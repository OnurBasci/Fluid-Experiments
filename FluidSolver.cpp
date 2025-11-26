#include"FluidSolver.h"
#include<iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#define PI 3.14159265358979323846f

FluidSolver::FluidSolver() {
	this->resX = RESX;
	this->resY = RESY;

    //initialize_sinusoidal_vel_field(2, 1);

    initialize_smoke_field();
    initialize_temperature_field();

	construct_vel_center();

    set_velocity_bytes();
}

void FluidSolver::solve_smoke() {
    add_smoke_inflow();
    add_temperature_inflow();
    //determine_time_step();
    advect_quantity(smoke);
    advect_quantity(temperature);
    advect_velocity();
    smoke_add_external_force();
    project();
    
    construct_vel_center();
    set_velocity_bytes();
}

void FluidSolver::solve() {
    advect_velocity();
    project();

    construct_vel_center();
    set_velocity_bytes();
}

void FluidSolver::determine_time_step() {
    /*
    choose a time step smaller that 5dx/umax
    */
    float umax = std::numeric_limits<float>::min();
    for (int i = 0; i < RESY; i ++) {
        for (int j = 0; j < RESX; j++) {
            if (vel_center[i][j].length() > umax)
            {
                umax = vel_center[i][j].length();
            }
        }
    }

    umax = umax + std::sqrt(5 * dx * gravity);
    dt = 0.5 * 5 * dx / umax;
    std::cout << "dt " << dt <<std::endl;
}

void FluidSolver::advect_velocity() {
    /*
    Update the velocity field (staggered grid) via semi lagrangian advection at timestep dt
    */

    //update velX
    float velX_temp[RESY][RESX + 1] = {};

    for (int i = 0; i < resY; i ++) {
        for (int j = 0; j < resX+1; j ++) {
            float vy;
            if (j == 0)
                vy = (velY[i][j] + velY[i + 1][j]) / 2.0;
            else if (j == resX)
                vy = (velY[i][j - 1] + velY[i + 1][j - 1]) / 2.0;
            else 
                vy = (velY[i][j - 1] + velY[i + 1][j - 1] + velY[i][j] + velY[i + 1][j]) / 4.0;
            //get the velocity vector at U_i+1/2, j
            float vx = velX[i][j];
            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

            velX_temp[i][j] = sample_velX(prev_pos);
        }
    }

    std::memcpy(velX, velX_temp, sizeof(velX_temp));

    //update velY
    float velY_temp[RESY+1][RESX] = {};

    for (int i = 0; i < resY+1; i++) {
        for (int j = 0; j < resX; j++) {
            //interpolalate vel y at i+1/2,j
            float vx;
            if (i == 0)
                vx = (velX[i][j] + velX[i][j + 1]) / 2.0;
            else if (i == resY)
                vx = (velX[i - 1][j] + velX[i - 1][j + 1]) / 2.0;
            else
                vx = (velX[i-1][j] + velX[i-1][j+1] + velX[i][j] + velX[i][j+1]) / 4.0;
            //get the velocity vector at U_i+1/2, j
            float vy = velY[i][j];
            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

            velY_temp[i][j] = sample_velY(prev_pos);
        }
    }

    std::memcpy(velY, velY_temp, sizeof(velX_temp));
}

void FluidSolver::advect_quantity(float arr[RESY][RESX]) {
    for (int i = 0; i < RESY; i ++) {
        for (int j = 0; j < RESX; j++) {
            float vx = (velX[i][j] + velX[i][j + 1]) / 2;
            float vy = (velY[i][j] + velY[i][j + 1]) / 2;

            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

            /*if (i == RESY - 5 && j == RESX / 2 - 10) {
                std::cout << "i, j" << Vec2(i, j) << " prev pos " << prev_pos << " val " << arr[i][j] << " vel " << Vec2(vx, vy) << std::endl;
            }*/
            arr[i][j] = sample_quantity(arr, prev_pos);
        }
    }
}

void FluidSolver::smoke_add_external_force() {
    /*
    We approximate the smokes movment due to the temperature change via approximative bouyancy force
    */
    for (int i = 0; i < RESY+1; i++) {
        for (int j = 0; j < RESX; j++) {
            //add gravity and bouyouncy
            float bouyouncy_force = -density * sample_quantity(smoke, Vec2(j, i - 0.5)) + bouyancy * (sample_quantity(temperature, Vec2(j, i-0.5)) - T_amb);
            float g_force = -gravity;
            velY[i][j] += dt * (bouyouncy_force);
        }
    }
}

void FluidSolver::project() {
    compute_divergence();
    gauss_seidel_pressure_solve();
    update_incompressible_velocity();
    compute_divergence_field();
}

void FluidSolver::compute_divergence() {
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            divergence[i][j] = ((velX[i][j + 1] - velX[i][j]) + (velY[i][j] - velY[i+1][j]))/dx;
        }
    }
}

void FluidSolver::compute_divergence_field() {
    divergence_field = 0.0;
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            divergence_field += std::abs(divergence[i][j]);
        }
    }
    std::cout << "divergence field " << divergence_field << std::endl;
}

void FluidSolver::gauss_seidel_pressure_solve() {
    /*
    solves pressure with gauss seidel iteration
    */

    for (int n = 0; n < gauss_seidel_iterations; n++) {
        for (int i = 0; i < RESY; i++) {
            for (int j = 0; j < RESX; j++) {
                int st = (i == 0); int sr = (j == RESX - 1); int sb = (i == RESY - 1); int sl = (j == 0);
                int sum_occ = st + sr + sb + sl;
                int free_neigh = 4 - sum_occ;
                //set occupancy coefficient, if outside of the grid, we apply drichlet condition, p = 0
                float pt = st ? 0.0 : pressure[i - 1][j];
                float pr = sr ? 0.0 : pressure[i][j + 1];
                float pb = sb ? 0.0 : pressure[i + 1][j];
                float pl = sl ? 0.0 : pressure[i][j - 1];

                float pressure_part = ((!sr)*pr + (!sl)*pl + (!st)*pt + (!sb)*pb);
                float div_part = -density * dx * (((!sr)*velX[i][j + 1] - (!sl)*velX[i][j]) + ((!st)*velY[i][j] - (!sb)*velY[i + 1][j])) / dt;
                //overrelaxation
                float pressure_new = (pressure_part + div_part) / free_neigh;
                float pressure_old = pressure[i][j];
                pressure[i][j] = pressure_old + (pressure_new - pressure_old) * gauss_seidel_over_relaxation;
            }
        }
    }
}

void FluidSolver::update_incompressible_velocity() {
    //update velX
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX+1; j ++) {
            float pressure_gradx;
            //Neumann condition for outside of domain, give the pressure value that make the velocity 0
            if (j == 0) {
                float pl = pressure[i][j] + (density * dx / dt) * (0.0-velX[i][j]);
                pressure_gradx = (pressure[i][j] - pl) / dx;
            }
            else if (j == RESX) {
                float pr = pressure[i][j-1] + (density * dx / dt) * (velX[i][j] - 0.0);
                pressure_gradx = (pr-pressure[i][j - 1])/dx;
            }
            else
            {
                pressure_gradx = (pressure[i][j] - pressure[i][j - 1]) / dx;
            }
            velX[i][j] = velX[i][j] - dt / density * pressure_gradx;
        }
    }

    //update velY
    for (int i = 0; i < RESY+1; i++) {
        for (int j = 0; j < RESX; j++) {
            float pressure_grady;
            //Neumann condition for outside of domain, give the pressure value that make the velocity 0
            if (i == 0) {
                float pt = pressure[i][j] + (density * dx / dt) * (velY[i][j] - 0.0);
                pressure_grady = (pt - pressure[i][j]) / dx;
            }
            else if (i == RESY) {
                float pb = pressure[i - 1][j] + (density * dx / dt) * (0.0 - velY[i][j]);
                pressure_grady = (pressure[i-1][j]-pb) / dx;
            }
            else
            {
                pressure_grady = (pressure[i-1][j] - pressure[i][j]) / dx;
            }
            velY[i][j] = velY[i][j] - dt / density * pressure_grady;
        }
    }
}

void FluidSolver::construct_vel_center() {
	for (int i = 0; i < resY; i++) {
		for (int j = 0; j < resX; j ++) {
			float velx = (velX[i][j] + velX[i][j + 1]) / 2;
			float vely = (velY[i][j] + velY[i + 1][j]) / 2;

			vel_center[i][j] = Vec2(velx, vely);
		}
	}
}


void FluidSolver::initialize_sinusoidal_vel_field(float frequency, float amplitude)
{
    // Grid spacing assuming domain [0, 1] x [0, 1]
    const float dx = 1.0f / static_cast<float>(resX);
    const float dy = 1.0f / static_cast<float>(resY);

    const float twoPiF = 2.0f * PI * frequency;

    for (int j = 0; j < resY; ++j)
    {
        float y = (j + 0.5f) * dy;
        for (int i = 0; i <= resX; ++i)
        {
            float x = i * dx;
            velX[j][i] = amplitude* std::sin(twoPiF * x);
        }
    }

    for (int j = 0; j <= resY; ++j)
    {
        float y = j * dy;
        for (int i = 0; i < resX; ++i)
        {
            float x = (i + 0.5f) * dx;
            velY[j][i] = amplitude* std::sin(twoPiF * y);
        }
    }
}

void FluidSolver::initialize_smoke_field() {
    /*
    set an initial smoke density field
    */

    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            if (std::pow((i - RESY/2 - 30),2) + std::pow((j - RESX/2), 2) < 100) {
                smoke[i][j] = 1.0;
            }
            else
            {
                smoke[i][j] = 0.0;
            }
        }
    }
}

void FluidSolver::add_smoke_inflow() {
    /*
    adds constant flow to the scene
    */

    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            if (std::pow((i - RESY / 2 - 30), 2) + std::pow((j - RESX / 2), 2) < 100) {
                smoke[i][j] = 1.0;
            }
        }
    }
}

void FluidSolver::initialize_temperature_field() {
    /*
    set an initial temperature density field
    */

    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            if (i > RESY - 8 && i < RESY - 1 && j > RESX / 2 - RESX / 6 && j < RESX / 2 + RESX / 6) {
                temperature[i][j] = 70.0;
            }
            else
            {
                temperature[i][j] = T_amb;
            }
        }
    }
}

void FluidSolver::add_temperature_inflow() {
    /*
    set an initial temperature density field
    */

    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            if (i > RESY - 8 && i < RESY - 1 && j > RESX / 2 - RESX / 6 && j < RESX / 2 + RESX / 6) {
                temperature[i][j] = 70.0;
            }
        }
    }
}

void FluidSolver::set_velocity_bytes() {
    /*
    Fills the velocity bytes from the velocity arrays 
    */
    //initialize the byte vector
    velX_bytes = {};
    velY_bytes = {};
    vel_bytes = {};

    //Find the maximum velocity x and y
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < resY; i ++) {
        for (int j = 0; j < RESX; j ++) {
            if (vel_center[i][j].x > max_x) {
                max_x = vel_center[i][j].x;
            }
            if (vel_center[i][j].y > max_y) {
                max_y = vel_center[i][j].y;
            }
            if (vel_center[i][j].x < min_x) {
                min_x = vel_center[i][j].x;
            }
            if (vel_center[i][j].y < min_y) {
                min_y = vel_center[i][j].y;
            }
        }
    }

    // For color mapping of velX, use symmetric range around 0
    float maxAbsX = std::max(std::fabs(min_x), std::fabs(max_x));
    if (maxAbsX == 0.0f) {
        maxAbsX = 1.0f; // avoid division by zero
    }

    //set the velocity values into the bytes
    for (int i = resY-1; i >= 0; i--) {
        for (int j = 0; j < resX; j++) {
            float vx = vel_center[i][j].x;
            float vy = vel_center[i][j].y;

            // Normalize to 0–1
            float nx = (vx - min_x) / (max_x - min_x);
            float ny = (vy - min_y) / (max_y - min_y);

            // Convert to byte 0–255
            unsigned char bx = static_cast<unsigned char>(nx * 255.0f);
            unsigned char by = static_cast<unsigned char>(ny * 255.0f);


            // --- 2) Color mapping for velX into vel_bytes (RGB) ---
            // t in [-1, 1]: negative -> blue, positive -> red
            float t = vx / 1;
            t = std::max(-1.0f, std::min(1.0f, t));

            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (t > 0.0f) {
                // positive: black -> red
                r = t;          // 0..1
                g = 0.0f;
                b = 0.0f;
            }
            else if (t < 0.0f) {
                // negative: black -> blue
                r = 0.0f;
                g = 0.0f;
                b = -t;         // t in [-1,0] -> 1..0
            }
            else {
                // exactly zero: black
                r = g = b = 0.0f;
            }

            unsigned char cr = static_cast<unsigned char>(r * 255.0f);
            unsigned char cg = static_cast<unsigned char>(g * 255.0f);
            unsigned char cb = static_cast<unsigned char>(b * 255.0f);

            velX_bytes.push_back(cr);
            velX_bytes.push_back(cg);
            velX_bytes.push_back(cb);

            // --- 2) Color mapping for velY into vel_bytes (RGB) ---
            // t in [-1, 1]: negative -> blue, positive -> red
            t = vy / 1;
            t = std::max(-1.0f, std::min(1.0f, t));

            if (t > 0.0f) {
                // positive: black -> red
                r = t;          // 0..1
                g = 0.0f;
                b = 0.0f;
            }
            else if (t < 0.0f) {
                // negative: black -> blue
                r = 0.0f;
                g = 0.0f;
                b = -t;         // t in [-1,0] -> 1..0
            }
            else {
                // exactly zero: black
                r = g = b = 0.0f;
            }

            cr = static_cast<unsigned char>(r * 255.0f);
            cg = static_cast<unsigned char>(g * 255.0f);
            cb = static_cast<unsigned char>(b * 255.0f);

            velY_bytes.push_back(cr);
            velY_bytes.push_back(cg);
            velY_bytes.push_back(cb);

            //set the velocity bytes height*width*3 (x,y, 0)
            vel_bytes.push_back(bx); //R
            vel_bytes.push_back(by); //G
            vel_bytes.push_back(0);  //B
        }
    }
}

std::vector<unsigned char> FluidSolver::array_to_bytes(float arr[RESY][RESX], float normalize_factor=1.0) {
    /*
    Transforms an array into bytes with a color mapping for visualization
    */
    std::vector<unsigned char> bytes;

    //Find the maximum velocity x and y
    float max = -std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < resY; i++) {
        for (int j = 0; j < RESX; j++) {
            if (arr[i][j] > max) {
                max = arr[i][j];
            }
            if (arr[i][j] < min) {
                min = arr[i][j];
            }
        }
    }

    // For color mapping of velX, use symmetric range around 0
    float maxAbsX = std::max(std::fabs(min), std::fabs(max));
    if (maxAbsX == 0.0f) {
        maxAbsX = 1.0f; // avoid division by zero
    }

    //set the velocity values into the bytes
    for (int i = RESY-1; i >= 0; i--) {
        for (int j = 0; j < resX; j++) {

            //Color mapping for velX into vel_bytes (RGB) ---
            // t in [-1, 1]: negative -> blue, positive -> red
            float t = arr[i][j] / normalize_factor; // maxAbsX;
            t = std::max(-1.0f, std::min(1.0f, t));

            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (t > 0.0f) {
                // positive: black -> red
                r = t;          // 0..1
                g = 0.0f;
                b = 0.0f;
            }
            else if (t < 0.0f) {
                // negative: black -> blue
                r = 0.0f;
                g = 0.0f;
                b = -t;         // t in [-1,0] -> 1..0
            }
            else {
                // exactly zero: black
                r = g = b = 0.0f;
            }

            unsigned char cr = static_cast<unsigned char>(r * 255.0f);
            unsigned char cg = static_cast<unsigned char>(g * 255.0f);
            unsigned char cb = static_cast<unsigned char>(b * 255.0f);

            bytes.push_back(cr);
            bytes.push_back(cg);
            bytes.push_back(cb);
        }
    }

    return bytes;
}

float FluidSolver::sample_quantity(float arr[RESY][RESX], Vec2 pos) {
    // Clamp position to valid index range
    float x = std::clamp(pos.x, 0.0f, static_cast<float>(RESX - 1));      // velX has RESX+1 in x
    float y = std::clamp(pos.y, 0.0f, static_cast<float>(RESY - 1));  // 0 .. RESY-1 in y

    // Integer cell indices
    int j0 = static_cast<int>(std::floor(x));
    int i0 = static_cast<int>(std::floor(y));

    int j1 = std::min(j0 + 1, RESX - 1);        // x index: 0 .. RESX
    int i1 = std::min(i0 + 1, RESY - 1);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    float q00 = arr[i0][j0];
    float q10 = arr[i0][j1];
    float q01 = arr[i1][j0];
    float q11 = arr[i1][j1];

    // Bilinear interpolation
    float qx0 = q00 * (1.0f - tx) + q10 * tx;
    float qx1 = q01 * (1.0f - tx) + q11 * tx;

    float interpolation = qx0 * (1.0f - ty) + qx1 * ty;

    return qx0 * (1.0f - ty) + qx1 * ty;
}

float FluidSolver::sample_velX(Vec2 pos)
{
    // Clamp position to valid index range
    float x = std::clamp(pos.x, 0.0f, static_cast<float>(RESX));      // velX has RESX+1 in x
    float y = std::clamp(pos.y, 0.0f, static_cast<float>(RESY-1));  // 0 .. RESY-1 in y

    // Integer cell indices
    int j0 = static_cast<int>(std::floor(x));
    int i0 = static_cast<int>(std::floor(y));

    int j1 = std::min(j0 + 1, RESX);        // x index: 0 .. RESX
    int i1 = std::min(i0 + 1, RESY - 1);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    float v00 = velX[i0][j0];
    float v10 = velX[i0][j1];
    float v01 = velX[i1][j0];
    float v11 = velX[i1][j1];
    
    // Bilinear interpolation
    float vx0 = v00 * (1.0f - tx) + v10 * tx;
    float vx1 = v01 * (1.0f - tx) + v11 * tx;

    float interpolation = vx0 * (1.0f - ty) + vx1 * ty;

    return vx0 * (1.0f - ty) + vx1 * ty;
}

float FluidSolver::sample_velY(Vec2 pos)
{
    // Clamp position to valid index range
    float x = std::clamp(pos.x, 0.0f, static_cast<float>(RESX-1));      // velX has RESX+1 in x
    float y = std::clamp(pos.y, 0.0f, static_cast<float>(RESY));  // 0 .. RESY-1 in y

    // Integer cell indices
    int j0 = static_cast<int>(std::floor(x));
    int i0 = static_cast<int>(std::floor(y));

    int j1 = std::min(j0 + 1, RESX-1);        // x index: 0 .. RESX
    int i1 = std::min(i0 + 1, RESY);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    float v00 = velY[i0][j0];
    float v10 = velY[i0][j1];
    float v01 = velY[i1][j0];
    float v11 = velY[i1][j1];

    // Bilinear interpolation
    float vx0 = v00 * (1.0f - tx) + v10 * tx;
    float vx1 = v01 * (1.0f - tx) + v11 * tx;

    float interpolation = vx0 * (1.0f - ty) + vx1 * ty;

    return vx0 * (1.0f - ty) + vx1 * ty;
}