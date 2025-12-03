#include"FluidSolver.h"
#include<iostream>
#include <limits>
#include <algorithm>
#include <cmath>

#define PI 3.14159265358979323846f

FluidSolver::FluidSolver() : velX(RESY, RESX + 1), velY(RESY + 1, RESX), pressure(RESY, RESX), divergence(RESY, RESX), smoke(RESY, RESX),
temperature(RESY, RESX), solid_map(RESY + 2, RESX + 2), air_map(RESY + 2, RESX + 2), vel_center(RESY, RESX), velX_temp(RESY, RESX + 1), velY_temp(RESY+1, RESX) {
	this->resX = RESX;
	this->resY = RESY;

    //allocate memory for velocity bytes
    velX_bytes.reserve(RESX * RESY * 3);
    velY_bytes.reserve(RESX * RESY * 3);
    vel_bytes.reserve(RESX * RESY * 3);

    //initialize_sinusoidal_vel_field(2, 1);

    initialize_smoke_field();
    initialize_temperature_field();
    initialize_environment();

	construct_vel_center();

    set_velocity_bytes();
}

void FluidSolver::solve_smoke() {
    update_scene(VisualizeField::Temperature);
    add_smoke_inflow();
    add_temperature_inflow();
    determine_time_step();
    advect_quantity(smoke);
    advect_quantity(temperature);
    advect_velocity();
    smoke_add_external_force();
    project();
    
    construct_vel_center();
    set_velocity_bytes();
}

void FluidSolver::update_scene(VisualizeField field_type=VisualizeField::Smoke) {
    /*
    fill the scene bytes array with the values to show
    */
    scene_bytes = {};

    for (int i = resY - 1; i >= 0; i--) {
        for (int j = 0; j < RESX; j++) {
            //set obstacles
            if (solid_map(i+1,j+1)) {
                unsigned char cr = static_cast<unsigned char>(0.0f);
                unsigned char cg = static_cast<unsigned char>(0.0f);
                unsigned char cb = static_cast<unsigned char>(255.0f);

                scene_bytes.push_back(cr);
                scene_bytes.push_back(cg);
                scene_bytes.push_back(cb);
            }
            else
            {
               //set the value to show
                float value = 0.0;
                unsigned char cr; unsigned char cb; unsigned char cg;
                float r; float g; float b;
                switch (field_type)
                {
                case VisualizeField::Smoke:
                    value = smoke(i,j) / 1.0; // maxAbsX;
                    value = std::min(1.0f, value);
                    cr = static_cast<unsigned char>(value * 255.0f);
                    cg = static_cast<unsigned char>(value * 255.0f);
                    cb = static_cast<unsigned char>(value * 255.0f);
                    break;
                case VisualizeField::Temperature:
                    value = temperature(i,j) / T_incoming; // maxAbsX;
                    value = std::min(T_incoming, value);
                    cr = static_cast<unsigned char>(value * 255.0f);
                    cg = static_cast<unsigned char>(value * 255.0f);
                    cb = static_cast<unsigned char>(value * 255.0f);
                    break;
                case VisualizeField::Pressure:
                    blue_red_color_map(pressure(i,j), 20.0, r, g, b);
                    cr = static_cast<unsigned char>(r * 255.0f);
                    cg = static_cast<unsigned char>(g * 255.0f);
                    cb = static_cast<unsigned char>(b * 255.0f);
                    break;
                case VisualizeField::Divergence:
                    blue_red_color_map(divergence(i,j), 1.0, r, g, b);
                    cr = static_cast<unsigned char>(r * 255.0f);
                    cg = static_cast<unsigned char>(g * 255.0f);
                    cb = static_cast<unsigned char>(b * 255.0f);
                    break;
                default:
                    break;
                }

                scene_bytes.push_back(cr);
                scene_bytes.push_back(cg);
                scene_bytes.push_back(cb);
            }
        }
    }
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
            if (vel_center(i,j).length() > umax)
            {
                umax = vel_center(i,j).length();
            }
        }
    }

    umax = umax + std::sqrt(5 * dx * gravity);
    dt =dt_coeff *  0.5 * 5 * dx / umax;
}

void FluidSolver::advect_velocity() {
    /*
    Update the velocity field (staggered grid) via semi lagrangian advection at timestep dt
    */

    //update velX
    //Field2D<float> velX_temp(resY, resX + 1);

    for (int i = 0; i < resY; i ++) {
        for (int j = 0; j < resX+1; j ++) {
            float vy;
            if (j == 0)
                vy = (velY(i,j) + velY(i + 1, j)) / 2.0;
            else if (j == resX)
                vy = (velY(i, j - 1) + velY(i + 1, j - 1)) / 2.0;
            else 
                vy = (velY(i, j - 1) + velY(i + 1, j - 1) + velY(i, j) + velY(i + 1, j)) / 4.0;
            //get the velocity vector at U_i+1/2, j
            float vx = velX(i, j);
            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

            velX_temp(i,j) = sample_velX(prev_pos);
        }
    }

    std::swap(velX, velX_temp);

    //update velY

    for (int i = 0; i < resY+1; i++) {
        for (int j = 0; j < resX; j++) {
            //interpolalate vel y at i+1/2,j
            float vx;
            if (i == 0)
                vx = (velX(i, j) + velX(i, j + 1)) / 2.0;
            else if (i == resY)
                vx = (velX(i - 1, j) + velX(i - 1, j + 1)) / 2.0;
            else
                vx = (velX(i-1, j) + velX(i-1, j+1) + velX(i, j) + velX(i, j+1)) / 4.0;
            //get the velocity vector at U_i+1/2, j
            float vy = velY(i, j);
            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

            velY_temp(i, j) = sample_velY(prev_pos);
        }
    }
    std::swap(velY, velY_temp);
}

void FluidSolver::advect_quantity(Field2D<float> &field) {
    for (int i = 0; i < RESY; i ++) {
        for (int j = 0; j < RESX; j++) {
            float vx = (velX(i, j) + velX(i, j + 1)) / 2;
            float vy = (velY(i, j) + velY(i, j + 1)) / 2;

            Vec2 dir(vx, -vy); //velocity in index coordinates

            //Semi lagrangian advection
            Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;
            field(i,j) = sample_quantity(field, prev_pos);
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
            float bouyouncy_force = -density_alpha * sample_quantity(smoke, Vec2(j, i - 0.5)) + bouyancy * (sample_quantity(temperature, Vec2(j, i-0.5)) - T_amb);
            float g_force = -gravity;
            velY(i, j) += dt * (bouyouncy_force+gravity);
        }
    }
}

void FluidSolver::project() {
    gauss_seidel_pressure_solve();
    update_incompressible_velocity();
    compute_divergence();
    compute_divergence_field();
}

void FluidSolver::compute_divergence() {
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            divergence(i,j) = ((velX(i, j + 1) - velX(i, j)) + (velY(i, j) - velY(i+1, j)))/dx;
        }
    }
}

void FluidSolver::compute_divergence_field() {
    divergence_field = 0.0;
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX; j++) {
            divergence_field += std::abs(divergence(i,j));
        }
    }
    //std::cout << "divergence field " << divergence_field << std::endl;
}

void FluidSolver::gauss_seidel_pressure_solve() {
    /*
    solves pressure with gauss seidel iteration
    */

    for (int n = 0; n < gauss_seidel_iterations; n++) {
        for (int i = 0; i < RESY; i++) {
            for (int j = 0; j < RESX; j++) {
                //continue if a solid cell
                if (solid_map(i + 1, j + 1)) continue;

                //solve pressure
                int st = solid_map(i, j+1); int sb = solid_map(i + 2, j+1); int sl = solid_map(i + 1,j); int sr = solid_map(i + 1, j + 2);
                int at = air_map(i, j + 1); int ab = air_map(i + 2,j + 1); int al = air_map(i + 1,j); int ar = air_map(i + 1,j + 2);
                int sum_occ = st + sr + sb + sl;
                int free_neigh = 4 - sum_occ;
                //the pressure is 0 if solid or air cell
                float pt = st||at ? 0.0 : pressure(i - 1,j);
                float pr = sr||ar ? 0.0 : pressure(i,j + 1);
                float pb = sb||ab ? 0.0 : pressure(i + 1,j);
                float pl = sl||al ? 0.0 : pressure(i,j - 1);

                float pressure_part = (pr + pl + pt + pb);
                float div_part = -density * dx * (((!sr)*velX(i, j + 1) - (!sl)*velX(i, j)) + ((!st)*velY(i, j) - (!sb)*velY(i + 1, j))) / dt;
                //overrelaxation
                float pressure_new = (pressure_part + div_part) / free_neigh;
                float pressure_old = pressure(i,j);
                pressure(i,j) = pressure_old + (pressure_new - pressure_old) * gauss_seidel_over_relaxation;
            }
        }
    }
}

void FluidSolver::update_incompressible_velocity() {
    //update velX
    for (int i = 0; i < RESY; i++) {
        for (int j = 0; j < RESX+1; j ++) {
            //continue if a solid cell
            if (solid_map(i + 1,j + 1)) continue;

            //get solid and air cells
            int sl = solid_map(i + 1,j); int sr = solid_map(i + 1,j + 1);
            int al = air_map(i + 1,j); int ar = air_map(i + 1,j + 1);

            float pressure_gradx;
            //solid boundary condition, find the pressure that makes the velocity 0
            if (sl) {
                float pl = pressure(i,j) + (density * dx / dt) * (0.0-velX(i, j));
                pressure_gradx = (pressure(i,j) - pl) / dx;
            }
            else if (sr) {
                float pr = pressure(i,j-1) + (density * dx / dt) * (velX(i, j) - 0.0);
                pressure_gradx = (pr-pressure(i,j - 1))/dx;
            }
            //air boundary condition, the pressure is 0 set the j=-1 or j=RESX to 0 if it is not solid to avoid error
            else if(al || j == 0)
            {
                pressure_gradx = (pressure(i,j) - 0.0) / dx;
            }
            else if (ar || j == RESX)
            {
                pressure_gradx = (0.0 - pressure(i,j - 1)) / dx;
            }
            else
            {
                pressure_gradx = (pressure(i,j) - pressure(i,j - 1)) / dx;
            }
            velX(i, j) = velX(i, j) - dt / density * pressure_gradx;
        }
    }

    //update velY
    for (int i = 0; i < RESY+1; i++) {
        for (int j = 0; j < RESX; j++) {
            //get solid and air cells
            int st = solid_map(i,j + 1); int sb = solid_map(i + 1,j + 1);
            int at = air_map(i,j + 1); int ab = air_map(i + 1,j + 1);

            //solid boundary condition, find the pressure that makes the velocity 0
            float pressure_grady;
            if (st) {
                float pt = pressure(i,j) + (density * dx / dt) * (velY(i, j) - 0.0);
                pressure_grady = (pt - pressure(i,j)) / dx;
            }
            else if (sb) {
                float pb = pressure(i - 1,j) + (density * dx / dt) * (0.0 - velY(i, j));
                pressure_grady = (pressure(i-1,j)-pb) / dx;
            }
            //air boundary condition, the pressure is 0, set the i=-1 or i=RESY to 0 if it is not solid to avoid error
            else if (at || i == 0)
            {
                pressure_grady = (0.0 - pressure(i,j)) / dx;
            }
            else if (ab || i == RESY)
            {
                pressure_grady = (pressure(i - 1,j) - 0.0) / dx;
            }
            else
            {
                pressure_grady = (pressure(i-1,j) - pressure(i,j)) / dx;
            }
            velY(i, j) = velY(i, j) - dt / density * pressure_grady;
        }
    }
}

void FluidSolver::construct_vel_center() {
	for (int i = 0; i < resY; i++) {
		for (int j = 0; j < resX; j ++) {
			float velx = (velX(i, j) + velX(i, j + 1)) / 2;
			float vely = (velY(i, j) + velY(i + 1, j)) / 2;

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
            velX(j, i) = amplitude* std::sin(twoPiF * x);
        }
    }

    for (int j = 0; j <= resY; ++j)
    {
        float y = j * dy;
        for (int i = 0; i < resX; ++i)
        {
            float x = (i + 0.5f) * dx;
            velY(j, i) = amplitude* std::sin(twoPiF * y);
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
                smoke(i,j) = 1.0;
            }
            else
            {
                smoke(i,j) = 0.0;
            }
        }
    }
}

void FluidSolver::initialize_environment() {
    /*
    Sets the air and solid cells
    */
    //solid cells
    for (int i = 0; i < RESX + 2; i++) {
        for (int j = 0; j < RESY+2;j++) {
            //make the border solid
            if (i == 0 || i == RESY + 1 || j == 0 || j == RESX + 1) {
                solid_map(i,j) = true;
            }

            //draw sphere obstical in the center
            /*if (std::pow((i - RESY / 2), 2) + std::pow((j - RESX / 2), 2) < 50) {
                solid_map[i][j] = true;
            }*/
        }
    }
    //air cells
    for (int i = 0; i < RESX + 2; i++) {
        for (int j = 0; j < RESY + 2; j++) {
            //make the border air
            
            if (i == 0 || i == RESY + 1 || j == 0 || j == RESX + 1) {
                air_map(i,j) = true;
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
                smoke(i,j) = 1.0;
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
                temperature(i,j) = T_incoming;
            }
            else
            {
                temperature(i,j) = T_amb;
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
                temperature(i,j) = 70.0;
            }
        }
    }
}

void FluidSolver::set_velocity_bytes() {
    /*
    Fills the velocity bytes from the velocity arrays 
    */
    velX_bytes.clear();
    velY_bytes.clear();
    vel_bytes.clear();

    //Find the maximum velocity x and y
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < resY; i ++) {
        for (int j = 0; j < resX; j ++) {
            if (vel_center(i,j).x > max_x) {
                max_x = vel_center[i][j].x;
            }
            if (vel_center(i,j).y > max_y) {
                max_y = vel_center[i][j].y;
            }
            if (vel_center(i,j).x < min_x) {
                min_x = vel_center[i][j].x;
            }
            if (vel_center(i,j).y < min_y) {
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
            float vx = vel_center(i,j).x;
            float vy = vel_center(i,j).y;

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

float FluidSolver::sample_quantity(Field2D<float> &field, Vec2 pos) {
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

    float q00 = field(i0,j0);
    float q10 = field(i0,j1);
    float q01 = field(i1,j0);
    float q11 = field(i1,j1);

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

    float v00 = velX(i0, j0);
    float v10 = velX(i0, j1);
    float v01 = velX(i1, j0);
    float v11 = velX(i1, j1);
    
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

    float v00 = velY(i0,j0);
    float v10 = velY(i0,j1);
    float v01 = velY(i1,j0);
    float v11 = velY(i1,j1);

    // Bilinear interpolation
    float vx0 = v00 * (1.0f - tx) + v10 * tx;
    float vx1 = v01 * (1.0f - tx) + v11 * tx;

    float interpolation = vx0 * (1.0f - ty) + vx1 * ty;

    return vx0 * (1.0f - ty) + vx1 * ty;
}

void FluidSolver::blue_red_color_map(float val, float normalization_factor, float& r, float& g, float& b) {
    float t = val / normalization_factor;
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
}