#include "FluidSolverGPU.cuh"
//#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
//#include <thrust/execution_policy.h>

CUDA_D
float sample_scalar_field(float* field, Vec2 pos, int resX, int resY);
Vec2 sample_vec2_field(Vec2* field, Vec2 pos, int resX, int resY);
Vec3 sample_vec3_field(Vec3* field, Vec2 pos, int resX, int resY);

//KERNELS

__global__
void construct_vel_center_kernel(Vec2* vel_center, float* velX, float* velY, int resX, int resY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    float vx = velX[i*(resX+1) + j] + velX[i*(resX+1) + j + 1];
    float vy = velY[i * resX + j] + velY[(i + 1) * resX + j];

    vel_center[i * resX + j] = Vec2(vx, vy);
}

__global__
void compute_divergence_kernel(float* divergence, float* velX, float* velY, int resX, int resY, float dx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int resX_1 = resX + 1;

    int i = idx / resX;
    int j = idx % resX;

    divergence[i*resX+j] = ((velX[i*resX_1+j + 1] - velX[i*resX_1+j]) + (velY[i*resX+j] - velY[(i + 1)*resX+j])) / dx;
}

__global__
void compute_vorticity_kernel(float* vorticity, Vec2* vel, int resX, int resY, float dx) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    // boundary: set to 0 (or copy later)
    if (i == 0 || i == resY - 1 || j == 0 || j == resX - 1) {
        vorticity[idx] = 0.0f;
        return;
    }

    // omega = dv/dx - du/dy
    float dv_dx = (vel[i * resX + (j + 1)].y - vel[i * resX + (j - 1)].y)/(2*dx);
    float du_dy = (vel[(i - 1) * resX + j].x - vel[(i + 1) * resX + j].x)/(2*dx);

    vorticity[idx] = dv_dx - du_dy;
}

__global__
void add_block_inflow_bottom_kernel(float* field, int resX, int resY, float val) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    if (i > resY - 8 && i < resY - 1 && j > resX / 2 - resX / 6 && j < resX / 2 + resX / 6) {
        field[i * resX + j] = val;
    }
}

__global__
void add_block_inflow_from_map(float* field, unsigned char* inflow_map, Vec3* color, Vec3* color_inflow, int resX, int resY, float val) {
    /*
    range [0,1] if 1 all the left covered
    */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    /*if (i == 0 && j == 20) {
        printf("on gpu %d\n", inflow_map[i*resX+j]);
    }*/

    if (inflow_map[i * resX + j]) {
        field[i * resX + j] = val;
        color[i * resX + j] = color_inflow[i*resX+j];
    }
}

__global__
void advect_velocityX_kernel(float* velX, float* velX_temp, float* velY, int resX, int resY, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int resX_1 = resX + 1;
    int num_cells = resX_1 * resY;
    if (idx >= num_cells) return;

    int i = idx / resX_1;
    int j = idx % resX_1;

    float vx = velX[i * resX_1 + j];
    float vy;

    if (j == 0)
        vy = (velY[i*resX+j] + velY[(i + 1)*resX+j]) / 2.0;
    else if (j == resX)
        vy = (velY[i*resX + (j - 1)] + velY[(i + 1)*resX + j - 1]) / 2.0;
    else
        vy = (velY[i*resX + j - 1] + velY[(i + 1)*resX + j - 1] + velY[i*resX+j] + velY[(i + 1)*resX+j]) / 4.0;

    Vec2 dir(vx*resX, -vy*resY); //velocity in index coordinates

    //Semi lagrangian advection
    Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

    velX_temp[i * resX_1 + j] = sample_scalar_field(velX, prev_pos, resX_1, resY);
}

__global__
void advect_velocityY_kernel(float* velY, float* velY_temp, float* velX, int resX, int resY, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * (resY+1);
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    int resX_1 = resX + 1;

    float vy = velY[i*resX+j];
    float vx;

    if (i == 0)
        vx = (velX[i*(resX_1)+j] + velX[i*(resX_1)+j + 1])/ 2.0;
    else if (i == resY)
        vx = (velX[(i - 1)*resX_1+j] + velX[(i - 1)*resX_1+j + 1])/ 2.0;
    else
        vx = (velX[(i - 1)*resX_1+j] + velX[(i - 1)*resX_1+j+1] + velX[i*resX_1+j] + velX[i*resX_1+j+1]) / 4.0;
    //get the velocity vector at U_i+1/2, j
    Vec2 dir(vx*resX, -vy*resY); //velocity in index coordinates

    //Semi lagrangian advection
    Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

    velY_temp[i * resX + j] = sample_scalar_field(velY, prev_pos, resX, resY+1);
}

__global__
void diffuse_scalar_field_kernel(float* field, float* swap_field, unsigned char* solid_map, unsigned char* air_map, int resX, int resY, float dx, float dt, float diffuse_factor) {
    /*
    diffuse the field with laplacian operator
    */
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    if (solid_map[(i + 1) * (resX + 2) + (j + 1)]) return;

    //Dirichlet condition for solid and air pixels
    unsigned char st = solid_map[i * (resX + 2) + j + 1]; unsigned char sb = solid_map[(i + 2) * (resX + 2) + j + 1]; unsigned char sl = solid_map[(i + 1) * (resX + 2) + j]; unsigned char sr = solid_map[(i + 1) * (resX + 2) + j + 2];
    unsigned char at = air_map[i * (resX + 2) + j + 1]; unsigned char ab = air_map[(i + 2) * (resX + 2) + j + 1]; unsigned char al = air_map[(i + 1) * (resX + 2) + j]; unsigned char ar = air_map[(i + 1) * (resX + 2) + j + 2];

    float top = st || at ? 0.0 : field[(i - 1) * resX + j];
    float right = sr || ar ? 0.0 : field[i * resX + j + 1];
    float bottom = sb || ab ? 0.0 : field[(i + 1) * resX + j];
    float left = sl || al ? 0.0 : field[i * resX + j - 1];
    float center = field[i * resX + j];

    float laplacian = (top + right + bottom + left - 4 * center) / (dx * dx);
    float diffused = center + laplacian * dt*diffuse_factor;

    swap_field[i * resX + j] = diffused;
}

__global__
void advect_vec3_kernel(Vec3* field, Vec3* swap_field, Vec2* vel, int resX, int resY, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int num_cells = resX * resY;

    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    Vec2 dir(vel[i * resX + j].x * resX, -vel[i * resX + j].y * resY);
    //Semi lagrangian advection
    Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

    Vec3 sample_val = sample_vec3_field(field, prev_pos, resX, resY);
    swap_field[i * resX + j] = sample_val;
}

__global__
void advect_quantity_kernel(float* field, float* swap_field, Vec2* vel, int resX, int resY, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int num_cells = resX * resY;

    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    Vec2 dir(vel[i*resX+j].x *resX, -vel[i * resX + j].y * resY);
    //Semi lagrangian advection
    Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

    float sample_val = sample_scalar_field(field, prev_pos, resX, resY);
    swap_field[i * resX + j] = sample_val;
}

__global__
void add_external_force_smoke_kernel(float* velY, float* smoke, float* temperature, int resX, int resY, float gravity, float dt, float d_a, float b, float T_amb) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * (resY + 1);
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    //add bouyouncy force depending on the smoke density and temperature
    float bouyouncy_force = -d_a * sample_scalar_field(smoke, Vec2(j, i - 0.5), resX, resY) + b * (sample_scalar_field(temperature, Vec2(j, i - 0.5), resX, resY) - T_amb);
    float g_force = -gravity;

    velY[i * resX + j] += dt* (bouyouncy_force + g_force);;
}

__global__
void add_external_force_wind_tunel_kernel(float* velX, int resX, int resY, float wind_force) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = (resX+1) * resY;
    if (idx >= num_cells) return;

    int resX_1 = (resX + 1);

    int i = idx / resX_1;
    int j = idx % resX_1;

    int center = resY / 2;

    if (j == 8 && i >= 0 * center && i < resY) {
        velX[i * resX_1 + j] = wind_force;
    }
}

__global__
void add_vorticity_confinement_kernel(float* velX, float* velY, float* vort, int resX, int resY, float dx, float dt, float v_coef) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    // boundary add no force
    if (i == 0 || i >= resY - 1 || j == 0 || j >= resX - 1) return;

    //compute for velY
    Vec2 up_pos(j+0.5, i);
    float gx = abs(sample_scalar_field(vort, Vec2(up_pos.x+0.5, i), resX, resY)) - abs(sample_scalar_field(vort, Vec2(up_pos.x-0.5, i), resX, resY));
    Vec2 grad_up(gx/dx ,(abs(vort[(i-1) * resX + j]) - abs(vort[i * resX + j]))/dx);
    Vec2 g_norm = grad_up / (grad_up.length() + 1e-6);

    float w = sample_scalar_field(vort, up_pos, resX, resY);
    float f_y = -v_coef * dx * w * g_norm.x;
    velY[i * resX + j] += f_y * dt;

    //compute for velX
    Vec2 left_pos(j, i+0.5);
    float gy = abs(sample_scalar_field(vort, Vec2(j, left_pos.y-0.5), resX, resY)) - abs(sample_scalar_field(vort, Vec2(j, left_pos.y+0.5), resX, resY));
    Vec2 grad_left((abs(vort[i*resX + j]) - abs(vort[i*resX+j-1])) / dx, gy/dx);
    g_norm = grad_left / (grad_left.length() + 1e-6);

    w = sample_scalar_field(vort, left_pos, resX, resY);
    float f_x = v_coef * dx * w * g_norm.y;
    velX[i * (resX + 1) + j] += f_x * dt;
}

__global__
void add_external_force_kernel(float* velX, float* velY, Vec2* adder_external_vel, Vec2* setter_external_vel, int resX, int resY, float dx, float dt) {
    //sets velx and vely from an external vel field
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    //add velocity from additional external velocity kernel
    Vec2 sampled_up_adder = sample_vec2_field(adder_external_vel, Vec2(j, i - dx * 0.5), resX, resY);
    Vec2 sampled_left_adder = sample_vec2_field(adder_external_vel, Vec2(j - dx * 0.5, i), resX, resY);

    velX[i * (resX + 1) + j] += sampled_left_adder.x*dt;
    velY[i * resX + j] += sampled_up_adder.y*dt;

    //sample the values at (i-1/2dx, j) and (i, j-1/2dx)
    Vec2 sampled_up_setter = sample_vec2_field(setter_external_vel, Vec2(j, i-dx*0.5), resX, resY);
    Vec2 sampled_left_setter = sample_vec2_field(setter_external_vel, Vec2(j-dx*0.5, i), resX, resY);

    if (fabs(sampled_left_setter.x) > 0)
        velX[i * (resX + 1) + j] = sampled_left_setter.x;
    if (fabs(sampled_up_setter.y) > 0)
        velY[i * resX + j] = sampled_up_setter.y;
}

__global__
void jacobi_pressure_solve(float* pressure_new, float* pressure_old, float* velX, float* velY, unsigned char* solid_map, unsigned char * air_map, int resY, int resX, float density, float dx, float dt) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * (resY);
    if (idx >= num_cells) return;

    int resX_1 = resX + 1;

    int i = idx / resX;
    int j = idx % resX;

    if (solid_map[(i + 1) * (resX+2) + (j + 1)]) return;

    //mark air and solid cells
    unsigned char st = solid_map[i*(resX + 2) + j + 1]; unsigned char sb = solid_map[(i + 2) * (resX + 2) + j + 1]; unsigned char sl = solid_map[(i + 1) * (resX + 2) + j]; unsigned char sr = solid_map[(i + 1) * (resX + 2) + j + 2];
    unsigned char at = air_map[i * (resX + 2) + j + 1]; unsigned char ab = air_map[(i + 2) * (resX + 2) + j + 1]; unsigned char al = air_map[(i + 1) * (resX + 2) + j]; unsigned char ar = air_map[(i + 1) * (resX + 2) + j + 2];
    int sum_occ = st + sb + sl + sr;
    unsigned char free_neigh = 4 - sum_occ;

    //the pressure is 0 if solid or air cell
    float pt = st||at ? 0.0 : pressure_old[(i - 1)*resX+ j];
    float pr = sr||ar ? 0.0 : pressure_old[i*resX + j + 1];
    float pb = sb||ab ? 0.0 : pressure_old[(i + 1)*resX+ j];
    float pl = sl||al ? 0.0 : pressure_old[i*resX+ j - 1];

    float pressure_part = (pr + pl + pt + pb);
    float div_part = -density * dx * (((!sr)*velX[i*resX_1 + j+1] - (!sl)*velX[i*resX_1 + j]) + ((!st)*velY[i*resX + j] - (!sb)*velY[(i + 1)*resX+j])) / dt;
    //overrelaxation
    float p_new = (pressure_part + div_part) / free_neigh;
    pressure_new[i * resX + j] = p_new;
}

__global__
void make_velX_incompressible(float *velX, float* pressure, unsigned char* solid_map, unsigned char* air_map, int resX, int resY, float dx, float dt, float density) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = (resX+1) * resY;
    if (idx >= num_cells) return;

    int resX_1 = resX + 1;

    int i = idx / resX_1;
    int j = idx % resX_1;

    unsigned char sl = solid_map[(i+1)*(resX + 2)+j]; unsigned char sr = solid_map[(i + 1)*(resX + 2)+j+1];
    unsigned char al = air_map[(i+1)*(resX + 2)+j]; unsigned char ar = air_map[(i+1)*(resX + 2)+j+1];

    float pressure_gradx;
    //solid boundary condition, find the pressure that makes the velocity 0
    if (sl) {
        float pl = pressure[i*resX+j] + (density * dx / dt) * (0.0 - velX[i*resX_1+j]);
        pressure_gradx = (pressure[i*resX+j] - pl) / dx;
    }
    else if (sr) {
        float pr = pressure[i*resX+(j - 1)] + (density * dx / dt) * (velX[i*resX_1+j] - 0.0);
        pressure_gradx = (pr - pressure[i*resX+(j - 1)]) / dx;
    }
    //air boundary condition, the pressure is 0 set the j=-1 or j=RESX to 0 if it is not solid to avoid error
    else if (al)
    {
        pressure_gradx = (pressure[i*resX+j] - 0.0) / dx;
    }
    else if (ar)
    {
        pressure_gradx = (0.0 - pressure[i*resX+j-1]) / dx;
    }
    else
    {
        pressure_gradx = (pressure[i*resX+j] - pressure[i*resX+j - 1]) / dx;
    }
    velX[i*resX_1+j] = velX[i*resX_1+j] - dt / density * pressure_gradx;
}

__global__
void make_velY_incompressible(float* velY, float* pressure, unsigned char* solid_map, unsigned char* air_map, int resX, int resY, float dx, float dt, float density) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
  
    int num_cells = resX * (resY+1);
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    unsigned char st = solid_map[i * (resX + 2) + j + 1]; unsigned char sb = solid_map[(i + 1) * (resX + 2) + j + 1];
    unsigned char at = air_map[i * (resX + 2) + j + 1]; unsigned char ab = air_map[(i + 1) * (resX + 2) + j + 1];

    float pressure_grady;
    //solid boundary condition, find the pressure that makes the velocity 0
    if (st) {
        float pt = pressure[i*resX+j] + (density * dx / dt) * (velY[i*resX+j] - 0.0);
        pressure_grady = (pt - pressure[i*resX+j]) / dx;
    }
    else if (sb) {
        float pb = pressure[(i - 1)*resX+j] + (density * dx / dt) * (0.0 - velY[i*resX+j]);
        pressure_grady = (pressure[(i - 1)*resX+j] - pb) / dx;
    }
    //air boundary condition, the pressure is 0, set the i=-1 or i=RESY to 0 if it is not solid to avoid error
    else if (at)
    {
        pressure_grady = (0.0 - pressure[i*resX+j]) / dx;
    }
    else if (ab)
    {
        pressure_grady = (pressure[(i - 1)*resX+j] - 0.0) / dx;
    }
    //air boundary condition, the pressure is 0, set the i=-1 or i=RESY to 0 if it is not solid to avoid error
    else
    {
        pressure_grady = (pressure[(i - 1)*resX+j] - pressure[i*resX+j]) / dx;
    }
    velY[i*resX+j] = velY[i*resX+j] - dt / density * pressure_grady;
}
 

//DEVICE HELPER FUNCTIONS
CUDA_D
float sample_scalar_field(float* field, Vec2 pos, int resX, int resY)
{
    // Clamp position to valid index range
    float x = CLAMP(pos.x, 0.0f, static_cast<float>(resX-1));
    float y = CLAMP(pos.y, 0.0f, static_cast<float>(resY-1));

    // Integer cell indices
    int j0 = static_cast<int>(floorf(x));
    int i0 = static_cast<int>(floorf(y));

    int j1 = min(j0 + 1, resX-1);        // x index: 0 .. RESX
    int i1 = min(i0 + 1, resY-1);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    float v00 = field[i0*resX+j0];
    float v10 = field[i0*resX+j1];
    float v01 = field[i1*resX+j0];
    float v11 = field[i1*resX+j1];

    // Bilinear interpolation
    float vx0 = v00 * (1.0f - tx) + v10 * tx;
    float vx1 = v01 * (1.0f - tx) + v11 * tx;

    return vx0 * (1.0f - ty) + vx1 * ty;
}

CUDA_D
Vec2 sample_vec2_field(Vec2* field, Vec2 pos, int resX, int resY)
{
    // Clamp position to valid index range
    float x = CLAMP(pos.x, 0.0f, static_cast<float>(resX - 1));
    float y = CLAMP(pos.y, 0.0f, static_cast<float>(resY - 1));

    // Integer cell indices
    int j0 = static_cast<int>(floorf(x));
    int i0 = static_cast<int>(floorf(y));

    int j1 = min(j0 + 1, resX - 1);        // x index: 0 .. RESX
    int i1 = min(i0 + 1, resY - 1);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    Vec2 v00 = field[i0 * resX + j0];
    Vec2 v10 = field[i0 * resX + j1];
    Vec2 v01 = field[i1 * resX + j0];
    Vec2 v11 = field[i1 * resX + j1];

    // Bilinear interpolation x
    float vx0x = v00.x * (1.0f - tx) + v10.x * tx;
    float vx1x = v01.x * (1.0f - tx) + v11.x * tx;

    // Bilinear interpolation y
    float vx0y = v00.y * (1.0f - tx) + v10.y * tx;
    float vx1y = v01.y * (1.0f - tx) + v11.y * tx;

    return Vec2(vx0x * (1.0f - ty) + vx1x * ty, vx0y * (1.0f - ty) + vx1y * ty);
}

CUDA_D
Vec3 sample_vec3_field(Vec3* field, Vec2 pos, int resX, int resY)
{
    // Clamp position to valid index range
    float x = CLAMP(pos.x, 0.0f, static_cast<float>(resX - 1));
    float y = CLAMP(pos.y, 0.0f, static_cast<float>(resY - 1));

    // Integer cell indices
    int j0 = static_cast<int>(floorf(x));
    int i0 = static_cast<int>(floorf(y));

    int j1 = min(j0 + 1, resX - 1);        // x index: 0 .. RESX
    int i1 = min(i0 + 1, resY - 1);    // y index: 0 .. RESY-1

    // Fractions inside the cell
    float tx = x - static_cast<float>(j0);
    float ty = y - static_cast<float>(i0);

    Vec3 v00 = field[i0 * resX + j0];
    Vec3 v10 = field[i0 * resX + j1];
    Vec3 v01 = field[i1 * resX + j0];
    Vec3 v11 = field[i1 * resX + j1];

    // Bilinear interpolation x
    float vx0x = v00.x * (1.0f - tx) + v10.x * tx;
    float vx1x = v01.x * (1.0f - tx) + v11.x * tx;

    // Bilinear interpolation y
    float vx0y = v00.y * (1.0f - tx) + v10.y * tx;
    float vx1y = v01.y * (1.0f - tx) + v11.y * tx;

    // Bilinear interpolation z
    float vx0z = v00.z * (1.0f - tx) + v10.z * tx;
    float vx1z = v01.z * (1.0f - tx) + v11.z * tx;

    return Vec3(vx0x * (1.0f - ty) + vx1x * ty, vx0y * (1.0f - ty) + vx1y * ty, vx0z * (1.0f - ty) + vx1z * ty);
}


//INITIALIZATION
FluidSolverGPU::FluidSolverGPU() : ResX(RESXGPU), ResY(RESYGPU), num_cells(ResX* ResY), block_size(256) {

    show_field_type = VisualizeField::Smoke;
    //initialize the gpu memory
    CUDA_CHECK(cudaMalloc(&velX, (ResY * (ResX + 1)) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&velX_temp, (ResY * (ResX + 1)) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&velY, ((ResY + 1) * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&velY_temp, ((ResY + 1) * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vel_center, ResX * ResY * sizeof(Vec2)));
    CUDA_CHECK(cudaMalloc(&vel_magnitude, ResX * ResY * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&setter_external_vel, ResX * ResY * sizeof(Vec2)));
    CUDA_CHECK(cudaMalloc(&adder_external_vel, ResX * ResY * sizeof(Vec2)));
    CUDA_CHECK(cudaMalloc(&pressure_new, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&pressure_old, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&smoke, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&color, (ResY*ResX) * sizeof(Vec3)));
    CUDA_CHECK(cudaMallocManaged(&swap_color, (ResY*ResX) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&color_inflow, (ResY*ResX) * sizeof(Vec3)));
    CUDA_CHECK(cudaMalloc(&swap_smoke, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&temperature, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&vorticity, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&swap_temperature, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&divergence, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&air_map, (ResY+2)*(ResX+2) * sizeof(char)));
    CUDA_CHECK(cudaMallocManaged(&solid_map, (ResY+2) * (ResX+2) * sizeof(char)));
    CUDA_CHECK(cudaMalloc(&scene_bytes, (ResY * ResX) * sizeof(float)));
    CUDA_CHECK(cudaMallocManaged(&smoke_inflow_map, ResY * ResX * sizeof(char)));

    //initialize cpu memory
    host_field_size = ResX * ResY * sizeof(float);
    host_vector_field_size = ResX * ResY * sizeof(Vec2);
    host_field = (float*)malloc(host_field_size);
    host_vector_field = (Vec2*)malloc(host_vector_field_size);

    construct_velocity_center();

    //copy memories
    CUDA_CHECK(cudaMemcpy(host_field, smoke, host_field_size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_vector_field, vel_center, host_vector_field_size, cudaMemcpyDeviceToHost));
}

void FluidSolverGPU::add_temperature_inflow() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    add_block_inflow_bottom_kernel<<<grid_size, block_size >>>(temperature, ResX, ResY, 70.0);
}

void FluidSolverGPU::add_smoke_inflow() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    //add_block_inflow_bottom_kernel<<<grid_size, block_size >>>(smoke, ResX, ResY);
    add_block_inflow_from_map <<<grid_size, block_size >>> (smoke, smoke_inflow_map, color, color_inflow, ResX, ResY, 1.0);
}

void FluidSolverGPU::set_solid_map_on_GPU(const unsigned char* s_map) {
    size_t map_size = (ResX + 2) * (ResY + 2) * sizeof(char);
    CUDA_CHECK(cudaMemcpy(solid_map, s_map, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_air_map_on_GPU(const unsigned char* a_map) {
    size_t map_size = (ResX + 2) * (ResY + 2) * sizeof(char);
    CUDA_CHECK(cudaMemcpy(air_map, a_map, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_smoke_field_on_GPU(const float* smoke_field) {
    size_t map_size = (ResX) * (ResY) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(smoke, smoke_field, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_smoke_inflow_map_on_GPU(const unsigned char* s_inflow_map) {
    size_t map_size = (ResX) * (ResY) * sizeof(char);
    CUDA_CHECK(cudaMemcpy(smoke_inflow_map, s_inflow_map, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_divergence_on_GPU(const float* div) {
    size_t map_size = (ResX) * (ResY) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(divergence, div, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_vorticity_on_GPU(const float* vort) {
    size_t map_size = (ResX) * (ResY) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(vorticity, vort, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_pressure_on_GPU(const float* press) {
    size_t map_size = (ResX) * (ResY) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(pressure_new, press, map_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(pressure_old, press, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_vel_field_on_GPU(const float* vX, const float* vY) {
    size_t map_size = (ResX+1) * (ResY) * sizeof(float);
    CUDA_CHECK(cudaMemcpy(velX, vX, map_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(velY, vY, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_setter_external_vel_field_on_GPU(const Vec2* ext_vel) {
    size_t map_size = ResX * ResY * sizeof(Vec2);
    CUDA_CHECK(cudaMemcpy(setter_external_vel, ext_vel, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_adder_external_vel_field_on_GPU(const Vec2* ext_vel) {
    size_t map_size = ResX * ResY * sizeof(Vec2);
    CUDA_CHECK(cudaMemcpy(adder_external_vel, ext_vel, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_color_field_on_GPU(const Vec3* c) {
    size_t map_size = ResX * ResY * sizeof(Vec3);
    CUDA_CHECK(cudaMemcpy(color, c, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::set_color_inflow_on_GPU(const Vec3* c) {
    size_t map_size = ResX * ResY * sizeof(Vec3);
    CUDA_CHECK(cudaMemcpy(color_inflow, c, map_size, cudaMemcpyHostToDevice));
}

void FluidSolverGPU::construct_velocity_center() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    construct_vel_center_kernel <<< grid_size, block_size >>> (vel_center, velX, velY, ResX, ResY);
}

void FluidSolverGPU::compute_vorticity() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    compute_vorticity_kernel <<<grid_size, block_size >>> (vorticity, vel_center, ResX, ResY, dx);
}

void FluidSolverGPU::compute_divergence() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    compute_divergence_kernel<<<grid_size, block_size>>>(divergence, velX, velY, ResX, ResY, dx);
}

//Solver Functions
void FluidSolverGPU::solve_smoke() {
    //add_temperature_inflow();
    add_smoke_inflow();

    //simulation steps
    diffuse_smoke();
    advect_quantities();
    add_external_force();
    project();
    determine_time_step();

    compute_divergence();
    construct_velocity_center();
    compute_vorticity();
}

//SIMULATION FUNCTIONS
void FluidSolverGPU::determine_time_step()
{
    const float cfl = 0.5f;     
    const float dtMax = 1.0f / 60.0f;
    const float eps = 1e-6f;

    float maxAbsU = thrust::transform_reduce(
        thrust::device,
        velX, velX + (ResX + 1) * ResY,
        AbsVal(),
        0.0f,
        thrust::maximum<float>());

    float maxAbsV = thrust::transform_reduce(
        thrust::device,
        velY, velY + ResX * (ResY + 1),
        AbsVal(),
        0.0f,
        thrust::maximum<float>());

    float umax = fmaxf(maxAbsU, maxAbsV);

    float dtCfl = (umax > eps) ? (cfl * dx / umax) : dtMax;  // at rest, allow dtMax
    dt = fminf(dtCfl, dtMax);
}

void FluidSolverGPU::diffuse_smoke() {
    int grid_size = (num_cells + block_size - 1) / block_size;
    diffuse_scalar_field_kernel<<<grid_size, block_size>>>(smoke, swap_smoke, solid_map, air_map, ResX, ResY, dx, dt, diffus_factor);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(smoke, swap_smoke);
}

void FluidSolverGPU::advect_quantities() {
    /*
    advect quantities, velocity, temperature, density...
    */
    
    //advect smoke
    int grid_size = (num_cells + block_size - 1) / block_size;
    advect_quantity_kernel<<<grid_size, block_size>>>(smoke, swap_smoke, vel_center, ResX, ResY, dt);

    //advect color
    advect_vec3_kernel <<<grid_size, block_size >>> (color, swap_color, vel_center, ResX, ResY, dt);

    //advect temperature
    advect_quantity_kernel <<<grid_size, block_size >>> (temperature, swap_temperature, vel_center, ResX, ResY, dt);

    //advect velX
    int number_of_cells = (ResX + 1) * ResY;
    grid_size = (number_of_cells + block_size - 1) / block_size;
    advect_velocityX_kernel <<<grid_size, block_size >>> (velX, velX_temp, velY, ResX, ResY, dt);

    //advect velY
    number_of_cells = ResX * (ResY+1);
    grid_size = (number_of_cells + block_size - 1) / block_size;
    advect_velocityY_kernel <<<grid_size, block_size >>> (velY, velY_temp, velX, ResX, ResY, dt);
    //wait for the gpu before swapping
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(velX, velX_temp);
    std::swap(velY, velY_temp);
    std::swap(smoke, swap_smoke);
    std::swap(color, swap_color);
    std::swap(temperature, swap_temperature);
}

void FluidSolverGPU::add_external_force() {
    int number_of_cells = ResX * (ResY+1);
    int grid_size = (number_of_cells + block_size - 1) / block_size;

    //smoke case
    //add_external_force_smoke_kernel <<<grid_size, block_size>>>(velY, smoke, temperature, ResX, ResY, gravity, dt, density_alpha, bouyancy, T_amb);

    //wind tunnel case
    //add_external_force_wind_tunel_kernel <<<grid_size, block_size>>> (velX, ResX, ResY, wind_force);

    //set external forces
    number_of_cells = ResX * ResY;
    grid_size = (number_of_cells + block_size - 1) / block_size;
    add_external_force_kernel <<<grid_size, block_size>>>(velX, velY, adder_external_vel, setter_external_vel, ResX, ResY, dx, dt);

    add_vorticity_confinement_kernel <<<grid_size, block_size>>> (velX, velY, vorticity, ResX, ResY, dx, dt, vorticity_coeff);
}

void FluidSolverGPU::project() {
    //solve pressure with jacobi iteration
    int number_of_cells = ResX * ResY;
    int grid_size = (number_of_cells + block_size - 1) / block_size;
    for (int i = 0; i < jacobi_iteration; i++) {
        jacobi_pressure_solve<<<grid_size, block_size>>>(pressure_new, pressure_old, velX, velY, solid_map, air_map, ResX, ResY, density, dx, dt);
        if (i < jacobi_iteration-1) {
            std::swap(pressure_new, pressure_old);
        }
    }

    //make velocity incompressible
    number_of_cells = (ResX + 1) * ResY;
    grid_size = (number_of_cells + block_size - 1) / block_size;
    make_velX_incompressible <<<grid_size, block_size >>> (velX, pressure_new, solid_map, air_map, ResX, ResY, dx, dt, density);
    make_velY_incompressible <<<grid_size, block_size>>> (velY, pressure_new, solid_map, air_map, ResX, ResY, dx, dt, density);
}

//VISUALIZATION FUNCTIONS
void FluidSolverGPU::set_host_field() {
    //copy memory to show on CPU
    switch (show_field_type)
    {
    case VisualizeField::Smoke:
        CUDA_CHECK(cudaMemcpy(host_field, smoke, host_field_size, cudaMemcpyDeviceToHost));
        break;
    case VisualizeField::Pressure:
        CUDA_CHECK(cudaMemcpy(host_field, pressure_new, host_field_size, cudaMemcpyDeviceToHost));
        break;
    case VisualizeField::Divergence:
        CUDA_CHECK(cudaMemcpy(host_field, divergence, host_field_size, cudaMemcpyDeviceToHost));
        break;
    case VisualizeField::VelocityMagnitude:
        //compute the velocity magnitude
        thrust::transform(
            thrust::device,
            vel_center,
            vel_center + ResX*ResY,
            vel_magnitude,
            MagFunctor{}
        );
        CUDA_CHECK(cudaMemcpy(host_field, vel_magnitude, host_field_size, cudaMemcpyDeviceToHost));
        break;
    case VisualizeField::Vorticity:
        CUDA_CHECK(cudaMemcpy(host_field, vorticity, host_field_size, cudaMemcpyDeviceToHost));
        break;
    default:
        CUDA_CHECK(cudaMemcpy(host_field, smoke, host_field_size, cudaMemcpyDeviceToHost));
        break;
    }
    CUDA_CHECK(cudaMemcpy(host_vector_field, vel_center, host_vector_field_size, cudaMemcpyDeviceToHost));
}

std::vector<unsigned char> FluidSolverGPU::scalar_field_to_bytes(float normalize_factor = 1.0) {
    /*
    Transforms an array into bytes with a color mapping for visualization
    */
    std::vector<unsigned char> bytes;
    bytes.reserve(ResY*ResX*3);

    //Find the maximum velocity x and y
    float max = -std::numeric_limits<float>::max();
    float min = std::numeric_limits<float>::max();
    for (int i = 0; i < ResY; i++) {
        for (int j = 0; j < ResX; j++) {
            if (host_field[i* ResY + j] > max) {
                max = host_field[i* ResY +j];
            }
            if (host_field[i* ResY + j] < min) {
                min = host_field[i* ResY + j];
            }
        }
    }

    // For color mapping of velX, use symmetric range around 0
    float maxAbsX = std::max(std::fabs(min), std::fabs(max));
    if (maxAbsX == 0.0f) {
        maxAbsX = 1.0f; // avoid division by zero
    }

    //set the velocity values into the bytes
    for (int i = ResY - 1; i >= 0; i--) {
        for (int j = 0; j < ResX; j++) {
            //handle the solid blocks
            if (solid_map[(i + 1) * (ResX + 2) + (j + 1)]) {
                bytes.push_back(0.0);
                bytes.push_back(0.0);
                bytes.push_back(255.0);
                continue;
            }

            float t = host_field[i* RESXGPU +j] / normalize_factor; // maxAbsX;
            t = std::max(-1.0f, std::min(1.0f, t));

            float r = 0.0f, g = 0.0f, b = 0.0f;
            
            switch (show_field_type)
            {
            case VisualizeField::Smoke:
            {
                //apply_gray_map(t, r, g, b);
                Vec3 col = t * color[i * ResX + j];
                r = col.x; g = col.y; b = col.z;
                break;
            }
            case VisualizeField::Pressure:
                apply_red_blue_map(t, r, g, b);
                break;
            case VisualizeField::Divergence:
                apply_red_blue_map(t, r, g, b);
                break;
            case VisualizeField::Temperature:
                apply_red_blue_map(t, r, g, b);
                break;
            case VisualizeField::VelocityMagnitude:
                apply_gray_map(t, r, g, b);
                break;
            case VisualizeField::Vorticity:
                apply_red_blue_map(t, r, g, b);
                break;
            default:
                apply_red_blue_map(t, r, g, b);
                break;
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

std::vector<unsigned char> FluidSolverGPU::vector_field_to_bytes() {
    /*
    Fills the velocity bytes from the velocity arrays
    */
    std::vector<unsigned char> bytes;

    //Find the maximum velocity x and y
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    for (int i = 0; i < ResY; i++) {
        for (int j = 0; j < ResX; j++) {
            if (host_vector_field[i*ResX + j].x > max_x) {
                max_x = host_vector_field[i * ResX + j].x;
            }
            if (host_vector_field[i * ResX + j].y > max_y) {
                max_y = host_vector_field[i * ResX + j].y;
            }
            if (host_vector_field[i * ResX + j].x < min_x) {
                min_x = host_vector_field[i * ResX + j].x;
            }
            if (host_vector_field[i * ResX + j].y < min_y) {
                min_y = host_vector_field[i * ResX + j].y;
            }
        }
    }

    //set the velocity values into the bytes
    for (int i = ResY - 1; i >= 0; i--) {
        for (int j = 0; j < ResX; j++) {
            float vx = host_vector_field[i*ResX+j].x;
            float vy = host_vector_field[i * ResX + j].y;

            // Normalize to 0–1
            float nx = (vx - min_x) / (max_x - min_x);
            float ny = (vy - min_y) / (max_y - min_y);

            // Convert to byte 0–255
            unsigned char bx = static_cast<unsigned char>(nx * 255.0f);
            unsigned char by = static_cast<unsigned char>(ny * 255.0f);

            //set the velocity bytes height*width*3 (x,y, 0)
            bytes.push_back(bx); //R
            bytes.push_back(by); //G
            bytes.push_back(0);  //B
        }
    }
    return bytes;
}

void FluidSolverGPU::apply_red_blue_map(const float val, float& r, float& g, float& b) {
    //Color mapping for velX into vel_bytes (RGB) ---
    // t in [-1, 1]: negative -> blue, positive -> red
    if (val > 0.0f) {
        // positive: black -> red
        r = val;
        g = 0.0f;
        b = 0.0f;
    }
    else if (val < 0.0f) {
        // negative: black -> blue
        r = 0.0f;
        g = 0.0f;
        b = -val;         // t in [-1,0] -> 1..0
    }
    else {
        // exactly zero: black
        r = g = b = 0.0f;
    }
}

void FluidSolverGPU::apply_gray_map(const float val, float& r, float& g, float& b) {
    if (val > 0.0f) {
        // positive: black -> red
        r = val;
        g = val;
        b = val;
    }
    else {
        // negative: black -> blue
        r = 0.0f;
        g = 0.0f;
        b = 0.0f;
    }
}

FluidSolverGPU::~FluidSolverGPU() {
	CUDA_CHECK(cudaFree(velX));
	CUDA_CHECK(cudaFree(velY));
    CUDA_CHECK(cudaFree(velX_temp));
    CUDA_CHECK(cudaFree(velY_temp));
    CUDA_CHECK(cudaFree(vel_center));
    CUDA_CHECK(cudaFree(vel_magnitude));
    CUDA_CHECK(cudaFree(setter_external_vel));
    CUDA_CHECK(cudaFree(adder_external_vel));
	CUDA_CHECK(cudaFree(pressure_new));
    CUDA_CHECK(cudaFree(pressure_old));
    CUDA_CHECK(cudaFree(smoke));
    CUDA_CHECK(cudaFree(swap_smoke));
    CUDA_CHECK(cudaFree(temperature));
    CUDA_CHECK(cudaFree(swap_temperature));
    CUDA_CHECK(cudaFree(divergence));
    CUDA_CHECK(cudaFree(solid_map));
    CUDA_CHECK(cudaFree(air_map));
    CUDA_CHECK(cudaFree(scene_bytes));
    CUDA_CHECK(cudaFree(smoke_inflow_map));
    CUDA_CHECK(cudaFree(color));
    CUDA_CHECK(cudaFree(swap_color));
    CUDA_CHECK(cudaFree(color_inflow));
    CUDA_CHECK(cudaFree(vorticity));

    free(host_field);
    free(host_vector_field);
}