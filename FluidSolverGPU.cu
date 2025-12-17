#include "FluidSolverGPU.cuh"

D
float sample_scalar_field(float* field, Vec2 pos, int resX, int resY);

//KERNELS
__global__
void initialize_sinusoidal_velX_kernel(float* velX, float frequency, float amplitude, int resX, int resY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //handle the out of bounds
    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    const float dx = 1.0f / static_cast<float>(resX);
    const float dy = 1.0f / static_cast<float>(resY);

    float x = dx * j;
    float y = dy * i;

    velX[i * resX + j] = 0.0; // cosf(x * frequency)* amplitude;
}

__global__
void initialize_sinusoidal_velY_kernel(float* velY, float frequency, float amplitude, int resX, int resY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    //handle the out of bounds
    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    const float dx = 1.0f / static_cast<float>(resX);
    const float dy = 1.0f / static_cast<float>(resY);

    float x = dx * j;
    float y = dy * i;

    velY[i * resX + j] = 0.0;// cosf(y * frequency)* amplitude;
}

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
void initialize_smoke_field(float* smoke, int resX, int resY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX; 
    int j = idx % resX;

    // Center of the domain (in indices)
    float cx = (resY - 1) * 0.5f;
    float cy = (resX - 1) * 0.5f;
    float radius = 0.2*resX;
    float r2 = radius * radius;

    float dx = (float)i - cx;
    float dy = (float)j - cy;

    float dist2 = dx * dx + dy * dy;

    smoke[i * resX + j] = (dist2 < r2) ? 1.0f : 0.0f;
}

__global__
void add_temperature_inflow_kernel(float* temperature, int resX, int resY) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * resY;
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    if (i > resY - 8 && i < resY - 1 && j > resX / 2 - resX / 6 && j < resX / 2 + resX / 6) {
        temperature[i*resX+j] = 70.0;
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

    float vy = velY[i*resX+j];
    float vx;

    if (i == 0)
        vx = (velX[i*resX+j] + velX[i*resX+j + 1])/ 2.0;
    else if (i == resY)
        vx = (velX[(i - 1)*resX+j] + velX[(i - 1)*resX+j + 1])/ 2.0;
    else
        vx = (velX[(i - 1)*resX+j] + velX[(i - 1)*resX+j+1] + velX[i*resX+j] + velX[i*resX+j+1]) / 4.0;
    //get the velocity vector at U_i+1/2, j
    Vec2 dir(vx*resX, -vy*resY); //velocity in index coordinates

    //Semi lagrangian advection
    Vec2 prev_pos = Vec2(static_cast<float>(j), static_cast<float>(i)) - dt * dir;

    velY_temp[i * resX + j] = sample_scalar_field(velY, prev_pos, resX, resY+1);
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
    /*if (i == resY / 2 && j == resX / 2) {
        printf("i: %d j: %d center val %f \n", i, j, swap_field[i * resX + j]);
    }*/
}

__global__
void add_external_force_kernel(float* velY, float* smoke, float* temperature, int resX, int resY, float gravity, float dt, float d_a, float b, float T_amb) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    int num_cells = resX * (resY + 1);
    if (idx >= num_cells) return;

    int i = idx / resX;
    int j = idx % resX;

    //add bouyouncy force depending on the smoke density and temperature
    float bouyouncy_force = -d_a * sample_scalar_field(smoke, Vec2(j, i - 0.5), resX, resY) + b * (sample_scalar_field(temperature, Vec2(j, i - 0.5), resX, resY) - T_amb);
    float g_force = -gravity;

    velY[i * resX + j] += dt * (bouyouncy_force + gravity);;
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
    //float pressure_old = pressure(i, j);
    //pressure(i, j) = pressure_old + (pressure_new - pressure_old) * gauss_seidel_over_relaxation;
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
D
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

    float interpolation = vx0 * (1.0f - ty) + vx1 * ty;

    return vx0 * (1.0f - ty) + vx1 * ty;
}


//INITIALIZATION
FluidSolverGPU::FluidSolverGPU() : ResX(RESXGPU), ResY(RESYGPU), num_cells(ResX* ResY), block_size(256) {

    std::cout << "res y " << ResY << "res x" << ResX << std::endl;
    //initialize the gpu memory
    cudaMalloc(&velX, (ResY * (ResX + 1)) * sizeof(float));
    cudaMalloc(&velX_temp, (ResY * (ResX + 1)) * sizeof(float));
    cudaMalloc(&velY, ((ResY + 1) * ResX) * sizeof(float));
    cudaMalloc(&velY_temp, ((ResY + 1) * ResX) * sizeof(float));
    cudaMalloc(&vel_center, ResX * ResY * sizeof(Vec2));
    cudaMalloc(&pressure_new, (ResY * ResX) * sizeof(float));
    cudaMalloc(&pressure_old, (ResY * ResX) * sizeof(float));
    cudaMalloc(&smoke, (ResY * ResX) * sizeof(float));
    cudaMalloc(&swap_smoke, (ResY * ResX) * sizeof(float));
    cudaMalloc(&temperature, (ResY * ResX) * sizeof(float));
    cudaMalloc(&swap_temperature, (ResY * ResX) * sizeof(float));
    cudaMalloc(&divergence, (ResY * ResX) * sizeof(float));
    cudaMalloc(&air_map, (ResY+2)*(ResX+2) * sizeof(char));
    cudaMalloc(&solid_map, (ResY+2) * (ResX+2) * sizeof(char));

    //initialize cpu memory
    host_field_size = ResX * ResY * sizeof(float);
    host_vector_field_size = ResX * ResY * sizeof(Vec2);
    host_field = (float*)malloc(host_field_size);
    host_vector_field = (Vec2*)malloc(host_vector_field_size);

    initialize_fields();
    initialize_environment();
    construct_velocity_center();

    //copy memories
    cudaMemcpy(host_field, smoke, host_field_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_vector_field, vel_center, host_vector_field_size, cudaMemcpyDeviceToHost);
}

void FluidSolverGPU::initialize_fields() {

    //Sinusoidal velocity field initialization
    float frequency = 16; float amplitude = 2;
	int number_of_cells = (ResX+1) * ResY;
	int grid_size = (number_of_cells + block_size - 1) / block_size;
    initialize_sinusoidal_velX_kernel <<<grid_size, block_size>>>(velX, frequency, amplitude, ResX+1, ResY);

    //initialize velocity y
    number_of_cells = ResX * (ResY + 1);
    grid_size = (number_of_cells + block_size - 1) / block_size;
    initialize_sinusoidal_velY_kernel <<<grid_size, block_size >>> (velY, frequency, amplitude, ResX, ResY+1);  

    //initialize smoke field
    grid_size = (num_cells + block_size - 1) / block_size;
    initialize_smoke_field <<<grid_size, block_size>>> (smoke, ResX, ResY);
}

void FluidSolverGPU::add_temperature_inflow() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    add_temperature_inflow_kernel<<<grid_size, block_size >>>(temperature, ResX, ResY);
}

void FluidSolverGPU::initialize_environment() {
    //allocate cpu field memories
    float* temperature_host = (float *) malloc(ResX*ResY*sizeof(float));
    unsigned char* solid_map_host = (unsigned char*)malloc((ResY+2)*(ResX+2)*sizeof(char));
    unsigned char* air_map_host = (unsigned char*)malloc((ResY+2) * (ResX+2) * sizeof(char));

    //initialize solid field
    for (int i = 0; i < ResX + 2; i++) {
        for (int j = 0; j < ResY + 2; j++) {
            //make the border solid
            if (i == 0 || i == ResY + 1 || j==0 || j==ResX+1) {
                solid_map_host[i * (ResX + 2) + j] = true;
            }
            else
            {
                solid_map_host[i * (ResX + 2) + j] = false;
            }

            //draw sphere obstical in the center
            /*if (std::pow((i - RESY / 2), 2) + std::pow((j - RESX / 2), 2) < 50) {
                solid_map[i][j] = true;
            }*/
        }
    }

    //initialize air field
    for (int i = 0; i < ResX + 2; i++) {
        for (int j = 0; j < ResY + 2; j++) {
            //make the border air
            if (i == 0 || i == ResY + 1 || j == 0 || j == ResX + 1) {
                air_map_host[i*(ResX+2)+j] = true;
            }
            else {
                air_map_host[i * (ResX + 2) + j] = false;
            }
        }
    }

    //initialize temperature
    for (int i = 0; i < ResY; i++) {
        for (int j = 0; j < ResX; j++) {
            if (i > ResY - 8 && i < ResY - 1 && j > ResX / 2 - ResX / 6 && j < ResX / 2 + ResX / 6) {
                temperature_host[i*ResX+j] = T_incoming;
            }
            else
            {
                temperature_host[i*ResX+j] = T_amb;
            }
        }
    }

    //copy memory to gpu
    cudaMemcpy(temperature, temperature_host, host_field_size, cudaMemcpyHostToDevice);
    size_t map_size = (ResX + 2) * (ResY + 2) * sizeof(char);
    cudaMemcpy(solid_map, solid_map_host, map_size, cudaMemcpyHostToDevice);
    cudaMemcpy(air_map, air_map_host, map_size, cudaMemcpyHostToDevice);

    free(temperature_host);
    free(solid_map_host);
    free(air_map_host);
}

void FluidSolverGPU::construct_velocity_center() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    construct_vel_center_kernel <<< grid_size, block_size >>> (vel_center, velX, velY, ResX, ResY);
}

void FluidSolverGPU::compute_divergence() {
    int grid_size = (num_cells + block_size - 1) / block_size;

    compute_divergence_kernel<<<grid_size, block_size>>>(divergence, velX, velY, ResX, ResY, dx);
}

//Solver Functions
void FluidSolverGPU::solve_smoke() {

    add_temperature_inflow();

    //simulation steps
    advect_quantities();
    add_external_force();
    project();

    compute_divergence();
    construct_velocity_center();
    //copy memory to show on CPU
    cudaMemcpy(host_vector_field, vel_center, host_vector_field_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_field, temperature, host_field_size, cudaMemcpyDeviceToHost);
}


void FluidSolverGPU::advect_quantities() {
    /*
    advect quantities, velocity, temperature, density...
    */
    
    //advect smoke
    int grid_size = (num_cells + block_size - 1) / block_size;
    advect_quantity_kernel<<<grid_size, block_size>>>(smoke, swap_smoke, vel_center, ResX, ResY, dt);

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
    cudaDeviceSynchronize();
    std::swap(velX, velX_temp);
    std::swap(velY, velY_temp);
    std::swap(smoke, swap_smoke);
    std::swap(temperature, swap_temperature);
}

void FluidSolverGPU::add_external_force() {
    int number_of_cells = ResX * (ResY+1);
    int grid_size = (number_of_cells + block_size - 1) / block_size;

    add_external_force_kernel<<<grid_size, block_size>>>(velY, smoke, temperature, ResX, ResY, gravity, dt, density_alpha, bouyancy, T_amb);
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

//Visualization Helper
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

            //Color mapping for velX into vel_bytes (RGB) ---
            // t in [-1, 1]: negative -> blue, positive -> red
            float t = host_field[i* RESXGPU +j] / normalize_factor; // maxAbsX;
            t = std::max(-1.0f, std::min(1.0f, t));

            float r = 0.0f, g = 0.0f, b = 0.0f;
            if (t > 0.0f) {
                // positive: black -> red
                r = t;          
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

            /*if (i == ResY / 2 && j == ResX / 2) {
                std::cout << "CPU: " << r * 255.0f << std::endl;
            }*/

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

FluidSolverGPU::~FluidSolverGPU() {
	cudaFree(velX);
	cudaFree(velY);
    cudaFree(velX_temp);
    cudaFree(velY_temp);
    cudaFree(vel_center);
	cudaFree(pressure_new);
    cudaFree(pressure_old);
    cudaFree(smoke);
    cudaFree(swap_smoke);
    cudaFree(temperature);
    cudaFree(swap_temperature);
    cudaFree(divergence);
    cudaFree(solid_map);
    cudaFree(air_map);

    free(host_field);
    free(host_vector_field);
}