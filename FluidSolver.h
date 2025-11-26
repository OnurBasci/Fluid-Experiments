#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H
#include<vector>
#include "Vec2.h"

const int RESX = 100;
const int RESY = 100;

class FluidSolver {
public:
	int resX;
	int resY;
	//staggered grid with array for x and y
	float velX[RESY][RESX +1];
	float velY[RESY+1][RESX];

	float dt = 1.0/RESX;
	float dx = 1.0 / RESX; //I suppose the fluid is in range [0,1][0,1]

	float density = 1.0;
	float gravity = 9.81;
	float bouyancy = 5.0;
	float density_alpha = 1.0; //coefficient for smoke external force
	float T_amb = 20.0;

	//center velocity field
	Vec2 vel_center[RESY][RESX];

	float pressure[RESY][RESX];
	float divergence[RESY][RESX];
	float smoke[RESY][RESX];
	float temperature[RESY][RESX];

	int gauss_seidel_iterations = 50;
	float gauss_seidel_over_relaxation = 1.7; //a parameter to accelerate pressure solve between 1 and 2
	float divergence_field = 0.0;

	//grids as bytes
	std::vector<unsigned char> velX_bytes;
	std::vector<unsigned char> velY_bytes;
	std::vector<unsigned char> vel_bytes;

	FluidSolver();
	void construct_vel_center();
	
	void initialize_sinusoidal_vel_field(float frequency, float amplitude);
	void initialize_smoke_field();
	void initialize_temperature_field();
	void add_smoke_inflow();
	void add_temperature_inflow();

	void solve_smoke();
    void solve();

	void determine_time_step();
	void advect_velocity();
	void advect_quantity(float arr[RESY][RESX]);
	void smoke_add_external_force();
	void project();
	void compute_divergence();
	void gauss_seidel_pressure_solve();
	void update_incompressible_velocity();

	float sample_velX(Vec2 pos);
	float sample_velY(Vec2 pos);
	float sample_quantity(float arr[RESY][RESX], Vec2 pos);

	void compute_divergence_field();
	void set_velocity_bytes();
	std::vector<unsigned char> array_to_bytes(float arr[RESY][RESX], float normalize_factor);
};

#endif