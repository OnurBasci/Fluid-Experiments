#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H
#include<vector>
#include "Vec2.h"
#include "Field2D.h"
#include "CudaConfig.h"

const int RESX = 100;
const int RESY = 100;

enum class VisualizeField {
	Smoke,
	Temperature,
	VelocityMagnitude,
	Pressure,
	Divergence,
	Vorticity
};

class FluidSolver {
public:
	int resX;
	int resY;
	
	float dt;
	float dt_coeff = 20;
	float dx = 1.0 / RESX; //I suppose the fluid is in range [0,1][0,1]

	float density = 1.0;
	float gravity = 9.81;
	float bouyancy = 1.0;
	float density_alpha = 1.0; //coefficient for smoke external force
	float T_amb = 20.0;
	float T_incoming = 70.0;
	float wind_force = 4.0;

	//staggered grid with array for x and y
	Field2D<float> velX;
	Field2D<float> velX_temp; //for swaping
	Field2D<float> velY;
	Field2D<float> velY_temp;

	//center velocity field
	Field2D<Vec2> vel_center;

	Field2D<float> pressure;
	Field2D<float> divergence;
	Field2D<float> smoke;
	Field2D<float> temperature;
	Field2D<unsigned char> solid_map;
	Field2D<unsigned char> air_map;
	std::vector<unsigned char> scene_bytes;

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
	void initialize_environment();
	void add_smoke_inflow();
	void add_temperature_inflow();
	void add_block_inflow(Field2D<float>& field, Vec2 center, Vec2 size, float val);

	void solve_smoke_temperature();
	void solve_smoke_wind_tunnel();
    void solve();

	//CPU code for the simulation
	void determine_time_step();
	void advect_velocity();
	void advect_quantity(Field2D<float> &field);
	void smoke_add_external_force_temperature();
	void smoke_add_external_force_wind_tunnel();
	void project();
	void compute_divergence();
	void gauss_seidel_pressure_solve();
	void update_incompressible_velocity();
	void update_scene(VisualizeField field_type);

	float sample_velX(Vec2 pos);
	float sample_velY(Vec2 pos);
	float sample_quantity(Field2D<float> &field, Vec2 pos);

	void compute_divergence_field();
	void set_velocity_bytes();
	
	static void blue_red_color_map(float val, float normalization_factor, float& r, float& g, float& b);
	std::vector<unsigned char> array_to_bytes(float arr[RESY][RESX], float normalize_factor);
};

#endif