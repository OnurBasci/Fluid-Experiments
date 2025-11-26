#ifndef FLUID_SOLVER_H
#define FLUID_SOLVER_H
#include<vector>
#include "Vec2.h"

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

	float dt = 1.0 / RESX;
	float dx = 1.0 / RESX; //I suppose the fluid is in range [0,1][0,1]

	float density = 1.0;
	float gravity = 9.81;
	float bouyancy = 1.0;
	float density_alpha = 1.0; //coefficient for smoke external force
	float T_amb = 20.0;
	float T_incoming = 70.0;

	//staggered grid with array for x and y
	float velX[RESY][RESX +1];
	float velY[RESY+1][RESX];

	//center velocity field
	Vec2 vel_center[RESY][RESX];

	float pressure[RESY][RESX];
	float divergence[RESY][RESX];
	float smoke[RESY][RESX];
	float temperature[RESY][RESX];
	bool solid_map[RESX + 2][RESX + 2];
	bool air_map[RESX + 2][RESX + 2];
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
	void update_scene(VisualizeField field_type);

	float sample_velX(Vec2 pos);
	float sample_velY(Vec2 pos);
	float sample_quantity(float arr[RESY][RESX], Vec2 pos);

	void compute_divergence_field();
	void set_velocity_bytes();

	static void blue_red_color_map(float val, float normalization_factor, float& r, float& g, float& b);
	std::vector<unsigned char> array_to_bytes(float arr[RESY][RESX], float normalize_factor);
};

#endif