#pragma once
#include<vector>
#include "Vec2.h"
#include "Vec3.h"
#include <cuda_runtime.h>
//#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>
#include "VisualizeField.h"
#include "FluidConfig.h"
#include "CudaConfig.h"

#define CLAMP(x, a, b) (( (x) < (a) ) ? (a) : ( ((x) > (b)) ? (b) : (x) ))

struct AbsVal
{
	__host__ __device__
		float operator()(float x) const
	{
		return fabsf(x);
	}
};

struct MagFunctor {
	__host__ __device__
		float operator()(const Vec2& v) const {
		return sqrtf(v.x * v.x + v.y * v.y);
	}
};

class FluidSolverGPU {
public:
	int ResX;
	int ResY;

	//parameters
	float dt=0.0005;
	float dt_coeff = 20;
	float dx = 1.0 / RESXGPU; //I suppose the fluid is in range [0,1][0,1]
	float density = 1.0;
	float gravity = 9.81;
	float bouyancy = 2.0;
	float density_alpha = 0.0; //coefficient for smoke external force
	float T_amb = 20.0;
	float T_incoming = 70.0;
	float wind_force = 4.0;
	float diffus_factor = 0.00001;
	float vorticity_coeff = 10;
	int jacobi_iteration = 100;

	//fields
	float* velX;
	float* velX_temp;
	float* velY;
	float* velY_temp;
	Vec2* vel_center;
	float* vel_magnitude;
	float* pressure_new;
	float* pressure_old;
	float* smoke;
	Vec3* color;
	Vec3* color_inflow;
	Vec3* swap_color;
	Vec2* setter_external_vel; //velocity field containing external forces that is set directly to the velocity field
	Vec2* adder_external_vel; //additional velocity field that is added to the velocity field each frame
	float* swap_smoke;
	float* temperature;
	float* swap_temperature;
	float* divergence;
	float* vorticity;
	float* scene_bytes; //bytes to render at each frame
	unsigned char* solid_map;
	unsigned char* air_map;
	unsigned char* smoke_inflow_map;
	VisualizeField show_field_type;

	float* host_field;
	Vec2* host_vector_field;
	size_t host_field_size;
	size_t host_vector_field_size;

	//constannts
	const int num_cells;
	const int block_size;

	FluidSolverGPU();

	//initialization
	void add_temperature_inflow();
	void add_smoke_inflow();
	void set_solid_map_on_GPU(const unsigned char* s_map);
	void set_air_map_on_GPU(const unsigned char* a_map);
	void set_smoke_field_on_GPU(const float* s_field);
	void set_smoke_inflow_map_on_GPU(const unsigned char* s_field);
	void set_vel_field_on_GPU(const float* velX, const float* velY);
	void set_setter_external_vel_field_on_GPU(const Vec2* external_vel);
	void set_adder_external_vel_field_on_GPU(const Vec2* external_vel);
	void set_divergence_on_GPU(const float* div);
	void set_pressure_on_GPU(const float* press);
	void set_color_field_on_GPU(const Vec3* color);
	void set_color_inflow_on_GPU(const Vec3* color_inflow);
	void set_vorticity_on_GPU(const float* vort);

	//solver functions
	void determine_time_step();
	void solve_smoke();
	void diffuse_smoke();
	void advect_quantities();
	void add_external_force();
	void project();
	void compute_divergence();

	//visualization
	void set_host_field();
	void construct_velocity_center();
	void compute_vorticity();
	std::vector<unsigned char> scalar_field_to_bytes(float normalize_factor);
	std::vector<unsigned char> vector_field_to_bytes();
	void apply_red_blue_map(const float val, float& r, float& g, float& b);
	void apply_gray_map(const float val, float& r, float& g, float& b);

	~FluidSolverGPU();
};