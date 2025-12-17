#pragma once
#include<vector>
#include "Vec2.h"
#include <cuda_runtime.h>
//#include <math_functions.h>
#include <device_launch_parameters.h>
#include <iostream>

#define CLAMP(x, a, b) (( (x) < (a) ) ? (a) : ( ((x) > (b)) ? (b) : (x) ))

constexpr int RESXGPU = 100;
constexpr int RESYGPU = 100;

class FluidSolverGPU {
public:
	int ResX;
	int ResY;

	//parameters
	float dt=0.001;
	float dt_coeff = 20;
	float dx = 1.0 / RESXGPU; //I suppose the fluid is in range [0,1][0,1]
	float density = 1.0;
	float gravity = 9.81;
	float bouyancy = 1.0;
	float density_alpha = 1.0; //coefficient for smoke external force
	float T_amb = 20.0;
	float T_incoming = 70.0;
	int jacobi_iteration = 100;

	//fields
	float* velX;
	float* velX_temp;
	float* velY;
	float* velY_temp;
	Vec2* vel_center;
	float* pressure_new;
	float* pressure_old;
	float* smoke;
	float* swap_smoke;
	float* divergence;
	unsigned char* solid_map;
	unsigned char* air_map;

	float* host_field;
	Vec2* host_vector_field;
	size_t host_field_size;
	size_t host_vector_field_size;

	//constannts
	const int num_cells;
	const int block_size;

	FluidSolverGPU();

	//initialization
	void initialize_fields();
	void initialize_environment();

	//solver functions
	void solve_smoke();
	void advect_quantities();
	void add_external_force();
	void project();
	void compute_divergence();

	//visualization helpers
	void construct_velocity_center();
	std::vector<unsigned char> scalar_field_to_bytes(float normalize_factor);
	std::vector<unsigned char> vector_field_to_bytes();

	~FluidSolverGPU();
};