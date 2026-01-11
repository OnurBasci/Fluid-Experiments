#pragma once
#include "FluidConfig.h"
#include <iostream>
#include "Vec2.h"
#include "imgui.h"
#include <algorithm>
#include "FluidSolverGPU.cuh"
#include <vector>
#include "DrawField.h"

/*
This class is used to determine the initial state of the simulation.
It sets up the boundaries, smoke density, temperature...
*/

class FieldInitializer {
public:
	int resX;
	int resY;
	unsigned char* solid_map;
	unsigned char* air_map;
	float* velX;
	float* velY;
	float* smoke;
	unsigned char* smoke_inflow_map;
	float brush_size = 5;
	FluidSolverGPU* fluidSolverGPU;
	bool add_constant_inflow; //if true the value will be add every frame

	//Predifined set ups
	void set_default_fields();
	void set_wind_tunnel();
	void reset_field();

	//Interactive set ups
	void setup_environment_by_mouse_interaction(DrawField draw_field);

	Vec2 convert_screen_pos_to_field_index(Vec2 screen_pos, Vec2 screen_res);
	std::vector<Vec2> get_brush_indices_from_index(Vec2 index);

	FieldInitializer(FluidSolverGPU& fluid_solver_gpu);
	~FieldInitializer();
};