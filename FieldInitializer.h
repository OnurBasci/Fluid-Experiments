#pragma once
#include "FluidConfig.h"
#include <iostream>
#include "Vec2.h"
#include "imgui.h"
#include <algorithm>
#include "FluidSolverGPU.cuh"
#include <vector>

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
	float brush_size = 5;
	FluidSolverGPU* fluidSolverGPU;

	//Predifined set ups
	void set_wind_tunnel();

	//Interactive set ups
	void update_solid_map_by_mouse_interaction();

	Vec2 convert_screen_pos_to_field_index(Vec2 screen_pos, Vec2 screen_res);
	std::vector<Vec2> get_brush_indices_from_index(Vec2 index);

	FieldInitializer(FluidSolverGPU& fluid_solver_gpu);
	~FieldInitializer();
};