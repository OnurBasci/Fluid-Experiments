#pragma once
#include "FluidConfig.h"
#include <iostream>
#include "Vec2.h"
#include "Vec3.h"
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
	float* divergence;
	float* pressure;
	unsigned char* solid_map;
	unsigned char* air_map;
	float* velX;
	float* velY;
	float* smoke;
	float* vorticity;
	Vec3* color;
	Vec3* color_inflow;

	Vec2* setter_external_vel; //velocity field containing external forces that is set directly to the velocity field on GPU (ex: constant wind)
	Vec2* adder_external_vel; //additional velocity field that is added to the velocity field each frame (ex: mouse forces)
	unsigned char* smoke_inflow_map; //additional velocity field that is added to the velocity field each frame (ex: mouse forces)
	float brush_size = 5;
	FluidSolverGPU* fluidSolverGPU;
	bool add_constant_inflow; //if true the value will be add every frame

	//paramaters
	float wind_force = 4.0;
	float mouse_force = 200;

	//Predifined set ups
	void set_default_fields(int wind_dir=0);
	void set_wind_tunnel();
	void reset_field(int wind_dir=0);
	void set_constant_velocity_inflow_from_border(int border_index);

	//Interactive set ups
	void setup_environment_by_mouse_interaction(DrawField draw_field, Vec3 smoke_color);
	void add_force_by_mouse_interaction(Vec2& prev_pos);

	Vec2 convert_screen_pos_to_field_index(Vec2 screen_pos, Vec2 screen_res);
	std::vector<Vec2> get_brush_indices_from_index(Vec2 index);

	FieldInitializer(FluidSolverGPU& fluid_solver_gpu);
	~FieldInitializer();
};