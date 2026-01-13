#include "FieldInitializer.h"

FieldInitializer::FieldInitializer(FluidSolverGPU& fluid_solver_gpu): resX(RESXGPU), resY(RESYGPU){
    fluidSolverGPU = &fluid_solver_gpu;

    velX = (float*)malloc((resX+1)*resY*sizeof(float));
    velY = (float*)malloc(resX*(resY+1)*sizeof(float));
    divergence = (float*)malloc(resX*resY*sizeof(float));
    pressure = (float*)malloc(resX*resY*sizeof(float));
    smoke = (float*)malloc(resX*resY*sizeof(float));
	solid_map = (unsigned char*)malloc((resY + 2) * (resX + 2) * sizeof(char));
	air_map = (unsigned char*)malloc((resY + 2) * (resX + 2) * sizeof(char));
    smoke_inflow_map = (unsigned char*)malloc(resY*resX * sizeof(char));
    setter_external_vel = (Vec2*)malloc(resX * resY * sizeof(Vec2));
    adder_external_vel = (Vec2*)malloc(resX * resY * sizeof(Vec2));

    set_default_fields();
}

void FieldInitializer::set_default_fields(int wind_dir) {
    //set velocity fields
    for (int i = 0; i < resY; i ++) {
        for (int j = 0; j < resX+1; j++) {
            velX[i * (resX+1) + j] = 0;
        }
    }

    for (int i = 0; i < resY + 1; i++) {
        for (int j = 0; j < resX; j++) {
            velY[i * resX + j] = 0;
        }
    }

    //set resY*resX fields
    for (int i = 0; i < resY; i++) {
        for (int j = 0; j < resX; j++) {
            smoke[i * resX + j] = 0.0;
            smoke_inflow_map[i * resX + j] = false;
            adder_external_vel[i * resX + j] = Vec2(0,0);
            divergence[i * resX + j] = 0;
            pressure[i * resX + j] = 0;
        }
    }

    set_constant_velocity_inflow_from_border(wind_dir);

    //set solid an air blocks
    for (int i = 0; i < resY + 2; i++) {
        for (int j = 0; j < resX + 2; j++) {
            solid_map[i * (resX + 2) + j] = false;
            //set free surface on borders
            if (i == 0 || i == resY + 1 || j == 0 || j == resX + 1) {
                air_map[i * (resX + 2) + j] = true;
            }
            else {
                air_map[i * (resX + 2) + j] = false;
            }
        }
    }

    fluidSolverGPU->set_solid_map_on_GPU(solid_map);
    fluidSolverGPU->set_air_map_on_GPU(air_map);
    fluidSolverGPU->set_smoke_field_on_GPU(smoke);
    fluidSolverGPU->set_vel_field_on_GPU(velX, velY);
    fluidSolverGPU->set_smoke_inflow_map_on_GPU(smoke_inflow_map);
    fluidSolverGPU->set_adder_external_vel_field_on_GPU(adder_external_vel);
    fluidSolverGPU->set_divergence_on_GPU(divergence);
    fluidSolverGPU->set_pressure_on_GPU(pressure);
}

void FieldInitializer::reset_field(int wind_dir) {
    set_default_fields(wind_dir);
}

void FieldInitializer::set_wind_tunnel() {
    //set constant velocity coming from left
    set_constant_velocity_inflow_from_border(1);


    //set solid blocks
    for (int i = 0; i < resY + 2; i++) {
        for (int j = 0; j < resX + 2; j++) {
            //make the border solid
            if (i == 2 || i == resY - 1) {
                solid_map[i * (resX + 2) + j] = true;
            }
            else
            {
                solid_map[i * (resX + 2) + j] = false;
            }

            //draw sphere obstical in the center
            Vec2 offset(-resX / 4, 0);
            float radius = (resX / 8);
            if (std::pow((i - resY / 2 - offset.y), 2) + std::pow((j - resX / 2 - offset.x), 2) < radius * radius) {
                solid_map[i * (resX + 2) + j] = true;
            }
        }
    }

    //set incoming smoke from left
    for (int i = 0; i < resY; i++) {
        for (int j = 0; j < resX; j++) {
            int center = resX / 2;

            if (j == 10 && i > center - 0.2 * center && i < center + 0.2 * center) {
                smoke[i * resX + j] = 1.0;
                smoke_inflow_map[i * resX + j] = true;
            }
        }
    }

    //set air blocks
    for (int i = 0; i < resY + 2; i++) {
        for (int j = 0; j < resX + 2; j++) {
            //make the border air
            if (i == 0 || i == resY + 1 || j == 0 || j == resX + 1) {
                air_map[i * (resX + 2) + j] = true;
            }
            else {
                air_map[i * (resX + 2) + j] = false;
            }
        }
    }

    //set the fields to fluid simulater
    fluidSolverGPU->set_solid_map_on_GPU(solid_map);
    fluidSolverGPU->set_air_map_on_GPU(air_map);
    fluidSolverGPU->set_smoke_inflow_map_on_GPU(smoke_inflow_map);
    fluidSolverGPU->set_smoke_field_on_GPU(smoke);
}

void FieldInitializer::set_constant_velocity_inflow_from_border(const int border_index) {
    /*
    adds constant velocity from borders id = 0 left, id = 1 top, id = 2 right, id = 3 bottom
    */

    for (int i = 0; i < resY; i++) {
        for (int j = 0; j < resX; j++) {

            setter_external_vel[i * resX + j] = Vec2(0, 0);

            switch (border_index)
            {
            case 0:
                break;
            case 1:
                //wind coming from left
                setter_external_vel[i * resX + j] = Vec2(0, 0);
                if (j == 0) {
                    setter_external_vel[i * resX + j] = Vec2(wind_force, 0);
                }
                break;
            case 2:
                //wind coming from top
                setter_external_vel[i * resX + j] = Vec2(0, 0);
                if (i == 0) {
                    setter_external_vel[i * resX + j] = Vec2(0, -wind_force);
                }
                break;
            case 3:
                //wind coming from right
                if (j == resX-2) {
                    setter_external_vel[i * resX + j] = Vec2(-wind_force, 0);
                }
                break;
            case 4:
                //wind coming from bottom
                if (i == resY-2) {
                    setter_external_vel[i * resX + j] = Vec2(0, wind_force);
                }
                break;
            default:
                break;
            }

        }
    }

    fluidSolverGPU->set_setter_external_vel_field_on_GPU(setter_external_vel);
}

void FieldInitializer::setup_environment_by_mouse_interaction(DrawField draw_field) {
    /*
    Sets the clicked block as a solid block
    */
    ImGuiIO& io = ImGui::GetIO();

    // If ImGui wants the mouse, do not handle it
    if (io.WantCaptureMouse)
        return;

    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        Vec2 center_id = convert_screen_pos_to_field_index(Vec2(mouse_pos.x, mouse_pos.y), Vec2(900, 900));

        std::vector<Vec2> brush_indices = get_brush_indices_from_index(center_id);

        for (Vec2 id : brush_indices) {
            int i = std::clamp(static_cast<int>(id.x),0 , resY+1);
            int j = std::clamp(static_cast<int>(id.y), 0, resX+1);

            if (draw_field == DrawField::Solid) {
                solid_map[i * (resX + 2) + j] = true;
            }
            else if (draw_field == DrawField::Smoke) {
                smoke[i * resX + j] = 1.0;
                if (add_constant_inflow) {
                    smoke_inflow_map[i * resX + j] = true;
                }
            }
        }
        fluidSolverGPU->set_solid_map_on_GPU(solid_map);
        fluidSolverGPU->set_smoke_field_on_GPU(smoke);
        fluidSolverGPU->set_smoke_inflow_map_on_GPU(smoke_inflow_map);
    }
}

void FieldInitializer::add_force_by_mouse_interaction(Vec2& prev_pos) {
    /*
    add velocity towards the direction of the mouse
    */
    ImGuiIO& io = ImGui::GetIO();

    // If ImGui wants the mouse, do not handle it
    if (io.WantCaptureMouse)
        return;

    if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        prev_pos = Vec2(mouse_pos.x, mouse_pos.y);
    }
    else if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        Vec2 center_id = convert_screen_pos_to_field_index(Vec2(mouse_pos.x, mouse_pos.y), Vec2(900, 900));
        std::vector<Vec2> brush_indices = get_brush_indices_from_index(center_id);
        Vec2 move_dir = Vec2(mouse_pos.x, mouse_pos.y) - prev_pos;
        move_dir = Vec2(move_dir.x, -move_dir.y);

        for (Vec2 id : brush_indices) {
            int i = std::clamp(static_cast<int>(id.x), 0, resY-1);
            int j = std::clamp(static_cast<int>(id.y), 0, resX-1);

            adder_external_vel[i * resX + j] = mouse_force* move_dir;
        }
        prev_pos = Vec2(mouse_pos.x, mouse_pos.y);
        fluidSolverGPU->set_adder_external_vel_field_on_GPU(adder_external_vel);
    }
    else if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
        for (int i = 0; i < resY; i++) {
            for (int j = 0; j < resX; j++) {
                adder_external_vel[i * resX + j] = Vec2(0, 0);
            }
        }
        fluidSolverGPU->set_adder_external_vel_field_on_GPU(adder_external_vel);
    }
}

Vec2 FieldInitializer::convert_screen_pos_to_field_index(Vec2 screen_pos, Vec2 screen_res) {
    float i = std::clamp((screen_pos.y * (resY / screen_res.y)), 0.0f, static_cast<float>(resY));
    float j = std::clamp((screen_pos.x * (resX / screen_res.x)), 0.0f, static_cast<float>(resX));
    Vec2 field_index(i, j);
    return field_index;
}

std::vector<Vec2> FieldInitializer::get_brush_indices_from_index(Vec2 index) {
    /*
    From an index position gets the indices around the index in the range of brush size
    */
    std::vector<Vec2> brush_indices;

    int radius = brush_size;
    int rSquared = radius * radius;

    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            // Circle check
            if (dx * dx + dy * dy <= rSquared)
            {
                brush_indices.emplace_back(
                    index.x + dx,
                    index.y + dy
                );
            }
        }
    }

    return brush_indices;

}


FieldInitializer::~FieldInitializer() {
    free(velX);
    free(velY);
    free(smoke);
	free(solid_map);
    free(divergence);
    free(pressure);
    free(air_map);
    free(smoke_inflow_map);
    free(setter_external_vel);
    free(adder_external_vel);
}