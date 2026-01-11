#include "FieldInitializer.h"

FieldInitializer::FieldInitializer(FluidSolverGPU& fluid_solver_gpu): resX(RESXGPU), resY(RESYGPU){
    fluidSolverGPU = &fluid_solver_gpu;

	solid_map = (unsigned char*)malloc((resY + 2) * (resX + 2) * sizeof(char));
	air_map = (unsigned char*)malloc((resY + 2) * (resX + 2) * sizeof(char));

    for (int i = 0; i < resY + 2; i++) {
        for (int j = 0; j < resX + 2; j++) {
            solid_map[i * (resX + 2) + j] = false;
            //set the air map on the borders
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
}

void FieldInitializer::set_wind_tunnel() {
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
}

void FieldInitializer::update_solid_map_by_mouse_interaction() {
    /*
    Sets the clicked block as a solid block
    */
    ImGuiIO& io = ImGui::GetIO();

    // If ImGui wants the mouse, DO NOT handle it
    if (io.WantCaptureMouse)
        return;

    if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        ImVec2 mouse_pos = ImGui::GetMousePos();
        Vec2 center_id = convert_screen_pos_to_field_index(Vec2(mouse_pos.x, mouse_pos.y), Vec2(900, 900));

        std::vector<Vec2> brush_indices = get_brush_indices_from_index(center_id);

        for (Vec2 id : brush_indices) {
            int i = std::clamp(static_cast<int>(id.x),0 , resY+1);
            int j = std::clamp(static_cast<int>(id.y), 0, resX+1);
            solid_map[i * (resX + 2) + j] = true;
        }
        fluidSolverGPU->set_solid_map_on_GPU(solid_map);
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
	free(solid_map);
    free(air_map);
}