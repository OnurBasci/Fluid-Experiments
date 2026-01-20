#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <filesystem>
#include "shader.h"
#include "stb_image.h"
#include "Texture.h"
#include "TextureGenerator.h"
#include "FluidSolver.h"
#include "File.cuh"
#include "FluidSolverGPU.cuh"
#include "VisualizeField.h"
#include "FieldInitializer.h"

#include "imgui.h"
#include"imgui_impl_glfw.h"
#include"imgui_impl_opengl3.h"

#include <iostream>

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// settings
const unsigned int SCR_WIDTH = 900;
const unsigned int SCR_HEIGHT = 900;

int main()
{

    add_vectors();

    // glfw: initialize and configure window
    // ------------------------------
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window creation
    // --------------------
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "LearnOpenGL", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // glad: load all OpenGL function pointers
    // ---------------------------------------
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // build and compile our shader zprogram
    // ------------------------------------
    auto base = std::filesystem::current_path(); // or your executable dir
    Shader ourShader((base / "vertex.vs").string().c_str(),
        (base / "frag.fs").string().c_str());

    // set up vertex data (and buffer(s)) and configure vertex attributes
    // ------------------------------------------------------------------
    float vertices[] = {
        // positions          // colors           // texture coords
         1.0f,  1.0f, 0.0f,   1.0f, 0.0f, 0.0f,   1.0f, 1.0f, // top right
         1.0f, -1.0f, 0.0f,   0.0f, 1.0f, 0.0f,   1.0f, 0.0f, // bottom right
        -1.0f, -1.0f, 0.0f,   0.0f, 0.0f, 1.0f,   0.0f, 0.0f, // bottom left
        -1.0f,  1.0f, 0.0f,   1.0f, 1.0f, 0.0f,   0.0f, 1.0f  // top left 
    };
    unsigned int indices[] = {
        0, 1, 3, // first triangle
        1, 2, 3  // second triangle
    };
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // color attribute
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1);
    // texture coord attribute
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
    glEnableVertexAttribArray(2);


    bool use_gpu = true;

    FluidSolverGPU fluid_solverGPU;
    FluidSolver fluid_solver;
    int show_field_index = 0; //index of the field to show
    int draw_field_index = 0;
    bool simulation_started=false;
    float brush_size = 5.0;
    int wind_direction = 0;
    Vec2 prev_mouse_pos(0,0);
    float smoke_color[3] = { 1.0f, 1.0f, 1.0f };
    float vorticity_coefficient = 10.0;

    //initialize field
    FieldInitializer fieldInitializer(fluid_solverGPU);

    Texture* texture1 = nullptr;
    if (use_gpu) {
        Texture tex(fluid_solverGPU.scalar_field_to_bytes(1.0), fluid_solverGPU.ResX, fluid_solverGPU.ResY, 3, "texture1", 0);
        texture1 = &tex;
    }
    else {
        Texture tex(fluid_solver.scene_bytes, fluid_solver.resX, fluid_solver.resY, 3, "texture1", 0);
        texture1 = &tex;
    }
    texture1->texUnit(ourShader, "texture1", 0);

    //setup IMGUI
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330");


    // render loop
    // -----------
    while (!glfwWindowShouldClose(window))
    {
        // input
        // -----
        processInput(window);

        // render
        // ------
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        //set new Imgui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        //solver loop
        if (simulation_started) {
            if (use_gpu) {
                //set the current field
                VisualizeField current_field = static_cast<VisualizeField>(show_field_index);
                fluid_solverGPU.show_field_type = current_field;
                fluid_solverGPU.solve_smoke();
                fieldInitializer.add_force_by_mouse_interaction(prev_mouse_pos);
            }
            else
            {
                fluid_solver.solve_smoke_wind_tunnel();
            }
        }
        else {
            //draw environment
            fieldInitializer.setup_environment_by_mouse_interaction(static_cast<DrawField>(draw_field_index), Vec3(smoke_color[0], smoke_color[1], smoke_color[2]));
        }

        //print the average ms per frame every 100 frames
        if (fluid_solverGPU.frame_counter % 100 == 0 && fluid_solverGPU.frame_counter > 0) {
            std::cout << "Avg simulation time: " << (fluid_solverGPU.simulation_time / fluid_solverGPU.frame_counter) << " ms\n";
        }

        // render fluid
        texture1->Bind();
        if (use_gpu) {
            fluid_solverGPU.set_host_field();
            texture1->update_texture_data(fluid_solverGPU.scalar_field_to_bytes(1.0));
        }
        else {
            texture1->update_texture_data(fluid_solver.scene_bytes);
        }

        ourShader.use();
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        //create a UI window
        ImGui::SetNextWindowSize(ImVec2(400, 300), ImGuiCond_FirstUseEver);
        ImGui::Begin("Environment Parameters");
        //Set Visualization Field
        ImGui::Text("Visualization field");
        ImGui::RadioButton("Smoke##show", &show_field_index, 0);
        ImGui::RadioButton("Pressure", &show_field_index, 1);
        ImGui::RadioButton("Divergence", &show_field_index, 2);
        ImGui::RadioButton("Vel_Mag", &show_field_index, 3);
        ImGui::RadioButton("vorticity", &show_field_index, 4);
        //Set Drawing Field
        ImGui::Text("Draw Field");
        ImGui::Checkbox("add constant flow", &fieldInitializer.add_constant_inflow);
        ImGui::SameLine();
        ImGui::ColorEdit3("Smoke color", smoke_color, ImGuiColorEditFlags_NoInputs);
        ImGui::RadioButton("Smoke##draw", &draw_field_index, 0);
        ImGui::RadioButton("Solid", &draw_field_index, 1);
        if (ImGui::DragFloat("brush size", &brush_size, 1.0, 1.0, 50.0)) {
            fieldInitializer.brush_size = brush_size;
        }
        //add wind
        ImGui::Text("wind direction");
        if (ImGui::RadioButton("None", &wind_direction, 0)) {
            fieldInitializer.set_constant_velocity_inflow_from_border(wind_direction);
        }
        if (ImGui::RadioButton("right", &wind_direction, 1)) {
            fieldInitializer.set_constant_velocity_inflow_from_border(wind_direction);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("down", &wind_direction, 2)) {
            fieldInitializer.set_constant_velocity_inflow_from_border(wind_direction);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("left", &wind_direction, 3)) {
            fieldInitializer.set_constant_velocity_inflow_from_border(wind_direction);
        }
        ImGui::SameLine();
        if (ImGui::RadioButton("up", &wind_direction, 4)) {
            fieldInitializer.set_constant_velocity_inflow_from_border(wind_direction);
        }
        if (ImGui::Button("Wind Tunnel State")) {
            fieldInitializer.set_wind_tunnel();
        }
        if (ImGui::Button("Start Simulation")) {
            simulation_started = true;
        }
        if (ImGui::Button("Restart Simulation")) {
            fieldInitializer.reset_field(wind_direction);
            fluid_solverGPU.frame_counter = 0;
            fluid_solverGPU.simulation_time = 0.0;
            simulation_started = false;
        }
        ImGui::Text("Simulation Parameters");
        if (ImGui::SliderFloat("vort coeff", &vorticity_coefficient, 0, 50)) {
            fluid_solverGPU.vorticity_coeff = vorticity_coefficient;
        }
        ImGui::RadioButton("Jacobi Solver", &fluid_solverGPU.presure_solver_id, 0);
        ImGui::SameLine();
        ImGui::RadioButton("Gauss Seidel Solver", &fluid_solverGPU.presure_solver_id, 1);
        
        ImGui::End();

        //render UI
        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    //end Imgui
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    // optional: de-allocate all resources once they've outlived their purpose:
    // ------------------------------------------------------------------------
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);

    // glfw: terminate, clearing all previously allocated GLFW resources.
    // ------------------------------------------------------------------
    glfwTerminate();
    return 0;
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}