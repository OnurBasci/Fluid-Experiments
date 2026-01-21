cmake_minimum_required(VERSION 3.24)

project(FluidExperiments
  VERSION 0.1.0
  LANGUAGES C CXX CUDA
)

# ----------------------------
# Standards
# ----------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Helpful default (users can still override with -DCMAKE_BUILD_TYPE=Release)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ----------------------------
# Options
# ----------------------------
option(FE_COPY_RUNTIME_ASSETS "Copy shaders/textures next to the built executable" ON)

# ----------------------------
# Dependencies: OpenGL
# ----------------------------
find_package(OpenGL REQUIRED)

# ----------------------------
# Dependencies: GLFW
# Try config package first, then pkg-config as a fallback.
# ----------------------------
set(GLFW_TARGET "")
find_package(glfw3 CONFIG QUIET)
if(glfw3_FOUND)
  # Common targets are glfw or glfw3 depending on package.
  if(TARGET glfw)
    set(GLFW_TARGET glfw)
  elseif(TARGET glfw3)
    set(GLFW_TARGET glfw3)
  elseif(TARGET glfw::glfw)
    set(GLFW_TARGET glfw::glfw)
  endif()
endif()

if(GLFW_TARGET STREQUAL "")
  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(GLFW3 QUIET glfw3)
    if(GLFW3_FOUND)
      add_library(glfw3_pkg INTERFACE)
      target_include_directories(glfw3_pkg INTERFACE ${GLFW3_INCLUDE_DIRS})
      target_link_libraries(glfw3_pkg INTERFACE ${GLFW3_LIBRARIES})
      target_link_directories(glfw3_pkg INTERFACE ${GLFW3_LIBRARY_DIRS})
      set(GLFW_TARGET glfw3_pkg)
    endif()
  endif()
endif()

if(GLFW_TARGET STREQUAL "")
  message(FATAL_ERROR
    "Could not find GLFW. Install it via vcpkg/your package manager, or provide a glfw3Config.cmake."
  )
endif()

# ----------------------------
# Dependencies: CUDA Toolkit
# ----------------------------
# Uses modern CMake CUDA Toolkit targets (CUDA::cudart, etc.).
find_package(CUDAToolkit REQUIRED)

# ----------------------------
# Sources
# ----------------------------
# Root sources (cpp/c/cuda)
file(GLOB ROOT_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB ROOT_C   CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.c")
file(GLOB ROOT_CU  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# ImGui sources (assumes imgui/ is vendored in the repo)
# Typical structure: imgui/*.cpp and imgui/backends/*.cpp
file(GLOB IMGUI_CPP CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends/*.cpp"
)

# Create an ImGui static lib
add_library(imgui STATIC
  ${IMGUI_CPP}
)

target_include_directories(imgui PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends"
)

# Many ImGui OpenGL backend setups want this when using glad
target_compile_definitions(imgui PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)

# ----------------------------
# Main executable
# ----------------------------
add_executable(FluidExperiments
  ${ROOT_CPP}
  ${ROOT_C}
  ${ROOT_CU}
)

# Allow CUDA object linking across translation units if needed
set_target_properties(FluidExperiments PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)

# Includes: project root (headers live in repo root) + ImGui
target_include_directories(FluidExperiments PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui"
  "${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends"
)

# Link libraries
target_link_libraries(FluidExperiments PRIVATE
  imgui
  ${GLFW_TARGET}
  OpenGL::GL
  CUDA::cudart
)

target_link_libraries(imgui PUBLIC ${GLFW_TARGET})

# ----------------------------
# Platform-specific bits
# ----------------------------
if(WIN32)
  # OpenGL::GL usually maps correctly, but keep this for older toolchains.
  target_link_libraries(FluidExperiments PRIVATE opengl32)
elseif(APPLE)
  # OpenGL is deprecated on macOS but still works; GLFW brings required frameworks.
  # Nothing extra usually needed here.
else()
  # Linux: sometimes you need dl and pthread explicitly
  find_package(Threads REQUIRED)
  target_link_libraries(FluidExperiments PRIVATE Threads::Threads ${CMAKE_DL_LIBS})
endif()

# ----------------------------
# Warnings (optional but nice)
# ----------------------------
if(MSVC)
  target_compile_options(FluidExperiments PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:/W4>
  )
else()
  target_compile_options(FluidExperiments PRIVATE
    $<$<COMPILE_LANGUAGE:C,CXX>:-Wall -Wextra -Wpedantic>
  )
endif()

# ----------------------------
# Runtime assets: copy shaders/textures next to the executable
# (your repo root has vertex.vs / frag.fs and some images)
# ----------------------------
if(FE_COPY_RUNTIME_ASSETS)
  set(_ASSETS
    "${CMAKE_CURRENT_SOURCE_DIR}/vertex.vs"
    "${CMAKE_CURRENT_SOURCE_DIR}/frag.fs"
    "${CMAKE_CURRENT_SOURCE_DIR}/awesomeface.png"
    "${CMAKE_CURRENT_SOURCE_DIR}/bulga_cat.jpg"
    "${CMAKE_CURRENT_SOURCE_DIR}/container.jpg"
  )

  foreach(_f IN LISTS _ASSETS)
    if(EXISTS "${_f}")
      add_custom_command(TARGET FluidExperiments POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${_f}"
          "$<TARGET_FILE_DIR:FluidExperiments>"
      )
    endif()
  endforeach()
endif()
