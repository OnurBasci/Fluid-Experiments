## Requirements
Windows 10/11 (64-bit) 

CUDA Toolkit ≥ 12.4

CMake ≥ 3.26

vcpkg 

Ninja

## Configuration
Install glfw3 and glad
```
.\vcpkg install glfw3:x64-windows
.\vcpkg install glad:x64-windows
```

Configure and build the project.
```
cmake -S . -B build -G -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows 
cmake --build build
```

This will generate a .exe file in the build directory.
