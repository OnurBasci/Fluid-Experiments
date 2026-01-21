## Requirements
Windows 10/11 (64-bit)
CUDA Toolkit ≥ 12.4
CMake ≥ 3.26
vcpkg 
Ninja

## Configuration
'''
.\vcpkg install glfw3:x64-windows
.\vcpkg install glfw3:x64-windows
'''

'''
cmake -S . -B build -G Ninja -DCMAKE_TOOLCHAIN_FILE=<VCPKG_ROOT>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows 
cmake --build build
'''
