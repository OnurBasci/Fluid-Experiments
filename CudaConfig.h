#pragma once

// Make a macro that is __host__ __device__ only when compiled with NVCC
#ifdef __CUDACC__
	#define CUDA_HD __host__ __device__
	#define CUDA_H __host__
	#define CUDA_D __device__
#else
	#define CUDA_HD
	#define CUDA_H
	#define CUDA_D
#endif