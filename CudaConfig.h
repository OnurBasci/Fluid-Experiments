#pragma once

// Make a macro that is __host__ __device__ only when compiled with NVCC
#ifdef __CUDACC__
	#define HD __host__ __device__
	#define H __host__
	#define D __device__
#else
	#define HD
	#define H
	#define D
#endif