#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <thrust\universal_vector.h>

__global__ void addKernel(int* c, const int* a, const int* b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    c[i] = a[i] + b[i];
}

void add_vectors() {
    const int N = 256;
    size_t size = N * sizeof(int);

    int* h_a = new int[N];
    int* h_b = new int[N];
    int* h_c = new int[N];

    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 2 * i;
    }

    int* d_a, * d_b, * d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    addKernel <<< blocksPerGrid, threadsPerBlock, 0 >>> (d_c, d_a, d_b);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    std::cout << "h_c[10] = " << h_c[10] << std::endl; // should be 30

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}