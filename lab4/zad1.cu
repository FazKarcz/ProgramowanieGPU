#include <cuda_runtime.h>
#include <iostream>

#define BLOCK_SIZE 512



__global__ void vectorScalarMulShared(float* d_out, const float* d_in, float scalar, int n) {
    __shared__ float s_data[BLOCK_SIZE]; // deklaracnja

    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int localIdx = threadIdx.x;

    if (globalIdx < n) {
        s_data[localIdx] = d_in[globalIdx];
    }

    __syncthreads(); // synchronizacja

    if (globalIdx < n) {
        d_out[globalIdx] = s_data[localIdx] * scalar;
    }
}

int main() {
    int n = 1000;
    size_t bytes = n * sizeof(float);
    float scalar = 5.0f;

    float* h_in = (float*)malloc(bytes);
    float* h_out = (float*)malloc(bytes);

    for (int i = 0; i < n; i++) {
        h_in[i] = (float)i;
    }

    float* d_in, * d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int gridSize = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    vectorScalarMulShared << <gridSize, BLOCK_SIZE >> > (d_out, d_in, scalar, n);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    std::cout << "Wyniki:" << std::endl;
    for (int i = 0; i <= 512; i++) {
        std::cout << "Indeks " << i << ": " << h_in[i] << " * " << scalar
            << " = " << h_out[i] << std::endl;
    }

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);


    return 0;
}
