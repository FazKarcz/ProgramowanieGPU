#include <iostream>
#include <cuda_runtime.h>

// Zad 1 --Profilowanie Kodu--

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}


int main() {
    int N = 1024;                     
    size_t size = N * sizeof(float);

    // Alokacja host
    float* hA = (float*)malloc(size);
    float* hB = (float*)malloc(size);
    float* hC = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        hA[i] = i;
        hB[i] = 2 * i;
    }

    float* dA, * dB, * dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    dim3 block(1);
    dim3 threads(N);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    vectorAdd << <block, threads >> > (dA, dB, dC, N);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << "Czas wykonania kernel = " << ms * 1000 << " us" << std::endl;

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
