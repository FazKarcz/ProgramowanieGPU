#include <stdio.h>

__global__ void vectorAddShared(const float* A, const float* B, float* C, int n) {
    extern __shared__ float shared[];

    float* sA = shared;                      
    float* sB = shared + blockDim.x;          

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        sA[threadIdx.x] = A[idx];
        sB[threadIdx.x] = B[idx];

        __syncthreads();  


        float result = sA[threadIdx.x] + sB[threadIdx.x];

        C[idx] = result;
    }
}

int main() {
    const int n = 1024;
    size_t size = n * sizeof(float);


    float *hA, *hB, *hC;
    hA = (float*)malloc(size);
    hB = (float*)malloc(size);
    hC = (float*)malloc(size);


    for (int i = 0; i < n; i++) {
        hA[i] = i;
        hB[i] = 2 * i;
    }


    float *dA, *dB, *dC;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);
    cudaMalloc(&dC, size);

    cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    vectorAddShared<<<gridSize, blockSize, 2 * blockSize * sizeof(float)>>>(dA, dB, dC, n);

    cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("%f + %f = %f\n", hA[i], hB[i], hC[i]);
    }

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
