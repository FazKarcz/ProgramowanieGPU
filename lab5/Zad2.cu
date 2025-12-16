#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

#define BLOCK 256 // Blok

// Średnia kolumn
__global__ void mean_cols(const float* A, float* mean,
    int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++)
            sum += A[i * cols + col];
        mean[col] = sum / rows;
    }
}

// Odchylenie standardowe
__global__ void std_cols(const float* A, const float* mean,
    float* std, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < cols) {
        float sum = 0.0f;
        for (int i = 0; i < rows; i++) {
            float diff = A[i * cols + col] - mean[col];
            sum += diff * diff;
        }
        std[col] = sqrtf(sum / rows);
    }
}

//Norm
__global__ void normalize_cols(float* A, const float* mean,
    const float* std,
    int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < rows && col < cols) {
        A[row * cols + col] =
            (A[row * cols + col] - mean[col]) / std[col];
    }
}

int main() {
    int rows = 8;
    int cols = 8;
    size_t size = rows * cols * sizeof(float);

    float* h_A = (float*)malloc(size);

    for (int i = 0; i < rows * cols; i++)
        h_A[i] = (float)(i);

    float* A, * mean, * std;
    cudaMalloc(&A, size);
    cudaMalloc(&mean, cols * sizeof(float));
    cudaMalloc(&std, cols * sizeof(float));

    cudaMemcpy(A, h_A, size, cudaMemcpyHostToDevice);

    mean_cols << <1, cols >> > (A, mean, rows, cols);
    std_cols << <1, cols >> > (A, mean, std, rows, cols);

    dim3 block(16, 16);
    dim3 grid((cols + 15) / 16, (rows + 15) / 16);
    normalize_cols << <grid, block >> > (A, mean, std, rows, cols);

    cudaMemcpy(h_A, A, size, cudaMemcpyDeviceToHost);

    printf("Wynik: \n \n");
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%6.2f ", h_A[i * cols + j]);
        printf("\n");
    }

    cudaFree(A);
    cudaFree(mean);
    cudaFree(std);
    cudaFree(h_A);

    return 0;
}
