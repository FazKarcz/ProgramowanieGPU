#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel_task7(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = (idx % 2 == 0) ? (A[idx] + B[idx]) : (A[idx] - B[idx]);
}

int main() {
    const int N = 8;
    int h_A[N] = {1,2,3,4,5,6,7,8};
    int h_B[N] = {8,7,6,5,4,3,2,1};
    int h_C[N];

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * sizeof(int));
    cudaMalloc(&d_B, N * sizeof(int));
    cudaMalloc(&d_C, N * sizeof(int));

    cudaMemcpy(d_A, h_A, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, N * sizeof(int), cudaMemcpyHostToDevice);

    kernel_task7<<<1, N>>>(d_A, d_B, d_C, N);

    cudaMemcpy(h_C, d_C, N * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++)
        std::cout << "C[" << i << "] = " << h_C[i] << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    return 0;
}
