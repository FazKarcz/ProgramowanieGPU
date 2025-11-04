#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel_task4() {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int value = 100 + thread_id + block_id;
    printf("Block %d, Thread %d -> wynik: %d\n", block_id, thread_id, value);
}

int main() {
    dim3 blocks(2);
    dim3 threads(5);
    kernel_task4<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
