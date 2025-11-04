#include <iostream>
#include <cuda_runtime.h>

__global__ void kernel_task5() {
    int thread_id = threadIdx.x;
    int block_id = blockIdx.x;
    int warp_id = thread_id / warpSize;  // warp to grupa 32 wątków
    int suma = thread_id + block_id + warp_id;

    if (suma % 2 == 0)
        printf("Block %d, Thread %d, Warp %d -> Suma %d jest PARZYSTA\n", block_id, thread_id, warp_id, suma);
    else
        printf("Block %d, Thread %d, Warp %d -> Suma %d jest NIEPARZYSTA\n", block_id, thread_id, warp_id, suma);
}

int main() {
    dim3 blocks(2);
    dim3 threads(64);  // dwa warpy po 32 wątki
    kernel_task5<<<blocks, threads>>>();
    cudaDeviceSynchronize();
    return 0;
}
