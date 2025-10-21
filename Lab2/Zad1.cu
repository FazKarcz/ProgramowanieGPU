#include <stdio.h>
#include <cuda_runtime.h>

__global__ void kernelwithId() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; // Wzór z internetu

    printf("Watek: globalny_id=%d, blockIdx=%d, threadIdx=%d\n",
        threadId, 
        blockIdx.x, 
        threadIdx.x);
}

int main() {
    int threadsPerBlock = 8;
    int numBlocks = 4;

    printf("Lacznie %d watkow\n\n", numBlocks * threadsPerBlock);

    kernelwithId << <numBlocks, threadsPerBlock >> > ();

    cudaDeviceSynchronize();

    printf("\nKoniec");

    return 0;
}