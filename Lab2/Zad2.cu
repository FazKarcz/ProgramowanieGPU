#include <cstdio>
#include <cuda_runtime.h>

__global__ void multiplyKernel(int a) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;

    int result = x * a;

    printf("Watek z Id %d: %d * %d = %d\n", 
        x, 
        x, 
        a, 
        result);
}

int main() {
    int a = 4;
    int numThreads = 8;  
    int numBlocks = 1;   

    multiplyKernel << <numBlocks, numThreads >> > (a);

    cudaDeviceSynchronize();

    printf("Koniec")

    return 0;
}
