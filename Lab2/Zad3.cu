#include <cstdio>
#include <cuda_runtime.h>

__global__ void kernelBy3() {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    if (idx % 3 == 0) {
        printf("Watek %d: indeks %d jest podzielny przez 3\n", 
            idx,
            idx);
    }
    else
    {
        printf("Watek %d: indeks %d nie jest podzielny przez 3\n",
            idx,
            idx);
    }
}

int main() {

    kernelBy3 << <4,8 >> > ();

    cudaDeviceSynchronize();

    printf("\nKoniec\n");

    return 0;
}
