#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda::wmma;

#define CHECK(call)                                                 
{                                                                   
    cudaError_t err = call;                                         
    if (err != cudaSuccess) {                                       
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                
            __FILE__, __LINE__, cudaGetErrorString(err));           
        exit(EXIT_FAILURE);                                         
    }                                                               
}

// Kernel 1: Naiwny - global memory
__global__ void hadamard_global(const half* A, const half* B, half* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = __hmul(A[idx], B[idx]);
    }
}

// Kernel 2: Shared memory tile copy, then multiply
template<int TILE>
__global__ void hadamard_shared(const half* A, const half* B, half* C, int N) {
    __shared__ half sA[TILE][TILE];
    __shared__ half sB[TILE][TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    if (row < N && col < N) {
        sA[ty][tx] = A[row * N + col];
        sB[ty][tx] = B[row * N + col];
    }
    else {
        sA[ty][tx] = __float2half(0.0f);
        sB[ty][tx] = __float2half(0.0f);
    }
    __syncthreads();

    if (row < N && col < N) {
        C[row * N + col] = __hmul(sA[ty][tx], sB[ty][tx]);
    }
}

// WMMA
__global__ void hadamard_wmma(const half* A, const half* B, half* C, int N) {
    const int TILE = 16;
    int tile_i = blockIdx.y;
    int tile_j = blockIdx.x;

    int row = tile_i * TILE;
    int col = tile_j * TILE;

    const half* ptrA = A + row * N + col;
    const half* ptrB = B + row * N + col;
    half* ptrC = C + row * N + col;

    fragment<matrix_a, 16, 16, 16, half, row_major> fragA;
    fragment<matrix_b, 16, 16, 16, half, row_major> fragB;
    fragment<accumulator, 16, 16, 16, half> fragC; 

    // 16x16
    load_matrix_sync(fragA, ptrA, N);
    load_matrix_sync(fragB, ptrB, N);

    for (int e = 0; e < fragA.num_elements; ++e) {
        fragC.x[e] = __hmul(fragA.x[e], fragB.x[e]);
    }

    store_matrix_sync(ptrC, fragC, N, mem_row_major);
}

void hadamard_cpu(const half* A, const half* B, half* C, int N) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            int idx = i * N + j;
            float a = __half2float(A[idx]);
            float b = __half2float(B[idx]);
            C[idx] = __float2half(a * b);
        }
    }
}


// Main
int main() {
    const int N = 256;
    const size_t bytes = N * N * sizeof(half);

    half* hA, * hB, * hCref;
    hA = (half*)malloc(bytes);
    hB = (half*)malloc(bytes);
    hCref = (half*)malloc(bytes);

    for (int i = 0; i < N * N; i++) {
        float v1 = (float)(rand() % 100) / 10.0f;
        float v2 = (float)(rand() % 100) / 20.0f;
        hA[i] = __float2half(v1);
        hB[i] = __float2half(v2);
    }

    half* dA, * dB, * dC;
    CHECK(cudaMalloc(&dA, bytes));
    CHECK(cudaMalloc(&dB, bytes));
    CHECK(cudaMalloc(&dC, bytes));

    CHECK(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    // 1) Naive global
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    hadamard_global << <grid, block >> > (dA, dB, dC, N);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(hCref, dC, bytes, cudaMemcpyDeviceToHost));

    half* hCcpu = (half*)malloc(bytes);
    hadamard_cpu(hA, hB, hCcpu, N);

    // check
    bool ok = true;
    for (int i = 0; i < N * N; i++) {
        float r = __half2float(hCref[i]);
        float g = __half2float(hCcpu[i]);
        if (fabs(r - g) > 1e-2f) { ok = false; break; }
    }
    printf("Naive global matches CPU: %s\n", ok ? "YES" : "NO");

    // 2) Shared memory
    const int TILE = 16;
    dim3 blockS(TILE, TILE);
    dim3 gridS(N / TILE, N / TILE);
    hadamard_shared<TILE> << <gridS, blockS >> > (dA, dB, dC, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hCref, dC, bytes, cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < N * N; i++) {
        float r = __half2float(hCref[i]);
        float g = __half2float(hCcpu[i]);
        if (fabs(r - g) > 1e-2f) { ok = false; break; }
    }
    printf("Shared memory matches CPU: %s\n", ok ? "YES" : "NO");

    // 3) WMMA
    hadamard_wmma << <gridS, 1 >> > (dA, dB, dC, N);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaMemcpy(hCref, dC, bytes, cudaMemcpyDeviceToHost));
    ok = true;
    for (int i = 0; i < N * N; i++) {
        float r = __half2float(hCref[i]);
        float g = __half2float(hCcpu[i]);
        if (fabs(r - g) > 1e-2f) { ok = false; break; }
    }
    printf("WMMA-fragment approach matches CPU: %s\n", ok ? "YES" : "NO");

    // cleanup
    free(hA); free(hB); free(hCref); free(hCcpu);
    CHECK(cudaFree(dA)); CHECK(cudaFree(dB)); CHECK(cudaFree(dC));

    return 0;
}
