#include <cstdio>
#include <cstdlib>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>          // WMMA
using namespace nvcuda;   // for wmma

// -------------------- Utility helpers --------------------
#define CHECK(call)                                                    \
do {                                                                    \
    cudaError_t err = call;                                             \
    if (err != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,   \
                cudaGetErrorString(err));                               \
        exit(1);                                                        \
    }                                                                   \
} while(0)

void fill_random_float(float *x, int N, float scale=1.0f) {
    for (int i=0;i<N;i++) x[i] = (rand() / (float)RAND_MAX) * scale;
}

void print_sample(const float *C, int M, int N) {
    for (int r=0;r<min(4,M);++r) {
        for (int c=0;c<min(8,N);++c) printf("%8.4f ", C[r*N + c]);
        printf("\n");
    }
}

// CPU reference (float)
void hadamard_cpu(const float *A, const float *B, float *C, int M, int N) {
    int S = M*N;
    for (int i=0;i<S;i++) C[i] = A[i]*B[i];
}

// -------------------- 1) Naive global memory kernel --------------------
__global__ void hadamard_global(const float* A, const float* B, float* C, int M, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int size = M * N;
    if (idx < size) {
        C[idx] = A[idx] * B[idx];
    }
}

// -------------------- 2) Tiled using shared memory --------------------
// We'll tile in 2D tiles of tileDim x tileDim
template<int TILE_DIM>
__global__ void hadamard_shared(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N) {
    // 2D block/grid mapping
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * TILE_DIM + ty;
    int col = bx * TILE_DIM + tx;

    __shared__ float sA[TILE_DIM][TILE_DIM];
    __shared__ float sB[TILE_DIM][TILE_DIM];

    // Load into shared if inside matrix
    if (row < M && col < N) {
        int idx = row * N + col;
        sA[ty][tx] = A[idx];
        sB[ty][tx] = B[idx];
    } else {
        sA[ty][tx] = 0.0f;
        sB[ty][tx] = 0.0f;
    }
    __syncthreads();

    // compute and store
    if (row < M && col < N) {
        int idx = row * N + col;
        C[idx] = sA[ty][tx] * sB[ty][tx];
    }
}

// -------------------- 3) WMMA fragments (half) --------------------
// We'll assume M and N are multiples of 16 (tile size). We load 16x16 tiles as fragments.
// Use wmma::fragment and do elementwise multiply inside fragment storage.
__global__ void hadamard_wmma(const half* A, const half* B, half* C, int M, int N) {
    // tile 16x16
    const int TILE_M = 16;
    const int TILE_N = 16;

    int tile_row = blockIdx.y;
    int tile_col = blockIdx.x;

    int global_row = tile_row * TILE_M;
    int global_col = tile_col * TILE_N;

    // fragment types for load/store. Use matrix_a and matrix_b fragments to load tiles.
    wmma::fragment<wmma::matrix_a, TILE_M, TILE_N, TILE_M, half, wmma::row_major> aFrag;
    wmma::fragment<wmma::matrix_b, TILE_M, TILE_N, TILE_M, half, wmma::row_major> bFrag;
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_M, half> cFrag; // we'll store half

    // load fragments from global memory; stride = N (row-major)
    const half* A_tile_ptr = A + global_row * N + global_col;
    const half* B_tile_ptr = B + global_row * N + global_col;

    // wmma load expects contiguous TILE_M x TILE_N block with leading dimension = N
    // Use load_matrix_sync
    wmma::load_matrix_sync(aFrag, A_tile_ptr, N);
    wmma::load_matrix_sync(bFrag, B_tile_ptr, N);

    // Elementwise multiply inside fragment
    // fragment::x is accessible (it's an array). Number of elements:
    const int nelemsA = aFrag.num_elements;
    // However aFrag is of type half elements for matrix_a; we will copy to cFrag elements.
    for (int i=0;i<nelemsA;i++) {
        // aFrag.x and bFrag.x are of type half
        // cFrag.x also exists; assign product
        // Note: accumulator fragment in many toolchains uses float for accumulation.
        // Here we declared accumulator as half -> some toolchains may not allow half accumulator;
        // if not supported, change accumulator to float and then convert on store.
        // To be maximally portable we'll use float accumulator below:
    }

    // To be portable, do this with float accumulator:
    wmma::fragment<wmma::accumulator, TILE_M, TILE_N, TILE_M, float> accFrag;
    for (int i=0;i<aFrag.num_elements;i++) {
        float aval = __half2float(aFrag.x[i]);
        float bval = __half2float(bFrag.x[i]);
        accFrag.x[i] = aval * bval;
    }

    // store back (store_matrix_sync expects half if C pointer half and fragment type matches);
    // wmma::store_matrix_sync works with accumulator fragment type float -> but requires an appropriate output layout.
    // We'll convert and write manually to global memory for simplicity:
    // write accFrag elements into C global memory (row-major)
    // accFrag layout corresponds to TILE_M x TILE_N in row-major with leading dimension = TILE_M in WMMA type param,
    // but it's easier to iterate rows/cols and write to C
    for (int r=0; r<TILE_M; ++r) {
        int row_idx = global_row + r;
        int base = row_idx * N + global_col;
        for (int c=0; c<TILE_N; ++c) {
            int fragIndex = r * TILE_N + c; // mapping for row-major fragment
            // convert to half and store
            half out = __float2half(accFrag.x[fragIndex]);
            C[base + c] = out;
        }
    }
}

// -------------------- Host helpers for half conversions & verification --------------------
void float_to_half_array(const float* in, half* out, int S) {
    for (int i=0;i<S;i++) out[i] = __float2half(in[i]);
}
void half_to_float_array(const half* in, float* out, int S) {
    for (int i=0;i<S;i++) out[i] = __half2float(in[i]);
}

// -------------------- Main: run and compare --------------------
int main(int argc, char** argv) {
    srand(123);
    // matrix dims
    int M = 512; // rows
    int N = 512; // cols
    if (argc >= 3) { M = atoi(argv[1]); N = atoi(argv[2]); }
    int S = M * N;
    printf("M=%d N=%d\n", M, N);

    // Allocate host
    float *h_A = (float*)malloc(S * sizeof(float));
    float *h_B = (float*)malloc(S * sizeof(float));
    float *h_C = (float*)malloc(S * sizeof(float));
    float *h_ref = (float*)malloc(S * sizeof(float));

    fill_random_float(h_A, S, 2.0f);
    fill_random_float(h_B, S, 2.0f);
    hadamard_cpu(h_A, h_B, h_ref, M, N);

    // Device allocations
    float *d_Af, *d_Bf, *d_Cf;
    CHECK(cudaMalloc(&d_Af, S*sizeof(float)));
    CHECK(cudaMalloc(&d_Bf, S*sizeof(float)));
    CHECK(cudaMalloc(&d_Cf, S*sizeof(float)));
    CHECK(cudaMemcpy(d_Af, h_A, S*sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_Bf, h_B, S*sizeof(float), cudaMemcpyHostToDevice));

    // 1) global kernel
    {
        int threads = 256;
        int blocks = (S + threads - 1) / threads;
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        hadamard_global<<<blocks, threads>>>(d_Af, d_Bf, d_Cf, M, N);
        CHECK(cudaGetLastError());
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("[global] time: %f ms\n", ms);
        // copy back and verify
        CHECK(cudaMemcpy(h_C, d_Cf, S*sizeof(float), cudaMemcpyDeviceToHost));
        // verify
        double maxerr=0.0;
        for (int i=0;i<S;i++) maxerr = fmax(maxerr, fabs(h_C[i]-h_ref[i]));
        printf("[global] max err = %e\n", maxerr);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }

    // 2) shared memory kernel
    {
        const int TILE = 16; // try 16 or 32
        dim3 block(TILE, TILE);
        dim3 grid((N + TILE - 1)/TILE, (M + TILE - 1)/TILE);
        cudaEvent_t start, stop;
        cudaEventCreate(&start); cudaEventCreate(&stop);
        cudaEventRecord(start);
        hadamard_shared<TILE><<<grid, block>>>(d_Af, d_Bf, d_Cf, M, N);
        CHECK(cudaGetLastError());
        cudaEventRecord(stop); cudaEventSynchronize(stop);
        float ms; cudaEventElapsedTime(&ms, start, stop);
        printf("[shared] time: %f ms (TILE=%d)\n", ms, TILE);
        CHECK(cudaMemcpy(h_C, d_Cf, S*sizeof(float), cudaMemcpyDeviceToHost));
        double maxerr=0.0;
        for (int i=0;i<S;i++) maxerr = fmax(maxerr, fabs(h_C[i]-h_ref[i]));
        printf("[shared] max err = %e\n", maxerr);
        cudaEventDestroy(start); cudaEventDestroy(stop);
    }
