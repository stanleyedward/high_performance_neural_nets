#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>

// Matrix A: MxK 
// Matrix B: KxN
// Matrix C: MxN
// C = A * B

#define M 10 //256
#define N 256 //32 batch size
#define K 256 //784
#define BLOCK_SIZE 16


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))


__global__ void mm1(float *A, float *B, float *C)
{ 
  // naive 
  // 2D blocks
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  // `if` condition is necessary for when M or N aren't multiples of 32.
  if (x < M && y < N) {
    float tmp = 0.0;
    for (int i = 0; i < K; ++i) {
      tmp += A[x * K + i] * B[i * N + y];
    }
    C[x * N + y] = tmp;
  }
}
__global__ void mm2(float *A, float *B, float *C)
{
// global memory coal.
// 1D blocks
const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

if (x < M && y < N) {
  float tmp = 0.0;
  for (int i = 0; i < K; ++i) {
    tmp += A[x * K + i] * B[i * N + y];
  }
  C[x * N + y] = tmp;
}
}

__global__ void mm3(float *A, float *B, float *C) {
  // M, N, K have to be multiples of BLOCK_SIZE
  // shared memory 
  // 2D blocks
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;

    for (int m = 0; m < N / BLOCK_SIZE; ++m) {
        sA[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }

    C[row * N + col] = sum;
}

__global__ void mm4(float *A, float *B, float *C){
  // siboehm 
  // M, N, K have to be multiples of BLOCK_SIZE
  // shared memory 
  // 1D blocks
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;
  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  A += cRow * BLOCK_SIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCK_SIZE;                        // row=0, col=cCol
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = tmp;
}

template <const int BM, const int BN, const int BK, const int TM>
__global__ void mm5(const float *A, const float *B, float *C) {

  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);
  const uint innerColA = threadIdx.x % BK; // warp-level GMEM coalescing
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; // warp-level GMEM coalescing
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {

      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = threadResults[resIdx];

  }
}


void runMM1(float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    mm1<<<blocks, threads>>>(A, B, C);
}

void runMM2(float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //1D blocks
    dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
    mm2<<<blocks, threads>>>(A, B, C);
}

void runMM3(float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    mm3<<<blocks, threads>>>(A, B, C);
}
void runMM4(float *A, float *B, float *C){

    dim3 blocks2((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //1D blocks
    dim3 threads2(BLOCK_SIZE* BLOCK_SIZE);
    mm4<<<blocks2, threads2>>>(A, B, C);
}

void runMM5(float *A, float *B, float *C){
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  mm5<BM, BN, BK, TM><<<gridDim, blockDim>>>(A, B, C);
}

void tiled_matrix_multiply_cpu(float *A, float *B, float *C){
  // Tiling for matrices:
  // A: M x K, B: K x N, C: M x N
  for (unsigned int rowTile = 0; rowTile < M / BLOCK_SIZE; ++rowTile){
    for (unsigned int colTile = 0; colTile < N / BLOCK_SIZE; ++colTile){
      for (unsigned int iTile = 0; iTile < K / BLOCK_SIZE; ++iTile){
        for (unsigned int row = rowTile * BLOCK_SIZE; row < (rowTile + 1) * BLOCK_SIZE; ++row){
          for (unsigned int col = colTile * BLOCK_SIZE; col < (colTile + 1) * BLOCK_SIZE; ++col){
            float sum = 0.0f;
            for (unsigned int i = iTile * BLOCK_SIZE; i < (iTile + 1) * BLOCK_SIZE; ++i){
              sum += A[row * K + i] * B[i * N + col];
            }
            if(iTile == 0)
              C[row * N + col] = sum;
            else
              C[row * N + col] += sum;
          }
        }
      }
    }
  }
}

void matrix_multiply_cpu(float *A, float *B, float *C, int m, int n, int k) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int l = 0; l < k; l++) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main()
{ 

  printf("Matrix Multiplication\n");
  printf("Matrix A: %d x %d\n", M, K);
  printf("Matrix B: %d x %d\n", K, N);
  printf("Matrix C: %d x %d\n", M, N);
  printf("Block size: %d x %d\n\n", BLOCK_SIZE, BLOCK_SIZE);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C1 = (float *)malloc(bytes_C);
    float *h_C2 = (float *)malloc(bytes_C);
    float *h_C_cpu = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i)
    {
      h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i)
    {
      h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    //warmup kernel
    runMM1(d_A, d_B, d_C);
    printf("Warmup kernel completed\n");

    cudaEventRecord(start);
    runMM2(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost);

    double gflops_kernel1 = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix Multiplication 1:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n\n", gflops_kernel1);
    
    
    cudaEventRecord(start);
    runMM5(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C2, d_C, bytes_C, cudaMemcpyDeviceToHost);
    double gflops_kernel2 = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix Multiplication 2:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops_kernel2);

    // Compute CPU reference
    matrix_multiply_cpu(h_A, h_B, h_C_cpu, M, N, K);
    // tiled_matrix_multiply_cpu(h_A, h_B, h_C_cpu);
    
    // Verify results
    float max_error1 = 0.0f;
    float max_error2 = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error1 = fmax(max_error1, fabs(h_C1[i] - h_C_cpu[i]));
        max_error2 = fmax(max_error2, fabs(h_C2[i] - h_C_cpu[i]));
    }
    
    printf("\nValidation Results:\n");
    printf("Kernel 1 max error: %e\n", max_error1);
    printf("Kernel 2 max error: %e\n", max_error2);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}