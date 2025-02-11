#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Matrix A: MxK
// Matrix B: KxN
// Matrix C: MxN
// C = A * B

#define M 1024
#define N 1024
#define K 1024
#define BLOCK_SIZE 16

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
  // shared memory 
  // 1D blocks
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  // allocate buffer for current block in fast shared mem
  // shared mem is shared between all threads in a block
  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  // the inner row & col that we're accessing in this thread
  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  // advance pointers to the starting positions
  A += cRow * BLOCK_SIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCK_SIZE;                        // row=0, col=cCol
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    // Have each thread load one of the elements in A & B
    // Make the threadCol (=threadIdx.x) the consecutive index
    // to allow global memory access coalescing
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    // block threads in this block until cache is fully populated
    __syncthreads();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    // execute the dotproduct on the currently cached block
    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];
    }
    // need to sync again at the end, to avoid faster threads
    // fetching the next block into the cache before slower threads are done
    __syncthreads();
  }
  C[threadRow * N + threadCol] = tmp;
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

    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1)/ BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    //1D blocks
    // dim3 threads(BLOCK_SIZE * BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaEventRecord(start);
    mm3<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost);

    double gflops_kernel1 = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix Multiplication 1:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n\n", gflops_kernel1);
    
    dim3 blocks2((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //1D blocks
    dim3 threads2(BLOCK_SIZE* BLOCK_SIZE);

    cudaEventRecord(start);
    mm4<<<blocks2, threads2>>>(d_A, d_B, d_C);
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
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}