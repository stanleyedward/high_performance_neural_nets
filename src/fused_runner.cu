#include "fused_kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>


#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void runFused1(int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C, float *bias){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
    linear_relu_gmem<<<blocks, threads>>>(BLOCK_SIZE, M, N, K, A, B, C, bias);
}

void runFused2(int M, int N, int K, float *A, float *B, float *C, float *bias){
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  linear_relu_1dBlocking<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C, bias);
}


void run_kernel_fused(int kernel_num, int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C, float *bias) {
  switch (kernel_num) {
  case 1:
    runFused1(BLOCK_SIZE, M, N, K, A, B, C, bias);
    break;
  case 2:
    runFused2(M, N, K, A, B, C, bias);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}
