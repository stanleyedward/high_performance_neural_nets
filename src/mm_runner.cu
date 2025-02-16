#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include "mm_kernels.cuh"
#include <fstream>

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

void verify_results_MM(uint M, uint N, uint K, float *gpu_results1, float *gpu_results2, float *cpu_results){
    // Verify results
    float max_error1 = 0.0f;
    float max_error2 = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error1 = fmax(max_error1, fabs(gpu_results1[i] - cpu_results[i]));
        max_error2 = fmax(max_error2, fabs(gpu_results2[i] - cpu_results[i]));
    }
    printf("Max error in MM Kernel 1: %e\n", max_error1);
    printf("Max error in MM Kernel 2: %e\n", max_error2);
}

void runCPU(int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C){
matrix_multiply_cpu(M, N, K, A, B, C);
// tiled_matrix_multiply_cpu(BLOCK_SIZE, M, N, K, A, B, C);
}

void runMM1(int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    mm1<<<blocks, threads>>>(M, N, K, A, B, C);
}

void runMM2(int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //2D blocks
    dim3 threads(BLOCK_SIZE * BLOCK_SIZE);
    mm2<<<blocks, threads>>>(BLOCK_SIZE, M, N, K, A, B, C);
}

void runMM3(uint BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C){
    dim3 blocks((M + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    //1D blocks
    dim3 threads(BLOCK_SIZE* BLOCK_SIZE);
    switch(BLOCK_SIZE){
      case 16:
        mm3<16><<<blocks, threads>>>(M, N, K, A, B, C);
        break;
      case 32:
        mm3<32><<<blocks, threads>>>(M, N, K, A, B, C);
        break;
      default:
        printf("Invalid block size\n");
        break;
    }
}

void runMM4(int M, int N, int K, float *A, float *B, float *C){
  const uint BM = 64;
  const uint BN = 64;
  const uint BK = 8;
  const uint TM = 8;

  dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
  dim3 blockDim((BM * BN) / TM);
  mm4<BM, BN, BK, TM><<<gridDim, blockDim>>>(M, N, K, A, B, C);
}

void run_kernel_MM(int kernel_num, int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C) {
  switch (kernel_num) {
  case 0:
    runCPU(BLOCK_SIZE, M, N, K, A, B, C);
    break;
  case 1:
    runMM1(BLOCK_SIZE, M, N, K, A, B, C);
    break;
  case 2:
    runMM2(BLOCK_SIZE, M, N, K, A, B, C);
    break;
  case 3:
    runMM3(BLOCK_SIZE, M, N, K, A, B, C);
    break;
  case 4:
    runMM4(M, N, K, A, B, C);
    break;
  default:
    throw std::invalid_argument("Unknown kernel number");
  }
}