#pragma once
#include <cuda_runtime.h>

__global__ void mm2(uint BLOCK_SIZE, uint M, uint N, uint K, float *A, float *B, float *C, float *bias)
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
  C[x * N + y] =  bias[y] + tmp;
}
}