#pragma once
#include <cuda_runtime.h>


__global__ void mm1(uint M, uint N, uint K, float *A, float *B, float *C, float *bias)
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
    C[x * N + y] = bias[y] + tmp;
  }
}