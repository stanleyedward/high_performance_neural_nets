#pragma once
#include <cuda_runtime.h>

__global__ void cross_entropy_backwards(int M, int N, float *preds, float *real, float *output)
{
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && col < N)
  {
    output[row * N + col] = preds[row * N + col] - real[row * N + col];
  }
}