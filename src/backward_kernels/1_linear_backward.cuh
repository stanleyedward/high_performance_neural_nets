#pragma once
#include <cuda_runtime.h>

__global__ void linear_backward(int M, int N, int K, float *weights, float *biases, float *d_l, float *out_d_l)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && column < N)
  {
    float dl = 0.f;
    for (int i = 0; i < K; i++)
    {
      float w = weights[i * N + column];
      dl += w * d_l[row * K + i];
    }
    out_d_l[row * N + column] = dl;
  }
}