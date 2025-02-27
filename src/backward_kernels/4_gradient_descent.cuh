#pragma once
#include <cuda_runtime.h>

__global__ void update_layer(int M, int N, int batch_size, float lr, float *weights, float *biases, float *activations, float *d_l)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < M && column < N)
  {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < batch_size; i++)
    {
      float act = activations[i * M + row];
      float dl = d_l[i * N + column];
      dw += act * dl;
      db += dl;
    }
    weights[row * N + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
  }
}