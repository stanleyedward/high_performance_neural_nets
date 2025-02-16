#pragma once
#include <cuda_runtime.h>


__global__ void softmax(int M, int N, float *input, float *output)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < M && col < N)
  {
    float maxval = input[row*N];

    for (int i = 1; i<N; i++)
    {
      maxval = fmaxf(maxval, input[row*N + i]);
    }
    float divisor = 0.f;

    for (int i = 0; i<N; i++)
    {
      divisor += __expf(input[row*N + i] - maxval);
    }
    output[row*N + col] = __expf(input[row*N + col]-maxval)/(divisor);
  }
}