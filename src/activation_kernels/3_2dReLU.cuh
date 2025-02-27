#pragma once
#include <cuda_runtime.h>

__global__ void ReLU2d_kernel(int M, int N, float *input, float *output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        output[row * N + col] = fmaxf(0.f, input[row * N + col]);
    }
}