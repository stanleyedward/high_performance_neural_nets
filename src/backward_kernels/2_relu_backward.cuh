#pragma once
#include <cuda_runtime.h>

__global__ void relu_backward_kernel(int M, int N, float *input, float *d_l, float *output)
{
    const uint row = blockIdx.y * blockDim.y + threadIdx.y;
    const uint col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        output[row * N + col] = input[row * N + col] > 0.f ? d_l[row * N + col] : 0.f;
    }
}