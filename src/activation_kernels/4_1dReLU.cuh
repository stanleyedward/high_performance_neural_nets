#pragma once
#include <cuda_runtime.h>

__global__ void ReLU1d_kernel(int M, int N, float *input, float *output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M*N)
    {
        output[idx] = fmaxf(0.f, input[idx]);
    }
}