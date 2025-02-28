#pragma once
#include <cuda_runtime.h>

__global__ void cross_entropy(int M, int N, float *preds, float *real, float *output)
{
    const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < M)
    {
        float loss = 0.f;
        for (int i = 0; i < N; i++)
        {
            loss -= real[idx * N + i] * log(max(1e-6, preds[idx * N + i]));
        }
        output[idx] = loss;
    }
}

void loss_fn(int M, int N, float *preds, float *real, float *output)
{
    dim3 threads = dim3(16, 1, 1);
    dim3 blocks = dim3((N + threads.x - 1) / threads.x, 1, 1);
    cross_entropy<<<blocks, threads>>>(M, N, preds, real, output);
}