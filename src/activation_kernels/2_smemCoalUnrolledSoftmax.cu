#pragma once
#include <cuda_runtime.h>

__global__ void smem_coal_unrolled_softmax(int BLOCK_DIM_Y, int M, int N, float *input, float *output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float reduction[BLOCK_DIM_Y];

    if (row < M)
    {
        // Compute max value for the row
        float maxval = 0;
        #pragma unroll URF
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            maxval = fmaxf(maxval, input[row * N + i]);
        }
        reduction[ty] = maxval;
        #pragma unroll URF
        for (int stride = BLOCK_DIM_Y / 2; stride >= 1; stride /= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction[ty] = fmaxf(reduction[ty], reduction[ty + stride]);
            }
        }
        __syncthreads();
        maxval = reduction[0];

        // Compute sum of exponentials for the row
        float divisor = 0.f;
        #pragma unroll URF
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            divisor += __expf(input[row * N + i] - maxval);
        }
        reduction[ty] = divisor;
        #pragma unroll URF
        for (int stride = BLOCK_DIM_Y / 2; stride >= 1; stride /= 2)
        {
            __syncthreads();
            if (ty < stride)
            {
                reduction[ty] += reduction[ty + stride];
            }
        }
        __syncthreads();
        divisor = reduction[0];

        // Compute final softmax outputs for the row
        #pragma unroll URF
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            output[row * N + i] = __expf(input[row * N + i] - maxval) / divisor;
        }
    }
}
