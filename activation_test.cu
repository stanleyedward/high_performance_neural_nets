#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "activation_runner.cuh"

#define M 1024  // Number of rows
#define N 1024  // Number of columns
#define THREADS_PER_BLOCK 1024

#ifndef UNROLL_FACTOR
#define UNROLL_FACTOR 8
#endif
constexpr int URF{UNROLL_FACTOR};

#define BLOCK_DIM_Y 1024  // adjust as needed

__global__ void act1(float *input, float *output)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = threadIdx.y;
    __shared__ float reduction[BLOCK_DIM_Y];

    if (row < M)
    {
        // Compute max value for the row
        float maxval = 0;
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            maxval = fmaxf(maxval, input[row * N + i]);
        }
        reduction[ty] = maxval;
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
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            divisor += __expf(input[row * N + i] - maxval);
        }
        reduction[ty] = divisor;
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
        for (int i = ty; i < N; i += BLOCK_DIM_Y)
        {
            output[row * N + i] = __expf(input[row * N + i] - maxval) / divisor;
        }
    }
}


__global__ void act2(float *input, float *output)
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
__global__ void act3(float *input, float *output)
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

void softmax_cpu(float *input, float *output, int rows, int cols) {
    for (int row = 0; row < rows; row++) {
        float *row_input = input + row * cols;
        float *row_output = output + row * cols;

        // Find the maximum value in the current row to avoid numerical instability
        float maxval = row_input[0];
        for (int i = 1; i < cols; i++) {
            if (row_input[i] > maxval) {
                maxval = row_input[i];
            }
        }

        // Compute the sum of exponentials
        float sum_exp = 0.0f;
        for (int i = 0; i < cols; i++) {
            float exp_val = expf(row_input[i] - maxval);
            row_output[i] = exp_val; // Temporarily store exp values
            sum_exp += exp_val;
        }

        // Normalize by the sum of exponentials
        for (int i = 0; i < cols; i++) {
            row_output[i] /= sum_exp;
        }
    }
}
int main()
{
    int size = M * N;
    size_t bytes = size * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    float *h_act1 = (float *)malloc(bytes);
    float *h_act2 = (float *)malloc(bytes);
    float *h_cpu_softmax = (float *)malloc(bytes);

    // init input with random values between -10 to 10
    for (int i = 0; i < size; i++)
    {
        h_input[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
    }

    float *d_input, *d_act1, *d_act2;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_act1, bytes);
    cudaMalloc(&d_act2, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    //warmup
    run_kernel_softmax(1, M, N, d_input, d_act1);
    
    cudaEventRecord(start);
    run_kernel_softmax(1, M, N, d_input, d_act1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("act1 execution time: %.4f ms\n", milliseconds);
    float gflops_act1 = (7.0f * M * N) / (milliseconds * 1e6);
    printf("act1 GFLOPS: %.4f\n", gflops_act1);

    cudaEventRecord(start);
    run_kernel_softmax(2, M, N, d_input, d_act2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("act2 execution time: %.4f ms\n", milliseconds);

    cudaGetLastError();
    float gflops_act2 = (7.0f * M * N) / (milliseconds * 1e6);
    printf("act2 GFLOPS: %.4f\n", gflops_act2);
    cudaMemcpy(h_act1, d_act1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_act2, d_act2, bytes, cudaMemcpyDeviceToHost);

    run_kernel_softmax(0, M, N, h_input, h_cpu_softmax);
    verify_results_activation(M, N, h_act1, h_act2, h_cpu_softmax);

    cudaFree(d_input);
    cudaFree(d_act1);
    cudaFree(d_act2);
    free(h_input);
    free(h_act1);
    free(h_act2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
