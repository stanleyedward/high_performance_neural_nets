#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 1024

__global__ void act1(float *input, float *output)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N)
    {
        output[row * N + col] = 1.0f / (1.0f + expf(-input[row * N + col]));
    }
}

__global__ void act2(float *input, float *output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

int main()
{
    int size = N * N;
    size_t bytes = size * sizeof(float);

    float *h_input = (float *)malloc(bytes);
    float *h_sigmoid = (float *)malloc(bytes);
    float *h_relu = (float *)malloc(bytes);

    // init input with random values between -10 to 10
    for (int i = 0; i < size; i++)
    {
        h_input[i] = (float)rand() / RAND_MAX * 20.0f - 10.0f;
    }

    float *d_input, *d_sigmoid, *d_relu;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_sigmoid, bytes);
    cudaMalloc(&d_relu, bytes);

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    dim3 numThreadsPerBlock = dim3(32, 32, 1);
    dim3 numBlocks = dim3((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;

    cudaEventRecord(start);
    act1<<<numBlocks, numThreadsPerBlock>>>(d_input, d_sigmoid);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Sigmoid execution time: %.4f ms\n", milliseconds);

    // Check for kernel errors

    int blockSize = THREADS_PER_BLOCK;
    int gridSize = (size + blockSize - 1) / blockSize;

    cudaEventRecord(start);
    act2<<<gridSize, blockSize>>>(d_input, d_relu, size);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("ReLU execution time: %.4f ms\n", milliseconds);

    cudaGetLastError();

    // cudaMemcpy(h_sigmoid, d_sigmoid, bytes, cudaMemcpyDeviceToHost);
    // cudaMemcpy(h_relu, d_relu, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_sigmoid);
    cudaFree(d_relu);
    free(h_input);
    free(h_sigmoid);
    free(h_relu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}