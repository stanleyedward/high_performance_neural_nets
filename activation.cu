#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>
#include "activation_runner.cuh"

#define M 1024  // Number of rows /OUTPUT
#define N 1024  // Number of columns /BATCH_SIZE

int main()
{
    int size = M * N;
    size_t bytes = size * sizeof(float);

    printf("Activation Function\n");
    printf("Matrix: %d x %d\n\n", M, N);

    float *h_input = (float *)malloc(bytes);
    float *h_act1 = (float *)malloc(bytes);
    float *h_act2 = (float *)malloc(bytes);
    float *h_cpu_out = (float *)malloc(bytes);

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
    // run_kernel_softmax(1, M, N, d_input, d_act1);
    run_kernel_relu(1, M, N, d_input, d_act1);
    
    cudaEventRecord(start);
    run_kernel_softmax(1, M, N, d_input, d_act1);
    // run_kernel_relu(1, M, N, d_input, d_act1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("act1 execution time: %.4f ms\n", milliseconds);
    float gflops_act1 = (8.0f * M * N) / (milliseconds * 1e6);
    printf("act1 GFLOPS: %.4f\n", gflops_act1);
    printf("\n");

    cudaEventRecord(start);
    run_kernel_softmax(2, M, N, d_input, d_act2);
    // run_kernel_relu(2, M, N, d_input, d_act2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("act2 execution time: %.4f ms\n", milliseconds);

    cudaGetLastError();
    float gflops_act2 = (7.0f * M * N) / (milliseconds * 1e6);
    printf("act2 GFLOPS: %.4f\n", gflops_act2);
    cudaMemcpy(h_act1, d_act1, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_act2, d_act2, bytes, cudaMemcpyDeviceToHost);
    printf("\n");

    run_kernel_softmax(0, M, N, h_input, h_cpu_out);
    // run_kernel_relu(0, M, N, h_input, h_cpu_out);
    verify_results_activation(M, N, h_act1, h_act2, h_cpu_out);

    cudaFree(d_input);
    cudaFree(d_act1);
    cudaFree(d_act2);
    free(h_input);
    free(h_act1);
    free(h_act2);
    free(h_cpu_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
