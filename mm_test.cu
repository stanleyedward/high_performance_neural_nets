#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cassert>
#include "mm_runner.cuh"

// Matrix A: MxK 
// Matrix B: KxN
// Matrix C: MxN
// C = A * B

#define M 1024 //256
#define N 1024 //32
#define K 1024 //784
#define BLOCK_SIZE 16

//nvcc test.cu src/mm_runner.cu -I./src -o test.o -L/usr/local/cuda/lib64 -lcudart

#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

int main()
{ 

  printf("Matrix Multiplication\n");
  printf("Matrix A: %d x %d\n", M, K);
  printf("Matrix B: %d x %d\n", K, N);
  printf("Matrix C: %d x %d\n", M, N);
  printf("Block size: %d x %d\n\n", BLOCK_SIZE, BLOCK_SIZE);

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float *)malloc(bytes_A);
    float *h_B = (float *)malloc(bytes_B);
    float *h_C1 = (float *)malloc(bytes_C);
    float *h_C2 = (float *)malloc(bytes_C);
    float *h_C_cpu = (float*)malloc(bytes_C);

    for (int i = 0; i < M * K; ++i)
    {
      h_A[i] = rand() / (float)RAND_MAX;
    }
    for (int i = 0; i < K * N; ++i)
    {
      h_B[i] = rand() / (float)RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);


    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    //warmup kernel
    run_kernel_MM(1, BLOCK_SIZE, M, N, K, d_A, d_B, d_C);
    printf("Warmup kernel completed\n");

    cudaEventRecord(start);
    run_kernel_MM(2, BLOCK_SIZE, M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C1, d_C, bytes_C, cudaMemcpyDeviceToHost);
    double gflops_kernel1 = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix Multiplication 1:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n\n", gflops_kernel1);
    
    cudaEventRecord(start);
    run_kernel_MM(3, BLOCK_SIZE, M, N, K, d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(h_C2, d_C, bytes_C, cudaMemcpyDeviceToHost);
    double gflops_kernel2 = (2.0 * M * N * K) / (milliseconds * 1e6);
    printf("Matrix Multiplication 2:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops_kernel2);
    
    // Compute CPU reference
    run_kernel_MM(0, BLOCK_SIZE, M, N, K, h_A, h_B, h_C_cpu);
    verify_results_MM(M, N, K, h_C1, h_C2, h_C_cpu);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    free(h_C_cpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}