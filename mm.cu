#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define BLOCK_SIZE 32

__global__ void matrix_multiply_naive(float *A, float *B, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

__global__ void matrix_multiply_shared(float *A, float *B, float *C) {
    __shared__ float sA[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float sB[BLOCK_SIZE][BLOCK_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;
    float sum = 0.0f;
    
    for (int m = 0; m < N/BLOCK_SIZE; ++m) {
        sA[ty][tx] = A[row * N + (m * BLOCK_SIZE + tx)];
        sB[ty][tx] = B[(m * BLOCK_SIZE + ty) * N + col];
        __syncthreads();
        
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += sA[ty][k] * sB[k][tx];
        }
        __syncthreads();
    }
    
    C[row * N + col] = sum;
}

int main() {
    size_t bytes = N * N * sizeof(float);
    
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C1 = (float*)malloc(bytes);
    float *h_C2 = (float*)malloc(bytes);
    
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    dim3 blocks(N/BLOCK_SIZE, N/BLOCK_SIZE);
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds;
    
    cudaEventRecord(start);
    matrix_multiply_naive<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    //each element needs 2*N ops (N multiplications and N additions)
    double gflops_naive = (2.0 * N * N * N) / (milliseconds * 1e6);
    printf("Naive Matrix Multiplication:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n\n", gflops_naive);
    
    cudaEventRecord(start);
    matrix_multiply_shared<<<blocks, threads>>>(d_A, d_B, d_C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    double gflops_shared = (2.0 * N * N * N) / (milliseconds * 1e6);
    printf("Shared Memory Matrix Multiplication:\n");
    printf("Time: %.4f ms\n", milliseconds);
    printf("Performance: %.2f GFLOPS\n", gflops_shared);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}