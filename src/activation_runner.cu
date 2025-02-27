#include "activation_kernels.cuh"
#include <cuda_runtime.h>
#include <stdio.h>

void run_softmaxCPU(int M, int N, float *input, float *output) {
    softmax_cpu(M, N, input, output);
}

void run_softmax1(int M, int N, float *d_input, float *d_output) {
    dim3 numThreadsPerBlock = dim3(32, 32, 1);
    dim3 numBlocks = dim3((N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, (M + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y, 1);

    softmax<<<numBlocks, numThreadsPerBlock>>>(M, N, d_input, d_output);

}

void run_softmax2(int M, int N, float *d_input, float *d_output) {
    const int BLOCK_DIM_Y = 16;
    dim3 block_size = dim3(1, BLOCK_DIM_Y, 1);
    dim3 grid_size = dim3(M, 1, 1);  
    smem_coal_unrolled_softmax<BLOCK_DIM_Y><<<grid_size, block_size>>>(M, N, d_input, d_output);

}

void run_reluCPU(int M, int N, float *input, float *output) {
    relu_cpu(M, N, input, output);
}

void run_relu1(int M, int N, float *d_input, float *d_output) {
    dim3 numThreadsPerBlock = dim3(32, 32, 1);
    dim3 numBlocks = dim3((N + numThreadsPerBlock.y - 1) / numThreadsPerBlock.y, (M + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 1);
    ReLU2d_kernel<<<numBlocks, numThreadsPerBlock>>>(M, N, d_input, d_output);
}

void run_relu2(int M, int N, float *d_input, float *d_output) {
    dim3 numThreadsPerBlock = dim3(32, 1, 1);
    dim3 numBlocks = dim3((M * N + numThreadsPerBlock.x - 1) / numThreadsPerBlock.x, 1, 1);
    ReLU1d_kernel<<<numBlocks, numThreadsPerBlock>>>(M, N, d_input, d_output);
}

void verify_results_activation(uint M, uint N, float *gpu_results1, float *gpu_results2, float *cpu_results){
    // Verify results
    float max_error1 = 0.0f;
    float max_error2 = 0.0f;
    for (int i = 0; i < M * N; i++) {
        max_error1 = fmax(max_error1, fabs(gpu_results1[i] - cpu_results[i]));
        max_error2 = fmax(max_error2, fabs(gpu_results2[i] - cpu_results[i]));
    }

    printf("Max error in Activation Kernel 1: %e\n", max_error1);
    printf("Max error in Activation Kernel 2: %e\n", max_error2);
}

void run_kernel_softmax(int kernel_num, int M, int N, float *d_input, float *d_output) {
    switch (kernel_num) {
        case 0:
            run_softmaxCPU(M, N, d_input, d_output);
            break;
        case 1:
            run_softmax1(M, N, d_input, d_output);
            break;
        case 2:
            run_softmax2(M, N, d_input, d_output);
            break;
        default:
            printf("Invalid kernel number\n");
            break;
    }
}

void run_kernel_relu(int kernel_num, int M, int N, float *d_input, float *d_output) {
    switch (kernel_num) {
        case 0:
            relu_cpu(M, N, d_input, d_output);
            break;
        case 1:
        run_relu1(M, N, d_input, d_output);
            break;
        case 2:
        run_relu2(M, N, d_input, d_output);
            break;
        default:
            printf("Invalid kernel number\n");
            break;
    }
}