#pragma once

#include <cuda_runtime.h>

void run_kernel_MM(int kernel_num, int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C, float *bias);
void verify_results_MM(uint M, uint N, uint K, float *gpu_results1, float *gpu_results2, float *cpu_results);