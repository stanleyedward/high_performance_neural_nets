#pragma once
#include <cuda_runtime.h>


void run_kernel_fused(int kernel_num, int BLOCK_SIZE, int M, int N, int K, float *A, float *B, float *C, float *bias);