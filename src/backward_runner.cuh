#pragma once

#include <cuda_runtime.h>

void run_kernel_BW(int kernel_num, int M, int N, int K, int batch_size,
                   float lr, float *weights, float *biases, float *inputs, float *activations,
                   float *outputs, float *labels, float *gradients, float *out_gradients);