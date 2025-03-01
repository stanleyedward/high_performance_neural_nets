#pragma once

#include <cuda_runtime.h>

void run_cross_entropy_backwards(int M, int N, float *outputs, float *labels, float *gradients);
void run_linearBW(int M, int N, int K, float *weights, float *biases, float *d_l, float *out_d_l);
void run_relu_backwards(int M, int N, float *inputs, float *gradients, float *out_gradients);
void run_update_layer(int M, int N, int batch_size, float lr,
                      float *weights, float *biases, float *activations, float *gradients);
void run_linear_backward_fused(int M, int N, int K, float *input, float *weights, float *biases, float *d_l, float *out_d_l);