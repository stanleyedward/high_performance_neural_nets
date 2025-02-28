#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include "backward_kernels.cuh"

void run_cross_entropy_backwards(int M, int N, float *outputs, float *labels, float *gradients)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    cross_entropy_backwards<<<blocks, threads>>>(M, N, outputs, labels, gradients);
}

void run_linearBW(int M, int N, int K, float *weights, float *biases, float *d_l, float *out_d_l)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    linear_backward<<<blocks, threads>>>(M, N, K, weights, biases, d_l, out_d_l);
}

void run_relu_backwards(int M, int N, float *inputs, float *gradients, float *out_gradients)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    relu_backward_kernel<<<blocks, threads>>>(M, N, inputs, gradients, out_gradients);
}

void run_update_layer(int M, int N, int batch_size, float lr,
                      float *weights, float *biases, float *activations, float *gradients)
{
    dim3 threads(16, 16, 1);
    dim3 blocks((N + threads.x - 1) / threads.x, (M + threads.y - 1) / threads.y);
    update_layer<<<blocks, threads>>>(M, N, batch_size, lr, weights, biases, activations, gradients);
}
