#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include "backward_kernels.cuh"

void run_cross_entropy_backwards(int M, int N, float *outputs, float *labels, float *gradients)
{
    dim3 threads(32, 32, 1);
    dim3 blocks((M + threads.y- 1) / threads.y, (N + threads.x - 1) / threads.x);
    cross_entropy_backwards<<<blocks, threads>>>(M, N, outputs, labels, gradients);
}

void run_linearBW(int M, int N, int K, float *weights, float *biases, float *d_l, float *out_d_l)
{
    dim3 threads(32, 32, 1);
    dim3 blocks((M + threads.y - 1) / threads.y, (N + threads.x - 1) / threads.x);
    linear_backward<<<blocks, threads>>>(M, N, K, weights, biases, d_l, out_d_l);
}

void run_relu_backwards(int M, int N, float *inputs, float *gradients, float *out_gradients)
{
    dim3 threads(32, 32, 1);
    dim3 blocks((M + threads.y - 1) / threads.y, (N + threads.x - 1) / threads.x);
    relu_backward_kernel<<<blocks, threads>>>(M, N, inputs, gradients, out_gradients);
}

void run_update_layer(int M, int N, int batch_size, float lr,
                      float *weights, float *biases, float *activations, float *gradients)
{
    dim3 threads(32, 32, 1);
    dim3 blocks((M + threads.y - 1) / threads.y, (N + threads.x - 1) / threads.x);
    update_layer<<<blocks, threads>>>(M, N, batch_size, lr, weights, biases, activations, gradients);
}

void run_kernel_BW(int kernel_num, int M, int N, int K, int batch_size,
                   float lr, float *weights, float *biases, float *inputs, float *activations,
                   float *outputs, float *labels, float *gradients, float *out_gradients)
{
    switch (kernel_num)
    {
    case 0:
        run_cross_entropy_backwards(M, N, outputs, labels, gradients);
        break;
    case 1:
        run_linearBW(M, N, K, weights, biases, gradients, out_gradients);
        break;
    case 2:
        run_relu_backwards(M, N, inputs, gradients, out_gradients);
        break;
    case 3:
        run_update_layer(M, N, batch_size, lr, weights, biases, activations, gradients);
        break;
    default:
        printf("dont know this one");
    }
}
