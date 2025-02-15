#pragma once

void verify_results_activation(uint M, uint N, float *gpu_results1, float *gpu_results2, float *cpu_results);
void run_kernel_softmax(int kernel_num, int M, int N, float *d_input, float *d_output);