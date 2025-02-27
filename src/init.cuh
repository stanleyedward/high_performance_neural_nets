#pragma once

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <string>
#include "helper_functions.cuh"

typedef struct
{
  float *weights1;
  float *weights2;
  float *weights3;
  float *biases1;
  float *biases2;
  float *biases3;
  float *grad_layer1;
  float *grad_layer2;
  float *grad_layer3;
} NeuralNetwork;

__global__ void init_rand(int w, int h, float *mat)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    curandState state;
    curand_init(42, row * w + column, 0, &state);
    mat[row * w + column] = curand_normal(&state) * sqrtf(2.f / h);
  }
}

template <const int BLOCK_SIZE>
void initLayer(float *weights, float *biases, int w, int h)
{
  dim3 numBlocks = dim3((w + BLOCK_SIZE - 1) / BLOCK_SIZE, (h + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  dim3 numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_rand<<<numBlocks, numThreadsPerBlock>>>(w, h, weights);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((h + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
  init_rand<<<numBlocks, numThreadsPerBlock>>>(1, h, biases);
  CUDA_CHECK(cudaPeekAtLastError());
}

template <const int INPUT_SIZE, const int HIDDEN_LAYER_1, const int HIDDEN_LAYER_2, const int OUTPUT_LAYER, const int BATCH_SIZE, const int BLOCK_SIZE, const int LABELS_SIZE>
void init_outputs(float **input, float **labels, float **x1, float **a1, float **x2, float **a2, float **x3, float **a3, float **loss)
{
  CUDA_CHECK(cudaMalloc((void **)input, INPUT_SIZE * BATCH_SIZE * sizeof(float))); // K*N
  CUDA_CHECK(cudaMalloc((void **)labels, LABELS_SIZE * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float))); // M*N
  CUDA_CHECK(cudaMalloc((void **)a1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)a2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float))); // softmax input
  CUDA_CHECK(cudaMalloc((void **)a3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float))); // softmax output

  CUDA_CHECK(cudaMalloc((void **)loss, BATCH_SIZE * sizeof(float)));
}

template <const int INPUT_SIZE, const int HIDDEN_LAYER_1, const int HIDDEN_LAYER_2, const int OUTPUT_LAYER, const int BATCH_SIZE, const int BLOCK_SIZE>
void initialize_network(NeuralNetwork *net) // change
{
  CUDA_CHECK(cudaMalloc((void **)&net->weights1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float))); // M*K
  CUDA_CHECK(cudaMalloc((void **)&net->biases1, HIDDEN_LAYER_1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float)));
  initLayer<BLOCK_SIZE>(net->weights1, net->biases1, HIDDEN_LAYER_1, INPUT_SIZE);

  CUDA_CHECK(cudaMalloc((void **)&net->weights2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->biases2, HIDDEN_LAYER_2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));
  initLayer<BLOCK_SIZE>(net->weights2, net->biases2, HIDDEN_LAYER_2, HIDDEN_LAYER_1);

  CUDA_CHECK(cudaMalloc((void **)&net->weights3, OUTPUT_LAYER * HIDDEN_LAYER_2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->biases3, OUTPUT_LAYER * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float)));
  initLayer<BLOCK_SIZE>(net->weights3, net->biases3, OUTPUT_LAYER, HIDDEN_LAYER_2);
}

void free_network(NeuralNetwork *net)
{
  cudaFree(net->weights1);
  cudaFree(net->biases1);
  cudaFree(net->grad_layer1);

  cudaFree(net->weights2);
  cudaFree(net->biases2);
  cudaFree(net->grad_layer2);

  cudaFree(net->weights3);
  cudaFree(net->biases3);
  cudaFree(net->grad_layer3);
}