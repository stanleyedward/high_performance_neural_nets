//nvcc nn_test.cu src/mm_runner.cu src/activation_runner.cu -I./src -o nn_test.o -L/usr/local/cuda/lib64 -lcudart

#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <string>
#include "helper_functions.cuh"
#include "init.cuh"
#include "mm_runner.cuh"
#include "activation_runner.cuh"

#define TEST_LENGTH 10000
#define TRAIN_LENGTH 60000

#define INPUT_SIZE 784 // K
#define LABELS_SIZE 10

#define HIDDEN_LAYER_1 256 // M
#define HIDDEN_LAYER_2 256
#define OUTPUT_LAYER 10

#define BLOCK_SIZE 16
#define BATCH_SIZE 256 // N 32
#define TILE_WIDTH 16
#define EPOCHS 10
#define LR 0.015


__global__ void linear_backward(int batch_size, int n, int out_w, float *weights, float *biases, float *d_l, float *out_d_l)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    float dl = 0.f;
    for (int i = 0; i < n; i++)
    {
      float w = weights[i * out_w + column];
      dl += w * d_l[row * n + i];
    }
    out_d_l[row * out_w + column] = dl;
  }
}

__global__ void update_layer(int w, int h, int batch_size, float lr, float *weights, float *biases, float *activations, float *d_l)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < batch_size; i++)
    {
      float act = activations[i * h + row];
      float dl = d_l[i * w + column];
      dw += act * dl;
      db += dl;
    }
    weights[row * w + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
  }
}

__global__ void relu(int w, int h, float *a, float *b)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = a[row * w + column];
    b[row * w + column] = activation > 0.f ? activation : 0.f;
  }
}

__global__ void relu_backwards(int w, int h, float *a, float *d_l, float *b)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float activation = a[row * w + column];
    b[row * w + column] = activation > 0.f ? d_l[row * w + column] : 0.f;
  }
}

__global__ void cross_entropy(int w, int h, float *preds, float *real, float *output)
{
  const uint idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < h)
  {
    float loss = 0.f;
    for (int i = 0; i < w; i++)
    {
      loss -= real[idx * w + i] * log(max(1e-6, preds[idx * w + i]));
    }
    output[idx] = loss;
  }
}

__global__ void cross_entropy_backwards(int w, int h, float *preds, float *real, float *output)
{
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    output[row * w + col] = preds[row * w + col] - real[row * w + col];
  }
}


void forward_pass(NeuralNetwork *net, float *input, float *x1, float *a1, float *x2, float *a2, float *x3, float *a3)
{
  dim3 numBlocks, numThreadsPerBlock;

  numBlocks = dim3((HIDDEN_LAYER_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  run_kernel_MM(4, BLOCK_SIZE, BATCH_SIZE, HIDDEN_LAYER_1, INPUT_SIZE, input, net->weights1, x1, net->biases1);
  CUDA_CHECK(cudaPeekAtLastError());
  relu<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_1, BATCH_SIZE, x1, a1);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  run_kernel_MM(4, BLOCK_SIZE, BATCH_SIZE, HIDDEN_LAYER_2, HIDDEN_LAYER_1, a1, net->weights2, x2, net->biases2);
  CUDA_CHECK(cudaPeekAtLastError());
  relu<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_2, BATCH_SIZE, x2, a2);
  CUDA_CHECK(cudaPeekAtLastError());

  run_kernel_MM(2, BLOCK_SIZE, BATCH_SIZE, OUTPUT_LAYER, HIDDEN_LAYER_2, a2, net->weights3, x3, net->biases3);
  CUDA_CHECK(cudaPeekAtLastError());
  run_kernel_softmax(2, BATCH_SIZE, OUTPUT_LAYER, x3, a3);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void backward_pass(NeuralNetwork *net, float *input, float *labels, float *x1, float *a1, float *x2, float *a2, float *x3, float *a3, float *loss)
{
  dim3 numBlocks, numThreadsPerBlock;

  numBlocks = dim3((OUTPUT_LAYER + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  cross_entropy_backwards<<<numBlocks, numThreadsPerBlock>>>(OUTPUT_LAYER, BATCH_SIZE, a3, labels, net->grad_layer3);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  linear_backward<<<numBlocks, numThreadsPerBlock>>>(BATCH_SIZE, OUTPUT_LAYER, HIDDEN_LAYER_2, net->weights3, net->biases3, net->grad_layer3, net->grad_layer2);
  CUDA_CHECK(cudaPeekAtLastError());
  relu_backwards<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_2, BATCH_SIZE, a2, net->grad_layer2, net->grad_layer2);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  linear_backward<<<numBlocks, numThreadsPerBlock>>>(BATCH_SIZE, HIDDEN_LAYER_2, HIDDEN_LAYER_1, net->weights2, net->biases2, net->grad_layer2, net->grad_layer1);
  CUDA_CHECK(cudaPeekAtLastError());
  relu_backwards<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_1, BATCH_SIZE, a1, net->grad_layer1, net->grad_layer1);
  CUDA_CHECK(cudaPeekAtLastError());

  // Update layers
  numBlocks = dim3((OUTPUT_LAYER + BLOCK_SIZE - 1) / BLOCK_SIZE, (HIDDEN_LAYER_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  update_layer<<<numBlocks, numThreadsPerBlock>>>(OUTPUT_LAYER, HIDDEN_LAYER_2, BATCH_SIZE, LR, net->weights3, net->biases3, a2, net->grad_layer3);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (HIDDEN_LAYER_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  update_layer<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_2, HIDDEN_LAYER_1, BATCH_SIZE, LR, net->weights2, net->biases2, a1, net->grad_layer2);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (INPUT_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  update_layer<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_1, INPUT_SIZE, BATCH_SIZE, LR, net->weights1, net->biases1, input, net->grad_layer1);
  CUDA_CHECK(cudaPeekAtLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}


int main(int argc, char **argv)
{
  printf("Hyperparameters:\n");
  printf("Hidden Layer 1: %d\n", HIDDEN_LAYER_1);
  printf("Hidden Layer 2: %d\n", HIDDEN_LAYER_2);
  printf("Output Layer: %d\n", OUTPUT_LAYER);
  printf("Batch Size: %d\n", BATCH_SIZE);
  printf("Learning Rate: %f\n", LR);
  printf("Epochs: %d\n", EPOCHS);
  printf("\n");

  float *input;
  float *labels;

  float *train_x = new float[INPUT_SIZE * TRAIN_LENGTH];
  float *train_y = new float[LABELS_SIZE * TRAIN_LENGTH];

  float *test_x = new float[INPUT_SIZE * TEST_LENGTH];
  float *test_y = new float[LABELS_SIZE * TEST_LENGTH];
  {
    Timer t("read mnist");
    read_mnist<INPUT_SIZE>("./mnist_train.csv", TRAIN_LENGTH, train_x, train_y);
    read_mnist<INPUT_SIZE>("./mnist_test.csv", TEST_LENGTH, test_x, test_y);
  }

  NeuralNetwork net;

  float *out_h = new float[BATCH_SIZE * OUTPUT_LAYER];
  float *loss_h = new float[BATCH_SIZE];

  float *x1;
  float *a1;
  float *x2;
  float *a2;
  float *x3;
  float *a3;
  float *loss;

  {
    Timer init("initialization");
    initialize_network<INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_LAYER, BATCH_SIZE, BLOCK_SIZE>(&net);
    init_outputs<INPUT_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, OUTPUT_LAYER, BATCH_SIZE, BLOCK_SIZE, LABELS_SIZE>(&input, &labels, &x1, &a1, &x2, &a2, &x3, &a3, &loss);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float total_time = 0.f;
  for (int epoch = 0; epoch < EPOCHS; epoch++)
  {
    float total_loss = 0.f;
    int correct = 0;
    int total = 0;
    auto start_time = std::chrono::system_clock::now();
    for (int batch = 0; batch < TRAIN_LENGTH / BATCH_SIZE; batch++)
    {
      total += BATCH_SIZE;
      CUDA_CHECK(cudaMemcpy(input, &train_x[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(labels, &train_y[batch * BATCH_SIZE * LABELS_SIZE], BATCH_SIZE * LABELS_SIZE * sizeof(float), cudaMemcpyHostToDevice));

      forward_pass(&net, input, x1, a1, x2, a2, x3, a3);

      // Compute loss
      dim3 numBlocks = dim3((OUTPUT_LAYER + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
      dim3 numThreadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
      cross_entropy<<<numBlocks, numThreadsPerBlock>>>(OUTPUT_LAYER, BATCH_SIZE, a3, labels, loss);
      CUDA_CHECK(cudaPeekAtLastError());

      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(out_h, a3, BATCH_SIZE * OUTPUT_LAYER * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(loss_h, loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());
      for (int i = 0; i < BATCH_SIZE; i++)
      {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j < LABELS_SIZE; j++)
        {
          if (out_h[i * LABELS_SIZE + j] > max_1)
          {
            max_1 = out_h[i * LABELS_SIZE + j];
            i1 = j;
          }

          if (train_y[batch * BATCH_SIZE * LABELS_SIZE + i * LABELS_SIZE + j] > max_2)
          {
            max_2 = train_y[batch * BATCH_SIZE * LABELS_SIZE + i * LABELS_SIZE + j];
            i2 = j;
          }
        }
        correct += (i1 == i2);
        total_loss += loss_h[i];
      }

      backward_pass(&net, input, labels, x1, a1, x2, a2, x3, a3, loss);
    }

    // Validation
    float val_loss = 0.f;
    int val_correct = 0;
    int val_total = 0;
    for (int batch = 0; batch < TEST_LENGTH / BATCH_SIZE; batch++)
    {
      val_total += BATCH_SIZE;
      CUDA_CHECK(cudaMemcpy(input, &test_x[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMemcpy(labels, &test_y[batch * BATCH_SIZE * LABELS_SIZE], BATCH_SIZE * LABELS_SIZE * sizeof(float), cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaDeviceSynchronize());
      forward_pass(&net, input, x1, a1, x2, a2, x3, a3);

      dim3 numBlocks = dim3((OUTPUT_LAYER + BLOCK_SIZE - 1) / BLOCK_SIZE, 1, 1);
      dim3 numThreadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
      cross_entropy<<<numBlocks, numThreadsPerBlock>>>(OUTPUT_LAYER, BATCH_SIZE, a3, labels, loss);
      CUDA_CHECK(cudaPeekAtLastError());
      CUDA_CHECK(cudaDeviceSynchronize());

      CUDA_CHECK(cudaMemcpy(out_h, a3, BATCH_SIZE * OUTPUT_LAYER * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(loss_h, loss, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaDeviceSynchronize());

      for (int i = 0; i < BATCH_SIZE; i++)
      {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j < LABELS_SIZE; j++)
        {
          if (out_h[i * LABELS_SIZE + j] > max_1)
          {
            max_1 = out_h[i * LABELS_SIZE + j];
            i1 = j;
          }

          if (test_y[batch * BATCH_SIZE * LABELS_SIZE + i * LABELS_SIZE + j] > max_2)
          {
            max_2 = test_y[batch * BATCH_SIZE * LABELS_SIZE + i * LABELS_SIZE + j];
            i2 = j;
          }
        }
        val_correct += (i1 == i2);
        val_loss += loss_h[i];
      }
    }

    float epoch_time = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count();
    total_time += epoch_time;
    std::cout << "epoch " << epoch << " " << epoch_time << "ms | total loss " << total_loss << " | accuracy " << (float)correct / total << " | val loss " << val_loss << " | val accuracy " << (float)val_correct / val_total << std::endl;
  }
  std::cout << "finished training, total time = " << total_time << " ms" << std::endl;

  // free mem
  free_network(&net);
  cudaFree(input);
  cudaFree(labels);
  cudaFree(x1);
  cudaFree(a1);
  cudaFree(x2);
  cudaFree(a2);
  cudaFree(x3);
  cudaFree(a3);
  cudaFree(loss);

  delete[] train_x;
  delete[] train_y;
  delete[] test_x;
  delete[] test_y;
  delete[] out_h;
  delete[] loss_h;

  return 0;
}