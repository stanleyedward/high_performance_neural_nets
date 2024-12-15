#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <chrono>

#define INPUT_SIZE 784
#define LABELS_SIZE 10
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define BLOCK_SIZE 16
#define LR 0.03
#define EPOCHS 10
#define TRAIN_LENGTH 60000
#define TEST_LENGTH 10000

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

typedef struct
{ 
  float *x1;
  float *a1;

  float *x2;
  float *a2;

  float *x3;
  float *a3;

  float *losses;
} Outputs;

__global__ void update_layer_params(int w, int h, int batch_size, float lr, float *weights, float *biases, float *layer_input, float *d_l)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && column < w)
  {
    float dw = 0.f;
    float db = 0.f;
    for (int i = 0; i < batch_size; i++)
    {
      float input = layer_input[i * h + row];
      float dl = d_l[i * w + column];
      dw += input * dl;
      db += dl;
    }
    weights[row * w + column] -= lr * dw / batch_size;
    biases[column] -= lr * db / batch_size;
  }
}
__global__ void linear_forward(int batch_size, int n, int out_w, float *input, float *weights, float *biases, float *output)
{
  const uint column = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    output[row * out_w + column] = biases[column];
    for (int i = 0; i < n; i++)
    {
      output[row * out_w + column] += weights[i * out_w + column] * input[row * n + i];
    }
  }
}

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

__global__ void relu_forward(int w, int h, float *a, float *b)
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

__global__ void softmax(int w, int h, float *a, float *b)
{
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
    // subtract with maxval for numeric stability
    float maxval = a[row * w];
    for (int i = 1; i < w; i++)
    {
      maxval = max(maxval, a[row * w + i]);
    }
    float divisor = 0.f;
    for (int i = 0; i < w; i++)
    {
      divisor += exp(a[row * w + i] - maxval);
    }
    b[row * w + col] = exp(a[row * w + col] - maxval) / (divisor);
  }
}

__global__ void init_kaiming_normal(int W, int H, float *matrix)
{
  const uint row = blockDim.x * blockIdx.x + threadIdx.x;
  const uint col = blockDim.y * blockIdx.y + threadIdx.y;

  if (row < H && col < W)
  {
    curandState state;
    curand_init(42, row * W + col, 0, &state);
    matrix[row * W + col] = curand_normal(&state) * sqrtf(2.f / H);
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

void init_parameters(float *weights, float *biases, int W, int H)
{
  // weights
  dim3 numBlocks = dim3(ceil(W / (float)BLOCK_SIZE), ceil(H / (float)BLOCK_SIZE), 1);
  dim3 numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_kaiming_normal<<<numBlocks, numThreadsPerBlock>>>(W, H, weights);

  // biases
  numBlocks = dim3(ceil(H / (float)BLOCK_SIZE), 1, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
  init_kaiming_normal<<<numBlocks, numThreadsPerBlock>>>(1, H, biases);
}

void initialize_nn(NeuralNetwork *nn)
{
  cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
  cudaMalloc(&nn->biases1, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&nn->grad_layer1, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  init_parameters(nn->weights1, nn->biases1, HIDDEN_SIZE, INPUT_SIZE);

  cudaMalloc(&nn->weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&nn->biases2, HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&nn->grad_layer2, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  init_parameters(nn->weights2, nn->biases2, HIDDEN_SIZE, HIDDEN_SIZE);

  cudaMalloc(&nn->weights3, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
  cudaMalloc(&nn->biases3, OUTPUT_SIZE * sizeof(float));
  cudaMalloc(&nn->grad_layer3, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
  init_parameters(nn->weights3, nn->biases3, OUTPUT_SIZE, HIDDEN_SIZE);
}

void init_outputs(Outputs *op)
{
  cudaMalloc(&op->x1, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  cudaMalloc(&op->a1, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));

  cudaMalloc(&op->x2, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
  cudaMalloc(&op->a2, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));

  cudaMalloc(&op->x3, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
  cudaMalloc(&op->a3, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));

  cudaMalloc(&op->losses, BATCH_SIZE * sizeof(float));
}

void forward(NeuralNetwork *nn, Outputs *op, float *input, float *labels)
{
  linear_forward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(BATCH_SIZE, INPUT_SIZE, HIDDEN_SIZE, input, nn->weights1, nn->biases1, op->x1);
  relu_forward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, BATCH_SIZE, op->x1, op->a1);
  linear_forward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, op->a1, nn->weights2, nn->biases2, op->x2);
  relu_forward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, BATCH_SIZE, op->x2, op->a2);
  linear_forward<<<dim3(OUTPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(BATCH_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, op->a2, nn->weights3, nn->biases3, op->x3);
  softmax<<<dim3(OUTPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(OUTPUT_SIZE, BATCH_SIZE, op->x3, op->a3);
  cross_entropy<<<dim3(OUTPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1, 1), dim3(BLOCK_SIZE, 1, 1)>>>(OUTPUT_SIZE, BATCH_SIZE, op->a3, labels, op->losses);
  cudaDeviceSynchronize();
}

void backward(NeuralNetwork *nn, Outputs *op, float *labels)
{
  cross_entropy_backwards<<<dim3(OUTPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(OUTPUT_SIZE, BATCH_SIZE, op->a3, labels, nn->grad_layer3);
  linear_backward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(BATCH_SIZE, OUTPUT_SIZE, HIDDEN_SIZE, nn->weights3, nn->biases3, nn->grad_layer3, nn->grad_layer2);
  relu_backwards<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, BATCH_SIZE, op->a2, nn->grad_layer2, nn->grad_layer2);
  linear_backward<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(BATCH_SIZE, HIDDEN_SIZE, HIDDEN_SIZE, nn->weights2, nn->biases2, nn->grad_layer2, nn->grad_layer1);
  relu_backwards<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, BATCH_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, BATCH_SIZE, op->a1, nn->grad_layer1, nn->grad_layer1);
  cudaDeviceSynchronize();
}

void optimizer_step(NeuralNetwork *nn, Outputs *op, float *inputs)
{
  update_layer_params<<<dim3(OUTPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(OUTPUT_SIZE, HIDDEN_SIZE, BATCH_SIZE, LR, nn->weights3, nn->biases3, op->a2, nn->grad_layer3);
  update_layer_params<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, HIDDEN_SIZE, BATCH_SIZE, LR, nn->weights2, nn->biases2, op->a1, nn->grad_layer2);
  update_layer_params<<<dim3(HIDDEN_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, INPUT_SIZE + BLOCK_SIZE - 1 / (float)BLOCK_SIZE, 1), dim3(BLOCK_SIZE, BLOCK_SIZE, 1)>>>(HIDDEN_SIZE, INPUT_SIZE, BATCH_SIZE, LR, nn->weights1, nn->biases1, inputs, nn->grad_layer1);
  cudaDeviceSynchronize();
}

void read_mnist(const std::string filename, int length, float *x, float *y)
{
  int input_size = 784;
  int labels = 10;

  std::fstream fin;
  fin.open(filename);
  std::string row;
  constexpr char delim = ',';
  for (int i = 0; i < length; i++)
  {
    fin >> row;
    int pos = row.find(delim);
    int label = std::stoi(row.substr(0, pos + 1));
    for (int j = 0; j < labels; j++)
    {
      y[labels * i + j] = (j == label);
    }
    row.erase(0, pos + 1);
    for (int j = 0; j < input_size; j++)
    {
      pos = row.find(delim);
      if (pos == std::string::npos)
      {
        pos = row.length() - 1;
      }
      x[i * input_size + j] = std::stof(row.substr(0, pos + 1)) / 255; // normalize value
      row.erase(0, pos + 1);
    }
  }
}

void train_loop(NeuralNetwork *nn, Outputs *op, float* train_x, float* train_y, float* input, float* labels, float* out_h, float* loss_h)
{

  float total_time = 0.f;
  for (int epoch = 0; epoch < EPOCHS; epoch++)
  {
    float cum_loss = 0.f;
    int correct = 0;
    int total = 0;
    auto start_time = std::chrono::system_clock::now();
    for (int batch = 0; batch < TRAIN_LENGTH / BATCH_SIZE; batch++)
    {
      total += BATCH_SIZE;
      cudaMemcpy(input, &train_x[batch * BATCH_SIZE * INPUT_SIZE], BATCH_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
      cudaMemcpy(labels, &train_y[batch * BATCH_SIZE * LABELS_SIZE], BATCH_SIZE * LABELS_SIZE * sizeof(float), cudaMemcpyHostToDevice);
      cudaDeviceSynchronize();

      forward(nn, op, input, labels);
      backward(nn, op, labels);
      optimizer_step(nn, op, input);

      
      cudaMemcpy(out_h, op->a3, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
      cudaMemcpy(loss_h, op->losses, BATCH_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

      for (int i = 0; i < BATCH_SIZE; i++)
      {
        float max_1 = 0.f;
        float max_2 = 0.f;
        int i1 = 0;
        int i2 = 0;
        for (int j = 0; j<LABELS_SIZE; j++)
        {
          if (out_h[i*LABELS_SIZE + j] > max_1)
          {
            max_1 = out_h[i*LABELS_SIZE + j];
            i1 = j;
          }
          
          if (train_y[batch*BATCH_SIZE*LABELS_SIZE + i*LABELS_SIZE + j] > max_2)
          {
            max_2 = train_y[batch*BATCH_SIZE*LABELS_SIZE + i*LABELS_SIZE + j];
            i2 = j;
          }
        }
        correct += (i1 == i2);
        cum_loss += loss_h[i];
      }
    }
    printf("[INFO] Epoch %d\n", epoch);
    printf("Total Accuracy: %f\n", (float)correct / total);
    printf("Total Loss: %f\n", cum_loss);
  }
}
  int main()
  {
    float *input;
    float *labels;
    cudaMalloc((void**) &input, INPUT_SIZE*BATCH_SIZE*sizeof(float));
    cudaMalloc((void**) &labels, LABELS_SIZE*BATCH_SIZE*sizeof(float));

    float *train_x = new float[INPUT_SIZE * TRAIN_LENGTH];
    float *train_y = new float[LABELS_SIZE * TRAIN_LENGTH];
    float *test_x = new float[INPUT_SIZE * TEST_LENGTH];
    float *test_y = new float[LABELS_SIZE * TEST_LENGTH];

    float *output_host = new float[OUTPUT_SIZE * BATCH_SIZE];
    float *loss_host = new float[BATCH_SIZE];

    // load data
    read_mnist("./mnist_train.csv", TRAIN_LENGTH, train_x, train_y);
    read_mnist("./mnist_test.csv", TEST_LENGTH, test_x, test_y);

    // init network
    NeuralNetwork nn;
    initialize_nn(&nn);
    Outputs op;
    init_outputs(&op);

    // train loop
    //  test loop

    return 0;
  }