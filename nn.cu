#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <string>

#define TEST_LENGTH 10000
#define TRAIN_LENGTH 60000

#define INPUT_SIZE 784
#define LABELS_SIZE 10

#define HIDDEN_LAYER_1 256
#define HIDDEN_LAYER_2 256
#define OUTPUT_LAYER 10

#define BLOCK_SIZE 16
#define BATCH_SIZE 32
#define EPOCHS 10
#define LR 0.003

// Modify the CUDA_CHECK macro to print more information
#define CUDA_CHECK(call)                                               \
  do                                                                   \
  {                                                                    \
    cudaError_t error = call;                                          \
    if (error != cudaSuccess)                                          \
    {                                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(error));                              \
      cudaDeviceReset();                                               \
      exit(EXIT_FAILURE);                                              \
    }                                                                  \
  } while (0)

class Timer
{
public:
  Timer(std::string in_name) : name(in_name)
  {
    start_time = std::chrono::system_clock::now();
  }
  ~Timer()
  {
    std::cout << name << " took " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - start_time).count() << " ms" << std::endl;
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start_time;
  std::string name;
};

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

__global__ void softmax(int w, int h, float *a, float *b)
{
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < h && col < w)
  {
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

void init_outputs(float **input, float **labels, float **x1, float **a1, float **x2, float **a2, float **x3, float **a3, float **loss)
{
  CUDA_CHECK(cudaMalloc((void **)input, INPUT_SIZE * BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)labels, LABELS_SIZE * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)a1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)a2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)x3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)a3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float)));

  CUDA_CHECK(cudaMalloc((void **)loss, BATCH_SIZE * sizeof(float)));
}

void initialize_network(NeuralNetwork *net) // change
{
  CUDA_CHECK(cudaMalloc((void **)&net->weights1, HIDDEN_LAYER_1 * INPUT_SIZE * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->biases1, HIDDEN_LAYER_1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer1, HIDDEN_LAYER_1 * BATCH_SIZE * sizeof(float)));
  initLayer(net->weights1, net->biases1, HIDDEN_LAYER_1, INPUT_SIZE);

  CUDA_CHECK(cudaMalloc((void **)&net->weights2, HIDDEN_LAYER_2 * HIDDEN_LAYER_1 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->biases2, HIDDEN_LAYER_2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer2, HIDDEN_LAYER_2 * BATCH_SIZE * sizeof(float)));
  initLayer(net->weights2, net->biases2, HIDDEN_LAYER_2, HIDDEN_LAYER_1);

  CUDA_CHECK(cudaMalloc((void **)&net->weights3, OUTPUT_LAYER * HIDDEN_LAYER_2 * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->biases3, OUTPUT_LAYER * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **)&net->grad_layer3, OUTPUT_LAYER * BATCH_SIZE * sizeof(float)));
  initLayer(net->weights3, net->biases3, OUTPUT_LAYER, HIDDEN_LAYER_2);
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

void forward_pass(NeuralNetwork *net, float *input, float *x1, float *a1, float *x2, float *a2, float *x3, float *a3)
{
  dim3 numBlocks, numThreadsPerBlock;

  numBlocks = dim3((HIDDEN_LAYER_1 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  linear_forward<<<numBlocks, numThreadsPerBlock>>>(BATCH_SIZE, INPUT_SIZE, HIDDEN_LAYER_1, input, net->weights1, net->biases1, x1);
  CUDA_CHECK(cudaPeekAtLastError());
  relu<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_1, BATCH_SIZE, x1, a1);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((HIDDEN_LAYER_2 + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  linear_forward<<<numBlocks, numThreadsPerBlock>>>(BATCH_SIZE, HIDDEN_LAYER_1, HIDDEN_LAYER_2, a1, net->weights2, net->biases2, x2);
  CUDA_CHECK(cudaPeekAtLastError());
  relu<<<numBlocks, numThreadsPerBlock>>>(HIDDEN_LAYER_2, BATCH_SIZE, x2, a2);
  CUDA_CHECK(cudaPeekAtLastError());

  numBlocks = dim3((OUTPUT_LAYER + BLOCK_SIZE - 1) / BLOCK_SIZE, (BATCH_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE, 1);
  linear_forward<<<numBlocks, numThreadsPerBlock>>>(BATCH_SIZE, HIDDEN_LAYER_2, OUTPUT_LAYER, a2, net->weights3, net->biases3, x3);
  CUDA_CHECK(cudaPeekAtLastError());
  softmax<<<numBlocks, numThreadsPerBlock>>>(OUTPUT_LAYER, BATCH_SIZE, x3, a3);
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

void read_mnist(const std::string &filename, int length, float *x, float *y)
{
  const int labels = 10;
  std::ifstream fin(filename);
  if (!fin.is_open())
  {
    throw std::runtime_error("Cannot open file: " + filename);
  }
  // Skip the header row
  std::string header;
  std::getline(fin, header);

  std::string row;
  for (int i = 0; i < length; i++)
  {
    if (!std::getline(fin, row))
    {
      throw std::runtime_error("Not enough rows in the file");
    }
    std::istringstream line_stream(row);
    std::string value;
    // Parse label
    if (!std::getline(line_stream, value, ','))
    {
      throw std::runtime_error("Cannot read label");
    }
    int label = std::stoi(value);
    // Initialize one-hot encoded label
    for (int j = 0; j < labels; j++)
    {
      y[labels * i + j] = (j == label) ? 1.0f : 0.0f;
    }
    // Parse input features
    for (int j = 0; j < INPUT_SIZE; j++)
    {
      if (!std::getline(line_stream, value, ','))
      {
        throw std::runtime_error("Not enough features in row");
      }
      x[i * INPUT_SIZE + j] = std::stof(value) / 255.0f;
    }
  }
}

int main(int argc, char **argv)
{
  float *input;
  float *labels;

  float *train_x = new float[INPUT_SIZE * TRAIN_LENGTH];
  float *train_y = new float[LABELS_SIZE * TRAIN_LENGTH];

  float *test_x = new float[INPUT_SIZE * TEST_LENGTH];
  float *test_y = new float[LABELS_SIZE * TEST_LENGTH];
  {
    Timer t("read mnist");
    read_mnist("./mnist_train.csv", TRAIN_LENGTH, train_x, train_y);
    read_mnist("./mnist_test.csv", TEST_LENGTH, test_x, test_y);
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
    initialize_network(&net);
    init_outputs(&input, &labels, &x1, &a1, &x2, &a2, &x3, &a3, &loss);
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

  // Free memory
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