#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 1024
#define OUTPUT_SIZE 10
#define BATCH_SIZE 64
#define BLOCK_SIZE 16

typedef struct {
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

typedef struct {
  float *x1;
  float *a1;

  float *x2;
  float *a2;
  
  float *x3;
  float *a3;

  float *losses;
} Outputs;

__global__ void matmut_add(int batch_size, int n, int out_w, float* input, float* weights, float* biases, float* output)
{
  int column = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row < batch_size && column < out_w)
  {
    output[row*out_w+column] = biases[column];
    for(int i = 0; i < n; i++)
    {
      output[row*out_w+column] += weights[i*out_w + column] * input[row*n + i];
    }
  }
}

__global__ void init_kaiming_normal(int W, int H, float* matrix){
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < H && col < W){
    curandState state;
    curand_init(42, row*W+col, 0, &state);
    matrix[row*W + col] = curand_normal(&state)*sqrtf(2.f/H);
    }
}

void init_parameters(float* weights, float* biases, int W, int H)
{
// weights
  dim3 numBlocks = dim3(ceil(W/(float)BLOCK_SIZE), ceil(H/(float)BLOCK_SIZE), 1);
  dim3 numThreadsPerBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  init_kaiming_normal<<<numBlocks, numThreadsPerBlock>>>(W, H, weights);

// biases
  numBlocks = dim3(ceil(H/(float)BLOCK_SIZE), 1, 1);
  numThreadsPerBlock = dim3(BLOCK_SIZE, 1, 1);
  init_kaiming_normal<<<numBlocks, numThreadsPerBlock>>>(1, H, biases);
}

void initialize_nn(NeuralNetwork* nn){
    cudaMalloc(&nn->weights1, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->biases1, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->grad_layer1, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
    init_parameters(nn->weights1, nn->biases1, HIDDEN_SIZE, INPUT_SIZE);

    cudaMalloc(&nn->weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->biases2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->grad_layer2, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
    init_parameters(nn->weights1, nn->biases1, HIDDEN_SIZE, HIDDEN_SIZE);
    
    cudaMalloc(&nn->weights3, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->biases3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->grad_layer3, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
    init_parameters(nn->weights3, nn->biases3, OUTPUT_SIZE, HIDDEN_SIZE);
}

void init_outputs(Outputs* op){
    cudaMalloc((void**) &op->x1, HIDDEN_SIZE*BATCH_SIZE*sizeof(float));
    cudaMalloc((void**) &op->a1, HIDDEN_SIZE*BATCH_SIZE*sizeof(float));

    cudaMalloc((void**) &op->x2, HIDDEN_SIZE*BATCH_SIZE*sizeof(float));
    cudaMalloc((void**) &op->a2, HIDDEN_SIZE*BATCH_SIZE*sizeof(float));

    cudaMalloc((void**) &op->x3, OUTPUT_SIZE*BATCH_SIZE*sizeof(float));
    cudaMalloc((void**) &op->a3, OUTPUT_SIZE*BATCH_SIZE*sizeof(float));

    cudaMalloc((void**) &op->losses, BATCH_SIZE*sizeof(float));
}

int main(){
    NeuralNetwork nn;
    initialize_nn(&nn);

    Outputs op;
    init_outputs(&op);

}