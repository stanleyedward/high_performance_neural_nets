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

__global__ void init_kaiming_normal(int W, int H, float* matrix){
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < H && col < W){
    curandState state;
    curand_init(42, row*W+col, 0, &state);
    matrix[row*W + col] = curand_normal(&state)*sqrtf(2.f/H);
    }
}

void init_parameters(float* weights, float* biases, int W, int H, int BLOCK_SIZE)
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
    init_parameters(nn->weights1, nn->biases1, HIDDEN_SIZE, INPUT_SIZE, BLOCK_SIZE);

    cudaMalloc(&nn->weights2, HIDDEN_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->biases2, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->grad_layer2, HIDDEN_SIZE * BATCH_SIZE * sizeof(float));
    init_parameters(nn->weights1, nn->biases1, HIDDEN_SIZE, HIDDEN_SIZE, BLOCK_SIZE);
    
    cudaMalloc(&nn->weights3, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&nn->biases3, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&nn->grad_layer3, OUTPUT_SIZE * BATCH_SIZE * sizeof(float));
    init_parameters(nn->weights3, nn->biases3, OUTPUT_SIZE, HIDDEN_SIZE, BLOCK_SIZE);
}

int main(){
    NeuralNetwork nn;

}