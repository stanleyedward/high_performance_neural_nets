#include <curand.h>
#include <curand_kernel.h>

__global__ void init_kaiming_normal(int W, int H, float* mat){
    const uint row = blockDim.x * blockIdx.x + threadIdx.x;
    const uint col = blockDim.y * blockIdx.y + threadIdx.y;

    if (row < H && col < W){
    curandState state;
    curand_init(42, row*W+col, 0, &state);
    mat[row*W + col] = curand_normal(&state)*sqrtf(2.f/H);
    }
}

void initLayer(float* weights, float* biases, int W, int H, int BLOCK_SIZE)
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