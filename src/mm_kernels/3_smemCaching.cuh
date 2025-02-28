#pragma once
#include <cuda_runtime.h>

template <uint BLOCK_SIZE>
__global__ void mm3(uint M, uint N, uint K, float *A, float *B, float *C, float *bias){
  // siboehm 
  // M, N, K have to be multiples of BLOCK_SIZE
  // shared memory 
  // 1D blocks
  const uint cRow = blockIdx.x;
  const uint cCol = blockIdx.y;

  __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

  const uint threadCol = threadIdx.x % BLOCK_SIZE;
  const uint threadRow = threadIdx.x / BLOCK_SIZE;

  A += cRow * BLOCK_SIZE * K;                    // row=cRow, col=0
  B += cCol * BLOCK_SIZE;                        // row=0, col=cCol
  C += cRow * BLOCK_SIZE * N + cCol * BLOCK_SIZE; // row=cRow, col=cCol

  float tmp = 0.0;
  for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCK_SIZE) {
    As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
    Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * N + threadCol];

    __syncthreads();
    A += BLOCK_SIZE;
    B += BLOCK_SIZE * N;

    for (int dotIdx = 0; dotIdx < BLOCK_SIZE; ++dotIdx) {
      tmp += As[threadRow * BLOCK_SIZE + dotIdx] *
             Bs[dotIdx * BLOCK_SIZE + threadCol];
    }
    __syncthreads();
  }
  C[threadRow * N + threadCol] = bias[cCol * BLOCK_SIZE + threadCol] + tmp;
}
