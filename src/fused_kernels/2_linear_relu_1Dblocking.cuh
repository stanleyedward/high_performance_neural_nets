#pragma once
#include <cuda_runtime.h>
#include <cassert>

template <const int BM, const int BN, const int BK, const int TM>
__global__ void linear_relu_1dBlocking(int M, int N, int K, const float *A, const float *B, float *C, float *bias) {
  const uint cRow = blockIdx.y;
  const uint cCol = blockIdx.x;

  const int threadCol = threadIdx.x % BN;
  const int threadRow = threadIdx.x / BN;

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  A += cRow * BM * K;
  B += cCol * BN;
  C += cRow * BM * N + cCol * BN;

  assert(BM * BK == blockDim.x);
  assert(BN * BK == blockDim.x);

  const uint innerColA = threadIdx.x % BK; 
  const uint innerRowA = threadIdx.x / BK;
  const uint innerColB = threadIdx.x % BN; 
  const uint innerRowB = threadIdx.x / BN;

  float threadResults[TM] = {0.0};

  for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
    As[innerRowA * BK + innerColA] = A[innerRowA * K + innerColA];
    Bs[innerRowB * BN + innerColB] = B[innerRowB * N + innerColB];
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
      float tmpB = Bs[dotIdx * BN + threadCol];
      for (uint resIdx = 0; resIdx < TM; ++resIdx) {
        threadResults[resIdx] +=
            As[(threadRow * TM + resIdx) * BK + dotIdx] * tmpB;
      }
    }
    __syncthreads();
  }

  for (uint resIdx = 0; resIdx < TM; ++resIdx) {
    C[(threadRow * TM + resIdx) * N + threadCol] = 
    threadResults[resIdx] + bias[cCol * BN + threadCol] > 0.f? 
    threadResults[resIdx] + bias[cCol * BN + threadCol] : 0.f;
  }
}