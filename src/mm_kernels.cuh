#pragma once
#include <cuda_runtime.h>
#include "mm_kernels/0_cpu.cuh"
#include "mm_kernels/1_naive.cuh"
#include "mm_kernels/2_gmemCoalesing.cuh"
#include "mm_kernels/3_smemCaching.cuh"
#include "mm_kernels/4_1dBlockTiling.cuh"