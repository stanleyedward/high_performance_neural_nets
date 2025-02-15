## nn cuda
cuda 121
nn


### TODO
- [x] add verify results to activation 
- [x] read mnist dataset
- [x] softmax optim
- [x] matmul optim
- [ ] modularize code 
- [ ] create make file
  
  -  pin mem
  -  coalesed accesses
  -  shared mem blocking
  -  tiling

- [ ] streams
  - data loading
  - kernel launches

### Matrix Multiplication
```sh
Matrix A: 1024 x 1024
Matrix B: 1024 x 1024
Matrix C: 1024 x 1024
Block size: 16 x 16

Naive Kernel:
Time: 13.4953 ms
Performance: 159.13 GFLOPS

SMEM Blocking Kernel:
Time: 2.5004 ms
Performance: 858.84 GFLOPS

Validation Results:
Naive Kernel max error: 9.155273e-05
SMEM Blocking Kernel max error: 9.155273e-05
```

### Activation
```sh
Matrix sizes: 1024 x 1024

naive softmax execution time: 52.6336 ms
naive softmax GFLOPS: 0.1395
optimized softmax execution time: 0.1956 ms
optimized softmax GFLOPS: 37.5288

Verifying Results
Max difference (act1 vs CPU): 5.587935e-09
Max difference (act2 vs CPU): 4.097819e-08
```
https://github.com/NVIDIA/cuda-samples.git

```sh
cuda-samples/Samples/1_Utilities/deviceQuery git:master  
(torch241cuda121) ❯ ./deviceQuery
./deviceQuery Starting...

 CUDA Device Query (Runtime API) version (CUDART static linking)

Detected 1 CUDA Capable device(s)

Device 0: "NVIDIA GeForce RTX 4050 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.7 / 12.6
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 5797 MBytes (6078595072 bytes)
  (020) Multiprocessors, (128) CUDA Cores/MP:    2560 CUDA Cores
  GPU Max Clock rate:                            2130 MHz (2.13 GHz)
  Memory Clock rate:                             8001 Mhz
  Memory Bus Width:                              96-bit
  L2 Cache Size:                                 25165824 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               65536 bytes
  Total amount of shared memory per block:       49152 bytes
  Total shared memory per multiprocessor:        102400 bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024
  Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
  Max dimension size of a grid size    (x,y,z): (2147483647, 65535, 65535)
  Maximum memory pitch:                          2147483647 bytes
  Texture alignment:                             512 bytes
  Concurrent copy and kernel execution:          Yes with 2 copy engine(s)
  Run time limit on kernels:                     No
  Integrated GPU sharing Host Memory:            No
  Support host page-locked memory mapping:       Yes
  Alignment requirement for Surfaces:            Yes
  Device has ECC support:                        Disabled
  Device supports Unified Addressing (UVA):      Yes
  Device supports Managed Memory:                Yes
  Device supports Compute Preemption:            Yes
  Supports Cooperative Kernel Launch:            Yes
  Supports MultiDevice Co-op Kernel Launch:      Yes
  Device PCI Domain ID / Bus ID / location ID:   0 / 1 / 0
  Compute Mode:
     < Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 12.7, CUDA Runtime Version = 12.6, NumDevs = 1
Result = PASS
```

### notes
- 1d relu is faster than 2d relu
- for smem tiling tile_dim has to be a multiple of M, N, K

### References

https://siboehm.com/articles/22/CUDA-MMM