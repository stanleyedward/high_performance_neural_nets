## High Performance NNs
cuda 121
nn


### TODO
- [x] add verify results to activation 
- [x] read mnist dataset
- [x] softmax optim
- [x] matmul optim
- [x] update the mm kernels to compute biases
- [x] backward pass module
- [x] modularize code 
- [x] create make file
- [x] operator fusion
- [ ] remove hardcoded blocksize in activation, loss and backward
  
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

### nn
unoptimized nn implementation

```sh
Hyperparameters:
Hidden Layer 1: 256
Hidden Layer 2: 256
Output Layer: 10
Batch Size: 256
Learning Rate: 0.015000
Epochs: 10

read mnist took 2389 ms
initialization took 1863 ms
epoch 0 270ms | total loss 5945.81 | accuracy 0.495126 | val loss 529.544 | val accuracy 0.766827
epoch 1 238ms | total loss 2439.71 | accuracy 0.810113 | val loss 346.389 | val accuracy 0.847957
epoch 2 238ms | total loss 1854.15 | accuracy 0.851613 | val loss 299.758 | val accuracy 0.869792
epoch 3 239ms | total loss 1638.33 | accuracy 0.870476 | val loss 276.336 | val accuracy 0.880609
epoch 4 241ms | total loss 1517.68 | accuracy 0.880826 | val loss 262.91 | val accuracy 0.887821
epoch 5 238ms | total loss 1440.33 | accuracy 0.886685 | val loss 254.229 | val accuracy 0.892127
epoch 6 239ms | total loss 1386.69 | accuracy 0.891476 | val loss 247.852 | val accuracy 0.897135
epoch 7 238ms | total loss 1344.75 | accuracy 0.894732 | val loss 242.446 | val accuracy 0.901242
epoch 8 239ms | total loss 1309.51 | accuracy 0.898438 | val loss 237.558 | val accuracy 0.903646
epoch 9 238ms | total loss 1279.85 | accuracy 0.901542 | val loss 232.655 | val accuracy 0.90625
finished training, total time = 2418 ms
```

optimized

```sh
Hyperparameters:
Hidden Layer 1: 256
Hidden Layer 2: 256
Output Layer: 10
Batch Size: 256
Learning Rate: 0.015000
Epochs: 10

read mnist took 1454 ms
initialization took 113 ms
epoch 0 144ms | total loss 5960.6 | accuracy 0.49379 | val loss 531.156 | val accuracy 0.766627
epoch 1 124ms | total loss 2440.52 | accuracy 0.810196 | val loss 347.35 | val accuracy 0.848057
epoch 2 123ms | total loss 1854.34 | accuracy 0.851345 | val loss 300.709 | val accuracy 0.870092
epoch 3 123ms | total loss 1636.67 | accuracy 0.870393 | val loss 277.957 | val accuracy 0.880208
epoch 4 123ms | total loss 1517.52 | accuracy 0.880626 | val loss 264.129 | val accuracy 0.887821
epoch 5 123ms | total loss 1439.88 | accuracy 0.886936 | val loss 255.306 | val accuracy 0.892929
epoch 6 123ms | total loss 1385.67 | accuracy 0.890875 | val loss 248.474 | val accuracy 0.897736
epoch 7 123ms | total loss 1343.11 | accuracy 0.895015 | val loss 242.408 | val accuracy 0.901342
epoch 8 123ms | total loss 1307.31 | accuracy 0.898738 | val loss 236.967 | val accuracy 0.905048
epoch 9 123ms | total loss 1276.57 | accuracy 0.901542 | val loss 231.689 | val accuracy 0.908153
finished training, total time = 1252 ms
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
- for smem tiling tile_dim has to be a multiple of M, N, K

### References

https://siboehm.com/articles/22/CUDA-MMM