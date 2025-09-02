## High Performance NNs
i was learning GPU computing by optimizing neural network kernels using cuda and night

todo
[ ] implement convolutional layers


##### Setup
```sh
conda install cuda -c nvidia/label/cuda-12.6.3
make all
# run ./nn.o or ./mm.o ./activation.o
# modify hyperparameters through macros in their respective files
```

### FMA/MatMul

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="matrix multiplication GFLOPS" src="assets/mm_gflops.png"> Higher is better |  <img width="1604" alt="matrix multiplication time" src="assets/mm_time.png"> Lower is better |

```sh
$ ./mm.o
Matrix A: 1024 x 1024
Matrix B: 1024 x 1024
Matrix C: 1024 x 1024
Block size: 16 x 16

Naive Kernel:
Time: 13.4953 ms
Performance: 159.13 GFLOPS

1D Block Tiling Kernel:
Time: 0.8926 ms
Performance: 2405.85 GFLOPS

Validation Results:
Naive Kernel max error: 9.155273e-05
1D Block Tiling Kernel max error: 9.155273e-05
```

### Softmax Function

| | |
|:-------------------------:|:-------------------------:|
|<img width="1604" alt="softmax GFLOPS" src="assets/softmax_gflops.png"> Higher is better |  <img width="1604" alt="softmax time" src="assets/softmax_time.png"> Lower is better |

```sh
$ ./activation.o
Matrix size: 1024 x 1024

Naive Softmax Kernel:
Time: 3.4529 ms
Performance: 2.4294 GLFOPS

SMEM Reduction Softmax Kernel:
Time: 0.0451 ms
Performance: 162.9091 GLFOPS

Verifying Results:
Naive Kernel max error: 5.587935e-09
SMEM reduction Kernel max error: 4.097819e-08
```

### Neural Network
  |             | Time per epoch   | Performance (%) | Performance (x)|
  | :---:          |  :---:           | :---:| :---:|
  | **Naive NN**      |  239 ms          |100%|1.00x|
  | **Optimized NN**  |  123 ms          |148.53%|1.94x|


#### Naive NN implementation

```sh
$ ./nn.o
Hyperparameters:
Hidden Layer 1: 256
Hidden Layer 2: 256
Output Layer: 10
Batch Size: 256
Learning Rate: 0.015000
Epochs: 10

read mnist took 2381 ms
initialization took 1860 ms
epoch 0 269ms | loss 5963.16 | acc 0.494525 | val loss 534.968 | val acc 0.768429
epoch 1 238ms | loss 2455.19 | acc 0.809145 | val loss 348.982 | val acc 0.847155
epoch 2 238ms | loss 1856.23 | acc 0.851179 | val loss 300.288 | val acc 0.86899
epoch 3 239ms | loss 1630.9 | acc 0.870593 | val loss 278.598 | val acc 0.88151
epoch 4 238ms | loss 1515.37 | acc 0.880809 | val loss 265.987 | val acc 0.888522
epoch 5 239ms | loss 1439.92 | acc 0.886301 | val loss 256.641 | val acc 0.89353
epoch 6 239ms | loss 1385.83 | acc 0.891026 | val loss 249.132 | val acc 0.898438
epoch 7 239ms | loss 1344.56 | acc 0.895065 | val loss 243.329 | val acc 0.901142
epoch 8 239ms | loss 1309.36 | acc 0.898237 | val loss 238.113 | val acc 0.903446
epoch 9 239ms | loss 1278.04 | acc 0.901159 | val loss 233.016 | val acc 0.90615
finished training, total time = 2417 ms
```
#### Optimized NN implementation

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
epoch 0 144ms | loss 5960.6 | acc 0.49379 | val loss 531.156 | val acc 0.766627
epoch 1 124ms | loss 2440.52 | acc 0.810196 | val loss 347.35 | val acc 0.848057
epoch 2 123ms | loss 1854.34 | acc 0.851345 | val loss 300.709 | val acc 0.870092
epoch 3 123ms | loss 1636.67 | acc 0.870393 | val loss 277.957 | val acc 0.880208
epoch 4 123ms | loss 1517.52 | acc 0.880626 | val loss 264.129 | val acc 0.887821
epoch 5 123ms | loss 1439.88 | acc 0.886936 | val loss 255.306 | val acc 0.892929
epoch 6 123ms | loss 1385.67 | acc 0.890875 | val loss 248.474 | val acc 0.897736
epoch 7 123ms | loss 1343.11 | acc 0.895015 | val loss 242.408 | val acc 0.901342
epoch 8 123ms | loss 1307.31 | acc 0.898738 | val loss 236.967 | val acc 0.905048
epoch 9 123ms | loss 1276.57 | acc 0.901542 | val loss 231.689 | val acc 0.908153
finished training, total time = 1252 ms
```

### References

- https://siboehm.com/articles/22/CUDA-MMM
- https://0mean1sigma.com/xgemm/
- https://www.youtube.com/playlist?list=PL5XwKDZZlwaY7t0M5OLprpkJUIrF8Lc9j
- https://www.youtube.com/watch?v=GetaI7KhbzM


### Nsight info
#### FMA
```sh
$ ncu ./mm.o
void smem_coal_unrolled_softmax<16>(int, int, float *, float *) (1024, 1, 1)x(1, 16, 1), Context 1, Stream 7, Device 0, CC 8.9
  Section: GPU Speed Of Light Throughput
  ----------------------- ----------- ------------
  Metric Name             Metric Unit Metric Value
  ----------------------- ----------- ------------
  DRAM Frequency                  Ghz         7.98
  SM Frequency                    Ghz         1.87
  Elapsed Cycles                cycle       94,083
  Memory Throughput                 %        43.71
  DRAM Throughput                   %        43.71
  Duration                         us        50.24
  L1/TEX Cache Throughput           %        32.99
  L2 Cache Throughput               %        21.22
  SM Active Cycles              cycle    88,773.05
  Compute (SM) Throughput           %        31.19
  ----------------------- ----------- ------------
  Section: Occupancy
  ------------------------------- ----------- ------------
  Metric Name                     Metric Unit Metric Value
  ------------------------------- ----------- ------------
  Block Limit SM                        block           24
  Block Limit Registers                 block           48
  Block Limit Shared Mem                block           28
  Block Limit Warps                     block           48
  Theoretical Active Warps per SM        warp           24
  Theoretical Occupancy                     %           50
  Achieved Occupancy                        %        44.73
  Achieved Active Warps Per SM           warp        21.47
  ------------------------------- ----------- ------------
```
#### Softmax
```sh
$ ncu ./activation.o
void mm4<64, 64, 8, 8>(int, int, int, const float *, const float *, float *, float *) (16, 16, 1)x(512, 1, 1), Context 1, Stream 7, Device 0, CC 8.9
  Section: GPU Speed Of Light Throughput
  ----------------------- ----------- ------------
  Metric Name             Metric Unit Metric Value
  ----------------------- ----------- ------------
  DRAM Frequency                  Ghz         7.99
  SM Frequency                    Ghz         1.87
  Elapsed Cycles                cycle    17,88,903
  Memory Throughput                 %        82.68
  DRAM Throughput                   %         4.59
  Duration                         us       954.50
  L1/TEX Cache Throughput           %        83.99
  L2 Cache Throughput               %        12.73
  SM Active Cycles              cycle 17,56,580.80
  Compute (SM) Throughput           %        82.68
  ----------------------- ----------- ------------

  Section: Occupancy
  ------------------------------- ----------- ------------
  Metric Name                     Metric Unit Metric Value
  ------------------------------- ----------- ------------
  Block Limit SM                        block           24
  Block Limit Registers                 block            2
  Block Limit Shared Mem                block            6
  Block Limit Warps                     block            3
  Theoretical Active Warps per SM        warp           32
  Theoretical Occupancy                     %        66.67
  Achieved Occupancy                        %        64.28
  Achieved Active Warps Per SM           warp        30.86
  ------------------------------- ----------- ------------
```
#### NN
##### unoptimized
```sh
$ nsys profiler --stats=true ./nn.o
Time (%)  Total Time (ns)  Instances   Avg (ns)    Med (ns)   Min (ns)  Max (ns)   StdDev (ns)                                   Name                                 
 --------  ---------------  ---------  ----------  ----------  --------  ---------  -----------  ----------------------------------------------------------------------
     67.4   1,34,78,67,971      8,190  1,64,574.8  1,13,024.0    29,728   4,18,846   1,34,706.8  linear_forward(int, int, int, float *, float *, float *, float *)     
     24.1     48,18,82,046      7,020    68,644.2    49,408.0     9,727   1,72,832     56,765.0  update_layer(int, int, int, float, float *, float *, float *, float *)
      6.2     12,39,93,574      4,680    26,494.4    25,760.0     3,136     59,807     23,202.9  linear_backward(int, int, int, float *, float *, float *, float *)    
      0.8      1,65,47,950      2,730     6,061.5     5,984.0     5,824      7,424        316.5  cross_entropy(int, int, float *, float *, float *)                    
      0.5        92,80,990      4,680     1,983.1     1,952.0     1,888      2,400        103.3  relu_backwards(int, int, float *, float *, float *)                   
      0.5        91,18,511      5,460     1,670.1     1,664.0     1,568      1,984         81.6  relu(int, int, float *, float *)                                      
      0.3        54,29,029          6  9,04,838.2    50,575.5       960  40,42,572  16,17,835.9  init_rand(int, int, float *)                                          
      0.2        36,34,095      2,730     1,331.2     1,312.0     1,279      1,600         66.8  softmax(int, int, float *, float *)                                   
      0.1        20,75,030      2,340       886.8       864.0       832      1,088         45.4  cross_entropy_backwards(int, int, float *, float *, float *) 
```


##### optimized
```sh
$ nsys profiler --stats=true ./nn.o
 Time (%)  Total Time (ns)  Instances   Avg (ns)   Med (ns)  Min (ns)  Max (ns)   StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ----------  --------  --------  ---------  -----------  ----------------------------------------------------------------------------------------------------
     54.3      5,69,59,125        702    81,138.4  58,815.0    11,744   1,72,799     67,150.7  update_layer(int, int, int, float, float *, float *, float *, float *)                              
     21.8      2,29,11,401        546    41,962.3  41,888.0    21,376     81,727     20,557.1  void linear_relu_1dBlocking<(int)64, (int)64, (int)8, (int)8>(int, int, int, const float *, const f…
     14.1      1,47,78,862        468    31,578.8  31,280.0     4,000     59,935     27,440.5  linear_backward_fused(int, int, int, float *, float *, float *, float *, float *)                   
      5.2        54,53,060          6  9,08,843.3  50,543.5       960  40,34,603  16,16,310.4  init_rand(int, int, float *)                                                                        
      2.0        20,46,642        273     7,496.9   7,488.0     7,424      7,552         20.0  mm2(unsigned int, unsigned int, unsigned int, unsigned int, float *, float *, float *, float *)     
      1.9        19,62,238        273     7,187.7   7,200.0     7,168      7,424         21.8  cross_entropy(int, int, float *, float *, float *)                                                  
      0.5         5,70,589        273     2,090.1   2,080.0     2,047      2,176         17.0  void smem_coal_unrolled_softmax<(int)16>(int, int, float *, float *)                                
      0.2         2,42,816        234     1,037.7   1,024.0     1,024      1,056         15.9  cross_entropy_backwards(int, int, float *, float *, float *)     
```
#### My device
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


