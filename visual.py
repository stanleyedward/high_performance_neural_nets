import matplotlib.pyplot as plt
import numpy as np

#manually noted values

matrix_sizes = [128, 256, 512, 1024, 2048, 4096]

mm_gflops = {"naive": [95.26, 149.63, 154.02, 158.08, 158.49, 195.17],
             "gmem": [308.40, 564.36, 624.01, 640.70, 644.52, 605.37],
             "smem": [206.74, 642.12, 808.07, 863.44, 880.88, 754.46],
             "1dBlocking": [131.33, 811.59, 1706.74, 2405.85, 2654.92, 2848.83]}

mm_times = {"naive": [0.0440, 0.2243, 1.7428, 13.5844, 108.3955, 704.1966],
             "gmem": [0.0136, 0.0595, 0.4302, 3.3518, 26.6551, 227.0318],
             "smem": [0.0203, 0.0523, 0.3322, 2.4871, 19.5030, 182.1685],
             "1dBlocking": [0.0319, 0.0413, 0.1573, 0.8926, 6.4709, 48.2440]}

softmax_times = {"naive": [0.0236, 0.0696, 0.4557, 3.4499, 28.3904, 220.0117],
                 "smem": [0.0451, 0.0410, 0.0543, 0.0788, 0.2796, 0.9288],
                 "unrolled_smem": [0.0143, 0.0151, 0.0191, 0.0460, 0.2488, 0.8243]}

softmax_gflops = {"naive": [5.5652, 7.5294, 4.6022, 2.4316, 1.1819, 0.6100],
                 "smem": [2.5455, 11.2000, 33.8113, 93.0909, 105.0256, 126.4476],
                 "unrolled_smem": [8.6154, 30.3087, 95.8930, 159.6214, 117.9918, 142.4696]}

if __name__ == '__main__':
#make a plot for matrix multiplication gflops with all methods
    fig, ax = plt.subplots()
    for method in mm_gflops:
        ax.plot(matrix_sizes, mm_gflops[method], label=method)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('GFLOPS')
    ax.set_title('Matrix Multiplication GFLOPS')
    ax.legend()
    plt.show()
#make a plot for matrix multiplication times with all methods
    fig, ax = plt.subplots()
    for method in mm_times:
        ax.plot(matrix_sizes, mm_times[method], label=method)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Matrix Multiplication Time')
    ax.legend()
    plt.show()
    
#make a plot for softmax gflops with all methods
    fig, ax = plt.subplots()
    for method in softmax_gflops:
        ax.plot(matrix_sizes, softmax_gflops[method], label=method)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('GFLOPS')
    ax.set_title('Softmax GFLOPS')
    ax.legend()
    plt.show()
    
#make a plot for softmax times with all methods in log scale
    fig, ax = plt.subplots()
    for method in softmax_times:
        ax.plot(matrix_sizes, softmax_times[method], label=method)
    ax.set_xlabel('Matrix Size')
    ax.set_ylabel('Time (s)')
    ax.set_title('Softmax Time')
    ax.legend()
    ax.set_yscale('log')
    plt.show()

