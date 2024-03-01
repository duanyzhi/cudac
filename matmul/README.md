# Matmul

## build and run

```shell
nvcc main.cc matmul_cpu.cc *.cu -o main -lcublas
./main 100 100 100  # M N K
```

install cublas if build with matmul_cublas.cu

## performance

| methods\M=N=K            | 500    | 1024    | 2048     | 4096                |
| ------------------------ | ------ | ------- | -------- | ------------------- |
| cpu                      | 220918 | 2361435 | 35805503 | 693826597           |
| mm naive                 | 0.618400   | 9.585    | 72.38       | 474.32ms/ 289 GFLOPS |
| Global memory coalescing | 0.20960   | 0.854    | 6.080     | 44.998/3 TFLOPS       |
| submatrix with shared memory | 0.178656 |  0.656      |   4.614    |  29.364 /4.67 TFLOPS    |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
| mm cublas                | 0.034406   | 0.158     | 1.183      | 8.818ms/ 113 TFLOPS |

time is ms without memory copy  ms/GFLOPS, notice: shoule waraup for cuda

## 计算GFLOPS
#### 基础信息
1GFLOPS = 10 ^ 9 FLOPS

1TFLOPS = 1000 GFLOPS

RTX 3060TI GPU 理论数据:

GPU : 8G, bandwidth = 448 GB/s = 448 * 1024 / 1000 = 458 M / ms

float32 compute throughput = 16.2 TFLOPS = 16200 GFLOPS

A100-SXM4-80GB理论数据:
Memory bandwidth: 2039 GB/s = 2039 M/ms
fp32 compute throughput: 156 TFLOPs = 156000 GFLOPS
bandwidth指理论上单位时间可以传输的数据量。

### 理论计算
说明:

A: [M, N], B[N, K], C[M, K]

C = A * B

compute flops = M * (N * + (N - 1)) * K

load memory from cpu to gpu(float type):

gpu memory = M * N * 4B + N * K * 4B + M * K * 4B



4096 compute flops  = 4096* (4096 + 4095) * 4096 = 137 GFLOP

4096 gpu memory = 4096 * 4096 * 4 + 4096 * 4096 * 4 + 4096 * 4096 *4 = 192 M

则A100上理论计算数据如下:
理论最快拷贝时间 =  192 / 2039 = 0.09416 ms.即192M的数据需要0.094ms的时间完成cpu到gpu的拷贝。

理论最快计算时间 = 137 / 156000 * 1000 = 0.878 ms  实际

理论计算时间远远大于数据拷贝时间，所以当前算子应该是compute-bound计算瓶颈的。


实测部分数据throughput分析如下:
mm naive 4096 实际计算GFLOPS =  137 / 474 * 1000 = 289 GFLOPS

cublas 409 size GFLOPS = 137 / 8.818 * 1000 = 113.4 TFLOPS

可以看出A100上理论fp32最大throughput为156TFLOPS,而cubls的计算能力基本快达到理论上限为113TFLOPS。

## 算子优化
### naive方法
如图，矩阵A: [M, K], B[K, N],矩阵乘结果C[M, N]。最直接的想法就是用A的一行乘上B的一列然后得到C中一个元素。这样遍历完所有的A的行和B的所有列就可以计算出所有C中的元素。kernel上启一个2纬的grid和block。一个x线程取A的一行元素计算，一个y线程取B的一列计算。
![naive](images/matmul_naive.png)
这种计算的实测throughput为289GFLOPS，只有理论上限的1%左右，利用率特别低。主要原因是：
1. 重复的内存读取
假设BLOCK_SIZE是16,则一个dimBlock里就有16 * 16个独立线程。thread id是(0, 0), (0, 1), ....(0, 15), (1, 0), ...(15, 15)。对线程组合(0, 0),naive的读取方式是用读取A的第0行和B的第0列。对thread(0, 1)则读取A的0行和B的1列。可以看出一个dimBlock里就对A的每一行都重复读取了16次。B也是。从global memory到shared memory的读取是很慢的。这种重复读取造成了极大的消耗。

2. 内存读取线程不连续
   内存读取不连续的含义是相邻的线程在读取global memory数据的时候不是读取的相邻的位置。还是以一个dimBlock的线程为例子，naive方法中相邻的线程(0,0),(0,1)...对应到A中就是第一个线程读取第一行，第二个线程读取第二行。假设两个线程是同时并行读取的，则读取的数据就是A[0,0]和A[1,0]两个元素，而这两个元素在A的内存上不是相邻的。一般A的存储肯定是A[0,0]和A[0,1]是相邻的。gpu在设计时从global memory中读取数据是遵循SIMD的，即单个指令执行多个数据，为了加速硬件读取的效率，从硬件上读取数据时一条读取指令会将返回一段内存的数据而不是只返回一个位置的数据。所以如果是按照A[0,0],A[1,0]这种读取的话，读取16个数据则需要16次硬件指令才能获取所有的数据，而且造成了很多数据的读取浪费。但是如果按照A[0,0],A[0,1]这种顺序读取数据的话由于是连续的，可能只需要一次的硬件指令就可以满足把16个数据读取出来。具体查看官网gpu global memory coalescing的介绍。
### global memory coalescing
根据naive的限制条件中内存读取不连续问题，可以将内存读取改成连续的版本。即相邻的线程让其读取的buffer是连续的。改变dimGrid和dimBlock如下，第一个dimBlock为(假设BLOCK_SIZE=16):
![global_memory_coalescing](images/global_memory_coalescing.png)
改变读取行列顺序:
```cuda
  const int row = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
  const int col = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);
```
对第一个dimBlock来说，前面16个线程组合(0,0)对应读取A的第0行的所有元素和B的第0列的所有元素。线程(0,1)对应读取A的第0行和B的第1列。则第一个dimBlock计算完则算完了C的第0行的前16个元素。这里对A而言在kernel内读取方式为:
```cuda
  for (int i = 0; i < K; ++i) {
    sum_c += A[row * K + i] *
             B[col + i * N];
  }
```
可以看出A的索引读取是连续的。对B而言，相邻线程(0,0)和(0,1)分别从B的[0,0]和B[0,1]开始读取，这两个地址也是连续的，即对B而要一个dimBlock里16个线程第一次读取的所有数据是连续的。在一个block内线程是按照warp大小个线程来操作的，即warp个线程会同时异步的读取数据。实测global memory coalescing方法可以达到44 TFLOPS的吞吐。