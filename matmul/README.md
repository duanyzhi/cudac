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
| mm naive                 | 0.54   | 9.32    | 75       | 524.57ms/ 261GFLOPS |
| Global memory coalescing | 0.38   | 2.56    | 19.5     | 154/889GFLOPS       |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
|                          |        |         |          |                     |
| mm cublas                | 0.16   | 0.6     | 1.9      | 12.12ms/11000GFLOPS |

time is ms without memory copy  ms/GFLOPS



1GFLOPS = 10 ^ 9 FLOPS

1TFLOPS = 1000 GFLOPS



说明:

A: [M, N], B[N, K], C[M, K]

C = A * B

compute flops = M * (N * + (N - 1)) * K

load memory from cpu to gpu(float type):

gpu memory = M * N * 4B + N * K * 4B + M * K * 4B



4096 compute flops  = 4096* (4096 + 4095) * 4096 = 137 GFLOP

4096 gpu memory = 4096 * 4096 * 4 + 4096 * 4096 * 4 + 4096 * 4096 *4 = 192 M



RTX 3060TI GPU 理论数据:

GPU : 8G, bandwidth = 448 GB/s = 448 * 1024 / 1000 = 458 M / ms

float32 compute throughput = 16.2 TFLOPS = 16200 GFLOPS



理论最快拷贝时间 =  192 / 458 = 0.4 ms.  (实际测试需要20 ms+.)

理论最快计算时间 = 137 / 16200 * 1000 = 8.45 ms  实际

理论计算时间远远大于数据拷贝时间，所以当前算子应该是compute-bound计算瓶颈的。



mm naive 4096 实际计算GFLOPS =  137 / 524 * 1000 = 261 GFLOPS

cublas 409 size GFLOPS = 137 / 12 * 1000 = 11 TFLOPS

可以看出cubls的计算能力基本快达到理论上限。



optimi method 1:   use shared memory