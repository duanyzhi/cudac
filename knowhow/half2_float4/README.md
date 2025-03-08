float4 ncu:
Memory Throughput [%]	0.71
Duration [us]	20.03

float ncu:
Memory Throughput [%]	0.27
Duration [us]	52.10

只使用float4数据结构，运行快了一倍多，memory throughput提升了很多,近4倍，因为本来需要读取4次的数据结构现在只需要读取一次，然后就可以进行4个数据的计算了。
