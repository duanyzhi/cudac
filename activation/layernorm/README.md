flash_layer_norm_v0.cu

问题:
Compute (SM) Throughput [%]	0.62
Duration [us]	26.82
Memory Throughput [%]	0.12
Elapsed Cycles [cycle]	59836
L1/TEX Cache Throughput [%]	7.83
SM Active Cycles [cycle]	447.63
L2 Cache Throughput [%]	0.11
SM Frequency [Ghz]	2.23
DRAM Throughput [%]	0.12
DRAM Frequency [Ghz]	10.22

Memory Throughput太小了, 可以通过使用float4数据结构来提升memory throughput
