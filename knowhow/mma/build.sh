# need arch
nvcc -arch=sm_89 mma_main.cu -o main

# nvcc  -ptx -arch=sm_80 bf16_naive.cu  -o bf16_naive.ptx
