规格
```
M4 SoC =
  CPU cores （ 4P + 6E)
    普通标量 + 128-bit NEON/AdvSIMD
    CPU 侧 ML accelerators
    M4 上可用的 SME / AMX-like 矩阵路径
  GPU
    10-core Apple GPU
    Metal / MPS / MLX 可用
  Neural Engine
    16-core NPU, up to 38 TOPS
  Media Engine
    H.264 / HEVC / ProRes / AV1 等视频专用单元
  统一内存
    base M4 Mac mini: 120 GB/s
```

https://arxiv.org/abs/2409.18779 这个论文还提到M4 第一个支持Arm SME 矩阵扩展，但是具体规格没有完全公开

### 理论算力计算
向量算力
（kmhv3 vector 是一拍处理2条128bit fp)
zen4 vector，一拍能处理2条256 bit
M4 P-core 一拍能有4条128bit FMLA(4个FP/SIMD unit), 算力和zen4 相同
FP32, 一拍 为 32 FLOPS/cycle, 单核 4.4GHz 为140 GFLOPS， 实测 100GFLOPS（频率在1G 到4G波动）
FP64 实测为 50GFLOPS

SME 矩阵算力
不是每core 一份，而是全chip 大约2个共享SME，论文逆向结果
```
单 P-core Neon FMLA:
FP32 113 GFLOPS
FP64 56 GFLOPS

单 P-core SME FMOPA:
FP32 2009 GFLOPS
FP64 503 GFLOPS

另一个E-core SME
FP32 357 GFLOPS
```
大约是向量的20倍
vector 是4 * 128 = 512bit
矩阵 
1. FP32: 16 * 16 * 2 = 512 FLOPS， 4GHz 大约2000 GFLOPS
2. FP64: 8 * 8 * 2 = 128 FLOPS

### 论文逆向
矩阵扩展 ZA matrix array. 大小4KB, 4个16* 16 tile, 512FLOPS。  
FMOPA 指令=两个向量做外积，累加到ZA tile 上 （似乎很像riscv VME)
只对FP32 很友好，I8/BF16/FP64 都提升不大。
https://developer.arm.com/community/arm-community-blogs/b/architectures-and-processors-blog/posts/arm-scalable-matrix-extension-introduction
这里是ARM SME 介绍


### 实际算力测试

https://docs.rs/crate/gemm-benchmark/latest
使用同一个来测试，注意这里有两种后端
1. accelerate: 能利用到M4 的SME 扩展（大概率只支持FP32)
2. openblas：也能用到SME 但是优化没那么好
两个数据类型
3. FP32 SGEMM: 能使用矩阵扩展
4. FP64 DGEMM: 只能使用neon 128

![Pasted image 20260526181922](../../assets/Pasted%20image%2020260526181922.png)
![Pasted image 20260526181935](../../assets/Pasted%20image%2020260526181935.png)

能有几个结论，
1. accelerate 分tile 机制和优化会比openblas 做的更好，算力高快一倍
2. FP32 比FP64 高挺多(高4倍），说明SME 只支持FP32
3. 多核下openblas DGEMM 算力线性增加，从
	1. 单核 50 GFLOPS ，和zen 4 单核接近
	2. 双核100 GFLOPS
	3. 4核 156 GFLOPS
	4. 更多核是P 核，没那么线性了
