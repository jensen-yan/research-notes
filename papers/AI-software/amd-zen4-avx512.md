https://www.amd.com/en/blogs/2026/understanding-avx-512---validating-usage-on-amd-epyc-.html

https://docs.amd.com/api/khub/documents/NgOfoW49HKdzztTeLBbekA/content
这是9004 zen4 芯片的文档

感觉这个和intel amx 相比，这个是添加很长的向量，有点像riscv AME vs IME 的争论，我得仔细学习一下

向量宽度不断扩展，
1. SSE 128,  XMM
2. AVX2 256, YMM
3. AVX-512 bit, ZMM
注意zen4 和之前的CPU 都是使用2个256bit vector 来执行AVX512, 虽然lscpu 看也支持avx512f
只有zen5 turin 才支持完整一拍512bit
(当前大机房服务器都是zen4 的，后续数据都是组合2个256 vector 数据)
在node037 上测试，
对于大机房node037

### 理论算力
一拍256 bit = 16 FP64 FLOPS
```
每 core 每 cycle: 16 FP64 FLOPs
= 2 条 256-bit FMA/cycle
= 2 * 4 个 double * 2 FLOPs(FMA mul+add)
```
( 256 = 4 * FP64 ,  一拍同时做mul add, 还能乘2，一拍能有2条256bit FMA  = 16 FP64 = 32 FP32)
192core * 3.6GHz * 16 FLOPS/cycle = 11  TFLOPS.   大约7到11 TFLOPS的纯算力
单核 1core * 3.6GHz * 16 FLOPS =  58GFLOPS

FP 32 单核106GFLOPS

### 实际算力测试
写一个存储计算密集型的测试
**关键结果**
```
单核：
AVX2    FP64:   58.19 GFLOPS
AVX512  FP64:   58.15 GFLOPS
AVX2    FP32:  116.31 GFLOPS
AVX512  FP32:  116.47 GFLOPS

多核：
AVX512 FP64,  96 threads:  4.39 TFLOPS
AVX512 FP64, 192 threads:  8.93 TFLOPS
AVX512 FP64, 384 threads:  8.80 TFLOPS

AVX512 FP32, 192 threads: 17.24 TFLOPS
```
1. AVX2, AVX512 峰值算力相同
2. 实际只有192 core, SMT 不会增加算力


测试了https://docs.rs/crate/gemm-benchmark/latest
之后再用openblas 测试下吧
默认情况下，使用 256 x 256 的矩阵对 `sgemm` 进行基准测试，迭代 1,000 次，使用 1 个线程。矩阵维度（ `-d` ）、迭代次数（ `-i` ）和线程数（ `-t` ）可以通过命令行标志设置。例如：

```shell
$ gemm-benchmark -d 1024 -i 2000 -t 4
```

SGEMM 表示single FP32, DGEMM 表示double FP64
```
DGEMM，OPENBLAS_NUM_THREADS=1，靠 -t 跑多个独立 GEMM：

N=1024, i=20
t=1      55.10 GFLOPS
t=2     108.04 GFLOPS
t=4     217.11 GFLOPS
t=8     415.71 GFLOPS
t=16    841.84 GFLOPS
t=32   1646.95 GFLOPS
t=64   1735.05 GFLOPS
t=96   3132.46 GFLOPS
t=192  4021.80 GFLOPS
```
![[Pasted image 20260526160900.png]]
上图是从1core to 192 core 的算力增长
测试发现，
1. FP64 理论最大能到8.9TFLOPS  ， GEMM 能到 4.55 TFLOPS, 接近一半了
2. FP32 理论能到17TFLOPS,  SGEMM 能到8.9
3. 单核情况下，55.10 GFLOPS vs 58.19 GFLOPS其实差异不大
4. 192cores 原先 4.5-5.1 TFLOPS 偏低，至少一部分是调度/SMT/迁移/benchmark 形态问题；绑物理核后能到 6.3 TFLOPS，约为手写 FP64 peak 8.93 TFLOPS 的 71%。

注意这里的GEMM 算数强度很高，N=1024, 计算量 = 2 * N^3 = 2.1BFLOPS
矩阵大小3 * 1024 * 1024 * 8 = 24MB, OI = 计算 / 存储 = 85 FLOPS/byte
并且openblas 能比较好做好blocking 和packing
而服务器192core, L3 = 2.3GB = 24 个L3 slice, 每个96MB 还挺大的。

**openblas优化思路总结**  
最核心目标是提高 arithmetic intensity：一次把 A/B 搬进 cache/register 后，多做很多 FMA。
徐岩也做了一个分tile 类似openblas 的GEMM, 效率能到openblas 70% 挺高了。

从外到内是：
```
大矩阵 blocking: 控制 cache 工作集
packing: 改成连续、规整、microkernel 友好的布局
microkernel: 用寄存器保存 C 小块，连续 FMA，最后再写回
prefetch/unroll: 避免 load 延迟打断 FMA
threading/NUMA: 多核时让每个线程处理独立 C block，减少共享和远端内存
```
相当于openblas 其实挺能感知cache size 的
测试了N = 1024 一直到8192， 随着矩阵算力增大，单核基本上都没有太多算力下降