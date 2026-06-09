
感觉现在XSAI 的AME 还是更接近intel 附加矩阵指令扩展的形式

GPU/TPU 是显示DMA + scratchpad + 编辑器调度 的大加速器
intel AMX/ Arm SME2/ Apple AMX 是CPU 内部tile/matrix 扩展 的形式，相对软件栈更简单一些
（AMD 更多是AVX-512, 类似IME 的形式，走大向量的形式）
述的是：

```
普通 CPU 指令流  ↓load 到 tileA / tileB 寄存器  ↓矩阵乘累加  ↓store 回内存
```
这非常像 Intel AMX：
```
tileconfig
tileload
tdpbf16ps / tdpbssd
tilestore
```
本地推理已经有人在用AMX, 主要收益在prefill, 大batch GEMM 上
生态大概是：

```
PyTorch / TensorFlow / ONNX Runtime / OpenVINO
        ↓
oneDNN / IPEX / vendor optimized kernels
        ↓
AMX microkernel
        ↓
tileload / tdpbf16ps / tilestore
```
也就是pytorch 后端会调用到Intel 自己写的oneDNN 使用AMX kernel ， 调用到AMX 指令来加速
llama.cpp 也是支持AMX 后端了


https://www.intel.com/content/www/us/en/developer/articles/code-sample/advanced-matrix-extensions-intrinsics-functions.html
这是intel AMX 介绍，介绍的很详细了。

核心也是 添加 一组二维寄存器tile,  一个Tile 矩阵乘法TMUL 加速器
eg: 8个tile, 每个支持16行* 64 字节= 1024B = 1KB,  8个tile 就是8KB

1. 先配置tile , (各个硬件tile 最大数量和尺寸不同，通过cpuid 获得)
	1. XTILECFG ：tile 配置信息 （64byte 的struct)
	2. XTILEDATA ：tile 寄存器里的真实数据
2. 然后TMUL 指令执行矩阵乘法，支持INT8, BF16
3. 当OS进程切换时候，默认不切换8KB 的额外架构状态，否则太亏了

eg： 做2个16* 64 INT8 的A，B 矩阵乘法，结果累加都16 * 16 的INT32 矩阵C 中
乘累加没说几拍，可能几十拍的量级吧

### packing
这里有一个重要概念是packing

核心就是，如果B 的存储还是按照 64 * 16 的形式存储的话，那么硬件每次计算C 的一个算数，都需要扫描读取A 的一行，B 的一列，
硬件上矩阵A 是row-major 存储，相当于行1 之后存储行2, 行3， 那么扫描读取A 的一行都是连续内存很好
如果B 也是row-major 存储，那么每次读B 的一列内容都是不连续的形式，地址0 , 16，32，48 的形式，那这样局部性太差了。
所以要把逻辑上的B 存储，packing 重新打包存储一下，让原本地址0，16，32，48 的是个元素，存储到地址0，1，2，3 的4个位子上，这样就能一次性读取了。
packing 像转置，但不是普通转置，而是 **blocked + interleaved packing**。
最终存储形式仍然是 16 * 64 的形式
注意这个packing 通常是软件来做
参考这个互动canvas
https://chatgpt.com/canvas/shared/6a0ee1c554188191ac923d3aa718b201

那么packing 每次做开销大吗？
1. 如果B 矩阵是weight 权重矩阵，那么只用packing 一次之后复用
2. B 是动态activation 输入，就比较慢了
所以尽可能让权重存储在B 矩阵