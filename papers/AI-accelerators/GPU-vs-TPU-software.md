# GPU-vs-TPU-software

## 基本信息

- 论文/文章：Part 2: From SIMT to Systolic A Kernel Author's Field Report
- 链接：[https://x.com/MainzOnX/article/2044804854020006223](https://x.com/MainzOnX/article/2044804854020006223)
- 作者：Adam Mainz
- 日期：2026.apr.16
- 领域：
- 标签：#paper 

---

## 1. 一句话总结

这篇文章最核心想说的是：

> 核心是在比较 **NVIDIA GPU 软件栈** 和 **Google TPU/JAX/Pallas 软件栈**，作者明显站在 TPU/Pallas 一边。
上一篇[GPU-vs-TPU-hardware](GPU-vs-TPU-hardware.html) 介绍了硬件差异
 
作者认为：**GPU/CUDA/Triton 的优势在生态和低门槛，但 TPU/JAX/Pallas 的优势在“组合性、编译器托管复杂度、确定性、规模化训练效率”，更适合大规模生产 ML workload。**


## 2. 原文关键内容

### 背景问题

#### tile 定义
**tile** 理解成：Tile不是“矩阵”，而是“固定形状的数据块/子张量”

> 从大 tensor / matrix 里切出来的一小块，通常大小在编译期或 kernel meta-parameter 里固定，比如 `128×128`、`128×64`、`64×256`。

tile 正好卡在一个很舒服的抽象层次：
太高层：整个 matmul / attention / conv
    优点：表达简单
    缺点：很难做特殊融合和局部优化

tile 层：一个固定大小的数据块
    优点：能表达数据复用、片上缓存、Tensor Core/MXU 使用
    缺点：还要调 block size、pipeline

太底层：thread / lane / warp / instruction
    优点：控制力最强
    缺点：程序员负担极大

现在 AI workload 里面大量计算都是 **矩阵乘、attention、norm、activation、reduction**，这些操作的数据访问很规则。

注意不同层次tile 大小不同
一个软件 tile 可能是 `128×128` 的 C block，但底层会被拆成很多个 Tensor Core MMA 指令。以 TPU/Pallas 为例，`BlockSpec((128, K), ...)` 描述的是 HBM 到 VMEM 的 tile

#### nvidia 技术栈
- ![](../../assets/Pasted%20image%2020260513162026.png)
- PTX:虚拟ISA, 不用管
- CUDA: 面向程序员API： 管理thread, warps, blocks, memory 能直接映射到硬件 
- CUTLASS: C++ 矩阵乘法库
- cuDNN： 闭源的，矩阵库， gemm, conv2d
- tirton: 开源的，基于tile 的DSL, 只需要描写tile, 编译器自动处理加载、共享内存暂存、张量核心调度
- NCCL: 集体分布式库，万卡采用
- 框架：pytorch. 2 模式：
	及时模式eager
	torch.compile: 捕获计算图并为热点路径生成融合的内核

>数一数有七层，其特点在于大多数有趣的选择点都位于中间层，而非顶层或底层。大多数生产级机器学习代码位于 PyTorch 层，大多数生产级内核位于 Triton 层，大多数追求极致性能的工作位于 CUTLASS 或 CUDA 层。而 CUTLASS 以下的所有层，在出问题之前你基本可以当它们不存在。

```python
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)  # 乘累加
        a_ptrs += BLOCK_K * stride_ak # 机选下一个地址指针
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))  # 写入C
```
tirton 代码参考，还是挺简单的，本质上

在核心源码中，这些细节完全不可见。Triton 的核心理念在于将“瓦片”作为抽象单元，由编译器自行决定如何在目标 GPU 上处理它。在 A100 上，同一份核心代码会生成`cp.async`（而非 TMA）、`mma.sync`（而非 WGMMA），以及传统的`__syncthreads()`模式而非 mbarrier。相同的源码，不同的代码生成。这正是 Triton 的职责所在。

#### google stack


![](../../assets/Pasted%20image%2020260513162704.png)

- JAX： 提供函数变换， jit 编译，vmap 批量操作，grad 求导，pmap 分布到多设备，可叠加
- Pallas: 同triton, 写内核的，面向tile
- Mosaic: TPU 后端编译器，Pallas -> TPU 机器码
* XLA、HLO 和 StableHLO 
	* XLA 是编译器。
	* HLO（高层操作）是其中间表示（IR）。当你对某个函数使用 `jax.jit` 时，该函数会被跟踪成 JAXPR，然后降级为 
	* StableHLO（一种可移植的 HLO 方言），位于 JAX 和 XLA 之间的可移植 IR
	影响性能-融合：将 HLO 操作的链条压缩成少量融合核
	在 TPU 上意味着一个单一核将中间结果保留在向量内存（VMEM）或寄存器中，而无需通过高带宽内存（HBM）进行往返。

![](../../assets/Pasted%20image%2020260513173928.png)

- StableHLO is the portability boundary.  
    StableHLO 是可移植性边界。
    
- XLA HLO is where fusion happens.  
    XLA HLO 是融合发生的地方。
    
- Mosaic TOP is where Pallas kernel bodies get translated into TPU-native ops.  
    Mosaic TOP 是将 Pallas 内核主体转换为 TPU 原生操作的地方。
    
- Mosaic TPU is where layout decisions get locked in (VMEM vs register, tile shape, DMA scheduling).  
    Mosaic TPU 是布局决策被锁定的地方（VMEM 与寄存器、瓦片形状、DMA 调度）。
    
- LLO is where instructions get packed into VLIW bundles. If you ever need to read your compiled output to understand why a kernel is slow, LLO is what you'll be reading.  
    LLO 是将指令打包成 VLIW 束的地方。如果你需要阅读编译输出以了解内核为何运行缓慢，那么你读的就是 LLO。

```python
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, w_ref, o_ref):
    o_ref[...] = jnp.dot(x_ref[...], w_ref[...]).astype(o_ref.dtype)
    # 矩阵点击 o = x * w


def matmul(x, w, block_m=128, block_n=128):
    M, K = x.shape
    _, N = w.shape
    return pl.pallas_call(
        matmul_kernel, # 执行矩阵乘法
        grid=(M // block_m, N // block_n), # 分块
        in_specs=[
        # A 的[128,K] 横条 * B【K,128】 竖条
            pl.BlockSpec((block_m, K), lambda i, j: (i, 0)), # 分块规模
            pl.BlockSpec((K, block_n), lambda i, j: (0, j)),
        ], # 从HBM 加载对应大小到VMEM 中
        # 写回C 到HBM 中
        out_specs=pl.BlockSpec((block_m, block_n), lambda i, j: (i, j)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        # 输出规模和大小
    )(x, w)
```
`pl.BlockSpec` 告诉 Pallas 如何将来自 HBM 的输入和输出分块为驻留在 VMEM 中的 `Ref` 对象。块规格包含两部分：
1. 块形状
2. 从网格坐标到块坐标的索引映射。
如果你的网格是 `(M // 128, N // 128)`，你的输入是 `(M, K)`，那么块规格 `BlockSpec((128, K), lambda i, j: (i, 0))` 表示“给我一个 128×K 的块，由网格的第一个轴索引”。Mosaic 会获取该规格，并生成 HBM↔VMEM 的 DMA 调度。

关键之处在于，这里没有出现：没有 TMA 描述符、没有 mbarrier 等待状态、没有 WGMMA 操作数暂存。Mosaic 自动生成了所有这些。作者只编写了数学逻辑。

> 如果 VMEM 装不下，不是硬件自动替换，而是你/编译器要换更小的 block，尤其是把 K 维也 tile 掉；否则可能编译失败、资源超限、或者生成非常低效的代码。


#### GPU vs TPU 差异

4点差异
1. composition 构图
google: 内核编写层pallas 位于框架转换层JAX 以下： 写一个内核，外层转换能免费升级
nvidia:   内核编写层triton 位于框架转换成cudnn 以外： triton 看不到下面的黑盒

2. 复杂度
nvidia: 复杂度给上层程序员， 上限高
google: 都交给编译器了，mosaic 包揽一切，用户描述数据块和流水。 下限高

3. 确定性
相同硬件的相同输入
google: 输出相同： 单线程， 有利于回归测试
nvidia： tirton 输出不同： 非确定调度

4. escape hatches 逃生通道
google:：比较难
nvidia: triton 不够了，降级到写cuda, ptx 汇编



### insight
#### insight1: 
其实感觉VMEM, HBM 这两层架构对程序员还挺简单的，感觉AME 的tensor core 也需要直接从Memory 拉一根线到加速器Scratchpad memory 中，显示管理计算分布
回到TPU 的内存布局，确实还比较简单，HBM -> VMEM -> MXU/VPU regFile。 没有硬件管理的L1, L2 Cache. 

其实感觉VMEM, HBM 这两层架构对程序员还挺简单的，感觉我们要做的CPU 紧耦合加速器AME 的tensor core 也需要直接从Memory 拉一根线到加速器Scratchpad memory 中，防止又到L2, 然后才到对应regfile or local memory 中。
我只是好奇，这种单层cache, 就很难做到多层cache 那样，兼顾容量 和延迟这种。不过TPU 确实不太考虑延迟，只要大容量和大吞吐就行，延迟通过并发来掩盖了。  
不过我们CPU 紧耦合加速器的L2 有个好处是，CPU 的vector 能帮忙从L2 读数据来处理，相当于要么加速器中vector 来处理这些，要么用CPU 中的vector, CPU 中的就得考虑和加速器的同步数据问题了，还是得定量看矩阵和向量数据搬运多不多。

> 如果 CPU vector 处理会导致中间结果写回 L2/HBM，然后 AME 又重新读一遍，那大概率应该融合进 AME/加速器侧。

大概率需要两套机制，
一个直接从HBM 到AME 中： 大GEMM, prefill, 不需要CPU 触碰； but 一致性困难点
一个从HBM -> L2 -> AME 中： 小矩阵，控制流复杂，cpu vector +AME 混合



#### insight 2:
我突然有点好奇了，为何感觉几乎所有的加速器都在抛弃传统Cache 结构
我最早在Diannao 论文中看过，当时作者认为通用 cache 对这种固定模式访问不划算：tag check、associativity、speculative read、conflict 都是额外成本。专用 scratchpad/buffer 因为访问模式可预测，可以更高效。

而GPU, TPU 也是慢慢都在使用软件管理的scarchpad memory

我理解CPU 时代，操作对象很小，几乎只有几byte, 难以软件管理，并且**未来访问地址太难预测**,所以硬件统一管理兼顾延迟和吞吐。
而AI 时代，操作对象都是tile, 很少有对tile 某个元素做操作的
管理粒度大，软件管理更高效，同时减少硬件管理部分的面积（但感觉如果能加面积，或许硬件管理算法也可以高效的）

此外，CPU 时代还在关注多cache 一致性问题，本质上控制流太复杂了，而AI 时代基本都不太管了，因为在运行前就知道哪一块要读写哪一块了，都通过软件来规避一致性问题了吧，编排上来减少？

>不是“cache 不好”，而是**当访问模式足够规则、数据块足够大、编译器足够知道未来时，硬件透明 cache 的很多优点会变成额外成本**。
>**现代 AI 加速器不是彻底不要 cache，而是把“发现局部性”的任务从硬件 cache 转移到编译器/程序员/运行时；cache 退到辅助层，scratchpad 成为主战场。**
>Cache 是“硬件猜测局部性”的机制；scratchpad/VMEM 是“软件显式声明局部性”的机制。AI workload 的局部性太强、太规则，所以不值得让硬件每次都猜。


大块规则张量数据：
    软件管理 scratchpad / shared memory / VMEM / LDS

不规则访问、metadata、普通 global load、跨 kernel 复用：
    仍然靠 L1/L2/L3 cache 或类似结构

一致性没有消失，而是从 cache-line 粒度，提升到 tensor/kernel/stream 粒度。

> 关键：一块 tensor 数据在某一段时间内，到底归 CPU cache 管，还是归 AME scratchpad 管？切换 ownership 的成本是多少？


### 作者的结论

- 偏好使用TPU
从组合  -> 编译器  -> 性能分析器

#### 组合
![](../../assets/Pasted%20image%2020260513175050.png)

编写 Triton 和编写 Pallas 之间最大的日常差异在于，Pallas 内核是 JAX 函数，而 Triton 内核不是。
google: 写一个内核，能自动扩展
cuda: 写一个内核要自己维护很多东西
当内核数量变多，复杂度提升很大

#### 编译器太强了
mosaic 能处理向量寄存器分块tiling检测，布局推断，DMA 调度等， 

VLIW（超长指令字）的编译时调度。 
在 LLO 层，Mosaic 将指令打包成 VLIW 字。标量操作、向量操作和 DMA 触发若互不冲突，则可并行执行；Mosaic 负责判断哪些指令可以打包在一起。在 GPU 上，SASS 调度是每个 SM 内部硬件调度器的任务，从 Triton 内核中无法看到。在 TPU 上，VLIW 调度是编译器的任务，如果愿意，你可以在 LLO 输出中看到它。
这个挺好，能自动并行打包

#### 性能分析
TPU 的XProf
![](../../assets/Pasted%20image%2020260513175655.png)

它能提供程序中每个操作的时间线，对于 TPU 上的常驻操作，还会标注每个操作上的脉动阵列占用率（MXU 忙碌的周期占比）、MXU 利用率（MXU 忙碌时执行有效工作的插槽比例）、HBM 带宽利用率、VMEM 占用率以及停顿情况。你可以看到哪个操作较慢、它在等待什么，以及它是计算密集型还是内存密集型。
一旦你能够阅读这些注释，性能分析器就会精确告诉你编译器对每个缓冲区做了什么：哪些操作数最终进入了 VMEM，哪些溢出到了 HBM，tile 的形状在哪里，
eg:
f32[32,32,4096]{2,1,0:T(8,128)(2,1)S(1)}


Nsight Compute 是 NVIDIA 的同类工具, 但看不到编译器决策细节


---


### 其他内容
Flash Attention 的核心思想是始终不实例化完整的注意力分数矩阵，而是将其保留在 SRAM/VMEM 中。

MFU = Model FLOPs Utilization，模型 FLOPs 利用率。
  MFU = 模型实际训练吞吐 / 理论峰值吞吐

GPU 优势
1. 第三方库的广度 diffusers、vLLM、SGLang、TRL，以及 arXiv 上每个发布 PyTorch 模型的研究仓库：这是 GPU 最大的单一优势，而且与性能无关。这是生态系统的问题。
2. Eager PyTorch 的易用性 即时执行的交互式 Python 开发——正是这一特性最初让 PyTorch 脱颖而出——在 GPU 上的表现仍然优于 TPU。
3. 单 GPU 调试速度 相关但不同。对于小模型、单设备工作以及任何项目的第一周，GPU 的迭代速度更快。JAX 有良好的 REPL 支持，但 TPU 后端的编译成本高于 GPU 的急切执行成本，并且调试工具（XProf）针对的是比你在第一小时编写的更大的内核而调优。这是我们正在进行的另一场战斗，但目前仍然是 GPU 的胜利。
相当于GPU pytorch 类似python 的JIT 模式，快速迭代；而GPU 有编译开销，长时间运行后更快

不要再以线程和线程束为单位思考，要想着数据块和流水线
