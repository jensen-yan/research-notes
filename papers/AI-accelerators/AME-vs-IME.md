## 整体思维导图

[XSAI](XSAI.md)  描述了当前我们使用AME 架构的基础设计

from 王曦爽老师

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d582d4b1e032b0deb17fd9988d142de1f37d816a2d103fc5a0c71b2c8acaa867e02a4baa6e034fa3f8?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

相关链接：

[https://riscv.org/blog/towards-an-integrated-matrix-extension-workload-analysis-of-cnn-inference-with-qemu-tcg-plugings/?utm_source=chatgpt.com](https://riscv.org/blog/towards-an-integrated-matrix-extension-workload-analysis-of-cnn-inference-with-qemu-tcg-plugings/?utm_source=chatgpt.com)

[https://riscv.atlassian.net/wiki/spaces/IMEX/pages/598867969/IME%2BRatification%2BPlan?utm_source=chatgpt.com](https://riscv.atlassian.net/wiki/spaces/IMEX/pages/598867969/IME%2BRatification%2BPlan?utm_source=chatgpt.com)

### 碎片想法

矩阵性能会受到vector register 容量和load bandwidth限制，这本身就是 IME 的核心约束

当然真实实现可能一个 16×16 tile 不是一个寄存器就装得下，还可能跨多个 vector registers。

IME 文档里有一句很关键的话：

矩阵操作性能本质上受两件事限制：处理器 load 带宽，以及能放在 vector registers 里的矩阵状态量。 这两者一起决定 IME 能达到的最大矩阵乘性能。

这时 IME 的收益就会明显下降。

因为 IME 的一个前提就是：在 vector RF 中保有足够矩阵状态，才能把 matrix FU 喂起来。

RVV：拿向量拼矩阵。

IME：矩阵语义进 ISA，但矩阵还住在 RVV 寄存器里，所以吃不吃得饱很看 RVV 家底。

AME：矩阵自己有房子、有路、有厨房，所以更容易把存储和计算一起围绕矩阵重做。

CPU OoO 理解为“面向单条/小粒度指令的动态调度”，把矩阵吞吐调度理解为“面向 tile/outer-product/数据流阶段的持续供给调度”；前者强在处理杂乱依赖，后者强在维持规则高吞吐。

## 调研

## 进迭时空A100: 1024bit IME

论文：SpacemiT K3: A RVA23 RISC-V AI CPU with 60 TOPS AI Compute

软件调度，如何自动调度到不同核心上吗？

lamma.cpp 如何固定把AI 请求，走socket 协议；作为server 一直运行，通推一体的优势？

协议栈需求对CPU 需求大，每个用户端口都需要走协议。

也有cli 版本，单用户；单用户多个CLI 如何做不同大模型保存切换？

同构（大模型天然分散在不同核，TP切分） vs 异构（操作系统难一点？小核无法用到x100 向量算力，运行和调度，需要操作系统支持，需要感知每个程序是否有矩阵指令）；

一个乱序核 + 一个AME 一定绑定吗？

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d5ed6307f4829c0b3d0fe6ead34bb73548e26d56d85eda30c1f97f3e669cd2346833c6a5ef51fdc8cf?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

8个x100 乱序核（KMH V1 改的? 9分/GHz） + 8 个A100 顺序AI 核心

一个A100 cluster 包含4个core + 2 个tensor + 2个local memory

对应下图 2个core 共享 1 tensor + 1 local memory

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d5b7b3560431cbe5f780dbd74f5dd1cc46070768bbe83d9d95ec01a4595de0655d4c9d40e7e0687a62?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

注意这里A100 标量是顺序发射流水线，非乱序核。

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d51ab3af29d321b26b883449fd0a0530164a918d6e1eeeb89acef5835e8a9ac4fbeb3e9840a28eedf0?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

IME 也分成两类，一个是在核内做，一拍处理256bits； 一个实在tensor core 做，一拍处理1024bits

> Each core is equipped with an independent
512-bit load channel. The front end supports dual issue in the form
of one 512-bit load + one MMA instruction per cycle.

这里每个core 只有512bit load 带宽，2核一起到1024bit 输入

> Implementing RVV VLEN = 1024 ( an ultra-wide datapath) on a 12
nm process makes timing closure often difficult to achieve in a single
pass with “default synthesis” alone and typically requires targeted
iteration in conjunction with physical implementation. For example,
the register-file read/write ports and the bypass/selection network
may be structurally partitioned and/or hierarchically pipelined;
critical paths are then compressed via retiming and fanout con-
trol. In the backend, floorplanning, critical-block placement, and
constraint-driven guidance are used to reduce timing degradation
caused by long interconnects and congestion. Ultimately, A100
closes PPA on the VLEN=1024–related critical paths at 2.1 GHz.

论文也提到在12nm 工艺下实现32个1024bits 的向量寄存器比较困难，在和后端协同优化后达到了2.1GHz

3.2.2 Tensor Core. 这一段比较绕，但也非常重要

这节 3.2.2 的核心结论其实是：

1. 单核裸跑一个 MMA 时，Tensor Core 很容易“等数据”，利用率只有 50%。
    
2. 如果利用 1024-bit vector register 做 panel load + register blocking，单核利用率可以从 50% 提高到 66.7%，operational intensity 也从 8 提高到 10.67 MAC/B。
    
3. 但即便如此，单核仍未必能把共享 Tensor Core 完全喂满；所以让两个 compute core 共享一个 Tensor Core，并由两个核并行供数，才更容易把这个昂贵的 MAC 阵列真正跑满
    

文中默认了这些前提：

- 两个 core 共享一个 Tensor Core。
    
- MMA 的输入/输出都放在 vector registers 里。
    
- 每个 core 有 32 个 vector registers，每个寄存器 128B。
    
- 每个 core 自己有一条 512-bit load channel。
    
- 前端每周期最多能发：1 条 512-bit load + 1 条 MMA。
    

这里其实已经包含了IME 非常受32个vector 寄存器影响，即便每个到1024bits

但是在考虑流水线复用的情况下，最多拿16个reg 存储C， rm * rn <= 8, 比较难做更大范围流水线

此外，512bit 的load 流水已经非常极限了，比较难扩大

也就是IME 即便用了tensor core, 也受到CPU core load 带宽和vector 寄存器个数影响，想要做的更多，只能让多个标量核一起喂给tensor core才能喂饱

论文之后还有一些有意思的

A100 通过向量化访存和一系列读带宽优化，把主存读吞吐提升到了 baseline 的约 3.2~3.3 倍。

并且它有一个local memory ，细节很少，猜测如下，最终每拍load 512 bit 的带宽

- vector regs：最贴近执行单元的 tile/accumulator 存放处
- local memory：软件管理的近核 scratchpad，放更大一点的 working set / panel / staged tiles
- L2 / main memory：再往外一层，负责更大范围的数据供给
- AI-DMA + async prefetch：负责把外层数据提前搬到 local memory / 近核层
- runtime / hand-written kernels：负责什么时候搬、搬多少、怎么 overlap 计算

这里虽然是IME, 但是增加了额外的指令来管理scratchpad memory, 也是生态负担很大

不过连IME 也需要local memory, 可能这个存储带宽需求真的很大，毕竟A100 同时面向两类场景

1. 计算密集，访存低，但对局部性要求高，使用local memory (对应LLM 计算部分)
2. 计算低，访存为流式带宽访问，增大访存带宽 （对应LLM decode ）

结论：即便是 IME 路线，真到本地/低延迟/大模型推理，也还是会被逼着走向“更专门的矩阵单元 + 更近的存储 + 更强的数据流控制”

待评估：softmax 是否在乱序核心上比顺序核有好处？

评估不同长度vector length 的性能差异？

窄的乱序vs 宽的顺序 性能差异。

## 玄铁C950

### 256bit vector + AME 0.5

[https://www.xrvm.cn/product/xuantie/C950](https://www.xrvm.cn/product/xuantie/C950)

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d54880e50be42c29a1a2bc814e59b329602331d8ccec634262c38a6d3b4e0ea75c48d93494f71bab3e?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d5aca3e234aceff4bc75bc578db57b195d222241d27b00c7335e5935b821292d0f805a80d201db87b0?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

AME 矩阵累加和VPU需要配合好，可能需要额外的VPU在协处理器中。

512 Byte = 4k bit vector length, N = 8

DMA

长向量评估？

coupled L2 改成高带宽，放松了矩阵一致性协议，标量一致性不变，时序更差2.5GHz。

原本标量需要访问目录2次，矩阵不记录inclusive，只读关系，直接写入L2 。

refill 32byte， 需要2拍。 矩阵 64B, 8 way = 512B

同步频率vs 异步

大模型对L2 Cache size 大小需求？

MOE 可能困难，一个专家内44M b 内部能预测。

包含tensor cache and local memory + DMA

支持 Softmax, SiLU, and GELU 等加速

### Vector 侧

支持：

- BF16 / FP16 / FP32 / FP64
    
- INT4 / INT8 / INT16 / INT32 / INT64
    
- 以及 Vector Convert Instruction Extension，支持  
    FP4 / FP8 / MX Scaling data format conversions。
    

### TPE 侧

支持：

- INT4 / INT8 / FP8 / MXFP8 / MXFP4 / RVFP4 / FP16 / BF16。
    
- TPE 的 DMA 大概率不是“和 CPU L2 紧耦合共享的数据通路”，而更像是能直接面向更外层共享内存系统（L3/NoC/主存）搬运到 TPE local memory / tensor cache。
    

这意味着一个很关键的架构分工：

- vector 更像通用向量处理 + 格式转换 + 非主干算子；
    
- TPE 负责真正的低精度矩阵/张量计算。
    

Support up to ８ TOPS per TPE（ 猜测 4T * 2GHz） Supported data types: INT4 / INT8 / FP8 / MXFP8 / MXFP4 / RVFP4 / FP16 / BF16

L1 data cache: 64KB, 4-way set-associative, 4-cycle load-to-use latency L2 cache: Private per core, configurable up to 3MB MMU: Sv39 / Sv48 / Sv57 virtual memory addressing modes, two-stage address translation

玄铁AME 实现细节

[https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/doc/slides/Matrix_Proposal_on_RISC-V_China_Summit_2023.pdf](https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/doc/slides/Matrix_Proposal_on_RISC-V_China_Summit_2023.pdf)

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d539e2287f3d11b9608d382e1ee9a03b69c97608882d4e31fb6f19c1d65d6c3b957f602427833c19e5?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

这个图挺好：本质上，vector 能一拍操作N个数，执行N 个操作

matrix 能一拍操作N*N 个数，执行N*N*N 个操作，是vector 的N*N 倍执行效率

![](https://alidocs.dingtalk.com/core/api/resources/img/5eecdaf48460cde5831f9db959d5cc270869a6ccbdb2bcf475b8339e1c4c2483b1b895eb8c846a833490e2fc8d4af258a156a98577f418d5b5543bdfc1ebf72725d2a10ec03ed42529c2eb50c7d01a05dfc535d5e6f665fa73f49268c8caa69a?tmpCode=e2bfc951-424c-4dc7-9d25-e113f85f8afc)

这里讲解AME vs IME 的差异

pro: 灵活性 + vector 并行

cons: 资源浪费+ 更多和vector 数据搬运（这个不确定）+ 更大上下文切换开销（待评估）

[https://riscv.org/blog/towards-an-integrated-matrix-extension-workload-analysis-of-cnn-inference-with-qemu-tcg-plugings/?utm_source=chatgpt.com](https://riscv.org/blog/towards-an-integrated-matrix-extension-workload-analysis-of-cnn-inference-with-qemu-tcg-plugings/?utm_source=chatgpt.com)

这里就提到vector 和 matrix 很少有数据搬运:矩阵指令读取的数据中，三分之一使用加载指令生成的值，三分之二访问其他矩阵指令生成的值。没有矩阵指令直接操作向量指令的结果

但是：这篇文章的“矩阵和向量几乎不互相使用”是对 CNN workload 的观察，不是对所有 AI workload 的普遍规律；像 Transformer 的 softmax、layernorm、rope 这类步骤，天然更偏 vector/reduction，因此更可能需要 matrix 与 vector 之间的高效衔接，要么寄存器间搬运，要么存回cache 再取，要么AME 支持softmax 等（难做）。

[https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/doc/slides/AME_workload_analysis_20240412.pdf](https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/doc/slides/AME_workload_analysis_20240412.pdf)

[https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/demos/README.md](https://github.com/XUANTIE-RV/riscv-matrix-extension-spec/blob/master/demos/README.md)

We use qemu and cpf to count the number of instructions of the program. Compared with vector extension 1.0, RISC-V Matrix Extension has an improvement of 5.28x - 7.36x on resnet50, speed up 9.76x - 15.44x on gemm (160 x 160 x 160)

这里做了一些统计，相对于全用vector 相比，用AME 指令平均少10倍的指令数，这是指令数的优势，能稍微减小一些前端压力（作用不大）。

## 结论

当前 XSAI 采用 AME 路线是可行且合理的。  
理由不是 AME 在所有场景下都优于 IME，而是：在 CPU 紧耦合、低延迟、本地推理的目标下，XSAI 已经围绕 AME 形成了从 AMU、DSA 接口到 HBL2 和 cache/coherence 的系统化方案；相比之下，IME 虽有较好的编程连续性，但其性能更依赖 RVV 寄存器容量与 load 带宽，真要做到高性能也常常需要额外引入 local memory、tensor core 与软件调度机制。公开的 K3/A100 方案就是一个例子。

### 风险边界

当前最大的未定项不是“AME 对不对”，而是目标 workload 还没有完全收敛。  
CNN 类 workload 的公开分析显示 matrix/vector 交互较少，这对 AME 是有利的；但 Transformer 尤其 decode 阶段，softmax/layernorm/rope 等向量化与规约算子占比更高，是否会放大 matrix↔vector 边界成本，需要通过 gem5 和代表性 kernel 进一步定量验证。

### 模型组任务

gem5 组接下来还要继续做“固定资源预算下的系统级验证”。  
把比较对象定为：

- XSAI 当前 AME+HBL2 方案
    
- 一个合理参数化的 IME-like baseline（VRF/L1/L2/load BW 固定）  
    然后比较 batch=1/小 batch 的 GEMM、attention、norm/softmax 混合链路，看 latency、load stall、working-set residency、matrix/vector 边界开销。
    

## 遗留事项

1. 找 yejingpeng 问他的模型 yuanbin
    vector 和 matrix 配合； 算力配比 1： 16
    vector 算softmax; matrix 算
    
2. IME vector length; 实现GEM5 IME 1024bit vector
    
    AME 用现有建模GEM5 评估.
    控制变量的情况下，同样L2 Cache, 固定算力 = 1cycle 计算ops 量，拉大ops 的差异
    机理分析 vs 数值建模
    对应的物理代价。先画出趋势。
    
3. 手写矩阵指令IME 的汇编测试（ygc 测试）
    
    问yangbibo老师的测试评估。用于IME 的测试和评估
    
    闭环：[https://gitee.com/RSPwFPGAs/roofline_model_spacemit_k1](https://gitee.com/RSPwFPGAs/roofline_model_spacemit_k1)
    
1. 填写大颗粒的变量
