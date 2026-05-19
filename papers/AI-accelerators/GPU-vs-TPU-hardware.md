https://x.com/MainzOnX/article/2044462083010662771
From SIMT to Systolic: A Foundation for GPU and TPU Architecture

这篇文章讲解GPU, TPU 的发展路径，来说明这段时间的硬件为何这样演进


### 为何要有加速器？
In-Datacenter Performance Analysis of a Tensor Processing Unit.
2017 这篇论文的激进观点
>缓存不好、乱序执行不好、同步多线程不好、推测执行不好、分支预测不好、硬件预取不好——所有形式的隐藏延迟动态机制都不好。所有这些硅片都被用于让不可预测的代码看起来快。而机器学习工作负载并非不可预测。你在动态机制上花费的每一字节硅片，都是本可以用于算术单元的一字节。

但感觉在大模型推理下，可能观点还有争议，毕竟现在
>ML 的核心 tensor kernels 仍然高度规则，应该用确定性数据流、scratchpad、编译器调度和专用矩阵单元；
>但 LLM serving 已经变成 tensor kernels + 动态系统控制的混合问题，所以需要异构架构：矩阵流要静态化，控制流要保留灵活性。
所以可能对于推测解码等，还是需要一些CPU 来参与乱序调度的

CPU: 分支多，内存访问不可预测，单线程低延迟最好 -> BPU, 乱序调度，预取等
ACC: 大量矩阵乘，向量操作，可预测，重度循环，几乎无分支，并且编译阶段、运行前 能知道循环边界和内存访问，算数强度

核心：关注矩阵、tile; 
GPU 着色器有些浪费，添加Tensor Core, 用数千个线程掩盖数据到达延迟
TPU 完全抛弃调度，用编译器来管理数据流动！

### 内存墙
regfile ->  片上SRAM -> Memory/HBM -> nvlink -> 光纤
每一代硬件：计算提升很快，但是访存提升慢

芯片更大， 内存更近，软件要更努力才能让硬件忙碌（分块）

算数强度AI =  FLOPS / Bytes  = FLOPS / (Byte/s) = 算力 / 带宽 
eg: N * N 矩阵乘法  AI= 2N^3  / N* N = N, N 更大，AI 更大
>H100 SXM5：峰值 Tensor Core BF16 为 1,979 TFLOPS，HBM3 带宽为 3.35 TB/s，得出脊点约为每字节 591 次浮点运算。 2000/ 3.3 = 600

[roofline](roofline.html) 介绍

![](../../assets/Pasted%20image%2020260513102423.png)关键： 拐点，屋脊点，决定到底是内存瓶颈还是计算瓶颈

> 图中还有一个没人画出的第三区域：指令延迟受限区。你有带宽，有算力，但无法足够快地发射指令来利用它们。这正是 GPU 占用率规划（occupancy discipline）的来源，也是 TPU 编译器通过指令打包工作获得回报的地方。
> 还不是很明白？


### 3种模式

参考[DNN-survey](DNN-survey.html)
1. SIMD: CPU SIMD or GPU 向量
	1. easy
	2. 如果填不满，必须mask, or stall
2. SIMT:
	> 单指令多线程。想象一位指挥家带领一支 32 人管弦乐队：一个起拍让每位乐手同时动作，但每个人各自演奏自己的声部。这就是 SIMT。你编写的代码看起来像是在单个线程上运行。硬件将线程分组为 32 个的线程束，并在底层以锁步方式运行线程束，因此本质上是在抽象层下的 SIMD。但编程模型看起来像线程：每个线程都有自己的程序计数器、自己的寄存器、自己的控制流。当线程束中的线程需要执行不同操作时，就会产生线程束分歧。硬件将分支串行化，先运行一条路径并屏蔽其他通路，再运行另一条路径。这虽然可行，但会损失吞吐量。
	关键：硬件负责锁步执行，内存合并，来隐藏延迟
3. systolic：
> 脉动阵列是一个由乘累加单元组成的二维网格，这些单元相互连接，使得数据像水流穿过网状结构一样在网格中流动。每个时钟周期，每个 MAC 单元从其邻居处获取一个值，与一个固定权重（或根据数据流方式流动的权重）执行乘加运算，然后将结果传递给下一个邻居。每个单元无需指令获取，无需控制逻辑，只有一种通过算术网格流动的数据节律。

一个好的PPT 讲解TPU: 
https://courses.grainger.illinois.edu/cs533/sp2025/notes/tpu_arch.pdf

pro: 密度：ALU 不用管获取指令和管理寄存器，同面积能放更多ALU
cons: 不灵活：如果填不满会浪费资源，需要高密度，大batch 来填充

![](../../assets/Pasted%20image%2020260513105007.png) 


### 两种哲学
NVidia: 从程序员，核心是线程，大量并行线程，来构建内存。为每个SM/ALU 加 tensor core

Google: 围绕矩阵数据流(Tile), MXU 是核心，添加向量单元，标量单元控制，添加VMEM 来让编译器处理缓存。添加ICI 扩大内存。 

> 这些不仅仅是组织上的差异，它们还会叠加放大。一旦 NVIDIA 选择了 SIMT 路线，后续的每一个设计决策都必须让线程更高效：共享内存、张量核心、线程束级内在函数、TMA、内存屏障、集群 API。一旦谷歌选择了脉动阵列，后续的每一个决策都必须为其提供支撑：确定性的 VMEM 分阶段处理、编译器调度的 DMA、环形网络结构、以及针对 MXU 无法吸收的操作的稀疏核心卸载。

>随着抽象层逐渐成熟，GPU 会变得更容易编程。随着编译器不断优化，TPU 会变得更具吞吐密度。GPU 的下限（编写直观代码所能达到的效果）很高，因为运行时为你承担了大量工作。TPU 的上限（通过编译器感知代码所能达到的效果）很高，因为没有任何资源消耗在运行时动态性上。这两种理念并没有严格意义上的优劣之分，它们各自成就了不同形式的卓越。

GPU： threads -> warps -> blocks ->        （cluster）-> grid
       工人  ->   32 工人  -> 车间（共享内存）->相邻园区 -> 工单
       其中SM 和blocks 解决，SM 是硬件，blocks 是软件，调度器选择不同blocks 放到SM 中
![](../../assets/Pasted%20image%2020260513105641.png)


### GPU 入门
GPU： threads -> warps -> blocks ->        （cluster）-> grid
GPU = 多个SM
SM = 标量 + 向量 + tensor core + + regfile + L1 Cache
所有SM 共享L2 Cache, HBM 是片外主存储

1. warp divergence: 同一个warp 线程走到不同分支会分叉执行，导致吞吐下降
2. coalescing: 同一个warp 对相邻地址加载会合并为一次事务
3. occupancy: 一个SM 保持多少个warps 运行来隐藏延时

![](../../assets/Pasted%20image%2020260513110246.png)

![](../../assets/Pasted%20image%2020260513110325.png)


A100(2021) -> H 100(2023) -> B200 (2025)

#### 1. Ampere/ A100: 
添加cp.aysnc: 把数据直接从全局内存移动到共享内存，不需要寄存器spill
80G HBM ， 2000GB/s 带宽， 312 BF16 TFLOPS tensor cores
MIG 多实例GPU: 方便云用户分区租用

拐点 AI =   312 TFLOPS/   2TB/s = 153 FLOPS/Byte

#### 2. Hopper/ H100:
数据移动不是副作用，而是程序本身！
描述：把HBM 中tile A 用某个布局放到 共享内存中，硬件会自动完成，线程也能异步做其他事情！

A. Thread Block Clusters
GPU： threads -> warps -> blocks ->        （cluster）-> grid
添加clusters ， 相邻车间概念，
之前相邻块无法通讯
now：一个大矩阵，放到相邻块中组成cluster(4 blocks), 调度到连续SM 中，可以访问彼此共享内存，分布式共享内存DSMEM.   速度： 本地SMEM > DSMEM > L2 > HBM
pro: 现在能跨多个SM 一起操作，不用来回从L2 搬运，减少数据移动！Flash-attention-style kernels 能每个cluster 处理一个head ? 

B. Tensor Memory Accelerator(TMA)
描述 src （base, shape, stride）  + dest， 发送指令，DMA 自动移动数据块

C. Warpgroup MMA (WGMMA)
1 warpgroup = 4 warp = 128 thread
old: 1warp mma
new: 1 warpgroup mma 更大矩阵乘法，1条指令操作更多thread 和 数据
能直接从共享内存获取数据，而不用先移动到寄存器，减少数据搬运

D. mbarrier
old: block 内所有thread 等待其他所有thread 到达
new: 可以某个warp arrive, wait 另一个warp， 能做到生产者-消费者 模式
warp A 发起TMA 加载，mbarrier 标记到达继续执行 （producer)
warp B 在mbarrier 等待，用WGMMA 执行矩阵乘消费数据，(consumer) 然后再另一个mbarrier 标记到达，表示空闲
![](../../assets/Pasted%20image%2020260513112045.png)

![](../../assets/Pasted%20image%2020260513112050.png)

添加FP8

H100 3.35TB/s, 50ML2 
old: 先算数运算，在围绕运算结果执行内存移动
now: 调度内存移动，算数运算在内存移动空隙中完成
（需要好的运行前调度，也就是编译器 Triton）


#### 3. BlackWell / B200
进一步放大，NVL72, NVL144, FP4
引入了Tensor memory: TMEM 256KB， 位于SMEM 和 张量reg 之间
tcgen05 核心张量运算，和warp 调度解耦，把算数运算和线程解耦，密集的运算中可以不要调度

#### 演进
Ampere 扩宽了高速路：更大tensor core, 更大HBM
Hopper 让移动更明确，以数据为中心
BlackWell 让多GPU 为一个整体
目的：让编译器而非程序员来调度数据移动和计算
毕竟memory 层次太多，同步异步比较难，数据结构也变化大，硬件对外暴露的细节太多了
![](../../assets/Pasted%20image%2020260513113133.png)


### TPU 入门

TPU 有少量大模块组成
v1 -> v2/v3/v4 -> v5 -> v6 -> v7
v5p 有两个tensor core = 多个MXU = 脉动阵列 + 向量 + 标量
128* 128   -> 256 * 256

1. MXU 脉动阵列
>MXU 负责执行密集矩阵运算。其数据流是权重驻留的变体，这意味着权重固定不动，激活值流过。一旦加载了权重，就通过网格流式传输激活值，累积结果从另一侧输出。当数据块适配网格时，吞吐量非常惊人。代价是，所有内容都必须针对 MXU 进行形状设计，否则 MXU 将无法发挥任何作用。

![](../../assets/Pasted%20image%2020260513113453.png)

2. VPU: 完成向量：逐元素操作、归约、Softmax 内部计算、层归一化数学运算以及激活函数。这是一个 SIMD 风格的单元，拥有独立的寄存器文件和独立的数据通道数。
3. **标量单元**负责控制流、地址生成以及无法在向量或矩阵单元中表达的少量运算。它规模很小。TPU 架构的核心在于，大部分芯片面积都分配给了 MXU。

内存：**无硬件管理的L1, L2 Cache.**
每个chips 有VMEM ， 都是软件管理的， v5e 是128MB. 
编译器负责数据从HBM -> VMEM -> MXU/VPU regFile
任何无法放入 VMEM 的操作其算术强度都会显著增加,  关注如何尽可能长时间地将数据保留在 VMEM 中。
需要很大batch size 来复用weight.  (GPU 友好一点)

ICI (Inter-Chip Interconnect) = NVLink， 带宽是4.8Tbps
一个pod 包含9000 个chip, 用3D torus 连接（包含6个邻居）

SparceCore: 负责MXU 不擅长的，稀疏嵌入、分散收集、以及哈希表密集型工作负载。

![](../../assets/Pasted%20image%2020260513114136.png)

![](../../assets/Pasted%20image%2020260513114151.png)

#### v1
training 看 throughput，inference serving 看 tail latency。
服务商不希望某个请求延迟特别慢
如果一个推理程序的 shape、tile、数据搬运、计算顺序都能在运行前（信息较少）排好，而且硬件执行时没有太多“临场发挥”，那同一个请求在加速器核心上的执行周期就更稳定，不容易出现偶发性长尾延迟。
而CPU/GPU  如果动态执行可能让某些时间变慢
>CPU 因为能获取运行时信息，所以在复杂、不规则程序上更强；TPU 因为假设 ML 主干足够规则，所以把动态机制拿掉，换来更高密度和更稳定延迟。  
   但如果 LLM 推理里出现变长、MoE、spec decode、KV cache 动态管理等复杂行为，TPU 静态编排确实可能浪费、变慢或需要 padding/bucketing/fallback。确定性执行保证的是“少抖动”，不是“永远最低延迟”。

Floorplan of a TPU Die
![](../../assets/Pasted%20image%2020260513115821.png)

控制单元只有2% 面积

#### v4: 
引入spacseCore
在嵌入密集型工作负载下，SparseCore 相比在 MXU 上运行相同任务可实现 5 至 7 倍的加速

光学电路交换（OCS）。OCS 让谷歌能够在配置时重新配置一个 Pod 的互连拓扑。物理网络是由光交换机构成的三维环面，这些交换机可以重新编程，将不同芯片连接到不同邻居.

#### V6e
MXU 跃升至 256×256。这意味着每周期 MAC 计数增加了 4 倍，使单芯片峰值性能较 v5e 大致提升了同等倍数。代价是**填充税**。如果你的矩阵乘法分块在两个维度上都无法被 256 整除，就会浪费周期。Trillium 上的内核作者需要比 v5p 上更仔细地考虑分块形状。编译器会处理大量填充工作，但填充税是真实存在的。
你不再需要将大量权重分片保留在每个芯片上，而是将权重分片到更多芯片上，并利用 ICI 来服务这些分片。这是一种以网络为先的内存层次结构，而非以容量为先。

#### V7
单芯片 4 FPLOPS FP8， 192GB HBM3e, 带宽7.3 TB/s
pod 支持9216 个芯片， 是NVL144 的 64倍！
等效显存是1.77PB = 1770TB,  算力 = 42.5 ExaFLOPS
并且保证通过ICI 保证带宽是1.2TB/s
每个芯片和B200 算力差不多，但是互联大很多！
pod 外通过光纤连接数万台服务器

#### 演进
v1 证明了加速器应去除动态特性，把芯片都交给算数单元
v4 打造为超级计算器
v5p 扩展规模更大，
v6 提高经济性
v7 互联优势是nvlink 64倍

TPU 系列一直押注于编译器调度的确定性、脉动矩阵密度以及以结构为先的扩展规模，这些因素比 SIMT 灵活性、缓存层次结构和单节点优化更能产生协同效应。

![](../../assets/Pasted%20image%2020260513120045.png)

### 对比
两个架构如何更相似
内存层次差异：by 添加
	GPU 添加更近计算的memory， SMEM, DSMEM, Tensor Memory
	TPU 一直有VMEM, 不用加
数据移动差异: by 承认
	GPU: TMA 承认异步传输有必要
	TPU: 一直靠编译器来做
执行模式差异：by 解耦
	 GPU: tcgen05 把Tensor 和 warp 调度解耦，相当于矩阵可以独立发射计算
	 TPU: 一直没有硬件调度
规模差异：by 互联
	 GPU: NVL72
	 TPU: 9216
精度差异：差不多，32->16->8->4
所以GPU 越来越像TPU 这样了

核心
1. 内存墙是主角，如何让数据始终保持在计算附近
2. 到底依靠程序员还是编译器，现在更多相信运行前信息，依靠编译器
3. 互联规模关键

### 启发
**ISA 还是需要暴露更多硬件细节，特别是需要能软件显示控制一些数据搬运位置和异步同步机制等** 



## 我自己的压缩总结

### 1. 这篇文章真正想说什么？

GPU 和 TPU 不是简单谁强谁弱，而是两条路线：
GPU 从线程灵活性出发，逐步把数据移动显式化、编译器化；
TPU 从矩阵数据流出发，牺牲动态灵活性，换取密度、确定性和规模化互联。

### 2. 我现在形成的三个判断

1. AI 加速器的主战场不是单纯算力，而是数据移动组织。
2. GPU 和 TPU 正在某些方向收敛：GPU 越来越显式管理数据流，TPU 需要补足非矩阵/稀疏/动态部分。
3. LLM serving 不是纯静态矩阵计算，因此 CPU/GPU/TPU/NPU 的协同边界会越来越重要。

### 3. 和 AME/IME 的关系

这篇文章提醒我：AME vs IME 不能只比较矩阵单元本身，而要比较：
- 数据从哪里来；
- 中间结果放在哪里；
- 编译器能不能安排 tile；
- 动态控制流由谁处理；
- decode / prefill / sampling / KV cache 分别适合哪种执行模型。

### 4. 我还没搞懂的问题

1. TPU VMEM 的软件管理具体在编译器 IR / runtime 中如何表达？
2. GPU TMA/WGMMA/tcgen05 和 TPU 的静态数据流到底差多远？
3. 对 batch=1 decode，TPU 的确定性优势和填充浪费哪个更重要？

### 5. 后续要查的资料

- TPU v1 paper
- TPU v4 / v5 / v6 / v7 architecture docs
- NVIDIA Hopper TMA / WGMMA docs
- Blackwell tcgen05 / TMEM 相关资料
- XLA / Pallas / Triton 对数据移动的表达方式
