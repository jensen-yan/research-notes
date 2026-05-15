https://ieeexplore.ieee.org/document/8695666/figures#figures
https://accelergy.mit.edu/timeloop.pdf
  
Timeloop: A Systematic Approach to DNN Accelerator Evaluation

主要是建立一个DNN 硬件加速器的评估框架/ 简单模拟器
给定3个输入
1. 硬件描述
![](../../assets/Pasted%20image%2020260513150407.png)
2. workload 特征（啥样的网络）
3. map 约束（RS , 哪些寄存器能传递）
![](../../assets/Pasted%20image%2020260513150650.png)

然后通过算法，自动找到最好的软件 mapping 算法，达到最好的性能
mapping 包括：数据如何切块，如何调度，放到哪里存储，哪些数据复用，哪些数据转发
同一个硬件不同的mapping 可能性能差异10倍

注意这里也区分了spatial vs temporal 加速器，对应综述论文

**其实感觉这些东西，应该会内嵌到编译器中，让对应的编译后端能自动根据当前硬件和同一个DNN 负载，自动给出一个mapping 策略！** 
类似现在的triton, TPU 编译器等
当然，现在编译器的难点：
1. 搜索所有空间太大了
2. 编译器需要更准确的cost model, 例如bank 冲突，寄存器压力等，很难通过模拟器建模好，所以需要auto-tuning or profiling 来弥补
3. 硬件后端信息暴露的比较少


## 细节
#### mapping 如何做
把一个 DNN layer 的 7D loop nest，分配到多级存储和 PE 阵列上的具体方式。
核心决策：

| 决策                       | 含义                                                     |
| ------------------------ | ------------------------------------------------------ |
| **Tiling / blocking**    | 每一级存储放多大的 tile                                         |
| **Loop permutation**     | loop 顺序怎么排，决定数据复用模式                                    |
| **Spatial partitioning** | 哪些维度展开到 PE 阵列上                                         |
| **Temporal scheduling**  | 哪些维度在一个 PE / RF / buffer 内按时间循环                        |
| **Bypass**               | 某些 tensor 是否跳过某一级存储，不占容量                               |
| **Dataflow constraints** | 例如 row-stationary / weight-stationary 本质是一组 mapping 约束 |

论文用 Eyeriss 举例：硬件组织包括 256 个 PE、每个 PE 一个 MAC 和私有 256-entry RF，一个 128KB shared global buffer，以及 DRAM；但光有这些还不够，还要用 mapspace constraints 描述 Eyeriss 的 row-stationary dataflow
#### 怎么搜索
Timeloop 认为一个 mapspace 是：

> 某个 workload 在某个 architecture 上所有合法 mapping 的集合。

它主要由几个子空间组合而成：

|子空间|含义|
|---|---|
|**IndexFactorization**|每个 loop 维度如何分解到不同 tiling level|
|**LoopPermutation**|每一级 loop 的排列顺序|
|**LevelBypass**|tensor 是否驻留或 bypass 某一级存储|

这些子空间做笛卡尔积，空间会爆炸。论文说，对于一个 7D CNN、4 个 tiling level 的架构，未约束 mapspace 会非常大；然后通过用户指定 constraints 和硬件资源约束进行剪枝/拒绝非法 mapping。
所以通过剪枝来大量减少对应的搜索算法！
## 6. Timeloop 的性能/能耗模型怎么做？

Timeloop 不是周期精确模拟器。它利用 DNN 计算和访存模式比较规则这一点，通过解析方法计算：

- MAC 次数
- 各级 memory access 次数
- network transfer 次数
- multicast 次数
- forwarding 次数
- partial sum reduction 次数

然后再根据每次访问的能量模型，估算总能耗。

性能估算是 throughput / bandwidth based：

- MAC 所需周期 = 总 MAC 数 / MAC 数量
- 通信接口周期 = 数据量 / 带宽
- buffer、network、arithmetic pipeline 化
- 总 latency 取各硬件组件 isolated cycles 的最大值

所以它本质上更像一个**结构化 roofline / bottleneck model**

### 思考：功耗估计
方案：
**总能耗 = Σ 每个硬件组件的访问次数 × 该组件单次访问能耗**

这里的“每个组件”包括 MAC、RF/SRAM/DRAM、片上网络、address generator、partial-sum accumulation 等。

- tile 分析，计算数据搬运次数；
- tile 访问变成组件访问，MAC/SRAM/regfile . 还有管SRAM 的bank 冲突等
- 每次访问计算能耗， 最后求和
	- memory 访问，最贵
	- 算数单元，便宜
	- wire/network. 片上大SRAM , 中间


局限性：感觉timeloop 可能比较适合DNN, LLM 推理的不确定。 毕竟DNN 全是对应的乘累加，而LLM 还有大量的数据搬运和向量等。
- 适合：大举证 + 规则算子：prefill
- 不适合：decode 和非GEMM 算子
> 低 reuse workload 被 DRAM 能耗主导，高 reuse workload 才更受 on-chip 组件影响。
- timeloop 所以选择RS 这种复用方式来提高数据复用性

不过要重建GEM5 的mcPat, 需要一定工作量

减少功耗几个方法：
1. 减少控制逻辑
2. 减少DRAM 访问次数
3. 减少片上大SRAM 访问次数
4. 减少广播次数

## 10. 局限性

这篇论文也有一些明显边界：

|局限|解释|
|---|---|
|**主要面向规则 DNN loop nest**|CONV / GEMM / FC / RNN 这类规则计算最适合|
|**不是 cycle-accurate simulator**|pipeline stall、bank conflict、network congestion、控制流扰动不一定准确|
|**搜索算法比较基础**|主要 exhaustive / random sampling，不是重点贡献|
|**主要是 per-layer 分析**|full network 的 inter-layer reuse 留作未来工作|
|**稀疏支持有限**|论文当时主要考虑能耗节省，未来工作才提到同时节省时间和能耗的稀疏架构建模|

论文自己也说，未来工作包括建模 inter-layer relationship、支持更复杂的稀疏架构、扩展到 graph analytics 和 sparse tensor