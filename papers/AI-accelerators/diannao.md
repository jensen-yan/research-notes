
https://www.cs.princeton.edu/courses/archive/spring16/cos598F/p269-chen.pdf

DianNao: A Small-Footprint High-Throughput Accelerator for Ubiquitous Machine-Learning

2014 年早期论文

DianNao 设计了一个面向 **CNN/DNN 前向推理** 的小面积、高吞吐 ASIC 加速器，通过定制化 scratchpad/buffer、DMA 预取、数据复用和 16-bit 定点计算，在 65nm 下实现 **3.02 mm²、485 mW、452 GOP/s**，相比 128-bit 2GHz SIMD 平均 **117.87× 更快、21.08× 更省能耗**。

别只把 CNN/DNN 当成一堆 MAC 来加速，真正决定大规模神经网络加速器效率的，是计算单元与片上/片外存储之间怎么配合。

**大规模 CNN/DNN 里，memory transfer 会变成一等公民**。

## 神经网络层的分析

论文把目标网络主要分成三类层：
![](../../assets/Pasted%20image%2020260513153603.png)

| 层类型                              | 特征                        | 主要瓶颈                      |
| -------------------------------- | ------------------------- | ------------------------- |
| **Classifier / fully-connected** | 输入神经元和权重矩阵相乘              | synapse/weight 巨大，访存压力大   |
| **Convolution**                  | 有 feature map、kernel、滑窗复用 | shared kernel 时复用weights好 |
| **Pooling**                      | 没有 synapse，只做窗口聚合         | 计算少，但输入复用机会也少             |

它先在处理器/SIMD baseline 上做 locality 分析

### 架构设计
![](../../assets/Pasted%20image%2020260513153742.png)

>图 11 是核心结构，主要包括：

- **NFU（Neural Functional Unit）**：计算阵列；
- **NBin**：输入神经元 buffer；
- **NBout**：输出/部分和 buffer，暂存还没算完的输出，下次还要用
- **Split Buffer**：synapse突触/weight buffer, 缓存所有weights；
- **CP**：类似可配置 FSM 的控制处理器；
- **三个 DMA**：分别服务输入、权重、输出。

NFU 分三段：

1. **NFU-1**：16-bit 定点乘法，主要做 synapse × input；
2. **NFU-2**：adder tree / pooling max / shift 等；
3. **NFU-3**：非线性函数，用 piecewise linear approximation 实现 sigmoid/tanh 等。


也是不相信Cache, 抛弃了传统Cache, 全部用软件管理的scratchpad memory 来保存数据
**作者认为通用 cache 对这种固定模式访问不划算：tag check、associativity、speculative read、conflict 都是额外成本。专用 scratchpad/buffer 因为访问模式可预测，可以更高效。**

DMA 预取 + reuse

DMA 指令提前发出，把数据搬到 NBin/SB，相当于非投机的 prefetch。这样能隐藏一部分片外访存延迟。论文指出这也是 DianNao 相比 SIMD 快很多的一个原因。
相当于软件来负责数据搬运

