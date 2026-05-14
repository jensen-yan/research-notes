Efficient Processing of Deep Neural Networks: A Tutorial and Survey

https://semiwiki.com/wp-content/uploads/2018/03/2017_pieee_dnn.pdf

前面都讲了很多的基础概念，包括卷积，全连接，等的基础概念
核心是
![](../../assets/Pasted%20image%2020260513152334.png)
区分了
Temporal: CPU/GPU
Spatial: NPU/TPU
这两个加速器
我自己理解：
Temporal: 都是软件+硬件来让ALU 从全局寄存器中读数据，计算，再写入。
Spatial 主要让PE=ALU + control + Regfile 为核心，PE 能直接存储数据，同时PE 之间能相互流动数据

感觉Spatial 能更有利于，让数据更多的停留在计算单元附近，同时减少数据搬运！
[diannao](diannao.md)  就是比较早的使用spatial 的加速器


activations 就是输入，输出数据 or feature map
weights 为权重 or filters
卷积 = 从大图中，通过channels 来提取特征
池化 = max or min or avg 来降维度
全连接 = 大量计算


一个关键的数据复用RS = row stationary 是最好的
![](../../assets/Pasted%20image%2020260513153052.png)

![](../../assets/Pasted%20image%2020260513153137.png)
![](../../assets/Pasted%20image%2020260513153119.png)

DNN 的能耗很大一部分来自数据搬运，尤其是 off-chip DRAM，而不是 MAC 本身。

spatial architecture 已经是在把 on-chip memory 放近 PE；near-data processing 更进一步，希望把计算靠近高密度存储，甚至放进 memory/sensor 里。

最后讲了一些算法-硬件协同优化
1. 降低精度，32->16->8 -> 4, 相同带宽下搬运数据更多了
2. 减少操作数，剪枝，MOE

DNN 加速器的本质不是“做很多 MAC”，而是围绕 DNN layer shape 设计 memory hierarchy + dataflow + precision/sparsity 支持，让有效计算尽量发生在靠近数据的位置，并且用完整 benchmark 指标证明 accuracy、latency、energy、area 的综合收益。