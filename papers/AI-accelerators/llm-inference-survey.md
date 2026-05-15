

## 基本信息

- 论文/文章 题目：A Survey on Efficient Inference for Large Language Models
- 链接: https://arxiv.org/abs/2404.14294
- https://arxiv.org/html/2404.14294v3
- 作者：
- 日期：2024
- 领域：
- 标签：#paper 

---

前置论文：[transformer](transformer.md)
## 1. 一句话总结

这篇文章最核心想说的是：

> 总结LLM 推理效率优化地图。LLM 满，贵，消耗显存大。
> 根因：模型大，attention 随着上下文二次增长，自回归decoding 难并行
> 优化：数据、模型、系统

---

## 2. 原文关键内容

### 背景问题

- LLM 推理分成两个阶段

| 阶段          | 做什么                                        | 主要瓶颈                   | 对加速器的含义                               |
| ----------- | ------------------------------------------ | ---------------------- | ------------------------------------- |
| **Prefill** | 处理输入 prompt，计算并存储初始 KV cache，同时生成第一个 token | 长上下文 attention 计算和显存访问 | 更像大矩阵/批量计算，比较容易并行                     |
| **Decode**  | 每次基于 KV cache 生成一个新 token，并更新 KV cache     | 自回归逐 token、batch 小、访存重 | 难以填满计算阵列，更像 GEMV/小 batch GEMM + KV 访存 |
![](../../assets/Pasted%20image%2020260514164448.png)
论文明确说：
prefill 计算并保存输入token 的KV Cache;
decoder: 逐token 输出，并持续更新KV Cache

定义：
* first-token latency: prefill 生成第一个token 延迟
* per-token latency： decode 生成token 延迟
* token throughput： 每秒生成token 数目
* request throughput： 每秒完成请求数目
* 峰值内存 =  模型权重 + KV Cache 大小

难点： 在预填充阶段，自注意力操作在输入长度上表现出**二次计算复杂度**。
因此，随着输入长度的增加，注意力操作的计算成本、内存访问成本和内存使用量会迅速上升。
原因：输入N, 那么QK 举证计算也是N* N 的矩阵计算，表示token i 要关注周围N 个关系，N 个token 就是N * N 组关系。 计算大致二次增长，显存占用通过FlashAttension 能降低一点，不是N * N 增长
FlashAttension: 不把完整N * N attention matrix 存入HBM, 而是分块算，变算边规约

prefill: 计算二次增长
decode: 计算线性增长，但kV Cache 线性增长，内存瓶颈大
![](../../assets/Pasted%20image%2020260514173742.png)

![](../../assets/Pasted%20image%2020260514164851.png)
内存占用：逐渐增大

### 核心方法 / 核心观点

- 推理优化方法分成三层：

| 层级               | 优化对象             | 典型方法                                                                           | 是否改模型     |
| ---------------- | ---------------- | ------------------------------------------------------------------------------ | --------- |
| **Data-level**   | 输入/输出组织          | prompt 压缩、RAG、Skeleton-of-Thought                                              | 通常不改模型    |
| **Model-level**  | 模型结构和数据表示        | MoE、MQA/GQA、量化、剪枝、蒸馏、SSM/Mamba                                                 | 通常会改模型或精度 |
| **System-level** | 推理引擎和 serving 系统 | FlashAttention、speculative decoding、offloading、continuous batching、KV cache 管理 | 通常不改模型能力  |
![](../../assets/Pasted%20image%2020260514165052.png)
#### Data 
输入压缩：优化prefill, 别把所有东西都塞入prompt, RAG or 记忆优化， 提示总结等
> 现在我也经常面对啊，所有有很多上下文压缩，精简AGENTS.md 等
> 计算会随着输入长度n 而平方增加！

输出组织：把decode 更并行
介绍了 Skeleton-of-Thought、SGD、APAR、SGLang 这类方法。
核心思路是：先让模型生成一个“骨架”或 DAG，再把独立子问题并行扩展。SoT 的第二阶段可以把多个 point-expanding 请求 batch 起来，从而提升硬件利用率；
SGLang 引入了一种基于 Python 的领域特定语言(DSL)，其特性原语能够灵活地促进 LLM 编程


#### Model

> FFN 挺占权重的，llama-7b 中，FFN 占据了全部参数的70%
> 注意力操作

有效结构优化：三类
1. FFN or MoE: 每次只激活部分token; but: routing , 负载均衡等问题
2. Attention 优化：多个attention共享KVCache
3. Transformer 替代：用线性复杂度建模长序列，难

model 压缩：
量化、稀疏化、结构分解、蒸馏、动态推理

#### System

1. Operator kernel 优化：
	1. 推理的 runtime 主要被 **attention operator 和 linear operator** 主导；系统优化重点就是围绕 attention、linear、decode 特性做专用 kernel、tiling、fusion。
2. Speculative decoding:
	1. 用小draft model 先猜测多个token, 然后大模型一次性验证
3. Serving System: 多个用户，暂时不关注


一个趋势：**LLM 专用硬件不能只做“大矩阵乘法阵列”，还必须照顾 decode 阶段的 KV cache、低 batch、稀疏/量化、长上下文和 serving 调度。**

### 关键图表 / 关键证据

- 图/表：
- 它说明了：

### 作者的结论

- 

---

## 3. 我的个人提炼

### 我现在怎么理解它？

用自己的话讲：

- 

### 这篇内容真正重要的点是：

1. 
2. 
3. 

### 我觉得作者说得最有启发的是：

- 

### 我不完全认同 / 还存疑的地方：

- 

---

## 4. 和我当前工作的关系

### 对 AME / IME 的启发

- 

### 对 gem5 建模的启发

- 

### 对 LLM 推理 / AI 加速器分析的启发

- 

### 可能能写进汇报 / 周报 / PPT 的观点

- 

---

## 5. 我还没搞懂的问题

- [ ] 
- [ ] 
- [ ] 

---

## 6. 后续行动

- [ ] 查：
- [ ] 对比：
- [ ] 整理到主题文档：