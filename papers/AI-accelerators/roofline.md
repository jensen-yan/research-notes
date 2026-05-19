
https://people.eecs.berkeley.edu/~kubitron/cs252/handouts/papers/RooflineVyNoYellow.pdf

Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures

核心是在说：用一个简单二维图，把程序性能受限于“算力”还是“内存带宽”这件事讲清楚。

Roofline 模型用
“Operational Intensity（操作强度，Flops/DRAM Byte）”作为横轴，
“可达到性能 GFlops/s”作为纵轴，
通过峰值算力水平线和内存带宽斜线，判断一个 kernel 是 compute-bound 还是 memory-bound，并进一步指导该优先做哪类优化。

一般横轴 = 算力 / 内存带宽 = 每从 DRAM 传输 1 byte，能完成多少次操作。得到一个当前算法下的操作强度，越大越好，说明从内存中搬运一次数据，能计算的很久，而不会频繁写回到memory 中，说明对这个数据时间复用性比较高（受到算法和对应的cache 影响？）

纵轴= 最终性能，越高越好

**ridge point**，也就是斜屋顶和平屋顶的交点。
> 想跑到峰值算力，kernel 至少需要多高的 Operational Intensity。

如果 ridge point 很靠右，说明这台机器算力很猛但内存跟不上，只有非常高 OI 的程序才能吃满算力，很难实现。
如果 ridge point 靠左，说明机器比较“均衡”，普通 kernel 更容易达到较高性能。

![](../../assets/Pasted%20image%2020260513154754.png)
这里的x4 相对x2，主要是算力提高了4倍，但是内存带宽没有变化。
![](../../assets/Pasted%20image%2020260513154829.png)

常见 compute ceilings：
- 提高 ILP
- 使用 SIMD
- 平衡浮点 add/mul 比例
- 使用 FMA

常见 memory ceilings：
- unit-stride 访问
- memory affinity，避免跨 socket 访问远端内存
- software prefetch
- 减少 conflict/capacity miss
- no-allocate store / streaming store


- 理论算法有一个“理想 OI”；
- kernel 写法、blocking、layout、reuse、prefetch、DMA、cache policy 会改变实际 OI；
- 芯片的 cache/local memory/DRAM 层次决定你统计哪个层级的 byte；
- 因此不同芯片上，同一个 kernel 的有效 OI 可能不同。