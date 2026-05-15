
https://arxiv.org/pdf/1704.04760
In-Datacenter Performance Analysis of a Tensor Processing Unit, 2017, google

https://courses.grainger.illinois.edu/cs533/sp2025/notes/tpu_arch.pdf


思考：脉动阵列的设计原理
以矩阵乘法为例：

```
C[i][j] = sum over k of A[i][k] * B[k][j]
```
脉动阵列很自然地把二维 PE 阵列对应到 `C[i][j]` 这个二维输出平面：

```
PE(i, j) 负责 C[i][j] 的一部分或全部计算
```

然后：

```
A[i][k] 沿着一行往右传B[k][j] 沿着一列往下传每个 PE 做 acc += A * B
```

于是一个 `A[i][k]` 被同一行多个 PE 复用，一个 `B[k][j]` 被同一列多个 PE 复用。
这就比“每个 PE 都从中央 SRAM 读 A/B”省很多数据搬运。
所以脉动阵列的关键直觉是：
> **数据不是被“广播到所有 PE”，而是像水波一样在 PE 阵列里逐拍传播，边走边被消费。**

https://zhuanlan.zhihu.com/p/650209037  
https://zhuanlan.zhihu.com/p/26522315
这个文章讲的比较好
