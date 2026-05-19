

## 基本信息

- 论文/文章 题目: Attention Is All You Need
- 链接: https://arxiv.org/abs/1706.03762  https://arxiv.org/html/1706.03762
- 作者：[Ashish Vaswani](https://arxiv.org/search/cs?searchtype=author&query=Vaswani,+A),
- 日期：2017
- 领域：
- 标签：#paper 

---

## 1. 一句话总结

这篇文章最核心想说的是：

> 传统的DNN,CNN 不行了，提出了新的Transformer 架构，用self-attension + Feed-forward Network + positonal Encoding 来组成架构，这是一切LLM 基础，也是硬件设计基础

---

## 2. 原文关键内容

### 背景问题
难以并行化计算，传统都是顺序依赖的。
现在改成常数增长，对应的再还是用多头机制来提升进度。
- **RNN / LSTM / GRU**
    - 优点：天然处理序列顺序。
    - 缺点：必须按 token 顺序一步步算，训练时难以并行。
    - 长距离依赖路径长，信息从前面传到后面要经过很多步。
- **CNN / ConvS2S / ByteNet**
    - 优点：比 RNN 更容易并行。
    - 缺点：局部卷积需要多层堆叠才能连接远距离 token。
    - 长距离依赖路径虽然比 RNN 短，但仍不是常数级。
都是长依赖问题，难以并行训练和更新

transfomer: 让序列任意两个位置能直接交互，并且每一层都能高度并行
### 核心方法 / 核心观点

#### 框架
经典的 **Encoder-Decoder** 架构：

```
输入序列 → Encoder → 中间表示 → Decoder → 输出序列
```
先编码到隐式层，然后再变回人类语言
论文里的 base model：encoder,decoder 各6层

![471](../../assets/Pasted%20image%2020260514144742.png)
#### 直觉理解：
**Attention 负责 token 之间交换信息，Feed-Forward 负责每个 token 内部加工信息。**
Attention 去和别人交流收集上下文，FFN 来自己消化一下
然后attention 类似查表，把Key, Query 差Value , 查出对应信息
类比：大概对CNN卷积有个抽象理解，相当于用channel 去抽取图像的不同细节，例如鼻子or 嘴巴，不同channel 抽取细节不同。然后再通过池化和全连接来降维这种.

每个encoder:
1. multi-head self-attention
2. position-wise feed-forward network, 全连接
3. 残差 + layerNorm（方便训练收敛）

每个decoder:
1. masked multi-head self-attention （多了一层）
2. multi-head self-attention
3.  position-wise feed-forward network
4. 残差 + layerNorm
> 这里masked 防止预测i 时候，不能看未来token, 只能看自己和之前的


#### 核心：Scaled Dot-Product Attention
input:
- Query: 我要找什么信息，是当前单词的表示形式，一般关注当前token 的query.
- Key：  是序列中所有单词的标签，是我们找单词的对照物
- Value： 是单词的实际表示
有点像你拿着query 便签，去文件柜中找到对应key 的value 内容。自注意力层会产生每个文件夹的相关度分数。
![](../../assets/Pasted%20image%2020260518182853.png)
> 注意这个注意力分数是动态算出来的，不是在模型参数中的。模型参数是被训练为：如何从当前token 表示抽出key or  注意力分数。注意力分数 = Query * Key, 其中某个key_i = input i * Wk

![](../../assets/Pasted%20image%2020260518184049.png)
然后把每个token 的value * 每个相关度分数/ 注意力分数，求和作为这一层输出

公式是：
```
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```
#### QKV直观理解：
1. Q 和 K 做点积，得到“相关性分数”。
2. 除以 sqrt(d_k)，防止点积过大导致 softmax 饱和。
3. softmax 得到权重。
4. 用权重对 V 加权求和。
这里的关键点是：**每个 token 可以直接看所有 token。**

eg: 句子：
```
The animal didn't cross the street because it was tired.
```
当模型处理 `it` 这个 token 时：
```
Query(it)：我是谁？我指代谁？
Key(animal)：我是一个可能被指代的实体
Value(animal)：animal 的语义信息
```
于是 `it` 可以高权重关注 `animal`，把 animal 的信息读过来。
所以相当于要查KV Cache, 这里存储了输入计算出的K, V 中很多全局信息，然后Query 有点类似查字典，从每层存储好的KV 中查出自己内容（查出多项按照概率聚合）

为何不一起存储？
三个角色不一定应该用同一套特征。
比如 `animal`：

```
Key 可能突出：  “我是一个名词实体”  “我是可被代词指代的对象”
Value 可能携带：  “animal 的语义内容”  “它是 cross 动作的主体”  “它可能对应后面的 it”Query 则是当 animal 自己作为当前位置时，它要去找别的信息。
```

所以 Q/K/V 分开投影，本质是给模型更大的自由度。

补充：
d_model = 512, 一个token 用512 长的向量来表示
d_k key 向量长度， = 64， 8个head, 每个head 分别处理64维子空间

#### multi-head: 为何要多头？
单attention: 一次注意关注一种关系
多attention: 多次注意关注多种关系

#### 如何关注token 顺序？ 位置编码
它没有RNN 这种循环卷积，通过添加  位置编码 Position Encoding, 随着input 一起进入
input = token embeding + position encoding
参考 https://blog.csdn.net/qq_36667170/article/details/124336971  图解位置编码
最简单想法1：为每个位置添加[0, 1] 范围数字，0=start, 1=end, 但是句子太长间隔太短。
想法2：1=单词1， 2=单词2， 固定步长增加，但是数字会变得很大，并且模型可能遇到比训练时候更长的句子。
希望：每个时间步编码相同，并且两个时间步 距离相同，不受句子长度影响。

作者方案：用d 维向量，不在模型中，而是附加到输入token 中
PE 采用固定的 sin/cos 位置编码：
```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```
这么做的好处是：
- 不引入额外可学习参数。
- 不同维度对应不同频率。
- 理论上可以外推到比训练时更长的序列，更好可扩展

直观理解：如果用2进制表示数字，那么pos0 每2个翻转一次，pos1 每4个翻转，pos2 每8个翻转
而用正弦函数，某个位置也是翻转。用多个不同频率的sin,cos 组合表示一个token 位置。（不同频率的sin 函数可以表征不同的位置范围，例如高频区分细粒度相邻token, 低频表示长粒度的）


#### 为何适合并行？
1. Self-Attention 并行度更高
	1. 所有token attention 可以一次性矩阵计算，不依赖上一拍
2. 长距离依赖路径短
	1. 任意2个token 一层就能交互，最大路径是O(1)
3. 短中序列复杂度 是O(n^2 d), n 少好，n 很大会成为瓶颈


---

## 3. 我的个人提炼

### 我现在怎么理解它？

用自己的话讲：

- 本质上把大语言模型，从依赖时间序列的计算，变成了大规模矩阵运算，更有利于加速器并行！
- 这里关键是position encoding, 把*序列位置信息*和计算执行顺序分离了，用位置编码显示注入顺序，所以允许所有token 在一层内并行计算。
- 主要是把prefill 那个encoder 阶段加速很大，
- 可惜decoder 阶段还是自回归一个一个输出，难以并行

insight: 突然觉得这个全局注意力机制，还有点像TAGE 的各种历史，偶尔需要短历史，偶尔需要长历史，偶尔还能用CNN 方式，把中间历史删除避免干扰。

注意：现在GPT 架构没有左边encoder 了，只有decoder-only Transformer! 且取消了cross-attension
两个阶段是

| 阶段      | 做什么                        | 是否能并行 token |
| ------- | -------------------------- | ----------- |
| Prefill | 一次性处理完整 prompt，建立 KV cache | 可以较大程度并行    |
| Decode  | 每次生成一个新 token，并追加 KV cache | 时间维度串行      |

所以更准确说：
prefill 并行性来自输入token 都已经知道


---

## 5. 我还没搞懂的问题

- [ ] transfomer 之前的RNN, 等
- [ ] 大模型GPT 和Transformer 的区别，没有encoder
- [ ] 

---

## 6. 后续行动

- [ ] 查：FlashAttension, PagedAttention/vLLM 等论文理解 KVCache 等术语
- [ ] 对比：
- [ ] 整理到主题文档：


https://www.zhihu.com/question/445556653/answer/3254012065
这个回答也还可以

### 和现代LLM 差异
先回顾下原始encoder 作用
我之前错误理解：需要先把一段人类语言 分词 变成token, 然后每个token映射为一段数学向量，相当于映射到一个隐式空间中。这个都是encoder 之前的工作，不是encoder 自己的工作。
encoder 核心是：每个输入token 都能看见完整输入序列，形成上下文表示
decoder 核心：每个token 只能看到自己左边的token, 用来做自回归生成。
相当于：读入源语言-> 构建源语言记忆 -> 逐步翻译目标语言。
相当于decoder query 去查encoder 输出的key/value

现代LLM:
```
input： prompt tokens + 已生成tokens
embedding/ position encoding
decoder-only transfomer + causal masking
输出下一个token 概率分布
采样、选择下一个token
```
相当于在一长串上下文中，每层重写每个token 上下文表示，并不断续写

为何要KV Cache?
生成token t 时候，当前query 会和前面所有token 的key 做匹配，然后加权汇聚前面所有value, 而之前所有token 的K/V 不会变，所以把每一层，每个历史token 的K/V 缓存复用

#### 什么是自注意力/self-attention?
假设我们要翻译下边这句话：  
`”The animal didn't cross the street because it was too tired”`
这里`it`指的是什么？是`street`还是`animal`？人理解起来很容易，但是对算法来讲就不那么容易了。
自注意力就是：看输入位置中其他位置的单词，试图寻找一种对当前单词更好的编码方式。
RNN 是把之前的隐藏状态 和 当前位置的输入 结合；
Transformer 是 把对其他单词的“理解” 嵌入到我们当前单词中，相当于对输入句子每个单词打分，分数决定我要对其他位置的单词给予多少关注度。
注意力分数= query * key  点积
然后让每个value  * 这个注意力分数，把其他不相干的单词扔掉


#### GPT2 介绍
https://lolitasian.blog.csdn.net/article/details/125529598    图解GPT2 翻译！
GPT2 不同规模，主要差异在decoder 层数不同（from 12 to 48 层）
GPT2 就是使用的transfomer 的decoder 模块；BERT 使用的encoder 模块
GPT2 会吧每个输出token ， 添加到输入token, 作为新序列来生成下一个token, 这是自动回归（BERT 没有）
此外，自注意力层有个关键差异：只关注之前的内容，把后面内容mask 了，不参与到自注意力计算。
![](../../assets/Pasted%20image%2020260518181311.png)
Transformer的自注意力和GPT-2的屏蔽式自注意力之间是有明显的区别的。一个正常的自注意计算在计算某位置的时候允许模型关注其右边的信息，屏蔽式自注意则不能关注到右侧信息：
此外，也把Transformer 的decoder 组件中的第二个自注意力层给去掉了。

此外，如果一直选择第一个token, 可能重复，该top-k 可以从k 个候选结果选其他的那个

embbding 矩阵也是预训练一部分，每行表示一个词语，标志一些含义信息。
![](../../assets/Pasted%20image%2020260518182119.png)
位置编码矩阵也是GPT-2模型的一部分，它包含输入中1024个位置的每个位置编码向量。
输入 = prompt 的embbding 矩阵 + PE 矩阵



相关资料：
https://jalammar.github.io/illustrated-gpt2/

https://blog.csdn.net/qq_36667170/article/details/124359818  图解transformer 好文！
https://github.com/karpathy/minGPT