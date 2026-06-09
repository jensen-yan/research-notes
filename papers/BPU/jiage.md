https://jia.je/cpu/comparison.html#l1-icache-itlb
先根据这个做个简单分类
1. amd： zen1, zen5
2. apple: M1, M5
3. arm: n2,v2, n3,v3
4. intel: golden, redwood, lion, sky

看BPU
amd 基本都有8k 左右的btb，并且之前是L3 级
苹果似乎使用了L1 ICache

然后基本都是8way 32KB or 64KB 的L1 ICache

ROB 基本最大也就1k
LSU 基本是3load 2store 这种，
甚至load 能支持64b，128b，256b

ALU 基本都是4-6个
但是FP/VEC 基本都能4 * 256b， 非常大类


	tage 这里还挺小的啊，intel 这些居然才3table，22,58,186， 4way， 2kentry，
v2 的最像我们， 8table，2way， 1k entry， 16k
