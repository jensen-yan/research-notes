https://eupilot.eu/wp-content/uploads/2023/03/Roger-Ferrer-ACM-2022_RISC-V-VectorExtension_v2.pdf

现代向量指令集都很复杂，并且包括predicate/MASK 都编码到向量指令中
例如比较指令先生成mask，然后用这个mask 去执行部分元素操作
默认用v0， 很丑。ARM SVE 用一组P 寄存器，intel 用K 寄存器
vsetvli

v s e t v l i   r d , r s , eN, mX, tP, mP
rs 是输入寄存器操作数，表达程序想要的向量长度AVL（application vector length）
其中N 是SEW，8，到64等
X 是LMUL
P 是尾部t和掩码m 的策略，u为undisturbed，a为agnostic
eg：
v s e t i v l i  x10, 3 , e32,m1,ta,ma
vl=3， SEW=32， LMUL=1， tail mask agnostic
vsetvli t0, x0, e32,m1,ta,ma
vl=0=VLMAX=128/32 = 4

注意，当AVL=9，VLMAX=8
传统上会处理8 + 1 来两拍完成，但会负载不均衡
所以运行 5 + 4， or 6 + 3 这种，这是不确定的

编译器很喜欢吧程序理解为虚拟寄存器之间的数据流动（每个值都有一个生产者和若干消费者），但是ISA 中包括一些全局状态寄存器，vl，vtype这种会让编译器很难受
