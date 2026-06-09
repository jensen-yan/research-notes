https://github.com/simplex-micro/riscv-vector-primer/blob/main/chapter-02.md

用一个512bit 的向量协处理器来举例（不是紧耦合到CPU 中）
关注
1. 可配置：在build决定，运行时不会修改，eg： L1C size， VLEN
2. 可扩展：可以在基础ISA 添加扩展指令
3. 可编程：软件在运行时决定：SEW，LMUL，VL， 通过向量配置指令+CSR来做

这里是CPU + VPU + 共享缓存，其中VPU有自己标量流水线？
512 VLEN， 4lane，
每个lane128，一组ALU，FPU等，loadstore 单元

第二节没讲啥

https://github.com/simplex-micro/riscv-vector-primer/blob/main/chapter-03.md

三种访存
1. unit-stride：连续
2. strided：固定偏移
3. indexed：离散
高性能实现来能支持不同内存（缓存，非缓存，本地向量存储）多个并发load store 操作，来隐藏内存延迟。

分bank来减少读写冲突，时分复用，pattern 感知预测来减少冲突

LMUL 可以加宽or 收窄同时使用多个寄存器
1. 加宽vwadd，只需要拆分为两个独立未操作
2. 收窄vnsr，结果是输入的一般，使用带有缓存区写回的专用流水线？

index load 缺乏规律性，临近元素可能一拍访问同一个bank，导致重放、串行化、多周期
1. bank interleaving：地址分散到多个bank
2. 重放队列在不暂停流水线重新发出冲突访问
3. 让向量寄存器内部重新排列数据

掩码效率
1. agnostic，vma，vta：允许忽略，减少对旧数据不必要读取
	1. vma：vector mask agnostic
	2. vta: vector tail agnostic
2. undisturbed，vum，vtu：读取之前值保留非活跃元素，增大端口使用
无寄存器重命名的是相同的，有rename的会区分
> 注意 无rename 是一些矩阵/向量的简单实现方法，应为要乱序回退一个大寄存器太难了，
> 传统： rename-> issue -> execute -> writeback -> commit
> new: decode/dispatch -> in ROB -> commit -> VPU queue -> execute
> 通过计分板来记录数据依赖性。某个ready or busy

香山使用有rename的，那么保留旧值需要读取就物理寄存器，会增大读写端口冲突

CSR 寄存器
1. 算数CSR：管理饱和标准和舍入模式： 复用定点和浮点的
2. 配置CSR：vl，vtyep
vector 配置推测执行
如果csr写安装传统的串行，新值生效前必须排空流水线，那么当vl， vtype 频繁变化时候会性能很差，所以需要推测vl，vtype等，
vtype 配置vma，vta，SEW，LMUL 等

![Pasted image 20260608162806](../../assets/Pasted%20image%2020260608162806.png)

LMUL 分数，主要用于类型扩展，把8bit 扩展为16bit 情况


![Pasted image 20260608163307](../../assets/Pasted%20image%2020260608163307.png)
这里向量mask
VLEN=512，SEW=64，那么有8个元素，每个reg需要8bit来做mask
而LMUL=8 时候，用低64bit做mask了
SEW=8，每个reg有64个元素，所以低64bit作为v0 的mask，最多支持LMUL=8 的mask
> 这个v0 使用独立寄存器而不是编码到指令中，确实感觉挺浪费的，很可能只用低几bit，确占用这么大个寄存器，而且比较耦合。不如ARM SVE

`vsetvli` 指令使用立即数来配置 vtype。汇编语法清晰明了："e8" 表示 SEW=8，"m4" 表示 LMUL=4，"mf4" 表示 LMUL=1/4。
所有向量配置，SEW，LMUL，VL；都通过vsetvli（立即数更新) ,vsetvl (寄存器),vsetivli (中间立即数) 来控制。VL和Vtype 同步更新

stripe mining： 让硬件自动处理剩余的VL
first-fault load: 假如一条向量load 部分数据后出现fault，不会立刻停止，而是把VL 设置为fault之前的长度，这样半截数据仍然可以继续处理了。

区别
1. complete：执行完成
2. commit：变成非推测状态，且无法被取消（前面的分支和控制流都确定了不会BP wrong而取消
3. retire：结果被写入架构寄存器，寄存器释放可以被重命名
普通乱序：先complete，然后同时commit+retire
向量：先commit然后同时complete+retire
并且后者也可以做乱序，只要维持好依赖关系就行

vstart 用于向量处理
例如对32个元素，第四个元素page fault，vstart=3， 并处理异常，恢复后只需要处理元素3到31
vlenb = vlen / 8, 只读，
