
现在实现了简单的Next line 预取器
核心设计：可控，可解释，可避让，宁可不预取也不乱污染
再顺序流很好，再乱序流中可能出现严重污染不太好
需要判断
1. 当前load miss 发生且开启预取
2. 下一行还不在Cache 中
3. 目前还没有别的正在预取的
4. MSHR 还不满
5. 顺手把下一行内容预取过来吧，并且标记这一行为Prefetch
6. 如果后面命中prefetch 行则改为normal 行；否则可能优先踢掉

紧急避让: 如果MSHR 还在预取未完成，正常load 请求来了，就把在预取的删除了
可能会专门区分普通MSHR 和预取MSHR

[预取交互](/Users/yanyue/workspace/claude-test/claude-first/risc-v-simulator/docs/prefetcher_interactive_guide.html)

[预取分类](/Users/yanyue/workspace/claude-test/claude-first/risc-v-simulator/docs/advanced_prefetchers_guide.html)

1. 时间预取：历史会重演：如果以前出现过A 后续访问B, 那么下次A 发生后都立刻把B 给取出来，例如图、指针链表等。
2. 空间预取：邻居可能有用：如果访问了当前块，大概率邻居也会被访问，例如Next-line, stride, SMS, 适合一维数组，结构体等连续地址空间

例如
1. next-line
2. stride:  pc + stide： 出现了几次（当前地址-上次地址= stride 定值）confidence ++
3. BOP(Best Offset prefetch): 空间：动态测试一组便宜，然后选一个最优offset 来预取（例如+1, +3, +5 等候选，多次有+3， 之后选+3 来预取）
4. SMS(Spatial Memory Streaming)：空间，以4K 页粒度，发现访问页1 时候经常0,4,8 这样访问，当访问页2 时候，就把4，8 一起预取出来，认为下一页和这一页的访问模式类似。 
5. berti(基于延迟关联)：时空一体：根据miss 的物理时间延迟计算任意两行访存的共生相关度， line A miss 后X 拍 line B 也一定miss, 则A-> B 相关，之后A 会预取B.