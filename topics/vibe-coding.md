https://guibesdev.medium.com/your-codebase-is-the-new-prompt-architecture-for-the-ai-era-8ad33d319489

这个说的挺好
决定AI 输出质量的，是你的代码块，而不是提示词。
AI 是一个放大器，会放大已经存在的东西

1. 深模块 优于  浅模块， 最好只暴露一点接口出来
2. 垂直切片 >  水平分层， 用功能文件夹
3. 拒绝大于500行的大文件
4. 类型 能阻止AI 产生幻觉
5. 规范，而非代码，是新的源代码，是基本单元-- grill me 来减少探索成本

如果我要把一切压缩成一个心理画面：

spec → types → tests → small files → vertical slices → AGENTS.md → ADRs  
  ↓       ↓        ↓        ↓               ↓               ↓        ↓  
  spec   shape   behavior  context cost  irrelevance   per-module   architectural  
  grader grader  grader    low           low           context      memory

https://github.com/mattpocock/skills
安装了一下这个skills 试试
感觉 CONTEXT.md 还挺重要，防止AI 一段时间后偏离之前的概念了
然后grill-me 也挺好，强迫自己让AI修改之前都理解清楚了
最后的TDD, DDD 也挺重要，此外多写垂直切片更好！

最关键：不要让AI 随便写代码，一种不好的代码风格，会不断污染后续的代码生成，让拉屎拉的越来越多！！！所以一定要写之前三思，同时，需要一段时间后来重构一轮代码！

关键skills:  Improve Codebase Architecture, grill me

https://schristoph.online/blog/software-fundamentals-matter-more/


今天又看了泽文 GTOC 公众号的内容，感觉还是挺有启发的
1. 再 生成 plan 之后，试着让AI 给自己出些题目，看看自己对AI 的plan, 各个约束度是否理解清楚了，coach mode, 不是给答案，而是不断互动，其实这个和 /grill me 还挺像的，也是确保自己的想法，在AI 生成代码之前，确保自己和它都理解并且对齐清楚了。
2. 传统学习方法
	1. 系统性理解一个概念，依赖自己检索能力，耗时长，但更能形成系统
	2. 从一个点切入，在解决过程中不断补齐相关知识面，适合AI 协同项目，多试试
3. 还可以再/goal 执行过程中，不要干别的项目，而是新开会话，试着反复追问它为何要做这些事情，有什么权衡，该如何学习理解它做的内容。
	1. 主执行线程，不断写代码，沉淀执行类skills
	2. 问答线程，不断学习文档，沉淀解释类和调试类skills，形成人类文档
	3. new: 流程规范skills, 写commit message, coding style, 目录形式，review 习惯等

