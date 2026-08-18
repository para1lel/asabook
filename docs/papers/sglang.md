---
title: 'SGLang: Structured Language Model Programs'
createTime: 2026/08/18 12:00:00
permalink: /papers/sglang/
pageClass: paper-reading
---

> [Lianmin Zheng](https://lmzheng.net/) [+equal-contribution], [Liangsheng Yin](https://www.lsyin.me/), [Zhiqiang Xie](https://zhiqiangxie.com/), [Chuyue Sun](https://dblp.org/pid/334/0722), [Jeff Huang](https://dblp.org/pid/68/4706-1), [Cody Hao Yu](https://comaniac.github.io/), [Shiyi Cao](https://shiyicao.com/), [Christos Kozyrakis](https://kozyraki.github.io/), [Ion Stoica](https://people.eecs.berkeley.edu/~istoica/), [Joseph E. Gonzalez](https://people.eecs.berkeley.edu/~jegonzal/), [Clark Barrett](https://cs.stanford.edu/~barrett/) 和 [Ying Sheng](https://sites.google.com/view/yingsheng/home) [+equal-contribution]. 论文于 2023 年 12 月 12 日首次提交至 arXiv, 当前版本为 v2. 后发表于 [Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Main Conference Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html). [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104). [原始 PDF](/paper/sglang.pdf). [DOI](https://doi.org/10.52202/079017-2000). [TeX 源码](https://export.arxiv.org/e-print/2312.07104). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

大语言模型 (LLM) 越来越多地用于复杂任务, 这些任务需要多次生成调用, 高级提示技术, 控制流以及结构化输入/输出. 然而, 目前缺少能高效编写和执行这类应用的系统. 我们提出 SGLang, 一个用于高效执行复杂语言模型程序的系统. SGLang 由前端语言和运行时组成. 前端提供生成与并行控制原语, 简化程序编写. 运行时通过 RadixAttention 等新优化加速执行, 前者用于复用 KV cache, 同时还利用压缩有限状态机加快结构化输出解码. 实验表明, 在多种大语言模型和多模态模型上, 面对智能体控制, 逻辑推理, few-shot 学习基准, JSON 解码, 检索增强生成流水线和多轮对话等任务, SGLang 的吞吐量最高可达到当前先进推理系统的 $6.4\times$. 代码已公开于 [https://github.com/sgl-project/sglang](https://github.com/sgl-project/sglang).

<span id="section-01"></span>

## 1 引言

近来 LLM 能力的提升拓宽了其用途, 使其能够处理范围更广的通用任务, 并作为自主智能体运行 [Ope23, Bub23, Par23, Wan23d, Sum23a]. 在这类应用中, LLM 会进行多轮规划, 推理, 并与外部环境交互. 这些能力通过工具使用 [Sch24, Pat23b], 多种输入模态 [Gem23, Ala22] 和众多提示技术 [Liu23l] 实现, 例如 few-shot 学习 [Bro20b], self-consistency [Wan22g], skeleton-of-thought [Nin24] 和 tree-of-thought [Yao23]. 这些新用例都需要多次且往往相互依赖的 LLM 生成调用, 反映出使用多调用结构完成复杂任务的趋势 [Yao22b, Kim23].

这些模式的出现标志着我们与 LLM 的交互发生变化: 从简单对话转向更复杂的程序化使用方式, 即用程序调度和控制 LLM 的生成过程. 我们把这类程序称为“语言模型程序” (Language Model Programs, LM Programs) [Beu23, Kha23]. 前述高级提示技术和智能体工作流都属于 LM 程序的范畴. LM 程序通常有两个共同属性: (1) LM 程序一般包含多次 LLM 调用, 其间穿插控制流. 这是完成复杂任务和提高整体质量所必需的. (2) LM 程序接收结构化输入并产生结构化输出. 这样才能组合多个 LM 程序, 并将它们集成到现有软件系统中.

尽管 LM 程序已经得到广泛使用, 现有系统仍无法高效表达和执行它们. 我们发现, 高效使用 LM 程序主要有两项挑战: *第一, LLM 具有非确定性, 因而编写 LM 程序既繁琐又困难.* 开发 LM 程序通常需要大量字符串操作, 对提示进行实验性调试, 处理脆弱的输出解析, 支持多种输入模态, 还要实现并行机制. 这种复杂性会明显降低程序的可读性, 即使程序本身很简单 ([第 2 节](#section-02)).

*第二点更为重要, 冗余计算和内存使用使 LM 程序的执行效率低下.* 当前先进的推理引擎 (例如 vLLM [Kwo23], TGI [Tex22] 和 TensorRT-LLM [Ten23]) 已针对延迟和吞吐量做过优化, 但并不了解具体工作负载. 这让系统更通用, 更稳健, 却也会对任何给定工作负载造成明显的效率损失. KV cache 复用就是一个典型例子 ([第 3 节](#section-03)). KV cache 由生成式推理必需且可复用的中间张量组成. LM 程序通常采用批量执行, 多次不同的 LLM 调用只要共享同一前缀, 就有大量机会复用 KV cache. 但现有系统缺少实现这种复用的有效机制, 因而产生不必要的计算和内存浪费. 另一个例子是结构化输出的约束解码 (例如 JSON 模式), 此时 LLM 的输出必须遵循由正则表达式定义的特定语法规则 ([第 4 节](#section-04)). 在这些约束下, 往往可以一次解码多个 Token. 现有系统却每次只解码一个 Token, 解码速度并不理想.

<span id="figure-01"></span>

![解释器通过 SGLang 运行时执行语言原语的系统架构](./sglang/figure-01.png)

**图 1.** 系统架构: 解释器通过经过优化的运行时执行语言原语.

为解决这些问题, 我们提出 SGLang, 一种面向 LLM 的结构化生成语言 (<u>S</u>tructured <u>G</u>eneration <u>Lang</u>uage). 核心思路是系统地利用 LM 程序的多调用结构来提高执行效率. 如[图 1](#figure-01) 所示, SGLang 分为前端语言和后端运行时两部分. 前端简化 LM 程序的编写, 运行时加快程序执行. 两部分协同工作时性能更好, 也可以彼此独立运行.

我们将 SGLang 设计为嵌入 Python 的领域特定语言. 它提供生成原语 (例如 `extend`, `gen`, `select`) 和并行控制原语 (例如 `fork`, `join`). SGLang 与 Python 的控制流和程序库兼容, 用户可以直接使用原生 Python 语法轻松开发高级提示工作流. 我们为 SGLang 提供了解释器和编译器. 解释器把提示状态作为流来管理, 将原语操作提交到流中异步执行, 从而正确控制同步和程序内并行. SGLang 程序还可以经过追踪和编译, 以应用更多优化.

在运行时一侧, 我们提出了多项新优化来加快 SGLang 程序的执行. *第一项技术* RadixAttention 可以在多次生成调用间自动复用 KV cache. 现有推理引擎会在请求处理完毕后丢弃其 KV cache, 因而无法在多次调用间复用 KV cache, 执行速度会大幅下降. 我们的系统改为在基数树中维护所有请求的 KV cache, 并按 LRU 策略进行缓存. 这种方法将 KV cache 作为传统缓存来管理, 再借助基数树高效完成匹配, 插入和驱逐. 配合 cache 感知调度策略, 运行时可以高效处理各种复用模式. *第二项技术*是压缩有限状态机, 用于加快结构化输出的约束解码. 现有系统只针对下一个 Token 施加约束, 将不允许 Token 的概率掩蔽掉, 所以每次只能解码一个 Token. 我们的系统会分析约束, 构造表示该约束的压缩有限状态机. 只要条件允许, 这种方法就把一条多 Token 路径压缩为单步路径, 使系统能够一次解码多个 Token, 从而加快解码. 最后, SGLang 也支持只能通过 API 访问的模型, 例如 OpenAI 的 GPT-4. 我们为此提出*第三项技术* API 推测执行, 用于优化只能调用 API 的多调用程序.

我们使用 SGLang 实现了多种 LLM 应用, 包括智能体控制, 逻辑推理, few-shot 学习基准, JSON 解码, 检索增强生成流水线, 多轮对话和多模态处理. 测试涵盖 Llama-7B/70B [Tou23a], Mistral-8x7B [Jia24], LLaVA-v1.5-7B (图像) [Liu24n] 和 LLaVA-NeXT-34B (视频) [Zha24j], 硬件则使用 NVIDIA A10G 和 A100 GPU. 实验结果表明, 面对各种工作负载, 模型和硬件配置, SGLang 的吞吐量最高可达到现有编程与推理系统的 $6.4\times$, 对比系统包括 Guidance [Gui22], vLLM [Kwo23] 和 LMQL [Beu23].

<span id="section-02"></span>

## 2 编程模型

本节通过一个贯穿始终的示例介绍 SGLang 编程模型, 说明它的语言原语和执行模式, 并概述运行时中的优化机会. 该编程模型提供灵活且可组合的原语, 可以简化多调用工作流中的繁琐操作 (例如字符串操作, API 调用, 约束定义和并行控制).

<span id="figure-02"></span>

![使用 SGLang 实现多维作文评审程序](./sglang/figure-02.png)

**图 2.** SGLang 中多维作文评审程序的实现采用 branch-solve-merge 提示技术 [Sah23]. SGLang 提供的原语以红色显示.

**贯穿示例.** 该语言是一种嵌入 Python 的领域特定语言. [图 2](#figure-02) 展示了一个使用 branch-solve-merge 提示方法 [Sah23] 评估图像相关作文的程序. 函数 `multi_dimensional_judge` 接受三个参数: `s`, `path` 和 `essay`. `s` 管理提示状态, `path` 是图像文件路径, `essay` 是作文文本. 使用 `+=` 运算符, 可以向状态 `s` 追加新字符串和 SGLang 原语并执行. 函数先将图像和作文加入提示, 再用 `select` 检查作文是否与图像相关, 并把结果存入 `s["related"]`. 如果相关, 提示会被 fork 为三份, 从不同维度并行评估, 各自用 `gen` 将结果存入 `f["judgment"]`. 随后, 函数合并这些评审结果, 生成摘要并给出字母等级. 最后, 它按照正则表达式约束 `regex` 定义的 schema, 以 JSON 格式返回结果. SGLang 显著简化了这个程序. 如果使用类似 OpenAI API 的接口编写等价程序, 由于需要手动进行字符串操作和并行控制, 代码行数将达到 $2.1\times$.

**语言原语.** SGLang 提供控制提示状态, 生成和并行的原语, 可与 Python 语法及程序库配合使用. 具体原语如下: “`gen`”调用模型进行生成, 并把结果存入名称由第一个参数指定的变量. 它支持“`regex`”参数, 用正则表达式定义语法 (例如 JSON schema), 以此约束输出. “`select`”调用模型, 从列表中选择概率最高的选项. 运算符“`+=`”或“`extend`”向提示追加字符串. 运算符“`[variable_name]`”取得生成结果. “`fork`”创建并行的提示状态分支, “`join`”重新合并提示状态. “`image`”和“`video`”接收图像和视频输入.

**执行模式.** 执行 SGLang 程序最简单的方式是使用解释器, 此时提示被视为异步流. `extend`, `gen`, `select` 等原语会提交到流中异步执行. 这些非阻塞调用不会等待生成结束, Python 代码可以继续运行. 这与异步启动 CUDA kernel 类似. 每个提示都由后台线程中的流执行器管理, 从而实现程序内并行. 取得生成结果时则会阻塞, 直至结果就绪, 以保证正确同步. 另一种方式是将 SGLang 程序编译为计算图, 再用图执行器运行, 从而支持更多优化. 本文默认使用解释器模式, 编译器模式的结果见[附录 D](#appendix-d). SGLang 既通过自有的 SGLang Runtime (SRT) 支持开放权重模型, 也支持 OpenAI 和 Anthropic 等 API 模型.

**比较.** LLM 编程系统可分为高层系统 (例如 LangChain, DSPy) 和低层系统 (例如 LMQL, Guidance, SGLang). 高层系统提供预定义或自动生成的提示, 例如 DSPy 的提示优化器. 低层系统通常不改写提示, 而是让用户直接操作提示和原语. SGLang 与 LMQL, Guidance 类似, 属于低层系统. [表 1](#table-01) 比较了它们的功能. SGLang 更关注运行时效率, 并带有协同设计的自有运行时, 因此可以采用后文介绍的新优化. 高层语言 (例如 DSPy) 可以编译为低层语言 (例如 SGLang). [第 6 节](#section-06) 会展示如何将 SGLang 集成为 DSPy 的后端, 以提高运行时效率.

**运行时优化.** [图 2](#figure-02) 展示了三种运行时优化机会: KV cache 复用, 快速约束解码和 API 推测执行. 后续几节将分别介绍它们.

<span id="table-01"></span>

![LMQL, Guidance 和 SGLang 的比较](./sglang/table-01.png)

**表 1.** LMQL, Guidance 和 SGLang 的比较.

<span id="section-03"></span>

## 3 使用 RadixAttention 高效复用 KV cache

SGLang 程序可以串联多次生成调用, 也可以用“`fork`”原语创建并行副本. 不同的程序实例也经常共享某些公共部分 (例如 system prompt). 这些情况会在执行过程中产生大量共享提示前缀, 因而有许多复用 KV cache 的机会. LLM 推理期间, KV cache 保存前向传播产生的中间张量, 供后续 Token 解码复用. 它得名于 self-attention 机制中的 key-value 对 [Vas17]. KV cache 的计算只取决于前缀 Token. 因此, 提示前缀相同的请求可以复用 KV cache, 减少冗余计算和内存使用. 更多背景和示例见[附录 A](#appendix-a).

既然 KV cache 有复用机会, 优化 SGLang 程序时的一项主要挑战就是在多次调用和多个实例间复用它. 一些系统已经研究了特定的 KV cache 复用情况 [Kwo23, Ye24, Jur24, Gim24], 但它们往往需要手动配置, 也无法处理所有复用模式 (例如动态树结构). 因此, 大多数当前先进的推理系统仍会为每个请求重新计算 KV cache. [第 7 节](#section-07) 将讨论这些系统的限制以及它们与本文方法的区别.

本节介绍 RadixAttention, 一种在运行时自动且系统地复用 KV cache 的新技术. 现有系统会在生成请求结束后丢弃 KV cache, 本系统则把提示与生成结果的 cache 保留在基数树中, 从而高效完成前缀搜索, 复用, 插入和驱逐. 我们实现了 LRU 驱逐策略和 cache 感知调度策略, 用于提高 cache 命中率. RadixAttention 与连续批处理 [Yu22a], paged attention [Kwo23] 和 tensor parallelism [Sho19] 等技术兼容. 没有 cache 命中时, 它带来的内存和时间开销也可以忽略.

**RadixAttention.** 基数树是一种数据结构, 相比经典 trie (前缀树) 更节省空间. 普通树的边只能用单个元素标记, 基数树的边还可以用长度不同的元素序列标记, 因而效率明显更高. 本系统用基数树管理 Token 序列与相应 KV cache 张量之间的映射. 这些 KV cache 张量采用不连续的分页布局存储, 每一页的大小相当于一个 Token. GPU 内存很快会被 KV cache 填满, 因而我们采用简单的 LRU 驱逐策略, 先驱逐最近最少使用的叶节点. 叶节点先被驱逐后, 它们的公共祖先仍可继续复用, 直至这些祖先也成为叶节点并被驱逐.

在连续批处理设置中, 当前运行批次正在使用的节点不能被驱逐. 因此, 每个节点都维护一个引用计数器, 记录正在使用它的请求数. 引用计数为零时, 节点才可以驱逐. 请注意, 我们不会预先分配固定大小的内存池作为 cache, 而是让缓存的 Token 与当前运行请求共享同一内存池. 因此, 系统会在 cache 与运行请求之间动态分配内存. 当等待队列中有足够多的请求需要运行时, 系统会驱逐所有缓存 Token, 以换取更大的批量. [图 3](#figure-03) 展示了多个请求到达时基数树的维护过程. 前端解释器将完整提示发送给运行时, 由运行时执行前缀匹配和复用. 树结构存放在 CPU 上, 维护开销可以忽略. 执行 `fork` 原语时, 前端先发送前缀作为提示, 保证该前缀正确插入树中, 再发送其余提示. 这个“Frontend Hint”简化了运行时调度和匹配, 也说明了前端与运行时协同设计的好处.

<span id="figure-03"></span>

![采用 LRU 驱逐策略的 RadixAttention 九步操作示例](./sglang/figure-03.png)

**图 3.** 采用 LRU 驱逐策略的 RadixAttention 操作示例, 共展示九个时间点. 图中说明了基数树响应不同请求时的动态变化. 这些请求包括两个对话会话, 一批 few-shot 学习查询和一次 self-consistency 采样. 每条树边都有一个标签, 表示子字符串或 Token 序列. 节点按状态着色: 绿色表示新增节点, 蓝色表示当前时间点访问的缓存节点, 红色表示已驱逐节点. 第 (1) 步中, 基数树起初为空. 第 (2) 步中, 服务器处理用户消息“Hello”, 并返回 LLM 输出“Hi”. System prompt“You are a helpful assistant”, 用户消息“Hello!”和 LLM 回复“Hi!”被合并为一条边, 连接到一个新节点. 第 (3) 步中, 新提示到达, 服务器在基数树里找到其前缀 (即第一轮对话), 并复用对应的 KV cache. 新一轮对话作为新节点追加到树中. 第 (4) 步中, 新的对话会话开始. 第 (3) 步的节点“b”被拆分成两个节点, 使两个会话能够共享 system prompt. 第 (5) 步中, 第二个会话继续进行. 但受内存限制, 第 (4) 步的节点“c”必须被驱逐. 新一轮对话追加到第 (4) 步的节点“d”之后. 第 (6) 步中, 服务器收到一个 few-shot 学习查询, 处理后将其插入树中. 新查询与现有节点不共享任何前缀, 因而根节点被拆分. 第 (7) 步中, 服务器收到一批新的 few-shot 学习查询. 这些查询共享同一组 few-shot 示例, 所以我们拆分第 (6) 步的节点“e”来实现共享. 第 (8) 步中, 服务器收到第一个会话的新消息. 第二个会话的所有节点 (节点“g”和“h”) 最近最少使用, 因而被驱逐. 第 (9) 步中, 服务器收到一个请求, 要为第 (8) 步节点“j”中的问题采样更多答案, 这可能用于 self-consistency 提示. 为给这些请求腾出空间, 我们驱逐第 (8) 步中的节点“i”, “k”和“l”.

**Cache 感知调度.** 我们将 cache 命中率定义为 $\frac{\text{缓存的提示 Token 数}}{\text{提示 Token 数}}$. 等待队列中有大量请求时, 执行顺序会明显影响 cache 命中率. 例如, 如果请求调度器频繁切换互不相关的请求, 就可能造成 cache 抖动, 使命中率下降. 我们设计了一种 cache 感知调度算法来提高命中率. 在批处理设置中, 我们按照匹配前缀长度对请求排序, 优先处理匹配前缀更长的请求, 而不是使用先到先服务调度. [算法 1](#algorithm-01) (附录) 给出了采用连续批处理时 cache 感知调度的伪代码. 该算法使用最长共享前缀优先顺序. 对延迟更敏感的设置仍有可能允许有限的批次重排, 以改善 cache 复用. 我们还针对离线情况证明了下面关于最优调度的定理. [+1]

<span id="theorem-03-01"></span>

**定理 3.1.** 对于一批请求, 如果 cache 大小 $\geq$ 最大请求长度, 按深度优先搜索顺序遍历请求的基数树即可达到最优 cache 命中率. 最长共享前缀优先顺序等价于深度优先搜索顺序.

证明见[附录 A.3](#appendix-a-3). 在线情况下, DFS 顺序会被打乱, 但如[附录 A.3](#appendix-a-3) 所述, 本文调度策略在完整基数树新增的部分上仍近似保持 DFS 行为. 贪心 cache 感知调度可以实现高吞吐量, 却可能导致饥饿. 我们把它与其他公平调度方法 [She23b] 的结合留作未来工作.

**分布式情况.** RadixAttention 可以扩展到多张 GPU. 对 tensor parallelism 而言, 每张 GPU 分别维护一部分 KV cache. 各处执行的树操作相同, 因而不需要额外同步. 多 worker 的 data parallelism 见[附录 A.4](#appendix-a-4).

<span id="section-04"></span>

## 4 使用压缩有限状态机进行高效约束解码

在 LM 程序中, 用户经常希望模型输出遵循特定格式, 例如 JSON schema. 这样可以提高可控性和稳健性, 也更容易解析输出. SGLang 提供 `regex` 参数, 使用正则表达式施加这类约束; 对许多实际场景而言, 正则表达式有足够的表达能力. 现有系统会将正则表达式转换为有限状态机 (FSM) [Wil23]. 解码期间, 系统维护当前 FSM 状态, 从后继状态中取得允许的 Token, 将无效 Token 的概率置零, 每次解码一个 Token. 但只要存在一次解码多个 Token 的机会, 这种逐 Token 方法效率就很低. 例如, [图 2](#figure-02) 中的常量序列 `{"summary": "` 在普通解码过程中会跨越多个 Token, 如[图 4](#figure-04) (c) 所示. 即使每一步只有一个有效的后继 Token, 仍需经过多个解码阶段. 因此, 整个序列其实可以在一步 (即一次前向传播) 中解码. 现有系统却只能逐个解码 Token, 因为 FSM 与模型运行器彼此分离, 无法处理多个 Token, 结果就是解码速度较慢.

<span id="figure-04"></span>

![普通与压缩有限状态机的解码过程](./sglang/figure-04.png)

**图 4.** 普通 FSM 和压缩 FSM 的解码过程 (下划线 `_` 表示空格).

SGLang 构建了采用压缩 FSM 的快速约束解码运行时, 克服了这个限制. 该运行时分析 FSM, 将相邻的单一转移边压缩为一条边, 如[图 4](#figure-04) (b) 所示, 从而识别哪些地方可以一起解码多个 Token. 在[图 4](#figure-04) (d) 中, 压缩转移边上的多个 Token 能在一次前向传播中解码, 大幅加快了解码过程. 该方法也很通用, 适用于所有正则表达式. 更多背景和实现细节见[附录 B](#appendix-b).

<span id="section-05"></span>

## 5 使用 API 推测执行高效调用端点

前几节介绍的优化面向开放权重模型, 需要修改模型推理过程. SGLang 也支持只能通过 API 访问的模型, 例如 OpenAI 的 GPT-4. 但对这类模型, 我们只能调用黑盒 API 端点.

本节针对黑盒 API 模型提出一种新优化, 通过推测执行加快多调用 SGLang 程序, 同时降低 API 成本. 例如, 某个程序可能用下面的多调用模式让模型生成角色描述: `s += context + "name:" + gen("name", stop="\n") + "job:" + gen("job", stop="\n")`. 如果直接执行, 两个 `gen` 原语对应两次 API 调用, 用户需要为 `context` 支付两次输入 Token 费用. 在 SGLang 中, 我们可以为第一次调用启用推测执行, 忽略停止条件, 让模型继续生成若干 Token. 解释器保留额外生成的输出, 将它们与后续原语匹配并复用. 在某些情况下, 经过谨慎的提示设计, 模型可以高准确率地匹配模板, 省去一次 API 调用的延迟和输入成本.

<span id="section-06"></span>

## 6 评测

我们在多种 LLM 工作负载上评测 SGLang 的性能, 随后进行消融实验和案例研究, 考察各组件的效果. SGLang 使用 PyTorch [Pas19a] 实现, 并采用 FlashInfer [Ye24a] 和 Triton [Til19] 中的定制 CUDA kernel.

<span id="section-06-01"></span>

### 6.1 设置

**模型.** 我们测试稠密 Llama-2 模型 [Tou23a], 稀疏 mixture of experts Mixtral 模型 [Jia24], 多模态 LLaVA 图像模型 [Liu23f] 和视频模型 [Zha24j], 以及 API 模型 OpenAI GPT-3.5. 开放权重模型的参数量从 70 亿到 700 亿不等, 使用 float16 精度.

**硬件.** 大多数实验在配备 NVIDIA A10G GPU (24GB) 的 AWS EC2 G5 实例上运行. 7B 模型使用一张 A10G GPU, 更大的模型使用多张 A10G GPU 并采用 tensor parallelism [Sho19]. 一些补充实验在 A100G (80GB) GPU 上运行.

**基线.** 我们将 SGLang 与两类系统比较: 一类是采用各自语言和默认运行时的高层编程系统, 另一类是使用标准 OpenAI 风格 Completion API 的低层推理引擎. 除非另有说明, 我们不会开启会改变计算结果的优化, 确保所有系统计算相同的结果. 基线包括:

- Guidance [Gui22], 一种控制 LLM 的语言. 我们使用 Guidance v0.1.8 和 llama.cpp 后端.
- vLLM [Kwo23], 一种高吞吐量推理引擎. 我们使用 vLLM v0.2.5 及其默认 API server [+2].
- LMQL [Beu23], 一种查询语言. 我们使用 LMQL v0.7.3 和 Hugging Face Transformers 后端.

**工作负载.** 测试包括 5-shot MMLU [Hen20] 和 20-shot HellaSwag [Zel19] 基准. 对 MMLU, 我们解码一个 Token; 对 HellaSwag, 则使用 `select` 原语选择概率最高的答案. 对 ReAct 智能体 [Yao22b] 和生成式智能体 [Par23], 我们从原论文提取 trace 并重放. GSM-8K 问题使用 Tree-of-thought [Yao23], 提示生成使用 Skeleton-of-thought [Nin24]. 其他工作负载包括: 采用 branch-solve-merge 技术 [Sah23] 的 LLM 评审器; 使用正则表达式指定 schema 的 JSON 解码; 4 轮多轮对话, 每轮输入从 256-512 Token 之间随机采样. 多轮对话 (短) 表示短输出 (4-8 Token), 多轮对话 (长) 表示长输出 (256-512 Token); 以及 DSPy 官方示例中的检索增强生成 (RAG) 流水线 [Kha23].

**指标.** 我们报告吞吐量和延迟两项性能指标. 测量吞吐量时, 会运行足够大的一批程序实例来计算最大吞吐量, 比较每秒执行的程序实例数 (programs per second, p/s). 测量延迟时, 每次只执行一个程序且不做批处理, 再报告多个实例的平均延迟.

<span id="figure-05"></span>

![Llama-7B 模型上的归一化吞吐量](./sglang/figure-05.png)

**图 5.** Llama-7B 模型上的归一化吞吐量. 越高越好.

<span id="figure-06"></span>

![Llama-7B 模型上的归一化延迟](./sglang/figure-06.png)

**图 6.** Llama-7B 模型上的归一化延迟. 越低越好.

<span id="section-06-02"></span>

### 6.2 端到端性能

**开放权重模型上的结果.** 延迟和吞吐量结果见[图 5](#figure-05) 与[图 6](#figure-06). SGLang 的吞吐量最高提高 $6.4\times$, 延迟最多降低 $3.7\times$. 这些改进来自 KV cache 复用, 单个程序内并行和更快的约束解码. 下面分别解释各基准中加速的原因.

在 MMLU 上, SGLang 可以通过 RadixAttention 复用 5-shot 示例的 KV cache. RadixAttention 对吞吐量和延迟都有帮助. 共享 KV cache 可减少总内存用量, 容纳更大的批量, 从而提高最大吞吐量. RadixAttention 也减少了预填充计算, 因此会降低首 Token 延迟. 在 HellaSwag 上, SGLang 同时复用 few-shot 示例的 KV cache 和多个选项共用的问题前缀, 形成两级共享. 对 ReAct 智能体和生成式智能体, SGLang 会复用智能体模板和先前调用的 KV cache. 在 Tree-of-thought 和 Skeleton-of-thought 中, SGLang 将单个程序内的生成调用并行化, 并尽可能复用 KV cache. 对 JSON 解码, SGLang 借助压缩有限状态机一次解码多个 Token, 从而加快解码. 在多轮对话中, SGLang 复用对话历史的 KV cache. 短输出的加速更明显, 因为 KV cache 复用主要缩短前缀处理时间. 对长输出而言, 不同会话间可共享的内容很少, 且解码时间占主导, 因而几乎没有加速. 在 DSPy RAG 流水线中, SGLang 复用公共上下文示例的 KV cache. 这些基准上的 cache 命中率为 50%-99%. [图 13](#figure-13) (附录) 列出了所有基准的实际和最优 cache 命中率, 结果显示本文 cache 感知调度平均可达到最优命中率的 96%.

最后五个基准中的一部分没有包含 LMQL 和 Guidance, 原因是二者速度较慢且缺少所需功能. LMQL 的问题来自缓慢的 Token 级处理和未经优化的后端, Guidance 则不支持批处理和并行.

<span id="figure-07"></span>

![采用 tensor parallelism 的 Mixtral-8x7B 模型归一化吞吐量](./sglang/figure-07.png)

**图 7.** 采用 tensor parallelism 的 Mixtral-8x7B 模型归一化吞吐量. 越高越好.

**采用 tensor parallelism 的大模型结果.** 我们在同一组基准上运行采用 tensor parallelism 的更大模型 Mixtral-8x7B 和 Llama-70B, 结果见[图 7](#figure-07) 和[图 12](#figure-12) (附录). 大模型上的加速趋势与小模型相近, 表明本文优化也能很好地泛化到大模型. 这里省略 Guidance 和 LMQL, 因为它们没有高效的 tensor parallelism 实现.

**多模态模型上的结果.** SGLang 通过 `image` 和 `video` 原语原生支持多模态模型. 本文的优化也与多模态模型兼容. 对 RadixAttention, 我们计算输入图像的哈希并将其作为基数树中的键, 因而可以为相同图像复用图像 Token 的 KV cache. 我们在 llava-bench-in-the-wild 上运行 LLaVA-v1.5-7B (图像), 在 ActivityNet 上运行 LLaVA-NeXT-34B (视频). 其他基线系统对这些模型的支持不完善, 因此我们以模型作者在 Hugging Face Transformers 中的原始实现作为基线. 如[表 2](#table-02) 所示, SGLang 在这些基准上的吞吐量最高可达到 $6\times$. llava-bench-in-the-wild 会针对同一张图像提出多个问题, SGLang 运行时可在这种情况下复用 KV cache.

<span id="table-02"></span>

![多模态 LLaVA 图像和视频模型的吞吐量比较](./sglang/table-02.png)

**表 2.** 多模态 LLaVA 图像和视频模型的吞吐量比较.

**生产部署.** SGLang 已部署到 Chatbot Arena [Chi24], 用于服务开放权重模型. 一些模型的流量较低, 因而每个模型只由一个 SGLang worker 提供服务. 运行一个月后, 我们观察到 LLaVA-Next-34B [Liu24n] 的 RadixAttention cache 命中率为 52.4%, Vicuna-33B [Chi23] 为 74.1%. Cache 命中来自公共 system message, 频繁复用的示例图像和多轮对话历史. 这使 Vicuna-33B 的首 Token 延迟平均降低 $1.7\times$.

**API 模型上的结果.** 我们测试了一个使用 OpenAI GPT-3.5 从 Wikipedia 页面提取三个字段的提示. Few-shot 提示使 API 推测执行具有较高准确率; 由于要提取三个字段, 该方法将输入 Token 成本降低到约三分之一.

<span id="figure-08"></span>

![Cache 命中率和 RadixAttention 消融实验](./sglang/figure-08.png)

**图 8.** (a)(b) Cache 命中率消融实验. (c) RadixAttention 消融实验.

<span id="section-06-03"></span>

### 6.3 消融实验

**Cache 命中率与延迟/吞吐量的关系.** [图 8](#figure-08)(a)(b) 展示了 tree-of-thought 基准上 cache 命中率与性能指标 (首 Token 延迟, 总延迟, 批量大小和吞吐量) 的关系. 该图通过在运行时禁用一部分已匹配 Token 得到. 结果表明, cache 命中率越高, 批量越大, 吞吐量越高, 延迟越低.

**RadixAttention 的效果.** 我们在多个有代表性的基准上测试 RadixAttention 及其组件的效果. 如[图 8](#figure-08)(c) 所示, “No Cache”表示完全不使用 cache; “No Tree-Structure”表示使用简单的表式 cache, 而不是树结构 cache; “FCFS Schedule”表示使用先到先服务策略, 而不是本文的 cache 感知调度; “Random Schedule”表示随机安排请求; “No Frontend Parallelism”表示关闭解释器中的并行; “No Frontend Hint”表示不由解释器发送 fork 提示; “Full optimizations”表示开启全部优化. 实验结果表明, 要达到最佳性能, 这些组件缺一不可. 关闭前端解释器的并行和提示也会降低运行时性能, 说明前端语言与运行时需要协同设计.

**RadixAttention 的开销.** 我们在一个没有任何 KV cache 复用机会的基准上测试 RadixAttention 的开销. 该基准测量 ShareGPT 数据集上的吞吐量. 运行 100 个请求耗时 74.3 秒, 其中只有 0.2 秒用于管理 RadixAttention 数据结构, 开销不足 0.3%, 可以忽略. 这是因为树操作的复杂度是线性的, 且常数很小. 因此, RadixAttention 可以默认开启.

**压缩有限状态机的效果.** 我们在 JSON 解码基准上测试压缩有限状态机及其组件的效果. 实验结果表明, 压缩有限状态机可以一次解码多个 Token, 因而将吞吐量提高 $1.6\times$. 我们还需要预处理状态机, 并让一批请求复用预处理结果. 否则, 每个请求都重新预处理会使吞吐量降低 $2.4\times$.

<span id="section-07"></span>

## 7 相关工作

许多工作研究过 KV cache 复用, 其中不少与本文同期完成. RadixAttention 首次提出将 KV cache 视为基于树的 LRU cache, 这是它的独特之处. 它是首个支持多级共享, cache 感知调度, 前端-运行时协同调度和分布式场景的方案.

vLLM [Kwo23] 和 ChunkedAttention [Ye24] 研究了 system prompt 共享等简单复用情况, 但不支持多级树结构共享或 LRU cache. PromptCache [Gim24] 提出超越前缀的模块化 KV cache 复用, 但可能使准确率最多下降 43%. HydraGen [Jur24], FlashInfer [Ye24a] 和 ChunkedAttention [Ye24] 关注 CUDA kernel 优化, 不包含 LRU cache. API Serve [Abh24] 和 LLM-SQL [Liu24o] 针对特定应用研究 KV cache 复用, 例如与外部 API 调用或关系数据库交错执行, 但它们没有采用本文的基数树或 cache 感知调度.

目前也有多种 LLM 编程和智能体框架, 例如 Guidance [Gui22], LMQL [Beu23], DSPy [Kha23], LangChain [Lan22], AutoGen [Wu23] 和 LLM Compiler [Kim23]. Guidance 和 LMQL 与 SGLang 最为相似, [第 2 节](#section-02) 对它们进行了比较. 本文的创新在于用新的运行时优化加速所提出的编程模型. SGLang 与其他框架兼容, 也可以加速它们 (例如评测中的 DSPy 示例). SGLang 还与许多常见推理优化兼容 [Yu22a, Pop22, Ami22, Kwo23, Ye24a, Dao22, Lin24, Hoo24, Kan24, Liu24c, Liu24p, Ge24].

<span id="section-08"></span>

## 8 未来方向与结论

**未来方向.** SGLang 已经取得进展, 但仍有若干限制, 也留下了可继续研究的方向: 扩展 SGLang 以支持更多输出模态; 让 RadixAttention 能够跨多级内存层次 (例如 DRAM, Disk) 运行 [She23]; 在 RadixAttention 中支持模糊语义匹配; 在 SGLang 之上提供更高层的原语; 解决 cache 感知调度中的饥饿问题 [She23b]; 改进 SGLang 编译器, 使其能够执行调度和内存规划等高级静态优化.

**结论.** 我们提出 SGLang, 一个用于高效编写和执行结构化语言模型程序的框架. SGLang 通过 RadixAttention, 压缩有限状态机和语言解释器等新优化, 显著提高复杂 LM 程序的吞吐量并降低延迟. 它可以用来开发高级提示技术和智能体工作流. 源代码已经公开.

## 致谢

本项目得到 Stanford Center for Automated Reasoning 的支持, 以及 Astronomer, Google, IBM, Intel, Lacework, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Nexla, Samsung SDS, Uber 和 VMware 的捐赠. Lianmin Zheng 获得 Meta Ph.D. Fellowship 支持. 感谢 Yuanhan Zhang 和 Bo Li 对 LLaVA-NeXT (视频) 的支持.

<span id="appendix-a"></span>

## 附录 A RadixAttention 补充细节

<span id="appendix-a-1"></span>

### A.1 KV cache 背景

目前使用的大多数 LLM, 例如 GPT-3 [Bro20b], PaLM [Cho22b] 和 LLaMA [Tou23a], 都基于自回归 Transformer 架构 [Vas17]. 这些模型根据序列中前面的 Token 预测下一个 Token 的概率. 推理时, 模型先通过一次前向传播处理输入 Token 序列 (这个过程称为“prefill”), 再依次解码输出 Token, 每个 Token 都依赖之前的 Token (这个过程称为“decoding”). 我们把接收输入 Token 序列并生成输出 Token 序列的过程称为一次生成调用. 在整个过程中, 每个 Token 都会产生一些中间张量, 用于解码后续 Token. 这些中间张量称为“KV Cache”, 名称来自 self-attention 机制中的 key-value 对. 讨论本文优化时有一项重要观察: KV cache 的计算只取决于之前的所有 Token, 因而前缀相同的不同序列可以复用前缀 Token 的 KV cache, 避免重复计算.

在 LM 程序中, 多个文本片段和多次生成调用经常会追加到同一个提示中. 跨多次串联调用缓存之前 Token 已计算的 KV cache 可以减少重复计算. 但这种优化并非没有代价, 实现起来也不简单, 因为它需要额外存储和更复杂的内存管理. LM 程序也经常从同一个提示生成多个输出, 或从当前状态 fork 新提示 [Li22]. vLLM [Kwo23] 已经研究了基本的前缀共享. 还可以采用不规则树结构共享等更高级的模式. [图 9](#figure-09) 展示了多次调用间 KV cache 共享的四种典型模式; 现有系统没有一个能自动处理全部四种模式. 相比之下, [第 3 节](#section-03) 的 RadixAttention 可以在运行时自动处理所有这些模式.

<span id="figure-09"></span>

![多次调用间 KV cache 共享的示例](./sglang/figure-09.png)

**图 9.** KV cache 共享示例. 蓝色框表示可共享的提示部分, 绿色框表示不可共享的部分, 黄色框表示不可共享的模型输出. 可共享元素包括 few-shot 学习示例, self-consistency [Wan22g] 中的问题, 多轮对话中的对话历史, 以及 tree-of-thought [Yao23] 中的搜索历史.

<span id="algorithm-01"></span>

**算法 1: 采用连续批处理的 RadixAttention Cache 感知调度.**

- **输入:** 基数树 $T$, 内存池 $P$, 当前运行批次 $B$, 等待队列 $Q$.
- **输出:** 已完成请求和更新后的系统状态.
- **从等待队列取得所有请求:**
  - 令 $requests\leftarrow Q.\text{get\_all\_requests}()$.
- **为所有等待请求搜索匹配前缀:**
  - **对于** $requests$ 中的每个 $req$:
    - 令 $req.prefix\_node, req.prefix\_len\leftarrow T.\text{match\_prefix}(req.input\_tokens)$.
- **按照匹配前缀长度排序请求:**
  - 对 $requests$ 排序.
- **为下一个批次选择请求:**
  - 令 $available\_size\leftarrow T.\text{evictable\_size}()+P.\text{available\_size}()$.
  - 令 $current\_size\leftarrow 0$.
  - 令 $new\_batch\leftarrow []$.
  - **对于** $requests$ 中的每个 $req$:
    - **如果** $req.\text{size}()+current\_size<available\_size$:
      - 将 $req$ 追加到 $new\_batch$.
      - 令 $delta\leftarrow T.\text{increase\_ref\_counter}(req.prefix\_node)$.
      - 令 $available\_size\leftarrow available\_size+delta$.
  - 从 $Q$ 中移除 $new\_batch$ 内的请求.
- **将请求插入当前运行批次:**
  - 将 $new\_batch$ 合并到 $B$.
- **分配新内存, 必要时执行驱逐:**
  - 令 $needed\_size\leftarrow B.\text{needed\_size}()$.
  - 令 $success, buffer\leftarrow P.\text{alloc}(needed\_size)$.
  - **如果** `not success`:
    - 从 $T$ 中驱逐 $needed\_size$.
    - 令 $success, buffer\leftarrow P.\text{alloc}(needed\_size)$.
  - 使用 $buffer$ 运行 $B$.
- **处理已完成请求:**
  - 令 $finished\_requests\leftarrow B.\text{drop\_finished\_requests}()$.
  - **对于** $finished\_requests$ 中的每个 $req$:
    - 在 $T$ 中减小 $req.prefix\_node$ 的引用计数.
    - 将 $req$ 插入 $T$.
- **返回:** $finished\_requests$.

<span id="appendix-a-2"></span>

### A.2 Cache 感知调度伪代码

[算法 1](#algorithm-01) 给出了采用连续批处理时 RadixAttention cache 感知调度的伪代码.

<span id="appendix-a-3"></span>

### A.3 [定理 3.1](#theorem-03-01) 的证明

**定理 3.1.** 对于一批请求, 如果 cache 大小 $\geq$ 最大请求长度, 按深度优先搜索顺序遍历请求的基数树即可达到最优 cache 命中率. 最长共享前缀优先顺序等价于深度优先搜索顺序.

::: details 证明
首先证明深度优先搜索 (DFS) 顺序可以达到最优 cache 命中率. 令 $R$ 表示批次中的请求集合, $T$ 表示由 $R$ 构建的基数树. 对 $T$ 的每条边 $e$, 至少需要计算一次与 $e$ 关联的 KV cache. 令 $|e|$ 表示与 $e$ 关联的 KV cache 大小, $C$ 表示 $R$ 的 KV cache 计算复杂度. 可以得到下界

$$
C \geq \sum_{e\in\text{edges}(T)}|e|.
$$

考虑按 DFS 顺序遍历基数树 $T$. 第一次计算与 $T$ 中某条边 $e$ 关联的 KV cache 后, 我们会继续计算 $e$ 的整个子树. 在计算 $e$ 的子树期间, 边 $e$ 会持续命中, 因而不会发生额外计算. 以 $e$ 为根的子树计算完毕后, 不会再次访问边 $e$. 请注意, cache 大小 $\geq$ 最大请求长度, 而后者等于基数树 $T$ 中的最长路径. 因为子树包含 $e$ 的公共前缀会持续命中, 计算该子树期间边 $e$ 不会被驱逐. 因此, 与每条边 $e$ 关联的 KV cache 只会计算一次. 由此达到下界

$$
C = \sum_{e\in\text{edges}(T)}|e|.
$$

Cache 命中率定义为

$$
\frac{\sum_{r\in R}r\text{ 中缓存的 prefill Token 数}}{\sum_{r\in R}r\text{ 中的 prefill Token 数}},
$$

它等于 $1-\frac{C}{\sum_{r\in R}\text{prefill Token 数}}$, 因而达到上界, 也就是最优值.

接下来用归纳法证明最长共享前缀优先顺序等价于 DFS 顺序.

- **基础情况:** 开始时没有任何 cache, 因而会处理一个随机请求, 它对应 $T$ 中的节点 $x$. 从根到 $x$ 的路径上, 与节点 $\{v_1,\ldots,v_n\}$ 对应的所有请求都无需重新计算. 与节点 $\{v_1,\ldots,v_n,x\}$ 对应的请求, 其计算复杂度与一种有效的 DFS 一致. 从根到 $x$ 的路径会被缓存.
- **归纳步骤:** 假设我们刚访问 $T$ 中的节点 $y$, 已访问节点符合 DFS 顺序. 令 $P$ 表示从根到 $y$ 的路径. 每个尚未访问的节点与已访问节点的最近公共祖先都位于 $P$ 上. $P$ 上的节点已经缓存, 因而在最近公共祖先位于 $P$ 的未访问节点中, 会有一个节点 $z$ 具备最长共享前缀. 最长共享前缀优先顺序将选择 $z$, 这是一种有效的 DFS 顺序. 从根到 $z$ 的路径是最近访问的路径, 因而会被缓存.

:::

在线情况下, DFS 顺序会被打乱, 但在隐含基数树新增的部分上, 最长共享前缀调度仍近似保持 DFS 行为. 下面考察加入一批新请求的步骤来说明这一点.

令 $T$ 表示目前已访问的基数树部分, $T'$ 表示加入一批新请求后的完整新基数树. 令 $C$ 表示 $T$ 中的缓存节点集合. 令 $\text{longest}(C)$ 表示 $C$ 中满足以下条件的节点: 它从根出发的路径最长, 且它在 $T'$ 中的子树尚未完全访问.

随后, 最长共享前缀调度会按 DFS 顺序处理 $T'$ 中以 $\text{longest}(C)$ 为根的子树. 在此过程中可能发生驱逐, 此时 $C$ 中剩余的缓存节点变为 $C^{(1)}\subseteq C$. 接下来对 $T'$ 中以 $\text{longest}(C^{(1)})$ 为根的子树执行 DFS.

同理, 我们会依次得到 $C^{(2)},\ldots,C^{(k)}$, 直到 $C^{(k)}$ 只包含一个叶节点, 且该节点在 $T'$ 中的子树尚未完全访问. 此时已经到达有效的 DFS 状态. $T'$ 的其余部分将按照[定理 3.1](#theorem-03-01) 的证明所述, 按 DFS 顺序访问.

<span id="appendix-a-4"></span>

### A.4 Data-Parallel 分布式 RadixAttention

为让 RadixAttention 适用于包含多个副本 worker 的分布式设置 (即 data parallelism), 我们开发了一套机制: 每个 worker 分别维护自己的子树, router 则维护一棵 meta-tree. Meta-tree 是一棵 trie, 用于追踪所有子树及其对应设备. 一批新请求到达 router 后, 系统在 meta-tree 上执行前缀匹配. 我们依据每个请求的亲和性制定了多种策略; 这里的亲和性通过请求与特定 worker 以及同组其他请求之间的共享前缀长度衡量, 用于高效决定如何分发请求, 尽量减少重复计算. 每次处理新请求时, router 和 worker 都会独立更新各自的树. 如果某个 worker 节点发生驱逐, 它会把这次驱逐提交到队列, router 在低活跃时段处理队列并更新 meta-tree. 我们使用四个 worker 和 MMLU 数据集测试了这种分布式配置, 结果显示它能够线性扩展并达到最优 cache 命中率, 弱一致分布式 cache 设计带来的开销很小. 最大化数据局部性与并行处理效率之间存在权衡. 如何用更高级的调度策略优化这项权衡, 留作未来研究. 同期工作 Preble [Sri24] 也基于 SGLang 的早期版本研究了 data-parallel 调度.

<span id="appendix-b"></span>

## 附录 B 压缩有限状态机补充细节

本节介绍压缩有限状态机的背景和实现细节, 它用于加快约束解码. 我们希望 LLM 遵循正则表达式 (regex). 正则表达式的表达能力更强, 可以表示 JSON schema 等常见格式. 为此, 我们把 regex 转换为有限状态机 (FSM), 在解码过程中引导生成 [Wil23]. FSM 本质上是一张图, 包含节点 (状态) 和边 (带字符串/字符的转移). 从初始状态开始, 每次转移都会追加边上的字符串并移动到下一状态, 最后以一组终止状态结束整个过程. 这种机制根据 FSM 当前状态的转移过滤无效 Token, 从而引导 LLM 解码, 如[图 10](#figure-10) 所示. 到达终止状态前, 解码过程可能需要在 FSM 中经历多次转移.

<span id="figure-10"></span>

![将正则表达式转换为 FSM 并用 FSM 引导解码的示例](./sglang/figure-10.png)

**图 10.** 将 regex 转换为 FSM, 再用 FSM 引导解码过程的示例.

约束解码之所以困难, 是因为约束通常采用自然语言格式表达, 也就是用字符/字符串描述 regex, 而 LLM 将这些内容作为 Token 来解释和处理. 字符串与 Token 之间的映射很复杂, 并非一一对应 [Kuc23].

本节内容源自此前的一篇博客文章 [+3]. 也建议读者阅读该文章, 以了解更多背景并更轻松地理解本节内容.

<span id="appendix-b-1"></span>

### B.1 压缩有限状态机的实现细节

为简化压缩 FSM 的构建, 我们在字符/字符串上而不是 Token 上构建原始 FSM. 下面正式定义单一转移边和压缩边:

- **单一转移边:** 如果 1) 一条边的源节点只有一个后继节点, 且 2) 边上只允许一个字符/字符串, 那么它就是单一转移边.
- **压缩边:** 当且仅当 $e_1,\ldots,e_k$ 都是单一转移边时, 才能把连续相邻的多条边 $(e_0,e_1,\ldots,e_k)$ 压缩成一条边. 压缩边上的文本是 $e_0,e_1,\ldots,e_k$ 各边文本的拼接结果.

从基于字符的 FSM 出发, 我们递归地将单一转移边合并到前一条边中, 直至无法继续压缩, 最终得到压缩 FSM. 这种方法可以加快解码过程; [图 11](#figure-11) 展示了 SGLang 运行时使用压缩 FSM 时的效率.

<span id="figure-11"></span>

![压缩 FSM 与普通 FSM 解码过程的比较](./sglang/figure-11.png)

**图 11.** 使用压缩 FSM 和普通 FSM 解码的比较: 左图显示每次前向传播的解码过程, 右图解释结果中各部分的来源.

<span id="appendix-b-2"></span>

### B.2 通过重新分词处理分词伪影

生成新 Token 后, 我们取得该 Token 对应的字符串, 搜索当前状态的所有出边, 找到以刚解码字符串开头的边, 然后沿边前进. 如果转移边压缩程度很高且包含很长的字符串, 我们还可以提前得到后续几轮要解码的字符串. 加速正是发生在这里, 我们把这个过程称为 *Jump Forward*. 但后续解码阶段仍需将字符串转换成 Token. 由于 LLM 特定的预训练和分词方法, 这个过程并不直接; 随意切分可能改变原意 [Tra24]. 例如, [图 2](#figure-02) 的 regex 中, 压缩文本为 `{"summary": "`. 按 tokenizer, 它只能分为 `{"`, `summary`, `":` 和 `_"`, 不能随意分成 `{"`, `summa`, `ry` 和 `":_`. 为解决这个问题, 我们使用原始 tokenizer 对之前的全部文本和压缩边文本重新分词, 保证结果与 LLM 的原始输入格式一致. 这只会带来很小的重新分词开销.

<span id="appendix-b-3"></span>

### B.3 未来扩展: 处理失真的概率

字符串与 Token 之间的差异还会造成概率分布偏斜 [Tra24]. 例如, [图 2](#figure-02) 中的 regex `"[ABCD][+-]?"` 表示从 `A+` 到 `D-` 的等级. 如果把它换成 `Excellent|Above Average|Fair|Below Average` 等更宽泛的词语, 运行时可能把 `A` 错误映射为 `Above Average`, 因为 `Above Average` 位于一条压缩转移上, 这会歪曲等级顺序. 出现这个问题是因为 LLM 不知道具体的选项范围, 因而产生了不合适的 Token 序列. 要准确计算每个选项的概率, 需要对能得到该选项的所有 Token 序列求概率和, 这会使解码更复杂并增加开销. 一种变通方法是把选项或 regex 直接放入 prefill prompt, 让 LLM 知道有哪些选项, 并以恰当的 Token 序列输出决定. 但这种方法没有解决概率失真的根本问题, 仍需进一步研究来提高压缩 FSM 的准确性.

<span id="appendix-c"></span>

## 附录 C 补充实验设置与结果

**补充实验设置.** [图 5](#figure-05) 和[图 6](#figure-06) 在一张 A10G (24GB) GPU 上运行 Llama-7B 得到. [图 7](#figure-07) 在 8 张 A10G (24GB) GPU 上使用 tensor parallelism 运行 Mixtral-8x7B 得到. [图 8](#figure-08)(c) 在一张 A10G (24GB) GPU 上运行 Llama-7B 得到. [图 12](#figure-12) 在 4 张 A100G (80GB) GPU 上使用 tensor parallelism 运行 Llama-70B 得到. [表 2](#table-02) 的 LLaVA-v1.5-7B 在一张 A10G (24GB) GPU 上运行, LLaVA-Next-34B 则在一张 A100G (80GB) GPU 上运行. 基准图中的每个柱形都需要几分钟到一小时才能完成.

**补充实验结果.** [图 13](#figure-13) 展示了[图 5](#figure-05) 所列基准的实际与最优 cache 命中率. [图 12](#figure-12) 展示了采用 tensor parallelism 的 Llama-2-70B 吞吐量.

<span id="figure-12"></span>

![采用 tensor parallelism 的 Llama-2-70B 模型归一化吞吐量](./sglang/figure-12.png)

**图 12.** 采用 tensor parallelism 的 Llama-2-70B 模型归一化吞吐量. 越高越好.

<span id="figure-13"></span>

![多个基准上的实际与最优 cache 命中率](./sglang/figure-13.png)

**图 13.** 多个基准上的实际 cache 命中率和最优 cache 命中率.

<span id="figure-14"></span>

![SGLang 程序及其对应的数据流图](./sglang/figure-14.png)

**图 14.** 一个 SGLang 程序及其对应的数据流图.

<span id="appendix-d"></span>

## 附录 D 编译器模式

除了正文使用的解释器模式, 还可以将 SGLang 程序编译为计算图, 再用图执行器运行. 这样可以改写计算图并执行更多静态规划, 为更多编译优化创造机会.

<span id="appendix-d-1"></span>

### D.1 设计与实现

我们为 SGLang 设计了一种中间表示 (IR), 将 SGLang 程序的结构和操作表示为计算图. 图中包含原语运算符节点和表示依赖关系的边. [图 14b](#figure-14) 是[图 14a](#figure-14) 中程序对应的计算图. 在该程序中, 每次调用装饰后的函数或 fork 都会创建新的提示状态或流.

其中有多种节点. SGLang 程序中运算符 `+=` 和 `+` 的每个操作数都表示为一个 IR 节点, 包括 `ConstantText`, `Argument`, `Gen`, `Select`, `Variable`, `Fork`, `GetForkItem` 和 `Join`. 依赖关系分为两类: 一类是流内依赖, 用 `+=` 提交到某条流中的操作必须在该流前面的所有操作之后执行; 另一类是流间依赖, 当一条流需要取得另一条流中的变量值时就会产生, 此时必须进行同步. Fork 等操作会操纵多条流, 因而会引入流间依赖.

为生成计算图, 我们使用 tracing, 以抽象参数运行程序并动态构建图. 该方法目前只适用于不含数据依赖控制流的程序, 我们计划在未来解决这项限制. 计算图构建完成后, 可以直接通过图执行器运行, 不再需要重新解释原始 Python 程序. 这样可以进行图改写优化, 降低运行时开销并序列化程序. 执行时, 系统为每条数据流启动流执行器, 按拓扑顺序将 IR 节点分派到各条流.

<span id="appendix-d-2"></span>

### D.2 编译器优化案例研究: 通过代码移动改善前缀共享

我们研究了一种针对 SGLang IR 的编译优化: 通过代码移动改善前缀共享. Auto-tuning 和指令选择等更多经典编译技术预计也可以应用到这里.

该优化通过重排图中节点来增加常量前缀长度, 从而改善前缀共享. 它不会严格保持原始计算, 属于激进优化. 例如, 将提示“Here is a question + {question}. Please act as a math expert and solve the given question.”改为“Please act as a math expert and solve the given question. Here is a question + {question}.”会得到更长的可共享前缀. 这个优化很有意思, 因为 SGLang 中包含自然语言指令, 传统程序分析无法完成它. 我们改为提示 GPT-4 重排图节点. 提示中包含多个示例, 用来教 GPT-4 理解 SGLang IR 的概念. 结果表明, GPT-4 能成功优化一些简单的 SGLang 程序.

我们评估了该优化的效果. 实验从互联网收集 20 个提示模板, 并使用 SGLang 实现. 其中 5 个模板作为 few-shot 训练示例, 其余 15 个作为测试用例. 结果显示, 在 15 个模板中的 12 个上, GPT-4 能成功重排图节点且不改变语义; 我们通过手动检查修改后的提示确认了这一点. 平均而言, 该优化使可共享前缀增加 60 个 Token, 说明 GPT-4 确实能完成这项工作. 优化提示顺序失败, 是因为 GPT-4 没有正确理解图节点背后的语义. 它过于激进, 即使把所有常量提前会改变原语义, 仍然这样处理. 这项案例研究旨在探索用 GPT-4 执行编译优化. 要让这类优化在未来变得可靠, 还需要更多工作.

[+equal-contribution]: Lianmin Zheng 和 Ying Sheng 对本文贡献相同.

[+1]: 在实践中, 计算过程与[定理 3.1](#theorem-03-01) 的证明所述并不相同, 因为输出 Token 数量不可预测, 可能导致 KV cache 重新计算.

[+2]: RadixAttention 的一部分已经作为可选的实验性功能集成到最新版本的 vLLM 中; 因此, 我们使用较早版本进行比较.

[+3]: [https://lmsys.org/blog/2024-02-05-compressed-fsm/](https://lmsys.org/blog/2024-02-05-compressed-fsm/)
