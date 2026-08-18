---
title: 'Asynchronous LLM Function Calling'
createTime: 2026/08/18 22:32:11
permalink: /papers/async-llm-function-calling/
---

> [In Gim](https://ingim.org/)、[Seung-seob Lee](https://www.seungseoblee.com/blog/) 和 [Lin Zhong](https://www.linzhong.org/). 首次提交 arXiv: 2024 年 12 月 9 日; 当前版本 v1. PDF 将本文标识为正在审稿的初步工作. [Asynchronous LLM Function Calling](https://arxiv.org/abs/2412.07017v1). [原始 PDF](/paper/async-llm-function-calling.pdf). [DOI](https://doi.org/10.48550/arXiv.2412.07017). [TeX 源码](https://arxiv.org/src/2412.07017v1). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

大语言模型 (LLM) 使用函数调用来连接外部工具和数据源. 然而, 当前的 LLM 函数调用方法本质上是同步的, 每次调用都会阻塞 LLM 推理, 限制 LLM 的运行和并发函数执行. 在本文中, 我们提出 AsyncLM, 一个用于异步 LLM 函数调用的系统. AsyncLM 通过让 LLM 并发生成和执行函数调用来提高 LLM 的运行效率. AsyncLM 不再等待每次调用完成, 而是引入中断机制, 在函数调用返回时异步通知正在运行的 LLM. 我们设计了函数调用和中断的上下文协议, 提供适配中断语义的微调策略, 并在 LLM 推理过程中高效实现这些机制. 我们在 Berkeley 函数调用排行榜 (BFCL) 的一组基准任务上展示了 AsyncLM 将端到端任务完成延迟降低至同步函数调用的 1.6$\times$-5.4$\times$. 此外, 我们讨论了如何扩展中断机制, 以支持新型的人类-LLM 或 LLM-LLM 交互.

<span id="section-01"></span>

## 1 引言

函数调用能力让大语言模型 (LLM) 可以访问外部数据源和工具, 例如天气预报和计算器. 商业和开源 LLM 都已集成这一功能 [Sch24, Pat23b], 为多种应用解锁了新的可能性, 从在动态环境中运行的自主 AI 智能体 [Wan24k], 到结合符号推理与 LLM 来解决复杂问题的神经符号系统 [Tri24].

LLM 函数调用是同步的, LLM 与函数调用执行器轮流生成和执行调用. 这种方法虽然实现简单, 但既不节省资源, 也没有响应性. 每次函数调用都会阻塞 LLM 推理, 这是一种资源消耗极高的过程, 直到函数返回. 从执行器的角度看, 这限制了并发性, 因为所有函数调用都必须按照 LLM 发起它们的顺序完成. 随着任务复杂度和函数数量增加, 这些低效问题会进一步恶化 [Zah24].

<span id="figure-01"></span>

![同步与异步函数调用的时间线](./async-llm-function-calling/figure-01.png)

**图 1. 同步与异步函数调用.** 异步函数调用通过让 LLM 在函数调用于后台执行的同时继续为独立任务生成 token, 提高了 LLM 的运行效率.

一些研究尝试解决这些问题, 包括使用编译器并行化函数调用 [Kim24a], 融合顺序调用以减少开销 [Sin24a], 设计紧凑的调用语法 [Che23c], 以及优化用于函数调用的 LLM 服务系统 [Abh24, Gao24b, Xu24b]. 虽然这些方法有助于减少函数执行时间或函数调用次数 ([第 2 节](#section-02)), 但它们从根本上受限于函数调用的同步特性, 例如 LLM 必须等待函数调用执行器完成.

我们提出 AsyncLM, 一个通过支持 LLM 与函数调用执行器之间的*异步*交互来克服这些限制的系统. 在 AsyncLM 中, LLM 和执行器彼此独立运行而不会互相阻塞, 这一设计借鉴了异步编程范式: 事件, 例如函数调用完成, 独立于主程序流程, 例如 LLM token 生成流而发生. [图 1](#figure-01) 展示了这一概念.

AsyncLM 的关键机制是可*中断*的 LLM 解码. 在 AsyncLM 中, 函数调用可以非阻塞地执行; 当函数调用返回时, 执行器会将*中断 token* 注入 LLM 的 token 生成流, 异步通知 LLM. 这个概念虽然简单, 但 AsyncLM 必须解决协调 LLM 与执行器时的两个挑战. (i) 中断 token 的时机必须谨慎选择, 以避免干扰 LLM 正在生成的其他任务或函数调用. 例如, 在另一个函数调用的参数中途注入中断, 可能同时破坏正在进行的函数调用生成和已经返回的函数调用. (ii) LLM 必须正确处理注入的中断, 即使这种中断不符合模型训练所依据的对话模式.

为了应对这些挑战, 我们协同设计了异步函数调用接口和 LLM 微调策略. (i) 我们开发了一种 token 高效的领域特定语言 CML, 用来表示函数调用和中断 ([第 3 节](#section-03)). AsyncLM 在 token 生成期间检测 CML 语法, 识别函数调用的开始和结束, 并据此延迟中断. CML 还通过在中断语法中嵌入必要上下文, 帮助 LLM 在中断后恢复处理. (ii) 我们通过微调 LLM 来教会它如何生成异步函数调用, 如何从处理中断的状态恢复, 以及何时必须暂停并等待函数调用返回时通知 LLM 服务系统 ([第 4 节](#section-04)). 这三种情况都使用 CML 作为接口. 此外, 当 LLM 选择等待时, AsyncLM 会确定管理阻塞状态的最高效策略, 例如丢弃、交换或保留 KV cache.

AsyncLM 相比同步函数调用有两点优势. 首先, AsyncLM 通过重叠函数的生成和执行来减少执行时间. 这一特性保证 AsyncLM 至少与带并行执行的同步函数调用一样快, 即使在最坏情况下也是如此 ([第 6.3 节](#section-06-03)). 其次, AsyncLM 无需预先知道未来函数调用的依赖图, 即可实现自动并行. 例如, 给定任务“使用数学谱系 API 告诉我 Leonhard Euler 是否是我的学术祖先”和 API 定义 `find_advisors(str)`$\rightarrow$`list[str]`, AsyncLM 可以递归并行地调用 `find_advisors`, 类似于执行并行深度优先搜索. 相比之下, 同步函数调用方法很难高效处理此类任务, 因为它需要预先确定可并行的函数调用列表 ([第 2 节](#section-02)).

我们在 Llama 3 模型上实现 AsyncLM [Dub24], 并使用上下文提示在 GPT-4o 上模拟微调, 即在输入提示中通过 CML 示例和说明引导模型 ([第 5 节](#section-05)). 我们在来自 Berkeley 函数调用排行榜 (BFCL) 的函数调用任务套件上评估这两个模型 [Yan24d] ([第 6 节](#section-06)). 基准比较显示, AsyncLM 相比同步顺序函数调用将端到端任务完成速度提高至 1.6$\times$-5.4$\times$, 相比同步并行函数调用最高可获得 2.1$\times$ 的加速. 我们还评估了 AsyncLM 对函数调用准确率的影响, 结果表明, 在经过微调的 Llama 模型中, AsyncLM 与同步函数调用保持相同的准确率. 值得注意的是, GPT-4o 无需显式微调即可处理异步函数调用. 最后, 我们讨论了所引入的中断机制用于新型 AI 应用的潜力, 包括可中断的 LLM 助手等人类-LLM 和 LLM-LLM 交互.

<span id="figure-02"></span>

![同步、并行和异步 LLM-执行器交互的比较](./async-llm-function-calling/figure-02.png)

**图 2. LLM-执行器交互的比较.** 并行函数调用通过预先生成函数调用并并行执行来减少端到端执行时间, 但仍要求 LLM 等待所有调用完成. 异步函数调用通过让 LLM 在函数调用于后台执行时生成 token, 克服了这一限制.**任务:** *“将 webpage.html 总结为 event.pdf, 按照 style.txt 的样式将其发送给 attendee.xls 中的人员, 如果目录中没有列出系主任, 则将其抄送给系主任. ”*

<span id="section-02"></span>

## 2 背景与相关工作

使用代码执行增强 LLM 的概念 [Mia23] 已得到广泛研究, 典型场景包括检索增强生成 [Kha22, Yao22b]、自主智能体 [Hua24c, Wan24k] 和神经符号问题求解 [Pan23, Tri24].

**学习生成函数调用.** LLM 与外部系统交互最常用的方法是工具或函数调用 [Sch24, She23c], 在这种方法中, 模型会自主生成对外部执行器 (例如 API 服务器、代码解释器) 的调用. 这种能力可以通过在多样化任务数据集上微调 [Pat23b] 或使用上下文指令 [Lia24] 来改进. 函数描述及其参数通常以 JSON 等结构化提示格式提供. 我们的方法在这一范式上支持异步函数调用, 要求 LLM (1) 在生成调用时考虑执行时间, 以及 (2) 使用中断语义决定后续调用.

**高效的 LLM 函数调用.** 函数调用的一项主要挑战是优化效率, 提高资源利用率并降低延迟. 已有研究探索了多种优化函数调用过程的策略. 例如, 并行函数调用方法 [Kim24a, Ope23d] 指示 LLM 将可以同时执行的调用打包, 使外部编译器能够优化这些批次以并行执行. 顺序函数调用优化包括函数调用融合 [Sin24a]、缓存 [Sin24b]、紧凑的调用表示语法 [Che23c] 和函数调用部分执行 [Xu24b], 这些方法允许单个代码块的生成和执行重叠.

**同步交互的限制.** 当前 LLM 执行同步函数调用. 生成和执行交错进行, 由于 LLM 的无状态特性, 每次函数调用都需要新会话, 这给 LLM 推理带来了额外开销. 这促使研究者探索等待期间 KV cache 的高效管理 [Abh24, Yu25b, Gao24b]. 更重要的是, 同步函数调用从根本上限制了资源利用, 因为 token 生成和函数执行无法同时进行. ReWOO [Xu23b] 通过提示 LLM 将推理与观察解耦, 将多个函数调用的执行合并为一次, 从而应对这一限制. 我们的工作在概念上与 ReWOO 一致, 但不依赖 LLM 的推理策略, 因此适用范围更广.

**我们的方法.** AsyncLM 通过支持异步执行来应对同步函数调用的限制. [图 2](#figure-02) 展示了异步函数调用相对于同步方案的优势. 虽然函数并行化优化 [Kim24a, Ope23d] 能够减少端到端执行时间, 但 LLM 仍然必须等待函数调用完成, 使资源处于空闲状态. 异步函数调用允许 LLM 在正在执行函数调用的同时继续为独立任务生成 token, 消除了这一瓶颈. 为了支持异步调用, AsyncLM 引入了在函数调用返回时异步通知运行中 LLM 的机制, 并微调 LLM 使其理解中断和利用并发执行.

AsyncLM 围绕两项关键创新构建. 首先, 它使用领域特定语言 (CML) 描述异步函数调用和中断, 将必要上下文嵌入接口以实现无缝集成 ([第 3 节](#section-03)). 其次, 它的微调策略训练 LLM 响应异步事件, 生成并发函数调用, 并在需要暂停等待先前调用完成时通知服务系统 ([第 4 节](#section-04)), 以确保函数调用依赖关系正确. 我们在 AsyncLM 的 LLM 服务系统中实现这些组件; 该系统监控 token 生成, 调度执行器上的函数调用, 并根据需要注入中断或暂停生成来管理 token 流 ([第 5 节](#section-05)).

<span id="section-03"></span>

## 3 使用 CML 表示异步交互

<span id="figure-03"></span>

![CML 中的异步函数执行工作流](./async-llm-function-calling/figure-03.png)

**图 3. 异步函数执行工作流.** LLM 以 CML 格式生成函数调用, 系统实时监控这些调用并在后台发送给代码执行器. 当函数调用完成时, 执行器插入包含执行结果的中断, 异步通知 LLM.

我们定义了一种简单的领域特定语言, 称为 *Context Markup Language* (CML), 用来表示异步函数调用和中断. CML 充当 LLM 与执行器之间的接口, 其语法确保各组件通过该接口交互时提供必要的上下文. 例如, LLM 可以在 CML 中生成函数调用以通知执行器, 执行器则通过在 CML 中插入中断 token 来表示完成, 如[图 3](#figure-03) 所示.

CML 使用一组最少的专用 token: `[CALL]`、`[INTR]`、`[TRAP]`、`[END]` 和 `[HEAD]`. `[CALL]`、`[INTR]` 和 `[TRAP]` token 启动一个*控制块*, 分别表示函数调用 ([第 3.1 节](#section-03-01))、中断 ([第 3.2 节](#section-03-02)) 或陷阱 (一种特殊的中断). `[END]` token 标记控制块的结束, `[HEAD]` 分隔可选元数据 (例如函数调用标识符) 和函数调用主体. 本节其余部分定义 CML 的语义.

<span id="section-03-01"></span>

### 3.1 发起函数调用

在 AsyncLM 中, LLM 使用以下格式发起函数调用: `[CALL] {function call} [END]`. 函数调用可以使用执行器支持的任意有效可执行语言编写, 例如 Python 代码或 JSON 抽象语法树 (AST). 但是, 每个函数调用块中只能始终使用一种语言. 独立的函数调用应放在不同的块中, 以支持并行执行.

生成的函数调用会在不阻塞 token 流的情况下启动执行, 从而实现隐式并行. 例如, 如[图 3](#figure-03) 所示, 如果 LLM 生成两个函数调用 (`search_nearby` 和 `put`), 执行器会在独立的 worker 中处理它们, 使函数调用的生成和执行可以流水线化. 这种重叠降低了延迟, 因为 `search_nearby` 的执行可以与生成写消息同时进行. 函数调用按照依赖顺序生成, 确保依赖调用按正确顺序执行. 例如, LLM 只有在搜索花店并完成消息撰写后, 才能生成函数调用 `text`.

**为中断分配标识符.** 如果 LLM 需要在后续调用或推理中引用函数结果, 可以在函数头中包含标识符 (例如 `[CALL] job1 [HEAD] {function call} [END]`), 其中 `job1` 充当唯一标识符. 为避免冲突, LLM 按照 Python 变量命名约定生成标识符. 标识符必须在整个会话中保持唯一. 虽然我们的原型尚未实现这一点, 但可以将唯一性检查作为语法验证的一部分 ([第 5.1 节](#section-05-01)).

<span id="section-03-02"></span>

### 3.2 触发中断

当执行器完成带有已注册标识符的函数调用时, 它会在 token 流末尾 (即 LLM 上下文) 插入中断块, 异步通知 LLM. 加入中断块后, token 生成继续进行. 中断块的格式为 `[INTR] {id} [HEAD] {value} [END]`, 其中 `{id}` 与对应 `[CALL]` 块中的标识符匹配 (例如前面示例中的 `job1`). `{value}` 包含执行器的结果, 例如函数输出或错误消息.

**临界区.** 由于中断会修改 LLM 的 token 生成流程, 必须确保中断之后的上下文符合 CML 语法. 例如, 如果在生成函数调用期间插入中断块, 生成的 token 可能在逻辑上不正确, 违反 CML 语法. 为避免这一点, AsyncLM 会在特定 token 生成期间临时禁用中断, 这一做法受到操作系统在处理其他中断时延迟低优先级中断的启发. 中断管理器中的一个名为*临界区*的标志 ([第 5.3 节](#section-05-03)) 决定执行器是否可以插入中断. 生成函数调用块时, 该标志设为 `false`; LLM 退出该块后恢复为 `true`. 在临界区期间发生的任何中断都会排队, 并在标志变为 `true` 时插入.

**等待中断.** 在同步函数调用中, LLM 发出函数调用后会停止生成 token, 等待调用完成. 相比之下, 异步函数调用让 LLM 无需等待即可继续为其他任务生成 token. 但是, 当任务间存在依赖时, LLM 有时必须暂停以等待函数结果. 标准 `[EOS]` token 无法满足这一目的, 因为它不能表示 token 生成是已经完成并可以释放资源, 还是 LLM 在函数结果到达前暂时暂停.

为解决这一歧义, AsyncLM 引入了由自身发起的中断, 称为*陷阱*, 用来通知 LLM 服务系统 ([第 5.4 节](#section-05-04)) 何时暂停 token 生成. 每个陷阱遵循简单结构 `[TRAP][END]`. 陷阱为异步函数调用创建明确边界, 帮助生成用于微调 LLM 异步函数处理能力的训练样本 ([第 4 节](#section-04)). 它们还为服务系统提供优化机会 ([第 5 节](#section-05)).

<span id="section-04"></span>

## 4 学习处理中断

我们提出一种微调方案, 训练 LLM (i) 使用 CML 生成异步函数调用和陷阱, 以及 (ii) 处理传递先前函数调用结果的中断. 核心思想是构造一个包含模拟函数调用和中断的数据集, 对 LLM 与执行器之间的理想交互建模. 我们从公开的函数调用数据集中提取任务描述和函数定义, 添加估计完成时间, 并将它们作为提示的一部分以 JSON 格式提供给 LLM [Ope23d, Pat23b, Lia24]. 为简单起见, 我们假设每个任务都可以通过有限次函数调用完成.

<span id="section-04-01"></span>

### 4.1 训练目标

AsyncLM 微调的主要目标是训练 LLM 通过有效利用异步函数调用, 在遵守任务依赖并考虑估计函数执行时间的同时, 最小化总任务完成时间 (即 makespan). 具体而言, LLM 必须在以下方面作出决策.

**决定下一个函数调用.** LLM 根据上下文选择下一个要调用的函数. [+1] 它识别出可以立即调用的函数, 即没有待处理依赖的函数, 并选择最合适的一个. AsyncLM 使用最长处理时间优先 (LPT) 策略, LLM 优先调用估计执行时间最长且已经就绪的函数. 这一方法通过最大化函数调用生成与执行的重叠来减少空闲时间, 在可以调用多个独立函数时尤其有效. 对于并行函数调用场景, LPT 是最优的 ([第 6 节](#section-06)).

[+1]: 我们用*上下文*指当前对 LLM 可见的 token 序列.

**处理中断.** 当上下文中出现中断时, LLM 必须决定继续为当前任务生成 token, 还是转而处理被中断的任务. 例如, 如果已完成任务的优先级低于当前任务, 或它是序列中的最终任务, LLM 可能忽略该中断. AsyncLM 使用与决定下一个函数调用相同的优先级原则处理中断. 中断可能引入新的可调用函数; 如果新可用函数在所有选项中具有最长的预计处理时间, LLM 就会被训练为下一个调用它. 例如, 在[图 2](#figure-02) 所示的任务调度场景中, 当“Read html”和“Read xls”完成后, LLM 会优先执行“Summarize & save pdf”, 而不是“Fetch contact”, 因为前者的预计处理时间更长.

**生成陷阱.** 当由于依赖未来中断而没有可调用函数, 且没有其他 token 可生成时, LLM 必须生成陷阱来暂停 token 生成.

<span id="section-04-02"></span>

### 4.2 生成训练样本

为了创建训练样本, 我们使用带有多轮场景的 LLM 基准中的现有函数调用轨迹, 模拟 LLM 与执行器之间的理想交互 [Lu24, Yan24d, Yao24].

**DAG 生成.** 我们开发了一个 Python 程序, 从数据集的每个样本中提取函数调用的有向无环图 (DAG). 在顺序场景中, DAG 是线性的, 节点表示函数调用, 边表示依赖关系. 在并行场景中, DAG 分支表示独立函数调用. 多轮场景由多个 DAG 组成.

**模拟中断.** 我们使用 LPT 策略在这些 DAG 上模拟 LLM 与执行器之间的交互. 在每次模拟运行中, 我们为函数调用分配 1 ms 到 1 s 范围内的随机估计执行时间, 以防止模型对特定函数名称过拟合. 这些估计值作为输入提示的一部分提供给 LLM. 为确定中断时机, 我们在每次模拟中随机设置每个输出 token 的时间 (TPOT), 范围为每个 token 5 ms 到 30 ms, 并通过统计生成的 token 数量来跟踪经过的时间. 当函数调用完成时, 我们插入包含执行结果的中断块. 如果 DAG 中所有函数都在等待依赖, 我们就插入陷阱块并注入下一个中断.

<span id="figure-04"></span>

![带有 token 监视器、执行器、中断管理器和陷阱处理器的 AsyncLM 推理过程](./async-llm-function-calling/figure-04.png)

**图 4. AsyncLM 的 LLM 推理过程概览.** LLM 自回归地生成 token. AsyncLM 通过增加额外步骤来监控 token 生成过程以识别 CML, 并根据识别出的 CML 语法管理异步函数调用、中断和陷阱, 从而实现异步函数调用. 编号步骤 1 至 5 展示了从生成函数调用到执行它并返回结果的逻辑流程.

<span id="section-05"></span>

## 5 实现

如[图 4](#figure-04) 所示, 我们通过拦截 LLM 的自回归 token 生成来实现 AsyncLM. 系统使用 Python `transformers` 库构建 [Wol20]. AsyncLM 包含四个主要组件: token 监视器、执行器、中断管理器和陷阱处理器. 本节介绍每个组件的实现以及对 LLM 推理所作的修改. 虽然其中一些组件可以在不修改服务系统的情况下通过云端 LLM API 实现 ([第 5.5 节](#section-05-05)), 但这种实现只在某些场景中实用 ([第 6.2 节](#section-06-02)).

<span id="section-05-01"></span>

### 5.1 Token 监视器

Token 监视器审计并调节 LLM 的 token 生成过程. 它的主要功能是 (i) 在 token 流中检测到函数调用或陷阱时立即通知执行器或陷阱处理器, 以及 (ii) 强制遵守 CML 语法. Token 监视器作为 LLM 下一个 token 采样过程的钩子实现, 约束下一个 token 的分布, 确保生成 token 遵循有效的 CML 语法 [Gen23c]. 它使用有限状态机 (FSM) 强制执行 CML 规则, 并阻止 LLM 生成 `[INTR]` token, 从而防止“幻觉式”中断, 因为中断必须由系统注入, 而不是由 LLM 生成. 每次采样 token 后, token 监视器都会检查函数调用或陷阱, 如果检测到, 就立即通知执行器 ([第 5.2 节](#section-05-02)) 或陷阱处理器 ([第 5.4 节](#section-05-04)). 此外, 在确定 LLM 是否处于可中断状态后, token 监视器会相应设置临界区标志. 它将生成的 token 和临界区标志一起发送给中断管理器 ([第 5.3 节](#section-05-03)).

<span id="section-05-02"></span>

### 5.2 执行器

执行器管理 LLM 生成的函数调用的执行. 它从 token 监视器接收每个函数调用 (在我们的原型中以 Python 语法格式化), 以及用于管理未来中断的可选标识符. 每个函数调用在专用 worker 上运行, 只要资源允许, 就可以并发执行多个调用. 这些 worker 直接与外部系统交互, 例如 API 服务器或代码解释器. 函数完成后, 执行器会将结果和标识符发送给中断管理器 ([第 5.3 节](#section-05-03)).

<span id="section-05-03"></span>

### 5.3 中断管理器

中断管理器有三个主要功能: (i) 管理中断队列, (ii) 根据临界区标志跟踪 token 生成过程何时可中断, 以及 (iii) 将 CML 格式的中断块插入 token 流. 当执行器完成函数调用时, 它会将结果加入中断队列. 在每个解码步骤中, 中断管理器从 token 监视器接收新生成的 token 和临界区标志 ([第 5.1 节](#section-05-01)). 如果进程可中断, 即临界区标志未设置, 它会将所有排队的中断格式化为 CML, 对其进行 token 化, 并将其追加到生成的 token 中. 这些 token 随后会在下一个解码步骤由 LLM 处理.

<span id="section-05-04"></span>

### 5.4 陷阱处理器

陷阱处理器的目标是在不增加任务完成延迟的情况下, 尽量减少 GPU 内存中闲置 KV cache 的使用. 当 token 监视器检测到陷阱, 表明需要暂停 token 生成时, 它会将当前上下文通知陷阱处理器. 陷阱处理器根据这一通知, 考虑当前上下文中的 token 数量和下一个中断完成前的估计时间, 决定暂停期间管理 KV cache 的最佳策略. 已有工作表明, 重新计算 KV cache 的开销随 token 数量呈二次增长, 而将其交换到主存的开销呈线性增长 [Abh24, Gim24]. 根据这些开销的增长特性, 如果重新计算和交换的时间都超过估计等待时间, 陷阱处理器会将 KV cache 保留在 GPU 内存中. 否则, 如果重新计算延迟更低, 就选择重新计算; 如果交换延迟更低, 就选择交换. 我们在[第 6.2 节](#section-06-02)提供了理想陷阱处理策略的真实示例.

<span id="section-05-05"></span>

### 5.5 在聊天补全 API 上实现

为了展示 AsyncLM 的适应性, 我们还使用 OpenAI 的 (流式) 聊天补全 API 实现它, 而不修改服务系统. 执行器实现可以直接复用. 对于 token 监视器, 我们只实现 CML 解析器而不进行约束解码, 因为 API 已经会流式传输采样 token. 为模拟中断管理器的中断插入机制, 每当触发中断时, 我们就向 API 服务器发起新请求并丢弃之前的会话. 陷阱处理器对于云端 API 没有必要, 因为云端 API 是无状态的, 插入中断时会为每个新会话从头重新计算整个 KV cache. [+2] 我们指出, 这一实现只有在首 token 时间 (TTFT) 延迟较低时才实用, 而大多数云服务通常并非如此 (详情见[第 6.2 节](#section-06-02)).

[+2]: 一些 LLM 服务支持缓存提示以便复用, 但这些缓存通常针对文档等频繁访问的信息设计, 而不是运行时使用.

<span id="section-06"></span>

## 6 评估

我们的评估回答两个问题: (i) *延迟* ([第 6.1 节](#section-06-01)-[第 6.4 节](#section-06-04)), 即异步函数调用相比同步方法能将任务完成延迟降低多少; 以及 (ii) *正确性* ([第 6.5 节](#section-06-05)), 即异步机制如何影响生成函数调用的正确性.

**工作负载.** 为评估任务完成延迟和函数调用准确率, 我们使用 Berkeley 函数调用排行榜 (BFCL) [Yan24d], 该排行榜覆盖车辆控制、旅行预订、文件系统操作和 Twitter API 等八个领域中的真实函数调用. BFCL 包含 84 个独特函数. 我们使用 BFCL 中的三个数据集覆盖不同函数调用场景: `v1-parallel`、`v2-parallel-live` 和 `v3-base-multi-turn`. 具体而言, `v1-parallel` 和 `v2-parallel-live` 提供 400 个并行函数调用场景, `v3-base-multi-turn` 提供 200 个多步函数调用场景. 为了模拟更复杂的多步并行函数调用, 我们创建了新数据集 `v3-multi-step-parallel`. 该数据集由 200 个场景组成, 每个场景从 `v3-base-multi-turn` 第一轮中随机组合三个不同的多步样本. 从总计 800 个样本中, 我们使用 200 个进行微调, 其余 600 个用于评估. 每个函数测得的执行时间范围为 30 ms 到 500 ms, 平均为 110 ms.

**AsyncLM 设置.** 我们考虑两种 LLM 部署设置: 本地和云端. 在本地部署中, LLM 推理和函数执行运行在同一台配备 NVIDIA RTX 4090 GPU 的机器上. 这一部署使用具有 3B 和 1B 参数的 Llama-3.2 模型 [Dub24]. 我们遵循 LlamaFactory 的默认配置微调 Llama 模型 [Zhe24a]. 本地模型使用 Text-Generation-Interface 提供服务 [Tex22]. 在云端部署中, 只有函数执行在本地进行. 这一部署使用 OpenAI 的 GPT-4o 和 GPT-4o-mini. 我们使用少样本提示适配它们; 从每个数据集中选择一个示例, 并提供关于如何解释这些示例的详细说明.

**基线.** 为与同步 LLM 函数调用进行比较, 我们采用两个同步基线:

- *Sync*: LLM 代码生成与执行交错进行的顺序函数调用.
- *Sync-Parallel*: 并行函数调用 [Ope23d, Kim24a], LLM 将独立函数调用打包, 执行器并行运行它们.

我们还比较 AsyncLM 的两种实现:

- *Async-Naive*: 基于聊天补全 API, 不修改底层 LLM 服务系统.
- *Async*: 使用协同设计推理过程的 AsyncLM.

对于云端 GPT-4o 上 *Async* 的延迟测量, 我们根据 OpenAI API 的平均 token 生成延迟统计 (每个输出 token 5 ms) 报告模拟结果. 这为在云端 LLM 上实现 AsyncLM 的延迟收益提供了粗略估计.

请注意, BFCL 官方报告的 LLM 函数调用生成准确率为 GPT-4o 的 62% 和 Llama-3.2-3B 的 43% [Yan24d]. 这可能使延迟评估产生偏差, 因为错误的函数调用可能遗漏必要步骤或包含冗余步骤. 为减少这种干扰, 我们在提示中加入包含真实答案的“作弊表”. 这确保了各次测量中生成函数调用的一致性.

<span id="section-06-01"></span>

### 6.1 并行函数调用

首先, 我们采用函数调用互不依赖的简单设置, 并在[图 5](#figure-05) 中报告结果. 并行函数调用很常见 [Ope23d]. 例如, 当用户询问“明天 Seattle 还是 Vancouver 哪个城市更可能下雨?”时, LLM 可以生成两个函数调用: 一个获取 Seattle 的天气数据, 另一个获取 Vancouver 的天气数据, 二者可以并行执行. 我们使用 BFCL 中的 `v1-parallel` 和 `v2-parallel-live` 进行该评估.

**结果.** *Async* 最高可获得 2.1$\times$ 的延迟改善, 相对于 *Sync*. 我们将端到端任务完成延迟测量为第一个 token 生成到最后一个 token 生成之间的时间. 结果表明, 本地部署中 *Async* 比 *Sync* 快 1.6$\times$, 云端部署中快 2.1$\times$. 相比之下, 本地部署中 *Sync-Parallel* 比 *Sync* 快 1.3$\times$, 云端部署中快 1.7$\times$. 虽然 *Async-Naive* 比 *Async* 慢, 但在本地部署中仍比 *Sync-Parallel* 快 1.2$\times$.

<span id="figure-05"></span>

![并行函数调用延迟结果](./async-llm-function-calling/figure-05.png)

**图 5. 并行函数调用延迟.** 合并 BFCL 并行数据集上的端到端任务完成延迟, 误差条表示第 10 和第 90 百分位范围. 柱状图标注表示相对于 *Sync* 的加速比.

<span id="figure-06"></span>

![多步并行函数调用延迟结果](./async-llm-function-calling/figure-06.png)

**图 6. 多步并行函数调用延迟.** 在 BFCL 多步数据集中同时处理三个并行任务时的端到端任务完成延迟.

<span id="section-06-02"></span>

### 6.2 多步并行函数调用

为了评估函数调用存在依赖时更复杂的设置, 我们使用 `v3-multi-step-parallel` 数据集评估 AsyncLM. 该数据集包含三个独立任务, 每个任务需要 1-5 次顺序函数调用. 结果见[图 6](#figure-06). 在这些函数调用场景中, AsyncLM 必须遵守任务顺序依赖, 并管理每次函数调用的中断标识符. 例如, “make pasta”任务可以包含两个独立序列: (i) `boil_water()` 后跟 `put_pasta_noodles`, 以及 (ii) `chop_vegetables()` 后跟 `stir_fry()`, 最后汇合到 `mix_everything()`.

**结果.** AsyncLM 通过并行化独立序列中的函数调用, 相比 *Sync* 将延迟最多降低 5.4$\times$; 而 *Sync-Parallel* 相比 *Sync* 将延迟降低 3.2$\times$. 如[图 2](#figure-02) 所示, 与 *Sync-Parallel* 中 LLM 必须等待所有打包函数完成不同, *Async* 使用 LPT 策略调度单个函数调用. 这使 LLM 能够更有效地优化 token 生成周期, 类似于 CPU 指令调度中的乱序执行.

<span id="section-06-03"></span>

### 6.3 延迟分析

为了理解异步函数调用如何降低延迟, 我们在并行函数调用场景下进行理论分析 ([第 6.1 节](#section-06-01)). 所有证明见[附录 A](#appendix-a).

**重叠生成与执行.** 异步函数调用至少与同步函数调用一样快. 简单并行函数调用中 *Sync* 的总延迟可以建模为:

$$
L_{\mathrm{Sync}}(F)=\sum_{f\in F}G(f)+\sum_{f\in F}E(f),
$$

给定 $F$ 为一组执行上互不依赖的函数, 其中 $G(f)$ 是 $f\in F$ 的 token 生成延迟, $E(f)$ 是执行时间. 对于 *Sync-Parallel*, 总延迟为:

$$
L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)=\sum_{f\in F}G(f)+\max_{f\in F}E(f),
$$

假设并行化它们的开销可以忽略. 对于使用 LPT 启发式的 *Async*, 总延迟为:

$$
L_{\mathrm{Async}}(F)=\max_{f\in F}\left(E(f)+\sum_{g\in\mathrm{pred}(f,F)}G(g)\right),
$$

其中 $\mathrm{pred}(f,F)=\{g\in F\mid E(f)\leq E(g)\}$. 直观地说, 这一表述表示每个函数调用 $f$ 都在生成所有具有相同或更高优先级的函数调用之后才启动, 即在 $\mathrm{pred}(f,F)$ 中生成时间的总和. 我们证明, 在生成 token 数量相同且 *Sync-Parallel* 或 *Async* 除 $G(f)$ 和 $E(f)$ 外没有额外开销的最坏情况下, *Async* 至少与 *Sync-Parallel* 一样快.

<span id="theorem-06-01"></span>

**定理 6.1.** 对任意独立函数集合 $F$, $L_{\mathrm{Async}}(F)\leq L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)<L_{\mathrm{Sync}}(F)$.

**根据平均函数执行时间刻画加速比.** 假设 $E(\cdot)$ 服从正态分布, 我们使用 $\overline{E}$ 和 $\overline{G}$ 估计 *Async* 相对于 *Sync* 的预期加速比, 其中 $\overline{E}$ 表示平均执行时间, $\overline{G}$ 表示集合 $F$ 中函数的平均生成时间.

<span id="theorem-06-02"></span>

**定理 6.2.** 当 $|F|$ 较大时, 加速比 $\frac{L_{\mathrm{Sync}}}{L_{\mathrm{Async}}}$ 约为 $1+\frac{\overline{E}}{\overline{G}}$, 误差为 $\mathcal{O}((\frac{\overline{E}}{\overline{G}})^2)$.

这一结果表明, 平均执行时间较长的任务 (例如昂贵的 I/O) 可以获得更高的加速比, 而执行时间较短的任务 (例如简单的算术操作) 可能无法从 AsyncLM 中获得同样多的收益.

**采用 LPT 启发式的理由.** LPT 对并行函数调用是最优的. 我们证明, 在异步并行函数调用中, LPT 启发式可以最小化总延迟. 直观地说, 更早开始执行较长函数可以最大化与 token 生成的重叠, 减少总完成时间. 为正式说明这一论点,

<span id="theorem-06-03"></span>

**定理 6.3.** 延续[定理 6.1](#theorem-06-01), 任何偏离 LPT 顺序的安排都不可能得到更低的总延迟.

这些分析粗略展示了异步函数调用通过重叠生成和执行所带来的优势, 以及 LPT 启发式的理由, 特别是在函数彼此独立时.

<span id="section-06-04"></span>

### 6.4 系统开销

**AsyncLM 的推理开销.** 与 *Sync* 相比, AsyncLM 引入了两类潜在的 LLM 推理开销: (i) 使用 CML 和函数调用标识符产生的语法开销, 以及 (ii) 中断开销, 由将函数执行结果及其标识符注入 LLM 上下文造成. 语法开销很小, 因为 *Sync* 和 *Sync-Parallel* 也要求函数调用使用特定格式, 例如用 Markdown 代码语法包裹函数调用. AsyncLM 每次函数调用增加两到三个控制 token, 相对于生成的总 token 数量可以忽略不计. 同样, 中断开销主要来自与 *Sync* 相比额外的上下文中断标识符, 因为两种方法都会在 LLM 上下文中接收函数返回值. 在 `v3-multi-step-parallel` 数据集上, *Async* 相比 *Sync* 平均增加 20 个 token, 在 3B 模型的 GPU 内存中增加约 90 ms 延迟和 27.5 MB KV cache, 少于总内存使用量的 5%.

**未与服务系统协同优化时的性能.** 在 OpenAI API 上实现 AsyncLM (*Async-Naive*) 对 GPT-4o 不实用, 因为首 token 时间 (TTFT) 延迟较高. 在 GPT-4o 上, *Async-Naive* 的延迟比 *Sync-Parallel* 高 1.5$\times$ ([图 6](#figure-06)). 延迟增加源于每次中断都要发起新的 API 调用, 平均 TTFT 为 310 ms. 有趣的是, 本地 Llama-3B 上的 *Async-Naive* 没有经历这么高的 TTFT, 平均 TTFT 只有 59 ms, 因而比 *Sync-Parallel* 快 1.1$\times$. 提示缓存或有状态推理等技术可能缓解 *Async-Naive* 的 TTFT 开销, 从而改善性能.

<span id="figure-07"></span>

![Llama-1B 和 Llama-3B 的陷阱处理策略图](./async-llm-function-calling/figure-07.png)

**图 7. 最小化内存中闲置 KV cache 且不牺牲延迟的陷阱处理策略图.** 最优策略取决于到下一个中断的估计时间和上下文中的 token 数量.

**存在任务依赖时的 LPT 启发式.** 实证实验表明, LPT 是一种有效的启发式方法. 在[图 6](#figure-06) 的本地 *Async* 评估中, LPT 平均比随机选择下一个函数快 8%. 不过, 当函数存在依赖关系时, LPT 不保证最优. 考虑函数 $a$、$b$ 和 $c$, 其中 $c$ 依赖于 $a$, 且它们的执行时间满足 $E(a)<E(b)<E(c)$. LPT 会安排 $b$-$a$-$c$, 因为它不考虑未来依赖, 而最优顺序是 $a$-$c$-$b$. 这一问题属于资源约束项目调度, 是一个 NP-hard 问题 [Bru99, Har22a]. LPT 是一种贪心启发式方法, 只考虑当前可用信息, 不预测未来依赖.

**陷阱处理策略.** 陷阱处理器可以根据上下文长度估计交换和重新计算的预期延迟, 从而减少闲置 KV cache 的内存使用. [图 7](#figure-07) 使用 KV cache 交换 (线性) 和重新计算 (二次) 的延迟模型, 分析了 AsyncLM 在 Llama-1B 和 Llama-3B 模型上的最优陷阱处理策略. 例如, 如果预计下一个中断将在 100 ms 后到达, 且上下文包含 300 个 token, 陷阱处理器可以在 1B 模型中丢弃 KV cache 并重新计算, 但在 3B 模型中由于重新计算开销更高, 应将其交换出去.

<span id="section-06-05"></span>

### 6.5 函数调用准确率

为了评估 AsyncLM 如何影响 LLM 生成准确函数调用的能力, 即正确的签名、参数和调用顺序, 我们使用多步并行数据集检查每个基线的函数调用轨迹. 我们确保提示中提供了所有必要的函数定义. 函数调用准确率通过将每个函数调用与真实调用进行精确 AST 匹配, 以及匹配其执行顺序来度量.

我们比较两种 LLM 适配策略: 微调和少样本提示. Llama 模型使用两种方法适配, 而 GPT-4o 只使用少样本提示适配. 为了理解 CML 语法的影响, 我们还测试了使用 CML 格式化 *Sync* 函数调用的场景. [表 1](#table-01) 展示了这些适配策略下 GPT-4o 和 Llama 模型的结果.

<span id="table-01"></span>

![同步和异步方法的函数组合准确率表](./async-llm-function-calling/table-01.png)

**表 1.** 多步并行函数调用中 *Sync* 和 *Async* 的平均函数组合准确率 (AST 匹配).  (FT) 表示经过微调的模型; (ICL) 表示少样本提示.

**CML 对准确率的影响.** 使用 CML 语法格式化 *Sync* 响应对准确率影响很小. 在 GPT-4o 中, 准确率几乎没有变化, 而 Llama-3B 略有提升. 使用 CML 时, 微调模型的准确率适度提高, 可能是因为微调样本使用了 CML 格式.

**异步函数调用对准确率的影响.** GPT-4o (使用少样本提示) 和经过微调的 Llama 模型在 *Sync* 和 *Async* 之间都保持相近的准确率, 这表明异步函数调用不会降低模型的函数调用准确率. 然而, 使用少样本提示适配的 Llama 模型从 *Sync* 切换到 *Async* 时准确率显著下降. 我们将其归因于小模型适应 CML 语法的上下文学习能力有限.

**微调与少样本提示对 LLM 适配的比较.** GPT-4o 可以通过少样本提示很好地适应异步函数调用, 而 GPT-4o-mini 等较小模型在 *Async* 中的准确率略有下降. 使用少样本提示的 Llama 模型总体准确率较低, 而微调显著改善了其性能. 值得注意的是, 经过微调的 Llama 模型在 *Async* 中的准确率高于 GPT-4o 在 *Sync* 中的准确率, 突出了微调对较小 LLM 的重要性.

<span id="section-06-06"></span>

### 6.6 讨论 — 人类触发的中断

<span id="figure-08"></span>

![用户触发中断时的延迟评估](./async-llm-function-calling/figure-08.png)

**图 8. 用户触发中断的评估.** 该实验重复了[图 6](#figure-06) 中的实验, 但每个任务由中断而不是提示表示, 到达时间为 0、200 和 400 ms.

AsyncLM 提出的中断机制不仅支持异步函数调用, 还支持新型的人类-LLM 和 LLM-LLM 交互. 我们先讨论这种机制的灵活性, 然后用真实示例说明它的潜力.

**中断的通用性和灵活性.** AsyncLM 将中断功能扩展到执行器通知之外, 允许中断表示外部事件, 例如用户输入或系统触发器. 这一能力支持实时交互, 用户可以在正在进行的 LLM 推理过程中中断它, 添加或调整任务, 而无需等待当前任务完成. 虽然本文没有详细介绍, 我们通过从多轮对话数据集中随机注入新任务, 将用户触发的中断纳入微调数据集. 虽然这一数据集可能无法完美代表真实的人类-LLM 交互, 我们仍在以下场景中探索了它的有效性.

**可中断的 AI 助手.** 当前的 LLM 聊天机器人以同步方式运行, 要求用户等待完整响应. 这种方法增加了感知延迟, 限制了实时任务处理. 借助 AsyncLM, 用户可以立即提出新请求, 即使 LLM 仍在生成响应. 例如, 用户常常不等待初始响应就提出后续请求. 用户可能先要求“在 Seattle 找一家酒店”, 然后很快补充“靠近 Space Needle”. 类似地, 用户可以纠正先前的请求, 例如在最初将会议安排在星期三后说“其实改成星期四”.

为了检验这一场景, 我们重复了多步并行函数调用实验, 将任务表示为一系列用户触发的中断而不是提示. 我们每隔 200 ms 插入这些中断, 并测量 *Async* 的端到端任务完成延迟, 如[图 8](#figure-08) 所示. 作为比较, 我们使用相同任务运行 *Sync* 和 *Sync-Parallel*, 二者都顺序处理任务. 与 *Sync* 相比, *Async* 将延迟降低了 2.4$\times$, 而 *Sync-Parallel* 仅降低 1.1$\times$. *Sync-Parallel* 的改善有限, 因为它要求一次生成全部函数调用, 不适合这一场景.

**多方通信的 LLM 智能体.** AsyncLM 还支持多个自主 LLM 智能体同时通信. 通常, 智能体之间的通信使用同步消息交换 [Wu23c, Cha24a], 智能体轮流行动. 这限制了它们同时处理多个任务的能力, 也使通信局限于轮询方式. 借助 AsyncLM, 可以使用中断机制实现一对多或多对多的智能体通信. 每个智能体都可以将消息作为中断发送给其他智能体, 从而实现更动态的交互, 并可能实现更逼真的基于 LLM 的社会模拟 [Zho24b, Par23]. 这种新的通信模式可能引入更多复杂性, 并要求 LLM 学会像人类对话那样判断何时插话.

<span id="section-07"></span>

## 7 结论

AsyncLM 通过让 LLM 与函数执行器彼此独立运行, 实现了异步 LLM 函数调用. 我们的核心创新是让 LLM 推理可被中断, 具体包括 (1) CML, 一个促进异步交互的上下文接口, (2) 让 LLM 利用异步语义以优化任务完成延迟的适配方法, 以及 (3) 在 LLM 推理流水线中高效实现中断机制. BFCL 基准上的实证评估表明, AsyncLM 相比同步方法将延迟降低了 1.6x–5.4x. AsyncLM 为提高与外部工具、数据源、人类和其他 LLM 交互的 LLM 运行效率开辟了新的可能性.

## 致谢

本工作部分得到 NSF Athena AI Institute (Award #2112562) 和 Yale University 的支持.

<span id="appendix-a"></span>

## 附录 A 延迟分析

本节分析一组独立函数在不同执行模型下的总延迟, 并给出证明草图, 以展示异步函数调用的优势.

### A.1 上下文与定义

令 $F=\{f_1,f_2,\dotsc,f_n\}$ 为包含 $n$ 个独立函数的集合. 每个函数 $f\in F$ 都有一个 token 生成延迟 $G(f)>0$ 和执行时间 $E(f)>0$. 我们定义平均生成延迟 $\overline{G}=\frac{1}{n}\sum_{f\in F}G(f)$ 和平均执行时间 $\overline{E}=\frac{1}{n}\sum_{f\in F}E(f)$.

不同执行模型下的总延迟如下.

- *同步执行 ($\mathrm{Sync}$)*: token 顺序生成, 函数顺序执行:

$$
L_{\mathrm{Sync}}(F)=\sum_{f\in F}G(f)+\sum_{f\in F}E(f)=n(\overline{G}+\overline{E}).
$$

- *并发同步执行 ($\mathrm{Sync}\text{-}\mathrm{Parallel}$)*: token 顺序生成, 函数并发执行:

$$
L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)=\sum_{f\in F}G(f)+\max_{f\in F}E(f).
$$

- 使用*最长处理时间 (LPT)* 启发式的*异步执行 ($\mathrm{Async}$)*: token 顺序生成, 函数按照 $E(f)$ 的降序调度:

$$
L_{\mathrm{Async}}(F)=\max_{f\in F}\left(E(f)+\sum_{g\in\mathrm{pred}(f,F)}G(g)\right),
$$

其中 $\mathrm{pred}(f,F)=\{g\in F\mid E(f)\leq E(g)\}$.

### A.2 定理 6.1 的证明草图

<span id="theorem-06-01-appendix"></span>

**定理 6.1.** 对任意独立函数集合 $F$, $L_{\mathrm{Async}}(F)\leq L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)<L_{\mathrm{Sync}}(F)$.

1. 建立 $L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)<L_{\mathrm{Sync}}(F)$:

由于最大执行时间小于所有执行时间之和,

$$
\max_{f\in F}E(f)<\sum_{f\in F}E(f),
$$

因此

$$
L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)<\sum_{f\in F}G(f)+\sum_{f\in F}E(f)=L_{\mathrm{Sync}}(F).
$$

2. 建立 $L_{\mathrm{Async}}(F)\leq L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)$:

在 LPT 启发式下, 函数的排序满足 $E(f_1)\geq E(f_2)\geq\dotsb\geq E(f_n)$. 函数 $f_i$ 的完成时间为

$$
C(f_i)=\left(\sum_{j=1}^{i}G(f_j)\right)+E(f_i).
$$

总延迟为 $L_{\mathrm{Async}}(F)=\max_{1\leq i\leq n}C(f_i)$. 对于所有 $i$, 有

$$
\sum_{j=1}^{i}G(f_j)\leq\sum_{f\in F}G(f),\quad E(f_i)\leq E(f_1)=\max_{f\in F}E(f),
$$

因此

$$
C(f_i)\leq\sum_{f\in F}G(f)+\max_{f\in F}E(f)=L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F),
$$

从而 $L_{\mathrm{Async}}(F)\leq L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)$.

### A.3 定理 6.2 的证明草图

<span id="theorem-06-02-appendix"></span>

**定理 6.2.** 当 $n=|F|$ 较大时, 比值 $\frac{L_{\mathrm{Sync}}(F)}{L_{\mathrm{Async}}(F)}$ 约为 $1+\frac{\overline{E}}{\overline{G}}$, 误差阶为 $\left(\frac{\overline{E}}{\overline{G}}\right)^2$.

当 $n$ 较大时, 假设 $E(f)$ 服从正态分布, 我们可以使用顺序统计量近似累积 token 生成延迟和最大执行时间:

$$
\sum_{f\in F}G(f)\approx n\overline{G},\quad \max_{f\in F}E(f)\approx\overline{E}+\sigma\sqrt{2\ln n}.
$$

利用不等式 $L_{\mathrm{Async}}(F)\leq L_{\mathrm{Sync}\text{-}\mathrm{Parallel}}(F)$, 我们将总延迟近似为

$$
L_{\mathrm{Async}}(F)\approx n\overline{G}+\overline{E}+\sigma\sqrt{2\ln n}.
$$

于是总延迟之比为

$$
\frac{L_{\mathrm{Sync}}(F)}{L_{\mathrm{Async}}(F)}\approx\frac{n\overline{G}+n\overline{E}}{n\overline{G}+\overline{E}+\sigma\sqrt{2\ln n}}.
$$

当 $n$ 较大时, 由于 $\varepsilon=\dfrac{\overline{E}+\sigma\sqrt{2\ln n}}{n\overline{G}}\ll 1$, 我们可以近似 $\dfrac{1}{1+\varepsilon}\approx 1-\varepsilon$. 应用该近似,

$$
\frac{L_{\mathrm{Sync}}(F)}{L_{\mathrm{Async}}(F)}\approx\left(1+\dfrac{\overline{E}}{\overline{G}}\right)\left(1-\dfrac{\overline{E}+\sigma\sqrt{2\ln n}}{n\overline{G}}\right).
$$

忽略 $O(1/n)$ 阶的项, 得到

$$
\frac{L_{\mathrm{Sync}}(F)}{L_{\mathrm{Async}}(F)}\approx 1+\dfrac{\overline{E}}{\overline{G}}.
$$

该近似的误差阶为 $\left(\dfrac{\overline{E}}{\overline{G}}\right)^2$.

### A.4 定理 6.3 的证明草图

<span id="theorem-06-03-appendix"></span>

**定理 6.3.** 任何偏离 LPT 顺序的安排都不可能得到更低的总延迟.

反设存在一个偏离 LPT 顺序的调度 $\sigma'$, 其总延迟低于 LPT 调度 $\sigma^*$. 这意味着在 $\sigma'$ 中存在函数 $f_i$ 和 $f_j$, 满足 $E(f_i)<E(f_j)$, 但 $f_j$ 被安排在 $f_i$ 之后.

设在 $\sigma'$ 中 $f_i$ 和 $f_j$ 之前的累积 token 生成延迟分别为 $S_i$ 和 $S_j$, 且 $S_j=S_i+G(f_i)$. 完成时间为:

$$
C_{\sigma'}(f_j)=S_j+G(f_j)+E(f_j)=S_i+G(f_i)+G(f_j)+E(f_j).
$$

考虑交换 $f_i$ 和 $f_j$ 以得到调度 $\sigma''$. 对于 $f_j$, token 生成延迟之和变为 $S_j''=S_i$; 对于 $f_i$, 则为 $S_i''=S_j''+G(f_j)$. 新的完成时间为

$$
C_{\sigma''}(f_i)=S_i''+G(f_i)+E(f_i)=S_i+G(f_j)+G(f_i)+E(f_i).
$$

比较交换前后的完成时间:

$$
C_{\sigma''}(f_i)-C_{\sigma'}(f_i)=E(f_i)-E(f_j)<0.
$$

因此, 交换 $f_i$ 和 $f_j$ 总会降低最大完成时间. 反复进行这样的交换, 可以在不增加总延迟的情况下将任意调度转换为 LPT 调度. 这与 $\sigma'$ 的总延迟低于 $\sigma^*$ 的假设矛盾. 因此, 任何偏离 LPT 顺序的安排都不可能得到更低的总延迟.
