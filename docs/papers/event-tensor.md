---
title: 'Event Tensor: Dynamic Megakernel Compiler'
createTime: 2026/09/14 22:50:53
permalink: /papers/event-tensor/
---

> [Hongyi Jin](https://dblp.org/pid/321/1565), [Bohan Hou](https://spectrometerhbh.github.io/), [Guanjie Wang](https://dblp.org/pid/166/5270.html), [Ruihang Lai](https://ruihanglai.com/), [Jinqi Chen](https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/Posters/MSCSThesis-5-JinqiChen25.pdf), [Zihao Ye](https://expye.com/), [Yaxing Cai](https://dblp.org/pid/290/7679.html), [Yixin Dong](https://github.com/Ubospica), [Xinhao Cheng](https://www.csd.cs.cmu.edu/people/doctoral-student/xinhao-cheng), [Zhihao Zhang](https://jackfram.github.io/), [Yilong Zhao](https://ylzhao.me/), [Yingyi Huang](https://mlsys26.flashinfer.ai/), [Lijie Yang](https://derrickylj.github.io/), [Jinchen Jiang](https://dblp.org/pid/307/3686.html), [Gabriele Oliaro](https://www.gabrieleoliaro.com/), [Jianan Ji](https://jiananji.me/), [Xupeng Miao](https://hsword.github.io/), [Vinod Grover](https://developer.nvidia.com/blog/author/vgrover/), [Todd C. Mowry](https://www.cs.cmu.edu/~tcm/), [Zhihao Jia](https://www.cs.cmu.edu/~zhihaoj2/) 与 [Tianqi Chen](https://tqchen.com/) 著. 论文于 2026 年 4 月 14 日首次提交至 arXiv, 当前版本为 2026 年 4 月 21 日修订的 v2, 并发表于 [Proceedings of Machine Learning and Systems 8 (MLSys 2026)](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html). 本阅读版转录自 [*Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel*](https://arxiv.org/abs/2604.13327v2), 并提供<a href="/paper/event-tensor.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>, [arXiv DOI](https://doi.org/10.48550/arXiv.2604.13327) 与 [TeX 源码](https://arxiv.org/src/2604.13327v2). 精确的印刷版式与参考文献仍以原始 PDF 为准.

## 摘要

现代 GPU 工作负载, 尤其是大语言模型 (LLM) 推理, 会受到内核启动开销和粗粒度同步的影响, 内核间并行性因而受限. 近期的巨型内核技术将多个算子融合进单个持久化内核, 以消除启动间隙并暴露内核间并行性, 但在处理真实工作负载中的动态形状和数据依赖计算时仍然力有不逮. 本文提出 *Event Tensor*, 一种面向动态巨型内核的统一编译器抽象. Event Tensor 对分块任务之间的依赖关系进行编码, 并将形状动态性和数据依赖动态性都作为一等能力支持. 基于这一抽象, 我们的 Event Tensor Compiler (ETC) 通过静态与动态调度变换生成高性能持久化内核. 评测表明, ETC 在显著降低系统预热开销的同时, 实现了当前领先的 LLM 服务延迟.

<span id="section-1"></span>

## 1 引言

高效部署机器学习 (ML) 应用, 需要尽可能降低延迟并提高硬件利用率, 因此系统性能优化 [Kwo23, Zhu25c, Ye25, Zho24] 已成为重要的研究前沿. 随着 GPU 的速度和并行能力持续提升, 传统 GPU 调度模型中的多种系统开销逐渐成为制约端到端效率的主要瓶颈.

第一类开销来自内核启动. PyTorch [Pas19a] 等现有系统由主机 CPU 依次启动 GPU 内核 ([图 1](#figure-01), 左上). 在 LLM 推理期间, 每个自回归解码步骤可能涉及数百乃至数千个细粒度操作, 启动开销难以得到有效摊销. 一次内核启动通常会产生 5-10 $\mu$s 延迟, 而最快的内核可能只需 2 $\mu$s 即可完成, 于是启动开销反而占据主导.

第二类开销来自内核边界, 它会在相邻内核之间强制执行隐式同步. 很多时候, 后续内核只依赖先前内核的一部分结果; 原则上, 这些内核可以重叠或形成流水线, 以提高吞吐量. 然而, 内核之间的边界阻碍了这种细粒度内核间并行, 造成了显著的性能损失.

<span id="figure-01"></span>

![图 1. 不同的 GPU 调度模型. 逐内核调度与 CUDA Graph 调度模型会强制执行粗粒度的串行过程. 巨型内核把操作拆成更小的任务, 从而实现内核间并行.](./event-tensor/figure-01.png)

**图 1.** 不同的 GPU 调度模型. 逐内核调度与 CUDA Graph 调度模型会强制执行粗粒度的串行过程. 巨型内核把操作拆成更小的任务, 从而实现内核间并行.

<span id="figure-02"></span>

![图 2. Event Tensor 抽象概览. 计算图 (左) 被划分为分块算子 (*任务*), Event Tensor 则把任务间的细粒度依赖表示为具有符号形状的一等对象, 用以处理 LLM 服务中两类主要的动态性来源: 形状动态性: 分块张量与 Event Tensor 均可包含符号维度, 例如动态批大小 **B**. 数据依赖动态性: 右侧 MoE 层展示了如何在运行时解析依赖. 依赖数据的更新与触发 (黄色箭头) 使用 `topk` 和 `exp_indptr` 等运行时计算值, 动态管理任务执行.](./event-tensor/figure-02.png)

**图 2.** Event Tensor 抽象概览. 计算图 (左) 被划分为分块算子 (*任务*), Event Tensor 则把任务间的细粒度依赖表示为具有符号形状的一等对象, 用以处理 LLM 服务中两类主要的动态性来源: 形状动态性: 分块张量与 Event Tensor 均可包含符号维度, 例如动态批大小 **B**. 数据依赖动态性: 右侧 MoE 层展示了如何在运行时解析依赖. 依赖数据的更新与触发 (黄色箭头) 使用 `topk` 和 `exp_indptr` 等运行时计算值, 动态管理任务执行.

近来已有一些工作试图部分解决这些局限. 首先, 许多系统采用 CUDA Graphs [Gra19] ([图 1](#figure-01), 右上) 等运行时技术, 通过捕获并回放固定的内核序列来降低内核启动开销. 然而, CUDA Graphs 保留了内核边界, 因而无法暴露内核间并行性.

更近期的 *巨型内核* 优化 [Spe25, Che25ae] 成为一种很有前景的替代方案 ([图 1](#figure-01), 下方). 其核心思想是把多个算子融合进单个持久化内核, 消除内核启动开销并实现内核间并行. 每个算子都会被分解为细粒度计算块, 也就是 *任务*, 并分配到各个流式多处理器 (SM) 上. 这些任务及其依赖组成任务图, 再由轻量级运行时信号协调执行, 在保持依赖关系的同时尽量提高并发度.

尽管前景可观, 将巨型内核用于 LLM 推理工作负载仍面临两项关键挑战:

**动态性挑战.** 现代 LLM 服务工作负载本身就是动态的. 引入连续批处理后, 系统必须应对可变的输入形状. 在单个巨型内核中支持这类动态形状并不容易, 因为它往往要求为每种可能的形状重新生成或重新编译内核. 这可能带来难以接受的启动延迟; 当潜在形状空间过大时, 甚至根本不可行. 此外, 混合专家 (MoE) 等模型包含依赖数据的控制流 (例如专家路由), 为了利用算子间并行性, 需要动态追踪细粒度任务依赖. 现有方法缺少表达这类细粒度数据依赖的抽象. 值得注意的是, 这项挑战并非巨型内核独有——运行时动态性同样给 CUDA Graphs 等传统方法带来很大困难; 在不同动态形状上反复捕获和管理 CUDA Graphs, 是生产级 LLM 服务系统的一大痛点. 对实时智能体工作流, 交互式编码助手等新兴的延迟敏感应用而言, 这些动态性挑战尤其重要, 因为这类应用以低批量推理为主, 而内核间并行是降低单次请求延迟的关键.

**可编程性挑战.** 巨型内核编程相当复杂. 开发者必须处理任务之间繁复的细粒度依赖, 这既容易出错, 也难以维护; 形状动态性和数据依赖动态性又让问题更加棘手. 此外, 不同工作负载可能适合不同的任务管理策略. 例如, *静态调度* 会在内核启动前为每个 SM 分配预先确定的任务队列, 而 *动态调度* 则由 GPU 上的调度器在运行时分派就绪任务. 理想情况下, 开发者应当能够顺畅地选择或组合这些调度策略, 而不必重新实现整个内核, 从而针对工作负载选择合适的调度方式.

本文提出 **Event Tensor** ([图 2](#figure-02)), 一种旨在简化动态巨型内核编译与执行的抽象. 我们把 *事件* 定义为一种原语, 用来表示一组任务在 GPU SM 粒度上的完成状态. 巨型内核会把算子划分成大量分块级任务, 因此相应的同步事件自然形成类似数据张量的多维结构. *Event Tensor* 是由这类事件组成的多维数组, 为巨型内核中的细粒度同步提供紧凑的一等表示. 基于信号量的同步虽是众所周知的原语, 但我们的核心创新在于把这些原语提升为编译器 IR 中的一等张量. Event Tensor 将事件统一为张量形式, 可以复用现有编译器对符号形状的支持 [Ans24, Lai25a], 让张量维度保持符号化, 从而紧凑表示动态形状计算. 此外, Event Tensor 通过把任务坐标映射到事件坐标的索引表达式来表达数据依赖动态性. 基于这一抽象, ETC 会自动把这些依赖变换为高度优化的持久化巨型内核, 将以往依赖人工专项工程的优化推广为通用方法.

在 Event Tensor 抽象之上, 我们构建了一套系统化编译流程, 自动融合并调度算子, 实现内核间并行. 流程从带有显式算子和 Event Tensor 标注的计算图出发, 通过一系列调度变换, 把程序逐步降低为可执行的巨型内核. 这些变换支持多种调度策略, 从完全静态到运行时动态负载均衡; 每种策略都对应同步开销与运行时适应能力之间不同的取舍. ETC 将以往依赖手工完成的融合和定制调度技术 [Spe25, Che25ae] 统一为编译器变换, 既显著减少了构建巨型内核所需的工程投入, 也提高了运行性能.

我们在多种 LLM 服务工作负载上评测 ETC, 并与极具竞争力的工业级基线 (例如 vLLM 和 SGLang) 比较; 这些基线已经采用 CUDA Graphs, Programmatic Dependent Launch (PDL) 和 `torch.compile` 等激进优化. 结果表明, ETC 在高效支持形状动态性与数据依赖动态性的同时, 相较这些系统取得显著加速, 而且不需要在运行时重新捕获计算图或重新编译. 对于张量并行工作负载, 由编译器驱动的计算与通信重叠, 使融合 GEMM 与 Reduce-Scatter 内核获得最高 1.40x 加速. 对于 MoE 等数据依赖工作负载, 我们的巨型内核比专用库最高快 1.23x. 在动态形状, 低批量推理场景中, ETC 的性能达到或超过这些高度优化的推理系统, 同时将引擎预热开销最多降低 3.5x. 面对这些强劲基线, 即使只是中等幅度的加速, 放大到数据中心规模也能产生可观的经济价值. 除纯粹的速度外, ETC 还对动态工作负载实现了真正的提前 (AOT) 编译, 完全消除运行时编译开销, 以及反复捕获 CUDA Graph 所带来的管理复杂度——后者正是生产服务系统的一大痛点. ETC 还会自动融合包含复杂数据依赖的子图 (例如 MoE 层, GEMM + 通信), 显著降低编程复杂度, 同时仍能与现有服务引擎组合使用. ETC 已被整合进一个重要的开源系统. Event Tensor 抽象简洁而通用, 配套的编译器框架也能惠及更广泛的机器学习系统与编译器社区.

<span id="section-2"></span>

## 2 Event Tensor 抽象

<span id="section-2-1"></span>

### 2.1 语言构件

<span id="figure-03"></span>

![图 3. 基于 Event Tensor 的示例程序.](./event-tensor/figure-03.png)

**图 3.** 基于 Event Tensor 的示例程序.

下面先介绍基于 Event Tensor 的程序中的主要语言构件.

**设备函数.** 设备函数定义一组在 GPU 上并行启动的任务网格. 每次启动都由一个多维坐标参数化, 其中每个坐标标识一个在流式多处理器 (SM) 上执行的任务块. 每项任务都可以包含专门逻辑, 例如 warp 专门化或张量核心调用.

**Event Tensor.** Event Tensor 是一种多维结构, 其元素表示事件, 也就是任务集合在 SM 层级上的完成状态; 这一设计沿用了并行编程系统中的成熟实践 [Blu95, Tre14, Bau12, Dag98]. 每个元素都有一个初始等待计数, 用于记录它所依赖的任务数, 并支持多种操作: `E[i].notify()` 表示任务完成, `E[i].wait()` 阻塞至事件触发; 在动态调度中, 事件还可以触发依赖它的任务. 与逐个管理独立事件的方法相比, 在真实 LLM 推理工作负载扩展到数百万个细粒度事件时, Event Tensor 能大幅降低任务图管理开销.

**图函数.** 图函数表示由 `call_device` 调用组成的计算图, 每个调用都会以指定的任务形状显式启动设备函数. 与传统计算图不同, 它同时包含数据张量和 Event Tensor. 每次设备函数启动都可以标注显式的输入/输出依赖与坐标映射, 通过 Event Tensor 追踪细粒度任务关系.

[图 3](#figure-03) 展示了一个基于 Event Tensor 的示例程序. 一般而言, 可以把程序看作任务图的一种紧凑表示, 其形式为"生产者任务 $\rightarrow$ 事件 $\rightarrow$ 消费者任务". 我们用通用 lambda 函数表示事件与任务之间的关系. [+1] 这些依赖标注会隐式映射为每个生产者任务结束时的事件通知, 以及每个消费者任务开始时的等待. 我们发现, 这套记法足以覆盖大多数用例. 更重要的是, 我们也允许设备函数把 Event Tensor 显式作为参数, 并在每项任务内部调用事件的通知与等待操作. 设备函数将 Event Tensor 作为一等对象支持, 使我们得以表示更高级的用例. 它还允许把显式融合优化表示为该表示内部的变换, 下一节将对此展开讨论.

<span id="figure-04"></span>

![图 4. **Event Tensor 通过符号形状张量处理形状动态性**, 这些张量定义了依赖图模板. 在运行时, 模板会使用具体形状值实例化 (例如, 批大小为 1 时生成 $1\times 2$ 图, 批大小为 2 时生成 $2\times 2$ 图), 无须重新编译或反复捕获计算图.](./event-tensor/figure-04.png)

**图 4.** **Event Tensor 通过符号形状张量处理形状动态性**, 这些张量定义了依赖图模板. 在运行时, 模板会使用具体形状值实例化 (例如, 批大小为 1 时生成 $1\times 2$ 图, 批大小为 2 时生成 $2\times 2$ 图), 无须重新编译或反复捕获计算图.

<span id="section-2-2"></span>

### 2.2 表示细粒度依赖

为了说明 Event Tensor 在实践中的用法, 我们详细分析 [图 3](#figure-03) 所示的例子. 该任务图沿输入张量 $A$ 的内轴求和; $A$ 的符号形状为 $(n\times 32,128)$: $C[i]=\sum_{k\in[0,128)}A[i,k]$. 示例采用 split-K 算法, 把求和分为两个阶段: $B[i,j]=\sum_{k\in[j*32,j*32+32)}A[i,k]$ 和 $C[i]=\sum_{k\in[0,4)}B[i,k]$. 第一阶段把每一行的部分和计算到 $B$ 中, 第二阶段再汇总它们, 得到 $C$. 在传统的逐内核方法中, 只有第一阶段的全部任务完成后, 第二阶段的任务才会启动. 然而, 每个输出行 $C[i]$ 只依赖对应的行 $B[i,:]$, 所以它的计算可以与其他行的部分和并行进行. 为捕获这种细粒度依赖, 我们把 $B$ 和 $C$ 的计算划分为更细的任务, 并引入 Event Tensor $E$:

$$
\begin{aligned}
\text{任务 }\hat{B}_{i,j}: && B[i*32:i*32+32,j], \\
\text{事件 }E_{i}: && E[i], \\
\text{任务 }\hat{C}_{i}: && C[i*32:i*32+32].
\end{aligned}
$$

完成任务划分后, 依赖关系变为 $\hat{B}_{i,j}\rightarrow E_{i}$ ($\hat{B}_{i,j}$ 产生 $E_{i}$) 和 $E_{i}\rightarrow\hat{C}_{i}$ ($\hat{C}_{i}$ 消费 $E_{i}$). 每个 $E_{i}$ 对应从第 $32*i$ 行开始, $B$ 中连续 32 行的完成状态.

`main_graph` 函数描述整体计算, 其中原语 `call_device` 先启动 `partial_sum` 函数, 再启动 `final_sum` 函数. `partial_sum` 与 `final_sum` 任务之间的依赖通过 `out_edges` 和 `in_edges` 参数指定.

<span id="section-2-3"></span>

### 2.3 将动态形状作为一等能力支持

基于 Event Tensor 的图可以看作并行系统中任务图表示 [Tre14] 的紧凑版本; 后者把每个事件元素和每项任务都显式表示为独立节点与边, 并在运行时将其物化为任务图. 而在我们的方法中, 一个张量就能表示数千个事件. Event Tensor 表示还让我们能够把符号维度作为一等能力引入. 例如, Event Tensor 的形状可以包含批大小或序列长度等符号变量. 符号形状的 Event Tensor 图充当通用模板, 在运行时对应不同的任务图 ([图 4](#figure-04)). 借助这种强大的表示, 我们可以克服 CUDA Graph 等静态任务图系统的局限, 提前优化适用于多种形状的动态形状 Event Tensor 图, 无须重新编译或反复捕获计算图, 从而更灵活地处理动态形状工作负载.

<span id="section-2-4"></span>

### 2.4 支持数据依赖动态性

<span id="figure-05"></span>

![图 5. **Event Tensor 对数据依赖动态性的处理**. (a) 具有静态依赖的规则工作负载. (b) 数据依赖的 MoE 工作负载, 其中运行时张量 `topk` 和 `exp_indptr` 定义不规则任务图, 并支持依赖数据的事件更新与任务触发.](./event-tensor/figure-05.png)

**图 5.** **Event Tensor 对数据依赖动态性的处理**. (a) 具有静态依赖的规则工作负载. (b) 数据依赖的 MoE 工作负载, 其中运行时张量 `topk` 和 `exp_indptr` 定义不规则任务图, 并支持依赖数据的事件更新与任务触发.

现代工作负载的一项关键挑战, 是如何处理不规则, 依赖数据的任务图. 混合专家 (MoE) 层 [Sha17] 就是典型例子: 输入 token 会根据运行时计算得到的路由决策, 动态分派到不同的专家子网络. 高效的 MoE 实现通常先按分配到的专家对 token 分组, 再使用 GroupGEMM 算子计算各组内所有 token 的结果.

动态路由带来了编译时未知的细粒度数据依赖任务关系 (具体来说, 就是哪个专家的哪一个 GroupGEMM 块负责处理哪些 token). 传统编译器和调度器假定任务图及其依赖固定不变, 因此很难应对这种动态性. 要高效表示此类动态工作负载, 需要一种抽象, 能够 (1) 在运行时确定每项消费者任务依赖哪些任务, 并 (2) 根据运行时数据触发数量可变的消费者任务. Event Tensor 抽象正是为此而设计. 它通过两项核心机制管理依赖数据的任务图:

**依赖数据的事件更新.** 传统任务图只支持静态依赖 ([图 5](#figure-05)a), 而 Event Tensor 抽象允许动态事件依赖 ([图 5](#figure-05)b). 在 MoE 示例中, 保存在 `topk` 张量中的运行时路由决策, 决定哪些分组块 (每个 token 一个) 更新哪些事件 (每个专家一个). 每个专家的事件计数器初始化为路由至该专家的 token 数, 这项初始化与 `topk` 的计算一起在运行时动态完成.

**依赖数据的任务触发.** 同样, 一个事件可以触发数量由运行时决定的任务. 根据 `topk` 中的路由决策, 我们可以计算每个专家需要处理多少 token, 进而确定每个专家需要多少个 GroupGEMM 块. 如 [图 5](#figure-05)b 所示, 这些信息编码在张量 `exp_indptr` 中; 该张量保存每个专家待触发 GroupGEMM 块数的前缀和. [+2] 借助 `exp_indptr`, 我们实现了依赖数据的触发: 专家 `i` 会激活区间 `(exp_indptr[i], exp_indptr[i+1])` 内的块.

这两项机制共同使编译器能够生成高效适应高度动态工作负载的巨型内核, 而静态任务图很难处理这类情形. 再结合前文所述的符号形状支持, ETC 的全部编译都在线下完成; 在推理时, 编译后的二进制文件以零编译开销同时处理形状动态性和数据依赖动态性.

需要注意, 整体依赖链仍然严格前馈 ([图 2](#figure-02), 右): Attention Output $\rightarrow$ Token Routing (TopK) $\rightarrow$ Token Grouping $\rightarrow$ Token Computation (GroupGEMM). TopK 的计算只依赖此前的 Attention 输出, 这是标准的静态依赖. 上文所述的数据依赖 Event Tensor 机制只负责后续阶段: 路由结果动态决定哪些分组任务通知哪些专家事件, 以及每个专家触发多少个 GroupGEMM 块.

<span id="section-3"></span>

## 3 Event Tensor 编译

本节介绍 ETC 中利用 Event Tensor 抽象的优化.

<span id="section-3-1"></span>

### 3.1 静态调度与变换

静态调度通过提前把任务显式分配到各个流式多处理器 (SM), 将多个设备函数融合在一起. 因此, 每项任务都会预先分配到某个特定 SM 的任务队列. 任务依赖则由基于计数器的信号量和事件触发等待等底层同步原语管理. 这种方法的同步开销极低, 尤其适合能够提前优化 SM 分区的可预测工作负载.

<span id="figure-06"></span>

![图 6. 静态调度变换前后的 GEMM + Reduce-Scatter. 两个独立设备函数被融合为一个持久化函数, 并通过对 Event Tensor 的显式 `notify` 与 `wait` 调用协调依赖.](./event-tensor/figure-06.png)

**图 6.** 静态调度变换前后的 GEMM + Reduce-Scatter. 两个独立设备函数被融合为一个持久化函数, 并通过对 Event Tensor 的显式 `notify` 与 `wait` 调用协调依赖.

<span id="figure-07"></span>

![图 7. 静态调度的通知与等待机制.](./event-tensor/figure-07.png)

**图 7.** 静态调度的通知与等待机制.

我们在 ETC 中通过三个主要步骤实现静态调度变换 ([算法 1](#algorithm-01)): (1) 在主机上为每个 SM 构建执行队列; (2) 生成持久化主循环, 让每个 SM 无须重新启动便可持续执行任务; (3) 把 Event Tensor 依赖降低为显式的 `notify()` 和 `wait()` 调用, 以落实细粒度执行顺序. [图 6](#figure-06) 以张量并行执行中的基础组合 GEMM + Reduce-Scatter 为例, 展示了这一过程. 它把设备函数融合为单个持久化内核; 其主循环不断从 `tile_scheduler` 的预计算队列中取出任务, 并在 GEMM 任务末尾发出 `notify()`, 在 Reduce-Scatter 任务开头执行 `wait()`.

[图 7](#figure-07) 展示了静态融合的 GEMM (MM) + Reduce-Scatter (RS) 内核在实践中的运行方式. 每项 RS 任务依赖两项 MM 任务 (也就是说, 一个 RS 块覆盖的大小是 MM 块的两倍), 所以每个事件的初始计数为 2. 在 $T_{1}$ 时, SM0 上的 MM0 完成并通知 Event Tensor, 把计数降为 1. 静态安排在 SM0 上紧随其后的 RS 任务尚不能继续, 于是进入自旋等待状态. 在 $T_{1}$ 与 $T_{2}$ 之间, SM1 继续执行 MM0, 使 GPU 保持忙碌. 在 $T_{2}$ 时, SM1 上的 MM0 完成, 将计数减为 0, 满足依赖条件; RS 任务随即退出等待循环, 并开始在 SM0 上执行.

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**算法 1: ETC 中的静态调度变换.**

- **输入:** 包含分块级数据流图 `G` 的模块 `mod`, 其中带有 Event Tensor 依赖.
- **输出:** 更新后的模块, 其中包含融合并静态调度的巨型内核.
- `mod_updated` $\leftarrow$ `mod.Copy()`.
- `static_schedule` $\leftarrow$ `GenerateStaticSchedule(G)`.
- `fused_kernel` $\leftarrow$ `NewPersistentKernel()`.
- 将预计算的调度嵌入全局内存.
- `fused_kernel.AddBuffer(static_schedule)`.
- **对所有** `G` 中的 `task_grid`:
  - `fused_kernel.AddDispatchLogic(task_grid)`.
  - **对所有** `task_grid.in_edges` 中的 `event`:
    - `fused_kernel.AddWaitLogic(event)`.
  - `fused_kernel.AddTileLogic(task_grid)`.
  - **对所有** `task_grid.out_edges` 中的 `event`:
    - `fused_kernel.AddNotifyLogic(event)`.
- `mod_updated.Replace(G, fused_kernel)`.
- **返回** `mod_updated`.

</div>

普通静态调度无法自然支持动态工作负载. 为处理形状动态性, 我们采样一组有代表性的形状; 未见过的形状会复用下一个更大采样值的执行队列. 为处理数据依赖动态性, 我们通过把相关的 `notify()` 和 `wait()` 操作重写为 `E[0].notify()` 和 `E[0].wait()`, 保守地按 Event Tensor 更新与触发的最坏情况处理. 为简化实现, 我们采用轮询策略构建执行队列.

<span id="section-3-2"></span>

### 3.2 动态调度与变换

<span id="figure-08"></span>

![图 8. 动态调度变换后的 GEMM + Reduce-Scatter. 代码中插入了任务 `push` 和 `pop`, 由调度器动态协调任务执行.](./event-tensor/figure-08.png)

**图 8.** 动态调度变换后的 GEMM + Reduce-Scatter. 代码中插入了任务 `push` 和 `pop`, 由调度器动态协调任务执行.

当任务执行时间不可预测时, 动态调度能改善 SM 之间的负载均衡. ETC 使用一个轻量级 GPU 端任务调度器, 实现基于 Event Tensor 的动态调度. 当一个事件在其依赖的所有任务完成后被触发时, 它会以原子方式把所有关联的消费者任务 **推入** 调度器, 将其标记为可执行. 随后, 任意空闲 SM 都可以原子地 **弹出** 一项就绪任务并执行.

<span id="figure-09"></span>

![图 9. 动态调度的推入与弹出机制.](./event-tensor/figure-09.png)

**图 9.** 动态调度的推入与弹出机制.

我们在编译器中实现了动态调度变换 [+3]. [图 8](#figure-08) 展示了 [第 3.1 节](#section-3-1) 所述同一 GEMM (MM) + Reduce-Scatter (RS) 示例变换后的代码. [图 9](#figure-09) 则展示推入—弹出机制. 在 $T_{1}$ 时, SM0 上的 MM0 完成, 把事件计数器减为 1; 此时空闲的 SM0 会立即从调度器弹出一项就绪任务 (MM1). 在 $T_{2}$ 时, SM1 上的 MM0 完成, 把计数器降为 0, 并触发 RS 任务推入调度器. 接着, SM1 弹出 RS 任务并开始执行. 整个依赖追踪与任务分派过程都在 GPU 上高效进行, 无须由主机预计算任务队列.

动态调度天然支持两类动态性, 因为分块执行顺序会在运行时即时决定; 此时符号形状值和运行时值已经由调度器解析. 我们的推入—弹出接口使用由所有 SM 共享, 位于全局内存中的集中式队列. 选择这一设计是因为它实现简单, 不过我们也承认, 它在大规模环境下可能发生争用. [第 11 节](#section-11) 还会讨论动态调度器的运行时优化.

**静态与动态调度之间的取舍.** 静态与动态调度的选择体现了一项经典取舍. 静态调度的调度开销最低, 适合可预测的工作负载. 相比之下, 动态调度对依赖数据的动态工作负载或不可预测的任务完成时间更灵活, 能够自然实现负载均衡, 但要承担少量任务队列推入与弹出的运行时开销.

<span id="section-3-3"></span>

### 3.3 降低到最小运行时

编译器中的静态与动态调度, 让我们可以直接把底层任务依赖及其处理封装进变换后的程序, 因此只需要极少的配套运行时支持 ([图 10](#figure-10)). 具体来说, 每个 Event Tensor 都会被降低为整数张量, 复用现有张量数据结构, 无须为事件提供专用运行时数据结构. 整数张量上的 `notify()` 和 `wait()` 操作由高效的硬件原子操作实现: `notify()` 执行原子递减, 而 `wait()` 则自旋等待计数器归零. 我们的运行时数据状态只包含这些整数张量和调度器的任务队列. 与典型任务图方法相比, 这种将逻辑编译进内核的方式所需运行时更小; 典型方法必须把完整任务图物化在内存中, 并依赖通用任务执行器遍历图, 启动设备函数.

<span id="figure-10"></span>

![图 10. 运行时架构对比. (a) 传统运行时执行器把任务图物化在内存中, 只编译分块算子. (b) ETC 把调度逻辑编译进巨型内核, 无须在运行时物化任务图.](./event-tensor/figure-10.png)

**图 10.** 运行时架构对比. (a) 传统运行时执行器把任务图物化在内存中, 只编译分块算子. (b) ETC 把调度逻辑编译进巨型内核, 无须在运行时物化任务图.

<span id="section-3-4"></span>

### 3.4 端到端编译流程

ETC 的端到端编译流程 [+4] 从未经优化的计算图开始; 图中已定义 Event Tensor, 算子也已被划分为 CTA 级块——可以由用户通过 Triton [Til19] 等内核 DSL 指定, 也可以由编译器内置提供. 在我们的实现中, 设备函数用基于 TVM 的 DSL [Hou26] 编写, 支持标准分块编程. 需要强调的是, Event Tensor 抽象与 DSL 无关: 它的依赖图和调度逻辑可以集成进其他编译器栈 (例如 Triton, CuteDSL), 不会产生根本性的设计冲突. op 图首先进行标准图级优化, 包括类似现有深度学习编译器 [Sab20, Lat21, Lai25a, Ans24] 的内存规划. 随后, 分块级优化通过确定硬件指令映射, 流水线策略等底层细节, 细化每个算子. 接着, 图会经过 [第 3.1 节](#section-3-1) 与 [第 3.2 节](#section-3-2) 所述的静态或动态调度变换. 最终得到的融合设备函数以持久化内核形式生成 GPU 代码. 之后的预取 pass 根据用户标注, 为分块算子生成权重预取函数, 使每个块能在输入激活到达前预取权重. 最后, 若选择静态调度, 编译器会计算各 SM 的任务顺序, 并将其物化为巨型内核的静态执行队列.

<span id="section-4"></span>

## 4 评测

我们在 Apache TVM 之上将 ETC 实现为一系列编译器 pass. 值得注意的是, 本文提出的抽象同样可以应用到其他编译器. 本节通过评测回答以下几个关键问题:

- Event Tensor 抽象管理细粒度依赖的效果如何, 无论工作负载采用静态任务图 ([第 4.1 节](#section-4-1)), 还是具有形状动态性与数据依赖动态性的任务图 ([第 4.2 节](#section-4-2))?
- 在动态低批量服务场景中, 由 ETC 编译的巨型内核能否取得更低的端到端延迟? ([第 4.3 节](#section-4-3))
- Event Tensor 对形状动态性的支持, 能否消除静态形状运行时即时 (JIT) 编译与计算图捕获系统所需的大量引擎预热开销? ([第 4.4 节](#section-4-4))
- 静态调度与动态调度在不同工作负载上是否体现出不同的性能取舍? ([第 4.5 节](#section-4-5))

所有实验均在一台配备 8 张 NVIDIA B200 GPU, 通过 NVLink 互连的服务器上进行; 系统运行 Ubuntu 24.04, PyTorch 2.8.0, CUDA 13.0, 驱动版本为 580.82.07. 我们选择 B200, 是因为它代表当前领先的硬件; Event Tensor 抽象工作在编译器 IR 层, 并不局限于某一代 GPU. 我们将 ETC 与当前领先的深度学习编译器, 专用库和高性能 LLM 服务系统比较. 现有巨型内核框架专为单批量推理设计, 因此无法在动态形状或数据依赖工作负载下进行公平比较.

<span id="section-4-1"></span>

### 4.1 融合通信与计算的性能

为评估 Event Tensor 抽象优化静态计算—通信模式的效果, 我们测试了张量并行 LLM 的两个基础融合内核: GEMM + Reduce-Scatter 与 All-Gather + GEMM. 这些内核对于降低分布式推理延迟, 提高硬件利用率至关重要. 我们使用由多种现代 LLM 派生的 MLP 配置, 所有实验都将张量并行规模固定为 8, token 数固定为 8192. 具体配置见 [第 9 节](#section-9). 我们将 ETC 生成的内核与多个基线比较, 并根据每种工作负载的特性选择实现:

- GEMM + Reduce-Scatter: Reduce-Scatter 集合通信使用 CUDA multimem PTX 指令实现. 我们采用 ETC 的 **动态调度器** 处理网络争用和波动造成的不可预测工作负载; 它能即时适应并平衡任务, 在这里最为有效.
- All-Gather + GEMM: All-Gather 操作使用由复制引擎 (DMA) 实现的环形算法. 我们采用 **静态调度器**, 因为只有 GEMM 块会按照环形算法规定的数据到达顺序在 SM 上执行. 预计算静态调度能以最低运行时开销有效重叠通信与计算.

比较所用基线包括:

- cuBLAS+NCCL: 一个不做重叠的基线, 依次执行 cuBLAS 与 NCCL 内核, 代表不融合时的性能.
- TP-Async [Lia24b]: 一种基于 PyTorch 的方法, 通过人工协调异步操作来实现重叠.
- Triton Distributed v0.0.2-rc [Zhe25i]: 生成重叠内核的编译器系统, 作为当前领先的开源基线.
- cuBLASMp [Cub23]: 一种高性能多进程融合内核库, 可重叠分布式计算与通信.

<span id="figure-11"></span>

![图 11. 使用动态调度器时, GEMM + Reduce-Scatter 在 8 张 B200 上的性能结果.](./event-tensor/figure-11.png)

**图 11.** 使用动态调度器时, GEMM + Reduce-Scatter 在 8 张 B200 上的性能结果.

<span id="figure-12"></span>

![图 12. 使用静态调度器时, All-Gather + GEMM 在 8 张 B200 上的性能结果.](./event-tensor/figure-12.png)

**图 12.** 使用静态调度器时, All-Gather + GEMM 在 8 张 B200 上的性能结果.

[图 11](#figure-11) 和 [图 12](#figure-12) 相较基线都有明显提升, 尤其是在较大的模型配置上; 两个工作负载相较 cuBLAS+NCCL 基线都取得最高 1.40x 的执行时间加速. 在各个融合基线中, TP-Async 的粗粒度切分可能产生太小, 无法使 SM 饱和的块, 也可能产生太大, 无法有效掩盖通信延迟的块; 而 Triton-Dist 对 B200 的支持尚处实验阶段, 其基于 Triton 的 GEMM 还没有针对 Blackwell 架构充分优化. 因此, 未融合的 cuBLAS+NCCL 基线有时也能与这些融合方法竞争, 足见高效融合并不容易. ETC 稳定的性能优势来自 Event Tensor 抽象. 编译器将细粒度依赖表示为一等 Event Tensor, 因而可以把单体操作变换为深度流水化的任务图. 这项抽象也让统一调度变换能够有效应用: 对 GEMM + Reduce-Scatter, 我们使用动态调度器应对通信延迟可能出现的不可预测性; 对 All-Gather + GEMM, 则使用静态调度器, 以最低开销配合可预测的环形算法安排重叠. 这种由编译器驱动的细粒度方法让计算资源 (SM) 和网络资源都能持续保持忙碌, 实现与现有系统相当或更高的重叠程度.

<span id="section-4-2"></span>

### 4.2 混合专家 (MoE) 层性能

为评估 Event Tensor 抽象管理具有形状动态性和数据依赖动态性的任务图的能力, 本小节测试 token 数可变的完整 MoE 层. 现有系统虽然采用了高效的持久化 GroupGEMM 内核, 但执行完整 MoE 层时仍需依次启动多个独立内核. ETC 可以把完整的数据依赖 MoE 数据流融合进单个巨型内核; 我们将其性能与现有优化后的多内核基线比较. 测试采用 Qwen3-30B-A3B 的完整 MoE 层, 该层包含 128 个专家, top-k 为 8. 工作负载负责处理数量可变的输入 token. 这里使用 ETC 的动态调度器, 因为它的自适应负载均衡非常适合这种不规则, 依赖数据的任务图. 比较的基线包括:

- Triton 3.4.0 [Til19]: 一种高度优化的 MoE 实现, 广泛用于 SGLang [She24] 和 vLLM [Kwo23] 等当前领先的服务系统.
- FlashInfer 0.2.14.post1 [Ye25]: 一个提供 LLM 推理优化内核的高性能库, 其中包括融合 MoE 内核.

<span id="figure-13"></span>

![图 13. MoE 层在单张 B200 上的性能结果.](./event-tensor/figure-13.png)

**图 13.** MoE 层在单张 B200 上的性能结果.

[图 13](#figure-13) 绘制了不同 token 数下 MoE 层的相对端到端性能. ETC 的巨型内核方法显著超过最佳基线, 在 1024 个 token 时达到最高 1.23x 加速. 两个基线之间, FlashInfer 的 GroupGEMM 针对较大 token 数做了更多优化, 而 Triton 得益于把 gather/scatter 融合进 GroupGEMM; 因此, 两者的相对排名会随 token 数变化. ETC 的性能增益直接来自 Event Tensor 抽象和动态调度变换. 首先, 依赖数据的 Event Tensor 打破了基线中的全局同步屏障, 在 MoE 的两个 GroupGEMM 阶段之间建立细粒度流水线; 同时, 通过平滑融合算子之间的 SM 分配, 还降低了波次量化效应. 其次, 对 MoE 更为重要的是, 片上动态调度器为不规则 token 路由提供了更好的负载均衡, 尽量减少 SM 空闲时间; 随着 token 数增长, 它始终优于其他方法.

<span id="section-4-3"></span>

### 4.3 端到端低批量服务性能

<span id="figure-14"></span>

![图 14. Qwen3-30B-A3B 与 Qwen3-32B 模型服务的端到端性能 (越低越好).](./event-tensor/figure-14.png)

**图 14.** Qwen3-30B-A3B 与 Qwen3-32B 模型服务的端到端性能 (越低越好).

本小节检验 ETC 编译的巨型内核能否在动态低批量服务场景中取得更低的端到端延迟; 这类场景在实时智能体工作流, 交互式编码助手等延迟敏感应用中日益常见. 我们重点关注解码阶段, 因为预填充与大批量服务通常已有很高的 GPU 利用率, 巨型内核带来的收益有限 (但 [第 4.1 节](#section-4-1) 所示的计算—通信重叠除外). 重要的是, ETC 不会降低大批量性能, 因为各算子的硬件利用率保持不变, 也没有额外开销. 我们为两个有代表性的模型集成 ETC 编译的巨型内核: Qwen3-30B-A3B MoE 模型和 Qwen3-32B 稠密模型; 二者均采用静态调度器以避免运行时开销. 编译后的巨型内核覆盖完整解码流水线 (Attention, RoPE, KV-Cache, Norm, MLP, MoE), 而非只有 GEMM. 我们选择 Qwen 系列 (包含稠密与 MoE 变体), 是因为其架构能够代表现代 LLM (例如 LLaMA 3, GPT). 基准测试使用合成数据集, 预填充长度为 512, 生成 100 个输出 token, 批大小从 1 变化到 128. 我们测量每个输出 token 的时间 (TPOT), 它最能反映 LLM 引擎的解码性能. 一次解码迭代的原始内核运行时间见 [第 10 节](#section-10); 该节只关注内核性能, 排除所有无关开销 (例如框架特定的调度延迟和其他 CPU 开销), 确保不同框架之间比较公平. 我们把 ETC 与领先的服务系统 vLLM (v0.11.0rc2) 和 SGLang (v0.5.3rc0) 比较, 两者均使用 CUDA Graph 和 `torch.compile` [Ans24] 优化性能.

[图 14](#figure-14) 左图给出 Qwen3-30B-A3B 的端到端性能; 批大小为 1 时, ETC 相较 vLLM 加速 1.48x, 相较 SGLang 加速 1.20x. 在 [图 14](#figure-14) 中图的 Qwen3-32B 实验中, ETC 始终具有最低延迟; 批大小为 1 时最多比 vLLM 快 1.15x, 批大小为 64 时最多比 SGLang 快 1.09x. Qwen3-32B 采用四路 TP 张量并行 (TP) 执行 [Sho19] 时 ([图 14](#figure-14), 右), ETC 与 vLLM 性能相当, 加速比在 0.99x 到 1.06x 之间. 此时 ETC 和 vLLM 的延迟都高于 SGLang, 因为 SGLang 高度优化的 CPU 调度器具有更低的分布式运行时开销. 少数情况下, ETC 略逊于最佳基线, 原因在于工程因素——具体来说, 某些配置下编译器生成的 GEMM 块不如 cuBLAS 调优充分, 而且我们的服务引擎 CPU 侧开销更高——并非抽象本身存在根本局限.

ETC 的强劲性能来自 Event Tensor 抽象与静态调度变换所实现的融合巨型内核架构. 传统方法会依次启动多个内核, 并在内核边界隐式同步; ETC 则在单个持久化内核中执行完整工作负载. 这种由编译器驱动的融合支持 CUDA Graph 系统难以实现的细粒度优化: 它可以暴露 attention 内部的并行执行 (例如, 让 Q 的 Norm+RoPE 与 K 的 Norm+RoPE+CacheAppend 并发运行), 把 MoE 中的 GroupGEMM 与 MLP 中的 GEMM 流水化以减少波次量化, 还能在输入激活就绪前预取模型权重, 掩盖内存延迟. ETC 能跨算子边界对操作进行流水化和重叠, 这是它相较基线降低延迟的关键. 更重要的是, ETC 打破了内核边界, 成功地在 MoE 等固有动态工作负载上达到与 CUDA Graphs 相当的性能.

<span id="section-4-4"></span>

### 4.4 预热开销

<span id="table-01"></span>

![表 1. 使用不同计算图捕获方法时, Qwen3-32B 模型服务的预热时间.](./event-tensor/table-01.png)

**表 1.** 使用不同计算图捕获方法时, Qwen3-32B 模型服务的预热时间.

本小节通过测量 LLM 引擎预热开销, 评估 ETC 编译策略对部署的影响. 我们把预热时间定义为从引擎启动到处理首个请求的总挂钟时间, 其中包含引擎初始化, 模型加载, 以及所有 JIT 编译或 CUDA Graph 捕获开销. 我们检验由形状动态性支持实现的 ETC 提前 (AOT) 编译, 能否消除即时 (JIT) 编译与 CUDA Graph 捕获带来的运行时成本. [表 1](#table-01) 显示出显著差异: 在 Qwen3-32B 上, vLLM 需要 123 s, SGLang 需要 583 s 才能完成预热, 而 ETC 只需 35 s 即可初始化. 加速来自 Event Tensor 抽象: 它将形状动态性作为一等能力支持, 从而实现 AOT 编译. 基线必须在运行时捕获许多静态 CUDA Graph 来覆盖不同形状 (例如 vLLM 捕获 67 个), 而 ETC 在线下只需编译一次持久化, 形状通用的巨型内核 (Qwen3-32B 需要 107 s). 运行时直接加载预编译图, 避免了 JIT 和运行时捕获方法固有的重复预热成本.

<span id="section-4-5"></span>

### 4.5 不同调度方法之间的取舍

<span id="table-02"></span>

![表 2. 不同 ETC 调度方法在 MoE 层上相对于未融合巨型内核的性能. 越高越好.](./event-tensor/table-02.png)

**表 2.** 不同 ETC 调度方法在 MoE 层上相对于未融合巨型内核的性能. 越高越好.

<span id="table-03"></span>

![表 3. 不同 ETC 调度方法在 TP=4 的 Qwen-3-32B 上相对于未融合巨型内核的性能. 越高越好.](./event-tensor/table-03.png)

**表 3.** 不同 ETC 调度方法在 TP=4 的 Qwen-3-32B 上相对于未融合巨型内核的性能. 越高越好.

本节分析 ETC 静态与动态调度策略的性能特征和取舍, 并通过与未融合巨型内核基线比较, 量化融合带来的收益. 该基线在不同算子阶段之间使用单个事件来强制执行全局同步屏障, 以此在一次内核启动内模拟串行执行模型. 关键在于, 未融合基线与 ETC 使用完全相同的算子代码, 因此 [表 2](#table-02) 和 [表 3](#table-03) 所报告的加速, 完全来自 ETC 的细粒度 Event Tensor 依赖所释放的内核间并行性——也就是 [第 4.2 节](#section-4-2) 与 [第 4.3 节](#section-4-3) 讨论的同一类收益来源 (减少波次量化, 权重预取, 并行执行和片上负载均衡), 而非更好的算子级实现.

**数据依赖工作负载.** 对 MoE 层等具有数据依赖控制流的工作负载 ([第 4.2 节](#section-4-2)), 动态调度器能够实现负载均衡. 如 [表 2](#table-02) 所示, 除单批量推理外, 动态调度器均优于静态调度器; 批大小为 1024 时, 相较静态调度器的最大加速为 4.0%, 相较未融合基线为 8.1%. 依赖数据的 token 路由会造成工作负载天然不均衡. 僵化的静态调度器可能让某些 SM 积累一队运行时间更长的块, 使它们成为拖后腿的长尾任务, 而其他 SM 则处于空闲状态.

**规则工作负载.** 相反, 对规则的稠密 Transformer 层工作负载 (在 [第 4.3 节](#section-4-3) 中以 TP=4 分析), 静态调度器显然更优 ([表 3](#table-03)). 在分布式环境下, 动态调度器的开销会变得很大, 尤其是尝试把任务推送到远程任务队列时. 相较 ETC 未融合版本, ETC 静态调度还稳定获得 6-8% 的加速, 且这一收益完全来自细粒度流水化.

这些结果也解释了为何 [图 11](#figure-11), [图 12](#figure-12) 和 [图 14](#figure-14) 中的多 GPU 评测似乎呈现不同趋势. [图 11](#figure-11) 与 [图 12](#figure-12) 代表采用大批量 (8192 个 token) 的带宽受限场景, ETC 通过细粒度信号重叠通信与计算, 在这里表现出色. 相比之下, [图 14](#figure-14) 右图代表低批量的延迟敏感场景, 通信开销与 CPU 调度开销都会暴露出来. 因此, 两者需要不同的调度策略: 动态调度能有效处理大批量下的抖动, 静态调度则把延迟敏感低批量任务的开销降至最低.

这些发现清楚地表明, 调度选择取决于工作负载, 也证实了 ETC 框架同时支持两种调度变换的价值.

<span id="section-5"></span>

## 5 相关工作

MLIR [Lat21], XLA [Sab20], TVM [Che18e, Fen22, Lai25a] 和 PyTorch 编译器 [Ans24] 等深度学习编译器, 为优化深度学习模型奠定了基础. 这些系统执行图级优化, 并逐内核运行. CUDA Graph [Gra19] 可以捕获并回放内核序列, 大幅降低启动开销, 但依赖静态输入. 机器学习编译器也在发展垂直融合 [Zhe20, Niu21] 和水平融合 [Jia19b, Li22e], 以优化内核启动开销. Rammer [Ma20] 与 Roller [Zhu22] 通过软件启动分块任务. 以上工作都没有用于追踪与优化细粒度依赖的显式抽象; 引入本文提出的 Event Tensor 后, 它们也能实现巨型内核优化. DynaTune [Zha21m], DietCode [Zhe22c] 和 SparseTIR [Ye23a] 等动态张量编译器在单内核层面处理动态形状或稀疏性; ETC 与它们互补, 可以把这些编译器的算子实现融合成巨型内核, 释放内核间并行性.

SGLang [She24], vLLM [Kwo23], TensorRT-LLM [Ten24a] 和 Orca [Yu22a] 等 LLM 推理系统通过连续批处理, 推测执行等系统优化取得高性能. 我们的 Event Tensor 编译器可以充当这些框架的后端, 提升 GPU 执行效率. 近期工作 [Che25ae, Spe25] 已经开始为 LLM 构建巨型内核. 这些方法只支持单批量稠密模型推理, 而且只关注单一调度策略. 本文提出的 Event Tensor 抽象为形状动态性和数据依赖动态性提供系统化编译器抽象支持, 补足了这些方法. Event Tensor 编译器还同时支持静态调度与动态调度. CuSync [Jan24a] 优化不同 CUDA stream 上独立内核的协同调度, FlashMoE [Aim25] 则为分布式 MoE 提供手工优化内核. ETC 与二者都不同: 它通过系统化编译流程, 把完整子图融合进单个持久化巨型内核, 能够推广到任何一种算子模式之外.

我们的方法与 Cilk [Blu95], Legion [Bau12], Realm [Tre14] 和 OpenMP Tasks [Dag98] 等基于任务的并行编程模型密切相关. 此前大多数方法关注通常由 CPU 协调的粗粒度任务. 我们的方法也与优化单个内核的 Graphene [Hag23] 和 Cypress [Yad25] 有关. Graphene 把线程建模为具有同步能力的张量, 在概念上与本文相关; 但它的目标是单内核优化, 而非支持动态形状和数据依赖的多算子巨型内核融合. 我们借鉴了这些已有洞见, 提出 Event Tensor 抽象, 以紧凑形式表示算子子任务之间的细粒度依赖, 并在 GPU 流式多处理器上同时运行静态与动态调度.

<span id="section-6"></span>

## 6 结论

本文提出 Event Tensor, 一种用于编译动态 GPU 巨型内核, 表达细粒度同步的统一抽象. Event Tensor 将形状动态性和数据依赖动态性都作为一等能力支持. 在此抽象之上, ETC 使用静态与动态调度, 系统地生成高性能持久化内核. ETC 在实现当前领先服务延迟的同时, 大幅降低预热开销. 未来, 我们设想通过更高层 pass, 从标准计算图自动生成 Event Tensor 任务图, 进一步减少人工同步工作. 我们希望这项工作能推动对巨型内核的更多研究, 并展示 ML 编译器的新可能.

## 致谢

感谢所有匿名 MLSys 审稿人和我们的 shepherd 提出的建设性反馈与意见. 本工作部分得到 NVIDIA, Google 和 Amazon 赠款支持. 我们还感谢 NVIDIA 为 DGX B200 提供支持.

<span id="section-7"></span>

## 7 动态调度伪代码

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**算法 2: ETC 中的动态调度变换.**

- **输入:** 包含分块级数据流图 `G` 的模块 `mod`, 其中带有 Event Tensor 依赖.
- **输出:** 更新后的模块, 其中包含融合并动态调度的巨型内核.
- `mod_updated` $\leftarrow$ `mod.Copy()`.
- `fused_kernel` $\leftarrow$ `NewPersistentKernel()`.
- 不是运行时调度器. 只提供 push/pop 函数.
- `scheduler` $\leftarrow$ `GPUScheduler()`.
- `fused_kernel.AddPopLogic(scheduler.f_pop_tasks)`.
- **对所有** `G` 中的 `task_grid`:
  - `fused_kernel.AddDispatchLogic(task_grid)`.
  - `fused_kernel.AddTileLogic(task_grid)`.
  - **对所有** `task_grid.out_edges` 中的 `event`:
    - `fused_kernel.AddCompleteOnLogic(event, scheduler.f_push_tasks)`.
- `mod_updated.Replace(G, fused_kernel)`.
- **返回** `mod_updated`.

</div>

[算法 2](#algorithm-02) 描述了把 Event Tensor 图变换为动态调度巨型内核的编译器 pass. 每当 SM 完成当前任务时, 代码就会插入一次 `scheduler.pop_tasks` 调用; 当一项任务完成, 将关联事件计数器减至 0 并由此解除依赖任务的阻塞时, 则会插入一次 `scheduler.push_tasks` 调用.

<span id="section-8"></span>

## 8 ETC 端到端编译流程

<span id="figure-15"></span>

![图 15. ETC 的端到端编译流水线.](./event-tensor/figure-15.png)

**图 15.** ETC 的端到端编译流水线.

[图 15](#figure-15) 总结了 [第 3.4 节](#section-3-4) 所述的 ETC 端到端编译流程.

<span id="section-9"></span>

## 9 [第 4.1 节](#section-4-1) 使用的 MLP 配置

[表 4](#table-04) 给出了融合通信与计算评测所用的 MLP 配置, 它们来自多种现代 LLM.

<span id="table-04"></span>

![表 4. MLP 模型配置, 其中 S = 序列长度, H = 隐藏维度, I = 中间层大小.](./event-tensor/table-04.png)

**表 4.** MLP 模型配置, 其中 S = 序列长度, H = 隐藏维度, I = 中间层大小.

<span id="section-10"></span>

## 10 端到端 LLM 服务的原始内核时间评测

<span id="figure-16"></span>

![图 16. Qwen-30B-A3B 在单张 B200 上的原始内核相对性能结果.](./event-tensor/figure-16.png)

**图 16.** Qwen-30B-A3B 在单张 B200 上的原始内核相对性能结果.

<span id="figure-17"></span>

![图 17. Qwen-32B 在单张 B200 上的原始内核相对性能结果.](./event-tensor/figure-17.png)

**图 17.** Qwen-32B 在单张 B200 上的原始内核相对性能结果.

<span id="figure-18"></span>

![图 18. Qwen-32B 在四张 B200 上采用张量并行时的原始内核相对性能结果.](./event-tensor/figure-18.png)

**图 18.** Qwen-32B 在四张 B200 上采用张量并行时的原始内核相对性能结果.

[第 4.3 节](#section-4-3) 报告 ETC 与基线在端到端 LLM 服务中的每个输出 token 时间 (TPOT) 指标. TPOT 不仅包含 GPU 内核执行总时间, 也包含 CPU 侧开销, 例如框架特定的请求调度延迟. 为了给出不受无关开销影响, 更直接也更公平的比较, 本节使用与 [第 4.3 节](#section-4-3) 相同的基线与实验设置, 评估端到端 LLM 服务中的原始 GPU 内核执行时间.

[图 16](#figure-16) 展示 ETC 与基线在 Qwen3-30B-A3B 上, 不同批大小下的原始内核端到端相对性能. ETC 始终比基线更快; 最显著的提升出现在批大小为 1 时, 相较 vLLM 加速 1.49x, 相较 SGLang 加速 1.27x. Event Tensor 抽象支持 MoE 中的数据依赖关系, 使 ETC 能在单个内核中执行完整 MoE 模型, 并实现 attention 算子间并行度提升, GroupGEMM 间细粒度流水化和模型权重预取等优化.

[图 17](#figure-17) 和 [图 18](#figure-18) 分别展示 Qwen3-32B 服务在单张 B200 以及四张 B200 张量并行环境下的原始内核性能. 在单 GPU 设置 (TP = 1) 下, ETC 在所有批大小上都稳定优于 vLLM 与 SGLang; 批大小为 1 时, 相较 vLLM 最高加速 1.13x, 平均提升约 7%. 在张量并行场景 (TP = 4) 中, ETC 在所有设置下均保持相当或更好的性能; 相较 vLLM 最高加速 1.08x, 同时随批大小增长维持类似的扩展能力. 这一相当的性能体现出 ETC 巨型内核设计, 以及基于 Event Tensor 抽象的静态调度卓有成效.

<span id="section-11"></span>

## 11 动态调度器运行时优化

我们采用提前推入策略来掩盖调度开销. 调度器不会等到某项任务的依赖 (即生产者任务) 执行完成, 而是在所有生产者任务都已分派到 SM 后, 立即主动把消费者任务推入就绪队列 (不要求任务已经完成); 消费者执行前的额外等待会确保依赖关系得到满足. 这项主动措施可以避免推入操作落在关键路径上. 推入与生产者任务的执行并发进行, 相当于用前序计算掩盖调度成本.

[+1]: 为简化表示, 我们在图和示例代码中使用类似 Numpy einsum 的记法 [Har20b].

[+2]: `indptr` 是压缩稀疏行 (CSR) 格式等稀疏矩阵表示中常用的术语.

[+3]: 动态调度变换的伪代码见 [第 7 节](#section-7).

[+4]: 编译流水线图见 [第 8 节](#section-8).
