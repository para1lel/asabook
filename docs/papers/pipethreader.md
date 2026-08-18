---
title: 'PipeThreader: Software-Defined Pipelining'
createTime: 2026/08/18 15:51:26
permalink: /papers/pipethreader/
---

> [Yu Cheng](https://chengyupku.github.io/) [+internship], [Lei Wang](https://x.com/Lei_Wang_1999), [Yining Shi](https://dblp.org/pid/161/3927-1.html), [Yuqing Xia](https://dblp.org/pid/211/8365.html), [Lingxiao Ma](https://xysmlx.github.io/), [Jilong Xue](https://dblp.org/pid/06/10336.html), [Yang Wang](https://dblp.org/pid/w/YangWang53.html), [Zhiwen Mo](https://hamerlate.github.io/), [Feiyang Chen](https://dblp.org/pid/41/10690.html), [Fan Yang](https://fanyangcs.github.io/), [Mao Yang](https://dblp.org/pid/89/1482-4.html) 和 [Zhi Yang](https://yangzhihome.github.io/). 论文发表于 [19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2025), *PipeThreader: Software-Defined Pipelining for Efficient DNN Execution*](https://www.usenix.org/conference/osdi25/presentation/cheng), 会议时间为 2025 年 7 月 7-9 日, 页码为 767-783. [原始 PDF](/paper/pipethreader.pdf). 本阅读版保留 OSDI 2025 论文的正文, 图, 表和致谢; 精确的印刷版式与参考文献以原始 PDF 为准.

[+internship]: 部分工作在 Microsoft Research 实习期间完成.

## 摘要

为了有效利用现代 GPU 中的 TensorCore 和 Tensor Memory Accelerator 等异构专用硬件单元, 本文提出新的 DNN 编译器 PipeThreader. PipeThreader 主张把调度功能从硬件移到软件, 以较少的人工工作实现更高效, 更复杂的计算流水线. 为此, 它引入新的 DNN 计算抽象 sTask-graph, 一种能够描述专用单元能力的分层硬件抽象, 以及新的调度原语. 因此, PipeThreader 能够为 FlashAttention 等已有大量研究的 DNN 架构找出高效的流水线调度, 性能相当甚至更高. 它还能为 Mamba2 等新模型发现新的流水线方案, 与当前最先进的手工实现相比, 性能显著提高. 代码已开源至 [https://github.com/tile-ai/tilelang](https://github.com/tile-ai/tilelang).

## 1 引言

不断扩大的深度神经网络 (DNN) 给 GPU 等现代 AI 加速器带来了沉重的计算和内存压力. 为满足持续增长的算力需求, 硬件厂商引入了 TensorCore [Amd20, Amd21a, Amd23, Nvi17a, Nvi20, Nvi23] 和 Tensor Memory Accelerator (TMA) [Nvi23] 等异构专用硬件单元. 与此同时, 软件开发者倾向于把多个 DNN 算子和矩阵乘法等计算密集单元融合到同一个 GPU kernel 中, 以最大限度复用数据并缓解内存压力 [Dao22, Shi23a].

然而, 这些硬件与软件趋势也给 DNN 的高效执行带来了新问题. 第一, 要充分利用专用硬件单元, 必须仔细调度 DNN 计算流水线. 过去, GPU 依靠硬件调度器执行线程, 用大量并发线程摊平单条流水线中可能出现的停顿. 如今这种方法不再有效, 因为专用单元需要以更大的张量粒度换取更高的计算密度, 可用并发线程数已大幅减少. 第二, 算子融合会加深计算流水线. 硬件调度器很难理解这种复杂流水线, 因而难以有效调度. 因此, FlashAttention [Dao23b] 等当前最先进的 DNN kernel 都依靠手工编写, 以细致组织执行流水线. 但这种做法很难泛化. 每种 GPU, 如 NVIDIA Hopper GPU 或 AMD GPU [Sha24b], 每种新 DNN 模型, 如 Mamba [Gu23], 甚至 DNN 模型采用的新张量形状, 如 FlashAttention2/3 [Dao23b, Sha24b], 都需要新的手工实现.

面对日益复杂的软件与硬件流水线以及硬件调度固有的局限, 本文提出 PipeThreader, 一种支持软件定义流水线的 DNN 编译器, 用于在配备异构专用硬件单元的现代 GPU 架构上高效执行 DNN. PipeThreader 把 DNN 计算抽象为 sTask-graph. 图中每个节点表示一个 sTask, 即可调度到专用单元上的细粒度任务; 有向边则表示 sTask 之间的依赖关系. sTask 在 tile 上计算, tile 是张量的一部分. 现代 DNN 编译器已广泛采用 tiling 概念 [Fen23, Nvi24a, Ma20, Til19, Zha19h, Zhe20, Zhu22]. 借助 sTask-graph, PipeThreader 可以提取适合在专用硬件单元上执行的复杂计算流水线.

为了公开硬件的细粒度调度能力, PipeThreader 把 GPU 设备抽象为分层硬件单元, 包括虚拟化的同构并行执行单元 (EU), 以及每个 EU 内的异构专用单元 (sEU). 随后可用 `append`, `wait` 和 `propagate` 三种原语调度 sTask, 从而通过软件实现复杂的流水线. 面对 sTask-graph 定义的新优化空间和新公开的硬件能力, PipeThreader 还使用两层调度策略寻找有效的调度方案.

我们基于开源 DNN 编译器 TVM [Che18] 和 Ladder [Wan24e], 用 8.5k 行 C++ 与 Python 代码实现了 PipeThreader. 评估结果表明, PipeThreader 能够在 NVIDIA H100 或 AMD MI300X GPU 等新硬件上发现类似 FlashAttention 的流水线方案, 无需手工实现即可达到相当或更高的性能. PipeThreader 也能为 Mamba2 [Dao24] 等新模型找到有效的流水线调度, 性能显著优于当前最先进的手工实现. 我们认为, PipeThreader 进一步推进了完整 tile 软件/硬件抽象范围的定义, 与现代 tile 编程工作 (如 Triton [Til19], CUTLASS [Nvi24a] 中的 CuTe 抽象等) 以及新硬件架构的发展相契合.

## 2 动机

**日益复杂的硬件与软件.** 大型 DNN 模型, 特别是大语言模型 (LLM) [Ope22] 的快速增长, 促使硬件厂商开发 TensorCore 和 Tensor Memory Accelerator (TMA) 等异构专用硬件单元, 以满足不断攀升的计算需求. 与此同时, FlashAttention [Dao22] 等复杂的算子融合技术越来越多地用于降低内存开销并最大化数据局部性 (详见[第 3.3 节](#_3-3-running-examples)). 这些趋势提高了计算密度与效率, 也给调度和执行带来很大困难, 在配备异构单元的现代 GPU 上尤其如此.

**传统数据并行 GPU 执行中的利用率不足.** CUDA [Cud25] 等传统 GPU 编程模型把 thread block 分派到各个 Streaming Multiprocessor (SM), 并将每个 SM 视作统一, 独立的执行单元. 这种抽象假设所有 SM 都可互换, 并隐藏其内部结构, 对 NVIDIA V100 [Nvi17a] 等早期架构十分有效. 然而, NVIDIA H100 [Nvi23] 等现代 GPU 在每个 SM 内集成了异构组件, 包括用于矩阵运算的 TensorCore, 用于通用计算的 CUDA core, 以及用于内存移动的 TMA. 这些组件的职责和执行特性各不相同. 如果不了解这些内部差异而统一分派 thread block, 资源利用就会低下. 要充分发挥此类架构的能力, 必须了解每个 SM 内的异构单元, 并据此协调任务放置, 调度和流水线. 缺少这种控制时, 大量性能无法释放.

[图 1](#figure-01) 给出了 H100 上不同 MatMul, FlashAttention 和 Mamba2 实现的各单元利用率. 对于 MatMul, 在没有流水线执行时, 内存移动成为瓶颈, TensorCore 利用率仅为 40%, 而专家优化的 cuBLAS 可达到 97%. 与基于 Triton 的 FlashAttention2 [Dao23b] 相比, FlashAttention-3 [Sha24b] 通过手工优化把 TensorCore 利用率从 40% 提高到 72%. FlashAttention 从第 2 版演进到第 3 版花了将近一年. 然而, 新近手工优化的 Mamba2 [Dao24] 仍未充分利用这些单元, TensorCore 利用率只有 15%. 因此, 面向新 DNN 模型充分利用现代硬件并不容易.

<span id="figure-01"></span>

![NVIDIA H100 上 MatMul, FlashAttention 和 Mamba2 ChunkScan 的各单元利用率](./pipethreader/figure-01.png)

**图 1.** NVIDIA H100 上代表性 AI 工作负载不同实现的各单元硬件利用率, 包括 MatMul, FlashAttention 和 Mamba2 的 ChunkScan. 每根柱表示某个工作负载实现中各硬件单元的利用率. 注意, MatMul 不使用 FMA 和 XU 单元.

由于设计空间庞大且对硬件敏感, 手工管理 kernel 中的流水线执行极其困难. 开发者必须在严格的片上资源限制下, 仔细权衡 tile 大小与流水线深度. 不同架构的内存层次和专用计算单元又不尽相同, 进一步加大了难度. 人工推理很快会变得不可处理, 因而必须借助自动推断和调度, 才能兼顾性能与可移植性. 然而, TVM [Che18] 和 Triton [Til19] 等现有编译器缺少用于表达 tile 流水线执行的明确机制. 它们抽象掉底层控制, 限制了开发者指定执行顺序, 资源分配和计算-通信重叠的能力, 因而无法发挥全部性能潜力. 要实现高效的 tile 流水线执行, 需要一种新编译器, 能够系统搜索这一设计空间, 生成优化后的调度, 并适配不同硬件平台.

**观察与机会.** 根据这些事实, 我们看到了解决该问题的独特机会. 新硬件单元以 tensor tile 等大粒度处理数据, 而已有工作 [Fen23, Nvi24a, Ma20, Til19, Zha19h, Zhe20, Zhu22] 表明, tile 级性能具有确定性, 因此可以在软件层高效调度 tile 级执行. 利用这一趋势, 我们主张把流水线调度从隐式硬件行为转为显式软件控制. 这里的流水线调度并非 GPU 对底层线程或 warp 的分派, 而是由软件引导, 把 tile 级操作映射到每个 SM 内的 TensorCore, CUDA core 或 TMA 等专用单元.

[图 2a](#figure-02) 展示了融合 MatMul-Sum 的执行过程, 其中 MatMul 在 TensorCore 上执行, Sum 在 CUDA core 上执行. 现有方法采用同构抽象, 即使 TensorCore 与 CUDA core 天然可以并行, 调度仍会把二者串行执行, 造成低效. 与之相对, [图 2b](#figure-02) 给出了利用专用执行单元的优化调度, 它可以流水线执行并充分利用异构硬件.

<span id="figure-02"></span>

![融合 MatMul-Sum 在专用执行单元上的低效调度与流水线调度](./pipethreader/figure-02.png)

**图 2.** (a) 现有方法中的低效调度; (b) 利用专用执行单元进行流水线执行的优化调度.

遗憾的是, 现有 DNN 模型表示和 GPU 硬件接口都没有明确公开 tile 级流水线执行所需的调度能力.

## 3 PipeThreader 抽象

[第 2 节](#_2-motivation)的观察促使我们提出 PipeThreader, 一个把基于 tile 的数据并行与流水线调度结合起来的 DNN 编译器框架. [图 3](#figure-03) 给出了系统概览. 当前最先进的 DNN 编译器 (如 Triton [Til19], Roller [Zhu22], Welder [Shi23a]) 把硬件加速器抽象为同构执行单元 (EU) 的集合, 用于 SPMD 式并行. 这种方法忽略了现代 GPU 固有的硬件异构性. 例如, 一个 SM 内的 TensorCore 和 CUDA core 分别针对不同工作负载优化, 现有编译器无法利用这种差异. 为弥补这些不足, PipeThreader 引入两种核心抽象: 专用任务 (sTask) 和专用执行单元 (sEU). PipeThreader 以算子的数据流图 (DFG) 为输入, 将算子转换为 sTask, 以利用 sEU 的异构能力并实现 MPMD 式并行. [第 3.1 节](#_3-1-specialized-tasks-and-execution-units)将详述 sTask 与 sEU. sTask 在任务粒度上保留原始 DFG 的数据依赖, 形成 sTask-graph. PipeThreader 以 sTask-program 这种结构化执行表示, 组织 sTask-graph 到 sEU 的映射. sTask-program 抽象为 PipeThreader 打开了新的搜索空间. [第 3.2 节](#_3-2-from-stask-graph-to-sprogram)将详述 sTask-graph, sTask-program 和搜索空间.

<span id="figure-03"></span>

![从 DFG 划分到 sProgram profiling 的 PipeThreader 系统概览](./pipethreader/figure-03.png)

**图 3.** PipeThreader 系统概览.

<span id="figure-04"></span>

![PipeThreader 的 sTask 与 sEU 抽象类结构](./pipethreader/figure-04.png)

**图 4.** sTask 和 sEU 抽象.

### 3.1 专用任务与执行单元

**sTask.** PipeThreader 引入 sTask (specialized task 的缩写), 作为算子中在加速设备的特定执行单元 (sEU) 上运行的基本计算单元. sTask 概念自然对应现代 DNN 加速器中的异构专用处理器, 如 H100 GPU 中的 TMA, CUDA core 和 TensorCore. 为尽可能提高效率, 必须针对每类专用处理器, 把加速器上的计算拆成多个并行的异构任务. 每类并行任务都可表示为一种 sTask, 从而同时向底层硬件的专用处理器和 PipeThreader 编译器公开潜在的任务并行性.

如[图 4](#figure-04) 所示, sTask 处理从输入张量切出的 data tile, 并在输出张量中生成 data tile, 其计算由基于索引的张量表达式描述. sTask 的 shape (第 8 行) 沿张量表达式 `expr` (第 7 行) 的每个循环轴定义. `target_sEU` 属性 (第 9 行) 还指定 sTask 可以在哪类专用单元上执行. 传统的 tile 任务没有明确分类, 因而无法充分利用专用处理器的并行性. 如[图 5](#figure-05) 所示, 在特定 EU 上, 传统 MatMul-Sum 任务 (FlashAttention 中的融合操作) 会依次从 $A$ 计算一个 $[2 \times 2]$ data tile, 从 $B$ 计算一个 $[2 \times 2 \times 2]$ data tile, 并为 $C$ 生成一个 $[2 \times 2]$ 输出 tile. PipeThreader 引入两类 sTask: 在 TensorCore 上执行矩阵乘加的 mma sTask, 以及在 CUDA core 上运行的 Sum sTask.

<span id="figure-05"></span>

![NVIDIA TensorCore 和 CUDA core 上采用流水线的 MatMul-Sum sTask](./pipethreader/figure-05.png)

**图 5.** NVIDIA GPU 上的 sTask MatMul-Sum.

这使流水线成为可能: 第二个 mma sTask 在 TensorCore 上将 $A$ 与分块 $B_1$ 相乘以生成 $C_1$, 可以和第一个 Sum sTask 在 CUDA core 上把 $C_0$ 归约为 $S_0$ 的过程重叠. 这种流水线执行支持任务并行, 能显著提高硬件利用率.

**sEU.** 现代加速器缺少把 sTask 映射到特定执行单元的接口. 为此, PipeThreader 明确公开 GPU 内的执行单元, 并将其抽象为分层执行数组, 同时描述并行性和通过流水线支持数据依赖执行的能力.

如[图 4](#figure-04) 所示, 抽象后的设备由多个并行执行单元 (EU) 组成 (第 11 行), 每个 EU 包含若干异构专用执行单元 (sEU) (第 12 行). 这些 sEU 是 PipeThreader 通过流水线有效调度数据依赖任务的硬件基础. 例如, 在现代 H100 GPU 上, Streaming Multiprocessor (SM) 是一种 EU, 其中包括用于 load sTask 的 Tensor Memory Accelerator (TMA) 和用于 mma sTask 的 TensorCore. sEU 使用 `Execute` 接口 (第 15 行) 执行给定的 sTask. `is_async` 属性 (第 14 行) 指定 sEU 采用同步方式 (如 CUDA core) 还是异步方式 (如 TMA) 运行. 异步 sEU 可以和异步或同步 sEU 并发执行.

### 3.2 从 sTask-Graph 到 sProgram

为了执行 DNN, PipeThreader 把输入 DFG 转换为针对现代异构硬件的专用表示. 该过程包含两个主要步骤: 构建描述计算和依赖的 sTask-graph, 再将该图映射到专用执行单元 (sEU), 形成协调高效执行的 sTask-program (sProgram).

**sTask-Graph.** 如[图 3](#figure-03) 所示, 输入 DFG 中的算子经由 sTask-partition 转换为 sTask, 形成 sTask-graph. 该图保留原始 DFG 的计算和数据依赖, 节点表示 sTask, 边则描述任务级细粒度依赖. sTask-partition 过程通过配置 sTask 的 `TileShape` (即 `Map<Axis, Dim>`) 来划分每个算子, 指定可划分的维度和大小. 传统编译器主要关注用于实现数据并行的空间划分. PipeThreader 将其扩展为同时支持空间划分与归约划分, 为流水线执行提供新的机会. 凭借这种灵活性, PipeThreader 可以根据不同划分策略生成多种 sTask-graph, 实现更灵活, 更高效的执行规划.

**sProgram.** 给定 sTask-graph 后, PipeThreader 以 sTask-program (sProgram) 的形式把它映射到硬件的 sEU. sProgram 是一个二维数组 `sProg[sEU][order]`, 每个条目指定 sTask 分配到哪个 sEU 及其执行顺序. 这种结构化表示有助于高效调度和执行任务. 为了让存在依赖的 sTask 按正确顺序执行, PipeThreader 引入 barrier-sTask, 它引用程序中由 `<EU_id, sEU_id, order>` 标识的一组 sTask 来同步执行. barrier-sTask 会等待所有被引用的 sTask 完成后再继续.

**搜索空间.** PipeThreader 的搜索空间由一组 sProgram 构成, 每个 sProgram 都是二维数组 `sProg[sEU][order]`, 用来定义图中各 sTask 的 tiling 大小和执行顺序 (包括同步 barrier). sTask 顺序和 tiling 大小可能存在很多组合. PipeThreader 的搜索空间包含所有既能执行算子又遵守数据依赖的有效 sProgram. 例如, FlashAttention 的搜索空间包含 37,440 个有效 sProgram. 对于复杂的融合算子, sTask 调度往往在搜索空间中占据更大比重. sTask 类型越多, 采用不同 sTask 执行顺序的有效 sProgram 就可能越多. 例如, FlashAttention 有 36 种 sTask 大小 (即 tiling) 配置, 但每种 tiling 大小对应 1,040 种 sTask 顺序配置.

### 3.3 示例

**Mamba2.** Mamba 是一种常用 DNN 模型, 使用带线性注意力机制的分块扫描处理序列. 它的线性注意力由多个模块组成; 这里以其中关键的 ChunkScan 算子为例.

**前端.** 对于 ChunkScan 函数, PipeThreader 以[图 6a](#figure-06) 中的简单 IR 为输入. 它先让 `cb` 乘以 `dA` 的指数与 `dt` 的乘积 (第 6 行), 再把 `cb` 与 `x` 的矩阵乘法结果累加到 `acc_o` (第 8 行). PipeThreader 把 `load_cb` (第 3 行), `load_dA` (第 4 行), `load_dt` (第 5 行), `exp` (第 6 行), `load_x` (第 7 行) 和 `mma` (第 8 行) 分别视为独立 sTask.

**sTask-graph.** PipeThreader 根据这些任务的依赖关系构建相应的 sTask-graph. sTask-graph 可以沿空间维度和归约维度划分. 空间划分 (即 batch size) 把图拆成较小子图, 分布到各 EU 上进行 tile 数据并行. 归约划分 (即序列长度) 在 EU 内生成粒度更细的 sTask, 提供流水线执行机会. 给定大小为 $(M, N)$ 的 `acc_o` ([图 6a](#figure-06) 第 8 行) 和大小为 $(K, N)$ 的 $X$, PipeThreader 像往常一样划分空间维度 $(M, N)$, 为每个 sTask 分配大小为 $(m, n)$ 的 tile. 它还把归约维度 $(K)$ 划分为 `loop_range` 次迭代, 使计算能够跨迭代重叠 [+window]. [图 6b](#figure-06) 展示了通过归约维度划分得到的 sTask-graph.

**sProgram.** 给定 sTask-graph 后, PipeThreader 可用多个 sProgram 表示将 sTask 映射到 sEU 的不同选择. [图 6c](#figure-06) 展示了从 sTask-graph 得到的三个 sProgram. 在 sProg-A 中, `load_x` 排在其他 load sTask 之前; sProg-B 与 sProg-C 则采用相反顺序. 与 sProg-B 相比, sProg-C 使用更大的 tiling 尺寸.

<span id="figure-06"></span>

![Mamba2 ChunkScan 前端, sTask-graph, sProgram 搜索空间与评估](./pipethreader/figure-06.png)

**图 6.** Mamba2-ChunkScan 示例. (a) 展示面向用户的前端. (b) 给出从 (a) 构建的 sTask-graph, 不同颜色表示不同迭代. (c) 展示从 (b) 中 sTask-graph 得到的多个 sProgram, (d) 比较其评估结果, 找出效率最高的 sProgram.

[+window]: 注意, sTask-graph 并不会完全展开所有循环, 而是通过调度一个包含 MAX_STREAM 次迭代的窗口来表示迭代结构.

**评估.** [图 6d](#figure-06) 展示了三个 sProgram 的评估. 表中列出了各时间步运行的 sTask 及相应的片上内存用量. 这里假设片上内存容量为 1 KiB. 在 sProg-A 中, `load_x` 调度得更早, 但 `exp` sTask 依赖 `load_cb_dA_dt` 完成, 因而 `exp` 的调度被延后, 总体效率低于 sProg-B. sProg-C 使用更大的 sTask, 也增加了 workspace 的片上内存用量. 可以看到, 在时间步 $t_4$ 时, workspace 超出可用片上内存容量, 使该 sProgram 无效. 因此, 最终选择 sProg-B 作为调度策略.

PipeThreader 的任务划分遵循 tiling 原则, 但把新的归约 tiling 提升为主要优化策略. 传统 tiling 优先通过空间划分实现数据复用, 归约 tiling 通常处于次要地位. PipeThreader 则主动利用归约 tiling 实现流水线, 提高执行效率. 这样既能保持数据复用, 又能高效地执行流水线. PipeThreader 把归约 tiling 作为一等优化, 带来了超越传统 tiling 策略的性能提升, 对大量使用流水线的工作负载尤其有效. 传统 tiling 策略还倾向于采用更大的 tile, 二者都会占用片上内存, 因而可能与提高流水线并行度发生冲突. PipeThreader 会进一步在 tiling 与流水线之间取舍.

**FlashAttention-3.** FlashAttention 是原始完整注意力机制的一种高效实现, 其 DNN 算子图的输入节点是三个张量: $Q$, $K$ 和 $V$. 首先执行矩阵乘法 MatMulQK, 计算 `acc_s` $= QK^\top$. 接着对 `acc_s` 执行 Softmax, 得到 $P$. 最后把 $P$ 和 $V$ 作为第二次矩阵乘法 MatMulPV 的输入, 计算输出 $O$.

<span id="figure-07"></span>

![FlashAttention sTask-graph 的伪代码](./pipethreader/figure-07.png)

**图 7.** FlashAttention sTask-graph 的伪代码.

在 FlashAttention 中, 这三个算子融合为一个 kernel. PipeThreader 标注该模式, 推导出存在依赖的 sTask, 形成 sTask-graph ([图 7](#figure-07)). 划分后, PipeThreader 把 `load_k` (第 4 行) 和 `load_v` (第 7 行) 分配给 TMA, 把 `mma_qk` (第 5 行) 和 `mma_pv` (第 9 行) 分配给 TensorCore, 再把 `softmax` (第 6 行) 和 `rescale` (第 8 行) 分配给 CUDA core. PipeThreader 使用两层调度策略搜索该空间, 生成的 sProgram 让 sTask 在各自的 sEU 上并行执行, 利用 sEU 的异步性与异构性. 我们的调度空间也包括最新 FlashAttention-3 [Sha24b] 的流水线方案.

## 4 PipeThreader 调度

sProgram 抽象打开了一个庞大的优化空间. PipeThreader 需要在该空间内生成高质量 sProgram. 为此, PipeThreader 将调度机制与策略分开. 在机制方面, 它提供两项能力: (1) 供策略生成 sProgram 的调度接口. (2) 提供调度策略所需 profiling 信息的 profiler. 在策略方面, PipeThreader 提供一种平衡 tiling 与流水线并行的两层策略. 这一简单策略已经能够超越当前最先进的方法, 有时优势还很明显. 我们认为, 该机制为今后研究更复杂的策略打下了基础, 使其可以继续利用 sProgram 公开的优化空间.

### 4.1 调度接口

如[图 8](#figure-08) 所示, PipeThreader 提供三个接口, 用于在新空间中生成高质量 sProgram. `Append` 接口把特定 sTask 分配给某个 EU 内的指定 sEU. `Wait` 接口让 sTask $s$ 等待 `list<sTask_uid>` 中的 sTask 完成, 相当于在 sTask $s$ 前隐式附加一个 barrier-sTask. 上述接口可以显式控制 sTask 在各 sEU 上的放置与执行顺序 (即 sProgram), 从而搜索并行化空间.

<span id="figure-08"></span>

![Append, Wait 和 Propagate 调度接口](./pipethreader/figure-08.png)

**图 8.** 调度接口.

PipeThreader 还提供 `Propagate` 接口, 它自动推断 sTask-graph 中每个 sTask 的 `TileShape`, 以搜索 sTask 划分空间. `Propagate` 从最后一个 sTask 的输出 tile shape 开始, 沿图反向进行一连串 shape 推断, 根据各 sTask 的张量表达式和输出 tile shape 确定它所依赖的输入区域. 例如, 如果 Softmax sTask 需要 $[4 \times 128]$ 的输出 tile shape, `Propagate` 会推导出其输入 tile shape 也必须是 $[4 \times 128]$. 再把它视为前一个 mma sTask 的输出 tile shape, 即可推断出输入 tile 分别为 $[4 \times k]$ 和 $[k \times 128]$, 其中 $k$ 是归约大小.

### 4.2 调度策略

我们的调度策略受到[图 3](#figure-03) 所述两层硬件抽象的启发: 同构 EU 支持 SPMD 式并行, 每个 EU 内的异构 sEU 则支持 MPMD 式并行. [图 9](#figure-09) 概述了 PipeThreader 采用的两层调度算法. 在 EU 间层, 该策略把模型划分为 sTask-subgraph, 并将其均匀分布到各 EU, 以尽量降低延迟 (第 1-10 行). 在 EU 内层, 该策略构建高效的流水线方案, 优化每个 sTask-subgraph 在给定 EU 上的执行代价 (第 11-32 行). EU 内调度提供各分块的执行代价估计, EU 间调度以此为依据.

策略首先根据每个算子的计算阶段, 把它表示为一个或多个 sTask (例如, MatMul 拆成 load 和 mma sTask). 在 EU 间 pass 中, 调度器通过 `GetsTaskPartitions` 函数 (第 2 行) 枚举输出 sTask 的不同划分. 对于每种 sTask 划分, 调度器使用 `Propagate` (第 4 行) 推导图中其他 sTask 的划分, 再利用 EU 等价的计算能力, 把 sTask 均匀分配到各 EU. 这种 SPMD 式方法大幅降低了 EU 间并行方案的复杂度. 策略调用 EU 内 pass, 优化一个 EU 内已分配 sTask (即 sTask-subgraph) 的执行. 在 EU 内 pass 中, 策略采用贪心方法把 sTask 调度到 sEU, 重复执行以下步骤, 直到分配给该 EU 的所有 sTask 均已调度: 1) 在 `get_complete_sTask` 中选择 `endtime` 早于当前时间 `cur_time` 的 sTask $t$ (第 15 行); 2) 找出前驱均已调度的 ready sTask 集合 (第 17-20 行), 再用 `get_high_priority` (第 22 行) 取出优先级最高的 sTask $u$; 3) 用 `Append()` (第 23 行) 把选中的 sTask 附加到 sEU, 并调用 `Wait()` (第 25 行) 保证 sTask 级依赖. 如果调度 $u$ 必须等待 $t$ 完成以释放内存, 我们还会调用 `Wait(u, t)` (第 26-27 行). 为提高流水线效率, 我们优先调度对已调度任务依赖最少, 且最有可能解锁下游 sTask 的异步 sTask. 如[图 6c](#figure-06) 所示, 过早调度 `load_x` 会推迟 `exp` 的执行, 而先调度 `load_y` 可以让 `exp` 更早开始. 因此, 我们的算法为 `load_y` 分配更高优先级, 自然更倾向于构造 sProg-B 和 sProg-C, 而不是 sProg-A.

<span id="figure-09"></span>

![PipeThreader 的 EU 间与 EU 内两层调度算法](./pipethreader/figure-09.png)

**图 9.** 调度算法.

增加 sTask 重叠 (即流水线并行度) 需要额外的片上内存 (如 GPU 上的 shared memory 和 register) 缓冲各阶段之间的中间结果. 但这一需求可能和较大的 tiling 尺寸冲突. 我们的方法通过 profiling 反馈引导的联合搜索策略, 平衡这些相互竞争的需求. 为保证内存可行, 我们调用 `check_valid`, 根据当前 sProgram 和 profiler 检查选中的 sTask 是否符合目标 sEU 的内存限制 (第 24 行). 超出限制的候选项会被跳过. 例如, [图 6](#figure-06) 中的 sProg-C 使用较大的 tiling 尺寸, 超出了可用片上内存. `check_valid` 步骤会检测到该问题, 避免生成这种无效调度.

**Profiler.** PipeThreader 引入 profiler, 用于指导在搜索空间中生成高效 sProgram ([图 9](#figure-09)). profiler 提供各 sTask 的以下信息, 用来生成有效的 sProgram 执行时间线: (1) 各 sTask 在特定 sEU 上的执行时间, (2) sTask 的资源用量, 包括 local memory 与 register 消耗, (3) sProgram 的总体执行时间. profiler 借助 TVM 等现有编译器的代码生成后端自动处理新的张量表达式, 并测量单个 sTask 的设备代码执行时间与资源用量. 在调度期间, PipeThreader 使用这些 profiling 结果估计任务启动时间, 以保持流水线效率并尽量减少空闲时间. sTask 调度完成后, profiler 还会测量整个生成调度的性能, 给出真实延迟. 这些 profiling 数据为调度策略提供依据, 指导高效调度方案的生成.

## 5 PipeThreader 实现

PipeThreader 基于开源 DNN 编译器 TVM [Che18] 和 Ladder [Wan24e] 实现, 包含 8.5k 行 C++ 与 Python 代码. [图 10](#figure-10) 总结了 PipeThreader 的整体工作流. 前端生成 sTask-graph, 随后由能感知 sTask 的编译器 (调度器) 处理, 生成 sProgram. 最后, mapping optimizer 为 sProgram 生成设备代码.

<span id="figure-10"></span>

![PipeThreader 前端, 调度器, 映射与设备代码工作流](./pipethreader/figure-10.png)

**图 10.** PipeThreader 的实现.

### 5.1 前端

PipeThreader 前端包括用于表示 sTask 级 DNN 计算的 sTask-IR, 以及把 DNN 模型转换为 sTask-graph 的 sTask-converter.

**sTask IR.** sTask 中间表示 (IR) 为程序员和编译器提供了一种灵活方法, 用来表示现有编译器 IR (如面向表达式的 IR) 难以描述的 sTask 级计算. [图 6a](#figure-06) 和[图 7](#figure-07) 中的伪代码可视作 sTask IR 的简化形式. 这些伪代码说明了如何把复杂的深度学习 kernel 建模为 data-flow pattern, 包括内存操作 (如在 DRAM 与 SRAM 之间移动 sTask) 和对 sTask 进行的一系列计算.

**带 sEU 的 sTask converter.** PipeThreader 前端还可以把 sTask IR 或 ONNX graph 表示的 DNN 模型转换为 sTask-graph. 在此过程中, 我们使用用于算子融合的先进 DNN 编译器 Ladder [Wan24e]. Ladder 输出 tile-graph, 以 TVM 的 TIR 作为中间表示. 我们标注每个 tile 任务的 `target_sEU` 属性, 根据 sEU 信息将其转换为 sTask. 例如, 在 NVIDIA H100 GPU 上, 我们把 Streaming Multiprocessor (SM) 视为 EU. 每个 SM 内的 sEU 包括用于矩阵乘加 `mma` 的 TensorCore, 用于 `reduce` 和 `parallel` 等通用浮点计算的 CUDA core, 以及在 global memory 与 shared memory 之间批量复制数据的 TMA. 这些基本操作可以组合起来, 表示大多数常见深度学习 kernel 中的数据操作. 例如, 数据移动会转换为 `parallel` 算子, 它可以表示任意逐元素 tile 操作. 用户也可以定义自定义函数来描述其他 sTask.

程序员虽然需要编写简单 IR 来生成 kernel (例如[图 7](#figure-07) 中的 FlashAttention kernel), 但无需了解任务, 如何把 graph (或 IR) 拆成 sTask, 以及哪个 sTask 可以在哪个 sEU 上运行. PipeThreader 可以推断这些信息, 例如设置[图 4](#figure-04) 中 sTask 类的 tiling shape 和目标 sEU 等属性. sTask converter 会标注每个操作 (如 `mma`) 应在哪类 sEU 上执行 (即 `target_sEU` 属性). 调度器随后自动确定如何将其划分为 sTask (如 tiling shape), 以及哪个 sTask 在哪个 sEU 上运行 (如 sProgram 中的 sEU 分配). 因此, PipeThreader 可以减少手工实现所需的繁琐工作和深厚领域知识. 对于 FlashAttention kernel, PipeThreader 只需要 68 行 Python 代码, 而手工实现的 FlashAttention-3 包含 840 行 CUDA kernel 代码.

### 5.2 NVIDIA CUDA GPU 上的 sTask 映射

TMA 和 TensorCore sEU 的 `is_async` 属性都设为 true, 因为我们可以利用 `cp.async.bulk` 和 `wgmma.mma_async` 指令. CUDA core 不支持异步指令, 因此其 `is_async` 属性设为 false. NVIDIA 已正式确认 CUDA core 与 TensorCore 的指令可以同时分派 [Nvi23a]. 两种单元使用同一组 register, 因而可能相互干扰. 我们在 register 上实现 double buffering, 让 TensorCore 与 CUDA core 的执行重叠, 从而减轻潜在干扰. barrier-sTask 使用 PTX [Nvi25] 中的 `mbarrier` 对象实现. 为了高效实现 sEU 中的 `Execute` 函数, 我们通过 layout inference 决定如何把 sTask 操作和数据映射到不同线程与物理内存. 我们还使用硬件专用指令 (hardware intrinsic) 加速操作.

PipeThreader 无需大量工程工作即可支持同一厂商的不同 GPU 型号. 面向 A100 [Nvi20], H100 [Nvi23] 或 B100 [Nvi25b] 等不同架构时, 只需少量更新 sEU layout, intrinsic 和资源限制等硬件专用配置. 核心编译与调度逻辑可以完全复用.

<span id="figure-11"></span>

![相连 mma 与 sum sTask 的简化 layout inference](./pipethreader/figure-11.png)

**图 11.** mma-sum sTask-subgraph 的简化 layout inference 示例.

**Layout inference.** 要在专用执行单元 (sEU) 上高效执行 sTask, 必须遵守特定的 layout 与 thread-binding 约束. 为此, PipeThreader 引入 Layout 对象, 用于描述 sTask 的 data layout 与 thread-binding. Layout 定义映射函数和迭代域, 指定逻辑数据元素如何转换到物理内存, 以及在需要时如何分配给线程.

PipeThreader 完全自动执行 layout inference, 无需手工指定. [图 11a](#figure-11) 展示了简化版 layout inference. sTask mma 分配给采用严格 layout 约束的 sEU TensorCore. 根据 layout 约束, 可以推导出相应的 layout 映射函数. 这里, $\{T(m), n\}$ 表示映射到线程 $m$ 第 $n$ 个位置的数据元素.

在 sTask-graph 中, 相连 sTask 的 layout 必须对齐, 才能彼此兼容. PipeThreader 根据特定 sEU 的 layout 要求推断 sTask 的 Layout, 并把这些要求传播到整个图. layout 冲突通过基于优先级的推断算法解决, mma 等高优先级 sTask 决定其依赖 sTask 的 layout. 例如, [图 11b](#figure-11) 中的 mma 和 sum sTask 相连. mma sTask 的 layout 已经确定, 因而可以据此推断 sum sTask 的 layout. 在这个例子中, 张量 `C_sum` 需要复制.

**Hardware intrinsic.** 对于需要在 sEU 上执行批量操作的 sTask, 我们将其 lower 为 tile 级函数模板. 例如, 矩阵乘加操作使用 CUTLASS/CuTe 模板 lower, 这些模板集成硬件专用的 TensorCore intrinsic. register allocation 等进一步的指令级优化交给 LLVM [Lat04] 等底层编译器. 对于 NVIDIA H100, 我们使用 Warp Specialization [Bau14, Cra24] 优化执行. 该技术把线程分成 producer warp 和 consumer warp, 每个 warp 负责不同的流水线阶段. producer warp 可以释放未使用的 register, 供 consumer warp 复用, 因而 Warp Specialization 改善了 register allocation 和效率. 根据 H100 中 TMA 单元的特点, 我们把在 global memory 与 shared memory 之间复制数据的 load sTask 分配给 producer warp. mma 和 Softmax 等其余 sTask 由 consumer warp 处理. producer 与 consumer 之间通过用 `mbarrier` 实现的 barrier-sTask 同步, 以保证正确的数据依赖.

### 5.3 AMD ROCm GPU 上的 sTask 映射

我们还在 AMD 最新的高性能 GPU MI300X [Amd23] 上实现了 PipeThreader. MI300X GPU 的并行执行单元称为 compute unit (CU), 类似 NVIDIA 的 SM. 每个 CU 包含多个 sEU, 包括用于矩阵乘加的 MatrixCore, Arithmetic Logic Unit (ALU) 和异步复制单元. 与 CUDA GPU 类似, PipeThreader 在 ROCm GPU 上执行 layout inference, 把 sTask 数据映射到物理地址和线程. 我们还明确使用 `lgkmcnt` 和 `s_waitcnt` 指令管理异步 barrier, 从而精确控制指令依赖和内存操作同步.

## 6 评估

本节在 DNN microbenchmark 和端到端模型上评估 PipeThreader, 与当前最先进的 DNN 编译器, 框架和库比较, 以验证 PipeThreader 的效果. 我们先总结主要发现: (1) PipeThreader 可以为成熟的 DNN 架构 (如 FlashAttention) 找出高效调度, 达到或超越当前最先进的性能; (2) PipeThreader 可以为新模型 (如 Mamba2) 发现新的调度, 显著提高性能; (3) PipeThreader 的抽象与设计可以适配 NVIDIA GPU 之外的硬件 (如 AMD MI300X), 并取得明显的性能提升.

### 6.1 实验设置

**硬件平台.** NVIDIA 和 AMD GPU 是目前最常用的硬件平台, 我们在二者上评估 PipeThreader. 评估使用两款最新的高性能 GPU: NVIDIA H100 (80GB) [Nvi23] 和 AMD Instinct MI300X GPU (192GB) [Amd23]. H100 GPU 使用 CUDA 12.4, MI300X GPU 使用 ROCm 6.1.0. 两款 GPU 都在 Ubuntu 20.04 操作系统上评估.

**DNN 工作负载.** 评估 benchmark 使用六种典型 DNN 模型, 包括 LLAMA3-8B [Dub24], LLAMA3-70B [Dub24], Mamba2-1.3B [Dao24], RetNet-65B [Sun23a], ResNet-50 [He16] 和 UNet [Ron15]. 对于 LLAMA3-8B, LLAMA3-70B 和 RetNet-65B 等大语言模型, 我们采用 (1, 1), (32, 1) 和 (1, 4096) 三种 (BS, SEQ) 配置测试; ResNet-50 和 UNet 等其他模型采用 batch size 1 和 128, 完整覆盖在线与离线推理场景. 对于 Mamba, 我们评估 BS=1, 序列长度分别为 1k, 2k, 4k 和 8k, 以及 BS=32 或 128, 序列长度为 1 的情形. 选择这些配置是因为 Mamba 相比 transformer 的主要优势在于处理长序列时计算效率更高, 而这些设置代表最常用的场景. 我们从每种模型中选择出现最频繁, 代价最高的操作构建 microbenchmark. [表 1](#table-01) 列出代表性算子, 其配置及各算子的缩写.

<span id="table-01"></span>

![原论文表 1, microbenchmark 算子配置子集](./pipethreader/table-01.png)

**表 1.** microbenchmark 中的部分算子配置.

**Baseline.** 我们将 PipeThreader 与 DNN 框架 ONNXRuntime (v1.19.2) [Onn24], 以及 Ladder [Wan24e] 和 PyTorch-Inductor (v2.4.0, 使用 Triton v3.0.0) [Pyt24, Til19] 等当前最先进的 DNN 编译器比较. 对于 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ 精度, PyTorch 集成了 HuggingFace transformers 的官方后端 bitsandbytes [Bit24]. 我们还与 NVIDIA GPU 的厂商专用推理库 TensorRT (v10.0.1) [Ten24] 比较. 其他比较对象包括用于 MatMul 的 cuBLAS [Nvi24] 与 rocBLAS (在 ROCm GPU 上) [Roc25], 用于低精度 MatMul 的 Ladder 与 bitsandbytes 库, 用于 Conv2D 的 MIOpen [Kha19], 以及用 CUTLASS [Nvi24a] 模板编写, 经专家优化的注意力操作 kernel FlashAttention-3 [Sha24b]. 对于 LLM 和 Mamba 模型, 我们还将 PipeThreader 与最常用的 LLM 推理库 vLLM (v0.6.3) [Kwo23] 比较. speedup 等平均性能指标使用所有实验的几何平均值计算. 所有评估都先进行 warm-up 迭代, 再反复执行各项工作负载至少 5 秒, 以得到准确, 稳定的结果.

### 6.2 NVIDIA H100 上的算子性能

[图 12](#figure-12) 展示了 microbenchmark 中所有算子配置的性能. 横轴表示不同算子, 纵轴表示相对于 PipeThreader 的归一化延迟.

<span id="figure-12"></span>

![NVIDIA H100 上 MatMul, convolution, attention 和 Mamba2 工作负载的归一化算子延迟](./pipethreader/figure-12.png)

**图 12.** NVIDIA H100 GPU 上的算子性能.

**MatMul.** [图 12](#figure-12) 第一行展示了 PipeThreader 与其他 baseline 在 LLAMA3-8B (M0-M7) 和 LLAMA3-70B (M8-M15) 所用 MatMul 算子上的性能. 现有编译器或库已经提供充分优化的 MatMul kernel, 但结果表明 PipeThreader 仍然取得了明显加速. 与 PyTorch, Triton 和 Ladder 相比, PipeThreader 的平均 speedup 分别为 1.24× (最高 1.40×), 1.13× (最高 1.26×) 和 2.07× (最高 2.25×). 这一提升来自 PipeThreader 的 sTask 抽象: 它可以把 MatMul 建模为 load 与 mma sTask 构成的流水线, 充分搜索高级调度机会. PipeThreader 的性能与 cuBLAS MatMul 相当, 平均 speedup 为 1.06×. 在大多数 MatMul 算子上, 它还能达到 750 TFLOPS 以上, 接近 H100 GPU 的 TensorCore 理论峰值性能.

**Convolution.** PipeThreader 使用 implicit GEMM [Li16c] 实现 convolution, 同样可以利用 load 和 mma 的流水线优化. 如[图 12](#figure-12) 第二行所示, 在来自 ResNet-50 模型的 convolution 算子 (batch size 为 1 和 128) 上, PipeThreader 明显优于 baseline. 与 PyTorch 和 Ladder 相比, PipeThreader 的平均性能提升分别为 1.94× (最高 3.52×) 和 2.56× (最高 8.66×). 由于没有官方 Conv2D 实现, 我们用 Triton 实现 Conv2D kernel 并执行 auto-tuning, 获得其最佳性能. 与 Triton 相比, PipeThreader 的平均 speedup 为 1.85× (最高 2.47×).

**低 bit MatMul.** [图 12](#figure-12) 第三行展示了来自 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ 量化 LLAMA3-8B (DM0-DM7) 和 LLAMA3-70B (DM8-DM15) 的低 bit MatMul 算子性能. 当前 TensorCore 不直接支持 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ MatMul, 因而低 bit MatMul 必须先在 CUDA core 上把数据转换为 FP16, 这会增加一个 dequant 阶段. 随着流水线阶段增多, 我们观察到 PipeThreader 的 speedup 大于标准 MatMul 上取得的结果. 例如, 在低 bit MatMul 上, PipeThreader 分别比 PyTorch (使用 bitsandbytes) 和 Ladder 快 3.92× (最高 4.76×) 和 2.48× (最高 3.81×), 而标准 MatMul 上的平均 speedup 仅为 1.24× 和 2.07×.

**FlashAttention 与 FlashDecoding.** 与 MatMul 和低 bit MatMul 相比, FlashAttention 包含更深的计算阶段, 为 PipeThreader 打开了更大的优化空间. PipeThreader 可以搜索 sTask-graph 定义的新优化空间与新公开的硬件能力, 自动找到有效的调度方案. [图 12](#figure-12) 第四行展示了来自 LLAMA3-8B (FA0-FA9) 和 LLAMA3-70B (FA10-FA19) 的 FlashAttention 算子性能. 评估覆盖从 512 到 8k 的序列长度, batch size 为 1 或 64, 包括使用和不使用 causal masking 的配置. PipeThreader 相比 Triton 的平均性能提升为 1.36× (最高 1.50×), 高于 MatMul 操作上的 1.13× (最高 1.26×). 这说明该系统能够利用 FlashAttention 更复杂计算流水线固有的更大优化空间.

凭借这种通用调度能力, PipeThreader 可以达到与专家优化的模型专用实现相当的性能, 在某些配置下甚至更快. 针对 NVIDIA Hopper GPU 手工优化的 attention kernel FlashAttention-3, 就是这种专家设计方法的例子. 与 FlashAttention-3 相比, PipeThreader 的平均性能提升为 1.07× (最高 2.18×). FlashAttention-3 是一种手工方法, 无法针对变化的工作负载高效优化. 我们观察到, 对于较短的序列长度, 固定 tile 大小会使 FlashAttention-3 的性能不够理想. PyTorch 使用手工编写的 FlashAttention-2 kernel, 没有加入粒度更细的流水线. 与 PyTorch 相比, PipeThreader 的平均 speedup 为 1.82× (最高 2.29×).

FlashDecoding 算子选自 LLAMA3-8B (FD0, FD1) 和 LLAMA3-70B (FD2, FD3) 模型, batch size 设为 1, context length 设为 8192, 用于模拟 decoding 场景. 与 FlashAttention-3 和 Triton 相比, PipeThreader 的平均 speedup 分别为 1.12× (最高 1.23×) 和 2.27× (最高 3.06×).

**Linear Attention.** [图 12](#figure-12) 第五行和第六行分别展示了 Mamba2 模型的关键 linear attention 操作: ChunkScan 和 ChunkState. 我们将 PipeThreader 与官方 Triton 实现比较. 测试配置的序列长度从 1k 到 16k, batch size 为 1 或 64. 在 ChunkScan 与 ChunkState 操作上, PipeThreader 相比 Triton 的平均 speedup 分别为 1.71× (最高 1.99×) 和 1.98× (最高 2.59×). Triton 在某些配置上还会失败, 如序列长度为 8k (CC14, CT14) 和 16k (CC7, CC15, CT7, CT15) 时. 这些结果说明 PipeThreader 能够适应新的 DNN 操作, 无需手工实现.

### 6.3 NVIDIA H100 上的端到端性能

[图 13](#figure-13) 展示了全部八种 DNN 模型在 NVIDIA H100 GPU 上的端到端性能. 受 GPU 内存限制, 我们使用单个 decoder layer 评估 LLAMA3, Mamba2 和 RetNet 等大语言模型的推理延迟; 由于所有 layer 均相同, 延迟随 layer 数量线性增长, 因而该结果可代表完整模型的性能.

<span id="figure-13"></span>

![八种 DNN 模型在 NVIDIA H100 上的归一化端到端延迟](./pipethreader/figure-13.png)

**图 13.** NVIDIA H100 GPU 上的端到端性能.

**LLM 模型.** 在 FP16 精度的 LLAMA3-8B 和 LLAMA3-70B 模型上, PipeThreader 相比 Ladder 和 ONNXRuntime 的平均 speedup 分别为 2.17× 和 2.45×. Ladder 的调度策略无法有效表示和生成 FlashAttention kernel, 而 ONNXRuntime 本身不支持 FlashAttention, 因而性能不够理想. PyTorch-Inductor, TensorRT 和 vLLM 虽然把业界常用的 FlashAttention kernel 集成为后端, PipeThreader 的平均性能仍分别高出 1.79× (最高 2.15×), 1.28× (最高 1.47×) 和 1.10× (最高 2.05×). 这一提升来自 PipeThreader 为模型中的操作 (如 MatMul 和 FlashAttention) 搜索出更高效的流水线配置. 在 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ 量化场景中, PipeThreader 会同时优化低 bit MatMul 和 FlashAttention 的流水线调度, 相比 Ladder, PyTorch-Inductor 和 vLLM 的平均 speedup 分别为 2.01× (最高 3.39×), 3.03× (最高 11.98×) 和 2.16× (最高 5.16×). [表 2](#table-02) 还列出了部分绝对性能数据.

<span id="table-02"></span>

![原论文表 2, LLAMA3-8B FP16 在 NVIDIA H100 上的延迟](./pipethreader/table-02.png)

**表 2.** LLAMA3-8B-$W_{\mathrm{FP}16}A_{\mathrm{FP}16}$ 在 NVIDIA H100 GPU 上的延迟 (毫秒).

**Linear Attention 模型.** 在 Mamba2-1.3B 模型上, Ladder, ONNXRuntime 和 TensorRT 无法生成高效的 linear attention kernel, 在某些配置下会出现内存错误. 使用 Triton 作为后端的 PyTorch-Inductor 可以运行融合 linear attention, 但性能不如 PipeThreader. PipeThreader 可以为模型中的操作搜索更高效的流水线配置, 相比 PyTorch-Inductor, ONNXRuntime, TensorRT, vLLM 和 Ladder 分别快 1.92× (最高 2.76×), 2.71× (最高 5.10×), 1.21× (最高 2.44×), 1.78× (最高 2.41×) 和 45.93× (最高 84.41×).

对于 linear attention 模型 RetNet-65B, PipeThreader 相比 PyTorch-Inductor, ONNXRuntime, TensorRT 和 Ladder 分别快 1.16× (最高 1.31×), 1.60× (最高 2.12×), 1.06× (最高 1.46×) 和 1.04× (最高 1.10×). PipeThreader 在 RetNet-65B 模型上的 speedup 相对较小. 这是因为 RetNet-65B 模型的 attention head 维度很大 (query 和 key 为 256, value 为 432), shared memory 用量较高, 限制了流水线调度优化.

**CNN 模型.** 对于 ResNet-50 和 UNet, PipeThreader 生成更高效的 Conv2D kernel, 端到端推理延迟相比 Ladder, PyTorch-Inductor 和 ONNXRuntime 分别加快 2.01×, 2.54× 和 3.99×. 与 TensorRT 相比, PipeThreader 的性能相当 (0.97×).

### 6.4 调度策略评估

**联合优化.** 我们的调度策略联合优化 sTask-graph 的划分与流水线调度. 为说明其好处, 我们创建了 PipeThreader 的一个变体 “PT-decouple”, 在两个独立 optimization pass 中分别优化划分 (如提高单个 sTask 的内存利用率) 和流水线调度 (如提高重叠程度). 如[表 3](#table-03) 所示, 在 Mamba2-ChunkScan (BS=64, SEQ=8k) 算子上使用 PT-decouple 时, 编译器侧重于最大化数据复用, 选择较大的 tile shape (如 64×128). 然而, 较大的 tile shape 会限制 sTask-graph 在单个 EU 上实现有效流水线并行的能力, 执行时间为 12.150 ms. 联合优化 sTask-graph 的划分与调度后, 编译器选择较小的 tile shape (如 64×64), 更易形成高效流水线, 执行时间降至 6.981 ms.

<span id="table-03"></span>

![原论文表 3, 解耦与联合 sTask-graph 优化的延迟](./pipethreader/table-03.png)

**表 3.** 解耦与联合 sTask-graph 优化的延迟比较 (毫秒).

**编译时间.** 联合优化需要相对更长的编译时间. [表 4](#table-04) 列出了 PipeThreader 在 FlashAttention 和 Mamba2 若干典型配置上的编译时间. 所有任务划分都在[图 9](#figure-09) 第 2 行生成, 因而调度过程可以并行化, 加快编译. 对于 MatMul 等流水线深度较少的简单 kernel, PipeThreader 只需 0.13 分钟完成编译, Triton 和 CUTLASS 则分别需要 0.17 分钟和 3.36 分钟. 对于 FlashAttention 等复杂融合 kernel, 即使要搜索比 Triton 大得多的流水线空间, PipeThreader 的编译时间仍只有 5.26 分钟.

<span id="table-04"></span>

![原论文表 4, NVIDIA H100 上的编译时间](./pipethreader/table-04.png)

**表 4.** H100 上的编译时间 (分钟).

### 6.5 AMD ROCm GPU 上的评估

**算子性能.** 我们从原本为 NVIDIA H100 GPU 设计的 microbenchmark suite 中选取部分算子, 在 AMD MI300X GPU 上运行 benchmark. 评估关注以下主要算子: MatMul (与 PyTorch, rocBLAS, Triton 和 Ladder 比较), Conv2D (与 PyTorch, MIOpen, Triton 和 Ladder 比较), FlashAttention (与 FlashAttention-2 和 Triton 比较), 以及 Linear Attention (与 Triton 比较). [图 14](#figure-14) 表明, 在不同类型的算子上, PipeThreader 相比 Triton 的 speedup 为 1.16× 至 5.42×, 相比 PyTorch 最高为 6.21×. 在 MatMul 和 Conv2D 上, PipeThreader 分别比 rocBLAS 和 MIOpen 最高快 1.77× 和 2.21×. PipeThreader 相比 FlashAttention-2 的 speedup 最高也达到 2.82×. PipeThreader 相比 Ladder 的平均 speedup 则为 1.45×, 说明它具有效率和可扩展性.

<span id="figure-14"></span>

![AMD MI300X 上的归一化算子延迟](./pipethreader/figure-14.png)

**图 14.** AMD MI300X GPU 上的算子性能.

**端到端性能.** 我们在 AMD Instinct MI300X GPU 上评估 PipeThreader, 与 Ladder, PyTorch-Inductor, ONNXRuntime 和 vLLM 比较. [图 15](#figure-15) 给出八种模型的端到端性能结果, 包括分别采用 $W_{\mathrm{FP}16}A_{\mathrm{FP}16}$ 和 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ 格式的 LLAMA3-8B 与 LLAMA3-70B, 以及 Mamba2-1.3B, RetNet-65B, ResNet-50 和 UNet.

<span id="figure-15"></span>

![八种 DNN 模型在 AMD MI300X 上的归一化端到端延迟](./pipethreader/figure-15.png)

**图 15.** AMD Instinct MI300X GPU 上的端到端性能.

在 FP16 精度的 LLAMA3-8B 和 LLAMA3-70B 模型上, PipeThreader 相比 PyTorch-Inductor, ONNXRuntime, vLLM 和 Ladder 的 speedup 分别为 1.48× (最高 2.77×), 6.33× (最高 15.51×), 1.02× (最高 1.32×) 和 1.07× (最高 1.29×). 在 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ LLAMA3-8B 和 LLAMA3-70B 模型上, PipeThreader 相比 PyTorch-Inductor 和 Ladder 的 speedup 分别为 3.97× (最高 12.66×) 和 1.12× (最高 1.34×), 而 ONNXRuntime 与 vLLM 不支持 ROCm 平台上的 $W_{\mathrm{FP}4}A_{\mathrm{FP}16}$ 量化.

对于 Mamba2-1.3B, PipeThreader 相比 Ladder 的平均 speedup 达到 32.93× (最高 61.33×). 这一大幅性能提升主要是因为 Ladder 无法融合 Linear Attention 组件. 与 PyTorch-Inductor 相比, PipeThreader 的平均 speedup 为 1.31× (最高 1.54×). 在 RetNet-65B 模型上, PipeThreader 相比 PyTorch-Inductor, ONNXRuntime 和 Ladder 的 speedup 分别为 1.03× (最高 1.36×), 4.75× (最高 5.73×) 和 1.01× (最高 1.02×).

在传统 CNN 模型 (ResNet-50 和 UNet) 上, PipeThreader 相比 PyTorch-Inductor, ONNXRuntime 和 Ladder 的 speedup 分别为 2.74× (最高 5.66×), 5.84× (最高 15.47×) 和 2.14× (最高 6.54×).

MI300X GPU 的异步能力相对较弱, shared memory 也小于 NVIDIA H100 GPU, 因而 PipeThreader 所实现流水线并行的潜在性能 (如相对于 baseline 的 speedup) 会有所降低.

## 7 讨论

**相比手工 kernel 的优势.** PipeThreader 相比手工实现有根本优势, 尤其体现在自动流水线调度与跨架构可移植性上.

第一, 它不再需要专家级手工调优. 手工设计高效流水线调度容易出错, 耗时很长, 而且对输入配置敏感. 即使 FlashAttention-3 (FA3) 这类专家编写的 kernel 最初也不支持某些维度 (如 head size 256), 足见其难度. PipeThreader 将该过程自动化, 在硬件约束下系统搜索调度空间. 它相比 FA3 的 speedup 最高为 2.18×, 相比 vLLM 中基于 Triton 的 Mamba2 快 2.41×.

第二, 它可以很好地泛化到不同硬件. 手工调优的 kernel 往往与特定平台紧密绑定, 特别是 NVIDIA GPU, 而 PipeThreader 在 AMD 硬件上也能取得明显提升. 它的抽象还可以自然映射到类似 TPU 的架构 [Jou21, Jou17a] (如 TPU core 和 DMA engine), 实现高效的流水线执行.

最后, 它降低了获得高性能的门槛. 在 Multihead Latent Attention (MLA) [Dee24, Dee24a] 上, PipeThreader 只用 80 行 Python 就达到相比 Triton 最高 5× 的 speedup, 性能与 DeepSeek 超过 500 行的 CUDA 实现 [Fla25] 相当, 但开发工作少得多.

**扩展到多 GPU.** PipeThreader 可以自然扩展到多 GPU, 并与 tensor parallelism 引入的 collective communication 配合, 方法是把 1) GPU 间的通信单元 (如 RDMA, NVLINK, IB) 纳入 sEU, 把 2) collective communication 纳入 sTask. 这样一来, PipeThreader 可以复用该策略, 在 GPU kernel 层搜索 collective communication 与计算之间的高效流水线, 并扩展到多 GPU 或多 node 环境. 当前结果表明, 在常见通信模式上, PipeThreader 的性能与 TileLink [Zhe25a] 等先进系统相当.

**支持新设备.** 我们发现, 广泛使用的硬件 (如 NVIDIA/AMD GPU 或 TPU) 符合 sEU 抽象, 其中包含所有 sEU 的偶数集合. 要把 sTask-graph 编译到设备上, 编程模型只需用各 sEU 自己的 load/store/compute 指令实现其 `Execute` 接口 ([图 4](#figure-04)). 这种设备虚拟化类似 Roller [Zhu22] 和 Welder [Shi23a] 的硬件抽象, 但进一步公开了细粒度异构 sEU.

**MoE FFN kernel.** PipeThreader 还可以支持 MoE FFN kernel 中的 grouped MatMul. 与 batched MatMul 不同, 每个 group 可以采用不同 shape. 为此, PipeThreader 可以把每个 group 分解为独立的 sTask-subgraph, 各自使用自己的输入 shape 并分别应用策略, 而非共享同一调度.

## 8 相关工作

**深度学习编译器与框架.** 大多数现有 DNN 编译器把硬件抽象为同构执行单元 (EU). Rammer [Ma20] 引入 rTask 概念, 用于跨 EU 并行执行; Welder [Shi23a] 则侧重通过纵向融合进行整体内存优化. PipeThreader 与二者不同, 它引入 sTask 和 sEU, 明确公开硬件异构性, 以便优化和调度流水线并行.

TVM [Che18], Ansor [Zhe20], XLA [Xla17] 和 TensorRT [Ten24] 等 DNN 编译器已广泛采用算子融合来减少内存开销, 因而产生了更深的计算阶段. Triton [Til19], Welder [Shi23a], Roller [Zhu22], Cocktailer [Zha23h], TensorIR [Fen23], ThunderKittens [Res24], FractalTensor [Liu24f] 和 Ladder [Wan24e] 等编译器以 tile 抽象为基础进行调度优化. 然而, 这些工作主要关注通过空间 tiling 改善数据局部性, 实现跨 EU 数据并行, 却忽略了跨 sEU 流水线并行的机会.

Triton [Til19] 和 CUTLASS [Nvi24a] 等工作虽然包含流水线执行, 但依赖针对特定算子的临时规则, 无法泛化到各种工作负载. PipeThreader 引入可自动调度和优化流水线并行的抽象, 弥补了这些不足.

ALCOP [Hua23b] 等框架关注数据加载与计算之间的流水线, 以优化内存层次利用率. 然而, 它们没有充分利用现代计算单元的异构性, 也没有为 FlashAttention 等具有深层计算阶段的工作负载搜索流水线调度. PipeThreader 引入粒度更细的抽象, 可以跨异构硬件组件进行完整的流水线优化, 填补了这一空缺.

**特定模式的优化.** 现有编译器缺少 sTask 和 sEU 抽象, 因此流水线并行优化往往需要针对特定模式手工定制. 例如, FlashAttention [Sha24b] 和 CUTLASS [Nvi24a] 中的 Hopper MatMul 提供特定模式的调度, 但需要大量人工工作. FlashAttention 还为不同输入提供独立调度, CUTLASS [Nvi24a] 则要求用户执行 profiling 并选择最佳调度. PipeThreader 利用 sTask 与 sEU 抽象泛化流水线并行, 无需人工干预即可自动调度大量算子和配置.

**分布式深度学习框架.** Centauri [Che24f], PrimePar [Wan24g] 和 TileLink [Zhe25a] 分别通过分层调度, 时序张量划分和基于 tile 的抽象来改善通信-计算重叠. PipeThreader 可以把通信和计算建模为独立 sTask, 因而既能表示这些工作提出的调度策略, 又能进行范围更广的调度优化.

## 9 结论

随着 DNN 模型变大, 异构专用硬件单元不断出现, 硬件调度器已经不足以高效执行流水线. 本文提出 DNN 编译器 PipeThreader, 它通过 sTask-graph 抽象与结合虚拟化 EU 和专用 sEU 的分层硬件能力, 实现软件定义流水线. 借助核心调度原语, PipeThreader 将流水线调度自动化, 在 H100 和 AMD GPU 上达到或超越 FlashAttention 等当前最先进方法的性能, 同时还能泛化到 Mamba2 等新模型. 我们认为, PipeThreader 为进一步发展基于编译器的优化奠定了基础, 使不断演进的 GPU 架构和 DNN 工作负载得到高效利用.

## 致谢

感谢匿名审稿人和 shepherd Deepti Raghavan 提出的详尽建议. 本工作部分得到国家自然科学基金项目 92464301 的资助. Zhi Yang 为通讯作者.
