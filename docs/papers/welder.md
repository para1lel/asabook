---
title: 'Welder: Scheduling DNN Memory Access'
createTime: 2026/09/11 12:00:00
permalink: /papers/welder/
pageClass: paper-reading
---

> [Yining Shi](https://dblp.org/pid/161/3927-1.html) [+internship], [Zhi Yang](https://yangzhihome.github.io/), [Jilong Xue](https://dblp.org/pid/06/10336.html), [Lingxiao Ma](https://xysmlx.github.io/), [Yuqing Xia](https://dblp.org/pid/211/8365.html), [Ziming Miao](https://dblp.org/pid/216/9568.html), [Yuxiao Guo](https://dblp.org/pid/22/329-1.html), [Fan Yang](https://dblp.org/pid/29/3081-24.html), [Lidong Zhou](https://www.microsoft.com/en-us/research/people/lidongz/). 论文发表于 2023 年 7 月的 *第 17 届 USENIX 操作系统设计与实现研讨会 (OSDI 23)*, 第 701-718 页. [Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/conference/osdi23/presentation/shi). <a href="/paper/welder.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源码; 精确措辞, 印刷版式和参考文献均以出版 PDF 为准.

## 摘要

随着处理更高保真度数据的需求不断增长, 新型硬件加速器又开始采用更快的计算核心, 现代深度神经网络 (DNN) 越来越受内存限制. 在多种常用 DNN 模型中, 人们已经观察到计算核心利用不足而内存带宽饱和的差距. 这种低效既源于传统上把 DNN 当作计算密集型工作负载, 也源于 DNN 模型缺少全局性的内存访问优化.

本文介绍 WELDER, 一种从全局内存访问角度优化执行效率的深度学习编译器. WELDER 的核心是 tile-graph, 这一抽象支持在 tile 粒度上进行细粒度数据管理. 利用不同内存层之间优化相互独立这一观察, WELDER 可以把整个组合式 DNN 优化空间分解成若干独立空间, 并借助基于 tile 流量的代价模型, 有效权衡算子内与算子间的数据复用. 由此, WELDER 把以往临时设计的内存优化统一到一个空间中, 生成包含 89 种更多优化模式的高效执行计划, 并显著优于当前最佳方案. WELDER 还可以把现有加速器内存和主机内存组合成一个整体系统, 从而处理输入规模任意大的 DNN 模型.

<span id="section-1"></span>

## 1 引言

深度神经网络 (DNN) 已广泛用于视觉和语言分析与生成等任务. 传统观点把 DNN 视为计算密集型工作负载. DNN 模型通常定义为数据流图 (DFG), 其中每个节点代表一个计算密集型算子, 例如矩阵乘法. 这些算子会卸载到具有大规模并行计算核心的现代加速器上, 例如 GPU 和 TPU [Jou21], 以加快计算. 为了高效利用加速器, DNN 框架与编译器探索了多种优化技术, 如代码特化 [Che18e, Zhe20, Zhu22] 和算子融合 [Che18e, Ma20a].

这些以计算为中心的优化在经典 DNN 模型上已经证明有效, 但我们观察到现代 DNN 正日益受内存限制. 对一系列当前最佳 DNN 模型的性能分析表明, 端到端 DNN 计算的瓶颈大多位于 GPU 内存. 内存带宽利用率最高可达 96.7%, 而计算核心的平均利用率只有 51.6% ([第 2 节](#section-2)). 此外, 随着硬件和 DNN 模型不断演进, 利用不足的核心与饱和的内存带宽之间的差距还可能继续扩大. 现代模型需要处理保真度更高的数据, 例如更大的图像, 更长的句子和高清图形, 因而在计算中消耗更多内存带宽. 同时, 更快的计算核心, 例如 TensorCore [Nvi17], 也会给内存带来更大压力.

优化内存密集型 DNN 工作负载并不容易, 因为这需要改进跨越多个内存层, 例如 GPU DRAM 与共享内存的复杂数据访问和复用模式. 从内存角度看, DNN 计算要求每个算子反复执行以下过程: 1) 跨内存层次加载输入张量, 2) 在计算核心上执行计算, 3) 跨内存层次存储结果张量. 要得到良好的数据访问模式, 必须仔细计算 tile, 即张量分块, 在各个维度上的大小. 在现有实践中, 得到这样的分块策略本身已经很困难 [Cut23, Zhe20, Zhu22]. 更棘手的是, 由于算法语义不同, 每个算子可能需要不同的数据访问模式. 算子之间的这种差异使算子间数据复用格外困难, 甚至常常无法实现. 如果某个算子在特定内存层得到的 tile 形状与下游算子不匹配, 就很难在该层复用这个 tile. 因此, 现有方法要么专注于算子内优化, 并把所有算子间中间张量都留在最低内存层, 例如 GPU 内存; 要么依赖基于规则的算子融合来减轻算子间内存开销. 这些规则只适用于特定算子组合, 例如逐元素算子的寄存器融合 [Pyt17, Aba16c, Che18e], 以及仅覆盖少数算子类型的共享内存融合 [Zhe22f]; 当输入规模或硬件配置发生变化时, 其结果可能并非最优.

本文介绍 WELDER, 一种对由通用算子组成的端到端 DNN 模型进行全局内存访问优化的深度学习编译器. WELDER 的设计建立在三项关键观察之上. 第一, 为了解决两个相邻算子之间可能存在的 tile 形状冲突, 我们观察到, 只要能精确保留每个算子的计算逻辑, 例如用张量表达式描述, 就可以从后向前传播输出 tile 形状, 自动推导出相互对齐的 tile 形状. 第二, 为了判断哪种 tile 形状能带来更好的性能, 只要让计算模式与硬件特性, 例如 TensorCore, 对齐, 就只需最小化所有内存层上的数据流量. 对于 tile 配置已经对齐的算子, 可以根据输入/输出 tile 大小和输入/输出张量形状轻松建立数据流量模型. 最后, 从整个内存层次来看, 我们观察到内存流量优化在不同内存层之间本质上彼此独立, 即层间独立性. 具体来说, 上述流量模型只由所关注内存层上的 tile 配置决定. 借助这些观察, 我们可以用一套有效流程优化整个空间: 先在各个独立内存层对齐两个相邻算子, 再由流量代价引导, 在适当内存层确定最优分块大小, 然后把优化扩展到更多算子.

WELDER 把这些认识融入一种新的 DNN 编译器设计. 第一, 为了支持细粒度数据管理, WELDER 提出 tile-graph, 一种用于建模 DNN 计算的 tile 级数据流图. 图中每个节点一次处理张量的一个数据 tile. 为了把 DNN 计算映射到多层内存层次, WELDER 允许控制每个节点的数据 tile 大小, 以及两个节点之间复用数据 tile 所在的内存层. 具体而言, WELDER 提供 SetConnect 接口来设定每条边的数据复用层, 并提供 Propagate 接口来推导一组相连节点中的 tile 配置. 第二, 为了高效地全局优化 tile 级数据流调度, WELDER 利用数据流计算的层间独立性, 把优化空间拆成多个子空间. 在此基础上, WELDER 提出两层调度策略: 为每条边枚举不同的内存连接选项, 再由流量代价模型引导, 为每个子空间确定高效的 tile 配置. 最后, 优化后的执行计划通过硬件层定义的四个抽象计算接口 Allocate, LoadTiles, ComputeTile 和 StoreTiles, 映射为特定硬件加速器上的可执行代码.

借助 tile 级全局数据流调度, WELDER 首次把所有常见算子融合, 例如基于寄存器的逐元素融合和共享内存融合, 统一到一个框架中. 这种通用性使 WELDER 能自动找到 89 种非常规算子融合模式, 其中大多数未被现有基于规则的方法探索过 ([第 5.2 节](#section-5-2)). 值得注意的是, 我们的方法还可以轻松支持处理任意大输入 DNN 模型的新需求, 例如高分辨率图像; 在这种情况下, 甚至单个算子也可能大到无法装入 GPU 内存. 具体来说, WELDER 可以在当前内存层次中加入额外层, 例如主机内存, 从而在主机与设备内存的组合层次上生成优化后的执行计划.

我们在 TVM [Che18e], Rammer [Ma20a] 和 Roller [Zhu22] 之上实现了 WELDER. 评测采用 10 个当前最佳 DNN 模型, 覆盖经典和近期模型结构, 涉及视觉, NLP, 3D 图形等多种任务. 结果表明, WELDER 在 NVIDIA 和 AMD GPU 上均显著优于 PyTorch, ONNXRuntime 和 Ansor 等当前最佳 DNN 框架及编译器, 最高加速比分别达到 21.4×, 8.7× 和 2.8×. WELDER 的自动优化甚至优于 TensorRT [Ten17a] 和 Faster Transformer [Fas21], 前者是高度优化的手工 DNN 推理库, 后者是 NVIDIA 面向特定模型的实现; 最高加速比分别达到 3.0× 和 1.7×. 此外, 在 TensorCore 等计算核心更快的硬件上运行这些模型时, 我们观察到更大的性能提升, 这凸显了内存优化对未来 AI 加速器的重要性.

<span id="section-2"></span>

## 2 动机

现代 DNN 受内存带宽限制. [图 1](#figure-01) 给出了一个代表性 DNN 基准在 ONNXRuntime [Onn21] 上运行时的平均 GPU 利用率, 同时包括计算 FLOPS 和全局内存吞吐量. 如图所示, 平均计算利用率只有 51.6%, 内存利用率却达到 96.7%. 从模型类型看, 以卷积和矩阵乘法算子为主, 计算利用率相对较高, 例如超过 80% 的 ResNet 与 BERT 是两个典型经典模型. 其余模型都是近年提出的常用模型, 由于在计算密集型算子之外引入了更多内存密集型模式, 计算效率较低. 我们还观察到, 与经典模型相比, 新型 DNN 模型的内存写流量与读流量之比往往更高. 主要原因是这些模型倾向于处理高保真数据, 并在各层产生大型激活. 但 ONNXRuntime 等现有系统对减少算子间流量的优化有限. 这意味着这些模型会频繁经由全局内存在算子之间交换大量中间数据. 这些结果说明, 有必要优化跨算子的内存访问效率.

<span id="figure-01"></span>

![图 1. 不同模型的计算和内存利用率.](./welder/figure-01.png)

**图 1.** NVIDIA V100 GPU 上不同模型的计算 FLOPS 和内存带宽利用率.

**相互冲突的算子内与算子间数据复用模式.** 同时优化算子内和算子间数据复用很困难. 一个算子通常实现为遍历所有张量维度的多层嵌套循环. 算子内部跨多个内存层的数据复用, 通常通过复杂的循环分块技术隐式优化 [Cut23, Zhe20, Zhu22]. 我们考虑两个连续算子 Matmul 和 Softmax 组成的典型模式. 当这两个算子独立优化时, 它们在共享内存中的最优 tile 大小不同, 例如 Matmul 为 $[32\times64]$, Softmax 为 $[4\times128]$. 因此, Softmax 无法在共享内存中复用 Matmul 的中间数据, 总延迟达到 0.36ms, 如[图 2](#figure-02) 所示. 但如果强制它们同时考虑算子内和算子间数据复用, 融合算子的延迟可以降到 0.29ms, 获得 1.26x 加速. 检查对齐后的 tile 大小, 即 $[16\times128]$ 可见, 两个算子都牺牲了自身效率; 由于算子内数据复用采用的 tile 并非最优, 两者独立运行时性能分别下降 15% 和 4%, 但整体效率得到提升. 这说明, 要全局优化内存访问, 就需要一种能同时覆盖算子内与算子间数据复用的高效方案.

<span id="figure-02"></span>

![图 2. 未融合与融合的 Matmul-Softmax 延迟.](./welder/figure-02.png)

**图 2.** 未融合, 融合以及 Matmul 和 Softmax 各自内核的延迟.

**关键观察.** 进一步分析[图 2](#figure-02) 中的例子后, 我们得到三项关键观察. 第一, 从一个输出 tile 形状出发进行链式形状推导, 可以得到跨算子对齐的 tile 配置. 例如, 如果要计算 Softmax 的 [4×128] 输出 tile, 就可以根据其计算逻辑, 例如张量表达式, 推导出它依赖的输入 tile 形状也是 [4×128]. 再把 [4×128] 作为 Matmul 的输出 tile, 还可以推导出 Matmul 的输入 tile 形状分别为 [4×k] 和 [k×128], 其中 k 是归约大小, 可以设为不超过 Matmul 归约维大小的任意数值. 这样, 两个算子便可以在共享内存中复用中间数据 tile [4×128], 从而完成融合.

第二, 给定对齐后的 tile 配置与原始张量形状, 可以通过解析方法轻松推导总内存流量. 在这个例子中, Matmul 的输入张量 A 和 B 的形状分别为 [98304×64] 和 [64×128], 输出张量 C 的形状为 [98304×128]. 随后, Softmax 以 C 为输入, 生成同形状的输出张量 D. 输入张量 A, B 和输出张量 D 位于全局内存中. 根据这些形状, 可以先计算张量 D 的单个输出 tile, 即 [4×128] 对应的内存流量. 为了完成计算, Matmul 会先从张量 A 加载形状为 [4×k] 的 tile, 再从张量 B 加载 [k×128] 的 tile; 中间 tile [4×128] 随后由 Softmax 在共享内存中使用, 最后把形状为 [4×128] 的 tile 写入张量 D. 依据输入张量 [98304×64] 的形状, k 可以代入 64. 因而, 单个输出 tile 在全局内存中产生的总流量为 35KB, 即 ((4*64+64*128+4*128)*4Bytes(FP32)); 由于在共享内存中复用了数据, 中间 tile [4×128] 的流量被省去. 要计算完整的输出张量 D, 总共需要 24,576 次这样的计算, 即 (98304*128)/(4*128), 因此全局内存总流量为 840MB, 即 24,576*35KB. 有意思的是, 按照同样的计算, 把输出 tile 改为 [16×128] 后, 总流量会降至 264MB.

最后, 张量形状确定后, 流量代价计算只取决于所关注内存层上的 tile 配置, 例如共享内存中的输出 tile 形状 [4×128] 或 [16×128]. 因此, 我们可以独立选择每一层的 tile 大小, 以优化来自较低内存层的流量代价.

这些观察提供了一种有效的全局内存访问优化方法: 通过输出 tile 形状对齐一组相邻算子, 根据内存流量确定最佳 tile 形状, 再独立优化每个内存层. 这样, WELDER 可以把原本粗粒度的算子间依赖转变为更细粒度的 tile 级依赖, 实际上消除算子之间的一部分虚假屏障, 提高并发性.

<span id="section-3"></span>

## 3 WELDER 设计

[第 2 节](#section-2) 的观察促使我们设计了 WELDER, 这是一种在全局内存访问调度空间中提升现代 DNN 性能的深度学习编译器. [图 3](#figure-03) 展示了系统概览. WELDER 接收完整的 DNN 模型作为输入, 把它转换成由 tile 计算任务, 即 operator-tile, 构成的数据流图, 称为 tile-graph ([第 3.1 节](#section-3-1)). tile-graph 可以细粒度控制数据 tile 配置和内存放置. 给定 tile-graph 后, WELDER 通过“先连接, 后调度”的方法解决算子内与算子间数据复用冲突: 它先假定两个相邻算子可以在某个内存层复用数据 tile, 即建立连接, 再推导最佳公共 tile 形状, 判断能否减少总内存流量. 为此, WELDER 提供两个 tile-graph 调度接口: SetConnect 和 Propagate, 后者用于形状推导链. 在此基础上, 我们提出由图连接和子图调度组成的两步调度算法, 递归确定适用于多层内存的高效 tile-graph 执行计划, 称为分层 tile-graph ([第 3.2 节](#section-3-2)). 最后, 该计划通过硬件层定义的四个抽象计算接口 Allocate, LoadTiles, ComputeTile 和 StoreTiles, 映射为特定硬件加速器的可执行代码 ([第 3.3 节](#section-3-3)). 抽象加速器的内存规格由 tile-graph 调度层用于引导优化过程.

<span id="figure-03"></span>

![图 3. WELDER 系统概览.](./welder/figure-03.png)

**图 3.** WELDER 系统概览.

<span id="section-3-1"></span>

### 3.1 Operator-tile 与 Tile-graph

WELDER 用一种名为 operator-tile 的细粒度任务粒度定义 DNN 计算. 卷积等 DNN 算子可以实现为多个同质 operator-tile, 它们以流式或并行方式执行, 从而计算输出张量中的所有数据 tile [Ma20a]. 每个 operator-tile 接收从输入张量切分出的一个数据 tile, 并在输出张量中计算一个数据 tile; 计算逻辑由基于索引的张量表达式描述 [Che18e]. [图 4a](#figure-04) 和[图 4b](#figure-04) 给出了 Conv 与 MaxPool 的 operator-tile 示例. Conv 算子接收一个 $[3\times3\times C]$ 数据 tile, 计算出一个 $[1\times1\times C]$ 数据 tile; MaxPool 算子接收 $[2\times2\times F]$ 输入 tile, 计算出 $[1\times1\times F]$ 输出 tile.

<span id="figure-04"></span>

![图 4. Operator-tile 与相连的 tile-graph.](./welder/figure-04.png)

**图 4.** 两个 operator-tile 的示意图: (a) Conv, (b) MaxPool; 以及 (c) 把它们连接成 tile-graph, 为简洁起见省略 Conv 的权重张量.

为了提高共享内存等分层内存资源的利用率, WELDER 允许两个相邻 operator-tile 通过一个公共中间数据 tile 建立“连接”, 这个 tile 也称为 reuse-tile. 这样, 第二个 operator-tile 可以直接使用第一个 operator-tile 产生的数据, 而不必把它实体化为完整的中间张量. [图 4](#figure-04)(c) 展示了 Conv 与 MaxPool 的两个 operator-tile 通过 [2 × 2 × F] reuse-tile 建立连接的例子. 多个 operator-tile 可以沿每条相邻边连接, 形成 operator-tile 数据流图, 即 tile-graph.

**Tile 传播.** 建立连接后, tile-graph 中的大多数 tile 会相互关联, 可以从一个输出 tile 形状向整张图传播, 自动推导出其他 tile. 具体做法是从输出节点向输入执行链式形状推导. 对每个 operator-tile, 分析其张量表达式与输出 tile 大小, 可以精确确定输入张量中依赖的区域. 如果输入区域可能包含稀疏或不连续访问等不规则模式, 例如 Gather 或带步幅的 Convolution, 表达式分析会给出一个保守的输入 tile 形状上界. 如果 tile-graph 有多个输出节点, 它们的输出形状也可能相关, 因为它们可能共享图中的同一祖先节点. 此时, 传播第一个输出 tile 后, 我们会为其余输出节点分别传播形状, 并将其与第一个输出对齐. 如果两次传播得到的 tile 形状不一致, 就不把后一个输出节点连接到当前图中.

**内存流量与占用.** 完成 tile 传播后, 可以确定 tile-graph 的内存流量与占用. 首先, 单个 tile-graph 的内存流量等于其输入和输出 tile 大小之和. 再把该值乘以计算完整输出张量所需的 tile-graph 数量, 例如用张量大小除以输出 tile 大小, 就能得到总流量. 其次, 可以采用内存分配算法, 例如 bestfit [Gar72], 按拓扑顺序分配所有数据 tile, 计算 tile-graph 的最小内存占用. 为了进一步优化占用, 还可以把含有归约轴的输入 tile 划分成更小的 tile, 按顺序加载和使用它们, 并把结果累加到输出 tile. 具体的策略可以在 tile 传播期间自动尝试沿归约轴采用不同的分块大小.

<span id="section-3-2"></span>

### 3.2 Tile-graph 调度

要把初始数据流图表示的 DNN 模型映射到加速器, 可以递归地把每个算子划分为多个 operator-tile, 使其装入各个内存层, 再在较高内存层连接 operator-tile, 利用算子间数据复用. 由此, 整个 DNN 计算可以建模为二维空间上的数据流式流水线: 数据 tile 在内存层次中纵向上下移动, 同时在各层横向传给后继算子. [图 5](#figure-05) 展示了把三个连续算子 Conv, ReLU 和 MaxPool 映射到三层内存层次, 例如从 L2 到 L0 的例子. Conv 算子的输入 tile 会反复从 L2 加载到 L1, 再加载到 L0 进行计算. 在 L0 连接 Conv 和 ReLU 后, Conv 的输出可以复用为 ReLU 的输入, 两个算子在 L0 形成一个 tile-graph. 同时, 它们在 L1 合并成一个虚拟节点, 即 Conv+ReLU. 随后, ReLU 的输出持续写入 L1 的数据 tile, 并通过在 L1 进一步连接而复用为 MaxPool 的输入. 这样, 三个算子便在 L1 层形成单个 tile-graph, 并在 L2 形成虚拟节点 Conv+ReLU+MaxPool. 完成这一递归过程后, 所有算子都会在最低层连接成单个 tile-graph.

<span id="figure-05"></span>

![图 5. 映射到三层内存层次的三个算子.](./welder/figure-05.png)

**图 5.** 把三个连续算子映射到三层内存层次, 省略 Conv 的权重.

**解耦优化空间.** 根据 DNN 计算大多受内存限制这一观察, 数据流式流水线的主要优化目标可以转化为最小化内存流量. 利用不同内存层之间流量优化固有的独立性, 可以把整个优化空间分解成若干子空间. 具体来说, 对于给定的 tile-graph, 从较低内存层加载和存入该层的数据总流量, 只需根据其输出 tile 形状估算, 因为这个形状可以推导出全部输入和输出 tile 形状. 借助这一性质, 同一或不同内存层上的不同 tile-graph 可以分别搜索最优 tile 形状, 独立优化各自的内存流量. 例如, 在[图 5](#figure-05) 中, L0 上由 Conv 和 ReLU 构成的 tile-graph 可以独立于 L1 tile-graph, 即由 Conv+ReLU 与 MaxPool 算子构成的图进行优化, 这称为层间独立性. 这还意味着, L0 上 Conv-Relu 与 MaxPool 子图的最优 tile 配置同样彼此独立, 因为它们分别独立于 L1 的 tile-graph; 我们把这种性质称为层内独立性. 实际上, 唯一约束是较低内存层的 tile 大小必须大于较高内存层的 tile 大小. 这一条件通常成立, 因为较低内存层的容量一般更大. 借助这些性质, 给定图连接计划后, 可以独立调度每个 tile-graph.

<span id="figure-06"></span>

![图 6. WELDER 调度接口.](./welder/figure-06.png)

**图 6.** WELDER 的调度接口.

**调度接口.** 如[图 6](#figure-06) 所示, WELDER 提供两个调度接口, 用于控制图连接和子图分块. 首先, 图连接通过 SetConnect 接口实现, 它为 tile-graph 中的一条边指定内存层, 默认为最低层. 建立连接后, 通过 Propagate 接口指定输出 tile 各维的大小, 以及输入 tile 中可选的归约轴, 即可推导图中的 tile 形状. 例如, 在[图 5](#figure-05) 中, 可以用 SetConnect 在 L0 连接 Conv 和 Relu, 在 L1 连接 Relu 和 MaxPool. 建立连接后, 对于 Conv+Relu 子图, 可以把输出 tile 形状指定为 $[1,1]$, 再用 Propagate 推导中间 reuse-tile 的形状, 即 $[1,1]$. 同样, 通过指定 $[1,1,F]$ 的输出 tile 形状, 也可以推导 Conv+Relu+MaxPool 子图的中间 reuse-tile 形状, 即 $[2,2,F]$. 这两个调度原语本质上是更新 tile-graph 边和顶点的两个接口. SetConnect 用于在两个节点之间添加连接, Propagate 用于设置节点的 tile 配置. 两者共同组成更新 tile-graph 的完整接口. 注意, 这些原语只由 WELDER 的调度策略使用, 对最终用户透明. WELDER 还提供 MemFootprint 和 MemTraffic 两个代价接口, 分别计算 tile-graph 的内存占用和总流量, 作为引导调度的代价模型.

<span id="figure-07"></span>

![图 7. 两步 tile-graph 调度算法.](./welder/figure-07.png)

**图 7.** 两步 tile-graph 调度算法.

**调度策略.** WELDER 采用两步调度算法有效优化数据流计算. 具体而言, 图连接调度器先为每条边设定不同的内存复用层, 枚举不同的图连接计划; 随后, 子图调度器为图连接调度器拆分出的每个子图快速搜索高效的 tile 配置. [图 7](#figure-07) 展示了 WELDER 的两步调度算法. 首先, 给定 DNN 数据流图 $g$ 和加速器设备 $d$, 图连接调度器按拓扑顺序枚举所有图节点及其输出边, 即第 1-3 行. 对每条边, WELDER 尝试不同的连接层, 例如使用 SetConnect 接口, 即第 5 行. 随后, 它提取所有边的连接层都高于 0 的相连子图. 这里用数字 0 表示最低内存层, 更大的数字表示更高层. ExtractSubgraph 函数在第 26-31 行实现. 对于提取出的子图, WELDER 调用 SubGraphTiling 函数得到若干高效 tile 配置, 再在硬件上进行性能分析, 选出最优配置, 即第 7-10 行. 与其他所有连接层比较后, WELDER 为当前边设定最佳连接层.

接着, 子图调度器, 即 SubGraphTiling 函数, 接收一个子图和上一层 tile 配置, 搜索当前层的高效 tile 配置. 首先, WELDER 使用类似 Roller [Zhu22] 的 tile 形状扩展方法, 枚举输出维度的 tile 大小, 即第 14 行的 EnumerateSubtiles. 该方法从初始 tile 形状, 例如大小为 1 开始, 朝着能够减少总流量并与硬件特性对齐的形状扩展. 得到输出 tile 形状后, 可以用 Propagate 接口推导完整 tile 配置, 再用 MemFootprint 接口检查其是否超出内存容量; 若未超出, 则以该配置的内存流量为键, 例如由 MemTraffic 接口取得, 把它加入有序结果列表, 即第 15-18 行. 最后, 选出当前层中内存流量最低的前 $K$ 个配置, 再提取更高层子图, 并通过递归调用 ExtractSubgraph 和 SubGraphTiling 确定其最佳 tile 配置, 即第 20-24 行.

WELDER 不对不同内存层的容量作任何假设, 因为调度策略总能尽力确定放置中间数据的最佳层与 tile 大小, 以最小化总延迟. 不过, WELDER 始终更适合拥有大容量高层高速内存, 例如共享内存的硬件, 因为其中可以容纳足够大的中间数据 tile; tile 过小会削弱算子内数据复用. WELDER 对数据流图的调度结果是分层 tile-graph: 它在最低内存层从完整图开始, 在更高层递归拆分为多个子图, 直至最高层.

<span id="section-3-3"></span>

### 3.3 映射到硬件加速器

WELDER 生成的分层 tile-graph 是一种抽象执行计划, 可以映射成特定硬件加速器的可执行代码. 为便于映射, WELDER 提供带有分层内存的抽象加速器设备. 内存层数, 每层容量与事务宽度等配置可以通过 MemLevels 接口获取, 例如[图 7](#figure-07) 中的用法. 有了这一抽象内存层, 就可以方便地为现有加速器加入额外内存层, 例如主机内存或 SSD, 将其扩展为新设备, 从而处理无法装入单设备内存的大型张量, 详见[第 5.4 节](#section-5-4). WELDER 的性能提升主要来自不同内存层之间的带宽差距. 因而, 只要较低层内存成为瓶颈, 且较高层内存能够容纳中间数据 tile, WELDER 就可以在速度更快的高层内存上自动流水化算子间数据传输.

<span id="table-01"></span>

![表 1. 抽象加速器中的设备接口.](./welder/table-01.png)

**表 1.** 抽象硬件加速器中的设备接口.

<span id="figure-08"></span>

![图 8. 分层 tile-graph 编译流程.](./welder/figure-08.png)

**图 8.** 分层 tile-graph 的编译流程.

为了在硬件加速器上执行分层 tile-graph, WELDER 提供 Allocate, LoadTiles, ComputeTile 和 StoreTiles 四个计算接口, 如[表 1](#table-01) 所列. [图 8](#figure-08) 展示了使用这些接口执行分层 tile-graph 的流程. 该过程从执行最底层 tile-graph, 即完整 DNN 图开始. 对每个 tile-graph, 首先在相应内存层分配必要的工作空间, 即 Allocate 接口, 并把输入 tile 加载到该空间, 即 LoadTiles. 随后, 按拓扑顺序执行子图中的所有节点. 如果当前内存层是最高层, 就直接在计算核心上执行节点, 即 ComputeTile; 否则递归调用更高层 tile-graph 的执行过程. 最后, 把当前空间中的结果 tile 存入较低内存层, 即 StoreTiles. 根据特定加速器把这些计算接口实现为代码发射器还是可执行函数调用, 这一执行流程既可作为代码生成过程, 也可作为运行时过程. 在 WELDER 中, 这些接口目前实现为代码发射器, 用于生成加速器专用计算逻辑. 执行这一递归流程后, 整个分层 tile-graph 会被展开, 自动生成包含全部必要优化的完整模型计算程序.

<span id="section-4"></span>

## 4 实现

WELDER 基于开源 DNN 编译器 TVM [Che18e], Roller [Zhu22] 和 Rammer [Ma20a] 实现. 它利用 TVM 编写内核调度, 利用 Roller 枚举高效 tile 配置, 并利用 Rammer 进行端到端图优化. WELDER 的核心机制, 包括 tile-graph, tile 传播, 调度算法和代码生成等, 共用 5.2k 行代码实现. WELDER 接收 ONNX 图作为输入, 先执行常量折叠和简单逐元素融合等常见图优化, 再把优化后的图转换成 tile-graph, 进行全局内存调度优化. WELDER 通过统一设备接口同时支持 CUDA GPU, ROCm GPU 与 GraphCore IPU ([表 1](#table-01)). 对 CUDA 和 ROCm GPU, WELDER 在全局内存 (DRAM), 共享内存与寄存器这三个内存层上调度数据 tile. 为了在 CUDA GPU 和 GraphCore IPU 上处理大型图像, 我们还加入主机内存层, 扩展其设备内存.

<span id="section-4-1"></span>

### 4.1 与硬件对齐的 Tile 搜索

**枚举高效数据 tile 大小.** WELDER 在流量代价模型中引入惩罚因子, 考虑多项可能影响数据访问效率的硬件因素. 第一, 如果存在非合并内存访问, 总内存流量会计入这些访问所需的额外事务. 例如在 CUDA GPU 中, 对连续 128 字节数据, 即一个事务, 采用合并内存访问总是更合适. 第二, 如果 tile 过大导致并行度不足, 就按照计算核心的利用率成比例增加内存流量. 最后, 如果某个 tile 配置的总内存占用超过容量, 则加入无穷大的惩罚. 为避免枚举低效候选项, WELDER 只枚举根据代价模型能最大程度降低流量的维度来搜索输出 tile, 并且只取流量最低的前 k 个候选项.

**确定对齐的计算并行度.** 在 GPU 中, 同一 threadblock 内执行的顶层 operator-tile 必须使用统一的 block 大小, 例如线程数. 为确保对齐, WELDER 首先在寄存器层强制保留足够多的并行 tile, 使其与硬件并行度对齐, 即枚举与硬件对齐的 tile. 例如在 NVIDIA V100 GPU 中, tile 数量应大于 128, 因为每个 SM 有 4 个 warp 调度器, 每个 warp 有 32 个线程. 随后, 如果所有算子的 tile 数量的最大公约数大于硬件并行度, 例如 128, 且小于最大限制, 例如 1024, 就把它作为公共 thread-block 大小; 否则, 把 block 大小设为等于硬件并行度的数值. 确定 block 大小后, 把寄存器层的所有 operator-tile 绑定到这些线程. 如果单个线程需要运行多个 tile, 则使用 TVM 的 virtual thread 绑定它们, 从而允许在所有内存 bank 上并发访问数据, 避免 bank 冲突.

**支持 TensorCore.** WELDER 利用 TensorCore 加速 CUDA GPU 上的 GEMM, BatchMatmul 和 Convolution 等算子, 其中卷积采用 implicit GEMM [Li16c]. 我们为这些算子添加注解, 指出哪些轴将绑定到 CUDA 的 Warp-Level Matrix Operations. 对于顶层 operator-tile, 我们把它们绑定到 warp 而不是线程, 以执行 MMA 操作. 此外, 枚举 tile 大小时还会加入额外约束, 例如确保线程数是 warp 大小的整数倍, 并确保每个 tile 中的 M, N, K 轴都是 MMA 操作 fragment 大小的整数倍.

<span id="section-4-2"></span>

### 4.2 代码生成与编译

WELDER 的内核生成以 TVM 为基础. 具体来说, 寄存器层的 tile 连接通过 TVM 的 compute_inline 调度原语实现. 对共享内存层连接, 我们只用 TVM 为共享内存之上的每个相连部分生成独立内核, 再执行若干额外 pass, 把这些独立内核组合成一个融合内核.

**重写加载/存储.** TVM 生成的独立内核从全局内存加载并向其存储数据. 我们在 TVM 的 lowering 流程中加入额外的 TIR [Ten20] pass, 把这些全局内存访问重写为共享内存访问. 此外, 还会加入内存栅栏以防止竞态条件, 并对缓冲区进行填充以处理 bank 冲突. 这样, 原始全局内核可以转换成设备函数, 纳入最终融合内核.

**重映射 block/thread 索引.** 部分算子无法直接连接到其他算子, 需要重映射其 blockIdx 和 threadIdx 值. BlockIdx 重映射用于 Transpose 等算子, 映射关系由其张量表达式推导. ThreadIdx 重映射用于连接 2D 与 1D thread block. 当线程间归约或 TensorCore 原语要求使用 2D thread block, 即同时使用 threadIdx.x 和 threadIdx.y, 而其他算子可能使用 1D thread block, 即只使用 threadIdx.x 时, 就需要这种映射. 只要线程总数相等, 2D thread block 就可以映射到 1D thread block.

**内存管理.** 我们统一管理全部共享内存, 包括每个独立内核中分配的内存, 以及算子间复用缓冲区. 首先, 按照拓扑执行顺序分析每个缓冲区的存活性, 把它们转换成一系列分配与释放操作. 随后使用 bestfit 算法计算每次共享内存分配的偏移量, 同时考虑数据类型与 TensorCore 操作的对齐要求, 例如按 32 字节对齐以避免地址访问未对齐.

**加快编译.** WELDER 通过并行编译与子图缓存提升编译速度. 第一, 利用不同配置相互独立的性质, WELDER 可以使用多个进程并行构建和评估各个配置. 第二, 多数 DNN 模型中常有一些子图模式反复出现. 为避免重复优化, WELDER 使用子图签名缓存每种唯一图模式. 例如在 12 层 BERT 模型中, 可以缓存第一层的优化结果, 即内核代码与实测延迟, 并在剩余 11 层中复用.

<span id="section-5"></span>

## 5 评测

<span id="section-5-1"></span>

### 5.1 实验设置

我们使用三台分别配备 NVIDIA GPU, AMD GPU 和 Graphcore IPU 的服务器评测 WELDER.

其中两台服务器配备 NVIDIA GPU. 第一台是 Azure NC24s_v3 VM, 配备 Intel Xeon E5-2690v4 CPU 和 NVIDIA Tesla V100 (16GB) GPU, 运行 Ubuntu 16.04 与 CUDA 11.0. 第二台是本地工作站, 配备 Intel(R) Xeon(R) E5-2678 v3 CPU 和 NVIDIA GeForce RTX 3090 GPU, 运行 Ubuntu 18.04 与 CUDA 11.3. AMD GPU 服务器配备 Intel Xeon CPU E5-2640 v4 CPU 和 AMD Radeon Instinct MI50 (16GB) GPU, 运行 Ubuntu 18.04 与 ROCm 5.2.3. IPU 服务器是 Azure ND40s_v3 VM, 配备 Intel Xeon Platinum 8168 CPU, 16 个 IPU 与 Poplar-sdk 3.0.

**DNN 工作负载.** WELDER 在 10 个不同类型的 DNN 模型上评测, 包括 CNN, Transformer, CNN-Transformer 和多层感知机 (MLP), 其中多数是相应任务上的当前最佳模型. [表 2](#table-02) 从模型类型, 任务和发表年份几个方面列出了这些模型. 表中所有模型都直接使用官方 PyTorch 实现, 未作修改.

<span id="table-02"></span>

![表 2. WELDER 评测的 DNN 模型.](./welder/table-02.png)

**表 2.** WELDER 评测的 DNN 模型.

**基线.** 我们把 WELDER 与多个 DNN 框架比较, 包括 PyTorch (v1.12) [Pyt17] 和 ONNXRuntime (v1.12) [Onn21], 以及 Ansor (v0.9) [Zhe20], Rammer [Ma20a] 等当前最佳 DNN 编译器. 我们还比较了 TensorRT (v8.4) [Ten17a], 这是面向 NVIDIA GPU 的厂商专用推理库. 对 Transformer 模型, 进一步比较 NVIDIA 的 FasterTransformer (v5.2) [Fas21], 这是为 Transformer 模型优化的手工 C++ 库. 另外还纳入 BladeDISC (v0.3.0) [Bla23], 它实现了最新的 AStitch [Zhe22f] 内核融合优化; 以及实现多流调度的 Nimble [Kwo20a], 作为 NVIDIA GPU 上的基线.

为了在这些基线上评测模型, 我们先在 PyTorch 中追踪模型, 并将其导出为 ONNX 格式. 随后把这一 ONNX 模型输入 WELDER, Ansor, ONNXRuntime 和 TensorRT 等框架. 对 ONNXRuntime, 使用其 CUDA execution provider, 并把图优化级别设为“ALL”, 以获得最佳性能. 对 TensorRT, 使用其 Python API 为输入 ONNX 模型构建 engine. 对 Ansor, 把总调优次数设为模型中任务数的 800×. 对所有框架, 都把输入和输出张量放在 GPU 设备内存中, 避免额外的数据移动成本. 评测时, 先执行若干预热迭代, 再让每项工作负载反复运行至少 5 秒. 由于所有情形中的变化都很小, 每个模型只报告平均速度. 所有实验中的跨模型平均性能, 例如加速比, 都按几何平均数计算.

<span id="section-5-2"></span>

### 5.2 NVIDIA GPU 上的评测

本节回答以下问题: 1) 与当前最佳 DNN 框架或编译器相比, WELDER 的性能如何? 2) TensorCore 能让 WELDER 的性能进一步提升多少? 3) WELDER 能否自动发现超出以往专家设计融合规则的新优化模式? 4) WELDER 能在多大程度上同时提升内存和计算效率? 5) WELDER 全局优化的搜索效率如何?

<span id="figure-09"></span>

![图 9. NVIDIA V100 GPU 上仅使用 SIMT Core 时的端到端模型推理性能.](./welder/figure-09.png)

**图 9.** NVIDIA V100 GPU 上的端到端模型推理性能, 仅使用 SIMT Core. 左: batch size 为 1; 右: batch size 为 64.

**端到端性能.** [图 9](#figure-09) 展示了 batch size 为 1 时 WELDER 与其他基线的性能, 结果表示为相对于最佳结果的归一化加速比. 与 DNN 框架相比, WELDER 相对 PyTorch 和 ONNXRuntime 的几何平均加速比分别为 4.29× 和 2.07×. 由于计算图具有很高的 Python 开销, PyTorch 在 batch size 为 1 的模型上表现不佳. 相比之下, ONNXRuntime 的优化更充分, 它消除了 Python 开销, 并实现了基于模式的图优化. WELDER 还比 Rammer 快 1.96×, 因为 Rammer 只能融合相互独立的并行内核, 无法通过共享内存融合有依赖关系的内核. 评测实现 AStitch 的 BladeDISC 时, 我们发现它会遇到“不支持算子”错误, 多数模型因而退回 PyTorch runtime. 对于没有发生错误的模型, 包括 BERT, MobileNet, BSRN 和 NeRF, WELDER 比 BladeDISC 快 2.70×. 对 Nimble 基线, 排除 Nimble 无法执行的模型后, WELDER 的平均加速比为 1.79×.

Ansor 通过生成高性能张量程序, 并在寄存器层采用基于规则的跨算子融合, 例如 Matmul+BiasAdd 和 Conv2D+ReLU, 提升 DNN 性能. 但它无法利用更多内存复用机会, 与 WELDER 的平均性能差距为 1.44×. 这一点在 NAFNet (1.70×) 和 BSRN (1.43×) 等 CNN 模型中尤为明显; 这些模型主要由通道数相对较小的卷积组成, WELDER 可以很好地优化它们. WELDER 在 BERT (1.71×), Swin-Transformer (1.45×) 和 ViT (1.56×) 等基于 Transformer 的模型上也显著优于 Ansor, 因为 Ansor 无法融合注意力块中的 LayerNorm 或 Softmax 等模式. 对 CNN+Transformer 模型, WELDER 同样表现良好, 在 MobileViT, Conformer 和 Restormer 上分别取得 1.64×, 1.39× 和 1.29× 加速; 这是因为 WELDER 可以覆盖这些模型中 CNN 与 Transformer 两部分的融合机会. 我们还观察到, WELDER 在 NeRF 上只略优于 Ansor (1.09×), 主要是因为计算密集型 MLP 占据了大部分计算, 没有更多优化机会.

最后, TensorRT 是 NVIDIA 提供的专用 DNN 推理库, 内含高度优化的算子. 在 BERT (1.02×) 和 Swin-T (0.97×) 等常用 Transformer 模型上, WELDER 与 TensorRT 性能相当. 原因是 TensorRT 已为一部分常用模型, 包括基于 Transformer 的模型, 纳入专家设计的融合规则和内部内核, 留给进一步优化的空间有限. 相比之下, WELDER 自动识别优化模式; 尽管计算密集型算子所依赖的内核性能较低, 仍能达到与 TensorRT 相当的性能. 值得注意的是, 内核优化与 WELDER 相互补充, 进一步优化的内核可能让 WELDER 获得更大收益. 对 NAFNet 等更新, 更多样的模型, WELDER 凭借通用性表现出比 TensorRT 更高的性能, 加速比最高达到 3.09×. 总体而言, 本系统平均比 TensorRT 快 1.47×.

[图 9](#figure-09) 还展示了更大 batch size 64 下的归一化性能. 由于输入规模很大, [表 2](#table-02) 中最后三个模型无法在大 batch size 下用 PyTorch 追踪. 在这一设置下, WELDER 仍优于所有其他基线, 相对 PyTorch, ONNXRuntime, Rammer, BladeDISC, Nimble, Ansor 和 TensorRT 的平均加速比分别为 1.83×, 1.90×, 2.1×, 1.57×, 1.49×, 1.47× 和 1.21×. 我们观察到, 与 batch size 为 1 的结果相比, 使用 CUDA 库的框架在大 batch size 下表现好得多. 因而, WELDER 相对 PyTorch, ONNXRuntime 和 TensorRT 的加速比有所缩小, 相对 Ansor 的加速比则与 batch size 为 1 时近似.

**TensorCore 性能.** TensorCore 更高的计算吞吐量会给内存访问带来更大压力. 为了理解 TensorCore 上的优化行为, 我们使用 PyTorch 把基准模型的权重与激活都转换为半精度浮点类型 (FP16), 因为 TensorCore 只支持 FP16. 除 TensorRT 外, 这一步都使用 onnxconverter_common 包 [Onn20] 中的工具完成; TensorRT 使用自己的转换器, 因为它往往能给出更好的结果.

<span id="figure-10"></span>

![图 10. NVIDIA V100 GPU 上启用 TensorCore 时的端到端模型推理性能.](./welder/figure-10.png)

**图 10.** NVIDIA V100 GPU 上的端到端模型推理性能, 启用 TensorCore. 左: batch size 为 1; 右: batch size 为 64.

[图 10](#figure-10) 比较了 batch size 为 1 和 64 时, WELDER 与其他使用 TensorCore 的框架之间的性能. 对 batch size 为 1 的 10 个测试案例, WELDER 均优于 PyTorch, ONNXRuntime, BladeDISC, Nimble, Rammer 和 TensorRT. 相对 PyTorch 的平均加速比为 7.18×, MobileNet 上最高 21.4×; 相对 ONNXRuntime 为 3.08×, Conformer 上最高 8.72×; 相对 BladeDISC 为 5.29×, MobileNet 上最高 16.9×; 相对 Nimble 为 2.72×, NeRF 上最高 5.58×; 相对 Rammer 为 2.76×, NAFNet 上最高 5.42×; 相对 TensorRT 为 1.53×, NAFNet 上最高 2.98×.

<span id="table-03"></span>

![表 3. WELDER 与 FasterTransformer 的性能.](./welder/table-03.png)

**表 3.** WELDER 与 FasterTransformer 的性能.

[图 10](#figure-10) 中其余 7 个 batch size 为 64 的测试案例里, WELDER 相对 PyTorch, ONNXRuntime, BladeDISC, Nimble, Rammer 和 TensorRT 的加速比分别为 1.98×, 2.13×, 1.97×, 3.84×, 3.45× 和 1.16×.

其中一些加速比远高于 SIMT core 上的结果. 尤其在 NeRF 模型上, WELDER 在 TensorCore 上比 TensorRT 快 2.34×, 而在 SIMT core 上的加速比只有 1.16×. 主要原因是 TensorCore 能大幅加速模型中计算密集的部分, 使其余内存密集部分的优化更加关键.

注意, 这项实验没有包含 Ansor, 因为它不支持 TensorCore. 为了公平比较, 我们关闭 WELDER 的 TensorCore 功能, 在 SIMT core 上评测这些 FP16 模型, 并在[图 11](#figure-11) 中与 Ansor 比较. 与 FP32 结果相比, 加速比略有提高, 平均为 1.74×, 最高达到 2.82×.

<span id="figure-11"></span>

![图 11. 在 FP16 且不使用 TensorCore 时与 Ansor 比较.](./welder/figure-11.png)

**图 11.** 在 FP16 且不使用 TensorCore 时与 Ansor 比较.

**另一款 NVIDIA GPU 上的性能.** 我们还在广泛使用的 RTX-3090 GPU 上进行评测, 它采用不同的 Ampere 架构. 与 V100 相比, RTX-3090 有多项新特性, 包括内存加载与 TensorCore 指令上的改进, 以及不同数量的流式多处理器 (SM). 为简洁起见, 在 RTX-3090 上只比较 WELDER 与 TensorRT, 因为 TensorRT 在 NVIDIA GPU 上始终优于其他基线. [图 12](#figure-12) 的结果表明, 按全部 34 个测试案例的几何平均数计算, WELDER 比 TensorRT 快 1.40×. 这一加速比与 V100 上的 1.36× 接近, 说明 WELDER 能适应不同 GPU 架构.

<span id="figure-12"></span>

![图 12. 在 NVIDIA RTX-3090 上与 TensorRT 比较.](./welder/figure-12.png)

**图 12.** 在 NVIDIA RTX-3090 上与 TensorRT 比较.

**自动发现的模式.** 在 10 个模型的全部 34 个编译测试案例中, 按唯一算子类型统计, WELDER 自动发现约 300 个不同的融合子图. 其中 89 种模式至少包含两个基于归约的算子, 无法由 Ansor 中简单的逐元素融合规则覆盖. 据我们所知, 其中许多是基于手工规则或自动融合优化均未探索过的非常规融合模式. [图 4](#figure-04) 给出了两个例子, 它们把多个 Convolution 或 MatMul, 即 Dot 算子与其他内存密集型算子融合为单个内核. 每种模式融合的算子数从 2 到 48 不等; 与 Ansor 所用的基础融合方法相比, 平均加速比为 1.87×, 最高达到 5.4×. 最常见的模式在所有模型中共使用 191 次.

<span id="table-04"></span>

![表 4. WELDER 发现的融合模式示例.](./welder/table-04.png)

**表 4.** WELDER 发现的融合模式示例.

这种通用融合能力常使 WELDER 优于专家优化的模型专用实现. 例如, FasterTransformer [Fas21] 是 NVIDIA 为 Transformer 模型手工优化的基准, 同时支持 BiasAdd+Transpose 等逐元素融合, 以及 Layernorm+Softmax 等非逐元素融合. WELDER 可以自动融合所有这些模式. 此外, 当序列不长时, WELDER 还能把 Q*K 与注意力块中随后的 Softmax 进一步融合. 例如, BERT 的序列长度为 128, 两者会被融合; Conformer 的序列长度为 512, 则不会融合. 这一决定由 WELDER 自动作出.

对于 FasterTransformer 支持的三个模型, [表 3](#table-03) 比较了它与 WELDER 的性能. 总体上, WELDER 平均比 FasterTransformer 快 1.11×, ViT 上最高达到 1.73×. 根据性能分析数据, ViT 在 batch size 为 1 时的显著加速来自一个形状非常规的卷积算子, 其 stride 和 kernel size 都是 32, 即 ViT 的 patch size. 对这个算子, WELDER 生成的内核快 4.4x. 这体现了 WELDER 处理新算子形状或模型模式的适应能力.

另一个例子是 NeRF, 这是一种常用的 3D 场景生成模型, 通常实现为 7 层 MLP. 为了充分利用 GPU, 领域专家往往需要从头实现这种模型, 才能获得更好的融合结果, 例如[Mul21a] 中的 fully-fused MLP. WELDER 可以自动把这个 7 层 MLP 融合成单个 GPU 内核. 生成的内核在前 6 层使用 TensorCore, 输出层使用 SIMT Core, 所有中间结果都存于共享内存. 我们观察到, 这种自动融合可达到与[Mul21a] 所报告数值相近的加速, 超过 5×; 由于其代码 [Mul21] 不支持 V100 GPU, 我们无法直接评测.

最后, 对 NAFNet, BSRN 和 MobileNet 等 CNN 模型, WELDER 可以把不同类型的卷积与 Pooling, PixelShuffle 等其他算子融合. 例如在 NAFNet 中, 本系统能把连续的 pointwise convolution 与其间的 normalization 操作一并融合. 另一个有意思的模式出现在包含多层 separable convolution 的模型中, 每层由 depthwise convolution (DWConv) 和 pointwise convolution (PWConv) 两个操作组成. WELDER 可以根据算子配置确定这两类算子的最优融合顺序. 例如, 在上层, feature map 较大而通道数较少, WELDER 会构造 DWConv+PWConv 融合组, 因为在共享内存中缓存完整 feature map 更合适. 到下层时, feature map 变小, WELDER 会构造 PWConv+DWConv 融合组, 为 DWConv 缓存一整个通道以供复用.

<span id="figure-13"></span>

![图 13. 所选模型的延迟, 内核数, 内存事务和中间结果大小.](./welder/figure-13.png)

**图 13.** 3 个所选模型的延迟, GPU 内核数量, 已执行全局内存事务数与中间结果大小 (IRS), FP32, batch size 64.

**消融与敏感性研究.** 为展示 WELDER 全局内存优化的收益, 我们建立两个变体: “WELDER-none”关闭所有算子间 tile 连接, 只搜索算子内调度; “WELDER-base”只启用寄存器层的算子间 tile 连接. 实验还纳入 Ansor, 因为它与本系统一样采用代码生成方法. 如[图 13](#figure-13) 所示, 启用寄存器层 tile 连接后, WELDER-base 相比 WELDER-none 的延迟平均降低 52%, 即加速 2.08×, 内核启动数降低 67%, 全局内存事务降低 52%, 中间结果大小 (IRS) 降低 66%. WELDER-base 的指标与 Ansor 相近, 说明通用的 tile 内存调度相比 Ansor 基于规则的融合同样高效. 在共享内存层进一步启用 tile 连接后, WELDER 相比 WELDER-base 的延迟平均再降低 29%, 最高加速 1.82x, 内核启动数降低 60%, 事务数降低 25%, IRS 降低 65%. 内存事务的降幅小于 IRS, 是因为融合无法优化模型权重部分的内存访问.

此外, 我们改变三个所选模型的输入大小进行敏感性研究: BERT 使用 128-512 的文本长度, Conformer 使用 128-512 个音频帧, NAFNet 使用 256x256-1024x1024 的图像输入. [图 14](#figure-14) 的结果显示, 使用较大图像时, NAFNet 的融合收益显著提高; 另外两个基于 Transformer 的模型则相反, 收益有所下降. 这是因为 Transformer 模型的计算量随输入序列长度呈二次增长, 内存密集程度因而降低.

<span id="figure-14"></span>

![图 14. 改变输入大小的敏感性研究.](./welder/figure-14.png)

**图 14.** 改变输入大小, 与 WELDER-base 比较.

**编译时间.** [表 5](#table-05) 比较了 WELDER 与 Ansor 的编译时间. Ansor 是基于搜索的编译器, 需要大量调优与性能分析试验. 其他基线会直接调用库内核, 不需要额外花费时间调优和生成代码, 因此没有纳入比较. 结果显示, WELDER 的端到端编译速度比 Ansor 快一个数量级以上. 原因是 Ansor 为所有算子生成非常大的搜索空间, 并通过基于机器学习的调优隐式优化数据复用. 这通常需要大量调优试验, 本评测中每个算子为 800 次, 并且还有即时训练代价模型的额外开销. 相比之下, WELDER 通过分层调度策略分解优化空间, 再使用解析式代价模型估算流量成本, 搜索高效分块配置. 因而, WELDER 所需的调优试验数显著少于 Ansor, 本评测中每个子图为 20 次.

<span id="table-05"></span>

![表 5. Ansor 与 WELDER 的编译时间.](./welder/table-05.png)

**表 5.** Ansor 与 WELDER 的编译时间.

**计算密集型模型上的性能.** ResNet [He16c], VGG [Sim14a] 和 UNet [Ron15] 等传统模型通常以卷积等大型算子为主. 对这些计算密集型模型, 尽管 WELDER 主要关注内存访问优化, 其性能大多仍能与 TensorRT 等当前最佳基线相当. 原因是即使在较高内存层连接 tile 的机会很少, WELDER 仍能使用类似 Ansor [Zhe20] 或 Roller [Zhu22] 的多层分块抽象, 生成高性能的单个算子. 不过, cuDNN [Cud23] 等现有库对部分卷积算子采用了优化过的数值算法, 例如 winograd [Lav16], 而这些算法很难从张量表达式自动推导. 如果没有额外的内存优化空间来弥补这一差距, WELDER 的性能可能低于 TensorRT. 例如, [表 6](#table-06) 比较了 WELDER, Ansor 和 TensorRT 在四个此类模型上的性能. 对 ResNet, 两个系统的性能相当, 因为该模型中多数卷积算子采用 DirectConv 算法时性能更好, Ansor 与 WELDER 都支持该算法, 而不是 winograd. 对 UNet 和 VGG16, 占主导的卷积算子在 TensorRT 中大多使用 winograd 实现, 同时 WELDER 也没有更多融合机会, 因而 TensorRT 性能更高. 这项问题与 WELDER 的优化正交, 我们把通过重写张量表达式支持 winograd 算法留作未来工作.

<span id="table-06"></span>

![表 6. 计算密集型模型上的性能.](./welder/table-06.png)

**表 6.** 计算密集型模型上的性能.

<span id="section-5-3"></span>

### 5.3 AMD ROCm GPU 上的评测

<span id="figure-15"></span>

![图 15. AMD ROCm MI50 GPU 上的端到端模型推理性能.](./welder/figure-15.png)

**图 15.** AMD ROCm MI50 GPU 上的端到端模型推理性能. 左: batch size 为 1; 右: batch size 为 64.

我们把 WELDER 的性能与 PyTorch, ONNXRuntime 和 Ansor 比较, 评测其在 AMD ROCm GPU 上的效率. TensorRT 和 AStitch 只支持 NVIDIA GPU, 因而未纳入. [图 15](#figure-15) 展示了 10 个 DNN 模型的端到端性能. 与 PyTorch, ONNXRuntime 和 Rammer 相比, WELDER 平均分别快 2.62×, 1.71× 和 2.14×; 相对 Ansor 的平均性能提升为 1.53×. [图 15](#figure-15) 还给出了更大 batch size 64 下的性能比较, 此时 WELDER 相对 PyTorch, ONNXRuntime, Rammer 和 Ansor 的平均加速比分别为 1.69×, 1.23×, 1.86× 和 1.47×. 由于部分 CNN 模型无法在 ONNXRuntime 上执行, 我们排除了这些结果. WELDER 在 MI50 上的加速比略小于 V100. 这是因为按照官方数据表, MI50 的峰值 FLOPS 低于 V100, 峰值带宽却更高. 这种差异使 MI50 上的工作负载更加偏向计算密集, 留给内存访问优化的机会较少.

<span id="section-5-4"></span>

### 5.4 使用主机内存扩展规模

借助抽象设备层, WELDER 可以扩展内存层次, 支持大型 DNN 任务. 例如, 当 UNet 或 VGG16 等经典 CNN 模型用于处理高分辨率医学图像 [Sid21] 时, 部分层中的单个张量常常大到无法装入 GPU 内存. 在这些场景中, SwapAdvisor [Hua20] 或 Capuchin [Pen20] 等基于张量的内存换入换出优化技术, 可能因为张量粒度太大而无效. WELDER 通过全局流量优化, 在扩展后的内存层次上生成基于 tile 的执行计划. 这样, 系统可以从主机内存加载一个数据 tile, 在设备内存中复用数据并计算多个相连的 operator-tile, 再把结果存回主机, 就像在单个设备上处理一样. 为评估这种调度方法的效率, 我们把 WELDER 与一个只关闭设备内存层数据复用的变体进行比较.

<span id="table-07"></span>

![表 7. 利用主机内存扩展大型 DNN 模型.](./welder/table-07.png)

**表 7.** 利用主机内存扩展大型 DNN 模型.

**扩展 GPU.** 作为初步实验, [表 7](#table-07) 给出了通过主机内存层扩展 GPU 内存后, WELDER 在大型图像数据上扩展 UNet 与 VGG16 的性能. 结果表明, 启用设备内存层 tile 连接后, WELDER 在两个模型上分别获得 2.63× 和 1.89× 的平均加速, 主机内存传输量也分别减少 3.11× 和 2.90×. 内存流量降幅高于实际加速比, 是因为我们实现了双缓冲, 并结合 pinned memory 和 CUDA stream, 让一部分内存传输与计算重叠.

**扩展 GraphCore IPU.** 我们还初步评测了 WELDER 在 Graphcore IPU [Ipu23] 上的扩展能力. 这是一种架构不同于 NVIDIA 与 AMD GPU 的 DNN 加速器. IPU 配有大规模并行 MIMD 处理器, 但设备内存相对较小, 只有 300MB, 连处理中等规模任务都颇具挑战. 我们在 IPU 上对两个模型采用相同的 tile 调度, 并把输入图像大小设为 2048*2048, 以适应 IPU 的内存容量. [表 7](#table-07) 的结果显示, WELDER 优化在两个模型上分别取得 3.63× 和 3.09× 的平均加速. 这一提升比例高于 GPU, 主要因为 IPU 内存有限, 我们关闭了双缓冲优化.

<span id="section-6"></span>

## 6 讨论

WELDER 的设计与实现主要面向静态模型. 对动态模型执行, 有两种实用处理方式. 第一, 可以通过 JIT 编译把动态图转换为静态子图, 例如 PyTorch JIT compile; 这已经成为 PyTorch 2.0 的标准做法. 随后, WELDER 可以集中优化静态子图, 它们通常是计算中的主导部分. 第二, 即使张量形状是动态的, 每个算子内部的 tile 也可以静态确定. 因而, WELDER 可以生成静态的 tile 级融合计划, 只让并行任务数量由输入张量形状决定.

<span id="section-7"></span>

## 7 相关工作

算子融合等编译器优化是 DNN 计算中常用的技术, 可以减少内核启动开销, 并提高较快内存中的数据局部性. TVM [Che18e], Ansor [Zhe20], XLA [Xla17] 和 DNNfusion [Niu21] 等编译器都支持寄存器层算子融合. 其他编译器试图把算子进一步融合到共享内存层, 这类方法要么依赖针对一组已知算子类型的融合规则, 例如 AStitch [Zhe22f], Apollo [Zha22h] 和 DeepCuts [Jun21]; 要么依赖针对少数算子组合的特定模板, 例如 Bolt [Xin22]. TensorRT [Ten17a] 和 ONNXRuntime [Onn21] 等专用 DNN runtime 已经为常用模型, 如 Transformer 模型中的一些常见模式加入专家设计的融合规则. 相比之下, WELDER 适用于以张量表达式实现的通用算子, 不对算子类型作假设, 并自动确定最佳融合内存层. 这是因为算子的资源使用特征, 即内存密集还是计算密集, 常常取决于形状, 进而影响融合决定. Rammer [Ma20a], HFuse [Li22e] 和 Nimble [Kwo20a] 等系统通过横向融合, 或用多流与 CUDA graph 调度并行任务, 更充分地利用硬件并行度并减少内核启动. WELDER 建立在 Rammer 之上, 进一步探索一种与这些系统互补的优化, 即通过纵向融合进行全局内存优化, 从而进一步加速内存密集型模型.

Ansor [Zhe20] 和 Roller [Zhu22] 是具有代表性的张量编译器, 分别通过循环优化或分块优化聚焦算子内优化. Roller [Zhu22] 和 Triton [Til19] 也使用 tile 概念优化内核性能, 例如算子内数据复用. WELDER 与它们互补, 统一优化算子内和算子间内存访问. WELDER 把 Roller 的 tile 概念推广为 tile-graph 抽象, 显式给出全局 tile 级调度空间, 并提出一种覆盖该全局空间与显式内存层次的高效调度机制.

一些工作针对特定模型类型中的特定模式, 采用更激进的算子融合, 例如 NeRF 模型的 fully-fused MLP [Mul21a], CNN 模型的手工融合内核 [Wan20h], 以及 Transformer 模型的注意力融合 [Fas21, Fan21b]. 评测表明, WELDER 可以自动实现其中大多数融合, 甚至生成新的融合模式, 进一步帮助优化.

内核融合技术也用于传统图像处理 [Qia18, Qia19] 或 HPC [Wah14] 领域. 这些工作通常针对自身工作负载采用领域专用融合规则. WELDER 关注 DNN 工作负载, 但也适用于以张量表达式表示的通用算子, 因而可能帮助其他领域中可以用张量表达式实现的工作负载.

<span id="section-8"></span>

## 8 结论

观察到现代 DNN 模型越来越受内存限制后, 我们提出 WELDER, 一种基于新型 tile-graph 抽象优化执行效率的 DNN 编译器. WELDER 可以在多层内存层次上全局优化高效的算子内与算子间数据复用. WELDER 首次把所有常见算子融合统一到一个框架中, 因而发现了 89 种非常规融合模式, 其中最大的一种把 48 个算子融合成单个内核. 凭借这种通用性, WELDER 显著优于当前最佳基线. 更重要的是, WELDER 提供了一种系统化方法, 可以利用未来 AI 加速器中内存层次的发展趋势, 例如容量更大, 连接更充分的片上内存.

## 致谢

我们感谢匿名审稿人和论文 shepherd Byung-Gon Chun 教授提出的广泛建议. 本工作得到中国国家重点研发计划 (No. 2021ZD0110202) 的部分支持.

<span id="section-9"></span>

## 9 Artifact 附录

### 摘要

WELDER 通过新的 tile-graph 抽象提供端到端 DNN 模型编译. 本 artifact 复现 NVIDIA V100 GPU 评测中的主要结果.

<span id="section-9-1"></span>

### 9.1 范围

本 artifact 将验证以下主张:

- 端到端模型性能, 通过复现[图 9](#figure-09), [图 10](#figure-10), [图 11](#figure-11), [表 3](#table-03) 和[表 6](#table-06) 的实验验证.

- [图 1](#figure-01) 与[图 2](#figure-02) 中的动机实验.

- [图 13](#figure-13) 中的消融研究.

- [表 5](#table-05) 中的编译时间.

- [表 7](#table-07) 中的 GPU stale out 实验.

<span id="section-9-2"></span>

### 9.2 内容

本 artifact 包含实现 WELDER 的全部源代码. 我们提供 Dockerfile 用于配置环境. 对上述每幅图和每张表, 都提供用于复现结果的脚本. 完整复现结果需要编译超过 50 个模型测试案例, 会花费很长时间, 尤其是 Ansor 基线. 因此, 我们还提供针对 NVIDIA V100 GPU 预编译的日志和模型. 详情请参阅仓库中的 README.md 文件.

<span id="section-9-3"></span>

### 9.3 托管位置

Artifact 托管在 GitHub 仓库[+artifact] . 请使用 git 克隆该仓库, 并 checkout 到 osdi2023welder 分支.

<span id="section-9-4"></span>

### 9.4 要求

本 artifact 需要 NVIDIA V100 GPU, CUDA 驱动须支持 11.0 以上版本的 CUDA runtime.

[+internship]: 本工作在 Microsoft Research 实习期间完成.

[+artifact]: <https://github.com/microsoft/nnfusion/tree/osdi2023welder>
