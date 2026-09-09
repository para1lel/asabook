---
title: 'Ladder: Hardware-Aware Tensor Transformation'
createTime: 2026/09/10 00:31:20
permalink: /papers/ladder/
---

> [Lei Wang](https://x.com/Lei_Wang_1999) [+intern], [Lingxiao Ma](https://xysmlx.github.io/), [Shijie Cao](https://caoshijie0501.github.io/), [Quanlu Zhang](https://dblp.org/pid/165/8284), [Jilong Xue](https://dblp.org/pid/06/10336.html), [Yining Shi](https://dblp.org/pid/161/3927-1.html) [+intern], [Ningxin Zheng](https://dblp.org/pid/234/5381), [Ziming Miao](https://dblp.org/pid/216/9568.html), [Fan Yang](https://fanyangcs.github.io/), [Ting Cao](https://www.microsoft.com/en-us/research/people/ticao/), [Yuqing Yang](https://dblp.org/pid/91/9064-1.html), [Mao Yang](https://www.microsoft.com/en-us/research/people/maoyang/). 论文发表于第 18 届 USENIX 操作系统设计与实现研讨会 (OSDI 24), 会议于 2024 年 7 月 10–12 日在美国加州圣克拉拉举行, 论文见第 307–323 页. [Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation](https://www.usenix.org/conference/osdi24/presentation/wang-lei). <a href="/paper/ladder.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源码; 本阅读版以正式发表的 PDF 为准, 原始 PDF 仍是精确版式和参考文献的权威来源.

[+intern]: 本工作在 Microsoft Research 实习期间完成.

## 摘要

为了提升深度学习模型的性能, 人们开始利用深度学习对误差的鲁棒性来支持低精度计算, 由此带来了范式转变. 尽管新的低精度数据类型与优化方法不断出现, 现有软硬件对这些演进中的数据类型支持不足且效率不高, 因而很难通过低精度计算获得实际的性能收益.

本文提出 LADDER, 一种用于弥合不断演进的自定义数据类型与现有硬件固定精度格式之间差距的新编译器. LADDER 利用通用类型系统 tType 和扩展的张量表达式, 将深度神经网络 (DNN) 计算转换为经过优化的计算流水线, 让自定义数据类型成为一等公民, 并暴露出一个可高效处理数据存储, 访问与类型转换的优化空间. LADDER 使用一组新的张量调度原语和硬件感知优化策略来搜索复杂的转换空间, 从而在不同内存层级与 DNN 算子之间取得最佳性能. 评估结果表明, LADDER 可以系统地支持多种低比特精度的自定义数据类型, 无需修改硬件即可显著提高现代加速器上的 DNN 计算性能. 这项工作让模型设计者能够探索数据类型优化, 也为硬件厂商扩展多种精度格式的支持提供了灵活方案.

<span id="section-1"></span>

## 1 引言

随着深度学习模型规模不断扩大 [Bro20, Dev18, Kap20], GPU 等硬件加速器需要提供更强的计算性能. 深度学习本身对误差较为鲁棒, 因而可以使用低精度算术; 这与科学计算等通常需要 float64 高精度的传统负载不同. 顺应这一趋势, 新一代前沿加速器正逐步加入 32 位, 16 位乃至 8 位浮点等更多低精度计算单元. 与此同时, 模型开发者也在积极研究混合精度等自定义低精度数据类型, 以便在模型精度与训练效率之间取得理想平衡. 到模型部署阶段, 计算还可以转成更紧凑的数据表示以追求极致效率, 例如 LLM 中的 2 位定点精度 [Che24b], 或多个数值共用缩放因子的分组类型 [Dar23].

然而, 硬件加速器很难跟上多样且快速变化的数据精度格式需求, 也就是自定义数据类型. 芯片面积有限, 硬件成本很高, 每种加速器只能集成少数针对标准数据类型的计算单元. 即便是近期获得硬件支持的低精度类型, 例如位宽不足 16 位的类型, 现有软件通常也效率不佳, 因为细粒度低比特数据访问很难与粗粒度内存系统对齐. 例如 NVIDIA GPU 的共享内存 bank 宽度为 4 字节, 直接加载或存储 8 位数据元素很容易浪费带宽. 这往往需要把多个数值打包在一起等并不简单的优化, 才能配合不同内存层级的特性. 因此, 要为所有新数据类型, 算子和形状组合优化内核库十分困难. 例如, NVIDIA GPU 上高度优化的 CUTLASS 库在 INT8 矩阵乘法中也只能达到 422 TFLOPS, 即 68% 的利用率. 对这些新型自定义数据类型支持不足且效率不高, 严重阻碍了模型与加速器两方面的创新.

为解决这些问题, 我们得到两点观察. 第一, 即使硬件加速器没有针对自定义数据类型的计算指令, 其内存系统仍可把数据转换为固定比特宽度的不透明数据块, 从而存储任意数据类型. 第二, 大多数自定义数据类型都能无损转换为现有硬件计算单元支持的, 更宽的标准数据类型. 例如, NF4 张量可以先转换数据类型, 再用 FP16 或 FP32 运算完成计算. 这些观察启发我们把数据存储与计算分开, 以一种通用方式支持所有自定义数据类型: 张量以自定义数据类型存储和传输, 再通过类型转换以标准数据类型计算. 现代 DNN 模型往往受内存制约, 最新硬件也面临内存墙问题 [Shi23a]; 这种方法可以减少内存流量和占用, 有效发挥低比特数据类型的性能优势, 因而愈发重要.

不过, 在现有加速器上高效支持面向通用自定义数据类型的这类计算流水线并不容易. 典型的张量计算流水线要从 DRAM, L2 cache, 共享内存和寄存器等多个内存层级加载数据. 首先, 在不同层级转换张量数据类型会显著影响内存占用, 数据访问流量和硬件开销等性能因素, 优化起来很复杂. 例如, 在寄存器中把低比特数据块转换为更高位宽的类型可能导致寄存器溢出, 使性能骤降. 其次, 涉及不同数据类型的流水线通常要采用不同的数据布局优化来适配内存系统, 例如与 memory bank 对齐, 从而尽量提高访问吞吐量. 现有的内存访问 swizzling 等优化 [Nvi24a] 大多只针对少数特定数据类型, 难以推广.

为此, 我们提出 LADDER, 一种面向通用自定义数据类型高效深度学习计算的编译器. 为了便于实现 MXFP 这类快速演进的分块数据类型, LADDER 首先引入名为 tType 的通用类型系统. tType 本质上是一种 tile 级数据类型, 可通过明确指定类型位宽, 元素形状和类型转换函数来定义各种常见自定义类型. 在 tType 基础上, LADDER 扩展了用于表示 DNN 算子的张量表达式, 使每个张量都能原生标注 tType. 这样, LADDER 便可系统地把采用自定义数据类型的 DNN 计算转换为标准计算流水线.

为了优化自定义数据的存储, 访问和类型转换流水线, 我们发现流水线中的张量存储与访问可以转换成多种逻辑等价的格式, 但它们对性能的影响差别很大. 例如, 子张量可以采用行优先, 列优先, 分块或自定义布局来存储, 可以填充到特定形状以匹配计算指令, 上层内存也可以按不同粒度访问它.

这些因素都会显著影响总体性能. 为支持此类转换, LADDER 引入 slice, map, pad 和 convert 等张量调度原语, 用它们把默认计算流水线转换为优化后的流水线.

要为特定计算流水线得到最佳张量转换, 必须统筹内存层级之间和算子之间的优化. 例如, 一种数据布局可以传播到相邻算子, 避免显式布局转换的开销. 某个内存层级中的数据布局还需要同时考虑该层的特性和上层的访问模式. 跨层级与跨算子优化共同形成了庞大的优化空间. LADDER 用分层硬件感知策略优化这个转换空间: 下层内存把偏好的数据访问粒度作为提示, 上层则让计算粒度与之对齐, 决定最佳粒度. 因此, LADDER 先把 DNN 计算建模成 tile 级数据流图, 再用粒度感知调度策略优化转换调度.

LADDER 基于 TVM [Che18], Roller [Zhu22] 和 Welder [Shi23a] 实现. 我们已经开源 LADDER [+source]. LADDER 的 DNN 算子编译能力也以 BitBLAS [+bitblas] 的形式发布; 该库可集成到现有 DNN 和 LLM 框架中, 为现有深度学习生态提供高效低精度计算能力. 在 NVIDIA A100, V100, RTX A6000 和 AMD Instinct MI250 GPU 上进行的 DNN 推理评估表明, LADDER 在硬件原生支持的数据类型上胜过先进 DNN 编译器, 对 GPU 不支持的自定义数据类型也能高效执行, 最高加速 14.6×. 因而, LADDER 是第一个在现代硬件加速器上系统支持通用低比特自定义数据类型 DNN 计算的系统. 它让模型设计者可以借助实际性能反馈探索更灵活的数据类型优化, 也让硬件厂商无需修改硬件就能支持更多类型.

[+source]: <https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact>
[+bitblas]: <https://github.com/microsoft/BitBLAS>

<span id="section-2"></span>

## 2 背景与动机

<span id="section-2-1"></span>

### 2.1 深度学习中的精度需求

随着大语言模型 (LLM) 等深度学习模型的规模不断扩大, 为提高计算效率并节省内存, 使用更低位宽和混合精度计算的需求也随之增长. 本节介绍深度学习中的一些新数据类型需求.

**低比特数值精度.** FP32 (32 位浮点数) 一直是深度学习模型常用的数据表示. 但近期实践表明, FP32 的高精度并非总有必要. 更低精度可以保持同等效果, 同时降低成本. 自动混合精度 (AMP) 训练中的 FP16/BF16 计算就是这种精度转变的重要实例 [Mic18]. 更进一步, Transformer Engine [Mic22] 和 MS-AMP [Pen23e] 已开始对权重, 梯度甚至优化器张量使用 FP8, 继续压低深度学习的计算精度. 推理时, 模型经常被量化到 8 位或 4 位等更低精度 [Det22, Fra22, Xia23]. 最新研究还在进一步突破这个界限, 尝试把权重量化到 2 位甚至 1 位 [Che24b, Wan23]. 这主要是因为预训练权重存在冗余, 而计算大多是前向传播. [图 1](#figure-01) 展示了深度学习模型中的多种数据格式, 可以清楚看到从高精度格式转向低比特格式的趋势.

<span id="figure-01"></span>

![深度学习训练与推理中的多种窄精度数据类型](./ladder/figure-01.png)

**图 1.** 深度学习训练与推理中的多种窄精度数据类型.

**分组精度缩放.** 为提高低精度深度学习模型的准确性和鲁棒性, 常见做法是用缩放因子重新缩放数值, 更准确地表示数据分布. 传统方法通常使用张量级或通道级缩放因子. 分组缩放的粒度更细, 能更好地捕捉子张量或分组的分布, 因而效果更好. 例如, 训练后量化 (PTQ) [Fra22] 通常采用 128 或 64 的组大小, 每组使用 FP16 缩放. OCP-MXFP [Dar23] 则让一组 32 个元素共用一个 8 位缩放值.

**混合精度运算.** 在数据量化中, 不同张量对低位量化的敏感程度不同, 因而会产生混合精度运算. 例如, 混合精度训练会组合使用 FP32, FP16 和 FP8 等较高位宽与较低位宽的张量. 这种精度搭配在计算效率与数值精度之间取得平衡, 从而优化性能. 类似地, 量化 LLM 时, 更容易量化的权重可以用更低位宽表示; 激活值的量化更困难, 需要更高位宽. 这种差异带来了 W4A16 (权重为 4 位数据类型, 激活为 16 位数据类型), W2A16 和 W1A8 等混合精度运算 [Che24b, Fra22, Wan23].

<span id="section-2-2"></span>

### 2.2 GPU 精度支持不足

GPU 等硬件加速器一直在适应深度学习不断变化的数据类型需求. NVIDIA Fermi 等早期 GPU 支持 FP32, FP64 等标准类型. 深度学习负载兴起后, Pascal 架构加入了 FP16 等低精度格式. Turing 架构进一步加入用于推理的 INT4 和 INT8. Ampere 架构随后支持 BF16, 在性能收益与机器学习所需的数值范围之间取得平衡. 最新的 Hopper 架构延续这一趋势, 加入 FP8 支持, 体现出通过调整精度与性能之间的取舍来追求效率. 这一演进说明 GPU 能处理的计算负载越来越多样. 然而, 硬件通常落后于算法或模型需求. 遇到不支持的数据类型时, 必须将其转换或模拟为硬件支持的更高精度类型, 这可能带来明显的性能问题与低效率.

<span id="section-2-3"></span>

### 2.3 低精度计算的低效率

由于数据访问粒度很细, 且使用 TensorCore 等特殊硬件单元, 低精度计算尤其难以优化. 我们在 NVIDIA V100, A100 和 AMD MI250 三种新型 GPU 上, 使用最新软件库与编译器测试了不同精度下的标准矩阵乘法基准, 结果见[表 1](#table-01). 由此得到三点观察. 第一, 低精度计算的硬件利用率普遍很低, 平均不足 60%. 即使是当前深度学习负载中最主要的 FP16 精度, 平均利用率也只有约 60%. 第二, 一些硬件已经支持的精度并未得到软件的良好支持. 例如 A100 和 MI250 都支持 INT8, 但大多数现有深度学习编译器无法在这些 GPU 上执行 INT8 计算. 第三, 硬件很难及时满足新的精度需求. 例如 FP8 直到下一代 NVIDIA Hopper 架构才得到支持. F16 × NF4 等混合精度计算则不受任何一种最新 GPU 支持.

<span id="table-01"></span>

![不同数据类型, GPU, 库与编译器上的矩阵乘法利用率](./ladder/table-01.png)

**表 1.** MatMul $[M,N]=[M,K]\times[N,K]$, 其中 $M,N,K=16384$. "X" 表示 tensor core 或 matrix core 不支持.

<span id="section-2-4"></span>

### 2.4 我们的洞察

我们以 FP16×INT8 混合精度矩阵乘法为例说明关键洞察, 如[图 2](#figure-02) 所示. DNN 算子通常实现为计算流水线, 它不断从输入张量中取出小数据 tile, 经多个内存层级加载到顶层计算核心. 每个内存层级通常有自己偏好的最小访问粒度, 例如 L1 层为 8 字节事务. 一些新型 GPU 还提供高效数据加载指令, 每次加载一个二维 tile, 例如 `ldmatrix.2x2.f16` 加载 2×2 tile. 数据 tile 通常存储在带步长的内存空间中, 访问容易与事务长度或指令形状不对齐, 造成带宽利用率偏低. 左图中, 两个张量从 L1 发起的每次内存访问都只达到一半利用率. 此外, 硬件没有 FP16×INT8 计算指令, 即使能把相应数据加载到寄存器, 也无法完成运算. 我们发现, 可以根据数据类型位宽, 内存事务长度和指令形状, 把张量布局转换为经过优化的形式, 从而绕过对齐问题. 右图把每个 2×2 tile 连续存放在 L1 中, 使上层加载指令充分利用带宽. 计算指令只支持 FP16, 因而可在数据从 L2 加载到 L1 时把第二个张量从 INT8 转为 FP16. 最终, L2 到 L1 的加载借助低比特类型减少流量, L1 到 L0 的加载通过事务对齐充分利用内存带宽, 运算再通过类型转换在硬件计算单元中加速. 这个例子说明, 即便硬件不支持某种自定义数据类型, 也能通过精心设计的布局与数据类型转换来调度和优化相应 DNN 计算.

<span id="figure-02"></span>

![张量转换前后的混合精度矩阵乘法流水线](./ladder/figure-02.png)

**图 2.** MatMul: $C_{\mathrm{FP16}}[2,2]=A_{\mathrm{FP16}}[2,4]\times B_{\mathrm{INT8}}[2,4]$.

<span id="section-3"></span>

## 3 LADDER 设计

[第 2 节](#section-2) 的观察促成了 LADDER. 它是一种把数据类型作为一等公民, 并通过张量转换来高效支持自定义数据类型 DNN 计算的编译器. [图 3](#figure-03) 给出了系统架构.

<span id="figure-03"></span>

![LADDER 系统概览](./ladder/figure-03.png)

**图 3.** LADDER 系统概览.

LADDER 的核心是 TypedTile (tTile) 抽象, 它为基于 tile 的张量抽象加入了数据类型 (即 tType, 见[第 3.1 节](#section-3-1)). 具体而言, 算法设计者可以把常见数据类型 (如 FP16) 或自定义数据类型 (如 MXFP8, NF4) 定义成 tType, 并在该数据类型上定义 DNN 计算. 随后, LADDER 以 DNN 模型为输入, 将其转换为基于 tTile 的数据流图 (即 tTile-graph), 其中的算子被定义为基于 tTile 的计算任务 (即 tTile-operator, 见[第 3.1 节](#section-3-1)).

此外, LADDER 把硬件加速器抽象为多层层次结构, 每一层的要求用一个 tTile 表示 (即 tTile-device, 见[第 3.1 节](#section-3-1)). tTile-device 明确描述各层支持的数据类型, 事务大小等要求. 只要把 tTile-graph 中的 tTile 与 tTile-device 对齐, 就能在硬件加速器上执行 tTile-graph 所表示的 DNN 计算.

给定初始 tTile-graph 和硬件规格后, LADDER 会把 DNN 模型编译为加速器上的高效执行方案. 为在 tTile-device 上调度 tTile-graph 并满足硬件层次结构的要求, LADDER 将调度机制与调度策略分开. 在机制方面, LADDER 提出四种 tTile 转换原语 (见[第 3.2 节](#section-3-2)). 调度器随后把初始 tTile-graph 变为能够细粒度控制 tTile 配置, 转换及其在硬件层次中放置位置的 tTile-graph. tTile 抽象扩大了 DNN 计算的调度空间, 并在内存占用效率与延迟效率之间引入新的取舍. 在策略方面, LADDER 根据观察使用启发式方法, 提供面向延迟优化的分层硬件感知策略 (见[第 3.3 节](#section-3-3)). 最后, 经过调度的 tTile-graph 被降低为硬件指令执行.

<span id="section-3-1"></span>

### 3.1 TypedTile: 带类型标注的张量 Tile

LADDER 提出 TypedTile (tTile), 一种带数据类型标注, 基于 tile 的张量抽象, 用于同时表示 DNN 计算与硬件要求. [图 4](#figure-04) 给出了 tType, tTile 和 tTile-operator 的定义.

<span id="figure-04"></span>

![tType, tTile 和 tTile-operator 的定义](./ladder/figure-04.png)

**图 4.** tType, tTile 和 tTile-operator 的定义.

**tType.** tType 是一种 tile 级张量数据类型, 对一个或多个元素的数值格式与精度作了概括. 具体而言, tType 包含每个元素的比特数 (`nElemBits`), 形状 (`shape`), 以及可转换到的 tType (`c_tTypes`) 和相应转换函数 (`c_funcs`). `shape` 表示一个 tType 所含元素的数量与布局. 当 `shape=[]` 时, tType 表示标量. 例如, FP16 可写为 `tType(nElemBits=16, shape=[])`. tType 也能表示一组元素, 例如分组数据类型. 转换函数说明如何把当前 tType 无损转换为其他 tType. 根据这一定义, INT4, NF4, FP8 和 MXFP 等常见低精度数据类型均可表示为 tType.

**tTile.** tTile 是用 tType 标注的张量 tile, 写作 `tTile(shape, dtype)`. `shape` 描述 tile 的逻辑形状, `dtype` 则是它的 tType. 一个张量可以划分为多个 tTile, 一个 tTile 在一个 tType 粒度内也可以包含多个元素. 该抽象明确表示了数据形状与存储粒度.

**tTile-operator 与 tTile-graph.** tTile-operator 是输入和输出均为 tTile 的计算任务. LADDER 为输入, 输出和中间张量标注 tType, 从而扩展张量表达式. DNN 模型随后被表示为由 tTile-operator 构成的数据流图, 称为 tTile-graph.

硬件加速器具有由内存层级 (如 DRAM, 寄存器) 和计算单元构成的硬件层次结构. 每一层都有偏好的数据访问方式. 具体而言, 内存层通常要求按事务访问, 一个事务是在某种粒度下连续排列或具有特定形状的数据. 例如, NVIDIA GPU 的共享内存要求一次事务访问 32 个 4 字节 bank. 计算单元通常也要求按某种粒度处理特定形状的数据. 例如, NVIDIA GPU 的 `hfma2` 指令以两个 FP16 值为处理粒度.

这些要求都能用 tTile 描述. 因此, LADDER 把硬件加速器抽象为由 tTile 描述的多层结构, 即 tTile-device. 每层都是内存层或计算单元; 它在特定粒度上对形状的要求由 tTile 表示, 该粒度则由 tType 表示.

<span id="figure-05"></span>

![使用 tTile 表示并转换的 FP16 与 NF4 混合矩阵乘法](./ladder/figure-05.png)

**图 5.** FP16 张量 A 与 NF4 张量 B 的 MatMul: (a) 带 tType 标注的张量表达式, (b) NVIDIA A100 的 tTile-device, (c) 计算流水线伪代码, (d) 使用 tTile 转换原语的 Transform-Load, (e) 张量 B 的转换.

[图 5](#figure-05)(b) 给出了带 FP16 tensor core 的 NVIDIA A100 GPU 所对应的 tTile-device. FP16 tensor core 的 MMA 指令 [+mma] 要求两个输入分别按 $[16,16]$ 与 $[8,16]$ 的粒度处理. 该要求可以表示为形状 $[16,16]$, `dtype=FP16` 的 tTile. FP16 tensor core 的数据加载指令 [+ldmatrix] 要求以 `half8` 粒度 (即 8 个 FP16 值) 加载 $[16,2]$ 数据, 可表示为形状 $[16,2]$, `dtype=16B` 的 tTile. 充分利用共享内存的要求可以表示为形状 $[32]$, `dtype=4B` 的 tTile. 全局内存的 32 字节事务要求则可以表示为形状 $[32]$, `dtype=1B` 的 tTile.

[+mma]: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f32.f32`
[+ldmatrix]: `ldmatrix.sync.aligned.m8n8.x4.shared.b16`

<span id="section-3-2"></span>

### 3.2 tTile 转换

tTile 明确描述细粒度张量存储和硬件层次结构的要求. 为高效执行, tTile-graph 中由 tTile 表示的 DNN 计算应与 tTile-device 对齐. 根据[第 2 节](#section-2) 的观察, 流水线中的张量存储与访问可以转换为逻辑等价的格式, 各种格式在硬件层次中的性能表现并不相同. 因此, LADDER 提出 tTile 转换机制, 把 tTile 的布局或 tType 转换为等价的 tTile. 具体而言, LADDER 把 tTile-operator 的计算流水线扩展为硬件层次上的三个阶段: Transform-Load, Compute 和 Transform-Store. Transform-Load 通过 tTile 转换把 tTile 从较低内存层加载到较高内存层. Compute 在计算单元上执行 tTile-operator 的计算任务. Transform-Store 通过 tTile 转换把 tTile 从较高内存层存储到较低内存层.

<span id="figure-06"></span>

![四种 tTile 转换原语](./ladder/figure-06.png)

**图 6.** tTile 转换原语.

LADDER 提供四种把 tTile 转换为等价 tTile 的原语, 如[图 6](#figure-06) 所示.

**Slice.** slice 原语从 `tTile_input` 的地址 `index` 开始切出一组形状为 `shape` 的元素, 并将其作为形状为 `out_shape` 的新 tTile 返回. slice 原语通常用于表示数据分块.

**Map.** map 原语修改 tTile 中元素的布局. 给定 `map_func` 后, map 原语把每个元素的地址映射到预期地址. 例如, [图 5](#figure-05)(d) 中从 L2 内存层到 L1 内存层的 `TransformLoad_L1B` 使用 map 原语, 通过 `map_func` 修改元素地址.

**Pad.** pad 原语按照 `pad_shape` 给出的各边界, 用 `pad_value` 填充 `tTile_input`. `pad_shape` 的长度是 `tTile_input` 形状维数的两倍, 分别描述各维度的左右边界.

**Convert.** convert 原语把 `tTile_input` 的 tType 转换为指定的 `new_tType`. `new_tType` 应属于 `tTile_input` 所用 tType 的 `c_tTypes`. convert 会对 `tTile_input` 中的每个元素调用 `new_tType` 对应的 `c_func`, 返回预期的 `new_tType` tTile. 例如, [图 5](#figure-05)(d) 中的 `TransformLoad_L1B` 用 convert 原语把 tType 从 NF4 转为 FP16, 以满足计算核心对 FP16 tType 的要求.

借助上述四种原语, 可以通过 slice 和 pad 改变形状, 通过 map 修改元素布局, 或通过 convert 转换 tType, 从而把一个 tTile 转换为另一个等价的 tTile. 这样就能转换 tTile-operator 的各个 tTile, 使其与 tTile-device 对齐, 并在硬件层次结构中得到高效处理.

[图 5](#figure-05) 给出了一个例子: 在四层 tTile-device 上, 一个 FP16 张量 $A[32, 63]$ 与一个 NF4 张量 $B[32, 63]$ 相乘, 以 FP32 累加, 输出 FP16 张量 $C[32, 32]$ ([图 5](#figure-05)(a)); 这四层从 L2 延伸到计算核心, 如[图 5](#figure-05)(b) 所示. [图 5](#figure-05)(c) 给出了执行过程的伪代码. A 和 B 的 tTile 经过转换, 以 FP16 类型从 L2 加载到 L1. 随后, tTile 通过 `ldmatrix` 加载到 L0, 再由 `mma` 指令处理; 中间结果在 L0 中以 FP32 累加. 最后, L0 中 C 的 tTile 经过转换, 以 FP16 存回 L2. [图 5](#figure-05)(d)(e) 详细展示了如何转换 NF4 张量 B 以对齐 tTile-device; 张量 A 的转换与之类似. 具体来说, `mma` 和 `ldmatrix` 指令要求 L1 中的数据为 FP16. 各层还有[图 5](#figure-05)(b) 所示的事务要求. 因此, `TransformLoad_L1B` 切出 $[16, 63]$ 并填充为 $[16, 64]$, 使其符合 L2 的事务要求. 然后, `TransformLoad_L1B` 将其转换为 FP16, 并映射到另一种元素布局, 以符合 L1 和 L0 的事务要求. 最终在 L1 中得到 FP16 的 `L1_B` $[16, 64]$. 接着, `TransformLoad_L0B` 利用 `ldmatrix` 切分 `L1_B`, 在 L0 上得到 FP16 的 `L0_B` $[16, 16]$, 同时满足 L1, L0 和 `mma` 核心的要求.

<span id="section-3-3"></span>

### 3.3 硬件感知的 tTile-Graph 调度

要把以 tTile-graph 表示的 DNN 计算调度到 tTile-device, 可以将每个 tTile-operator 面向 tTile 的计算流水线, 即 Transform-Load, Compute 和 Transform-Store, 映射到 tTile-device. 具体来说, 可以把每个 tTile-operator 划分成多个 tTile, 以适应各内存层的容量; 调度 tTile 转换, 使 tTile 满足各硬件层的要求; 还可以协调算子之间的 tTile 配置和转换, 做整体优化. 最终, 整个 tTile-graph 被调度为一条数据流水线: tTile-operator 节点的 tTile 沿硬件层次上下移动, 并跨边传递给后继 tTile-operator 节点.

tTile 为 DNN 计算调度增加了张量转换这一维度, 因而显著扩大了 tTile-graph 的调度空间. 此外, tTile 转换在内存占用效率与延迟效率之间引入了新的权衡, 也让调度更加复杂. 以 NVIDIA GPU 上 FP16 张量与 NF4 张量的 MatMul 为例, 受硬件支持限制, 必须把 NF4 转换为 FP16. 该转换要在从 L1 到 L0 的 Transform-Load 之前完成, 因而可以调度到 L2 或 L1. 若在 L2 转换, L2 和 L1 会占用更多内存, 但随后 tTile 从 L2 移到 L1, 再移到 L0 时不会占用计算单元. 若在 L1 转换, 则能节省 L2 内存和 L2 内存带宽, 但类型转换会占用计算单元. 当算子受计算单元限制时, 前一种方案延迟更低, 但内存占用更大; 当算子受内存 I/O 限制时, 后一种方案在延迟和内存占用两方面都更好. 另外, convert 只需在 L1 到 L0 的 Transform-Load 之前完成, 因此还可以把它融合进前一个算子执行, 改善端到端性能.

面对如此庞大的调度空间, LADDER 提供了一种面向延迟的策略, 目标是尽量降低端到端延迟. 具体而言, LADDER 提出一种硬件感知的分层调度策略: 较低层内存把偏好的数据访问粒度表示为一个 tTile, 作为提示; 上层通过转换与该 tTile 所表示的粒度对齐, 决定最优计算粒度. 为了缩小调度空间, 并在合理时间内找到合适方案, LADDER 采用了基于观察得到的启发式方法.

**调度策略.** 算法 1 描述了基于提示的分层调度策略. 它接收以 tTile-graph 表示的 DNN 模型 $g$ 和以 tTile-device 表示的硬件规格 $D$, 返回调度后的 tTile-graph $g_{ret}$. 首先, 该策略把图调度为若干子图 (第 33 行). 每个子图表示一条计算流水线: 从最低内存层把 tTile 加载到计算核心, 再把结果存回最低内存层. 一个子图可以是单个 tTile-operator, 也可以是一组能够融合的 tTile-operator. `ExtractConnectedGraph` 可以利用现有 DNN 编译器的工作 [Che18, Shi23a].

**算法 1. 基于提示的分层调度**

- **输入:** $g$: tTile-graph; $D$: tTile-device
- **输出:** $g_{ret}$: scheduled tTile-graph
- **函数** `GetDeviceHint(g, D)`:
  - $D=$ `SelectDeviceConfig(g, D)`
  - `HintShape = None`, `HintGranularity = None`
  - **对每个** `layer ∈ D.layers`:
    - `HintGranularity = LCM(HintGranularity, layer.tTile.type)`
  - **对每个** `layer ∈ D.layers`:
    - `layer.tTile = convert(layer.tTile, HintGranularity)`
    - `HintShape = LCM(HintShape, layer.tTile.shape)`
  - **对每个** `layer ∈ D.layers`:
    - `layer.tTile.shape = HintShape`
  - **返回** $D$
- **函数** `ScheduleTransform(op, D, l_id)`:
  - `tTile_h = op.tTile[l_id - 1]`
  - `tTile_l = op.tTile[l_id]`
  - `ScheduleSlice(tTile_l, tTile_h)`
  - **如果** `LCM(tTile_l.shape, tTile_h.shape) != tTile_l.shape`:
    - `SchedulePad(tTile_l, tTile_h, D)`
  - **如果** `tTile_l.type != tTile_h.type`:
    - `ScheduleConvert(tTile_l, tTile_h, D)`
  - **如果** `nBits(tTile_h.shape[-1]) != nBits(D.layers[l_id].shape[-1])`:
    - `ScheduleMap(tTile_l, tTile_h, D)`
  - **返回** `op.transform[l_id - 1]`
- **函数** `ScheduleConnectedGraph(g, D)`:
  - `D = GetDeviceHint(g, D)`
  - **对每个** `l_id` in `length(D.layers)`:
    - **对每个** `op ∈ g[l_id]`:
      - `op.tTile[l_id] = ScheduleTiling(op, D, l_id)`
      - **如果** `l_id > 0`:
        - `op.transform[l_id] = ScheduleTransform(op, D, l_id)`
  - `g = ProfileAndSelect(g)`
  - **返回** $g$
- **函数** `Schedule(g, D)`:
  - `g = ExtractConnectedGraph(g, D)`
  - **对每个** $g_{\mathit{conn}}\in g$:
    - `g_conn = ScheduleConnectedGraph(g_conn, D)`
  - **返回** $g$

给定一个子图, 策略先从硬件推断提示. 它首先选择合适的硬件配置, 例如计算核心 (第 2 行), 并优先选用硬件支持, 位宽最接近的 tType. 位数更多的数值类型通常需要更多晶体管来实现硬件指令, 性能也往往更低. 例如在 NVIDIA A100 GPU 上, NF4 可以转换为 FP16 或 FP32 处理, LADDER 会选择 FP16 核心 (312 TFlops), 而不是 FP32 核心 (19.5 TFlops). 随后, 策略通过位对齐找出各硬件层对齐后的粒度和形状, 并配置提示 (第 1-11 行). 仍以 NVIDIA A100 为例 ([图 5](#figure-05)(b)), `HintGranularity` 是 `ldmatrix` 所要求的 16B, `HintShape` 是 $[4, 8]$; 其内维度为 128B, 可同时对齐全局内存的 32B 事务和共享内存的 128B 事务. 接下来, 策略从顶层, 即计算核心, 到底层, 即 DRAM, 逐层调度该子图 (第 25-29 行). 在每一层, 策略先利用提示, 通过 `ScheduleTiling` 调度 tTile-operator 分块 (第 27 行), 再调度 tTile 转换 (第 29 行). 如果 `ScheduleTiling` 以 16B 为粒度, 把算子分块调度为 $[4, 8]$ 的倍数, 后续 `ScheduleTransform` 就能让该调度与 tTile-device 对齐. `ScheduleTiling` 还可以利用现有张量编译器 [Che18, Zhe20, Zhu22]. 在 `ScheduleTransform` 中, 策略会检查形状和类型是否都与 tTile-device 对齐, 再调度相应转换以对齐 tTile (第 12-22 行). 调度后可能得到多个候选方案, 系统会逐一剖析并返回其中最优者 (第 30 行).

**ScheduleMap.** 调度 map 转换时, `map_func` 的确定并不简单. LADDER 提出了一种推断 `map_func` 的方法: 按行主序把 tTile 中的元素映射到所需的事务大小. [图 5](#figure-05)(e) 给出了一个例子: 以 16B 为粒度, 要把 L0 中形状为 $[16, 2]$ 的数据映射为 L1 所需的形状 $[8]$, 就按行主序展平元素, 得到形状 $[4, 8]$. map 也支持其他 `map_func`. 该调度策略不保证最优. 不过, [第 5 节](#section-5) 表明, 它已经能够超过现有最佳方案, 并在 GPU 上实现高效的低精度 DNN 计算. 我们也希望未来研究能以更先进的调度策略, 继续探索这一调度机制所打开的优化空间.

<span id="section-4"></span>

## 4 实现

LADDER 基于开源 DNN 编译器 TVM [Che18], Welder [Shi23a] 和 Roller [Zhu22] 实现, 包含约 5000 行 Python 与 C++ 代码. LADDER 修改 TVM, 用于实现内核调度和生成内核代码; Roller 用于推断高效的 tTile 配置. Welder 是能够整体优化 DNN 模型的先进 DNN 编译器, LADDER 用它完成端到端图优化.

LADDER 的输入是 PyTorch 程序. 对 PyTorch 内置数据类型, 无需修改 DNN 模型程序. 对 PyTorch 不支持的新数据类型, LADDER 则用自定义算子扩展 PyTorch, 以表达用户定义数据类型上的张量表达式. LADDER 接收 PyTorch 程序后, 将其导出为 ONNX 图. LADDER 还扩展了 ONNX, 用以表示新数据类型上的计算; 带 tType 标注的张量表达式保存在 ONNX 图节点的属性中. 根据导出的 ONNX 图和目标硬件加速器基于 tTile 的规格文件, LADDER 自动把 ONNX 图转换为 tTile-graph 并完成调度, 随后生成目标硬件加速器的设备代码.

NVIDIA GPU 和 AMD GPU 是使用最广泛的 DNN 加速器, 因此我们为二者实现了 LADDER. 本节余下部分会详细介绍 NVIDIA GPU 上的实现, 并简要说明 AMD GPU 上的实现. 只要新的硬件指令, 如最新 Hopper GPU 的 FP8 张量核心, 或其他硬件加速器, 如 Graphcore IPU, 符合基于 tTile 的硬件抽象, 并提供在硬件层次结构中加载和存储数据的编程接口, LADDER 也可以移植到这些平台.

<span id="section-4-1"></span>

### 4.1 NVIDIA CUDA GPU 上的 LADDER

<span id="section-4-1-1"></span>

#### 4.1.1 tType 与 tTile

LADDER 已为常见数据类型实现 tType, 包括 FP32, FP16, INT8, FP8, MXFP, INT4, NF4 和 INT1.

GPU 采用单指令多线程 (SIMT) 架构, 倾向于让一组线程在不同数据上执行同一条指令. 因此, LADDER 在 tTile 中分别存储元素和各项元数据. [图 7](#figure-07) 展示了 NVIDIA GPU 上形状为 $\left[32, 32\right]$ 的 MXFP8 tTile 如何存储. 元素保存在一个数组中, 共享缩放因子保存在另一个数组中. 访问 tTile 时, 连续线程处理连续元素, 从而形成合并访问. 需要注意, 某些数据类型的 `nElemBits` 并不是 $2^{n}$, 例如 3-bit [Fra22]. 受 GPU 规格约束, LADDER 以 4B 为粒度存储这类数据类型; 例如, 10 个 3-bit 值可以存入一个 4B (32-bit) 粒度.

<span id="figure-07"></span>

![NVIDIA GPU 上 MXFP8 tTile 的存储方式](./ladder/figure-07.png)

**图 7.** NVIDIA GPU 上形状为 $\left[32, 32\right]$ 的 MXFP8 tTile 的存储方式. E: 元素. S: 元数据中的共享缩放因子.

<span id="section-4-1-2"></span>

#### 4.1.2 使用 PTX 指令优化代码生成

NVIDIA 不直接提供可供编程的汇编指令, 而是引入并行线程执行 (Parallel-Thread-Execution, PTX), 作为 NVIDIA GPU 的低级虚拟机. PTX 虚拟机上的指令集架构 (ISA) 可以看作 NVIDIA GPU 的指令级 API [Nvi25]. CUDA C++ 代码先编译为 PTX 代码, 再编译为可执行的机器码. 对部分单元, CUDA 同时提供 C++ API 和 PTX API. 例如, 张量核心既提供 WMMA C++ API, 也提供 MMA PTX API; nvcc 编译器会把一个 WMMA API 编译为一组 MMA 指令. MMA PTX API 比 WMMA C++ API 更灵活, 性能也更好. LADDER 在张量核心上使用 MMA PTX API 生成代码, 并用 `cp.async` 指令支持 Ampere GPU 新增的异步内存复制功能 [Nvi20]. 我们还发现, 把低位整数, 如 INT4, 转换为浮点数, 如 FP16, 可能产生显著开销. LADDER 使用 LOP3 指令实现低于 4-bit 整数的转换 [Nvi25]. 为实现这些优化, 我们修改了 TVM 的代码生成模块.

<span id="section-4-2"></span>

### 4.2 AMD ROCm GPU 上的 LADDER

AMD GPU 与 NVIDIA GPU 相似, 同样具有由所有 CU 共享的全局内存, 每个 CU 内的本地数据存储 (类似共享内存), 寄存器和计算核心组成的硬件层次结构. 因此, AMD GPU 也可以抽象为四层 tTile-device, 各层采用不同的 tTile 配置. ROCm 为 AMD GPU 提供 HIP 编程模型 [Roc16], 功能与 CUDA 相近, 支持大多数 CUDA 语句. 我们在 TVM 中实现了新的 HIP 代码生成后端, 以支持 AMD ROCm GPU. 此外, 我们通过 MFMA (Matrix Fused-Multiply Add) ISA 级 API 使用矩阵核心, 即 NVIDIA 张量核心在 AMD 平台上的对应单元.

<span id="section-5"></span>

## 5 评估

<span id="section-5-1"></span>

### 5.1 评估设置

**硬件平台.** 我们在 NVIDIA 和 AMD 的多款 GPU 上评估 LADDER, 以全面考察其在不同硬件生态中的性能. NVIDIA 平台包括三款高性能 GPU: Tesla V100 (16GB), A100 (80GB) 和 RTX A6000 (48GB), 使用 CUDA toolkit 12.1. AMD 平台使用 AMD Instinct MI250 GPU (128GB) 和 ROCm toolkit 5.7.0. 所有平台均运行 Ubuntu 20.04.

**DNN 模型.** 我们在一组涵盖不同领域和架构的先进 DNN 模型上测试推理性能, 以评估 LADDER 的有效性. 其中包括大语言模型 LLAMA-70B [Tou23a] 和 BLOOM-176B [Les23], 计算机视觉模型 ResNet-50 [He16], ShuffleNet-V2 [Ma18] 和 ViT-Base [Dos20], 以及音频模型 transducer Conformer-L [Gul20]. 这些模型的数据类型配置均来自先进研究, 并经过深度学习社区评估. LADDER 沿用这些配置, 不会额外损害模型质量. 下文列出各模型的权重和激活数据类型配置, 记为 $W_{\mathrm{type}}A_{\mathrm{type}}$:

- **LLAMA-70B 和 BLOOM-176B:** 评估 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Tou23a, Les23], $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ [Fra22, Lin23d], $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ [Det23a], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Mic22], $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$ [Dar23] 和 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ [Wan23].
- **ResNet-50:** 评估 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [He16], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Mic22], $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$ [Dar23] 和 $W_{\mathrm{INT1}}A_{\mathrm{INT4}}$ [Hua19d].
- **ShuffleNet-V2:** 评估 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Ma18] 和 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [She23f].
- **ViT-Base:** 评估 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Dos20], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Kuz22] 和 $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ [Li22c].
- **Conformer-L:** 评估 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Gul20], $W_{\mathrm{INT8}}A_{\mathrm{INT4}}$ [Din22] 和 $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ [Din22].

我们设置了多种批大小 (BS) 和序列长度 (SEQ), 覆盖不同部署场景. 对 LLAMA-70B 和 BLOOM-176B 等大语言模型, 测试 (BS, SEQ) 为 (1, 1), (32, 1) 和 (1, 4096) 的情况, 以覆盖在线与离线推理, 以及预填充与解码阶段. 对 ResNet-50, ShuffleNet-V2, ViT-Base 和 Conformer-L, 则分别以批大小 1 和 128 评估在线与离线推理性能.

**基线.** 我们在不同 GPU 平台上将 LADDER 与多种成熟编译器和框架比较. NVIDIA GPU 上的基线包括 Welder [Shi23a], PyTorch-Inductor [Pas19], ONNXRuntime [Onn24], TensorRT [Ten24], AMOS [Zhe22d], TensorIR [Fen23], vLLM [Kwo23], 以及 vLLM-$W_{\mathrm{INT4}}A_{\mathrm{FP16}}$, 即 vLLM 对 4-bit 量化模型的支持 [Kwo23]. AMD GPU 上的基线包括 Welder [Shi23a], PyTorch-Inductor [Pas19], ONNXRuntime [Onn24] 和 TensorIR [Fen23].

为了使用 ROCm 设备的 MatrixCore, 我们把 MIOpen 和 rocBLAS 集成到 Welder, 并为 TensorIR 加入 rocWMMA Auto Tensorize 支持. 在算子基准测试中, LADDER 与 cuBLAS [Cub16], CUTLASS [Nvi24a], vLLM [Kwo23], cuDNN [Nvi25c], AMOS [Zhe22d] 和 TensorIR [Fen23] 比较.

<span id="section-5-2"></span>

### 5.2 NVIDIA GPU 上的评估

<span id="section-5-2-1"></span>

#### 5.2.1 端到端性能

**推理延迟.** 我们在 Tesla A100, V100 和 RTX A6000 GPU 上测试前述 DNN 模型的推理延迟. 对 LLAMA-70B 和 BLOOM-176B 等大语言模型, 受 GPU 内存容量限制, 我们使用单个解码器层评估推理延迟. 由于各层相同, 且延迟随层数线性增长, 单层结果可以代表完整模型的性能.

<span id="figure-08"></span>

![NVIDIA A100 GPU 上的端到端性能](./ladder/figure-08.png)

**图 8.** NVIDIA A100 GPU 上的端到端性能.

[图 8](#figure-08) 汇总了 A100 GPU 上的推理延迟. 在 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 配置下, LADDER 显著提升了性能. 与 Welder 相比, LLAMA, BLOOM, ResNet, ShuffleNet, Conformer 和 ViT 的平均加速比分别为 1.0×, 1.2×, 2.0×, 1.2×, 1.1× 和 1.4×. 原因在于 Welder 使用 Roller [Zhu22], cuBLAS [Cub16] 和 CUTLASS [Nvi24a] 生成内核, 会遇到共享内存 bank conflict 等内核性能问题; ResNet 的 Conv2D 算子包含更多不规则形状, 问题尤为明显. LADDER 通过张量转换调度解决这些问题, 因而效率更高. 例如, ResNet 在 BS1 和 BS128 下的延迟分别为 1.1 ms 和 7.6 ms. 对 LLM 中常用的 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 配置, LADDER 相比 vLLM 平均加速 2.3×. LADDER 还能支持其他系统通常无法处理的自定义数据类型. 例如在 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 配置下, BLOOM-176B-BS1SEQ1 单层延迟为 0.32 ms, 相比 Welder 最多加速 10×.

<span id="figure-09"></span>

![NVIDIA V100 GPU 上的端到端性能](./ladder/figure-09.png)

**图 9.** NVIDIA V100 GPU 上的端到端性能.

<span id="figure-10"></span>

![NVIDIA RTX A6000 GPU 上的端到端性能](./ladder/figure-10.png)

**图 10.** NVIDIA RTX A6000 GPU 上的端到端性能.

我们还在 Tesla V100 和 RTX A6000 GPU 上评估推理延迟, 结果见[图 9](#figure-09) 和[图 10](#figure-10). 这些平台上的结果与 A100 基本一致. V100 只有 16GB 内存, 即使运行 BLOOM 模型的单个解码器层也会受限, 出现内存不足错误. 在 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 配置下, LADDER 相比 Welder 在 V100 上平均加速 1.1×, 在 A6000 上平均加速 1.2×. 在 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 配置下, LADDER 在 A6000 上相比 vLLM 平均加速 2.0×, 同时也让 V100 能够高效执行 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 推理. 在 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 配置下, 相比 Welder, LADDER 在 V100 和 A6000 上最多分别加速 13.3× 和 14.6×.

**内存占用.** 使用低精度数据类型是缓解大语言模型巨大内存需求的重要手段. 为量化这一收益, 我们详细考察了 A100 GPU 上不同数据类型配置在 LLM 推理期间的内存占用. [图 11](#figure-11) 显示, 内存占用会随位宽降低而近似线性下降. 这一趋势在序列长度为 1 的解码阶段尤其明显, 说明精度缩放对内存密集的推理解码阶段很有价值.

<span id="figure-11"></span>

![NVIDIA A100 GPU 上 LLM 推理的内存占用](./ladder/figure-11.png)

**图 11.** NVIDIA A100 GPU 上不同数据类型配置的 LLM 推理内存占用.

在最极端的 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 场景中, 权重精度为 1-bit, 激活精度为 8-bit, 内存节省十分显著. 与全精度 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 配置相比, LLAMA 模型在三种批大小与序列长度组合下的推理内存占用分别降低 74%, 74% 和 24%; BLOOM 模型在对应设置下分别降低 85%, 85% 和 6%.

**编译时间.** 为评估系统效率, [表 2](#table-02) 比较了 LADDER, AMOS, TensorIR 和 Welder 的编译时间. 我们在 NVIDIA A100 GPU 上, 分别以批大小 1 和 128 对 ResNet 与 ShuffleNet 两个代表性神经网络模型做端到端编译计时. 结果表明, LADDER 的平均编译时间明显短于 AMOS 和 TensorIR: 比 TensorIR 快一个数量级, 比 AMOS 快两个数量级. LADDER 通过张量转换支持低精度算术, 因而天生具有更大的调度空间. 这让 LADDER 能够利用低精度算术的性能收益, 也会在编译过程中增加一些额外开销, 因此其编译时间略高于 Welder.

<span id="table-02"></span>

![NVIDIA A100 GPU 上的编译时间比较](./ladder/table-02.png)

**表 2.** NVIDIA A100 GPU 上端到端模型的编译时间比较, 单位为分钟.

<span id="section-5-2-2"></span>

#### 5.2.2 算子基准测试

为评估 LADDER 的内核性能, 我们构建了一个算子基准, 包含 LLAMA 和 ResNet 模型中的常用算子. 该基准由 6 个矩阵乘 (MatMul) 算子 M0-M5 和 8 个二维卷积 (Conv2d) 算子 C0-C7 组成. 每个算子都在多种数据类型配置下测试, 包括 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$, $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$, $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$, $W_{\mathrm{FP8}}A_{\mathrm{FP16}}$, $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 和 $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$. 所有实验都在 NVIDIA A100 GPU 上执行, 以保证性能评估的一致性和可靠性.

<span id="figure-12"></span>

![NVIDIA A100 GPU 上的算子基准](./ladder/figure-12.png)

**图 12.** NVIDIA A100 GPU 上的算子基准.

如[图 12](#figure-12) 所示, LADDER 在 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 配置下表现最佳. 切换到 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 后, LADDER 平均加速 1.8×; $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 配置进一步达到 4.5× 的平均加速. Ada Lovelace, Hopper 和 Blackwell GPU 支持 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ 张量核心. 我们还在配备 CUDA 12.4 的 NVIDIA RTX 4090 GPU 上执行算子基准, 评估硬件支持的 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ 性能, 结果见[图 13](#figure-13).

<span id="figure-13"></span>

![NVIDIA RTX 4090 GPU 上的算子基准](./ladder/figure-13.png)

**图 13.** NVIDIA RTX 4090 GPU 上的算子基准.

对 $W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$, LADDER 超过 cuBLAS, 并达到与 CUTLASS 相当的性能. 对 $W_{\mathrm{FP8\_E5M2}}A_{\mathrm{FP8\_E5M2}}$, LADDER 同样与 CUTLASS 性能相当, 而 cuBLAS 不支持该情况. RTX 4090 只开放使用 FP32 累加的 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$, 其理论性能与 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 相同. 因而在 M2, M5 这类大型矩阵上, cuBLAS, CUTLASS 和 LADDER 的 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ 性能都与 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 接近. 使用 FP16 累加的 $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ 虽然理论性能可翻倍, NVIDIA 目前却没有开放. 由于 RTX 4090 的数据类型转换核心更强, LADDER 在 $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ 和 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 等数据类型上的加速比也高于 A100.

<span id="section-5-2-3"></span>

#### 5.2.3 优化分解

<span id="figure-14"></span>

![逐步优化拆解](./ladder/figure-14.png)

**图 14.** 优化拆解.

[图 14](#figure-14) 展示了 LADDER 如何针对不同数据格式, 逐步优化 LLAMA-70B 模型在单 token 序列 (BS1 SEQ1) 和长序列 (BS1 SEQ4096) 下的内核. 分块感知的内核转换让数据处理更顺畅, 相比 Roller 基线加速 2.0×, 同时支持多种数据类型. PTX 级优化减少了 GPU 内存负载; 借助对张量操作与布局的细粒度控制, LADDER 又获得最多 1.7× 的加速. 综合调度策略通过优化转换, 最多带来 2.5× 加速, 对 MXFP8 这类受内存约束的数据类型尤其有效. 总体而言, LADDER 的优化提升了计算效率和适应性, 在多种操作上都取得了明显的性能收益.

<span id="section-5-2-4"></span>

#### 5.2.4 缩放位宽

借助 LADDER 的通用能力, 权重和激活都可以采用任意位宽的多种数据类型. 为全面评估精度缩放对性能的影响, 我们在位宽逐步降低的数据类型设置上进行实验, 并在两种不同的批大小与序列长度组合下, 同时考察端到端性能和单算子性能.

<span id="figure-15"></span>

![缩放权重和激活位宽时的性能](./ladder/figure-15.png)

**图 15.** 缩放权重和激活的位宽.

[图 15](#figure-15) 给出了实验结果. 随着 W 和 A 的位宽降低, 加速比相应提高, 体现出低精度算术的效率收益. 序列长度为 1 的解码场景受内存限制; 实验表明, 当 W 位宽从 $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ 降到 $W_{\mathrm{INT2}}A_{\mathrm{INT4}}$, 再降到 $W_{\mathrm{INT1}}A_{\mathrm{INT4}}$ 时, 加速比明显上升. 序列长度为 4096 的编码场景则受计算限制. 由于混合精度操作仍依赖更高精度计算, 这些配置的加速比基本不变.

<span id="section-5-2-5"></span>

#### 5.2.5 低精度 LLM 的效率与准确度

低精度计算同时关注模型质量和效率, 因此设计低精度模型时通常需要权衡效率与准确率. 我们以 LLAMA2-3B, LLAMA2-7B, LLAMA2-13B 和 LLAMA2-70B 等 LLM 为例, 同时评估先进低精度方法的效率与准确率. 具体包括 $W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$ 的 PTQ [Aut23, Mic22], $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 的 GPTQ [Fra22], $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ 的 PTQ [Det22c], $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$ 的 BitDistiller [Du24], $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 的 OneBit [Xu24h], 以及 $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$ 的 BitNet-b1.58 [Ma24]. PTQ 和 GPTQ 都是训练后量化方法, 不涉及模型训练. BitDistiller 和 OneBit 是量化感知训练方法, 通过蒸馏实现 2-bit 和 1-bit 权重量化. BitNet-b1.58 从头训练 LLM, 获得以 $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$ 表示的三值权重.

<span id="figure-16"></span>

![LLM 低精度方法的困惑度与延迟](./ladder/figure-16.png)

**图 16.** 不同 LLM 低精度方法在 WikiText-2 上的 PPL (↓), 以及在 A100 上解码单个 token 的延迟 (ms). $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$ 的 G64 表示每 64 个元素共享一个缩放因子. LLAMA2-70B 使用流水线并行.

[图 16](#figure-16) 展示了 WikiText-2 上的困惑度 (PPL), 以及 A100 上解码单个 token 的延迟. PPL 越低, 模型质量越好. $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 和 $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ 的 PPL 来自 AFPQ [Zha23aa], $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 的 PPL 来自 OneBit [Xu24h], LLAMA2-3B 上 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 的 PPL 来自 BitNet-b1.58 [Ma24]. 其他模型的 PPL 使用开源模型检查点和开源实现评估. $W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$, $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ 和 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 对 PPL 影响很小, 平均加速比分别为 1.6×, 1.7× 和 2.5×. 使用 PTQ 和 GPTQ 把 LLM 权重量化到 2-bit 会得到 NaN PPL [Du24, Xu24h], BitDistiller 和 OneBit 则利用蒸馏, 在 2-bit 和 1-bit 量化下取得稳定结果. 不过, 分组缩放给 $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$-G64 引入了额外计算开销, 因而其加速比与 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 接近.

值得注意的是, 在 LLAMA2-3B 配置上, 与使用相同数据集和相同 token 数训练的 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 模型相比, BitNet-b1.58 不但 PPL 更好, 还加速 1.8× [Ma24]. 该加速没有达到理论值, 因为 LLAMA2-3B 太小, 无法充分利用 GPU. 我们进一步在 LLAMA2-70B 配置上评估 BitNet-b1.58 的 $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$, 相比 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 加速 4.6×. 因此, BitNet-b1.58 在准确率和效率两方面都展现出良好潜力.

比较不同模型配置时, 模型规模对准确率和效率都有显著影响. 采用 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 的 LLAMA2-13B 在准确率和效率上都优于采用 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 的 LLAMA2-7B; 量化后的 LLAMA2-7B 也在两方面都优于采用 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 的 LLAMA2-3B. 这说明了低精度计算的价值.

社区正在积极探索低精度计算. 我们希望 LADDER 能够提供效率反馈, 帮助研究者推进这一方向.

<span id="section-5-3"></span>

### 5.3 AMD GPU 上的评估

我们在 AMD Instinct MI250 GPU 上评估 LADDER, 并与 Welder, PyTorch-Inductor 和 ONNXRuntime 比较. [图 17](#figure-17) 给出了 6 个模型的端到端性能.

<span id="figure-17"></span>

![AMD Instinct MI250 GPU 上的端到端性能](./ladder/figure-17.png)

**图 17.** AMD Instinct MI250 GPU 上的端到端性能.

在 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 数据类型下, LADDER 相比 Welder 在 LLAMA, BLOOM, ResNet, ShuffleNet, Conformer 和 ViT 上的平均加速比分别为 2.1×, 2.35×, 1.5×, 10.5×, 1.6× 和 1.5×. Welder 在 ShuffleNet 上表现不佳, 因为它使用 rocBLAS 和 MIOpen 驱动矩阵核心, 从而破坏了融合机会. LADDER 不仅能为矩阵核心生成高效计算内核, 还能保留更多融合机会. 在 ShuffleNet-BS1 上, LADDER 延迟为 0.43 ms, 相比 Welder 加速 14.1×. 对 LLM 的 $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ 配置, 与 Welder 相比, LADDER 在 LLAMA-BS1SEQ1 上延迟为 0.73 ms, 最多加速 3.8×; 在 BLOOM-BS1SEQ1 上延迟为 1.75 ms, 最多加速 4.5×.

<span id="section-6"></span>

## 6 讨论

LADDER 当前的实现主要面向模型推理. 本节讨论它的一些局限和未来工作.

**多 GPU 服务.** BLOOM-176B 和 LLAMA2-70B 等大规模模型无法装入单块 GPU, 部署时需要多块 GPU. 多 GPU 支持与 LADDER 互补. LADDER 关注单个硬件加速器上的低精度计算, 多 GPU 框架 [Kwo23, Sto23d, Lin24i, Zhe22] 则负责划分模型, 并在多块 GPU 之间调度并行计算. 二者可以协作: 多 GPU 框架先划分模型, 再把各设备上的分块计算交给 LADDER 执行, 从而支持低精度模型的多 GPU 并行计算. 将 LADDER 与多 GPU 框架集成留作未来工作.

**低精度训练.** LADDER 的设计并不限于推理. 低精度模型的训练和推理都需要系统与硬件提供低精度支持, 而训练中的反向计算也与前向计算相似. 低精度模型训练可以从两方面受益: 1) 使用更高效的低精度计算单元. 例如, A100 支持的 $W_{\mathrm{INT8}}A_{\mathrm{INT8}}$ 张量核心吞吐量是 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ 的 2×, $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ 张量核心则为 4×; 2) 低精度模型表示减少内存占用, 可以采用更大批大小, 从而改善硬件利用率. 低精度训练留作未来工作.

<span id="section-7"></span>

## 7 相关工作

**深度学习编译器与框架.** 现有多数深度学习编译器 [Ans24, Che18, Ma20, Pas19, Shi23a, Zha23h, Zhe20, Zhe23d, Zhu22] 主要优化 FP16, FP32 等主流数据类型上的算子或模型计算, 很少关注低精度数据类型. 不过, 其中不少优化与低精度计算互补. 例如, LADDER 使用 Roller [Zhu22] 推断高效的 tTile 配置, 使用 Welder [Shi23a] 做端到端图优化. SparTA [Zhe22e] 把模型剪枝和量化视作模型稀疏性, 整体优化稀疏模型的推理与训练; LADDER 可以提供高效低精度内核, 进一步改善性能. AMOS [Zhe22d] 优化了覆盖 FP16 和 INT8 类型的 TensorCore 计算, 但仅适用于 NVIDIA GPU. 相比之下, LADDER 是首个面向通用低精度计算优化, 并在不同 GPU 上支持通用自定义数据类型的编译器. ONNXRuntime [Onn24] 和 TensorRT [Ten24] 等深度学习库或框架支持部分推理用低位算子, 但要实现各种组合需要大量工作, 覆盖范围仍然有限. Triton [Til19] 和 TensorIR [Fen23] 等近期编译器允许用户直接编写 DNN 算子的计算流水线, 可以灵活指定各阶段的调度. 但这些编译器主要关注计算调度, 对自定义数据类型的数据调度支持很少, 后者正是 LADDER 的重点.

**面向特定模型的低精度优化.** 由于现有编译器和框架缺乏高效低精度支持, 许多工作针对特定工作负载开展低精度优化. 例如, 一些研究针对大语言模型 (LLM) 优化低精度类型上的量化和模型训练 [Fra22, Kwo23, Lin23d, Ma24, Dar23, Tou23a, Wan23, Les23]. 另一些工作 [Gul20, He16, Hua19d, Ma18, She23f] 则把 ShuffleNet, Conformer 等模型优化到 FP8 或 FP16 精度. 相比之下, LADDER 提供一种机制, 让用户更容易实现自定义数据类型和优化策略. 因此, 这些优化方法与 LADDER 互补, 可以在 LADDER 中实现, 或由 LADDER 自动优化.

<span id="section-8"></span>

## 8 结论

本文提出 LADDER, 首个面向 GPU 等加速器上的通用低精度计算进行优化的深度学习编译器. LADDER 公开通用类型系统 tType 和扩展张量表达式, 让用户可以方便地实现和表达深度学习中的新数据类型. 它还引入一组新的张量调度原语, 用于优化计算流水线中的张量存储, 访问和类型转换. LADDER 的分层硬件感知优化策略能够处理复杂的转换空间, 系统地支持多种低位自定义数据类型, 无需修改硬件即可提升现代加速器上的 DNN 计算性能. 这让模型设计者能够探索数据类型优化, 也为硬件厂商扩展多种精度格式的支持提供了灵活方案.

## 致谢

感谢匿名审稿人和匿名论文指导人提出的详尽建议.


