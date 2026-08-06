---
title: 'DeepSpeed Ulysses'
createTime: 2026/08/05 00:26:41
permalink: /papers/deepspeed-ulysses/
---

> [Sam Ade Jacobs](https://www.microsoft.com/en-us/research/people/samjacobs/), [Masahiro Tanaka](https://tohtana.github.io/), [Chengming Zhang](https://chengmingzh8.github.io/), [Minjia Zhang](https://minjiazhang.github.io/), [Shuaiwen Leon Song](https://sites.google.com/site/shuaiwenleonsongresearch/), [Samyam Rajbhandari](https://dblp.org/pid/115/9021), and [Yuxiong He](https://x.com/yuxionghe). 首次提交至 arXiv: September 25, 2023; 当前版本为 v2. [DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models](https://arxiv.org/abs/2309.14509). [原始 PDF](/paper/deepspeed-ulysses.pdf). [TeX 源码](https://export.arxiv.org/e-print/2309.14509). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

在典型的基于 Transformer 的大型语言模型 (LLM) 中的计算可以通过批量大小, 隐藏维度, 层数和序列长度来描述. 到目前为止, 加速 LLM 训练的系统工作主要集中在前三个维度: 针对批量大小的数据并行, 针对隐藏维度的张量并行以及针对模型深度或层数的流水线并行. 这些被广泛研究的并行形式并未针对长序列 Transformer 模型进行优化或专门设计. 鉴于长序列 LLM 的实际应用需求, 对序列并行的关注正在重新增长. 然而, 现有的序列并行工作受制于内存- 通信效率低下, 限制了其在长序列大模型上的可扩展性. 在本工作中, 我们提出了 DeepSpeed-Ulysses, 一种新颖, 可移植且高效的方法论, 用于实现具有极长序列长度的 LLM 训练的高效和可扩展性. DeepSpeed-Ulysses 的核心是沿序列维度划分输入数据, 并采用高效的全对全集合通信进行注意力计算. 理论通信分析表明, 当序列长度增加时, 其他方法会产生通信开销, 而 DeepSpeed-Ulysses 在序列长度和计算设备按比例增加时保持通信量恒定. 另外, 实验评估显示, DeepSpeed-Ulysses 在序列长度增加 4 倍的情况下, 比现有方法 SOTA 基线训练速度快 2.5 倍.

## 1 引言

使用长序列训练大型模型在从生成式 AI 到科学发现模型的各个领域都变得非常重要. 在生成式 AI 方面, 会话式 AI, 知识丰富的长文档摘要和视频生成都需要在空间和时间域中对长上下文进行推理. 例如, 多模态基础模型, 如同时处理语音, 图像和波形的模型, 需要对高维输入的长序列进行长上下文推理. 同样, 章节和书籍级别的摘要 (估计为数万至数十万字) 在会话式 AI 和抽象摘要任务中也非常重要 [Beltaa20, Kry22, Mosaic23], 并且已经显示出从长序列训练中受益 [Xiong23, Peng23, Touvrb23]. ChatGPT 的首次亮相 (以及随后类似的开源和“产品”级大语言模型品牌) 将聊天应用推向现代 AI 的前沿, 使聊天应用比以往任何时候都更相关. 处理长序列对于在聊天应用中支持更长的历史记录至关重要 [Touvrb23].

长序列长度对于科学人工智能同样至关重要, 它为更好地理解结构生物学, 医疗, 气候和天气预测打开了大门 [Nguyen23] 以及大分子模拟 [Zvyagi22]. 例如, 通过将大语言模型与基因序列相结合, 我们可以创建能够使用简单字母和极长序列 (人类基因组有 64 亿个字母) 学习基因组进化模式的语言模型 [Zvyagi22]. 在医疗保健中, 基于整个病人护理记录的诊断预测模型需要长序列的上下文 [Refe22, Gaoa21].

尽管长序列长度对于生成式人工智能和科学人工智能的重要性日益凸显, 但现有的大模型训练系统及其底层并行技术 (数据并行, 张量并行, 流水线并行, 序列并行) 在支持高效长序列训练方面能力有限. 现有并行方法面临两个主要挑战. 首先, 现有的并行方法如数据并行, 张量并行和流水线并行无法解决序列维度的扩展问题. 其次, 现有的序列并行方法由于内存与通信效率低下而效果不佳. 另外, 现有方法的可用性有限, 通常需要侵入式且易出错的代码重构.

在本文中, 我们介绍了 DeepSpeed-Ulysses (或称 Ulysses, 一本非常长的小说), 这是一种简单, 可移植且有效的方法, 用于实现高度高效和可扩展的超长序列 LLM 训练. DeepSpeed-Ulysses 沿序列维度将单个样本在参与 GPU 之间进行分区. 然后, 在注意力计算之前, 它对分区后的查询, 键和值使用全到全通信集合, 使每个 GPU 接收完整序列, 但仅针对非重叠的注意力头子集. 这样可以让参与的 GPU 并行计算不同注意力头的注意力. 最后, DeepSpeed-Ulysses 使用另一个全到全通信来沿注意力头汇总结果, 同时沿序列维度重新分区.

在这项工作中, 我们提出了 DeepSpeed-Ulysses 对推进长序列并行技术的以下贡献:

- •

DeepSpeed-Ulysses 能够训练比现有系统大 4 倍序列长度的 Transformer 模型, 同时支持超过一百万 token 的序列训练.

- •

与现有系统相比, 通信减少超过 10 倍, 从而带来高达 2.5 倍的吞吐量提升, 并实现持续吞吐量超过 175 TFlops/GPU (超过硬件峰值的 54%).

- •

完全通用且实现不可知的注意力机制: DeepSpeed 序列并行 (Ulysses) 支持稠密和稀疏注意力, 并且可以与高效的注意力实现 (例如 FlashAttention v2) 配合使用 [Daoa23].

- •

支持大规模模型训练: DeepSpeed 序列并行与 ZeRO-3 协同工作, 不仅支持大序列长度, 还支持超大模型规模.

- •

易于使用且可移植, 对现有训练框架仅需最少的代码修改.

在后续部分, 我们将提供背景和相关工作, 详述 DeepSpeed 序列并行核心设计, 通信复杂度分析, 实验评估及与现有工作的对比.

## 2 背景和相关工作

本节将简要概述 Transformer 架构, 加速 Transformer 训练的并行模式, 并讨论与我们方法密切相关的工作.

### 2.1 背景

本节简要介绍 Transformer 架构, 并重点说明深度神经网络的一般并行模式以及 Transformer 模型的特定并行模式. 随后将特别关注密切相关的工作.

#### 2.1.1 Transformer 架构

<span id="figure-01"></span>

![参见图注](./deepspeed-ulysses/figure-01.png)

**图 1.** 多头注意力 Transformer

如[图 1](#figure-01) 所示, 是典型多头注意力 Transformer 架构 [Vaswad17] 的构建模块示意图. 它由输入序列组成, 这些序列被投影为查询 (*Q*) , 键 (*K*) 和值 (*V*) 嵌入. *QKV* 通常是大小为 $N,b,d$ 的三维张量, 其中 $N$ 是序列长度, $b$ 是微批量大小, $d$ 是隐藏维度. $\mathrm{QKV}$ 张量被输入到注意力模块, 这是 Transformer 模型的核心组件. 注意力的输出是 Transformer 架构中多层感知机 (MLP) 或逐位置前馈模块的输入.

注意力块之后的 MLP 块会被重复多次, 以形成编码器, 解码器或编码器- 解码器 Transformer 网络.

#### 2.1.2 并行模式

数据并行 [Dean12] 是加速神经网络训练的事实方法, 并已广泛应用于不同的神经网络架构和应用中. 最简单形式的数据并行将输入数据按样本或批次维度进行划分, 同时在计算设备上复制模型参数. 当批次大小足够大以掩盖计算中的通信成本时, 数据并行是有效的. 然而, 当模型很大且跨设备复制模型参数在实际操作中不可行时, 它的效果是有限的. ZeRO [Rajbha20, Rajbha21] 优化通过将模型参数划分到可用的计算设备上来解决这个问题. 另外, 大批次已知会对模型质量产生影响 [Keskar16].

我们提出的方法与数据并行和 ZeRO 方法是正交的. 我们提出的方法可以与这两种方法结合使用. 另外, 通过利用序列并行来保持大型系统上的全局批量大小在合理范围内, 我们有效地减轻了大批量大小对模型收敛的影响. 在这方面, 序列并行具有两个作用. 首先, 序列并行可以加快相同 (已探索) 长序列长度的求解时间; 换句话说, 序列并行能根据额外计算资源的增加按比例减少迭代时间. 其次, 序列并行使得较长序列的训练或持续预训练成为可能, 其中训练上下文长度会随着时间逐渐增加 [Xiong23]. 考虑一个在 1024 个 GPU 上进行大规模训练的现实场景. 初始的探索性或预训练的 (代理) 大语言模型设置的序列长度为 8192 (8K), 每个 GPU 的微批量大小为 1 (因此全局大小为 800 万个 token). 为了提高预训练模型的质量, 一个简单的改动是将序列长度从 8K 改为 32K, 这将导致全局批量大小约为 3200 万. 然而, 由于对模型质量的负面影响, 增加全局批量大小并不可行. 因此, 序列并行作为一种系统优化技术显得非常有用, 无需费力进行超参数搜索. 在这种情况下, 序列并行允许将大批量大小在多个 GPU 上拆分, 而无需增加全局批量大小, 无论序列长度如何.

张量 [Shoeyd19] 和流水线并行 [Naraya19, Huang18, Naraya21] 是大规模训练的另外两种流行方法. 总体而言, 张量并行和流水线并行被称为模型并行, 主要针对大模型中的计算算子. 与数据并行相比, 当模型太大 (如许多大型语言模型中) 而无法在数据并行的不同计算单元中完全复制时, 就会使用模型并行. 张量并行在层内拆分计算算子 (即注意力和多层感知机), 而流水线并行则按深度 (按层) 拆分模型. 3D 并行 [Majumd20, Smitha22] 将数据并行, 张量并行和流水线并行结合, 以在与三个组成部分相比获得更高吞吐量, 但代价是需要大量代码重写和生产力开销 [Wang23].

### 2.2 相关工作

有关深度神经网络分布式训练方法的广泛概述和综述, 请参见 [Hoefle19]. 这些方法大致可以分为上述的数据并行和模型并行. 然而, 现有的所有并行方法在处理与极长序列相关的中间激活记忆开销方面都是有限的.

尽管最近关于序列并行的研究解决了内存开销问题, 但它们在通信效率方面仍然不足, 因此扩展能力有限. 与我们的工作类似, 所有现有的序列并行研究都是沿序列维度对输入数据进行划分, 但在划分哪些输入投影以及如何对注意力计算中的划分进行聚合和通信方面有所不同.

在 [Reff22] 中的作者 (以下称为 *ColAI-SP*) 引入了环形自注意力, 一种环状的通信集合, 其中查询投影是局部的, 而键和值的投影以环形方式传输以计算全局注意力, 从而使通信复杂度与消息大小呈线性关系, $M$. Megatron-LM 序列并行 [Korthi22] 方法与 Megatron 张量并行紧密结合. Megatron LM 沿序列维度划分序列, 并应用 allgather 和 reduce scatter 集合来聚合 *QKV* 投影以进行注意力计算. 通信复杂度分析显示, 与我们的方法不同, Megatron-LM 序列并行的通信量会随消息大小线性增加 ($M$), 而与计算设备数量无关. 另一方面, DeepSpeed-Ulysses 通过按消息大小或序列长度比例增加 GPU 来保持通信量一致, 更多细节请参见 [3.2](#S3.SS2).

[表 1](#table-01) 总结了 DeepSpeed-Ulysses 与其他现有方法的不同之处. DeepSpeed-Ulysses 在通信效率方面优于其他两种方法. 它还受益于利用 ZeRO [Rajbha20, Rajbha21] 优化在序列和数据并行组之间进行模型参数分区. DeepSpeed-Ulysses 支持不同类型的注意力, 并且易于使用. Megatron-LM 序列并行性与 Megatron-LM 张量并行性紧密集成, 限制了其内存效率和易用性. *ColAI-SP* 需要一种不同 (特定) 类型的注意力, 并且不易使用. 目前尚不清楚 *ColAI-SP* 环自注意力在其他注意力类型和机制上的泛化能力如何.

<span id="table-01"></span>

|方法|通信|激活|参数|注意力|易用性|
| --- | --- | --- | --- | --- | --- |
|复杂性|内存效率|内存效率|不相关|易用性||
|ColAI-SP [Reff22]|$O(M)$| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓|x|x|x|
|Megatron-SP [Korthi22]|$O(M)$| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓|x| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓|x|
|DS-Ulysses|$O(M/P)$| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓| \textpdfrender TextRenderingMode=FillStroke, LineWidth=. 75pt, ✓|

**表 1.** 我们的工作 (DS-Ulysses) 与其他序列并行方法的比较.

在稀疏 Transformer 中有相关工作, 特别关注全注意力的近似, 如稀疏注意力 [Childa19, Choroa20, Zaheer21, Beltaa20]. 近期也有关于单 GPU 内存和计算高效注意力的研究. 这类研究的一个流行例子是 Flash attention [Daod22, Daoa23], 它利用已知技术如分块和重计算以提高计算和内存效率. 这些工作与我们的工作是正交的, 并相应地被利用.

## 3 DeepSpeed-Ulysses 核心设计

### 3.1 系统设计

<span id="figure-02"></span>

![参见图注](./deepspeed-ulysses/figure-02.png)

**图 2.** DeepSpeed 序列并行 (DeepSpeed-Ulysses) 设计

[图 2](#figure-02) 展示了 DeepSpeed-Ulysses 的核心设计. 与已知的 Transformer 架构类似, 该设计由输入序列 *N* 组成, 并划分到 *P* 个可用设备上. 每个本地 *N/P* 分区被投影为查询 (*Q*), 键 (*K*) 和值 (*V*) 嵌入. 接下来, 通过参与计算设备之间高度优化的 all-to-all 集体操作, 将 (*QKV*) 嵌入收集到全局 *QKV* 中. all-to-all 集体操作之后, 是每个头的注意力计算, 形式如下:

$$
\mathrm{Outputcontext}=\mathrm{Softmax}((\mathrm{QK}^\top)/\sqrt{(}d))V\tag{1}
$$

注意力计算之后, 另一次 all-to-all 集体操作将注意力计算的输出上下文张量变换为序列 (*N/P*) 并行, 以便在 Transformer 层块的后续模块中进行后续运算 (MLP 矩阵乘法, 层归一化等).

### 3.2 通信分析

DeepSpeed-Ulysses 与现有的长序列方法的区别在于, 我们的整体通信量更小, 并且随着序列并行度的增加, 其整体可扩展性优于现有解决方案, 如下面的通信量分析所示:

在具有节点内 NVSwitch 互连和节点间 fat tree IB 拓扑的现代集群中, 对于大小为 *M* 的聚合消息在 *P* 个 GPU 上的 all-to-all 操作, 每条链路传输的通信量为 *M/P*. 对于隐藏层大小为 h, 序列长度为 N, 并行度为 P 的 Transformer 模型, DS-Sequence 在注意力计算之前对 *QKV* 投影执行 all-to-all 操作, 聚合消息大小为 *3Nh*, 并且对于每个 Transformer 层的输出上下文投影执行另一组大小为 *Nh* 的 all-to-all 操作. 因此, DeepSpeed 序列并行每条链路的聚合通信量为 *4Nh/P* (或复杂度为 *O(N/P)*). 注意, 当 *N* 和 *P* 按比例增加时, 该通信量保持不变.

相比之下, 现有的方法如 Megatron-LM 会产生随 N 线性增加的通信量, 而不考虑 P, 从而导致通信复杂度为 *O(N)*. 例如, Megatron-LM 对每一层 Transformer 都执行两次 all-gather, 消息量为 *Nh*, 以及两次 reduce-scatter, 消息量为 *Nh*. 然而, 当 *P >> 1* 时, 每次大小为 *M* 的 all-gather 和 reduce-scatter 的成本仍然是 *M*, 而不是 *M/P*. 因此, Megatron-LM 的序列并行在每个链路上的通信量为 *4Nh*, 是 DeepSpeed 序列并行的 *P* 倍. 这使得 DeepSpeed 序列并行能够在训练极长序列的同时, 实现比现有方法显著更高的训练效率. 我们的评估结果与该分析一致.

### 3.3 内存效率

虽然 DeepSpeed 序列并行在训练更长序列时可以减少激活内存, 但它不会影响模型状态所消耗的内存. 因此, 为了支持大语言模型的大序列长度训练, DeepSpeed 序列并行与 ZeRO-3 集成. ZeRO 冗余优化器第 3 阶段 (ZeRO-3) [Rajbha20, Rajbha21] 是一种用于训练大模型的内存优化技术. 与神经网络的经典数据并行训练中模型状态在各数据并行等级中被复制不同, ZeRO-3 通过在数据并行等级间分区模型状态来优化内存使用. 然而, 在序列并行中, 训练数据可以在批次 (样本) 和序列维度上进行考虑, 并将相关并行组结合形成 ZeRO 并行更大的组. 因此, 我们将 ZeRO-3 的分区扩展到数据并行和序列并行等级的组合. 换句话说, 在 DeepSpeed 序列并行中, ZeRO 将模型状态在序列和数据并行组之间进行分区, 并在需要时收集每个等级的分区 (allgather). 类似地, 梯度也会在数据并行和序列并行等级之间进行规约以进行参数更新. ZeRO 的支持在序列和数据维度上都允许巨大的内存节省, 并使得扩展不仅适用于大序列长度, 也适用于大模型.

### 3.4 通用且与注意力无关的解决方案

DeepSpeed 对分布式注意力模块的实现具有足够的通用性, 可以支持任何注意力类型: 例如自注意力, 交叉注意力, 因果注意力, 无论是其稠密还是稀疏版本, 以及支持局部注意力下长序列的各种优化内核, 例如不同版本的 FlashAttention. DeepSpeed-Ulysses 的通用特性源自其核心设计的模块化特性: 一种以注意力为中心的序列并行设计. 在注意力计算之前是 N/P 分区的序列并行, 注意力计算是基于每个头的全注意力的头并行, 只是头的数量更少, 因此注意力计算可以被替换为任何类型的注意力机制, 例如稠密注意力和各种形式的稀疏注意力.

## 4 评估

我们在 GPT [Radfob19] 上评估 DeepSpeed-Ulysses (DeepSpeed 序列), 这是一个支持多种自然语言处理任务的基础模型, 评估环境为最多 256 个 A100 GPU. 我们的评估内容分为五个方面: i) 序列长度可扩展性, ii) 稠密注意力的吞吐量及与现有系统的比较, iii) 稀疏注意力的吞吐量及与现有系统的比较, iv) 并行扩展研究, v) 深度序列并行的收敛性研究. 接下来我们将讨论并呈现这些类别的评估结果.

### 4.1 序列长度可扩展性

第一组实验是对序列长度进行强扩展性测试, 序列长度最高可达 100 万个标记, 使用 12 亿参数的 GPT 模型. 该评估的结果显示在 [图 3](#figure-03). DeepSpeed 序列并行性允许随着 GPU 数量增加线性地增加序列长度, 并且在适当的 GPU 数量下, 序列长度与计算吞吐量线性相关并保持相似的计算吞吐率.

<span id="figure-03"></span>

![参见图注](./deepspeed-ulysses/figure-03.png)

**图 3.** DeepSpeed 序列并行性在不同序列长度和 GPU 数量下的强扩展性评估

### 4.2 密集注意力评估

接下来, 我们对 DeepSpeed 序列并行性在 70 亿 (7B) 和 300 亿 (30B) 参数的 GPT 密集注意力模型上进行评估, 并分别与 Megatron-LM 在 32 和 64 个 A100 GPU 上的序列并行性进行比较. 这些评估结果显示在 [图 4](#figure-04) 和 [图 5](#figure-05).

我们比较了 7B 和 30B 模型在不同序列长度下运行时 DeepSpeed 序列并行性与 Megatron-LM 的性能. 为了评估, 我们选择了能够为 DeepSpeed 序列并行性和 Megatron-LM 提供最佳性能 (以吞吐量或 TFLOPs 衡量) 的序列并行度和微批量大小, 我们称之为最优 (批量大小- 序列长度) 配置. 对于 DeepSpeed 序列并行性, 我们始终为 7B 和 30B 模型分别使用 32 和 64 的 ZeRO 并行度.

[图 4](#figure-04) 和 [图 5](#figure-05) 显示 DeepSpeed 序列并行在可以运行的序列长度上始终优于 Megatron-LM. 另外, DeepSpeed 序列并行可以运行比 Megatron-LM 更长的序列. DeepSpeed 序列并行性能优势体现在两个方面: (1) 结合 ZeRO-3 的 DeepSpeed 序列并行由于内存优化, 可以容纳比 Megatron-LM 更多的样本, 从而带来更高的吞吐量; (2) DeepSpeed 序列并行相对于 Megatron-LM 序列并行中使用的 all-gather 通信, 受益于高效的 all-to-all 通信.

<span id="figure-04"></span>

![参见图注](./deepspeed-ulysses/figure-04.png)

**图 4.** 在 32 个 GPU 上对 7B 参数模型 (密集注意力) 进行 DeepSpeed-Ulysses 和 Megatron LM 的评估

<span id="figure-05"></span>

![参见图注](./deepspeed-ulysses/figure-05.png)

**图 5.** 在 64 个 GPU 上对 30B 参数模型 (密集注意力) 进行 DeepSpeed-Ulysses 和 Megatron LM 的评估

### 4.3 稀疏注意力评估

同样地, 我们在 70 亿和 300 亿参数稀疏注意力模型上评估 DeepSpeed 序列并行性, 并与 Megatron-LM 序列并行性进行基准测试. 我们的评估结果显示在 [图 6](#figure-06) 和 [图 7](#figure-07) 中. 我们观察到与稠密注意力实验类似的趋势. 在吞吐量方面, DeepSpeed 序列并行性相比 Megatron-LM 提高了超过 2 倍. 为了节省内存, 利用 ZeRO-3 的 DeepSpeed 序列并行性能够扩展至比 Megatron-LM 长 4 倍的序列长度.

在可以同时运行的序列长度下, DeepSpeed 序列并行性优于 Megatron-LM. 实际上, 目前 DeepSpeed 的吞吐量受到本地稀疏注意力实现的瓶颈影响, 因此随着序列长度增加, DeepSpeed 的吞吐量会下降. 我们预计, 随着未来我们改进本地稀疏注意力实现, DeepSpeed 与 Megatron-LM 之间的性能差距在更长序列长度下将进一步增加.

<span id="figure-06"></span>

![参见图注](./deepspeed-ulysses/figure-06.png)

**图 6.** 在拥有块稀疏注意力的 70 亿参数模型上对 DeepSpeed-Ulysses 与 Megatron LM 的评估 (32 张 GPU)

<span id="figure-07"></span>

![参见图注](./deepspeed-ulysses/figure-07.png)

**图 7.** 在拥有块稀疏注意力的 300 亿参数模型上对 DeepSpeed-Ulysses 与 Megatron LM 的评估 (64 张 GPU)

### 4.4 并行扩展研究

<span id="table-02"></span>

|序列长度|GPUs|时间 (毫秒)|TFLOPs|
| --- | --- | --- | --- |
|131072|64|32432.1333|165.526667|
|131072|128|17052.5143|157.41|
|131072|256|9886.7|136.09|

**表 2.** 固定序列长度的并行扩展研究

<span id="table-03"></span>

|序列长度|GPUs|时间 (毫秒)|TFLOPs|
| --- | --- | --- | --- |
|65536|64|9676.76|161.3626667|
|131072|128|17052.5143|157.41|
|262144|256|33486.5|147.4|

**表 3.** 不同序列长度的并行扩展研究

另外, 我们沿两个轴对 DeepSpeed-Ulysses 进行了并行扩展研究. 首先, 我们将序列长度固定为 131, 072 个标记, 并将 GPU 数量从 64 增加到 256. 其次, 我们按序列长度增加的比例增加 GPU 数量. 这些实验的结果分别显示在 [表 2](#table-02) 和 [表 3](#table-03) 中. 在这两种评估中, 我们使用 GPT-7B 密集模型, 全球批量大小为 8. 表中显示了迭代时间 (微秒) 以及每 GPU TFLOPs 测量的吞吐量. [表 2](#table-02) 可被理解为强扩展, 并显示随着 GPU 数量的增加, 执行时间几乎呈线性下降. 另一方面, [表 3](#table-03) 是一种弱扩展形式 (非传统意义上的弱扩展), 需注意注意力计算是序列长度的函数, 其复杂度为二次方. 换句话说, 随着序列长度的增加, 工作量呈二次增长.

通信开销可以归因于随着通信工作量 (即序列长度或 GPU 数量) 的增加, 吞吐量略有下降. 尽管存在这种开销, 我们在两个研究中观察到了在理论峰值 GPU 性能的高百分比下的良好扩展性. 这些良好的扩展结果表明 DeepSpeed-Ulysses 的并行效率较高.

### 4.5 收敛性研究

最后, [图 8](#figure-08) 显示了在 8 块 A100 GPU 上, 序列长度为 32K 的 13 亿参数 GPT 模型的收敛情况, 其中 DeepSpeed-Ulysses 和 Megatron-LM 的序列并行度均设置为 4. 对于 DeepSpeed 序列并行, 我们评估了不同 ZeRO 阶段下的收敛情况. DeepSpeed 序列并行是一种纯粹的系统优化技术, 可以实现长序列 Transformer 模型的训练, 因此不会对训练出的模型质量产生 (负面) 影响, 这一论断通过实验得到验证, 并在 [图 8](#figure-08) 中展示.

<span id="figure-08"></span>

![参见图注](./deepspeed-ulysses/figure-08.png)

**图 8.** 使用不同 ZeRO 内存优化阶段的 DeepSpeed-Ulysses 收敛评估

## 5 结论

总之, 我们提出了一种内存和通信高效的 DeepSpeed 序列作为长序列大规模 Transformer 训练的使能技术. DeepSpeed 序列实现了跨 GPU (以及其他 AI 加速器) 的序列并行, 将序列并行化应用于 Transformer 模型的所有组件, 包括对 SOTA Flash (稠密和稀疏) 注意力的流线型支持. 使用 DeepSpeed 序列进行训练允许模型尺寸和序列长度几乎无限制增长, 不受单 GPU 内存限制, 并能以接近峰值计算性能的高比例运行.
