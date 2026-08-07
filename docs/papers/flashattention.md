---
title: 'FlashAttention'
createTime: 2026/08/05 00:15:31
permalink: /papers/flashattention/
pageClass: paper-reading
---

> [Tri Dao](https://tridao.me/), [Daniel Y. Fu](https://danfu.org/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Atri Rudra](http://www.cse.buffalo.edu/~atri/), and [Christopher Ré](http://cs.stanford.edu/people/chrismre/). 首次提交至 arXiv: May 27, 2022; 当前版本为 v2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). [原始 PDF](/paper/flashattention.pdf). [TeX 源码](https://export.arxiv.org/e-print/2205.14135). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

变压器在处理长序列时速度慢且占用大量内存, 因为自注意力的时间和内存复杂度与序列长度呈平方关系. 近似注意力方法尝试通过牺牲模型质量来降低计算复杂度, 但通常无法实现实际时间上的加速. 我们认为一个缺失的原则是使注意力算法具备 IO 感知能力——即考虑 GPU 内存各层之间的读写操作. 我们提出了 FlashAttention, 一种 IO 感知的精确注意力算法, 通过分块技术减少 GPU 高带宽内存 (HBM) 和 GPU 片上 SRAM 之间的内存读写次数. 我们分析了 FlashAttention 的 IO 复杂度, 显示其所需的 HBM 访问次数比标准注意力更少, 并且对于一定范围的 SRAM 大小是最优的. 我们还将 FlashAttention 扩展到块稀疏注意力, 从而产生一种比现有任何近似注意力方法都更快的近似注意力算法. FlashAttention 在训练变压器时比现有基线速度更快: 在 BERT-large (序列长度 512) 上端到端实际时间加速 15%, 超过 MLPerf 1.1 训练速度记录, 在 GPT-2 (序列长度 1K) 上加速 3 $\times$, 在 long-range arena (序列长度 1K-4K) 上加速 2.4 $\times$. FlashAttention 和块稀疏 FlashAttention 使变压器能够处理更长的上下文, 从而产生更高质量的模型 (GPT-2 困惑度提升 0.7, 长文档分类提升 6.4 分) 并带来全新能力: 首批在 Path-X 挑战 (序列长度 16K, 准确率 61.4%) 和 Path-256 (序列长度 64K, 准确率 63.1%) 上实现优于随机性能的变压器.

## 1 引言

Transformer 模型 [Vaswan17] 已成为自然语言处理和图像分类等应用中最广泛使用的架构. Transformer 模型变得越来越大 [Brown20], 越来越深 [Wang22], 但为它们提供更长的上下文仍然很困难 [Tay20], 因为其核心的自注意力模块在时间和内存复杂度上随序列长度呈二次增长. 一个重要的问题是, 使注意力更快, 更节省内存是否能帮助 Transformer 模型解决长序列的运行时间和内存挑战.

许多近似注意力方法旨在降低注意力的计算和内存需求. 这些方法包括稀疏近似 [Kitaev20, Roy21], 低秩近似 [Wang20, Kathar20, Chorom20] 及其组合 [Beltag20, Zaheer20, Chen21]. 尽管这些方法将计算需求降低到与序列长度呈线性或近线性关系, 但许多方法在实际运行时间上并未显示出相对于标准注意力的加速, 并且尚未被广泛采用. 一个主要原因是它们关注 FLOP 的减少 (这可能与实际运行时间无关) 并且往往忽略了内存访问 (IO) 带来的开销.

<span id="figure-01"></span>

![参见图注](./flashattention/figure-01.png)

**图 1.** 左: FlashAttention 使用分块处理来防止在 (相对) 较慢的 GPU HBM 上实现大型 $N\times N$ 注意力矩阵 (虚线框). 在外层循环 (红色箭头) 中, FlashAttention 遍历 $\mathbf{K}$ 和 $\mathbf{V}$ 矩阵的块并将它们加载到快速的片上 SRAM 中. 在每个块中, FlashAttention 遍历 $\mathbf{Q}$ 矩阵的块 (蓝色箭头), 将它们加载到 SRAM, 并将注意力计算的输出写回 HBM. 右: 在 GPT-2 上, 相较于 PyTorch 注意力实现的加速效果. FlashAttention 不将大型 $N\times N$ 注意力矩阵读写到 HBM, 从而在注意力计算上实现了 7.6 $\times$ 的加速.

在本文中, 我们认为一个缺失的原则是使注意力算法具备 IO 感知能力 [Vitter88]——也就是说, 要仔细考虑对不同层级的快速和慢速存储器的读写 (例如, 在快速 GPU 片上 SRAM 和相对较慢的 GPU 高带宽存储器 (HBM [Jiaa18], [图 1](#figure-01) 左)之间). 在现代 GPU 上, 计算速度已经超过了内存速度 [NVIDIA17, NVIDIA20, NVIDIA22], 而变压器中的大多数操作都受到内存访问的瓶颈影响 [Ivanov21]. 当读取和写入数据可能占据运行时间的大部分时, IO 感知算法对于类似的内存受限操作至关重要——例如数据库连接 [Ramakr03], 图像处理 [RaganK13], 数值线性代数 [Blackf02] 等等 [Willia09, Patter03]. 然而, 诸如 PyTorch 和 Tensorflow 之类的常见 Python 深度学习接口并不允许对内存访问进行细粒度控制.

我们提出了 FlashAttention, 一种新的注意力算法, 它可以在大大减少内存访问的情况下计算精确的注意力. 我们的主要目标是避免将注意力矩阵读写到 HBM. 这需要 (i) 在不访问整个输入的情况下计算 softmax 归约 (ii) 不为反向传播存储大的中间注意力矩阵. 我们应用了两种成熟的技术来解决这些挑战. (i) 我们重构了注意力计算, 将输入拆分为块, 并对输入块进行多次遍历, 从而逐步执行 softmax 归约 (也称为分块). (ii) 我们在前向传播中存储 softmax 归一化因子, 以便在反向传播中快速在芯片上重新计算注意力, 这比从 HBM 读取中间注意力矩阵的标准方法更快. 我们在 CUDA 中实现了 FlashAttention, 以实现对内存访问的精细控制, 并将所有注意力操作融合到一个 GPU 内核中. 即使由于重新计算导致 FLOPs 增加, 我们的算法仍然运行更快 (在 GPT-2 [Radfor19] 上最高可达 7.6 倍, [图 1](#figure-01) 右侧) 并且使用的内存更少——与序列长度线性相关——这得益于大幅减少的 HBM 访问量.

我们分析了 FlashAttention 的 IO 复杂度 [Vitter88], 证明它需要 $O(N^{2}d^{2}M^{-1})$ 次 HBM 访问, 其中 $d$ 是头部维度, $M$ 是 SRAM 的大小, 相比之下标准注意力需要 $\Omega(\mathrm{Nd}+N^{2})$ 次访问. 对于典型的 $d$ 和 $M$ 值, FlashAttention 所需的 HBM 访问次数比标准注意力少很多倍 (如 [图 2](#figure-02) 所示, 最多少 9 $\times$ 次). 另外, 我们提供了一个下界, 表明没有任何精确注意力算法可以在所有 SRAM 大小上渐近地减少 HBM 访问次数.

我们还表明, FlashAttention 可以作为实现近似注意力算法潜力的有用原语, 克服它们在内存访问开销上的问题. 作为概念验证, 我们实现了块稀疏 FlashAttention, 一种稀疏注意力算法, 其速度比 FlashAttention 快 2-4 $\times$, 序列长度可扩展到 64k. 我们证明块稀疏 FlashAttention 的 IO 复杂度比 FlashAttention 更优, 其提升比例与稀疏率成正比. 我们在 [第 5 节](#S5) 讨论了进一步扩展到其他操作 (多 GPU 上的注意力, 核回归, 块稀疏矩阵乘法). 我们将 FlashAttention 开源, 以便更容易在此原语基础上进行开发. [+1]

我们通过实证验证了 FlashAttention 可以通过建模更长的上下文来加快模型训练速度并提升模型质量. 我们还对比了 FlashAttention 和块稀疏 FlashAttention 的运行时间和内存占用情况, 相对于之前的注意力实现进行了基准测试.

- 更快的模型训练. FlashAttention 可以在实际时间上更快地训练 Transformer 模型. 我们在 MLPerf 1.1 [Mattso20] 中训练 BERT-large (序列长度 512) 的速度比训练速度记录快 15%, GPT2 (序列长度 1K) 比 HuggingFace [Wolf20] 和 Megatron-LM [Shoeyb19] 的基线实现快 3 $\times$ 倍, 以及 long-range arena (序列长度 1K-4K) 比基线快 2.4 $\times$ 倍.
- 更高质量的模型. FlashAttention 可将 Transformer 扩展到更长的序列, 从而提高其质量并启用新的功能. 我们在 GPT-2 上观察到困惑度 (perplexity) 提升 0.7, 在长文档分类上通过建模更长序列提升 6.4 个点 [Dai22]. FlashAttention 使得第一个在 Path-X [Tay20] 挑战中仅通过使用更长序列长度 (16K) 就能达到优于随机表现的 Transformer 成为可能. 块稀疏 FlashAttention 可以让 Transformer 扩展到更长序列 (64K), 从而诞生了首个在 Path-256 上能够达到优于随机表现的模型.
- 注意力基准测试. FlashAttention 在常见的序列长度范围从 128 到 2K 时, 比标准的注意力实现快多达 3 $\times$ 倍, 并可扩展到 64K. 对于序列长度达到 512, FlashAttention 在速度和内存效率上都优于任何现有的注意力方法, 而当序列长度超过 1K 时, 一些近似注意力方法 (例如 Linformer) 开始变得更快. 另一方面, 块稀疏 FlashAttention 比我们所知道的所有现有近似注意力方法都更快.

<span id="S2"></span>

## 2 背景

我们提供了一些关于现代硬件 (GPU) 上常见深度学习操作性能特性的背景信息. 我们还描述了注意力的标准实现.

<span id="S2.SS1"></span>

### 2.1 硬件性能

我们在这里关注 GPU. 其他硬件加速器上的性能类似 [Jouppi17, Jiaa19].

GPU 内存层次结构. GPU 内存层次结构 ([图 1](#figure-01) 左侧) 由多种不同大小和速度的内存组成, 越小的内存速度越快. 例如, A100 GPU 具有 40-80GB 的高带宽内存 (HBM), 带宽为 1.5-2.0TB/s, 以及每个 108 个流多处理器配备的 192KB 芯片上 SRAM, 估计带宽约为 19TB/s [Jiaa18, Sandt21]. 芯片上 SRAM 的速度比 HBM 快一个数量级, 但容量却小几个数量级. 随着计算速度相对于内存速度的增加 [NVIDIA17, NVIDIA20, NVIDIA22], 操作越来越受到内存 (HBM) 访问的瓶颈限制. 因此, 利用快速的 SRAM 变得更加重要.

执行模型. GPU 拥有大量线程来执行一个操作 (称为内核). 每个内核从 HBM 加载输入到寄存器和 SRAM, 进行计算, 然后将输出写回 HBM.

性能特性. 根据计算和内存访问的平衡, 操作可被分类为计算受限或内存受限. 这通常通过*算术强度* [Willia09] 来衡量, 即每字节内存访问的算术操作数.

1. 计算受限: 操作所需的时间由算术操作的数量决定, 而访问 HBM 的时间要小得多. 典型的例子是内维度较大的矩阵乘法和通道数较多的卷积运算.
2. 内存受限: 操作所需的时间由内存访问次数决定, 而计算所花费的时间要小得多. 例子包括大多数其他操作: 按元素操作 (例如, 激活, dropout) 和归约操作 (例如, 求和, softmax, 批量归一化, 层归一化).

内核融合. 加速内存受限操作最常见的方法是内核融合: 如果对同一输入应用多个操作, 可以只从 HBM 加载一次输入, 而不是对每个操作都加载多次. 编译器可以自动融合许多按元素操作 [Refa20, Paszke19, Sabne20]. 然而, 在模型训练的上下文中, 中间值仍需写入 HBM 以便在反向传播中使用, 这会降低简单内核融合的效果.

<span id="S2.SS2"></span>

### 2.2 标准注意力实现

给定输入序列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 其中 $N$ 是序列长度, $d$ 是头维度, 我们希望计算注意力输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\begin{aligned}
\mathbf{S} &= \mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\\
\mathbf{O} &= \mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
\end{aligned}
$$

其中 $\mathrm{softmax}$ 是按行应用的.

标准注意力实现将矩阵 $\mathbf{S}$ 和 $\mathbf{P}$ 物化到 HBM, 这需要 $O(N^{2})$ 内存. 通常 $N\gg d$ (例如, 对于 GPT2, $N=1024$ 和 $d=64$). 我们在 [算法](#alg0) 中描述了标准注意力实现. 由于部分或大部分操作是受内存限制的 (例如, softmax), 大量的内存访问会导致实时时间变慢.

这一问题因应用于注意力矩阵的其他逐元素操作而加剧, 例如应用于 $\mathbf{S}$ 的掩码或应用于 $\mathbf{P}$ 的 dropout. 因此, 已经有许多尝试将几个逐元素操作融合, 例如将掩码与 softmax [Shoeyb19] 融合.

在 [第 3.2 节](#S3.SS2) 中, 我们将展示标准注意力实现执行的 HBM 访问随着序列长度 $N$ 呈二次增长. 我们还比较了标准注意力和我们的方法 (FlashAttention) 的 FLOPs 数量和 HBM 访问次数.

<span id="alg0"></span>

**算法 0: 标准注意力实现**

- **输入:** 矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 在 HBM 中.
- <span id="alg0.l1"></span> 从 HBM 分块加载 $\mathbf{Q},\mathbf{K}$, 计算 $\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$, 将 $\mathbf{S}$ 写入 HBM.
- <span id="alg0.l2"></span> 从 HBM 读取 $\mathbf{S}$, 计算 $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$, 将 $\mathbf{P}$ 写入 HBM.
- <span id="alg0.l3"></span> 从 HBM 分块加载 $\mathbf{P}$ 和 $\mathbf{V}$, 计算 $\mathbf{O}=\mathbf{P}\mathbf{V}$, 将 $\mathbf{O}$ 写入 HBM.
- **返回:** $\mathbf{O}$.

<span id="S3"></span>

## 3 FlashAttention: 算法, 分析与扩展

我们展示了如何在减少 HBM 读/写次数且无需存储大型中间矩阵以便进行反向传播的情况下计算精确注意力. 这产生了一种既节省内存又在实际时间上更快的注意力算法. 我们分析了它的 IO 复杂度, 显示出与标准注意力相比, 我们的方法所需的 HBM 访问次数大大减少. 我们进一步展示了 FlashAttention 作为一个有用原语的潜力, 通过扩展它以处理块稀疏注意力.

这里我们重点介绍正向传播以便说明; [附录 B](#A2) 包含了反向传播的详细信息.

<span id="S3.SS1"></span>

### 3.1 使用平铺和重新计算的高效注意力算法

给定在 HBM 中的输入 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 我们的目标是计算注意力输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$, 并将其写入 HBM. 我们的目标是减少 HBM 访问次数 (至 $N$ 的亚二次级别).

我们应用两种已建立的技术 (分块, 重计算) 来克服在亚二次 HBM 访问中计算精确注意力的技术挑战. 我们在 [算法 1](#alg1) 中描述了这一点. 其主要思想是将输入 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 拆分为块, 从慢速 HBM 加载到快速 SRAM, 然后针对这些块计算注意力输出. 通过在相加前按正确的归一化系数缩放每个块的输出, 我们最终可以得到正确的结果.

分块. 我们按块计算注意力. Softmax 将 $\mathbf{K}$ 的列耦合在一起, 因此我们分解带缩放 [Gimels18, Kitaev20, Staats21] 的大 Softmax. 为了数值稳定, 向量 $x\in\mathbb{R}^{B}$ 的 Softmax 计算如下:

$$
\begin{aligned}
m(x) &:= \max_{i} x_{i},\\
f(x) &:= \begin{bmatrix}e^{x_{1}-m(x)} & \ldots & e^{x_{B}-m(x)}\end{bmatrix},\\
\ell(x) &:= \sum_{i} f(x)_{i},\\
\mathrm{softmax}(x) &:= \frac{f(x)}{\ell(x)}.
\end{aligned}
$$

对于向量 $x^{(1)},x^{(2)}\in\mathbb{R}^{B}$, 我们可以将连接的 $x=\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix}\in\mathbb{R}^{2B}$ 的 Softmax 分解如下:

$$
\begin{aligned}
m(x) &= m\left(\begin{bmatrix}x^{(1)} & x^{(2)}\end{bmatrix}\right)
 = \max\left(m(x^{(1)}),m(x^{(2)})\right),\\
f(x) &= \begin{bmatrix}
e^{m(x^{(1)})-m(x)}f(x^{(1)}) & e^{m(x^{(2)})-m(x)}f(x^{(2)})
\end{bmatrix},\\
\ell(x) &= \ell\left(\begin{bmatrix}x^{(1)} & x^{(2)}\end{bmatrix}\right)
 = e^{m(x^{(1)})-m(x)}\ell(x^{(1)})
 + e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\\
\mathrm{softmax}(x) &= \frac{f(x)}{\ell(x)}.
\end{aligned}
$$

因此, 如果我们跟踪一些额外的统计信息 ($m(x),\ell(x)$), 我们可以一次计算一块的 softmax. [+2] 因此我们将输入 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 分成块 ([算法 1](#alg1) 第 3 行), 计算 softmax 值及额外统计信息 ([算法 1](#alg1) 第 10 行), 并合并结果 ([算法 1](#alg1) 第 12 行).

重计算. 我们的目标之一是不存储 $O(N^{2})$ 中间值用于反向传播. 反向传播通常需要矩阵 $\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$ 来计算相对于 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的梯度. 然而, 通过存储输出 $\mathbf{O}$ 和 softmax 归一化统计信息 $(m,\ell)$, 我们可以在反向传播中从 SRAM 中的 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 块轻松地重新计算注意力矩阵 $\mathbf{S}$ 和 $\mathbf{P}$. 这可以视为一种选择性梯度检查点 [Walthe08, Chen16]. 虽然已经有人建议通过梯度检查点来减少所需的最大内存量 [Staats21], 但我们所知的所有实现都必须在速度和内存之间进行权衡. 相反, 即使 FLOPs 更多, 由于减少了 HBM 访问, 我们的重计算加快了反向传播的速度 ([图 2](#figure-02)). 完整的反向传播描述见 [附录 B](#A2).

实现细节: 内核融合. 平铺 (Tiling) 使我们能够在一个 CUDA 内核中实现算法, 从 HBM 加载输入, 执行所有计算步骤 (矩阵乘法, softmax, 可选的掩码和 dropout, 矩阵乘法), 然后将结果写回 HBM (掩码和 dropout 在 [附录 B](#A2) 中). 这避免了对输入和输出在 HBM 之间重复读写.

<span id="alg1"></span>

**算法 1: FlashAttention**

- **输入:** 在 HBM 中的矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 片上 SRAM 大小为 $M$.
- 设置块大小 $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- <span id="alg1.l2"></span> 在 HBM 中初始化 $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$.
- <span id="alg1.l3"></span> 将 $\mathbf{Q}$ 分成 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 个块 $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$, 每块大小为 $B_{r}\times d$, 并将 $\mathbf{K},\mathbf{V}$ 分成 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 个块 $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ 和 $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, 每块大小为 $B_{c}\times d$.
- 将 $\mathbf{O}$ 分成 $T_{r}$ 块, 每块 $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$, 大小为 $B_{r}\times d$, 将 $\ell$ 分成 $T_{r}$ 块, 每块 $\ell_{i},\dots,\ell_{T_{r}}$, 大小为 $B_{r}$, 将 $m$ 分成 $T_{r}$ 块, 每块 $m_{1},\dots,m_{T_{r}}$, 大小为 $B_{r}$.
- <span id="alg1.l5"></span> **对** $1\leq j\leq T_{c}$ **执行:**
  - <span id="alg1.l6"></span> 从 HBM 将 $\mathbf{K}_{j},\mathbf{V}_{j}$ 加载到片上 SRAM.
  - **对** $1\leq i\leq T_{r}$ **执行:**
    - <span id="alg1.l8"></span> 从 HBM 将 $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ 加载到片上 SRAM.
    - <span id="alg1.l9"></span> 在芯片上, 计算 $\mathbf{S}_{\mathrm{ij}}=\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - <span id="alg1.l10"></span> 在芯片上, 计算 $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (逐点) 和 $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
    - 在芯片上, 计算 $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
    - <span id="alg1.l12"></span> 将 $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j})$ 写入 HBM.
    - 将 $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ 写入 HBM.
- **返回:** $\mathbf{O}$.

我们展示了 FlashAttention 的正确性, 运行时间和内存需求 (证明见 [附录 C](#A3)).

<span id="Thmtheorem1"></span>

###### 定理 1.

[算法 1](#alg1) 返回 $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$, 计算量为 $O(N^{2}d)$ FLOPs, 并且要求除输入和输出外额外的 $O(N)$ 内存.

<span id="S3.SS2"></span>

### 3.2 分析: FlashAttention 的 IO 复杂度

我们分析了 FlashAttention 的 IO 复杂度, 显示与标准注意力机制相比, 其 HBM 访问显著减少. 我们还提供了下界, 证明没有任何精确的注意力算法能够在所有 SRAM 大小下渐近地减少 HBM 访问. 证明见 [附录 C](#A3).

<span id="Thmtheorem2"></span>

###### 定理 2.

令 $N$ 为序列长度, $d$ 为头维度, $M$ 为 SRAM 的大小, 并设 $d\leq M\leq \mathrm{Nd}$. 标准注意力 ([算法](#alg0)) 需要 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问, 而 FlashAttention ([算法 1](#alg1)) 需要 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问.

对于典型的 $d$ (64-128) 和 $M$ (约 100KB), $d^{2}$ 比 $M$ 小很多, 因此 FlashAttention 所需的 HBM 访问次数比标准实现少很多. 这既带来了更快的执行速度, 也降低了内存占用, 我们在 [第 4.3 节](#S4.SS3) 中进行了验证.

证明的主要思想是, 给定 SRAM 大小为 $M$, 我们可以将 $\Theta(M)$ 个 $\mathbf{K},\mathbf{V}$ 块装入片上 SRAM ([算法 1](#alg1) 第 6 行). 对于每个 $\mathbf{K}$ 和 $\mathbf{V}$ 块, 我们遍历所有 $\mathbf{Q}$ 块 ([算法 1](#alg1) 第 8 行) 计算中间值, 从而需要 $\Theta(\mathrm{NdM}^{-1})$ 次遍历 $\mathbf{Q}$. 每次遍历加载 $\Theta(\mathrm{Nd})$ 个元素, 相当于 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问. 我们同样证明, 标准注意力的反向传播需要 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问, 而 FlashAttention 的反向传播需要 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问 ([附录 B](#A2)).

我们证明了一个下界: 在计算精确注意力时, 对于所有 $M$ (SRAM 大小) 的值, 无法渐近地改善 HBM 访问次数.

<span id="Thmtheorem3"></span>

###### 命题 3.

设 $N$ 为序列长度, $d$ 为头部维度, $M$ 为具有 $d\leq M\leq \mathrm{Nd}$ 的 SRAM 大小. 在 $[d,\mathrm{Nd}]$ 范围内的所有 $M$ 中, 不存在一种算法可以通过 $o(N^{2}d^{2}M^{-1})$ 次 HBM 访问来精确计算注意力.

该证明基于如下事实: 对于 $M=\Theta(\mathrm{Nd})$, 任何算法都必须执行 $\Omega(N^{2}d^{2}M^{-1})=\Omega(\mathrm{Nd})$ 次 HBM 访问. 这种对 $M$ 子范围的下界在流算法文献 [Woodru04] 中很常见. 我们将以 $M$ 为参数证明 [Grohe06] 的复杂性下界作为未来令人兴奋的工作.

我们验证了 HBM 访问次数是决定注意力运行时间的主要因素. 在 [图 2](#figure-02) (左图) 中, 我们看到尽管 FlashAttention 的 FLOP 数相比标准注意力更高 (由于反向传递中的重计算), 但是它的 HBM 访问次数远少于标准注意力, 从而导致运行时间更快. 在 [图 2](#figure-02) (中图) 中, 我们改变了 FlashAttention 的块大小 $B_{c}$, 这会导致不同数量的 HBM 访问, 并测量前向传递的运行时间. 随着块大小增加, HBM 访问次数减少 (因为我们对输入的访问次数更少), 运行时间也随之减少. 对于足够大的块大小 (超过 256), 运行时将受到其他因素 (例如算术运算) 的瓶颈. 另外, 更大的块大小将无法适应较小的 SRAM 容量.

<span id="figure-02"></span>

![参见图注](./flashattention/figure-02.png)

**图 2.** 左图: 在 A100 GPU 上, GPT-2 medium (序列长度 1024, 头维度 64, 16 个头, 批量大小 64) 标准注意力和 FlashAttention 的前向及反向运行时间. HBM 访问是影响运行时间的主要因素. 中图: 在 A100 GPU 上, FlashAttention (序列长度 1024, 头维度 64, 16 个头, 批量大小 64) 的前向运行时间. 较少的 HBM 访问导致更快的运行时间, 但有一定限度. 右图: 块稀疏 FlashAttention (序列长度 4K) 的运行时间比 FlashAttention 快, 其加速比例与稀疏度成正比.

<span id="S3.SS3"></span>

### 3.3 拓展: 块稀疏 FlashAttention

我们将 FlashAttention 扩展到近似注意力: 我们提出了块稀疏 FlashAttention, 其 IO 复杂度比 FlashAttention 小一个与稀疏性成比例的因子.

给定输入 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 和一个掩码矩阵 $\tilde{\mathbf{M}}\in\{0,1\}^{N\times N}$, 我们希望计算:

$$
\begin{aligned}
\mathbf{S} &= \mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S}\odot\mathbf{1}_{\tilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\\
\mathbf{O} &= \mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
\end{aligned}
$$

其中 $(\mathbf{S}\odot\mathbf{1}_{\tilde{\mathbf{M}}})_{\mathrm{kl}}=\mathbf{S}_{\mathrm{kl}}$ 如果 $\tilde{\mathbf{M}}_{\mathrm{kl}}=1$, 并且 $-\infty$ 如果 $\tilde{\mathbf{M}}_{\mathrm{kl}}=0$. 我们要求 $\tilde{\mathbf{M}}$ 具有块状形式: 对于某些块大小 $B_{r},B_{c}$, 对于所有 $k,l$, $\tilde{\mathbf{M}}_{k,l}=\mathbf{M}_{\mathrm{ij}}$ 具有 $i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor$, 且存在某些 $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$.

给定一个预定义的块稀疏掩码 $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$, 我们可以轻松地将 [算法 1](#alg1) 适配为只计算注意力矩阵中的非零块. 该算法与 [算法 1](#alg1) 相同, 只是我们跳过零块. 我们在 [附录 B](#A2) 中重现了 [算法 5](#alg5) 的算法描述.

我们还分析了块稀疏 FlashAttention 的 IO 复杂度.

<span id="Thmtheorem4"></span>

###### 命题 4.

设 $N$ 为序列长度, $d$ 为头部维度, $M$ 为 SRAM 的大小, 其为 $d\leq M\leq \mathrm{Nd}$. 块稀疏 FlashAttention ([算法 5](#alg5)) 需要 $\Theta(\mathrm{Nd}+N^{2}d^{2}M^{-1}s)$ 次 HBM 访问, 其中 $s$ 是块稀疏掩码中非零块的比例.

我们看到, 应用块稀疏会通过稀疏性直接改善 IO 复杂度中较大的项. 对于大序列长度 $N$, $s$ 通常采用两种典型设置: $N^{-1/2}$ [Child19] 和 $N^{-1}\log N$ [Zaheer20, Beltag20, Daoa22], 对应的 IO 复杂度分别为 $\Theta(N\sqrt{N})$ 和 $\Theta(N\log N)$. 在下游实验中, 我们使用固定的蝴蝶稀疏模式 [Daoa22], 已显示能够近似任意稀疏 [Dao20].

在 [图 2](#figure-02) (右) 中, 我们验证了随着稀疏性的增加, 块稀疏 FlashAttention 的运行时间成比例改善. 在 LRA 基准测试中, 块稀疏 FlashAttention 达到 2.8 $\times$ 的加速, 同时性能与标准注意力 ([第 4 节](#S4)) 相当.

<span id="S4"></span>

## 4 实验

我们评估了使用 FlashAttention 训练 Transformer 模型的影响. 我们验证了关于训练时间和模型准确性的两个声明, 并报告了注意力运行时间和内存基准.

- 训练速度. FlashAttention 在 BERT 上的表现比 MLPerf 1.1 [Mattso20] 速度记录快 15%, 并且在 GPT-2 上比 HuggingFace [Wolf20] 快 3 $\times$, 比 Megatron $1.8\times$ 快 [Shoeyb19], 相比标准 Transformer 更快. FlashAttention 在长距离竞赛 (LRA) 基准测试中提升了 2.4 $\times$ 的速度.
- 质量. FlashAttention 可将 Transformer 扩展到更长的序列, 从而产生更高的质量. FlashAttention 在上下文长度为 4K 时训练 GPT-2, 比 Megatron 在上下文长度为 1K 时训练 GPT-2 更快, 同时困惑度提高了 0.7. 建模更长的序列在两个长文档分类任务中带来了 6.4 个点的提升. 最后, FlashAttention 创造了首个在挑战性 Path-X 任务 (序列长度 16K) 上达到随机以上表现的 Transformer, 而块稀疏 FlashAttention 创造了我们所知首个在 Path-256 (序列长度 64K) 上达到随机以上表现的序列模型.
- 注意力基准. 我们基于序列长度测量了 FlashAttention 和块稀疏 FlashAttention 的运行时间和内存性能. 我们确认 FlashAttention 的内存占用与序列长度成线性关系, 并且在常见序列长度 (最大 2K) 下比标准注意力快 3 $\times$. 我们确认块稀疏 FlashAttention 的运行时间随序列长度线性增长, 并且比所有现有的近似注意力基线更快.

额外的实验细节在 [附录 E](#A5) 中提供.

<span id="S4.SS1"></span>

### 4.1 使用 FlashAttention 的更快模型

##### BERT.

FlashAttention 提供了我们所知的最快的单节点 BERT 训练速度. 我们在维基百科数据上使用 FlashAttention 训练了一个 BERT-large [Devlin19] 模型. [表 1](#table-01) 将我们的训练时间与 Nvidia 的实现进行比较, 该实现曾在 MLPerf 1.1 中创下训练速度记录 [Mattso20]. 我们的实现快了 15%.

<span id="table-01"></span>

![论文原表 1](./flashattention/table-01.png)

**表 1.** 从 MLPerf 基准提供的相同初始化开始训练 BERT-large, 以达到掩码语言建模的目标准确率 72.0%. 在 8 个 A100 GPU 上平均运行 10 次.

##### GPT-2.

FlashAttention 在大型 OpenWebtext 数据集 [Gokasl19]上对 GPT-2 [Radfor19] 的训练时间比广泛使用的 HuggingFace [Wolf20] 和 Megatron-LM [Shoeyb19] 实现更快. [表 2](#table-02) 显示, 与 Huggingface 相比端到端速度提升高达 3 $\times$, 与 Megatron-LM 相比提升 1.7 $\times$. FlashAttention 达到与其他两种实现相同的困惑度, 因为我们没有改变模型定义. [附录 E](#A5) 包含训练过程中验证困惑度的图表, 确认 FlashAttention 在数值上与基线一样稳定, 并产生相同的训练/验证曲线.

<span id="table-02"></span>

![论文原表 2](./flashattention/table-02.png)

**表 2.** 使用 FlashAttention 的 GPT-2 小型和中型相比 Huggingface 实现最高可获得 3 $\times$ 的加速, 相比 Megatron-LM 最高可获得 1.7 $\times$ 的加速. 训练时间在 8 $\times$ A100s GPU 上报告.

##### 远程范围竞赛.

我们在远程范围竞赛 (LRA [Tay20]) 基准测试中比较了原始 Transformer (使用标准实现或 FlashAttention). 我们测量所有模型的准确性, 吞吐量和训练时间. 每个任务的序列长度不同, 介于 1024 到 4096 之间. 我们遵循 [Tay20] 和 [Xiong21] 的实现和实验设置. [+3] [表 3](#table-03) 显示 FlashAttention 相比标准注意力最高可获得 2.4 $\times$ 的加速. 块稀疏 FlashAttention 比我们测试的所有近似注意力方法都更快.

<span id="table-03"></span>

![论文原表 3](./flashattention/table-03.png)

**表 3.** 标准注意力, FlashAttention, 块稀疏 FlashAttention 和近似注意力基线在长程竞技场 (Long-Range-Arena) 基准测试中的表现.

<span id="S4.SS2"></span>

### 4.2 使用更长序列的更好模型

##### 长上下文的语言建模.

FlashAttention 的运行时和内存效率使我们能够将 GPT-2 的上下文长度增加 4 $\times$, 同时仍然比 Megatron-LM 的优化实现运行得更快. [表 4](#table-04) 显示, 使用 FlashAttention 且上下文长度为 4K 的 GPT-2, 仍比上下文长度为 1K 的 Megatron GPT-2 快 30%, 同时获得更低 0.7 的困惑度.

<span id="table-04"></span>

![论文原表 4](./flashattention/table-04.png)

**表 4.** 使用 FlashAttention 的 GPT-2 小型, 在上下文长度比 Megatron-LM 大 4 $\times$ 的情况下, 速度仍快 30%, 同时困惑度提高了 0.7. 在 8 $\times$ A100 GPU 上的训练时间已报告.

##### 长文档分类.

使用 FlashAttention 训练具有更长序列的 Transformer, 可提高 MIMIC-III [Johnsa16] 和 ECtHR [Chalki19, Chalki21] 数据集上的性能. MIMIC-III 包含重症监护病房患者出院小结, 每个小结都标注了多个标签. ECtHR 包含欧洲人权法院的法律案件, 每个案件都对应涉嫌违反的人权公约的条款. 这两个数据集都包含非常长的文本文档; MIMIC 平均令牌数为 2, 395, 最长文档包含 14, 562 个令牌, 而 ECtHR 的平均和最长令牌数分别为 2, 197 和 49, 392. 我们评估了在预训练 RoBERTa 模型 [Liua19] 上增加序列长度带来的提升 (我们重复位置嵌入, 如 [Beltag20] 所示).

[表 5](#table-05) 显示, 在 MIMIC 上, 序列长度为 16K 比 512 长度高出 4.3 个点, 而在 ECtHR 上, 长度为 8K 比 512 高出 8.5 个点. 这些差异可能是由于微小的分布变化: MIMIC-III 包含专业医疗文本, 因此可能更容易受到文档长度分布变化的影响, 而 ECtHR 包含一般语言文本.

<span id="table-05"></span>

![论文原表 5](./flashattention/table-05.png)

**表 5.** 使用 FlashAttention 在不同序列长度下的长文档性能 (micro $F_{1}$).

<span id="table-06"></span>

![论文原表 6](./flashattention/table-06.png)

**表 6.** 我们报告了第一个能够在 Path-X 和 Path-256 上实现非随机性能的 Transformer 模型.

##### Path-X 和 Path-256.

Path-X 和 Path-256 基准测试是来自长距离领域基准的挑战性任务, 旨在测试长上下文. 该任务是判定黑白 128 $\times$ 128 (或 256 $\times$ 256) 图像中的两个点是否有路径连接它们, 并且图像是一次一个像素地输入给 Transformer. 在以前的工作中, 所有 Transformer 模型要么内存溢出, 要么仅达到随机性能 [Tay20]. 一直在寻找可以建模如此长上下文的替代架构 [Ref22]. 我们在这里展示了 Transformer 模型首次能够解决 Path-X 和 Path-256 的结果 ([表 6](#table-06)). 我们在 Path-64 上预训练一个 Transformer, 然后通过空间插值位置嵌入将其转移到 Path-X. FlashAttention 在 Path-X 上实现了 61.4 的准确率. 另外, 块稀疏 FlashAttention 使 Transformer 能够扩展到序列长度 64K, 在 Path-256 上达到 63.1 的准确率 [+4].

<span id="S4.SS3"></span>

### 4.3 注意力基准测试

<span id="figure-03"></span>

![参见图注](./flashattention/figure-03.png)

**图 3.** 左: 前向和反向传递的运行时间. 右: 注意力内存使用量.

我们改变序列长度, 并在一块具有 40 GB HBM 的 A100 GPU 上测量 FlashAttention 和块稀疏 FlashAttention 与各种注意力基线的运行时间和内存使用情况, 使用 dropout 和填充掩码. 我们与精确注意力, 近似注意力和稀疏注意力的参考实现进行比较. 我们在正文中报告了一部分基线; 附录 [E](#A5) 包含更多基线和完整细节.

##### 运行时间.

[图 3](#figure-03) (左) 报告了 FlashAttention 和块稀疏 FlashAttention 在前向和反向传播中的运行时间 (单位: 毫秒), 并与精确, 近似和稀疏注意力的基线进行比较 (精确数字见附录 [E](#A5)). 运行时间随序列长度呈二次增长, 但 FlashAttention 的运行速度明显快于精确注意力基线, 最高比 PyTorch 实现快 3 $\times$. 许多近似/稀疏注意力机制的运行时间随序列长度呈线性增长, 但对于短序列, FlashAttention 由于访问内存次数较少, 仍然比近似和稀疏注意力运行更快. 在序列长度在 512 到 1024 之间时, 近似注意力的运行时间开始与 FlashAttention 交叉. 另一方面, 块稀疏 FlashAttention 在所有我们已知的精确, 稀疏和近似注意力实现中, 在所有序列长度下都更快.

##### 内存占用.

[图 3](#figure-03) (右) 显示了 FlashAttention 和块稀疏 FlashAttention 的内存占用情况, 并将其与各种精确, 近似和稀疏注意力基线进行了比较. FlashAttention 和块稀疏 FlashAttention 的内存占用相同, 并且随序列长度线性增长. 相比精确注意力基线, FlashAttention 的内存效率提高了最多 20 $\times$, 并且比近似注意力基线更节省内存. 除 Linformer 外的所有其他算法在 A100 GPU 上在 64K 之前都会耗尽内存, 而 FlashAttention 的效率仍比 Linformer 高 2 $\times$.

<span id="S5"></span>

## 5 限制与未来方向

我们讨论了我们方法的局限性以及未来方向. 相关工作见 [附录 A](#A1).

编译到 CUDA. 我们当前构建注意力的 IO 感知实现的方法需要为每个新的注意力实现编写新的 CUDA 内核. 这要求用比 PyTorch 低得多的语言来编写注意力算法, 并且需要大量工程工作. 实现可能也无法在不同 GPU 架构之间迁移. 这些限制表明, 需要一种方法, 能够支持在高级语言 (例如 PyTorch) 中编写注意力算法, 并编译为 CUDA 中的 IO 感知实现——类似于图像处理中 Halide 的尝试 [RaganK13].

IO 感知深度学习. 我们相信 IO 感知方法可以超越注意力机制. 注意力是 Transformer 中最耗内存的计算, 但深度网络中的每一层都会接触 GPU HBM. 我们希望我们的工作能激发对其他模块的 IO 感知实现. 在 [附录 D](#A4) 中, 我们讨论了这些潜在的扩展.

多 GPU IO 感知方法. 我们对注意力的 IO 感知实现是在单 GPU 上计算注意力的常数范围内的最优实现. 然而, 注意力计算可能可以在多个 GPU 上并行化 [Recht13]. 使用多 GPU 为 IO 分析增加了一层复杂性——需要考虑 GPU 之间的数据传输. 我们希望我们的工作能激发未来在这一方向上的研究.

#### 致谢

我们的实现以 [Apex FMHA 代码](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha) 为起点. 我们感谢 Young-Jun Ko 对其 FMHA 实现的深入讲解, 以及他对我们有关 CUDA 的问题所提供的详尽解答. 我们感谢 Sabri Eyuboglu, Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen 和 Xun Huang 对论文早期稿件提出的建设性反馈和建议. 我们也感谢 Markus Rabe 和 Charles Staats 对其注意力算法的有益讨论.

我们衷心感谢 NIH 对编号 U54EB020405 (Mobilize) 的支持, NSF 对编号 CCF1763315 (Beyond Sparsity) , CCF1563078 (Volume to Velocity) 和 1937301 (RTML) 的支持; ARL 对编号 W911NF-21-2-0251 (Interactive Human-AI Teaming) 的支持; ONR 对编号 N000141712266 (Unifying Weak Supervision) , N00014-20-1-2480 (Understanding and Applying Non-Euclidean Geometry in Machine Learning) 以及 N000142012275 (NEPTUNE) 的支持; NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, HAI-GCP & HAI-Azure Cloud Credits for Research 项目, 斯坦福数据科学计划 (SDSI) , 通过国家国防科学与工程研究生奖学金 (NDSEG) 计划的国防部 (DoD), 以及斯坦福 DAWN 项目的成员: Facebook, Google 和 VMWare 的支持. 尽管文中有任何版权标注, 美国政府仍被授权为政府目的复制和分发此材料的再版. 本文中表达的任何观点, 发现和结论或建议均为作者个人观点, 不一定反映 NIH, ONR 或美国政府的意见, 政策或认可, 无论是明示还是暗示. Atri Rudra 的研究得到 NSF 资助 CCF-1763481 的支持.

<span id="A1"></span>

## 附录 A 相关工作

IO 感知运行时优化. 优化对快速/慢速内存的读写这一广泛概念在计算机科学中有着悠久的历史, 并且已经有许多名称. 我们在这项工作中 [Vitter88] 与分析 I/O 复杂度的文献建立了最直接的联系, 但内存层次结构的概念是基础性的, 并以多种形式出现, 从工作集模型 [Dennin68], 到数据局部性 [Lama91], 再到算术强度的 Roofline 模型 [Willia09], 到可扩展性分析 [McSher15], 再到计算机体系结构标准教材的讨论 [Patter03]. 我们希望这项工作能鼓励社区在深度学习堆栈的更多部分采用这些思想.

使用结构化矩阵的高效机器学习模型. 矩阵乘法是大多数机器学习模型中的核心计算瓶颈. 为了降低计算复杂性, 已经有许多方法尝试使用更高效的矩阵集合进行学习. 这些矩阵被称为*结构化矩阵*, 其参数数量和运行时间均为亚二次方 ($o(n^{2})$ 对于维度 $n\times n$). 最常见的结构化矩阵示例是稀疏矩阵和低秩矩阵, 以及在信号处理 (傅里叶, 切比雪夫, 正弦/余弦, 正交多项式) 中常见的快速变换. 在机器学习中, 还提出了几类更通用的结构化矩阵: Toeplitz 类 [Sindhw15], 低位移秩 [Kailat79], 准可分离 [Gohber99]. 我们用于块稀疏注意力的蝴蝶模式的设计灵感来源于这样一个事实: 蝴蝶矩阵 [Parker95, Dao19] 及其乘积已被证明能够以几乎最优的运行时间和参数数量 [Ref18, Dao20] 表示任何结构化矩阵. 然而, 即使结构化矩阵在理论上是高效的, 它们的应用仍然不广, 因为很难将其效率转化为实壁时加速, 因为密集无限制矩阵乘法已有非常优化的实现, 这一现象被称为硬件彩票 [Hooker20]. 为了让蝴蝶矩阵更适合硬件使用, 也提出了蝴蝶矩阵的扩展 [Daoa22, Daob22].

稀疏训练. 我们的块稀疏 FlashAttention 可以被视为在使稀疏模型训练更高效方面的一步. 稀疏模型在压缩推理模型方面 (剪枝) 已经取得了成功, 通过稀疏化权重矩阵 [Han16, Hana15, Sanh20, Lin17, Dong17]. 对于模型训练, 彩票假设 [Carbin18, Frankl19, Frankl20] 表明, 从较大的密集网络导出的某些小型子网络, 其性能与原始密集网络一样好. 我们的块稀疏 FlashAttention 也可以被视为注意力上下文中的固定彩票票: 我们在训练过程中将稀疏模式固定为蝶形模式, 并观察到它在 Long-range Arena 任务中的表现几乎与 (密集) FlashAttention 一样好.

高效变换器. 基于变换器的模型已经成为自然语言处理 [Devlin19] 和计算机视觉 [Dosovi20, Yuan21] 中最广泛使用的架构. 然而, 它们的计算瓶颈之一是时间和内存的规模随序列长度呈二次增长. 为了克服这一瓶颈, 有许多方法, 包括使用哈希进行近似 (即稀疏) 的方法, 如 Reformer [Kitaev20] 和 Smyrf [Daras20], 以及使用低秩近似的方法, 如 Performer [Chorom20, Likhos20]. 甚至可以将稀疏和低秩近似结合以获得更好的准确性 (例如 Longformer [Beltag20], BigBird [Zaheer20], Scatterbrain [Chen21], 长短变换器 [Zhu21], Combiner [Ren21]). 其他方法包括沿序列维度压缩以一次关注多个标记 [Refb19, Sukhba19, Lan20, Refa21]. 还可以关注来自先前序列的状态以帮助延长上下文 (例如 Transformer-XL [Dai19] 和 Compressive Transformer [Rae20]). 更多详情请参考综述 [Taya20].

在开发其他模块以替代关注模型较长上下文的工作上有几条研究路线. HiPPO [Ref20] 及其扩展项目, 最著名的是 S4 [Ref21, Ref22, Goel22], 将历史投影到多项式基上, 通过状态空间模型实现对历史的准确重建. 它们结合了 CNN (高效训练) , RNN (高效推理) 和连续模型 (对采样率变化具有鲁棒性) 的优点. LambdaNetworks [Bello21], AFT [Zhai21] 和 FLASH [Hua22] 是在图像分类和语言建模背景下尝试替代注意力的其他方法.

<span id="A2"></span>

## 附录 B 算法细节

我们首先推导注意力的前向和反向传播, 并展示它们可以以节省内存的方式计算 (所需额外内存与序列长度呈线性而非平方关系). 虽然它们减少了所需的额外内存, 但如果使用简单方法, 仍会产生平方级的高带宽内存访问, 导致执行速度较慢. 我们描述了 FlashAttention 算法, 在 GPU 上实现前向和反向传播, 从而减少高带宽内存访问, 实现更快的运行速度和更小的内存占用.

<span id="A2.SS1"></span>

### B. 1 节省内存的前向传播

在使注意力机制节省内存方面的主要挑战是 softmax, 它会将 $\mathbf{K}$ 的列 (以及 $\mathbf{V}$ 的列) 联系在一起. 我们的方法是分别计算 softmax 的归一化常数, 以解耦列. 这个技术 [Gimels18] 已在文献中 [Kitaev20, Staats21] 用于表明注意力计算不需要二次 *额外* 内存 (尽管 HBM 访问次数仍是二次的, 这会导致运行时间较慢).

为了简单起见, 我们在此省略 softmax 过程中的最大值偏移步骤. [附录 B.3](#A2.SS3) 中的完整算法包含所有步骤.

回顾一下, 给定输入序列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 我们希望计算注意力输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\begin{aligned}
\mathbf{S} &= \mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\\
\mathbf{O} &= \mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
\end{aligned}
$$

我们有 $S_{\mathrm{ij}}=q_{i}^{\top}k_{j}$, 其中 $q_{i}$ 和 $k_{j}$ 分别是 $i$ 和 $j$ 的 $\mathbf{Q}$ 列和 $\mathbf{K}$ 列. 定义 softmax 的归一化常数:

<span id="A2.E1"></span>

$$
L_{i}=\sum_{j}e^{q_{i}^{\top}k_{j}}.\tag{1}
$$

令 $v_{j}$ 为 $j$-th 列的 $\mathbf{V}$, 则输出的 $i$-th 列为:

<span id="A2.E2"></span>

$$
\begin{aligned}
o_{i} &= P_{i:}\mathbf{V}
 = \sum_{j}P_{\mathrm{ij}}v_{j}\\
 &= \sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}.\tag{2}
\end{aligned}
$$

我们看到, 一旦计算出 $L_{i}$, 我们可以通过反复求和 $\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}$ 来在不使用额外内存的情况下计算 $o_{i}$. 因此前向传播可以用 $O(n)$ 额外内存计算:

1. 根据 [公式 1](#A2.E1), 为所有 $i$ 计算 $L_{i}$, 这需要额外的 $O(n)$ 内存.
2. 根据 [公式 2](#A2.E2), 为所有 $i$ 计算 $o_{i}$, 这需要额外的 $O(d)$ 内存.

<span id="A2.SS2"></span>

### B. 2 内存高效的反向传递

我们推导了注意力的反向传递, 并展示了它也可以用线性内存计算. [Staats21] 表明, 通过对内存高效的前向传递应用梯度检查点, 可以无需二次额外内存完成反向传递. 我们则明确推导了反向传递, 并展示如何以内存高效的方式进行计算.

假设有一个标量损失函数 $\phi$, 并且让输出梯度为 $\mathbf{\mathrm{dO}}\in\mathbb{R}^{n\times d}$ (其中 $\mathbf{\mathrm{dO}}$ 表示 $\frac{\partial\phi}{\partial\mathbf{O}}$). 我们想要计算输入梯度 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{n\times d}$ (其中 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ 分别表示 $\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}$).

梯度 $\mathbf{\mathrm{dV}}$ 很容易看出. 手动应用反向模式自动微分 (也称链式法则), 我们得到 (矩阵表示) $\mathbf{\mathrm{dV}}=\mathbf{P}^\top\mathbf{\mathrm{dO}}$. 因此:

<span id="A2.E3"></span>

$$
\begin{aligned}
\mathrm{dv}_{j} &= \sum_{i}P_{\mathrm{ij}}\mathrm{do}_{i}
 = \sum_{i}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}\mathrm{do}_{i}.\tag{3}
\end{aligned}
$$

由于我们已经计算了 $L_{i}$, $\mathrm{dv}_{j}$ 可以通过重复求和在不增加额外内存的情况下计算.

梯度 $\mathbf{\mathrm{dQ}}$ 和 $\mathbf{\mathrm{dK}}$ 稍微复杂一些. 我们先来看梯度 $\mathbf{\mathrm{dP}}$ 和 $\mathbf{\mathrm{dS}}$. 根据 [方程 2](#A2.E2), 我们有 $\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^\top$, 因此:

$$
\mathrm{dP}_{\mathrm{ij}}=\mathrm{do}_{i}^{\top}v_{j}.
$$

回忆 $P_{i:}=\mathrm{softmax}(S_{i:})$. 利用 $y=\mathrm{softmax}(x)$ 的雅可比矩阵是 $\mathrm{diag}(y)-yy^\top$ 这一事实, 我们得到

$$
\begin{aligned}
\mathrm{dS}_{i:}
 &= (\mathrm{diag}(P_{i:})-P_{i:}P_{i:}^{\top})\mathrm{dP}_{i:}\\
 &= P_{i:}\circ \mathrm{dP}_{i:}-(P_{i:}^{\top}\mathrm{dP}_{i:})P_{i:}.
\end{aligned}
$$

其中 $\circ$ 表示逐点相乘.

定义

<span id="A2.E4"></span>

$$
\begin{aligned}
D_{i} &= P_{i:}^{\top}\mathrm{dP}_{i:}
 = \sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}\mathrm{do}_{i}^{\top}v_{j}\\
&= \mathrm{do}_{i}^{\top}\sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}
 = \mathrm{do}_{i}^{\top}o_{i}.
\tag{4}
\end{aligned}
$$

然后

$$
\mathrm{dS}_{i:}=P_{i:}\circ \mathrm{dP}_{i:}-D_{i}P_{i:}.
$$

因此

$$
\begin{aligned}
\mathrm{dS}_{\mathrm{ij}}
 &= P_{\mathrm{ij}}\mathrm{dP}_{\mathrm{ij}}-D_{i}P_{\mathrm{ij}}\\
 &= P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i}).
\end{aligned}
$$

现在我们可以得到梯度 $\mathbf{\mathrm{dQ}}$ 和 $\mathbf{\mathrm{dK}}$. 回想一下 $S_{\mathrm{ij}}=q_{i}^{\top}k_{j}$, 所以

<span id="A2.E5"></span>

$$
\begin{aligned}
\mathrm{dq}_{i} &= \sum_{j}\mathrm{dS}_{\mathrm{ij}}k_{j}
 = \sum_{j}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})k_{j}\\
&= \sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}(\mathrm{do}_{i}^{\top}v_{j}-D_{i})k_{j}.
\tag{5}
\end{aligned}
$$

类似地,

<span id="A2.E6"></span>

$$
\begin{aligned}
\mathrm{dk}_{j} &= \sum_{i}\mathrm{dS}_{\mathrm{ij}}q_{i}
 = \sum_{i}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})q_{i}\\
&= \sum_{i}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}(\mathrm{do}_{i}^{\top}v_{j}-D_{i})q_{i}.
\tag{6}
\end{aligned}
$$

因此反向传递也可以使用 $O(n)$ 的额外内存计算:

1. 根据 [公式 3](#A2.E3) 计算所有 $j$ 的 $\mathrm{dv}_{j}$, 这需要 $O(d)$ 的额外内存.
2. 根据 [公式 4](#A2.E4) 计算所有 $i$ 的 $D_{i}$, 这将占用 $O(n)$ 额外内存.
3. 根据 [公式 5](#A2.E5) 计算所有 $i$ 的 $\mathrm{dq}_{i}$, 这需要 $O(d)$ 的额外内存.
4. 根据 [公式 6](#A2.E6) 计算所有 $j$ 的 $\mathrm{dk}_{j}$, 这需要 $O(d)$ 的额外内存.

<span id="A2.SS3"></span>

### B. 3 FlashAttention: 前向传递

我们描述了 FlashAttention 前向传递的完整细节. 给定输入序列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 我们希望计算注意力输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\begin{aligned}
\mathbf{S} &= \tau\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\\
\mathbf{S}^{\mathrm{masked}} &= \mathrm{mask}(\mathbf{S})\in\mathbb{R}^{N\times N},\\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S}^{\mathrm{masked}})\in\mathbb{R}^{N\times N}.
\end{aligned}
$$

$$
\begin{aligned}
\mathbf{P}^{\mathrm{dropped}} &= \mathrm{dropout}(\mathbf{P},p_{\mathrm{drop}}),\\
\mathbf{O} &= \mathbf{P}^{\mathrm{dropped}}\mathbf{V}\in\mathbb{R}^{N\times d}.
\end{aligned}
$$

其中 $\tau\in\mathbb{R}$ 是某种 softmax 缩放 (通常是 $\frac{1}{\sqrt{d}}$), mask 是一种掩码函数, 它将输入的某些项设置为 $-\infty$, 而保持其他项不变 (例如, 当批次中的序列长度不同且被填充时的 key padding mask), $\mathrm{dropout}(x,p)$ 对 $x$ 进行逐元素的 dropout (即, 对于每个元素 $x$, 以概率 $1-p$ 输出 $\frac{x}{1-p}$, 以概率 $p$ 输出 0).

完整的算法见 [算法 2](#alg2). 我们保存输出 $\mathbf{O}$, softmax 统计信息 $\ell$ 和 $m$, 以及伪随机数生成器状态 ${\cal R}$, 用于反向传播.

<span id="alg2"></span>

**算法 2: FlashAttention 前向传播**

- **输入:** 在 HBM 中的矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 尺寸为 $M$ 的片上 SRAM, softmax 缩放常数 $\tau\in\mathbb{R}$, 掩码函数 mask, dropout 概率 $p_{\mathrm{drop}}$.
- 初始化伪随机数生成器状态 ${\cal R}$ 并保存到 HBM.
- 设置块大小 $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- 在 HBM 中初始化 $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$.
- 将 $\mathbf{Q}$ 分成 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 块, 每块大小为 $B_{r}\times d$, 然后将 $\mathbf{K},\mathbf{V}$ 分成 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 块 $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ 和 $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, 每块大小为 $B_{c}\times d$.
- 将 $\mathbf{O}$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}\times d$, 将 $\ell$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}$, 将 $m$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}$.
- **对于** $1\leq j\leq T_{c}$ **执行:**
  - 将 $\mathbf{K}_{j},\mathbf{V}_{j}$ 从 HBM 加载到片上 SRAM.
  - **对于** $1\leq i\leq T_{r}$ **执行:**
    - 将 $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ 从 HBM 加载到片上 SRAM.
    - 在芯片上, 计算 $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - 在芯片上, 计算 $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
    - 在芯片上, 计算 $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (逐点) , $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
    - 在芯片上, 计算 $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
    - 在芯片上, 计算 $\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$.
    - 将 $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ 写入 HBM.
    - 将 $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ 写入 HBM.
- **返回:** $\mathbf{O},\ell,m,{\cal R}$.

<span id="A2.SS4"></span>

### B. 4 FlashAttention: 反向传播

我们描述了 FlashAttention 反向传播的完整细节. 给定输入序列 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, 输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$ 和输出梯度 $\mathbf{\mathrm{dO}}$, 我们希望计算输入梯度 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$.

为了完整起见, 我们首先在 [算法 3](#alg3) 中描述标准注意力反向传播.

<span id="alg3"></span>

**算法 3: 标准注意力反向传播**

- **输入:** 矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$, $\mathbf{P}\in\mathbb{R}^{N\times N}$ 在 HBM 中.
- 从 HBM 按块加载 $\mathbf{P},\mathbf{\mathrm{dO}}$, 计算 $\mathbf{\mathrm{dV}}=\mathbf{P}^{\top}\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$, 将 $\mathbf{\mathrm{dV}}$ 写入 HBM.
- 从 HBM 按块加载 $\mathbf{\mathrm{dO}},\mathbf{V}$, 计算 $\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^{\top}\in\mathbb{R}^{N\times N}$, 将 $\mathbf{\mathrm{dP}}$ 写入 HBM.
- 从 HBM 读取 $\mathbf{P},\mathbf{\mathrm{dP}}$, 计算 $\mathbf{\mathrm{dS}}\in\mathbb{R}^{N\times N}$, 其中 $\mathrm{dS}_{\mathrm{ij}}=P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-\sum_{l}P_{\mathrm{il}}\mathrm{dP}_{\mathrm{il}})$, 将 $\mathbf{\mathrm{dS}}$ 写入 HBM.
- 从 HBM 按块加载 $\mathbf{\mathrm{dS}}$ 和 $\mathbf{K}$, 计算 $\mathbf{\mathrm{dQ}}=\mathbf{\mathrm{dS}}\mathbf{K}$, 将 $\mathbf{\mathrm{dQ}}$ 写入 HBM.
- 从 HBM 按块加载 $\mathbf{\mathrm{dS}}$ 和 $\mathbf{Q}$, 计算 $\mathbf{\mathrm{dK}}=\mathbf{\mathrm{dS}}^{\top}\mathbf{Q}$, 将 $\mathbf{\mathrm{dK}}$ 写入 HBM.
- **返回:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$.

现在我们对 FlashAttention 反向传播提出两点观察:

1. 我们不需要存储前向传播中大小为 $O(N^{2})$ 的 dropout 掩码. 相反, 我们可以保存前向传播中的伪随机数生成器状态, 并在反向传播中重新生成 dropout 掩码. 这只需使用 $O(N)$ 的额外内存.
2. 在计算 softmax 梯度时, 我们使用 [公式 4](#A2.E4) 来计算 $D_{i}=P_{i:}^{\top}\mathrm{dP}_{i:}$, 而不对大小为 $N$ 的 $P_{i:}$ 和 $\mathrm{dP}_{i:}$ 进行压缩 (它们可能无法放入 SRAM). 相反, 我们可以重写 $D_{i}=\mathrm{do}_{i}^{\top}o_{i}$, 并计算大小为 $d$ 的向量之间的点积.

完整的 FlashAttention 反向传播算法在 [算法 4](#alg4) 中. 概念上, 它只是 [第 B. 2 节](#A2.SS2) 推导的块版本.

<span id="alg4"></span>

**算法 4: FlashAttention 反向传播**

- **输入:** HBM 中的矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$, HBM 中的向量 $\ell,m\in\mathbb{R}^{N}$, 大小为 $M$ 的芯片上 SRAM, softmax 缩放常数 $\tau\in\mathbb{R}$, 掩码函数 mask, dropout 概率 $p_{\mathrm{drop}}$, 来自前向传播的伪随机数生成器状态 ${\cal R}$.
- 将伪随机数生成器状态设置为 ${\cal R}$.
- 设置块大小 $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- 将 $\mathbf{Q}$ 分成 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 块, 每块 $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$, 大小为 $B_{r}\times d$, 并将 $\mathbf{K},\mathbf{V}$ 分成 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 块 $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ 和 $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, 每块大小为 $B_{c}\times d$.
- 将 $\mathbf{O}$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}\times d$, 将 $\mathbf{\mathrm{dO}}$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}\times d$, 将 $\ell$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}$, 将 $m$ 分成 $T_{r}$ 块, 每块大小为 $B_{r}$.
- 在 HBM 中初始化 $\mathbf{\mathrm{dQ}}=(0)_{N\times d}$ 并将其划分为 $T_{r}$ 块, 每块大小为 $B_{r}\times d$. 在 HBM 中初始化 $\mathbf{\mathrm{dK}}=(0)_{N\times d},\mathbf{\mathrm{dV}}=(0)_{N\times d}$ 并将 $\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ 划分为 $T_{c}$ 块 $\mathbf{\mathrm{dK}}_{1},\dots,\mathbf{\mathrm{dK}}_{T_{c}}$ 和 $\mathbf{\mathrm{dV}}_{1},\dots,\mathbf{\mathrm{dV}}_{T_{c}}$, 每块大小为 $B_{c}\times d$.
- **对于** $1\leq j\leq T_{c}$ **执行:**
  - 将 $\mathbf{K}_{j},\mathbf{V}_{j}$ 从 HBM 加载到片上 SRAM.
  - 在 SRAM 上初始化 $\tilde{\mathbf{\mathrm{dK}}}_{j}=(0)_{B_{c}\times d},\tilde{\mathbf{\mathrm{dV}}}_{j}=(0)_{B_{c}\times d}$.
  - **对于** $1\leq i\leq T_{r}$ **执行:**
    - 将 $\mathbf{Q}_{i},\mathbf{O}_{i},\mathbf{\mathrm{dO}}_{i},\mathbf{\mathrm{dQ}}_{i},\ell_{i},m_{i}$ 从 HBM 加载到片上 SRAM.
    - 在芯片上计算 $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - 在芯片上计算 $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
    - 在芯片上计算 $\mathbf{P}_{\mathrm{ij}}=\mathrm{diag}(l_{i})^{-1}\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-m_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - 在芯片上, 计算丢弃掩码 $\mathbf{Z}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}\times B_{c}}$, 其中每个条目的值为 $\frac{1}{1-p_{\mathrm{drop}}}$ 的概率为 $1-p_{\mathrm{drop}}$, 值为 0 的概率为 $p_{\mathrm{drop}}$.
    - 在芯片上, 计算 $\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{P}_{\mathrm{ij}}\circ\mathbf{Z}_{\mathrm{ij}}$ (逐点相乘).
    - 在芯片上, 计算 $\tilde{\mathbf{\mathrm{dV}}_{j}}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}+(\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}})^{\top}\mathbf{\mathrm{dO}}_{i}\in\mathbb{R}^{B_{c}\times d}$.
    - 在芯片上, 计算 $\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{\mathrm{dO}}_{i}\mathbf{V}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - 在芯片上, 计算 $\mathbf{\mathrm{dP}}_{\mathrm{ij}}=\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}\circ\mathbf{Z}_{\mathrm{ij}}$ (逐点相乘).
    - 在芯片上, 计算 $D_{i}=\mathrm{rowsum}(\mathbf{\mathrm{dO}}_{i}\circ\mathbf{O}_{i})\in\mathbb{R}^{B_{r}}$.
    - 在芯片上, 计算 $\mathbf{\mathrm{dS}}_{\mathrm{ij}}=\mathbf{P}_{\mathrm{ij}}\circ(\mathbf{\mathrm{dP}}_{\mathrm{ij}}-D_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - 将 $\mathbf{\mathrm{dQ}}_{i}\leftarrow\mathbf{\mathrm{dQ}}_{i}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}\mathbf{K}_{j}\in\mathbb{R}^{B_{r}\times d}$ 写入 HBM.
    - 在芯片上, 计算 $\tilde{\mathbf{\mathrm{dK}}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}}_{j}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}^{\top}\mathbf{Q}_{i}\in\mathbb{R}^{B_{c}\times d}$.
  - 将 $\mathbf{\mathrm{dK}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}_{j}},\mathbf{\mathrm{dV}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}$ 写入 HBM.
- **返回:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$.

我们看到, 类似于前向传播, 反向传播执行 $O(N^{2})$ FLOPs, 只需要除了输入, 输出, 输出梯度和输入梯度之外的 $O(N)$ 额外内存.

我们分析了反向传播的 IO 复杂度, 类似于前向传播 ([定理 2](#Thmtheorem2)).

###### 定理 5.

设 $N$ 为序列长度, $d$ 为头维度, $M$ 为 SRAM 的大小, 使用 $d\leq M\leq \mathrm{Nd}$. 标准注意力 ([算法](#alg0)) 反向传播需要 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问, 而 FlashAttention 反向传播 ([算法 4](#alg4)) 需要 $\Theta(N^{2}d^{2}M^{-1})$ 次 HBM 访问.

证明见 [附录 C](#A3).

### B. 5 与 [Staats21] 的比较

我们在这里描述了我们的 FlashAttention 算法与 [Staats21] 算法的一些相似性和差异.

从概念上讲, FlashAttention 和 [Staats21] 都使用成熟的平铺 (或 softmax 缩放) 技术 [Gimels18, Kitaev20] 对注意力矩阵的块进行操作. 为了减少内存占用, 这两种方法在前向传播中都避免存储大注意力矩阵, 并在反向传播中重新计算它.

第一个主要区别是 [Staats21] 侧重于减少总内存占用 (所需的最大 GPU 内存量), 而 FlashAttention 侧重于减少内存访问 (内存读/写的次数). 如 [第 2 节](#S2) 所述, 内存访问量是运行时间的主要决定因素. 减少内存访问也必然减少所需的总内存量 (例如, 如果某个操作产生 $A$ 次内存访问, 那么其总内存需求最多为 $A$). 因此, FlashAttention 比标准注意力更快 (2-4 $\times$), 而 [Staats21] 的速度与标准注意力差不多或略慢. 在所需总内存方面, 这两种方法都提供了显著的内存节省.

两种方法之间的第二个区别是从每个块总结信息以传递到下一个块的方式. [Staats21] 使用其临时输出以及 softmax 归一化统计来总结每个块. 在前向传播结束时, 所有块的临时输出结合这些统计生成最终输出. FlashAttention 则在处理每个块后增量更新输出 ([算法 1](#alg1) 第 12 行), 因此只需要一份输出副本 (而不是 $K$ 份对应 $K$ 个块). 这意味着 FlashAttention 的总内存需求比 [Staats21] 小.

最后一个主要区别是反向传播的计算方式. [Staats21] 使用梯度检查点重新计算注意力矩阵和每个模块的临时输出. 而 FlashAttention 则通过解析简化了反向传播 ([B. 2 节](#A2.SS2) 和 [B. 4](#A2.SS4)). 它只重新计算注意力矩阵, 而不重新计算每个模块的临时输出. 这减少了反向传播的内存需求并提升了速度.

<span id="A3"></span>

## 附录 C 证明

###### [定理 1](#Thmtheorem1) 的证明.

我们首先计算所需的 FLOPs 数量和额外内存.

主导的浮点运算 (FLOPs) 来自矩阵乘法. 在内循环中, 我们计算 $\mathbf{Q}_{i}\mathbf{K}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$ (见 [算法 1](#alg1) 第 9 行), 这需要 $O(B_{r}B_{c}d)$ FLOPs. 我们还计算 $\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j}\in\mathbb{R}^{B_{r}\times d}$ (见 [算法 1](#alg1) 第 12 行), 需要 $O(B_{r}B_{c}d)$ FLOPs. 我们执行内循环 $T_{c}T_{r}=\left\lceil\frac{N}{B_{c}}\right\rceil\left\lceil\frac{N}{B_{r}}\right\rceil$ 次. 因此 FLOPs 总数为

$$
O\left(\frac{N^{2}}{B_{c}B_{r}}B_{r}B_{c}d\right)=O(N^{2}d).
$$

在额外所需内存方面, 我们看到我们需要 $O(N)$ 内存来存储统计信息 $(\ell,m)$.

我们现在通过对 $j$ 关于 $0\leq j\leq T_{c}$ 进行归纳来证明算法的正确性. 设 $\mathbf{K}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ 为 $\mathbf{K}$ 的前 $\mathrm{jB}_{c}$ 行, 类似地, $\mathbf{V}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ 为 $\mathbf{V}$ 的前 $\mathrm{jB}_{c}$ 行. 设 $\mathbf{S}_{:,:j}=\mathbf{Q}\mathbf{K}_{:j}^{\top}\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$, 以及 $\mathbf{P}_{:,:j}=\mathrm{softmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$ (按行应用 softmax). 设 $m^{j},\ell^{(j)},\mathbf{O}^{(j)}$ 为在外层循环第 $j$ 次迭代后, HBM 中 $m,\ell,\mathbf{O}$ 的值 ([算法 1](#alg1) 第 5 行). (注意这些 $m,\ell,\mathbf{O}$ 的值在外层循环的每次迭代后都会被更新.) 我们想要证明, 在外层循环第 $j$ 次迭代后, 我们已经在 HBM 中计算了:

$$
\begin{aligned}
m^{(j)} &= \mathrm{rowmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N},\\
\ell^{(j)} &= \mathrm{rowsum}(\exp(\mathbf{S}_{:,:j}-m^{(j)}))\in\mathbb{R}^{N},\\
\mathbf{O}^{(j)} &= \mathbf{P}_{:,:j}\mathbf{V}_{:j}\in\mathbb{R}^{N\times d}.
\end{aligned}
$$

根据我们的初始化 ([算法 1](#alg1) 第 2 行), 这个结论对于 $j=0$ (即在外循环的任何迭代执行之前) 是成立的. 假设该结论对某个 $j=0,\dots,T_{c}-1$ 成立. 我们想要证明该结论也对 $j+1$ 成立. 确实, 当我们在外循环的 $(j+1)$ 次迭代中在内循环中更新统计数据 ([算法 1](#alg1) 第 10 行) 时, 我们更新 $m^{(j+1)}=\max(m^{(j)},\tilde{m})$, 其中 $\tilde{m}\in\mathbb{R}^{N}$ 是 $\mathbf{S}_{:,j:j+1}$ 的行最大值, $\mathbf{S}_{:,j:j+1}$ 是从 $\mathbf{S}$ 的第 $\mathrm{jB}_{c}$ 列到第 $(j+1)B_{c}-1$ 列的切片. 这意味着

$$
m^{(j+1)}=\mathrm{rowmax}(\mathbf{S}_{:,:j+1})\in\mathbb{R}^{N}.
$$

同样地, 我们更新

$$
\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell},
$$

其中 $\tilde{\ell}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,j:j+1}-\tilde{m}))\in\mathbb{R}^{N}$. 通过在 [第 3.1 节](#S3.SS1) 中相同的代数操作, 我们得到:

$$
\ell^{(j+1)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j+1}-m^{(j+1)}))\in\mathbb{R}^{N}.
$$

设 $\mathbf{V}_{j:j+1}$ 为从列 $\mathrm{jB}_{c}$ 到列 $(j+1)B_{c}-1$ 的 $\mathbf{V}$ 的切片, 我们还更新如下:

$$
\begin{aligned}
\mathbf{O}^{(j+1)}
 &= \mathrm{diag}(\ell^{(j+1)})^{-1}\bigl(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{O}^{(j)}\\
 &\qquad +e^{\tilde{m}-m^{(j+1)}}\exp(\mathbf{S}_{:,j:j+1}-\tilde{m})\mathbf{V}_{j:j+1}\bigr)\\
 &= \mathrm{diag}(\ell^{(j+1)})^{-1}\bigl(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{P}_{:,:j}\mathbf{V}_{:j}\\
 &\qquad +e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,j:j+1})\mathbf{V}_{j:j+1}\bigr)\\
 &= \mathrm{diag}(\ell^{(j+1)})^{-1}\bigl(e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,:j})\mathbf{V}_{:j}\\
 &\qquad +e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,j:j+1})\mathbf{V}_{j:j+1}\bigr)\\
 &= \mathrm{diag}(\ell^{(j+1)})^{-1}\bigl(\exp(\mathbf{S}_{:,:j}-m^{(j+1)})\mathbf{V}_{:j}\\
 &\qquad +\exp(\mathbf{S}_{:,j:j+1}-m^{(j+1)})\mathbf{V}_{j:j+1}\bigr)\\
 &= \mathrm{diag}(\ell^{(j+1)})^{-1}\exp\left(\begin{bmatrix}\mathbf{S}_{:,:j}&\mathbf{S}_{:,j:j+1}\end{bmatrix}-m^{(j+1)}\right)\begin{bmatrix}\mathbf{V}_{:j}\\
 \mathbf{V}_{j:j+1}\end{bmatrix}\\
 &= \mathrm{softmax}(\mathbf{S}_{:,:j+1})\mathbf{V}_{:j+1}.
\end{aligned}
$$

然后我们看到该结论对于 $j+1$ 也是成立的. 通过归纳法, 该结论对所有 $j=0,\dots,T_{c}$ 都成立.

当 $j=T_{c}$ 时, 我们得出 HBM 中 $\mathbf{O}$ 的最终值为 $\mathrm{softmax}(\mathbf{S})\mathbf{V}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$.

∎

###### [定理 2](#Thmtheorem2) 证明.

我们首先分析标准注意力实现的 IO 复杂度. 输入 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 位于 HBM 中, 在算法结束时, 输出 $\mathbf{O}\in\mathbb{R}^{N\times d}$ 被写入 HBM.

在计算矩阵乘法 $\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$ 的第一步中, 输入 $\mathbf{Q},\mathbf{K}$ 从 HBM 中读取, 输出 $\mathbf{S}\in\mathbb{R}^{N\times N}$ 写入 HBM ([算法 0](#alg0) 第 1 行). 这会产生 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问.

在计算 $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$ 的第二步中, 输入 $\mathbf{S}$ 从 HBM 读取, 输出 $\mathbf{P}$ 写入 HBM ([算法 0](#alg0) 第 2 行). 这将产生 $\Theta(N^{2})$ 次 HBM 访问.

在计算 $\mathbf{O}=\mathbf{P}\mathbf{V}$ 的最后一步中, 输入 $\mathbf{P},\mathbf{V}$ 从全局内存读取, 输出 $\mathbf{O}$ 写入 HBM ([算法 0](#alg0) 第 3 行). 这会产生 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问.

总体来看, 标准注意力实现需要 $\Theta(\mathrm{Nd}+N^{2})$ 次全局内存访问.

我们现在分析流式注意力的 IO 复杂度.

根据 [算法 1](#alg1), 我们看到 $\mathbf{K}$ 和 $\mathbf{V}$ 的每个元素只需从 HBM 加载一次 ([算法 1](#alg1) 第 6 行). 我们对 $\mathbf{Q}$ 和 $\mathbf{O}$ 进行 $T_{c}$ 次遍历, 每次遍历将所有 $\mathbf{Q}$ 和所有 $\mathbf{O}$ 加载到 HBM ([算法 1](#alg1) 第 8 行). 因此 HBM 访问的次数为 $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$.

我们推导块大小 $B_{c}$ 和 $B_{r}$ 的条件. 我们需要大小为 $B_{c}\times d$ 的块 $\mathbf{K}_{j}$ 和 $\mathbf{V}_{j}$ 能够适合片上内存, 这可转换为:

$$
B_{c}d=O(M)\Leftrightarrow B_{c}=O\left(\frac{M}{d}\right).
$$

同样, 我们需要大小为 $B_{r}\times d$ 的块 $\mathbf{Q}_{i},\mathbf{O}_{i}$ 能够适合片上内存, 这可转换为:

$$
B_{r}d=O(M)\Leftrightarrow B_{r}=O\left(\frac{M}{d}\right).
$$

最后, 我们需要将大小为 $B_{r}\times B_{c}$ 的块 $\mathbf{S}_{\mathrm{ij}}$ 放入片上存储器中, 这意味着:

$$
B_{r}B_{c}=O(M).
$$

因此我们设定:

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},\frac{M}{B_{c}}\right)\right)=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

然后我们有:

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

因此, HBM 访问次数为:

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

###### [命题 3](#Thmtheorem3) 的证明.

为了矛盾, 假设存在一个算法可以计算精确注意力, 其中对所有 $M\in[d,\mathrm{Nd}]$ 的 HBM 访问次数为

$$
o\left(\frac{N^{2}d^{2}}{M}\right).
$$

在 $M=\Theta(\mathrm{Nd})$ 的情况下, 这导致 HBM 访问次数为:

$$
o\left(\frac{N^{2}d^{2}}{\mathrm{Nd}}\right)=o(\mathrm{Nd}).
$$

然而, 注意力的输入 (矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V}$) 和输出 $\mathbf{O}$ 的大小为 $\mathrm{Nd}$, 并且它们一开始就在 HBM 中, 因此如果算法计算精确注意力, 它必须至少进行 $\Omega(\mathrm{Nd})$ 次 HBM 访问. 这是一个矛盾. ∎

<span id="Thmtheorem5"></span>

###### [定理 5](#Thmtheorem5) 的证明.

注意力反向传播的 IO 复杂度与注意力前向传播的 IO 复杂度非常相似 ([定理 2](#Thmtheorem2)). 这里我们提供一个证明的概略.

我们首先分析标准注意力反向传递的 IO 复杂度. 输入 $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$ 位于 HBM 中, 而在算法结束时输出 $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$ 写入 HBM.

在标准注意力反向传递的每一步中, 需要从 HBM 加载大小为 $\mathrm{Nd}$ 或 $N^{2}$ 的输入, 并需要将大小为 $N^{2}$ 或 $\mathrm{Nd}$ 的输出写入 HBM. 这会导致 $\Theta(\mathrm{Nd}+N^{2})$ 次 HBM 访问.

现在我们分析 FlashAttention 反向传播的 IO 复杂度.

类似于 [定理 2](#Thmtheorem2), 我们看到 $\mathbf{K}$ 和 $\mathbf{V}$ 的每个元素只从 HBM 加载一次. $\mathbf{\mathrm{dK}}$ 和 $\mathbf{\mathrm{dV}}$ 的每个元素只写入 HBM 一次. 我们对 $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$ 做 $T_{c}$ 次遍历, 每次遍历将所有 $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$ 加载到 HBM. 我们还对 $\mathbf{\mathrm{dQ}}$ 做 $T_{c}$ 次遍历, 每次遍历从/向 HBM 读取/写入所有 $\mathbf{\mathrm{dQ}}$. 因此 HBM 访问次数为 $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$.

如同 [定理 2](#Thmtheorem2) 的证明中, 对块大小的约束为:

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

然后我们有:

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

因此, HBM 访问次数为:

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

<span id="A4"></span>

## 附录 D 扩展详细信息

### D. 1 块稀疏 FlashAttention

我们在 [算法 5](#alg5) 中描述了完整的块稀疏 FlashAttention 算法. 该算法与 [算法 2](#alg2) 相同, 只是我们跳过了零块.

<span id="alg5"></span>

**算法 5: 块稀疏 FlashAttention 前向传播**

- **输入:** 矩阵 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 位于 HBM 中, 片上 SRAM 大小为 $M$, softmax 缩放常数为 $\tau\in\mathbb{R}$, 掩码函数为 mask, dropout 概率为 $p_{\mathrm{drop}}$, 块大小为 $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$, 块稀疏掩码为 $M\in\{0,1\}^{N/B_{r}\times N/B_{c}}$.
- 初始化伪随机数生成器状态 ${\cal R}$ 并保存到 HBM.
- 在 HBM 中初始化 $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$.
- 将 $\mathbf{Q}$ 分成 $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ 块, 每块 $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$, 大小为 $B_{r}\times d$, 并将 $\mathbf{K},\mathbf{V}$ 分成 $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ 块 $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ 和 $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, 每块大小为 $B_{c}\times d$.
- 将 $\mathbf{O}$ 分成 $T_{r}$ 块, 每块 $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$, 大小为 $B_{r}\times d$, 将 $\ell$ 分成 $T_{r}$ 块, 每块 $\ell_{i},\dots,\ell_{T_{r}}$, 大小为 $B_{r}$, 将 $m$ 分成 $T_{r}$ 块, 每块 $m_{1},\dots,m_{T_{r}}$, 大小为 $B_{r}$.
- **对** $1\leq j\leq T_{c}$ **执行:**
  - 从 HBM 加载 $\mathbf{K}_{j},\mathbf{V}_{j}$ 到片上 SRAM.
  - **对** $1\leq i\leq T_{r}$ **执行:**
    - **如果** $M_{\mathrm{ij}}\neq 0$:
      - 从 HBM 加载 $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ 到片上 SRAM.
      - 在芯片上, 计算 $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
      - 在芯片上, 计算 $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
      - 在芯片上, 计算 $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (逐点) 和 $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
      - 在芯片上, 计算 $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
      - 在芯片上, 计算 $\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$.
      - 将 $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ 写入 HBM.
      - 将 $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ 写入 HBM.
- **返回:** $\mathbf{O},\ell,m,{\cal R}$.

我们证明了块稀疏 FlashAttention 的 IO 复杂度.

###### [命题 4](#Thmtheorem4) 的证明.

该证明与 [定理 2](#Thmtheorem2) 的证明非常相似. 对于块稀疏情况, 注意我们只需要加载与非零块对应的块. 因此, HBM 访问次数按 $s$ 缩放, 即块稀疏掩码中非零块的比例. 然而, 对于较小的 $s$ 值, 我们仍然需要写入结果 $\mathbf{O}\in\mathbb{R}^{N\times d}$. 因此 HBM 访问次数为

$$
\Theta\left(\mathrm{Nd}+\frac{N^{2}d^{2}}{M}s\right).
$$

∎

### D. 2 潜在扩展

我们在这里讨论几种可将 IO 感知方法扩展以加速深度学习训练的潜在方法.

多 GPU 注意力. 大规模语言模型在数百或数千个 GPU 上进行训练, 通常会在同一个节点上的 4-8 个 GPU 之间分配注意力计算 [Shoeyb19]. 这引入了另一个内存层次结构: 除了 GPU SRAM 和 GPU HBM 之外, 我们还拥有其他 GPU 的 HBM. 对于非常长的序列, 同一节点上的不同 GPU 可以协作计算注意力, 同时考虑不同层次内存结构的非对称性.

稀疏 MLP 层. 典型的密集 MLP 层是计算受限而非内存受限. 为了提高它们的效率, 可以使用稀疏权重矩阵的 MLP 层 [Daoa22]. 然而, 许多稀疏 MLP 层反而是内存受限, 其加速效果通常与稀疏程度不成比例. 我们认为, Io 感知的实现可以缓解这个问题, 并实现稀疏性的优势. 我们对该方向的未来工作感到兴奋, 以减少大型模型的计算需求并改善它们的实际运行时间.

核机器学习. 我们在 FlashAttention 中的方法依赖于 $N\times N$ 注意力矩阵是低秩矩阵 $\mathbf{Q}\mathbf{K}^{\top}$ (秩为 $d\ll N$) 的函数这一事实. 因此, 我们可以重复加载输入 $\mathbf{Q},\mathbf{K}$ 并重新计算所需的注意力矩阵块, 从而显著减少 HBM 访问. 在核机器学习中也存在类似情景: $N\times N$ 核矩阵 $\mathbf{K}$ 的每个元素 $K_{\mathrm{ij}}$ 都是两个大小为 $d\ll N$ 的向量的函数, 因为它测量了两个数据点 $x_{i}$ 和 $x_{j}$ 之间的相似性. KeOps 库 [Feydy20, Charli21] 是减少内存读写从而加快核操作的一个成功例子. 我们希望这能激励核方法更多关注减少 IO 而不仅仅是 FLOPs.

<span id="A5"></span>

## 附录 E 完整实验结果

### E. 1 BERT

我们依据参考 MLPerf 1.1 实现的训练程序和超参数来训练 BERT-large. 特别地, 我们使用 LAMB 优化器, 学习率为 3.75e-3, 批量大小为 448, 最多训练 7100 步. 当验证准确率 (用于掩码语言建模) 达到目标 72.0% 时, 停止训练, 并测量实际运行时间. 我们使用 FP16 精度和 Apex AMP (优化等级 O2) 进行训练.

我们将结果与 Nvidia 提交到 MLPerf 1.1 的训练速度报告进行比较 ([表 1](#table-01)).

我们使用 MLPerf 1.1 参考实现提供的相同训练/验证数据划分. 特别是, 我们在与 Nvidia 基线相同的 10000 个验证样本上进行评估.

我们在 8 个 $\times$ A100-80GB GPU 上训练模型. 每次训练运行耗时在 16 到 19 分钟之间, 我们取 10 次运行的结果平均值.

### E. 2 GPT-2

我们使用 Huggingface transformers 库和 Nvidia 的 Megatron-LM 仓库提供的 GPT-2 标准实现. [Radfor19] 我们遵循 Megatron-LM 仓库的训练方案.

我们使用有效批量大小为 512, 并使用梯度累积以适应可用 GPU 内存. 我们使用 AdamW 优化器, GPT-2 small 的学习率为 6e-4, GPT-2 medium 的学习率为 1.5e-4, 权重衰减为 0.1. 所有模型使用相同的超参数训练 400K 步. 我们所有实现都运行混合精度训练 (PyTorch AMP).

我们使用 Openwebtext 数据集, 并使用 GPT-2 BPE 分词器. 我们随机选择 0.5% 的数据集作为验证集, 其余作为训练集. 验证集的这一随机选择只进行一次, 所有模型都在相同的验证集上进行评估.

我们在 8 个 $\times$ A100-40GB GPU 上训练模型, 并测量实际训练时间. 训练 GPT-2 small 耗时在 2.7 到 9.5 天之间, 训练 GPT-2 medium 耗时在 6.9 到 21.0 天之间 ([表 2](#table-02)).

在 [图 4](#figure-04) 中, 我们绘制了 GPT-2 small/medium 在整个训练过程中使用 HuggingFace 实现或我们的 FlashAttention 实现的验证困惑度. 我们看到 FlashAttention 的表现与基线实现相同, 并且两种实现的验证困惑度曲线几乎重合.

<span id="figure-04"></span>

![参见图注](./flashattention/figure-04.png)

**图 4.** 使用两种实现方式的 GPT-2 small/medium 验证困惑度. 我们确认 FlashAttention 产生的验证曲线与 HuggingFace 的基线实现相同.

##### 长文档分类.

对于 MIMIC-III 和 ECtHR, 我们遵循 [Dai22] 中的超参数.

### E. 3 LRA 详细资料

我们遵循 Long-range arena 论文 [Tay20], [Long-range arena 仓库](https://github.com/google-research/long-range-arena)以及 Nyströmformer 复现 [Xiong21] 中的超参数. 为了给基线方法宽松, 如果我们无法在任意五个任务中的任何一个再现某个基线的性能, 我们将报告 [Tay20] 或 [Xiong21] 中该基线在该任务上的更好性能.

在超参数调优后, 几乎所有注意力方法在五个 LRA 任务上都达到类似的准确率.

我们对所有方法都使用混合精度训练, 除了 Performer (在混合精度下不稳定) 和 Local Attention (实现不支持 FP16).

为了计算总体实际时钟时间加速, 我们取五个任务的实际时钟时间加速的几何平均值.

##### Path-X

对于 Path-X 和 Path-256, 我们遵循 long-range arena 论文中 PathFinder-32 实验的超参数. [Tay20] 对于两者, 我们首先在 Path-64 上预训练一个模型. 我们在训练 200 个 epoch 后获取检查点, 将其位置嵌入上采样 (我们在空间中按网格复制位置嵌入), 并在下游任务上对其进行 200 个 epoch 的微调, 采用一个 epoch 的线性预热和余弦退火学习率. 对于 Path-X, 我们选择表现最好的检查点 (根据验证集准确率), 并额外使用相同的预热和学习率对其微调 200 个 epoch (这为 Path-X 的 FlashAttention 提高大约 4 个点的准确率, 但模型随后开始过拟合).

### E. 4 与 Apex FMHA 的比较

我们将我们的方法/实现与 [Apex FMHA](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha)进行比较。

当我们开始这个项目时, Apex FMHA 是我们所知道的最快的注意力实现, 专为长度最多为 512 的短序列量身定制. 实际上, 截止到 MLPerf 1.1 [Mattso20], 几乎所有针对 BERT 训练基准在 Nvidia GPU 上运行的 MLPerf 提交都在其模型代码中使用 FMHA. 由于 FMHA 针对 BERT 模型, 它只支持头部维度为 64, 并且只在 A100 GPU 上运行. FMHA 将注意力计算 $\mathrm{dropout}(\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^{\top})))\mathbf{V}$ 融合到一个 CUDA 内核中. 在前向传播中, 它将注意力矩阵 $\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^\top))$ 存储到 HBM 中, 以便在梯度计算中使用. 因此, 它并没有提供显著的内存节省 (尽管对于较短的序列, 内存占用通常不是主要关注点).

我们以 FMHA 代码作为起点, 并应用两种经过验证的技术 (平铺和重新计算) 来处理长序列并节省内存, 如 [第 3 节](#S3) 中提到的. 因此, 我们可以支持更长的序列 (例如, 长度最高可达 64K). 我们还支持更多的头维度 (16, 32, 64, 128) 和更广泛的 GPU 类型 (在撰写本文时的所有图灵和安培 GPU).

在 [表 7](#table-07) 中, 我们比较了 FlashAttention 和 Apex FMHA 在短序列上的性能 (因为 FMHA 只支持最多 512 的序列长度). 总体上, FlashAttention 在前向计算中略快于 FMHA, 在反向计算中略慢于 FMHA. 这是因为我们在前向计算中不存储注意力矩阵, 而在反向计算中重新计算它. 与 FMHA 相比, FlashAttention 的整体运行时间在序列长度为 128 时约慢 4%, 序列长度为 256 时快 8%, 序列长度为 512 时快 5%.

<span id="table-07"></span>

![论文原表 7](./flashattention/table-07.png)

**表 7.** FlashAttention 与 FMHA 在不同序列长度下的运行时间 (毫秒), 包含掩码和 dropout, 在 A100-SXM4-40GB GPU 上测量. 批量大小 64, 16 个头, 每个头维度 64 (即 BERT-large 尺寸).

### E. 5 不同硬件和配置下的加速

加速因 GPU 类型和代际不同而有所差异, 这取决于 HBM 带宽和 SRAM 大小. 在本节中, 我们将对不同 GPU 和配置下的 FlashAttention 加速情况进行分析.

<span id="figure-05"></span>

![参见图注](./flashattention/figure-05.png)

**图 5.** 在不同序列长度下, 相对于标准 PyTorch 注意力的加速情况, 使用 A100.

##### A100

[图 5](#figure-05) 展示了在 A100 GPU 上, 批量大小为 8, 头维度为 64, 12 个注意力头, 在不同序列长度下的加速情况. 我们一般看到 2-4 倍的加速, 当使用 dropout 和掩码(kernel fusion)时, 加速效果更明显.

<span id="figure-06"></span>

![参见图注](./flashattention/figure-06.png)

**图 6.** 在不同序列长度下, 相对于标准 PyTorch 注意力的加速情况, 使用 A100, 头维度为 128.

##### A100, 头维度 128

当增加头维度时, 加速也会变化. 每个块需要更多的内存, 因此我们需要使用更小的块大小以适应 SRAM. [图 6](#figure-06) 展示了在 A100 上头维度为 128 时的加速情况(批量大小 16, 12 个头). 总体上加速效果较小——但在使用因果掩码时, 我们仍然可以看到显著加速 (最高可达 3 倍), 因为一半的块被屏蔽.

<span id="figure-07"></span>

![参见图注](./flashattention/figure-07.png)

**图 7.** 在不同序列长度下, 相对于标准 PyTorch 注意力的加速情况, 使用 RTX 3090.

##### RTX 3090

[图 7](#figure-07) 显示了在 RTX 3090 GPU 上的加速情况. 在这里, 我们使用批量大小为 12, 注意力头数为 12. 由于 RTX 3090 的内存带宽低于 A100 (大约 900 GB/s 对 1.5 TB/s), 我们观察到在 RTX 3090 上的加速略高一些 (在 2.5-4.5 $\times$ 之间).

<span id="figure-08"></span>

![参见图注](./flashattention/figure-08a.png)

![参见图注](./flashattention/figure-08b.png)

**图 8.** 在不同序列长度下, 相较于标准 PyTorch 注意力的加速情况, 在 T4 上. 顶部: 前向传递和反向传递的组合. 底部: 仅前向传递.

##### T4

[图 8](#figure-08) 显示了在 T4 GPU 上的加速情况. T4 的 SRAM 比 A100 小, 因此我们需要在 FlashAttention 中减小块大小. 因此, 我们在 T4 上观察到的加速较少, 这与第 [3.2](#S3.SS2) 节中的 IO 复杂度分析相匹配. T4 GPU 通常用于推理, 因此我们还报告了仅前向传递的加速情况.

### E. 6 完整基准测试结果

我们报告了在 A100 上的完整基准测试结果和实验细节.

##### 基线

我们与 PyTorch/HuggingFace 和 Megatron 的精确注意力参考实现, 近似注意力和稀疏注意力进行了比较. 对于近似注意力, 我们与 Reformer [Kitaev20], Local Attention [Razavi20], Linformer Attention [Wang20], Smyrf [Daras20] 和 LongShortFormer (LSFormer) [Zhu21] 的参考实现进行了比较. 对于稀疏注意力, 我们与 OpenAI 的 Block-Sparse Attention [Child19], Longformer [Beltag20] 和 BigBird Attention [Zaheer20] 的参考实现进行了比较. 对于近似和稀疏注意力, 我们使用 1/8 的压缩比或压缩后的序列长度 256, 以较小者为准.

##### 设置

我们在一台配备 40 GB GPU HBM 的 A100 GPU 的机器上, 以 8 个维度为 64 的头, 批次大小为 16 测量注意力计算的运行时间和内存使用情况. 我们在实验中改变序列长度. 我们在随机向量上计算注意力 $\mathbf{Q}$, $\mathbf{K}$ 和 $\mathbf{V}$ (我们不测量隐藏层的投影). 对于 dropout, 我们使用 dropout 0.1; 对于掩码, 我们使用填充掩码, 其掩码长度在总序列长度和总序列长度减 20 之间均匀随机. 为了测量运行时间, 我们取注意力调用的 100 次测量平均值. 我们只测量内存占用一次, 因为其在多次运行间不变化.

我们报告了前向传播, 反向传播以及前后向传播组合的时间结果. 我们对每种方法分别测量了有无 dropout, masking 或两者同时存在的情况——Block Sparse, Longformer 和 BigBird 除外. 由于外部库的 bug, 这些方法在使用 masking 时无法成功运行反向传播, 因此为了宽容起见, 我们在测量它们时不使用 masking. 我们所有测量都使用 FP16, Local Attention 除外, 因为其实现仅支持 FP32.

对于每个基线方法, 我们增加序列长度, 直到 GPU 内存不足, 以下是例外情况: Megatron 实现不支持超过 2048 的序列长度. Block-Sparse (OpenAI) 不支持超过 4096 的序列长度. Longformer 和 BigBird 不支持超过 8092 的序列长度.

我们测量了在不使用 dropout 和 masking 的情况下, 前后向传播组合的内存使用情况.

##### 结果

[表 8](#table-08) 总结了所有实验配置, 并包含指向结果表的指针.

<span id="table-08"></span>

![论文原表 8](./flashattention/table-08.png)

**表 8.** 结果表指针.

<span id="table-09"></span>

![论文原表 9](./flashattention/table-09.png)

**表 9.** 各种精确/近似/稀疏注意力机制在不同序列长度下的前向传播运行时间 (毫秒), 包含丢弃和掩码. 最佳以粗体显示, 第二最佳加下划线.

<span id="table-10"></span>

![论文原表 10](./flashattention/table-10.png)

**表 10.** 各种精确/近似/稀疏注意力机制的反向传递运行时间 (毫秒), 按序列长度划分, 包含 dropout 和掩码. 最优值加粗, 次优值下划线.

<span id="table-11"></span>

![论文原表 11](./flashattention/table-11.png)

**表 11.** 各种精确/近似/稀疏注意力机制在不同序列长度下的前向传递和反向传递运行时间 (毫秒), 包含 dropout 和 mask. 最佳值加粗, 次佳值下划线.

<span id="table-12"></span>

![论文原表 12](./flashattention/table-12.png)

**表 12.** 各种精确/近似/稀疏注意力机制在不同序列长度下的前向传递运行时间 (毫秒), 含掩码. 最佳值加粗, 次佳值下划线.

<span id="table-13"></span>

![论文原表 13](./flashattention/table-13.png)

**表 13.** 各种精确/近似/稀疏注意力机制在不同序列长度下的反向传递运行时间 (毫秒), 带掩码. 最佳结果加粗, 次佳结果下划线处理.

<span id="table-14"></span>

![论文原表 14](./flashattention/table-14.png)

**表 14.** 不同序列长度下各种精确/近似/稀疏注意力机制的前向传递和反向传递运行时间 (毫秒), 带掩码. 最佳加粗, 第二最佳下划线.

<span id="table-15"></span>

![论文原表 15](./flashattention/table-15.png)

**表 15.** 各种精确/近似/稀疏注意力机制在不同序列长度下的前向传递运行时间 (毫秒), 带 dropout. 最佳用粗体表示, 第二佳用下划线表示.

<span id="table-16"></span>

![论文原表 16](./flashattention/table-16.png)

**表 16.** 各种精确/近似/稀疏注意力机制在不同序列长度下的反向传播运行时间 (毫秒), 含 dropout. 最佳加粗, 次佳加下划线.

<span id="table-17"></span>

![论文原表 17](./flashattention/table-17.png)

**表 17.** 各种精确/近似/稀疏注意力机制在不同序列长度下的前向传递和反向传递运行时间 (毫秒), 含 dropout. 最佳值加粗, 次佳值下划线.

<span id="table-18"></span>

![论文原表 18](./flashattention/table-18.png)

**表 18.** 各种精确/近似/稀疏注意力机制的前向传递运行时间 (毫秒), 按序列长度分类. 最佳为粗体, 次优为下划线.

<span id="table-19"></span>

![论文原表 19](./flashattention/table-19.png)

**表 19.** 各种精确/近似/稀疏注意力机制的反向传播运行时间 (毫秒) 按序列长度列出. 最佳值加粗, 第二最佳值加下划线.

<span id="table-20"></span>

![论文原表 20](./flashattention/table-20.png)

**表 20.** 按序列长度划分的各种精确/近似/稀疏注意力机制的前向传递后向传递运行时间 (ms). 最好用加粗, 第二好的下划线.

<span id="table-21"></span>

![论文原表 21](./flashattention/table-21.png)

**表 21.** 各种精确/近似/稀疏注意力机制在不同序列长度下的内存使用 (MB). 最佳结果加粗, 第二佳结果下划线.

[+1]: FlashAttention 代码可在 [FlashAttention GitHub 仓库](https://github.com/HazyResearch/flash-attention) 获取.

[+2]: 这种聚合方式称为*代数聚合* [Gray97].

[+3]: LRA 精度结果已知高度依赖于调优过程 [Xiong21]. 我们复现的基线表现优于原始比较中报告的结果 [Tay20].

[+4]: Path-256 需要更长的序列, 但路径相对比 Path-X 短, 因此更容易获得更高的精度.
