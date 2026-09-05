---
title: 'Gated Linear Attention'
createTime: 2026/09/05 22:11:56
permalink: /papers/gated-linear-attention/
pageClass: paper-reading
---

> [Songlin Yang](https://sustcsonglin.github.io/) [+author-note], [Bailin Wang](https://berlino.github.io/) [+author-note], [Yikang Shen](https://dblp.org/pid/152/8226), [Rameswar Panda](https://rpand002.github.io/), [Yoon Kim](https://people.csail.mit.edu/yoonkim/). 2023 年 12 月 11 日首次提交至 arXiv, 当前版本为 v6. 2024 年 7 月发表于 *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:56501-56523. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635). <a href="/paper/gated-linear-attention.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [ICML 2024](https://proceedings.mlr.press/v235/yang24ab.html). [DOI](https://doi.org/10.48550/arXiv.2312.06635). [TeX 源码](https://export.arxiv.org/e-print/2312.06635v6). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

采用线性注意力的 Transformer 既能高效并行训练, 又能写成具有二维矩阵隐状态的 RNN, 因而推理复杂度随序列长度线性增长. 但线性注意力的效果通常不及普通 softmax 注意力; 现有实现也没有充分考虑 I/O, 实际速度反而慢于高度优化的 softmax 注意力. 本文提出一种硬件高效的线性注意力算法, 在内存搬运与并行能力之间取舍. 其实现称为 FlashLinearAttention, 即便序列很短 (如 1K), 作为独立层也快于 FlashAttention-2 [Dao23b]. 作者进一步把算法推广到表达能力更强、带数据依赖门控的线性注意力. 用它替换 Transformer 的标准注意力层后, 得到的门控线性注意力 (GLA) Transformer 在中等规模语言建模实验中可与 LLaMA 架构的 Transformer [Tou23] 以及 RetNet [Sun23b]、Mamba [Gu23] 等近期线性时间推理基线竞争. GLA Transformer 尤其擅长长度泛化: 在 2K 序列上训练的模型可以推广到 20K 以上, 困惑度没有明显恶化. 训练速度方面, GLA Transformer 的吞吐量也高于规模相近的 Mamba 模型.

[https://github.com/sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

<span id="section-1"></span>

## 1 引言

采用 softmax 注意力的 Transformer [Vas17a] 可以高效并行训练, 但复杂度随序列长度平方增长, 这推动了能在线性时间内处理序列、形式更接近 RNN 的模型. 线性注意力把指数相似度函数换成 (可能先经变换的) 键向量与查询向量的简单点积, 已成为经典 softmax 注意力的一种有前景的替代方案 [Kat20, Cho20a, Kas21, Pen21]. 它的一个重要性质是存在“递归形式”: 可以把它写成具有二维隐状态的线性 RNN [Kat20], 从而实现线性时间推理. 训练时, 线性注意力还可以采用次二次复杂度的“分块并行形式”. 这种形式把序列划分为互不重叠的块, 先串行计算块间递归, 再并行计算块内结果 [Hua22a, Sun23b, Lin23], 因此仍保留一部分并行训练能力. 不过, 现有线性注意力算法没有考虑 I/O, 在中等序列长度下实际慢于优化过的 softmax 注意力实现 [Dao22, Dao23b].

从效果看, 线性注意力通常不及普通 softmax 注意力, 在语言建模中差距往往很大 [Kas21]. RetNet [Sun23b]、TransNormerLLM [Qin23c] 等近期变体会在 RNN 更新前给当前隐状态乘上衰减因子, 因而取得明显提升. 但它们使用的是全局且*不依赖数据*的衰减因子, 而在一维 RNN 中, *依赖数据*的门控机制已被证明对性能至关重要 [Wes18, Qin23a]. 即便加入衰减因子, 从头预训练的线性注意力 Transformer 仍落后于最强的 Transformer 架构.

本文为线性注意力设计了一种硬件高效算法, 并用它训练带门控的线性注意力变体, 使其能够与 softmax 注意力竞争. 作者先讨论如何在现代 GPU 上优化普通线性注意力, 再据此给出两种适配不同训练场景的 I/O 感知算法 ([第 3 节](#section-3)). 其实现 FlashLinearAttention 即使面对 1K 一类短序列也快于 FlashAttention-2 [Dao23b]. 随后, 作者给出带数据依赖门控的线性注意力层, 并说明如何把 FlashLinearAttention 推广到门控情形 ([第 4 节](#section-4)). 实验在中等规模语言建模基准上研究由此得到的*门控线性注意力 (GLA) Transformer*: 分别用 15B 和 100B token 训练 340M 与 1.3B 参数模型. 结果表明, GLA Transformer 相比采用近期训练配方的强 LLaMA 架构 Transformer 基线 [Tou23], 以及 RetNet [Sun23b]、Mamba [Gu23] 等近期线性时间序列模型, 都有良好表现. 在线性递归模型中, GLA Transformer 的长度泛化与高召回需求任务尤其突出; 它的训练吞吐量也明显高于规模相近的 Mamba 模型.

<span id="section-2"></span>

## 2 背景: 线性注意力

先简要介绍线性注意力层. 记号方面, 粗体大写字母表示矩阵 (如 ${\mathbf{S}}$、${\mathbf{Q}}$), 粗体小写字母表示向量 (如 ${\bm{q}}_{t}$、${\bm{k}}_{t}$), 斜体大写字母表示可学习的参数矩阵 (如 ${\bm{W}}_{K}$). 矩阵的行通常沿用同一字母: 例如 ${\bm{q}}_{t}$ 是 ${\mathbf{Q}}$ 的第 $t$ 行.

<span id="section-2-1"></span>

### 2.1 并行形式与递归形式

标准自回归 Transformer 使用 softmax 注意力. 给定输入序列 ${\mathbf{X}}\in\mathbb{R}^{L\times d}$ (其中 $L$ 为长度, $d$ 为隐层维度), 输出 ${\mathbf{O}}\in\mathbb{R}^{L\times d}$ 的计算方式为

$$
\begin{aligned}
{\mathbf{Q}},{\mathbf{K}},{\mathbf{V}} & ={\mathbf{X}}{\bm{W}}_{Q},{\mathbf{X}}{\bm{W}}_{K},{\mathbf{X}}{\bm{W}}_{V}, \\
{\mathbf{O}} & =\mathrm{softmax}\big(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\mathbf{M}}\big){\mathbf{V}},
\end{aligned}
$$

其中 ${\bm{W}}_{Q},{\bm{W}}_{K},{\bm{W}}_{V}\in\mathbb{R}^{d\times d}$ 是可学习矩阵, ${\mathbf{M}}\in\{-\infty,1\}^{L\times L}$ 是阻止模型关注未来 token 的掩码: $i\geq j$ 时 ${\mathbf{M}}_{ij}=1$, $i<j$ 时 ${\mathbf{M}}_{ij}=-\infty$. 为简单起见, 这里假设只有一个注意力头. 给定完整输入 ${\mathbf{X}}$ 后, 上述注意力*并行形式*可以并行计算 ${\mathbf{O}}$, 因而便于高效训练. 但推理时 Transformer 必须使用下面的*递归形式*:

$$
\begin{aligned}
{\bm{q}}_{t},\ {\bm{k}}_{t},\ {\bm{v}}_{t} & ={\bm{x}}_{t}{\bm{W}}_{Q},\ {\bm{x}}_{t}{\bm{W}}_{K},\ {\bm{x}}_{t}{\bm{W}}_{V}, \\
{\bm{o}}_{t} & =\frac{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top){\bm{v}}_{i}}{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top)},
\end{aligned}
$$

该式根据当前 token 的表示 ${\bm{x}}_{t}\in\mathbb{R}^{1\times d}$ 计算查询 ${\bm{q}}_{t}$、键 ${\bm{k}}_{t}$ 与值 ${\bm{v}}_{t}$, 再对持续增长的键集合 $\{{\bm{k}}_{1},\dots,{\bm{k}}_{t}\}$ 和值集合 $\{{\bm{v}}_{1},\dots,{\bm{v}}_{t}\}$ (即“KV cache”) 计算注意力.

线性注意力机制 [Kat20] 用带特征映射 $\phi$ 的核 $k({\bm{x}},{\bm{y}})$ 替换 $\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top)$, 其中 $k({\bm{x}},{\bm{y}})=\langle\phi({\bm{x}}),\phi({\bm{y}})\rangle$. 这样便可简化 ${\bm{o}}_{t}$ 的计算:

$$
\begin{aligned}
{\bm{o}}_{t} & =\frac{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}}{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^\top}=\frac{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}}{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top}.
\end{aligned}
$$

令 ${\mathbf{S}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}$、${\bm{z}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top$, 其中 ${\mathbf{S}}_{t}\in\mathbb{R}^{d\times d}$、${\bm{z}}_{t}\in\mathbb{R}^{d\times 1}$, 上式可改写为 RNN:

$$
\begin{aligned}
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1} & +\phi({\bm{k}}_{t})^\top{\bm{v}}_{t},\hskip 2.84526pt{\bm{z}}_{t}={\bm{z}}_{t-1}+\phi({\bm{k}}_{t})^\top,\hskip 2.84526pt{\bm{o}}_{t}=\frac{\phi({\bm{q}}_{t}){\mathbf{S}}_{t}}{\phi({\bm{q}}_{t}){\bm{z}}_{t}}.
\end{aligned}
$$

尽管已有工作研究了多种核 [Kas21, Pen21], 近期研究发现, 不带归一化项的线性核 (即把 $\phi$ 设为恒等映射) 在实践中效果很好 [Sun23b]. 由此得到未归一化的线性注意力层, 更新式为

<span id="equation-01"></span>

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+{\bm{k}}_{t}^\top{\bm{v}}_{t},\quad{\bm{o}}_{t}={\bm{q}}_{t}{\mathbf{S}}_{t}.
$$

[公式 1](#equation-01) 清楚表明, 线性注意力层本质上是具有矩阵隐状态 ${\mathbf{S}}_{t}$ 的线性递归层, 该状态通过外积 ${\bm{k}}_{t}^\top{\bm{v}}_{t}=({\bm{x}}_{t}{\bm{W}}_{K})^\top({\bm{x}}_{t}{\bm{W}}_{V})$ 更新. [+1] 因果线性注意力的并行形式仍具有关于 $L$ 的二次复杂度: ${\mathbf{O}}=\big(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\mathbf{M}}\big){\mathbf{V}}$. 这里 ${\mathbf{M}}\in\{0,1\}^{L\times L}$ 是掩码, $i\geq j$ 时 ${\mathbf{M}}_{ij}=1$, $i<j$ 时 ${\mathbf{M}}_{ij}=0$. 由于存在 ${\mathbf{M}}$, 不能利用矩阵乘法结合律把并行形式的复杂度从二次降为线性. [+2]

<span id="section-2-2"></span>

### 2.2 分块并行形式

线性注意力的*分块*并行形式在并行形式与递归形式之间取得平衡 [Hua22a, Sun23b], 可以做次二次复杂度、部分并行的训练. 形式化地说, 将输入 ${\mathbf{X}}$ 划分为互不重叠、长度均为 $C$ 的块. 令 ${\mathbf{S}}_{[i]}\in\mathbb{R}^{d\times d}$ 表示处理完 $i$ 个块后的块级隐状态, 即 ${\mathbf{S}}_{[i]}:={\mathbf{S}}_{iC}$. 再令 ${\mathbf{Q}}_{[i]}:={\mathbf{Q}}_{iC+1:(i+1)C+1}\in\mathbb{R}^{C\times d}$ 表示第 $i$ 块对应的查询向量, ${\mathbf{K}}_{[i]}$、${\mathbf{V}}_{[i]}$、${\mathbf{O}}_{[i]}$ 同理. 对 $i\in[0,1,\dots\frac{L}{C}-1]$, 块间递归为

<span id="equation-02"></span>

$$
{\mathbf{S}}_{[i+1]}={\mathbf{S}}_{[i]}+\underbrace{\sum_{j=iC+1}^{(i+1)C}{\bm{k}}_{j}^\top{\bm{v}}_{j}}_{{\mathbf{K}}^\top_{[i]}{\mathbf{V}}_{[i]}}\quad\hskip 2.84526pt\in\mathbb{R}^{d\times d}.
$$

${\mathbf{S}}_{[0]}$ 可以初始化为零, 也可以取上一段的隐状态. 一个块内全部 RNN 输入之和 ${\mathbf{K}}^\top_{[i]}{\mathbf{V}}_{[i]}$ 可用 $O(C^{2}d)$ 并行算出. 输出的块内并行计算为

$$
{\mathbf{O}}_{[i+1]}=\underbrace{{\mathbf{Q}}_{[i+1]}{\mathbf{S}}_{[i]}}_{\mathrm{inter-chunk}:{\mathbf{O}}^{\mathrm{inter}}_{[i+1]}}+\underbrace{\big(({\mathbf{Q}}_{[i+1]}{\mathbf{K}}_{[i+1]}^\top)\odot{\mathbf{M}}\big){\mathbf{V}}_{[i+1]}}_{\mathrm{intra-chunk}:{\mathbf{O}}^{\mathrm{intra}}_{[i+1]}},
$$

其中 ${\mathbf{O}}_{[i+1]}\in\mathbb{R}^{C\times d}$. “块内”分量 ${\mathbf{O}}^{\mathrm{intra}}_{[i+1]}$ 与 [公式 1](#equation-01) 的并行形式完全相同, 耗时 $O(C^{2}d+Cd^{2})$; “块间”分量 ${\mathbf{O}}^{\mathrm{inter}}_{[i+1]}$ 计入上一块隐状态的贡献, 耗时 $O(Cd^{2})$. 因而训练复杂度为 $O\left(\frac{L}{C}(C^{2}d+Cd^{2})\right)=O(L C d+L d^{2})$. 当 $L>d$ 时, 它小于 $O(L^{2}d)$. $C=L$ 时恢复并行形式, $C=1$ 时恢复递归形式.

<span id="section-3"></span>

## 3 硬件高效的线性注意力

本节介绍 FlashLinearAttention, 一种沿袭 FlashAttention [Dao22, Dao23b] 思路、兼顾 I/O 的硬件高效线性注意力算法. 首先讨论实现真正高效的算法时必须考虑的硬件特性.

<span id="section-3-1"></span>

### 3.1 硬件高效算法的原则

高效算法需要了解现代硬件的计算模型、内存层次和专用计算单元.

**占用率.** GPU 会并行执行大量线程; 线程组成线程块, 再由流式多处理器 (SM) 执行. 要维持较高的 GPU 占用率 (即被使用的 GPU 资源比例), 就必须启用足够多的 SM. 大规模训练与长序列建模的 batch size 往往较小, 此时沿时间维并行有助于保持高占用率 [Dao23b].

**专用计算单元.** 现代神经网络训练硬件通常配有专用计算单元, 例如 NVIDIA GPU 的 tensor core 和 TPU 的矩阵乘法单元, 可以显著加速矩阵乘法. 以 A100 为例, tensor core 上的半精度矩阵乘法约比 CUDA core 快 16 倍. 这些专用单元对大规模训练很重要.

**内存层次.** GPU 的内存层次包括容量较大但较慢的全局显存 (高带宽内存, HBM), 以及容量较小但更快的共享内存 (SRAM). 因此, 充分利用 SRAM、减少 HBM I/O 能带来明显加速.

<span id="section-3-2"></span>

### 3.2 线性注意力的硬件考量

下面分别讨论各种线性注意力形式的硬件效率.

**递归形式.** 递归形式的基本实现会把所有时间步的二维隐状态存入 HBM, I/O 成本很高 [Mao22]. 可以像 [Kat20] 那样不物化这些状态, 在反向传播时重新计算, 从而减少 I/O; 但递归更新中的逐元素操作无法利用 tensor core, 算术强度也很低. 因此, 虽然递归形式在三种形式中通常 FLOPs 最少, 实际墙钟时间却未必更短. 理论上还可以用并行扫描算法并行计算线性递归, 可这种方法需要物化每个时间步的二维隐状态. 大量内存 I/O 会抵消沿序列维并行的收益, 实际运行仍然很慢 [Kat23].

**并行形式.** [Qin23c] 表明, 采用类似的 I/O 优化后, 并行形式可以达到与 FlashAttention 相近的效率. 但其二次复杂度带来大量 FLOPs, 长序列训练成本很高, 这与 softmax 注意力的朴素实现存在同样的问题.

**分块形式.** 分块并行形式通过额外参数 $C$ 在并行形式与递归形式之间插值, 更便于细致权衡上述因素. 与递归形式不同, 它的大多数操作都可写成矩阵乘法; 只要把 $C$ 设为 16 的倍数, 就能使用 tensor core. 文献已讨论过分块训练算法 [Hua22a, Sun23b], 但多数实现不感知 I/O, 因而在 2K-4K 等中等序列长度上慢于 FlashAttention.

<span id="figure-01"></span>

![图 1. FlashLinearAttention 的非物化与物化版本.](./gated-linear-attention/figure-01.png)

**图 1.** (a) 不物化隐状态的 FlashLinearAttention, 内存效率更高. (b-c) 物化隐状态的 FlashLinearAttention, 可以沿序列维做分块并行.

<span id="section-3-3"></span>

### 3.3 FlashLinearAttention: 采用分块形式的硬件高效线性注意力

下面给出分块形式的 I/O 感知硬件高效实现. 根据是否把块级隐状态 ${\mathbf{S}}_{[n]}$ 物化到 HBM, 前向和反向传播各有两个版本. 前向传播见[算法 1](#algorithm-01)和[图 1](#figure-01), 反向传播见附录[算法 2](#algorithm-02). 总体上, 算法用 tiling 逐块加载张量, 并在片上复用张量块, 尽量避免重复访问 HBM. 例如, ${\mathbf{Q}}_{[n]}$ 载入 SRAM 后, ${\mathbf{Q}}_{[n]}{\mathbf{S}}$ 与 $({\mathbf{Q}}_{[n]}{\mathbf{K}}_{[n]}^{\top}\odot{\mathbf{M}}){\mathbf{V}}_{[n]}$ 都能在片上完成, 无须再次加载 ${\mathbf{Q}}_{[n]}$, 因而节省 HBM I/O.

<span id="algorithm-01"></span>

**算法 1: FlashLinearAttention 前向传播.**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf V}\in\mathbb R^{L\times d}$, 块大小 $C\in[L]$, `materialize` $\in\{$`True`,`False`$\}$.
- 将 ${\mathbf Q},{\mathbf K},{\mathbf V}$ 各自划分为 $N=L/C$ 个 $C\times d$ 块.
- 在 SRAM 中初始化 ${\mathbf S}=\mathbf 0\in\mathbb R^{d\times d}$, 并在片上构造因果掩码 ${\mathbf M}\in\mathbb R^{C\times C}$.
- **若** `materialize` (物化版本):
  - **循环** $n\gets1,N$:
    - 将 ${\mathbf S}$ 作为 ${\mathbf S}_{[n]}$ 存入 HBM.
    - 将 ${\mathbf K}_{[n]},{\mathbf V}_{[n]}$ 从 HBM 载入 SRAM.
    - 在片上计算 ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
  - **并行循环** $n\gets1,N$:
    - 将 ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf S}_{[n]}$ 从 HBM 载入 SRAM.
    - 在片上计算 ${\mathbf O}'={\mathbf Q}_{[n]}{\mathbf S}_{[n]}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M}){\mathbf V}_{[n]}$.
    - 将 ${\mathbf O}'$ 作为 ${\mathbf O}_{[n]}$ 存入 HBM.
  - **返回** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$ 与 ${\mathbf S}=\{{\mathbf S}_{[1]}\dots{\mathbf S}_{[N]}\}$.
- **否则** (非物化版本):
  - **循环** $n\gets1,N$:
    - 将 ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]}$ 从 HBM 载入 SRAM.
    - 在片上计算 ${\mathbf O}'={\mathbf Q}_{[n]}{\mathbf S}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M}){\mathbf V}_{[n]}$.
    - 在片上计算 ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
    - 将 ${\mathbf O}'$ 作为 ${\mathbf O}_{[n]}$ 存入 HBM.
  - **返回** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$.

</div>

**非物化版本**按 $n\in[N]$ 依次计算 ${\mathbf{O}}_{[n]}$, 并用 SRAM 临时保存 ${\mathbf{S}}_{[n]}$, 因而更节省内存. 它可以沿 batch size、头数与头维度并行, 却不能沿序列维并行. batch size 较大时, 这些并行维度足以保持较高的 GPU 占用率; 长序列与大规模训练的 batch size 较小, 此时无法充分利用 SM. **物化版本**先计算块间递归 ([公式 2](#equation-02)), 把所有 $n\in[N]$ 的 ${\mathbf{S}}_{[n]}$ 存入 HBM, 随后并行计算各块的 ${\mathbf{O}}_{[n]}$. 这种做法并行性更好, 但内存占用会增加约 10%-20%. 作者通过*重计算*缓解这一问题: 前向传播结束后丢弃隐状态, 反向传播时再算一遍. 实验发现它只增加少量运行时间, 却能显著减少内存占用, 因此默认采用这一策略.

<span id="figure-02"></span>

![图 2. 单张 H100 GPU 上的速度比较.](./gated-linear-attention/figure-02.png)

**图 2.** 单张 H100 GPU 上的速度对比: batch size 为 32, 头数为 16, 头维度与块大小均为 64. 横纵轴均采用对数刻度. *w/ m.* 与 *w/o m.* 分别表示 FlashLinearAttention 会与不会把隐状态物化到 HBM.

[图 2](#figure-02) 给出了实现的速度和内存占用. 两个 FlashLinearAttention 版本都明显快于 FlashAttention-2 [Dao23b] 以及纯 PyTorch (即不感知 I/O) 的分块线性注意力实现, 说明 I/O 感知优化确实有效.

<span id="section-4"></span>

## 4 门控线性注意力

[公式 1](#equation-01) 中的线性递归没有衰减项或遗忘门, 而这两者对 RNN 的性能十分重要 [Bec24, Cho14a, Wes18]. 没有衰减项, 模型就很难“忘记”信息; 有研究推测, 这也是线性注意力在长上下文任务中不稳定的部分原因 [Buc24]. 近期工作 [Sun23b, Qin23c] 给线性注意力加入全局且*不依赖数据*的衰减因子 [+3] $\gamma\in(0,1)$, 即 ${\mathbf{S}}_{t}=\gamma{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^\top{\bm{v}}_{t}$, 从而提升性能. 使用单一 $\gamma$ 是为了保留类似注意力的并行形式, 便于高效训练. 本文改为考虑依赖数据的线性注意力门控机制, 并证明即使门控因子的表达能力更强, 得到的门控线性注意力 (GLA) 层仍有适合高效训练的硬件高效分块形式.

<span id="section-4-1"></span>

### 4.1 GLA 的递归形式与并行形式

<span id="table-01"></span>

![表 1. 近期模型中的门控线性注意力形式.](./gated-linear-attention/table-01.png)

**表 1.** 近期模型的门控线性注意力形式; 各模型对 ${\mathbf{G}}_{t}$ 的参数化不同. 表中省略偏置项.

**递归形式.** GLA 具有随时间变化的二维遗忘门 ${\mathbf{G}}_{t}\in(0,1)^{d_{k}\times d_{v}}$:

$$
{\mathbf{S}}_{t}={\mathbf{G}}_{t}\odot{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t},
$$

这里允许隐状态使用不同维度. 这种基于 Hadamard 积的递归形式十分通用, 涵盖[表 1](#table-01) 所列的多种近期二维隐状态 RNN.

门控线性注意力设计的核心是 ${\mathbf{G}}_{t}$ 的参数化, 需要在*参数效率*、*状态大小*和*训练效率*之间取得平衡. 若直接用 ${\bm{x}}_{t}\mapsto{\mathbf{G}}_{t}$ 得到依赖数据的门控矩阵, 就需要大小为 $d\cdot d_{k}\cdot d_{v}$ 的矩阵, 参数效率很低. [Mao22] 提出更高效、基于外积的低秩参数化 ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}{\bm{\beta}}_{t}$, 只需 $d\cdot d_{v}+d\cdot d_{k}$ 个参数. [+4]

Mamba [Gu23] 将*数据无关*的可学习矩阵 ${\bm{A}}$ 与数据依赖向量 ${\bm{\alpha}}_{t}$ 结合得到 ${\mathbf{G}}_{t}$, 因而门控矩阵可以是满秩的. 但如 [Dao24] 所述, 这种形式无法改写成矩阵乘法, 也就不能使用 tensor core. 缺少紧凑的矩阵乘法形式意味着必须物化每个时间步的隐状态. 为降低高昂的 I/O 成本, [Gu23] 设计了硬件感知算法, 只在 SRAM 而非 HBM 中物化隐状态. SRAM 容量有限, 这种方案无法扩展到更大的隐状态; 后文实验表明, 这会使高召回需求任务的效果不够理想. Mamba-2 [Dao24] 用限制更强的门控机制解决这一问题: ${\mathbf{G}}_{t}=\gamma_{t}\mathbf{1}^\top\mathbf{1}$, 其中 $\gamma_{t}\in(0,1)$ 为标量. 递归由此可以改写成矩阵乘法, 使用 tensor core 和更大的状态. [Pen21]、[Sun24b] 与 [Bec24] 也采用这种*标量*数据依赖门控.

本文在标量参数化与完整低秩参数化之间取中间方案, 采用 ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}\mathbf{1}$. [+5] 得到的递归形式为

<span id="equation-03"></span>

$$
{\mathbf{S}}_{t}=({\bm{\alpha}}_{t}^{\top}\mathbf{1})\odot{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t}=\mathrm{Diag}({\bm{\alpha}}_{t}){\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t},
$$

${\bm{\alpha}}_{t}$ 由作用于 ${\bm{x}}_{t}$ 的低秩线性层与随后的 sigmoid 参数化, 详见 [第 4.4 节](#section-4-4). 上述形式具有通用性, 涵盖多种近期 RNN [Kat23, Qin24a, Pen24a], 因而下文的硬件高效 GLA 实现可以直接用于其他模型, 或经调整后使用.

**并行形式.** 为了沿序列长度并行, 下面给出 GLA 的并行形式. 展开 [公式 3](#equation-03) 可得

$$
{\mathbf{S}}_{t}=\sum_{i=1}^{t}\left(\left(\prod_{j=i+1}^{t}{\bm{\alpha}}_{j}^{\top}\mathbf{1}\right)\odot{\bm{k}}_{i}^{\top}{\bm{v}}_{i}\right)
$$

令 ${\bm{b}}_{t}:=\prod_{j=1}^{t}{\bm{\alpha}}_{j}$, 上式可改写为

$$
\begin{aligned}
{\bm{o}}_{t}={\bm{q}}_{t}{\mathbf{S}}_{t} & ={\bm{q}}_{t}\sum_{i=1}^{t}\left(\left(\frac{{\bm{b}}_{t}}{{\bm{b}}_{i}}\right)^{\top}\mathbf{1}\right)\odot{\bm{k}}_{i}^{\top}{\bm{v}}_{i} \\
=\sum_{i=1}^{t}({\bm{q}}_{t}\odot{\bm{b}}_{t})\left(\frac{{\bm{k}}_{i}}{{\bm{b}}_{i}}\right)^{\top}{\bm{v}}_{i}
\end{aligned}
$$

其中除法均为逐元素运算. 将 ${\bm{b}}_{t}$ 堆叠成矩阵 ${\mathbf{B}}\in(0,1)^{L\times d}$ 后, 并行形式为

$$
{\mathbf{O}}=\left(\left(\underbrace{({\mathbf{Q}}\odot{\mathbf{B}})\left(\frac{{\mathbf{K}}}{{\mathbf{B}}}\right)^{\top}}_{\mathbf{P}}\right)\odot{\mathbf{M}}\right){\mathbf{V}}.
$$

这种形式数值不稳定. ${\bm{b}}_{t}$ 是门值 ${\bm{\alpha}}_{j}\in(0,1)^{1\times d}$ 的累乘, $t$ 较大时会极小, 导致 $\frac{\mathbf{K}}{{\mathbf{B}}}$ 爆炸. 为此, 可以在对数空间计算 $\mathbf{P}$: [+6]

<span id="equation-04"></span>

$$
\mathbf{P}_{ij}=\sum_{k=1}^{d}\mathbf{Q}_{ik}\mathbf{K}_{jk}\,\exp(\log{\mathbf{B}}_{ik}-\log{\mathbf{B}}_{jk}),\quad i\geq j.
$$

其中 $k$ 表示特征索引. 与普通线性注意力不同, [公式 4](#equation-04) 不能用标准矩阵乘法表示, 因而无法利用 tensor core 的半精度矩阵乘法. [第 4.3 节](#section-4-3) 将介绍二级分块机制: 在保持数值稳定的同时, 让大部分计算仍能使用半精度矩阵乘法, 如[图 3](#figure-03) 所示.

<span id="figure-03"></span>

![图 3. GLA 分块计算的注意力式示意图.](./gated-linear-attention/figure-03.png)

**图 3.** 用类似注意力图的方式展示 GLA 分块计算. 灰色的块间依赖不会在分块形式中直接计算, 只在并行形式中计算. 块内依赖通过二级分块或 tiling 建模: 橙色的子块间部分采用半精度矩阵乘法, 粉色的子块内部分在对数空间以全精度计算.

<span id="section-4-2"></span>

### 4.2 GLA 的分块并行形式

与基本线性注意力的分块形式类似 ([第 2.2 节](#section-2-2)), GLA 也可以推导出分块形式. 块内操作在块级实现上述并行形式, 得到 ${\mathbf{O}}^{\mathrm{intra}}$; 块间计算则为

$$
\begin{aligned}
\mathbf{\Lambda}_{iC+j} & =\frac{{\bm{b}}_{iC+j}}{{\bm{b}}_{iC}},\mathbf{\Gamma}_{iC+j}=\frac{{\bm{b}}_{(i+1)C}}{{\bm{b}}_{iC+j}},{\bm{\gamma}}_{i+1}=\frac{{\bm{b}}_{(i+1)C}}{{\bm{b}}_{iC}}, \\
{\mathbf{S}}_{[i+1]} & =\left({\bm{\gamma}}_{i+1}^{\top}\mathbf{1}\right)\odot{\mathbf{S}}_{[i]}+\left({\mathbf{K}}_{[i+1]}\odot\mathbf{\Gamma}_{[i+1]}\right)^{\top}{\mathbf{V}}_{[i+1]}, \\
{\mathbf{O}}^{\mathrm{inter}}_{[i+1]} & =\left({\mathbf{Q}}_{[i+1]}\odot\mathbf{\Lambda}_{[i+1]}\right){\mathbf{S}}_{[i]}.
\end{aligned}
$$

直观地说, $\mathbf{\Lambda}_{[i+1]}$ 编码从块首开始的累积衰减, 用于传播上一块的隐状态 ${\mathbf{S}}_{[i]}$; $\mathbf{\Gamma}_{[i+1]}$ 编码直到块尾的衰减, 用于累积将加入下一隐状态 ${\mathbf{S}}_{[i+1]}$ 的信息.

<span id="section-4-3"></span>

### 4.3 硬件高效的 GLA

有了分块形式, 就能把 [第 3 节](#section-3) 的 FlashLinearAttention 算法推广到门控情形. 这一推广还依赖下述两项关键技术. 本节先给出高层直觉, 完整算法见附录[第 9 节](#section-9)的[算法 3–6](#algorithm-03).

**二级分块.** 与普通线性注意力不同, GLA 的块内计算包含对数空间运算 ([公式 4](#equation-04)), 不能直接使用半精度矩阵乘法和 tensor core. 为了更充分地利用 tensor core, 本文沿用经典 tiling 思路 [Dao22], 再把每个块划分成子块. 随后按块计算类似注意力的矩阵 ${\mathbf{P}}\in\mathbb{R}^{L\times L}$, 如[图 3](#figure-03) 所示. 具体而言, 子块间交互采用半精度矩阵乘法: [+7]

$$
\begin{aligned}
{\mathbf{P}}_{[i][j]} & =\Big({\mathbf{Q}}_{[i]}\odot{\mathbf{\Lambda}}_{[i]}\Big)\Big({\mathbf{K}}_{[j]}\odot{\mathbf{\Gamma}}_{[j]}\odot\frac{{\bm{b}}_{iC}}{{\bm{b}}_{(j+1)C}}\Big)^\top\in\mathbb{R}^{C\times C}.
\end{aligned}
$$

这对应[图 3](#figure-03) 中的橙色块. 对粉色的子块内部分, 为保证稳定性, 仍须按照 [公式 4](#equation-04) 以全精度计算矩阵乘法. 两级 tiling 大幅减少了非半精度矩阵乘法的 FLOPs, 从而缩短墙钟时间. Pytorch 风格的伪代码见附录[第 9 节](#section-9)的[代码 1](#listing-01).

**内存高效的 ${\mathbf{d}\bm{\alpha}}_{t}$ 计算.** 以往工作 [Mao22] 认为, 由于 ${\mathbf{d}\bm{\alpha}}_{t}=({\mathbf{S}}_{t-1}\odot\mathbf{d}{\mathbf{S}}_{t})\mathbf{1}$, GLA 类模型要计算全部梯度 ${\mathbf{d}\bm{\alpha}}_{t}$, 就必须在 HBM 中物化大小为 $L\times d\times d$ 的矩阵隐状态. 本文改为给出 ${\mathbf{d}\log\bm{\alpha}}_{t}$ 的如下*闭式*表达:

$$
\begin{aligned}
{\mathbf{d}\log\bm{b}}_{t} & ={\bm{q}}_{t}\odot{\mathbf{d}\bm{q}}_{t}-{\bm{k}}_{t}\odot{\mathbf{d}\bm{k}}_{t},\hskip 11.38109pt{\mathbf{d}\log\bm{\alpha}}_{t}=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm{b}}_{i},
\end{aligned}
$$

对 [公式 4](#equation-04) 求导即可得到该式, 完整推导见附录[第 9 节](#section-9). ${\mathbf{d}\bm{q}}_{t}$ 与 ${\mathbf{d}\bm{k}}_{t}$ 可按[算法 4 和 6](#algorithm-04)计算.

<span id="section-4-4"></span>

### 4.4 GLA Transformer

将 GLA 层推广到多头情形. 给定 $H$ 个头, 对每个 $h\in[1,H]$ 有

$$
\begin{aligned}
{\mathbf{S}}^{h}_{t}=\left(\left({\bm{\alpha}}_{t}^{h}\right)^{\top}\mathbf{1}\right)\odot{\mathbf{S}}_{t-1}^{h}+{\bm{k}}_{t}^{h\top}\,{\bm{v}}^{h}_{t}\in\mathbb{R}^{d^{\prime}_{k}\times d^{\prime}_{v}}, \\
{\bm{o}}^{h}_{t}={\bm{q}}_{t}^{h}{\mathbf{S}}_{t}^{h}\in\mathbb{R}^{1\times d^{\prime}_{v}}, \\
{\bm{o}}^{\prime}_{t}=\mathrm{concat}(\mathrm{LN}({\bm{o}}^{1}_{t}),\dots,\mathrm{LN}({\bm{o}}^{H}_{t}))\in\mathbb{R}^{1\times d_{v}}, \\
{\bm{r}}_{t}=\mathrm{Swish}({\bm{x}}_{t}{\bm{W}}_{r}+{\bm{b}}_{r})\in\mathbb{R}^{1\times d_{v}}, \\
{\bm{y}}_{t}=({\bm{r}}_{t}\odot{\bm{o}}^{\prime}_{t}){\bm{W}}_{O}\in\mathbb{R}^{1\times d}.
\end{aligned}
$$

这里分别设置键维度 $d_{k}$ 与值维度 $d_{v}$; $d^{\prime}_{k}=d_{k}/H$、$d^{\prime}_{v}=d_{v}/H$ 是每个头的键和值维度. 每个头输出后应用 LayerNorm ($\mathrm{LN}$), 输出投影与输出门控则作用于所有头输出的拼接结果 [Sun23b].

随后交替堆叠多头 GLA 层与前馈网络 (FFN), 构造类似 Transformer 的模型. 具体而言, 给定第 $l$ 层的上下文化表示 ${\mathbf{X}}^{(l)}$, 按下式得到 ${\mathbf{X}}^{(l+1)}$:

$$
\begin{aligned}
{\mathbf{Y}}^{(l)}=\mathrm{GLA}(\mathrm{LN}({\mathbf{X}}^{(l)}))+{\mathbf{X}}^{(l)} \\
{\mathbf{X}}^{(l+1)}=\mathrm{SwiGLU}(\mathrm{LN}({\mathbf{Y}}^{(l)}))+{\mathbf{X}}^{(l)},
\end{aligned}
$$

其中 SwiGLU FFN 层 [Tou23] 为

$$
\mathrm{SwiGLU}({\mathbf{Z}})=(\mathrm{Swish}({\mathbf{Z}}{\bm{W}}_{1})\odot{\mathbf{Z}}{\bm{W}}_{2}){\bm{W}}_{3}.
$$

<span id="table-02"></span>

![表 2. GLA Transformer 与 Transformer++、RetNet、Mamba 的结果对比.](./gated-linear-attention/table-02.png)

**表 2.** GLA Transformer 与 Transformer++ [Tou23]、RetNet [Sun23b]、Mamba [Gu23] 的对比. 所有模型都在 SlimPajama 的同一子集上用 Mistral tokenizer 训练; 340M/1.3B 模型分别训练 15B/100B token. 单项任务采用 zero-shot. 主要结果覆盖 [Gu23] 报告的同一组任务; 其他基准及 5-shot 结果见附录[第 11 节](#section-11). 最后一列为所有采用准确率或归一化准确率指标的基准平均值.

**参数分配.** 与普通 softmax 注意力层相比, 这里的 GLA 层多用两个矩阵预测 ${\bm{\alpha}}_{t}$ 和 ${\bm{r}}_{t}$, 即 ${\bm{W}}_{\alpha}$ 与 ${\bm{W}}_{r}$. 为提高参数效率, 采用低秩参数化

$$
{\bm{\alpha}}_{t}=\sigma(({\bm{x}}_{t}{\bm{W}}^{1}_{\alpha}{\bm{W}}^{2}_{\alpha}+{\bm{b}}_{\alpha})))^{\frac{1}{\tau}}\in\mathbb{R}^{1\times d_{k}},
$$

其中 ${\bm{W}}^{1}_{\alpha}\in\mathbb{R}^{d\times 16}$、${\bm{W}}^{2}_{\alpha}\in\mathbb{R}^{16\times d_{k}}$, 温度项 $\tau=16$ 用来促使模型更慢地遗忘. 另设 $d_{k}=\frac{d}{2}$、$d_{v}=d$, 并对 ${\bm{W}}_{Q},{\bm{W}}_{K},{\bm{W}}_{V},{\bm{W}}_{O},{\bm{W}}_{r}$ 采用满秩参数化. 最终一个 GLA 层总共约需 $4d^{2}$ 个参数, 与普通 softmax 注意力相同.

<span id="section-5"></span>

## 5 实证研究

<span id="section-5-1"></span>

### 5.1 实验设置

主要实验针对语言建模, 考察 GLA 能否与两类模型竞争: (i) 采用现代架构方案的强 Transformer 基线; (ii) 近期线性时间模型. 实验使用 SlimPajama 数据集 [Sob23] 和 Mistral tokenizer [Jia23e]. 原始数据集含 627B token, 本文取其中 100B.

**基线.** GLA 与三个基线比较: Transformer++ [Tou23]、RetNet [Sun23b] 和 Mamba [Gu23]. Transformer++ 是采用 Rotary Positional Embedding [Su24]、SwiGLU [Sha20] 与 RMSNorm [Zha19] 的 LLaMA 架构; 为公平比较, RetNet 的原始 FFN 也替换为 SwiGLU. Mamba 使用开源代码. 所有基线都在同一数据集上训练完全相同数量的 token.

**训练细节.** 所有模型均从头训练, 规模为 340M 和 1.3B. 采用 AdamW [Los18], 最大学习率 3e-4. 340M 模型用 15B token 训练, batch size 为 0.5M token; 1.3B 模型用 100B token 训练, batch size 为 2M token. 使用余弦学习率调度, 340M/1.3B 设置分别用 0.5B/1B token warmup. 初始与最终学习率均为 3e-5, weight decay 为 0.01, gradient clipping 为 1.0.

<span id="section-5-2"></span>

### 5.2 主要结果

除 Wikitext (Wiki.) 的困惑度 (ppl) 外, 实验还采用 [Gu23] 使用的一组常识推理与问答下游任务: LAMBADA [Pap16]、PiQA [Bis20]、HellaSwag [Zel19]、WinoGrande [Sak19]、ARC-easy (ARC-e) 和 ARC-challenge (Arc-c) [Cla18]. 附录[第 11 节](#section-11)还给出 Copa [Roe11]、SciQA [Aue23]、OpenbookQA [Mih18b]、BoolQA [Cla19] 的结果. WikiText 与 LAMBADA 报告困惑度; HellaSwag、ARC-challenge 和 OpenbookQA 报告按长度归一化的准确率; 其余任务报告准确率. 所有评测均使用 LM evaluation harness [Gao21c].

<span id="figure-04"></span>

![图 4. 合成 MQAR 任务上的准确率.](./gated-linear-attention/figure-04.png)

**图 4.** 合成 MQAR 任务的准确率 (%).

<span id="figure-05"></span>

![图 5. SlimPajama 与 PG19 测试集上的长度外推.](./gated-linear-attention/figure-05.png)

**图 5.** SlimPajama 与 PG19 测试集上的长度外推. 在 SlimPajama 上以不同训练长度从头预训练 1.3B 模型, 共训练 100B token. <sup>*∗*</sup> 表示采用 truncated BPTT, 将序列划成 12 段, 每段长度 2K.

主要结果见[表 2](#table-02). 与采用数据无关衰减率的 RetNet 相比, 带数据依赖门控的 GLA Transformer 在所有任务上都有提升. GLA Transformer 和 Mamba 的表现都与 Transformer++ 相当.

**高召回需求任务.** 次二次模型的语言建模性能虽能与 Transformer 竞争, [Aro24] 发现它们在高召回需求任务上落后于 softmax 注意力. 下面在侧重召回的真实与合成任务上评测 GLA.

合成 MQAR 任务 [Aro23] 是 induction head 任务 [Dao22g] 更难的多查询版本, 模型需要多次回忆查询 token 后面的 token. 实验沿用 [Aro23] 的设置, 将 GLA 与 RetNet [Sun23b]、Mamba [Gu23]、Hyena [Pol23a] 和 RWKV-4 [Pen23b] 等近期次二次模型比较. RetNet 与 GLA 的头数设为 2, 其余模型采用 [Aro23] 的默认设置. 结果见[图 4](#figure-04). 标准二次注意力在所有设置中都得到满分, 因而没有列出. 具有矩阵隐状态的模型 (Mamba/RetNet/GLA) 优于 Hyena/RWKV, GLA 又优于 RetNet, 验证了数据依赖门控的价值.

按照 [Aro24], 作者还在三个真实的高召回需求任务上测试模型: FDA [Aro23a]、SWDE [Loc19] 和 SQUAD [Raj18]. 这些任务侧重信息抽取或阅读理解. [表 3](#table-03) 显示, 在 FDA 与 SWDE 两项信息抽取任务上, 次二次模型明显落后于 Transformer. 不过 GLA 优于其他次二次模型, 可能因为它的递归状态比 Mamba 更大, 同时又比 RetNet 多了选择机制.

<span id="table-03"></span>

![表 3. 三项高召回需求任务上的模型对比.](./gated-linear-attention/table-03.png)

**表 3.** [Aro24] 测试的三项高召回需求任务上不同模型的对比. 所有任务均为越高越好.

**长序列训练与长度外推.** 线性注意力模型的一项优势是能在线性时间内高效训练长序列. 实验考虑两种设置: (i) 直接在 8K 上下文训练; (ii) 将 24K 上下文划成 2K 段, 用 truncated backpropagation through time (TBPTT) 训练. [+8] 后一种设置不跨段反向传播梯度, 额外开销很小, 与每次都把初始隐状态置零的标准 2K 训练策略相当. 作者在这些设置下用 SlimPajama 的 100B token 预训练 1.3B Mamba、RetNet 与 GLA, 再在 SlimPajama 和 PG19 [Rae20] 测试集上评测.

[图 5](#figure-05) 给出不同位置分组中 token 的困惑度. 对在 2K 上下文训练的模型, GLA 在 PG19 测试集的大多数位置桶上比 Mamba/RetNet 外推得更好; Mamba 很难超过 4K, 而 GLA/RetNet 在 SlimPajama 测试集上可泛化到 18K. Transformer 无法外推到训练长度之外, 这是已知的失效模式. [+9] 长序列预训练始终能改善三种模型的困惑度. GLA 在两种长序列设置下的困惑度差异很小, 表明 TBPTT 可能是更经济的长序列训练方式. Mamba 从 8K 训练中获益明显, 相同训练设置下表现与 GLA 接近.

**消融.** 小规模消融实验用 7B token 训练不同 340M GLA 变体, 研究 (i) 同时具备*细粒度*与*数据依赖*门控的重要性; (ii) 头维度大小的影响. 结果见[表 4](#table-04). 数据依赖标量门虽明显优于 RetNet, 仍需要更细粒度的门控机制. 默认 GLA 使用 4 个头; 增至 8 个头 (更小头维度) 会使困惑度明显变差; 减至 1 个头 (更大头维度) 效果最好, 但提升很小且需要更多 GPU 内存. 因而实验采用 4 个头.

<span id="table-04"></span>

![表 4. 340M 模型的消融实验.](./gated-linear-attention/table-04.png)

**表 4.** 340M 模型训练 7B token 后的消融结果. 用最后 200 个训练 step 的平均困惑度评测各变体.

<span id="section-5-3"></span>

### 5.3 训练效率

<span id="figure-06"></span>

![图 6. 单张 H100 GPU 上 1.3B 模型的训练吞吐量与显存占用.](./gated-linear-attention/figure-06.png)

**图 6.** 单张 H100 GPU 上 1.3B 模型的训练吞吐量与 GPU 内存占用.

[图 6](#figure-06) 展示单张 H100 GPU 上, 不同 1.3B 模型的吞吐量和内存占用如何随序列长度与 batch size 变化. [+10] GLA 采用 FlashLinearAttention 的物化版本, 并重计算隐状态 ([第 3.3 节](#section-3-3)). 所有模型的空间复杂度均为线性, 总 GPU 占用差异很小. 训练吞吐量方面, Mamba 落后于 Transformer++ 与 GLA; 训练长度超过 4096 后, GLA 的优势更明显.

<span id="section-5-4"></span>

### 5.4 局限与未来工作

GLA Transformer 的实验规模已经不小, 但受算力限制, 作者未能开展更大规模的实验. 目前还不清楚 GLA 在更大模型和数据集上如何扩展, 不过其训练效率相对 Mamba 的优势预计会随规模增大. 具体而言, 模型扩展到更大规模 (如 $>7$B) 后, GLA 能更充分地利用 tensor core, 也兼容 tensor parallelism, 因而可能比 Mamba 更高效. [+11] 若要发挥线性注意力的效率, 还值得把 GLA 用到其他模态, 尤其是存在长程依赖的模态; 近期已有工作把先进状态空间模型用于其他数据类型 [Yan23f, Zhu24f, Ma24d, Liu24z, Xin24a, Wan24ad, Wan24ae, Yan24n].

<span id="section-6"></span>

## 6 相关工作

这里简要回顾相关工作, 更完整的讨论见附录[第 8 节](#section-8).

传统 RNN 的隐状态之间存在非线性依赖, 且基于矩阵乘法的顺序隐状态更新成本很高, 因而难以扩展. 线性 RNN、状态空间模型 (SSM) 与 Transformer 消除了非线性依赖, 使训练可以沿时间维并行 [Mar18, Gu22, Smi23]. 作为有竞争力的次二次 Transformer 替代方案, 这类模型近来受到广泛研究 [Pen23b, Gu23, Qin23a, Qin23c, Sun23b, Wan22m].

数据依赖衰减率一直被视为 RNN 的重要组成 [Ger00, Wes18]. 典型遗忘门同时依赖上一隐状态与当前输入. [Mar18] 则提出, 为实现并行训练, 遗忘门值应只依赖当前输入. HGRN [Qin23a] 的中等规模实验已证明这种简单策略有效. RWKV-v6 [Pen24a] 和 Mamba [Gu23] 也采用类似遗忘门的数据依赖衰减率. 在线性 Transformer 中, [Pen21] 使用粗粒度的逐位置遗忘门, [Mao22] 与 [Kat23] 使用更细粒度的遗忘门.

RNN 依靠固定维度的隐状态编码全部历史. 隐状态维度可以近似反映内存容量, 因而显著影响表达能力. 如 [第 2.1 节](#section-2-1) 所述, 线性 Transformer 通过外积参数化扩大 RNN 隐状态维度, 线性 SSM 则采用单输入单输出 (SISO) 策略. SSM 参数不依赖数据时, 可以用快速傅里叶变换 (FFT) 高效训练; 参数依赖数据时, FFT 训练不再可行. 因此 [Gu23] 实现自定义 CUDA kernel, 用并行扫描算法 [Smi23] 训练选择性状态空间模型. 为把所有隐状态放入 SRAM, 扩展率最多只能到 16. 本文的硬件感知训练算法提供另一条高效路径, 可以在更大范围内扩展隐状态维度; 实验已显示这对高召回需求任务有帮助.

<span id="section-7"></span>

## 7 结论

本文提出一种高效算法, 用于训练带数据依赖门控机制的线性注意力 Transformer. 该算法可以权衡 FLOPs 与并行度, 同时仍使用半精度矩阵乘法, 发挥现代 GPU tensor core 的能力. 语言建模实验表明, 门控线性注意力 Transformer 能与强基线取得相当的表现.

**影响声明.** 本文旨在提高一类新型 (门控) 线性注意力模型的训练效率. 这类模型的效率优势或许有助于降低语言模型的使用门槛. 另一方面, 新架构是否会影响语言模型偏见、有害输出等已知问题, 仍有待研究.

## 致谢

本研究得到 MIT-IBM Watson AI Lab 支持. 感谢 Yutao Sun、Zhen Qin、Li Dong、Xinyu Yang、Jiacheng You、Huanqi Cao、Yu Zhang 和 Shida Wang 提供富有启发的讨论. 也感谢 Yu Zhang、Fares Obeid、Daniel Goldstein 和 Liliang Ren 校对论文. 特别感谢 Yu Zhang 对 FlashLinearAttention 库的贡献.

<span id="section-8"></span>

## 8 扩展相关工作

<span id="section-8-1"></span>

### 8.1 线性注意力

**特征映射 $\phi$.** 线性注意力机制 [Kat20] 用具有特征映射 $\phi$ 的核 $k({\bm x},{\bm y})$ 替换 $\exp({\bm q}_t{\bm k}_i^\top)$, 即 $k({\bm x},{\bm y})=\langle\phi({\bm x}),\phi({\bm y})\rangle$, 其中 $\phi\in\mathbb R^{d_{\text{key}}}\rightarrow\mathbb R^{d_{\text{dot}}}$. $\phi$ 通常由两部分组成: $\phi=\phi_0\circ\phi_1$. $\phi_1$ 可以是由随机样本构成的线性映射 [Pen21, Cho20a]、可学习 MLP [Kas21, Zha24aa, Kac23], 或简单的恒等映射 [Mao22]. $\phi_2$ 通常是逐元素激活函数, 使 $\phi$ 成为正特征映射, 例如 $1+\operatorname{elu}$ [Kat20]、$\mathrm{ReLU}$ [Kas21]、$\exp(\cdot)$ [Zha24aa, Cho20a]. 也有工作认为正特征映射未必必要 [Qin23c, Sun23b, Mao22].

本文沿用 [Sun23b] 和 [Mao22], 采用恒等映射 $\phi=\mathbf I$. 近期研究表明, 缩放逐元素指数映射 [Nah23, Zha24aa]、高阶多项式映射 [Aro24, Kac23] 等非恒等特征映射在实验中效果很好. 将其他特征映射整合进 GLA 留作未来工作.

**注意力尖锐度.** 线性注意力存在“注意力稀释”问题 [Qin22a]: 分布过于均匀 (熵过高), 难以集中到相关 token. [Qin22a] 提出加入局部注意力层, 更关注相邻 token; [Lin23, Nah23, Zha23p] 沿用这一方法, 并证明它对性能至关重要. 近期工作发现, 缩放逐元素指数映射 $\phi(\mathbf x)=\mathbf{\exp}(t\cdot\mathbf x)$ ($t\geq2$) 有助于集中注意力 [Nah23, Zha24aa]. [Zha24aa] 还发现, 高阶多项式核会产生低熵、尖锐的注意力分布, 部分解释了 Based Linear Attention [Aro24] 与 PolySketchFormer [Kac23] 的实验成功.

**内存容量.** 线性注意力的内存大小有界 [Pen22], softmax 注意力则有无界内存 [Ore24]. 高效扩大内存并充分利用它, 是缩小两者性能差距的关键. 直接增大 $d_{\operatorname{key}}$ 确实有效 [Sun23b, Mao22, Zha22f], 但总参数量会难以控制. 参数高效方法通常保持 $d_{\text{key}}$ 不变, 转而增大 $d_{\text{dot}}$. 阶数 $p\geq2$ 的高阶多项式核把 $d_{\text{key}}$ 映射到更高维的 $d_{\text{dot}}=O(d_\text{key}^p)$ [Aro23, Kac23]. [Sch21] 提出确定性无参数投影 (DPFP), [Pra23] 则用参数化外积, 以参数高效或无参数的方式扩大 $d_{\text{dot}}$.

为改善内存利用率, [Sch21] 用 delta rule 动态编辑内存. 不过实验表明, 它不及门控机制 [Mao22]; 后者是门控 RNN 擦除无关历史信息的经典方法. [Zha23p] 近期通过约束内存向量正交, 试图进一步提高利用率.

**带衰减或门控的线性注意力.** [Pen21] 用逐位置标量门把近因偏置引入线性注意力, 近期工作又重新采用这一方法 [Dao24, Bec24, Sun24b]; [Mao22, Pra23] 则使用由外积得到的矩阵门, 更细致地控制内存.

标量衰减很容易整合进分块线性注意力, 提高训练效率 [Sun23b, Qin24c]. 矩阵门的高效训练则困难得多. [Mao22] 与 [Kat23] 的算法都要在 HBM 中物化全部时间步的隐状态, I/O 成本很高, 也都无法利用 tensor core. 本文的硬件高效训练算法减少或消除了物化, 同时可以使用 tensor core.

**I/O 感知的分块线性注意力.** 线性注意力的分块形式在文献中已广为人知. [Hua22a] 首先提出这种形式, 指出 [Kat20] 的训练算法实际很慢. [Sun23b] 与 [Qin24c] 将其推广到带指数衰减 (或 ALiBi) 的线性注意力, [Kac23, Lin23] 也推导出类似形式.

不过, 多数分块线性注意力并不感知 I/O. 据作者所知, 只有同期的 LightningAttention2 [Qin24c] 感知 I/O, 且与 FlashLinearAttention 的非物化版本非常相似. 本文还提出物化版本, 用略高的内存占用换取序列级并行与更高训练吞吐量.

**其他次二次模型.** 除本文讨论的线性注意力 Transformer [Kat20, Sch21] 外, 以往研究还通过预定义固定模式 [Chi19, Bel20a, Zah20] 或上下文感知可学习模式 [Roy21, Kit20, Ren23] 稀疏化注意力, 使序列长度维的复杂度降到次二次. 也有工作用卷积高效建模序列, 包括 Dynamic Convolution [Wu19]、Long Convolution [Fu23b, Qin23d, Pol23a, Mas23, Li23y, Rom21] 与状态空间模型 [Gu22, Gup22, Gu21, Has22, Smi23, Ma23b].

<span id="section-8-2"></span>

### 8.2 序列并行

线性 Transformer 的分块并行形式类似两阶段并行前缀和 (或并行扫描) 算法 [Ble90], 后者同样把块内并行计算与块间通信结合起来 [Cha15]. 它也类似加速注意力 Transformer 的序列并行 [Li23g], 这项技术近来在长序列建模中备受关注 [Liu23, Li23q, Bra23]. 序列级并行也是 FlashAttention-2 [Dao23b] 相对 FlashAttention-1 [Dao22] 的主要改进. 区别在于: (i) 线性 Transformer 的块级并行形式因复杂度为线性, 只需一遍扫描; Transformer 的序列并行因固有二次复杂度, 需要 $L/C$ 遍扫描, 即对每个查询块从左到右扫描键/值块; (ii) 矩阵乘法顺序不同. 分布式训练时, 与 softmax 注意力相比, 分块线性注意力还可大幅降低设备间通信成本, 为超长序列训练创造条件.

<span id="algorithm-02"></span>

**算法 2: FlashLinearAttention 反向传播.**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d}$, 块大小 $C\in[L]$, `materialize` $\in\{$`True`,`False`$\}$, 以及 ${\mathbf S}\in\mathbb R^{(L/C)\times d\times d}$ (`materialize` 为 `True` 时可用).
- 在 SRAM 中初始化 ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d\times d}$, 并在片上构造 ${\mathbf M}\in\mathbb R^{C\times C}$.
- **若** `materialize`:
  - **反向循环** $n\gets N,1$:
    - 将 ${\mathbf{dS}}$ 作为 ${\mathbf{dS}}_{[n]}$ 存入 HBM; 将 ${\mathbf Q}_{[n]},{\mathbf{dO}}_{[n]}$ 载入 SRAM; 计算 ${\mathbf{dS}}={\mathbf{dS}}+{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
  - **并行循环** $n\gets1,N$:
    - 将 ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf{dO}}_{[n]},{\mathbf S}_{[n]},{\mathbf{dS}}_{[n]}$ 从 HBM 载入 SRAM.
    - 在片上计算 ${\mathbf{dQ}}={\mathbf{dO}}_{[n]}{\mathbf S}_{[n]}^\top+({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top\odot{\mathbf M}){\mathbf K}_{[n]}$.
    - 在片上计算 ${\mathbf{dK}}={\mathbf V}_{[n]}{\mathbf{dS}}_{[n]}^\top+({\mathbf V}_{[n]}{\mathbf{dO}}_{[n]}^\top\odot{\mathbf M}^\top){\mathbf Q}_{[n]}$.
    - 在片上计算 ${\mathbf{dV}}={\mathbf K}_{[n]}{\mathbf{dS}}_{[n]}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M})^\top{\mathbf{dO}}_{[n]}$, 再将 ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$ 写入 HBM.
- **否则**:
  - 在 SRAM 中初始化 ${\mathbf S}=\mathbf0\in\mathbb R^{d\times d}$.
  - **循环** $n\gets1,N$ (重算隐状态): 载入 ${\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf{dO}}_{[n]}$, 计算 ${\mathbf{dQ}}={\mathbf{dO}}_{[n]}{\mathbf S}^\top+({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top\odot{\mathbf M}){\mathbf K}_{[n]}$, 并更新 ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
  - **反向循环** $n\gets N,1$: 载入块张量, 计算 ${\mathbf{dS}}={\mathbf{dS}}+{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$ 以及上面的 ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$, 再写入 HBM.
- **返回** ${\mathbf{dQ}}=\{{\mathbf{dQ}}_{[1]}\dots{\mathbf{dQ}}_{[N]}\}$、${\mathbf{dK}}=\{{\mathbf{dK}}_{[1]}\dots{\mathbf{dK}}_{[N]}\}$ 与 ${\mathbf{dV}}=\{{\mathbf{dV}}_{[1]}\dots{\mathbf{dV}}_{[N]}\}$.

</div>

<span id="section-8-3"></span>

### 8.3 硬件感知算法

许多算法理论上很快, 却因不符合硬件特性而在实践中很慢 [Hoo20, Sap23]. 例如, 蝶形矩阵乘法用 FFT 可降低理论复杂度, 但大量内存搬运使其实际很慢, 因而出现了更适合 GPU 的矩阵设计 [Dao22b, Fu23c]. 实践中必须用 tiling、重计算等技术降低 HBM I/O, 并尽量发挥 tensor core. FlashLinearAttention 的思路与 FlashAttention [Dao22, Dao23b]、FlashConvFFT [Fu23] 相似: 都为神经网络层实现 I/O 感知版本, 获得实际墙钟加速. 同期工作 [Qin24c] 也提出 I/O 感知的线性注意力, 与 FlashLinearAttention 的非物化版本相似. 本文另提出物化版本, 以略增内存占用为代价利用序列级并行, 提高训练吞吐量.

<span id="section-9"></span>

## 9 分块 (门控) 线性注意力的细节

**FlashLinearAttention 的反向传播.** 线性注意力的反向传播伪代码见[算法 2](#algorithm-02).

**GLA 伪代码.** 先给出不采用二级分块时, 把 FlashLinearAttention 直接改造成 GLA 训练算法的结果. [算法 3](#algorithm-03)与[算法 4](#algorithm-04)分别是物化版本的前向和反向传播; [算法 5](#algorithm-05)与[算法 6](#algorithm-06)对应非物化版本.

二级分块的 Pytorch 风格伪代码见[代码 1](#listing-01).

<span id="listing-01"></span>

**代码 1: 用于训练 GLA 的两级分块算法, Pytorch 风格. 为清楚起见, 省略 batch size 与头数维度.**

```python
def gated_linear_attention_forward(Q, K, V, a, C, c):
  '''
  Q/K/V: query/key/value
  a: log forget gate
  C/c: chunk size, subchunk size
  '''
  # L: sequence length, d: head dimension
  L, d_k = Q.shape
  d_v = V.shape[-1]
  S = torch.zeros(d_k, d_v)
  O = torch.empty_like(V)
  # cumsum of log decay within a chunk
  B = torch.empty_like(a)
  # local compute of cumulative product of decay within a chunk
  for i in range(0, L//C):
    b = torch.zeros(d_k)
    for j in range(0, C):
      b += a[i]
      B[i] = b

  for i in range(0, L // C):
    r = range(i*C,(i+1)*C)
    # (C, d) chunking
    bq, bk, bv, bb = Q[r], K[r], V[r], B[r]
    b = bb[-1,None]
    #inter-chunk w/ matmul
    q, k, g = bq*(bb.exp()), bk*((b-bb).exp()), b.exp()
    o = q @ S
    #hidden state update
    S = g.t() * S + k.t() @ bv
    #intra-chunk (secondary chunking)
    for j in range(0, C // c):
      t = range(j*c, (j+1)*c)
      #(c, head_dim) subchunking
      q, k, v, b = bq[t], bk[t], bv[t], bb[t]
      p = torch.zeros(c,c)
      #intra-subchunk w/o matmul.
      for m in range(c):
        for n in range(m+1):
          p[m,n]=torch.sum(q[m]*k[n]*((b[m]-b[n]).exp()))
      o[t] += p @ v
      # inter-subchunk w/ matmul
      z = b[0, None]
      q = q * (b-z).exp()
      for u in range(0, j):
        y = range(u*c, (u+1)*c)
        p = q @ (bk[y]*(z-bb[y]).exp()).t()
        o[t] += p@bv[y]
    O[r] = o
  return O
```

<span id="algorithm-03"></span>

**算法 3: 门控线性注意力前向传播 (物化版本).**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V}\in\mathbb R^{L\times d_v}$, ${\mathbf G}=[{\bm\alpha}_1\dots{\bm\alpha}_L]$, 块大小 $C$.
- 将 ${\mathbf Q},{\mathbf K},{\mathbf G}$ 划分为 $N=L/C$ 个 $C\times d_k$ 块, 将 ${\mathbf V}$ 划分为 $N$ 个 $C\times d_v$ 块, 并在 SRAM 中初始化 ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$.
- **循环** $n\gets1,N$: 将 ${\mathbf S}$ 作为 ${\mathbf S}_{[n]}$ 写入 HBM; 载入 ${\mathbf K}_{[n]},{\mathbf G}_{[n]},{\mathbf V}_{[n]}$; 计算 ${\bm\gamma}_{[n]}$、${\mathbf\Gamma}_{[n]}$、$\widetilde{\mathbf K}_{[n]}={\mathbf K}_{[n]}\odot{\mathbf\Gamma}_{[n]}$ 与 ${\mathbf S}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf S}+\widetilde{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
- **并行循环** $n\gets1,N$: 载入块张量与 ${\mathbf S}_{[n]}$; 构造 ${\mathbf M}$; 计算 ${\mathbf\Lambda}_{[n]}$、${\mathbf\Gamma}_{[n]}$、$\widetilde{\mathbf Q}_{[n]}={\mathbf Q}_{[n]}\odot{\mathbf\Lambda}_{[n]}$、$\widetilde{\mathbf K}_{[n]}={\mathbf K}_{[n]}\odot{\mathbf\Gamma}_{[n]}$ 与 $\overline{\mathbf K}_{[n]}={\mathbf K}_{[n]}/{\mathbf\Lambda}_{[n]}$.
  - 计算 ${\mathbf O}_{[n]}^{\mathrm{inter}}=\widetilde{\mathbf Q}_{[n]}{\mathbf S}_{[n]}$、${\mathbf P}=(\widetilde{\mathbf Q}_{[n]}\overline{\mathbf K}_{[n]}^\top)\odot{\mathbf M}$、${\mathbf O}^{\mathrm{intra}}={\mathbf P}{\mathbf V}_{[n]}$ 与 ${\mathbf O}_{[n]}={\mathbf O}^{\mathrm{inter}}+{\mathbf O}^{\mathrm{intra}}$; 将 ${\mathbf O}_{[n]}$ 存入 HBM.
- **返回** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$ 与 ${\mathbf S}=\{{\mathbf S}_{[1]}\dots{\mathbf S}_{[N]}\}$.

</div>

<span id="algorithm-04"></span>

**算法 4: 门控线性注意力反向传播 (物化版本).**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d_v}$, 块大小 $C$.
- 在 SRAM 中初始化 ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d_k\times d_v}$.
- **循环** $n\gets N,1$: 将 ${\mathbf{dS}}$ 作为 ${\mathbf{dS}}_{[n]}$ 保存; 载入 ${\mathbf G}_{[n]},{\mathbf Q}_{[n]},{\mathbf{dO}}_{[n]}$; 计算 ${\bm\gamma}_{[n]}$、${\mathbf\Gamma}_{[n]}$、$\widetilde{\mathbf Q}_{[n]}={\mathbf Q}_{[n]}\odot{\mathbf\Gamma}_{[n]}$ 与 ${\mathbf{dS}}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf{dS}}+\widetilde{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
- **并行循环** $n\gets1,N$: 载入块张量、状态与状态梯度; 构造 ${\mathbf M}$ 以及门控后的 query/key; 计算 ${\mathbf P}$ 与 ${\mathbf{dP}}=({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top)\odot{\mathbf M}$.
  - 计算 ${\mathbf{d\bar K}}_{[n]}=\widetilde{\mathbf Q}_{[n]}{\mathbf{dP}}^\top$、${\mathbf{d\widetilde K}}_{[n]}={\mathbf V}_{[n]}{\mathbf{dS}}_{[n]}^\top$ 与 ${\mathbf{dK}}_{[n]}={\mathbf{d\widetilde K}}_{[n]}\odot{\mathbf\Gamma}_{[n]}+{\mathbf{d\bar K}}_{[n]}/{\mathbf\Lambda}_{[n]}$.
  - 计算 ${\mathbf{d\widetilde Q}}_{[n]}={\mathbf{dP}}\overline{\mathbf K}_{[n]}+{\mathbf{dO}}_{[n]}{\mathbf S}_{[n]}^\top$、${\mathbf{dQ}}_{[n]}={\mathbf{d\widetilde Q}}_{[n]}\odot{\mathbf\Lambda}_{[n]}$ 与 ${\mathbf{dV}}_{[n]}={\mathbf P}^\top{\mathbf{dO}}_{[n]}+\widetilde{\mathbf K}_{[n]}{\mathbf{dS}}_{[n]}$; 将梯度存入 HBM.
- 将各块梯度拼接为 ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$, 再计算 ${\mathbf{dA}}={\mathbf Q}\odot{\mathbf{dQ}}-{\mathbf K}\odot{\mathbf{dK}}$ 与 ${\mathbf{dG}}=\mathrm{revcum}({\mathbf{dA}})$.
- **返回** ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}},{\mathbf{dG}}$.

</div>

<span id="algorithm-05"></span>

**算法 5: 门控线性注意力前向传播 (非物化版本).**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V}\in\mathbb R^{L\times d_v}$, ${\mathbf G}=[{\bm\alpha}_1\dots{\bm\alpha}_L]$, 块大小 $C$.
- 将输入划分为 $N$ 个块, 并在 SRAM 中初始化 ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$.
- **循环** $n\gets1,N$: 载入块张量; 计算 ${\bm\gamma}_{[n]}$、${\mathbf\Lambda}_{[n]}$、${\mathbf\Gamma}_{[n]}$、$\widetilde{\mathbf Q}_{[n]}$、$\widetilde{\mathbf K}_{[n]}$ 与 $\overline{\mathbf K}_{[n]}$; 构造 ${\mathbf M}$.
  - 计算 ${\mathbf O}_{[n]}^{\mathrm{inter}}=\widetilde{\mathbf Q}_{[n]}{\mathbf S}$、${\mathbf P}=(\widetilde{\mathbf Q}_{[n]}\overline{\mathbf K}_{[n]}^\top)\odot{\mathbf M}$、${\mathbf O}^{\mathrm{intra}}={\mathbf P}{\mathbf V}_{[n]}$ 与 ${\mathbf O}_{[n]}={\mathbf O}^{\mathrm{inter}}+{\mathbf O}^{\mathrm{intra}}$; 保存 ${\mathbf O}_{[n]}$.
  - 更新 ${\mathbf S}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf S}+\widetilde{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
- **返回** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$.

</div>

<span id="algorithm-06"></span>

**算法 6: 门控线性注意力反向传播 (非物化版本).**

<div class="paper-algorithm">

- **输入:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d_v}$, 块大小 $C$.
- 在 SRAM 中初始化 ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$.
- **循环** $n\gets1,N$: 载入 gate、query 与输出梯度块; 计算 gate 摘要、${\mathbf{dP}}={\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top$、${\mathbf{d\widetilde Q}}_{[n]}={\mathbf{dP}}\widetilde{\mathbf K}_{[n]}+{\mathbf{dO}}_{[n]}{\mathbf S}^\top$ 与 ${\mathbf{dQ}}={\mathbf{d\widetilde Q}}_{[n]}\odot{\mathbf\Gamma}_{[n]}$; 保存 ${\mathbf{dQ}}_{[n]}$ 并更新 ${\mathbf S}$.
- 在 SRAM 中初始化 ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d_k\times d_v}$.
- **循环** $n\gets N,1$: 载入块张量与梯度; 构造 ${\mathbf M}$ 以及门控后的 query/key; 按[算法 4](#algorithm-04)计算 ${\mathbf P}$、${\mathbf{dP}}$、${\mathbf{dK}}_{[n]}$ 与 ${\mathbf{dV}}_{[n]}$, 存入 HBM; 更新 ${\mathbf{dS}}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf{dS}}+\widetilde{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
- 拼接各块梯度, 计算 ${\mathbf{dA}}={\mathbf Q}\odot{\mathbf{dQ}}-{\mathbf K}\odot{\mathbf{dK}}$ 与 ${\mathbf{dG}}=\mathrm{revcum}({\mathbf{dA}})$.
- **返回** ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}},{\mathbf{dG}}$.

</div>

**${\mathbf{d}\log\bm\alpha}_t$ 的推导.** 下面推导这种梯度形式.

$$
{\mathbf{d}\log\bm b}_t={\bm k}_t\odot{\mathbf{d}\bm k}_t-{\bm q}_t\odot{\mathbf{d}\bm q}_t,\qquad
{\mathbf{d}\log\bm\alpha}_t=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm b}_i.
$$

展开递归可得

$$
{\bm o}_t={\bm q}_t{\mathbf S}_t=\sum_{i=1}^{t}({\bm q}_t\odot{\bm b}_t)\left(\frac{{\bm k}_i}{{\bm b}_i}\right)^\top{\bm v}_i
=\sum_{i=1}^{t}({\bm q}_t\odot\exp(\log{\bm b}_t))\left({\bm k}_i\odot\exp(-\log{\bm b}_i)\right)^\top{\bm v}_i,
$$

第二步使用了恒等式 $\exp(\log x)=x$.

先推导查询与键向量的梯度:

$$
{\mathbf{d}\bm q}_t=\sum_{i=1}^{t}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle{\bm b}_t\odot{\bm k}_i/{\bm b}_i,
\qquad
{\mathbf{d}\bm k}_i=\sum_{t=i}^{L}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle{\bm q}_t\odot{\bm b}_t/{\bm b}_i.
$$

再推导累积门 logit 的梯度:

$$
{\mathbf{d}\log\bm b}_t={\bm q}_t\odot\underbrace{\sum_{i=1}^{t}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle\odot{\bm b}_t\odot{\bm k}_i/{\bm b}_i}_{{\mathbf{d}\bm q}_t}
-{\bm k}_t\odot\underbrace{\sum_{i=t}^{L}\langle{\mathbf{d}\bm o}_i,{\bm v}_t\rangle{\bm q}_i\odot{\bm b}_i/{\bm b}_t}_{{\mathbf{d}\bm k}_t}.
$$

这里改变了 ${\mathbf{d}\bm k}$ 项的索引记号. 于是 ${\mathbf{d}\log\bm b}_t={\bm q}_t\odot{\mathbf{d}\bm q}_t-{\bm k}_t\odot{\mathbf{d}\bm k}_t$ 显而易见. 又因为 $\log{\bm b}_t=\sum_{i=1}^{t}\log{\bm\alpha}_i$, 所以 ${\mathbf{d}\log\bm\alpha}_t=\sum_{t=i}^{L}{\mathbf{d}\log\bm b}_i$.

<span id="section-10"></span>

## 10 一般门控线性注意力

正文采用简化参数化, 在下面的一般门控线性注意力中把 ${\bm\beta}$ 固定为 $\mathbf1$.

$$
{\mathbf S}_t=({\bm\alpha}_t^\top{\bm\beta}_t)\odot{\mathbf S}_{t-1}+{\bm k}_t^\top{\bm v}_t.
$$

实验中, 把 ${\bm\beta}$ 设为可学习并未带来性能提升. 不过, 这里证明一般形式仍有并行形式与分块形式, 可能对未来开发线性注意力模型有用.

<span id="section-10-1"></span>

### 10.1 并行形式

展开递归可得

$$
{\bm o}_t={\bm q}_t{\mathbf S}_t={\bm q}_t\sum_{i=1}^{t}\left(\left(\prod_{j=i+1}^{t}{\mathbf G}_j\right)\odot({\bm k}_i^\top{\bm v}_i)\right).
$$

利用 Kronecker 积或外积的混合积性质, 可得

$$
\left(\prod_{j=i+1}^{t}{\mathbf G}_j\right)\odot({\bm k}_i^\top{\bm v}_i)
=\left(\frac{{\bm b}_t}{{\bm b}_i}\odot{\bm k}_i\right)^\top\left(\frac{{\bm d}_t}{{\bm d}_i}\odot{\bm v}_i\right),
$$

其中 ${\bm b}_t=\prod_{j=1}^{t}{\bm\alpha}_j$, ${\bm d}_t=\prod_{j=1}^{t}{\bm\beta}_j$. 代入展开后的递归, 得到

$$
\begin{aligned}
{\bm o}_t
&=\sum_{i=1}^{t}\left({\bm q}_t\left(\frac{{\bm b}_t}{{\bm b}_i}\odot{\bm k}_i\right)^\top\right)\left(\frac{{\bm d}_t}{{\bm d}_i}\odot{\bm v}_i\right)\\
&=\sum_{i=1}^{t}\left(\left({\bm q}_t\odot{\bm b}_t\right)\left(\frac{{\bm k}_i}{{\bm b}_i}\right)^\top\left(\frac{{\bm v}_i}{{\bm d}_i}\right)\right)\odot{\bm d}_t\in\mathbb R^{1\times d_v}.
\end{aligned}
$$

第一处等号来自线性性与矩阵乘法结合律, 第二处依据 $\langle{\bm a},{\bm b}\odot{\bm c}\rangle=\langle{\bm a}\odot{\bm b},{\bm c}\rangle$. 最终形式等价于下面的并行形式, 与线性或 softmax 注意力的并行形式相似.

$$
\widetilde{\mathbf Q}={\mathbf Q}\odot{\mathbf B},\quad
\widetilde{\mathbf K}={\mathbf K}/{\mathbf B},\quad
\widetilde{\mathbf V}={\mathbf V}/{\mathbf D},\qquad
\widetilde{\mathbf O}=(\widetilde{\mathbf Q}\widetilde{\mathbf K}^\top\odot{\mathbf M})\widetilde{\mathbf V},\quad
{\mathbf O}=\widetilde{\mathbf O}\odot{\mathbf D},
$$

其中 ${\mathbf Q},{\mathbf K},{\mathbf B}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf D}\in\mathbb R^{L\times d_v}$, ${\mathbf M}\in\mathbb R^{L\times L}$ 表示因果掩码.

<span id="section-10-2"></span>

### 10.2 分块并行形式

下面给出一般线性注意力用于高效训练的分块并行形式. 将 ${\mathbf X}$ 划分成 $L/C$ 个长度为 $C$ 的块. 令 ${\mathbf S}_{[i]}\in\mathbb R^{d_k\times d_v}$ 为处理 $i$ 个块后的块级隐状态, 即 ${\mathbf S}_{[i]}:={\mathbf S}_{iC}$. 再令 ${\mathbf K}_{[i+1]}:={\mathbf K}_{iC+1:(i+1)C}\in\mathbb R^{C\times d_k}$、${\mathbf V}_{[i+1]}:={\mathbf V}_{iC+1:(i+1)C}\in\mathbb R^{C\times d_v}$. 块间递归为

$$
{\mathbf S}_{[i+1]}=\left(\left(\frac{{\mathbf B}_{(i+1)C}}{{\mathbf B}_{iC}}\right)^\top\left(\frac{{\mathbf D}_{(i+1)C}}{{\mathbf D}_{iC}}\right)\right)\odot{\mathbf S}_{[i]}
+({\mathbf B}'_{[i+1]}\odot{\mathbf K}_{[i+1]})^\top({\mathbf D}'_{[i+1]}\odot{\mathbf V}_{[i+1]}),
$$

其中, 对 $j\in[1,C]$、$i\in[0,L/C-1]$, $({\mathbf B}'_{[i+1]})_j={\mathbf B}_{(i+1)C}/{\mathbf B}_{iC+j}\in\mathbb R^{1\times d_k}$, $({\mathbf D}'_{[i+1]})_j={\mathbf D}_{(i+1)C}/{\mathbf D}_{iC+j}\in\mathbb R^{1\times d_v}$. 块内并行计算为

$$
\begin{aligned}
\widetilde{\mathbf O}_{[i+1]}&=\underbrace{(({\mathbf Q}_{[i+1]}\odot{\mathbf B}^{\dagger}_{[i+1]}){\mathbf S}_{[i]})\odot{\mathbf D}^{\dagger}_{[i+1]}}_{\mathrm{inter-chunk}}\\
&\quad+\underbrace{(\widetilde{\mathbf Q}_{[i+1]}\widetilde{\mathbf K}_{[i+1]}^\top\odot{\mathbf M})\widetilde{\mathbf V}_{[i+1]}}_{\mathrm{intra-chunk}},\\
{\mathbf O}_{[i+1]}&=\widetilde{\mathbf O}_{[i+1]}/{\mathbf D}^{\dagger}_{[i+1]}.
\end{aligned}
$$

其中 $({\mathbf B}_{[i+1]}^{\dagger})_j={\mathbf B}_{iC+j}/{\mathbf B}_{iC}$, $({\mathbf D}_{[i+1]}^{\dagger})_j={\mathbf D}_{iC+j}/{\mathbf D}_{iC}$. 于是 $\widetilde{\mathbf Q}_{[i+1]}={\mathbf Q}_{[i+1]}\odot{\mathbf B}_{[i+1]}^{\dagger}$, $\widetilde{\mathbf K}_{[i+1]}={\mathbf K}_{[i+1]}/{\mathbf B}_{[i+1]}^{\dagger}$, $\widetilde{\mathbf V}_{[i+1]}={\mathbf V}_{[i+1]}\odot{\mathbf D}_{[i+1]}^{\dagger}$. 初值设为 ${\mathbf S}_0=\mathbf0$、${\mathbf B}_0=\mathbf1$、${\mathbf D}_0=\mathbf1$. 直观而言, ${\mathbf B}'_{[i]}$ 编码从块首开始的累积衰减, 用于传播上一块的隐状态 ${\mathbf S}_{[i]}$; ${\mathbf B}^{\dagger}_{[i]}$ 编码到块尾为止的衰减, 用于累积将加入下一隐状态 ${\mathbf S}_{[i+1]}$ 的信息.

这里的分块形式推广了多种现有线性注意力形式. 令 ${\mathbf A}_{ij}=1$、${\mathbf B}_{ij}=1$, 就退化为正文普通线性注意力的分块形式; 令 ${\mathbf A}_{ij}=1$、${\mathbf B}_{ij}=\gamma^{i+1}$, 就得到 RetNet 的分块形式 [Sun23b]. 因此, 该表达可以视为允许细粒度数据依赖衰减的一般线性注意力分块并行形式.

**${\mathbf{d}\bm\alpha}$ 与 ${\mathbf{d}\bm\beta}$ 的内存高效计算.** 一般形式中, ${\bm\alpha}$ 与 ${\bm\beta}$ 的梯度具有如下闭式表达, 因而无需在 HBM 中实例化 ${\mathbf S}$ 即可计算 ${\mathbf{d}\bm\alpha}$ 和 ${\mathbf{d}\bm\beta}$.

$$
\begin{aligned}
{\mathbf{d}\log\bm b}_t&={\bm k}_t\odot{\mathbf{d}\bm k}_t-{\bm q}_t\odot{\mathbf{d}\bm q}_t,&
{\mathbf{d}\log\bm\alpha}_t&=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm b}_i,\\
{\mathbf{d}\log\bm d}_t&={\bm o}_t\odot{\mathbf{d}\bm o}_t-{\bm v}_t\odot{\mathbf{d}\bm v}_t,&
{\mathbf{d}\log\bm\beta}_t&=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm d}_i.
\end{aligned}
$$

其中 $\log{\bm b}_t=\sum_{i=1}^{t}\log{\bm\alpha}_i$, $\log{\bm d}_t=\sum_{i=1}^{t}{\bm\beta}_i$; 等价地, ${\bm b}_t=\prod_{i=1}^{t}{\bm\alpha}_i$, ${\bm d}_t=\prod_{i=1}^{t}{\bm\beta}_i$. 对上述累加形式采用同一技巧计算 ${\mathbf{d}\log\bm b}_t$ 与 ${\mathbf{d}\log\bm d}_t$. $\log{\bm b}_t$ 的梯度来自 ${\bm q}_t$ 和 ${\bm k}_i$ 两部分, $\log{\bm d}_t$ 同样来自 ${\bm o}_t$ 和 ${\bm v}_i$. 所用恒等关系是 $\partial f({\bm a}\odot{\bm b})/\partial\log{\bm b}={\bm a}\odot\partial f({\bm a}\odot{\bm b})/\partial{\bm a}$, 以及 $\partial f({\bm a}/{\bm b})/\partial\log{\bm b}=-\partial f({\bm a}/{\bm b})/\partial{\bm a}\odot{\bm a}$.

<span id="section-11"></span>

## 11 更多实验结果

<span id="table-05"></span>

![表 5. 扩展的零样本与五样本结果.](./gated-linear-attention/table-05.png)

**表 5.** 扩展的 zero-shot 与 five-shot 性能结果. 所有模型均在 SlimPajama 的同一子集上用 Mistral tokenizer 训练. 340M/1.3B 模型分别训练 15B/100B token. 最后一列为所有准确率的平均值.

全部 11 项任务的完整结果, 包括 1.3B 模型的 5-shot 结果, 见[表 5](#table-05).

[+1]: 这类具有随时间变化的矩阵值隐状态的模型也称为“快速权重” [Hin87, Sch92, Ba16a], 近期研究讨论了它与 Transformer 的联系 [Sch21, Iri21, Mao22].

[+2]: 若没有 ${\mathbf{M}}$, 可将 $({\mathbf{Q}}{\mathbf{K}}^\top){\mathbf{V}}$ 改写为 ${\mathbf{Q}}({\mathbf{K}}^\top{\mathbf{V}})$, 使复杂度从二次 $O(L^{2}d)$ 降至线性 $O(Ld^{2})$.

[+3]: 这可以看作采用 ALiBi 位置编码 [Pre21] 的线性注意力. 实际上, 这些工作还会加入旋转位置嵌入 [Su24].

[+4]: 不过, [Mao22] 只处理递归形式, 并在 HBM 中物化所有时间步的隐状态. 附录[第 10 节](#section-10)给出一种新算法, 将模型改写为基于矩阵乘法的并行形式, 从而使用 FlashLinearAttention 的扩展进行高效训练.

[+5]: 初步实验中, ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}{\bm{\beta}}_{t}$ 相比 ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}\mathbf{1}$ 只有很小的提升.

[+6]: 这种形式类似可外推的位置编码 [Sun22], 指数内部的项可视作*数据依赖*的相对位置因子.

[+7]: 为减少符号负担, 这里沿用第一级分块的记号表达核心思路; 实际实现采用第二级分块.

[+8]: 将 24K 输入序列分成 12 段, 并把前一段的最终状态作为当前段的初始状态.

[+9]: 有些位置编码方案能改善长度外推, 但仍难以显著泛化到训练时未见的更长上下文 [Pre21, Sun22, Li23x].

[+10]: Mamba 使用官方实现; Transformer++ 与 GLA 使用融合版 SwiGLU; Transformer++ 使用 FlashAttention-2.

[+11]: 尤其是 Mamba 并非多头模型, 因而不太适合张量并行.

[+author-note]: 同等贡献.
