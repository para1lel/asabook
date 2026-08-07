---
title: 'Gated Delta Networks'
createTime: 2026/08/05 00:34:32
permalink: /papers/gated-delta-networks/
---

> [Songlin Yang](https://sustcsonglin.github.io/), [Jan Kautz](https://www.jankautz.com/), and [Ali Hatamizadeh](https://ahatamiz.github.io/). 首次提交至 arXiv: December 9, 2024; 当前版本为 v3. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). [原始 PDF](/paper/gated-delta-networks.pdf). [TeX 源码](https://export.arxiv.org/e-print/2412.06464). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

线性 Transformer 作为标准 Transformer 的高效替代方案, 已引起关注, 但在检索和长上下文任务中的表现有限. 为了解决这些限制, 最近的研究探索了两种不同的机制: 用于自适应记忆控制的门控机制和用于精确记忆修改的 delta 更新规则. 我们观察到这两种机制是互补的——门控机制能够快速清除记忆, 而 delta 规则则有助于定向更新. 在此基础上, 我们引入了门控 delta 规则, 并开发了一种针对现代硬件优化的并行训练算法. 我们提出的架构 Gated DeltaNet 在多个基准测试中, 如语言建模, 常识推理, 上下文检索, 长度外推和长上下文理解, 始终超越现有模型如 Mamba2 和 DeltaNet. 我们进一步通过开发将 Gated DeltaNet 层与滑动窗口注意力或 Mamba2 层结合的混合架构来提升性能, 实现了训练效率提升和任务表现优化.
代码: [https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet]

## 1 引言

Transformer 架构显著提升了大型语言模型 (LLM) 的能力, 由于其高效的注意力机制, 在广泛的任务中展现出卓越的性能. 这一机制在精确的序列建模方面表现出色, 并在训练过程中利用现代 GPU 的并行处理能力. 然而, 自注意力组件的计算复杂度随序列长度呈二次增长, 从而导致巨大的计算需求, 对训练和推理都构成挑战.

为了缓解这些问题, 研究人员探索了替代方案, 如线性 Transformer [Kathaa20], 它们用基于核的点积线性注意力替代了传统的基于 softmax 的注意力, 通过将其重新组织为具有矩阵值状态的线性 RNN, 从而大幅减少推理期间的内存需求. 尽管早期版本的线性 Transformer 在语言建模任务中的表现不如标准 Transformer, 但最近的改进——例如引入类似于 LSTM 的数据依赖门控机制, 以 GLA [Yang24] 和 Mamba2 [Daoa24] 等模型为例——显示出了有希望的提升. 然而, 在处理长序列信息方面仍然存在挑战, 特别是在上下文检索任务中, 传统 Transformer 仍然保持其优势 [Arora23, Arora24, Jelass24, Wen24, Aky24].

这种现象并不令人惊讶: 线性 Transformer 可以被解释为实现基于外积的键值关联记忆, 这让人想起张量积表示 [Smolen90]. 然而, 它们能够存储的正交键值对的数量 *受限于* 模型的维度. 当序列长度超过该维度时, “记忆冲突”不可避免, 从而阻碍精确检索 [Schlag21].

Mamba2 通过引入一个简单的门控更新规则来解决这一限制, ${\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}$, 该规则在每个时间步以动态比率均匀衰减所有键值关联 $\alpha_{t}$. 然而, 该方法并未考虑不同键值关联的重要性差异, 可能导致记忆利用效率低下. 如果模型需要忘记某个特定的键值关联, 所有键值关联都会被同等忘记, 使得这一过程不够有针对性和高效.

相比之下, 使用 delta 规则的线性 Transformer [Widrow60], 被称为 DeltaNet [Schlag21, Yanga24], 通过按顺序 (软性地) 用输入的键值对替换旧的键值对来选择性地更新记忆. 这种方法在上下文检索的合成基准测试中表现出令人印象深刻的性能. 然而, 由于此过程一次只修改一个键值对, 模型缺乏快速清除过时或无关信息的能力, 尤其是在上下文切换过程中需要擦除以前的数据. 因此, 研究发现 DeltaNet 在现实任务中表现中等 [Yanga24], 可能是由于缺乏强大的记忆清除机制.

认识到门控更新规则和 delta 规则在记忆管理中的互补优势, 我们提出了*门控 delta 规则*, 一种结合两种方法的简单直观机制. 这一统一规则使记忆控制更加灵活: 可以通过设置 $\alpha_{t}\rightarrow 0$ 迅速清除记忆, 同时通过设置 $\alpha_{t}\rightarrow 1$ 选择性地更新特定内容而不影响其他信息 (有效地切换到纯 delta 规则).

剩下的挑战在于如何以硬件高效的方式实现门控 delta 规则. 在 [Yanga24] 的高效算法基础上, 该算法利用 WY 表示 [Loan85] 并行计算 delta 规则, 我们仔细扩展了他们的方法以加入门控项. 我们的扩展保留了分块并行的优势 [Huaa22, Suna23, Yang24], 从而实现硬件高效的训练.

我们得到的架构 Gated DeltaNet 在包括语言建模, 常识推理, 上下文检索, 长度外推和长上下文理解的一系列综合基准测试中, 一直优于 Mamba2 和 DeltaNet. 基于这些结果, 我们还开发了混合架构, 策略性地将 Gated DeltaNet 层与滑动窗口注意力或 Mamba2 层结合, 进一步提高训练效率和模型性能.

## 2 预备知识

### 2.1 带分块并行形式的线性注意力

已知线性 Transformer [Kathab20] 可以在排除归一化和查询/键激活的情况下, 表示为以下线性递推式:

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad\qquad{\bm{o}}_{t}={\mathbf{S}}_{t}{\bm{q}}_{t}\in\mathbb{R}^{d_{v}}
$$

其中 $d_{k}$ 和 $d_{v}$ 分别表示查询/键和数值的 (头) 维度. 通过展开递推关系, 我们可以将其分别以向量形式 (左) 和矩阵形式 (右) 表示如下:

$$
{\bm{o}}_{t}=\sum_{i=1}^{t}({\bm{v}}_{i}{\bm{k}}_{i}^{\top}){\bm{q}}_{t}=\sum_{i=1}^{t}{\bm{v}}_{i}({\bm{k}}_{i}^{\top}{\bm{q}}_{t})\in\mathbb{R}^{d_{v}},\qquad{\mathbf{O}}=({\mathbf{Q}}{\mathbf{K}}^{\top}\odot{\mathbf{M}}){\mathbf{V}}\in\mathbb{R}^{L\times d_{v}}
$$

其中 $L$ 是序列长度, 而 ${\mathbf{M}}\in\mathbb{R}^{L\times L}$ 是因果遮罩, 由以下定义 ${\mathbf{M}}_{\mathrm{ij}}=0$: 当 $i<j$ 时为 1, 其他情况下 $1$ 为 0.

这个公式表明, 线性注意力消除了传统注意力机制中使用的 softmax 操作, 而是利用矩阵乘法的线性和结合性, 从而实现线性复杂度. 但是, 无论是递归形式还是并行形式在高效训练 [Yang24] 时都不是理想的, 这促使采用块状并行形式 [Huaa22, Suna23, Yang24] 进行硬件高效的线性时间训练, 如下所述.

#### 分块并行形式.

总结而言, 块状并行形式将输入和输出拆分为若干大小为 $C$ 的块, 并根据前一区块的最终状态以及当前区块的查询/键/值块来计算每个区块的输出. 按照 [Suna23, Yang24, Yanga24] 的符号, 让我们以查询块 ${\bm{q}}$ 为例. 我们将 ${\mathbf{Q}}_{[t]}:={\bm{q}}_{\mathrm{tC}+1:(t+1)C+1}$ 表示为区块 $t$ 的查询块, ${\bm{q}}_{[t]}^{r}:={\bm{q}}_{\mathrm{tC}+r}$ 表示区块 $t$ 中的第 $r$ 个查询. 区块 $t$ 的初始状态定义为 ${\mathbf{S}}_{[t]}:={\mathbf{S}}_{[t]}^{0}={\mathbf{S}}_{[t-1]}^{C}$. 通过部分展开递归, 我们有

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad{\bm{o}}_{[t]}^{r}={\mathbf{S}}_{[t]}^{r}{\bm{q}}_{[t]}^{r}={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^{r}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}\left({\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^{r}\right)\in\mathbb{R}^{d_{v}}
$$

等价地, 用矩阵形式表示为:

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad{\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\mathbf{M}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_{v}}
$$

其中 ${\mathbf{M}}\in\mathbb{R}^{C\times C}$ 是因果掩码. 上述方程式中包含大量矩阵乘法 (matmuls), 通过将 $C$ 设置为 16 的倍数, 可以利用张量核心——用于高效半精度矩阵乘法操作的专用 GPU 单元——实现硬件高效训练. 通常, $C$ 被设置为一个小常数 (例如, 在 FLA [Zhang24] 中实现的 64), 确保总体计算复杂度保持与序列长度的线性关系, 从而能够高效建模非常长的序列.

### 2.2 Mamba2: 具有标量值数据相关衰减的线性注意力

Mamba2 [Daoa24] 可以用以下线性递归表示 (取决于具体参数化):

$$
{\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top},\qquad{\bm{o}}_{t}={\mathbf{S}}_{t}{\bm{q}}_{t}
$$

其中 $\alpha_{t}\in(0,1)$ 是一个数据相关的标量值衰减项. 接下来, 我们将以蓝色突出显示衰减项, 以便与普通线性注意力进行更清晰的比较. 定义累积衰减乘积 $\gamma_{j}=\prod_{i=1}^{j}\alpha_{i}$, 通过展开递推式, 我们可以将结果表示为向量形式 (左) 和矩阵并行形式 (右):

$$
{\bm{o}}_{t}=\sum_{i=1}^{t}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}{\bm{v}}_{i}{\bm{k}}_{i}^{\top}\right){\bm{q}}_{t}=\sum_{i=1}^{t}{\bm{v}}_{i}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}{\bm{k}}_{i}^{\top}{\bm{q}}_{t}\right),\qquad{\mathbf{O}}=\left(\left({\mathbf{Q}}{\mathbf{K}}^{\top}\right)\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma}\right){\mathbf{V}}
$$

这里, $\Gamma\in\mathbb{R}^{L\times L}$ 是一个衰减感知的因果掩码, 其中 $\Gamma_{\mathrm{ij}}=\frac{\gamma_{i}}{\gamma_{j}}$ 如果 $i\geq j$, 否则为 $\Gamma_{\mathrm{ij}}=0$.

这种平行且递归的公式在 [Daoa24] 中被称为状态空间对偶 (SSD). 这种递归结构也被用于 Gated RFA [Peng21], xLSTM [Beck24] 和 Gated RetNet [Sunb24].

#### 分块并行形式.

略微滥用符号, 我们定义块内衰减的局部累积乘积为 $\gamma_{[t]}^{j}=\prod_{i=\mathrm{tC}+1}^{\mathrm{tC}+j}\alpha_{i}$. 另外, 我们定义 $(\Gamma_{[t]})_{\mathrm{ij}}=\frac{\gamma_{[t]}^{j}}{\gamma_{[t]}^{i}}$ 对于 $i\geq j$, 而否则为 0. 通过部分展开递归, 我们得到如下方程:

$$
{\mathbf{S}}_{[t]}^{r}={\color[\mathrm{rgb}]{0,0,1}\gamma^{r}_{[t]}}{\mathbf{S}}_{[t]}+\sum_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}}}{\bm{v}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top},\qquad{\bm{o}}_{[t]}^{r}={\color[\mathrm{rgb}]{0,0,1}\gamma^{r}_{[t]}}{\mathbf{S}}_{[t]}^{r}{\bm{q}}_{[t]}^{r}={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^{r}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}}}{\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^{r}\right)
$$

这可以等效地表示为矩阵形式:

$$
{\mathbf{S}}_{[t+1]}\qquad =\color[\mathrm{rgb}]{0,0,1}{\gamma_{[t]}^{C}}\color[\mathrm{rgb}]{0,0,0}{\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}^{\top}{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\frac{\gamma_{[t]}^{C}}{\gamma_{[t]}}\right)}{\mathbf{K}}_{[t]}\tag{1}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\right){\mathbf{V}}_{[t]}\tag{2}
$$

我们观察到, (累积) 衰减项可以无缝地整合到矩阵乘法中, 几乎不增加计算开销. 这确保了块级并行形式保持高效, 并兼容基于高性能张量核心的加速.

### 2.3 Delta 网络: 具有 Delta 规则的线性注意力

Delta 更新规则 [Widrow60, Schlaa21] *动态* 擦除与当前输入键 (${\bm{k}}_{t}$) 相关的值 (${\bm{v}}_{t}^{\mathrm{old}}$) 并写入一个新值 (${\bm{v}}_{t}^{\mathrm{new}}$), 该新值是当前输入值与旧值的线性组合. 该过程在每个时间步更新一个键- 值关联对, 其中标量 $\beta_{t}\in(0,1)$ 决定旧关联被新关联替换的程度, 如下所示.

$$
{\mathbf{S}}_{t}\qquad ={\mathbf{S}}_{t-1}-\underbrace{\left({\mathbf{S}}_{t-1}{\bm{k}}_{t}\right)}_{ {\bm{v}}_{t}^{\mathrm{old}}}{\bm{k}}_{t}^{\top}+\underbrace{\left(\beta_{t}{\bm{v}}_{t}+(1-\beta_{t}){\mathbf{S}}_{t-1}{\bm{k}}_{t})\right)}_{ {\bm{v}}_{t}^{\mathrm{new}}}{\bm{k}}_{t}^{\top}={\mathbf{S}}_{t-1}\left({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}
$$

#### 分块并行形式.

通过部分展开递归, 我们得到

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^{r}{\mathbf{I}}-\beta_{[t]}^{i}{\bm{k}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)}_{:={\mathbf{P}}_{[t]}^{r}}+\underbrace{\sum_{i=1}^{r}\left(\beta^{i}_{[t]}{\bm{v}}^{i}_{[t]}{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^{r}\left({\mathbf{I}}-\beta_{[t]}^{j}{\bm{k}}^{j}_{[t]}{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{t}^{r}}\tag{3}
$$

其中 ${\mathbf{P}}_{[t]}^{j}$ 涉及过渡矩阵的累积乘积. [Yanga24] 表明这些矩阵呈 (广义) Householder 矩阵的形式, 通过经典 WY 表示法 [Loan85] 可以实现内存高效计算. 基于此, 他们引入了两种紧凑表示以优化该过程:

$$
{\mathbf{P}}_{[t]}^{r}\qquad ={\mathbf{I}}-\sum_{i=1}^{r}\mathbf{w}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{k}\times d_{k}}\qquad{\mathbf{H}}_{[t]}^{r}=\sum_{i=1}^{r}\mathbf{u}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{v}\times d_{k}}\tag{4}
$$

$$
\mathbf{w}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{k}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{w}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\quad\mathbf{u}_{[t]}^{r}=\beta_{[t]}^{r}\left({\bm{v}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{u}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\tag{5}
$$

其中 $\mathbf{w}_{[t]}^{r}\in\mathbb{R}^{d_{k}}$ 和 $\mathbf{u}_{[t]}^{r}\in\mathbb{R}^{d_{v}}$. 将这些代回方程 [3](#S2.E3) 并以矩阵形式, 我们得到:

$$
{\mathbf{S}}_{[t+1]}\qquad ={\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)^{\top}{\mathbf{K}}_{[t]}\tag{6}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\mathbf{M}})\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)\tag{7}
$$

## 3 门控增量网络

### 3.1 表述: 门控 delta 规则

所提出的门控 delta 规则既简单又有效:

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}\left(\color[\mathrm{rgb}]{0,0,1}{\alpha_{t}}\color[\mathrm{rgb}]{0,0,0}({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top})\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}\tag{8}
$$

其中数据依赖的门控项 $\color[\mathrm{rgb}]{0,0,1}{\alpha_{t}}\color[\mathrm{rgb}]{0,0,0}\in(0,1)$ 控制状态衰减. 该公式统一了两种门控机制和 delta 规则的优势: 门控项使自适应记忆管理成为可能, 而 delta 更新结构则有助于有效的键值关联学习.

我们通过 [Liua24] 引入的在线学习框架, 对门控 delta 规则进行了正式分析. 在该框架下, 递归状态更新表现为带有目标函数 $\bm{L_{t}}(S)$ 的在线学习问题的解. 如 [表 1](#table-01) 所示, 最近的线性 RNN 架构通常在在线学习目标中加入正则项, 以防止状态偏离先前值, 从而实现记忆保持. 然而, 当状态充满信息时, 这种保持机制会成为问题. 在这种情况下, 每个状态必须编码多个信息的叠加, 使得精确检索变得具有挑战性. 为了解决这一限制, Mamba2 和 Gated DeltaNet 引入了自适应缩放因子 $\alpha_{t}$, 该因子放宽了正则项, 从而允许 $S_{t}$ 和 $S_{t-1}$ 之间的受控偏差. 此修改通过选择性遗忘实现动态记忆管理.

另一方面, 线性注意力 (Linear Attention, LA) 和 Mamba2 使用简单的线性预测损失 $\langle{\mathbf{S}}_{t}{\bm{k}}_{t},{\bm{v}}_{t}\rangle$, 而 Longhorn [Liua24] 使用更具表现力的在线回归目标 $\|{\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}\|^{2}$, 以更好地建模键值关联. 由此得到的 Longhorn 更新规则与 delta 更新规则 [+1] 非常相似, 这表明 (门控) delta 规则在上下文关联回忆方面优于 Mamba2.

从快速权重编程 [Irie22] 和测试时训练 [Suna24] 的角度来看, 隐藏状态 ${\mathbf{S}}$ 可以被理解为权重矩阵, delta 规则通过*在线*随机梯度下降 (SGD) 优化目标 $L({\mathbf{S}}_{t})=\frac{1}{2}\|{\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}\|^{2}$:

$$
{\mathbf{S}}_{t+1}\qquad ={\mathbf{S}}_{t}-\beta_{t}\nabla_{S}L({\mathbf{S}}_{t})={\mathbf{S}}_{t}-\beta_{t}({\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}){\bm{k}}_{t}^{\top}={\mathbf{S}}_{t}\left({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}
$$

其中 $\beta_{t}$ 表示 (自适应) 学习率. 从这个角度来看, 门控 delta 规则可以被视为在 SGD 更新中引入自适应权重衰减项 $\alpha_{t}$, 这是一种在深度学习中广泛使用的技术 [Hertz91, Andriu23].

<span id="table-01"></span>

![论文原表 1](./gated-delta-networks/table-01.png)

**表 1.** 使用 [Liua24] 框架比较不同线性 RNN 模型及其对应的在线学习目标. 为了方便, 我们将 Longhorn 的向量值 ${\bm{\beta}}$ 简化为标量 $\beta$.

### 3.2 算法: 硬件高效的分块训练

在本小节中, 我们描述了一种用于门控 delta 规则的高效分块算法.

#### 分块并行形式.

通过部分展开递归, 我们得到

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^{i}}\left({\mathbf{I}}-\beta_{[t]}^{i}{\bm{k}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)\right)}_{:={\mathbf{P}}_{[t]}^{r}}+\underbrace{\sum_{i=1}^{r}\left(\beta^{i}_{[t]}{\bm{v}}^{i}_{[t]}{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^{r}{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^{j}}\left({\mathbf{I}}-\beta_{[t]}^{j}{\bm{k}}^{j}_{[t]}{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{[t]}^{r}}
$$

我们采用公式 [4](#S2.E4)-[5](#S2.E5) 中的 WY 表示, 将衰减项整合如下,

$$
{\mathbf{P}}_{[t]}^{r}\qquad =\color[\mathrm{rgb}]{0,0,1}{\gamma_{[t]}^{r}}\color[\mathrm{rgb}]{0,0,0}\left({\mathbf{I}}-\sum_{i=1}^{r}\mathbf{w}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)\qquad {\mathbf{H}}_{[t]}^{r}\qquad =\sum_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}^{r}}{\gamma_{t}^{i}}}\mathbf{u}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\tag{9}
$$

$$
\mathbf{w}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{k}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{w}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\qquad \mathbf{u}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{v}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{u}_{[t]}^{i}({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{[t]}^{r}}{\gamma_{[t]}^{i}}}{\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\tag{10}
$$

正确性证明可在附录中找到. 等效地, 用矩阵形式表示为:

$$
{\mathbf{S}}_{[t+1]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\gamma_{[t]}^{C}}{\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)^{\top}{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\frac{\gamma_{[t]}^{C}}{\gamma_{[t]}}\right)}{\mathbf{K}}_{[t]}\tag{11}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}})\left({\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)\tag{12}
$$

#### 与公式 [1](#S2.E1)-[2](#S2.E2) 的比较

我们可以看到, 关键区别在于将值块 ${\mathbf{V}}_{[t]}$ 替换为“伪”值项 ${\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}$. 此修改类似于 Eq. [6](#S2.E6)-[7](#S2.E7), 但显著地引入了衰减感知.

#### UT 变换.

为了最大化硬件效率, 我们将 UT 变换 [Joffra06] 应用于公式 [10](#S3.E10). 该技术将操作重新构造为矩阵乘法形式, 从而减少非矩阵乘法的 FLOPs, 这对于在训练期间实现更好的硬件利用率至关重要 [Daob23, Refc23, Yang24].

$$
{\mathbf{W}}_{[t]}\qquad ={\mathbf{A}}^{W}_{[t]}\mathrm{Diag}(\beta_{[t]}){\mathbf{K}}_{[t]},\qquad {\mathbf{A}}^{W}_{[t]}=\left({\mathbf{I}}-\mathrm{lower}(\mathrm{Diag}(\beta_{[t]}){\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^{\top})\right)^{-1}
$$

$$
{\mathbf{U}}_{[t]}\qquad ={\mathbf{A}}^{U}_{[t]}\mathrm{Diag}\left(\beta_{[t]}\right){\mathbf{V}}_{[t]},\qquad {\mathbf{A}}^{U}_{[t]}=\left({\mathbf{I}}-\mathrm{lower}\left(\mathrm{Diag}(\beta_{[t]})\left({\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\odot{\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\right)\right)\right)^{-1}
$$

其中 $\mathrm{lower}(\cdot):=\mathrm{tril}(\cdot,-1)$; 下三角矩阵的逆可以通过前向代换高效计算.

#### 关于速度的备注.

类似于 Mamba2, 门控项 (蓝色标记) 仅对 (中间) 变量进行元素级乘法, 而不影响矩阵乘法结构, 从而实现张量核心 GPU 优化. 如 [图 3](#figure-03) 所示, Gated DeltaNet 的速度与 DeltaNet 相同, 尽管具有更复杂和更具表现力的转移矩阵, 但与 Mamba2 仅有小的性能差距.

### 3.3 门控 Delta 网络与混合模型

<span id="figure-01"></span>

![图 1](./gated-delta-networks/figure-01.png)

**图 1.** Gated DeltaNet 模型 (混合) 架构和模块设计的可视化. Gated DeltaNet-H1 和 H2 分别使用 Gated DeltaNet SWA 和 Mamba2 Gated DeltaNet SWA 模式. 在模块设计中, 查询/键路径包括线性映射, 短卷积, SiLU 和 L2 归一化; 值路径包括线性映射, 短卷积和 SiLU; alpha/beta 使用线性映射; 输出门使用带 SiLU 的线性映射.

#### 令牌混合块.

基本的门控增量网络遵循 Llama 的宏观架构, 将 token 混合层与 SwiGLU MLP 层堆叠, 但用门控 delta 规则 token 混合替换了自注意力机制. [图 1](#figure-01) (右) 显示了其模块设计. 对于门控 delta 规则 (公式 [8](#S3.E8) ), 查询, 键和值 $\{ {\bm{q}},{\bm{k}},{\bm{v}}\}$ 通过线性投影, 短卷积和 SiLU 生成, 并对 ${\bm{q}},{\bm{k}}$ 进行 L2 归一化以保证训练稳定性. $\alpha,\beta$ 仅使用线性投影. [+2] 按照 [Suna23], 输出在应用输出投影之前会经过归一化和门控处理.

#### 混合模型

线性 Transformer 在建模局部偏移和比较方面存在局限性, 并且其固定状态大小使检索任务变得困难 [Arora24]. 遵循最近的一些混合架构如 Griffin [De24] 和 Samba [Ren24], 我们将线性循环层与滑动窗口注意力 (SWA) 结合, 形成 GatedDeltaNet-H1. 我们还将 Mamba2, GatedDeltaNet 与 SWA 堆叠, 形成 GatedDeltaNet-H2.

## 4 实验

#### 设置

我们的实验涵盖了对近期最先进架构的全面比较, 包括纯 Transformer 模型, 基于 RNN 的方法和混合架构. 我们评估了以下基线模型: RetNet [Suna23], HGRN2 [Qin24a], Mamba [Daoc23], Mamba2 [Daob24], Samba [Ren24] 和 DeltaNet [Yanga24]. 为了公平比较, 所有模型在相同条件下在 1.3B 参数上训练, 使用从 FineWeb-Edu 数据集中采样的 100B tokens [Penedo24]. 我们使用 AdamW 优化器, 峰值学习率为 4e-4, 权重衰减为 0.1, 梯度裁剪为 1.0. 学习率采用余弦退火调度, 预热期为 1B tokens, 批量大小为 0.5M tokens. 所有模型都使用词汇量为 32, 000 的 LLaMA 2 分词器. 对于序列建模, 我们将训练长度设置为 4K tokens, Samba 和我们的混合模型使用 2K 的滑动窗口大小. 评估设置见附录.

<span id="table-02"></span>

![论文原表 2](./gated-delta-networks/table-02.png)

**表 2.** 语言建模和零样本常识推理的性能比较.

#### 常识推理

在[表 2](#table-02) 中, 我们展示了具有 4 亿和 13 亿参数模型在常识推理基准上的语言建模困惑度和零样本准确率. 门控 DeltaNet 在两种规模下均持续优于其他线性模型, 包括 RetNet, HGRN2, Mamba, Mamba2 和 DeltaNet. 如预期, 混合变体进一步提升了性能.

<span id="table-03"></span>

![论文原表 3](./gated-delta-networks/table-03.png)

**表 3.** S-NIAH 基准测试套件的性能比较.

#### 合成数据的上下文检索

[表 3](#table-03) 显示了来自 RULER [Hsieh24] 的单针寻草 (S-NIAH) 基准套件的结果. 在最简单的 S-NIAH-1 设置中使用合成输入时, DeltaNet 在所有序列长度上都实现了接近完美的性能, 这得益于其 delta 更新规则, 该规则对上下文记忆回溯特别有利 (§[3.1](#S3.SS1)). 相比之下, Gated DeltaNet 的检索准确率略低, 因为其门控机制会丢弃信息, 从而影响完美记忆的保持, 而 Mamba2 在超过 2K 序列后性能显著下降.

然而, 记忆检索不仅依赖于保持能力, 还依赖于“遗忘”的能力: 在固定状态大小下, 缺乏记忆清理会导致状态饱和时出现记忆冲突——多条信息叠加在一起, 使其无法分辨. 这在 NIAH-2 和 NIAH-3 中尤为明显, 这些基准中的针扎基于现实世界文本数据: DeltaNet 的性能显著下降, 而 Gated DeltaNet 的自适应记忆管理显示出相较于 Mamba2 和 DeltaNet 的明显优势.

#### 现实世界数据中的上下文检索

<span id="table-04"></span>

![论文原表 4](./gated-delta-networks/table-04.png)

**表 4.** 在回忆世界检索任务中的准确率, 输入截断为 2K 令牌. SQD: SQUADE. TQA: 简单问答.

[表 4](#table-04) 展示了由 [Aroraa24] 使用的现实世界高召回任务的结果. 如预期, 线性递归模型与 Transformer 相比表现出明显的性能差距, 而结合线性递归和注意力的混合模型在检索任务中优于纯注意力模型.

对于纯循环模型, 尽管 DeltaNet 在合成的上下文检索任务 [Yanga24] 上表现优越, 其在真实世界的检索性能仍落后于 Mamba2, 这与我们在 S-NIAH-2 和 S-NIAH-3 中的观察一致 ([表 3](#table-03)). 得益于其门控 delta 规则, Gated DeltaNet 的表现优于 DeltaNet 和 Mamba2, 但改进幅度比 [表 3](#table-03) 中的要小. 我们将这一性能差距的缩小归因于未按指令对齐的小型语言模型容易出现重复错误, 而重复错误是这些任务中错误的主要来源 (参见 [Aroraa24]). 由于这一问题在很大程度上与更新规则的选择无关, 不同模型之间的性能差异相比 [表 3](#table-03) 更不明显.

<span id="figure-02"></span>

![图 2](./gated-delta-networks/figure-02.png)

**图 2.** 六个长序列基准上的长度外推.

#### 长序列的长度外推.

如图[2](#figure-02) 所示, 我们评估了模型在六个长上下文基准测试中对最多 20K 标记序列的外推能力. 在 RNN 模型中, Gated DeltaNet 在各任务中实现了最低的整体困惑度. 虽然我们在长度外推中观察到结果参差不齐, 但 Gated DeltaNet 表现出相对更稳健的性能, 表明其内存管理能力更好. 通过利用注意力进行局部上下文建模, 混合模型进一步改进了这一点, 从而减轻了其递归组件的内存管理负担. 未来工作将探索这些模型在更长序列上的能力.

#### 长上下文理解

如[表 5](#table-05) 所示, 我们评估了模型在 LongBench [Bai23] 上的表现. 在递归模型中, Gated DeltaNet 显示出持续的优势, 尤其在单文档问答, 少量样本的上下文学习及代码任务中, 分别展示了其在检索, 上下文学习和状态跟踪方面的优越能力.

<span id="table-05"></span>

![论文原表 5](./gated-delta-networks/table-05.png)

**表 5.** LongBench 上 14 个任务的准确率: 按顺序分别为 Narrative QA, QasperQA, MultiField QA, HotpotQA, 2WikiMulti QA, Musique, GovReport, QMSum, MultiNews, TRec, Trivia QA, SamSum, LCC 和 RepoBench-P.

<span id="figure-03"></span>

![图 3](./gated-delta-networks/figure-03.png)

**图 3.** 单个 H100 GPU 上 1.3B 模型的训练吞吐量比较.

#### 吞吐量比较

不同模型的训练吞吐量比较如[图 3](#figure-03) 所示. 正如我们的分析显示的, 提出的门控 Delta 规则相比原始 Delta 规则仅引入了极少的开销, Gated DeltaNet 实际上实现了与 DeltaNet 相同的吞吐量. 由于其更具表现力的转移矩阵, 两者的速度略低于 Mamba2 (2-3K token/秒).

Transformer 在 2K 上下文窗口领域表现最佳, 这得益于高度优化的 Flash-Attention-2 内核 [Daob23]. 因此, 将 2K 窗口大小的 SWA 注意力与其他 token 混合器结合的混合方法表现出比单独混合器更高的吞吐量: Samba 的表现优于 Mamba, 而 Gated DeltaNet-H1 和 -H2 的表现优于 Gated DeltaNet. Gated DeltaNet-H1 在所有序列长度下都保持了强劲的训练吞吐量, 即使是在短序列上.

## 5 相关工作

#### 门控线性 RNN.

大型线性递归语言模型因其训练和推理效率而受到广泛关注. 线性 RNN 领域已经迅速发展, 从使用与数据无关的衰减机制 (如 S4 [Refg22], S5 [Smith23], LRU [Orviet23], RWKV4/5 [Penga23] 和 RetNet [Suna23] 等模型) 到在更新的架构中融合与数据相关的衰减机制, 例如 HGRN1/2 [Qin24a, Qina23], Mamba1/2 [Daoc23, Daoa24], RWKV6 [Peng24] 和 GSA [Zhanga24]. 这一转变源于门控/遗忘机制 (在 Mamba 中称为选择性机制) 的验证优势——这是源自门控 RNN 文献 [Gersa00] 的经典概念, 其重要性已得到持续确认 [Greff15, Lasenb18, Qin24a, Qina23, Daoc23].

现代遗忘门不同于 LSTM 等传统设计, 它们取消了对先前隐藏状态的依赖, 仅依赖输入数据. 这一修改使得跨序列长度的高效并行成为可能 [Cundy18, Qina23]. 在 DeltaNet 中缺乏遗忘门一直是一个显著局限, 我们的门控扩展以自然且有效的方式弥补了这一空白.

#### Delta 法则.

已经证明, 增量学习规则在记忆容量方面优于赫布学习规则 [Gardne88, Kak89]. 虽然线性 Transformer 依赖于类似赫布的学习规则, 但 DeltaNet 使用 delta 规则, 而这一记忆容量优势在合成的上下文学习任务中有实验证据. 另外, 这种优势延伸到各种应用, 包括语言建模 [Irie21, Yanga24], 强化学习 [Iriea22] 以及图像生成 [Schmid23]. [Yanga24] 进一步将 delta 规则计算在序列长度上并行化, 并展示了 DeltaNet 转移矩阵的增强表现力. 具体而言, DeltaNet 的依赖数据的恒等加低秩结构 (${\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}$) 相比 Mamba2 的依赖数据的对角矩阵 ($\alpha_{t}{\mathbf{I}}$) 提供了更大的灵活性. 这一从对角矩阵到结构化密集矩阵的架构转变显著提升了模型在复杂推理任务中的能力, 包括正则语言识别 [Fan24, Grazzi24] 以及超出 TC0 复杂度类的状态跟踪任务 [Merril24]——这些能力对于像编码这样的应用尤为关键. [Grazzi24] 的最新研究表明, 允许 DeltaNet 中存在负特征值可能进一步增强其状态跟踪能力, 这也可以直接整合到门控 DeltaNet 中.

德尔塔规则与通过梯度下降的在线 (元) 学习之间存在一个有趣的联系 [Munkhd19, Irie22]. 像 Longhorn [Liua24] 和 TTT [Suna24] 这样的最新架构通过将状态空间学习重新表述为基于梯度的在线学习问题来重新审视这种关系 (另见 §[3.1](#S3.SS1)). 虽然 Longhorn 提供了更理论上严谨的表述, 但其依赖对角近似显著削弱了表现力. TTT 提出了一个有趣的案例: 其不带 Layernorm 的线性变体等同于 DeltaNet, 但添加 Layernorm 后, 它会转变为非线性 RNN 模型. 这种转变需要一种混合训练方法, 其中“类似德尔塔规则”的方法在每个 N 个标记的块级别应用 (N 为块大小).

尽管具有优势, 德尔塔规则仍存在理论局限 [Irie23], 并且在实际数据集上的表现适中 [Yanga24]. 之前的扩展通过严格的*非线性*递归提升了表现力 [Irie21, Iriea22], 但牺牲了训练并行性. 我们的门控 DeltaNet 保持了线性 RNN, 既能够高效训练, 又通过门控机制提升表现力, 从而在各项任务中实现一致改进. 未来的工作可以采用类似 GLA 的对角门控 [Yang24], 以进一步放宽门控限制.

## 6 结论

在这项工作中, 我们引入了 Gated DeltaNet, 它相比 Mamba2 可以实现更好的键值关联学习, 并且相比 DeltaNet 拥有更自适应的记忆清理, 从而在各种任务中实现了持续更好的实证结果. 我们将 [Yanga24] 中的并行算法进行了扩展, 以实现 Gated DeltaNet 的硬件高效训练. 我们的混合 Gated DeltaNet 模型实现了更高的训练吞吐量和整体性能, 使其非常适合实际部署.

## 致谢

感谢 Yu Zhang 协助绘制图表, Simeng Sun 和 Zhixuan Lin 在评估方面提供宝贵讨论, 以及 Eric Alcaide 对 DeltaNet 在线学习视角的深刻反馈.

## 附录 A

### A. 1 门控 delta 规则的扩展 WY 表示

为了减少符号混乱, 这里我们仅考虑第一个块.

对 ${\mathbf{S}}_{t}$, 扩展的 WY 表示为

$$
{\mathbf{S}}_{t}=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top},\qquad\mathbf{u}_{t}=\beta_{t}\left({\bm{v}}_{t}-\sum_{i=1}^{t-1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^\top{\bm{k}}_{t}\right)
$$

我们通过数学归纳法证明这一点.

###### 证明

$$
\centering{\mathbf{S}}_{t+1}\@\mathrm{add}@\mathrm{centering}\qquad ={\mathbf{S}}_{t}\left({\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}({\mathbf{I}}-\beta_{t+1}{\bm{k}}_{t+1}{\bm{k}}_{t+1}^{\top})\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
={\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}(\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top})-{\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}\beta_{t+1}(\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}{\bm{k}}_{i}{\bm{k}}_{t+1}^{\top})+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}+\underbrace{\beta_{t+1}\left({\bm{v}}_{t+1}-\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^\top{\bm{k}}_{t+1}\right)}_{\mathbf{u}_{t+1}}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}+\underbrace{ {\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{t+1}}}}_{1}\mathbf{u}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t+1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}
$$

∎

对 ${\mathbf{P}}_{t}$, ${\mathbf{P}}_{t}$ 以及

$$
{\mathbf{P}}_{t}\qquad =\prod_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\alpha_{t}}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)
$$

$$
={\color[\mathrm{rgb}]{0,0,1}\underbrace{\left(\prod_{i=1}^{t}\alpha_{t}\right)}_{\gamma_{t}}}\underbrace{\left(\prod_{i=1}^{t}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)\right)}_{ {\mathbf{I}}-\sum_{i=1}^{t}\mathbf{w}_{i}{\bm{k}}_{i}^{\top}}
$$

已经在 [Yanga24] 中得到证明.

$$
\prod_{i=1}^{t}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)={\mathbf{I}}-\sum_{i=1}^{t}\mathbf{w}_{i}{\bm{k}}_{i}^{\top},\quad\mathbf{w}_{n}=\beta_{n}{\bm{k}}_{n}-\beta_{n}\sum_{t=1}^{n-1}\left(\mathbf{w}_{t}({\bm{k}}_{t}^{\top}{\bm{k}}_{n}\right)
$$

已经在 [Yanga24] 中得到证明.

### A. 2 消融研究

<span id="table-06"></span>

![论文原表 6](./gated-delta-networks/table-06.png)

**表 S. 1.** 关于 Gated DeltaNet 模块的消融研究. Avg-PPL 和 Avg-Acc 分别表示平均困惑度和零样本常识推理准确率 (如 [表 2](#table-02) 所示). 所有模型均有 4 亿参数, 并在相同的 FineWeb-Edu 数据集子集上训练 150 亿 token [Penedo24].

<span id="table-07"></span>

![论文原表 7](./gated-delta-networks/table-07.png)

**表 S. 2.** Gated DeltaNet 模型的消融研究. 所有评估均使用 lm-evaluation-harness [Gaob21] 进行. 所有模型都使用 Llama 分词器, 并在 FineWeb-Edu 数据集 [Penedo24] 的相同子集上进行训练.

[表 S. 1](#table-06) 展示了对 Gated DeltaNet 模块各组件的消融研究. 我们的实验表明, 短卷积和输出门对模型性能都至关重要, 而输出归一化带来的改善有限. 与 [Yanga24] 的结果一致, 我们发现 L2 归一化对实现最佳性能是必不可少的, 尽管特征图的选择影响较小. 尽管如此, SiLU 一直优于其他激活函数, 这与 [Qin23] 的观察结果相符. 通过经验分析, 我们确定将头部维度设置为 128 能在性能和计算效率之间提供最佳折衷. 另外, [表 S. 2](#table-07) 表明, 在各种混合架构中, 按特定顺序组合 Mamba2, Gated DeltaNet 和 SWA 可以产生更优的结果.

## 附录 B 实验设置

### B. 1 评估

#### 常识推理

遵循 [Daoc23], 我们在多个常识推理基准上评估了我们的模型: PIQA [Biska20], HellaSwag [Zellea19], WinoGrande [Sakagu20], ARC-easy (ARC-e) 与 ARC-challenge (ARC-c) [Clark18], SIQA [Sap19], BoolQ [Clark19], Wikitext [Merity17] 及 LAMBADA [Papera16].

#### 上下文检索

我们的评估包括合成任务和真实世界任务. 对于合成任务, 我们使用来自 RULER [Hsieh24] 的 Needle-In-A-Haystack Single (NIAH-S) 基准套件, 其中包括三个逐渐复杂的任务: S-NIAH-1 (密钥检索) , S-NIAH-2 (数字大海捞针) , S-NIAH-3 (基于单词的大海捞针). 对于真实世界任务, 遵循 [Aroraa24] 的方法, 我们在多样化数据集上进行评估: 用于结构化 HTML 关系抽取的 SWDE [Lockar19], 用于 PDF 键值检索的 FDA [Aroraa23], 以及多个问答数据集, 包括 SQuAD [Rajpur18], TriviaQA [Joshi17], Drop [Duaa19] 和 NQ [Kwiatk19]. 由于我们预训练模型缺乏指令微调, 我们采用 [Aroraa24] 提供的完形填空格式提示, 这更符合我们模型的下一个词预测训练目标.

#### 长上下文理解

我们在 Longbench [Bai23] 的 14 个任务上进行评估, 涵盖: 叙事理解 (Narrative QA [Refa18]) , 科学理解 (QasperQA [Dasigi21]) , 多跳推理 (MultiField QA, HotpotQA [Yang18], 2WikiMulti QA [Refd20], Musique [Trived22]) , 文档摘要 (GovReport [Huang21], QMSum [Zhong21], MultiNews [Fabbri19]) , 以及各种专业任务 (TRec [Roth02], Trivia QA [Joshia17], SamSum [Gliwa19], LCC [Guoa23] 和 RepoBench-P [Liub23]).

[+1]: 理论上的区别在于优化方法: Longhorn 使用隐式在线学习 [Bartle10] 来推导闭式全局最优更新, 而 DeltaNet 则通过一步显式梯度下降来优化相同的目标, 如 [Liua24] 所指出的. 尽管 Longhorn 在理论上基础更强, 但我们发现这些方法在实际性能上没有显著差异, 因此仍保持 DeltaNet 的原始公式.

[+2]: 我们使用 Mamba2 的参数化方式 $\alpha$, 但为简洁起见省略它.
