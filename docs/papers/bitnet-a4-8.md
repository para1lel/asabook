---
title: 'BitNet a4.8: 4-bit Activations for 1-bit LLMs'
createTime: 2026/09/08 16:34:10
permalink: /papers/bitnet-a4-8/
pageClass: paper-reading
---

> [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Shuming Ma](https://shumingma.com/) [+author-note], 以及 [Furu Wei](https://thegenerality.com/)◇. [GeneralAI](https://aka.ms/GeneralAI). 论文于 2024 年 11 月 7 日首次提交至 arXiv; 当前版本为 v1; 仍在完善中. [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965). <a href="/paper/bitnet-a4-8.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2411.04965). [TeX 源文件](https://export.arxiv.org/e-print/2411.04965v1). 精确的印刷排版和参考文献以原始 PDF 为准.

[+author-note]: 贡献相同. ◇ 通讯作者. S. Ma 和 F. Wei 就职于 Microsoft Research. H. Wang 就职于中国科学院大学.

## 摘要

近期针对 1-bit 大语言模型 (LLM) 的研究, 如 BitNet b1.58 [Ma24], 为降低 LLM 推理成本并保持其性能提供了一条颇有前景的路径. 本文提出 **BitNet a4.8**, 使 1-bit LLM 能够使用 4-bit 激活值. BitNet a4.8 采用混合量化与稀疏化策略, 以缓解离群通道带来的量化误差. 具体而言, 我们对注意力层和前馈网络层的输入使用 4-bit 激活值, 对中间状态则先做稀疏化, 再进行 8-bit 量化. 大量实验表明, 在训练成本相同的情况下, BitNet a4.8 的性能与 BitNet b1.58 相当; 借助 4-bit (INT4/FP4) 内核, 其推理速度更快. 此外, BitNet a4.8 仅激活 55% 的参数, 并支持 3-bit KV 缓存, 进一步提高了大规模 LLM 部署和推理的效率.

<span id="figure-01"></span>

![BitNet a4.8 的权重和激活值量化概览](./bitnet-a4-8/figure-01.png)

**图 1.** BitNet a4.8 的权重与激活值量化概览. 所有参数均为三值 (即与 BitNet b1.58 [Ma24] 相同的 1.58-bit). 我们使用混合量化与稀疏化策略处理部分 Transformer 子层中的离群激活值.

<span id="section-1"></span>

## 1 引言

近期研究 [Ma24] 表明, 在参数量和训练 token 数相同的情况下, 1-bit LLM 的性能可以媲美全精度模型, 同时在延迟, 内存, 吞吐量和能耗方面显著降低成本. 当模型权重以 1.58-bit (即 $\{-1, 0, 1\}$) 表示时, 推理瓶颈已从有限的内存带宽转向高昂的计算成本. 在 LLM 中采用低比特或稀疏激活值, 有望进一步降低计算预算, 同时保持下游任务的性能.

一种常见做法是利用激活稀疏性 [Liu23t, Son24c, Liu24aa], 剪除幅值较小的激活值项, 从而减少推理 FLOPs 和权重 I/O. 稀疏化尤其适合处理呈高度不平衡长尾分布的激活值. 近期研究 [Wan24ag] 表明, 激活值全面稀疏化的 LLM 可以用少得多的活跃参数取得与稠密模型相当的结果.

除稀疏化之外, 激活值量化也是加速矩阵乘法的一种方法. 然而, 随着训练推进和模型规模增大, 离群维度随之出现, 因而使用低比特激活值优化神经网络颇具挑战. 尽管这些离群值只占激活值的极小部分 [Det22, Xia23], 其幅值却大得多, 会在下游任务中造成显著的量化误差和性能下降. 以往研究 [Xi23, Ash24, Liu24b, Lin24b] 多使用 Hadamard 变换或可学习的旋转变换, 将离群特征分摊到其他项中. 不过, 这些方法大多面向精度更高的 LLM (如 4-bit). 对 1-bit LLM 而言, 权重的位宽极低, 很难将这些变换矩阵直接吸收到权重中; 若保留为在线变换, 又会增加计算开销, 限制整体推理性能.

本文提出 **BitNet a4.8**, 这是一种混合量化与稀疏化策略, 使 1-bit LLM 能够采用 4-bit 激活值. 通过细致分析 1-bit LLM 的激活值分布, 我们根据这些激活值的分布模式, 有选择地应用 4-bit 量化或稀疏化. 具体而言, 如 [图 1](#figure-01) 所示, BitNet a4.8 对注意力和 FFN 的输入采用 4-bit 激活值, 对中间状态则使用 8-bit 稀疏化表示. 为提高训练效率, BitNet a4.8 采用两阶段方案, 将激活值从 8-bit 训练到 4-bit; 只需在训练末期使用少量训练 token, 即可让 BitNet b1.58 适应低比特激活值. 大量实验表明, 在训练成本相同的情况下, BitNet a4.8 的性能可与 BitNet b1.58 竞争, 推理效率则明显更高. 此外, BitNet a4.8 仅有 55% 的参数处于激活状态, 并支持 3-bit KV 缓存, 进一步提高了 LLM 的部署效率.

<span id="section-2"></span>

## 2 BitNet a4.8

<span id="section-2-1"></span>

### 2.1 架构

如 [图 1](#figure-01) 所示, BitNet a4.8 采用与 BitNet b1.58 相同的布局. 按照 [Wan23, Ma24] 的做法, 我们将注意力和前馈网络 (FFN) 中的线性投影替换为 BitLinear, 从头学习 1.58-bit 权重. 对激活值, 我们采用混合量化与稀疏化策略, 以缓解离群维度引入的误差.

[图 2](#figure-02) 展示了模型规模为 7B 的 BitNet b1.58 模型各组件输入的分布. 注意力层和 FFN 层的输入通常近似高斯分布, 而 FFN 下投影之前以及注意力输出投影处的激活值具有更多离群通道, 还有大量数值集中在零附近. [Liu24aa] 也报告了全精度 LLM 中的类似现象. 如 [图 3](#figure-03) 所示, 直接对这些中间状态应用低比特量化会引入大量量化误差.

<span id="figure-02"></span>

![各投影输入的激活值分布](./bitnet-a4-8/figure-02.png)

**图 2.** 各投影输入的分布. 此可视化使用 7B BitNet b1.58 模型, 数据取自 C4 验证集的一个子集. 对呈类高斯分布的层, 我们采用 4-bit 激活值量化. 对分布尖锐的层, 我们采用 Q-Sparse [Wan24ag] 对激活值进行稀疏化.

因此, 我们使用 Q-Sparse [Wan24ag] 的稀疏化方法, 在将这些中间状态保持为 8-bit 的同时消除计算瓶颈. 对自注意力层的输出投影, 我们使用先稀疏化, 后量化的函数:

<span id="equation-01"></span>

$$
\mathbf{Y} = \left(\mathrm{Q}_{\mathrm{INT}8}(\mathbf{X}) \odot \mathbf{M}\right) \cdot \mathrm{Q}_{w}(\mathbf{W})^\top, \quad \mathbf{M} = \mathrm{Top}_k\left(|\mathbf{X}|\right)
$$

其中, $\mathrm{Q}_{w}(\cdot)$ 和 $\mathrm{Q}_{\mathrm{INT}8}(\cdot)$ 分别表示权重 $\mathbf{W}$ 和激活值 $\mathbf{X}$ 的量化函数. $\mathbf{M}$ 是掩码张量, 表示按激活值 $\mathbf{X}$ 的绝对值计算得到的最大 top-K 元素; $\odot$ 表示逐元素乘法.

具体而言, 权重量化和激活值量化函数可以写为:

<span id="equation-02"></span>

$$
\mathrm{Q}_{w}(\mathbf{W}) = \alpha\mathrm{RoundClip}\left(\frac{\mathbf{W}}{\alpha+\epsilon}, -1, 1\right), \quad \alpha = \mathrm{mean}(|\mathbf{W}|)
$$

<span id="equation-03"></span>

$$
\mathrm{Q}_{\mathrm{INT}8}(\mathbf{X}) = \frac{\gamma}{127}\mathrm{RoundClip}\left(\frac{127}{\gamma+\epsilon}\mathbf{X}, -128, 127\right), \quad \gamma = \max(|\mathbf{X}|)
$$

<span id="equation-04"></span>

$$
\mathrm{RoundClip}(X, a, b) = \min\left(\max(\mathrm{round}(X), a), b\right)
$$

对于 FFN, 我们采用平方 ReLU [So21, Wan24ag] 和门控线性单元 (GLU), 进一步提高激活稀疏性. 其定义如下:

<span id="equation-05"></span>

$$
\mathrm{ReLU}^2\mathrm{GLU}(\mathbf{X}) = \mathbf{X}\mathbf{W}_{\mathrm{up}}^\top \odot \mathrm{ReLU}^2\left(\mathbf{X}\mathbf{W}_{\mathrm{gate}}^\top\right)
$$

根据初步实验, 使用平方 ReLU 后, 下投影输入的稀疏度超过 80%, 对性能的影响很小. 此外, 我们观察到门控投影的输出 $\mathrm{ReLU}^2(\mathbf{X}\mathbf{W}_{\mathrm{gate}}^\top)$ 同样具有很高的激活稀疏度 (例如, 7B 模型为 67.5%). 这一特性可以进一步减少上投影的推理 FLOPs: 先计算门控投影, 再只对门控中的非零通道执行上投影.

注意力和 FFN 的输入具有少得多的离群特征, 因此我们使用 absmean 函数将激活值量化为 4-bit 整数:

<span id="equation-06"></span>

$$
\mathbf{Y} = \mathrm{Q}_{\mathrm{INT}4}(\mathbf{X}) \cdot \mathrm{Q}_{w}(\mathbf{W})^\top
$$

<span id="equation-07"></span>

$$
\mathrm{Q}_{\mathrm{INT}4}(\mathbf{X}) = \frac{\beta}{\sqrt{7}}\mathrm{RoundClip}\left(\frac{\sqrt{7}}{\beta+\epsilon}\mathbf{X}, -8, 7\right), \quad \beta = \mathrm{mean}(|\mathbf{X}|)
$$

<span id="section-2-2"></span>

### 2.2 训练

**从 BitNet b1.58 继续训练.** BitNet a4.8 采用两阶段方案, 从 W1.58A8 训练到 W1.58A4. 第一阶段以 8-bit 激活值和 $\mathrm{ReLU}^2\mathrm{GLU}$ 训练模型. 第二阶段采用 [第 2.1 节](#section-2-1) 所示的混合量化与稀疏化. BitNet a4.8 只需少量训练 token, 即可迅速适应 4-bit 和稀疏激活值, 性能损失可以忽略.

**梯度近似.** 按照 [Wan23, Wan24ag] 的做法, 我们使用直通估计器 (STE) [Ben13] 对 BitNet a4.8 进行梯度近似, 并使用混合精度训练更新参数. 反向传播期间, 我们直接绕过量化函数和 top-K 稀疏化函数等不可微函数. 在混合精度训练中, 我们保留一个全精度潜在权重, 用于累积参数更新. 前向传播时, 我们即时将潜在权重量化为 1.58-bit.

<span id="figure-03"></span>

![量化与稀疏化下的输出投影输入分布](./bitnet-a4-8/figure-03.png)

**图 3.** 不同量化与稀疏化方式下, 注意力输出投影输入的分布. 此可视化使用 7B BitNet b1.58 模型, 数据取自 C4 验证集的一个子集.

<span id="section-2-3"></span>

### 2.3 浮点量化

浮点量化比基于整数的量化具有更宽的动态范围, 这对处理激活值的长尾分布很重要. 在浮点精度方案中, 我们只将 FFN 下投影的输入保留为 8-bit 整数, 其余激活值则使用 MinMax 量化器 [Liu23c] 量化为 FP4. 其定义如下:

<span id="equation-08"></span>

$$
\mathrm{Q}_{\mathrm{FP}4}(\mathbf{X}) = \frac{\gamma}{2^{M+b}}\mathrm{Round}\left(\frac{2^{M+b}}{\gamma}\mathbf{X}\right), \quad \gamma = 2^{\max\left(\left\lfloor\left\lfloor\log_2|\mathbf{X}|\right\rfloor+b\right\rfloor,1\right)}
$$

<span id="equation-09"></span>

$$
b = \log_2\left(\frac{2-2^{-M}}{|\mathbf{X}|_{\max}}\right) + 2^E - 1
$$

其中, $E$ 和 $M$ 分别表示指数部分和尾数部分的位宽. 我们采用动态范围更大的 E2M1 格式. 如 [表 1](#table-01) 所示, 使用 FP4 量化的 BitNet a4.8 与采用基于整数的混合量化与稀疏化策略时性能相近.

<span id="table-01"></span>

![BitNet a4.8, BitNet b1.58 与 LLaMA LLM 的困惑度和下游任务结果](./bitnet-a4-8/table-01.png)

**表 1.** BitNet a4.8, BitNet b1.58 和 LLaMA LLM 的困惑度及下游任务结果. 平均分数的误差标准差为 1.06%.

<span id="section-3"></span>

## 3 实验

我们比较了不同规模的 BitNet a4.8, BitNet b1.58 和自行复现的 FP16 LLaMA LLM. 对 1.58-bit 模型, 我们按照 BitNet b1.58 [Ma24] 的训练方案, 采用两阶段权重衰减和学习率调度. 更多细节见 [第 5 节](#section-5). 为确保公平比较, 所有模型均使用 RedPajama 数据集 [Tog23a] 中的 100B token 进行训练. 对 BitNet a4.8, 我们先使用 8-bit 激活值训练 95B token. 随后复用优化器状态, 再用本文提出的混合量化与稀疏化方法继续训练 5B token. 注意力输出投影的 topK 设为 50%.

我们使用 *lm-evaluation-harness* 工具包 [Gao24h], 在一系列语言任务上评估这些模型的 zero-shot 准确率, 包括 ARC-Easy (ARCe) [Yad19], ARC-Challenge (ARCc) [Yad19], Hellaswag (HS) [Zel19], Winogrande (WGe) [Sak20] 和 PIQA (PQ) [Bis20]. 我们还报告了 C4 数据集 [Raf19] 验证集上的困惑度.

<span id="section-3-1"></span>

### 3.1 主要结果

[表 1](#table-01) 汇总了 BitNet a4.8, BitNet b1.58 和 FP16 LLaMA LLM 的详细结果. 随着模型规模增大, 全精度 (即 FP16) LLaMA LLM 与 BitNet b1.58 之间的性能差距逐渐缩小. 对 7B 模型, BitNet b1.58 在语言模型困惑度和下游任务平均准确率两方面均与 LLaMA LLM 相当. 此外, BitNet a4.8 的性能与 BitNet b1.58 相当, 平均准确率几乎没有损失.

<span id="table-02"></span>

![BitNet a4.8, BitNet b1.58 与 LLaMA LLM 的各组件详细稀疏度](./bitnet-a4-8/table-02.png)

**表 2.** BitNet a4.8, BitNet b1.58 和 LLaMA LLM 在 C4 验证集上的详细稀疏度.

**稀疏度.** [表 2](#table-02) 展示了不同规模下 BitNet a4.8, BitNet b1.58 和 FP16 LLaMA LLM 各组件的详细稀疏度. 稀疏度根据 C4 验证集上的非嵌入参数计算. BitNet a4.8 的稀疏度明显高于 BitNet b1.58 和 LLaMA LLM. 例如, 7B 模型的整体稀疏度达到 44.5%, 仅有 3.4B 个活跃参数. 下投影输入的稀疏度尤其高, 这与我们观察到的中间状态分布尖锐集中于零附近相符. 我们还观察到门控投影的输出非常稀疏. 因为只需对门控选出的非零通道执行投影, 上投影也因此具有很高的稀疏度. 具体来说, 对 7B BitNet a4.8, 门控和上投影输入的稀疏度分别为 67.5% 和 12.0%. 因此, 上投影的稀疏度可估算为 $1 - (1 - 12.0\%)\times(1 - 67.5\%)$, 即 71.4%.

<span id="table-03"></span>

![3B 与 7B BitNet a4.8 的低比特 QKV 结果](./bitnet-a4-8/table-03.png)

**表 3.** BitNet a4.8 在 QKV 状态采用不同位宽时的详细下游任务结果. 我们报告了所有模型的 zero-shot 准确率.

**低比特注意力.** [表 3](#table-03) 给出了 3B 和 7B 模型规模下, BitNet a4.8 采用低比特注意力的详细结果. 低比特注意力对高效长序列建模十分重要, 因为它能减少 KV 缓存的内存占用和 I/O, 并加速注意力计算. 实验采用 post-RoPE 量化. QKV 头使用 absmax 函数直接量化为无符号整数, 不需要任何校准数据集. 对 3-bit KV 量化, 我们将 bos token 的头保留为 4-bit, 因为其中包含更多离群特征. 如 [表 3](#table-03) 所示, 在 3B 和 7B 模型中, BitNet a4.8 使用 4-bit KV 或 QKV 头时, 准确率损失可以忽略. 此外, BitNet a4.8 的 KV 缓存可量化为 3-bit 整数, 平均准确率几乎不受影响.

<span id="section-3-2"></span>

### 3.2 消融研究

<span id="figure-04"></span>

![混合量化与稀疏化消融实验的损失曲线](./bitnet-a4-8/figure-04.png)

**图 4.** 混合量化与稀疏化的消融研究.

<span id="figure-05"></span>

![FFN 下投影量化与激活函数消融实验的损失曲线](./bitnet-a4-8/figure-05.png)

**图 5.** FFN 下投影输入采用不同量化方法或激活函数时的消融研究.

<span id="table-04"></span>

![注意力输出投影输入的 TopK 稀疏化消融实验](./bitnet-a4-8/table-04.png)

**表 4.** 注意力输出投影输入采用 TopK 稀疏化时的消融实验.

**混合架构.** [图 4](#figure-04) 给出了 700M BitNet a4.8 在全 INT4/FP4 量化以及混合量化与稀疏化下的训练损失曲线. 我们使用 RedPajama 数据集中的 25B token, 按第一阶段的调度方案训练这些模型. 对全 INT4 和 FP4 量化, 我们分别采用 absmean 和 MinMax 量化器. 此外, 对全 INT4 量化, 由于 FFN 下投影输入中存在幅值更大的离群值, 我们采用 $\beta = 2\mathrm{mean}(|X|)$ 的 absmean 量化器. 如 [图 4](#figure-04) 所示, 全 INT4 量化会导致训练发散. 而且, 混合架构在训练困惑度上明显优于全 FP4 架构.

**FFN 的下投影.** 我们比较了 1.3B BitNet a4.8 在 FFN 下投影采用不同量化方法或激活函数时的表现. 所有模型均使用 RedPajama 数据集中的 50B token, 按第一阶段的调度方案训练. 为确保公平比较, 其他激活值均保留为 8-bit. 对 INT8 量化, 我们采用 absmax 量化器; 对 FP4 量化, 则采用 MinMax 量化器. absmean 量化器的 $\beta$ 设为 $2\mathrm{mean}(|X|)$. [图 5](#figure-05) 展示了这些模型的训练损失曲线. 平方 ReLU 的训练困惑度略优于 Swish, 同时稀疏度更高. 此外, 对下投影输入应用 FP4 量化会显著降低性能, 而使用带 STE 的 INT4 激活值则会导致训练发散.

<span id="figure-06"></span>

![注意力和 FFN 输入采用 4-bit 量化器时的损失曲线](./bitnet-a4-8/figure-06.png)

**图 6.** 注意力和 FFN 输入采用 4-bit 量化器时的消融实验.

**注意力的输出投影.** [表 4](#table-04) 展示了 3B BitNet a4.8 在注意力输出投影输入使用和不使用 Top-K 稀疏化时的详细结果. 两个模型均采用相同的两阶段方案, 将激活值从 8-bit 训练到 4-bit. 稀疏化的 $K$ 设为 50%. 基线对输出投影的输入使用 INT8 absmax 量化器. 结果表明, TopK 稀疏化带来的困惑度和准确率损失可以忽略.

**4-bit 量化.** 我们给出了 3B BitNet a4.8 在注意力和 FFN 输入采用不同 4-bit 量化器时的损失曲线. 我们比较了使用 MinMax 量化器和 E2M1, E1M2 格式进行浮点量化, 以及使用 absmax, absmean 量化器进行整数量化时 BitNet a4.8 的性能. 如 [图 6](#figure-06) 所示, 采用 E2M1 格式的 FP4 和采用 absmean 量化器的 INT4 可获得略低的训练困惑度, 因为二者都适合处理小幅值激活值项.

<span id="table-05"></span>

![BitNet a4.8 与 BitNet b1.58 使用两万亿训练 token 时的结果](./bitnet-a4-8/table-05.png)

**表 5.** 参数量为 2B, 训练 token 数为 2T 时, BitNet a4.8 和 BitNet b1.58 的结果.

<span id="section-3-3"></span>

### 3.3 更多训练 token

已有研究 [Det22] 表明, 语言模型中激活离群值的出现频率与训练 token 数呈正相关. 为严格评估 BitNet a4.8 的可扩展性, 我们使用参数量为 2B, 训练 token 数为 2T 的模型配置开展了大量实验. 我们使用完全相同的训练数据和配置, 与 BitNet b1.58 进行了受控比较. [表 5](#table-05) 所示的实验结果表明, BitNet a4.8 在将激活值压缩至 4-bit 的同时, 性能仍与基线相当, 各项准确率指标的下降可以忽略. 这些结果有力说明, 本文所提方法在大规模场景下依然有效.

<span id="section-4"></span>

## 4 结论

本文提出 BitNet a4.8, 使 1-bit LLM 能够采用 4-bit 激活值. BitNet a4.8 使用一种新的混合量化与稀疏化架构, 减少激活值离群通道引入的量化误差. 具体而言, 我们对注意力层和 FFN 层的输入使用 4-bit 量化, 对中间状态则使用 8-bit 整数进行稀疏化. BitNet a4.8 通过继续训练从 W1.58A8 转为 W1.58A4. 实验结果表明, 在训练成本相同的情况下, BitNet a4.8 的结果与 BitNet b1.58 相当, 同时显著提高了推理效率.

## 致谢

感谢 Lei Wang 围绕推理效率所做的讨论.

<span id="section-5"></span>

## 5 超参数

<span id="table-06"></span>

![BitNet a4.8, BitNet b1.58 与 LLaMA LLM 的模型配置](./bitnet-a4-8/table-06.png)

**表 6.** BitNet a4.8, BitNet b1.58 和 LLaMA LLM 的模型配置.

<span id="table-07"></span>

![BitNet a4.8 与 LLaMA LLM 的训练超参数](./bitnet-a4-8/table-07.png)

**表 7.** BitNet a4.8 和 LLaMA LLM 训练所用的超参数.
