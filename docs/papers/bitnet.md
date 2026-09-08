---
title: 'BitNet: Scaling 1-bit Transformers'
createTime: 2026/09/08 15:00:00
permalink: /papers/bitnet/
pageClass: paper-reading
---

> [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Shuming Ma](https://shumingma.com/) [+author-note], [Li Dong](https://dong.li/), [Shaohan Huang](https://buaahsh.github.io/), [Huaijie Wang](https://dblp.dagstuhl.de/pid/346/1061.html), [Lingxiao Ma](https://xysmlx.github.io/), [Fan Yang](https://fanyangcs.github.io/), [Ruiping Wang](https://www.jdl.link/user/rpwang/index.htm), [Yi Wu](https://jxwuyi.weebly.com/) 和 [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]. 论文于 2023 年 10 月 17 日首次提交至 arXiv; 当前版本为 v1; 仍在完善中. [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453). <a href="/paper/bitnet.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2310.11453). [TeX 源码](https://export.arxiv.org/e-print/2310.11453v1). 精确排版和完整参考文献以原始 PDF 为准.

[+author-note]: Hongyu Wang 和 Shuming Ma 对本文贡献同等. Furu Wei 是通讯作者. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Lingxiao Ma, Fan Yang 和 Furu Wei 就职于 Microsoft Research. Hongyu Wang 和 Ruiping Wang 就职于中国科学院大学. Huaijie Wang 和 Yi Wu 就职于清华大学. [GeneralAI](https://aka.ms/GeneralAI).

## 摘要

大语言模型规模不断增长, 给部署带来了困难, 高能耗造成的环境影响也引发了担忧. 本文提出 BitNet, 一种面向大语言模型, 可扩展且训练稳定的 1-bit Transformer 架构. 具体而言, 我们提出 `BitLinear`, 用它直接替换 `nn.Linear` 层, 从头训练 1-bit 权重. 语言建模实验表明, 与当前先进的 8-bit 量化方法和 FP16 Transformer 基线相比, BitNet 在保持有竞争力的性能时, 大幅减少了内存占用和能耗. 此外, BitNet 呈现出与全精度 Transformer 相似的缩放定律, 表明它可能有效扩展到更大的语言模型, 同时保留效率和性能优势.

<span id="figure-01"></span>

![BitNet 的性能, 后训练量化对比, 能耗降幅和缩放曲线](./bitnet/figure-01.png)

**图 1.** BitNet 从头训练 1-bit Transformer, 以较低能耗取得有竞争力的结果. BitNet 显著优于当前先进的量化方法. 随着模型规模增大, 在达到与 FP16 训练模型相当性能的同时, 成本降幅也会更加明显.

> 我认为人类智能没有什么独一无二之处. 大脑中形成知觉和情感的所有神经元都以二元方式运作.
>
> — William Henry Gates III

<span id="section-1"></span>

## 1 引言

大语言模型 [Bro20, Ope23, Cho22, Ani23, Tou23, Tou23a] 的快速发展显著改善了多种任务的表现. 但由于推理成本和能耗很高, 托管大语言模型十分昂贵. 随着模型规模增长, 访问和处理模型参数所需的内存带宽成为主要瓶颈, 限制了整体推理性能. 在分布式系统或多设备平台上部署模型时, 设备间通信开销还会显著影响推理延迟和能耗. 模型量化 [Fra23, Che24b, Xia23] 已成为一种可行的解决方案, 它能在保持有竞争力性能的同时, 显著降低大规模模型的内存占用和计算成本.

现有的大语言模型量化方法大多采用训练后量化. 这类方法简单易用, 因为不需要修改训练流程或重新训练模型. 但随着精度降低, 它会造成更严重的准确率损失, 因为模型在训练时并未针对量化表示进行优化.

量化感知训练是深度神经网络量化的另一条路线. 与训练后量化相比, 它通常能取得更高的准确率, 因为模型从训练开始便会适应较低精度. 它还允许模型继续训练或微调, 这对大语言模型十分重要. 量化感知训练的难点主要在优化方面, 即精度越低, 模型越难收敛. 此外, 量化感知训练是否遵循神经语言模型的缩放定律仍不清楚.

本文关注应用于大语言模型的二值化 (即 1-bit), 这是量化的极端情形. 以往关于二值神经网络 [Ras16, Bul19a] 的研究主要围绕卷积神经网络展开. 最近也有一些二值 Transformer 研究. 但这些研究聚焦机器翻译或 BERT 预训练, 与大语言模型相去甚远. 例如, 机器翻译使用编码器-解码器架构, BERT 预训练使用双向编码器, 大语言模型则使用单向解码器. 而且, 大语言模型通常会扩展到远大于 BERT 和机器翻译模型的规模.

据我们所知, 本文首次研究面向 1-bit 大语言模型的量化感知训练. 我们提出 BitNet, 一种面向大语言模型的 1-bit Transformer 架构, 目标是在内存和计算两方面高效扩展. BitNet 使用低精度二值权重和量化激活值, 同时在训练期间为优化器状态和梯度保留高精度. 我们的方法兼顾可扩展性与稳定性, 能够高效处理大语言模型. BitNet 的实现很简单, 只需替换 Transformer 中的线性投影 (即 PyTorch 的 *nn.Linear*). 它还能与 PagedAttention [Kwo23], FlashAttention [Dao22, Dao24a] 和推测解码 [Lev23] 等大语言模型加速方法配合使用.

我们在一系列语言建模基准上评估 BitNet, 并将其与当前先进的量化方法和 FP16 Transformer 对比. 实验结果表明, BitNet 在困惑度和下游任务准确率上都取得了有竞争力的性能. 更重要的是, BitNet 的内存占用和能耗显著低于基线. 我们还发现, BitNet 遵循与全精度 Transformer 相似的缩放定律, 说明它可以有效扩展到更大的语言模型, 并可能同时获得性能和效率优势.

<span id="section-2"></span>

## 2 BitNet

如[图 2](#figure-02) 所示, BitNet 沿用 Transformer 的整体布局, 堆叠自注意力和前馈网络模块. 与普通 Transformer 相比, BitNet 用 `BitLinear` ([公式 11](#equation-11)) 取代常规矩阵乘法, 其中使用二值化 (即 1-bit) 模型权重. 其他组件保持高精度, 在我们的实验中为 8-bit. 原因如下. 第一, 残差连接和层归一化在大语言模型中的计算成本可以忽略. 第二, 随着模型增大, QKV 变换的计算成本远小于参数投影. 第三, 输入/输出嵌入需要保留精度, 因为语言模型必须用高精度概率进行采样.

<span id="figure-02"></span>

![BitLinear 的计算流程和 BitNet 架构](./bitnet/figure-02.png)

**图 2.** (a) `BitLinear` 的计算流程. (b) BitNet 的架构, 由堆叠的注意力模块和 FFN 组成, 其中矩阵乘法用 `BitLinear` 实现.

<span id="section-2-1"></span>

### 2.1 BitLinear

我们首先用符号函数将权重二值化为 $+1$ 或 $-1$. 按照 [Liu22] 的方法, 我们在二值化前将权重中心化为零均值, 以提高有限数值范围内的容量. 二值化后使用缩放因子 $\beta$, 以减小实值权重和二值权重之间的 $l2$ 误差. 权重 $W \in \mathcal{R}^{n \times m}$ 的二值化可写为:

<span id="equation-01"></span>

$$
\widetilde{W} = \mathrm{Sign}(W - \alpha),
$$

<span id="equation-02"></span>

$$
\mathrm{Sign}(W_{ij}) = \left\{
\begin{aligned}
&+1, \quad &&\text{if } W_{ij} > 0, \\
&-1, \quad &&\text{if } W_{ij} \leq 0,
\end{aligned}
\right.
$$

<span id="equation-03"></span>

$$
\alpha = \frac{1}{nm}\sum_{ij} W_{ij}
$$

我们进一步将激活值量化到 $b$-bit 精度. 按照 [Det22] 的方法, 我们使用 absmax 量化: 将激活值乘以 $Q_b$ 并除以输入矩阵绝对值的最大值, 从而缩放到 $[-Q_b, Q_b]$ 范围内, 其中 $Q_b=2^{b-1}$:

<span id="equation-04"></span>

$$
\widetilde{x} = \mathrm{Quant}(x) = \mathrm{Clip}\left(x \times \frac{Q_b}{\gamma}, -Q_b+\epsilon, Q_b-\epsilon\right),
$$

<span id="equation-05"></span>

$$
\mathrm{Clip}(x, a, b) = \max(a, \min(b, x)), \quad \gamma = \|x\|_{\infty},
$$

其中 $\epsilon$ 是一个较小的浮点数, 用于防止裁剪时发生溢出.

对于非线性函数 (如 ReLU) 之前的激活值, 我们减去输入的最小值, 使所有值均为非负数, 再将其缩放到 $[0, Q_b]$ 范围:

<span id="equation-06"></span>

$$
\widetilde{x} = \mathrm{Quant}(x) = \mathrm{Clip}\left((x-\eta) \times \frac{Q_b}{\gamma}, \epsilon, Q_b-\epsilon\right), \quad \eta = \min_{ij} x_{ij}.
$$

本文将激活值量化到 8-bit, 更低精度留待未来研究. 为兼顾稳定性和效率, 训练时按张量量化, 推理时则按 token 量化.

使用上述量化公式, 矩阵乘法可以写为:

<span id="equation-07"></span>

$$
y = \widetilde{W} \widetilde{x}
$$

我们假设 $W$ 和 $x$ 中的元素相互独立且服从相同分布, 同时 $W$ 与 $x$ 彼此独立. 于是输出 $y$ 的方差估计为:

<span id="equation-08"></span>

$$
\mathrm{Var}(y) = n \mathrm{Var}(\widetilde{w} \widetilde{x})
$$

<span id="equation-09"></span>

$$
=n E[\widetilde{w}^2] E[\widetilde{x}^2]
$$

<span id="equation-10"></span>

$$
=n \beta^2 E[\widetilde{x}^2] \approx E[\widetilde{x}^2]
$$

采用标准初始化方法 (如 Kaiming 初始化或 Xavier 初始化) 时, 全精度计算的输出方差 $\mathrm{Var}(y)$ 量级为 $1$, 这对训练稳定性很有帮助. 为在量化后保留方差, 我们在激活值量化之前引入 LayerNorm [Ba16]. 此时输出 $y$ 的方差可估计为 $\mathrm{Var}(y) \approx E[\mathrm{LN}(\widetilde{x})^2] = 1$, 与全精度版本的 $\mathrm{Var}(y)$ 具有相同量级. 在 Transformer 中, 它与 `SubLN` [Wan22l] 的实现完全相同. 结合 `SubLN` 和上述量化方法, 即可得到 `BitLinear`, 其形式为:

<span id="equation-11"></span>

$$
y = \widetilde{W} \widetilde{x} = \widetilde{W}\,\mathrm{Quant}(\mathrm{LN}(x)) \times \frac{\beta\gamma}{Q_b}
$$

<span id="equation-12"></span>

$$
\mathrm{LN}(x) = \frac{x-E(x)}{\sqrt{\mathrm{Var}(x)+\epsilon}}, \quad \beta = \frac{1}{nm}\|W\|_1
$$

[图 2](#figure-02) 给出了 `BitLinear` 的计算流程. 执行 SubLN 后, 用 absmax 函数量化激活值. 矩阵乘法在 1-bit 权重和量化激活值之间进行. 最后用 $\{\beta, \gamma\}$ 重新缩放输出激活值, 将其反量化到原始精度.

**结合分组量化和归一化的模型并行.** 扩展大语言模型的一项重要技术是模型并行 [Sho19], 它把矩阵乘法划分到多个设备上. 现有模型并行方法要求张量在划分维度上相互独立. 然而, 参数 $\alpha$, $\beta$, $\gamma$ 和 $\eta$ 都由完整张量计算, 破坏了这一独立性要求. 一种解决办法是为每个参数引入一次 *all-reduce* 操作. 但即使每个参数的通信量很小, 同步次数也会随模型层数增加, 从而显著拖慢前向传播. `SubLN` 同样存在这个问题, 因为均值和方差都需要跨划分维度估计.

为此, 我们提出一种简单的方法, 让模型并行更加高效. 我们把权重和激活值划分为若干组, 再分别估计各组参数. 这样便可在本地计算参数, 无需额外通信. 这种方法称为分组量化, 其形式如下:

对于权重矩阵 $W \in \mathcal{R}^{n \times m}$, 我们沿划分维度将其分成 $G$ 组, 每组大小为 $\frac{n}{G} \times m$. 随后分别估计各组的参数:

<span id="equation-13"></span>

$$
\alpha_g = \frac{G}{nm}\sum_{ij} W_{ij}^{(g)}, \quad \beta_g = \frac{G}{nm}\|W^{(g)}\|_1,
$$

其中 $W^{(g)}$ 表示权重矩阵的第 $g$ 组. 对激活值也可以采用相同方法: 将输入矩阵 $x \in \mathcal{R}^{n \times m}$ 分成 $G$ 组, 并计算各组参数:

<span id="equation-14"></span>

$$
\gamma_g = \|x^{(g)}\|_{\infty}, \quad \eta_g = \min_{ij} x_{ij}^{(g)}
$$

对于 LN, 可以使用分组归一化 [Wu20a], 分别计算各组的均值和方差:

<span id="equation-15"></span>

$$
\mathrm{LN}(x^{(g)}) = \frac{x^{(g)}-E(x^{(g)})}{\sqrt{\mathrm{Var}(x^{(g)})+\epsilon}}
$$

这样即可用分组量化和归一化高效实现模型并行, 无需额外通信, 并能扩展到大语言模型.

<span id="section-2-2"></span>

### 2.2 模型训练

**直通估计器.** 为训练 1-bit 模型, 我们使用直通估计器 (STE) [Ben13] 近似反向传播中的梯度. 这种方法在反向传播时绕过 Sign ([公式 2](#equation-02)) 和 Clip ([公式 5](#equation-05)) 等不可微函数. STE 让梯度可以不受这些不可微函数影响而流经网络, 从而使量化模型能够训练.

**混合精度训练.** 权重和激活值会被量化到低精度, 但梯度和优化器状态以高精度保存, 以保证训练稳定性和准确率. 按照以往工作 [Liu21f] 的方法, 我们为可学习参数保留高精度的潜在权重, 用于累积参数更新. 潜在权重在前向传播期间即时二值化, 推理过程不会使用它们.

**大学习率.** 优化中的一个难点是, 潜在权重的小幅更新往往不会改变 1-bit 权重. 这会导致根据 1-bit 权重估计的梯度和更新产生偏差. 在训练初期, 模型理应尽快收敛, 这个问题会更加严重. 为解决这一问题, 我们探索了多种方法, 最后发现提高学习率是加速优化最简单也最有效的办法. 实验表明, 大学习率能改善 BitNet 的收敛, 而 FP16 Transformer 在相同学习率下会在训练初期发散. 详见[第 3 节](#section-3).

<span id="table-01"></span>

![三种模型规模下 BitNet 和 Transformer 的能耗](./bitnet/table-01.png)

**表 1.** 不同模型规模下 BitNet 和 Transformer 的能耗. 结果使用 512 的输入长度.

<span id="section-2-3"></span>

### 2.3 计算效率

我们从算术运算能耗和内存占用两方面估算 BitNet 的计算效率. 矩阵乘法占大语言模型成本的主要部分, 因此我们重点计算这部分开销.

**算术运算能耗.** 根据 [Hor14, Zha22g] 的能耗模型, 不同算术运算的能耗可以估算如下:

<span id="table-02"></span>

![45nm 和 7nm 工艺下 FP32, FP16 与 INT8 的 ADD 和 MUL 能耗](./bitnet/table-02.png)

**表 2.** 45nm 和 7nm 工艺节点下, 不同 bit 表示的 ADD 和 MUL 能耗 [Hor14, Zha22g].

在普通 Transformer 中, 对维度分别为 $m \times n$ 和 $n \times p$ 的矩阵进行乘法时, 能耗可计算为:

<span id="equation-16"></span>

$$
E_{\mathrm{add}} = m \times (n-1) \times p \times \hat{E}_{\mathrm{add}}
$$

<span id="equation-17"></span>

$$
E_{\mathrm{mul}} = m \times n \times p \times \hat{E}_{\mathrm{mul}}
$$

对于 BitNet, 由于权重为 1-bit, 矩阵乘法的能耗主要来自加法. 乘法只用于用标量 $\beta$ 和 $\frac{\gamma}{Q_b}$ 缩放输出, 因而乘法能耗可计算为:

<span id="equation-18"></span>

$$
E_{\mathrm{mul}} = (m \times p + m \times n) \times \hat{E}_{\mathrm{mul}}
$$

这远小于 Transformer 的乘法能耗. [表 1](#table-01) 给出了 W1A8 BitNet 相比全精度 (32-32) 和半精度 (16-16) Transformer 的节能幅度. 可以看到, BitNet 显著降低了能耗, 尤其是矩阵乘法能耗的主要部分: 乘法运算.

<span id="section-3"></span>

## 3 与 FP16 Transformer 的比较

<span id="section-3-1"></span>

### 3.1 设置

我们训练了一系列不同规模的 BitNet 自回归语言模型, 参数量从 125M 到 30B. 模型在英文语料库上训练, 其中包含 Pile 数据集, Common Crawl 快照, RealNews 和 CC-Stories 数据集. 我们使用 Sentencpiece tokenizer 预处理数据, 词表大小为 16K. 除 BitNet 外, 我们还用相同的数据集和设置训练 Transformer 基线, 以便公平比较. 更多细节见附录.

<span id="section-3-2"></span>

### 3.2 推理最优缩放定律

使用普通 Transformer 架构的神经语言模型已被证明能够按规律缩放 [Kap20]. 损失随训练计算量呈幂律变化. 因此, 我们可以确定计算预算的最优分配, 也能用小模型预测大语言模型的性能.

为研究二值 Transformer 的缩放定律, 我们首先绘制 BitNet 和 FP16 Transformer 基线相对于参数量的缩放曲线. 我们固定训练 token 数量, 改变模型规模. [图 3](#figure-03) 表明, BitNet 的损失缩放与 FP16 Transformer 相似, 同样遵循幂律. 随后, 我们用含不可约损失项的函数拟合缩放定律:

<span id="equation-19"></span>

$$
L(N)=aN^b+c
$$

为检验该缩放定律能否准确预测损失, 我们用 125M 到 6.7B 的模型拟合幂律参数, 再据此预测 13B 和 30B 模型的损失. 结果显示, 拟合出的缩放定律能准确预测 BitNet 的损失. 此外, BitNet 与 FP16 Transformer 的差距会随模型增大而缩小.

上述幂律描述了 BitNet 的缩放趋势, 但不能准确刻画损失与实际计算量之间的关系. 以往工作 [Kap20, Hen20a, Hof22] 通过计算 FLOPs 估算计算量. 这种方法不适用于以整数计算为主要成本的 1-bit 模型. 而且, 它衡量的主要是训练计算量, 并非推理成本. 为更好地理解神经语言模型的缩放效率, 我们提出推理最优缩放定律. 它预测损失与能耗之间的关系. 训练只进行一次, 而推理能耗会随模型使用量增长, 因此我们关注推理能耗. 能耗按照[第 2.3 节](#section-2-3) 中的方法估算. [图 3](#figure-03) 展示了相对于 7nm 工艺节点推理能耗的缩放曲线. 结果证明 BitNet 的缩放效率高得多. 在固定计算预算下, BitNet 能取得显著更低的损失. 同时, 达到与 FP16 模型相同性能所需的推理成本也低得多.

<span id="figure-03"></span>

![BitNet 和 FP16 Transformer 相对于能耗和模型规模的缩放曲线](./bitnet/figure-03.png)

**图 3.** BitNet 和 FP16 Transformer 的缩放曲线.

<span id="section-3-3"></span>

### 3.3 下游任务结果

除了损失, 我们还关注 BitNet 在缩放过程中的能力. 与损失相比, 神经语言模型会出现涌现现象, 因而更难预测能力. 为用可解释的指标评估能力, 我们在 Hellaswag [Zel19], Winogrande [Sak20], Winograd [Lev12] 和 Storycloze [Mos16] 四项下游任务上测试 0-shot 和 4-shot 结果. [图 4](#figure-04) 给出了不同规模 BitNet 与 FP16 Transformer 的平均结果. 与损失缩放曲线相似, 下游任务性能会随计算预算增长而提高. 无论是 zero-shot 还是 few-shot 性能, BitNet 的能力缩放效率都远高于 FP16 Transformer 基线.

<span id="figure-04"></span>

![BitNet 和 FP16 Transformer 的 zero-shot 与 few-shot 下游性能](./bitnet/figure-04.png)

**图 4.** BitNet 和 FP16 Transformer 相对于推理成本的 zero-shot (左) 与 few-shot (右) 性能.

<span id="section-3-4"></span>

### 3.4 稳定性测试

训练低 bit Transformer 的主要难点是优化稳定性. 因此, 我们用不同的峰值学习率训练一系列模型, 对 BitNet 和 FP16 基线进行稳定性测试. [图 5a](#figure-05) 给出了稳定性测试结果. BitNet 能在大学习率下收敛, FP16 Transformer 则不能, 说明 BitNet 的训练更加稳定. 这一优化优势允许使用更大的学习率. [图 5b](#figure-05) 表明, 提高学习率可以改善 BitNet 的收敛, 降低 PPL.

<span id="figure-05"></span>

![BitNet 和 FP16 Transformer 的训练稳定性及不同学习率下的收敛情况](./bitnet/figure-05.png)

**图 5.** 在相同学习率下, BitNet 比 FP16 Transformer 更稳定 (左). 训练稳定性使 BitNet 能够使用更大的学习率, 从而实现更好的收敛 (右).

<span id="section-4"></span>

## 4 与训练后量化的比较

<span id="section-4-1"></span>

### 4.1 设置

我们按照[第 3.1 节](#section-3-1) 的设置训练 BitNet. 对比方法包括 Absmax [Det22], SmoothQuant [Xia23], GPTQ [Fra23] 和 QuIP [Che24b] 等当前先进的量化方法. 这些方法都在 FP16 Transformer 模型上进行训练后量化, 该模型采用与 BitNet 相同的训练设置和数据. 其中, Absmax 和 SmoothQuant 同时量化权重与激活值, GPTQ 和 QuIP 则只降低权重精度. 我们在多种量化精度下应用这些方法. 对于仅量化权重的方法 (即 GPTQ 和 QuIP), 我们实验了 W4A16 和 W2A16. 对于同时量化权重和激活值的方法 (即 Absmax 和 SmoothQuant), 我们将 FP16 Transformer 量化为 W8A8, W4A4 和 W1A8. 本文的 BitNet 实现采用二值权重和 8-bit 激活值 (W1A8), bit 数不高于各基线.

<span id="section-4-2"></span>

### 4.2 结果

[表 3](#table-03) 详细比较了我们提出的 BitNet 与多种基线方法在 Winogrande, Winograd, Storycloze 和 Hellaswag 四个基准数据集上的 zero-shot 性能. 为公平比较, 所有模型的参数量均为 6.7B. 各方法的权重精度从 16 bit 降至 1 bit. 除下游任务的 zero-shot 准确率外, 评估指标还包括验证集上的语言模型困惑度, 从而完整反映各方法的性能.

结果表明, BitNet 与基线方法相比具有竞争力, 尤其在较低 bit 下. BitNet 的 zero-shot 得分与 8-bit 模型相当, 推理成本却低得多. 对于 4-bit 模型, 仅量化权重的方法优于同时量化权重和激活值的方法, 主要原因是激活值更难量化. 作为 1-bit 模型, BitNet 的结果显著优于同时量化权重和激活值的方法, 也优于仅量化权重的方法. 在更低 bit 下, BitNet 的得分始终高于所有基线. 这证明量化感知训练优于训练后量化. [图 6](#figure-06) 汇总了模型规模从 1.3B 扩展到 6.7B 时, 本文方法和各基线的 zero-shot 与 few-shot 准确率. 结果表明, 这一优势在不同规模下都能保持.

<span id="figure-06"></span>

![BitNet 和训练后量化基线的 zero-shot 与 few-shot 结果](./bitnet/figure-06.png)

**图 6.** BitNet 与训练后量化基线在下游任务上的 zero-shot (左) 和 few-shot (右) 结果.

<span id="table-03"></span>

![BitNet 和训练后量化基线的 zero-shot 结果](./bitnet/table-03.png)

**表 3.** BitNet 和基线的 zero-shot 结果 (`PTQ`: 训练后量化, `WGe`: Winogrande, `WG`: Winograd, `SC`: Storycloze, `HS`: Hellaswag 数据集).

<span id="section-5"></span>

## 5 消融研究

[表 4](#table-04) 给出了我们的方法与几种替代方案的消融对比. 我们分别消融激活值量化方法的选择和稳定模型训练的技术. BitNet 使用 absmax 量化激活值, 并用 `SubLN` 保证训练稳定性. 另一种量化方案是 [Liu22] 的 elastic 函数, 它通过可学习参数动态调整缩放比例. 实验中, absmax 的性能优于 elastic 函数. absmax 还能让训练更稳定, 因而 BitNet 可以使用更大的学习率. 我们还将 `SubLN` 与 Pre-LN 及 BMT 架构 [Zha23s] 对比. Pre-LN 是 GPT 预训练的默认架构, BMT 则已被证明能提高二值模型的训练稳定性. 实验显示, `SubLN` 的表现优于 Pre-LN 和 BMT. 因此, BitNet 采用 absmax 和 `SubLN`.

<span id="table-04"></span>

![BitNet 量化和归一化选择的消融结果](./bitnet/table-04.png)

**表 4.** BitNet 的消融结果 (`WGe`: Winogrande, `WG`: Winograd, `SC`: Storycloze, `HS`: Hellaswag 数据集). Elastic 是 [Liu22] 提出的激活值量化方法, BMT 是 [Zha23s] 提出的低 bit 模型稳定训练架构.

<span id="section-6"></span>

## 6 结论与未来工作

本文提出 BitNet, 一种面向大语言模型的新型 1-bit Transformer 架构. 我们的方法兼顾可扩展性与稳定性, 能够高效处理大语言模型. 实验结果表明, BitNet 在困惑度和下游任务性能上均具有竞争力, 同时内存占用和能耗显著低于基线. 此外, BitNet 遵循与全精度 Transformer 相似的缩放定律, 说明它可以有效扩展到更大的语言模型, 并可能同时获得性能和效率优势. 今后, 我们计划从模型规模和训练步数两方面扩展 BitNet. 我们也希望把 BitNet 应用于其他架构 (如 RetNet [Sun23a]) 以训练大语言模型.

<span id="section-7"></span>

## 7 超参数

<span id="table-05"></span>

![BitNet 缩放实验的模型配置](./bitnet/table-05.png)

**表 5.** BitNet 缩放实验的模型配置.

<span id="table-06"></span>

![BitNet 和 FP16 Transformer 缩放实验的超参数](./bitnet/table-06.png)

**表 6.** BitNet 和 FP16 Transformer 缩放实验的超参数. 对于 13B 和 30B 模型, 为保证训练稳定性, 权重衰减设为 0.05.

<span id="table-07"></span>

![BitNet 和 FP16 Transformer 稳定性测试的超参数](./bitnet/table-07.png)

**表 7.** BitNet 和 FP16 Transformer 稳定性测试的超参数.

<span id="table-08"></span>

![BitNet 消融实验的超参数](./bitnet/table-08.png)

**表 8.** BitNet 消融实验的超参数.
