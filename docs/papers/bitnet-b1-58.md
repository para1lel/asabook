---
title: 'The Era of 1-bit LLMs'
createTime: 2026/09/08 13:02:00
permalink: /papers/bitnet-b1-58/
pageClass: paper-reading
---

> [Shuming Ma](https://shumingma.com/) [+author-note], [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Lingxiao Ma](https://xysmlx.github.io/), [Lei Wang](https://dblp.org/pid/181/2817-222), [Wenhui Wang](https://www.microsoft.com/en-us/research/people/wenwan/), [Shaohan Huang](https://buaahsh.github.io/), [Li Dong](https://dong.li/), [Ruiping Wang](https://www.jdl.link/user/rpwang/index.htm), [Jilong Xue](https://dblp.org/pid/06/10336), [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]. 论文于 2024 年 2 月 27 日首次提交至 arXiv; 当前版本为 v1; 仍在完善中. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764). <a href="/paper/bitnet-b1-58.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2402.17764). [TeX 源码](https://export.arxiv.org/e-print/2402.17764v1). 精确排版和完整参考文献以原始 PDF 为准.

[+author-note]: Shuming Ma 和 Hongyu Wang 贡献相同. Furu Wei 是通讯作者. Shuming Ma, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Jilong Xue 和 Furu Wei 就职于 Microsoft Research. Hongyu Wang 和 Ruiping Wang 就职于中国科学院大学. [GeneralAI](https://aka.ms/GeneralAI).

## 摘要

BitNet [Wan23] 等近期研究正推动大语言模型 (LLM) 进入 1-bit 时代. 本文提出一种名为 **BitNet b1.58** 的 1-bit LLM 变体, 其中 LLM 的每个参数 (或权重) 都是三值的, 即 $\{-1, 0, 1\}$. 在模型规模和训练 token 数相同的情况下, 它在困惑度和下游任务性能上均可比肩全精度 (即 FP16 或 BF16) Transformer LLM, 同时在延迟, 内存, 吞吐量和能耗方面显著降低成本. 更重要的是, 1.58-bit LLM 为训练兼具高性能与低成本的新一代 LLM 定义了新的缩放定律和方法. 此外, 它带来了新的计算范式, 为设计针对 1-bit LLM 优化的专用硬件提供了可能.

<span id="figure-01"></span>

![BitNet b1.58 与全精度 Transformer LLM 的 Pareto 对比及计算范式](./bitnet-b1-58/figure-01.png)

**图 1.** 1-bit LLM (如 BitNet b1.58) 在保持模型性能的同时, 为降低 LLM 的推理成本 (延迟, 吞吐量和能耗) 提供了 Pareto 解. BitNet b1.58 的新计算范式需要人们着手设计针对 1-bit LLM 优化的新硬件.

<span id="section-1"></span>

## 1 1-bit LLM 时代

近年来, AI 领域中的大语言模型 (LLM) 在规模和能力上快速增长. 这些模型在许多自然语言处理任务上表现突出, 但模型规模的增长给部署带来了困难, 高能耗所造成的环境与经济影响也引发了担忧.

应对这些问题的一种方法是使用训练后量化, 生成用于推理的低比特模型 [Xia23, Fra23, Che24b, Tse24]. 这项技术降低权重和激活值的精度, 从而显著减少 LLM 的内存与计算需求. 发展趋势是从 16 bit 转向更低的位数, 例如 4-bit 变体 [Fra23, Lin24]. 不过, 尽管训练后量化已广泛用于工业界的 LLM, 它仍非最优方案.

BitNet [Wan23] 等近期的 1-bit 模型架构为降低 LLM 成本同时保持其性能提供了一个有前景的方向. 常规 LLM 使用 16-bit 浮点值 (即 FP16 或 BF16), 而任何 LLM 的计算主体都是矩阵乘法. 因此, 主要计算成本来自浮点加法和乘法运算. 相比之下, BitNet 的矩阵乘法只涉及整数加法, 可将 LLM 的能耗成本降低若干数量级. 许多芯片的计算性能从根本上受功耗限制, 因而节省的能量也能转化为更快的计算速度.

除计算外, 推理期间将模型参数从 DRAM 传输到片上加速器的内存 (如 SRAM) 也可能代价高昂. 人们曾尝试扩大 SRAM 以提高吞吐量, 但这会带来远高于 DRAM 的成本. 与全精度模型相比, 1-bit LLM 无论从容量还是带宽看, 内存占用都低得多. 这样可以显著降低从 DRAM 加载权重的成本和时间, 使推理更快, 效率更高.

本文提出一种重要的 1-bit LLM 变体 **BitNet b1.58**, 其中每个参数都是三值的, 取值为 $\{-1, 0, 1\}$. 我们在原始 1-bit BitNet 中加入了取值 0, 因此在二进制系统中得到 1.58 bit. BitNet b1.58 保留了原始 1-bit BitNet 的所有优点, 包括新的计算范式: 矩阵乘法几乎不需要乘法运算, 因而可以进行高度优化. 此外, 它的能耗与原始 1-bit BitNet 相同; 与 FP16 LLM 基线相比, 其内存消耗, 吞吐量和延迟也高效得多. BitNet b1.58 还有另外两个优点. 第一, 模型权重中加入 0 后可以显式支持特征过滤, 因此建模能力更强, 并能显著改善 1-bit LLM 的性能. 第二, 实验表明, 在采用相同配置 (如模型规模, 训练 token 数等) 时, BitNet b1.58 从 3B 规模开始便能在困惑度和下游任务性能上比肩全精度 (即 FP16) 基线.

<span id="section-2"></span>

## 2 BitNet b1.58

BitNet b1.58 基于 BitNet 架构, 后者是一种用 *BitLinear* 替换 *nn.Linear* 的 Transformer. 它以 1.58-bit 权重和 8-bit 激活值从头训练. 与原始 BitNet 相比, 它引入了若干修改, 总结如下.

**量化函数.** 为了将权重限制为 -1, 0 或 +1, 我们采用 *absmean* 量化函数. 该函数先用权重矩阵的绝对值均值进行缩放, 再把每个值舍入为 $\{-1, 0, +1\}$ 中最接近的整数:

<span id="equation-01"></span>

$$
\widetilde{W}=\mathrm{RoundClip}\left(\frac{W}{\gamma+\epsilon},-1,1\right),
$$

<span id="equation-02"></span>

$$
\mathrm{RoundClip}(x,a,b)=\max(a,\min(b,\mathrm{round}(x))),
$$

<span id="equation-03"></span>

$$
\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|.
$$

激活值的量化函数沿用 BitNet 的实现, 不同之处是我们不会在非线性函数之前将激活值缩放到 $[0,Q_b]$ 范围. 相反, 每个 token 的激活值都缩放到 $[-Q_b,Q_b]$, 以消除零点量化. 这样在实现和系统级优化中更方便, 也更简单, 而实验中的性能影响可以忽略不计.

**类 LLaMA 组件.** LLaMA [Tou23, Tou23a] 架构事实上已成为开源 LLM 的主干. 为了融入开源社区, BitNet b1.58 采用了类 LLaMA 组件. 具体而言, 它使用 RMSNorm [Zha19], SwiGLU [Sha20] 和旋转位置嵌入 [Su24], 并移除所有 bias. 因此, 只需很少改动, BitNet b1.58 就能集成到常见的开源软件中 (如 Huggingface, vLLM [Kwo23] 和 llama.cpp [+llama-cpp]).

[+llama-cpp]: [llama.cpp](https://github.com/ggerganov/llama.cpp).

<span id="table-01"></span>

![BitNet b1.58 与 LLaMA LLM 的困惑度, 内存和延迟结果](./bitnet-b1-58/table-01.png)

**表 1.** BitNet b1.58 与 LLaMA LLM 的困惑度和成本.

<span id="table-02"></span>

![BitNet b1.58 与 LLaMA LLM 在七项下游任务上的零样本准确率](./bitnet-b1-58/table-02.png)

**表 2.** BitNet b1.58 与 LLaMA LLM 在下游任务上的零样本准确率.

<span id="section-3"></span>

## 3 结果

我们比较了不同规模的 BitNet b1.58 与自行复现的 FP16 LLaMA LLM. 为保证公平比较, 两类模型都在 RedPajama 数据集 [Tog23a] 上预训练 1000 亿个 token. 我们在一系列语言任务上评估零样本性能, 包括 ARC-Easy [Yad19], ARC-Challenge [Yad19], Hellaswag [Zel19], Winogrande [Sak19], PIQA [Bis20], OpenbookQA [Mih18b] 和 BoolQ [Cla19]. 我们还报告了 WikiText2 [Mer16] 和 C4 [Raf19] 数据集上的验证困惑度.

我们比较了 LLaMA LLM 和 BitNet b1.58 的运行时 GPU 内存与延迟. 结果使用 FasterTransformer [+fastertransformer] 代码库测得, 该代码库针对 GPU 设备上的 LLM 推理延迟做了充分优化. BitNet b1.58 还集成了 Ladder [Wan24e] 的 2-bit kernel. 我们报告每个输出 token 的耗时, 因为它是推理的主要成本.

[+fastertransformer]: [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

[表 1](#table-01) 汇总了 BitNet b1.58 和 LLaMA LLM 的困惑度与成本. 结果表明, 从 3B 模型规模开始, BitNet b1.58 的困惑度便能比肩全精度 LLaMA LLM, 同时速度提高 2.71 倍, GPU 内存用量减少 3.55 倍. 特别是, 3.9B 规模的 BitNet b1.58 速度提高 2.4 倍, 内存用量减少 3.32 倍, 性能却显著优于 3B 的 LLaMA LLM.

[表 2](#table-02) 给出了各下游任务的零样本准确率详情. 我们按照 *lm-evaluation-harness* [+lm-evaluation-harness] 的流程进行评估. 结果表明, 随着模型规模增大, BitNet b1.58 与 LLaMA LLM 之间的性能差距会缩小. 更重要的是, BitNet b1.58 从 3B 规模开始即可比肩全精度基线. 与困惑度的观察结果相似, 下游任务结果显示, 3.9B 的 BitNet b1.58 在内存和延迟成本更低的同时, 性能优于 3B 的 LLaMA LLM. 这说明 BitNet b1.58 相比当前最先进的 LLM 模型实现了 Pareto 改进.

[+lm-evaluation-harness]: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

**内存和延迟** 我们进一步将模型规模扩展到 7B, 13B 和 70B, 并评估其成本. [图 2](#figure-02) 展示了延迟和内存的变化趋势, 可以看到加速比随模型规模增大而上升. 其中, 70B 的 BitNet b1.58 比 LLaMA LLM 基线快 4.1 倍. 原因在于 *nn.Linear* 的时间成本会随模型规模增长. 内存消耗呈现相似趋势, 因为 embedding 仍保持全精度, 且它在更大模型中的内存占比更小. 延迟和内存均使用 2-bit kernel 测量, 因此仍有继续优化并降低成本的空间.

<span id="figure-02"></span>

![BitNet b1.58 在不同模型规模下的解码延迟和内存消耗](./bitnet-b1-58/figure-02.png)

**图 2.** BitNet b1.58 随模型规模变化的解码延迟 (左) 与内存消耗 (右).

**能耗** 我们还估算了 BitNet b1.58 和 LLaMA LLM 的算术运算能耗. 由于矩阵乘法在 LLM 成本中占比最高, 我们主要关注这部分计算. [图 3](#figure-03) 展示了能耗成本的组成. BitNet b1.58 的大部分运算是 INT8 加法, 而 LLaMA LLM 同时包含 FP16 加法和 FP16 乘法. 根据 [Hor14, Zha22g] 中的能耗模型, 在 7nm 芯片上, BitNet b1.58 可将矩阵乘法的算术运算能耗降低 71.4 倍. 我们还报告了模型处理 512 个 token 时的端到端能耗成本. 结果显示, 随着模型规模扩大, BitNet b1.58 相比 FP16 LLaMA LLM 基线在能耗方面越来越高效. 原因是 *nn.Linear* 所占比例随模型规模上升, 而其他组件在较大模型中的成本更小.

<span id="figure-03"></span>

![BitNet b1.58 与 LLaMA LLM 的算术运算和端到端能耗](./bitnet-b1-58/figure-03.png)

**图 3.** 在 7nm 工艺节点上, BitNet b1.58 与 LLaMA LLM 的能耗对比. 左图为算术运算能耗的组成, 右图为不同模型规模下的端到端能耗成本.

**吞吐量** 我们在两张 80GB A100 卡上比较参数量为 70B 的 BitNet b1.58 和 LLaMA LLM 的吞吐量, 并使用流水线并行 [Hua19c], 使 70B 的 LLaMA LLM 能够在这些设备上运行. 在序列长度为 512 的条件下, 我们逐步增加 batch size, 直至达到 GPU 内存上限. [表 3](#table-03) 显示, 70B 的 BitNet b1.58 所支持的 batch size 最高可达 LLaMA LLM 的 11 倍, 因而吞吐量提高了 8.9 倍.

<span id="table-03"></span>

![70B 参数下 BitNet b1.58 与 LLaMA LLM 的最大 batch size 和吞吐量](./bitnet-b1-58/table-03.png)

**表 3.** 70B 的 BitNet b1.58 与 70B 的 LLaMA LLM 的吞吐量对比.

**BitNet b1.58 正在为模型性能和推理成本带来新的缩放定律**. 参照 [图 2](#figure-02) 和 [图 3](#figure-03) 的结果, 不同规模的 1.58-bit 与 16-bit 模型之间可得到以下对应关系.

- 就延迟, 内存用量和能耗而言, 13B BitNet b1.58 比 3B FP16 LLM 更高效.
- 就延迟, 内存用量和能耗而言, 30B BitNet b1.58 比 7B FP16 LLM 更高效.
- 就延迟, 内存用量和能耗而言, 70B BitNet b1.58 比 13B FP16 LLM 更高效.

**使用 2T Token 训练** 训练 token 数是影响 LLM 的一个重要因素. 为了检验 BitNet b1.58 随 token 数扩展的能力, 我们按照当前最先进的开源 3B 模型 StableLM-3B [Tow23] 的数据配方, 使用 2T token 训练了一个 BitNet b1.58 模型. 两个模型都在由 Winogrande [Sak19], PIQA [Bis20], SciQ [Wel17], LAMBADA [Pap16] 和 ARC-easy [Yad19] 组成的基准上进行评估. 我们在 [表 4](#table-04) 中报告零样本准确率. 对于同时采用准确率和归一化准确率衡量的任务, 我们取两者的平均值. StableLM 3b 使用 2T token 训练的结果直接取自其技术报告. 结果表明, BitNet b1.58 在所有下游任务上的性能均更好, 说明 1.58-bit LLM 也有很强的泛化能力.

<span id="table-04"></span>

![使用 2T token 训练后 BitNet b1.58 与 StableLM-3B 的零样本结果](./bitnet-b1-58/table-04.png)

**表 4.** 使用 2T token 训练的 BitNet b1.58 与 StableLM-3B 对比.

<span id="section-4"></span>

## 4 讨论与未来工作

**1-bit 混合专家 (MoE) LLM** 混合专家 (MoE) 已被证明是一种低成本的 LLM 方案. 它能显著减少计算 FLOPs, 但高内存消耗和芯片间通信开销限制了部署与应用. 1.58-bit LLM 可以解决这些问题. 首先, 更小的内存占用能够减少部署 MoE 模型所需的设备数量. 同时, 它也能显著降低通过网络传输激活值的开销. 如果整个模型都能放入单颗芯片, 最终将不再产生这项开销.

**LLM 对长序列的原生支持** 在 LLM 时代, 处理长序列已成为一项重要需求. 长序列推理的一个主要难题是 KV cache 引入的内存消耗. BitNet b1.58 将激活值从 16 bit 降至 8 bit, 因此在相同资源下可将上下文长度翻倍, 向原生支持长序列迈出了一大步. 对于 1.58-bit LLM, 还可以无损压缩至 4 bit 甚至更低, 我们将此留作未来工作.

**边缘和移动设备上的 LLM** 1.58-bit LLM 有望大幅提升语言模型在边缘和移动设备上的性能. 这类设备的内存和算力往往有限, 会限制 LLM 的性能和规模. 1.58-bit LLM 降低了内存和能耗, 因而可以部署在这些设备上, 支持许多以往无法实现的应用. 这可以提升边缘与移动设备的能力, 并支持新的 LLM 应用. 此外, 1.58-bit LLM 对 CPU 设备更加友好, 而 CPU 正是边缘和移动设备使用的主要处理器. 这意味着 BitNet b1.58 可以在这些设备上高效运行, 进一步改善其性能和能力.

**面向 1-bit LLM 的新硬件** Groq [+groq] 等近期工作表明, 为 LLM 构建专用硬件 (如 LPU) 有良好效果和巨大潜力. 再进一步, 鉴于 BitNet [Wan23] 带来的新计算范式, 我们设想并呼吁着手设计专门针对 1-bit LLM 优化的新硬件和系统.

[+groq]: [Groq](https://groq.com/).
