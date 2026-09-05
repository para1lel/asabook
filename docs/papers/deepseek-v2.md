---
title: 'DeepSeek-V2'
createTime: 2026/09/05 18:30:00
permalink: /papers/deepseek-v2/
pageClass: paper-reading
---

> [DeepSeek-AI](https://www.deepseek.com/). 于 2024 年 5 月 7 日首次提交至 arXiv; 当前版本为 v5 (2024 年 6 月 19 日). [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434). [原始 PDF](/paper/deepseek-v2.pdf). [DOI](https://doi.org/10.48550/arXiv.2405.04434). [TeX 源码](https://export.arxiv.org/e-print/2405.04434v5). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

我们提出 DeepSeek-V2, 一个性能强劲的混合专家 (MoE) 语言模型, 其特点是训练经济且推理高效. 它总计包含 236B 参数, 每个 token 激活其中的 21B, 并支持 128K token 的上下文长度. DeepSeek-V2 采用了包括多头潜在注意力 (MLA) 和 DeepSeekMoE 在内的创新架构. MLA 通过将键值 (KV) cache 显著压缩为潜在向量来保障高效推理, 而 DeepSeekMoE 则借助稀疏计算实现以经济成本训练出强大的模型. 与 DeepSeek 67B 相比, DeepSeek-V2 取得了显著更强的性能, 同时节省了 42.5% 的训练成本, 将 KV cache 减少了 93.3%, 并将最大生成吞吐量提升至 5.76 倍. 我们在一个高质量且多来源、包含 8.1T token 的语料上对 DeepSeek-V2 进行预训练, 并进一步执行监督微调 (SFT) 和强化学习 (RL) 以充分释放其潜力. 评测结果表明, 即使在激活参数仅有 21B 的情况下, DeepSeek-V2 及其对话版本依然在开源模型中取得顶尖性能. 模型检查点见 https://github.com/deepseek-ai/DeepSeek-V2.

<span id="figure-01"></span>

![DeepSeek-V2 图 1](./deepseek-v2/figure-01.png)

**图 1.** (a) 不同开源模型间 MMLU 准确率与激活参数的关系. (b) DeepSeek 67B (Dense) 与 DeepSeek-V2 的训练成本和推理效率.

<span id="section-1"></span>

## 1 引言

过去几年里, 大语言模型 (LLM) [Ope22, Ope23, Ant23, Goo23] 经历了快速发展, 让我们得以一窥通用人工智能 (AGI) 的曙光. 总体而言, LLM 的智能往往随着参数数量的增加而提升, 使其能够在各项任务上展现涌现能力 [Wei22d]. 然而, 这种提升的代价是训练时需要更大的计算资源, 以及推理吞吐量可能下降. 这些约束带来了显著挑战, 阻碍了 LLM 的广泛采用与利用. 为了解决这一问题, 我们推出了 DeepSeek-V2, 一个性能强劲的开源混合专家 (MoE) 语言模型, 通过创新的 Transformer 架构实现训练经济与推理高效. 它总计配备 236B 参数, 每个 token 激活其中的 21B, 并支持 128K token 的上下文长度.

我们借助所提出的 **多头潜在注意力 (MLA)** 与 **DeepSeekMoE**, 优化了 Transformer 框架 [Vas17d] 中的注意力模块和前馈网络 (FFN).

1. 在注意力机制方面, 多头注意力 (MHA) [Vas17d] 的键值 (KV) cache 对 LLM 的推理效率构成了重大障碍. 为解决这一问题, 人们探索了多种方法, 包括分组查询注意力 (GQA) [Ain23] 和多查询注意力 (MQA) [Sha19]. 然而, 这些方法在尝试减小 KV cache 时往往以牺牲性能为代价. 为了兼得两者之优, 我们提出 MLA, 一种配备低秩键值联合压缩的注意力机制. 实验中, MLA 相比 MHA 取得了更优性能, 同时显著降低了推理期间的 KV cache, 从而提升了推理效率.
2. 对于前馈网络 (FFN), 我们沿用 DeepSeekMoE 架构 [Dai24], 该架构采用细粒度专家切分与共享专家隔离, 以提升专家专业化的潜力. 与 GShard [Lep20] 等传统 MoE 架构相比, DeepSeekMoE 架构展现出巨大优势, 使我们能够以经济成本训练出强大的模型. 由于我们在训练中采用了专家并行, 还设计了补充机制来控制通信开销并确保负载均衡. 通过结合这两项技术, DeepSeek-V2 同时具备强大的性能 ([图 1](#figure-01)(a)), 经济的训练成本, 以及高效的推理吞吐量 ([图 1](#figure-01)(b)).

<span id="figure-02"></span>

![DeepSeek-V2 图 2](./deepseek-v2/figure-02.png)

**图 2.** DeepSeek-V2 的架构示意图. MLA 通过显著减少生成时的 KV cache 来保障高效推理, 而 DeepSeekMoE 则借助稀疏架构实现以经济成本训练出强大的模型.

我们构建了一个高质量、多来源、包含 8.1T token 的预训练语料. 与 DeepSeek 67B (我们此前发布的版本) [Dee24e] 所用的语料相比, 该语料数据量更大, 尤其是中文数据, 且数据质量更高. 我们首先在完整预训练语料上对 DeepSeek-V2 进行预训练. 随后, 我们收集了 150 万个对话会话, 涵盖数学, 代码, 写作, 推理, 安全等诸多领域, 为 DeepSeek-V2 Chat (SFT) 执行监督微调 (SFT). 最后, 我们沿袭 DeepSeekMath [Sha24d], 采用组相对策略优化 (GRPO) 进一步使模型与人类偏好对齐, 从而得到 DeepSeek-V2 Chat (RL).

我们在英语和中文的广泛基准上评测 DeepSeek-V2, 并将其与代表性开源模型比较. 评测结果表明, 即使仅有 21B 激活参数, DeepSeek-V2 依然在开源模型中取得顶尖性能, 并成为最强的开源 MoE 语言模型. [图 1](#figure-01) 突出显示, 在 MMLU 上 DeepSeek-V2 仅凭少量激活参数便取得了顶级表现. 此外, 如 [图 1](#figure-01) 所示, 与 DeepSeek 67B 相比, DeepSeek-V2 节省了 42.5% 的训练成本, 将 KV cache 减少了 93.3%, 并将最大生成吞吐量提升至 5.76 倍. 我们还在开放式基准上评测了 DeepSeek-V2 Chat (SFT) 与 DeepSeek-V2 Chat (RL). 值得注意的是, DeepSeek-V2 Chat (RL) 在 AlpacaEval 2.0 [Dub24a] 上取得了 38.9 的长度控制胜率, 在 MT-Bench [Sto23e] 上取得 8.97 的总分, 在 AlignBench [Liu23m] 上取得 7.91 的总分. 这些英语开放式对话评测表明, DeepSeek-V2 Chat (RL) 在开源对话模型中具有顶尖性能. 此外, AlignBench 上的评测表明, 在中文上 DeepSeek-V2 Chat (RL) 超越了所有开源模型, 甚至击败了大多数闭源模型.

为了推动 MLA 与 DeepSeekMoE 的进一步研究与开发, 我们还向开源社区发布了 DeepSeek-V2-Lite, 一个同样配备 MLA 与 DeepSeekMoE 的较小模型. 它总计包含 15.7B 参数, 其中每个 token 激活 2.4B. 关于 DeepSeek-V2-Lite 的详细描述见 [第 7 节](#section-7).

在本文余下部分, 我们首先详细描述 DeepSeek-V2 的模型架构 ([第 2 节](#section-2)). 随后介绍了我们的预训练努力, 包括训练数据构建, 超参数设置, 基础设施, 长上下文扩展, 以及模型性能与效率的评测 ([第 3 节](#section-3)). 接着, 我们展示了在对齐方面的工作, 涵盖监督微调 (SFT), 强化学习 (RL), 评测结果以及其他讨论 ([第 4 节](#section-4)). 最后, 我们总结结论, 思考 DeepSeek-V2 目前的局限, 并展望未来的工作 ([第 5 节](#section-5)).

<span id="section-2"></span>

## 2 架构

总体而言, DeepSeek-V2 仍采用 Transformer 架构 [Vas17d], 其中每个 Transformer 块由一个注意力模块和一个前馈网络 (FFN) 组成. 然而, 对于注意力模块和 FFN, 我们都设计并采用了创新的架构. 对于注意力, 我们设计了 MLA, 利用低秩键值联合压缩来消除推理时键值 cache 的瓶颈, 从而支持高效推理. 对于 FFN, 我们采用了 DeepSeekMoE 架构 [Dai24], 一种高性能的 MoE 架构, 能够以经济成本训练出强大的模型. [图 2](#figure-02) 展示了 DeepSeek-V2 的架构示意图, 本节将介绍 MLA 与 DeepSeekMoE 的细节. 对于其他细微之处 (例如层归一化和 FFN 中的激活函数), 除非另有说明, DeepSeek-V2 均沿用 DeepSeek 67B [Dee24e] 的设置.

<span id="section-2-1"></span>

### 2.1 多头潜在注意力: 提升推理效率

传统 Transformer 模型通常采用多头注意力 (MHA) [Vas17d], 但在生成过程中, 其庞大的键值 (KV) cache 会成为限制推理效率的瓶颈. 为了减小 KV cache, 人们提出了多查询注意力 (MQA) [Sha19] 和分组查询注意力 (GQA) [Ain23]. 它们所需的 KV cache 数量更小, 但性能不及 MHA (我们在 [第 9.1 节](#section-9-1) 提供了 MHA, GQA 与 MQA 的消融实验).

对于 DeepSeek-V2, 我们设计了一种名为多头潜在注意力 (MLA) 的创新注意力机制. 凭借低秩键值联合压缩, MLA 取得了优于 MHA 的性能, 同时所需的 KV cache 数量显著更少. 下面我们将介绍其架构, 并在 [第 9.2 节](#section-9-2) 中给出 MLA 与 MHA 的对比.

<span id="section-2-1-1"></span>

#### 2.1.1 预备知识: 标准多头注意力

我们首先介绍标准的 MHA 机制以作背景. 设 $d$ 为嵌入维度, $n_h$ 为注意力头数量, $d_h$ 为每个头的维度, 且 $\mathbf{h}_{t} \in \mathbb{R}^{d}$ 为某一注意力层中第 $t$ 个 token 的注意力输入. 标准 MHA 首先通过三个矩阵 $W^{Q}, W^{K}, W^{V} \in \mathbb{R}^{d_h n_h \times d}$ 分别产生 $\mathbf{q}_{t}, \mathbf{k}_{t}, \mathbf{v}_{t} \in \mathbb{R}^{d_h n_h}$:

<span id="equation-01"></span>

$$
\begin{aligned}
    \mathbf{q}_{t} &= W^{Q} \mathbf{h}_{t}, \\
    \mathbf{k}_{t} &= W^{K} \mathbf{h}_{t}, \\
    \mathbf{v}_{t} &= W^{V} \mathbf{h}_{t},
\end{aligned}
$$

随后, 为进行多头注意力计算, $\mathbf{q}_{t}, \mathbf{k}_{t}, \mathbf{v}_{t}$ 将被切分为 $n_h$ 个头:

$$
\begin{aligned}
    [\mathbf{q}_{t, 1};&\mathbf{q}_{t, 2};...;\mathbf{q}_{t, n_{h}}] = \mathbf{q}_{t}, \\
    [\mathbf{k}_{t, 1};&\mathbf{k}_{t, 2};...;\mathbf{k}_{t, n_{h}}] = \mathbf{k}_{t}, \\
    [\mathbf{v}_{t, 1};&\mathbf{v}_{t, 2};...;\mathbf{v}_{t, n_{h}}] = \mathbf{v}_{t}, \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h}}}) \mathbf{v}_{j, i}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

其中 $\mathbf{q}_{t, i}, \mathbf{k}_{t, i}, \mathbf{v}_{t, i} \in \mathbb{R}^{d_h}$ 分别表示第 $i$ 个注意力头的查询, 键和值; $W^{O} \in \mathbb{R}^{d \times d_h n_h}$ 表示输出投影矩阵. 推理期间, 所有键和值都需要缓存以加速推理, 因此 MHA 需要为每个 token 缓存 $2 n_{h} d_{h} l$ 个元素. 在模型部署中, 这一庞大的 KV cache 是限制最大批大小和序列长度的主要瓶颈.

<span id="figure-03"></span>

![DeepSeek-V2 图 3](./deepseek-v2/figure-03.png)

**图 3.** 多头注意力 (MHA), 分组查询注意力 (GQA), 多查询注意力 (MQA) 与多头潜在注意力 (MLA) 的简化示意图. 通过将键和值联合压缩为潜在向量, MLA 在推理期间显著降低了 KV cache.

<span id="section-2-1-2"></span>

#### 2.1.2 低秩键值联合压缩

MLA 的核心是对键和值进行低秩联合压缩以减小 KV cache:

<span id="equation-10"></span>

$$
\begin{aligned}
    \mathbf{c}_{t}^{\mathit{KV}} &= W^{\mathit{DKV}} \mathbf{h}_{t}, \\
    \mathbf{k}_{t}^{C} &= W^{\mathit{UK}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{v}_{t}^{C} &= W^{\mathit{UV}} \mathbf{c}_{t}^{\mathit{KV}},
\end{aligned}
$$

其中 $\mathbf{c}_{t}^{\mathit{KV}} \in \mathbb{R}^{d_c}$ 是键和值的压缩潜在向量; $d_c (\ll d_h n_h)$ 表示 KV 压缩维度; $W^{\mathit{DKV}} \in \mathbb{R}^{d_c \times d}$ 是下投影矩阵; 而 $W^{\mathit{UK}},W^{\mathit{UV}} \in \mathbb{R}^{d_h n_h \times d_c}$ 分别是键和值的上投影矩阵. 推理期间, MLA 只需缓存 $\mathbf{c}_{t}^{\mathit{KV}}$, 因此其 KV cache 仅有 $d_{c}l$ 个元素, 其中 $l$ 表示层数. 此外, 推理期间由于 $W^{\mathit{UK}}$ 可被吸收进 $W^{Q}$, 且 $W^{\mathit{UV}}$ 可被吸收进 $W^{O}$, 我们甚至无需为注意力计算出键和值. [图 3](#figure-03) 直观地展示了 MLA 中的 KV 联合压缩如何降低 KV cache.

此外, 为了降低训练期间的激活内存, 我们即使无法因此减小 KV cache, 也仍对查询进行低秩压缩:

$$
\begin{aligned}
    \mathbf{c}_{t}^{Q} &= W^{\mathit{DQ}} \mathbf{h}_{t}, \\
    \mathbf{q}_{t}^{C} &= W^{\mathit{UQ}} \mathbf{c}_{t}^{Q},
\end{aligned}
$$

其中 $\mathbf{c}_{t}^{Q} \in \mathbb{R}^{d_c^{\prime}}$ 是查询的压缩潜在向量; $d_c^{\prime} (\ll d_h n_h)$ 表示查询压缩维度; 而 $W^{\mathit{DQ}} \in \mathbb{R}^{d_c^{\prime} \times d}, W^{\mathit{UQ}} \in \mathbb{R}^{d_h n_h \times d_c^{\prime}}$ 分别是查询的下投影矩阵和上投影矩阵.

<span id="section-2-1-3"></span>

#### 2.1.3 解耦旋转位置编码

沿袭 DeepSeek 67B [Dee24e], 我们打算为 DeepSeek-V2 使用旋转位置编码 (RoPE) [Su24]. 然而, RoPE 与低秩 KV 压缩不兼容. 具体而言, RoPE 对键和查询都具有位置敏感性. 如果我们将 RoPE 应用于键 $\mathbf{k}_{t}^{C}$, 那么 [式 10](#equation-10) 中的 $W^{\mathit{UK}}$ 将与一个位置敏感的 RoPE 矩阵耦合. 这样一来, 推理期间 $W^{\mathit{UK}}$ 便不能再被吸收进 $W^{Q}$, 因为与当前正在生成的 token 相关的 RoPE 矩阵会位于 $W^{Q}$ 与 $W^{\mathit{UK}}$ 之间, 而矩阵乘法不满足交换律. 其结果是, 我们必须在推理期间为所有前缀 token 重新计算键, 这会显著阻碍推理效率.

作为解决方案, 我们提出了解耦 RoPE 策略, 它使用额外的多头查询 $\mathbf{q}_{t, i}^{R} \in \mathbb{R}^{d_h^R}$ 和一个共享键 $\mathbf{k}_{t}^{R} \in \mathbb{R}^{d_h^R}$ 来承载 RoPE, 其中 $d_h^R$ 表示解耦查询与键的每头维度. 配备解耦 RoPE 策略后, MLA 执行如下计算:

$$
\begin{aligned}
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{QR}}} \mathbf{c}_{t}^{Q}), \\
    \mathbf{k}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{KR}}} \mathbf{h}_{t}), \\
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \\
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

其中 $W^{\mathit{QR}} \in \mathbb{R}^{d_h^R n_h \times d_c^{\prime}}$ 与 $W^{\mathit{KR}} \in \mathbb{R}^{d_h^R \times d}$ 是分别产生解耦查询与键的矩阵; $\mathop{\mathrm{RoPE}}(\cdot)$ 表示施加 RoPE 矩阵的操作; 而 $[\cdot;\cdot]$ 表示拼接操作. 推理期间, 解耦键也应被缓存. 因此 DeepSeek-V2 所需的 KV cache 总计包含 $(d_{c} + d_h^R)l$ 个元素.

为了展示 MLA 的完整计算过程, 我们还在 [第 8 节](#section-8) 整理并给出了其完整公式.

<span id="table-01"></span>

![DeepSeek-V2 表 1](./deepseek-v2/table-01.png)

**表 1.** 不同注意力机制之间每个 token 的 KV cache 比较. $n_{h}$ 表示注意力头数量, $d_{h}$ 表示每个注意力头的维度, $l$ 表示层数, $n_{g}$ 表示 GQA 中的分组数, 而 $d_{c}$ 与 $d_h^R$ 分别表示 MLA 中 KV 压缩维度和解耦查询与键的每头维度. KV cache 的数量以元素个数衡量, 与存储精度无关. 对于 DeepSeek-V2, $d_{c}$ 设为 $4d_{h}$, $d_h^R$ 设为 $\frac{d_{h}}{2}$. 因此, 其 KV cache 相当于仅含 2.25 个分组的 GQA, 但性能却强于 MHA.

<span id="section-2-1-4"></span>

#### 2.1.4 键值缓存比较

我们在 [表 1](#table-01) 中展示了不同注意力机制之间每个 token 的 KV cache 比较. MLA 仅需少量 KV cache, 相当于仅含 2.25 个分组的 GQA, 却能达到优于 MHA 的性能.

<span id="section-2-2"></span>

### 2.2 DeepSeekMoE: 以经济成本训练强大的模型

<span id="section-2-2-1"></span>

#### 2.2.1 基本架构

对于 FFN, 我们采用 DeepSeekMoE 架构 [Dai24]. DeepSeekMoE 有两个核心思想: 将专家划分为更细的粒度, 以提高专家专业化并实现更准确的知识获取; 以及隔离部分共享专家, 以缓解路由专家之间的知识冗余. 在激活参数和总专家参数数量相同的情况下, DeepSeekMoE 可以大幅超越 GShard [Lep20] 等传统 MoE 架构.

设 $\mathbf{u}_{t}$ 为第 $t$ 个 token 的 FFN 输入, 我们按如下方式计算 FFN 输出 $\mathbf{h}_{t}^{\prime}$:

$$
\begin{aligned}
    \mathbf{h}_{t}^{\prime} & = \mathbf{u}_{t} + \sum_{i=1}^{N_{s}} {\mathop{\mathrm{FFN}}^{(s)}_{i}\left( \mathbf{u}_{t} \right)} + \sum_{i=1}^{N_r} {g_{i,t} \mathop{\mathrm{FFN}}^{(r)}_{i}\left( \mathbf{u}_{t} \right)}, \\
    g_{i,t} & = \begin{cases}
    s_{i,t}, & s_{i,t} \in \mathop{\mathrm{Topk}} (\{ s_{j, t} | 1 \leq j \leq N_r \}, K_{r}), \\
    0, & \mathrm{otherwise},
    \end{cases} \\
    s_{i,t} & = \mathop{\mathrm{Softmax}}_i \left( {\mathbf{u}_{t}}^\top \mathbf{e}_{i} \right),
\end{aligned}
$$

其中 $N_{s}$ 与 $N_r$ 分别表示共享专家和路由专家的数量; $\mathop{\mathrm{FFN}}^{(s)}_{i}(\cdot)$ 与 $\mathop{\mathrm{FFN}}^{(r)}_{i}(\cdot)$ 分别表示第 $i$ 个共享专家和第 $i$ 个路由专家; $K_{r}$ 表示被激活的路由专家数量; $g_{i,t}$ 是第 $i$ 个专家的门控值; $s_{i,t}$ 是 token 与专家之间的亲和度; $\mathbf{e}_{i}$ 是该层第 $i$ 个路由专家的质心; 而 $\mathop{\mathrm{Topk}}(\cdot, K)$ 表示由针对第 $t$ 个 token 与所有路由专家计算出的亲和度分数中得分最高的 $K$ 个分数构成的集合.

<span id="section-2-2-2"></span>

#### 2.2.2 设备受限路由

我们设计了一种设备受限路由机制来限定 MoE 相关的通信开销. 当采用专家并行时, 路由专家会被分布到多个设备上. 对于每个 token, 其 MoE 相关的通信频率与其目标专家所覆盖的设备数量成正比. 由于 DeepSeekMoE 中的细粒度专家切分, 被激活的专家数量可能很大, 因此如果我们采用专家并行, MoE 相关的通信会更为昂贵.

对于 DeepSeek-V2, 除了对路由专家进行朴素的 top-K 选择外, 我们还额外确保每个 token 的目标专家分布在至多 $M$ 个设备上. 具体而言, 对于每个 token, 我们首先选择具有最高亲和度分数的专家所在的 $M$ 个设备. 然后, 我们在这 $M$ 个设备上的专家之间进行 top-K 选择. 实践中我们发现, 当 $M \geq 3$ 时, 设备受限路由能够取得与不受限 top-K 路由大致对齐的良好性能.

<span id="section-2-2-3"></span>

#### 2.2.3 负载均衡的辅助损失

我们考虑了自动学习路由策略中的负载均衡问题. 首先, 不均衡的负载会增大路由坍缩 [Sha17] 的风险, 使部分专家无法被充分训练和利用. 其次, 当采用专家并行时, 不均衡的负载会降低计算效率. 在 DeepSeek-V2 的训练过程中, 我们设计了三种辅助损失, 分别用于控制专家级负载均衡 ($\mathcal{L}_{\mathrm{ExpBal}}$), 设备级负载均衡 ($\mathcal{L}_{\mathrm{DevBal}}$) 和通信均衡 ($\mathcal{L}_{\mathrm{CommBal}}$).

**专家级均衡损失.** 我们使用专家级均衡损失 [Fed22, Lep20] 来缓解路由坍缩的风险:

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{ExpBal}} & = \alpha_1 \sum_{i=1}^{N_r}{f_i P_i}, \\
    f_i & = \frac{N_r}{K_r T} \sum_{t=1}^{T}{ \mathds{1}( \mathrm{Token}\ t\ \text{选择专家}\ i )}, \\
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s_{i,t}},
\end{aligned}
$$

其中 $\alpha_1$ 是一个称为专家级均衡因子的超参数; $\mathds{1}(\cdot)$ 表示指示函数; 而 $T$ 表示一个序列中的 token 数量.

**设备级均衡损失.** 除了专家级均衡损失之外, 我们还额外设计了一种设备级均衡损失, 以确保不同设备之间的计算均衡. 在 DeepSeek-V2 的训练过程中, 我们将所有路由专家划分为 $D$ 组 $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D \}$, 并把每组部署在单个设备上. 设备级均衡损失的计算如下:

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{DevBal}} & = \alpha_{2} \sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}, \\
    f_i^{\prime} & = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i}{ f_j }, \\
    P_i^{\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

其中 $\alpha_{2}$ 是一个称为设备级均衡因子的超参数.

**通信均衡损失.** 最后, 我们引入一种通信均衡损失, 以确保每个设备的通信是均衡的. 尽管设备受限路由机制保证了每个设备的发送通信是有界的, 但如果某个设备接收的 token 多于其他设备, 实际的通信效率也会受到影响. 为了缓解这一问题, 我们设计了如下的通信均衡损失:

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{CommBal}} & = \alpha_{3} \sum_{i=1}^{D}{f_i^{\prime\prime} P_i^{\prime\prime}}, \\
    f_i^{\prime\prime} & = \frac{D}{M T} \sum_{t=1}^{T}{ \mathds{1}( \mathrm{Token}\ t\ \text{被发送至设备}\ i )}, \\
    P_i^{\prime\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

其中 $\alpha_{3}$ 是一个称为通信均衡因子的超参数. 设备受限路由机制遵循的原则是确保每个设备向其他设备传输至多 $M T$ 个隐藏状态. 同时, 通信均衡损失被用来鼓励每个设备从其他设备接收约 $M T$ 个隐藏状态. 通信均衡损失保证了设备之间信息的均衡交换, 从而促进高效通信.

<span id="section-2-2-4"></span>

#### 2.2.4 Token 丢弃策略

尽管均衡损失旨在鼓励均衡的负载, 但必须承认它们无法保证严格的负载均衡. 为了进一步缓解不均衡负载造成的计算浪费, 我们在训练期间引入了一种设备级 token 丢弃策略. 该方法首先计算每个设备的平均计算预算, 这意味着每个设备的容量因子相当于 1.0. 然后, 受 [Riq21] 启发, 我们在每个设备上丢弃亲和度分数最低的 token, 直至达到计算预算. 此外, 我们还确保属于约 10% 训练序列的 token 永远不会被丢弃. 这样, 我们可以根据效率需求灵活决定是否在推理期间丢弃 token, 并始终确保训练与推理之间的一致性.

<span id="section-3"></span>

## 3 预训练

<span id="section-3-1"></span>

### 3.1 实验设置

<span id="section-3-1-1"></span>

#### 3.1.1 数据构建

在保持与 DeepSeek 67B [Dee24e] 相同的数据处理阶段的同时, 我们扩充了数据量并提升了数据质量. 为了扩大预训练语料, 我们挖掘了互联网数据的潜力并优化了清洗流程, 从而恢复了大量被误删的数据. 此外, 我们还纳入了更多中文数据, 旨在更好地利用中文互联网上可用的语料. 除了数据量, 我们也关注数据质量. 我们用来自多种来源的高质量数据丰富了预训练语料, 同时改进了基于质量的过滤算法. 改进后的算法确保大量无益数据被移除, 而有价值的数据则大多被保留. 此外, 我们从预训练语料中过滤掉具有争议的内容, 以缓解特定地域文化引入的数据偏差. 关于这一过滤策略影响的详细讨论见 [第 10 节](#section-10).

我们采用了与 DeepSeek 67B 相同的 tokenizer, 它基于字节级字节对编码 (BBPE) 算法构建, 词汇表大小为 100K. 我们分词后的预训练语料包含 8.1T token, 其中中文 token 大约比英文多 12%.

<span id="section-3-1-2"></span>

#### 3.1.2 超参数

**模型超参数.** 我们将 Transformer 层数设为 60, 隐藏维度设为 5120. 所有可学习参数均以标准差 0.006 随机初始化. 在 MLA 中, 我们将注意力头数量 $n_h$ 设为 128, 每头维度 $d_h$ 设为 128. KV 压缩维度 $d_c$ 设为 512, 查询压缩维度 $d_c^{\prime}$ 设为 1536. 对于解耦查询与键, 我们将每头维度 $d_h^R$ 设为 64. 按照 [Dai24], 我们除了第一层外, 将所有 FFN 替换为 MoE 层. 每个 MoE 层包含 2 个共享专家和 160 个路由专家, 其中每个专家的中间隐藏维度为 1536. 在路由专家中, 每个 token 会激活 6 个专家. 此外, 低秩压缩与细粒度专家切分会影响到一层的输出尺度. 因此, 实践中我们在压缩潜在向量之后额外加入 RMS Norm 层, 并在宽度瓶颈处 (即压缩潜在向量与路由专家的中间隐藏状态) 乘以额外的缩放因子, 以确保训练稳定. 在此配置下, DeepSeek-V2 总计包含 236B 参数, 其中每个 token 激活 21B.

**训练超参数.** 我们采用 AdamW 优化器 [Los17], 超参数设为 $\beta_1=0.9$, $\beta_2=0.95$, 且 $\mathrm{weight\_decay}=0.1$. 学习率采用预热加阶梯衰减策略 [Dee24e] 进行调度. 最初, 学习率在前 2K 步内从 0 线性增至最大值. 随后, 学习率在大约训练了 60% 的 token 后乘以 0.316, 并在大约训练了 90% 的 token 后再次乘以 0.316. 最大学习率设为 $2.4 \times 10^{-4}$, 梯度裁剪范数设为 1.0. 我们还使用了批大小调度策略, 即在前 225B token 的训练中, 批大小从 2304 逐步增至 9216, 其余训练则保持 9216. 我们将最大序列长度设为 4K, 并在 8.1T token 上训练 DeepSeek-V2. 我们利用流水线并行将模型的不同层部署到不同设备上, 且对于每一层, 路由专家被均匀部署在 8 个设备上 ($D=8$). 至于设备受限路由, 每个 token 至多会被发送到 3 个设备 ($M=3$). 至于均衡损失, 我们将 $\alpha_{1}$ 设为 0.003, $\alpha_{2}$ 设为 0.05, $\alpha_{3}$ 设为 0.02. 我们在此训练期间采用 token 丢弃策略以加速, 但在评测时不丢弃任何 token.

<span id="section-3-1-3"></span>

#### 3.1.3 基础设施

DeepSeek-V2 基于 HAI-LLM 框架 [Hig23] 训练, 这是我们工程师内部开发的一个高效轻量的训练框架. 它采用了 16 路零气泡流水线并行 [Qi23], 8 路专家并行 [Lep20] 与 ZeRO-1 数据并行 [Raj20]. 鉴于 DeepSeek-V2 相对较少的激活参数, 以及为节省激活内存而对部分算子进行的重计算, 它无需张量并行即可训练, 从而降低了通信开销. 此外, 为了进一步提升训练效率, 我们将共享专家的计算与专家并行的 all-to-all 通信重叠起来. 我们还为通信, 路由算法以及跨不同专家的融合线性计算定制了更快的 CUDA 内核. 此外, MLA 也基于 FlashAttention-2 [Dao24] 的改进版本进行了优化.

我们在配备 NVIDIA H800 GPU 的集群上进行了所有实验. H800 集群中的每个节点包含 8 个 GPU, 节点内部通过 NVLink 和 NVSwitch 连接. 节点之间则利用 InfiniBand 互连来促进通信.

<span id="figure-04"></span>

![DeepSeek-V2 图 4](./deepseek-v2/figure-04.png)

**图 4.** "Needle In A Haystack" (NIAH) 测试的评测结果. DeepSeek-V2 在直至 128K 的所有上下文窗口长度上都表现出色.

<span id="section-3-1-4"></span>

#### 3.1.4 长上下文扩展

在 DeepSeek-V2 的初始预训练之后, 我们采用 YaRN [Pen23] 将默认上下文窗口长度从 4K 扩展到 128K. YaRN 被专门应用于解耦共享键 $\mathbf{k}^R_t$, 因为它负责承载 RoPE [Su24]. 对于 YaRN, 我们将缩放尺度 $s$ 设为 40, $\alpha$ 设为 1, $\beta$ 设为 32, 目标最大上下文长度设为 160K. 在这些设置下, 我们预期模型对 128K 的上下文长度能够良好响应. 与原始 YaRN 略有不同, 由于我们独特的注意力机制, 我们调整了长度缩放因子以调节注意力熵. 该因子 $\sqrt{t}$ 计算为 $\sqrt{t} = 0.0707 \ln{s} + 1$, 旨在最小化困惑度.

我们另外训练了模型 1000 步, 序列长度为 32K, 批大小为 576 个序列. 尽管训练仅在 32K 的序列长度上进行, 但模型在 128K 上下文长度下评测时仍展现出稳健的性能. 如 [图 4](#figure-04) 所示, "Needle In A Haystack" (NIAH) 测试的结果表明, DeepSeek-V2 在直至 128K 的所有上下文窗口长度上都表现出色.

<span id="section-3-2"></span>

### 3.2 评测

<span id="section-3-2-1"></span>

#### 3.2.1 评测基准

DeepSeek-V2 在一个双语语料上预训练, 因此我们在一系列英语和中文基准上对其评测. 我们的评测基于集成在 HAI-LLM 框架中的内部评测框架. 纳入的基准按如下分类列出, 其中带下划线的基准为中文:

**多学科多项选择** 数据集包括 MMLU [Hen20], C-Eval [Hua23] 和 CMMLU [Li23e].

**语言理解与推理** 数据集包括 HellaSwag [Zel19], PIQA [Bis20], ARC [Cla18] 和 BigBench Hard (BBH) [Suz22].

**闭卷问答** 数据集包括 TriviaQA [Jos17] 和 NaturalQuestions [Kwi19a].

**阅读理解** 数据集包括 RACE [Lai17], DROP [Dua19], C3 [Sun19c] 和 CMRC [Cui19].

**指代消解** 数据集包括 WinoGrande [Sak19] 和 CLUEWSC [Xu20].

**语言建模** 数据集包括 Pile [Gao20].

**中文理解与文化** 数据集包括 CHID [Zhe19] 和 CCPM [Li21e].

**数学** 数据集包括 GSM8K [Cob21], MATH [Hen21] 和 CMath [Wei23b].

**代码** 数据集包括 HumanEval [Che21e], MBPP [Aus21b] 和 CRUXEval [Gu24].

**标准化考试** 包括 AGIEval [Zho23]. 请注意, AGIEval 同时包含英语和中文子集.

沿袭我们此前的工作 [Dee24e], 我们对包括 HellaSwag, PIQA, WinoGrande, RACE-Middle, RACE-High, MMLU, ARC-Easy, ARC-Challenge, CHID, C-Eval, CMMLU, C3 和 CCPM 在内的数据集采用基于困惑度的评测, 对 TriviaQA, NaturalQuestions, DROP, MATH, GSM8K, HumanEval, MBPP, CRUXEval, BBH, AGIEval, CLUEWSC, CMRC 和 CMath 采用基于生成的评测. 此外, 我们对 Pile-test 进行基于语言建模的评测, 并使用每字节比特数 (BPB) 作为指标, 以保证不同 tokenizer 模型之间的公平比较.

为了对这些基准提供直观的概览, 我们还在 [第 12 节](#section-12) 提供了每个基准的评测格式.

<span id="section-3-2-2"></span>

#### 3.2.2 评测结果

<span id="table-02"></span>

![DeepSeek-V2 表 2](./deepseek-v2/table-02.png)

**表 2.** DeepSeek-V2 与其他代表性开源模型的比较. 所有模型均在内部框架中评测, 并共享相同的评测设置. **加粗** 表示最佳, 下划线表示次佳. 差距小于 0.3 的分数视为同一水平. 仅有 21B 激活参数时, DeepSeek-V2 在开源模型中取得了顶尖性能.

在 [表 2](#table-02) 中, 我们将 DeepSeek-V2 与几个代表性开源模型比较, 包括 DeepSeek 67B [Dee24e] (我们此前发布的版本), Qwen1.5 72B [Bai23b], LLaMA3 70B [Dub24] 和 Mixtral 8x22B [Mis24]. 我们用内部评测框架评测所有这些模型, 并确保它们共享相同的评测设置. 总体而言, 仅有 21B 激活参数时, DeepSeek-V2 在几乎所有基准上都显著优于 DeepSeek 67B, 并在开源模型中取得顶尖性能.

进一步, 我们逐一详细比较 DeepSeek-V2 与其开源同类.

1. 与同时支持中文和英语的 Qwen1.5 72B 相比, DeepSeek-V2 在大部分英语, 代码和数学基准上展现出压倒性优势. 至于中文基准, Qwen1.5 72B 在多学科多项选择任务上表现更好, 而 DeepSeek-V2 在其他任务上与之相当或更优. 请注意, 对于 CHID 基准, Qwen1.5 72B 的 tokenizer 在我们的评测框架中会报错, 因此我们将 Qwen1.5 72B 的 CHID 分数留空.
2. 与 Mixtral 8x22B 相比, DeepSeek-V2 取得了相当或更好的英语性能, 除了与英语常识知识密切相关的 TriviaQA, NaturalQuestions 和 HellaSwag. 值得注意的是, DeepSeek-V2 在 MMLU 上优于 Mixtral 8x22B. 在代码和数学基准上, DeepSeek-V2 展现出与 Mixtral 8x22B 相当的性能. 由于 Mixtral 8x22B 并未专门在中文数据上训练, 其中文能力远落后于 DeepSeek-V2.
3. 与 LLaMA3 70B 相比, DeepSeek-V2 训练的英语 token 不足其四分之一. 因此, 我们承认 DeepSeek-V2 在基础英语能力上与 LLaMA3 70B 仍存在轻微差距. 然而, 即使训练 token 和激活参数都少得多, DeepSeek-V2 仍展现出与 LLaMA3 70B 相当的代码和数学能力. 此外, 作为双语语言模型, DeepSeek-V2 在中文基准上以压倒性优势超越 LLaMA3 70B.

最后, 值得提及的是, 某些先前研究 [Hu24] 在预训练阶段纳入了 SFT 数据, 而 DeepSeek-V2 在预训练期间从未接触过 SFT 数据.

<span id="section-3-2-3"></span>

#### 3.2.3 训练与推理效率

**训练成本.** 由于 DeepSeek-V2 每个 token 激活的参数更少, 所需的 FLOPs 也少于 DeepSeek 67B, 因此理论上训练 DeepSeek-V2 比训练 DeepSeek 67B 更经济. 尽管训练 MoE 模型会引入额外的通信开销, 但通过我们的算子与通信优化, DeepSeek-V2 的训练可以达到相对较高的模型 FLOPs 利用率 (MFU). 在 H800 集群上的实际训练中, 每训练一万亿 token, DeepSeek 67B 需要 300.6K GPU 小时, 而 DeepSeek-V2 仅需 172.8K GPU 小时, 即稀疏的 DeepSeek-V2 相比稠密的 DeepSeek 67B 可节省 42.5% 的训练成本.

**推理效率.** 为了高效部署 DeepSeek-V2 提供服务, 我们首先将其参数转换为 FP8 精度. 此外, 我们还对 DeepSeek-V2 执行 KV cache 量化 [Hoo24, Zha24e], 以进一步将其 KV cache 中的每个元素平均压缩为 6 比特. 得益于 MLA 与这些优化, 实际部署的 DeepSeek-V2 所需的 KV cache 远少于 DeepSeek 67B, 因此能够服务更大的批大小. 我们基于实际部署的 DeepSeek 67B 服务中的提示与生成长度分布来评测 DeepSeek-V2 的生成吞吐量. 在配备 8 个 H800 GPU 的单个节点上, DeepSeek-V2 实现了超过 5 万 token 每秒的生成吞吐量, 是 DeepSeek 67B 最大生成吞吐量的 5.76 倍. 此外, DeepSeek-V2 的提示输入吞吐量超过 10 万 token 每秒.

<span id="section-4"></span>

## 4 对齐

<span id="section-4-1"></span>

### 4.1 监督微调

在我们此前研究 [Dee24e] 的基础上, 我们精心构建了包含 150 万个实例的指令调优数据集, 其中包括 120 万个用于有用性的实例和 30 万个用于安全性的实例. 与初始版本相比, 我们提升了数据质量, 以缓解幻觉式回复并增强写作能力. 我们对 DeepSeek-V2 进行 2 个 epoch 的微调, 学习率设为 $5 \times 10^{-6}$. 对于 DeepSeek-V2 Chat (SFT) 的评测, 我们主要纳入基于生成的基准, 除少数代表性多项选择任务 (MMLU 和 ARC) 外. 我们还为 DeepSeek-V2 Chat (SFT) 进行了指令遵循评测 (IFEval) [Zho23a], 以提示级别的宽松准确率作为指标. 此外, 我们使用 2023 年 9 月 1 日至 2024 年 4 月 1 日期间的 LiveCodeBench [Jai25a] 问题来评测对话模型. 除了标准基准, 我们还在包括 MT-Bench [Sto23e], AlpacaEval 2.0 [Dub24a] 和 AlignBench [Liu23m] 在内的开放式对话基准上进一步评测我们的模型. 为了比较, 我们还在内部评测框架和设置中评测了 Qwen1.5 72B Chat, LLaMA-3-70B Instruct 和 Mistral-8x22B Instruct. 至于 DeepSeek 67B Chat, 我们直接参考此前发布中报告的评测结果.

<span id="section-4-2"></span>

### 4.2 强化学习

为了进一步释放 DeepSeek-V2 的潜力并使其与人类偏好对齐, 我们进行了强化学习 (RL) 以调整其偏好.

**强化学习算法.** 为了节省 RL 的训练成本, 我们采用组相对策略优化 (GRPO) [Sha24d], 它舍弃了通常与策略模型同规模的评论家模型, 转而从组分数中估计基线. 具体而言, 对于每个问题 $q$, GRPO 从旧策略 $\pi_{\theta_{\mathrm{old}}}$ 中采样一组输出 $\{o_1, o_2, \cdots, o_G\}$, 然后通过最大化如下目标来优化策略模型 $\pi_{\theta}$:

<span id="equation-25"></span>

$$
\begin{aligned}
    \mathcal{J}_{\mathrm{GRPO}}(\theta) &= \mathbb{E}{[q \sim P(Q), \{o_i\}_{i=1}^G \sim \pi_{\theta_{\mathrm{old}}}(O|q)]}  \\
    & \frac{1}{G}\sum_{i=1}^G \left( \min \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{\mathrm{old}}}(o_i |q)} A_i, \mathop{\mathrm{clip}} \left( \frac{\pi_\theta(o_i |q)}{\pi_{\theta_{\mathrm{old}}}(o_i |q)}, 1 - \epsilon, 1 + \epsilon \right)  A_i \right) - \beta \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta} \| \pi_{\mathrm{ref}}\right)\right) ,
\end{aligned}
$$

$$
\begin{aligned}
    \mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta} \| \pi_{\mathrm{ref}}\right) = \frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)}- \log\frac{\pi_{\mathrm{ref}}(o_i|q)}{\pi_{\theta}(o_i|q)} - 1,
\end{aligned}
$$

其中 $\epsilon$ 与 $\beta$ 是超参数; 而 $A_i$ 是优势, 由与每组内输出对应的一组奖励 $\{r_1, r_2, \ldots, r_G\}$ 计算得出:

$$
\begin{aligned}
    A_i = \frac{r_i - {\mathrm{mean}(\{r_1, r_2, \cdots, r_G\})}}{{\mathrm{std}(\{r_1, r_2, \cdots, r_G\})}}.
\end{aligned}
$$

**训练策略.** 在我们的初步实验中, 我们发现对推理数据 (例如代码和数学提示) 进行的 RL 训练展现出与通用数据训练不同的独特特征. 例如, 我们模型的数学和编码能力可以在更长的训练步数内持续提升. 因此, 我们采用两阶段 RL 训练策略, 它首先进行推理对齐, 然后进行人类偏好对齐. 在第一阶段的推理对齐中, 我们为代码和数学推理任务训练一个奖励模型 $\mathit{RM}_{\mathrm{reasoning}}$, 并用 $\mathit{RM}_{\mathrm{reasoning}}$ 的反馈来优化策略模型:

$$
\begin{aligned}
    r_i=\mathit{RM}_{\mathrm{reasoning}}(o_i).
\end{aligned}
$$

在第二阶段的类偏好对齐中, 我们采用一个多奖励框架, 它从一个有用性奖励模型 $\mathit{RM}_{\mathrm{helpful}}$, 一个安全性奖励模型 $\mathit{RM}_{\mathrm{safety}}$ 和一个基于规则的奖励模型 $\mathit{RM}_{\mathrm{rule}}$ 获取奖励. 一个回复 $o_i$ 的最终奖励为

$$
\begin{aligned}
    r_i = c_1 \cdot \mathit{RM}_{\mathrm{helpful}}(o_i) + c_2 \cdot \mathit{RM}_{\mathrm{safety}}(o_i) + c_3 \cdot \mathit{RM}_{\mathrm{rule}}(o_i),
\end{aligned}
$$

其中 $c_1$, $c_2$, 与 $c_3$ 是对应的系数.

为了获得在 RL 训练中发挥关键作用的可靠奖励模型, 我们精心收集偏好数据, 并细致地进行质量过滤和比例调整. 我们基于编译器反馈获得代码偏好数据, 并基于真实标签获得数学偏好数据. 对于奖励模型训练, 我们用 DeepSeek-V2 Chat (SFT) 初始化奖励模型, 并使用逐点或成对损失进行训练. 在我们的实验中, 我们观察到 RL 训练能够充分挖掘并激活我们模型的潜力, 使其能够从可能的回复中选出正确且令人满意的答案.

**针对训练效率的优化.** 在超大模型上进行 RL 训练对训练框架提出了很高要求. 它需要精细的工程优化来管理 GPU 内存和 RAM 压力, 同时保持较快的训练速度. 为实现这一目标, 我们实施了以下工程优化. (1) 首先, 我们提出一种混合引擎, 它对训练和推理分别采用不同的并行策略, 以实现更高的 GPU 利用率. (2) 其次, 我们利用带大批大小的 vLLM [Kwo23] 作为推理后端, 以加速推理速度. (3) 第三, 我们精心设计了一种将模型卸载到 CPU 并加载回 GPU 的调度策略, 在训练速度与内存消耗之间取得了接近最优的平衡.

<span id="section-4-3"></span>

### 4.3 评测结果

**标准基准上的评测.** 首先, 我们在标准基准上评测 DeepSeek-V2 Chat (SFT) 与 DeepSeek-V2 Chat (RL). 值得注意的是, 与其基础版本相比, DeepSeek-V2 Chat (SFT) 在 GSM8K, MATH 和 HumanEval 评测上展现出显著改进. 这一进展可归因于我们纳入了 SFT 数据, 其中包含大量与数学和代码相关的内容. 此外, DeepSeek-V2 Chat (RL) 进一步提升了数学和代码基准上的性能. 我们在 [第 11 节](#section-11) 展示了更多代码和数学评测.

至于与其他模型的比较, 我们首先将 DeepSeek-V2 Chat (SFT) 与 Qwen1.5 72B Chat 比较, 发现 DeepSeek-V2 Chat (SFT) 在几乎所有英语, 数学和代码基准上都超越了 Qwen1.5 72B Chat. 在中文基准上, DeepSeek-V2 Chat (SFT) 在多学科多项选择任务上的得分略低于 Qwen1.5 72B Chat, 这与它们基础版本所观察到的性能一致. 当与最先进的开源 MoE 模型 Mixtral 8x22B Instruct 比较时, DeepSeek-V2 Chat (SFT) 在大多数基准上表现出更好的性能, 除了 NaturalQuestions 和 IFEval. 此外, 与最先进的开源模型 LLaMA3 70B Chat 相比, DeepSeek-V2 Chat (SFT) 在代码和数学相关基准上表现出相近的性能. LLaMA3 70B Chat 在 MMLU 和 IFEval 上表现更好, 而 DeepSeek-V2 Chat (SFT) 在中文任务上展现出更强的性能. 最终, 与 DeepSeek-V2 Chat (SFT) 相比, DeepSeek-V2 Chat (RL) 在数学和编码任务上都展现出进一步增强的性能. 这些比较凸显了 DeepSeek-V2 Chat 在不同领域和语言中相对于其他语言模型的优势.

<span id="table-03"></span>

![DeepSeek-V2 表 3](./deepseek-v2/table-03.png)

**表 3.** DeepSeek-V2 Chat (SFT), DeepSeek-V2 Chat (RL) 与其他代表性开源对话模型的比较. 关于 TriviaQA 和 NaturalQuestions, 值得注意的是, LLaMA3 70B Instruct 等对话模型可能不会严格遵守少样本设置中通常指定的格式约束. 因此, 这可能导致在我们的评测框架中低估某些模型.

**开放式生成的评测.** 我们继续在开放式对话基准上评测我们的模型. 对于英语开放式对话生成, 我们使用 MT-Bench 和 AlpacaEval 2.0 作为基准. [表 4](#table-04) 中呈现的评测结果表明, DeepSeek-V2 Chat (RL) 相比 DeepSeek-V2 Chat (SFT) 具有显著的性能优势. 这一结果彰显了我们的 RL 训练在实现更好对齐方面的有效性. 与其他开源模型相比, DeepSeek-V2 Chat (RL) 在这两个基准上都表现出优于 Mistral 8x22B Instruct 和 Qwen1.5 72B Chat 的性能. 当与 LLaMA3 70B Instruct 比较时, DeepSeek-V2 Chat (RL) 在 MT-Bench 上展现出有竞争力的性能, 并在 AlpacaEval 2.0 上显著超越它. 这些结果凸显了 DeepSeek-V2 Chat (RL) 在生成高质量且上下文相关回复方面的强大性能, 尤其在基于指令的对话任务中.

<span id="table-04"></span>

![DeepSeek-V2 表 4](./deepseek-v2/table-04.png)

**表 4.** 英语开放式对话评测. 对于 AlpacaEval 2.0, 我们使用长度控制胜率作为指标.

此外, 我们基于 AlignBench 评测中文开放式生成能力. 如 [表 5](#table-05) 所示, DeepSeek-V2 Chat (RL) 相比 DeepSeek-V2 Chat (SFT) 展现出轻微优势. 值得注意的是, DeepSeek-V2 Chat (SFT) 以显著优势超越所有开源中文模型. 它在中式推理和语言两方面都显著优于次优的开源模型 Qwen1.5 72B Chat. 此外, DeepSeek-V2 Chat (SFT) 与 DeepSeek-V2 Chat (RL) 都超越了 GPT-4-0613 和 ERNIEBot 4.0, 巩固了我们的模型在支持中文的顶级 LLM 中的地位. 具体来说, DeepSeek-V2 Chat (RL) 在中文语言理解方面表现卓越, 超越了包括 GPT-4-Turbo-1106-Preview 在内的所有模型. 另一方面, DeepSeek-V2 Chat (RL) 的推理能力仍落后于 Erniebot-4.0 和 GPT-4 等巨型模型.

<span id="table-05"></span>

![DeepSeek-V2 表 5](./deepseek-v2/table-05.png)

**表 5.** 由 GPT-4-0613 评分的 AlignBench 排行榜. 模型按总分降序排列. 标有 * 的模型表示我们通过其 API 服务或开放权重模型评测, 而非参考其原论文报告的结果. Erniebot-4.0 与 Moonshot 的后缀表示我们调用其 API 时的时间戳.

<span id="section-4-4"></span>

### 4.4 讨论

**SFT 数据量.** 关于是否需要大规模 SFT 语料的讨论一直是激烈争论的话题. 先前工作 [You24a, Zho24a] 认为少于 1 万实例的 SFT 数据就足以产生令人满意的结果. 然而, 在我们的实验中, 如果使用少于 1 万实例, 我们在 IFEval 基准上观察到显著的性能下降. 一种可能的解释是, 语言模型需要一定量的数据来培养特定技能. 尽管所需的数据量可能随模型规模增大而减少, 但它无法被完全消除. 我们的观察强调了用充足数据赋予 LLM 所需能力的必要性. 此外, SFT 数据的质量也至关重要, 尤其对于涉及写作或开放式问题的任务.

**强化学习的对齐税.** 在人类偏好对齐期间, 我们观察到开放式生成基准上显著的性能提升, 这体现在 AI 和人类评测者给出的分数上. 然而, 我们也注意到一种"对齐税" [Ouy22] 现象, 即对齐过程会对某些标准基准 (如 BBH) 的性能产生负面影响. 为了缓解对齐税, 我们在 RL 阶段在数据处理和优化训练策略方面付出了大量努力, 最终在标准与开放式基准的性能之间取得了可容忍的权衡. 探索如何在不损害通用性能的前提下将模型与人类偏好对齐, 是未来研究的一个有价值方向.

**在线强化学习.** 在我们的偏好对齐实验中, 我们发现在线方法显著优于离线方法. 因此, 我们投入大量精力构建用于对齐 DeepSeek-V2 的在线 RL 框架. 关于在线或离线偏好对齐的结论可能因语境而异, 我们将更全面的比较与分析留待未来工作.

<span id="section-5"></span>

## 5 结论, 局限与未来工作

本文中, 我们介绍 DeepSeek-V2, 一个支持 128K 上下文长度的大型 MoE 语言模型. 除了强大的性能, 它还具有训练经济与推理高效的特点, 这得益于包括 MLA 与 DeepSeekMoE 在内的创新架构. 实践中, 与 DeepSeek 67B 相比, DeepSeek-V2 取得了显著更强的性能, 同时节省了 42.5% 的训练成本, 将 KV cache 减少了 93.3%, 并将最大生成吞吐量提升至 5.76 倍. 评测结果进一步表明, 仅有 21B 激活参数时, DeepSeek-V2 在开源模型中取得顶尖性能, 并成为最强的开源 MoE 模型.

DeepSeek-V2 及其对话版本具有其他 LLM 常见的公认局限, 包括预训练后缺乏持续的知识更新, 可能生成不实信息 (例如未经证实的建议), 以及可能产生幻觉. 此外, 由于我们的数据主要由中文和英文内容组成, 我们的模型在其他语言上的熟练度可能有限. 在超出中文和英语的场景中, 应谨慎使用.

DeepSeek 将以长期主义持续投入开源大模型, 旨在逐步逼近人工通用智能的目标.

- 在我们持续的探索中, 我们致力于设计能够在保持经济训练与推理成本的同时进一步扩展 MoE 模型的方法. 我们下一步的目标是在即将到来的版本中取得与 GPT-4 相当的性能.
- 我们的对齐团队不断努力增强模型, 旨在开发一个不仅有用, 而且对全球用户诚实且安全的模型. 我们的最终目标是将模型的价值与人类价值对齐, 同时尽量减少对人工监督的需求. 通过优先考虑伦理考量和负责任的发展, 我们致力于为社会创造积极且有益的贡献.
- 目前, DeepSeek-V2 仅支持文本模态. 在我们前瞻的议程中, 我们打算让模型支持多模态, 以增强其在更广泛场景中的通用性和实用性.

<span id="section-6"></span>

## 6 贡献与致谢

**研究与工程.** Aixin Liu, Bingxuan Wang, Bo Liu, Chenggang Zhao, Chengqi Deng, Chong Ruan, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangyun Lin, Fuli Luo, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Yang, Haowei Zhang, Honghui Ding, Huajian Xin, Huazuo Gao, Hui Qu, Jianzhong Guo, Jiashi Li, Jingyang Yuan, Junjie Qiu, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Lean Wang, Lecong Zhang, Liang Zhao, Liyue Zhang, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Panpan Huang, Peiyi Wang, Qihao Zhu, Qinyu Chen, Qiushi Du, Ruiqi Ge, Ruizhe Pan, Runxin Xu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shengfeng Ye, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Size Zheng, Tian Pei, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjun Gao, Wentao Zhang, Xiao Bi, Xiaohan Wang, Xiaodong Liu, Xiaokang Chen, Xiaotao Nie, Xin Liu, Xin Xie, Xingkai Yu, Xinyu Yang, Xuan Lu, Xuecheng Su, Y. Wu, Y.K. Li, Y.X. Wei, Yanhong Xu, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yichao Zhang, Yiliang Xiong, Yilong Zhao, Ying He, Yishi Piao, Yixin Dong, Yixuan Tan, Yiyuan Liu, Yongji Wang, Yongqiang Guo, Yuduan Wang, Yuheng Zou, Yuxiang You, Yuxuan Liu, Z.Z. Ren, Zehui Ren, Zhangli Sha, Zhe Fu, Zhenda Xie, Zhewen Hao, Zhihong Shao, Zhuoshu Li, Zihan Wang, Zihui Gu, Zilin Li, Ziwei Xie

**数据标注.** Bei Feng, Hui Li, J.L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R.J. Chen, R.L. Jin, Ruyi Chen, S.S. Li, Shuang Zhou, Tian Yuan, Tianyu Sun, X.Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xiaowen Sun, Xiaoxiang Wang, Xinnan Song, Xinyi Zhou, Y.X. Zhu, Yanhong Xu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang

**商务与合规.** Bin Wang, Dongjie Ji, Jian Liang, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Shaoqing Wu, Shengfeng Ye, T. Wang, W.L. Xiao, Wei An, Xianzu Wang, Ying Tang, Yukun Zha, Yuting Yan, Zhen Zhang, Zhiniu Wen

在每个角色内, 作者按名字的字母顺序排列. 特别地, Huazuo Gao 与 Wangding Zeng 在 MLA 架构的研究中做出了关键创新. 此外, 我们感谢 Jianlin Su 在位置编码方面给予的有益讨论. 我们感谢所有为 DeepSeek-V2 作出贡献但未在本文中提及的人. DeepSeek 相信, 在通往 AGI 的道路上, 创新, 新颖与好奇至关重要.

<span id="section-7"></span>

## 7 DeepSeek-V2-Lite: 配备 MLA 与 DeepSeekMoE 的 16B 模型

<span id="section-7-1"></span>

### 7.1 模型描述

**架构.** DeepSeek-V2-Lite 有 27 层, 隐藏维度为 2048. 它也采用 MLA, 有 16 个注意力头, 每个头的维度为 128. 其 KV 压缩维度为 512, 但与 DeepSeek-V2 略有不同, 它不压缩查询. 对于解耦查询与键, 它每头维度为 64. DeepSeek-V2-Lite 也采用 DeepSeekMoE, 除第一层外所有 FFN 都被替换为 MoE 层. 每个 MoE 层包含 2 个共享专家和 64 个路由专家, 其中每个专家的中间隐藏维度为 1408. 在路由专家中, 每个 token 会激活 6 个专家. 在此配置下, DeepSeek-V2-Lite 总计包含 15.7B 参数, 其中每个 token 激活 2.4B.

<span id="table-06"></span>

![DeepSeek-V2 表 6](./deepseek-v2/table-06.png)

**表 6.** DeepSeek-V2-Lite, DeepSeekMoE 16B 与 DeepSeek 7B 的性能.

**训练细节.** DeepSeek-V2-Lite 同样在 DeepSeek-V2 的同一预训练语料上从头训练, 该语料未受到任何 SFT 数据污染. 它使用 AdamW 优化器, 超参数设为 $\beta_1=0.9$, $\beta_2=0.95$, 且 $\mathrm{weight\_decay}=0.1$. 学习率采用预热加阶梯衰减策略进行调度. 最初, 学习率在前 2K 步内从 0 线性增至最大值. 随后, 学习率在大约训练了 80% 的 token 后乘以 0.316, 并在大约训练了 90% 的 token 后再次乘以 0.316. 最大学习率设为 $4.2 \times 10^{-4}$, 梯度裁剪范数设为 1.0. 我们未对它采用批大小调度策略, 它以恒定的批大小 4608 个序列训练. 在预训练期间, 我们将最大序列长度设为 4K, 并在 5.7T token 上训练 DeepSeek-V2-Lite. 我们利用流水线并行将其不同层部署到不同设备上, 但对每一层, 所有专家都会被部署在同一个设备上. 因此, 我们仅采用一个小的专家级均衡损失, $\alpha_{1}=0.001$, 并对其不采用设备级均衡损失与通信均衡损失. 在预训练之后, 我们还对 DeepSeek-V2-Lite 进行长上下文扩展与 SFT, 得到一个名为 DeepSeek-V2-Lite Chat 的对话模型.

<span id="table-07"></span>

![DeepSeek-V2 表 7](./deepseek-v2/table-07.png)

**表 7.** DeepSeek-V2-Lite Chat, DeepSeekMoE 16B Chat 与 DeepSeek 7B Chat 的性能.

<span id="section-7-2"></span>

### 7.2 性能评测

**基础模型.** 我们在 [表 6](#table-06) 中评测 DeepSeek-V2-Lite 的性能, 并将其与我们先前的小型基础模型比较. DeepSeek-V2-Lite 展现出压倒性的性能优势, 尤其在推理, 编码和数学方面.

**对话模型.** 我们在 [表 7](#table-07) 中评测 DeepSeek-V2-Lite Chat 的性能, 并将其与我们先前的小型对话模型比较. DeepSeek-V2-Lite 也以显著优势超越我们先前的小型对话模型.

<span id="section-8"></span>

## 8 MLA 的完整公式

为了展示 MLA 的完整计算过程, 我们在下面给出其完整公式:

$$
\begin{aligned}
    \mathbf{c}_{t}^{Q} &= W^{\mathit{DQ}} \mathbf{h}_{t}, \\
    [\mathbf{q}_{t, 1}^{C};\mathbf{q}_{t, 2}^{C};...;\mathbf{q}_{t, n_{h}}^{C}] = \mathbf{q}_{t}^{C} &= W^{\mathit{UQ}} \mathbf{c}_{t}^{Q}, \\
    [\mathbf{q}_{t, 1}^{R};\mathbf{q}_{t, 2}^{R};...;\mathbf{q}_{t, n_{h}}^{R}] = \mathbf{q}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{QR}}} \mathbf{c}_{t}^{Q}), \\
    \mathbf{q}_{t, i} &= [\mathbf{q}_{t, i}^{C}; \mathbf{q}_{t, i}^{R}], \\
    \mathbf{c}_{t}^{\mathit{KV}} &= W^{\mathit{DKV}} \mathbf{h}_{t}, \\
    [\mathbf{k}_{t, 1}^{C};\mathbf{k}_{t, 2}^{C};...;\mathbf{k}_{t, n_{h}}^{C}] = \mathbf{k}_{t}^{C} &= W^{\mathit{UK}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{k}_{t}^{R} &= \mathop{\mathrm{RoPE}}({W^{\mathit{KR}}} \mathbf{h}_{t}), \\
    \mathbf{k}_{t, i} &= [\mathbf{k}_{t, i}^{C}; \mathbf{k}_{t}^{R}], \\
    [\mathbf{v}_{t, 1}^{C};\mathbf{v}_{t, 2}^{C};...;\mathbf{v}_{t, n_{h}}^{C}] = \mathbf{v}_{t}^{C} &= W^{\mathit{UV}} \mathbf{c}_{t}^{\mathit{KV}}, \\
    \mathbf{o}_{t, i} &= \sum_{j=1}^{t} \mathop{\mathrm{Softmax}}_j(\frac{\mathbf{q}_{t, i}^\top \mathbf{k}_{j, i}}{\sqrt{d_{h} + d_{h}^{R}}}) \mathbf{v}_{j, i}^{C}, \\
    \mathbf{u}_{t} &= W^{O} [\mathbf{o}_{t, 1};\mathbf{o}_{t, 2};...;\mathbf{o}_{t, n_{h}}],
\end{aligned}
$$

其中蓝色的方框向量需要在生成时缓存. 推理期间, 朴素公式需要从 $\mathbf{c}_{t}^{\mathit{KV}}$ 恢复 $\mathbf{k}_{t}^{C}$ 与 $\mathbf{v}_{t}^{C}$ 以进行注意力. 幸运的是, 由于矩阵乘法的结合律, 我们可以将 $W^{\mathit{UK}}$ 吸收进 $W^{\mathit{UQ}}$, 将 $W^{\mathit{UV}}$ 吸收进 $W^{O}$. 因此, 我们无需为每个查询计算出键和值. 通过这一优化, 我们避免了推理期间重新计算 $\mathbf{k}_{t}^{C}$ 与 $\mathbf{v}_{t}^{C}$ 带来的计算开销.

<span id="section-9"></span>

## 9 注意力机制消融

<span id="section-9-1"></span>

### 9.1 MHA, GQA 与 MQA 的消融

我们在 [表 8](#table-08) 中展示了分别在四个困难基准上采用 MHA, GQA 与 MQA 的 7B 稠密模型的评测结果. 这三个模型都训练了 1.33T token, 除注意力机制外共享相同的架构. 此外, 为公平比较, 我们通过调整层数将其参数数量对齐到约 7B 左右. 从表中我们可以发现, MHA 在这些基准上相比 GQA 与 MQA 展现出显著优势.

<span id="table-08"></span>

![DeepSeek-V2 表 8](./deepseek-v2/table-08.png)

**表 8.** 分别采用 MHA, GQA 与 MQA 的 7B 稠密模型的比较. MHA 在困难基准上相比 GQA 与 MQA 展现出显著优势.

<span id="section-9-2"></span>

### 9.2 MLA 与 MHA 的比较

在 [表 9](#table-09) 中, 我们展示了分别在四个困难基准上装备 MLA 与 MHA 的 MoE 模型的评测结果. 为得出可靠结论, 我们在两个规模上训练并评测模型. 两个小型 MoE 模型总计约 16B 参数, 我们在 1.33T token 上训练它们. 两个大型 MoE 模型总计约 250B 参数, 我们在 420B token 上训练它们. 同样, 两个小型 MoE 模型与两个大型 MoE 模型除注意力机制外分别共享相同的架构. 从表中我们可以观察到, MLA 表现出优于 MHA 的性能. 更重要的是, MLA 所需的 KV cache 显著少于 MHA (小型 MoE 模型为 14%, 大型 MoE 模型为 4%).

<span id="table-09"></span>

![DeepSeek-V2 表 9](./deepseek-v2/table-09.png)

**表 9.** 四个困难基准上 MLA 与 MHA 的比较. DeepSeek-V2 表现出优于 MHA 的性能, 但所需的 KV cache 显著更少.

<span id="section-10"></span>

## 10 关于预训练数据去偏的讨论

在预训练数据准备期间, 我们识别并过滤掉具有争议的内容, 例如受地域文化影响的价值观, 以避免模型在这些有争议的话题上表现出不必要的主观偏见. 因此, 我们观察到 DeepSeek-V2 在与特定地域文化密切相关的测试集上表现略差. 例如, 在 MMLU 上评测时, 尽管 DeepSeek-V2 在大多数测试集上取得了与 Mixtral 8x22B 等竞争对手相当或更优的性能, 但它仍在主要与美国价值观相关的 Humanity-Moral 子集上落后.

进一步, 我们对这一子集进行了人工分析. 三位受过良好教育的人工标注者对 MMLU Humanity-Moral 子集中的 420 个道德场景进行了独立标注. 然后, 我们计算了它们的标注与真实标签之间的一致性. 如 [表 10](#table-10) 所示, 三位人工标注者与真实标签之间的一致性都较低. 因此, 我们将 DeepSeek-V2 在这些价值敏感测试集上的异常性能归因于我们对预训练语料进行去偏的努力.

<span id="table-10"></span>

![DeepSeek-V2 表 10](./deepseek-v2/table-10.png)

**表 10.** 三位受过良好教育的人工标注者对 MMLU Humanity-Moral 子集中的 420 个道德场景进行独立标注, DeepSeek-V2 及其竞争模型在这些场景上展现出性能不一致. 三位标注者与真实标签之间的一致性都较低. 这表明 Humanity-Moral 子集的答案可能因特定地域文化而具有争议.

<span id="section-11"></span>

## 11 数学与代码的额外评测

该评测采用 SC-Math6 语料, 其中包含数千道中文数学题. DeepSeek-V2 Chat (RL) 超越了包括开源与闭源模型在内的所有中文 LLM.

<span id="table-11"></span>

![DeepSeek-V2 表 11](./deepseek-v2/table-11.png)

**表 11.** SC-Math6 模型推理水平. "R Level" 代表 Reasoning Level (推理水平), "Comp. Score" 代表 Comprehensive Score (综合得分), "Reas. Steps Score" 代表 Reasoning Steps Score (推理步骤得分), "OvrAcc Score" 代表 Overall Accuracy Score (总体准确率得分).

我们进一步在 [图 5](#figure-05) 中分享 HumanEval 与 LiveCodeBench 上的更多结果, 其中 LiveCodeBench 的题目选自 2023 年 9 月 1 日至 2024 年 4 月 1 日期间. 如图所示, DeepSeek-V2 Chat (RL) 在 LiveCodeBench 上展现出相当强的熟练度, 其 Pass@1 分数甚至超越了一些巨型模型. 这一性能凸显了 DeepSeek-V2 Chat (RL) 在处理实时编码任务方面的强大能力.

<span id="figure-05"></span>

![DeepSeek-V2 图 5](./deepseek-v2/figure-05.png)

**图 5.** HumanEval 与 LiveCodeBench 上的评测结果. LiveCodeBench 的题目选自 2023 年 9 月 1 日至 2024 年 4 月 1 日期间.

<span id="section-12"></span>

## 12 评测格式

我们在 [表 12](#table-12)-[表 37](#table-37) 中分别呈现每个基准的评测格式.

<span id="table-12"></span>

![DeepSeek-V2 表 12](./deepseek-v2/table-12.png)

**表 12.** AGIEval 的一个示例.

<span id="table-13"></span>

![DeepSeek-V2 表 13](./deepseek-v2/table-13.png)

**表 13.** ARC 的一个示例.

<span id="table-14"></span>

![DeepSeek-V2 表 14](./deepseek-v2/table-14.png)

**表 14.** BBH 的一个示例.

<span id="table-15"></span>

![DeepSeek-V2 表 15](./deepseek-v2/table-15.png)

**表 15.** C-Eval 的一个示例.

<span id="table-16"></span>

![DeepSeek-V2 表 16](./deepseek-v2/table-16.png)

**表 16.** CHID 的一个示例.

<span id="table-17"></span>

![DeepSeek-V2 表 17](./deepseek-v2/table-17.png)

**表 17.** CLUEWSC 的一个示例.

<span id="table-18"></span>

![DeepSeek-V2 表 18](./deepseek-v2/table-18.png)

**表 18.** CMMLU 的一个示例.

<span id="table-19"></span>

![DeepSeek-V2 表 19](./deepseek-v2/table-19.png)

**表 19.** CMRC 的一个示例.

<span id="table-20"></span>

![DeepSeek-V2 表 20](./deepseek-v2/table-20.png)

**表 20.** CRUXEval-I 的一个示例.

<span id="table-21"></span>

![DeepSeek-V2 表 21](./deepseek-v2/table-21.png)

**表 21.** AGIEval 英语子集的一个示例.

<span id="table-22"></span>

![DeepSeek-V2 表 22](./deepseek-v2/table-22.png)

**表 22.** CRUXEval-O 的一个示例.

<span id="table-23"></span>

![DeepSeek-V2 表 23](./deepseek-v2/table-23.png)

**表 23.** DROP 的一个示例.

<span id="table-24"></span>

![DeepSeek-V2 表 24](./deepseek-v2/table-24.png)

**表 24.** GSM8K 的一个示例.

<span id="table-25"></span>

![DeepSeek-V2 表 25](./deepseek-v2/table-25.png)

**表 25.** HumanEval 的一个示例.

<span id="table-26"></span>

![DeepSeek-V2 表 26](./deepseek-v2/table-26.png)

**表 26.** MATH 的一个示例.

<span id="table-27"></span>

![DeepSeek-V2 表 27](./deepseek-v2/table-27.png)

**表 27.** MBPP 的一个示例.

<span id="table-28"></span>

![DeepSeek-V2 表 28](./deepseek-v2/table-28.png)

**表 28.** MMLU 的一个示例.

<span id="table-29"></span>

![DeepSeek-V2 表 29](./deepseek-v2/table-29.png)

**表 29.** NaturalQuestions 的一个示例.

<span id="table-30"></span>

![DeepSeek-V2 表 30](./deepseek-v2/table-30.png)

**表 30.** PIQA 的一个示例.

<span id="table-31"></span>

![DeepSeek-V2 表 31](./deepseek-v2/table-31.png)

**表 31.** C3 的一个示例.

<span id="table-32"></span>

![DeepSeek-V2 表 32](./deepseek-v2/table-32.png)

**表 32.** RACE 的一个示例.

<span id="table-33"></span>

![DeepSeek-V2 表 33](./deepseek-v2/table-33.png)

**表 33.** CMath 的一个示例.

<span id="table-34"></span>

![DeepSeek-V2 表 34](./deepseek-v2/table-34.png)

**表 34.** TriviaQA 的一个示例.

<span id="table-35"></span>

![DeepSeek-V2 表 35](./deepseek-v2/table-35.png)

**表 35.** CCPM 的一个示例.

<span id="table-36"></span>

![DeepSeek-V2 表 36](./deepseek-v2/table-36.png)

**表 36.** AGIEval 英语子集的一个示例.

<span id="table-37"></span>

![DeepSeek-V2 表 37](./deepseek-v2/table-37.png)

**表 37.** CCWSC 的一个示例.
