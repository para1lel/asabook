---
title: 'DeepSeekMoE'
createTime: 2026/09/06 19:41:12
permalink: /papers/deepseek-moe/
---

> [Damai Dai](https://dblp.org/pid/199/2097.html) [+internship], [Chengqi Deng](https://dblp.org/pid/255/4939.html), [Chenggang Zhao](https://dblp.org/pid/254/2607.html) [+internship], [R.X. Xu](https://dblp.org/pid/267/5291.html), [Huazuo Gao](https://dblp.org/pid/366/3356.html), [Deli Chen](https://dblp.org/pid/50/2637.html), [Jiashi Li](https://dblp.org/pid/241/9364.html), [Wangding Zeng](https://dblp.org/pid/315/5319.html), [Xingkai Yu](https://dblp.org/pid/257/4432.html) [+internship], [Y. Wu](https://dblp.org/pid/22/0-24.html), [Zhenda Xie](https://dblp.org/pid/239/8676.html), [Y.K. Li](https://dblp.org/pid/16/8783.html), [Panpan Huang](https://dblp.org/pid/19/6338.html), [Fuli Luo](https://dblp.org/pid/220/4216.html), [Chong Ruan](https://dblp.org/pid/159/9956.html), [Zhifang Sui](https://dblp.org/pid/22/5834.html), [Wenfeng Liang](https://dblp.org/pid/59/9456.html). 2024 年 1 月 11 日首次提交至 arXiv, 当前版本为 v1. 2024 年 8 月发表于 *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, 页码 1280-1297. [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066v1). <a href="/paper/deepseek-moe.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [ACL 2024](https://aclanthology.org/2024.acl-long.70/). [DOI](https://doi.org/10.18653/v1/2024.acl-long.70). [TeX 源码](https://export.arxiv.org/e-print/2401.06066v1). [代码与模型](https://github.com/deepseek-ai/DeepSeek-MoE). 精确的印刷版式与参考文献以原始 PDF 为准.

[+internship]: 在 DeepSeek-AI 实习期间完成的贡献。

## 摘要

在大语言模型时代，混合专家（Mixture-of-Experts，MoE）是一种很有前景的架构，能够在扩大模型参数规模时控制计算成本。不过，GShard 等传统 MoE 架构从 $N$ 个专家中激活得分最高的 $K$ 个，难以保证专家特化，即每个专家都掌握互不重叠且聚焦的知识。为此，我们提出以实现极致专家特化为目标的 **DeepSeekMoE** 架构。它包含两项主要策略：(1) 将专家细分为 $mN$ 个，并从中激活 $mK$ 个，从而能更灵活地组合被激活的专家；(2) 分离出 $K_s$ 个专家作为共享专家，用来捕获公共知识并缓解路由专家之间的冗余。我们先从参数规模适中的 2B 模型入手，证明 DeepSeekMoE 2B 的性能可与 GShard 2.9B 相当，而后者的专家参数量和计算量均为前者的 1.5$\times$。此外，DeepSeekMoE 2B 的性能已接近总参数量相同的稠密模型，而该稠密模型给出了 MoE 模型的性能上界。随后，我们将 DeepSeekMoE 扩展到 16B 参数，并表明它仅用约 40% 的计算量便取得了与 LLaMA2 7B 相当的性能。最后，我们将 DeepSeekMoE 初步扩展至 145B 参数；结果仍然持续验证了它相较 GShard 架构的显著优势，并表明它仅用 28.5%（甚至可能是 18.2%）的计算量便能取得与 DeepSeek 67B 相当的性能。

<span id="figure-01"></span>

![图 1. DeepSeekMoE 16B 与开源模型在 Open LLM Leaderboard 上的比较。红色虚线由除 DeepSeekMoE 16B 外所有模型的数据点线性拟合得到。DeepSeekMoE 16B 始终大幅优于激活参数量相近的模型，并取得了与 LLaMA2 7B 相当的性能；后者的激活参数量约为前者的 2.5 倍。](./deepseek-moe/figure-01.png)

**图 1.** DeepSeekMoE 16B 与开源模型在 Open LLM Leaderboard 上的比较。红色虚线由除 DeepSeekMoE 16B 外所有模型的数据点线性拟合得到。DeepSeekMoE 16B 始终大幅优于激活参数量相近的模型，并取得了与 LLaMA2 7B 相当的性能；后者的激活参数量约为前者的 2.5 倍。

<span id="section-1"></span>

## 1 引言

近期的研究和实践通过实验表明，在训练数据充足时，增加参数量与计算预算来扩展语言模型，可以得到性能明显更强的模型 [Bro20, Ope23, Tou23, Hof22]。不过也必须承认，将模型扩展到极大规模会带来极高的计算成本。考虑到这笔巨大开销，混合专家（MoE）架构 [Jac91, Jor94, Sha17] 已成为一种常用方案。它可以扩大参数规模，同时把计算成本维持在适中的水平。近来，把 MoE 架构用于 Transformer [Vas17d]，已成功将语言模型扩展至相当大的规模 [Fed22, Lep20, Du22, Zop22]，并取得了出色的性能。这些成果表明 MoE 语言模型具有很大的潜力。

尽管 MoE 架构前景良好，现有架构仍可能存在知识混杂与知识冗余问题，限制了专家特化，即每个专家掌握互不重叠且聚焦的知识。传统 MoE 架构用 MoE 层替换 Transformer 中的前馈网络（FFN）。每个 MoE 层由多个专家组成，每个专家的结构都与标准 FFN 相同，而每个词元会被分配给一个 [Fed22] 或两个 [Lep20] 专家。这种架构可能产生两个问题：(1) **知识混杂：** 现有 MoE 实践往往只使用有限数量的专家（例如 8 个或 16 个），因此分配给某个专家的词元很可能涉及多种知识。于是，该专家会试图在参数中容纳差异很大的知识，而这些知识很难同时得到利用。(2) **知识冗余：** 分配给不同专家的词元可能需要相同的公共知识。因此，多个专家可能会在各自参数中学到相同的知识，造成专家参数冗余。这两个问题共同阻碍了现有 MoE 实践中的专家特化，使其无法达到 MoE 模型的理论性能上界。

针对上述问题，我们提出 **DeepSeekMoE**，一种专为实现极致专家特化而设计的新型 MoE 架构。该架构包含两项主要策略：(1) **细粒度专家划分：** 在保持参数量不变的前提下，我们通过拆分 FFN 的中间隐藏维度，将专家划分得更细。相应地，在计算成本不变的情况下，我们也会激活更多细粒度专家，使被激活专家的组合更灵活、更适应任务。细粒度划分使不同知识可以拆得更细，并由不同专家更精确地学习，因此每个专家都能保持更高的特化程度。被激活专家组合的灵活性提高，也有助于更准确、更有针对性地获取知识。(2) **共享专家隔离：** 我们分离出一部分始终激活的专家作为共享专家，用来捕获和整合不同上下文中的公共知识。把公共知识压缩到这些共享专家后，其他路由专家之间的冗余会得到缓解。这能提高参数效率，并让每个路由专家专注于不同方面，从而保持特化。DeepSeekMoE 的这些架构改进使我们有机会训练出参数高效、各专家高度特化的 MoE 语言模型。

我们先从参数规模适中的 2B 模型开始，验证 DeepSeekMoE 架构的优势。我们在涵盖不同任务的 12 个零样本或少样本基准上进行评估。实验结果显示，DeepSeekMoE 2B 大幅超过 GShard 2B [Lep20]，甚至可以匹敌更大的 MoE 模型 GShard 2.9B；后者的专家参数量和计算量均为前者的 1.5$\times$。值得注意的是，DeepSeekMoE 2B 的性能已接近参数量相同的稠密模型，而该稠密模型给出了 MoE 语言模型的严格上界。为获得更深入的认识，我们对 DeepSeekMoE 进行了详尽的消融实验和专家特化分析。这些研究验证了细粒度专家划分和共享专家隔离的有效性，并为 DeepSeekMoE 能够实现高度专家特化这一结论提供了实验证据。

借助这一架构，我们随后把模型参数扩展到 16B，并在包含 2T 词元的大规模语料库上训练 DeepSeekMoE 16B。评估结果显示，DeepSeekMoE 16B 仅用约 40% 的计算量，就取得了与在同一 2T 语料库上训练的稠密模型 DeepSeek 7B [Dee24e] 相当的性能。我们还将 DeepSeekMoE 与开源模型比较；评估表明，DeepSeekMoE 16B 始终大幅优于激活参数量相近的模型，并取得了与 LLaMA2 7B [Tou23a] 相当的性能，而后者的激活参数量约为前者的 2.5 倍。[图 1](#figure-01) 给出了 Open LLM Leaderboard [+1] 上的评估结果。此外，我们还通过监督微调（SFT）进行对齐，把模型转化为聊天模型。评估结果表明，在聊天场景中，DeepSeekMoE Chat 16B 也取得了与 DeepSeek Chat 7B 和 LLaMA2 SFT 7B 相当的性能。受这些结果鼓舞，我们进一步尝试将 DeepSeekMoE 初步扩展到 145B。实验结果仍然持续验证了它相较 GShard 架构的显著优势。此外，它仅用 28.5%（甚至可能是 18.2%）的计算量，就取得了与 DeepSeek 67B 相当的性能。

我们的贡献概括如下：

- **架构创新。** 我们提出 DeepSeekMoE，一种以实现极致专家特化为目标的新型 MoE 架构，它采用细粒度专家划分和共享专家隔离两项主要策略。

- **实验验证。** 我们进行了广泛实验，从实验上验证 DeepSeekMoE 架构的有效性。实验结果证实 DeepSeekMoE 2B 具有很高的专家特化程度，并表明 DeepSeekMoE 2B 的性能可以接近 MoE 模型的上界。

- **可扩展性。** 我们扩展 DeepSeekMoE 并训练出一个 16B 模型，结果表明 DeepSeekMoE 16B 仅用约 40% 的计算量，便取得了与 DeepSeek 7B 和 LLaMA2 7B 相当的性能。我们还初步尝试将 DeepSeekMoE 扩展到 145B，结果显示它相较 GShard 架构仍有稳定优势，性能也与 DeepSeek 67B 相当。

- **MoE 对齐。** 我们成功对 DeepSeekMoE 16B 进行了监督微调，得到一个对齐的聊天模型，展现出 DeepSeekMoE 16B 的适应能力与通用性。

- **公开发布。** 本着开放研究的精神，我们向公众发布 DeepSeekMoE 16B 的模型检查点。值得一提的是，该模型无需量化，便可部署在一张配有 40GB 显存的 GPU 上。

<span id="section-2"></span>

## 2 预备知识：用于 Transformer 的混合专家

我们先介绍一种常用于 Transformer 语言模型的通用 MoE 架构。标准 Transformer 语言模型由 $L$ 层标准 Transformer 块堆叠而成，每个块可以表示为：

$$
\begin{aligned}
    \mathbf{u}_{1:T}^{l} &= \operatorname{Self-Att}\left( \mathbf{h}_{1:T}^{l-1} \right) + \mathbf{h}_{1:T}^{l-1}, \\
    \mathbf{h}_{t}^{l} &= \operatorname{FFN}\left( \mathbf{u}_{t}^{l} \right) + \mathbf{u}_{t}^{l},
\end{aligned}
$$

其中，$T$ 表示序列长度，$\operatorname{Self-Att}(\cdot)$ 表示自注意力模块，$\operatorname{FFN}(\cdot)$ 表示前馈网络（FFN），$\mathbf{u}_{1:T}^{l} \in \mathbb{R}^{T \times d}$ 是第 $l$ 个注意力模块之后所有词元的隐藏状态，$\mathbf{h}_{t}^{l} \in \mathbb{R}^{d}$ 是第 $l$ 个 Transformer 块之后第 $t$ 个词元的输出隐藏状态。为简洁起见，上述公式省略了层归一化。

构建 MoE 语言模型的一种典型做法，是按指定间隔用 MoE 层替换 Transformer 中的 FFN [Fed22, Lep20, Du22, Zop22]。一个 MoE 层由多个专家组成，每个专家的结构都与标准 FFN 相同。然后，每个词元会被分配给一个 [Fed22] 或两个 [Lep20] 专家。如果第 $l$ 个 FFN 被替换为 MoE 层，其输出隐藏状态 $\mathbf{h}_{t}^{l}$ 可表示为：

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{N} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leqslant j \leqslant N \}, K), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right),
\end{aligned}
$$

其中，$N$ 表示专家总数，$\operatorname{FFN}_{i}(\cdot)$ 是第 $i$ 个专家 FFN，$g_{i,t}$ 表示第 $i$ 个专家的门控值，$s_{i,t}$ 表示词元与专家之间的亲和度，$\operatorname{Topk}(\cdot, K)$ 表示为第 $t$ 个词元计算出的全部 $N$ 个专家亲和度分数中最高的 $K$ 个所构成的集合，$\mathbf{e}_{i}^{l}$ 是第 $l$ 层中第 $i$ 个专家的中心。注意，$g_{i,t}$ 是稀疏的，即 $N$ 个门控值中仅有 $K$ 个非零。这一稀疏性质保证了 MoE 层的计算效率，也就是说，每个词元只会被分配给 $K$ 个专家，并只在这些专家中计算。此外，为简洁起见，上述公式同样省略了层归一化操作。

<span id="figure-02"></span>

![图 2. DeepSeekMoE 示意图。子图 (a) 展示采用传统 top-2 路由策略的 MoE 层。子图 (b) 展示细粒度专家划分策略。随后，子图 (c) 展示共享专家隔离策略的整合，构成完整的 DeepSeekMoE 架构。值得注意的是，在这三种架构中，专家参数量与计算成本保持不变。](./deepseek-moe/figure-02.png)

**图 2.** DeepSeekMoE 示意图。子图 (a) 展示采用传统 top-2 路由策略的 MoE 层。子图 (b) 展示细粒度专家划分策略。随后，子图 (c) 展示共享专家隔离策略的整合，构成完整的 DeepSeekMoE 架构。值得注意的是，在这三种架构中，专家参数量与计算成本保持不变。

<span id="section-3"></span>

## 3 DeepSeekMoE 架构

在[第 2 节](#section-2)所述通用 MoE 架构的基础上，我们提出专为发掘专家特化潜力而设计的 DeepSeekMoE。如[图 2](#figure-02) 所示，该架构包含两项主要策略：细粒度专家划分和共享专家隔离。两项策略都用于提高专家特化程度。

<span id="section-3-1"></span>

### 3.1 细粒度专家划分

当专家数量有限时，分配给某个专家的词元更可能涉及多种知识。因此，该专家会试图在参数中学习差异很大的知识，而这些知识很难同时得到利用。不过，如果每个词元可以路由到更多专家，不同知识就有机会被拆开，分别由不同专家学习。在这种情况下，每个专家仍能维持较高的特化程度，使知识更集中地分布在不同专家中。

为了实现这一目标，我们在保持专家参数量和计算成本不变的同时，把专家划分得更细。更细的专家划分使被激活专家的组合更灵活、更适应任务。具体来说，在[图 2](#figure-02)(a) 所示的典型 MoE 架构上，我们将 FFN 中间隐藏维度缩小到原来的 $\frac{1}{m}$，从而把每个专家 FFN 划分为 $m$ 个较小的专家。由于每个专家变小，我们也相应地把激活专家数量提高到原来的 $m$ 倍，以保持相同的计算成本，如[图 2](#figure-02)(b) 所示。采用细粒度专家划分后，一个 MoE 层的输出可表示为：

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leqslant j \leqslant mN \}, mK), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right),
\end{aligned}
$$

其中，专家参数总量等于标准 FFN 参数量的 $N$ 倍，$mN$ 表示细粒度专家总数。采用细粒度专家划分策略后，非零门控的数量也会增至 $mK$。

从组合的角度看，细粒度专家划分策略大幅提高了被激活专家的组合灵活性。举例来说，考虑 $N=16$ 的情况。典型的 top-2 路由策略可以产生 $\binom{16}{2}=120$ 种组合。相比之下，如果把每个专家拆成 $4$ 个较小的专家，细粒度路由策略就能产生 $\binom{64}{8}=4,426,165,368$ 种可能的组合。组合灵活性的激增提高了更准确、更有针对性地获取知识的可能性。

<span id="section-3-2"></span>

### 3.2 共享专家隔离

使用传统路由策略时，分配给不同专家的词元可能需要某些相同的知识或信息。因此，多个专家可能会在各自参数中学到相同的知识，从而造成专家参数冗余。不过，如果有专门捕获和整合不同上下文中公共知识的共享专家，其他路由专家之间的参数冗余就会减轻。冗余的减少有助于得到一个参数效率更高、专家更为特化的模型。

为此，除了细粒度专家划分策略外，我们还进一步分离出 $K_{s}$ 个专家作为共享专家。无论路由器模块的输出如何，每个词元都会被确定性地分配给这些共享专家。为了保持计算成本不变，其他路由专家中的激活数量会减少 $K_{s}$ 个，如[图 2](#figure-02)(c) 所示。整合共享专家隔离策略后，完整 DeepSeekMoE 架构中的 MoE 层可表示为：

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{K_{s}} {\operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} + \sum_{i=K_{s} + 1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | K_{s} + 1 \leqslant j \leqslant mN \}, mK - K_{s}), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right).
\end{aligned}
$$

最终，在 DeepSeekMoE 中，共享专家数量为 $K_{s}$，路由专家总数为 $mN - K_{s}$，非零门控数量为 $mK - K_{s}$。

值得一提的是，共享专家隔离的原型应归功于 [Raj22]。主要区别在于，他们从工程角度推导这一策略，而我们从算法角度出发。

<span id="section-3-3"></span>

### 3.3 负载均衡考量

自动学习的路由策略可能遇到负载不均衡问题，主要表现为两个缺陷。第一，存在路由坍缩 [Sha17] 的风险，即模型始终只选择少数几个专家，使其他专家无法得到充分训练。第二，如果专家分布在多个设备上，负载不均衡会加剧计算瓶颈。

**专家级均衡损失。** 为降低路由坍缩的风险，我们也采用专家级均衡损失。均衡损失的计算方式如下：

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{ExpBal}} & = \alpha_1 \sum_{i=1}^{N^{\prime}}{f_i P_i}, \\
    f_i & = \frac{N^{\prime}}{K^{\prime}T} \sum_{t=1}^{T}{ \mathbb{1}( \text{Token } t \text{ selects Expert } i )}, \\
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s_{i,t}},
\end{aligned}
$$

其中，$\alpha_1$ 是称为专家级均衡因子的超参数；为简洁起见，$N^{\prime}$ 等于 $(mN - K_s)$，$K^{\prime}$ 等于 $(mK - K_s)$。$\mathbb{1}(\cdot)$ 表示指示函数。

**设备级均衡损失。** 除专家级均衡损失外，我们还引入设备级均衡损失。为了缓解计算瓶颈，并无必要在专家级施加严格的均衡约束，因为对负载均衡施加过度约束会损害模型性能。我们的主要目标是保证设备间计算均衡。如果把所有路由专家划分为 $D$ 个组 $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D \}$，并将每个组部署在一台设备上，则设备级均衡损失计算如下：

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{DevBal}} & = \alpha_{2} \sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}, \\
    f_i^{\prime} & = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i}{ f_j }, \\
    P_i^{\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

其中，$\alpha_{2}$ 是称为设备级均衡因子的超参数。实际使用时，我们设置较小的专家级均衡因子来降低路由坍缩风险，同时设置较大的设备级均衡因子来促进设备间计算均衡。

<span id="section-4"></span>

## 4 验证实验

<span id="section-4-1"></span>

### 4.1 实验设置

<span id="section-4-1-1"></span>

#### 4.1.1 训练数据与分词

我们的训练数据取样自 DeepSeek-AI 构建的大规模多语言语料库。该语料库以英语和中文为主，也涵盖其他语言。它汇集了多种来源，包括网络文本、数学材料、代码脚本、已出版文献以及其他各类文本材料。在验证实验中，我们从语料库中抽取一个包含 100B 词元的子集来训练模型。分词方面，我们使用 HuggingFace Tokenizer [+2] 工具，在训练语料的一个较小子集上训练字节对编码（BPE）[Sen16] 分词器。验证实验采用词表大小为 8K 的分词器；训练更大的模型时，词表大小也会随之增加。

<span id="section-4-1-2"></span>

#### 4.1.2 基础设施

我们的实验基于高效、轻量的训练框架 HAI-LLM [Hig23] 开展，该框架集成了多种并行策略，包括张量并行 [Sho19, Nar21, Kor22]、ZeRO 数据并行 [Raj20]、PipeDream 流水线并行 [Har18]，以及更具体地说，通过组合数据并行与张量并行实现的专家并行 [Lep20]。为了优化性能，我们使用 CUDA 和 Triton [Til19] 开发 GPU 内核，用于门控算法及不同专家中线性层的融合计算。

所有实验均在配有 NVIDIA A100 或 H800 GPU 的集群上运行。A100 集群的每个节点包含 8 张 GPU，通过 NVLink 桥两两连接。H800 集群的每个节点同样配有 8 张 GPU，节点内使用 NVLink 和 NVSwitch 互连。A100 和 H800 集群均使用 InfiniBand 互连来支持节点间通信。

<span id="section-4-1-3"></span>

#### 4.1.3 超参数

**模型设置。** 在验证实验中，我们把 Transformer 层数设为 9，隐藏维度设为 1280。模型采用多头注意力机制，共有 10 个注意力头，每个头的维度为 128。初始化时，所有可学习参数都以 0.006 的标准差随机初始化。我们用 MoE 层替换全部 FFN，并保证专家总参数量等于标准 FFN 参数量的 16 倍。此外，包括共享专家参数和被激活路由专家参数在内的激活专家参数量，保持为标准 FFN 参数量的 2 倍。在此配置下，每个 MoE 模型的总参数量约为 2B，激活参数量约为 0.3B。

**训练设置。** 我们使用 AdamW 优化器 [Los17]，超参数设为 $\beta_1=0.9$、$\beta_2=0.95$ 和 $\mathrm{weight\_decay}=0.1$。学习率采用预热加阶梯衰减策略。前 2K 步中，学习率从 0 线性增至最大值。随后，在训练进度达到 80% 时将学习率乘以 0.316，在达到 90% 时再次乘以 0.316。验证实验的最大学习率设为 $1.08 \times 10^{-3}$，梯度裁剪范数设为 1.0。批大小设为 2K，最大序列长度为 2K，因此每个训练批次包含 4M 个词元。相应地，总训练步数设为 25,000，以达到 100B 个训练词元。由于训练数据充足，训练过程中不使用 dropout。考虑到模型相对较小，包括专家参数在内的所有参数都部署在单张 GPU 上，以免出现计算不均衡。相应地，训练时不丢弃任何词元，也不使用设备级均衡损失。为了防止路由坍缩，专家级均衡因子设为 0.01。

为便于阅读，我们还在[第 10 节](#section-10)给出了不同规模 DeepSeekMoE 的超参数总览表。

<span id="section-4-1-4"></span>

#### 4.1.4 评估基准

我们在覆盖多类任务的众多基准上进行评估。所用基准如下。

**语言建模。** 语言建模使用 Pile [Gao20] 的测试集评估模型，评估指标为交叉熵损失。

**语言理解与推理。** 语言理解与推理采用 HellaSwag [Zel19]、PIQA [Bis20]、ARC-challenge 和 ARC-easy [Cla18]。这些任务的评估指标是准确率。

**阅读理解。** 阅读理解采用 RACE-high 和 RACE-middle [Lai17]，评估指标是准确率。

**代码生成。** 代码生成使用 HumanEval [Che21e] 和 MBPP [Aus21b] 评估模型。评估指标为 Pass@1，表示只生成一次时的通过率。

**闭卷问答。** 闭卷问答采用 TriviaQA [Jos17] 和 NaturalQuestions [Kwi19a]。评估指标为精确匹配（EM）率。

<span id="table-01"></span>

![表 1. 验证实验的评估结果。粗体表示最佳结果。与其他 MoE 架构相比，DeepSeekMoE 表现出显著的性能优势。](./deepseek-moe/table-01.png)

**表 1.** 验证实验的评估结果。**粗体**表示最佳结果。与其他 MoE 架构相比，DeepSeekMoE 表现出显著的性能优势。

<span id="section-4-2"></span>

### 4.2 评估

**基线。** 包括 DeepSeekMoE 在内，我们在验证实验中比较五个模型。**Dense** 表示一个标准稠密 Transformer 语言模型，总参数量为 0.2B。**Hash Layer** [Rol21] 是基于 top-1 哈希路由的 MoE 架构，总参数量为 2.0B，激活参数量为 0.2B，与稠密基线相同。**Switch Transformer** [Fed22] 是另一个知名的 MoE 架构，采用 top-1 可学习路由，总参数量和激活参数量与 Hash Layer 相同。**GShard** [Lep20] 采用 top-2 可学习路由策略；由于比 top-1 路由方法多激活一个专家，其总参数量为 2.0B，激活参数量为 0.3B。**DeepSeekMoE** 有 1 个共享专家和 63 个路由专家，每个专家的大小是标准 FFN 的 0.25 倍。包括 DeepSeekMoE 在内，所有比较模型使用相同的训练语料和训练超参数。所有参与比较的 MoE 模型具有相同的总参数量，GShard 与 DeepSeekMoE 的激活参数量也相同。

**结果。** 评估结果见[表 1](#table-01)。对于所有展示的模型，我们报告其在 100B 词元上训练完成后的最终评估结果。从表中可以得到以下观察：(1) 得益于稀疏架构和更多总参数，Hash Layer 与 Switch Transformer 的性能明显强于激活参数量相同的稠密基线。(2) 与 Hash Layer 和 Switch Transformer 相比，GShard 的激活参数更多，性能也略好于 Switch Transformer。(3) 在总参数量和激活参数量相同的情况下，DeepSeekMoE 相较 GShard 有压倒性优势。这些结果说明，在现有 MoE 架构中，DeepSeekMoE 架构具有优越性。

<span id="table-02"></span>

![表 2. DeepSeekMoE、较大的 GShard 模型与较大的稠密模型之间的比较。在“# Experts”一行中，$a$ + $b$ 表示 $a$ 个共享专家和 $b$ 个路由专家。在“# Activated Experts”一行中，$a$ + $b$ 表示 $a$ 个被激活的共享专家和 $b$ 个被激活的路由专家。DeepSeekMoE 的性能可与专家参数量和计算量均为其 1.5 倍的 GShard 模型相当。此外，DeepSeekMoE 的性能接近 FFN 参数量为其 16 倍的稠密模型；从模型容量来看，该稠密模型给出了 MoE 模型的上界。](./deepseek-moe/table-02.png)

**表 2.** DeepSeekMoE、较大的 GShard 模型与较大的稠密模型之间的比较。在“# Experts”一行中，$a$ + $b$ 表示 $a$ 个共享专家和 $b$ 个路由专家。在“# Activated Experts”一行中，$a$ + $b$ 表示 $a$ 个被激活的共享专家和 $b$ 个被激活的路由专家。DeepSeekMoE 的性能可与专家参数量和计算量均为其 1.5 倍的 GShard 模型相当。此外，DeepSeekMoE 的性能接近 FFN 参数量为其 16 倍的稠密模型；从模型容量来看，该稠密模型给出了 MoE 模型的上界。

<span id="section-4-3"></span>

### 4.3 DeepSeekMoE 非常接近 MoE 模型的上界

我们已经证明，DeepSeekMoE 优于稠密基线和其他 MoE 架构。为了更准确地认识 DeepSeekMoE 的性能，我们将它与总参数量或激活参数量更多的较大基线比较。通过这些比较，可以估计 GShard 或稠密基线要达到 DeepSeekMoE 的同等性能，需要多大的模型规模。

**与 GShard$\times 1.5$ 比较。** [表 2](#table-02) 比较了 DeepSeekMoE 与一个专家规模为其 1.5 倍的较大 GShard 模型，后者的专家参数量和专家计算量都因此成为前者的 1.5 倍。总体而言，DeepSeekMoE 的性能与 GShard$\times 1.5$ 相当，说明 DeepSeekMoE 架构具有明显优势。除了与 GShard$\times 1.5$ 的比较外，我们还在[第 11 节](#section-11)给出了与 GShard$\times 1.2$ 的比较。

我们还把 DeepSeekMoE 的总参数量增至 13.3B，并将它与总参数量分别为 15.9B 和 19.8B 的 GShard$\times 1.2$、GShard$\times 1.5$ 比较。结果发现，在更大的规模上，DeepSeekMoE 甚至能明显超过 GShard$\times 1.5$。这些结果也在[第 11 节](#section-11)中给出。

**与 Dense$\times 16$ 比较。** [表 2](#table-02) 还比较了 DeepSeekMoE 与更大的稠密模型。为公平比较，我们没有采用常见的注意力参数与 FFN 参数之比（1:2）。相反，我们配置了 16 个共享专家，每个专家的参数量与一个标准 FFN 相同。这种架构模拟了一个标准 FFN 参数量放大 16 倍的稠密模型。从表中可以看出，DeepSeekMoE 的性能接近 Dense$\times 16$；从模型容量来看，后者给出了 MoE 模型的严格上界。这些结果表明，***至少在约 2B 参数和 100B 训练词元的规模上，DeepSeekMoE 的性能已经非常接近 MoE 模型的理论上界***。我们还在[第 11 节](#section-11)给出了与 Dense$\times 4$ 的额外比较。

<span id="figure-03"></span>

![图 3. DeepSeekMoE 的消融实验。为清晰展示，性能按最佳结果归一化。所有比较模型的参数量和激活参数量都相同。可以看出，细粒度专家划分与共享专家隔离都能提高整体性能。](./deepseek-moe/figure-03.png)

**图 3.** DeepSeekMoE 的消融实验。为清晰展示，性能按最佳结果归一化。所有比较模型的参数量和激活参数量都相同。可以看出，细粒度专家划分与共享专家隔离都能提高整体性能。

<span id="section-4-4"></span>

### 4.4 消融实验

为了证实细粒度专家划分与共享专家隔离策略的有效性，我们对 DeepSeekMoE 进行了消融实验，结果见[图 3](#figure-03)。为公平比较，我们保证所有参与比较的模型具有相同的总参数量和激活参数量。

**共享专家隔离。** 为了评估共享专家隔离策略的影响，我们在 GShard 的基础上分离出一个专家作为共享专家。从[图 3](#figure-03) 可以看出，与 GShard 相比，有意分离一个共享专家提高了模型在大多数基准上的性能。这些结果支持了共享专家隔离策略有助于增强模型性能这一主张。

**细粒度专家划分。** 为评估细粒度专家划分策略的有效性，我们进一步把专家划分得更细，进行了更详细的比较。具体来说，我们把每个专家划分为 2 个或 4 个较小的专家，最终得到 32 个（1 个共享专家 + 31 个路由专家）或 64 个（1 个共享专家 + 63 个路由专家）专家。[图 3](#figure-03) 呈现出一个稳定趋势：随着专家划分粒度不断细化，模型的整体性能也持续提高。这些结果为细粒度专家划分策略的有效性提供了实验证据。

**共享专家与路由专家之比。** 我们还研究了共享专家与路由专家的最佳比例。在最细的划分粒度下，专家总数为 64；我们保持专家总数和激活专家数不变，尝试分别分离出 1、2、4 个专家作为共享专家。结果发现，共享专家与路由专家的不同比例不会显著影响性能；使用 1、2、4 个共享专家时，Pile 损失分别为 1.808、1.806 和 1.811。考虑到 1:3 的比例能得到略优的 Pile 损失，在扩展 DeepSeekMoE 时，我们将共享专家与激活路由专家之比保持为 1:3。

<span id="section-4-5"></span>

### 4.5 专家特化分析

本节对 DeepSeekMoE 2B 的专家特化进行实证分析。本节所说的 DeepSeekMoE 2B 是[表 1](#table-01) 中报告的模型，即总参数量为 2.0B，包含 1 个共享专家，并激活 63 个路由专家中的 7 个。

<span id="figure-04"></span>

![图 4. 禁用不同比例、路由概率最高的路由专家时的 Pile 损失。值得注意的是，DeepSeekMoE 对这一禁用比例更加敏感，表明 DeepSeekMoE 的路由专家之间冗余更少。](./deepseek-moe/figure-04.png)

**图 4.** 禁用不同比例、路由概率最高的路由专家时的 Pile 损失。值得注意的是，DeepSeekMoE 对这一禁用比例更加敏感，表明 DeepSeekMoE 的路由专家之间冗余更少。

**DeepSeekMoE 的路由专家之间冗余更少。** 为了评估路由专家之间的冗余，我们禁用不同比例、路由概率最高的路由专家，并评估 Pile 损失。具体来说，对于每个词元，我们屏蔽一定比例、路由概率最高的专家，再从剩余路由专家中选择 top-K 个专家。为公平起见，我们将 DeepSeekMoE 与 GShard$\times 1.5$ 比较，因为在不禁用专家时，两者的 Pile 损失相同。如[图 4](#figure-04) 所示，与 GShard$\times 1.5$ 相比，DeepSeekMoE 对禁用路由概率最高的专家更为敏感。这种敏感性说明 DeepSeekMoE 的参数冗余较低，因为每个路由专家都更难被替代。相比之下，GShard$\times 1.5$ 的专家参数冗余更多，因此禁用路由概率最高的专家时，它能够缓冲性能下降。

**共享专家无法由路由专家替代。** 为研究共享专家在 DeepSeekMoE 中的作用，我们禁用共享专家，并多激活一个路由专家。Pile 上的评估显示，尽管计算成本保持不变，Pile 损失仍从 1.808 显著升至 2.414。这一结果说明共享专家的作用非常重要，也表明共享专家捕获了路由专家不具备的基础、核心知识，因此无法由路由专家替代。

<span id="figure-05"></span>

![图 5. DeepSeekMoE 激活不同数量路由专家时的 Pile 损失。只激活 4 个路由专家时，DeepSeekMoE 便取得了与 GShard 相当的 Pile 损失。](./deepseek-moe/figure-05.png)

**图 5.** DeepSeekMoE 激活不同数量路由专家时的 Pile 损失。只激活 4 个路由专家时，DeepSeekMoE 便取得了与 GShard 相当的 Pile 损失。

<span id="figure-06"></span>

![图 6. GShard 与激活专家减半的 DeepSeekMoE 之间的比较（从头训练）。在专家总参数量相同、激活专家参数量只有一半的情况下，DeepSeekMoE 仍然优于 GShard。](./deepseek-moe/figure-06.png)

**图 6.** GShard 与激活专家减半的 DeepSeekMoE 之间的比较（从头训练）。在专家总参数量相同、激活专家参数量只有一半的情况下，DeepSeekMoE 仍然优于 GShard。

**DeepSeekMoE 获取知识更准确。** 为了验证“更灵活地组合激活专家有助于更准确、更有针对性地获取知识”这一主张，我们研究了 DeepSeekMoE 能否用更少的激活专家获得所需知识。具体来说，我们把激活路由专家数从 3 调至 7，并评估由此产生的 Pile 损失。如[图 5](#figure-05) 所示，即使只激活 4 个路由专家，DeepSeekMoE 也能取得与 GShard 相当的 Pile 损失。这一观察结果支持了 DeepSeekMoE 能够更准确、更高效地获取所需知识这一主张。

受这些结果鼓舞，为了更严格地验证 DeepSeekMoE 的专家特化与知识获取准确性，我们从头训练了一个新模型。该模型包含 1 个共享专家和 63 个路由专家，但只激活 3 个路由专家。[图 6](#figure-06) 中的评估结果显示，在专家总参数量相同、激活专家参数量只有一半的情况下，DeepSeekMoE 仍然优于 GShard。这表明 DeepSeekMoE 可以更高效地利用专家参数，即其激活专家中有效参数的比例远高于 GShard。

<span id="section-5"></span>

## 5 扩展至 DeepSeekMoE 16B

借助 DeepSeekMoE 架构，我们把 MoE 模型扩展至总参数量为 16B 的更大规模，并用 2T 个词元进行训练。结果表明，与 LLaMA2 7B 相比，DeepSeekMoE 16B 仅用约 40% 的计算量便取得了更好的性能。

<span id="section-5-1"></span>

### 5.1 实验设置

<span id="section-5-1-1"></span>

#### 5.1.1 训练数据与分词

我们从[第 4.1.1 节](#section-4-1-1)所述的同一语料库中抽取训练数据。与验证实验不同，这里抽取了更多数据，共 2T 个词元，与 LLaMA2 7B 的训练词元数一致。我们同样使用 HuggingFace Tokenizer 工具训练 BPE 分词器，但 DeepSeekMoE 16B 的词表大小设为 100K。

<span id="section-5-1-2"></span>

#### 5.1.2 超参数

**模型设置。** 对于 DeepSeekMoE 16B，我们把 Transformer 层数设为 28，隐藏维度设为 2048。模型采用多头注意力机制，共有 16 个注意力头，每个头的维度为 128。初始化方面，所有可学习参数都以 0.006 的标准差随机初始化。我们观察到第一层的负载均衡状态收敛得尤其慢，因此除第一层外，把其余全部 FFN 都替换为 MoE 层。每个 MoE 层由 2 个共享专家和 64 个路由专家组成，每个专家的大小是标准 FFN 的 0.25 倍。每个词元会被路由到这 2 个共享专家和 64 个路由专家中的 6 个。专家过小可能降低计算效率，因此我们没有采用更细的专家划分粒度。规模超过 16B 时，仍可以采用更细的粒度。在这一配置下，DeepSeekMoE 16B 的总参数量约为 16.4B，激活参数量约为 2.8B。

**训练设置。** 我们使用 AdamW 优化器 [Los17]，超参数设为 $\beta_1=0.9$、$\beta_2=0.95$ 和 $\mathrm{weight\_decay}=0.1$。学习率同样采用预热加阶梯衰减策略。前 2K 步中，学习率从 0 线性增至最大值。随后，在训练进度达到 80% 时将学习率乘以 0.316，在达到 90% 时再次乘以 0.316。DeepSeekMoE 16B 的最大学习率设为 $4.2 \times 10^{-4}$，梯度裁剪范数设为 1.0。批大小设为 4.5K，最大序列长度为 4K，因此每个训练批次包含 18M 个词元。相应地，总训练步数设为 106,449，以达到 2T 个训练词元。由于训练数据充足，训练时不使用 dropout。我们通过流水线并行把模型的不同层部署到不同设备；对于每一层，所有专家都会部署在同一设备上。因此，训练时同样不丢弃任何词元，也不使用设备级均衡损失。为了防止路由坍缩，我们把专家级均衡因子设为很小的 0.001，因为我们发现，在这种并行策略下，提高专家级均衡因子并不能提升计算效率，反而会损害模型性能。

<span id="section-5-1-3"></span>

#### 5.1.3 评估基准

除验证实验使用的基准外，我们又加入一些基准，使评估更加全面。与验证实验所用基准的差异如下。

**语言建模。** 语言建模仍使用 Pile [Gao20] 的测试集评估模型。由于 DeepSeekMoE 16B 与 LLaMA2 7B 使用的分词器不同。为公平比较，我们采用每字节比特数（BPB）作为评估指标。

**阅读理解。** 阅读理解还加入了 DROP [Dua19]。评估指标为精确匹配（EM）率。

**数学推理。** 数学推理还加入了 GSM8K [Cob21] 和 MATH [Hen21]，以 EM 为评估指标。

**多学科选择题。** 多学科选择题还使用 MMLU [Hen20] 评估模型。评估指标为准确率。

**歧义消解。** 歧义消解还加入了 WinoGrande [Sak19]，评估指标为准确率。

**中文基准。** DeepSeekMoE 16B 在双语语料库上预训练，因此我们也在四个中文基准上评估模型。CLUEWSC [Xu20] 是中文歧义消解基准。CEval [Hua23] 与 CMMLU [Li23e] 是两个形式与 MMLU 相似的中文多学科选择题基准。CHID [Zhe19] 是一个中文成语填空基准，用来评估对中国文化的理解。上述中文基准的评估指标为准确率或 EM。

**Open LLM Leaderboard。** 我们基于内部评估框架评估了上述所有基准。为了方便、公平地将 DeepSeekMoE 16B 与开源模型比较，我们还在 Open LLM Leaderboard 上评估了 DeepSeekMoE 16B。Open LLM Leaderboard 是 HuggingFace 支持的公开排行榜，由六项任务组成：ARC [Cla18]、HellaSwag [Zel19]、MMLU [Hen20]、TruthfulQA [Lin22]、Winogrande [Sak19] 和 GSM8K [Cob21]。

<span id="section-5-2"></span>

### 5.2 评估

<span id="table-03"></span>

![表 3. DeepSeek 7B 与 DeepSeekMoE 16B 的比较。粗体表示最佳或接近最佳的结果。DeepSeekMoE 16B 仅用 40.5% 的计算量，便取得了与 DeepSeek 7B 相当的性能。](./deepseek-moe/table-03.png)

**表 3.** DeepSeek 7B 与 DeepSeekMoE 16B 的比较。**粗体**表示最佳或接近最佳的结果。DeepSeekMoE 16B 仅用 40.5% 的计算量，便取得了与 DeepSeek 7B 相当的性能。

<span id="section-5-2-1"></span>

#### 5.2.1 与 DeepSeek 7B 的内部比较

我们先在 DeepSeekMoE 16B 和总参数量为 6.9B 的稠密语言模型 DeepSeek 7B [Dee24e] 之间进行内部比较。为保证公平，两种模型都在包含 2T 个词元的同一语料库上训练。这样可以排除训练数据的影响，准确评估 MoE 架构的有效性。

评估结果见[表 3](#table-03)，可以得到以下观察：(1) 总体而言，DeepSeekMoE 16B 仅用约 40% 的计算量，便取得了与 DeepSeek 7B 相当的性能。(2) DeepSeekMoE 16B 在语言建模及 Pile、HellaSwag、TriviaQA、NaturalQuestions 等知识密集型任务上优势明显。由于 MoE 模型中的 FFN 参数远多于注意力参数，这些结果符合“Transformer 中的 FFN 具有知识记忆能力”这一主张 [Dai22a]。(3) 与在其他任务上的出色表现相比，DeepSeekMoE 处理选择题时存在局限。这是因为 DeepSeekMoE 16B 的注意力参数有限（DeepSeekMoE 16B 只有约 0.5B 注意力参数，而 DeepSeek 7B 有 2.5B）。我们早先对 DeepSeek 7B 的研究发现，注意力容量与选择题任务性能正相关。例如，采用多查询注意力机制 [Sha19] 的 DeepSeek 7B MQA 在 MMLU 类任务上也表现不佳。此外，为了更全面地了解 DeepSeekMoE 16B 的训练过程，我们还在[第 12 节](#section-12)给出 DeepSeekMoE 16B 和 DeepSeek 7B（Dense）训练期间的基准曲线，以供参考。

重要的是，DeepSeekMoE 16B 的参数量不大，可以单设备部署在一张配有 40GB 显存的 GPU 上。配合适当的算子优化，它的推理速度可以达到 7B 稠密模型的近 2.5 倍。

<span id="table-04"></span>

![表 4. LLaMA2 7B 与 DeepSeekMoE 16B 的比较。DeepSeekMoE 16B 仅用 39.6% 的计算量，便在大多数基准上优于 LLaMA2 7B。](./deepseek-moe/table-04.png)

**表 4.** LLaMA2 7B 与 DeepSeekMoE 16B 的比较。DeepSeekMoE 16B 仅用 39.6% 的计算量，便在大多数基准上优于 LLaMA2 7B。

<span id="section-5-2-2"></span>

#### 5.2.2 与开源模型比较

**与 LLaMA2 7B 的内部比较。** 在开源模型中，我们主要将 DeepSeekMoE 16B 与 LLaMA2 7B [Tou23a] 比较，后者是一个知名且性能很强的开源语言模型，总参数量为 6.7B。DeepSeekMoE 16B 与 LLaMA2 7B 都使用 2T 个词元进行预训练。与 LLaMA2 7B 相比，DeepSeekMoE 的总参数量是其 245%，所需计算量却只有 39.6%。内部基准的结果见[表 4](#table-04)，可以得到以下观察。(1) 在所评估的基准中，DeepSeekMoE 16B 仅用约 40% 的计算量，便在大多数基准上优于 LLaMA2 7B。(2) DeepSeekMoE 16B 的数学推理与代码生成能力强于 LLaMA2 7B，这是因为我们的预训练语料库中有更多数学和代码相关文本。(3) 由于预训练语料库包含中文文本，DeepSeekMoE 16B 在中文基准上明显优于 LLaMA2 7B。(4) 尽管训练所用的英文文本更少，DeepSeekMoE 16B 在英语理解或知识密集型基准上仍取得了与 LLaMA2 7B 相当或更好的性能，说明 DeepSeekMoE 16B 的能力非常强。

**Open LLM Leaderboard 上的评估。** 除内部评估外，我们还在 Open LLM Leaderboard 上评估 DeepSeekMoE 16B，并与其他开源模型比较。除 LLaMA2 7B 外，我们还考虑了范围更广的开源模型，包括 LLaMA 7B [Tou23]、Falcon 7B [Alm23b]、GPT-J 6B [Wan21g]、RedPajama-INCITE 7B 和 3B [Tog23a]、Open LLaMA 7B 和 3B [Gen23a]、OPT 2.7B [Zha22]、Pythia 2.8B [Bid23]、GPT-neo 2.7B [Bla21] 以及 BLOOM 3B [Les23]。[图 1](#figure-01) 给出的评估结果显示，DeepSeekMoE 16B 始终大幅优于激活参数量相近的模型。此外，它取得了与 LLaMA2 7B 相当的性能，而后者的激活参数量约为其 2.5 倍。

<span id="section-6"></span>

## 6 DeepSeekMoE 16B 的对齐

以往研究表明，MoE 模型通常无法从微调中获得明显收益 [Fed22, Art22]。不过，[She23a] 的研究结果显示，MoE 模型确实可以受益于指令微调。为了评估 DeepSeekMoE 16B 能否从微调中获益，我们进行了监督微调，以 DeepSeekMoE 16B 为基础构建聊天模型。实验结果显示，DeepSeekMoE Chat 16B 同样取得了与 LLaMA2 SFT 7B 和 DeepSeek Chat 7B 相当的性能。

<span id="section-6-1"></span>

### 6.1 实验设置

**训练数据。** 为训练聊天模型，我们使用内部整理的数据进行监督微调（SFT），其中包含 1.4M 个训练样本。该数据集覆盖数学、代码、写作、问答、推理、摘要等众多类别。我们的 SFT 训练数据大部分为英文和中文，因此聊天模型能够灵活用于双语场景。

**超参数。** 监督微调时，我们把批大小设为 1024 个样本，使用 AdamW 优化器 [Los17] 训练 8 个 epoch。最大序列长度设为 4K，并尽可能紧密地打包训练样本，直到达到序列长度上限。监督微调不使用 dropout，只采用恒定的 $10^{-5}$ 学习率，不引入任何学习率调度策略。

**评估基准。** 评估聊天模型时，我们使用与[第 5.1.3 节](#section-5-1-3)相似的基准，但作出以下调整：(1) 排除 Pile [Gao20]，因为聊天模型很少用于纯语言建模。(2) 排除 CHID [Zhe19]，因为观察到结果不稳定，难以得出可靠结论。(3) 额外加入 BBH [Suz22]，以便更全面地评估聊天模型的推理能力。

<span id="table-05"></span>

![表 5. LLaMA2 SFT 7B、DeepSeek Chat 7B 与 DeepSeekMoE Chat 16B 的比较，三个模型都在相同的 SFT 数据上微调。与两个 7B 稠密模型相比，DeepSeekMoE Chat 16B 仅用 40% 的计算量，仍能在大多数基准上取得相当或更好的性能。](./deepseek-moe/table-05.png)

**表 5.** LLaMA2 SFT 7B、DeepSeek Chat 7B 与 DeepSeekMoE Chat 16B 的比较，三个模型都在相同的 SFT 数据上微调。与两个 7B 稠密模型相比，DeepSeekMoE Chat 16B 仅用 40% 的计算量，仍能在大多数基准上取得相当或更好的性能。

<span id="section-6-2"></span>

### 6.2 评估

**基线。** 为验证 DeepSeekMoE 16B 对齐后的潜力，我们对 LLaMA2 7B、DeepSeek 7B 与 DeepSeekMoE 16B 进行监督微调，并使用完全相同的微调数据来保证公平。相应地，我们构建了三个聊天模型：LLaMA2 SFT 7B [+3]、DeepSeek Chat 7B 与 DeepSeekMoE Chat 16B。随后，我们在众多下游任务上，将 DeepSeekMoE Chat 16B 与另外两个稠密聊天模型（FLOPs 约为其 2.5 倍）比较。

**结果。** 评估结果见[表 5](#table-05)。主要观察如下：(1) DeepSeekMoE Chat 16B 仅消耗约 40% 的计算量，却在语言理解与推理（PIQA、ARC、BBH）、机器阅读理解（RACE）、数学（GSM8K、MATH）以及知识密集型任务（TriviaQA、NaturalQuestions）上取得了与 7B 稠密模型相当的性能。(2) 在代码生成任务上，DeepSeekMoE Chat 16B 明显优于 LLaMA2 SFT 7B，在 HumanEval 和 MBPP 上都有显著提升。此外，它也超过了 DeepSeek Chat 7B。(3) 在 MMLU、CEval、CMMLU 等选择题问答基准上，DeepSeekMoE Chat 16B 仍落后于 DeepSeek Chat 7B，这与基础模型上的观察结果一致（[第 5.2.1 节](#section-5-2-1)）。不过，值得注意的是，监督微调后，DeepSeekMoE 16B 与 DeepSeek 7B 之间的性能差距缩小了。(4) 得益于在双语语料库上的预训练，DeepSeekMoE Chat 16B 在所有中文基准上都明显优于 LLaMA2 SFT 7B。这些结果表明，DeepSeekMoE 16B 在中文和英文上能力均衡，因而能够灵活用于不同场景。总的来说，聊天模型评估说明 DeepSeekMoE 16B 能够从对齐中获益，也验证了它只用约 40% 的计算量便能持续取得与稠密模型相当的性能。

<span id="section-7"></span>

## 7 仍在进行中的 DeepSeekMoE 145B

受 DeepSeekMoE 16B 出色性能的鼓舞，我们进一步初步尝试将 DeepSeekMoE 扩展到 145B。在这项初步研究中，DeepSeekMoE 145B 仅使用 245B 个词元训练，却已持续表现出相较 GShard 架构的优势，并有望达到或超过 DeepSeek 67B（Dense）的性能。此外，等 DeepSeekMoE 145B 最终版本完成全部训练后，我们也计划将其公开。

<span id="section-7-1"></span>

### 7.1 实验设置

**训练数据与分词。** DeepSeekMoE 145B 使用与 DeepSeekMoE 16B 完全相同的训练语料和分词器；唯一的区别是，在初步研究中，DeepSeekMoE 145B 使用 245B 个词元训练。

**模型设置。** 对于 DeepSeekMoE 145B，我们把 Transformer 层数设为 62，隐藏维度设为 4096。模型采用多头注意力机制，共有 32 个注意力头，每个头的维度为 128。初始化方面，所有可学习参数都以 0.006 的标准差随机初始化。与 DeepSeekMoE 16B 一样，除第一层外，我们同样把其余所有 FFN 替换为 MoE 层。每个 MoE 层由 4 个共享专家和 128 个路由专家组成，每个专家的大小是标准 FFN 的 0.125 倍。每个词元会被路由到这 4 个共享专家和 128 个路由专家中的 12 个。在这一配置下，DeepSeekMoE 145 的总参数量约为 144.6B，激活参数量约为 22.2B。

**训练设置。** 我们使用 AdamW 优化器 [Los17]，超参数设为 $\beta_1=0.9$、$\beta_2=0.95$ 和 $\mathrm{weight\_decay}=0.1$。在 DeepSeekMoE 145B 的初步研究中，学习率采用预热后恒定的调度策略。前 2K 步中，学习率从 0 线性增至最大值。随后，余下训练过程中学习率保持不变。DeepSeekMoE 145B 的最大学习率设为 $3.0 \times 10^{-4}$，梯度裁剪范数设为 1.0。批大小设为 4.5K，最大序列长度为 4K，因此每个训练批次包含 18M 个词元。我们将 DeepSeekMoE 145B 训练 13,000 步，共使用 245B 个训练词元。训练时同样不使用 dropout。我们通过流水线并行把模型的不同层部署到不同设备；对于每一层，所有路由专家会均匀部署在 4 台设备上（即专家并行与数据并行相结合）。由于 DeepSeekMoE 145B 采用专家并行，需要考虑设备级负载均衡，以减轻计算瓶颈。因此，我们把设备级均衡因子设为 0.05，以促进设备间计算均衡。同时，仍采用较小的专家级均衡因子 0.003 来防止路由坍缩。

**评估基准。** 我们在与 DeepSeekMoE 16B 完全相同的内部基准上评估 DeepSeekMoE 145B（见[第 5.1.3 节](#section-5-1-3)）。

<span id="table-06"></span>

![表 6. DeepSeek 67B（Dense）与总参数规模约 140B 的 MoE 模型之间的比较。在“# Experts”和“# Activated Experts”两行中，$a$ + $b$ 分别表示 $a$ 个共享专家和 $b$ 个路由专家。粗体表示除最后一列外最佳或接近最佳的性能。DeepSeekMoE 145B，甚至激活专家参数量只有一半的 DeepSeekMoE 142B（Half Activated），都大幅优于 GShard 137B。此外，DeepSeekMoE 145B 仅用 28.5% 的计算量，便取得了与 DeepSeek 67B 相当的性能。](./deepseek-moe/table-06.png)

**表 6.** DeepSeek 67B（Dense）与总参数规模约 140B 的 MoE 模型之间的比较。在“# Experts”和“# Activated Experts”两行中，$a$ + $b$ 分别表示 $a$ 个共享专家和 $b$ 个路由专家。**粗体**表示除最后一列外最佳或接近最佳的性能。DeepSeekMoE 145B，甚至激活专家参数量只有一半的 DeepSeekMoE 142B（Half Activated），都大幅优于 GShard 137B。此外，DeepSeekMoE 145B 仅用 28.5% 的计算量，便取得了与 DeepSeek 67B 相当的性能。

<span id="section-7-2"></span>

### 7.2 评估

**基线。** 除 **DeepSeekMoE 145B** 外，我们还考虑三个模型进行比较。**DeepSeek 67B（Dense）** 是一个总参数量为 67.4B 的稠密模型（模型与训练细节参见 [Dee24e]）。**GShard 137B** 的隐藏维度与层数和 DeepSeekMoE 145B 相同，但采用 GShard 架构。需要注意的是，为提高计算效率，DeepSeekMoE 145B 将每个专家的中间隐藏维度对齐为 64 的倍数，因此模型规模比 GShard 137B 大 6%。**DeepSeekMoE 142B（Half Activated）** 的架构与 DeepSeekMoE 145B 相似，但只包含 2 个共享专家，且只激活 128 个路由专家中的 6 个。值得注意的是，包括 DeepSeekMoE 145B 在内，所有比较模型均使用相同的训练语料。此外，参与比较的所有 MoE 模型都从头训练，并采用相同的训练超参数。

**结果。** 从[表 6](#table-06) 中的评估结果可以得到以下观察：(1) 尽管总参数量和计算量相当，DeepSeekMoE 145B 仍明显优于 GShard 137B，再次说明 DeepSeekMoE 架构的优势。(2) 总体而言，DeepSeekMoE 145B 只用 28.5% 的计算量，便取得了与 DeepSeek 67B（Dense）相当的性能。与 DeepSeekMoE 16B 的结果一致，DeepSeekMoE 145B 在语言建模和知识密集型任务上优势明显，但在选择题任务上存在局限。(3) 在更大的规模上，DeepSeekMoE 142B（Half Activated）的性能与 DeepSeekMoE 145B 相差不大。此外，尽管激活专家参数量只有一半，DeepSeekMoE 142B（Half Activated）仍然只用 18.2% 的计算量就匹敌 DeepSeek 67B（Dense）。它也优于 GShard 137B，与[第 4.5 节](#section-4-5)的结论一致。

<span id="section-8"></span>

## 8 相关工作

混合专家（MoE）技术最早由 [Jac91] [Jor94] 提出，使用相互独立的专家模块处理不同样本。[Sha17] 将 MoE 引入语言模型训练，并构建了大规模、基于 LSTM [Hoc97] 的 MoE 模型。随着 Transformer 成为 NLP 中最常用的架构，许多工作把 Transformer 中的 FFN 扩展为 MoE 层，以构建 MoE 语言模型。GShard [Lep20] 和 Switch Transformer [Fed22] 是其中的先驱，它们分别采用可学习的 top-2 或 top-1 路由策略，把 MoE 语言模型扩展到极大规模。Hash Layer [Rol21] 和 StableMoE [Dai22b] 使用固定路由策略，使路由和训练更加稳定。[Zho22a] 提出专家选择路由策略，其中每个词元可以分配到不同数量的专家。[Zop22] 关注 MoE 模型的训练不稳定和微调困难问题，并提出 ST-MoE 来解决这些难题。除研究 MoE 架构与训练策略外，近年来还出现了许多基于现有 MoE 架构的大规模语言或多模态模型 [Lin21a, Du22, Ren23a, Xue23]。总体来看，以往大多数 MoE 模型都基于传统的 top-1 或 top-2 路由策略，专家特化仍有很大的改进空间。为此，DeepSeekMoE 架构力求把专家特化程度提升到极致。

<span id="section-9"></span>

## 9 结论

本文面向 MoE 语言模型提出 DeepSeekMoE 架构，目标是实现极致的专家特化。通过细粒度专家划分与共享专家隔离，与主流 MoE 架构相比，DeepSeekMoE 实现了明显更高的专家特化程度和性能。我们从参数规模适中的 2B 模型开始，验证了 DeepSeekMoE 的优势，并证明其性能可以接近 MoE 模型的上界。此外，我们还提供实验证据，表明 DeepSeekMoE 的专家特化程度高于 GShard。

把总参数量扩展到更大的 16B 规模后，我们使用 2T 个词元训练 DeepSeekMoE 16B，并表明它仅用约 40% 的计算量，便取得了与 DeepSeek 7B 和 LLaMA2 7B 相当的出色性能。此外，我们还通过监督微调进行对齐，以 DeepSeekMoE 16B 为基础构建 MoE 聊天模型，进一步体现了它的适应能力与通用性。接下来，我们初步尝试将 DeepSeekMoE 扩展至 145B 参数。结果发现，DeepSeekMoE 145B 相较 GShard 架构仍保持显著优势，并仅用 28.5%（甚至可能是 18.2%）的计算量便取得了与 DeepSeek 67B 相当的性能。

出于研究目的，我们向公众发布 DeepSeekMoE 16B 的模型检查点，它可以部署在一张配有 40GB 显存的 GPU 上。我们希望这项工作能为学术界和工业界提供有价值的认识，并推动大规模语言模型更快发展。

<span id="section-10"></span>

## 10 超参数总览

[表 7](#table-07) 给出了不同规模 DeepSeekMoE 的超参数总览。

<span id="table-07"></span>

![表 7. 不同规模 DeepSeekMoE 的超参数总览。相对专家大小以标准 FFN 为参照。](./deepseek-moe/table-07.png)

**表 7.** 不同规模 DeepSeekMoE 的超参数总览。相对专家大小以标准 FFN 为参照。

<span id="section-11"></span>

## 11 DeepSeekMoE 与更大模型的比较

[表 8](#table-08) 给出了 DeepSeekMoE、GShard$\times 1.2$ 与 GShard$\times 1.5$ 之间的比较。[表 9](#table-09) 给出了 DeepSeekMoE、Dense$\times 4$ 与 Dense$\times 16$ 之间的比较。

<span id="table-08"></span>

![表 8. DeepSeekMoE 与较大 GShard 模型的比较。](./deepseek-moe/table-08.png)

**表 8.** DeepSeekMoE 与较大 GShard 模型的比较。

<span id="table-09"></span>

![表 9. DeepSeekMoE 与较大稠密基线的比较。](./deepseek-moe/table-09.png)

**表 9.** DeepSeekMoE 与较大稠密基线的比较。

在总参数量为 13B 的更大规模上，我们还将 DeepSeekMoE 与 GShard$\times 1.2$、GShard$\times 1.5$ 比较，结果见[表 10](#table-10)。在更大的规模上，DeepSeekMoE 甚至明显优于 GShard$\times 1.5$。

<span id="table-10"></span>

![表 10. 更大规模下 DeepSeekMoE 与较大 GShard 模型的比较。](./deepseek-moe/table-10.png)

**表 10.** 更大规模下 DeepSeekMoE 与较大 GShard 模型的比较。

<span id="section-12"></span>

## 12 DeepSeekMoE 16B 的训练基准曲线

我们在[图 7](#figure-07) 中给出 DeepSeekMoE 16B 和 DeepSeek 7B（Dense）训练期间的基准曲线，以供参考。

<span id="figure-07"></span>

![图 7. DeepSeekMoE 16B 与 DeepSeek 7B（Dense）训练期间的基准曲线。](./deepseek-moe/figure-07.png)

**图 7.** DeepSeekMoE 16B 与 DeepSeek 7B（Dense）训练期间的基准曲线。

[+1]: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
[+2]: [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
[+3]: 我们用 LLaMA2 SFT 指代不同于官方 LLaMA2 Chat [Tou23a] 的模型。
