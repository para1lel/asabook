---
title: 'Native Sparse Attention'
createTime: 2026/09/06 14:00:00
permalink: /papers/native-sparse-attention/
pageClass: paper-reading
---

> [Jingyang Yuan](https://dblp.org/pid/244/7491.html) [+internship], [Huazuo Gao](https://dblp.org/pid/366/3356.html), [Damai Dai](https://dblp.org/pid/199/2097.html), [Junyu Luo](https://dblp.org/pid/198/0850-2.html), [Liang Zhao](https://dblp.org/pid/63/5422-26.html), [Zhengyan Zhang](https://aclanthology.org/people/zhengyan-zhang/unverified/), [Zhenda Xie](https://dblp.org/pid/239/8676.html), [Yuxing Wei](https://aclanthology.org/people/yuxing-wei/unverified/), [Lean Wang](https://aclanthology.org/people/lean-wang/), [Zhiping Xiao](https://dblp.org/pid/176/5397-1.html), [Yuqing Wang](https://aclanthology.org/people/yuqing-wang/), [Chong Ruan](https://dblp.org/pid/159/9956.html), [Ming Zhang](https://dblp.org/pid/73/1844-4.html), [Wenfeng Liang](https://dblp.org/pid/59/9456.html) 和 [Wangding Zeng](https://dblp.org/pid/315/5319.html). 论文于 2025 年 2 月 16 日首次提交至 arXiv; 当前版本为 v2. 本阅读版转录并翻译自 [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089v2), 论文后来发表于 ACL 2025, 获 Best Paper, 页码为 23078–23097. <a href="/paper/native-sparse-attention.pdf" target="_blank">原始 PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2502.11089). [ACL 论文](https://doi.org/10.18653/v1/2025.acl-long.1126). [TeX 源码](https://export.arxiv.org/e-print/2502.11089v2). 精确的印刷版式和参考文献以原始 PDF 为准.

[+internship]: 在 DeepSeek-AI 实习期间完成的贡献.

## 摘要

长上下文建模是下一代语言模型的关键能力, 但标准注意力机制的高计算成本也带来了严峻的计算挑战.

稀疏注意力有望在保持模型能力的同时提高效率.

我们提出 NSA, 一种原生可训练的稀疏注意力机制. 它把算法创新与适配硬件的优化结合起来, 以实现高效的长上下文建模.

NSA 采用动态的分层稀疏策略, 将粗粒度 token 压缩与细粒度 token 选择结合起来, 兼顾全局上下文感知和局部精度. 我们从两个方面推进了稀疏注意力设计: (1) 通过平衡算术强度的算法设计, 配合面向现代硬件的实现优化, 取得显著加速; (2) 支持端到端训练, 在不牺牲模型性能的前提下降低预训练计算量.

如[图 1](#figure-01) 所示, 实验表明, 使用 NSA 预训练的模型在通用 benchmark、长上下文任务和基于指令的推理任务上达到或超过 Full Attention 模型.

同时, 对于长度为 64k 的序列, NSA 在解码、前向传播和反向传播阶段都比 Full Attention 快得多, 说明它能贯穿模型的整个生命周期提高效率.

<span id="section-1"></span>

## 1 引言

<span id="figure-01"></span>

![Full Attention 模型与 NSA 在性能和效率上的对比. 左: NSA 虽然采用稀疏注意力, 但在通用 benchmark、长上下文任务和推理评测上的平均表现超过 Full Attention baseline. 右: 处理 64k 长度序列时, NSA 在解码、前向传播和反向传播三个阶段都比 Full Attention 获得了显著的计算加速.](./native-sparse-attention/figure-01.png)

**图 1. Full Attention 模型与 NSA 在性能和效率上的对比. 左: NSA 虽然采用稀疏注意力, 但在通用 benchmark、长上下文任务和推理评测上的平均表现超过 Full Attention baseline. 右: 处理 64k 长度序列时, NSA 在解码、前向传播和反向传播三个阶段都比 Full Attention 获得了显著的计算加速.**

研究界日益认识到, 长上下文建模是下一代大语言模型的关键能力. 现实应用对这一能力有广泛需求, 包括深度推理 [Zel22a, Dee25c]、repo 级代码生成 [Zha23z, Zha24zz] 和多轮自主 agent 系统 [Par23]. OpenAI o 系列模型、DeepSeek-R1 [Dee25c] 和 Gemini 1.5 Pro [Tea24a] 等近期突破, 使模型能够处理完整代码库和长篇文档, 在数千 token 的多轮对话中保持连贯, 并对长程依赖进行复杂推理. 然而, 随着序列变长, vanilla Attention [Zah20, Vas17] 的高复杂度逐渐成为主要的延迟瓶颈. 理论估算表明, 解码 64k 长度上下文时, 带 softmax 的注意力计算占总延迟的 70–80%, 因而亟需更高效的注意力机制.

一种自然的思路是利用 softmax 注意力固有的稀疏性 [Ge24, Jia23z]: 只计算关键的 query-key 对, 在保持性能的同时显著减少计算开销. 近期研究从多条路径展示了这种潜力, 包括 KV-cache 淘汰 [Zha23g, Li24c, Zho24z]、分块 KV-cache 选择 [Tan24, Xia24z, Gao24z], 以及基于采样、聚类或哈希的选择 [Che25ab, Liu24zz, Des24z]. 尽管如此, 现有稀疏注意力方法在实际部署中往往达不到预期. 很多方法的实际加速远低于理论收益; 另外, 大多数方法无法有效支持训练阶段, 因而不能充分利用注意力的稀疏模式.

要让稀疏注意力真正高效, 需要解决两个关键问题: (1) ***适配硬件的推理加速***: 要把理论上的计算量缩减转化为实际速度提升, 必须在 prefill 和解码阶段都采用适合硬件的算法设计, 缓解内存访问和硬件调度瓶颈; (2) ***面向训练的算法设计***: 通过可训练算子支持端到端计算, 在保持模型性能的同时降低训练成本. 这两点是长上下文推理或训练走向实际应用的前提. 同时考察二者, 现有方法仍有明显缺口.

为此, 我们提出 NSA, 一种原生可训练、采用分层 token 建模的稀疏注意力架构. 如[图 2](#figure-02) 所示, NSA 把 key 和 value 按时间组织成 block, 并通过三条注意力分支处理: 压缩后的粗粒度 token、经过选择而保留的细粒度 token, 以及负责局部上下文的滑动窗口. 随后, 我们用专用 kernel 最大限度提高实际效率. NSA 的两项核心创新分别对应上述要求: (1) 适配硬件的系统: 针对 Tensor Core 利用率和内存访问优化 blockwise 稀疏注意力, 使算术强度保持平衡; (2) 面向训练的设计: 通过高效算法和反向算子支持稳定的端到端训练. 这些优化让 NSA 可以兼顾高效部署和端到端训练.

我们在真实语言语料上全面评测了 NSA. 以 27B 参数 transformer 为 backbone、使用 260B token 预训练后, 我们从通用语言能力、长上下文能力和 chain-of-thought 推理三个方面评估 NSA. 我们还在 A100 GPU 上把 kernel 速度与优化过的 Triton [Til19] 实现进行比较. 实验结果表明, NSA 的性能与 Full Attention baseline 相当或更好, 同时超过现有稀疏注意力方法. 与 Full Attention 相比, NSA 在解码、前向和反向三个阶段都取得了显著加速, 且序列越长, 加速比越高. 这些结果说明, 分层稀疏注意力能够有效兼顾模型能力与计算效率.

<span id="section-2"></span>

## 2 重新审视稀疏注意力方法

现代稀疏注意力方法在降低 transformer 理论计算复杂度方面已经取得了长足进展. 但多数方法只在推理时引入稀疏性, backbone 仍沿用预训练的 Full Attention, 这可能带来架构偏差, 限制稀疏注意力优势的发挥. 在介绍原生稀疏架构之前, 我们从两个关键角度系统分析这些局限.

<span id="figure-02"></span>

![NSA 架构概览. 左: 框架通过三条并行的注意力分支处理输入序列. 对于给定 query, 先前的 key 和 value 分别进入压缩注意力以捕捉粗粒度模式、选择注意力以保留重要 token block, 以及滑动注意力以处理局部上下文. 右: 各分支产生的不同注意力模式. 绿色区域表示需要计算注意力分数的位置, 白色区域表示可以跳过的位置.](./native-sparse-attention/figure-02.png)

**图 2. NSA 架构概览. 左: 框架通过三条并行的注意力分支处理输入序列. 对于给定 query, 先前的 key 和 value 分别进入压缩注意力以捕捉粗粒度模式、选择注意力以保留重要 token block, 以及滑动注意力以处理局部上下文. 右: 各分支产生的不同注意力模式. 绿色区域表示需要计算注意力分数的位置, 白色区域表示可以跳过的位置.**

<span id="section-2-1"></span>

### 2.1 高效推理的假象

很多方法虽然减少了注意力计算, 推理延迟却没有相应下降, 主要有两个原因:

**仅限特定阶段的稀疏性.** H2O [Zha23g] 等方法在自回归解码时应用稀疏注意力, 但 prefill 阶段仍要执行计算密集的预处理, 如计算 attention map、建立索引. MInference [Jia24e] 则只关注 prefill 阶段的稀疏性. 这类方法无法覆盖整个推理过程, 至少有一个阶段的计算成本仍与 Full Attention 相当. 这种阶段局限性会削弱它们在不同 workload 下的加速能力, 无论是图书摘要、代码补全等由 prefill 主导的任务, 还是长 chain-of-thought [Wei22z] 推理等由解码主导的任务.

**与先进注意力架构不兼容.** 一些稀疏注意力方法无法适配更适合解码的现代架构, 如 Multi-Query Attention (MQA) [Sha19] 和 Grouped-Query Attention (GQA) [Ain23]. 这些架构让多个 query head 共享 KV, 从而大幅缓解解码时的内存访问瓶颈. 以 Quest [Tan24] 为例, 每个 attention head 会独立选择自己的 KV-cache 子集. 对 Multi-Head Attention (MHA) 模型而言, 它的计算稀疏度和内存访问稀疏度可以保持一致; 但在 GQA 等架构中, 同一 GQA group 内的 query head 共用 KV, 所需加载的 KV-cache 实际上是所有 head 选择结果的并集. 因此, 这类方法虽能减少计算操作, 却仍要访问较多 KV-cache. 它们由此面临一个关键矛盾: 计算量下降了, 分散的内存访问模式却与先进架构的高效内存访问设计冲突.

这些问题源于很多现有方法只关注 KV-cache 或理论计算量的缩减, 却难以在先进框架或 backend 上显著降低延迟. 因此, 我们需要把先进架构与硬件高效实现结合起来, 真正利用稀疏性提升模型效率.

<span id="section-2-2"></span>

### 2.2 可训练稀疏性的误区

我们追求原生可训练稀疏注意力, 源于对纯推理方案的两点认识: (1) ***性能下降***: 事后施加稀疏性会迫使模型偏离预训练时的优化轨迹. Chen 等人 [Che25ab] 表明, 得分最高的 20% 注意力也只能覆盖总注意力分数的 70%, 因而预训练模型中的 retrieval head 等结构容易在推理剪枝时受损. (2) ***训练效率需求***: 高效处理长序列训练对现代 LLM 开发十分重要, 既包括在更长文档上预训练以提升模型能力, 也包括后续的长上下文 fine-tuning 和 reinforcement learning. 但现有稀疏注意力方法主要面向推理, 基本没有解决训练时的计算问题, 阻碍了更强长上下文模型的高效开发. 将现有稀疏注意力方法改用于训练, 还会暴露以下困难:

**不可训练的组件.** ClusterKV [Liu24zz] 中的 k-means 聚类、MagicPIG [Che25ab] 中基于 SimHash 的选择等离散操作, 会让计算图出现不连续. 梯度无法穿过这些不可训练组件流向 token 选择过程, 模型因而不能学习最优的稀疏模式.

**低效的反向传播.** 一些理论上可训练的稀疏注意力方法, 实际训练效率却很低. HashAttention [Des24z] 等方法采用 token 粒度选择, 计算注意力时必须从 KV cache 加载大量单独的 token. 这种不连续的内存访问无法高效适配 FlashAttention 一类快速注意力技术, 后者依赖连续内存访问和 blockwise 计算来获得高吞吐. 实现只能退回到硬件利用率较低的方式, 训练效率因此明显下降.

<span id="section-2-3"></span>

### 2.3 原生稀疏势在必行

推理效率和训练可行性上的这些局限, 促使我们从根本上重新设计稀疏注意力机制. 我们提出 NSA, 一个同时满足计算效率与训练要求的原生稀疏注意力框架. 下文将详述 NSA 的算法设计和算子实现.

<span id="section-3"></span>

## 3 方法

我们的技术方案同时覆盖算法设计和 kernel 优化. 接下来先介绍方法背景, 再给出 NSA 的整体框架和关键算法组件, 最后说明面向硬件优化的 kernel 设计, 以最大限度提高实际效率.

<span id="section-3-1"></span>

### 3.1 背景

**注意力机制.** 注意力机制广泛用于语言建模. 每个 query token $\mathbf{q}_t$ 会与之前所有 key $\mathbf{k}_{:t}$ 计算相关性分数, 再对 value $\mathbf{v}_{:t}$ 求加权和. 对长度为 $t$ 的输入序列, 注意力操作定义为:

<span id="equation-01"></span>

$$
\mathbf{o}_t = \mathrm{Attn}\left(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}\right)
$$

其中, $\mathrm{Attn}$ 表示注意力函数:

<span id="equation-02"></span>

$$
\mathrm{Attn}\left(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}\right) = \sum_{i=1}^t\frac{ \alpha_{t,i} \mathbf{v}_i}{\sum_{j=1}^t \alpha_{t,j}}, \quad \alpha_{t,i} = e^{\frac{\mathbf{q}_t^\top \mathbf{k}_i}{\sqrt{d_k}}}\,.
$$

这里, $\alpha_{t,i}$ 表示 $\mathbf{q}_t$ 与 $\mathbf{k}_i$ 之间的注意力权重, $d_k$ 是 key 的特征维度. 随着序列变长, 注意力计算在总计算成本中的占比越来越高, 给长上下文处理带来严峻挑战.

**算术强度.** 算术强度是计算操作数与内存访问量之比, 它从根本上影响硬件上的算法优化. 每款 GPU 都有一个由峰值算力与内存带宽之比决定的临界算术强度. 计算任务的算术强度高于临界值时, 受算力限制, 即 compute-bound; 低于临界值时, 受内存带宽限制, 即 memory-bound.

对 causal self-attention 而言, 训练和 prefill 阶段的 batch matrix multiplication 与注意力计算具有较高算术强度, 在现代加速器上属于 compute-bound. 自回归解码则不同: 每次前向传播只生成一个 token, 却需要加载完整的 KV cache, 因而算术强度低, 受内存带宽限制. 两类阶段的优化目标也不同: 训练和 prefill 要减少计算量, 解码则要减少内存访问.

<span id="section-3-2"></span>

### 3.2 整体框架

为发挥注意力天然稀疏模式的潜力, 我们针对每个 query $\mathbf{q}_t$, 用更紧凑、信息密度更高的表示 key-value 对 $\tilde{K}_t,\tilde{V}_t$ 取代[公式 1](#equation-01) 中的原始 key-value 对 $\mathbf{k}_{:t},\mathbf{v}_{:t}$. 优化后的注意力输出形式化定义如下:

<span id="equation-03"></span>

$$
\tilde{K}_t = f_K(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}), \quad \tilde{V}_t = f_V(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t})
$$

<span id="equation-04"></span>

$$
\mathbf{o}^*_t=\mathrm{Attn}\left(\mathbf{q}_t,\tilde{K}_t, \tilde{V}_t \right)
$$

$\tilde{K}_t,\tilde{V}_t$ 根据当前 query $\mathbf{q}_t$ 和上下文记忆 $\mathbf{k}_{:t},\mathbf{v}_{:t}$ 动态构造. 我们可以设计不同的映射策略, 得到多类 $\tilde{K}_t^c,\tilde{V}_t^c$, 再按下式合并:

<span id="equation-05"></span>

$$
\mathbf{o}^*_t = \sum_{c \in \mathcal{C}} g_t^c \cdot \mathrm{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c).
$$

如[图 2](#figure-02) 所示, NSA 有三种映射策略 $\mathcal{C}=\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}$, 分别对 key 和 value 执行压缩、选择与滑动窗口处理. $g_t^c\in[0,1]$ 是策略 $c$ 对应的 gate 分数, 由输入特征经 MLP 和 sigmoid activation 得到. 令 $N_t$ 表示重新映射后的 key/value 总数:

<span id="equation-06"></span>

$$
N_t = \sum_{c \in \mathcal{C}}\mathrm{size}[\tilde{K}^c_t].
$$

通过确保 $N_t\ll t$, 我们维持很高的稀疏率.

<span id="section-3-3"></span>

### 3.3 算法设计

本节介绍 $f_K$ 和 $f_V$ 三种重映射策略的设计: token 压缩、token 选择和滑动窗口.

<span id="section-3-3-1"></span>

#### 3.3.1 Token 压缩

把连续的 key 或 value block 聚合为 block 级表示, 即可得到包含整个 block 信息的压缩 key 和 value. 压缩 key 的形式化定义为:

<span id="equation-07"></span>

$$
\tilde{K}^\mathrm{cmp}_t = f_K^\mathrm{cmp}(\mathbf{k}_{:t}) = \left\{\varphi(\mathbf{k}_{i d+1: i d+l})\middle| 0\leqslant i\leqslant\left\lfloor\frac{t-l}{d}\right\rfloor\right\}
$$

其中, $l$ 是 block 长度, $d$ 是相邻 block 之间的滑动 stride, $\varphi$ 是带有 block 内 position encoding 的可学习 MLP, 用于把一个 block 内的 key 映射为单个压缩 key. $\tilde{K}_t^\mathrm{cmp}\in\mathbb{R}^{d_k\times\left\lfloor\frac{t-l}{d}\right\rfloor}$ 是由压缩 key 构成的 tensor. 通常取 $d<l$, 以减轻信息碎片化. 压缩 value $\tilde{V}_t^\mathrm{cmp}$ 采用类似形式. 压缩表示可以捕捉粒度更粗、层次更高的语义信息, 同时减轻注意力计算负担.

<span id="section-3-3-2"></span>

#### 3.3.2 Token 选择

只使用压缩 key 和 value 可能丢失重要的细粒度信息, 因而还需要有选择地保留单独的 key 和 value. 下面介绍一种高效的 token 选择机制, 它能以较低计算开销识别并保留最相关的 token.

**Blockwise 选择.** 我们以空间连续的 block 处理 key 和 value 序列, 原因有二: 硬件效率以及注意力分数固有的分布模式. ***Blockwise 选择是现代 GPU 上实现高效计算的关键.*** 与随机索引读取相比, 现代 GPU 对连续 block 访问的吞吐量高得多; blockwise 计算还可以充分利用 Tensor Core. 这种硬件特性使 blockwise 内存访问和计算成为高性能注意力实现的基本原则, FlashAttention 的分块设计就是一例. ***Blockwise 选择符合注意力分数固有的分布模式.*** 先前研究 [Jia24e] 表明, 注意力分数通常具有空间连续性, 相邻 key 的重要程度往往相近. [第 6.2 节](#section-6-2)的可视化也显示了这种空间连续模式.

具体实现时, 我们先把 key 和 value 序列划分为选择 block. 为找出注意力计算中最重要的 block, 需要为每个 block 分配重要性分数. 下面介绍 block 级重要性分数的计算方法.

**重要性分数计算.** 直接计算 block 重要性分数可能产生很大开销. 所幸, 压缩 token 的注意力计算会产生中间注意力分数, 可以利用它推导选择 block 的重要性分数:

<span id="equation-08"></span>

$$
\mathbf{p}_t^\mathrm{cmp} = \mathrm{Softmax}\left(\mathbf{q}_t^\top \tilde{K}_t^\mathrm{cmp}\right),
$$

其中, $\mathbf{p}_t^\mathrm{cmp}\in\mathbb{R}^{\left\lfloor\frac{t-l}{d}\right\rfloor+1}$ 是 $q_t$ 与压缩 key $\tilde{K}_t^\mathrm{cmp}$ 之间的注意力分数. 令 $l'$ 表示选择 block 的大小. 当压缩 block 与选择 block 采用同一种分块方式, 即 $l'=l=d$ 时, 可直接令 $\mathbf{p}_t^\mathrm{slc}=\mathbf{p}_t^\mathrm{cmp}$, 得到选择 block 的重要性分数 $\mathbf{p}_t^\mathrm{slc}$. 分块方式不同时, 我们根据二者的空间关系推导选择 block 的重要性分数. 给定 $l\leqslant l'$、$d\mid l$ 且 $d\mid l'$, 有:

<span id="equation-09"></span>

$$
\mathbf{p}_t^\mathrm{slc}[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d} -1} \mathbf{p}_t^\mathrm{cmp}\left[\frac{l'}{d}j -m -n \right],
$$

其中, $[\cdot]$ 是访问 vector element 的索引算子. 对采用 GQA 或 MQA 的模型, 同一组 query head 会共享 KV cache. 为尽量减少解码时的 KV cache 加载量, 这些 head 必须选择一致的 block. 一个 group 内各 head 共享的重要性分数定义为:

<span id="equation-10"></span>

$$
{\mathbf{p}_t^{\mathrm{slc}}}' = \sum_{h=1}^{H} \mathbf{p}_{t}^{\mathrm{slc}, (h)},
$$

其中, 上标中的 $(h)$ 表示 head 索引, $H$ 是每组的 query head 数. 这种聚合保证了同组 head 选择相同的 block.

**Top-$\pmb{n}$ block 选择.** 得到选择 block 的重要性分数后, 我们保留按该分数排序后的 top-$n$ 稀疏 block 中的 token:

<span id="equation-11"></span>

$$
\mathcal{I}_t = \{i \mid \mathrm{rank}({\mathbf{p}_t^\mathrm{slc}}'[i]) \leqslant n\}
$$

<span id="equation-12"></span>

$$
\tilde{K}^\mathrm{slc}_t = \mathrm{Cat}\left[\{\mathbf{k}_{il'+1:(i+1)l'}\mid i \in \mathcal{I}_t\}\right],
$$

其中, $\mathrm{rank}(\cdot)$ 表示降序排名位置, rank = 1 对应最高分; $\mathcal{I}_t$ 是被选 block 的索引集合; $\mathrm{Cat}$ 表示拼接操作. $\tilde{K}_t^\mathrm{slc}\in\mathbb{R}^{d_k\times nl'}$ 是由压缩 key 构成的 tensor. 细粒度 value $\tilde{V}_t^\mathrm{slc}$ 采用类似形式. 随后, 被选中的 key 和 value 按[公式 5](#equation-05) 与 $\mathbf{q}_t$ 一同参与注意力计算.

<span id="section-3-3-3"></span>

#### 3.3.3 滑动窗口

在注意力机制中, 局部模式通常适应得更快, 也可能主导学习过程, 使模型无法从压缩 token 和选择 token 中有效学习. 为此, 我们加入一条专门处理局部上下文的滑动窗口分支. 这样, 其他分支, 即压缩和选择分支, 就能专注学习各自的特征, 不会被局部模式走捷径. 具体而言, 我们在窗口 $w$ 中保留最近的 token $\tilde{K}_t^\mathrm{win}=\mathbf{k}_{t-w:t},\tilde{V}_t^\mathrm{win}=\mathbf{v}_{t-w:t}$, 并把不同信息源, 即压缩 token、选择 token 和滑动窗口, 的注意力计算放在独立分支中. 各分支输出再由可学习的 gating 机制聚合. 为进一步避免分支间的 shortcut learning, 同时只增加很少计算开销, 我们为三条分支分别提供独立的 key 和 value. 这种架构可防止局部模式与长程模式识别之间的梯度干扰, 从而稳定学习, 额外开销也很小.

得到三类 key 和 value, 即 $\tilde{K}_t^\mathrm{cmp},\tilde{V}_t^\mathrm{cmp}$; $\tilde{K}_t^\mathrm{slc},\tilde{V}_t^\mathrm{slc}$; $\tilde{K}_t^\mathrm{win},\tilde{V}_t^\mathrm{win}$ 后, 按[公式 5](#equation-05) 计算最终的注意力输出. 它们与上述压缩、选择和滑动窗口机制共同构成 NSA 的完整算法框架.

<span id="section-3-4"></span>

### 3.4 Kernel 设计

为在训练和 prefill 阶段达到 FlashAttention 级别的加速, 我们基于 Triton 实现了适配硬件的稀疏注意力 kernel. MHA 在解码时内存开销大、效率低, 因而我们遵循当前先进 LLM 的做法, 重点面向 GQA、MQA 等共享 KV cache 的架构. 压缩注意力和滑动窗口注意力可以直接适配现有 FlashAttention-2 kernel, 但稀疏选择注意力需要专门设计. 如果沿用 FlashAttention 的方式, 把时间上连续的 query block 加载到 SRAM, 由于一个 block 内的 query 可能需要互不相交的 KV block, 内存访问会很低效. 我们的核心优化是改用另一种 query 分组策略: 对 query 序列上的每个位置, 把同一 GQA group 内的所有 query head 一并载入 SRAM, 因为它们共享相同的稀疏 KV block. [图 3](#figure-03) 展示了前向传播实现. 该 kernel 架构有以下特点:

1. **以 group 为中心加载数据.** 每次 inner loop 中, 加载位置 $t$ 上该 group 内所有 head 的 query $Q\in\mathbb{R}^{[h,d_k]}$, 以及它们共享的稀疏 key/value block 索引 $\mathcal{I}_t$.

2. **共享 KV 获取.** 在 inner loop 中, 按 $\mathcal{I}_t$ 连续加载 key/value block 到 SRAM, 形成 $K\in\mathbb{R}^{[B_k,d_k]}$、$V\in\mathbb{R}^{[B_k,d_v]}$, 以尽量减少内存加载. 其中, $B_k$ 是满足 $B_k\mid l'$ 的 kernel block size.

3. **在 grid 上执行 outer loop.** 不同 query block 的 inner-loop 长度, 即与被选 block 数 $n$ 成正比的长度, 几乎一致, 因而把 query/output loop 放入 Triton grid scheduler, 可以简化并优化 kernel.

该设计通过两点让算术强度接近最优: (1) 在 group 内共享, 消除多余的 KV 传输; (2) 在 GPU streaming multiprocessor 之间平衡计算 workload.

<span id="figure-03"></span>

![NSA 的 kernel 设计. Kernel 按 GQA group 加载 query (Grid Loop), 获取相应的稀疏 KV block (Inner Loop), 并在 SRAM 上执行注意力计算. 绿色 block 表示 SRAM 上的数据, 蓝色 block 表示 HBM 上的数据.](./native-sparse-attention/figure-03.png)

**图 3. NSA 的 kernel 设计. Kernel 按 GQA group 加载 query (Grid Loop), 获取相应的稀疏 KV block (Inner Loop), 并在 SRAM 上执行注意力计算. 绿色 block 表示 SRAM 上的数据, 蓝色 block 表示 HBM 上的数据.**

<span id="section-4"></span>

## 4 实验

我们从三个方面评测 NSA: (1) 通用 benchmark 性能; (2) 长上下文 benchmark 性能; (3) chain-of-thought 推理性能, 并与 Full Attention baseline 和先进的稀疏注意力方法比较. 训练和推理速度的详细分析放在[第 5 节](#section-5).

<span id="section-4-1"></span>

### 4.1 预训练设置

遵循先进 LLM 的常见做法, 实验 backbone 结合 Grouped-Query Attention (GQA) 与 Mixture-of-Experts (MoE), 总参数量为 $27\mathrm{B}$, 激活参数量为 $3\mathrm{B}$. 模型包含 30 层, hidden dimension 为 2560. GQA 设置 4 个 group, 共 64 个 attention head. 每个 head 的 query、key、value hidden dimension 分别设为 $d_q=d_k=192$ 和 $d_v=128$. MoE 采用 DeepSeekMoE [Dai24, Dee24] 结构, 包含 72 个 routed expert 和 2 个 shared expert, top-k expert 数设为 6. 为保持训练稳定, 第一层的 MoE 换成 SwiGLU 形式的 MLP. 该架构在计算成本和模型性能之间取得了有效平衡.

NSA 的压缩 block size 设为 $l=32$, 滑动 stride 为 $d=16$, 选择 block size 为 $l'=64$, 选择 block 数为 $n=16$, 其中固定激活第 1 个初始 block 和 2 个局部 block; 滑动窗口大小为 $w=512$. Full Attention 与稀疏注意力模型都先在 $270\mathrm{B}$ 个长度为 $8\mathrm{k}$ 的文本 token 上预训练, 随后在长度为 $32\mathrm{k}$ 的文本上继续训练并执行 supervised fine-tuning, 同时使用 YaRN [Pen23] 完成长上下文适配. 两个模型都训练至完全收敛, 以保证比较公平. 如[图 4](#figure-04) 所示, NSA 与 Full Attention baseline 的预训练 loss curve 都稳定、平滑下降, 而 NSA 始终优于 Full Attention 模型.

<span id="figure-04"></span>

![27B 参数模型上 Full Attention 与 NSA 的预训练 loss 对比. 两个模型都稳定收敛, NSA 的 loss 更低.](./native-sparse-attention/figure-04.png)

**图 4. 27B 参数模型上 Full Attention 与 NSA 的预训练 loss 对比. 两个模型都稳定收敛, NSA 的 loss 更低.**

<span id="table-01"></span>

![Full Attention baseline 与 NSA 在通用 benchmark 上的预训练性能对比, 覆盖知识任务 (MMLU、MMLU-PRO、CMMLU)、推理任务 (BBH、GSM8K、MATH、DROP) 和代码任务 (MBPP、HumanEval). NSA 虽然高度稀疏, 仍在多数 benchmark 上取得更好的平均表现.](./native-sparse-attention/table-01.png)

**表 1. Full Attention baseline 与 NSA 在通用 benchmark 上的预训练性能对比, 覆盖知识任务 (MMLU、MMLU-PRO、CMMLU)、推理任务 (BBH、GSM8K、MATH、DROP) 和代码任务 (MBPP、HumanEval). NSA 虽然高度稀疏, 仍在多数 benchmark 上取得更好的平均表现.**

<span id="section-4-2"></span>

### 4.2 Baseline 方法

除了 Full Attention, 我们还评测了几种先进的推理阶段稀疏注意力方法: H2O [Zha23g]、infLLM [Xia24z]、Quest [Tan24] 和 Exact-Top. Exact-Top 先计算完整注意力分数, 为每个 query 选出分数最高的 top-$n$ 个 key, 再只在这些位置计算注意力. 这些方法覆盖了多种稀疏注意力范式, 包括 KV-cache 淘汰、query-aware 选择和精确 top-$n$ 稀疏选择.

通用评测中, 大多数样本的长度都在稀疏注意力 baseline 的局部上下文窗口内, 因而这些方法实际上等价于 Full Attention. 所以, 这一场景只给出 NSA 和 Full Attention baseline 的对比. 长上下文评测会比较全部 baseline, 并统一所有稀疏注意力方法的稀疏度, 以保证公平. Chain-of-thought 推理评测需要长文本 supervised fine-tuning, 其他稀疏注意力 baseline 不支持训练, 因而只与 Full Attention 比较.

<span id="section-4-3"></span>

### 4.3 性能对比

<span id="table-02"></span>

![NSA 与各 baseline 在 LongBench 上的性能对比, 包括单文档 QA、多文档 QA、合成任务和代码任务子集. NSA 超过了包括 Full Attention 在内的大多数 baseline.](./native-sparse-attention/table-02.png)

**表 2. NSA 与各 baseline 在 LongBench 上的性能对比, 包括单文档 QA、多文档 QA、合成任务和代码任务子集. NSA 超过了包括 Full Attention 在内的大多数 baseline.**

**通用评测.** 我们在涵盖知识、推理和代码能力的一组综合 benchmark 上评测预训练后的 NSA 与 Full Attention baseline, 包括 MMLU [Hen20]、MMLU-PRO [Wan24c]、CMMLU [Li23e]、BBH [Suz22]、GSM8K [Cob21]、MATH [Hen20]、DROP [Dua19]、MBPP [Aus21] 和 HumanEval [Che21]. 结果见[表 1](#table-01). NSA 尽管采用稀疏注意力, 整体性能仍然更好, 9 项指标中有 7 项超过了包括 Full Attention 在内的全部 baseline. 这说明, NSA 在短序列上即使不能充分发挥效率优势, 性能依然很强. 尤其是在推理类 benchmark 上, NSA 提升明显, DROP 为 +0.042, GSM8K 为 +0.034. 这表明预训练帮助模型形成了专门的注意力机制. 稀疏注意力预训练迫使模型集中处理最重要的信息, 过滤无关注意力路径带来的噪声, 因而可能提高性能. 在不同评测中都保持稳定表现, 也说明 NSA 可以作为稳健的通用架构.

**长上下文评测.** 如[图 5](#figure-05) 所示, NSA 在 64k 上下文的 needle-in-a-haystack [Kam23z] 测试中, 对所有上下文位置都达到 100% 检索准确率. 这一表现来自分层稀疏注意力设计: 压缩 token 用于高效扫描全局上下文, 选择 token 用于精确检索局部信息. 粗粒度压缩以较低计算成本识别相关上下文 block, 再在选中 token 上执行 token 级注意力, 保留关键的细粒度信息. 因而 NSA 可以兼顾全局感知与局部精度.

<span id="figure-05"></span>

![64k 上下文长度下, 不同上下文位置的 needle-in-a-haystack 检索准确率. NSA 依靠分层稀疏注意力设计达到 100% 准确率.](./native-sparse-attention/figure-05.png)

**图 5. 64k 上下文长度下, 不同上下文位置的 needle-in-a-haystack 检索准确率. NSA 依靠分层稀疏注意力设计达到 100% 准确率.**

我们还在 LongBench [Bai23] 上将 NSA 与先进稀疏注意力方法及 Full Attention baseline 比较. 为统一稀疏度, 所有稀疏注意力 baseline 中每个 query 激活的 token 数都设为 2560, 等于 NSA 处理 32k 序列时的平均激活 token 数. 按照 StreamLLM [Xia24a] 的做法, 该预算包含开头的 128 个 token 和局部的 512 个 token. LongBench 中有些子集在所有模型上的得分都很低, 难以形成有意义的比较, 因而没有纳入. 如[表 2](#table-02) 所示, NSA 的平均分最高, 达到 0.469, 超过所有 baseline, 比 Full Attention 高 0.032, 比 Exact-Top 高 0.046. 这一提升源于两项创新: (1) 原生稀疏注意力设计让稀疏模式可以在预训练中端到端优化, 使稀疏注意力模块与模型其他组件同步适应; (2) 分层稀疏注意力机制在局部和全局信息处理之间取得平衡.

NSA 在需要对长上下文进行复杂推理的任务上尤其出色: 多跳 QA 任务 HPQ 和 2Wiki 分别比 Full Attention 高 0.087 和 0.051; 代码理解任务 LCC 比 baseline 高 0.069; 段落检索任务 PassR-en 比其他方法高 0.075. 这些结果表明, NSA 能够处理多种长上下文挑战, 而原生预训练的稀疏注意力还能为学习适合任务的最优模式带来额外收益.

**Chain-of-thought 推理评测.** 为评估 NSA 与先进下游训练范式的兼容性, 我们考察它能否通过 post-training 获得 chain-of-thought 数学推理能力. Reinforcement learning 在较小模型上的效果有限, 因而我们从 DeepSeek-R1 蒸馏知识, 使用 10B 个长度为 32k 的数学推理轨迹 token 执行 supervised fine-tuning (SFT). 由此得到两个可比模型: Full Attention-R, 即 Full Attention baseline, 以及 NSA-R, 即稀疏变体. 两个模型都在较难的 American Invitational Mathematics Examination (AIME 24) benchmark 上评测. 采样温度设为 0.7, top-$p$ 为 0.95, 每道题生成 16 个回答并取平均分. 为检验推理深度的影响, 实验采用 8k 和 16k token 两种生成上下文上限, 观察更长的推理链能否提高准确率. 模型预测样例见[第 9 节](#section-9).

<span id="table-03"></span>

![Supervised fine-tuning 后的 AIME 指令评测. 在 8k 和 16k 序列长度下, NSA-R 都优于 Full Attention-R.](./native-sparse-attention/table-03.png)

**表 3. Supervised fine-tuning 后的 AIME 指令评测. 在 8k 和 16k 序列长度下, NSA-R 都优于 Full Attention-R.**

如[表 3](#table-03) 所示, 在 8k 上下文设置下, NSA-R 的准确率比 Full Attention-R 高 0.075; 上下文增至 16k 后, 仍领先 0.054. 结果验证了原生稀疏注意力的两点优势: (1) 预训练得到的稀疏注意力模式能高效捕捉复杂数学推导所需的长程逻辑依赖; (2) 适配硬件的架构设计维持了足够的上下文密度, 可以支持越来越深的推理而不发生 catastrophic forgetting. NSA-R 在两种上下文长度下都稳定胜出, 说明只要从训练阶段原生集成, 稀疏注意力同样适用于先进推理任务.

<span id="section-5"></span>

## 5 效率分析

我们在 8 张 A100 GPU 上比较 NSA 与 Full Attention 的计算效率. 模型同样采用 4 个 GQA group, 每组 16 个 head, query/key dimension 为 $d_k=192$, value dimension 为 $d_v=128$. 设置与[第 4 节](#section-4)一致: NSA 压缩 block size 为 $l=32$, 滑动 stride 为 $d=16$, 选择 block size 为 $l'=64$, 选择 block 数为 $n=16$, 滑动窗口大小为 $w=512$.

<span id="figure-06"></span>

![基于 Triton 的 NSA kernel 与基于 Triton 的 FlashAttention-2 kernel 对比. NSA 实现在所有上下文长度下都显著降低延迟, 且输入越长, 改善越明显.](./native-sparse-attention/figure-06.png)

**图 6. 基于 Triton 的 NSA kernel 与基于 Triton 的 FlashAttention-2 kernel 对比. NSA 实现在所有上下文长度下都显著降低延迟, 且输入越长, 改善越明显.**

<span id="section-5-1"></span>

### 5.1 训练速度

为在相同 backend 上公平比较速度, 我们将基于 Triton 的 NSA 注意力和 Full Attention 实现与基于 Triton 的 FlashAttention-2 比较. 如[图 6](#figure-06) 所示, 上下文越长, NSA 的加速幅度越大; 上下文长度为 64k 时, 前向最高加速 9.0$\times$, 反向最高加速 6.0$\times$. 这一加速来自适配硬件的算法设计, 它从两方面提高稀疏注意力架构的效率: (1) blockwise 内存访问通过 coalesced load 最大限度利用 Tensor Core; (2) kernel 中精细的 loop 调度消除多余的 KV 传输.

<span id="table-04"></span>

![解码时每次注意力操作的内存访问量, 以等价 token 数表示. 解码的算术强度低, 并且受内存带宽限制, 因而预期加速比与内存访问量大致呈线性关系.](./native-sparse-attention/table-04.png)

**表 4. 解码时每次注意力操作的内存访问量, 以等价 token 数表示. 解码的算术强度低, 并且受内存带宽限制, 因而预期加速比与内存访问量大致呈线性关系.**

<span id="section-5-2"></span>

### 5.2 解码速度

注意力的解码速度主要取决于内存访问瓶颈, 而这一瓶颈与 KV cache 加载量密切相关. 每个解码 step 中, NSA 至多只需加载 $\left\lfloor\frac{s-l}{d}\right\rfloor$ 个压缩 token、$nl'$ 个选择 token 和 $w$ 个相邻 token, 其中 $s$ 是缓存的序列长度. 如[表 4](#table-04) 所示, 解码长度越大, 本方法的延迟降幅越明显; 上下文长度为 64k 时最高加速 11.6$\times$. 序列越长, 内存访问效率上的优势同样越大.

<span id="section-6"></span>

## 6 讨论

本节回顾 NSA 的开发过程, 并总结探索不同稀疏注意力策略时得到的重要认识. NSA 已经取得了良好效果, 但理解其他策略遇到的困难、分析注意力模式, 能为后续研究提供有价值的背景. 我们先讨论促成当前设计选择的其他 token 选择策略及其问题, 再通过可视化观察注意力分布模式.

<span id="section-6-1"></span>

### 6.1 其他 Token 选择策略的困难

在设计 NSA 之前, 我们曾尝试把现有稀疏注意力方法用于训练阶段. 这些尝试遇到了不同问题, 促使我们另行设计稀疏注意力架构:

**基于 key 聚类的策略.** 我们考察了 ClusterKV [Liu24zz] 等聚类策略. 这类方法把同一 cluster 的 key 和 value 存放在连续内存区域. 理论上可以用于训练和推理, 但有三个明显问题: (1) 动态聚类机制会带来不可忽略的计算开销; (2) cluster 之间的不均衡增加了算子优化难度, 尤其在 Mixture-of-Experts (MoE) 系统中, Expert Parallelism (EP) group 的执行时间偏斜会造成持续的 load imbalance; (3) 实现上必须定期重新聚类, 训练也只能按 chunk 顺序进行. 这些因素共同形成严重瓶颈, 限制了它们在实际部署中的效果.

<span id="figure-07"></span>

![在 3B 参数模型上比较 Full Attention 与不同 token 选择策略的训练 loss. NSA 的表现更好.](./native-sparse-attention/figure-07.png)

**图 7. 在 3B 参数模型上比较 Full Attention 与不同 token 选择策略的训练 loss. NSA 的表现更好.**

<span id="figure-08"></span>

![Full Attention transformer 的 Attention Map 可视化. 颜色越浅, 注意力值越高. 图中注意力分数呈 blockwise 聚集分布.](./native-sparse-attention/figure-08.png)

**图 8. Full Attention transformer 的 Attention Map 可视化. 颜色越浅, 注意力值越高. 图中注意力分数呈 blockwise 聚集分布.**

**其他 blockwise 选择策略.** 我们也考察了 Quest [Tan24] 和 InfLLM [Xia24z] 等不同于 NSA 的 blockwise key/value 选择策略. 这些方法为每个 KV block 计算重要性分数, 再依据它与 $q_t$ 的相似度选择 top-$n$ block. 现有方法有两个关键问题: (1) 选择操作不可微, 基于神经网络的重要性分数计算只能依靠辅助 loss, 这会增加算子开销, 并且常常降低模型性能; (2) 无参数的启发式重要性分数计算策略召回率较低, 性能不理想. 我们在架构相近的 3B 参数模型上评测这两种方法, 并把 loss curve 与 NSA、Full Attention 比较. 对基于辅助 loss 的选择方法, 我们为每个 token 增加 query, 为每个 block 增加代表 key, 用于估计 block 重要性分数. 我们对每个 key block 内的注意力分数取 mean pooling, 得到 block 级监督信号, 再用 KL divergence 监督 block 重要性预测. 为适配高效解码, query 仍保持单独 token 粒度, 不按 block 取平均. 这种基于辅助 loss 的重要性估计在概念上与 SeerAttention [Gao24z] 相似. 对无参数的启发式选择方法, 我们沿用 Quest 的策略, 直接计算 query 与 key chunk 各坐标 min-max 的乘积来执行选择, 不增加参数. 我们还尝试了 cold-start 训练: 前 1000 step 使用 Full Attention, 此后切换到启发式 blockwise 选择. 如[图 7](#figure-07) 所示, 两种方法的 loss 都更差.

<span id="section-6-2"></span>

### 6.2 可视化

为了寻找 transformer 注意力分布中的潜在模式, 并为设计提供启发, 我们在[图 8](#figure-08) 中可视化了预训练 27B Full Attention 模型的 attention map. 图中呈现出有趣的现象: 注意力分数往往按 block 聚集, 相邻 key 的分数通常也很接近. 这一观察启发了 NSA 的设计, 表明根据空间连续性选择 key block 可能是一条可行路径. Blockwise 聚集现象说明, 序列中相邻 token 与 query token 之间可能具有某种共同的语义关系, 但这种关系究竟是什么, 还需要进一步研究. 因此, 我们开始探索在连续 token block 而不是单独 token 上运行的稀疏注意力机制, 以提高计算效率并保留高注意力模式.

<span id="section-7"></span>

## 7 相关工作

我们回顾通过稀疏注意力提高注意力计算效率的现有方法. 按照核心策略, 这些方法大体可以分为三类: (1) 固定稀疏模式; (2) 动态 token 剪枝; (3) query-aware 选择. 下面分别介绍各类的代表性工作.

<span id="section-7-1"></span>

### 7.1 固定稀疏模式

SlidingWindow 是常用方法, 只允许 query 在固定窗口内计算注意力. StreamingLLM [Xia24a] 把 attention sink 与局部窗口结合起来, 处理连续文本流. MoA [Fu24z] 和 DuoAttention [Xia24d] 也采用类似的局部信息和 sink 信息来建模长序列. Longformer [Bel20] 交替使用局部窗口注意力与全局 token, 处理长序列. NSA 与这些方法不同, 它不依赖预定义的稀疏模式, 而是自动学习模式, 从而发挥完整上下文的潜力.

<span id="section-7-2"></span>

### 7.2 动态 Token 剪枝

一些方法通过动态剪枝 KV-cache, 降低推理时的内存和计算成本. H2O [Zha23g]、BUZZ [Zha24za] 和 SepLLM [Che24z] 采用自适应方案, 减少解码时的 KV-cache 内存用量, 动态淘汰对未来预测不重要的 token. FastGen 和 HeadKV [Fu24za] 为不同 attention head 分配不同策略, 以优化计算. SnapKV [Li24c] 有选择地保留最关键的特征, 由此剪枝 token、缩减 KV-cache, 提高内存使用效率. 与这些专注推理的方法不同, NSA 从训练阶段就原生加入稀疏性.

<span id="section-7-3"></span>

### 7.3 Query-Aware 选择

另一些工作根据 query 选择 token, 在保持注意力质量的同时减少计算量. Quest [Tan24] 采用 blockwise 选择: 用 query 与 key chunk 各坐标 min-max 的乘积估计每个 chunk 的重要性. InfLLM [Xia24z] 结合固定模式和检索, 同时维护 attention sink、局部上下文和可检索 chunk, 并从每个 chunk 中选择代表 key 来估计重要性. HashAttention [Des24z] 把关键 token 识别表述为推荐问题, 使用学习到的函数把 query 和 key 映射到 Hamming space. ClusterKV [Liu24zz] 先对 key 聚类, 再按 query-cluster 相似度选出最相关的 cluster, 以实现稀疏性. MInference [Jia24e] 和 TokenSelect [Wu24z] 根据 token 级重要性分数选择参与计算的 KV 对. SeerAttention [Gao24z] 把 query 和 key 划分为空间 block, 再执行 blockwise 选择以提高计算效率. 与这些方法相比, NSA 在训练、prefill 和解码的整个模型生命周期中都能执行适配硬件的稀疏注意力计算.

<span id="section-8"></span>

## 8 结论

本文提出 NSA, 一种面向高效长上下文建模的原生可训练稀疏注意力机制. NSA 把分层 token 压缩和选择与适配硬件的实现结合起来, 在不牺牲性能的前提下, 显著加快训练和推理. 在 27B 参数模型上的大量实验表明, NSA 在通用 benchmark、长上下文 benchmark 和 chain-of-thought 推理任务上均达到或超过 Full Attention baseline, 并在 64k 序列上实现最高 9.0$\times$ 的前向加速、6.0$\times$ 的反向加速和 11.6$\times$ 的解码加速. 这些结果说明, 原生稀疏注意力可以作为下一代语言模型高效长上下文建模的一条可行路径.

<span id="section-9"></span>

## 9 AIME 结果样例

**样例 1 — Prompt**

> “|User| 设 $x,y,z$ 为正实数, 且满足下列方程组:
>
> $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$
>
> $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$
>
> $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$
>
> 那么 $\left|\log_2(x^4y^3z^2)\right|=\frac{m}{n}$, 其中 $m,n$ 是互质的正整数. 求 $m+n$.
>
> 请逐步推理, 并把最终答案放在 $\boxed{}$ 中. |Assistant|”

**我们的结果**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (2275 tokens)
>
> 给定方程组:
>
> 1. $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$
> 2. $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$
> 3. $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$
>
> 将这些对数方程化为指数形式:
>
> 1. $\frac{x}{yz}=\sqrt{2}$
> 2. $\frac{y}{xz}=2^{1/3}$
> 3. $\frac{z}{xy}=2^{1/4}$
>
> 分别用 $x,y,z$ 表示各变量:
>
> 1. $x=\sqrt{2}\cdot yz$
> 2. $y=2^{1/3}\cdot xz$
> 3. $z=2^{1/4}\cdot xy$
>
> 将这些表达式相互代入, 得到:
>
> - 由 $x=\sqrt{2}\cdot yz$ 和 $y=2^{1/3}\cdot xz$, 可得 $x=\sqrt{2}\cdot(2^{1/3}\cdot xz)\cdot z$, 因而 $z=2^{-5/12}$.
> - 把 $z$ 代入 $y$ 的方程, 得 $y=2^{-3/8}$.
> - 把 $z,y$ 代入 $x$ 的方程, 得 $x=2^{-7/24}$.
>
> 接着计算 $x^4y^3z^2$:
>
> - $x^4=(2^{-7/24})^4=2^{-7/6}$
> - $y^3=(2^{-3/8})^3=2^{-9/8}$
> - $z^2=(2^{-5/12})^2=2^{-5/6}$
>
> 三者相乘: $x^4y^3z^2=2^{-7/6}\cdot2^{-9/8}\cdot2^{-5/6}=2^{-75/24}=2^{-25/8}$.
>
> 取以 2 为底的对数: $\log_2(x^4y^3z^2)=-25/8$.
>
> 绝对值为 $|\log_2(x^4y^3z^2)|=25/8$.
>
> 因此 $m=25,n=8$, 所以 $m+n=33$.
>
> $\boxed{33}$

**Baseline 结果**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (9392 tokens)
>
> 给定对数方程组:
>
> $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$, $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$, $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$.
>
> 将它们化为指数形式:
>
> $\sqrt{2}=\frac{x}{yz}\implies x=\sqrt{2}\cdot yz$,
>
> $2^{\frac{1}{3}}=\frac{y}{xz}\implies y=2^{\frac{1}{3}}\cdot xz$,
>
> $2^{\frac{1}{4}}=\frac{z}{xy}\implies z=2^{\frac{1}{4}}\cdot xy$.
>
> 相互代入, 得到:
>
> 1. $x=\sqrt{2}\cdot yz$
> 2. $y=2^{\frac{1}{3}}\cdot xz$
> 3. $z=2^{\frac{1}{4}}\cdot xy$
>
> 逐步求解这些方程, 得到:
>
> - $z=2^{-5/12}$
> - $xy=2^{-2/3}$
> - $y=2^{-1/24}$
> - $x=2^{1/24}$
>
> 但检查第三个方程后, 我们发现这个方程组可能无解. 因此改用矩阵求逆来求解:
>
> $\begin{cases}\log_2(x)-\log_2(y)-\log_2(z)=\frac{1}{2}\\ \log_2(y)-\log_2(x)-\log_2(z)=\frac{1}{3}\\ \log_2(z)-\log_2(x)-\log_2(y)=\frac{1}{4}\end{cases}$
>
> 令 $A=\log_2(x),B=\log_2(y),C=\log_2(z)$. 方程组变为:
>
> $\begin{cases}A-B-C=\frac{1}{2}\\ B-A-C=\frac{1}{3}\\ C-A-B=\frac{1}{4}\end{cases}$
>
> 用 Cramer 法则和矩阵求逆求解. 矩阵行列式为 -2, 解为 $A=\frac{61}{48},B=\frac{13}{16},C=-\frac{1}{6}$.
>
> 因而 $x=2^{61/48},y=2^{13/16},z=2^{-1/6}$.
>
> 需要求 $\left|\log_2(x^4y^3z^2)\right|$:
>
> $4\log_2x+3\log_2y+2\log_2z=4\cdot\frac{61}{48}+3\cdot\frac{13}{16}+2\cdot\left(-\frac{1}{6}\right)$.
>
> 化为公分母 48: $4\cdot\frac{61}{48}=\frac{244}{48}$, $3\cdot\frac{13}{16}=\frac{117}{48}$, $2\cdot\left(-\frac{1}{6}\right)=-\frac{16}{48}$.
>
> 相加可得 $\frac{244}{48}+\frac{117}{48}-\frac{16}{48}=\frac{345}{48}=\frac{115}{16}$.
>
> $\frac{115}{16}$ 已是最简分数, 所以 $m=115,n=16$. 因此 $m+n=115+16=131$.
>
> $\boxed{131}$

**样例 2 — Prompt**

> “|User| 实数 $x,y$ 满足 $x,y>1$ 以及 $\log_x(y^x)=\log_y(x^{4y})=10$.
>
> $xy$ 的值是多少?
>
> 请逐步推理, 并把最终答案放在 $\boxed{}$ 中. |Assistant|”

**我们的结果**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (15147 tokens)
>
> 给定方程:
>
> 1. $\log_x(y^x)=10$
> 2. $\log_y(x^{4y})=10$
>
> 先对两个方程应用对数的幂法则.
>
> 对第一个方程: $\log_x(y^x)=x\cdot\log_x(y)=10$.
>
> 令 $z=\log_y(x)$. 则 $\log_x(y)=\frac{1}{z}$. 代入方程, 得 $x\cdot\frac{1}{z}=10\implies x=10z$.
>
> 对第二个方程: $\log_y(x^{4y})=y\cdot\log_y(x^4)=10$.
>
> 使用幂法则, 得 $y\cdot4\log_y(x)=10\implies4y\log_y(x)=10$.
>
> 因为 $\log_y(x)=z$, 可化简为 $4yz=10\implies yz=\frac{5}{2}$.
>
> 现在有两个表达式:
>
> 1. $x=10z$
> 2. $y=\frac{5/2}{z}$
>
> 把 $x=10z$ 和 $y=\frac{5/2}{z}$ 代入乘积 $xy$, 得 $xy=(10z)\cdot\left(\frac{5/2}{z}\right)=10z\cdot\frac{5}{2z}=10\cdot\frac{5}{2}=25$.
>
> 因此 $xy=\boxed{25}$.

**Baseline 结果**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (16223 tokens)
>
> 题目给出实数 $x,y>1$, 满足 $\log_x(y^x)=10$ 和 $\log_y(x^4y)=10$, 需要求 $xy$.
>
> 先使用换底公式, 把对数方程化为指数形式.
>
> 1. $\log_x(y^x)=10$
>
> $x\log_x(y)=10$.
>
> 根据换底公式, $\log_x(y)=\frac{\ln y}{\ln x}$. 因为 $\log_yx=\frac{9}{4}$, 所以 $\log_xy=\frac{4}{9}$. 因而 $x\cdot\frac{4}{9}=10\implies x=\frac{10\cdot9}{4}=\frac{90}{4}=\frac{45}{2}$.
>
> 2. $\log_y(x^4y)=10$
>
> $4\log_yx+1=10\implies4\log_yx=9\implies\log_yx=\frac{9}{4}$.
>
> 再用换底公式, $\log_xy=\frac{4}{9}$.
>
> 接着用彼此表示 $x,y$: $x=\frac{45}{2}$, $y=\left(\frac{45}{2}\right)^{4/9}$.
>
> 计算乘积 $xy$: $xy=\left(\frac{45}{2}\right)\cdot\left(\frac{45}{2}\right)^{4/9}=\left(\frac{45}{2}\right)^{1+4/9}=\left(\frac{45}{2}\right)^{13/9}$.
>
> 但重新检查后, 改用 $x=2^a,y=2^b$, 得到:
>
> $\log_x(y^x)=10\implies\frac{x\log_xy}{\log_xy}=10\implies x\cdot\frac{4}{9}=10\implies x=\frac{45}{2}$.
>
> $\log_y(x^4y)=10\implies\frac{4\log_yx+1}{\log_yx}=10\implies4\cdot\frac{9}{4}+1=10\implies9+1=10\quad\mathrm{consistent}$.
>
> 最后, 乘积 $xy$ 可化简为 $xy=\left(\frac{45}{2}\right)$.
