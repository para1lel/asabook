---
title: 'Native Sparse Attention'
createTime: 2026/09/06 14:00:00
permalink: /en/papers/native-sparse-attention/
pageClass: paper-reading
---

> [Jingyang Yuan](https://dblp.org/pid/244/7491.html) [+internship], [Huazuo Gao](https://dblp.org/pid/366/3356.html), [Damai Dai](https://dblp.org/pid/199/2097.html), [Junyu Luo](https://dblp.org/pid/198/0850-2.html), [Liang Zhao](https://dblp.org/pid/63/5422-26.html), [Zhengyan Zhang](https://aclanthology.org/people/zhengyan-zhang/unverified/), [Zhenda Xie](https://dblp.org/pid/239/8676.html), [Yuxing Wei](https://aclanthology.org/people/yuxing-wei/unverified/), [Lean Wang](https://aclanthology.org/people/lean-wang/), [Zhiping Xiao](https://dblp.org/pid/176/5397-1.html), [Yuqing Wang](https://aclanthology.org/people/yuqing-wang/), [Chong Ruan](https://dblp.org/pid/159/9956.html), [Ming Zhang](https://dblp.org/pid/73/1844-4.html), [Wenfeng Liang](https://dblp.org/pid/59/9456.html), and [Wangding Zeng](https://dblp.org/pid/315/5319.html). First submitted to arXiv on February 16, 2025; current version v2. This reading edition transcribes [Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention](https://arxiv.org/abs/2502.11089v2), later published as an ACL 2025 Best Paper (pages 23078–23097). <a href="/paper/native-sparse-attention.pdf" target="_blank">Original PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2502.11089). [ACL paper](https://doi.org/10.18653/v1/2025.acl-long.1126). [TeX source](https://export.arxiv.org/e-print/2502.11089v2). The original PDF remains authoritative for the exact print layout and bibliography.

[+internship]: Contribution during internship at DeepSeek-AI.

## Abstract

Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges.

Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities.

We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling.

NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance.

As shown in [Figure 1](#figure-01), experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning.

Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle.

<span id="section-1"></span>

## 1 Introduction
<span id="figure-01"></span>

![Comparison of performance and efficiency between Full Attention model and our NSA. Left: Despite being sparse, NSA surpasses Full Attention baseline on average across general benchmarks, long-context tasks, and reasoning evaluation. Right: For 64k-length sequence processing, NSA achieves substantial computational speedup compared to Full Attention in all stages: decoding, forward propagation, and backward propagation.](../../papers/native-sparse-attention/figure-01.png)

**Figure 1. Comparison of performance and efficiency between Full Attention model and our NSA. Left: Despite being sparse, NSA surpasses Full Attention baseline on average across general benchmarks, long-context tasks, and reasoning evaluation. Right: For 64k-length sequence processing, NSA achieves substantial computational speedup compared to Full Attention in all stages: decoding, forward propagation, and backward propagation.**

The research community increasingly recognizes long-context modeling as a crucial capability for next-generation large language models, driven by diverse real-world applications ranging from in-depth reasoning [Zel22a, Dee25c], repository-level code generation [Zha23z, Zha24zz] and multi-turn autonomous agent systems [Par23]. Recent breakthroughs, including OpenAI’s o-series models, DeepSeek-R1 [Dee25c], and Gemini 1.5 Pro [Tea24a], enabling models to process entire codebases, lengthy documents, maintain coherent multi-turn conversations over thousands of tokens, and perform complex reasoning across long-range dependencies. However, the high complexity [Zah20] of vanilla Attention [Vas17] mechanisms emerges as a critical latency bottleneck as sequence length increases. Theoretical estimates indicate that attention computation with softmax architectures accounts for 70–80% of total latency when decoding 64k-length contexts, underscoring the urgent need for more efficient attention mechanisms.

A natural approach to efficient long-context modeling is to take advantage of the inherent sparsity of softmax attention [Ge24, Jia23z], where selectively computing critical query-key pairs can significantly reduce computational overhead while preserving performance. Recent advances demonstrate this potential through diverse strategies: KV-cache eviction methods [Zha23g, Li24c, Zho24z], blockwise KV-cache selection methods [Tan24, Xia24z, Gao24z], and sampling, clustering or hashing-based selection methods [Che25ab, Liu24zz, Des24z]. Despite these promising strategies, existing sparse attention methods often fall short in practical deployments. Many approaches fail to achieve speedups comparable to their theoretical gains; moreover, most methods lack effective training-time support to fully exploit the sparsity patterns of attention.

To address these limitations, the deployment of effective sparse attention must tackle two key challenges: (1) ***Hardware-aligned inference speedup***: Converting theoretical computation reductions into actual speed improvements requires hardware-friendly algorithm design during both prefilling and decoding stages to mitigate memory access and hardware scheduling bottlenecks; (2) ***Training-aware algorithm design***: Enabling end-to-end computation with trainable operators to reduce training costs while maintaining model performance. These requirements are crucial for real-world applications to achieve fast long-context inference or training. When considering both aspects, existing methods still exhibit a noticeable gap.

To achieve more effective and efficient sparse attention, we present NSA, a Natively trainable Sparse Attention architecture that integrates hierarchical token modeling. As shown in [Figure 2](#figure-02), NSA reduces per-query computation by organizing keys and values into temporal blocks and processing them through three attention paths: compressed coarse-grained tokens, selectively retained fine-grained tokens, and sliding windows for local contextual information. Then we implement specialized kernels to maximize its practical efficiency. NSA introduces two core innovations corresponding to the key requirements above: (1) Hardware-aligned system: Optimize blockwise sparse attention for Tensor Core utilization and memory access, ensuring balanced arithmetic intensity. (2) Training-aware design: Enable stable end-to-end training through efficient algorithms and backward operators. This optimization enables NSA to support both efficient deployment and end-to-end training.

We evaluate NSA through comprehensive experiments on real-world language corpora. Pretraining on a 27B-parameter transformer backbone with 260B tokens, we assess NSA’s performance across general language evaluations, long-context evaluations, and chain-of-thought reasoning evaluation. We further compare the kernel speed on A100 GPUs with optimized Triton [Til19] implementations. Experimental results demonstrate that NSA achieves comparable or superior performance to full attention baseline, while outperforming existing sparse attention approaches. Additionally, NSA delivers substantial speedups across decoding, forward, and backward stages compared to Full Attention, with the speedup ratio increasing for longer sequences. These results validate that our hierarchical sparse attention design effectively balances model capability and computational efficiency.

<span id="section-2"></span>

## 2 Rethinking Sparse Attention Methods
Modern sparse attention methods have made significant strides in reducing the theoretical computational complexity of transformer models. However, most approaches predominantly apply sparsity during inference while retaining a pretrained Full Attention backbone, potentially introducing architectural bias that limits their ability to fully exploit sparse attention’s advantages. Before introducing our native sparse architecture, we systematically analyze these limitations through two critical lenses.

<span id="figure-02"></span>

![Overview of NSA's architecture. Left: The framework processes input sequences through three parallel attention branches: For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context. Right: Visualization of different attention patterns produced by each branch. Green areas indicate regions where attention scores need to be computed, while white areas represent regions that can be skipped.](../../papers/native-sparse-attention/figure-02.png)

**Figure 2. Overview of NSA's architecture. Left: The framework processes input sequences through three parallel attention branches: For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context. Right: Visualization of different attention patterns produced by each branch. Green areas indicate regions where attention scores need to be computed, while white areas represent regions that can be skipped.**

<span id="section-2-1"></span>

### 2.1 The Illusion of Efficient Inference
Despite achieving sparsity in attention computation, many methods fail to achieve corresponding reductions in inference latency, primarily due to two challenges:

**Phase-Restricted Sparsity.** Methods such as H2O [Zha23g] apply sparsity during autoregressive decoding while requiring computationally intensive pre-processing (e.g. attention map calculation, index building) during prefilling. In contrast, approaches like MInference [Jia24e] focus solely on prefilling sparsity. These methods fail to achieve acceleration across all inference stages, as at least one phase remains computational costs comparable to Full Attention. The phase specialization reduces the speedup ability of these methods in prefilling-dominated workloads like book summarization and code completion, or decoding-dominated workloads like long chain-of-thought [Wei22z] reasoning.

**Incompatibility with Advanced Attention Architecture.** Some sparse attention methods fail to adapt to modern decoding efficient architectures like Mulitiple-Query Attention (MQA) [Sha19] and Grouped-Query Attention (GQA) [Ain23], which significantly reduced the memory access bottleneck during decoding by sharing KV across multiple query heads. For instance, in approaches like Quest [Tan24], each attention head independently selects its KV-cache subset. Although it demonstrates consistent computation sparsity and memory access sparsity in Multi-Head Attention (MHA) models, it presents a different scenario in models based on architectures like GQA, where the memory access volume of KV-cache corresponds to the union of selections from all query heads within the same GQA group. This architectural characteristic means that while these methods can reduce computation operations, the required KV-cache memory access remains relatively high. This limitation forces a critical choice: while some sparse attention methods reduce computation, their scattered memory access pattern conflicts with efficient memory access design from advanced architectures.

These limitations arise because many existing sparse attention methods focus on KV-cache reduction or theoretical computation reduction, but struggle to achieve significant latency reduction in advanced frameworks or backends. This motivates us to develop algorithms that combine both advanced architectural and hardware-efficient implementation to fully leverage sparsity for improving model efficiency.

<span id="section-2-2"></span>

### 2.2 The Myth of Trainable Sparsity
Our pursuit of native trainable sparse attention is motivated by two key insights from analyzing inference-only approaches: (1) ***Performance Degradation***: Applying sparsity post-hoc forces models to deviate from their pretrained optimization trajectory. As demonstrated by Chen et al. [Che25ab], top 20% attention can only cover 70% of the total attention scores, rendering structures like retrieval heads in pretrained models vulnerable to pruning during inference. (2) ***Training Efficiency Demands***: Efficient handling of long-sequence training is crucial for modern LLM development. This includes both pretraining on longer documents to enhance model capacity, and subsequent adaptation phases such as long-context fine-tuning and reinforcement learning. However, existing sparse attention methods primarily target inference, leaving the computational challenges in training largely unaddressed. This limitation hinders the development of more capable long-context models through efficient training. Additionally, efforts to adapt existing sparse attention for training also expose challenges:

**Non-Trainable Components.** Discrete operations in methods like ClusterKV [Liu24zz] (includes k-means clustering) and MagicPIG [Che25ab] (includes SimHash-based selecting) create discontinuities in the computational graph. These non-trainable components prevent gradient flow through the token selection process, limiting the model’s ability to learn optimal sparse patterns.

**Inefficient Back-propagation.** Some theoretically trainable sparse attention methods suffer from practical training inefficiencies. Token-granular selection strategy used in approaches like HashAttention [Des24z] leads to the need to load a large number of individual tokens from the KV cache during attention computation. This non-contiguous memory access prevents efficient adaptation of fast attention techniques like FlashAttention, which rely on contiguous memory access and blockwise computation to achieve high throughput. As a result, implementations are forced to fall back to low hardware utilization, significantly degrading training efficiency.

<span id="section-2-3"></span>

### 2.3 Native Sparsity as an Imperative
These limitations in inference efficiency and training viability motivate our fundamental redesign of sparse attention mechanisms. We propose NSA, a natively sparse attention framework that addresses both computational efficiency and training requirements. In the following sections, we detail the algorithmic design and operator implementation of NSA.

<span id="section-3"></span>

## 3 Methodology
Our technical approach spans algorithm design and kernel optimization. In the following subsections, we first introduce the background of our methodology. Then we present the overall framework of NSA, followed by its key algorithmic components. Finally, we detail our hardware-optimized kernel design that maximizes practical efficiency.

<span id="section-3-1"></span>

### 3.1 Background
**Attention Mechanism** is widely used in language modeling where each query token $\mathbf{q}_t$ computes relevance scores against all preceding keys $\mathbf{k}_{:t}$ to generate a weighted sum of values $\mathbf{v}_{:t}$. Formally, for an input sequence of length $t$, the attention operation is defined as:
<span id="equation-01"></span>

$$
\mathbf{o}_t = \mathrm{Attn}\left(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}\right)
$$
where $\mathrm{Attn}$ denotes the attention function:
<span id="equation-02"></span>

$$
\mathrm{Attn}\left(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}\right) = \sum_{i=1}^t\frac{ \alpha_{t,i} \mathbf{v}_i}{\sum_{j=1}^t \alpha_{t,j}}, \quad \alpha_{t,i} = e^{\frac{\mathbf{q}_t^\top \mathbf{k}_i}{\sqrt{d_k}}}\,.
$$
Here, $\alpha_{t,i}$ represents the attention weight between $\mathbf{q}_t$ and $\mathbf{k}_i$, and $d_k$ is the feature dimension of keys. As sequence length increases, attention computation becomes increasingly dominant in the overall computational cost, presenting significant challenges for long-context processing.

**Arithmetic Intensity** is the ratio of compute operations to memory accesses. It intrinsically shapes algorithm optimization on hardware. Each GPU has a critical arithmetic intensity determined by its peak compute capability and memory bandwidth, calculated as the ratio of these two hardware limits. For computation tasks, arithmetic intensity above this critical threshold becomes compute-bound (limited by GPU FLOPS), while below it becomes memory-bound (limited by memory bandwidth).

Specifically for causal self-attention mechanism, during training and prefilling phases, batched matrix multiplications and attention computations exhibit high arithmetic intensity, making these stages compute-bound on modern accelerators. In contrast, auto-regressive decoding becomes memory-bandwidth constrained because it generates one token per forward pass while requiring loading the entire key-value cache, resulting in low arithmetic intensity. This leads to different optimization goals — reducing computation cost during training and prefilling, while reducing memory access during decoding.

<span id="section-3-2"></span>

### 3.2 Overall Framework
To leverage the potential of attention with natural sparse pattern, we propose replacing the original key-value pairs $\mathbf{k}_{:t}, \mathbf{v}_{:t}$ in [Equation 1](#equation-01) with a more compact and information-dense set of representation key-value pairs $\tilde{K}_t, \tilde{V}_t$ given each query $\mathbf{q}_t$. Specifically, we formally define the optimized attention output as follows:
<span id="equation-03"></span>

$$
\tilde{K}_t = f_K(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t}), \quad \tilde{V}_t = f_V(\mathbf{q}_t, \mathbf{k}_{:t}, \mathbf{v}_{:t})
$$
<span id="equation-04"></span>

$$
\mathbf{o}^*_t=\mathrm{Attn}\left(\mathbf{q}_t,\tilde{K}_t, \tilde{V}_t \right)
$$

where $\tilde{K}_t, \tilde{V}_t$ are dynamically constructed based on the current query $\mathbf{q}_t$ and the contextual memory $\mathbf{k}_{:t}, \mathbf{v}_{:t}$. We can design various mapping strategies to get different categories of $\tilde{K}_t^c, \tilde{V}_t^c$, and combine them as follows:

<span id="equation-05"></span>

$$
\mathbf{o}^*_t = \sum_{c \in \mathcal{C}} g_t^c \cdot \mathrm{Attn}(\mathbf{q}_t, \tilde{K}_t^c, \tilde{V}_t^c).
$$
As illustrated in [Figure 2](#figure-02), NSA have three mapping strategies $\mathcal{C}=\{\mathrm{cmp},\mathrm{slc},\mathrm{win}\}$, representing compression, selection, and sliding window for keys and values. $g_t^c\in [0, 1]$ is the gate score for corresponding strategy $c$, derived from input features via an MLP and sigmoid activation. Let $N_t$ denote the total number of remapped keys/values:
<span id="equation-06"></span>

$$
N_t = \sum_{c \in \mathcal{C}}\mathrm{size}[\tilde{K}^c_t].
$$
We maintain a high sparsity ratio by ensuring${N_t}{} \ll t$.

<span id="section-3-3"></span>

### 3.3 Algorithm Design
In this subsection, we introduce the design of our remapping strategies $f_K$ and $f_V$: token compression, token selection, and sliding window.

<span id="section-3-3-1"></span>

#### 3.3.1 Token Compression
By aggregating sequential blocks of keys or values into block-level representations, we obtain compressed keys and values that capture the information of the entire block. Formally, the compressed key representation is defined as:
<span id="equation-07"></span>

$$
\tilde{K}^\mathrm{cmp}_t = f_K^\mathrm{cmp}(\mathbf{k}_{:t}) = \left\{\varphi(\mathbf{k}_{i d+1: i d+l})\middle| 0\leqslant i\leqslant\left\lfloor\frac{t-l}{d}\right\rfloor\right\}
$$
where $l$ is the block length, $d$ is the sliding stride between adjacent blocks, and $\varphi$ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key. $\tilde{K}_t^\mathrm{cmp}\in \mathbb{R}^{ d_k \times \left\lfloor\frac{t-l}{d}\right\rfloor }$ is tensor composed by compresion keys. Usually, we adopt $d < l$ to mitigate information fragmentation. An analogous formulation holds for the compressed value representation $\tilde{V}_t^\mathrm{cmp}$. Compressed representations capture coarser-grained higher-level semantic information and reduce computational burden of attention.

<span id="section-3-3-2"></span>

#### 3.3.2 Token Selection
Using only compressed keys, values might lose important fine-grained information, motivating us to selectively preserve individual keys, values. Below we describe our efficient token selection mechanism that identifies and preserves the most relevant tokens with low computational overhead.

**Blockwise Selection.** Our selection strategy processes key and value sequences in spacial continuous blocks, motivated by two key factors: hardware efficiency considerations and inherent distribution patterns of attention scores. ***Blockwise selection is crucial to achieve efficient computation on modern GPUs.*** That is because modern GPU architectures exhibit significantly higher throughput for continuous block accesses compared to random index-based reads. Also, blockwise computation enables optimal utilization of Tensor Cores. This architectural characteristic has established blockwise memory access and computation as a fundamental principle in high-performance attention implementations, as exemplified by FlashAttention’s block-based design. ***Blockwise selection follows the inherent distribution patterns of attention scores.*** Prior works [Jia24e] have shown that attention scores often exhibit spatial continuity, suggesting that neighboring keys tend to share similar importance levels. Our visualization in [Section 6.2](#section-6-2) also shows this spatial continuous pattern.

To implement blockwise selection, we first divide key, value sequences into selection blocks. To identify the most important blocks for attention computation, we need to assign importance scores to each block. Below we present our method for computing these block-level importance scores.

**Importance Score Computation.** Computing block importance scores could introduce significant overhead. Fortunately, the attention computation of compression tokens produces intermediate attention scores that we can leverage to induce selection block importance scores, formulated as:
<span id="equation-08"></span>

$$
\mathbf{p}_t^\mathrm{cmp} = \mathrm{Softmax}\left(\mathbf{q}_t^\top \tilde{K}_t^\mathrm{cmp}\right),
$$
where $\mathbf{p}_t^\mathrm{cmp} \in \mathbb{R}^{\left\lfloor\frac{t-l}{d}\right\rfloor+1}$ is the attention scores between $q_t$ and compression keys $\tilde{K}_t^\mathrm{cmp}$. Let $l'$ denote the selection block size. When compression blocks and selection blocks share the same blocking scheme, i.e., $l'=l=d$, we can directly obtain the selection block importance scores $\mathbf{p}_t^\mathrm{slc}$ by $\mathbf{p}_t^\mathrm{slc} = \mathbf{p}_t^\mathrm{cmp}$ straightforwardly. For cases where the blocking schemes differ, we derive the importance scores for selection blocks according to their spatial relationship. Given $l\leqslant l',$ $d \mid l$ and $d \mid l'$, we have:
<span id="equation-09"></span>

$$
\mathbf{p}_t^\mathrm{slc}[j] = \sum_{m=0}^{\frac{l'}{d}-1}\sum_{n=0}^{\frac{l}{d} -1} \mathbf{p}_t^\mathrm{cmp}\left[\frac{l'}{d}j -m -n \right],
$$
where$[\cdot]$ denotes the indexing operator for accessing vector element. For models employing GQA or MQA where key-value caches are shared across query heads, consistent block selection across these heads has to be ensured to minimize KV cache loading during decoding. The shared importance scores across heads in a group are formally defined as:

<span id="equation-10"></span>

$$
{\mathbf{p}_t^{\mathrm{slc}}}' = \sum_{h=1}^{H} \mathbf{p}_{t}^{\mathrm{slc}, (h)},
$$
where $(h)$ in the superscript denotes the head index, and $H$ is the number of query heads in each group. This aggregation ensures consistent block selection across heads within the same group.

**Top-$\pmb{n}$ Block Selection.** After obtaining the selection block importance scores, We retain tokens within the top-$n$ sparse blocks ranked by block importance scores, formulated as:
<span id="equation-11"></span>

$$
\mathcal{I}_t = \{i \mid \mathrm{rank}({\mathbf{p}_t^\mathrm{slc}}'[i]) \leqslant n\}
$$
<span id="equation-12"></span>

$$
\tilde{K}^\mathrm{slc}_t =  \mathrm{Cat}\left[\{\mathbf{k}_{il'+1:(i+1)l'}|i \in \mathcal{I}_t\}\right],
$$
where rank$(\cdot)$ denotes the ranking position in descending order, with rank = 1 corresponding to the highest score, $\mathcal{I}_t$ is the set of selected blocks’ indices, $\mathrm{Cat}$ denotes the concatenation operation. $\tilde{K}_t^\mathrm{slc}\in \mathbb{R}^{ d_k \times nl' }$ is tensor composed by compresion keys. An analogous formulation applies to the fine-grained value $\tilde{V}^\mathrm{slc}_t$. The selected keys and values then participate in the attention computation with $\mathbf{q}_t$ as defined in [Equation 5](#equation-05).

<span id="section-3-3-3"></span>

#### 3.3.3 Sliding Window
In attention mechanisms, local patterns typically adapt faster and can dominate the learning process, potentially preventing the model from effectively learning from compression and selection tokens. To address this issue, we introduce a dedicated sliding window branch that explicitly handles local context, allowing other branches (compression and selection) to focus on learning their respective features without being shortcutted by local patterns. Specifically, we maintain recent tokens $\tilde{K}_t^\mathrm{win}=\mathbf{k}_{t-w:t}, \tilde{V}_t^\mathrm{win}=\mathbf{v}_{t-w:t}$ in a window $w$, and isolate attention computations of different information sources (compression tokens, and selected tokens, sliding window) into separate branches. These branch outputs are then aggregated through a learned gating mechanism. To further prevent shortcut learning across attention branches with marginal computational overhead, we provide independent keys and values for three branches. This architectural design enables stable learning by preventing gradient interference between local and long-range pattern recognition, while introducing minimal overhead.

After obtaining all three categories of keys and values ($\tilde{K}_t^\mathrm{cmp}, \tilde{V}_t^\mathrm{cmp}$; $\tilde{K}_t^\mathrm{slc}, \tilde{V}_t^\mathrm{slc}$; and $\tilde{K}_t^\mathrm{win}, \tilde{V}_t^\mathrm{win}$), we compute the final attention output following [Equation 5](#equation-05). Together with the compression, selection, and sliding window mechanisms described above, this forms the complete algorithmic framework of NSA.

<span id="section-3-4"></span>

### 3.4 Kernel Design
To achieve FlashAttention-level speedup during the training and prefilling, we implement hardware-aligned sparse attention kernels upon Triton. Given MHA is memory-intensive and inefficient for decoding, we focus on architectures with shared KV caches like GQA and MQA following the current state-of-the-art LLMs. While compression and sliding window attention computations are readily compatible with existing FlashAttention-2 kernels, we introduce the specialized kernel design for sparse selection attention. If we were to follow FlashAttention’s strategy of loading temporally continuous query blocks into SRAM, it would result in inefficient memory access since queries within a block may require disjoint KV blocks. To address this, our key optimization lies in a different query grouping strategy: for each position on the query sequence, we load all query heads within a GQA group (they share the same sparse KV blocks) into SRAM. [Figure 3](#figure-03) illustrates our forward pass implementation. The proposed kernel architecture is characterized by the following key features:

1.  **Group-Centric Data Loading**. For each inner loop, load all heads’ queries $Q\in \mathbb{R}^{[ h, d_k]}$ in the group at position $t$ and their shared sparse key/value block indices $\mathcal{I}_t$.

2.  **Shared KV Fetching**. In the inner loop, Sequentially load continuous key/value blocks indexed by $\mathcal{I}_t$ into SRAM as $K\in\mathbb{R}^{[B_k, d_k]}, V\in\mathbb{R}^{[B_k, d_v]}$ to minimize memory loading, where $B_k$ is the kernel block size satisfying $B_k | l'$.

3.  **Outer Loop on Grid**. Since the inner-loop length (proportional to the selected block count $n$) remains nearly identical for different query blocks, we put query/output loops in Triton’s grid scheduler to simplify and optimize the kernel.

This design achieves near-optimal arithmetic intensity by (1) eliminating redundant KV transfers through group-wise sharing, and (2) balancing compute workloads across GPU streaming multiprocessors.

<span id="figure-03"></span>

![Kernel design for NSA. The kernel loads queries by GQA groups (Grid Loop), fetches corresponding sparse KV blocks (Inner Loop), and performs attention computation on SRAM. Green blocks indicate data on SRAM, while blue indicates data on HBM.](../../papers/native-sparse-attention/figure-03.png)

**Figure 3. Kernel design for NSA. The kernel loads queries by GQA groups (Grid Loop), fetches corresponding sparse KV blocks (Inner Loop), and performs attention computation on SRAM. Green blocks indicate data on SRAM, while blue indicates data on HBM.**

<span id="section-4"></span>

## 4 Experiments
We evaluate NSA through three lenses: (1) general benchmarks performance, (2) long-context benchmarks performance, and (3) chain-of-thought reasoning performance, comparing against Full Attention baseline and state-of-the-art sparse attention methods. We defer the efficiency analysis of our sparse computation paradigm to [Section 5](#section-5), where we provide detailed discussions on training and inference speed.

<span id="section-4-1"></span>

### 4.1 Pretraining Setup
Following the common practice in state-of-the-art LLMs, our experiments adopt a backbone combining Grouped-Query Attention (GQA) and Mixture-of-Experts (MoE), featuring $27\text{B}$ total parameters with $3\text{B}$ active parameters. The model consists of 30 layers with a hidden dimension of 2560. For GQA, we set the number of groups to 4, with a total of 64 attention heads. For each head, the hidden dimensions of the query, key, and value are configured as $d_q = d_k = 192$ and $d_v = 128$, respectively. For MoE, we utilize the DeepSeekMoE [Dai24, Dee24] structure, with 72 routed experts and 2 shared experts, and set the top-k experts to 6. To ensure training stability, the MoE in the first layer is replaced by an MLP in the form of SwiGLU. The proposed architecture achieves an effective trade-off between computation cost and model performance. For NSA, we set compression block size $l=32$, sliding stride $d=16$, selected block size $l'=64$, selected block count $n=16$ (including fixed activating the 1 initial block and 2 local blocks), and sliding window size $w=512$. Both Full Attention and sparse attention models are pretrained on $270\text{B}$ tokens of $8\text{k}$-length texts, followed by continued training and supervised fine-tuning on $32\text{k}$-length texts with YaRN [Pen23] to achieve long-context adaptation. Both models are trained to full convergence to ensure fair comparison. As shown in  [Figure 4](#figure-04), the pretraining loss curve of our NSA and Full Attention baseline demonstrates stable and smooth decline, with NSA consistently outperforming the Full Attention model.

<span id="figure-04"></span>

![Pretraining loss comparison between Full Attention and our NSA on 27B-parameter model. Both models exhibit stable convergence, with NSA achieving lower loss values.](../../papers/native-sparse-attention/figure-04.png)

**Figure 4. Pretraining loss comparison between Full Attention and our NSA on 27B-parameter model. Both models exhibit stable convergence, with NSA achieving lower loss values.**

<span id="table-01"></span>

![Pretraining performance comparison between the full attention baseline and NSA on general benchmarks, across knowledge (MMLU, MMLU-PRO, CMMLU), reasoning (BBH, GSM8K, MATH, DROP), and coding (MBPP, HumanEval) tasks. NSA achieves superior average performance on most benchmarks despite high sparsity.](../../papers/native-sparse-attention/table-01.png)

**Table 1. Pretraining performance comparison between the full attention baseline and NSA on general benchmarks, across knowledge (MMLU, MMLU-PRO, CMMLU), reasoning (BBH, GSM8K, MATH, DROP), and coding (MBPP, HumanEval) tasks. NSA achieves superior average performance on most benchmarks despite high sparsity.**

<span id="section-4-2"></span>

### 4.2 Baselines Methods
In addition to comparing with Full Attention, we evaluate several state-of-the-art inference-stage sparse attention methods: H2O [Zha23g], infLLM [Xia24z], Quest [Tan24], and Exact-Top, which first computes full attention score and select the top-$n$ scores keys corresponding to each query and then calculates attention on these positions. These methods span diverse sparse attention paradigms, including KV-cache eviction, query-aware selection, and exact top-$n$ sparse selection.

For general evaluation, where most samples have lengths within the local context window of sparse attention baselines, these methods are effectively equivalent to Full Attention. Therefore, we present only the comparison results between NSA and Full Attention baseline in this setting. In the long-context evaluation, we conduct comparisons across all baseline methods, with the sparsity of all sparse attention methods set to the same to ensure a fair comparison. For chain-of-thought reasoning evaluation, which requires long-text supervised fine-tuning, we limit our comparison to Full Attention, as sparse attention baselines do not support training.

<span id="section-4-3"></span>

### 4.3 Performance Comparison
<span id="table-02"></span>

![Performance comparison between our NSA and baselines on LongBench, including subsets in single document QA, multi-document QA, synthetic and code task categories. NSA outperformed most of the baselines including Full Attention.](../../papers/native-sparse-attention/table-02.png)

**Table 2. Performance comparison between our NSA and baselines on LongBench, including subsets in single document QA, multi-document QA, synthetic and code task categories. NSA outperformed most of the baselines including Full Attention.**

**General Evaluation.** We evaluated the pretrained NSA and Full Attention baseline, on a comprehensive suite of benchmarks spanning knowledge, reasoning, and coding capabilities, including MMLU [Hen20], MMLU-PRO [Wan24c], CMMLU [Li23e], BBH [Suz22], GSM8K [Cob21], MATH [Hen20], DROP [Dua19], MBPP [Aus21], and HumanEval [Che21]. The results are shown in [Table 1](#table-01). Despite its sparsity, NSA achieves superior overall performance, outperforming all baselines including Full Attention on 7 out of 9 metrics. This indicates that although NSA may not fully leverage its efficiency advantages on shorter sequences, it shows strong performance. Notably, NSA demonstrates significant gains in reasoning-related benchmarks (DROP: +0.042, GSM8K: +0.034), suggesting that our pretraining helps models to develop specialized attention mechanisms. This sparse attention pretraining mechanism forces model to focus on the most important information, potentially enhancing performance by filtering out noise from irrelevant attention pathways. The consistent performance across diverse evaluations also validates NSA’s robustness as a general-purpose architecture.

**Long-Context Evaluation.** As shown in [Figure 5](#figure-05), NSA achieves perfect retrieval accuracy across all positions in 64k-context needle-in-a-haystack [Kam23z] test. This performance stems from our hierarchical sparse attention design, which combines compression tokens for efficient global context scanning, and selection tokens for precise local information retrieval. The coarse-grained compression identifies relevant context blocks at low computational cost, while the token-level attention on selected tokens ensures the preservation of critical fine-grained information. This design enables NSA to maintain both global awareness and local precision.

<span id="figure-05"></span>

![Needle-in-a-Haystack retrieval accuracy across context positions with 64k context length. NSA achieves perfect accuracy through its hierarchical sparse attention design.](../../papers/native-sparse-attention/figure-05.png)

**Figure 5. Needle-in-a-Haystack retrieval accuracy across context positions with 64k context length. NSA achieves perfect accuracy through its hierarchical sparse attention design.**

We also evaluate NSA on LongBench [Bai23] against state-of-the-art sparse attention methods and Full Attention baseline. To ensure consistent sparsity, we set the token activated by each query in all sparse attention baselines to 2560 tokens, which corresponds to the average number of tokens activated in NSA when handling 32k sequence lengths. Following StreamLLM [Xia24a], this token budget includes the leading 128 tokens and 512 local tokens. We exclude certain subsets from LongBench due to their low scores across all models, which may not provide meaningful comparisons. As shown in [Table 2](#table-02), NSA achieves the highest average score 0.469, outperforming all baselines (+0.032 over Full Attention and +0.046 over Exact-Top). This improvement arises from two key innovations: (1) our native sparse attention design, which enables end-to-end optimization of sparse patterns during pretraining, facilitates synchronized adaptation between the sparse attention module and other model components; and (2) the hierarchical sparse attention mechanism achieves a balance between local and global information processing.

Notably, NSA demonstrates exceptional performance on tasks requiring complex reasoning over long contexts, achieving +0.087 and +0.051 improvements over Full Attention on multi-hop QA tasks (HPQ and 2Wiki), exceeding the performance of baselines on code understanding (LCC: +0.069), and outperforming other methods on passage retrieval (PassR-en: +0.075). These results validate NSA’s capability to handle diverse long-context challenges, with its natively pretrained sparse attention providing additional benefits in learning task-optimal patterns.

**Chain-of-Thought Reasoning Evaluation.** To evaluate NSA’s compatibility with advanced downstream training paradigms, we investigate its capacity to acquire chain-of-thought mathematical reasoning abilities via post-training. Given the limited effectiveness of reinforcement learning on smaller-scale models, we employ knowledge distillation from DeepSeek-R1, conducting supervised fine-tuning (SFT) with 10B tokens of 32k-length mathematical reasoning traces. This produces two comparable models: Full Attention-R (Full Attention baseline) and NSA-R (our sparse variant). We assess both models on the challenging American Invitational Mathematics Examination (AIME 24) benchmark. We use a sampling temperature of 0.7 and a top-$p$ value of 0.95 to generate 16 responses for each question and obtain the average score. To validate the impact of reasoning depth, we conduct experiments with two generation context limits: 8k and 16k tokens, measuring whether extended reasoning chains improve accuracy. Example comparisons of model predictions are provided in [Section 9](#section-9).

<span id="table-03"></span>

![AIME Instruction-based Evaluating after supervised fine-tuning. Our NSA-R demonstrates better performance than Full Attention-R at both 8k and 16k sequence lengths](../../papers/native-sparse-attention/table-03.png)

**Table 3. AIME Instruction-based Evaluating after supervised fine-tuning. Our NSA-R demonstrates better performance than Full Attention-R at both 8k and 16k sequence lengths**

As shown in [Table 3](#table-03), NSA-R achieves significantly higher accuracy than Full Attention-R under the 8k context setting (+0.075), with this advantage persisting at 16k contexts (+0.054). These results validate two key benefits of native sparse attention: (1) The pretrained sparse attention patterns enable efficient capture of long-range logical dependencies critical for complex mathematical derivations; (2) Our architecture’s hardware-aligned design maintains sufficient context density to support growing reasoning depth without catastrophic forgetting. The consistent outperformance across context lengths confirms sparse attention’s viability for advanced reasoning tasks when natively integrated into the training pipeline.

<span id="section-5"></span>

## 5 Efficiency Analysis
We evaluate the computational efficiency of NSA against Full Attention on an 8-GPU A100 system. In efficiency analysis, we also configure the model with GQA group $g=4$, heads per group $h=16$, query/key dimension $d_k=192$, and value dimension $d_v=128$. Following the same settings in [Section 4](#section-4), we set NSA compression block size $l=32$, sliding stride $d=16$, selected block size $l'=64$, selected block count $n=16$, and sliding window size $w=512$.

<span id="figure-06"></span>

![Comparison of Triton-based NSA kernel with Triton-based FlashAttention-2 kernel. Our implementation significantly reduces latency across all context lengths, with the improvement becoming more pronounced as input length increases.](../../papers/native-sparse-attention/figure-06.png)

**Figure 6. Comparison of Triton-based NSA kernel with Triton-based FlashAttention-2 kernel. Our implementation significantly reduces latency across all context lengths, with the improvement becoming more pronounced as input length increases.**

<span id="section-5-1"></span>

### 5.1 Training Speed
We compare the Triton-based implementations of our NSA attention and Full Attention with Triton-based FlashAttention-2 to ensure fair speed comparison across the same backend. As shown in [Figure 6](#figure-06), our NSA achieves progressively greater speedups as context length increases, up to 9.0$\times$ forward and 6.0$\times$ backward speedup at 64k context-length. Notably, the speed advantage becomes more pronounced with longer sequences. This speedup stems from our hardware-aligned algorithm design to maximize the efficiency of sparse attention architecture: (1) The Blockwise memory access pattern maximizes Tensor Core utilization through coalesced loads, (2) The delicate loop scheduling in the kernel eliminates redundant KV transfers.

<span id="table-04"></span>

![Memory access volume (in equivalent number of tokens) per attention operation during decoding. Due to the low arithmetic intensity and memory-bound nature of decoding, the expected speedup is approximately linear with the volume of memory access.](../../papers/native-sparse-attention/table-04.png)

**Table 4. Memory access volume (in equivalent number of tokens) per attention operation during decoding. Due to the low arithmetic intensity and memory-bound nature of decoding, the expected speedup is approximately linear with the volume of memory access.**

<span id="section-5-2"></span>

### 5.2 Decoding Speed
The decoding speed of Attention is primarily determined by the memory access bottleneck, which is closely tied to the amount of KV cache loading. In each decoding step, Our NSA just needs to load at most $\left\lfloor\frac{s-l}{d}\right\rfloor$ compression tokens, $nl'$ selected tokens, and $w$ neighbor tokens, where $s$ is the cached sequence length. As shown in [Table 4](#table-04), our method exhibits a significant reduction in latency as the decoding length increases, achieving up to 11.6$\times$ speedup at 64k context-length. This advantage in memory access efficiency also amplifies with longer sequences.

<span id="section-6"></span>

## 6 Discussion
In this section, we reflect on the development process of NSA and discuss key insights gained from our exploration of different sparse attention strategies. While our approach demonstrates promising results, understanding the challenges encountered with alternative strategies and analyzing attention patterns provides valuable context for future research directions. We first examine challenges with alternative token selection strategies that motivated our design choices, followed by visualizations that offer insights into attention distribution patterns.

<span id="section-6-1"></span>

### 6.1 Challenges with Alternative Token Selection Strategies
Before designing NSA, we explored adapting existing sparse attention methods to the training stage. However, these attempts encountered various challenges, prompting us to design a different sparse attention architecture:

**Key-Clustering Based Strategies.** We examined clustering-based strategies like ClusterKV [Liu24zz]. These methods store Keys and Values from the same cluster in contiguous memory regions. While theoretically feasible for training and inference, they face three significant challenges: (1) Non-trivial computational overhead introduced by dynamic clustering mechanisms; (2) Operator optimization difficulties exacerbated by inter-cluster imbalances, especially in Mixture-of-Experts (MoE) systems, where skewed Expert Parallelism (EP) group execution times lead to persistent load imbalances; (3) Implementation constraints arising from the need for mandatory periodic reclustering and chunk-sequential training protocols. These combined factors create substantial bottlenecks, significantly limiting their effectiveness for real-world deployment.

<span id="figure-07"></span>

![Compare training loss on a 3B-parameter model with Full Attention and different token selection strategies and. Our NSA achieves better performance.](../../papers/native-sparse-attention/figure-07.png)

**Figure 7. Compare training loss on a 3B-parameter model with Full Attention and different token selection strategies and. Our NSA achieves better performance.**

<span id="figure-08"></span>

![Visualization of Attention Map on a Full Attention transformer. Light-colored regions indicate higher attention values. As shown in the figure, attention scores exhibit blockwise clustering distribution.](../../papers/native-sparse-attention/figure-08.png)

**Figure 8. Visualization of Attention Map on a Full Attention transformer. Light-colored regions indicate higher attention values. As shown in the figure, attention scores exhibit blockwise clustering distribution.**

**Other Blockwise Selection Strategies.** We also considered blockwise key, value selection strategies different from NSA, such as Quest [Tan24] and InfLLM [Xia24z]. These methods rely on computing an importance score for each KV block and selecting the top-$n$ blocks based on their similarity with $q_t$. However, existing methods face two critical issues: (1) Since the selection operation is non-differentiable, importance score computation based on neural networks relies on auxiliary loss, which increases operator overhead and often degrades model performance; (2) Heuristic parameter-free importance score computation strategy suffer from low recall rates, leading to suboptimal performance. We evaluate both approaches on a 3B-parameter model with similar architecture and compare their loss curve with NSA and Full Attention. For the auxiliary loss-based selection method, we introduce additional queries for each token and representative keys for each block to estimate the block importance scores. We compute block-level supervision signals by mean-pooling attention scores within each key block, and use KL divergence to supervise block importance prediction. We maintain individual query granularity instead of block-averaged queries to accommodate efficient decoding. This auxiliary loss-based importance estimation shares conceptual similarity with SeerAttention [Gao24z]. For the heuristic parameter-free selection method, following the strategy of Quest, we implement direct selection using the product between queries and coordinate-wise min-max of the key chunks, without introducing additional parameters. We also explore a cold-start training approach where Full Attention is applied for the initial 1000 steps before transitioning to the heuristic blockwise selection. As shown in  [Figure 7](#figure-07), both methods exhibited inferior loss.

<span id="section-6-2"></span>

### 6.2 Visualization
To explore potential patterns in transformer attention distributions and seek inspiration for our design, we visualize the attention map from our pretrained 27B Full Attention model in [Figure 8](#figure-08). The visualization reveals interesting patterns where attention scores tend to exhibit blockwise clustering characteristics, with nearby keys often showing similar attention scores. This observation inspired our design of NSA, suggesting that selecting key blocks based on spatial continuity might be a promising approach. The blockwise clustering phenomenon indicates that tokens adjacent in the sequence may share certain semantic relationships with query tokens, though the exact nature of these relationships requires further investigation. This observation motivated us to explore a sparse attention mechanism that operates on continuous token blocks rather than individual tokens, aiming to enhance computational efficiency and preserve high-attention patterns.

<span id="section-7"></span>

## 7 Related Works
We review existing approaches that improve the efficiency of attention computation through sparse attention. These methods can be broadly categorized into three groups based on their core strategies: (1) fixed sparse pattern, (2) dynamic token pruning, and (3) query-aware selection. We introduce several representative works from each category.

<span id="section-7-1"></span>

### 7.1 Fixed Sparse Pattern
SlidingWindow is a commonly used approach that allows the query to compute attention only within a fixed window. StreamingLLM [Xia24a] combines attention sinks with local windows to process continuous text streams. MoA [Fu24z] and DuoAttention [Xia24d] also adopt similar local and sink information for long sequence modeling. Longformer [Bel20] interleaves local windowed attention with global tokens to process long sequences. Compared with them, our NSA does not rely on pre-defined sparse patterns, but learns the patterns automatically, unlocking the potential to utilize full context.

<span id="section-7-2"></span>

### 7.2 Dynamic Token Pruning
Some methods are designed to reduce memory and computation costs during inference through dynamic KV-cache pruning. H2O [Zha23g], BUZZ [Zha24za], and SepLLM [Che24z] implement adaptive approaches to reduce KV-cache memory usage during decoding. These methods dynamically evict tokens deemed less important for future predictions. FastGen  and HeadKV [Fu24za] optimize computation by assigning different strategies to individual attention heads. SnapKV [Li24c] introduces a token pruning strategy that reduces the KV-cache by selectively retaining only the most crucial features, enabling efficient memory usage. Unlike these inference-focused approaches, our NSA incorporates sparsity natively during the training phase.

<span id="section-7-3"></span>

### 7.3 Query-Aware Selection
Other works focus on query-dependent token selection methods to reduce attention computation while preserving attention quality. Quest [Tan24] employs a blockwise selection strategy where each chunk’s importance is estimated by product between query and coordinate-wise min-max of the key chunks. InfLLM [Xia24z] combines fixed patterns with retrieval by maintaining attention sinks, local context, and retrievable chunks. This method selects representative keys from each chunk to estimate chunk importance. HashAttention [Des24z] formulates pivotal token identification as a recommendation problem by mapping queries and keys to Hamming space using learned functions. ClusterKV [Liu24zz] achieves sparsity by firstly clustering keys and then selecting the most relevant clusters for attention computation based on query-cluster similarity. MInference [Jia24e] and TokenSelect [Wu24z] select KV pairs for computation based on token-level importance scoring. SeerAttention [Gao24z] separates queries and keys into spatial blocks and performs blockwise selection for efficient computation. Compared to these methods, our NSA achieves hardware-aligned sparse attention computation throughout the full model lifecycle, including training, prefilling, and decoding.

<span id="section-8"></span>

## 8 Conclusion
We present NSA, a hardware-aligned sparse attention architecture for efficient long-context modeling. By integrating hierarchical token compression with blockwise token selection within a trainable architecture, our architecture achieves accelerated training and inference while maintaining Full Attention performance. NSA advances the state-of-the-art by demonstrating general benchmark performance matches full-attention baselines, exceeding modeling capability in long-context evaluations, and enhanced reasoning ability, all accompanied by measurable reductions in computational latency and achieving significant speedup.

<span id="section-9"></span>

## 9 Example of AIME Results

**Example 1 — Prompt**

> “|User| Let $x,y$ and $z$ be positive real numbers that satisfy the following system of equations:
>
> $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$
>
> $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$
>
> $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$
>
> Then the value of $\left|\log_2(x^4y^3z^2)\right|$ is $\frac{m}{n}$ where $m$ and $n$ are relatively prime positive integers. Find $m+n$.
>
> Please reason step by step, and put your final answer within $\boxed{}$. |Assistant|”

**Our Result**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (2275 tokens)
>
> Given the system of equations:
>
> 1. $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$
> 2. $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$
> 3. $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$
>
> We convert these logarithmic equations to their exponential forms:
>
> 1. $\frac{x}{yz}=\sqrt{2}$
> 2. $\frac{y}{xz}=2^{1/3}$
> 3. $\frac{z}{xy}=2^{1/4}$
>
> Expressing each variable in terms of $x$, $y$, and $z$:
>
> 1. $x=\sqrt{2}\cdot yz$
> 2. $y=2^{1/3}\cdot xz$
> 3. $z=2^{1/4}\cdot xy$
>
> Substituting these expressions into each other, we find:
>
> - From $x=\sqrt{2}\cdot yz$ and $y=2^{1/3}\cdot xz$, we get $x=\sqrt{2}\cdot(2^{1/3}\cdot xz)\cdot z$, leading to $z=2^{-5/12}$.
> - Substituting $z$ into the equation for $y$, we get $y=2^{-3/8}$.
> - Substituting $z$ and $y$ into the equation for $x$, we get $x=2^{-7/24}$.
>
> Next, we compute $x^4y^3z^2$:
>
> - $x^4=(2^{-7/24})^4=2^{-7/6}$
> - $y^3=(2^{-3/8})^3=2^{-9/8}$
> - $z^2=(2^{-5/12})^2=2^{-5/6}$
>
> Multiplying these together: $x^4y^3z^2=2^{-7/6}\cdot2^{-9/8}\cdot2^{-5/6}=2^{-75/24}=2^{-25/8}$.
>
> Taking the logarithm base 2: $\log_2(x^4y^3z^2)=-25/8$.
>
> The absolute value is $|\log_2(x^4y^3z^2)|=25/8$.
>
> Thus, $m=25$ and $n=8$, giving $m+n=33$.
>
> $\boxed{33}$

**Baseline Result**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (9392 tokens)
>
> Given the system of logarithmic equations:
>
> $\log_2\left(\frac{x}{yz}\right)=\frac{1}{2}$, $\log_2\left(\frac{y}{xz}\right)=\frac{1}{3}$, $\log_2\left(\frac{z}{xy}\right)=\frac{1}{4}$.
>
> We convert these equations to exponential form:
>
> $\sqrt{2}=\frac{x}{yz}\implies x=\sqrt{2}\cdot yz$,
>
> $2^{\frac{1}{3}}=\frac{y}{xz}\implies y=2^{\frac{1}{3}}\cdot xz$,
>
> $2^{\frac{1}{4}}=\frac{z}{xy}\implies z=2^{\frac{1}{4}}\cdot xy$.
>
> Substituting these into each other, we find:
>
> 1. $x=\sqrt{2}\cdot yz$
> 2. $y=2^{\frac{1}{3}}\cdot xz$
> 3. $z=2^{\frac{1}{4}}\cdot xy$
>
> By solving these equations step-by-step, we find:
>
> - $z=2^{-5/12}$
> - $xy=2^{-2/3}$
> - $y=2^{-1/24}$
> - $x=2^{1/24}$
>
> However, upon verifying the third equation, we realize the system might not have a solution. Therefore, we use matrix inversion to solve the system:
>
> $\begin{cases}\log_2(x)-\log_2(y)-\log_2(z)=\frac{1}{2}\\ \log_2(y)-\log_2(x)-\log_2(z)=\frac{1}{3}\\ \log_2(z)-\log_2(x)-\log_2(y)=\frac{1}{4}\end{cases}$
>
> Let $A=\log_2(x)$, $B=\log_2(y)$, $C=\log_2(z)$. The system becomes:
>
> $\begin{cases}A-B-C=\frac{1}{2}\\ B-A-C=\frac{1}{3}\\ C-A-B=\frac{1}{4}\end{cases}$
>
> We solve this system using Cramer's rule and matrix inversion. The determinant of the matrix is -2, and the solutions are $A=\frac{61}{48}$, $B=\frac{13}{16}$, $C=-\frac{1}{6}$.
>
> Thus, we have $x=2^{61/48}$, $y=2^{13/16}$, $z=2^{-1/6}$.
>
> We need to find $\left|\log_2(x^4y^3z^2)\right|$:
>
> $4\log_2x+3\log_2y+2\log_2z=4\cdot\frac{61}{48}+3\cdot\frac{13}{16}+2\cdot\left(-\frac{1}{6}\right)$.
>
> Converting to a common denominator (48): $4\cdot\frac{61}{48}=\frac{244}{48}$, $3\cdot\frac{13}{16}=\frac{117}{48}$, $2\cdot\left(-\frac{1}{6}\right)=-\frac{16}{48}$.
>
> Summing these: $\frac{244}{48}+\frac{117}{48}-\frac{16}{48}=\frac{345}{48}=\frac{115}{16}$.
>
> The fraction $\frac{115}{16}$ is in simplest form, so $m=115$ and $n=16$. Therefore, $m+n=115+16=131$.
>
> $\boxed{131}$

**Example 2 — Prompt**

> “|User| Real numbers $x$ and $y$ with $x,y>1$ satisfy $\log_x(y^x)=\log_y(x^{4y})=10.$
>
> What is the value of $xy$?
>
> Please reason step by step, and put your final answer within $\boxed{}$. |Assistant|”

**Our Result**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (15147 tokens)
>
> Given the equations:
>
> 1. $\log_x(y^x)=10$
> 2. $\log_y(x^{4y})=10$
>
> We start by applying the logarithm power rule to both equations.
>
> For the first equation: $\log_x(y^x)=x\cdot\log_x(y)=10$.
>
> Let $z=\log_y(x)$. Then, $\log_x(y)=\frac{1}{z}$. Substituting this into the equation, we get $x\cdot\frac{1}{z}=10\implies x=10z$.
>
> For the second equation: $\log_y(x^{4y})=y\cdot\log_y(x^4)=10$.
>
> Using the power rule, we get $y\cdot4\log_y(x)=10\implies4y\log_y(x)=10$.
>
> Since $\log_y(x)=z$, this simplifies to $4yz=10\implies yz=\frac{5}{2}$.
>
> We now have two expressions:
>
> 1. $x=10z$
> 2. $y=\frac{5/2}{z}$
>
> Substituting $x=10z$ and $y=\frac{5/2}{z}$ into the product $xy$, we get $xy=(10z)\cdot\left(\frac{5/2}{z}\right)=10z\cdot\frac{5}{2z}=10\cdot\frac{5}{2}=25$.
>
> Thus, the value of $xy$ is $\boxed{25}$.

**Baseline Result**

> $\langle$think$\rangle$...$\langle$/think$\rangle$ (16223 tokens)
>
> Given the problem where real numbers $x$ and $y$ with $x,y>1$ satisfy the equations $\log_x(y^x)=10$ and $\log_y(x^4y)=10$, we need to find the value of $xy$.
>
> First, we convert the logarithmic equations to their exponential forms using the change of base formula.
>
> 1. $\log_x(y^x)=10$
>
> $x\log_x(y)=10$.
>
> Using the change of base formula, $\log_x(y)=\frac{\ln y}{\ln x}$. Since $\log_yx=\frac{9}{4}$, we have $\log_xy=\frac{4}{9}$. Thus, $x\cdot\frac{4}{9}=10\implies x=\frac{10\cdot9}{4}=\frac{90}{4}=\frac{45}{2}$.
>
> 2. $\log_y(x^4y)=10$
>
> $4\log_yx+1=10\implies4\log_yx=9\implies\log_yx=\frac{9}{4}$.
>
> Again, using the change of base formula, $\log_xy=\frac{4}{9}$.
>
> Next, we express $x$ and $y$ in terms of each other: $x=\frac{45}{2}$ and $y=\left(\frac{45}{2}\right)^{4/9}$.
>
> We then compute the product $xy$: $xy=\left(\frac{45}{2}\right)\cdot\left(\frac{45}{2}\right)^{4/9}=\left(\frac{45}{2}\right)^{1+4/9}=\left(\frac{45}{2}\right)^{13/9}$.
>
> However, upon re-evaluating and using another approach by setting $x=2^a$ and $y=2^b$, we find:
>
> $\log_x(y^x)=10\implies\frac{x\log_xy}{\log_xy}=10\implies x\cdot\frac{4}{9}=10\implies x=\frac{45}{2}$.
>
> $\log_y(x^4y)=10\implies\frac{4\log_yx+1}{\log_yx}=10\implies4\cdot\frac{9}{4}+1=10\implies9+1=10\quad\mathrm{consistent}$.
>
> Finally, we find that the product $xy$ simplifies to $xy=\left(\frac{45}{2}\right)$.
