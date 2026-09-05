---
title: 'Gated Linear Attention'
createTime: 2026/09/05 22:11:56
permalink: /en/papers/gated-linear-attention/
pageClass: paper-reading
---

> [Songlin Yang](https://sustcsonglin.github.io/) [+author-note], [Bailin Wang](https://berlino.github.io/) [+author-note], [Yikang Shen](https://dblp.org/pid/152/8226), [Rameswar Panda](https://rpand002.github.io/), and [Yoon Kim](https://people.csail.mit.edu/yoonkim/). First submitted to arXiv on December 11, 2023; current version v6. Published in the *Proceedings of the 41st International Conference on Machine Learning*, PMLR 235:56501-56523, July 2024. [Gated Linear Attention Transformers with Hardware-Efficient Training](https://arxiv.org/abs/2312.06635). <a href="/paper/gated-linear-attention.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [ICML 2024](https://proceedings.mlr.press/v235/yang24ab.html). [DOI](https://doi.org/10.48550/arXiv.2312.06635). [TeX source](https://export.arxiv.org/e-print/2312.06635v6). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Transformers with linear attention allow for efficient parallel training but can simultaneously be formulated as an RNN with 2D (matrix-valued) hidden states, thus enjoying linear-time inference complexity. However, linear attention generally underperforms ordinary softmax attention. Moreover, current implementations of linear attention lack I/O-awareness and are thus slower than highly optimized implementations of softmax attention. This work describes a hardware-efficient algorithm for linear attention that trades off memory movement against parallelizability. The resulting implementation, dubbed FlashLinearAttention, is faster than FlashAttention-2 [Dao23b] as a standalone layer even on short sequence lengths (e.g., 1K). We then generalize this algorithm to a more expressive variant of linear attention with data-dependent gates. When used as a replacement for the standard attention layer in Transformers, the resulting gated linear attention (GLA) Transformer is found to perform competitively against the LLaMA-architecture Transformer [Tou23] as well recent linear-time-inference baselines such as RetNet [Sun23b] and Mamba [Gu23] on moderate-scale language modeling experiments. GLA Transformer is especially effective at length generalization, enabling a model trained on 2K to generalize to sequences longer than 20K without significant perplexity degradations. For training speed, the GLA Transformer has higher throughput than a similarly-sized Mamba model.

[https://github.com/sustcsonglin/flash-linear-attention](https://github.com/sustcsonglin/flash-linear-attention)

<span id="section-1"></span>

## 1 Introduction

Transformers with softmax attention [Vas17a] enjoy efficient parallel training but suffer from quadratic (in sequence length) complexity, thus motivating more RNN-like models that allow for linear-time sequence modeling. Linear attention, which replaces the exponential similarity function with a simple dot product over (possibly transformed) key/query vectors, has emerged as a promising alternative to classic softmax attention [Kat20, Cho20a, Kas21, Pen21]. An attractive property of linear attention is that it admits a “recurrent form” in which it can be formulated as a linear RNN with 2D hidden states [Kat20], thus enabling linear-time inference. For training, linear attention also admits a subquadratic “chunkwise parallel form” which divides the sequence into non-overlapping chunks and performs (serial) inter-chunk recurrent computations followed by (parallel) intra-chunk computations [Hua22a, Sun23b, Lin23], thus (partially) maintaining parallel training. However, existing algorithms for linear attention are not I/O aware and thus, in practice, slower than optimized implementations of softmax attention [Dao22, Dao23b] on moderate sequence lengths.

From a performance standpoint, linear attention has generally been found to underperform ordinary softmax attention, often by a significant margin in language modeling [Kas21]. Recent variants of linear attention such as RetNet [Sun23b] and TransNormerLLM [Qin23c] obtain significant improvements by multiplying the current hidden state with a decay factor before the RNN update. However, these works use a global, *data-independent* decay factor, despite the fact that in 1D RNNs, a *data-dependent* gating mechanism has been shown to be crucial for performance [Wes18, Qin23a]. And even with the decay factor, linear attention Transformers underperform the strongest Transformer architectures when pretrained from scratch.

This work develops a hardware-efficient algorithm for linear attention, and applies it to train a gated variant of linear attention that is competitive with softmax attention. We first discuss aspects of optimizing ordinary linear attention on modern GPUs and give two I/O-aware algorithms (tailored for different training settings) based on these principles (§[Section 3](#section-3)). Our implementation of the algorithm, called FlashLinearAttention, is faster than FlashAttention-2 [Dao23b] even on short (e.g., 1K) sequences. We then describe a gated linear attention layer with a data-dependent gating mechanism and show how FlashLinearAttention can be generalized to the gated case (§[Section 4](#section-4)). We study the resulting *gated linear attention (GLA) Transformer* on moderate-scale language modeling benchmarks, where we train models with 340M/1.3B parameters on 15B/100B tokens, respectively. We find that the GLA Transformer performs favorably against a strong LLaMA architecture Transformer baseline that makes use of recent recipes [Tou23] as well as recent linear-time sequence models such as RetNet [Sun23b] and Mamba [Gu23]. GLA Transformer is found to be particularly strong at length generalization and recall-intensive tasks among linear recurrent models. For training speed, the GLA Transformer has significantly higher throughput than a similarly sized Mamba model.

<span id="section-2"></span>

## 2 Background: Linear Attention

We first give a brief background on linear attention layers. For notation we use bold upper-case letters for matrices (e.g., ${\mathbf{S}}$, ${\mathbf{Q}}$), bold lower-case letters for vectors (e.g., ${\bm{q}}_{t}$, ${\bm{k}}_{t}$), and italic upper-case for learnable parameters matrices (e.g., ${\bm{W}}_{K}$). We generally use the same alphabet to show the rows of a matrix, e.g., ${\bm{q}}_{t}$ is the $t$-th row of ${\mathbf{Q}}$.

<span id="section-2-1"></span>

### 2.1 Parallel and Recurrent Forms

Standard autoregressive Transformers employ a softmax attention mechanism which takes an input sequence ${\mathbf{X}}\in\mathbb{R}^{L\times d}$ (here $L$ is the length and $d$ is the hidden dimension) and computes the output ${\mathbf{O}}\in\mathbb{R}^{L\times d}$ through,

$$
\begin{aligned}
{\mathbf{Q}},{\mathbf{K}},{\mathbf{V}} & ={\mathbf{X}}{\bm{W}}_{Q},{\mathbf{X}}{\bm{W}}_{K},{\mathbf{X}}{\bm{W}}_{V}, \\
{\mathbf{O}} & =\mathrm{softmax}\big(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\mathbf{M}}\big){\mathbf{V}},
\end{aligned}
$$

where ${\bm{W}}_{Q},{\bm{W}}_{K},{\bm{W}}_{V}\in\mathbb{R}^{d\times d}$ are learnable matrices and ${\mathbf{M}}\in\{-\infty,1\}^{L\times L}$ is a mask that prevents the model from attending to future tokens, i.e., ${\mathbf{M}}_{ij}=1$ if $i\geq j$ and ${\mathbf{M}}_{ij}=-\infty$ if $i<j$. (Here we assume a single attention head for simplicity.) The above *parallel form* of attention can compute ${\mathbf{O}}$ in parallel given the full input ${\mathbf{X}}$, thus enabling efficient training. However, during inference Transformers must use the following *recurrent form*,

$$
\begin{aligned}
{\bm{q}}_{t},\ {\bm{k}}_{t},\ {\bm{v}}_{t} & ={\bm{x}}_{t}{\bm{W}}_{Q},\ {\bm{x}}_{t}{\bm{W}}_{K},\ {\bm{x}}_{t}{\bm{W}}_{V}, \\
{\bm{o}}_{t} & =\frac{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top){\bm{v}}_{i}}{\sum_{i=1}^{t}\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top)},
\end{aligned}
$$

which calculates the query (${\bm{q}}_{t}$), key (${\bm{k}}_{t}$), and value (${\bm{v}}_{t}$) vectors given the current token’s representation ${\bm{x}}_{t}\in\mathbb{R}^{1\times d}$ and the performs attention over the (growing) set of keys $\{{\bm{k}}_{1},\dots,{\bm{k}}_{t}\}$ and values $\{{\bm{v}}_{1},\dots,{\bm{v}}_{t}\}$ (i.e., the “KV cache”).

Linear attention mechanisms [Kat20] replace $\exp({\bm{q}}_{t}{\bm{k}}_{i}^\top)$ with a kernel $k({\bm{x}},{\bm{y}})$ with an associated feature map $\phi$ (i.e., $k({\bm{x}},{\bm{y}})=\langle\phi({\bm{x}}),\phi({\bm{y}})\rangle$). This simplifies the calculation of ${\bm{o}}_{t}$ since we have

$$
\begin{aligned}
{\bm{o}}_{t} & =\frac{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}}{\sum_{i=1}^{t}\phi({\bm{q}}_{t})\phi({\bm{k}}_{i})^\top}=\frac{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}}{\phi({\bm{q}}_{t})\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top}.
\end{aligned}
$$

Letting ${\mathbf{S}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top{\bm{v}}_{i}$ and ${\bm{z}}_{t}=\sum_{i=1}^{t}\phi({\bm{k}}_{i})^\top$ where ${\mathbf{S}}_{t}\in\mathbb{R}^{d\times d},{\bm{z}}_{t}\in\mathbb{R}^{d\times 1}$, we can rewrite the above as an RNN,

$$
\begin{aligned}
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1} & +\phi({\bm{k}}_{t})^\top{\bm{v}}_{t},\hskip 2.84526pt{\bm{z}}_{t}={\bm{z}}_{t-1}+\phi({\bm{k}}_{t})^\top,\hskip 2.84526pt{\bm{o}}_{t}=\frac{\phi({\bm{q}}_{t}){\mathbf{S}}_{t}}{\phi({\bm{q}}_{t}){\bm{z}}_{t}}.
\end{aligned}
$$

Although various kernels have been explored [Kas21, Pen21], recent work has found that a linear kernel (i.e., setting $\phi$ to be the identity) without a normalizer works well in practice [Sun23b]. This results in an (unnormalized) linear attention layer with the following update equation,

<span id="equation-01"></span>

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+{\bm{k}}_{t}^\top{\bm{v}}_{t},\quad{\bm{o}}_{t}={\bm{q}}_{t}{\mathbf{S}}_{t}.
$$

[Equation 1](#equation-01) makes it clear that a linear attention layer is essentially a linear recurrent layer with matrix-valued hidden states ${\mathbf{S}}_{t}$ that is updated via the outer-product ${\bm{k}}_{t}^\top{\bm{v}}_{t}=({\bm{x}}_{t}{\bm{W}}_{K})^\top({\bm{x}}_{t}{\bm{W}}_{V})$. [+1] The parallel form of causal linear attention, whose complexity is still quadratic in $L$, is given by ${\mathbf{O}}=\big(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\mathbf{M}}\big){\mathbf{V}}$, where ${\mathbf{M}}\in\{0,1\}^{L\times L}$ is a mask such that ${\mathbf{M}}_{ij}=1$ if $i\geq j$ and ${\mathbf{M}}_{ij}=0$ if $i<j$. Due to ${\mathbf{M}}$ it is not possible to exploit the associative property of matrix multiplication to reduce the parallel form complexity from quadratic to linear. [+2]

<span id="section-2-2"></span>

### 2.2 Chunkwise Parallel Form

The *chunkwise* parallel form of linear attention strikes a balance between parallel and recurrent form [Hua22a, Sun23b], and allows for subquadratic, partially parallel training. Formally, suppose the input ${\mathbf{X}}$ is now split into non-overlapping chunks, where each chunk is of length $C$. Let ${\mathbf{S}}_{[i]}\in\mathbb{R}^{d\times d}$ be the chunk-level hidden state after processing $i$ chunks, i.e., ${\mathbf{S}}_{[i]}:={\mathbf{S}}_{iC}$. Further let ${\mathbf{Q}}_{[i]}:={\mathbf{Q}}_{iC+1:(i+1)C+1}\in\mathbb{R}^{C\times d}$ be the query vectors corresponding to the $i$-th chunk; let ${\mathbf{K}}_{[i]}$, ${\mathbf{V}}_{[i]}$, ${\mathbf{O}}_{[i]}$ be similarly defined. We then have the following inter-chunk recurrence (for $i\in[0,1,\dots\frac{L}{C}-1]$):

<span id="equation-02"></span>

$$
{\mathbf{S}}_{[i+1]}={\mathbf{S}}_{[i]}+\underbrace{\sum_{j=iC+1}^{(i+1)C}{\bm{k}}_{j}^\top{\bm{v}}_{j}}_{{\mathbf{K}}^\top_{[i]}{\mathbf{V}}_{[i]}}\quad\hskip 2.84526pt\in\mathbb{R}^{d\times d}.
$$

Here ${\mathbf{S}}_{[0]}$ can be initialized to zero or from the previous segment’s hidden state. The sum of all RNN inputs from a chunk (i.e., ${\mathbf{K}}^\top_{[i]}{\mathbf{V}}_{[i]}$) can be computed in $O(C^{2}d)$ in parallel. The intra-chunk parallel computation for the output is given by

$$
{\mathbf{O}}_{[i+1]}=\underbrace{{\mathbf{Q}}_{[i+1]}{\mathbf{S}}_{[i]}}_{\mathrm{inter-chunk}:{\mathbf{O}}^{\mathrm{inter}}_{[i+1]}}+\underbrace{\big(({\mathbf{Q}}_{[i+1]}{\mathbf{K}}_{[i+1]}^\top)\odot{\mathbf{M}}\big){\mathbf{V}}_{[i+1]}}_{\mathrm{intra-chunk}:{\mathbf{O}}^{\mathrm{intra}}_{[i+1]}},
$$

where ${\mathbf{O}}_{[i+1]}\in\mathbb{R}^{C\times d}$. Here the “intra-chunk” component ${\mathbf{O}}^{\mathrm{intra}}_{[i+1]}$ has exactly the same parallel form as [Equation 1](#equation-01) and thus takes $O(C^{2}d+Cd^{2})$. The “inter-chunk” component ${\mathbf{O}}^{\mathrm{inter}}_{[i+1]}$ accounts for the contribution from the hidden state from the previous chunk, and takes $O(Cd^{2})$. Training complexity is thus $O\left(\frac{L}{C}(C^{2}d+Cd^{2})\right)=O(L C d+L d^{2})$, which is less than $O(L^{2}d)$ when $L>d$. Note that setting $C=L$ recovers the parallel form, and $C=1$ recovers the recurrent form.

<span id="section-3"></span>

## 3 Hardware-Efficient Linear Attention

We describe FlashLinearAttention, an I/O-aware, hardware-efficient algorithm for linear attention in the spirit of FlashAttention [Dao22, Dao23b]. We first discuss aspects of hardware that should be taken into account for a practically efficient implementation.

<span id="section-3-1"></span>

### 3.1 Principles of Hardware-Efficient Algorithms

An efficient algorithm should be aware of the compute model, memory hierarchy, and specialized compute units on modern hardware.

**Occupancy.** GPUs have many threads executed in parallel; threads are grouped into thread blocks, which execute on streaming multiprocessors (SMs). To maintain a high GPU occupancy (i.e., fraction of GPU resources being used), it is necessary to use a sufficient number of SMs. In large-scale training and long-sequence modeling scenarios where the batch size tends to be small, parallelizing over the temporal dimension enables high GPU occupancy [Dao23b].

**Specialized compute units.** Modern hardware for neural network training typically have specialized compute units (e.g., tensor cores on NVIDIA GPUs, matrix mutiply units on TPUs), which can significantly accelerate matmuls; for example half-precision matmuls on an A100 can be roughly 16 times faster on tensor cores than on CUDA cores. These specialized units are crucial for large-scale training.

**Memory hierarchy.** GPUs have a memory hierarchy with larger but slower global GPU memory (high bandwidth memory; HBM) and smaller but faster shared memory (SRAM). Optimal utilization of SRAM to reduce HBM I/O cost can therefore lead to significant speed-ups.

<span id="section-3-2"></span>

### 3.2 Hardware Considerations for Linear Attention

We now discuss hardware considerations pertaining to the efficiency of the different forms of linear attention.

**Recurrent form.** A basic implementation of the recurrent form stores the 2D hidden states of all time steps in HBM, resulting in high I/O cost [Mao22]. I/O cost could be reduced by avoiding such materialization and recomputing the hidden states during the backward pass, as in [Kat20], but the elementwise operations in the recurrent update cannot make use of tensor cores and result in low arithmetic intensity. Hence, while the recurrent form generally has the lowest total FLOPs among the three forms, this does not translate to actual wall-time efficiency. And while it is theoretically possible to parallelize linear recurrences via the parallel scan algorithm, this method requires materializing the 2D hidden state for each time step. This incurs a significant memory I/O burden, thereby offsetting the benefits of parallelism over the sequence length and resulting in slow actual running speeds, as in [Kat23].

**Parallel form.** The parallel form could be as efficient as FlashAttention using similar I/O optimization techniques, as demonstrated by [Qin23c]. However, the high number of FLOPs (due to the quadratic complexity) makes the long-sequence training expensive, the same issue that the naïve implementation of softmax attention would suffer from.

**Chunkwise form.** The chunkwise parallel form, which interpolates between the parallel and recurrent forms with an extra “parameter” $C$, makes it possible to more easily make the above tradeoffs for fine-grained optimization. Unlike the recurrent form, most operations can be done via matmuls, enabling the use of tensor cores (if $C$ is set to a multiple of 16). Though the chunkwise training algorithm has been discussed before in the literature [Hua22a, Sun23b], most implementations are not I/O-aware and thus slower than FlashAttention for moderate sequence lengths (e.g., 2K-4K).

<span id="figure-01"></span>

![Figure 1. (a) FlashLinearAttention without materialization. This version is more memory-efficient. (b-c) FlashLinearAttention with materialization. This version enables sequence-level chunkwise parallelism.](../../papers/gated-linear-attention/figure-01.png)

**Figure 1.** (a) FlashLinearAttention without materialization. This version is more memory-efficient. (b-c) FlashLinearAttention with materialization. This version enables sequence-level chunkwise parallelism.

<span id="section-3-3"></span>

### 3.3 FlashLinearAttention: Hardware-Efficient Linear Attention with the Chunkwise Form

We describe our I/O-aware, hardware-efficient implementation of the chunkwise form. We give two versions, whose forward and backward passes differ depending on whether the chunk-level hidden states ${\mathbf{S}}_{[n]}$ are materialized in HBM. See [Algorithm 1](#algorithm-01) and [Figure 1](#figure-01) for the forward pass. ([Algorithm 2](#algorithm-02) in the appendix describes the backward pass.) At a high level, we use tiling to load tensors block-by-block and re-use tensor blocks on chip to avoid multiple HBM I/O as much as possible. For example, when ${\mathbf{Q}}_{[n]}$ is loaded to SRAM, both ${\mathbf{Q}}_{[n]}{\mathbf{S}}$ and $({\mathbf{Q}}_{[n]}{\mathbf{K}}_{[n]}^{\top}\odot{\mathbf{M}}){\mathbf{V}}_{[n]}$ can be computed on chip, which avoids loading ${\mathbf{Q}}_{[n]}$ twice, thus saving HBM I/O.

<span id="algorithm-01"></span>

**Algorithm 1: FlashLinearAttention: Forward Pass.**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf V}\in\mathbb R^{L\times d}$, chunk size $C\in[L]$, `materialize` $\in\{$`True`,`False`$\}$.
- Divide ${\mathbf Q},{\mathbf K},{\mathbf V}$ into $N=L/C$ blocks of size $C\times d$ each.
- Initialize ${\mathbf S}=\mathbf 0\in\mathbb R^{d\times d}$ on SRAM and construct the causal mask ${\mathbf M}\in\mathbb R^{C\times C}$ on chip.
- **If** `materialize` (the materialization version):
  - **For** $n\gets1,N$:
    - Store ${\mathbf S}$ to HBM as ${\mathbf S}_{[n]}$.
    - Load ${\mathbf K}_{[n]},{\mathbf V}_{[n]}$ from HBM to SRAM.
    - On chip, compute ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
  - **Parallel for** $n\gets1,N$:
    - Load ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf S}_{[n]}$ from HBM to SRAM.
    - On chip, compute ${\mathbf O}'={\mathbf Q}_{[n]}{\mathbf S}_{[n]}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M}){\mathbf V}_{[n]}$.
    - Store ${\mathbf O}'$ to HBM as ${\mathbf O}_{[n]}$.
  - **Return** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$ and ${\mathbf S}=\{{\mathbf S}_{[1]}\dots{\mathbf S}_{[N]}\}$.
- **Else** (the non-materialization version):
  - **For** $n\gets1,N$:
    - Load ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]}$ from HBM to SRAM.
    - On chip, compute ${\mathbf O}'={\mathbf Q}_{[n]}{\mathbf S}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M}){\mathbf V}_{[n]}$.
    - On chip, compute ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
    - Store ${\mathbf O}'$ to HBM as ${\mathbf O}_{[n]}$.
  - **Return** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$.

</div>

**The non-materialization version** computes ${\mathbf{O}}_{[n]}$ sequentially for $n\in[N]$, using SRAM to temporarily store ${\mathbf{S}}_{[n]}$, which is memory-efficient. This version parallelizes across batch size, number of heads, and head dimensions, but lacks sequence-level parallelim. When the batch size is large, this level of parallelism is sufficient to enable high GPU occupancy. In long-sequence and large scale training settings where batch size is small, the SMs cannot be fully exploited in this case. **The materialization version** first performs the inter-chunk recurrence ([Equation 2](#equation-02)) and stores all ${\mathbf{S}}_{[n]}$ for $n\in[N]$ in HBM. Then, the ${\mathbf{O}}_{[n]}$’s can be computed in parallel for all chunks. This approach offers better parallelism but increases the memory footprint by approximately 10-20%. We mitigate this through *recomputation*, where the hidden states discarded after the forward pass and recomputed during the backward pass. We find this introduces a small runtime overhead but significantly reduces the memory footprint, and we adopt this strategy by default.

<span id="figure-02"></span>

![Figure 2. Speed comparison on a single H100 GPU with batch size 32, number of heads 16, head dimension 64, and chunk size 64. Both x- and y-axes are on log scale. *w/ m.* and *w/o m.* denotes using FlashLinearAttention *with* or *without materialization* of hidden states in HBM.](../../papers/gated-linear-attention/figure-02.png)

**Figure 2.** Speed comparison on a single H100 GPU with batch size 32, number of heads 16, head dimension 64, and chunk size 64. Both x- and y-axes are on log scale. *w/ m.* and *w/o m.* denotes using FlashLinearAttention *with* or *without materialization* of hidden states in HBM.

[Figure 2](#figure-02) shows the speed and memory footprint of our implementation. Both versions of FlashLinearAttention are substantially faster than FlashAttention-2 [Dao23b] and a pure PyTorch (i.e., I/O-unaware) implementation of chunkwise linear attention, showing the benefits of I/O-awareness.

<span id="section-4"></span>

## 4 Gated Linear Attention

The linear recurrence in [Equation 1](#equation-01) does not have a decay term or a forget gate, which has been shown to be crucial in RNNs [Bec24, Cho14a, Wes18]. The lack of a decay term makes it difficult for a model to “forget” information, and has been hypothesized to be partially responsible for the instability of linear attention in long-context tasks [Buc24]. Recent works [Sun23b, Qin23c] obtain better performance through incorporating a global, *non-data-dependent* decay factor [+3] $\gamma\in(0,1)$ into linear attention: ${\mathbf{S}}_{t}=\gamma{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^\top{\bm{v}}_{t}$. The use of a single $\gamma$ is designed to preserve the attention-style parallel form for efficient training. In this work, we consider a data-dependent gating mechanism for linear attention. We show that despite having a more expressive gating factor, the resulting gated linear attention (GLA) layer still admits a hardware-efficient chunkwise form for efficient training.

<span id="section-4-1"></span>

### 4.1 Recurrent and Parallel Form of GLA

<span id="table-01"></span>

![Table 1. Gated linear attention formulation of recent models, which vary in their parameterization of ${\mathbf{G}}_{t}$. The bias terms are omitted.](../../papers/gated-linear-attention/table-01.png)

**Table 1.** Gated linear attention formulation of recent models, which vary in their parameterization of ${\mathbf{G}}_{t}$. The bias terms are omitted.

**Recurrent form.** GLA has a 2D forget gate ${\mathbf{G}}_{t}\in(0,1)^{d_{k}\times d_{v}}$ that varies over time:

$$
{\mathbf{S}}_{t}={\mathbf{G}}_{t}\odot{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t},
$$

where we now allow the hidden state to have varying dimensions. This Hadamard product-based recurrent form is very general and encompasses many recent RNNs with 2D hidden states, as listed in [Table 1](#table-01).

Central to the design of gated linear attention is the parameterization of ${\mathbf{G}}_{t}$ which requires a balance between *parameter-efficiency*, *state size*, and *training efficiency*. A naïve mapping ${\bm{x}}_{t}\mapsto{\mathbf{G}}_{t}$ to obtain a data-dependent gating matrix would require a matrix of size $d\cdot d_{k}\cdot d_{v}$, which would be parameter-inefficient. [Mao22] propose a more efficient outer-product-based low-rank parameterization (${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}{\bm{\beta}}_{t}$), which requires $d\cdot d_{v}+d\cdot d_{k}$ parameters. [+4]

In Mamba [Gu23], ${\mathbf{G}}_{t}$ is obtained by combining a *data-independent* learnable matrix ${\bm{A}}$ with a data-dependent vector ${\bm{\alpha}}_{t}$, which allows the matrix to be full rank. However, this prevents the use of tensor cores because it cannot be reformulated into a matrix-multiply format, as discussed in [Dao24]. The lack of a compact matrix-multiply form necessitates the materialization of each time step’s hidden states. To reduce high I/O costs, [Gu23] develop a hardware-aware algorithm that materializes the hidden states exclusively in SRAM rather than in HBM. Due to limited SRAM capacity, this approach cannot scale to larger hidden states, which, as we will show in our experiments, results in suboptimal performance on recall-intensive tasks. Mamba-2 [Dao24] addresses this limitation with a more restricted gating mechanism: ${\mathbf{G}}_{t}=\gamma_{t}\mathbf{1}^\top\mathbf{1}$, where $\gamma_{t}\in(0,1)$ is a scalar, which makes it possible to to reformulate the recurrence in matrix-multiply form, enabling the use of tensor cores and larger state sizes. This *scalar* data-dependent gating is also used in [Pen21], [Sun24b], and [Bec24].

This paper adopts a middle ground between the scalar and the fully low-rank parameterization by using ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}\mathbf{1}$. [+5] This results in the following recurrent form,

<span id="equation-03"></span>

$$
{\mathbf{S}}_{t}=({\bm{\alpha}}_{t}^{\top}\mathbf{1})\odot{\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t}=\mathrm{Diag}({\bm{\alpha}}_{t}){\mathbf{S}}_{t-1}+{\bm{k}}_{t}^{\top}{\bm{v}}_{t},
$$

where ${\bm{\alpha}}_{t}$ is parameterized via a low-rank linear layer followed by sigmoid on ${\bm{x}}_{t}$ (see §[Section 4.4](#section-4-4)). Note that the above formulation is general and encompasses several recent RNNs [Kat23, Qin24a, Pen24a]. Thus, the hardware-efficient GLA implementation (described next) could be directly used or adapted to other models.

**Parallel form.** We now describe a parallel form GLA for parallelizing across sequence length. Unrolling [Equation 3](#equation-03) gives

$$
{\mathbf{S}}_{t}=\sum_{i=1}^{t}\left(\left(\prod_{j=i+1}^{t}{\bm{\alpha}}_{j}^{\top}\mathbf{1}\right)\odot{\bm{k}}_{i}^{\top}{\bm{v}}_{i}\right)
$$

Letting ${\bm{b}}_{t}:=\prod_{j=1}^{t}{\bm{\alpha}}_{j}$, we can rewrite the above as

$$
\begin{aligned}
{\bm{o}}_{t}={\bm{q}}_{t}{\mathbf{S}}_{t} & ={\bm{q}}_{t}\sum_{i=1}^{t}\left(\left(\frac{{\bm{b}}_{t}}{{\bm{b}}_{i}}\right)^{\top}\mathbf{1}\right)\odot{\bm{k}}_{i}^{\top}{\bm{v}}_{i} \\
=\sum_{i=1}^{t}({\bm{q}}_{t}\odot{\bm{b}}_{t})\left(\frac{{\bm{k}}_{i}}{{\bm{b}}_{i}}\right)^{\top}{\bm{v}}_{i}
\end{aligned}
$$

where the division is element-wise. Letting ${\mathbf{B}}\in(0,1)^{L\times d}$ be the matrix obtained from stacking ${\bm{b}}_{t}$’s, the parallel form is:

$$
{\mathbf{O}}=\left(\left(\underbrace{({\mathbf{Q}}\odot{\mathbf{B}})\left(\frac{{\mathbf{K}}}{{\mathbf{B}}}\right)^{\top}}_{\mathbf{P}}\right)\odot{\mathbf{M}}\right){\mathbf{V}}.
$$

However, this form is not numerical stable as ${\bm{b}}_{t}$ is the cumulative product of gate values in ${\bm{\alpha}}_{j}\in(0,1)^{1\times d}$, and thus can be extremely small when $t$ is large, making $\frac{\mathbf{K}}{{\mathbf{B}}}$ explode. To handle this, we can compute in log space for $\mathbf{P}$, [+6]

<span id="equation-04"></span>

$$
\mathbf{P}_{ij}=\sum_{k=1}^{d}\mathbf{Q}_{ik}\mathbf{K}_{jk}\,\exp(\log{\mathbf{B}}_{ik}-\log{\mathbf{B}}_{jk}),\quad i\geq j.
$$

where $k$ denotes feature indices. However, unlike vanilla linear attention, as [Equation 4](#equation-04) cannot be represented via a standard matmul, and it cannot make use of half-precision matmuls on tensor cores. We will show in §[Section 4.3](#section-4-3) how a secondary-level chunking mechanism can enable the use of half-precision matmuls for most computations while maintaining numerical stability, as illustrated in [Figure 3](#figure-03).

<span id="figure-03"></span>

![Figure 3. Attention-style map to illustrate the chunkwise computations in GLA. The inter-chunk dependencies (in gray) are not directly computed in the chunkwise form (only computed in the parallel form). The intra-chunk dependencies are modeled via secondary chunking/tiling where the inter-sub-chunk part (in orange) is computed by half-precision matmuls while the intra-sub-chunk part (in pink) is computed in full precision in log space.](../../papers/gated-linear-attention/figure-03.png)

**Figure 3.** Attention-style map to illustrate the chunkwise computations in GLA. The inter-chunk dependencies (in gray) are not directly computed in the chunkwise form (only computed in the parallel form). The intra-chunk dependencies are modeled via secondary chunking/tiling where the inter-sub-chunk part (in orange) is computed by half-precision matmuls while the intra-sub-chunk part (in pink) is computed in full precision in log space.

<span id="section-4-2"></span>

### 4.2 Chunkwise Parallel Form of GLA

We derive a chunkwise form of GLA similar to the chunkwise form of basic linear attention (§[Section 2.2](#section-2-2)). Here the intra-chunk operation implements the above parallel form at the chunk-level to obtain ${\mathbf{O}}^{\mathrm{intra}}$. For inter-chunk, we have

$$
\begin{aligned}
\mathbf{\Lambda}_{iC+j} & =\frac{{\bm{b}}_{iC+j}}{{\bm{b}}_{iC}},\mathbf{\Gamma}_{iC+j}=\frac{{\bm{b}}_{(i+1)C}}{{\bm{b}}_{iC+j}},{\bm{\gamma}}_{i+1}=\frac{{\bm{b}}_{(i+1)C}}{{\bm{b}}_{iC}}, \\
{\mathbf{S}}_{[i+1]} & =\left({\bm{\gamma}}_{i+1}^{\top}\mathbf{1}\right)\odot{\mathbf{S}}_{[i]}+\left({\mathbf{K}}_{[i+1]}\odot\mathbf{\Gamma}_{[i+1]}\right)^{\top}{\mathbf{V}}_{[i+1]}, \\
{\mathbf{O}}^{\mathrm{inter}}_{[i+1]} & =\left({\mathbf{Q}}_{[i+1]}\odot\mathbf{\Lambda}_{[i+1]}\right){\mathbf{S}}_{[i]}.
\end{aligned}
$$

Intuitively, $\mathbf{\Lambda}_{[i+1]}$ encodes the cumulative decay from the start of a chunk which will be used to propagate the hidden states from the previous chunk ${\mathbf{S}}_{[i]}$, while $\mathbf{\Gamma}_{[i+1]}$ encodes the decay to the end of a chunk which will be used to accumulate information to be added to the next hidden state ${\mathbf{S}}_{[i+1]}$.

<span id="section-4-3"></span>

### 4.3 Hardware-Efficient GLA

With the chunkwise form in hand, we can adapt the FlashLinear Attention algorithm presented in §[Section 3](#section-3) to the gated case. The adaptation additionally relies on two crucial techniques described below. We give high-level intuitions in this section and defer the full algorithms to [Algorithms 3–6](#algorithm-03) of Appendix [Section 9](#section-9).

**Secondary-level chunking.** Unlike in ordinary linear attention, the intra-chunk computations in GLA cannot leverage half-precision matmuls (and thus tensor cores) due to log space computations ([Equation 4](#equation-04)). To make better use of tensor cores, we use secondary-level chunking scheme, where a chunk is further divided into sub-chunks (i.e., another level of tiling) in the spirit of classic tiling techniques [Dao22]. The attention-like matrix ${\mathbf{P}}\in\mathbb{R}^{L\times L}$ is then computed in a chunkwise manner, as illustrated in [Figure 3](#figure-03). Concretely, the interactions between sub-chunks are computed via half-precision matmuls, [+7]

$$
\begin{aligned}
{\mathbf{P}}_{[i][j]} & =\Big({\mathbf{Q}}_{[i]}\odot{\mathbf{\Lambda}}_{[i]}\Big)\Big({\mathbf{K}}_{[j]}\odot{\mathbf{\Gamma}}_{[j]}\odot\frac{{\bm{b}}_{iC}}{{\bm{b}}_{(j+1)C}}\Big)^\top\in\mathbb{R}^{C\times C}.
\end{aligned}
$$

This corresponds to the orange tiles in [Figure 3](#figure-03). For the intra-sub-chunk part (pink tiles in [Figure 3](#figure-03)) we have to resort to [Equation 4](#equation-04) and perform the matmul in full precision for stability. With this two-level tiling strategy, the total amount of non-half-precision matmul FLOPs are greatly reduced, thus leading to wallclock improvements. We provide the Pytorch-style pseudo-code in [Listing 1](#listing-01) of Appendix [Section 9](#section-9).

**Memory-efficient ${\mathbf{d}\bm{\alpha}}_{t}$ computation.** Past work [Mao22] has claimed that GLA-like models have to materialize the matrix-valued hidden states of size $L\times d\times d$ in HBM to compute all the gradients ${\mathbf{d}\bm{\alpha}}_{t}$, since ${\mathbf{d}\bm{\alpha}}_{t}=({\mathbf{S}}_{t-1}\odot\mathbf{d}{\mathbf{S}}_{t})\mathbf{1}$. We instead give the following *closed form* formula for ${\mathbf{d}\log\bm{\alpha}}_{t}$,

$$
\begin{aligned}
{\mathbf{d}\log\bm{b}}_{t} & ={\bm{q}}_{t}\odot{\mathbf{d}\bm{q}}_{t}-{\bm{k}}_{t}\odot{\mathbf{d}\bm{k}}_{t},\hskip 11.38109pt{\mathbf{d}\log\bm{\alpha}}_{t}=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm{b}}_{i},
\end{aligned}
$$

which can be easily obtained by taking the derivative with respect to [Equation 4](#equation-04) (see Appendix [Section 9](#section-9) for full derivation). ${\mathbf{d}\bm{q}}_{t}$ and ${\mathbf{d}\bm{k}}_{t}$ can be computed as in [Algorithms 4 and 6](#algorithm-04).

<span id="section-4-4"></span>

### 4.4 GLA Transformer

We generalize the GLA layer to the multi-head case. Given $H$ heads, we have the following for each head $h\in[1,H]$,

$$
\begin{aligned}
{\mathbf{S}}^{h}_{t}=\left(\left({\bm{\alpha}}_{t}^{h}\right)^{\top}\mathbf{1}\right)\odot{\mathbf{S}}_{t-1}^{h}+{\bm{k}}_{t}^{h\top}\,{\bm{v}}^{h}_{t}\in\mathbb{R}^{d^{\prime}_{k}\times d^{\prime}_{v}}, \\
{\bm{o}}^{h}_{t}={\bm{q}}_{t}^{h}{\mathbf{S}}_{t}^{h}\in\mathbb{R}^{1\times d^{\prime}_{v}}, \\
{\bm{o}}^{\prime}_{t}=\mathrm{concat}(\mathrm{LN}({\bm{o}}^{1}_{t}),\dots,\mathrm{LN}({\bm{o}}^{H}_{t}))\in\mathbb{R}^{1\times d_{v}}, \\
{\bm{r}}_{t}=\mathrm{Swish}({\bm{x}}_{t}{\bm{W}}_{r}+{\bm{b}}_{r})\in\mathbb{R}^{1\times d_{v}}, \\
{\bm{y}}_{t}=({\bm{r}}_{t}\odot{\bm{o}}^{\prime}_{t}){\bm{W}}_{O}\in\mathbb{R}^{1\times d}.
\end{aligned}
$$

Here we use separate key ($d_{k}$) and value ($d_{v}$) dimensions; $d^{\prime}_{k}=d_{k}/H,d^{\prime}_{v}=d_{v}/H$ are the per-head key/value dimensions. LayerNorm ($\mathrm{LN}$) is applied after the output of each head, while the output projection and output gating operate on the concatenation of head outputs [Sun23b].

We then build up a Transformer-like model by interleaving multi-head GLA layers with feed-forward networks (FFN). Concretely, given layer $l$’s contextualized representation ${\mathbf{X}}^{(l)}$, we obtain ${\mathbf{X}}^{(l+1)}$ via,

$$
\begin{aligned}
{\mathbf{Y}}^{(l)}=\mathrm{GLA}(\mathrm{LN}({\mathbf{X}}^{(l)}))+{\mathbf{X}}^{(l)} \\
{\mathbf{X}}^{(l+1)}=\mathrm{SwiGLU}(\mathrm{LN}({\mathbf{Y}}^{(l)}))+{\mathbf{X}}^{(l)},
\end{aligned}
$$

where the SwiGLU FFN layer [Tou23] is,

$$
\mathrm{SwiGLU}({\mathbf{Z}})=(\mathrm{Swish}({\mathbf{Z}}{\bm{W}}_{1})\odot{\mathbf{Z}}{\bm{W}}_{2}){\bm{W}}_{3}.
$$

<span id="table-02"></span>

![Table 2. GLA Transformer results against Transformer++, RetNet, and Mamba.](../../papers/gated-linear-attention/table-02.png)

**Table 2.** GLA Transformer results against Transformer++ [Tou23], RetNet [Sun23b], and Mamba [Gu23]. All models are trained on the same subset of the SlimPajama dataset with the Mistral tokenizer. The 340M/1.3B models are trained for 15B/100B tokens respectively. The individual task performance is via zero-shot. We report the main results on the same set of tasks reported by [Gu23]. See Appendix [Section 11](#section-11) for results on other benchmarks, including 5-shot results. The last column shows the average over all benchmarks that use (normalized) accuracy as the metric.

**Parameter allocation.** As presented, our GLA layer employs two additional matrices for predicting ${\bm{\alpha}}_{t},{\bm{r}}_{t}$ (i.e., ${\bm{W}}_{\alpha},{\bm{W}}_{r}$) compared to a regular softmax attention layer. For parameter-efficiency, we use a low-rank parameterization

$$
{\bm{\alpha}}_{t}=\sigma(({\bm{x}}_{t}{\bm{W}}^{1}_{\alpha}{\bm{W}}^{2}_{\alpha}+{\bm{b}}_{\alpha})))^{\frac{1}{\tau}}\in\mathbb{R}^{1\times d_{k}},
$$

where ${\bm{W}}^{1}_{\alpha}\in\mathbb{R}^{d\times 16}$, ${\bm{W}}^{2}_{\alpha}\in\mathbb{R}^{16\times d_{k}}$, and $\tau=16$ is a temperature term to encourage model to have a slower forgetting rate. We further set $d_{k}=\frac{d}{2}$ and $d_{v}=d$ and use full-rank parameterizations for (${\bm{W}}_{Q},{\bm{W}}_{K},{\bm{W}}_{V},{\bm{W}}_{O},{\bm{W}}_{r}$). Ultimately, one GLA layer collectively needs (roughly) $4d^{2}$ parameters, as in regular softmax attention.

<span id="section-5"></span>

## 5 Empirical Study

<span id="section-5-1"></span>

### 5.1 Experimental Setup

Our main experiments are on language modeling, where we study whether GLA can perform competitively against a (i) strong Transformer baseline with modern architectural recipes and (ii) recent linear-time models. We use the SlimPajama dataset [Sob23] and tokenize it using the Mistral tokenizer [Jia23e]. The original dataset contains 627B tokens; we use a 100B subset.

**Baselines.** We evaluate GLA against three baselines: Transformer++ [Tou23], RetNet [Sun23b], and Mamba [Gu23]. Transformer++ is the LLaMA architecture with Rotary Positional Embeddings [Su24], SWiGLU [Sha20], and RMSNorm [Zha19]; we also use SwiGLU in the RetNet to replace its original FFN for fair comparison. For Mamba, we use the open source code. All our baselines are trained for the exact same number of tokens on the same dataset for fair comparison.

**Training details.** We train all models from scratch at two scales: 340M and 1.3B. All models are trained with AdamW [Los18] using a maximum learning rate of 3e-4. The 340M models are trained on 15B tokens with a batch size of 0.5M tokens, while the 1.3B models are trained on 100B tokens with a batch size of 2M tokens. We use a cosine learning rate schedule with a warmup of 0.5B/1B tokens for the 340M/1.3B settings, respectively. The initial and final learning rates are 3e-5. We use a weight decay of 0.01, and gradient clipping of 1.0.

<span id="section-5-2"></span>

### 5.2 Main Results

In addition to perplexity (ppl) on Wikitext (Wiki.), we consider a wide range of downstream tasks covering common-sense reasoning and question-answering as was used in [Gu23]: LAMBADA [Pap16], PiQA [Bis20], HellaSwag [Zel19], WinoGrande [Sak19], ARC-easy (ARC-e) and ARC-challenge (Arc-c) [Cla18]. In Appendix [Section 11](#section-11), we also include results on additional tasks: Copa [Roe11], SciQA [Aue23], OpenbookQA [Mih18b], BoolQA [Cla19]. We report perplexity (ppl) on WikiText and LAMBADA, accuracy normalized by length on HellaSwag, ARC-challenge and OpenbookQA, and accuracy on the other tasks. All evaluations are performed using the LM evaluation harness [Gao21c].

<span id="figure-04"></span>

![Figure 4. Accuracy (%) on the synthetic MQAR task.](../../papers/gated-linear-attention/figure-04.png)

**Figure 4.** Accuracy (%) on the synthetic MQAR task.

<span id="figure-05"></span>

![Figure 5. Length extrapolation on the test set of SlimPajama and PG19. We pretrain 1.3B models from scratch on SlimPajama for 100B tokens with different training length. <sup>*∗*</sup> indicates models using truncated BPTT with over 12 segments that are each of 2K-length.](../../papers/gated-linear-attention/figure-05.png)

**Figure 5.** Length extrapolation on the test set of SlimPajama and PG19. We pretrain 1.3B models from scratch on SlimPajama for 100B tokens with different training length. <sup>*∗*</sup> indicates models using truncated BPTT with over 12 segments that are each of 2K-length.

Our main results are shown in [Table 2](#table-02). Compared to RetNet which uses a data-independent decay rate, the GLA Transformer with data-dependent gates shows improved results on all tasks. Both GLA Transformer and Mamba show comparable performance to Transformer++.

**Recall-intensive tasks.** While subquadratic models can achieve competitive language modeling performance to Transformers, [Aro24] show that they lag behind softmax attention in recall-intensive tasks. We next evaluate GLA on real and synthetic tasks that focus on recall.

The synthetic MQAR task [Aro23] is a more challenging multi-query version of the induction head task [Dao22g] in which a model has to recall the token following a query token multiple times. We follow [Aro23]’s experimental setting and compare GLA against recent subquadractic models, including RetNet [Sun23b], Mamba [Gu23], Hyena [Pol23a] and RWKV-4 [Pen23b]. For RetNet and GLA the number of heads is set to 2; for other models we follow the default settings in [Aro23]. The results are shown in [Figure 4](#figure-04). Standard quadratic attention achieves perfect scores in all settings and is thus omitted. We find that models with matrix-valued hidden states (i.e., Mamba/RetNet/GLA) outperform Hyena/RWKV, and our GLA outperforms RetNet, confirming the benefits of using data-dependent gates.

Following [Aro24], we also test our models on three real recall-intensive tasks: FDA [Aro23a], SWDE [Loc19], and SQUAD [Raj18]. These tasks focus on information extraction or reading comprehension. As illustrated in [Table 3](#table-03), subquadratic models significantly underperform Transformers on the FDA and SWDE, both of which are information extraction tasks. However, GLA outperforms other subquadractic models, likely due to its larger recurrent state (compared to Mamba) and selection mechanism (compared to RetNet).

<span id="table-03"></span>

![Table 3. Comparison of different models in three recall-intensive tasks tested in Aro24. Higher is better for all tasks.](../../papers/gated-linear-attention/table-03.png)

**Table 3.** Comparison of different models in three recall-intensive tasks tested in [Aro24]. Higher is better for all tasks.

**Long sequence training and length extrapolation.** One advantage of linear attention models is that they allow for efficient long sequence training in linear time. To showcase this feature, we consider two training settings: (i) direct training on 8K-length contexts, (ii) training on 24K-length contexts through truncated backpropagation through time (TBPP) over 2K-length segments. [+8] In the latter case the gradients are not back-propagated across segments, and hence this approach has minimal overhead comparable to the standard 2K-length training strategy (where the initial hidden state is always set to zero). We pretrain 1.3B Mamba, RetNet, and GLA models on SlimPajama for 100B tokens on these settings and test them on both SlimPajama test set and PG19 [Rae20] test set.

[Figure 5](#figure-05) shows the perplexities of the tokens calculated in different position groups. For models trained on 2K-length contexts, GLA extrapolates better than Mamba/RetNet in most position buckets on the PG19 test set; Mamba struggles to extrapolate beyond 4K, while GLA/RetNet can generalize to 18K on the Slimpajama test set. Transformers cannot extrapolate beyond training length, which is a known failure mode. [+9] Pretraining in a long sequence consistently improves perplexities for all three models. We found marginal perplexity difference in the two settings for GLA, indicating that TBPTT might be a more economic approach to long-sequence training. Mamba benefits significantly from 8K-length training, and it performs similarly as GLA in the same training setting.

**Ablations.** We conduct a small-scale ablation study by training the 340M GLA variants for 7B tokens. We investigate (i) the importance of having both *fine-grained* and *data-dependent* gating and (ii) the influence of head dimension size. The results are shown in [Table 4](#table-04). For (i), we find that while data dependent scalar gates substantially improve upon RetNet, a finer-grained gating mechanism is still necessary. For (ii) we tune the number of heads to vary head dimensions, where by default GLA uses 4 heads. Increasing it to 8 (i.e., smaller head dimension) leads to relatively large perplexity degradation; reducing it to 1 (i.e., larger head dimension) actually performs best, but results in only marginal improvement while requiring much higher GPU memory. We thus choose 4 heads for our experiments.

<span id="table-04"></span>

![Table 4. Ablation study results on the 340M model trained for 7B tokens. We evaluate the model variants via the average perplexity of the last 200 training steps.](../../papers/gated-linear-attention/table-04.png)

**Table 4.** Ablation study results on the 340M model trained for 7B tokens. We evaluate the model variants via the average perplexity of the last 200 training steps.

<span id="section-5-3"></span>

### 5.3 Training Efficiency

<span id="figure-06"></span>

![Figure 6. Training throughput and GPU memory usage for 1.3B models on a single H100 GPU.](../../papers/gated-linear-attention/figure-06.png)

**Figure 6.** Training throughput and GPU memory usage for 1.3B models on a single H100 GPU.

[Figure 6](#figure-06) shows the throughput and memory usage as a function of the sequence length and batch size for the different 1.3B models on a single H100 GPU. [+10] Here GLA adopts the materialization version of FlashLinearAttention with recomputation of hidden state (§[Section 3.3](#section-3-3)). All models have linear space complexity, and the total GPU footprint difference among them is minimal. In terms of training throughput, Mamba lags behind Transformer++ and GLA, with GLA shows greater advantages in training lengths beyond 4096.

<span id="section-5-4"></span>

### 5.4 Limitations & Future Work

While our experiments with the GLA Transformer were on a respectable scale, we were unable to perform larger-scale experiments due to limited compute resources. Although it is unclear at this point how GLA would scale to even larger models/datasets, we anticipate that training efficiency of GLA become even more favorable compared to Mamba at larger scales. Specifically, when scaled to larger sizes (e.g., $>7$B), GLA can be more efficient than Mamba because of better use of tensor cores and GLA’s compatibility with tensor parallelism. [+11] Insofar as we are interested in leveraging the efficiency of linear attention, it would be interesting to apply GLA to other modalities (especially modalities with long-range dependencies), in line with recent work on applying state-of-the-art state-space models to other types of data [Yan23f, Zhu24f, Ma24d, Liu24z, Xin24a, Wan24ad, Wan24ae, Yan24n].

<span id="section-6"></span>

## 6 Related Work

We briefly discuss related work here and give an extended discussion of the related work in Appendix [Section 8](#section-8).

Traditional RNNs are difficult to scale due to the nonlinear dependencies between the hidden states and expensive matmul-based sequential hidden state updates. Linear RNNs/State-Space Models (SSMs)/Transformers eliminate nonlinear dependencies, making training parallelizable along the temporal dimension [Mar18, Gu22, Smi23]. Such models have been the focus of much recent work as a competitive sub-quadratic alternative to the Transformer architecture [Pen23b, Gu23, Qin23a, Qin23c, Sun23b, Wan22m].

Data-dependent decay rates have always been regarded important for RNNs [Ger00, Wes18]. Typical forget gate values depend on both the previous hidden state and the current input. However [Mar18] suggest that forget gate values should depend solely on the current inputs to enable parallel training. This simple strategy has been shown to be effective in moderate-scale experiments conducted by HGRN [Qin23a]. RWKV-v6 [Pen24a] and Mamba [Gu23] also use data-dependent decay rates that are reminiscent of forget gates. In the context of linear Transformers, [Pen21] employ a coarse-grained position-wise forget gate, while [Mao22] and [Kat23] use a more fine-grained forget gate.

RNNs rely on fixed-dimensional hidden states to encode their entire history. The hidden state dimension serves as a proxy for memory capacity and thus significantly influences their expressive power. Linear Transformers expand the hidden dimension of RNNs via the outer-product parameterization, as discussed §[Section 2.1](#section-2-1). Linear SSMs on the other hand expand their hidden dimension via a single-input-single-output (SISO) strategy. Without data-dependent SSM parameters, this can be done efficiently during training via the Fast Fourier Transform (FFT). However, with data-dependent SSM parameters, FFT-based training is not possible, and thus [Gu23] implements a custom CUDA kernel to train a selective state-space model using the parallel scan algorithm [Smi23]. To fit all the hidden states into SRAM, they can only afford an expansion rate up to 16. In contrast our hardware-aware training algorithm provides an alternative, efficient approach for expanding the hidden dimension to a wider range, which we have shown useful in recall-intensive tasks.

<span id="section-7"></span>

## 7 Conclusion

We propose an efficient algorithm for training linear attention Transformers with data-dependent gating mechanisms. Our algorithm makes it possible to balance FLOPs against parallellism, while still allowing for the use of half-precision matmuls which can take advantage of tensor core units on modern GPUs. Experiments on language modeling demonstrate that gated linear attention Transformers can perform respectably compared to strong baselines.

**Impact Statement.** This paper aims to improve the training efficiency of a new model family of (gated) linear attention models. The efficiency advantage of such models might help democratize access of language models. On the other hand, whether such new architectures would affect known issues such as biased and harmful outputs of language models remains an unexplored research question.

## Acknowledgments

This work was supported by MIT-IBM Watson AI Lab. We thank Yutao Sun, Zhen Qin, Li Dong, Xinyu Yang, Jiacheng You, Huanqi Cao, Yu Zhang, and Shida Wang for their insightful discussions. We also thank Yu Zhang, Fares Obeid, Daniel Goldstein, and Liliang Ren for their proofreading. Special thanks to Yu Zhang for contributing to the FlashLinearAttention library.

<span id="section-8"></span>

## 8 Extended Related Work

<span id="section-8-1"></span>

### 8.1 Linear Attention

**Feature map $\phi$.** Linear attention mechanisms [Kat20] replace $\exp({\bm q}_t{\bm k}_i^\top)$ with a kernel $k({\bm x},{\bm y})$ having an associated feature map $\phi$ (i.e., $k({\bm x},{\bm y})=\langle\phi({\bm x}),\phi({\bm y})\rangle$) where $\phi\in\mathbb R^{d_{\text{key}}}\rightarrow\mathbb R^{d_{\text{dot}}}$. $\phi$ often consists of two parts: $\phi=\phi_0\circ\phi_1$. $\phi_1$ could be a linear map made up by random samples [Pen21, Cho20a], learnable MLPs [Kas21, Zha24aa, Kac23] or simply an identity map [Mao22]. $\phi_2$ is often an element-wise (activation) function that makes the resulting $\phi$ a positive feature map, such as $1+\operatorname{elu}$ [Kat20], $\mathrm{ReLU}$ [Kas21], $\exp(\cdot)$ [Zha24aa, Cho20a]. Some work [Qin23c, Sun23b, Mao22] suggests that a positive feature map might not be necessary.

Our work follows [Sun23b] and [Mao22] by using an identity map $\phi=\mathbf I$. Recent work suggests that non-identity feature maps such as scaled element-wise exponential map [Nah23, Zha24aa] and higher-order polynomial map [Aro24, Kac23] work well empirically. We leave the exploration of integrating other types of feature map into GLA to future work.

**Attention spikiness.** Linear attention suffers from the “attention dilution” issue [Qin22a], where the attention distribution is too uniform (i.e., high entropy) to concentrate on relevant tokens. [Qin22a] propose adding local attention layers to focus more on adjacent tokens, a method adopted in [Lin23, Nah23, Zha23p] and proven crucial for performance. Recent work finds that a scaled element-wise exponential map—i.e., $\phi(\mathbf x)=\mathbf{\exp}(t\cdot\mathbf x)$ with $t\geq2$—helps to concentrate attention [Nah23, Zha24aa]. [Zha24aa] also find that higher-order polynomial kernels induce low-entropy and spiky attention distribution, partially explaining the empirical success of Based Linear Attention [Aro24] and PolySketchFormer [Kac23].

**Memory capacity.** Linear attention has bounded memory size [Pen22] while softmax attention enjoys unbounded memory [Ore24]. We believe that increasing the memory size efficiently and utilizing memory effectively are the keys to bridging the performance gap between linear attention and softmax attention. To increase memory size, it is shown that directly increasing $d_{\operatorname{key}}$ is effective [Sun23b, Mao22, Zha22f]; however, the total parameters are hard to control with the increase of $d_{\operatorname{key}}$. Parameter-efficient methods often keep $d_{\text{key}}$ intact and increase $d_{\text{dot}}$ instead. Higher order polynomial kernels with order $p\geq2$ map $d_{\text{key}}$ to a much higher $d_{\text{dot}}=O(d_\text{key}^p)$ [Aro23, Kac23]. [Sch21] propose the Deterministic Parameter-Free Projection (DPFP), while [Pra23] use parameterized outer product to expand $d_{\text{dot}}$ in a parameter-efficient/free manner.

For better memory utilization, [Sch21] use the delta rule to edit the memory dynamically. However, this is shown to underperform the gating mechanism [Mao22], which is a classic method to erase irrelevant historical information in gated RNNs. Recently, [Zha23p] enforce orthogonality of memory vectors to potentially increase utiliziation.

**Linear attention with decay or gates.** [Pen21] use position-wise scalar gates for incorporating recency bias into linear attention, and has been revisited in recent work [Dao24, Bec24, Sun24b], while [Mao22, Pra23] use matrix-valued gates (obtained by the outer product) for more fine-grained memory control.

Scalar decays can be easily incorporated into chunkwise linear attention for training efficiency [Sun23b, Qin24c]. With matrix-valued gates, the training efficiency becomes much more challenging. Both [Mao22] and [Kat23]’s training algorithms involve materializing hidden states of all steps in HBM, which suffers from high I/O costs. Moreover, both approaches cannot take advantage of tensor cores. Our hardware-efficient training algorithm reduces or eliminates materialization and enables usage of tensor cores.

**I/O-aware chunkwise linear attention.** The chunkwise form of linear attention is well-known in the literature. [Hua22a] first propose the chunkwise linear attention form, arguing that the training algorithm of [Kat20] is slow in practice. [Sun23b] and [Qin24c] generalize this form to linear attention with exponential decay (or ALiBi). [Kac23, Lin23] also derive similar chunkwise forms.

However, most chunkwise linear attention is not I/O-aware. To the best of our knowledge, only LightningAttention2 [Qin24c] (concurrent to our work) is I/O aware, and it is very similar to the non-materialization version of our FlashLinearAttention. We additionally propose a materialization version, which leverages sequence-level parallelism and thus allows for higher training throughput at the cost of a slightly increasing memory footprint.

**Other subquadratic models.** Besides the Linear attention Transformer [Kat20, Sch21] discussed in this work, previous studies have explored sparsifying attention with either a predefined fixed pattern [Chi19, Bel20a, Zah20] or a context-aware learnable pattern [Roy21, Kit20, Ren23] for sequence modeling with subquadratic complexity in the sequence length dimension. Leveraging convolutions for efficient sequence modeling has also been studied in works such as Dynamic Convolution [Wu19], Long Convolution [Fu23b, Qin23d, Pol23a, Mas23, Li23y, Rom21], and State Space Models [Gu22, Gup22, Gu21, Has22, Smi23, Ma23b].

<span id="section-8-2"></span>

### 8.2 Sequence Parallelism

The chunk-wise parallel form of linear Transformers resembles the two-stage parallel prefix sum (or parallel scan) algorithm [Ble90], which also combine chunk-wise parallel computations with inter-chunk communication [Cha15]. It also resembles sequence parallelism used for accelerating attention-based Transformers [Li23g], which has recently received much attention for long-sequence modeling [Liu23, Li23q, Bra23]. Sequence-level parallelism also constitutes the main improvement of FlashAttention-2 [Dao23b] over FlashAttention-1 [Dao22]. The main differences between these works are that (i) the chunk-level parallel form of linear Transformer needs only a single pass due to the linear complexity, while the sequence parallelism in Transformers needs $L/C$ passes (i.e., left-to-right scan of key/value blocks for each query block) due to the inherent quadratic complexity, and (ii) the order of matrix multiplications is different. We also note that chunkwise linear attention could greatly reduce the communication cost between devices in the distributed training setting compared to softmax attention, which could open the door for extremely long sequence training.

<span id="algorithm-02"></span>

**Algorithm 2: FlashLinearAttention: Backward Pass.**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d}$, chunk size $C\in[L]$, `materialize` $\in\{$`True`,`False`$\}$, ${\mathbf S}\in\mathbb R^{(L/C)\times d\times d}$ (available when `materialize` is `True`).
- Initialize ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d\times d}$ on SRAM and construct ${\mathbf M}\in\mathbb R^{C\times C}$ on chip.
- **If** `materialize`:
  - **For** $n\gets N,1$ in reverse order:
    - Store ${\mathbf{dS}}$ in HBM as ${\mathbf{dS}}_{[n]}$; load ${\mathbf Q}_{[n]},{\mathbf{dO}}_{[n]}$ to SRAM; compute ${\mathbf{dS}}={\mathbf{dS}}+{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
  - **Parallel for** $n\gets1,N$:
    - Load ${\mathbf Q}_{[n]},{\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf{dO}}_{[n]},{\mathbf S}_{[n]},{\mathbf{dS}}_{[n]}$ from HBM to SRAM.
    - On chip, compute ${\mathbf{dQ}}={\mathbf{dO}}_{[n]}{\mathbf S}_{[n]}^\top+({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top\odot{\mathbf M}){\mathbf K}_{[n]}$.
    - On chip, compute ${\mathbf{dK}}={\mathbf V}_{[n]}{\mathbf{dS}}_{[n]}^\top+({\mathbf V}_{[n]}{\mathbf{dO}}_{[n]}^\top\odot{\mathbf M}^\top){\mathbf Q}_{[n]}$.
    - On chip, compute ${\mathbf{dV}}={\mathbf K}_{[n]}{\mathbf{dS}}_{[n]}+({\mathbf Q}_{[n]}{\mathbf K}_{[n]}^\top\odot{\mathbf M})^\top{\mathbf{dO}}_{[n]}$, then write ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$ to HBM.
- **Else**:
  - Initialize ${\mathbf S}=\mathbf0\in\mathbb R^{d\times d}$ on SRAM.
  - **For** $n\gets1,N$ (hidden state recomputation): load ${\mathbf K}_{[n]},{\mathbf V}_{[n]},{\mathbf{dO}}_{[n]}$, compute ${\mathbf{dQ}}={\mathbf{dO}}_{[n]}{\mathbf S}^\top+({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top\odot{\mathbf M}){\mathbf K}_{[n]}$, and update ${\mathbf S}={\mathbf S}+{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
  - **For** $n\gets N,1$ in reverse order: load the chunk tensors, compute ${\mathbf{dS}}={\mathbf{dS}}+{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$, compute ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$ as above, and write them to HBM.
- **Return** ${\mathbf{dQ}}=\{{\mathbf{dQ}}_{[1]}\dots{\mathbf{dQ}}_{[N]}\}$, ${\mathbf{dK}}=\{{\mathbf{dK}}_{[1]}\dots{\mathbf{dK}}_{[N]}\}$, and ${\mathbf{dV}}=\{{\mathbf{dV}}_{[1]}\dots{\mathbf{dV}}_{[N]}\}$.

</div>

<span id="section-8-3"></span>

### 8.3 Hardware-ware Algorithm

Many algorithms are fast in theory, but slow in practice, due to misalignment with hardware properties [Hoo20, Sap23]. For example, matmuls with butterfly matrices have theoretically lower complexity by using FFT, but in practice it is slow due to extensive memory transportation operations, motivating matrices [Dao22b, Fu23c] which can better align butterfly operators to GPUs. In practice it is important to reduce HBM I/O cost using techniques such as tiling and recomputation and leverage tensor cores as much as possible. Our FlashLinearAttention is similar in spirit to FlashAttention [Dao22, Dao23b] and FlashConvFFT [Fu23], which implement I/O-aware versions of neural network layers to enable practical wallclock speedups. Concurrent work by [Qin24c] also proposes an I/O-aware version of linear attention, which is similar to the non-materialization version of FlashLinearAttention. We additionally propose a materialization version, which leverages sequence-level parallelism and thus allows for higher training throughput at the cost of a slightly increasing memory footprint.

<span id="section-9"></span>

## 9 Details for Chunkwise (Gated) Linear Attention

**Backward pass of FlashLinearAttention.** The pseduocode for backward pass of linear attention is listed in [Algorithm 2](#algorithm-02).

**Pseudo codes of GLA.** We first present the direct adaptions of FlashLinearAttention to training GLA without secondary-level chunking. Specifically, [Algorithms 3](#algorithm-03) and [4](#algorithm-04) shows the forward/backward pass for the materialization version; [Algorithms 5](#algorithm-05) and [6](#algorithm-06) for the non-materialization version.

We show the psuedo code of our secondary-level chunking in Pytorch style in [Listing 1](#listing-01).

<span id="listing-01"></span>

**Listing 1: Pytorch-like code snippet of our two-level chunking algorithm for training GLA. We omit the dimensions of batch size and number of heads for clarity.**

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

**Algorithm 3: Forward pass for gated linear attention (w. materialization).**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V}\in\mathbb R^{L\times d_v}$, ${\mathbf G}=[{\bm\alpha}_1\dots{\bm\alpha}_L]$, and chunk size $C$.
- Divide ${\mathbf Q},{\mathbf K},{\mathbf G}$ into $N=L/C$ blocks of size $C\times d_k$, and ${\mathbf V}$ into $N$ blocks of size $C\times d_v$. Initialize ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$ on SRAM.
- **For** $n\gets1,N$: write ${\mathbf S}$ to HBM as ${\mathbf S}_{[n]}$; load ${\mathbf K}_{[n]},{\mathbf G}_{[n]},{\mathbf V}_{[n]}$; compute ${\bm\gamma}_{[n]}$, ${\mathbf\Gamma}_{[n]}$, $\widetilde{\mathbf K}_{[n]}={\mathbf K}_{[n]}\odot{\mathbf\Gamma}_{[n]}$, and ${\mathbf S}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf S}+\widetilde{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
- **Parallel for** $n\gets1,N$: load the chunk tensors and ${\mathbf S}_{[n]}$; construct ${\mathbf M}$; compute ${\mathbf\Lambda}_{[n]}$, ${\mathbf\Gamma}_{[n]}$, $\widetilde{\mathbf Q}_{[n]}={\mathbf Q}_{[n]}\odot{\mathbf\Lambda}_{[n]}$, $\widetilde{\mathbf K}_{[n]}={\mathbf K}_{[n]}\odot{\mathbf\Gamma}_{[n]}$, and $\overline{\mathbf K}_{[n]}={\mathbf K}_{[n]}/{\mathbf\Lambda}_{[n]}$.
  - Compute ${\mathbf O}_{[n]}^{\mathrm{inter}}=\widetilde{\mathbf Q}_{[n]}{\mathbf S}_{[n]}$, ${\mathbf P}=(\widetilde{\mathbf Q}_{[n]}\overline{\mathbf K}_{[n]}^\top)\odot{\mathbf M}$, ${\mathbf O}^{\mathrm{intra}}={\mathbf P}{\mathbf V}_{[n]}$, and ${\mathbf O}_{[n]}={\mathbf O}^{\mathrm{inter}}+{\mathbf O}^{\mathrm{intra}}$; store ${\mathbf O}_{[n]}$ to HBM.
- **Return** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$ and ${\mathbf S}=\{{\mathbf S}_{[1]}\dots{\mathbf S}_{[N]}\}$.

</div>

<span id="algorithm-04"></span>

**Algorithm 4: Backward pass for gated linear attention (w. materialization).**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d_v}$, and chunk size $C$.
- Initialize ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d_k\times d_v}$ on SRAM.
- **For** $n\gets N,1$: store ${\mathbf{dS}}$ as ${\mathbf{dS}}_{[n]}$; load ${\mathbf G}_{[n]},{\mathbf Q}_{[n]},{\mathbf{dO}}_{[n]}$; compute ${\bm\gamma}_{[n]}$, ${\mathbf\Gamma}_{[n]}$, $\widetilde{\mathbf Q}_{[n]}={\mathbf Q}_{[n]}\odot{\mathbf\Gamma}_{[n]}$, and ${\mathbf{dS}}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf{dS}}+\widetilde{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
- **Parallel for** $n\gets1,N$: load the chunk tensors, states, and state gradients; construct ${\mathbf M}$ and the gated query/key forms; compute ${\mathbf P}$ and ${\mathbf{dP}}=({\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top)\odot{\mathbf M}$.
  - Compute ${\mathbf{d\bar K}}_{[n]}=\widetilde{\mathbf Q}_{[n]}{\mathbf{dP}}^\top$, ${\mathbf{d\widetilde K}}_{[n]}={\mathbf V}_{[n]}{\mathbf{dS}}_{[n]}^\top$, ${\mathbf{dK}}_{[n]}={\mathbf{d\widetilde K}}_{[n]}\odot{\mathbf\Gamma}_{[n]}+{\mathbf{d\bar K}}_{[n]}/{\mathbf\Lambda}_{[n]}$.
  - Compute ${\mathbf{d\widetilde Q}}_{[n]}={\mathbf{dP}}\overline{\mathbf K}_{[n]}+{\mathbf{dO}}_{[n]}{\mathbf S}_{[n]}^\top$, ${\mathbf{dQ}}_{[n]}={\mathbf{d\widetilde Q}}_{[n]}\odot{\mathbf\Lambda}_{[n]}$, and ${\mathbf{dV}}_{[n]}={\mathbf P}^\top{\mathbf{dO}}_{[n]}+\widetilde{\mathbf K}_{[n]}{\mathbf{dS}}_{[n]}$; store gradients in HBM.
- Let ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}}$ be the concatenated chunk gradients. Compute ${\mathbf{dA}}={\mathbf Q}\odot{\mathbf{dQ}}-{\mathbf K}\odot{\mathbf{dK}}$ and ${\mathbf{dG}}=\mathrm{revcum}({\mathbf{dA}})$.
- **Return** ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}},{\mathbf{dG}}$.

</div>

<span id="algorithm-05"></span>

**Algorithm 5: Forward pass for gated linear attention (w/o. materialization).**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V}\in\mathbb R^{L\times d_v}$, ${\mathbf G}=[{\bm\alpha}_1\dots{\bm\alpha}_L]$, and chunk size $C$.
- Divide the inputs into $N$ chunks and initialize ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$ on SRAM.
- **For** $n\gets1,N$: load the chunk tensors; compute ${\bm\gamma}_{[n]}$, ${\mathbf\Lambda}_{[n]}$, ${\mathbf\Gamma}_{[n]}$, $\widetilde{\mathbf Q}_{[n]}$, $\widetilde{\mathbf K}_{[n]}$, and $\overline{\mathbf K}_{[n]}$; construct ${\mathbf M}$.
  - Compute ${\mathbf O}_{[n]}^{\mathrm{inter}}=\widetilde{\mathbf Q}_{[n]}{\mathbf S}$, ${\mathbf P}=(\widetilde{\mathbf Q}_{[n]}\overline{\mathbf K}_{[n]}^\top)\odot{\mathbf M}$, ${\mathbf O}^{\mathrm{intra}}={\mathbf P}{\mathbf V}_{[n]}$, and ${\mathbf O}_{[n]}={\mathbf O}^{\mathrm{inter}}+{\mathbf O}^{\mathrm{intra}}$; store ${\mathbf O}_{[n]}$.
  - Update ${\mathbf S}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf S}+\widetilde{\mathbf K}_{[n]}^\top{\mathbf V}_{[n]}$.
- **Return** ${\mathbf O}=\{{\mathbf O}_{[1]}\dots{\mathbf O}_{[N]}\}$.

</div>

<span id="algorithm-06"></span>

**Algorithm 6: Backward pass for gated linear attention (w/o. materialization).**

<div class="paper-algorithm">

- **Require:** ${\mathbf Q},{\mathbf K},{\mathbf G}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf O},{\mathbf{dO}}\in\mathbb R^{L\times d_v}$, and chunk size $C$.
- Initialize ${\mathbf S}=\mathbf0\in\mathbb R^{d_k\times d_v}$ on SRAM.
- **For** $n\gets1,N$: load the gate, query, and output-gradient chunks; compute the gate summaries, ${\mathbf{dP}}={\mathbf{dO}}_{[n]}{\mathbf V}_{[n]}^\top$, ${\mathbf{d\widetilde Q}}_{[n]}={\mathbf{dP}}\widetilde{\mathbf K}_{[n]}+{\mathbf{dO}}_{[n]}{\mathbf S}^\top$, and ${\mathbf{dQ}}={\mathbf{d\widetilde Q}}_{[n]}\odot{\mathbf\Gamma}_{[n]}$; store ${\mathbf{dQ}}_{[n]}$ and update ${\mathbf S}$.
- Initialize ${\mathbf{dS}}=\mathbf0\in\mathbb R^{d_k\times d_v}$ on SRAM.
- **For** $n\gets N,1$: load the chunk tensors and gradients; construct ${\mathbf M}$ and the gated query/key forms; compute ${\mathbf P}$, ${\mathbf{dP}}$, ${\mathbf{dK}}_{[n]}$, and ${\mathbf{dV}}_{[n]}$ as in [Algorithm 4](#algorithm-04); store them in HBM; update ${\mathbf{dS}}=({\bm\gamma}_{[n]}^\top\mathbf1)\odot{\mathbf{dS}}+\widetilde{\mathbf Q}_{[n]}^\top{\mathbf{dO}}_{[n]}$.
- Concatenate the chunk gradients. Compute ${\mathbf{dA}}={\mathbf Q}\odot{\mathbf{dQ}}-{\mathbf K}\odot{\mathbf{dK}}$ and ${\mathbf{dG}}=\mathrm{revcum}({\mathbf{dA}})$.
- **Return** ${\mathbf{dQ}},{\mathbf{dK}},{\mathbf{dV}},{\mathbf{dG}}$.

</div>

**Derivations of ${\mathbf{d}\log\bm\alpha}_t$.** We show the derivations for the following gradient form.

$$
{\mathbf{d}\log\bm b}_t={\bm k}_t\odot{\mathbf{d}\bm k}_t-{\bm q}_t\odot{\mathbf{d}\bm q}_t,\qquad
{\mathbf{d}\log\bm\alpha}_t=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm b}_i.
$$

By unrolling the recurrence, we have

$$
{\bm o}_t={\bm q}_t{\mathbf S}_t=\sum_{i=1}^{t}({\bm q}_t\odot{\bm b}_t)\left(\frac{{\bm k}_i}{{\bm b}_i}\right)^\top{\bm v}_i
=\sum_{i=1}^{t}({\bm q}_t\odot\exp(\log{\bm b}_t))\left({\bm k}_i\odot\exp(-\log{\bm b}_i)\right)^\top{\bm v}_i,
$$

where at the second step, we apply a trivial identity: $\exp(\log x)=x$.

We first derive the gradients wrt. query/key vectors,

$$
{\mathbf{d}\bm q}_t=\sum_{i=1}^{t}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle{\bm b}_t\odot{\bm k}_i/{\bm b}_i,
\qquad
{\mathbf{d}\bm k}_i=\sum_{t=i}^{L}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle{\bm q}_t\odot{\bm b}_t/{\bm b}_i.
$$

Then for the gradients wrt. the logits of the accumulative gates,

$$
{\mathbf{d}\log\bm b}_t={\bm q}_t\odot\underbrace{\sum_{i=1}^{t}\langle{\mathbf{d}\bm o}_t,{\bm v}_i\rangle\odot{\bm b}_t\odot{\bm k}_i/{\bm b}_i}_{{\mathbf{d}\bm q}_t}
-{\bm k}_t\odot\underbrace{\sum_{i=t}^{L}\langle{\mathbf{d}\bm o}_i,{\bm v}_t\rangle{\bm q}_i\odot{\bm b}_i/{\bm b}_t}_{{\mathbf{d}\bm k}_t}.
$$

where we change the index notation for the ${\mathbf{d}\bm k}$ term. It now becomes clear that ${\mathbf{d}\log\bm b}_t={\bm q}_t\odot{\mathbf{d}\bm q}_t-{\bm k}_t\odot{\mathbf{d}\bm k}_t$. Since $\log{\bm b}_t=\sum_{i=1}^{t}\log{\bm\alpha}_i$, we get ${\mathbf{d}\log\bm\alpha}_t=\sum_{t=i}^{L}{\mathbf{d}\log\bm b}_i$.

<span id="section-10"></span>

## 10 General Gated Linear Attention

In the main paper, we use a simplified parameterization where ${\bm\beta}$ is fixed to $\mathbf1$ in the following gated linear attention.

$$
{\mathbf S}_t=({\bm\alpha}_t^\top{\bm\beta}_t)\odot{\mathbf S}_{t-1}+{\bm k}_t^\top{\bm v}_t.
$$

Though empirically we found that making ${\bm\beta}$ learnable does not lead to performance gain, we show here that the general form still enjoys parallel form and chunk-wise form, which could be potentially useful for future development of linear attention models.

<span id="section-10-1"></span>

### 10.1 Parallel Form

By unrolling the recurrence we have,

$$
{\bm o}_t={\bm q}_t{\mathbf S}_t={\bm q}_t\sum_{i=1}^{t}\left(\left(\prod_{j=i+1}^{t}{\mathbf G}_j\right)\odot({\bm k}_i^\top{\bm v}_i)\right).
$$

By taking advantage of the mixed product property of Kronercker/outer product, we have

$$
\left(\prod_{j=i+1}^{t}{\mathbf G}_j\right)\odot({\bm k}_i^\top{\bm v}_i)
=\left(\frac{{\bm b}_t}{{\bm b}_i}\odot{\bm k}_i\right)^\top\left(\frac{{\bm d}_t}{{\bm d}_i}\odot{\bm v}_i\right),
$$

where ${\bm b}_t=\prod_{j=1}^{t}{\bm\alpha}_j$, ${\bm d}_t=\prod_{j=1}^{t}{\bm\beta}_j$. By plugging it into the expanded recurrence, we have the following form.

$$
\begin{aligned}
{\bm o}_t
&=\sum_{i=1}^{t}\left({\bm q}_t\left(\frac{{\bm b}_t}{{\bm b}_i}\odot{\bm k}_i\right)^\top\right)\left(\frac{{\bm d}_t}{{\bm d}_i}\odot{\bm v}_i\right)\\
&=\sum_{i=1}^{t}\left(\left({\bm q}_t\odot{\bm b}_t\right)\left(\frac{{\bm k}_i}{{\bm b}_i}\right)^\top\left(\frac{{\bm v}_i}{{\bm d}_i}\right)\right)\odot{\bm d}_t\in\mathbb R^{1\times d_v}.
\end{aligned}
$$

The first equality is by linearity and associative property of matrix multiplication, and the second is derived based on $\langle{\bm a},{\bm b}\odot{\bm c}\rangle=\langle{\bm a}\odot{\bm b},{\bm c}\rangle$. The final form has following equivalent parallel form similar to the parallel form of linear/softmax attention.

$$
\widetilde{\mathbf Q}={\mathbf Q}\odot{\mathbf B},\quad
\widetilde{\mathbf K}={\mathbf K}/{\mathbf B},\quad
\widetilde{\mathbf V}={\mathbf V}/{\mathbf D},\qquad
\widetilde{\mathbf O}=(\widetilde{\mathbf Q}\widetilde{\mathbf K}^\top\odot{\mathbf M})\widetilde{\mathbf V},\quad
{\mathbf O}=\widetilde{\mathbf O}\odot{\mathbf D},
$$

where ${\mathbf Q},{\mathbf K},{\mathbf B}\in\mathbb R^{L\times d_k}$, ${\mathbf V},{\mathbf D}\in\mathbb R^{L\times d_v}$, ${\mathbf M}\in\mathbb R^{L\times L}$ denotes the causal mask.

<span id="section-10-2"></span>

### 10.2 Chunkwise Parallel Form

Now we show that the chunkwise parallel form for efficient training of general linear attention. Suppose ${\mathbf X}$ is now split into $L/C$ chunks, each of length $C$. Let ${\mathbf S}_{[i]}\in\mathbb R^{d_k\times d_v}$ be the chunk-level hidden state after processing $i$ chunks, i.e., ${\mathbf S}_{[i]}:={\mathbf S}_{iC}$. Further let ${\mathbf K}_{[i+1]}:={\mathbf K}_{iC+1:(i+1)C}\in\mathbb R^{C\times d_k}$, ${\mathbf V}_{[i+1]}:={\mathbf V}_{iC+1:(i+1)C}\in\mathbb R^{C\times d_v}$. The inter-chunk recurrence is then given by,

$$
{\mathbf S}_{[i+1]}=\left(\left(\frac{{\mathbf B}_{(i+1)C}}{{\mathbf B}_{iC}}\right)^\top\left(\frac{{\mathbf D}_{(i+1)C}}{{\mathbf D}_{iC}}\right)\right)\odot{\mathbf S}_{[i]}
+({\mathbf B}'_{[i+1]}\odot{\mathbf K}_{[i+1]})^\top({\mathbf D}'_{[i+1]}\odot{\mathbf V}_{[i+1]}),
$$

where $({\mathbf B}'_{[i+1]})_j={\mathbf B}_{(i+1)C}/{\mathbf B}_{iC+j}\in\mathbb R^{1\times d_k}$ and $({\mathbf D}'_{[i+1]})_j={\mathbf D}_{(i+1)C}/{\mathbf D}_{iC+j}\in\mathbb R^{1\times d_v}$ for $j\in[1,C]$, $i\in[0,L/C-1]$. The intra-chunk parallel computation is then given by,

$$
\begin{aligned}
\widetilde{\mathbf O}_{[i+1]}&=\underbrace{(({\mathbf Q}_{[i+1]}\odot{\mathbf B}^{\dagger}_{[i+1]}){\mathbf S}_{[i]})\odot{\mathbf D}^{\dagger}_{[i+1]}}_{\mathrm{inter-chunk}}\\
&\quad+\underbrace{(\widetilde{\mathbf Q}_{[i+1]}\widetilde{\mathbf K}_{[i+1]}^\top\odot{\mathbf M})\widetilde{\mathbf V}_{[i+1]}}_{\mathrm{intra-chunk}},\\
{\mathbf O}_{[i+1]}&=\widetilde{\mathbf O}_{[i+1]}/{\mathbf D}^{\dagger}_{[i+1]}.
\end{aligned}
$$

where $({\mathbf B}_{[i+1]}^{\dagger})_j={\mathbf B}_{iC+j}/{\mathbf B}_{iC}$ and $({\mathbf D}_{[i+1]}^{\dagger})_j={\mathbf D}_{iC+j}/{\mathbf D}_{iC}$. Subsequently, we have $\widetilde{\mathbf Q}_{[i+1]}={\mathbf Q}_{[i+1]}\odot{\mathbf B}_{[i+1]}^{\dagger}$, $\widetilde{\mathbf K}_{[i+1]}={\mathbf K}_{[i+1]}/{\mathbf B}_{[i+1]}^{\dagger}$, $\widetilde{\mathbf V}_{[i+1]}={\mathbf V}_{[i+1]}\odot{\mathbf D}_{[i+1]}^{\dagger}$. For initial values, we set ${\mathbf S}_0=\mathbf0$, ${\mathbf B}_0=\mathbf1$, ${\mathbf D}_0=\mathbf1$. Intuitively, ${\mathbf B}'_{[i]}$ encodes the cumulative decay from the start of a chunk which will be used to propagate the hidden states from the previous chunk ${\mathbf S}_{[i]}$; ${\mathbf B}^{\dagger}_{[i]}$ encodes the decay to the end of a chunk which will be used to accumulate information to be added to the next hidden state ${\mathbf S}_{[i+1]}$.

The chunkwise form given here is a generalization of several existing forms for linear attention. If we set ${\mathbf A}_{ij}=1$, ${\mathbf B}_{ij}=1$, it reduces to the chunk-wise form presented in the main paper for vanilla linear attention; if we set ${\mathbf A}_{ij}=1$, ${\mathbf B}_{ij}=\gamma^{i+1}$, it becomes RetNet's chunk-wise form [Sun23b]. As such, our formulation can be regarded as a generalized chunk-wise parallel form for linear attention that enables fine-grained data-dependent decay.

**Memory-efficient computation of ${\mathbf{d}\bm\alpha}$ and ${\mathbf{d}\bm\beta}$.** In the general form, we show that the gradient wrt. ${\bm\alpha}$ and ${\bm\beta}$ admits the following closed form, which allows computing ${\mathbf{d}\bm\alpha}$ and ${\mathbf{d}\bm\beta}$ without instantiating ${\mathbf S}$ in HBM.

$$
\begin{aligned}
{\mathbf{d}\log\bm b}_t&={\bm k}_t\odot{\mathbf{d}\bm k}_t-{\bm q}_t\odot{\mathbf{d}\bm q}_t,&
{\mathbf{d}\log\bm\alpha}_t&=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm b}_i,\\
{\mathbf{d}\log\bm d}_t&={\bm o}_t\odot{\mathbf{d}\bm o}_t-{\bm v}_t\odot{\mathbf{d}\bm v}_t,&
{\mathbf{d}\log\bm\beta}_t&=\sum_{t\leq i\leq L}{\mathbf{d}\log\bm d}_i.
\end{aligned}
$$

where $\log{\bm b}_t=\sum_{i=1}^{t}\log{\bm\alpha}_i$, $\log{\bm d}_t=\sum_{i=1}^{t}{\bm\beta}_i$ (or alternatively ${\bm b}_t=\prod_{i=1}^{t}{\bm\alpha}_i$, ${\bm d}_t=\prod_{i=1}^{t}{\bm\beta}_i$). We apply the trick to compute ${\mathbf{d}\log\bm b}_t$ and ${\mathbf{d}\log\bm d}_t$ for the cumulative-sum form above. The gradient of $\log{\bm b}_t$ comes from two sources: one associated with ${\bm q}_t$, the other associated with ${\bm k}_i$. Similarly, $\log{\bm d}_t$ comes from both ${\bm o}_t$ and ${\bm v}_i$. The trick applied there is that $\partial f({\bm a}\odot{\bm b})/\partial\log{\bm b}={\bm a}\odot\partial f({\bm a}\odot{\bm b})/\partial{\bm a}$ and $\partial f({\bm a}/{\bm b})/\partial\log{\bm b}=-\partial f({\bm a}/{\bm b})/\partial{\bm a}\odot{\bm a}$.

<span id="section-11"></span>

## 11 Additional Experimental Results

<span id="table-05"></span>

![Table 5. Extended zero- and five-shot performance results.](../../papers/gated-linear-attention/table-05.png)

**Table 5.** Extended zero- and five-shot performance results. All models are trained on the same subset of SlimPajama dataset with Mistral tokenizer. The 340M/1.3B models are trained for 15B/100B tokens respectively. The last column shows the average of all accuracies.

The complete results on all 11 tasks, including the 5-shot results for the 1.3B models, are shown in [Table 5](#table-05).

[+1]: This type of model with matrix-valued hidden states that change over time is also known as “fast weights” [Hin87, Sch92, Ba16a], whose connection to Transformers was explored in recent work [Sch21, Iri21, Mao22].

[+2]: Without ${\mathbf{M}}$, one can transform $({\mathbf{Q}}{\mathbf{K}}^\top){\mathbf{V}}$ to ${\mathbf{Q}}({\mathbf{K}}^\top{\mathbf{V}})$ reducing the complexity from quadratic ($O(L^{2}d)$) to linear ($O(Ld^{2})$).

[+3]: This can be viewed as linear attention with ALiBi position encodings [Pre21]. In practice these works also incorporate rotary position embeddings [Su24].

[+4]: However, [Mao22] works with only the recurrent form and materializes the hidden states for all time steps in HBM. In Appendix [Section 10](#section-10) we give a new algorithm that reformulates the model in a matrix-multiply-based parallel form, which can make use of (an extension of) FlashLinearAttention for efficient training.

[+5]: Our preliminary experiments with the ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}{\bm{\beta}}_{t}$ parameterization resulted in only marginal improvements over ${\mathbf{G}}_{t}={\bm{\alpha}}_{t}^{\top}\mathbf{1}$.

[+6]: This form resembles extrapolatable position encoding [Sun22] in that the term inside the exponential can be viewed as a *data-dependent* relative position factor.

[+7]: To reduce notational clutter, here we use the notations from the first-level chunking to express the key idea. The actual implementation is done with secondary-level chunks.

[+8]: We split a 24K input sequence into 12 segments. The final state of the previous segment is used as the initial state for the current segment.

[+9]: Although there are positional encoding schemes that enable better length extrapolation, these methods still have difficulty generalizing significantly beyond context lengths seen during training [Pre21, Sun22, Li23x].

[+10]: We use the official implementation for Mamba, the fused version of SwiGLU for Transformer++ and GLA, and FlashAttention-2 for Transformer++.

[+11]: In particular, since Mamba is not a multi-head model it is not as amenable to tensor parallelism.

[+author-note]: Equal contribution.
