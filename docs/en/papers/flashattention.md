---
title: 'FlashAttention'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/flashattention/
---

> [Tri Dao](https://tridao.me/), [Daniel Y. Fu](https://danfu.org/), [Stefano Ermon](https://cs.stanford.edu/~ermon/), [Atri Rudra](http://www.cse.buffalo.edu/~atri/), and [Christopher Ré](http://cs.stanford.edu/people/chrismre/). First submitted to arXiv on May 27, 2022; current version v2. [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135). [Original PDF](/paper/flashattention.pdf). [TeX source](https://export.arxiv.org/e-print/2205.14135). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Transformers are slow and memory-hungry on long sequences, since the time and memory complexity of self-attention are quadratic in sequence length. Approximate attention methods have attempted to address this problem by trading off model quality to reduce the compute complexity, but often do not achieve wall-clock speedup. We argue that a missing principle is making attention algorithms IO-aware—accounting for reads and writes between levels of GPU memory. We propose FlashAttention, an IO-aware exact attention algorithm that uses tiling to reduce the number of memory reads/writes between GPU high bandwidth memory (HBM) and GPU on-chip SRAM. We analyze the IO complexity of FlashAttention, showing that it requires fewer HBM accesses than standard attention, and is optimal for a range of SRAM sizes. We also extend FlashAttention to block-sparse attention, yielding an approximate attention algorithm that is faster than any existing approximate attention method. FlashAttention trains Transformers faster than existing baselines: 15% end-to-end wall-clock speedup on BERT-large (seq. length 512) compared to the MLPerf 1.1 training speed record, 3$\times$ speedup on GPT-2 (seq. length 1K), and 2.4$\times$ speedup on long-range arena (seq. length 1K-4K). FlashAttention and block-sparse FlashAttention enable longer context in Transformers, yielding higher quality models (0.7 better perplexity on GPT-2 and 6.4 points of lift on long-document classification) and entirely new capabilities: the first Transformers to achieve better-than-chance performance on the Path-X challenge (seq. length 16K, 61.4% accuracy) and Path-256 (seq. length 64K, 63.1% accuracy).

## 1 Introduction

Transformer models [Vaswan17] have emerged as the most widely used architecture in applications such as natural language processing and image classification. Transformers have grown larger [Brown20] and deeper [Wang22], but equipping them with longer context remains difficult [Tay20], since the self-attention module at their heart has time and memory complexity quadratic in sequence length. An important question is whether making attention faster and more memory-efficient can help Transformer models address their runtime and memory challenges for long sequences.

Many approximate attention methods have aimed to reduce the compute and memory requirements of attention. These methods range from sparse-approximation [Kitaev20, Roy21] to low-rank approximation [Wang20, Kathar20, Chorom20], and their combinations [Beltag20, Zaheer20, Chen21]. Although these methods reduce the compute requirements to linear or near-linear in sequence length, many of them do not display wall-clock speedup against standard attention and have not gained wide adoption. One main reason is that they focus on FLOP reduction (which may not correlate with wall-clock speed) and tend to ignore overheads from memory access (IO).

<span id="figure-01"></span>

![Refer to caption](../../papers/flashattention/figure-01.png)

**Figure 1.** Left: FlashAttention uses tiling to prevent materialization of the large $N\times N$ attention matrix (dotted box) on (relatively) slow GPU HBM. In the outer loop (red arrows), FlashAttention loops through blocks of the $\mathbf{K}$ and $\mathbf{V}$ matrices and loads them to fast on-chip SRAM. In each block, FlashAttention loops over blocks of $\mathbf{Q}$ matrix (blue arrows), loading them to SRAM, and writing the output of the attention computation back to HBM. Right: Speedup over the PyTorch implementation of attention on GPT-2. FlashAttention does not read and write the large $N\times N$ attention matrix to HBM, resulting in an 7.6$\times$ speedup on the attention computation.

In this paper, we argue that a missing principle is making attention algorithms IO-aware [Vitter88]—that is, carefully accounting for reads and writes to different levels of fast and slow memory (e.g., between fast GPU on-chip SRAM and relatively slow GPU high bandwidth memory, or HBM [Jiaa18], [Figure 1](#figure-01) left). On modern GPUs, compute speed has out-paced memory speed [NVIDIA17, NVIDIA20, NVIDIA22], and most operations in Transformers are bottlenecked by memory accesses [Ivanov21]. IO-aware algorithms have been critical for similar memory-bound operations, when reading and writing data can account for a large portion of the runtime—such as database joins [Ramakr03], image processing [RaganK13], numerical linear algebra [Blackf02], and more [Willia09, Patter03]. However, common Python interfaces to deep learning such as PyTorch and Tensorflow do not allow fine-grained control of memory access.

We propose FlashAttention, a new attention algorithm that computes exact attention with far fewer memory accesses. Our main goal is to avoid reading and writing the attention matrix to and from HBM. This requires (i) computing the softmax reduction without access to the whole input (ii) not storing the large intermediate attention matrix for the backward pass. We apply two well-established techniques to address these challenges. (i) We restructure the attention computation to split the input into blocks and make several passes over input blocks, thus incrementally performing the softmax reduction (also known as tiling). (ii) We store the softmax normalization factor from the forward pass to quickly recompute attention on-chip in the backward pass, which is faster than the standard approach of reading the intermediate attention matrix from HBM. We implement FlashAttention in CUDA to achieve fine-grained control over memory access and fuse all the attention operations into one GPU kernel. Even with the increased FLOPs due to recomputation, our algorithm both runs faster (up to 7.6x on GPT-2 [Radfor19], [Figure 1](#figure-01) right) and uses less memory—linear in sequence length—than standard attention, thanks to the massively reduced amount of HBM access.

We analyze the IO complexity [Vitter88] of FlashAttention, proving that it requires $O(N^{2}d^{2}M^{-1})$ HBM accesses where $d$ is the head dimension and $M$ is the size of SRAM, as compared to $\Omega(\mathrm{Nd}+N^{2})$ of standard attention. For typical values of $d$ and $M$, FlashAttention requires many times fewer HBM accesses compared to standard attention (up to 9$\times$ fewer, as shown in [Fig. 2](#figure-02)). Moreover, we provide a lower bound, showing that no exact attention algorithm can asymptotically improve on the number of HBM accesses over all SRAM sizes.

We also show that FlashAttention can serve as a useful primitive for realizing the potential of approximate attention algorithms by overcoming their issues with memory access overhead. As a proof of concept, we implement block-sparse FlashAttention, a sparse attention algorithm that is 2-4$\times$ faster than even FlashAttention, scaling up to sequence length of 64k. We prove that block-sparse FlashAttention has better IO complexity than FlashAttention by a factor proportional to the sparsity ratio. We discuss further extensions to other operations (attention on multi-GPU, kernel regression, block-sparse matrix multiply) in [Section 5](#S5 "5 Limitations and Future Directions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). We open-source FlashAttention to make it easier to build on this primitive. [+1]

We empirically validate that FlashAttention speeds up model training and improves model quality by modeling longer context. We also benchmark the runtime and memory footprint of FlashAttention and block-sparse FlashAttention compared to prior attention implementations.

-   •

    Faster Model Training. FlashAttention trains Transformer models faster in wall-clock time. We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1 [Mattso20], GPT2 (seq. length 1K) 3$\times$ faster than baseline implementations from HuggingFace [Wolf20] and Megatron-LM [Shoeyb19], and long-range arena (seq. length 1K-4K) 2.4$\times$ faster than baselines.

-   •

    Higher Quality Models. FlashAttention scales Transformers to longer sequences, which improves their quality and enables new capabilities. We observe a 0.7 improvement in perplexity on GPT-2 and 6.4 points of lift from modeling longer sequences on long-document classification [Dai22]. FlashAttention enables the first Transformer that can achieve better-than-chance performance on the Path-X [Tay20] challenge, solely from using a longer sequence length (16K). Block-sparse FlashAttention enables a Transformer to scale to even longer sequences (64K), resulting in the first model that can achieve better-than-chance performance on Path-256.

-   •

    Benchmarking Attention. FlashAttention is up to 3$\times$ faster than the standard attention implementation across common sequence lengths from 128 to 2K and scales up to 64K. Up to sequence length of 512, FlashAttention is both faster and more memory-efficient than any existing attention method, whereas for sequence length beyond 1K, some approximate attention methods (e.g., Linformer) start to become faster. On the other hand, block-sparse FlashAttention is faster than all existing approximate attention methods that we know of.

## 2 Background

We provide some background on the performance characteristics of common deep learning operations on modern hardware (GPUs). We also describe the standard implementation of attention.

### 2.1 Hardware Performance

We focus here on GPUs. Performance on other hardware accelerators are similar [Jouppi17, Jiaa19].

GPU Memory Hierarchy. The GPU memory hierarchy ([Fig. 1](#figure-01) left) comprises multiple forms of memory of different sizes and speeds, with smaller memory being faster. As an example, the A100 GPU has 40-80GB of high bandwidth memory (HBM) with bandwidth 1.5-2.0TB/s and 192KB of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around 19TB/s [Jiaa18, Sandt21]. The on-chip SRAM is an order of magnitude faster than HBM but many orders of magnitude smaller in size. As compute has gotten faster relative to memory speed [NVIDIA17, NVIDIA20, NVIDIA22], operations are increasingly bottlenecked by memory (HBM) accesses. Thus exploiting fast SRAM becomes more important.

Execution Model. GPUs have a massive number of threads to execute an operation (called a kernel). Each kernel loads inputs from HBM to registers and SRAM, computes, then writes outputs to HBM.

Performance characteristics. Depending on the balance of computation and memory accesses, operations can be classified as either compute-bound or memory-bound. This is commonly measured by the *arithmetic intensity* [Willia09], which is the number of arithmetic operations per byte of memory access.

1.  1.

    Compute-bound: the time taken by the operation is determined by how many arithmetic operations there are, while time accessing HBM is much smaller. Typical examples are matrix multiply with large inner dimension, and convolution with large number of channels.

2.  2.

    Memory-bound: the time taken by the operation is determined by the number of memory accesses, while time spent in computation is much smaller. Examples include most other operations: elementwise (e.g., activation, dropout), and reduction (e.g., sum, softmax, batch norm, layer norm).

Kernel fusion. The most common approach to accelerate memory-bound operations is kernel fusion: if there are multiple operations applied to the same input, the input can be loaded once from HBM, instead of multiple times for each operation. Compilers can automatically fuse many elementwise operations [Refa20, Paszke19, Sabne20]. However, in the context of model training, the intermediate values still need to be written to HBM to save for the backward pass, reducing the effectiveness of naive kernel fusion.

### 2.2 Standard Attention Implementation

Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ where $N$ is the sequence length and $d$ is the head dimension, we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

where $\mathrm{softmax}$ is applied row-wise.

Standard attention implementations materialize the matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM, which takes $O(N^{2})$ memory. Often $N\gg d$ (e.g., for GPT2, $N=1024$ and $d=64$). We describe the standard attention implementation in [Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") . As some or most of the operations are memory-bound (e.g., softmax), the large number of memory accesses translates to slow wall-clock time.

This problem is exacerbated by other elementwise operations applied to the attention matrix, such as masking applied to $\mathbf{S}$ or dropout applied to $\mathbf{P}$. As a result, there have been many attempts to fuse several elementwise operations, such as fusing masking with softmax [Shoeyb19].

In [Section 3.2](#S3.SS2 "3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), we will show that the standard attention implementation performs HBM accesses quadratic in the sequence length $N$. We also compare the number of FLOPs and number of HBM accesses of standard attention and of our method (FlashAttention).

<span id="alg0"></span>

**Algorithm 0: Standard Attention Implementation**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM.
- <span id="alg0.l1"></span> Load $\mathbf{Q},\mathbf{K}$ by blocks from HBM, compute $\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$, write $\mathbf{S}$ to HBM.
- <span id="alg0.l2"></span> Read $\mathbf{S}$ from HBM, compute $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$, write $\mathbf{P}$ to HBM.
- <span id="alg0.l3"></span> Load $\mathbf{P}$ and $\mathbf{V}$ by blocks from HBM, compute $\mathbf{O}=\mathbf{P}\mathbf{V}$, write $\mathbf{O}$ to HBM.
- **Return:** $\mathbf{O}$.

## 3 FlashAttention: Algorithm, Analysis, and Extensions

We show how to compute exact attention with fewer HBM reads/writes and without storing large intermediate matrices for the backward pass. This yields an attention algorithm that is both memory efficient and faster in wall-clock time. We analyze its IO complexity, showing that our method requires much fewer HBM accesses compared to standard attention. We further show that FlashAttention can serve as a useful primitive by extending it to handle block-sparse attention.

We focus here on the forward pass for ease of exposition; [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") contains details for the backward.

### 3.1 An Efficient Attention Algorithm With Tiling and Recomputation

Given the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM, we aim to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$ and write it to HBM. Our goal is to reduce the amount of HBM accesses (to sub-quadratic in $N$).

We apply two established techniques (tiling, recomputation) to overcome the technical challenge of computing exact attention in sub-quadratic HBM accesses. We describe this in [Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). The main idea is that we split the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks, load them from slow HBM to fast SRAM, then compute the attention output with respect to those blocks. By scaling the output of each block by the right normalization factor before adding them up, we get the correct result at the end.

Tiling. We compute attention by blocks. Softmax couples columns of $\mathbf{K}$, so we decompose the large softmax with scaling [Gimels18, Kitaev20, Staats21]. For numerical stability, the softmax of vector $x\in\mathbb{R}^{B}$ is computed as:

$$
m(x):=\max_{i}\ \ x_{i},\quad f(x):=\begin{bmatrix}e^{x_{1}-m(x)}&\ldots&e^{x_{B}-m(x)}\end{bmatrix},\quad\ell(x):=\sum_{i}f(x)_{i},\quad\mathrm{softmax}(x):=\frac{f(x)}{\ell(x)}.
$$

For vectors $x^{(1)},x^{(2)}\in\mathbb{R}^{B}$, we can decompose the softmax of the concatenated $x=\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix}\in\mathbb{R}^{2B}$ as:

$$
m(x)=m(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=\max(m(x^{(1)}),m(x^{(2)})),\quad f(x)=\begin{bmatrix}e^{m(x^{(1)})-m(x)}f(x^{(1)})&e^{m(x^{(2)})-m(x)}f(x^{(2)})\end{bmatrix},
$$

$$
\ell(x)=\ell(\begin{bmatrix}x^{(1)}\ x^{(2)}\end{bmatrix})=e^{m(x^{(1)})-m(x)}\ell(x^{(1)})+e^{m(x^{(2)})-m(x)}\ell(x^{(2)}),\quad\mathrm{softmax}(x)=\frac{f(x)}{\ell(x)}.
$$

Therefore if we keep track of some extra statistics ($m(x),\ell(x)$), we can compute softmax one block at a time. [+2] We thus split the inputs $\mathbf{Q},\mathbf{K},\mathbf{V}$ into blocks ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [3](#alg1.l3 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")), compute the softmax values along with extra statistics ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [10](#alg1.l10 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")), and combine the results ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [12](#alg1.l12 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")).

Recomputation. One of our goals is to not store $O(N^{2})$ intermediate values for the backward pass. The backward pass typically requires the matrices $\mathbf{S},\mathbf{P}\in\mathbb{R}^{N\times N}$ to compute the gradients with respect to $\mathbf{Q},\mathbf{K},\mathbf{V}$. However, by storing the output $\mathbf{O}$ and the softmax normalization statistics $(m,\ell)$, we can recompute the attention matrix $\mathbf{S}$ and $\mathbf{P}$ easily in the backward pass from blocks of $\mathbf{Q},\mathbf{K},\mathbf{V}$ in SRAM. This can be seen as a form of selective gradient checkpointing [Walthe08, Chen16]. While gradient checkpointing has been suggested to reduce the maximum amount of memory required [Staats21], all implementations (that we know off) have to trade speed for memory. In contrast, even with more FLOPs, our recomputation speeds up the backward pass due to reduced HBM accesses ([Fig. 2](#figure-02)). The full backward pass description is in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

Implementation details: Kernel fusion. Tiling enables us to implement our algorithm in one CUDA kernel, loading input from HBM, performing all the computation steps (matrix multiply, softmax, optionally masking and dropout, matrix multiply), then write the result back to HBM (masking and dropout in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). This avoids repeatedly reading and writing of inputs and outputs from and to HBM.

<span id="alg1"></span>

**Algorithm 1: FlashAttention**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM, on-chip SRAM of size $M$.
- Set block sizes $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- <span id="alg1.l2"></span> Initialize $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$ in HBM.
- <span id="alg1.l3"></span> Divide $\mathbf{Q}$ into $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ blocks $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ of size $B_{r}\times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ blocks $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, of size $B_{c}\times d$ each.
- Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ of size $B_{r}\times d$ each, divide $\ell$ into $T_{r}$ blocks $\ell_{i},\dots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\dots,m_{T_{r}}$ of size $B_{r}$ each.
- <span id="alg1.l5"></span> **For** $1\leq j\leq T_{c}$ **do:**
  - <span id="alg1.l6"></span> Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.
  - **For** $1\leq i\leq T_{r}$ **do:**
    - <span id="alg1.l8"></span> Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.
    - <span id="alg1.l9"></span> On chip, compute $\mathbf{S}_{\mathrm{ij}}=\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - <span id="alg1.l10"></span> On chip, compute $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (pointwise), $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
    - On chip, compute $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
    - <span id="alg1.l12"></span> Write $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j})$ to HBM.
    - Write $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ to HBM.
- **Return:** $\mathbf{O}$.

We show FlashAttention’s correctness, runtime, and memory requirement (proof in [Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")).

###### Theorem 1.

[Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") returns $\mathbf{O}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$ with $O(N^{2}d)$ FLOPs and requires $O(N)$ additional memory beyond inputs and output.

### 3.2 Analysis: IO Complexity of FlashAttention

We analyze the IO complexity of FlashAttention, showing significant reduction in HBM accesses compared to standard attention. We also provide a lower bound, proving that no exact attention algorithm can asymptotically improve on HBM accesses over all SRAM sizes. Proofs are in [Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

###### Theorem 2.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d\leq M\leq \mathrm{Nd}$. Standard attention ([Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") ) requires $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses, while FlashAttention ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) requires $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses.

For typical values of $d$ (64-128) and $M$ (around 100KB), $d^{2}$ is many times smaller than $M$, and thus FlashAttention requires many times fewer HBM accesses than standard implementation. This leads to both faster execution and lower memory footprint, which we validate in [Section 4.3](#S4.SS3 "4.3 Benchmarking Attention ‣ 4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

The main idea of the proof is that given the SRAM size of $M$, we can load blocks of $\mathbf{K},\mathbf{V}$ of size $\Theta(M)$ each ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [6](#alg1.l6 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). For each block of $\mathbf{K}$ and $\mathbf{V}$, we iterate over all blocks of $\mathbf{Q}$ ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [8](#alg1.l8 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) to compute the intermediate values, resulting in $\Theta(\mathrm{NdM}^{-1})$ passes over $\mathbf{Q}$. Each pass loads $\Theta(\mathrm{Nd})$ elements, which amounts to $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses. We similarly prove that the backward pass of standard attention requires $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses while the backward pass of FlashAttention requires $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses ([Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")).

We prove a lower-bound: one cannot asymptotically improve on the number of HBM accesses for all values of $M$ (the SRAM size) when computing exact attention.

###### Proposition 3.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d\leq M\leq \mathrm{Nd}$. There does not exist an algorithm to compute exact attention with $o(N^{2}d^{2}M^{-1})$ HBM accesses for all $M$ in the range $[d,\mathrm{Nd}]$.

The proof relies on the fact that for $M=\Theta(\mathrm{Nd})$ any algorithm must perform $\Omega(N^{2}d^{2}M^{-1})=\Omega(\mathrm{Nd})$ HBM accesses. This type of lower bound over a subrange of $M$ is common in the streaming algorithms literature [Woodru04]. We leave proving parameterized complexity [Grohe06] lower bounds in terms of $M$ as exciting future work.

We validate that the number of HBM accesses is the main determining factor of attention run-time. In [Fig. 2](#figure-02) (left), we see that even though FlashAttention has higher FLOP count compared to standard attention (due to recomputation in the backward pass), it has much fewer HBM accesses, resulting in much faster runtime. In [Fig. 2](#figure-02) (middle), we vary the block size $B_{c}$ of FlashAttention, which results in different amounts of HBM accesses, and measure the runtime of the forward pass. As block size increases, the number of HBM accesses decreases (as we make fewer passes over the input), and runtime decreases. For large enough block size (beyond 256), the runtime is then bottlenecked by other factors (e.g., arithmetic operations). Moreover, larger block size will not fit into the small SRAM size.

<span id="figure-02"></span>

![Refer to caption](../../papers/flashattention/figure-02.png)

**Figure 2.** Left: Forward + backward runtime of standard attention and FlashAttention for GPT-2 medium (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. HBM access is the primary factor affecting runtime. Middle: Forward runtime of FlashAttention (seq. length 1024, head dim. 64, 16 heads, batch size 64) on A100 GPU. Fewer HBM accesses result in faster runtime, up to a point. Right: The runtime (for seq. length 4K) of block-sparse FlashAttention is faster than FlashAttention by a factor proportional to the sparsity.

### 3.3 Extension: Block-Sparse FlashAttention

We extend FlashAttention to approximate attention: we propose block-sparse FlashAttention, whose IO complexity is smaller than FlashAttention by a factor proportional to the sparsity.

Given inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ and a mask matrix $\tilde{\mathbf{M}}\in\{0,1\}^{N\times N}$, we want to compute:

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}\odot\vmathbb{1}_{\tilde{\mathbf{M}}})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

where $(\mathbf{S}\odot\vmathbb{1}_{\tilde{\mathbf{M}}})_{\mathrm{kl}}=\mathbf{S}_{\mathrm{kl}}$ if $\tilde{\mathbf{M}}_{\mathrm{kl}}=1$ and $-\infty$ if $\mathbf{M}_{\mathrm{kl}}=0$. We require $\tilde{\mathbf{M}}$ to have block form: for some block sizes $B_{r},B_{c}$, for all $k,l$, $\tilde{\mathbf{M}}_{k,l}=\mathbf{M}_{\mathrm{ij}}$ with $i=\lfloor k/B_{r}\rfloor,j=\lfloor l/B_{c}\rfloor$ for some $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$.

Given a predefined block sparsity mask $\mathbf{M}\in\{0,1\}^{N/B_{r}\times N/B_{c}}$ we can easily adapt [Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") to only compute the nonzero blocks of the attention matrix. The algorithm is identical to [Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), except we skip zero blocks. We reproduce the algorithm description in [Algorithm 5](#alg5 "In D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") in [Appendix B](#A2 "Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

We also analyze the IO complexity of block-sparse FlashAttention.

###### Proposition 4.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d\leq M\leq \mathrm{Nd}$. Block-sparse FlashAttention ([Algorithm 5](#alg5 "In D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) requires $\Theta(\mathrm{Nd}+N^{2}d^{2}M^{-1}s)$ HBM accesses where $s$ is the fraction of nonzero blocks in the block-sparsity mask.

We see that applying block-sparsity yields a direct improvement by the sparsity to the larger term in the IO complexity. For large sequence lengths $N$, $s$ is often set to $N^{-1/2}$ [Child19] or $N^{-1}\log N$ [Zaheer20, Beltag20, Daoa22], resulting in $\Theta(N\sqrt{N})$ or $\Theta(N\log N)$ IO complexity. For downstream experiments, we use the fixed butterfly sparsity pattern [Daoa22], which has been shown to be able to approximate arbitrary sparsity [Dao20].

In [Fig. 2](#figure-02) (right), we validate that as the sparsity increases, the runtime of block-sparse FlashAttention improves proportionally. On the LRA benchmark, block-sparse FlashAttention achieves 2.8$\times$ speedup, while performing on par with standard attention ([Section 4](#S4 "4 Experiments ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")).

## 4 Experiments

We evaluate the impact of using FlashAttention to train Transformer models. We validate two claims about training time and model accuracy, and report attention runtime and memory benchmarks.

-   •

    Training Speed. FlashAttention outperforms the MLPerf 1.1 [Mattso20] speed record for BERT by 15%, and speeds up GPT-2 up to 3$\times$ over HuggingFace [Wolf20] and $1.8\times$ over Megatron [Shoeyb19] over standard Transformers. FlashAttention speeds up the long-range arena (LRA) benchmark 2.4$\times$.

-   •

    Quality. FlashAttention scales Transformers to longer sequences, yielding higher quality. FlashAttention trains GPT-2 with context length 4K faster than Megatron trains GPT-2 with context length 1K, while achieving 0.7 better perplexity. Modeling longer sequences yields 6.4 points of lift on two long-document classification tasks. Finally, FlashAttention yields the first Transformer that can achieve better-than-random performance on the challenging Path-X task (sequence length 16K), and block-sparse FlashAttention yields the first sequence model that we know of that can achieve better-than-random performance on Path-256 (sequence length 64K).

-   •

    Benchmarking Attention. We measure the runtime and memory performance of FlashAttention and block-sparse FlashAttention based on sequence length. We confirm that the memory footprint of FlashAttention scales linearly with seq. length and is up to 3$\times$ faster than standard attention for common seq. lengths (up to 2K). We confirm that runtime of block-sparse FlashAttention scales linearly in seq. length and is faster than all existing approximate attention baselines.

Additional experiment details are in [Appendix E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

### 4.1 Faster Models with FlashAttention

##### BERT.

FlashAttention yields the fastest single-node BERT training speed that we know of. We train a BERT-large [Devlin19] model with FlashAttention on Wikipedia. [Table 1](#table-01) compares our training time to the implementation from Nvidia that set the training speed record for MLPerf 1.1 [Mattso20]. Our implementation is 15% faster.

<span id="table-01"></span>

![Original paper Table 1](../../papers/flashattention/table-01.png)

**Table 1.** Training time of BERT-large, starting from the same initialization provided by the MLPerf benchmark, to reach the target accuracy of 72.0% on masked language modeling. Averaged over 10 runs on 8$\times$A100 GPUs.

##### GPT-2.

FlashAttention yields faster training times for GPT-2 [Radfor19] on the large OpenWebtext dataset [Gokasl19] than the widely used HuggingFace [Wolf20] and Megatron-LM [Shoeyb19] implementations. [Table 2](#table-02) shows up to 3$\times$ end-to-end speedup compared to Huggingface and 1.7$\times$ speedup compared to Megatron-LM. FlashAttention achieves the same perplexity as the other two implementations, as we do not change the model definition. [Appendix E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") includes plots of the validation perplexity throughout training, confirming that FlashAttention is as numerically stable as the baselines and produces the same training / validation curves.

<span id="table-02"></span>

![Original paper Table 2](../../papers/flashattention/table-02.png)

**Table 2.** GPT-2 small and medium using FlashAttention achieve up to 3$\times$ speed up compared to Huggingface implementation and up to 1.7$\times$ compared to Megatron-LM. Training time reported on 8$\times$A100s GPUs.

##### Long-range Arena.

We compare vanilla Transformer (with either standard implementation or FlashAttention) on the long-range arena (LRA [Tay20]) benchmark. We measure accuracy, throughput, and training time of all models. Each task has a different sequence length varying between 1024 and 4096. We follow the implementation and experimental setting in [Tay20]and [Xiong21]. [+3] [Table 3](#table-03) shows that FlashAttention achieves up 2.4$\times$ speed-up compared to standard attention. Block-sparse FlashAttention is faster than all of the approximate attention methods that we have tested.

<span id="table-03"></span>

![Original paper Table 3](../../papers/flashattention/table-03.png)

**Table 3.** The performance of standard attention, FlashAttention, block-sparse FlashAttention, and approximate attention baselines on the Long-Range-Arena benchmarks.

### 4.2 Better Models with Longer Sequences

##### Language Modeling with Long Context.

The runtime and memory-efficiency of FlashAttention allow us to increase the context length of GPT-2 by 4$\times$ while still running faster than the optimized implementation from Megatron-LM. [Table 4](#table-04) shows that that GPT-2 with FlashAttention and context length 4K is still 30% faster than GPT-2 from Megatron with context length 1K, while achieving 0.7 better perplexity.

<span id="table-04"></span>

![Original paper Table 4](../../papers/flashattention/table-04.png)

**Table 4.** GPT-2 small with FlashAttention, with 4$\times$ larger context length compared to Megatron-LM, is still 30% faster while achieving 0.7 better perplexity. Training time on 8$\times$A100 GPUs is reported.

##### Long Document Classification.

Training Transformers with longer sequences with FlashAttention improves performance on the MIMIC-III [Johnsa16] and ECtHR [Chalki19, Chalki21] datasets. MIMIC-III contains intensive care unit patient discharge summaries, each annotated with multiple labels. ECtHR contains legal cases from the European Court of Human Rights, each of which is mapped to articles of the Convention of Human Rights that were allegedly violaged. Both of these datasets contain very long text documents; the average number of tokens in MIMIC is 2,395 tokens, and the longest document contains 14,562 tokens, while the average and longest numbers in ECtHR are 2,197 and 49,392, respectively. We evaluate lift from increasing the sequence length of a pretrained RoBERTa model [Liua19] (we repeat the positional embeddings, as in [Beltag20]).

[Table 5](#table-05) shows that sequence length 16K outperforms length 512 by 4.3 points on MIMIC, and that length 8K outperforms length 512 by 8.5 points on ECtHR. The discrepancies may be due to subtle distribution shifts: MIMIC-III contains specialized medical text and thus may be more susceptible to a distribution shift in the document length, whereas ECtHR contains general language.

<span id="table-05"></span>

![Original paper Table 5](../../papers/flashattention/table-05.png)

**Table 5.** Long Document performance (micro $F_{1}$) at different sequence lengths using FlashAttention.

<span id="table-06"></span>

![Original paper Table 6](../../papers/flashattention/table-06.png)

**Table 6.** We report the first Transformer model that can achieve non-random performance on Path-X and Path-256.

##### Path-X and Path-256.

The Path-X and Path-256 benchmarks are challenging tasks from the long-range arena benchmark designed to test long context. The task is to classify whether two points in a black and white 128$\times$128 (or 256$\times$256) image have a path connecting them, and the images are fed to the transformer one pixel at a time. In prior work, all transformer models have either run out of memory, or only achieved random performance [Tay20]. There has been a search for alternative architectures that can model such long context [Ref22]. We present here the first result of Transformer models being able to solve Path-X and Path-256 ([Table 6](#table-06)). We pretrain a transformer on Path-64, and then transfer to Path-X by spatially interpolating the positional embeddings. FlashAttention achieves 61.4 accuracy on Path-X. Additionally, block-sparse FlashAttention enables the Transformers to scale to sequence length 64K, achieving 63.1 accuracy [+4] on Path-256.

### 4.3 Benchmarking Attention

<span id="figure-03"></span>

![Refer to caption](../../papers/flashattention/figure-03.png)

**Figure 3.** Left: runtime of forward pass + backward pass. Right: attention memory usage.

We vary sequence length and measure runtime and memory usage of FlashAttention and block-sparse FlashAttention against various attention baselines on one A100 GPU with 40 GB HBM, with dropout and a padding mask. We compare against reference implementations for exact attention, approximate attention, and sparse attention. We report a subset of baselines in the main body; Appendix [E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") contains more baselines and full details.

##### Runtime.

[Figure 3](#figure-03) (left) reports the runtime in milliseconds of the forward + backward pass of FlashAttention and block-sparse FlashAttention compared to the baselines in exact, approximate, and sparse attention (exact numbers in Appendix [E](#A5 "Appendix E Full Experimental Results ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). Runtime grows quadratically with sequence length, but FlashAttention runs significantly faster than exact attention baselines, up to 3$\times$ faster than the PyTorch implementation. The runtimes of many approximate/sparse attention mechanisms grow linearly with sequence length, but FlashAttention still runs faster than approximate and sparse attention for short sequences due to fewer memory accesses. The approximate attention runtimes begin to cross over with FlashAttention at sequences between 512 and 1024. On the other hand, block-sparse FlashAttention is faster than all implementations of exact, sparse, and approximate attention that we know of, across all sequence lengths.

##### Memory Footprint.

[Figure 3](#figure-03) (right) shows the memory footprint of FlashAttention and block-sparse FlashAttention compared to various exact, approximate, and sparse attention baselines. FlashAttention and block-sparse FlashAttention have the same memory footprint, which grows linearly with sequence length. FlashAttention is up to 20$\times$ more memory efficient than exact attention baselines, and is more memory-efficient than the approximate attention baselines. All other algorithms except for Linformer run out of memory on an A100 GPU before 64K, and FlashAttention is still 2$\times$ more efficient than Linformer.

## 5 Limitations and Future Directions

We discuss limitations of our approach and future directions. Related work is given in [Appendix A](#A1 "Appendix A Related Work ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

Compiling to CUDA. Our current approach to building IO-aware implementations of attention requires writing a new CUDA kernel for each new attention implementation. This requires writing the attention algorithm in a considerably lower-level language than PyTorch, and requires significant engineering effort. Implementations may also not be transferrable across GPU architectures. These limitations suggest the need for a method that supports writing attention algorithms in a high-level language (e.g., PyTorch), and compiling to IO-aware implementations in CUDA—similar to efforts such as Halide in image processing [RaganK13].

IO-Aware Deep Learning. We believe that the IO-aware approach can extend beyond attention. Attention is the most memory-intensive computation in Transformers, but every layer in a deep network touches GPU HBM. We hope our work inspires IO-aware implementations of additional modules. We discuss these potential extensions in [Appendix D](#A4 "Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

Multi-GPU IO-Aware Methods. Our IO-aware implementation of attention is optimal within constants for computing attention on a single GPU. However, the attention computation may be parallelizable across multiple GPUs [Recht13]. Using multiple GPUs adds an additional layer to IO analysis—accounting for data transfer between GPUs. We hope our work inspires future work in this direction.

#### Acknowledgments

Our implementation uses Apex’s FMHA code ([https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha)) as a starting point. We thank Young-Jun Ko for the in-depth explanation of his FMHA implementation and for his thoughtful answers to our questions about CUDA. We thank Sabri Eyuboglu, Megan Leszczynski, Laurel Orr, Yuhuai Wu, Beidi Chen, and Xun Huang for their constructive feedback and suggestions on early drafts of the paper. We thank Markus Rabe and Charles Staats for helpful discussion of their attention algorithm.

We gratefully acknowledge the support of NIH under No. U54EB020405 (Mobilize), NSF under Nos. CCF1763315 (Beyond Sparsity), CCF1563078 (Volume to Velocity), and 1937301 (RTML); ARL under No. W911NF-21-2-0251 (Interactive Human-AI Teaming); ONR under No. N000141712266 (Unifying Weak Supervision); ONR N00014-20-1-2480: Understanding and Applying Non-Euclidean Geometry in Machine Learning; N000142012275 (NEPTUNE); NXP, Xilinx, LETI-CEA, Intel, IBM, Microsoft, NEC, Toshiba, TSMC, ARM, Hitachi, BASF, Accenture, Ericsson, Qualcomm, Analog Devices, Google Cloud, Salesforce, Total, the HAI-GCP & HAI-Azure Cloud Credits for Research program, the Stanford Data Science Initiative (SDSI), Department of Defense (DoD) through the National Defense Science and Engineering Graduate Fellowship (NDSEG) Program, and members of the Stanford DAWN project: Facebook, Google, and VMWare. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views, policies, or endorsements, either expressed or implied, of NIH, ONR, or the U.S. Government. Atri Rudra’s research is supported by NSF grant CCF-1763481.

## Appendix A Related Work

IO-Aware Runtime Optimization. The broad concept of optimizing for reading and writing to fast/slow memory has a long history in computer science and has been known by many names. We draw the most direct connection to the literature of analyzing I/O complexity in this work [Vitter88], but concepts of memory hierarchies are fundamental and has appeared in many forms, from the working set model [Dennin68], to data locality [Lama91], to the Roofline model of arithmetic intensity [Willia09], to analyses of scalability [McSher15], to standard textbook treatments of computer architecture [Patter03]. We hope that this work encourages the community to adopt these ideas in more parts of the deep learning stack.

Efficient ML Models with Structured Matrices. Matrix multiply is the core computational bottleneck of most machine learning models. To reduce the computational complexity, there have been numerous approaches to learn over a more efficient set of matrices. These matrices are called *structured matrices*, which have subquadratic ($o(n^{2})$ for dimension $n\times n$) number of parameters and runtime. Most common examples of structured matrices are sparse and low-rank matrices, along with fast transforms commonly encountered in signal processing (Fourier, Chebyshev, sine/cosine, orthogonal polynomials). There have been several more general classes of structured matrices proposed in machine learning: Toeplitz-like [Sindhw15], low-displacement rank [Kailat79], quasi-separable [Gohber99]). The butterfly pattern we use for our block-sparse attention is motivated by the fact that butterfly matrices [Parker95, Dao19] and their products have been shown to be able to express any structured matrices with almost optimal runtime and number of parameters [Ref18, Dao20]. However, even though structured matrices are efficient in theory, they have not seen wide adoption since it is hard to translate their efficiency to wall-clock speedup since dense unconstrained matrix multiply has very optimize implementation, a phenomenon known as the hardware lottery [Hooker20]. Extensions of butterfly matrices [Daoa22, Daob22] aimed to make butterfly matrices more hardware-friendly.

Sparse Training. Our block-sparse FlashAttention can be seen as a step towards making sparse model training more efficient. Sparse models have seen success in compressing models for inference (pruning) by sparsifying the weight matrices [Han16, Hana15, Sanh20, Lin17, Dong17]. For model training, the lottery tickets hypothesis [Carbin18, Frankl19, Frankl20] suggests that there are a set of small sub-networks derived from a larger dense network that performs as well as the original dense network. Out block-sparse FlashAttention can also be seen as a fixed lottery ticket in the context of attention: we fix the sparsity pattern to be the butterfly pattern through training, and observe that it performs almost as well as the (dense) FlashAttention on the Long-range Arena tasks.

Efficient Transformer. Transformer-based models have become the most widely-used architecture in natural language processing [Devlin19] and computer vision [Dosovi20, Yuan21]. However, one of their computational bottlenecks is that their time and memory scales quadratic in the sequence length. There are numerous approaches to overcome this bottleneck, including approximation with hashing (i.e., sparse) such as Reformer [Kitaev20] and Smyrf [Daras20] and with low-rank approximation such as Performer [Chorom20, Likhos20]. One can even combine sparse and low-rank approximation for better accuracy (e.g., Longformer [Beltag20], BigBird [Zaheer20], Scatterbrain [Chen21], Long-short transformer [Zhu21], Combiner [Ren21]). Other approaches include compressing along the sequence dimension to attend to multiple tokens at once [Refb19, Sukhba19, Lan20, Refa21]. One can also attend over the states from previous sequences to help lengthen the context (e.g., Transformer-XL [Dai19] and Compressive Transformer [Rae20]). We recommend the survey [Taya20] for more details.

There are several lines of work on developing other modules instead of attention to model longer context. HiPPO [Ref20] and its extensions, most notably S4 [Ref21, Ref22, Goel22] projects the history on a polynomial basis, allowing accurate reconstruction of the history through state-space models. They combine the strengths of CNNs (efficient training), RNNs (efficient inference), and continuous models (robust to change in sampling rates). LambdaNetworks [Bello21], AFT [Zhai21] and FLASH [Hua22] are other attempts at replacing attention in the context of image classification and language modeling.

## Appendix B Algorithm Details

We first derive the forward and backward passes of attention and show that they can be computed in a memory-efficient manner (requiring extra memory linear instead of quadratic in the sequence length). Though they reduce the amount of extra memory required, naively they still incur quadratic HBM accesses, resulting in slower execution speed. We describe the FlashAttention algorithm to implement both the forward and the backward passes on GPUs that reduces HBM accesses, leading to both faster runtime and smaller memory footprint.

### B.1 Memory-efficient forward pass

The main challenge in making attention memory-efficient is the softmax that couples the columns of $\mathbf{K}$ (and columns of $\mathbf{V}$). Our approach is to compute the softmax normalization constant separately to decouple the columns. This technique [Gimels18] has been used in the literature [Kitaev20, Staats21] to show that attention computation does not need quadratic *extra* memory (though the number of HBM accesses is still quadratic, resulting in slow run-time).

For simplicity, we omit here the max-shifting step during softmax. The full algorithm in [Section B.3](#A2.SS3 "B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") contains all the steps.

Recall that given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d}.
$$

We have that $S_{\mathrm{ij}}=q_{i}^\topk_{j}$ where $q_{i}$ and $k_{j}$ are the $i$-th and $j$-th columns of $\mathbf{Q}$ and $\mathbf{K}$ respectively. Define the normalization constants of softmax:

$$
L_{i}=\sum_{j}e^{q_{i}^\topk_{j}}.\tag{1}
$$

Let $v_{j}$ be the $j$-th column of $\mathbf{V}$, then the $i$-th columns of the output is

$$
o_{i}=P_{i:}\mathbf{V}=\sum_{j}P_{\mathrm{ij}}v_{j}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}v_{j}.\tag{2}
$$

We see that once $L_{i}$ is computed, we can compute $o_{i}$ without extra memory by repeatedly summing $\frac{e^{q_{i}^\topk_{j}}}{L_{i}}v_{j}$. Therefore the forward pass can be computed with $O(n)$ extra memory:

1.  1.

    Compute $L_{i}$ for all $i$ according to [Eq. 1](#A2.E1 "In B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(n)$ extra memory.

2.  2.

    Compute $o_{i}$ for all $i$ according to [Eq. 2](#A2.E2 "In B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(d)$ extra memory.

### B.2 Memory-efficient backward pass

We derive the backward pass of attention and show that it can also be computed with linear memory. [Staats21] suggests that the backward pass can be done without quadratic extra memory by applying gradient checkpointing to the memory-efficient forward pass. We instead derive the backward pass explicitly and show how it can be computed in a memory-efficient manner.

Suppose that there is a scalar loss function $\phi$, and let the output gradient be $\mathbf{\mathrm{dO}}\in\mathbb{R}^{n\times d}$ (where $\mathbf{\mathrm{dO}}$ denotes $\frac{\partial\phi}{\partial\mathbf{O}}$). We want to compute the input gradients $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{n\times d}$ (where $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ denote $\frac{\partial\phi}{\partial\mathbf{Q}},\frac{\partial\phi}{\partial\mathbf{K}},\frac{\partial\phi}{\partial\mathbf{V}}$ respectively).

The gradient $\mathbf{\mathrm{dV}}$ is easy to see. Applying reverse-mode autodiff by hand (aka the chain rule), we obtain (in matrix notation) $\mathbf{\mathrm{dV}}=\mathbf{P}^\top\mathbf{\mathrm{dO}}$. Thus:

$$
\mathrm{dv}_{j}=\sum_{i}P_{\mathrm{ij}}\mathrm{do}_{i}=\sum_{i}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}\mathrm{do}_{i}.\tag{3}
$$

Since we already computed $L_{i}$, $\mathrm{dv}_{j}$ can be computed without extra memory by repeated summing.

The gradients $\mathbf{\mathrm{dQ}}$ and $\mathbf{\mathrm{dK}}$ are a little more complicated. We go through the gradients $\mathbf{\mathrm{dP}}$ and $\mathbf{\mathrm{dS}}$ first. From [Eq. 2](#A2.E2 "In B.1 Memory-efficient forward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), we have that $\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^\top$, and so:

$$
\mathrm{dP}_{\mathrm{ij}}=\mathrm{do}_{i}^\topv_{j}.
$$

Recall that $P_{i:}=\mathrm{softmax}(S_{i:})$. Using the fact that the Jacobian of $y=\mathrm{softmax}(x)$ is $\mathrm{diag}(y)-\mathrm{yy}^\top$, we have that

$$
\mathrm{dS}_{i:}=(\mathrm{diag}(P_{i:})-P_{i:}P_{i:}^\top)\mathrm{dP}_{i:}=P_{i:}\circ \mathrm{dP}_{i:}-(P_{i:}^\topdP_{i:})P_{i:},
$$

where $\circ$ denotes pointwise multiplication.

Define

$$
D_{i}=P_{i:}^\topdP_{i:}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}\mathrm{do}_{i}^\topv_{j}=\mathrm{do}_{i}^\top\sum_{j}\frac{e^{q_{i}^{\top}k_{j}}}{L_{i}}v_{j}=\mathrm{do}_{i}^\topo_{i},\tag{4}
$$

then

$$
\mathrm{dS}_{i:}=P_{i:}\circ \mathrm{dP}_{i:}-D_{i}P_{i:}.
$$

Hence

$$
\mathrm{dS}_{\mathrm{ij}}=P_{\mathrm{ij}}\mathrm{dP}_{\mathrm{ij}}-D_{i}P_{\mathrm{ij}}=P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i}).
$$

Now we can get the gradients $\mathbf{\mathrm{dQ}}$ and $\mathbf{\mathrm{dK}}$. Recall that $S_{\mathrm{ij}}=q_{i}^\topk_{j}$, so

$$
\mathrm{dq}_{i}=\sum_{j}\mathrm{dS}_{\mathrm{ij}}k_{j}=\sum_{j}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})k_{j}=\sum_{j}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}(\mathrm{do}_{i}^\topv_{j}-D_{i})k_{j}.\tag{5}
$$

Similarly,

$$
\mathrm{dk}_{j}=\sum_{i}\mathrm{dS}_{\mathrm{ij}}q_{i}=\sum_{i}P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-D_{i})q_{i}=\sum_{i}\frac{e^{q_{i}^\topk_{j}}}{L_{i}}(\mathrm{do}_{i}^\topv_{j}-D_{i})q_{i}.\tag{6}
$$

Therefore the backward pass can also be computed with $O(n)$ extra memory:

1.  1.

    Compute $\mathrm{dv}_{j}$ for all $j$ according to [Eq. 3](#A2.E3 "In B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(d)$ extra memory.

2.  2.

    Compute $D_{i}$ for all $i$ according to [Eq. 4](#A2.E4 "In B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(n)$ extra memory.

3.  3.

    Compute $\mathrm{dq}_{i}$ for all $i$ according to [Eq. 5](#A2.E5 "In B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(d)$ extra memory.

4.  4.

    Compute $\mathrm{dk}_{j}$ for all $j$ according to [Eq. 6](#A2.E6 "In B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), which takes $O(d)$ extra memory.

### B.3 FlashAttention: Forward Pass

We describe the full details of FlashAttention forward pass. Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, we want to compute the attention output $\mathbf{O}\in\mathbb{R}^{N\times d}$:

$$
\mathbf{S}=\tau\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{S}^{\mathrm{masked}}=\mathrm{mask}(S)\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S}^{\mathrm{masked}})\in\mathbb{R}^{N\times N},
$$

$$
\mathbf{P}^{\mathrm{dropped}}=\mathrm{dropout}(\mathbf{P},p_{\mathrm{drop}}),\quad\mathbf{O}=\mathbf{P}^{\mathrm{dropped}}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

where $\tau\in\mathbb{R}$ is some softmax scaling (typically $\frac{1}{\sqrt{d}}$), mask is some masking function that sets some entries of the input to $-\infty$ and keep other entries the same (e.g., key padding mask when sequences in the batch don’t have the same lengths and are padded), and $\mathrm{dropout}(x,p)$ applies dropout to $x$ elementwise (i.e., output $\frac{x}{1-p}$ with probability $1-p$ and output 0 with probability $p$ for each element $x$).

The full algorithm is in [Algorithm 2](#alg2 "In B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). We save the output $\mathbf{O}$, the softmax statistics $\ell$ and $m$, and the pseudo-random number generator state ${\cal R}$ for the backward pass.

<span id="alg2"></span>

**Algorithm 2: FlashAttention Forward Pass**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant $\tau\in\mathbb{R}$, masking function mask, dropout probability $p_{\mathrm{drop}}$.
- Initialize the pseudo-random number generator state ${\cal R}$ and save to HBM.
- Set block sizes $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- Initialize $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$ in HBM.
- Divide $\mathbf{Q}$ into $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ blocks $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ of size $B_{r}\times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ blocks $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, of size $B_{c}\times d$ each.
- Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ of size $B_{r}\times d$ each, divide $\ell$ into $T_{r}$ blocks $\ell_{i},\dots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\dots,m_{T_{r}}$ of size $B_{r}$ each.
- **For** $1\leq j\leq T_{c}$ **do:**
  - Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.
  - **For** $1\leq i\leq T_{r}$ **do:**
    - Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.
    - On chip, compute $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - On chip, compute $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
    - On chip, compute $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (pointwise), $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
    - On chip, compute $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
    - On chip, compute $\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$.
    - Write $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ to HBM.
    - Write $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ to HBM.
- **Return:** $\mathbf{O},\ell,m,{\cal R}$.

### B.4 FlashAttention: Backward Pass

We describe the full details of FlashAttention backward pass. Given input sequences $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$, the output $\mathbf{O}\in\mathbb{R}^{N\times d}$, and the output gradient $\mathbf{\mathrm{dO}}$, we want to compute the input gradients $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$.

We first describe the standard attention backward pass in [Algorithm 3](#alg3 "In B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") for completeness.

<span id="alg3"></span>

**Algorithm 3: Standard Attention Backward Pass**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$, $\mathbf{P}\in\mathbb{R}^{N\times N}$ in HBM.
- Load $\mathbf{P},\mathbf{\mathrm{dO}}$ by blocks from HBM, compute $\mathbf{\mathrm{dV}}=\mathbf{P}^{\top}\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$, write $\mathbf{\mathrm{dV}}$ to HBM.
- Load $\mathbf{\mathrm{dO}},\mathbf{V}$ by blocks from HBM, compute $\mathbf{\mathrm{dP}}=\mathbf{\mathrm{dO}}\mathbf{V}^{\top}\in\mathbb{R}^{N\times N}$, write $\mathbf{\mathrm{dP}}$ to HBM.
- Read $\mathbf{P},\mathbf{\mathrm{dP}}$ from HBM, compute $\mathbf{\mathrm{dS}}\in\mathbb{R}^{N\times N}$ where $\mathrm{dS}_{\mathrm{ij}}=P_{\mathrm{ij}}(\mathrm{dP}_{\mathrm{ij}}-\sum_{l}P_{\mathrm{il}}\mathrm{dP}_{\mathrm{il}})$, write $\mathbf{\mathrm{dS}}$ to HBM.
- Load $\mathbf{\mathrm{dS}}$ and $\mathbf{K}$ by blocks from HBM, compute $\mathbf{\mathrm{dQ}}=\mathbf{\mathrm{dS}}\mathbf{K}$, write $\mathbf{\mathrm{dQ}}$ to HBM.
- Load $\mathbf{\mathrm{dS}}$ and $\mathbf{Q}$ by blocks from HBM, compute $\mathbf{\mathrm{dK}}=\mathbf{\mathrm{dS}}^{\top}\mathbf{Q}$, write $\mathbf{\mathrm{dK}}$ to HBM.
- **Return:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$.

We now make two observations about FlashAttention backward pass:

1.  1.

    We do not need to store the dropout mask of size $O(N^{2})$ from the forward pass. Instead, we can save the pseudo-random number generator states from the forward pass and re-generate the dropout mask in the backward pass. This allows us to only use $O(N)$ extra memory.

2.  2.

    When computing the softmax gradient, we use [Eq. 4](#A2.E4 "In B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") to compute $D_{i}=P_{i:}^{\top}\mathrm{dP}_{i:}$ without reducing over $P_{i:}$ and $\mathrm{dP}_{i:}$ of size $N$ (they might not fit into SRAM). Instead we can rewrite $D_{i}=\mathrm{do}_{i}^{\top}o_{i}$ and compute the dot product between vectors of size $d$.

The full FlashAttention backward pass algorithm is in [Algorithm 4](#alg4 "In B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). Conceptually it is just a block version of the derivation in [Section B.2](#A2.SS2 "B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

<span id="alg4"></span>

**Algorithm 4: FlashAttention Backward Pass**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{O},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$ in HBM, vectors $\ell,m\in\mathbb{R}^{N}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant $\tau\in\mathbb{R}$, masking function mask, dropout probability $p_{\mathrm{drop}}$, pseudo-random number generator state ${\cal R}$ from the forward pass.
- Set the pseudo-random number generator state to ${\cal R}$.
- Set block sizes $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$.
- Divide $\mathbf{Q}$ into $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ blocks $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ of size $B_{r}\times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ blocks $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, of size $B_{c}\times d$ each.
- Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ of size $B_{r}\times d$ each, divide $\mathbf{\mathrm{dO}}$ into $T_{r}$ blocks $\mathbf{\mathrm{dO}}_{i},\dots,\mathbf{\mathrm{dO}}_{T_{r}}$ of size $B_{r}\times d$ each, divide $\ell$ into $T_{r}$ blocks $\ell_{i},\dots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\dots,m_{T_{r}}$ of size $B_{r}$ each.
- Initialize $\mathbf{\mathrm{dQ}}=(0)_{N\times d}$ in HBM and divide it into $T_{r}$ blocks $\mathbf{\mathrm{dQ}}_{1},\dots,\mathbf{\mathrm{dQ}}_{T_{r}}$ of size $B_{r}\times d$ each. Initialize $\mathbf{\mathrm{dK}}=(0)_{N\times d},\mathbf{\mathrm{dV}}=(0)_{N\times d}$ in HBM and divide $\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$ in to $T_{c}$ blocks $\mathbf{\mathrm{dK}}_{1},\dots,\mathbf{\mathrm{dK}}_{T_{c}}$ and $\mathbf{\mathrm{dV}}_{1},\dots,\mathbf{\mathrm{dV}}_{T_{c}}$, of size $B_{c}\times d$ each.
- **For** $1\leq j\leq T_{c}$ **do:**
  - Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.
  - Initialize $\tilde{\mathbf{\mathrm{dK}}}_{j}=(0)_{B_{c}\times d},\tilde{\mathbf{\mathrm{dV}}}_{j}=(0)_{B_{c}\times d}$ on SRAM.
  - **For** $1\leq i\leq T_{r}$ **do:**
    - Load $\mathbf{Q}_{i},\mathbf{O}_{i},\mathbf{\mathrm{dO}}_{i},\mathbf{\mathrm{dQ}}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.
    - On chip, compute $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - On chip, compute $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
    - On chip, compute $\mathbf{P}_{\mathrm{ij}}=\mathrm{diag}(l_{i})^{-1}\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-m_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - On chip, compute dropout mask $\mathbf{Z}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}\times B_{c}}$ where each entry has value $\frac{1}{1-p_{\mathrm{drop}}}$ with probability $1-p_{\mathrm{drop}}$ and value 0 with probability $p_{\mathrm{drop}}$.
    - On chip, compute $\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{P}_{\mathrm{ij}}\circ\mathbf{Z}_{\mathrm{ij}}$ (pointwise multiply).
    - On chip, compute $\tilde{\mathbf{\mathrm{dV}}_{j}}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}+(\mathbf{P}_{\mathrm{ij}}^{\mathrm{dropped}})^{\top}\mathbf{\mathrm{dO}}_{i}\in\mathbb{R}^{B_{c}\times d}$.
    - On chip, compute $\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathbf{\mathrm{dO}}_{i}\mathbf{V}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - On chip, compute $\mathbf{\mathrm{dP}}_{\mathrm{ij}}=\mathbf{\mathrm{dP}}_{\mathrm{ij}}^{\mathrm{dropped}}\circ\mathbf{Z}_{\mathrm{ij}}$ (pointwise multiply).
    - On chip, compute $D_{i}=\mathrm{rowsum}(\mathbf{\mathrm{dO}}_{i}\circ\mathbf{O}_{i})\in\mathbb{R}^{B_{r}}$.
    - On chip, compute $\mathbf{\mathrm{dS}}_{\mathrm{ij}}=\mathbf{P}_{\mathrm{ij}}\circ(\mathbf{\mathrm{dP}}_{\mathrm{ij}}-D_{i})\in\mathbb{R}^{B_{r}\times B_{c}}$.
    - Write $\mathbf{\mathrm{dQ}}_{i}\leftarrow\mathbf{\mathrm{dQ}}_{i}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}\mathbf{K}_{j}\in\mathbb{R}^{B_{r}\times d}$ to HBM.
    - On chip, compute $\tilde{\mathbf{\mathrm{dK}}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}}_{j}+\tau\mathbf{\mathrm{dS}}_{\mathrm{ij}}^{\top}\mathbf{Q}_{i}\in\mathbb{R}^{B_{c}\times d}$.
  - Write $\mathbf{\mathrm{dK}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dK}}_{j}},\mathbf{\mathrm{dV}}_{j}\leftarrow\tilde{\mathbf{\mathrm{dV}}_{j}}$ to HBM.
- **Return:** $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}$.

We see that similar to the forward pass, the backward pass performs $O(N^{2})$ FLOPs and only requires $O(N)$ extra memory beyond inputs, output, output gradient, and input gradients.

We analyze the IO-complexity of the backward pass, similar to the forward pass ([Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")).

###### Theorem 5.

Let $N$ be the sequence length, $d$ be the head dimension, and $M$ be size of SRAM with $d\leq M\leq \mathrm{Nd}$. Standard attention ([Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") ) backward pass requires $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses, while FlashAttention backward pass ([Algorithm 4](#alg4 "In B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) requires $\Theta(N^{2}d^{2}M^{-1})$ HBM accesses.

The proof is in [Appendix C](#A3 "Appendix C Proofs ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

### B.5 Comparison with [Staats21]

We describe here some similarities and differences between our FlashAttention algorithm and the algorithm of [Staats21].

Conceptually, both FlashAttention and [Staats21] operate on blocks of the attention matrix using the well-established technique of tiling (or softmax scaling) [Gimels18, Kitaev20]. To reduce the memory footprint, both methods avoid storing the large attention matrix in the forward pass and recompute it in the backward pass.

The first major difference is that [Staats21] focuses on the reducing the total memory footprint (maximum amount of GPU memory required) while FlashAttention focuses on reducing memory accesses (the number of memory reads/writes). As mentioned in [Section 2](#S2 "2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), the amount of memory access is the primary determining factor of runtime. Reducing memory accesses also necessarily reduces the total amount of memory required (e.g., if an operation incurs $A$ memory accesses, then its total memory requirement is at most $A$). As a result, FlashAttention is faster than standard attention (2-4$\times$) while [Staats21] is around the same speed or slightly slower than standard attention. In terms of total memory required, both methods offer substantial memory saving.

The second difference between the two methods is the way information is summarized from each block to pass to the next block. [Staats21] summarizes each block with its temporary output along with the softmax normalization statistics. At the end of the forward pass, the temporary outputs of all the blocks are combined using the statistics to produce the final output. FlashAttention instead incrementally updates the output ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [12](#alg1.l12 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) after processing each block, so only one copy of the output is needed (instead of $K$ copies for $K$ blocks). This means that FlashAttention has smaller total memory requirement compared to [Staats21].

The final major difference is the way the backward pass is computed. [Staats21] uses gradient checkpointing to recompute the attention matrix and the temporary output of each block. FlashAttention instead simplifies the backward pass analytically ([Sections B.2](#A2.SS2 "B.2 Memory-efficient backward pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") and [B.4](#A2.SS4 "B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). It only recomputes the attention matrix and does not recompute the temporary output of each block. This reduces the memory requirement for the backward pass and yields speedup.

## Appendix C Proofs

###### Proof of [Theorem 1](#Thmtheorem1 "Theorem 1. ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

We first count the number of FLOPs and extra memory required.

The dominating FLOPs are from matrix multiplication. In the inner loop, ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [9](#alg1.l9 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")), we compute $\mathbf{Q}_{i}\mathbf{K}_{j}^{\top}\in\mathbb{R}^{B_{r}\times B_{c}}$ for $\mathbf{Q}_{i}\in\mathbb{R}^{B_{r}\times d}$ and $\mathbf{K}_{j}\in\mathbb{R}^{B_{c}\times d}$, which takes $O(B_{r}B_{c}d)$ FLOPs. We also compute ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [12](#alg1.l12 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) $\tilde{\mathbf{P}}_{\mathrm{ij}}\mathbf{V}_{j}\in\mathbb{R}^{B_{r}\times d}$ for $\tilde{\mathbf{P}}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}\times B_{c}}$ and $\mathbf{V}_{j}\in\mathbb{R}^{B_{c}\times d}$, which takes $O(B_{r}B_{c}d)$ FLOPs. We execute the inner loops $T_{c}T_{r}=\left\lceil\frac{N}{B_{c}}\right\rceil\left\lceil\frac{N}{B_{r}}\right\rceil$ times. Therefore the total number of FLOPs is

$$
O\left(\frac{N^{2}}{B_{c}B_{r}}B_{r}B_{c}d\right)=O(N^{2}d).
$$

In terms of extra memory required, we see that we need $O(N)$ memory to store the statistics $(\ell,m)$.

We now prove the algorithm’s correctness by induction on $j$ for $0\leq j\leq T_{c}$. Let $\mathbf{K}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ be the first $\mathrm{jB}_{c}$ rows of $\mathbf{K}$, and similarly $\mathbf{V}_{:j}\in\mathbb{R}^{\mathrm{jB}_{c}\times d}$ the the first $\mathrm{jB}_{c}$ rows of $\mathbf{V}$. Let $\mathbf{S}_{:,:j}=\mathbf{Q}\mathbf{K}_{:j}^{\top}\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$, and $\mathbf{P}_{:,:j}=\mathrm{softmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N\times \mathrm{jB}_{c}}$ (softmax applied row-wise). Let $m^{j},\ell^{(j)},\mathbf{O}^{(j)}$ be the values of $m,\ell,\mathbf{O}$ in HBM after the $j$-th iteration of the outer loop ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [5](#alg1.l5 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). (Note that these values of $m,\ell,\mathbf{O}$ are updated after each iteration of the outer loop.) We want to show that after the $j$-th iteration of the outer loop, we have computed in HBM:

$$
m^{(j)}=\mathrm{rowmax}(\mathbf{S}_{:,:j})\in\mathbb{R}^{N},\quad\ell^{(j)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j}-m^{(j)}))\in\mathbb{R}^{N},\quad\mathbf{O}^{(j)}=\mathbf{P}_{:,:j}\mathbf{V}_{:j}\in\mathbb{R}^{N\times d}.
$$

Based on our initialization ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [2](#alg1.l2 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")), this claim is true for $j=0$ (i.e., before the any iteration of the outer loop is executed). Suppose that the claim holds for some $j=0,\dots,T_{c}-1$. We want to show that the claim also holds for $j+1$. Indeed, when we update the statistics in the inner loop ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [10](#alg1.l10 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")) on the $(j+1)$-th iteration of the outer loop, we update $m^{(j+1)}=\max(m^{(j)},\tilde{m})$ where $\tilde{m}\in\mathbb{R}^{N}$ is the row-max of $\mathbf{S}_{:,j:j+1}$, the slice of $\mathbf{S}$ from column $\mathrm{jB}_{c}$ to column $(j+1)B_{c}-1$. This implies that

$$
m^{(j+1)}=\mathrm{rowmax}(\mathbf{S}_{:,:j+1})\in\mathbb{R}^{N}.
$$

Similarly, we update

$$
\ell^{(j+1)}=e^{m^{(j)}-m^{(j+1)}}\ell^{(j)}+e^{\tilde{m}-m^{(j+1)}}\tilde{\ell},
$$

where $\tilde{\ell}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,j:j+1}-\tilde{m}))\in\mathbb{R}^{N}$. By the same algebraic manipulation in [Section 3.1](#S3.SS1 "3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), we obtain:

$$
\ell^{(j+1)}=\mathrm{rowsum}(\exp(\mathbf{S}_{:,:j+1}-m^{(j+1)}))\in\mathbb{R}^{N}.
$$

Let $\mathbf{V}_{j:j+1}$ be the slice of $\mathbf{V}$ from column $\mathrm{jB}_{c}$ to column $(j+1)B_{c}-1$, we also update:

$$
\mathbf{O}^{(j+1)}\qquad =\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{O}^{(j)}+e^{\tilde{m}-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1}-\tilde{m})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathbf{P}_{:,:j}\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\mathrm{diag}(\ell^{(j)})e^{m^{(j)}-m^{(j+1)}}\mathrm{diag}(\ell^{(j)})\exp(\mathbf{S}_{:,:j}-m^{(j)})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(e^{-m^{(j+1)}}\exp(\mathbf{S}_{:,:j})\mathbf{V}_{:j}+e^{-m^{(j+1)}}\exp(\mathbf{S}_{j:j+1})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}(\exp(\mathbf{S}_{:,:j}-m^{(j+1)})\mathbf{V}_{:j}+\exp(\mathbf{S}_{j:j+1}-m^{(j+1)})\mathbf{V}_{j:j+1})
$$

$$
=\mathrm{diag}(\ell^{(j+1)})^{-1}\left(\exp\left(\begin{bmatrix}\mathbf{S}_{:,:j}&\mathbf{S}_{j:j+1}\end{bmatrix}-m^{(j+1)}\right)\right)\begin{bmatrix}\mathbf{V}_{:j}\\
\mathbf{V}_{j:j+1}\end{bmatrix}
$$

$$
=\mathrm{softmax}(\mathbf{S}_{:j+1})\mathbf{V}_{:j+1}.
$$

We then see that the claim is also true for $j+1$. By induction, the claim is true for all $j=0,\dots,T_{c}$.

When $j=T_{c}$, we conclude that the final value of $\mathbf{O}$ in HBM is $\mathrm{softmax}(\mathbf{S})\mathbf{V}=\mathrm{softmax}(\mathbf{Q}\mathbf{K}^{\top})\mathbf{V}$.

∎

###### Proof of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

We first analyze the IO complexity of standard attention implementation. The inputs $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ reside in HBM, and the at the end of the algorithm the output $\mathbf{O}\in\mathbb{R}^{N\times d}$ is written to HBM.

In the first step of computing the matrix multiply $\mathbf{S}=\mathbf{Q}\mathbf{K}^{\top}$, the inputs $\mathbf{Q},\mathbf{K}$ are read from HBM and the output $\mathbf{S}\in\mathbb{R}^{N\times N}$ is written to HBM ([Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")  line [1](#alg0.l1 "In Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). This incurs $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses.

In the second step of computing $\mathbf{P}=\mathrm{softmax}(\mathbf{S})$, the input $\mathbf{S}$ is read from HBM and the output $\mathbf{P}$ is written to HBM ([Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")  line [2](#alg0.l2 "In Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). This incurs $\Theta(N^{2})$ HBM accesses.

In the last step of computing $\mathbf{O}=\mathbf{P}\mathbf{V}$, the inputs $\mathbf{P},\mathbf{V}$ are read from global memory and the output $\mathbf{O}$ is written to HBM ([Algorithm](#alg0 "In 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")  line [3](#alg0.l3 "In Algorithm 0 ‣ 2.2 Standard Attention Implementation ‣ 2 Background ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). This incurs $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses.

Overall, standard attention implementation requires $\Theta(\mathrm{Nd}+N^{2})$ global memory accesses.

We now analyze the IO complexity of streaming attention.

Following [Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [6](#alg1.l6 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). We make $T_{c}$ passes over $\mathbf{Q}$ and $\mathbf{O}$, each pass loading all of $\mathbf{Q}$ and all of $\mathbf{O}$ to HBM ([Algorithm 1](#alg1 "In 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness") line [8](#alg1.l8 "In Algorithm 1 ‣ 3.1 An Efficient Attention Algorithm With Tiling and Recomputation ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). Therefore the number of HBM accesses is $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$.

We derive the conditions on the block sizes $B_{c}$ and $B_{r}$. We need the blocks $\mathbf{K}_{j}$ and $\mathbf{V}_{j}$ of size $B_{c}\times d$ to fit into on-chip memory, which translates to:

$$
B_{c}d=O(M)\Leftrightarrow B_{c}=O\left(\frac{M}{d}\right).
$$

Similarly, we need the blocks $\mathbf{Q}_{i},\mathbf{O}_{i}$ of size $B_{r}\times d$ to fit into on-chip memory, which translates to:

$$
B_{r}d=O(M)\Leftrightarrow B_{r}=O\left(\frac{M}{d}\right).
$$

Finally, we need the block $\mathbf{S}_{\mathrm{ij}}$ of size $B_{r}\times B_{c}$ to fit into on-chip memory, which translates to:

$$
B_{r}B_{c}=O(M).
$$

We therefore set:

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},\frac{M}{B_{c}}\right)\right)=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

We then have:

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

As a result, the number of HBM accesses is:

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

###### Proof of [Proposition 3](#Thmtheorem3 "Proposition 3. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

For contradiction, suppose that there exists an algorithm that computes exact attention where the number for HBM access for all $M\in[d,\mathrm{Nd}]$ is

$$
o\left(\frac{N^{2}d^{2}}{M}\right).
$$

In the regime of $M=\Theta(\mathrm{Nd})$, this results in the number of HBM accesses:

$$
o\left(\frac{N^{2}d^{2}}{\mathrm{Nd}}\right)=o(\mathrm{Nd}).
$$

However, the input to attention (matrices $\mathbf{Q},\mathbf{K},\mathbf{V}$) and the output $\mathbf{O}$ have size $\mathrm{Nd}$ and they start out being in HBM, so if the algorithm computes exact attention it must incur at least $\Omega(\mathrm{Nd})$ HBM accesses. This is a contradiction. ∎

###### Proof of [Theorem 5](#Thmtheorem5 "Theorem 5. ‣ B.4 FlashAttention: Backward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

The IO complexity of the attention backward is very similar to the IO complexity of the attention forward ([Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness")). Here we provide a sketch of the proof.

We first analyze the IO complexity of standard attention backward pass. The inputs $\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{\mathrm{dO}}\in\mathbb{R}^{N\times d}$ reside in HBM, and the at the end of the algorithm the outputs $\mathbf{\mathrm{dQ}},\mathbf{\mathrm{dK}},\mathbf{\mathrm{dV}}\in\mathbb{R}^{N\times d}$ are written to HBM.

At each step of the standard attention backward pass, one needs to load inputs of size $\mathrm{Nd}$ or $N^{2}$ from HBM, and needs to write the outputs of size $N^{2}$ or $\mathrm{Nd}$ to HBM. This incurs $\Theta(\mathrm{Nd}+N^{2})$ HBM accesses.

We now analyze the IO complexity of FlashAttention backward pass.

Similar to [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), we see that each element of $\mathbf{K}$ and $\mathbf{V}$ is loaded from HBM once. Each element of $\mathbf{\mathrm{dK}}$ and $\mathbf{\mathrm{dV}}$ is only written to HBM once. We make $T_{c}$ passes over $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$, each pass loading all of $\mathbf{Q},\mathbf{O},\mathbf{\mathrm{dO}}$ to HBM. We also make $T_{c}$ passes over $\mathbf{\mathrm{dQ}}$, each pass reading/writing all of $\mathbf{\mathrm{dQ}}$ from/to HBM. Therefore the number of HBM accesses is $\Theta\left(\mathrm{Nd}+\mathrm{NdT}_{c}\right)=\Theta(\mathrm{NdT}_{c})$.

As in the proof of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), the constraints on the block sizes are that:

$$
B_{c}=\Theta\left(\frac{M}{d}\right),\qquad B_{r}=\Theta\left(\min\left(\frac{M}{d},d\right)\right).
$$

We then have:

$$
T_{c}=\frac{N}{B_{c}}=\Theta\left(\frac{\mathrm{Nd}}{M}\right).
$$

As a result, the number of HBM accesses is:

$$
\Theta\left(\mathrm{NdT}_{c}\right)=\Theta\left(\frac{N^{2}d^{2}}{M}\right).
$$

∎

## Appendix D Extension Details

### D.1 Block-sparse FlashAttention

We describe the full block-sparse FlashAttention algorithm in [Algorithm 5](#alg5 "In D.1 Block-sparse FlashAttention ‣ Appendix D Extension Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). The algorithm is identical to [Algorithm 2](#alg2 "In B.3 FlashAttention: Forward Pass ‣ Appendix B Algorithm Details ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"), except that we skip zero blocks.

<span id="alg5"></span>

**Algorithm 5: Block-Sparse FlashAttention Forward Pass**

- **Input:** Matrices $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ in HBM, on-chip SRAM of size $M$, softmax scaling constant $\tau\in\mathbb{R}$, masking function mask, dropout probability $p_{\mathrm{drop}}$, block sizes $B_{c}=\left\lceil\frac{M}{4d}\right\rceil,B_{r}=\min\left(\left\lceil\frac{M}{4d}\right\rceil,d\right)$, block sparsity mask $M\in\{0,1\}^{N/B_{r}\times N/B_{c}}$..
- Initialize the pseudo-random number generator state ${\cal R}$ and save to HBM.
- Initialize $\mathbf{O}=(0)_{N\times d}\in\mathbb{R}^{N\times d},\ell=(0)_{N}\in\mathbb{R}^{N},m=(-\infty)_{N}\in\mathbb{R}^{N}$ in HBM.
- Divide $\mathbf{Q}$ into $T_{r}=\left\lceil\frac{N}{B_{r}}\right\rceil$ blocks $\mathbf{Q}_{1},\dots,\mathbf{Q}_{T_{r}}$ of size $B_{r}\times d$ each, and divide $\mathbf{K},\mathbf{V}$ in to $T_{c}=\left\lceil\frac{N}{B_{c}}\right\rceil$ blocks $\mathbf{K}_{1},\dots,\mathbf{K}_{T_{c}}$ and $\mathbf{V}_{1},\dots,\mathbf{V}_{T_{c}}$, of size $B_{c}\times d$ each.
- Divide $\mathbf{O}$ into $T_{r}$ blocks $\mathbf{O}_{i},\dots,\mathbf{O}_{T_{r}}$ of size $B_{r}\times d$ each, divide $\ell$ into $T_{r}$ blocks $\ell_{i},\dots,\ell_{T_{r}}$ of size $B_{r}$ each, divide $m$ into $T_{r}$ blocks $m_{1},\dots,m_{T_{r}}$ of size $B_{r}$ each.
- **For** $1\leq j\leq T_{c}$ **do:**
  - Load $\mathbf{K}_{j},\mathbf{V}_{j}$ from HBM to on-chip SRAM.
  - **For** $1\leq i\leq T_{r}$ **do:**
    - **If** $M_{\mathrm{ij}}\neq 0$:
      - Load $\mathbf{Q}_{i},\mathbf{O}_{i},\ell_{i},m_{i}$ from HBM to on-chip SRAM.
      - On chip, compute $\mathbf{S}_{\mathrm{ij}}=\tau\mathbf{Q}_{i}\mathbf{K}_{j}^\top\in\mathbb{R}^{B_{r}\times B_{c}}$.
      - On chip, compute $\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}=\mathrm{mask}(\mathbf{S}_{\mathrm{ij}})$.
      - On chip, compute $\tilde{m}_{\mathrm{ij}}=\mathrm{rowmax}(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}})\in\mathbb{R}^{B_{r}}$, $\tilde{\mathbf{P}}_{\mathrm{ij}}=\exp(\mathbf{S}_{\mathrm{ij}}^{\mathrm{masked}}-\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}\times B_{c}}$ (pointwise), $\tilde{\ell}_{\mathrm{ij}}=\mathrm{rowsum}(\tilde{\mathbf{P}}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$.
      - On chip, compute $m_{i}^{\mathrm{new}}=\max(m_{i},\tilde{m}_{\mathrm{ij}})\in\mathbb{R}^{B_{r}}$, $\ell_{i}^{\mathrm{new}}=e^{m_{i}-m_{i}^{\mathrm{new}}}\ell_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\ell}_{\mathrm{ij}}\in\mathbb{R}^{B_{r}}$.
      - On chip, compute $\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}=\mathrm{dropout}(\tilde{\mathbf{P}}_{\mathrm{ij}},p_{\mathrm{drop}})$.
      - Write $\mathbf{O}_{i}\leftarrow\mathrm{diag}(\ell_{i}^{\mathrm{new}})^{-1}(\mathrm{diag}(\ell_{i})e^{m_{i}-m_{i}^{\mathrm{new}}}\mathbf{O}_{i}+e^{\tilde{m}_{\mathrm{ij}}-m_{i}^{\mathrm{new}}}\tilde{\mathbf{P}}_{\mathrm{ij}}^{\mathrm{dropped}}\mathbf{V}_{j})$ to HBM.
      - Write $\ell_{i}\leftarrow\ell_{i}^{\mathrm{new}}$, $m_{i}\leftarrow m_{i}^{\mathrm{new}}$ to HBM.
- **Return:** $\mathbf{O},\ell,m,{\cal R}$.

We prove the IO-complexity of block-sparse FlashAttention.

###### Proof of [Proposition 4](#Thmtheorem4 "Proposition 4. ‣ 3.3 Extension: Block-Sparse FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness").

The proof is very similar to the proof of [Theorem 2](#Thmtheorem2 "Theorem 2. ‣ 3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). For the block-sparse case, notice that we only need to load blocks corresponding to nonzero blocks. As a result, the number of HBM accesses are scaled by $s$, the fraction of nonzero blocks in the block-sparsity mask. However, for small values of $s$, we would still need to write the result $\mathbf{O}\in\mathbb{R}^{N\times d}$. Therefore the number of HBM accesses is

$$
\Theta\left(\mathrm{Nd}+\frac{N^{2}d^{2}}{M}s\right).
$$

∎

### D.2 Potential Extensions

We discuss here a few potential extensions of the IO-aware approach to speed up deep learning training.

Multi-GPU Attention. Large language models are trained on hundreds or thousands of GPUs, and one typically splits the attention computation between 4-8 GPUs on the same node [Shoeyb19]. This introduces another level of memory hierarchy: beside GPU SRAM and GPU HBM, we also have the HBM of other GPUs. For very long sequences, the different GPUs on the same node can cooperate to compute attention by taking into account the asymmetry of different levels of memory hierarchy.

Sparse MLP layers. Typical dense MLP layers are compute-bound and not memory-bound. To improve their efficiency, MLP layers with sparse weight matrices can be used [Daoa22]. However, many sparse MLP layers are instead memory-bound, and their speedup is often not proportional to the sparsity. We believe that an IO-aware implementation can alleviate this issue and realize the benefits of sparsity. We are excited about future work in this direction, to reduce the computational requirement of large models and improve their wall-block runtime.

Kernel machine learning. Our approach in FlashAttention relies on the fact that the $N\times N$ attention matrix is a function of a low-rank matrix $\mathbf{Q}\mathbf{K}^{\top}$ (of rank $d\ll N$). As a result, we can repeatedly load the inputs $\mathbf{Q},\mathbf{K}$ and recompute the block of the attention matrix that we need, significantly reducing HBM access. As similar scenario happens in kernel machine learning: each element $K_{\mathrm{ij}}$ of the $N\times N$ kernel matrix $\mathbf{K}$ is a function of two vectors of size $d\ll N$, as it measures the similarity between two datapoints $x_{i}$ and $x_{j}$. The KeOps library [Feydy20, Charli21] is a successful example of how reducing memory reads/writes can speed up kernel operations. We hope that this will motivate kernel methods that focus more on reducing IOs instead of just FLOPs.

## Appendix E Full Experimental Results

### E.1 BERT

We train BERT-large following the training procedure and hyperparameters of the reference MLPerf 1.1 implementation. In particular, we use the LAMB optimizer with learning rate 3.75e-3, with batch size 448, trained for at most 7100 steps. The training is stopped once the validation accuracy (for masked language modeling) reaches the target 72.0%, and the wall-clock run-time is measured. We train with FP16 precision using Apex AMP (with O2 optimization level).

We compare our results with the reported training speed from Nvidia that was submitted to MLPerf 1.1 ([Table 1](#table-01)).

We use the same train / validation data split provided by MLPerf 1.1 reference implementation. In particular, we evaluate on the same 10000 validation examples as the baseline from Nvidia.

We train the model on 8$\times$A100-80GB GPUs. Each training run takes between 16 and 19 minutes, and we average the results of 10 runs.

### E.2 GPT-2

We use the standard implementations of GPT-2 [Radfor19] from Huggingface transformers library and from Nvidia’s Megatron-LM repo. We follow the training recipe of the Megatron-LM repo.

We use an effective batch size of 512, and use gradient accumulation to fit into available GPU memory. We use the AdamW optimizer, with learning rate 6e-4 for GPT-2 small and 1.5e-4 for GPT-2 medium, and weight decay of 0.1. All models are trained with the same hyperparameters for 400K steps. We run all implementations with mixed-precision training (PyTorch AMP).

We use the Openwebtext dataset, with the GPT-2 BPE tokenizer. We randomly select 0.5% of the dataset as the validation set, with the rest being used as training set. This random selection of validation set is done once, and all models are evaluated on the same validation set.

We train the model on 8$\times$A100-40GB GPUs, and we measure the wall-clock training time. Training GPT-2 small takes between 2.7-9.5 days, and training GPT-2 medium takes between 6.9-21.0 days ([Table 2](#table-02)).

In [Fig. 4](#figure-04), we plot of the validation perplexity throughout training of GPT-2 small/medium, using either HuggingFace implementation or our FlashAttention implementation. We see that FlashAttention behaves the same as the baseline implementation and the validation perplexity curves of the two implementations almost lie on top of each other.

<span id="figure-04"></span>

![Refer to caption](../../papers/flashattention/figure-04.png)

**Figure 4.** Validation perplexity of GPT-2 small/medium using two implementations. We confirm that FlashAttention yields the same validation curves as the baseline implementation from HuggingFace.

##### Long Document Classification.

For MIMIC-III and ECtHR, we follow the hyperparameters of [Dai22].

### E.3 LRA details

We follow the hyperparameters from the Long-range arena paper [Tay20], the Long-range arena repo ([https://github.com/google-research/long-range-arena](https://github.com/google-research/long-range-arena)), and the Nyströmformer reproduction [Xiong21]. To be generous to the baseline methods, if we are unable to reproduce the performance of any baseline for any of the five tasks, we report the better performance from [Tay20] or [Xiong21] for that baseline on that task.

After hyperparameter tuning, almost all of the attention methods achieve similar accuracy on all of the five LRA tasks.

We run all methods with mixed-precision training, except for Performer (not stable with mixed precision) and Local Attention (implementation does not support FP16).

To calculate the overall wallclock-time speedup, we take the geometric mean of the wallclock-time speedup of each of the five tasks.

##### Path-X

For Path-X and Path-256, we follow the hyperparameters from the PathFinder-32 experiments from the long-range arena paper[Tay20]. For both, we first pretrain a model on Path-64. We take the checkpoint after 200 epochs, upsample its positional embedding (we duplicate the positional embeddings gridwise in space), and fine-tune it on the downstream task for 200 epochs with one epoch of linear warmup, and cosine decay of the learning rate. For Path-X, we take the best performing checkpoint (according to val accuracy), and additionally fine-tune it for 200 epochs with the same warmup and learning rate (this adds roughly 4 points of accuracy to FlashAttention for Path-X, but the model starts overfitting afterwards).

### E.4 Comparison with Apex FMHA

We compare our method/implementation with Apex FMHA ([https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha](https://github.com/NVIDIA/apex/tree/master/apex/contrib/csrc/fmha)).

When we started this project, Apex FMHA was the fastest implementation of attention (that we knew of), tailored for short sequences of length at most 512. In fact, almost all MLPerf submissions for BERT training benchmark running on Nvidia GPUs use FMHA for their model code, as of MLPerf 1.1 [Mattso20]. Since FMHA targets BERT models, it only supports head dimension 64, and only runs on A100 GPUs. FMHA fuses the attention computation $\mathrm{dropout}(\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^{\top})))\mathbf{V}$ into one CUDA kernel. In the forward pass, it stores the attention matrix $\mathrm{softmax}(\mathrm{mask}(\mathbf{Q}\mathbf{K}^\top))$ to HBM to be used in gradient computation. As a result, it does not offer substantial memory saving (though for shorter sequences memory footprint is often not a primary concern).

We use FMHA code as a starting point, and apply two well-established techniques (tiling and recomputation) to deal with long sequences and to save memory as mentioned in [Section 3](#S3 "3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). As a result, we can support much longer sequences (e.g., up to length 64K). We also support more head dimensions (16, 32, 64, 128) and broader GPU types (all Turing and Ampere GPUs at the time of writing).

In [Table 7](#table-07), we compare the performance of FlashAttention and Apex FMHA for short sequences (as FMHA only supports sequence length at most 512). Generally FlashAttention is slightly faster than FMHA in the forward pass and slightly slower than FMHA in the backward pass. This is because we do not store the attention matrix in the forward pass and recompute it in the backward pass. Compared to FMHA, the overall runtime of FlashAttention is about 4% slower for sequence length 128, 8% faster for sequence length 256, and 5% faster for sequence length 512.

<span id="table-07"></span>

![Original paper Table 7](../../papers/flashattention/table-07.png)

**Table 7.** Runtime (ms) of FlashAttention compared to FMHA by sequence length, with masking and dropout, measured on an A100-SXM4-40GB GPU. Batch size 64, 16 heads, head dimension 64 (i.e., BERT-large size).

### E.5 Speedup On Different Hardware and Configurations

Speedup varies between different types of GPU types and generations depending on HBM bandwidth and SRAM size. In this section, we profile FlashAttention speedup on different GPUs and configurations.

<span id="figure-05"></span>

![Refer to caption](../../papers/flashattention/figure-05.png)

**Figure 5.** Speedup over standard PyTorch attention at different sequence lengths, on A100.

##### A100

[Figure 5](#figure-05) shows speedup on an A100 GPU with batch size 8, head dimension 64, and 12 attention heads, across different sequence lengths. We generally see 2-4$\times$ speedup, and we see more speedup when using dropout and masking due to kernel fusion.

<span id="figure-06"></span>

![Refer to caption](../../papers/flashattention/figure-06.png)

**Figure 6.** Speedup over standard PyTorch attention at different sequence lengths, on A100, with head dimension 128.

##### A100, Head Dimension 128

Speedup also changes when we increase the head dimension. Each block requires more memory, so we need to use smaller block sizes to fit into SRAM. [Figure 6](#figure-06) shows speedup with head dimension 128 on an A100 (batch size 16, 12 heads). We see less speedup overall—but we can still see significant speedup (up to 3$\times$) with a causal mask, where half the blocks are masked out.

<span id="figure-07"></span>

![Refer to caption](../../papers/flashattention/figure-07.png)

**Figure 7.** Speedup over standard PyTorch attention at different sequence lengths, on RTX 3090.

##### RTX 3090

[Figure 7](#figure-07) shows speedup on an RTX 3090 GPU. Here, we use batch size 12 with 12 attention heads. We observe slightly higher speedups on the RTX 3090 (between 2.5-4.5$\times$), since the memory bandwidth on an RTX 3090 is lower than on an A100 (roughly 900 GB/s vs. 1.5 TB/s).

<span id="figure-08"></span>

![Refer to caption](../../papers/flashattention/figure-08a.png)

![Refer to caption](../../papers/flashattention/figure-08b.png)

**Figure 8.** Speedup over standard PyTorch attention at different sequence lengths, on T4. Top: Combined forward pass + backward pass. Bottom: Forward pass only.

##### T4

[Figure 8](#figure-08) shows speedup on a T4 GPU. T4 SRAM is smaller than A100, so we need to make the block sizes smaller in FlashAttention. As a result, we observe less speedup on T4, which matches the IO complexity analysis in Section [3.2](#S3.SS2 "3.2 Analysis: IO Complexity of FlashAttention ‣ 3 FlashAttention: Algorithm, Analysis, and Extensions ‣ FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"). T4 GPUs are commonly used for inference, so we also report speedup on the forward pass only.

### E.6 Full Benchmarking Results

We report the full benchmarking results and experimental details on A100.

##### Baselines

We compare against reference implementations for exact attention from PyTorch/HuggingFace and Megatron, approximate attention, and sparse attention. For approximate attention, we compare against reference implementations of Reformer [Kitaev20], Local Attention [Razavi20], Linformer Attention [Wang20], Smyrf [Daras20], and LongShortFormer (LSFormer) [Zhu21]. For sparse attention, we compare against reference implementations of Block-Sparse Attention form OpenAI [Child19], Longformer[Beltag20], and BigBird Attention [Zaheer20]. For the approximate and sparse attention, we use a compression ratio of 1/8, or a compressed sequence length of 256, whichever is smaller.

##### Setup

We measure runtime and memory usage of the attention computation with 8 heads of dimension 64, and batch size 16 on a machine with one A100 GPU with 40 GB of GPU HBM. We vary sequence length in our experiments. We compute attention on random vectors for $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ (we do not measure the projection from the hidden layer). For dropout, we use dropout 0.1; for masking, we use a padding mask with uniformly-random mask lengths between the total sequence length and the total sequence length minus 20. To measure runtime, we take the average of 100 measurements of the attention call. We only measure memory footprint once, since it does not vary between runs.

We report timing results on the forward pass, backward pass, and combined forward + backward pass. We measure each method with and without dropout, masking, or both—except for Block Sparse, Longformer, and BigBird. These methods did not successfully run the backward pass with masking due to a bug in external libraries, so we measured them without masking to be generous. We use FP16 for all measurements, except for Local Attention, whose implementation only supports FP32.

For each baseline, we increase sequence length until it runs out of memory on the GPU, except for the following exceptions: The Megatron implementation does not support sequence lengths longer than 2048. Block-Sparse (OpenAI) does not support sequence lengths longer than 4096. Longformer and BigBird do not support sequence lengths longer than 8092.

We measure memory usage on the combined forward + backward pass, without dropout or masking.

##### Results

[Table 8](#table-08) summarizes all the experimental configurations and contains pointers to the results tables.

<span id="table-08"></span>

![Original paper Table 8](../../papers/flashattention/table-08.png)

**Table 8.** Pointers to results tables.

<span id="table-09"></span>

![Original paper Table 9](../../papers/flashattention/table-09.png)

**Table 9.** Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

<span id="table-10"></span>

![Original paper Table 10](../../papers/flashattention/table-10.png)

**Table 10.** Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

<span id="table-11"></span>

![Original paper Table 11](../../papers/flashattention/table-11.png)

**Table 11.** Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout and masking. Best in bold, second best underlined.

<span id="table-12"></span>

![Original paper Table 12](../../papers/flashattention/table-12.png)

**Table 12.** Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

<span id="table-13"></span>

![Original paper Table 13](../../papers/flashattention/table-13.png)

**Table 13.** Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

<span id="table-14"></span>

![Original paper Table 14](../../papers/flashattention/table-14.png)

**Table 14.** Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with masking. Best in bold, second best underlined.

<span id="table-15"></span>

![Original paper Table 15](../../papers/flashattention/table-15.png)

**Table 15.** Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

<span id="table-16"></span>

![Original paper Table 16](../../papers/flashattention/table-16.png)

**Table 16.** Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

<span id="table-17"></span>

![Original paper Table 17](../../papers/flashattention/table-17.png)

**Table 17.** Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length, with dropout. Best in bold, second best underlined.

<span id="table-18"></span>

![Original paper Table 18](../../papers/flashattention/table-18.png)

**Table 18.** Forward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

<span id="table-19"></span>

![Original paper Table 19](../../papers/flashattention/table-19.png)

**Table 19.** Backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

<span id="table-20"></span>

![Original paper Table 20](../../papers/flashattention/table-20.png)

**Table 20.** Forward pass + backward pass runtime (ms) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

<span id="table-21"></span>

![Original paper Table 21](../../papers/flashattention/table-21.png)

**Table 21.** Memory usage (MB) of various exact/approximate/sparse attention mechanisms by sequence length. Best in bold, second best underlined.

[+1]: FlashAttention code is available at [https://github.com/HazyResearch/flash-attention](https://github.com/HazyResearch/flash-attention)

[+2]: This style of aggregation is called *algebraic aggregation* [Gray97].

[+3]: LRA accuracy results are known to be highly dependent on the tuning procedure [Xiong21]. Our reproduced baselines perform better than as reported in the original comparison [Tay20].

[+4]: Path-256 requires longer sequences but has relatively shorter paths than Path-X, so it is easier to obtain a higher accuracy.
