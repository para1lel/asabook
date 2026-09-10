---
title: 'FlashAttention-3'
createTime: 2026/09/10 14:13:02
permalink: /en/papers/flashattention-3/
pageClass: paper-reading
---

> [Jay Shah](https://developer.nvidia.com/blog/author/jayshah/) [+equal], [Ganesh Bikshandi](https://dblp.org/pid/68/2188.html) [+equal], [Ying Zhang](https://x.com/ipiszy), [Vijay Thakkar](https://cse.gatech.edu/people/vijay-thakkar), [Pradeep Ramani](https://developer.nvidia.com/blog/author/prramani/), and [Tri Dao](https://tridao.me/). First submitted to arXiv on July 11, 2024; current version v2, revised July 12, 2024. Published in the [Advances in Neural Information Processing Systems 37 (NeurIPS 2024) Main Conference Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608v2). <a href="/paper/flashattention-3.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2407.08608). [Proceedings DOI](https://doi.org/10.52202/079017-2193). [TeX source](https://export.arxiv.org/e-print/2407.08608v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Attention, as a core layer of the ubiquitous Transformer architecture, is the bottleneck for large language models and long-context applications. FlashAttention elaborated an approach to speed up attention on GPUs through minimizing memory reads/writes. However, it has yet to take advantage of new capabilities present in recent hardware, with FlashAttention-2 achieving only 35% utilization on the H100 GPU. We develop three main techniques to speed up attention on Hopper GPUs: exploiting asynchrony of the Tensor Cores and TMA to (1) overlap overall computation and data movement via warp-specialization and (2) interleave block-wise matmul and softmax operations, and (3) block quantization and incoherent processing that leverages hardware support for FP8 low-precision. We demonstrate that our method, FlashAttention-3, achieves speedup on H100 GPUs by 1.5-$2.0\times$ with FP16 reaching up to 740 TFLOPs/s (75% utilization), and with FP8 reaching close to 1.2 PFLOPs/s. We validate that FP8 FlashAttention-3 achieves $2.6\times$ lower numerical error than a baseline FP8 attention.

<span id="section-1"></span>

## 1 Introduction

For the Transformer architecture [Vas17], the attention mechanism constitutes the primary computational bottleneck, since computing the self-attention scores of queries and keys has quadratic scaling in the sequence length. Scaling attention to longer context will unlock new capabilities (modeling and reasoning over multiple long documents [Guo21a, Sha22a, Pen23] and files in large codebases [Roz23, Li23o]), new modalities (high-resolution images [Che22a], audio [Gul20], video [Ho22]), and new applications (user interaction with long history [Sun19d], agent workflow with long horizon [Yao22b]). This has generated significant interest in making attention faster in the long-context regime, including by approximation [Kat20, Cho20a, Tay20a] and software optimization ([Rab21, Dao22, Kwo23]), or even alternative architectures [Pen23b, Sun23b, Gu23].

In this work, we build on the work of [Dao22] on developing exact-attention algorithms that integrate knowledge of the GPU’s execution model and hardware characteristics into their high-level design. In [Dao22], Dao et al. introduced FlashAttention, a novel tiling strategy for parallelizing attention that eliminates intermediate reads/writes to slow global memory through fusing all of the attention operations into a single GPU kernel. [Dao23b] restructured the algorithm as FlashAttention-2 to also parallelize over the sequence length dimension and perform the inner loop of the forward pass over blocks of the key and value matrices, thus improving the occupancy and distribution of work on the GPU. However, we observe that FlashAttention-2 nonetheless achieves poor utilization on newer GPUs relative to optimized matrix-multiplication (GEMM) kernels, such as 35% vs. 80-90% on the Hopper H100 GPU. Partially, this may be attributed to implementation-level differences, such as not using Hopper-specific instructions in place of Ampere ones when targeting the Tensor Cores. Several work such as ThunkerKitten [Res24] and cuDNN 9 [Nvi24g] has shown that with Hopper-specific instructions and tile-based abstractions, one can speedup attention computation and simplify the implementation.

More fundamentally, FlashAttention-2’s algorithm adheres to a simplified synchronous model and makes no explicit use of asynchrony and low-precision in its design. Asynchrony is a result of hardware specialization to accelerate the most important operations in a ML workload: specific hardware units performing matrix multiplication (Tensor Cores) or memory loading (Tensor Memory Accelerator – TMA), separate from the rest of the CUDA cores performing logic, integer, and floating point computation. Low precision such as FP8 in Hopper and FP4 in Blackwell, continuing the trend of FP16 (Pascal in 2017) and BF16 (Ampere in 2020), is a proven technique to get double or quadruple throughput for the same power and chip area. We review the capabilities afforded by Hopper in these directions in [Section 2.2](#section-2-2). The technical challenge is to redesign FlashAttention-2 to make use of these hardware features: asynchrony requires overlapping computation between matmul and softmax even though one depends on the output of the other, and low-precision requires care to minimize quantization error, especially in the case of outlier features in LLMs [Det22, Sun24c].

To this end, we propose FlashAttention-3, which contributes and synthesizes three new ideas to further improve performance on newer GPU architectures: [+1]

- **Producer-Consumer asynchrony:** We define a warp-specialized software pipelining scheme that exploits the asynchronous execution of data movement and Tensor Cores by splitting producers and consumers of data into separate warps, thereby extending the algorithm’s ability to hide memory and instruction issue latencies.
- **Hiding softmax under asynchronous block-wise GEMMs:** We overlap the comparatively low-throughput non-GEMM operations involved in softmax, such as floating point multiply-add and exponential, with the asynchronous WGMMA instructions for GEMM. As part of this, we rework the FlashAttention-2 algorithm to circumvent certain sequential dependencies between softmax and the GEMMs. For example, in the 2-stage version of our algorithm, while softmax executes on one block of the scores matrix, WGMMA executes in the asynchronous proxy to compute the next block.
- **Hardware-accelerated low-precision GEMM:** We adapt the forward pass algorithm to allow for targeting the FP8 Tensor Cores for GEMM, nearly doubling the measured TFLOPs/s. This requires bridging the different layout conformance requirements of WGMMA in terms of how blocks of FP32 accumulator and FP8 operand matrices are assumed to be laid out in memory. We use the techniques of block quantization and incoherent processing to mitigate the loss of accuracy that results from moving to FP8 precision.

To validate our method empirically, we benchmark FlashAttention-3 on the H100 SXM5 GPU over a range of parameters and show that (1) FP16 achieves 1.5-$2.0\times$ speedup over FlashAttention-2 in the forward pass (reaching up to 740 TFLOPs/s) and 1.5-$1.75\times$ in the backward pass, (2) FP8 achieves close to 1.2 PFLOPs/s, and (3) for large sequence length, FP16 outperforms and FP8 is competitive [+2] with a state-of-the-art implementation of attention from NVIDIA’s cuDNN library. We also validate that FP16 FlashAttention-3 yields the same numerical error as FlashAttention-2 and is better than the standard attention implementation as intermediate results (e.g., softmax rescaling) are kept in FP32. Moreover, FP8 FlashAttention-3 with block quantization and incoherent processing is $2.6\times$ more accurate than standard attention with per-tensor quantization in cases with outlier features.

We open-source FlashAttention-3 with a permissive license [+3] and plan to integrate it with PyTorch and Hugging Face libraries to benefit the largest number of researchers and developers.

<span id="section-2"></span>

## 2 Background: Multi-Head Attention and GPU Characteristics

<span id="section-2-1"></span>

### 2.1 Multi-Head Attention

Let $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ be the query, key and value input sequences associated to a single head, where $N$ is the sequence length and $d$ is the head dimension. Then the attention output $\mathbf{O}$ is computed as:

$$
\mathbf{S}=\alpha\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

where $\mathrm{softmax}$ is applied row-wise and one typically sets $\alpha=1/\sqrt{d}$ as the scaling factor. In practice, we subtract $\mathrm{rowmax}(\mathbf{S})$ from $\mathbf{S}$ to prevent numerical instability with the exponential function. For multi-head attention (MHA), each head has its own set of query, key and value projections, and this computation parallelizes across multiple heads and batches to produce the full output tensor.

Now let $\phi$ be a scalar loss function and let $\mathbf{d}(-)=\partial\phi/\partial(-)$ be notation for the gradient. Given the output gradient $\mathbf{dO}\in\mathbb{R}^{N\times d}$, we compute $\mathbf{dQ}$, $\mathbf{dK}$, and $\mathbf{dV}$ according to the chain rule as follows:

$$
\begin{aligned}
\mathbf{dV} & =\mathbf{P}^{\top}\mathbf{dO}\in\mathbb{R}^{N\times d} \\
\mathbf{dP} & =\mathbf{dO}\mathbf{V}^{\top}\in\mathbb{R}^{N\times N} \\
\mathbf{dS} & =\mathrm{dsoftmax}(\mathbf{dP})\in\mathbb{R}^{N\times N} \\
\mathbf{dQ} & =\alpha\mathbf{dS}\mathbf{K}\in\mathbb{R}^{N\times d} \\
\mathbf{dK} & =\alpha\mathbf{dS}^{\top}\mathbf{Q}\in\mathbb{R}^{N\times d},
\end{aligned}
$$

Here, we have that $\mathbf{d}s=(\mathrm{diag}(p)-pp^{\top})\mathbf{d}p$ for $p=\mathrm{softmax}(s)$ as a function of a vector $s$, and we write $\mathrm{dsoftmax}(\mathbf{dP})$ for this formula applied row-wise. Finally, this computation again parallelizes across the number of heads and batches for the backward pass of MHA.

<span id="section-2-2"></span>

### 2.2 GPU hardware characteristics and execution model

We describe the aspects of the GPU’s execution model relevant for FlashAttention-3, with a focus on the NVIDIA Hopper architecture as a concrete instantiation of this model.

**Memory hierarchy:** The GPU’s memories are organized as a hierarchy of data locales, with capacity inversely related to bandwidth ([Table 1](#table-01)) [+4]. Global memory (GMEM), also known as HBM, is the off-chip DRAM accessible to all streaming multiprocessors (SMs). Data from GMEM gets transparently cached into an on-chip L2 cache. Next, each SM contains a small on-chip, programmer-managed highly banked cache called shared memory (SMEM). Lastly, there is the register file within each SM.

**Thread hierarchy:** The GPU’s programming model is organized around logical groupings of execution units called threads. From the finest to coarsest level, the thread hierarchy is comprised of threads, warps (32 threads), warpgroups (4 contiguous warps), threadblocks (i.e., cooperative thread arrays or CTAs), threadblock clusters (in Hopper), and grids.

These two hierarchies are closely interlinked. Threads in the same CTA are co-scheduled on the same SM, and CTAs in the same cluster are co-scheduled on the same GPC. SMEM is directly addressable by all threads within a CTA, whereas each thread has at most 256 registers (RMEM) private to itself.

<span id="table-01"></span>

![Table 1. Thread-Memory hierarchy for the NVIDIA Hopper H100 SXM5 GPU.](../../papers/flashattention-3/table-01.png)

**Table 1.** Thread-Memory hierarchy for the NVIDIA Hopper H100 SXM5 GPU.

**Asynchrony and warp-specialization:** GPUs are throughput processors that rely on concurrency and asynchrony to hide memory and execution latencies. For async memory copy between GMEM and SMEM, Hopper has the Tensor Memory Accelerator (TMA) as a dedicated hardware unit [Nvi24c]. Furthermore, unlike prior architectures such as Ampere, the Tensor Core of Hopper, exposed via the warpgroup-wide WGMMA instruction [Ptx24], is also asynchronous and can source its inputs directly from shared memory.

Hardware support for asynchrony allows for warp-specialized kernels, where the warps of a CTA are divided into producer or consumer roles that only ever issue either data movement or computation. Generically, this improves the compiler’s ability to generate optimal instruction schedules [Bau11]. In addition, Hopper supports the dynamic reallocation of registers between warpgroups via `setmaxnreg` [Ptx24], so those warps doing MMAs can obtain a larger share of RMEM than those just issuing TMA (for which only a single thread is needed).

**Low-precision number formats:** Modern GPUs have specialized hardware units for accelerating low-precision computation. For example, the WGMMA instruction can target the FP8 Tensor Cores on Hopper to deliver 2x the throughput per SM when compared to FP16 or BF16.

However, correctly invoking FP8 WGMMA entails understanding the layout constraints on its operands. Given a GEMM call to multiply $A\times B^{\top}$ for an $M\times K$-matrix $A$ and an $N\times K$-matrix $B$, we say that the $A$ or $B$ operand is *mn-major* if it is contiguous in the outer $M$ or $N$ dimension, and *k-major* if is instead contiguous in the inner $K$-dimension. Then for FP16 WGMMA, both mn-major and k-major input operands are accepted for operands in SMEM, but for FP8 WGMMA, only the k-major format is supported. Moreover, in situations such as attention where one wants to fuse back-to-back GEMMs in a single kernel, clashing FP32 accumulator and FP8 operand layouts pose an obstacle to invoking dependent FP8 WGMMAs.

In the context of attention, these layout restrictions entail certain modifications to the design of an FP8 algorithm, which we describe in [Section 3.3](#section-3-3).

<span id="section-2-3"></span>

### 2.3 Standard Attention and Flash Attention

Following [Dao22], we let **standard attention** denote an implementation of attention on the GPU that materializes the intermediate matrices $\mathbf{S}$ and $\mathbf{P}$ to HBM. The main idea of FlashAttention was to leverage a local version of the softmax reduction to avoid these expensive intermediate reads/writes and fuse attention into a single kernel. Local softmax corresponds to lines 18-19 of the consumer mainloop in [Algorithm 1](#algorithm-01) together with the rescalings of blocks of $\mathbf{O}$. The simple derivation that this procedure indeed computes $\mathbf{O}$ can be found in [Dao23b].

<span id="section-3"></span>

## 3 FlashAttention-3: Algorithm

In this section, we describe the FlashAttention-3 algorithm. For simplicity, we focus on the forward pass, with the backward pass algorithm described in [Section 7.1](#section-7-1). We first indicate how to integrate warp-specialization with a circular SMEM buffer into the base algorithm of FlashAttention-2. We then explain how to exploit asynchrony of WGMMA to define an overlapped GEMM-softmax 2-stage pipeline. Finally, we describe the modifications needed for FP8, both in terms of layout conformance and accuracy via block quantization and incoherent processing.

<span id="section-3-1"></span>

### 3.1 Producer-Consumer asynchrony through warp-specialization and pingpong scheduling

**Warp-specialization.** As with FlashAttention-2, the forward pass of FlashAttention-3 is embarrassingly parallel in the batch size, number of heads, and query sequence length. Thus, it will suffice to give a CTA-level view of the algorithm, which operates on a tile $\mathbf{Q}_{i}$ of the query matrix to compute the corresponding tile $\mathbf{O}_{i}$ of the output. To simplify the description, we first give the warp-specialization scheme with a circular SMEM buffer that does *not* have in addition the GEMM-softmax overlapping. Let $d$ be the head dimension, $N$ the sequence length, and fix a query block size $B_{r}$ to divide $\mathbf{Q}$ into $T_{r}=\lceil\frac{N}{B_{r}}\rceil$ blocks $\mathbf{Q}_{1},..,\mathbf{Q}_{T_{r}}$.

<span id="algorithm-01"></span>

**Algorithm 1: FlashAttention-3 forward pass without intra-consumer overlapping — CTA view.**

- **Require:** Matrices $\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}$ and $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ in HBM, key block size $B_c$ with $T_c = \lceil \frac{N}{B_c} \rceil$.
- Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
- **If** in producer warpgroup:
  - Deallocate predetermined number of registers.
  - Issue load $\mathbf{Q}_i$ from HBM to shared memory.
  - Upon completion, commit to notify consumer of the load of $\mathbf{Q}_i$.
  - **For** $0 \le j < T_c$:
    - Wait for the $(j\,\%\,s)$th stage of the buffer to be consumed.
    - Issue loads of $\mathbf{K}_j, \mathbf{V}_j$ from HBM to shared memory at the $(j\,\%\,s)$th stage of the buffer.
    - Upon completion, commit to notify consumers of the loads of $\mathbf{K}_j, \mathbf{V}_j$.
- **Else:**
  - Reallocate predetermined number of registers as function of number of consumer warps.
  - On-chip, initialize $\mathbf{O}_i = (0) \in \mathbb{R}^{B_r \times d}$ and $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
  - Wait for $\mathbf{Q}_i$ to be loaded in shared memory.
  - **For** $0 \le j < T_c$:
    - Wait for $\mathbf{K}_j$ to be loaded in shared memory.
    - Compute $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top$ (SS-GEMM). Commit and wait.
    - Store $m_i^{\mathrm{old}} = m_i$ and compute $m_i = \max(m_i^{\mathrm{old}}, \mathrm{rowmax}(\mathbf{S}_i^{(j)}))$.
    - Compute $\widetilde{\mathbf{P}}_i^{(j)} = \exp(\mathbf{S}_i^{(j)} - m_i)$ and $\ell_i = \exp(m_i^{\mathrm{old}} - m_i) \ell_i + \mathrm{rowsum}(\widetilde{\mathbf{P}}_i^{(j)})$.
    - Wait for $\mathbf{V}_j$ to be loaded in shared memory.
    - Compute $\mathbf{O}_i = \mathrm{diag}(\exp(m_i^{\mathrm{old}} - m_i))^{-1} \mathbf{O}_i + \widetilde{\mathbf{P}}_i^{(j)} \mathbf{V}_j$ (RS-GEMM). Commit and wait.
    - Release the $(j\,\%\,s)$th stage of the buffer for the producer.
  - Compute $\mathbf{O}_i = \mathrm{diag}(\ell_i)^{-1} \mathbf{O}_i$ and $L_i = m_i + \log(\ell_i)$.
  - Write $\mathbf{O}_i$ and $L_i$ to HBM as the $i$th block of $\mathbf{O}$ and $L$.

For our implementation of [Algorithm 1](#algorithm-01) on Hopper, we use `setmaxnreg` for (de)allocations, TMA for loads of $\mathbf{Q}_{i}$ and $\{\mathbf{K}_{j},\mathbf{V}_{j}\}_{0\leq j<T_{c}}$, and WGMMA to execute the GEMMs in the consumer mainloop, with the SS or RS prefix indicating whether the first operand is sourced from shared memory or register file. For interpreting the execution flow of [Algorithm 1](#algorithm-01), note that issuing TMA loads does not stall on the completion of other loads due to asynchrony. Moreover, in the producer mainloop, no waits will be issued for the first $s$ iterations as the buffer gets filled.

**Pingpong scheduling.** The asynchronous nature of WGMMA and TMA, along with warp-specialization, opens up the opportunity to overlap the softmax computation of one warpgroup with the GEMM of another warpgroup. To motivate this, notice that non-matmul operations have much lower throughput than matmul operations on modern hardware accelerators. As an example, the H100 SXM5 GPU has 989 TFLOPS of FP16 matmul but only 3.9 TFLOPS of special functions such as exponential [+5] (necessary for softmax). For the attention forward pass in FP16 with head dimension 128, there are 512x more matmul FLOPS compared to exponential operations, but the exponential has 256x lower throughput, so exponential can take 50% of the cycle compared to matmul. The situation is even worse with FP8, where the matmul throughput doubles but the exponential throughput stays the same.

Since the exponential is performed by a separate hardware unit (the multi-function unit), ideally we’d want the exponential calculation to be scheduled when the Tensor Cores are performing the matmul. To do so, we use synchronization barriers (`bar.sync` instructions) to force the GEMMs (GEMM1 – $\mathbf{P}\mathbf{V}$ of one iteration, and GEMM0 – $\mathbf{Q}\mathbf{K}^{\top}$ of the next iteration) of warpgroup 1 to be scheduled before the GEMMs of warpgroup 2. As a result, the softmax of warpgroup 1 will be scheduled while warpgroup 2 is performing its GEMMs. Then the roles swap, with warpgroup 2 doing softmax while warpgroup 1 doing GEMMs (hence, “pingpong” scheduling). This is illustrated in [Figure 1](#figure-01). Though in practice the pingpong scheduling is not as clean as depicted in the figure, we generally find this to improve performance (e.g., from 570 TFLOPS to 620-640 TFLOPS for FP16 forward with head dimension 128 and sequence length 8192).

<span id="figure-01"></span>

![Figure 1. Pingpong scheduling for 2 warpgroups to overlap softmax and GEMMs: the softmax of one warpgroup should be scheduled when the GEMMs of another warpgroup are running. The same color denotes the same iteration.](../../papers/flashattention-3/figure-01.png)

**Figure 1.** Pingpong scheduling for 2 warpgroups to overlap softmax and GEMMs: the softmax of one warpgroup should be scheduled when the GEMMs of another warpgroup are running. The same color denotes the same iteration.

**Attention variants.** For multi-query attention [Sha19] and grouped query attention [Ain23a], we follow the approach in FlashAttention-2 and adjust the tensor indexing to avoid duplicating $\mathbf{K}$ and $\mathbf{V}$ in HBM.

<span id="section-3-2"></span>

### 3.2 Intra-warpgroup overlapping GEMMs and softmax

Even within one warpgroup, we can overlap some instructions in the softmax with some instructions in the GEMMs. We describe one technique to do so.

In the attention algorithm, operations within the inner loop (main loop) have sequential dependencies that impede parallelization within a single iteration. For example, (local) softmax (lines 18 to 19) relies on the output $\mathbf{S}_{i}^{(j)}$ of the first GEMM, while the second GEMM takes its result $\widetilde{\mathbf{P}}_{i}^{(j)}$ as an operand. Indeed, the wait statements in lines 17 and 21 of [Algorithm 1](#algorithm-01) serialize the execution of softmax and GEMMs. However, we can break these dependencies by pipelining across iterations through additional buffers in registers. Pursuing this idea, we propose the following two-stage [+6] GEMM-softmax pipelining algorithm:

<span id="figure-02"></span>

![Figure 2. 2-stage WGMMA-softmax pipelining](../../papers/flashattention-3/figure-02.png)

**Figure 2.** 2-stage WGMMA-softmax pipelining

<span id="algorithm-02"></span>

**Algorithm 2: FlashAttention-3 consumer warpgroup forward pass.**

- **Require:** Matrices $\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}$ and $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ in HBM, key block size $B_c$ with $T_c = \lceil \frac{N}{B_c} \rceil$.
- Reallocate predetermined number of registers as function of number of consumer warps.
- On-chip, initialize $\mathbf{O}_i = (0) \in \mathbb{R}^{B_r \times d}$ and $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
- Wait for $\mathbf{Q}_i$ and $\mathbf{K}_0$ to be loaded in shared memory.
- Compute $\mathbf{S}_{\mathrm{cur}} = \mathbf{Q}_i \mathbf{K}_0^\top$ using WGMMA. Commit and wait.
- Release the $0$th stage of the buffer for $\mathbf{K}$.
- Compute $m_i$, $\widetilde{\mathbf{P}}_{\mathrm{cur}}$ and $\ell_i$ based on $\mathbf{S}_{\mathrm{cur}}$, and rescale $\mathbf{O}_i$.
- **For** $1 \le j < T_c - 1$:
  - Wait for $\mathbf{K}_j$ to be loaded in shared memory.
  - Compute $\mathbf{S}_{\mathrm{next}} = \mathbf{Q}_i \mathbf{K}_{j}^\top$ using WGMMA. Commit but do not wait.
  - Wait for $\mathbf{V}_{j-1}$ to be loaded in shared memory.
  - Compute $\mathbf{O}_{i} = \mathbf{O}_{i} + \widetilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$ using WGMMA. Commit but do not wait.
  - Wait for the WGMMA $\mathbf{Q}_i \mathbf{K}_{j}^\top$.
  - Compute $m_i$, $\widetilde{\mathbf{P}}_{\mathrm{next}}$ and $\ell_i$ based on $\mathbf{S}_{\mathrm{next}}$.
  - Wait for the WGMMA $\widetilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$ and then rescale $\mathbf{O}_i$.
  - Release the $(j\,\%\,s)$th, resp. $(j-1\,\%\,s)$th stage of the buffer for $\mathbf{K}$, resp. $\mathbf{V}$.
  - Copy $\mathbf{S}_{\mathrm{next}}$ to $\mathbf{S}_{\mathrm{cur}}$.
- Wait for $\mathbf{V}_{T_c - 1}$ to be loaded in shared memory.
- Compute $\mathbf{O}_{i} = \mathbf{O}_{i} + \widetilde{\mathbf{P}}_{\mathrm{last}} \mathbf{V}_{T_c - 1}$ using WGMMA. Commit and wait.
- **Epilogue:**
  - Rescale $\mathbf{O}_{i}$ based on $m_i$.
  - Compute $L_i$ based on $m_i$ and $\ell_i$.
  - Write $\mathbf{O}_{i}$ and $L_i$ to HBM as the $i$th block of $\mathbf{O}$ and $L$.

[Algorithm 2](#algorithm-02) functions as a replacement for the consumer path of [Algorithm 1](#algorithm-01) to comprise the complete FlashAttention-3 algorithm for FP16 precision. At a high-level, we use WGMMA as a metonym for asynchronous GEMM. Within the mainloop (lines 8 to 16), the second WGMMA operation of iteration $j$ (line 11) is overlapped with softmax operations from iteration $j+1$ (line 13).

While the pipelined structure illustrated above offers theoretical performance gains, there are several practical aspects to consider:

**Compiler reordering.** The pseudocode represents an idealized execution order but the compiler (NVCC) often rearranges instructions for optimization. This can disrupt the carefully crafted WGMMA and non-WGMMA operation pipelining sequence, potentially leading to unexpected behavior or diminished performance gains. An analysis of the SASS code shows that the compiler generates overlapped code as expected ([Section 7.2](#section-7-2)).

**Register pressure.** To maintain optimal performance, register spilling should be minimized. However, the 2-stage pipeline requires additional registers to store intermediate results and maintain context between stages. Specifically, an extra $\mathbf{S}_{\mathrm{next}}$ must be kept in registers, leading to extra register usage of size $B_{r}\times B_{c}\times\text{sizeof}(\text{float})$ per threadblock. This increased register demand may conflict with using larger block sizes (another common optimization), which is also register-hungry. In practice, trade-offs should be made based on profiling results.

**3-stage pipelining.** Extending the 2-stage algorithm described above, we propose a 3-stage variant that would further overlap the second WGMMA with softmax. While this approach offers the potential for even higher Tensor Core utilization, it requires even more registers due to an additional stage in the pipeline, making the trade-off between tile size and pipeline depth more difficult to balance. A detailed description of the 3-stage algorithm and its evaluation results can be found in [Section 7.3](#section-7-3).

<span id="section-3-3"></span>

### 3.3 Low-precision with FP8

<span id="figure-03"></span>

![Figure 3. FP32 accumulator register WGMMA layout – rows 0 and 8, threads 0-3, entries 0-7.](../../papers/flashattention-3/figure-03.png)

**Figure 3.** FP32 accumulator register WGMMA layout – rows 0 and 8, threads 0-3, entries 0-7.

<span id="figure-04"></span>

![Figure 4. FP8 operand A register WGMMA layout – rows 0 and 8, threads 0-3, entries 0-7.](../../papers/flashattention-3/figure-04.png)

**Figure 4.** FP8 operand A register WGMMA layout – rows 0 and 8, threads 0-3, entries 0-7.

**Efficiency: layout transformations.** Computing the forward pass of FlashAttention-3 in FP8 precision poses additional challenges not encountered for FP16 in terms of layout conformance.

First, we note that the input tensors $\mathbf{Q}$, $\mathbf{K}$, and $\mathbf{V}$ are typically given as contiguous in the head dimension, while to satisfy the k-major constraint on FP8 WGMMA for the second GEMM we need $\mathbf{V}$, or rather the tiles of $\mathbf{V}$ loaded into SMEM, to be contiguous in the sequence length dimension. Since the TMA load itself cannot change the contiguous dimension, we then need to either (1) transpose $\mathbf{V}$ in GMEM as a pre-processing step, or (2) do an in-kernel transpose of tiles of $\mathbf{V}$ after loading them into SMEM. To implement option (1), we can either (1a) fuse the transpose to the epilogue of a preceding step such as the rotary embedding, or (1b) call a standalone pre-processing transpose kernel [+7] to exchange the strides of the sequence length and head dimensions. However, (1a) is difficult to integrate into a standard library, and (1b) is too wasteful in a memory-bound situation such as inference.

Instead, for FP8 FlashAttention-3 we opt for option (2). For the in-kernel transpose, we take advantage of the LDSM (`ldmatrix`) and STSM (`stmatrix`) instructions, which involve a warp of threads collectively loading SMEM to RMEM and storing RMEM to SMEM at a granularity of 128 bytes. [+8] The LDSM/STSM instructions are both register efficient, allowing us to execute them in the producer warpgroup, and capable of transposing layouts when doing memory copy. Moreover, after the first iteration we can arrange for the transpose of the next $\mathbf{V}$ tile to be executed in the shadow of the two WGMMAs that involve the preceding $\mathbf{V}$ and current $\mathbf{K}$ tile.

Second, we observe that unlike with FP16, the memory layout of the FP32 accumulator of an FP8 WGMMA is different from that assumed for its operand A when held in registers. We depict fragments of these two layouts in [Figure 3](#figure-03) and [Figure 4](#figure-04), where the entries are held in registers per thread in the listed order. By using byte permute instructions, we can then transform the first WGMMA’s accumulator into a format suitable for the second WGMMA, and compatibly with the layout of the $\mathbf{V}$ tile produced by the in-kernel transpose. Specifically, with reference to [Figure 3](#figure-03), we change the order in sequence to

$$
\{\verb|d0 d1 d4 d5 d2 d3 d6 d7|\},
$$

and this register permutation is then replicated over every 8 bytes. In terms of the logical shape of the $\mathbf{P}$ tile, this manuever permutes its columns (e.g., columns $0189$ now become the first four columns). For WGMMA to then compute the correct output tile, we can correspondingly arrange for the in-kernel transpose to write out a matching row permutation of the $\mathbf{V}$ tile. [+9]

**Accuracy: block quantization and incoherent processing.** With FP8 (e4m3) format, one only uses 3 bits to store the mantissa and 4 bits for the exponent. This results in higher numerical error than FP16/BF16. Moreover, large models typically have outlier values [Det22, Sun24c] that are much larger in magnitude than most other values, making quantization difficult. One typically use per-tensor scaling [Mic22] by keeping one scalar per tensor (e.g., one for $\mathbf{Q}$, for $\mathbf{K}$, and for $\mathbf{V}$). To reduce the numerical error of attention in FP8, we employ two techniques:

- **Block quantization**: we keep one scalar per block, so that for each of $\mathbf{Q}$, $\mathbf{K}$, $\mathbf{V}$ we split the tensor into blocks of size $B_{r}\times d$ or $B_{c}\times d$ and quantize them separately. This quantization can be fused with an operation right before attention (e.g., rotary embedding) with no additional slow down (since rotary embedding is memory-bandwidth bound). As the FlashAttention-3 algorithm naturally operates on blocks, we can scale each block of $\mathbf{S}$ to account for this block quantization at no computation cost.
- **Incoherent processing**: to even out outliers, we multiply $\mathbf{Q}$ and $\mathbf{K}$ with a random orthogonal matrix $\mathbf{M}$ before quantizing to FP8. Since $\mathbf{M}$ is orthogonal, $\mathbf{M}\mathbf{M}^{\top}=I$ and so $(\mathbf{Q}\mathbf{M})(\mathbf{K}\mathbf{M})^{\top}=\mathbf{Q}\mathbf{K}^{\top}$, i.e., multiplying both $\mathbf{Q}$ and $\mathbf{K}$ with $\mathbf{M}$ does not change the attention output. This serves to “spread out” the outliers since each entry of $\mathbf{Q}\mathbf{M}$ or $\mathbf{K}\mathbf{M}$ is a random sum of entries of $\mathbf{Q}$ or $\mathbf{K}$, thus reducing quantization error. In practice, we follow [Che24b] and [Tse24] and choose $\mathbf{M}$ to be the product of random diagonal matrices of $\pm 1$ and a Hadamard matrix, which can be multiplied in $O(d\log d)$ instead of $O(d^{2})$, and can also be fused with the rotary embedding at no extra computation cost.

We validate that these two techniques reduces numerical error by up to $2.6\times$ in [Section 4.3](#section-4-3).

<span id="section-4"></span>

## 4 Empirical Validation

We use the primitives from CUTLASS [Nvi24a] such as WGMMA and TMA abstractions to implement FlashAttention-3 and evaluate its efficiency and accuracy.

- **Benchmarking attention.** We measure the runtime of FlashAttention-3 across different sequence lengths and compare it to a standard implementation in PyTorch, FlashAttention-2, FlashAttention-2 in Triton (which uses H100-specific instructions), as well as a vendor’s implementation of FlashAttention-2 optimized for H100 GPUs from cuDNN. We confirm that FlashAttention-3 is up to $2.0\times$ faster than FlashAttention-2 and $1.5\times$ faster than FlashAttention-2 in Triton. FlashAttention-3 reaches up to 740 TFLOPs/s, 75% of the theoretical maximum TFLOPs/s on H100 GPUs.
- **Ablation study.** We confirm that our algorithmic improvements with warp-specialization and GEMM-softmax pipelining contribute to the speedup of FlashAttention-3.
- **Accuracy of FP8 attention.** We validate that block quantization and incoherent processing reduces the numerical error of FP8 FlashAttention-3 by $2.6\times$.

<span id="section-4-1"></span>

### 4.1 Benchmarking Attention

We measure the runtime of different attention methods on an H100 80GB SXM5 GPU for different settings (without / with causal mask, head dimension 64 or 128) for FP16 inputs. We report the results in [Figure 5](#figure-05) and [Figure 6](#figure-06), showing that FlashAttention-3 is around 1.5-$2.0\times$ faster than FlashAttention-2 in the forward pass and 1.5-$1.75\times$ faster in the backward pass. Compared to a standard attention implementation, FlashAttention-3 can be up to 3-$16\times$ faster. For medium and long sequences (1k and above), FlashAttention-3 even surpasses the speed of a vendor’s library (cuDNN – closed source) that has been optimized for H100 GPUs.

**Benchmark settings:** We vary the sequence length as 512, 1k, …, 16k, and set batch size so that the total number of tokens is 16k. We set the hidden dimension to 2048, and head dimension to be either 64, 128, or 256 (i.e., 32 heads, 16 heads, or 8 heads). To calculate the FLOPs of the forward pass, we use:

$$
\begin{aligned}
4\cdot\text{seqlen}^{2}\cdot\text{head dimension}\\
{}\cdot\text{number of heads}.
\end{aligned}
$$

With causal masking, we divide this number by 2 to account for the fact that approximately only half of the entries are calculated. To get the FLOPs of the backward pass, we multiply the forward pass FLOPs by 2.5 (since there are 2 matmuls in the forward pass and 5 matmuls in the backward pass, due to recomputation).

<span id="figure-05"></span>

![Figure 5. Attention forward speed (FP16/BF16) on H100 GPU](../../papers/flashattention-3/figure-05.png)

**Figure 5.** Attention forward speed (FP16/BF16) on H100 GPU

<span id="figure-06"></span>

![Figure 6. Attention backward speed (FP16/BF16) on H100 GPU](../../papers/flashattention-3/figure-06.png)

**Figure 6.** Attention backward speed (FP16/BF16) on H100 GPU

We also measure the runtime for FP8 for the forward pass under similar settings. We report the results for headdim 256 in [Figure 7](#figure-07) and give the full results in [Section 8.2](#section-8-2).

<span id="figure-07"></span>

![Figure 7. Attention forward speed (FP8) on H100 GPU](../../papers/flashattention-3/figure-07.png)

**Figure 7.** Attention forward speed (FP8) on H100 GPU

<span id="section-4-2"></span>

### 4.2 Ablation Study: 2-Stage Pipelining Experiments

We ablate both the 2-stage WGMMA-softmax pipelining and warp-specialization for non-causal FP16 FlashAttention-3 with fixed parameters $\{\text{batch},\text{seqlen},\text{nheads},\text{hdim}\}=\{4,8448,16,128\}$. The result in [Table 2](#table-02) confirms that our algorithmic improvements (asynchrony with warp-specialization and overlapping between GEMM and softmax) lead to significant speedup, from 570 to 661 TFLOPs.

<span id="table-02"></span>

![Table 2. Pipelining ablation measurements](../../papers/flashattention-3/table-02.png)

**Table 2.** Pipelining ablation measurements

<span id="section-4-3"></span>

### 4.3 Numerical Error Validation

As there has been interest in the numerical error [Gol24] of FlashAttention, we compare FlashAttention-2, FlashAttention-3, and a standard implementation of attention against a reference implementation in FP64. To simulate outlier features and activations in LLMs [Det22, Sun24c], we generate the entries of $\mathbf{Q},\mathbf{K},\mathbf{V}$ with the following distribution:

$$
\mathcal{N}(0,1)+\mathcal{N}(0,100)\cdot\mathrm{Bernoulli}(0.001).
$$

That is, each entry is normally distributed with zero mean and standard deviation 1, but for 0.1% of entries we add an independent term that’s normally distributed with standard deviation 10. We then measure the root mean squared error (RMSE) in [Table 3](#table-03). In FP16, both FlashAttention-2 and FlashAttention-3 achieves $1.7\times$ lower RMSE compared to the standard implementation since intermediate results (softmax) are kept in FP32. The baseline attention in FP8 uses per-tensor scaling, with matmul accumulator in FP32 and intermediate softmax results kept in FP16. Thanks to block quantization and incoherent processing, FlashAttention-3 in FP8 is $2.6\times$ more accurate than this baseline.

<span id="table-03"></span>

![Table 3. Numerical error comparisons in FP16 and FP8 (e4m3).](../../papers/flashattention-3/table-03.png)

**Table 3.** Numerical error comparisons in FP16 and FP8 (e4m3).

<span id="section-5"></span>

## 5 Dicussion, Limitations, Conclusion

With FlashAttention-3, we have demonstrated that new programming techniques and hardware features such as asynchrony and low-precision can have a dramatic impact on the efficiency and accuracy of attention. We are able to speed up attention by 1.5-$2.0\times$ times compared to FlashAttention-2, and reduce FP8 numerical error by $2.6\times$ compared to standard per-tensor quantization. Some limitations of our work that we hope to address in the future include: optimizing for LLM inference, integrating a persistent kernel design into the FP8 kernel, [+10] and understanding the effects of low-precision attention in large-scale training. Though we have focused on Hopper GPUs in this work, we expect that the techniques developed here will apply to other hardware accelerators. We hope that a faster and more accurate primitive such as attention will unlock new applications in long-context tasks.

## Acknowledgments

We are grateful to the NVIDIA CUTLASS team (especially Haicheng Wu, Aniket Shivam, and Cris Cecka) for helping us understand Hopper’s programming model and for their library, which provides clean and powerful building blocks for the implementation of FlashAttention-3. We thank the cuDNN team for the idea of in-kernel transpose for FP8. The idea of overlapping GEMMs and softmax was inspired by insightful conversations with Christopher Ré, Benjamin Spector, Aniket Shivam, and Markus Hoehnerbach. The pingpong scheduling is adapted from the warp-specialized pingpong GEMM implementation in CUTLASS. We appreciate Driss Guessous for integrating FlashAttention to PyTorch. FlashAttention-3 has benefited from helpful discussions with Horace He on different attention variants, with Hao Liu and Phil Wang on distributed attention, and with Daniel Haziza and Chris De Sa on quantization. We thank Meta, Together AI, and Princeton Language and Intelligence (PLI) for compute support.

<span id="section-6"></span>

## 6 Related Work

**Attention variants and distributed attention.** Ever since attention became popular with the Transformer architecture [Vas17], there has been a large body of work on approximating attention to scale it to longer sequences. These approximation methods can generally be categorized into two classes: sparse and low-rank. Sparse attention only computes some entries of the attention matrix ($\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)$) and assumes that other entries are zero. Different methods have different ways of choosing which entries should be zero, either with a fixed pattern [Chi19], with a sliding window [Bel20a], or with a dynamic pattern through hashing [Kit20] or routing [Roy21]. The low-rank approach instead assumes that the attention matrix has a low-rank structure, and apply a pointwise nonlinearity to the query and key [Kat20] with random projection [Cho20a, Pen21, Xio21]. One can also combine the sparse and low-rank approximation for better quality [Zah20, Che21b]. However, these approximation methods typically do not offer the same model quality as standard attention [Tay20a], and so most large-scale models do not employ these techniques.

There are other variants of attention aimed at reducing the size of the KV cache to improve inference efficiency. Multi-query attention [Sha19] and grouped query attention [Ain23a] tie different heads of $\mathbf{K}$ and $\mathbf{V}$, and multiple query heads interact with the same key and value head. Multi-head latent attention [Dee24] parameterizes the $\mathbf{K}$ and $\mathbf{V}$ as low-rank projections of a shared matrix to further reduce the KV cache size. However, all of these approaches do not change the core computation $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ during training and simply change how $\mathbf{Q},\mathbf{K},\mathbf{V}$ are obtained. As a result, any efficiency or accuracy improvement to the standard attention computation benefits these methods.

To extend to even longer context, attention computation can be distributed across multiple GPUs. Methods such as Ring attention [Liu23, Liu24l] and variants [Bra23] can reach a context length of up to 1 million. They use FlashAttention (or FlashAttention-2) as a primitive, and so the improvement from FlashAttention-3 would benefit these distributed attention methods as well.

**Alternative architectures.** Motivated by the limitations of attention, a variety of alternative architectures have been proposed. They build on the connection between linear attention [Kat20] and recurrent neural networks (RNNs). RWKV [Pen23b], H3 [Dao22g], MEGA [Ma23b], Retnet [Sun23b] enhance the expressivity of the simple cumulative sum in linear attention with more sophisticated recurrences. Mamba [Gu23] and xLSTM [Bec24] use learnable weighting for the recurrence and can match the quality of Transformers in language modeling at small or medium scale. These approaches can be connected to generalizations of linear attention through the lens of the structure of the token-mixing matrix [Dao24]. These models have started to see some traction, seeing usage in some medium to large-scale models such as Jamba [Jam24], Zamba [Zam24], Megalodon [Ma24e], and Mamba2-hybrid [Wal24]. For the highest quality, these SSM- and RNN-based models still employ many layers of attention. We expect that techniques to speed up attention presented in this work will be useful to speedup these alternative architectures.

**Low-precision attention.** Quantization is a promising approach to speed up attention, but they have mostly focused on reducing the space for KV cache for inference efficiency. QuIP [Che24b] and QuIP#[Tse24] use incoherent processing to reduce the quantization, and we adapted this technique for FP8 FlashAttention-3. Recent work suggests that for inference the KV cache is highly compressible down to 4-, 3-, or even 2-bits [Hoo24, Liu24c]. However, quantization during training is still challenging as higher precision is typically required for stable training.

**Hardware-aware Algorithms.** Our work presented in this paper focuses on the micro-architecture specific tuning to leverage new instruction sets and adopt a natively asynchronous programming model. There are other orthogonal axes for hardware-aware algorithm co-design being explored. A recent example of this is LeanAttention [San24a], which recognizes the poor GPU occupancy and high memory bandwidth requirements of the sequential token generation phase as primary bottlenecks for inference and optimizes it via a smarter load balancing strategy similar to Stream-K load balancing [Osa23] to achieve nearly peak occupancy. There is a large literature on optimizing GEMM for specific hardware that employs many of the same techniques. As an example, [Abd16] presents a high performance batched GEMM kernel on K40c Graphics Processing Units (GPU) for both fixed and variable sizes, proposing specialized GEMM designs and a comprehensive autotuning process to deliver state-of-the-art performance.

<span id="section-7"></span>

## 7 Addition Details on Algorithms

<span id="section-7-1"></span>

### 7.1 Asynchrony Through Warp Specialization for the Backward Pass

Similar to the forward pass [Section 3.1](#section-3-1), we use warp specialization to handle asynchrony. Instead of just a simple producer-consumer pattern in the forward pass, we add one extra role of a $\mathbf{dQ}$ writer, since we need to accumulate the value of $\mathbf{dQ}$ produced by each thread block to the global value of $\mathbf{dQ}$. This $\mathbf{dQ}$ accumulation introduces memory contention (many thread blocks writing to the same location) so having a separate warp to handle this (along with asynchrony) will avoid blocking the rest of the warps in the thread block to perform the next computation (matmul).

We include the backward pass with warp specialization in [Algorithm 3](#algorithm-03).

<span id="algorithm-03"></span>

**Algorithm 3: FlashAttention-3 backward pass with warp specialization.**

- **Require:** Matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N \times d}$ in HBM, logsumexp vector $L \in \mathbb{R}^N$ in HBM, block sizes $B_c$, $B_r$.
- In a preprocessing kernel, compute $D = \mathrm{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^d$ (pointwise multiply), write $D$ to HBM and divide it into $T_r$ blocks $D_1, \dots, D_{T_r}$ of size $B_r$ each.
- Divide $\mathbf{Q}$ into $T_r = \left\lceil\frac{N}{B_r} \right\rceil$ blocks $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_r}$ of size $B_r \times d$ each, and divide $\mathbf{K}, \mathbf{V}$ in to $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ and $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$, of size $B_c \times d$ each.
- Divide $\mathbf{dO}$ into $T_r$ blocks $\mathbf{dO}_i, \dots, \mathbf{dO}_{T_r}$ of size $B_r \times d$ each, and divide $L$ into $T_r$ blocks $L_i, \dots, L_{T_r}$ of size $B_r$ each.
- Initialize pipeline object to manage barrier synchronization with $s$-stage circular SMEM buffer.
- **If** in producer warpgroup:
  - Deallocate predetermined number of registers.
  - Issue load $\mathbf{K}_j$ and $\mathbf{V}_j$ from HBM to shared memory.
  - Upon completion, commit to notify consumer of the load of $\mathbf{K}_j$ and $\mathbf{V}_j$.
  - **For** $1 \le i \leq T_r$:
    - Wait for the $(i\,\%\,s)$th stage of the buffer to be consumed.
    - Issue loads of $\mathbf{Q}_i, \mathbf{dO}_i$ from HBM to shared memory at the $(i\,\%\,s)$th stage of the buffer.
    - Upon completion, commit to notify consumers of the loads of $\mathbf{Q}_i, \mathbf{dO}_i$.
- **Else if** in consumer warpgroups:
  - Reallocate predetermined number of registers as function of number of consumer warps.
  - On-chip, Initialize $\mathbf{dK}_j = (0)_{B_c \times d}, \mathbf{dV}_j = (0)_{B_c \times d}$.
  - Wait for $\mathbf{K}_j$ and $\mathbf{V}_j$ to be loaded in shared memory.
  - **For** $1 \le i \leq T_r$:
    - Wait for $\mathbf{Q}_i$ to be loaded in shared memory.
    - Load $L_i, D_i$ from HBM to on-chip SRAM.
    - On chip, compute $\mathbf{S}_{i}^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
    - Wait for $\mathbf{dO}_i$ to be loaded in shared memory.
    - On chip, compute $\mathbf{dP}_{i}^{(j)} = \mathbf{dO}_{i} \mathbf{V}_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). Commit.
    - On chip, wait for $\mathbf{S}_{i}^{(j)}$, then compute $\mathbf{P}_{i}^{(j)} = \exp(\mathbf{S}_{ij} - L_{i}) \in \mathbb{R}^{B_r \times B_c}$.
    - On chip, wait for $\mathbf{dP}_i^{(j)}$, then compute $\mathbf{dS}_{i}^{(j)} = \mathbf{P}_{i}^{(j)} \circ (\mathbf{dP}_{i}^{(j)} - D_i) \in \mathbb{R}^{B_r \times B_c}$.
    - On chip, compute $\mathbf{dV}_j \leftarrow \mathbf{dV}_j + (\mathbf{P}_{i}^{(j)})^\top \mathbf{dO}_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM). Commit.
    - On chip, compute $\mathbf{dK}_{j} \leftarrow \mathbf{dK}_j + {\mathbf{dS}_{i}^{(j)}}^\top \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM). Commit and wait for both $\mathbf{dV}_j$ and $\mathbf{dK}_j$.
    - On chip, compute $\mathbf{dQ}_{i}^{(\mathrm{local})} = \mathbf{dS}_{i}^{(j)} \mathbf{K}_j \in \mathbb{R}^{B_r \times d}$ (SS-GEMM), and write $\mathbf{dQ}_i^{(\mathrm{local})}$ to SMEM. Notify the $\mathbf{dQ}$-writer.
- **Else if** in $\mathbf{dQ}$-writer warp:
  - **For** $1 \le i \leq T_r$:
    - Wait for $\mathbf{dQ}_i^{(\mathrm{local})}$ to be ready in SMEM.
    - Using a semaphore, atomically add $\mathbf{dQ}_i^{(\mathrm{local})}$ to $\mathbf{dQ}_i$ in global memory.

<span id="section-7-2"></span>

### 7.2 2-Stage Pipelining SASS Analysis

We give simplified SASS code for the inside of the consumer warpgroup mainloop.

```text
// Compute row_max
FMNMX.FTZ R0, R24, R6, !PT ;
SHFL.BFLY PT, R185, R2, 0x2, 0x1f ;
… FMNMX and SHFL.BFLY …

// Apply exp2 and row_sum. Rescale O.
FMUL.FTZ R2, R4, UR9 ;
MUFU.EX2 R185, R184 ;
FFMA.FTZ R24, R24, UR9, -R6.reuse ;
FADD.FTZ R24, R211, R24 ;
… FMUL, FFMA, FMUL, MUFU.EX2, FADD …

// FP32 -> FP16 conversion are interleaved with exp2, row_sum and O rescaling.
F2FP.F16.F32.PACK_AB R231, R25, R231 ;
… F2FP, FMUL, MUFU, FFMA, FADD ...

// Start the first WGMMA. Broken down into 8 HGMMAs.
// The first 7 HGMMAs are packed together.
WARPGROUP.ARRIVE ;
HGMMA.64x192x16.F32 R24, gdesc[UR44], RZ, !UPT ;
... HGMMA x 6 ...

// FP32->FP16, exp2, row_sum, O rescaling are interleaved with HGMMA.
F2FP.F16.F32.PACK_AB R214, R214, R187 ;
MUFU.EX2 R234, R5 ;
FADD.FTZ R237, R187, R2 ;
… F2FP, MUFU, FADD …

// The last HGMMA is issued here. No need to wait.
HGMMA.64x192x16.F32 R24, gdesc[UR44], R24, gsb0 ;

// Start the second WGMMA. Broken down into 12 HGMMAs.
// All 12 HGMMAs are packed together. Not interleaved with other instructions.
WARPGROUP.ARRIVE ;
HGMMA.64x128x16.F32 R120, R228, gdesc[UR8].tnspB, R120 ;
... HGMMA x 10 ...
HGMMA.64x128x16.F32 R120, R184, gdesc[UR8].tnspB, R120, gsb0 ;

// wgmma.wait_group at the end.
WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
```

We make the following observations:

- Softmax is reordered to the very beginning, even before the first WGMMA.
- The first WGMMA is interleaved with softmax and FP32 $\rightarrow$ FP16 datatype conversion of $\mathbf{S}$. This indicates that WGMMA and non-WGMMAs are executed in parallel.
- `exp2`, `row\_sum`, O rescaling and FP32 $\rightarrow$ FP16 conversions are interleaved together.
- The second WGMMA is not overlapped with other instructions, as expected.

Overall, SASS shows that the 2-stage pipelining idea works as expected.

<span id="section-7-3"></span>

### 7.3 3-Stage Pipelining Algorithm

We experiment with a 3-stage pipelining algorithm to parallelize the first WGMMA from iteration $j+2$, softmax from iteration $j+1$, and the second WGMMA from iteration $j$. We describe this algorithm in [Algorithm 4](#algorithm-04). This algorithm behaves worse than the 2-stage pipelining algorithm due to the reasons below:

<span id="figure-08"></span>

![Figure 8. 3-Stage Pipelining](../../papers/flashattention-3/figure-08.png)

**Figure 8.** 3-Stage Pipelining

<span id="algorithm-04"></span>

**Algorithm 4: FlashAttention 3-stage pipelining consumer warpgroup forward pass.**

- **Require:** Matrices $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ in HBM, block sizes $B_c$, $B_r$. Each warpgroup reads 1 block $\mathbf{Q}_i$ of size $B_r \times d$, $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ blocks $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ and $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$ of size $B_c \times d$. Each warpgroup writes 1 output block $\mathbf{O}_i$ of size $B_r \times d$, and 1 logsumexp block $L_i$ of size $B_r$.
- Initialization. Load $\mathbf{Q}_i$ from HBM to on-chip SRAM. Initialize $\mathbf{O}_i, \ell_i, m_i, \mathrm{scale}_o$.
- Wait for the producer warpgroup loading $\mathbf{K}_0$ from HBM to on-chip SRAM.
- Compute $\mathbf{S} = \mathbf{Q}_i \mathbf{K}_0^\top$ using WGMMA. Commit and wait.
- Compute $m_i$, $\widetilde{\mathbf{P}}_i$, $\ell_i$, $\mathrm{scale}_o$ based on $\mathbf{S}$.
- Wait for the producer warpgroup loading $\mathbf{K}_1$ from HBM to on-chip SRAM.
- Compute $\mathbf{S} = \mathbf{Q}_i \mathbf{K}_1^\top$ using WGMMA. Commit and wait.
- **For** $2 \le j < T_c - 2$:
  - Wait for the producer warpgroup loading $\mathbf{K}_j$ from HBM to on-chip SRAM.
  - Compute $\mathbf{S}_{\mathrm{next}} = \mathbf{Q}_i \mathbf{K}_{j}^\top$ using WGMMA. Commit but do not wait.
  - Wait for the producer warpgroup loading $\mathbf{V}_{j-2}$ from HBM to on-chip SRAM.
  - Rescale $\mathbf{O}_i$ based on $\mathrm{scale}_o$.
  - Compute $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{j-2}$ using WGMMA. Commit but do not wait.
  - Compute $m_i$, $\widetilde{\mathbf{P}}_{i,\mathrm{next}}$, $\ell_i$, $\mathrm{scale}_o$ based on $\mathbf{S}$.
  - Wait for all previous WGMMAs.
  - Copy $\mathbf{S}_{\mathrm{next}}$ to $\mathbf{S}$.
  - Copy $\widetilde{\mathbf{P}}_{i,\mathrm{next}}$ to $\widetilde{\mathbf{P}}_i$.
- Wait for the producer warpgroup loading $\mathbf{V}_{T_c-2}$ from HBM to on-chip SRAM.
- Rescale $\mathbf{O}_i$ based on $\mathrm{scale}_o$.
- Compute $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{T_c-2}$ using WGMMA. Commit and wait.
- Compute $m_i$, $\widetilde{\mathbf{P}}_i$, $\ell_i$, $\mathrm{scale}_o$ based on $\mathbf{S}$.
- Wait for the producer warpgroup loading $\mathbf{V}_{T_c-1}$ from HBM to on-chip SRAM.
- Rescale $\mathbf{O}_i$ based on $\mathrm{scale}_o$.
- Compute $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{T_c-1}$ using WGMMA. Commit and wait.
- Epilogue. Rescale $\mathbf{O}_i$ based on $\ell_i$. Compute $L_i$ based on $\ell_i$ and $m_i$. Write $\mathbf{O}_i$ and $L_i$ to HBM as the $i$th block of $\mathbf{O}$ and $L$.

**Overlapping.** We expected that softmax can be overlapped with (the first WGMMA + the second WGMMA). However, the compiler doesn’t cooperate in this way. SASS code shows that only the first WGMMA is overlapped with softmax, while the second WGMMA is not. It’s not clear why the compiler chooses to reorder instructions in this way.

**Register pressure.** This algorithm requires more registers compared to the 2-stage pipelining algorithm. In theory, it needs to store an extra $\tilde{\mathbf{P}}_{i}$ and $\mathrm{scale}_o$, which is of size $B_{r}\times B_{c}\times\text{sizeof}(\text{input\_data\_type})+B_{r}\times\text{sizeof}(\text{float})$. As a result, a smaller block size needs to be chosen.

<span id="section-8"></span>

## 8 Addition Details on Experiments and Benchmarking

<span id="section-8-1"></span>

### 8.1 System and libraries

We benchmark the speed on an H100 80GB SXM5 (700W). We generally use the latest versions of the libraries, at the time of writing (May 2024). Specifically, we use:

- CUDA 12.3
- cuDNN 9.1.1.17
- CUTLASS 3.5
- FlashAttention 2.5.8
- Triton nightly 3.0.0.post20240424212437
- PyTorch 2.3.0

To reduce variability, we fix the GPU clock speed to 1830MHz (clock speed used to calculate the 989 TFLOPS FP16 theoretical max throughput). We repeat the benchmarks 100 times and take the average timing.

<span id="section-8-2"></span>

### 8.2 FP8 Attention Full Results

We use following sequence lengths: 512, 1024, 2048, 4224, 8448, 16896. When sequence length $\geq$ 4k, we make it also divisible by 132 (number of SMs in H100 SXM5) to avoid wave quantization.

<span id="figure-09"></span>

![Figure 9. Attention forward speed (FP8) on H100 GPU](../../papers/flashattention-3/figure-09.png)

**Figure 9.** Attention forward speed (FP8) on H100 GPU

[+1]: We describe our results in the context of NVIDIA’s Hopper architecture. However, our algorithm is operative for any GPU architecture with sufficiently robust asynchronous execution and low-precision capabilities.

[+2]: More precisely, for head dimension 64 FlashAttention-3 FP8 is ahead, while for head dimensions 128 and 256 it is at par for those cases without causal masking and behind with causal masking.

[+3]: FlashAttention-3 is available at [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

[+4]: [Luo24b] reports shared memory bandwidth of 128 bytes per clock cycle per SM, and we multiply that by 132 SMs and the boost clock of 1830 MHz.

[+5]: The CUDA programming guide specifies that 16 operations of special functions can be performed per streaming multiprocessor (SM) per clock cycle. We multiply 16 by 132 SMs and 1830 MHz clock speed to get 3.9 TFLOPS of special functions.

[+6]: Note that the number of stages of the overlapping scheme is bounded by, but need not equal, the number $s$ of stages in the circular SMEM buffer.

[+7]: An optimized transpose kernel will achieve speed near the bandwidth of the device [Cut24].

[+8]: In the PTX documentation, LDSM/STSM are described as copying $8\times 8$ matrices with 16-bit entries [Ptx24], but we can pack 8-bit entries two at a time to use LDSM/STSM in the context of FP8 precision. However, the transpose versions of LDSM/STSM cannot split packed 8-bit entries, which necessitates certain register movements in between LDSM and STSM to actually perform a tile-wise transpose; we omit the details.

[+9]: This additional freedom afforded by doing the in-kernel transpose eliminates having to use shuffle instructions to change register ownership across threads, which we previously described in [Bik24].

[+10]: For our benchmarks, FP16 FlashAttention-3 has a persistent kernel and load balancing strategy, while FP8 FlashAttention-3 does not. This partly explains why FP8 FlashAttention-3 does not perform as well for small sequence length and causal masking compared to the FP8 cuDNN kernels.

[+equal]: Equal contribution.
