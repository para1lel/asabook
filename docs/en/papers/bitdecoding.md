---
title: 'BitDecoding'
createTime: 2026/08/23 12:32:45
permalink: /en/papers/bitdecoding/
pageClass: paper-reading
---

> [Dayou Du](https://openalex.org/A5113196534) [+internship], [Shijie Cao](https://openalex.org/A5140976747) [+corresponding], [Jianyi Cheng](https://openalex.org/A5010206458), [Luo Mai](https://luomai.github.io/), [Ting Cao](https://openalex.org/A5061455084), and [Mao Yang](https://dblp.org/pid/89/1482-4.html). First submitted to arXiv on March 24, 2025; current version v3. This reading edition transcribes [BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache](https://arxiv.org/abs/2503.18773v3). <a href="/paper/bitdecoding.pdf" target="_blank">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2503.18773). [TeX source](https://export.arxiv.org/e-print/2503.18773v3). The original PDF remains authoritative for the exact print layout and bibliography.

[+internship]: Work partially done during an internship at Microsoft Research.

[+corresponding]: Corresponding author.

## Abstract

The growth of long-context Large Language Models (LLMs) significantly increases memory and bandwidth pressure during autoregressive decoding due to the expanding Key-Value (KV) cache. While accuracy-preserving KV-cache quantization (e.g., 4-bit or 2-bit) reduces memory footprint, existing systems decode inefficiently by relying solely on CUDA cores, underutilizing Tensor Cores—the dominant compute resource on GPUs.

We present BitDecoding, the first inference system to efficiently decode low-bit KV caches by cooperatively leveraging CUDA cores and Tensor Cores. BitDecoding smartly induces Tensor-Core-friendly layouts, introduces warp-level dequantization parallelism, and provides unified system support through query transformation, high-performance tensor- and channel-wise quantization, and a software-pipelined dequantization kernel enabling mixed-precision execution. Architecture-aware optimizations further leverage Hopper's warpgroup tensor instructions and Blackwell's NVFP4 (MXFP4) tensor formats.

Evaluated on Blackwell, Hopper, and Ampere GPUs, BitDecoding achieves an average 7.5$\times$ decoding speedup over FP16 FlashDecoding-v2, up to 8.6$\times$ on Blackwell with NVFP4, and up to 4.3$\times$ over state-of-the-art approaches. On LLaMA-3.1-8B with a 128K context, BitDecoding reduces single-batch decoding latency by 3$\times$. BitDecoding is open-sourced at [https://github.com/OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding).

<span id="section-1"></span>

## 1 Introduction

The ability of Large Language Models (LLMs) to process **long contexts** [Pen23, Din24d, Tea24a] has unlocked new capabilities, such as book summarization [Cha23c], multi-modal understanding [Yan23d], and test-time scaling [Dee25c, Ope25k]. However, these advancements come with significant memory and computational challenges, primarily due to the large size of the Key-Value (KV) cache in long-context scenarios. During autoregressive decoding, LLMs must repeatedly access this growing cache for each generated token, which increases memory usage and slows down decoding. The problem worsens with larger batch sizes, as the KV cache scales linearly with the number of concurrent queries. For example, a 7B model requires approximately 14GB GPU memory for its parameters, but with a 32K context length and a batch size of 8, the KV cache alone consumes 128GB GPU memory [Hoo24], creating a significant memory bottleneck.

To address this growing bottleneck, **KV cache quantization** has emerged as a promising solution. By reducing the bit-width of the KV cache, quantization lowers memory overhead and improves overall efficiency. Recent quantization algorithms have shown that low-bit KV cache can retain high accuracy.

QServe [Lin24a] demonstrates 4-bit KV cache improves throughput on models like LLaMA-3 and Qwen-1.5 while maintaining strong accuracy, even together with 4-bit weight and 8-bit activation.

Further research [Liu24c, Kan24, Su25c] shows that 2-bit KV cache can achieve near fp16 accuracy.

Kivi [Liu24c], for instance, incurs only a 0.6% accuracy drop on LongBench [Bai23] with a 2-bit KV cache on LLaMA-2-7B-Chat.

Recent studies [Zha24b, Tao24a] explore 1-bit quantization for KV cache, maintaining acceptable accuracy under specific conditions.

These results confirm that KV cache quantization strikes an effective balance between efficiency and accuracy, making it viable for long-context LLM deployment.

*Despite the memory savings, current system support for low-bit KV cache struggles to deliver the expected speedup.* Previous implementations [Liu24c, Zha24e, Lin24a] remain preliminary and case-specific, with significant room for further systematic optimization. A major bottleneck lies in the overhead introduced by quantization and dequantization. Although the KV cache is low-bit, the query (Q) values and attention scores remain in high precision. This results in mixed-precision matrix multiplications (mpGEMM), which existing hardware does not natively support, requiring dequantization before multiplication. Previous mpGEMM kernels like Ladder [Wan24e] and Marlin [Fra24] are designed for low-bit weights but cannot be directly applied to low-bit KV caches. This is because weights are *static and stored offline*, while KV caches are *dynamic and generated online*. In autoregressive decoding, each newly generated token requires quantization, packing, and dequantization of the low-bit KV cache, introducing significant overhead and complexity in GPU kernel design, as illustrated in [Figure 1](#figure-01).

To address this, our insight is to leverage Tensor Cores for intensive matrix multiplications while efficiently utilizing CUDA cores for KV cache dequantization. Previous work either implemented with separated kernels or fused attention operations relied solely on CUDA cores, leaving Tensor Cores underutilized, as shown in [Figure 2](#figure-02). Our approach is based on three key observations: First, modern language models employ Grouped-Query Attention (GQA) and Multi-Query Attention (MQA), which share a group of keys across multiple queries, enabling Tensor Cores to accelerate dot products in the self-attention mechanism. Second, leveraging Tensor Cores can alleviate computational pressure on CUDA cores, enabling more efficient execution of low-bit operations. Finally, newer GPU architectures provide distinct mechanisms: Hopper's support for asynchronous execution and warp specialization allows low-bit operations to overlap with computation [Luo24b], while Blackwell's native support for low-precision formats (e.g., MXFP4) reduces these overheads by minimizing the need for on-the-fly data conversion.

<span id="figure-01"></span>

![Comparison of mixed-precision matrix multiplication for low-bit weight and low-bit KV cache.](../../papers/bitdecoding/figure-01.png)

**Figure 1.** Comparison of mixed-precision matrix multiplication for low-bit weight and low-bit KV cache. (a) Quantized weights can be preprocessed offline. (b) KV cache requires online quantization and packing for each newly generated token.

Efficiently leveraging Tensor Cores for decoding with low-bit KV caches poses significant challenges. First, Tensor Cores require dequantized low-bit data to be aligned with high-precision formats, which is difficult in autoregressive decoding as the KV cache grows dynamically and must conform to Tensor Cores-specific layouts. Without optimized layouts, Tensor Cores may exhibit poor utilization or even produce incorrect results. Second, the high cost of dequantization can stall Tensor Cores execution, reducing GPU occupancy due to mismatched workloads between CUDA cores and Tensor Cores. Third, supporting low-bit KV caches across diverse attention mechanisms and quantization algorithms—with varying tensor-wise and channel-wise scaling—demands a general yet highly optimized implementation. Without careful design, either CUDA cores or Tensor Cores become performance bottlenecks during long-context generation.

To address the above challenges, we have designed and implemented **BitDecoding**, a high-performance long-context LLMs inference system with low-bit KV cache. The design of BitDecoding delivers several contributions essential for exploiting Tensor Cores, including: (i) inducing low-bit optimized layouts based on hardware instructions, (ii) aligning warps with residual buffer to saturate Tensor Cores, (iii) re-mapping layouts for faster dequantization, and (iv) coordinating kernels for quantization and dequantization. In addition, we contribute new strategies for parallelizing GPU warps to mitigate low-bit operations overhead, including (i) efficient warp parallelism layout, and (ii) enhancing attention algorithms for fast warp synchronization leveraging the GPU memory hierarchy.

We further contribute implementation techniques in BitDecoding for LLMs inference, including: (i) a query transformation approach that enables efficient execution of diverse attention variants, allowing BitDecoding to be easily adopted in existing LLMs; (ii) a high-performance quantization kernel that supports both channel-wise and tensor-wise scaling, ensuring generality across quantization algorithms; and (iii) a dequantization kernel with a software-defined pipeline that coordinates CUDA and Tensor Cores for GEMM and dequantization, while overlapping data movement, including extra low-bit metadata; furthermore, BitDecoding incorporates architecture-specific optimizations that unlock Hopper's warpgroup tensor operations and Blackwell's native low-precision tensor formats to maximize decoding performance on the latest GPU generations.

BitDecoding is evaluated at both the kernel and end-to-end levels across Blackwell, Hopper, Ada, and Ampere GPU architectures. At the kernel level, it outperforms FP16 FlashDecoding-v2 by up to 8.6$\times$ on Blackwell (e.g., RTX 5090, using native MXFP4 format support), 8.0$\times$ on Hopper, 7.5$\times$ on Ada, and 4.8$\times$ on Ampere, while surpassing QServe by up to 4.3$\times$. At the end-to-end model level, BitDecoding reduces single-batch decoding latency by 3$\times$ on LLaMA-3.1-8B with a 128K sequence length and achieves over 4$\times$ higher serving throughput than QServe.

<span id="section-2"></span>

## 2 Background and Motivation

<span id="figure-02"></span>

![Comparison of different low-bit KV cache systems against half-precision FlashAttention.](../../papers/bitdecoding/figure-02.png)

**Figure 2.** Comparison of different low-bit KV cache systems against half-precision FlashAttention. Each system follows the attention formulation $\mathrm{Out}=\mathrm{softmax}(Q\,\mathcal{D}(K'^\top))\,\mathcal{D}(V')$, where $K'$ and $V'$ are low-bit quantized Key and Value tensors, and $\mathcal{D}(\cdot)$ denotes the dequantization function.

**LLMs inference and low-bit KV cache.** LLMs inference comprises two stages: (i) *Prefill*, which processes the prompt and computes Key (K) and Value (V) tensors for caching; and (ii) *Decode*, which updates the KV cache token-by-token for autoregressive generation. For a model with $n$ layers, $h_{kv}$ KV heads, and hidden size $d$, the KV cache requires $2 \cdot 16 \cdot n \cdot h_{kv} \cdot d \cdot b \cdot l$ bits (assuming FP16), where $b$ is the batch size and $l$ is the sequence length. Because this requirement grows linearly with both $b$ and $l$, the KV cache often dominates memory usage, especially for long-context and large-batch workloads. In batched inference, each sequence has an independent past context, so there is little batch-level parallelism or reuse when loading cached Keys and Values; *consequently, KV-cache access is typically bound by memory bandwidth*. These constraints have spurred extensive research and industrial efforts on lower-bit KV caches [Liu24c, Hoo24, Zha24b] to reduce memory footprint and improve throughput while preserving accuracy close to non-quantized baselines.

**Tensor Cores and CUDA cores on modern GPUs.** When optimizing LLM inference and low-bit KV caches on GPUs, it is crucial to exploit both Tensor Cores and CUDA cores. Tensor Cores deliver the majority of compute FLOPS in modern GPUs but are specialized for matrix operations (e.g., GEMM), whereas CUDA cores provide more flexible vector, scalar, and control-flow capabilities at substantially lower peak FLOPS. For example, on the A100, Tensor Cores deliver up to 312 TFLOPS in FP16/BF16—far exceeding the 19.5 TFLOPS FP32 offered by CUDA cores.

This performance gap has widened significantly in recent generations. The Hopper architecture introduces Warpgroup Matrix Multiply-Accumulate (WGMMA) instructions and warp-specialized pipelines to maximize asynchronous execution efficiency. The Blackwell architecture further exacerbates this disparity by supporting native micro-scaling formats (e.g., MXFP4, NVFP4), delivering up to 20 PFLOPS.

For fast LLM inference, substantial effort has gone into optimizing attention variants to exploit Tensor Cores. SOTA LLMs [Dee24a, Dub24, Yan25a] increasingly adopt MQA [Sha19] and GQA [Ain23], which reduce memory bandwidth by reusing KV heads across multiple queries. This reuse increases arithmetic intensity and improves compute efficiency [Sun25e], aligning well with the high-throughput, matrix-centric design of Tensor Cores. Consequently, leveraging Tensor Cores is becoming essential for efficient inference in long-context and grouped-attention LLMs.

**Limitations of existing low-bit KV cache systems.** To support low-bit KV caches for long-context LLM inference, a number of systems have been proposed [Liu24c, Lin24a, Zha24e]. However, they often leave GPUs underutilized, leading to suboptimal performance. We summarize the key reasons below.

- *Attention with separated low-bit KV-cache kernels:* The most straightforward approach, exemplified by Kivi [Liu24c], decomposes mixed-precision attention into multiple standalone kernels and embeds them in a non-fused attention implementation. This design is highly flexible and readily supports many attention variants [Ain23, Sha19]. Yet the isolated launches repeatedly load and store intermediate data, inflate global-memory traffic, and break on-chip data reuse. The result is high launch overhead, increased memory bandwidth pressure, and lower effective throughput.
- *Fused attention with low-bit KV-cache kernels on CUDA cores solely:* Given the generality of CUDA cores for mixed-precision operations, a natural extension of FlashAttention-style fusion [Dao24a] is a CUDA-cores-only implementation of low-bit KV caches. While this outperforms non-fused designs, it still underutilizes Tensor Cores. In these systems, both dequantization and matrix operations (GEMV/GEMM) are executed on CUDA cores via fused multiply-add (FMA) instructions. Under mixed precision, CUDA cores must handle expensive dequantization (e.g., int4/8 $\rightarrow$ FP16/BF16), scaling, and element-wise ops—tasks that are memory-bound and consume instruction slots, register bandwidth, and L1/L2 capacity. This reduces occupancy and limits tile sizes, leaving fewer resources for the compute-heavy matrix multiplications. Consequently, running both dequantization and matmul on CUDA cores introduces significant overhead, especially for attention variants with higher arithmetic intensity.

<span id="section-3"></span>

## 3 Proposed Solutions and Challenges

<span id="section-3-1"></span>

### 3.1 Solution: Cooperative use of Tensor Cores & CUDA Cores

In this paper, we want to explore a solution that can achieve a *cooperative* use of Tensor Cores and CUDA cores to support low-bit KV caches during long-context LLMs inference. Our design introduces new designs and implementations that (i) construct and schedule matrix multiplications on Tensor Cores, and (ii) execute non-matrix-multiplication operations—quantization, packing and dequantization—efficiently on CUDA cores. To make this cooperation effective, we balance workloads across the Tensor Cores and CUDA cores and carefully orchestrate data movement so that dequantization feeds Tensor-Core GEMM without stalls, memory traffic is minimized, and end-to-end decoding throughput is maximized.

To ensure broad adoption, we aim to realize this cooperative design as a system that (i) supports low-bit KV caches across multiple attention variants (including MHA, MQA, and GQA), and (ii) spans multiple GPU generations. The former requires a clean interface that integrates with existing attention implementations; the latter requires designs that are easy to adapt, enabling rapid targeting of different GPU backends while sustaining high decoding throughput.

We expect significant benefits from this proposed solution. For example, by enabling low-bit decoding that builds on FlashAttention-3 (FA-3) [Sha24b], we can leverage SM90-specific features—such as warp-specialized pipelines—that yield up to $6\times$ speedups over prior implementations, avoiding the 35% throughput penalty associated with legacy SM80 instructions. Furthermore, this design anticipates the architectural capabilities of Blackwell, where native support for low-precision formats will drive even more substantial throughput improvements.

<span id="section-3-2"></span>

### 3.2 Open challenges

Although promising, the *cooperative* use of Tensor Cores and CUDA cores for low-bit KV caches is particularly challenging to implement for several reasons:

<span id="figure-03"></span>

![FP16 and INT4 fragment layouts illustrating a low-bit layout mismatch.](../../papers/bitdecoding/figure-03.png)

**Figure 3.** (a) `mma.m16n8k16` fragment layout for matrix B. Each thread ($T_i$) is assigned a specific set of values based on the instruction-defined interleaved mapping. (b) For INT4, quantization packs values contiguously per thread. After dequantization, the layout misaligns with the expected interleaved pattern.

**Challenge 1: Tensor Cores often suffer from low-bit layout mismatches.** Aligning low-bit data layouts with Tensor Cores requirements is difficult, especially in autoregressive generation where KV caches expand dynamically.

At runtime, after quantization and packing, the low-bit KV cache must dequantize into a half-precision layout that matches what Tensor Cores expect. This matching is challenging for three reasons.

First, fragment layouts vary across instructions and GPU generations. After using the optimized data-movement instruction `ldmatrix`, the fragment residing in registers enforces a strict value-to-thread mapping. [Figure 3a](#figure-03) illustrates the registers read by each thread ($T$) for `mma.m16n8k16` with repeat tiling along the $N$ dimension. However, this mapping differs from other Tensor Core instructions (e.g., `mma.m16n8k8`) and from Hopper's `wgmma` family (e.g., `wgmma.m64n64k16`).

Second, low-precision bitwidths exacerbate alignment issues. Although Tensor Cores instructions require specific compute types, their rigid, interleaved register layout makes lower-precision data hard to match directly. Without a layout transform, the low-bit register layout becomes an **invalid layout** for MMA execution due to misalignment with the interleaved access patterns. As shown in [Figure 3b](#figure-03), two FP16 values originally computed by Thread 0 (T0) may be quantized and packed as eight consecutive low-bit values in the KV cache; after unpacking and dequantization, they no longer align with the expected Tensor Core register layout, yielding incorrect values. Even with native low-precision formats in Blackwell, hardware support remains limited, especially for the KV cache, which still depends on continuous quantization and packing; software must therefore carefully handle low-precision values and micro-scaling factors [Nvi25e].

Finally, dequantization can bottleneck execution: naive low-bit$\rightarrow$FP16 casts are slow [Kim22] and require a **friendly layout** to run efficiently. Prior work such as Ladder [Wan24e] and Marlin [Fra24] mitigates mismatch for static weights by inserting separate layout-transformation kernels, but this adds substantial overhead and is unsuitable for dynamic decoding. Experimental details are given in [Table 2](#table-02).

<span id="figure-04"></span>

![Original warp design and a micro-level comparison with and without dequantization.](../../papers/bitdecoding/figure-04.png)

**Figure 4.** (a) A single warp along $N$ for register-level operations will experience stalls due to dequantization (DQ) (b) Micro-level comparision with and without dequantization.

**Challenge 2: Frequent stalls limit Tensor Cores utilization.** We observe that empirically tuned warp layouts and partitioning in high-performance attention kernels often inadvertently degrade low-bit KV-cache performance.

Under FlashAttention's original warp partitioning, the additional dequantization (DQ) can substantially reduce throughput and Tensor Core utilization. As shown in [Figure 4a](#figure-04), FlashAttention assigns a single warp along the $N$ dimension to perform register-level softmax and the matrix multiplication $P V$, with $P$ stored in registers aligned to the Tensor Core layout. When DQ is inserted before the matmul, this strategy becomes inefficient: small warp tiles of $K$ or $V$ must traverse $N$ sequentially, so DQ frequently stalls the warp. Nsight Compute profiling [Nvi25a] in [Figure 4b](#figure-04) confirms that the added DQ overhead increases memory-access stalls and depresses compute throughput and Tensor Cores utilization, consistent with prior observations [Fan25e].

Furthermore, native low-precision formats introduce their own overhead despite eliminating dequantization. Specifically, to utilize low-precision Tensor Cores for the second matrix multiplication ($P V$), the probability matrix $P$ must be dynamically re-quantized after the softmax operation: $P_{f16}=\mathrm{softmax}(Q_{f4}K_{f4}^\top), \quad O_{f16}=\mathrm{Quant}(P_{f16})V_{f4}$. This on-the-fly quantization creates a new computational bottleneck that can similarly stall Tensor Cores execution.

**Challenge 3: Lack of generalizable system optimizations for different low-bit KV-cache methods.** Popular KV-cache quantization methods use diverse scaling granularities for the Key tensor—tensor-wise [Zha24e, Hoo24] and channel-wise [Liu24c, Kan24]—which complicates building a unified system that supports them all. Online quantization and packing require reductions and element-wise transforms, adding nontrivial runtime overhead. Moreover, auxiliary metadata (scale and zero-point) increases memory traffic and, without careful scheduling, disrupts the load-compute pipeline. Prior mixed-precision kernel optimizations [Wan24e, Fra24] target static weight quantization and do not generalize to the dynamic, step-by-step nature of KV caches. To date, generalizable system-level optimization techniques for high-performance, low-bit KV-cache quantization are lacking.

<span id="section-4"></span>

## 4 BitDecoding Design

In this section, we present the design of BitDecoding system which realizes the cooperative use of Tensor Cores and CUDA cores in supporting low-bit KV cache. The design primarily contains (i) new methods and principles for optimizing the low-bit layout in using Tensor Cores, and (ii) new strategies for parallelizing and coordinating GPU warps that can minimize the stalls due to dequantization.

<span id="figure-05"></span>

![Overview of methods for optimizing low-bit layout on Tensor Cores.](../../papers/bitdecoding/figure-05.png)

**Figure 5.** Overview of methods for optimizing low-bit layout on Tensor Cores. (1) Fused computation and quantization within Tensor Cores fragments. (2) The low-bit packing data preserves FP16 values. (3) Low-bit Layout matches with the dequantized half-precision layout. (4) Layout remapping for faster dequantization.

<span id="section-4-1"></span>

### 4.1 Methods for optimizing low-bit layout on Tensor Cores

The first challenge our design aims to address is to ensure BitDecoding can automatically generate an optimized layout that can fully utilize Tensor Cores across different GPU generations and different configurations of the low-bit KV caches. For this, we have designed the following principles and methods:

**(1) Inducing low-bit optimized layout with hardware instructions.** Our design is motivated by a novel insight: the thread-to-register mapping of `ldmatrix` loads data in Tensor Core's interleaved fragment layout. As shown in [Figure 5](#figure-05)-(2), if each thread then quantizes and packs locally, the resulting low-bit packing *implicitly preserves* the half-precision (FP16) interleaved layout. On unpacking and dequantization, values already match Tensor Core registers—no global reshape is required. Thus, rather than relying on heavyweight global transforms via manual implementations [Fra24] or iterative search [Wan24e] as in prior methods, we use hardware instructions to automatically induce a valid low-bit packing layout while computing. This yields zero-overhead remapping that is efficient, compatible with Tensor Cores execution, and avoids extra data movement.

Building on this insight, we design a dedicated GPU *Residual Kernel* that fuses computation, quantization, and packing for newly generated FP16 KV tensors. Using `ldmatrix`, we load the high-precision KV tensor into registers structured for Tensor Cores, perform the matrix operation (e.g., $Q K^\top$ or $P V$), and then have each thread quantize and pack its portion in registers (see [Figure 5](#figure-05)-(1)). The result is interleaved, layout-compatible low-bit data written directly to global memory, updating the low-bit KV cache.

To consume this cache, we introduce a *Packing Kernel* that fuses dequantization with computation. To guarantee correct register layout during unpacking, it mirrors the Residual Kernel's instruction configuration which (i) uses the same `ldmatrix` variant and (ii) follows the same `mma` variant and warp-tiling configuration. Consequently, when the Packing Kernel loads packed low-bit data via `ldmatrix`, the unpacked values are inherently aligned with Tensor Core registers and can participate in matrix multiplication immediately, without explicit layout correction.

**(2) Aligning warps with residual KV cache to saturate Tensor Cores.** Tensor Cores execute warp-tiled matrix operations, which require input tiles to be fully populated to achieve optimal throughput. Based on this, *our insight* is that by allocating a residual buffer with size matching the tiling capacity of Tensor Cores, we ensure that low-bit data aligns with the compute granularity of the hardware to fully utilize the computing ability of the computing unit.

To implement this idea, we introduce a half-precision residual KV cache with a residual block size $N_r$. Let $X \in \mathbb{R}^{L \times d}$ denote the entire KV cache. We partition $X$ into:

$$
X=X_{\mathrm{pack}}\cup X_{\mathrm{res}},
$$

where

$$
\begin{cases}
X_{\mathrm{pack}}=X[:L-N_r] \\
X_{\mathrm{res}}=X[L-N_r:]
\end{cases}
$$

We define $\beta$ as the bit-width for low-bit quantization (e.g., $\beta=4$ or $2$), and $\omega$ as the word size used for packed storage (e.g., $\omega=16$ for INT16). The corresponding *packing ratio* is given by $R=\omega/\beta$. Let $W_n$ denote the number of warps along the N dimension, and $P_n$ the number of elements each warp tile processes (e.g., $P_n=8$ under `mma.m16n8k16`). To ensure each Tensor Cores fragment is fully populated for each warp, the residual block size is computed as:

<span id="equation-01"></span>

$$
N_r=P_n\times W_n\times R
$$

This guarantees that low-bit KV cache fragments align precisely with the warp-level tiling of Tensor Core operations, enabling dense, layout-compatible packing and maximizing compute unit occupancy.

**(3) Re-mapping layout for faster dequantization.** Though compatible with Tensor Cores layout, the layout is inefficient to dequantization due to directly casting low-bit values to FP16 using `static_cast` introduces significant overhead.

To mitigate this inefficiency, we further design a faster dequantization mapping approach based on low-level bitwise operations and instructions inspired by [Kim22]. After loading packed data into registers using `ldmatrix`, we cast them to INT32 before mapping them to the interleaved Tensor Core layout following the 75316420 pattern. This layout enables efficient conversion of INT4/INT2 data to FP16 using the `lop3` instruction for bitwise manipulation while aligning with the Tensor Core computation pattern.

**(4) Coordinating Residual and Packing Kernels with Configuration Setup.** This design is executed by coordinating the Residual and Packing kernels under a unified instruction configuration. First, the hardware instruction configuration—including `ldmatrix` and `mma` variants—can be determined based on GPU architectures. With this configuration, the residual block size $N_r$ is computed based on the bit-width of the low-bit KV cache. As shown in [Figure 5](#figure-05), the Residual kernel loads high-precision KV entries into registers via `ldmatrix`, performs computation using Tensor Cores, and then fuses quantization and packing before storing the results into the low-bit KV cache. The Packing kernel, using the same instruction configuration, loads the packed data into registers, performs efficient dequantization, and proceeds with Tensor Core computation.

<span id="section-4-2"></span>

### 4.2 Strategies for parallelizing warps

The second challenge is ensuring BitDecoding avoids the pitfalls of existing warp-parallelization strategies for mixed-precision attention, which suffer from low hardware utilization due to frequent warp stalls. Our key insight is that low-bit data moves at much higher bandwidth than full precision, shifting the bottleneck from memory to compute. We therefore design a warp layout that exploits the GPU memory hierarchy to parallelize low-precision operations efficiently, minimizing data movement and substantially improving Tensor Cores utilization ([Table 3](#table-03) demonstrates minimal overhead).

**(1) Enhancing warps parallelism for low-precision operations.** We introduce a novel warps layout to enable parallel operations of multiple packed data chunks. Using dequantization as an example, we modify the warp partitioning strategy to better exploit parallelism. As illustrated in [Figure 6](#figure-06), instead of the original strategy that allocates multiple warps along the $M$ dimension, we constrain the allocation to $W_m=1$—leveraging the fact that the decoding query length is typically small ($<16$)—and reallocate resources to increase the number of warps along the $N$ dimension ($W_n$).

By increasing $W_n$, dequantization stalls can be effectively mitigated by the Streaming Multiprocessor (SM) warp scheduler [San15], as multiple warps concurrently execute dequantization on packed data before proceeding to Tensor Cores-based matrix multiplication.

Similarly, this parallelism strategy alleviates the stalls introduced by on-the-fly quantization in native low-precision attention, ensuring that neither quantization nor dequantization becomes a serialization bottleneck.

<span id="figure-06"></span>

![Warp layout and cooperative softmax design for Tensor Cores utilization.](../../papers/bitdecoding/figure-06.png)

**Figure 6.** Enhancing parallism for efficient Tensor Cores utilization with (1) new warp layout design reduces dequantization stalls and (2) cooperative softmax leverages data movement between GPU register and shared memory for cross-warp reduction with minimal overhead.

**(2) Leveraging memory hierarchy for warps synchronization.** However, with results now distributed across different registers and warps, the original register-level softmax becomes infeasible. Moreover, *a key challenge emerges* due to the incompatibility between the new warp layout and the expected format for MMA operations on $P V$.

To address this, we leverage a multi-level memory hierarchy—spanning registers and shared memory—to enable cross-warp reduction and synchronization for the softmax computation. As illustrated in Algorithm 1, we extend existing high-performance attention algorithms, such as FlashAttention, by introducing two additional shared memory buffers: $\mathit{sTMP} \in \mathbb{R}^{W_n}$ and $\mathit{sAcc} \in \mathbb{R}^{T_m \times T_n}$. The buffer $\mathit{sTMP}$ facilitates cross-warp reduction for computing the row-wise maximum during softmax. This is achieved by first performing intra-warp reduction within registers, followed by inter-warp reduction via shared memory. The buffer $\mathit{sAcc}$ temporarily stores the attention scores $P$ computed in Tensor Core registers and later reloads them via `ldmatrix`, ensuring proper alignment for subsequent Tensor Core `mma` operations.

Since $W_n$ is typically small, we reuse the shared memory pointer of $\mathit{sTMP}$ for $\mathit{sAcc}$ to minimize memory overhead. Moreover, on Hopper Tensor Cores, WGMMA supports direct shared memory access, eliminating the need for explicit data movement from shared memory to registers.

**Algorithm 1: Multi-warps Cooperative Softmax**

- **Require:** $\mathit{sTMP} \in \mathbb{R}^{W_n}$ and $\mathit{sAcc} \in \mathbb{R}^{T_m \times T_n}$ in SMEM.
- **Require:** Load $Q_i \in \mathbb{R}^{T_m \times d}$ and $K_i,V_i \in \mathbb{R}^{T_n \times d}$ to REG.
- $S_i=Q_i K_j^\top$ where $S_i \in \mathbb{R}^{T_m \times T_n}$.
- $m_i^{\mathrm{new}}=\max(m_i,\mathrm{rowmax}(S_i,\mathit{sTMP}))$.
- $P_i=\exp(S_i-m_i^{\mathrm{new}})$ where $P_i \in \mathbb{R}^{T_m \times T_n}$.
- $\mathit{sAcc}=\mathit{tiled\_copy\_r2s}(P_i)$.
- $P_i'=\mathit{tiled\_copy\_s2r}(\mathit{sAcc})$.
- $O_i^{\mathrm{new}}=P_i'V_j+\mathrm{diag}(e^{m_i-m_i^{\mathrm{new}}})O_i$.

<span id="figure-07"></span>

![System overview of BitDecoding.](../../papers/bitdecoding/figure-07.png)

**Figure 7.** System overview of BitDecoding. (1) **Query Transformation** restructures the query tensor layout to enable efficient warp-level execution for attention variants on Tensor Cores. (2) **Residual Kernel** performs quantization and packing with minimal overhead, supporting both tensor-wise and channel-wise scaling. (3) **Packing Kernel** executes dequantization and matrix multiplication using a fine-grained, asynchronous pipeline, maximizing Tensor Cores and CUDA Cores utilization with low-bit parameters.

<span id="section-5"></span>

## 5 System Implementation

In this section, we describe how we implement BitDecoding, as illustrated in [Figure 7](#figure-07). Our implementation consists of three major components: (i) a *query transformation* component that supports diverse attention variants in LLMs; (ii) a *Residual Kernel* that performs low-cost quantization and packing while remaining general to both tensor-wise and channel-wise scaling across quantization algorithms; and (iii) a *Packing Kernel* with a fine-grained pipeline that fully utilizes both Tensor Cores and CUDA cores. Finally, we discuss architecture-specific optimizations that leverage the advanced features of the latest GPU generations (e.g., Hopper and Blackwell) to further enhance decoding throughput.

<span id="section-5-1"></span>

### 5.1 Query Transformation

Modern LLMs adopt diverse attention variants [Dub24, Yan25a, Dee24a] with different key-value (KV) sharing patterns. BitDecoding aims to support all these variants.

For instance, in GQA and MQA, multiple query heads share a KV head, reducing the number of KV projections and memory accesses. The degree of sharing is measured by $g_q=h_q/h_{kv}$, where $h_q$ and $h_{kv}$ are the numbers of query and KV heads, respectively: $g_q=1$ corresponds to MHA, $g_q>1$ denotes GQA, and $h_{kv}=1$ (i.e., $g_q=h_q$) characterizes MQA.

A challenge arises in decoding: since $Q\_len=1$ (one token at a time), the query tensor has a very small batch dimension, and a naive $Q\cdot K^\top$ underfills Tensor Cores, yielding poor warp occupancy and low throughput.

To address this, we perform a *query transformation* that reorganizes the query layout to better match Tensor Core tiling. As illustrated in [Figure 7](#figure-07) (left), we reshape the query tensor from $[1,(g_q,h_{kv})]$ to $[g_q,h_{kv}]$, effectively forming a larger $Q$ tile without changing the semantics of attention or its KV-sharing pattern. Grouped query heads are then processed in parallel as a larger GEMM block, fully populating Tensor Core fragments, improving warp occupancy, and increasing throughput.

<span id="section-5-2"></span>

### 5.2 Residual Kernel

A primary challenge in low-bit KV-cache design is supporting diverse quantization algorithms—especially differing scaling granularities (e.g., tensor-wise, channel-wise)—without sacrificing performance. Quantization involves reductions and element-wise operations to compute scale and zero-point, followed by bit-packing; during decoding these must run online, adding runtime overhead and risking misalignment with the rigid layouts expected by Tensor Cores. To address this, we design the *Residual Kernel* with two key optimizations:

**(1) Partitioning KV cache based on residual block size.** During prefill with context length $L$, we split the KV cache based on a Tensor Cores-aligned residual block size $N_r$ (see [Equation 1](#equation-01)). The first $N_p=L-(L\mod N_r)$ entries are quantized and packed into the low-bit KV cache using a fused quantization and packing operation. The remaining KV Tensor with size $\texttt{res\_len}=L\mod N_r$ are stored in the half-precision residual KV cache. At each decode step, the newly generated $K,V$ tensors are appended to the residual cache and used for attention computation. This cache grows incrementally until it reaches the residual block size $N_r$. Once per token generation, the Residual Kernel computes attention using the half-precision residual KV cache and optionally quantizes it (when $\texttt{res\_len}=N_r$) into packed format.

With this KV cache partitioning during decoding, we can naturally perform channel-wise quantization along the $seq\_len$ and tensor-wise quantization along the hidden dimension within the residual block.

**(2) Optimizing reduction with warp-level instructions.** As shown in [Figure 7](#figure-07) (mid), once the half-precision KV data is computed, it remains in registers as Tensor Cores fragments—structured in the native interleaved layout used by `mma` operations. To efficiently compute the quantization parameters (scale and zero-point), we first perform thread-level reductions to obtain local min/max statistics within each group.

These local results are then aggregated across the warp using the PTX instruction `__shfl_xor_sync`, enabling efficient warp-level reduction without shared memory. When the warp repetition factor $W_n>1$, we introduce a small shared memory buffer to coordinate the final reduction across warps.

After computing the quantization parameters, each thread performs in-register quantization and packs the low-bit values into INT16 format. This avoids extra memory movement and keeps data in a computation-ready state. To minimize overhead, both the scale and zero-point are stored in a compact `half2` format, enabling efficient memory access and fused multiply-add during dequantization in the decode phase.

<span id="section-5-3"></span>

### 5.3 Packing Kernel

Another challenge is the auxiliary low-bit metadata (scale and zero-point), which increases memory traffic, while dequantization still runs on CUDA cores. Without careful scheduling, this disrupts the load-compute pipeline and prevents overlap with Tensor Core operations. We therefore design a fine-grained asynchronous pipeline: CUDA cores handle dequantization, Tensor Cores execute matrix multiplications, and both are orchestrated to overlap with memory transfers through the GPU hierarchy—enabling efficient mixed-precision computation.

**(1) Optimizing asynchronous data movement.** *From Global to Shared Memory*, we follow FlashAttention [Dao24a] via block-wise tiling [Wan25] and strategic recomputation. It processes input matrices $Q \in \mathbb{R}^{T_m \times d}$, $K,V \in \mathbb{R}^{T_n \times d}$ in tiles within shared memory, using block sizes $T_m$ and $T_n$. The number of key-value tiles is $C_n=\lceil L/T_n\rceil$.

To efficiently manage quantization parameters, we introduce dedicated shared memory buffers for quantization paramenter $K_{\mathrm{pack}}$ params ($K_p$) and $V_{\mathrm{pack}}$ param ($V_p$), facilitating efficient tiling for memory copy. These buffers store `scale` and `zeros` in the `half2` format, allowing them to be loaded in a single instruction.

The shape of $K_p$ is determined by the quantization granularity setting, and the $V_p$ follows a Tensor-wise layout:

- **Channel-wise:** $(T_n/\mathit{group\_size},d)$.
- **Tensor-wise:** $(T_n,d/\mathit{group\_size})$.

To achieve optimal memory overlapping, all global-to-shared memory transfers are executed asynchronously using the `cp.async` intrinsic, ensuring efficient pipeline execution, as shown in [Figure 7](#figure-07) (right). We optimize memory transactions using instructions with different caching strategies:

- **`cp.async.cg`:** Used for $Q$, $K_{\mathrm{pack}}$, and $V_{\mathrm{pack}}$, which cache only in global memory as they are not reused within the same kernel.
- **`cp.async.ca`:** Applied to $K_p$ and $V_p$, ensuring smaller byte-level alignment for fine-grained memory access.

In Hopper architecture, we follow FA3, leveraging the `tma.copy` instruction for data loading. This facilitates warp-specialized scheduling, improving data locality and reducing memory latency across multiple warps.

*From Shared Memory to Register*, we use the PTX instruction `ldmatrix` to efficiently load $K_{\mathrm{pack}}$, $V_{\mathrm{pack}}$ and $\mathit{sAcc}$ from shared memory into registers with the Tensor Cores tiling layout. To eliminate bank conflicts, we use a sizzling scheme [Nvi24a] defined as:

<span id="equation-02"></span>

$$
\mathrm{col}_{id}=\mathrm{row}_{id}\oplus\mathrm{col}_{id}
$$

achieve bank conflict-free access. Additionally, we restructure the shared memory layout of $K_p$ and $V_p$ to further reduce bank conflict and maximize throughput efficiency.

**(2) Asynchronous pipeline for overlapping CUDA Cores and Tensor Cores.** To fully utilize both CUDA cores and Tensor Cores, we implement a register-level, asynchronous pipeline that overlaps computation with memory operations. In this pipeline, shared-memory loads via `ldmatrix` and dequantization (`Dequant`) run concurrently with Tensor Core matrix multiplications (`mma`) under the SM warp scheduler.

As shown in [Figure 7](#figure-07) (right), while the $i$-th slice is being processed by `mma` on Tensor Cores, the $(i+1)$-th slice is simultaneously loaded from shared memory (`ldmatrix`) and dequantized. This sustains a continuous producer-consumer flow, improving instruction throughput and maximizing utilization of both CUDA cores and Tensor Cores.

<span id="figure-08"></span>

![Kernel performance with MXFP4 on Blackwell architectures.](../../papers/bitdecoding/figure-08.png)

**Figure 8.** Kernel performance with mxfp4 on Blackwell architectures. (a) RTX 5090. (b) RTX PRO 6000.

<span id="section-5-4"></span>

### 5.4 Latest Architectures Support

While the design presented thus far effectively targets pre-Hopper architectures (e.g., Ampere), newer generations introduce distinct hardware features that require tailored optimization strategies. Below, we detail how our approach adapts to leverage the specialized instructions and native data formats of the Hopper and Blackwell architectures.

**(1) Unlocking Hopper for warpgroup acceleration capabilities via smart uses of PTX-level instructions.** Hopper Tensor Cores, increasingly introduce Warpgroup Matrix Multiply-Accumulate (`wgmma`) instruction. This instruction however imposes a key constraint: in a matrix multiplication $C=A B$, only $A$ and $C$ can be sourced from registers, while $B$ must reside in shared memory. This presents a challenge for low-bit quantized data, as values are typically upconverted to FP16 in registers before computation. To resolve this, we leverage Hopper's `STSM` PTX instruction to store dequantized FP16 values in shared memory efficiently, accessible for `wgmma_SS` operations. Remarkably, the asynchronous nature of WGMMA overlaps storage with computation, optimizing performance.

**(2) Accelerating Blackwell with native low-precision format.** The Blackwell architecture introduces native support for low-precision tensor operations, eliminating the need for explicit dequantization. Consequently, the `lop3`-based register remapping described earlier is bypassed in favor of direct execution. We target Blackwell's low-precision `mma` instructions—specifically those supporting the micro-scaling formats (e.g., `mxfp4 / nvfp4`)—to execute GEMM operations directly on packed 4-bit data. While these instructions enforce rigid layout constraints for both the packed values and their block-scaling factors, the layout transformation strategy proposed in [Section 4.1](#section-4-1) is designed to be layout-agnostic. It automatically aligns the packed KV data with the hardware-mandated format, ensuring seamless integration with Blackwell's native tensor pipelines.

<span id="section-6"></span>

## 6 Evaluation

In this section, we comprehensively evaluate BitDecoding against state-of-the-art approaches and systems. Our evaluation highlights the following key results:

1. BitDecoding outperforms FP16 FlashDecoding-v2 by significant margins across GPU generations, achieving speedups of up to 8.6$\times$ on Blackwell (using native MXFP4), 8.0$\times$ on Hopper, and 7.5$\times$ on Ada architectures, while surpassing the state-of-the-art low-bit system QServe by up to 4.3$\times$ ([Section 6.1](#section-6-1)).
2. In end-to-end long-context inference, BitDecoding reduces single-batch latency by 3x (on LLaMA-3.1-8B with 128K context) and achieves over 4x higher serving throughput than QServe, demonstrating superior scalability in GQA settings where prior CUDA Core-only methods degrade ([Section 6.2](#section-6-2)).
3. BitDecoding preserves near-FP16 accuracy while deriving significant performance gains from each system component, demonstrating only a 0.2% accuracy degradation with 4-bit quantization, while our ablation study confirms that every design module contributes to the overall speedup ([Section 6.3](#section-6-3)).

<span id="section-6-1"></span>

### 6.1 Kernels Performance Across GPU Architectures

**Kernels Settings.** Since different LLM serving scenarios require varying workloads and attention kernel designs, we evaluate performance under the following three representative settings:

- **Single:** A scenario where $\mathit{batch\_size}=1$, representing inference for edge users with long context.
- **Batches:** A setting with a larger $\mathit{batch\_size}$, maintaining the same input length while applying simple padding.
- **Page:** A high-throughput scenario where a larger $\mathit{batch\_size}$ is managed using the page management technique [Kwo23].

**Baselines.** We compare BitDecoding against several representative attention kernel implementations. For FP16 KV cache, we use FlashDecoding [Dao24a, Sha24b]—a split-partitioned variant of FlashAttention optimized for long-context decoding—as our baseline for speedup normalization. For low-bit KV cache, we evaluate Kivi [Liu24c], a non-fused kernel supporting 4-bit and 2-bit quantization; Atom [Zha24e] and QServe [Lin24a], both fused-kernel implementations with CUDA Cores-only approach and supporting 4-bit cache with page management. Notably, Atom does not support GQA.

**Quantization Settings.** We evaluate BitDecoding under various quantization configurations, supporting 4-bit and 2-bit Key tensors with both Channel-wise (KC) and Tensor-wise (KT) schemes.

<span id="figure-09"></span>

![Kernel performance on Hopper H100.](../../papers/bitdecoding/figure-09.png)

**Figure 9.** Kernel performance on Hopper (H100).

**Results on MXFP4 / NVFP4 (RTX5090, RTX PRO 6000).** The Blackwell architecture provides native support for low-precision data formats, eliminating on-the-fly dequantization overhead while delivering very high compute throughput on low-bit operations. As shown in [Figure 8a](#figure-08), BitDecoding achieves remarkable performance, reaching up to 8.6$\times$ speedup in batched scenarios and over 4.3$\times$ in single-batch long-context decoding (128k), significantly outpacing the non-fused attention baseline. Similarly, [Figure 8b](#figure-08) demonstrates that the RTX PRO 6000 attains substantial gains, peaking at 6.5$\times$ speedup with large batch sizes.

<span id="figure-10"></span>

![Kernel performance on RTX4090.](../../papers/bitdecoding/figure-10.png)

**Figure 10.** Kernel performance on RTX4090.

**Results on Advanced Tensor Cores Acceleration (H100).** Newer GPU architectures often introduce advanced compute instructions that significantly accelerate kernel execution. As illustrated in [Figure 9](#figure-09), FlashDecoding-v3, optimized for Hopper Tensor Cores, delivers notable performance gains over its v2 counterpart. While BitDecoding-v2 reaches up to 4.1$\times$ speedup, the v3 implementation further boosts performance to 8.0$\times$. This is enabled by BitDecoding's use of Hopper's `wgmma` and asynchronous memory instructions, ensuring high Tensor Cores utilization even in mixed-precision settings.

<span id="figure-11"></span>

![Kernel performance on A100.](../../papers/bitdecoding/figure-11.png)

**Figure 11.** Kernel performance on A100.

**Results on Bandwidth-constrained GPU (RTX 4090).** Leveraging low-precision data is critical for accelerating inference on bandwidth-constrained GPUs. As shown in [Figure 10](#figure-10), BitDecoding achieves roughly $4\times$ (4-bit) and over $7\times$ (2-bit) speedups over FlashDecoding-v2 in Single and Batches settings, gains that stem directly from alleviating DRAM bottlenecks via low-bit KV caching.

BitDecoding significantly outperforms baselines across all scenarios; unlike the non-fused KIVI, which relies on separate kernels and suffers severe degradation in GQA, BitDecoding's fully fused design maintains high efficiency. In Page settings, it surpasses fused CUDA-core baselines: for MHA, BitDecoding achieves over $6\times$ speedup compared to QServe's $3.5\times$. Crucially, in compute-intensive GQA, it maintains a $3\times$ speedup while QServe drops to $1.4\times$, confirming that leveraging Tensor Cores provides robust acceleration where CUDA-only approaches falter.

**Results on High-Bandwidth GPU (A100).** On architectures with high memory bandwidth like the A100, computation pressure becomes more pronounced, as performance bottlenecks shift from memory access to compute utilization—especially when kernel designs fail to fully exploit available compute resources. As shown in [Figure 11](#figure-11), both KIVI and QServe suffer from poor performance—KIVI due to its non-fused kernel design, and QServe due to underutilization of Tensor Cores—even performing worse than the FP16 baseline. In contrast, BitDecoding consistently outperforms all baselines across workloads, achieving up to $3\times$ speedup, thanks to its efficient utilization of Tensor Cores and fused execution pipeline. An interesting observation is that the performance gap between 4-bit and 2-bit variants narrows on A100, as the increased DRAM bandwidth reduces memory bottlenecks and shifts the performance balance toward compute-bound execution.

<span id="section-6-2"></span>

### 6.2 Performance across LLMs Inference Systems

**Model settings.** We evaluate on a range of LLMs, including LLaMA-2-7B, LLaMA-3.1-8B, LLaMA-3.1-70B, Qwen3-8B, and Qwen3-14B. Among them, only LLaMA-2-7B adopts MHA, while the others use GQA. All models are run on a single A100 GPU, except LLaMA-3.1-70B, which is evaluated on 8$\times$A100 GPUs.

**Quantization settings.** We choose channel-wise quantization for LLMs KV cache as it brings better accuracy and aligns with the Kivi.

**Compared with Non-fused Attention.** As illustrated in [Figure 12](#figure-12), in the Single setting, BitDecoding achieves up to 3.3$\times$ speedup at a 128K context length, where KV cache loading becomes the dominant bottleneck in LLMs inference. In contrast, Kivi suffers from limited scalability and encounters out-of-memory (OOM) failures at 128K due to the lack of block-tiling kernel support. For the Batches setting, BitDecoding significantly outperforms KIVI in throughput: BitDecoding-KC-4 and KC-2 reach up to 900 and 1200 tokens/s, respectively, while KIVI-4 and KIVI-2 peak below 700 tokens/s.

<span id="figure-12"></span>

![End-to-end generation time and decoding throughput compared with Kivi.](../../papers/bitdecoding/figure-12.png)

**Figure 12.** Comparing Kivi with (a) end-to-end generation time and (b) decoding throughput.

**Compared with CUDA Cores-only fused Attention.** We compare BitDecoding with Qserve for page-setting inference, as Qserve supports both MHA and GQA attention structures. The maximum throughput is evaluated under the largest batch sizes available within GPU memory. As illustrated in [Figure 13](#figure-13), Qserve achieves higher throughput than FlashDecoding-v2 on LLaMA-2-7B but suffers from degraded performance on all other models due to inefficiencies in handling GQA. In contrast, BitDecoding consistently outperforms QServe across both LLaMA and Qwen architectures, under both single-GPU and multi-GPU settings, achieving more than 2$\times$ higher maximum throughput compared to QServe.

<span id="figure-13"></span>

![Decoding throughput compared with Qserve.](../../papers/bitdecoding/figure-13.png)

**Figure 13.** Comparing Qserve with decoding throughput.

<span id="section-6-3"></span>

### 6.3 Accuracy, Overhead and Performance Breakdown

**Accuracy analysis.** As shown in [Table 1](#table-01), we evaluate throughput and accuracy across different bit widths. The 2-bit quantization reduces memory consumption significantly, enabling larger batch sizes and achieving a $4.25\times$ higher throughput compared to FP16. Meanwhile, the 4-bit quantization achieves a $2.98\times$ speedup while maintaining near full-precision accuracy with only a minimal $0.2\%$ degradation. These results highlight the trade-off, with 4-bit quantization offering balance and 2-bit maximizing throughput at a slight accuracy cost.

<span id="table-01"></span>

![Efficiency and accuracy tradeoff with a low-bit KV cache.](../../papers/bitdecoding/table-01.png)

**Table 1.** Efficiency and accuracy tradeoff with low-bit KV cache. We use Llama-3.1-8B-Instruct with $seq\_len=32K$, and evaluate average accuracy on longbench [Bai23].

<span id="table-02"></span>

![Latency comparison of quantization and packing during inference.](../../papers/bitdecoding/table-02.png)

**Table 2.** Latency (ms) comparison of quantization and packing during inference.

<span id="table-03"></span>

![Impact of cooperative softmax and warps on performance and validity.](../../papers/bitdecoding/table-03.png)

**Table 3.** Impact of cooperative softmax and warps on performance and validity.

**Half-precision Residual Kernel Overhead.** Half-precision residual KV Cache would introduce quite a small portion memory overhead as $seq\_len\gg N_r$, while $seq\_len$ would be more than 32K and $N_r$ is always less than 256. The half-precision residual KV cache introduces only a slight runtime overhead due to an extra kernel launch, as shown in [Figure 14](#figure-14). Moreover, this overhead becomes increasingly negligible as the sequence length grows, since the residual portion constitutes a smaller fraction of the total KV cache.

<span id="figure-14"></span>

![Runtime overhead of the residual KV cache.](../../papers/bitdecoding/figure-14.png)

**Figure 14.** Runtime overhead of the residual KV cache.

**Quantization and Packing Overhead.** We evaluate the latency of quantization and packing under a sequence length of $seq\_len=128K$, comparing BitDecoding with Marlin [Fra24] and Ladder [Wan24e]. As shown in [Table 2](#table-02), the pre-transformation and packing step in previous mixed-precision computing methods introduce significant overhead, which cannot be ignored. Our kernel incurs minimal overhead after the Prefill phase, primarily due to kernel launch overhead. Moreover, during decoding, we achieves nearly negligible overhead, as it is fully fused into kernel computation.

**Dequantization Overhead.** [Figure 15a](#figure-15) illustrates the high computational overhead of dequantization in Atom and QServe, consuming nearly half the kernel execution time. In contrast, BitDecoding significantly reduces this overhead to less than 15% (4-bit) and 35% (2-bit), thanks to better Tensor Cores overlap.

A further microbenchmark comparing Atom and BitDecoding ([Figure 15b](#figure-15)) reveals BitDecoding's superior memory throughput from effective Tensor Core usage. Conversely, Atom relies heavily on CUDA cores, increasing pressure on FMA and ALU operations.

<span id="figure-15"></span>

![Dequantization overhead and micro-level analysis.](../../papers/bitdecoding/figure-15.png)

**Figure 15.** Dequantization overhead analysis. (a) Dequantization Overhead. (b) Micro Analysis.

**Multi-warps Cooperative Softmax Overhead.** [Table 3](#table-03) shows that increasing $W_n$ improves Tensor Cores utilization and reduces latency, but breaks correctness without cooperative softmax. Enabling cooperative softmax restores correctness with only 0.5% overhead. Although it introduces shared memory access, the overhead is minimal since low-bit data reduces memory bandwidth pressure and shifts the kernel from memory-bound to compute-bound.

**BreakDown Analysis.** To further analyze the performance gains of BitDecoding, we decompose our optimizations in [Figure 16](#figure-16). Following [Ash24], we use a continuous-packing baseline that quantizes and packs the KV cache at every generation step, which introduces substantial overhead and requires manual effort to maintain valid layouts. In contrast, our layout design automatically induces Tensor Core-compatible layouts for arbitrary low-bit formats, fully unlocking the compute potential of Tensor Cores. On top of this, the warp-parallelism strategy contributes significant additional speedups, while the pipeline optimizations further enhance end-to-end performance.

<span id="figure-16"></span>

![Breakdown of BitDecoding optimizations across architectural generations.](../../papers/bitdecoding/figure-16.png)

**Figure 16.** Breakdown of BitDecoding optimizations across architectural generations.

<span id="section-7"></span>

## 7 Related Works

**KV Cache Quantization Algorithms.** KV cache quantization reduces memory usage in LLMs with long contexts while maintaining performance. Recent works explore 4-bit, 2-bit, and even 1-bit KV cache quantization, aiming to push the limits of compression. Methods like KIVI [Liu24c], Gear [Kan24], and KVQuant [Hoo24] use per-channel quantization to handle key-value outliers, while RotateKV [Su25c] applies rotation to smooth channel-wise distributions. Although effective at higher compression ratios, these methods lack efficient system implementations, leading to suboptimal performance.

**Mixed-precision Matrix Multiplication.** Low-bit weight and low-bit KV cache in LLMs create a unique requirement for mixed-precision matrix multiplication (mpGEMM), where one input matrix is in lower precision (e.g., INT4/2/1) while the other matrix remains in higher precision (e.g., FP16/8). Optimized kernels like Ladder [Wan24e] and Marlin [Fra24] improve performance via layout transformations and efficient dequantization. However, these methods require pre-packing and pre-transforming weights, limiting applicability to low-bit KV cache in autoregressive decoding.

**System Implementation for Low-bit KV Cache.** KIVI [Til19] uses Triton with separate kernels for low-bit KV Cache implementation. Atom [Zha24e] integrates quantization within the preceding linear layer, while QServe [Lin24a] fuses quantization directly into FlashAttention kernels. However, they both rely on GEMV operations with fused multiply-add (FMA) instructions, missing Tensor Core acceleration.

<span id="section-8"></span>

## 8 Conclusion

BitDecoding establishes a new system foundation for efficient low-bit KV-cache decoding by demonstrating how CUDA cores and Tensor Cores can be cooperatively orchestrated using principled system designs. Its layout-induction and warp-level coordination techniques generalize across attention variants, quantization schemes, and GPU generations, and naturally extend to emerging architectures such as Blackwell and even beyond. We expect BitDecoding to enable future work on algorithm-system co-design for KV-cache quantization, near-lossless test-time scaling, and more capable GPU execution models for long-context LLMs inference.
