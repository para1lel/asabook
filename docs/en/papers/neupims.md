---
title: 'NeuPIMs'
createTime: 2026/08/23 12:00:00
permalink: /en/papers/neupims/
pageClass: paper-reading
---

> [Guseul Heo](https://dblp.org/pid/299/7465.html), [Sangyeop Lee](https://dblp.org/pid/290/9678.html), [Jaehong Cho](https://dblp.org/pid/297/4770.html), [Hyunmin Choi](https://dblp.org/pid/288/1825.html), [Sanghyeon Lee](https://dblp.org/pid/286/7145.html), [Hyungkyu Ham](https://dblp.org/pid/322/2213.html), [Gwangsun Kim](https://dblp.org/pid/335/5960.html), [Divya Mahajan](https://divyamahajan.github.io/), and [Jongse Park](https://casys.kaist.ac.kr/). First submitted to arXiv on March 1, 2024; current version v1. Published in [ASPLOS '24](https://doi.org/10.1145/3620666.3651380), pp. 722-737. [NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing](https://arxiv.org/abs/2403.00579). [Original PDF](/paper/neupims.pdf). [DOI](https://doi.org/10.1145/3620666.3651380). [TeX source](https://export.arxiv.org/e-print/2403.00579). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Modern transformer-based Large Language Models (LLMs) are constructed with a series of decoder blocks. Each block comprises three key components: (1) QKV generation, (2) multi-head attention, and (3) feed-forward networks. In batched processing, QKV generation and feed-forward networks involve compute-intensive matrix-matrix multiplications (GEMM), while multi-head attention requires bandwidth-heavy matrix-vector multiplications (GEMV). Machine learning accelerators like TPUs or NPUs are proficient in handling GEMM but are less efficient for GEMV computations. Conversely, Processing-in-Memory (PIM) technology is tailored for efficient GEMV computation, while it lacks the computational power to handle GEMM effectively.

Inspired by this insight, we propose NeuPIMs, a heterogeneous acceleration system that jointly exploits a conventional GEMM-focused NPU and GEMV-optimized PIM devices. The main challenge in efficiently integrating NPU and PIM lies in enabling concurrent operations on both platforms, each addressing a specific kernel type. First, existing PIMs typically operate in a "blocked" mode, allowing only either NPU or PIM to be active at any given time. Second, the inherent dependencies between GEMM and GEMV in LLMs restrict their parallel processing. To tackle these challenges, NeuPIMs is equipped with dual row buffers in each bank, facilitating the simultaneous management of memory read/write operations and PIM commands. Further, NeuPIMs employs a runtime sub-batch interleaving technique to maximize concurrent execution, leveraging batch parallelism to allow two independent sub-batches to be pipelined within a single NeuPIMs device. Our evaluation demonstrates that compared to GPU-only, NPU-only, and a naive NPU+PIM integrated acceleration approach, NeuPIMs achieves 3x, 2.4x, and 1.6x throughput improvement, respectively.

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) are being widely deployed across various sectors such as natural language understanding [Ope23, Hof22, Bla22, Tou23], content generation, and decision support. However, a key challenge with these models is the substantial resource requirement they impose, both memory and compute. This paper specifically addresses the inference challenges in contemporary LLMs, with an emphasis on models like GPT-4 [Ope23] and LLaMA [Tou23].

The algorithmic commonality of these state-of-the-art LLMs is that their model architecture constitutes a stack of decoder blocks. As illustrated in [Figure 1](#figure-01)(a), each block is structured around three primary layers: (1) Query-Key-Value (QKV) generation, (2) Multi-Head Attention (MHA), and (3) Feed-Forward Networks (FFNs). For efficient computation of these blocks, a prevalent strategy is batching multiple inference requests. Batching allows QKV generation and feed-forward layers to reuse weights across multiple requests, resulting in General Matrix Multiplication (GEMM) operations between weight and activation matrices. Conversely, the multi-head attention layer requires multiplication between activation matrices and activation vectors with no data reuse opportunity, leading to General Matrix-Vector Multiplication (GEMV) operations.

Overall, LLM inference involves the computation of numerous large-scale GEMMs and GEMVs. To address this computational demand, a common practice is to utilize high-performance machine learning (ML) accelerators, such as GPUs and TPUs. In this paper, we refer to these ML accelerators as Neural Processing Units (NPUs). NPUs are often optimized for compute-intensive tasks, particularly for the efficient execution of GEMMs. However, their utility for GEMVs is less optimal due to the latter's lower arithmetic intensity, which leads to under-utilization of the NPU's computational resources. On the other hand, Processing-in-Memory (PIM) technology [Att24, New20, Tra22, Har21, Neu24a] is less effective for GEMMs but shows promise for the bandwidth-intensive GEMV operations.

To this end, this work proposes NeuPIMs, a heterogeneous acceleration system for batched inference of LLMs. We architect NeuPIMs so that it balances memory bandwidth and computational resources to improve overall inference throughput. NeuPIMs jointly exploits (1) a conventional GEMM-centric NPU using a 2D cluster of multiple systolic arrays and (2) a multitude of GEMV-friendly processing-in-memory (PIM) accelerators.

In designing NeuPIMs, we identify two major challenges:

- **Microarchitectural Challenge:** Current PIMs operate in a "blocked" mode, preventing the simultaneous execution of NPU and PIM. This serialization leads to under-utilization of resources.
- **Algorithmic Challenge:** In an LLM decoder block, GEMM and GEMV operations have a data dependency. This algorithmic limitation restricts parallel NPU+PIM computation.

NeuPIMs addresses these challenges through a hardware-algorithm co-design approach:

1. **Microarchitectural Contribution.** To facilitate NPU+PIM parallel execution, NeuPIMs introduces a modified PIM bank architecture that enables regular memory accesses to occur concurrently with GEMV operations within the PIM. This is achieved by employing distinct row buffers for the two functions, hereafter called *dual row buffers*. Dual row buffers use the property of DRAM that multiple rows can be activated independently. The design also handles and schedules mixed memory-access and PIM commands at the memory controllers without violating DRAM timing parameters. NeuPIMs strategically intersperses the two types of commands to minimize row activation delays. A few composite commands are added to the baseline PIM ISA to perform multiple GEMV operations and amortize control cost.
2. **Algorithmic Contribution.** To enable parallel execution of GEMM and GEMV operators within the decoder block, we introduce *sub-batch interleaving*, which concurrently processes two sub-batch inference computations on NeuPIMs. Because the two sub-batches are independent, GEMM operations from one sub-batch can run in parallel with GEMV operations from another. This creates simultaneous execution opportunities and balances the GEMM and GEMV workloads. To balance the pipeline, we estimate mappings from sequence lengths to MHA execution latency on the PIM, then partition each batch so that the sum of sequence lengths in the sub-batches is balanced.

<span id="figure-01"></span>

![Figure 1. Decoder blocks, baseline accelerators, and the proposed NeuPIMs accelerator.](../../papers/neupims/figure-01.png)

**Figure 1.** (a) Mathematical components of decoder blocks that constitute LLMs, (b) NPU-only baseline accelerator equipped with non-PIM memory, (c) NPU+PIM integrated baseline accelerator, and (d) the proposed NeuPIMs accelerator.

Combining the proposed microarchitectural and algorithmic innovations, NeuPIMs achieves high utilization on both NPU and PIM accelerators. We evaluate NeuPIMs using four variants of GPT-3 and the real-world ShareGPT and Alpaca inference datasets. We develop the NeuPIMs simulator by integrating ONNXim with a PIM simulator built on DRAMsim3. Compared with NPU-only and naive NPU+PIM integrated baseline accelerators, NeuPIMs achieves 2.4x and 1.6x throughput improvement, respectively. These gains are attributed to improved NPU and PIM utilization, from 28% and 17% to 65% and 26%.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 Computational Characteristics of LLM Inference

<span id="figure-02"></span>

![Figure 2. Model architecture and inference in LLMs.](../../papers/neupims/figure-02.png)

**Figure 2.** Model architecture and inference in LLMs.

**Model architecture and execution of LLMs.** [Figure 2](#figure-02) illustrates the model architecture shared by state-of-the-art large language models [Tou23, Bla22]. For an input prompt, the model first enters a summarization phase, encoding the input to establish context for the subsequent generation phase. In the generation phase, the model produces one token per iteration in an autoregressive manner, using the generated key-value projections for the next iteration. Both phases comprise a sequence of decoder blocks, each with (1) QKV generation, (2) multi-head attention (MHA), and (3) feed-forward networks (FFNs).

<span id="figure-03"></span>

![Figure 3. Operators in an LLM decoder block.](../../papers/neupims/figure-03.png)

**Figure 3.** Operators in an LLM decoder block.

**Batched inference of LLMs.** The MHA layers have different computational characteristics from QKV generation and FFN layers. [Figure 3](#figure-03) shows tensor operations for (a) weight-activation and (b) activation-activation multiplication. QKV generation and FFNs multiply a per-token Q/K/V activation or attention vector with trained weight matrices (GEMV). These GEMV operators become GEMMs when they are located in decoders during summarization, where multiple token vectors are available, or when multiple inferences are batched. In contrast, MHA multiplies two activations, one for the current token and one for all preceding tokens, yielding GEMV. Because activation operands are unique to each request, batching is not possible and the computation is highly memory-bandwidth bound.

**Analysis of arithmetic intensity.** To understand LLM inference, we conduct a roofline analysis using GPT3-13B and GPT3-175B. [Figure 4](#figure-04) shows arithmetic intensity (FLOPS/byte) against performance (TFLOPS). Generation is severely memory-bound for both models, while summarization is compute-bound. The two phases have algorithmic dependencies and alternate sequentially, making high utilization on a homogeneous platform difficult. This motivates a heterogeneous system combining a compute-centric systolic-array NPU for GEMMs with memory-centric Processing-in-Memory (PIM) accelerators for GEMVs.

<span id="figure-04"></span>

![Figure 4. Arithmetic intensities of LLM layers.](../../papers/neupims/figure-04.png)

**Figure 4.** Arithmetic intensities of LLM layers.

<span id="section-2-2"></span>

### 2.2 LLM Inference Serving

As LLMs demand substantial resources, the de facto practice is to build large-scale inference serving frameworks such as DeepSpeed, Orca, and vLLM [Vll23]. These frameworks provide inference services for prompts and enable batching.

**Selective batching.** Batching is generally effective for neural-network inference because it improves utilization without sacrificing latency requirements. MHA layers, however, do not allow batching. Orca therefore computes attention layers individually while batching QKV generation and FFN layers. The system benefits from batching when possible and serializes the computation otherwise. This property requires simultaneous GEMM and GEMV computation, which motivates this work.

**Iteration-level scheduling.** Inference serving receives requests as a stream without a deterministic schedule. Orca schedules batched inference at the beginning of every iteration, allowing new requests to enter and finished requests to leave the batch. Newly arrived requests therefore do not wait for an already-started batch to finish its generation phase. NeuPIMs builds on this technique and manages requests at iteration boundaries.

**Memory paging for attention.** vLLM focuses on memory management. QKV generation produces a KV cache that is reused during generation. Inference systems cache these KV projections in memory, and the cache can become large for long sequences. vLLM introduces paging for this data so that memory is not allocated long before use. NeuPIMs employs the same page-based allocation mechanism for KV cache, increasing the effective batch size.

*NeuPIMs is designed for an inference serving system that incorporates all of these techniques.*

<span id="section-3"></span>

## 3 Motivation

This section provides the motivation behind the design decisions in NeuPIMs. We first identify problems in existing GPU-based LLM inference serving systems, then discuss the limitations of naive NPU-PIM integration and define the target research challenges.

<span id="section-3-1"></span>

### 3.1 GPU-based LLM Inference Serving

Because LLMs require considerable memory, they are commonly deployed on clusters of multiple GPUs, using pipeline and/or tensor parallelism.

<span id="figure-05"></span>

![Figure 5. GPU resource utilization for four LLMs.](../../papers/neupims/figure-05.png)

**Figure 5.** GPU resource utilization for four different LLMs.

**Under-utilization of the GPU system.** We analyze a GPU-equipped baseline to understand compute, memory, and bandwidth utilization for LLM inference. We compare NVIDIA GeForce RTX 3090 24GB and NVIDIA A100 40GB systems running GPT-NeoX, LLaMA2, OPT, and MPT. [Figure 5](#figure-05) presents utilization results with layer-wise variations as error bars. Capacity utilization approaches 100% despite imperfections in the parallelization schemes because the number of GPUs is determined by capacity constraints. Computational-resource utilization, however, is consistently below 40%, showing the cost inefficiency of GPU-based LLM inference. The under-utilization is caused by insufficient bandwidth, even though A100 provides an aggregate 1,555 GB/s through HBM. This imbalance is inevitable while serial dependencies between GEMMs and GEMVs persist.

<span id="figure-06"></span>

![Figure 6. NPU and PIM utilization for a decoder block.](../../papers/neupims/figure-06.png)

**Figure 6.** NPU-PIM resource utilization for a decoder block.

<span id="figure-07"></span>

![Figure 7. Overview of the proposed NeuPIMs system.](../../papers/neupims/figure-07.png)

**Figure 7.** Overview of the proposed NeuPIMs system.

<span id="section-3-2"></span>

### 3.2 A Naive NPU-PIM Approach

A straightforward way to resolve the bandwidth bottleneck is to use PIM to offload bandwidth-bound computation. We therefore design a naive NPU-PIM accelerator with a systolic-array NPU and the Newton PIM GEMV accelerator [New20]. We use the methodology described in [Section 8](#section-8), replacing GPUs with a standard NPU-PIM device.

[Figure 6](#figure-06) presents NPU and PIM compute utilization for different layers in decoder blocks. NPU is busy with QKV generation, projection, and FFN layers while PIM utilization is zero. Conversely, NPU utilization is almost zero while PIM runs MHA. Combined utilization over the complete execution is below 40% for both resources.

**Necessity of concurrent NPU and PIM executions.** The under-utilization is primarily caused by a limitation in PIM microarchitecture that disallows concurrent host (NPU) and PIM execution, serializing otherwise disjoint resource use. The critical challenge for practical PIM in NPU accelerators is therefore enabling parallel execution of the two units.

<span id="section-4"></span>

## 4 Overview of NeuPIMs

[Figure 7](#figure-07) illustrates the proposed system. It alleviates low resource utilization in LLM inference serving through (1) an NPU with systolic arrays, vector units, and HBM-based PIM channels, and (2) a scheduler that partitions a batch into two sub-batches and interleaves their execution.

1. **NeuPIMs system.** The system comprises a host CPU, multiple NeuPIMs devices, standalone NPUs, and a high-bandwidth interconnect such as PCIe or CXL. Summarization is entirely GEMM, so standalone NPUs handle it while NeuPIMs focuses on generation. Requests arrive as a stream and are assigned to PIM channels in a request pool until the current iteration ends.
2. **NeuPIMs accelerator.** The bank architecture extends standard PIM with dual row buffers, one for PIM execution and one for regular memory accesses. This allows the NPU to access rows not currently used by PIM.
3. **NeuPIMs scheduling algorithm.** The prototype has 32 HBM-based PIM channels, each with its own memory controller. Controllers interleave memory and PIM commands without violating timing constraints while maximizing control-path throughput.
4. **NeuPIMs compiler framework.** The compiler frontend accepts LLM and system specifications whose syntax resembles ONNX. It translates the model configuration into intermediate representations and emits NPU and NeuPIMs instruction binaries, adjusting tile sizes and instruction order to match the system specification.

<span id="section-5"></span>

## 5 NeuPIMs Architecture

<span id="section-5-1"></span>

### 5.1 PIM Microarchitecture for Concurrent Execution

**Single row buffer for PIM-based accelerator.** [Figure 8](#figure-08)(a) depicts a PIM GEMV accelerator with banks equipped with a single row buffer. The vector operand is first placed in a global buffer shared by all banks in a channel. Rows of the matrix operand are read from multiple banks in parallel and placed in their row buffers. Parallel multipliers and an adder tree then compute partial dot products from the broadcast vector and per-bank row buffers.

**Limitation of current PIM-based GEMV accelerators.** Current PIM accelerators [New20, Har21] operate in a "blocked" mode that prevents simultaneous NPU and PIM execution. A memory bank's single row buffer serves both regular memory read/write operations and PIM GEMV, so the modes are managed sequentially. This is acceptable for PIM-only GEMV, but it is a problem for LLM inference, which requires both GEMM and GEMV. NeuPIMs therefore enables parallel execution of both modes.

**Extending PIM with dual row buffers.** [Figure 8](#figure-08)(b) shows the NeuPIMs bank. Each bank has dual row buffers, MEM and PIM, connected to independent data paths. The MEM row buffer handles regular read/write accesses; the PIM row buffer handles GEMV. We minimize changes to the microarchitecture and place complexity in the command interface and memory-control mechanism. We prototype with Newton [New20], but the techniques apply to any GEMV accelerator that follows the standard DRAM microarchitecture and command interface.

<span id="figure-08"></span>

![Figure 8. Memory-bank microarchitecture with one and two row buffers.](../../papers/neupims/figure-08.png)

**Figure 8.** Microarchitecture of memory banks in (a) existing PIM accelerators with single row-buffer banks and (b) NeuPIMs with dual row-buffer banks.

<span id="section-5-2"></span>

### 5.2 Memory Command Interface

**Existing command interface for PIM-based GEMV.** NeuPIMs uses a PIM accelerator with a command interface built on the DRAM standard. Four commands operate the PIM. `PIM_GWRITE` copies a row from a bank to the global vector buffer. Grouped `PIM_ACTIVATION` commands activate PIM row buffers in multiple banks, usually four at a time under the tFAW power constraint. `PIM_DOTPRODUCT` performs parallel dot products, and `PIM_RDRESULT` transfers accumulated results to the host.

<span id="figure-09"></span>

![Figure 9. Timing comparison of PIM commands.](../../papers/neupims/figure-09.png)

**Figure 9.** PIM command timing comparison.

**NeuPIMs command interface.** We add three commands to support NeuPIMs.

<span id="table-01"></span>

![Table 1. NeuPIMs command set.](../../papers/neupims/table-01.png)

**Table 1.** List of NeuPIMs commands.

- **PIM_HEADER:** Allows varying GEMV dimensionalities. The controller receives the dimensions before execution and can estimate latency without conflicting with DRAM refresh.
- **PIM_GEMV:** Replaces a sequence of fine-grained dot-product and result-read commands with one composite command. Its argument $k$ gives the number of dot products.
- **PIM_PRECHARGE:** Precharges the PIM row buffers after GEMV. It is like regular PRECHARGE but targets the PIM row buffer.

<span id="section-5-3"></span>

### 5.3 Memory Controller

NeuPIMs has multiple channels, each with multiple PIM banks. Requests are assigned to channels, and each request's MHA execution is distributed across its banks. Each channel has a PIM command queue, and PIM commands are broadcast to every bank in that channel.

**Interleaved scheduling of memory read/write and PIM commands.** A NeuPIMs memory controller must interleave memory read/write and PIM commands so that command/address bus bandwidth does not become a bottleneck. NeuPIMs prioritizes PIM commands because their issuing delay is greater than that of memory commands. The resulting PIM command traffic is small enough to share the C/A bus without significant degradation.

<span id="section-6"></span>

## 6 NeuPIMs Scheduling

Dual row buffers let NeuPIMs handle NPU memory accesses and PIM commands simultaneously. This section describes overlap opportunities in MHA and the *sub-batch interleaving* technique for batched LLM inference.

<span id="section-6-1"></span>

### 6.1 Overlapping Opportunities in MHA Layer

<span id="figure-10"></span>

![Figure 10. Overlapping opportunities in multi-head attention.](../../papers/neupims/figure-10.png)

**Figure 10.** Overlapping opportunities of multi-head attention layers. NPU-S denotes systolic arrays and NPU-V denotes vector units.

<span id="figure-11"></span>

![Figure 11. Serialized execution and sub-batch interleaving.](../../papers/neupims/figure-11.png)

**Figure 11.** Example execution timelines of LLM decoder blocks: (a) serialized execution and (b) sub-batch interleaving. $N$ is the number of decoder blocks.

[Figure 10](#figure-10) illustrates overlap between (1) logit and attend operations on the PIM side and (2) softmax operations on the NPU side. MHA operations can be decomposed at head granularity, so a naive NPU-PIM architecture appears to have an opportunity to overlap both resources. It cannot exploit that opportunity because results cannot be transferred between PIM units and vector units through PIM channels. With dual row buffers, NeuPIMs concurrently uses NPU and PIM, allowing vector units to store partial logit and softmax values before all PIM GEMVs finish. The overlap is possible only because MHA exposes head-level parallelism, and it exists between PIM and NPU vector units; the NPU systolic arrays remain largely unused during MHA.

<span id="section-6-2"></span>

### 6.2 Sub-batch Interleaving

**Limitation of serialized executions.** [Figure 11](#figure-11)(a) shows a decoder block on a naive NPU-PIM device. QKV generation, MHA, and projection and FFNs have data dependencies, so they execute serially and leave both NPU and PIM under-utilized.

**Interleaving the two sub-batches.** We partition one large batch into two sub-batches and alternate them. [Figure 11](#figure-11)(b) shows how PIM-friendly and NPU-friendly operations from the two sub-batches execute at the same time, improving both utilizations.

**Comparative analysis on the execution timelines.** Let $N$ denote the number of decoder blocks on one NeuPIMs device. Without interleaving, the operators in each decoder block execute sequentially, for a total time of $N$ times the per-block time: QKV generation on NPU systolic arrays, MHA on PIM and NPU vector units, and projection and FFNs on NPU systolic arrays. With interleaving, MHA execution is hidden by NPU systolic-array execution. The total time is $(N-1)$ times the per-sub-batch partial execution time plus one decoder-block time split between the start and end. During interleaving, NPU and PIM utilization improves because their executions overlap. Our study shows that the interleaved period is mostly bounded by NPU GEMM time, hiding the PIM MHA time.

**Challenges.** NeuPIMs must address two issues. First, it must balance execution time for each sub-batch, especially MHA. Because MHA latency is determined by the channel processing the longest sequence, token lengths must be balanced across channels. This is handled by the channel load-balancing algorithm in [Section 6.4](#section-6-4). Second, the two sub-batches should have similar execution times. Each interleaving stage is bounded by the slower sub-batch, so NeuPIMs uses the sub-batch partitioning algorithm in [Section 6.5](#section-6-5).

<span id="section-6-3"></span>

### 6.3 Multi-Head Attention Latency Estimation

NPU operation latency depends largely on inference batch size. To optimize MHA, we estimate its execution time using the key-value mapping to PIM memory. The GEMV vector is shared across banks and the GEMV matrix is interleaved row-wise. Key caches at the same row and column share layer and head indices but have different sequence indices. Value caches at the same row and column share layer, head, and sequence indices, with each head embedding interleaved across banks. [Algorithm 1](#algorithm-01) uses this mapping to estimate MHA latency.

<span id="algorithm-01"></span>

**Algorithm 1: MHA Latency Estimation.**

- **Input:** `seq_len`, the request sequence length.
- **Parameters:** model embedding size $E$, one-tile GEMV latency $L_{\mathrm{tile}}$, global-buffer write latency $L_{\mathrm{GWRITE}}$, DRAM page size $P_{\mathrm{DRAM}}$, PIM banks per channel $B_{\mathrm{chnl}}$, and number of heads $N_{\mathrm{head}}$.
- **Output:** estimated MHA latency $L_{\mathrm{MHA}}$.
- Set $L_{\mathrm{MHA}}\leftarrow 0$.
- **For** the GEMV $K^\top\times\mathrm{Query}$:
  - Set $N_{\mathrm{tiles}}\leftarrow (\mathrm{seq\_len}/B_{\mathrm{chnl}})(E/P_{\mathrm{DRAM}})$.
  - Add $L_{\mathrm{GWRITE}}(E/P_{\mathrm{DRAM}})$ and $L_{\mathrm{tile}}N_{\mathrm{tiles}}$ to $L_{\mathrm{MHA}}$.
- **For** the GEMV $\mathrm{Logits}\times\mathrm{Value}$:
  - Set $N_{\mathrm{tiles}}\leftarrow ((E/N_{\mathrm{head}})/B_{\mathrm{chnl}})((\mathrm{seq\_len}/P_{\mathrm{DRAM}})N_{\mathrm{head}})$.
  - Add $L_{\mathrm{GWRITE}}((\mathrm{seq\_len}/P_{\mathrm{DRAM}})N_{\mathrm{head}})$ and $L_{\mathrm{tile}}N_{\mathrm{tiles}}$ to $L_{\mathrm{MHA}}$.
- **Return:** $L_{\mathrm{MHA}}$.

<span id="section-6-4"></span>

### 6.4 Greedy Min-Load Bin Packing Algorithm

NeuPIMs allocates requests to PIM channels, each of which contains banks that partially execute MHA. To minimize the difference between the most congested and least loaded channels, the greedy min-load bin-packing algorithm uses the latency estimate above. It sorts requests in decreasing sequence length, places each request in the least-loaded channel, and updates the estimated latency.

<span id="algorithm-02"></span>

**Algorithm 2: Greedy Min-Load Bin Packing.**

- **Input:** $L_{\mathrm{req}}$, a list of sequence lengths for new requests, and $L_{\mathrm{chnl}}$, the current request allocation for each channel.
- Set $L_{\mathrm{load}}\leftarrow []$.
- **For** each channel $\mathrm{chnl}$ in $L_{\mathrm{chnl}}$:
  - Set $S_{\mathrm{load}}\leftarrow 0$.
  - **For** each request $\mathrm{req}$ in $\mathrm{chnl}$, add `MHALatencyEstimation(req)` to $S_{\mathrm{load}}$.
  - Append $S_{\mathrm{load}}$ to $L_{\mathrm{load}}$.
- **For** each new request $\mathrm{new\_req}$ in $L_{\mathrm{req}}$:
  - Set $\mathrm{min\_index}$ to the index of the minimum element of $L_{\mathrm{load}}$.
  - Append $\mathrm{new\_req}$ to $L_{\mathrm{chnl}}[\mathrm{min\_index}]$.
  - Add `MHALatencyEstimation(new_req)` to $L_{\mathrm{load}}[\mathrm{min\_index}]$.
- **Return:** $L_{\mathrm{chnl}}$.

<span id="section-6-5"></span>

### 6.5 Sub-batch Partitioning Algorithm

Because NPU-friendly operations depend on batch size, the two sub-batches should have balanced sizes. [Algorithm 3](#algorithm-03) divides the requests in each channel into halves and appends each half to one sub-batch.

<span id="algorithm-03"></span>

**Algorithm 3: Sub-Batch Partitioning.**

- **Input:** $L_{\mathrm{req}}$, the active request set in each channel.
- **Output:** $\mathrm{SB}_1$ and $\mathrm{SB}_2$, the sub-batches for interleaving.
- Set `turn` to `True`, and set $\mathrm{SB}_1,\mathrm{SB}_2\leftarrow [],[]$.
- **For** each channel request list $\mathrm{req}_{\mathrm{chnl}}$ in $L_{\mathrm{req}}$:
  - Set $\mathrm{bsize}\leftarrow |\mathrm{req}_{\mathrm{chnl}}|/2$.
  - **If** $|\mathrm{req}_{\mathrm{chnl}}|\bmod 2\ne 0$, set $\mathrm{bsize}$ to $\lceil\mathrm{bsize}\rceil$ when `turn` is true and to $\lfloor\mathrm{bsize}\rfloor$ otherwise, then flip `turn`.
  - Convert $\mathrm{bsize}$ to an integer.
  - Append $\mathrm{req}_{\mathrm{chnl}}[:\mathrm{bsize}]$ to $\mathrm{SB}_1$ and $\mathrm{req}_{\mathrm{chnl}}[\mathrm{bsize}:]$ to $\mathrm{SB}_2$.
- **Return:** $\mathrm{SB}_1,\mathrm{SB}_2$.

<span id="section-7"></span>

## 7 Scaling NeuPIMs System

Model parallelism partitions model parameters across multiple NeuPIMs devices. This is essential because a single device has limited memory. We discuss pipeline parallelism and tensor parallelism, two common model-parallel techniques for LLM inference. These techniques are not contributions of this work, but the section shows how they adapt to NeuPIMs.

<span id="section-7-1"></span>

### 7.1 Pipeline Parallelism of NeuPIMs System

Pipeline parallelism divides the model by layer, placing several layers on each device. The batch is divided into micro-batches corresponding to pipeline depth, and devices process them in a pipeline. This approach applies to NeuPIMs. The number of decoder blocks on each device decreases proportionally, reducing the performance benefit of co-located blocks. Pipeline parallelism also reduces batch size; sub-batch interleaving reduces it further and can under-utilize the NPU systolic arrays.

<span id="section-7-2"></span>

### 7.2 Tensor Parallelism of NeuPIMs System

Tensor parallelism splits model tensors into shards distributed across devices. Devices execute their shards in parallel and aggregate results, which requires communication. Sub-batch interleaving doubles communication frequency, but total traffic remains unchanged from a non-partitioned batch, resulting in modest communication overhead. The sub-batch that completes first can communicate while the other computes, further reducing latency. We therefore prioritize tensor parallelism over pipeline parallelism and use pipeline parallelism when the model is too large for tensor parallelism alone.

<span id="section-8"></span>

## 8 Evaluation

<span id="section-8-1"></span>

### 8.1 Methodology

**Baseline.** We compare GPU-only, NPU-only, NPU+PIM, and NeuPIMs.

- **GPU-only:** A real NVIDIA A100 40GB GPU running a batched LLM workload compiled with PyTorch.
- **NPU-only:** An NPU such as a TPU without PIM. It has equivalent memory bandwidth to the alternatives and includes systolic arrays and vector units.
- **NPU+PIM:** An off-the-shelf NPU integrated with a Newton PIM GEMV accelerator. MHA GEMV operations are mapped to PIM and all other operations run on the NPU. Requests are assigned to channels round-robin.

**Cycle-level simulation.** We build the NeuPIMs simulator on ONNXim and DRAMsim3, linking the simulators by modifying ONNXim's memory interface and offloading memory accesses to DRAMsim3.

<span id="table-02"></span>

![Table 2. NeuPIMs hardware specification.](../../papers/neupims/table-02.png)

**Table 2.** NeuPIMs hardware specification.

**Hardware specifications.** Our prototype is a multi-chiplet design containing eight systolic arrays, each integrated with a SIMD vector unit. Each memory channel controls 32 PIM banks and provides 1 GB of capacity. The architecture is orthogonal to these choices and can use configurations suited to model size and sequence lengths.

<span id="figure-12"></span>

![Figure 12. End-to-end throughput comparison.](../../papers/neupims/figure-12.png)

**Figure 12.** Throughput comparison for GPU-only, NPU-only, NPU+PIM, and NeuPIMs on Alpaca and ShareGPT with batch sizes 64, 128, 256, 384, and 512.

<span id="table-03"></span>

![Table 3. Evaluated LLM configurations.](../../papers/neupims/table-03.png)

**Table 3.** The evaluated LLM configurations.

**LLM models.** We use four GPT-3 variants, as listed in [Table 3](#table-03). Although the experiments focus on GPT-3 variants, NeuPIMs can host any decoder-based generation model.

**Datasets.** We use the ShareGPT and Alpaca datasets. ShareGPT contains conversations scraped from real user logs of ChatGPT. Alpaca is an instruction dataset generated by OpenAI's text-davinci-003. ShareGPT has average input and output lengths of 80 and 296 tokens; Alpaca has 12 and 56.

**Workload.** Cycle-accurate simulation is infeasible for full inference-serving experiments, so we synthesize workloads for system-level evaluation. We vary model type, batch size, and tensor/pipeline parallelism. For each combination we simulate a fixed interval, randomly sampling sequence lengths from the datasets to warm up a batch with varied requests. We sample ten batches and use them to measure throughput.

<span id="section-8-2"></span>

### 8.2 Results

**Throughput.** [Figure 12](#figure-12) compares the three baselines and NeuPIMs. GPU-only and NPU-only show marginal differences because both execute complete decoder blocks without PIM, including bandwidth-bound MHA. Integrating PIM with the NPU gives an average 1.5x improvement over NPU-only by offloading MHA GEMV. NeuPIMs consistently surpasses NPU+PIM, adding 13% to 3x throughput across models and datasets. Gains are larger for ShareGPT because its longer sequences provide more acceleration opportunities. As batch size grows from 64 to 512, gains grow substantially because NeuPIMs shifts the bottleneck from bandwidth to NPU compute. The improvement uses the same memory capacity, which is useful for datacenter inference serving.

<span id="table-04"></span>

![Table 4. Average resource utilization.](../../papers/neupims/table-04.png)

**Table 4.** Average utilization of NPU/PIM compute resources and memory bandwidth.

**Utilization.** [Table 4](#table-04) compares average utilization for GPT3-30B, batch size 256, and ShareGPT. NPU+PIM raises NPU utilization to 28.0% by offloading MHA. It still suffers temporal blocking from GEMM-GEMV dependencies. NeuPIMs reaches 64.9% NPU utilization and 26.4% PIM utilization through concurrent execution.

<span id="figure-13"></span>

![Figure 13. Batch-size sensitivity and ablation study.](../../papers/neupims/figure-13.png)

**Figure 13.** GPT3-7B and ShareGPT experiment. DRB is dual row buffers, GMLBP is greedy min-load bin packing, and SBI is sub-batch interleaving.

**Ablation study.** Starting from NPU+PIM, we add the three proposed techniques in [Figure 13](#figure-13). Dual row buffers provide an average 69.7% throughput improvement and have the largest effect because they enable concurrent NPU+PIM execution. Greedy min-load bin packing distributes requests evenly and always helps. Sub-batch interleaving can hurt at small batch sizes because partitioning under-utilizes the NPU systolic array and pipeline overhead can exceed the benefit. At batch size 256 or larger, NeuPIMs achieves the highest throughput.

<span id="figure-14"></span>

![Figure 14. Throughput under tensor and pipeline parallelism.](../../papers/neupims/figure-14.png)

**Figure 14.** Throughput of a multi-NeuPIMs system as tensor and pipeline parallelism change.

**Implication of parallelization schemes.** As LLM size increases, NeuPIMs scales the number of devices using tensor and pipeline parallelism. [Figure 14](#figure-14) fixes the total number of requests at 256 while batch size per device varies. Tensor parallelism is preferable because it maintains a larger batch and better NPU efficiency. The trend is consistent across model variants, while throughput decreases when per-device batch size becomes small.

**Area overhead.** The main area overhead in NeuPIMs is the dual row buffer. We use CACTI 7.0 with 22 nm technology and double the row-buffer resource in the configuration. The measured overhead is 3.11%, which is small relative to the performance gain.

<span id="table-05"></span>

![Table 5. NeuPIMs power overhead.](../../papers/neupims/table-05.png)

**Table 5.** NeuPIMs power overhead.

**Power overhead.** NeuPIMs uses more memory power than NPU-only because it operates NPU and PIM concurrently. We measure this with Micron's DRAM power model from DRAMsim3. We assume an all-bank computation command consumes four times the power of a read command, and the additional row buffer consumes background power. NeuPIMs consumes 1.8x more power while providing 2.4x speedup, corresponding to a 25% energy reduction.

<span id="figure-15"></span>

![Figure 15. NeuPIMs speedup over TransPIM.](../../papers/neupims/figure-15.png)

**Figure 15.** Speedup of NeuPIMs over TransPIM [Tra22].

**Comparison with TransPIM.** TransPIM is a standalone PIM-only solution that executes all transformer operators in PIM. Because no open-source simulator exists, we develop a simulator based on DRAMsim3 and align HBM timing and capacity with NeuPIMs and the NPU+PIM baseline. [Figure 15](#figure-15) shows that NeuPIMs has 228x higher average throughput, with speedups from 79x to 431x. The gap comes from effective GEMM execution on the NPU, whereas TransPIM targets single-batch encoder-style inference and is unsuitable for batched decoder-based LLM inference.

<span id="section-9"></span>

## 9 Discussion

**Model training.** Training uses fixed-length input and output sequences and therefore consists of GEMMs, not GEMVs. PIM targets bandwidth-bound GEMVs and performs poorly on GEMMs. NeuPIMs can be used for training, but its efficiency is limited.

**Integration with production software stack.** The NeuPIMs compiler has an interface similar to modern machine-learning libraries such as ONNX, PyTorch, and JAX. Integrating NeuPIMs with the existing software stack requires a translator from those model representations to the NeuPIMs specification. The rest of the system remains the same because it already includes an inference-serving scheduler, NPU and PIM operator compilers, and an inference runtime.

<span id="section-10"></span>

## 10 Related Work

**LLM inference serving.** LLM serving systems improve inference performance by reducing memory footprint, optimizing kernel execution, choosing intra- and inter-operator partitioning, or combining these techniques. This work targets utilization of hardware platforms suited to compute and I/O, NPUs and PIMs, and adds a scheduling policy. Existing GPU kernel optimizations cannot fully remove the I/O and bandwidth bottleneck of GEMV kernels, so NeuPIMs builds on selective batching and KV caching to use the transformer hardware more effectively.

**PIM for language model support.** TransPIM [Tra22] accelerates end-to-end transformer inference with PIM and a customized dataflow. It targets encoder attention and single-request inference, so it is unsuitable for decoder-based batched LLM inference. AttAcc [Att24] uses PIM for attention and reduces KV movement, whereas NeuPIMs combines a PIM accelerator with scheduling for end-to-end LLM inference. Other PIM work targets GEMV because of its bandwidth-bound nature [New20, Har21], but does not enable simultaneous PIM and NPU execution.

**Heterogeneous acceleration pipeline for deep learning.** Prior work proposes pipelined machine-learning accelerators, and other work targets specific models, but these designs do not use PIM to relieve LLM bandwidth demand or address the under-utilization of GEMV and GEMM in decoder blocks.

<span id="section-11"></span>

## 11 Conclusion

LLM inferencing demands dedicated resources that can be deployed at scale. These models require high memory capacity, high compute intensity, and high bandwidth. We propose NeuPIMs, which integrates a general ML accelerator NPU with PIM to address different operations and dataflow in transformer layers. We introduce a scheduling and execution strategy that uses HBM, the compute-intensive NPU, and the PIM accelerator more effectively for inference serving. NeuPIMs improves throughput by 1.6x over a baseline that naively integrates an NPU with PIM.

## Acknowledgments

We thank our shepherd Vidushi Goyal and the anonymous reviewers for their comments and feedback. This research is supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grants No. 2022-0-01037, No. 2018-0-00503, and IITP-2024-2020-0-01795, and by the Artificial Intelligence Graduate School Program (KAIST) No. 2019-0-00075, funded by the Korean government (MSIT).
