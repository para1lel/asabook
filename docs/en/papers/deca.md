---
title: DECA Accelerator
createTime: 2026-08-23
permalink: /en/papers/deca/
---

> Gerasimos Gerogiannis, Stijn Eyerman, Evangelos Georganas, Wim Heirman, and Josep Torrellas. First submitted to arXiv on 2025-05-25 (current version v2, 2025-08-08). Intel Corporation, Intel Labs, and University of Illinois at Urbana-Champaign. [arXiv:2505.19349](https://arxiv.org/abs/2505.19349). [Original PDF](/paper/deca.pdf). [TeX source](https://arxiv.org/e-print/2505.19349). The full source title is “DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model”.

## Abstract

To alleviate the memory bandwidth bottleneck in Large Language Model (LLM) inference workloads, weight matrices are stored in memory in quantized and sparsified formats. Hence, before tiles of these matrices can be processed by in-core generalized matrix multiplication (GeMM) hardware engines, they need to be dequantized and de-sparsified. This is currently performed in software with vector operations. Unfortunately, this approach delivers only modest performance. Moreover, it is hard to understand how to improve the system, as the overall GeMM performance depends on the interaction between memory resources, vector units, and hardware matrix engines.

To improve the performance of LLM inference in advanced platforms equipped with in-core GeMM engines and HBM, this paper makes three main contributions. First, it develops an analytical performance model with a 3D visual representation that provides insights into how memory resources, vector units, and hardware matrix engines interact to deliver compressed GeMM performance. Second, it proposes DECA, a new near-core ML-model decompression accelerator. DECA offloads tile de-sparsification and dequantization from the CPU, producing ready-to-use tiles for in-core GeMM engines. Third, it introduces a new ISA extension that enables out-of-order invocation of the near-core accelerator. With this extension, accelerator and core computations can interleave and overlap with high performance. Our evaluation shows that, in a simulated 56-core Xeon 4 server with HBM, DECA accelerates the execution of compressed GeMMs by up to 4x over the use of optimized Intel software kernels. Further, DECA reduces the next-token generation time of Llama2-70B and OPT-66B by 1.6x-2.6x.

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) are important Machine Learning (ML) workloads for chatbots, translation, text summarization, and content creation. LLMs use transformers and mainly consist of multi-head attention and fully connected (FC) layers. The largest models contain trillions of parameters in the FC layers. During inference, these weights have low reuse in small-batch scenarios, stressing memory capacity and bandwidth.

GPUs are a standard platform for LLM inference because of their compute and memory bandwidth. Recent Intel Xeon 4 servers add in-core generalized matrix multiplication (GeMM) engines called TMUL and can be equipped with High Bandwidth Memory (HBM). TMUL is programmed with AMX instructions and increases GeMM throughput by an order of magnitude over vector SIMD units, while HBM supplies three to four times the bandwidth of DDR systems.

In Xeon servers, LLM inference is memory-bandwidth bound. Large GeMMs in FC layers account for more than 90% of next-token generation time for Llama2-70B [Lla23]. Thus, accelerating these GeMMs is central to accelerating inference.

Model compression techniques such as low-bit weight quantization and sparsification reduce memory traffic. However, TMUL expects dense BF16 or INT8 tiles and cannot directly consume arbitrary quantization schemes or sparse patterns. Intel libxsmm therefore uses AVX vector instructions to read compressed tiles, de-sparsify and dequantize them, and feed dense tiles to the AMX unit. This cooperative mode combines vector and matrix domains, each with separate instructions and functional units.

Profiling shows that libxsmm is effective for moderately compressed GeMMs and DDR memory, but its performance degrades with HBM. A traditional two-dimensional roofline model cannot explain this degradation because it omits the interaction among memory, vector, and matrix resources. We develop a three-dimensional model, called Roof-Surface, to expose this interaction and attribute the degradation to AVX decompression.

We propose DECA, a near-core accelerator that offloads tile de-sparsification and dequantization from CPU cores. DECA handles quantized formats from 1 to 8 bits, unstructured sparsity, and group quantization. We also introduce Tile External Preprocess and Load (TEPL), an ISA extension that invokes DECA out of order and hides CPU-accelerator communication latency.

Our evaluation of BF8 and MXFP4 with different sparsity levels shows that DECA accelerates compressed GeMMs by up to 4x on a simulated 56-core Xeon 4 server with HBM. It reduces next-token generation time for Llama2-70B and OPT-66B by 1.6x-2.6x over software decompression and by 2.5x-5.0x over an uncompressed baseline.

The contributions are the Roof-Surface performance model, the DECA near-core decompression accelerator, the TEPL extension for out-of-order invocation, and a simulation-based evaluation for compressed GeMMs in LLM inference.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 LLM Inference

LLMs consist of embedding, fully connected, and attention layers. Inference has a prompt phase that encodes input tokens and generates the first token, followed by a generation phase that produces output tokens. We focus on the low-arithmetic-intensity generation phase, which dominates end-to-end inference in many practical cases [Yua24]. CPUs with HBM and in-core GeMM engines are an attractive platform, so this work studies inference on modern CPU servers.

<span id="section-2-2"></span>

### 2.2 Model Compression

Quantization stores weights in lower-bit formats such as FP8 or FP4. Group quantization adds a per-group scaling factor. We evaluate BF8 and MXFP4, where MXFP4 uses four-bit values and a shared scale for every 32 weights. Sparsification removes weights close to zero. Unstructured sparsity allows arbitrary positions to be removed; a bitmask reconstructs nonzero positions. We evaluate densities from 50% to 5%.

For a dense BF16 model, a Q-bit model with density factor $d$ reduces model size by $16/(Qd+1)$, where the 1 accounts for the bitmask. We call this factor the Compression Factor (CF). Compression is performed offline; decompression is performed online and can affect performance.

<span id="figure-01"></span>

![Figure 1. Offline compression and online decompression.](../../papers/deca/figure-01.png)

**Figure 1.** Offline compression and online decompression.

<span id="figure-02"></span>

```text
// Decompress Ti+1
For each row r of Ti+1:
  apply AVX vector operations
// GeMM Ti
load Ti with AMX
compute Tout with AMX
// Decompress Ti+2 and GeMM Ti+1
```

**Figure 2.** Libxsmm compressed GeMM kernel pseudocode.

<span id="section-2-3"></span>

### 2.3 Matrix Extensions

Intel Advanced Matrix Extensions (AMX) add eight tile registers. Each register holds up to 16 rows and 64 bytes per row, interpreted as BF16 or INT8 elements. TMUL multiplies an activation tile $A$ and a weight tile $W^\top$. With batch size $N\leq16$, one operation performs $512N$ fused multiply-adds in 16 cycles. We refer to fused multiply-adds as FLOPs.

<span id="section-2-4"></span>

### 2.4 GeMM Decompression

TMUL accepts only specific dense formats, so compressed weights must be decompressed into tiles before multiplication. Libxsmm overlaps AVX decompression of tile $T_{i+1}$ with AMX processing of tile $T_i$ using double buffers. The decompression sequence uses permutes and masked vector expands. AVX instructions greatly outnumber AMX instructions because AVX processes cache-line-sized rows while AMX processes whole tiles.

<span id="section-3"></span>

## 3 Motivation

<span id="section-3-1"></span>

### 3.1 GeMMs in FC Layers Dominate Inference

<span id="table-01"></span>

![Table 1. Contribution of FC-layer GeMMs to next-token time.](../../papers/deca/table-01.png)

**Table 1.** Contribution of FC-layer GeMMs to next-token time.

Most next-token time is spent in FC-layer GeMMs: more than 95% on DDR5 and 85%-90% on HBM. Accelerating these GeMMs therefore provides a direct end-to-end benefit.

<span id="section-3-2"></span>

### 3.2 GeMMs in FC Layers are Bandwidth Bound

<span id="figure-03"></span>

![Figure 3. Traditional rooflines for a GeMM with N=4.](../../papers/deca/figure-03.png)

**Figure 3.** Traditional rooflines for a GeMM with $N=4$.

The uncompressed BF16 point is memory-bandwidth bound. Compression increases arithmetic intensity and moves points rightward. At high compression, observed performance falls below the roofline because AVX decompression cannot keep up with memory bandwidth or TMUL throughput.

<span id="section-3-3"></span>

### 3.3 Compressed GeMMs Can Introduce Inefficiencies

Roofline analysis does not show how much vector throughput is needed to remove the decompression bottleneck. Scaling the core risks overprovisioning vector units, superscalar width, and cache ports. The next section develops a model that guides this hardware support.

<span id="section-4"></span>

## 4 The Roof-Surface Model

We model the interaction of matrix, vector, and memory operations with a three-dimensional Roof-Surface and its two-dimensional projection, the Bounding Region Diagram (BORD).

<span id="section-4-1"></span>

### 4.1 The 3D Roof-Surface Performance Model

Memory supplies compressed tiles at $\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}$ tiles per second. Vector hardware decompresses at $\mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}$. Matrix hardware processes at $\mathrm{MOS}$. The achievable tile rate is

<span id="equation-01"></span>

$$
\mathrm{TPS} = \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

Since one TMUL tile performs $512N$ FMAs, the Roof-Surface equation is

<span id="equation-02"></span>

$$
\mathrm{FLOPS} = 512N\cdot\min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

The two kernel-dependent variables are $\mathrm{AI}_{\mathrm{XM}}$ and $\mathrm{AI}_{\mathrm{XV}}$. Their values form a kernel signature, so two kernels with the same signature have the same projected performance.

<span id="figure-04"></span>

![Figure 4. The 3D Roof-Surface model and its performance predictions.](../../papers/deca/figure-04.png)

**Figure 4.** The 3D Roof-Surface model and its performance predictions.

The surface has memory-, vector-, and matrix-bound regions. Measured points below the surface reveal the limiting resource, while points above it are not achievable.

<span id="section-4-2"></span>

### 4.2 The 2D Bounding Region Diagram

BORD projects the Roof-Surface onto the $\mathrm{AI}_{\mathrm{XM}}$-$\mathrm{AI}_{\mathrm{XV}}$ plane. Its boundary lines are $y=(\mathrm{MBW}/\mathrm{VOS})x$, $x=\mathrm{MOS}/\mathrm{MBW}$, and $y=\mathrm{MOS}/\mathrm{VOS}$.

<span id="figure-05"></span>

![Figure 5. BORDs for HBM and DDR systems.](../../papers/deca/figure-05.png)

**Figure 5.** BORDs for HBM and DDR systems.

The HBM system places most kernels in the vector-bound region, while DDR's smaller memory bandwidth enlarges the memory-bound region.

<span id="figure-06"></span>

![Figure 6. BORD for HBM with four times the vector throughput.](../../papers/deca/figure-06.png)

**Figure 6.** BORD for HBM with four times the vector throughput.

Even a fourfold increase in vector throughput does not remove the vector bottleneck for all kernels. Cores already spend most dynamic instructions on decompression and use 40%-80% of commit slots, making conventional scaling expensive.

<span id="section-5"></span>

## 5 DECA Overview and Out-of-Order Invocation

<span id="section-5-1"></span>

### 5.1 DECA Placement and System Integration

Each CPU core is associated with a DECA containing a processing element, control registers, and tile-output registers. The core configures quantization and sparsity through privileged stores. DECA reads compressed tiles through the L2, performs decompression, and writes ready-to-use tiles to tile-output registers.

<span id="figure-07"></span>

![Figure 7. DECA placement next to a core.](../../papers/deca/figure-07.png)

**Figure 7.** DECA placement next to a core.

DECA shares the L2 TLB and CPU virtual address space. Its state can be saved across context switches or trapped and reconfigured when another process uses it.

<span id="section-5-2"></span>

### 5.2 DECA-Core Cooperative Tile Processing

Two loaders and two tile-output registers provide hardware double buffering. A loader fetches data, a bitmask, and scaling factors. DECA loads, decompresses, and writes tile $T_i$ while the core reads and multiplies $T_{i-1}$, then starts fetching $T_{i+1}$.

<span id="figure-08"></span>

![Figure 8. DECA and CPU core cooperative tile processing.](../../papers/deca/figure-08.png)

**Figure 8.** DECA and CPU core cooperative tile processing.

<span id="figure-09"></span>

```text
TLoad TReg1, TOut1
TComp TReg2, TReg1
store metadata for T(i+1) to Loader2
fence
```

**Figure 9.** CPU pseudocode for memory-mapped DECA invocation.

<span id="section-5-3"></span>

### 5.3 ISA Support for Out-of-Order Invocation

Tile External Preprocess and Load (TEPL) combines metadata submission and tile loading. It returns when DECA has decompressed the tile into a core tile register. At most two TEPL instructions execute simultaneously, matching the two loaders.

<span id="figure-10"></span>

```text
TEPL TReg1, M(i+1)
TComp TReg2, TReg1
TEPL TReg2, M(i+2)
```

**Figure 10.** CPU pseudocode using TEPL instructions.

TEPLs enter a dedicated queue and execute as soon as their source registers and execution ports are available. Speculative invocation is safe because DECA does not update memory; a pipeline flush sends a squash signal. This removes fences, overlaps multiple tiles, and hides core-DECA communication.

<span id="section-6"></span>

## 6 DECA Microarchitecture Design

<span id="section-6-1"></span>

### 6.1 DECA Processing Element

<span id="figure-11"></span>

![Figure 11. DECA processing-element microarchitecture.](../../papers/deca/figure-11.png)

**Figure 11.** DECA processing-element microarchitecture.

Each processing element has two Load Queues (LDQ) and prefetchers, bitmask and scale-factor queues, a vector pipeline, look-up-table (LUT) arrays, an expansion stage, a scaling stage, and tile-output registers. The pipeline reads compressed data, expands nonzeros according to the bitmask, applies LUT-based dequantization and scaling, and writes a dense BF16 tile.

<span id="section-6-2"></span>

### 6.2 Quantitative Microarchitecture Design

The vector pipeline has width $W$ and latency $L$. The LUT array supports 8-bit quantization and stores values for each 8-bit input. Bitmask and scale-factor queues are dimensioned for a tile. For sparse inputs, fewer elements need processing, so the pipeline can naturally achieve higher throughput. The Roof-Surface model selects $W=32$ and $L=8$ as a balanced design.

<span id="section-7"></span>

## 7 Alternatives to DECA for Handling the Decompression Bottleneck

Scaling conventional vector resources requires more AVX units, wider vectors, larger superscalar width, and additional cache ports. In-core matrix extensions can support selected sparse or low-bit formats, but they need format-specific hardware and cannot adapt to future schemes. Decoupled vector accelerators avoid some core changes but do not exploit TMUL throughput for moderately sparse ML models. DECA keeps decompression separate, supports many formats through LUT configuration, and cooperates with the matrix unit with small ISA changes.

<span id="table-02"></span>

![Table 2. Comparison with other in- and near-core accelerators.](../../papers/deca/table-02.png)

**Table 2.** Comparison with other in- and near-core accelerators.

<span id="section-8"></span>

## 8 Methodology

We simulate a 56-core Xeon 4-like server at 2.5 GHz with either 260 GB/s DDR5 or 850 GB/s HBM, extending a Sniper-based simulator with DECA processing elements and TEPL queues. The baseline DECA uses $W=32$ and $L=8$.

The software baseline is Intel libxsmm. We replace its AVX decompression sequence with TEPL instructions and evaluate cascades of fully connected layers with about 250 million parameters. We test batch sizes from 1 to 16 and use the Intel Tensor Processing Primitives framework for Llama2 and OPT inference.

We evaluate Q16, Q8, and Q4 compression, with density from 50% to 5% for Q16 and Q8. Area is estimated with CACTI and published circuit models. Fifty-six DECA processing elements occupy about $2.51\,\mathrm{mm}^2$, less than 0.2% of a 56-core Xeon 4 die.

<span id="section-9"></span>

## 9 Evaluation

<span id="section-9-1"></span>

### 9.1 DECA for Compressed GeMMs

<span id="figure-12"></span>

![Figure 12. Compressed GeMM speedup for DDR and N=1.](../../papers/deca/figure-12.png)

**Figure 12.** Compressed GeMM speedup for DDR and $N=1$.

<span id="figure-13"></span>

![Figure 13. Compressed GeMM speedup for HBM and N=1.](../../papers/deca/figure-13.png)

**Figure 13.** Compressed GeMM speedup for HBM and $N=1$.

DECA reaches 1.7x speedup for DDR and 4.0x for HBM. Its performance is near-optimal because vector overheads are hidden. Similar results hold for batch sizes up to 16.

<span id="figure-14"></span>

![Figure 14. TFLOPS across compressions for DDR and N=4.](../../papers/deca/figure-14.png)

**Figure 14.** TFLOPS across compressions for DDR and $N=4$.

Sixteen DECA-augmented cores outperform 56 conventional cores, freeing cores for other workloads or power gating.

<span id="table-03"></span>

![Table 3. Component utilization for Q8, N=1, and HBM.](../../papers/deca/table-03.png)

**Table 3.** Component utilization for Q8, $N=1$, and HBM.

DECA raises memory utilization while keeping the vector decompression unit from becoming the bottleneck. This validates the Roof-Surface prediction.

<span id="figure-15"></span>

![Figure 15. DECA versus conventional vector scaling.](../../papers/deca/figure-15.png)

**Figure 15.** DECA versus conventional vector scaling for HBM and $N=1$.

Four additional AVX units or fourfold wider AVX units remain far below DECA because conventional systems do not scale superscalar width or L1 ports.

<span id="section-9-2"></span>

### 9.2 Design Space Exploration with Roof-Surface

<span id="figure-16"></span>

![Figure 16. HBM BORDs without DECA and with different DECA sizes.](../../papers/deca/figure-16.png)

**Figure 16.** HBM BORDs without DECA and with different DECA sizes.

The model selects $W=32,L=8$ as the smallest design that moves all kernels out of the vector-bound region. The overprovisioned $W=64,L=64$ design is less than 3% faster but has eight times more LUTs.

<span id="section-9-3"></span>

### 9.3 Analysis of DECA Integration and TEPLs

<span id="figure-17"></span>

![Figure 17. DECA integration features for HBM and N=4.](../../papers/deca/figure-17.png)

**Figure 17.** DECA integration features for HBM and $N=4$.

Reading compressed weights from L2, using a DECA prefetcher, writing to tile-output registers, and using TEPL progressively improve performance. At 5% density, TEPL doubles performance by hiding communication latency.

<span id="section-9-4"></span>

### 9.4 DECA for LLM Inference

<span id="table-04"></span>

![Table 4. Llama2-70B and OPT-66B next-token latency.](../../papers/deca/table-04.png)

**Table 4.** Llama2-70B and OPT-66B next-token latency.

For 128 input and output tokens, DECA reduces next-token time by 1.6x-2.6x over software decompression and provides 2.5x-5.0x speedup over the uncompressed baseline.

<span id="section-10"></span>

## 10 Other Related Work

Decoupled accelerators target sparsity, quantization, and attention, but often incur area, power, and data-movement costs. CPU-integrated accelerators reduce these costs. Cooperative vector-matrix processors include Tandem, AWS Trainium, TPUs, and GPUs with Tensor Cores and SIMT cores. GPU Tensor Cores also accept limited formats, so a DECA-like decompression engine could augment TMA and reduce pressure on shared memory.

<span id="section-11"></span>

## 11 Conclusion

For CPU platforms with in-core GeMM engines and HBM, this paper presented the Roof-Surface performance model, the DECA near-core ML-model decompression accelerator, and the TEPL ISA extension for out-of-order accelerator invocation. DECA accelerates compressed GeMMs and LLM inference while requiring less than 0.2% area overhead in the evaluated system.
