---
title: DECA Accelerator
createTime: 2026-08-24
permalink: /en/papers/deca/
---

> [Gerasimos Gerogiannis](https://dblp.org/pid/314/4553), [Stijn Eyerman](https://dblp.org/pid/99/4678), [Evangelos Georganas](https://dblp.org/pid/121/2450), [Wim Heirman](https://heirman.net/), and [Josep Torrellas](https://dblp.org/pid/t/JosepTorrellas). First submitted to arXiv on 2025-05-25 (current version v2, 2025-08-08). Published in the 58th IEEE/ACM International Symposium on Microarchitecture (MICRO 2025), pages 184-200, online 2025-10-17 and in print 2025-10-18. [arXiv:2505.19349](https://arxiv.org/abs/2505.19349). [Original PDF](/paper/deca.pdf). [DOI: 10.1145/3725843.3756073](https://doi.org/10.1145/3725843.3756073). [TeX source](https://arxiv.org/e-print/2505.19349). The full source title is “DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model”. The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

To alleviate the memory bandwidth bottleneck in Large Language Model (LLM) inference workloads, weight matrices are stored in memory in quantized and sparsified formats. Hence, before tiles of these matrices can be processed by in-core generalized matrix multiplication (GeMM) hardware engines, they need to be dequantized and de-sparsified. This is currently performed in software with vector operations. Unfortunately, this approach delivers only modest performance. Moreover, it is hard to understand how to improve the system, as the overall GeMM performance depends on the interaction between memory resources, vector units, and hardware matrix engines.

To improve the performance of LLM inference in advanced platforms equipped with in-core GeMM engines and HBM, this paper makes three main contributions. First, it develops an analytical performance model with a 3D visual representation that provides insights into how memory resources, vector units, and hardware matrix engines interact to deliver compressed GeMM performance. Second, it proposes *DECA*, a new near-core ML-model decompression accelerator. DECA offloads tile de-sparsification and dequantization from the CPU, producing ready-to-use tiles for in-core GeMM engines. Third, it introduces a new ISA extension that enables out-of-order invocation of the near-core accelerator. With this extension, accelerator and core computations can interleave and overlap with high-performance. Our evaluation shows that, in a simulated 56-core Xeon 4 server with HBM, DECA accelerates the execution of compressed GeMMs by up to 4x over the use of optimized Intel software kernels. Further, DECA reduces the next-token generation time of Llama2-70B and OPT-66B by 1.6$\times$—2.6$\times$.

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) are one of the most important Machine Learning (ML) workloads, excelling at tasks such as chatbots, translation, text summarization, and content creation [Zho24j, Zha23m, Kal23, Yao23d]. LLMs use transformers [Vas17] and mainly consist of multi-head attention and fully connected (FC) layers. The largest models contain trillions of parameters (weights) in the FC layers [Ope24a, Zha23d]. During inference, these weights have low reuse (e.g., in small batch scenarios), stressing not only the memory capacity of modern platforms but also their memory bandwidth [Yua24].

GPUs are regarded as the standard platform for LLM inference because of their high compute and memory bandwidth. However, recent advances introduced by Intel Xeon 4 servers (codenamed Sapphire Rapids (SPR)) [Bis21], make CPUs an additional attractive option for LLM inference. First, such processors are equipped with an in-core generalized matrix multiplication (GeMM) engine called TMUL [Int24a]. The TMUL serves the same purpose as the GPU Tensor Cores [Mar18b]. It is programmed with the AMX ISA extensions [Int24a] to perform GeMMs on matrix tiles. The result is an order of magnitude increase in GeMM computational throughput compared to relying solely on vector SIMD units. Second, SPR servers can be equipped with High Bandwidth Memory (HBM), increasing the available memory bandwidth by 3-4$\times$ over their DDR-based counterparts.

In SPR CPUs, we observe that, similar to GPUs [Yua24], LLM inference is memory-bandwidth bound. The large GeMMs in the FC layers account for more than 90% of the next token generation time for LLama2-70B [Tou23a]. Such GeMMs have low arithmetic intensity and load a large number of weights from main memory. To a large extent, accelerating LLM inference on CPUs means speeding-up these large GeMMs.

Deep neural network (DNN) model compression techniques [Lia21a, Den20a], such as low-bit weight quantization [Gho21a] and sparsification/pruning [Hoe21, Xia23b, Zhu23b] can improve GeMM performance: the amount of data that needs to be loaded from memory is reduced, leading to significant speedups in memory-bound kernels. Sadly, like systolic arrays [Jou23] and Tensor Cores [Xia23b], the TMUL cannot handle arbitrary quantization schemes or sparse patterns. Consequently, the SPR TMUL engine expects well-formed dense input tiles (i.e., zero values must be included) either in BF16 [Kal19] or INT8 format.

To benefit from both model compression and TMUL GeMM throughput, Intel has recently introduced specialized kernels in the libxsmm framework [Hei16]. Libxsmm uses a sequence of vector (AVX) instructions to read compressed tiles from memory, de-sparsify and/or dequantize them, and feed them to the TMUL AMX unit. This cooperative processing mode involves two different computational domains (vector and matrix), each with its own instructions (AVX and AMX), and functional units (SIMD units and TMUL).

We profiled the performance of the libxsmm kernels for different quantized and sparsified workloads. Our analysis shows that, although they are very effective for moderately compressed GeMMs and with the relatively low-bandwidth DDR memory, their performance degrades with HBM. This degradation cannot be explained using a traditional two-dimensional (2D) roofline performance model [Zha15] that only considers the memory bandwidth and the (matrix) compute throughput as bounding factors.

To guide performance optimization, we first construct an analytical performance model that captures the interactions between memory, matrix and vector resources. In contrast to the 2D roofline, this model has a 3D visual representation with a surface separating achievable from non-achievable performance. For this reason, we call the model *Roof-Surface*. The Roof-Surface offers useful performance insights and accurately attributes the libxsmm performance degradation to the AVX vector decompression sequence. Further, it reveals that overcoming the decompression inefficiencies would require a prohibitive scaling of the CPU core's resources.

To address this problem, this paper proposes *DECA*, a new near-core *accelerator of ML model decompression*. DECA offloads tile de-sparsification and dequantization from the CPU, producing ready-to-use tiles for the TMUL. DECA can be programmed to handle quantized number formats with any number of bits between 1 and 8, supports any level of unstructured sparsity, and supports group quantization [Gho21a]. The DECA microarchitecture performs decompression by utilizing a pipeline with advanced vector operations. Importantly, we use the *Roof-Surface* model, (1) to make decisions about the vector pipeline microarchitecture and (2) to perform design space exploration and derive a well-balanced DECA design.

We observe that if the CPU cores use regular memory-mapped load/store instructions to communicate with DECA, the communication latency gets exposed and hurts performance. To this end, we introduce a new ISA extension that hides the CPU-DECA communication latency by invoking the accelerator out-of-order. We call this extension *Tile External Preprocess and Load* (TEPL).

Our evaluation for two different low-bit quantization formats (BF8 and MXFP4) and different unstructured sparsity levels shows that DECA is very effective. In a simulated 56-core SPR with HBM, DECA accelerates the execution of compressed GeMMs by up to 4x over the optimized Intel libxsmm software kernels. In addition, by speeding-up the FC layers, DECA reduces the next-token generation time of Llama2-70B and OPT-66B [Zha22] by 1.6$\times$—2.6$\times$ over the software-only solution, and by 2.5$\times$—5.0$\times$ over the uncompressed baseline model.

This paper's contributions are:

- The *Roof-Surface* performance model that models the interaction between vector units, matrix units, and memory.
- The *DECA* near-core accelerator, designed to accelerate the de-sparsification and dequantization of compressed ML models.
- The Tile External Preprocess & Load (*TEPL*) extension that enables out-of-order invocation of near-core accelerators.
- A simulation-based evaluation of the performance of *DECA* for compressed GeMMs in LLM inference.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 LLM Inference

Large Language Models (LLMs) consist of different layers, such as Embedding layers, Fully-Connected (FC) layers and Attention layers [Vas17]. LLM inference has two phases [Pat23]. The first one encodes the input tokens and generates the first token (prompt phase). The second one generates the next output tokens (generation phase). In this work, we focus on executing the low arithmetic-intensity generation phase efficiently, since for many practical use cases it dominates the end-to-end LLM inference time [Yua24].

GPUs are regarded as the standard platform for LLM inference [Pat23, Su23a] because of their high compute and memory bandwidth. However, recent advances, such as HBM and in-core GeMM engines [Bis21], make CPUs an additional attractive option for LLM inference. There has been increasing research and industrial interest in making CPUs better at Machine Learning (ML) and scientific workloads, by either incorporating extensions or small accelerators on the CPU die  [Jeo23, Gon20, Gon22, Sir23, Nas22, Ore22]. For these reasons, in this paper, we focus on LLM inference on modern CPU servers.

<span id="section-2-2"></span>

### 2.2 Model Compression

For low arithmetic-intensity LLM FC layers, compressing the weight matrices reduces data movement and, therefore, can directly improve performance in both GPUs and CPUs. There are two main ways to compress an ML model [Han15, Zhu23b]:

- **Quantization** involves storing weights in a lower-bit format, e.g., FP8 or FP4 instead of FP16. Multiple quantization schemes exist [Lin23d, Zha24e, Kim24, Wei23]. Some of them additionally split weights in groups and introduce a per-group scaling factor (*group quantization*) to achieve higher accuracy. We evaluate two types of weight quantization in our work: BF8 (8-bit brain floating point) and MXFP4 [Bit23]. The latter uses a 4-bit floating point and group quantization with a shared scaling factor for every 32 weights (8 bit exponent); it has been shown to not degrade LLM accuracy [Bit23].
- **Sparsification** consists of eliminating (*pruning*) weights that are close to zero and/or that do not contribute much to the model's accuracy [Lec89, Bla20, Hoe21]. *Unstructured sparsity* does not impose restrictions on which weights can be removed. It achieves higher accuracy than structured sparsity for the same sparsity level [Liu21d, Fra23a]. In this work, we assume unstructured sparsity and a bitmask-based sparse format, to avoid storing zeros. To reconstruct the position of the non-zeros in the original weight matrix, a bitmask is used that has as many bits as the number of elements in the original matrix. The '1' bits in the bitmask indicate the location of the nonzeros, which are stored consecutively in a nonzero array. Recently proposed LLM weight pruning methods such as SparseGPT [Fra23a] have achieved unstructured sparsity levels of up to 60-70% without significant loss in accuracy. For traditional ML models like ResNet50 [He16], unstructured sparsity levels up to 95% are easy to achieve [Pes21]. Since we believe that LLM research advances will soon enable higher sparsity levels, we evaluate a large 50%-95% range of sparsity.

Models may be both sparse and quantized [Har24]. Starting from a dense BF16 model, a $Q$ bit quantized model with a density factor of $d$ (e.g., $d=10\%$ means only 10% of the weights are nonzeros), reduces the model size by a factor of $16/(Q\times d+1)$, where the '1' comes from the bitmask bit. We assume that the footprint of activations is negligible. We refer to this factor as *Compression Factor (CF)*.

The compression process is executed offline (e.g., after training). It is shown on the left part of Figure [1](#figure-01). In this paper, we assume an already compressed model that we want to use online for inference.

<span id="section-2-3"></span>

### 2.3 Matrix Extensions

There are several matrix extensions [Int22, Wil22, Car22a, Bha21] to improve the efficiency of matrix multiplication on CPUs. In this work, we use Intel's Advanced Matrix Extensions (AMX)[Int22]. AMX extends the register file with 8 matrix registers, called tile registers. Each one can hold up to 16 rows, with 64 bytes of data per row that can be interpreted as 32 2-byte elements (BF16) or 64 1-byte elements (INT8). Each tile can contain up to 1 KB of data.

Each core has the tile registers and a matrix multiplication TMUL unit that multiplies the tiles. To load/store data to/from the tile registers, AMX includes tload/tstore instructions. For the next token generation phase in LLMs with a batch size of $N\leq16$ and BF16, a weight tile $W$ contains $M=16$ rows each with $K=32$ columns. An activation tile $A$ contains $N$ rows each with $K=32$ columns. The TMUL performs the operation $A\times W^\top$ to produce an $N\times M$ output tile. The TMUL operation takes 16 cycles to execute regardless of the N value. and performs a total of $N\times K\times M =N\times32\times16=512N$ fused multiply-adds (FMA) - or equivalently $32N$ FMAs per cycle. For N>16, the TMUL throughput saturates at 512 FMAs per cycle, since the activation tile can hold no more than 16 rows. Any mention of FLOPs in this work refers to FMAs.

<span id="section-2-4"></span>

### 2.4 GeMM Decompression

The TMUL, similar to other GeMM engines [Mar18b, Jou23], can handle data in very specific data formats (i.e., BF16 or I8) and cannot handle unstructured sparsity. If a GeMM contains compressed weights, decompression is needed to produce tiles that conform to the TMUL requirements. Unlike compression, decompression is performed online (Figure [1](#figure-01)). Thus, it can impact performance.

<span id="figure-01"></span>

![Figure 1. Offline compression and online decompression.](../../papers/deca/figure-01.png)

**Figure 1.** Offline compression and online decompression.

<span id="figure-02"></span>

```text
............
//Decompress Ti+1
for(r=0 to 16):
{
  //Decompress
  //row r of Ti+1
  VectorOps AVX
  ...
}

//GeMM Ti
MatrixOps AMX
TLoad Ti
TComp Tout, Ti

//Decompress Ti+2
//GeMM Ti+1
............
```

**Figure 2.** Libxsmm compressed GeMM kernel pseudocode.

To achieve high performance in compressed GeMM kernels and hide the decompression overhead, Intel recently introduced a software solution integrated in the Libxsmm framework [Hei16] (Figure [2](#figure-02)). The decompression sequence is handled using AVX vector operations, while the actual GeMM is executed using AMX matrix operations. Libxsmm adopts a smart method to overlap the execution of the two: software allocates a double software buffer, and tries to keep it in the L1 cache. The output of the AVX decompression sequence for tile *Ti+1* is written in one of the two software buffers. At the same time, AMX instructions load data from the other software buffer that contains *Ti*, which has been previously decompressed by the AVX sequence. Overlapping AMX with AVX is enabled by out-of-order execution and dependencies are naturally honored.

The decompression sequence uses vector operations such as permutes for the decompression, and masked vector expands to insert zeros in the appropriate positions of the nonzero array. Although we omit the specifics due to limited space, the first takeaway is that decompression is done using AVX, utilizing a different "domain" (i.e., separate instructions and functional units) than AMX. The second takeaway is that the AVX dynamic instructions vastly outnumber the AMX ones, since AMX uses tile-sized operands (1KB), while AVX operates on cache-line sized ones (64B, one tile row).

<span id="section-3"></span>

## 3 Motivation

<span id="section-3-1"></span>

### 3.1 GeMMs in FC Layers Dominate Inference

Table [1](#table-01) shows the fraction of the next-token (i.e., generation) time spent in the GeMMs of the different Fully Connected (FC) layers of Llama2-70B [Tou23a] on an SPR server with either DDR5 or HBM. We show results for an uncompressed model with BF16 weights, with different numbers of input tokens and batch sizes ($N$). The rest of the time is spent on kernels such as attention, for which weight compression does not apply. We see that the time spent in such GeMMs is over 95% for DDR5 and 85-90% for HBM. Hence, accelerating these GeMMs can greatly improve the next-token time.

<span id="table-01"></span>

![Table 1. Contribution of the GeMMs of FC layers to the next-token time.](../../papers/deca/table-01.png){.paper-table-narrow}

**Table 1.** Contribution of the GeMMs of FC layers to the next-token time.

<span id="section-3-2"></span>

### 3.2 GeMMs in FC Layers are Bandwidth Bound

Figure [3](#figure-03) shows the roofline models for one of the large GeMMs of the FC layers in LLama2-70B for an SPR with either DDR5 or HBM, and N=4. We use the TMUL FLOPS limit ([Section 2.3](#section-2-3)) for the maximum achievable GeMM FLOPS in the compute-bound area. In this work, when calculating the Arithmetic Intensity (AI) in FLOPs per memory byte, we assume that the footprint of the weight matrices is much larger than that of activations, which is true for small values of N. The leftmost circle in both graphs, labeled as 'BF16', is our baseline uncompressed execution. We see that this execution is memory-bandwidth bound in both cases due to a low AI. This motivates model compression, to reduce the amount of data that needs to be read from memory.

<span id="figure-03"></span>

![Figure 3. Traditional rooflines for a GeMM with $N=4$.](../../papers/deca/figure-03.png)

**Figure 3.** Traditional rooflines for a GeMM with $N=4$.

<span id="section-3-3"></span>

### 3.3 Compressed GeMMs Can Introduce Inefficiencies

The other data points in Figure [3](#figure-03) represent compressed models with 4-bit quantization (MXFP4) or with 8-bit quantization (BF8) with density levels (i.e., fraction of nonzeros) ranging from 5% to 100%. Compression reduces the amount of data fetched from memory, which increases the AI, moving the circles to the right as the compression factor increases. For each design point, we show two circles: one at the *Observed* performance, and one on the roofline for the same AI. We call the latter *Optimal* performance.

We see that, as we increase the compression factor, the Observed and Optimal points increasingly diverge. In the DDR5 graph, the divergence appears at BF8 with 5% density. However, in the HBM graph, all the compressed models are below their Optimal performance; at BF8 with 5% density, the ratio between Optimal and Observed performance is 4.94x. This means that performance is limited by some inefficiency that is not captured by the roofline model. By manual profiling, we find that the root cause is the overhead of the AVX decompress instruction sequence. Effectively, the AVX SIMD processing units of the cores are unable to keep up with the memory bandwidth and/or the throughput of TMUL.

Considering the importance of LLM workloads, some form of hardware support for the decompression overhead could be justified. However, one has to be cautious when making changes in the resource-constrained CPU setting. The roofline model does not inform us on the required vector throughput improvement for the kernels to shift from being bounded by vector processing to being bounded by memory or matrix computation. There is a danger of constructing hardware solutions that are either underprovisioned or overprovisioned. To avoid this danger, in the next section, we propose an alternative analytical model. This model can theoretically guide the required hardware support for eliminating the decompression overhead in compressed GeMMs.

<span id="section-4"></span>

## 4 The Roof-Surface Model

To guide performance optimization for our kernels that involve matrix, vector, and memory operations, we develop a performance model that captures their interaction. This model, which we call *Roof-Surface*, has a three-dimensional (3D) visualization. We also present a 2D projection, called the *Bounding Region Diagram* (BORD).

<span id="section-4-1"></span>

### 4.1 The 3D Roof-Surface Performance Model

In cases where multiple interacting factors can affect performance, the slowest factor ends-up determining the performance. Thus, we should first express how fast (1) memory can provide compressed tiles (MEM), (2) vector hardware can process compressed tiles (VEC), and (3) matrix hardware can process decompressed tiles (MTX).

**Memory.** Memory can provide compressed tiles at a rate of $\mathrm{MBW}/\mathrm{Bytes}_{\mathrm{tile}}$ tiles per second, where $MBW$ is the memory bandwidth and $\mathrm{Bytes}_{\mathrm{tile}}$ is the number of bytes in a compressed tile. Since a compressed weight tile will be used for a single TMUL matrix operation, we refer to $1/\mathrm{Bytes}_{\mathrm{tile}}$ as matriX-to-Memory arithmetic intensity or $\mathrm{AI}_{\mathrm{XM}}$. It expresses how many matrix operations can be executed per byte loaded from memory, and it is very similar to the traditional arithmetic intensity used in the rooflines of Figure [3](#figure-03). The main difference is that its units are matrix operations per byte and not FLOPs per byte. In our setting, compression schemes with higher compression factors (CF) ([Section 2.2](#section-2-2)) have a higher $\mathrm{AI}_{\mathrm{XM}}$. Overall, the MEM rate in compressed tiles per second is $\mathrm{MBW}\cdot\mathrm{AI}_{\mathrm{XM}}$.

**The Vector Hardware.** The vector hardware decompresses tiles at a rate of $\mathrm{VOS}/\mathrm{VO}_{\mathrm{tile}}$, where $\mathrm{VOS}$ is the number of vector operations per second that can be executed by the architecture, and $\mathrm{VO}_{\mathrm{tile}}$ is the number of vector operations needed per tile. $\mathrm{VOS}$ is the vector throughput and is an architecture-dependent parameter. For example, for our SPR system, it is given by the product of the processor frequency ($f$), the number of cores ($c$), and the number of SIMD units per core. $\mathrm{VO}_{\mathrm{tile}}$ is a kernel-dependent parameter. Since only the weight matrix in a GeMM needs to be decompressed, $\mathrm{VO}_{\mathrm{tile}}$ effectively expresses how many vector operations are needed per matrix operation. We refer to $1/\mathrm{VO}_{\mathrm{tile}}$ as the matriX-to-Vector arithmetic intensity or $\mathrm{AI}_{\mathrm{XV}}$, since it expresses how many matrix operations can be executed per vector operation. Overall, the VEC rate in tiles per second is $\mathrm{VOS}\cdot\mathrm{AI}_{\mathrm{XV}}$.

**The Matrix Hardware.** The matrix hardware can perform $\mathrm{MOS}$ matrix operations per second. $\mathrm{MOS}$ depends on the architecture and not on the kernel. For example, in SPR systems, it is given by $fc/16$, since each core has a TMUL that takes 16 cycles to perform a tile multiplication. Overall, the MTX rate in tiles per second is simply $\mathrm{MOS}$.

**The Final Performance.** The final performance is determined by the lowest tile processing rate among the three rates considered. Specifically, the number of tiles per second (*TPS*) that the architecture can process is:

<span id="equation-01"></span>

$$
\mathrm{TPS} = \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}
$$

We can easily get the rate of FLOPs per second (*FLOPS*) by recalling from [Section 2.3](#section-2-3) that a TMUL tile operation corresponds to $512N$ FMAs. Thus:

<span id="equation-02"></span>

$$
\mathrm{FLOPS} = 512N\cdot \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}
$$

We call this equation the *Roof-Surface* equation. Any of the three terms inside the *min* clause can be the one limiting performance. For a given architecture (i.e., fixed *MBW*, $\mathrm{VOS}$, and $\mathrm{MOS}$), there are *two kernel-dependent variables* inside the *min* clause: $\mathrm{AI}_{\mathrm{XM}}$ and $\mathrm{AI}_{\mathrm{XV}}$. These are the kernel's "signature"—if two kernels have the same signature, they have the same projected performance. In contrast, in the roofline model, the kernel signature is just one variable: the traditional FLOP-to-memory AI. Now, the illustration of the performance model can no longer be done in the two dimensions of Figure [3](#figure-03) (FLOP-to-memory AI and FLOPS). We need three dimensions: one for $\mathrm{AI}_{\mathrm{XM}}$ (x dimension), one for $\mathrm{AI}_{\mathrm{XV}}$ (y dimension), and one for FLOPS (z dimension).

<span id="figure-04"></span>

![Figure 4. (a) The 3D Roof-Surface model. (b) The optimal performance based on the Roofline (R-L), the Roof-Surface (R-S), and real performance measurements in TFLOPs.](../../papers/deca/figure-04.png)

**Figure 4.** (a) The 3D Roof-Surface model. (b) The optimal performance based on the Roofline (R-L), the Roof-Surface (R-S), and real performance measurements in TFLOPs.

Figure [4a](#figure-04) shows the result of plotting Equation [2](#equation-02) (for N=4,HBM) in three dimensions to form the *Roof-Surface* plot. A Roof-Surface plot has three regions, depicted in different colors. In each of the regions, a different term of the Roof-Surface Equation is the smallest one, and thus bounds performance. The operation points below the blue subsurface are bound by the MTX factor, the ones below the green subsurface are bound by the MEM factor, and the ones below the orange subsurface are bound by the VEC factor. Kernel performance is depicted by points in the 3D space. The achievable performance is bounded by the overall surface, rather than by a line like in the roofline model. For this reason, we call the model Roof-Surface. Points above the overall surface are not achievable.

Figure [4a](#figure-04) also includes red points that correspond to the observed performance points for different compression schemes. We see that the red points under the VEC-bound region (MXFP4, BF16_10%, BF8_5%) are very near to the top of the corresponding tangent triangles (i.e. almost exactly on the roofsurface). This visually reveals that they are bounded by vector operations. The red point in the MEM-bound region (BF16_30%) is slightly below the roofsurface, revealing that, for this point, a non-plotted factor such as memory latency is leaving a little bit of performance on the table.

In Figure [4b](#figure-04) we show the optimal performance values as predicted by the roofline (R-L) and the Roof-Surface (R-S) models, and the real observed values. For almost all kernels, the Roof-Surface produces accurate performance bounds, while the roofline can be way off. If we were to plot many of the roofline predictions on the 3D space they would float above the roofsurface. Note that for kernels BF8, BF16_50%, and BF16_30%, the performance estimates of R-L and R-S are the same. The reason is that these kernels are classified as MEM-bound by both models.

<span id="section-4-2"></span>

### 4.2 The 2D Bounding Region Diagram

We introduce an easier to visualize 2D representation of the Roof-Surface plot that we call the Bounding Region Diagram (BORD). BORD is the projection of the roofsurface on the xy plane. A BORD does not depict FLOPS information, but accurately identifies which one of the plotted factors bounds the performance of a given kernel.

<span id="figure-05"></span>

![Figure 5. 2D bounding-region diagrams (BORD).](../../papers/deca/figure-05.png)

**Figure 5.** 2D bounding-region diagrams (BORD).

Figure [5a](#figure-05) shows the BORD for HBM SPR. The figure shows the equations of the lines that separate the three regions. They are: $y=(\mathrm{MBW}/\mathrm{VOS})x$, $x=\mathrm{MOS}/\mathrm{MBW}$, and $y=\mathrm{MOS}/\mathrm{VOS}$. It also shows the positions of the different compressed GeMM kernels that use BF8 and MXFP4 from Figure [3](#figure-03)b, and of additional kernels that use BF16 with different density levels. We observe that the vast majority of kernels are VEC-bound. To reach the performance of the roofline in Figure [3](#figure-03)b, these points should be pushed away from the VEC-bound region.

Figure [5b](#figure-05) shows the BORD for DDR SPR, which has a smaller *MBW* value. Now, the area of the MEM-bound region increases. The MTX-bound region is no longer visible for the $\mathrm{AI}_{\mathrm{XM}}$ and $\mathrm{AI}_{\mathrm{XV}}$ value ranges we are plotting in the BORD. Its area is consumed by the MEM region. The BORD also shows that all of our kernels except BF8 with 20% and lower density are in the MEM-bound area or very close to it. This explains that the software decompression solution reaches the roofline in most design points of Figure [3](#figure-03)a.

<span id="figure-06"></span>

![Figure 6. 2D BORD for HBM with 4x VOS.](../../papers/deca/figure-06.png){.paper-figure-half}

**Figure 6.** 2D BORD for HBM with 4x VOS.

Finally, Figure [6](#figure-06) shows the BORD when we take the HBM SPR variant and increase the vector throughput in VOS by 4x, in an attempt to eliminate the vector bottleneck. When compared to Figure [5a](#figure-05), we see that the area of the VEC-bound region decreases and the MEM-bound region covers more kernels. However, even a 4x VOS increase is not enough to make all kernels not VEC-bound.

We found that in the HBM SPR variant of Figure [5a](#figure-05), cores typically spend over 95% of their dynamic instructions on tile decompression, and that cores are already using 40-80% of their commit slots. Hence, increasing the VOS by 4x would require not only a 4x increase in the number of SIMD AVX units, but also a prohibitive increase in the core's superscalar width. We further discuss the limitations of this and other conventional solutions (such as increasing the vector width without increasing the number of AVX units) in [Section 7](#section-7) and evaluate those limitations in [Section 9](#section-9).

<span id="section-5"></span>

## 5 DECA Overview and Out-of-Order Invocation

The previous analysis reveals that, to hide the decompression overheads with a conventional solution, one would need a very expensive scaling of the general-purpose core's resources. This motivates us to propose *DECA*, a *near-core decompression accelerator for ML models*. DECA offloads vector processing for decompression from the cores. In this section, we first describe DECA's integration. We then introduce a new mechanism and ISA extensions for efficiently overlapping the operation of CPU cores and near-core accelerators.

<span id="section-5-1"></span>

### 5.1 DECA Placement & System Integration

We envision a processor to have a DECA associated with each core as shown in Figure [7](#figure-07). A DECA has a memory-mapped interface that allows the core to write commands and read data. A DECA has a processing element (PE), control registers, and tile output (*TOut*) registers. The core uses privileged stores to the control registers to configure the PE to perform decompression of tiles with a given quantization scheme and with or without sparsity. A configuration includes filling look-up tables (LUTs) that DECA employs for efficient dequantization ([Section 6](#section-6)).

<span id="figure-07"></span>

![Figure 7. DECA placement next to a core.](../../papers/deca/figure-07.png){.paper-figure-half}

**Figure 7.** DECA placement next to a core.

The DECA PE reads a compressed tile from memory, processes it, and then writes the decompressed tile to the TOut registers. Then, the CPU core reads the TOut registers and uses the data to execute the GeMM using AMX instructions. The PE accesses memory through the L2, issuing both regular loads (but never stores) and prefetch requests, generated by a prefetcher integrated in the PE. DECA shares the L2 TLB with the core like prior work [Gon22, Ger23, Sir23] and, therefore, uses the virtual space of the CPU core.

A DECA can potentially be used by multiple processes. One approach is to save and restore the DECA state on context switches. Alternatively, we propose that DECA retains its state across context switches and when a new process attempts to use DECA, it causes a trap to the OS, which saves the state and reconfigures DECA.

<span id="section-5-2"></span>

### 5.2 DECA-Core Cooperative Tile Processing

To execute GeMMs with high performance, we introduce a mechanism that overlaps vector operations in a DECA with AMX operations in a CPU core using hardware double buffering. The design is shown in Figure [8](#figure-08). A DECA has two Loader modules and two TOut registers. A Loader reads a compressed tile from the memory system, which includes three data structures: the data, a bitmask, and scaling factors. A Loader can also issue prefetches to load a tile in advance. The journey of a tile involves DECA loading it into a Loader (D1 in Figure  [8](#figure-08)), decompressing it in the DECA vector pipeline (D2), and storing it in a TOut register (D3). Then, the core reads it (C1), uses it to perform the AMX operation (C2), and prompts a Loader (C3) to initiate the fetching of the next tile by passing the starting address and the length of the three data structures of the tile. As shown in the figure, the double buffers enable overlapping of the operations on two tiles. While the core is reading and processing Tile *i-1*, DECA reads, processes, and writes out Tile *i*. After the core finishes *i-1*, it triggers the fetching of Tile *i+1*.

<span id="figure-08"></span>

![Figure 8. DECA-CPU core cooperative tile processing.](../../papers/deca/figure-08.png)

**Figure 8.** DECA-CPU core cooperative tile processing.

We explore two options for a CPU core to communicate with a DECA. The first one uses regular stores to the memory-mapped DECA interface; the second uses ISA extensions that we describe in [Section 5.3](#section-5-3). Using the first approach, Figure [9](#figure-09) shows the pseudocode of the core as it processes tiles as shown in Figure [8](#figure-08). The key instructions are those in Lines 4-6. The core uses *TLoad* (an AMX instruction) to load tile $T_{i-1}$ from a DECA TOut register into a tile register TReg$_1$ (Line 4). It then uses this tile in a *TComp* instruction (an AMX instruction that performs a GeMM), saving the output in a tile register TReg$_2$ (Line 5). Finally, it writes the metadata for tile $T_{i+1}$ (shown as $M_{i+1}$) to a memory-mapped register in DECA's Loader2 using a plain store. The write prompts Loader2 to initiate the fetch of tile $T_{i+1}$. In parallel with Lines 4-6, DECA is decompressing $T_{i}$.

<span id="figure-09"></span>

```text
............
DECA_ldr1 <- ST M_i
Fence
TReg_1 <- TLoad T_i-1
TReg_2 <- TComp TReg_1
DECA_ldr2 <- ST M_i+1
Fence
TReg_1 <- TLoad T_i
TReg_2 <- TComp TReg_1
............
```

**Figure 9.** CPU core pseudocode for store-based DECA invocation.

<span id="figure-10"></span>

```text
............
TReg_1 <- TEPL M_i-1
TReg_2 <- TComp TReg_1
TReg_1 <- TEPL M_i
TReg_2 <- TComp TReg_1
TReg_1 <- TEPL M_i+1
TReg_2 <- TComp TReg_1
............
```

**Figure 10.** CPU pseudocode for TEPL-based DECA invocation. The architectural tile registers TReg$_1$ and TReg$_2$ get renamed to different physical tile registers in each iteration.

Figure [9](#figure-09) also shows a piece of the previous iteration (Line 2) and of the subsequent iteration (Lines 8-9). To prevent incorrect memory operation reordering, we add a memory fence per iteration. Specifically, the load of tile $T_{i}$ (Line 8) should not execute before the metadata for $T_{i}$ is written to the control register in DECA's Loader1 (Line 2), which resets TOut Register 1 and initiates the tile fetch from memory. Since these two instructions do not depend on each other, we place a fence in Line 3. There is a fence in each iteration.

Unfortunately, this approach is likely to deliver limited performance for two reasons. First, each iteration has a fence that prevents cross-iteration overlap. Second, within an iteration, no instruction overlaps: the instructions in Lines 4 and 5 have a true dependence, and the store in Line 6 can only perform the update when it is at the head of the reorder buffer (ROB). The execution of all instructions is serialized, as if the core was in-order. As a result, in every iteration, the latency of the communication between core and DECA (both the load and the store) is fully exposed.

<span id="section-5-3"></span>

### 5.3 ISA Support for Out-of-Order Invocation

To reinstate out-of-order execution and hide the core-DECA communication, we propose a different approach that relies on an extension to the CPU AMX ISA. We call the extension *Tile External Preprocess and Load* (TEPL). The main idea is to eliminate the per-iteration fence in Figure [9](#figure-09) by combining, in hardware, the instructions in Lines 2 and 8 into a single instruction. This instruction updates the control register of a loader with metadata, triggering a tile fetch, and only returns to the core when DECA has decompressed the tile and stored it in a core tile register (e.g., TReg$_1$).

A TEPL instruction takes as arguments a source register with the metadata for a tile, and a destination core tile register. The metadata is transferred to the DECA to initiate decompression. Moreover, the maximum number of TEPL instructions that can execute at any point in time is equal to the total number of DECA Loaders (i.e. two). A structural hazard prevents more TEPLs from executing. This is done to avoid overwriting accelerator invocations, since each DECA loader is able to handle only one tile at a time.

With this design, the code in Figure [9](#figure-09) is rewritten as Figure [10](#figure-10). Fences are removed and an iteration has only two instructions (e.g., Lines 4,5). There are no register dependencies between the iterations because TReg$_1$ and TReg$_2$ are renamed. However, a structural hazard causes the TEPL in Line 6 to stall until one of the previous two completes.

A context switch can only occur in between two instructions. Hence, the DECA state that needs to be saved and restored when a new process attempts to use the DECA is only the DECA control registers and LUTs, and not any tile data.

To support these instructions, the core has a *TEPL Queue* akin to a load-store queue, and two *TEPL execution ports*, each leading to a DECA loader. As a TEPL instruction $i$ enters the ROB, it is deposited in this Queue. When its source register is available and there is a free TEPL execution port, $i$ is issued to the DECA.

To attain high performance, TEPLs are issued to the DECA as soon as possible—they do not wait until they reach the ROB head. Hence, like a load instruction, they execute speculatively and out-of-order. Invoking a DECA speculatively is always safe, as a DECA does not update memory state. If the core needs to flush the pipeline (e.g., on a branch misprediction or exception) while a TEPL instruction is outstanding, the core sends a squash signal to the DECA. At that point, the DECA aborts any tile operation in progress, no matter the state it is in. The core may safely reissue the same TEPL.

Overall, this design hides the communication between the core and DECA. The core executes without fences and overlaps the operation on multiple tiles. TEPLs are not only useful for DECA. A core can potentially use them to communicate with other DECA-like near-core tile preprocessing accelerators.

<span id="section-6"></span>

## 6 DECA Microarchitecture Design

We now describe the microarchitecture that enables DECA to sustain high decompression performance and, at the same time, support a rich set of compression schemes. For simplicity, in the rest of the paper, we assume that DECA's output tile is in BF16 format. DECA can be trivially configured to produce I8 output tiles.

<span id="section-6-1"></span>

### 6.1 DECA Microarchitecture

Figure [11](#figure-11) displays the DECA PE microarchitecture. To understand it, we describe its multiple components.

<span id="figure-11"></span>

![Figure 11. DECA PE microarchitecture.](../../papers/deca/figure-11.png)

**Figure 11.** DECA PE microarchitecture.

**Accessing Memory.** DECA has two Loaders, each composed of a *Load Queue* (LDQ) and a prefetcher (PF). The LDQ accesses memory to read compressed weights, bitmasks, and scaling factors. The memory address bases and lengths of these structures are part of the metadata provided by the CPU on DECA invocation. When a requested cache line arrives from memory, depending on which of the three types of data it contains, it is placed in the *Sparse Quantized Queue* (SQQ), *Bitmask Queue*, or *Scale Factor Queue*. The PF observes the address bases and lengths used for a Tile, and predicts the ones for future Tiles. The PF then generates prefetch requests that will bring this data to the L2 cache. The PF aggressiveness is dynamically adjusted so that a high L2 MSHR occupancy is preserved.

**Pipeline Stages.** The pipeline is split into three stages, responsible for dequantization, expansion (i.e., de-sparsification) and scaling. Each stage has its own output register to enable pipelining (SD, DD and TOut). The Dequantization stage reads values from the SQQ, dequantizes them using an array of $L$ Lookup Tables (*LUT Array*), and writes dequantized BF16 values to the *Sparse Dequantized* (SD) register. These values are potentially sparse—stored contiguously with zero values skipped. The Expansion stage de-sparsifies data by inserting zeros in the positions indicated by the bitmask. This operation is performed using a crossbar (*XBAR*) that is controlled using expansion indices. The latter are generated from the bitmask using the *Parallel Prefix Sum* circuitry. The result is written to the *Dense Dequantized* (DD) register, which contains dense (i.e., with explicit zeros) dequantized data. Finally, if group quantization is used, the Scaling stage applies appropriate scaling to the BF16 values by multiplying them with the scaling factors. It then writes the final values to the *TOut* register. The critical path is shown with red arrows in the figure.

**Duplicated Modules.** A DECA PE contains two Loaders and two TOut registers to enable the overlapping of DECA and CPU operation. Hence, as shown in Figure [11](#figure-11), the PE replicates LDQ, PF, the input queues (SQQ, Bitmask queue, and Scale Factor queue), and TOut. One Loader can be supplying data while the pipeline is processing data that was provided by the other Loader. The bitmask processing circuitry mainly performs additions of 1-bit data, and is also duplicated so we can hide its latency. The rest of the pipeline is not duplicated and used by one Loader-TOut pair at a time.

**Vector Operations (vOps).** It takes multiple cycles to generate a decompressed BF16 tile, which always contains 512 BF16 elements. This is because the pipeline generates output chunks of *W* elements at a time, each using one DECA Vector Operation (vOp). In the absence of pipeline bubbles, a new chunk is generated every cycle. A vOp reads data from the SQQ, executes in the pipeline stages and finally writes W elements to a TOut. vOps exploit pipelining: if a vOp enters the Expansion stage, the next vOp can enter the Dequantization stage. The vOps of a tile are processed in-order and can enter the pipeline as long as (1) their input has arrived from memory and (2) the first pipeline stage is free.

Without sparsity, a vOp reads W elements from the SQQ. With sparsity, less than W elements are needed, since the SQQ does not contain zero values. We refer to the elements that a given vOp needs to read from the SQQ as the vOp's window (*Wnd*). To determine the size of a Wnd, the POPCNT circuitry counts the number of "1s" in the bitmask, and determines the end of the current Wnd and the start of the next Wnd. The latter is the next SQQ position from which data will be read into the pipeline.

**LUT Array Organization.** The DECA dequantization stage supports up to 8-bit quantized numbers, which can represent a maximum of 256 different values. For this reason, each of the $L$ LUTs in the LUT array stores 256 ($2^8$) BF16 values. Dequantizing an 8-bit value corresponds to a lookup using the 8-bit value as the LUT address. DECA contains $L$ LUTs to allow for parallel dequantization of multiple values. Each LUT is internally divided into 4 smaller sub-LUTs, each one with a read port and 64 ($2^6$) entries. If the quantized data bitwidth is 6 bits or less, the 4 sub-LUTs can be used in isolation to enable 4 reads from one 256-entry "big" LUT. For less than 6-bit quantization, some of the LUT entries are redundant and will not be used at runtime.

**Bubbles and the Roof-Surface.** We set the number of "big" LUTs to $L < W$ to limit DECA's area. If the Wnd of a vOp is larger than $L$ elements, the vOp occupies the Dequantization stage for more than one cycle. This injects one or more *bubbles* in the pipeline, which reduce the vOp throughput. For example, the Wnd of a dense 8-bit quantization scheme is W and, therefore, a vOp will always require $W/L$ cycles for dequantization. Although setting $L$ < $W$ limits the DECA throughput for dense quantization schemes, this is not a major concern because dense schemes like BF8_100% and MXFP4 require less vector throughput (i.e., VOS) in order to escape the vector (VEC) region. This can seen in the BORDs of Figure [5](#figure-05).

On the other hand, sparser schemes require a higher VOS to escape the VEC-bound region. Luckily, this is naturally achieved by the DECA pipeline: the probability that the Wnd of a vOp is larger than $L$ decreases with sparsity. Thus, fewer bubbles are introduced for sparse schemes, naturally achieving higher throughput than their dense counterparts for the same $L$. The same behavior is achieved for lower bitwidth schemes because they can perform more than $L$ reads in parallel from the LUT array.

**Generality and Performance.** DECA supports quantization formats of 8 bits and lower, group quantization, and unstructured sparsity, which cover most current and likely future model compression schemes. DECA's design is flexible, since by changing the values in its LUT array and/or using different scale factors, it enables the support for a rich set of formats without redesigning the hardware. Additionally, individual stages can be skipped if they are unneeded (e.g., quantization without sparsity). In terms of performance, the main benefit of DECA is that it replaces multiple vector (AVX) instructions by a single vOp that performs the whole decompression: dequantization, expansion, and scaling. The decreased vOp count *increases the $\mathrm{AI}_{\mathrm{XV}}$* ([Section 4](#section-4)), moving the points away from the VEC region. Finally, note that DECA efficiently dequantizes only the non-zeros, which is hard to do on a CPU with a traditional vector ISA due to data dependent branches during expansion.

<span id="section-6-2"></span>

### 6.2 Quantitative Microarchitecture Design

In previous sections, we discussed how the Roof-Surface model influenced DECA's design *qualitatively*. For example, it suggested designing a higher-performance accelerator by optimizing the $\mathrm{AI}_{\mathrm{XV}}$, and not just by blindly scaling the CPU's width and AVX resources. We now discuss how it can be used *quantitatively* to dimension DECA's $W$ and $L$ parameters and derive a well-balanced design.

Consider Equation [2](#equation-02). We should express how the parameters in the equation depend on $W$ and $L$. In reality, only $\mathrm{AI}_{\mathrm{XV}}$ depends on $W$ and $L$. $\mathrm{VOS}$ is $c\cdot1\cdot f$, since each of the $c$ CPU cores has one DECA PE that can complete at most one vOp per cycle and operates at the core frequency. On the other hand, the $\mathrm{AI}_{\mathrm{XV}}$ of different kernels depends on DECA's $W$ and $L$ parameters. To calculate it, we need to add-up the number of vOps that are needed per tile and the number of bubbles that are generated per tile.

<span id="table-02"></span>

![Table 2. Comparison of DECA with other in/near-core accelerators.](../../papers/deca/table-02.png)

**Table 2.** Comparison of DECA with other in/near-core accelerators.

The number of vOps per tile is $\#\mathrm{vOps}=512/W$, since each tile has 512 elements and we produce $W$ with a single vOp. We express the number of bubbles per tile as $\#\mathrm{bbl}=\#\mathrm{vOps}\cdot\mathit{bpv}$, where $\mathit{bpv}$ is the number of bubbles per vOp. Since bubbles can only be generated due to insufficient resources in the Dequantization stage, we use $L_q$ to denote the maximum number of elements that can be dequantized in a cycle. $L_q$ is equal to $L$ for 8-bit quantization schemes, $2*L$ for 7-bit, and $4*L$ for 6-bit and below. Without sparsity, $\mathit{bpv}=\lceil W/L_q\rceil-1$. With sparsity, the bubble generation is not deterministic, as it depends on the number of nonzeros in a compressed tile. For a matrix of density $d$, if we assume that nonzeros are uniformly distributed, then the number of nonzeros in $W$ consecutive matrix elements is a binomial distribution with parameters $W$, $d$. We compute the expected number of bubbles as:

$$
\begin{aligned}
\mathit{bpv} &=  \sum\nolimits_{k=0}^{\frac{W}{L_q}-1} k \cdot [F((k+1)L_q; W, d) - F(kL_q; W, d)]
\end{aligned}
$$

where $F(i;W,d)$ is the binomial cumulative distribution function. Finally, the $\mathrm{AI}_{\mathrm{XV}}$ is given by $1/[\#\mathrm{vOps}\cdot(1+\mathit{bpv})]$.

Now we have all we need to perform an analytical Design Space Exploration (DSE) using the Roof-Surface model. For example, we can plot the BORDs of different ($W$, $L$) pairs and pick the one that pushes all kernels out of the VEC-bound area at the minimum DECA hardware cost (see [Section 9.2](#section-9-2)).

<span id="section-7"></span>

## 7 Alternatives to DECA for Handling the Decompression Bottleneck

In [Section 5](#section-5) and [Section 6](#section-6), we discussed how DECA can sustain high decompression performance while maintaining support for a rich set of compression schemes. We now discuss the shortcomings of two alternatives to using DECA: scaling the CPU core's vector resources or using other in/near-core accelerator designs.

1. **Traditional scaling of the CPU vector resources.** Our Roof-Surface analysis of [Section 4](#section-4) reveals that, to hide most of the decompression overheads, one would need more than a 4x increase in vector throughput (VOS). Supporting such increase by conventional scaling of a core's vector resources is very challenging. One approach would be to increase the number of SIMD AVX vector units by more than 4x. However, as discussed in [Section 4](#section-4), cores are already using 40-80% of their commit slots. Hence, such a substantial increase in the number of vector units would require a major increase in the superscalar core width. This is undesirable, as a core's area scales quadratically with the superscalar width [Pal97]. Another approach would be to increase the SIMD AVX vector width. This requires new AVX instructions that operate on multi-cache line operands of at least 2048. However, supporting AVX2048 would require significant ISA and pipeline changes (e.g., redesigning wider versions of all the vector instructions, new register files, etc.). In addition, feeding the core with so large vectors would require, at a minimum, increasing the number of ports in the L1 cache. This would in turn hurt the L1 access latency and the core's cycle time, affecting the core's performance for general-purpose workloads. In [Section 9](#section-9), we quantitatively compare DECA to these alternatives.
2. **In-core accelerators using matrix operations.** Traditional matrix units such as the TMUL, and RASA [Jeo21] cannot deal with compressed tiles. To avoid the need for tile decompression, some in-core accelerator designs [Jeo23, Pel24, Nvi24d] such as VEGETA [Jeo23] augment matrix units with support for specific structured sparsity patterns. Such an approach increases hardware complexity in the core (e.g., larger matrix unit, more architectural registers, changes in register renaming). Further, although this approach can increase the matrix throughput ($\mathrm{MOS}$) by skipping some computations involving zero values, our *Roof-Surface* analysis of [Section 4](#section-4) reveals that such an increase is unnecessary for our kernels: most of them become bound by memory after escaping the vector-bound region. Other designs augment the matrix units with native support for more efficient lower bit quantization formats [Jan24, Nvi24d]. However, such designs require extra hardware to be included in the matrix unit for each one of the supported formats. Further, the hardware needs to be redesigned if a new, previously unseen, quantization format emerges. Instead, DECA can support a very rich set of quantization formats without requiring extra hardware for each one of them (i.e., by changing the values in its LUT array and/or using different scale factors). DECA's flexibility enables support for future quantization formats without redesigning the hardware. In principle, all the DECA hardware (i.e., LUT array, expansion and scaling circuitry, etc.) could be integrated in the matrix multiplication unit. However, the decoupled approach of DECA has some important advantages. First, it adds flexibility: the output of the decompressor can also be fed into another accelerator, stored back to memory, or be used for other use-cases. Second, by attaching the accelerator with its own Loaders at the L2, DECA can more effectively fetch and prefetch data. Finally, the CPU core ISA and pipeline changes required are minimal, decreasing the risk of impacting the core's performance in general-purpose workloads.
3. **In/near-core accelerators using vector operations.** SPADE [Ger23] and SAVE [Gon20] are accelerators for sparse applications designed to be integrated with CPUs. However, instead of relying on matrix units, they use vector units to execute the actual GeMM. While this approach might work for highly sparse matrices, utilizing the high throughput of matrix units is necessary for the moderately sparse matrices found in machine learning models [Yan24m].

Table [2](#table-02) summarizes the unique combination of characteristics enabled by DECA, when compared to other state-of-the-art in/near-core accelerators. First, DECA is the first design that offers support for a rich set of quantization schemes combined with structured or unstructured sparsity. At the same time it enables high GeMM throughput, by cooperating with the TMUL matrix units. Second, through speculative invocation, it is the first near-core accelerator design that enables fine-grained interleaving with the core. Finally, it introduces only few changes to the core's pipeline, which can be reused for other near-core accelerators ([Section 5.3](#section-5-3)).

<span id="section-8"></span>

## 8 Methodology

**Simulation and System Parameters.** To evaluate our work, we simulate a 56-core server with SPR-like parameters using an internal simulator based on Sniper [Car14] with full support for AMX. We evaluate the DDR5-based and the HBM-based designs with about 260GB/s and 850GB/s achievable memory bandwidth, respectively. We extend the simulator with: (1) DECA PEs, and (2) a TEPL queue and ports to support TEPLs in the core pipeline. Both cores and DECA PEs run at 2.5GHz. Our baseline PE is dimensioned with W=32 and L=8, but we also evaluate other options in [Section 9.2](#section-9-2).

**Software and DECA Control Code Generation.** We use the Intel Libxsmm compressed GeMM kernels ([Section 2.4](#section-2-4)) as our software baseline. To invoke DECA, we modify the libxsmm JIT compiler by replacing the AVX decompression sequence with TEPL instructions.

To evaluate the effectiveness of DECA for compressed GeMMs in isolation, we implement a large cascade of Fully Connected (FC) layers (without other types of layers) and use Parlooper [Geo23] for loop parallelization. The weight matrices in those layers have $\approx250$ million parameters, similar to the large FC layers of Llama-2-70B. Libxsmm and Parlooper are already integrated in the Intel Tensor Processing Primitives (TPP) Framework [Geo21a], which supports end-to-end Llama-2 and OPT inference on CPUs. Hence, we use TPP as is for software-only LLM inference, and by invoking the TEPL-augmented libxsmm kernels for inference with DECA. We test batch sizes of 1-16. Our simulator is compatible with all frameworks.

**Compression Schemes.** In our evaluation, we refer to BF16, BF8, and MXFP4 as Q16, Q8, and Q4. We limit the compression schemes we evaluate to these, since these are the ones for which libxsmm already includes support. We also evaluate unstructured sparsity with weight density ranging from 50% to 5% for Q16 (only sparsity) and Q8 (quantization plus sparsity). The Q4 sparse kernels are currently not included in libxsmm, so we don't have reference data to compare with DECA performance. For end-to-end Llama-2-70B and OPT-66B inference, the uncompressed Q16 baseline, Q16 with 50% density (Q16_50%) and Q8_100% do not fit in the 64GB of HBM. Hence, we simulate a larger HBM capacity for those schemes. Note that the Q4 performance is also representative of INT4 compression schemes with scaling factors such as AWQ [Lin23d].

**Area estimation.** We estimate the area of our proposed DECA design with W=32 and L=8. For the memory structures (e.g., LDQ and SQQ), registers, and LUT array, we use CACTI [Bal17]. For the crossbar and for the BF16 multipliers, we use numbers from [Cak15] and  [Zha19j], respectively. We then use [Sca17] to scale down the numbers to 7nm. We estimate the total area for 56 DECA PEs to be around 2.51 $\mathrm{mm}^2$. The Loaders, SQQs, Bitmask queues, Scale Factor queues, and TOut registers consume about 55% of DECA's area, the LUT array consumes 22%, while the rest consumes 23%. Given that the total die area of a 56-core SPR is around 1600 $\mathrm{mm}^2$ [Wik24], the DECA area overhead is less than 0.2%.

<span id="section-9"></span>

## 9 Evaluation

<span id="section-9-1"></span>

### 9.1 DECA for Compressed GeMMs

Figures [12](#figure-12) and [13](#figure-13) show, for different compression schemes, the speedups of the libxsmm software solution (*Software-only*) and of DECA over the baseline uncompressed BF16 scheme. We also add the *Optimal* speedup from the *roofline* model, which assumes that all VEC overheads are hidden. The compression schemes appear with increasing compression factor. We show results for N=1.

<span id="figure-12"></span>

![Figure 12. Compressed GeMM speedup for DDR and $N=1$.](../../papers/deca/figure-12.png)

**Figure 12.** Compressed GeMM speedup for DDR and $N=1$.

<span id="figure-13"></span>

![Figure 13. Compressed GeMM speedup for HBM and $N=1$.](../../papers/deca/figure-13.png)

**Figure 13.** Compressed GeMM speedup for HBM and $N=1$.

For the DDR setting (Figure [12](#figure-12)), DECA offers speedups over software only for high compression factors. This is expected since, according to the BORD in Figure [5b](#figure-05), only high compression factors are VEC-bound. The speedups reach 1.7$\times$. For the HBM setting (Figure [13](#figure-13)), DECA offers speedups for almost all the compression schemes. This is because, as shown by the BORD in Figure [5a](#figure-05), almost all schemes are VEC-bound. The speedups reach 4.0$\times$. In both DDR and HBM, the performance of DECA is near-optimal, revealing that the VEC overheads are successfully hidden. We repeated this analysis for batch sizes of up to N=16 and observed similar results.

DECA-augmented cores are much more capable at vector processing than conventional cores. Figure [14](#figure-14) compares the performance of both types of cores for the DDR setting with N=4, averaged across all the compression schemes. The figure compares different core counts: 8, 16, ...56. We see, e.g., that 16 DECA-augmented cores achieve higher performance than 56 conventional cores. The extra cores can either be freed-up for other workloads that do not consume much memory bandwidth, or power-gated to save energy.

<span id="figure-14"></span>

![Figure 14. TFLOPS across all compressions for DDR and $N=4$.](../../papers/deca/figure-14.png)

**Figure 14.** TFLOPS across all compressions for DDR and $N=4$.

To provide further insights into the performance of the system with software only and the one with DECA, Table [3](#table-03) displays the percent utilization of the memory bandwidth, of the TMUL, and of either the CPU's AVX units or DECA. Since performance is proportional to the utilization of the TMUL, the table shows that the system with DECA has much higher performance than software-only. Further, since the operations of the three components overlap, the one with the highest utilization ends up being the bottleneck. In the software-only system, for almost all of the densities, the bottleneck is the AVX vector units. This observation validates the Roof-Surface prediction. With DECA, the memory is much better utilized, leading to direct performance improvements. Note that, although sparser kernels take less time to execute, the utilization of DECA remains fairly constant. As explained in [Section 6](#section-6), DECA naturally achieves higher throughput for sparse schemes.

<span id="table-03"></span>

![Table 3. Component utilization for Q8, $N=1$, and HBM.](../../papers/deca/table-03.png){.paper-table-narrow}

**Table 3.** Component utilization for Q8, $N=1$, and HBM.

Figure [15](#figure-15) compares DECA with the alternative of scaling the CPU core's vector resources as a method to alleviate the decompression overhead. We compare a DECA-augmented core to a core with: (1) 4$\times$ more vector AVX units (*More AVX Units*) or (2) 4$\times$ wider AVX units (*Wider AVX Units*). We optimistically model the wider AVX2048 units by removing the dynamic instructions from 3 out of 4 iterations of the decompression loop. Since we do not modify the system cache line, each AVX2048 memory operation is executed as 4 cache-line sized operations. For the non-DECA systems, we do not scale the superscalar width of the core or the number of L1 ports since, as explained in [Section 7](#section-7), such changes are prohibitive. From the figure, we see that the performance of conventional vector scaling methods is far below DECA's performance.

<span id="figure-15"></span>

![Figure 15. DECA vs traditional vector scaling for HBM & $N=1$.](../../papers/deca/figure-15.png)

**Figure 15.** DECA vs traditional vector scaling for HBM & $N=1$.

<span id="section-9-2"></span>

### 9.2 Design Space Exploration with Roof-Surface

The DECA W and L parameters determine how fast DECA can decompress, but too large values may increase area without real benefit. To this end, we use the *Roof-Surface* to examine the performance for different {W,L} pairs. To dimension DECA, we pick the smallest {W,L} pair for which the predicted performance saturates (i.e., all the kernels are predicted *not* to be VEC-bound anymore). According to our model, this value pair is {W=32,L=8}. In Figure [16](#figure-16), we compare the BORDs for the HBM SPR system without DECA (a) and with DECA (b) with different {W,L} sizes: {W=8,L=4} (underprovisioned), {W=32,L=8} (best), and {W=64,L=64} (overprovisioned).

<span id="figure-16"></span>

![Figure 16. HBM BORDs with no DECA and with different-sized DECAs.](../../papers/deca/figure-16.png)

**Figure 16.** HBM BORDs with no DECA and with different-sized DECAs.

We observe that, in comparison with CPU, DECA has a smaller vector operations per second ($VOS$) parameter, since its VEC-bound region is larger. However, DECA decreases the number of vector operations needed per matrix operation (i.e., it increases $\mathrm{AI}_{\mathrm{XV}}$) as discussed in [Section 6](#section-6). The underprovisioned DECA with {W=8,L=4} is unable to push the kernels out of the VEC-bound region. The overprovisioned one with {W=64,L=64} pushes them out, but more than needed. We simulate the performance of these pairs to validate the model's accuracy. We find that the DECA-best system is 2$\times$ faster than the DECA-underprovisioned one. The DECA-overprovisioned system is less than 3% faster than the DECA-best one. At the same time, DECA-best is much cheaper than DECA-overprovisioned: it has 8$\times$ fewer LUTs and half the W. Overall, the Roof-Surface model accurately captures the dynamics of the matrix-vector-memory interaction and can guide microarchitectural decisions.

<span id="section-9-3"></span>

### 9.3 Analysis of DECA Integration and TEPLs

We now evaluate different decisions we made regarding DECA's integration with a core. We start with a base configuration where DECA reads compressed tiles from the LLC (bypassing L2), writes decompressed tiles in the L2 for the core to read, and is invoked using normal loads, stores, and fences. Then, we progressively enhance it to: (1) allow the accelerator to read compressed weights from the L2 and use the L2 prefetcher (*+Reads L2*), (2) use its own prefetcher instead of the L2 prefetcher (*+DECA prefetcher*), (3) write to the TOut Regs instead of to the L2 (*+TOut Regs*), and (4) use TEPL instructions instead of loads, stores, and fences (*+TEPL (DECA)*).

<span id="figure-17"></span>

![Figure 17. DECA integration features for HBM and $N=4$.](../../papers/deca/figure-17.png)

**Figure 17.** DECA integration features for HBM and $N=4$.

Figure [17](#figure-17) shows the speedups over the base design that the progressive application of these optimizations obtains for Q8 with different densities. We see that *+Reads L2* improves performance for all densities. The benefit comes from the L2 hardware prefetcher already available in the system, which fetches future tiles, hiding the memory and LLC access latencies. *+DECA prefetcher* further improves performance by using the DECA prefetcher rather than the default L2 one. *+TOut Regs* and *+TEPL (DECA)* reduce or hide the DECA-core communication latency and are necessary for out-of-order invocation. Specifically, *+TOut Regs* enables the core to directly fetch data from DECA, instead of taking the longer path through the L2. Further, *+TEPL (DECA)* overlaps communication with computation, effectively hiding the former. We see that the effectiveness of *+TOut Regs* and *+TEPL (DECA)* increases as the density decreases. This is because DECA takes less time to process a lower density tile, while the overhead of communication with the core remains constant. Thus, for lower densities, the communication cost gets more exposed. Note that TEPLs are very effective for low-density models: for 5% density, they double the performance.

<span id="section-9-4"></span>

### 9.4 DECA for LLM Inference

Lastly, we show the performance benefit of DECA for LLM next token generation (including the non-GeMM stages). Table [4](#table-04) shows the next token latencies of the Llama2-70B and OPT-66B models, respectively, on SPR with HBM, for 128 input tokens, 128 output tokens, batch sizes 1 and 16, and different compression schemes. We compare software decompression (*SW*) with our proposal (*DECA*). As explained, we simulate the uncompressed baseline BF16 model assuming a larger HBM size. We see that DECA reduces the next token time by 1.6$\times$-2.6$\times$ over *SW*. This translates into a 2.5$\times$-5.0x speedup over the uncompressed base model. We observed similar results for shorter/longer token sequences.

<span id="table-04"></span>

![Table 4. Llama2-70B/OPT-66B next-token latency (ms).](../../papers/deca/table-04.png){.paper-table-narrow}

**Table 4.** Llama2-70B/OPT-66B next-token latency (ms).

<span id="section-10"></span>

## 10 Other Related work

**Decoupled Accelerators.** A variety of stand-alone decoupled accelerators that target sparsity in ML and scientific applications have been proposed [Zha16c, Lu19, Ger24, Par17a, Che22b, Heg19, Sri20, Gon19a, Han16a, Adi23, Aan23]. Other decoupled accelerators rely on quantization  [Zhu24d, Ryu22, Jan24]. Recently, accelerators for attention are also becoming popular [Wan20b, Kac24, Lu21, Ham21, Ham20]. Decoupled accelerators come with large area-power budgets [Jeo23], and suffer from data movement overheads [Ger23]. For those reasons, CPU-integrated accelerators have been proposed [Ger23, Jeo23, Gon20, Gon22, Jeo21, Nas22]. DECA falls in this line of works. We discussed the shortcomings of other in/near-core accelerators in [Section 7](#section-7).

**Cooperative Vector-Matrix Processing.** A variety of architectures include heterogeneous matrix and vector units, whose interaction could be modeled with the *Roof-Surface* model. Examples include the Tandem processor [Gho24], the AWS Trainium [Bsh24, Fan24d], the TPU [Nor21], and GPUs with their Tensor Cores and SIMT Cores.

**Utility of a DECA-inspired decompression engine for GPUs.** Similar to the TMUL, the GPU Tensor Cores support only limited quantization formats and do not support unstructured sparsity. For this reason, GPU kernels such as Flash-LLM [Xia23b] adopt a similar approach to libxsmm: compressed data is decompressed through software and fed to the Tensor Cores. Although effective, Flash-LLM puts pressure on the L1/shared memory of the SMs, preventing full TensorCore/HBM utilization. We thus believe that DECA-inspired decompression engines could also be useful for GPUs. NVIDIA recently introduced the TMA accelerator [Luo24b] for supplying data from memory to Tensor Cores. Augmenting TMA with DECA-inspired decompression capabilities is an interesting future direction.

<span id="section-11"></span>

## 11 Conclusion

To improve LLM inference in advanced CPU platforms with in-core GeMM engines and HBM, this paper made three contributions: the *Roof-Surface* performance model, the *DECA* near-core ML-model decompression accelerator, and the TEPL ISA extension for out-of-order accelerator invocation. Our evaluation shows that DECA effectively accelerates compressed GeMMs and LLM inference.
