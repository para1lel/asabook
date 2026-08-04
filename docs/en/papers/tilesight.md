---
title: 'TileSight: Tile-Centric GPU Performance Model'
createTime: 2026/08/04 08:48:07
permalink: /en/papers/tilesight/
pageClass: tilesight-paper
---

> [Zhiwen Mo](https://hamerlate.github.io/), [Yu Cheng](https://chengyupku.github.io/), [Lei Wang](https://dblp.org/pid/181/2817-222), [Zhengju Tang](https://dblp.org/pid/371/5817), [Lei Xu](https://orcid.org/0000-0002-6226-3063), [Guoyu Li](https://dblp.org/pid/61/8379), [Yuqi Dong](https://dblp.org/pid/294/5118), [Lingxiao Ma](https://xysmlx.github.io/), [Yuqing Xia](https://dblp.org/pid/211/8365), [Jilong Xue](https://dblp.org/pid/06/10336), [Fan Yang](https://fanyangcs.github.io/), [Luo Mai](https://luomai.github.io/), [Zhi Yang](https://yangzhihome.github.io/), [Wayne Luk](https://profiles.imperial.ac.uk/w.luk), and [Hongxiang Fan](https://os-hxfan.github.io/). First submitted to arXiv on July 24, 2026. This reading edition transcribes version 1 from the [arXiv page, *TileSight: A First-Principles Tile-Centric Analytical GPU Performance Model from Cores to Clusters*](https://arxiv.org/abs/2607.22432), with the [original PDF](/paper/tilesight.pdf), [arXiv DOI](https://doi.org/10.48550/arXiv.2607.22432), and [TeX source](https://arxiv.org/src/2607.22432). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Recent GPU programming frameworks, such as Triton, TileLang, and CUDA Tile, have adopted the tile as a first-class language primitive, making tile-centric programming the prevailing approach for writing high-performance GPU kernels. Performance-analysis tooling for tile-based programs, however, has not followed suit: programmers still fall back on coarse roofline bounds, opaque ML-based predictors, or post-hoc profilers to reason about how their kernels actually run. This gap is increasingly painful for modern AI workloads, in which kernel fusion and distributed inference hinge on the interplay of tensor cores, CUDA cores, the cache hierarchy, memory pipelines, and inter-GPU networks. We bridge this gap with TileSight, a tile-centric performance-modeling tool that leverages the tile from a programming primitive to an analysis primitive. Within a single GPU core, TileSight models the overlap of compute and memory pipelines. Across cores on the same chip, TileSight models the cache hierarchy. Across GPUs, it models inter-node communication. All three layers share the tile abstraction: *1)* the intra-tile layer expresses each tile's work as a resource vector spanning the network, memory, and compute pipelines; *2)* the inter-tile layer schedules dependent and ordered tile actions to expose legal overlap and infers multi-level cache hit rates from a tile reuse distance; *3)* the cross-device layer maps remote tensor accesses to placements and routes them through an $\alpha$-$\beta$ stage cost. Evaluated on A100, H200, B200, and B6000, TileSight matches measured single-GPU kernel latency to within a pooled 12.35% mean absolute percentage error (MAPE), beating state-of-the-art baselines and transferring better across the four architectures. Its L2 cache-hit-rate predictions land within roughly one percentage point of the measured rate on every GPU. Pushed up to 32-GPU deployments, TileSight reaches 16.18% weighted MAPE (wMAPE) on fused distributed kernels and 13.52% wMAPE on end-to-end vLLM serving. When driven into the optimization loop, TileSight picks tile configurations competitive with strong vendor and expert baselines on the case studies we report. TileSight will be open-sourced upon publication.

## 1 Introduction

Large language model (LLM) scaling keeps pushing training and serving systems closer to hardware limits, making kernel efficiency central to both latency and cost. To extract this performance, developers increasingly fuse multiple LLM operations into large tile programs whose bottlenecks are determined by tiling, memory movement, pipeline overlap, and wavefront scheduling. Accurate and fast white-box performance modeling is therefore needed to expose the performance boundary and guide optimization.

To facilitate kernel optimization, the GPU programming community has converged on *tile-centric programming* as the common paradigm: Triton [Til19] pioneered tile-level loads, stores and dot products, which has become the de facto standard for custom kernels in PyTorch. TileLang [Til25g] further decouples dataflow from scheduling at the tile level. NVIDIA's CUDA Tile [Nvi26] (CUDA 13.1, 2025), described as the most significant CUDA advancement in roughly 20 years [Fut26], officially adopts tiles as the programming primitive, while CuteDSL [NviCut] exposes CUTLASS's tile abstractions as a Python domain-specific language (DSL). As a result, tiles have emerged as a central abstraction in modern GPU programming. However, **the performance analysis has not kept pace with this tile-centric abstraction**: Triton relies on black-box autotuning over thousands of configurations [Til19], roofline models [Wil09] cannot distinguish an L2 cache miss from a shared memory bank conflict, and ML-based predictors [Lee25, Geo21] require per-architecture training and are opaque. Profilers such as Nsight Compute (NCU) and profiling-based tools [Gua25, Hua25] are post-hoc: they can perturb execution through instrumentation and clock changes, and their counter reports do not explain which tile, reuse pattern, or pipeline stage caused the observed bottleneck. Table 1 summarizes this abstraction mismatch. As tile-centric programming becomes increasingly adopted, an accurate and efficient *tile-centric performance model* is urgently needed to predict how tile-configuration changes affect performance without running the kernel.

The need for such a model becomes even more pressing when we consider what happens *inside* a tiled kernel's main loop. Even for GEMM, performance depends on software-pipeline depth, resident tiles per streaming multiprocessor (SM), and load-compute overlap rather than only on FLOP and byte counts. The issue becomes sharper in fused kernels: as illustrated in Figure 1, FlashAttention-3 (FA-3) on H100 involves over ten distinct operations, including two GEMMs on Tensor Cores, reductions and softmax on CUDA cores, and special functions on special function units (SFUs), with fine-grained data dependencies. These operations occupy different hardware resources and can potentially overlap, but the degree of overlap depends critically on their scheduling order and pipeline depth. Existing tools, including roofline, profilers, and autotuners, are largely blind to this intra-tile scheduling structure. Furthermore, these complex kernels are increasingly needed in *distributed* settings. For instance, Tensor parallelism (TP), expert parallelism (EP), and sequence parallelism (SP) partition workloads across multiple GPUs [Sve25], introducing collective communication that needs to overlap with computation. The performance of a distributed kernel depends on how the global tile grid is partitioned, what communication primitives are used, and how compute and communication pipelines interleave. These decisions are currently made by intuition or expensive trial-and-error.

To address these challenges, we observe that tiles provide *a natural first-class abstraction for performance modeling* of GPU systems. This stems from three properties: **(1) Deterministic**: given a tile configuration (shape, pipeline depth, memory layout), the resource usage of each tile is fully determined, enabling analytical modeling without simulation. **(2) Composable**: tile information composes hierarchically. Each tile carries its own per-pipeline resource decomposition (intra-tile), tiles are related through dependencies, concurrent issue, and execution order (inter-tile), and tile grids extend across devices through placement (cross-device). Each level can be modeled independently then composed. **(3) Portable**: the tile abstraction is adopted across various GPU architectures (in this paper we exercise NVIDIA A100, H100, H200, B200, RTX PRO 6000 Blackwell (B6000), and AMD MI210), since all modern GPUs execute tile-shaped workloads through similar hierarchical memory and compute structures.

Building on these insights, we present TileSight, a *unified tile-centric analytical execution engine*. Unlike roofline models that attribute performance to a single bottleneck resource, TileSight analytically simulates how a tile execution plan unfolds across the hardware, capturing the prologue, steady-state overlap, and epilogue structure that determines real kernel performance. This simulation composes three hierarchical levels with unified tile-based abstractions:

- **Intra-tile**: each tile is characterized by an operation, an src/dst placement descriptor, and a footprint, which together produce a per-tile *resource vector* that decomposes work into times on independently schedulable hardware pipelines spanning network, memory, and compute. The same placement descriptor unifies fusion (intermediates kept in registers or shared memory (SMEM)) and cross-device movement.
- **Inter-tile**: tiles are related through producer-consumer dependencies, concurrent issue, and execution order. These together drive a topological-order search over the tile-action directed acyclic graph (DAG) that picks the best legal pipeline overlap inside a fused kernel body, and a multi-level *tile reuse distance* analysis with stochastic distance-based cache modeling (SDCM) that derives implicit cache hit rates from grid traversal.
- **Cross-device**: cross-device execution is a placement case of the same intra-tile abstraction — a tile whose source or destination crosses devices simply gains a `Net` entry computed from the routed $\alpha$-$\beta$ cost of the underlying remote tensor access, so the same envelope still applies.

Crucially, these three levels are jointly designed with shared core abstractions: `HardwareUsage` as a per-pipeline time decomposition, the *tile action* as the composable scheduling unit, and the `TileGrid` as the workload descriptor. In summary, we make the following contributions:

**(1) A unified tile-centric analytical execution engine** that simulates how tile execution plans unfold across hardware pipelines, including per-tile resource decomposition (intra-tile), dependency-driven DAG ordering with tile reuse-distance cache modeling (inter-tile), and placement-based cross-device tile accesses, all under one framework with shared abstractions (Section 3).

**(2) Tile-pipeline overlap analysis** that models both regular software pipelined loops (e.g., GEMM load-compute overlap) and complex fused kernels (e.g., FlashAttention and multi-head latent attention (MLA) decode) as repeated tile pipelines over dependency-constrained tile-action DAGs. By combining pipeline depth, resident tile interleaving, and legal tile-action ordering, TileSight predicts the prologue, steady-state, and epilogue costs that simple roofline models miss (Section 3.4).

**(3) Tile reuse-distance cache modeling** that makes cache behavior a natural consequence of the tile execution plan rather than a separate trace-simulation problem. By reasoning about reuse at the same granularity as GPU schedules, TileSight enables fast, schedule-sensitive multi-level cache modeling inside an analytical performance model, while preserving accuracy through lightweight approximation and sampling techniques (Section 3.5).

**(4) Composable distributed extension via tile placement** where cross-device execution is a placement case of the same tile abstraction: remote tensor accesses are inferred from producer-consumer placement and decomposed into ordered stages of logical exchanges, whose routed $\alpha$-$\beta$ cost populates the network entry of the per-tile resource vector so cross-device movement composes with local compute through the same envelope (Section 3.6).

## 2 Background & Motivation

### 2.1 GPU Performance Modeling

![FlashAttention-3 execution on H100](../../papers/tilesight/figure-01.png)

**Figure 1.** FlashAttention-3 on H100: (a) the 10+ heterogeneous operations spanning Tensor Cores, CUDA cores, and SFUs; (b) their data-dependency DAG; (c) how scheduling order determines compute-memory pipeline overlap.

![L2 bandwidth versus working-set size](../../papers/tilesight/figure-02.png)

**Figure 2.** L2 bandwidth vs. working-set size on B200 and B6000, revealing the multi-level cache hierarchy. B200 (dual-die) exposes a level-1.5 (L1.5)/LRC tier at ${\sim}22.5$ TB/s and a smeared L2 cliff at ${\sim}83$ MB; B6000 (single-die) shows a sharp cliff at ${\sim}130$ MB. TileSight uses these sweeps to calibrate effective cache capacity per GPU.

Existing GPU performance tools can be grouped into three categories. **Learned and hybrid predictors** [Lee25, Zha26p] fit either end-to-end runtime or analytical-model residuals from per-architecture traces, arguing that pure analytical models cannot capture modern GPU interaction complexity; both nevertheless require retraining and provide limited interpretability. **Analytical models** [Wil09, Par19, Zhe23] are portable and explainable, but typically collapse GPU execution into aggregate compute and bandwidth terms. **Profiling and simulation tools** [Gua25, Hua25, Agr24, Wan25s] expose measured behavior after execution, but do not forecast how a tile shape, pipeline depth, or swizzle change will perform before rerunning the kernel. As motivated in Section 1, the shared limitation is an abstraction mismatch: these tools do not model performance at the same tile granularity used by modern GPU programs. In particular, where hybrid predictors delegate this complexity to learned components, TileSight shows that a first-principles tile-centric simulation can capture it while remaining fully white-box.

### 2.2 Modeling Gap for Tile-Centric Programs

The missing abstraction appears at three levels. **Intra-tile**: each tile uses heterogeneous pipelines spanning compute, memory, and network, so a single bottleneck scalar misses the per-pipeline structure that determines overlap (Figure 1). **Inter-tile**: tile dependencies determine the legal action orderings inside a fused body, and tile execution order across the grid determines cache reuse; a single flat bandwidth number is insufficient on modern GPUs (Figure 2). **Cross-device**: partitioned tile grids exchange data through communication pipelines that must overlap with compute rather than be added as standalone times. Table 1 summarizes how existing tools miss one or more of these levels.

<div class="paper-wide-table">

| Feature | Roofline [Wil09] | NeuSight [Lee25] | PipeWeave [Zha26p] | GenZ [Bam24] | Vidur [Agr24] | SimAI [Wan25s] | TileSight |
| --- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| No kernel profiling/training¹ | ✓ | ✗ | ✗ | ✓ | ✗ | ✗ | ✓ |
| Pipeline-aware² | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✓ |
| Cache-aware³ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✓ |
| Explicit fused program⁴ | ✗ | ✗ | ○ | ✗ | ✗ | ✗ | ✓ |
| Distributed⁵ | ✗ | ○ | ○ | ○ | ✓ | ✓ | ✓ |
| Compute-comm. overlap⁶ | ✗ | ✗ | ✗ | ✗ | ✗ | ○ | ✓ |
| Interpretable⁷ | ✓ | ✗ | ○ | ✓ | ✗ | ✗ | ✓ |

</div>

**Table 1.** Comparison with prior performance modeling tools. ✓ Full support. ○ Partial. ✗ Not supported. ¹No kernel profiling/training: no kernel execution traces or ML training are required; TileSight uses only one-time per-architecture microbenchmarks (bandwidth/throughput/latency sweeps, $\sim$minutes). ²Pipeline-aware: intra-tile DAG scheduling and compute-memory pipeline overlap. ³Cache-aware: predicts L2/L1.5 hit rates and schedule-dependent tile locality effects. ⁴Explicit fused program: user-described arbitrary multi-op DAG kernels (e.g., FA-3, MLA), not limited to a fixed set of supported patterns. ⁵Distributed: multi-GPU collective communication modeling. ⁶Compute-communication overlap: fused compute-communication kernels (e.g., AllGather+GEMM). SimAI accepts user-specified overlap ratios but does not derive them analytically. ⁷Interpretable: white-box model supporting bottleneck diagnosis.

## 3 Hierarchical Tile-Pipeline Model

TileSight treats the *tile* as the first-class modeling unit and adopts a prologue-steady-epilogue pipeline envelope to recursively apply at every level of the program. A tile carries *intra-tile* information (operation, src/dst placement, footprint, and the resources it occupies on each independently schedulable hardware pipeline) and participates in *inter-tile* relationships (producer-consumer dependencies, concurrent issue, and execution order across loops, the tile grid, and waves). A tiled workload is therefore a *tile execution plan*: a graph of tiles annotated with these two kinds of information. Distributed execution shares the same tile-based abstraction: a tile whose source or destination crosses devices simply gains a `Net` entry in its resource vector, and the same envelope still applies.

### 3.1 From Workload to Tile Execution Plan

The input to TileSight is a high-level workload, such as a tiled GEMM, a fused attention kernel, an all-gather followed by a GEMM, or Mixture-of-Experts (MoE) routing across GPUs; the workload fixes tensors and their placements but leaves the schedule unspecified. TileSight lifts it to a tile execution plan that exposes the schedule-relevant choices needed for performance modeling: tile shape, loop and reduction order, block swizzle, software-pipeline depth, resident blocks per SM, distributed partitioning, and collective implementation. Tile-centric DSLs such as Triton and TileLang expose most of this information directly; for hand-written kernels, the same fields are supplied manually from the kernel schedule.

![TileSight design overview](../../papers/tilesight/figure-03.png)

**Figure 3.** **TileSight design overview** on all-gather-GEMM (AG-GEMM). **(a)** A workload is described by an operator and tensor placement only ($X$ column-sharded across $N$ GPUs). **(b)** TileSight lifts it to a tile schedule whose DAG spans memory levels $L_0$-$L_4$. **(c)** A single hardware abstraction exposes registers, SMEM, L2, HBM, and the inter-GPU fabric as a 5-level hierarchy. **(d)** Intra-tile resource vectors and inter-tile DAG/concurrency analysis feed a recursive prologue-steady-epilogue envelope. **(e)** The engine renders the envelope as a timeline: software-pipelined loads overlap with compute, and the AllGather is *inferred from placement* on the `Net` lane. **(f)** A per-tile performance report with latency, utilization, cache hit, and overlap rate.

| Field | Side | Role in the model |
| --- | --- | --- |
| Tensor accesses | Intra | Per-tile footprint and a placement descriptor recording where a tensor tile is produced and where it resides: register, architecture-specific tensor memory (TMEM), SMEM, local cache/DDR, or shard/replica on a GPU group. Reuse dimensions feed inter-tile cache modeling. |
| Operation type | Intra | The action a tile performs: load, store, tensor-core or CUDA-core matmul, reduction, exponential, rescaling, remote transfer, or fused composite. |
| Resource vector | Intra | Per-tile time on independently schedulable resources (tensor cores (TC), CUDA cores, SFU, TMEM, SMEM, L1.5, L2, DDR, Net); derived from the operation, footprint, placement, and calibrated hardware rates. |
| Tile grid | Inter | Spatial tile shape, launch order, swizzle, loop/reduction depth, and distributed partition. Determines tile execution order, waves, and local work per device. |
| Producer-consumer DAG | Inter | Edges among tiles based on tensor production and consumption; together with placement, fixes the legal orderings within an iteration. |
| Concurrency & depth | Inter | Software-pipeline stages, resident blocks per SM, and which tiles may issue together. Sets the effective pipeline depth. |

**Table 2.** The tile execution plan groups its fields by what they describe: a single tile in isolation (intra) or relationships among tiles (inter).

The plan deliberately avoids thread-level details. It keeps only the choices that tile programmers and distributed runtimes actually change, and those choices change which tiles enter the pipeline, what resources they occupy, and how they depend on or run concurrently with each other. Cache traffic, wave effects, communication stages, and pipeline overlap are then derived rather than separately added.

### 3.2 Intra-Tile: A Tile and Its Resource Vector

A tile is characterized by its *operation* (load, store, tensor-core or CUDA-core matmul, reduction, exponential, rescaling, remote transfer, or fused composite), its *footprint* (per-tile bytes and FLOPs), and its *src*/*dst* *placement descriptors* that record where its inputs are produced and where its output resides — a register, architecture-specific tensor memory (TMEM), the shared-memory scratchpad, the L1.5 or L2 cache, DDR on the local device, or a shard or replica on a GPU group. Placement is the central abstraction that lets the same intra-tile representation describe both fusion and cross-device movement: marking an intermediate output as register-, TMEM-, or SMEM-scope removes a global-memory store (fusion), while marking a load source as a remote shard turns the load into a cross-device transfer (distribution).

For each tile, TileSight converts these properties into a vector of times on independently schedulable hardware resources:

$$
\mathbf{u}(o)=
\langle t_{\mathrm{TC}}, t_{\mathrm{CUDA}}, t_{\mathrm{SFU}}, t_{\mathrm{TMEM}},
t_{\mathrm{SMEM}}, t_{\mathrm{L1.5}}, t_{\mathrm{L2}}, t_{\mathrm{DDR}}, t_{\mathrm{Net}}\rangle .
\tag{1}
$$

computed from the tile's operation, footprint, src/dst placement, and one-shot microbenchmark-calibrated rates. A pure tensor-core matmul tile populates only the TC entry; a Blackwell attention tile also charges explicit TMEM traffic for softmax and correction loads/stores; a load tile from DDR populates DDR (and L1.5/L2 if the access hits cache); a remote-load tile populates Net. This vector is more expressive than a roofline scalar because tiles on different pipelines may overlap, while tiles contending for the same pipeline serialize, and remote movement composes with local compute through the same machinery. Two entries of $\mathbf{u}(o)$ are not fixed by the tile in isolation: the L1.5/L2/DDR split for a memory tile depends on whether its access hits cache, derived from tile reuse distance in Section 3.5; the `Net` entry for a remote tile depends on the routed cost of the underlying communication stage, derived in Section 3.6. Algorithm 1 sketches how all components plug into the master loop; subsequent subsections detail each block.

**Algorithm 1: Hierarchical Tile-Pipeline Evaluation.**

- **Input:** tile execution plan $P$, hardware specification $H$, optional distributed mapping $\Pi$.
- **Output:** predicted latency $T$ and per-resource utilization.
- $G \leftarrow$ tile grid, launch order, and swizzle from $P$.
- $A \leftarrow$ tensor accesses, reuse dimensions, and placement descriptors from $P$.
- $D \leftarrow$ tile-action DAG from $P$.
- $S \leftarrow$ software-pipeline parameters from $P$.
- **If** $\Pi$ is not empty:
  - $G,A,D \leftarrow \mathrm{PartitionTilePlan}(G,A,D,\Pi)$; single device is the local-only case.
  - $\mathcal{O}_{\mathrm{net}} \leftarrow \mathrm{InferRemoteTensorAccesses}(G,A,D,\Pi)$.
  - $\mathcal{N} \leftarrow$ network topology and calibrated $\alpha,\beta$ parameters from $H$.
  - **For each** remote tensor access sequence $c \in \mathcal{O}_{\mathrm{net}}$:
    - $\mathcal{K}_c \leftarrow \mathrm{DecomposeIntoStages}(c)$; e.g., ring steps or tree levels.
    - **For each** stage $k \in \mathcal{K}_c$:
      - $\mathcal{E}_k \leftarrow \mathrm{LogicalExchanges}(k)$; tuples $(\mathrm{src},\mathrm{dst},\mathrm{bytes})$.
      - $\mathcal{R}_k \leftarrow \mathrm{Route}(\mathcal{E}_k,\mathcal{N})$.
      - $T_k, U_k \leftarrow \mathrm{AlphaBetaStageTime}(\mathcal{R}_k,\mathcal{N})$.
    - Annotate the corresponding transfer tile in $D$ with $\sum_k T_k$ on `Net`.
- $C \leftarrow \mathrm{CacheTraffic}(G,A,H)$.
- Annotate memory tiles in $D$ with L1.5/L2/DDR entries from $C$.
- $p \leftarrow \mathrm{ResidentTilesPerSM}(P,H)$.
- $d \leftarrow S.\mathrm{stages}\times p - 1$.
- $\mathcal{E}_{\mathrm{tile}} \leftarrow \mathrm{PipelineEnvelope}(D,d,H,\mathrm{active\ SMs})$.
- $T, U \leftarrow \mathrm{WaveAggregate}(G,\mathcal{E}_{\mathrm{tile}},H)$.
- **Return** $T, U$.

### 3.3 Inter-Tile: Dependency, Concurrency and Order

Tiles connect through three kinds of inter-tile information. *1)* *Producer-consumer dependencies* fix the legal orderings within an iteration: in FlashAttention, $Q$/$K$ loads precede gemm1 ($Q\!@\!K$), gemm1 precedes softmax, and softmax precedes gemm2 ($P\!@\!V$). Together with placement and dependencies, it determines which intermediates stay in registers/TMEM/SMEM and which spill to global memory. *2)* *Concurrent issue* lets non-dependent tiles run together when their resource vectors do not contend, e.g., concurrently loading the next $K$-block of attention while computing on the current one, or issuing the A and B loads of a GEMM along the same $K$ slice. The same set of tiles can be ordered in multiple legal ways that yield different overlap on shared pipelines. *3)* *Tile execution order* across loop iterations and the tile grid determines which loads find their data already resident in cache: row-panel traversal preserves B-tile reuse for adjacent $M$-rows, block swizzle reorders the sequence, and persistent-block schedules pin tiles to SMs. These three pieces are exactly the input the pipeline envelope needs.

### 3.4 Pipeline Envelope: Prologue-Steady-Epilogue

Given a set of tiles with resource vectors and inter-tile relationships, TileSight evaluates execution as a pipeline. For a repeated unit with $N$ logical iterations and effective depth $d$:

$$
T =
T_{\mathrm{pro}} +
\max(N-d,0)\,T_{\mathrm{steady}} +
T_{\mathrm{epi}},
\tag{2}
$$

where $T_{\mathrm{pro}}$ is the fill cost, $T_{\mathrm{steady}}$ is the overlapped cost per repeated unit, and $T_{\mathrm{epi}}$ is the drain cost. The same envelope applies recursively at every level of the tile execution plan: the steady-state body of an outer envelope (over tile-block waves) can itself be a pipeline (over a $K$-loop), whose steady body can in turn be a pipeline over the inner action sequence. The effective depth combines explicit software-pipeline stages with resident tile interleaving:

$$
d = \mathrm{stages} \times \mathrm{resident\_tiles\_per\_SM} - 1 .
\tag{3}
$$

A two-block-per-SM schedule is therefore not a special case: it deepens the pipeline because an SM can issue work from one resident tile-block while another waits on memory.

**Steady-state overlap.** The steady-state cost of a tile sequence depends on which legal ordering is chosen, since tiles using the same hardware dimension in Eq. 1 accumulate on that dimension while independent dimensions overlap:

$$
T_{\mathrm{steady}}(\sigma)
=
\max_{r}
\sum_{o \in \sigma} u_r(o),
\tag{4}
$$

subject to all data-dependency edges in the DAG. The selected steady state is the best legal ordering:

$$
T_{\mathrm{steady}} =
\min_{\sigma \in \mathrm{Topo}(D)} T_{\mathrm{steady}}(\sigma).
\tag{5}
$$

This is a small search in practice because real fused-kernel DAGs are heavily constrained. For MLA decode, 11 tile actions reduce from $11!$ unconstrained permutations to 132 legal topological orders. The search is not an autotuning run: it is an analytical scheduling step over the tile plan, so it remains cheap enough to run inside a cost model.

**Boundary costs.** The prologue and epilogue are computed from the same resource vectors but with reduced overlap. For a load-compute pipeline, the prologue consists primarily of memory tiles that fill the pipeline, while the epilogue consists of remaining compute and final stores. Fused tile bodies add reductions or normalizations to one or both boundaries. This separation matters because two schedules with the same steady-state bottleneck can have different end-to-end time when the loop count is short or when only a few waves are launched.

**Resident tiles and waves.** Occupancy changes overlap structure, not only utilization. If $p$ tile-blocks reside on one SM, the model treats them as interleaved instances of the same tile pipeline; the resident count is bounded by shared memory, registers, warp limits, and architecture-specific maximum blocks per SM. The same wave decomposition handles tail effects: a tail wave may use only a subset of SMs, and those active SMs receive a larger share of shared L2/DDR bandwidth, so the envelope is recomputed for the tail using its active-SM count. Algorithm 2 expands this evaluation, recursively traversing the tile loop structure and enumerating dependency-valid orderings.

**Algorithm 2: Recursive Pipeline-Envelope Evaluation.**

- **Function** $\mathrm{OverlapAnalysis}(P,H)$:
  - $p \gets \mathrm{ResidentTilesPerSM}(P,H)$.
  - $(n_{\mathrm{full}}, n_{\mathrm{tail}}) \gets \mathrm{WaveDecompose}(P.\mathrm{grid}, H.\mathrm{SMs}, p)$.
  - $(T^{\mathrm{full}}, U^{\mathrm{full}}) \gets \mathrm{AnalyzeLoop}(P.\mathrm{root}, p, H.\mathrm{SMs})$.
  - **If** $n_{\mathrm{tail}} > 0$:
    - $(T^{\mathrm{tail}}, U^{\mathrm{tail}}) \gets \mathrm{AnalyzeLoop}(P.\mathrm{root}, p, n_{\mathrm{tail}})$.
  - **Else:**
    - $T^{\mathrm{tail}} \gets 0,\ U^{\mathrm{tail}} \gets \emptyset$.
  - **Return** $n_{\mathrm{full}}T^{\mathrm{full}} + T^{\mathrm{tail}},\ \mathrm{MergeMetrics}(U^{\mathrm{full}},U^{\mathrm{tail}})$.
- **Function** $\mathrm{AnalyzeLoop}(\mathrm{node}, \mathrm{stage}, \mathrm{active\_SMs})$:
  - $\mathrm{groups} \gets \mathrm{GetSubNodes}(\mathrm{node})$.
  - **If** $\mathrm{node}$ is an inner loop:
    - $s \gets \mathrm{GetPipelineStage}(\mathrm{node})$.
    - $d \gets s \times \mathrm{stage} - 1$; software stages $\times$ resident tiles.
    - **Return** $\mathrm{ModelOverlap}(\mathrm{groups},d,\mathrm{active\_SMs})$.
  - $\mathrm{metrics} \gets [\,]$.
  - **For each** $g \in \mathrm{groups}$:
    - **If** $g$ is a loop:
      - $\mathrm{metrics.append}($ $\mathrm{AnalyzeLoop}(g,\mathrm{stage},\mathrm{active\_SMs}))$.
    - **Else:**
      - $\mathrm{metrics.append}($ $\mathrm{ModelOverlap}([g],\mathrm{stage}-1,\mathrm{active\_SMs}))$.
  - **Return** $\mathrm{MergeMetrics}(\mathrm{metrics})$.
- **Function** $\mathrm{ModelOverlap}(\mathrm{groups},d,\mathrm{active\_SMs})$:
  - $N \gets$ repeated-iteration count represented by $\mathrm{groups}$.
  - $\mathrm{best} \gets \infty$.
  - **For each** $\sigma \in \mathrm{Topo}(\mathrm{groups})$:
    - $\mathbf{u}_{\sigma} \gets$ resource-vector accumulation under order $\sigma$ and $\mathrm{active\_SMs}$.
    - $T_{\mathrm{pro}},T_{\mathrm{steady}},T_{\mathrm{epi}} \gets$ boundary and steady costs from $\mathbf{u}_{\sigma}$.
    - $T \gets T_{\mathrm{pro}}+\max(N-d,0)T_{\mathrm{steady}}+T_{\mathrm{epi}}$.
    - **If** $T < \mathrm{best}$: $\mathrm{best} \gets T$.
  - **Return** $\mathrm{best}$ and the corresponding utilization.

### 3.5 Cache Traffic via Tile Reuse Distance

For a memory tile, the L1.5/L2/DDR split is not a property of the tile in isolation: the same load-tile coordinate can hit cache or fall through to DDR depending on swizzle, wave occupancy, and which neighboring tiles share tensor data. Preserving B-tile reuse across GEMM $M$-axis tiles can cut DDR traffic by ${\sim}4\times$, and block swizzling shifts L2 hit rate from 35% to 72% in our motivating case; modern GPUs further add intermediate L1.5/LRC tiers (H200, B200), making a single flat bandwidth term insufficient. Reuse-distance analysis is well established for cache modeling [Lam91], [Con98], [Nug14], [Ara19], [Ara20], [Niu12], but conventional formulations operate on cache-line traces and are too low-level to place inside analytical schedule search. TileSight instead lifts reuse distance to the tile execution plan, with the symbolic tile order as the analyzed sequence and tile-sized tensor blocks as the reuse universe—to our knowledge, the first analytical GPU performance model to make schedule-sensitive, multi-level cache modeling practical through a tile-granular reuse-distance abstraction.

#### 3.5.1 Tensor Access and Tile Reuse Distance

TileSight introduces a *tensor access* for each tensor associated with the tile grid: per-tile footprint, placement descriptor, repeated-access count, and the grid dimensions along which the same data block is reused. The reuse dimensions `reuse_dims` make one rule cover diverse operators: a tensor's reuse key is the tile coordinate projected onto the non-reuse dimensions. For a GEMM grid $(M_t,N_t)$, A tiles are reused across $N_t$ and B tiles across $M_t$. For MLA decode, key-value (KV)-cache tiles are reused across attention heads of the same batch element. For convolution, weights and activations have different reuse dimensions over batch, output-channel, and spatial axes. This avoids operator-specific cache formulas while preserving the schedule information that determines reuse.

The *tile reuse distance* $D_T$ is the number of distinct tile-sized data blocks accessed between two consecutive accesses to the same tensor block. Traditional reuse distance asks how many cache lines or memory transactions intervene between two accesses, tile reuse distance asks the same question at the unit GPU kernel schedules expose. Modeling an 8 KB tile instead of 128-byte cache lines reduces tracked entries by $64\times$, matches the granularity tile-centric schedules expose, makes block swizzles and traversal orders directly visible to the cache model, and avoids trace-level cache simulation.

![Figure 4: Tile vs. cache-line reuse distance. Left: traditional cache-line reuse distance tracks tens of thousands of line entries and evaluates exact SDCM at line granularity. Right: TileSight lifts reuse distance to tile-sized blocks, applies a Gaussian SDCM approximation, and samples along reduction axes, preserving schedule sensitivity while making cache modeling lightweight.](../../papers/tilesight/figure-04.png)

For a tensor with `reuse_dims`, TileSight computes a reuse key from the tile's non-reuse coordinates:

$$
\mathrm{key}(\mathbf{x}, R)=\mathrm{Linearize}\bigl(x_d\mid d\notin R\bigr),
\tag{6}
$$

where $\mathbf{x}$ is the tile coordinate and $R$ is the set of reuse dimensions. For GEMM's A matrix with $R=\{N_t\}$, all tiles in the same M-row share the same A key. For B, all tiles in the same N-column share the same B key. The concrete tile execution order, including swizzles and row-panel traversal, determines the sequence in which these keys appear and therefore their reuse distances.

#### 3.5.2 Hit Probability and Fast Evaluation

Given reuse distance $D_T$, associativity $A$, and cache capacity $B_T$ measured in tile units, the stochastic distance cache model estimates the hit probability of a least-recently-used (LRU)-like cache. The exact SDCM hit probability can be expressed with a binomial form:

$$
P(h \mid D_T) =
\sum_{a=0}^{A-1}
\binom{D_T}{a}
\left(\frac{A}{B_T}\right)^a
\left(\frac{B_T-A}{B_T}\right)^{D_T-a},
\tag{7}
$$

where $A$ is cache associativity and $B_T$ is cache capacity measured in tiles. While accurate, this binomial form is expensive to compute for every tensor key in a large tile grid.

TileSight therefore adopts a Gaussian approximation for efficient evaluation:

$$
P(h \mid D_T)_{\mathrm{approx}}
=
1 - Q\!\left(
\frac{|A-1-\mu|}{\sqrt{\sigma^2}}
\right),
\tag{8}
$$

where

$$
\mu = D_T \cdot \frac{A}{B_T},
\qquad
\sigma^2 =
D_T \cdot \frac{A}{B_T}
\cdot
\left(1-\frac{A}{B_T}\right).
\tag{9}
$$

$Q(x)$ denotes the complementary cumulative distribution function (CDF) of the standard normal distribution. To further reduce overhead, we apply the Zelen-Severo approximation [Abr65] for the CDF $\Phi(x)$:

$$
\Phi(x)
\approx
1 -
\left(a_1t-a_2t^2+a_3t^3\right)
\frac{e^{-x^2/2}}{\sqrt{2\pi}},
\tag{10}
$$

where $t=(1+0.33267x)^{-1}$ and $a_1,a_2,a_3$ are constants.

**Sampling along reduction axes.** Tile execution plans expose reduction axes (e.g., the $K$ axis in GEMM). TileSight samples reuse events at this granularity rather than replaying every inner-loop access (a GEMM with $K{=}8192$, $\mathrm{tile}_K{=}32$ reduces checks by $256\times$ with negligible accuracy loss). Together with tile-level reuse distance and the Gaussian approximation, this reduces cache-model evaluation by roughly five orders of magnitude, enabling cache modeling inside the analytical loop rather than as offline trace analysis.

#### 3.5.3 Two-Level Cascade, Swizzle, and Waves

On GPUs with an intermediate L1.5/LRC tier, TileSight applies SDCM as a cascade—L1.5 within each physical SM group, L2 globally, and DDR carrying the residual miss traffic; without this design, L1.5 hit probability is zero and the model collapses to a single L2 evaluation. A block swizzle, row-panel, Z-order, or persistent-block schedule is just a concrete sequence of tile coordinates fed to the reuse-distance simulation. Within a wave, TileSight perturbs $D_T$ for hardware nondeterminism, sequential tensor loads, and cross-tensor cache aging, all derived from the tile execution plan and hardware grouping with no kernel-specific profiling. Tail waves use a subset of SMs and therefore receive a larger share of shared bandwidth, so the envelope is recomputed for the tail. The resulting L1.5/L2/DDR byte counts populate the corresponding entries of Eq. 1, so cache behavior changes the pipeline envelope itself, not only the final latency.

### 3.6 Cross-Device Tiles

Cross-device execution is a placement extension of the same intra-tile abstraction: a tile's source or destination can point to a shard or replica on another GPU, and its resource vector picks up a non-zero `Net` entry. A tensor-, expert-, sequence- or data-parallel mapping partitions both the tile grid and its tensor tiles, producing placement descriptors over GPU groups. After partitioning, a local tile wave may need a tensor tile produced by another device, a replicated activation, or a partial result that must be reduced before later tiles can consume it. TileSight treats these as *remote tensor accesses*: the required collectives or point-to-point transfers are inferred directly from producer-consumer placement, and each becomes a tile with source/destination devices, byte volume, and `Net` resource usage.

**Logical exchanges and topology.** For each inferred remote tensor access, TileSight decomposes the required tensor-tile movement into ordered stages. A stage is represented by logical source-destination exchanges $(s,d,b)$, where $s$ is the device that owns or produces the tensor tile, $d$ denotes the device whose tile wave consumes it, and $b$ means the tile or shard byte volume derived from the tensor access. Collective algorithms simply provide different stage decompositions: a ring all-reduce uses reduce-scatter and all-gather steps, tree algorithms use reduction and broadcast levels, and irregular routing remains point-to-point. This representation is tile-level rather than packet-level. It preserves the tensor-placement information needed to reason about communication volume, while leaving the hardware topology to determine which physical network-on-chip (NoC) or interconnect links carry each exchange.

**Per-stage routed cost.** After routing the exchanges in a stage, TileSight estimates the stage time with an $\alpha$-$\beta$ communication model [Tha05] that matches the decomposition into hop latency and bottleneck-link serialization:

$$
T_k
=
\underbrace{
\max_{(s,d,b)\in\mathcal{E}_k}
\sum_{l\in\mathcal{P}_{sd}} \alpha_l
}_{\mathrm{routed\ hop\ latency}}
+
\underbrace{
\max_{l\in\mathcal{L}} \beta_l B_{l,k}
}_{\mathrm{bottleneck\ link\ serialization}},
\tag{11}
$$

where $\mathcal{E}_k$ is the set of logical exchanges in stage $k$, $\mathcal{P}_{sd}$ the physical route for $(s,d,b)$, $B_{l,k}$ the bytes routed through link $l$, and $\alpha_l,\beta_l$ are the calibrated startup latency and inverse bandwidth of link $l$. The cost of an inferred communication sequence is the ordered sum over its stages, $T_c=\sum_{k\in\mathcal{K}_c}T_k$. For algorithms with repeated identical stages such as ring collectives, TileSight evaluates one stage and multiplies by the stage count. The result enters the `Net` dimension of Eq. 1, so cross-device movement is represented as an intra-tile resource requirement and overlaps with local compute through the same steady-state machinery as everything else.

### 3.7 Putting It Together

With the pieces in place, Algorithm 1 regains its full meaning: cache analysis (Section 3.5) populates the L1.5/L2/DDR entries of $\mathbf{u}(o)$ from inter-tile execution order; remote tensor accesses (Section 3.6) populate the `Net` entry from routed $\alpha$-$\beta$ stage cost; the envelope (Section 3.4) then consumes the completed resource vectors and the dependency/concurrency edges (Section 3.3), applied recursively across nested loops, waves, and network stages. None of these are post-hoc corrections—each piece either populates or consumes the same per-tile resource vector that flows through the envelope.

### 3.8 Portable Hardware Abstraction

TileSight requires only the parameters that affect the tile execution plan and its placement descriptors. The abstraction mirrors the tensor-placement hierarchy: local placements map to register/TMEM/SMEM/cache/DDR resources, remote placements map to a calibrated network hierarchy across GPUs and nodes (Table 3). Values come from vendor specifications and lightweight microbenchmarks for practical bandwidth, utilization caps, and network parameters.

**Table 3: Hardware specifications: theoretical peak (spec) / microbenchmark-calibrated (meas.) for GPU architectures evaluated in this paper.**

| GPU | SMs | VEC FP32 T spec / meas. | TC FP16 T spec / meas. | SFU T spec / meas. | L2 TB/s meas. | DDR TB/s spec / meas. |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A100 | 108 | 19.5 / 19.0 | 312 / 299 | 2.4 / 2.4 | 3.2 | 1.9 / 1.7 |
| H200\* | 132 | 61.8 / 49.5 | 989 / 928 | 3.9 / 4.1 | 9.2 | 4.8 / 4.2 |
| B6000 | 188 | 117 / 88.6 | 468 / 433 | 7.3 / 6.7 | 7.6 | 1.8 / 1.4 |
| B200 | 148 | 74.5 / 57.7 | 2382 / 2185 | 4.7 / 4.5 | 20.5 | 8.0 / 7.0 |
| MI210 | 104 | 45.3 / 34.4 | 181 / 167 | 2.8 / 1.1 | 4.8 | 1.6 / 1.4 |

*Note*: TileSight's hardware abstraction also includes cache hierarchy, architecture-specific TMEM bandwidth, SMEM/occupancy limits, and network hierarchy across GPU groups. Not listed here for simplicity. \*H200 has a maximum clock of 1980 MHz and a default clock of 1830 MHz.

TileSight does not model warp-level instruction issue, compiler register allocation, hardware scheduling at instruction granularity, or packet-level network effects. Instead, it models the schedule-visible effects that tile-level programmers and distributed runtimes control: tile shape, tensor placement, reuse pattern, swizzle order, pipeline depth, resident blocks per SM, distributed partitioning, collective algorithm, and topology-aware routing. This is what makes the model both portable across GPU generations and fast enough to use inside schedule search.

## 4. Implementation

TileSight is implemented in Python ($\sim$6K lines) and supports NVIDIA and AMD GPUs. Users describe kernels as tile-based programs, either extracted from Triton or TileLang code, or written by hand for non-DSL kernels, and TileSight produces a full performance breakdown without running the kernel.

**Describing arbitrary fused programs.** To represent arbitrary kernels, TileSight describes the operations executed within each tile as a tile-action DAG (Section 3.1). Each tile action is annotated with the `HardwareUsage` resource vector (Section 3.1) and two additional attributes: (1) explicit data dependencies among actions, and (2) the *scratchpad memory level at which intermediate results reside*: the register file, shared memory, or architecture-specific tensor memory (TMEM) on Blackwell. The scratchpad annotation determines the bandwidth tier charged for each data movement between actions and how much on-chip capacity is consumed, which in turn constrains occupancy. Data dependencies are declared between tile-action nodes. TileSight automatically enumerates all valid topological orderings consistent with these dependencies and selects the schedule minimizing tile latency.

**Software pipeline and occupancy.** For pipelined kernels, the user provides the pipeline depth, corresponding to `num_stages` in Triton or explicit stage counts in TileLang. Given kernel resource usage, such as shared memory per tile and register count, TileSight computes the number of resident tiles per SM as the resource-limited minimum. This determines the effective pipeline depth and per-SM bandwidth allocation. TileSight models head and tail waves separately: the tail wave has fewer active SMs, so each SM has a larger per-SM share of L2 and DDR bandwidth, which is reflected in the per-tile latency computation.

**Single GPU to cluster.** At the single-GPU level, the entire tile grid is scheduled on one device. At the node level, a `DistributedTileMap` partitions the grid across GPUs and a `NetworkHierarchy` captures the intra-node interconnect, including NVLink or PCIe. TileSight selects collective algorithms, such as ring, recursive-doubling, Rabenseifner, based on message size and device count. For multi-node clusters, the same `NetworkHierarchy` is extended with inter-node links, such as InfiniBand or NVLink Bridge. Users can specify custom topologies by providing per-hop bandwidth and latency for any link. Given a `DistributedTileMap`, TileSight infers the required remote tensor accesses from producer-consumer placement of the partitioned tile grid, decomposes each into ordered stages of $(s,d,b)$ logical exchanges, and applies the $\alpha$-$\beta$ stage cost over the `NetworkHierarchy` to produce a per-tile `Net` resource time that flows through the same pipeline envelope as local compute and memory.

![Figure 5: GEMM latency prediction vs. measured latency across A100, B200, B6000, H200, and MI210. Each point is one BF16/FP16 tensor-core GEMM shape; the diagonal indicates exact prediction.](../../papers/tilesight/figure-05.png)

**Table 4: FlashAttention-3 modeling compared with NCU on H100 (Qwen configuration: batch 1, 64 heads, head-dim 128). NCU is ground truth.**

|  | Time (ms) | L2 hit (%) | L2 util. (%) | SMEM (%) | TC (%) | SFU (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| NCU | 5.58 | 96.50 | 38.66 | 51.14 | 74.78 | 38.58 |
| TileSight | 5.73 | 95.26 | 35.72 | 43.13 | 70.30 | 35.42 |

**Composition and calibration.** The modeling chain runs bottom-up: our cache model computes L1.5/L2/DDR traffic fractions based on tile schedule and reuse distances. These traffic data feed into the per-tile pipeline overlap model. The wave model aggregates per-tile results into per-device time, while the distributed model adds communication and computes overlap. Memory/TMEM bandwidth, per-unit compute throughput, and other hardware parameters are calibrated once per architecture with small microbenchmarks. This consists of bandwidth sweeps over working-set sizes, as in Figure 2, and short matrix-multiply probes that only take seconds.

![Figure 6: TileSight L2 hit-rate prediction vs. NCU ground truth across 4,680 GEMM persistent-kernel cases.](../../papers/tilesight/figure-06.png)

## 5. Evaluation

![Figure 7: Pure-collective prediction on H200 $\times 8$ and B200 $\times 8$ across AllGather, AllReduce, ReduceScatter, and All-to-All.](../../papers/tilesight/figure-07.png)

To demonstrate that a unified tile-level abstraction can model single-operator latency, cache behavior, distributed kernels, and end-to-end serving without per-target retraining or profiling, we evaluate TileSight from single-GPU kernels to multi-GPU LLM serving across A100, H200, B200, B6000, H200-NVL, and B200 $\times 32$ systems.

We first describe the experimental setup in Section 5.1, including the hardware and framework configuration, workloads, and baselines. We then validate the core tile-level model on single-GPU operators in Section 5.2, followed by a deeper analysis of L2 cache prediction for persistent kernels in Section 5.3. Sections 5.4 and 5.5 extend the evaluation to distributed settings, covering both collective/fused compute-communication kernels and end-to-end vLLM serving. Finally, Section 5.6 shows how TileSight can be used for performance diagnosis and cost-model-guided schedule pruning.

### 5.1 Experimental Setup

**Hardware and framework.** To ensure broad hardware coverage, we evaluate A100 $\times 1$, H200-SXM $\times 8$, B200 $\times 8$, an InfiniBand-connected B200 $\times 32$ cluster, B6000 $\times 2$, and H200-NVL $\times 8$ systems, spanning SXM, PCIe, NVLink4/5, and multi-node settings. We use CUDA 12.9 on Hopper and Ampere machines and CUDA 13.1 on Blackwell machines. We additionally evaluate AMD MI210 (CDNA2) on ROCm 6.2 for cross-vendor coverage. GEMM measurements use `cutlass_profiler` on NVIDIA GPUs and Composable Kernel (CK) on MI210; distributed kernels use Parallel Kittens [Sul25]; and end-to-end serving uses vLLM 0.19.0.

**Workloads.** Kernel-level experiments cover BF16/FP16 GEMMs, persistent-kernel cache sweeps, collectives, and fused compute-communication kernels. End-to-end vLLM experiments include dense and MoE models from the Qwen, Llama, and DeepSeek families, ranging from single-GPU serving to tensor-, expert-, and data-parallel serving on up to 32 GPUs. In total, we evaluate 703 GEMM shapes, 4,680 persistent-kernel cases for cache modeling, and 166 vLLM decode configurations.

**Baselines.** For single-operator prediction, we compare against Roofline [Wil09], NeuSight [Lee25][+1], PipeWeave [Zha26p], and GenZ [Bam24]. NeuSight is trained on BF16/FP16 GEMM data from six PipeWeave GPUs, including A100. The PipeWeave dataset covers A100 and Hopper-class machines, so PipeWeave is not a zero-shot baseline on those architectures. For distributed kernels and end-to-end serving, we compare against PipeWeave and GenZ. PipeWeave's collective model is a per-GPU random forest with no configurable $\alpha$-$\beta$ or topology parameters; among our targets, A100 and B6000 (RTX PRO 6000 Blackwell) have native PipeWeave collective datasets, while H200-NVL and B200 are unsupported and we use its H800 dataset as the closest available substitute. For end-to-end serving, we provide PipeWeave with the required vendor hardware specifications.

[+1]: The original NeuSight is trained only on FP32 GEMMs; we retrain it with PipeWeave's FP16 dataset for a fair comparison.

### 5.2 Single-Operator Prediction Accuracy

![Figure 8: Fused compute-communication kernel prediction on H200 $\times 8$ and B200 $\times 8$ (AllGather+GEMM, GEMM+ReduceScatter, Ulysses Attention).](../../papers/tilesight/figure-08.png)

![Figure 9: vLLM decode throughput prediction across dense LLMs, MoE models, and multi-node configurations. Dense rows cover A100 $\times 1$, B6000 $\times 2$, B200 $\times 8$, and H200-NVL. MoE rows cover B200 $\times 8$, B200 $\times 32$, and H200-NVL $\times 8$. Bars compare measured vLLM tokens per second with TileSight and PipeWeave where supported. PipeWeave does not support MoE.](../../papers/tilesight/figure-09.png)

![Figure 10: Predicted vs. measured decode throughput across all healthy configurations. TileSight: 13.52% wMAPE overall. PipeWeave: 31.84% wMAPE on supported dense rows.](../../papers/tilesight/figure-10.png)

Figure 5 evaluates 703 BF16/FP16 tensor-core GEMM shapes on A100, B200, B6000, and H200, using `cutlass_profiler` measurements as ground truth after filtering stream-K and single-instruction, multiple-thread (SIMT) fallback paths. TileSight achieves 12.35% pooled MAPE, compared with 21.97% for PipeWeave, 32.95% for retrained NeuSight, 33.85% for Roofline, and 34.89% for GenZ. TileSight is best on the newer B200, B6000, and H200 targets. NeuSight narrowly leads on A100 because A100 appears in its training distribution, but this advantage does not transfer to newer GPUs, illustrating the overfitting risk of architecture-specific learned predictors. On MI210, because CK provides no explicit rasterization (along-$M$/along-$N$) or swizzle control as `cutlass_profiler` does, TileSight runs in its default cache mode, yet still leads at 23.4% MAPE, ahead of PipeWeave (25.5%), NeuSight (26.4%), Roofline (38.8%), and GenZ (40.4%). Non-GEMM fused operators are evaluated in the distributed and end-to-end workloads below.

Table 4 compares TileSight with NCU on a fused FA-3 kernel. The final model predicts latency within 2.7% and tracks the major resource-utilization components, providing a compact sanity check for the tile-pipeline model on non-GEMM fused execution.

**Table 5: Performance improvements in TileSight diagnosed kernels.**

| Kernel | Framework | Device | Baseline | Issue | Solution | Optimized | Speedup |
| --- | --- | --- | ---: | --- | --- | ---: | ---: |
| ReLU | Triton | MI210 | 1.40 ms | Indirect addr. | Unroll addr. | 1.10 ms | $1.27\times$ |
| Avg_Pool | Triton | MI210 | 0.20 ms | Indirect addr. + Not Overlapped | Unroll addr. + Small tile | 0.10 ms | $2.00\times$ |
| Avg_Pool | Torch | MI210 | 0.15 ms | Not Overlapped | Small tile | 0.10 ms | $1.50\times$ |
| GEMM(M128) | CK | MI210 | 3.68 ms | Not Overlapped | Multi Thread Block per SM | 2.68 ms | $1.37\times$ |
| GEMM(K57344) | CK | MI210 | 55.63 ms | Large K with L2 hit rate issue | large tilek->1 TB per CU | 51.90 ms | $1.07\times$ |
| RMS_Norm | Torch.Compile | H100 | 0.21 ms | Not Overlapped | Multi Thread Block per SM | 0.18 ms | $1.17\times$ |
| MLA(kv8192 b128 h128) | Triton | MI210 | 66.38 ms | Tiling, Memory Alloc., SMEM Conflict | Register alloc., larger Tile, Conflict Elim. | 7.40 ms | **$8.97\times$** |

![Figure 11: Kernel performance on H100 and MI210 when TileSight guides tile configuration selection in Triton and TileLang, replacing exhaustive autotuning. Reference lines are FlashAttention-3 for multi-head attention/grouped-query attention (MHA/GQA), FlashMLA for MLA, cuBLAS/rocBLAS for matrix multiplication, and vendor libraries for dequantized matrix multiplication.](../../papers/tilesight/figure-11.png)

![Figure 12: TileSight as cost model in TileLang: pruning 95% of candidate schedules and retaining the predicted top 5% reaches 99.66% of exhaustive-search best performance on average across 10 LLaMA-derived GEMM-FP16 workloads.](../../papers/tilesight/figure-12.png)

### 5.3 L2 Cache Prediction Accuracy

Figure 6 evaluates tile reuse-distance cache modeling against NCU on 4,680 GEMM persistent-kernel cases. With effective cache capacity calibrated by the bandwidth sweep in Figure 2, mean absolute L2 hit-rate error stays near one percentage point on every GPU: 1.46 pp on A100, 0.88 pp on H200, 1.05 pp on B200, and 0.78 pp on B6000. The results demonstrate the effectiveness of tile reuse-distance cache modeling.

**Effect of inter-SM execution skew.** The reuse-distance model assumes tiles advance at a uniform rate, but SMs desynchronize and work on different $K$-slices at once, spreading the concurrently accessed tiles beyond L2; for deep-$K$ GEMMs this pushes the measured hit rate below TileSight's lockstep prediction (e.g., a GEMM with $M{=}N{=}8192$, $K{=}28672$ on H200: 82% predicted vs. 43% measured). Such configurations are rare, so aggregate error stays near one percentage point, but the model is systematically optimistic in this regime; we revisit it in Section 7.

### 5.4 Distributed Validation

Figures 7 and 8 validate 304 distributed cases on H200 $\times 8$ and B200 $\times 8$: 152 pure collectives and 152 fused compute-communication kernels. TileSight extracts logical source-destination exchanges, routes them over calibrated NVLink topologies, and evaluates each stage with the $\alpha$-$\beta$ model from Section 3.6. On pure collectives, TileSight achieves 12.22% wMAPE, compared with 20.82% for GenZ and 65.72% for PipeWeave on supported rows. PipeWeave has no native configurable H200/B200 backend for these collectives and falls back to an H800 random-forest model, so it cannot reflect the NVLink4/5 bandwidth differences in our machines. For B200 Ulysses Attention, the local compute stage uses the source-aligned SM100 $128\!\times\!128$ FA4 tile pipeline with TMEM traffic, packed grids, and the sectioned LPT mapping, composed with four all-to-all stages. On fused kernels, where both baselines are unsupported, TileSight achieves 14.83% wMAPE.

### 5.5 vLLM End-to-End Decode

Figures 9 and 10 evaluate end-to-end vLLM decode throughput on 166 healthy configurations spanning dense, MoE, single-node, and multi-node serving. The evaluated systems range from A100 $\times 1$ and B6000 $\times 2$ to B200 $\times 32$ and H200-NVL $\times 8$, exercising both local tile execution and routed distributed stages. Overall, TileSight achieves 13.52% wMAPE, while PipeWeave with the B200 extension reaches 31.84% wMAPE on 114/117 dense configurations. PipeWeave does not support MoE. PipeWeave uses native collective datasets for A100 and B6000, but falls back to H800 for H200-NVL and B200. For B200, we extend PipeWeave by supplying B200 hardware specifications while using its closest available H800 samples for GEMM-configuration lookup and its Hopper calculator. The B200 extension produces valid predictions for 19/22 dense configurations. In the remaining three large-batch cases, the prefill RMSNorm sequence lengths exceed PipeWeave's 131K-token MLP training maximum. Although PipeWeave bounds its learned utilization factor to $[0,1]$ with a sigmoid, these out-of-range inputs drive it to zero, triggering division by zero and preventing robust end-to-end prediction for these cases. This highlights a robustness limitation of ML-based predictors when extrapolating to unseen cases. TileSight achieves 7.5-18.0% per-machine wMAPE and 10.35% wMAPE on MoE configurations.

### 5.6 Key Applications: Diagnosis and Cost Model

Due to its interpretable nature, TileSight can be used as a white-box optimization aid. Figure 11 shows that tile configurations selected by TileSight can match or exceed strong vendor and expert baselines across attention, MLA, GEMM, and dequantized matmul kernels on H100 and MI210. Figure 12 shows the same model used as a TileLang cost model: retaining the predicted top 5% schedules prunes 95% of candidates while reaching 99.66% of exhaustive-search best performance on average. This is especially useful on less-supported targets, where learned or vendor-tuned cost models provide weak guidance but the analytical model can still surface high-quality schedule candidates.

The diagnosis cases fall into four recurring bottleneck classes: indirect addressing, insufficient pipeline overlap, poor L2 locality, and architecture-specific memory-layout issues. In each case, TileSight maps the bottleneck to concrete tile-level changes, such as address unrolling, tile-size adjustment, higher resident-block occupancy, or shared-memory/register-layout fixes. Table 5 summarizes diagnosis cases where TileSight identifies indirect addressing, pipeline stalls, and L2 locality bottlenecks, leading to 1.07-8.97$\times$ improvements.

## 6. Related Work

**Tile-Centric Programming Frameworks.** Triton [Til19], TileLang [Til25g], TileLink [Zhe25t], CUTLASS/CUTE [Nvi24], CuteDSL [NviCut], ThunderKittens [Ben25], FractalTensor [Liu24t], and NVIDIA's CUDA Tile [Nvi26] have driven GPU programming toward tile-centric abstractions. Yet none ships a tile-centric performance model: Triton relies on black-box autotuning, TileLang on heuristics, and tritonBLAS [Swa25] on GEMM-specific analytical selection. TileSight fills this gap as a unified tile-centric cost model and diagnosis backend for these frameworks.

**Performance Modeling and Prediction.** Roofline [Wil09] and its variants (e.g., GenZ [Bam24], [Mor24], [Yua24t], [Pat25t], [Dav25]) provide useful first-order bounds for LLM inference but cannot distinguish kernels with different schedules at identical FLOP/byte counts, nor capture schedule-dependent effects such as L2 reuse under different tile orders. Karami et al. [Kar25] further show non-GEMM ops account for up to 74% of inference latency, challenging GEMM-centric assumptions. Dataflow exploration frameworks [Par19], [Gao19t], [Kwo20], [Zhe23], [Wu22], [Cai23] model loop nests and data reuse for spatial accelerators but rely on simplified hardware assumptions that limit GPU applicability. Hybrid and ML-based approaches—PipeWeave [Zha26p], NeuSight [Lee25], CDMPP [Hu24t], TAO [Pan24], Omniwise [Wan25o], among others [Geo21], [Li23t]—predict runtime via learned models (either end-to-end or as residuals on top of analytical estimates), often accurately but as black boxes without exposing *why* a kernel is slow. TileSight differs by being fully first-principles with no learned component, yet matches or exceeds these predictors' accuracy while offering schedule-aware, tile-granular diagnosis that decomposes performance into actionable components.

**GPU Profiling and Instrumentation.** Vendor profilers (Nsight Compute [Nvi25n], OmniPerf [AMD25]) report metrics but little root-cause guidance. KPerfIR [Gua25] and Neutrino [Hua25] advance compiler- and probe-based GPU instrumentation, while binary-level tools [She18], [Zho21a], [Zho21b], [Zen24] provide low-level visibility. All are *post-hoc*, requiring execution and unable to predict unseen configurations. TileSight predicts performance before execution and maps bottlenecks to tile-level scheduling decisions.

**Distributed Multi-GPU Performance Modeling.** Vidur [Agr24], Lumos [Lia25], SimAI [Wan25s], TokenSim [Wu25t], Maya [Yar25], and Echo [Fen24] simulate distributed training or inference at scale using profiling-based kernel estimators, while DistServe [Zho24], CrossPipe [Che25t], Sailor [Str25], Metis [Um24], and RAPID-LLM [Kar25a] optimize parallel strategies with various communication and scheduling models. All treat single-GPU kernel execution as a black box. TileSight operates at the complementary intra-kernel level, providing white-box tile-level cost estimation that can plug into these distributed simulators, while its own distributed extension composes tile-level predictions with communication models under a unified tile abstraction.

## 7. Limitations and Future Work

TileSight targets regular, tile-structured programs whose runtime is dominated by resource utilization. It does not model data-dependent control flow, highly irregular memory access, instruction-level compiler decisions, undocumented warp/cooperative thread array (CTA) scheduling, or closed-source runtime behavior. The hardware abstraction focuses on throughput. Latency-bound cases such as small-batch decode attention, and multi-die effects such as B200 SM-to-HBM affinity, require finer-grained latency and topology parameters. TileSight also assumes tiles execute at a uniform rate across SMs; in reality SMs desynchronize, which makes L2 hit-rate prediction mildly optimistic for large-$K$ GEMMs (Section 5.3). Finally, although TileSight has been validated as a TileLang cost model on selected GEMM workloads, broader compiler integration and non-GEMM schedule search remain future work.

## 8. Conclusion

TileSight shows that the tile, already a universal GPU programming unit across Triton, TileLang, CUDA Tile, and CuteDSL, can also unify performance reasoning. With tile-level modeling of resource use, dependencies, cache reuse, and cross-device placement, TileSight accurately predicts performance from single kernels to multi-node clusters without per-architecture training or profiling. The broader lesson is a first-principles one: begin with a compact set of physically grounded mechanisms, and let their composition explain complex execution. For regular tile-structured workloads, this approach can yield accurate and interpretable predictions that transfer across architectures.
