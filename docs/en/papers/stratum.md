---
title: 'Stratum'
createTime: 2026/08/22 20:00:00
permalink: /en/papers/stratum/
pageClass: paper-reading
---

> [Yue Pan](https://dblp.org/pid/385/3702), [Zihan Xia](https://dblp.org/pid/244/0846) [+equal-contribution], [Po-Kai Hsu](https://shimeng.ece.gatech.edu/people/), [Lanxiang Hu](https://snyhlx.github.io/), [Hyungyo Kim](https://cubic.engineering.columbia.edu/directory/hyungyo-kim), [Janak Sharda](https://grad.gatech.edu/events/phd-dissertation-defense-janak-sharda), [Minxuan Zhou](https://zhouminxuan.github.io/), [Nam Sung Kim](https://ece.illinois.edu/about/directory/faculty/nskim), [Shimeng Yu](https://ece.gatech.edu/directory/shimeng-yu), [Tajana Rosing](https://cseweb.ucsd.edu/~trosing/), and [Mingu Kang](https://jacobsschool.ucsd.edu/node/3664). First submitted to arXiv on October 6, 2025; current version v1. [Stratum: System-Hardware Co-Design with Tiered Monolithic 3D-Stackable DRAM for Efficient MoE Serving](https://arxiv.org/abs/2510.05245). [Original paper PDF](/paper/stratum.pdf). [DOI](https://doi.org/10.48550/arXiv.2510.05245). [MICRO '25 DOI](https://doi.org/10.1145/3725843.3756043). [TeX source](https://export.arxiv.org/e-print/2510.05245v1). The original PDF remains authoritative for exact print layout and bibliography.

## Abstract

As Large Language Models (LLMs) continue to evolve, Mixture of Experts (MoE) architecture has emerged as a prevailing design for achieving state-of-the-art performance across a wide range of tasks. MoE models use sparse gating to activate only a handful of expert sub-networks per input, achieving billion-parameter capacity with inference costs akin to much smaller models. However, such models often pose challenges for hardware deployment due to the massive data volume introduced by the MoE layers. To address the challenges of serving MoE models, we propose Stratum, a system–hardware co-design approach that combines the novel memory technology Monolithic 3D-Stackable DRAM (Mono3D DRAM), near-memory processing (NMP), and GPU acceleration. The logic and Mono3D DRAM dies are connected through hybrid bonding, whereas the Mono3D DRAM stack and GPU are interconnected via silicon interposer. Mono3D DRAM offers higher internal bandwidth than HBM thanks to the dense vertical interconnect pitch enabled by its monolithic structure, which supports implementations of higher-performance near-memory processing. Furthermore, we tackle the latency differences introduced by aggressive vertical scaling of Mono3D DRAM along the $z$-dimension by constructing internal memory tiers and assigning data across layers based on access likelihood, guided by topic-based expert usage prediction to boost NMP throughput. The Stratum system achieves up to $8.29\times$ improvement in decoding throughput and $7.66\times$ better energy efficiency across various benchmarks compared to GPU baselines.

<span id="section-1"></span>

## 1 Introduction

Transformer-based Large Language Models (LLMs) have become central to a wide range of applications, delivering state-of-the-art performances across diverse domains [Vas17c, Dub24, Dos20, Zha22, Gro25a, Jia24a, Ope23b, Qwe25, Den20, Dee24a, Dee25c]. To improve various task performances, LLMs are reaching unprecedented scales, with models such as LLaMA 3.1 (405B) [Dub24], DeepSeek-V3 (671B) [Dee24a], and Kimi-K2 (1T) [Kim25e] pushing the boundaries of model size and performance. Training and deploying these large models present significant challenges to the underlying infrastructure, particularly in terms of memory capacity and compute capability.

Among various efforts to reduce the inference cost, exploiting activation sparsity offers a promising solution by directly reducing the computational and data movement demands. One of the most widely adopted approaches is the Mixture of Experts (MoE) architecture [Dee24a, Ope23b, Olm24, Jia24a, Du21a, Dbr25, Gro25a, Fed22, Lla25], which replaces conventional dense Multi-Layer Perceptron (MLP) blocks with a pool of expert MLPs that are sparsely selected during inference, as illustrated in [Figure 1](#figure-01). MoE models utilize a routing mechanism to activate only a small subset of experts per token during inference. Since MLP dominates the overall model size, this selective activation leads to substantial savings in both inference and training costs [Sca24]. As a result, the MoE architecture has become a preferred choice in many state-of-the-art LLMs.

<span id="figure-01"></span>

![Figure 1. Architectures of dense transformer-based LLM (left) and Mixture of Experts (MoE) LLM (right).](../../papers/stratum/figure-01.png)

**Figure 1.** Architectures of dense transformer-based LLM (left) and Mixture of Experts (MoE) LLM (right).

While MoE models reduce practical memory access and computation requirements, they do not address the overall size of the model. The rapid growth in model size necessitates high-bandwidth and high-density memory technologies. Along this line, die-stacked High Bandwidth Memory (HBM) has emerged as the dominant solution in high-performance GPUs such as the NVIDIA A100 and H100 [Nvi21, Nvi23b], achieving high density per footprint with six stacked DRAM dies and 1024-bit I/O interfaces, delivering up to 800 GB/s of memory bandwidth per stack to the GPU compute die via silicon interposers. Although HBM offers increased bandwidth compared to conventional 2D DRAMs, the bandwidth available through the interposer remains insufficient. This limitation often leads to underutilization of GPU computing resources, particularly for memory-bound operations such as LLM decoding [Att24]. To mitigate the memory wall between HBM and the GPU, recent approaches have adopted near-memory processing (NMP) for LLM inference [New20, Tra22, Har21, Att24, Dup24, Neu24a, Hig21]. Prior studies [Neu24a, Att24, Tra22, Dup24] have utilized NMP units to compute attention during the decoding stage by placing the computing logic on the HBM base die. However, the NMP on the base die still suffers from limited bandwidth due to vertical data traversal through a constrained number of TSV I/O connections. To mitigate this limitation, prior work has integrated compute units directly into the memory dies to exploit extensive internal memory bandwidth [Har21, Tra22, Dri17, Att24, Hig21, Pri24], commonly known as processing in memory (PIM). However, compute logic embedded in DRAM dies suffers from expensive intra-memory data transmission and large performance-area-power (PPA) overhead of implementing logic using the DRAM technology, as DRAM dies are inherently optimized for storage rather than computation [Dri17]. Moreover, integrating logic and memory on the same die introduces additional thermal concerns and manufacturing overheads.

As a strong alternative to HBM, Monolithic 3D-Stackable DRAM, referred to as Mono3D DRAM throughout this paper, has recently emerged as a promising solution for continued DRAM scaling beyond sub-10-nanometer technologies. It offers improved vertical integration through a cost-effective fabrication process that eliminates costly TSV and bonding processes, gaining growing attention in both industry and academia [Ong23, A24, Sig25, Mon25]. By fabricating multiple additional DRAM layers sequentially on the same wafer, Mono3D DRAM achieves higher density without a proportional increase in cost per bit, making it an attractive candidate for future high-capacity memory systems. Compared to HBM-based NMP, Mono3D DRAM-based NMP introduces key architectural benefits. Mono3D DRAM offers significantly greater internal bandwidth due to its monolithic construction within DRAM and direct face-to-face hybrid bonding between DRAM and logic dies, leveraging the full chip area. On the other hand, TSVs in HBM require a certain area on both the logic base die and DRAM dies as vertical interconnects. The TSV area cannot be unbounded, thus limiting the HBM internal bandwidth. Moreover, hybrid bonding pitch of 1 $\mu m$ [Che20a] has around $5\times$ finer pitch for vertical interconnects than HBM [Exp24], offering denser internal connectivity. The higher internal bandwidth of Mono3D DRAM can enable stronger NMP capability with the logic-die implementation than prior HBM-based memory-die NMP architectures. In addition, thinner dies and improved vertical thermal conduction enabled by monolithic integration enhance heat dissipation, supporting higher power density and allowing a larger power budget for NMP.

Despite the numerous potential benefits offered by Mono3D DRAM, fully leveraging its advantages presents several critical challenges. Recent studies have demonstrated the feasibility of integrating several hundred vertically stacked layers through sequential layer fabrication [Sig25, Mon25]. However, such aggressive vertical scaling inherently leads to substantial variability in access latencies across different layers. Adopting a simplistic design based on the worst-case latency significantly undermines the available internal bandwidth. Additionally, the drastically increased density of vertical interconnects, enabled by the fine-pitch monolithic 3D integration, facilitates simultaneous access to large volumes of data. Consequently, a carefully tailored data mapping strategy is essential to effectively harness local Mono3D DRAM bank bandwidth while minimizing inter-bank and inter-channel data access. Furthermore, given the extremely high local DRAM data access bandwidth, the overhead of on-chip communication between processing units can become comparable to the computation latency if data is mapped inefficiently. Therefore, achieving a balanced overlap between computation and communication is crucial for minimizing the overall execution time.

To address the challenges in serving large MoE models, we propose the Stratum system that integrates Mono3D DRAM, NMP, and GPU. This work makes the following key contributions:

- For the first time, we propose a system-hardware co-design solution Stratum for MoE serving that leverages Monolithic 3D-Stackable DRAM. Our approach heterogeneously integrates high-density Mono3D DRAM dies with high-performance logic dies via 3D hybrid bonding, and further integrates this Mono3D DRAM stack with GPUs using a 2.5D silicon interposer. This architecture serves as a high-throughput and cost-effective alternative to conventional GPU-HBM-based MoE serving systems.
- At the hardware level, we introduce an in-memory tiering mechanism that exploits the inherent access latency variations across Mono3D DRAM layers resulting from vertical scaling. Additionally, we propose an NMP processor tailored for hybrid-bonding-based Mono3D DRAM, incorporating optimized data mapping and communication strategies for both expert and attention execution.
- At the system level, we observe the nonuniform activation frequency of experts depending on user request topics. Based on this, we classify experts into hot and cold categories and assign them to fast and slow tiers of Mono3D DRAM, respectively. The proposed topic-aware serving system queues and dispatches requests according to their topics, predicted by our lightweight topic classifier, while adhering to defined service-level objectives (SLOs).
- Cross-layer evaluations (device, circuit, algorithm, and system) demonstrate that Stratum achieves up to $8.29\times$ better decoding throughput and $7.66\times$ better energy efficiency in practical MoE serving scenarios, compared to state-of-the-art GPU-baselines.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 Monolithic 3D-Stackable DRAM

Mono3D DRAM is a promising technology for continued DRAM scaling, drawing significant attention from both academia and industry [Har21, Tra22, Dri17, Att24, Hig21]. Compared to conventional 2D DRAM technologies, it offers significantly higher memory density by leveraging vertical scaling—enabled by advanced techniques such as nanosheet field-effect transistors (FETs), which provide tighter gate control and support stacked channel architectures, and fabrication techniques inspired by 3D NAND Flash processes, including layer-by-layer deposition, high-aspect-ratio etching for ultra-thin dielectric isolation, and dense vertical integration [Ong23, Sig25, A24, Mon25].

Mono3D DRAM employs monolithic 3D stackable horizontal 1T1C DRAM cells, incorporating wordline (WL) staircases and vertically connected bitlines (BL) to interconnect memory cells across multiple layers, as seen in [Figure 2](#figure-02). While HBM incurs high costs due to low manufacturing yield from TSV fabrication and the sophisticated packaging required for die stacking, Mono3D DRAM offers cost advantages through improved scalability by avoiding TSVs and leveraging monolithic 3D integration, which sequentially constructs additional DRAM layers on the same wafer. Mono3D DRAM also achieves thermal benefit using thinner dies and improved vertical thermal conduction enabled by monolithic integration.

On top of its cost and thermal benefits, Mono3D DRAM also delivers enhanced memory bandwidth to the logic layer. It leverages heterogeneous integration [A24, Mon25] and employs Cu–Cu hybrid bonding for high-speed data transfer between memory cells and logic peripherals. [Figure 3](#figure-03) compares Mono3D DRAM with HBM on the same 2.5D integration platform. HBM’s internal bandwidth is constrained by TSVs, which have a coarse pitch of 10 $\mu$m [Sma16], resulting in limited bandwidth and significant area overhead that reduces memory density. In contrast, Mono3D DRAM utilizes Cu–Cu hybrid bonding between DRAM and logic base dies with a much finer pitch of 1 $\mu$m [Che20a], connected via back-end-of-line (BEOL) metal routing, to achieve exceptionally high internal bandwidth.

Despite its higher internal bandwidth, Mono3D DRAM, as shown in [Figure 3](#figure-03), still has external bandwidth limitations similar to HBM due to the limited bandwidth of the interposer I/O interface. Additionally, prior work [Fin17] highlights the significant energy consumption incurred during data transfers to the external processor, including routing across the logic base die and through the interposer I/O interface. These inefficiencies underscore the necessity of NMP integration on the logic die alongside Mono3D DRAM to utilize internal bandwidth and improve energy efficiency.

Despite the potential for exceptional memory capacity in Mono3D DRAM, its vertical scalability is limited by substantial variation in access latency across layers. As shown in [Figure 2](#figure-02), WLs at the bottom of the staircase structure experience increased parasitic capacitance and resistance, resulting from the linearly extended WL routing. This latency imbalance becomes significant when Mono3D DRAM is scaled to hundreds of layers. Rather than designing around the worst-case access latency, system-level performance can be improved by embracing this latency heterogeneity. This challenge naturally motivates an architectural approach dubbed *in-memory tiering*, discussed in detail in [Section 3](#section-3). Note that the scaling trend of Mono3D DRAM aligns with that of 3D NAND Flash, as Mono3D DRAM leverages similar fabrication processes that have already been scaled beyond 400 layers [A25]. Furthermore, recent white papers suggest the feasibility of extending this scaling to 500 to even 1000 layers [New25, Sca24a]. Given these advancements and the projected trajectory of vertical scaling, we assume up to 1024 wordline (WL) stacks to reflect the near-future feasibility.

<span id="figure-02"></span>

![Figure 2. Monolithic 3D-Stackable DRAM with vertically stacked horizontal 1T1C DRAM cells. Bitlines are vertically routed to avoid sense margin variations, and wordlines are routed through staircases. The activation latency varies by layers due to wordline staircases.](../../papers/stratum/figure-02.png)

**Figure 2.** Monolithic 3D-Stackable DRAM with vertically stacked horizontal 1T1C DRAM cells. Bitlines are vertically routed to avoid sense margin variations, and wordlines are routed through staircases. The activation latency varies by layers due to wordline staircases.

<span id="figure-03"></span>

![Figure 3. HBM versus Mono3D DRAM on 2.5D integration platform with a xPU die. The HBM and Mono3D DRAM are attached to the logic base die through TSVs and Cu-Cu hybrid bonding, respectively.](../../papers/stratum/figure-03.png)

**Figure 3.** HBM versus Mono3D DRAM on 2.5D integration platform with a xPU die. The HBM and Mono3D DRAM are attached to the logic base die through TSVs and Cu-Cu hybrid bonding, respectively.

<span id="section-2-2"></span>

### 2.2 Mixture of Expert LLMs

As indicated by LLM scaling laws [Kap20], the accuracy of dense transformer models improves with size, but so do their training and serving costs. Recent MoE models, such as OLMoE [Olm24], Mixtral [Jia24a], Deepseek V3 [Dee24a], Time MoE [Tim24], DBRX [Dbr25], LLaMA-4 [Lla25], and Kimi-K2 [Kim25e], offer a compelling alternative by activating only a small subset of experts per token. This sparse activation improves training scalability and enables large parameter counts without proportional increases in pre-training cost [Sca24], while keeping inference costs comparable to smaller dense models [Fed22]. On the other hand, MoE models require a routing mechanism, where a gating network computes expert assignment scores from token representations (FFN input or intermediate activations) using learned router parameters that determine sparse expert selection patterns [Fed22]. Each token is then dispatched to its selected expert(s) for independent processing, and when multiple experts are used per token, their outputs are combined—typically via weighted aggregation using the routing scores—to produce the final output of the layer [Jia24a, Dee24a, Fed22].

The switching nature of MLP modules in MoE models introduces unique hardware deployment challenges. First, MoE models are large, with expert weights dominating the total size, e.g., over 95% of the model in Mixtral $8 \times 7$B [Jia24a], placing substantial pressure on GPU memory. Second, expert usage varies dynamically for each token and is unknown beforehand, leading to load imbalance when experts are distributed across different computing units [Dee24a]. Recent efforts aim to reduce communication overhead by predicting expert usage in advance. ExpertFlow [Exp24a] employs a lightweight surrogate model to forecast routing paths, while MoE Infinity [Moe25] uses cross-layer activation profiling to statistically predict expert selection. In hybrid GPU and near-memory processing systems, Duplex [Dup24] dynamically dispatches expert computation to either GPU or NMP units based on the latency models and batch size.

During training, MoE models typically include an expert imbalance loss to prevent starvation, where one or more experts are selected far less frequently, thereby encouraging more uniform expert utilization [Fed22, Jia24a]. However, as training progresses, domain specialization tends to emerge naturally among experts [Acc23, Chi19, Exp24b]. This specialization becomes increasingly pronounced as the number of experts increases and shared experts are introduced, consolidating common knowledge and enhancing the domain specificity of the routed experts [Olm24, Dee24a, Dai24, Lla25]. Building on this observation, recent work has explored leveraging expert affinity to specific domains to accelerate inference in GPU-only environments [Exp24b, Apt24, Moe25a].

<span id="figure-04"></span>

![Figure 4. Expert hit profiling from LLaMA-4 Scout (16 Experts).](../../papers/stratum/figure-04.png)

**Figure 4.** Expert hit profiling from LLaMA-4 Scout (16 Experts).

We profile and observe that the expert usage has a distinct relationship with the topic of the query: a particular topic activates certain experts significantly more frequently. An example is shown in [Figure 4](#figure-04), where LLaMA-4 Scout exhibits over 90% domain-specific expert affinity on math- and logic-related topics within MMLU subsets. In our serving system, we exploit topic-specific expert affinity by first conducting offline profiling to collect statistics on expert hit rates (i.e., usage probabilities) across various topics. During online serving, a lightweight topic classifier in the scheduler assigns topic labels to all incoming queries in a batch. Based on this classification, the system maps frequently used experts to faster Mono3D DRAM layers to optimize access latency, as discussed in [Section 5](#section-5).

<span id="section-3"></span>

## 3 Stratum Overview

<span id="section-3-1"></span>

### 3.1 System Overview

<span id="figure-05"></span>

![Figure 5. Example Stratum configurations.](../../papers/stratum/figure-05.png)

**Figure 5.** Example Stratum configurations.

<span id="figure-06"></span>

![Figure 6. Serving system based on Stratum.](../../papers/stratum/figure-06.png)

**Figure 6.** Serving system based on Stratum.

<span id="figure-07"></span>

![Figure 7. Stratum NMP architecture. (a) Overview of the processor at the chip level. Microarchitectures of (b) the processing unit (PU) at the channel level, and (c) the processing element (PE) at the bank level.](../../papers/stratum/figure-07.png)

**Figure 7.** Stratum NMP architecture. (a) Overview of the processor at the chip level. Microarchitectures of (b) the processing unit (PU) at the channel level, and (c) the processing element (PE) at the bank level.

The Stratum processing system consists of an xPU die and a configurable number of Monolithic 3D-Stackable DRAM chips, interfaced through silicon interposers, with near-memory computing capabilities. We demonstrate three different example configurations ([Figure 5](#figure-05)) to accommodate models of varying sizes, using different numbers of Mono3D DRAM chips. *Stratum-L* uses an NVIDIA H100 compute die as the xPU die with six Mono3D DRAM chips interconnected through interposers. *Stratum-S* uses a NVIDIA RTX A6000 die as the xPU die with a single Mono3D DRAM chip providing 32GB memory. *Stratum-XL* consists of two *Stratum-L* modules, providing a total of 384 GB of memory for serving larger models. These configurations suit diverse compute and memory requirements, and can scale up using cross-chip interconnects like NVLink [The22].

Each Mono3D DRAM chip consists of a memory die on top and a logic die at the bottom, which are interconnected by Cu-Cu hybrid bonding to provide high internal bandwidth. Additionally, to exploit access latency differences across the vertical layers of Mono3D DRAM, we introduce internal memory tiering within the memory die. The bottom logic die implements a powerful near-memory processor (NMP) to support LLM inference without always fetching data to the host processor, as detailed in [Section 3.2](#section-3-2).

[Figure 6](#figure-06) describes the flow of a serving system based on Stratum. In a realistic serving scenario, queries submitted by users are of varying topics. When users send inference requests, the host processor uses a lightweight topic classifier to determine the topic of the query. These requests are then enqueued in the serving queue with a topic tag. Periodically, the scheduler groups inference requests from the serving queue and later dispatches them to the Stratum processing system. To enhance user experience, a key Service-Level Objective (SLO) is Time to First Token (TTFT), which ensures that a request does not wait too long before processing begins. When SLO permits, the scheduler prioritizes batching requests of the same topic to maximize the benefits of expert placements. The memory mapper constructs the aggregated expert hit prediction for the batch by consulting the pre-profiled expert usage table and produces a target placement as a mapping between experts to Mono3D DRAM layers. Expert swaps are executed before every new batch with different topic tags to meet the target layout. Considering the arithmetic intensity of each stage, the Computation Mapper assigns the prefill phase to xPU and the decode phase to the Stratum NMP, following a similar strategy as in [Att24]. Additionally, the lightweight topic classification is executed by the host processor.

<span id="section-3-2"></span>

### 3.2 Stratum Near Memory Processing

[Figure 7](#figure-07) illustrates the architecture of Stratum NMP, which organizes processing components across multiple levels of the memory hierarchy—including chip, channel, and bank levels—to exploit the benefits of 3D integration. This architectural decision targets the acceleration of attention and expert computations, which are fundamental bottlenecks in MoE models.

[Figure 7](#figure-07)(a) illustrates the integration of the logic die processor with the Mono3D DRAM die. The logic die consists of multiple processing units (PUs), each coupled with a dedicated Mono3D DRAM channel. These PUs interconnect via a bidirectional ring-based on-chip network designed to optimize data communication patterns in LLM workloads, such as reduce-scatter and all-gather. Note that the ring network is only utilized in NMP mode. In regular memory operation mode, the logic die NMP remains inactive, ensuring minimal interference with traditional memory access patterns. In NMP mode, the xPU streams inputs (e.g., queries, hidden token vectors, etc.) to reserved rows in Mono3D DRAM banks with a standard DRAM interface. Upon computation completion, the xPU retrieves processed results by accessing the dedicated address space.

Each PU aims to handle data assigned to its respective DRAM channel to avoid cross-channel DRAM access—a critical consideration given the massive volume of vertical routing between Mono3D DRAM and the logic die. [Figure 7](#figure-07)(b) presents the PU microarchitecture, consisting of a near-bank processing element (PE) cluster, a shared memory, a special function engine, a ring router, and a reducer. The near-bank PE cluster integrates multiple PEs optimized for both GeMM and GeMV operations. The intra-channel reducer implemented with parallel reduction trees aggregates partial sums (psums) across multiple PEs within the channel as required. The ring router incorporates a local switch for efficient data routing during inter-PU communication and an aggregator for in-situ data reduction. Incoming data streams can be immediately accumulated in the router without going through the shared memory. The accumulated results can be stored locally in the PU or forwarded to neighboring PUs as needed. The special function engine performs special operations such as `Softmax` for attention mechanisms and other common activation functions (e.g., `SiLU`, `GeLU`) in expert layers. It includes a vector register file, a scalar register file, and multiple arithmetic units. Operating in a single-instruction-multiple-data (SIMD) manner, the special function engine maximizes data reuse by decomposing complex functions into simple primitives and sourcing and storing operands or intermediate results within the vector and scalar register files.

At the bank level, detailed in [Figure 7](#figure-07)(c), each PE is designed to execute GeMM and GeMV operations. The bank-level PE consists of a tensor core integrated with specialized memory components: a matrix register file, a psum memory, and a simple local memory controller. The memory controller, directly interfacing with its corresponding DRAM bank, dynamically translates row addresses to specific memory tier identifiers through a programmable tiering table, enabling adaptive DRAM latency control (`tRCD`) for performance optimization. The row swap buffer stores temporary row data to support tier-to-tier data movement without requiring explicit external data fetching. The tensor core incorporates $n$ parallel $k$-tap dot-product engines and $n$ local accumulators. The double-buffered psum memory structure concurrently supports intermediate result accumulation and output transfers. The processed outputs can be delivered to the special function engine for element-wise function evaluation or returned to the channel-level shared memory for subsequent computational steps.

Stratum’s architecture, specifically optimized for hybrid bonding-based Mono3D DRAM integration, differs from HBM-centric NMP approaches such as AttAcc [Att24], Neupims [Neu24a], and Duplex [Dup24]. The on-chip ring network is designed to support MoE inference communication patterns (e.g., all-gather, reduce-scatter), eliminating the centralized global buffer and crossbar used in Duplex [Dup24], which improves scalability and simplifies physical design. Unlike Duplex [Dup24] and AttAcc [Att24], which rely on dedicated `Softmax` units, our SIMD-based engine executes general non-linear operators with programming instructions. In addition, the processor is fully implemented on the logic die and hybrid-bonded to the Mono3D DRAM die, avoiding the DRAM fabrication process constraints and TSV bandwidth limitations observed in AttAcc [Att24] and Neupims [Neu24a]. At the circuit level, Stratum introduces Mono3D DRAM-specific primitives—including tiering tables and row swap buffers—to exploit tiered memory latency and accelerate expert migration for MoE model serving.

<span id="section-4"></span>

## 4 Stratum Operator Mapping and Execution

<span id="section-4-1"></span>

### 4.1 Expert Processing

The execution flow of an MoE layer consists of three main stages: token routing, expert computation, and result aggregation. As illustrated in [Figure 8](#figure-08)(a), tokens from a batch may be routed to different experts based on routing decisions computed on the xPU. This is feasible due to the negligible computational cost of the routing step, which typically involves a lightweight linear layer (e.g., 4096 input and 8 output dimensions). Subsequently, only the activated experts—i.e., those assigned at least one token—are executed. Finally, the outputs from all experts are merged using a weighted sum to produce the final output tokens. Both the expert computation and result aggregation are executed by Stratum NMP processor.

<span id="figure-08"></span>

![Figure 8. (a) Example of MoE’s token-to-expert mapping. (b) The computation stages of an expert with $M$ routed tokens and matrix partition, assuming four PUs for simplicity. (c) The step-by-step execution of the MoE layer in Stratum.](../../papers/stratum/figure-08.png)

**Figure 8.** (a) Example of MoE’s token-to-expert mapping. (b) The computation stages of an expert with $M$ routed tokens and matrix partition, assuming four PUs for simplicity. (c) The step-by-step execution of the MoE layer in Stratum.

The computation of a single expert in MoE models typically consists of three cascaded GeMM operations [Jia24a, Lla25], as shown in [Figure 8](#figure-08)(b). Let $M$ denote the number of tokens routed to one expert in the current batch, $K$ the hidden dimension, and $N$ the intermediate dimension. First, the input hidden matrix $\mathbf{X_{1}}$ of size $M\times K$ is multiplied by two weight matrices of size $K\times N$ to produce intermediate matrices $\mathbf{Z_{1}}$ and $\mathbf{Z_{2}}$ (both of size $M\times N$). A non-linear, element-wise activation is applied to $\mathbf{Z_{1}}$, and the result is combined with $\mathbf{Z_{2}}$ via a Hadamard product to form $\mathbf{X_{2}}$. Finally, $\mathbf{X_{2}}$ is multiplied by a projection-down weight matrix of size $N\times K$, producing the output $\mathbf{Z_{3}}$ of size $M\times K$.

**Partitioning Strategy.** In practice, different experts may receive different numbers of tokens. Furthermore, experts may be mapped to different tiers within the Mono3D DRAM hierarchy, each with varying memory access latency, further exacerbating load imbalance. Thus, distributing multiple experts across PUs could cause serious workload imbalance issues between PUs. To address this, the execution of multiple chosen experts is scheduled sequentially, e.g., one expert at a time. All PUs collaborate to process one expert at a time using tensor parallelism. This requires each matrix involved in all three GeMM operations to be partitioned into tiles, each assigned to a PU for parallel execution. [Figure 8](#figure-08)(b) illustrates the matrix partitioning scheme used in Stratum, where only four PUs are assumed for simplicity. Partitioning along different dimensions introduces trade-offs among input duplication, weight duplication, and partial sum aggregation. We avoid splitting along the $M$ dimension to prevent duplication of expert weights, which dominate memory usage. Instead, we split the weight matrix of the GeMM1 and GeMM2 vertically, while horizontally for GeMM3. Such a method eliminates data communication between projection-up and projection-down stages at the cost of duplicating ${\bf{X}}_{t}$ to multiple PUs initially and then gathering partial results from multiple PUs for ${\bf{Z}}_{3}$. Note that the cost of duplicating ${\bf{X}}_{t}$ is well amortized, as the input matrix ${\bf{X}}_{1}$ for all active experts is derived from ${\bf{X}}_{t}$ (i.e., the collection of tokens in the batch). In addition, the gathering from multiple PUs and reduction for ${\bf{Z}}_{3}$ can be computed in parallel with the next expert processing, effectively hiding the latency.

**Execution Stages.** [Figure 8](#figure-08)(c) illustrates the step-by-step execution flow of the MoE layer. The xPU begins by sending the batch of input tokens, along with the corresponding expert IDs and scaling weights, to the Mono3D DRAM and switches the Mono3D DRAM to NMP mode (step 1). Due to the adopted matrix partitioning strategy, each Mono3D DRAM channel must receive the entire input token matrix. Next, the Stratum NMP processor executes the activated experts sequentially through steps 2–7. In steps 2 and 3, the tensor cores in all PEs execute the two projection-up GeMM operations to compute the intermediate results $\mathbf{Z}_{1}$ and $\mathbf{Z}_{2}$. Steps 4 and 5 involve applying the activation function and performing the Hadamard product using the special function engines. Thanks to the matrix splitting strategy, no inter-PU communication is needed for each PU to obtain its required input slice for the third GeMM. The third GeMM is executed in step 6, followed by a reduce-scatter operation to accumulate the final output matrix $\mathbf{Z}_{3}$ across PUs. Steps 2–7 are then repeated for each of the remaining activated experts. In step 9, the special function engines perform a weighted sum across expert outputs to produce the final output tokens, which are written back to the designated DRAM memory space. Finally, in step 10, the Mono3D DRAM exits NMP mode, and the xPU retrieves the computed tokens by accessing the designated address space.

<span id="figure-09"></span>

![Figure 9. Optimized timing diagram of the expert processing.](../../papers/stratum/figure-09.png)

**Figure 9.** Optimized timing diagram of the expert processing.

**Execution Optimization.** [Figure 9](#figure-09) presents an optimized execution pipeline designed to maximize utilization of compute and communication resources. First, to mitigate the latency of xPU-to-Mono3D DRAM data transfer, the input token matrix is partitioned into multiple slices, with each slice sent to a distinct Mono3D DRAM channel. This reduces input preparation overhead, and a subsequent all-gather operation, enabled by the high-speed logic die ring network, reconstructs the full input matrix for all PUs. Second, the computation of GeMM2 is overlapped with the activation function evaluation, as there are no data dependencies between them, enabling better pipeline utilization. Third, the reduce-scatter communication associated with GeMM3 is parallelized with the GeMM1 execution of the next expert, thereby hiding communication latency behind computation. Finally, the weighted-sum operation is performed immediately by the special function engines as soon as each expert’s output becomes available, minimizing idle cycles and improving overall throughput.

Within each PU, communication overhead among PEs is negligible due to the high-bandwidth shared memory. As a result, intra-PU matrix partitioning is primarily focused on maximizing tensor core mapping utilization. To this end, the longer dimension of the weight matrix is partitioned, and the resulting sub-tiles are distributed across PEs for parallel processing. Therefore, the projection-up weight slices ${\mathbf{W}}_{1,2}[i]$ are typically partitioned horizontally, while the projection-down weight slice ${\mathbf{W}}_{3}[i]$ is partitioned vertically across PEs to optimize compute efficiency.

<span id="section-4-2"></span>

### 4.2 Attention Processing

The generation task in Large Language Models (LLMs) is often bottlenecked by data access to the key–value (KV) cache. Stratum addresses this issue efficiently by leveraging the high bandwidth between Mono3D DRAM and the NMP logic on the base die. However, to fully exploit this bandwidth, it is critical to effectively process the data fetched vertically from the DRAM layers on time. Otherwise, the available bandwidth may be underutilized due to computational or communication bottlenecks within the logic die.

<span id="figure-10"></span>

![Figure 10. Execution of attention layer. (a) Heads (e.g., eight) assignment across PU groups (e.g., four). Intra-PU group: (b) Attention operator mapping. (c) Concurrent processing of multiple heads (e.g., two).](../../papers/stratum/figure-10.png)

**Figure 10.** Execution of attention layer. (a) Heads (e.g., eight) assignment across PU groups (e.g., four). Intra-PU group: (b) Attention operator mapping. (c) Concurrent processing of multiple heads (e.g., two).

Stratum leverages head-level parallelism to efficiently execute attention operations due to the absence of data dependencies across attention heads. [Figure 10](#figure-10)(a) illustrates the assignment of attention head tasks on the logic die. Multiple attention heads from a group of requests can be assigned across Mono3D DRAM devices. The number of assigned heads can change depending on the network models, such as the common grouped query attention in MoE models [Lla25, Jia24a] and the concurrency of requests under a service latency requirement. To provide a processing architecture for diverse head-level parallelism, the PUs on the logic die can be flexibly partitioned into multiple PU groups of variable sizes, provided that the PUs within a group are neighbors connected through the on-chip ring topology as shown in [Figure 10](#figure-10)(a), where PUs connected with arrows indicate the PUS on the ring. This arrangement also allows efficient intra-group communication via high-speed bi-directional links. We assign at least two heads per group to enable interleaved processing across different computation stages for the enhanced throughput and hardware utilization—for example, one head may perform a linear operation while another executes the `Softmax`.

[Figure 10](#figure-10)(b) depicts how key and value matrices of a single head are partitioned across PUs within a PU group. Typically, the sequence length dimension (e.g., 512–32k tokens) is significantly larger than the attention head dimension (e.g., 64–128), motivating us to partition along the sequence length dimension. However, the `Softmax` operation inherently requires global information across all tokens, i.e., the global maximum (i.e., $\mathrm{row}_{\max}(\mathrm{Scores})$) and the global sum of exponentials (i.e., $\sum\exp(\mathrm{Scores}-\mathrm{row}_{\max}(\mathrm{Scores}))$) for normalization [A20]. Fortunately, each PU can independently compute local maxima and sums using its dedicated special function engine, requiring only scalar exchanges between PUs to derive global values. To balance the workloads of PUs in the decoding stage, the newly generated key-value pairs are distributed across different PUs within a PU group in a round-robin manner.

[Figure 10](#figure-10)(c) presents the optimized execution flow of multiple attention heads within a PU group. Initially, the xPU writes computed key-value pairs into the corresponding DRAM channels. Queries (which may be grouped query matrices) are partitioned into slices, each allocated to a distinct DRAM channel within a PU group. Subsequently, all PUs in the group obtain the complete query matrix via a sub-ring all-gather operation, analogous to the MoE layer. When multiple heads are assigned to the same PU group, the `Softmax` operation can be interleaved with the $\mathrm{query}\times\mathrm{key}$ and $\mathrm{attn.}\times\mathrm{value}$ operators to minimize the overall latency. Note that the `Softmax` operator is split into three steps with two rounds of inter-PU communications as shown in [Figure 10](#figure-10). Finally, the latency of the reduce-scatter of the first head can be hidden in the $\mathrm{attn.}\times\mathrm{value}$ operation of the second head.

In summary, Stratum best utilizes the vertical bandwidth enabled by hybrid bonding through optimized data placement, operator mapping, and scheduling. The system applies tensor parallelism across all PU for expert computation and uses grouped-PU head parallelism for attention. Both strategies direct most memory accesses to local Mono3D DRAM banks through hybrid bonding I/Os. The remaining inter-PU communication, such as all-gather, reduce-scatter, or scalar exchange, is efficiently supported by the on-chip ring network. Additionally, the scheduler overlaps matrix operations (e.g., GeMM and GeMV) with special-function computations (e.g., `SiLU` and `Softmax`), coordinating on-chip communication and compute to improve overall parallelism.

<span id="section-4-3"></span>

### 4.3 Design with Physical Constraints

The integration of Mono3D DRAM and the logic die processor via hybrid bonding must satisfy both thermal and area constraints. In the NMP mode, the system could be limited by a peak power budget, $P_{\mathrm{peak}}$, determined by thermal analysis (see [Section 6.2.2](#section-6-2-2)), leading to the power constraint as follows:

<span id="equation-01"></span>

$$
\begin{aligned}
P_{\mathrm{dram}} + P_{\mathrm{compute}} + P_{\mathrm{misc}} &\leq P_{\mathrm{peak}},\\
P_{\mathrm{dram}} &= \mathrm{BW}_{\mathrm{fast\_tier}} \cdot E_b,\\
P_{\mathrm{compute}} &= N_{mac} \cdot f_{\mathrm{logic}} \cdot E_{\mathrm{mac}}.
\end{aligned}
$$

Here, $\mathrm{BW}_{\mathrm{fast\_tier}}$ is the peak bandwidth of the fastest tier in Mono3D DRAM tier, $E_{b}$ represents the energy per bit for the data transfer from the DRAM layer to the logic die via hybrid bonding, $N_{mac}$ is the total number of multiply-accumulate (MAC) units in tensor cores, $f_{\mathrm{logic}}$ is the logic die operating frequency, and $E_{\mathrm{mac}}$ is the energy per MAC operation. The miscellaneous power, $P_{\mathrm{misc}}$, includes logic die SRAMs, register files, routers, special function engines, intra-PU reducers, and local memory controllers, varying according to the operator type and dataflow.

While hybrid bonding-based data I/O does not consume an active area in the logic die, TSVs remain necessary for power delivery to both DRAM and logic dies [Exp24]. Consequently, the following area constraint must hold:

<span id="equation-02"></span>

$$
A_{\mathrm{PD}}+N_{mac}\cdot A_{\mathrm{mac}}+A_{\mathrm{PHY}}+A_{\mathrm{peri}}+A_{\mathrm{misc}}\leq\alpha A_{\mathrm{chip}},
$$

where $A_{\mathrm{PD}}$ is the total TSV for power delivery, $A_{\mathrm{mac}}$ is the area per MAC unit operating at $f_{\mathrm{logic}}$, $A_{\mathrm{PHY}}$ represents the area of the physical communication layer of xPU-DRAM interface, $A_{\mathrm{peri}}$ is the area of low-voltage Mono3D DRAM peripherals on the logic die such as D/Q buffer, level shifters and others, and $A_{\mathrm{misc}}$ captures miscellaneous logic area components similar to those outlined for $P_{\mathrm{misc}}$, and $\alpha$ is the target utilization. Assuming a single TSV with area $A_{\mathrm{TSV}}$ can deliver $I_{\mathrm{TSV}}$ current, the total TSV area is given by:

<span id="equation-03"></span>

$$
\begin{aligned}
A_{\mathrm{PD}} &= \left(\frac{P_{\mathrm{dram\_c}}}{V_{\mathrm{dram\_c}}} + \frac{P_{\mathrm{dram\_p}}}{V_{\mathrm{dram\_p}}} + \frac{P_{\mathrm{compute}} + P_{\mathrm{misc}}}{V_{\mathrm{logic}}}\right) \frac{A_{\mathrm{TSV}}}{I_{\mathrm{TSV}}},\\
P_{\mathrm{dram\_c}} + P_{\mathrm{dram\_p}} &= P_{\mathrm{dram}}.
\end{aligned}
$$

where $V_{\mathrm{dram\_c}}$, $V_{\mathrm{dram\_p}}$, and $V_{\mathrm{logic}}$ denotes the supply voltage of Mono3D DRAM core, high-voltage peripherals, and low-voltage logic die. Equations ([1](#equation-01)), ([2](#equation-02)), and ([3](#equation-03)) will be used to guide the design configuration of the logic die processor (see [Section 6.2.3](#section-6-2-3)).

<span id="section-5"></span>

## 5 Stratum Algorithm-System Co-Optimizations

<span id="section-5-1"></span>

### 5.1 Expert Usage Prediction

As discussed in [Section 2.2](#section-2-2), pre-trained MoE models often exhibit domain-specific expert specialization at inference time [Exp24b], as shown in [Figure 4](#figure-04). Given that one of the main challenges in MoE inference is handling the large total parameter size across all experts, this specialization presents a valuable opportunity for efficient inference and serving. When expert specialization aligns with specific query topics, it becomes possible to optimize the placement of MoE experts. For a given topic, experts with higher usage probabilities (hit rates) can be mapped to faster Mono3D DRAM tiers, reducing the latency for the data transfer from DRAM to the base logic dies.

To enable MoE expert mapping, a key component of Stratum is a topic classifier that tags incoming queries. This allows the Stratum scheduler to estimate the topic distribution of each query. Combined with a per-topic expert usage table (as shown in [Figure 6](#figure-06)), the scheduler assigns experts’ weight matrices to the appropriate expert tiers. Our implementation trains a DistillBERT-based [Ber19, San19] topic classifier with 67M parameters on 6 topics as part of our online serving system built on Stratum. To account for distribution shifts from standard NLP datasets to the diverse prompting styles observed in real serving queries, we employ a data synthesis pipeline that uses GPT-4o-based rewriting to augment the training data. Due to their compact size, our topic classifiers introduce less than 2% latency overhead per decoding step at moderate request rates (fewer than four queries per second) on our experimental setup, while achieving 85.0% and 81.0% classification accuracy on real-world serving datasets (Chatbot Arena conversations [Chi24]) for the 6-topic model, respectively. Further details on data augmentation, training, and evaluation are provided in [Section 6.3.1](#section-6-3-1).

<span id="section-5-2"></span>

### 5.2 Data Placement Strategy

<span id="figure-11"></span>

![Figure 11. Example expert placement optimization for Mono3D DRAM-NMP system with tiered memory.](../../papers/stratum/figure-11.png)

**Figure 11.** Example expert placement optimization for Mono3D DRAM-NMP system with tiered memory.

<span id="algorithm-01"></span>

**Algorithm 1. Expert Weight Placement.**

- **Require:** number of layers $L$; experts per layer $K$; active experts $k$; usage frequencies $F = \{f_p^l\}$; expert size $S_E$ (bytes); DRAM banks $N_{\mathrm{bank}}$; row-buffer size $S_{\mathrm{rb}}$ (bytes); rows reserved for NMP data $\Phi$.
- **Ensure:** DRAM row-address intervals $[a_p^l, b_p^l]$ for every expert.
- $\Delta \leftarrow \lceil S_E / (N_{\mathrm{bank}} S_{\mathrm{rb}}) \rceil$ (rows occupied by one expert).
- $\tau \leftarrow kL$ (threshold of fast experts).
- **Sort** $F$ in descending order as $\langle f_{p_1}^{l_1}, \ldots, f_{p_{KL}}^{l_{KL}} \rangle$.
- **For** $i \leftarrow 1$ to $KL$:
  - **If** $i \leq \tau$:
    - $a_{p_i}^{l_i} \leftarrow (i - 1)\Delta$.
  - **Else:**
    - $a_{p_i}^{l_i} \leftarrow \Phi - (KL - i + 1)\Delta$.
  - $b_{p_i}^{l_i} \leftarrow a_{p_i}^{l_i} + \Delta - 1$.
- **Return:** $\{[a_p^l, b_p^l] \mid p \in [1,K], l \in [1,L]\}$.

Stratum categorizes the data within the MoE model into four types: hot expert weights, cold expert weights, KV cache, and non-NMP data. Hot experts include shared experts and other experts exhibiting high routing-hit probabilities for a given topic. Non-NMP data primarily consists of miscellaneous parameters such as positional embedding parameters, layer norm shift and scale parameters, and others. These are generally used for computation in the external processor rather than the NMP. By leveraging heterogeneous access latencies across different memory tiers, a data placement strategy can be optimized to enhance the serving performance.

As shown in [Figure 11](#figure-11), Stratum assigns non-NMP data, which is processed by the xPU, to the slowest memory tier, as accessing it requires traversing the interposer bottleneck, which is an order of magnitude slower than the internal DRAM bandwidth of the slowest tier. This helps preserve the faster memory tiers exclusively for NMP-related workloads. Stratum classifies experts into hot and cold categories based on offline profiling of topic-specific requests, assigning hot experts to faster memory tiers and cold experts to slower ones. This placement ensures that hot experts benefit from low-latency access provided by faster Mono3D DRAM memory tiers. The expert weight placement is detailed in Algorithm 1. Each expert weight is partitioned into shards and distributed across Mono3D DRAM banks according to the tensor parallelism strategy (see [Section 4.1](#section-4-1)). The mapping from physical row addresses obtained from Algorithm 1 to logical memory tiers functions as a quantization process, configurable via the tiering table (see [Section 3.2](#section-3-2)). In our evaluation, we adopt a uniform mapping strategy that assigns an equal number of rows to each memory tier (see [Section 6.2.1](#section-6-2-1)). KV cache data, whose capacity dynamically changes as request generation progresses, is stored in intermediate-speed memory. Upon completing the processing of one topic (e.g., topic A), the Stratum scheduler transitions to a new topic (e.g., topic B) and initiates expert swapping based on the expert activation frequencies of the new topic. To avoid costly host-processor transfers, this swapping is executed using near-memory operations, as detailed in [Section 3.2](#section-3-2). Specifically, the local memory controller performs the swap between two DRAM rows by temporarily buffering them in a dedicated row-swap buffer (see [Figure 7](#figure-07)(c)) before writing them back to their new row addresses.

<span id="section-6"></span>

## 6 Evaluation

<span id="section-6-1"></span>

### 6.1 Experimental Setup

<span id="section-6-1-1"></span>

#### 6.1.1 Monolithic 3D-Stackable DRAM Configuration

<span id="figure-12"></span>

![Figure 12. Mono3D DRAM bank configuration. The performance is simulated from NeuroSim Neu24b and Coventor process simulator Cov24.](../../papers/stratum/figure-12.png)

**Figure 12.** Mono3D DRAM bank configuration. The performance is simulated from NeuroSim [Neu24b] and Coventor process simulator [Cov24].

For Mono3D DRAM technology, we adopt the vertical bitline connections for 3D stackable horizontal 1T1C. We design the Mono3D DRAM scaled to 1024 layers and define the bank structure as in [Figure 12](#figure-12), where 1024 BLs $\times$ 1024 WLs form a MAT and 1024 MATs form a bank. To illustrate the impact of heterogeneous integration, [Figure 13](#figure-13) presents a 3D view of the proposed Mono3D DRAM bank. The high-voltage circuits are implemented beneath the memory array using a mature CMOS-under-array process, while the low-voltage circuits are fabricated on an advanced CMOS die and later hybrid-bonded to the memory tiers through Cu–Cu bonding pads. In this work, we leverage the 32 nm technology node for the CUA process and the 7 nm technology node for the bonded CMOS tier. To obtain the bank-level results, we utilize the Coventor process model [Cov24] for RC parameter extraction of the 3D DRAM array, and combine it with the peripheral circuit results extracted from NeuroSim [Neu24b] merging with the timing of DDR5 Standards [Ddr20], as shown in [Figure 12](#figure-12). The 1T1C model of Mono3D DRAM is built by the Coventor SEMulator3D process simulator [Cov24] based on a 3D DRAM structure specification in [Ong23]. The detailed parameters are listed in [Table 1](#table-01). The overall Mono3D DRAM achieves a memory density of 2.156 Gb/mm<sup>2</sup>, which is $5.2\times$ higher than that of the latest 32Gb DDR5 die (0.417 Gb/mm<sup>2</sup>[A24a]). It provides an internal bandwidth ranging from 19.01 TB/s to 30.34 TB/s, depending on the memory tier.

<span id="figure-13"></span>

![Figure 13. Mono3D DRAM array with heterogeneous integration, hybrid-bonding and CMOS-under-array (CUA).](../../papers/stratum/figure-13.png)

**Figure 13.** Mono3D DRAM array with heterogeneous integration, hybrid-bonding and CMOS-under-array (CUA).

<span id="table-01"></span>

![Table 1. Monolithic 3D-Stackable DRAM Parameters](../../papers/stratum/table-01.png)

**Table 1.** Monolithic 3D-Stackable DRAM Parameters

<span id="section-6-1-2"></span>

#### 6.1.2 Logic Die Processor Modeling

The components of the Stratum logic die processor are implemented using SystemVerilog and synthesized using Cadence Genus [Gen24] with the 7nm predictive process design kit ASAP7 [Asa16]. The hardware employs the IEEE754 FP-16 arithmetic data format [Iee19], widely adopted for LLM inference serving. The local psum memory and shared memory on the logic die are implemented with SRAMs modeled by FinCACTI [Fin14], calibrated with publicly available SRAM specifications [Coo24, A17]. The area measurements for the Stratum NMP processor components are obtained from synthesis reports. Energy consumption is determined through the simulations with post-synthesis netlists, which include annotated switching activity derived from random stimulus inputs. Execution cycles, on-chip communication cycles, and associated energy metrics are derived from an in-house simulator. The simulator takes as input tensor size information, parameter tier assignments (e.g., expert parameters or KV cache), attention head mappings, and routed expert IDs, along with the delay and energy parameters for each component. It outputs the overall execution time as well as detailed energy breakdowns at the component level.

<span id="section-6-1-3"></span>

#### 6.1.3 System modeling

<span id="table-02"></span>

![Table 2. Evaluation Workload Setup](../../papers/stratum/table-02.png)

**Table 2.** Evaluation Workload Setup

We evaluate with models (both MoE and regular LLMs) and system configurations shown in [Table 2](#table-02). Each GPU baseline and Stratum configuration is chosen to support the maximum evaluated context length without degrading performance. The GPU baselines are evaluated using vLLM 0.8.1 [Kwo23a] under benchmark throughput mode using NVIDIA RTX A6000 or H100 SXM5 HBM3 GPUs for different Stratum configurations. The GPU energy is derived from the NVIDIA-SMI tool.

The system-level simulator contains a Request Generator, SLO-Aware Scheduler, Memory and Computation Mapper, and interfaces to Stratum NMP simulator, in accordance with [Figure 6](#figure-06). The Request Generator models a Poisson process in which the incoming queries of certain topics arrive at defined rates. Taking into consideration serving SLO, the scheduler dynamically batches input queries to the Stratum processor for inference and prioritizes dispatching input queries of the same topic to maximize hot expert hits. Using the prior knowledge of the expert usage table, the memory mapper aggregates the topics in the batch and calculates expert placements for Mono3D DRAM that maximize hot expert hit, as shown in Algorithm 1. A memory reconfiguration is executed between dispatches to relocate experts. Energy and latency consumed by xPU and NMP are accumulated during simulated serving.

<span id="section-6-2"></span>

### 6.2 Hardware Evaluation

<span id="section-6-2-1"></span>

#### 6.2.1 Tiering in 3D-DRAM

<span id="figure-14"></span>

![Figure 14. Mono3D DRAM latency across WL layers. The inset illustrates various access latencies according to the increasing WL RC delay when scaling the staircase for increasing WL layers.](../../papers/stratum/figure-14.png)

**Figure 14.** Mono3D DRAM latency across WL layers. The inset illustrates various access latencies according to the increasing WL RC delay when scaling the staircase for increasing WL layers.

As illustrated in [Figure 14](#figure-14), Mono3D DRAM exhibits the almost linearly scaled access latency associated with the extending WL staircase structure for accessing various WL layers. As Mono3D DRAM vertically scaled with increasing WL layers, WL parasitics corresponding to the area of the staircase are also scaled, leading to a longer RC delay. Although the critical path for the bottommost WL suffers from long latency, the topmost WL has a shorter access latency, facilitating further optimization at the system level. In this work, we introduce the memory tiering technique for Mono3D DRAM. We define 8 timing tiers in Mono3D DRAM corresponding to different layers as shown in [Figure 14](#figure-14). The fast tier achieves $1.6\times$ faster access than the slowest tier.

<span id="section-6-2-2"></span>

#### 6.2.2 Power and area budget.

**Power.** The vertically integrated memory and logic dies require precise thermal modeling to determine the logic die’s power budget. We performed thermal simulations using the HotSpot [Tem03, Hot03] simulator for 3D IC. We consider high-end liquid cooling solutions with vapor chamber heat sinks. The heat sink is characterized by the following parameters: a convection capacitance of 75 J/K, a convection resistance of 0.01 W/K, and a thickness of 1 mm. The material properties include a thermal conductivity of 5000 J/(m$\cdot$K) and a specific heat capacity of $10^{6}$ J/(m${}^{3}\cdot$K). The thermal conductivity values are adopted from previous studies on vapor chamber thermal modeling [Per22a, Hea25a]. Additionally, advanced cooling fluids, such as phase change materials, achieve significantly reduced convection resistance of approximately $\mathrm{0.01\,W/K}$ [A22, Liq24]. Furthermore, we derived convection capacitance, heat sink thickness, and vapor specific heat parameters, explicitly considering the differences between conventional and vapor chamber heat sinks. Prior research demonstrates that state-of-the-art cooling methods for 3D ICs effectively manage power densities ranging up to $\mathrm{200\,W/cm^{2}}$ [The10]. Assuming full utilization of Mono3D DRAM internal bandwidth at 30.34 TB/s, each Mono3D DRAM die consumes approximately 104 W. Given the safe temperature for memory and data [Pow21], we conclude the logic die power caps at around 45W per chip.

**Area.** The Mono3D DRAM maintains compatibility with the xPU-DRAM interposer interface utilized by HBM3 [A22a], thereby requiring an HBM3 PHY module. The PHY module’s area overhead, computed for 16 physical channels each supporting 64-bit data I/O at 6.4 Gbps, totals 23.94 mm<sup>2</sup> [A24b, Sca17]. The logic die also has low-voltage Mono3D DRAM peripherals such as DQ buffer, level shifter, and address decoder, occupying 14.80 mm<sup>2</sup>. Power delivery to both Mono3D DRAM and the logic dies involves TSVs extending through the logic die from the interposer. Each TSV with an area of 25 $\mu$m<sup>2</sup> can deliver up to 36 mA [Exp24]. To accommodate peak power of 104 W for the Mono3D DRAM and 45W for the logic processor, the TSVs introduce an area overhead of 0.21 mm<sup>2</sup> when considering a 2:1 redundancy scheme. The logic die matches the Mono3D DRAM die area of 121 mm<sup>2</sup> (i.e., the base die dimensions of HBM3 [A22a]). Thus, the available area budget for the logic die processor is 82 mm<sup>2</sup>.

<span id="section-6-2-3"></span>

#### 6.2.3 logic die processor.

<span id="table-03"></span>

![Table 3. Stratum Logic Die Processor Specification](../../papers/stratum/table-03.png)

**Table 3.** Stratum Logic Die Processor Specification

<span id="figure-15"></span>

![Figure 15. (a) Area breakdown of logic die processor; (b) Power breakdown of Mono3D DRAM-Logic Die at peak performance.](../../papers/stratum/figure-15.png)

**Figure 15.** (a) Area breakdown of logic die processor; (b) Power breakdown of Mono3D DRAM-Logic Die at peak performance.

[Table 3](#table-03) summarizes the specifications of the Stratum logic die processor at the PE, PU, and chip hierarchy levels. We calculated the maximum number of MAC units using [Equation 1](#equation-01), employing a simulated per-MAC-operation energy of $E_{\mathrm{mac}}=0.604$ pJ. The processor achieves a peak performance of 128 TFLOPS with 64k MAC units operating at 1 GHz. The PE tensor core is arranged into a $16 \times 16$ array, providing a balanced matrix tile size to optimize utilization across diverse GeMM sizes. Additionally, a programmable tiering table stores row addresses of the last Mono3D DRAM layer and the tRCD for each tier. The incoming row addresses are compared with eight stored addresses to expedite tRCD lookup. The communication-computation optimizations adopted enable the on-chip ring to require only 128 GB/s bandwidth per link without performance degradation based on the system-level simulation. [Figure 15](#figure-15) presents the area and power breakdown of the Stratum NMP stack. The total area occupied by the active logic is 76.63 mm<sup>2</sup>, which falls within the 121 mm<sup>2</sup> area budget, yielding a utilization of 63%. The area is predominantly consumed by the PEs, which dominate the PU-level area. The tiering table introduces only a minimal overhead of 0.1% of the PE area within each PE. The Stratum NMP stack reaches a peak power of 144.53 W when the fastest Mono3D DRAM tier is accessed concurrently with full tensor core utilization. The total power of the logic die is 42.67 W, including compute, on-chip communication, and logic-die memory access, under the 45W power budget.

<span id="section-6-3"></span>

### 6.3 System Evaluation

<span id="section-6-3-1"></span>

#### 6.3.1 Algorithm Evaluation

<span id="figure-16"></span>

![Figure 16. Evaluation and comparison of system decoding throughput and energy efficiency.](../../papers/stratum/figure-16.png)

**Figure 16.** Evaluation and comparison of system decoding throughput and energy efficiency.

**Model.** Our model is based on DistilBERT [San19] with 67M parameters and designed for multi-topic text classification, supporting sequences of up to 1024 tokens. It features a compact architecture with 6 transformer layers and 12 attention heads, with a hidden dimension of 3072.

**Data.** Our model training involves a customized data mix across 6 topics. The datasets include a 2% split of Pile of Law for legal topic [Pil22], 1 out of 3 splits from atlas converse and INCLUDE for humanity topic [Atl23, Rom24], 5% split of Programming books for CS topic [Pro25], SciQ and ARC-easy for science topic [Cro17, Cla18], GSM8K and MATH for math topic [Cob21, Hen21], Atlas reasoning for logic topic [Atl25]. For the above-mentioned 6-topic configuration, the data encompasses approximately 70 million tokens.

**Training and Evaluation.** To address distribution shifts from standard NLP datasets to diverse real-world prompts, we use a GPT-4o-based data synthesis pipeline. We sample 500 prompts from the Chatbot Arena dataset [Chi24] to reflect natural user styles, then use GPT-4o with a fixed system prompt to rewrite 50% of our training data into a QA format. We use a mix of rewritten and original data to train our topic classifier on a single A100 GPU for 3 epochs of 3 hours each. For evaluation, we use the MMLU test sets [Li23e] and hand-curated 180-example subsets of Chatbot arena conversations dataset [Chi24] with the 6 topics. Our trained classifier achieves 94.5% and 85.0% accuracy on MMLU and Chatbot arena test sets, close to the performance of OpenAI O3-mini-high (96.2%, 91.1%). The inference overhead of the model is less than 10ms with ONNX runtime on a regular laptop CPU. We use OpenAI-O3 LLM-as-a-judge to classify 33,000 real-world queries from LMArena [Chi24], which shows that our six coarse-grained topics cover 93% of queries, confirming the robustness and generality of TopicBERT’s taxonomy.

<span id="section-6-3-2"></span>

#### 6.3.2 System Performance

[Figure 16](#figure-16) shows the normalized decoding throughput and energy efficiency when serving requests with equal input and output length. For Mono3D DRAM designs, we evaluate *no-tiering* and *tiering* approaches. In *no-tiering* design of Mono3D DRAM, Mono3D DRAM is treated as a single tier, therefore, the logic die is limited to operating under the worst memory access latency of the memory die. In *tiering*, Mono3D DRAM is divided into 8 tiers with fine-grained memory latency and data mapping optimizations given tiering. Stratum *tiering* consistently outperforms GPU baselines across all cases, averaging $8.29\times$, $5.39\times$, $6.13\times$, $4.48\times$ better decoding throughput for OLMoE, Mixtral, Qwen2.5, and Llama-4, respectively. Specifically, as decoding length grows, decoding on conventional GPUs with limited memory bandwidth becomes increasingly memory-bound, due to the quadratic complexity of the attention mechanism, explaining the growing gap of Stratum over GPU baselines. Stratum *no-tiering* as well outperforms GPU due to its higher internal bandwidth compared to HBM, even considering the worst-case latency. The internal memory tiering ([Section 3.2](#section-3-2)) and MoE-specific data mapping optimizations ([Section 5.2](#section-5-2)) further improve decoding throughput by averages of $1.45\times$, $1.39\times$, $1.32\times$, $1.34\times$ over *no-tiering* for the 4 models, respectively. Energy-wise, Stratum achieves up to $7.66\times$, $2.74\times$, $3.51\times$, $4.87\times$ better energy efficiency for the same decoding tasks across OLMoE, Mixtral, Qwen2.5, and Llama-4, respectively, due to cheaper memory access. We also extracted data from the previous work Duplex [Dup24] and made conservative scaling to compare with Stratum. Stratum achieves up to $2.9\times$, $2.5\times$, $3.0\times$, $2.2\times$ better throughput and $2.7\times$, $1.9\times$, $2.9\times$, $2.1\times$ energy over Duplex [Dup24] for OLMoE, Mixtral, Qwen2.5, and Llama-4.

<span id="section-6-3-3"></span>

#### 6.3.3 Expert Placement Optimizations

**Effectiveness.** To study the effectiveness of expert placement in the tiered Mono3D DRAM, we scan the hot expert hit rate for Mixtral $8 \times 7$B on Stratum-L as shown in [Figure 17](#figure-17). The hot expert hit rate is defined as the ratio of aggregated hot expert to total expert accesses at the token level. Across decoding lengths, accurate hot expert usage prediction brings $1.32\times$ to $1.51\times$ better throughput over a uniformly distributed expert usage, or equivalently a naively managed tiered memory. The benefit is more noticeable on smaller decoding lengths, as the MLP dominates the decoding latency more. Using our topic prediction model, we achieve 31.6%, 48.5%, and 68.9% aggregated hot expert hit rates when serving Mixtral, OLMoE, and Llama-4.

<span id="figure-17"></span>

![Figure 17. Impact of hot expert hit rates on (a) MLP (MoE layer) latency and (b) overall system throughput for Stratum-L.](../../papers/stratum/figure-17.png)

**Figure 17.** Impact of hot expert hit rates on (a) MLP (MoE layer) latency and (b) overall system throughput for Stratum-L.

<span id="table-04"></span>

![Table 4. Overhead of Expert Swap across Mono3D DRAM Tiers](../../papers/stratum/table-04.png)

**Table 4.** Overhead of Expert Swap across Mono3D DRAM Tiers

**Costs.** The scheduler ([Section 3.1](#section-3-1)) may trigger expert swaps *between batches*. To evaluate the worst-case scenario, we consider 1) short sequences, ${L_{\mathrm{in}}}={L_{\mathrm{out}}}=256$ with batch size one, and 2) consecutive batches assigned to different topics. [Table 4](#table-04) reports the time and energy overheads of expert swaps, which remain well below 1% across all benchmarks. This negligible cost stems from two factors: expert swaps occur within the same bank, avoiding cross-bank movement, and NMP logic includes dedicated row-swap buffers that enables swapping at the high internal Mono3D DRAM tier bandwidth without traversing the DRAM–xPU interface.

<span id="section-6-3-4"></span>

#### 6.3.4 Performance scaling with batch size

<span id="figure-18"></span>

![Figure 18. Impacts of (a) batch size and (b) Mono3D DRAM layers on system-level metrics, evaluated with Llama-4-Scout on Stratum-XL](../../papers/stratum/figure-18.png)

**Figure 18.** Impacts of (a) batch size and (b) Mono3D DRAM layers on system-level metrics, evaluated with Llama-4-Scout on Stratum-XL

[Figure 18](#figure-18)(a) evaluates Stratum’s performance scaling across different query batch sizes using the large-scale Llama-4-Scout [Lla25] benchmark. Batch sizes are chosen to ensure the full model fits within the Mono3D DRAM of Stratum or the HBM of the GPU baseline. Stratum consistently outperforms the GPU baseline across all settings by 4.7–$9.8\times$. However, the relative performance advantage reduces with larger batches, particularly at shorter sequence lengths (e.g., 1024 tokens), due to the GPU die’s higher compute-to-bandwidth ratio and the increased dominance of MoE layers in the overall runtime.

<span id="section-6-3-5"></span>

#### 6.3.5 Performance scaling with Mono3D DRAM layers

[Figure 18](#figure-18)(b) reports Stratum’s performance scaling across different Mono3D DRAM layer configurations. All variants have the same DRAM capacity and use the same NMP logic die processor, and throughput is normalized to the die area of each Mono3D DRAM to ensure a fair, cost-aware comparison. On average, the 1024-layer design achieves $1.21\times$ and $2.96\times$ higher throughput per area than the 256-layer and 64-layer Mono3D DRAM, respectively, demonstrating the cost-efficiency benefits of adopting >1k-layer Mono3D DRAM.

<span id="section-6-3-6"></span>

#### 6.3.6 Tiering mechanism on Mono3D DRAM with less layers.

The proposed tiering mechanism exploits wordline latency variation resulting from vertical stacking in monolithic 3D DRAM. Mono3D DRAM employs the similar fabrication process as 3D NAND Flash, which has already scaled beyond 400 layers [A25]. Thus, we consider a 512-layer configuration by partitioning the original 1024-layer mat into two horizontally connected 512-layer segments while preserving the NMP logic design. Device-level simulations reveal a $1.3\times$ access latency difference between the fastest and slowest tiers. System-level evaluations demonstrate overall (including both MoE and attention layers) performance improvements of 17.7%, 18.3%, and 18.3% under our topic-aware tiering placement at a sequence length of ${L_{\mathrm{in}}}={L_{\mathrm{out}}}=1024$ on LLama-4-Scout [Lla25], Mixtral $8 \times 7$B [Jia24a], and OLMoE-1B-7B [Olm24] benchmarks, respectively. These results validate the efficacy of the proposed tiering strategy across a wide number of Mono3D DRAM layers.

<span id="section-7"></span>

## 7 Related Works

**3D Stackable DRAM.** Monolithic 3D-Stackable DRAM has emerged as a promising alternative to HBM by sequentially fabricating multiple DRAM layers on the same wafer. Unlike HBM, which depends on TSVs and costly die-stacking, Mono3D DRAM employs fine-pitch hybrid bonding for higher internal bandwidth and integration density [Ong23, A23, A22b, A23a, Sig25, Mon25]. Leading Mono3D DRAM technologies include Horizontal 1T1C [Ong23, A23], which reorients and stacks 1T1C DRAM cells, and Gate-Control Thyristors [A22b, A23a], which leverage avalanche mechanisms. Recent work further shows that Mono3D DRAM ’s $\sim$1$\mu$m bonding pitch [Che20a] enables up to $5\times$ denser vertical interconnects than HBM [Exp24].

**Processing In/Near Memory Acceleration for Transformers.** While Processing In/Near Memory (PIM/PNM) has been a long-standing concept, MAT [Mat21a] first applied PIM to Transformer models, targeting a single encoder block with a memory-efficient pipelined sub-sequence flow. TransPIM [Tra22] extends this with a hybrid PIM-PNM architecture for full-model execution. Neupims [Neu24a] and AttAcc [Att24] focus on Decoder-only Transformer models, offloading attention layers in the decoding stage to the PNM on a xPU-PNM hybrid-processing system. Duplex [Dup24] further expanded support to MoE, GQA, and continuous batching with dynamic compute partitioning. However, all these designs rely on 2D DRAM or die-stacked HBM, limiting their effectiveness when applied to Mono3D DRAM-based systems.

<span id="section-8"></span>

## 8 Conclusion

We present Stratum, a novel system–hardware co-design for efficient MoE serving that, for the first time, leverages high-density Mono3D DRAM dies integrated with logic through 3D hybrid bonding, and further connected to GPUs via a 2.5D silicon interposer. This architecture offers a cost-effective and high-throughput alternative to conventional GPU–HBM-based systems. At the hardware level, Stratum introduces in-memory tiering to exploit vertical access latency variations in Mono3D DRAM, and a near-memory processor (NMP) optimized for expert and attention execution. At the system level, we exploit topic-dependent expert activation patterns to classify and map experts across memory tiers and design a topic-aware scheduler guided by a lightweight classifier to meet service-level objectives. Cross-layer evaluations spanning device, circuit, algorithm, and system levels show that Stratum achieves up to $8.29\times$ better decoding throughput and up to $7.66\times$ less energy consumption compared to GPU baselines.

Acknowledgements. This work was supported in part by PRISM and CoCoSys, centers in JUMP 2.0, an SRC program sponsored by DARPA. This research is also supported by National Science Foundation (NSF) grants 2112665, 2112167, 2003279, 2120019, and 2211386.

[+equal-contribution]: Yue Pan and Zihan Xia contributed equally.
