---

title: 'LLMCompass'

createTime: 2026/08/22 00:00:00

permalink: /en/papers/llmcompass/

---

> [Hengrui Zhang](https://dblp.org/pid/228/2533), [August Ning](https://dblp.org/pid/284/6904), [Rohan Prabhakar](https://dblp.org/pid/294/3826), and [David Wentzlaff](https://www.cs.princeton.edu/people/profile/dwentzlaff). Submitted to arXiv on December 5, 2023; current version v1. [A Hardware Evaluation Framework for Large Language Model Inference](https://arxiv.org/abs/2312.03134). [Original PDF](/paper/llmcompass.pdf). [TeX source](https://export.arxiv.org/e-print/2312.03134). The original PDF remains authoritative for exact print layout and bibliography.

## Abstract

The past year has witnessed the increasing popularity of Large Language Models (LLMs). Their unprecedented scale and associated high hardware cost have impeded their broader adoption, calling for efficient hardware designs. With the large hardware needed to simply run LLM inference, evaluating different hardware designs becomes a new bottleneck. This work introduces LLMCompass, a hardware evaluation framework for LLM inference workloads. LLMCompass is fast, accurate, versatile, and able to describe and evaluate different hardware designs. LLMCompass includes a mapper to automatically find performance-optimal mapping and scheduling. It also incorporates an area-based cost model to help architects reason about their design choices. Compared to real-world hardware, LLMCompass’ estimated latency achieves an average 10.4% error rate across various operators with various input sizes and an average 4.1% error rate for LLM inference. With LLMCompass, simulating a 4-NVIDIA A100 GPU node running GPT-3 175B inference can be done within 16 minutes on commodity hardware, including 26,400 rounds of the mapper’s parameter search. With the aid of LLMCompass, this work draws architectural implications and explores new cost-effective hardware designs. By reducing the compute capability or replacing High Bandwidth Memory (HBM) with traditional DRAM, these new designs can achieve as much as 3.41x improvement in performance/cost compared to an NVIDIA A100, making them promising choices for democratizing LLMs. LLMCompass is planned to be fully open-source.

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs), the technology behind OpenAI ChatGPT [Ope22], Github Copilot [Git21], and Google Bard [Bar23], are gaining widespread attention from the whole society. The capability of LLMs is related to their model size [Kap20a, Hof22a], and larger models [Bro20, Cho22b] show impressive abilities [Wei22f] compared to smaller counterparts [Rad19, Dev18], with future models expected to exceed trillions of parameters [Fed22].

This unprecedented scale of LLMs poses challenges to deployment. Serving a GPT-3 (175B parameters) inference requires a minimum of five NVIDIA A100s solely to accommodate the model parameters (in half precision). This substantial hardware cost impedes the broader adoption of LLMs and motivates computer architects to design more cost-effective hardware. We identify three challenges that exist in designing hardware for LLM inference:

**Lack of tools to evaluate hardware designs.** Before diving into writing the RTL code, hardware designers may want to first sketch and compare different design choices. There are many properties we want for such a hardware evaluation tool before writing RTL. ① **Fast and accurate.** Due to the intense compute and memory hardware demand required for LLM inference, this tool needs to be as fast as possible without sacrificing accuracy. ② **Architecturally descriptive.** This tool should be general enough to describe different design choices: If it only applies to a specific architecture, the design space for computer architects will be limited. ③ **Performance-optimal.** The hardware performance is also affected by how the software is programmed (*e.g.*, how to map the workload to the hardware). The evaluation tool should optimize this software domain to fully demonstrate the hardware capability of each design. ④ **Cost-aware.** We also want to know how different hardware design choices affect the hardware cost to reason about cost-performance trade-offs.

Existing tools fail to meet these requirements. Roofline model analysis is fast but not accurate, and cycle-level simulators are accurate but slow. FPGA emulation is accurate and provides area statistics but requires significant engineering effort. To evaluate large-scale hardware designs in the era of LLMs, a new hardware evaluation tool is needed.

<span id="figure-01"></span>

![Figure 1](../../papers/llmcompass/figure-01.png)

**Figure 1.** An Overview of LLMCompass. LLMCompass can aid the hardware design process as a versatile evaluation tool.

**Lack of knowledge on how different hardware design choices affect LLM inference performance.** As an emerging application, the hardware characteristics of LLMs remain to be understood. Besides the large volume of compute and memory requirements, LLMs are also unique in their auto-regressive way of generating tokens. We are interested in exploring whether these properties of LLMs will change common architecture wisdom.

**Lack of cost-effective hardware designs to democratize LLMs.** LLMs are powerful and capable, but are cost-prohibitive to deploy. To serve GPT-3, a DGX A100 compute node can cost over $100,000 USD [Dgx20], with each NVIDIA A100 featuring 54B transistors and 80 GB of High Bandwidth Memory (HBM). This high hardware cost hinders democratizing LLMs.

In this paper, we tackle these challenges and make three main contributions.

**(1) We introduce LLMCompass, a hardware evaluation framework for LLM inference workloads (Sec. [Section 3](#section-3)).** LLMCompass leverages the fact that mainstream ML hardware platforms share many architectural commonalities, allowing us to develop a general hardware description template for them. We also observe LLMs’ computational graphs are composed of dense operators: matrix multiplication, softmax, layer normalization, *etc.*, all of which have a structural and hence predictable compute and memory access pattern. This allows LLMCompass to perform faster, higher-level tile-by-tile (block-by-block) simulations without losing accuracy compared to cycle-accurate simulators. The framework implements a mapper to manually manage the memory hierarchy and find the performance-optimal mapping and schedule scheme for dense workloads. LLMCompass also features a cost and area model based on public parameters to help designers reason about different design choices.

LLMCompass is validated on three commercial hardware designs: NVIDIA A100 [Nvi20], AMD MI210 [Amd21a], and Google TPUv3 [Nor21, Jou20]. Compared to real-world hardware, LLMCompass’ estimated latency achieves 10.4% error rate across various operators with various input sizes and 4.1% error rate for LLM inference. Implemented in Python, LLMCompass is still fast. It takes only 15-16 minutes to simulate a 4-A100 GPU node running GPT-3 175B inference, including 26,400 rounds of the mapper’s parameter search ([Figure 5](#figure-05)(i), tested on one core of Intel Xeon Gold 6242R CPU @ 3.10GHz).

**(2) We leverage LLMCompass to draw architectural implications and explore how hardware design choices affect LLM inference (Sec. [Section 4](#section-4)).** We find that *prefill* and *decoding* pose different hardware requirements. *Prefill* can significantly benefit from more compute capability and buffers, while *decoding* barely gains from these and is more sensitive to memory bandwidth. These insights inspire us to think about new hardware design paradigms.

**(3) We propose two cost-effective hardware designs different from conventional wisdom (Sec. [Section 5](#section-5)).** We find that today’s hardware design paradigms tend to fit massive compute capability and SRAMs in a huge die connected to high-end HBMs. We analyze the LLM inference characteristics and show how current hardware designs are inefficient. ① As LLM inference is mostly IO-bound, HBMs can be used to achieve low latency. However, HBM memory capacity limits the batch size, making it hard to fully utilize the massive compute capability. Based on this observation, we find that 95.3% of the original performance can still be achieved even if we prune the compute capability and buffer size by half. ② Larger batch size can significantly improve throughput as the model parameters are only read once for the whole batch. As memory capacity limits the batch size therefore limiting throughput, we propose to replace HBMs with traditional DRAM. We find that a larger batch size can compensate for the loss in memory bandwidth and can bring a 1.42x improvement in throughput and a 3.41x improvement in performance/cost.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 Large Language Models and Transformers

Large Language Models are variations of Transformer models [Vas23] with a considerable amount of parameters that have been pre-trained on large corpora of data [Min23]. Today’s LLMs can have as much as one trillion parameters [Fed22]. Compared to smaller models, larger models (*e.g.* GPT-3 175B [Bro20]) showcase a remarkable set of capabilities such as emergent abilities [Wei22f] and few-shot learning [Bro20]. This increase in model size and the consequent memory and compute requirements have posed unique challenges for hardware.

<span id="figure-02"></span>

![Figure 2](../../papers/llmcompass/figure-02.png)

**Figure 2.** A Decoder-Only Transformer Layer with Tensor Parallelism. GPT-3 175B [Bro20] consists of a stack of 96 such layers.

<span id="figure-03"></span>

![Figure 3](../../papers/llmcompass/figure-03.png)

**Figure 3.** LLMCompass’ Hardware Description Template. In this example, each device has 2 cores and each core has 2 lanes.

We focus on Decoder-only Transformer models [Phu22], which is the architecture adopted by most of the LLMs today: LLaMA [Oth23], GPTs [Bro20, Rad19], Bloom [Wor23], PaLM [Cho22b], *etc.* The basic building blocks of these models are Transformer layers. As illustrated in [Figure 2](#figure-02), each layer comprises a Multi-Head Attention block followed by an MLP block. These layers are then stacked together, forming the bulk of an LLM’s memory and compute requirement. Transformers also use learned Vocabulary and Position embeddings, but for large models like GPT-3, these do not contribute significantly to either the memory or compute requirement ($<2$%). Without losing generality, we focus on Multi-Head Attention Transformers (GPT-style). There are other variations such as Multi-Query Attention [Cho22b], Mixture-of-Experts [Fed22], and parallel Attention and MLP [Cho22b]. LLMCompass seamlessly supports all these possible variations as they share a common set of operators.

<span id="section-2-2"></span>

### 2.2 LLM Inference

Given an input prompt and the required number of output tokens, LLM inference can be divided into two stages [Pop23]. ① *Prefill*: Processing the input prompt and computing the KV cache. The Key Value (KV) cache refers to the stored Key and Value tensors of the Attention block in each layer [Pop23]. ② *Decoding*: Generating output tokens one after another in an auto-regressive manner: The Key and Value of the newly generated token will be concatenated to the KV cache and used for generating the next token. The latency of *prefill* and *decoding* is mostly determined by the input and output sequence lengths, respectively. In *prefill*, as the entire input sequence needs to be multiplied by all the parameters, it is usually bounded by compute. In *decoding*, each new token needs to be multiplied by all the parameters and concatenated to the KV cache, so *decoding* is usually bounded by reading parameters and KV cache.

Latency and throughput are the key metrics to evaluate LLM inference systems. For interactive use cases such as chatbots [Ope22], it is imperative to optimize latency. For background data processing use cases such as data wrangling [Nar22] or form processing [Che21f], throughput is more important. The tradeoff between latency and throughput is determined by batch size: larger batch increases throughput at the cost of higher latency.

<span id="section-2-3"></span>

### 2.3 Parallelizing LLM Inference

Due to the large volume of compute and memory operations, it is beneficial to parallelize LLM inference across multiple devices. This leads to much better performance and can be necessary if the model’s parameters along with the KV cache do not fit in a single device’s memory. For LLM inference, there are two model parallelization schemes: pipeline parallelism and tensor parallelism. In pipeline parallelism, different layers of the model are grouped into sequential partitions and assigned to different devices like a hardware pipeline. This scheme has the effect of considerably increasing throughput at the expense of increased latency. On the other hand, tensor parallelism, as proposed by Megatron-LM [Sho20], partitions each layer of the model across the available devices, thereby decreasing latency at the cost of frequent device-device communication and synchronization. As shown in [Figure 2](#figure-02), this scheme requires two *all-reduce* for each Transformer layer, one after the Attention block and another after the MLP block.

<span id="section-3"></span>

## 3 LLMCompass

An overview of LLMCompass (**L**arge **L**anguage **M**odel **Com**putation **P**erformance and **A**rea **S**ynthesi**s**) is shown in [Figure 1](#figure-01). To evaluate the performance (*e.g.*, throughput and latency) of running a Transformer-based large language model on a hardware system, two inputs are needed: the computational graph of the LLM and a **hardware description** ([Section 3.1](#section-3-1)). Given the input, the **performance model** ([Section 3.2](#section-3-2)) generates a performance report. The **mapper** conducts a parameter search along with the **architecture simulator** to find the best mapping and scheduling scheme. At the same time, the **area model** ([Section 3.4](#section-3-4)) generates the area and cost report.

<span id="section-3-1"></span>

### 3.1 Hardware Description Template

The hardware description template of LLMCompass is introduced below, as shown in [Figure 3](#figure-03): • A **system** (*e.g.*, a DGX node) is composed of multiple devices connected through a device-device interconnect (*e.g.*, NVLink or Infinity Link). • Each **device** (*e.g.*, a GPU) is composed of multiple cores, a shared global buffer, and an off-chip main memory. The **global buffer** (*e.g.*, L2 cache in NVIDIA GPUs) is connected to the main memory, device-device interconnect, and all the cores. • Each **core** (*e.g.*, a Stream Multiprocessor in NVIDIA GPUs) can have multiple lanes sharing a **local buffer** (*e.g.*, L1 cache in NVIDIA GPUs). The local buffer is connected to the global buffer through the on-chip interconnect. • Each **lane** is independent from each other and has its own **vector unit**, **systolic array**, registers and control logic.

<span id="table-01"></span>

![Table 1](../../papers/llmcompass/table-01.png)

**Table 1.** Examples of LLMCompass’s Hardware Description [+1]

<span id="figure-04"></span>

![Figure 4](../../papers/llmcompass/figure-04.png)

**Figure 4.** Visualization of a Matrix Multiplication in LLMCompass as in [Section 3.2.1](#section-3-2-1).

In existing devices, the local and global buffers are usually on-chip SRAM: cache, scratchpad, or a combination of both. LLMCompass doesn’t distinguish between cache and scratchpad because the memory is explicitly managed by the mapper. We believe this assumption does not lose generality as a highly optimized library will also carefully manage the memory. The main memory is usually off-chip DRAM: HBM, DDR memory, CXL memory, *etc*, all of which can be described by our parameterized hardware description template.

We find this hardware description is general enough to describe the mainstream machine learning platforms of today: NVIDIA GPUs, AMD GPUs, and Google TPUs, as shown in [Table 1](#table-01) with a sample of key specifications listed. It is also flexible enough to explore future architectures.

<span id="section-3-2"></span>

### 3.2 Performance Model

The computational graph of a Transformer is composed of a stack of Transformer layers. Each layer is composed of a series of operators, including matrix multiplication (*Matmul*), *Softmax*, layer normalization (*LayerNorm*), and activation functions (*e.g.*, GELU [Hen16b] as in GPTs [Bro20, Rad19]). In a multi-device setup, communication primitives such as *all-reduce* operators are also needed to perform tensor parallelism. The key challenge is how to simulate the performance of different operators and communication primitives on a given hardware system - this requires knowledge about the hardware and how to map and schedule operators on a multi-level compute system with a multi-level memory hierarchy.

To solve this, LLMCompass introduces a mapper and an architecture simulator to build a performance model. Conceptually, we simulate running an operator on the chosen hardware in a recursive manner: we first partition the problem into smaller sub-problems that can fit in the global buffer. The sub-problem is then divided into smaller sub-sub-problems that can fit in each core’s local buffer. The partitioning, mapping, and scheduling are generated by the mapper and a parameter search is conducted to find the optimal mapping and scheduling. LLMCompass always tries to find the performance-optimal mapping to fully demonstrate the hardware capability.

<span id="section-3-2-1"></span>

#### 3.2.1 Matrix Multiplication

The process of simulating a matrix multiplication is visualized in [Figure 4](#figure-04). $\mathbf{A}$ is a $M\times K$ matrix with $M$ rows and $K$ columns. Similarly, $\mathbf{B}$ and $\mathbf{C}$ are $K\times N$ and $M\times N$ matrices respectively. A generalized matrix multiplication is defined as $\mathbf{C}=\mathbf{AB}+\mathbf{C}$.

**From main memory to global buffer:** To maximize data reuse, matrix multiplication is usually calculated in a tile-by-tile manner [Lam91a]. As shown on the left of [Figure 4](#figure-04), matrix $\mathbf{A}$, $\mathbf{B}$, and $\mathbf{C}$ are divided into tiles small enough to fit into the global buffer. In each step, one $A_{\mathrm{tile}_{m,k}}$, $B_{\mathrm{tile}_{k,n}}$, and $C_{\mathrm{tile}_{m,n}}$ are read into the global buffer, the cores then perform the computation, and the results are written back.

**From global buffer to local buffer:** With tiles inside the global buffer, we now need to parallelize the computation of $C_{\mathrm{tile}_{m,n}}=A_{\mathrm{tile}_{m,k}}B_{\mathrm{tile}_{k,n}}+C_{\mathrm{tile}_{m,n}}$ on multiple cores. As shown in the middle of [Figure 4](#figure-04), these tiles are further divided into smaller sub-tiles to fit in each core’s local buffer. It then becomes a scheduling problem to map sub-tiles onto cores.

The right of [Figure 4](#figure-04) shows two possible schedule schemes: • **Schedule Scheme 1:** Different cores working on different $C_{\mathrm{subtile}}$s in the same column. At *wave 0*, as *core 0* and *core 1* both need to read the same $B_{\mathrm{subtile}}$, their memory access to the global buffer should be merged. In our simulator, this memory access merging is automatically identified and taken care of. As the same core keeps updating the same $C_{\mathrm{subtile}}$, there is no need to first write the partial result and then read it from the global buffer. This *Read-After-Write* dependency is also automatically taken care of by the simulator. • **Schedule Scheme 2:** Different cores working on the same $C_{\mathrm{subtile}}$. *Core 0* and *core 1* first read the data and calculate the partial results, then perform a reduction and write back the final results.

In reality, with more cores and more tiles, the schedule space can be more complicated than the example shown in [Figure 4](#figure-04).

**From local buffer to lanes:** Similarly, within each core, the sub-tiles are further partitioned into sub-sub-tiles to be mapped to lanes sharing a local buffer. After that, the sub-sub-tiles are finally passed to the systolic arrays. LLMCompass leverages SCALE-Sim [Sam18, Sam20], a cycle-level systolic array simulator, to mimic the behavior of a systolic array and get the cycle count. LLCompass caches the results of SCALE-Sim into a look-up table to avoid duplicated simulation. A reduction will be performed by the vector unit if needed.

**Mapper:** A parameter search is performed by the mapper to determine the best tiling scheme and schedule scheme. To overlap computation with memory accesses, we also add software pipelines (double buffering) at each level of the memory hierarchy as scheduling options. The downside of enabling software pipeline is that it requires extra buffer space so the maximal tile size will be reduced, causing potentially lower utilization of systolic arrays. However, we find software pipeline to be beneficial in most cases.

<span id="section-3-2-2"></span>

#### 3.2.2 Communication Primitives

We use the link model as in AHEAD [Abd19] and LogGP [Ale95]. Suppose $L$ is the link latency, $O$ is the additional overhead associated with the data transfer, and $B$ is the link bandwidth. The latency $T$ to transfer $n$ bytes of data through a link is expressed in [Equation 1](#equation-01) and [Equation 2](#equation-02):

<span id="equation-01"></span>

$$
T=L+O+\frac{\hat{n}}{B}
$$

<span id="equation-02"></span>

$$
\hat{n}=\left\lceil\frac{n}{\mathit{MaxPayload}}\right\rceil*{\mathit{\mathit{Flit\_size}}}+n
$$

On top of this, we implement ring all-reduce [Pat09], which is a bandwidth-optimal all-reduce algorithm. We use a 16-byte $\mathit{\mathit{Flit\_size}}$ and a 256-byte $\mathit{MaxPayload}$ based on NVLinks [Fol17]. We don’t model more communication primitives as LLM inference only requires *all-reduce* for tensor parallelism and *peer-to-peer* for pipeline parallelism.

<span id="section-3-2-3"></span>

#### 3.2.3 Other Operators

We also model *Softmax*, *LayerNorm*, and *GELU* following a similar methodology as in [Section 3.2.1](#section-3-2-1). The only differences are: ① They have fewer dimensions and are therefore simpler: *Softmax* and *LayerNorm* operate on two-dimensional data, and *GELU* operates on one-dimensional data, while *Matmul* operates on three-dimensional data. As each dimension requires tiling and scheduling, the mapper search space is much smaller. ② They do not use systolic arrays. *Softmax* is implemented with the online algorithm [Mil18a]. *GELU* is approximated with $\tanh$ [Hen16b].

<span id="figure-05"></span>

![Figure 5](../../papers/llmcompass/figure-05.png)

**Figure 5.** *Matmul with $M=8192$.*

<span id="section-3-3"></span>

### 3.3 Performance Model Validation

In this section, we validate our framework against three real hardware platforms: (1) a datacenter GPU node with 4 NVIDIA A100 SXM4 GPUs (80 GB) fully connected by NVLinks; (2) a Google Cloud TPU node with 8 TPUv3 cores connected in a 2D torus topology; (3) an AMD MI210 GPU [+2]. The results are shown in [Figure 5](#figure-05). For NVIDIA GPUs, CUDA 11.7 and PyTorch 2.0 are used to benchmark operators in half precision (FP16) with `torch.compile` enabled for *LayerNorm* and *GELU* to maximize performance. Communication primitive *all-reduce* is benchmarked with nccl-tests [Ncc21], a communication primitive performance benchmark for NVIDIA GPUs. For Google TPUs, JAX 0.4.18 is used to benchmark operators and communication primitives. Due to the hardware feature of TPUs, *Matmul* is benchmarked in bfloat16 (BF16) and all the other operators are in FP32. For AMD GPU, ROCm 5.4.2 and PyTorch 2.0 [+3] are used along with FP16 for *Matmul* and FP32 for other operators. The kernel launch overhead including the framework overhead is measured by running the operator with an input of size 1.

As shown in [Figure 5](#figure-05), for *Matmul*, *Softmax*, *LayerNorm*, *GELU*, and *all-reduce*, LLMCompass achieves an average error rate of 9.0%, 12.0%, 11.3%, 5.0%, and 14.9% respectively. For LLM inference, LLMCompass achieves an average error rate of 0.69% and 7.5% for *prefill* and *decoding* respectively. **On average, LLMCompass achieves a 10.4% error rate for different operators at various input sizes and a 4.1% error rate across the *prefill* and *decoding* stages.**

Although not a perfect match to real-world hardware, LLMCompass is able to show a similar trend that a naive roofline model fails to show. For example, in [Figure 5](#figure-05)(d), as the reduction dimension of *LayerNorm* increases to an extreme, the throughput should drop due to the increasing reduction cost. LLMCompass is able to catch this trend.

LLMCompass’ results are totally interpretable without incorporating any fudge factor and we believe this interpretability is more important than perfectly matched results. Here are some possible causes of the mismatch between LLMCompass and real hardware: • Lack of hardware knowledge. We have little knowledge about the micro-architecture of GPUs and TPUs (*e.g.*, hardware pipeline design or scheduler design). With a large input size, the hardware is well utilized and some overhead can be hidden. However, with a small input size, it’s hard to hide the overhead and micro-architecture details affect performance significantly. Also, the Tensor Cores in NVIDIA GPUs and Matrix Cores in AMD GPUs are simulated as systolic arrays in LLMCompass, which may not be true in reality. • Lack of software knowledge. We don’t know how operators and communication primitives are implemented on these platforms as they are closed-source libraries. We conduct a thorough parameter search for each input size to maximize performance, but in reality those libraries probably use heuristics to determine mapping and scheduling, which may not be optimal at all input sizes (*e.g.*, we find that for a *Matmul* with $M=64$ and $N=K=12288$, AMD MI210 is less than 25% of its roofline performance while a NVIDIA A100 can achieve 50% of its roofline performance.). Also, some key information is not available. For example, we cannot find the packet format for TPU-TPU communication and have to use the NVLink packet format instead. • Non-ideal hardware. LLMCompass assumes a fixed frequency, but when testing real-world hardware, we have no control over the frequency of the datacenter GPU or TPU nodes. LLMCompass also assumes bandwidth can be utilized at full rate, but in reality there may be some other overhead (*e.g.*, error correction code).

<span id="section-3-4"></span>

### 3.4 Area and Cost Model

As chip designers increase die area to improve single chip performance, fewer chips fit per wafer and may also risk decreased yield, leading to increased costs. LLMCompass incorporates area and cost models to allow designers to reason about these performance-area trade-offs. These models use the provided hardware description with estimated transistor counts and/or die areas from known components to find the total device die area - our methodology is explained as follows.

Within each core’s lanes, we estimate the vector units’ and systolic arrays’ transistor counts from open-source designs, tape-outs, and generators [Zar19, Mck18a, Gen21]. We estimate each lane’s register file’s area overhead using an empirical area model [Rag09]. For the local buffer shared amongst lanes in each core as well as the global buffer shared amongst cores, we model them as SRAM caches and derive their areas using CACTI [Mur09] and scale results down to a 7nm process. For memory and device-device interconnect, we estimate PHY and controller area based on annotated A100 and MI210 die photos [Pat22a, Smi22b]. In our calculations, the controller area scales based on the process node, but the PHY area remains fixed as they do not scale well due to internal analog devices.

<span id="table-02"></span>

![Table 2](../../papers/llmcompass/table-02.png)

**Table 2.** A Sample of Area Model Parameters (7nm)

<span id="figure-06"></span>

![Figure 6](../../papers/llmcompass/figure-06.png)

**Figure 6.** Die Area Breakdown of NVIDIA GA100 and AMD Aldebaran.

We account for extra per lane overheads (*e.g.*, control signals) by calculating the core area using our model and taking the difference from the expected die areas taken from annotated photos. We then divide the overhead per lane, per scheduler width (32 in A100s, 16 in MI210). Similarly, we account for extra per core overheads (*e.g.*, core-to-core crossbars) by calculating the expected die area with our model and splitting the area between the cores. These per-lane and per-core overhead estimates are averaged between AMD and NVIDIA chips.

To estimate cost, LLMCompass uses supply chain modeling [Nin23] for wafer costs to calculate per-die costs. These per-die costs do not incorporate any IP, masks, or packaging costs. For memory costs, we use average DRAM spot prices for DDR [Tre23] and consumer estimates for HBM2e [Lap19].

[Table 2](#table-02) shows a sample of the transistor counts and corresponding 7nm die areas of the parameters used in the area model. Using their respective architecture white papers, we model GA100 [Nvi20] (the die used in NVIDIA A100) and Aldebaran [Amd21a] (the die used in AMD MI210) dies to estimate their total die areas, shown in [Figure 6](#figure-06)(a). For the accounted-for components, LLMCompass’ area model estimates for GA100 and Aldebaran dies achieve a 5.1% and 8.1% error respectively. We attribute these differences to the core’s microarchitecture and core-to-core communication overheads which are proprietary and difficult to estimate. Our model also allows users to break down a single core’s area into its individual components, shown in [Figure 6](#figure-06)(b).

<span id="section-4"></span>

## 4 Architectural Implications

With LLMCompass, we are able to conduct a design space exploration and shed light on how to design efficient hardware systems for LLM inference. In this section, we use LLMCompass to study how different compute system configurations, memory bandwidth, and buffer sizes affect LLM inference performance and draw architectural implications. These insights inspire us to propose new designs as in [Section 5](#section-5).

<span id="section-4-1"></span>

### 4.1 Experimental Setup

For all the unmentioned specifications, we use the specifications of an NVIDIA A100 (as in [Table 1](#table-01)) and 4-way tensor parallelism. *Prefill* latency is measured by running one GPT-3 layer with batch size 8 (a balancing point between latency and throughput) and input sequence length 2048 (a medium-long sequence for GPT-3). *Decoding* latency is measured as the latency of generating the 1024th output token when running one GPT-3 layer with batch size 8 and input sequence length 2048. We use FP16 for all the operators.

<span id="section-4-2"></span>

### 4.2 Compute System

We test five different compute system designs as shown in [Table 3](#table-03). From A to E, we increase each core’s systolic array, vector unit, and local buffer capacities. B represents a full GA100. We keep B, C, D, and E to have the same total compute capability and total buffer size to compare the design choice of fewer big cores or more tiny cores. Configuration A only has a quarter of the compute capability compared to others. All the designs have the same amount of total buffer size and register file size scales with vector width.

[Figure 7](#figure-07) shows *prefill* and *decoding* latencies for these designs. Compared to the GA100, design A has 3.25x higher *prefill* latency but is only 0.1% slower at *decoding* and uses only 57.8% of the area. Design E with the largest cores see *prefill* and *decoding* latency increase by 12.4% and 30.8% respectively, but can reduce die area up to 7.7%.

**Analysis:** For the *prefill* stage, B is much faster than A because *prefill* is compute-bound. As per core systolic arrays and vector units scale, the tile size needs to increase to fully utilize larger computing units. Bigger tiles can cause more padding as the problem size needs to be quantized to the tile size and hardware size. Although large systolic arrays and vector units can be more area-efficient, they are harder to schedule and fully utilize.

Since *decoding* is IO-bound, increasing compute capability barely helps, which explains why A and B have similar performance. As the matrix multiplications during *decoding* are narrow (*e.g.* $16\times{}12288$), it is even harder to fully utilize larger systolic arrays and performance degrades.

**Implications:** ① *Increasing compute capability significantly helps prefill but barely helps decoding.* ② *Large systolic arrays are more efficient for prefill compared to decoding.*

<span id="table-03"></span>

![Table 3](../../papers/llmcompass/table-03.png)

**Table 3.** Five Compute System Designs.

<span id="figure-07"></span>

![Figure 7](../../papers/llmcompass/figure-07.png)

**Figure 7.** *Prefill Latency per GPT-3 Layer.*

<span id="section-4-3"></span>

### 4.3 Main Memory

As main memory capacity is considered more of a constraint (enough capacity is required to hold the parameters and KV cache), we will focus on the impact of main memory bandwidth. [Figure 8](#figure-08) details the performance results for sweeping memory bandwidth from 400 to 3200 GB/s. For *prefill*, increasing memory bandwidth from 800GB/s to 2000GB/s reduces latency by 14.3%, and further increasing to 3200GB/s has a marginal performance gain of 3.5%. For *decoding*, increasing from 800GB/s to 2000GB/s has a speedup of 1.88x, and further increasing to 3200GB/s brings another 26% gain.

<span id="figure-08"></span>

![Figure 8](../../papers/llmcompass/figure-08.png)

**Figure 8.** *Prefill Latency per GPT-3 Layer.*

**Analysis:** In the *prefill* stage, *Matmul*s are significantly faster when increasing memory bandwidth from 400GB/s to 800GB/s. Further increasing bandwidth does not significantly affect *Matmul* performance as it becomes compute-bound. For IO-bound *GELU*, *LayerNorm*, and *Softmax*, larger memory bandwidth realizes significant speedup.

In the *decoding* stage, *Matmul*s are significantly faster with increased memory bandwidth, mainly because they are narrow (turn into a vector-matrix multiplication at batch size 1) and IO-bound. In this stage, *GELU*, *LayerNorm*, and *Softmax* have a small input size. They are dominated by kernel launch overhead and barely affected by memory bandwidth.

③ *Decoding is much more sensitive to memory bandwidth than prefill.*

<span id="section-4-4"></span>

### 4.4 Local and Global Buffer

**Local Buffer**. We fix the hardware specifications to an NVIDIA A100 (as in [Table 1](#table-01)) and sweep local buffer size. The results are shown in [Figure 9](#figure-09). For *prefill*, increasing the local buffer size from 64KB to 192KB improves the performance by 18.0% while increasing the area by 5.8%. Further increasing to 1024KB has a negligible performance gain of only 0.2% at the cost of 28.8% bigger area. For the *decoding* stage, increasing the local buffer size from 64KB to 1024KB only increases the performance by 0.5%.

<span id="figure-09"></span>

![Figure 9](../../papers/llmcompass/figure-09.png)

**Figure 9.** *Prefill Latency per GPT-3 Layer.*

**Analysis:** The reduced *prefill* latency with larger local buffers is mainly because of reduced matrix multiplication latencies. A larger local buffer enables larger matrix tiles and therefore higher systolic array utilization rate. A local buffer size of 192KB is just enough for matrix multiplication of $128\times 128\times 128$ at FP16 with double buffering technique. It can fully utilize the $16\times 16$ systolic arrays, shedding some insight on the NVIDIA A100’s design choices. Increasing local buffer size when the systolic array is already fully utilized leads to marginal performance gains. For *decoding* stage, increasing local buffer size does not help because it’s IO-bound.

**Global Buffer.** The performance trends for global buffer size are similar to [Figure 9](#figure-09). Increasing the global buffer size from 10MB to 40MB speeds up *prefill* by 11.8% while increasing area by 9.6%. Further increasing to 80MB only brings a performance gain of 0.01% at the cost of 11.7% bigger area. For *decoding*, increasing global buffer size from 10MB to 80MB has a performance gain of only 0.7%.

**Analysis:** Larger global buffers enable larger matrix tiles, increasing systolic array utilization and data reuse at the global buffer level. Similarly, increasing global buffer size has diminishing returns once the systolic arrays are saturated. The *decoding* stage is not bounded by computation so it barely benefits from the larger global buffer.

④ *Large buffers help prefill but not decoding.* ⑤ *Buffers should be large enough to fully utilize the systolic arrays.*

<span id="section-5"></span>

## 5 Efficient Hardware Design with LLMCompass

Ideally, efficient hardware design will optimize for both performance and cost. This section draws from the insights in [Section 4](#section-4) and proposes two efficient hardware designs: a latency-oriented design and a throughput-oriented design. Both of these designs aim to reduce hardware costs while maintaining or improving performance. The key specifications are shown in [Table 4](#table-04). All the other specifications (*e.g.*, frequency, register file size, device-device interconnect, kernel launch overhead, and framework overhead *etc.*) are the same as an NVIDIA GA100 for fair comparison.

<span id="table-04"></span>

![Table 4](../../papers/llmcompass/table-04.png)

**Table 4.** Comparison with NVIDIA GA100

<span id="section-5-1"></span>

### 5.1 Latency-Oriented Design

LLM inference latency means the total time between receiving the request and generating the last token. It is a critical metric for interactive use cases like chatbots. It is composed of *prefill* latency, the time to process the input sequence, and *decoding* latency, the time to generate the output sequence in an auto-regression way. Inference latency is usually dominated by *decoding* unless the input sequence is much longer than the output sequence. *Decoding* is IO-intensive and is mostly bounded by reading model parameters and KV cache.

**Observation:** As latency is mostly IO-bound, memory bandwidth is the key to reducing latency, making HBM the best choice. However, due to the capacity limit of HBM, the batch size cannot be too large: the size of the KV cache and intermediate values is proportional to batch size. Therefore, the massive compute capability is not fully utilized.

**Proposal:** We propose an efficient latency-oriented design by pruning half of the compute capability while using the same memory system as a GA100, as shown in the left of [Table 4](#table-04).

**Results:** Compared to an NVIDIA GA100, the die area is reduced by 42.1% while keeping 95.3% of the performance on average. The results are shown in [Figure 10](#figure-10).

<span id="figure-10"></span>

![Figure 10](../../papers/llmcompass/figure-10.png)

**Figure 10.** End-to-End Performance of Latency-Oriented Design Normalized to GA100. Performance metric: inverse of latency (higher is better). Settings: batch size 16 [+5], 4-way tensor parallelism, running 48 GPT-3 layers (half of GPT-3).

**Discussion:** Due to the IO-bound *decoding* stage, the over-provisioned GA100 is not able to realize significantly improved inference performance compared to our latency-oriented design. As shown in [Figure 11](#figure-11), our pruned design achieves identical *decoding* performance as a GA100. The GA100 is an enormous die and is susceptible to yield issues - A100 dies are already binned to have 108 functioning SMs out of 128. Our latency-oriented design shows that even with half the cores and SRAM disabled, the device can still achieve similar performance. This may motivate designers to salvage previously deemed faulty chips and manufacture them into separate products focused on LLM inference.

<span id="figure-11"></span>

![Figure 11](../../papers/llmcompass/figure-11.png)

**Figure 11.** Comparison of *Decoding* Latency per GPT-3 Layer.

Pruning the compute capability only hurts the compute-bound *prefill* performance. As *prefill* is more dominant at long input sequence and short output sequence, the performance degradation will be more visible under these cases, which explains why we only achieve 80% of the GA100 performance at input length 2048 and output length 256. With a smaller input length and larger output length, our pruned latency-aware design can achieve 99% the performance as GA100.

<span id="section-5-2"></span>

### 5.2 Throughput-Oriented Design

For background use cases such as form processing or data wrangling, throughput can be more important than latency. There are generally two ways to improve throughput: • Decrease latency - As latency is mostly IO-bound by reading parameters and KV cache, the best way to improve latency is to further improve memory bandwidth. As HBM is already expensive, this may not be easily achieved without increasing cost. • Increase batch size - Generally, larger batch sizes are more efficient for throughput because the parameters are only read once for the whole batch. Larger batch sizes can also improve the hardware utilization rate. The downside is that a larger batch size consumes more compute power and increases KV cache accesses.

**Observation:** Increasing batch size is a more efficient way to improve throughput compared to decrease latency, which requires expensive high-end HBMs or even SRAMs. With a larger batch size, more memory capacity is needed to hold the larger KV cache and intermediate values.

**Proposal:** We propose a throughput-oriented design as shown in the right of [Table 4](#table-04). To hold larger batches, we use 512GB of DRAM powered by 256 PCIe 5.0 channels with an aggregated memory bandwidth of 1TB/s. (According to our area model, an 800$mm^{2}$ die’s perimeter is able to fit around 400 PCIe 5.0 channels.) Considering the high cost and limited capacity of HBMs, this design is more cost-effective. With larger batch sizes comes a greater need for compute capability, so we quadruple the systolic arrays and the local buffer. We halve the core count and vector unit to maintain a similar die area as GA100.

<span id="figure-12"></span>

![Figure 12](../../papers/llmcompass/figure-12.png)

**Figure 12.** Throughput of Throughput-Oriented Design (Tokens/s).

**Results:** Compared to an NVIDIA GA100, the die area is slightly smaller and the throughput is improved by 1.42x on average. The results are shown in [Figure 12](#figure-12). By replacing HBMs with traditional DRAMs, the cost is reduced by 58.3%, making a total of 3.41x gain in performance/cost.

**Discussion:** Our design has 6.4x the memory capacity of a GA100, which allows more than 12x bigger batch size after subtracting the fixed space occupied by model parameters. Ideally, with half the bandwidth of a GA100, this configuration can achieve more than 6x improvement in throughput. However, batching only reduces model parameter accesses but not KV cache reads. With a much larger batch, KV cache accesses become the new bottleneck, which diminishes the benefits of batching. As input length and output length increase, throughput decreases due to longer KV cache reads, as shown in [Figure 12](#figure-12)(a).

From a latency perspective, this throughput-oriented design may not be promising: the latency is 9.21x worse than GA100 on average. While model parameters are only read once for each batch, a larger batch size means more KV cache and intermediate values to read. In LLM inference, there is no free lunch between latency and throughput.

<span id="section-6"></span>

## 6 Related Work

<span id="section-6-1"></span>

### 6.1 Evaluating Large-scale Hardware Design

Evaluating the various characteristics of a hardware design, including performance, area, and cost, is extremely useful for hardware designers. To this end, the options are as follows:

**Roofline Model Analysis** [Wil09]. Roofline models are analytical, fast to evaluate, and can be applied to various architectures for performance comparison. However, they can be overly optimistic relative to actual hardware capabilities.

**Cycle-level Simulation** [Bak09, Bec15, Gut18, Sun19b, Ger18, Kim12, Gon17, Uba12, Kha20a, Par19, Sam18, Sam20]. With a typical simulation rate of less than 100K instructions per second, cycle-level simulators become infeasible for evaluating LLM scale workloads. As these simulators are often designed for specific architectures, it is hard to describe a hardware design very different from its design purpose (*e.g.*, it’s almost impossible to use GPGPU-sim [Bak09] to evaluate a TPU-like design because it relies on the GPU ISA). These simulators often require the user to provide the program for evaluation. If the software program is not optimized, it may lead to unfair comparisons.

**FPGA Emulation.** Another way is to implement the design in RTL code and emulate it on FPGAs. The RTL code can be either handwritten or generated by accelerator generators [Ven19, Nvd18, Tin23, Gen21]. Although the emulation is fast, the synthesis process may take a long time for a large design capable of running large language models, and we may need multiple FPGAs to hold the design. Additionally, users need to rewrite the RTL code and redo the synthesis to evaluate a new design. For each new design, users are responsible for mapping their workloads efficiently to fully utilize the hardware.

**Comparison.** As shown in [Table 5](#table-05), LLMCompass is more accurate than roofline model analysis, faster and more versatile than cycle-level simulators, and less engineering-intensive than FPGA emulation. For evaluating large-scale hardware design in the era of LLMs, we believe LLMCompass is helpful for the initial design stage to determine high-level hardware characteristics (*e.g.*, number of cores, memory bandwidth, *etc*.). While this work describes LLMCompass in the context of large Transformer models, it can also be applied to other dense neural network models.

**LLMCompass can complement FPGA emulation.** Designers can perform initial design space exploration before incurring the heavy costs associated with FPGA emulation and the necessary RTL implementation of the proposed design.

<span id="section-6-2"></span>

### 6.2 Accelerator Design Space Exploration

Since the era of CNN, various works have focused on exploring optimal hardware designs as well as mapping [Par19, Dav19, Dav20, Lu17, Ven19, Yan20c, Heg21, Li21d, Rea17, Zha22d]. LLMCompass is different from these works in design considerations and emphasis: ① Mainly targeting Convolutional Neural Networks (CNNs), these works focus on loop parallelization, loop order, and data flows (*e.g.*, weight stationary or output stationary), which are not the primary design considerations in Transformer-based LLMs. LLMCompass is more tailored for matrix multiplication tiling and scheduling as well as other Transformer operators such as *LayerNorm*. ② LLMCompass is designed for GPU-scale designs, which are much larger than CNN accelerators like Eyeriss [Che16h]. LLM workloads are also significantly larger than CNN workloads.

**LLMCompass can also complement design space explorations.** Implemented as a Python library, LLMCompass can be seamlessly integrated into design space exploration frameworks such as FAST [Zha22d]. FAST uses an internal TPU performance simulator, limiting its broader utility. Fast and accurate, we believe the fully open-source LLMCompass can democratize hardware design space exploration research.

<span id="table-05"></span>

![Table 5](../../papers/llmcompass/table-05.png)

**Table 5.** Comparison of Hardware Evaluation Methods

<span id="section-6-3"></span>

### 6.3 Accelerating LLM Inference

Many Transformer accelerators have been proposed [Tam21, Wan22j, Wan21f, Ham21], mainly focusing on accelerating the Transformer with hardware-software co-design such as pruning or approximate-computing. Whether these techniques are effective for the largest of models remains to be seen. Additionally, the major challenge of LLMs today comes from the massive scale of the models, which is the main scope of this paper.

Many efforts have also been made to accelerate LLM inference at the software domain [Ami22a, Pop23, Dao22f, Dao23c, She23d]. LLMCompass is compatible with these optimization techniques by modeling their compute and memory access patterns. We don’t discuss techniques like FlashAttention [Dao22f] because they are orthogonal to the focus of this paper: They focus on the software domain and are usually implemented on a specific hardware platform such as NVIDIA GPUs.

<span id="section-7"></span>

## 7 Conclusion

This work introduces LLMCompass, a fast, accurate, and architecturally descriptive hardware evaluation framework for LLM inference workloads. LLMCompass’ hardware description template, mapper, and architectural simulator allow hardware designers to evaluate large-scale chip designs for LLMs, which are infeasible for cycle-level simulators. The incorporated area and cost models can also help designers reason about performance-cost trade-offs. With the aid of LLMCompass, we draw implications on how hardware designs affect LLM inference. Based on these findings, we propose a latency-oriented design and a throughput-oriented design that achieve 1.06x and 3.41x performance per cost improvements respectively, compared to NVIDIA GA100. We plan to extend LLMCompass to support more machine learning workloads as well as LLM fine-tuning in the future.

## Acknowledgements

We would like to thank Qixuan (Maki) Yu, Zhongming Yu, Haiyue Ma, Christopher Batten, and the entire Princeton Parallel Group, for their feedback, suggestions, and encouragement. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-2039656, the National Science Foundation under Grant No. CCF-1822949, Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) under agreement No. FA8650-18-2-7862. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation. The U.S. Government is authorized to reproduce and distribute reprints for Governmental purposes notwithstanding any copyright notation thereon. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies or endorsements, either expressed or implied, of Air Force Research Laboratory (AFRL) and Defense Advanced Research Projects Agency (DARPA) or the U.S. Government.

[+1]: One TPUv3 core. Each TPUv3 chip has two TPUv3 cores. TPUv3 cores within the same chip are connected by internal links.

[+2]: We set the frequency to 1400 Mhz to avoid frequency fluctuation

[+3]: We encountered “load binary” error so we didn’t benchmark *LayerNorm* on AMD MI210.

[+5]: In reality, a batch size of 16 with input length 2048 and output length 2048 will slightly exceed the memory capacity.
