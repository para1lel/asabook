---
title: 'MegaScale-MoE'
createTime: 2026/08/22 13:02:12
permalink: /en/papers/megascale-moe/
---

> [Chao Jin](https://dblp.org/pid/19/4764-7) [+equal], [Ziheng Jiang](https://dblp.org/pid/14/8980) [+equal], [Zhihao Bai](https://dblp.org/pid/234/8717), [Zheng Zhong](https://dblp.org/pid/69/7279), [Juncai Liu](https://dblp.org/pid/304/3355), [Xiang Li](https://dblp.org/pid/40/1491-67), [Ningxin Zheng](https://dblp.org/pid/234/5381), [Xi Wang](https://dblp.org/pid/08/5760), [Cong Xie](https://dblp.org/pid/130/0102), [Qi Huang](https://dblp.org/pid/46/4397-1), [Wen Heng](https://dblp.org/pid/201/7460), [Yiyuan Ma](https://dblp.org/pid/234/3589), [Wenlei Bao](https://dblp.org/pid/162/4919), [Size Zheng](https://dblp.org/pid/254/6617-1), [Yanghua Peng](https://dblp.org/pid/195/5934), [Haibin Lin](https://dblp.org/pid/142/1829), [Xuanzhe Liu](https://dblp.org/pid/08/2161), [Xin Jin](https://dblp.org/pid/68/3340-8), and [Xin Liu](https://dblp.org/pid/76/1820-86). First submitted to arXiv on May 16, 2025; current version v3. Accepted to EuroSys '26; [DOI 10.1145/3767295.3769325](https://doi.org/10.1145/3767295.3769325). [MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432). [Original PDF](/paper/megascale-moe.pdf). [TeX source](https://export.arxiv.org/e-print/2505.11432v3). The original PDF remains authoritative for the exact print layout and bibliography.

[+equal]: Equal contribution.

## Abstract

We present MegaScale-MoE, a production system tailored for the efficient training of large-scale mixture-of-experts (MoE) models. MoE emerges as a promising architecture to scale large language models (LLMs) to unprecedented sizes, thereby enhancing model performance. However, existing MoE training systems experience a degradation in training efficiency, exacerbated by the escalating scale of MoE models and the continuous evolution of hardware.

Recognizing the pivotal role of efficient communication in enhancing MoE training, MegaScale-MoE customizes communication-efficient parallelism strategies for attention and FFNs in each MoE layer and adopts a holistic approach to overlap communication with computation at both inter- and intra-operator levels. Additionally, MegaScale-MoE applies communication compression with adjusted communication patterns to lower precision, further improving training efficiency. When training a 352B MoE model on 1,440 NVIDIA Hopper GPUs, MegaScale-MoE achieves a training throughput of 1.41M tokens/s, improving the efficiency by $1.88\times$ compared to Megatron-LM. We share our operational experience in accelerating MoE training and hope that by offering our insights in system design, this work will motivate future research in MoE systems.

<span id="section-1"></span>

## 1 Introduction

As the size of Large Language Models (LLMs) [Cho22b, Tou23a, Jia24a] grow, so does the scale of their training regimes. The escalation in training scale has made efficiency improvements not just desirable but crucial [Jia24f]. As a company building AI products for billions of users, we remain committed to training LLMs with hundreds of billions of parameters on thousands of GPUs. Consequently, even marginal gains in training efficiency can significantly reduce computational resource consumption and training time, directly influencing the feasibility and sustainability of developing state-of-the-art LLMs.

Within the landscape of LLM architectures, Mixture-of-Experts (MoE) models stand out for their sparse activation [Cho22b, Jia24a, Fed22, Sha17], which dynamically routes input tokens to a selected set of specialized network components, known as *experts*, rather than to all parameters. This design leads to sub-linear scaling of FLOPs required as the model size increases, thereby significantly reducing the computational cost. Recent industrial advancements [Du22, Raj22, Dbr25, Xai24, Dee24a] have demonstrated the potential of MoE models, achieving an order-of-magnitude reduction in training cost compared to dense models with equivalent model quality.

Despite the lower training costs of MoE models, we observe a critical performance bottleneck during training from a systems perspective—communication. For instance, when training an internal model on NVIDIA Hopper GPUs, communication accounts for 43.6% of the total time during the forward pass and 32% over the entire training process. Two primary factors contribute to this bottleneck. First, MoE models inherently introduce more communication overhead. Compared to dense model training, MoE model training requires distribution across more GPUs for model parallelism due to its larger parameter size. Second, enabling sparse computation requires two extra all-to-all communications in both the forward and backward passes to dispatch and aggregate tokens, respectively, which hinders ongoing computation.

Moreover, as hardware advances, the imbalance between computation and communication becomes increasingly pronounced, with communication overhead growing more dominant. Alongside improvements in model architectures, hardware capabilities have evolved rapidly, with GPUs achieving significantly higher processing speeds ([Figure 1](#figure-01)). Concurrently, reductions in training precision have been adopted to enhance efficient and cost-effective training [Pen23e, Dee24a]. These trends lead to a scenario where the raw computation time decreases, making the relative impact of communication overhead a more critical bottleneck. For instance, simply extending existing tensor parallelism to multi-node setups has been observed to push communication overhead beyond 50% in certain cases. As a result, optimizing communication is essential for sustaining and improving the scalability of MoE model training, particularly in distributed environments where frequent data synchronization across multiple GPUs is required.

<span id="figure-01"></span>

![Figure 1. Evolution of NVIDIA GPUs.](../../papers/megascale-moe/figure-01.png)

**Figure 1.** Evolution of NVIDIA GPUs.

In this paper, we present the design, implementation, and operational experience of MegaScale-MoE, a production system optimized for efficient large-scale MoE training. By meticulously addressing the communication bottleneck, MegaScale-MoE strives to push the boundaries of MoE training, achieving significant improvements in performance and efficiency. Based on the insight that the key architectural distinctions between MoE and dense models are intra-layer, which is the primary source of the communication overhead, MegaScale-MoE confines each MoE layer to within a single node, utilizing high-bandwidth NVLink. Our analysis ([Section 3](#section-3)) and evaluation ([Section 6](#section-6)) show that despite the cross-node expert parallelism common in existing systems [Dee24a, Hwa23], our approach effectively scales MoE training to models of several hundred billion parameters on thousands of GPUs.

Specifically, MegaScale-MoE addresses the communication problem in MoE training from three key aspects. First, MegaScale-MoE reduces the communication volume by customizing parallelism strategies for the attention and FFN modules in each MoE layer. We compare the parallelism strategies in existing LLM training frameworks, comprehensively considering their impact on large-scale training, including the communication volume and whether communication can be effectively overlapped (i.e., whether it lies on the critical path). Based on this analysis, we select the optimal combination of parallelism strategies for MoE training.

Second, MegaScale-MoE fully overlaps communication with computation at the operator level. MegaScale-MoE partitions the forward and backward passes of each MoE layer into distinct computation and communication operators. For inter-operator overlap, MegaScale-MoE employs a holistic scheduling strategy that carefully reorders communication and computation operators during both forward and backward propagation, hiding communication within independent computations. This approach also optimizes GPU memory usage. MegaScale-MoE utilizes selective activation rematerialization, retaining only a subset of activations in GPU memory during the forward pass, and recomputing or re-communicating to obtain the required activations during the backward pass. With this holistic scheduling, MegaScale-MoE effectively hides the rematerialization overhead, achieving comparable performance while storing only half of the activations.

To overlap communication on the critical paths, MegaScale-MoE employs a fine-grained approach that splits communication into tiles and aligns with the GPU compute pattern, fusing these tile-level communications into the compute kernels. For MoE models with token dispatch, MegaScale-MoE fuses an efficient local scatter operation into the kernel and reorganizes the computation tasks along the scattered dimension to mitigate communication bottlenecks from multiple data sources. This fine-grained overlap occurs within each node, leveraging the high-bandwidth connectivity between GPUs.

Third, MegaScale-MoE leverages communication compression to further enhance MoE training efficiency. Specifically, for widely-used BF16 mixed-precision training, MegaScale-MoE reduces the inter-node parameter synchronization precision from FP32 to BF16, halving the associated overhead. In FP8 training, MegaScale-MoE replaces BF16 reduce-scatter with FP8 communication, incorporating tailored quantization strategies and FP32 reduction to decrease communication volume while preserving convergence stability.

MegaScale-MoE is deployed in our datacenters to train MoE models for our products. Compared to the state-of-the-art open-source LLM training framework, Megatron-LM [Sho19], MegaScale-MoE achieves up to $1.88\times$ higher MFU (Model FLOPs Utilization) when training a 352B MoE model on 1,440 NVIDIA Hopper GPUs. With comprehensive communication optimizations, MegaScale-MoE powers large-scale training in our production, efficiently scaling to trillions of parameters and thousands of GPUs while saving millions of GPU hours.

<span id="section-2"></span>

## 2 Background

<span id="figure-02"></span>

![Figure 2. Mixture-of-Experts (MoE) layer.](../../papers/megascale-moe/figure-02.png)

**Figure 2.** Mixture-of-Experts (MoE) layer.

<span id="section-2-1"></span>

### 2.1 Mixture-of-Experts for Transformer

The Mixture of Experts (MoE) mechanism is an advanced approach designed to boost the performance of Transformer [Vas17] models, which are increasingly pivotal in the realm of LLMs [Jia24a, Cho22b, Dbr25, Dee24a]. It extends the Transformer architecture by integrating multiple expert networks within the feed-forward network (FFN) component. As illustrated in [Figure 2](#figure-02), MoE models dynamically route input tokens to the most relevant experts based on their characteristics. This routing is managed by a trainable gating mechanism that selects the best-suited experts for each token. This architectural innovation enables MoE models to scale in capacity without a proportional increase in inference costs, as only a subset of experts is activated for each input.

<span id="section-2-2"></span>

### 2.2 Large-scale LLM Training

Training large language models at scale on tens of thousands of GPUs is a complex system engineering challenge that requires multiple systems techniques. To distribute the training workload, a combination of parallelism strategies such as data, tensor, and pipeline parallelism is necessary [Sho19, Ras20, Jia24f], as each approach has limitations that prevent relying on a single method for effective scaling.

**Data parallelism** uniformly distributes the training data across all devices, with each device replicating the model parameters and optimizer states. To synchronize the parameters after each training iteration, data parallelism performs an all-reduce communication operation. Zero Redundancy Optimizer (ZeRO) [Raj20] improves over data parallelism by distributing model states across all participating devices. ZeRO unfolds across three progressive stages, each designed to increasingly conserve memory, though this comes with the trade-off of elevated communication.

**Tensor parallelism** distributes compute-intensive tensor operations over multiple devices, enabling parallel computation and significantly accelerating the training process. The specific partitioning strategy and the dependencies among operators within the model dictate that tensor parallelism may necessitate gathering split inputs (all-gather) or merging outputs (reduce-scatter). In LLM training, operators like LayerNorm and Dropout, though less compute-intensive, require substantial activation memory. To tackle this problem, a variant of tensor parallelism known as **sequence parallelism** [Kor22] is proposed, which partitions these operators along the dimension of sequence length. For long-context training, several works [Sho19, Sam23, Con25a] apply sequence parallelism or tensor parallelism to different operators in self-attention. [Figure 3](#figure-03) illustrates the mainstream parallelism strategies for attention, namely tensor, sequence, and context parallelism (TP, SP, and CP), which we analyze in [Section 3.1](#section-3-1).

<span id="figure-03"></span>

![Figure 3. Different parallelism strategies for self-attention. "TP" denotes partitioning along the dimension of hidden size, while "SP" denotes partitioning along the dimension of sequence length.](../../papers/megascale-moe/figure-03.png)

**Figure 3.** Different parallelism strategies for self-attention. "TP" denotes partitioning along the dimension of hidden size, while "SP" denotes partitioning along the dimension of sequence length.

**Pipeline parallelism** enhances efficiency by dividing model layers into stages that are processed on different devices, enabling pipelined execution. Each batch is split into several micro-batches for this purpose. To minimize pipeline bubbles, various scheduling strategies have been developed, e.g., GPipe [Hua19], PipeDream 1F1B [Nar19] and Interleaved 1F1B [Nar21], etc. Megatron-LM adopts Interleaved 1F1B pipeline scheduling, further dividing each stage on one device into multiple virtual stages to reduce the pipeline bubble rate.

**Expert parallelism** is tailored for training MoE models by distributing experts across multiple devices, alleviating memory pressure and enabling parallel processing. To efficiently assign tokens to the appropriate experts and retrieve their outputs, all-to-all communication is typically employed.

<span id="section-3"></span>

## 3 Communication-Efficient Parallelism

With the rise of MoE models and the evolution of hardware compute capabilities, communication overhead has become increasingly critical in MoE training in production. In this section, we delve into the parallelism strategies employed to reduce communication volume and meet other training requirements, such as high GEMM (General Matrix Multiplication) efficiency.

<span id="figure-04"></span>

![Figure 4. Design space for large-scale MoE training.](../../papers/megascale-moe/figure-04.png)

**Figure 4.** Design space for large-scale MoE training.

[Figure 4](#figure-04) shows the design space of parallelism strategies for large-scale MoE training, excluding the outermost data parallelism. We start with inter-node parallelism. Expert parallelism alleviates memory pressure from MoE models’ large parameter size by distributing experts across nodes but incurs per-layer cross-node communication, harming training efficiency. Similarly, tensor parallelism’s high communication overhead makes it more efficient to limit TP to a single node. Following prior work [Jia24f], we adopt pipeline parallelism to distribute model parameters, reduce communication, and overlap communication of different micro-batches.

Prior large-scale MoE training systems, such as Megatron-LM [Sho19] and DeepSpeed-MoE [Raj22], incorporate tensor parallelism to scale up training by partitioning the model parameters within the node. However, in our practice, we observe two issues with this approach: (1) TP partitions the expert dimension, which negatively impacts GEMM efficiency; and (2) TP introduces significant communication overhead, which remains constant as the parallelism size increases, eventually causing communication to exceed computation on modern hardware.

To address these issues, we tailor parallelism strategies for MoE model components. For feed-forward networks (i.e., experts), we replace tensor parallelism with expert parallelism and use custom communication modes optimized for varying top-k and expert sizes, ensuring communication overhead stays lower than tensor parallelism. For other components, we apply sequence parallelism, partitioning along the sequence dimension instead of the batch dimension, allowing scaling without increasing global batch size. This also reduces communication on critical paths compared to tensor parallelism. The additional memory and DP communication overhead remain manageable due to the parameter asymmetry across components. We detail the rationale and analysis of this intra-node parallelism strategy in the following sections. [Table 1](#table-01) lists the key symbols.

<span id="section-3-1"></span>

### 3.1 Sequence Parallelism for Attention

Due to the inherent parallelizability of the expert components in MoE models, most prior work on MoE training [Raj22, Li23i] focuses on optimizing expert parallelism, while data parallelism (DP) is typically applied to the non-MoE components such as attention. However, when scaling up MoE training, this approach proves insufficient due to the $n\times$ activation memory consumption. This issue arises because DP splits the batch dimension both across and within nodes. Compared to other intra-node parallelism strategies shown in [Figure 4](#figure-04), applying DP to attention forces each GPU within a node to process one micro-batch simultaneously, increasing the activation size by $8\times$, which often results in out-of-memory issues.

<span id="table-01"></span>

![Table 1. Description of symbols.](../../papers/megascale-moe/table-01.png)

**Table 1.** Description of symbols.

To enable scalable MoE training, implementing intra-node parallelism for the attention module is crucial. Tensor parallelism (TP) is commonly employed to parallelize attention operations within nodes. However, it introduces inevitable communication costs due to all-gathering and reduce-scattering activations along the critical path. With the increasing gap between computational FLOPs and communication bandwidth, we find that the TP communication overhead can even surpass the computation time of self-attention. This communication-dominated bottleneck limits the ability to overlap communication and computation, ultimately reducing training efficiency.

We adopt sequence parallelism (SP), as proposed in DeepSpeed-Ulysses [Sam23], to scale MoE training and effectively reduce communication along the critical path. SP is commonly used in long-context training to address memory challenges associated with long inputs. We find it also works well in large-scale MoE training. First, it significantly reduces communication overhead compared to TP, especially when using grouped-query attention [Ain23a]. Second, while it introduces some parameter redundancy and increased communication overhead during parameter synchronization, the unique characteristics of MoE models make these trade-offs manageable and acceptable.

**Communication efficiency.** When utilizing TP, the communication volume in attention is

<span id="equation-01"></span>

$$
2bsh(n-1)/n.
$$

With SP, the communication volume decreases to

<span id="equation-02"></span>

$$
2bsh(n-1)/n\times(2+2/m)/n,
$$

where $m$ represents the ratio between the number of query heads and that of key-value heads. Assuming the model is trained on an NVIDIA Hopper GPU workstation with an NVLink domain of size 8, the communication latency for sequence parallel attention can be reduced to about one-fourth of that required by tensor-parallel attention.

<span id="figure-05"></span>

![Figure 5. Hierarchical communication for parameter synchronization in SP attention.](../../papers/megascale-moe/figure-05.png)

**Figure 5.** Hierarchical communication for parameter synchronization in SP attention.

**Data communication & memory overhead.** A notable difference between SP and TP attention is how parameters are distributed across devices: TP shards the attention weights, while SP replicates them. This raises the concern about the potential increase in communication overhead for synchronizing gradients and parameters. Counterintuitively, given the intra- and inter-node bandwidth asymmetry and the adoption of hierarchical communication operations in modern communication libraries [Ncc21] as shown in [Figure 5](#figure-05) and analyzed in [Section 10.1](#section-10-1), although SP attention requires synchronization of $n\times$ more parameters compared to TP attention, the difference in communication overhead is minimal in practical scenarios.

On the other hand, the additional GPU memory consumption introduced by SP attention is minimal in MoE training. For large-scale MoE models with tens to hundreds of experts, the majority of GPU memory is consumed by the expert parameters. Our experiments, detailed in [Section 6.2](#section-6-2), confirm that the extra parameter synchronization and memory overhead of SP attention remain manageable.

**Balanced vs. imbalanced.** In addition to the Ulysses-style SP attention, we also explored other forms, including context parallelism (CP) [Con25a], which partitions all activations along the sequence dimension. CP attention, however, faces workload imbalance due to causal masking in attention, as each token only attends to previous tokens. To mitigate this, we attempted the zigzag strategy by grouping the head and tail partitions of the sequence on the same GPU, although achieving perfect balance remains challenging. Consequently, in large-scale training, the entire training process is often constrained by the most imbalanced data batch. Moreover, this imbalance disturbs the training pipeline, thereby reducing overall training efficiency.

<span id="section-3-2"></span>

### 3.2 Expert Parallelism for Feed-forward Network

<span id="figure-06"></span>

![Figure 6. Communication-efficient expert parallelism. $e$ represents the number of tokens routed to the worker.](../../papers/megascale-moe/figure-06.png)

**Figure 6.** Communication-efficient expert parallelism. $e$ represents the number of tokens routed to the worker.

In the choice of parallelism strategies for the feed-forward network component, expert parallelism (EP) consistently outperforms tensor parallelism. TP partitions the hidden dimension of each expert, reducing GEMM efficiency, whereas EP maintains full expert computation on each device. Theoretically, the communication cost for EP is

<span id="equation-03"></span>

$$
2k/n\times bsh(n-1)/n,
$$

while for TP it is

<span id="equation-04"></span>

$$
2bsh(n-1)/n.
$$

Although their relative efficiency depends on the ratio $k/n$, we design an adaptive communication strategy for different top-$k$ values to minimize the communication volume of EP.

<span id="figure-07"></span>

![Figure 7. Comparison of AG, RS, and A2A for token dispatch.](../../papers/megascale-moe/figure-07.png)

**Figure 7.** Comparison of AG, RS, and A2A for token dispatch.

**Efficient communication pattern.** [Figure 6](#figure-06) compares the typical EP implementation with MegaScale-MoE’s approach. The standard EP implementation requires two all-to-all communications for token dispatch and aggregation. Additionally, a scatter operation may be required before sending and after receiving tokens to ensure that tokens assigned to the same expert reside in a contiguous memory space.

When the top-$k$ value exceeds $n$, we replace traditional all-to-all communication with all-gather and reduce-scatter. First, an all-gather operation collects tokens from all workers. Then, a local scatter operation discards unneeded tokens, retaining only those required by the experts on the current worker. After expert computation, the tokens are assembled into a complete tensor. This approach enables a gather operation before communication, followed by a reduce-scatter to produce the final result, ensuring that EP’s communication overhead remains equal to or lower than TP’s.

In practical training, all-to-all communication is less efficient than all-gather and reduce-scatter, as it requires each worker to communicate with all others, whereas all-gather and reduce-scatter follow a ring-based communication pattern with only neighboring workers. As shown in [Figure 7](#figure-07), the communication time for these three operations in Mixtral-$8\times$7B reveals that when top-$k$ > 6, the all-gather-based EP implementation is more efficient.

**Efficient operators.** Instead of using `torch.scatter_add` and `torch.gather` for tensor scattering and gathering like Megatron-LM, we develop efficient scatter and gather operators directly using CUDA. Based on the token routing results, we pre-calculate the mapping from each row of the input tensor (representing a token) to the corresponding row in the output tensor. The scatter and gather operators then perform data transfers efficiently according to this mapping.

**Load balance.** A well-known challenge in MoE model training is load balancing across experts [Li23i, Dee24d]. To address this, we use auxiliary loss and token dropping to balance the workload across GPUs within each node. Similar to DeepSeek-V2 [Dee24d], we treat the experts placed on the same GPU as a group and calculate the balance loss and computational capacity for each device rather than for each individual expert.

<span id="figure-08"></span>

![Figure 8. Selective activation rematerialization.](../../papers/megascale-moe/figure-08.png)

**Figure 8.** Selective activation rematerialization.

<span id="section-4"></span>

## 4 Communication-computation Overlap

After optimizing parallelism strategies to minimize communication volume, we further reduce the communication overhead to nearly zero using comprehensive communication-computation overlapping techniques. Training large models involves integrating various techniques, which increases the complexity of communication overlap. For instance, at any given moment, the device might concurrently handle computation and communication kernels, overlap PP and DP communications, and manage data transfers between the device and host. Existing frameworks like Megatron-LM assemble attention and FFN modules into MoE layers and rely on the `torch.autograd` package for backward propagation, which limits the flexibility of communication overlap. In contrast, MegaScale-MoE decomposes the attention and FFN modules of each MoE layer into operators that run as GPU kernels, enabling fine-grained communication overlap through flexible scheduling.

<span id="section-4-1"></span>

### 4.1 Inter-operator Overlap

We overlap communication operators with independent computation operators by executing them asynchronously on different CUDA streams. To achieve optimal performance during the training process, we adopt a specifically hand-tailored, holistic scheduling strategy.

**Holistic scheduling.** From the caller’s perspective, we implement a unified macro module to execute the entire MoE layer’s forward and backward passes, thereby expanding our scheduling flexibility. For instance, during the backward pass, various communication operators can be overlapped with dependency-free computations, such as activation recomputation, to improve efficiency. From the runtime perspective, a key challenge is efficiently managing concurrent communication tasks by resolving resource conflicts to prevent blocking and maximize throughput. This requires careful coordination, such as determining the number of SMs allocated to each communication operator, to minimize interference and optimize overall throughput.

<span id="figure-09"></span>

![Figure 9. Activation shapes in rematerialization.](../../papers/megascale-moe/figure-09.png)

**Figure 9.** Activation shapes in rematerialization.

<span id="figure-10"></span>

![Figure 10. Fine-grained intra-operator communication-computation overlap.](../../papers/megascale-moe/figure-10.png)

**Figure 10.** Fine-grained intra-operator communication-computation overlap.

**Selective activation rematerialization.** The holistic scheduling strategy also helps reduce memory usage without compromising training speed. Compared to dense models with equivalent computational requirements, MoE models exert significantly higher memory pressure during training due to their parameter count being several times larger. In addition to employing ZeRO optimizations [Raj20] to eliminate redundant optimizer states across DP groups, we further optimize memory usage through selective activation rematerialization. This approach reduces activation memory requirements by re-performing computation and communication operators that can be overlapped with other necessary operators.

[Figure 8a](#figure-08) illustrates the forward pass of a Mixtral [Jia24a] MoE layer and highlights key activations produced during this process. MegaScale-MoE strategically retains activations that are computationally expensive to recompute, while recalculating others generated by memory-intensive operations or communication operations. This minimizes dependencies on backward computation, enabling rematerialization operations to overlap with other computations and communications, avoiding delays in the critical path. For example, as shown in [Figure 8b](#figure-08), the backward pass of the GroupedGEMM operator for FC2 requires the activation `fc2_in` and the gradient of `fc2_out` (denoted as $\Delta$`fc2_out`) as inputs. MegaScale-MoE recomputes `fc2_in` and overlaps this operator with gradient communication (i.e., all-gather for $\Delta$`ffn_out`). Similarly, `ffn_in` is obtained through re-performing `RMSNorm` and all-gather, with these operators hidden within the preceding communication and the FC2 GroupedGEMM, respectively. MegaScale-MoE also places the weighted sum of `ffn_out` immediately after the SwiGLU [Sha20] activation function to eliminate the need to store `ffn_out`. This reordering ensures computational consistency by avoiding operators that cross non-linear boundaries.

[Figure 9](#figure-09) illustrates the shapes of the key activations produced during forward propagation, with the highlighted activations retained for backward propagation. Let the model parallelism size within one MoE layer be $n$ and the intermediate hidden size of one expert be $fh$. The total activation of a single MoE layer is

$$
(2n+2k+3kf+12+5/m)bsh/n,
$$

which we have reduced to

$$
(2kf+4+2/m)bsh/n.
$$

MegaScale-MoE reduces the activation memory by $\sim 50\%$ while maintaining the same training speed.

<span id="section-4-2"></span>

### 4.2 Intra-operator Overlap

Although inter-operator overlap effectively hides communication latency, squeezing all bubbles in the execution timeline remains non-trivial—especially in the forward pass, where no rematerialization or gradient computation operators exist to overlap with communication. Some forward operators directly depend on communication, such as token dispatch for expert computation, making overlap impossible unless another micro-batch is introduced, which increases memory pressure.

A widely adopted solution [Jia24f, Tra25, Wan22b] is to decompose operators into smaller parallel ones to enable pipelining by executing them on separate CUDA streams. However, this approach introduces non-negligible overhead: $(i)$ complex stream control, involving host interference and causing random bubbles due to the non-deterministic feature of CPU control; $(ii)$ imperfect tail computation, increasing overall computation latency.

To address the above issues, we adopt intra-operator overlap to parallelize communication and computation operators with direct dependencies. The core idea is to fuse these operators and break down the workloads into tiles. Following prior work [Jan22, Cha24c, Zha25e, Zhe25a], we implement barriers in device memory between communication and computation operators. These barriers enable fine-grained tile-level notifications and remove the need for host interference, further improving training performance. We implement two types of kernels, overlapping with GEMMs and overlapping with MoE GroupedGEMMs, for the attention and FFN modules, respectively.

**Overlapping with GEMMs.** We first introduce the intra-operator communication-computation overlap for GEMM kernels. Specifically, we implement all-to-all(A2A)+GEMM and GEMM+A2A kernels for Output and QKV Projections in SP attention, respectively, where X+Y means Y executed after X. [Figure 10](#figure-10) shows the data flow and overlapping pattern in A2A+GEMM. The GEMM on local data and communication for remote data starts simultaneously. We leverage dedicated GPU copy engines for data transfer, ensuring that all SMs (streaming multiprocessors) are fully utilized for computation. Once a remote data tile arrives at local memory, a signal notifies the GEMM kernel to continue its computation on the arrived tile. For GEMM+A2A, the all-to-all operation is fused into the GEMM kernel. Each tile of GEMM computation ends with a remote data transfer that writes the output data tile to remote ranks. We also implement all-gather+GEMM and GEMM+reduce-scatter kernels for tensor parallelism, which are similar to A2A+GEMM and GEMM+A2A.

For A2A+GEMM and GEMM+A2A, we allocate a small number of SMs for communication as all-to-all is more complex than all-gather and reduce-scatter. The number of SMs for communication is tuned to make communication and computation exhibit similar latency. Moreover, multiple ranks may simultaneously read from or write to the same device, potentially causing contention in NVLink. To mitigate this, we apply swizzling [Cha24c, Zha25e, Zhe25a] to reorder tile communication and computation so that the arrival of communication tiles aligns with the pace of computation tiles.

**Overlapping with GroupedGEMMs** For expert parallelism with token dispatch and combine, we aim to overlap communication with GroupedGEMMs. We implement two types of overlapping kernels: all-gather+scatter+GroupedGEMM and GroupedGEMM+gather+reduce-scatter. Unlike the overlapping techniques for GEMM kernels, MoE GroupedGEMMs require token shuffling (scatter/gather). As a result, each computation tile may depend on tokens from multiple ranks. To effectively overlap computation with communication, we sort the token order to minimize the number of dependent ranks for each computation tile. Additionally, since each tile has its own dependencies, the signal control for each tile varies depending on the MoE routing, which is determined dynamically.

In detail, for AG+scatter+GroupedGEMM, we reorder tokens along the sequence dimension based on their routed expert index. Then, for each expert, we sort the routed tokens according to their source rank index. Finally, we slice the sorted sequence into blocks and perform GroupedGEMM using a sequence of computation tiles. Specifically, as shown in [Figure 10c](#figure-10), we fuse the local scatter into the kernel by selecting rows of input data based on the index mapping. The GroupedGEMM computation for each expert is divided into tiles, with each tile depending on only a subset or even a single source rank. This reduces the overall waiting time for each computation block, avoids redundant loading of expert parameters, and improves the overlap between computation and communication tiles.

<span id="section-5"></span>

## 5 Communication Compression

We further reduce communication overhead by applying communication compression. To maintain convergence stability, mixed-precision training frameworks typically transfer tensors awaiting reduction in higher precision, such as FP32, to ensure more accurate accumulation. A common example of this is gradient reduce-scatter in data parallelism.

**DP communication compression.** As MoE model parameters increase, so does the communication overhead for parameter and gradient synchronization in data parallelism. Prior work has explored gradient compression to mitigate this cost. In our BF16 mixed-precision training, we carefully apply FP32-to-BF16 precision reduction for gradient synchronization, balancing efficiency and convergence stability.

<span id="figure-11"></span>

![Figure 11. DP communication compression.](../../papers/megascale-moe/figure-11.png)

**Figure 11.** DP communication compression.

Specifically, as shown in [Figure 11](#figure-11), we retain the main gradients in FP32 during local gradient accumulation in pipeline parallelism. After each model stage completes accumulation, instead of relying solely on reduce-scatter for gradient synchronization, we cast gradients to BF16 and perform all-to-all communication within the data parallel group to gather the required gradient shards, which are then locally aggregated in FP32. Our results show that this approach introduces negligible precision loss compared to directly performing reduce-scatter with FP32, while reducing gradient communication overhead by 50%.

This approach minimizes risk for two key reasons. First, it performs a one-time conversion of accumulated gradients to BF16 during communication, while the local gradient accumulation is maintained in FP32 precision. Second, instead of using ring-style reduce for BF16 gradient communication, it employs all-to-all communication, with the final reduction computed using FP32 summation. This design prevents precision loss that could arise from repeated accumulation of BF16 values in ring-based reductions.

We observe that casting large gradients and performing all-to-all communication increases peak memory consumption, potentially causing out-of-memory errors. To mitigate this, we develop a memory-efficient operator that in-places BF16 gradients into half of the FP32 input buffer while using the remaining half as the output buffer for BF16 all-to-all communication, preventing peak memory growth.

<span id="table-02"></span>

![Table 2. Model configurations in evaluation.](../../papers/megascale-moe/table-02.png)

**Table 2.** Model configurations in evaluation.

**Communication compression for FP8 training.** In low-precision FP8 training, the proportion of communication time increases due to reduced computation time. To mitigate communication overhead, we explore compressing communication volume using FP8 precision with appropriate quantization techniques. Currently, we apply communication compression in FP8 MoE training with tensor parallelism, focusing on reduction scenarios prone to overflow or underflow. For example, we adopt the E4M3 format (4-bit exponent and 3-bit mantissa) for all tensors. Similar to DP reduce-scatter compression, we replace BF16 TP reduce-scatter with FP8 all-to-all in forward propagation and perform reduction in FP32 precision. In the corresponding backward propagation, we apply FP8 all-gather for gradients. Notably, simply reducing precision leads to loss misalignment with BF16 training. To mitigate this, we apply per-token activation quantization for forward communication and per-channel quantization for backward communication. In backward propagation, we further group quantization along the token dimension using a small group size (e.g., 128).

<span id="section-6"></span>

## 6 Evaluation

In this section, we present a comprehensive evaluation of MegaScale-MoE, covering overall training performance ([Section 6.1](#section-6-1)), ablation studies of MegaScale-MoE’s key optimizations ([Section 6.2](#section-6-2)), and the effectiveness of the precision-communication co-design ([Section 6.3](#section-6-3)). [Table 2](#table-02) lists the configurations of the MoE models used in our evaluation, detailing hidden size ($h$), FFN intermediate size ($h_{\mathrm{ffn}}$), number of experts, and top-$k$ values. The evaluation is conducted on NVIDIA H800 GPUs unless otherwise specified, with the specifications provided in [Table 4](#table-04).

<span id="table-03"></span>

![Table 3. Strong-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs. The number in parentheses in the throughput column represents the speedup of MegaScale-MoE compared to Megatron-LM.](../../papers/megascale-moe/table-03.png)

**Table 3.** Strong-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs. The number in parentheses in the throughput column represents the speedup of MegaScale-MoE compared to Megatron-LM.

<span id="section-6-1"></span>

### 6.1 Training Performance

MegaScale-MoE is built on top of Megatron-LM [Sho19], a state-of-the-art open-source LLM training system that supports 3D parallelism strategies and is continuously updated to incorporate the latest optimizations from the community. Our evaluation uses the Megatron-LM on GitHub [Meg25] with commit hash f1f03922, selected for its stability at the commencement of our experiments months ago. For fair comparison, we use the same global batch size for Megatron-LM and MegaScale-MoE and choose the optimal parallelism configurations for the two systems, respectively. Specifically, MegaScale-MoE employs SP attention and EP within each node, while Megatron-LM adopts TP within each node, with both systems configured with a PP size of 15. We tune the configuration of Megatron-LM to meet its requirement of a uniform TP size across all components. As discussed in [Section 3.1](#section-3-1), for Megatron-LM, a TP size of 1 leads to a prohibitive $8\times$ activation memory (addressable only with slow recomputation via gradient checkpointing), while a TP size of 8 forces EP to operate across nodes, incurring more communication costs than PP. Notably, both systems in the evaluation enable the communication-computation overlap techniques from MegaScale [Jia24f] for data and pipeline parallelism. Therefore, the communication overhead mainly comes from intra-node model parallelism, e.g. TP, SP and EP. Sequence length is 8,192 and vocabulary size is 65,536.

<span id="figure-12"></span>

![Figure 12. Weak-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs.](../../papers/megascale-moe/figure-12.png)

**Figure 12.** Weak-scaling training performance for the 352B MoE model with NVIDIA H800 GPUs.

**Scalability.** [Table 3](#table-03) compares the strong-scaling training performance of Megatron-LM and MegaScale-MoE on the 352B MoE model. We scale the number of GPUs while keeping the global batch size fixed at 720. Across all settings, MegaScale-MoE achieves 1.65–$1.88\times$ speedups over Megatron-LM. As the number of GPUs increases, the MFU (Model FLOPs Utilization) of MegaScale-MoE declines from 32.48% to 27.89%. This is expected, as the batch size is fixed and the number of micro-batches for each pipeline decreases with more GPUs, leading to more bubbles.

[Figure 12](#figure-12) presents the weak-scaling training performance of Megatron-LM and MegaScale-MoE on the same model. We scale the global batch size from 360 to 1,080 in proportion to the number of GPUs (from 480 to 1,440). MegaScale-MoE achieves a 1.74-$1.79\times$ training throughput compared to Megatron-LM. As the scale increases, Megatron-LM’s throughput degrades by 2.74% due to increased communication overhead. In contrast, MegaScale-MoE exhibits near-linear scalability, with its throughput declining by only 0.2%, benefiting from comprehensive communication-computation overlap.

<span id="figure-13"></span>

![Figure 13. Performance breakdown of training Mixtral-$8\times$7B on different GPUs.](../../papers/megascale-moe/figure-13.png)

**Figure 13.** Performance breakdown of training Mixtral-$8\times$7B on different GPUs.

<span id="table-04"></span>

![Table 4. Specifications of different NVIDIA GPUs.](../../papers/megascale-moe/table-04.png)

**Table 4.** Specifications of different NVIDIA GPUs.

**Performance breakdown on different GPUs.** We conduct a deep dive into MegaScale-MoE to further understand the performance of training a MoE model in production environments. We train Mixtral-$8\times$7B on 32 NVIDIA H800, H20, and A100 GPUs, respectively. The specifications of GPUs we used are listed in [Table 4](#table-04). We set the DP size as four, the TP size as eight for Megatron-LM, and the SP and EP size as eight for MegaScale-MoE. As shown in [Figure 13b](#figure-13), across the four kinds of GPUs, MegaScale-MoE consistently outperforms Megatron-LM by up to $1.58\times$ in MFU. [Figure 13a](#figure-13) demonstrates the iteration time breakdown of Megatron-LM and MegaScale-MoE. Exposed communication time represents the communication time that is not overlapped with computation operations. FlashAttention and GEMMs are the operations we count when calculating MFU. The performance gain primarily results from MegaScale-MoE’s communication-efficient parallelism strategies and fine-grained overlapped communication.

Note that the MFU value decreases as GPU compute capability increases. This is because, unlike dense models, MoE models involve many memory-intensive operations like routing, local scatter, and gather, which remain time-consuming since memory bandwidth does not scale as quickly as compute capabilities. Additionally, GEMM efficiency declines with increasing compute capability, as it also relies on memory loading, constrained by memory bandwidth.

<span id="section-6-2"></span>

### 6.2 Ablation Study

<span id="table-05"></span>

![Table 5. Throughput improvement breakdown when training the 352B MoE model with 240 NVIDIA H800 GPUs and batch size is 720.](../../papers/megascale-moe/table-05.png)

**Table 5.** Throughput improvement breakdown when training the 352B MoE model with 240 NVIDIA H800 GPUs and batch size is 720.

We evaluate the effectiveness of the optimization techniques of MegaScale-MoE. First, we conduct an experiment about systematic breakdown by incrementally enabling each technique to isolate its contribution to the overall performance. [Table 5](#table-05) shows the throughput improvement breakdown with different optimizations when training the 352B MoE model on 240 GPUs with a global batch size of 720. The baseline is a version of MegaScale-MoE that adopts TP for both attention and FFNs and disables communication-computation overlap. First, by applying communication-efficient strategies—namely, SP for attention and EP for experts—we achieve an initial 13% throughput improvement over this baseline. We then target the primary bottleneck in large-scale MoE training: communication overhead. Our inter-operator and intra-operator overlap methods effectively hide these costs, further accelerating training by an additional 9% and 6%, respectively.

<span id="figure-14"></span>

![Figure 14. Parallelism efficiency for different models.](../../papers/megascale-moe/figure-14.png)

**Figure 14.** Parallelism efficiency for different models.

<span id="figure-15"></span>

![Figure 15. Parameter synchronization time under SP and TP attention.](../../papers/megascale-moe/figure-15.png)

**Figure 15.** Parameter synchronization time under SP and TP attention.

Following the systematic breakdown, we perform ablation studies on each component, varying a single setting at a time while keeping all others constant, to gain deeper insights into its behavior.

<span id="figure-16"></span>

![Figure 16. Overlapped communication-computation time vs. non-overlapped time of each layer. M1-M6 represent the six models listed from top to bottom in Table 2; A2A, AG, and RS refer to all-to-all, all-gather, and reduce-scatter, respectively.](../../papers/megascale-moe/figure-16.png)

**Figure 16.** Overlapped communication-computation time vs. non-overlapped time of each layer. M1-M6 represent the six models listed from top to bottom in [Table 2](#table-02); A2A, AG, and RS refer to all-to-all, all-gather, and reduce-scatter, respectively.

**Parallelism strategy.** We compare the training efficiency under various intra-node parallelism strategies using a single node with eight NVIDIA H800-SXM GPUs. We denote parallelism strategies as X+Y, where X represents the parallelism strategy for attention, and Y corresponds to that for experts. The available parallelism strategies for attention include TP and our SP, whereas for experts, the choices are TP and EP. To isolate the performance benefits of optimized parallelism, we disable other system optimizations.

<span id="figure-17"></span>

![Figure 17. Ablation study of selective activation rematerialization (SAR).](../../papers/megascale-moe/figure-17.png)

**Figure 17.** Ablation study of selective activation rematerialization (SAR).

We measure the training MFU of one internal and five open-source MoE models with diverse model configurations as listed in [Table 2](#table-02). The global batch size is set to 32, and we adjust the number of layers for each model to fit within the GPU memory. [Figure 14](#figure-14) shows that MegaScale-MoE’s parallelism strategy, SP+EP, consistently outperforms the other three parallelism strategies, achieving 14.9%-32.9% higher MFU compared to TP+TP. The performance gains are attributed to two main factors. First, as discussed in [Section 3](#section-3), SP and EP effectively reduce the communication volume compared to TP, thereby decreasing communication overhead. Second, TP partitions the FFN module along the intermediate size dimension, which results in lower GEMM efficiency.

To provide a more comprehensive evaluation of the parallelism strategy, we also report the additional overhead introduced by the replicated attention parameters in SP. In terms of memory usage, SP incurs a 1.2%–5.4% higher memory footprint compared to TP, requiring 1.7%–8.1% more memory to store parameters, gradients, and optimizer states across all seven models. This overhead is manageable considering the significant performance gains achieved by SP.

For the parameter synchronization time, we follow large-scale training setups and set the size of the TP or SP to 8, effectively parallelizing each layer within a single node. The attention parameter size on each GPU is varied from 384 MB to 1536 MB, while the FFN parameter size is fixed at 10 GB per GPU, reflecting typical real-world training setups. We run MegaScale-MoE with SP and TP attention, using 4 and 8 DP groups, which correspond to a total of 32 and 64 GPUs, respectively. [Figure 15](#figure-15) shows that the synchronization times for SP and TP attention are consistently comparable, differing by only 0.3%–3.1%. This aligns with our hypothesis that SP and TP would exhibit similar performance characteristics in DP communication latency.

**Intra-operator commmunication overlap.** We then measure the duration of four key communication and the corresponding computation operators in the forward pass: $(i)$ QKV Projection paired with all-to-all, $(ii)$ all-to-all with Output Projection, $(iii)$ all-gather with scatter and GroupedGEMM, and $(iv)$ GroupedGEMM with gather and reduce-scatter, as depicted in [Figure 8](#figure-08). [Figure 16](#figure-16) demonstrates that across all six models, MegaScale-MoE achieves a 1.2–$4.7\times$ reduction in the combined time of communication and computation operators compared to the baseline lacking fine-grained overlap. And MegaScale-MoE reduces the training iteration time by 7.1%-12.9% due to intra-operator communication-computation overlap.

<span id="figure-18"></span>

![Figure 18. The training loss curve of MegaScale-MoE with DP communication compression.](../../papers/megascale-moe/figure-18.png)

**Figure 18.** The training loss curve of MegaScale-MoE with DP communication compression.

**Selective activation rematerailization.** We compare MegaScale-MoE to a baseline that disables selective activation rematerialization (No SAR), which stores all activations in GPU memory during training. We evaluate both methods by training Mixtral-$8\times$7B and Mixtral-$8\times$22B on 128 NVIDIA H800 GPUs. [Figure 17](#figure-17) shows the memory usage breakdown and the training MFU. Compared to No SAR, MegaScale-MoE reduces activation memory consumption by 45.5% and 57.2% for the two models, respectively, resulting in overall memory reductions of 21.3% and 35%, while maintaining the training performance difference within 0.5%.

**Data parallelism communication compression.** We validate the effectiveness of our communication compression technique by training a 7B MoE model using BF16 all-to-all DP communication and FP32 reduce-scatter communication, as described in [Section 5](#section-5). [Figure 18](#figure-18) illustrates the training loss curves, which are nearly identical. This optimization compresses only the accumulated gradients of the batch and performs conversions between BF16 and FP32 exclusively during communication, introducing minimal risk.

<span id="section-6-3"></span>

### 6.3 Model Convergence

We evaluate model convergence with MegaScale-MoE. [Figure 19](#figure-19) demonstrates the loss curves of training a 35B MoE model from scratch and continuing training a 176B MoE model from a checkpoint, with results shown for both BF16 and FP8 precision. MegaScale-MoE ensures stable convergence and consistent training loss across BF16 and FP8 formats.

<span id="figure-19"></span>

![Figure 19. The loss curve of MegaScale-MoE in FP8 and BF16.](../../papers/megascale-moe/figure-19.png)

**Figure 19.** The loss curve of MegaScale-MoE in FP8 and BF16.

<span id="section-7"></span>

## 7 Experience

In this section, we describe our deployment and operational experience of MegaScale-MoE.

**Deployment experience.** MegaScale-MoE has been deployed in our production environment and is responsible for the majority of large-scale MoE training tasks within our company. It enables the training of models with trillions of parameters, supports single training jobs scaling beyond 10,000 GPUs, with individual training tasks running for several months. By combining the aforementioned techniques, MegaScale-MoE minimizes idle communication time and optimizes memory usage in MoE training without compromising model performance, ultimately saving millions of GPU hours in large-scale MoE training. [Figure 20](#figure-20) shows the model convergence from a real production job, which trains a proprietary MoE model with 200B parameters, 20B activated for each token. This job uses over 10,000 GPUs and lasts for months. The loss continues to converge with a stable training process.

**FP8 training.** We have made extensive efforts to maintain the convergence stability of FP8 training. For example, we observe that the SwiGLU operator significantly expands the numerical range. To address this, we replace per-tensor quantization with higher-precision per-token quantization ($1\times h$). Additionally, since multiplying SwiGLU with the gating weight further amplifies the dynamic numerical range, we shift the gating weight multiplication back to after the FC2 output, reducing quantization errors.

Beyond ensuring training convergence, we introduce additional engineering optimizations. Existing FP8 training implementations [Tra25, Lia24b] store model parameters in BF16, requiring frequent FP8 conversion for GEMM computations, adding casting and transpose overhead. To address this, we use a multi-precision optimizer to store model parameters directly in FP8, while keeping main parameters in FP32 with separate buffers for different data types. This lowers memory consumption and halves parameter all-gather communication in data parallelism.

**Scale up.** When training MoE models, an intriguing engineering question arises: can we indefinitely scale the training size by increasing model parameters without raising computational load? This approach is impractical in tensor parallelism, as scaling up the model necessitates a higher TP degree to accommodate additional parameters. While increased TP reduces per-GPU computation, the communication overhead remains constant, as shown in [Equation 1](#equation-01) and [Equation 4](#equation-04), leading to progressively longer communication times and reduced training efficiency. In other words, TP has inherent scalability limitations and often relies on high-speed intra-node links to mitigate communication delays.

<span id="figure-20"></span>

![Figure 20. The normalized training loss curve of a real production job on more than 10,000 GPUs for months, training a MoE model with 20B activated and 200B total parameters on multi-trillion tokens. Different colors indicate training restarts.](../../papers/megascale-moe/figure-20.png)

**Figure 20.** The normalized training loss curve of a real production job on more than 10,000 GPUs for months, training a MoE model with 20B activated and 200B total parameters on multi-trillion tokens. Different colors indicate training restarts.

In contrast, when scaling training with SP and EP, the communication volume decreases as the parallel size $n$ increases, as shown in [Equation 2](#equation-02) and [Equation 3](#equation-03). This implies that, in theory, this parallelism strategy can scale to significantly larger sizes. However, in practical hierarchical infrastructures, a critical challenge emerges: can this approach maintain training efficiency when scaling beyond the NVLink domain, where bandwidth drops to RDMA levels?

Formally, for a SwiGLU structure incorporating a MoE mechanism, the ratio $R$ between computation time and communication time is defined as:

<span id="equation-05"></span>

$$
\mathrm{comm\_time}=\frac{2k\times bsh(n-1)/n/n}{\mathrm{bandwidth}},
$$

<span id="equation-06"></span>

$$
\mathrm{comp\_time}=\frac{3k\times bsh\times h_{\mathrm{ffn}}/n}{\mathrm{peak}}.
$$

<span id="equation-07"></span>

$$
R=\frac{\mathrm{comp\_time}}{\mathrm{comm\_time}}
$$

<span id="equation-08"></span>

$$
=3/2\times h_{\mathrm{ffn}}\times\frac{\mathrm{bandwidth}}{\mathrm{peak}}\times n/(n-1)
$$

<span id="equation-09"></span>

$$
\approx 3/2\times h_{\mathrm{ffn}}\times\frac{\mathrm{bandwidth}}{\mathrm{peak}}
$$

To sustain training efficiency, the FFN’s computation time must exceed the communication time, ensuring effective overlap of communication overhead. Therefore, our goal is to maintain $R>1$, leading to two key insights:

- The value of $R$ is independent of the number of experts, top-$k$, hidden dimension, parallelism size, or input size, providing flexibility in selecting algorithm parameters.
- $R$ is solely determined by the expert’s intermediate dimension, computational peak, and communication bandwidth. Consequently, on fixed hardware, as long as the expert dimension is sufficiently large, the MoE model can be scaled while maintaining training efficiency from an engineering perspective.

**Holistic vs. automatic.** We have invested substantial engineering efforts in inter-operator communication-computation overlap, including determining operator execution order, concurrency of communication and computation, and SM allocation for communication. These manual interventions provide deeper insights into training dynamics, enabling targeted optimizations. As training progresses and experience accumulates, we seek to automate operator scheduling within the search space to optimize the training process at a fine-grained level and achieve optimal performance. We leave automatic optimization for future work.

**MoE vs. dense model training.** In our continued efforts to optimize MoE model training, we have identified several critical distinctions from the training of dense models. In a dense Transformer layer, optimization efforts are concentrated on self-attention and GEMMs. The former is often accelerated by techniques like FlashAttention [Dao22], while the latter, as a dense computation, generally achieves high utilization on the GPU’s parallel processing units. In contrast, as shown in [Figure 13a](#figure-13), the combined runtime of attention and GroupedGEMM accounts for only about one-third of a layer’s execution time. The remainder is consumed by communication and other operators. While MegaScale-MoE effectively addresses the communication overhead, we observe that the computational operators in MoE models, which are inherently more complex than their dense counterparts, also introduce performance degradation. Specifically, they are a primary source of stragglers for three main reasons:

First, the intermediate dimension of each expert is smaller than the FFN layer in a dense model. To efficiently process computations for multiple experts concurrently, GroupedGEMM employs a single CUDA kernel for numerous small matrix multiplications. The resource usage of this kernel—including shared memory, L1 cache, and number of threads—is finely controlled via `cuFuncSetAttribute`. This granular control, however, can introduce synchronization delays. Second, due to the imbalanced number of tokens routed to each expert, the inputs and outputs for GroupedGEMM are dynamically shaped tensors. The frequent allocation and deallocation of these tensors exacerbate GPU memory fragmentation. Third, the MoE gating mechanism involves a multitude of small operators for tasks like calculating routing scores and communicating routing decisions. Jitter in CPU performance can delay the launch of these kernels to the point where the launch latency exceeds their actual execution time on the GPU, creating pipeline bubbles.

<span id="section-8"></span>

## 8 Related Work

**Large model training.** LLM research has led to the development of scalable, efficient, and robust training techniques [Ras20, Sho19, Jia24f, Zha25ax] to meet the substantial computational demands of these models. DeepSpeed [Ras20] features the Zero Redundancy Optimizer (ZeRO) [Raj20, Raj21, Ren21], which shards model parameters, gradients, and optimizer states across participating GPUs in data parallelism, enabling the scaling of LLMs with manageable memory consumption. Megatron-LM [Sho19] focuses on intra-layer model parallelism techniques, partitioning the parameters and computation of each layer. Pipeline parallelism assigns the parameters and computation of a contiguous subset of layers to each GPU[Hua19, Nar19], breaks a batch into micro-batches, and processes the micro-batches in a pipelined fashion. MegaScale [Jia24f] shows how combining tensor, pipeline, and data parallelism can be an efficient strategy to train large multi-billion parameter models at unprecedented scale.

**Mixture-of-Expert training.** To address the computational challenges of training advanced neural networks, the machine learning field has increasingly adopted Mixture-of-Experts architectures. Subsequently, a number of deep learning frameworks have been proposed for training or running inference on MoEs on multi-GPU clusters. DeepSpeed-MoE [Raj22] significantly reduces training costs through model architecture designs and compression techniques. HetuMoE [Nie22] utilizes a hierarchical all-to-all communication strategy to achieve performance speedup. SE-MoE [She22] distinguishes itself by focusing on scalable and efficient training with heterogeneous resources like CPU memory and SSDs. FasterMoE [He22] introduces a comprehensive suite of optimizations such as dynamic shadowing, fine-grained scheduling, and congestion-avoiding expert selection strategies. Janus [Liu23r] proposes a data-centric paradigm shift for MoE models, aiming to lower communication demands and boost training efficiency. Tutel [Hwa23] offers a dynamic solution for MoE models, employing adaptive parallelism and pipelining. However, its dynamic parallelism switching and hierarchical all-to-all can cause significant overheads for models with hundreds of billions of parameters. To avoid such overhead, latest MoE training systems [Dee24d, Dee24a] use auxiliary loss or routing bias for load balancing and limit cross-node token dispatch. By mapping each MoE layer to intra-node, MegaScale-MoE eliminates cross-node token dispatch.

Recently, DeepSeek-V3 [Dee24a] introduced two key optimizations for training production-scale MoE models: DeepEP, for high-performance cross-node all-to-all communication, and DualPipe, for overlapping communication with computation. Due to the relatively low cross-node InfiniBand bandwidth, DeepEP limits the token dispatch to a maximum of 4 nodes to maintain a constant cross-node communication volume, restricting its routing flexibility. In contrast, MegaScale-MoE places each MoE layer intra-node to ensure efficient routing to any top-k experts. DualPipe leverages pipeline parallelism for communication-computation overlap across different micro-batches, which requires storing $2\times$ the model parameters. In contrast, MegaScale-MoE’s overlap occurs within a single micro-batch’s forward or backward pass, incurring no additional memory overhead and remaining compatible with systems both with and without pipeline parallelism.

**Long-context training.** While Megatron-LM [Sho19, Kor22] opts to partition only specific operations along the sequence dimension, various methods of sequence parallelism [Li24s, Liu23, Li23g, Gu24a] have been explored for training models requiring long contexts. The Blockwise Parallel Transformer [Liu24w] method implements blockwise computation of self-attention and the fusion of FFNs based on online softmax calculations. Ring Attention [Liu23, Li23g] introduces a ring-style communication mechanism integrated with self-attention calculations, facilitating the exchange of key and value chunks. We adopt the all-to-all style of SP attention from DeepSpeed Ulysses [Sam23], which partitions attention by heads rather than sequence length, due to its reduced communication volume and balanced computation pattern.

**Communication-computation overlap.** Several frameworks [Has19, Li20c, Mah23, Pen19, Zha23] focus on overlapping communication with computation in distributed deep learning training with a single parallelism strategy. Some compiler-style work [Jan22, Wan22b, Pat24b] provides fine-grained overlap among kernels, but excessive partitioning of GEMM kernels can result in low GPU utilization. Centauri [Che24f] enhances communication overlap for LLM training with 3D parallelism by communication partitioning and hierarchical scheduling. Similar to Centauri, our inter-operator communication overlap hides communication within independent computation by reordering operators. We further conceal communication on critical paths through intra-operator overlap, without compromising GPU utilization.

<span id="section-9"></span>

## 9 Conclusion

In this paper, we offer an in-depth look at the design, implementation, and deployment of MegaScale-MoE, a production-grade system built to efficiently train MoE models. MegaScale-MoE exploits communication-efficient approaches, including parallelism strategies with lower communication volume, inter- and intra-operator communication-computation overlap, and communication compression with adjusted communication patterns to unleash the compute capabilities of high-performance GPUs. MegaScale-MoE achieves 1.41M tokens/s in throughput when training a 352B MoE model on 1,440 NVIDIA Hopper GPUs, a $1.88\times$ improvement over Megatron-LM. By sharing our insights on accelerating large-scale MoE training, we hope our work will inspire future research.

## Acknowledgements

We thank our shepherd, Cheng Li, and the anonymous reviewers for their valuable feedback and suggestions. This work was supported in part by the National Key Research and Development Program of China under Grant 2022YFB4500700, the Scientific Research Innovation Capability Support Project for Young Faculty under Grant ZYGXQNJSKYCXNLZCXM-I1, the Fundamental Research Funds for the Central Universities, Peking University, and the National Natural Science Foundation of China under Grant 62172008 and Grant 62325201. Xin Jin and Xin Liu are the corresponding authors. Chao Jin, Xuanzhe Liu, and Xin Jin are also with the Key Laboratory of High Confidence Software Technologies (Peking University), Ministry of Education.

<span id="section-10"></span>

## 10 Appendix

<span id="section-10-1"></span>

### 10.1 Hierarchical Communication for Parameter Synchronization

Let the full attention weights size be $P$, the dimension of model parallelism (TP or SP) be $n$, and the data parallel size be $d$. Typically, GPUs for model parallelism are located on the same node, requiring intra-node communication, whereas data parallelism spans across nodes, requiring inter-node communication. Consider a data parallelism group containing $d$ devices, each holding the identical partition of the parameter.

For parameter synchronization in TP attention, communication involves data of size $P/n$ across $d$ devices in two primary steps in LLM training:

- inter-node `reduce-scatter` operation, where the data size is $P/n$, on $d$ devices.
- inter-node `all-gather` operation, where the data size is $P/n$, on $d$ devices.

leading to primarily inter-node communication, with a communication volume of $2P/n(d-1)/d$.

With SP attention, the parameter synchronization involves the entire data of size $P$ across $n\times d$ devices. Considering the discrepancy between intra-node and inter-node network bandwidth, this process can be implemented by four-step hierarchical communication, where the replicated parameters are first reduced within a node and then reduced across nodes, before being distributed back to each device. [Figure 5a](#figure-05) illustrates a hierarchical communication example where $n=3$ and $d=2$. The detailed steps are as follows.

- intra-node `reduce-scatter` operation, where the data size is $P$, on $n$ devices.
- inter-node `reduce-scatter` operation, where the data size is $P/n$, on $d$ devices.
- inter-node `all-gather` operation, where the data size is $P/n$, on $d$ devices.
- intra-node `all-gather` operation, where the data size is $P$, on $n$ devices.

The inter-node communication volume in SP attention remains at $2P/n(d-1)/d$, with additional intra-node volume of $2P(n-1)/n$.

Moreover, due to the distinct resources for intra-node and inter-node communications, these steps can be segmented into small chunks and pipelined to efficiently hide each other as shown in [Figure 5b](#figure-05). The ratio of inter-node communication latency and intra-node communication latency is

<span id="equation-10"></span>

$$
\frac{1}{n}\times\frac{\mathrm{intra\text{-}node\ bandwidth}}{\mathrm{inter\text{-}node\ bandwidth}}\times\frac{n(d-1)}{d(n-1)}
$$

Consider a typical training scenario involving an H100 SXM machine, where the NVLink bandwidth is 450 GB/s, and the inter-device NIC communication bandwidth is 50 GB/s. In this context, the latency of inter-node communication can easily surpass that of intra-node communication. This implies that the communication within a node can overshadow that between nodes. Consequently, in such scenarios, the synchronization of gradients and parameters with SP attention is, in fact, consistent with TP attention.
