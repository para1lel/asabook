---
title: 'ZeRO: Training Trillion-Parameter Models'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/zero/
---

> [Samyam Rajbhandari](https://dblp.org/pid/115/9021) [+1], [Jeff Rasley](https://x.com/jeffra45) [+1], [Olatunji Ruwase](https://www.cs.cmu.edu/~oor/), and [Yuxiong He](https://x.com/yuxionghe). First submitted to arXiv on October 4, 2019; current version v3. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054). [Original PDF](/paper/zero.pdf). [TeX source](https://export.arxiv.org/e-print/1910.02054v3). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Large deep learning models offer significant accuracy gains, but training billions to trillions of parameters is challenging. Existing solutions such as data and model parallelisms exhibit fundamental limitations to fit these models into limited device memory, while obtaining computation, communication and development efficiency. We develop a novel solution, Zero Redundancy Optimizer (*ZeRO*), to optimize memory, vastly improving training speed while increasing the model size that can be efficiently trained. *ZeRO* eliminates memory redundancies in data- and model-parallel training while retaining low communication volume and high computational granularity, allowing us to scale the model size proportional to the number of devices with sustained high efficiency. Our analysis on memory requirements and communication volume demonstrates: *ZeRO* has the potential to scale beyond 1 *Trillion* parameters using today’s hardware.

We implement and evaluate ZeRO: it trains large models of over 100B parameter with super-linear speedup on 400 GPUs, achieving throughput of 15 Petaflops. This represents an 8x increase in model size and 10x increase in achievable performance over state-of-the-art. In terms of usability, *ZeRO* can train large models of up to 13B parameters (e.g., larger than Megatron GPT 8.3B and T5 11B) without requiring model parallelism which is harder for scientists to apply. Last but not the least, researchers have used the system breakthroughs of *ZeRO* to create the world’s largest language model (17B parameters) with record breaking accuracy.

<span id="section-01"></span>

## 1 Extended Introduction

Deep Learning (DL) models are becoming larger, and the increase in model size offers significant accuracy gain. In the area of Natural Language Processing (NLP), the transformers have paved way for large models like Bert-large (0.3B) [Dev18], GPT-2 (1.5B) [Rad19], Megatron-LM (8.3B) [Sho20], T5 (11B) [Raf19]. To enable the continuation of model size growth from 10s of billions to trillions of parameters, we experience the challenges of training them—they clearly do not fit within the memory of a single device, e.g., GPU or TPU, and simply adding more devices will not help scale the training.

Basic data parallelism (DP) does not reduce memory per device, and runs out of memory for models with more than 1.4B parameters on current generation of GPUs with 32 GB memory. Other existing solutions such as Pipeline Parallelism (PP), Model Parallelism (MP), CPU-Offloading, etc, make trade-offs between functionality, usability, as well as memory and compute/communication efficiency, but all of which are crucial to training with speed and scale.

Among different existing solution for training large models, MP is perhaps the most promising. The largest models in the current literature, the 11B T5 model [Raf19], and Megatron-LM 8.3B [Sho20], were both powered by model parallelism, implemented in Mesh-Tensorflow [Sha18a] and Megatron-LM[Sho20], respectively. However, MP cannot scale much further beyond these models sizes. MP splits the model vertically, partitioning the computation and parameters in each layer across multiple devices, requiring significant communication between each layer. As a result, they work well within a single node where the inter-GPU communication bandwidth is high, but the efficiency degrades quickly beyond a single node [Sho20]. We tested a 40B parameter model using Megatron-LM across two DGX-2 nodes and observe about $5\,\mathrm{Tflops}$ per V100 GPU (less than 5% of hardware peak).

So, how can we overcome the limitations of existing solutions and train large models more efficiently? To answer this question, we first analyze the full spectrum of memory consumption of the existing systems on model training and classify it into two parts: 1) For large models, the majority of the memory is occupied by *model states* which include the optimizer states (such as momentum and variances in Adam [Kin15]), gradients, and parameters. 2) The remaining memory is consumed by activation, temporary buffers and unusable fragmented memory, which we refer to collectively as *residual* states. We develop *ZeRO*—Zero Redundancy Optimizer—to optimize memory efficiency on both while obtaining high compute and communication efficiency. As these two parts face different challenges, we develop and discuss their solutions correspondingly.

**Optimizing Model State Memory** Model states often consume the largest amount of memory during training, but existing approaches such as DP and MP do not offer satisfying solution. DP has good compute/communication efficiency but poor memory efficiency while MP can have poor compute/communication efficiency. More specifically, DP replicates the entire model states across all data parallel process resulting in redundant memory consumption; while MP partition these states to obtain high memory efficiency, but often result in too fine-grained computation and expensive communication that is less scaling efficient. Furthermore, all of these approaches maintain all the model states required over the entire training process statically, even though not all model states are required all the time during the training.

<span id="figure-01"></span>

![Refer to caption](../../papers/zero/figure-01.png)

**Figure 1.** Comparing the per-device memory consumption of model states, with three stages of *ZeRO*-DP optimizations. $\Psi$ denotes model size (number of parameters), $K$ denotes the memory multiplier of optimizer states, and $N_{d}$ denotes DP degree. In the example, we assume a model size of $\Psi=7.5B$ and DP of $N_{d}=64$ with $K=12$ based on mixed-precision training with Adam optimizer.

Based on these observations, we develop *ZeRO*-DP, ZeRO-powered data parallelism, that achieves the computation/communication efficiency of DP while achieving memory efficiency of MP. *ZeRO*-DP removes the memory state redundancies across data-parallel processes by *partitioning* the model states instead of replicating them, and it retains the compute/communication efficiency by retaining the computational granularity and communication volume of DP using a dynamic communication schedule during training.

*ZeRO*-DP has three main optimization stages (as depicted in [Figure 1](#figure-01)), which correspond to the partitioning of optimizer states, gradients, and parameters. When enabled cumulatively:

1. Optimizer State Partitioning ($P_{\mathrm{os}}$): 4x memory reduction, same communication volume as DP;
2. Add Gradient Partitioning ($P_{\mathrm{os}+g}$): 8x memory reduction, same communication volume as DP;
3. Add Parameter Partitioning ($P_{\mathrm{os}+g+p}$): Memory reduction is linear with DP degree $N_{d}$. For example, splitting across 64 GPUs ($N_{d}=64$) will yield a 64x memory reduction. There is a modest 50% increase in communication volume.

ZeRO-DP eliminates memory redundancies and makes the full aggregate memory capacity of a cluster available. With all three stages enabled, ZeRO can train a trillion-parameter model on just 1024 NVIDIA GPUs. A trillion-parameter model with an optimizer like Adam [Kin15] in 16-bit precision requires approximately 16 terabytes (TB) of memory to hold the optimizer states, gradients, and parameters. 16TB divided by 1024 is 16GB, which is well within a reasonable bound for a GPU (e.g., with 32GB of on-device memory).

**Optimizing Residual State Memory** After *ZeRO*-DP boosts memory efficiency for model states, the rest of the memory consumed by activations, temporary buffers, and unusable memory fragments could become a secondary memory bottleneck. We develop *ZeRO*-R to optimize the residual memory consumed by these three factors respectively.

1. For activations (stored from forward pass in order to perform backward pass), we noticed checkpointing [Che16] helps but not sufficient for large models. Thus *ZeRO*-R optimizes activation memory by identifying and removing activation replication in existing MP approaches through activation partitioning. It also offloads activations to CPU when appropriate.
2. *ZeRO*-R defines appropriate size for temporary buffers to strike for a balance of memory and computation efficiency.
3. We observe fragmented memory during training due to variations in the lifetime of different tensors. Lack of contiguous memory due to fragmentation can cause memory allocation failure, even when enough free memory is available. *ZeRO*-R proactively manages memory based on the different lifetime of tensors, preventing memory fragmentation.

*ZeRO*-DP and *ZeRO*-R combined together forms a powerful system of memory optimizations for DL training that we collectively refer to as *ZeRO*.

<strong><em>ZeRO</em> and MP</strong>: Since *ZeRO* eliminates the memory inefficiency in DP, it is natural to ask: Do we still need MP, and when? How does *ZeRO* work with MP? With *ZeRO*, MP becomes a less attractive option for the purpose of fitting large models alone. *ZeRO*-DP is at least as effective on reducing per-device memory footprint as MP, or more effective sometimes when MP cannot divide the model evenly. It also has comparable or better scaling efficiency. Furthermore, data parallelism is so easy to use that it is widely applicable across different workloads, while MP approaches today often need some work from model developers to revise their model, system developers to work out distributed operators, and existing work like Megatron-LM only supports a limited set of operators and models.

That being said, there are still cases where we want to leverage MP: i) When used with *ZeRO*-R, MP can reduce activation memory footprint for very large models. ii) For smaller models where activation memory is not an issue, MP can also have benefits when aggregated batch size using DP alone is too big to have good convergence. [+2] In those case, one can combine *ZeRO* with MP to fit the model with an acceptable aggregated batch size.

We show that *ZeRO* can be combined with MP, resulting in a max theoretical memory reduction of $N_{d}\times N_{m}$ times on each device with a DP degree of $N_{d}$ and MP degree of $N_{m}$. This could allow us to fit a trillion parameter model on 1024 GPUs with 16-way model parallelism (within each DGX2 node) and 64-way data parallelism across nodes, and run it efficiently using a modest batch size!

<span id="figure-02"></span>

![Refer to caption](../../papers/zero/figure-02.png)

**Figure 2.** *ZeRO* training throughput and speedup w.r.t SOTA baseline for varying model sizes. For *ZeRO*, the MP always fit in a node, while for baseline, models larger than 40B require MP across nodes.

<span id="figure-03"></span>

![Refer to caption](../../papers/zero/figure-03.png)

**Figure 3.** Superlinear scalability and per GPU training throughput of a 60B parameter model using *ZeRO*-100B.

**Implementation & Evaluation** The complete set of optimizations in *ZeRO* could allow us to run models with trillion parameters on the high-end hardware cluster today (e.g., with 1K V100 GPUs), however, the hardware compute capacity is still too limited and training time can be impractically long ($>$1 year). Therefore, our focus for this implementation is to efficiently support models with 10x parameters ($\sim 100\mathrm{B}$ parameters) than state-of-the-art (SOTA) while still being within reach of the compute capabilities of current hardware. We implement and evaluate a subset of optimizations in *ZeRO* called *ZeRO*-100B—$P_{\mathrm{os}+g}$ of *ZeRO*-DP plus ZeRO-R—that allow us to achieve this goal. The results show:

**Model Size** Combined with MP, *ZeRO*-100B runs 170B parameter models efficiently, while the existing system like using Megatron alone cannot scale efficiently beyond 40B parameters, as shown in [Figure 2](#figure-02). This is an over 8x increase in model size compared to SOTA.

**Speed** Improved memory efficiency powers higher throughput and faster training. As shown in [Figure 2](#figure-02), *ZeRO* runs 100B parameter models on a 400 Nvidia V100 GPU cluster with over 38 TFlops per GPU, and aggregate performance over 15 Petaflops. This is more than 10x improvement in training speed compared to SOTA for the same model size.

**Scalability** We observe super linear speedup in the regime of 64-400 GPUs, where the performance more than doubles when we double the number of GPUs. This is a property of *ZeRO*-DP which reduces the memory footprint of the model states as we increase the DP degree allowing us to fit larger batch sizes per GPU resulting in better performance. We expect this behaviour to continue further as we increase the number of GPUs beyond 400.

**Democratization of Large Model Training** *ZeRO*-100B powers data scientist to train models with up to 13B parameters without any MP or PP that requires model refactoring, where 13B is more parameters than the largest model in literature (T5 with 11B parameters). Data scientists can thus experiment freely with large models without worrying about parallelism. In comparison, exist systems (e.g., PyTorch Distributed Data Parallel) runs out of memory with 1.4B parameter models.

**New SOTA Model** *ZeRO* powers the largest language model with 17B parameters and record-breaking accuracy, Turing-NLG [Mic20].

We share *ZeRO* as a part of our open source DL training optimization library called DeepSpeed [+3]. We plan to release all implementations described in this paper by end of May 2020 and extend it further to support 1 trillion parameters by enabling *ZeRO*-DP stage 3 partitioning parameters ($P_{\mathrm{os}+g+p}$). We plan to make *ZeRO* fully accessible to the DL community to catalyze the evolution and democratization of large model training at scale.

## 2 Related Work

### 2.1 Data, Model and Pipeline Parallelism

Parallelization is a key strategy on training large models at scale. For a model that fits in the device memory for training, data parallelism (DP) is used to scale training to multiple devices. In DP, model parameters are replicated on each device. At each step, a mini-batch is divided evenly across all the data parallel processes, such that each process executes the forward and backward propagation on a different subset of data samples, and uses averaged gradients across processes to update the model locally.

When a model does not fit in the device memory, model parallelism (MP) [Sha18a, Sho20] and pipeline parallelism (PP) [Hua19, Har18] split the model among processes, in vertical and horizontal way respectively. Sec. [1](#section-01) discussed how *ZeRO* relates to DP and MP. We now discuss PP and how it relates to reducing memory consumption.

PP splits a model horizontally across layers running each partition on a different device and use micro-batching to hide the pipeline bubble [Hua19, Har18]. Model functionalities such as tied-weights and batch-normalization are difficult to implement due to horizontal splitting and micro-batching, respectively. Popular PP implementation such as G-pipe [Hua19] partitions both model parameters and total activations but requires a batch size proportional to number of pipeline partitions to hide the pipeline bubble. The large batch size can affect the convergence rate, while also requiring significant memory to store activations. A different implementation of PP in PipeDream [Nar19] keeps multiple copies of stale parameters to hide the pipeline bubble without increasing the batch size significantly, making it less memory efficient. Additionally, the implementation is not equivalent to the standard DL training and has implications on training convergence. In contrast, *ZeRO* obtains the same or better memory efficiency than PP without incurring functionality, performance and convergence related restrictions of PP.

### 2.2 Non-parallelism based approach to reduce memory

In addition to MP and PP, there are multiple lines of work that target reducing memory overheads of DL training.

#### 2.2.1 Reducing Activation Memory

Multiple efforts have focused on reducing the memory footprint of activations through compression [Jai18], activation checkpointing [Che16, Jai19], or live analysis [Wan18a]. These efforts are complimentary and can work together with *ZeRO*. In fact, activation memory reduction in *ZeRO*-R works in parallel with activation checkpointing.

#### 2.2.2 CPU Offload

[Pud20, Rhu16] exploit heterogeneous nature of today’s compute nodes, offloading model states to CPU memory through algorithmic design or virtualized memory, respectively. Up to $50\%$ of training time can be spent on GPU-CPU-GPU transfers [Pud20]. *ZeRO* differs in that it reduces the memory consumption significantly without storing the model states to CPU memory whose bandwidth is severely constrained due to PCI-E. On rare cases, *ZeRO*-R may offload just the activation checkpoints for very large models to improve performance (see Sec. [6.1](#section-06-01) for details).

#### 2.2.3 Memory Efficient Optimizer

[Sha18, Ani19] focus on reducing memory consumption of adaptive optimization methods by maintaining coarser-grained statistics of model parameters and gradients, with potential impact on model convergence guarantees. *ZeRO* is orthogonal to these efforts, and its optimizations do not change the model optimization method or affect model convergence, but effectively reduce memory footprint of optimizer states and gradients per device.

### 2.3 Training Optimizers

Adaptive optimization methods [Duc11, Kin15, You17, You19] are crucial to achieving SOTA performance and accuracy for effective model training of large models. Compared to SGD, by maintaining fine-grained first-order and second-order statistics for each model parameter and gradient at the cost of significant memory footprint. *ZeRO* can reduce the memory footprint of these optimizers by orders of magnitude, making these sophisticated optimization methods practical for training large models on hardware with modest device memory. It also makes it possible to develop and use even more complex and memory hungry optimizers that may have better convergence.

## 3 Where Did All the Memory Go?

Let’s take a step back to examine the memory consumption of the current training system. For example, a 1.5B parameter GPT-2 model requires 3GB of memory for its weights (or parameters) in 16-bit precision, yet, it cannot be trained on a single GPU with 32GB memory using Tensorflow or PyTorch. One may wonder where all the memory goes. During model training, most of the memory is consumed by model states, i.e., tensors comprising of optimizer states, gradients, and parameters. Besides these model states, the rest of the memory is consumed by activations, temporary buffers and fragmented memory which we call *residual states*. We look at the memory consumption from both in details.

### 3.1 Model States: Optimizer States, Gradients and Parameters

Majority of the device memory is consumed by model states during training. Consider for instance, Adam [Kin15], one of the most popular optimizers for DL training. Adam requires storing two optimizer states, i) the time averaged momentum and ii) variance of the gradients to compute the updates. Therefore, to train a model with ADAM, there has to be enough memory to hold a copy of both the momentum and variance of the gradients. In addition, there needs to be enough memory to store the gradients and the weights themselves. Of these three types of the parameter-related tensors, the optimizer states usually consume the most memory, specially when mixed-precision training is applied.

**Mixed-Precision Training** The state-of-the-art approach to train large models on the current generation of NVIDIA GPUs is via mixed precision (fp16/32) training [Mic17], where parameters and activations are stored as fp16, enabling the use of the high throughput tensor core units [Nvi17a] on these GPUs. During mixed-precision training, both the forward and backward propagation are performed using fp16 weights and activations. However, to effectively compute and apply the updates at the end of the backward propagation, the mixed-precision optimizer keeps an fp32 copy of the parameters as well as an fp32 copy of all the other optimizer states.

Let’s take Adam as a concrete example. Mixed precision training of a model with $\Psi$ parameters using Adam requires enough memory to hold an $\mathrm{fp}16$ copy of the parameters and the gradients, with memory requirements of $2\Psi$ and $2\Psi$ bytes respectively. In addition, it needs to hold the optimizer states: an $\mathrm{fp}32$ copy of the parameters, momentum and variance, with memory requirements of $4\Psi$, $4\Psi$, and $4\Psi$ bytes, respectively. Let’s use $K$ to denote the memory multiplier of the optimizer states, i.e., the additional memory required to store them is $K\Psi$ bytes. Mixed-precision Adam has $K=12$. In total, this results in $2\Psi+2\Psi+K\Psi=16\Psi$ bytes of memory requirement. For a model such as GPT-2 with $1.5$ Billion parameters, this leads to a memory requirement of at least $24\,\mathrm{GB}$, which is significantly higher than the meager $3\,\mathrm{GB}$ of memory required to hold the $\mathrm{fp}16$ parameters alone.

### 3.2 Residual Memory Consumption

**Activations** can take up a significant amount of memory [Che16] during training. As a concrete example, the 1.5B parameter GPT-2 model trained with sequence length of 1K and batch size of 32 requires about 60 GB of memory [+4]. Activation checkpointing (or activation recomputation) is a common approach to reduce the activation memory by approximately the square root of the total activations at the expense of $33\%$ re-computation overhead [Che16]. This would reduce the activation memory consumption of this model to about 8 GB.

Despite the significant reduction, the activation memory can grow quite large for bigger models even with activation checkpointing. For example, a GPT-like model with 100 billion parameters requires around 60 GB of memory for batch size 32, even when using activation checkpointing.

**Temporary buffers** used for storing intermediate results consumes non-trivial amount of memory for large models. Operations such as gradient all-reduce, or gradient norm computation tend to fuse all the gradients into a single flattened buffer before applying the operation in an effort to improve throughput. For example, the bandwidth of all-reduce across devices improves with large message sizes. While the gradient themselves are usually stored as fp16 tensors, the fused buffer can be an fp32 tensor depending on the operation. When the size of the model is large, these temporary buffer sizes are non-trivial. For example, for a model with 1.5B parameters, a flattened fp32 buffer would required $6\,\mathrm{GB}$ of memory.

**Memory Fragmentation:** So far we have discussed the actual memory consumption during training. Additionally, it is possible to run out of usable memory even when there is plenty of available memory. This can happen with memory fragmentation. A request for a memory will fail if there isn’t enough contiguous memory to satisfy it, even if the total available memory is larger than requested. We observe significant memory fragmentation when training very large models, resulting in out of memory issue with over 30% of memory still available in some extreme cases.

<span id="section-04"></span>

## 4 *ZeRO*: Insights and Overview

*ZeRO* has two sets of optimizations: i) *ZeRO*-DP aimed at reducing the memory footprint of the model states, and ii) *ZeRO*-R targeted towards reducing the residual memory consumption. We present an overview of the optimizations and the insights behind, which allows *ZeRO* to reduce memory footprint while remaining efficient. Please note efficiency is a key here: without this constraint, trivial solutions like moving all the parameter states to the CPU memory, or increasing the MP degree arbitrarily can reduce memory footprint.

### 4.1 Insights and Overview: *ZeRO*-DP

*ZeRO* powered DP is based on three key insights:

1. DP has better scaling efficiency than MP because MP reduces the granularity of the computation while also increasing the communication overhead. Beyond a certain point, lower computational granularity reduces the efficiency per GPU, while the increased communication overhead, hiders the scalability across GPUs, especially when crossing node boundaries. On the contrary, DP has both higher computational granularity and lower communication volume, allowing for much higher efficiency.
2. DP is memory inefficient as model states are stored redundantly across all data-parallel processes. On the contrary, MP partitions the model states to obtain memory efficiency.
3. Both DP and MP keep all the model states needed over the entire training process, but not everything is required all the time. For example, parameters corresponding to each layer is only needed during the forward propagation and backward propagation of the layer.

Based on these insights, *ZeRO*-DP retains the training efficiency of DP while achieving the memory efficiency of MP. *ZeRO*-DP *partitions* the model states instead of replicating them (Section [5](#section-05)) and uses a dynamic communication schedule that exploits the intrinsically temporal nature of the model states while minimizing the communication volume (Section [7](#section-07)). By doing so, *ZeRO*-DP reduces per-device memory footprint of a model *linearly* with the increased DP degree while maintaining the communication volume close to that of the default DP, retaining the efficiency.

<span id="section-04-02"></span>

### 4.2 Insights and Overview: *ZeRO*-R

#### 4.2.1 Reducing Activation Memory

Two key insights are:

1. MP partitions the model states but often requires replication of the activation memory. For example, if we split the parameters of a linear layer vertically and compute them in parallel across two GPUs, each GPU requires the entire activation to compute its partition
2. For models such as GPT-2 or larger, the arithmetic intensity (ratio of the amount of computation per iteration to amount of activation checkpoints per iteration) is very large ($\geq 10K$) and increases linearly with hidden dimension making it possible to hide the data-movement cost for the activation checkpoints, even when the bandwidth is low.

*ZeRO* removes the memory redundancies in MP by *partitioning* the activations checkpoints across GPUs, and uses allgather to reconstruct them on demand. The activation memory footprint is reduced proportional to the MP degree. For very large models, *ZeRO* can even choose to move the activation partitions to the CPU memory, while still achieving good efficiency due to large arithmetic intensity in these models.

#### 4.2.2 Managing Temporary buffers

*ZeRO*-R uses constant size buffers to avoid temporary buffers from blowing up as the model size increases, while making them large enough to remain efficient.

#### 4.2.3 Managing fragmented Memory

Memory fragmentation is a result of interleaving between short lived and long lived memory objects. During the forward propagation activation checkpoints are long lived but the activations that recomputed are short lived. Similarly, the backward computation, the activation gradients are short lived while the parameter gradients are long lived. Based on this insight, *ZeRO* performs on-the-fly memory defragmentation by moving activation checkpoints and gradients to pre-allocated contiguous memory buffers. This not only increases memory availability but also improves efficiency by reducing the time it takes for the memory allocator to find free contiguous memory.

<span id="section-05"></span>

## 5 Deep Dive into *ZeRO*-DP

While the existing DP approach replicates the model states at each device and introduces significant memory overhead, *ZeRO*-DP eliminates this memory redundancy by partitioning them—optimizer states, gradients and parameters—across data parallel processes. [Figure 1](#figure-01) quantifies and visualizes the memory requirement with and without *ZeRO*-DP. The figure shows the memory footprint after partitioning (1) optimizer state, (2) gradient and (3) parameter redundancies accumulatively. We refer to them as the three optimization phases of *ZeRO*-DP: $P_{\mathrm{os}}$, $P_{g}$, and $P_{p}$, which we elaborate below.

### 5.1 $P_{\mathrm{os}}$: Optimizer State Partitioning

For a DP degree of $N_{d}$, we group the optimizer states into $N_{d}$ equal partitions, such that the $i^{\mathrm{th}}$ data parallel process only updates the optimizer states corresponding to the $i^{\mathrm{th}}$ partition. Thus, each data parallel process only needs to store and update $\frac{1}{N_{d}}$ of the total optimizer states and then only update $\frac{1}{N_{d}}$ of the parameters. We perform an all-gather across the data parallel process at the end of each training step to get the fully updated parameters across all data parallel process.

**Memory Savings:** As shown in [Figure 1](#figure-01), the memory consumption after optimizing state partition reduces from $4\Psi+K\Psi$ to $4\Psi+\frac{K\Psi}{N_{d}}$. As the concrete example depicted in [Figure 1](#figure-01), a 7.5 B parameter model requires 31.4GB of memory using $P_{\mathrm{os}}$ with 64-way DP ($N_{d}=64$), while requiring 120 GB with standard DP. Furthermore, when $N_{d}$ is large, the memory requirement on model states reduces from $4\Psi+12\Psi=16\Psi$ bytes to $4\Psi+\frac{12\Psi}{N_{d}}\approx 4\Psi$ bytes, leading to a 4x reduction.

### 5.2 $P_g$: Gradient Partitioning

As each data parallel process only updates its corresponding parameter partition, it only needs the reduced gradients for the corresponding parameters. Therefore, as each gradient of each layer becomes available during the backward propagation, we only reduce them on the data parallel process responsible for updating the corresponding parameters. After the reduction we no longer need the gradients and their memory can be released. This reduces the memory footprint required to hold the gradients from $2\Psi$ bytes to $\frac{2\Psi}{N_{d}}$.

Effectively this is a Reduce-Scatter operation, where gradients corresponding to different parameters are reduced to different process. To make this more efficient in practice, we use a bucketization strategy, where we bucketize all the gradients corresponding to a particular partition, and perform reduction on the entire bucket at once. This is similar in spirit to how NVIDIA’s AMP [Nvi19] optimizer bucketizes the all-reduce gradient computation to overlap communication and computation. In our case we perform a reduction instead of an all-reduce at the partition boundaries to reduce memory footprint and overlap computation and communication.

**Memory Savings:** By removing both gradient and optimizer state redundancy, we reduce the memory footprint further down to $2\Psi+\frac{14\Psi}{N_{d}}\approx 2\Psi$. As the example in [Figure 1](#figure-01), a 7.5 B parameter model requires only 16.6 GB of memory using $P_{\mathrm{os}+g}$ with 64-way DP ($N_{d}=64$), while requiring 120 GB with standard DP. When $N_{d}$ is large, the memory requirement of model states reduces from $2\Psi+14\Psi=16\Psi$ bytes to $2\Psi+\frac{14\Psi}{N_{d}}\approx 2\Psi$ bytes, leading to a 8x reduction.

<span id="table-01"></span>

![Original paper Table 1](../../papers/zero/table-01.png)

**Table 1.** Per-device memory consumption of different optimizations in *ZeRO*-DP as a function of DP degree . Bold-faced text are the combinations for which the model can fit into a cluster of 32GB V100 GPUs.

### 5.3 $P_p$: Parameter Partitioning

Just as with the optimizer states, and the gradients, each process only stores the parameters corresponding to its partition. When the parameters outside of its partition are required for forward and backward propagation, they are received from the appropriate data parallel process through broadcast. While this may seem to incur significant communication overhead at first glance, we show that this approach only increases the total communication volume of a baseline DP system to $1.5$x, while enabling memory reduction proportional to $N_{d}$.

**Memory Savings:** With parameter partitioning, we reduce the memory consumption of an $\Psi$ parameter model from $16\Psi$ to $\frac{16\Psi}{N_{d}}$. As the example in [Figure 1](#figure-01), a 7.5 B parameter model requires 1.9 GB of model-state memory using $P_{\mathrm{os}+p+g}$ with 64-way DP ($N_{d}=64$), while requiring 120 GB with standard DP. This has a profound implication: *ZeRO powers DP to fit models with arbitrary size—as long as there are sufficient number of devices to share the model states*.

### 5.4 Implication on Model Size

The three phases of partitioning $P_{\mathrm{os}}$, $P_{\mathrm{os}+g}$, and $P_{\mathrm{os}+g+p}$ reduces the memory consumption of each data parallel process on model states by up to 4x, 8x, and $N_{d}$ respectively. [Table 1](#table-01) analyzes model-state memory consumption of a few example models under the 3 stages of *ZeRO*-DP optimizations for varying DP degree. Without *ZeRO*, the memory consumption is equal to the first row in the table, regardless of the DP degree. Note that, with $N_{d}=64$, *ZeRO* can train models with up to 7.5B, 14B, and 128B parameters using $P_{\mathrm{os}}$, $P_{\mathrm{os}+g}$, and $P_{\mathrm{os}+g+p}$, respectively. When $N_{d}=1024$, *ZeRO* with all of its optimizations enabled ($P_{\mathrm{os}+g+p}$) could train models with 1 Trillion parameters! Or potentially, models with Arbitrary size! Without *ZeRO*, the largest model DP alone can run has less than 1.5 Billion parameters.

## 6 Deep Dive into *ZeRO*-R

<span id="section-06-01"></span>

### 6.1 $P_{a}$: Partitioned Activation Checkpointing

As discussed in [4.2](#section-04-02), MP by design requires a replication of the activations, resulting in redundant copies of the activations across model parallel GPUs. *ZeRO* eliminates this redundancy by partitioning the activations, and only materializes them in a replicated form one activation layer at a time, right before the activation is used in computation. More specifically, once the forward propagation for a layer of a model is computed, the input activations are partitioned across all the model parallel process, until it is needed again during the backprogation. At this point, *ZeRO* uses an all-gather operation to re-materialize a replicated copy of the activations. We refer to this optimization as $P_{a}$. It works in conjunction with activation checkpointing [Che16], storing partitioned activation checkpoints only instead of replicated copies. Furthermore, in the case of very large models and very limited device memory, these partitioned activation checkpoints can also be offloaded to the CPU reducing the activation memory overhead to nearly zero at an additional communication cost, which we will discuss in [7](#section-07). We refer to this as $P_{a+\mathrm{cpu}}$.

**Memory Saving** With partitioned activation checkpointing, *ZeRO* reduces the activation footprint by a factor proportional to the MP degree. Consider training a 100B model shown in [Table 4](#table-04) with a batch size of 32, sequence length of 1024 and a MP degree of 16. If we checkpoint a single activation for each transformer layer, it would require about 33 GB of memory per GPU just to store the activation checkpoints. But with $P_a$ in *ZeRO*, it can be reduced to about 2 GB per GPU. Furthermore, this 2GB can be offloaded to the CPU reducing the memory footprint for activations to nearly zero.

### 6.2 $C_B$: Constant Size Buffers

*ZeRO* carefully selects the sizes of the temporal-data buffers to balance memory and compute efficiency. During training, the computational efficiency of some operations can be highly dependent on the input size, with larger inputs achieving higher efficiency. For example, a large all-reduce operation achieves much higher bandwidth than a smaller one. Hence, to get better efficiency, high performance libraries such as NVIDIA Apex or Megatron fuses all the parameters into a single buffer before applying these operations. However, the memory overhead of the fused buffers is proportional to the model size, and can become inhibiting. For example, for a 3B parameter model, a 32-bit fused buffer will require 12 GB of memory. To address this issue, we simply use a performance-efficient constant-size fused buffer when the model becomes too large. By doing so, the buffer size does not depend on the model size, and by keeping the buffer size large enough, we can still achieve good efficiency.

### 6.3 $M_D$: Memory Defragmentation

Memory fragmentation in model training occurs as a result of activation checkpointing and gradient computation. During the forward propagation with activation checkpointing, only selected activations are stored for back propagation while most activations are discarded as they can be recomputed again during the back propagation. This creates an interleaving of short lived memory (discarded activations) and long lived memory (checkpointed activation), leading to memory fragmentation. Similarly, during the backward propagation, the parameter gradients are long lived, while activation gradients and any other buffers required to compute the parameter gradients are short lived. Once again, this interleaving of short term and long term memory causes memory fragmentation.

Limited memory fragmentation is generally not an issue, when there is plenty of memory to spare, but for large model training running with limited memory, memory fragmentation leads to two issues, i) OOM due to lack of contiguous memory even when there is enough available memory, ii) poor efficiency as a result of the memory allocator spending significant time to search for a contiguous piece of memory to satisfy a memory request.

*ZeRO* does memory defragmentation on-the-fly by pre-allocating contiguous memory chunks for activation checkpoints and gradients, and copying them over to the pre-allocated memory as they are produced. $M_D$ not only enables *ZeRO* to train larger models with larger batch sizes, but also improves efficiency when training with limited memory.

<span id="section-07"></span>

## 7 Communication Analysis of *ZeRO*-DP

As *ZeRO* boosts model size by removing memory redundancy, it is only natural to ask if we are trading communication volume for memory efficiency. In other words, what is the communication volume of *ZeRO*-powered DP approach compared to a baseline DP approach? The answer is in two parts: i) *ZeRO*-DP incurs no additional communication using $P_{\mathrm{os}}$ and $P_{g}$, while enabling up to 8x memory reduction, ii) *ZeRO*-DP incurs a maximum of $1.5$x communication when using $P_{p}$ in addition to $P_{\mathrm{os}}$ and $P_{g}$, while further reducing the memory footprint by $N_{d}$ times. We present the analysis in this section. We begin by first presenting a brief overview of the communication volume for standard DP.

### 7.1 Data Parallel Communication Volume

During data parallel training, gradients across all data parallel processes are averaged at the end of the backward propagation before computing the updates for the next step. The averaging is performed using an all-reduce communication collective. For a large model size, the all-reduce communication is entirely communication bandwidth bound, and therefore, we limit our analysis to the total communication volume send to and from each data parallel process.

State-of-art implementation of all-reduce uses a two-step approach, where the first step is a reduce-scatter operation, which reduces different part of the data on different process. The next step is an all-gather operation where each process gathers the reduced data on all the process. The result of these two steps is an all-reduce. Both reduce-scatter and all-gather are implemented using a pipelined approach, that results in a total data movement of $\Psi$ elements (for a data with $\Psi$ elements) for each. Therefore, the standard DP incurs $2\Psi$ data movement during each training step.

<span id="table-02"></span>

![Original paper Table 2](../../papers/zero/table-02.png)

**Table 2.** Maximum model size through memory analysis (left) and the measured model size when running with *ZeRO-OS* (right). The measured model size with $P_{\mathrm{os}}$ matches the theoretical maximum, demonstrating that our memory analysis provides realistic upper bounds on model sizes.

### 7.2 *ZeRO*-DP Communication Volume

#### 7.2.1 Communication Volume with $P_{\mathrm{os}+g}$

With gradient partitioning, each process only stores the portion of the gradients, that is required to update its corresponding parameter partition. As such, instead of an all-reduce, *ZeRO* only requires a scatter-reduce operation on the gradients, incurring communication volume of $\Psi$. After each process updates the partition of the parameters that it is responsible for, an all-gather is performed to collect all the updated parameters from all the data parallel process. This also incurs a communication volume of $\Psi$. So the total communication volume per training step is $\Psi+\Psi=2\Psi$, exactly the same as the baseline DP.

#### 7.2.2 Communication Volume with $P_{\mathrm{os}+g+p}$

After parameter partitioning, each data parallel process only stores the parameters that it updates. Therefore, during the forward propagation it needs to receives the parameters for all the other partitions. However, this can be pipelined to avoid the memory overhead. Before computing the forward propagation on the part of the model corresponding to a particular partition, the data parallel process responsible for that partition can broadcast the weights to all the data parallel processes. Once the forward propagation for that partition is done, the parameters can be discarded. The total communication volume is thus $\frac{\Psi\times N_{d}}{N_{d}}=\Psi$. In other words, we reschedule the parameter all-gather by spreading it across the entire forward propagation, and discarding the parameters once they have been used. Note however that this all-gather needs to happen once again for the backward propagation in the reverse order.

The total communication volume is therefore the sum of the communication volumes incurred by these all-gathers in addition to the communication volume incurred by the reduce-scatter of the gradients. The total volume is therefore $3\Psi$ which is 1.5x compared to the baseline. Both gradient and parameter partitioning leverage the insight that—not all states of gradients and parameters are needed all the time—to optimize memory by communicating the states judiciously.

## 8 Communication Analysis of *ZeRO*-R

We compare the communication volume of partitioned activation checkpointing ($P_{a}$) in *ZeRO*-R with baseline MP, and show that $P_{a}$ incurs a communication volume increase that is in general less than one tenth of the baseline MP. Furthermore, we analyze the communication overhead of $P_{a}$ in relation to DP communication volume to identify scenarios when $P_{a}$ improves efficiency by allowing for a larger batch size and reducing DP communication. We leverage such analysis to decide if and when to apply $P_{a}$ as well as $P_{a+\mathrm{cpu}}$.

Communication volume trade-off of partitioning activation checkpoints depends on the model size, checkpointing strategy and the MP strategy. To share concrete insights, we perform the analysis in the context of transformer based models implemented using SOTA MP approach, Megatron-LM.

In Megatron-LM with activation checkpointing, each transformer block performs two all-reduce operations of size $\mathit{batch}\times\mathit{seq\_length}\times\mathit{hidden\_dim}$ in the forward propagation, two all-reduce for forward re-computation and two more in the backward propagation. The total communication per block is $12\times\mathit{seq\_length}\times\mathit{hidden\_dim}$ since communication volume of an all-reduce is $2\times\mathit{message\_size}$.

When *ZeRO*-R partitions activation checkpoints, it requires an additional all-gather operation before the forward recomputation of the back-propagation on each activation checkpoint. In general, we checkpoint the input activation for each transformer block, requiring one all-gather per transformer block. The communication overhead $P_{a}$ is therefore $\mathit{seq\_length}*\mathit{hidden\_dim}$, since the communication volume of an all-gather is $\mathit{message\_size}$. Therefore, the total communication overhead of $P_{a}$ is less than $10\%$ of the original communication volume for model parallelism.

When MP is used in conjunction with DP, $P_{a}$ can be used to reduce the data-parallel communication volume by an order of magnitude at the expense of a $10\%$ increase in model-parallel communication volume, and significantly boost efficiency when data-parallel communication is a performance bottleneck. Notice that $P_{a}$ reduces the activation memory consumption by the MP degree allowing for a proportional increase in batch size. For large models, MP can be as large as 16 (#GPUs on a DGX-2 node), allowing for up to 16x increase in the batch size. The communication volume of a data-parallel training is inversely proportional to the batch size. Therefore, an order of magnitude increase in batch size due to $P_{a}$ could result in an order-of-magnitude decrease in data-parallel communication volume.

Finally if $P_{a+\mathrm{cpu}}$ is applied, partitioned activation checkpoints are offloaded to CPU, reducing the activation memory requirement to nearly zero at the expense of 2x added data movement to and from CPU memory compared to $P_{a}$. In extreme cases where DP communication volume is the major bottleneck due to a small batch size even with $P_{a}$, $P_{a+\mathrm{cpu}}$ can improve efficiency by increasing the batch size as long as the CPU data transfer overhead is less than the DP communication volume overhead, which is generally true for small batch sizes.

Given model and hardware characteristics, we leverage the above analysis to decide if and when to apply $P_{a}$ and $P_{a+\mathrm{cpu}}$.

## 9 Step Towards 1 Trillion Parameters

The largest published models today are in the range of 10 billion parameters, which are already challenging to train. Getting to a trillion parameters, 3-orders of magnitude larger, will inevitably happen, but the road will be full of hurdles, surprises and innovations. While we do not claim knowing or addressing all of them, *ZeRO* addresses one of the most fundamental challenges from a system perspective: the ability to fit a model of this scale on current hardware while allowing it to train with good system scalability.

**A Leap from State-of-Art** The largest model that the state-of-art framework, Megatron, can train with acceptable throughput is a 16-20B parameter model in a DGX-2 system. Scaling further by having model parallelism across multiple DGX nodes results in significant efficiency drop due to limited internode bandwidth.

*ZeRO* vastly increase the efficiently-runnable model size. It enables the current generation of hardware to run significantly larger models without requiring fine-grained model parallelism to go across the node boundaries. As demonstrated in [Table 1](#table-01), *ZeRO*, with all optimizations turned on ($P_{\mathrm{os}+g+p}$), could fit more than 1 *Trillion* parameters on 1024 GPUs using DP only. Alternatively, when combined with model parallelism (as shown in [Table 2](#table-02)), *ZeRO* could fit more than 1 *Trillion* parameters on 1024 GPUs with 16-way model parallelism (within each DGX2 node) and 64-way data parallelism across nodes. Running a model with a trillion parameters efficiently is no longer impossible!

**Compute Power Gap** Training a trillion parameter model end-to-end within an acceptable time range, however, could still require significant amount of compute power, which is lacking in today’s AI clusters.

To understand the resource requirement, we present a brief comparison with Bert-Large. Bert-Large can be trained in $67$ minutes on a $1024$ GPU DGX-2H cluster [Nar19a]. A 1 Trillion Parameter model can easily contain $3000$x (1 trillion / 330 million) more computation than a Bert-Large model for a data sample. Even if we assume the same sequence length and the total number of samples required to train the model, training a 1T model would take 140 days, assuming the same hardware and similar computational efficiency. In practice, both data samples and sequence length are likely to increase with the increased model size requiring over a year to train. It would require an exa-flop system to train a 1T parameter model in a reasonable time. But when such compute capacity becomes available, we hope *ZeRO* will provide the system technology to run the 1T models efficiently.

<span id="figure-04"></span>

![Refer to caption](../../papers/zero/figure-04.png)

**Figure 4.** Max model throughput with *ZeRO*-DP.

<span id="figure-05"></span>

![SOTA Turing-NLG enabled by *ZeRO*.](../../papers/zero/source-x5.png)

**Figure 5.** SOTA Turing-NLG enabled by *ZeRO*.

## 10 Implementation and Evaluation

We focus our implementation on supporting efficient training of models with $\sim 100\mathrm{B}$ parameters, which are an order-of-magnitude larger than the largest published models today (e.g., T5-11B [Raf19]) while trainable within a reasonable time frame on current hardware (e.g., with 1K V100 GPUs). We implement and evaluate a subset of optimizations in *ZeRO*—$P_{\mathrm{os}+g}$ in *ZeRO*-DP plus ZeRO-R—that allows us to achieve this goal. We will refer to this implementation as *ZeRO*-100B. Our results show that *ZeRO*-100B can efficiently train models with up to 170B parameters, 8x bigger than SOTA, up to 10x faster and with improved usability. *ZeRO*-100B powers Turing-NLG, the largest published model in the world with new SOTA accuracy.

<span id="figure-06"></span>

![Refer to caption](../../papers/zero/figure-06.png)

**Figure 6.** Max model size.

<span id="figure-07"></span>

![Refer to caption](../../papers/zero/figure-07.png)

**Figure 7.** Max cache allocated.

<span id="figure-08"></span>

![Throughput per GPU.](../../papers/zero/source-x8.png)

**Figure 8.** Throughput per GPU.

### 10.1 Implementation and Methodology

**Implementation** We implemented *ZeRO*-100B in PyTorch including the full set of optimizations in $P_{\mathrm{os}+g}$ and *ZeRO*-R. Its interface is compatible with any model implemented as an torch.nn.module. Users can simply wrap their models using this interface and leverage *ZeRO*-powered DP as they use classic DP. Users do not need to modify their model. *ZeRO*-powered DP can be combined with any form of MP including Megatron-LM.

**Hardware** We conducted our experiments on a cluster of 400 V100 GPUs ($25$ DGX-2 nodes) with 800 Gbps internode communication bandwidth.

**Baseline** For experiments without MP, we use torch’s distributed data parallel (DDP) as baseline. For experiments with MP, we use Megatron-LM because it is, to our knowledge, the state-of-art. We use the open-source version of Megatron-LM from NVIDIA [+5] with a date of September 2019. The most recent Megatron-LM results report the ability to scale up to 16B parameter models using 32 DGX-2 nodes (total of 512 32GB V100 GPUs) [Sho20].

***ZeRO*** Experiments without MP, use the *ZeRO*-powered DP implementation in *ZeRO*-100B. Experiments with MP, combine *ZeRO*-powered DP with MP of Megatron-LM.

**Model Configurations** The models presented in this section are GPT-2 [Rad19] like transformer based models. We vary the hidden dimension and the number of layers to obtain models with different number of parameters. [Table 4](#table-04) shows the configuration parameters used in our experiments with additional details in AE Appendix.

### 10.2 Speed and Model Size

*ZeRO*-100B efficiently run models with up to 170B parameters on 400 GPUs, more than 8x bigger than Megatron-LM. [Figure 2](#figure-02) shows throughput per GPU for varying model sizes using *ZeRO*-100B with MP versus using Megatron MP alone. *ZeRO*-100B achieves a sustained throughput of 15 PetaFlops (over 30% of the peak) on average for models with 8B to 100B parameters. In comparison, the baseline MP performance degrades quickly with the increase in model size: MP incurs high communication volume between GPUs, and going beyond a single node to fit larger models causes a communication bandwidth drop from 300GB/sec per link (NVSwitch) to 12.5 GB/sec per link (Infiniband EDR), resulting in a significant performance drop. *ZeRO*-100B achieves up to 10x speedup over baseline, significantly outperforming on large models.

For *ZeRO*-100B, the slight reduction in performance beyond 100B is due to lack of enough memory to run larger batch sizes. We expect the performance to improve as we increase the number of GPUs due to super-linear speedup of *ZeRO*-100B as we discuss next.

<span id="table-03"></span>

![Original paper Table 3](../../papers/zero/table-03.png)

**Table 3.** *ZeRO* configurations.

### 10.3 Super-Linear Scalability

*ZeRO*-100B demonstrates super-linear scalability for very large model sizes. [Figure 3](#figure-03) shows scalability results for a 60B parameter model going from 64 to 400 GPUs and we expect this trend to continue further for more GPUs. $P_{\mathrm{os}+g}$ reduces per GPU memory consumption of *ZeRO*-100B with increase in DP degree, allowing *ZeRO*-100B to fit larger batch sizes per GPU [+6], which in turn improves throughput as a result of increasing arithmetic intensity.

### 10.4 Democratizing Large Model Training

Using MP and PP is challenging for many data scientists, which is a well-known hurdle to train large models. *ZeRO* does not require any changes to the model itself and it can be used as simple as baseline DP while delivering significantly boosted model size and speed. Fig. [4](#figure-04) shows that *ZeRO*-100B can train models with up to 13B parameters without MP on 128 GPUs, achieving throughput over 40 TFlops per GPU on average. In comparison, without *ZeRO*, the largest trainable model with DP alone has 1.4B parameters with throughput less than 20 TFlops per GPU. Furthermore, in the absence of the communication overhead from MP, these models can be trained with lower-end compute nodes without very fast intra-node interconnect such as NVLINK or NVSwitch, which is required to achieve good efficiency with MP.

### 10.5 Memory and Performance Analysis

We look into the benefits and impact of different optimizations on maximum model size, memory consumption and performance. These optimizations are referred to as Config 1 to 5 (C1-C5) in [Table 3](#table-03).

**Maximum Model Size** [Figure 6](#figure-06) shows the largest trainable model by enabling different *ZeRO* optimizations for a fixed batch size and MP of 16. The model size increase from 40B to 60B when trained with C1 vs C2 due to a 16x (MP degree) reduction in activation memory from using $P_{a}$, while the jump to 140B using C4 is from enabling $P_{\mathrm{os}+g}$ which halves the memory requirement by the model states compared to $P_{\mathrm{os}}$ in C2. The increase to 150B using C5 is solely due to further reduction in activation memory from offloading the partitioned activation checkpoints to the CPU memory.

**Max Cached Memory** [Figure 7](#figure-07) shows the maximum memory cached by PyTorch during each training iteration for a 40B and a 100B parameter model. The decrease of the cached memory size is as expected from C1 to C2. The difference in memory consumption between C2 and C3 depends on the size of the model states in comparison to the activation memory, and can increase when activation memory is larger, or decrease when the model states are larger. It is note worthy that the cached memory does not decrease from C4 to C5 for 40B but it does for 100B. This is simply because the activation memory for 100B is much larger for the decrease to be noticeable. This makes $P_{a+\mathrm{cpu}}$ a valuable tool to fit a larger batch size when we get to very large models. In [Figure 8](#figure-08), $P_{a+\mathrm{cpu}}$ is needed for 170B model to execute without running out of memory.

<span id="table-04"></span>

![Original paper Table 4](../../papers/zero/table-04.png)

**Table 4.** Configurations for different model sizes, number of layers, and hidden dimensions (HD) across [Figures 2](#figure-02), [3](#figure-03), [4](#figure-04).

**Max Achievable Performance** [Figure 8](#figure-08) shows the best achievable performance for different set of optimizations. Notice that performance improvement corresponds to decrease in memory consumption between the optimizations. As mentioned earlier, lower memory consumption allows for larger batch size which improves performance. The only caveat is the performance drop between C4 and C5 for 60B parameter model. Despite lower memory consumption, C5 incurs activation movement to and from the CPU, this will result in worse performance in most cases, except for a few where the model is so large that the model simply cannot run without C5 or the batch size that can run without C5 is very small (such as model with 170B parameters in [Figure 8](#figure-08)). During training, $P_{a+\mathrm{cpu}}$ is turned on only when it is beneficial.

### 10.6 Turing-NLG, the SOTA language model with 17B parameters

As of May 12th, 2020, Turing-NLG is the largest model in the world with over 17B parameters. It achieved the new SOTA for language models with Webtext-103 perplexity of 10.21. Turing-NLG was trained end-to-end using *ZeRO*-100B and Fig. [5](#figure-05) shows the validation perplexity over 300K iterations compared to previous SOTA, Megatron-LM 8.3B parameter model. *ZeRO*-100B achieves a sustained throughput of 41.4 TFlops/GPU for this model.

## 11 Concluding Remarks

From a HPC and system perspective, we believe that *ZeRO* represents a revolutionary transformation in the large model training landscape. While our implementation, *ZeRO*-100B, enables 8x increase in model sizes, over 10x in throughput improvement, achieves super-linear speedups on modern GPU clusters, and trains the largest model in the world, it is still just a tip of the iceberg. *ZeRO* in its entirety has the potential to increase the model size by yet another order of magnitude, enabling the training of trillion parameter models of the future.

Perhaps, what we feel most optimistic about *ZeRO* is that it imposes no hurdles on the data scientists. Unlike existing approaches such as MP and PP, no model refactoring is necessary, and it is as easy to use as standard DP, making *ZeRO* a prime candidate for future investigations on large model training. Through open sourcing and community feedback, we plan to make *ZeRO* fully accessible to the DL community to catalyze the evolution and democratization of large model training at scale.

## Acknowledgement

We thank Junhua Wang for his valuable support and advice. We thank Minjia Zhang, Elton Zheng, Shaden Smith, Reza Yazdani Aminabadi, Arash Ashari, and Niranjan Uma Naresh for their great feedback and help on evaluating the work. We thank Brandon Norick, Corby Rossett, Gopi Kumar, Jack Zhang, Jing Zhao, Payal Bajaj, Rangan Majumder, Saksham Singhal, Saurabh Tiwary, and Xia Song for many helpful discussions and suggestions.

<span id="table-05"></span>

![Original paper Table 5](../../papers/zero/table-05.png)

**Table 5.** Model configurations for [Figure 2](#figure-02) related to ZeRO throughput compared with baseline.

<span id="table-06"></span>

![Original paper Table 6](../../papers/zero/table-06.png)

**Table 6.** Model configurations for [Figure 3](#figure-03) related to superlinear scalability.

<span id="table-07"></span>

![Original paper Table 7](../../papers/zero/table-07.png)

**Table 7.** Model configurations for [Figure 4](#figure-04) related to max model size with different ZeRO configurations.

<span id="table-08"></span>

![Original paper Table 8](../../papers/zero/table-08.png)

**Table 8.** Model configurations for [Figure 5](#figure-05) related to memory allocated with different ZeRO configurations.

<span id="table-09"></span>

![Original paper Table 9](../../papers/zero/table-09.png)

**Table 9.** Model configurations for [Figure 6](#figure-06) related to throughput with different ZeRO configurations.

.

These tables contain all the model configurations and batch sizes used for the experiments presented in the paper. In [Figure 2](#figure-02), notice that the total number of GPUs for some baseline experiment is 384 or 256 compared to 400 for ZeRO. This is because the total number of GPUs must be a product of the number of MP, and we only had access to a total of 400 GPUs. There exist a handful of additional constraints in model configuration values, such as hidden size must be divisible by attention heads, hidden size divisible by MP, and attention heads divisible by MP. For baseline we used the lowest number of GPUs that was a power of 2 that would fit the model. So for example, for 170B parameter model this was 256 for the baseline. Since we only had 400 GPUs, we could only run baseline with 256 GPUs.

We do want to point out that this gives the baseline an advantage over ZeRO because fewer GPUs means better communication throughput for the baseline. For example, in case of the 170B parameter model, DP=1 for the baseline so it in fact incurs no communication for DP. The results presented in this paper are despite this advantage for the baseline.

Also, we want to point out that we are comparing the performance per GPU, not the aggregate performance, and therefore the results are still apples-to-apples while giving a slight advantage to the baseline.

<span id="table-10"></span>

![Original paper Table 10](../../papers/zero/table-10.png)

**Table 10.** Model configurations for [Figure 7](#figure-07) related to evaluating maximum model sizes vs throughput while using only data-parallelism.

[+1]: Equal Contributors

[+2]: Prior work [Mcc18] shows, very large batch size could slow down convergence. For given model and data, there is a measure of critical-batch size, where increasing batch size further slows down convergence. The detailed discussion of this topic is beyond the scope of the paper.

[+3]: https://github.com/microsoft/deepspeed

[+4]: The activation memory of a transformer-based model is proportional to the number of transformer layers $\times$ hidden dimensions $\times$ sequence length $\times$ batch size. For a GPT-2 like architecture the total activations is about $12\times\mathit{hidden\_dim}\times\mathit{batch}\times\mathit{seq\_length}\times\mathit{transformer\_layers}$.

[+5]: https://github.com/nvidia/Megatron-LM

[+6]: Increasing batch size too much can lead to poor convergence, but for these large models, we are still in a regime where batch size is small enough even with 1K GPU and it does not affect convergence rate
