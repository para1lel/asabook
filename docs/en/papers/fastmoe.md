---
title: 'FastMoE: A Fast Mixture-of-Expert Training System'
createTime: 2026/09/07 21:06:24
permalink: /en/papers/fastmoe/
---

> [Jiaao He](https://dblp.org/pid/249/2660), [Jiezhong Qiu](https://jiezhongqiu.com/), [Aohan Zeng](https://blog.sengxian.com/), [Zhilin Yang](https://kimiyoung.github.io/), [Jidong Zhai](https://pacman.cs.tsinghua.edu.cn/~zjd/), and [Jie Tang](https://keg.cs.tsinghua.edu.cn/persons/jietang/). First submitted to arXiv on March 24, 2021; current version v1. [FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262). <a href="/paper/fastmoe.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2103.13262). [TeX source](https://export.arxiv.org/e-print/2103.13262). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Mixture-of-Expert (MoE) presents a strong potential in enlarging the size of language model to trillions of parameters. However, training trillion-scale MoE requires algorithm and system co-design for a well-tuned high performance distributed training system. Unfortunately, the only existing platform that meets the requirements strongly depends on Google's hardware (TPU) and software (Mesh Tensorflow) stack, and is not open and available to the public, especially GPU and PyTorch communities.

In this paper, we present *FastMoE*, a distributed MoE training system based on PyTorch with common accelerators. The system provides a hierarchical interface for both flexible model design and easy adaption to different applications, such as Transformer-XL and Megatron-LM. Different from direct implementation of MoE models using PyTorch, the training speed is highly optimized in *FastMoE* by sophisticated high-performance acceleration skills. The system supports placing different experts on multiple GPUs across multiple nodes, enabling enlarging the number of experts linearly against the number of GPUs. The source of *FastMoE* is available at [https://github.com/laekov/fastmoe](https://github.com/laekov/fastmoe) under Apache-2 license.

<span id="section-1"></span>

## 1 Introduction

Recent emergence of large-scale language models, examplified by BERT [Dev18], GPT-2/-3 [Rad19b, Bro20b], XLNet [Yan19], RoBERTa [Liu19a], T5 [Raf20b], GShard [Lep20] and Switch Transformer [Fed22], has drastically reshaped the landscape of the natural language processing research, reestablishing the new state-of-the-art baselines in various benchmarks such as GLUE [Wan18d] and SuperGLUE [Wan19h].

Among many possible solutions, scaling model size has been proved to be one of the simplest and most effective way [Kap20] toward more powerful models. From BERT [Dev18] with $340$ million parameters, to T5 [Raf20b] with $11$ billion parameters, to GPT-3 [Bro20b] with $175$ billion parameters, the model size is enlarged by $500\times$ in merely two years. More recently, GShard [Lep20] scales to a record number of $600$ billion parameters, which is quickly broken by Switch Transformer [Fed22] with $1.6$ trillion parameters. The main contributor toward the huge model size of GShard and Switch Transformer is a novel neural network architecture named mixture of experts (MoE) [Sha17].

<span id="figure-01"></span>

![An illustrative example of an MoE layer](../../papers/fastmoe/figure-01.png)

**Figure 1.** An illustrative example of an MoE layer. In this example, expert 1 and expert 3 are selected by the gate for computation.

An MoE layer (an illustrative example can be found in [Figure 1](#figure-01)) consists of a gate and a pool of experts. For each input, only a tiny minority of experts are chosen by the gate for computation. The special architecture of MoE is a double-edge sword for large-scale distributed training. On the one hand, due to its sparse activation of experts, MoE is able to enlarge the model size by orders of magnitude without significantly increasing the amount of computation (FLOPs). On the other hand, when scaling up to thounds of experts, the imbalanced all-to-all communication pattern of MoE brings new challenges to the co-design of algorithm and system. Therefore, MoE can not be directly supported by traditional deep learning libraries such as PyTorch [Pas19] and TensorFlow [Aba16].

Due to the challenge raised by the new model architecture, both the research community and the industry need an MoE implementation that support large-scale distributed training. However, despite the existence of some naive single-GPU implementation in PyTorch [Rau19], the only present system that supports scalable MoE training is based on Google's private hardware and software stack—TPU [Jou17a] and Mesh TensorFlow [Sha18a]. Thus, there is urgent need to develope an MoE system on publicly available hardware (e.g., GPU) and platforms (e.g., PyTorch [Pas19]).

Motivated by the desire to obtain easy-to-use, flexible, efficient, scalable, and open-source solution to large-scale MoE training, we release *FastMoE* with the following design goals:

- **Easy-to-use:** provide a user-friendly interface to define an MoE layer, and seamless support for popular language model training system, Megatron-LM [Sho19].
- **Flexible:** make it easy for users to customize gate networks and expert networks.
- **Efficient:** integrate a highly optimized feadforward (FFN) layer for Transformer.
- **Scalable:** support scaling up the size of MoE models by training across multiple GPUs on multiple nodes.

Different from previous single-GPU PyTorch implementation [Rau19], *FastMoE* concentrates on efficiency and scalability. Dedicated CUDA kernels are included in *FastMoE* for high performance with specialized optimizations. *FastMoE* is able to run across multiple GPUs on multiple nodes using NCCL [Jea17]. The details of communication is hidden from model developers by *FastMoE*. The model-parallel method of *FastMoE* allows distributing experts across different GPUs, while other parts of the model remain parallelized by batch dimension (data parallel) or tensor dimension (model parallel). Chances are that the model size, proportional to the number of experts, can scale up with the number of GPUs used for training, being the key to train trillion-scale models.

In our experiment, we observe that *FastMoE* is faster than a baseline [Rau19] implemented by pure PyTorch API on a single GPU. *FastMoE* also shows reasonable scalability when running across nodes on a cluster connected by Infiniband network. We train a real GPT model with $96$ experts per layer using distributed *FastMoE* with promising end-to-end training speed. Compared to a non-MoE model of the same amount of computation, its performance benefits from the enlarged model size that the MoE architecture.

This paper is organized as follows. [Section 2](#section-2) introduces the background of MoE and compares existing systems. [Section 3](#section-3) presents the *FastMoE* system in detail. [Section 4](#section-4) introduces the challenges of achieving high-performance and the *FastMoE*'s solutions. [Section 5](#section-5) shows results of experiments that demonstrate the efficiency of *FastMoE* and the performance gain of an MoE model using *FastMoE* in training. [Section 6](#section-6) summarizes the paper and indicates directions our future work.

<span id="section-2"></span>

## 2 Mixture-of-Experts (MoE)

In this section, we review the architecture of MoE and current systems for training MoE.

<span id="section-2-1"></span>

### 2.1 MoE: Model Structure

Mixture-of-Expert is short for Sparsely-Gated Mixture-of-Experts layers proposed by [Sha17]. An MoE layer consists of multiple experts, each can be an arbitrary neural network. The only constraint of the experts is that they should take the same input, and give output in the same vector space. [Figure 1](#figure-01) shows a detailed example of an MoE layer. A special neural network, namely *the gate network*, is introduced to score each expert over a given input. According to the scores, selection of experts is made by a policy, which may vary from model to model. Then, the selected experts, e.g. experts $1$ and $3$ in the example, are activated to process the input sample. The outputs of the experts, together with the score, are combined into the final output using a certain algorithm.

A popular way of the expert selection is to select the experts with top $k$ highest score. In synthesis process, the score is used as the weight for the output of the experts to be added into the overall output. This enables training the gate network, as the gradient can be propagated by the score. [Algorithm 1](#algorithm-01) formalizes the method above.

<span id="algorithm-01"></span>

**Algorithm 1: Forward computation of an MoE layer with top-$k$ gating.**

- **Require:** A pool of $n$ experts: $\{E_1,E_2,\cdots,E_n\}$.
- **Require:** Gate $G$.
- **Require:** The number of experts $k$ to be selected.
- **Function** $\operatorname{MoE}(x)$:
  - $\mathit{score}\leftarrow G(x)$.
  - $\mathit{indices}\leftarrow \operatorname{ArgMax}_k(\mathit{score})$.
  - $y\leftarrow$ zero tensor like $x$.
  - **For** each index $i\in\mathit{indices}$:
    - $x_i\leftarrow E_i(x)$.
    - $y\leftarrow \mathit{score}_i*x_i+y$.
  - **Return** $y$.

<span id="section-2-2"></span>

### 2.2 Current Systems for MoE Training

The GShard system [Lep20] implements a distributed version of the MoE model. It trains a language model on up to $2048$ TPUs, with $1$ expert per layer placed on each TPU. As a result, the MoE layers contain $2048\times$ more parameters than a non-MoE layer. In Switch Transformer [Fed22], the model is further enlarged to $1.6$ trillion, showing strong ability for the system to support training models in large scale. Unfortunately, this system is not publicly available yet. It is strongly binded with the TPU cluster, which makes it hard to reproduce the experiments on commodity devices. Additionally, the design of GShard lacks flexibility to use different number and size of experts with different replication strategy.

In Tensor2tensor [Vas18c], a MoE Transformer model is provided. However, this implementation uses Mesh TensorFlow [Sha18a], which does not support GPUs very well. To implement a FFN in Transformer, it takes more than $100$ lines of code in TensorFlow with complicated `einsum` operators, making it burdening for developers to understand the structure and explore other model structures based on the code.

PyTorch [Pas19], as a popular deep learning framework among researchers, provides more straightforward coding style and flexibility against TensorFlow. Efforts are made to train MoE models with PyTorch [Rau19]. However, as the PyTorch community lacks multi-dimension parallel training tools, none of the PyTorch-based implementations support training on multiple GPUs. As the ultimate target of adopting MoE is training even larger models, the PyTorch-based implementations fails to be a candidate.

<span id="section-3"></span>

## 3 FastMoE: System Design

In this section, we introduce our design of *FastMoE* with distributed training support.

<span id="section-3-1"></span>

### 3.1 A Flexible System for Diverse Model Explorers

**The Backbone to Run Arbitrary Expert Networks.** *FastMoE* supports using arbitrary network as the expert. The `FMoE` interface of *FastMoE* takes any neural network module constructor as input, and replicates the module for multiple times as the expert instances. The expert is defined to take a batch of aligned contiguous input features, and the output should be in the same batch order. Therefore, the expert module implementation is decoupled from the MoE architecture so that developers can focus on the design of their own expert network.

For even stronger flexibility, the `FMoE` class contains a member function `expert_fn`, where the expert modules are used to conduct the forward computation. This function can be overloaded for further customized MoE behavior. For example, in the `FMoETransformerMLP` network, which will be mentioned later in this section. the list of experts is replaced by a specially optimized module that applies the experts in parallel to extremely lower the latency.

Moreover, *FastMoE* supports placing multiple experts together on the same worker, enabling more flexible configuration space of the number of experts (i.e., the number of experts does not have to be equal to the number of data parallels), which is different from the the design of GShard.

**A Highly-optimized FFN for Transformer.** To better support training Transformer with MoE, *FastMoE* provides a standard and high-performance FFN implementation (`FMoETransformerMLP`). The detailed optimization strategy is hidden from the developers.

In particular, when placing multiple experts on the same worker, a naive implementation is to loop over these experts and conduct forward in sequence. However, for certain types of expert networks, it is possible to explore the potential speedup brought by parallel execution. In *FastMoE*, we mainly optimize the parallel execution of fully-connected layers by a dedicated `FMoELinear` module. Instead of computing the expert modules sequentially, the specially optimized expert module maintains a pool of available hardware resources, and applies the expert computation in parallel.

**Plugin-style Support for PyTorch and Megatron-LM.** The flexibility of *FastMoE* allows convenient adaption to existing training applications. Take Megatron-LM [Sho19] as an example, a plugin-style module is integrated in *FastMoE* to quickly replace the FFNs in the original Megatron-LM model with MoE networks. As shown in listing 1, the transformation can be achieved by only 2 lines of code.

**Listing 1: Sample code to use FastMoE in Megatron-LM.**

```python
from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=<number of experts per worker>)
```

The `fmoefy` function can find the FFNs in the Transformer layers. Then, an MoE network using *FastMoE* is created, which is a module that wraps up the `FMoETransformerMLP` module for interface-level compatibility.

<span id="section-3-2"></span>

### 3.2 Enlarging the Model Capacity Distributedly

**The Model Parallel Method of *FastMoE*.** As one of the most effective way to enlarge the model capacity, the ability to accommodate a large expert population and train them in parallel is demanded in many MoE models. It is hard for the model developers to handle the complicated data transfer among GPUs and even across nodes. Achieving high training performance and good hardware resource utilization requires expertise in computer architecture and parallel programming, which is beyond the technique stack of common model developers.

*FastMoE* supports distributing experts across multiple workers on multiple nodes, which is called *the model parallel method in FastMoE*. The detail of input data exchange is hidden within the `FMoE` interface. For model developers, they only need to write code for a single expert, and each expert is given all the input data gathered from all workers by *FastMoE*. As a result, the model developers do not have to consider the implementation details about cross-worker communication.

In the design of *FastMoE*, when the feature to distribute expert across workers is enabled, extra communication operations are included in the forward and backward computation. To better identify the operations, we call them global data exchange operations, in contrast to the local data shuffle process, which will be mentioned in [Section 4](#section-4).

A major challenge in the distributed context is that the total number of input samples assigned to all experts on a worker may vary a lot. It is impossible to have the number of incoming samples before the gate output is available. However, allocating the buffer to place the input samples is dependent on the number. Therefore, before actual exchange of input samples between workers happens after exchanging the amount information between workers, and allocating memory according to the inspection of the expert count information.

<span id="figure-02"></span>

![An example of the global operations](../../papers/fastmoe/figure-02.png)

**Figure 2.** An example of the global operations.

An example of the global operations in *FastMoE* is shown in [Figure 2](#figure-02). The workers first count the number of samples assigned to each expert on each worker. Then, they exchange the size of expert inputs, so that all workers get the number of incoming input samples, and where they are from. After the offsets of each receiving buffer is calculated, the workers start exchanging data directly. It is worth nothing that the statistics of the incoming and outgoing samples can be reused through the whole process of a training iteration.

**Heterogeneity-aware Synchronization Module.** Heterogeneity is introduced as different parts of the network may be replicated across different groups of workers. It is a challenge that the distributed module has to identify whether the gradient of a parameter should be synchronized, and with whom it is synchronized. *FastMoE* introduces the *data parallel communication group* tag on each parameter to address the issue.

The tag can be one of `world`, `data parallel` or `none`, which respectively indicates that the gradient should be synchronized with (1) all other workers, (2) the workers in a data-parallel group that is orthogonal to the model-parallel group, or (3) no worker. For example, the gate network is replicated across all workers, regardless of model parallel settings. The attention layer may be divided into model-parallel sub-layers, so its tag is `data parallel`. Each worker serves several unique expert networks, whose tag is `none`. A customized data parallel module instead of PyTorch's original distributed data parallel module is provided by *FastMoE*, which can identify the tags and perform correct synchronization.

<span id="section-4"></span>

## 4 Optimizations to Achieve High-performance

The performance of MoE computation on a single node is significant, as it determines the theoretical upper bound of the system scaling up to any scale.

The most intuitive way to compute an MoE layer is slicing the input batch into samples, and compute sample by sample. After that, output features are stacked in the original order. However, it is observed that implementing an MoE model using simple PyTorch operators can hardly achieve high performance. Less than $5\%$ the peak performance of GPUs can be achieved.

<span id="figure-03"></span>

![GeMM performance at different problem sizes](../../papers/fastmoe/figure-03.png)

**Figure 3.** `GeMM` performance of different problem sizes using `cuBLAS` on NVIDIA V100.

Without loss of generality, we assume the expert network is an FFN. Note that the major operator within an FFN is from fully-connected layers, which consist of several `GeMM` operators. When the batch is split up into single samples, the `GeMM` operation is degraded into `GeMV`. [Figure 3](#figure-03) shows the float-point computation throughput of an example fully-connected layer using different batch size. Given that in modern heterogeneous compute devices, matrix multiplication operators are fine tuned with sophisticated tiling techniques applied on all dimensions, it is not surprising that the throughput can only approach the theoretical peak when the batch size is large enough. This leads to the principle that to achieve high performance in MoE computation, the samples should be batched to fully utilize the hardware resources.

*FastMoE* batches all input samples to the same expert together. Due to the limit of data representation, *FastMoE* performs memory movement with a specially developed CUDA kernel to reduce overhead. Given the index of gate that each sample is going to, the process to put all input samples to the same gate in a contiguous memory space is called `scatter`. However, in other parts of the neural network, the batch may have to be organized as its original order, e.g., the attention layer in Transformer. A reverse operation is performed after the experts output to another contiguous memory space, i.e. place the scattered feature vectors back to their original order according to the gate indices. This process is denoted as `gather` in *FastMoE*.

<span id="figure-04"></span>

![An example of the reordered computation of an MoE layer](../../papers/fastmoe/figure-04.png)

**Figure 4.** An example of the reordered computation of an MoE layer

The reordering computation process is shown as [Figure 4](#figure-04). When the assignment from input samples to experts is balance enough, each expert is expected to have a relatively large input batch size that can reach satisfying hardware utilization according to [Figure 3](#figure-03). However, load imbalance always occurs because of the nature of random sampling of the input training data. It is highly possible that one expert receives very few input samples during millions training iterations. Additionally, as multiple experts are placed on one worker, local batch sizes of the experts are, on average, statistically lower than that in data parallel. *FastMoE* uses a customized stream manager to simultaneously execute the computation of multiple experts to extract the potential throughput gain.

<span id="section-5"></span>

## 5 Evaluation

In this section, the training speed of *FastMoE* is compared with another PyTorch MoE implementation [Rau19] on a single GPU. We also report the scalability of *FastMoE* when distributed training. To the best of our knowledge, *FastMoE* is the only PyTorch-based MoE system can run across different nodes and GPUs. We also show the end-to-end performance of an MoE Transformer model trained using *FastMoE*.

<span id="section-5-1"></span>

### 5.1 Experiment Setup

We use the following notations to characterize the computation task: $n_e$ experts are placed on each GPU. Each expert applies two linear layers of sizes $d_m\times d_h$ and $d_h\times d_m$ respectively. The input contains $n_b$ samples. The gate module scores the fitness of each sample to be processed by each expert. For each input sample, the experts of top $k$ highest score are selected to process the sample.

Additionally, several warm-up rounds are performed, which perform the same computation but are not counted in the results. For each experiment, the task is executed $16$ times, and the average time is used to calculate the performance. The standard deviation values of the execution time are also inspected. All of them are negligible.

<span id="section-5-2"></span>

### 5.2 Training Speed on a Single GPU

The performance of the `FMoETransformerMLP` is tested, which completes similar task with `MoE` module in the baseline [Rau19], on a NVIDIA TESLA V100 PCIe GPU. The baseline is implemented by pure PyTorch API with hard-coded model structure. For fairness of the comparison, both modules uses a randomly initialized matrix as the weight of the gate network, which consists of one fully-connected layer. The experts also perform the same computation.

<span id="figure-05"></span>

![Computation time comparison between FastMoE and the baseline](../../papers/fastmoe/figure-05.png)

*The latency is tested with $n_b=4096,d_m=1024,d_h=4096,k=2$.*

**Figure 5.** Computation time comparison between *FastMoE* and the baseline implementation.

As [Figure 5](#figure-05) shows, the baseline implementation is constantly slower than *FastMoE*. As the number of experts grows, the baseline spends much more time in the forward computation, while the latency of *FastMoE* remains stable, thanks to its customized stream manager mentioned in [Section 4](#section-4). Considering that *FastMoE* is targeted on training, the backward time is stacked over the forward time. We observed that *FastMoE* outperforms the baseline in the overall time spent in each iteration.

<span id="section-5-3"></span>

### 5.3 Cross-GPU and Cross-node Scalability

To examine the performance of *FastMoE* expanding on multiple GPUs across nodes, we conduct an experiment on a cluster of $8$ nodes, with $1$ NVIDIA Tesla V100 GPU on each node. The cluster is interconnected via an Infiniband EDR switch and $8$ HCA cards. The FLOPs of the matrix multiplication operations is calculated to represent the training throughput.

<span id="figure-06"></span>

![Scalability of FastMoE across GPUs and nodes](../../papers/fastmoe/figure-06.png)

*The throughput is tested with $n_e=4,n_b=4096,d_m=1024,d_h=4096,k=2$.*

**Figure 6.** Scalability of *FastMoE* across multiple GPUs on multiple nodes

According to the result shown in [Figure 6](#figure-06), *FastMoE* shows scalability across nodes. The overall throughput increases from $10$ `TFLOPs` to $25$ `TFLOPs`, as the number of GPUs increases from $2$ to $8$, sub-linearly scaling up. We observe that when expanding to $2$ GPUs, the performance is half of that on a single GPU, which suggests that *FastMoE* is bounded by communication. When more GPUs are used for computation, more experts are introduced, and the granularity of exchanging input samples becomes smaller, lowering the efficiency in data transfer over the network.

As a conclusion, the scalability of *FastMoE* can support training large MoE model using multiple GPUs across multiple nodes with performance gain. However, space is left for further optimization on throughput.

<span id="section-5-4"></span>

### 5.4 End-to-end Performance Gain Using FastMoE

We test the end-to-end performance gain using *FastMoE* by training a 12-layer GPT model on $8$ GPUs using Megatron-LM [Sho19]. As mentioned in [Section 3](#section-3), the Megatron adapter of *FastMoE* is used for MoE structure. For each layer, $96$ experts are distributed across the GPUs, i.e. $12$ experts are placed on each GPU. For each input token, the top $2$ experts with highest score are used to process it. The $d_h$ in expert MLP layer is halved so that the valid FLOPs of the model are almost identical, except for the extra FLOPs introduced by the gate, which is negligible. Both the baseline model and the MoE model are trained for $70$ hours. The `lm loss` metric in training indicates the convergence tendency of the models.

<span id="figure-07"></span>

![Loss curves for a GPT model trained with FastMoE](../../papers/fastmoe/figure-07.png)

*The narrow dark lines are smoothed exponentially by $0.97$ from the original loss curve, represented by the brighter wide curves respectively.*

**Figure 7.** Loss curve of training a GPT model by *FastMoE*

From [Figure 7](#figure-07), we observed that the training speed of the baseline model is about $3\times$ of *FastMoE*. As *FastMoE* performs more computation and communication, it is a reasonable slow down. Fortunately, the MoE model achieves much lower loss with the same training iterations. Also, as a benefit from the efficiency of *FastMoE*, the MoE model achieves lower loss in the same training time.

<span id="section-6"></span>

## 6 Summary and Future Work

In this paper, we present *FastMoE*, an open-source system to train Mixture-of-Experts models. The system is based on the popular PyTorch framework, and currently supports efficient training on GPUs. Friendly interfaces of multiple levels are provided for different users to explore different aspects of the MoE architecture. The performance of *FastMoE* on a single GPU is well-optimized to exploit the power of GPUs. *FastMoE* can also run across GPUs on multiple nodes with reasonable scalability, enabling further enlarging model size. Real model performance advantage is observed in our end-to-end model training experiment using *FastMoE*.

We are still working on *FastMoE* for more features and faster training. Compared to the GShard model [Lep20], *FastMoE* lacks functionalities to support load-balancing among experts. The work of load-balance monitor and support for load-balance loss is in progress. We are also trying to make the system more user-friendly on utilities, such as loading and saving of MoE models. The performance across multiple GPUs requires joint efforts from the view of both high-performance computing and machine learning. Any contributions to the open-source project will be appreciated. We are looking forward to your participation.
