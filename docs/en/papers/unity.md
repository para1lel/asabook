---
title: 'Unity: Joint DNN Training Optimization'
createTime: 2026/09/11 18:33:53
permalink: /en/papers/unity/
pageClass: paper-reading
---

> [Colin Unger](https://dblp.org/pid/294/3598.html) [+equal], [Zhihao Jia](https://www.cs.cmu.edu/~zhihaoj2/) [+equal], [Wei Wu](https://dblp.org/pid/95/6985-16.html), [Sina Lin](https://dblp.org/pid/116/4935.html), [Mandeep Baines](https://dblp.org/pid/277/0899.html), [Carlos Efrain Quintero Narvaez](https://dblp.org/pid/302/0542.html), [Vinay Ramakrishnaiah](https://dblp.org/pid/231/3756.html), [Nirmal Prajapati](https://dblp.org/pid/169/1015.html), [Pat McCormick](https://dblp.org/pid/147/4103.html), [Jamaludin Mohd-Yusof](https://dblp.org/pid/69/5800.html), [Xi Luo](https://dblp.org/pid/69/7449.html), [Dheevatsa Mudigere](https://dblp.org/pid/87/8721.html), [Jongsoo Park](https://dblp.org/pid/13/9348.html), [Misha Smelyanskiy](https://dblp.org/pid/22/4090.html), and [Alex Aiken](https://theory.stanford.edu/~aiken/). Published in the [16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2022)](https://www.usenix.org/conference/osdi22/presentation/unger), July 11-13, 2022, pp. 267-284. [Unity: Accelerating DNN Training Through Joint Optimization of Algebraic Transformations and Parallelization](https://www.usenix.org/conference/osdi22/presentation/unger). <a href="/paper/unity.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. No arXiv record or TeX source is available; the published PDF remains authoritative for the exact wording, print layout, and bibliography.

## Abstract

This paper presents Unity, the first system that jointly optimizes algebraic transformations and parallelization in distributed DNN training. Unity represents both parallelization and algebraic transformations as substitutions on a unified *parallel computation graph* (PCG), which simultaneously expresses the computation, parallelization, and communication of a distributed DNN training procedure.

Optimizations, in the form of graph substitutions, are automatically generated given a list of operator specifications, and are formally verified correct using an automated theorem prover. Unity then uses a novel hierarchical search algorithm to jointly optimize algebraic transformations and parallelization while maintaining scalability. The combination of these techniques provides a generic and extensible approach to optimizing distributed DNN training, capable of integrating new DNN operators, parallelization strategies, and model architectures with minimal manual effort.

We evaluate Unity on seven real-world DNNs running on up to 192 GPUs on 32 nodes and show that Unity outperforms existing DNN training frameworks by up to $3.6\times$ while keeping optimization times under 20 minutes. Unity is available to use as part of the open-source DNN training framework FlexFlow at [https://github.com/flexflow/flexflow](https://github.com/flexflow/flexflow).

<span id="section-1"></span>

## 1 Introduction

Deep neural networks (DNNs) are becoming progressively larger and computationally more expensive to train, and as they have grown, so has interest in optimizing their execution to reduce training times and improve scalability. Two key classes of optimizations shown to yield significant performance improvements across diverse model architectures are algebraic transformations and parallelization.

Algebraic transformations exploit operator identities to perform the underlying computation in a more efficient way, but ignore parallelization and distribution of training. Common examples of algebraic transformations include operator fusion, which merges two operators into a single semantically equivalent operator whose computation is more efficient, and operator reordering, where the associativity or commutativity of sets of operators allows them to be reordered into more efficient configurations or to expose further optimization opportunities. More explanation of algebraic transformations, along with examples, is provided in [Section 2.2](#section-2-2).

Parallelization, in contrast, distributes operators over multiple devices, but does not change the way in which the underlying computation is performed. DNN training exploits a class of parallelism named *partition-n-reduce* [Wan19b], in which every distributed subcomputation of an operator must perform the same computation, and may only differ in the input data it consumes. The tensor computations in DNN training are particularly well-suited to this form of parallelism, and many parallelism dimensions along which to divide distributed operators have been identified, such as data [Aba16], model [Dea12], spatial [Jia19a], reduction [Sho19], and pipeline [Nar19]. For a detailed overview of these various approaches, see [Section 2.1](#section-2-1).

When applied effectively, these two techniques can improve training times by more than an order of magnitude. However, effective application is nontrivial. Rewriting the computation graph for maximum speedup can require many transformations, some of which may harm performance except in the context of a longer sequence of transformations [Jia19]. The optimal parallel execution strategy for a model often requires simultaneously exploiting multiple parallelization dimensions and using different parallelization schemes for each operator [Jia18]. Early work relied on the programmer to manually determine the correct optimizations to apply [Aba16]. While manual optimization allows fine-grained control over the model's performance, it requires many hours of tuning by experts to achieve good performance. As the pace of new developments in model design has increased, manual optimization has struggled to scale beyond the most commonly used models.

<span id="figure-01"></span>

![Computation graph for a two-layer MLP](../../papers/unity/figure-01.png)

**Figure 1.** Computation graph for a 2-layer MLP.

Recent work has focused on automating optimizations. MetaFlow [Jia19], TASO [Jia19b], and PET [Wan21] propose algorithms for automatically generating and applying algebraic transformations by posing optimization as a search problem. FlexFlow [Jia19a], automap [Sch21c], Tofu [Wan19b], and Whale [Jia21c] bring a similar approach to parallelism. These works present impressive benchmarks, yielding the impression that automating algebraic and parallelization optimization is a solved problem.

However, to reduce training time as much as possible, we want to apply both of these optimizations, but the most effective way to combine algebraic and parallelization optimizations is not obvious. The simplest solution is to apply them independently, in one of two orders: algebraic optimization followed by parallelization, or the reverse. The reverse order turns out to be problematic: since algebraic transformations can introduce new operations or replace existing ones, running algebraic optimization after parallelizations have been assigned can lead to the final solution having operations without assigned parallelizations (if the operation was created) or invalid parallelizations (if the operation was replaced). Workarounds can be used to fix invalid solutions by using default parallelization strategies or copying the strategies of nearby operators, but it is easy to find cases in which these workarounds lead to suboptimal solutions. As such, applying algebraic optimization before parallelization is the only option, but as we see in the next example, it can miss significant optimization opportunities.

Consider the computation graph shown in [Figure 1](#figure-01), which represents a 2-layer multilayer perceptron (MLP). If we are optimizing independently (also referred to as "sequentially"), we start by applying algebraic transformations without considering parallelism. A typical algebraic optimizer will fuse the MatMul and ReLU operators to remove redundant memory loads and stores. The model is then parallelized (we consider only 2 GPUs for simplicity) resulting in [Figure 2a](#figure-02): data parallelism is used for both operators and thus the weight gradients must be synchronized with an AllReduce. Since weight 1 has size $c_i h$ and weight 2 has size $hc_o$, the total communication is $2(c_i h+hc_o)$. Using a set of parameters for a basic image classification model for MNIST ($b=64$, $h=512$, $c_o=10$, $c_i=28\times 28=784$) yields a total communication of $813,056d$ bytes, where $d$ is the element size.

<span id="figure-02"></span>

![Sequential and joint optimization of a two-layer MLP on two GPUs](../../papers/unity/figure-02.png)

**Figure 2.** Comparing joint and sequential optimizations.

Instead of independently applying algebraic transformations and parallelization, we can combine them and solve a single joint optimization problem that discovers the solution in [Figure 2b](#figure-02). By not fusing the first MatMul and ReLU, more efficient reduction parallelism can be used. This requires synchronizing the activation and gradient of the first MatMul's output, but not the weights: a total inter-GPU communication of $4bh$, or $131,072d$ bytes for our MNIST example. Joint optimization reduces communication by $6\times$, which far exceeds the cost of not fusing the first ReLU.

As this example shows, joint optimization is necessary to maximize performance. However, it also poses significant challenges. The first is representation: existing frameworks perform optimizations on a model's computation graph. As discussed above, algebraic transformations can leave operators in the computation graph with unassigned or invalid parallelizations. To prevent such invalid solutions from arising during search, we need a representation that allows algebraic transformations to consider the current parallelization before being applied. Further discussion of the representation challenges is in [Section 3.4](#section-3-4).

The second challenge is scalability: existing search-based approaches already struggle to scale up to large models and GPU counts. Improvements have been made for algebraic transformations alone [Yan21d], but the complexity of these solutions makes adding parallelization a daunting task. Simultaneously considering both optimization classes only exacerbates this problem by exponentially increasing the search space size. For joint optimization to be practical, search algorithms must improve on the scalability of past techniques.

<span id="section-1-1"></span>

### 1.1 Unity's Approach

The key idea behind Unity is to represent both algebraic transformations and parallelization as graph substitutions on a unified parallel computation graph, and then to use a hierarchical search algorithm to efficiently identify which combination of substitutions yields the best performance. [Figure 3b](#figure-03) shows an overview of Unity, which differs from existing frameworks in the following ways:

<span id="figure-03"></span>

![Existing DNN optimization flow and the unified Unity flow](../../papers/unity/figure-03.png)

**Figure 3.** Comparing existing DNN frameworks and Unity.

**Unified graph representation.** We introduce the parallel computation graph (PCG) [+1] as a unified representation of distributed DNN training that simultaneously expresses computation, parallelism, and data movement. All parallelization strategies used in existing frameworks can be represented as specific PCGs, and parallelization and algebraic transformations as sequences of graph substitutions. pONNX [Wan19j] previously proposed merging computation and parallelism into a single graph, but certain design decisions prevent Unity-style joint optimization. For a detailed comparison, see [Section 3.4](#section-3-4).

**Transformation generation and verification.** Unity does not require users to explicitly define possible parallelization strategies for DNN training. Unlike prior work that automatically generates parallelization strategies [Wan19b] or algebraic transformations [Jia19b], by using the PCG Unity is able to generate both kinds of transformations with a single approach, as well as hybrid algebraic-parallelization optimizations absent in prior automated approaches. Automatically generating and verifying transformations greatly reduces the engineering effort required to support different parallelism dimensions and enables extensibility to new operators.

**Joint optimization.** Unity uses a hierarchical search algorithm to discover highly optimized PCG substitutions and device placements while maintaining scalability to models with hundreds of operators distributed over hundreds of GPUs. Unity's cost model includes both computation and communication time, and the search algorithm handles custom network topologies and heterogeneous compute devices. Despite the exponentially larger search space being considered, Unity outperforms existing search-based approaches (see [Section 6](#section-6)). The rest of this paper provides additional background ([Section 2](#section-2)), discusses Unity's design and implementation ([Section 3](#section-3), [Section 4](#section-4), and [Section 5](#section-5)), and evaluates its performance on seven real-world DNNs ([Section 6](#section-6)). For widely-used DNNs highly optimized by existing frameworks, such as BERT [Dev19], Unity matches the performance of existing expert-designed strategies while being completely automated. For complex DNN architectures with a mixture of compute- and communication-intensive operators, such as DLRM [Nau19] and CANDLE-Uno [Can18], Unity is up to $3.6\times$ faster than existing frameworks.

<span id="section-2"></span>

## 2 Background

We first provide a brief overview of the two classes of optimizations that Unity exploits, parallelization ([Section 2.1](#section-2-1)) and algebraic transformations ([Section 2.2](#section-2-2)), as well as a discussion of how they are represented in existing systems ([Section 2.3](#section-2-3)). For a discussion of how Unity interacts with other classes of optimizations, see [Section 8](#section-8).

<span id="section-2-1"></span>

### 2.1 Parallelization

The massively parallel nature of tensor algebra creates many opportunities for parallelizing DNN training. We identify six primary forms of parallelism leveraged in DNN systems:

1. *Data parallelism* is the most common approach used in existing frameworks [Aba16, Che15b, Pyt17]. Data parallelism keeps a replica of the entire DNN model on every device and assigns each a subset of the training data.
2. *Model parallelism* divides a DNN model into disjoint submodels and trains each sub-model on a dedicated device.
3. *Spatial parallelism* [+2] divides the spatial dimensions of a tensor (e.g., the height and width of images) into multiple partitions, each of which is assigned to a specific device [Jia19a]. Spatial parallelism often requires synchronizing the shared elements (e.g., the shared pixels along the boundary of different sub-images) between devices.
4. *Reduction parallelism* exploits the linearity of tensor algebra operators. For a matrix multiplication $C=A\times B$, reduction parallelism splits $A$ along its columns and $B$ along its rows as follows: $A=[A_1,\ldots,A_n]$, $B=[B_1^\top,\ldots,B_n^\top]^\top$. The matrix multiplication is distributed across $n$ devices, with the $i$-th device computing $C_i=A_i\times B_i$. An extra reduction afterward recovers the original result: $C=\sum_i C_i$.
5. *Pipeline parallelism* exploits the opportunity to parallelize across different training iterations [Nar19].
6. *Operator-specific parallelism.* The introduction of new DNN operators provides operator-specific parallelization opportunities. For example, the following equation shows the batched matrix multiplication used in Transformer [Vas17]: $\mathit{output}(s,h,o)=\sum_i \mathit{input}(s,h,i)\times \mathit{weight}(h,o,i)$. This differs from typical matrix multiplication in that it applies a different weight for each input sample. As a result, these batched matrix multiplications across attention heads can be run in parallel (i.e., the $h$ dimension) without any tensor replication or synchronization.

Most parallelizations are not pure performance optimizations, but are instead trade-offs among different cost metrics. For example, applying data parallelism reduces per-device computation time at the cost of increased memory usage and data movement for storing and synchronizing model parameters. Thus, DNN operators typically require a combination of these forms of parallelism to achieve optimal performance.

<span id="section-2-2"></span>

### 2.2 Algebraic Transformations

Algebraic transformations are very diverse and are not as easily categorized as the forms of parallelism, so we instead provide examples. For a more comprehensive exploration of algebraic transformations, see [Jia19b].

<span id="figure-04"></span>

![Basic operator fusion and a more complex DepthwiseConv transformation](../../papers/unity/figure-04.png)

**Figure 4.** Example algebraic transformations. DWC stands for DepthwiseConv (i.e., depth-wise separable convolution).

The most basic algebraic transformation is operator fusion, shown in [Figure 4a](#figure-04). Unfused, the device needs to load and store activations to and from memory twice, once before and after each operator. If the two operators are fused, however, the combined kernel can compute the ReLU operation as it stores the outputs of the MatMul back to memory. For a more complex example, see [Figure 4b](#figure-04). By exploiting DepthwiseConv's linearity, a computation that previously required two DepthwiseConv operations now only requires one plus an additional Add, effectively halving the amount of computation needed.

<span id="figure-05"></span>

![A composition of three algebraic graph transformations](../../papers/unity/figure-05.png)

**Figure 5.** Compositions of small algebraic transformations can lead to significant changes.

Small algebraic transformations can be composed to create large changes. Consider the sequence of transformations shown in [Figure 5](#figure-05): while each individual transformation is relatively small, the final output is radically different from the original computation graph. Also, notice that not all performance gains are realizable in a single transformation: for example, moving from graph 2 to graph 4 reduces the amount of computation by reducing the number of DepthwiseConv operations performed, but it is first necessary to pass through graph 3 which performs worse than either graph 2 or 4.

<span id="section-2-3"></span>

### 2.3 Intermediate Representations

Most existing optimizing frameworks represent a DNN architecture as a computation graph [+3]: a node is a mathematical tensor operator (e.g., matrix multiplication, etc.), and an edge is a tensor (i.e., $n$-dimensional array) passed between operators. An example computation graph is shown in [Figure 6a](#figure-06). Algebraic transformations are performed by iteratively applying graph substitutions, and the model is parallelized by assigning each node a set of parallelism annotations.

<span id="figure-06"></span>

![A computation graph and its parallel computation graph representation](../../papers/unity/figure-06.png)

**Figure 6.** Comparing computation graph and PCG. Both graphs describe the same parallelization of two consecutive matrix multiplications $(A\times B)\times C$ (a simplified form of attention). The green and orange boxes denote regular DNN operators and Unity’s new parallelization operators (see [Section 3.3](#section-3-3)) respectively.

This representation has two limitations. First, while using distinct representations for algebraic transformations (i.e., graph substitutions) and parallelization (i.e., node annotations) is convenient, it hinders joint optimization. The key issue is that algebraic transformations can add or replace nodes in the graph, while parallelization views the computation graph as static and thus cannot handle these newly-created, unannotated nodes. This prevents interleaving the two search algorithms, since at any time a substitution can transform a valid parallelization into an invalid one.

Second, a computation graph does not explicitly capture the communication costs associated with parallelism. This absence makes it difficult for algebraic transformations to reason about the impact on the performance of the final model.

<span id="section-3"></span>

## 3 Parallel Computation Graph

To solve the shortcomings of the existing model representations described in [Section 2.3](#section-2-3), we introduce the parallel computation graph (PCG) [+1] as a unified representation of distributed DNN training that is capable of simultaneously expressing computation, parallelism, and communication. The PCG allows Unity to consider both algebraic transformations and parallelization as graph substitutions on a common graph. While the PCG is not the first to merge computation and parallelization into a single graph, the PCG is tailored for optimization and as such differs from prior unified graph representations in key aspects, which we discuss in [Section 3.4](#section-3-4).

PCGs extend the existing computation graph representation by allowing nodes to represent changes in parallelization in addition to mathematical tensor operations, and edges to represent distributed movement of tensor data in addition to data dependence. A set of parallelization operators are added that allow PCGs to express all existing parallelization strategies and provide an explicit representation of data movement and its associated costs during training. Additionally, each operator in a PCG is associated with a machine mapping, denoting how the execution of the operator is mapped to individual processors in a parallel machine. [Figure 6b](#figure-06) shows an example of a PCG. [Section 3.1](#section-3-1), [Section 3.2](#section-3-2), and [Section 3.3](#section-3-3) provide a brief description of the tensor representation, machine mappings, and parallelization operators, respectively. Finally, [Section 3.4](#section-3-4) discusses the design decisions that make the PCG uniquely suited for joint optimization, and how it differs from alternative unified graph representations.

<span id="section-3-1"></span>

### 3.1 Tensor Representation

Unity models tensors as a set of data dimensions, each of which has two fields: a size and a degree. The degree field specifies the number of partitions the tensor has been divided into along that dimension. Every tensor also includes a special replica dimension, which represents the number of replicas of that tensor's data.

<span id="section-3-2"></span>

### 3.2 Machine Mappings

Each operator in a PCG is associated with a machine mapping, an $n$-dimensional array of devices/processors that specifies on which device to run each piece of the operator's computation. More formally, given an operator and a set of $n$ applicable parallel dimensions with degrees $d_1,\ldots,d_n$, Unity divides the operator into $d_1\times d_2\times\cdots\times d_n$ parallel tasks, which we reference with tuple indices of the form $(i_1,\ldots,i_n)$ where $0\le i_k<d_k$. A machine mapping is a map from task indices $(i_1,\ldots,i_n)$ to individual GPUs that will be used to run that parallel task. For convenience, we also define the machine mapping of an entire PCG to be the set of machine mappings of each of its constituent operators. [Figure 7](#figure-07) shows some example machine mappings for the Summit compute nodes [Vaz18] used in our evaluation. The hardware architecture is depicted in [Figure 7a](#figure-07). [Figure 7b](#figure-07) shows a basic 1-D machine mapping for data parallelism, while [Figure 7c](#figure-07) shows a 2-D machine mapping of a hybrid parallelization strategy combining data and model parallelism, where model parallelism is applied across GPUs within the same compute node and data parallelism across distinct compute nodes. [Figure 7d](#figure-07) shows a 3-D machine mapping where we apply model parallelism across GPUs attached to the same CPU, reduction parallelism across GPUs attached to different CPUs but on the same compute node, and data parallelism across different compute nodes.

<span id="figure-07"></span>

![Summit node hardware and one-, two-, and three-dimensional machine mappings](../../papers/unity/figure-07.png)

**Figure 7.** Example machine mapping for a compute node in our evaluation. (a) shows the node’s hardware architecture, where and orange and grey arrows denote NVLink and X-Bus. Numbers in mapping examples denote GPU ids.

Unity includes a comprehensive set of machine mappings that capture effective usages of a parallel machine. In addition, developers can register custom machine mappings tailored to specific hardware architectures. For example, when node pairs in a cluster have different network bandwidths and latencies, an extra dimension can be added to the existing machine mappings to represent node-level locality.

Machine mappings provide two key desirable properties: expressiveness and scalability. All effective distributions of parallel tasks in a PCG can be captured in just a few machine mappings, and complex features of a machine's hardware architecture can be easily leveraged through adding additional machine mappings. Machine mappings also allow Unity to capture all effective device assignments while remaining linear in the number of devices and aid Unity's search algorithm by removing inefficient assignments from consideration.

<span id="section-3-3"></span>

### 3.3 Parallelization Operators

Unity uses six parallelization operators to capture the computation and communication costs associated with different parallelization strategies. These six are further divided into three pairs, where one operator is the "back propagation" of the other (e.g., when back propagation is done on Partition it becomes semantically equivalent to Combine, and the same in reverse). The three pairs are:

1. *Partition and Combine:* Partition and Combine change a tensor's degree of parallelism. More specifically, Partition increases the parallelism degree of a tensor dimension by splitting the dimension into multiple equal-sized partitions, as shown in [Figure 8a](#figure-08). Combine performs the reverse: reducing a tensor's degree of parallelism by concatenating multiple partitions into one.

2. *Replicate and Reduce:* Replicate and Reduce control the parallelism degree of the replica dimension by copying and summing tensors, as shown in [Figure 8b](#figure-08). Parameter synchronization is naturally captured as the back propagation of Replicate operations applied to weight tensors.

3. *Pipeline and Batch:* Pipeline splits a tensor dimension into equal size partitions and processes one partition at a time, while Batch aggregates tensors across iterations (see [Figure 8c](#figure-08)). Note that Pipeline does not modify the parallelism degree of a tensor dimension, but instead reduce its size.

<span id="figure-08"></span>

![Partition, Combine, Replicate, Reduce, Pipeline, and Batch operators](../../papers/unity/figure-08.png)

**Figure 8.** Parallelization operators in Unity.

As a basic demonstration of the PCG's expressiveness, [Figure 9](#figure-09) illustrates how Unity's six parallelization operators can represent some example parallelization strategies from [Section 2.1](#section-2-1). These parallelization operators can also be composed to create hybrid parallelism. [Figure 8d](#figure-08) shows an example that applies Replicate and Partition on the same tensor dimension, replicating the tensor and partitioning each replica. To improve efficiency, Unity replaces particular sequences of parallelization operators with fused versions at run time (e.g., a Reduce followed by a Replicate can be implemented as an AllReduce).

<span id="figure-09"></span>

![Data, reduction, and pipeline strategies for batched matrix multiplication in a PCG](../../papers/unity/figure-09.png)

**Figure 9.** Representing different parallelization strategies for batched matrix multiplication with a PCG. $s$, $i$, $o$, and $h$ indicate the sample, input channel, output channel, and attention head dimensions, respectively.

<span id="section-3-4"></span>

### 3.4 Discussion and Comparison

Unity's decision to use the PCG instead of an annotated computation graph is driven by how easily the representations lend themselves to joint search and not a fundamental limitation of annotated computation graphs. Theoretically, there exist annotation languages isomorphic to the PCG, but attempts to design such a language quickly lead to a number of difficulties.

First, because each operator can use different forms of parallelism, including operator-specific forms of parallelism, the number of annotations quickly grows prohibitively large. Which annotations are supported by which operators, along with their semantics and composition, must then be baked into the representation itself. By comparison, Unity's PCG moves this knowledge into the PCG substitutions, which are generated automatically. This separation of concerns makes the core of Unity simpler and easier to maintain.

Second, not explicitly representing communication forces communication patterns along dataflow edges to be reconstructed from their source and destination node annotations, which is difficult due to the expressive forms of parallelism Unity considers. Specifically, supporting $n$ parallelism dimensions requires considering up to $2^n$ different subsets of these dimensions and thus $2^n\times 2^n=4^n$ potential communication patterns between operators. Unity explicitly represents communication patterns throughout search, obviating the need for a complex analysis to reconstruct them. Representing these patterns via a small set of parallelization operators also allows Unity to easily recognize and optimize common communication patterns, such as executing a pair of Reduce-Replicate operators as an AllReduce. In an annotated computation graph, these optimizations become entangled with the code for reconstructing the communication patterns themselves, adding significant complexity and implementation effort.

Finally, jointly optimizing an annotated computation graph is challenging, as algebraic transformations can introduce new operators which, since they have not yet been parallelized, lack annotations. As such, the internal representation becomes underspecified and the cost becomes undefined. It is possible to add an additional mechanism to "fill in" these missing annotations such as inserting a random annotation, a fixed value, a value from a neighboring node (though this becomes challenging when neighboring nodes have differing parallelizations), or evaluating the valid parallelizations and choosing the best one. However, Unity's PCG avoids this additional complexity by representing each parallelization strategy for the new operator as one or multiple PCG substitutions, offering an efficient and uniform approach to joint optimization.

**pONNX.** Unity is not the first to integrate computation and parallelism into a single graph: pONNX [Wan19j] proposed doing so using Split, Concat, and custom operators Send and Recv. However, Unity focuses on optimization while pONNX is designed as a serialization format, leading to critical differences.

First, an operator in pONNX with a parallelism degree of $n$ is duplicated $n$ times, requiring an optimizer to reconstruct the operator from multiple nodes. Unity simply adds a parallelism operator so the operator remains a single node in a PCG.

Second, pONNX assigns every communication its own Send/Recv node, which dramatically increases the size of the graph. Since communication patterns in DNN training are highly regular, Unity eschews materializing every communication in favor of optimizing communication patterns (e.g., Reduce, Replicate, etc.), which allows Unity to represent communication costs without reasoning about individual communications.

Finally, pONNX makes device placement part of the operator, while Unity represents it separately as a machine mapping. This allows Unity's search to optimize device assignments separately and to ignore the symmetries created by a large number of compute devices with identical capabilities.

Additional unified representations have been proposed [San21, Sch21c], which are discussed in [Section 7](#section-7).

<span id="section-4"></span>

## 4 Graph Substitutions

Since Unity represents both algebraic transformations and parallelization as graph substitutions, the effectiveness of joint optimization relies on having an appropriate set of graph substitutions. The number of potential substitutions increases exponentially with size, so Unity represents large and complex algebraic transformations and parallelization strategies as compositions of small PCG substitutions. For example, [Figure 10](#figure-10) shows the sequence of substitutions for the hand-tuned parallelization strategy used in Megatron-LM [Sho19].

<span id="figure-10"></span>

![Megatron-LM attention-head parallelism represented as graph substitutions](../../papers/unity/figure-10.png)

**Figure 10.** Representing the hand-tuned parallelization strategies used in Megatron-LM [Sho19] as a sequence of basic graph substitutions in Unity. `BatchMM` and `MatMul` are batched and regular matrix multiplications, respectively. Each arrow denotes a graph substitution, where the dotted subgraphs in the same color are the source and target graph of the substitution. For `Partition` and `Combine`, the parentheses indicate the data dimension for which they are performed.

**Substitution generation.** To reduce the engineering effort to support new parallelization strategies, Unity automatically generates and formally verifies all valid PCG substitutions up to a fixed size to serve as a "basis set" from which the search algorithm can construct sophisticated optimizations. This also allows Unity to not only automatically discover algebraic transformations and parallelization strategies, but also to find novel hybrids of the two missed by prior approaches. To do so, Unity adopts TASO's super-optimization approach [Jia19b].

As in TASO, Unity discovers substitutions in two steps: first it uses a fast heuristic to identify candidate substitutions, and then it uses a more expensive formal verification to ensure correctness. To find candidate substitutions, Unity enumerates all possible PCGs up to a fixed size. Note that this fixed size does not limit the size of the transformations Unity can apply, as many larger substitutions are compositions of smaller ones.

For each generated PCG, Unity computes a fingerprint: a hash of the PCG's output tensors generated by evaluating the PCG on some fixed input tensors. To allow Unity to account for parallelization, we extend the fingerprint function in TASO [Jia19b] to include the parallelism degree of each tensor dimension. A pair of PCGs is considered a candidate substitution if both PCGs have an identical fingerprint. The addition of parallelism causes Unity to discover 651 new candidate substitutions beyond the 743 previously identified by TASO.

**Substitution verification.** Similar to TASO, Unity formally verifies the new substitutions using an automated theorem prover (Z3 [Dem08] in our implementation). Operator specifications are provided in first-order logic, where an operator is represented as a function of its inputs and configuration parameters. For example, `Reduce(d,x)` defines a `Reduce` operator with input $x$ and parallelism degree $d$. The fact that `Reduce` commutes with matrix multiplication is captured by the following operator property (where `Replicate(d,y)` represents a `Replicate` with input $y$ and parallelism degree $d$):

$$
\forall d,x,y.\;\mathrm{Matmul}(\mathrm{Reduce}(d,x),y)=\mathrm{Reduce}(d,\mathrm{Matmul}(x,\mathrm{Replicate}(d,y)))
$$

We follow TASO's methodology for developing operators' parallelization properties: we attempt to formally verify all candidate substitutions using Z3, and when a substitution cannot be verified but is correct, we add the missing operator properties. This procedure was repeated until all 651 new substitutions discovered by Unity were verified. Overall, we introduced 33 operator properties in addition to the 43 properties from TASO [Jia19b] ([Table&nbsp;2](./taso.md#table-02)) to verify all PCG substitutions.

Combined, the substitution generation and verification process takes a total of 30 minutes. Since the available substitutions only change on the addition of new operators or forms of parallelism, this process can be run entirely offline so as not to impact the execution time of Unity's joint search algorithm.

<span id="figure-11"></span>

![A ReLU spatial substitution and a hybrid Add-to-Concat-Reduce substitution](../../papers/unity/figure-11.png)

**Figure 11.** Substitution (a) shows that spatial parallelism is valid for `ReLU`. Substitution (b) demonstrates a hybrid algebraic-parallel transformation: transforming an `Add` into a `Concat` followed by a `Reduce` allows Unity to use the more efficient AllReduce communication pattern.

**Example Substitutions.** Most new substitutions generated by Unity simply state the parallelism valid for an operator. For instance, the substitution in [Figure 11a](#figure-11) indicates that ReLU supports spatial parallelism in the row dimension. However, combining algebraic transformations and parallelization also yields novel hybrids, such as the example shown in [Figure 11b](#figure-11), where Unity identifies that an Add operator is equivalent to a Concat followed by a Reduce. In the left PCG, when Input 1 and Input 2 are located on separate devices and Output is required to be replicated across those same devices, Input 1 and Input 2 would have to be sent to and from a single device to be added. By applying this transformation, Unity is able to merge the input tensors into a single distributed tensor through a Concat (which moves no data) and replace the communication with a Reduce followed by a Replicate (which is implemented as an AllReduce).

<span id="section-5"></span>

## 5 Joint Optimization

This section describes Unity's search algorithm for jointly optimizing algebraic transformations and parallelization. The core problem is as follows: given a PCG ([Section 3](#section-3)), a set of operator-level machine mappings ([Section 3.2](#section-3-2)), and a set of PCG substitutions ([Section 4](#section-4)), find (1) a sequence of PCG substitutions and (2) a machine mapping for the resulting PCG that minimize the per-iteration training time. A key challenge is the exponentially larger search space created by unifying algebraic transformations and parallelization. The search must also scale to both complex DNNs (i.e., a large input PCG) and large numbers of compute devices (i.e., a large set of operator-level machine mappings).

Unity uses a three-level hierarchical search algorithm, depicted in [Figure 12](#figure-12). In simplified form, Unity breaks an input PCG into subgraphs, determines an optimized sequence of substitutions for each subgraph (which requires determining the optimized machine mapping for each candidate), and then combines these sub-solutions to produce the final output. This allows Unity to scale to DNNs with over 300 operators and machines with 192 GPUs while keeping search times below 20 minutes, which is negligible compared to the hours or days needed to train modern DNNs.

<span id="figure-12"></span>

![Three-level hierarchical search in Unity](../../papers/unity/figure-12.png)

**Figure 12.** High-level depiction of Unity’s hierarchical search.

In the following section, we provide a more detailed description of Unity's search algorithm. [Section 5.1](#section-5-1), [Section 5.2](#section-5-2), and [Section 5.3](#section-5-3) describe the three levels of Unity's search algorithm, starting from the middle layer (substitution selection), then the lowest (machine mapping selection), and finally introducing the highest level (graph splitting) as an optimization to help Unity scale to large DNNs. Afterward, we briefly address Unity's cost estimation and how the search algorithm can be tweaked to integrate pipeline parallelism.

<span id="section-5-1"></span>

### 5.1 Substitution Selection

Unity uses the cost-based backtracking search algorithm from TASO [Jia19b] to identify a sequence of substitutions that minimizes the execution time of an input PCG. Unity maintains a queue of candidate PCGs sorted by their execution times, and until the queue is emptied or a fixed budget is exceeded, Unity iteratively removes the best candidate from the queue and uses it to generate new candidates by applying every available substitution at every location in the PCG whenever applicable. Candidate PCGs with execution times that are a threshold factor times worse than the best candidate PCG seen so far are pruned, while the rest are inserted into the queue. The threshold factor allows the user to balance the search time and amount of exploration. In our experiments, we use a threshold factor of 1.05 [+4]. This algorithm allows Unity to explore arbitrary sequences of substitutions, but requires an accurate cost estimator to evaluate the execution time of each candidate PCG. Since a PCG contains only the parallelization of each operator but not the devices to which it is assigned (i.e., the machine mapping), this cost estimator must first determine an optimized machine mapping. An efficient algorithm must be used to identify this mapping, as the cost estimator is called for every candidate PCG. [Section 5.2](#section-5-2) introduces our algorithm to find optimized machine mappings.

<span id="figure-13"></span>

![Sequence and parallel graph splits on a data-parallel ResNeXt module](../../papers/unity/figure-13.png)

**Figure 13.** Applying sequence and parallel graph splits on a data-parallel ResNeXt module. Horizontal and vertical dotted lines refer to sequence and parallel splits, respectively, and numbers indicate the order they are applied.

<span id="section-5-2"></span>

### 5.2 Finding Optimized Machine Mappings

The lowest level of Unity's search algorithm identifies the optimized machine mapping for a candidate PCG. The key observation behind this level is that most modern DNN architectures consist of linear chains of independent strands of parallel computation. For example, ResNeXt [He16] is built around two parallel strands of convolutions (see [Figure 13](#figure-13)), which are repeated to form the final model. Unity leverages this structure by recursively decomposing these linear chains and parallel strands into independent subgraphs through sequence and parallel graph splits respectively. [Figure 13](#figure-13) demonstrates how sequence and parallel graph splits can be iteratively applied to decompose a ResNeXt module into recursive sub-problems which can be solved via dynamic programming.

A sequence graph split partitions an input PCG $G$ by finding a postdominator node $n$, such that all paths from the inputs to the outputs of $G$ go through $n$. This post-dominator node splits $G$ into two disjoint subgraphs $G_1$ and $G_2$. Since all of $G_2$ depends on $n$, and $n$ depends on all of $G_1$, every operator in $G_1$ must complete before any in $G_2$ can start. This reduces the task of finding an optimized machine mapping for $G$ to optimizing machine mappings for $G_1$, $G_2$, and $n$. For example, for split ❶ in [Figure 13](#figure-13), assuming no other splits (such as ⓿) had already been applied, $n$ would be the Add node, $G_1$ would be all the nodes from Input up to but not including Add, and $G_2$ would be all the nodes after the Add until Output.

A parallel graph split partitions a PCG $G$ into independent subgraphs whose computations can be performed in parallel. In this case, Unity considers two potential ways of running the sides $G_1$ and $G_2$: in sequence (with access to the full machine resources) or in parallel (with each side given a disjoint share of the available resources) and chooses the faster one. Unity does not allow combinations of serial and parallel execution, in which branches are run partially in parallel and partially in serial. While this eliminates certain strategies, considering them would significantly reduce Unity's scalability as it requires analyzing exponentially many interleavings of operators, and as evidenced by the results in [Section 6](#section-6), these strategies are not necessary to achieve good performance.

To determine how to partition the available resources when running in parallel, Unity iterates over all possible resource quantities that can be assigned to each side. By considering resource quantities, Unity ignores redundant divisions that differ only in which GPUs are assigned and not in the number and location of these GPUs, replacing an exponential search over all subsets of devices with a quadratic search over resource quantities.

As an additional optimization, Unity maintains a cache of the selected machine mappings for all subgraphs. Since substitution selection generates a new candidate for each substitution, and each substitution modifies only a small part of a PCG, many candidate PCGs have most of their subgraphs in common with other candidates. This allows Unity to skip computing the cost and machine mapping of all but the part of the PCG modified by the substitution under consideration.

<span id="section-5-3"></span>

### 5.3 Scaling to Large Graphs

Even with the dynamic programming algorithm and cross-invocation caching, the search algorithm described so far fails to scale to large models. To understand why, we examine how the number of candidate PCGs in substitution selection scales with the size of the input PCG.

As described in [Section 5.1](#section-5-1), at each iteration Unity generates a candidate PCG for every possible application of each substitution. In the worst case this would require examining $O(2^{gs})$ candidates, where $g$ is the number of nodes in the PCG and $s$ is the number of substitutions Unity considers. In practice $s$ has limited impact on search time as only a small fraction of the substitutions Unity considers can be applied to any one model, but for large models the exponential behavior of $g$ becomes problematic.

To solve this, we borrow from [Section 5.1](#section-5-1) and decompose the PCG into independent sequential subgraphs. However, this approach prevents applying substitutions across these splits, which is problematic since Unity uses substitutions to represent parallelization. Thus, naive graph splitting would reduce the parallelism degree across all splits to 1, eliminating many common and important parallelization strategies, such as using data parallelism across the entire model.

Unity addresses this issue by explicitly searching for the optimal parallelization across every split location. More specifically, for every possible partitioning of the tensor communicated across the split, Unity optimizes the resulting two subgraphs under the condition that the first subgraph's output and the second subgraph's input must both match the partitioning under consideration. When either subgraph does not meet this condition, parallelization operators are inserted to ensure any communication cost arising from a change in partitioning is accounted for.

This method works for Partition and Combine but encounters a problem with Replicate and Reduce. For example, consider the case of the tensor crossing the split location having its replica degree fixed to 2 by the search algorithm. To coerce the first subgraph to output a tensor in this format, the search algorithm could insert a Replicate as its final operation, and the algorithm similarly could insert a Reduce as the first operation of the second subgraph. However, this will incorrectly scale the tensor by a factor of 2! The core issue is that unlike Partition and Combine, Replicate and Reduce are not inverses of each other. Fortunately, since reduction parallelism that spans many nodes of a computation graph is rarely useful in practice, we limit the partitionings across splits to only those with a replica degree of 1.

To reduce the number of algebraic transformations these splits prevent, Unity follows MetaFlow [Jia19] and chooses split locations that disrupt the fewest substitutions while maintaining a minimum subgraph size $k$ [+5]. Thus graph splitting reduces the worst-case number of candidate PCGs from exponential in $g$ to linear in $g$, specifically from $O(2^{gs})$ to $O\left(\frac{gp}{k}\times 2^{ks}\right)$ where $p$ is the number of valid tensor partitionings.

**Cost estimation.** To estimate operator run times and communication costs we use similar methods as prior work [Jia18, Jia19a]. More accurate cost models are possible [San21], but we have not noticed any issues caused by inaccuracies in our model.

**Pipeline parallelism.** When considering pipeline parallelism, Unity adopts the 1F1B schedule (i.e., interleaving forward and backward micro-batches on each device) and the weight update semantics from PipeDream-2BW [Nar21b], which achieves both high training throughput and low memory footprint. To reduce the search space, Unity only considers strategies where pipeline parallelism is applied to all operators in a PCG, since a non-pipeline-parallel operator in the PCG would disable the benefits of pipeline parallelism. In addition, similar to prior work [Hua19, Nar19, Zhe22], Unity only considers sequential pipeline parallelism where each stage only communicates with a single next stage in the pipeline (except for the last stage, which directly performs back propagation after forward processing). Unity also follows prior work [Fan21a, Hua19, Tar21] in assuming that the number of micro-batches in a mini-batch is much larger then the number of pipeline stages so the additional latency introduced by pipeline initialization can be ignored. These constraints allow Unity to explore a comprehensive search space that includes existing pipeline parallelism strategies while maintaining reasonable search time. The search algorithm is also slightly modified: instead of using per-iteration run time as a proxy for throughput, we maximize the throughput directly.

<span id="table-01"></span>

![Original paper table listing the seven evaluated DNNs](../../papers/unity/table-01.png)

**Table 1.** Overview of the seven DNNs evaluated.

<span id="section-6"></span>

## 6 Evaluation

<span id="section-6-1"></span>

### 6.1 Implementation and Experimental Setup

Unity is implemented on top of FlexFlow [Jia19a], a distributed multi-GPU runtime for DNN training. We modified FlexFlow to represent models with PCGs, added support for Unity's additional forms of parallelism, and replaced FlexFlow's randomized search with the algorithm described in [Section 5](#section-5). The substitution generator ([Section 4](#section-4)) is implemented on top of TASO [Jia19b], and extends its fingerprint function to consider parallelization. We also add 33 parallelization-specific properties that are used by the substitution verifier as axioms capturing the semantics of the parallelization operators.

All experiments were performed on the Summit supercomputer [Sum18, Vaz18]. Each compute node is equipped with two IBM POWER9 CPUs, 512 GB main memory, and six NVIDIA Volta V100 GPUs. Three of the GPUs within a node are connected to the same CPU and interconnected via NVLink. Nodes are connected with Mellanox EDR 100Gb InfiniBand.

**DNNs.** [Table 1](#table-01) summarizes the seven DNN models used in our evaluation. ResNeXt-50 [Xie16] and Inception-v3 are commonly used DNNs for image classification. BERT [Dev19] is a language model with state-of-the-art accuracy on a spectrum of language tasks. DLRM [Nau19] and XDL [Jia19d] are deep learning recommendation models for personalization and ads recommendation. CANDLE-Uno [Uno18] is a DNN architecture for precision medicine. Multi-layer perceptron [Gar98] (MLP) is a widely used architecture for a variety of regression tasks and a core component in many DNNs.

We follow prior work in setting hyperparameters for training (e.g., batch sizes, learning rates) [Uno18, Dev19, Mud21, Nau19, Xie16]. We report per-GPU minibatch size $B$: for runs with $n$ GPUs, the global minibatch size is $n\times B$. The global minibatch sizes are consistent with those reported in the literature. We use a per-GPU minibatch size of 64 for ResNeXt-50 and Inception-v3, 4 for BERT-Large, 1024 for DLRM and XDL, and 256 for CANDLE-Uno and MLP. The MLP model includes 16 dense layers, each of which has a hidden dimension of 8192. We use Adam [Kin15] with a learning rate of 0.0001 for BERT-Large, and SGD [Goy17a] with a learning rate of 0.01 for the other DNNs.

Unless stated, pipeline parallelism is disabled when comparing against frameworks that do not support this feature.

We evaluate the impact of pipeline parallelism in [Figure 15a](#figure-15).

**Search Time.** For all DNNs except Inception-v3, Unity's search times are under 10 minutes even for the largest GPU count (i.e., 192). Even for Inception-v3, the most complex architecture in our evaluation with 323 operators, search terminates within 20 minutes. These times are negligible compared to the hours or days needed to train these DNNs.

<span id="section-6-2"></span>

### 6.2 End-to-end Evaluation

We compare the end-to-end training performance of Unity and existing frameworks such as Megatron [Sho19] and Deep-Speed [Raj19]. We also compare against using TASO [Jia19b] and FlexFlow [Jia19a] to perform sequential optimization (i.e., TASO first and FlexFlow second). Since Megatron [Sho19] and Deep-Speed [Raj19] require the user to manually optimize each model, these baselines are only present for a subset of the models, while the automated approaches of FlexFlow and Unity can be used across all seven. [Figure 14](#figure-14) shows the results.

<span id="figure-14"></span>

![Training throughput of seven DNNs under Unity and baseline frameworks](../../papers/unity/figure-14.png)

**Figure 14.** Training throughput comparison among existing frameworks and Unity. The experiments were performed on the Summit supercomputer [Sum18] with 6 GPUs per node. All numbers were measured by averaging 1,000 training iterations.

BERT-Large has been highly optimized by existing frameworks such as Megatron and DeepSpeed which use expert-designed strategies combining multiple forms of parallelism. As such, Unity is not expected to outperform these strategies. Instead, the primary purpose of this evaluation is to determine if Unity can re-discover these hand-tuned strategies within a few minutes of automated search. Note that since Megatron and DeepSpeed require users to manually specify all parallelism degrees for data, tensor-model, and pipeline parallelism, we explore different combinations of the supported parallelism degrees and report the best performance.

Unity achieves on-par performance with Megatron and outperforms both DeepSpeed and FlexFlow. We find that the best strategy discovered by Unity is almost the same as the expert-designed strategy in Megatron: the only difference is that for some matrix multiplications Megatron uses reduction parallelism while Unity uses data parallelism, which has a negligible impact on overall training performance. This shows that even on highly-optimized models Unity is able to automatically generate parallelization optimizations that match those manually designed by domain experts. Megatron is customized for Transformer-based language models and does not support the other DNNs in our evaluation. The fact that the parallelization strategy discovered by Unity matches the expert-designed strategy in Megatron is, in our view, a positive outcome of Unity.

DLRM and CANDLE-Uno both exceed the memory capacity of a single GPU, preventing data parallel training. For both models we use the expert-designed strategy proposed in [Mud21] as a baseline, which parallelizes communication-intensive operators (e.g., embedding tables) in model parallelism and compute-intensive operators (e.g., matrix multiplications) in data parallelism. Unity outperforms both expert-designed strategies and TASO+FlexFlow by up to $3.6\times$ on DLRM and $1.6\times$ on CANDLE-Uno. For all other models, we compare Unity against data parallelism and TASO+FlexFlow.

<span id="figure-15"></span>

![Parallelism-dimension ablation and joint-versus-sequential optimization speedups](../../papers/unity/figure-15.png)

**Figure 15.** (a) End to end performance of BERT-Large integrating different parallelization dimensions. Speedups relative to data+model parallelism. (b) Speedups solely attributable to joint vs sequential optimization on 96 V100 GPUs (16 nodes). Search space and algorithm are fixed to remove effects from Unity’s larger search space and improved search scalability.

Unity outperforms the best existing approaches by $1.0\times$ on ResNeXt-50, $1.3\times$ on Inception-v3, $2.0\times$ on MLP, and $1.9\times$ on XDL. The lack of improvement on ResNeXt-50 is expected as the model's optimal strategy (data parallelism) is already the default used by most frameworks.

We observe that the performance improvement is achieved by (1) supporting operator-specific parallelism and (2) jointly optimizing algebraic transformations and parallelization. We further analyze these details in the following experiments.

<span id="section-6-3"></span>

### 6.3 Parallelism Dimensions

To evaluate how different parallelism dimensions improve training performance, we perform an ablation study of Unity on BERT-Large by iteratively adding new dimensions to Unity and measuring the training throughput. [Figure 15a](#figure-15) shows the results. Compared to data and model parallelism, adding reduction parallelism does not improve training performance, but combining reduction and attention-head parallelism increases performance by up to $1.2\times$ because optimizing the attention operators in BERT-Large requires both reduction and attention-head parallelism, as shown in [Figure 10](#figure-10). Enabling pipeline parallelism achieves an overall speedup of $1.4\times$. This result shows that hybrid strategies and operator-specific dimensions are critical for DNN training performance.

<span id="section-6-4"></span>

### 6.4 Joint Optimization

To evaluate Unity's joint optimization, we compare against sequential optimization of algebraic transformations and parallelization. Results are shown in [Figure 15b](#figure-15). Unlike the TASO+FlexFlow baseline in [Figure 14](#figure-14), in [Figure 15b](#figure-15) we include Unity's additional parallelism dimensions and improved scalability to isolate the effects of joint optimization. As a result, the performance improvement (up to $1.4\times$ speedup) comes solely from the ability to optimize jointly rather than sequentially. We study three examples in detail.

<span id="figure-16"></span>

![Three joint computation and parallelization optimizations discovered by Unity](../../papers/unity/figure-16.png)

**Figure 16.** Example joint optimizations of computation graph and parallelization discovered by Unity. For `Partition`, $i$ and $o$ indicate the input and output channel dimensions of a matrix multiplication.

The first ([Figure 16a](#figure-16)) is a slight generalization of the example introduced in [Figure 2](#figure-02). By not fusing the first MatMul and ReLU, which would be done in sequential optimization as the algebraic optimizer would ignore parallelism, Unity is able to significantly reduce the amount of communication by using reduction parallelism and a more efficient AllReduce (represented by the Reduce followed by Replicate).

The second is shown in [Figure 16b](#figure-16). `Concatenation` is the main performance bottleneck in DLRM and XDL, since it cannot be parallelized in the same dimension as the `Embedding` operators and requires an all-to-all synchronization. The optimization eliminates the `Concatenation` by replacing the subsequent `MatMul` with independent `MatMul`s executed using the same model parallel strategy as the `Embedding` operators, which reduces communication costs as the `Embedding` operators’ outputs are only used locally.

The third optimization is shown in [Figure 16c](#figure-16). An EmbeddingBag [Pyt21] operator computes the sum of a bag of embeddings for each training sample. Unity discovers a joint optimization that transforms an EmbeddingBag to a normal Embedding to enable additional parallelization opportunities.

<span id="table-02"></span>

![Original paper table showing the search-algorithm ablation](../../papers/unity/table-02.png)

**Table 2.** Search algorithm ablation study. “Scaled” numbers are relative to the 2 GPU time with all optimizations enabled.

<span id="section-6-5"></span>

### 6.5 Search Algorithm

To evaluate the impact of the three search optimizations (graph splitting, cross-invocation cache, and dynamic programming) presented in [Section 5](#section-5), we perform an ablation study of the search time for ResNeXt-50. With all three techniques enabled (the "All" column), we see roughly linear scaling as we move from 6 to 48 GPUs. This, along with [Figure 14](#figure-14), demonstrates that Unity's search algorithm scales to nontrivial node counts.

Disabling graph splitting increases search times by $4.3$-$8.8\times$ and causes them to scale nonlinearly, while disabling the cross-invocation cache adds an additional $8.9\times$. Disabling the dynamic programming algorithm causes even the smallest cases to time out. These results indicate that the three proposed techniques are necessary for adequate performance.

<span id="section-7"></span>

## 7 Related Work

**Manually-designed parallelization strategies.** Manually-designed parallelization strategies are used in most existing DNN frameworks to optimize distributed DNN training [Onn21, Aba16, Pyt17, Raj19, Sha18a]. For example, Neo [Mud21] optimizes DLRM by using data parallelism for compute-intensive operators and model parallelism for communication-intensive operators. Megatron-LM [Sho19] proposes a model-specific customized strategy that combines data, reduction, and attention-head parallelism for training large language models. These strategies only work for specific DNN models and do not generalize. We use these expert-designed strategies as baselines in our evaluation and show that Unity can automatically discover strategies with improved performance.

**Automated DNN parallelization.** Recent work has proposed automated approaches to optimizing distributed DNN training. For example, ColocRL [Mir18, Mir17] and Placeto [Add19] use reinforcement learning to find efficient device placement for model parallelism. Baechi [Jeo20a] achieves fast device placement for model parallelism using two memory-constrained algorithms. FlexFlow [Jia19a] uses randomized search to optimize data, model, and spatial parallelism. GSPMD [Xu21], a generalization of GShard [Lep20], finds parallelization strategies based on user-provided hints. PipeDream [Nar19] uses dynamic programming to find optimized strategies combining pipeline and data parallelism. Tofu [Wan19b] uses recursive search to minimize communication time and automatically discovers parallelization dimensions via interval analysis. Tarnawski et al. [Tar20, Tar21] propose a two-level dynamic programming algorithm to partition a DNN computation graph across devices by combining data, pipeline, and tensor model parallelism. Alpa [Zhe22] automates inter-operator (i.e., pipeline) parallelism using dynamic programming and intra-operator (i.e., data and tensor model) parallelism using integer linear programming. Whale [Jia21c] uses computation-balanced partitioning to accomodate heterogenous compute devices and allows specifying parallelization strategies through small parallelization primitives. TensorOpt [Cai21] introduces the cost frontier to simultaneously reason about multiple objectives (e.g., execution time and cloud resource cost) in automatic parallelization. Finally, AutoSync [Zha20a] learns to optimize synchronization strategies for data-parallel training from a few thousand samples. However, existing approaches (except Tofu) only support limited parallelism dimensions and none jointly optimizes algebraic transformations and parallelization. Unity supports all existing parallelism dimensions, is extensible to new operators and forms of parallelism, and jointly optimizes algebraic transformations and parallelization.

**Automated algebraic transformations.** TASO [Jia19b] automatically discovers algebraic transformations for DNNs but does not support parallelization. Unity adopts the superoptimization idea from TASO to generate and verify PCG substitutions. However, unlike the algebraic transformation task considered by TASO, Unity deals with a significantly larger search space and considers additional tasks, such as device assignments. We observe that TASO's search algorithm alone is incapable of exploring the larger search space. To address this challenge, Unity introduces three novel elements of the search technique: the dynamic programming algorithm for finding optimized machine mappings, the subgraph cache for exploiting the locality of graph substitutions, and the additional parallelism-compatible divide-and-conquer approach to enabling scalability to complex models.

**Automated DNN code generation.** Recent work has proposed approaches for generating hardware-specific code for DNN operators. TVM [Che18, Che18a] uses a learning-based algorithm to generate optimized code for a diverse set of hardware backends. Ansor [Zhe20] extends TVM by utilizing a hierarchical search algorithm to explore a much larger search space of program candidates. Unity optimizes DNN computation at a higher level than these approaches. Therefore, Unity's optimizations are orthogonal and can be combined with existing code generation techniques. We leave integrating code generation into Unity as future work.

**Intermediate representations for DNN parallelization.** TensorFlow [Aba16], MLIR [Lat21, Lat20], Relay [Roe19], and ONNX [Le20] represent DNN computation with graph-based intermediate representations (IRs). Distributed training of a model is represented by annotating each operator with a parallelization strategy describing how the operator is parallelized across devices. These approaches represent algebraic transformations and parallelization separately and optimize them sequentially, missing joint optimizations. pONNX [Wan19j], automap [Sch21c], and DistIR [San21] propose IRs that express both computation and communication, but are too low-level to be used for Unity-style joint optimization (see [Section 3.4](#section-3-4) for details). Unity uses a higher-level representation better suited to optimization, the PCG, and represents both parallelization and algebraic transformations as graph substitutions on PCGs.

<span id="section-8"></span>

## 8 Limitations and Future Work

To scale to large DNNs and machines, Unity's search algorithm exploits the sequential and parallel structure of modern DNNs (see [Section 5.2](#section-5-2)). However, there exist DNN architectures (e.g., NASNet [Zop18]) that violate this structure. Extending Unity to include these DNNs would improve generality, but potentially at the cost of decreased scalability.

While Unity successfully optimizes two of the most prominent classes of optimizations (i.e., algebraic transformations and parallelization), there are a variety of additional optimizations currently not considered, such as tensor offloading and rematerialization [Jai20, Kir20, Ren21]. The PCG can be extended to represent these optimizations, but the search algorithm as presented in [Section 5](#section-5) does not reason about memory usage and therefore may generate parallelization strategies that violate memory constraints. While these invalid strategies can be made valid by applying the necessary tensor offloading and rematerialization afterward, not including these optimizations in Unity's joint search potentially leads to suboptimal performance. Thus, integrating memory optimizations into Unity's search algorithm is a promising area for future research.

Another limitation of Unity is its support for pipeline parallelism. While PCGs are capable of representing parallelization strategies that interleave pipeline-parallel and nonpipeline-parallel operators in a PCG, our search algorithm excludes these cases to reduce the search space. In addition, Unity's search algorithm does not consider non-sequential pipeline parallelism strategies, where a stage can have multiple predecessor/successor stages.

<span id="section-9"></span>

## 9 Conclusion

This paper presents Unity, the first system that jointly optimizes algebraic transformations and parallelization in distributed DNN training. Unity represents both parallelization and algebraic transformations as substitutions on a unified graph representation, uses a novel hierarchical search algorithm to identify an optimized sequence of substitutions, and scales to large numbers of GPUs and complex DNNs.

Our evaluation with seven real-world DNN benchmarks on up to 192 GPUs show that Unity outperforms state-of-the-art parallelization approaches by up to $3.6\times$ while keeping optimization times under 20 minutes. As nearly half of this speedup is attributable solely to the use of joint optimization over sequential optimization, Unity demonstrates that joint optimization is practical and that future systems will need to include it or else miss significant performance gains.

## Acknowledgement

We thank the anonymous reviewers for their comments, and are grateful to our shepherd Byung-Gon Chun for his feedback. This material is based upon work supported by the National Science Foundation Graduate Research Fellowship Program under Grant No. DGE-1656518, and an NSF award CNS-2147909. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation.

[+equal]: Contributed equally.

[+1]: To prevent ambiguity, we use the term *computation graph* strictly to refer to the conventional computation graph used in prior work, and *parallel computation graph* or PCG to refer to Unity’s new unified representation.

[+2]: Spatial parallelism was called *attribute parallelism* in [Jia19a].

[+3]: Alternative representations are discussed in [Section 3.4](#section-3-4) and [Section 7](#section-7).

[+4]: This specific value was chosen to match [Jia19b].

[+5]: Our experiments use $k=10$ as it strikes a balance between keeping subgraph sizes small enough for good scalability while blocking relatively few substitutions.
