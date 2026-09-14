---
title: 'Mirage: Multi-Level Tensor Superoptimization'
createTime: 2026/09/14 21:22:00
permalink: /en/papers/mirage/
pageClass: paper-reading
---

> [Mengdi Wu](https://wmdi.github.io/), [Xinhao Cheng](https://www.csd.cs.cmu.edu/people/doctoral-student/xinhao-cheng), [Shengyu Liu](https://interestinglsy.github.io/), [Chunan Shi](https://dblp.org/pid/334/3643.html), [Jianan Ji](https://jiananji.me/), [Man Kit Ao](https://dblp.org/pid/411/1681.html), [Praveen Velliengiri](https://www.linkedin.com/in/praveen-velliengiri-97556a104), [Xupeng Miao](https://hsword.github.io/), [Oded Padon](https://www.wisdom.weizmann.ac.il/~padon/), and [Zhihao Jia](https://www.cs.cmu.edu/~zhihaoj2/). First submitted to arXiv on May 9, 2024; current version v3, revised June 6, 2025. Published in the [19th USENIX Symposium on Operating Systems Design and Implementation (OSDI 2025)](https://www.usenix.org/conference/osdi25/presentation/wu-mengdi), pages 21–38. [Mirage: A Multi-Level Superoptimizer for Tensor Programs](https://arxiv.org/abs/2405.05751v3). <a href="/paper/mirage.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2405.05751). [TeX source](https://export.arxiv.org/e-print/2405.05751v3). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

We introduce Mirage, the first multi-level superoptimizer for tensor programs. A key idea in Mirage is $\mu$Graphs, a uniform representation of tensor programs at the kernel, thread block, and thread levels of the GPU compute hierarchy. $\mu$Graphs enable Mirage to discover novel optimizations that combine algebraic transformations, schedule transformations, and generation of new custom kernels. To navigate the large search space, Mirage introduces a pruning technique based on abstraction that significantly reduces the search space and provides a certain optimality guarantee. To ensure that the optimized $\mu$Graph is equivalent to the input program, Mirage introduces a probabilistic equivalence verification procedure with strong theoretical guarantees. Our evaluation shows that Mirage significantly outperforms existing approaches even for DNNs that are widely used and heavily optimized. Mirage is publicly available at [https://github.com/mirage-project/mirage](https://github.com/mirage-project/mirage).

<span id="section-1"></span>

## 1 Introduction

Enabling high-performance execution of deep neural networks (DNNs) on GPUs is critical for modern ML applications. Today’s DNN frameworks generally specify DNN computation using tensor programs, which are directed acyclic graphs whose nodes and edges represent tensor algebra operators (e.g., matrix multiplication) and tensors (i.e., $n$-dimensional arrays) shared between operators.

To optimize an input tensor program, existing frameworks (e.g., PyTorch [Pyt17] and TensorFlow [Aba16]) use manually designed rules to map the tensor program to expert-written GPU kernels. These approaches generally require extensive engineering efforts to design and implement optimization rules, and they may miss certain optimization opportunities. To address these challenges, recent work has introduced *automated* approaches that optimize tensor programs by searching over a comprehensive space of program transformations and applying them based on their performance on target GPUs. These approaches generally fall into two categories.

The first category of work, including Halide [Rag13], TVM [Che18], and Ansor [Zhe20], is motivated by the idea of algorithm and schedule separation [+1] introduced in Halide and optimizes the *schedule* of a tensor program while fixing the algorithm. For a given algorithm, these optimizers automatically generate performant kernels by searching for possible strategies to execute the kernel on the target hardware. However, due to the linear algebra nature of DNNs, a tensor program can be represented by a wide spectrum of mathematically equivalent algorithms. Existing schedule-based optimizers only consider kernels whose algorithms are manually specified by users, resulting in missed optimization opportunities.

The second category of work, including TASO, Grappler, Tensat, and PET, considers *algebraic transformations*, which exploit mathematical equivalence among different algorithms for a tensor program [Jia19b, Opt19, Yan21f, Wan21]. Examples of algebraic transformations include (1) converting one linear algebra operator into another, such as transforming a convolution to a matrix multiplication; (2) fusing multiple operators to reduce memory access and kernel overhead; and (3) reorganizing operators based on commutativity, associativity, and distributivity. These optimizers perform algebraic transformations at the algorithm level and require programmers to manually specify the set of available operators and their implementations. They are thus limited by the performance of the provided kernels.

All existing automated optimization approaches, from both categories, still require programmers to manually specify a set of kernels (each defined by a tensor function), and then explore the search space of algebraic *or* schedule transformations. However, some advanced performance optimizations require coordinated transformations across the kernel, thread block, and thread levels of the GPU compute hierarchy, and involve introducing completely new kernel computations (e.g., a custom kernel that decomposes standard kernels and fuses only certain computations). Such optimizations are not included in the search space of existing automated methods and must still be implemented manually.

One such example is FlashAttention [Dao23] (see [Section 8.2](#section-8-2) for details), which optimizes attention [Wol22a] on GPUs by reordering operators at the algorithm level (algebraic transformations), reorganizing the computation across GPU kernels (yielding new custom kernels), and adapting the parallelization strategy of each kernel to the GPU architecture (schedule transformations). The transformations required for this example cannot be automatically discovered by existing frameworks and must therefore be implemented manually. An implementation of FlashAttention in Triton [Til19], a widely used tensor program optimizer, contains more than 700 lines of code [Htm23].

We present Mirage, the first *multi-level superoptimizer* for tensor programs. Mirage automatically discovers and verifies sophisticated optimizations of tensor programs that require joint optimization of algebraic transformations, schedule transformations, and the discovery of new custom kernels.

A key idea in Mirage is $\mu$Graphs, a *hierarchical graph representation* that specifies tensor programs across multiple levels of the GPU compute hierarchy. By uniformly treating the kernel, thread block, and thread levels, $\mu$Graphs can capture both algebraic and schedule transformations across these levels. Moreover, optimizing a $\mu$Graph can introduce new custom kernels, which go beyond both algebraic and schedule transformations. For example, Mirage automatically discovers the $\mu$Graphs representing FlashAttention [Dao23] and its inference variant FlashDecoding [Htm23a] as well as other $\mu$Graphs that outperform these manually designed kernels by up to $2.2\times$ for certain use cases. Most of these optimizations discovered by Mirage are outside the search space of existing methods.

<span id="figure-01"></span>

![Figure 1. An overview of Mirage.](../../papers/mirage/figure-01.png)

**Figure 1.** An overview of Mirage.

[Figure 1](#figure-01) shows an overview of Mirage. Mirage first splits an input tensor program into subprograms that fall into the restricted Lax fragment. The Lax fragment, formally defined in [Section 5](#section-5), includes multi-linear operators such as matrix multiplication and convolution, division (useful for normalizations), and limited exponentiation (useful for activations). Partitioning a tensor program into Lax subprograms reduces the optimization search space while preserving most optimization opportunities; it also enables Mirage’s probabilistic equivalence verifier.

**Expression-guided $\mu$Graph generator.** For each Lax subprogram, Mirage’s *expression-guided generator* exhaustively searches for possible $\mu$Graphs equivalent to it. A key challenge Mirage must address is its significantly larger search space compared to prior superoptimization techniques. For example, TASO [Jia19b] and PET [Wan21] search only for tensor programs at the kernel level, using a fixed set of pre-defined kernels, while Mirage considers superoptimization across the kernel, thread block, and thread levels. To efficiently navigate this significantly larger search space, Mirage introduces a novel pruning technique based on *abstract expressions*, which greatly reduces the number of $\mu$Graphs Mirage must consider while providing a certain theoretical guarantee on the optimality of the discovered $\mu$Graphs. Mirage further reduces the search space by focusing the search on the kernel and block levels and using a rule-based approach for the thread level.

**Probabilistic equivalence verifier.** For a $\mu$Graph discovered by Mirage, verifying its functional equivalence with the input program introduces another challenge, since the input and output tensors of a program include up to many millions of elements. A key idea behind Mirage is *probabilistic equivalence verification*, which performs random tests over finite fields to check equivalence between $\mu$Graphs. While random tests typically provide limited correctness guarantees for general programs, Mirage leverages a novel theoretical result showing that the restrictions imposed by the Lax fragment ensure that, for Lax programs, random tests over finite fields offer strong correctness guarantees. Specifically, we show that a polynomial identity testing (PIT) algorithm [Sch80, Zip79] can be generalized to Lax programs, yielding a randomized algorithm for Lax program equivalence that can be made arbitrarily precise. Mirage uses this randomized algorithm to (probabilistically) ensure that each optimized program is equivalent to the input program.

**$\mu$Graph optimizer.** For each verified $\mu$Graph, Mirage’s $\mu$*Graph optimizer* maximizes its runtime performance by further considering potential tensor layouts, scheduling operator execution orders, and planning memory allocation at all of the kernel, thread block, and thread levels. Finally, Mirage returns an optimized tensor program based on the best discovered $\mu$Graph for each individual Lax subprogram.

**Evaluation results.** We evaluate Mirage on a variety of commonly used DNN benchmarks on NVIDIA A100 and H100 GPUs. Even for DNN benchmarks that are widely used and heavily optimized by existing systems, such as the group-query attention used in LLMs [Mod24], Mirage still outperforms current approaches by up to $3.3\times$ by exploiting subtle custom kernels and optimizations missing in existing systems.

<span id="section-2"></span>

## 2 Multi-Level Graph Representation

Mirage uses a $\mu$Graph to specify the execution of a tensor program on GPUs. A $\mu$Graph contains hierarchical graphs at multiple levels to represent computation at the kernel, block, and thread levels [+2]. This section first describes the GPU hierarchy and uses [Figure 3](#figure-03) as a running example to introduce the key components of a $\mu$Graph.

<span id="figure-02"></span>

![Figure 2. GPU compute and memory hierarchy.](../../papers/mirage/figure-02.png)

**Figure 2.** GPU compute and memory hierarchy.

**GPU hierarchy.** [Figure 2](#figure-02) shows the hierarchy of today’s GPUs. Computations on GPUs are organized as *kernels*, each of which is a function executed simultaneously on multiple GPU cores in a single-program-multiple-data (SPMD) fashion. A kernel includes a grid of *thread blocks*, each of which is executed on one GPU streaming multiprocessor and includes multiple *threads* to perform computation on individual data elements. Each thread is associated with a per-thread *register file*, and all threads within a thread block can access *shared memory* to enable collective operations. Finally, all inputs and outputs of a kernel are stored in GPU *device memory*.

<span id="figure-03"></span>

![Figure 3. The computation graph and the best $\mu$Graph discovered by Mirage for RMSNorm and MatMul.](../../papers/mirage/figure-03.png)

**Figure 3.** [Figure 3a](#figure-03) is the computation graph for RMSNorm and MatMul. [Figure 3b](#figure-03) shows the best $\mu$Graph discovered by Mirage for computing RMSNorm and MatMul, which fuses the computation in a single kernel to reduce device memory access and kernel launch overhead, outperforms existing approaches by $1.9\times$. Numbers in brackets indicate tensor shapes, and numbers in braces show the *imap*, *omap*, or *fmap* for the corresponding operators.

**Kernel graph.** Each tensor program corresponds to one *kernel graph*, where each node represents a kernel running on an entire GPU, and each edge is a tensor shared between kernels. All tensors in a kernel graph are stored in GPU device memory since different kernels cannot share data in register files or shared memory. Each node in a kernel graph can be a *pre-defined* kernel operator supported by existing kernel libraries such as convolution by cuDNN [Che14] and matrix multiplication by cuBLAS [Cub16]. In addition, to enable fine-grained inter-kernel optimizations such as kernel fusion, a node in a kernel graph can also be a *graph-defined* kernel operator, whose semantic and behavior are defined by a lower-level (i.e., block) graph. As an example, the kernel operator in [Figure 3b](#figure-03) is a graph-defined operator specified by a block graph.

**Block graph.** A *block* graph specifies computation associated with a thread block [+3], where each node denotes a *block operator* specifying computation within a block, and each edge (blue arrows in [Figure 3b](#figure-03)) is a tensor shared between block operators. Mirage stores all intermediate tensors within a block graph in GPU *shared memory* for two considerations. First, GPU shared memory offers much higher bandwidth than device memory, and this design allows Mirage to reduce device memory access by maximally saving intermediate results in shared memory. Second, for tensors whose sizes exceed shared memory capacity and must be stored in device memory, Mirage uses these tensors to split computation into multiple block graphs, each of which only contains tensors in shared memory. This separation does not introduce additional access to device memory.

Each block graph is also associated with properties specifying its execution, which we introduce below.

<span id="figure-04"></span>

![Figure 4. Demonstrating how an input tensor is partitioned across blocks and for-loop iterations with *imap* and *fmap*.](../../papers/mirage/figure-04.png)

**Figure 4.** Demonstrating how an input tensor is partitioned across blocks and for-loop iterations with *imap* and *fmap*.

**Grid dimensions.** All blocks within a kernel are organized into a mesh with up to 3 dimensions, identified as $x$, $y$, and $z$. A block graph is associated with up to three *grid dimensions* that specify the number of blocks along the $x$, $y$, and $z$ dimensions. The block graph in [Figure 3b](#figure-03) launches $128$ blocks.

First, for each input tensor to a graph-defined kernel operator (e.g., $X$, $G$, and $W$ in the kernel graph in [Figure 3b](#figure-03)), the associated block graph contains an *imap*, which specifies how the input tensor is partitioned into sub-tensors for individual blocks. For each grid dimension (i.e., $x$, $y$, or $z$), the *imap* maps it to either (1) a data dimension of the input tensor or (2) a special *replica* dimension $\phi$. For (1), the mapped data dimension is *equally partitioned* across blocks along the grid dimension. For (2), the input tensor is *replicated* across these blocks. For example, the block graph in [Figure 3b](#figure-03) takes three inputs—$\overline{X}$, $\overline{G}$, and $\overline{W}$—representing the input tensors to each block. For $\overline{W}$, its $\operatorname{imap}=\{x\leftrightarrow d\}$ indicates that the $d$ dimension of tensor $W$ is partitioned into 128 equally sized chunks. As a result, $\overline{W}$ has shape $[h=1024,d=32]$.

Second, for each output tensor of a block graph (e.g., $\overline{Z}$ in [Figure 3b](#figure-03)), the block graph includes an *omap*, which specifies how the outputs of all blocks are concatenated to construct the final output of the kernel operator. In an *omap*, each grid dimension must map to a data dimension of the output tensor, since different blocks must store disjoint tensors in device memory. For $\overline{Z}$ with shape $[b=16,d=32]$ in [Figure 3b](#figure-03), its $\operatorname{omap}=\{x\leftrightarrow d\}$ indicates that blocks with the same $x$ index are concatenated along the $d$ dimension, resulting in a tensor $Z$ with shape $[b=16,d=4096]$.

**For-loop body.** To fit large input tensors in shared memory and to overlap data loading from device memory with computation, a block graph can include a *for-loop body*, which is executed multiple times to complete a kernel. Often, the for loop in a kernel is followed by some post-processing. For example, when computing an average value, the for loop would perform the summation of $n$ values and the post-processing would divide by $n$. Mirage specifies the for-loop body of a block graph using *input iterators*, *for-loop accumulators*, and all operators in between, as shown in the orange box in [Figure 3b](#figure-03)). Each input tensor to a block graph first passes through an *input iterator*, which loads part of the tensor (e.g., $\overline{X}$, $\overline{G}$, and $\overline{W}$) from device memory into shared memory. Each input iterator is associated with an *fmap* to specify which part of the input tensor to load in each iteration. Formally, the *fmap* maps each for-loop dimension to either (1) a data dimension of the input tensor or (2) the replica dimension $\phi$. Similar to *imap*, the tensor is equally partitioned along that dimension for (1) and replicated for (2). [Figure 4](#figure-04) shows how an input matrix is partitioned across blocks and for-loop iterations using different *imap* and *fmap*.

Each block graph is also associated with a *for-loop dimension*, which determines how many iterations the for-loop body is executed to complete the kernel. In addition, Mirage uses *for-loop accumulators* (e.g., the two `Accum` operators in [Figure 3b](#figure-03)) to accumulate intermediate results computed in each iteration (using standard accumulators, e.g., summation and max) and store the accumulated results in shared memory. Once the for-loop body is completed, Mirage proceeds to execute the remaining operators outside the for-loop body directly on the accumulated results. An *output saver* then saves the final result from shared memory back to device memory.

**Thread graph.** A *thread graph* further reduces computation scope from a block to a single thread. Similar to a block graph, each thread graph is also associated with *block dimensions*, which specify the organization of threads within the block, and *for-loop dimensions*, which define the total number of iterations to finish the defined computation. Each thread graph includes *input iterators*, each of which loads an input tensor (e.g., $\overline{\overline{A}}$ and $\overline{\overline{B}}$ in [Figure 3b](#figure-03)) from shared memory into register files, and *output savers*, each of which stores an output tensor from register files back to shared memory (e.g., $\overline{\overline{C}}$). A thread graph is the lowest-level graph in a $\mu$Graph and contains only pre-defined thread operators.

**Tensor layout.** Each tensor in the kernel, block, or thread graph is associated with a *tensor layout* (omitted in [Figure 3](#figure-03) for simplicity), specifying how the tensor is linearized in memory. Note that tensor layouts affect only the performance of a $\mu$Graph and have no impact on its output correctness.

**Definition 2.1 (μGraph Validity).** A $\mu$Graph $G$ is *valid* if: (1) for each kernel, block, and thread operator $o\in G$, its input and output tensors match the specification of $o$; (2) all tensors in each kernel, block, and thread graph can reside in GPU device memory, shared memory, and register file, respectively; and (3) for each block and thread graph with a for-loop body, any path from an input to an output passes through exactly one input-iterator, one for-loop accumulator, and one output saver.

**Comparison with prior work.** Prior work separately considers algebraic [Jia19b, Wan21] or schedule transformations [Rag13, Che18, Mul16], while $\mu$Graphs can represent both in a uniform way. Specifically, the grid and for-loop dimensions and their corresponding mappings (i.e., *imap*, *omap*, and *fmap*) to tensor dimensions constitute a comprehensive search space of possible schedules for graph-defined operators. The hierarchical graphs across the kernel, block, and thread levels allow Mirage to explore algebraic transformations at these levels.

<span id="section-3"></span>

## 3 Case Study: RMSNorm

In this section, we use root mean square layer normalization (RMSNorm) [Zha19l] as a case study to demonstrate the advantages of the $\mu$Graph representation and Mirage’s superoptimization approach. RMSNorm is a widely used normalization technique in recent large language models [Mod24]. Formally, RMSNorm takes two tensors, $X$ and $G$, as inputs and normalizes their element-wise products according to the root mean square:

<span id="equation-01"></span>

$$
Y_{ij}=\frac{X_{ij}G_{j}}{\operatorname{RMS}(X_{i})},\operatorname{RMS}(X_{i})=\sqrt{\frac{1}{d}\sum_{j=1}^{d}X_{ij}^{2}},
$$

where $d$ is the hidden dimension size of $X$.

RMSNorm is often followed by a matrix multiplication (MatMul). [Figure 3a](#figure-03) shows the computation graph of an RMSNorm followed by a MatMul operator, where $X$ is the input tensor, and $G$ and $W$ denote two weight tensors. Existing ML compilers generally launch two separate kernels for RMSNorm and MatMul computations, since both operations internally perform reductions across an input dimension, making it challenging to fuse their computations into a single kernel. This approach requires storing intermediate results (i.e., $Y$) in device memory since different kernels cannot share data in shared memory or register files.

[Figure 3b](#figure-03) shows the best $\mu$Graph automatically discovered by Mirage for computing RMSNorm and MatMul in a single kernel. The computation is fused in a single graph-defined kernel operator to avoid saving intermediate results (i.e., $Y$) in device memory and reduce kernel launch overheads.

We highlight the key differences between the $\mu$Graph discovered by Mirage and the original $\mu$Graph. These differences involve discovering new custom kernels and combining algebraic and schedule transformations, making it infeasible to discover the final $\mu$Graph by separately considering algebraic and schedule transformations. First, Mirage reorders MatMul and the division of RMSNorm by leveraging the commutativity of matrix multiplication and element-wise division (algebraic transformation). Second, Mirage performs the accumulation in the root mean square (i.e., $A_{i}=\sum_{j}X_{ij}^{2}$) and the accumulation in the matrix multiplication (i.e., $B_{ik}=\sum_{j}X_{ij}G_{j}W_{jk}$) in parallel (schedule transformation), avoiding writing the accumulation results to device memory. Next, Mirage instantiates a thread graph to perform a sequence of element-wise operators while maintaining all intermediate results in register files (schedule transformation). Finally, the best discovered $\mu$Graph uses a new custom kernel to fuse the computation of RMSNorm and MatMul, reducing device memory access and kernel launch overheads. This $\mu$Graph outperforms the hand-written kernels in existing systems by $1.5\times$ and $1.9\times$ on NVIDIA A100 and H100 GPUs respectively.

<span id="section-4"></span>

## 4 Expression-Guided μGraph Generator

This section introduces the Mirage $\mu$Graph generator, which automatically discovers potential $\mu$Graphs for an input tensor program. To generate $\mu$Graphs that capture optimizations at the kernel, block, and thread levels, Mirage must explore a significantly larger search space than existing superoptimizers, which only consider optimizations at the kernel level. Mirage employs two key techniques to address this challenge. First, based on the observation that optimizations at the kernel and block levels are substantially more critical to performance than optimizations at the thread level—since accessing device and shared memory is orders of magnitude more expensive than accessing register file—Mirage’s $\mu$Graph generator employs a *hybrid approach*: it exhaustively considers all possible graphs up to a certain size at the kernel and block levels, and uses a rule-based strategy to construct graphs at the thread level. This approach reduces the search space while retaining most performance-critical optimizations. Second, to further prune the search space, Mirage introduces a pruning technique based on an abstraction of $\mu$Graphs called *abstract expression*, which reduces the number of $\mu$Graphs Mirage must consider while providing a certain theoretical guarantee on the optimality of the discovered $\mu$Graphs. We introduce the hybrid $\mu$Graph generation algorithm in [Section 4.1](#section-4-1) and [Section 4.2](#section-4-2), and the expression-guided pruning techniques in [Section 4.3](#section-4-3).

<span id="figure-05"></span>

![Figure 5. An overview of the $\mu$Graph generator.](../../papers/mirage/figure-05.png)

**Figure 5.** An overview of the $\mu$Graph generator.

<span id="algorithm-01"></span>

**Algorithm 1.** Mirage’s hybrid $\mu$Graph generation algorithm.

- **Input:** A Lax program with a computation graph $G_{\mathrm{ref}}$.
- **Output:** A set of $\mu$Graphs $\mathcal{S}$.
- **1.** $E_O\gets E(G_{\mathrm{ref}})$.
- **2.** $\mathcal{S}_0,\mathcal{S}\gets\varnothing$.
- **3.** $\operatorname{GenerateNextKernelOperator}(\mathrm{Inputs}(G_{\mathrm{ref}}))$.
- **4.** For all $G\in\mathcal{S}_0$ do:
- **5.** $\mathcal{S}\gets\mathcal{S}\cup\{\operatorname{ThreadGraphConstruction}(G)\}$.
- **6.** Function $\operatorname{GenerateNextKernelOperator}(G_K)$:
- **7.** $\mathcal{S}_0\gets\mathcal{S}_0\cup\{G_K\}$.
- **8.** For all kernel graph operator types $t$ and input sets $I$ do:
- **9.** If $\operatorname{rank}(I,t)>\operatorname{rank}(op.I,op.t)$ for each $op\in G_K$, then:
- **10.** If $t$ is a pre-defined operator, then:
- **11.** If $o:=\operatorname{ConstructOp}(G_K,I,t)$ is valid, then:
- **12.** $\operatorname{GenerateNextKernelOperator}(G_K\cup\{o\})$.
- **13.** Else, $t$ is a graph-defined operator:
- **14.** For all $\mathit{gridDims}$ and $\mathit{forloopDims}$ do:
- **15.** $G_B\gets\operatorname{TBGraph}(I,\mathit{gridDimd},\mathit{forloopDims})$.
- **16.** $\operatorname{GenerateNextBlockOperator}(G_K,G_B)$.
- **17.** Function $\operatorname{GenerateNextBlockOperator}(G_K,G_B)$:
- **18.** If all shared tensors in $G_B$ are consumed, then:
- **19.** If $o:=\operatorname{ConstructOp}(G_K,G_B.I,G_B)$ is valid, then:
- **20.** $\operatorname{GenerateNextKernelOperator}(G_K\cup\{o\})$.
- **21.** For all block graph operator types $t$ and input sets $I$ do:
- **22.** If $\operatorname{rank}(I,t)>\operatorname{rank}(op.I,op.t)$ for each $op\in G_B$, then:
- **23.** If $o:=\operatorname{ConstructOp}(G_B,I,t)$ is valid, then:
- **24.** $\operatorname{GenerateNextBlockOperator}(G_K,G_B\cup\{o\})$.
- **25.** Function $\operatorname{ConstructOp}(G,I,\mathit{attrs})$:
- **26.** $E\gets\operatorname{ExprInfr}(E(I),\mathit{attrs})$. Refer to [Table 1](#table-01).
- **27.** If $\operatorname{Subexpr}(E,E_O)$, then prune via abstract expressions.
- **28.** $S\gets G.\operatorname{outputTensorShapeInfr}(I,\mathit{attrs})$. Check tensor shape.
- **29.** If $S.\mathit{valid}$ and $G.\mathit{mAlloc}+S.\mathit{size}\leq G.\mathit{mLimit}$, then check memory.
- **30.** Return $G.\operatorname{constructOp}(I,\mathit{attrs})$.
- **31.** Return Invalid.
- **32.** Function $\operatorname{ThreadGraphConstruction}(G)$:
- **33.** $G_{\mathrm{fused}}\gets G$.
- **34.** While there exists $o\in G_{\mathrm{fused}}$ that can be fused with a preceding operator, do:
- **35.** $G_{\mathrm{fused}}\gets\operatorname{FuseOp}(G_{\mathrm{fused}},o)$.
- **36.** Return $G_{\mathrm{fused}}$.

<span id="section-4-1"></span>

### 4.1 Kernel and Block Graph Generation

Mirage generates kernel and block graphs incrementally and leverages several pruning techniques to reduce the search space, as shown in the second part of [Figure 5](#figure-05). Specifically, Mirage maintains a *prefix* of a valid $\mu$Graph and iteratively extends it with new operators. For a graph $G=(V,E)$ we say that $G^{\prime}=(V^{\prime},E^{\prime})$ is a *prefix* of $G$ if it is a subgraph of $G$ such that $\forall u\in V^{\prime},\forall(v,u)\in E,v\in V^{\prime}$.

To generate the next operator in the kernel graph, Mirage enumerates the kernel operator type $t$ and the input tensor set $I$. If $t$ represents the graph-defined operator type, Mirage generates the associated block graph that defines its kernel computation by (1) enumerating the grid and for-loop dimensions (introduced in [Section 2](#section-2)), which enables Mirage to calculate the input tensor shapes of the block graph; and (2) performing a nested generation procedure similar to that used at the kernel level but without considering graph-defined operators. Lines 6-16 and 17-24 in [Algorithm 1](#algorithm-01) show how Mirage generates kernel and block operators, respectively. Mirage checks tensor shape (line 28) and memory usage (line 29) before adding an operator, ensuring a valid prefix.

To ensure that identical $\mu$Graphs are generated only once, Mirage defines the *canonical form* of $\mu$Graphs. Given a $\mu$Graph $G$ with its operators in topological order $o_{1},\ldots,o_{n}$, the *index* of the $j$-th output of $o_{i}$ is defined as a tuple $(i,j)$. Each operator $o_{i}$ in $G$ is assigned a *rank* $(\mathit{input}_{i},\mathit{type}_{i})$, where $\mathit{input}_{i}$ is the list of input tensor indices of $o_{i}$, and $\mathit{type}_{i}$ is the operator type. A $\mu$Graph is in canonical form if its operators are ordered in increasing rank. Mirage generates only $\mu$Graphs in canonical form by requiring that operators be added in increasing order of rank (lines 9 and 22). This approach does not prune out any valid solutions, since each $\mu$Graph can be transformed to canonical form by reordering the operators.

In addition, Mirage utilizes the *abstract expression* technique to prune out prefixes that do not satisfy certain constraints, which will be introduced in [Section 4.3](#section-4-3).

<span id="section-4-2"></span>

### 4.2 Thread Graph Construction

While a similar nested generation strategy can be applied to thread graphs, Mirage instead constructs them using a transformation-based approach (see the third panel of [Figure 5](#figure-05) and lines 4–5 in [Algorithm 1](#algorithm-01)) to reduce the search space. Mirage applies operator fusion when constructing thread graphs, which reduces access to shared memory by reusing tensors in register file whenever possible. For example, Mirage fuses the three element-wise operators (`Mul`, `Sqrt`, and `Div`) in [Figure 3b](#figure-03) into a thread graph, avoiding saving intermediate results to shared memory and keeping the entire computation of these operators in register file. While our current implementation focuses on operator fusion, additional rule-based transformations can be used to construct thread graphs.

<span id="table-01"></span>

![Table 1. Operators supported by Mirage. The second column shows the graph levels supporting each operator (K, B and T denote kernel, block, and thread graphs, respectively). The last column defines the abstract expressions for the outputs of each operator, where $E$ maps tensors to their abstract expressions.](../../papers/mirage/table-01.png)

**Table 1.** Operators supported by Mirage. The second column shows the graph levels supporting each operator (K, B and T denote kernel, block, and thread graphs, respectively). The last column defines the abstract expressions for the outputs of each operator, where $E$ maps tensors to their abstract expressions.

<span id="section-4-3"></span>

### 4.3 Pruning via Abstract Expressions

When searching the space of possible $\mu$Graphs, we aim to avoid $\mu$Graph prefixes whose intermediate results cannot contribute to the desired computation. For example, for the input program $X\cdot Z+Y\cdot Z$, we can prune a prefix that computes $X\cdot Y$, but we should not prune one that computes $X+Y$, as $(X+Y)\cdot Z$ is equivalent to the input program. However, how can we determine whether a prefix can contribute to a desired computation while searching for that computation? Below, we develop a pruning technique driven by this intuition that circumvents the “chicken and egg” problem via *abstraction*. We first present the abstraction—*abstract expressions*—and then explain how it is used for pruning. Finally, we offer a theoretical guarantee that, under certain conditions, this pruning does not exclude the optimal $\mu$Graph.

**Abstract expressions.** Recall that an edge in a $\mu$Graph corresponds to a tensor-valued function of the input tensors. Intuitively, abstract expressions abstract these functions by ignoring the differences between elements of the same input tensor. Formally, abstract expressions are first-order logic terms over the theory of integers and uninterpreted functions. In a $\mu$Graph, the abstract expression of each edge, denoted by $\mathrm{E}(\cdot)$, is defined in [Table 1](#table-01). When computing a $\mu$Graph’s abstract expression, all graph-defined operators are “inlined”. Specifically, the expressions computed for a graph-defined operator’s inputs are passed into its lower-level graph, and the resulting output expressions of that lower-level graph become the output expressions of the graph-defined operator. [Figure 6](#figure-06) shows the abstract expressions for a subgraph of attention.

While abstract expressions capture some information about the function computed at each edge, they also abstract away many details. For example, if $X$ is a $k\times k$ matrix, summing over the rows and summing over the columns both yield the same abstract expression—$\mathsf{sum}(k,\mathrm{E}(X))$. But keeping $k$ as part of the abstract expression is crucial for effective pruning.

<span id="figure-06"></span>

![Figure 6. Illustration of abstract expressions. The abstract expressions of tensors are annotated on edges. A human-friendly notation is used here: $\mathrm{e}^{a}$ denotes $\operatorname{exp}(a)$, $\sum_{k}a$ denotes $\mathsf{sum}(k,a)$, $a/b$ denotes $\mathsf{div}(a,b)$, and $a*b$ denotes $\mathsf{mul}(a,b)$. The tensors $I_{1}$, $I_{2}$ and $O$ are all $64\times 64$ matrices.](../../papers/mirage/figure-06.png)

**Figure 6.** Illustration of abstract expressions. The abstract expressions of tensors are annotated on edges. A human-friendly notation is used here: $\mathrm{e}^{a}$ denotes $\operatorname{exp}(a)$, $\sum_{k}a$ denotes $\mathsf{sum}(k,a)$, $a/b$ denotes $\mathsf{div}(a,b)$, and $a*b$ denotes $\mathsf{mul}(a,b)$. The tensors $I_{1}$, $I_{2}$ and $O$ are all $64\times 64$ matrices.

**Abstract subexpression and pruning.** We use abstract expressions to prune the search space of $\mu$Graphs by formalizing two relations over abstract expressions: equivalence and abstract subexpression. Specifically, we prune any $\mu$Graph prefix whose abstract expression is not a subexpression of some abstract expression equivalent to that of the input program. We formalize abstract expressions as uninterpreted functions in first-order logic over the theory of integer arithmetic and uninterpreted functions, and use an SMT solver to reason about them based on two sets of axioms in [Table 2](#table-02): $A_{\text{eq}}$ and $A_{\text{sub}}$.

First, $A_{\text{eq}}$ axiomatizes equivalence between abstract expressions. As will become clear below, these axioms need not be sound—it is not required that $\mu$Graphs with equivalent abstract expressions are functionally equivalent, since non-equivalent $\mu$Graphs can have the same abstract expression. Second, $A_{\text{sub}}$ axiomatizes the subexpression relation between abstract expressions. A key property of $A_{\text{sub}}$ is that whenever a $\mu$Graph $G_{1}$ is a prefix of $G_{2}$—meaning $G_{2}$ can be constructed by extending $G_{1}$ with additional operators—$\mathrm{E}(G_{1})$ is an abstract subexpression of $\mathrm{E}(G_{2})$; formally, $A_{\text{sub}}\models\operatorname{subexpr}(\mathrm{E}(G_{1}),\mathrm{E}(G_{2}))$, where $\models$ denotes entailment modulo the theory of integer arithmetic and uninterpreted functions.

During the search, Algorithm 1 first computes the abstract expression of the input Lax program, denoted $E_{O}$, and prunes any $\mu$Graph prefix $G$ if $A_{\text{eq}}\cup A_{\text{sub}}\not\models\operatorname{subexpr}(\mathrm{E}(G),E_{O})$. That is, a graph is pruned if its abstract expression is not a subexpression of $E_{O}$. This check is performed using an SMT solver (Z3 [Dem08]). As an optimization, the results of these checks are cached and reused, since Mirage may encounter multiple $\mu$Graphs with identical abstract expressions during the search.

<span id="table-02"></span>

![Table 2. Axiomatization of abstract expressions used for pruning. Mirage checks whether an abstract expression $E_{1}$ is a subexpression of $E_{2}$ by querying an SMT solver to check if $\operatorname{subexpr}(E_{1},E_{2})$ is entailed by these axioms. All variables in these axioms are universally quantified.](../../papers/mirage/table-02.png)

**Table 2.** Axiomatization of abstract expressions used for pruning. Mirage checks whether an abstract expression $E_{1}$ is a subexpression of $E_{2}$ by querying an SMT solver to check if $\operatorname{subexpr}(E_{1},E_{2})$ is entailed by these axioms. All variables in these axioms are universally quantified.

**Theoretical guarantee and the pruning-optimality tradeoff.** Intuitively, our pruning would keep any prefix that can lead to a $\mu$Graph whose abstract expression is equivalent (according to $A_{\text{eq}}$) to that of the input Lax program. Formally:

**Theorem 1 (Pruning via Abstract Expressions).** For an input $\mu$Graph $G_{0}$, and a $\mu$Graph $G$ equivalent to $G_{0}$, if $A_{\text{eq}}\models E(G_{0})=E(G)$ then $G$ will be generated by Algorithm 1.

::: details Proof
By [Tables 1](#table-01) and [2](#table-02), we show that for any operator *op*, if $Y=\mbox{\rm op}(X_{1},\ldots,X_{n})$, then $A_{\text{sub}}\models\operatorname{subexpr}(\mathrm{E}(X_{i}),\mathrm{E}(Y))$ for $1\leq i\leq n$. That is, the abstract expression of each input to *op* is always a subexpression of *op*’s output. Given that $A_{\text{sub}}$ includes reflexivity and transitivity axioms, it follows that for any $G^{\prime}$ that is a prefix of $G$, $A_{\text{sub}}\models\operatorname{subexpr}(\mathrm{E}(G^{\prime}),\mathrm{E}(G))$. Together with the assumption that $A_{\text{eq}}\models E(G_{0})=E(G)$, we have $A_{\text{eq}}\cup A_{\text{sub}}\models\operatorname{subexpr}(\mathrm{E}(G^{\prime}),\mathrm{E}(G_{0}))$. Thus, no prefix of $G$ will be pruned, and Mirage will generate $G$.
:::

The theorem highlights the role of abstract expressions in solving the “chicken and egg” problem outlined above. To decide if a prefix $\mu$Graph is useful, we reason about whether it is a prefix of a useful computation *in the abstract*. The choice of abstraction and the axioms $A_{\text{eq}}$ represents a tradeoff between optimality and pruning. As Theorem 1 shows, we are only guaranteed to find the optimal $\mu$Graph whose abstract expression is equivalent to that of the input program under $A_{\text{eq}}$. Stronger axioms expand the set of $\mu$Graphs covered by the theorem but reduce pruning effectiveness, since more prefixes would pass the subexpression test. In particular, note that $A_{\text{eq}}$ does not include cancellation rules (e.g., $\mathsf{div}(\mathsf{mul}(x,y),y)=y$). As a result, Mirage may miss some equivalent $\mu$Graphs. However, including such axioms would make everything a subexpression of everything, therefore nulling desired pruning. As our evaluation shows, the chosen $A_{\text{eq}}$ yields a good balance between pruning and optimality.

<span id="section-5"></span>

## 5 Probabilistic Equivalence Verifier

Mirage’s *probabilistic equivalence verifier* checks if a candidate $\mu$Graph is equivalent to the desired Lax program. The key idea is to evaluate both on *random inputs* in two finite fields. Using finite fields instead of floating point numbers not only avoids floating point errors but also provides a strong theoretical guarantee: the probability of accepting a non-equivalent $\mu$Graph can be made arbitrarily low.

For general programs, random testing can hardly provide any correctness guarantee. However, we show that for Lax programs (formally defined below), random testing offers a probabilistic correctness guarantee, and repeated tests can reduce the error probability to an arbitrarily small threshold.

Prior work [Wan21] has applied a similar technique to check equivalence between tensor programs that contain only linear operators (e.g., matrix multiplication, convolution). We develop a random testing technique that also supports division and exponentiation, which are needed for many DNN optimizations (e.g., the RMSNorm example in [Section 3](#section-3)).

Mirage verifies equivalence between Lax $\mu$Graphs (linear, division, and an exponentiation) defined below. We introduce the main theoretical results in [Section 5.1](#section-5-1) and present Mirage’s verification methodology in [Section 5.2](#section-5-2).

**Definition 5.1 (Lax μGraph).** A $\mu$Graph $G$ is a Lax $\mu$Graph if (1) $G$ contains only multi-linear operators [+4], division, and exponentiation, and (2) every path from an input to an output in $G$ includes at most one exponentiation.

<span id="section-5-1"></span>

### 5.1 Theoretical Foundations

Without loss of generality, we assume a Lax $\mu$Graph $G$ takes $n$ input tensors and produces one output tensor. Our theoretical results directly generalize to Lax $\mu$Graph with multiple outputs. Since each Lax $\mu$Graph includes linear operators, divisions, and at most one exponentiation along each path, the computation for each entry of the output tensor can be expressed in the following form (by using standard identities such as $\frac{\frac{a}{b}}{\frac{c}{d}}=\frac{ad}{bc}$, $\frac{a}{b}+\frac{c}{d}=\frac{ad+bc}{bd}$, $e^{x}e^{y}=e^{x+y}$):

<span id="equation-02"></span>

$$
\frac{\sum_{i=1}^{k}f_{i}\exp(g_{i}/h_{i})}{\sum_{i=1}^{k^{\prime}}f^{\prime}_{i}\exp(g^{\prime}_{i}/h^{\prime}_{i})}
$$

where $f_{i}$, $g_{i}$, $h_{i}$, $f_{j}^{\prime}$, $g_{j}^{\prime}$ and $h_{j}^{\prime}$ ($1\leq i\leq k$, $1\leq j\leq k^{\prime}$) are polynomials over the entries of the input tensors.

The main theoretical result that underpins our randomized equivalence verification is the following theorem, which extends polynomial identity testing (PIT) [Sch80, Zip79] on finite fields to Lax $\mu$Graphs. Note that the difference of two Lax $\mu$Graphs is also of the form of [Equation 2](#equation-02). Therefore, identity testing of two Lax $\mu$Graphs reduces to testing if an expression of that form is zero. Due to the presence of exponentiation, we use two finite fields instead of one. [+5]

**Theorem 2.** Let $P$ be a function of the form described in [Equation 2](#equation-02), where $f_{i},g_{i},h_{i},f_{i}^{\prime},g_{i}^{\prime},h_{i}^{\prime}$ are non-zero polynomials of degree at most $d$ with integer coefficients between $[-w,w]$. Let $p,q$ be primes such that $q\mid p-1$ and $q>2w$. Let $\mathcal{G}$ be the set of $q$-th roots of unity in $\mathbb{Z}_{p}$. If $P$ is not a zero function, then [Li25af]

$$
\Pr_{(\vec{u},\vec{v},\omega)\leftarrow\mathbb{Z}_{p}^{N}\times\mathbb{Z}_{q}^{N}\times\mathcal{G}}\left[\frac{\sum_{i=1}^{k}f_{i}(\vec{u})\omega^{g_{i}(\vec{v})/h_{i}(\vec{v})}}{\sum_{i=1}^{k^{\prime}}f^{\prime}_{i}(\vec{u})\omega^{g^{\prime}_{i}(\vec{v})/h^{\prime}_{i}(\vec{v})}}\right]\leq 8dk^{4}/q+q^{-1/k^{2}}.
$$

<span id="section-5-2"></span>

### 5.2 Random Tests over Finite Fields

<span id="table-03"></span>

![Table 3. Arithmetic operations for random testing. Mirage selects two prime numbers $p$ and $q$ such that $q$ divides $p-1$. $x_{p}$ and $x_{q}$ are values from the finite fields $\mathbb{Z}_{p}$ and $\mathbb{Z}_{q}$, respectively. The notation $x^{-1}$ and $\sqrt{x}$ represents the multiplicative inverse and square root of $x$ in the corresponding finite field. Specifically, $xx^{-1}\bmod p=1$ and $\sqrt{x}\sqrt{x}\bmod p=x$.](../../papers/mirage/table-03.png)

**Table 3.** Arithmetic operations for random testing. Mirage selects two prime numbers $p$ and $q$ such that $q$ divides $p-1$. $x_{p}$ and $x_{q}$ are values from the finite fields $\mathbb{Z}_{p}$ and $\mathbb{Z}_{q}$, respectively. The notation $x^{-1}$ and $\sqrt{x}$ represents the multiplicative inverse and square root of $x$ in the corresponding finite field. Specifically, $xx^{-1}\bmod p=1$ and $\sqrt{x}\sqrt{x}\bmod p=x$.

Mirage leverages Theorem 2 to probabilistically verify the equivalence of two $\mu$Graphs by performing random testing over the finite fields $\mathbb{Z}_{p}$ and $\mathbb{Z}_{q}$ as defined in Theorem 2. To check the equivalence of two $\mu$Graphs, Mirage first generates input tensors, with each entry uniformly sampled from $\mathbb{Z}_{p}\times\mathbb{Z}_{q}$. Mirage also samples $\omega$ uniformly from the set of $q$-roots of unity in $\mathbb{Z}_{p}$, which is used for exponentiation. Mirage then evaluates the two $\mu$Graphs on these inputs using the operations defined in [Table 3](#table-03). As explained in [Section 5.1](#section-5-1), $\mathbb{Z}_{p}$ and $\mathbb{Z}_{q}$ are used for computations outside and inside the exponent, respectively. All operations except exponentiation are implemented via modular arithmetic in $\mathbb{Z}_{p}$ and $\mathbb{Z}_{q}$ independently. For exponentiation, Mirage uses the value $x_{q}$ from $\mathbb{Z}_{q}$ and computes $\omega^{x_{q}}\bmod p$ to obtain a result in $\mathbb{Z}_{p}$.

Note that in a Lax $\mu$Graph, exponentiation is performed at most once along each path. Finally, Mirage checks whether the two $\mu$Graphs produce identical outputs. This process is repeated multiple times, and the two $\mu$Graphs are considered equivalent if they pass all random tests. The following theorem, which follows from Theorem 2, shows that this process can yield an arbitrarily low error rate.

**Theorem 3.** Equivalent $\mu$Graphs always pass $\mu$Graph verification. For two non-equivalent $\mu$Graphs and a given probability threshold $0<\delta\leq 1$, the $\mu$Graphs pass all $\Omega(\frac{k^{2}}{\ln q}\cdot\ln\frac{1}{\delta})$ random tests with probability at most $\delta$.

**Numerical stability.** While the theorem bridges finite fields and real-number computations, discrepancies can arise between real-number computations and floating-point operations, particularly involving overflow or underflow due to large intermediate values. Mirage employs floating-point tests to filter out $\mu$Graphs with significant numerical errors.

<span id="section-6"></span>

## 6 μGraph Optimizer

For each verified $\mu$Graph, Mirage’s $\mu$*Graph optimizer* maximizes its performance by further performing *layout optimization*, *operator scheduling*, and *memory planning*, as shown in [Figure 1](#figure-01). Mirage defers these $\mu$Graph optimizations until after verification for two reasons. First, these optimizations *do not* affect the correctness of the generated $\mu$Graphs; omitting them when generating $\mu$Graphs reduces the search space Mirage must consider, since $\mu$Graphs with the same graph topology but different choices of tensor layouts, operator orders, or memory allocation plans are considered identical by the $\mu$Graph generator. Second, applying these optimizations after verification also reduces the search space for these optimizations, since the $\mu$Graph optimizer only needs to optimize $\mu$Graphs that are functionally equivalent to the input.

**Tensor layouts.** The $\mu$Graph optimizer explores possible data layouts for all intermediate tensors at the kernel, block, and thread levels and chooses the best combinations to maximize performance. We formulate layout selection as a constrained optimization problem and solve it optimally using an integer linear programming (ILP) algorithm. Specifically, for each tensor $t$ and each possible layout $l$ for $t$, we introduce a boolean variable $B_{t,l}$ to indicate whether tensor $t$ uses layout $l$. Operators at the kernel, block, and thread levels may impose various constraints on tensor layouts. For example, to use kernels from the cuBLAS library [Cub16] for matrix multiplication, the innermost dimension of the two input tensors must be among the last two dimensions. These restrictions are converted into a series of linear constraints on $B_{t,l}$. Different tensor layouts may lead to varying performance. For example, some input tensor layouts support bulk copies from device to shared memory, while others do not. Mirage introduces a cost function to model the performance of each operator under different layout choices. Mirage uses an off-the-shelf ILP solver (i.e., Z3 [Dem08]) to find an optimal layout strategy that satisfies all layout constraints while minimizing cost.

**Operator scheduling.** In a $\mu$Graph, there are multiple topological orders to execute operators, and different orders may yield different performance. For a given input $\mu$Graph, the $\mu$Graph optimizer identifies an efficient strategy to schedule operators by minimizing thread-level synchronization within each thread block (i.e., `__syncthreads()` in CUDA). To achieve this goal, Mirage labels each node with a *depth*, defined as the length of the longest path from any input operator to that node. Mirage uses a dynamic programming algorithm to compute the depth of each node and schedules all operators in ascending order of their depths. This approach minimizes the number of thread-level synchronizations required in the generated CUDA kernel, as Mirage only needs to insert synchronization points between operators with different depths.

**Memory planning.** A third class of post-verification optimizations is memory planning, which determines memory offsets for all intermediate tensors at the kernel, block, and thread levels. Mirage formulates memory planning as a *dynamic storage allocation* problem and exhaustively enumerates all possible allocation plans to discover an optimal strategy.

<span id="section-7"></span>

## 7 Implementation

<span id="table-04"></span>

![Table 4. DNN benchmarks used in our evaluation.](../../papers/mirage/table-04.png)

**Table 4.** DNN benchmarks used in our evaluation.

Mirage is implemented in 30K lines of code in C++, CUDA, and Python. Kernel operators are implemented with the cuDNN and cuBLAS libraries [Che14, Cub16], and block and thread operators are implemented using cuTLASS [Ker22] and CUDA PTX. For each input tensor program, Mirage automatically generates and verifies potential $\mu$Graphs. For each verified $\mu$Graph, Mirage produces CUDA source code for all custom kernels of the $\mu$Graph and compiles the code into binary using the CUDA compiler. This approach enables just-in-time (JIT) compilation and deployment for general tensor programs, and the generated kernels can be directly integrated into a PyTorch program with a few lines of code changes. Mirage’s SMT and ILP solvers are implemented using Z3 4.12.6 [Dem08].

Our implementation supports the operators listed in [Table 1](#table-01). Mirage can be extended to include new operators, such as variants of convolution or matrix multiplication, at the kernel, block, and/or thread levels. To support a new linear operator, Mirage requires (1) a float-pointing implementation of the operator at the kernel, block, and/or thread levels, which is used by the $\mu$Graph optimizer to generate CUDA kernels; (2) an implementation of the operator over modular arithmetic (see [Section 5](#section-5)); and (3) an extension to the abstract expression axioms $A_{\text{eq}}$ and $A_{\text{sub}}$ for the operator (see [Section 4.3](#section-4-3)).

To utilize Theorems 2 and 3, random tests should be performed with sufficiently large prime numbers $p$ and $q$ and iterated multiple times. Our current implementation uses the largest values of $p$ and $q$ whose product fits in 16-bit integers (i.e., $p=227,q=113$) to run these random tests on GPUs. We leverage Mirage’s GPU optimizations–such as keeping intermediate results in shared memory–to accelerate the search procedure. We also perform a single random test without iterating it and compare all elements of the output tensors. We note that this equivalence verification procedure does not introduce false negatives. While it could, in theory, introduce false positives, we have not observed any in practice. For these reasons, we consider this procedure sufficient for the search process and plan to add a final verification step that provides the theoretical guarantees only for the best $\mu$Graph at the end of the optimization process.

**Equivalence verification for non-Lax programs.** While Mirage can generate $\mu$Graphs for arbitrary tensor programs, the probabilistic equivalence verifier is limited to Lax programs and does not support certain DNN operators such as ReLU [Nai10]. As an alternative, we have developed a solver-based verifier for arbitrary tensor programs. The verifier relies on user-provided mathematical properties of individual operators (e.g., linearity, associativity, commutativity, and distributivity) defined in first-order logic and uses these properties to verify equivalence using an automated theorem prover. Compared to the probabilistic equivalence verifier, the solver-based verifier supports more general programs, while requiring additional manual effort to specify the properties of each new operator. A detailed discussion of the solver-based verifier is beyond the scope of this paper.

<span id="section-8"></span>

## 8 Evaluation

<span id="section-8-1"></span>

### 8.1 Experimental Setup

<span id="figure-07"></span>

![Figure 7. Comparing Mirage with existing systems for 6 benchmarks on an A100 and an H100 GPU. The performance of all systems are normalized by Mirage (higher is better). Numbers above the Mirage bars show the speedup over the best baselines.](../../papers/mirage/figure-07.png)

**Figure 7.** Comparing Mirage with existing systems for 6 benchmarks on an A100 and an H100 GPU. The performance of all systems are normalized by Mirage (higher is better). Numbers above the Mirage bars show the speedup over the best baselines.

Since Mirage is a superoptimizer for Lax programs, we focus our evaluation on various DNN benchmarks commonly used in existing DNNs, each of which is a Lax program. These benchmarks provide the most fine-grained way to compare the performance of Mirage and existing systems. [Table 4](#table-04) shows the six benchmarks in our evaluation. GQA, RMSNorm, and GatedMLP are the main building blocks of large language models (LLMs). QKNorm introduces query-key normalization before attention to enhance model convergence [Cha24]. LoRA enables low-rank adaptation for fine-tuning a DNN on different tasks. We use a context length of 8K for GQA and 4K for QKNorm, corresponding to the maximum supported by LLaMA-3-70B [Mod24] and Chameleon-7B [Cha24], respectively. In addition, we also evaluate how Mirage-generated kernels improve the end-to-end performance of full DNNs, including Chameleon [Cha24], nGPT [Los24], LLaMA-3 [Mod24], and LoRA [Hu21].

The experiments were conducted on NVIDIA A100 and H100 GPUs, each with 40GB of memory. All our benchmarks fit on a single GPU except GQA (used for LLaMA-2-70B), which is generally parallelized across four GPUs using tensor model parallelism [Sho19]. Therefore, we evaluate GQA under this parallelism strategy, where the eight key-value heads are equally partitioned across four GPUs. Since the performance of Mirage and all baselines depends only on the shapes of the input tensors, we repeat each experiment 1,000 times using random inputs and report the average run time.

One of our benchmarks, LoRA, requires concatenation to express a common optimization: fusing two matrix multiplications via concatenation. To support this optimization in Mirage, we introduce a new linear operator that takes four inputs and computes $f(W,X,Y,Z)=(W\|X)\times(Y\|Z)$, where $\|$ is tensor concatenation. This operator is equivalent to computing $W\times Y+X\times Z$. We define the abstract expression associated with this operator as: $\mathrm{E}(f(W,X,Y,Z))=\mathsf{add}(\mathsf{sum}(k_{1},\mathsf{mul}(\mathrm{E}(W),\mathrm{E}(Y))),\mathsf{sum}(k_{2},\mathsf{mul}(\mathrm{E}(X),\mathrm{E}(Z))))$, where $k_{1}$ and $k_{2}$ are the last dimensions of $W$ and $X$.

Unless otherwise stated, Mirage considers up to 5 operators in the kernel graph and up to 11 operators in each block graph.

<span id="section-8-2"></span>

### 8.2 Benchmark Results

[Figure 7](#figure-07) compares the performance of Mirage with systems on six DNN benchmarks on NVIDIA A100 and H100 GPUs. All systems use half-precision floating points to run these DNN benchmarks. TASO [Jia19b] and PET [Wan21] are DNN superoptimizers that automatically generate algebraic transformations at the kernel level. We report a combined TASO/PET baseline, as the latest TASO implementation includes PET’s partially equivalent transformations as special substitutions. PyTorch [Pyt17] uses the highly optimized cuDNN and cuBLAS libraries [Cub16, Che14] to perform DNN operators on GPUs. For the PyTorch baseline, we enable `torch.compile` and use FlashAttention kernels to maximize performance. TensorRT and its LLM variant TensorRT-LLM include a set of manually designed and highly optimized kernels for common tensor operators such as attention [Ten17a]. FlashAttention and its inference variant FlashDecoding are manually written kernels for efficient attention [Dao23, Hon24d]. Finally, Triton is a schedule-based optimizer to generate high-performance kernels and has been adopted in production systems, outperforming other schedule-based approaches [Til19]. All baselines use CUDA Graphs to minimize kernel launch overhead.

Compared to the best existing approaches, Mirage improves the performance of these benchmarks by up to $3.3\times$ by combing algebraic transformations, schedule transformations, and the generation of new custom kernels. [Section 3](#section-3) shows the best discovered $\mu$Graphs for RMSNorm. Next, we present a case study for the remaining benchmarks.

<span id="figure-08"></span>

![Figure 8. Comparing the $\mu$Graphs used by existing optimizers and Mirage for QKNorm and attention.](../../papers/mirage/figure-08.png)

**Figure 8.** Comparing the $\mu$Graphs used by existing optimizers and Mirage for QKNorm and attention.

**GQA.** Group-query attention is the backbone of LLMs and has been heavily optimized by existing frameworks. For example, FlashAttention and FlashDecoding are expert-designed attention kernels and have been adopted in existing LLM inference systems [Dao23]. Mirage discovers these expert-designed kernels as well as other $\mu$Graphs that outperform them by up to $2.2\times$. The speedup is achieved by two additional optimizations on top of existing hand-written kernels. First, current approaches rely on fixed heuristics to determine the grid dimensions for GQA, which are suboptimal in certain scenarios. For example, TensorRT-LLM launches the GQA kernel with grid dimensions of (8, 2, 1) and (8, 2, 8) when the batch sizes are 1 and 8, respectively. However, both configurations cannot fully utilize all SMs on A100 (108 SMs) and H100 (132 SMs) GPUs. In contrast, Mirage automatically searches for the best grid dimensions for each $\mu$Graph, resulting in full SM utilization. Further ablation study shows that the performance of the best $\mu$Graph discovered by Mirage degrades by 18% when using the same grid dimensions as TensorRT-LLM.

Second, existing approaches use fixed tensor dimensions to parallelize GQA across thread blocks. For example, FlashAttention [Dao23] parallelizes attention across the *sample*, *head*, and *query sequence* dimensions, while FlashDecoding and TensorRT-LLM leverage the *sample*, *head*, and *key-value sequence* dimensions. Both strategies are efficient for conventional multi-head attention with many heads but suboptimal for GQA with fewer attention heads. In contrast, Mirage automatically selects the most efficient parallelization strategy by choosing among the sample, KV heads, query sequence, and key-value sequence dimensions. Moreover, Mirage generates different $\mu$Graphs tailored to different attention scenarios, reducing device memory access by up to $7\times$ compared to the heuristics used in existing systems.

Implementing Mirage’s $\mu$Graphs in existing systems is possible but requires extensive engineering effort to support different kernels for different scenarios. In contrast, Mirage automatically generates them and verify their correctness.

**QKNorm.** To reduce model divergence, several recent DNNs introduce query-key normalization (QKNorm) into the Transformer architecture [Cha24]. QKNorm applies layer normalization to the query and key vectors before attention, as shown in [Figure 8a](#figure-08). These additional normalization layers are not yet supported by existing attention implementations (e.g., FlashAttention and TensorRT-LLM) and require launching separate kernels for normalization and attention.

Mirage automatically discovers a $\mu$Graph that integrates QKNorm and attention computation into a custom kernel, as shown in [Figure 8b](#figure-08). The $\mu$Graph reorganizes the attention computation to enable fusion with the two layer normalizations, which avoids writing intermediate results to GPU device memory and reduces the kernel execution time by up to $1.4\times$.

<span id="figure-09"></span>

![Figure 9. Comparing the tensor programs used by existing optimizers and by Mirage for LoRA: $O=W\times X+B\times A\times X$. Note that both matrices $A$ and $B$ are low-rank.](../../papers/mirage/figure-09.png)

**Figure 9.** Comparing the tensor programs used by existing optimizers and by Mirage for LoRA: $O=W\times X+B\times A\times X$. Note that both matrices $A$ and $B$ are low-rank.

**LoRA.** Low-rank adaptation (LoRA) introduces a pair of low-rank adapters to the linear operators of a pre-trained DNN to improve its performance for downstream tasks. Existing tensor program optimizers launch separate kernels for the original linear operator and the two additional linear operators introduced by LoRA ([Figure 9a](#figure-09)), which introduces high kernel launch overheads since these LoRA operators involve minimal computation. [Figure 9b](#figure-09) shows the best $\mu$Graph discovered by Mirage for LoRA, which fuses the three `Matmul`s and the subsequent `Add` into a single kernel. Mirage reorganizes the computation into two block-level `Matmul`s by leveraging the following algebraic transformation: $W\times X+B\times A\times X=(W\|B)\times\big(X\|(A\times X)\big)$. The `Concat`s in [Figure 9b](#figure-09) do not involve any computation and are performed by updating tensor offsets in GPU shared memory. This $\mu$Graph reduces the execution cost of LoRA by 1.1-$2.4\times$.

<span id="figure-10"></span>

![Figure 10. Comparing the $\mu$Graphs used by existing optimizers and Mirage for GatedMLP.](../../papers/mirage/figure-10.png)

**Figure 10.** Comparing the $\mu$Graphs used by existing optimizers and Mirage for GatedMLP.

**GatedMLP.** Gated multi-layer perceptrons are commonly used in DNNs to capture non-linear representations. We use the GatedMLP configuration introduced in Falcon-7B [Alm23b], whose kernel graph is shown in [Figure 10a](#figure-10). Existing tensor program optimizers generally fuse the two `Matmul`s in a single kernel to reduce GPU device memory access, since the input tensor $X$ only needs to be loaded once. However, this approach still requires launching multiple kernels and storing intermediate results—specifically, the output of the two `Matmul`s—in device memory, as the `SiLU` activation and elementwise multiplication are not fused with the `Matmul`s.

In contrast, the best $\mu$Graph discovered by Mirage ([Figure 10b](#figure-10)) performs the two `Matmul`s in parallel within the same block graph and fuses the remaining computation (i.e., `SiLU` and `Mul`) as post-processing steps within the same block graph. This approach yields $1.5\times$ speedups on A100 GPUs and 2.7-$3.3\times$ speedups on H100 GPUs.

**nTrans.** To accelerate model training, nGPT introduces normalized Transformer, which normalizes all intermediate results in Transformer [Los24]. Formally, the computation is defined as $y=\texttt{Norm}(x+\alpha(\texttt{Norm}(h-x)))$, where `Norm` is a normalization layer, and $x$, $h$, and $\alpha$ are input tensors. Existing systems launch three separate kernels for nTrans, since it interleaves normalization and elementwise addition and multiplication. Mirage automatically discovers a $\mu$Graph that fuses the computation into a single kernel and stores all intermediate results in GPU shared memory. Mirage outperforms other baselines but is slower than TensorRT. This performance gap is because Mirage loads data from global memory to shared memory and writes it back for each tensor in graph-defined kernels. This design improves memory efficiency and enables asynchronous pipelines. However, for kernels with light computation, the overhead of these memory transfers can dominate the kernel runtime. To mitigate this overhead, we plan to extend Mirage to support bypassing shared memory during data loading, therefore avoiding unnecessary data movement.

<span id="figure-11"></span>

![Figure 11. Comparing the end-to-end inference performance of PyTorch and PyTorch with Mirage-generated kernels.](../../papers/mirage/figure-11.png)

**Figure 11.** Comparing the end-to-end inference performance of PyTorch and PyTorch with Mirage-generated kernels.

<span id="section-8-3"></span>

### 8.3 End-to-end Results

In addition to the microbenchmark performance, we also evaluate how Mirage-generated kernels impact the end-to-end latency of commonly used DNNs. Mirage supports just-in-time compilation and deployment, and its generated kernels can be directly integrated into PyTorch programs. We compare PyTorch with its native handwritten CUDA kernels and PyTorch with Mirage-generated kernels on four DNN models. [Figure 11](#figure-11) shows the results. Mirage reduces the end-to-end latency of these models by 0.9-$1.9\times$ by automatically generating highly optimized kernels. The improvement is achieved with a few lines of code changes to the PyTorch programs.

<span id="section-8-4"></span>

### 8.4 Search Time

<span id="table-05"></span>

![Table 5. Ablation study on Mirage’s techniques to accelerate $\mu$Graph generation. We evaluate the impact of multi-threading and abstract expressions on search time for RMSNorm.](../../papers/mirage/table-05.png)

**Table 5.** Ablation study on Mirage’s techniques to accelerate $\mu$Graph generation. We evaluate the impact of multi-threading and abstract expressions on search time for RMSNorm.

In our evaluation, Mirage takes up to 4 hours to optimize a Lax program. This optimization is a one-time cost before deployment on the target hardware. This subsection provides detailed results and an ablation study of Mirage’s search procedure, focusing on how its techniques enable the exploration of large $\mu$Graphs while maintaining low search time. In particular, we evaluate the impact of two techniques: pruning via abstract expressions ([Section 4.3](#section-4-3)) and multi-threading. [Table 5](#table-05) reports the search times for RMSNorm as we vary the maximum number of operators allowed in a block graph.

Multi-threading significantly reduces the search time, while pruning via abstract expressions is crucial for the scalability of Mirage. Specifically, the pruning techniques allow Mirage to explore $\mu$Graphs whose block graphs can each have at most 11 operators, while disabling abstract expression pruning restricts Mirage to handle block graphs with up to 6 operators within a 10-hour search window. Note that discovering the optimized $\mu$Graph for RMSNorm shown in [Figure 3](#figure-03) requires exploring block graphs with 11 operators.

<span id="section-8-5"></span>

### 8.5 Ablation Study on Optimizations

<span id="figure-12"></span>

![Figure 12. Ablation study on optimizations used in Mirage. We evaluate the performance degradation when disabling each optimization independently. The evaluation is performed on A100 for GQA with batch size $1$.](../../papers/mirage/figure-12.png)

**Figure 12.** Ablation study on optimizations used in Mirage. We evaluate the performance degradation when disabling each optimization independently. The evaluation is performed on A100 for GQA with batch size $1$.

We conduct an ablation study to evaluate the impact of thread graph construction and optimizations introduced in [Section 6](#section-6), including layout optimization, operator scheduling, and memory planning. Specifically, we measure the performance degradation of the best $\mu$Graph discovered by Mirage when each optimization is disabled independently. The study is conducted on an A100 using the GQA benchmark with a batch size of $1$. The results, shown in [Figure 12](#figure-12), indicate that disabling any individual optimization leads to a performance degradation ranging from $5\%$ to $70\%$.

<span id="section-9"></span>

## 9 Related Work

**Manually-designed kernels.** Many existing frameworks, such as TensorFlow XLA [Xla17, Aba16], PyTorch [Pyt17], and TensorRT [Ten17a], rely on GPU experts to manually design kernels for ML operators. Recently, significant engineering effort has been dedicated to hand-optimizing GPU kernels for commonly used DNNs, particularly foundation models [Bom22]. For example, to accelerate attention computation [Wol22a], several specialized kernels have been developed based on FlashAttention [Dao23, Htm23a, Hon24d, Fas21]. Due to the increasing complexity of modern GPUs—such as tensor cores in A100s [Mar18b] and thread block clusters in H100s [Htt23]—manually designed kernels may miss subtle optimizations that are hard to discover manually.

**Superoptimization-based approaches.** Superoptimization was originally introduced to find optimal instruction sequences [Mas87, Sch13, Ban06]. Recent work has applied superoptimization techniques to tensor programs [Jia19b, Wan21, Zhe23f, Yan21d, Ung22, Jia19a, Hu24d, Jeo25]. However, all these attempts only consider algebraic transformations at the kernel level and cannot discover more sophisticated optimizations that require jointly considering algebraic and schedule transformations at all of the kernel, block, and thread levels. Our evaluation shows that Mirage largely outperforms existing DNN superoptimizers, demonstrating the importance of multi-level joint optimization.

**Schedule-based approaches.** Recent work has introduced ML compilers that automatically optimize the execution schedule of kernel GPUs. Systems such as TVM [Che18, Che18a], Ansor [Zhe20], and Triton [Til19], along with others [Zhe20a, Hag23, Fen22], build on the idea of algorithm-schedule separation introduced in Halide. They search for optimized schedules to execute a user-specified algorithm on GPUs. However, schedule-based approaches require users to explicitly specify the algorithm for each kernel, and their performance is limited to the quality of these provided algorithms.

**Multi-level graph representations.** Welder [Shi23a] and ASPEN [Par23c] introduce multi-level tile graphs that share similarities with Mirage’s $\mu$Graphs, as both representations follow the GPU hierarchy. However, prior work focuses on scheduling transformations, while Mirage extends beyond scheduling by also considering algebraic transformations and the discovery of new custom kernels. Most optimizations presented in this paper fall outside the scope of these prior approaches.

<span id="section-10"></span>

## 10 Conclusion

This paper proposes Mirage, the first multi-level superoptimizer for tensor programs. Mirage introduces a hierarchy graph representation to specify a tensor program at the kernel, thread block, and thread levels of the GPU execution hierarchy, and uses a novel pruning technique based on abstraction to significantly reduce the search space Mirage needs to consider while providing a certain optimality guarantee. Mirage outperforms existing tensor program optimizers by up to $3.3\times$, even for widely used and heavily optimized DNNs.

## Acknowledgment

We would like to thank the anonymous reviewers and our shepherd, Stephanie Wang, for their valuable comments and suggestions. We thank Tianqi Chen, Phillip Gibbons, Bohan Hou, Muyan Hu, Jinchen Jiang, Xiaoyu Jiang, Ruihang Lai, Yu Zhou, and other CMU Catalyst members for their feedback on this work. This research is partially supported by NSF awards CNS-2147909, CNS-2211882, and CNS-2239351, and research awards from Amazon, Cisco, Google, Meta, NVIDIA, Oracle, Qualcomm, and Samsung. This research is also partially supported by a research grant from the Center for New Scientists at the Weizmann Institute of Science and by a grant from the Azrieli Foundation.

[+1]: In the schedule optimization literature, an algorithm describes what to compute in a kernel and a schedule specifies how to compute the kernel.

[+2]: For simplicity, we use the term *block* to refer to a thread block of a CUDA kernel and *thread* to refer to a single CUDA thread.

[+3]: In the CUDA programming model, a kernel’s computation is defined as computations for independent thread blocks.

[+4]: Operator *op* with $n$ inputs is multi-linear if *op* is linear to all inputs $I_{k}$: <br>(1) $\forall X,Y.\mbox{\rm op}(I_{1},...,I_{k-1},X,I_{k+1},...,I_{n})+\mbox{\rm op}(I_{1},...,I_{k-1},Y,I_{k+1},...,I_{n})=\mbox{\rm op}(I_{1},...,I_{k-1},X+Y,I_{k+1},...,I_{n})$, and <br>(2) $\alpha\cdot\mbox{\rm op}(I_{1},...,I_{k-1},X,I_{k+1},...,I_{n})=\mbox{\rm op}(I_{1},...,I_{k-1},\alpha\cdot X,I_{k+1},...,I_{n}).$

[+5]: We use two primes $p$ and $q$ for polynomial identity testing [Sch80, Zip79] outside and inside the exponents, respectively. The condition $q$ divides $p-1$ is to ensure the existence of $q$-th roots of unity in $\mathbb{Z}_{p}$.
