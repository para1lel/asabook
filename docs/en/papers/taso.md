---
title: 'TASO: Automatic Graph Optimization'
createTime: 2026/07/28 16:00:00
permalink: /en/papers/taso/
---

> [Zhihao Jia](https://www.cs.cmu.edu/~zhihaoj2/), [Oded Padon](https://www.wisdom.weizmann.ac.il/~padon/), [James Thomas](https://cs.stanford.edu/~jjthomas/), [Todd Warszawski](https://dblp.org/pid/143/7244), [Matei Zaharia](https://people.eecs.berkeley.edu/~matei/), and [Alex Aiken](https://theory.stanford.edu/~aiken/). Published at SOSP 2019. This HTML transcription preserves the text, figures, tables, and references of the [original PDF](/paper/TASO.pdf). DOI: [10.1145/3341301.3359630](https://doi.org/10.1145/3341301.3359630).

## Abstract

Existing deep neural network (DNN) frameworks optimize the computation graph of a DNN by applying graph transformations manually designed by human experts. This approach misses possible graph optimizations and is difficult to scale, as new DNN operators are introduced on a regular basis.

We propose TASO, the first DNN computation graph optimizer that automatically generates graph substitutions. TASO takes as input a list of operator specifications and generates candidate substitutions using the given operators as basic building blocks. All generated substitutions are formally verified against the operator specifications using an automated theorem prover. To optimize a given DNN computation graph, TASO performs a cost-based backtracking search, applying the substitutions to find an optimized graph, which can be directly used by existing DNN frameworks.

Our evaluation on five real-world DNN architectures shows that TASO outperforms existing DNN frameworks by up to 2.8×, while requiring significantly less human effort. For example, TensorFlow currently contains approximately 53,000 lines of manual optimization rules, while the operator specifications needed by TASO are only 1,400 lines of code.

## 1 Introduction

Deep neural network (DNN) frameworks represent a neural architecture as a computation graph, where each node is a mathematical tensor operator (e.g., matrix multiplication, convolution, etc.). To improve the runtime performance of a computation graph, the most common form of optimization is graph substitutions that replace a subgraph matching a specific pattern with a functionally equivalent subgraph with improved performance.

Existing DNN frameworks optimize a computation graph by applying graph substitutions that are manually designed by domain experts, as depicted in [Figure 1a](#figure-01). For example, TensorFlow, PyTorch, TensorRT, and TVM use a greedy rule-based optimization strategy and directly perform all applicable substitutions (i.e., rules) on an input computation graph [Aba16, Che18, PyT17, Ten17a]. MetaFlow [Jia19] allows substitutions that may either increase or decrease performance to enable a larger search space of equivalent computation graphs and uses back-tracking search to explore this space, but it still requires manually specified substitutions. Although manually designed substitutions improve the performance of DNN computations, they fall short in several respects.

<span id="figure-01"></span>

![Comparison of computation graph optimization in existing DNN frameworks and TASO](../../papers/taso/figure-01.png)

**Figure 1.** Comparing computation graph optimization in existing DNN frameworks with TASO.

### Maintainability

Hand-written graph substitutions require significant engineering effort. For example, TensorFlow r1.14 includes 155 substitutions implemented in approximately 53K lines of C++ code. The maintenance problem is aggravated by the fact that new operators are continuously introduced; recent work, for example, has proposed depth-wise [How17], grouped [Xie16], and transposed convolutions [Dum16] for different image classification tasks. TensorFlow r1.14 includes 17 graph substitutions (written in 4K lines of code) to optimize ordinary convolution, such as by fusing it with different operator types. Supporting each new convolution variant would require similar implementation effort because each has slightly different semantics and cannot be directly optimized using existing substitutions.

### Data layout

Tensor data can be stored in memory in various layouts, and this choice has a high impact on runtime performance. The best layout depends on both the operator and the hardware. For example, on a P100 GPU, convolution performs best with row-major layout (i.e., the inner-most dimension is contiguously stored), while matrix multiplication performs best with column-major layout (i.e., the outer-most dimension is contiguously stored). On a Tesla V100 GPU with tensor cores [Ten18] supporting 4×4 matrix operations, optimal performance may require tiling tensors into 4×4 chunks. However, considering layout transformations together with graph substitutions adds another level of complexity. For example, a graph substitution may only improve performance if it is combined with a particular layout transformation (see [Section 7.5](#_7-5-joint-optimization-of-graph-substitutions-and-data-layout)). Current frameworks avoid this complexity by treating data layout and graph substitution as separate optimization problems and solve them sequentially [Che18, Mkl16], as shown in [Figure 1a](#figure-01), but this separation misses many possible optimization opportunities.

### Correctness

Hand-written graph substitutions are error-prone, and a bug in graph substitutions can lead to incorrect computation graphs [Ten17, Gra18]. The same issue arises in compiler optimization, where an incorrect optimization leads to incorrect programs. In the compiler literature, significant effort has been devoted to formally verifying optimizations [Ban06, Chu19, Dah17, Le14, Nec00, Pnu98, Sha13, Tat11]. However, to the best of our knowledge, such techniques have not been applied to graph substitution optimizations performed by DNN frameworks.

### 1.1 Our Approach

In this paper, we present TASO (Tensor Algebra SuperOptimizer), the first DNN computation graph optimizer that automatically generates graph substitutions. [Figure 1b](#figure-01) shows an overview of TASO, which differs from existing frameworks in three aspects. First, TASO only requires operator definitions and specifications, and automatically generates graph substitutions, reducing manual effort. Second, TASO employs formal verification to ensure correctness of the generated graph substitutions. Finally, TASO jointly optimizes graph substitution and data layout, achieving significantly better runtime performance.

#### Generating substitutions

TASO's graph substitution generator enumerates all possible computation graphs over a given set of DNN operators (e.g., the cuDNN kernels [Che14]) up to a fixed size, and executes them on a set of random input tensors. Any pair of computation graphs that have identical results on the random inputs are considered as a candidate substitution. To efficiently find all such pairs, TASO constructs a hash table where computation graphs are stored based on the hash of their outputs for the random inputs.

#### Formal verification

TASO's graph substitution verifier is used to ensure correctness of the generated graph substitutions, relying on user provided operator properties. Operator properties capture mathematical properties of operators, e.g., linearity of convolution. The full list of 43 operator properties we used appears in [Table 2](#table-02). As our evaluation shows, a small set of properties for each operator suffices to prove the correctness of complex substitutions.

Formally, we model tensor operators using a symbolic representation based on first-order logic that is agnostic to the size of the underlying tensors, and can succinctly express operator properties. The verifier uses the specified properties to check the correctness of all generated graph substitutions using an automated theorem prover.

We also present a methodology for developing operator properties that assists developers in two ways: the graph substitution generator guides discovery of required properties, and symbolic execution on small tensors provides further validation. During development, this methodology uncovered several bugs in both the operator specifications and the graph substitution generator.

#### Joint optimization

TASO jointly optimizes graph substitutions and data layout transformations by integrating them into a common representation. TASO uses the cost-based backtracking search algorithm of MetaFlow [Jia19] and extends its cost model to also capture performance differences that arise from different data layouts. During the search, TASO measures the performance of a proposed DNN operator with a specific proposed data layout on the hardware. These individual measurements are used to predict the performance of an entire computation graph with specific data layouts.

#### Evaluation

We evaluate TASO on five real-world DNN architectures. For widely used DNNs optimized by existing frameworks, such as ResNet-50 [He16], TASO matches the performance of these frameworks with hand-written rules by using operator definitions and specifications 1,400 lines long.

For new DNN architectures such as ResNeXt-50 [Xie16], Nas-RNN [Zop16], NasNet-A [Zop18], and BERT [Dev18], TASO is up to 2.8× faster than state-of-the-art frameworks, by automatically discovering novel graph substitutions to optimize these architectures. Compared to sequentially optimizing graph substitutions and data layout, we show that the joint optimization can further improve performance by 1.2×. In all experiments, TASO discovered an optimized graph in less than ten minutes, making it feasible to use when optimizing a DNN architecture before large-scale deployment.

## 2 Graph Substitution Generator

This section describes the TASO substitution generator that automatically generates potential substitutions given a list of primitive operators. The generation algorithm finds all valid substitutions up to a certain size.

To find all potential substitutions, a straightforward approach is to test all pairs of graphs for equivalence, which requires a quadratic number of tests between graphs. We adopt an idea from compiler superoptimization [Ban06] and compute a fingerprint for each graph, which is a hash of the graph outputs on some specific inputs. Two graphs are certainly not equivalent if they have different fingerprints, and so by only comparing graphs with the same fingerprint, TASO significantly reduces the number of equivalence tests. In the experiments, we observe that all graphs with the same fingerprint are verified equivalent by TASO.

### 2.1 Graph Substitution Definition

A graph substitution consists of three components: (1) a source graph that is matched to subgraphs in a computation graph;

A target graph [+terminology] defines a functionally equivalent new subgraph to replace the matched source subgraph. The third component is a mapping relation between input and output tensors in the source and target graphs. [Figure 2a](#figure-02) shows a substitution based on matrix multiplication associativity. [Figure 2b](#figure-02) fuses two matrix multiplications using concatenation and split along the row dimension. A, B, C, X, and Y identify the mapping between source and target tensors.

A graph substitution is specified independently of the concrete tensor shapes. For example, the substitutions of [Figure 2](#figure-02) can be applied to tensors A, B, and C of any concrete shape. Some operators also depend on configuration parameters to determine the behavior of the operator. For example, the parameters of convolution determine the strides, padding, and activation (e.g., applying the relu function [Nai10] as part of convolution); and the parameters of split or concatenation determine the axis along which to apply the operator.

[+terminology]: In some of the superoptimization literature, what we call the source is called the target, and what we call the target is called the rewrite.

<span id="figure-02"></span>

![Two examples of graph substitutions](../../papers/taso/figure-02.png)

**Figure 2.** Graph substitution examples: (a) associativity of matrix multiplication; (b) fusing two matrix multiplications using concatenation and split.

#### Concatenation and split operators

Concatenation and split operators are commonly used in fusing operators with shared inputs, as illustrated in [Figure 2b](#figure-02). A split operator partitions a tensor into two disjoint sub-tensors along a dimension determined by its parameter. This presents a complication, as the split point cannot be inferred from the input tensors or the parameter. To solve this problem, we observe that a split operator always partitions a tensor at previous concatenation points to "undo" the most recent concatenation operator. We use this fact to define a suitable semantics for the split operator.

Formally, we maintain a split tree for each dimension of a tensor to track the concatenation history. [Figure 3](#figure-03) shows the split trees of the row dimension for all tensors in [Figure 2b](#figure-02). The split trees allow the substitution to recover the split point without introducing any additional parameters. Our approach also supports multi-way concatenation and split by nesting of concatenation and split operators.

<span id="figure-03"></span>

![A graph substitution for fusing matrix multiplications with a shared input](../../papers/taso/figure-03.png)

**Figure 3.** A graph substitution for fusing matrix multiplications with a shared input. The target graph has a concat and a split operator, both performed along the row dimension. The split tree of the row dimension for each tensor is shown in a gray box.

### 2.2 Generation Algorithm

For a given set of operator specifications, TASO generates potential graph substitutions in two steps, as shown in Algorithm 1.

Step 1: Enumerating potential graphs and collecting their fingerprints. TASO first enumerates all potential graphs up to a certain size by using a given set of operators. To construct a graph, TASO iteratively adds an operator in the current graph by enumerating the type of the operator and the input tensors to the operator. The input tensors can be initial input tensors to the graph (e.g., A, B, and C in [Figure 2](#figure-02)) or the output tensors of previous operators (e.g., the output of the matmul and concat operators in [Figure 2](#figure-02)). Algorithm 1 (line 7-18) shows a depth-first search algorithm for constructing all acyclic computation graphs that do not contain duplicated computation. We say a graph contains duplicated computation if it has two operators performing the same computation on the same input tensors. The generator ignores such graphs as they do not represent useful computation graphs.

For each graph, TASO collects its fingerprint, which is a hash of the output tensors obtained by evaluating the graph on some input tensors. TASO uses both randomly initialized tensors and a number of constants as inputs to allow finding substitutions involving constant tensors, such as the identity matrix (see examples in [Section 7.3](#_7-3-substitution-case-study)). To avoid floating-point errors in computing a fingerprint, all tensors are represented with integers, following the method introduced in [Wu18].

```pseudocode:line-numbers title="Algorithm 1: Graph substitution generation algorithm"
Input: A set of operators P, and a set of input tensors I.
Output: Candidate graph substitutions S.

// Step 1: enumerating potential graphs.
D = {}  // D is a graph hash table indexed by fingerprints.
Build(1, {}, I)
function Build(n, G, I)
  if G contains duplicated computation then
    return
  D = D + (Fingerprint(G), G)
  if n < threshold then
    for op in P do
      for i in I where i is a valid input to op do
        Add operator op into graph G.
        Add the output tensors of op into I.
        Build(n + 1, G, I)
        Remove operator op from G.
        Remove the output tensors of op from I.

// Step 2: testing graphs with identical fingerprints.
S = {}
for G1, G2 in D with the same Fingerprint(.) do
  if G1 and G2 are equivalent for all test cases then
    S = S + (G1, G2)
return S
```

Since a graph can have an arbitrary number of output tensors, the hash function must ensure that its fingerprint is independent of any permutation of the output tensors. To guarantee this property, TASO employs a two-step hash function:

$$\operatorname{Fingerprint}(G)=\operatorname{hash}_2\left(\{\operatorname{hash}_1(t_i)\mid i\in\operatorname{Outputs}(G)\}\right).$$

Here, $t_i$ are the output tensors of graph $G$. $\operatorname{hash}_1$ computes the state and content of an output tensor, including its size, shape, and content. $\operatorname{hash}_2$ is a symmetric hash function applied to an unordered set of hash values.

#### Step 2: Testing graphs with identical fingerprints

For graphs with the same fingerprint, TASO further examines their equivalence on a set of test cases. Similar to collecting fingerprints, each test case contains a set of randomized input tensors, and two graphs pass if they produce equivalent output tensors for all test cases. Unlike the fingerprints, these tests use floating point numbers ranging between $[-1,1]$, and classify two output tensors as equivalent if their outputs differ by no more than a small threshold value, which is $10^{-5}$ in the evaluation. For this threshold, we observed no discrepancy from the integer tests. However, it is possible to use a smaller threshold to filter out substitutions that are valid for real numbers but result in floating point errors.

Each pair of graphs passing the random testing becomes the source and target graphs of a candidate graph substitution, and the mapping relation between the input/output tensors in the source and target graphs can be automatically inferred from the test cases. All candidate graph substitutions are then sent to the substitution verifier to check their correctness ([Section 3](#_3-graph-substitution-verifier)), and later pruned to eliminate redundant substitutions ([Section 4](#_4-pruning-redundant-substitutions)).

The algorithm described so far is generic, in the sense that it does not depend on the specific tensor operators used. However, we observed that for DNN applications, there are two operators that require special handling. The relu operator [Nai10], which is commonly used in DNN applications, returns 0 for all negative inputs. As relu often returns 0, it results in many superfluous substitutions being valid. To prevent these substitutions from being generated, the generator replaces relu by an arbitrary non-linear function (our implementation uses $x \mapsto x(x+1)+1$). The enlarge operator increases the size of a tensor by padding it with zeros, which is useful for fusing convolutions with different kernel sizes [Jia19]. However, the presence of zeros also results in many superfluous substitutions. To overcome this, the generator only considers computation graphs in which enlarge is applied to an input tensor, i.e., not to the output of another operator. This restriction captures the intended use of enlarge for fusing convolutions, while avoiding the superfluous substitutions.

<span id="table-01"></span>

![Table of tensor operators and constant tensors included in TASO](../../papers/taso/table-01.png)

**Table 1.** Tensor operators and constant tensors included in TASO.

It is worth noting that prior work [Ban06] reported false positives when random testing was used to examine code transformations in compiler superoptimization: a number of incorrect transformations passed a set of tests. We have not observed any false positives in our experiments. We use a single test case to examine all graph pairs with the same fingerprint, and every substitution that passes the test case is correct and verified by the substitution verifier. This is likely due to the high arithmetic density of DNN operators and the lack of branching in computation graphs. As a reference, [Gul03] shows that for programs with only linear operators, the probability that two nonequivalent programs produce identical output on a random input is at most $1/d$, where $d$ is the number of possible values for a variable ($d=2^{32}$ in TASO).

## 3 Graph Substitution Verifier

The key idea behind our approach to formally verifying substitutions is to use a small set of operator properties expressed in first-order logic. These properties are manually written and reviewed, and are further validated by symbolically executing operators on tensors of small sizes and confirming that the operator properties are satisfied for these tensor sizes. Development of operator properties is guided by the substitutions discovered by the substitution generator.

For verification, tensor operators are modeled in first-order logic as functions of both their parameters and input tensors. For example, $\operatorname{conv}(s, p, c, x, y)$ represents convolution applied to tensors $x$ and $y$; $s$ determines the stride, $p$ the padding mode, and $c$ the activation mode. The fact that convolution without activation ($A_{\mathrm{none}}$) is linear in its first argument is captured by the following property, where $\operatorname{ewadd}$ is element-wise tensor addition:

$$\forall s,p,x,y,z.\ \operatorname{conv}(s,p,A_{\mathrm{none}},\operatorname{ewadd}(x,y),z)=\operatorname{ewadd}(\operatorname{conv}(s,p,A_{\mathrm{none}},x,z),\operatorname{conv}(s,p,A_{\mathrm{none}},y,z)).$$

[Table 1](#table-01) lists all operators and tensor constants used in the evaluation, and [Table 2](#table-02) gives the complete list of operator properties used to verify graph substitutions.

<span id="table-02"></span>

![Operator properties used for verification](../../papers/taso/table-02.png)

**Table 2.** Operator properties used for verification.

Given the operator properties, we use a first-order theorem prover; our implementation uses Z3 [DeM08]. Verification amounts to entailment checking in first-order logic: the operator properties must entail the functional equivalence of the source and target graphs for each generated substitution.

Modeling the operators using first-order logic involves a degree of abstraction (e.g., the shapes of tensors are not modeled). We found this level of abstraction to be suitable for verifying graph substitutions. We also note that the data layout is abstracted for verification purposes-layout does not affect operator semantics, and the optimizer ([Section 5](#_5-joint-optimizer)) ensures that layouts are used consistently.

#### Methodology for developing operator properties

We developed operator properties as needed to determine the correctness of generated graph substitutions using an iterative process. During the development process, we ran the substitution generator and tried to verify all discovered substitutions. If a substitution could not be verified and appeared correct, we added an appropriate property (or properties). To safeguard against mistakes in operator properties, we used further validation steps.

To validate operator properties, TASO verifies the operator properties themselves for all combinations of parameter values and tensor sizes up to a small bound-in our evaluation the bound was 4×4×4×4. For this, TASO requires a basic symbolic implementation of each tensor operator in Python. TASO symbolically executes this implementation for tensors of small size, effectively elaborating the tensor operations into symbolic real arithmetic expressions, where activation functions (e.g., relu) are modeled using uninterpreted functions. TASO then uses Z3, here as an SMT solver for the theory of real arithmetic, to verify the operator properties. For example, if a user would try to add the (wrong) property stating the convolution operator is linear for all activation modes (including relu activation), then this check would show that this property is not satisfied by the actual operators.

As an additional validation step that assists the development process, TASO checks that the set of operator properties is consistent and does not contain redundancies (i.e., a property entailed by other properties), which amounts to first-order entailment checks. These checks are also useful for discovering erroneous properties, and are cheaper to perform than the verification for small tensor sizes.

During our development process, the verification methodology revealed several subtle bugs. Some bugs in the graph substitution generator were found when it generated substitutions that could not be verified, and the validation steps described above revealed several bugs in candidate operator properties. In our experience, a new operator can be supported with a small amount of effort, usually a few hours of work by an expert. Typically a few properties must be written for each operator. In our evaluation, we were able to Matmul and Add refer to matrix multiplication and elementwise addition, respectively. For each subgraph, A, B, and C refer to its input tensors, while X refers to the output tensor. verify all 743 generated graph substitutions using 43 operator properties (see [Table 2](#table-02)).

## 4 Pruning Redundant Substitutions

A graph substitution is redundant if it is subsumed by a more general valid substitution. This section describes TASO's pruning techniques. All pruning steps preserve every optimization opportunity: if graph $G$ can be transformed into $G'$ using a sequence of substitutions, then $G$ can still be transformed into $G'$ after pruning, possibly using a different set of transformations.

<span id="figure-04"></span>

![Examples of redundant substitutions pruned by TASO](../../papers/taso/figure-04.png)

**Figure 4.** Example redundant substitutions pruned by TASO. Matmul and Add refer to matrix multiplication and element-wise addition. For each subgraph, A, B, and C are its input tensors, while X is the output tensor.

#### Input tensor renaming

TASO eliminates graph substitutions identical to other substitutions modulo input tensor renaming. [Figure 4a](#figure-04), for example, is equivalent to [Figure 2a](#figure-02) after renaming input tensor C to A. For substitutions that are equivalent through input tensor renaming, TASO prunes all but a single most general substitution.

#### Common subgraph

TASO also tries to eliminate substitutions whose source and target graphs have a common subgraph. TASO identifies two forms of common subgraphs that can lead to pruning. The first is illustrated in [Figure 4b](#figure-04). The source and target graphs both contain a common operator with the same input tensors, highlighted in gray. This common subgraph is an input to other operators in both graphs. A more general substitution can therefore be obtained by replacing the common subgraph with a fresh input tensor. If that more general substitution is valid, TASO prunes the less general substitution.

The second form of common subgraph is demonstrated in [Figure 4c](#figure-04). Here, the common subgraph (highlighted in gray boxes) includes all the outputs in both the source and target graphs. In this case, a more general substitution can be obtained by completely removing the common subgraph, making its inputs new outputs of the source and target graphs. TASO prunes the less general substitution if the more general one is valid.

[Table 3](#table-03) shows the effect of the TASO pruning techniques on the number of substitutions. We observe that both pruning techniques play an important role in eliminating redundant substitutions and their combination reduces the number of substitutions TASO must consider by 39×.

<span id="table-03"></span>

![Number of graph substitutions remaining after pruning](../../papers/taso/table-03.png)

**Table 3.** The number of remaining graph substitutions after applying the pruning techniques in order.

## 5 Joint Optimizer

We now describe the TASO optimizer for jointly optimizing data layout and graph substitution. The optimizer uses the MetaFlow [Jia19] cost-based backtracking search algorithm to search for an optimized computation graph by applying verified substitutions. TASO extends MetaFlow's search algorithm to also consider possible layout optimization opportunities when performing substitutions.

When applying a substitution to a matched subgraph, TASO enumerates possible layouts for tensors in the target graph based on the source tensors and the layouts supported by each operator. Applying a substitution may therefore produce multiple graphs with identical structure but different data layouts.

<span id="figure-05"></span>

![A graph substitution using the transpose of matrix multiplication](../../papers/taso/figure-05.png)

**Figure 5.** A graph substitution using the transpose of matrix multiplication. The parentheses show potential tensor layouts, where C and R indicate column-major and row-major layouts.

[Figure 5](#figure-05) shows the potential graphs derived by applying the transpose-of-matrix-multiplication substitution to a source graph with a default column-major layout. Matrix multiplication and transpose also support row-major layout. The layouts of mapped tensors A, B, and X must match between source and target graphs, while the two intermediate tensors can independently use row- or column-major layout. TASO consequently considers CC, CR, RC, and RR, allowing graph substitutions to capture layout transformation opportunities.

```pseudocode:line-numbers title="Algorithm 2: Cost-based backtracking search"
Input: An input graph G_in, verified substitutions S,
  a cost model Cost(.), and hyperparameter alpha.
Output: An optimized graph.

P = {G_in}  // Priority queue sorted by Cost.
while P is not empty do
  G = P.dequeue()
  for substitution s in S do
    for layout l in Layout(G, s) do
      G' = Apply(G, s, l)
      if G' is valid then
        if Cost(G') < Cost(G_opt) then
          G_opt = G'
        if Cost(G') < alpha * Cost(G_opt) then
          P.enqueue(G')
return G_opt
```

Algorithm 2 shows our cost-based backtracking search algorithm for jointly optimizing substitution and data layout. The cost model is motivated by the fact that DNN operators perform dense linear algebra with no branches, and therefore their performance on hardware is highly consistent and predictable given the same data layouts and configuration parameters (e.g., the strides and padding of a convolution). Similar to MetaFlow [Jia19], TASO measures the execution time of a DNN operator once for each configuration and data layout, and estimates the performance of a graph by summing up the measured execution time of its operators.

To search for an optimized graph, all candidate graphs are maintained in a priority queue $P$ and are dequeued in increasing order of cost. For each dequeued graph $G$, TASO considers each verified substitution and possible layouts applicable to the substitution, and applies them to obtain functionally equivalent new graphs $G'$.

A non-obvious property of graph substitutions is that applying them can introduce cycles into a graph. [Figure 6](#figure-06) shows one example where applying a valid substitution results in a cyclic graph. Since computation graphs must be acyclic, TASO checks the acyclicity of $G'$ (line 12 of Algorithm 2) before enqueuing it in $P$.

<span id="figure-06"></span>

![A graph substitution that introduces a cycle](../../papers/taso/figure-06.png)

**Figure 6.** A valid graph substitution can introduce a cycle into a computation graph. The source and target subgraphs are shown by dotted boxes, and the resulting cycle is highlighted in red.

Finally, the best discovered graph $G_{\mathrm{opt}}$ is returned by the search algorithm. The search space is pruned by a hyperparameter $\alpha$, which directly eliminates all graphs whose cost is $\alpha$ times worse than the best discovered graph. The parameter $\alpha$ trades off between the search time and the best discovered graph. Setting $\alpha=1$ reduces the search to a simple greedy algorithm without backtracking, and a high value for $\alpha$ makes the search explore more possible candidates and causes more backtracking. We observe that $\alpha=1.05$ achieves good performance in our evaluation.

## 6 Implementation

TASO is designed and implemented as a generic and extensible computation graph optimizer for tensor computations, such that new tensor operators can be easily added. [Table 1](#table-01) lists the tensor operators included in the current implementation of TASO. Some operators also depend on additional parameters to determine the behavior of the operator, such as the strides, padding, and activation of a convolution. In addition to operators, TASO also includes four types of constant tensors that are useful in substitutions. In particular, $I_{\mathrm{ewmul}}$, $I_{\mathrm{matmul}}$, and $I_{\mathrm{conv}}$ are identity tensors for element-wise multiplication, matrix multiplication, and convolution, respectively. $C_{\mathrm{pool}}$ allows converting an average pooling operator to a depth-wise convolution (see examples in [Section 7.3](#_7-3-substitution-case-study)).

As explained in [Section 3](#_3-graph-substitution-verifier), TASO uses operator properties specified by the user to verify the generated graph substitutions. [Table 2](#table-02) lists the 43 properties used to verify all substitutions in our evaluation.

TASO can easily be extended to include new tensor operators. For each operator, TASO requires two forms of input: (1) reference implementations for the operator, and (2) specifications of operator properties. (1) consists of a concrete implementation (in C++) used by the substitution generator and a symbolic implementation (in Python) used to validate the operator specifications. In our experience, adding a new operator requires a few hours of work by an expert.

For a new operator whose specifications are currently missing, TASO treats it as an opaque operator and can still optimize the rest of the graph using verified substitutions.

TASO is implemented on top of MetaFlow, and reuses the MetaFlow cost-based backtracking search [Jia19]. Overall, our implementation of TASO contains around 8,000 lines of code for the core components (i.e., the substitution generator, verifier, and optimizer), and 1,400 lines of code for the operator reference implementations, including the 43 operator properties.

TASO is framework-agnostic and can be plugged in to existing DNN frameworks such as TensorRT and TVM by simply emitting the optimized graph in the target framework's input format. In the evaluation, we demonstrate this portability on TensorRT and TVM, and show that they can directly use TASO's optimizations to improve performance.

## 7 Evaluation

This section evaluates the following questions:

- Can TASO automatically generate and verify graph substitutions in an acceptable amount of time?
- Can TASO improve end-to-end performance on real-world DNN architectures, especially emerging architectures with recently introduced operators?
- Can joint optimization of computation graphs and data layouts outperform separate optimizations?

### 7.1 Experimental Setup

We use five real-world DNN architectures to evaluate TASO. ResNet-50 [He16] is a widely used convolutional neural network for image classification and achieved the best classification performance in the ILSVRC [Rus15] competition. ResNeXt-50 [Xie16] improves the model accuracy and runtime efficiency of ResNet-50 by introducing a new grouped convolution operator. NasNet-A [Zop18] and NasRNN [Zop16] are two DNN architectures automatically discovered by machines through neural architecture search. NasNet-A and NasRNN exceed the best human-designed DNN architectures for image classification and language modeling tasks, respectively. Finally, BERT [Dev18] is a new language representation architecture that obtained the state-of-the-art model accuracy on a spectrum of language tasks. All experiments were performed on an Amazon p3.2xlarge instance [Ama17] with an 8-core Intel E5-2600 CPU, 64 GB DRAM, and one NVIDIA Tesla V100 GPU.

To generate candidate graph substitutions, TASO enumerates all potential graphs with up to four operators by using all DNN operators listed in [Table 1](#table-01). TASO generated 743 candidate substitutions in around 5 minutes.

In the cost-based backtracking search for optimized DNN graphs, we set the hyperparameter $\alpha$ to be 1.05, which is identical to the value used in MetaFlow [Jia19]. In all experiments, the end-to-end search time to discover an optimized computation graph is less than ten minutes.

### 7.2 End-to-End Evaluation

We first compare the end-to-end inference performance among TensorFlow [Aba16], TensorFlow XLA [XLA17], TensorRT [Ten17a], TVM [Che18], MetaFlow [Jia19], and TASO on a V100 GPU. [Figure 7](#figure-07) shows the results. TensorFlow, TensorFlow XLA, TensorRT, and MetaFlow use the highly-engineered cuDNN and cuBLAS libraries [Che14, Cub16] to perform DNN operators on GPUs, while TVM generates customized GPU kernels for the DNN operators. To eliminate the impact of different operator libraries, we evaluate the performance of TASO on both backends.

To generate GPU kernels in TVM, we allow the auto tuner [Che18a] to run 2000 trials and use the best discovered configuration for each DNN operator. It takes 2 hours on average to tune a GPU kernel for each DNN operator. The TASO graph optimizer needs to query the execution time of hundreds of DNN operators for its cost model, therefore, for the TVM backend, we reuse the best discovered computation graph for the cuDNN backend, assuming the cost of an operator in cuDNN is a reasonable estimate for its cost in TVM.

Among the five DNN architectures, ResNet-50 has been commonly used and heavily optimized by existing DNN frameworks. TASO achieves on-par performance for ResNet-50 with existing frameworks, showing that TASO is able to automatically discover graph substitutions manually designed by domain experts. For the remaining four DNN architectures with new operators and graph structures, TASO outperforms existing DNN frameworks with speedups ranging from 1.3× to 2.8× on the cuDNN backend and 1.1× to 1.8× on the TVM backend. The speedup is achieved by (1) automatically discovering optimizing substitutions for the new operators and (2) jointly optimizing graph substitution and data layout. We analyze the substitutions discovered by TASO in [Section 7.3](#_7-3-substitution-case-study) and [Section 7.4](#_7-4-analysis-of-used-substitutions), and the joint optimization of substitution and data layout in [Section 7.5](#_7-5-joint-optimization-of-graph-substitutions-and-data-layout).

<span id="figure-07"></span>

![End-to-end inference performance of existing DNN frameworks and TASO](../../papers/taso/figure-07.png)

**Figure 7.** End-to-end inference performance on a single NVIDIA V100 GPU. Each result averages 1,000 runs; the labels above TASO bars show speedup over the best existing approach using the same backend.

### 7.3 Substitution Case Study

To understand how the substitutions generated and verified by TASO improve runtime performance, we study a few graph substitution examples in detail.

NasNet-A is the best discovered CNN architecture for the CIFAR-10 dataset, obtained by neural architecture search. [Figure 8a](#figure-08) shows a convolutional cell in NasNet-A. Unlike human-designed architectures, NasNet-A contains unconventional graph structures, making it hard to optimize with manual substitutions designed for more standard DNN architectures. To illustrate how TASO optimizes this architecture, we show two example substitutions discovered by TASO; neither is present in any existing DNN framework.

<span id="figure-08"></span>

![NasNet-A architecture and substitutions discovered by TASO](../../papers/taso/figure-08.png)

**Figure 8.** The NasNet-A architecture and substitutions discovered by TASO. Colored boxes identify source and target subgraphs; gray regions depend only on pretrained weights and can be precomputed.

[Figure 8b](#figure-08) shows graph substitutions that transform two average pooling operators followed by element-wise addition to a single depth-wise convolution, by using a constant tensor $C_{\mathrm{pool}}$ defined in [Table 1](#table-01). The mathematical formula for average pooling is:

$$o(n,c,x,y)=\frac{1}{K_XK_Y}\sum_{k_x}\sum_{k_y}i(n,c,x+k_x,y+k_y).$$

Here, $K_X$ and $K_Y$ are the height and width of the pooling filter. The corresponding formula for depth-wise convolution is:

$$o(n,c,x,y)=\sum_{k_x}\sum_{k_y}i(n,c,x+k_x,y+k_y)\times w(c,k_x,k_y),$$

which is mathematically equivalent to average pooling when $w(c,k_x,k_y)=1/(K_XK_Y)$. TASO also fuses the two depth-wise convolutions using linearity.

In addition, TASO also fuses the two depth-wise convolutions into one using its linearity.

A second new sequence of substitutions for NasNet-A is shown in [Figure 8c](#figure-08), which fuses two depth-wise convolutions and two convolutions followed by addition to a depth-wise convolution followed by a standard convolution. This substitution increases the operator granularity and reduces the operator launch overhead by using larger operators.

For inference workloads, DNN weights such as $W_i$ and $C_{\mathrm{pool}}$ in [Figure 8](#figure-08) are fixed and independent of the inputs. TASO preprocesses operators whose inputs are all pretrained weights, such as the gray regions in [Figure 8](#figure-08), to further reduce inference time.

ResNeXt-50 replaces large ResNet-50 convolutions with multiple branches of smaller convolutions to improve accuracy and runtime efficiency, as shown in [Figure 9a](#figure-09). Launching each small convolution separately incurs high kernel-launch overhead. Grouped convolution kernels execute multiple convolutions in a single CUDA kernel, but grouping all 32 convolutions requires more cache for intermediate state and can also reduce performance. [Figure 9d](#figure-09) shows that neither separate convolutions nor one group of 32 is optimal.

<span id="figure-09"></span>

![Multi-batch convolution strategies in ResNeXt-50 and their performance](../../papers/taso/figure-09.png)

**Figure 9.** Multi-batch convolution strategies in ResNeXt-50. TASO uses four grouped convolutions of eight convolutions each, a mixture it discovered automatically, and achieves a 2.8x speedup over the best existing approach.

### 7.4 Analysis of Used Substitutions

We now present a detailed analysis of how the graph substitutions discovered by TASO impact the performance of the optimized graphs. [Figure 10](#figure-10) shows a heat map of the substitutions used to optimize each of the five DNN architectures. Each DNN uses 4-10 different substitutions to achieve optimized performance, and different DNNs require different sets of substitutions. This shows the difficulty of manually designing a few core substitutions to optimize today's DNN architectures with increasingly high diversity. TASO is better positioned for optimizing new DNNs by automatically discovering performance critical substitutions.

<span id="figure-10"></span>

![Heat map of graph substitutions used by five DNN architectures](../../papers/taso/figure-10.png)

**Figure 10.** How often verified substitutions are used to optimize five DNN architectures. Only substitutions used by at least one DNN are shown.

Additionally, we evaluate the scalability of TASO by considering substitutions with different size limitations, and measuring the runtime performance of the optimized graphs. [Figure 11](#figure-11) shows the results. For all three DNN architectures, performance improvement is consistently achieved by using larger substitutions up to size 3. ResNeXt-50 and BERT do not obtain additional speedups by using substitutions with 4 operators, while NasNet-A achieves 1.2× by considering larger substitutions. Our current implementation of TASO does not scale to generate all substitutions with 5 or more operators, since the generator is limited by the memory needed to hold the fingerprints of all potential graphs, which scales exponentially with graph size. A distributed fingerprint generator could potentially handle graphs of size 5 and even more, which we leave as future work.

<span id="figure-11"></span>

![Performance with different graph substitution size limits](../../papers/taso/figure-11.png)

**Figure 11.** Relative speedups obtained with different maximum graph substitution sizes.

### 7.5 Joint Optimization of Graph Substitutions and Data Layout

To evaluate joint optimization, TASO is compared with three baselines: graph substitution optimization only, data layout optimization only, and the two optimizations applied sequentially.

[Figure 12](#figure-12) compares the four strategies on BERT. TASO outperforms the baselines by 1.2-1.3x using substitutions that transform both graph structure and data layout. One example appears in [Figure 5](#figure-05). BERT's most expensive operation is $A\times B$, where A is 64 by 1024 and B is 1024 by 4096. In cuBLAS, the transposed form, $(B^T\times A^T)^T$, is 1.5x faster when $B^T$ and $A^T$ use column-major and row-major layout, respectively. This optimization can only be captured when graph substitution and data layout are considered jointly.

<span id="figure-12"></span>

![BERT inference time for graph and layout optimization strategies](../../papers/taso/figure-12.png)

**Figure 12.** End-to-end BERT inference performance with separate, sequential, and joint graph substitution and data layout optimization.

### 7.6 Graph Substitution Verifier

We evaluate the performance of the graph substitution verifier for its two key tasks: verifying generated substitutions against operator specifications, and validating the operator specifications themselves to aid in the development process ([Section 3](#_3-graph-substitution-verifier)). Our implementation uses Z3 [DeM08] to automatically discharge all proof obligations, and our experiments were performed with Z3 version 4.8.5. Generating the 743 graph substitutions takes around five minutes, and verifying them against the 43 specified operator properties takes less than 10 minutes. When checking the specification for redundancies we use Z3 to search for a proof of an invalid formula (stating that a specified property is entailed by the rest of the specification). This search can continue indefinitely, and in our evaluation we used a timeout of 10 seconds per query, resulting in a run time of less than 10 minutes (for 43 axioms). During the development process, when we had some redundant specifications they were discovered in a few seconds.

The validation check that verifies the operator specification for all combinations of parameter values and tensor sizes up to 4×4×4×4 is more computationally expensive, with roughly one million proof obligations. We parallelized it using 128 CPU cores, which resulted in a run time of roughly one hour. During the development process, we also found it useful to verify the operators for more restricted combinations. For example, verifying the specification for tensors of size exactly 4×4×4×4 (rather than all tensors up to that size) takes under 10 minutes using a single CPU core.

## 8 Related Work

Manually designed graph substitutions are used in existing DNN frameworks to optimize DNN architectures. For example, TensorFlow, TensorRT, and TVM use a rule-based strategy and directly perform all applicable substitutions on an input graph [Aba16, Che18, Ten17a]. MetaFlow [Jia19] allows users to define performance-decreasing substitutions to obtain a larger space of potential graphs. The key difference between TASO and these frameworks is that TASO can automatically generate candidate substitutions, and also provides semi-automatic support for verifying their correctness. In the evaluation, we also show that existing frameworks can directly use TASO's optimized graphs to improve performance.

### Automated DNN code generation

Recent work has proposed various approaches to generate hardware-specific code for DNN operators. For example, TVM [Che18, Che18a] uses a learning-based approach and automatically generates low-level optimized code for a diverse set of hardware backends. Astra [Siv19] optimizes DNN computation by exploring the optimization space of multi-version complication during training. Compared to these approaches, TASO aims at optimizing DNN computation at a higher graph level, and therefore TASO's optimizations are orthogonal and can be combined with code generation techniques. It still remains an open problem of how to jointly optimize DNN computation at both graph-level and operator-level.

### Automated DNN parallelization

ColocRL [Mir17] uses reinforcement learning to automatically discover an efficient device placement for parallelizing DNN training across multiple GPUs. FlexFlow [Jia18, Jia19a] introduces a comprehensive search space of parallelization strategies for DNN training, and uses a randomized search algorithm to find efficient strategies in the search space. These frameworks optimize distributed DNN training assuming a fixed computation graph. We believe it is possible to combine TASO's graph optimizations with training parallelization techniques.

### Superoptimization

Superoptimization is a compiler optimization technique that was originally designed to find the optimal code for a sequence of instructions [Mas87]. TASO's approach to identifying potential substitutions via enumeration of graphs and fingerprinting is similar to work in automatically generating peephole optimizers using superoptimization techniques [Ban06]. TASO's approach to verification, however, is significantly different. Verification in superoptimization typically relies on "bit blasting", that is, modeling every bit in a computation explicitly in a logical formula (e.g., as a boolean variable). This approach is possible only when all aspects of a program transformation, including the computation and the data, can be expressed using a known number of bits. For TASO, where the input tensor sizes for graph substitutions are unknown, we must take a different approach. While not fully automatic like verification via bit blasting, our methodology based on writing operator specifications is much more flexible in being able to model future operators with almost arbitrary semantics, in addition to smoothly handling the issue of unknown tensor dimensions and split points.

### Data layout optimizations

Existing DNN frameworks that support data layout optimizations treat data layouts and graph transformations as separate optimization problems [Che18, Li16, Mkl16]. TASO formulates the problem of performing graph substitutions and deciding the data layout of each DNN operator as a joint optimization problem and considers layout conversions as a part of graph substitutions. As a result, TASO can automatically generate graph substitutions that optimize both graph structures and data layouts, and our evaluation shows that jointly optimizing the two tasks can significantly improve the end-to-end performance, compared to optimizing the them separately.

## 9 Limitations and Future Work

One limitation of TASO is the reliance on user provided operator properties. While our experience has been that the required effort is manageable, it would be better to eliminate it altogether. One possible approach is to automatically verify substitutions directly against the implementations of the operators, e.g., cuDNN kernels.

Another limitation of TASO is the scalability of the generator, which requires saving the fingerprints of all computation graphs up to a fixed size. This approach currently does not scale beyond graphs of size 4. One possible approach to scale to larger graphs is to implement a distributed generator. A second possibility is to replace the brute-force enumeration with more efficient algorithms or heuristics.

An additional avenue for future research is combining graph-level and operator-level optimizations. This joint optimization is challenging as both problems involve large and complex search spaces, and optimizations at one level affect the search space of the other.

## 10 Conclusion

TASO is the first DNN computation graph optimizer that automatically generates graph substitutions. TASO formally verifies the substitutions, and considers graph substitutions and layout transformations together as a joint optimization problem, exploiting more optimization opportunities. TASO matches the performance of existing frameworks on DNNs for which these frameworks have been heavily optimized such as ResNet-50, and outperforms existing frameworks by up to 2.8× on other DNNs, finding novel optimizations not present in the hundreds of optimization rules in existing frameworks. TASO achieves these results with dramatically less human effort than existing frameworks, and provides a higher level of correctness guarantees.

## Acknowledgments

We thank Nikolaj Bjørner, Mingyu Gao, Vinod Grover, Sina Lin, Feng Ruan, Xi Wang, the anonymous SOSP reviewers, and our shepherd, Joey Gonzalez, for their helpful feedback. This work was supported by NSF grant CCF-1409813, the Exascale Computing Project (17-SC-20-SC), a collaborative effort of the U.S. Department of Energy Office of Science and the National Nuclear Security Administration, and is based on research sponsored by DARPA under agreement number FA84750-14-2-0006. This research was supported in part by affiliate members and other supporters of the Stanford DAWN project-Ant Financial, Facebook, Google, Infosys, Intel, Microsoft, NEC, SAP, Teradata, and VMware-as well as Cisco and the NSF under CAREER grant CNS-1651570. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the NSF.
