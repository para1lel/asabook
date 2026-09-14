---
title: 'Event Tensor: Dynamic Megakernel Compiler'
createTime: 2026/09/14 22:50:53
permalink: /en/papers/event-tensor/
---

> [Hongyi Jin](https://dblp.org/pid/321/1565), [Bohan Hou](https://spectrometerhbh.github.io/), [Guanjie Wang](https://dblp.org/pid/166/5270.html), [Ruihang Lai](https://ruihanglai.com/), [Jinqi Chen](https://www.cs.cmu.edu/afs/cs.cmu.edu/Web/Posters/MSCSThesis-5-JinqiChen25.pdf), [Zihao Ye](https://expye.com/), [Yaxing Cai](https://dblp.org/pid/290/7679.html), [Yixin Dong](https://github.com/Ubospica), [Xinhao Cheng](https://www.csd.cs.cmu.edu/people/doctoral-student/xinhao-cheng), [Zhihao Zhang](https://jackfram.github.io/), [Yilong Zhao](https://ylzhao.me/), [Yingyi Huang](https://mlsys26.flashinfer.ai/), [Lijie Yang](https://derrickylj.github.io/), [Jinchen Jiang](https://dblp.org/pid/307/3686.html), [Gabriele Oliaro](https://www.gabrieleoliaro.com/), [Jianan Ji](https://jiananji.me/), [Xupeng Miao](https://hsword.github.io/), [Vinod Grover](https://developer.nvidia.com/blog/author/vgrover/), [Todd C. Mowry](https://www.cs.cmu.edu/~tcm/), [Zhihao Jia](https://www.cs.cmu.edu/~zhihaoj2/), and [Tianqi Chen](https://tqchen.com/). First submitted to arXiv on April 14, 2026; current version v2, revised April 21, 2026. Published in [Proceedings of Machine Learning and Systems 8 (MLSys 2026)](https://proceedings.mlsys.org/paper_files/paper/2026/hash/53d3f45797970d323bd8a0d379c525aa-Abstract-Conference.html). This reading edition transcribes [*Event Tensor: A Unified Abstraction for Compiling Dynamic Megakernel*](https://arxiv.org/abs/2604.13327v2), with the <a href="/paper/event-tensor.pdf" target="_blank" rel="noopener noreferrer">original PDF</a>, [arXiv DOI](https://doi.org/10.48550/arXiv.2604.13327), and [TeX source](https://arxiv.org/src/2604.13327v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Modern GPU workloads, especially large language model (LLM) inference, suffer from kernel launch overheads and coarse synchronization that limit inter-kernel parallelism. Recent megakernel techniques fuse multiple operators into a single persistent kernel to eliminate launch gaps and expose inter-kernel parallelism, but struggle to handle dynamic shapes and data-dependent computation in real workloads. We present *Event Tensor*, a unified compiler abstraction for dynamic megakernels. Event Tensor encodes dependencies between tiled tasks, and enables first-class support for both shape and data-dependent dynamism. Built atop this abstraction, our Event Tensor Compiler (ETC) applies static and dynamic scheduling transformations to generate high-performance persistent kernels. Evaluations show that ETC achieves state-of-the-art LLM serving latency while significantly reducing system warmup overhead.

<span id="section-1"></span>

## 1 Introduction

Efficient deployment of machine learning (ML) applications requires minimizing latency and maximizing hardware utilization, making system performance optimization [Kwo23, Zhu25c, Ye25, Zho24] a critical research frontier. As GPUs continue to scale in speed and parallelism, several forms of system overhead in conventional GPU scheduling models have emerged as dominant bottlenecks that constrain end-to-end efficiency.

The first source of overhead arises from kernel launches. Current systems such as PyTorch [Pas19a] launch GPU kernels sequentially from the host CPU ([Figure 1](#figure-01), upper left). During LLM inference, each auto-regressive decoding step may involve hundreds or even thousands of fine-grained operations, where the launch overhead cannot be effectively amortized. Each kernel launch typically incurs 5–10 $\mu$s of latency, while the fastest kernels may complete in 2 $\mu$s, making the launch overhead dominant.

The second source of overhead stems from kernel boundaries, which enforce implicit synchronization between consecutive kernels. In many cases, later kernels depend only on a subset of results from prior ones; in principle, these kernels could be overlapped or pipelined to improve throughput. However, the boundaries between kernels hinder such fine-grained inter-kernel parallelism, leaving significant performance on the table.

<span id="figure-01"></span>

![Figure 1. Different GPU scheduling models. Kernel-by-kernel and CUDA Graph scheduling models enforce a coarse-grained sequential execution. Megakernels break operations into smaller tasks, achieving inter-kernel parallelism.](../../papers/event-tensor/figure-01.png)

**Figure 1.** Different GPU scheduling models. Kernel-by-kernel and CUDA Graph scheduling models enforce a coarse-grained sequential execution. Megakernels break operations into smaller tasks, achieving inter-kernel parallelism.

<span id="figure-02"></span>

![Figure 2. Event Tensor abstraction overview. A computation graph (left) is partitioned into tiled operators (*tasks*), and the Event Tensor captures fine-grained dependencies between tasks as a first-class, symbolic-shaped object, handling the primary sources of dynamism inherent to LLM serving: Shape Dynamism: Tiled tensors and Event Tensors have symbolic dimensions, such as the dynamic batch size **B**. Data-Dependent Dynamism: The MoE layer (right) details how dependencies are resolved at runtime. Data-dependent updates and triggers (yellow arrows) use runtime-computed values such as `topk` and `exp_indptr` to dynamically manage the task execution.](../../papers/event-tensor/figure-02.png)

**Figure 2.** Event Tensor abstraction overview. A computation graph (left) is partitioned into tiled operators (*tasks*), and the Event Tensor captures fine-grained dependencies between tasks as a first-class, symbolic-shaped object, handling the primary sources of dynamism inherent to LLM serving: Shape Dynamism: Tiled tensors and Event Tensors have symbolic dimensions, such as the dynamic batch size **B**. Data-Dependent Dynamism: The MoE layer (right) details how dependencies are resolved at runtime. Data-dependent updates and triggers (yellow arrows) use runtime-computed values such as `topk` and `exp_indptr` to dynamically manage the task execution.

Several recent efforts have attempted to partially address these limitations. First, many systems adopt runtime techniques such as CUDA Graphs [Gra19] ([Figure 1](#figure-01), upper right), which reduce kernel launch overhead by capturing and replaying a fixed sequence of kernels. However, CUDA Graphs preserve kernel boundaries and thus cannot expose inter-kernel parallelism.

More recently, *megakernel* optimizations [Spe25, Che25ae] have emerged as a promising alternative ([Figure 1](#figure-01), lower). The key idea is to fuse multiple operators into a single persistent kernel, eliminating kernel launch overheads and enabling inter-kernel parallelism. Each operator is decomposed into fine-grained tiles of computation, or *tasks*, which are distributed across streaming multiprocessors (SMs). The tasks and their dependencies form a task graph, whose execution is orchestrated through lightweight runtime signaling to preserve dependency while maximizing concurrency.

Despite its promise, deploying megakernels for LLM inference workloads remains challenging for two key reasons:

**Dynamism challenges.** Modern LLM serving workloads are inherently dynamic. With the introduction of continuous batching, the system must handle variable input shapes. Supporting such dynamic shapes within a single megakernel is challenging, as it often requires regenerating or recompiling the kernel for every possible shape. This can incur prohibitive startup latency or become impossible when the space of potential shapes is too large. Furthermore, models such as Mixture-of-Experts (MoE) introduce data-dependent control flow (e.g., expert routing), which requires dynamic tracking of fine-grained task dependencies to exploit inter-operator parallelism. Current approaches lack abstractions to express such fine-grained data dependency. Notably, this challenge is not unique to megakernels—runtime dynamism also poses significant difficulties for conventional methods such as CUDA Graphs, where recapturing and managing CUDA Graphs across dynamic shapes is a major pain point for production LLM serving systems. These dynamism challenges are particularly important for emerging latency-sensitive applications such as real-time agentic workflows and interactive coding assistants, where low-batch inference dominates and inter-kernel parallelism is critical for reducing per-request latency.

**Programmability challenges.** Megakernel programming introduces substantial complexity. Developers must reason about complex, fine-grained dependencies among tasks, which are error-prone and difficult to maintain. The shape and data-dependent dynamism further complicate this process. Moreover, multiple task-management strategies may be desirable depending on the workload. For instance, *static scheduling* assigns each SM a predetermined queue of tasks before kernel launch, while *dynamic scheduling* employs an on-GPU scheduler to dispatch ready tasks at runtime. Ideally, developers should be able to seamlessly select or combine these scheduling strategies without reimplementing the entire kernel, enabling workload-specific scheduling.

In this paper we present **Event Tensor** ([Figure 2](#figure-02)), an abstraction designed to simplify the compilation and execution of dynamic megakernels. We define an *event* as a primitive representing the completion of a set of tasks at the granularity of GPU SMs. Because megakernels partition operators into a large number of tile-level tasks, the corresponding synchronization events naturally form multi-dimensional structures similar to data tensors. An *Event Tensor* is a multi-dimensional array of such events, providing a compact, first-class representation for fine-grained synchronization within a megakernel. While semaphore-based synchronization is a well-known primitive, our core novelty lies in elevating these primitives into first-class tensors within the compiler IR. By unifying events into tensor form, Event Tensors leverage existing compiler support for symbolic shapes [Ans24, Lai25a], allowing tensor dimensions to remain symbolic and thereby compactly representing dynamic-shape computations. Furthermore, Event Tensors express data-dependent dynamism through index expressions that map task coordinates to event coordinates. Built upon this abstraction, ETC automatically transforms these dependencies into highly optimized, persistent megakernels, generalizing optimizations that previously required manual, specialized engineering.

Building on the Event Tensor abstraction, we develop a systematic compiler pipeline that automatically fuses and schedules operators for inter-kernel parallelism. Starting from a computational graph annotated with explicit operators and Event Tensors, the compiler applies a series of scheduling transformations to lower the program into an executable megakernel. These transformations support multiple scheduling strategies—ranging from fully static to dynamically load-balanced execution—each representing a different trade-off between synchronization overhead and runtime adaptability. By unifying previously manual fusion and hand-crafted scheduling techniques [Spe25, Che25ae] into compiler transformations, ETC significantly reduces engineering effort to construct megakernels, while improving their runtime performance.

We evaluate ETC on a diverse set of LLM serving workloads, comparing against highly competitive, industry-level baselines (e.g., vLLM and SGLang) that already employ aggressive optimizations such as CUDA Graphs, Programmatic Dependent Launch (PDL), and `torch.compile`. Our results show that ETC achieves substantial speedups over these systems while efficiently supporting both shape- and data-dependent dynamism, without requiring runtime graph recapture or recompilation. For tensor-parallel workloads, our compiler-driven overlap of computation and communication achieves up to 1.40x speedup on fused GEMM and Reduce-Scatter kernels. For data-dependent workloads such as MoE, our megakernels outperform specialized libraries by up to 1.23x. In dynamic-shape, low-batch inference scenarios, ETC matches or exceeds the performance of these highly optimized inference systems, while reducing engine warm-up overheads by up to 3.5x. Achieving even moderate speedups over such strong baselines translates to substantial economic value at datacenter scale. Beyond raw speed, ETC achieves true ahead-of-time (AOT) compilation for dynamic workloads, completely eliminating runtime compilation overhead and the management complexity of repeated CUDA Graph recapture—a major pain point in production serving systems. Furthermore, ETC automates megakernel fusion of complex, data-dependent subgraphs (e.g., MoE layers, GEMM+communication), significantly reducing programming complexity while remaining composable with existing serving engines. ETC has been incorporated into a major open-source system. The simplicity and generality of the Event Tensor abstraction, together with the accompanying compiler framework, can benefit the broader machine learning systems and compiler community.

<span id="section-2"></span>

## 2 Event Tensor Abstraction

<span id="section-2-1"></span>

### 2.1 Language Constructs

<span id="figure-03"></span>

![Figure 3. Example Event Tensor–based program.](../../papers/event-tensor/figure-03.png)

**Figure 3.** Example Event Tensor–based program.

We first introduce the main language constructs in Event Tensor–based programs.

**Device Function.** A device function defines a grid of tasks launched in parallel on the GPU. Each launch is parameterized by a multidimensional coordinate, where each coordinate identifies a task tile executed on a streaming multiprocessor (SM). Each task can include specialized logic, such as warp specialization or tensor core calls.

**Event Tensor.** An Event Tensor is a multi-dimensional structure whose elements represent events—the completion of task sets at the SM level—following established practices in parallel programming systems [Blu95, Tre14, Bau12, Dag98]. Each element has an initial wait count recording the number of tasks it depends on and supports several operations: `E[i].notify()` signals task completion, `E[i].wait()` blocks until the event is triggered, and in dynamic scheduling, events can also trigger dependent tasks. Compared with approaches that manage standalone events individually, Event Tensors greatly reduce task-graph management overhead when scaling to millions of fine-grained events in real LLM inference workloads.

**Graph function.** A graph function represents a computational graph consisting of `call_device` calls that explicitly launch device functions with specified task shapes. Unlike traditional computational graphs, it includes both data tensors and Event Tensors. Each device function launch can annotate explicit input/output dependencies and coordinate mappings that track fine-grained task relationships through Event Tensors.

[Figure 3](#figure-03) shows an example Event Tensor–based program. The general program can be viewed as a compact representation of task graphs in the form of “producer task $\rightarrow$ event $\rightarrow$ consumer task’’. We use generic lambda functions to represent event task relations. [+1] These dependency annotation implicitly maps to event notifications at the end of each producer task and waiting at the beginning of each consumer task. We find this notation to be sufficient for most use cases. Importantly, we also allow device function to explicitly take in Event Tensors as arguments and call the event notify and wait inside each task. The first-class support of the Event Tensor in device function enables us to represent advanced use cases. It also enables explicit fusion optimizations as transformations within the representation that we will discuss in detail in the next section.

<span id="figure-04"></span>

![Figure 4. **Event Tensor handles shape dynamism** with symbolic-shape tensors that define a template for dependency graphs. At runtime, the template is instantiated with concrete shape values (e.g., producing a $1\times 2$ graph for batch size 1 or a $2\times 2$ graph for batch size 2) without recompilation or repeated graph capture.](../../papers/event-tensor/figure-04.png)

**Figure 4.** **Event Tensor handles shape dynamism** with symbolic-shape tensors that define a template for dependency graphs. At runtime, the template is instantiated with concrete shape values (e.g., producing a $1\times 2$ graph for batch size 1 or a $2\times 2$ graph for batch size 2) without recompilation or repeated graph capture.

<span id="section-2-2"></span>

### 2.2 Representing Fine-Grained Dependencies

To illustrate how Event Tensors are used in practice, we walk through the example shown in [Figure 3](#figure-03). It shows a task graph that performs a summation over the inner axis of the input tensor $A$, which has symbolic shape $(n\times 32,128)$: $C[i]=\sum_{k\in[0,128)}A[i,k]$. The example adopts a split-K algorithm that divides the summation into two stages $B[i,j]=\sum_{k\in[j*32,j*32+32)}A[i,k]$ and $C[i]=\sum_{k\in[0,4)}B[i,k]$, where the first stage computes partial sums of each row into $B$, and the second stage aggregates them to produce $C$. In a traditional kernel-by-kernel approach, tasks in the second stage are launched only after all tasks in the first stage have completed. However, each output row $C[i]$ depends only on the corresponding row $B[i,:]$, meaning that its computation can proceed concurrently with partial sums of other rows. To capture this fine-grained dependency, we partition the computations of $B$ and $C$ into finer tasks and introduce an Event Tensor $E$:

$$
\begin{aligned}
\text{Task }\hat{B}_{i,j}: && B[i*32:i*32+32,j], \\
\text{Event }E_{i}: && E[i], \\
\text{Task }\hat{C}_{i}: && C[i*32:i*32+32].
\end{aligned}
$$

With task partitioning, the dependency relations becomes $\hat{B}_{i,j}\rightarrow E_{i}$ ($\hat{B}_{i,j}$ produces $E_{i}$) and $E_{i}\rightarrow\hat{C}_{i}$ ($\hat{C}_{i}$ consumes $E_{i}$), where each $E_{i}$ corresponds to the completion of 32 consecutive rows of $B$ starting from row $32*i$.

The `main_graph` function provides the description of the overall computation where the primitive `call_device` first launches `partial_sum` and then `final_sum` function. The dependencies between `partial_sum` and `final_sum` tasks are specified through `out_edges` and `in_edges` arguments.

<span id="section-2-3"></span>

### 2.3 First-Class Dynamic Shape Support

The Event Tensor-based graph can be viewed as a more compact representation of the task graph representations [Tre14] in parallel systems, where each event element and tasks are explicitly represented as individual nodes and edges materialized as task graphs in the runtime. In our case, we can use a single tensor to represent thousands of events. The Event Tensor representation also allows us to bring in first-class support for symbolic dimensions. For example, we can have an Event Tensor shape to contain symbolic variables such as batch size of sequence length. The symbolic-shape Event Tensor graph serves as a generic template that corresponds to different task graphs ([Figure 4](#figure-04)) at runtime. This powerful representation gives us the ability to overcome limitations in static task-graph systems such as CUDA Graph, and *ahead of time* optimize dynamic shape Event Tensor graph for multiple shapes without recompilation or repeated graph capture, providing greater flexibility in handling dynamic shape workloads.

<span id="section-2-4"></span>

### 2.4 Supporting Data-Dependent Dynamism

<span id="figure-05"></span>

![Figure 5. **Event Tensor handles data-dependent dynamism** by (a) A regular workload with static dependencies. (b) A data-dependent MoE workload where runtime tensors `topk` and `exp_indptr` define an irregular task graph, enabling data-dependent event updates and task triggering.](../../papers/event-tensor/figure-05.png)

**Figure 5.** **Event Tensor handles data-dependent dynamism** by (a) A regular workload with static dependencies. (b) A data-dependent MoE workload where runtime tensors `topk` and `exp_indptr` define an irregular task graph, enabling data-dependent event updates and task triggering.

A critical challenge in modern workloads is handling irregular, data-dependent task graphs. A representative example is the Mixture-of-Experts (MoE) layer [Sha17], where input tokens are dynamically routed to different expert sub-networks based on routing decisions computed at runtime. An efficient MoE implementation typically first groups tokens according to their assigned experts and then uses GroupGEMM operators to compute the results for all tokens within each group.

Dynamic routing introduces fine-grained, data-dependent task dependencies that are unknown at compile time (specifically, which expert’s GroupGEMM tile processes which tokens). This dynamism poses a significant challenge for traditional compilers and schedulers, which assume a static task graph with fixed dependencies. To efficiently represent such dynamic workloads, we need an abstraction that can (1) determine at runtime which tasks each consumer task depends on, and (2) trigger a variable number of consumer tasks based on runtime data. The Event Tensor abstraction is designed precisely for this purpose. It manages data-dependent task graphs through two core mechanisms:

**Data-Dependent Event Update.** Unlike conventional task graphs that support only static dependencies ([Figure 5](#figure-05)a), the Event Tensor abstraction allows dynamic event dependencies ([Figure 5](#figure-05)b). In the MoE example, runtime routing decisions stored in the `topk` tensor determine which grouping tiles (one per token) update which events (one per expert). Each expert’s event counter is initialized to the number of tokens routed to it, and this initialization occurs dynamically at runtime, together with the computation of `topk`.

**Data-Dependent Task Triggering.** Similarly, an event can trigger a runtime-dependent number of tasks. Based on routing decisions in `topk`, we can compute how many tokens each expert must process and, therefore, how many GroupGEMM tiles each expert requires. As illustrated in [Figure 5](#figure-05)b, this information is encoded in the tensor `exp_indptr`, which stores the prefix sum of GroupGEMM tiles to be triggered per expert. [+2] Leveraging `exp_indptr`, we enable data-dependent triggering, where expert `i` activates tiles in the range `(exp_indptr[i], exp_indptr[i+1])`.

Together, these two mechanisms allow the compiler to generate megakernels that efficiently adapt to highly dynamic workloads—cases that static task graphs handle poorly. Combined with the symbolic-shape support described above, all compilation in ETC occurs offline; at inference time, the compiled binary handles both shape and data-dependent dynamism with zero compilation overhead.

Note that the overall dependency chain remains strictly feed-forward ([Figure 2](#figure-02), right): Attention Output $\rightarrow$ Token Routing (TopK) $\rightarrow$ Token Grouping $\rightarrow$ Token Computation (GroupGEMM). Computing TopK depends only on the preceding Attention output—a standard static dependency. The data-dependent Event Tensor mechanisms described above govern only the later stages, where routing results dynamically determine which grouping tasks notify which expert events and how many GroupGEMM tiles each expert triggers.

<span id="section-3"></span>

## 3 Event Tensor Compilation

This section describes optimizations in ETC that make use of the proposed Event Tensor abstractions.

<span id="section-3-1"></span>

### 3.1 Static Scheduling and Transformation

Static scheduling fuses multiple device functions together by explicitly distributing tasks across streaming multiprocessors (SMs) ahead of time. As a result, each task is pre-assigned to the task queue of a specific SM. Task dependencies are managed through low-level synchronization primitives, such as counter-based semaphores and event-triggered waits. This approach achieves minimal synchronization overhead and is particularly effective for predictable workloads where SM partitions can be optimized ahead of time.

<span id="figure-06"></span>

![Figure 6. GEMM + Reduce-Scatter before and after static scheduling transformation. Two separate device functions are fused into a single persistent function, with explicit `notify` and `wait` calls on the Event Tensor to coordinate dependencies.](../../papers/event-tensor/figure-06.png)

**Figure 6.** GEMM + Reduce-Scatter before and after static scheduling transformation. Two separate device functions are fused into a single persistent function, with explicit `notify` and `wait` calls on the Event Tensor to coordinate dependencies.

<span id="figure-07"></span>

![Figure 7. Notify-and-wait mechanism for static scheduling.](../../papers/event-tensor/figure-07.png)

**Figure 7.** Notify-and-wait mechanism for static scheduling.

We implement a static scheduling transformation in ETC with three main steps ([Algorithm 1](#algorithm-01)): (1) construct per-SM execution queues on the host; (2) generate a persistent main loop that lets each SM execute tasks continuously without relaunching; and (3) lower Event Tensor dependencies into explicit `notify()` and `wait()` calls to enforce fine-grained execution order. [Figure 6](#figure-06) illustrates this process for GEMM + Reduce-Scatter, fundamental to tensor-parallel execution. It fuses the device functions into a single persistent kernel, whose main loop continuously fetches tasks from a precomputed queue in `tile_scheduler` and issues `notify()` at the end of GEMM tasks and `wait()` at the beginning of Reduce-Scatter tasks.

[Figure 7](#figure-07) illustrates how the statically fused GEMM (MM) + Reduce-Scatter (RS) kernel operates in practice. Each RS task depends on two MM tasks (i.e., an RS tile spans twice the size of an MM tile), so the initial counter for each event is two. At $T_{1}$, MM0 on SM0 finishes and notifies the Event Tensor, reducing the counter to one. The RS task, statically scheduled next on SM0, cannot yet proceed and enters a spin-wait state. Between $T_{1}$ and $T_{2}$, SM1 continues executing MM0, keeping the GPU busy. At $T_{2}$, MM0 on SM1 completes, decrementing the counter to zero and satisfying the dependency, which releases the RS task from its wait loop and allows execution to begin on SM0.

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**Algorithm 1: Static Scheduling Transformation in ETC.**

- **Input:** A module `mod` containing a tile-level dataflow graph `G` with Event Tensor dependencies.
- **Output:** An updated module with a fused, statically scheduled megakernel.
- `mod_updated` $\leftarrow$ `mod.Copy()`.
- `static_schedule` $\leftarrow$ `GenerateStaticSchedule(G)`.
- `fused_kernel` $\leftarrow$ `NewPersistentKernel()`.
- Embed the pre-computed schedule into global memory.
- `fused_kernel.AddBuffer(static_schedule)`.
- **For all** `task_grid` in `G`:
  - `fused_kernel.AddDispatchLogic(task_grid)`.
  - **For all** `event` in `task_grid.in_edges`:
    - `fused_kernel.AddWaitLogic(event)`.
  - `fused_kernel.AddTileLogic(task_grid)`.
  - **For all** `event` in `task_grid.out_edges`:
    - `fused_kernel.AddNotifyLogic(event)`.
- `mod_updated.Replace(G, fused_kernel)`.
- **Return** `mod_updated`.

</div>

Plain static scheduling does not naturally support dynamic workloads. To handle shape dynamism, we sample a set of representative shapes; unseen shapes reuse the execution queue of the next larger sampled value. To handle data-dependent dynamism, we conservatively assume the worst case for Event Tensor updates and triggers by rewriting related `notify()` and `wait()` operations to `E[0].notify()` and `E[0].wait()`. For simplicity, we use a round-robin policy to construct execution queues.

<span id="section-3-2"></span>

### 3.2 Dynamic Scheduling and Transformation

<span id="figure-08"></span>

![Figure 8. GEMM + Reduce-Scatter after dynamic scheduling transformation. Task `push` and `pop` are inserted, and task execution is dynamically coordinated by the scheduler.](../../papers/event-tensor/figure-08.png)

**Figure 8.** GEMM + Reduce-Scatter after dynamic scheduling transformation. Task `push` and `pop` are inserted, and task execution is dynamically coordinated by the scheduler.

When task execution time is unpredictable, dynamic scheduling improves load balance across SMs. ETC implements Event Tensor–based dynamic scheduling using a lightweight on-GPU task scheduler. When an event is triggered—after all its dependent tasks complete—it atomically **pushes** all associated consumer tasks into the scheduler, marking them ready for execution. Any available SM can then atomically **pop** a ready task and execute it.

<span id="figure-09"></span>

![Figure 9. Push-and-pop mechanism for dynamic scheduling.](../../papers/event-tensor/figure-09.png)

**Figure 9.** Push-and-pop mechanism for dynamic scheduling.

We implement a dynamic scheduling transformation in our compiler [+3]. [Figure 8](#figure-08) shows the transformed code for the same GEMM (MM) + Reduce-Scatter (RS) example discussed in [Section 3.1](#section-3-1). [Figure 9](#figure-09) illustrates the push–pop mechanism. At $T_{1}$, MM0 on SM0 finishes and decrements the event counter to one; SM0, now idle, immediately pops a ready task (MM1) from the scheduler. At $T_{2}$, MM0 on SM1 completes, reducing the counter to zero and triggering the RS task to be pushed into the scheduler. SM1 then pops the RS task and begins execution. This entire process of dependency tracking and task dispatching occurs efficiently on the GPU, without requiring any host-precomputed task queue.

Dynamic scheduling inherently supports both kind of dynamism, because tile execution order is decided on the fly at runtime, when the symbolic shape value and runtime value are already resolved by scheduler. Our implementation of push-pop interface uses a centralized queue in global memory shared across all SMs. We choose this design for its implementation simplicity, though we acknowledge potential contention at scale. We also discuss the runtime optimization for dynamic scheduler in [Section 11](#section-11).

**Trade-off between static and dynamic scheduling.** The choice between static and dynamic scheduling reflects a classic trade-off. Static scheduling leads to minimal scheduling overhead, making it well-suited for predictable workloads. Dynamic scheduling, by contrast, offers flexibility for data-dependent dynamic workload or unpredictable task completion times, naturally achieving load balance at the cost of a small runtime overhead of task queue pushes and pops.

<span id="section-3-3"></span>

### 3.3 Lowering to Minimal Runtime

The static and dynamic scheduling in our compiler allows us to encapsulate low-level task dependencies and their handling directly in the transformed program, resulting in minimal needs for corresponding supporting runtime ([Figure 10](#figure-10)). Specifically, each Event Tensor is lowered to an integer tensor, reusing the existing tensor data structure and avoiding any dedicated runtime data structures for events. The `notify()` and `wait()` operations on this integer tensor are implemented with efficient hardware atomics: `notify()` performs an atomic decrement, while `wait()` spin-waits for the counter to reach zero. Our runtime data state consists solely of these integer tensors and scheduler’s task queue. This compiled-in approach brings a smaller runtime requirement compared to a typical task-graph approach, where the entire task graph needs to be materialized in memory and relies on a generic task executor that traverses this graph to launch device functions.

<span id="figure-10"></span>

![Figure 10. Comparison of runtime architectures. (a) The task graph in traditional runtime executor is materialized in memory, and only tiled operators are compiled. (b) ETC compiles scheduling logic into megakernels without runtime task graph materialization.](../../papers/event-tensor/figure-10.png)

**Figure 10.** Comparison of runtime architectures. (a) The task graph in traditional runtime executor is materialized in memory, and only tiled operators are compiled. (b) ETC compiles scheduling logic into megakernels without runtime task graph materialization.

<span id="section-3-4"></span>

### 3.4 End-to-End Compilation Flow

The end-to-end compilation flow of ETC [+4] starts from an unoptimized computational graph where Event Tensors are defined and operators are already partitioned into CTA-level tiles—either user-specified through a kernel DSL such as Triton [Til19] or provided as compiler builtins. In our implementation, device functions are written in a TVM-based DSL [Hou26] that supports standard tile-based programming. Notably, the Event Tensor abstraction is DSL-agnostic: its dependency graph and scheduling logic can be integrated into other compiler stacks (e.g., Triton, CuteDSL) without fundamental design conflicts.op The graph first undergoes standard graph-level optimizations, including memory planning, similar to existing deep learning compilers [Sab20, Lat21, Lai25a, Ans24]. Next, tile-level optimizations refine each operator by determining low-level details such as hardware instruction mapping and pipelining strategies. The graph is then transformed via either the static or dynamic scheduling pass described in [Section 3.1](#section-3-1) and [Section 3.2](#section-3-2). The resulting fused device function is emitted as GPU code in the persistent-kernel style. A subsequent prefetching pass generates weight-prefetching functions for tiled operators based on user annotations, enabling each tile to prefetch weights before input activations arrive. Finally, if static scheduling is chosen, the compiler computes the per-SM task order and materializes it as the megakernel’s static execution queue.

<span id="section-4"></span>

## 4 Evaluation

We implement ETC as a series of compiler passes building upon Apache TVM. Notably, our proposed abstraction can be applied to other compilers as well. This section provides evaluations to answer the following key questions:

- How effectively does the Event Tensor abstraction manage fine-grained dependencies for workloads with both static task graphs ([Section 4.1](#section-4-1)) and task graphs with shape dynamism and data-dependent dynamism ([Section 4.2](#section-4-2))?
- Do ETC-compiled megakernels achieve lower end-to-end latency in dynamic low-batch serving scenarios? ([Section 4.3](#section-4-3))
- Does the Event Tensor’s support for shape dynamism eliminate the significant engine warmup overhead required by static-shape runtime just-in-time (JIT) compilation and graph capture systems? ([Section 4.4](#section-4-4))
- Do static and dynamic scheduling exhibit distinct performance trade-offs on different workloads? ([Section 4.5](#section-4-5))

All experiments are conducted on a server equipped with 8 NVIDIA B200 GPUs connected via NVLink, running Ubuntu 24.04 with PyTorch 2.8.0, CUDA 13.0 and driver version 580.82.07. We choose B200 as it represents state-of-the-art hardware; the Event Tensor abstraction operates at the compiler IR level and is not specific to any particular GPU generation. We evaluate ETC against state-of-the-art deep learning compilers, specialized libraries, and high-performance LLM serving systems. Existing megakernel frameworks are tailored to single-batch inference and thus cannot be fairly compared under dynamic-shape or data-dependent workloads.

<span id="section-4-1"></span>

### 4.1 Fused Communication and Computation Performance

To evaluate how effectively the Event Tensor abstraction optimizes static compute-communication patterns, we benchmark two fundamental fused kernels for tensor-parallel LLMs: GEMM + Reduce-Scatter and All-Gather + GEMM. These kernels are critical for minimizing latency and maximizing hardware utilization in distributed inference. We use MLP configurations derived from a range of modern LLMs, fixing the tensor-parallel size to 8 and the number of tokens to 8192 in all experiments. The configuration details are provided in [Section 9](#section-9). We compare ETC’s generated kernels against several baselines, with implementation choices tailored to each workload’s characteristics:

- GEMM + Reduce-Scatter: The Reduce-Scatter collective is implemented using CUDA multimem PTX instructions. We employ ETC’s **dynamic scheduler** to handle unpredictable workloads arising from network contention and fluctuation, where its ability to adapt and balance tasks on the fly is most effective.
- All-Gather + GEMM: The All-Gather operation uses a ring algorithm implemented via the copy engine (DMA). We use the **static scheduler**, as only GEMM tiles execute on SMs following the data arrival order dictated by the ring algorithm. A precomputed static schedule effectively overlaps communication and computation with minimal runtime overhead.

The baselines for comparison include:

- cuBLAS+NCCL: A non-overlapped baseline executing cuBLAS and NCCL kernels sequentially, representing performance without fusion.
- TP-Async [Lia24b]: A PyTorch-based approach that manually orchestrates asynchronous operations for overlap.
- Triton Distributed v0.0.2-rc [Zhe25i]: A compiler-based system that generates overlapping kernels, serving as a state-of-the-art open-source baseline.
- cuBLASMp [Cub23]: A high-performance, multi-process fused-kernel library that overlaps distributed computation and communication.

<span id="figure-11"></span>

![Figure 11. Performance results of GEMM + Reduce-Scatter on 8 B200s with dynamic scheduler.](../../papers/event-tensor/figure-11.png)

**Figure 11.** Performance results of GEMM + Reduce-Scatter on 8 B200s with dynamic scheduler.

<span id="figure-12"></span>

![Figure 12. Performance results of All-Gather + GEMM on 8 B200s with static scheduler.](../../papers/event-tensor/figure-12.png)

**Figure 12.** Performance results of All-Gather + GEMM on 8 B200s with static scheduler.

[Figure 11](#figure-11) and [Figure 12](#figure-12) show clear improvements over the baselines, particularly on larger model configurations, achieving up to a 1.40x execution time speedup over the cuBLAS+NCCL baseline for both workloads. Among the fused baselines, TP-Async’s coarse-grained splitting can lead to chunks that are either too small to saturate SMs or too large to effectively hide communication latency, and Triton-Dist’s experimental B200 support means its Triton-based GEMM is not yet fully optimized for the Blackwell architecture. As a result, the unfused cuBLAS+NCCL baseline is sometimes competitive with these fused approaches, underscoring the difficulty of achieving efficient fusion. The consistent performance advantage of ETC stems from the Event Tensor abstraction. By representing fine-grained dependencies as a first-class Event Tensor, our compiler can transform monolithic operations into a deeply pipelined task graph. This abstraction also allows our unified scheduling transformations to be applied effectively: we use the dynamic scheduler for GEMM + Reduce-Scatter to handle any potential unpredictability of communication latency, and the static scheduler for All-Gather + GEMM to orchestrate overlap with the predictable ring algorithm with minimal overhead. This fine-grained, compiler-driven approach keeps both compute (SMs) and network resources continuously busy, achieving a degree of overlap that matches or exceeds existing systems.

<span id="section-4-2"></span>

### 4.2 Mixture-of-Experts (MoE) Layer Performance

To evaluate our Event Tensor abstraction’s ability to manage task graphs with shape dynamism and data-dependent dynamism, this subsection benchmarks a complete MoE layer with variable number of tokens. While existing systems use efficient persistent GroupGEMM kernels, they still require a sequence of separate launches for a complete MoE layer. ETC allows us to fuse the entire data-dependent MoE dataflow into a single megakernel, and we evaluate its performance against existing optimized multi-kernel baselines. We benchmark a complete MoE layer in Qwen3-30B-A3B, which has 128 experts with a top-k of 8. The workload consists of processing a variable number of input tokens. For this workload, we use ETC’s dynamic scheduler, as its adaptive load balancing is ideal for the irregular, data-dependent task graph. We compare against several baselines:

- Triton 3.4.0 [Til19]: A highly optimized MoE implementation widely used in state-of-the-art serving systems, including SGLang [She24] and vLLM [Kwo23].
- FlashInfer 0.2.14.post1 [Ye25]: A high-performance library providing optimized kernels for LLM inference, including fused MoE kernels.

<span id="figure-13"></span>

![Figure 13. Performance results of MoE layer on a single B200.](../../papers/event-tensor/figure-13.png)

**Figure 13.** Performance results of MoE layer on a single B200.

[Figure 13](#figure-13) plots the relative end-to-end performance for the MoE layer across different token counts. ETC’s megakernel approach significantly outperform the best baseline, achieving up to a 1.23x speedup at 1024 tokens. Between the two baselines, FlashInfer’s GroupGEMM is more optimized for larger token counts, while Triton benefits from fusing gather/scatter into GroupGEMM; their relative ranking thus varies with token count. The performance gains of ETC are a direct result of the Event Tensor abstraction and dynamic scheduling transformation. Firstly, data-dependent Event Tensors break the global synchronization barrier in the baselines, creating a fine-grained pipeline between the two GroupGEMM stages in MoE, which also reduces wave quantization by smoothing out SM allocation across fused operators. Secondly, and more importantly for MoE, our on-chip dynamic scheduler provides superior load balancing for the irregular token routing, minimizing SM idle time and consistently surpassing all other methods as token counts grow.

<span id="section-4-3"></span>

### 4.3 End-to-End Low-Batch Serving Performance

<span id="figure-14"></span>

![Figure 14. End-to-end performance of model serving on Qwen3-30B-A3B and Qwen3-32B (Lower is better).](../../papers/event-tensor/figure-14.png)

**Figure 14.** End-to-end performance of model serving on Qwen3-30B-A3B and Qwen3-32B (Lower is better).

This subsection tests whether ETC-compiled megakernels achieve lower end-to-end latency in dynamic low-batch serving scenarios, which are increasingly dominant for latency-sensitive applications such as real-time agentic workflows and interactive coding assistants. We focus on the decoding stage, as prefilling and large-batch serving usually have high GPU utilization and benefit marginally from megakernels (except for computation-communication overlap as we showcased in [Section 4.1](#section-4-1)). Importantly, ETC does not degrade large-batch performance, as hardware utilization of individual operators remains unchanged without extra overhead. We integrate ETC-compiled megakernels for two representative models: the Qwen3-30B-A3B MoE model and the Qwen3-32B dense model, using the static scheduler for both to avoid runtime overhead. The compiled megakernels cover the full decoding pipeline (Attention, RoPE, KV-Cache, Norm, MLP, MoE), not just GEMM. We chose the Qwen family (both dense and MoE variants) as it is architecturally representative of modern LLMs (e.g., LLaMA 3, GPT). The benchmark uses a synthetic dataset with a prefill length of 512 and generates 100 output tokens, with batch size varying from 1 to 128. We measure the time-per-output-token (TPOT) metric, which best reflects the LLM engine decoding performance. The raw kernel running time of one decoding iteration can be found in [Section 10](#section-10), which concentrates only on the kernel performance and excludes all irrelative overhead (e.g., framework-specific scheduling latencies and other CPU overhead), ensuring a fair comparison of different frameworks. We compare ETC against leading serving systems vLLM (v0.11.0rc2) and SGLang (v0.5.3rc0) both of which use CUDA Graph and torch.compile [Ans24] for performance optimization.

[Figure 14](#figure-14) (left) shows the end-to-end performance for Qwen3-30B-A3B, where ETC achieves a 1.48x speedup over vLLM and 1.20x over SGLang at batch size 1. In [Figure 14](#figure-14) (mid), for Qwen3-32B, ETC consistently delivers the lowest latency, outperforming vLLM by up to 1.15x at batch size 1, and SGLang by up to 1.09x at batch size 64. In four-way TP tensor-parallel (TP) execution [Sho19] for the Qwen3-32B ([Figure 14](#figure-14), right), ETC matches performance of vLLM with a speedup varying between 0.99x and 1.06x. The latency of both ETC and vLLM is higher than SGLang in this setting because SGLang’s highly optimized CPU scheduler incurs lower distributed runtime overhead. The occasional small gaps where ETC trails the best baseline are attributable to engineering factors—specifically, compiler-generated GEMM tiles that are less tuned than cuBLAS in certain configurations and higher CPU-side overhead in our serving engine—rather than fundamental limitations of the abstraction.

The strong performance of ETC stems from its fused megakernel architecture, enabled by the Event Tensor abstraction and static scheduling transformation. Unlike conventional approaches that launch a sequence of kernels with implicit synchronization at kernel boundaries, ETC executes the entire workload within a single persistent kernel. This compiler-driven fusion enables fine-grained optimizations that are difficult for CUDA-Graph-based systems: it exposes parallel execution in attention (e.g., Q’s Norm+RoPE running concurrently with K’s Norm+RoPE+CacheAppend), pipelines GroupGEMMs in MoE and GEMMs in MLP to reduce wave quantization, and prefetches model weights before input activations are ready to hide memory latency. ETC’s ability to pipeline and overlap operations across operator boundaries is the key for the latency reduction compared to baselines. Crucially, ETC breaks kernel boundaries, successfully applying performance on par with CUDA Graphs to inherently dynamic workloads like MoE.

<span id="section-4-4"></span>

### 4.4 Warmup Overhead

<span id="table-01"></span>

![Table 1. Warmup time of Qwen3-32B model serving using different graph capturing methods.](../../papers/event-tensor/table-01.png)

**Table 1.** Warmup time of Qwen3-32B model serving using different graph capturing methods.

This subsection evaluates the deployment impact of ETC’s compilation strategy by measuring LLM engine warmup overhead. We define warmup time as the total wall-clock time from engine launch to the first request served, including engine initialization, model loading, and all JIT compilation or CUDA Graph capture overheads. We test whether ETC’s ahead-of-time (AOT) compilation, enabled by shape dynamism support, can eliminate the runtime cost of just-in-time (JIT) compilation and CUDA Graph capture. [Table 1](#table-01) shows substantial difference: vLLM requires 123 s and SGLang 583 s to warm up on Qwen3-32B, whereas ETC initializes in only 35 s. This speedup stems from the Event Tensor abstraction, whose first-class shape dynamism support enables AOT compilation. While baselines must capture many static CUDA Graphs at runtime to cover different shapes (e.g., 67 for vLLM), ETC compiles a single persistent, shape-generic megakernel offline (107 s for Qwen3-32B). At runtime, it simply loads the precompiled graph, avoiding the repetitive warmup penalty inherent to JIT and runtime-capture approaches.

<span id="section-4-5"></span>

### 4.5 Tradeoff Between Different Scheduling Methods

<span id="table-02"></span>

![Table 2. Relative performance of different ETC scheduling methods on MoE layer against unfused megakernel. Higher is better.](../../papers/event-tensor/table-02.png)

**Table 2.** Relative performance of different ETC scheduling methods on MoE layer against unfused megakernel. Higher is better.

<span id="table-03"></span>

![Table 3. Relative performance of different ETC scheduling methods on Qwen-3-32B with TP=4 against unfused megakernel. Higher is better.](../../papers/event-tensor/table-03.png)

**Table 3.** Relative performance of different ETC scheduling methods on Qwen-3-32B with TP=4 against unfused megakernel. Higher is better.

This section analyzes the performance characteristics and trade-offs of ETC’s static and dynamic scheduling strategies, and quantifies the gains from fusion by comparing them to an unfused megakernel baseline. This baseline uses a single event between different operator stages to enforce a global synchronization barrier, simulating a sequential execution model within a single kernel launch. Crucially, the unfused baseline uses identical operator code as ETC, so the speedups reported in [Table 2](#table-02) and [Table 3](#table-03) are driven purely by the inter-kernel parallelism unlocked by ETC’s fine-grained Event Tensor dependencies—the same sources of gain discussed in [Section 4.2](#section-4-2) and [Section 4.3](#section-4-3) (reduced wave quantization, weight prefetching, parallel execution, and on-chip load balancing)—rather than better operator-level implementations.

**Data-Dependent Workloads.** For workloads with data-dependent control flow, such as the MoE layer in [Section 4.2](#section-4-2), the dynamic scheduler provides load balancing. As shown in [Table 2](#table-02), dynamic scheduler outperforms static scheduler except on single-batch inference, with the largest speedup being 4.0% over static scheduler and 8.1% over unfused baseline when batch size is 1024. The data-dependent routing of tokens creates an inherent workload imbalance. A rigid static scheduler can cause some SMs to accumulate a queue of longer-running tiles, forcing them to become stragglers while other SMs sit idle.

**Regular Workloads.** Conversely, for the regular, dense transformer layer workload (analyzed with TP=4 in [Section 4.3](#section-4-3)), the static scheduler is the clear winner ([Table 3](#table-03)). The dynamic scheduler’s overhead becomes very large on distributed setting, especially when trying to push tasks to remote task queue. There is also a consistent 6-8% speedup of ETC-static over the ETC-unfused version, a gain purely from fine-grained pipelining.

These results also explain why the multi-GPU evaluation results in [Figure 11](#figure-11) and [Figure 12](#figure-12) and [Figure 14](#figure-14) appear to show different trends. [Figure 11](#figure-11) and [Figure 12](#figure-12) represent a bandwidth-bound regime with large batches (8192 tokens), where ETC excels by utilizing fine-grained signaling to overlap communication and computation. In contrast, [Figure 14](#figure-14) (right) represents a latency-critical regime with low batches, where communication overhead and CPU scheduling overhead are exposed. This necessitates different scheduling strategies: dynamic scheduling handles large-batch jitter effectively, while static scheduling minimizes overhead for latency-sensitive low-batch tasks.

These findings demonstrate a clear, workload-dependent trade-off, confirming the value of supporting both scheduling transformations within the ETC framework.

<span id="section-5"></span>

## 5 Related Work

Deep learning compilers such as MLIR [Lat21], XLA [Sab20], TVM [Che18e, Fen22, Lai25a], and the PyTorch compiler [Ans24] have laid the foundation for optimizing deep learning models. These systems perform graph-level optimizations and run execution kernel-by-kernel. CUDA Graph [Gra19] provides a way to drastically reduce launch overhead by capturing and replaying a sequence of kernels, but relies on static input. Machine learning compilers are also developing vertical fusion [Zhe20, Niu21] and horizontal fusion [Jia19b, Li22e] to optimize kernel launch overhead. Rammer [Ma20] and Roller [Zhu22] perform software launch of tile-based tasks. All the previous works do not have explicit abstraction for fine-grained dependencies tracking and optimizations, and can benefit from our proposed Event Tensor to enable megakernel optimizations. Dynamic tensor compilers such as DynaTune [Zha21m], DietCode [Zhe22c], and SparseTIR [Ye23a] handle dynamic shapes or sparsity at the single-kernel level; ETC complements them by fusing their operator implementations into megakernels to unlock inter-kernel parallelism.

LLM inference systems such as SGLang [She24], vLLM [Kwo23], TensorRT-LLM [Ten24a], and Orca [Yu22a] achieve high performance through system optimizations such as continuous batching and speculative execution. Our event tensor compiler can serve as a backend for these frameworks to enable more efficient GPU execution. Recent works [Che25ae, Spe25] start to build megakernels for LLMs. These approaches only supports single-batch dense model inference, and focus on a single scheduling strategy. The event tensor abstraction proposed in this paper complement these approaches by providing systematic compiler abstraction support for shape dynamism and data-dependent dynamism. The event tensor compiler also supports both static scheduling and dynamic scheduling. CuSync [Jan24a] optimizes co-scheduling of separate kernels on distinct CUDA streams, and FlashMoE [Aim25] provides a hand-optimized kernel for distributed MoE. ETC differs from both by fusing entire subgraphs into a single persistent megakernel via a systematic compiler pipeline, generalizing beyond any single operator pattern.

Our approach is closely related to task-based parallel programming models such as Cilk [Blu95], Legion [Bau12], Realm [Tre14], and OpenMP Tasks [Dag98]. Most of the previous approaches focus on coarse-grained tasks typically orchestrated by the CPU. Our approach is also related to Graphene [Hag23] and Cypress [Yad25] that optimizes a single kernel. Graphene models threads as a tensor with synchronization capabilities, which is conceptually related; however, it targets single-kernel optimization rather than multi-operator megakernel fusion with dynamic shape and data-dependent support. Our approach builds on these previous insights and proposes the event tensor abstraction that compactly represents fine-grained dependencies across operator sub-tasks and runs both static and dynamic scheduling in the GPU streaming multiprocessors.

<span id="section-6"></span>

## 6 Conclusion

This work introduces Event Tensor, a unified abstraction that expresses fine-grained synchronization for compiling dynamic GPU megakernels. Event Tensor provides first-class support for both shape and data-dependent dynamism. Built on this abstraction, ETC systematically generates high-performance persistent kernels using static and dynamic scheduling. ETC achieves state-of-the-art serving latency while substantially reducing warmup overhead. In the future, we envision higher-level passes that automatically generate Event Tensor task graphs from standard computational graphs, further reducing manual synchronization effort. We hope this work will encourage additional studies of megakernels and highlight new possibilities for ML compilers.

## Acknowledgements

We thank all anonymous MLSys reviewers and our shepherd for their constructive feedback and comments. This work is supported in part by gifts from NVIDIA, Google and Amazon. We also acknowledge support from NVIDIA for the DGX B200.

<span id="section-7"></span>

## 7 Dynamic Scheduling Pseudocode

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**Algorithm 2: Dynamic Scheduling Transformation in ETC.**

- **Input:** A module `mod` containing a tile-level dataflow graph `G` with Event Tensor dependencies..
- **Output:** An updated module with a fused, dynamic scheduled megakernel.
- `mod_updated` $\leftarrow$ `mod.Copy()`.
- `fused_kernel` $\leftarrow$ `NewPersistentKernel()`.
- Not runtime scheduler. Only provides push/pop functions.
- `scheduler` $\leftarrow$ `GPUScheduler()`.
- `fused_kernel.AddPopLogic(scheduler.f_pop_tasks)`.
- **For all** `task_grid` in `G`:
  - `fused_kernel.AddDispatchLogic(task_grid)`.
  - `fused_kernel.AddTileLogic(task_grid)`.
  - **For all** `event` in `task_grid.out_edges`:
    - `fused_kernel.AddCompleteOnLogic(event, scheduler.f_push_tasks)`.
- `mod_updated.Replace(G, fused_kernel)`.
- **Return** `mod_updated`.

</div>

[Algorithm 2](#algorithm-02) describes the compiler pass to transform an event tensor graph to a dynamically scheduled megakernel. A call to `scheduler.pop_tasks` is inserted whenever an SM finishes its current task, while a call to `scheduler.push_tasks` is inserted when the completion of a task decrements the associated event counters to zero, thereby unblocking dependent tasks.

<span id="section-8"></span>

## 8 ETC End-to-End Compilation Flow

<span id="figure-15"></span>

![Figure 15. End-to-end compilation pipeline in ETC.](../../papers/event-tensor/figure-15.png)

**Figure 15.** End-to-end compilation pipeline in ETC.

[Figure 15](#figure-15) summarizes the end-to-end compilation flow of ETC, as described in [Section 3.4](#section-3-4).

<span id="section-9"></span>

## 9 MLP configuration used in [Section 4.1](#section-4-1)

[Table 4](#table-04) shows MLP configurations used in fused communication and computation evaluation, which are derived from a range of modern LLMs.

<span id="table-04"></span>

![Table 4. Model configurations for MLP, where S = sequence length, H = hidden dim, I = intermediate size.](../../papers/event-tensor/table-04.png)

**Table 4.** Model configurations for MLP, where S = sequence length, H = hidden dim, I = intermediate size.

<span id="section-10"></span>

## 10 Raw Kernel Time Evaluation in End-to-End LLM Serving

<span id="figure-16"></span>

![Figure 16. Raw kernel relative performance results of Qwen-30B-A3B on a single B200.](../../papers/event-tensor/figure-16.png)

**Figure 16.** Raw kernel relative performance results of Qwen-30B-A3B on a single B200.

<span id="figure-17"></span>

![Figure 17. Raw kernel relative performance results of Qwen-32B on a single B200.](../../papers/event-tensor/figure-17.png)

**Figure 17.** Raw kernel relative performance results of Qwen-32B on a single B200.

<span id="figure-18"></span>

![Figure 18. Raw kernel relative performance results of Qwen-32B on four B200s with tensor parallelism.](../../papers/event-tensor/figure-18.png)

**Figure 18.** Raw kernel relative performance results of Qwen-32B on four B200s with tensor parallelism.

[Section 4.3](#section-4-3) reports the time-per-output-token (TPOT) metric of ETC and baselines in end-to-end LLM serving. The TPOT number includes not only the total GPU kernel execution time but also CPU-side overheads, such as framework-specific request scheduling latency. To provide a more direct and fair comparison unaffected by unrelated overheads, this section evaluates the raw GPU kernel execution time in end-to-end LLM serving, using the same baselines and experimental settings as in [Section 4.3](#section-4-3).

[Figure 16](#figure-16) shows the relative raw kernel end-to-end performance of ETC and baselines across multiple batch sizes on Qwen3-30B-A3B, where ETC achieves consistent speedups over the baselines, with the most significant improvement being 1.49x over vLLM and 1.27x over SGLang at batch size 1. With the Event Tensor abstraction supporting data-dependent dependencies in MoE, ETC executes the entire MoE model within a single kernel, enabling optimizations such as increased parallelism across attention operators, fine-grained pipelining between GroupGEMMs, and model-weight prefetching.

[Figure 17](#figure-17) and [Figure 18](#figure-18) present the raw kernel performance of Qwen3-32B serving on a single B200 and on four B200s with tensor parallelism, respectively. Under the single-GPU setting (TP = 1), ETC achieves consistent gains over both vLLM and SGLang across all batch sizes, with up to a 1.13x speedup over vLLM at batch size 1 and an average improvement of about 7%. In the tensor-parallel case (TP = 4), ETC maintains comparable or better performance across all settings, achieving up to a 1.08x speedup over vLLM while sustaining similar scalability as batch size increases. This on-par performance reflects the effectiveness of ETC’s megakernel design and static scheduling based on the Event Tensor abstraction.

<span id="section-11"></span>

## 11 Dynamic Scheduler Runtime Optimization

We adopt an early push strategy to hide scheduling overhead. Rather than waiting for a task’s dependencies (its producer tasks) to finish executing, the scheduler proactively pushes a consumer task into the ready queue as soon as all of its producer tasks have been dispatched to SMs (not requiring task completion), and the dependency is guaranteed by the extra wait prior to the consumer’s execution. This proactive measure prevents the overhead of the push operation from falling onto the critical path. The push is performed concurrently with the execution of the producer tasks, effectively overlapping the scheduling cost with the preceding computation.

[+1]: For simplicity, we use Numpy einsum-like notations [Har20b] in figures and example codes.

[+2]: `indptr` is a term commonly used in sparse matrix representations such as the compressed sparse row (CSR) format.

[+3]: The pseudocode of the dynamic scheduling transformation is provided in [Section 7](#section-7).

[+4]: A compilation pipeline figure can be found in [Section 8](#section-8).
