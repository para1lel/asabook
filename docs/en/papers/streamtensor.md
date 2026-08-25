---
title: 'StreamTensor: Dataflow Tensor Streaming'
createTime: 2026/08/25 15:18:17
permalink: /en/papers/streamtensor/
pageClass: paper-reading
---

> [Hanchen Ye](https://hanchenye.com/) [+internship], and [Deming Chen](https://dchen.ece.illinois.edu/). First submitted to arXiv on September 17, 2025; current version v2. Published in the [58th IEEE/ACM International Symposium on Microarchitecture (MICRO '25)](https://doi.org/10.1145/3725843.3762817), October 18-22, 2025, Seoul, Republic of Korea. [StreamTensor: Make Tensors Stream in Dataflow Accelerators for LLMs](https://arxiv.org/abs/2509.13694v2). <a href="/paper/streamtensor.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.1145/3725843.3762817). [TeX source](https://arxiv.org/src/2509.13694v2). The original PDF remains authoritative for the exact print layout and bibliography.

[+internship]: *Work was done during an internship at Inspirit IoT, Inc.*

## Abstract

Efficient execution of deep learning workloads on dataflow architectures is crucial for overcoming memory bottlenecks and maximizing performance. While streaming intermediate results between computation kernels can significantly improve efficiency, existing approaches struggle with inter-kernel correlations, external memory access management, and buffer optimization. In this work, we propose StreamTensor, a compiler framework that automatically constructs and optimizes stream-based dataflow accelerators. StreamTensor introduces a novel iterative tensor type system to explicitly encode stream layouts, enabling seamless kernel fusion, buffer allocation, and memory optimization. By systematically exploring three hierarchical design spaces, including tensor tiling, kernel fusion, and resource allocation, StreamTensor balances computational intensity, memory efficiency, and data streaming to maximize performance. Based on FPGA evaluations on Large Language Models (LLM), StreamTensor achieves up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs, making it a promising approach for scalable dataflow-based deep learning acceleration.

<span id="section-1"></span>

## 1 Introduction

<span id="section-1-1"></span>

### 1.1 Dataflow Architecture

Dataflow architecture, as an alternative to Von Neumann-style architectures such as the NVIDIA H100 [Nvi23b] and Google TPUv4 [Jou23a], is increasingly adopted and studied to overcome the memory wall in emerging AI applications, such as Large Language Models (LLM). Because of LLMs' autoregressive nature, the decoding stage is highly memory-bound, demanding more memory-efficient architectures. AMD Versal [Gai19], Sambanova SN40L [Pra24], and IBM AIU [Bur22] are commercial AI accelerators with reconfigurable dataflow architectures; many studies [Pra17, Now17, Che24u] have also demonstrated the latency and energy efficiency advantages of dataflow architecture.

[Figure 1](#figure-01) shows the typical computation pattern of dataflow accelerators. As shown in [Figure 1(b)](#figure-01), a dataflow accelerator contains the following on-chip components:

1. **Kernel**: Computes an operator or coarse-grained *task* (e.g., matrix multiply) using a parallel processor (e.g., a systolic array), and provides stream interfaces for input and output.
2. **Token**: Atomic element communicated between kernels.
3. **First-in First-out (FIFO)**: Holds accumulated stream tokens to balance different token rates of the producer and consumer, and avoids deadlock or unnecessary kernel stalls.
4. **Stream Layout Converter**: Converts stream layout on-the-fly to accommodate different computation patterns of producer and consumer kernels through a local ping-pong buffer.
5. **Direct Memory Access (DMA)**: Communicates with external memory, and converts memory-mapped interfaces to stream interfaces or vice versa.

<span id="figure-01"></span>

![Figure 1. Computation pattern of dataflow accelerators.](../../papers/streamtensor/figure-01.png)

**Figure 1.** Computation pattern of dataflow accelerators.

Kernels may be designed using *dataflow* circuits through dynamic scheduling [Jos18], or may adopt different *dataflow* strategies (e.g., input stationary) for efficient on-chip data reuse [Che17a]. Although using the same terminology, these *dataflow* concepts are conceptually orthogonal to the dataflow architecture and accelerators discussed in this paper.

The key idea of dataflow architecture is to stream intermediate results between kernels through on-chip FIFOs instead of triggering frequent external memory accesses. For example, in [Figure 1(b)](#figure-01), the intermediate results produced by *Kernel0* are streamed directly to *Kernel1* and *Converter0* without going through external memory, as in [Figure 1(a)](#figure-01). Following the convention proposed in [Pra24], we refer to enabling streaming between dataflow kernels as *stream-based kernel fusion*. Additionally, as illustrated in [Figure 1(c)](#figure-01), the schedule of the dataflow accelerator allows *Kernel1* and *Converter0* to start execution before *Kernel0* completes. This overlapped execution can significantly improve both the overall throughput and latency.

<span id="figure-02"></span>

![Figure 2. Current paradigm of dataflow accelerator design.](../../papers/streamtensor/figure-02.png)

**Figure 2.** Current paradigm of dataflow accelerator design.

<span id="section-1-2"></span>

### 1.2 Dataflow Accelerator Programming

[Figure 2](#figure-02) shows the current paradigm of dataflow accelerator programming. As dataflow accelerators generally fall into two categories, application-specific accelerators and domain-specific accelerators (DSAs), we discuss each separately.

<span id="section-1-2-1"></span>

#### 1.2.1 Application-specific Accelerator

In this category, the dataflow components and schedule are tailored for a single application. Thus, *programming* typically refers to the *design* or *generation* of architecture and microarchitecture. Traditionally, Hardware Description Languages (HDLs), High-level Synthesis (HLS), and meta-HDLs like Chisel [Bac12] are used for this purpose [Che05, Zha18a, Chi18a, Sar23]. More recently, Accelerator Design Languages (ADLs) have emerged to improve productivity [Che24v, Dur20, Tho20], introducing typing systems and primitives to describe computation, memory layout, and dataflow schedules. As shown in [Figure 2](#figure-02), existing solutions require manual effort to convert applications into dataflow schedules and components, which are then passed to HLS, meta-HDL transpilers, or vendor EDA tools for hardware generation. While ADLs and HLS frameworks incorporate Design Space Exploration (DSE) [Koe16, Koe18, Ben19a, Ye22, Ago22, Zha24z], these efforts focus mainly on optimizing individual kernels.

<span id="section-1-2-2"></span>

#### 1.2.2 Dataflow DSA

DSAs are designed to efficiently perform computations for a particular class of applications or a specific domain, rather than being a general-purpose processor. DSAs are often realized using Coarse-grained Reconfigurable Architecture (CGRA)-like architectures [Gai19, Pra24, Pra17, Now17], where on-chip resources are reconfigured to implement different dataflow designs. Modern DSAs are programmed using C/C++ primitives [Gai19, Zhu23c, Zhu24e] or Domain-specific Languages (DSLs), such as Spatial [Koe18], Halide [Rag13], and TVM [Che18f], to generate domain-optimized code. As illustrated in [Figure 2](#figure-02), developers must manually transform applications into logical components using these DSLs or APIs. Software compilers then map them to physical resources and generate the final binaries for on-chip execution. While these DSLs often provide auto-tuning capabilities for dataflow kernels, their primary focus is on optimizing individual kernels instead of the entire dataflow application, leaving substantial performance gains unrealized.

<span id="section-1-3"></span>

### 1.3 Pitfalls

<span id="section-1-3-1"></span>

#### 1.3.1 Pitfall 1: Inter-kernel Correlation

Prior works [Ye22, Ye24c] show that inter-kernel correlation can affect accelerator performance. Since kernels execute in a pipelined manner, their latencies must be balanced for optimal throughput. Moreover, buffer-connected kernels need aligned parallelization strategies to avoid inefficient memory use. However, previous work only considered ping-pong buffers, which support memory-mapped access. FIFOs are more restrictive, as data must be pushed/pulled in order. This introduces the following challenges for each kernel:

1. *Tiling*: Choosing tile sizes that enable streaming, minimize local buffering, and preserve memory efficiency.
2. *Permutation*: Reordering loops to reduce memory utilization during data streaming.
3. *Vectorization*: Selecting unrolling strategies to balance latency and improve streaming efficiency.

These decisions are interdependent across kernels, making global optimization challenging for analytical models or manual design.

<span id="section-1-3-2"></span>

#### 1.3.2 Pitfall 2: External Memory Access

Most existing compilers [Ye22, Ago22, Ye24c, Zha22e, Zha24z, Bas25] assume that all data fits on-chip, which is unrealistic for large applications. When off-chip memory is involved, each DMA must address the following issues:

1. How to overlap memory access with kernel execution?
2. What data layout best matches the streaming pattern?
3. How to pack/vectorize data to maximize bandwidth?

These require nontrivial pattern analysis and are error-prone when handled manually. DMA design is also tightly coupled with kernel tiling and scheduling, compounding the complexity.

<span id="section-1-3-3"></span>

#### 1.3.3 Pitfall 3: Stream-based Kernel Fusion

The goal of stream-based kernel fusion is to stream all intermediate results on-chip, limiting external memory use to inputs and outputs. However, producer and consumer kernels often have incompatible stream layouts due to different computation patterns. This requires:

1. Checking layout compatibility between kernels.
2. Generating minimal on-the-fly stream layout converters.
3. Ensuring the converter fits within available on-chip memory.

These steps involve complex pattern analysis and require a global view of the system, making manual solutions impractical.

<span id="section-1-3-4"></span>

#### 1.3.4 Pitfall 4: FIFO Sizing

As shown in [Figure 1](#figure-01), if *Kernel1* is slower than *Converter0*, FIFOs may overflow or underflow, leading to a stall cascade and eventual deadlock. Though dynamic scheduling solutions exist [Jos21], coarse-grained accelerators still rely on manual sizing [Che24v, Che24u], which does not scale to a large number of FIFOs. A recent automated approach [Hon24c] uses simulation to determine FIFO sizes, but it is time-consuming and lacks scalability.

<span id="figure-03"></span>

![Figure 3. Proposed paradigm of dataflow accelerator design.](../../papers/streamtensor/figure-03.png)

**Figure 3.** Proposed paradigm of dataflow accelerator design.

<span id="section-1-4"></span>

### 1.4 Our Proposal

Due to the pitfalls described in [Section 1.3](#section-1-3), the current paradigm shown in [Figure 2](#figure-02) is difficult to scale up to large dataflow accelerators. Therefore, we propose a shift in the design paradigm shown in [Figure 3](#figure-03). We do not advocate for full automation, as ADL/HLS/HDL or DSLs remain essential for designing individual dataflow kernels, such as local buffers and vectorization. However, once individual kernels are designed or generated, we argue that compilers should automatically generate the dataflow schedule, assemble the kernels into an application-level dataflow accelerator, and resolve the pitfalls identified in [Section 1.3](#section-1-3) algorithmically. This is analogous to the GPU software ecosystem, where DSLs like CUDA and Triton [Til19b] are used to design or auto-tune individual GPU kernels, while kernel assembly and scheduling are handled automatically by compilers, resulting in a programming paradigm that is both efficient and scalable.

In this spirit, we propose *StreamTensor*, a compiler that enables automatic tensor streaming in dataflow architectures. This paper describes how each pitfall is addressed in a systematic and hierarchical manner. As a pioneering work, StreamTensor proposes algorithmic solutions for each challenge and demonstrates their effectiveness through large benchmarks. While these solutions may not be optimal, they clearly expose well-defined optimization subproblems and enable co-optimization opportunities across different design spaces. Overall, this paper makes the following contributions:

1. We propose StreamTensor, the first PyTorch-to-device dataflow compiler that automatically generates stream-based dataflow accelerators and their corresponding runtime systems.
2. We propose an iterative tensor (`itensor`) type that systematically encodes the stream information for the first time. This typing system forms the foundation for stream-based kernel fusion and dataflow component generation, improving the scalability and productivity of dataflow accelerator design.
3. We propose three design spaces, including tensor tiling space, kernel fusion space, and resource allocation space, that cover the sophisticated design space of dataflow architecture in an algorithmic and hierarchical manner. We further propose an exploration algorithm for each design space to reduce resource utilization and improve latency and throughput.
4. We propose a piecewise function-based token behavior model that transforms the dataflow FIFO sizing problem of dataflow accelerators into a scheduling problem. We further propose a linear programming (LP) algorithm to solve this problem, reducing resource utilization while avoiding deadlock.
5. We evaluate StreamTensor on FPGA platforms with LLMs and observe up to 0.76x and 0.64x lower latency compared to the state-of-the-art FPGA LLM accelerators and GPUs, and up to 1.99x higher energy efficiency compared to GPUs.

<span id="figure-04"></span>

![Figure 4. Proposed StreamTensor framework.](../../papers/streamtensor/figure-04.png)

**Figure 4.** Proposed StreamTensor framework.

<span id="section-2"></span>

## 2 StreamTensor Framework

StreamTensor is a compilation framework designed to transform PyTorch models into optimized dataflow implementations. It is built upon the MLIR [Lat21a] compilation framework. The overall architecture of StreamTensor is depicted in [Figure 4](#figure-04). The compilation process begins with a PyTorch model from Torch-MLIR [Tor21] and proceeds through several stages. Initially, tensor operations are converted into a structured Intermediate Representation (IR) using MLIR's built-in Linear Algebra (Linalg) operations. This IR is then optimized by MLIR's Linalg passes like element-wise operation fusion. StreamTensor subsequently applies Design Space Exploration (DSE) algorithms to determine optimal tiling strategies, considering factors such as tile sizes, unrolling factors, and permutations based on computational patterns. The Linalg IR is then transformed into a dataflow IR, where computations are organized as hierarchical tasks. All dataflow components, including DMAs, stream layout converters, and FIFOs, are generated during this stage. Critical optimizations are also performed here, such as stream-based kernel fusion to minimize external memory access and FIFO sizing to balance producer-consumer executions. In the final stages, StreamTensor generates hardware-specific code and a host runtime. StreamTensor handles memory allocation, stream connectivity, and directive materialization, which allows vendor compilers like HLS to generate the target dataflow architectures. Concurrently, it produces host runtime code that manages data transfer, kernel execution, and synchronization between the host CPU and the dataflow accelerator.

<span id="figure-05"></span>

![Figure 5. Iterative tensor (`itensor`) typing system.](../../papers/streamtensor/figure-05.png)

**Figure 5.** Iterative tensor (`itensor`) typing system.

<span id="section-3"></span>

## 3 Intermediate Representation

<span id="section-3-1"></span>

### 3.1 Typing System

StreamTensor introduces a typing system to enable efficient verification and optimization of the IR. Through StreamTensor's dedicated type and operation verifiers, the typing system helps ensure the IR's validity after any transformation pass is applied.

<span id="section-3-1-1"></span>

#### 3.1.1 Motivation

Traditionally, `tensor` type encodes a data type and a list of integers representing its shape [Che18f, Rag13, Lat21a, Hag23a]. Tensors can be accessed in a memory-mapped manner, e.g., a slice can be extracted or inserted based on offsets and its shape. However, dataflow kernels communicate via FIFOs, which enforce a strict access order and follow a streamed access pattern rather than a memory-mapped one. Consequently, traditional tensor types may fail to ensure correctness in dataflow communication. Even when a producer and a consumer share the same tensor type, the stream access order may remain ambiguous, causing unintended behaviors. For example, in Graphene [Hag23a], the tensor type only encodes memory-mapped layout. As a result, a mismatch between a producer's row-major stream generation and a consumer's column-major expectation, when both operate on the same tensor type, leads to incorrect data interpretation and logical corruption. Therefore, although existing solutions are sufficient for Linalg-level optimizations like tiling, they are error-prone and unscalable for generating dataflow components and applying dataflow optimizations.

<span id="table-01"></span>

![Table 1. Iterative tensor (`itensor`) operations.](../../papers/streamtensor/table-01.png)

**Table 1.** Iterative tensor (`itensor`) operations.

<span id="section-3-1-2"></span>

#### 3.1.2 Iterative Tensor Type

To address this, we propose a new `itensor` type that explicitly encodes stream layout information, making type-based verification and optimization both possible and efficient. [Figure 5](#figure-05) shows three examples of `itensor`s converted from the same `tensor` with type `tensor<8x8xf32>`. To convert a tensor to an `itensor`, we first partition it into identical tensor slices or vectors. For example, in [Figure 5(b)](#figure-05), the tensor is partitioned into eight tensor slices of shape `4x2`. These slices are then accessed iteratively within a defined iteration space, typically nested loops. The iteration space is defined by two lists: tripcounts and step sizes. In [Figure 5(b)](#figure-05), the iteration space is `[4,2]*[2,4]`, which produces iteration indices `[0,0]`, `[0,4]`, `[2,0]`, `[2,4]`, etc. The mapping from iteration space to data space is specified by an affine map—for example, `(d0,d1)->(d1,d0)` in [Figure 5(b)](#figure-05), which transposes the iteration indices. Thus, the data access indices become `[0,0]`, `[4,0]`, `[0,2]`, `[4,2]`, etc., reflecting this transposition as shown in [Figure 5(b)](#figure-05). In `itensor`, tensor slices can be accessed multiple times, with the pattern explicitly encoded in the iteration map. For instance, in [Figure 5(c)](#figure-05), the iteration space is `[4,2,2]*[2,1,4]` and the iteration map is `(d0,d1,d2)->(d2,d0)`, where dimension `d1` does not correspond to any data dimension. As `d1` iterates from 0 to 1, all less significant dimensions (like `d2`) are reiterated. Consequently, the corresponding data dimensions (e.g., row dimension) are also re-accessed, producing indices like `[0,0]`, `[4,0]`, `[0,0]`, `[4,0]`, `[0,2]`, etc., for tensor slices of shape `4x2`. By encoding the element shape, iteration space, and iteration map in the `itensor` type, the stream pattern of a dataflow kernel can be uniquely determined. When the `itensor` types of a producer and consumer match, streaming communication can be safely established between them (*Case1* of [Figure 5](#figure-05)). Otherwise, a stream layout converter must be inserted in between (*Case2* of [Figure 5](#figure-05)), and the minimal ping-pong buffer size for layout conversion can be analytically inferred from the `itensor` types. The details of layout converter generation will be discussed in [Section 5.2.1](#section-5-2-1). Due to the lack of stream information, existing tensor-based typing systems are not sufficient for stream-based kernel fusion, limiting their usability in stream-based dataflow optimizations.

<span id="section-3-1-3"></span>

#### 3.1.3 Stream Type

In traditional tensor compilers, high-level tensor IR must be *bufferized* into a low-level memory/buffer IR to enable low-level optimizations and code generation. Following this convention, we propose a `stream` type, which is lowered from `itensor` type during bufferization. Unlike immutable `itensor` objects, `stream` objects represent hardware FIFOs and support mutation through operations such as stream reads and writes. The `stream` type encodes only the data type and FIFO depth, while the stream layout information is stripped during bufferization. As a result, dataflow component generation and optimization must be completed at the `itensor` level IR. After bufferization, the `stream` IR is reserved for lower-level hardware/runtime optimizations and code generation.

<span id="table-02"></span>

![Table 2. Stream (`stream`) and buffer operations.](../../papers/streamtensor/table-02.png)

**Table 2.** Stream (`stream`) and buffer operations.

<span id="table-03"></span>

![Table 3. Structure operations.](../../papers/streamtensor/table-03.png)

**Table 3.** Structure operations.

<span id="section-3-2"></span>

### 3.2 Operations

Built upon the typing system in [Section 3.1](#section-3-1), StreamTensor introduces `itensor` and `stream` operations to represent different dataflow behaviors. Additionally, structure operations are introduced to represent the multi-level hierarchy of a dataflow accelerator, and are shared by both `itensor` and `stream`-level IRs.

<span id="section-3-2-1"></span>

#### 3.2.1 Iterative Tensor Operations

[Table 1](#table-01) lists the complete set of operations at the `itensor` level. Overall, these operations are self-explanatory; we highlight those whose semantics are less obvious. `itensor_write` can be conceptually understood as writing or pushing an element into a FIFO. It is a destination-carried operation, where the destination is an `itensor` passed through a `dest` operand. For example, iteratively writing the `itensor` in [Figure 5(b)](#figure-05) (referred to as `itensor(b)`) can be expressed as:

```text
%empty = itensor_empty() : itensor(b)
%res0 = scf.for 0 to 8 step 2 iter_args={%arg0 = %empty} {
  %res1 = scf.for 0 to 8 step 4 iter_args={%arg1 = %arg0} {
    %value = ... : tensor<4x2xf32> // %value is defined
    %output = itensor_write %value into %arg1 : ...
    scf.yield %output : itensor(b)
  } : itensor(b)
  scf.yield %res1 : itensor(b)
} : itensor(b)
```

Here, `scf` is an MLIR built-in dialect for structured control flow, including `for` loops. `scf.for` is also destination-carried, where `%empty` is passed as an argument and iteratively pushed through an `itensor_write`. Eventually, `%res0` is returned as the final result. In contrast, `itensor_read` represents pulling an element from a FIFO. For example, reading `itensor(b)` can be expressed as:

```text
%source = ... : itensor(b) // %source is defined
scf.for 0 to 8 step 2 {
  scf.for 0 to 8 step 4 {
    %empty = tensor.empty() : tensor<4x2xf32>
    %value = itensor_read %source init %empty : ...
    ... = ... %value ... // %value is used
  }
}
```

`itensor_converter` contains a local ping-pong buffer that performs on-the-fly stream layout conversion. For example, in *Case1* of [Figure 5](#figure-05), the source and target share the same `itensor` type and can connect via a FIFO. In *Case2*, they differ, so a converter must be inserted. A minimum `8x2` ping-pong buffer is required to accommodate the stream layouts. While the source writes to the ping buffer, the target reads the pong buffer twice, then they swap.

<span id="section-3-2-2"></span>

#### 3.2.2 Stream Operations

[Table 2](#table-02) lists the operations at the `stream` level. These are mostly self-explanatory; we highlight the key difference from `itensor` operations. As discussed in [Section 3.1.3](#section-3-1-3), `stream` objects are mutable, and destination-carried semantics are no longer used. A FIFO push and pull can be written as:

```text
%stream = stream() : stream<f32, depth: 32>
scf.for 0 to 8 step 2 {
  scf.for 0 to 8 step 4 {
    %value = ... : f32 // %value is defined
    stream_write %value into %stream : ...
  }
}
scf.for 0 to 8 step 2 {
  scf.for 0 to 8 step 4 {
    %value = stream_read %stream : ...
    ... = ... %value ... // %value is used
  }
}
```

Note that the same `%stream` is used throughout without creating new duplicates, unlike the destination-carried style of `itensor`. `stream` IR is more efficient for code generation, but complicates define-use analysis. Hence, `itensor` is preferred for high-level dataflow optimization. The correctness of `stream` operations is guaranteed by construction as they are lowered from `itensor` operations, which are strictly verified by the `itensor` typing system.

<span id="figure-06"></span>

![Figure 6. Linalg tiling, Linalg to dataflow conversion, and dataflow kernel fusion. A solid arrow indicates an operation on the left is transformed into the operation on the right, whereas a dashed arrow indicates an operation that remains unchanged.](../../papers/streamtensor/figure-06.png)

**Figure 6.** Linalg tiling, Linalg to dataflow conversion, and dataflow kernel fusion. A solid arrow indicates an operation on the left is transformed into the operation on the right, whereas a dashed arrow indicates an operation that remains unchanged.

<span id="section-3-2-3"></span>

#### 3.2.3 Structure Operations

While `itensor` and `stream` operations model behavior, structure operations model hierarchy. [Table 3](#table-03) lists all the structure operations in StreamTensor. The `kernel` operation represents a dataflow kernel (as in [Figure 1](#figure-01)), containing a graph of `task` operations. It takes `tensor`s as inputs/outputs, which are converted to/from `itensor`s at the boundary. These implicit conversions act as DMAs. Intra-`kernel` uses on-chip streaming, while inter-`kernel` uses external memory. For example:

```text
%source = ... : tensor<8x8xf32> // %source is defined
%result = kernel(
  %arg : itensor<b> = %source : tensor<8x8xf32>
) {
  ... = ... %arg ... // %arg is used
  %output = ... : itensor<c> // %output is defined
  yield %output : itensor<c>
} : tensor<8x8xf32>
```

By converting at the kernel boundary, we avoid explicit DMA handling during kernel fusion, improving transformation efficiency and analyzability. In contrast, the `task` operation is transparent and does not convert types at its boundary. It represents a dataflow task within a kernel and may be nested for hierarchical dataflow designs. At the `itensor` level, `task` is destination-carried where outputs are written into destinations via `inits`, improving the efficiency of define-use analysis. For example:

```text
%empty = ... : itensor(b)
%result = task @example inits={%arg = %empty} {
  %value = ... : tensor<4x2xf32> // %value is defined
  %output = itensor_write %value into %arg : ...
  yield %output : itensor(b)
} : itensor(b)
```

After lowering and bufferization, the same code becomes:

```text
%stream = stream() : stream<f32, depth: 32>
task @example {
  %value = ... : f32 // %value is defined
  stream_write %value into %stream : ...
}
```

We can observe that `task` combines both `itensor` and `stream` operations, making it a unifying structure abstraction across both IRs that serve different levels of dataflow optimizations. Eventually, all dataflow `task`s are lowered to MLIR built-in `call` and `func` operations for code generation.

<span id="section-4"></span>

## 4 Compilation Pipeline

Building on the type system and operations, we introduce a compilation pipeline that compiles Linalg IR into hardware implementations and a corresponding runtime. All compilation passes are shown in [Figure 4](#figure-04). In this section, we focus on the Linalg-to-dataflow conversion, dataflow kernel fusion, and dataflow optimizations that are unique and essential to understanding the compiler.

<span id="section-4-1"></span>

### 4.1 Linalg to Dataflow

[Figures 6(a)-(c)](#figure-06) illustrate the Linalg-to-dataflow conversion process. The original Linalg operations ([Figure 6(a)](#figure-06)) are first tiled into [Figure 6(b)](#figure-06), where `scf.for`s represent the loop nests for tiling. In each iteration, `extract_slice`s extract input tensor tiles to feed into the tiled Linalg operation. After the operation produces output tiles, `insert_slice`s insert them back into the full tensor. Then, each tiled loop nest is converted into a `kernel` operation in place as shown in [Figure 6(c)](#figure-06). The input and output `tensor`s are converted into/from `itensor`s at the boundary of `kernel`s. The `itensor` types are inferred from:

1. The nested `scf.for` loops — iteration tripcounts and step sizes define the `itensor` iteration space.
2. The `extract_slice` and `insert_slice` operations' offsets and sizes — offsets define the iteration mapping, while sizes define the element shape. For example, offsets `[%iv2, %iv0]` result in the iteration map `(d0,d1,d2)->(d2,d0)`.

After conversion, `extract_slice` and `insert_slice` operations are replaced with `itensor_read` and `itensor_write` operations, respectively. The resulting `scf.for` loop nest is wrapped in a `task` to form a single-level dataflow hierarchy: a dataflow kernel containing a dataflow task. By converting the Linalg semantics to dataflow, we open opportunities for subsequent dataflow-oriented transforms and optimizations.

<span id="section-4-2"></span>

### 4.2 Dataflow Kernel Fusion

After all tiled Linalg operations are converted to dataflow kernels, all these kernels initially communicate via traditional `tensor`s, which are eventually stored in external memory. To reduce this communication overhead, StreamTensor applies stream-based kernel fusion. [Figures 6(c)-(d)](#figure-06) show this process. To fuse *Kernel0* and *Kernel1*, we first compare the output `itensor` type of *Kernel0* with the input `itensor` type of *Kernel1*. As described in [Section 3.1.2](#section-3-1-2), if the types match, we can directly fuse the kernels. If not, we insert a stream layout converter as shown in [Figure 6(d)](#figure-06). The fused kernel comprises two `task`s and a `converter`, all communicating via `itensor`s that will be lowered to on-chip stream FIFOs. The `itensor` typing system enables any dataflow kernels to be fused *by design* at the cost of potential on-chip memory utilization for converters. In [Section 5.2](#section-5-2), we will discuss the exploration of kernel fusion space given memory constraints.

After fusion, StreamTensor applies additional optimization passes to improve the efficiency of external memory access. In particular, `tensor` `pack` and `unpack` operations are inserted before and after the `kernel` to convert between default and tiled memory layouts for burst memory access. For example, with a tiling size of `[16,16]` on a `64x64` tensor, the packed tensor has shape `4x4x16x16`. To maximize the usage of external memory bandwidth, StreamTensor widens the tensor with vectors. For instance, with 512-bit DDR or HBM and `uint8` elements, grouping 64 elements into `vector<64>` fully utilizes the bandwidth. In [Figure 6](#figure-06), the packed tensor is widened to shape `4x4x2x2xvector<8x8>`. Note that `pack` and `widen` operations are eventually lowered to runtime operations on the host CPU, which prepares data for the accelerator and causes some latency and memory overhead. However, for static tensors (e.g., pre-trained parameters), `pack` and `widen` can fuse directly into these tensors, eliminating any runtime costs. For dynamic tensors (e.g., activations), `pack` and `widen` operations can be folded with their `unpack` and `unwiden` counterparts from the preceding layer via effective Linalg tiling space exploration. As a result, the `pack` and `widen` operations, being necessary only for the model's inputs and outputs, contribute negligible memory and latency overhead at runtime.

<span id="figure-07"></span>

![Figure 7. Materialize converter & DMA, fold `itensor`, and vectorize `itensor`. A solid arrow indicates an operation on the left is transformed into the operation on the right, whereas a dashed arrow indicates an operation that remains unchanged.](../../papers/streamtensor/figure-07.png)

**Figure 7.** Materialize converter & DMA, fold `itensor`, and vectorize `itensor`. A solid arrow indicates an operation on the left is transformed into the operation on the right, whereas a dashed arrow indicates an operation that remains unchanged.

<span id="section-4-3"></span>

### 4.3 Dataflow Optimization

<span id="section-4-3-1"></span>

#### 4.3.1 Materialization

[Figures 7(a) and (b)](#figure-07) illustrate the *materialization* process for converters and DMAs. Materialization involves transforming a high-level dataflow component into its low-level implementation, typically `scf.for` loop nests containing `tensor` and `itensor` operations. Initially, converters are represented by `itensor_converter`, while DMAs are implicitly handled via `tensor` to or from `itensor` conversions at `kernel` boundaries. This abstraction facilitates kernel fusion and converter optimization. For instance, redundant converters generated for multiple consumers of a producer can be removed using MLIR’s Common Sub-expression Elimination (CSE), which becomes harder after materialization. In contrast, after materialization, all dataflow components are expressed as nested `task`s, making further dataflow optimizations efficient and accessible. For converters, as shown in [Figure 7(a)](#figure-07), *Converter0* contains two `scf.for` loop nests connected with a `16x64` ping-pong buffer. These two loop nests are wrapped by a *shared* parent `scf.for` loop to iterate through the original full `64x64` tensor. Therefore, the `16x64` ping-pong buffer is reused four times, effectively reducing on-chip memory resource utilization by a factor of four. In [Section 5.2](#section-5-2), we will discuss how the ping-pong buffer shape and shared loops are inferred from `itensor` types.

For DMAs, as shown in [Figure 7(a)](#figure-07), the input type conversion from `tensor<4x4x2x2xvector<8x8>>` to `itensor<16x16...>` indicates a DMA that will: 1) load `4x4x2x2` times `vector<8x8>` data from external memory; 2) store this data in a `16x16` ping-pong buffer to hide external memory access latency; and 3) push the data to a FIFO with a layout encoded in the `itensor` type. In [Figure 7(b)](#figure-07), we observe that *DMA0* is automatically generated to implement these three behaviors using `scf.for` loop nests. Note that our `itensor`-based typing system encodes all the converter and DMA information. This is a capability that traditional tensor types lack, limiting their utility in dataflow component generation.

<span id="section-4-3-2"></span>

#### 4.3.2 Iterative Tensor Folding

[Figures 7(b)-(c)](#figure-07) show the `itensor` folding. Suppose we have an `itensor_write` in *DMA0* and an `itensor_read` in *Kernel0*, connected via a FIFO. These represent two separate local buffers connected by streaming. By folding, we eliminate the FIFO and merge the two buffers. This optimization can reduce on-chip memory utilization while improving the overall latency by increasing the overlap between kernels. As shown in [Figure 7(c)](#figure-07), the fetched tile is directly passed to the `linalg.generic` op in *Kernel0*, eliminating redundant buffering and communication. `itensor` folding requires an exact match in memory access patterns between producer and consumer. This makes it more restrictive than stream-based kernel fusion, which can be applied between any dataflow kernels. Consequently, we implement `itensor` folding as an additional optimization upon already fused kernels.

<span id="section-4-3-3"></span>

#### 4.3.3 Iterative Tensor Vectorization

As dataflow kernels often run in parallel, we must vectorize dataflow FIFOs to provide sufficient bandwidth. [Figures 7(c)-(d)](#figure-07) show the vectorization of an `itensor` into `vector<2x4>`. On the *DMA0+Kernel0* side, the `itensor_write` becomes a loop with `transfer_read` (from the buffer) followed by `itensor_write` (to the FIFO). On the *Converter0* side, similar transformations are applied for reading. This process aligns FIFO bandwidth with the parallelism of the dataflow kernel.

<span id="section-5"></span>

## 5 Design Spaces

To generate realizable and optimized accelerators, we must configure the compilation pass parameters properly. As shown in [Figure 4](#figure-04), we divide the overall design space into three sub-spaces: Linalg tiling space, kernel fusion space, and resource allocation space.

<span id="section-5-1"></span>

### 5.1 Linalg Tiling Space

The Linalg tiling space determines tiling factors, unrolling factors, permutation strategies, and input/output vectorization for each dataflow kernel. In StreamTensor, this space is represented by a graph of Linalg operations, with properties such as loop trip counts, step sizes, and loop types (reduction or parallel) annotated on each node. The results of the exploration are also written back to this graph to configure transformation passes.

For tiling, a hyperparameter `default_tile_size` is exposed to users and applied across all dimensions of all kernels. For unrolling, we develop an intensity-aware algorithm, which iteratively selects the kernel with the longest latency through a max-heap and increases its unroll factor until a user-defined hyperparameter `overall_unroll_size` is reached. This approach balances kernel latencies to improve throughput. Once unroll sizes are determined, vectorization factors are inferred by analyzing the loop iteration space and tensor shapes. Permutation is handled by a heuristic that moves reduction loops outward while keeping parallel loops innermost, reducing initiation intervals (II) of pipeline loops. In StreamTensor, the hyperparameters of the Linalg tiling space are automatically explored through a blackbox optimizer, Optuna [Aki19], with the feedback from dataflow kernel fusion results.

<span id="section-5-2"></span>

### 5.2 Kernel Fusion Space

As described in [Section 4.2](#section-4-2), kernel fusion enables streaming between kernels. If the producer and consumer have different `itensor` types, a converter must be inserted. The exploration of the Linalg tiling space determines all data layouts and shapes, thereby fixing the `itensor` types at the interfaces of all dataflow kernels. Consequently, the memory overhead of fusing any pair of kernels is also established. Due to limited on-chip memory, fusing all kernels is generally not feasible. To effectively select which kernel pairs to fuse while adhering to memory resource constraints, we propose two algorithms: [Algorithm 1](#algorithm-01) that infers the minimal ping-pong buffer shape required by the stream layout converter; and [Algorithm 2](#algorithm-02) that determines a global fusion plan under on-chip memory constraints.

<span id="algorithm-01"></span>

**Algorithm 1: Pseudo code of stream layout converter generation.**

- **Input:** $\mathit{src}$, Source `itensor` type; $\mathit{res}$, Result `itensor` type.
- **Output:** $\mathit{bufShape}$, Shape of the ping-pong buffer; $\mathit{beforeLoop}$, Loop index where the ping-pong buffer is inserted.
- Set $\mathit{bufShape}\gets []$ and $\mathit{beforeLoop}\gets 0$.
- Set $\mathit{sharedLoops}\gets []$, the indices of loops shared by $\mathit{src}$ and $\mathit{res}$.
- **For** $\mathit{dim}\gets 0$ **to** $\mathit{src}.\mathrm{rank}()-1$:
  - **If** $\mathit{src}.\mathrm{elementSize}(\mathit{dim})\neq\mathit{res}.\mathrm{elementSize}(\mathit{dim})$: **break**.
  - Set $\mathit{srcExpr}\gets\mathit{src}.\mathit{iterMap}[\mathit{dim}]$.
  - Set $\mathit{resExpr}\gets\mathit{res}.\mathit{iterMap}[\mathit{dim}]$.
  - **If** both $\mathrm{Expr}$s are dimensions with same position:
    - Set $\mathit{bufShape}.\mathrm{append}(\mathit{src}.\mathrm{elementSize}(\mathit{dim}))$.
    - Set $\mathit{sharedLoops}.\mathrm{append}(\mathit{srcExpr}.\mathit{pos})$.
    - Set $\mathit{beforeLoop}\gets\mathit{beforeLoop}+1$.
  - **Else:** **break**.
- **While** any $\mathit{loop}\in\mathit{sharedLoops}$ where $\mathit{loop}\geq\mathit{beforeLoop}$:
  - Set $\mathit{bufShape}.\mathrm{pop}()$ and $\mathit{loop}\gets\mathit{sharedLoops}.\mathrm{pop}()$.
  - **If** $\mathit{loop}\neq -1$: set $\mathit{beforeLoop}\gets\mathit{beforeLoop}-1$.
- Set $\mathit{bufShape}.\mathrm{append}(\mathit{src}.\mathit{shape}[\mathit{bufShape}.\mathrm{size}():])$.
- **Return:** $\{\mathit{bufShape},\mathit{beforeLoop}\}$.

<span id="section-5-2-1"></span>

#### 5.2.1 Stream Layout Converter Generation

[Algorithm 1](#algorithm-01) compares the source and target `itensor`s across each data dimension (lines 3-16). The ping-pong buffer size can be reduced along a data dimension only if: 1) their element sizes are equal (lines 4-5); and 2) their corresponding iteration dimensions are equal, referring to the same loop nesting level (lines 8-16). For instance, in [Figure 5](#figure-05), the second data dimensions of `itensor(b)` and `itensor(c)` both correspond to iteration dimension `d0`, allowing this dimension to be reduced; we only need to buffer a single column of tiles. In materialization, shared loops will be generated to reuse the buffer along this reduced dimension. Conversely, their first data dimensions correspond to iteration dimensions `d1` and `d2`, respectively, making them non-reducible. Thus, we must buffer all rows of tiles. Consequently, as [Figure 5](#figure-05) illustrates, two tiles (four tiles after ping-pong buffering) are required in the layout converter.

After identifying reducible data dimensions and corresponding shared loops, the algorithm filters out those that have parent loops that are not shareable, ensuring buffer realizability (lines 17-19). For example, if loop-`{0,1,2,4}` are shareable but loop-3 is not, loop-4 must be excluded. Finally, the buffer shape and shared loops are returned. This process's worst case occurs when no dimension is reducible, demanding that the entire data be held on-chip for fusion. This may result in significant memory overhead.

<span id="algorithm-02"></span>

**Algorithm 2: Pseudo code of kernel fusion exploration.**

- **Input:** $G$, kernel fusion design space; $C_{\max}$, max fusion cost.
- **Output:** $F$, sets of nodes to be fused; $C$, costs of fused nodes.
- Set $F\gets[\emptyset]$, $C\gets[0]$, and $M\gets\{\}$, a map from node to index of fusion.
- **For** $n$ **in** $\mathrm{topo\_sort}(G)$:
  - Set $\mathit{cand}\gets\{\}$, a map from index of fusion candidate to cost.
  - **For** $p$ **in** $G.\mathrm{predecessors}(n)$:
    - Set $\mathit{cost}\gets\mathrm{compute\_memory\_cost}(G.\mathit{edges}[p,n,0])$.
    - Set $\mathit{cand}[M[p]]\gets\mathit{cand}.\mathrm{get}(M[p],0)+\mathit{cost}$.
  - Set $\mathit{f\_idx}\gets\mathrm{len}(F)$ and $\mathit{f\_cost}\gets 0$.
  - **If** $\mathrm{len}(\mathit{cand})>0$:
    - Set $\mathit{f\_idx}\gets\max(\mathit{cand}.\mathrm{keys}())$ and $\mathit{f\_cost}\gets\mathit{cand}[\mathit{f\_idx}]$.
  - **If** $\mathit{f\_idx}=\mathrm{len}(F)$ **or** $\mathit{f\_cost}+C[\mathit{f\_idx}]>C_{\max}$:
    - Set $F.\mathrm{append}(\{n\})$, $C.\mathrm{append}(0)$, and $M[n]\gets\mathrm{len}(F)-1$.
  - **Else:**
    - Set $F[\mathit{f\_idx}].\mathrm{add}(n)$ and $C[\mathit{f\_idx}]\gets C[\mathit{f\_idx}]+\mathit{f\_cost}$.
    - Set $M[n]\gets\mathit{f\_idx}$.
  - Set $G.\mathit{nodes}[n][\texttt{"fusion\_index"}]\gets M[n]$.
- **Return:** $F,C$.

<span id="figure-08"></span>

![Figure 8. Token behavior modeling with piecewise linear function and linear-programming-based FIFO sizing formulation.](../../papers/streamtensor/figure-08.png)

**Figure 8.** Token behavior modeling with piecewise linear function and linear-programming-based FIFO sizing formulation.

<span id="section-5-2-2"></span>

#### 5.2.2 Kernel Fusion Exploration

The input $C_{\max}$ (*max fusion cost*) for [Algorithm 2](#algorithm-02) represents the maximum on-chip memory a single fused kernel can utilize. For FPGAs, this is typically set to the total on-chip memory size. Consequently, the kernel fusion process can also be viewed as a graph partitioning problem. After fusion, each resulting fused kernel will occupy a single FPGA. If a computation graph comprises multiple such kernels, they can be executed across multiple FPGAs, on a single FPGA sequentially, or with a hybrid approach. StreamTensor supports all these approaches as a compiler. However, mapping $M$ kernels to $N$ FPGAs and managing inter-FPGA communication are beyond the scope of this paper. [Algorithm 2](#algorithm-02) traverses all `kernel`s in a topological order (line 3). For each kernel, it first gathers fusion candidates from predecessors and computes the fusion cost (lines 4-11). The kernel is fused with the nearest valid candidate (lines 13-14) if it does not exceed the resource limit (lines 15-20). Fusion results are written back to the graph (line 22) and used to configure the optimizations discussed in [Section 4.2](#section-4-2). Dataflow kernel fusion always has a feasible solution unless a single kernel occupies more resources than a single FPGA. In such a case, the result is fed back to the tiling space for refinement, for example, reducing tiling and/or unrolling factors.

<span id="section-5-3"></span>

### 5.3 Resource Allocation Space

On hardware like FPGAs, due to limited on-chip memory and compute resources, effective resource allocation greatly affects routing congestion and clock frequency. In this space, we need to solve:

1. **FIFO sizing**: Determine FIFO depths to avoid deadlocks and improve execution overlap. This section will cover more details.
2. **Graph partitioning**: On multi-die hardware, we need to assign `task`s to dies. This assignment problem is formulated and solved using Integer Linear Programming (ILP). In our ILP model, a binary list represents each `task`'s assignment. A constraint ensures that only one element in this list can be "1", with its position indicating the assigned die. The ILP objective is to minimize both inter-die communication and resource imbalance across the dies. Since similar formulations have been studied [Guo21b, Du23b], we omit further details.
3. **Memory allocation**: Place each buffer in LUTRAM, BRAM, or URAM on FPGAs, prioritized by size. Since this algorithm is straightforward, we omit further details.

<span id="section-5-3-1"></span>

#### 5.3.1 Token Behavior Model

To address the FIFO sizing problem discussed in [Section 1.3](#section-1-3), we first propose a token production and consumption model based on piecewise linear functions. [Figure 8(a)](#figure-08) illustrates the token communication between *Source* and *Target* kernels fused through *InterFIFO*. Pipeline II is the cycle count between two consecutive output tokens, while initial delay is the cycle count required to produce the first output token. A token is defined as the atomic data element communicated between kernels. At *time0*, all five input tokens are in *InputFIFO*, and tokens begin to stream into *Source* at *time1*. At *time5*, *Source* pushes *token1* into *InterFIFO*, while *Target* consumes *token0*, leaving one token in *InterFIFO*. At *time6*, *Target* cannot consume *token1* because it requires two cycles to process *token0*. Meanwhile, *token2* is pushed into *InterFIFO*, increasing its token count to two. At *time8*, *Source* finishes processing tokens, when *InterFIFO* holds its maximum capacity of three tokens. *Target* then continues to consume and process the remaining tokens until *time15*, when all tokens are fully processed.

To model these complex behaviors with an analyzable function, we reorganize the token statuses from [Figure 8(a)](#figure-08) into [Figure 8(b)](#figure-08), aligning the statuses of the same token in the same row. We observe that the boundary between the *Source* (blue) and *InterFIFO* (red) sections can be perfectly modeled with a piecewise linear function (blue curve). This function represents the token count *produced* by *Source*. Similarly, we can model the token count *consumed* by *Target* with the orange curve. The difference between these two curves represents the token count in *InterFIFO*. These curves can be represented by the kernel's latency, initial delay, and pipeline II. StreamTensor automatically invokes vendor tools like HLS to profile these metrics for each kernel in the middle of the flow. Since these metrics are specific to vendor platform's architecture, technology node, and mapping strategy, they must be obtained through this profiling process. As resource allocation is the last design space, the kernel designs remain unchanged in the subsequent StreamTensor flow. As long as the vendor tools use a deterministic scheduling algorithm, the final accelerator's metrics will match those profiled earlier. This consistency guarantees the validity of our algorithm.

<span id="section-5-3-2"></span>

#### 5.3.2 Maximum Token Calculation

As shown in [Figure 8(c)](#figure-08), we define $L$ as the total latency of *Source* execution; $D$ as the initial delay from the start of *Source* execution to the production of its first output token; $\mathit{delay}$ as the time from the start of *Source* execution to the start of *Target* execution. Naturally, $\mathit{delay}$ is always greater than or equal to $D$ since *Target* cannot start its execution before the first token is produced by *Source*. We define $T$ as the exact number of tokens passed from *Source* to *Target* for a single accelerator execution. $T$ is a static value that can be analytically inferred from tensor shapes in StreamTensor. We will address how to handle dynamic tensor shapes in [Section 5.3.5](#section-5-3-5). With a static $T$ value, the maximum token count in *InterFIFO*, $\mathit{max\_tokens}$, can be analytically calculated from $\mathit{delay}$:

<span id="equation-01"></span>

$$
\mathit{max\_tokens}=\min\left(T,~T-\left\lfloor\frac{L-\mathit{delay}}{\mathrm{II}_{\mathrm{Target}}}\right\rfloor\right)
$$

The pipeline $\mathrm{II}$ determines the slope of the curve, i.e., the kernel throughput. [Figure 8(c)](#figure-08) illustrates the case where *Source*'s throughput is greater than *Target*'s. Conversely, when *Source*'s throughput is lower, data starvation may limit *Target*'s throughput. [Figure 8(d)](#figure-08) shows that *Target* is unaffected with a sufficiently large $\mathit{delay}$, whereas [Figure 8(e)](#figure-08) shows that *Target* is eventually starved and its throughput is equalized to *Source*'s throughput. In both cases, $\mathit{max\_tokens}$ can be calculated from $\mathit{delay}$:

<span id="equation-02"></span>

$$
\mathit{max\_tokens}=\min\left(T,~\left\lceil\frac{\mathit{delay}-D}{\mathrm{II}_{\mathrm{Source}}}\right\rceil\right)
$$

Equations [1](#equation-01) and [2](#equation-02) both reveal a positive correlation between $\mathit{max\_tokens}$ and $\mathit{delay}$. As shown in [Figure 8(c)-(e)](#figure-08), setting the *InterFIFO* depth to $\mathit{max\_tokens}$ prevents back-pressure from *Target* onto *Source*. This ensures steady, periodic behavior between any pair of *Source* and *Target* across multiple accelerator executions. By preventing stalls from back-pressure, the analytical relationship between $\mathit{max\_tokens}$ and $\mathit{delay}$ is preserved.

<span id="section-5-3-3"></span>

#### 5.3.3 Equalization

The approach described in [Section 5.3.2](#section-5-3-2) is named as the *Normal* equalization strategy, which assumes that kernels always produce tokens at their original throughput. However, the throughput of a dataflow accelerator is ultimately determined by its slowest kernel. Based on this, we propose a *Conservative* equalization strategy, which *scales* the pipeline II of all kernels to match the throughput of the slowest kernel. The resulting *max_tokens* values are smaller than or equal to those from the *Normal* strategy because the gap between any pair of *Source* and *Target* curves is minimized. The drawback is that faster kernels are frequently stalled by back-pressure, potentially increasing the latency. Therefore, the *Normal* and *Conservative* strategies present a trade-off between area and performance, where the *Conservative* strategy minimizes FIFO buffer sizes at the cost of increased overall latency. The key difference between the *Conservative* and *Normal* strategies lies in how their IIs are initially scaled. Because this scaling preserves the piecewise-linear nature of the kernel curves, the equations for calculating $\mathit{max\_tokens}$ from $\mathit{delay}$ remain identical for both strategies.

<span id="section-5-3-4"></span>

#### 5.3.4 LP-based FIFO Sizing

By introducing the token behavior model, we transform the FIFO sizing problem into a problem of determining the $\mathit{delay}$ values between kernels. [Figure 8(f)](#figure-08) shows an example of dataflow graph. *Kernel0* has two outputs; *Kernel1* depends on *Kernel0*; *Kernel2* has two operands and must wait for both *Kernel0*'s and *Kernel1*'s first tokens. Given that *Kernel1* produces its first token after `D[0]+D[1]`, `delay[0][2]` must be greater than or equal to this value. Their relationship is depicted in [Figure 8(f)](#figure-08), with the green curve representing *Kernel1*. The maximum token count for the FIFO between *Kernel0* and *Kernel2*, `max_token[0][2]`, can then be calculated using `delay[0][2]`. If the FIFO size is smaller than this maximum, *Kernel0* will stall due to back-pressure, which harms overall performance. This stall can propagate to *Kernel1* and *Kernel2*, preventing the back-pressure from resolving and potentially causing a deadlock. A FIFO size equal to `max_token[0][2]` is sufficient to prevent back-pressure and avoid a deadlock; it is also required to prevent performance degradation from unintended kernel stalls. We propose an LP formulation to optimally solve for the $\mathit{delay}$ values. Given $G=(V,E)$, where $V$ is the set of kernels and $E$ is the set of edges between the kernels, the objective and constraints of LP are:

<span id="equation-03"></span>

<span id="equation-04"></span>

$$
\begin{aligned}
\mathrm{minimize} & \sum_{e_{i,j}\in E}\mathit{delay}(i,j) \\
\forall u,v\in V,\forall \mathit{path}\in P_{u,v}, & \sum_{e_{i,j}\in \mathit{path}}\mathit{delay}(i,j)\geq \mathrm{threshold}(u,v)
\end{aligned}
$$

$e_{i,j}\in E$ covers all edges in the graph; $\mathit{path}\in P_{u,v}$ covers all full paths connecting any pair of kernels, named `u` and `v`; $e_{i,j}\in \mathit{path}$ covers all edges along a $\mathit{path}$ connecting the two kernels `u` and `v`. We minimize the summation of `delay`s on all edges, which serves as a proxy for optimizing FIFO sizes due to the positive correlation between $\mathit{max\_tokens}$ and $\mathit{delay}$. $\mathrm{threshold}(u,v)$ is the maximum accumulated $D$ over all paths connecting the two kernels `u` and `v`:

<span id="equation-05"></span>

$$
\mathrm{threshold}(u,v)=\max_{\mathit{path}\in P_{u,v}}\sum_{e_{i,j}\in \mathit{path}}D(i)
$$

The LP formulation for the example above is shown in [Figure 8(f)](#figure-08). Note that in this example, the two paths diverging from *Kernel0* re-converge to *Kernel2* as two distinct input operands, rather than joining into a single input. We will discuss the handling of dynamic behaviors like path joining in [Section 5.3.5](#section-5-3-5). Resource constraints are not needed for the LP problem for two reasons: First, as discussed in [Section 5.2](#section-5-2), dataflow kernel fusion guarantees that all fused kernels will fit within available on-chip resources by restricting the fusion cost. Second, the memory utilization of stream FIFOs is negligible compared to that of dataflow kernels and converters. Consequently, the LP problem can be optimally solved in polynomial time. Notably, we do not need to enforce vendor tools to implement the $\mathit{delay}$s. Instead, the $\mathit{delay}$s are automatically fulfilled through the FIFO dependencies between dataflow kernels. In the example above, *Kernel2* automatically waits for *Kernel1* because it depends on *Kernel1*'s output token.

<span id="table-04"></span>

![Table 4. Comparison with previous works on GPT-2 model. *TTFT* measures the time to first token in ms, the lower the better. *Speed* measures the decoding speed in token/s, the higher the better. All results of previous works are directly from their papers.](../../papers/streamtensor/table-04.png)

**Table 4.** Comparison with previous works on GPT-2 model. *TTFT* measures the time to first token in ms, the lower the better. *Speed* measures the decoding speed in token/s, the higher the better. All results of previous works are directly from their papers.

<span id="table-05"></span>

![Table 5. Comparison with NVIDIA GPUs on GPT-2 model. *TTFT* measures the time to first token in ms, the lower the better. *Speed* measures the decoding speed in token/s, the higher the better.](../../papers/streamtensor/table-05.png)

**Table 5.** Comparison with NVIDIA GPUs on GPT-2 model. *TTFT* measures the time to first token in ms, the lower the better. *Speed* measures the decoding speed in token/s, the higher the better.

<span id="section-5-3-5"></span>

#### 5.3.5 Dynamic Behaviors

StreamTensor uses different approaches to manage dynamic behaviors within dataflow accelerators:

1. **Control flow**: StreamTensor leverages Torch-MLIR [Tor21] as its front-end. Torch-MLIR can infer the static tensor shapes as much as possible from inputs, eliminating `if`s and unrolling `for`s associated with static tensor shapes. If the control flow relies on runtime values, the corresponding subgraph will fall back to naive PyTorch execution [Ans24a] on the host.
2. **Path joining**: This often arises in the presence of control flow, particularly when a dataflow kernel is reused with inputs from different sources. By eliminating control flows, Torch-MLIR resolves the corresponding path joining problems.
3. **Dynamic tensor shape**: Tensors with dynamic shapes, like input tokens and KV-caches, require shape hints to define their maximum possible dimension sizes (e.g., maximum sequence length). These hints determine the total number of tokens, $T$, that can be processed between any two dataflow kernels. From these maximum $T$ values, StreamTensor infers $\mathit{max\_tokens}$ based on the method discussed in [Section 5.3](#section-5-3).
4. **FIFO stall**: StreamTensor does not generate a static schedule for the dataflow accelerator. Instead, all dataflow kernels automatically honor their dependencies via FIFO interconnections. As a result, unexpected FIFO stalls caused by runtime events, e.g., external memory traffic, do not require specific handling. Once the event causing the stall resolves, the dataflow accelerator seamlessly resumes operation from the stall point.

<span id="section-6"></span>

## 6 Experiments

To evaluate the performance of dataflow accelerators generated by StreamTensor, we deploy multiple LLMs on AMD U55C FPGA with Vitis 2024.1. As shown in [Figure 4](#figure-04), HLS C++ code is generated by StreamTensor and compiled into bitstreams using Vitis to program the FPGA. [Table 6](#table-06) shows the experimental setup of the platforms evaluated in this section. All experimental results of StreamTensor reported are obtained via *on-board measurement*. All LLM models evaluated on StreamTensor are modified from Huggingface models to accommodate the requirements of Torch-MLIR front-end.

<span id="table-06"></span>

![Table 6. Experiment setup of evaluated platforms.](../../papers/streamtensor/table-06.png)

**Table 6.** Experiment setup of evaluated platforms.

<span id="figure-09"></span>

![Figure 9. Energy efficiency (tokens/J) comparison with NVIDIA GPUs on emerging LLMs.](../../papers/streamtensor/figure-09.png)

**Figure 9.** Energy efficiency (tokens/J) comparison with NVIDIA GPUs on emerging LLMs.

<span id="figure-10"></span>

![Figure 10. Ablation studies on GPT-2 model and emerging LLMs.](../../papers/streamtensor/figure-10.png)

**Figure 10.** Ablation studies on GPT-2 model and emerging LLMs.

<span id="section-6-1"></span>

### 6.1 GPT-2

Most prior works [Che24v, Che24u, Hon22] on FPGAs evaluate their frameworks using GPT-2 [Rad19]. [Table 4](#table-04) shows a comparison between StreamTensor and previous works under different input/output sequence length configurations. For GPT-2, we successfully fuse an entire transformer block onto a single FPGA by inserting layout converters and stream FIFOs, ensuring all intermediate results are communicated on-chip. Subsequently, this single FPGA accelerator is triggered multiple times with different weight parameters to execute all transformer blocks in a sequential manner. StreamTensor achieves 0.76x shorter total latency and 0.40x shorter TTFT than Allo [Che24v, Che24u]. Compared to DFX [Hon22], StreamTensor delivers even greater improvements, e.g., 0.19x TTFT. These gains come from StreamTensor's automated dataflow architecture exploration. In contrast, both Allo and DFX require manual design of all dataflow kernels and components. For example, all the layout converters, DMAs, and FIFOs are manually written and configured, a process that is error-prone and may lead to suboptimal design choices. Note that GPT-2 is the only LLM reported in Allo and DFX due to their limited flexibility and productivity on other emerging LLMs. As shown in [Table 4](#table-04), TTFT scales roughly linearly with input length, demonstrating the design's scalability. We also compare StreamTensor with NVIDIA GPUs in [Table 5](#table-05), where StreamTensor achieves 0.64x and 0.25x shorter total latency compared to A100 and 2080Ti, respectively. We can observe that GPUs outperform StreamTensor by a large margin for the TTFT metric due to their abundant computation resources. However, because the decoding stage of LLM inference is highly memory-bound, the dataflow accelerators generated by StreamTensor can outperform GPUs due to their reduced external memory access, leading to better decoding speed and overall latency.

<span id="table-07"></span>

![Table 7. Configurations of LLMs, collected from their Huggingface model cards Gpt19, Qwe24a, Lla24a, Gem25b.](../../papers/streamtensor/table-07.png)

**Table 7.** Configurations of LLMs, collected from their Huggingface model cards [Gpt19, Qwe24a, Lla24a, Gem25b].

<span id="section-6-2"></span>

### 6.2 Emerging LLMs

To evaluate the flexibility of StreamTensor, we test it on several emerging LLMs, including Qwen [Bai23b], Llama [Tou23], and Gemma [Gem24b]. Model configurations are shown in [Table 7](#table-07). For all three of these models, we also successfully fuse an entire transformer block onto a single FPGA and execute it in the same manner as GPT-2. From [Figure 9](#figure-09), we observe that StreamTensor can outperform A100 on energy efficiency on Qwen and Gemma models by 1.99x and 1.59x due to the lower power of FPGAs. [Figure 10(a)](#figure-10) shows that the Llama model generates more intermediate results than other models. This leads StreamTensor to adopt a more conservative dataflow FIFO sizing strategy, which, in turn, reduces the execution overlap between dataflow kernels and results in lower performance compared to Qwen and Gemma.

<span id="section-6-2-1"></span>

#### 6.2.1 On-chip Memory Reduction Study

[Figure 10(a)](#figure-10) shows on-chip memory usage before and after kernel fusion across all evaluated LLMs. This study focuses on the *intermediate results* within a single LLM layer. Model parameters are excluded in this study, as they are too large to fit on-chip. Kernel fusion reduces memory usage to just 14.8%–16.8% of the original design. Without fusion, LLMs cannot be deployed in a fully dataflow fashion due to excessive intermediate buffer sizes.

<span id="section-6-2-2"></span>

#### 6.2.2 Compilation Time Study

[Figure 10(b)](#figure-10) shows the breakdown of execution time for generating RTL from PyTorch. The HLS process (generating RTL from C++) consumes the majority of the total time. The downstream tool profiling also accounts for a large portion, since resource allocation decisions depend on accurate profiling results. In comparison, StreamTensor compilation and parameter packing take only a small fraction of the total time. As discussed in [Section 4.2](#section-4-2), StreamTensor automatically packs and widens interfaces to optimize external memory efficiency. As a result, model parameters must be packed accordingly to match the desired memory layout. After packing, binary files are generated and loaded at runtime. In [Figure 10(c)](#figure-10), we further break down StreamTensor’s compilation time based on the stages shown in [Figure 4](#figure-04). Total compilation time ranges from 26.8s to 63.4s in our experiments. High-level stages (from Linalg optimization to resource allocation) are relatively fast. In contrast, low-level stages (bufferization, HLS optimization, and code generation) take more time. This validates the efficiency of our high-level `itensor` optimizations.

<span id="section-7"></span>

## 7 Related Works

Pioneering works [Lee87, Bil96, Bha01, Thi02, Neu04] established the foundation of stream-based dataflow modeling and compilation. Later works [Gov02, Ven06, Naj13] explored buffer minimizing and slack matching problems in dataflow networks. [Con14a, Che16i] explored the deadlock analysis and buffer sizing for sequential programs. Note that these papers focused on steady-state scenarios (i.e., the *Conservative* equalization strategy in [Section 5.3.3](#section-5-3-3)), overlooking the trade-off between area and performance. [Guo21b, Du23b] improved the floorplanning and clock frequency for streaming applications on FPGAs. [Jos21, Xu24g] tackled the buffer insertion and placement problem in dynamically scheduled dataflow circuits [Jos18].

Compilers are essential for mapping applications onto spatial architectures like DSAs and FPGAs. SARA [Zha21h] provided a compiler stack for large-scale DSAs like Plasticine [Pra17], translating an imperative DSL with nested control flow, virtualizing resources, and managing memory consistency. The compiler for Revet [Ruc23] mapped its “dataflow threads” abstraction, which supports data-dependent control flow, onto vectorized DSAs [Ruc21] using streaming tensor operations. Works like DSAGEN [Wen20a] synthesized programmable spatial accelerators directly from dataflow graph descriptions. Constraint-based scheduling techniques [Now13] often use ILP for optimal or near-optimal instruction scheduling on spatial platforms. Higher-level programming abstractions are also crucial, such as Sigma [Zha23n], which compiled Einstein summations to dataflow hardware. Targeting FPGAs, Stream-HLS [Bas25] automatically generated optimized HLS-based dataflow architectures from C/C++ or PyTorch. These diverse compilers and frameworks automated critical optimizations. However, they often only enable partial design space exploration, and lack a systematic typing system to enable flexible stream-based kernel fusion and other optimizations. Here, we use Stream-HLS [Bas25] as an example to analyze its differences with StreamTensor:

- Due to the lack of a systematic typing system, Stream-HLS cannot automatically generate DMAs for external memory, limiting its practical usage and scalability on real-world applications.
- Stream-HLS overlooked the FIFO sizing problem, which is essential to avoid deadlocks in dataflow accelerators and scale out to real-world applications.
- Stream-HLS demanded two conditions to enable streaming between dataflow kernels: 1) the number of writes and reads to/from the shared buffer must be equal; and 2) the write order of the producer must match the read order of the consumer. Although both conditions are often difficult to meet, Stream-HLS cannot perform kernel fusion without meeting either of them. In contrast, StreamTensor resolves these two conditions through the `itensor`-based typing system, making any dataflow kernels fuseable by design.
- Due to the reasons above, Stream-HLS did not support the kernel fusion space exploration like StreamTensor, limiting its application on large-scale workloads that cannot be fully deployed on-chip without kernel fusion. For example, Stream-HLS only reports the performance of the multi-head attention layer and feed-forward layer separately, rather than for the entire transformer block.

<span id="section-8"></span>

## 8 Conclusion and Future Works

This paper introduces StreamTensor, a compiler framework that automates the generation and optimization of stream-based dataflow accelerators. StreamTensor's main contributions include an `itensor`-based typing system that forms the foundation of the entire framework, a PyTorch-to-device compilation pipeline, and a set of design spaces for exploring key architectural parameters. By addressing common pitfalls in existing frameworks, StreamTensor effectively improves the efficiency of dataflow accelerators. As the demand for efficient AI continues to grow, StreamTensor paves the way for future work in scalable and extensible dataflow compilation.

Looking ahead, StreamTensor's modular design and `itensor` typing system open promising avenues for future work, particularly in extending its compatibility with diverse dataflow architectures and specialized kernel languages. StreamTensor can be adapted to programmable architectures like AMD Versal [Gai19], Sambanova RDU [Pra24], and Groq LPU [Abt22] by retargeting its low-level compilation and code generation stages. This process would map the dataflow kernels, FIFOs, and layout converters in StreamTensor IR into platform-specific components, such as the AI engines and routing networks in AMD Versal. Similarly, StreamTensor can integrate with kernel languages like Allo [Che24v], allowing developers to incorporate manually-optimized kernels as black-box components. In both scenarios, the `itensor` system serves as a crucial abstraction layer, enabling StreamTensor to perform high-level dataflow optimizations, including kernel fusion and dataflow component generation, while interfacing with target-specific back-ends and black-box components. This promises to broaden StreamTensor's applicability by leveraging the unique strengths of various hardware platforms and programming languages.

## Acknowledgments

We thank all anonymous reviewers, especially our anonymous shepherd, for their valuable feedback and suggestions. We thank Vikram Adve, Jian Huang, and Stephen Neuendorffer for their insightful feedback on this work. We thank Kaiwen Cao for collecting the FPGA experimental results during his internship at Inspirit IoT, Inc. We thank Jinghua Wang for collecting the GPU experimental results. We thank AMD for supporting the FPGA boards to Inspirit IoT, Inc., which are used in this work.
