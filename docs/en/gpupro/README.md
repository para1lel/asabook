---
title: "Modern GPU Programming For MLSys"
createTime: 2026/08/01 00:00:00
permalink: /en/gpupro/
pageClass: gpupro-page
---

::: note Source
This local edition mirrors the [MLC Community book](https://mlc.ai/modern-gpu-programming-for-mlsys/) from [upstream commit `d289e0a79218`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/d289e0a7921852b620115e04136ba2ac6495ac33). Copyright 2026 MLC Community. The text and assets are unchanged except for the formatting and links required by VuePress. The upstream repository currently has no repository-wide license file; the TIRx reference pages retain their Apache License 2.0 notices.
:::

Machine learning systems power many of today's AI workloads. As models grow and deployment
settings become more complex, end-to-end performance increasingly depends on a small number of
critical GPU kernels. Attention, LLM prefill and decode, low-precision block-scaled GEMM, fused MoE
layers, and other large fused kernels directly affect both training and serving speed.

Making these kernels fast requires more than a list of optimization tricks. Recent GPU
architectures introduce richer memory spaces, new data-movement mechanisms, and increasingly
specialized execution units. Using them effectively requires both a clear understanding of how the
hardware executes a program and practical knowledge of how a basic kernel evolves into a
high-performance implementation. This book develops both.

The book proceeds from hardware to programming model to complete kernels. It first introduces GPU
organization and execution, then presents the programming model used throughout the book, and
finally builds high-performance kernels step by step. The main target is NVIDIA Blackwell, and the
running examples are General Matrix-Matrix Multiplication (GEMM) and FlashAttention. Along the way,
the book develops the key ideas behind GPU optimization: data layout, asynchronous data movement,
and asynchronous coordination.

The material grows out of the [Machine Learning Systems](https://mlsyscourse.org/) course series at
Carnegie Mellon University. The examples use the **TIRx** Python DSL so that the ideas can be
studied, run, and verified in real kernels. TIRx keeps hardware-level choices explicit, making it
possible to reason about control flow, memory access, and synchronization while working with
runnable code.

This book is open source. Contributions, corrections, and examples are welcome through the
[GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## How This Book Is Organized

- **Part I, Understanding the GPU.** This part introduces the overall organization of the GPU,
  general techniques for writing fast kernels, and key concepts such as data layout, asynchronous
  memory operations, and coordination. It builds the hardware intuition that the rest of the book
  relies on.
- **Part II, TIRx Overview.** This part introduces the key elements of TIRx, which serve as the
  foundation for the code examples throughout the book.
- **Part III, GEMM: Tiled to SOTA.** A complete guide to optimizing a tiled GEMM, built up through
  TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV, Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference, compiler internals, and a guide to debugging asynchronous
  kernels.

## Contents

### Part I: Understanding the GPU

- [GPU Execution Model](./gpu-execution-model/)
- [What Makes a Kernel Fast](./kernel-performance/)
- [Data Layout and Its Notation](./data-layout/)
- [The Evolution of Tensor Core Data Layouts](./tensor-core-data-layouts/)
- [Async Data Movement: TMA](./tma/)
- [Blackwell Tensor Core: `tcgen05.mma`](./blackwell-tensor-core/)
- [Tensor Memory (TMEM)](./tensor-memory/)
- [Async Coordination: mbarrier](./mbarrier/)
- [Advanced Scheduling: Cluster Launch Control](./cluster-launch-control/)

### Part II: TIRx Overview

- [Introduction to TIRx](./tirx-introduction/)
- [TIRx Layout API](./tirx-layout-api/)

### Part III: GEMM, Tiled to SOTA

- [Building a Tiled GEMM](./tiled-gemm/)
- [Pipelining GEMM with TMA](./pipelined-gemm/)
- [Scaling GEMM with Warp Specialization and Clusters](./warp-specialized-gemm/)

### Part IV: Flash Attention 4

- [Flash Attention 4](./flash-attention-4/)

### Reference

- [Reference Overview](./reference/)
- [TIRx Language Reference](./tirx-language-reference/)
- [Debugging Warp-Specialized Kernels](./debugging-warp-specialized-kernels/)
- [Compiler Internals](./compiler-internals/)
