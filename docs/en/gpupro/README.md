---
title: "Modern GPU Programming For MLSys"
createTime: 2026/08/01 00:00:00
permalink: /en/gpupro/
pageClass: gpupro-page
---

::: note Source
This local edition mirrors the [MLC Community book](https://mlc.ai/modern-gpu-programming-for-mlsys/) from [upstream commit `a5ed072`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/a5ed072f0d35c35722bbe86dec6926baad2aa46a). Copyright 2026 MLC Community. The text and assets are unchanged except for the formatting and links required by VuePress.
:::

Machine learning systems sit at the heart of modern AI workloads. In these systems, performance
often comes down to the quality of a small number of GPU kernels. Attention kernels, LLM prefill
and decode kernels, low-precision block-scaled GEMMs, fused MoE layers, and other large fused
kernels all directly shape end-to-end speed in both training and serving.

To make these kernels fast, however, we need more than a list of optimization tricks. Modern GPUs
are no longer simple variations of the same old design. Recent architectures introduce richer
memory spaces, new access patterns, and increasingly specialized execution units. To program them
well, we need both a clear mental model of the hardware and a practical understanding of how
high-performance kernels are built. This book is about developing both.

The book follows a simple progression: first understand the GPU hardware, then learn the
programming model we will use, and finally build state-of-the-art kernels step by step. Our main
target is the Blackwell generation, and our main running examples are General Matrix-Matrix
Multiplication (GEMM) and FlashAttention. Along the way, we will also study the core ingredients
behind GPU optimization: data layout, asynchronous data movement, and asynchronous coordination.

The material grows out of the [Machine Learning Systems](https://mlsyscourse.org/) course series
at Carnegie Mellon University. To make the ideas easier to study and easier to run, this book uses
the **TIRx** Python DSL to build real GPU kernel examples step by step. TIRx stays close to the
hardware, which lets us reason about low-level control while still learning through runnable code.

This book is open source. Contributions, corrections, and examples are welcome through the
[GitHub repository](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys).

## How This Book Is Organized

- **Part I, Understanding the GPU.** This part introduces the overall organization of the GPU,
  general recipes for writing fast kernels, and key concepts such as data layout, asynchronous
  memory operations, and coordination. It builds the hardware intuition that the rest of the book
  relies on.
- **Part II, TIRx Overview.** This part introduces the key elements of TIRx, which serve as the
  foundation for the code examples throughout the book.
- **Part III, GEMM: Tiled to SOTA.** A complete guide to optimizing a tiled GEMM, built up through
  TMA pipelining, persistent scheduling, warp specialization, and 2-CTA clusters.
- **Part IV, Flash Attention 4.** A complete attention kernel built from the Part III techniques:
  two MMAs with softmax between them, online-softmax rescaling, causal masking, and GQA.
- **Reference.** TIRx language reference and compiler internals.

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
