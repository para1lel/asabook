---
title: "现代 GPU 编程"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/
pageClass: gpupro-page
---

::: note 来源
本地版本镜像自 [MLC Community 原书](https://mlc.ai/modern-gpu-programming-for-mlsys/)的[上游提交 `a5ed072f0d35`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/a5ed072f0d35c35722bbe86dec6926baad2aa46a). Copyright 2026 MLC Community. 正文和资源采用上游中文稿, 仅调整了 VuePress 所需的格式, 链接与本站标点规范. 上游仓库目前没有覆盖全仓库的许可证文件; TIRx 参考页面保留其 Apache License 2.0 声明.
:::

机器学习系统支撑着现代 AI 的核心计算任务. 随着模型规模扩大, 部署场景变得更加复杂, 系统性能越来越依赖少数关键 GPU kernel 的实现质量. Attention kernel, LLM prefill 和 decode kernel, 低精度 block-scaled GEMM, 融合 MoE 层, 以及其他大型融合 kernel, 都会直接影响训练和服务的端到端速度.

因此, 要理解和优化现代 AI 系统, 就必须理解高性能 GPU kernel 是如何写出来的. 然而, 高性能 kernel 并不是简单堆叠优化技巧的结果. 现代 GPU 架构已经发生了显著变化: 新的架构引入了更丰富的内存空间, 新的数据搬运机制, 以及越来越专用化的执行单元. 要充分利用这些硬件能力, 我们既需要建立清晰的硬件心智模型, 也需要理解一个高性能 kernel 是如何从基础版本一步步演化出来的. 本书重点关注的正是这两个方面.

基于这一目标, 本书将按照从硬件到代码, 再到高性能 kernel 的顺序展开. 我们会先介绍 GPU 的硬件组织和执行模型, 然后学习本书使用的编程模型, 最后在这些基础上逐步构建先进的 GPU kernel. 具体来说, 本书将以 NVIDIA Blackwell 架构为例, 详细讲解 General Matrix-Matrix Multiplication (GEMM), 以及 FlashAttention. 在这些 kernel 的构建过程中, 我们还会系统学习数据布局, 异步数据搬运, 异步协作等 GPU 优化中的关键主题.

本书内容源自卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程系列. 为了让这些概念可以通过真实代码学习, 运行和验证, 本书使用 TIRx Python DSL 逐步构建 GPU kernel 示例. TIRx 贴近硬件, 并暴露底层执行抽象, 因此读者可以一边运行代码, 一边推理其背后的控制流, 内存访问和同步逻辑.

本书是开源项目, 欢迎通过 [GitHub 仓库](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys) 提交贡献, 修正和示例.


## 本书结构

- **第一部分: 理解 GPU.** 这一部分介绍 GPU 的整体架构组织, 编写高性能 kernel 的通用方法, 以及数据布局, 异步内存操作和协作等关键概念, 并建立后续章节所依赖的硬件理解.
- **第二部分: TIRx 概览.** 这一部分介绍 TIRx 的核心组成部分, 为理解后续章节中的代码示例做准备.
- **第三部分: GEMM: 从 Tiled 到 SOTA.** 这一部分完整讲解如何优化一个 tiled GEMM, 并逐步加入 TMA pipelining, persistent scheduling, warp specialization 和 2-CTA cluster.
- **第四部分: Flash Attention 4.** 这一部分基于第三部分的技术构建完整的 attention kernel: 两个 MMA, 中间插入 softmax, 并包含 online-softmax rescaling, causal masking 和 GQA.
- **参考资料.** TIRx 语言参考, 编译器内部机制, 以及异步 kernel 调试指南.

## 目录

### 第一部分: 理解 GPU

- [GPU 执行模型](./gpu-execution-model/)
- [高性能 Kernel 的关键](./kernel-performance/)
- [数据布局及其表示方法](./data-layout/)
- [Tensor Core 数据布局的演进](./tensor-core-data-layouts/)
- [异步数据搬运: TMA](./tma/)
- [Blackwell Tensor Core: tcgen05.mma](./blackwell-tensor-core/)
- [Tensor Memory (TMEM)](./tensor-memory/)
- [异步协作: mbarrier](./mbarrier/)
- [高级调度: Cluster Launch Control](./cluster-launch-control/)

### 第二部分: TIRx 概览

- [TIRx 简介](./tirx-introduction/)
- [TIRx Layout API](./tirx-layout-api/)

### 第三部分: GEMM, 从 Tiled 到 SOTA

- [构建 Tiled GEMM](./tiled-gemm/)
- [使用 TMA 流水线化 GEMM](./pipelined-gemm/)
- [使用 Warp Specialization 和 Cluster 扩展 GEMM](./warp-specialized-gemm/)

### 第四部分: Flash Attention 4

- [Flash Attention 4](./flash-attention-4/)

### 参考资料

- [参考资料](./reference/)
- [TIRx 语言参考](./tirx-language-reference/)
- [调试 Warp-Specialized Kernel](./debugging-warp-specialized-kernels/)
- [编译器内部机制](./compiler-internals/)
