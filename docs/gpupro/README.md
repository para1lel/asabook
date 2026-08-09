---
title: "面向机器学习系统的现代 GPU 编程"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/
pageClass: gpupro-page
---

::: note 来源
本地版本镜像自 [MLC Community 原书](https://mlc.ai/modern-gpu-programming-for-mlsys/)的[上游提交 `d289e0a79218`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/d289e0a7921852b620115e04136ba2ac6495ac33). Copyright 2026 MLC Community. 正文和资源以上游中文稿为基础, 本地版本调整了 VuePress 所需的格式和链接, 并校订中文表达. 上游仓库目前没有覆盖全仓库的许可证文件; TIRx 参考页面保留 Apache License 2.0 声明.
:::

机器学习系统承担着现代 AI 中许多核心计算任务. 随着模型规模扩大、部署场景变得更复杂, 端到端性能越来越依赖少数关键 GPU kernel 的实现质量. Attention, LLM prefill 和 decode, 低精度 block-scaled GEMM, 融合 MoE 层以及其他大型融合 kernel, 都会直接影响训练和服务的速度.

要让这些 kernel 真正跑得快, 不能只罗列优化技巧. 近年来的 GPU 架构引入了更多内存空间、新的数据搬运机制和更专用的执行单元. 要用好这些硬件, 既要理解 GPU 如何执行程序, 也要掌握基础 kernel 如何一步步演变成高性能实现. 本书围绕这两个方面展开.

本书按硬件、编程模型到完整 kernel 的顺序展开. 我们先介绍 GPU 的组织方式和执行模型, 再学习本书使用的编程模型, 最后逐步构建高性能 kernel. 全书主要面向 NVIDIA Blackwell, 以 General Matrix-Matrix Multiplication (GEMM) 和 FlashAttention 为贯穿始终的示例, 并在构建这些 kernel 的过程中介绍数据布局、异步数据搬运和异步协作.

本书内容源自卡内基梅隆大学的 [Machine Learning Systems](https://mlsyscourse.org/) 课程. 书中的示例使用 TIRx Python DSL, 读者可以通过真实 kernel 学习、运行和验证这些概念. TIRx 会明确表示与硬件执行有关的选择, 便于结合可运行的代码分析控制流、内存访问和同步逻辑.

本书是开源项目, 欢迎在 [GitHub 仓库](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys) 提交代码、勘误和示例.


## 本书结构

- **第一部分: 理解 GPU.** 介绍 GPU 的整体架构、编写高性能 kernel 的通用方法, 以及数据布局、异步内存操作和协作等概念, 为后续章节建立硬件基础.
- **第二部分: TIRx 概览.** 介绍 TIRx 的核心组成, 为后续章节中的代码示例做准备.
- **第三部分: GEMM: 从 Tiled 到 SOTA.** 逐步优化一个 tiled GEMM, 加入 TMA pipelining、persistent scheduling、warp specialization 和 2-CTA cluster.
- **第四部分: Flash Attention 4.** 基于前面的技术构建完整的 attention kernel: 两个 MMA 之间插入 softmax, 并处理 online-softmax rescaling、causal mask 和 GQA.
- **参考资料.** TIRx 语言参考, 编译器内部机制, 以及异步 kernel 调试指南.

## 目录

### 第一部分: 理解 GPU

- [GPU 执行模型](./gpu-execution-model/)
- [高性能 Kernel 的关键](./kernel-performance/)
- [数据布局及其表示方法](./data-layout/)
- [Tensor Core 数据布局的演进](./tensor-core-data-layouts/)
- [异步数据搬运: TMA](./tma/)
- [Blackwell Tensor Core: `tcgen05.mma`](./blackwell-tensor-core/)
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

- [参考资料概览](./reference/)
- [TIRx 语言参考](./tirx-language-reference/)
- [调试 Warp-Specialized Kernel](./debugging-warp-specialized-kernels/)
- [编译器内部机制](./compiler-internals/)
