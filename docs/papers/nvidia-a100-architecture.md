---
title: 'NVIDIA A100 Tensor Core GPU Architecture'
createTime: 2026/09/13 18:00:00
permalink: /papers/nvidia-a100-architecture/
pageClass: paper-reading
---

> [NVIDIA Corporation](https://www.nvidia.com/en-us/about-nvidia/). *[NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)*, V1.0, PDF 版本日期为 2020 年 8 月 26 日. <a href="/paper/nvidia-a100-architecture.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源文件; 准确措辞、印刷版式和参考文献均以已发布的 PDF 为准.

<span id="section-1"></span>

## 1 引言

现代云数据中心运行着多种计算密集型应用, 推动了 NVIDIA GPU 加速云计算的迅猛发展. 这类应用包括 AI 深度学习训练与推理、数据分析、科学计算、基因组学、边缘视频分析与 5G 服务、图形渲染、云游戏等. 无论是扩展 AI 训练与科学计算的规模、横向扩展推理应用, 还是实现实时对话式 AI, NVIDIA GPU 都能提供所需算力, 加速当今云数据中心内众多复杂且难以预测的工作负载.

NVIDIA® GPU 是推动 AI 革命的主要计算引擎, 可显著加速 AI 训练与推理工作负载. NVIDIA GPU 还可加速多种 HPC 和数据分析应用与系统, 帮助客户有效分析和可视化数据, 并从中获得洞察. NVIDIA 加速计算平台已成为许多重要且快速增长行业的核心基础设施.

HPC 早已不局限于在超级计算机上运行天气预报、油气勘探和金融建模等计算密集型应用. 如今, 数百万块 NVIDIA GPU 正在加速云数据中心、服务器、边缘系统乃至桌边工作站中的各类 HPC 应用, 服务于数百个行业和科学领域.

AI 网络在规模、复杂度和多样性上持续增长, 基于 AI 的应用与服务也在迅速普及. NVIDIA GPU 加速的 AI 系统和应用包括: 深度学习推荐系统、自动驾驶汽车和工厂机器人等自主机器、对话式 AI 和实时语言翻译等自然语言处理、智慧城市视频分析、可在边缘提供 AI 服务的软件定义 5G 网络、分子模拟、无人机控制、医学图像分析等.

<span id="figure-01"></span>

![现代云数据中心的工作负载需要 NVIDIA GPU 加速](./nvidia-a100-architecture/figure-01.png)

**图 1.** 现代云数据中心的工作负载需要 NVIDIA GPU 加速

2017 年, NVIDIA Tesla® V100 GPU 引入了强大的新型“Tensor Core”, 大幅加速深度学习神经网络训练和推理核心环节中的矩阵计算. 2018 年, 采用 NVIDIA Turing™ Tensor Core 的 NVIDIA Tesla® T4 GPU 配合 TensorRT™ 推理优化器和运行时, 以高能效显著提升了数据中心推理速度. Turing Tensor Core 还为采用 Turing GPU 的 GeForce® 游戏 PC 和 Quadro® 工作站带来了出色的 AI 能力.

在业界标准 MLPerf AI 基准测试中, NVIDIA Volta™ GPU 在训练类别中胜出, Turing GPU 则在新推出的 MLPerf 推理基准的数据中心和边缘类别中夺得第一. NVIDIA Jetson AGX Xavier™ 也在所有商用 SoC 设备中取得了最佳推理性能.

十多年来, NVIDIA CUDA® 开发平台不断释放 GPU 的能力, 加速广泛的应用领域. API、软件栈、库和代码优化器的创新与改进, 与 GPU 硬件的进步同样重要. [NVIDIA CUDA 工具包](https://developer.nvidia.com/cuda-toolkit) 为开发者提供多种软件工具, 其中包括面向 AI、HPC 和数据分析的 NVIDIA [CUDA-X™ GPU 加速库](https://developer.nvidia.com/gpu-accelerated-libraries). [NVIDIA GPU Cloud™ (NGC)](https://ngc.nvidia.com/catalog/all) 还免费提供许多 AI 框架和 HPC 应用容器, 其中含有模型与脚本, 可简化编程并加快 GPU 加速应用的开发和部署. [NVIDIA GPU 上的 Kubernetes](https://developer.nvidia.com/kubernetes-gpu) 同样免费提供, 使企业能够在多云 GPU 集群上顺畅地纵向和横向扩展训练与推理部署.

<span id="section-2"></span>

## 2 NVIDIA A100 Tensor Core GPU: 面向弹性计算时代的第八代数据中心 GPU

全新的 NVIDIA® A100 Tensor Core GPU 延续了上一代 NVIDIA Tesla V100 GPU 的能力, 在增加众多新特性的同时, 显著提升 HPC、AI 和数据分析工作负载的性能. A100 由基于 NVIDIA Ampere 架构的 GA100 GPU 驱动, 能够有力扩展运行于单 GPU 和多 GPU 工作站、服务器、集群、云数据中心、边缘系统及超级计算机上的 GPU 计算与深度学习应用. A100 GPU 可用于构建弹性、通用且高吞吐量的数据中心.

A100 GPU 引入了全新的“多实例 GPU” (Multi-Instance GPU, MIG) 虚拟化与 GPU 分区功能, 尤其适合云服务提供商 (CSP). 配置为 MIG 模式后, CSP 可以提高 GPU 服务器的利用率, 无需增加成本便可提供最多 7 倍的 GPU 实例. 可靠的故障隔离让客户能够安全地划分单块 A100 GPU.

A100 新增强大的第三代 Tensor Core, 在提高 V100 吞吐量的同时全面支持 DL 和 HPC 数据类型, 并借助新的稀疏性功能进一步将吞吐量翻倍.

A100 中全新的 TensorFloat-32 (TF32) Tensor Core 运算为加速 DL 框架和 HPC 中的 FP32 输入/输出数据提供了一条简便路径, 速度比 V100 的 FP32 FMA 运算快 10 倍, 加入稀疏性后可快 20 倍. 对于 FP16/FP32 混合精度 DL, A100 Tensor Core 的性能是 V100 的 2.5 倍, 加入稀疏性后提升至 5 倍.

全新的 Bfloat16 (BF16)/FP32 混合精度 Tensor Core 运算速度与 FP16/FP32 混合精度相同. Tensor Core 对 INT8、INT4 和二进制运算的加速完善了对 DL 推理的支持, 其中 A100 稀疏 INT8 的速度比 V100 INT8 快 20 倍. 对 HPC 而言, A100 Tensor Core 新增符合 IEEE 标准的 FP64 处理能力, FP64 性能达到 V100 的 2.5 倍.

<span id="figure-02"></span>

![NVIDIA A100 中的新技术](./nvidia-a100-architecture/figure-02.png)

**图 2.** NVIDIA A100 中的新技术

A100 GPU 面向广泛的性能扩展需求而设计. 客户既可使用 MIG GPU 分区技术共享一块 A100, 也可在强大的 NVIDIA DGX™、NVIDIA HGX™ 和 NVIDIA EGX™ 系统中, 通过全新的第三代 NVIDIA NVLink® 高速互连连接多块 A100 GPU. 基于 A100 的系统可借助新的 NVIDIA NVSwitch™ 以及 Mellanox® 先进的 InfiniBand™ 和以太网解决方案, 在计算集群、云实例或大型超级计算机中横向扩展至数十、数百乃至数千块 A100, 加速多种应用与工作负载. A100 GPU 的全新硬件能力还得到 CUDA 11 新特性的配合, 从而改善可编程性, 降低 AI 和 HPC 软件的复杂度.

NVIDIA A100 GPU 是首款弹性 GPU 架构: 它既能利用 NVLink、NVSwitch 和 InfiniBand 纵向扩展为巨型 GPU, 也能通过 MIG 横向扩展以支持多个独立用户, 同时取得高性能和最低的单 GPU 实例成本.

NVIDIA A100 Tensor Core GPU 实现了 NVIDIA GPU 加速计算迄今最大的一次代际飞跃.

<span id="section-3"></span>

## 3 NVIDIA A100 Tensor Core GPU 概览

<span id="section-3-1"></span>

### 3.1 下一代数据中心与云 GPU

日益复杂多样的 AI、HPC 和数据分析工作负载, 需要更强的 GPU 计算能力、更完善的多 GPU 连接能力, 以及一套全面的软件栈. NVIDIA 将基于 NVIDIA Ampere GPU 架构的新款 NVIDIA A100 Tensor Core GPU 与 CUDA 软件的新进展结合起来, 应对这些不断增长的 GPU 计算需求.

本文随后将说明 A100 GPU 的多项核心架构增强, 与 V100 相比, 这些增强可显著加速 AI、HPC 和数据分析工作负载. 新的稀疏性功能还能使数学运算速度再提高最多 2 倍. 高带宽 HBM2 内存以及容量更大、速度更快的缓存, 可为数量增加的 CUDA Core 和 Tensor Core 输送数据.

全新的第三代 NVLink 和 PCIe Gen 4 可加速多 GPU 系统配置. 其他多项增强则为超大规模数据中心带来强扩展能力, 并为云服务提供商 (CSP) 的系统及其客户提供可靠的多实例 GPU (MIG) 虚拟化. NVIDIA Ampere 架构还简化了编程, 同时降低延迟及 AI 和 HPC 软件的复杂度. 在提供所有这些新特性的同时, NVIDIA Ampere 架构 GPU 的每瓦性能也高于上一代 NVIDIA Volta GPU.

NVIDIA A100 GPU 的架构不仅能加速大型复杂工作负载, 也能高效加速许多较小的工作负载. A100 可用于构建能够应对不可预测需求的数据中心, 同时提供细粒度的工作负载配置、更高的 GPU 利用率和更优的总体拥有成本 (TCO).

<span id="figure-03"></span>

![采用新型 SXM4 模块的 NVIDIA A100 GPU](./nvidia-a100-architecture/figure-03.png)

**图 3.** 采用新型 SXM4 模块的 NVIDIA A100 GPU

A100 的通用性有助于基础设施管理员充分发挥数据中心内每块 GPU 的价值, 满足从最小作业到最大多节点工作负载的不同性能需求. A100 驱动的 NVIDIA 数据中心平台还包括 Mellanox HDR InfiniBand (IB)、NVSwitch、HGX A100 和用于纵向扩展的 Magnum IO SDK. 这些集成技术可高效扩展至数万块 GPU, 以前所未有的速度训练最复杂的 AI 网络.

要在企业和云环境中普及加速计算, 必须让小型工作负载也能取得较高的资源利用率. 借助新的多实例 GPU 技术, 每块 A100 最多可划分为七个 GPU 实例, 以优化利用率, 让每个用户和应用都能使用 GPU.

<span id="section-3-2"></span>

### 3.2 AI、HPC 和数据分析的业界领先性能

如 [图 4](#figure-04) 所示, NVIDIA A100 GPU 相比 V100 可显著加速 AI 训练和推理工作负载. 同样, [图 5](#figure-05) 展示了不同 HPC 应用取得的大幅性能提升.

<span id="figure-04"></span>

![面向 BERT-LARGE 训练和推理的统一 AI 加速](./nvidia-a100-architecture/figure-04.png)

**图 4.** 面向 BERT-LARGE 训练和推理的统一 AI 加速

<span id="figure-05"></span>

![A100 GPU 上 HPC 应用相较 NVIDIA Tesla V100 的加速比](./nvidia-a100-architecture/figure-05.png)

**图 5.** A100 GPU 上 HPC 应用相较 NVIDIA Tesla V100 的加速比

<span id="section-3-3"></span>

### 3.3 A100 GPU 主要特性概览

NVIDIA A100 Tensor Core GPU 是面向计算密集型 AI、HPC 和数据分析应用而设计的高速云与数据中心 GPU 加速器.

驱动 A100 的 GA100 GPU 以 TSMC 7nm N7 制程制造, 基于 NVIDIA Ampere 架构, 集成 542 亿个晶体管, 芯片面积为 826 mm².

以下概要列出 A100 的主要特性, 便于快速了解 A100 的重要新技术和性能水平. 后续章节将详细介绍其架构.

<span id="section-3-3-1"></span>

#### 3.3.1 A100 GPU 流式多处理器 (SM)

基于 NVIDIA Ampere 架构的 A100 Tensor Core GPU 采用新型 SM, 它以 Volta 和 Turing SM 架构引入的特性为基础, 显著提升性能并增加多项新能力.

A100 第三代 Tensor Core 改进了操作数共享并提高效率, 还新增了多种强大的数据类型, 包括:

- 用于加速 FP32 数据处理的 TF32 Tensor Core 指令
- 面向 HPC、符合 IEEE 标准的 FP64 Tensor Core 指令
- 吞吐量与 FP16 相同的 BF16 Tensor Core 指令

<span id="table-01"></span>

![NVIDIA A100 Tensor Core GPU 性能规格](./nvidia-a100-architecture/table-01.png)

**表 1.** NVIDIA A100 Tensor Core GPU 性能规格

A100 Tensor Core 新增的稀疏性支持可利用深度学习网络中的细粒度结构化稀疏性, 使 Tensor Core 运算吞吐量翻倍. 下文 [第 4.3 节](#section-4-3) 将详细介绍稀疏性功能.

A100 配备容量更大、速度更快的 L1 缓存与共享内存单元, 每个 SM 的总容量是 V100 的 1.5 倍 (每个 SM 192 KB, V100 为 128 KB), 可进一步加速多种 HPC 和 AI 工作负载.

其他多项 SM 新特性也改善了可编程性, 并降低软件复杂度.

<span id="section-3-3-2"></span>

#### 3.3.2 40 GB HBM2 与 40 MB L2 缓存

为了满足庞大的计算吞吐需求, NVIDIA A100 GPU 配备 40 GB 高速 HBM2 内存, 内存带宽达 1555 GB/s, 比 Tesla V100 提高 73%. A100 GPU 的片上内存也大幅增加, 其中包括 40 MB 二级 (L2) 缓存, 容量接近 V100 的 7 倍, 可充分发挥计算性能. 新的分区式交叉开关结构使 A100 L2 缓存的读取带宽达到 V100 的 2.3 倍.

为了优化容量利用率, NVIDIA Ampere 架构提供 L2 缓存驻留控制, 供开发者管理应保留在缓存中或从中逐出的数据. A100 还新增计算数据压缩, 最多可使 DRAM 和 L2 带宽再提高 4 倍, L2 容量提高 2 倍.

<span id="section-3-3-3"></span>

#### 3.3.3 多实例 GPU (MIG)

新的多实例 GPU (MIG) 功能可将 A100 Tensor Core GPU 安全地划分为最多七个独立的 GPU 实例, 供 CUDA 应用使用, 让多个用户分别利用独立的 GPU 资源加速应用和开发项目.

使用 MIG 时, 每个实例的处理器在整个内存系统中都有独立且隔离的路径: 片上交叉开关端口、L2 缓存组、内存控制器和 DRAM 地址总线都单独分配给一个实例. 因此, 即便其他任务频繁置换各自的缓存或占满 DRAM 接口, 单个用户的工作负载仍能以可预测的吞吐量和延迟运行, 并获得固定的 L2 缓存配额和 DRAM 带宽.

MIG 在提高 GPU 硬件利用率的同时, 可在不同客户 (如 VM、容器和进程) 之间提供明确的 QoS 与隔离. MIG 特别适合有多租户场景的云服务提供商, 可确保一个客户不会影响其他客户的工作或调度, 同时加强安全性并允许向客户保证 GPU 利用率.

<span id="section-3-3-4"></span>

#### 3.3.4 第三代 NVLink

A100 GPU 与新款 NVSwitch 采用第三代 NVIDIA 高速 NVLink 互连, 可显著增强多 GPU 扩展性、性能和可靠性. 由于每块 GPU 和每个交换机配备更多链路, 新一代 NVLink 可提供高得多的 GPU-GPU 通信带宽, 并改进错误检测与恢复功能.

第三代 NVLink 每对信号线的数据速率为 50 Gbit/s, 接近 V100 所用 25.78 Gbit/s 的两倍. 单条 A100 NVLink 在每个方向上都提供与 V100 相近的 25 GB/s 带宽, 但每条链路使用的信号线对数量只有 V100 的一半. A100 的链路总数从 V100 的 6 条增至 12 条, 因而总带宽从 300 GB/s 提升至 600 GB/s.

<span id="section-3-3-5"></span>

#### 3.3.5 支持 NVIDIA Magnum IO™ 和 Mellanox 互连解决方案

NVIDIA A100 Tensor Core GPU 完全兼容 NVIDIA Magnum IO 以及 Mellanox 先进的 InfiniBand 和以太网互连解决方案, 可加速多节点连接. NVIDIA Magnum IO API 集成计算、网络、文件系统和存储, 使多 GPU、多节点加速系统取得最高 IO 性能. 它与 CUDA-X™ 库衔接, 可加速从 AI、数据分析到可视化等多种工作负载的 IO.

<span id="section-3-3-6"></span>

#### 3.3.6 支持 SR-IOV 的 PCIe Gen 4

A100 GPU 支持 PCI Express Gen 4 (PCIe Gen 4), x16 连接带宽为 31.5 GB/s, 相比 PCIe 3.0/3.1 的 15.75 GB/s 翻倍. 更高速度尤其有利于 A100 GPU 连接支持 PCIe 4.0 的 CPU, 也有助于支持 200 Gbit/s InfiniBand 等高速网络接口. A100 还支持单根 I/O 虚拟化 (SR-IOV), 可让多个进程或虚拟机 (VM) 共享并虚拟化一条 PCIe 连接.

<span id="section-3-3-7"></span>

#### 3.3.7 改进错误与故障检测、隔离和遏制

通过检测、遏制并经常纠正错误和故障来最大限度提高 GPU 的运行时间与可用性, 而不是强制重置 GPU, 这一点在大型多 GPU 集群以及 MIG 配置等单 GPU 多租户环境中尤为重要. NVIDIA A100 Tensor Core GPU 包含多项新技术, 用于改进错误或故障的归因、隔离和遏制, 下文 [第 4 节](#section-4) 将作详细说明.

<span id="section-3-3-8"></span>

#### 3.3.8 异步复制

A100 GPU 新增异步复制指令, 可将数据从全局内存直接加载至 SM 共享内存, 无需中间寄存器文件 (RF). 异步复制可降低寄存器文件带宽需求, 更高效地利用内存带宽, 并减少功耗. 顾名思义, SM 执行其他计算时, 异步复制可在后台进行.

<span id="section-3-3-9"></span>

#### 3.3.9 异步屏障

A100 GPU 在共享内存中提供硬件加速屏障. CUDA 11 以符合 ISO C++ 标准的屏障对象形式提供这些屏障. 异步屏障将到达与等待操作分开, 可使从全局内存到共享内存的异步复制和 SM 中的计算重叠执行. 它们可用 CUDA 线程实现生产者-消费者模型. 屏障还能以不同粒度同步 CUDA 线程, 不再局限于线程束或线程块级别.

<span id="section-3-3-10"></span>

#### 3.3.10 任务图加速

CUDA 任务图提供了一种更高效的 GPU 工作提交模型. 任务图由内存复制、内核启动等一系列通过依赖关系连接的操作组成. 任务图支持“一次定义、重复运行”的执行流程. 预定义任务图能够通过一次操作启动任意数量的内核, 大幅提高应用效率和性能. A100 新增硬件特性, 可显著加快任务图中网格之间的路径.

<span id="section-4"></span>

## 4 NVIDIA A100 Tensor Core GPU 架构详解

基于 NVIDIA Ampere 架构的 NVIDIA A100 GPU 通过多项全新架构特性与优化, 力求提供尽可能强的 AI 和 HPC 算力. A100 采用 TSMC 7nm N7 FinFET 制程制造; 与 Tesla V100 所用的 12nm FFN 制程相比, 该制程具有更高的晶体管密度、更好的性能和能效. 新的多实例 GPU (MIG) 功能可在多租户和虚拟化 GPU 环境中加强客户与应用的故障隔离和 QoS, 对云服务提供商尤其有用. 更快且更具容错能力的第三代 NVIDIA NVLink 互连, 可改善超大规模数据中心的多 GPU 性能扩展.

NVIDIA GA100 GPU 由多个 GPU 处理集群 (GPC)、纹理处理集群 (TPC)、流式多处理器 (SM) 和 HBM2 内存控制器组成.

完整实现的 GA100 GPU 包含以下单元:

- 8 个 GPC, 每个 GPC 8 个 TPC, 每个 TPC 2 个 SM, 每个 GPC 16 个 SM, 完整 GPU 共 128 个 SM
- 每个 SM 64 个 FP32 CUDA Core, 完整 GPU 共 8192 个 FP32 CUDA Core
- 每个 SM 4 个第三代 Tensor Core, 完整 GPU 共 512 个第三代 Tensor Core
- 6 组 HBM2 堆栈和 12 个 512 位内存控制器. NVIDIA A100 Tensor Core GPU 中实现的 GA100 GPU 包含以下单元:
- 7 个 GPC, 每个 GPC 7 或 8 个 TPC, 每个 TPC 2 个 SM, 每个 GPC 最多 16 个 SM, 共 108 个 SM
- 每个 SM 64 个 FP32 CUDA Core, 每块 GPU 共 6912 个 FP32 CUDA Core
- 每个 SM 4 个第三代 Tensor Core, 每块 GPU 共 432 个第三代 Tensor Core
- 5 组 HBM2 堆栈和 10 个 512 位内存控制器. 用于制造 GA100 GPU 的 TSMC 7nm N7 制程, 使尺寸与 Volta GV100 GPU (采用 TSMC 12nm FFN 制程) 相近的芯片可以容纳更多 GPC、TPC 和 SM 单元, 以及许多其他新硬件特性.

[图 6](#figure-06) 展示了配备 128 个 SM 的完整 GA100 GPU. A100 基于 GA100, 配备 108 个 SM.

<span id="figure-06"></span>

![配备 128 个 SM 的完整 GA100 GPU (A100 Tensor Core GPU 配备 108 个 SM)](./nvidia-a100-architecture/figure-06.png)

**图 6.** 配备 128 个 SM 的完整 GA100 GPU (A100 Tensor Core GPU 配备 108 个 SM)

<span id="section-4-1"></span>

### 4.1 A100 SM 架构

新款 A100 SM 以 Volta 和 Turing SM 架构中引入的特性为基础, 显著提升性能, 并新增多项能力与改进.

[图 7](#figure-07) 展示了 A100 SM. Volta 和 Turing 的每个 SM 有八个 Tensor Core, 每个 Tensor Core 每时钟可执行 64 次 FP16/FP32 混合精度融合乘加 (FMA) 运算. A100 SM 采用全新的第三代 Tensor Core, 每个核心每时钟可执行 256 次 FP16/FP32 FMA 运算. A100 的每个 SM 有四个 Tensor Core, 合计每时钟可执行 1024 次稠密 FP16/FP32 FMA 运算, 单个 SM 的算力是 Volta 和 Turing 的 2 倍.

下面简要列出 SM 的主要特性 (后续章节会详细说明):

- 第三代 Tensor Core:
  - 加速 FP16、BF16、TF32、FP64、INT8、INT4 和二进制等所有数据类型.
  - 新的 Tensor Core 稀疏性功能利用深度学习网络中的细粒度结构化稀疏性, 使标准 Tensor Core 运算性能翻倍.
  - A100 的 TF32 Tensor Core 运算为加速 DL 框架和 HPC 中的 FP32 输入/输出数据提供简便路径, 速度比 V100 FP32 FMA 运算快 10 倍, 加入稀疏性后快 20 倍.
  - FP16/FP32 混合精度 Tensor Core 运算为 DL 提供强大的处理能力, 速度比 Tesla V100 Tensor Core 运算快 2.5 倍, 加入稀疏性后快 5 倍.
  - BF16/FP32 混合精度 Tensor Core 运算速度与 FP16/FP32 混合精度相同.
  - FP64 Tensor Core 运算为 HPC 提供强大的双精度处理能力, 速度比 V100 FP64 DFMA 运算快 2.5 倍.
  - 采用稀疏性的 INT8 Tensor Core 运算为 DL 推理提供强大的处理能力, 速度最高可达 V100 INT8 运算的 20 倍.
- 共享内存与 L1 数据缓存合计 192 KB, 比 V100 SM 大 1.5 倍
- 新的异步复制指令可将数据从全局内存直接加载至共享内存, 可选择绕过 L1 缓存, 且无需使用中间寄存器文件 (RF)
- 配合新异步复制指令使用、基于共享内存的新型屏障单元 (异步屏障)
- 用于 L2 缓存管理与驻留控制的新指令
- CUDA Cooperative Groups 支持的新型线程束级规约指令
- 多项可降低软件复杂度的可编程性改进

<span id="figure-07"></span>

![GA100 流式多处理器 (SM)](./nvidia-a100-architecture/figure-07.png)

**图 7.** GA100 流式多处理器 (SM)

<span id="section-4-2"></span>

### 4.2 第三代 NVIDIA Tensor Core

Tensor Core 是专门执行矩阵数学运算的高性能计算核心, 可为 AI 和 HPC 应用带来大幅性能提升. Tensor Core 执行矩阵乘加 (MMA) 计算. 一块 NVIDIA GPU 中的数百个 Tensor Core 并行工作, 可大幅提升吞吐量和效率. Tensor Core 最初在 NVIDIA Tesla V100 GPU 中引入, 随后又在 NVIDIA 较新的 Turing GPU 中得到增强. (有关 Tensor Core 运算的背景信息, 请参阅 [NVIDIA Tesla V100 GPU 架构](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).)

<span id="table-02"></span>

![A100 相对 V100 的加速比 (TC=Tensor Core, GPU 采用各自的时钟速度)](./nvidia-a100-architecture/table-02.png)

**表 2.** A100 相对 V100 的加速比 (TC=Tensor Core, GPU 采用各自的时钟速度)

<span id="section-4-2-1"></span>

#### 4.2.1 A100 Tensor Core 提高吞吐量

A100 的新型第三代 Tensor Core 架构使每个 SM 的原始稠密 Tensor 吞吐量达到 V100 的两倍, 可加速更多数据类型, 并使稀疏矩阵计算再取得 2 倍的大幅加速.

通用矩阵乘法 (GEMM) 是神经网络训练与推理的核心运算, 用于将各层的大型输入数据矩阵与权重矩阵相乘. GEMM 运算计算矩阵乘积 $D = A * B + C$, 其中 C 和 D 是 $m$×$n$ 矩阵, A 是 $m$×$k$ 矩阵, B 是 $k$×$n$ 矩阵. 在 Tensor Core 上运行的这类 GEMM 运算, 其问题规模由矩阵尺寸决定, 通常记作 $m$×$n$×$k$.

以 FP16/FP32 混合精度 Tensor Core 运算为例, 在硬件层面, Volta 架构的每个 Tensor Core 每时钟可执行 64 次采用 FP32 累加的 FP16 融合乘加运算 (FMA), 因而每时钟可完成一次 4×4×4 混合精度矩阵乘法. 每个 Volta SM 有八个 Tensor Core, 因此单个 SM 每时钟可执行 512 次 FP16 FMA 运算, 即 1024 次独立的 FP16 浮点运算. A100 的每个 Tensor Core 每时钟可执行 256 次 FP16 FMA 运算, 因而每时钟可得出一次 8×4×8 混合精度矩阵乘法的结果. A100 GPU 的每个 SM 都有四个重新设计的新型 Tensor Core, 因此每个 SM 每时钟可执行 1024 次 FP16 FMA 运算 (即 2048 次独立的 FP16 浮点运算).

若比较整块 GPU 而非只比较 SM 级性能, 配备 108 个 SM 的 NVIDIA A100 Tensor Core GPU 共包含 432 个 Tensor Core, 可提供最高 312 TFLOPS 的稠密 FP16/FP32 混合精度性能. 这相当于整块 Tesla V100 GPU 混合精度 Tensor Core 性能的 2.5 倍, 也是 V100 标准 FP32 吞吐量 (由传统 FP32 CUDA Core 执行 FMA 运算) 的 20 倍.

[图 8](#figure-08) 比较了 V100 与 A100 的 FP16 Tensor Core 运算, 还将 V100 的标准 FP32、FP64 和 INT8 运算分别与 A100 的 TF32、FP64 和 INT8 Tensor Core 运算作了比较. 图中吞吐量为每块 GPU 的总吞吐量; A100 在 FP16、TF32 和 INT8 中采用稀疏 Tensor Core 运算. 请注意, 左上图显示两个 V100 FP16 Tensor Core, 因为 V100 SM 的每个 SM 分区含两个 Tensor Core, 而 A100 SM 只有一个.

<span id="figure-08"></span>

![A100 与 V100 Tensor Core 运算对比](./nvidia-a100-architecture/figure-08.png)

**图 8.** A100 与 V100 Tensor Core 运算对比

<span id="section-4-2-2"></span>

#### 4.2.2 A100 Tensor Core 支持所有 DL 数据类型

除 Volta Tensor Core 引入的 FP16 精度, 以及 Turing Tensor Core 新增的 INT8、INT4 和二进制 1 位精度外, A100 Tensor Core 还支持 TF32、BF16 和 FP64 格式. ([第 4.2.3 节](#section-4-2-3) 将讨论 FP64 双精度 MMA.)

Volta GPU 架构引入了可处理 IEEE FP16 数据类型的 Tensor Core, 数学吞吐量是 V100 FP32 的 8 倍. 混合精度训练会将结果累加至 FP32, 推理则累加至 FP16. 从架构的原始性能看, 若 A100 和 V100 以相同时钟速度运行, 单个 A100 SM 的 FP16 Tensor Core 性能是 V100 SM 的 2 倍, 是标准 V100 (以及 A100) FP32 FFMA 运算的 16 倍.

Turing 架构通过增加 INT8、INT4 和二进制支持, 将 Tensor Core 扩展至更多推理场景. 在 Turing 上, 这三类运算的数学吞吐量分别是 FP32 的 16、32 和 128 倍. A100 SM 的 INT8、INT4 和二进制 Tensor Core 性能分别是 Turing SM 的 2 倍, 是 A100 FP32 FFMA 的 32、64 和 256 倍.

NVIDIA Ampere 架构为 Tensor Core 新增 BF16、TF32 和 FP64 三种格式. BF16 是 IEEE FP16 的一种替代格式, 包含 8 位指数、7 位尾数和 1 位符号. 实践表明, FP16 和 BF16 均可在混合精度模式下成功训练神经网络, 无需调整超参数便可达到 FP32 的训练结果. A100 GPU 上的 FP16 和 BF16 Tensor Core 模式都能提供 FP32 的 16 倍数学吞吐量.

目前, AI 训练默认采用不经 Tensor Core 加速的 FP32 数学运算. NVIDIA Ampere 架构新支持 TF32, 让 AI 训练无需用户操作便可默认使用 Tensor Core. 非 Tensor 运算继续使用 FP32 数据路径; TF32 Tensor Core 读取 FP32 数据, 在保持与 FP32 相同数值范围的同时降低内部精度, 随后生成标准 IEEE FP32 输出. TF32 包含 8 位指数 (与 FP32 相同)、10 位尾数 (精度与 FP16 相同) 和 1 位符号.

与 Volta 一样, 自动混合精度 (AMP) 只需改动几行代码, 便可让用户在 AI 训练中使用 FP16 混合精度. 借助 AMP, A100 的 Tensor Core 性能可在 TF32 基础上再提高 2 倍.

<span id="figure-09"></span>

![TensorFloat-32 (TF32)](./nvidia-a100-architecture/figure-09.png)

**图 9.** TensorFloat-32 (TF32)

<span id="table-03"></span>

![A100 Tensor Core 输入/输出格式及相对 FP32 FFMA 的性能](./nvidia-a100-architecture/table-03.png)

**表 3.** A100 Tensor Core 输入/输出格式及相对 FP32 FFMA 的性能.

对于 NVIDIA Ampere 架构上的深度学习训练数学运算, 用户可作如下选择:

- 默认使用 TF32 Tensor Core, 无需调整用户脚本. 吞吐量最高可达 A100 上 FP32 的 8 倍, 以及 V100 上 FP32 的 10 倍.
- 如需最快训练速度, 应使用 FP16 或 BF16 混合精度训练. 吞吐量最高可达 TF32 的 2 倍、A100 上 FP32 的 16 倍, 以及 V100 上 FP32 的 20 倍.

<span id="section-4-2-3"></span>

#### 4.2.3 A100 Tensor Core 加速 HPC

高性能计算 (HPC) 应用的性能需求正在快速增长. 众多科学和研究领域的应用依赖双精度 (FP64) 计算. 为满足迅速增长的 HPC 算力需求, A100 Tensor Core 支持加速符合 IEEE 标准的 FP64 计算, FP64 性能最高可达 NVIDIA Tesla V100 GPU 的 2.5 倍. A100 新增的双精度矩阵乘加指令可取代 V100 上的 8 条 DFMA 指令, 从而减少指令获取、调度开销、寄存器读取、数据路径功耗和共享内存读取带宽. 借助 Tensor Core, A100 的每个 SM 每时钟共执行 64 次 FP64 FMA 运算 (即 128 次 FP64 运算), 吞吐量是 Tesla V100 的两倍. 配备 108 个 SM 的 A100 Tensor Core GPU 可提供 19.5 TFLOPS 的 FP64 峰值吞吐量, 为 Tesla V100 的 2.5 倍.

由于支持这些新格式, A100 Tensor Core 可用于加速 HPC 工作负载、迭代求解器和多种新的 AI 算法.

<span id="section-4-2-3-1"></span>

##### 4.2.3.1 面向 HPC 的混合精度 Tensor Core

混合精度 Tensor Core 在 HPC 中颇具前景的应用之一是迭代求精方法. 迭代求精通常用于求解线性方程组, 而线性方程组广泛存在于地球科学、流体动力学、医疗保健、材料科学、核能以及油气勘探等领域的 HPC 应用中.

cuSOLVER 中的 Tensor Core 加速迭代求精求解器 (TCAIRS) 可自动在此类应用中使用混合精度. 去年, 一项面向国际热核聚变实验反应堆的聚变反应研究表明, 对此类求解器使用 V100 的 FP16 Tensor Core 时, 混合精度技术可在 V100 上实现 3.5 倍加速. 该研究采用的同一项技术, 还使 Summit 超级计算机在 HPL-AI 基准测试中的性能提高到三倍.

CUDA 11.0 中的 cuSOLVER 增加了对 A100 新型 Tensor Core 格式的支持, 其中包括 TF32. 下方 [图 10](#figure-10) 和 [图 11](#figure-11) 展示了 TCAIRS 求解器对 SuiteSparse Matrix 集合中 37 项测试的结果, 比较 FP32、带输入缩放的 FP16、BF16 和 TF32 的收敛率与性能. 这些结果与参考 FP64 求解器的性能作了比较, 后者利用 A100 上的 FP64 Tensor Core. 若混合精度求解器因收敛缓慢或不收敛而自动回退至 FP64 求解器, 迭代次数记为负数; 由于结果包含失败尝试的成本, 加速比小于 1.

如 [图 10](#figure-10) 和 [图 11](#figure-11) 所示, 与其他 Tensor Core 模式相比, TF32 的结果最快且最稳健. 在各 Tensor Core 模式中, TF32 收敛所需的迭代次数最少. FP32 有一次回退, TF32 仅有两次, 带输入缩放的 FP16 有三次, BF16 Tensor Core 模式则有六次. 相对 FP64 求解器, TF32 Tensor Core 的几何平均加速比为 2.0 倍, FP16 为 1.9 倍, BF16 为 1.8 倍. 对尺寸约为 40K 的大型复数矩阵, A100 上使用 TF32 的 TCAIRS 求解器可实现最高 4 倍加速.

混合精度 Tensor Core 加速不仅适用于稠密线性求解器, 也可扩展至稀疏问题, 以及矩阵乘法在算法复杂度中占较大比重的其他数值方法.

<span id="figure-10"></span>

![TCAIRS 求解器收敛至 FP64 精度所需的迭代次数](./nvidia-a100-architecture/figure-10.png)

**图 10.** TCAIRS 求解器收敛至 FP64 精度所需的迭代次数

<span id="figure-11"></span>

![TCAIRS 求解器相对基准 FP64 直接求解器的加速比](./nvidia-a100-architecture/figure-11.png)

**图 11.** TCAIRS 求解器相对基准 FP64 直接求解器的加速比

<span id="section-4-3"></span>

### 4.3 A100 引入细粒度结构化稀疏性

NVIDIA 随 A100 GPU 推出细粒度结构化稀疏性, 这种新方法可使深度神经网络的计算吞吐量翻倍.

深度学习可以利用稀疏性, 是因为各个权重的重要性会在学习过程中变化; 网络训练结束时, 只有部分权重会对确定学习结果产生实质作用. 其余权重不再需要.

细粒度结构化稀疏性会约束允许的稀疏模式, 让硬件能够更高效地完成必要的输入操作数对齐. NVIDIA 工程师发现, 深度学习网络能根据训练反馈在训练过程中调整权重, 因此这种结构约束通常不会影响训练后网络的推理准确率. 这样便可利用稀疏性加速推理. 如要加速训练, 必须在训练过程早期引入稀疏性才能获得性能收益; 如何在不损失准确率的情况下加速训练, 仍是一个活跃的研究方向.

有关稀疏性的更多背景信息, 请参阅 [第 14 节](#section-14).

<span id="section-4-3-1"></span>

#### 4.3.1 稀疏矩阵定义

新的 2:4 稀疏矩阵定义规定, 每个包含四个元素的向量中允许有两个非零值, 以此强制形成特定结构.

如下面的 [图 12](#figure-12) 所示, A100 支持按行应用 2:4 结构化稀疏性. 由于矩阵结构有明确规定, 它可以高效压缩, 使内存存储量和带宽需求均降低近 2 倍.

<span id="figure-12"></span>

![A100 细粒度结构化稀疏性](./nvidia-a100-architecture/figure-12.png)

**图 12.** A100 细粒度结构化稀疏性

NVIDIA 为使用这种 2:4 结构化稀疏模式稀疏化用于推理的深度神经网络, 开发了一套简单且通用的流程. 首先用稠密权重训练网络, 然后应用细粒度结构化剪枝, 最后再执行额外的训练步骤, 微调剩余的非零权重. 对视觉、目标检测、分割、自然语言建模和翻译等数十种网络的评估表明, 该方法几乎不会损失推理准确率.

<span id="section-4-3-2"></span>

#### 4.3.2 稀疏矩阵乘加 (MMA) 运算

A100 新增的稀疏 MMA 指令会跳过值为零的元素, 从而使 Tensor Core 计算吞吐量翻倍. 例如在下面的 [图 13](#figure-13) 中, 矩阵 A 是一个符合 2:4 结构模式要求、稀疏度为 50% 的稀疏矩阵, 矩阵 B 则是尺寸减半的稠密矩阵. 标准 MMA 运算不会跳过零值, 需要用 N 个周期计算完整的 16×8×16 矩阵乘法. 使用稀疏 MMA 指令时, 只会将矩阵 A 每一行中的非零元素与矩阵 B 的对应元素匹配. 计算由此转化为规模较小的矩阵乘法, 只需 N/2 个周期, 加速 2 倍.

<span id="figure-13"></span>

![稠密 MMA 与稀疏 MMA 运算示例](./nvidia-a100-architecture/figure-13.png)

**图 13.** 稠密 MMA 与稀疏 MMA 运算示例

<span id="section-4-4"></span>

### 4.4 统一的 L1 数据缓存和共享内存

NVIDIA 在 Volta V100 中首次引入统一的 L1 数据缓存和共享内存子系统架构, 它显著提升性能, 同时简化编程, 并减少达到或接近应用峰值性能所需的调优. 将数据缓存和共享内存功能合并至同一个内存块, 可为两类内存访问提供最佳整体性能. A100 的 L1 数据缓存和共享内存总容量为每个 SM 192 KB, V100 则为每个 SM 128 KB.

将 L1 缓存集成至共享内存块可保证低延迟和高带宽. L1 既是流数据的高吞吐量通道, 又能为频繁复用的数据提供高带宽、低延迟访问, 兼顾两方面的优势. A100 更大的 L1/共享内存子系统, 进一步提高了通过 L1 数据缓存访问设备内存的应用性能, 可使性能接近显式使用并管理高速共享内存的水平. (有关统一 L1 数据缓存与共享内存子系统如何使 L1 缓存操作获得共享内存性能优势的示例, 请参阅 [NVIDIA Tesla V100 白皮书](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf).)

<span id="section-4-5"></span>

### 4.5 同时执行 FP32 和 INT32 运算

与 Tesla V100 和 Turing GPU 类似, A100 SM 也配有独立的 FP32 和 INT32 核心, 可在满吞吐量下同时执行 FP32 与 INT32 运算, 并提高指令发射吞吐量. 许多应用的内层循环会将指针算术 (整数内存地址计算) 与浮点计算结合, 因此能够受益于 FP32 和 INT32 指令的同时执行. 流水线循环在每次迭代中都可更新地址 (INT32 指针算术), 并为下一次迭代加载数据, 同时使用 FP32 处理当前迭代.

<span id="section-4-6"></span>

### 4.6 A100 HBM2 与 L2 缓存内存架构

GPU 内存架构与层次结构的设计对应用性能至关重要, 也会影响 GPU 的尺寸、成本、功耗和可编程性. GPU 包含多种不同的内存子系统, 从大量芯片外 DRAM (帧缓冲区) 设备内存, 到不同层级和类型的片上内存, 再到 SM 计算所用的寄存器文件. A100 GPU 采用高性能 HBM2 作为 DRAM 技术.

CUDA 程序访问的全局与本地内存区域位于 HBM2 内存空间中, 在 CUDA 术语中称为设备内存. 常量内存空间位于设备内存中, 并由常量缓存加速. 纹理和表面内存空间位于设备内存中, 并由纹理缓存加速. L2 缓存负责缓存对 HBM2 (设备) 内存的读写. 所有 SM 以及 GPU 上运行的所有应用都可访问 HBM2 和 L2 内存空间.

<span id="section-4-6-1"></span>

#### 4.6.1 A100 HBM2 DRAM 子系统

随着 HPC、AI 和分析数据集持续增长, 待解决的问题也日趋复杂, 更大的 GPU 内存容量和更高的内存带宽已不可或缺. Tesla P100 是全球首个支持高带宽 HBM2 内存技术的 GPU 架构, Tesla V100 则提供了速度更快、效率更高且容量更大的 HBM2 实现. A100 再次提高了 HBM2 性能与容量的标准.

HBM2 内存由与 GPU 位于同一物理封装内的内存堆栈组成, 相比传统 GDDR5/6 内存设计可大幅节省功耗和面积, 从而在系统中安装更多 GPU. 我们的 [Pascal 架构白皮书](https://images.nvidia.com/content/pdf/tesla/whitepaper/pascal-architecture-whitepaper.pdf) 介绍了 HBM2 技术的基本细节.

A100 GPU 的 SXM4 式电路板上配有 40 GB 高速 HBM2 DRAM. 内存由五组工作的 HBM2 堆栈组成, 每组包含八个内存裸片. 数据速率为 1215 MHz (DDR) 时, A100 HBM2 可提供 1555 GB/s 内存带宽, 比 Tesla V100 高 1.7 倍以上.

<span id="section-4-6-2"></span>

#### 4.6.2 ECC 内存容错能力

A100 HBM2 内存子系统支持单错纠正、双错检测 (SECDED) 纠错码 (ECC), 用于保护数据. ECC 可提高对数据损坏敏感的计算应用的可靠性. 在 GPU 处理超大型数据集和/或长时间运行应用的大规模集群计算环境中, 它尤其重要. A100 中的其他主要内存结构也受 SECDED ECC 保护, 包括 L2 缓存以及所有 SM 内的 L1 缓存和寄存器文件.

<span id="section-4-6-3"></span>

#### 4.6.3 A100 L2 缓存

A100 Tensor Core GPU 所用的 A100 GPU 配备 40 MB L2 缓存, 容量是 Tesla V100 L2 缓存的 6.7 倍. L2 缓存容量大幅增加, 可显著提高许多 HPC 和 AI 工作负载的性能, 因为数据集和模型中更大的部分现在可以缓存起来并反复访问, 速度远高于 HBM2 内存的读写. 某些受 DRAM 带宽限制的工作负载也会受益于更大的 L2 缓存, 例如采用较小批量大小的深度神经网络.

A100 L2 缓存位于 GPC 外部, 是 GPC 和 SM 的共享资源. L2 缓存被分为两个分区, 以实现更高带宽和更低延迟的内存访问. 每个 L2 分区会对直接连接到该分区的 GPC 内 SM 所发起的内存访问进行本地化和缓存. 这种结构使 A100 的 L2 带宽比 V100 提高 2.3 倍. 硬件缓存一致性在整块 GPU 上维持 CUDA 编程模型, 应用会自动利用 A100 新型 L2 缓存的带宽和延迟优势.

每个 L2 缓存分区划分为 40 个 L2 缓存切片. 每个内存控制器对应八个 512 KB L2 切片. 如下文 [第 6 节](#section-6) 所述, MIG 配置下一个 GPU 实例的每个 GPU 切片中, 都有一个由 10 个 L2 缓存切片组成的 L2 切片组. A100 的 L2 读取带宽为每时钟 5120 字节, V100 的 L2 缓存读取带宽则为每时钟 2048 字节.

NVIDIA Ampere 架构提供 L2 缓存驻留控制, 供程序员管理应保留在缓存中或从中逐出的数据 (更多细节见下文 [第 11 节](#section-11)).

NVIDIA Ampere 架构新增计算数据压缩, 可加速非结构化稀疏性及其他可压缩数据模式. L2 中的压缩最多可使 DRAM 读写带宽提高 4 倍、L2 读取带宽提高 4 倍, 并使 L2 容量提高 2 倍.

<span id="table-04"></span>

![NVIDIA 数据中心 GPU 对比](./nvidia-a100-architecture/table-04.png)

**表 4.** NVIDIA 数据中心 GPU 对比

<span id="section-5"></span>

## 5 最大限度提高深度学习应用的 Tensor Core 性能与效率

如上文所述, NVIDIA Tensor Core 最初在 NVIDIA Volta GPU 架构中引入, 用于显著加速神经网络训练和推理中常见的矩阵乘法运算. 与标准 FP32 精度运算相比, Volta 上的 Tensor Core 可实现 8 倍的混合精度矩阵乘法峰值加速.

与 Tesla V100 相比, 基于 NVIDIA Ampere 架构的 A100 GPU 拥有更多 SM (108 个, V100 为 80 个), 其第三代 Tensor Core 还能执行规模更大的 Tensor 运算. A100 GPU 中的 Tensor Core 所支持的混合精度峰值计算性能, 比标准 FP32 FMA 运算高 16 倍.

NVIDIA Ampere 架构引入了多项新特性和优化, 可提高 Tensor Core 利用率、改善可编程性、降低软件复杂度和内存带宽用量, 并减少延迟及其他开销.

<span id="section-5-1"></span>

### 5.1 深度学习性能的强扩展

深度学习需要庞大的计算资源, 但其并行任务会分解为一小段一小段存在顺序依赖的工作. 典型的深度神经网络由很长的互连层链构成. 每一层执行类似通用矩阵乘法 (GEMM) 的运算: 将输入值矩阵与权重矩阵相乘, 得到输出矩阵. 输出矩阵通常还会经过某种激活运算, 再传递至网络的下一层. 每次 GEMM 的输出矩阵会拆分为较小的分块, 映射至 GPU 中的多个 SM.

NVIDIA Ampere 架构着眼于强扩展, 以加速现有深度神经网络. 弱扩展较容易实现, 因为工作负载的并行度可以随架构加速能力一起增长. 在强扩展中, 从一种架构换到下一种架构时, 每块 GPU 的工作负载保持不变. 对深度学习而言, 这意味着即使 A100 Tensor Core 消耗数据的速度是 V100 的 2.5 倍, 每个 SM 的 GEMM 分块大小也不能增加. NVIDIA Ampere 架构实现了下述多项特性和优化, 以更快、更高效地向 Tensor Core 供给数据.

<span id="section-5-2"></span>

### 5.2 提高 Tensor Core 性能的 NVIDIA Ampere 架构新特性

**数据共享改进.** NVIDIA Ampere 架构的第三代 Tensor Core 允许一个线程束中的全部 32 个线程共享数据, 而 Volta Tensor Core 只允许 8 个线程共享. 在更多线程间共享数据可降低为 Tensor Core 供给数据所需的寄存器文件带宽. 它还减少了从共享内存 (SMEM) 加载到寄存器文件的重复数据量, 从而节省带宽和寄存器文件存储空间. 为进一步提高效率, A100 Tensor Core 指令将每条指令所执行矩阵乘法的 k 维相对 V100 最多增大 4 倍. 总体而言, 执行矩阵乘法时, A100 发出的指令数比 V100 少 8 倍, 寄存器文件访问次数少 2.9 倍.

<span id="figure-14"></span>

![A100 Tensor Core 吞吐量与效率](./nvidia-a100-architecture/figure-14.png)

**图 14.** A100 Tensor Core 吞吐量与效率

**数据提取改进.** NVIDIA Ampere 架构包含一条新的异步复制指令, 可将数据从全局内存 (通常来自 L2 缓存和 DRAM) 直接加载至 SM 共享内存. 在 Volta 上, 首先使用全局加载指令经 L1 缓存将数据加载至寄存器文件, 然后通过共享存储指令将数据从寄存器文件传至共享内存, 最后再用共享加载指令将数据从共享内存加载至多个线程和线程束的寄存器. NVIDIA Ampere 架构 GPU 新增的“全局加载-共享存储”异步复制指令无需让数据往返寄存器文件, 因而节省 SM 内部带宽, 也不必为正在传输的数据分配寄存器文件存储空间. 本文后面会进一步介绍异步复制指令.

<span id="figure-15"></span>

![A100 SM 数据移动效率](./nvidia-a100-architecture/figure-15.png)

**图 15.** A100 SM 数据移动效率

新的异步屏障与异步复制指令配合, 可实现高效的数据提取流水线; A100 还将每个 SM 可分配的最大 SMEM 增至 164 KB, 是 V100 的 1.7 倍 (V100 为 96 KB). 这些改进使 A100 SM 能够持续流式传送数据, 让 L2 缓存始终得到充分利用.

**L2 缓存与 DRAM 带宽改进.** NVIDIA A100 GPU 的 SM 数量增加, Tensor Core 也更强大, 因而需要从 DRAM 和 L2 缓存以更高速率提取数据. 为了向 Tensor Core 供给数据, A100 实现了五堆栈 HBM2 内存子系统, 带宽为 1555 GB/s, 比 V100 快 1.7 倍以上. A100 的 L2 缓存读取带宽还达到 V100 的 2.3 倍.

除了提高原始数据带宽, A100 还以容量接近 Tesla V100 7 倍的 40 MB L2 缓存改善数据提取效率, 降低 DRAM 带宽需求. 为充分利用 L2 容量, A100 加入了改进的缓存管理控制. 新控制针对神经网络训练与推理以及通用计算工作负载进行了优化, 可尽量减少向内存回写, 并将复用数据保留在 L2 中以减少重复的 DRAM 流量, 从而提高缓存数据的使用效率.

例如, 对 DL 推理工作负载, 可以将乒乓缓冲区长期缓存在 L2 中以加快数据访问, 同时避免回写 DRAM. 对 DL 训练中常见的生产者-消费者链, L2 缓存控制可以针对先写后读的数据依赖优化缓存. 在 LSTM 网络中, 多次 GEMM 运算共享的循环权重可以优先缓存于 L2 并重复使用.

<span id="figure-16"></span>

![A100 L2 缓存驻留控制](./nvidia-a100-architecture/figure-16.png)

**图 16.** A100 L2 缓存驻留控制

**压缩.** 为提高效率并加强强扩展能力, A100 新增计算数据压缩. 压缩最多可节省 4 倍 DRAM 读写带宽和 4 倍 L2 读取带宽, 并使 L2 容量提高最多 2 倍.

<span id="figure-17"></span>

![A100 计算数据压缩](./nvidia-a100-architecture/figure-17.png)

**图 17.** A100 计算数据压缩

**总结.** 下方 [图 18](#figure-18) 概括了 A100 在计算与内存层次结构各层带来的改进. 这些创新让 A100 能够把深度学习强扩展至前所未有的性能水平.

<span id="figure-18"></span>

![A100 强扩展创新](./nvidia-a100-architecture/figure-18.png)

**图 18.** A100 强扩展创新

<span id="section-5-3"></span>

### 5.3 计算能力

A100 GPU 支持新的计算能力 8.0. [表 5](#table-05) 比较了 NVIDIA GPU 架构不同计算能力的参数.

<span id="table-05"></span>

![计算能力: GP100、GV100 与 GA100](./nvidia-a100-architecture/table-05.png)

**表 5.** 计算能力: GP100、GV100 与 GA100

<span id="section-6"></span>

## 6 MIG (多实例 GPU) 架构

许多数据中心工作负载的规模和复杂度仍在增长, 但某些加速任务的要求并不高, 如早期开发或使用较小批量对简单模型进行推理. 数据中心管理者希望维持较高的资源利用率, 因此理想的数据中心加速器既要能处理大型任务, 也要能高效加速许多较小的工作负载.

<span id="section-6-1"></span>

### 6.1 背景

2017 年, NVIDIA Tesla V100 GPU 引入了硬件加速的多进程服务 (MPS), 使多个应用能够同时在相互独立的 GPU 执行资源 (SM) 上运行.

深度学习推理应用使用 Volta MPS 后, 相比传统 GPU 工作提交方法可获得高得多的吞吐量和更低延迟, 允许将许多独立推理作业同时提交至 GPU, 并提高 GPU 整体利用率. (有关 Volta MPS 的更多细节, 请参阅 [NVIDIA Tesla V100 GPU 架构](https://images.nvidia.com/content/volta-architecture/pdf/volta-architecture-whitepaper.pdf) 白皮书.)

<span id="figure-19"></span>

![Pascal 中基于软件的 MPS 与 Volta 中硬件加速的 MPS](./nvidia-a100-architecture/figure-19.png)

**图 19.** Pascal 中基于软件的 MPS 与 Volta 中硬件加速的 MPS

不过, 由于所有应用共享内存系统资源, 若一个应用需要很高的 DRAM 带宽, 或其请求超出 L2 缓存容量, 就可能干扰其他应用. Volta MPS 在 Ampere 上仍得到完整支持; 它用于让单个用户的多个应用共享 GPU, 并不面向多用户或多租户场景.

<span id="section-6-2"></span>

### 6.2 NVIDIA Ampere GPU 架构的 MIG 能力

新的 MIG 功能可将每块 A100 划分为最多七个 GPU 实例, 以优化利用率, 实际上让每个用户和应用都更容易使用 GPU.

A100 GPU 的新 MIG 功能可将一块 GPU 分成多个称为 GPU 实例的 GPU 分区. 每个实例的 SM 在整个内存系统中都有独立且隔离的路径: 片上交叉开关端口、L2 缓存组、内存控制器和 DRAM 地址总线都单独分配给一个实例. 因此, 即便其他任务频繁置换各自的缓存或占满 DRAM 接口, 单个用户的工作负载仍能以可预测的吞吐量和延迟运行, 并获得固定的 L2 缓存配额和 DRAM 带宽.

MIG 可以据此划分可用的 GPU 计算资源, 为虚拟机、容器、进程等不同客户提供明确的服务质量 (QoS) 与故障隔离. 多个 GPU 实例可在一块物理 A100 GPU 上并行运行. MIG 还保持 CUDA 编程模型不变, 尽量减少编程工作.

CSP 可以利用 MIG 提高 GPU 服务器的利用率, 无需增加成本便可提供最多 7 倍的 GPU 实例. MIG 提供 CSP 所需的 QoS 和隔离保证, 确保一个客户 (VM、容器或进程) 不会影响另一客户的工作或调度.

CSP 经常根据客户的使用模式划分硬件. 只有硬件资源在运行期间提供稳定的带宽、妥善的隔离和良好性能, 分区才会有效.

使用基于 NVIDIA Ampere 架构的 GPU 时, 用户可以像面对物理 GPU 一样查看新的虚拟 GPU 实例并在其上调度作业. MIG 可与 Linux 操作系统及其虚拟机监控程序配合. 用户可通过 Docker Engine 等运行时, 在容器中使用 MIG; 对 Kubernetes 容器编排的支持也将很快推出.

<span id="section-6-3"></span>

### 6.3 MIG 的重要用例

MIG 的一个重要用例称为“多租户” (通过 NVIDIA vGPU 技术使用), CSP 可借此将独立的 GPU 实例租给不同客户. 每个 GPU 实例中运行的应用均受隔离保护, 不会受到其他 GPU 实例内同时运行的应用所发生故障的影响. 此类用例必须具备数据保护、故障隔离和 QoS.

另一种称为“单租户、单用户”的 MIG 用例, 可支持单个用户在一台工作站上运行多个 GPU 应用, 且应用之间的故障隔离至关重要. “单租户、多用户”场景则适合企业支持内部工作组, 或向多个外部用户提供 AI 推理服务及其他 GPU 加速服务.

MIG 允许在不同虚拟机 (VM) 之间划分计算资源, 并让多个 VM 在保持故障隔离的同时并发执行. 即使 VM 迁移至另一块 GPU, 仍可维持一致的性能. 将多个 VM 紧密部署在同一块 GPU 上, 还能提高 GPU 利用率.

<span id="figure-20"></span>

![当前的 CSP 多用户节点](./nvidia-a100-architecture/figure-20.png)

**图 20.** 当前的 CSP 多用户节点

<span id="figure-21"></span>

![CSP MIG 配置示例](./nvidia-a100-architecture/figure-21.png)

**图 21.** CSP MIG 配置示例

<span id="section-6-4"></span>

### 6.4 MIG 架构与 GPU 实例详解

创建 GPU 实例可以理解为把一块大型 GPU 拆分为多块较小的 GPU, 每个 GPU 实例都有专用的计算与内存资源. 每个 GPU 实例都像一块较小但功能完整的独立 GPU, 包含预定数量的 GPC、SM、L2 缓存切片、内存控制器和帧缓冲内存.

GPU 实例由多个“GPU 切片”构成, 每个 GPU 切片包括一条“系统管线” (定义见下文)、一个 GPC、一个 L2 切片组 (含 10 个 L2 缓存切片), 以及一部分帧缓冲内存的访问权. A100 GPU 总共支持 7 个 GPU 切片. 注意: 在 MIG 工作模式下, 每个 GPU 切片的单个 GPC 都启用七个 TPC (14 个 SM), 从而使所有 GPU 切片具有相同且稳定的计算性能.

系统管线是新型 A100 GigaThread™ 引擎的一部分, 负责与主机 CPU 通信, 并将工作调度至 GPU 切片中的一个 GPC (及其 SM). A100 Tensor Core GPU 共包含七条系统管线以支持 MIG. 其中一条 A100 系统管线与以往的 GPU 架构相似, 同时支持图形和计算工作; 另外六条新系统管线只支持计算工作负载. 在图形模式下运行时, A100 与过去的 GPU 类似, 使用唯一支持图形的系统管线控制整块 GPU (A100 GPU 的全部七个 GPC), 并运行单个图形上下文. 在计算模式下, 七条系统管线均可同时运行多个计算上下文. 注意, A100 GPU 处于 MIG 模式时不支持图形流水线操作. MIG 只用于计算模式.

GPU 内存切片是另一种 MIG 结构, 包含 GPU 实例所有 GPU 切片中的全部 L2 切片组 (由若干 L2 缓存切片组成的块) 及对应的帧缓冲内存. 在一个 GPU 实例中运行的应用上下文不会使用另一 GPU 实例的 L2 切片, 从而有效隔离并按比例分配不同 GPU 实例所用的内存带宽.

<span id="figure-22"></span>

![包含三个 GPU 实例的 MIG 计算配置示例](./nvidia-a100-architecture/figure-22.png)

**图 22.** 包含三个 GPU 实例的 MIG 计算配置示例.

单个 GPU 实例可为其上运行的所有客户应用提供内存 QoS. 多个实例按比例共享 GPU 帧缓冲内存.

可以为每个 GPU 实例静态分配不同数量的 GPU 切片, 从而按需要划分计算与内存带宽资源, 并改善 QoS、故障隔离、错误遏制和错误恢复.

<span id="section-6-5"></span>

### 6.5 计算实例

“计算实例”是另一种分组方式, 可在 GPU 实例内部配置不同级别的计算能力, 封装可在 GPU 实例中执行工作的所有计算资源 (GPC 数量、复制引擎、NVDEC 单元等). 默认情况下, 每个 GPU 实例下会创建一个计算实例, 公开 GPU 实例内全部可用的 GPU 计算资源. GPU 实例还可进一步细分为多个较小的计算实例, 继续划分其计算资源.

每个计算实例都支持 Volta 式 MPS 功能, 可将多个不同的 CPU 进程 (主机应用上下文) 合并到一个 CUDA 上下文中并在 GPU 上运行. MPS 客户的最大数量与计算实例大小成比例. A100 完整支持 MPS, 对需要利用 MPS 吞吐量执行 MPI 的 HPC 用例尤其重要.

<span id="figure-23"></span>

![包含多个独立 GPU 计算工作负载的 MIG 配置](./nvidia-a100-architecture/figure-23.png)

**图 23.** 包含多个独立 GPU 计算工作负载的 MIG 配置

<span id="figure-24"></span>

![MIG 分区过程示例](./nvidia-a100-architecture/figure-24.png)

**图 24.** MIG 分区过程示例

<span id="section-6-6"></span>

### 6.6 计算实例支持上下文同时执行

计算实例使多个上下文能够同时在 GPU 上运行. 一个计算实例可包含一个或多个 GPU 切片, 也可将单条系统管线配置为连接 GPU 实例中其他 GPU 切片的多个 GPC、L2 切片和内存. 在这种情况下, 该 GPU 实例内其他 GPU 切片的系统管线会被禁用.

<span id="figure-25"></span>

![包含三个 GPU 实例和四个计算实例的 MIG 配置示例](./nvidia-a100-architecture/figure-25.png)

**图 25.** 包含三个 GPU 实例和四个计算实例的 MIG 配置示例.

从技术上讲, 计算实例定义为一条系统管线加上 GPU 实例内最多 7 个 GPC. 共享一个计算实例的所有应用也共享一条系统管线, 且每个计算实例可独立于其他计算实例切换上下文. 在 A100 之前, 所有 GPC 一起切换上下文; 在 A100 中, 不同计算实例内的 GPC 会分别切换上下文, 因而每条系统管线可仅为部分 GPC 切换上下文, 使多个计算实例能够独立运行.

请注意, 每个计算实例也支持 Volta 式 MPS 功能, 可将多个不同 CPU 进程 (应用上下文) 合并为一个应用上下文并在 GPU 上运行.

总体而言, MIG 支持许多配置方式, 开发者和系统管理员文档会介绍具体细节.

<span id="section-6-7"></span>

### 6.7 MIG 迁移

为了管理、调优、维护 vGPU (虚拟 GPU) 虚拟机 (VM) 配置并对其进行负载均衡, MIG 的一项重要特性是可以在一块 GPU 的不同 GPU 实例之间迁移 vGPU, 更常见的情况则是在集群中的不同 GPU 之间迁移. 迁移过程在概念上很直接. GPU 实例中属于某个 vGPU 的 GPU 切片状态信息会先被保存, 再恢复至另一具有相同 GPU 切片数量的 GPU 实例.

当集群中的多块 GPU 只得到部分利用时, MIG 迁移可将作业移动并集中至更少的 GPU 上, 从而减少碎片, 并经常减少支持给定数量 vGPU 所需的物理 GPU 数量. 这样可以释放某些 GPU 来运行更大的作业, 也可以让未使用的 GPU 进入节能模式, 降低数据中心成本. MIG 迁移还允许卸载 GPU 以便维护, 而不必终止作业.

<span id="section-7"></span>

## 7 第三代 NVLink

NVIDIA Ampere 架构的 A100 GPU 和新款 NVSwitch 采用第三代 NVIDIA 高速 NVLink 互连. NVLink 是一种无损、高带宽、低延迟的共享内存互连, 还包含链路级错误检测和数据包重放等容错功能, 用于保证数据成功传输.

新型 NVLink 为每块 GPU 提供更多链路和快得多的 GPU-GPU 通信带宽, 并改进错误检测与恢复功能, 因而显著增强多 GPU 扩展性、性能和可靠性. A100 GPU 可通过 NVLink 链路访问对等 GPU 内存, 带宽远高于 PCI Express 所能达到的水平.

新型 NVLink 每对信号线的数据速率为 50 Gbit/s, 接近 Tesla V100 中 25.78 Gbit/s 的两倍. 每条链路在各方向均使用 4 对差分信号线 (4 条通道), Volta 则使用 8 对信号线 (8 条通道). 单条链路每个方向可提供 25 GB/s 带宽, 与 Volta GPU 相近, 但所用信号线只有 Volta 的一半. A100 的 NVLink 链路总数增至 12 条, Tesla V100 为 6 条, 因而整块 A100 的总带宽达到 600 GB/s, Tesla V100 则为 300 GB/s.

每块 A100 上的 12 条 NVLink 链路支持多种配置, 可与其他 GPU 和交换机建立高速连接. 为满足规模更大、结构更复杂的 DNN 和 HPC 模拟日益增长的计算需求, 新款 DGX A100 系统 (见 [第 13 节](#section-13)) 包含八块 A100 GPU, 通过支持新 NVLink 的 NVSwitch 相连. 多个 DGX A100 系统可通过 Mellanox InfiniBand 或 Mellanox 以太网等网络结构互连, 横向扩展数据中心, 构建性能强大乃至超级计算机级的系统. 更强大的 NVIDIA DGX POD™ 和 [NVIDIA DGX SuperPOD](https://www.nvidia.com/en-us/data-center/resources/nvidia-dgx-superpod-reference-architecture/)™ 系统会包含多个 DGX A100 系统, 以强扩展方式提供高得多的算力.

<span id="figure-26"></span>

![配备八块 A100 GPU 的 NVIDIA DGX A100](./nvidia-a100-architecture/figure-26.png)

**图 26.** 配备八块 A100 GPU 的 NVIDIA DGX A100

第三代 NVLink 中的所有写入现在都采用非投递方式, 因而可在请求方执行同步, 并将错误归因信息返回至特定执行上下文. 新增的特性还能提高小型载荷写入和无数据响应的效率.

<span id="section-8"></span>

## 8 支持 SR-IOV 的 PCIe Gen 4

A100 GPU 支持 PCI Express Gen 4 (PCIe Gen 4), x16 连接每个方向可提供 31.5 GB/s 带宽, 是 PCIe 3.0/3.1 的两倍. 更高速度尤其有利于 A100 GPU 连接支持 PCIe 4.0 的 CPU, 也有助于更快的网络接口, 例如支持 200 Gbit/s InfiniBand 以提高 GPU 集群性能. A100 还支持单根 I/O 虚拟化 (SR-IOV), 可让多个进程或虚拟机 (VM) 共享并虚拟化一块通过 PCIe 连接的 GPU. A100 还允许一块通过 SR-IOV PCIe 连接的 GPU 所提供的虚拟功能 (VF) 或物理功能 (PF), 通过 NVLink 访问对等 GPU.

<span id="section-9"></span>

## 9 错误与故障检测、隔离和遏制

通过检测、遏制并经常纠正错误和故障来提高 GPU 的运行时间与可用性, 而不是强制重置 GPU, 这一点至关重要, 在大型多 GPU 集群以及 MIG 配置等单 GPU 多租户环境中尤其如此. NVIDIA A100 Ampere 架构 GPU 包含许多新技术, 用于改进错误或故障归因 (确定哪些应用引发错误)、隔离 (隔离故障应用, 使其不影响同一 GPU 或 GPU 集群上运行的其他应用) 和遏制 (确保一个应用中的错误不会泄漏并影响其他应用).

新的 NVIDIA Ampere 架构故障处理技术对 MIG 环境尤为重要, 可确保共享同一块 GPU 的客户之间得到妥善隔离并保持安全. 如上文 [第 7 节](#section-7) 所述, 通过 NVLink 连接的 GPU 现在也具备更可靠的错误检测和恢复功能. 远端 GPU 上的页面错误会经 NVLink 传回源 GPU. 远程访问故障通信是大型 GPU 计算集群的一项重要容错特性, 有助于确保一个进程或 VM 中的故障不会导致其他进程或 VM 停止运行.

<span id="section-10"></span>

## 10 A100 架构的其他特性

NVIDIA A100 GPU 还包含多项新增或改进的特性, 可提高应用性能和可编程性. 下面列出其中几项. 另请访问 [NVIDIA 开发者网站](https://developer.nvidia.com/) 了解更多信息.

<span id="section-10-1"></span>

### 10.1 用于 DL 训练的 NVJPG 解码

A100 GPU 新增了基于硬件的 JPEG 解码功能. 对图像进行 DL 训练或推理时, 要实现高吞吐量, 一个根本问题是 JPEG 解码造成的输入瓶颈. 由于图像位处理采用串行操作, CPU 和 GPU 的 JPEG 解码效率并不高. 如果由 CPU 执行 JPEG 解码, PCIe 还会成为另一处瓶颈. A100 通过增加硬件 JPEG 解码引擎解决这些问题.

A100 包含一个名为 NVJPG 的五核心硬件 JPEG 解码引擎. 应用可将图像分成每批最多五张, 再传给 NVJPG 处理. 这些图像可以采用不同尺寸, 但为了获得最佳性能, 应尽可能将尺寸相近的图像放在同一批次.

支持的 JPEG 解码格式:

- YUV420
- YUV422
- YUV444
- YUV400
- RGBA. 对 1 百万像素及以上的大型图像, GPU 加速时钟 (1410 MHz) 下以百万像素/秒计的性能如下:

<span id="table-06"></span>

![不同视频格式下的 NVJPG 解码速率](./nvidia-a100-architecture/table-06.png)

**表 6.** 不同视频格式下的 NVJPG 解码速率

<span id="section-10-2"></span>

### 10.2 光流加速器

光流和立体视差是计算机视觉中两种基础且相互关联的图像分析方法. 光流测量两幅图像之间各点的表观运动, 立体视差则通过一组平行且经过标定的双目相机, 测量物体的 (逆) 深度.

<span id="figure-27"></span>

![光流与立体视差示意图](./nvidia-a100-architecture/figure-27.png)

**图 27.** 光流与立体视差示意图

光流和立体视差广泛用于汽车与机器人导航、电影制作、视频分析与理解、增强现实和虚拟现实等计算机视觉任务. 对光流和立体视差的测量已经研究了数十年; 尽管当前最佳技术进步很大, 这些问题仍颇具挑战, 尤其难以按现代相机的像素速率实时获得稠密数据, 而现代相机的速率通常超过每秒 5000 万像素, 且很容易达到这一数字的 10 倍.

GA100 光流加速器是一种硬件模块, 能以较高像素速率完成光流和立体视差估计. 可以通过参数选择来调整质量和性能.

<span id="section-10-3"></span>

### 10.3 原子操作改进

A100 GPU 延续 V100 的原子操作进展, 提高全局内存中原子操作的吞吐量, 对 DL 工作负载尤其有利. 许多 DL 工作负载会在训练和推理中使用 FP16 原子操作. 相比 V100, A100 将 FP16 原子操作吞吐量提高 11 倍, FP32 原子操作吞吐量提高 2.7 倍.

<span id="section-10-4"></span>

### 10.4 用于 DL 的 NVDEC

与 V100 相比, A100 大幅改进了视频解码能力. 在 DL 平台中, 输入视频采用 H264、HEVC、VP9 等业界标准压缩. DL 平台要实现较高的端到端吞吐量, 一项重大挑战是让输入视频解码性能跟上训练或推理性能, 否则就无法充分利用 GPU 的完整 DL 性能. A100 在这一方面有了大幅进步, 新增五个 NVDEC (NVIDIA DECode) 单元.

与 V100 对比:

- A100 配备 5 个 NVDEC, V100 配备 1 个
- 每个 NVDEC 的 HEVC 解码性能提升
- A100 支持 HEVC 4:4:4

<span id="table-07"></span>

![GA100 硬件解码支持](./nvidia-a100-architecture/table-07.png)

**表 7.** GA100 硬件解码支持

<span id="table-08"></span>

![GPU 加速时钟 (1410 MHz) 下的解码性能](./nvidia-a100-architecture/table-08.png)

**表 8.** GPU 加速时钟 (1410 MHz) 下的解码性能

(以支持的并发流数量衡量)

<span id="table-09"></span>

![1080p30 下 A100 与 V100 的解码对比](./nvidia-a100-architecture/table-09.png)

**表 9.** 1080p30 下 A100 与 V100 的解码对比

(以支持的并发流数量衡量)

视频用例:

- 视频分类/理解
- 边缘平台上的智能视频分析
- 自动驾驶 DL 训练

<span id="section-11"></span>

## 11 面向 NVIDIA Ampere 架构 GPU 的 CUDA 进展

NVIDIA® CUDA® 是 NVIDIA 创建的并行计算平台和编程模型, 让应用开发者可以使用 NVIDIA GPU 的大规模并行处理能力. CUDA 是 GPU 加速深度学习的基础, 也为从天文学、分子动力学模拟到计算金融等众多计算与内存密集型应用提供加速. 数千款 GPU 加速应用构建于 NVIDIA CUDA 并行计算平台之上. CUDA 的灵活性与可编程性, 使其成为研究和部署新型深度学习及并行计算算法的常用平台.

NVIDIA Ampere 架构 GPU 在提高 GPU 可编程性和性能的同时, 也着力降低软件复杂度. NVIDIA Ampere 架构 GPU 与 CUDA 编程模型的进展可以加快程序执行, 并降低许多操作的延迟和开销. CUDA 11 为第三代 Tensor Core、稀疏性功能、[CUDA 图](https://developer.nvidia.com/blog/cuda-graphs/)、多实例 GPU、L2 缓存驻留控制以及 NVIDIA Ampere 架构的多项其他新能力提供编程与 API 支持.

下面几节介绍与 NVIDIA Ampere 架构有关的部分主要 CUDA 进展.

<span id="section-11-1"></span>

### 11.1 CUDA 任务图加速

<span id="section-11-1-1"></span>

#### 11.1.1 CUDA 任务图基础

深度神经网络训练和科学模拟等许多 GPU 密集型应用都具有迭代结构, 会反复执行相同的工作流程. 这类工作流程若使用 CUDA 流, 每次迭代都需要 CPU 将工作重新提交至 GPU, 会同时占用时间和 CPU 资源. CUDA 任务图作为 CUDA 10 的一部分于 2018 年推出, 提供了一种更高效的 GPU 工作提交模型. 任务图由内存复制、内核启动等一系列通过依赖关系连接的操作组成, 其定义与执行相互分离. 任务图支持“一次定义、重复运行”的执行流程. 预定义任务图能够通过一次操作启动任意数量的内核, 大幅提高应用效率和性能.

GPU 上的工作执行分为三个阶段: 启动、网格初始化和内核执行. 特别是对于运行时间很短的 GPU 内核, 这些开销可能占到整体端到端执行时间的很大一部分.

将任务图的定义与执行分开 (任务图会重复执行), 可以显著降低 CPU 内核启动成本. 由于驱动能够看到整个工作流程, 包括执行、数据移动和同步交互, 任务图还使 CUDA 驱动可以执行多项优化, 在多种情况下提高执行性能.

<span id="figure-28"></span>

![顺序执行 2 μs 内核时的执行时间分解](./nvidia-a100-architecture/figure-28.png)

**图 28.** 顺序执行 2 μs 内核时的执行时间分解.

<span id="section-11-1-2"></span>

#### 11.1.2 NVIDIA Ampere 架构 GPU 上的任务图加速

A100 GPU 加速了任务图支持的多项优化. 这些优化分为两类: 启动优化和执行依赖优化. 它们旨在降低内核间延迟和开销; 之所以能够实现, 是因为任务图预先知道整个工作流程. 对含有持续时间很短内核的强扩展工作负载, 这些优化尤其有效, 因为此时开销在运行时间中占很大比例.

启动优化依靠图拓扑识别整个工作流程, 从而高效上传启动和运行工作所需的内核数据. 首先, 图的初次启动能够通过一次操作向 GPU 提交多个工作项. 这会直接大幅降低 CPU 看到的启动开销. 随后, A100 GPU 可利用图中内置的依赖信息, 更高效地将内核信息上传至 SM 执行, 显著降低内核第一条指令开始运行前的延迟.

<span id="figure-29"></span>

![任务图加速对 CPU 启动延迟的影响](./nvidia-a100-architecture/figure-29.png)

**图 29.** 任务图加速对 CPU 启动延迟的影响

执行依赖优化处理工作流程发生分叉后又重新汇合的复杂图. A100 GPU 架构能够沿分叉中的多条依赖关系执行, 自动以尽可能低的延迟运行依赖内核. 对拓扑复杂的图而言, 这会直接显著改善启动延迟和网格间执行延迟.

<span id="figure-30"></span>

![使用 CUDA 图取得的网格间延迟加速](./nvidia-a100-architecture/figure-30.png)

**图 30.** 使用 CUDA 图取得的网格间延迟加速

有关 CUDA 任务图用法的更多细节, 请参阅 [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 以及 CUDA 图 [入门文章](https://developer.nvidia.com/blog/cuda-graphs/).

<span id="section-11-2"></span>

### 11.2 CUDA 异步复制操作

CUDA 11 新增异步复制 API, 用于发挥 A100 GPU 硬件加速的直接复制至共享内存功能. 异步复制会把数据从全局内存直接异步 (非阻塞) 传输至共享内存, 绕过 SM 线程, 并将“从全局内存加载到寄存器”和“从寄存器写入共享内存”两项独立操作合并为一次高效操作.

异步复制无需通过寄存器文件 (RF) 暂存中间数据, 可降低寄存器文件带宽需求. 它还能高效利用内存带宽并减少功耗. 顾名思义, 异步复制以异步方式工作, 从全局内存向共享内存复制期间可以执行其他计算. 复制完成后, 异步复制能够通过 GPU 的新屏障功能通知程序 (见[第 11.3 节](#section-11-3)).

绕过 L1 和寄存器文件可以显著提高内存复制性能, 对连续执行多次异步复制、从全局内存向共享内存复制大量数据的情况尤其如此.

异步复制指令有两种变体, 分别适用于不同场景. BYPASS 如上所述会绕过 L1 缓存和寄存器文件; ACCESS 则会将数据存入 L1, 供后续访问和复用.

<span id="figure-31"></span>

![A100 使用异步复制与不使用异步复制的对比](./nvidia-a100-architecture/figure-31.png)

**图 31.** A100 使用异步复制与不使用异步复制的对比

从用户角度看, 异步复制的行为类似于分别执行全局加载和共享存储指令, 但不会占用线程资源作临时存储. 该指令允许每个线程使用独立的全局内存与共享内存地址. 程序必须使用屏障来保证写入顺序, 并确保同一线程块内各线程可以看到其他线程的加载与存储. [第 11.3 节](#section-11-3) 介绍的每线程异步屏障可以完成这项工作.

<span id="figure-32"></span>

![同步复制与异步复制至共享内存的对比](./nvidia-a100-architecture/figure-32.png)

**图 32.** 同步复制与异步复制至共享内存的对比

<span id="section-11-3"></span>

### 11.3 异步屏障

NVIDIA A100 GPU 在共享内存中提供硬件加速屏障. CUDA 11 以符合 ISO C++ 标准的屏障对象形式提供这些屏障. 异步屏障与普通单阶段屏障不同: 线程通知自己已到达屏障的操作 (“到达”) 与等待其他线程到达屏障的操作 (“等待”) 相互分离. 线程可以在等待期间执行与屏障无关的其他操作, 更充分地利用等待时间, 因而提高执行效率. 异步屏障可用 CUDA 线程实现生产者-消费者模型, 也可按需仅作为单阶段屏障使用.

<span id="figure-33"></span>

![A100 异步屏障](./nvidia-a100-architecture/figure-33.png)

**图 33.** A100 异步屏障

与以往架构中的屏障相比, 新型异步屏障还显著细化了同步粒度, 可以对线程块中任意一组 CUDA 线程进行硬件加速同步. 以往架构只能在线程束整体或线程块整体级别加速同步. 异步复制操作从全局内存向共享内存复制完成后, 可向屏障发出 (“到达”) 信号, 因而屏障可让这种异步复制 ([第 11.2 节](#section-11-2) 所述) 与 SM 中的其他执行重叠. 这样可以隐藏复制延迟并提高效率.

<span id="section-11-4"></span>

### 11.4 L2 缓存驻留控制

CUDA 内核反复访问全局内存中的某个数据区域时, 可以认为这些数据具有“持久性”. 反之, 数据若只访问一次, 则可视为“流式”数据. DL 工作负载尤其依赖持久数据访问.

从 CUDA 11.0 开始, A100 等计算能力 8.0 设备可以影响数据在 L2 缓存中的持久性, 并划出一部分 L2 缓存供持久数据访问使用, 从而以更高带宽和更低延迟访问全局内存.

影响 L2 缓存中数据持久性的能力, 让 A100 GPU 可以更高效地利用 40 MB 大容量 L2 缓存. 例如, 许多 LSTM 网络中的循环权重可以常驻 L2, 并在多次 GEMM 运算间复用. A100 允许以 1/16 (2.5 MB) 为增量划出 L2 缓存供持久访问使用.

持久访问可优先使用 L2 缓存中划出的这部分空间. 对全局内存的普通访问或流式访问, 只有在持久访问未使用这部分空间时才能使用它. 可以通过 CUDA 流或 [CUDA 图](https://developer.nvidia.com/blog/cuda-graphs/) 设置 L2 持久性. 但请注意, GPU 配置为多实例 GPU (MIG) 模式时, 划出 L2 缓存的功能会被禁用.

[CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html) 详细说明了如何为持久访问设置并使用划出的 L2 缓存区域、多个并发执行的 CUDA 内核如何共享这部分 L2 缓存, 以及如何清除并重置该区域以供后续非持久访问使用.

可以通过基于地址范围的窗口管理 L2 缓存中的数据驻留情况, 该窗口指定一个地址范围, 这个范围内的所有读写访问都将持久缓存于 L2 中. 内存操作本身无需注解.

A100 还支持粒度更细的逐内存操作控制, 可分别为每次访问指定 L2 驻留. 基于访问的控制包括按地址哈希作比例分配, 并涵盖生产者-消费者缓冲区等用例. 有关这些功能的具体支持方式, 请参阅 [CUDA 编程指南](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).

<span id="figure-34"></span>

![A100 L2 驻留控制示例](./nvidia-a100-architecture/figure-34.png)

**图 34.** A100 L2 驻留控制示例

使用大量无法装入共享内存的桶计算直方图时, 必须直接在 GPU 全局内存中执行原子操作. 在上方 [图 34](#figure-34) 中, 我们使用含 2.56 亿个整数的数据集, 计算一个有 500 万个整数桶的直方图. 500 万个整数桶占用 20 MB, 因此无法装入共享内存, 但可以装入 GPU 的 L2 缓存. 将直方图桶区域标记为持久后, 相比 V100 可取得 2.5 倍加速; 与未使用驻留控制的 A100 场景相比, 也能加速 43%.

<span id="section-11-5"></span>

### 11.5 Cooperative Groups

Cooperative Groups 扩展了最初在 CUDA 9 中引入的编程模型, 将异步内存复制封装为整个组共同执行的集合操作. 它既利用 A100 对全局内存至共享内存非阻塞复制的硬件加速, 也为反方向复制和早期架构提供阻塞式软件回退方案.

Cooperative Groups 使用组中指定的线程, 以尽可能高效的方式自动分配工作负载, 并推导每个线程正确的对齐方式和数据传输大小. 默认操作相当于单阶段流水线, 同时还提供重载, 可配合 CUDA 新增的内存流水线对象扩展为多阶段流水线.

传输启动后, 调用 wait 即可读取数据; wait 表示流水线已清空, 或对应阶段已将数据移入共享内存.

Cooperative Groups 利用 A100 强大的新型线程束规约指令, 通过 reduce API 扩展其集合操作. 该 API 对传入组中各个指定线程提供的数据执行规约操作. 硬件可加速算术 ADD、MIN 或 MAX 运算, 以及逻辑 AND、OR 或 XOR 运算. 其他类型和运算由软件实现, 老一代硬件上的回退方案也是如此.

协作启动继续为 CUDA 开发者提供实用价值, 我们已将网格同步开销最多降低 30%, 并且在编写协作内核和利用网格组时, 不再需要单独编译.

<span id="figure-35"></span>

![线程束范围规约](./nvidia-a100-architecture/figure-35.png)

**图 35.** 线程束范围规约

<span id="section-12"></span>

## 12 结论

NVIDIA 的使命是加速当代达·芬奇和爱因斯坦们的工作. 科学家、研究人员和工程师正利用高性能计算 (HPC) 与人工智能 (AI), 致力于解决世界上重大的科学、工业和大数据难题. NVIDIA® A100 Tensor Core GPU 推动我们的加速数据中心平台实现下一次重大飞跃, 在各种规模下提供强劲加速, 让这些创新者能在有生之年完成毕生事业. A100 驱动的应用领域包括 HPC、基因组学、5G、渲染、深度学习、数据分析、数据科学和机器人技术.

要推进个性化医疗、对话式 AI 和深度推荐系统等当今重要的 HPC 与 AI 应用, 研究人员必须扩大规模. A100 驱动的 NVIDIA 数据中心平台包含用于纵向扩展的 Mellanox HDR InfiniBand (IB)、NVIDIA NVSwitch、NVIDIA HGX-A100 和 Magnum IO SDK. 这些集成技术可高效扩展至数万块 GPU, 以前所未有的速度训练最复杂的 AI 网络.

A100 GPU 的新型多实例 GPU (MIG) 可将每块 A100 划分为最多七个 GPU 加速器, 以优化利用率, 实际上既提高了 GPU 资源利用率, 也让更多用户和 GPU 加速应用能够使用 GPU. A100 的通用性使基础设施管理员可以充分发挥数据中心内每块 GPU 的价值, 满足从最小作业到最大多节点工作负载的不同性能需求.

<span id="section-13"></span>

## 13 NVIDIA DGX A100

当今企业需要扩展 AI 来推动业务转型, 才能在充满挑战的时期生存并发展. 然而, 大多数企业缺乏大规模投入 AI 实际运营所需的基础设施和知识. 对传统系统和架构的依赖, 使其数据中心在服务器上投入过多、效率低下, 且无法满足训练、推理和分析的独特需求.

<span id="section-13-1"></span>

### 13.1 NVIDIA DGX A100: 通用 AI 基础设施系统

DGX A100 是专为 AI 构建的先进系统的第三代产品. 单套系统可提供前所未有的 5 PFLOPS 性能. DGX A100 为企业数据中心带来新的基础设施构建方式, 旨在用全新的通用平台和架构统一所有 AI 工作负载. 由 A100 和 MIG 驱动的 DGX A100 改变了企业数据中心. 它让架构师能够规划、部署和扩展数据中心, 使用同构基础设施处理异构工作负载并为其优化. 从开发到大规模部署, DGX A100 为 AI 创新者提供完成重要工作所需的能力.

<span id="figure-36"></span>

![NVIDIA DGX A100 系统](./nvidia-a100-architecture/figure-36.png)

**图 36.** NVIDIA DGX A100 系统

连接数千套 DGX A100 系统即可轻松向上扩展性能; 也可将系统中的每块 A100 GPU 划分为七个独立 GPU 实例, 向下扩展并取得很高的效率. 多实例 GPU (MIG) 是 NVIDIA 的突破性技术, 可在一套 DGX A100 中提供多达 56 个独立加速器, 每个加速器都在硬件层面完全隔离且安全, 拥有自己的高带宽内存、缓存和计算核心. MIG 允许用户在同一套系统上并行混合运行多个训练与推理作业, 通过专用资源优化利用率.

DGX A100 运行来自 NVIDIA GPU Cloud (NGC) 的优化软件, 将密集算力与完整的工作负载灵活性结合起来, 因而同样适合单节点部署, 以及使用 NVIDIA DeepOps 部署的大型 Slurm 和 Kubernetes 集群.

<span id="section-13-2"></span>

### 13.2 改变局面的性能

DGX A100 中的八块 NVIDIA A100 GPU 使用高性能的新型第三代 NVLink, 经由六个新 NVSwitch 互连, 双向总带宽达到 4.8 TB/s (全双工 2.4 TB/s). 每块 NVIDIA A100 GPU 都采用支持 TF32 精度和稀疏性的第三代 Tensor Core, 无需更改代码, 性能最高可达 V100 上标准 FP32 FMA 运算的 20 倍. 在 AI 训练中, DGX A100 的性能最高可达基于 V100 的 DGX-1 的 6 倍. DGX A100 还以 6U 机身容纳最高 5 PFLOPS 的 AI 性能, 将计算密度提升到新的水平.

<span id="figure-37"></span>

![DGX A100 为训练与推理带来前所未有的 AI 性能](./nvidia-a100-architecture/figure-37.png)

**图 37.** DGX A100 为训练与推理带来前所未有的 AI 性能.

<span id="section-13-3"></span>

### 13.3 出色的数据中心扩展能力

NVIDIA DGX A100 拥有所有 DGX 系统中最快的 IO 架构, 是 [NVIDIA DGX SuperPOD](https://www.nvidia.com/en-us/data-center/resources/nvidia-dgx-superpod-reference-architecture/) 等大型 AI 集群的基础构建模块, 后者是可扩展 AI 基础设施的企业蓝图. DGX A100 首次采用速度为 PCIe Gen 4 10 倍的下一代 NVLink、新型 NVSwitch, 以及八个各以 200 Gb/s 运行的 Mellanox ConnectX-6 HDR InfiniBand 适配器, 为大规模 AI 工作负载提供高速结构. DGX A100 还支持 Magnum IO 软件 SDK, 可将应用高效扩展至数万块 GPU. 大规模 GPU 加速计算、先进网络硬件与软件优化相结合, 使 NVIDIA DGX A100 可扩展至数百或数千个节点, 应对对话式 AI 和大规模图像分类等重大挑战.

<span id="section-13-4"></span>

### 13.4 全面优化的 DGX 软件栈

DGX A100 软件为大规模运行 AI 工作负载而构建. 其中一项目标是尽可能减少设置工作, 让从业者在 DGX A100 上部署深度学习框架、数据分析和 HPC 应用. 平台软件的设计以服务器上精简的操作系统和驱动程序安装为中心, 所有应用与 SDK 软件均通过 [NGC Private Registry](https://docs.nvidia.com/ngc/ngc-private-registry-user-guide/index.html) 配置.

[NGC Private Registry](https://docs.nvidia.com/ngc/ngc-private-registry-user-guide/index.html) 提供针对 GPU 优化的深度学习 (DL)、机器学习 (ML) 和高性能计算 (HPC) 应用容器, 以及预训练模型、模型脚本、Helm 图表和软件开发工具包 (SDK). 这些软件均在 DGX 系统上开发、测试并调优, 兼容 DGX-1、DGX-2、DGX Station 和 DGX A100 等全部 DGX 产品. NGC Private Registry 还提供安全空间, 用于存储可与企业内其他人共享的自定义容器、模型、模型脚本和 Helm 图表. 可从 [这篇博客文章](https://developer.nvidia.com/blog/securing-and-accelerating-end-to-end-ai-workflows-with-the-ngc-private-registry/) 进一步了解 NGC Private Registry.

[图 38](#figure-38) 展示了这些组成部分如何共同构成 DGX 软件栈.

<span id="figure-38"></span>

![NVIDIA DGX 软件栈](./nvidia-a100-architecture/figure-38.png)

**图 38.** NVIDIA DGX 软件栈

DGX 软件栈包含以下主要组件.

- [NVIDIA CUDA 工具包](https://developer.nvidia.com/cuda-toolkit) 是创建高性能 GPU 加速应用的开发环境. CUDA 11 使软件开发人员和 DevOps 工程师可以利用新款 NVIDIA A100 GPU 的主要创新, 包括:
  - CUDA 线性代数库支持新的输入数据类型格式和性能优化
  - 在 Linux 操作系统上配置和管理 MIG 实例, 它是 DGX 软件栈的一部分

有关新增内容的详细信息, 请阅读 [CUDA 11 特性揭秘开发者博客](https://developer.nvidia.com/blog/cuda-11-features-revealed/).

- NVIDIA Container Toolkit 允许用户构建和运行 GPU 加速的 Docker 容器. 该工具包包含容器运行时库和实用工具, 可自动配置容器以利用 NVIDIA GPU.
- GPU 加速容器包含支持以下应用的软件:
  - 用于训练的深度学习框架, 如 [PyTorch](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)、[MXNet](http://docs.nvidia.com/deeplearning/dgx/mxnet-release-notes/index.html) 和 [TensorFlow](http://docs.nvidia.com/deeplearning/dgx/tensorflow-release-notes/index.html)
  - [TensorRT](https://docs.nvidia.com/deeplearning/frameworks/tf-trt-user-guide/index.html) 等推理平台
  - [RAPIDS](http://rapids.ai/) 等数据分析工具; RAPIDS 是一套完全在 GPU 上执行端到端数据科学与分析流水线的软件库.
  - [CUDA-X HPC](https://developer.nvidia.com/gpu-accelerated-libraries)、[OpenACC](https://developer.nvidia.com/openacc) 和 CUDA® 等高性能计算 (HPC) 工具.

有关 DGX A100 的更多信息, 请参阅博客文章 [《以 DGX A100 定义 AI 创新》](https://developer.nvidia.com/blog/defining-ai-innovation-with-dgx-a100/).

<span id="section-13-5"></span>

### 13.5 NVIDIA DGX A100 系统规格

<span id="table-10"></span>

![NVIDIA DGX A100 系统规格](./nvidia-a100-architecture/table-10.png)

**表 10.** NVIDIA DGX A100 系统规格

<span id="section-14"></span>

## 14 稀疏神经网络入门

GPU 计算性能虽已快速提升以跟上 DNN 日益增长的复杂度, 但科学家仍在研究新技术, 以减少训练这些稠密网络所需的计算、内存和能量, 并缩小训练后网络的体积, 使其可以装入内存较小的边缘设备. 如何将稠密网络剪枝为准确率相当的稀疏网络, 正受到业界和学术界的广泛研究.

<span id="figure-39"></span>

![稠密神经网络](./nvidia-a100-architecture/figure-39.png)

**图 39.** 稠密神经网络

深度神经网络 (DNN) 由若干相互连接的神经元层或节点层组成. 通常, 全连接 DNN 中的每个神经元或节点都会与网络下一层的每个神经元相连. 这意味着, 若网络某一层有 n 个节点, 并与下一层的 n 个节点相连, 两层之间的连接数将为 $n^2$. 过去几年, 新型神经网络的复杂度迅速增长, 形成了包含数十至数百层、数千个神经元和数百万条连接的 DNN.

<span id="section-14-1"></span>

### 14.1 剪枝与稀疏性

简而言之, 剪枝是一种将对网络最终准确率贡献很小或没有贡献的节点与连接置零或移除的技术. 对 AI 训练而言, 这可能意味着将许多接近零值的权重与激活矩阵置零, 再重新训练以优化剩余权重. 对推理而言, 可以把接近零的权重值向下舍入为零, 或从网络中移除接近零值的连接与节点. 换言之, 剪枝后的网络因节点和连接减少而变得稀疏. 许多研究论文探讨了稀疏网络的不同剪枝技术, 可在网上找到以供深入阅读.

利用神经网络的稀疏性可以带来多项性能收益. 第一, 跳过矩阵中的零值元素可以提高计算吞吐量. 第二, 只提取非零元素可以降低内存带宽用量. 第三, 对受延迟限制的推理应用, 从内存提取更多非零值并将其存储于片上, 可以降低延迟.

<span id="section-14-2"></span>

### 14.2 细粒度与粗粒度稀疏性

稀疏性研究大体可分为两类: 细粒度稀疏性研究如何将神经网络中分散的特定权重置零; 粗粒度稀疏性研究如何将神经网络中的整个子网络置零.

<span id="figure-40"></span>

![细粒度稀疏性](./nvidia-a100-architecture/figure-40.png)

**图 40.** 细粒度稀疏性

采用细粒度稀疏性的网络具有相同数量的节点, 但在网络中不规则分布的边会减少. 如 [图 40](#figure-40) 所示, 计算每个节点输出时从内存提取的数据量和所需计算量因节点而异. 这会造成不规则内存访问和负载均衡问题, 降低计算工作负载的并行程度, 进而降低 GPU 计算吞吐量.

按粗粒度稀疏性剪枝的网络 ([图 41](#figure-41)) 会移除整个网络子区域. 这有助于保持工作负载的并行性并提高吞吐量, 但可能造成不可接受的较大准确率损失.

<span id="figure-41"></span>

![粗粒度稀疏性](./nvidia-a100-architecture/figure-41.png)

**图 41.** 粗粒度稀疏性

NVIDIA A100 GPU 支持的细粒度结构化稀疏性允许网络保持稀疏, 但要求每个节点执行相同数量的数据提取与计算. 下方 [图 42](#figure-42) 表示采用细粒度结构化稀疏性的网络, 其第二层和第三层的每个节点都有相同数量的稀疏连接.

<span id="figure-42"></span>

![细粒度结构化稀疏性](./nvidia-a100-architecture/figure-42.png)

**图 42.** 细粒度结构化稀疏性

学术界和 AI 业界已发表大量关于稀疏性的研究. 但尚未形成利用稀疏性在不牺牲准确率的情况下优化计算吞吐量的标准做法. NVIDIA A100 实现的细粒度结构化稀疏性, 配合 NVIDIA 提供的简单通用深度神经网络稀疏化流程, 在多种常用神经网络的评估中几乎不会损失准确率. 下方 [表 11](#table-11) 比较了使用 2:4 稀疏性微调所达到的准确率与采用稠密矩阵训练所达到的准确率.

<span id="table-11"></span>

![多种网络采用 2:4 细粒度结构化稀疏性后取得的准确率](./nvidia-a100-architecture/table-11.png)

**表 11.** 多种网络采用 2:4 细粒度结构化稀疏性后取得的准确率

> 准确的印刷版式和参考文献以原始 PDF 为准.
