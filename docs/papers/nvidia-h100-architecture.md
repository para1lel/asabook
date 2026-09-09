---
title: 'NVIDIA H100 Tensor Core GPU Architecture'
createTime: 2026/09/09 14:56:59
permalink: /papers/nvidia-h100-architecture/
pageClass: paper-reading
---

> [NVIDIA Corporation](https://www.nvidia.com/en-us/about-nvidia/). *[NVIDIA H100 Tensor Core GPU Architecture](https://resources.nvidia.com/en-us-hopper-architecture/nvidia-h100-tensor-c)*, V1.04, 发布于 2023 年 5 月. <a href="/paper/nvidia-h100-architecture.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源码; 准确措辞、印刷版式和参考文献均以已发布的 PDF 为准. 本版包含最终 GPU / 显存时钟频率和最终 TFLOPS 性能规格.

<span id="section-1"></span>

## 1 引言

NVIDIA® 加速计算技术解决了远超普通计算机能力范围的计算难题. 加速计算需要的不只是强大的 GPU. NVIDIA® CUDA® 通用可编程 GPU 与众多 GPU 加速 SDK、API 和算法相结合, 提供全栈计算解决方案, 在多个领域显著加速应用. 分布式 GPU 计算系统和软件可将处理能力扩展到整个数据中心. 全球云数据中心越来越多地采用 NVIDIA GPU 加速系统和架构进行纵向与横向扩展, 运行各种 AI、HPC 和数据分析应用.

15 年多以前, NVIDIA 随 G80 GPU 推出了 CUDA 并行计算平台. 从那时起, CUDA 工具和库的下载量已超过 3000 万次, 使用者接近 300 万名开发者. CUDA 平台持续改进、优化和扩展, 引入性能更强的 CUDA GPU、新的多样化 GPU 加速库、工作站、服务器及应用, 让 NVIDIA 加速计算得到更广泛的应用.

如今, NVIDIA 已为不同产业、科学领域和应用提供全栈解决方案. 超过 450 个 NVIDIA SDK、工具包、库和模型服务于游戏、设计、生命科学、地球科学、机器人、自动驾驶汽车、量子计算、供应链物流、网络安全、5G、气候科学、数字生物学等行业和应用. 目前有超过 25000 家公司使用 NVIDIA AI 技术.

NVIDIA CUDA 平台易于编程且功能丰富, 让设计师、研究人员和工程师能够快速创新. 随着平台软件不断优化, 用户在 NVIDIA 产品的整个生命周期内获得数倍加速也很常见.

NVIDIA GPU 用于全球许多大型数据中心, 可大幅加速 AI、HPC 和数据分析系统及应用. 云数据中心正通过 NVIDIA GPU 快速纵向扩展 AI 训练, 并横向扩展推理应用. 目前, 多种 AI 模型已趋于成熟并实现产业化, 可供企业广泛使用; 这些模型使用 NVIDIA GPU 完成训练并持续改进. 成熟的 AI 模型包括计算机视觉模型、语音识别、推荐系统、图与树、时间序列模型、生成模型、变量编码器和大语言模型. 事实上, 针对新语言和新领域定制大语言模型, 很可能会成为有史以来规模最大的超级计算应用之一.

NVIDIA 新推出的 [Omniverse™ 平台](https://developer.nvidia.com/nvidia-omniverse-platform) 将为众多元宇宙环境提供支持, 并需要庞大的 GPU 计算能力. 除了使用 NVIDIA RTX GPU 为许多支持 Omniverse 的元宇宙提供实时渲染和仿真能力之外, 我们预计, 配备 H100 的系统还会为复杂的数字孪生任务增加 AI 和仿真算力. 规模最大的超级计算项目之一将是 NVIDIA 自己的 [Earth-2 超级计算机项目](https://blogs.nvidia.com/blog/2021/11/12/earth-2-supercomputer/). 它会不断将海量数据流式传入在 Omniverse 中运行的地球数字孪生, 通过物理仿真预测全球未来天气模式.

<span id="figure-01"></span>

![现代云计算中的多样化工作负载](./nvidia-h100-architecture/figure-01.png)

**图 1.** 现代云数据中心工作负载需要 NVIDIA GPU 加速

本白皮书介绍新款 NVIDIA H100 Tensor Core GPU, 这是我们的下一代高性能数据中心 GPU. H100 基于 NVIDIA Hopper GPU 架构, 将加速云数据中心、服务器、边缘系统和工作站中的 AI 训练与推理、HPC 及数据分析应用.

下文首先概述 H100、基于 H100 的新款 DGX、DGX SuperPOD 和 HGX 系统, 以及基于 H100 的新型融合加速器, 随后深入介绍 H100 硬件架构、效率改进和新的编程功能.

<span id="section-2"></span>

## 2 NVIDIA H100 Tensor Core GPU 概述

人工智能 (AI)、高性能计算 (HPC) 和数据分析的复杂度呈指数增长, 科学家和工程师因此需要使用最先进的计算平台. NVIDIA Hopper GPU 架构以低延迟安全地提供高性能计算能力, 并集成了一整套用于数据中心规模计算的功能.

采用 NVIDIA Hopper GPU 架构的 NVIDIA® H100 Tensor Core GPU, 让 NVIDIA 数据中心平台的加速计算性能再次实现大幅跃升. H100 可以安全地加速各种工作负载, 从小型企业工作负载、百亿亿次级 HPC, 到万亿参数 AI 模型.

H100 采用为 NVIDIA 定制的 TSMC 4N 工艺实现, 集成 800 亿个晶体管并包含多项架构进步, 是世界上迄今制造的最先进芯片.

<span id="figure-02"></span>

![安装在 SXM5 模块上的 NVIDIA H100 GPU](./nvidia-h100-architecture/figure-02.png)

**图 2.** 安装在新款 SXM5 模块上的 NVIDIA H100 GPU

H100 是 NVIDIA 第 9 代数据中心 GPU, 旨在让大规模 AI 和 HPC 的性能较上一代 NVIDIA A100 Tensor Core GPU 实现数量级的跃升. H100 延续了 A100 的主要设计重点, 改进 AI 与 HPC 工作负载的强扩展能力, 同时大幅提高架构效率.

对于当今主流的 AI 和 HPC 模型, 采用 InfiniBand 互连的 H100 可提供高达 A100 30 倍的性能 (参见 [图 3](#figure-03)).

新的 NVLink Switch System 互连面向一些规模最大、难度最高的计算工作负载. 这些工作负载必须跨多个 GPU 加速节点执行模型并行才能容纳; 与采用 InfiniBand 的 H100 相比, 该互连再次带来代际性能跃升, 在某些情况下还能再将性能提高至 3 倍.

<span id="figure-03"></span>

![H100 在 HPC、AI 推理和 AI 训练工作负载上的性能](./nvidia-h100-architecture/figure-03.png)

**图 3.** H100 推动下一代 AI 和 HPC 突破. 所有性能数据均为基于当前预期的初步结果, 最终出货产品可能会有变化. A100 集群: HDR IB 网络. H100 集群: NDR IB 网络, 并在注明处采用 NVLink Switch System. 请注意, H100 系统目前尚不提供 NVLink Switch System 技术, 相关系统和可用时间将在未来公布. GPU 数量: 气候建模 1K、LQCD 1K、基因组学 8、3D-FFT 256、MT-NLG 32 (批量大小: 1 秒时 A100 为 4、H100 为 60, 1.5 秒和 2 秒时 A100 为 8、H100 为 64)、MRCNN 8 (批量大小 32)、GPT-3 16B 512 (批量大小 256)、DLRM 128 (批量大小 64K)、GPT-3 16K (批量大小 512)、MoE 8K (批量大小 512, 每个 GPU 一个专家).

在 2022 年春季 GTC 上, NVIDIA 发布了新款 Grace Hopper Superchip 产品. Hopper H100 Tensor Core GPU 将为 NVIDIA Grace Hopper Superchip CPU+GPU 架构提供动力. 该架构专为 TB 级加速计算而打造, 可在大模型 AI 和 HPC 上实现 10 倍性能提升.

NVIDIA Grace CPU 利用 Arm® 架构的灵活性, 从零开始打造面向加速计算的 CPU 和服务器架构. H100 通过 NVIDIA 的超高速芯片间互连与 Grace 配对, 提供 900 GB/s 带宽, 比 PCIe Gen5 快 7 倍. 与当今速度最快的服务器相比, 这项创新设计的总带宽最高可提升至 30 倍; 对运行 TB 级数据的应用, 性能最高可提升至 10 倍.

<span id="figure-04"></span>

![Grace Hopper Superchip](./nvidia-h100-architecture/figure-04.png)

**图 4.** Grace Hopper Superchip

<span id="section-2-1"></span>

### 2.1 NVIDIA H100 GPU 主要特性概览

- 新款流式多处理器 (SM) 在性能和效率方面有多项改进. 主要新特性包括:
  - 新款第四代 Tensor Core 的芯片间性能最高可达 A100 的 6 倍, 其中包括单 SM 加速、更多 SM, 以及 H100 更高的时钟频率. 以单个 SM 计, 对于相同数据类型, Tensor Core 的 MMA (矩阵乘加) 计算速率是 A100 SM 的 2 倍; 使用新的 FP8 数据类型时, 速率是 A100 的 4 倍, 对比基准是上一代 16 位浮点选项. 稀疏性功能利用深度学习网络中的细粒度结构化稀疏, 将标准 Tensor Core 运算的性能提高一倍.
  - 新的 DPX 指令对动态规划算法的加速幅度最高可达 A100 GPU 的 7 倍. 两个例子分别是用于基因组处理的 Smith-Waterman 算法, 以及用于在动态仓库环境中为机器人群寻找最优路线的 Floyd-Warshall 算法.
  - 与 A100 相比, 芯片间 IEEE FP64 和 FP32 处理速率提高至 3 倍, 原因是每个 SM 在相同时钟频率下的性能提高至 2 倍, 加上 H100 拥有更多 SM 和更高时钟频率.
  - 新的线程块集群 (Thread Block Cluster) 功能允许程序在比单个 SM 上的单个线程块更大的粒度上控制局部性. 它为编程层次再增加一级, 将 CUDA 编程模型扩展为线程、线程块、线程块集群和网格. 集群让跨多个 SM 并发运行的多个线程块能够同步, 并协同提取和交换数据.
  - 新的异步执行功能包括新款张量内存加速器 (Tensor Memory Accelerator, TMA), 它可以在全局内存和共享内存之间高效传输大型数据块. TMA 还支持集群内线程块之间的异步复制. 此外还有新的异步事务屏障, 用于执行原子数据移动和同步.
- 新的 Transformer Engine 结合软件和定制的 Hopper Tensor Core 技术, 专为加速 Transformer 模型训练与推理而设计. Transformer Engine 会智能管理并动态选择 FP8 或 16 位计算, 在每一层自动处理 FP8 与 16 位格式之间的重新转换和缩放. 与上一代 A100 相比, 它在大语言模型上可将 AI 训练加速至 9 倍, 将 AI 推理加速至 30 倍.
- HBM3 内存子系统的带宽较上一代提高近 2 倍. H100 SXM5 GPU 是全球首款采用 HBM3 内存的 GPU, 可提供领先同类产品的 3 TB/s 内存带宽.
- 50 MB L2 缓存架构可以缓存模型和数据集的大部分内容, 供重复访问, 从而减少对 HBM3 的访问.
- 第二代多实例 GPU (MIG) 技术相比 A100, 每个 GPU 实例可提供约 3 倍的计算容量和近 2 倍的内存带宽. 现在还首次提供具备 MIG 级可信执行环境 (TEE) 的机密计算能力. 最多支持 7 个独立 GPU 实例, 每个实例都有专用 NVDEC 和 NVJPG 单元. 现在, 每个实例还包含一组可配合 NVIDIA 开发者工具使用的性能监视器.
- 新增的机密计算支持可保护用户数据、防御硬件和软件攻击, 并在虚拟化与 MIG 环境中更好地隔离和保护各个 VM. H100 实现了世界上首款原生机密计算 GPU, 并以完整 PCIe 线路速率将 CPU 的可信执行环境扩展到 GPU.
- 第四代 NVIDIA NVLink® 使全归约操作的带宽提高至 3 倍, 常规带宽较上一代 NVLink 提高 50%, 多 GPU IO 总带宽达到 900 GB/s, 是 PCIe Gen 5 的 7 倍.
- 第三代 NVSwitch 技术包括位于节点内部和外部的交换机, 用于连接服务器、集群和数据中心环境中的多个 GPU. 节点内的每个 NVSwitch 提供 64 个第四代 NVLink 端口, 可加速多 GPU 连接. 交换机总吞吐量从上一代的 7.2 Tbit/s 增至 13.6 Tbit/s. 新的第三代 NVSwitch 技术还通过组播和 NVIDIA SHARP 网络内归约, 为集合操作提供硬件加速.
- 新的 NVLink Switch System 互连技术和基于第三代 NVSwitch 技术的新型第二级 NVLink 交换机引入地址空间隔离与保护, 可通过 2:1 渐缩胖树拓扑, 使用 NVLink 连接最多 32 个节点或 256 个 GPU. 这些相连节点可以提供 57.6 TB/s 的全互连带宽, 并能提供 1 exaFLOP 的 FP8 稀疏 AI 计算性能.
- PCIe Gen 5 提供 128 GB/s 总带宽 (每个方向 64 GB/s), 而 Gen 4 PCIe 为 64 GB/s 总带宽 (每个方向 32 GB/s). PCIe Gen 5 让 H100 可以与最高性能的 x86 CPU 和 SmartNIC / DPU (数据处理单元) 连接.

此外还包含许多新特性, 用于改进强扩展能力、降低延迟和开销, 并整体简化 GPU 编程.

本白皮书在 [第 3 节](#section-3) 中介绍了基于 H100 的新款 DGX、HGX、融合加速器和 AI 超级计算系统.

[第 4 节](#section-4) 详细介绍 H100 GPU 架构特性、新的编程能力和性能改进.

<span id="figure-05"></span>

![Hopper H100 的六项新技术](./nvidia-h100-architecture/figure-05.png)

**图 5.** Hopper H100 的新技术

<span id="section-3"></span>

## 3 NVIDIA GPU 加速数据中心

从 AI 和数据分析到高性能计算 (HPC), 数据中心是解决若干重大难题的关键. NVIDIA 端到端加速计算平台实现软硬件集成, 为企业提供稳健、安全的基础设施蓝图, 支持各类现代工作负载从开发到部署的完整实施过程.

深度学习数据集的规模越来越大, 结构也越来越复杂; 对话式 AI、推荐系统和计算机视觉等工作负载在各行业日益普及. NVIDIA 数据中心平台涵盖硬件和软件, 可显著加速 AI 训练, 让数据科学团队获得很高的生产效率, 大幅节省成本, 并更快实现投资回报.

要加速数据中心的推理工作负载, 需要能够横向扩展并充分利用所有可用计算资源的灵活弹性基础设施. 借助多实例 GPU (MIG) 等新技术, NVIDIA 解决方案尤其适合加速图像识别、推荐系统和自然语言处理等推理工作负载, 可提供将 AI 引入应用所需的高吞吐量和实时响应能力.

HPC 是推动数据中心科学进步的主要工具之一. NVIDIA GPU 是现代 HPC 数据中心的引擎. NVIDIA 数据中心平台以更少的服务器提供突破性性能, 从而更快获得洞见并大幅降低成本, 为科学发现提供助力.

企业正在生成和收集前所未有的海量数据. 可供分析的数据越多, 能从中了解的信息就越多. 借助 NVIDIA 数据中心平台和分析解决方案, 企业能以前所未有的速度从数据中获得可操作的洞见.

NVIDIA 庞大的服务器制造商合作伙伴生态提供多种服务器, 可为数据中心提供 NVIDIA GPU 加速. H100 GPU 有多种配置, 可满足不同服务器设计的要求.

以下各节简要介绍面向 NVIDIA 数据中心的 H100 系统和板卡, 包括采用 SMX5 和 PCIe Gen 5 外形规格的 H100 GPU、DGX H100 和 DGX SuperPOD 系统、HGX H100, 以及 H100 CNX 融合加速器. 后者将 NVIDIA H100 GPU 的强大计算能力与 NVIDIA® ConnectX-7 SmartNIC 的先进网络能力结合起来. 有关 DGX H100 系统的更多详情, 请参见 [第 7 节](#section-7).

<span id="section-3-1"></span>

### 3.1 H100 SXM5 GPU

H100 SXM5 配置采用 NVIDIA 定制的 SXM5 板卡, 板上装有 H100 GPU 和 HBM3 内存堆栈, 并提供第四代 NVLink 和 PCIe Gen 5 连接, 因而能够提供最高的应用性能. 这种配置非常适合应用需要在一台服务器内扩展到多个 GPU, 并继续跨服务器扩展的客户. 它通过 HGX H100 服务器板卡提供, 有 4 GPU 和 8 GPU 两种配置. 4 GPU 配置在 GPU 之间采用点对点 NVLink 连接, 并在服务器中提供更高的 CPU 与 GPU 比例; 8 GPU 配置则包含 NVSwitch, 可提供 SHARP 网络内归约, 并在任意一对 GPU 之间提供 900 GB/s 的完整 NVLink 带宽. 强大的新款 DGX H100 服务器和 DGX SuperPOD 系统也采用 H100 SXM5 GPU.

<span id="section-3-2"></span>

### 3.2 H100 PCIe Gen 5 GPU

H100 PCIe Gen 5 配置仅需 350 W 热设计功耗 (TDP), 即可提供 H100 SXM5 GPU 的全部能力. 该配置可选择使用 NVLink 桥接器, 以 600 GB/s 带宽连接最多两个 GPU, 其速度接近 PCIe Gen5 的 5 倍. H100 PCIe 很适合装入标准机架、每台服务器功耗较低的主流加速服务器, 可为一次扩展到 1 个或 2 个 GPU 的应用提供出色性能, 包括 AI 推理和部分 HPC 应用. 对 10 个领先的数据分析、AI 和 HPC 应用进行综合测试时, 单个 H100 PCIe GPU 仅消耗 50% 的功率, 即可高效提供 H100 SXM5 GPU 65% 的实际性能.

<span id="section-3-3"></span>

### 3.3 DGX H100 与 DGX SuperPOD

NVIDIA DGX H100 是一套适用于训练、推理和分析的通用高性能 AI 系统. DGX H100 配备 Bluefield-3、NDR InfiniBand 和第二代 MIG 技术. 单台 DGX H100 系统可提供无可比拟的 16 petaFLOPS FP16 稀疏 AI 计算性能. 将多台 DGX H100 系统连接成称为 DGX POD 乃至 DGX SuperPOD 的集群, 即可轻松扩展这项性能. DGX SuperPOD 从 32 台 DGX H100 系统起步, 这组系统称为一个"可扩展单元". 它集成 256 个 H100 GPU, 通过基于第三代 NVSwitch 技术的新型第二级 NVLink 交换机连接, 可提供前所未有的 1 exaFLOP FP8 稀疏 AI 计算性能. DGX H100 SuperPOD 将支持 InfiniBand 和 NVLINK Switch 两种网络选项.

更多详情请参见 [第 7 节](#section-7).

<span id="section-3-4"></span>

### 3.4 HGX H100

随着工作负载复杂度激增, 多个 GPU 必须协同工作并在彼此之间进行极高速通信. NVIDIA HGX H100™ 将多个 H100 GPU 与 NVLink 和 NVSwitch 支持的高速互连相结合, 可用于打造世界上性能最强的纵向扩展服务器.

HGX H100 以集成底板的形式作为服务器构建模块提供, 有四个或八个 H100 GPU 两种配置. 四 GPU HGX H100 在 GPU 之间提供完全互连的点对点 NVLink 连接, 八 GPU 配置则通过 NVSwitch 提供完整的 GPU 间带宽. 8 路 HGX H100 利用 H100 多精度 Tensor Core 的计算能力, 通过稀疏 FP8 运算可提供超过 32 petaFLOPS 的深度学习计算性能. HGX H100 让标准化高性能服务器可以在不同应用工作负载上提供可预测的性能, 同时帮助 NVIDIA 服务器制造商合作伙伴生态更快将产品推向市场.

<span id="section-3-5"></span>

### 3.5 H100 CNX 融合加速器

NVIDIA H100 CNX 将 NVIDIA H100 GPU 的强大计算能力与 NVIDIA® ConnectX-7 SmartNIC 的先进网络能力结合起来. ConnectX-7 SmartNIC 可提供最高 400 Gb/s 带宽, 并包含 NVIDIA ASAP2 (加速交换与数据包处理), 以及 TLS/IPsec/MACsec 加密与解密的内联硬件加速等创新功能. 这种独特架构可为 GPU 驱动的 I/O 密集型工作负载提供前所未有的性能, 例如企业数据中心中的分布式 AI 训练, 或边缘端的 5G 信号处理.

<span id="section-4"></span>

## 4 NVIDIA H100 GPU 架构详解

基于全新 Hopper GPU 架构的 NVIDIA H100 GPU 包含多项创新:

- 新款第四代 Tensor Core 在更多种类的 AI 与 HPC 任务上, 能以前所未有的速度执行矩阵计算.
- 新的 Transformer Engine 让 H100 在大语言模型上可提供高达上一代 A100 9 倍的 AI 训练加速和 30 倍的 AI 推理加速.
- 新的 NVLink Network 互连支持最多 256 个 GPU 跨多个计算节点进行 GPU 间通信.
- Secure MIG 将 GPU 划分为隔离、大小适当的实例, 为较小的工作负载提供最高的 QoS (服务质量).

NVIDIA H100 是首款真正实现异步执行的 GPU. H100 将 A100 从全局内存到共享内存的异步传输扩展到所有地址空间, 并增加对张量内存访问模式的支持. 应用因此可以构建端到端异步流水线, 将数据移入和移出芯片, 让数据移动与计算完全重叠, 从而隐藏数据移动开销.

现在只需少量 CUDA 线程, 便可通过新的张量内存加速器管理 H100 的完整内存带宽; 其他大多数 CUDA 线程则可以专注于通用计算, 例如为新一代 Tensor Core 预处理和后处理数据.

H100 在线程块层次中增加了名为线程块集群的新层级, 扩展了 CUDA 线程组层次结构. 集群是一组可保证并发调度的线程块, 让跨多个 SM 的线程可以高效协作和共享数据. 集群还可以更高效地协同驱动张量内存加速器和 Tensor Core 等异步单元.

协调越来越多的片上加速器和各种通用线程组需要同步. 例如, 使用输出的线程和加速器必须等待生成这些输出的线程和加速器.

NVIDIA 异步事务屏障让集群内的通用 CUDA 线程和片上加速器能够高效同步, 即使它们位于不同 SM 上也不例外. 这些新特性让每位用户和每个应用都能始终充分利用 H100 GPU 的所有单元, 使 H100 成为迄今性能最强、可编程性最高且能效最佳的 GPU.

为 H100 GPU 提供核心能力的完整 GH100 GPU 采用面向 NVIDIA 定制的 TSMC 4N 工艺制造, 集成 800 亿个晶体管, 裸片面积为 814 mm², 并采用更高频率的设计.

NVIDIA GH100 GPU 由多个 GPU 处理集群 (GPC)、纹理处理集群 (TPC)、流式多处理器 (SM)、L2 缓存和 HBM3 内存控制器组成.

完整 GH100 GPU 实现包含以下单元:

- 8 个 GPC、72 个 TPC (每个 GPC 9 个 TPC)、每个 TPC 2 个 SM, 完整 GPU 共 144 个 SM
- 每个 SM 128 个 FP32 CUDA Core, 完整 GPU 共 18432 个 FP32 CUDA Core
- 每个 SM 4 个第四代 Tensor Core, 完整 GPU 共 576 个
- 6 个 HBM3 或 HBM2e 堆栈、12 个 512 位内存控制器
- 60 MB L2 缓存
- 第四代 NVLink 和 PCIe Gen 5

采用 SXM5 板卡外形规格的 NVIDIA H100 GPU 包含以下单元:

- 8 个 GPC、66 个 TPC、每个 TPC 2 个 SM, 每个 GPU 共 132 个 SM
- 每个 SM 128 个 FP32 CUDA Core, 每个 GPU 共 16896 个 FP32 CUDA Core
- 每个 SM 4 个第四代 Tensor Core, 每个 GPU 共 528 个
- 80 GB HBM3、5 个 HBM3 堆栈、10 个 512 位内存控制器
- 50 MB L2 缓存
- 第四代 NVLink 和 PCIe Gen 5

采用 PCIe Gen 5 板卡外形规格的 NVIDIA H100 GPU 包含以下单元:

- 7 个或 8 个 GPC、57 个 TPC、每个 TPC 2 个 SM, 每个 GPU 共 114 个 SM
- 每个 SM 128 个 FP32 CUDA Core, 每个 GPU 共 14592 个 FP32 CUDA Core
- 每个 SM 4 个第四代 Tensor Core, 每个 GPU 共 456 个
- 80 GB HBM2e、5 个 HBM2e 堆栈、10 个 512 位内存控制器
- 50 MB L2 缓存
- 第四代 NVLink 和 PCIe Gen 5

采用 TSMC 4N 制造工艺, H100 可以提高 GPU 核心频率和每瓦性能, 并集成比上一代 GA100 GPU 更多的 GPC、TPC 和 SM. GA100 GPU 采用的是 TSMC 7 nm N7 工艺.

[图 6](#figure-06) 展示了拥有 144 个 SM 的完整 GH100 GPU. H100 SXM5 GPU 有 132 个 SM, PCIe 版本有 114 个 SM. 请注意, H100 GPU 主要用于执行 AI、HPC 和数据分析的数据中心及边缘计算工作负载, 而非图形处理. SXM5 和 PCIe H100 GPU 都只有两个 TPC 具备图形处理能力 (即可以运行顶点、几何和像素着色器).

<span id="figure-06"></span>

![包含 144 个 SM 的完整 GH100 GPU 框图](./nvidia-h100-architecture/figure-06.png)

**图 6.** 包含 144 个 SM 的完整 GH100 GPU

<span id="section-4-1"></span>

### 4.1 H100 SM 架构

H100 SM 以 NVIDIA A100 Tensor Core GPU SM 架构为基础, 由于引入 FP8, 每个 SM 的峰值浮点计算能力达到 A100 的 4 倍; 在相同时钟频率下, 对此前所有 Tensor Core 和 FP32 / FP64 数据类型的原始 SM 计算能力则达到 A100 的 2 倍.

新款 Transformer Engine 与 Hopper FP8 Tensor Core 相结合, 在大语言模型上可提供高达上一代 A100 9 倍的 AI 训练加速和 30 倍的 AI 推理加速. Hopper 的新 DPX 指令可将基因组与蛋白质测序所用 Smith-Waterman 算法的处理速度提高至 7 倍.

Hopper 新款第四代 Tensor Core、张量内存加速器, 以及其他多项 SM 和 H100 通用架构改进结合起来, 在许多其他情况下可提供高达 3 倍的 HPC 和 AI 性能.

<span id="table-01"></span>

![NVIDIA H100 Tensor Core GPU 性能规格](./nvidia-h100-architecture/table-01.png)

**表 1.** NVIDIA H100 Tensor Core GPU 性能规格

<span id="figure-07"></span>

![GH100 流式多处理器框图](./nvidia-h100-architecture/figure-07.png)

**图 7.** GH100 流式多处理器 (SM)

<span id="section-4-1-1"></span>

#### 4.1.1 H100 SM 主要特性概览

- 第四代 Tensor Core:
  - 与 A100 相比, 芯片间性能最高达到 6 倍, 其中包括单 SM 加速、更多 SM, 以及 H100 更高的时钟频率.
  - 以单个 SM 计, 对于相同数据类型, Tensor Core 的 MMA (矩阵乘加) 计算速率是 A100 SM 的 2 倍; 使用新的 FP8 数据类型时, 速率是 A100 的 4 倍, 对比基准是上一代 16 位浮点选项.
  - 稀疏性功能利用深度学习网络中的细粒度结构化稀疏, 将标准 Tensor Core 运算的性能提高一倍.
- 新的 DPX 指令对动态规划算法的加速幅度最高可达 A100 GPU 的 7 倍. 两个例子分别是用于基因组处理的 Smith-Waterman 算法, 以及用于在动态仓库环境中为机器人群寻找最优路线的 Floyd-Warshall 算法.
- 与 A100 相比, 芯片间 IEEE FP64 和 FP32 处理速率提高至 3 倍, 原因是每个 SM 在相同时钟频率下的性能提高至 2 倍, 加上 H100 拥有更多 SM 和更高时钟频率.
- 共享内存与 L1 数据缓存合计 256 KB, 比 A100 大 1.33 倍.
- 新的异步执行功能包括新款张量内存加速器 (TMA), 它可以在全局内存和共享内存之间高效传输大型数据块. TMA 还支持集群内线程块之间的异步复制. 此外还有新的异步事务屏障, 用于执行原子数据移动和同步.
- 新的线程块集群功能开放了跨多个 SM 的局部性控制.
- 分布式共享内存允许通过加载、存储和原子操作, 在多个 SM 的共享内存块之间直接进行 SM 间通信.

<span id="section-4-1-2"></span>

#### 4.1.2 H100 Tensor Core 架构

Tensor Core 是专用于矩阵乘加 (MMA) 数学运算的高性能计算核心, 能为 AI 和 HPC 应用提供突破性的性能. 一块 NVIDIA GPU 中的多个 Tensor Core 跨 SM 并行运行, 与标准浮点 (FP)、整数 (INT) 和 FMA (融合乘加) 运算相比, 吞吐量和效率均大幅提高. Tensor Core 最早在 NVIDIA Tesla® V100 GPU 中引入, 此后每一代 NVIDIA GPU 新架构都对其作了进一步增强.

与 A100 相比, H100 的新款第四代 Tensor Core 架构在相同时钟频率下, 每个 SM 的原始稠密和稀疏矩阵数学吞吐量均提高至 2 倍; 如果再考虑 H100 高于 A100 的 GPU Boost 时钟频率, 提升幅度还会更大. 它支持 FP8、FP16、BF16、TF32、FP64 和 INT8 MMA 数据类型. 新款 Tensor Core 的数据管理效率也更高, 可节省最高 30% 的操作数传输功耗.

<span id="figure-08"></span>

![A100 与 H100 FP16 Tensor Core 结构](./nvidia-h100-architecture/figure-08.png)

**图 8.** H100 FP16 Tensor Core 的吞吐量是 A100 FP16 Tensor Core 的 3 倍

<span id="section-4-1-3"></span>

#### 4.1.3 Hopper FP8 数据格式

H100 GPU 增加了 FP8 Tensor Core, 用于加速 AI 训练和推理. 如 [图 9](#figure-09) 所示, FP8 Tensor Core 支持 FP32 和 FP16 累加器, 以及两种新的 FP8 输入类型:

- E4M3: 4 个指数位、3 个尾数位和 1 个符号位
- E5M2: 5 个指数位、2 个尾数位和 1 个符号位.

E4M3 为动态范围要求较低的计算提供更高精度, E5M2 则提供更大的动态范围和较低精度. 与 FP16 或 BF16 相比, FP8 将数据存储需求减半, 并将吞吐量提高一倍.

新的 Transformer Engine (将在下文介绍) 同时使用 FP8 和 FP16 精度, 可减少内存用量并提高性能, 同时仍能维持大语言模型和其他模型的准确度.

<span id="figure-09"></span>

![Hopper FP8 精度格式和累加器类型](./nvidia-h100-architecture/figure-09.png)

**图 9.** 新款 Hopper FP8 精度格式 - H100 FP16 / BF16 两倍的吞吐量和一半的存储占用

<span id="figure-10"></span>

![A100 FP16 与 H100 FP8 Tensor Core 结构](./nvidia-h100-architecture/figure-10.png)

**图 10.** H100 FP8 Tensor Core 的吞吐量是 A100 FP16 Tensor Core 的 6 倍

<span id="figure-11"></span>

![H100 TF32、FP64 和 INT8 Tensor Core 吞吐量比较](./nvidia-h100-architecture/figure-11.png)

**图 11.** H100 TF32、FP64 和 INT8 Tensor Core 的吞吐量均为 A100 的 3 倍

H100 对多种数据类型的数学运算相对 A100 的加速幅度列于下方 [表 2](#table-02).

<span id="table-02"></span>

![H100 对多种数据类型相对 A100 的加速幅度](./nvidia-h100-architecture/table-02.png)

**表 2.** H100 相对 A100 的加速幅度 (H100 性能, TC=Tensor Core)

<span id="section-4-1-4"></span>

#### 4.1.4 用于加速动态规划的新 DPX 指令

许多"暴力"优化算法都有这样一种性质: 求解较大问题时会多次复用子问题的解. 动态规划是一种算法技术, 它将复杂的递归问题分解为较简单的子问题来求解. 动态规划算法存储子问题的结果, 以后需要时不必重新计算, 从而将指数规模问题集的计算复杂度降至线性规模.

动态规划广泛用于各种优化、数据处理和基因组学算法. 在快速发展的基因组测序领域, Smith-Waterman 动态规划算法是最重要的方法之一. 在机器人领域, Floyd-Warshall 是一项关键算法, 用于在动态仓库环境中实时为机器人群寻找最优路线.

H100 引入 DPX 指令, 与 Ampere GPU 相比, 可将动态规划算法的性能加速至 7 倍. 这些新指令为许多 DP 算法的内层循环提供高级融合操作数支持. 这会显著缩短疾病诊断、物流路线优化乃至图分析得出结果的时间.

<span id="figure-12"></span>

![DPX 指令的用途与加速效果](./nvidia-h100-architecture/figure-12.png)

**图 12.** DPX 指令加速动态规划

<span id="section-4-1-5"></span>

#### 4.1.5 合并的 L1 数据缓存与共享内存

NVIDIA 合并 L1 数据缓存与共享内存子系统的架构最早随 Volta V100 引入. 它显著提高了性能, 同时简化编程, 并减少应用达到或接近峰值性能所需的调优. 将数据缓存和共享内存功能合并到同一个内存块中, 能为两种内存访问提供最佳的综合性能. H100 的 L1 数据缓存与共享内存合计容量为 256 KB/SM, A100 则为 192 KB/SM. H100 的 SM 共享内存自身最大可配置为 228 KB.

<span id="section-4-1-6"></span>

#### 4.1.6 H100 计算性能总结

将 H100 中所有新的计算技术进步都考虑在内, H100 的整体计算性能约为 A100 的 6 倍. [图 13](#figure-13) 以层层叠加的方式总结了 H100 的改进: 首先, H100 拥有 132 个 SM, 比 A100 的 108 个多 22%. 每个 H100 SM 得益于新的第四代 Tensor Core, 速度提高至 2 倍. 在每个 Tensor Core 内, 新的 FP8 格式和相应的 Transformer Engine 又带来 2 倍提升. 最后, H100 更高的时钟频率又使性能提高约 1.3 倍. 这些改进合计让 H100 的峰值计算吞吐量约为 A100 的 6 倍, 对全球计算需求最庞大的工作负载而言是一次重大跃升.

<span id="figure-13"></span>

![H100 相对 A100 层层叠加的计算能力改进](./nvidia-h100-architecture/figure-13.png)

**图 13.** H100 计算能力改进总结. H100 为全球计算需求最庞大的工作负载提供 6 倍吞吐量.

<span id="section-4-2"></span>

### 4.2 H100 GPU 层次结构与异步性改进

在并行程序中实现高性能的两个基本要点是数据局部性和异步执行. 程序员将程序数据尽可能移近执行单元, 便可利用低延迟、高带宽本地数据访问所带来的性能. 异步执行需要找出可与内存传输及其他处理重叠的独立任务, 目标是让 GPU 中所有单元都得到充分利用. 我们将探讨 Hopper 为 GPU 编程层次新增的一项重要层级, 它开放了比单个 SM 上单个线程块更大规模的局部性. 我们还会介绍新的异步执行功能, 它们可以提高性能并减少同步开销.

<span id="section-4-2-1"></span>

#### 4.2.1 线程块集群

CUDA 编程模型长期依赖这样一种 GPU 计算架构: 使用包含多个线程块的网格来利用程序中的局部性. 一个线程块包含多个在单个 SM 上并发运行的线程, 这些线程可通过快速屏障进行同步, 并使用 SM 的共享内存交换数据. 然而, 随着 GPU 的规模超过 100 个 SM、计算程序越来越复杂, 如果编程模型中只有线程块这一种局部性单元, 已无法最大限度地提高执行效率.

H100 引入新的线程块集群架构, 在比单个 SM 上的单个线程块更大的粒度上开放局部性控制. 线程块集群扩展了 CUDA 编程模型, 为 GPU 的物理编程层次再增加一级, 使其包含线程、线程块、线程块集群和网格. 集群是一组可保证并发调度到一组 SM 上的线程块, 目的是让跨多个 SM 的线程能够高效协作.

H100 中的集群在一个 GPC 内跨 SM 并发运行. GPC 是硬件层次中一组始终在物理位置上彼此接近的 SM. 集群具有硬件加速屏障, 以及后续各节将讨论的全新内存访问协作能力. GPC 中的 SM 配有专用 SM 间网络, 让集群中的线程能够快速共享数据. 在 CUDA 中, 如 [图 14](#figure-14) 所示, 网格中的线程块可以在内核启动时选择性地组成集群; 集群能力可通过 CUDA [cooperative_groups API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cooperative-groups) 使用.

<span id="figure-14"></span>

![传统网格与采用线程块集群的 H100 网格](./nvidia-h100-architecture/figure-14.png)

**图 14.** 线程块集群与包含集群的网格. 在 A100 等传统 CUDA 编程模型中, 网格由线程块组成, 如上图左半部分所示. Hopper 架构增加了可选的集群层次, 如图右半部分所示.

<span id="section-4-2-2"></span>

#### 4.2.2 分布式共享内存

有了集群, 所有线程便可通过加载、存储和原子操作直接访问其他 SM 的共享内存. 此功能称为分布式共享内存 (DSMEM), 因为共享内存的虚拟地址空间在逻辑上分布于集群中的所有线程块. DSMEM 能够更高效地在 SM 之间交换数据, 不再需要先将数据写入全局内存再读出, 以此传递数据. 集群专用 SM 间网络可确保快速、低延迟地访问远程 DSMEM. 与使用全局内存相比, DSMEM 可将线程块之间的数据交换加速约 7 倍.

<span id="figure-15"></span>

![A100 与 H100 上的线程块数据交换](./nvidia-h100-architecture/figure-15.png)

**图 15.** 线程块之间的数据交换 (A100 与采用集群的 H100 对比)

在 CUDA 层面, 集群中所有线程块的全部 DSMEM 段都会映射到每个线程的通用地址空间, 因此所有 DSMEM 都可以用普通指针直接引用. CUDA 用户可以利用 cooperative_groups API 构造指向集群中任意线程块的通用指针. DSMEM 传输也可以表示为异步复制操作, 并通过基于共享内存的屏障来跟踪完成状态和进行同步.

下方的 [图 16](#figure-16) 展示了在不同算法上使用集群的性能优势. 集群让程序员能直接控制比单个 SM 更大的 GPU 区域, 从而提高性能. 集群支持更多线程协同执行, 并能访问比单个线程块更大的共享内存池.

<span id="figure-16"></span>

![使用集群与不使用集群的性能对比](./nvidia-h100-architecture/figure-16.png)

**图 16.** 使用集群与不使用集群的性能比较. H100 的初步性能估计基于当前预期, 最终出货产品可能会有变化.

<span id="section-4-2-3"></span>

#### 4.2.3 异步执行

每一代 NVIDIA 新 GPU 都包含多项架构增强, 用于改进性能、可编程性、能效、GPU 利用率等方面. 近几代 NVIDIA GPU 已包含异步执行能力, 让数据移动、计算和同步可以更多地重叠. Hopper 架构提供新的功能来改进异步执行, 让内存复制可以进一步与计算及其他独立工作重叠, 同时尽量减少同步点.

下面将介绍名为张量内存加速器 (TMA) 的新型异步内存复制单元, 以及新的异步事务屏障.

<span id="figure-17"></span>

![Hopper 中异步执行的并发性](./nvidia-h100-architecture/figure-17.png)

**图 17.** Hopper 中异步执行的并发性与增强功能. 以编程方式重叠数据移动、计算和同步. 异步并发和尽量减少同步点是实现高性能的要点.

<span id="section-4-2-4"></span>

#### 4.2.4 张量内存加速器 (TMA)

为满足性能强大的新款 H100 Tensor Core 的数据需求, 新增的张量内存加速器 (TMA) 提高了数据提取效率. 它可以在全局内存与共享内存之间双向传输大型数据块和多维张量.

TMA 操作使用复制描述符启动; 它通过张量维度和块坐标而非逐元素寻址来指定数据传输 (参见下方 [图 18](#figure-18)). 可以指定大型数据块 (最大为共享内存容量), 将其从全局内存加载到共享内存, 或从共享内存存回全局内存. TMA 支持不同张量布局 (1D-5D 张量)、不同内存访问模式、归约等功能, 可显著减少寻址开销并提高效率.

<span id="figure-18"></span>

![TMA 通过复制描述符生成地址](./nvidia-h100-architecture/figure-18.png)

**图 18.** TMA 通过复制描述符生成地址

TMA 操作是异步的, 并利用 A100 中引入的基于共享内存的异步屏障. 此外, TMA 采用单线程编程模型: 从一个 warp 中选出单个线程, 由它发出异步 TMA 操作 ([`cuda::memcpy_async`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#async_data_operations))来复制张量; 随后, 多个线程可以等待 [`cuda::barrier`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#aw-barrier), 直至数据传输完成. 为了进一步提高性能, H100 SM 增加了用于加速这些异步屏障等待操作的硬件.

TMA 的一项主要优点是能够释放线程, 让它们执行其他独立工作. 在 [图 19](#figure-19) 左侧的 A100 上, 异步内存复制通过一种特殊的 LoadGlobalStoreShared 指令执行, 因此线程要负责生成所有地址, 并循环遍历整个复制区域.

在 Hopper 上, TMA 会接管一切. 单个线程在启动 TMA 前创建复制描述符, 此后的地址生成和数据移动由硬件负责. TMA 在复制张量片段时会接管步长、偏移量和边界的计算, 因此提供了更简单的编程模型.

<span id="figure-19"></span>

![H100 上采用 TMA 与 A100 上采用 LDGSTS 的异步内存复制](./nvidia-h100-architecture/figure-19.png)

**图 19.** H100 上采用 TMA 与 A100 上采用 LDGSTS 的异步内存复制对比

<span id="section-4-2-5"></span>

#### 4.2.5 异步事务屏障

异步屏障最早在 Ampere GPU 架构中引入. 请参见 [图 20](#figure-20) 左侧. 考虑这样一个例子: 一组线程正在生成数据, 并会在越过屏障后共同使用这些数据. 异步屏障将同步过程分为两个步骤. 首先, 每个线程完成自己负责的共享数据后发出"到达"信号. "到达"不会阻塞线程, 因此线程可继续执行其他独立工作.

这些线程最终需要其他所有线程生成的数据. 此时, 它们会执行"等待", 这会阻塞线程, 直至所有线程都发出"到达"信号.

异步屏障的优点是, 提前到达的线程可以在等待时执行独立工作. 这种重叠可以带来额外性能. 如果所有线程都有足够多的独立工作, 屏障实际上就会变得"免费", 因为所有线程都已经到达, 等待指令可以立即退役.

Hopper 的一项新能力是让"等待"中的线程休眠, 直至其他所有线程均已到达. 在此前的芯片上, 等待中的线程会轮询共享内存中的屏障对象.

异步屏障仍是 Hopper 编程模型的一部分, 同时 Hopper 还增加了一种称为异步事务屏障的新屏障形式. 异步事务屏障与异步屏障非常相似. 请参见 [图 20](#figure-20) 右侧. 它同样是分离式屏障, 但除了统计到达的线程, 还会统计事务. Hopper 包含一条用于写入共享内存的新命令, 它会同时传入待写数据和事务计数. 事务计数本质上是字节计数. 在所有生产者线程均已执行"到达", 且全部事务计数之和达到预期值之前, 异步事务屏障会在"等待"命令处阻塞线程.

异步事务屏障是一种强大的新原语, 可用于异步内存复制或数据交换. 如前所述, 集群可以通过隐含同步完成线程块间通信以交换数据, 这种集群能力便构建在异步事务屏障之上.

<span id="figure-20"></span>

![A100 异步屏障与 H100 异步事务屏障](./nvidia-h100-architecture/figure-20.png)

**图 20.** A100 中的异步屏障与 H100 中的异步事务屏障对比

<span id="section-4-3"></span>

### 4.3 H100 HBM 与 L2 缓存架构

GPU 内存架构和层次的设计对应用性能至关重要, 也会影响 GPU 尺寸、成本、功耗和可编程性. GPU 中存在许多不同的内存子系统, 从大量芯片外 DRAM (帧缓冲区) 设备内存, 到不同层级和类型的片上内存, 再到 SM 中用于计算的寄存器文件.

H100 SXM5 和 PCIe H100 GPU 分别采用高性能 HBM3 与 HBM2e DRAM 技术. HBM 内存由与 GPU 位于同一个物理封装上的内存堆栈组成. 与传统 GDDR5/6 内存设计相比, 它能大幅节省功耗和面积, 让系统可以安装更多 GPU.

CUDA 程序访问的全局和局部内存区域位于 HBM 内存空间中, 在 CUDA 术语中称为"设备内存". 常量内存空间位于设备内存中, 并由常量缓存进行缓存. 纹理和表面内存空间位于设备内存中, 并由纹理缓存进行缓存. 二级 (L2) 缓存会缓存 HBM (设备) 内存的读写, 并处理来自 GPU 内部各种子系统的内存请求. HBM 和 L2 内存空间可由所有 SM 和 GPU 上运行的所有应用访问.

<span id="section-4-3-1"></span>

#### 4.3.1 H100 HBM3 与 HBM2e DRAM 子系统

随着 HPC、AI 和数据分析数据集的规模持续扩大, 计算问题日益复杂, 更大的 GPU 内存容量和带宽已不可或缺. NVIDIA P100 是全球首个支持高带宽 HBM2 内存技术的 GPU 架构, NVIDIA V100 则提供速度更快、效率更高且容量更大的 HBM2 实现. NVIDIA A100 GPU 又进一步提高了 HBM2 的性能和容量.

H100 SXM5 GPU 对此作出大幅提升, 支持 80 GB (五个堆栈) 的高速 HBM3 内存, 内存带宽超过 3 TB/s, 实际达到两年前发布的 A100 内存带宽的 2 倍. PCIe H100 提供 80 GB 高速 HBM2e, 内存带宽超过 2 TB/s.

<span id="figure-21"></span>

![各代 GPU 的实际 HBM 带宽](./nvidia-h100-architecture/figure-21.png)

**图 21.** 全球首款 HBM3 GPU 内存架构, 实际带宽提高至 2 倍. 内存数据速率尚未最终确定, 最终产品可能会有变化.

<span id="section-4-3-2"></span>

#### 4.3.2 H100 L2 缓存

H100 的 L2 缓存为 50 MB, 是 A100 40 MB L2 缓存的 1.25 倍. 它可以缓存模型和数据集的更大部分, 供重复访问, 从而减少对 HBM3 或 HBM2e DRAM 的访问并提高性能. L2 缓存采用分区交叉开关结构, 将与相应分区直接相连的 GPC 内 SM 所访问的数据就地缓存. L2 缓存驻留控制可优化容量利用率, 让程序员能够选择性地管理哪些数据应留在缓存中, 哪些应被逐出.

HBM3 或 HBM2e DRAM 与 L2 缓存子系统都支持数据压缩和解压缩技术, 可优化内存与缓存的使用和性能.

<span id="section-4-3-3"></span>

#### 4.3.3 内存子系统 RAS 特性

H100 的 HBM3 和 HBM2e 内存子系统实现了以下两项主要 RAS (可靠性、可用性与可维护性) 特性.

**ECC 内存弹性.** H100 HBM3/2e 内存子系统支持单比特错误纠正、双比特错误检测 (SECDED) 错误纠正码 (ECC), 用于保护数据. ECC 为容易受数据损坏影响的计算应用提供更高可靠性. 在大规模集群计算环境中, GPU 需要处理极大数据集和/或长时间运行应用, 因此 ECC 尤其重要. H100 的 HBM3/2e 内存支持"旁带 ECC", 即使用一块与主 HBM 内存分开的较小内存区域存储 ECC 位 (与之相对的是"内联 ECC", 它会从主内存中划出一部分来存储 ECC 位). H100 中其他主要内存结构也受到 SECDED ECC 保护, 包括 L2 缓存, 以及所有 SM 内部的 L1 缓存和寄存器文件.

**内存行重映射.** H100 HBM3/HBM2e 子系统可以将含有产生 ECC 错误的存储单元的内存行置为无效, 并在启动时使用行重映射逻辑, 以预留且已知良好的内存行替换这些内存行. 每个 HBM3/HBM2e 内存 Bank 中都会留出一些内存行作为备用行, 必要时可以启用, 替换被判定为损坏的内存行.

<span id="table-03"></span>

![NVIDIA A100 与 H100 数据中心 GPU 对比](./nvidia-h100-architecture/table-03.png)

**表 3.** NVIDIA A100 与 H100¹ 数据中心 GPU 对比

> **注:** H100 和 A100 Tensor Core GPU 设计用于安装在高性能服务器和数据中心机架中, 为 AI 与 HPC 计算工作负载提供动力, 因此不包含显示接口、用于加速光线追踪的 NVIDIA RT Core 或 NVENC 编码器.

<span id="section-4-4"></span>

### 4.4 计算能力

H100 GPU 支持新的计算能力 9.0. [表 4](#table-04) 比较了不同 NVIDIA GPU 架构计算能力的参数.

<span id="table-04"></span>

![V100、A100 与 H100 的计算能力对比](./nvidia-h100-architecture/table-04.png)

**表 4.** 计算能力: V100 与 A100、H100 对比

<span id="section-4-5"></span>

### 4.5 第二代安全 MIG

NVIDIA 多实例 GPU (MIG) 技术随基于 NVIDIA Ampere 架构的 A100 Tensor Core GPU 一同推出. MIG 为共享同一 GPU 的多个用户提供独立、完全隔离且安全的 GPU 实例, 已成为云服务提供商 (CSP) 数据中心横向扩展的一项极其重要的功能.

<span id="section-4-5-1"></span>

#### 4.5.1 MIG 技术回顾

MIG 技术可将每个 A100 或 H100 GPU (包括 H100 SXM5 和 H100 PCIe 两种版本) 划分为最多 7 个 GPU 实例, 以优化 GPU 利用率, 并在不同客户端 (例如 VM、容器和进程) 之间提供明确的 QoS 和隔离. MIG 对具有多租户使用场景的云服务提供商尤其有益. 它除了提高安全性并允许向客户保证 GPU 利用率之外, 还能确保一个客户端不会影响其他客户端的工作或调度.

<span id="figure-22"></span>

![CSP 多实例 GPU 配置示例](./nvidia-h100-architecture/figure-22.png)

**图 22.** CSP MIG 配置示例. 此 CSP MIG 示意图展示了如何将来自同一组织或不同组织的多个独立用户, 分别分配到单个物理 GPU 内各自专用、受保护且隔离的 GPU 实例.

一项用于管理、调优、维护 vGPU (虚拟 GPU) 虚拟机 (VM) 配置并对其进行负载均衡的重要 MIG 功能, 是在同一 GPU 上的 GPU 实例之间迁移 vGPU, 更常见的则是在集群中的不同 GPU 之间进行迁移.

每个 GPU 实例在整个内存系统中都有独立且隔离的路径, 片上交叉开关端口、L2 缓存 Bank、内存控制器和 DRAM 地址总线都会单独分配给相应实例. 这样, 即使其他任务频繁置换自身缓存或占满 DRAM 接口, 单个用户的工作负载仍能以可预测的吞吐量和延迟运行, 并获得相同的 L2 缓存分配和 DRAM 带宽.

(有关 MIG 基础技术的更多详情, 请参阅 [NVIDIA A100 Tensor Core GPU 白皮书](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf).)

<span id="section-4-5-2"></span>

#### 4.5.2 H100 MIG 增强功能

H100 新一代 MIG 技术相比 A100, 每个 GPU 实例可提供约 3 倍的计算容量和近 2 倍的内存带宽. NVIDIA Hopper 架构通过完全安全、云原生的多租户、多用户 MIG 配置增强 MIG 技术. 借助硬件和 Hypervisor 层面的新机密计算功能, 最多 7 个 GPU 实例可彼此安全隔离 (有关机密计算的更多详情, 请参见下文 [第 5 节](#section-5)).

[图 23](#figure-23) 展示了一种示例系统配置, 其中 CPU 和 GPU 协同为共享单个 GPU 的多个用户提供多个可信执行环境 (TEE). CPU 一侧提供多个配备安全 NVIDIA 驱动程序的机密 VM. 此例中的 H100 GPU 被划分为 4 个安全 MIG 实例. CPU 与 GPU 之间的传输会进行加密. GPU 硬件虚拟化通过 PCIe SR-IOV 提供 (每个 MIG 实例有一个虚拟功能 (VF)). 多项基于硬件的安全功能提供机密性和数据完整性, 硬件防火墙则在各个 GPU 实例之间提供内存隔离.

<span id="figure-23"></span>

![单个 H100 GPU 上包含四个租户的安全 MIG 示例](./nvidia-h100-architecture/figure-23.png)

**图 23.** 多租户单 GPU 配置中的安全 MIG 示例

Hopper 架构现在还允许每个 GPU 实例使用专用图像和视频解码器, 在共享基础设施上提供安全、高吞吐量的智能视频分析 (IVA). 每个 MIG GPU 实例可以获得至少一个 NVDEC 和 NVJPG 单元.

此外, H100 MIG 实例现在包含各自的一组性能监视器, 可配合 NVIDIA 开发者工具使用. 借助 Hopper 的并发性能分析, 管理员可以监视大小适当的 GPU 加速实例, 并在用户之间无缝优化资源分配.

<span id="section-4-6"></span>

### 4.6 Transformer Engine

Transformer 模型是从 BERT 到 GPT-3 等当今广泛使用的语言模型的骨干, 需要庞大的计算资源. Transformer 最初为自然语言处理 (NLP) 而开发, 如今正越来越多地应用于计算机视觉、药物发现等众多领域. 它们的规模持续呈指数增长, 目前已达到数万亿个参数, 训练时间也延长到数月. 如此庞大的计算需求并不符合企业的实际需要. 例如, 训练 Megatron Turing NLG (MT-NLG) 需要 2048 个 NVIDIA A100 GPU 连续运行 8 周. 总体而言, 在过去 5 年中, Transformer 模型的增长速度远超大多数其他 AI 模型, 每两年扩大 275 倍 (参见 [图 24](#figure-24)).

<span id="figure-24"></span>

![不同使用场景中 Transformer 模型规模的增长](./nvidia-h100-architecture/figure-24.png)

**图 24.** 不同使用场景中的 Transformer 模型规模呈指数增长

H100 包含新的 Transformer Engine, 这是一项定制的 Hopper Tensor Core 技术, 可大幅加速 Transformer 的 AI 计算.

<span id="figure-25"></span>

![Transformer Engine 概念性工作过程](./nvidia-h100-architecture/figure-25.png)

**图 25.** Transformer Engine 概念性工作过程.

混合精度的目标是智能管理精度, 在维持准确度的同时, 获得更小、更快数值格式的性能. 在 Transformer 模型的每一层, Transformer Engine 都会分析 Tensor Core 生成的输出值统计信息. Transformer Engine 知道后续神经网络层的类型及其所需精度, 因而还会决定在将张量存入内存之前, 应把它转换为哪种目标格式. FP8 的范围比其他数值格式更有限. 为了充分利用可用范围, Transformer Engine 还会使用根据张量统计信息计算的缩放因子, 动态地将张量数据缩放到可表示范围内. 因此, 每一层都恰好在自身所需的范围内运行, 并以最佳方式获得加速.

<span id="section-4-7"></span>

### 4.7 第四代 NVLink 与 NVLink Network

百亿亿次级 HPC 和万亿参数 AI 模型是正在兴起的一类任务, 例如超越人类水平的对话式 AI, 即使在超级计算机上也需要数月才能完成训练. 要把几个月的漫长训练时间压缩到几天, 以便企业实际使用, 服务器集群中的每个 GPU 之间都需要高速、无缝的通信. PCIe 的有限带宽会形成瓶颈. 要构建性能最强的端到端计算平台, 需要速度更快、扩展性更强的 NVLink 互连.

NVLink 是 NVIDIA 的高带宽、高能效、低延迟、无损 GPU 间互连. 它包含链路级错误检测和数据包重放机制等弹性功能, 可保证数据传输成功. H100 GPU 实现了新的第四代 NVLink, 与 NVIDIA A100 Tensor Core GPU 所用的上一代第三代 NVLink 相比, 通信带宽达到 1.5 倍.

新的 NVLink 为多 GPU IO 和共享内存访问提供 900 GB/s 总带宽, 是 PCIe Gen 5 带宽的 7 倍. A100 GPU 中的第三代 NVLink 在每个方向使用 4 个差分对 (4 条通道) 组成一条链路, 每个方向提供 25 GB/s 有效带宽; 第四代 NVLink 在每个方向只用 2 个高速差分对组成一条链路, 同样在每个方向提供 25 GB/s 有效带宽. H100 包含 18 条第四代 NVLink 链路, 可提供 900 GB/s 总带宽; A100 则包含 12 条第三代 NVLink 链路, 提供 600 GB/s 总带宽.

在第四代 NVLink 之上, H100 还引入了新的 NVLink Network 互连. 它是可扩展版本的 NVLink, 支持最多 256 个 GPU 跨多个计算节点进行 GPU 间通信.

常规 NVLink 中的所有 GPU 共享一个地址空间, 请求会使用 GPU 物理地址直接路由; NVLink Network 则引入新的网络地址空间, 由 H100 中的新型地址转换硬件提供支持, 将所有 GPU 的地址空间彼此隔离, 并与网络地址空间隔离. 这使 NVLink Network 能够安全地扩展到更多 GPU.

由于 NVLink Network 端点不共享同一个内存地址空间, NVLink Network 连接不会在整个系统中自动建立. 相反, 与 InfiniBand 等其他网络接口类似, 用户软件应按需在端点之间显式建立连接.

<span id="section-4-7-1"></span>

#### 4.7.1 第三代 NVSwitch

新的第三代 NVSwitch 技术包括位于节点内部和外部的交换机, 用于连接服务器、集群和数据中心环境中的多个 GPU. 节点内每个新的第三代 NVSwitch 提供 64 个第四代 NVLink 端口, 可加速多 GPU 连接. 交换机总吞吐量从上一代的 7.2 Tbit/s 增至 13.6 Tbit/s.

新的第三代 NVSwitch 还通过组播和 [NVIDIA SHARP](https://docs.nvidia.com/networking/display/SHARPv200) 网络内归约, 为集合操作提供硬件加速. 加速的集合操作包括写广播 (all_gather)、reduce_scatter 和广播原子操作. 相比在 A100 上使用 [NCCL (NVIDIA 集合通信库)](https://developer.nvidia.com/nccl), 结构内组播与归约可以在小块集合操作上提供高达 2 倍的吞吐量, 同时显著降低延迟. NVSwitch 对集合操作的加速大幅减轻了集合通信给 SM 带来的负载.

<span id="section-4-7-2"></span>

#### 4.7.2 新款 NVLink Switch System

新的 NVLINK Network 技术与第三代 NVSwitch 相结合, 让 NVIDIA 可以构建大规模纵向扩展 NVLink Switch System 网络, 提供前所未有的通信带宽. 每个 GPU 节点以 2:1 渐缩层级开放节点中所有 GPU 的 NVLink 带宽. 各节点通过第二级 NVSwitch 相连. 这些交换机装在计算节点之外的 NVLink Switch 模块中, 将多个节点连接起来.

NVLink Switch System 最多支持 256 个 GPU. 相连节点可以提供 57.6 TB/s 的全互连带宽, 并能提供 1 exaFLOP 的 FP8 稀疏 AI 计算性能. [图 26](#figure-26) 比较了分别基于 A100 和 H100 的 32 节点、256 GPU DGX SuperPOD. 请注意, [图 26](#figure-26) 中的性能数据仅供参考. H100 系统目前尚不提供 NVLink Switch System 技术, 相关系统和可用时间将在未来公布.

<span id="figure-26"></span>

![DGX A100 与 DGX H100 256 GPU SuperPOD 对比](./nvidia-h100-architecture/figure-26.png)

**图 26.** DGX A100 与 DGX H100 32 节点、256 GPU NVIDIA SuperPOD 对比. DGX H100 SuperPOD 最多可跨 256 个 GPU, 并通过基于第三代 NVSwitch 技术的新款 NVLink Switch, 以 NVLink Switch System 实现完全互连. 2:1 渐缩胖树拓扑中的 NVLink Network 互连, 可将二分带宽大幅提升至 9 倍, 例如用于全互连交换时; 全归约吞吐量则达到上一代 InfiniBand 系统的 4.5 倍. 请注意, 图中的性能数据仅供参考. H100 系统目前尚不提供 NVLink Switch System 技术, 相关系统和可用时间将在未来公布.

交换机间的最大线缆长度从 5 m 增至 20 m. 现在支持 NVIDIA 制造的 OSFP (八通道小型可插拔) LinkX 线缆. 每个 OSFP 配备四端口光收发器, 以及 8 个 100G PAM4 信号通道. 这些四端口 OSFP 收发器创新使单台 1 RU、32 笼式 NVLink Switch 可以总共提供 128 个 NVLink 端口, 每个端口均以 25 GB/s 传输数据.

<span id="section-4-7-3"></span>

#### 4.7.3 PCIe Gen 5

H100 集成 PCI Express Gen 5 x16 通道接口, 可提供 128 GB/s 总带宽 (每个方向 64 GB/s); A100 配备的 Gen 4 PCIe 总带宽为 64 GB/s (每个方向 32 GB/s).

H100 可通过 PCIe Gen 5 接口连接最高性能的 x86 CPU 与 [SmartNIC / DPU (数据处理单元)](https://www.nvidia.com/en-us/networking/products/data-processing-unit/). H100 针对 NVIDIA BlueField-3 DPU 的连接进行了优化, 可通过 400 Gb/s 以太网或 NDR (下一代数据速率) 400 Gb/s InfiniBand 网络加速安全的 HPC 和 AI 工作负载.

H100 增加对原生 PCIe 原子操作的支持, 例如用于 32 位和 64 位数据类型的原子 CAS、原子交换和原子取数相加, 可加速 CPU 与 GPU 之间的同步和原子操作. H100 还支持单根输入/输出虚拟化 (SR-IOV), 允许多个进程或虚拟机 (VM) 共享并虚拟化单个 PCIe 连接的 GPU. H100 还允许单个通过 SR-IOV 连接 PCIe 的 GPU 上的虚拟功能 (VF) 或物理功能 (PF), 通过 NVLink 访问对等 GPU.

<span id="section-5"></span>

## 5 安全增强功能与机密计算

NVIDIA 向安全敏感型市场销售的 GPU 越来越多. 云服务提供商 (CSP)、汽车制造商、国家实验室、医疗卫生、金融等众多行业和组织都要求很高的安全水平. 每一代 NVIDIA 新 GPU 都在持续改进安全功能.

每天都有海量敏感数据生成、存储和处理, 面临越来越高的监管风险与网络攻击业务风险. 先进的加密技术可以保护存储中的静态数据, 以及通过网络传输的数据, 但对于处理或使用中的数据, 当今的保护仍存在很大空白. 新的机密计算技术通过保护使用中的数据和应用来填补这项空白, 为管理敏感数据和受监管数据的组织提高安全性.

NVIDIA H100 包含多项安全功能, 可限制对 GPU 内容的访问, 确保只有授权实体有权访问, 提供安全启动和证明功能, 并在系统运行期间主动监视攻击. 此外还配有专用片上安全处理器, 支持多种类型和级别的加密、硬件保护内存区域、特权访问控制寄存器、裸片传感器等许多功能, 为我们的客户及其数据提供安全的 GPU 处理能力.

H100 是世界上首款具备机密计算能力的 GPU. 用户在使用 H100 GPU 前所未有的加速能力时, 可以保护"使用中"数据和应用的机密性与完整性. H100 还提供多种其他安全功能, 用于保护用户数据、防御硬件和软件攻击, 并在虚拟化与 MIG 环境中更好地隔离和保护各个 VM.

NVIDIA H100 GPU 综合安全功能的主要目标包括:

- **数据保护与隔离:** 防止未经授权的实体访问其他用户的数据. 实体可以是用户、操作系统、Hypervisor 或 GPU 固件.
- **内容保护:** 防止未经授权的实体访问 GPU 上存储或处理的受保护内容.
- **物理损坏防护:** 防止恶意行为或意外造成 GPU 物理损坏.

<span id="section-5-1"></span>

### 5.1 NVIDIA 机密计算

NVIDIA 是 [机密计算联盟 (C3)](https://confidentialcomputing.io/) 成员. C3 由来自世界各地的供应商、学术机构、开源项目和软件开发者组成, 共同开发相关举措和技术, 以减少安全威胁, 保护公有云服务、本地数据中心、边缘系统与设备中正在使用的敏感数据及应用.

机密计算的正式定义是"在基于硬件的可信执行环境 (TEE) 中执行计算, 以保护使用中的数据". 这一定义与数据在哪里使用无关, 无论是在云端、最终用户设备上, 还是两者之间的某处. 它也与保护数据的处理器或所用保护技术无关. C3 将 TEE 定义为"为数据机密性、数据完整性和代码完整性这三项主要属性提供一定保证的环境".

如今, 静态存储的数据和通过网络传输的数据通常会受到保护, 但使用时并不会与操作系统 / Hypervisor 隔离. 必须信任操作系统 / Hypervisor 的要求, 给用户的数据和代码保护留下很大空白. 此外, 传统计算基础设施对使用中数据和代码的保护能力有限. 处理个人身份信息 (PII)、金融和健康数据等敏感数据, 或必须遵守数据本地化法规的组织, 需要缓解在所有阶段威胁其应用、模型和数据机密性及完整性的风险.

<span id="figure-27"></span>

![跨云端、本地与边缘部署的机密计算](./nvidia-h100-architecture/figure-27.png)

**图 27.** 机密计算保护多种 ISV 场景. 机密计算可保护云端、本地和边缘的 ISV 客户数据及已训练 AI 模型的机密性.

现有机密计算解决方案基于 CPU, 对 AI 和 HPC 等计算密集型工作负载而言速度过慢. 基于 CPU 的机密计算通常会降低系统性能, 这可能影响生产效率, 在延迟敏感的数据处理工作负载中甚至无法使用.

NVIDIA 机密计算是 NVIDIA Hopper 架构引入的一项新安全功能. H100 因此成为世界上首款能够保护使用中数据与代码机密性和完整性的 GPU. H100 将加速计算带入机密计算领域, 并把 CPU 的可信执行环境扩展到 GPU. 在过去, 由于必须保护使用中的数据和代码, 加上此前的机密计算解决方案在许多工作负载上性能不足或不够灵活, 共享基础设施 (云端、托管、边缘) 无法用于许多使用场景. H100 为这些场景打开了大门.

NVIDIA 机密计算创建基于硬件的可信执行环境 (TEE), 为单个 H100 GPU、一个节点内的多个 H100 GPU, 或各个安全多实例 GPU (MIG) 实例上运行的完整工作负载提供保护与隔离. 可信执行环境 (TEE) 会在 GPU 上的机密 VM 与 CPU 上的对应实例之间建立安全通道. TEE 提供两种运行模式.

1. 将整个 GPU 独占分配给单个 VM (也可同时为单个 VM 分配多个 GPU).
2. 对 NVIDIA H100 GPU 进行分区, 并使用 MIG 技术支持多个 VM, 以实现多租户机密计算. GPU 加速应用可以在 TEE 中不作修改地运行, 无需手动分区.

用户可以将丰富、强大的 NVIDIA AI 与 HPC 软件组合, 与 NVIDIA 机密计算提供的硬件信任根安全性相结合, 从 GPU 架构的最低层面实现安全和数据保护. 用户可以在共享或远程基础设施上运行并证明应用, 确保任何未经授权的实体均无法查看或修改应用在 TEE 中使用时的代码和数据, 包括 Hypervisor、主机操作系统、系统管理员、基础设施所有者或任何能够物理接触设备的人.

<span id="figure-28"></span>

![适用于不同使用场景的机密计算](./nvidia-h100-architecture/figure-28.png)

**图 28.** 适用于不同使用场景的机密计算

Hopper 架构的机密计算能力进一步增强并加速联邦学习等多方协作计算场景的安全性. 联邦学习让多个组织可以共同训练或评估 AI 模型, 无需共享各自的专有数据集. 采用 H100 的机密联邦学习可确保每个参与地点的数据和 AI 模型都免受外部或内部威胁的未授权访问, 各个地点还可以了解并证明对等方运行的软件. 这提高了安全协作的可信度, 可推动医学研究、加快药物开发、减少保险和金融欺诈, 并服务许多其他应用, 同时维持安全性、隐私和监管合规性.

<span id="figure-29"></span>

![跨三个地点的机密联邦学习](./nvidia-h100-architecture/figure-29.png)

**图 29.** 机密联邦学习

尽管 GPU 的机密计算能力涉及许多组件, 安全且可度量的启动仍是其中较重要的功能之一, 下文将对此作出说明.

<span id="section-5-2"></span>

### 5.2 成功度量

NVIDIA Ampere GPU 架构包含安全启动技术, 但并不支持满足机密计算合规要求所需的度量启动. 我们将简要讨论 H100 所实现安全启动和度量启动的概念与组件.

安全启动是一组硬件和软件系统, 可确保 GPU 从已知的安全状态启动, 并且启动 GPU 时只允许运行由 NVIDIA 编写和审查且通过身份验证的固件与微代码. 度量启动是收集、安全存储和报告启动过程特征的流程, 这些特征决定了 GPU 的安全状态. 证明与验证是将度量值和参考值进行比较的方法, 用来确保设备处于预期的安全状态. NVIDIA 提供证明方、参考值和背书签名.

部署工作流利用度量启动提供的度量值, 将其与 NVIDIA 或服务提供商给出的参考值比较, 以确定系统是否已准备就绪且处于安全状态, 可以开始处理客户数据. 系统通过验证后, 客户便可以像在非机密计算环境中运行相同应用一样启动应用.

<span id="section-5-3"></span>

### 5.3 NVIDIA 机密计算实现概述

如 [图 30](#figure-30) 所示, 左侧的 NVIDIA CC 关闭状态展示了传统 PC 架构, 主机操作系统和 Hypervisor 可完全访问 GPU 等设备. 右侧的 NVIDIA CC 开启状态展示了 VM 与其他要素完全隔离的情形.

<span id="figure-30"></span>

![NVIDIA CC 关闭与开启状态下的 VM 隔离](./nvidia-h100-architecture/figure-30.png)

**图 30.** NVIDIA CC 关闭与开启状态下的 VM 隔离对比

完整 VM TEE 和 GPU TEE 隔离由强大的硬件安全功能提供, 以形成机密计算环境. 其中包括前文已作部分说明的三个主要要素:

- **裸片信任根 (RoT)** - 在操作系统能够与 GPU 通信前, GPU 使用 RoT 确保设备上运行的固件真实可信, 且未被设备所有者 (CSP 等) 篡改.
- **设备证明** - 让用户可以确保自己正与启用机密计算的真实 NVIDIA GPU 通信, 且 GPU 的安全状态与包含固件和硬件配置在内的已知可信安全状态相符.
- **AES-GCM 256** - CPU 与 H100 GPU 之间的数据传输使用 AES256-GCM 的硬件实现, 以 PCIe 线路速率进行加密/解密. 这为总线上传输的数据同时提供机密性和完整性, 相应密钥只有 CPU 与 GPU TEE 可以使用. 此加密实现将通过 FIPS 140-3 2 级认证.

请注意, 使用 NVIDIA 机密计算技术不需要修改 CUDA 应用代码.

<span id="section-6"></span>

## 6 H100 视频 / IO 特性

<span id="section-6-1"></span>

### 6.1 用于 DL 的 NVDEC

与 A100 相比, H100 大幅改进了视频解码能力. 在 DL 平台中, 输入视频采用 H264 / HEVC / VP9 等任一行业标准进行压缩. DL 平台要实现较高的端到端吞吐量, 一项主要难题是让视频解码性能与训练及推理性能保持平衡. 否则, GPU 的完整 DL 性能就无法得到利用. H100 支持 7 个 NVDEC (NVidia DECode) 单元, A100 则有 5 个, 因此大幅提高了解码吞吐量. 这还可以确保在 MIG 模式下, 每个 MIG 分区至少能获得一个 NVDEC 单元.

<span id="table-05"></span>

![A100 与 H100 视频解码流数量对比](./nvidia-h100-architecture/table-05.png)

**表 5.** A100 与 H100 视频解码对比 (流数量):

<span id="table-06"></span>

![H100 硬件解码支持](./nvidia-h100-architecture/table-06.png)

**表 6.** H100 硬件解码支持

<span id="section-6-2"></span>

### 6.2 NVJPG (JPEG) 解码

在图像 DL 训练和推理中实现高吞吐量的一项根本瓶颈, 是图像的 JPEG 解码过程 (压缩格式 -> 原始格式). 用于处理图像位的串行操作, 使 CPU 和 GPU 执行 JPEG 解码时效率不高. 此外, 如果 JPEG 解码在 CPU 中完成, PCIe 还会成为另一个瓶颈.

H100 包含 7 个单核 NVJPG 硬件引擎, 用于加速 JPEG 解码; A100 则包含一个 5 核引擎.

H100 NVJPG 引擎的主要特性:

- NVJPG 支持 YUV420、YUV422、YUV444、YUV400 和 RGBA 格式
- 相比 A100 改进的 JPEG 架构: H100 不再采用 A100 的 5 核引擎, 而是增加 7 个单核引擎. 这大幅简化了软件使用模型, 因为 JPEG 图像可以独立分配给各个引擎, 无需收集成包含 5 张图像的批次. 如果同一批次中的图像分辨率不同, 吞吐量也会得到提高.
- 在 MIG 模式下, 每个 MIG 分区至少能获得一个 NVJPG 引擎
- JPEG 吞吐量较 A100 大幅提高

<span id="table-07"></span>

![H100 与 A100 NVJPG 解码性能](./nvidia-h100-architecture/table-07.png)

**表 7.** NVJPG 解码性能

NVIDIA 提供数据加载库 (DALI), 它通过自动调用 NVDEC / NVJPG, 管理视频 / 图像流水线的硬件加速. AI 开发者可以方便地通过它, 在 DL 工作负载中使用视频 / 图像硬件引擎. 它还支持灵活的计算图, 可用于创建自定义视频 / 图像流水线. DALI 的详细说明和用户指南见 [https://docs.nvidia.com/deeplearning/dali/user-guide/docs/](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/). DALI 库可从 [https://github.com/NVIDIA/DALI](https://github.com/NVIDIA/DALI) 下载.

<span id="section-7"></span>

## 7 附录 A - NVIDIA DGX - 数据中心 AI 的基础构建模块

人工智能 (AI) 如今已成为解决棘手业务难题的首选方法. 无论是改善客户服务、优化供应链、提取商业智能, 还是在几乎每个行业设计前沿产品与服务, AI 都为组织提供了实现创新的机制. 作为 AI 基础设施的先驱, NVIDIA DGX 系统提供性能最强、最完整的 AI 平台, 将这些重要构想变为现实.

<span id="section-7-1"></span>

### 7.1 NVIDIA DGX H100 - 世界上最完整的 AI 平台

NVIDIA DGX H100 推动业务创新与优化. DGX H100 是 NVIDIA 久负盛名的 DGX 系统的最新版本, 也是 NVIDIA DGX SuperPOD 的基础; 它是一套性能强大的 AI 系统, 采用突破性的 NVIDIA H100 Tensor Core GPU. 该系统只有一个设计目标: 尽可能提高 AI 吞吐量, 为企业提供高度完善、系统化且可扩展的平台, 帮助企业在自然语言处理、推荐系统、数据分析等众多领域实现突破. DGX H100 可部署在本地, 也可通过多种访问和部署选项使用, 能够提供企业用 AI 解决最大难题所需的性能.

<span id="section-7-2"></span>

### 7.2 DGX H100 概述

NVIDIA DGX H100 是一套适用于训练、推理和分析的通用高性能 AI 系统. DGX H100 支持云原生环境, 配备 Bluefield-3、NDR InfiniBand 和第二代 MIG 技术. 单台 DGX H100 系统可提供无可比拟的 32 petaFLOPS 性能. 将多台 DGX H100 系统连接成称为 DGX POD 乃至 DGX SuperPOD 的集群, 即可轻松扩展这项性能.

每台 DGX H100 系统包含:

- 8 个 H100 Tensor Core GPU
- 第四代 Tensor Core
- 第四代 NVLink
- 第三代 NVSwitch (4 个)
- 8 个 ConnectX-7 (400 Gb/s InfiniBand / 以太网)
- 2 个 Bluefield-3 DPU
- 支持 PCIe Gen5

<span id="section-7-3"></span>

### 7.3 无与伦比的数据中心扩展能力

NVIDIA DGX H100 是 [NVIDIA DGX SuperPOD](https://www.nvidia.com/en-us/data-center/resources/nvidia-dgx-superpod-reference-architecture/) 等大型 AI 集群的基础构建模块, 后者是可扩展 AI 基础设施的企业蓝图. DGX H100 中的 8 个 NVIDIA H100 GPU 使用新的高性能第四代 NVLink 技术, 通过 4 个第三代 NVSwitch 实现互连. 第四代 NVLink 技术提供上一代 1.5 倍的通信带宽, 最高可比 PCIe Gen5 快 7 倍. 它提供最高 7.2 TB/s 的 GPU 间总吞吐量, 相比上一代 DGX A100 提高近 1.5 倍. DGX H100 系统还包含 8 个 NVIDIA ConnectX-7 InfiniBand / 以太网适配器, 每个均以 400 Gb/s 运行, 从而为大规模 AI 工作负载提供性能强大的高速网络结构.

每台 DGX H100 还包含两个 NVIDIA BlueField-3 DPU (数据处理单元), 用于以智能化硬件加速方式完成存储、安全和网络管理功能. BlueField-3 DPU 将传统计算环境转变为安全且经过加速的虚拟私有云, 让组织可以在安全的多租户环境中运行应用工作负载. BlueField-3 将数据中心基础设施与业务应用解耦, 可以增强数据中心安全性、简化运维并降低总体拥有成本. BlueField-3 采用 NVIDIA 网络内计算技术, 可支持下一代超级计算平台, 提供最佳裸机性能和对多节点租户隔离的原生支持.

大规模 GPU 加速计算、先进网络硬件和软件优化相结合, 意味着 NVIDIA DGX H100 可扩展到数百或数千个节点, 应对下一代 AI 应用中的重大难题.

<span id="section-7-4"></span>

### 7.4 NVIDIA DGX H100 系统规格

<span id="table-08"></span>

![NVIDIA DGX H100 与 DGX A100 系统规格对比](./nvidia-h100-architecture/table-08.png)

**表 8.** NVIDIA DGX H100 系统规格

<span id="section-8"></span>

## 8 附录 B - NVIDIA CUDA 平台更新

[NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) 是一个功能全面、高效且性能优异的加速计算平台. 从系统软件到针对特定应用的库和框架 (参见 [图 31](#figure-31)), 它使用 GPU、CPU、DPU 和网络内计算, 加速各个层级的最终用户应用. 其成熟且易于使用的工具链、开发者工具和文档, 为加速异构应用提供出色的开发者体验.

<span id="section-8-1"></span>

### 8.1 高性能库与框架

CUDA 库可以充分发挥以下常见任务的性能: 数学运算 ([CUDA Math Library](https://developer.nvidia.com/cuda-math-library))、并行算法 ([CUB](https://github.com/NVIDIA/cub) 和 [Thrust](https://developer.nvidia.com/thrust))、线性代数 ([cuBLAS](https://developer.nvidia.com/cublas))、稠密和稀疏线性求解器 ([cuSOLVER](https://developer.nvidia.com/cusolver) 和 [cuSPARSE](https://developer.nvidia.com/cusparse))、FFT ([cuFFT](https://developer.nvidia.com/cufft))、随机数生成 ([cuRAND](https://developer.nvidia.com/curand))、张量操作 ([cuTENSOR](https://developer.nvidia.com/cutensor))、图像和信号处理 ([NPP](https://developer.nvidia.com/npp))、JPEG 解码 ([nvJPEG](https://developer.nvidia.com/nvjpeg)) 和 GPU 管理 ([NVML](https://developer.nvidia.com/nvidia-management-library-nvml)). [cuNumeric](https://developer.nvidia.com/cunumeric) 通过 Legate 与 Legion 运行时, 将 NumPy 程序透明加速并分布到任意规模的机器, 无需修改任何代码. [libcu++](https://nvidia.github.io/libcudacxx/) 提供异构同步和数据移动原语, 用于实现高并发、异构且符合 ISO 标准的 C++ 应用.

此外, CUDA 平台通信库支持基于标准的可扩展系统编程. [HPC-X](https://developer.nvidia.com/networking/hpc-x) 是支持 CUDA 的 MPI 库, 支持使用 GPUDirect 通过 RDMA 直接收发 GPU 缓冲区. [NVIDIA 集合通信库 (NCCL)](https://developer.nvidia.com/nccl) 实现了高度优化的多节点集合通信原语. [NVSHMEM](https://developer.nvidia.com/nvshmem) 以 OpenSHMEM 为基础, 为主机线程和设备线程提供异构多节点通信原语. [cuFile](https://docs.nvidia.com/gpudirect-storage/api-reference-guide/index.html) 与 [MAGNUM IO](https://developer.nvidia.com/magnum-io) 让异构应用可以通过 [GPUDirect Storage](https://developer.nvidia.com/gpudirect-storage) 进行高性能文件 I/O.

一整套面向特定领域的库和框架进一步加速了众多应用领域中的主要算法, 例如深度神经网络 ([cuDNN](https://developer.nvidia.com/cudnn))、用于仿真和隐式非结构化方法的线性求解器 ([AmgX](https://developer.nvidia.com/amgx))、量子计算 ([cuQuantum](https://developer.nvidia.com/cuquantum-sdk))、数据科学与机器学习 ([RAPIDS](https://rapids.ai/))、用于机器学习的数据加载和预处理 ([DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/)), 以及实时 3D 仿真与设计协作 ([Omniverse](https://developer.nvidia.com/nvidia-omniverse-platform)) 等. 超过 150 个 [软件开发工具包](https://developer.nvidia.com/) 利用这些库, 帮助开发者在大量应用领域中高效工作, 其中包括高性能计算 ([NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk))、AI、[机器学习](https://developer.nvidia.com/machine-learning)、[深度学习](https://developer.nvidia.com/deep-learning) 和数据科学、基因组学 ([NVIDIA CLARA](https://developer.nvidia.com/clara))、智慧城市 ([NVIDIA Metropolis](https://developer.nvidia.com/metropolis))、自动驾驶 ([NVIDIA Drive SDK](https://developer.nvidia.com/drive))、电信 ([NVIDIA Aerial SDK](https://developer.nvidia.com/aerial-sdk))、机器人 ([NVIDIA Isaac SDK](https://developer.nvidia.com/isaac-sdk))、网络安全 ([NVIDIA Morpheus SDK](https://developer.nvidia.com/morpheus-cybersecurity))、[计算机视觉](https://developer.nvidia.com/computer-vision) 等.

<span id="figure-31"></span>

![NVIDIA CUDA 平台及其生态系统](./nvidia-h100-architecture/figure-31.png)

**图 31.** NVIDIA CUDA 平台及其生态系统

<span id="section-8-2"></span>

### 8.2 系统软件

NVIDIA CUDA 平台还提供灵活的系统软件组件, 帮助用户高效部署、管理和优化大型异构系统. 这些组件包括设备驱动程序 (CUDA 驱动程序)、设备管理软件 (NVML、NVIDIA-smi、DCGM 和 Unified Fabric Manager)、用于异构网络和文件 I/O 的 GPUDirect, 以及容器感知作业调度系统和操作系统 (DGX OS).

<span id="section-8-3"></span>

### 8.3 文档与培训

庞大的 CUDA 软件生态配有完善的文档, 涵盖我们的编程模型 (例如 C++ 并行算法)、库 (例如 libcu++)、框架 (例如 RAPIDS AI) 和 SDK (例如 HPC SDK).

NVIDIA 深度学习学院 (DLI) 提供自主学习和现场培训, 例如在 Supercomputing 大会与国际超级计算大会等会议上开设课程, 帮助个人提升 AI、加速计算、加速数据科学、图形与仿真等领域的知识. DLI 还在研究机构和 HPC 中心培训合格教育工作者并将其认证为 DLI 大使, 让他们可以根据自身需要教授和调整 DLI 内容.

除官方文档之外, NVIDIA 还与不同社区和 HPC 站点合作, 提供 GPU Hackathon 和 Bootcamp 计划. 该计划让领域科学家与研究软件工程师 (RSE) 团队同来自 NVIDIA 及 HPC 社区的 GPU 导师结对, 传授有效使用现代异构计算系统所需的软件开发、并行计算和优化技能. NVIDIA 每年举办 GPU 技术大会 (GTC), 重点向开发者介绍最新 NVIDIA 平台和技术. 演讲内容涵盖 NVIDIA 编程模型、硬件细节, 以及加速计算在众多领域中的应用. 所有演讲都会录制, 并可在 GTC 点播平台观看.

<span id="section-8-4"></span>

### 8.4 语言与编译器

CUDA 平台提供统一而灵活的编译器栈, 通过 NVIDIA NVVM IR 和 NVIDIA libNVVM 生成经过高度优化的设备二进制文件. NVVM IR 是一种以 LLVM 7 为基础的编译器中间表示 (IR), 为生成 GPU 计算内核提供前端编译器目标. libNVVM 是将 NVVM IR 编译并优化为 PTX 的库, PTX 是 NVIDIA GPU 的虚拟 ISA. 所有 NVIDIA Compute 编译器都使用 libNVVM, 将 NVIDIA GPU 作为目标 ([图 32](#figure-32)); 它让用户和框架可以将自己选择的编程语言引入 CUDA 平台, 并获得与 CUDA C++ 本身相同的代码生成质量和优化水平.

<span id="figure-32"></span>

![使用 libNVVM 的高级语言前端](./nvidia-h100-architecture/figure-32.png)

**图 32.** 高级语言前端. 前端使用 libNVVM 将 NVVM IR 程序编译为 PTX, 并在 GPU 上运行.

PTX 是 NVIDIA GPU 的虚拟 ISA, 同时也是一种公开 ISA. 第三方生成器可将其作为目标, 在我们的目标架构上高效运行. PTX 还具有向前兼容的优点, 并可离线或在运行时汇编.

在许多应用中, 待生成的 GPU 计算内核取决于程序输入. 这些应用可以生成 NVVM IR, 但 NVIDIA 运行时编译器允许它们改为生成开发者熟悉的 CUDA C++, 从而显著提高应用及其用户的生产效率. NVRTC 在运行时使用 libNVVM 将 CUDA C++ 编译为 PTX, 也可以使用嵌入式 PTX 汇编器将其编译为原生 GPU 二进制代码. 因此, Python 程序等应用可以针对用户输入的程序动态生成内核, C++ 程序等应用则可以根据程序输入在运行时特化计算内核.

[NVIDIA HPC SDK](https://developer.nvidia.com/hpc-sdk) 是一套面向异构系统的工具链. NVCC 是 CUDA C++ 编译器, 提供分离式编译模型, 将 GPU 编译与 GCC 等外部主机编译器配合使用 ([图 33](#figure-33) 左侧). NVIDIA HPC 编译器 NVC、NVC++ 和 NVFortran 提供统一的异构编译模型 ([图 33](#figure-33) 右侧).

<span id="figure-33"></span>

![NVCC 分离式编译与 NVC++ 统一编译模型](./nvidia-h100-architecture/figure-33.png)

**图 33.** NVCC 分离式编译模型与 NVC++ 统一编译模型

统一编译器只会解析和优化程序一次, 然后再针对不同目标拆分编译过程. 这种模型支持某些 nvcc 无法提供的功能. 例如, 使用 nvcc 时, CUDA C++ 设备代码需要添加 `__device__` 注解 ([图 34](#figure-34) 左侧). NVC++ 编译器不需要这些注解 ([图 34](#figure-34) 右侧); 如果程序使用来自某个特定目标的函数, 且其定义可达, 编译器就会尝试编译它.

<span id="figure-34"></span>

![NVCC 设备注解与 NVC++ 推断执行空间](./nvidia-h100-architecture/figure-34.png)

**图 34.** 统一工具链支持执行空间推断

统一编译简化了开发, 让初学者更容易进行 GPU 编程, 也让经验丰富的开发者提高效率. 它还增加了主机与设备目标之间的代码复用, 简化 GPU 应用的加速过程.

<span id="section-9"></span>

## 9 附录 C - 使用 DPX 指令加速基因组学

与上一代 GPU 和 CPU 相比, NVIDIA H100 能以不同倍数加速许多种应用与算法. 本节重点介绍 H100 在基因组学领域带来的一项显著加速. 随着近几年传染病增多、全球流行病带来危险, 基因组和蛋白质分析对人类的重要性达到了前所未有的程度.

H100 引入新的 DPX 指令, 这是一组专用硬件指令, 用于加速动态规划算法, 例如 DNA 基因测序、蛋白质分类和折叠所用的 Smith-Waterman 算法. 与 NVIDIA Ampere A100 GPU 相比, H100 可以让 Smith-Waterman 加速至 7 倍, 大幅缩短疾病诊断、病毒变异研究和疫苗开发得出结果的时间. 下面简要介绍基因组学和基因测序.

基因组学领域正呈指数增长, 改变医疗卫生、农业和生命科学行业; 它也是我们对抗 SARS-CoV-2 和 COVID-19 最锐利的武器之一. 对人类基因组进行全基因组或选定部分的测序, 对了解其工作原理至关重要. 借此我们能够识别可能导致疾病、提供保护或成为治疗靶点的遗传变异. 随着组织利用基因组来了解疾病、发现药物和改善患者护理, 数据分析与管理正成为发掘基因组价值的主要工具.

自 2005 年下一代测序 (NGS) 问世以来, 该行业经历了数据爆炸, 也催生了围绕人类基因组建立的新产业, 从梳理家族史到临床护理皆包括在内. 基因组学可以受益于先进计算系统, 这些系统能够加速把原始仪器数据转化为生物学洞见所需的计算密集型步骤. 一个人的基因组原始数据约为 100 GB. 经过分析后, 总数据占用会增至 225 GB 以上; 分析过程会使用深度学习和自然语言处理等复杂算法和应用. 使用 GPU 加速数学模型, 显然有利于测序读段处理、变异识别等传统基因组分析; 它还有望彻底改变我们对特定基因组变异如何影响疾病与健康的理解.

NVIDIA Clara™ Parabricks® 是面向下一代测序数据的加速计算框架, 支持 DNA 和 RNA 应用的端到端数据分析工作流. Clara Parabricks 在一系列 NVIDIA GPU 平台上运行, 提供超过 50 款加速工具, 包括 GPU 加速的 Burrows-Wheeler Aligner (BWA-MEM)、Picard 和 Samtools, 以及一套用于注释、筛选和合并多种变异调用格式 (VCF) 的实用工具. 整个工作流中的加速工具结合起来, 能让结果在几分钟内生成, 而不再需要数小时或数天.

<span id="figure-35"></span>

![NVIDIA Clara Parabricks 加速框架](./nvidia-h100-architecture/figure-35.png)

**图 35.** NVIDIA CLARA Parabricks 加速框架

基因组是生物体内脱氧核糖核酸 (DNA) 的完整集合. DNA 是一种化合物, 包含生物体发育和调控其各种活动所需的遗传指令. DNA 分子由两条相互缠绕且成对的链组成. 每条链由 4 种称为核苷酸碱基的化学单元组成. 这些碱基是腺嘌呤 (A)、胸腺嘧啶 (T)、鸟嘌呤 (G) 和胞嘧啶 (C). 相对链上的碱基会特异性配对; A 始终与 T 配对, C 始终与 G 配对. 人类基因组包含约 30 亿个这样的碱基对, 位于我们所有细胞的细胞核内 23 对染色体中. 对基因组进行测序, 就是确定一段 DNA 中碱基对的确切顺序.

个人 DNA 的测序过程首先通过化学过程将 DNA 分成互补对, 把 DNA 链切成特定大小的片段 (长度可以是 100 至 2000 个碱基对), 再使用测序机器对这些小片段 (称为读段) 测序, 生成计算机可读的碱基对编码序列. 随后, 可以在参考基因组中搜索这些序列的位置, 将测序片段重新组装起来; 也可以采用 De Novo 方法, 在不依赖参考基因组序列的情况下寻找碱基的重叠模式, 以此组装测序片段.

从计算角度看, 问题可归结为在包含数十亿碱基对的参考基因组中搜索并匹配一组"读段", 或通过模式匹配算法从零开始组装基因组. 后一种算法需要比较数百万个读段来寻找重叠, 再按正确顺序将其对齐. 在此过程中, 算法可能需要插入、编辑或删除序列以解决不匹配, 还要指定可能遇到的各种不匹配所对应的成本. 因此, 模式匹配所用的计算硬件架构需要足够灵活, 既能满足这些要求, 也能支持蛋白质测序等其他基因组学问题所用的类似算法.

NVIDIA CLARA Parabricks 加速计算框架的 GPU 加速 BWA-MEM 模块, 使用 Smith-Waterman 算法进行 DNA 测序. 该算法首先比较两个碱基读段字符串, 生成一个评分矩阵, 然后回溯矩阵中的分数, 确定两个字符串之间的最佳匹配模式. [此处](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm) 对该算法在基因组测序中的用法作了很好的说明.

<span id="figure-36"></span>

![Smith-Waterman 评分矩阵与回溯](./nvidia-h100-architecture/figure-36.png)

**图 36.** 用于基因组测序的 Smith-Waterman 算法 [+1]

在上图中, 每次更新矩阵单元格都需要 5 项基本计算.

1. 匹配时, 将对角元素加上数值 x (本图中 x = 3)
2. 不匹配时, 将对角元素减去数值 x
3. 垂直元素不匹配时, 减去数值 y (本图中 y = 2)
4. 水平元素不匹配时, 减去数值 z (本图中 z = 2)
5. 求上述四项运算结果的最大值 (如果结果为负, 则将该单元格置零).

H100 的新 DPX 指令针对上述计算及其他类似算法进行了优化, 可以加速这些操作.

[+1]: 来源为 [https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm](https://en.wikipedia.org/wiki/Smith%E2%80%93Waterman_algorithm)

<span id="section-10"></span>

## 10 声明

我们相信, 本规格说明中提供的信息截至注明日期准确且可靠. 然而, NVIDIA Corporation ("NVIDIA") 不对这些信息的准确性或完整性作出任何明示或默示的陈述或保证. NVIDIA 不对这些信息的使用或其后果, 或因使用这些信息而可能造成的专利或其他第三方权利侵权承担任何责任. 本出版物取代并替换此前可能提供的本产品所有其他规格说明.

NVIDIA 保留随时更正、修改、增强、改进和以其他方式变更本规格说明, 和/或停止提供任何产品或服务而不另行通知的权利. 客户应在下单前取得最新的相关规格说明, 并验证其中信息是否最新且完整.

除非 NVIDIA 授权代表与客户签署的单独销售协议另有约定, NVIDIA 产品的销售受订单确认时所提供 NVIDIA 标准销售条款和条件的约束. 对于购买本规格说明所述 NVIDIA 产品一事, NVIDIA 在此明确反对适用任何客户通用条款和条件.

NVIDIA 产品的设计、授权或保证范围不包括在医疗、军事、航空器、航天或生命支持设备中使用, 也不包括用于 NVIDIA 产品发生故障或功能失常时可合理预期会造成人身伤害、死亡、财产损失或环境破坏的应用. NVIDIA 不对在这些设备或应用中包含和/或使用 NVIDIA 产品承担任何责任, 因此这种包含和/或使用的风险由客户自行承担.

NVIDIA 不陈述或保证基于这些规格的产品无须进一步测试或修改, 即适用于任何特定用途. NVIDIA 不一定会测试每款产品的所有参数. 客户独自负责确保产品适合并符合客户规划的应用, 并负责对应用进行必要测试, 以避免应用或产品出现故障. 客户产品设计中的薄弱环节可能会影响 NVIDIA 产品的质量和可靠性, 并可能造成超出本规格说明所含条件和/或要求的其他条件和/或要求. NVIDIA 不承担与下述原因可能引起或可归因于下述原因的任何故障、损害、成本或问题相关的责任: (i) 以任何违反本规格说明的方式使用 NVIDIA 产品, 或 (ii) 客户产品设计.

本规格说明不授予 NVIDIA 的任何专利权、版权或其他知识产权的明示或默示许可. NVIDIA 发布的第三方产品或服务相关信息, 不构成 NVIDIA 对这些产品或服务的使用许可, 也不构成相关保证或认可. 使用这些信息可能需要取得第三方专利或其他知识产权项下的第三方许可, 或取得 NVIDIA 专利或其他知识产权项下的 NVIDIA 许可. 只有在取得 NVIDIA 书面批准、不作修改地复制, 并随附所有相关条件、限制和声明的情况下, 才允许复制本规格说明中的信息.

所有 NVIDIA 设计规格、参考板卡、文件、图纸、诊断程序、列表及其他文档 (合称或分别称为"材料") 均按"原样"提供. NVIDIA 不对这些材料作出任何明示、默示、法定或其他保证, 并明确否认关于不侵权、适销性和特定用途适用性的所有默示保证. 无论客户因任何原因遭受何种损害, NVIDIA 对客户承担的本说明所述产品相关责任总额和累计责任, 均以 NVIDIA 针对该产品的销售条款和条件为限.

<span id="section-10-1"></span>

### 10.1 商标

NVIDIA、NVIDIA 徽标、NVIDIA CUDA、NVIDIA Omniverse、NVIDIA RTX、NVIDIA Tesla、NVIDIA Turing、NVIDIA Volta、NVIDIA Jetson AGX Xavier、NVIDIA DGX、NVIDIA HGX、NVIDIA EGA、NVIDIA CUDA-X、NVIDIA GPU Cloud、GeForce、Quadro、CUDA、GeForce RTX、NVIDIA NVLink、NVIDIA NVSwitch、NVIDIA DGX POD、NVIDIA DGX SuperPOD 和 NVIDIA TensorRT 是 NVIDIA Corporation 在美国及其他国家/地区的商标和/或注册商标. 其他公司和产品名称可能是与其关联的相应公司的商标.

版权所有 © 2023 NVIDIA Corporation. 保留所有权利.
