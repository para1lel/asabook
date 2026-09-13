---
title: 'NVIDIA Tesla P100 Pascal Architecture'
createTime: 2026/09/13 13:21:06
permalink: /papers/nvidia-pascal-architecture/
pageClass: paper-reading
---

> [NVIDIA Corporation](https://www.nvidia.com/en-us/about-nvidia/). *[NVIDIA Tesla P100: The Most Advanced Datacenter Accelerator Ever Built](https://www.nvidia.com/en-us/data-center/resources/pascal-architecture-whitepaper/)*, 搭载 Pascal GP100, 当时全球最快的 GPU, V1.2, 2017 年. 该白皮书最早随 Tesla P100 于 2016 年 4 月 5 日发布. <a href="/paper/nvidia-pascal-architecture.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源码; 准确措辞, 印刷版式和参考文献以发布的 PDF 为准.

<span id="section-1"></span>

## 1 引言

大约十年前, NVIDIA® 随 G80 GPU 和 NVIDIA® CUDA® 并行计算平台的推出, 率先使用 GPU 加速计算密集型工作负载. 如今, NVIDIA® Tesla® GPU 已为数千种高性能计算 (HPC) 应用提供加速, 涵盖计算流体力学, 医学研究, 机器视觉, 金融建模, 量子化学, 能源勘探等众多领域.

NVIDIA Tesla GPU 安装在许多全球顶级超级计算机中, 加快了科学发现, 并支持多个领域日益复杂的模拟. 数据中心使用 NVIDIA Tesla GPU 加速大量 HPC 和大数据应用, 同时支持前沿的人工智能 (AI) 与深度学习系统.

NVIDIA 新推出的 NVIDIA Tesla P100 加速器 (见[图 1](#figure-01)) 采用开创性的 NVIDIA® Pascal™ GP100 GPU, 将 GPU 计算推进到新的水平. 本文详细介绍 Tesla P100 加速器和 Pascal GP100 GPU 的架构.

本文还讨论 NVIDIA 功能强大的新型 DGX-1 服务器, 它使用 8 个 Tesla P100 加速器, 实际上是一台装在机箱里的 AI 超级计算机. DGX-1 专为推进 AI 研究的人员以及需要集成式深度学习系统的数据科学家打造.

<span id="figure-01"></span>

![搭载 Pascal GP100 GPU 的 NVIDIA Tesla P100 加速器](./nvidia-pascal-architecture/figure-01.png)

**图 1.** 搭载 Pascal GP100 GPU 的 NVIDIA Tesla P100

<span id="section-2"></span>

## 2 Tesla P100: 面向 GPU 计算的革新性性能与特性

Tesla P100 拥有 153 亿晶体管的 GPU, 新型高性能互连, 高效简化 GPU 编程的新技术和出色的能效. 这种互连可大幅加快 GPU 点对点及 GPU 到 CPU 的通信. 因此, Tesla P100 不只是当时性能最强的 GPU 加速器, 也是当时架构最复杂的 GPU 加速器.

Tesla P100 的主要特性包括:

- **极致性能**<br>
  为 HPC, 深度学习以及更多 GPU 计算领域提供动力
- **NVLink™**<br>
  NVIDIA 面向最高应用扩展能力推出的新型高速, 高带宽互连
- **HBM2**<br>
  高速, 大容量且极为高效的 CoWoS (Chip-on-Wafer-on-Substrate) 堆叠内存架构
- **统一内存, 计算抢占和新型 AI 算法**<br>
  显著改进的编程模型, 以及针对 Pascal 架构优化的先进 AI 软件;
- **16nm FinFET**<br>
  支持更多特性, 更高性能和更好的能效

<span id="figure-02"></span>

![Tesla P100 引入的五项技术](./nvidia-pascal-architecture/figure-02.png)

**图 2.** Tesla P100 的新技术

<span id="section-2-1"></span>

### 2.1 面向高性能计算与深度学习的极致性能

Tesla P100 为要求最严苛的计算应用提供出色性能, 具体为:

- 5.3 TFLOPS 双精度浮点 (FP64) 性能
- 10.6 TFLOPS 单精度 (FP32) 性能
- 21.2 TFLOPS 半精度 (FP16) 性能

<span id="figure-03"></span>

![Tesla P100 与历代 GPU 的计算性能对比](./nvidia-pascal-architecture/figure-03.png)

**图 3.** Tesla P100 的计算性能显著超过历代 GPU

NVIDIA GPU 多年来加速了高性能计算中的众多领域, 而最近, 深度学习成为 GPU 加速极受重视的领域. NVIDIA GPU 现已处在深度神经网络 (DNN) 和人工智能 (AI) 的前沿. 与 CPU 相比, 它们可将各种应用中的 DNN 加速 10-20 倍, 并把训练时间从数周缩短到数天. 在过去三年中, 基于 NVIDIA GPU 的计算平台已将深度学习网络训练速度提高了 50 倍. 在过去两年中, 与 NVIDIA 合作开展深度学习的公司数量增长近 35 倍, 超过 3,400 家.

Pascal 架构的新创新包括原生 16 位浮点 (FP) 精度, 使 GP100 能够大幅加快许多深度学习算法. 这些算法不要求很高的浮点精度, 却能从 FP16 提供的额外计算能力和 16 位数据类型更低的存储需求中获得很大收益.

<span id="section-2-2"></span>

### 2.2 NVLink: 面向多 GPU 和 GPU 到 CPU 连接的超大带宽

随着 GPU 加速计算日益普及, 从工作站, 服务器到超级计算机, 各个层级都在部署更多多 GPU 系统. 许多 4-GPU 和 8-GPU 系统配置已经用于解决规模更大, 更加复杂的问题. 多组多 GPU 系统通过 InfiniBand® 和 100 Gb 以太网互连, 组成规模和能力都大得多的系统. GPU 与 CPU 的比例也有所提高. 2012 年最快的超级计算机 Titan 位于 Oak Ridge National Labs, 每个 CPU 配置 1 个 GK110 GPU. 如今, 随着开发者在应用中不断发掘并利用可用的并行性, 每个 CPU 更常与 2 个或更多 GPU 配对. 这一趋势延续下去时, PCIe 带宽会成为多 GPU 系统层面更严重的瓶颈.

为解决这个问题, Tesla P100 配备 NVIDIA 新型高速接口 NVLink, GPU 到 GPU 的双向数据传输带宽最高可达 160 Gigabytes/second, 是 PCIe Gen 3 x16 的 5 倍. [图 4](#figure-04) 展示 NVLink 以混合立方网格拓扑连接 8 个 Tesla P100 加速器.

<span id="figure-04"></span>

![以混合立方网格连接的 8 个 Tesla P100 加速器](./nvidia-pascal-architecture/figure-04.png)

**图 4.** NVLink 以混合立方网格拓扑连接 8 个 Tesla P100 加速器

[图 5](#figure-05) 展示多种工作负载的性能, 说明服务器使用 NVLink 连接最多 8 个 GP100 GPU 时能够达到的性能扩展能力. (注意: 这些数字在预生产 P100 GPU 上测得.)

<span id="figure-05"></span>

![通过 NVLink 连接 1, 2, 4 和 8 个 P100 GPU 时的工作负载加速](./nvidia-pascal-architecture/figure-05.png)

**图 5.** 连接 8 个 P100 时获得最大的性能提升

<span id="section-2-3"></span>

### 2.3 HBM2 高速 GPU 内存架构

Tesla P100 是全球首个支持 HBM2 内存的 GPU 架构. HBM2 提供 Maxwell GM200 GPU 3 倍的内存带宽. 因此, P100 能以更高带宽处理大得多的工作集, 提高效率和计算吞吐量, 并减少从系统内存传输数据的频率.

HBM2 是堆叠内存, 与 GPU 位于同一物理封装中, 相比传统 GDDR5 大幅节省空间, 使我们能以前所未有的便利构建更高密度的 GPU 服务器.

<span id="figure-06"></span>

![Tesla P100 HBM2 与历代 GPU 的内存带宽对比](./nvidia-pascal-architecture/figure-06.png)

**图 6.** 采用 HBM2 的 Tesla P100 内存带宽显著超过历代 GPU

<span id="section-2-4"></span>

### 2.4 通过统一内存和计算抢占简化开发者编程

统一内存是 NVIDIA GPU 计算的一项重大进展, 也是 Pascal GP100 GPU 架构中一项重要的新硬件与软件特性. 它为 CPU 和 GPU 内存提供单一, 无缝的统一虚拟地址空间. 统一内存大幅简化 GPU 编程和应用向 GPU 的移植, 也降低了 GPU 计算的学习门槛. 程序员不再需要操心如何在两个不同的虚拟内存系统之间管理数据共享.

GP100 是首个支持硬件缺页的 NVIDIA GPU, 结合新的 49 位 (512 TB) 虚拟寻址后, 可以在 GPU 与 CPU 的完整虚拟地址空间之间透明迁移数据.

计算抢占是 GP100 新增的另一项重要软硬件特性, 可在指令级粒度抢占计算任务, 而不是像此前的 Maxwell 和 Kepler GPU 架构那样只能在线程块粒度抢占. 计算抢占可以防止长时间运行的应用独占系统 (使其他应用无法运行) 或超时. 程序员不再需要修改长时间运行的应用, 让它们与其他 GPU 应用和平共处. 借助 GP100 的计算抢占, 应用可以按处理大型数据集或等待各种条件所需的时长运行, 同时与其他任务共同调度. 例如, 交互式图形任务和交互式调试器都能与长时间运行的计算任务同时运行.

<span id="section-3"></span>

## 3 深入了解 GP100 GPU 硬件架构

GP100 旨在成为全球性能最高的并行计算处理器, 满足 Tesla P100 加速器平台所服务的 GPU 加速计算市场需求. 与此前的 Tesla 级 GPU 一样, GP100 由图形处理集群 (GPC), 纹理处理集群 (TPC), 流式多处理器 (SM) 和内存控制器阵列组成. 完整的 GP100 包含 6 个 GPC, 60 个 Pascal SM, 30 个 TPC (每个包含 2 个 SM), 以及 8 个 512 位内存控制器 (合计 4096 位).

GP100 中的每个 GPC 有 10 个 SM. 每个 SM 有 64 个 CUDA Core 和 4 个纹理单元. 60 个 SM 使 GP100 总计拥有 3840 个单精度 CUDA Core 和 240 个纹理单元. 每个内存控制器连接 512 KB L2 缓存, 每个 HBM2 DRAM 堆栈由一对内存控制器控制. 完整 GPU 总计包含 4096 KB L2 缓存.

[图 7](#figure-07) 展示带 60 个 SM 单元的完整 GP100 GPU (不同产品可以使用 GP100 的不同配置). Tesla P100 加速器使用 56 个 SM 单元.

<span id="figure-07"></span>

![带 60 个 SM 单元的完整 Pascal GP100 GPU 框图](./nvidia-pascal-architecture/figure-07.png)

**图 7.** 带 60 个 SM 单元的完整 Pascal GP100 GPU

<span id="section-3-1"></span>

### 3.1 出色的性能和能效

提高性能与改善能效是新 GPU 架构的两个主要目标. Maxwell 架构对 SM 做出多项修改, 相比 Kepler 提高了效率. Pascal 在此基础上继续改进, 使每瓦性能进一步超过 Maxwell. TSMC 的 16-nm FinFET 制程发挥了重要作用, 同时也通过多项 GPU 架构修改, 在保持高性能的同时进一步降低功耗.

<span id="table-01"></span>

![Tesla K40, Tesla M40 与 Tesla P100 规格](./nvidia-pascal-architecture/table-01.png)

**表 1.** Tesla P100 与前代 Tesla 产品对比

<span id="section-3-2"></span>

### 3.2 Pascal 流式多处理器

GP100 的第六代 SM 架构提高 CUDA Core 利用率和能效, 从而显著改善 GPU 整体性能, 并能使用比前代 GPU 更高的核心时钟频率.

GP100 的 SM 包含 64 个单精度 (FP32) CUDA Core. 相比之下, Maxwell 和 Kepler SM 分别有 128 个和 192 个 FP32 CUDA Core. GP100 SM 分为两个处理块, 每个处理块有 32 个单精度 CUDA Core, 1 个指令缓冲区, 1 个 warp 调度器和 2 个分派单元. GP100 SM 的 CUDA Core 总数只有 Maxwell SM 的一半, 但保持相同的寄存器文件大小, 并支持相近的 warp 与线程块占用率. GP100 SM 的寄存器数量与 Maxwell GM200 和 Kepler GK110 SM 相同, 但完整 GP100 GPU 拥有多得多的 SM, 因而总体寄存器数量也多得多. 这意味着整个 GPU 上的线程可以使用更多寄存器, 与前代 GPU 相比, GP100 可同时维持更多线程, warp 和线程块.

GP100 GPU 的 SM 数量增加, 也提高了整个 GPU 的共享内存总量, 有效地让共享内存总带宽增加一倍以上. GP100 中每个 SM 配有更高比例的共享内存, 寄存器和 warp, 使 SM 能更高效地执行代码. 指令调度器可从更多 warp 中选择, 可以发起更多加载, 每个线程也能获得更高的共享内存带宽.

[图 8](#figure-08) 展示由此得到的 GP100 SM 框图.

与 Kepler 相比, Pascal SM 的数据路径组织更简单, 管理 SM 内数据传输所需的芯片面积和功耗都更少. Pascal 还通过更好的调度和重叠的加载/存储指令提高浮点单元利用率. GP100 的新型 SM 调度器架构延续 Maxwell 调度器的改进并进一步提高智能程度, 在改善性能的同时降低功耗. 每个 warp 调度器 (每个处理块 1 个) 每个时钟周期可以分派 2 条 warp 指令.

GP100 的 FP32 CUDA Core 新增了一项能力, 可以处理 16 位与 32 位精度的指令和数据, 后文将详细说明. FP16 运算吞吐量最高可达 FP32 的两倍.

<span id="figure-08"></span>

![Pascal GP100 流式多处理器框图](./nvidia-pascal-architecture/figure-08.png)

**图 8.** Pascal GP100 SM 单元

<span id="section-3-3"></span>

### 3.3 为高性能双精度计算而设计

双精度算术是线性代数, 数值模拟和量子化学等许多 HPC 应用的核心. 因此, GP100 的一个主要设计目标是显著提高这些用例的实际性能.

GP100 的每个 SM 有 32 个双精度 (FP64) CUDA Core, 数量为 FP32 单精度 CUDA Core 的一半. 完整 GP100 GPU 有 1920 个 FP64 CUDA Core. 单精度 (SP) 单元与双精度 (DP) 单元的 2:1 比例更适合 GP100 的新数据路径配置, 使 GPU 能更高效地处理 DP 工作负载. 与此前的 GPU 架构一样, GP100 完全支持符合 IEEE 754-2008 的单精度和双精度算术, 包括融合乘加 (FMA) 运算和全速非正规值处理.

> **注意:** Kepler GK110 的 SP 单元与 DP 单元之比为 3:1.

<span id="section-3-4"></span>

### 3.4 FP16 算术支持加快深度学习

深度学习是发展最快的计算领域之一. 它是许多重要应用中的必要组成部分, 包括实时语言翻译, 高精度图像识别, 自动图像描述, 自动驾驶目标识别, 最优路径计算和避碰等. 深度学习分为两个步骤.

- 首先, 必须训练神经网络.
- 其次, 将网络部署到实际环境中运行推理计算, 使用此前的训练结果对未知输入进行分类, 识别和一般性处理.

与 CPU 相比, GPU 可以大幅加快深度学习训练和推理.

其他技术计算应用要求高精度浮点计算, 而深度神经网络架构由于训练时使用反向传播算法, 天然具有一定的容错能力. 实际上, dropout 等方法为了避免网络对训练数据集过拟合, 会让训练后的网络具备较好的泛化能力, 不会过分依赖任何给定单元计算结果的准确性 (或误差).

与更高精度的 FP32 或 FP64 相比, 存储 FP16 数据可以减少神经网络的内存用量, 因而能够训练和部署更大的网络. 使用 FP16 计算可将性能提升到 FP32 算术的两倍, FP16 数据传输所需的时间也少于 FP32 或 FP64 传输.

> **注意:** 在 GP100 中, 一条成对运算指令可以执行两个 FP16 运算.

GP100 的架构改进结合 FP16 数据类型支持, 与仅仅一年前所能达到的水平相比, 可显著缩短深度学习处理时间.

<span id="section-3-5"></span>

### 3.5 改进的原子操作

原子内存操作在并行编程中很重要, 它允许并发线程正确地对共享数据结构执行读取-修改-写入操作.

Kepler 的共享内存原子操作与 Fermi 形式相同. 两种架构都使用锁定/更新/解锁模式实现共享内存原子操作, 当多个线程高度争用共享内存中的特定位置时, 这种方式可能代价很高.

Maxwell 通过原生硬件支持 32 位整数共享内存原子操作, 以及原生共享内存 32 位和 64 位比较并交换 (CAS), 改进了原子操作. CAS 可以用来实现其他原子函数, 开销低于以软件实现这些操作的 Fermi 和 Kepler.

GP100 在 Maxwell 基础上继续改进, 还使用新的统一内存和 NVLink 特性改善原子操作 (以下段落将介绍这些特性). 全局内存中的原子加法操作扩展到 FP64 数据. CUDA 中的 atomicAdd() 函数现在适用于 32 位和 64 位整数及浮点数据. 所有浮点原子加法操作的舍入模式均为就近取偶 (此前 FP32 原子加法使用向零舍入).

<span id="section-3-6"></span>

### 3.6 GP100 的 L1/L2 缓存变化

Fermi 和 Kepler GPU 配有可配置的 64 KB 共享内存与 L1 缓存, 可以根据工作负载在 L1 和共享内存功能之间分配容量; 从 Maxwell 开始, 缓存层次结构发生了变化. GP100 SM 有自己的专用共享内存池 (64 KB/SM), 另有一个可根据工作负载兼作纹理缓存的 L1 缓存. 统一 L1/纹理缓存充当内存访问的合并缓冲区, 先收集一个 warp 中各线程请求的数据, 再将这些数据交给该 warp.

> **注意:** 一个 CUDA 线程块无法独占 64 KB 共享内存, 但两个线程块可以各使用 32 KB, 依此类推..

每个 SM 配有专用共享内存后, 应用不再需要选择最佳的 L1/共享内存划分, 每个 SM 的完整 64 KB 始终可供共享内存使用. GP100 配有统一的 4096 KB L2 缓存, 可在整个 GPU 上高效, 高速地共享数据. 相比之下, GK110 的 L2 缓存为 1536 KB, GM200 则配有 3072 KB L2 缓存. 芯片上缓存增多后, 对 GPU DRAM 的请求减少, 从而降低整板功耗和内存带宽需求并提高性能.

<span id="section-3-7"></span>

### 3.7 GPUDirect 改进

无论是处理海量地质数据, 还是研究复杂科学问题的解法, 都需要能够提供最高数据吞吐量和最低延迟的计算平台. GPUDirect 能让同一台计算机中的 GPU 或跨网络位于不同服务器中的 GPU 直接交换数据, 无需经过 CPU/系统内存.

Kepler GK110 引入的 GPUDirect RDMA 特性允许 InfiniBand (IB) 适配器, 网络接口卡 (NIC) 和 SSD 等第三方设备直接访问同一系统中多个 GPU 的内存, 省去不必要的内存复制, 大幅降低 CPU 开销, 并显著减少 MPI 向 GPU 内存发送和从 GPU 内存接收消息的延迟. 它还降低系统内存带宽需求, 释放 GPU DMA 引擎供其他 CUDA 任务使用.

在通过 PCIe 从源 GPU 内存读取数据并写入目标 NIC 内存时, GP100 将实际 RDMA 带宽翻倍. GPUDirect 带宽翻倍对许多用例都很重要, 尤其是深度学习. 深度学习机器的 GPU 与 CPU 比例很高 (有时每个 CPU 配 8 个 GPU), 因此 GPU 必须能与 I/O 快速交互, 而无需退回 CPU 传输数据.

<span id="section-3-8"></span>

### 3.8 Compute Capability

GP100 GPU 支持新的 Compute Capability 6.0. [表 2](#table-02) 对比不同 NVIDIA GPU 架构的 Compute Capability 参数.

<span id="table-02"></span>

![Kepler GK110, Maxwell GM200 与 Pascal GP100 的 Compute Capability](./nvidia-pascal-architecture/table-02.png)

**表 2.** Compute Capability: GK110, GM200 与 GP100 对比

<span id="section-3-9"></span>

### 3.9 Tesla P100: 全球首款采用 HBM2 的 GPU

近年来, 随着使用 GPU 加速计算应用的情况大幅增加, 许多应用对数据的需求也随之增长. GPU 正在解决规模大得多的问题, 需要大得多的数据集, 对 DRAM 带宽的需求也更高. 为满足这种原始带宽需求, Tesla P100 成为首款采用 High Bandwidth Memory 2 (HBM2) 的 GPU 加速器. HBM2 从根本上改变 DRAM 的封装方式及其与 GPU 的连接方式, 从而显著提高 DRAM 带宽.

传统 GDDR5 GPU 板卡设计需要在 GPU 周围布置大量分立内存芯片, HBM2 则包含一个或多个由多颗内存裸片垂直叠放形成的堆栈. 内存裸片通过硅通孔和微凸点形成的微型导线连接. 一颗 8 Gb HBM2 裸片包含超过 5,000 个硅通孔. 随后使用无源硅中介层连接内存堆栈与 GPU 裸片. HBM2 堆栈, GPU 裸片和硅中介层共同封装在一个 55mm x 55mm BGA 封装内. [图 9](#figure-09) 给出 GP100 和两个 HBM2 堆栈的示意图, [图 10](#figure-10) 则是实际 P100 中 GPU 和内存的显微照片.

<span id="figure-09"></span>

![GP100 与相邻 HBM2 堆栈的剖面](./nvidia-pascal-architecture/figure-09.png)

**图 9.** GP100 与相邻 HBM2 堆栈的剖面示意图

<span id="figure-10"></span>

![P100 HBM2 堆栈与 GP100 GPU 的剖面显微照片](./nvidia-pascal-architecture/figure-10.png)

**图 10.** P100 HBM2 堆栈与 GP100 GPU 的剖面显微照片

[图 10](#figure-10) 的显微照片展示 Tesla P100 HBM2 堆栈和 GP100 GPU 的剖面. 左上方的 HBM2 堆栈由 5 颗裸片构成-1 颗基底裸片和上方的 4 颗内存裸片. 最上方的内存裸片层很厚. 组装时会将顶部裸片和 GPU 研磨到相同高度, 为散热器提供共面表面.

与前代 HBM1 相比, HBM2 提供更高的内存容量和内存带宽. HBM2 每个堆栈支持 4 颗或 8 颗 DRAM 裸片, HBM1 每个堆栈只支持 4 颗 DRAM 裸片. HBM2 每颗 DRAM 裸片最高支持 8 Gb, HBM1 每颗裸片仅支持 2 Gb. HBM1 每个堆栈的带宽上限为 125 GB/sec, P100 采用 HBM2 后每个堆栈可支持 180GB/sec.

如 GP100 完整芯片框图 ([图 7](#figure-07)) 所示, GP100 GPU 连接 4 个 HBM2 DRAM 堆栈. 每个 HBM2 堆栈连接 2 个 512 位内存控制器, 构成有效位宽 4096 位的 HBM2 内存接口. 首批 Tesla P100 加速器将配备 4 个由 4 颗裸片组成的 HBM2 堆栈, HBM2 内存总容量为 16 GB.

<span id="section-3-9-1"></span>

#### 3.9.1 内存韧性

HBM2 内存的另一项优势是原生支持纠错码 (ECC) 功能. ECC 为易受数据损坏影响的计算应用提供更高可靠性. 在 GPU 处理超大数据集和/或长时间运行应用的大规模集群计算环境中, 它尤其重要.

ECC 技术可在单位软错误影响系统前检测并纠正它. 相比之下, GDDR5 不为内存内容提供内部 ECC 保护, 只能检测 GDDR5 总线上的错误. 内存控制器或 DRAM 本身的错误无法检测.

GK110 Kepler GPU 从可用内存中划出一部分显式存储 ECC, 从而为 GDDR5 提供 ECC 保护. GDDR5 总容量的 6.25% 会留作 ECC 位. 以 12 GB Tesla K40 为例, 总内存中的 750 MB 留作 ECC, 因而开启 ECC 后可用内存为 11.25 GB (总计 12 GB). 与不使用 ECC 相比, 访问 ECC 位还会使典型工作负载的内存带宽降低 12-15%. HBM2 原生支持 ECC, 因此 Tesla P100 不会承担这种容量开销, ECC 可以始终启用而不损失带宽. 与 GK110 GPU 一样, GP100 GPU 的寄存器文件, 共享内存, L1 缓存和 L2 缓存, 以及 Tesla P100 加速器的 HBM2 DRAM 都由单错纠正双错检测 (SECDED) ECC 码保护.

<span id="section-3-10"></span>

### 3.10 Tesla P100 设计

Tesla P100 系统架构最令人兴奋的新特性之一是它的新板卡设计, 可容纳 GP100 GPU 和 HBM2 内存堆栈, 并提供 NVLink 与 PCIe 连接. 工作站, 服务器和大型计算系统都可以使用一个或多个 P100 加速器. P100 加速器尺寸为 140mm x 78mm, 配有高效稳压器, 为 GPU 提供所需的多种电压. P100 的额定功率为 300W.

[图 11](#figure-11) 展示 Tesla P100 加速器正面, [图 12](#figure-12) 展示背面.

<span id="figure-11"></span>

![Tesla P100 加速器正面](./nvidia-pascal-architecture/figure-11.png)

**图 11.** Tesla P100 加速器正面

<span id="figure-12"></span>

![Tesla P100 加速器背面](./nvidia-pascal-architecture/figure-12.png)

**图 12.** Tesla P100 加速器背面

<span id="section-4"></span>

## 4 NVLink 高速互连

NVLink 是 NVIDIA 面向 GPU 加速计算推出的新型高速互连技术. 它已在 Tesla P100 加速器板卡和 Pascal GP100 GPU 中实现, 可显著提高 GPU 间通信和 GPU 访问系统内存的性能.

高性能计算集群的节点通常使用多个 GPU. 如今每个节点配备最多 8 个 GPU 十分常见, 在多处理器系统中, 高性能互连极有价值. NVIDIA 设计 NVLink 的目标是为 GPU 提供一种带宽远高于 PCI Express Gen 3 (PCIe) 的互连, 同时兼容 GPU ISA, 以支持共享内存多处理工作负载.

<span id="figure-13"></span>

![配备 8 个 NVIDIA Tesla P100 GPU 的 NVIDIA DGX-1](./nvidia-pascal-architecture/figure-13.png)

**图 13.** 配备 8 个 NVIDIA Tesla P100 GPU 的 NVIDIA DGX-1

使用 NVLink 连接 GPU 后, 程序既可直接在连接到其他 GPU 的内存上执行, 也可在本地内存上执行, 并且内存操作仍能保持正确 (例如完整支持 Pascal 的原子操作).

NVLink 使用 NVIDIA 新型高速信号互连 NVHS. NVHS 通过最高运行于 20 Gb/sec 的差分对传输数据. 8 条这样的差分连接构成一个单向发送数据的 Sub-Link, 两个方向各一个 sub-link 构成连接两个处理器 (GPU 到 GPU 或 GPU 到 CPU) 的 Link. 单个 Link 在端点之间支持最高 40 GB/sec 双向带宽. 多个 Link 可以组合为 Gang, 在处理器之间提供更高带宽的连接. Tesla P100 的 NVLink 实现最多支持 4 个 Link, 可以组成总计最高 160 GB/sec 双向带宽的组合配置, 如[图 14](#figure-14) 和[图 15](#figure-15) 所示.

<span id="section-4-1"></span>

### 4.1 NVLink 配置

NVLink 可以组成多种拓扑, 不同配置可针对不同应用优化. 本节讨论以下 NVLink 配置:

- GPU 到 GPU 的 NVLink 连接
- CPU 到 GPU 的 NVLink 连接

<span id="section-4-1-1"></span>

#### 4.1.1 GPU 到 GPU 的 NVLink 连接

[图 14](#figure-14) 展示一个 8-GPU 混合立方网格, 其中包含两组通过 NVLink 完全互连的四 GPU 单元, 两组单元之间也有 NVLink 连接, 而且每组内的 GPU 通过 PCIe 直接连接各自的 CPU. 使用独立 NVLink 跨越两个四 GPU 单元之间的距离, 可以减轻每个 CPU 的 PCIe 上行链路压力, 同时避免经由系统内存和 CPU 间链路传输数据.

<span id="figure-14"></span>

![8-GPU 混合立方网格架构](./nvidia-pascal-architecture/figure-14.png)

**图 14.** 8-GPU 混合立方网格架构

8-GPU 混合立方网格的每一半都能作为共享内存多处理器运行, 远端节点也能通过对等 DMA 共享内存. 所有 GPU 到 GPU 流量都通过 NVLINK 后, PCIe 可完全用于连接 NIC (图中未显示) 或访问系统内存流量. 这种配置通常推荐用于通用深度学习应用, NVIDIA 新型 DGX-1 服务器采用了这一配置.

[图 15](#figure-15) 展示一个四 GPU 集群, 每个 GPU 通过一条 NVLink 连接到其他各 GPU. 在这种情况下, 对等端可以 40 GB/sec 双向通信 (双链路的双向带宽为 80GB/sec), 支持 GPU 之间可靠地共享数据.

<span id="figure-15"></span>

![4 个 GPU 通过 NVLink 互连且 CPU 通过 PCIe 连接](./nvidia-pascal-architecture/figure-15.png)

**图 15.** NVLink 连接 4 个 GPU, CPU 通过 PCIe 连接

<span id="section-4-1-2"></span>

#### 4.1.2 CPU 到 GPU 的 NVLink 连接

NVLink 主要用于连接多个 NVIDIA Tesla P100 加速器, 但也能作为 CPU 到 GPU 的互连. 例如, Tesla P100 加速器可以连接采用 NVIDIA NVLink 技术的 IBM POWER8. POWER8 with NVLink™ 支持 4 条 NVLink.

[图 16](#figure-16) 展示单个 GPU 连接支持 NVLink 的 CPU. 在这种配置中, GPU 访问系统内存的双向带宽最高可达 160 GB/sec, 是 PCIe 的 5 倍.

<span id="figure-16"></span>

![1 个 GPU 通过 4 条 NVLink 连接 CPU](./nvidia-pascal-architecture/figure-16.png)

**图 16.** GPU 到 CPU 的 NVLink 互连

[图 17](#figure-17) 展示 CPU 通过 2 条 NVLink 连接每个 GPU 的系统. 每个 GPU 剩余的 2 条链路用于对等通信.

<span id="figure-17"></span>

![2 个 GPU 与 1 个 CPU 连接, 每条 CPU-GPU 路径有 2 条 NVLink](./nvidia-pascal-architecture/figure-17.png)

**图 17.** 2 个 GPU 和 1 个 CPU 以 80 GB/sec 双向带宽连接

<span id="section-4-2"></span>

### 4.2 Tesla P100 的 NVLink 接口

如 Tesla P100 设计一节所述, P100 加速器包含 NVLink 互连. P100 有两个 400 引脚高速连接器. 其中一个连接器用于连接模块内外的 NVLink 信号, 另一个用于提供电源, 控制信号和 PCIe I/O.

Tesla P100 加速器可以安装到更大的 GPU 载板或系统板上. GPU 载板负责连接其他 P100 加速器或 PCIE 控制器. P100 加速器比传统 GPU 板卡更小, 客户可以轻松构建 GPU 密度前所未有的服务器. NVLink 提供额外带宽后, GPU 间通信不会受 PCIe 带宽限制而形成瓶颈, 从而带来此前无法实现的 GPU 集群方案.

在 GPU 架构接口层面, NVLink 控制器通过另一个名为 High-Speed Hub (HSHUB) 的新模块与 GPU 内部通信. HSHUB 可以直接访问 GPU 全局交叉开关和其他系统组件, 例如能以 NVLink 峰值速率将数据移入和移出 GPU 的 High-Speed Copy Engine (HSCE). [图 18](#figure-18) 展示 NVLink 与 HSHUB 以及 GP100 GPU 中若干高层模块的关系.

<span id="figure-18"></span>

![NVLink 与 GP100 中 HSHUB, 交叉开关, 复制引擎和内存控制器的关系](./nvidia-pascal-architecture/figure-18.png)

**图 18.** NVLink 与 GP100 中其他主要模块的关系

更多详情请参阅[第 9 节](#section-9).

<span id="section-5"></span>

## 5 统一内存

统一内存是 CUDA 编程模型的一项重要特性, 它提供一个访问系统中全部 CPU 与 GPU 内存的统一虚拟地址空间, 大幅简化 GPU 编程和应用向 GPU 的移植. Pascal GP100 的新特性扩展了统一内存的能力并提高其性能, 是 GPU 计算的一项重大进展.

现代处理器获得高性能的要点是让硬件计算单元能够快速, 直接地访问数据. 多年来, NVIDIA 持续改善并简化 GPU 内存访问与数据共享, 让 GPU 程序员可以把更多精力放在构建并行应用上, 减少对内存分配和 GPU 与 CPU 间数据传输的管理.

许多年来, 在典型 PC 或集群节点中, CPU 与各 GPU 的内存在物理上彼此独立, 由通常为 PCIe 的互连总线隔开. 在早期 CUDA 版本中, GPU 程序员必须显式管理 CPU 与 GPU 的内存分配和数据传输. 由于 CPU 与 GPU 共享的任何数据都需要分配两份, 一份位于系统内存, 另一份位于 GPU 内存, 这项工作颇具挑战. 程序员必须使用显式内存复制调用, 在两者之间移动最新数据. 在正确时间把数据放到正确位置会增加应用复杂度, 也抬高新 GPU 程序员的学习门槛.

对于稀疏内存访问, 显式数据传输也会损失性能. 例如, CPU 只随机写入几个字节后便将整个数组复制回 GPU, 会增加传输延迟开销. 管理内存传输, 改善内存局部性以及使用异步内存复制等技术可以提高性能, 但都要求编程时投入更多精力.

<span id="section-5-1"></span>

### 5.1 统一内存的历史

2009 年推出的 NVIDIA Fermi GPU 架构实现了统一 GPU 地址空间, 覆盖三种主要 GPU 内存空间 (线程私有局部内存, 线程块共享内存和全局内存). 这一统一地址空间仅适用于 GPU 内存寻址, 主要通过一条加载/存储指令和一个指针地址访问任意 GPU 内存空间 (全局, 局部或共享内存), 而不再为每种空间使用不同指令和指针, 从而简化编译. 它还支持完整的 C 和 C++ 指针, 在当时是一项重大进展.

CUDA 4 于 2011 年引入统一虚拟寻址 (UVA), 为 CPU 与 GPU 内存提供单一虚拟内存地址空间, 使 GPU 代码可以访问指向系统中任意位置的指针, 无论它位于 GPU 内存 (同一或另一 GPU), CPU 内存还是片上共享内存. UVA 支持零复制内存, 即 GPU 代码无需 memcpy, 可以直接通过 PCIe 访问的固定 CPU 内存. 零复制提供统一内存的部分便利, 却没有它的性能, 因为 GPU 总要通过低带宽, 高延迟的 PCIe 访问它.

CUDA 6 引入统一内存, 创建 CPU 和 GPU 共享的托管内存池, 跨越 CPU 与 GPU 之间的鸿沟. CPU 和 GPU 都可通过一个指针访问托管内存. CUDA 系统软件会自动在 GPU 与 CPU 之间迁移分配在统一内存中的数据, 因此 CPU 上运行的代码将其视为 CPU 内存, GPU 上运行的代码将其视为 GPU 内存. 但 CUDA 6 统一内存受 Kepler 和 Maxwell GPU 架构特性限制: CPU 接触过的全部托管内存必须在每次内核启动前与 GPU 同步. CPU 和 GPU 不能同时访问一块托管内存分配, 而统一内存地址空间也受限于 GPU 物理内存容量.

<span id="figure-19"></span>

![CPU 与 GPU 访问 CUDA 6 统一内存](./nvidia-pascal-architecture/figure-19.png)

**图 19.** CUDA 6 统一内存

[图 20](#figure-20) 展示 CUDA 6 统一内存如何通过单一数据指针简化代码向 GPU 的移植, 使显式 CPU-GPU 内存复制从必要条件变成一项优化.

<span id="figure-20"></span>

![展示 CUDA 6 统一内存如何简化 GPU 移植的代码对比](./nvidia-pascal-architecture/figure-20.png)

**图 20.** CUDA 6 统一内存简化代码向 GPU 的移植

(具体方式是提供新的托管内存分配器, 返回 CPU 或 GPU 代码均可访问的数据指针.)

<span id="section-5-2"></span>

### 5.2 Pascal GP100 统一内存

Pascal GP100 扩展 CUDA 6 统一内存的优势, 新增多项特性, 进一步简化 CPU 与 GPU 间的编程和内存共享, 也让 CPU 并行计算应用更容易移植到 GPU 上获得大幅加速. 两项主要硬件特性支持这些改进: 大地址空间和缺页能力.

GP100 扩展 GPU 寻址能力, 支持 49 位 (512 TB) 虚拟内存寻址 (注意, GP100 还支持 47 位 (128 TB) 物理内存寻址). 这足以覆盖现代 CPU 的 48 位虚拟地址空间以及 GPU 自身的内存. GP100 统一内存程序因而可以把系统中所有 CPU 和 GPU 的完整地址空间作为单一虚拟地址空间访问, 不受任何一个处理器物理内存容量限制 (见[图 21](#figure-21)).

GP100 的内存缺页支持是一项重要的新特性, 使统一内存的功能更加无缝. 缺页与系统级虚拟地址空间结合后带来多项好处. 首先, 缺页意味着 CUDA 系统软件不必在每次内核启动前把所有托管内存分配同步到 GPU. 如果 GPU 上运行的内核访问一个不在其内存中的页面, 就会触发缺页, 使该页面按需自动迁移到 GPU 内存. 另一种方式是将该页面映射到 GPU 地址空间, 通过 PCIe 或 NVLink 互连访问 (访问时映射有时比迁移更快). 请注意, 统一内存覆盖整个系统: GPU (以及 CPU) 可以从 CPU 内存或系统中其他 GPU 的内存触发缺页并迁移内存页面.

<span id="figure-21"></span>

![跨越 CPU 与 GPU 物理内存的统一虚拟内存](./nvidia-pascal-architecture/figure-21.png)

**图 21.** Pascal GP100 统一内存不受 GPU 物理内存容量限制.

借助新的缺页机制, 统一内存可保证全局数据一致性. 这意味着在 GP100 上, CPU 与 GPU 无需程序员同步即可访问统一内存分配. Kepler 和 Maxwell GPU 不允许这样做, 因为 GPU 内核运行时若 CPU 访问统一内存分配, 系统无法保证一致性.

> **注意:** 与任何并行应用一样, 开发者需要确保正确同步, 避免处理器之间的数据冒险.

最后, 在支持的操作系统平台上, 使用默认 OS 分配器 (例如 malloc 或 new) 分配的内存可由 GPU 代码和 CPU 代码通过同一指针访问 (见[图 22](#figure-22)). 在这些系统中, 统一内存可以成为默认内存: 无需特殊分配器, 也无需创建专门的托管内存池. GP100 的大虚拟地址空间和缺页能力还让应用能够访问完整的系统虚拟内存. 这意味着应用可以超额分配内存系统: 换言之, 它们能够分配, 访问和共享大于系统物理总容量的数组, 对超大数据集进行核外处理.

要让统一内存配合系统分配器使用, 操作系统需要做出一些修改. NVIDIA 正与 Red Hat 合作, 并在 Linux 社区中推进这项功能.

<span id="figure-22"></span>

![Pascal 统一内存使用默认系统分配器](./nvidia-pascal-architecture/figure-22.png)

**图 22.** 在操作系统支持下, Pascal 可以使用默认系统分配器支持统一内存.

(此处只需 malloc, 即可分配系统中任意 CPU 或 GPU 都能访问的内存.)

<span id="section-5-3"></span>

### 5.3 统一内存的优势

程序员主要通过两方面受益于统一内存.

- 更简单的编程与内存模型. 统一内存将显式设备内存管理从必要条件变为一项优化, 降低 GPU 并行编程的入门门槛. 程序员可以专注开发并行代码, 不必困在设备内存分配和复制的细节中. 学习 GPU 编程和将现有代码移植到 GPU 都因此变得更容易.
- 但它并非只适合初学者; 统一内存也让复杂数据结构和 C++ 类更容易在 GPU 上使用. 在默认系统分配器支持统一内存的系统中, 任意层次化或嵌套数据结构都能由系统中的任何处理器自动访问. 借助 GP100, 应用可以对大于系统内存总容量的数据集进行核外操作.
- 通过数据局部性提高性能. 统一内存按需在 CPU 和 GPU 之间迁移数据, 在提供全局共享数据便利性的同时, 也可获得 GPU 本地数据的性能. CUDA 驱动程序和运行时会在内部处理这一功能的复杂性, 使应用代码更容易编写. 迁移的目的在于获得各处理器的完整带宽; HBM2 的高内存带宽对于满足 GP100 GPU 的计算吞吐量至关重要. GP100 支持缺页, 即使程序进行稀疏数据访问, 无法预先得知 CPU 或 GPU 将访问哪些页面, 或者 CPU 和 GPU 同时访问同一数组分配的不同部分, 也能保证局部性.

重要的是, CUDA 程序员仍然拥有所需工具, 可以在必要时显式优化数据管理和 CPU-GPU 并发: CUDA 8 将引入实用 API, 向运行时提供内存使用提示并进行显式预取. 这些工具提供与显式内存复制和固定 API 相同的能力, 却不会重新受制于显式 GPU 内存分配.

<span id="section-6"></span>

## 6 计算抢占

Pascal GP100 的新计算抢占特性允许在指令级粒度中断 GPU 上运行的计算任务, 并将其上下文换出到 GPU DRAM. 随后可换入并运行其他应用, 再换回原任务的上下文, 从中断处继续执行.

计算抢占解决长时间运行或行为不良的应用独占系统这一严重问题. 系统等待任务完成时可能失去响应, 任务还可能超时和/或被 OS 或 CUDA 驱动程序终止. 在 Pascal 之前, 如果计算和显示任务在同一 GPU 上运行, 长时间运行的计算内核可能使 OS 和其他可视化应用失去响应和交互能力, 直至内核超时. 因此, 程序员要么必须安装专用的纯计算 GPU, 要么要针对前代 GPU 的限制仔细编写应用, 将工作负载拆成更短的执行时间片, 避免超时或被 OS 终止.

事实上, 许多应用确实需要长时间运行的进程. 借助 GP100 的计算抢占, 这些应用在处理大型数据集或等待特定条件时, 现在可以按实际需要一直运行, 可视化应用仍能保持流畅和交互性, 而程序员不必费力把代码限制在很短的时间片中.

计算抢占还支持在单 GPU 系统上交互调试计算内核. 这项能力对开发者的工作效率很重要. 相比之下, Kepler GPU 架构只提供计算内核中线程块级别, 粒度更粗的抢占. 这种块级抢占要求线程块中的所有线程全部完成后, 硬件才能将上下文切换到另一上下文. 但使用调试器时, GPU 断点若在该线程块内的一条指令上触发, 线程块便尚未完成, 从而阻止块级抢占. Kepler 和 Maxwell 仍可在编译期间添加插桩, 提供调试器的核心功能, 而 GP100 能支持更可靠, 更轻量的调试器实现.

<span id="section-7"></span>

## 7 NVIDIA DGX-1 深度学习超级计算机

数据科学家和人工智能研究人员要求深度学习系统具备准确性, 易用性和速度. 更快的训练与迭代最终意味着更快的创新和更短的上市时间. NVIDIA DGX-1 是全球首台专为深度学习打造的服务器, 集成完整的硬件和软件, 可以快速, 轻松地部署. 它具有最高 170 FP16 TFLOPS 的革新性性能, 可显著缩短训练时间, 成为首台装在机箱里的 AI 超级计算机.

NVIDIA DGX-1 是首台使用 NVLink 互连 Tesla P100 加速器的服务器. DGX-1 系统提供 8 个 Tesla P100 加速器的配置, 在 3U 机架式机箱中采用高性能, 高可靠性组件, 可以单独使用或集成到集群中.

8-GPU 配置包含两组通过 NVLink 完全互连的四 P100 GPU 单元, 两组之间另有 4 条 NVLink, 组成[图 14](#figure-14) 所示的混合立方网格拓扑. 每组内的各 GPU 还通过 PCIe 直接连接到与 CPU 相连的 PCIe 交换机.

NVIDIA DGX-1 ([图 23](#figure-23)) 将强大硬件与针对深度学习定制的软件结合, 为开发者和研究人员提供一套交钥匙解决方案, 用于高性能 GPU 加速的深度学习应用开发, 测试和网络训练.

<span id="figure-23"></span>

![NVIDIA DGX-1 服务器](./nvidia-pascal-architecture/figure-23.png)

**图 23.** NVIDIA DGX-1 服务器

<span id="section-7-1"></span>

### 7.1 一个机箱相当于 250 台服务器

[表 3](#table-03) 对比双路 Xeon 系统与 DGX-1 服务器的 Alexnet 训练时间. 可以看到, 无论原始 TFLOPS 还是节点总带宽, DGX-1 的原始处理能力都远超双路 Xeon. 双路 Xeon 需要超过 250 个节点才能在两小时周转时间 (TAT) 内训练 Alexnet, DGX-1 只需一个节点!

<span id="table-03"></span>

![Pascal GP100 与 Xeon 系统的 Alexnet 训练时间](./nvidia-pascal-architecture/table-03.png)

**表 3.** Alexnet 训练时间: Pascal GP100 与 Xeon 对比

<span id="section-7-2"></span>

### 7.2 一年内将 DNN 加速 12 倍

[图 24](#figure-24) 对比 Pascal DGX-1 和 Maxwell 在一年内对 Alexnet 的 DNN 加速.

<span id="figure-24"></span>

![4 个 Maxwell GPU 与 8 个 Pascal GPU 的 Alexnet 训练时间](./nvidia-pascal-architecture/figure-24.png)

**图 24.** 自 NVIDIA 上届 GTC 活动以来, Pascal DGX-1 与 Maxwell 的 DNN 年度加速对比

<span id="section-7-3"></span>

### 7.3 DGX-1 软件特性

DGX-1 基础 OS 软件使用户可以轻松快速地开始深度学习工作. DGX-1 软件栈基于针对 GPU 调优的行业标准 Linux 发行版, 包含 CUDA 8.0 和最新版 NVIDIA 深度学习 SDK, 使深度学习应用能够利用 Tesla P100 的高性能特性, 加速所有主要深度学习框架及使用这些框架的应用.

<span id="section-7-4"></span>

### 7.4 NVIDIA DGX-1 系统规格

NVIDIA DGX-1 是全球首台专为深度学习打造的服务器, 集成完整的硬件和软件, 可以快速, 轻松地部署. 它的革新性性能显著缩短训练时间, 使 NVIDIA DGX-1 成为首台装在机箱里的 AI 超级计算机. [表 4](#table-04) 列出 NVIDIA DGX-1 的系统规格.

<span id="table-04"></span>

![NVIDIA DGX-1 系统规格](./nvidia-pascal-architecture/table-04.png)

**表 4.** NVIDIA DGX-1 系统规格

<span id="section-8"></span>

## 8 结论

NVIDIA 采用 Pascal 架构的新型 NVIDIA Tesla P100 GPU 加速器汇集多项突破, 让客户能够计算此前无法解决的问题. NVIDIA Tesla P100 从上到下都有惊人的创新, 包括计算性能, 内存带宽, 容量, 连接和能效, 具备成为下一代 HPC 与 AI 系统计算引擎所需的能力.

<span id="section-9"></span>

## 9 附录 A: NVLink 信号与协议技术

NVLink 使用 NVIDIA 高速信号技术 (NVHS). 每个信号对以 20 Gbit/sec 进行差分传输. 每个方向的 8 个差分对组合成一条链路, 这是基本构件. 单条链路的原始双向带宽为 40 GB/sec. 信号采用 NRZ (Non-Return-to-Zero). 链路采用直流耦合, 差分阻抗为 85 Ohms. 链路可容忍极性反转和通道反转, 便于有效进行 PCB 布线. 在裸片上, 数据以 1.25GHz 数据速率, 使用 128 位 Flit (Flow control digit) 从 PHY (物理层电路) 发送到 NVLink 控制器. NVHS 使用嵌入式时钟. 接收端使用恢复出的时钟捕获传入数据.

<span id="section-9-1"></span>

### 9.1 NVLink 控制器各层

NVLink 控制器包含三层: 物理层 (PL), 数据链路层 (DL) 和事务层 (TL). 协议使用变长数据包, 包大小从 1 个 flit (例如简单的读取请求命令) 到 18 个 flit (带数据的写入请求, 用扩展地址传输 256B 数据). [图 25](#figure-25) 展示 NVLink 各层和链路, 包括物理层 (PHY), 数据链路层 (DL) 和事务层 (TL).

<span id="section-9-1-1"></span>

#### 9.1.1 物理层 (PL)

PL 与 PHY 对接. PL 负责去偏斜 (跨全部 8 条通道), 成帧 (确定各数据包的起点), 加扰/解扰 (确保足够的位跳变密度以支持时钟恢复), 极性反转, 通道反转, 并将收到的数据交给数据链路层. [图 25](#figure-25) 展示 NVLink 各层和链路, 包括物理层 (PHY), 数据链路层 (DL) 和事务层 (TL).

<span id="figure-25"></span>

![跨 4 条链路的 NVLink 物理层, 数据链路层和事务层](./nvidia-pascal-architecture/figure-25.png)

**图 25.** NVLink 各层和链路: 物理层 (PHY), 数据链路层 (DL), 事务层 (TL)

<span id="section-9-1-2"></span>

#### 9.1.2 数据链路层 (DL)

数据链路层主要负责通过链路可靠传输数据包. 待传输数据包使用 25 位 CRC (循环冗余校验) 保护. 发送的数据包会保存在重放缓冲区, 直至链路另一端的接收器发出肯定确认 (ACK). 如果 DL 在传入数据包中检测到 CRC 错误, 就不会发送 ACK, 并准备接收重传数据. 与此同时, 发送器因未收到 ACK 而超时, 从重放缓冲区发起数据重传. 数据包只有在得到确认后才会从重放缓冲区中移除. 25 位 CRC 可以检测最多 5 个随机位错误, 或任意通道中最长 25 位的突发错误. CRC 根据当前包头和前一个有效载荷 (如果有) 计算.

DL 还负责链路启动和维护. DL 将数据继续发送到事务层 (TL).

<span id="section-9-1-3"></span>

#### 9.1.3 事务层

事务层处理同步, 链路流量控制和虚拟通道, 还能聚合多条链路, 在处理器之间提供很高的通信带宽.

<span id="section-10"></span>

## 10 附录 B: 使用 GPU 加速深度学习与人工智能

计算领域的终极目标是人工智能: 构建一台足够智能, 无需明确指令即可自行学习的机器. 深度学习是实现现代 AI 的必要组成部分.

深度学习让 AI 大脑能够感知周围世界; 机器学习并最终自行做出决策. 训练机器做到这一点需要海量数据, 还需要高度复杂的深度神经网络来处理全部数据. 2012 年, Google 的深度学习项目 Google Brain 通过观看 YouTube 视频学会识别猫. 但它需要 Google 一座数据中心内供电和冷却的 2,000 个 CPU (16,000 个 CPU 核心). 很少有组织拥有如此规模的机器. 大约同一时期, NVIDIA Research 与 Stanford University 合作使用 GPU 进行深度学习. 最终发现, 12 个 NVIDIA GPU 可以提供 2,000 个 CPU 的深度学习性能.

许多人认为, 2012 年 Krizhevsky, Sutskever 和 Hinton 参加 ImageNet 竞赛的作品是深度学习革命的起点. 该作品采用一种如今称为 "AlexNet" 的卷积神经网络, 借助 GPU 的并行处理性能, 大幅领先整个传统计算机视觉竞赛. 这是人工智能与深度学习历史上的里程碑事件. Krizhevsky 及其团队没有编写任何计算机视觉代码. 相反, 他们的计算机使用深度学习自行学会识别图像. 他们设计了一个神经网络 (AlexNet), 用 100 万张示例图像训练它, 训练需要在 NVIDIA GPU 上执行数万亿次数学运算. Krizhevksy 的 AlexNet 击败了最佳人工编写软件.

此后, 在 GPU 上运行的深度神经网络 (DNN) 逐一攻克了多种算法领域, 尤其是计算机视觉, 以及更广泛的机器感知. 潜在用例无穷无尽: 从自动驾驶汽车到更快的药物开发, 从在线图像数据库的自动图像描述到视频聊天应用中的智能实时语言翻译, 只要机器与人类世界交互, 深度学习就带来令人兴奋的机会. 如今, 使用深度神经网络离不开 GPU. 各地的深度学习用户从 CPU 转向一个或多个大规模并行 GPU 加速器, 大幅缩短训练时间.

<span id="section-10-1"></span>

### 10.1 深度学习概述

深度学习是一种对人脑神经学习过程建模的技术, 它不断学习, 变得更聪明, 随时间更快地给出更准确的结果. 成人起初会教儿童正确识别各种形状并进行分类, 儿童最终无需指导也能识别形状. 类似地, 深度学习或神经学习系统必须接受物体识别与分类训练, 才能更智能, 更高效地识别基本物体, 被遮挡物体等, 同时为物体赋予上下文.

在最简单的层面上, 人脑中的神经元会查看送入的各种输入, 为各输入分配重要性级别, 再把输出传递给其他神经元处理.

[图 26](#figure-26) 所示的感知机是最基本的神经网络模型, 类似人脑中的神经元. 如图所示, 感知机有多个输入, 代表它正在学习识别和分类的物体的各种特征; 每个特征会根据它对物体形状定义的重要程度获得相应权重.

<span id="figure-26"></span>

![感知机的输入, 权重, 求和, 激活函数与输出](./nvidia-pascal-architecture/figure-26.png)

**图 26.** 感知机是最简单的神经网络模型

例如, 考虑训练一个感知机识别手写数字 0. 显然, 不同书写风格能用许多方式写出数字 0. 感知机取得数字 0 的图像, 将其分成不同部分, 并把这些部分分配给特征 x1 到 x4. 数字 0 的右上方曲线可以分配给 x1, 底部下方曲线分配给 x2, 依此类推. 与特定特征关联的权重决定该特征对正确判断手写数字是否为 0 有多重要. 图中央的绿色区域是感知机计算图像中所有特征加权和的位置, 用以判断数字是否为 0. 随后对结果应用一个函数, 输出该数字是否为 0 的真值或假值.

神经网络的主要环节是训练网络做出更好的预测. 用于检测手写数字 0 的感知机模型 ([图 26](#figure-26)) 首先为定义数字 0 的每项特征分配一组权重, 从而接受训练. 随后向感知机提供数字 0, 检查它能否正确识别. 数据在网络中流动, 直至得出该数字是否为 0 的结论, 这一过程称为*前向传播*阶段. 如果神经网络未能正确识别数字, 就需要了解识别错误的原因和误差大小, 并调整各项特征的权重, 直至感知机正确识别数字 0. 权重还需进一步调整, 直至它能正确识别各种书写风格的数字 0. 把误差反馈回来并调整定义数字 0 的各项特征权重, 这一过程称为*反向传播*. 图中公式看似复杂, 但基本上只是上述训练过程的数学表达.

感知机虽然是很简单的神经网络模型, 但基于类似概念的高级多层神经网络如今已被广泛使用. 网络经过训练, 能够正确识别物体并进行分类后, 就会部署到实际环境中, 反复执行*推理*计算. 推理 (DNN 从给定输入中提取有用信息的过程) 示例包括: 识别存入 ATM 的支票上的手写数字, 识别 Facebook 照片中的朋友, 为 5,000 多万 Netflix 用户推荐电影, 识别并分类无人驾驶汽车中的各类车辆, 行人和道路危险, 或实时翻译人类语音.

[图 27](#figure-27) 所示的多层神经网络模型可以包含多个相互连接的复杂类感知机节点, 每个节点查看若干输入特征, 并将其输出馈送到接下来几层相互连接的节点.

<span id="figure-27"></span>

![多层神经网络为识别 Audi A7 学到的层次化特征](./nvidia-pascal-architecture/figure-27.png)

**图 27.** 复杂的多层神经网络模型需要更多计算能力

在[图 27](#figure-27) 所示模型中, 第一层神经模型把图像分成多个区域, 寻找线条和角等基本模式; 第二层组合这些线条, 寻找车轮, 挡风玻璃和后视镜等更高层模式; 下一层识别车辆类型; 最后几层识别特定品牌的车型 (这里是 Audi A7).

全连接层之外, 神经网络还可以使用卷积层. 卷积层中的一个神经元只与下一层某个小区域内的神经元相连. 这个区域通常可能是 5×5 的神经元网格 (也可能是 7×7 或 11×11). 该网格的大小称为滤波器尺寸. 因而, 卷积层可以看作对其输入执行卷积. 这种连接模式模拟了大脑感知区域中的模式, 例如视网膜神经节细胞或初级视觉皮层中的细胞.

在 DNN 卷积层中, 该层每个神经元使用相同的滤波器权重. 一个卷积层通常实现为许多 "子层", 每个子层使用不同的滤波器. 一个卷积层可以使用数百种不同的滤波器. 可以把 DNN 卷积层理解为同时对输入执行数百种不同的卷积, 并把这些卷积的结果提供给上一层. 包含卷积层的 DNN 称为卷积神经网络 (CNN).

<span id="section-10-2"></span>

### 10.2 NVIDIA GPU: 深度学习的引擎

当时最先进的 DNN 和 CNN 可以有数百万乃至超过 10 亿个参数需要通过反向传播调整. DNN 还需要大量训练数据才能达到高准确率, 这意味着必须让数十万到数百万个输入样本同时经过前向和反向传递.

学术界和业界现在普遍认为, GPU 在训练深度神经网络方面代表当时最高水平, 因为它在速度和能效上均优于较传统的 CPU 平台. 神经网络由大量相同神经元组成, 天生高度并行. 这种并行性自然适合 GPU, 相比纯 CPU 训练可以显著加速.

神经网络严重依赖矩阵数学运算, 复杂的多层网络需要大量浮点性能和带宽才能兼顾效率与速度. GPU 拥有数千个处理核心, 针对矩阵数学运算优化, 可提供数十到数百 TFLOPS 性能, 显然适合作为基于深度神经网络的人工智能和机器学习应用的计算平台.

NVIDIA 处在这场由 GPU 驱动的 DNN 与人工智能 (AI) 革命前沿. NVIDIA GPU 可将各种应用中的 DNN 加速 10-20 倍, 把训练时间从数周缩短到数天. 通过与该领域专家合作, 我们不断改进 GPU 设计, 系统架构, 编译器和算法. 在过去三年中, 基于 NVIDIA GPU 的计算平台已将深度学习网络训练速度提高 50 倍.

<span id="section-10-3"></span>

### 10.3 Tesla P100: 训练深度神经网络最快的加速器

NVIDIA 最新, 最先进的 Pascal GPU 架构为深度神经网络训练提供一个数量级的性能提升, 并显著缩短训练时间. Tesla P100 拥有 3584 个处理核心, 可为深度学习应用提供超过 21 TFLOPS 的 FP16 处理能力. 通过高速 NVLink 互连连接 8 个 Tesla P100 加速器, 可用性能显著提高到 170 TFLOPS/sec, 用于训练高度复杂的多层 DNN

除 HBM2 内存, 统一内存, 高速 NVLink 互连, 更大缓存和更低延迟等主要架构进步外, Tesla P100 还包含可提高深度学习性能的特性. Maxwell GPU 架构最先引入 16 位存储和算术支持, Pascal GP100 GPU 同样支持. 16 位浮点 (FP16) 存储和算术进一步提高了神经网络算法的性能, 并缩短推理时间.

<span id="section-10-4"></span>

### 10.4 完整的深度学习软件开发套件

AI 创新的发展速度极快. 编程便利性和开发者生产力至关重要. NVIDIA CUDA 平台易于编程且内容丰富, 使研究人员能够快速创新. NVIDIA 通过深度学习软件开发套件 (SDK) 提供 NVDIA DIGITS™, cuDNN 和 cuBLAS 等高性能工具与库, 为云, 数据中心, 工作站和嵌入式平台中的创新 GPU 加速机器学习应用提供动力. 开发者希望随处创建, 随处部署. 全球各地均可获得 NVIDIA GPU: 每家 PC OEM 都提供相应产品; 它们用于台式机, 笔记本电脑, 服务器和超级计算机; Amazon, Google, IBM, Facebook, Baidu 与 Microsoft 等大公司也在云端提供它们. 从互联网公司到研究机构和初创企业, 所有主要 AI 开发框架都使用 NVIDIA GPU 加速. 无论偏好哪种 AI 开发系统, GPU 加速都能让它运行得更快. 我们还为几乎每种计算形态打造 GPU, 让 DNN 能够驱动各种智能机器. GeForce 面向 PC. Tesla 面向云和超级计算机. Jetson 面向机器人和无人机. DRIVE PX 面向汽车. 它们使用同一种架构, 都能加速深度学习 (见[图 28](#figure-28)).

<span id="figure-28"></span>

![NVIDIA GPU 加速的深度学习框架和平台](./nvidia-pascal-architecture/figure-28.png)

**图 28.** 加速框架

<span id="section-10-5"></span>

### 10.5 使用 NVIDIA GPU 与 DNN 解决大数据问题

Baidu, Google, Facebook 和 Microsoft 是最早采用 NVIDIA GPU 进行深度学习和 AI 处理的公司之一. AI 技术让这些公司能够响应语音, 把语音或文本翻译成另一种语言, 识别图像并自动添加标签, 以及推荐针对每位用户定制的动态消息, 娱乐内容和产品. 初创企业和成熟公司现在竞相使用 AI 创建新产品与服务, 或改进运营. 短短两年内, 与 NVIDIA 合作开展深度学习的公司数量增长近 35 倍, 超过 3,400 家 (见[图 29](#figure-29)). 医疗, 生命科学, 能源, 金融服务, 汽车, 制造和娱乐等行业将通过从海量数据中推断洞见而获益. 随着 Facebook, Google 和 Microsoft 向所有人开放其深度学习平台, AI 驱动的应用会迅速普及. 鉴于这一趋势, Wired 最近宣告 GPU 已经崛起.

<span id="figure-29"></span>

![2013, 2014 与 2015 年同 NVIDIA 合作开展深度学习的组织](./nvidia-pascal-architecture/figure-29.png)

**图 29.** 同 NVIDIA 合作开展深度学习的组织

<span id="section-10-5-1"></span>

#### 10.5.1 自动驾驶汽车

无论是用超越人类的副驾驶辅助人类, 革新个人出行服务, 还是减少城市对大面积停车场的需求, 自动驾驶汽车都有可能带来巨大的社会效益. 驾驶很复杂. 意外随时会发生. 冻雨会把道路变成溜冰场. 通往目的地的道路可能封闭. 儿童可能突然跑到车前. 你无法编写软件, 预见自动驾驶汽车可能遇到的每一种情形. 这正是深度学习的价值; 它可以学习, 适应并改进. 我们正在使用 NVIDIA DRIVE PX, NVIDIA DriveWorks 和 NVIDIA DriveNet (见[图 30](#figure-30)) 构建端到端自动驾驶深度学习平台, 从训练系统一直延伸到车载 AI 计算机. 结果令人振奋. 拥有超人类计算机副驾驶和无人驾驶接驳车的未来不再是科幻小说.

<span id="figure-30"></span>

![NVIDIA DriveNet 在普通与雪天道路场景中的环境感知](./nvidia-pascal-architecture/figure-30.png)

**图 30.** NVIDIA DriveNet

<span id="section-10-5-2"></span>

#### 10.5.2 机器人

领先的制造机器人厂商 FANUC 最近演示了一台装配线机器人, 它学会从箱中拾取朝向随机的物体. 这台 GPU 驱动的机器人通过试错学习. 这项深度学习技术由 Preferred Networks 开发, 《The Wall Street Journal》最近在题为 Japan Seeks Tech Revival with Artificial Intelligence 的文章中报道了该公司.

<span id="section-10-5-3"></span>

#### 10.5.3 医疗与生命科学

Deep Genomics 正在应用基于 GPU 的深度学习, 了解基因变异如何导致疾病. Arterys 使用 GPU 驱动的深度学习加快医学图像分析. 其技术将部署在 GE Healthcare MRI 设备中, 帮助诊断心脏病. Enlitic 使用深度学习分析医学图像, 识别肿瘤, 几乎不可见的骨折和其他疾病.

这些只是在各领域中 GPU 和 DNN 如何革新人工智能与机器学习的少数示例. 这样的示例还有数千个.

深度学习的突破正在多个层面加快 AI 能力发展, GPU 加速的深度学习与 AI 系统及算法让该领域呈指数级进步.

<span id="section-11"></span>

## 11 声明

本白皮书提供的全部信息, 包括评注, 意见, NVIDIA 设计规格, 参考板, 文件, 图纸, 诊断, 列表及其他文档 (合称或单独称为 "材料"), 均按 "原样" 提供. 对于这些材料, NVIDIA 不作任何明示, 默示, 法定或其他形式的保证, 并明确否认对不侵权, 适销性和特定用途适用性的所有默示保证.

NVIDIA 有权随时对本规格进行修正, 修改, 增强, 改进及其他更改, 和/或停止任何产品或服务, 恕不另行通知. 客户应在下单前取得最新相关规格, 并确认这些信息当前有效且完整.

NVIDIA 产品按订单确认时提供的 NVIDIA 标准销售条款与条件销售, 除非 NVIDIA 授权代表与客户签署的单独销售协议另有约定. 对于将任何客户通用条款与条件应用于购买本规格所述 NVIDIA 产品, NVIDIA 在此明确表示反对.

NVIDIA 产品的设计, 授权或保证范围不包括医疗, 军事, 航空器, 航天或生命支持设备, 也不包括产品故障或失灵可被合理预期会造成人身伤害, 死亡, 财产或环境损害的应用. NVIDIA 不对其产品在此类设备或应用中的纳入和/或使用承担责任, 因此此类纳入和/或使用由客户自行承担风险.

NVIDIA 不声明或保证基于这些规格的产品无需进一步测试或修改即可适合任何指定用途. NVIDIA 不一定测试每个产品的所有参数. 客户独自负责确认产品适合计划中的应用, 并为该应用进行必要测试, 避免应用或产品出现故障. 客户产品设计中的缺陷可能影响 NVIDIA 产品的质量与可靠性, 并导致本规格所载条件和/或要求之外的其他条件和/或要求. 对以下原因可能造成或引发的任何故障, 损害, 成本或问题, NVIDIA 不承担责任: (i) 以违反本规格的任何方式使用 NVIDIA 产品; (ii) 客户产品设计.

本规格不明示或默示授予任何 NVIDIA 专利权, 著作权或其他 NVIDIA 知识产权许可. NVIDIA 发布的第三方产品或服务信息, 不构成 NVIDIA 对使用此类产品或服务的许可, 保证或认可. 使用此类信息可能需要根据第三方的专利权或其他知识产权取得第三方许可, 或根据 NVIDIA 的专利权或其他知识产权取得 NVIDIA 许可. 只有在 NVIDIA 书面批准复制, 复制内容未经改动并附带所有相关条件, 限制和声明时, 才允许复制本规格中的信息.

所有 NVIDIA 设计规格, 参考板, 文件, 图纸, 诊断, 列表及其他文档 (合称或单独称为 "材料") 均按 "原样" 提供. 对于这些材料, NVIDIA 不作任何明示, 默示, 法定或其他形式的保证, 并明确否认对不侵权, 适销性和特定用途适用性的所有默示保证. 无论客户因任何原因可能遭受何种损害, NVIDIA 对本文所述产品承担的责任总额与累计责任均以 NVIDIA 产品销售条款与条件为限.

<span id="section-11-1"></span>

### 11.1 VESA DisplayPort

DisplayPort, DisplayPort Compliance Logo, DisplayPort Compliance Logo for Dual-mode Sources 和 DisplayPort Compliance Logo for Active Cables 是 Video Electronics Standards Association 在美国和其他国家/地区拥有的商标.

<span id="section-11-2"></span>

### 11.2 HDMI

HDMI, HDMI 标志和 High-Definition Multimedia Interface 是 HDMI Licensing LLC 的商标或注册商标.

<span id="section-11-3"></span>

### 11.3 ARM

ARM, AMBA 和 ARM Powered 是 ARM Limited 的注册商标. Cortex, MPCore 和 Mali 是 ARM Limited 的商标. 其他所有品牌或产品名称均为其各自所有者的财产. "ARM" 用于指 ARM Holdings plc; 其运营公司 ARM Limited; 以及地区子公司 ARM Inc.; ARM KK; ARM Korea Limited.; ARM Taiwan Limited; ARM France SAS; ARM Consulting (Shanghai) Co. Ltd.; ARM Germany GmbH; ARM Embedded Technologies Pvt. Ltd.; ARM Norway, AS 和 ARM Sweden AB.

<span id="section-11-4"></span>

### 11.4 OpenCL

OpenCL 是 Apple Inc. 的商标, 经许可由 Khronos Group Inc. 使用.

<span id="section-11-5"></span>

### 11.5 商标

NVIDIA, NVIDIA 标志, CUDA, FERMI, KEPLER, MAXWELL, PASCAL, TITAN, Tesla, GeForce, NVIDIA DRIVE PX, NVIDIA DriveWorks, NVIDIA DriveNet 和 NVLink 是 NVIDIA Corporation 在美国和其他国家/地区的商标和/或注册商标. 其他公司和产品名称可能是与之相关的各公司的商标.

<span id="section-11-6"></span>

### 11.6 版权

© 2017 NVIDIA Corporation. 保留所有权利.

准确的印刷版式和参考文献以原始 PDF 为准.
