---
title: 'NVIDIA Blackwell Architecture Technical Brief'
createTime: 2026/09/09 12:00:00
permalink: /papers/nvidia-blackwell-architecture/
---

> [Nick Stam](https://developer.nvidia.com/blog/author/nstam/). NVIDIA Blackwell Architecture Technical Brief, V2.1, 发布于 2025-10-01. [NVIDIA 资源页面](https://resources.nvidia.com/en-us-blackwell-architecture/blackwell-architecture-technical-brief). <a href="/paper/nvidia-blackwell-architecture.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. 本文没有 arXiv 记录或 TeX 源码; 准确措辞, 印刷版式和参考文献以正式发布的 PDF 为准.
>
> 为 AI 推理时代而生. 已更新, 加入 Blackwell Ultra GB300 Superchip, Blackwell Ultra GB300 NVL72 Rack-scale System 和 HGX B300 Server System.

<span id="section-1"></span>

## 1 NVIDIA GB300 NVL72 为 AI 推理时代而生

多年来, AI 的进步一直沿着预训练扩展这一清晰路径发展: 更大的模型, 更多的数据和更多的计算资源带来突破性能力. 构建更智能的系统已不再只是扩大预训练模型. 现在还要对模型进行精调, 使其能够思考和推理.

后训练扩展通过针对特定任务精调 AI 模型, 让模型给出更符合对话习惯的回答. 使用领域专用数据和合成数据调优模型, 可以增强模型理解细微上下文并给出准确输出的能力. 合成数据的生成没有上限, 因而后训练扩展需要大量计算资源.

如今, 一条放大智能的新扩展定律已经出现: 测试时扩展.

测试时扩展也称为长思考, 它在 AI 推理期间动态增加计算量, 以实现更深入的推理. AI 推理模型不会只用一次前向过程生成回答, 而是会实时思考, 权衡多种可能性并修正答案.

这种向后训练扩展和测试时扩展的转变, 对计算量, 实时处理和高速互连提出了指数级增长的需求. 为开发定制的衍生模型, 后训练所需的计算量可能是预训练的 30 倍; 要解决极其复杂的任务, 长思考所需的计算量可能是单次推理过程的 100 倍.

[NVIDIA Blackwell 架构](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/?ncid=so-link-252431-vt04)由此登场. 它专为数据中心规模的推理 AI 工作流构建, [能效](https://www.nvidia.com/en-us/glossary/energy-efficiency/)最高可达上一代 NVIDIA Hopper GPU 的 30 倍.

本技术简报详细介绍 NVIDIA Blackwell 的优势, 包括 Blackwell Ultra GPU, [GB300 NVL72 机架级系统](https://www.nvidia.com/en-us/data-center/gb300-nvl72/)和 HGX B300 服务器. 文中还包含 GB200 NVL72 和 HGX B200 的系统信息.

<span id="section-2"></span>

## 2 NVIDIA Blackwell 和 Blackwell Ultra 概览

NVIDIA Blackwell 和 Blackwell Ultra 产品旨在应对持续增长的 AI 复杂度需求, 包括更大的模型规模和 AI 推理, 并引入了一系列创新.

借助 NVIDIA Blackwell 和 Blackwell Ultra 产品, 每家企业都能以可负担的经济成本使用和部署先进的 LLM, 利用推理 AI 优化业务. 同时, NVIDIA Blackwell 和 Blackwell Ultra 产品开启了下一代 AI 模型, 以实时性能支持高吞吐量; 没有 Blackwell 的架构创新, 这种能力无法实现.

<span id="figure-01"></span>

![图 1. 配备 ConnectX-8 SuperNIC 的 NVIDIA Grace Blackwell Ultra Superchip.](./nvidia-blackwell-architecture/figure-01.png)

**图 1.** 配备 ConnectX-8 SuperNIC 的 NVIDIA Grace Blackwell Ultra Superchip

<span id="section-3"></span>

## 3 NVIDIA Blackwell 架构创新

Blackwell 架构通过 Blackwell 和 Blackwell Ultra 产品, 为 AI 推理和加速计算带来突破性进展. 它采用新的第二代 Transformer Engine, 并配备速度更快, 带宽更宽的 [NVIDIA® NVLink®](https://www.nvidia.com/en-us/data-center/nvlink/) 互连, 让数据中心进入性能比上一代架构高出数个数量级的新时代.

[NVIDIA 机密计算](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/)技术的进一步进展提高了大规模 AI 推理的安全等级, 同时不牺牲性能. NVIDIA Blackwell 新增的 Decompression Engine 与 [Spark RAPIDS™](https://docs.nvidia.com/spark-rapids/index.html) 库结合, 为数据分析应用提供无与伦比的数据库性能. NVIDIA Blackwell 的多项进展建立在数代加速计算技术之上, 以无与伦比的性能, 能效和规模开启推理 AI 的下一篇章.

<span id="figure-02"></span>

![图 2. NVIDIA Blackwell 架构的技术突破.](./nvidia-blackwell-architecture/figure-02.png)

**图 2.** NVIDIA Blackwell 架构的技术突破

<span id="section-3-1"></span>

### 3.1 新一代 AI GPU

Blackwell 拥有 2080 亿个晶体管, 数量超过 NVIDIA Hopper GPU 的 2.5 倍, 并采用专为 NVIDIA 定制的 [TSMC](https://www.tsmc.com/english) 4NP 工艺, 是迄今制造的最大 GPU. NVIDIA Blackwell 实现了单芯片最高计算性能 20 petaFLOPS.

该架构通过合并两个 GPU die, 可容纳大量计算能力. 两个 GPU die 都达到了光刻掩模版尺寸所允许的最大 die 面积, 也是目前能够制造的最大尺寸. 两个 die 通过一条 10 TB/s 的芯片间 NVIDIA High-Bandwidth Interface (NV-HBI) 连接并统一起来, 形成一颗完全一致的芯片.

Blackwell 架构远不只是一颗具有高每秒浮点运算次数 (FLOPS) 的芯片. 它继续以 NVIDIA 丰富的开发工具生态, CUDA-X™ 库, 超过 400 万名开发者和 3000 多个应用为基础并从中受益, 将性能扩展到数千个节点.

新的 NVIDIA Blackwell Ultra GPU 面向 AI 推理时代构建, 计算能力更强, 内存容量更大. 与 NVIDIA GB200 NVL72 相比, GB300 NVL72 的 AI 性能提高 1.5 倍; 与 [NVIDIA Hopper™](https://www.nvidia.com/en-us/data-center/technologies/hopper-architecture/) 系统相比, AI 推理生产力提高 50 倍, AI 推理速度提高 35 倍, 能效提高 30 倍, 每 token 成本降低 25 倍.

<span id="section-3-2"></span>

### 3.2 Blackwell Tensor Core 架构

Tensor Core 是专门执行矩阵乘法累加 (MMA) 数学运算的高性能计算核心, 可为 AI 和 HPC 应用提供突破性性能. 一颗 NVIDIA GPU 中跨 SM 并行运行的 Tensor Core, 与标准 Floating-Point (FP), Integer (INT) 和 FMA (Fused Multiply-Accumulate) 运算相比, 可大幅提高吞吐量和效率. Tensor Core 最早在 NVIDIA Tesla® V100 GPU 中引入, 此后每一代 NVIDIA GPU 架构都对其进行了增强.

随着生成式和推理 AI 模型的规模与复杂度不断增长, 必须改进训练和推理性能. 为满足这些计算需求, Blackwell 的新第五代 Tensor Core 架构支持 FP4 等新数值格式, 其中包括社区定义的 microscaling (OCP) 格式. Blackwell 架构支持[表 1](#table-01) 所列的全部数据类型和数值格式.

<span id="table-01"></span>

![表 1. Blackwell 架构支持的数据类型.](./nvidia-blackwell-architecture/table-01.png)

**表 1.** Blackwell 架构支持的数据类型

<span id="section-3-3"></span>

### 3.3 第二代 Transformer Engine

Blackwell 引入了新的第二代 Transformer Engine. 第二代 Transformer Engine 将定制 Blackwell Tensor Core 技术与 [NVIDIA Dynamo](https://www.nvidia.com/en-us/ai/dynamo/), [TensorRT-LLM](https://developer.nvidia.com/tensorrt) 和 [Nemo Framework](https://www.nvidia.com/en-us/ai-data-science/generative-ai/nemo-framework/) 的创新结合起来, 加速 LLM, AI 推理和 Mixture-of-Experts (MoE) 模型的推理与训练.

Blackwell Transformer Engine 使用先进的动态范围管理算法和称为 micro-tensor scaling 的细粒度缩放技术, 优化推理性能和准确度, 并支持 FP4 AI. 这让 Blackwell FP4 Tensor Core 的性能翻倍, 到 HBM 内存的参数带宽翻倍, 每颗 GPU 支持的模型规模也翻倍.

Dynamo 和 TensorRT-LLM 的创新包括 4-bit 精度量化, 采用 expert parallelism 映射的定制 kernel 和解耦架构, 它们使当前的 MoE 模型可以用更少的硬件, 更少的能量和更低的成本进行实时推理.

在训练方面, 第二代 Transformer Engine 与 Nemo Framework 和 Megatron-Core 的新 expert parallelism 技术配合; 这些技术再与其他并行技术和第五代 NVLink 结合, 实现前所未有的模型性能. 更低精度的格式为进一步加速大规模训练提供了可能.

借助 Blackwell 第二代 Transformer Engine, 企业能够以可负担的经济成本使用和部署先进的 AI 推理模型, 利用生成式 AI 优化业务. NVIDIA Blackwell 让下一代 AI 推理模型成为可能, 同时支持训练和实时推理.

<span id="section-3-4"></span>

### 3.4 注意力层加速

Blackwell Ultra GPU 新增了用于改善长输入序列性能的指令, 注意力层计算速度是 Blackwell GPU 的 2 倍. Blackwell Ultra GPU 架构把注意力运算能力翻倍, 通过降低延迟并让 AI 推理模型更快, 更智能地作出决策来提高 AI 性能. 这种加速还会缩短处理时间, 从而降低计算成本, 节省能源和基础设施开支. 企业可以更高效地扩展, 使用相同资源处理更大的工作负载, 最终获得更高的效率, 更低的成本以及 AI 驱动业务中的竞争优势.

<span id="section-3-5"></span>

### 3.5 高性能机密计算与安全 AI

生成式 AI 为企业提供了巨大潜力. 优化收入, 提供业务洞察和辅助生成内容只是其中几项收益. 但对于需要使用受隐私法规约束或包含专有信息的私有数据进行训练的企业, 采用生成式 AI 可能并不容易.

NVIDIA Confidential Computing 将 Trusted Execution Environment (TEE) 从 CPU 扩展到 GPU. NVIDIA Blackwell 上的 Confidential Computing 经过专门设计, 为 LLM 和其他敏感数据提供速度最快, 安全性最高且可验证 (基于证据) 的保护. NVIDIA Blackwell 引入了业界首款支持 TEE-I/O 的 GPU, 同时通过支持 TEE-I/O 的主机和 NVLink 上的内联保护 (同时提供机密性与完整性), 提供性能最高的机密计算解决方案.

Blackwell Confidential Computing 的吞吐性能与未加密模式几乎相同. 客户现在不仅能保护 AI 知识产权 (IP), 安全地启用机密 AI 训练, 推理和联邦学习, 还可以高效保护规模最大的模型.

<span id="section-3-6"></span>

### 3.6 第五代 NVLink 和 NVLink Switch

要充分发挥 exascale 计算和 AI 推理模型的潜力, 服务器集群中的每颗 GPU 都必须快速, 顺畅地通信. 借助为第五代 NVLink 构建的 NVLink Switch 芯片, 第五代 NVLink 可扩展到 576 颗 GPU, 加速推理 AI 模型. 第五代 NVLink 的性能是 NVIDIA Hopper 中第四代 NVLink 的两倍. Blackwell 和 Blackwell Ultra GPU 中的新 NVLink 与 Hopper GPU 一样, 也在每个方向使用两对高速差分线组成一条链路, 但 NVIDIA Blackwell 架构把每条链路每个方向的有效带宽翻倍到 50 GB/sec.

Blackwell 和 Blackwell Ultra GPU 包含 18 条第五代 NVLink 链路, 提供 1.8 TB/sec 总带宽, 每个方向 900 GB/sec. 每颗 GPU 的 1.8TB/s 双向吞吐量是 PCIe Gen5 带宽的 14 倍以上, 可为当前最复杂的大模型提供高速通信. 在包含 72 颗 GPU 的 NVLink 域中, 汇总传输带宽为 130 TB/s, 数据移动量超过整个互联网.

NVIDIA NVLink Switch 在一个包含 72 颗 GPU 的 NVLink 域 (NVL72) 中为模型并行提供 130TB/s GPU 带宽, 并通过新的 NVIDIA Scalable Hierarchical Aggregation and Reduction Protocol (SHARP)™ FP8 支持, 将带宽效率提高 4 倍. NVLink 与 NVLink Switch 配合使用, 可让集群超出单台服务器, 同时保持同样出色的 1.8 TB/s 互连. 使用 NVLink Switch 的多服务器集群能够按新增计算能力均衡扩展 GPU 通信, 使 GB300 NVL72 支持的 GPU 吞吐量达到单个八 GPU 系统的 9 倍.

<span id="section-3-7"></span>

### 3.7 Decompression Engine

数据分析和数据库工作流传统上依赖 CPU 计算, 速度慢且操作繁琐. 加速数据科学可显著提高端到端分析的性能, 更快产生价值并获得洞察, 同时降低成本. 数据库在处理, 加工和分析大量数据的过程中承担重要作用. Blackwell 架构新增专用 NVIDIA Decompression Engine, 数据解压缩速率最高可达 800GB/s. 它结合 GB200 中单颗 GPU 提供的 8TB/s HBM3e (High Bandwidth Memory) 和 Grace CPU 的高速 NVLink-C2C (Chip-to-Chip) 互连, 让 Blackwell 与 Blackwell Ultra 加速数据库查询的完整流水线, 为数据分析和数据科学提供最高性能. 它支持 LZ4, Snappy 和 Deflate 等最新压缩格式; 在查询基准测试中, [NVIDIA Blackwell 的速度是 CPU 的 18 倍](https://developer.nvidia.com/blog/nvidia-gb200-nvl72-delivers-trillion-parameter-llm-training-and-real-time-inference/), 是 NVIDIA H100 GPU 的 6 倍.

<span id="figure-03"></span>

![图 3. GB200 Grace Blackwell 使用 Decompression Engine 执行数据库 Join 查询.](./nvidia-blackwell-architecture/figure-03.png)

**图 3.** GB200 Grace Blackwell 使用 Decompression Engine 执行数据库 Join 查询

<span id="section-3-8"></span>

### 3.8 RAS Engine

Blackwell 架构新增专用的 Reliability, Availability, and Serviceability (RAS) Engine, 提供智能韧性; 它会及早识别可能出现的故障, 以尽量减少停机时间. NVIDIA 由 AI 驱动的预测管理功能会持续监控硬件和软件中的数千个数据点以了解总体健康状况, 从而预测并拦截停机和低效的来源. 由此建立的智能韧性可以节省时间, 能源和计算成本.

NVIDIA RAS engine 提供深入的诊断信息, 可以识别需要关注的区域并安排维护. RAS engine 通过快速定位问题来源来缩短周转时间, 并通过有效修复尽量减少停机时间. 管理员可以灵活调整计算资源和最佳 checkpoint 策略, 使大规模训练作业不中断. 如果 RAS engine 发现需要更换组件, 系统会启用备用容量, 让工作按时完成并把性能下降降到最低. 所需的硬件更换可以安排在适当时间, 避免计划外停机.

<span id="section-4"></span>

## 4 NVIDIA Grace Blackwell Ultra / Blackwell NVL72 机架级系统

<span id="table-02"></span>

![表 2. GB300 NVL72 和 GB200 NVL72 的系统规格.](./nvidia-blackwell-architecture/table-02.png)

**表 2.** GB300 NVL72 和 GB200 NVL72 的系统规格

<span id="figure-04"></span>

![图 4. NVIDIA GB300 NVL72.](./nvidia-blackwell-architecture/figure-04.png)

**图 4.** NVIDIA GB300 NVL72

<span id="section-5"></span>

## 5 Blackwell Ultra GB300 NLV72

NVIDIA GB300 NVL72 在一个机架级设计中连接 36 颗 Grace CPU 和 72 颗 Blackwell Ultra GPU, 大幅增强推理, 训练和数据处理能力. GB300 NVL72 是液冷机架级解决方案, 具有包含 72 颗 GPU 的 NVLink 域; 它像一颗巨型 GPU 一样运行, 并提供为 AI 推理设计的优化连接.

<span id="section-5-1"></span>

### 5.1 最大化 AI 工厂性能和收入

<span id="section-5-1-1"></span>

#### 5.1.1 将 AI 工厂产出提高 50 倍

Jensen Huang 将 Pareto Frontier 曲线定义为 AI 工厂大语言模型推理的优化框架, 在每兆瓦吞吐量 (y 轴上的 TPS / MW) 和单用户延迟体验 (x 轴上的 TPS for 1 User) 之间取得平衡. AI 工厂效率需要平衡原始吞吐量与快速响应, 最佳运行点位于曲线的转角处. 工厂产出可以看作曲线下的面积, 也可以用平衡点构成的矩形面积近似. NVIDIA Dynamo 是实时编排器, 它在 GPU, GPU 内存和 NVLink 之间动态划分 GPU 资源, 使系统可以沿 Pareto 曲线流动, 而不被锁在固定运行点上. Dynamo 通过优化 Blackwell Ultra GPU 的并行策略 (expert/tensor/pipeline) 和管理, 在维持低延迟的同时, 将 AI 工厂产出或生产力提高到 Hopper 系统的 50 倍, 并通过高效的 token 制造让每个 token 的收入最大化.

<span id="figure-05"></span>

![图 5. GB300 将 AI 推理的 AI 工厂产出提高 50 倍.](./nvidia-blackwell-architecture/figure-05.png)

**图 5.** GB300 将 AI 推理的 AI 工厂产出提高 50 倍

<span id="section-5-2"></span>

### 5.2 通过性能与能效降低 TCO

<span id="section-5-2-1"></span>

#### 5.2.1 将性能提高 35 倍

NVIDIA GB300 NVL72 的推理性能提高了 35 倍; 与使用 H100 进行 AI 推理 (DeepSeek-R1) 相比, 它改变了 AI 应用的速度和可扩展性, 使 AI 聊天机器人能够解决更复杂的策略问题.

<span id="figure-06"></span>

![图 6. GB300 将 AI 推理性能提高 35 倍.](./nvidia-blackwell-architecture/figure-06.png)

**图 6.** GB300 将 AI 推理性能提高 35 倍

<span id="section-5-2-2"></span>

#### 5.2.2 将能效提高 30 倍

运行 AI 工作负载的数据中心经常受到能源和冷却能力的限制, 因此效率十分重要, 目标是让每单位能耗发挥最大性能. Blackwell Ultra 的能效是 Hopper 一代的 30 倍.

<span id="section-5-2-3"></span>

#### 5.2.3 将 TCO 降低 25 倍

希望以一小部分成本获得显著更高 AI 性能的企业仍然关注总体拥有成本 (TCO). 借助 GB300 NVL72, 客户不仅可以降低硬件开支, 还可以减少 AI 部署整个生命周期内的冷却和维护成本. 企业因此能够更有效地分配资源, 在不过度增加资本支出的情况下加快 AI 驱动的创新.

<span id="figure-07"></span>

![图 7. GB300 降低能源使用量和总体拥有成本.](./nvidia-blackwell-architecture/figure-07.png)

**图 7.** GB300 降低能源使用量和总体拥有成本

<span id="section-5-3"></span>

### 5.3 机架级端到端 AI 加速

Blackwell Ultra 每颗 GPU 配备最高 279 GB HBM3e 内存, 每个机架配备 37 TB 高速内存, 同时拥有超过 1 exaFLOP 的 FP4 计算能力和统一的 72-GPU NVLink 域, 因而可以支持大得多的模型, 并以更少的节点扩展, 为 AI 突破打开大门. 再结合用于加速计算的 CUDA-X 库, NVIDIA 加速了整个软硬件计算栈.

<span id="section-5-4"></span>

### 5.4 针对每个数据中心优化

投资建设经过优化的数据中心不只是性能优势, 对于希望在 AI 驱动的未来保持竞争力的组织而言也是战略需要. GB300 NVL72 的 AI FLOPS 是 HGX H100 的 65 倍, 可以为 AI 模型提供显著更多的推理能力. 通过 NVLink 将 CPU, GPU 和高速互连结合起来, 多个系统之间的数据传输达到了前所未有的效率.

<span id="section-5-5"></span>

### 5.5 用于可扩展, 安全和高吞吐 AI 性能的加速网络平台

GB300 NVL72 像一个性能极强的单一计算单元一样运行, 需要强大的网络来实现最佳应用性能. GB300 NVL72 与 NVIDIA Quantum-X800 InfiniBand, Spectrum-X Ethernet, Connect-X SuperNIC 和 BlueField-3 DPU 配合, 在超大规模 AI 数据中心提供前所未有的可扩展性能, 效率和安全性.

系统中每颗 GPU 都可获得 800 Gb/s 总数据吞吐量. GB300 NVL72 与 NVIDIA 网络平台无缝集成, 使 AI 工厂和云数据中心能够处理万亿参数模型, 而不会遇到瓶颈. GB300 NVL72 架构率先在 GPU 与 ConnectX-8 SuperNIC 之间引入 PCIe Gen6 连接, 从而不再需要独立的 PCIe switch 接口.

<span id="section-5-5-1"></span>

#### 5.5.1 ConnectX-8 SuperNIC 实现卓越的推理性能

新的 NVIDIA ConnectX®-8 SuperNIC™ 为 GB300 NVL72 系统中的每颗 GPU 提供完整的 800 gigabits per second (Gb/s) 网络连接. 它与 NVIDIA Quantum-X800 InfiniBand 或 Spectrum-X™ Ethernet 网络平台配合, 可提供一流的 remote direct-memory access (RDMA) 能力, 实现最高的 AI 工作负载效率. 此外, ConnectX-8 SuperNIC 支持线速 Internet Protocol Security (IPsec) 和 PSP Protocol Security (PSP) 网络加密, 通过 GPU 间连接加密增强平台安全性.

<span id="section-5-6"></span>

### 5.6 将实时视频生成和多模态 AI 能力提高 30 倍

先进 LLM 的上下文窗口仅为 128,000 个 token, 而生成一段 5 秒视频需要处理 400 万个 token, 在当前先进的 NVIDIA Hopper GPU 上需要近 90 秒. Blackwell Ultra 平台可使用 NVIDIA Cosmos 等 world foundation model 实时生成视频, 性能比 Hopper 一代提高 30 倍, 让客户能够为 Physical AI 应用创建定制, 照片级逼真, 时空稳定的视频和 3D 世界模拟器.

<span id="figure-08"></span>

![图 8. GB300 NVL72 将 Physical AI 应用的视频生成性能提高 30 倍.](./nvidia-blackwell-architecture/figure-08.png)

**图 8.** GB300 NVL72 将 Physical AI 应用的视频生成性能提高 30 倍

<span id="section-6"></span>

## 6 Blackwell GB200 NVL72

<span id="section-6-1"></span>

### 6.1 最大化 AI 工厂性能和收入

<span id="section-6-1-1"></span>

#### 6.1.1 将 AI 工厂产出提高 40 倍

Frontier Pareto 曲线说明了 AI 工厂推理生产力如何平衡吞吐量和单用户响应时间. NVIDIA Dynamo 在 GB200 NVL72 上实时编排 GPU 资源, 执行并行化并管理 Blackwell GPU, 相比 Hopper 系统可将生产力提高最多 40 倍, 在固定 1 megawatt 功率条件下最大化 token 制造效率.

<https://docs.nvidia.com/multi-node-nvlink-systems/partition-guide.pdf>

<https://www.theregister.com/2025/03/23/nvidia_dynamo/>

<span id="figure-09"></span>

![图 9. GB200 将 AI 推理的 AI 工厂产出提高 40 倍.](./nvidia-blackwell-architecture/figure-09.png)

**图 9.** GB200 将 AI 推理的 AI 工厂产出提高 40 倍

<span id="section-6-2"></span>

### 6.2 面向下一代大语言模型的实时推理

GB200 NVL72 引入先进功能和第二代 Transformer Engine, 可显著加速 LLM 推理工作负载, 为多万亿参数语言模型等资源密集型应用提供实时性能. 对于 GPT-MoE-1.8 等巨型模型, GB200 NVL72 使用相同数量的 GPU, 相比 H100 可加速 30 倍, TCO 降低 25 倍, 能耗也降低 25 倍. 这项进步来自新一代 Tensor Core, 它引入了包括 FP4 在内的新精度. 此外, GB200 使用 NVLink 和液冷构建一个包含 72 颗 GPU 的巨型单一机架, 可以克服通信瓶颈.

GB200 是用于高性能推理任务的革命性解决方案, 体现了 NVIDIA 推动 AI 边界的承诺.

<span id="figure-10"></span>

![图 10. GB200 1.8T GPT-MoE 使用第二代 Transformer Engine 的实时推理性能与新一代 AI 训练性能.](./nvidia-blackwell-architecture/figure-10.png)

**图 10.** GB200 1.8T GPT-MoE 使用第二代 Transformer Engine 的实时推理性能与新一代 AI 训练性能

GB200 配备速度更快, 支持 FP8 精度的 Transformer Engine; 与 NVIDIA Hopper GPU 一代相比, 它可将 GPT-MoE-1.8T 等大语言模型的训练性能提高 4 倍. 这种性能提升让机架空间减少 9 倍, TCO 和能耗降低 3.5 倍. 第五代 NVLink (提供 1.8 TB/s GPU 间互连和更大的 72-GPU NVLink 域), InfiniBand 网络和 NVIDIA Magnum IO™ 软件进一步补充了这项突破. 这些技术共同为企业提供高效的可扩展性, 并便于实施大规模 GPU 计算集群.

<span id="figure-11"></span>

![图 11. GB200 1.8T GPT-MoE 模型使用 Transformer Engine 的训练加速.](./nvidia-blackwell-architecture/figure-11.png)

**图 11.** GB200 1.8T GPT-MoE 模型使用 Transformer Engine 的训练加速

<span id="section-6-3"></span>

### 6.3 加速数据处理和基于物理的模拟

GB200 通过紧密耦合 CPU 和 GPU, 为数据处理, 工程设计及模拟领域的加速计算带来新机会.

数据库在处理, 加工和分析企业的大量数据时承担重要作用. GB200 利用高带宽 NVLink-C2C 和 Blackwell 中的专用 Decompression Engine, 让关键数据库查询速度达到 CPU 的 18 倍, 同时减少 7 倍能耗并将 TCO 降低 5 倍.

基于物理的模拟仍是产品设计和开发的支柱. 从硅芯片到药品, 通过模拟而不是物理测试来检验和改进产品, 每年可节省数十亿美元.

专用集成电路几乎完全使用 CPU 设计, 其工作流漫长而复杂, 通常还包括用于确定各处电压和电流的模拟分析. Cadence SpectreX simulator 就是其中一种 solver, 它在 GB200 上的运行速度是 x86 CPU 的 13 倍.

GPU 加速计算流体动力学 (CFD) 是工程师和设备设计人员研究或预测设计行为的重要工具. Cadence Fidelity 是一种 large eddy simulator (LES), 在 GB200 上的模拟速度最高可达 x86 CPU 的 22 倍.

<span id="section-6-4"></span>

### 6.4 可持续计算

计算密度和计算功率正在推动数据中心从风冷转向液冷. 使用液体而不是空气冷却, 会在数据中心内外产生许多积极影响, 包括提高每机架性能, 减少冷却用水量, 并允许数据中心在更高环境温度下运行, 进一步降低能耗.

<span id="figure-12"></span>

![图 12. 能耗和 TCO 降低 25 倍.](./nvidia-blackwell-architecture/figure-12.png)

**图 12.** 能耗和 TCO 降低 25 倍

<span id="section-7"></span>

## 7 AI-Ready Enterprise Platform

[NVIDIA AI Enterprise](https://www.nvidia.com/en-us/data-center/products/ai-enterprise/) 是端到端软件平台, 让每家企业都能使用生成式 AI, 并为生成式 AI foundation model 提供速度最快, 效率最高的 runtime. 它包括 NVIDIA NIM™ 推理 microservice, AI framework, 库和工具; 这些组件经过认证, 可以在常见的数据中心平台和配备 NVIDIA GPU 的主流 NVIDIA-Certified Systems™ 上运行. 使用 AI 经营业务的企业依赖 NVIDIA AI Enterprise 提供的安全性, 支持, 可管理性和稳定性, 确保从试点平稳转向生产.

NVIDIA AI Enterprise 与 NVIDIA Blackwell Ultra 加速计算结合, 不仅简化了 AI-ready 平台的构建, 还缩短了产生价值的时间.

可通过 [build.nvidia.com](http://build.nvidia.com/) 了解 NVIDIA AI Enterprise 的 AI 工作负载工作流.

<span id="section-8"></span>

## 8 NVIDIA Blackwell HGX

NVIDIA Blackwell HGX B300 和 HGX B200 系统为生成式 AI, 数据分析和高性能计算带来突破性进展,

**NVIDIA HGX™ B300:** NVIDIA HGX™ B300 面向 AI 推理时代构建, 计算能力更强, 内存容量更大. 它的 AI 计算能力是 Hopper 平台的 7 倍, 配备超过 2 TB HBM3E 内存, 并与 NVIDIA ConnectX-8 SuperNIC 进行高性能网络集成; 从训练, agentic system 和推理到实时视频生成, HGX B300 为每个数据中心中最复杂的工作负载提供突破性性能.

**HGX B200:** 这是基于八 Blackwell GPU baseboard 的 Blackwell x86 平台, 可提供 144 petaFLOPS AI 性能. 对于 x86 scale-up 平台和基础设施, HGX B200 提供最佳性能 (是 HGX H100 的 15 倍) 和 TCO (是 HGX H100 的 12 倍). 每颗 GPU 可配置到最高 1000 Watts.

<span id="table-03"></span>

![表 3. HGX B300 和 HGX B200 的系统规格.](./nvidia-blackwell-architecture/table-03.png)

**表 3.** HGX B300 和 HGX B200 的系统规格

<span id="section-9"></span>

## 9 NVIDIA Blackwell 架构在 AI 推理时代的作用

AI 已演变到需要三条不同的扩展定律, 它们说明以不同方式投入计算资源会如何影响模型性能.

<span id="figure-13"></span>

![图 13. 三条 AI 扩展定律.](./nvidia-blackwell-architecture/figure-13.png)

**图 13.** 三条 AI 扩展定律

**预训练扩展:** 这是 AI 发展的原始定律. 它表明, 通过增加训练数据集规模, 模型参数量和计算资源, 开发者可以预期模型智能与准确度得到可预测的提升. 预训练扩展造就了具备突破性能力的模型. 它推动了模型架构中的重大创新, 包括数十亿和数万亿参数 Transformer 模型的兴起; 五年内, 计算需求增长了 5000 万倍.

**后训练扩展:** 预训练扩展让模型学习互联网中的知识, 后训练则教模型如何思考, 并进一步提高模型针对组织预期用例的具体性和相关性. 如果说预训练像把 AI 模型送进学校学习基础技能, 那么后训练会为模型补充适用于预期工作的技能. 为支持后训练, 开发者可以使用合成数据扩充或补充微调数据集. 后训练所需的总计算量是预训练的 30 倍.

**测试时扩展 (也称为长思考或推理):** LLM 可以快速响应输入 prompt, 并正确回答简单问题, 但在复杂查询上可能不够有效. agentic AI 工作负载需要 LLM 在给出答案前对问题进行推理; 这一过程发生在推理阶段. 据估计, 使用测试时扩展的模型所需计算量是传统推理的 100 倍, 因此能在得出最佳答案前推理多个可能的回答.

[NVIDIA Blackwell](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/?ncid=so-link-252431-vt04) 是一代仅有一次的平台, 具备有效服务三条扩展定律所需的计算能力和能效, 是 AI 推理时代的基础.

<span id="section-10"></span>

## 10 万亿参数模型 AI 推理中的高级并行技术

GPT 1.8T MoE (Mixture of Experts) 等万亿参数模型的部署给 AI 推理带来独特挑战, 尤其需要有效管理计算资源, 同时确保最佳用户体验. 本附录介绍可用于应对这些挑战的多种并行技术, 重点讨论数据, 张量, 流水线和专家并行.

<span id="section-10-1"></span>

### 10.1 AI 推理中的并行技术

1. **Data Parallelism (DP)** 数据并行会在不同 GPU 或集群上托管完整模型的多个副本, 同时处理彼此独立的用户请求. 该方法随 GPU 数量线性扩展, 可以提高吞吐量而不影响用户交互性. 但它需要大量内存, 因为每颗 GPU 都保存一份完整模型副本.
2. **Tensor Parallelism (TP)** 张量并行把模型的每一层拆分到多颗 GPU, 让一个用户请求的不同部分并行处理. 该方法通过为每个请求分配更多资源来缩短处理时间, 因而可以改善用户交互性. 但它严重依赖 GPU 间的高带宽通信, 在大规模系统上可能形成瓶颈.
3. **Pipeline Parallelism (PP)** 在流水线并行中, 不同的模型层组分布到多颗 GPU, 一个用户请求的各部分依次经过流水线处理. 该技术通过分散权重来管理大模型, 但可能造成处理效率低下, 也不会显著改善用户交互性.
4. **Expert Parallelism (EP)** 专家并行把请求路由到模型中的特定专家, 并让不同 GPU 处理这些专家, 减少与不需要参数的交互. 专家完成处理后, 结果需要通过高带宽 GPU 互连进行 all-to-all 通信. 该方法需要复杂的数据路由和重组管理, 效果也受可用专家数量限制.

<span id="section-10-2"></span>

### 10.2 组合并行技术

组合不同并行方法可以缓解单项技术的局限. 同时使用专家并行和流水线并行, 能让用户交互性翻倍, 且吞吐量损失很小. 类似地, 集成张量, 专家和流水线并行可以让 GPU 吞吐量提高三倍, 而不牺牲用户交互性. 为合适的部署场景组合不同并行方式, 需要穷举探索整个解空间, 并使用大量计算资源.

<span id="section-10-3"></span>

### 10.3 最大化吞吐量并管理运行阶段

高效管理 prefill 和 decode 阶段, 即上下文处理和生成阶段, 是最大化吞吐量的必要条件. inflight batching 和 chunking 等技术可以动态管理请求处理, 防止这些阶段出现瓶颈, 从而优化 GPU 利用率.

**Inflight Batching and Chunking** Inflight batching 和 chunking 是优化 GPU 利用率并改善 LLM 部署用户体验的重要技术. 这些方法通过管理数据在 GPU 资源间的处理方式, 应对 AI 推理的 prefill 和 decode 运行阶段.

- **Chunk Size Considerations:** chunk 大小在平衡 GPU 吞吐量和用户交互性时很重要. 更大的 chunk 会减少 prefill 阶段所需的迭代次数, 从而缩短 time to first token (TTFT). 但它也会延长 decode 阶段, 降低 tokens per second (TPS). 相反, 更小的 chunk 可以加快 token 输出并提高 TPS, 但会增加 TTFT. 这种权衡决定了特定部署场景的最佳 chunk 大小.

**Impact of Chunk Size on GPT 1.8T MoE Model** 以 GPT 1.8T MoE 模型为例, 研究分析了 chunk 大小从 128 到 8,192 个 token 变化时的影响, 涵盖 2,700 多种并行和 chunk 长度配置组合. 这项大规模分析有助于理解不同设置如何影响吞吐量和交互性之间的平衡.

<span id="section-10-4"></span>

### 10.4 结论

部署万亿参数模型需要成熟的并行策略, 才能有效平衡吞吐量和用户交互性. 企业理解并实施数据, 张量, 流水线和专家并行的组合后, 可以优化 AI 推理部署, 同时满足计算需求和用户期望.

如需进一步了解如何优化大规模模型的 AI 推理, 并深入了解不同类型的并行方式, 请阅读技术解析 [Demystifying AI Inference Deployments for Trillion Parameter Large Language Models](https://developer.nvidia.com/blog/demystifying-ai-inference-deployments-for-trillion-parameter-large-language-models/).

<span id="section-11"></span>

## 11 声明

截至所示日期, 本规格中提供的信息被认为准确可靠. 但是, NVIDIA Corporation ("NVIDIA") 不对这些信息的准确性或完整性作出任何明示或暗示的陈述或保证. 对于使用这些信息造成的后果, 或者因使用这些信息可能导致的专利或其他第三方权利侵害, NVIDIA 不承担任何责任. 本出版物取代此前提供的该产品所有其他规格.

NVIDIA 保留随时更正, 修改, 增强, 改进及以其他方式更改本规格和/或停止提供任何产品或服务而不另行通知的权利. 客户在下单前应获取最新的相关规格, 并确认其中信息是最新且完整的.

NVIDIA 产品按订单确认时提供的 NVIDIA 标准销售条款和条件销售, 除非 NVIDIA 与客户的授权代表在单独签署的销售协议中另有约定. 对于将任何客户通用条款和条件应用于购买本规格所述 NVIDIA 产品, NVIDIA 在此明确表示反对.

NVIDIA 产品并非为医疗, 军事, 飞机, 航天或生命支持设备而设计, 授权或担保, 也不适用于 NVIDIA 产品发生故障或异常时可合理预期会造成人身伤害, 死亡或财产及环境损害的应用. NVIDIA 不对在这些设备或应用中包含和/或使用 NVIDIA 产品承担责任, 因而客户应自行承担包含和/或使用这些产品的风险.

NVIDIA 不陈述或保证基于这些规格的产品无需进一步测试或修改即可适合任何指定用途. NVIDIA 不一定会测试每款产品的全部参数. 客户独自负责确保产品适合并符合其计划的应用, 并执行应用所需的测试, 以避免应用或产品发生故障. 客户产品设计中的缺陷可能影响 NVIDIA 产品的质量和可靠性, 并可能产生本规格之外的额外或不同条件和/或要求. 对于以下原因造成或与之相关的任何故障, 损害, 成本或问题, NVIDIA 不承担任何责任: (i) 以违反本规格的任何方式使用 NVIDIA 产品; 或 (ii) 客户的产品设计.

本规格不明示或暗示授予任何 NVIDIA 专利权, 著作权或其他 NVIDIA 知识产权下的许可. NVIDIA 发布的第三方产品或服务信息不构成 NVIDIA 授予使用这些产品或服务的许可, 也不构成对其作出的保证或认可. 使用这些信息可能需要按第三方的专利或其他知识产权获得第三方许可, 或按 NVIDIA 的专利或其他知识产权获得 NVIDIA 许可. 只有在 NVIDIA 书面批准, 信息未经修改地复制, 且附带所有相关条件, 限制和声明时, 才允许复制本规格中的信息.

所有 NVIDIA 设计规格, 参考板, 文件, 图纸, 诊断信息, 列表和其他文档 (合称及分别称为 "材料") 均按 "原样" 提供. 对于这些材料, NVIDIA 不作任何明示, 暗示, 法定或其他保证, 并明确否认所有关于不侵权, 适销性和特定用途适用性的暗示保证. 无论客户因何种原因遭受任何损害, NVIDIA 对客户承担的产品总计和累计责任均以 NVIDIA 的产品销售条款和条件为限.

<span id="section-11-1"></span>

### 11.1 商标

NVIDIA, NVIDIA logo, NVIDIA CUDA, NVIDIA Omniverse, NVIDIA RTX, NVIDIA Tesla, NVIDIA Turing, NVIDIA Volta, NVIDIA Jetson AGX Xavier, NVIDIA DGX, NVIDIA HGX, NVIDIA EGX, NVIDIA CUDA-X, NVIDIA GPU Cloud, GeForce, Quadro, CUDA, GeForce RTX, NVIDIA NVLink, NVIDIA NVSwitch, NVIDIA DGX POD, NVIDIA DGX SuperPOD 和 NVIDIA TensorRT 是 NVIDIA Corporation 在美国和其他国家/地区的商标和/或注册商标. 其他公司名和产品名可能是其各自关联公司的商标.

Copyright © 2025 NVIDIA Corporation. 保留所有权利.
