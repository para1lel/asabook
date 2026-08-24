---
title: 'FRED'
createTime: 2026/08/24 20:00:00
permalink: /papers/fred/
pageClass: paper-reading
---

> [Saeed Rashidi](https://orcid.org/0000-0002-6472-9920), [William Won](https://orcid.org/0000-0002-1715-9144), [Sudarshan Srinivasan](https://orcid.org/0009-0002-8662-5820), [Puneet Gupta](https://orcid.org/0000-0002-6188-1134), and [Tushar Krishna](https://orcid.org/0001-5738-6942). 论文于 2024 年 6 月 28 日首次提交至 arXiv, 当前版本为 v2 (2025 年 6 月 9 日). 发表于 [第 52 届 IEEE/ACM 计算机体系结构国际研讨会 (ISCA 2025)](https://doi.org/10.1145/3695053.3731055), 2025 年 6 月 21-25 日. [FRED: A Wafer-scale Fabric for 3D Parallel DNN Training](https://arxiv.org/abs/2406.19580v2). [原始 PDF](/paper/fred.pdf). [arXiv DOI](https://doi.org/10.48550/arXiv.2406.19580). [TeX 源码](https://export.arxiv.org/e-print/2406.19580v2). 原始 PDF 对确切的印刷版式和参考文献保持权威性.

## 摘要

晶圆级系统将高端加速器芯粒与高速晶圆级互连紧密集成, 从而提供低延迟和高带宽连接. 这使它适合用于深度神经网络 (DNN) 训练. 但是, 现有的晶圆上网络拓扑 (例如 2D Mesh) 缺少有效支持不同并行策略所需的灵活性. 本文提出 Fred, 一种针对 DNN 训练通信需求设计的晶圆级互连结构. Fred 使用微型交换机在晶圆上构建分布式拓扑, 为任意加速器组之间的集合通信提供无阻塞连接, 并支持交换机内集合通信. 在示例并行策略上, 与基线晶圆级 Mesh 相比, Fred 将 ResNet-152, Transformer-17B, GPT-3 和 Transformer-1T 的端到端训练时间平均分别缩短至 $1.76\times$, $1.87\times$, $1.34\times$ 和 $1.4\times$.

<span id="section-1"></span>

## 1 引言

 DNN 模型的规模正呈指数增长. 一项近期研究表明, 不到两年时间里, DNN 训练所需的计算量和内存分别增加了 1,$800\times$ 和 1,$500\times$ [Thi21]. 目前常见的做法是将训练任务分布到多个加速器或神经处理单元 (NPU) 上, 以缩短训练时间. 但分布式训练有一个关键副作用: 根据并行策略的不同, NPU 之间需要通信以同步模型梯度和/或激活值. 随着 NPU 数量增加, 通信开销也会增加, 直到成为分布式训练延迟的主要因素 [Ast20, Sca20b, Jia19a, An20].

 即使是高速机架级互连 (如 NVLink [Eva19]), 可提供的带宽也存在根本上限, 因此人们越来越关注在同一封装中集成多个 NPU 的平台. Cerebras [Cer21] 以单片晶圆的形式展示了这一思路的极端实现, 晶圆上的 NPU 彼此互连. 更具成本和良率优势的方案包括基于硅/有机中介层的方法 [Sim19, Cen20], 或采用 Silicon Interconnect Fabric (Si-IF), 将芯粒直接键合到整厚度硅 *晶圆* 上而无需封装 [Arc19, Des21, Waf24]. *本文假设使用一种仅包含互连的无源晶圆级基底, 芯粒以类似 Si-IF 或 TSMC-SoW [Tsm24] 的细间距键合在其上. 与 Cerebras 的单片方案不同, 这种方式可以异构集成来自不同工艺的计算, 内存和网络芯粒.*

 人们普遍认可晶圆级基底在可扩展性和带宽方面的优势, 但 *连接 NPU 的互连结构架构* 仍是一个开放问题. 迄今为止的晶圆级加速器方案 (如 Cerebras CS2 [Cer21], NVIDIA SIMBA [Sim19], UCLA waferscale GPU [Arc19], Chiplet Cloud [Chi24a], Chen 等人的 TTO [Enh24]) 都为互连采用了 2D Mesh 拓扑. 选择 Mesh 是可以理解的: 它易于完成布局布线并具备良好的可扩展性, 是多核芯片中最普遍的拓扑, 在晶圆级基底上也很自然. *但我们证明, 2D Mesh 拓扑固有的阻塞特性对于 DNN 训练通信场景极其低效.*

<span id="figure-01"></span>

![图 1. 用于优化 DNN 训练通信的软硬件协同设计栈. 本文处理图中红色标出的三个阶段. 左侧展示了 3D 并行中训练工作器的一个逻辑视图. 并行策略大小为 MP(4)-DP(3)-PP(2), 表示 MP/DP/PP 三个维度分别有 4/3/2 个对等工作器. 每个工作器用 3 位数字命名, 分别表示其在 MP, DP 和 PP 维度中的 ID. 沿同一维度对齐的工作器需要针对该维度的并行类型进行 *通信*. 例如, 工作器 000, 100, 200 和 300 需要执行 MP 通信 (即在前向传播/反向传播期间同步激活值和输入梯度), 而工作器 300, 310 和 320 需要执行 DP 通信 (在反向传播期间同步权重梯度).](./fred/figure-01.png)

**图 1.** 用于优化 DNN 训练通信的软硬件协同设计栈. 本文处理图中红色标出的三个阶段. 左侧展示了 3D 并行中训练工作器的一个逻辑视图. 并行策略大小为 MP(4)-DP(3)-PP(2), 表示 MP/DP/PP 三个维度分别有 4/3/2 个对等工作器. 每个工作器用 3 位数字命名, 分别表示其在 MP, DP 和 PP 维度中的 ID. 沿同一维度对齐的工作器需要针对该维度的并行类型进行 *通信*. 例如, 工作器 000, 100, 200 和 300 需要执行 MP 通信 (即在前向传播/反向传播期间同步激活值和输入梯度), 而工作器 300, 310 和 320 需要执行 DP 通信 (在反向传播期间同步权重梯度).

 DNN 训练中的通信取决于所采用的并行方式. 数据并行 (DP) [Li20c, An20], 模型并行 (MP) [Lep20, Jia19a] 和流水线并行 (PP) [Hua19, Pip19] 是各种并行策略的基本组成. 在 DP 中, DNN 模型被复制到多个 NPU, 每个 NPU 处理一组不同的训练样本 (即一个 minibatch). 在 MP 中, 每个 DNN 层被分片到多个 NPU, 各 NPU 处理相同的训练样本. 在 PP 中, 每个 NPU 承载一部分 DNN 层, 训练样本以流水线方式流过这些 NPU. 3D 并行 [Eff21] 在 NPU 之间建立不同的 MP/DP/PP 组, 综合使用上述策略. DP, MP 和 PP 之间的最优平衡高度依赖工作负载和底层平台, 在不同工作负载/平台配置之间可能 *存在显著差异* [Jia19a, Ali24]. [图 1](#figure-01) 展示了一个 3D 并行策略示例.

<span id="figure-02"></span>

![图 2. 不同并行策略下的归一化计算和通信开销.](./fred/figure-02.png)

**图 2.** Transformer-17B 在连接晶圆上 20 个 NPU 的 2D Mesh 拓扑上运行时, 不同并行策略 (见[第 2 节](#section-2)) 的归一化计算和通信开销 (拓扑见[第 6 节](#section-6)).

 从通信角度看, 3D 并行需要在分布式训练的不同阶段, 在同一 MP/DP/PP 组内的 NPU 之间执行 *多个并发通信操作*. 不同并行策略对计算和通信的压力也不同. [图 2](#figure-02) 对此进行了量化. 如图所示, 较高的通信开销可能使计算效率更高的策略拥有比计算效率较低策略更高的总训练开销 (例如 MP(20)-DP(1)-PP(1) 对比 MP(5)-DP(4)-PP(1)). 除了由工作负载决定的通信量外, [图 2](#figure-02) 中较高网络开销的主要原因是基线拓扑对网络资源的 *低效利用*. 具体来说: (i) 大多数通信操作只会激活一半或更少的 NPU 链路 (详见[第 3.2.4 节](#section-3-2-4)); (ii) MP/DP/PP 并行组之间存在网络竞争 (详见[第 3.2.2 节](#section-3-2-2)). 基线拓扑面临的全部挑战见[第 3.2 节](#section-3-2).

总的来说, 用于分布式 DNN 训练的理想晶圆级互连应满足以下三点:

- 以最小拥塞处理 *多个* 无阻塞集合通信.
- 对 *所有* 3D 并行配置都保持高效.
- 在 NPU 之间提供 *高带宽* 连接.

本文提出 **Fred**, 一种带有 Flexible REduction-Distribution 特性的晶圆级互连, 用于支持任意 3D 并行. Fred 包括: (i) 一种新型拓扑, 其中的交换机原生支持归约和广播, 用于放大带宽; (ii) 支持无阻塞的集合通信路由算法; (iii) 用于降低拥塞的设备放置算法. 我们在晶圆级基底上部署 Fred [Des21]. 架构中的每个 NPU 都由高端计算芯粒和 3D 堆叠 DRAM 芯粒混合集成而成 (类似 H100 [Nvi23]). 我们还讨论了如何在晶圆基底上完成 Fred 拓扑的物理布局和扩展.

据我们所知, *Fred 是首个针对 DNN 训练设计的晶圆级互连方案, 能够高效支持混合并行策略 (例如 3D 并行) 下的多个并发集合通信*. 因此, 编译器可以使用任意并行策略, 无需担心其在网络上的执行效率. 具体而言:

- 我们阐述了为 3D 并行设计晶圆级互连时面临的挑战 ([第 3 节](#section-3)).
- 我们提出 Fred, 一种新型网络互连, 包含若干创新特性: 通过可扩展拓扑连接的灵活归约-分发树组成的 *交换机互连* ([第 4 节](#section-4)), 以及能够并发路由多个集合通信的 *新型路由算法* 和面向 3D 并行的拥塞感知设备放置策略 ([第 5 节](#section-5)).
- 我们展示了 Fred 的晶圆级互连实现方式 ([第 6 节](#section-6)).
- 我们针对部分示例工作负载和并行策略, 将 Fred 与基线互连进行比较 ([第 8 节](#section-8)).

结果表明, 与基线 2D Mesh 相比, Fred 可使 ResNet-152, Transformer-17B, GPT-3 和 Transformer-1T 的端到端训练时间平均分别提升 $1.76\times$, $1.87\times$, $1.34\times$ 和 $1.4\times$.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 集合通信模式

尽管 DNN 模型差异很大, 但分布式训练中的大部分通信都可以通过集合通信模式处理 [An20]. 根据模型类型和并行策略, 在前向传播/反向传播期间同步激活值或梯度时可能需要不同类型的集合通信 [Ast20]. [图 3](#figure-03) 展示了三个工作器之间常见集合通信模式的数学含义. 在 *Reduce-Scatter* 中, 工作器以一种最终让每个工作器获得全局归约数据一部分的方式通信. 在 *All-Gather* 中, 每个工作器将本地数据广播给所有其他工作器. *All-Reduce* 是分布式训练中最常见的模式 [An20], 可以看作先执行 *Reduce-Scatter*, 再执行 *All-Gather*. 在 *Reduce* 中, 多个 NPU 参与数据归约, 结果只存储在一个 NPU 上; *Gather* 从所有 NPU 收集数据并将其存储在单个 NPU 上. *Multicast* 表示一个 NPU 将数据发送给多个 NPU. 在 *All-to-All* 中, 每个工作器都向每个工作器发送本地数据的一部分.

<span id="figure-03"></span>

![图 3. 三个工作器之间的集合通信模式.](./fred/figure-03.png)

**图 3.** 三个工作器之间的集合通信模式.

<span id="section-2-2"></span>

### 2.2 集合通信算法

[第 2.1 节](#section-2-1) 所述模式可以通过不同的 *集合通信算法* 处理. 总体上, 这类算法有两种不同的实现方式:

**1) 基于端点.** NPU 之间以点对点的分布式方式通信, 通过显式 send/recv 消息完成, 不需要中央协调. 在这种情况下, 最优算法通常取决于物理网络拓扑和集合通信规模. 例如, 当物理拓扑是环时, 基于环的 All-Reduce 最优; 对于树形拓扑或较小消息, 基于树的 All-Reduce 更优 [Tha05].

基于 NPU 到 NPU 的方法的一个缺点是会产生大量流量. 例如, 带宽最优的 NPU 到 NPU 算法在 N 个 NPU 之间执行大小为 $D$ 字节的 All-Reduce 时, 要求每个 NPU 发送/接收近似 $\frac{2(N-1)}{N}D$ 字节数据 [An20], 几乎是 All-Reduce 大小 ($D$ 字节) 的 $2$ 倍 [Tha05, Col06, Col07]. 这是因为所有基于端点的算法都必须分别执行归约和收集阶段, 每个阶段都需要每个 NPU 发送/接收 $\frac{(N-1)}{N}D$ 字节 [Tha05, Col06, Col07].

**2) 网络内集合通信执行.** 为减少基于端点方法产生的额外流量, 近期方案通过向交换机加入计算能力, 引入了网络内集合通信算法 [Sca21, An20, Acc19], 使归约和收集可以同时完成. 例如, 对大小为 $D$ 字节的 All-Reduce, 每个 NPU 只需向交换机或交换机层级发送/接收 $D$ 字节. 交换机或交换机层级从每个 NPU 接收 $D$ 字节, 对来自全部 $N$ 个 NPU 的数据执行归约, 再将 $D$ 字节广播回所有 NPU. 因此, 与基于端点的方法相比, 每个 NPU 发送/接收的流量几乎减半 ($D$ 字节, 而不是 $\frac{2(N-1)}{N}D$ 字节) [Acc19]. 此外, 网络内集合通信执行可以将端点资源用于训练计算任务, 由网络交换机高效处理集合通信.

<span id="section-2-3"></span>

### 2.3 3D 并行中的通信

将分布式训练任务分配到多个 NPU 有多种方式 (统称为并行策略): MP (又称张量并行) [Sho19], DP [Li20c] 和 PP [Hua19, Pip19]. 这些策略的组合可以概括为 3D 并行 [Eff21]. [图 1](#figure-01) 展示了 3D 并行的概念. 在该方案中, 每个训练工作器都属于一个 MP 组, 一个 DP 组和一个 PP 组; 每个 NPU 在其 MP/DP/PP 组内的 ID (偏移量), 分别由三位工作器 ID 的第一/第二/第三位确定. 因此, DP 和 PP 位相同的 NPU 属于同一个 MP 组 (例如 000, 100, 200 和 300).

同一 DP 组内的 NPU 应在反向传播期间通过 *All-Reduce* 集合通信模式同步本地计算的模型梯度, 并在开始下一次训练迭代前更新模型 [An20]. 对于 MP 组, NPU 需要在前向传播/反向传播期间通信, 以同步输出激活值和输入梯度. 具体通信模式取决于层类型及其分片方式. 常见模式包括: *All-Reduce* [Sho19], *All-to-All* [Nau19], *Reduce-Scatter* [Lep20] 和 *All-Gather* [Lep20]. 对于 PP 组, NPU 需要在边界层的前向传播/反向传播期间传输输出激活值和输入梯度, 并将数据传递给承载下一组层的 NPU. [表 1](#table-01) 汇总了各并行策略产生的集合通信模式.

[图 1](#figure-01) 还说明了同时处理多个集合通信的必要性. 例如, 这里有 8 个不同的 DP 组, 因此 DP 通信最多需要同时处理 8 个 All-Reduce (MP/PP 通信则分别有 6/12 个并发通信操作). 此外, MP, DP 和 PP 组的通信类型及对等工作器各不相同. 因此, *底层网络互连必须能够灵活处理并发且不同的集合通信模式*.

<span id="table-01"></span>

![表 1. 不同并行策略产生的集合通信模式.](./fred/table-01.png)

**表 1.** 不同并行策略产生的集合通信模式.

<span id="section-2-4"></span>

### 2.4 多芯粒集成

在基于芯粒的集成中, NPU 芯片先完成制造, 然后键合到封装互连 (如 Si-IF) 上 [Sim19, Des21, Tsm24]. 这种方法可以在封装中集成来自不同工艺的组件 (甚至包括 DRAM). 此外, 芯粒可以在集成前进行测试, 因而与 Cerebras 等完全单片式方案相比, 具有更高良率, 支持异构集成, 且所需冗余更少 [Cer21].

**多芯粒互连拓扑.** 近期多芯粒平台的产品和研究都通过 2D Mesh 拓扑互连 NPU [Arc19, Cer21, Des21, Sim19, Enh24, Chi24a]. 在封装内/晶圆上的 2D Mesh 易于布局布线, 且在二维基底上具有面积优势, 这是选择它的主要原因 [Arc19]. *因此, 本文选择 2D Mesh 作为主要基线拓扑, 并以此比较所提出的方案*.

<span id="section-3"></span>

## 3 晶圆级互连结构的目标指标

<span id="section-3-1"></span>

### 3.1 通信需求

首先, 我们讨论在晶圆级基底上运行 DNN 训练的两种执行模式.

<span id="section-3-1-1"></span>

#### 3.1.1 权重驻留

当 DNN 模型可以完全装入晶圆上的片上内存时, 将完整模型参数加载到封装并将训练结果收集回封装的操作只需执行一次, 属于一次性开销. [+1] 预训练模型的加载和训练模型的存储成本可以分摊到数千次训练迭代中. 但输入样本需要在每次训练迭代开始时加载. 由于样本远小于模型, 这些 I/O 操作对整体训练性能影响很小. 因此, *在此模式下, 主要性能因素是计算核心效率和 NPU 到 NPU 的通信性能*. 未优化的互连可能导致某些并行策略的 NPU 到 NPU 通信性能很差, 编译器不得不放弃这些策略, 即使它们能更好地利用计算资源和片上内存, 仅仅因为通信性能不佳 ([第 3.2 节](#section-3-2)).

<span id="section-3-1-2"></span>

#### 3.1.2 权重流式传输

当片上内存不足以容纳模型时, 执行模式会转为 *权重流式传输* [Cer21, Thi21]. 在这种情况下, 任意时刻只有部分 DNN 层被加载到封装中. 处理完这些层后, 片上存储空间会被回收, 用于加载下一组层. 因此, 在模型训练期间必须多次将完整模型加载到芯片上 (至少在前向传播和反向传播期间各加载一次). 此外, NPU 计算出模型梯度后, 会将数据写入片外存储, 再由存储侧的轻量计算核心更新模型, 供下一次迭代使用 [+2] [Cer21]. 这种方式使性能受 I/O 限制, 即训练性能上限按 $\propto\frac{\mathrm{model}\_\mathrm{size}}{I/O\_\mathrm{BW}}$ 缩放. 因此, *除了计算效率和 NPU 到 NPU 的通信性能外, 保持最大 I/O 带宽也至关重要*. 在向 I/O 通道分发/收集模型或梯度时, 刚性的拓扑可能形成热点, *限制 I/O 数据速率* ([第 3.2 节](#section-3-2)), 直接影响训练性能.

<span id="section-3-2"></span>

### 3.2 2D Mesh 的挑战

下面讨论 2D Mesh 在支持 DNN 训练通信需求时面临的具体挑战.

<span id="section-3-2-1"></span>

#### 3.2.1 高效 I/O

如前所述, 在权重流式传输执行模式下, 保持高 I/O 带宽是获得最佳性能的关键. 但 2D Mesh 往往无法提供最大 I/O 性能. [图 4](#figure-04) 用纯 DP 并行策略下的 $4\times 4$ Mesh 拓扑说明了这一问题. 在这种场景中, 从片外内存通道读取的每个权重都必须广播给所有 NPU. [图 4](#figure-04)(A) 展示了从两个不同内存通道读取数据时的广播算法 (红色和蓝色流), 该算法基于 2D Mesh 上一对多模式的 *MPI* 实现 [For09].

理想情况下, 所有内存通道应同时以线速传输 (不同的) 权重, 以最大化 I/O 带宽. 但 *2D Mesh 拓扑的形状会产生固有热点, 从而降低 I/O 带宽*. [图 4](#figure-04)(B) 展示了所有内存通道同时读取权重时某条热点链路的最大通道负载. 如果每个内存通道的带宽为 $P$ 字节/秒, 那么在 $4\times 4$ Mesh 上实现最大 I/O 带宽时, 热点链路需要具备 $7P$ 字节/秒的容量 (带宽).

<span id="figure-04"></span>

![图 4. (A) 从两个不同 I/O 通道读取数据时的广播通信模式 (红色和蓝色箭头). 每个箭头旁的数字表示一个数据包穿过该链路的时间戳. 实际上, 每条路径上会以流水线方式传输多个数据包. 在该示例中, 并行策略为 MP(1)-DP(16)-PP(1), 模型权重在权重流式传输执行模式下广播给所有 NPU. 注意, 反向传播和将最终结果写回远程存储时, 使用相反的顺序对权重梯度求和. (B) 对应[图 4](#figure-04)(A)的最大通道负载分析, 此时所有 I/O 通道同时工作.](./fred/figure-04.png)

**图 4.** (A) 从两个不同 I/O 通道读取数据时的广播通信模式 (红色和蓝色箭头). 每个箭头旁的数字表示一个数据包穿过该链路的时间戳. 实际上, 每条路径上会以流水线方式传输多个数据包. 在该示例中, 并行策略为 MP(1)-DP(16)-PP(1), 模型权重在权重流式传输执行模式下广播给所有 NPU. 注意, 反向传播和将最终结果写回远程存储时, 使用相反的顺序对权重梯度求和. (B) 对应[图 4](#figure-04)(A)的最大通道负载分析, 此时所有 I/O 通道同时工作.

In general, for an $N\times N$ mesh and $4\times N$ external I/O channels, the wafer-scale fabric links should have a bandwidth of $\mathbf{(2N-1)P}$ bytes/s to fully utilize the I/O bandwidth in all parallelization strategies, assuming each I/O channel has a bandwidth of $P$ bytes/s. As the formula indicates, the required link bandwidth grows $O(N)$ with the mesh width. For larger packages, the technology might not support such high-bandwidth requirements on the package. In such cases, the I/O channel rate must be scaled down proportionally to accommodate the maximum link bandwidth, i.e., $P=\frac{\mathrm{\mathrm{link}}\_\mathrm{\mathrm{BW}}}{(2N-1)}$.

**Fred 的方案.** Fred prevents network hotspots by adaptively routing the traffic through all of its links equally, enabling further scalability of the wafer-scale systems.

<span id="section-3-2-2"></span>

#### 3.2.2 设备放置

<span id="figure-05"></span>

![图 5. MP(2)-DP(4)-PP(2) 策略下的两种设备放置映射. (A) 偏向 MP 和 DP 通信但使 PP 通信拥塞的放置. (B) 偏向 DP 和 PP 通信但使 MP 通信拥塞的放置.](./fred/figure-05.png)

**图 5.** MP(2)-DP(4)-PP(2) 策略下的两种设备放置映射. (A) 偏向 MP 和 DP 通信但使 PP 通信拥塞的放置. (B) 偏向 DP 和 PP 通信但使 MP 通信拥塞的放置.

Device placement involves assigning each logical training worker to a physical NPU. With $N$ NPUs, there are $N!$ possible device placement mappings. This becomes critical in 3D parallelism, as each training worker may have different communication volumes and patterns with other workers across distinct parallelization groups (refer to [图 1](#figure-01)). Therefore, finding a device placement that minimizes network contention is essential.

However, this is challenging with rigid topologies, especially 2D Mesh, where certain communication patterns are inherently prioritized over others. [图 5](#figure-05) illustrates two different mappings for a given MP(2)-DP(4)-PP(2) strategy. In [图 5](#figure-05)(A), the MP and DP communications are free of congestion, but PP communications cause congestion between different PP groups. Conversely, in [图 5](#figure-05)(B), DP and PP communications are optimized, but MP communications face congestion between MP groups. Ultimately, as 2D mesh offers two logically disjoint dimensions ($x$ and $y$), *it is mathematically impossible for all 3D parallelism dimensions to be optimally mapped onto a 2D Mesh*. This is trivial by observing the four corner NPUs, where each NPU offers two outgoing links. Consequently, due to the limited path diversity, one out of the three parallelization groups must experience network congestion and reduced communication performance. Determining which communication patterns to prioritize, unavoidable on 2D Mesh, requires a thorough analysis of the end-to-end workload and understanding the impact of different communication operations.

**Fred 的方案.** Fred supports congestion-free routing for all communication patterns simultaneously.

<span id="section-3-2-3"></span>

#### 3.2.3 非对齐并行策略

<span id="figure-06"></span>

![图 6. 非对齐 MP(5)-DP(3)-PP(1) 并行策略在 $4\times 4$ Mesh 拓扑上的网络通信. (A) 同一 MP 组内 NPU 执行通信模式 (如 All-Reduce) 时的非优化情况. (B) 假设采用 X-Y 路由时两个 DP 组之间的流量拥塞, 以红色和蓝色表示.](./fred/figure-06.png)

**图 6.** 非对齐 MP(5)-DP(3)-PP(1) 并行策略在 $4\times 4$ Mesh 拓扑上的网络通信. (A) 同一 MP 组内 NPU 执行通信模式 (如 All-Reduce) 时的非优化情况. (B) 假设采用 X-Y 路由时两个 DP 组之间的流量拥塞, 以红色和蓝色表示.

When searching for the best parallelization strategy itself, there are many possible configurations where the size of MP/DP/PP is not aligned with the physical topology dimensions. Such configurations create extra challenges on a 2D Mesh, due to the limited path diversity with distinct NPU-to-NPU distances.

[图 6](#figure-06)illustrates the communication issues within a $4\times 4$ 2D-mesh topology for an MP(5)-DP(3)-PP(1) strategy. [图 6](#figure-06)(A) demonstrates how NPUs in the same MP group need to communicate. Collective communications are often optimized for well-structured topologies (e.g., rings, trees, switches). However, as shown in [图 6](#figure-06)(A), the MP groups form non-standard shapes, making it challenging to identify the most optimized collective algorithm for each shape. For example, the distance between NPU 420 and 020 is two hops, due to the rigid shape of 2D Mesh, *making it impossible to construct a well-constructed ring*, even without considering network congestion. [图 6](#figure-06)(B) depicts the extra traffic congestion between two different DP groups, marked in red and blue, caused by non-aligned dimensions.

**Fred 的方案.** Fred provides congestion-free topology and routing mechanisms for any size/placement of MP/DP/PP.

<span id="section-3-2-4"></span>

#### 3.2.4 网络带宽利用率

Maintaining high bandwidth utilization is challenging for a 2D Mesh. For instance, MP communications are required during both forward-pass and back-propagation phases, while DP communications occur only during back-propagation. However, these links cannot be utilized by MP communications due to the limited paths and lack of optimal routing. Consequently, the links used for DP communication during back-propagation remain underutilized during the forward-pass phase, detrimenting full bandwidth utilization for many strategies on a 2D Mesh.

**Fred 的方案.** Fred can utilize the full bandwidth of each NPU for every communication phase.

<span id="section-3-2-5"></span>

#### 3.2.5 网络内集合通信执行

Supporting in-network collectives can significantly reduce network traffic and improve execution performance as described in [第 2.2 节](#section-2-2). This feature, currently employed in off-chip switches [An20, Mel20], requires centralized or hierarchical switches which can perform the collection, reduction, and broadcast of multiple data. A 2D Mesh with distributed NPUs and without a shared central entity, however, impedes the adaptation of the in-network collective support.

**Fred 的方案.** Fred employs a switch-based topology that supports in-network collective execution.

<span id="section-3-2-6"></span>

#### 3.2.6 小结

Ideally, a fabric for DNN training should enable each NPU to fully utilize its network bandwidth for any communication phase of 3D-parallel training without congestion and with support for in-network collectives. These requirements cannot be met via a 2D Mesh, due to their natural shape and rigidity. This underscores the need for the adaptation of new topology and routing mechanisms, such as Fred.

<span id="section-4"></span>

## 4 Fred 网络互连结构

<span id="figure-07"></span>

![图 7. (a) 带有 P 个端口的 Fred 交换机概览. (b) 端口数为偶数 ($2r$) 或奇数 ($2r+1$) 时递归构造的 Fred 互连. (c) Fred<sub>*m*</sub>($2$) 交换机. (d) Fred<sub>*m*</sub>($3$) 交换机. (e) R-$\mu$Switch. (f) D-$\mu$Switch. (g) RD-$\mu$Switch. (h) Fred<sub>*2*</sub>($8$) 互连实现示例及两种已路由的 All-Reduce 通信模式. (i) Fred<sub>2</sub>($8$) 上三条 All-Reduce 通信流的路由算法和冲突图. (j) 路由冲突示例.](./fred/figure-07.png)

**图 7.** (a) An overview of the Fred switch with P ports. (b) Fred interconnect (recursively constructed) when the number of ports is even ($2r$) or odd ($2r+1$). (c) Fred<sub>*m*</sub>($2$) switch. (d) Fred<sub>*m*</sub>($3$) switch. (e) R-$\mu$Switch. (f) D-$\mu$Switch. (g) RD-$\mu$Switch. (h) An example of a Fred<sub>*2*</sub>($8$) interconnect implementation and two routed All-Reduce communication patterns (green and orange). (i) Routing Algorithm for three All-Reduce comm flows on Fred<sub>2</sub>($8$) with conflict graph. (j) Example of Routing conflict.

A Fred switch forms the backbone of the fabric. Hierarchical connections of the Fred switches form the full Fred fabric, which is described in [第 6.1 节](#section-6-1). The key idea behind a Fred switch is simple: **break the switch into the most fundamental components, and add small compute capability to each component.** The fine-grained distribution of compute enables supporting flexible and concurrent in-switch collective execution for 3D parallelism communication patterns. In addition, distributed computation of collectives is more scalable to map over the high-BW wafer-scale links than having centralized compute and memory entities.

[图 7](#figure-07)(a) shows a Fred switch, which consists of a control unit, input port buffers, and the Fred interconnect. The control unit performs routing between the input ports and the output ports.

The Fred interconnect, shown in [图 7](#figure-07)(b), is inspired by *Clos* networks [A53]. Clos networks are identified through the tuple $(m,n,r)$, where $m\geq 2$ is the number of middle stage switches, $n$ is the number of input/output ports per each input/output micro-switch ($\mu$Switch), and $r$ is the number of input/output $\mu$Switches. Fred’s connectivity is similar to the $(m,n=2,r)$ Clos network, which is denoted as Fred<sub>*m*</sub>($P$). $m$ denotes to the number of middle-stage switches, and $P$ identifies the number of input(output) ports. Fred can be designed for an arbitrary number of ports by building on top of the previous works [Arb97]. $P$ is $\frac{2r}{2r+1}$ when $P$ is an $\frac{\mathrm{\mathrm{even}}}{\text{odd}}$ number. Similar to the Clos network, Fred interconnect is constructed recursively, where the middle stage switches are the $\frac{m\times\text{{\mathrm{\mathrm{Fred}}}${}_{m}$($r$)}}{\text{$m\times${\mathrm{\mathrm{Fred}}}${}_{m}$($r+1$)}}$ switches for the $\frac{\mathrm{\mathrm{even}}}{\text{odd}}$ number of ports, as shown in [图 7](#figure-07)(b). The recursive design of Fred ends when encountering the base Fred<sub>*m*</sub>($2$) or Fred<sub>*m*</sub>($3$) Switches, which are depicted in [图 7](#figure-07)(c) and [图 7](#figure-07)(d), respectively.

*The main difference of Fred, compared to a baseline Clos, is adding the reduction and/or distribution (broadcast) support* to the baseline $\mu$Switches. This creates three types of $\mu$Switches depending on which of these two features is present in the $\mu$Switch. [图 7](#figure-07)(e) shows the *R-$\mu$Switch* structure that has the reduction feature, i.e., reducing data on the two input ports and routing to one of the output ports. [图 7](#figure-07)(f) shows the *D-$\mu$Switch*, which is able to perform distribution by broadcasting one of the input data to both output ports. *RD-$\mu$Switch* is a $2\times 2$ $\mu$Switch and can perform both reduction and distribution, as shown in [图 7](#figure-07)(g). The entire Fred switch is built using these three $\mu$Switch types (plus *Muxes* and *Demuxes* to connect the last port to all intermediate stage switches when $P$ is odd) through the recursive process explained earlier.

[图 7](#figure-07)(h) shows the complete structure of a Fred<sub>*2*</sub>($8$) switch with two concurrent All-Reduce operations (green and orange). The highlighted $R/D/\mathrm{\mathrm{RD}}$ means that the reduction/distribution/reduction-distribution features of the corresponding $\mu$Switch are activated. For instance, the input $\mu$Switch connecting the input ports $4,5$ performs the reduction and routes the result to one of its output ports. Other non-highlighted $\mu$Switches operate like Clos $\mu$Switches.

<span id="section-5"></span>

## 5 无冲突集合通信路由

<span id="section-5-1"></span>

### 5.1 Fred 上的通信模式

The fine-grained reduction and broadcast features enable Fred $\mu$Switches to perform all different types of collective communication patterns observed in distributed training. Collective implementation on Fred, however, can be abstracted through the notation of *communication flow* (or *flow* in short).

A *flow* on Fred<sub>*m*</sub>($P$) includes a set of input ports ($\mathrm{\mathrm{IPs}}$)={ip<sub>1</sub>, ip<sub>2</sub>, …., ip<sub>*i*</sub>} and output ports ($\mathrm{\mathrm{OPs}}$)={op<sub>1</sub>, op<sub>2</sub>, …., op<sub>*j*</sub>}, where $|\mathrm{\mathrm{IPs}}|\leq P$ and $|\mathrm{\mathrm{OPs}}|\leq P$. The *flow* results in reducing the data across the input ports determined in $\mathrm{\mathrm{IPs}}$ and broadcasting the final result to the output ports identified in $\mathrm{\mathrm{OPs}}$. The port numbers and cardinality of $\mathrm{\mathrm{IPs}}$ and $\mathrm{\mathrm{OPs}}$ can be set independently, depending on the communication pattern. Each communication algorithm can be expressed in terms of performing one or more *flows*. For example, the orange All-Reduce pattern in [图 7](#figure-07)(h) is a single *flow* with $\mathrm{\mathrm{IPs}}=\{3,4,5\}$ and $\mathrm{\mathrm{OPs}}=\{3,4,5\}$.

**Simple Communication Algorithms.** Simple communication algorithms refer to communication patterns that can be realized on Fred by performing only one *flow*. [表 2](#table-02) summarizes different simple communication patterns on Fred and the number of involved input/output ports.

**Compound Communication Algorithms.** Compound communication algorithms realize the communication patterns through multiple *flows* on Fred. [表 2](#table-02) summarizes different compound communication patterns on Fred. For example, *Reduce-Scatter* among $i$ inputs is broken into $i$ serial steps of the *reduce* *flow*, and during step $1\leq j\leq i$, the *reduce* operation corresponding to the result of the $op_{j}$ is done. The process is similar for other compound communication algorithms.

<span id="section-5-2"></span>

### 5.2 路由协议

Fred considers a *flow* as a unit of routing, and supports concurrent routing of multiple *flows*. Similar to the previous methods [Nov22], Fred routing protocol is also recursive, meaning that first the status of outermost $\mu$Switch levels (i.e., input/output $\mu$Switches) are determined, and then routing is recursively called on the middle stage switches. The difference is, however, supporting reduction/distribution features on the Fred $\mu$switches, and the dependency between the input/output ports of a *flow*, which requires a new routing algorithm to realize these differences. Fred’s routing protocol is built upon the following intuitions:

- If two flows share the same input or output $\mu$Switch, they should be routed through different middle-stage switches (subnetworks).
- If both input ports of an R-$\mu$Switch or RD-$\mu$Switch belong to the same *flow*, the reduction feature is activated.
- If both output ports of a D-$\mu$Switch or RD-$\mu$Switch belong to the same *flow*, the distribution (broadcast) feature of the $\mu$Switch is activated.

The latter two points are easy to realize. To satisfy the first point, Fred routing protocol creates a *conflict graph*. [图 7](#figure-07)(i) shows the first step of a routing example for a Fred<sub>2</sub>($8$) interconnect with the associated conflict graph for this step.

In the conflict graph, each node represents a *flow* and the edges between the nodes represent a conflict (i.e., sharing an input or output $\mu$Switch) between the two nodes (*flows*). Fred routing applies the graph coloring on the conflict graph to find the routing of each *flow*. The number of colors is the number of intermediate stage switches (i.e., $m$). [图 7](#figure-07)(i) also shows the results of the graph coloring. Here, there are only two colors since $m=2$. Two flows are routed to the up subnetwork (blue), and one to the down subnetwork (red). After this step, the routing protocol and the conflict graph generation are recursively called on the middle blue and red Fred<sub>2</sub>($4$) switches. Note that a desired property of DL training is the deterministic and repetitive nature of its communication patterns that can be inferred at compile time. Therefore, the routing algorithm for different comm phases of the training workload can be executed at compile time and then saved at the control unit of the Fred switches and used during the training to minimize the routing overhead.

<span id="section-5-3"></span>

### 5.3 路由冲突及解决方法

There are certain cases where not all *flows* can be routed at the same time, causing *routing conflict*. The routing conflict is identified when the graph coloring fails to color all of the nodes within the conflict graph. [图 7](#figure-07)(j) shows an example of a routing conflict when there are four *flows* to be routed on a Fred<sub>2</sub>($8$) and the resulting conflict graph. The conflict graph cannot be colored using only two colors due to the circular dependencies between *flows: 0, 1, 2*. Note that the routing conflict may happen during any recursive call to the routing algorithm (for routing the subnetworks). If the routing conflict is identified, the entire routing is marked to have a conflict.

<span id="table-02"></span>

![表 2. 简单 (阴影部分) 和复合集合通信算法.](./fred/table-02.png)

**表 2.** 简单 (阴影部分) 和复合集合通信算法.

We now discuss ways to address such conflicts.

**(1) Blocking the Conflicting *Flows*.** The first trivial way is to block some of the conflicting *flows* and run them after the other *flows* are finished. This translates to removing some of the nodes in the conflict graph. For example, in [图 7](#figure-07)(j), if any of the *flows* $1,2,$ or $3$ is blocked, the routing can proceed to the next step (i.e., subnetworks). This option is, however, costly in terms of performance since it blocks some of the flows.

**(2) Increasing the Number of Middle Stages.** Another method is to design Fred switches with more intermediate stage switches (i.e., increase $m$). This method increases the number of colors for the graph coloring algorithm. Therefore, more conflicting *flows* can be routed simultaneously. [+3] However, this comes at the expense of more HW overhead.

**(3) Decomposing the Communication Algorithms.** For the unicast-only traffic, Fred interconnect is *rearrangeably nonblocking* when $m=2$ and *strict-sense nonblocking* when $m\geq 3$. This fact can be leveraged to decompose some of the communication algorithms into multiple steps and break the dependency among input/output ports in each step (i.e., making them unicast traffic). In the worst case, any collective algorithm can be decomposed into complete unicast traffic. For example, All-Reduce can be handled through a ring-based algorithm at the endpoints (NPUs), rather than in-network execution, which is complete unicast traffic. As a result, *flows* $0,1,$ and $2$ in [图 7](#figure-07)(j) can switch to ring-based All-Reduce at the endpoint, while *flow* $3$ uses an in-network All-Reduce algorithm. This method solves the routing by degrading the communication performance of the conflicting *flows* (but it does not block any *flow*).

**(4) Intelligent Device Placement.** Another method to prevent conflicts is through intelligent device placement (mapping) of the training workers to the physical NPUs at the start time. For example, if in [图 7](#figure-07)(j) the workers mapped to NPUs of ports $1$ and $4$ swap their locations, the conflict does not happen.

*In Fred, we prioritize the communication performance and do not use options (1) and (3). We use option (2) to simplify the device placement algorithm by only using Fred<sub>*3*</sub>($P$) switches, ensuring that we have three colors in our routing algorithm protocol. Then, for the device placement algorithm, we map the training workers within the same MP group on consecutive physical NPUs, followed by iterating over workers within PP and DP, respectively. This is sufficient to prevent routing conflicts for 3D-Parallelism communication patterns.*

<span id="section-5-4"></span>

### 5.4 处理重叠通信

In training, the workload at a given time may require multiple communication operations. For example, while handling the DP communication in backpropagation, the workload may initiate the PP communication to exchange the next microbatch between the workers. However, FRED’s circuit switch configuration may handle one communication phase at a given time. Additionally, different NPUs might issue communication at different times, due to variations in the compute latencies. Hence, there should be a mechanism to safely preempt the current executing communication operation and execute the new communication, with minimal effects to the in-flight packets, if the latter has a higher priority.

We address this issue by allocating multiple Virtual Circuits (VCs) per port, each dedicated to a specific communication group (e.g., MP), and the FRED’s interconnect to be reconfigured between different overlapping communication operations. While it is possible to frequently reconfigure FRED’s interconnect in short intervals to handle overlapping communication operations concurrently, we choose to reconfigure FRED to execute the highest priority communication operation among the currently pending operations (and preempt the current communication if a new higher priority communication is issued). This decision simplifies the design and minimizes the FRED’s reconfiguration overhead, and is in line with the training workload requirements, since the workload is usually blocking on one communication operation (highest priority) at any given point in time. In our 3D-parallel case, the priority of communication operations in descending order is: MP, PP, and DP. More discussion on FRED’s buffer management and flow control is described in [第 6.2.3 节](#section-6-2-3).

<span id="section-6"></span>

## 6 晶圆级架构

We present an instance of a wafer-scale NPU system connected using Fred, for evaluation purposes. We note that alternate configurations are also feasible.

<span id="section-6-1"></span>

### 6.1 Fred 互连结构布局

A Fred switch builds a foundation to connect multiple wafer-scale NPUs. However, for large wafer-scale systems, due to physical limitations such as wiring, area, etc., it is not feasible to connect all of the NPUs through a single Fred switch. Hence, the *Fred fabric* provides a hierarchical design for the scalable connection of large wafer-scale systems. [图 8](#figure-08) shows an example of the Fred fabric that shows a 2-level tree connection of the Fred switches and the NPUs connected to the leaf (*L1*) switches [+4]. In general, tree height and the BW across different levels are determined by the system size and physical constraints (see [第 6.2 节](#section-6-2)).

When there are multiple levels of Fred switches, the communication algorithms might need to cross several switches and hence, need to be optimized accordingly. For example, [图 8](#figure-08)(a) shows the flow path for an All-Reduce between NPUs $1,5,$ and $6$. In this case, the data of NPUs $1\text{ \mathrm{\mathrm{and}} }5$ are reduced on their local L1 switch (to reduce the traffic going to the L2 switch), and the result along with the data of NPU $6$ are reduced on the L2 switch. The final result is sent back to the corresponding L1 switches. The L1 switch attached to NPUs $1\text{ \mathrm{\mathrm{and}} }5$ also multicasts the result to the NPUs.

<span id="figure-08"></span>

![图 8. 两级 Fred 拓扑的物理视图和逻辑视图.](./fred/figure-08.png)

**图 8.** Physical and Logical Views of 2-level Fred Topologies.

<span id="section-6-2"></span>

### 6.2 晶圆级架构配置

We assume a standard 300 $mm$ wafer diameter, similar to the prior works [Arc19, Sca21a], resulting in a 70000 $mm^{2}$ wafer area.

<span id="section-6-2-1"></span>

#### 6.2.1 约束

Fundamentally, there are two physical limitations that limit the amount of compute and other resources on the wafer: (i) Thermal constraints, and (ii) Power delivery network [Des21, Arc19, Sca21a]. Thermal constraints limit the amount of power that can be delivered to the wafer, depending on the cooling mechanism. Previous works report the maximum power limit within the $9.6\>\mathrm{\mathrm{KW}}$ [Arc19] to $15\>kW$ [Cer21] range. In this paper, we assume $\boldsymbol{15\>kW}$ power is available for the wafer-scale system. The other limitation is the power delivery network, which might necessitate using big on-wafer *voltage regulator modules (VRMs)*, limiting the available area for NPUs [Arc19]. However, alternative solutions can eliminate the need for on-wafer VRMs by either supplying the voltage from the top of the wafer [Cer21], or delivering the power from the back of the wafer by using the *through-wafer-vias (TWVs)* [Pro19]. In this paper, we assume the **on-wafer VRMs are not used** by using any of the solutions described earlier.

<span id="section-6-2-2"></span>

#### 6.2.2 物理系统参数

[表 3](#table-03) shows the other set of physical parameters. We assume that the NPU chiplets are tested before bonding. If Known Good Die testing is difficult, larger chiplets such as NPU Compute may need to be broken into smaller constituents. Recent work [Chi23c] has suggested that these chiplets actually need to be moderately large (40$mm^{2}$-400$mm^{2}$) in size for cost-optimality. For the purposes of our evaluation, we assume an H100 GPU-like NPU compute chiplet, each equipped with five stacks of HBM3 chiplet memories, resulting in combined power consumption of $700\>W$ and an area of $1314\>mm^{2}$ [Nvi23].

The NPU compute chiplet perimeter can support up to 12 TBps wafer-scale BW, where $6$ TBps of it is allocated to support the 3 TBs local HBM memory BW ($3$ TBps for read + $3$ TBps for write), and the other $6$ TBps is allocated to support $3$ TBps bi-directional total NPU-to-NPU BW ($3$ TBps for send + $3$ TBps for receive).

The $15\>\mathrm{\mathrm{KW}}$ power budget limits the total amount of NPUs on the wafer to $15\>\mathrm{\mathrm{KW}}/700\>W\approx 21$, excluding other component power overheads (e.g., I/O controller, wafer-scale wires). This anticipated power density of 22W/cm<sup>2</sup> is well within the projection of cooling capability in heterogeneous integration roadmaps [Iee23]. In this paper, we consider a $20$-NPU wafer-scale system to make room for other component power overheads. Additionally, $18\times$I/O Controllers are used to connect the wafer to the external memory. Hence, the total NPU $+$ I/O Controller area overhead is $26640\>mm^{2}$.

Similar to [Arc19], we assume in the baseline, the NPU chips are placed with a 100 $um$ distance from each other. Combined with the I/O controllers, the entire baseline can be fit within a rectangle with the size of 190.8 $mm$ $\times$ 150.4 $mm$ in the center of the wafer, leaving the rest of the wafer area unclaimed.

<span id="table-03"></span>

![表 3. 物理系统参数.](./fred/table-03.png)

**表 3.** 物理系统参数.

<span id="table-04"></span>

![表 4. [图 8](#figure-08)(b) 所示 Fred 实现的硬件开销.](./fred/table-04.png)

**表 4.** [图 8](#figure-08)(b) 所示 Fred 实现的硬件开销.

<span id="section-6-2-3"></span>

#### 6.2.3 Fred 拓扑和参数

To motivate Fred, we leverage the fact that the combination of a constrained power budget and high-end NPUs results in utilizing $26640\>mm^{2}$ out of $70000\>mm^{2}$ area, **making room to utilize otherwise unclaimed area for flexible fabrics like Fred**. However, any fabric proposal must have low power consumption since most of the power budget is allocated to the NPUs. **We demonstrate that Fred meets these properties.**

Our target Fred topology is similar to [图 8](#figure-08)(a), where $20$ NPUs and I/O controllers are connected through a 2-level (almost) fat-tree topology. Similar to the baseline, the BW/NPU is still $3$ TBps, but the bisection BW is increased to $30$ TBps. It is almost fat-tree since the L1-to-L2 BW is the summation of attached NPU BW only (and not NPU $+$ I/O Controller). The reason is that if one participant of any *flow* (e.g., *Reduce*) is an I/O controller, then the entire *flow’s* BW requirement is determined by the I/O controller’s BW (e.g., 128 GBps), which is significantly less than NPU-to-NPU BW. Hence, an almost fat-tree gives the same performance as the full fat-tree.

Looking at the BW requirements of Fred L1/L2 switches in [图 8](#figure-08)(a), it is clear that each switch chiplet requires a perimeter, to connect the wafer-scale network wires, that is not feasible to build. Hence, in reality, each of the Fred switches in [图 8](#figure-08)(a) is decomposed into multiple lower-BW Fred chiplets. [图 8](#figure-08)(b) shows a logical view of implementing the (almost) fat-tree based topology of [图 8](#figure-08)(a) using feasible Fred chiplets. As [图 8](#figure-08)(b) shows, each switch of [图 8](#figure-08)(a) is implemented by decomposing it into multiple smaller, but feasible, Fred switches (enclosed in the strip line). For our evaluations, we use Fred<sub>3</sub>($P$) switches.

As [图 8](#figure-08)(b) shows, in Fred fabric, L1 switches have hybrid BW downstream links to connect to the NPUs and I/O controllers. This requires Fred L1 switches to use different interface circuitry for NPU vs. I/O controller links, which is accounted for in the overhead numbers in [表 4](#table-04). In general, hybrid on-chip interconnects are widely used in many designs (e.g., to connect on-chip routers vs. memory controllers in multi-core processors) [Pri04].

**Flow Control.** We assume a Virtual Cut-Through flow control with a credit-based backpressure mechanism to guarantee the switch buffer as packets flow through FRED’s fabric. To enable preemptive communication execution, we consider four VCs per port: three data VCs dedicated to MP, DP, and PP packets and one control VC for the ACK/NACK and other control messages. The data/control packet size is 4KB/512B, with each flit size set to be 512B. The packet header size is 6B to allow for large sequence numbers. Each packet header also has the index to the $\mu$Switch configuration bits, stored in the control unit for a specific communication phase [+5]. If all ports receive a packet belonging to a higher priority phase, Fred changes its $\mu$Switch configuration to that phase and starts forwarding the packets from that phase. Additionally, there is a default header index, which refers to a phase where all flows are unicast and Fred falls back to the online routing to determine the $\mu$Switch configs. While not present in our workloads, this mode is useful when dealing with communication patterns such as *alltoallv* where different src/dst pairs have different size flows that are changing dynamically.

The retransmission protocol is set to be simple Go-Back-N, with an accumulative ack per every 16 data packets to reduce the ack overhead to less than $1\%$ of the network BW. If a switch receives a NACK from an NPU, it forwards it to all input ports participating in that flow, which is then propagated to all NPUs serving as the source of the flow, and retransmission starts from the NACKed packet.

Additionally, each input port has a 24KB buffer per data VC and a 2KB buffer for the control VC. These policies ensure that in the case of communication preemption, there are enough buffers available (i.e., $\mathrm{\mathrm{link}}\_\mathrm{\mathrm{BW}}\times \mathrm{\mathrm{RTT}}=\text{24\mathrm{\mathrm{KB}}}$) for the new communication operation to send at the full link BW.

**HW Overhead.** [表 4](#table-04) shows the overheads of our proposed Fred implementation shown in [图 8](#figure-08). We assume 1.5KB SRAM per FRED switch to store the $\mu$Switch configurations for different communication operations. The numbers are obtained post layout using 15nm NanGate PDK. The total power overhead is $179.35\>W$, which is about $1.2\%$ of the total power budget. The total area overhead is $25195\>mm^{2}$, which can be accommodated by using the unclaimed area available on the wafer. Note that, as discussed in [第 6.2.3 节](#section-6-2-3), the main area overhead of the Fred chiplets is due to I/O for supporting high-BW wafer-scale interconnects, and not because of the switch logic overhead.

**讨论: Fred Area Overhead.** As we discussed earlier, the unclaimed area on the wafer allows for designing large (but low power) Fred switches to deliver high I/O BW requirements for our topology. In fact, Fred’s internal logic occupies less than 5% of the chip area. Hence, the area overhead of Fred can be significantly reduced if the I/O density increases.

<span id="table-05"></span>

![表 5. 目标配置.](./fred/table-05.png)

**表 5.** 目标配置.

In our design, we conservatively assume the switch chips use the same interconnect technology as the NPUs (e.g., pitch, frequency, etc.). However, switch area can be further reduced by applying more aggressive network bandwidth technologies. Next generation of I/O technology is expected to deliver up to 250 GBps/mm (compared to 107.4 GBps/mm in our design) [Het24]. This results in designing Fred switch chips with only 18.4% of current area with the same I/O BW.

The other I/O technology alternative is using the serialized high-speed links such as UCIe Advanced [Uni24], which can deliver up to 1 TBps/$mm$. This results in designing Fred switch chips with only 5% of the current area. Note that even with the high area assumption of Fred, we don’t expect the yield issue to be a practical problem since compared to the compute NPUs, Fred switches have much less internal logic and hence encounter fewer defects.

<span id="table-06"></span>

![表 6. 目标工作负载.](./fred/table-06.png)

**表 6.** 目标工作负载.

<span id="section-7"></span>

## 7 评估方法

<span id="section-7-1"></span>

### 7.1 基线和 Fred 配置

**Baseline.** The baseline topology is a $5\times 4$ 2D-mesh with I/O controllers attached to the edge NPUs, similar to prior multi-chiplet wafer-scale prototypes [Arc19, Des21, Sim19, Enh24, Chi24a]. Since each NPU has 3 TBps bandwidth ([第 6.2.2 节](#section-6-2-2)), each NPU-to-NPU link in the 2D-Mesh is equal to $750$ GBps, resulting in the bisection BW of 3.75 TBps. The I/O Controller-to-NPU is $128$ GBps.

**Fred.** We test four different variations of Fred to show how different features of Fred contribute to the overall performance. [表 5](#table-05) shows the target configurations. *Fred-A* shows the effect of going from mesh to switch-based topology with the same bisection and without in-network collective execution. *Fred-B* builds on top of Fred-A and adds the in-network collective execution feature. *Fred-C* increases the bisection BW without in-network collective execution. Finally, Fred-D is the most optimal variant of Fred by adding the in-network collective execution to the previous variant.

<span id="section-7-2"></span>

### 7.2 Collective Algorithm

For the baseline 2D mesh and when there is a wafer-wide collective, we use the hierarchical 2D algorithm with two concurrent chunks (in reverse direction) to enhance utilization [Hig20, Enh24]. For collectives between arbitrary NPUs, we build logical rings between involved NPUs and perform the ring algorithm. We also use X-Y routing, which is common in real systems [Hig20]. For Fred-A and Fred-C, we use the hierarchical 2-D ring algorithm to reduce the traffic of L1-L2 links, similar to [Cho19]. Fred-B and Fred-D use the in-network capability and use the hierarchical Fred switch topology to perform the collective, as explained in [第 6.1 节](#section-6-1).

<span id="section-7-3"></span>

### 7.3 目标工作负载和执行模式

In the interest of space, we evaluate four training workloads, ranging from 60M to 1T parameters to be the representative for a broad range of ML workloads. [表 6](#table-06) shows the target workloads and their corresponding parallelization strategy and execution models studied in [第 8.2 节](#section-8-2). ResNet-152 and Transformer-17B (Transformer model with 17 billion parameters) can fit on the on-wafer memory and hence, use the *权重驻留* execution mode ([第 3.1 节](#section-3-1)). In contrast, GPT-3 and Transformer-1T (Transformer model with 1 trillion parameters) use the *权重流式传输* execution mode ([第 3.1 节](#section-3-1)). Workers within the same DP group perform All-Reduce together during the back-propagation to sync on weight gradients. In *权重驻留* mode, the workers use the Microsoft ZeRO optimizer stage 2 [Raj20b] along the DP dimension to reduce the memory footprint. Note that in *权重流式传输* mode, the DP groups should reduce the gradients as they stream them out to the external memory through the I/O controller. The pattern is the reverse communication direction of [图 4](#figure-04). For Transformer-17B, GPT-3 and Transformer-1T, the model split is based on the Megatron-LM method [Sho19], which requires two All-Reduces (along the MP dimension) for each transformer layer stack during forward-pass & back-propagation. For the PP split on Transformer-17B, we assume the minibatch is divided into 8 microbatches to hide the effect of pipeline bubbles [Hua19]. For GPT-3, however, pipelining works differently since it is combined with the 权重流式传输. In this case, $\mathrm{\mathrm{PP}}\>=\>2$ indicates that each time $2$ consecutive layers are brought to the wafer and distributed among different NPUs along the PP dimension. Thus, splitting the minibatch into two microbatches is enough to hide the pipeline latency. In [第 8.1 节](#section-8-1) and [第 8.2 节](#section-8-2), the minibatch size for all workloads is set to DP_size$\times 16$, while in [第 8.3 节](#section-8-3) (and also [图 2](#figure-02)) the minibatch size is increased to DP_size$\times 40$ to allow for finer-grain pipelining when PP_size increases [+6]. All workloads use FP16 gradient precision.

<span id="section-7-4"></span>

### 7.4 仿真框架

We use ASTRA-SIM [Ast20, Ast20a], which is an open-source simulation methodology for modeling distributed training systems. ASTRA-SIM enables the profiling of compute and communication performance of distinct wafer-scale fabrics, including Fred. It can model various parallelization strategies and the overlapping of compute with comm kernels. Additionally, its network back-end can simulate the comm operations in detail. We extend ASTRA-SIM to model the I/O-to-wafer transfers for both the 权重驻留 and 权重流式传输 scenarios. For each workload, we run the simulation for two training iterations (i.e., two forward + two backward-pass).

Previous works have shown that endpoint-based collective execution (our baseline) puts more pressure on the endpoint’s compute and memory BW resources, hindering the compute kernel efficiency [Ena21]. To favor the baseline and only focus on the network characteristics, we omit such effects in our baseline system and assume the compute kernels can run as efficient as the in-network collective execution systems such as Fred.

**Metric of Evaluation.** In [第 8 节](#section-8), we report the end-to-end training times and their breakdowns into total compute time and different *exposed* communication times. Since the minibatch size per training iteration may be different depending on the parallelization strategy, we normalize the reported times by dividing the latencies by the minibatch size when comparing the different parallelization strategies of the same workload (e.g., [图 2](#figure-02)). The exposed communication time refers to the amount of time that is not overlapped with the compute time and the workload is waiting for the communication to be finished. Depending on the parallelization strategy and execution model, there might be multiple sources of exposed communication times—load, DP, MP, PP, and/or 权重流式传输.

<span id="section-8"></span>

## 8 结果

<span id="section-8-1"></span>

### 8.1 微基准测试结果

[图 9](#figure-09) presents the communication breakdown across 3D parallelism phases for two parallelization strategies for Transformer-17B. For the MP(20)-DP(1)-PP(1) strategy, there are only wafer-wide All-Reduce operations for the MP communication. The baseline effective BW utilization is bounded by the corner NPUs since they have only 2 links to other NPUs. This limits the average network BW utilization of each NPU to be around $2\times 750\mathrm{\mathrm{GBps}}=1500\mathrm{\mathrm{GBps}}$. In Fred-A, each NPU-L1 BW is 3 TBps, but NPU-L2 BW is 375GBps. [+7] Using a similar analysis as [The22a], we see that hierarchical collectives result in NPU-L2 BW being the bottleneck and the effective NPU BW utilization is $375\mathrm{\mathrm{GBps}}+4\times 375\mathrm{\mathrm{GBps}}=1850\mathrm{\mathrm{GBps}}$. In Fred-B, the L1 switches first perform the All-Reduce and then use the entire L1-L2 BW to forward the data to the L2 switches for the second All-Reduce. Therefore, each NPU can send the data to L2 switch at the speed of $1500\mathrm{\mathrm{GBps}}$ (L1-L2 BW). However, since it is an in-network collective execution, the amount of traffic each NPU sends out is almost half of the traffic in the endpoint-based collective. Fred-C has much more L1-L2 BW and therefore each NPU can drive the BW utilization to $3\mathrm{\mathrm{TBps}}$. In Fred-D, an additional in-network collective execution reduces the traffic by half in addition to the $3\mathrm{\mathrm{TBps}}$ NPU BW utilization.

<span id="figure-09"></span>

![图 9. Transformer-17B 两种并行策略在 3D 并行不同阶段的通信微基准结果, 仅比较通信性能 (策略见[图 2](#figure-02)).](./fred/figure-09.png)

**图 9.** Communication microbenchmark results for comparing only communication performance at different phases of 3D-parallelism, for two different parallelization strategies of Transformer-17B from [图 2](#figure-02).

The MP(2)-DP(5)-PP(2) case has all MP (All-Reduce), DP (All-Reduce), and PP (multicast) communications. For the MP communications, the baseline NPU can only utilize 1 link (out of its up to 4 links), resulting in only $750\mathrm{\mathrm{GBps}}$ BW utilization. Since all the communicating NPUs are below the same L1 switch in Fred topologies, they can use the entire $3\mathrm{\mathrm{TBps}}$ of NPU-L1 BW to communicate. Additionally, in the special case when the number of peer NPUs is two, the amount of traffic for endpoint-based vs. in-network execution is the same. Hence, all Fred variants have the same performance for MP communication.

Again, the baseline is limited by the corner NPUs, which can utilize only one of their links for DP communication. Hence, the baseline NPU BW is $750\mathrm{\mathrm{GBps}}$. In Fred, and for the DP communication, each NPU should communicate with four other NPUs under different L1 switches. Therefore, in Fred variants the L1-L2 BW should be shared across four collective flows. Therefore, L1-L2 BW plays a significant role in the performance of this collective. In Fred-A, each NPU has an average NPU-L2 BW of $375\mathrm{\mathrm{GBps}}$, and hence, the NPU BW utilization is only $375\mathrm{\mathrm{GBps}}$, which is worse than the baseline. In Fred-B, however, the L2 switch is used to perform All-Reduce for each flow. This reduces the traffic generated by each NPU roughly by $37.5\%$, which makes its overall performance closer to the baseline. In Fred-C, however, the NPU-L2 BW is increased to $3\mathrm{\mathrm{TBps}}$. Finally, Fred-D Improves the Fred-C by performing in-network collective and reducing the traffic by $37.5\%$.

For the PP comm, the baseline NPU can utilize one of its links to forward data to the next pipeline stage and hence, its BW utilization is 750GBps. Note that this is possible since in the case of language models such as Transformer-17B, one NPU within the mp group is sufficient to multicast the output to all NPUs at the next stage, [+8] and hence, there is no contention between NPUs of the same MP group at the same stage. In Fred, all peer NPUs are below the same L1 switch and can utilize the entire $3{\mathrm{\mathrm{TBps}}}$ BW for the PP comm.

**讨论: Fred’s NPU to L1 Topology Logic.** Now that we have presented the microbenchmark results, we can discuss why we preferred to choose a tree-based topology to connect every four NPUs to the L1 switches. An alternative solution can be a fully-connected topology to connect every four NPUs and then use only one switch level. However, this design choice still suffers from the endpoint-based effects (i.e., increased use of compute and memory BW at the endpoint) discussed in [第 7.4 节](#section-7-4). Furthermore, as explained in [第 2.2 节](#section-2-2), endpoint-based methods produce more communication traffic. For example, in the case of four NPUs, the most endpoint-based BW optimal algorithms produce 1.5D traffic per NPU to perform an All-Reduce of size D [Tha05, The22a], while the in-network collective execution produces only D traffic per NPU [Sca21], 50% lower than the fully connected topology.

<span id="figure-10"></span>

![图 10. 端到端训练时间分解为计算时间和不同通信时间. 每个工作负载的运行时间均相对于对应基线归一化.](./fred/figure-10.png)

**图 10.** End-to-end training times are decomposed into compute times and different communication times. The runtime of each workload is normalized to its corresponding baseline.

<span id="section-8-2"></span>

### 8.2 全工作负载结果: 深入分析

[图 10](#figure-10) shows the end-to-end runtimes of the training workloads for the baseline vs. Fred. Due to space limitations, we only show the Fred-C and Fred-D in comparison with the baseline. However, we note that Fred-A and Fred-B results are between the baseline and Fred-C, in terms of performance. In general, input activations, compared to the model parameters, are relatively small in size and hence, do not have significant overhead on the total iteration time. Additionally, the input activations of the next iteration can be prefetched to the wafer whenever the wafer-scale interconnect is idle. Hence, we observe **no** *initial_input_load* exposed comm for any of our target workloads, except for the Transformer-1T.

ResNet-152 uses pure DP with a 权重驻留 model. Hence, the only communication costs that repeat on each training iteration are the input minibatch loading and DP communication. As explained earlier, in wafer-wide All-Reduce collective, the baseline is able to utilize $1.5$ TBps of NPU BW. Fred-C and Fred-D can achieve $3$ TBps NPU BW but Fred-D can further reduce the network traffic by $\approx 2\times$, resulting in a significant reduction of DP exposed comm. Thus, Fred-C and Fred-D can improve the end-to-end training runtime by $1.41\times$ and $1.76\times$, respectively, for ResNet-152.

Transformer-17B uses all dimensions of the 3D-parallelism and therefore, has all DP, MP, and PP communication overheads. The baseline device placement favors MP, but compromises the PP and DP comms, especially due to the non-aligned parallelization strategy dimensions as explained in [第 3 节](#section-3). Another drawback of the baseline is the underutilized links due to the non-overlapping nature of MP/DP/PP comms (see [第 3 节](#section-3)). Fred-C, on the other hand, does not have the problem of underutilized links and non-aligned parallelization strategies. It also does not require favoring any of DP, MP, or PP over the other strategies. Fred-D can further improve the MP and DP collectives’ performance due to in-switch collective execution capability. As a result, Fred-C and Fred-D can improve the overall end-to-end training performance by $1.75\times$ and $1.87\times$, respectively.

GPT-3 combines 权重流式传输 with 3D-parallelism. Using the analysis of [第 3 节](#section-3), the baseline topology is unable to stream weights with the full line-rate of I/O controllers. The reason is that the hotspot link requires $(2\times 5\>-1)\times 128\text{ \mathrm{\mathrm{GBps}}}\>=\>1152\text{ \mathrm{\mathrm{GBps}}}$, while link capacity is only $750$ GBps. Therefore, the I/O channels should work with $\frac{750}{1152}=0.65\times$ of the line-rate. The MP/PP comm performance of Fred-C and Fred-D is $\approx 4\times$ better than the baseline, due to the underutilized links in the baseline. Note that the reason why Fred-C and Fred-D have the same performance for MP collective comm is because dim(MP)=2. In this special case, as explained earlier, end-to-end and in-switch collective execution have the same amount of networking traffic and hence, have the same performance. In total, Fred-D and Fred-C outperform the baseline by $1.34\times$ in terms of overall training time for GPT-3.

Transformer-1T is another 权重流式传输 workload, but with only DP parallelism. As a result, the 权重流式传输 delay is the only communication overhead in addition to the initial input load. The high-performance compute NPUs and limited off-chip I/O BW puts the 权重流式传输 performance directly on the critical path. This means that the NPUs can work with the line-rate of the weight being streamed, and the main limiting factor is how fast all the weights can be streamed. In this case, both Fred-C and Fred-D can leverage the full I/O BW, while the baseline topology can only work with $0.65\times$ of the total I/O BW as explained earlier. Additionally, since I/O controllers are always being utilized for 权重流式传输, there is no idle time to prefetch the input minibatch of the next iteration during the current training iteration. Hence, the initial input load cannot be hidden, although its overhead is very negligible. In total, using Fred-C/Fred-D improves the training time by $1.4\times$.

<span id="figure-11"></span>

![图 11. Transformer-1T 和 Transformer-17B 在不同并行策略下的基线与 Fred-D 对比.](./fred/figure-11.png)

**图 11.** Baseline vs. Fred-D for various parallelization strategies of Transformer-1T and Transformer-17B

<span id="section-8-3"></span>

### 8.3 不同并行策略

To test the efficiency of Fred for different parallelization strategies, we pick two workloads, Transformer-17B and Transformer-1T, and compare the baseline performance vs. Fred-D in [11(a)](#figure-11) and [11(b)](#figure-11), respectively. The Avg. bars are obtained across all parallelization strategies similar to [图 2](#figure-02), however, not all individual parallelization strategies of [图 2](#figure-02) are shown in [11(a)](#figure-11) and [11(b)](#figure-11) due to lack of space. As can be observed from both figures, Fred-D can significantly improve communication performance and reduce the total exposed communication in all parallelization strategies.

Such improvements make the most compute-efficient (i.e., least compute time) parallelization strategy also to be the be the best parallelization strategy overall. For example, for Transformer-17B the most compute efficient strategy is MP(20)-DP(1)-PP(1). However, this configuration does not have the lowest overall training time in the baseline system due to its huge exposed communication overheads. Thanks to the benefits of Fred-D in reducing the share of exposed communication overheads, this configuration is now the most optimal compared to other parallelization strategies. This is also true for Transformer-1T, where the most compute-efficient strategy (i.e., MP(5)-DP(1)-PP(4)) is now the most optimal strategy.

Overall, when averaged across all parallelization strategies, Fred-D can improve the exposed communication time by $4.22\times$ and $3.92\times$, resulting in training speedup by $1.63\times$ and $1.44\times$ for Transformer-17B and Transformer-1T, respectively.

**讨论: going beyond a single wafer.** While the main focus of this paper is on providing flexible on-wafer interconnects to allow for more flexible parallelization strategies, here we discuss the possible scenarios when the model cannot fit on a single wafer. The first method is to pyramidically load and unload parts of the model (i.e., 权重流式传输) as we considered and evaluated in the paper. However, in some cases more than one wafer is needed for training to reduce the training time. In that case, the optimal inter-wafer topology is an open question. Some methods use reduction trees to accumulate the gradients obtained from different wafers [Cer21]. This method, although efficient for data-parallel strategy across the wafers, is not flexible if we consider other parallelization strategies across wafers. A Fred-like inter-wafer interconnect can be constructed to allow for more flexibility across the wafers. In any case, on-wafer Fred topology can work in tandem with the inter-wafer interconnect to form efficient hierarchical collectives. For example, a global all-reduce can be broken into: i) a special intra-wafer reduce scatter performed by Fred where only the boundary NPUs with access to the I/O maintain the results, followed by ii) an All-Reduce facilitated by the inter-wafer interconnect where boundary NPUs reduce the data across different wafers, iii) followed by the final intra-wafer special All-Gather done by Fred where the boundary NPUs broadcast the final result to all NPUs within the same wafer.

**讨论: going beyond 3D Parallelism.** While the main focus of this paper was on the MP/DP/PP parallelism, recently, more parallelization strategies have been proposed. Examples include Expert-Parallelism (EP) [A23b], Context Parallelism (CP) [Dis25], and more customized and non-homogeneous strategies where the parallelization strategy might change layer by layer [Jia19a]. While not quantitatively studied in this paper, we expect that increasing the parallelization strategy dimensions further increases the network congestion and reduces the effective network BW for each parallelism dimension on the baseline 2D Mesh. This highlights the need to have a flexible network fabric such as Fred.

<span id="section-9"></span>

## 9 相关工作

**Accelerator Fabrics.** Prior works on *flexible* DNN Accelerators [Mae18, Sig20, Fle23, Eye19, Cus17] have explored indirect topologies such as Benes/Fat-tree/Clos for efficiently distributing operands and reducing partial sums. This work leverages this concept to build a topology optimized for collectives.

**In-switch Collectives.** The idea of in-switch collective execution has been proposed in many previous works for different network levels. The *P4* language [P14] allows for offloading application-specific tasks to network switches that support the P4 abstract architecture. [Sca21, Atp21] proposed programming datacenter Ethernet switches for offloading the All-Reduce collective for data-parallel training. iSwitch [Acc19] utilizes FPGA logic within switches to offload the All-Reduce functionality for the distributed training of reinforcement learning. Mellanox SHARP [Mel20] is an Infiniband switch architecture for performing collectives. Klenk *et al.* [An20] propose a method to offload collectives to the scale-up (e.g., NVlink[Eva19]) NPU fabric. Clos topologies have also been explored in prior works [Cus17, Atp21]. A fundamental difference between these works and Fred is that they are proposed for *off-chip* networks, which have significantly less BW compared to on-package networks. In many of these solutions, the internal switch BW should be at least $2\times$ and $P\times$ the link BW to be efficient (i.e., line-rate) for All-Reduce and Reduce between $P$ ports, respectively. This is due to the switch architecture that performs the reductions only after the routing and on the output port. While the difference between off-chip links and on-chip switch architectures allows for provisioning such BW differences, it is not applicable for on-package/on-wafer platforms where the links are on-chip and can have the same BW as the switches. In contrast, Fred performs the reduction operations in multiple steps ($\mu$Switches) during the routing on the Fred interconnect. Hence, the Fred switch works with the same BW as the links and can provide line-rate throughput.

<span id="section-10"></span>

## 10 结论与未来工作

We propose Fred, a high-BW wafer-scale fabric that is flexible for different configurations of the 3D parallelization strategies of distributed training workloads. Fred is able to support concurrent in-network collective execution efficiently, enabling the upper-level compiler to further optimize the parallelization strategy for compute and memory utilization. We plan to study Fred for distributed inference as a part of our future work.

Acknowledgements.

[+1]: All model updates over different training iterations happen on-chip.

[+2]: Model updates involve low operational intensity. Hence, performing these updates off-chip prevents wasting I/O bandwidth by avoiding loading optimizer states onto the chip for lightweight operations.

[+3]: For example, Fred<sub>3</sub>($8$) can route all the flows in [图 7](#figure-07)(j).

[+4]: We note that Fred layout shown is not tiled. This means that the substrate (where chiplets are bonded) may not be able to use stepper-based lithography. But direct-written maskless lithography is not uncommon for substrate patterning. This was used in a commercial packaging provider ThinkDeca [Ada24a]. Such patterning has no symmetry requirement, albeit it has a lower throughput. Also, note that using maskless lithography increases the substrate manufacturing moderately [Cos10], but substrate manufacturing is a small fraction of the total system cost [Des21].

[+5]: Compound collectives have multiple phases

[+6]: For these results, we assume the number of microbatches is 1, 10, 20, 20, 20, 40 for the Transformer-17B with $\mathrm{\mathrm{PP}}$ size of 1, 2, 4, 5, 10, 20, respectively. For Transformer-1T, the number of microbatches is equal to the $\mathrm{\mathrm{PP}}$ size.

[+7]: Assuming the L1-L2 BW is equally shared among all NPUs.

[+8]: All NPUs within the same MP group produce the same output in this case
