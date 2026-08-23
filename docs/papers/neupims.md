---
title: 'NeuPIMs'
createTime: 2026/08/23 12:00:00
permalink: /papers/neupims/
pageClass: paper-reading
---

> [Guseul Heo](https://dblp.org/pid/299/7465.html), [Sangyeop Lee](https://dblp.org/pid/290/9678.html), [Jaehong Cho](https://dblp.org/pid/297/4770.html), [Hyunmin Choi](https://dblp.org/pid/288/1825.html), [Sanghyeon Lee](https://dblp.org/pid/286/7145.html), [Hyungkyu Ham](https://dblp.org/pid/322/2213.html), [Gwangsun Kim](https://dblp.org/pid/335/5960.html), [Divya Mahajan](https://divyamahajan.github.io/), 和 [Jongse Park](https://casys.kaist.ac.kr/). 论文于 2024 年 3 月 1 日首次提交至 arXiv, 当前版本为 v1. 发表于 [ASPLOS '24](https://doi.org/10.1145/3620666.3651380), pp. 722-737. [NeuPIMs: NPU-PIM Heterogeneous Acceleration for Batched LLM Inferencing](https://arxiv.org/abs/2403.00579). [原始 PDF](/paper/neupims.pdf). [DOI](https://doi.org/10.1145/3620666.3651380). [TeX 源文件](https://export.arxiv.org/e-print/2403.00579). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

现代基于 Transformer 的大语言模型 (LLM) 由一系列解码器块构成. 每个块包含 3 个主要组件: (1) QKV 生成, (2) 多头注意力, (3) 前馈网络. 批处理时, QKV 生成和前馈网络涉及计算密集的矩阵-矩阵乘法 (GEMM), 多头注意力则需要带宽密集的矩阵-向量乘法 (GEMV). TPU 或 NPU 等机器学习加速器擅长 GEMM, 但处理 GEMV 的效率较低. 相反, 存内处理 (PIM) 适合高效执行 GEMV, 却缺少有效处理 GEMM 所需的计算能力.

基于这一观察, 我们提出 NeuPIMs, 一种同时利用传统 GEMM 型 NPU 和 GEMV 优化型 PIM 设备的异构加速系统. 将 NPU 与 PIM 高效整合的难点在于让两种平台同时运行各自负责的内核. 现有 PIM 通常工作在"阻塞"模式, 任意时刻只能启用 NPU 或 PIM; 同时, LLM 中 GEMM 与 GEMV 的依赖关系限制了并行处理. 为解决这些问题, NeuPIMs 在每个 bank 中加入双行缓冲区, 让内存读写和 PIM 命令可以同时进行. NeuPIMs 还采用运行时子批次交错, 利用批次并行在同一设备中流水执行 2 个独立子批次. 实验表明, 相比仅 GPU, 仅 NPU 和朴素 NPU+PIM 集成方案, NeuPIMs 的吞吐量分别提升 3 倍, 2.4 倍和 1.6 倍.

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) 已广泛用于自然语言理解, 内容生成和决策支持 [Ope23, Hof22, Bla22, Tou23]. 但这类模型对内存和计算资源的需求很高. 本文关注 GPT-4 [Ope23] 和 LLaMA [Tou23] 等当代 LLM 的推理问题.

这些模型的共同架构是由多个解码器块堆叠而成. [图 1](#figure-01)(a) 展示了每个块的 3 个主要层: (1) 查询-键-值 (QKV) 生成, (2) 多头注意力 (MHA), (3) 前馈网络 (FFN). 为高效计算这些块, 系统通常会批处理多个推理请求. 批处理使 QKV 生成和前馈层可以在请求间复用权重, 形成权重矩阵与激活矩阵之间的通用矩阵乘法 (GEMM). 相反, 多头注意力在激活矩阵和激活向量之间进行乘法, 几乎没有数据复用, 因而形成通用矩阵-向量乘法 (GEMV).

LLM 推理包含大量大规模 GEMM 和 GEMV. 常见做法是使用 GPU 或 TPU 等高性能机器学习加速器, 本文将它们统称为神经处理单元 (NPU). NPU 针对计算密集任务, 尤其是 GEMM, 进行了优化. GEMV 的算术强度较低, 因此无法充分利用 NPU 的计算资源. PIM 对 GEMV 更有优势 [Att24, New20, Tra22, Har21, Neu24a], 但不擅长 GEMM.

本文提出用于 LLM 批量推理的 NeuPIMs. 系统同时利用 (1) 由多个脉动阵列组成的 GEMM 型 NPU, (2) 多个适合 GEMV 的 PIM 加速器. 设计中有两个问题:

- **微架构问题:** 现有 PIM 工作在"阻塞"模式, 无法同时执行 NPU 和 PIM.
- **算法问题:** 解码器块中的 GEMM 和 GEMV 存在数据依赖, 限制了 NPU+PIM 并行.

NeuPIMs 通过硬件-算法协同设计解决这些问题.

1. **微架构贡献.** NeuPIMs 修改 PIM bank, 让常规内存访问与 GEMV 同时发生. 具体做法是为两种功能设置独立的行缓冲区, 即双行缓冲区. 设计还在内存控制器中交错调度内存命令和 PIM 命令, 避免违反 DRAM 时序, 并增加少量复合命令以一次执行多个 GEMV.
2. **算法贡献.** 我们提出子批次交错, 同时处理两个独立子批次. 一个子批次的 GEMM 可以和另一个子批次的 GEMV 并行, 从而提高 NPU 和 PIM 的利用率. 我们估计序列长度到 PIM 侧 MHA 延迟的映射, 再划分批次, 使两个子批次的序列长度总和接近.

<span id="figure-01"></span>

![图 1. 解码器块, 基线加速器和 NeuPIMs 加速器.](./neupims/figure-01.png)

**图 1.** (a) 构成 LLM 的解码器块数学组件, (b) 使用非 PIM 内存的仅 NPU 基线, (c) NPU+PIM 集成基线, (d) NeuPIMs 加速器.

NeuPIMs 结合上述两项改进, 同时提高 NPU 和 PIM 利用率. 我们使用 4 个 GPT-3 变体以及 ShareGPT 和 Alpaca 数据集进行评估, 并将 ONNXim 与基于 DRAMsim3 的 PIM 模拟器结合. 相比仅 NPU 和朴素 NPU+PIM 基线, 吞吐量分别提升 2.4 倍和 1.6 倍; NPU 和 PIM 利用率从 28% 和 17% 提升到 65% 和 26%.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 LLM 推理的计算特征

<span id="figure-02"></span>

![图 2. LLM 的模型架构和推理过程.](./neupims/figure-02.png)

**图 2.** LLM 的模型架构和推理过程.

**LLM 的模型架构和执行.** [图 2](#figure-02) 展示了当前大语言模型共有的架构 [Tou23, Bla22]. 对输入提示, 模型先进入摘要阶段, 编码输入并建立生成阶段所需的上下文. 生成阶段以自回归方式每次产生一个 token, 并在下一次迭代使用已生成的键值投影. 两个阶段都由解码器块组成, 每个块包含 QKV 生成, 多头注意力和前馈网络.

<span id="figure-03"></span>

![图 3. LLM 解码器块中的算子.](./neupims/figure-03.png)

**图 3.** LLM 解码器块中的算子.

**LLM 的批处理推理.** MHA 层与 QKV 生成和 FFN 层的计算特征不同. [图 3](#figure-03) 展示了权重-激活和激活-激活乘法. QKV 生成和 FFN 将每个 token 的 Q/K/V 激活或注意力向量与训练得到的权重矩阵相乘. 在摘要阶段有多个 token 向量, 或者多个推理被批处理时, 这些 GEMV 会转为 GEMM. MHA 则将当前 token 的激活与此前所有 token 的激活相乘, 形成 GEMV. 每个请求的激活操作数都不同, 无法批处理, 因而计算受内存带宽限制.

**算术强度分析.** 我们使用 GPT3-13B 和 GPT3-175B 进行 roofline 分析. [图 4](#figure-04) 给出算术强度 (FLOPS/byte) 与性能 (TFLOPS) 的关系. 两个模型的生成阶段都严重受内存限制, 摘要阶段则受计算限制. 两个阶段交替执行且存在依赖, 难以在同构平台上获得高利用率. 因此我们将面向 GEMM 的脉动阵列 NPU 与面向 GEMV 的 PIM 结合.

<span id="figure-04"></span>

![图 4. LLM 各层的算术强度.](./neupims/figure-04.png)

**图 4.** LLM 各层的算术强度.

<span id="section-2-2"></span>

### 2.2 LLM 推理服务

LLM 需要大量资源, 因此通常使用 DeepSpeed, Orca 和 vLLM [Vll23] 等大规模推理服务框架. 这些框架为提示请求提供推理服务并支持批处理.

**选择性批处理.** 批处理通常可以提高神经网络推理的资源利用率, 但 MHA 层不支持批处理. Orca 因此逐个计算注意力层, 同时批处理 QKV 生成和 FFN. 这种特性需要 GEMM 和 GEMV 同时计算, 正是本文的动机.

**迭代级调度.** 推理服务以流式方式接收请求, 没有确定的调度顺序. Orca 在每次迭代开始时调度批处理, 允许新请求加入并移除已完成请求. NeuPIMs 沿用这一策略, 在迭代边界管理请求.

**注意力的内存分页.** vLLM 关注 KV cache 的内存管理. QKV 生成会产生在生成阶段复用的 KV cache, 长序列会使它占用大量内存. vLLM 为缓存数据引入分页, 避免过早分配内存. NeuPIMs 采用相同的分页机制来提高有效批次大小.

*NeuPIMs 面向包含上述技术的推理服务系统.*

<span id="section-3"></span>

## 3 动机

本节说明 NeuPIMs 的设计依据. 我们先分析 GPU LLM 推理服务的问题, 再讨论朴素 NPU-PIM 集成的局限.

<span id="section-3-1"></span>

### 3.1 基于 GPU 的 LLM 推理服务

LLM 需要大量内存, 通常部署在多 GPU 集群上, 并使用流水并行和张量并行.

<span id="figure-05"></span>

![图 5. 4 个 LLM 的 GPU 资源利用率.](./neupims/figure-05.png)

**图 5.** 4 个不同 LLM 的 GPU 资源利用率.

**GPU 系统的低利用率.** 我们分析配备 GPU 的基线, 比较 RTX 3090 24GB 和 A100 40GB 运行 GPT-NeoX, LLaMA2, OPT 和 MPT 的情况. [图 5](#figure-05) 展示利用率及层间变化. 由于 GPU 数量按容量约束决定, 容量利用率接近 100%. 但计算资源利用率始终低于 40%. 造成低利用率的原因是带宽不足, 即使 A100 配备 HBM 并提供 1,555 GB/s 的总带宽. 只要 GEMM 与 GEMV 之间存在串行依赖, 这种失衡就无法避免.

<span id="figure-06"></span>

![图 6. 解码器块中的 NPU 和 PIM 利用率.](./neupims/figure-06.png)

**图 6.** 解码器块中的 NPU-PIM 资源利用率.

<span id="figure-07"></span>

![图 7. NeuPIMs 系统概览.](./neupims/figure-07.png)

**图 7.** NeuPIMs 系统概览.

<span id="section-3-2"></span>

### 3.2 朴素 NPU-PIM 方法

解决带宽瓶颈的直接办法是用 PIM 卸载带宽受限的计算. 我们设计了由脉动阵列 NPU 和 Newton PIM GEMV 加速器 [New20] 组成的朴素 NPU-PIM 加速器, 并采用[第 8 节](#section-8)的方法进行模拟.

[图 6](#figure-06) 给出了不同解码器层的 NPU 和 PIM 利用率. NPU 执行 QKV 生成, 投影和 FFN 时, PIM 利用率为 0; PIM 执行 MHA 时, NPU 利用率接近 0. 因此在整个执行过程中两者的利用率都低于 40%.

**NPU 与 PIM 并行执行的必要性.** 根本原因是 PIM 微架构禁止主机 (NPU) 与 PIM 单元同时执行, 只能串行使用彼此独立的资源. 要让 PIM 真正用于 NPU 加速器, 必须让两者并行执行.

<span id="section-4"></span>

## 4 NeuPIMs 概览

[图 7](#figure-07) 展示了系统. NeuPIMs 由带脉动阵列和向量单元的 NPU, 多个基于 HBM 的 PIM 通道, 以及将批次划分为两个子批次并交错执行的调度器组成.

1. **NeuPIMs 系统.** 系统包含主机 CPU, 多个 NeuPIMs 设备, 独立 NPU 和 PCIe 或 CXL 等高带宽互连. 摘要阶段完全由 GEMM 构成, 因此由独立 NPU 处理, NeuPIMs 负责生成阶段. 请求以流式方式到达, 被分配到 PIM 通道并在当前迭代完成前存放在请求池中.
2. **NeuPIMs 加速器.** bank 使用双行缓冲区: 一个用于 PIM 执行, 一个用于常规内存访问. NPU 可以访问当前未被 PIM 使用的行.
3. **NeuPIMs 调度算法.** 原型包含 32 个基于 HBM 的 PIM 通道, 每个通道都有内存控制器. 控制器交错发送内存和 PIM 命令, 同时满足时序约束.
4. **NeuPIMs 编译器框架.** 编译器前端接收类似 ONNX 的 LLM 和系统规格, 将模型转换为中间表示, 并生成 NPU 与 NeuPIMs 指令二进制, 调整 tile 大小和指令顺序.

<span id="section-5"></span>

## 5 NeuPIMs 架构

<span id="section-5-1"></span>

### 5.1 支持并发执行的 PIM 微架构

**基于 PIM 的单行缓冲区.** [图 8](#figure-08)(a) 展示了带单行缓冲区的 PIM GEMV 加速器. 向量操作数先放入通道内所有 bank 共享的全局缓冲区, 矩阵行从多个 bank 并行读取并放入对应行缓冲区, 随后乘法器和加法树计算部分点积.

**现有 PIM GEMV 加速器的局限.** 现有 PIM [New20, Har21] 以"阻塞"模式工作, 禁止 NPU 与 PIM 同时执行. 单个行缓冲区同时服务常规读写和 GEMV, 两种模式只能串行. 这对仅执行 GEMV 的 PIM 影响不大, 但会限制同时需要 GEMM 和 GEMV 的 LLM 推理.

**加入双行缓冲区.** [图 8](#figure-08)(b) 展示 NeuPIMs bank. 每个 bank 具有 MEM 和 PIM 两个行缓冲区, 并连接到独立数据通路. MEM 行缓冲区服务常规访问, PIM 行缓冲区服务 GEMV. 我们尽量减少微架构改动, 将复杂性放到命令接口和内存控制机制中. 原型使用 Newton [New20], 但方法适用于遵循标准 DRAM 架构和命令接口的 GEMV 加速器.

<span id="figure-08"></span>

![图 8. 单行和双行缓冲区的内存 bank 微架构.](./neupims/figure-08.png)

**图 8.** (a) 现有单行缓冲区 PIM 加速器, (b) NeuPIMs 双行缓冲区.

<span id="section-5-2"></span>

### 5.2 内存命令接口

**现有 PIM GEMV 命令接口.** NeuPIMs 在 DRAM 标准接口之上使用 PIM 命令. `PIM_GWRITE` 将某个 bank 的行复制到全局向量缓冲区; 分组的 `PIM_ACTIVATION` 同时激活多个 bank 的 PIM 行缓冲区; `PIM_DOTPRODUCT` 执行并行点积; `PIM_RDRESULT` 将结果传给主机.

<span id="figure-09"></span>

![图 9. PIM 命令时序比较.](./neupims/figure-09.png)

**图 9.** PIM 命令时序比较.

**NeuPIMs 命令接口.** 我们增加 3 个命令.

<span id="table-01"></span>

![表 1. NeuPIMs 命令集合.](./neupims/table-01.png)

**表 1.** NeuPIMs 命令列表.

- **PIM_HEADER:** 允许 GEMV 使用不同维度. 控制器可据此估计延迟并避开 DRAM 刷新.
- **PIM_GEMV:** 用一个复合命令替代多个细粒度点积和结果读取命令, 参数 $k$ 指定点积数量.
- **PIM_PRECHARGE:** GEMV 完成后预充电 PIM 行缓冲区, 类似常规 PRECHARGE 但目标是 PIM 行缓冲区.

<span id="section-5-3"></span>

### 5.3 内存控制器

NeuPIMs 包含多个通道, 每个通道有多个 PIM bank. 请求被分配到通道, MHA 在通道的多个 bank 上分布执行. 每个通道有独立的 PIM 命令队列, PIM 命令广播到该通道的所有 bank.

**内存读写与 PIM 命令的交错调度.** 控制器需要交错两类命令, 避免命令/地址总线成为瓶颈. NeuPIMs 优先发送 PIM 命令, 因为其发出延迟高于内存命令, 这样两类命令可以共享总线而不会显著降低性能.

<span id="section-6"></span>

## 6 NeuPIMs 调度

双行缓冲区让 NeuPIMs 可以同时处理 NPU 内存访问和 PIM 命令. 本节介绍 MHA 的重叠机会和批量推理中的子批次交错.

<span id="section-6-1"></span>

### 6.1 MHA 层的重叠机会

<span id="figure-10"></span>

![图 10. 多头注意力中的重叠机会.](./neupims/figure-10.png)

**图 10.** 多头注意力层的重叠机会. NPU-S 表示脉动阵列, NPU-V 表示向量单元.

<span id="figure-11"></span>

![图 11. 串行执行和子批次交错.](./neupims/figure-11.png)

**图 11.** LLM 解码器块的执行时间线: (a) 串行执行, (b) 子批次交错. $N$ 是解码器块数量.

[图 10](#figure-10) 展示 PIM 侧的 logit 和 attend 操作与 NPU 侧 softmax 操作的重叠. MHA 可在 head 粒度拆分, 但朴素 NPU-PIM 架构无法通过 PIM 通道在 PIM 单元和向量单元之间传递中间结果. 双行缓冲区使 NeuPIMs 可以并行使用 NPU 和 PIM, 向量单元无需等待所有 PIM GEMV 完成即可保存部分 logit 和 softmax. 这种重叠依赖 MHA 的 head 级并行, 且仅发生在 PIM 与 NPU 向量单元之间, NPU 脉动阵列在 MHA 执行期间仍基本空闲.

<span id="section-6-2"></span>

### 6.2 子批次交错

**串行执行的局限.** [图 11](#figure-11)(a) 展示朴素 NPU-PIM 设备中的解码器块. QKV 生成, MHA, 投影和 FFN 存在依赖, 必须串行执行.

**交错两个子批次.** 我们将大批次划分为两个子批次并交替执行. [图 11](#figure-11)(b) 展示一个子批次的 PIM 操作与另一个子批次的 NPU 操作如何同时执行.

**执行时间线比较.** 设单个 NeuPIMs 设备上有 $N$ 个解码器块. 不交错时每个块的算子依次执行, 总时间为 $N$ 倍的块时间. 交错后, MHA 时间被 NPU 脉动阵列的执行隐藏, 交错阶段主要受 NPU GEMM 时间限制. 因此 NPU 和 PIM 的利用率都得到提高.

**挑战.** 首先需要平衡每个子批次的 MHA 时间. MHA 延迟由处理最长序列的通道决定, 因而必须在通道间平衡 token 长度, 这由[第 6.4 节](#section-6-4)的算法完成. 其次两个子批次的执行时间应相近, 这由[第 6.5 节](#section-6-5)的划分算法完成.

<span id="section-6-3"></span>

### 6.3 多头注意力延迟估计

NPU 操作延迟很大程度取决于推理批次大小. 为优化 MHA, 我们根据键值映射到 PIM 内存布局来估计执行时间. GEMV 向量在 bank 间共享, 矩阵按行交错. 同一行列的 key cache 共享 layer 和 head 索引但序列索引不同; value cache 共享 layer, head 和序列索引, 每个 head embedding 交错放入各 bank. [算法 1](#algorithm-01) 使用该映射估计延迟.

<span id="algorithm-01"></span>

**算法 1: MHA 延迟估计.**

- **输入:** `seq_len`, 请求序列长度.
- **参数:** embedding 大小 $E$, 单个 PIM tile 的 GEMV 延迟 $L_{\mathrm{tile}}$, 全局缓冲区写延迟 $L_{\mathrm{GWRITE}}$, DRAM 页大小 $P_{\mathrm{DRAM}}$, 每通道 PIM bank 数 $B_{\mathrm{chnl}}$, head 数 $N_{\mathrm{head}}$.
- **输出:** MHA 延迟 $L_{\mathrm{MHA}}$.
- 将 $L_{\mathrm{MHA}}$ 初始化为 0.
- **对于** $K^\top\times\mathrm{Query}$ GEMV, 设置 $N_{\mathrm{tiles}}\leftarrow (\mathrm{seq\_len}/B_{\mathrm{chnl}})(E/P_{\mathrm{DRAM}})$, 并累加写延迟和 tile 延迟.
- **对于** $\mathrm{Logits}\times\mathrm{Value}$ GEMV, 设置 $N_{\mathrm{tiles}}\leftarrow ((E/N_{\mathrm{head}})/B_{\mathrm{chnl}})((\mathrm{seq\_len}/P_{\mathrm{DRAM}})N_{\mathrm{head}})$, 并累加相应延迟.
- **返回:** $L_{\mathrm{MHA}}$.

<span id="section-6-4"></span>

### 6.4 贪心最小负载装箱算法

NeuPIMs 将请求分配到 PIM 通道, 每个通道的 bank 分担 MHA. 为减少最拥塞通道与空闲通道的差异, 算法按序列长度降序处理请求, 将请求放入当前负载最小的通道并更新延迟估计.

<span id="algorithm-02"></span>

**算法 2: 贪心最小负载装箱.**

- **输入:** 新请求序列长度列表 $L_{\mathrm{req}}$, 各通道当前请求列表 $L_{\mathrm{chnl}}$.
- 初始化 $L_{\mathrm{load}}\leftarrow []$.
- **对于**每个通道, 累加其中请求的 MHA 延迟并加入 $L_{\mathrm{load}}$.
- **对于**每个新请求, 找到 $L_{\mathrm{load}}$ 中最小值的索引, 将请求加入对应通道, 再更新其负载.
- **返回:** $L_{\mathrm{chnl}}$.

<span id="section-6-5"></span>

### 6.5 子批次划分算法

由于 NPU 操作依赖推理批次大小, 两个子批次应保持规模平衡. [算法 3](#algorithm-03) 将每个通道中的请求分成两半, 分别加入两个子批次.

<span id="algorithm-03"></span>

**算法 3: 子批次划分.**

- **输入:** 每个通道的活跃请求集合 $L_{req}$.
- **输出:** 交错执行的 $\mathrm{SB}_1$ 和 $\mathrm{SB}_2$.
- 初始化 `turn=True` 和 $\mathrm{SB}_1,\mathrm{SB}_2\leftarrow [],[]$.
- **对于**每个通道, 取请求数量的一半. 若数量为奇数, 交替将多出的请求放入一侧, 然后将前半部分加入 $\mathrm{SB}_1$, 后半部分加入 $\mathrm{SB}_2$.
- **返回:** $\mathrm{SB}_1,\mathrm{SB}_2$.

<span id="section-7"></span>

## 7 NeuPIMs 系统扩展

模型并行将参数分布到多个 NeuPIMs 设备上, 这是因为单个设备的内存容量有限. 本节讨论流水并行和张量并行在 NeuPIMs 上的适用性.

<span id="section-7-1"></span>

### 7.1 NeuPIMs 的流水并行

流水并行按层划分模型, 每个设备放置若干层, 再将批次切成微批次并流水处理. NeuPIMs 可以采用相同方法, 但每个设备上的解码器块数量和批次大小都会下降; 子批次交错会进一步降低批次大小, 造成脉动阵列利用率不足.

<span id="section-7-2"></span>

### 7.2 NeuPIMs 的张量并行

张量并行将模型张量切成多个分片并分发到设备. 各设备并行执行后需要通信聚合结果. 子批次交错使通信频率增加一倍, 但总通信量不变, 因而额外开销有限. 一个子批次通信时, 另一个子批次可以继续计算. 因此我们优先使用张量并行, 只有当模型太大而无法仅靠张量并行容纳时才加入流水并行.

<span id="section-8"></span>

## 8 评估

<span id="section-8-1"></span>

### 8.1 方法

**基线.** 我们比较仅 GPU, 仅 NPU, NPU+PIM 和 NeuPIMs.

- **仅 GPU:** 使用 NVIDIA A100 40GB 和 PyTorch 编译的批量推理工作负载.
- **仅 NPU:** 不含 PIM 的 NPU, 具有等效内存带宽, 同时包含脉动阵列和向量单元.
- **NPU+PIM:** 将 Newton PIM GEMV 加速器与现成 NPU 集成, 仅把 MHA GEMV 映射到 PIM, 请求按轮询方式分配到通道.

**周期级模拟.** NeuPIMs 模拟器基于 ONNXim 和 DRAMsim3, 通过修改 ONNXim 的内存接口将访问卸载到 DRAMsim3.

<span id="table-02"></span>

![表 2. NeuPIMs 硬件规格.](./neupims/table-02.png)

**表 2.** NeuPIMs 硬件规格.

**硬件规格.** 原型是包含 8 个脉动阵列和 SIMD 向量单元的多芯粒设计. 每个内存通道控制 32 个 PIM bank, 总容量为 1 GB. 这些配置可以根据模型大小和序列长度调整.

<span id="figure-12"></span>

![图 12. 端到端吞吐量比较.](./neupims/figure-12.png)

**图 12.** 仅 GPU, 仅 NPU, NPU+PIM 和 NeuPIMs 在 Alpaca 与 ShareGPT 上的吞吐量比较.

<span id="table-03"></span>

![表 3. 评估所用的 LLM 配置.](./neupims/table-03.png)

**表 3.** 评估所用的 LLM 配置.

**LLM 模型.** 我们使用[表 3](#table-03) 中的 4 个 GPT-3 变体. NeuPIMs 也适用于其他仅解码器生成模型.

**数据集.** 我们使用 ShareGPT 和 Alpaca. ShareGPT 是从 ChatGPT 用户日志抓取的对话集合; Alpaca 是由 OpenAI text-davinci-003 生成的指令数据集. ShareGPT 的平均输入和输出长度为 80 和 296 token, Alpaca 为 12 和 56.

**工作负载.** 完整推理服务的周期级模拟不可行, 因此我们合成系统级工作负载. 我们改变模型, 批次大小以及张量/流水并行组合, 从数据集随机采样序列长度, 让批次包含不同长度的请求. 每种组合采样 10 个批次并测量吞吐量.

<span id="section-8-2"></span>

### 8.2 结果

**吞吐量.** [图 12](#figure-12) 比较了各基线与 NeuPIMs. 仅 GPU 与仅 NPU 差异很小, 因为两者都执行包含带宽受限 MHA 的完整解码器块. NPU+PIM 将 MHA GEMV 卸载到 PIM, 平均比仅 NPU 提升 1.5 倍. NeuPIMs 在所有模型和数据集上都超过 NPU+PIM, 额外提升 13% 到 3 倍. ShareGPT 的序列更长, 因而收益更大. 批次从 64 增加到 512 时, NeuPIMs 将瓶颈转移到 NPU 计算, 吞吐量收益继续增长.

<span id="table-04"></span>

![表 4. 平均资源利用率.](./neupims/table-04.png)

**表 4.** NPU/PIM 计算资源和内存带宽的平均利用率.

**利用率.** [表 4](#table-04) 比较 GPT3-30B, 批次 256 和 ShareGPT 的资源利用率. NPU+PIM 将 MHA 卸载到 PIM, 使 NPU 利用率达到 28.0%, 但仍受 GEMM-GEMV 依赖造成的时间阻塞影响. NeuPIMs 通过并发执行将 NPU 和 PIM 利用率提高到 64.9% 和 26.4%.

<span id="figure-13"></span>

![图 13. 批次敏感性和消融实验.](./neupims/figure-13.png)

**图 13.** GPT3-7B 和 ShareGPT 实验. DRB 为双行缓冲区, GMLBP 为贪心最小负载装箱, SBI 为子批次交错.

**消融实验.** 从 NPU+PIM 开始逐步加入 3 项技术. 双行缓冲区平均带来 69.7% 的吞吐量提升, 影响最大. 贪心装箱均衡分布请求, 始终带来收益. 子批次交错在小批次上可能造成脉动阵列利用率不足, 流水开销会超过收益; 批次达到 256 或更大时, NeuPIMs 获得最高吞吐量.

<span id="figure-14"></span>

![图 14. 张量并行和流水并行下的吞吐量.](./neupims/figure-14.png)

**图 14.** 多 NeuPIMs 系统在不同并行方案下的吞吐量.

**并行方案的影响.** LLM 规模增大时, NeuPIMs 需要扩展设备数量. [图 14](#figure-14) 固定总请求数为 256, 让每个设备的批次大小随并行方案变化. 张量并行保持更大的批次, NPU 效率更好, 因而优于流水并行. 当每设备批次过小时, NPU 利用率下降, 总吞吐量也下降.

**面积开销.** NeuPIMs 的主要面积开销来自双行缓冲区. 我们使用 22 nm CACTI 7.0, 将行缓冲区资源加倍, 测得面积开销为 3.11%.

<span id="table-05"></span>

![表 5. NeuPIMs 功耗开销.](./neupims/table-05.png)

**表 5.** NeuPIMs 功耗开销.

**功耗开销.** NeuPIMs 同时运行 NPU 和 PIM, 内存功耗高于仅 NPU. 我们使用 DRAMsim3 提供的 Micron DRAM 功耗模型测量功耗, 假设全 bank 计算命令功耗是读命令的 4 倍, 额外行缓冲区还会产生后台功耗. NeuPIMs 功耗高 1.8 倍, 但速度提升 2.4 倍, 对应节省 25% 能耗.

<span id="figure-15"></span>

![图 15. NeuPIMs 相对 TransPIM 的加速比.](./neupims/figure-15.png)

**图 15.** NeuPIMs 相对 TransPIM [Tra22] 的加速比.

**与 TransPIM 比较.** TransPIM 是在 PIM 中执行全部 Transformer 算子的独立 PIM 方案. 我们基于 DRAMsim3 构建模拟器并对齐 HBM 时序和容量. [图 15](#figure-15) 表明 NeuPIMs 的平均吞吐量高 228 倍, 加速范围为 79 倍到 431 倍. 差距来自 NPU 对 GEMM 的高效执行; TransPIM 针对单批次编码器推理, 不适合批量解码器 LLM 推理.

<span id="section-9"></span>

## 9 讨论

**模型训练.** 训练使用固定长度的输入和输出序列, 计算主要是 GEMM 而不是 GEMV. PIM 面向带宽受限 GEMV, 处理 GEMM 的效率较低, 因此 NeuPIMs 可用于训练但效率有限.

**与生产软件栈集成.** NeuPIMs 编译器提供类似 ONNX, PyTorch 和 JAX 的接口. 集成现有软件栈需要将这些模型表示转换为 NeuPIMs 规格的翻译器. 其他部分无需改变, 因为系统已经包含推理调度器, NPU 和 PIM 算子编译器以及执行运行时.

<span id="section-10"></span>

## 10 相关工作

**LLM 推理服务.** 现有服务系统通过减少内存占用, 优化内核执行和选择算子划分来提升推理性能. 本文针对适合计算和 I/O 的 NPU 与 PIM, 并加入调度策略. 这些系统可以继续使用选择性批处理和 KV cache 等优化, 但仅 GPU 优化无法完全消除 GEMV 的带宽瓶颈.

**面向语言模型的 PIM.** TransPIM [Tra22] 通过定制数据流加速 Transformer, 但面向编码器注意力和单请求推理, 不适合基于解码器的批量 LLM. AttAcc [Att24] 使用 PIM 减少 KV 矩阵移动, NeuPIMs 则把 PIM 加速器和端到端调度结合起来. 其他 PIM 工作针对 GEMV [New20, Har21], 但没有支持 PIM 与 NPU 并发.

**深度学习的异构流水加速.** 既有工作提出机器学习流水加速器, 另一些工作针对特定模型, 但都没有用 PIM 缓解 LLM 带宽需求或解决解码器块中 GEMV 与 GEMM 的利用率问题.

<span id="section-11"></span>

## 11 结论

LLM 推理需要可规模部署的专用资源, 同时面临内存容量, 计算强度和带宽约束. 我们提出 NeuPIMs, 将通用 ML 加速器 NPU 与 PIM 集成, 并设计针对 Transformer 数据流的调度和执行策略. 相比朴素 NPU+PIM 基线, NeuPIMs 将吞吐量提升 1.6 倍.

## 致谢

感谢 shepherd Vidushi Goyal 和匿名审稿人的意见. 本研究由韩国政府 (MSIT) 资助的 IITP 项目 (No. 2022-0-01037, No. 2018-0-00503, IITP-2024-2020-0-01795) 以及 KAIST 人工智能研究生院项目 (No. 2019-0-00075) 支持.
