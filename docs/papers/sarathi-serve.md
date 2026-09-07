---
title: 'Sarathi-Serve'
createTime: 2026/09/07 19:29:55
permalink: /papers/sarathi-serve/
---

> [Amey Agrawal](https://ameya.info/) [+author-note], [Nitin Kedia](https://kedianitin.com/), [Ashish Panwar](https://apanwariisc.github.io/), [Jayashree Mohan](https://www.microsoft.com/en-us/research/people/jamohan/), [Nipun Kwatra](https://www.microsoft.com/en-us/research/people/nkwatra/), [Bhargav S. Gulavani](https://x.com/bhargavgulavani), [Alexey Tumanov](https://faculty.cc.gatech.edu/~atumanov/) 和 [Ramachandran Ramjee](https://x.com/ramaramjee). 论文于 2024 年 3 月 4 日首次提交至 arXiv, 当前版本为 v3. 2024 年 7 月发表于第 18 届 USENIX 操作系统设计与实现研讨会 (OSDI 24), 第 117-134 页. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). <a href="/paper/sarathi-serve.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2403.02310). [TeX 源文件](https://export.arxiv.org/e-print/2403.02310). 精确的印刷排版和参考文献以原始 PDF 为准.

## 摘要

每个 LLM 服务请求都会经历两个阶段. 第一个阶段是 *prefill*, 它处理整个输入提示并生成第一个输出 token; 第二个阶段是 *decode*, 它逐个生成其余输出 token. Prefill 迭代延迟较高, 但输入提示可并行处理, 因而能够让 GPU 计算单元饱和. 相比之下, decode 迭代延迟较低, 计算利用率也较低, 因为一次 decode 迭代对每个请求只处理一个 token. 因此, 批处理对 decode 很有效, 也能提高整体吞吐量. 可是, 将多个请求组成批次会让 prefill 和 decode 迭代交错执行, 难以同时获得高吞吐量和低延迟.

我们提出高效的 LLM 推理调度器 Sarathi-Serve, 用来处理这种吞吐量与延迟之间的权衡. Sarathi-Serve 引入 *chunked-prefills*, 将一个 prefill 请求切成大小近似相等的块; 它还构造 *stall-free* 调度, 在不暂停已有 decode 的前提下向批次加入新请求. Stall-free 调度可以用大批次提高吞吐量, 同时尽量减小批处理对延迟的影响. 此外, Sarathi-Serve 中各批次的计算量较为一致, 缓解了迭代之间的不平衡, 因而只产生很少的流水线气泡.

在尾延迟约束下, 这些技术在不同模型和硬件上都显著改善了推理性能. 对单张 A100 GPU 上的 Mistral-7B, 服务容量提高了 $2.6\times$; 对两张 A100 GPU 上的 Yi-34B, 相比 vLLM 最多提高 $3.7\times$. 在 Falcon-180B 上配合流水线并行时, Sarathi-Serve 可将端到端服务容量最多提高 $5.6\times$. Sarathi-Serve 的源代码位于 [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) [Wei22d, Bro20, Cho23a, Ope24a, Kap20] 已在自然语言处理、问答和代码生成等多种任务上展现出很强的能力. 这使它们在聊天机器人 [Ope24a, Cha22a, Ant23, Cha22]、搜索 [Bin23, Kom22, You21, Per22, Bar23] 和代码助手 [Git21a, Rep22, Ama22] 等应用中的用量大幅增加. 大模型推理需要大量 GPU 计算, 同时其使用量显著增长, 因而 LLM 推理已经成为主要的 GPU 工作负载. 于是, 许多近期系统都把优化 LLM 推理作为研究重点 [Pop22, She23, Yu22a, Kwo23, Pat23, Zho24, Agr23].

<span id="figure-01"></span>

![Yi-34B 在两张 A100 GPU 上为 arxiv-summarisation 轨迹中的 128 个请求提供服务](./sarathi-serve/figure-01.png)

**图 1.** Yi-34B 在两张 A100 GPU 上为 *arxiv-summarisation* 轨迹中的 128 个请求提供服务. [图 1(a)](#figure-01) 标出 vLLM 中一次持续数秒的生成停顿, 这是许多同类停顿之一 [Kwo23]. [图 1(b)](#figure-01) 展示负载增加对尾延迟的影响. Sarathi-Serve 在提高吞吐量的同时消除了生成停顿.

吞吐量和延迟都是 LLM 推理的重要目标: 前者有助于把服务成本控制在可接受范围, 后者则是满足应用要求的必要条件. 本文说明, 当前 LLM 服务系统必须在吞吐量和延迟之间取舍. 批处理可以显著提高 LLM 推理吞吐量, 但现有系统组织多个请求的批处理方式会牺牲吞吐量或延迟中的一项. 例如, [图 1(b)](#figure-01) 显示, 在先进的 LLM 服务系统 vLLM 中, 增加负载会显著提高尾延迟 [Kwo23].

每个 LLM 推理请求都先后经历 *prefill* 和 *decode* 两个阶段. *Prefill* 阶段处理输入提示, *decode* 阶段进行自回归 token 生成. Prefill 并行处理输入提示的全部 token, 因而受计算能力限制; decode 每次只处理每个请求的一个 token, 因而受内存带宽限制. 所以, 较大的批次可以更有效地使用 GPU, decode 能从批处理中明显受益, 而 prefill 几乎不能受益.

按照请求批处理时安排 prefill 和 decode 阶段的方式, 当前 LLM 推理调度器大致可分为两类 [+1]: *prefill-prioritizing* 和 *decode-prioritizing*. 本文认为, 两种策略都有根本缺陷, 不适合在线推理服务 (见[图 2](#figure-02)).

FasterTransformer [Fas21] 等传统请求级批处理系统采用 *decode-prioritizing* 调度. 这类系统向执行引擎提交一批请求, 先计算所有请求的 prefill 阶段, 再调度它们的 decode 阶段. 只有批次内所有请求完成 decode 后, 该批次才结束; 换言之, 只要仍有请求正在 decode, 系统就不会调度新的 prefill. 这项策略针对 token 间时间 (TBT) 这一重要的 LLM 延迟指标优化推理. 原因是新请求不会影响正在进行的 decode. 可是, *decode-prioritizing* 调度器会严重损失吞吐量: 即使批次中的部分请求提前结束, 系统仍以缩小后的批次继续执行, 直到最后一个请求完成.

<span id="figure-02"></span>

![当前 LLM 服务系统的调度策略在吞吐量与延迟之间形成权衡](./sarathi-serve/figure-02.png)

**图 2.** 当前 LLM 服务系统会因调度策略不同而在吞吐量与延迟之间取舍. 优先执行 prefill 可优化吞吐量, 但会牺牲 TBT (token 间时间) 尾延迟; 优先执行 decode 则相反. Sarathi-Serve 通过 stall-free batching 兼顾高吞吐量和低 TBT 延迟. (本图仅作示意, 实际数值取决于模型和工作负载特征.)

Orca [Yu22a] 引入了迭代级批处理, 请求可以按单次迭代的粒度动态进入或离开批次. 迭代级批处理避免了请求级批处理系统的低效, 因而提高吞吐量. Orca 和 vLLM [Vll23] 等近期系统把迭代级批处理与 *prefill-prioritizing* 调度结合起来, 一有 GPU 内存可用便立即调度一个或多个请求的 prefill 阶段. 由于先计算 prefill 会让随后的 decode 使用较大的批次, *prefill-prioritizing* 调度器能获得更高吞吐量. 可是, 优先处理 prefill 会干扰正在执行的 decode, 导致高延迟. Prefill 的耗时会随提示长度任意增长, 因此 *prefill-prioritizing* 调度器会产生本文所称的 *generation stall*. 例如, [图 1(a)](#figure-01) 显示 vLLM 的一次生成停顿可能持续数秒.

Orca [Yu22a] 这类传统迭代级调度系统还会引入流水线停顿或气泡 [Hua19]. 跨多个节点扩展 LLM 推理时需要部署流水线并行 (PP), 气泡就会出现在这种部署中. 在 NVIDIA DGX A100 [Nvi16a] 这类具有高带宽互连的服务器中, 张量并行 (TP) [Sho19] 可以把一个 LLM 部署到最多 8 张 GPU 上, 以低延迟支持大批次. 但没有超大规模集群时, TP 的延迟可能高得无法接受 [Ath22]. 因此, 系统通常在普通网络上使用流水线并行 (PP) [Pip19, Ath22] 代替 TP. 现有系统依靠微批次缓解流水线停顿或气泡 [Hua19]. 不过, LLM 推理有自身特点, 标准的微批次调度仍会产生流水线气泡. 具体来说, LLM 推理由长度各异的 prefill 和 decode 混合组成, 不同微批次的运行时间可能相差很大, 从而浪费 GPU 周期并降低系统总吞吐量.

为处理这些问题, 我们提出 Sarathi-Serve, 一种为可扩展在线 LLM 推理服务平衡吞吐量与延迟的调度器. Sarathi-Serve 基于两个思路: *chunked-prefills* 和 *stall-free* 调度. *Chunked-prefills* 把一个 prefill 请求切分为计算量相同的块, 并在多次迭代中完成提示的 prefill 阶段 (每次只处理一部分提示 token). *Stall-free* 调度允许*新请求加入正在运行的批次, 而无需暂停已有 decode*. 具体做法是把所有正在进行的 decode 与新请求的一个或多个 prefill 块合并, 使每个批次达到预设的块大小. Sarathi-Serve 建立在迭代级批处理之上, 但有一处重要区别: 它在向运行中的批次接纳新请求时, 会限制每次迭代中的 prefill token 数. 这样既限制了单次迭代的延迟, 又使该延迟基本不受输入提示总长度影响. 因此, Sarathi-Serve 尽量减小新 prefill 对已有 decode 的 TBT 影响, 同时获得高吞吐量和低 TBT 延迟.

Sarathi-Serve 构造的混合批次 (包含 prefill 和 decode token) 还具有近似一致的计算量. 配合流水线并行, 这种特性可以构造计算均衡的微批次调度, 显著减少流水线气泡并提高 GPU 利用率, 支持高效且可扩展的部署.

我们在不同模型和硬件上评估 Sarathi-Serve: 单张 A100 上的 Mistral-7B、两张 A100 上采用 2 路张量并行的 Yi-34B、八张 A40 上的 LLaMA2-70B, 以及在通过普通以太网互连的八张 A100 上采用 2 路流水线并行和 4 路张量并行的 Falcon-180B. 对 Yi-34B, Sarathi-Serve 在不同 SLO 目标下最多把系统服务容量提高 $3.7\times$. 对 Mistral-7B, 服务容量最多提高 $2.6\times$. Sarathi-Serve 还减少了流水线气泡, 对采用流水线并行部署的 Falcon-180B, 端到端服务容量最多提高 $5.6\times$.

本文的主要贡献如下:

- 找出当前 LLM 服务系统的多项缺陷, 尤其是它们在处理吞吐量与延迟权衡时的问题.
- 提出 *chunked-prefills* 和 *stall-free batching* 两项简单而有效的技术, 改善 LLM 服务系统的性能.
- 在多种模型、硬件和并行策略上进行全面评估, 说明 Sarathi-Serve 具有通用性, 可将模型服务容量最多提高一个数量级.

<span id="section-2"></span>

## 2 背景

本节介绍典型的 LLM 模型架构及其自回归推理过程, 并概述调度策略和重要的性能指标.

<span id="section-2-1"></span>

### 2.1 Transformer 架构

GPT-3 [Ope22a]、LLaMA [Tou23c] 和 Yi [Yi23] 等常见大语言模型都是仅含解码器的 Transformer 模型, 通过下一 token 预测任务训练. 这些模型由结构相同的多层堆叠而成. 每层包含自注意力和前馈网络 (FFN) 两个模块.

**自注意力模块.** 自注意力模块是 Transformer 架构的核心 [Vas17], 序列的每个部分都可以参考之前的所有部分来生成上下文表示. 计算自注意力时, 系统先通过线性变换得到每个输入 token 对应的 Query ($Q$)、Key ($K$) 和 Value ($V$) 向量. 随后, *attention* 算子计算序列中所有 token 之间的语义关系. 它先将每个 $Q$ 向量与序列中所有前序 token 的 $K$ 向量求点积, 再通过 softmax 操作得到权重向量, 最后用该向量计算 $V$ 向量的加权平均. 注意力计算可以分成多个 *head*, 各头的输出再通过线性变换合并.

**前馈网络 (FFN).** FFN 通常包含两次线性变换, 中间夹有非线性激活. 第一个线性层把维度为 $h$ 的输入 token 嵌入变换到更高维度 $h2$. 随后使用激活函数, 通常是 ReLU 或 GELU [Aga19, Hen23]. 最后, 第二个线性层把 token 嵌入变换回原始维度 $h$.

<span id="section-2-2"></span>

### 2.2 LLM 推理过程

**自回归解码.** LLM 推理由 *prefill* 和随后的 *decode* 两个阶段组成. Prefill 阶段处理用户输入提示并生成第一个输出 token. 随后, decode 阶段逐个生成输出 token: 上一步生成的 token 会送入模型, 用来生成下一个 token, 直到生成特殊的 *end-of-sequence* token. 执行注意力操作时, decode 阶段需要访问此前处理过的所有 token 所对应的 key 和 value. 为避免反复计算, 当代 LLM 推理系统会把激活值保存在 KV-cache 中 [Sho19, Yu22a, Fas21].

典型的 LLM 提示包含数百到数千个输入 token [表 2](#table-02), [Zhe23c]. Prefill 阶段在一次迭代中并行处理所有提示 token, 因而能有效利用 GPU 计算单元. Decode 阶段则对上一次迭代生成的一个 token 执行模型的完整前向传播. 这会使计算利用率偏低, decode 因而受内存带宽限制.

**多租户环境中的批量 LLM 推理.** 生产服务系统必须处理来自多个用户的并发请求. 依次处理请求会严重浪费 GPU 计算能力. 为提高 GPU 利用率, LLM 服务系统通过批处理并发执行多个请求. 在小批次下计算强度较低的 decode 阶段, 这种做法尤其有效. 批次越大, 获取模型参数的成本就能分摊到越多请求上.

近期还有多项互补技术通过支持更大的批次来优化吞吐量. Kwon 等人提出 PagedAttention [Kwo23], 消除 *KV-cache* 碎片, 让更多请求能够并发执行. Multi Query Attention (MQA) [Sha19b] 和 Group Query Attention (GQA) [Ain23a] 用于 LLaMA2 [Tou23c]、Falcon [Alm23] 和 Yi [Yi23] 等前沿 LLM, 也明显缓解了 LLM 推理的内存瓶颈. 例如, LLaMA2-70B 的 KV-cache 占用比 LLaMA-65B 小 $8\times$.

<span id="section-2-3"></span>

### 2.3 多 GPU LLM 推理

随着模型规模不断增长, LLM 必须扩展到多 GPU 甚至多节点部署 [Pop22, Usi23]. 此外, LLM 推理吞吐量, 尤其是 decode 阶段的吞吐量, 受单张 GPU 能够容纳的最大批次大小限制. 模型并行把模型权重切分到多张 GPU, 允许使用更大的批次, 因而能提高推理效率. 既有工作采用张量并行 (TP) [Sho19] 和流水线并行 (PP) [Yu22a, Fas21, Wu23a] 实现这一点.

TP 通过在各参与 GPU 之间切分模型权重和 KV-cache 来拆分每一层. 因而, TP 能够线性扩展每张 GPU 上的批次大小. 不过, 每层都要执行两次 all-reduce, 一次用于注意力计算, 另一次用于 FFN [Sho19], 通信成本很高. 这些通信操作位于关键路径上, 因此 TP 通常只在单个节点内使用, 其中的 GPU 通过 NVLink 等高带宽互连相连.

PP 按层拆分模型, 每张 GPU 负责其中一部分层. 为使流水线中的所有 GPU 保持忙碌, 系统采用*微批处理*. 每次迭代时, 微批次会从一个流水线阶段移到下一阶段. PP 的计算通信比远高于 TP, 因为经过多层计算后只需发送一次激活值. 此外, PP 只需点对点通信, TP 则需要成本更高的 all-reduce. 所以在跨节点部署等缺少高带宽互连的环境中, PP 比 TP 更高效.

<span id="section-2-4"></span>

### 2.4 性能指标

LLM 服务主要关注两个延迟指标: TTFT (首 token 时间) 和 TBT (token 间时间). 对某个请求而言, TTFT 衡量请求进入系统到生成第一个输出 token 的延迟, 反映模型的初始响应速度. TBT 衡量一个请求连续生成两个输出 token 之间的间隔, 会影响用户感受到的响应流畅度. 系统承受负载时, 低吞吐量会造成很长的调度延迟, 继而提高 TTFT.

我们还使用吞吐量指标 *Capacity*, 即系统在满足给定延迟目标的前提下能够承受的最大请求负载 (每秒查询数). 容量越高, 服务成本越低.

<span id="section-2-5"></span>

### 2.5 LLM 推理调度策略

调度器负责准入控制和批处理策略. 为便于说明, 我们把现有 LLM 推理调度器大致分为 *prefill-prioritizing* 和 *decode-prioritizing* 两类.

FasterTransformer [Fas21] 和 Triton Inference Server [Tri20] 等传统推理引擎采用带请求级批处理的 *decode-prioritizing* 调度, 即选取一批请求并执行, 直到批内*所有*请求完成 ([算法 1](#algorithm-01)). 这种方法降低了调度框架的运行复杂度, 代价是资源利用率低. 一个批次内不同请求的输入和输出 token 数通常相差很大. 请求级调度器会用零填充较短的请求, 使其长度与批内最长请求一致, 造成无效计算, 也让待处理请求等待更久 [Yu22a].

<span id="algorithm-01"></span>

**算法 1: 请求级批处理. 仅当没有剩余 decode 时才接纳新请求 (第 3 行). 这会优化 TBT, 但许多仅含 decode 的迭代 (第 10 行) 批次可能很小, 从而浪费 GPU 计算.**

- 初始化当前批次 $B \leftarrow \emptyset$.
- **当** True:
  - **如果** $B = \emptyset$:
    - $R_{new} \leftarrow$ `get_next_request()`.
    - **当** `can_allocate_request`$(R_{new})$:
      - $B \leftarrow B + R_{new}$.
      - $R_{new} \leftarrow$ `get_next_request()`.
    - `prefill`$(B)$.
  - **否则:**
    - `decode`$(B)$.
    - $B \leftarrow$ `filter_finished_requests`$(B)$.

为避免请求级批处理浪费计算, Orca [Yu22a] 引入了细粒度的迭代级批处理机制, 请求可以在每次模型迭代后动态进入或离开批次 ([算法 2](#algorithm-02)). 这种方法可以显著提高系统吞吐量, 如今已用于许多 LLM 推理服务系统, 包括 vLLM [Vll23]、TensorRT-LLM [Ten23] 和 LightLLM [Lig23c].

<span id="algorithm-02"></span>

**算法 2: 迭代级批处理 (vLLM). Prefill 会被立即执行 (第 8-9 行), 可能让已有 decode 发生生成停顿 (第 12 行).**

- 初始化当前批次 $B \leftarrow \emptyset$.
- **当** True:
  - $B_{new} \leftarrow \emptyset$.
  - $R_{new} \leftarrow$ `get_next_request()`.
  - **当** `can_allocate_request`$(R_{new})$:
    - $B_{new} \leftarrow B_{new} + R_{new}$.
    - $R_{new} \leftarrow$ `get_next_request()`.
  - **如果** $B_{new} \neq \emptyset$:
    - `prefill`$(B_{new})$.
    - $B \leftarrow B + B_{new}$.
  - **否则:**
    - `decode`$(B)$.
  - $B \leftarrow$ `filter_finished_requests`$(B)$.

vLLM [Vll23] 和 Orca [Yu22a] 等当前迭代级批处理系统使用 *prefill-prioritizing* 调度, 一有机会就立即把新请求接纳到运行中的批次, 例如 GPU 内存可用时. 优先执行 prefill 可以增大随后 decode 迭代的批次, 因而提高吞吐量.

<span id="section-3"></span>

## 3 动机

<span id="figure-03"></span>

![Mistral-7B 在单张 A100 GPU 上以不同批次大小运行时 prefill 与 decode 阶段的吞吐量](./sarathi-serve/figure-03.png)

**图 3.** Mistral-7B 在单张 A100 GPU 上以不同批次大小运行时 prefill 与 decode 阶段的吞吐量. Prefill 和 decode 实验都使用长度为 1024 的提示. 两幅图的 y 轴不同, 可见 prefill 比 decode 高效得多. 还可以看到, *批处理会近似线性地提高 decode 吞吐量, 对 prefill 吞吐量却几乎没有影响.*

本节先分析 prefill 和 decode 操作的成本, 再说明 LLM 服务中的吞吐量-延迟权衡和流水线气泡.

<span id="section-3-1"></span>

### 3.1 Prefill 和 decode 的成本分析

如[第 2.2 节](#section-2-2) 所述, *prefill* 阶段并行处理所有输入 token, 能够让 GPU 计算单元饱和; *decode* 阶段一次只处理一个 token, 效率很低. [图 3](#figure-03) 展示吞吐量随批次大小的变化. Decode 迭代的吞吐量近似随批次大小线性增长, prefill 吞吐量却在只有一个请求时就几乎饱和.

**结论 1.** *LLM 推理的 prefill 与 decode 两个阶段表现不同: 批处理可以大幅提高 decode 阶段的吞吐量, 对 prefill 吞吐量则影响很小.*

<span id="figure-04"></span>

![Mistral-7B 在单张 A100 GPU 上使用不同输入大小时的 prefill 与 decode 时间](./sarathi-serve/figure-04.png)

**图 4.** Mistral-7B 在单张 A100 GPU 上使用不同输入大小时的 prefill 与 decode 时间. 线性层占 prefill 和 decode 两个阶段的大部分运行时间. Decode 批次的算术强度较低, 因此 1 个 decode token 的线性操作成本几乎等于 128 个 prefill token.

[图 4](#figure-04) 把 prefill 和 decode 的计算时间分解为 linear、attention 和 others, 并展示各部分所占比例. 线性算子贡献了大部分运行成本. 虽然注意力成本随序列长度呈二次增长, 即使在序列很长时, 线性算子仍占总时间的 80% 以上. 因此, 优化线性算子对于改善 LLM 推理很重要.

**Decode 期间的计算利用率偏低.** Decode 阶段的计算利用率低, 浪费了 GPU 的处理能力. 为进一步理解这一点, 我们分析 prefill 和 decode 迭代的算术强度. LLM 推理的大部分时间花在线性算子上, 因此分析也集中于这些算子.

矩阵乘法内核会让内存访问与数学运算重叠执行. 一个操作的总执行时间可近似为 $T=\max(T_{\text{math}},T_{\text{mem}})$, 其中 $T_{\text{math}}$ 和 $T_{\text{mem}}$ 分别表示数学运算与内存读取所用时间. 若 $T_{\text{math}}<T_{\text{mem}}$, 该操作受内存限制. 受内存限制的操作具有较低的模型 FLOPs 利用率 (MFU) [Cho23a]. 受计算限制的操作则具有较低的模型带宽利用率 (MBU). 当 $T_{\text{math}}=T_{\text{mem}}$ 时, 计算与内存带宽利用率同时达到最大值. 算术强度表示每读取一字节数据所执行的数学运算数. 在最佳点, 操作的算术强度与设备的 FLOPS-带宽比相同. [图 5](#figure-05) 展示在四张 A100 GPU 上运行 LLaMA2-70B 时, 线性层的算术强度如何随批次 token 数变化. Prefill 批次可把从 HBM 向 GPU 缓存读取线性算子权重的成本分摊到大量 token 上, 因而具有较高算术强度. Decode 批次的计算强度则很低. [图 6](#figure-06) 展示 LLaMA2-70B 一次迭代中线性算子的总执行时间随 token 数的变化. 开始阶段, 即批次仍处于内存受限区域时, 执行时间只略有增加; 之后, 即批次转为计算受限时, 执行时间会线性增长. [+2]

<span id="figure-05"></span>

![LLaMA2-70B 在四张 A100 上运行时线性操作的算术强度趋势](./sarathi-serve/figure-05.png)

**图 5.** LLaMA2-70B 在四张 A100 上以不同 token 数运行线性操作时的算术强度趋势. Decode 批次的算术强度较低, 即瓶颈是内存读取时间, 计算利用率因而偏低. Prefill 批次受计算限制, 带宽利用率较低. Sarathi-Serve 把 decode 和 prefill 块合并成均衡批次, 同时提高计算与带宽利用率.

**结论 2.** *Decode 批次处于内存受限区域, 计算单元没有得到充分利用. 因此, 可以在不显著增加延迟的情况下, 随 decode 批次一起处理更多 token.*

<span id="figure-06"></span>

![不同张量并行度下 LLaMA2-70B 线性层的执行时间随批次 token 数变化](./sarathi-serve/figure-06.png)

**图 6.** 不同张量并行度下, LLaMA2-70B 在 A100 上运行时, 线性层执行时间随批次 token 数的变化. Token 数较少时, 执行时间取决于从 HBM 读取权重的成本. 因此, 在 128-512 个 token 范围内, 执行时间基本不变, 张量并行度越高时越明显. 批内 token 数超过临界值后, 操作转为计算受限, 运行时间随 token 数线性增长.

<span id="figure-07"></span>

![不同调度策略中的生成停顿](./sarathi-serve/figure-07.png)

**图 7.** 在一个请求的连续两次 decode 迭代之间调度一个或多个 prefill 时, 会发生生成停顿. A、B、C 和 D 表示不同请求. 下标 $d$ 表示一次 decode 迭代, $p$ 表示完整 prefill, $p0$ 和 $p1$ 表示同一提示的两个 prefill 块. vLLM 会先调度尽可能多的 prefill, 再恢复已有 decode, 因而引发生成停顿. Orca 虽然支持混合批次, 但含长提示的批次执行时间仍然很长, 无法缓解生成停顿. FasterTransformer 会在调度新 prefill 前完成所有已有 decode, 因而没有生成停顿, 但 decode 批次很小, 吞吐量会受损. Sarathi-Serve 生成的调度既消除生成停顿, 又提供高吞吐量.

<span id="section-3-2"></span>

### 3.2 吞吐量-延迟权衡

迭代级批处理提高了系统吞吐量, 但本文说明, 它会因一种称为*生成停顿*的现象而付出高 TBT 延迟的代价.

[图 7](#figure-07) 比较了不同调度策略. 示例按从左到右的方向展示请求 A、B、C 和 D 的时间线. 区间开始时, 请求 A 和 B 处于 decode 阶段; 一次迭代后, 请求 C 和 D 进入系统. Orca 与 vLLM 都采用 FCFS 迭代级批处理, 立即接纳 prefill 请求, 但批次组成策略不同. Orca 支持由 prefill 和 decode 请求共同组成的混合批次, vLLM 的批次则只能全部是 prefill 或全部是 decode. 无论采用哪种组成方式, Orca 和 vLLM 都能在随后的 decode 迭代中通过扩大批次提高吞吐量. 但是, 立即调度请求 C 和 D 的 prefill 会推迟已有请求 A 和 B 的 decode, 因为计算一个或多个 prefill 的迭代会随输入提示长度增长到数秒. 所以, *prefill-prioritizing* 调度器可能让已有 decode 发生*生成停顿*, 造成 TBT 延迟尖峰.

FasterTransformer [Fas21] 等请求级批处理系统与迭代级批处理不同, 在已有请求*全部*完成 decode 阶段前不会调度新请求 ([算法 1](#algorithm-01) 第 3 行). 在[图 7](#figure-07) 中, 请求 C 和 D 的 prefill 会一直等待, 直到请求 A 和 B 都离开系统. 因此, *decode-prioritizing* 系统的 TBT 延迟较低, 代价是系统吞吐量偏低. 例如, Kwon 等人 [Kwo23] 证明, 采用 PagedAttention 的迭代级批处理可比 FasterTransformer 获得高一个数量级的吞吐量.

一种减少迭代级批处理延迟尖峰的方法是按照 Orca [Yu22a] 的建议使用较小批次. 然而, 如[第 2.2 节](#section-2-2) 所示, 缩小批次会降低吞吐量. 因而, 现有系统只能根据期望的 SLO 在吞吐量与延迟之间取舍.

**结论 3.** *对当前 LLM 推理调度器而言, prefill 与 decode 交错执行会引入吞吐量与延迟之间的权衡. 当今先进系统采用 prefill-prioritizing 调度, 用 TBT 延迟换取高吞吐量.*

<span id="figure-08"></span>

![Orca 与 Sarathi-Serve 的两路流水线并行调度](./sarathi-serve/figure-08.png)

**图 8.** Orca 在四个请求 (A、B、C、D) 上采用两路流水线并行迭代级调度时, 批次执行时间不一致, 因而存在流水线气泡. Sarathi-Serve 通过构造计算量一致的批次把这些停顿降到最低.

<span id="section-3-3"></span>

### 3.3 流水线气泡浪费 GPU 周期

流水线并行 (PP) 的通信开销低于张量并行 (TP), 因而是跨节点部署大模型的常用策略. 不过, 后续流水线阶段必须等待前一阶段完成相应微批次, PP 会产生*流水线气泡*, 即 GPU 空闲的时段. 流水线气泡是训练任务中的已知问题. 前面的阶段需要等待反向传播到达, 气泡会出现在前向传播与反向传播之间. 微批处理是 PP 训练任务中缓解流水线气泡的常用技术 [Ath22, Pip19, Hua19].

推理任务只需要前向计算, 因而人们可能认为微批处理可以消除推理中的流水线气泡. FasterTransformer [Fas21] 和 FastServe [Wu23a] 等 Transformer 推理工作确实使用微批次, 但没有提到流水线气泡. 近期提出的 Orca [Yu22a] 也认为迭代级调度能消除流水线调度中的气泡 (见 [Yu22a] 的[图&#32;8](https://arxiv.org/pdf/2206.02672#page=11)). 然而, 实验表明即使使用迭代级调度, 流水线气泡仍会在 PP 中浪费大量 GPU 周期 (见[第 5.3 节](#section-5-3)).

LLM 推理中的每个微批次 (或迭代) 都可能需要不同的计算量, 执行时间也会随微批次中 prefill 与 decode token 的组成而变化 (见[图 8](#figure-08)). 我们找出推理中的三类气泡: (1) $\mathrm{PB}_{1}$ 这类气泡来自连续两个微批次中 prefill token 数不同; (2) $\mathrm{PB}_{2}$ 这类气泡出现在 prefill 阶段与 decode 阶段先后执行且计算时间不同时; (3) $\mathrm{PB}_{3}$ 这类气泡来自不同微批次的 decode 计算时间差异, 因为注意力成本取决于累积上下文长度 (KV-cache 大小), 在不同请求间并不相同. 对 Falcon-180B, 一个含 4k token 的提示需要约 $\approx 1150$ ms, 而批次大小为 32 的仅 decode 迭代约需 $\approx 200$ ms. 两类迭代交错执行可能产生约 $\approx 950$ ms 的气泡. 流水线气泡浪费 GPU 周期, 会直接损失服务吞吐量并增加延迟. 提示越长, prefill 迭代就越久; 批次越大, prefill 迭代越频繁, 问题都会进一步加重. 如果能让每个微批次执行一致的计算量, 就可以缓解这些流水线气泡.

**结论 4.** *LLM 迭代的计算时间会随批次中 prefill 和 decode token 的组成产生很大差异. 使用流水线并行时, 这会造成显著的气泡.*

<span id="section-4"></span>

## 4 Sarathi-Serve: 设计与实现

下面介绍 Sarathi-Serve 的设计与实现. 该系统通过 *chunked-prefills* 和 *stall-free batching* 两项技术, 提供高吞吐量和可预测的尾延迟.

<span id="figure-09"></span>

![把 prefill 与 decode 批次合并后的增量成本](./sarathi-serve/figure-09.png)

**图 9.** 把 prefill 与 decode 批次合并后的增量成本. 我们考虑两种批处理方案: (i) Decode + Full Prefill 表示 Orca 的混合批处理, 整个 prefill 在一次迭代中与已有 decode 一起执行. (ii) Decode + Chunked Prefill 表示 Sarathi-Serve, prefill 先按固定 token 预算分块, 再与已有 decode 合并. Sarathi-Serve 处理 prefill token 时对 decode 延迟的影响要小得多. 随着 decode 批次和上下文长度增加, Sarathi-Serve 对延迟的相对影响还会降低.

<span id="section-4-1"></span>

### 4.1 Chunked-prefills

如[第 3.1 节](#section-3-1) 所示, decode 批次严重受内存限制, 算术强度较低. 算术强度中的余量使我们有机会在 decode 批次上附带额外计算. 最直接的做法是构造混合批次, 把受内存限制的 decode 与受计算限制的 prefill 合并. 不过, 实际场景中的输入提示平均常有数千个 token. 例如, [表 2](#table-02) 显示 *openchat_sharegpt4* 和 *arxiv_summarization* 数据集的提示长度中位数分别是 1730 和 7059. 把这些长 prefill 与 decode 迭代合并会造成很高的 TBT 延迟.

为处理这个问题, 我们提出 *chunked-prefills*, 在多次迭代中以小块计算大型 prefill. *Chunked-prefills* 是一种基于两个观察的 prefill 拆分机制. 首先, 如[第 3.1 节](#section-3-1) 所述, 序列长度适中的 prefill 请求就能让 GPU 计算单元饱和. 例如, [图 4](#figure-04) 中的 prefill 吞吐量在序列长度约为 512 个 token 时就开始饱和. 其次, 许多实际场景的输入提示平均包含数千个 token ([表 2](#table-02)). 因而, 大型 prefill 请求可以拆成更小的计算单元, 这些单元仍足以让 GPU 计算单元饱和. Sarathi-Serve 利用这项机制构造 token 数合适的批次, 在不违反 TBT SLO 的前提下使用 decode 批次中闲置的计算能力.

<span id="algorithm-03"></span>

**算法 3: Sarathi-Serve 的 stall-free batching. 首先用已有 decode token 填充批次 (第 6-8 行), 也可以加入已有请求的一个 prefill 块 (第 10-12 行). 最后在 token 预算内加入新请求 (第 13-20 行), 尽量提高吞吐量, 同时尽量减小推迟已有 decode 对 TBT 的影响.**

- **输入:** $T_{\max}$, 应用的 TBT SLO.
- 初始化 *token_budget*, $\tau \leftarrow$ `compute_token_buget`$(T_{\max})$.
- 初始化 *batch_num_tokens*, $n_t \leftarrow 0$.
- 初始化当前批次 $B \leftarrow \emptyset$.
- **当** True:
  - **对每个** $B$ 中的 $R$:
    - **如果** `is_prefill_complete`$(R)$:
      - $n_t \leftarrow n_t + 1$.
  - **对每个** $B$ 中的 $R$:
    - **如果** not `is_prefill_complete`$(R)$:
      - $c \leftarrow$ `get_next_chunk_size`$(R, \tau, n_t)$.
      - $n_t \leftarrow n_t + c$.
  - $R_{new} \leftarrow$ `get_next_request()`.
  - **当** `can_allocate_request`$(R_{new}) \land n_t < \tau$:
    - $c \leftarrow$ `get_next_chunk_size`$(R_{new}, \tau, n_t)$.
    - **如果** $c > 0$:
      - $n_t \leftarrow n_t + c$.
      - $B \leftarrow R_{new}$.
    - **否则:**
      - **跳出循环.**
  - `process_hybrid_batch`$(B)$.
  - $B \leftarrow$ `filter_finished_requests`$(B)$.
  - $n_t \leftarrow 0$.

<span id="section-4-2"></span>

### 4.2 Stall-free batching

Sarathi-Serve 调度器采用迭代级调度, 借助 *chunked-prefills* 以及 prefill 与 decode 的合并, 在尽量降低延迟的同时提升吞吐量.

Orca 和 vLLM 会暂停已有 decode 来执行 prefill, 而 Sarathi-Serve 利用 decode 迭代在算术强度上的余量执行 prefill, 不会延后系统中 decode 请求的执行. 我们把这种方法称为 *stall-free batching* ([算法 3](#algorithm-03)). Sarathi-Serve 首先根据用户指定的 SLO, 计算一个批次内最多可执行多少个 token. [第 4.3 节](#section-4-3) 会详细说明确定这一 token 预算时需要考虑的因素. 每轮调度时, 我们先把所有正在运行的 decode 装入下一个批次 ([算法 3](#algorithm-03) 第 6-8 行), 再加入尚未完成的 prefill (第 9-12 行). 只有容纳了所有正在运行的请求后, 才接纳新请求 (第 13-20 行). 将 prefill 请求加入批次时, 我们会计算该批次剩余 token 预算所能容纳的最大块大小 (第 11、15 行). *Stall-free batching* 限制每轮迭代的计算负载, 从而保证 decode 不会因并行执行的 prefill 块而遭遇生成停顿. [图 9](#figure-09) 比较了混合批次在采用和不采用 chunked prefill 时的延迟. 朴素混合批处理会让 TBT 延迟相对纯 decode 批次骤增至多 $28.3\times$, 而 Sarathi-Serve 通过分块给出了紧得多的延迟上界.

[图 7](#figure-07) 用[第 3.2 节](#section-3-2) 的同一个例子展示了 Sarathi-Serve 的实际调度过程. 第一轮迭代只包含 decode, 因为此时没有待计算的 prefill. 新请求 C 进入系统后, Sarathi-Serve 先把 C 的 prefill 拆成两块, 在之后的迭代中依次调度. 与此同时, *stall-free batching* 会把这些 prefill 块与 A、B 正在进行的 decode 合并. 这样一来, Sarathi-Serve 不会像现有系统那样暂停 decode 或 prefill, 因而能在不牺牲吞吐量的情况下基本消除 TBT 延迟尖峰. 此外, *stall-free batching* 与 *chunked-prefills* 结合后, 多数情况下还能形成计算量均匀的混合批次, 减少流水线并行中的气泡, 支持高效且可扩展的部署.

<span id="section-4-3"></span>

### 4.3 确定 token 预算

Token 预算取决于两个相互制约的因素: TBT SLO 要求与 *chunked-prefills* 的开销. 若只考虑降低 TBT, 较小的 token 预算更合适, 因为 prefill token 较少的迭代延迟更低. 但预算太小会把 prefill 切得过碎, 从两方面带来开销: 1) GPU 利用率降低; 2) 注意力运算反复访问 KV-cache. 下面具体说明第二点.

计算 *chunked-prefills* 时, 提示中每个块的注意力运算都要访问同一提示此前**所有**块的 KV-cache. 计算成本虽然不变, 从 GPU HBM 读取的数据量却会增加. 例如, 一个 prefill 序列若被拆成 $N$ 块, 第一块的 KV-cache 会被加载 $N-1$ 次, 第二块会被加载 $N-2$ 次, 依此类推. 不过我们发现, 即便块很小, prefill 的注意力运算仍受计算限制. 实际中, 内核启动等固定成本会让分块产生少量开销. [第 5.4 节](#section-5-4) 会详细分析 *chunked-prefills* 的开销.

因此, 确定 token 预算时必须权衡 prefill 开销与 decode 延迟. 可以一次性分析含不同 token 数的批次, 把预算设为不违反 TBT SLO 时一个批次可容纳的最大 token 数.

影响 token 预算选择的另一个因素是 *tile-quantization* 效应 [Mat23]. GPU 会把给定矩阵划分为多个 tile, 分配给不同线程块并行计算矩阵乘法. 每个线程块由一组 GPU 线程组成, 执行相同数量的算术运算. 因此, 当矩阵维度能被 tile 大小整除时, 矩阵乘法的 GPU 利用率最高; 否则, *tile-quantization* 会让某些线程块执行多余计算 [Mat23]. 我们观察到, tile-quantization 会显著延长 prefill 计算时间. 例如在某些情况下, 块大小为 257 时的 prefill 时间比块大小为 256 时高 32%.

采用流水线并行时, 还应考虑 token 预算对流水线气泡的影响. 较大的块会加剧批次间的运行时间差异, 形成流水线气泡并降低系统总吞吐量. 反过来, token 预算过小也会因算术强度较低和其他固定成本而增加开销.

所以, 合适的 token 预算取决于目标 TBT SLO、并行配置和具体硬件特性, 并不是一个简单的选择. 我们利用 LLM 推理分析与模拟工具 Vidur [Agr24], 针对具体部署场景找出能使系统容量最大的 token 预算.

<span id="section-4-4"></span>

### 4.4 实现

我们在 vLLM [Kwo23, Vll23] 的开源实现上构建了 Sarathi-Serve, 使用 FlashAttention v2 [Dao23a] 和 FlashInfer [Ye24a] 内核加入分页分块 prefill 支持. 本文所有评估均使用 FlashAttention 后端, 因为它支持的模型范围更广. 我们还扩展了 vLLM 基础代码, 支持多种调度策略、chunked prefill、流水线并行以及完备的遥测系统. 流水线并行和张量并行通信均使用 NCCL [Ncc15]. 项目源代码见 [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).

<span id="table-01"></span>

![表 1. 模型与 GPU 配置 (GQA: grouped-query attention, SW: sliding window).](./sarathi-serve/table-01.png)

**表 1.** 模型与 GPU 配置 (GQA: grouped-query attention, SW: sliding window).

<span id="section-5"></span>

## 5 评估

我们在多种常用模型和 GPU 配置 ([表 1](#table-01)) 以及两个数据集 ([表 2](#table-02)) 上评估 Sarathi-Serve. 基线选择 vLLM 和 Orca, 因为二者代表了当时 LLM 推理的先进水平. 评估试图回答以下问题:

- 在给定服务等级目标 (SLO) 约束时, 不同推理服务系统的单个模型副本最多能承载多少负载 ([第 5.1 节](#section-5-1))? 该负载会如何随 SLO 约束改变 ([第 5.2 节](#section-5-2))?
- Sarathi-Serve 在 TP、PP 等不同部署方式下表现如何 ([第 5.3 节](#section-5-3))?
- *Chunked-prefills* 的开销是多少 ([第 5.4.1 节](#section-5-4-1))?
- 单独使用 *chunked-prefills* 或 *stall-free batching* 与二者结合使用相比, 各自有何影响 ([第 5.4.2 节](#section-5-4-2))?

<span id="table-02"></span>

![表 2. 评估所用数据集.](./sarathi-serve/table-02.png)

**表 2.** 评估所用数据集.

<span id="table-03"></span>

![表 3. 不同模型配置的 SLO.](./sarathi-serve/table-03.png)

**表 3.** 不同模型配置的 SLO.

**模型与环境:** 我们评估四种模型: Mistral-7B [Jia23]、Yi-34B [Yi23]、LLaMA2-70B [Tou23c] 和 Falcon-180B [Alm23]. 它们都属于各自规模类别中表现最好的模型. 实验使用两种服务器配置. 除 LLaMA2-70B 外, 其余模型均使用 Azure NC96ads v4 VM, 每台配有 4 块通过成对 NVLINK 互联的 NVIDIA 80GB A100 GPU, 机器之间使用 100 Gbps 以太网连接. LLaMA2-70B 使用一台配有 8 块成对互联的 NVIDIA 48GB A40 GPU 的服务器. Yi-34B 采用二路张量并行 (TP-2); LLaMA2-70B 和 Falcon-180B 采用四个张量并行 worker、两个流水线阶段的混合并行配置 (TP4-PP2).

**工作负载:** 为模拟实际服务场景, 我们根据 *openchat_sharegpt4* [Wan23m] 和 *arxiv_summarization* [Coh18] 数据集的请求长度特征生成轨迹 ([表 2](#table-02)). *openchat_sharegpt4* 轨迹包含用户分享的 ChatGPT-4 [Cha22a] 对话. 一段对话可能有多轮用户与聊天机器人的交互, 每一轮都作为独立请求提交给系统. 这种多轮特性使提示长度的相对方差较大. 相比之下, *arxiv_summarization* 收集了 arXiv.org [Arx91] 上的科学论文及其摘要. 该数据集的提示更长, 输出 token 数方差更低, 能代表 Microsoft M365 Copilot [Cop23]、Google Duet AI [Due23] 等 LLM 工作负载. 请求到达时间按泊松分布生成. 我们移除总长度分别超过 8192 和 16384 个 token 的请求, 过滤两个数据集中的离群值.

**指标:** 每个用户请求只会得到一次 TTFT, 因而我们关注其中位数; 每个 decode token 都会产生一个 TBT 延迟值, 因而关注 TBT 的第 99 百分位数 (P99).

<span id="figure-10"></span>

![图 10. 在严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下, Mistral-7B 和 Yi-34B 使用不同调度器时的容量 (每秒查询数).](./sarathi-serve/figure-10.png)

**图 10.** 在严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下, Mistral-7B 和 Yi-34B 使用不同调度器时的容量 (每秒查询数).

<span id="section-5-1"></span>

### 5.1 容量评估

<span id="figure-11"></span>

![图 11. 在严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下, 使用不同调度器时 LLaMA2-70B 和 Falcon-180B (采用流水线并行的模型) 的容量.](./sarathi-serve/figure-11.png)

**图 11.** 在严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下, 使用不同调度器时 LLaMA2-70B 和 Falcon-180B (采用流水线并行的模型) 的容量.

我们在两种延迟配置下, 用全部四种模型和两个数据集评估 Sarathi-Serve、Orca 与 vLLM: **宽松**和**严格**. 与 Patel 等人 [Pat23] 类似, 为了纳入模型与硬件组合自身的性能限制, 我们把**严格**和**宽松**配置的 P99 TBT SLO 分别定义为某个请求在不受 prefill 干扰时运行一次 decode 迭代所需时间的 $5\times$ 和 $25\times$ (该请求 prefill 长度为 4k, 批大小为 32). [表 3](#table-03) 汇总了各 SLO 的绝对阈值. **严格** SLO 代表聊天机器人等交互式应用所需的延迟目标. **宽松**配置则代表另一类系统: 完整输出 token 序列必须在可预测的时限内生成, 但单个 token 的 TBT 约束没有那么严格. 在所有负载实验中, 我们都会保证最大负载可以持续承载, 即排队延迟不会失控 (调度延迟中位数上限设为 2 秒).

[图 10](#figure-10) 和[图 11](#figure-11) 给出了容量实验的结果. 在所有模型和工作负载组合中, Sarathi-Serve 始终优于 Orca 与 vLLM. 在**严格** SLO 下, Sarathi-Serve 可承载的负载至多比 Orca 高 $4.0\times$, 比 vLLM 高 $3.7\times$ (Yi-34B, *openchat_sharegpt4*). 对使用流水线并行的大模型, Sarathi-Serve 相比 Orca 和 vLLM 分别实现至多 $6.3\times$ 和 $4.3\times$ 的提升 (LLaMA2-70B, *openchat_sharegpt4*), 原因是流水线气泡更少.

我们观察到, 在多数场景中, Orca 和 vLLM 尚未达到最高可服务吞吐量, P99 TBT 延迟就已经违反 SLO. 因此, 放宽延迟目标会显著提高它们的模型服务容量. Sarathi-Serve 可以根据目标 SLO 调整块大小. 在**严格**延迟 SLO 下, 我们使用严格的 token 预算, 把提示拆成更小的块. 这会略微降低系统效率, 却能压低尾延迟. 延迟约束放宽时, 我们会增加 token 预算, 让 prefill 更高效. 除 LLaMA2-70B 的**宽松**配置使用 1536 以减轻流水线气泡外, 所有模型在**宽松**与**严格**配置下的 token 预算分别是 2048 和 512. 若能根据工作负载特征动态调整 token 预算, 系统性能还可进一步提高, 留待未来研究.

我们还发现, 宽松配置下 vLLM 明显优于 Orca, 原因有二. 第一, Orca 会把多个请求的提示合为一批 (*最大序列长度 × 批大小*, 而 vLLM 为*最大序列长度*), 某些情况下会让尾延迟更高. 第二, vLLM 支持的批大小远大于 Orca. Orca 的批大小较低, 一方面是没有 PagedAttention, 另一方面是处理 token 数过多的批次会占用大量激活内存.

最后, 每套系统在 *openchat_sharegpt4* 数据集上的容量都高于 *arxiv_summarization*. 这是预期结果, 因为[表 2](#table-02) 显示 *arxiv_summarization* 的提示要长得多, 中位数分别为 7059 与 1730 个 token. 较长提示的处理时间更久, Orca 和 vLLM 因而更容易违反延迟约束.

<span id="section-5-2"></span>

### 5.2 吞吐量与延迟的权衡

为完整理解 LLM 服务系统中吞吐量与延迟的权衡, 我们改变 P99 TBT 延迟 SLO, 观察 vLLM 和 Sarathi-Serve 的系统容量如何变化. [图 12](#figure-12) 给出了 Mistral-7B 与 Yi-34B 在 *openchat_sharegpt4* 数据集、五种不同 SLO 下的结果.

按照 Yu 等人 [Yu22a] 的建议, 我们用三种不同批大小评估 vLLM, 尝试在延迟与吞吐量之间取舍. 严格 TBT SLO 下, 生成停顿会限制 vLLM 的最大容量. 尤其值得注意的是, vLLM 在三种批大小设置下的容量基本相同. 这意味着 PagedAttention 虽然能通过高效内存管理支持大批次, 但面对实际的延迟约束时, vLLM 的 *prefill-prioritizing* 调度器要付出过于陡峭的延迟—吞吐量代价, 无法利用大批次.

Sarathi-Serve 则可以通过改变 token 预算, 精确控制延迟与吞吐量的权衡. 在严格 SLO 下 (100ms, Mistral-7B), Sarathi-Serve 使用 512 的小 token 预算, 容量比 vLLM 高 $3.5\times$. 在 SLO 较为宽松的场景中, 选择 2048 的较大 token 预算能让 Sarathi-Serve 更高效地运行, 容量比 vLLM 高 $1.65\times$ (1s, Yi-34B).

<span id="figure-12"></span>

![图 12. vLLM 与 Sarathi-Serve 在 *openchat_sharegpt4* 数据集上运行 Mistral-7B 和 Yi-34B 时的延迟—吞吐量权衡. vLLM 采用 32、64、128 三种最大批大小; Sarathi-Serve 的最大批大小为 128, token 预算取 512 和 2048. 借助 *stall-free batching*, Sarathi-Serve 在严格 SLO 下为 Yi-34B 提供了高 $3.5\times$ 的容量.](./sarathi-serve/figure-12.png)

**图 12.** vLLM 与 Sarathi-Serve 在 *openchat_sharegpt4* 数据集上运行 Mistral-7B 和 Yi-34B 时的延迟—吞吐量权衡. vLLM 采用 32、64、128 三种最大批大小; Sarathi-Serve 的最大批大小为 128, token 预算取 512 和 2048. 借助 *stall-free batching*, Sarathi-Serve 在严格 SLO 下为 Yi-34B 提供了高 $3.5\times$ 的容量.

<span id="section-5-3"></span>

### 5.3 让流水线并行切实可用

下面说明 Sarathi-Serve 如何利用高效的流水线并行, 让 LLM 推理可以跨普通网络高效服务. 实验在两个节点上运行 Falcon-180B, 每个节点有 4 块 A100 GPU, 节点间使用 100 Gbps 以太网连接. 我们在三种配置下评估模型容量: 采用八路 TP 的 vLLM、采用我们流水线并行实现的 vLLM, 以及采用流水线并行的 Sarathi-Serve. 两种 PP 配置均在节点内使用四路 TP, 节点间使用二路 PP.

[图 13(a)](#figure-13) 比较了 Falcon-180B 采用纯张量并行 TP-8 部署和 TP-4 PP-2 混合并行配置时, 纯 decode 批次的延迟. 张量并行的中位延迟约为流水线并行的 $2\times$, 原因是跨节点 all-reduce 给 TP 带来了很高的通信开销.

[图 13(b)](#figure-13) 给出了 Falcon-180B 在 *openchat_sharegpt4* 数据集上采用张量并行与混合并行时的容量. 与混合并行配置不同, TP 的延迟很高, 即便在**宽松** SLO 下容量也很低. vLLM 在**宽松** SLO 下使用混合并行虽然能支持相当高的负载, 但由于流水线气泡, 进入**严格**区间后性能会急剧下降. Sarathi-Serve 利用 *chunked-prefills* 减小微批次间执行时间的差异, 避免流水线气泡, 因而在**宽松** SLO 下把容量提高 $1.48\times$, 在**严格** SLO 下提高 $3.6\times$.

<span id="figure-13"></span>

![图 13. TP 跨节点扩展的效果很差. (a) 纯 decode 批次的中位 TBT: 与节点内四路 TP、节点间 PP 相比, 跨节点 TP 使中位 TBT 增加超过 $2\times$. (b) 严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下的容量: 在严格 SLO 下, Sarathi-Serve 相比 vLLM 的纯 TP 和混合并行配置, 分别将 Falcon-180B 的服务容量提高 $4.3\times$ 与 $3.6\times$.](./sarathi-serve/figure-13.png)

**图 13.** TP 跨节点扩展的效果很差. (a) 纯 decode 批次的中位 TBT: 与节点内四路 TP、节点间 PP 相比, 跨节点 TP 使中位 TBT 增加超过 $2\times$. (b) 严格 (SLO-S) 与宽松 (SLO-R) 延迟 SLO 下的容量: 在严格 SLO 下, Sarathi-Serve 相比 vLLM 的纯 TP 和混合并行配置, 分别将 Falcon-180B 的服务容量提高 $4.3\times$ 与 $3.6\times$.

<span id="section-5-4"></span>

### 5.4 消融研究

本节从不同角度对 Sarathi-Serve 做消融研究, 主要回答两个问题: 1) 分块如何影响 prefill 吞吐量; 2) 混合批处理与分块如何影响延迟. 本节只给出少数实验的结果, 但下述趋势在各种模型与硬件组合中都一致.

<span id="section-5-4-1"></span>

#### 5.4.1 Chunked-prefills 的开销

[图 14](#figure-14) 展示了分块给 Yi-34B 的 prefill 总运行时间增加多少开销. 与预期相同, [图 14](#figure-14) 中的柱高逐渐下降, 表明块越小, 开销越高. 不过, 即便使用最小的 512 token 块, 观察到的开销也至多约为 25%. 使用 2048 的较大 token 预算时, chunked prefill 的开销几乎可以忽略.

<span id="figure-14"></span>

![图 14. Yi-34B (TP-2) 计算 prefill 时 *chunked-prefills* 的开销, 相对于不分块的成本归一化. 图中给出不同提示长度在块长度为 512、1024、2048 时的结果.](./sarathi-serve/figure-14.png)

**图 14.** Yi-34B (TP-2) 计算 prefill 时 *chunked-prefills* 的开销, 相对于不分块的成本归一化. 图中给出不同提示长度在块长度为 512、1024、2048 时的结果.

<span id="table-04"></span>

![表 4. 单独使用 *hybrid-batching*、*chunked-prefills* 以及二者结合时测得的 TTFT 与 TBT 延迟 (秒). 实验在两块 A100 上运行 Yi-34B, 使用 128 个请求, token 预算为 1024. 同时采用 *hybrid-batching* 与 *chunked-prefills* 后, Sarathi-Serve 能同时降低 TTFT 和 TBT.](./sarathi-serve/table-04.png)

**表 4.** 单独使用 *hybrid-batching*、*chunked-prefills* 以及二者结合时测得的 TTFT 与 TBT 延迟 (秒). 实验在两块 A100 上运行 Yi-34B, 使用 128 个请求, token 预算为 1024. 同时采用 *hybrid-batching* 与 *chunked-prefills* 后, Sarathi-Serve 能同时降低 TTFT 和 TBT.

<span id="section-5-4-2"></span>

#### 5.4.2 各项技术的影响

最后, [表 4](#table-04) 给出了分别单独使用 Sarathi-Serve 各组件时的 TTFT 和 TBT 延迟, 包括只用 *chunked-prefills*、只用 *hybrid-batching* (把 prefill 与 decode 请求放入混合批次), 以及二者结合. 结果说明两项技术配合使用时效果最好: 只用 *chunked-prefills* 会增加 TTFT, 因为 prefill 块的效率稍低; 只用 *hybrid-batching* 会增加 TBT, 因为较长的 prefill 仍会造成生成停顿. 两者结合后, Sarathi-Serve 能同时改善这两个维度.

<span id="section-6"></span>

## 6 相关工作

**模型服务系统:** Clipper [Cra17]、TensorFlow-Serving [Ols17]、Clockwork [Guj20] 和 BatchMaker [Gao18] 等系统研究了模型服务中的多种放置、缓存与批处理策略, 但没有处理自回归 Transformer 推理的特殊挑战. 较近的 Orca [Yu22a]、vLLM [Kwo23]、FlexGen [She23]、FasterTransformers [Fas21]、LightSeq [Wan21a] 和 TurboTransformers [Fan21] 等系统, 则针对 Transformer 推理提出专门优化. FlexGen [She23] 面向资源受限的离线场景优化 LLM 推理吞吐量, 并不适合在线服务. FastServe [Wu23a] 提出抢占式 LLM 推理调度框架, 以缩短作业完成时间. Orca 与 vLLM 代表 LLM 推理的先进水平, 本文对二者做了详细比较.

最近出现的另一种方法是把 prefill 与 decode 阶段拆分到不同副本上, SplitWise、DistServe 和 TetriInfer [Pat23, Zho24, Hu24b] 都采用了这一思路. 这类方案可以彻底消除 prefill 与 decode 的相互干扰. 然而, 每个请求完成 prefill 后都需要迁移 KV cache; 如果不同副本之间没有高带宽互联, 这会相当困难. 此外, 这种方法没有充分利用 prefill 副本的 GPU 内存容量, 因为只有 decode 副本负责存储 KV cache. 好的一面是, 拆分式方法能以最高效率执行 prefill, 因而获得更好的 TTFT; chunked prefill 则比完整 prefill 稍慢. Sarathi-Serve 与拆分式方案之间的量化比较留待未来研究.

Sheng 等人 [She23b] 最近修改了迭代级批处理算法, 以保证多租户环境中各客户端的公平性. FastServe [Wu23a] 使用基于抢占的调度机制缓解队首阻塞. 这些算法优化与我们的方法互补, 也能受益于 Sarathi-Serve 所降低的 prefill-decode 干扰. 另一套近期系统 APIServe [Abh24] 采用 Sarathi 的 chunked prefill, 利用 decode 批次中浪费的计算能力, 为多轮 API 服务提前重新计算 prefill.

**提高 Transformer 的 GPU 利用率:** 近期工作从不同方面优化 Transformer 的硬件利用率. FasterTransformer 使用模型专用的 GPU 内核实现. CocoNet [Jan22] 和 [Wan22b] 尝试重叠计算与通信来提高 GPU 利用率. 当分布式模型采用较高的张量并行度、通信时间可能超过计算时间时, 这些技术尤其有用. 此外, 自注意力的计算成本随序列长度呈二次增长, 在长上下文中可能相当可观. [Rab22, Dao22c, Dao23a] 通过精心设计的 tile 与工作划分, 提出多种减轻自注意力内存瓶颈的方法. 研究者还探索了多种并行策略以优化模型放置. 这些技术与 Sarathi-Serve 相互独立.

**模型优化:** 大量模型创新工作试图弥补基于 Transformer 的语言模型的不足, 或者越过 Transformer, 进一步探索新的模型架构. 例如, multi-query attention [Sha19b] 在所有注意力头之间共享同一组 key 和 value, 以缩小 KV-cache, 让 GPU 能容纳更大的批次. 多项近期工作还表明, 量化可以显著压缩模型大小 [Xia23a, Fra23, Det23a, Det22b]. 混合专家模型主要用于减少每轮迭代中被激活的模型参数量 [Art22, Li23i, Hua23a]. 更近期的工作提出 retentive network 作为 Transformer 的后继架构 [Sun23a]. 与这些工作不同, 我们从 GPU 角度着手解决主流 Transformer 模型的性能问题.

<span id="section-7"></span>

## 7 结论

LLM 推理希望同时取得高吞吐量和低延迟, 但这并不容易. 我们广泛分析了现有 LLM 推理调度器, 将其分为 *prefill-prioritizing* 与 *decode-prioritizing* 两类. 总体而言, 前者更擅长优化吞吐量, 后者更擅长优化 TBT 延迟. 当吞吐量与延迟同样重要时, 两者都不理想.

为解决这一权衡, 我们提出 Sarathi-Serve, 这套系统用 *chunked-prefills* 和 *stall-free batching* 实现了一种新方法. Sarathi-Serve 把输入提示拆成更小的工作单元, 构成无停顿调度. 因而, 它可以在不暂停已有 decode 的情况下, 向正在运行的批次加入新请求. 评估表明, Sarathi-Serve 在单块 A100 GPU 上能将 Mistral-7B 的服务容量提高至多 $2.6\times$, 在 8 块 A100 GPU 上能将 Falcon-180B 的服务容量提高至多 $5.6\times$.

## 致谢

感谢 OSDI 审稿人和 shepherd 提出的深刻意见. 本研究部分由 GT Cloud Hub 支持, 该项目隶属数据工程与科学研究所 (Institute for Data Engineering and Science, IDEaS), 资金来自 Microsoft 与 Georgia Tech 的新型计算层次结构研究中心 (Center for Research into Novel Compute Hierarchies, CRNCH).

<span id="section-8"></span>

## 8 Artifact 附录

### 摘要

我们的开源 artifact 发布在 [GitHub](https://github.com/microsoft/sarathi-serve). 仓库既包含 Sarathi-Serve 的实现, 也包含运行和绘制本文实验结果所需的 harness 与脚本.

该仓库最初从 vLLM 项目 fork 而来. Sarathi-Serve 是一套轻量、高性能的研究原型, 功能并未与开源 vLLM 完全对齐. 我们只保留了最关键的功能, 并调整代码库, 以便更快地开展研究迭代.

<span id="section-8-1"></span>

### 8.1 范围

该 artifact 可以帮助读者验证 Sarathi-Serve 论文中的主张 (各幅图), 也提供复现实验的途径. 借助它可以搭建所需环境、执行主要实验并完成微基准测试, 从而全面理解 Sarathi-Serve 的核心主张.

<span id="section-8-2"></span>

### 8.2 内容

仓库结构如下: 系统的主要源代码位于 */sarathi* 目录, 自定义 CUDA 内核的实现位于 */csrc* 目录, 复现实验所需的全部脚本位于 */osdi-experiments*, 实验使用的轨迹文件则保存在 */data*.

<span id="section-8-3"></span>

### 8.3 托管

可以从 GitHub 获取我们的 artifact: [GitHub](https://github.com/microsoft/sarathi-serve). GitHub 仓库的主分支仍在持续更新, 但我们会在容易找到的 README 文件中维护清晰、可用的 artifact 说明. 复现 OSDI 论文实验所需的详细说明与 README 文件均位于 *osdi-sarathi-serve* 分支.

<span id="section-8-4"></span>

### 8.4 要求

Sarathi-Serve 已在配备 A100 和 A40 GPU 的 CUDA 12.1 环境中通过测试. 为便于复现, artifact 中与各幅图对应的 README 已清楚说明实验使用的具体 GPU SKU 与并行策略.

[+1]: 我们把近期的 Splitwise [Pat23] 和 DistServe [Zho24] 调度器归入第三类“拆分式”系统, 并在[第 6 节](#section-6) 讨论.

[+2]: 理论上, 这些算子在 A100 GPU 上处理约 200 个 token 时就会转为受计算限制. 实际中, 由于固定开销, 我们观察到张量并行维度较高时, 转折点约为 500-600 个 token.

[+author-note]: 本工作的一部分在 Microsoft Research India 实习期间完成.
