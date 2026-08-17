---
title: 'LMCache: Enterprise-Scale KV Caching'
createTime: 2026/08/16 08:48:59
permalink: /papers/lmcache/
---

> [Yuhan Liu](https://yuhanliu11.github.io/), [Jiayi Yao](https://dblp.org/pid/71/10222), [Yihua Cheng](https://www.tensormesh.ai/about), [Yuwei An](https://dblp.org/pid/18/10645), [Xiaokun Chen](https://dblp.org/rec/journals/corr/abs-2510-09665), [Shaoting Feng](https://shaoting-feng.github.io/), [Yuyang Huang](https://royyhuang.github.io/), [Samuel Shen](https://dblp.org/pid/417/0856), [Rui Zhang](https://www.linkedin.com/in/rui-zhang-0279aa201), [Kuntai Du](https://kuntaidu.github.io/aboutme.html) 和 [Junchen Jiang](https://people.cs.uchicago.edu/~junchenj/) [+equal]. 首次提交至 arXiv: October 8, 2025; 当前版本为 v2, December 5, 2025. [LMCache: An Efficient KV Cache Layer for Enterprise-Scale LLM Inference](https://arxiv.org/abs/2510.09665). [原始 PDF](/paper/lmcache.pdf). [DOI](https://doi.org/10.48550/arXiv.2510.09665). [TeX 源码](https://export.arxiv.org/e-print/2510.09665v2). 精确的印刷版式和参考文献以原始 PDF 为准.

[+equal]: *Yuhan Liu, Jiayi Yao 和 Yihua Cheng 贡献相同. *

## 摘要

传统上, KV cache 存放在 GPU 内存中, 用于加速大语言模型 (LLM) 推理的解码阶段. 不过, 现在越来越有必要把 KV cache 移到 GPU 设备之外, 以便在不同查询和推理引擎之间复用 cache. 我们的真实使用统计证实了这一趋势: 用户存储的 KV cache 总量持续快速增长, 远超 GPU 内存容量.

尽管有这样的需求, 现有系统还没有高效卸载和传输 KV cache 的方案. 我们提出 **LMCache**, 这是首个, 也是目前效率最高的开源 KV cache 方案. 它从现代 LLM 引擎 (vLLM 和 SGLang) 的 GPU 内存中提取并存储 KV cache, 再在不同引擎和查询之间共享. LMCache 同时支持 *cache 卸载* (跨查询的前缀复用) 和 *预填充-解码 (PD) 分离* (跨引擎/GPU 的 KV cache 传输).

LMCache 的高性能和广泛采用来自以下贡献: *(i)* 通过批量数据移动, 计算与 I/O 流水线实现高效的 KV cache 数据移动; *(ii)* 模块化的 KV cache connector, 将 LMCache 与快速演进的推理引擎解耦; *(iii)* 一组一等控制 API, 包括 pin, lookup, cleanup, move 和 compression, 让 cache 可以在 GPU, CPU, 存储和网络层之间灵活编排.

评估表明, LMCache 与 vLLM 结合后, 在多轮问答和文档分析等工作负载上的吞吐量最高提升 $15\times$. LMCache 在企业中的大规模部署也带来了一些经验: 从远程存储获取 KV cache 可以降低预填充延迟, 而业界常用的上下文截断会让前缀 cache 命中率降低一半.

LMCache 的源代码见 [https://github.com/LMCache/LMCache](https://github.com/LMCache/LMCache).

<span id="section-01"></span>

## 1 引言

如今, 大语言模型 (LLM) *推理* 的增长速度已经超过训练. LLM 推理支撑着数百万种应用, 从交互式客服, 代码生成, 到基于检索的文档分析和智能体工作流. 为了构建 LLM 推理系统, KV cache, 即 LLM 推理产生的中间状态, 已经成为加速推理的事实标准优化.

传统上, KV cache 用于新 token 生成, 以跳过同一查询中输入 prompt 的 KV cache 重算. 这样, 每个查询都由推理引擎的*一个实例*独立处理, 一个 LLM 查询的整个生命周期, 包括计算和 I/O, 都发生在同一个推理引擎的 GPU 和 GPU 内存中.

<span id="figure-01"></span>

![图 1: LMCache 的用户数量, 存储的 KV cache 大小和 Docker 拉取次数随时间增长.](./lmcache/figure-01.png)

**图 1.** 上: 使用 LMCache 的用户越来越多. 中: LMCache 存储的 KV cache 越来越大. 下: LMCache Docker 镜像的拉取次数持续增加.

不过, 近期出现了把 KV cache 移出 GPU 内存的趋势, 主要包括两个方向:

**跨查询缓存以避免重复计算:** KV cache 可以在查询生命周期结束后*持久化* (例如放到低层存储设备), 从而避免另一个查询重新计算共享前缀.

**预填充-解码分离以提高利用率:** 一个新趋势是把预填充和解码放到不同 GPU 上, 让对延迟敏感的解码阶段不受吞吐导向的预填充阶段影响. 这种预填充-解码 (PD) 分离需要把预填充 GPU 产生的 KV cache 传到解码 GPU.

我们的真实使用统计证实了 KV cache 正在离开 GPU 内存. 第 [2.2 节](#section-02-02) 会详细讨论这一点: 用户存储的 KV cache 总量持续增长, 远超 GPU 内存容量. 随着请求增多, GPU 内存中的 KV cache 可能需要频繁驱逐. 要让另一个查询复用它, 还必须先把 cache 从 GPU 内存卸载出来, 再加载回 GPU 内存.

要让这些思路真正可用, LLM 推理系统需要加入新的 KV cache 语义. 具体来说, 推理引擎应支持从一次普通推理调用中*提取* KV cache, 并按需把它*重新加载*到后续查询中. 系统还必须支持持久化存储提取出的 KV cache, 并在分布式推理引擎之间传输它. 更重要的是, KV cache 的提取, 重新加载, 存储和传输必须高效, 新接口也必须兼容快速演进的推理引擎, 例如 vLLM [Kwo23] 和 SGLang [Lm01].

我们提出 **LMCache**, 一个高性能实现这些 KV cache 语义的开源库. 借助 LMCache, KV cache 可以高效地从推理引擎中提取并重新加载, 存放在分层存储设备中 (CPU 内存, 本地磁盘, 远程磁盘和 Redis), 也可以通过不同网络 (Ethernet, RDMA, NVLink) 传输.

LMCache 有三项贡献.

**#1. 高度优化的性能:** LMCache 采用一系列优化, 让真实部署中的 KV cache 存储和加载变得高效可用. 例如, LMCache 批量处理操作, 把 KV cache 的存取做成流水线, 也把 GPU 计算与数据加载/存储做成流水线 (例如在计算当前层时加载下一层的 KV cache). LMCache 不再按推理引擎原生的小 page 存取 KV cache, 而是使用可配置的, 通常远大于 page 的 chunk, 充分利用存储设备与 GPU 内存之间的带宽. 不同存储层之间移动 KV cache 时, LMCache 还通过 zero-copy 操作减少数据复制.

**#2. 与推理引擎的标准化接口:** LMCache 定义了标准化 connector 接口, 兼容快速变化的推理引擎后端. 2025 年, 平均每周会发布 15-20 个新的开放权重模型. 为了在新模型上利用新硬件, 推理引擎必须快速演进, 这可能改变 GPU 内存中的 KV cache 布局, 进而影响 LMCache 接口. 为此, LMCache 设计并实现了模块化 KV cache connector, 将 LMCache 与推理引擎后端解耦, 让 LMCache 能够适应推理引擎不断变化的 API.

**#3. 灵活的 KV cache 管理接口:** 这组接口把 KV cache 暴露为 LLM 推理中的一种新数据结构. LMCache 提供 API, 允许开发者和运维人员定位, 移动, 固定甚至压缩从推理引擎中提取的 KV cache. 这些一等 API 让查询调度器或路由器等上层应用能够做出更好的决策, 例如按 KV cache 感知的方式路由查询.

我们的评估显示, LMCache 在多种场景下都优于开源推理框架内置的 KV cache 机制和商业推理 API, 吞吐量最高提高 $15\times$, 延迟至少降低 $2\times$. 这些场景包括本地前缀缓存, 分布式前缀复用和 PD 分离.

除了量化结果, LMCache 已被多家企业和开源项目采用, 也让我们积累了 KV cache 驱动优化的生产经验. 其中包括远程存储带来的延迟收益, 上下文截断导致的前缀命中率下降, 以及快速演进和顺畅集成比语言性能本身更重要这一点.

本文接下来介绍动机 ([第 2 节](#section-02)) 和挑战 ([第 3 节](#section-03)), LMCache 的架构和主要设计 ([第 4 节](#section-04), [第 5 节](#section-05) 和 [第 6 节](#section-06)), 部署经验 ([第 9 节](#section-09)) 以及实验评估 ([第 8 节](#section-08)).

<span id="figure-02"></span>

![图 2: 上下文缓存跨查询复用 KV cache, PD 分离则在推理引擎之间传输 KV cache.](./lmcache/figure-02.png)

**图 2.** LMCache 同时支持上下文缓存 (跨查询卸载和共享 KV cache) 与 PD 分离 (跨引擎传输 KV cache).

<span id="section-02"></span>

## 2 动机与真实使用统计

### 2.1 LLM 推理中的 KV Cache

*KV cache* 最初用于加速单个推理查询: 它把输入 token 和此前生成 token 的注意力状态, 也就是 $K$ 和 $V$ 张量, 直接存放在 GPU 内存中. KV cache 实际记录了这个查询中已出现 token 两两之间的注意力信息. 简言之, 它是一种 *LLM 原生* 的知识表示.

如今, 上下文越来越长, 人们也开始用背景知识增强推理. 于是, 在不同用户查询之间共享 KV cache, 以减少长上下文或背景知识带来的重复计算, 已经很常见.

<span id="figure-03"></span>

![图 3: GPU 内存容量以内和以外的 KV cache 大小按周增长.](./lmcache/figure-03.png)

**图 3.** KV cache 大小按周增长, 包括能放入 GPU 内存的部分以及超出 GPU 内存的部分.

<span id="section-02-02"></span>

### 2.2 真实使用统计

**KV Cache 大小超过 GPU 内存:** 传统 LLM 推理系统一直把 KV cache 保存在 GPU 内存中. 根据用户自愿开启的使用跟踪器收集的真实统计, 我们发现 KV cache 的需求规模已经远超 GPU 内存容量.

[图 3](#figure-03) 展示了过去五周 KV cache 大小的周增长, 绿色表示能放入 GPU 内存的 cache, 蓝色表示超过 GPU 内存容量的 cache. 随着时间推移, 无法放入 GPU 内存的部分显著增加, 说明仅靠 GPU 内存无法保存所有 cache. 要在查询之间复用 KV cache, 尤其是复用很久以前生成的 cache, 就必须把它移出 GPU 内存, 例如卸载到 CPU 内存或其他存储层.

**每 token 的复用次数大幅上升:** 我们还观察到, 每 token 的复用次数随时间明显增加.

[图 4](#figure-04) 左侧绘出了超出 GPU 内存的 cache 中, 已复用 token 与所有已存储 token 的比值, 并列出复用量最高的 10 个用户. 我们把这个比值称为每 token 的复用次数. 过去几周, 每 token 的复用次数明显上升, 说明那些无法放入 GPU 内存的 token 被推理越来越频繁地复用. 这意味着越来越多的 token 需要重新加载到 GPU 内存.

图右侧展示了过去一周不同用户的每 token 复用次数分布. 超过 19% 的用户会把已存储 token 复用 1.5 次以上, 可以看出用户在存储 token 后多次访问它的趋势.

<span id="figure-04"></span>

![图 4: 头部用户的每 token 平均复用次数及其用户分布.](./lmcache/figure-04.png)

**图 4.** 左: 头部用户的每 token 平均复用次数. 右: 不同用户的每 token 平均复用次数分布.

### 2.3 需要高效的 KV Cache 层来移动 KV Cache

根据真实部署统计, 我们看到 KV cache 有两个趋势. 第一, 无法放入 GPU 内存的 KV cache 持续增长, 可能是上下文变长或用户流量增加造成的. 第二, GPU 内存之外存储的 token 的复用次数也在增加. 两个趋势都说明 KV cache 需要离开 GPU 内存. 在当前业界, 主要有两种场景会把 KV cache 移出 GPU:

1. *上下文缓存 (即跨查询 KV cache 复用)*: 保存一次查询中的 KV cache 片段, 并在后续共享相同前缀的查询中复用. 例子包括对同一文档 (或文档片段) 进行多次分析, 以及带固定 system prompt 或长前导文本的多轮对话. 前缀缓存能减少预填充阶段的重复计算, 直接降低 TTFT 和每次查询的 GPU-hours [Liu24d, Lm02, Lm03, Lm04, Qin24, Lm05, Lm06, Lm07].
2. *预填充-解码 (PD) 分离 (即跨引擎 KV cache 传输)*: 把推理拆成预填充阶段 (处理完整输入 prompt) 和解码阶段 (自回归生成 token), 分别放在不同 GPU 或节点上. 这种方法让解码速度不被预填充打断, 因而降低尾延迟 [Zho24, Lm08, Lm09].

不过, 由于后文将讨论的系统挑战 ([第 3 节](#section-03)), 目前还没有一个库能高效地从 GPU 内存提取和加载 KV cache.

<span id="section-03"></span>

## 3 高效 KV 缓存的挑战与相关工作

### 3.1 高效 KV 缓存的挑战

虽然前缀缓存和 PD 分离有潜力, 但它们的实际采用受到三个相互关联的系统挑战限制:

<span id="table-01"></span>

![表 1: RCCL 传输库中消息大小与传输吞吐量.](./lmcache/table-01.png)

**表 1.** 使用 RCCL 传输库时, 传输消息大小与实际传输吞吐量的关系 [Lm10].

#### 3.1.1 挑战 #1: 分页内存下的 I/O 低效

KV cache 的存储和传输过去依赖 PyTorch 序列化 (`torch.save` / `torch.load`) 或简单的张量复制, 典型传输速度只有 1GB/s 以下. 处理 KV cache 这类大型数据结构时, 这些方法会引入明显的延迟开销; 它们也不支持多种存储设备 (本地或远程) 的 zero-copy, 因而需要额外的 CPU-GPU 数据复制.

近年来的高吞吐推理引擎, 例如 vLLM [Kwo23] 和 SGLang [Lm01], 让 KV cache 的存储和传输更难. 它们使用分页注意力内存, 把注意力缓冲区分成小的定长 page (通常为 16-64 KB). 例如, Llama-3.1-8B-Instruct 中 vLLM 使用 62.5 KB 的 page. 分页内存架构之所以广泛使用, 是因为它能改善批处理和内存利用率.

然而, KV cache 的 page 不一定连续, 分页内存架构会大幅增加持久化或传输 KV cache 所需的小 I/O 操作数量. 已知传输这么小的数据块会造成网络带宽利用率不足, 降低吞吐量 [Lm11, Lm12, Lm13]. [表 1](#table-01) 所引用的工作显示, 在两台 AMD GPU 节点通过 8 个 Broadcom Thor-2 400Gbps 网卡连接的环境中, 传输块至少要达到 16 MB 才能占满网络带宽 [Lm14]. 另一项工作显示, 只有把传输大小提高到 MB 量级 (例如 1-2MB), 才能达到 PCIe 5.0 理论带宽的 75-80% [Lm15].

#### 3.1.2 挑战 #2: 兼容快速演进的推理引擎

随着 AI 普及, 新的 LLM 和硬件加速器不断出现. 2025 年, 平均每 4 天就会发布一个重要 LLM [Lm16]. 推理引擎也必须同样快速地演进.

为了支持新模型或硬件, 每次更新往往都会改变 GPU 内存分配方式, 从而改变 KV cache 接口. 例如, vLLM 采用新的注意力 kernel 后, 生成的 KV cache 可能具有不同维度; KV cache 库就必须把新的 kernel 输出格式转换为自身支持的格式. 面对快速变化的推理引擎, 持续跟进这些变化需要大量工作.

#### 3.1.3 挑战 #3: 缺少管理 API

KV cache 成为 LLM 推理后端中的一等数据结构后, 除推理引擎外的各种组件和 ML 运维团队也需要基于 cache 做决定. 但如果没有统一的管理接口来定位, 驱逐, 固定或压缩 cache, 上层模块就无法合理决定放置和驱逐策略. 这会造成 cache 利用率低, 存储重复和不可预测的驱逐行为.

例如, 查询路由器负责把每个查询分配给一个推理引擎实例. 它需要知道 KV cache 的位置, 才能把查询路由到本地 (例如 CPU 内存中) 已经保存匹配前缀的实例.

应用也开始要求这样的 KV cache 管理接口. 2025 年初, 一家与 LMCache 密切合作的金融公司 [+1] 要求提供一个接口, 让用户能够*显式*固定经常访问的金融文档, 以便更快访问热门上下文. 另一个智能体公司则要求提供一组 API, 用来定位给定内容对应的 KV cache, 压缩它, 并在节点之间传输压缩后的 cache.

[+1]: 出于保密原因, 本文不披露企业用户的名称.

### 3.2 相关工作与已有方案

已经有多种 KV cache 处理机制, 但它们都没有完全解决上面的挑战:

**推理框架:** vLLM Production Stack [Lm17] 于 2025 年 1 月发布后, 出现了多个开源分布式推理栈, 包括 Nvidia 的 Dynamo [Lm18], AIBrix [Lm19], `llm-d` [Lm20], SGLang OME [Lm21] 和 KServe [Lm22]. 它们都关注在 Kubernetes 上简化推理引擎部署, 在技术上也都支持按负载或前缀 cache 感知的查询路由和 KV cache, LMCache 已被用于 vLLM production stack, Dynamo, llm-d 和 KServe.

**推理引擎原生的 KV cache:** vLLM 和 SGLang 等开源推理引擎也提供 GPU 到 CPU 的 KV cache 传输, 但它们面向单节点推理, 缺少跨节点传输优化和 KV cache 的分层存储支持. 第 [8 节](#section-08) 会评估它们的性能并与 LMCache 对比.

**KV cache 存储层:** Mooncake [Qin24], Redis [Lm23], InfiniStore [Lm24] 和 3FS [Lm25] 提供分布式对象存储或缓存, 但缺少推理引擎与存储层之间的高效"胶水层", 无法频繁地在不同存储层之间移动小张量, 或者与某一个推理框架绑定得很紧.

**专有实现:** Fireworks AI, Together AI 等专有推理 API 在内部实现了前缀缓存, 但它们绑定自己的闭源服务栈, 不向自行部署基础设施的运维人员开放.

**研究代码:** 一些研究工作开源了 KV cache 优化原型, 包括前缀缓存 [Kwo23, Lm01, Lm26, Lm27, Lm28, Lm29, Lm30, Lm05, Lm02, Lm07, Lm03, Lm31], PD 分离 [Zho24, Lm08, Lm09] 和 KV cache 压缩 [Liu24c, Lm32, Lm33, Xia24a, Li24c, Lm34, Lm35, Lm36, Lm37, Lm38, Lm39]. 不过, 这些原型通常基于 HuggingFace Transformers 等面向研究的推理框架, 尚未达到企业可用的程度, 也没有为 SGLang 和 vLLM 这样的快速变化的推理引擎生态设计.

<span id="section-04"></span>

## 4 LMCache 概览

**LMCache** 是一个统一的高性能 KV cache 层, 能为分页内存推理引擎高效地存储, 移动和显式管理 KV cache, 让前缀缓存和 PD 分离在企业规模上可用.

作为 KV cache 层, LMCache 位于 LLM 推理引擎与异构存储/网络设备之间 ([图 5](#figure-05)). 它提供标准化, 高性能的 KV cache 移动和管理基础, 同时兼容快速演进的 vLLM, SGLang 等推理框架.

<span id="figure-05"></span>

![图 5: LMCache 位于推理引擎与异构存储, 网络设备之间.](./lmcache/figure-05.png)

**图 5.** LMCache 位于 LLM 推理引擎与异构存储/网络设备之间.

[图 6](#figure-06) 展示了端到端系统. 下面通过两个工作流说明 KV cache 的存储和读取.

**Store:** 新查询到达后, 首先经过 *KV connector*, 生成 tokenized input prompt, 相关 page 的 GPU 内存地址等元数据. 查询随后进入 *token processor*, 由它判断后端还没有存储多少新 token. 最后, storage manager 通过负责数据传输的 *transfer channel* 把这些 token 对应的 KV cache 保存到后端.

**Retrieve:** 查询需要从后端加载 KV cache 时, 同样先经过 KV connector 准备元数据. token processor 找出后端已有的前缀匹配 token 数量. 接着, event manager 检查是否见过相同的 query ID. 如果见过, 已缓存的内存地址已经被记录, 可以直接返回给 *GPU connector*, 后者把 KV cache 加载回 GPU 内存. event manager 还会启动异步的逐层加载事件, 如 [第 5.2 节](#section-05-02) 所述. 如果是新的 query ID, 查询会转发给 storage manager, 由它查找已存储 KV cache 的 CPU 内存地址.

**Lookup:** 查询需要确认后端是否存在特定 token 的 KV cache 时, 路由器等上层组件会查询 cache controller. cache controller 维护一个 token pool, 记录 KV cache 后端当前存储的所有 token. 每当一个 LMCache 实例存储或驱逐 KV cache 时, 该实例中的 LMCache worker 就更新 token pool 的状态. 因此, token pool 始终记录后端 token 的最新状态.

<span id="figure-06"></span>

![图 6: LMCache 的端到端系统工作流.](./lmcache/figure-06.png)

**图 6.** LMCache 的端到端系统工作流.

<span id="section-05"></span>

## 5 性能优化

LMCache 的一个重点是提高 KV cache 在设备之间移动的效率. 在企业规模的 LLM 推理中, LMCache 处理三个主要问题:

- 现代 LLM 推理引擎以 page 为粒度管理 KV cache [+2], 对 Llama, Qwen, GPT-OSS 等常见模型来说, page 通常只有 20 KB-63 KB. 这种小单元无法占满带宽, 传输效率很低 [Lm15, Lm14].
- KV cache 传输经常需要与 LLM 推理并行进行, 会产生两类开销. 第一, 如果传输和计算使用同一个 CUDA stream, 数据移动会阻塞推理. 第二, 启动内存复制 CUDA 函数需要 CPU 开销; 当层数和 page 数很多时, 每次调用都消耗 CPU 周期, 累积开销很大.
- LLM 推理过程中, 大量查询会生成大量 KV cache. 在存储设备上复制这些 cache 会浪费空间并带来复制开销, 使推理变慢.

[+2]: 在 vLLM 中, 每个 page 对应单层的 16 个 token.

这些问题都来自开源项目和企业部署中的实际经验. 本节分别介绍它们, 并说明 LMCache 的设计选择.

### 5.1 批量操作

为解决 KV cache 单元过小导致的 I/O 低效, LMCache 引入了一组优化.

**可配置的 Chunk 大小:** LMCache 不再按 page 传输 KV cache, 而是把多层的多个 page 组合成更大的 chunk, 默认每个 chunk 包含 256 个 token [+3]. 实现上使用一个中间的 *streaming GPU buffer*. 存储时, 先用定制 CUDA kernel 把分散的分页 GPU 内存复制到连续的 streaming buffer, 再由 DMA 引擎以 chunk 为粒度整体卸载到低层存储 (例如 CPU 内存), 而不是逐个 page 卸载. 加载时, 先由 DMA 引擎把 chunk 从存储层取到 GPU buffer, 再用 CUDA kernel 拆回分页内存.

[+3]: Chunk 大小可以根据 I/O 速度配置.

**并行 Store/Load 操作:** LMCache 支持在多个存储层之间并行存取 KV cache, 包括本地 CPU DRAM 或磁盘, 远程 CPU DRAM 或磁盘以及对象存储 (例如 S3). 实际 LLM 服务中, KV cache 经常需要同时在多个设备间迁移, 例如把热门上下文从 GPU 传到 CPU 内存, 同时把冷门上下文从 CPU 内存卸载到本地磁盘. 为了充分利用链路, LMCache 的 store 和 load API 接受多个源设备和目标设备, 可以在异构链路上并行移动数据. 互连支持全双工通信 (例如 PCIe) 时, 这些操作也可以并行执行.

**解码 KV Cache 延迟存储:** LMCache 还支持在解码期间存储新生成的 KV cache. 与立即卸载每个 token 的 KV cache 的朴素做法不同, 后者会频繁执行小写入, LMCache 会先缓存 KV cache, 等生成预设数量的 token (即一个 chunk) 后再批量存储. 按 chunk 延迟写入可以减少写入次数和 I/O 开销, 提高整体存储吞吐量.

<span id="section-05-02"></span>

### 5.2 计算与 I/O 重叠

LMCache 采用多种优化, 让 LLM 推理计算与 I/O 重叠, 以提高 GPU 利用率.

**逐层流水线:** LMCache 通过逐层流水线让 KV cache 传输与推理计算重叠. 它为每层的推理计算和数据移动分配不同的 CUDA stream. 例如, 执行第一层推理前, 先把该层 KV cache 加载到 GPU buffer 并转换成 page. 第一层运行推理时, 异步把第二层的 KV cache 取到 buffer 并做同样的转换. 第二层的 KV cache 会在第一层 KV cache 放入正确的分页内存后再加载. 这样只需要一个固定大小的 GPU buffer, 大小等于单层 KV cache, 同时可以重叠数据传输和计算.

**异步计算与预取:** 在许多场景中, 推理调度器接纳查询与真正需要该查询 KV cache 之间有一段空闲时间. 例如, 一个命中 cache 的查询在推理引擎处理其他查询时到达, 就必须先在队列中等待. LMCache 利用这段空闲时间, 把排队查询的 KV cache 从较慢的存储层预取到更快的存储层 (例如从远程磁盘取到本地 CPU 内存或 GPU 内存). 实际推理开始时, 所需 KV cache 已经可以从更快的存储层加载或直接使用, 加载延迟因此降低. 用户可以根据延迟 SLO 和资源限制配置预取目标层.

### 5.3 最少数据复制

朴素的 KV cache 移动实现会在每个传输步骤创建额外副本, 异构存储尤其如此, 造成多余的内存使用和开销. LMCache 只保留必要的最少副本.

**Zero-Copy 操作:** 同时把 KV cache 传给多个设备时, LMCache 用引用计数器减少数据复制. 例如, 同一份 KV cache 同时从本地 CPU 内存写到本地磁盘和远程对象存储时, LMCache 为每次传输增加共享数据的引用计数, 而不是创建副本. 每次读或写完成后计数减一, 计数归零时释放数据. 这样, 并发读写可以共享数据而不产生多余复制, 减少内存压力并提高效率. 这种技术与操作系统中的 PCB counter 类似 [Lm40].

<span id="figure-07"></span>

![图 7: 动态卸载 KV cache 的四种状态.](./lmcache/figure-07.png)

**图 7.** LMCache 动态卸载示意图.

**动态卸载:** vLLM 等现代推理引擎会在 GPU 内存中维护一组 *free page*, 即当前没有被活跃查询使用的 KV cache page. LMCache 不把所有 free page 都复制到 CPU 内存, 而只复制其中一部分. 这个机制使用三个指针:

- **Start pointer:** GPU 内存中 free-page 区域的起始地址.
- **Current pointer:** 已经卸载到 CPU 内存的 free page 索引.
- **End pointer:** 计划卸载的 free page 区域结束地址.

如 [图 7](#figure-07) 所示, 动态卸载有四种状态:

- **State #1 (Initialization):** start 和 current 指针重合. start/current 指针与 end 指针之间的区域表示等待复制的 page.
- **State #2 (In-progress):** current 指针向 end 指针移动. start 与 current 之间的 page 已经卸载到 CPU 内存.
- **State #3 (Query Arrival):** 新查询占用一部分 free page 时, end 指针按分配 page 数量向前移动, 确保后续活跃查询仍有足够 GPU 内存可用.
- **State #4 (Steady state):** current 指针与 end 指针重合, 表示所有计划的 page 都已复制.

如果查询尝试分配 current 指针之后的 page, 就必须等 current 指针继续向右移动, 覆盖所需 page 后才能分配. 因此, 这个设计的权衡是 GPU 与 CPU 之间复制的 page 数量, 即 end pointer - start pointer 定义的区域. 复制窗口越小, 复制比例越低, 但分配停顿的概率越高. 例如, 只复制一个 page, 而查询需要三个 page 时, 查询必须等 current 和 end 指针再向前移动两个 page. 相反, 如果复制了三个 page, 查询可以立即执行, 代价是更高的复制比例. 虽然目前不支持, 同样的动态卸载策略也可以扩展到 CPU 和 GPU 之外的其他存储层.

<span id="section-06"></span>

## 6 连接 KV 缓存层与推理引擎的标准化接口

vLLM 和 SGLang 等现代 LLM 推理引擎为了支持不断发布的多种架构新模型, 演进速度很快. 例如, 2025 年平均每周会发布 15-20 个新模型. 支持这些新架构通常需要对推理引擎做并非微不足道的修改, 例如增加对 Sliding Window Attention 或 Multi-Head Latent Attention 的支持. 这些代码变更常常会改变 KV cache 的内部管理方式, LMCache 无法通过临时适配来跟上变化.

为解决这个问题, LMCache 引入了标准化 KV cache connector 接口, 把 KV cache 管理与推理引擎后端解耦. 无论上游推理引擎如何演进, 这个设计都能让 LMCache 保持兼容.

这个 API 的设计由 LMCache 团队发起, 但它的实现与维护是 LMCache 团队和 vLLM 团队的共同工作.

**设计目标:** 主要设计目标如下:

- **最大灵活性:** 尽可能支持更多 KV cache 操作.
- **vLLM 原生:** 与 vLLM 的设计方向一致, 包括严格分离 scheduler 和 worker, 把前缀缓存作为一等功能, 以及采用分段 CUDA graph, 即 vLLM 只为非注意力操作捕获 CUDA graph.
- **对树外 connector 友好:** 不修改 vLLM 端代码也能集成树外 connector.
- **最低 API 层开销:** 不在 API 层引入进程间通信等开销.

为了兼容 vLLM 分离 scheduler 与 model runner 的理念, connector API 包含两组接口: 1) scheduler 接口, connector 额外命中的 cache token 在 vLLM 中被当作普通的前缀缓存 token, 它们由 LMCache 改变并直接影响调度决策, 例如 LMCache 命中 cache 后, 需要新做预填充的 token 数量会改变; 2) model runner 接口, 在模型执行前后以及注意力计算前后加入 hook, 从而同时支持整体 KV cache 卸载和逐层 KV cache 卸载.

本节余下部分列出 [表 2](#table-02) 中的所有接口, 讨论重要 API 的设计, 再追踪一个查询端到端与这些接口交互的过程.

<span id="table-02"></span>

![表 2: LMCache connector 中的函数.](./lmcache/table-02.png)

**表 2.** LMCache connector 中的函数.

[表 2](#table-02) 列出的接口构成了 LMCache 在低层存储中加载和存储 KV cache 的基础. 前三个接口实现在 vLLM scheduler 内, 它们根据 LMCache KV cache 后端找到的匹配 token 数量准备所需元数据. 其余四个接口位于 model runner, 负责在推理引擎与 LMCache KV cache 后端之间实际传输 KV cache.

把这些过程合在一起, 查询到达后, scheduler 先调用 `get_num_new_matched_tokens`, 向 LMCache 查询后端命中的 cache token. 如果 LMCache 决定让 vLLM 把当前请求放回等待队列, 先处理其他请求, 以便将本请求的 I/O 与其他请求的计算重叠, 该函数可以返回 `None`.

然后, `update_state_after_alloc` 根据 LMCache 提供的匹配 token 信息, 决定 vLLM 中每个 page 是否需要从外部存储后端加载. 如果命中的 cache token 数大于零, 就调用 `build_connector_meta` 准备从存储设备加载或向其存储 KV cache 所需的元数据.

查询进入 model runner 后, 在逐层流水线模式下, 调用 `start_load_kv` 开始把第一层 KV cache 加载到 GPU 内存. 每层 LLM 推理计算开始前, 调用 `wait_load_kv` 同步本层 KV cache 的加载, 并开始加载下一层 KV cache.

每层推理计算完成后, 在逐层模式下, 调用 `wait_store_kv` 等待前一层 KV cache 存储完成, 再调用 `start_store_kv` 开始存储新生成的本层 KV cache.

在非逐层流水线模式下, 第一层 LLM 推理开始前, `start_load_kv` 会以阻塞方式把整个 KV cache 加载到 GPU 内存. KV cache 放入正确的 GPU 分页内存地址后, LLM 推理才会执行. 当前调度迭代的 LLM 推理完成后, 调用 `start_store_kv` 把生成的 KV cache 同步存到低层存储.

**影响:** 这个 API 已在 vLLM 中发布超过六个月. 期间已有多个开源项目采用, 包括 NVIDIA Dynamo, RedHat 的 llm-d, ByteDance 的 AIBrix 和 vLLM production stack. 我们也看到了多家公司使用 KV connector API 的专有 connector.

<span id="section-07"></span>

## 7 Controller 接口

LMCache 作为一个分布式缓存系统运行, 核心是负责全局元数据管理, cache 操作和请求路由的集中式 KV cache controller. 为支持这些功能, LMCache 提供两类 API: (1) 用户或系统运维人员可以直接访问的外部 API; (2) 个别 LMCache 实例使用的内部 API.

在机制上, KV cache controller 包含两层: 集中式 controller manager 和每实例 worker. controller manager 作为独立进程运行, 是全局协调点; 每实例 worker 与对应的 LMCache 实例共置, 负责本地操作, 或向 manager 发出全局请求. 外部 API 调用由集中式 manager 处理, 必要时再把适当操作分发给各 worker. 每实例 worker 也可以通过内部 API 主动与集中式 manager 交互, 用于更新或查询元数据.

KV cache controller 为跨节点 KV cache 共享, cache 感知的请求路由和动态 KV cache 迁移等一系列高级优化提供基础. 本节余下部分通过具体示例说明这些优化如何利用 controller 接口.

<span id="table-03"></span>

![表 3: LMCache Controller 中的内部和外部 API.](./lmcache/table-03.png)

**表 3.** LMCache Controller 中的 API.

**KV cache 感知路由:** 在这种情况下, 上层路由器希望把请求导向预期 cache 命中率最高的实例. 每个 LMCache 实例通过 `batched_admit` 和 `batched_evict` 接口向 controller manager 报告其 cache 接纳与驱逐决定. controller manager 汇总这些更新, 并在内存中维护一份跨所有实例的全局 KV cache 状态. 当路由器调用 `lookup(tokens)` 时, controller 查询内存中的全局 KV cache 状态, 并返回 `(instance_id, storage_device, hit_tokens)` 列表, 表示所请求 token 当前缓存的位置与命中数量.

**KV cache 迁移:** 保有 KV cache 的实例即将缩容, 或需要负载均衡时, KV cache 可能需要迁移到另一个实例. controller manager 通过 `move((src_inst_id, src_device), (dst_inst_id, dst_device), tokens)` API 调用处理这类操作, 将请求分发给源实例. 如果源实例与目标实例之间还没有连接, 源实例会尝试建立连接, 再把指定 KV cache 从源存储设备 `src_device` 传到 `(dst_inst_id, dst_device)` 指定的目标位置.

**P2P KV cache 共享:** LMCache 支持点对点 KV cache 共享, 本地 cache 未命中时, 实例可以从另一个 peer 获取 KV cache. 发生 cache miss 后, 实例的本地 worker 可以通过 `batched_p2p_lookup` 查询集中式 controller manager. manager 会返回 `(inst_id, device, hit_chunks)` 列表, 表示命中 chunk 数量及保存这些 chunk 的位置. 然后, 该实例可以选择从 `hit_chunks` 最大的 peer 加载 KV cache.

**清理 KV cache:** 应用切换模型或回收内存时可能需要清理 cache. 收到 `clear(tokens, inst_id, location)` 调用后, controller manager 会把操作分发给 `inst_id` 标识的对应实例. 该实例的 worker 随后从特定存储设备 `device` 中删除与 `tokens` 关联的 KV cache.

上述应用没有涵盖部分 API, 例如 `compress/decompress(tokens, inst_id, device, compression_method)` 会按指定的 `compression_method` 压缩或解压保存在 (`inst_id`, `device`) 位置的 KV cache, `pin/unpin(tokens, inst_id, device)` 则可以在特定位置 `(inst_id, device)` 固定或取消固定指定的 KV cache. 用户可以按照自己的应用需求自由调用这些 API, 显式管理 KV cache.

<span id="section-08"></span>

## 8 评估

<span id="figure-08"></span>

![图 8: LMCache 与四个基线在五个模型上的 TTFT 和 ITL.](./lmcache/figure-08.png)

**图 8.** 与基础 vLLM, 基础 vLLM CPU 卸载和两个商用方案相比, LMCache 的 TTFT 低 1.9-8.1$\times$, 支持的推理吞吐量高 2.3-14$\times$. 基础 vLLM CPU 卸载无法在 `Qwen3-Coder-480B` 上运行, 商用方案也不支持部署 `Qwen3-Coder-480B`.

### 8.1 实验设置

<span id="table-04"></span>

![表 4: 评估场景设置.](./lmcache/table-04.png)

**表 4.** 评估场景设置.

我们在 [表 4](#table-04) 所示的三种场景下评估 LMCache. 这三种场景都是 LMCache 用户常用的代表性配置.

**模型:** 我们在业界采用的常见开源模型上对比 LMCache 和基线方案: `meta-llama/Llama-3.1-8B-Instruct`, `Sao10K-L3-8B`, `meta-llama/Llama-3.1-70B-Instruct`, `Qwen/Qwen2.5-Coder-32B-Instruct`, `Qwen/Qwen3-Coder-480B-A35B-Instruct-FP8`, `Qwen/Qwen2.5-72B-Instruct`.

**数据集:** LMCache 在多个数据集上评估, 包括模拟多轮问答, LongBench [Bai23] 的长上下文问答, 以及 vLLM 官方基准测试脚本的随机数据集 [Kwo23].

**硬件:** 单节点评估在 GMI Cloud [Lm41] 提供的一台 $8\times$H100 服务器上运行 LMCache. 由于不同模型服务所需的 GPU 数量不同, 我们为每个模型分配能使其成功启动的最少 H100 GPU. 多节点评估使用与单节点设置相同数量的 GPU, 并配置一个集中式远程存储后端, 用 CPU 内存存储 KV cache. 在 PD 分离中, prefiller 和 decoder 实例的 GPU 数量也与单节点评估相同, 两个实例通过 NVLink 连接.

**指标:** 每项实验都报告 time-to-first-token (TTFT), 即预填充延迟, 以及 inter-token-latency (ITL), 即连续两个输出 token 生成之间的平均延迟. 对于分解 CPU 卸载或 PD 分离延迟的组件级分析, 我们分别报告各个组件的延迟.

**基线:** 我们将 LMCache `v0.3.6` 与多个基线对比, 包括:

- 基础 vLLM: vLLM `v0.10.2` 默认启用前缀缓存, 但只把 KV cache 保留在 GPU 内存中, 因此只能保留很小一部分;
- 基础 vLLM CPU 卸载: vLLM `v0.11.0` 及其自带的 CPU 卸载实现;
- 商用方案 #1 和 #2: 提供为用户保留 GPU, 以运行自定义模型的专用 endpoint 服务. 我们于 9 月 10 日访问并运行了这些基线.

### 8.2 单节点 CPU 卸载

我们先在 [表 4](#table-04) 的 CPU 卸载场景下评估 LMCache. 实验使用模拟典型聊天机器人文档分析场景的多轮问答负载. 默认情况下, 每个 LLM 查询包含 10K token, 由一份用作上下文的文档, 大约相当于 12 页 PDF, 以及一个唯一的简短问题组成. Llama-3.1-8B-Instruct 模型的输入为 20K token, 因为小模型通常更善于处理数量更多, 长度更长的查询. LLM 输出是最多 100 token 的短答案. 聊天会话从 40 个用户开始, 其他用户按指定到达率 (QPS) 加入. 我们把 LMCache 可用于卸载 KV cache 的 CPU 内存上限设为 500 GB.

如 [图 8](#figure-08) 所示, LMCache 在 TTFT 和 ITL 上都始终优于所有基线. 例如, 在低 QPS 下, 如 QPS = 1, LMCache 的 TTFT 低 1.9-8.1$\times$. 在 TTFT 相同时, 对五个评估模型而言, LMCache 的查询处理速率, 即吞吐量, 比最强基线高 2.3-14$\times$. 在 ITL 方面, LMCache 也优于基线, 因为基线在生成第一个 token 之前延迟很长, 进而导致后续 token 生成排队. 具体而言, 在 QPS=1 时, LMCache 的 ITL 比最佳基线低 7%-92%. 对于 Qwen3-Coder-480B, 商用方案 #1 和 #2 都不支持托管该模型.

**理解 LMCache 的收益:** LMCache 优于基线有多个原因. 基础 vLLM 只在 GPU 内存中缓存 KV 数据, LMCache 则利用 CPU 卸载. CPU 内存可以容纳比 GPU 内存多得多的 KV cache, 因此 LMCache 的 cache 命中率显著更高. 基础 vLLM CPU 卸载按层, 每 16 token 传输, 无法充分利用加载带宽; LMCache 按 chunk 级别加载 KV cache, 并使用高性能数据加载 CUDA kernel, 传输模块实现更高效. 由于闭源商用方案的内部实现不公开, 我们与它们的对比以黑盒方式进行. 根据端到端结果, 我们推测商用方案 #1 没有向次级存储卸载 KV cache 的机制. 相比之下, 商用方案 #2 可能支持向次级存储卸载 KV cache, 但性能仍不及 LMCache.

### 8.3 真实 Trace 驱动的评估

<span id="figure-09"></span>

![图 9: 公司 F 的三个模型在真实 trace 下的 TTFT 和 ITL.](./lmcache/figure-09.png)

**图 9.** 根据从公司 F 输入和输出分布提取的真实 trace, 对比 LMCache 和基础 vLLM 在三个不同模型上的表现. 在高 QPS 下, LMCache 的 TTFT 至少低 4.4-6.6$\times$, ITL 低 34%-58%.

<span id="figure-10"></span>

![图 10: 公司 F 和 G 的模型在真实 trace 下的 TTFT 和 ITL.](./lmcache/figure-10.png)

**图 10.** 上图和中图: 根据从公司 F 获取的真实 trace, 对比 LMCache 和基础 vLLM 在 Qwen 2.5 Coder 32B Instruct 和 Sao10K L3 8B 模型上的表现. 下图: 根据从公司 G 输入/输出数据分布获取的 trace, 对比 LMCache 和基础 vLLM 在 Llama 3.1 70B Instruct 上的表现. LMCache 的 TTFT 低 3.7-6.8$\times$, ITL 低 19%-44%.

我们用从公司 F 和公司 G 的输入输出 token 分布中提取的真实 trace 评估 LMCache. 由于无法使用公司 $F$ 的专有模型, 我们使用五个不同模型运行 trace. 为了让实验可以执行, 我们拉伸了原本持续数天的 trace, 使实际运行的负载在一小时内完成. LMCache 最多使用 500 GB CPU DRAM, 并与启用 GPU 前缀缓存的最新基础 vLLM 对比.

如 [图 9](#figure-09) 和 [图 10](#figure-10) 所示, 对五个模型的真实 trace, LMCache 在不同 QPS 下始终优于基础 vLLM. 具体而言, 在高 QPS 下, LMCache 在五个模型上把 TTFT 至少降低 3.7-6.8$\times$, 把 ITL 至少降低 19%-58%.

### 8.4 集中式存储服务器

<span id="figure-11"></span>

![图 11: 四个模型在集中式远程后端下的 TTFT 和 ITL.](./lmcache/figure-11.png)

**图 11.** 与基础 vLLM 相比, 在 TTFT 相同时, 使用远程后端卸载的 LMCache 将推理吞吐量提高 1.3-3$\times$.

接下来, 我们按照 [表 4](#table-04) 的集中式存储设置, 通过一台带宽为 15 Gbps 的集中式远程服务器连接 GPU 实例, 运行 LMCache 共享 KV cache. 本实验使用 LongBench [Bai23] 的 TriviaQA 数据集评估, 这是广泛使用的长上下文评估基准. 我们遵循 vLLM 官方基准测试脚本 [Kwo23], 按指定 QPS 的泊松分布生成推理查询.

如 [图 11](#figure-11) 所示, LMCache 在不同 QPS 下始终优于所有基线, 将推理吞吐量提高 1.3-3$\times$. 提升的原因是远程后端可以存放比 CPU 内存多得多的 KV cache, 从而达到更高的 cache 命中率.

但是需要注意, 由于远程后端的带宽低得多, 从远程后端加载 KV cache 比从 CPU 内存加载延迟更高. 因此, 加载延迟甚至可能超过预填充延迟, 尤其是输入上下文较短或模型较小, 导致预填充过快时. 我们稍后会在第 [8.7 节](#section-08-07) 展示这种情况. 所以, KV cache 位于远程存储服务器时, 需要在加载 KV cache 与预填充之间做自适应决策.

<span id="section-08-05"></span>

### 8.5 PD 分离

<span id="figure-12"></span>

![图 12: LMCache 与 vLLM 原生 PD 分离的 TTFT 和 ITL 累积分布.](./lmcache/figure-12.png)

**图 12.** 与 vLLM 原生 PD 分离相比, LMCache 的 PD 分离尾延迟明显更低, 平均 TTFT 低 1.5-1.8$\times$, 平均 ITL 低 1.1-1.7$\times$.

<span id="figure-13"></span>

![图 13: 异步和同步 I/O 下的查询时间线.](./lmcache/figure-13.png)

**图 13.** 将请求异步化后, LMCache 使 KV cache 加载与推理计算重叠, 推理计算可以是预填充或解码.

<span id="figure-14"></span>

![图 14: LMCache 与 vLLM PD 分离的延迟分解.](./lmcache/figure-14.png)

**图 14.** 与 vLLM 原生 PD 分离相比, LMCache 的传输延迟小得多, 因而降低了端到端延迟.

本实验评估 PD 分离场景中的性能. 我们使用官方基准测试脚本的随机输入输出负载, 将 LMCache 与 vLLM 原生 PD 分离对比. 输入为 8K token, 输出为 200 token. [图 12](#figure-12) 展示了 LMCache 与 vLLM 原生 PD 分离的第 95 百分位 TTFT, 可以看出 LMCache 的尾延迟明显更好. 在平均 TTFT 上, LMCache 也大幅优于 vLLM 原生 PD 分离. 具体而言, 在四个模型上, LMCache 将平均 TTFT 降低 1.53-1.84$\times$, 将平均 ITL 降低 1.12-1.66$\times$.

LMCache 相对基线的性能收益来自更高效的 PD 分离设计. 具体而言, LMCache 把分块预填充期间生成的每个 KV cache chunk 复制到 prefiller 实例 GPU 内存的一个 buffer, 再传到 decoder 实例上对应的 buffer. 收到后, KV cache 被复制到 decoder 实例的分页内存中.

相比之下, vLLM 原生 PD 分离使用 NIXL 的内存复制函数, 直接把 prefiller 生成的分页 KV cache 发送到 decoder. 该函数接收 prefiller 端 KV cache page 的内存地址, 并把它们复制到 decoder 端的目标地址. 但是, 如果 KV cache 的分页内存散布在 prefiller GPU 内存中, 传输就会逐 page 进行, 造成带宽利用不足, 如第 [5 节](#section-05) 所述.

### 8.6 组件级评估

为进一步理解 LMCache 带来的收益, 我们还做了组件级分析, 分解端到端系统中各个组件的延迟.

**PD 分离:** [图 14](#figure-14) 展示了 LLM 推理的延迟分解, 包括预填充与解码计算, 以及 prefiller 与 decoder 实例之间的 KV cache 传输. LMCache 和 vLLM 原生 PD 分离的预填充与解码计算时间相同. 但是, 如第 [8.5 节](#section-08-05) 所述, vLLM 原生设计以更细的粒度传输 KV cache, 造成带宽利用不足. LMCache 则使用更高效的 KV cache 传输机制, 传输速度显著提高, 从而降低 PD 分离的整体端到端延迟.

<span id="table-05"></span>

![表 5: LMCache 与 vLLM 原生卸载的 CPU 加载带宽.](./lmcache/table-05.png)

**表 5.** 从 CPU 内存加载 KV cache 时, LMCache 达到的加载带宽远高于 vLLM 原生 CPU 卸载.

**CPU 卸载:** 我们在 [表 5](#table-05) 中做了消融实验, 测试 LMCache 与 vLLM 原生 CPU 卸载从 CPU 达到的加载带宽. LMCache 的传输带宽高于 vLLM 原生 CPU 卸载, 原因在于传输粒度. 原生 CPU 卸载逐 page 移动数据, LMCache 则逐 chunk 传输数据. 每次传输操作都会触发 CUDA 内存复制, 还需要提前准备元数据, 事后发送完成信号. 这些逐传输操作会给每个内存复制 kernel 带来开销. 每次复制传输更大的数据 chunk 后, LMCache 降低了总体开销, 有效带宽因而大幅提高.

**异步计算:** 我们还展示了 LMCache 异步计算减少端到端延迟的收益. [图 13](#figure-13) 展示了查询加载和推理计算的时间线. 为了方便说明, 该图截取自一次较长运行的中段. 如图所示, 没有查询异步化时, 预填充/解码计算和加载顺序执行. 将查询异步化后, 预填充/解码计算可以与 KV cache 加载重叠, 把端到端延迟降低 $1.46\times$.

<span id="section-08-07"></span>

### 8.7 敏感性研究

我们还做了多项敏感性评估, 考察 LMCache 的延迟如何随上下文长度和远程后端类型变化.

<span id="figure-15"></span>

![图 15: 不同上下文长度和网络带宽下的预填充与 KV cache 加载延迟.](./lmcache/figure-15.png)

**图 15.** 网络带宽为 32 Gbps 时, 只有在输入长度超过 256K token 后, LMCache 的 KV cache 卸载才优于基础 vLLM 的预填充. 网络带宽为 64 或 128 Gbps 时, 在所有输入长度下, LMCache 的 KV cache 卸载都优于预填充.

**上下文长度的影响:** [图 15](#figure-15) 展示了 B200 机器上的预填充延迟, 以及不同网络带宽下的加载延迟. 网络带宽较低时, 即 32 Gbps, 只有当输入上下文长度超过 256K token 后, LMCache 的 KV cache 加载才优于朴素预填充. 相比之下, 带宽较高时, 即 64 或 128 Gbps, 在所有上下文长度下, LMCache 的加载延迟都始终低于朴素预填充. 这些结果说明 LMCache 的 KV cache 加载应该自适应: 带宽低时, 只有上下文长度超过加载快于预填充的交叉点后, 才应启用加载.

### 8.8 SGLang 结果

<span id="figure-16"></span>

![图 16: SGLang 在 LMCache 和原生 CPU 卸载下的吞吐量, 平均 TTFT 和平均端到端延迟.](./lmcache/figure-16.png)

**图 16.** 在 `Qwen3-32B` 模型上, LMCache 的 CPU 卸载与 SGLang 原生 CPU 卸载的性能相当.

我们的主要评估使用 vLLM, 但也评估了与 SGLang 集成的 LMCache. [图 16](#figure-16) 报告了在两张 H100 GPU (TP=2) 上服务 Qwen3-32B, 并启用 LMCache CPU 卸载时的结果. 与不启用 CPU 卸载的 SGLang 相比, LMCache 的吞吐量更高, 平均 TTFT 和平均端到端延迟更低. 与 SGLang 原生 CPU 卸载相比, LMCache 性能相当. 这些结果证实 LMCache 在另一种推理引擎上也有效. 虽然 SGLang 原生 CPU 卸载在 SGLang 上的性能与 LMCache CPU 卸载相当, 但它没有分布式存储后端, 无法在本地磁盘, 远程 CPU/磁盘资源等分层存储设备之间高效卸载数据.

<span id="section-09"></span>

## 9 真实世界的经验与教训

**从远程存储加载比预填充更快:** 传统观点认为, 从远程存储加载 KV cache 主要是通过使用更廉价的存储设备提高 cache 命中率并降低存储成本, 但会以提高推理延迟为代价, 因为人们认为从远程设备加载数据比做一次完整预填充更慢. 这个假设主要源于 Amazon S3 等远程对象存储历史上吞吐量很低, 加载速度最低只有 100 MBps. 不过, 近年远程存储性能大幅提高, 例如 Amazon S3 Express 的吞吐量已从 100 MBps 提高到接近 1 GBps. 公司 C 等用户已采用 LMCache 从自己的远程对象存储加载 KV cache, 与完整预填充相比, TTFT 降低 22%-32%. 这项经验说明, 远程后端可以同时提高 cache 命中率并降低 TTFT.

**上下文截断会降低前缀 cache 命中率:** 许多业界用户采用滑动窗口机制, 处理受限于模型上下文窗口或 GPU 内存的长上下文输入. 例如, 当输入 token 超出上下文窗口上限时, 一些公司会截断输入, 只保留最近的 token. 但是, 由于截断后的输入不再匹配之前缓存上下文的前缀, 这种做法会显著降低前缀 cache 命中率. 在实际运行中, 我们使用公司 F 的真实 trace 发现, 截断输入上下文, 只保留最新 token 后, 前缀 cache 命中率从约 85% 降到 45%. 其他研究也讨论过这一现象, 指出应避免动态添加或删除上下文 token, 因为这会使前缀 KV cache 复用失效 [Lm42].

**更受偏好的容器化代码:** 随着 LLM 推理规模增长, 大多数生产环境现在使用 Kubernetes 管理 GPU 集群. 因此, 通过容器化环境, 通常是 Docker 镜像, 部署 vLLM 或 SGLang 等推理引擎和 LMCache, 已成为业界用户的标准做法. 有趣的是, 许多用户只使用官方 Docker 镜像, 不会深入阅读 LMCache 源代码.

**生产系统中出乎意料的高 cache 命中率:** 在把 LMCache 部署到系统之前, 客户没有想到前缀 cache 命中率会这么高, 例如公司 G 的生产环境中达到 50%. 以前, 人们认为 KV cache 只能在固定 system prompt 中复用. 但现代应用中越来越多地出现"动态可复用上下文", 例如编程助手, 聊天应用和检索增强生成 (RAG) pipeline 中的对话历史. 这些新模式大幅提高了真实部署中的总体 cache 命中率.

**业界用户与学界用户:** 2024 年 5 月, LMCache 最初被设计为一个统一的原型框架, 我们可以将研究工作放入其中, 以获得更大影响. 但我们发现, 随着 KV cache 和并发用户规模增长, 当时的业界需要一个高效的 KV cache 卸载方案. 于是, 我们的重心转向提高性能, 稳定性和兼容性. 大多数公司不太关心定制注意力算法, 因此我们降低了为选择性 token 丢弃等特殊注意力机制设计灵活集成 API 的优先级. 这使 LMCache 在学界不那么流行, 因为学界用户的研究原型常常集中于修改注意力机制. 下一步, LMCache 将设计更灵活的 API, 让业界和学界都能方便使用.

**编程语言的灵活性与性能:** Python 一直是 ML 中的事实标准语言. 但当前业界的重心正从广泛兼容逐步转向更高效率. 具体而言, 许多公司正在用 Rust 或 C++ 等高性能语言重写 ML 库, 或仔细优化基于 Python 的系统, 在保留灵活性的同时隐藏运行时开销. 虽然一些替代方案已使用 Rust 编写 ML 库, 我们仍然使用 Python, 并做了精心设计的优化. 这种方式让我们可以更快演进, 获得更多社区贡献, 同时保持与替代方案相当的性能.

**LMCache 如今是一项社区工作:** LMCache 能从研究原型快速演变为广泛采用的业界框架, 一个重要原因是社区贡献者的积极参与. 大约一年前, LMCache 只支持与 NVIDIA GPU 上 vLLM 集成的本地 CPU, 本地磁盘和 Redis 后端. 如今, 它在四种处理器, 即 NVIDIA, AMD, Ascend 和 TPU 上, 支持了八个新存储后端, 即 NFS, WEKA, GPU-Direct Storage, Mooncake Store, NIXL, S3, InfiniStore 和 Valkey, 以及两种推理引擎 vLLM 和 SGLang. 这些贡献全部来自业界合作伙伴, 他们积极将代码提交到上游, 以便与持续开发保持一致, 避免偏离 LMCache 的最新更新.

## 10 结论与展望

本文介绍了 LMCache, 这是第一个开源, 也是采用最广泛的生产就绪型 KV cache 层, 面向企业规模的 LLM 推理. LMCache 把 KV cache 当作一等数据结构, 而不是推理的内部副产品, 从而将 LLM 引擎从孤立的 token 处理器转变为分布式的计算与存储生态系统. 对多种负载和模型的评估表明, 与开源基线和商用推理 API 相比, LMCache 一直能大幅提高吞吐量, 降低延迟.

除性能之外, LMCache 已在生产环境中快速普及. 企业通过它的 CPU 卸载, 分层存储和 PD 分离能力, 使万亿 token 规模部署保持低延迟并降低成本. 真实部署还显示出了新机会, 例如推荐系统中的 KV cache 复用和开放式聊天机器人中的有损压缩, 说明 LMCache 可以用于多种应用领域.

展望未来, LMCache 指向一个更广泛的转变: **KV cache 等 AI 原生数据将日益成为扩展 LLM 推理和智能体负载的基础**. LMCache 把 KV cache 建立为标准化存储与通信介质, 为未来的系统奠定基础, 让它们不再把推理当作孤立会话, 而是当作持久, cache 感知的计算网络. 我们希望本文介绍的设计, 优化和部署经验能为下一代 LLM 基础设施提供参考. 在这样的基础设施中, KV cache 等 AI 原生数据不只是一种优化, 更是高效, 可靠, 可扩展推理的核心原语.

LMCache 的源代码见 [https://github.com/LMCache/LMCache](https://github.com/LMCache/LMCache).

## 11 致谢

我们感谢 LMCache 社区的宝贵支持与贡献, 包括负责管理远程 connector 的 Baolong Mao 和 Chunxiao Zheng, 负责 GitHub 基础设施的 Martin Hickey, 撰写并维护文档的 Huaizheng Zhang, Siddhant Ray, Zhuohan Gu 和 Hanchen Li, 以及提供深刻反馈的 Qizheng Zhang 和 Hussain Mohammad. 我们也感谢 GMI Cloud 提供 GPU 集群, 供我们运行实验.
