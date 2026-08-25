---
title: 'HiSparse: Hierarchical KV Cache'
createTime: 2026/08/25 11:34:32
permalink: /papers/hisparse/
---

> [Zhiqiang Xie](https://zhiqiangxie.com/) [+corresponding-author], [Zhangheng Huang](https://github.com/hzh0425), [Tingwei Huang](https://github.com/huangtingwei9988), [Ziyi Xu](https://orcid.org/0009-0000-4411-9773), [Ruiyang Ma](https://orcid.org/0009-0003-9067-9538), [Christos Kozyrakis](https://kozyraki.github.io/). 论文于 2026 年 8 月 7 日首次提交至 arXiv, 当前版本为 v1. [HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv.org/abs/2608.07009v1). [原始 PDF](/paper/hisparse.pdf). [DOI](https://doi.org/10.48550/arXiv.2608.07009). [TeX 源码](https://arxiv.org/src/2608.07009v1). 精确的印刷版式和参考文献以原始 PDF 为准.

[+corresponding-author]: *通讯作者: `xiezhq@cs.stanford.edu`.*

## 摘要

Top-$k$ 稀疏注意力降低了长上下文 LLM 解码的计算成本: 每一步只读取几千个选中的 KV 条目, 而非完整上下文. 但服务系统通常会把整个 KV cache 保留在 GPU HBM 中, 以保证每个位置都可被选中, 因此请求的内存成本仍随完整上下文长度增长, 解码会在计算资源耗尽前很久撞上容量墙, KV cache 超过 HBM 的上下文甚至完全无法提供服务. 我们提出 HiSparse, 一种用于稀疏注意力服务的精确、与索引器无关的分层 KV cache. HiSparse 在主机内存中保留每个请求的完整 KV 历史, 并用一个小型定长 GPU cache 限制其解码占用; 一个融合 CUDA kernel 在解码 CUDA graph 内完成每层选择结果的解析, 包括命中检测、LRU 替换和从主机到设备的获取; 对于跨层共享选择结果的模型, 精确的逐层预取还能隐藏约一半的剩余未命中开销. 由于改变的只有 KV 放置位置, 模型输出保持不变. HiSparse 已合入上游 SGLang, 并在 H200、B200 和 GH200 平台上用三类稀疏注意力 (DSA、NSA 和 Quest) 进行评估: 在长上下文工作负载中, 它将峰值生成吞吐量最高提高 $4.7\times$, 同时保持相近的单 token 延迟, 并降低高负载下的首 token 时间; no-IO oracle 则表明解析机制本身不会增加可测量的单 token 成本, 有界驻留付出的唯一代价是主机与设备之间的 IO.

<span id="section-1"></span>

## 1 引言

长上下文推理正在成为标准的 LLM 工作负载: 编程 agent 检查整个代码仓库, 助手综合长文档中的信息, 近期模型则把上下文窗口扩展到数万乃至数百万 token [Dee26, Zen26, Qwe25a]. 为这些上下文提供服务仍然代价高昂, 因为每个正在解码的请求都携带一个随完整历史增长的 KV cache.

Top-$k$ 稀疏注意力为扩展长上下文提供了一条可行路径. 每个解码步骤不再关注此前的所有 token, 而只关注一小组随查询变化的 $k$ 个 KV 条目, 通常是几千个 token, 比这些模型面向的上下文少一到两个数量级. DeepSeek-V3.2 表明, 学习得到的 top-$k$ 选择可以保持模型质量, 同时显著降低长上下文注意力成本 [Dee25a, Dee25d]; 同样的模式也见于 DeepSeek Sparse Attention (DSA) 和 Native Sparse Attention (NSA) [Yua25e] 等训练式架构, 以及 Quest [Tan24] 等免训练选择器. 选中集合确定后, 注意力 kernel 只需读取 $k$ 个 KV 条目, 而不是完整上下文, 因此长上下文解码理应能以低得多的成本提供服务.

然而, 稀疏注意力并没有缩小 KV 容量瓶颈. 选中集合会随生成的 token 和层而变化, 当前跳过的条目之后可能被选中, 所以服务系统通常会把完整 KV cache 驻留在 GPU HBM 中, 让每个逻辑位置都可寻址. 这造成了失衡的服务成本: 每个解码步骤只读取 $k$ 个条目, 完整上下文中的每个条目却都要占用 HBM, 只因为它*可能*被读取, 注意力的成本降低了几个数量级, 内存账单却一个字节也没有减少. 即使采用压缩 KV 布局, 这笔成本也很可观: 一个 $128$K-token 的 GLM-5.1 请求包含 $13.09\,\mathrm{GB}$ 的 BF16 KV 状态, 单个 $1$M-token 请求就会占用整张 H200, 在计算权重之前便需要其 $141\,\mathrm{GB}$ HBM 中的 $100\,\mathrm{GB}$ 以上; 权重驻留后, 该请求根本无法准入, 可服务的上下文因此远小于模型窗口. 实际上, 容量墙受上下文长度与并发数的*乘积*约束: 每个请求包含 $32$K token 时, 几十个并发请求便会耗尽 HBM ([图 1](#figure-01)); 到 $128$K 时, 同样的 HBM 只能容纳四分之一的请求. 因此, 长上下文稀疏解码会在注意力计算资源耗尽前很久先耗尽内存容量.

<span id="figure-01"></span>

![HiSparse 分层结构下的解码吞吐量和首 token 时间](./hisparse/figure-01.png)

**图 1.** HiSparse 将长上下文工作负载中的解码吞吐量与 GPU 内存容量解耦. **(a)** 解码吞吐量与并发数: HBM 饱和后, baseline 的吞吐量进入平台期, HiSparse 则继续扩展. 上下文越长, 在越低的并发数下就会撞上同一容量墙. **(b)** PD-colocated 模式下的平均 TTFT 与吞吐量: HiSparse 在更高吞吐量下仍保持较低 TTFT. GLM-5.1-FP8 (DSA, $k=2048$), $8\times$H200, $32$K 输入, $8$K 输出.

[图 1](#figure-01) 展示了这种容量约束对服务的影响: HBM 被占满后, 解码吞吐量进入平台期; 在 colocated 服务中, 解码 KV cache 挤占 prefill 工作所需的空间, 首 token 时间 (TTFT) 随之上升. 然而, full-KV 服务忽略了一个免费的 oracle. 在每个解码步骤的每一层, 稀疏选择器都会准确给出注意力将读取的 $k$ 个位置, 这为 KV 放置提供了精确的逐步需求信号, 而密集注意力系统从未拥有过这种信息. 这种需求还有明确结构: 连续解码步骤会重新选择大量重叠位置 [Che25aa], 相邻层也会选择相关位置; 近期模型通过跨层共享索引器输出来明确利用这种局部性 [Bai26, Glm26]. 稀疏选择的行为类似具有强时间局部性的内存访问, 很适合用小型高速 cache 配合更大、更慢的后备层. 这一观察促成了 HiSparse, 一个用于 top-$k$ 稀疏注意力服务的精确分层 KV cache 系统: HBM 应当为注意力实际读取的内容付费, 而不是为所有可能读取的内容付费.

HiSparse 在主机内存中保留每个请求的完整 KV 历史, 并在 HBM 中为其分配一个小型定长 *GPU cache*. 每层的 top-$k$ 选择结果会在注意力 kernel 启动前对照该 cache 进行解析: 已驻留的条目直接使用, 少量未命中条目则通过一次批量传输从主机副本获取. 选中位置、注意力分数和输出均不改变; 由于 HiSparse 只使用每层发出的选中位置, 它与*索引器无关*: 无需重新训练或修改模型, 即可置于 DSA、NSA 或 Quest 之下. 因此, 请求的解码期 HBM 占用随 GPU cache 大小而增长, 不再随上下文长度增长.

HiSparse 的核心问题是让这套层级结构避开解码关键路径, 其设计贡献应对了三个挑战. 第一, *保留哪些内容*: 只暂存当前 top-$k$ 集合会丢弃局部性, 在一条 LongBenchV2 选择 trace 上, 每一步的选择有 $30\%$ 未命中, 因此 HiSparse 用 LRU 管理 cache, 在 cache 大小为 top-$k$ 两倍时, 将选择局部性转化为 $87\%$ 的命中率 ([第 4.3 节](#section-4-3)). 第二, *隐藏剩余的未命中*: 每次未命中都会让该层注意力等待主机内存获取, batch 较大时, 并发请求的未命中还会争用同一条主机链路 ([第 4.4 节](#section-4-4)); HiSparse 使用 GPU-assisted IO 提高每次获取的带宽效率, 并在模型跨层共享选择结果时, 通过精确预取把剩余传输与中间层的计算重叠 ([第 3.5 节](#section-3-5)). 第三, *降低解析本身的成本*: 每个解码步骤的每个稀疏层都要重复执行命中检测、victim 选择、元数据更新和主机获取, 因此 HiSparse 把它们融合到解码 CUDA graph 内的一个 CUDA kernel 中 ([第 3.4 节](#section-3-4)).

HiSparse 并非研究原型: 我们的实现已经合入上游 SGLang [She24], 这一框架已被广泛部署, HiSparse 也作为受支持的服务功能发布 [Xie26a, Sgl26a] ([第 8 节](#section-8) 说明了集成细节). 我们在三种硬件平台 (H200、B200 和 GH200) 上评估了三类稀疏注意力 (DSA、NSA 和 Quest). HiSparse 在长上下文工作负载上将峰值生成吞吐量最高提高 $4.7\times$, 在重叠的吞吐量区间保持相近的单输出 token 时间 (TPOT), 降低高负载下的 TTFT, 且不会改变模型输出. 所有收益都来自同一个杠杆: 有界驻留让 HiSparse 能在相同 HBM 中运行大得多的解码 batch. 这个杠杆也可以向另一个方向使用, 用显著更少的 HBM 服务固定 batch ([第 4.2 节](#section-4-2)), 或服务仅靠 HBM 根本无法容纳的上下文 ([第 5 节](#section-5)).

本文作出以下贡献:

- 我们指出了长上下文 top-$k$ 稀疏注意力服务的容量墙: 活跃 KV 读取量随 $k$ 增长, 驻留 HBM 的 KV 占用仍随完整上下文长度增长 ([第 2 节](#section-2)).
- 我们提出 HiSparse, 一种精确且与索引器无关的分层 KV cache, 在主机内存中保持完整 KV 状态可用, 同时用定长 GPU cache 限制每个请求的解码 HBM 占用 ([第 3 节](#section-3)).
- 我们设计了保持局部性的未命中解析路径: LRU 管理把选择局部性转化为 cache 命中, 逐层预取隐藏剩余的未命中延迟, 融合 CUDA resolve kernel 则降低解码关键路径上的解析成本 ([第 3.4 节](#section-3-4), [第 3.5 节](#section-3-5)).
- 我们在 H200、B200 和 GH200 平台上评估了 DSA、NSA 和 Quest 工作负载, 展示了长上下文吞吐量的大幅提升, 并分析决定性能的 cache policy、kernel 和主机-设备带宽权衡 ([第 4 节](#section-4)).

<span id="section-2"></span>

## 2 背景与动机

<span id="section-2-1"></span>

### 2.1 Top-$k$ 稀疏注意力

Top-$k$ 稀疏注意力不再对整个上下文执行完整注意力, 而是关注一小组随查询变化的 key. 在解码步骤 $t$, 一个*索引器*产生选中集合 $\mathcal{S}_t \subseteq \{1,\dots,L_{\mathrm{ctx}}\}$, 其中 $|\mathcal{S}_t| = k$, 注意力 kernel 只读取对应的 key 和 value 条目. 对模型的稀疏注意力规则而言, 由此得到的稀疏注意力仍然精确: $\mathcal{S}_t$ 一旦确定, 未选中的条目就不参与当前注意力计算.

近期系统的主要差异在于如何产生 $\mathcal{S}_t$ ([表 1](#table-01)). DeepSeek Sparse Attention (DSA) 引入了与 backbone 一同训练的 "lightning indexer", 并用于 DeepSeek-V3.2 和 GLM-5.1 [Dee25a, Dee25d, Zen26, Xie26a]: 它为每个 token 保留一个紧凑的 indexer key, 先用轻量 kernel 为完整历史评分, 再在 token 粒度取 top-$k$. Native Sparse Attention (NSA) 在可训练架构中结合压缩、选择和滑动窗口分支; 其选择分支为每个 token block 保留一个压缩 key, 并根据 block 分数取 top-$k$ [Yua25e]. Quest 是面向预训练密集模型的免训练 page-granular 选择器: 它维护每页 key 的 min/max 摘要, 并选取 query-aware upper bound 最高的页面 [Tan24]. 因此, 这些选择器在维护的状态、执行的评分和选择粒度上有所不同, 粒度分别可以是 token、block 或 page.

<span id="table-01"></span>

![随查询变化的 top-k 选择器对比](./hisparse/table-01.png)

**表 1.** 随查询变化的 top-$k$ 选择器. 每种选择器都维护紧凑且驻留 HBM 的选择状态, 并输出逻辑 token 位置; 注意力读取的 KV 记录占据主要内存, HiSparse 对它们进行分层管理.

尽管存在这些差异, 三种选择器仍共享定义统一系统接口的三项性质. 第一, 用于*选择*的状态很紧凑: indexer key、block key 和 page 摘要都远小于注意力读取的 KV 记录, 即使上下文很长, 选择状态也可以驻留 HBM. 第二, 选择会在注意力 kernel 接触任何 KV 记录*之前*完成, 因而在选择和读取之间形成自然的介入点. 第三, 各种方法的输出形式相同: 每层一组逻辑 token 位置, 对于步骤 $t$ 的第 $\ell$ 层, 记为 $\mathcal{S}_t^{(\ell)}$. 每个稀疏层通常选择自己的集合, 但近期模型会让连续的多层*共享*某一层的选择结果, 以摊销索引器成本 [Bai26, Glm26], HiSparse 随后会利用这一设计进行预取 ([第 3.5 节](#section-3-5)). HiSparse 正是建立在这些性质之上 ([第 3 节](#section-3)): 它不改变 GPU 上的选择状态与计算, 在位置接口处介入, 只对体积较大的 KV 记录应用分层放置, 因而与索引器无关.

<span id="section-2-2"></span>

### 2.2 KV cache 与容量墙

在自回归解码期间, 之前每个 token 的 key 和 value 都作为 KV cache 保留下来. Top-$k$ 稀疏注意力缩小的是单步*读取*的范围, 而非必须保持*可用*的范围: $\mathcal{S}_t$ 从完整历史中选取并随步骤变化, 当前跳过的条目之后可能被选中, 因此服务系统通常会让整个 cache 驻留 HBM, 使索引器和注意力后端可以寻址每个位置 ([第 1 节](#section-1)). 由此产生的准入约束就是容量墙的定量形式: 上下文长度为 $L_{\mathrm{ctx}}$ 的 $N_{\mathrm{batch}}$ 个解码请求, 必须在模型权重占用后剩余的 HBM 中容纳 $N_{\mathrm{batch}} \times L_{\mathrm{ctx}}$ 个 token 的 KV 状态, 而每个解码步骤的注意力只读取其中 $N_{\mathrm{batch}} \times k$ 个 token.

两种常见部署模式都会撞上这堵墙. 增加并发请求只能在 KV 存储填满 HBM 前提高吞吐量; 此后即使注意力 kernel 仍有计算余量, 调度器也无法准入更多解码工作, 这就是 [图 1(a)](#figure-01) 中的平台期. 在 *PD-disaggregated* 服务中, prefill 和 decode 运行在不同 GPU pool 上, HBM 容量直接限制 decode pool 的吞吐量. 在 *PD-colocated* 服务中, prefill 和 decode 共享 GPU 与 HBM: 完整上下文的解码 KV cache 会占用 prefill chunk 所需的内存, 因此新请求必须同时等待内存和 decode slot 才能产生第一个 token, 平均 TTFT 主要由排队时间决定, 并随负载快速上升 ([图 1(b)](#figure-01)).

我们的 GLM-5.1 部署 ([第 4.1 节](#section-4-1): $8\times$H200, 合计 $1.1\,\mathrm{TB}$ HBM) 给出了具体数字. 对于 $32$K 输入 / $8$K 输出, 每个请求最多持有 $4\,\mathrm{GB}$ KV 状态, full-KV baseline 在约 $60$ 个并发请求时饱和, 即约 $240\,\mathrm{GB}$ KV, 这是权重、activation 和 CUDA graph 状态驻留后剩余的 HBM ([图 1(a)](#figure-01)). 在 colocated 模式下, TTFT 从这一饱和点开始上升; 在 disaggregated 模式下, 无论前方有多少 prefill 容量, 同样硬件组成的 decode pool 都会被限制在相应的 decode-only 速率, 即测量中的 $777$ tokens/s. 上下文达到 $128$K 时, 同样的容量只能容纳约 ${\sim}15$ 个请求.

<span id="section-2-3"></span>

### 2.3 将可用性与驻留解耦

容量墙源于把两个不必相同的要求绑在一起. 模型需要每个过去的 KV 条目在逻辑上*可用*, 因为未来的 top-$k$ 选择可能引用历史中的任何位置. 但 GPU 只需要当前步骤实际读取的条目. Full-HBM 服务把 "之后可能被选中" 当作 "现在必须留在 HBM 中", 这种做法简单, 但约束过强. 服务系统只需保证索引器发出 $\mathcal{S}_t^{(\ell)}$ 后, 选中的 KV 条目在该层注意力运行前位于设备上; $\mathcal{S}_t^{(\ell)}$ 之外的条目可以驻留在其他位置, 而不改变选中位置、注意力分数或输出.

但是, 单纯解耦并不能让 offloading 可行, CPU-GPU 互连会立即成为瓶颈. 如果每一步都要从主机内存获取每层选中的 $k$ 个条目, 一个 GLM-5.1 请求每生成一个 token 就要移动约 $200\,\mathrm{MB}$ KV 记录 ($k=2048$, 各层合计每个 token 约 ${\sim}100\,\mathrm{KB}$ KV), TPOT 为 $30\,\mathrm{ms}$ 时, 每个请求需要约 $7\,\mathrm{GB/s}$ 的持续 host-to-device 流量, 因此每张 GPU 上十几个并发请求产生的未命中就会占满 PCIe Gen5 $\times16$ 链路 (每个方向约 ${\sim}64\,\mathrm{GB/s}$). 弥合这一差距的是选择本身的局部性, 即 [第 1 节](#section-1) 所述的结构: 连续解码步骤会重新选择大量重叠位置 [Che25aa], 相邻层也会选择相关位置, 更新的模型则通过跨层共享索引器输出来明确利用这一点 [Bai26, Glm26]. 在一条 LongBenchV2 选择 trace 上, 一个大小适中的 LRU cache (top-$k$ 大小的两倍) 能让 $87\%$ 的选择在设备上命中 ([第 4.3 节](#section-4-3)). 这种局部性支撑了整个设计: cache 有效是因为最近选中的记录大多会再次被选中, 预取有效则是因为后续选择可以预测; 使用共享索引器时, 后续选择甚至可以提前确知.

要在不损失这些收益的前提下完成解耦, 系统需要在保留上述局部性的同时限制每个请求的 cache, 用计算隐藏剩余未命中延迟, 并通过只使用输出位置的接口完成这两点, 这些正是 [第 1 节](#section-1) 概述的挑战. 下一节给出满足这些要求的设计, [第 4 节](#section-4) 则量化各项机制.

<span id="section-3"></span>

## 3 HiSparse 设计

<span id="section-3-1"></span>

### 3.1 设计目标与不变量

HiSparse 实现了 [第 2.3 节](#section-2-3) 提出的解耦: 每个 KV 条目对稀疏注意力算法仍在逻辑上可用, 但 GPU 内存中只驻留一个有界 working set. 以下目标和不变量指导了系统设计, [表 2](#table-02) 定义了全文使用的符号.

<span id="table-02"></span>

![HiSparse 设计中使用的符号](./hisparse/table-02.png)

**表 2.** 符号.

**完整 KV 可用性.** 对每个活跃请求, HiSparse 都在解码 GPU 的 HBM 外保留完整 KV cache 副本. 因此, 稀疏索引器选中的任何逻辑 KV 位置都可以恢复, 无需重新计算.

**有界设备占用.** 对每个请求和每一层, HiSparse 在 HBM 中预留一个定长 *GPU cache*, 容量为 $B$ 个逻辑 KV-record slot, 用于缓存该请求在该层最近选中的 KV 记录. $B$ 是服务配置参数, 满足 $B \ge k$, 且与 $L_{\mathrm{ctx}}$ 无关. 如果模型每个 token 每层存储 $W_{\mathrm{KV}}$ 个元素, 那么解码侧 KV 占用按 $N_{\mathrm{batch}} N_{\ell} B W_{\mathrm{KV}} s$ 增长, 而非 $N_{\mathrm{batch}} N_{\ell} L_{\mathrm{ctx}} W_{\mathrm{KV}} s$, 另有少量元数据.

**精确的稀疏注意力输出.** 在某层的稀疏注意力 kernel 运行前, 该层选中集合 $\mathcal{S}_t^{(\ell)}$ 中的所有 KV 条目都必须在设备上物化. HiSparse 可以改变未选中 KV 条目的驻留位置, 但不会改变选中位置、注意力分数或注意力输出.

**与索引器无关的接口.** HiSparse 不假设选中集合如何产生. DSA、NSA 和 Quest 可以使用不同的索引器, HiSparse 只使用每个请求和每一层输出的选中位置.

**让未命中延迟避开关键路径.** 限制驻留不能以吞吐量换取单 token 延迟. 选中集合未命中会给每个稀疏注意力层增加工作, 因此 HiSparse 直接处理这项成本: cache 管理保留 [第 2.3 节](#section-2-3) 所述的选择局部性, 使大多数选择命中 ([第 3.2 节](#section-3-2)); 剩余部分只需启动一次融合 kernel 即可解析 ([第 3.4 节](#section-3-4)); 预取则将 host-to-device 传输与较早层的计算重叠 ([第 3.5 节](#section-3-5)).

<span id="section-3-2"></span>

### 3.2 KV 层级结构与元数据

HiSparse 把 KV 状态组织为两级层次结构 ([图 2](#figure-02)). **Host KV pool** 分配在 pinned host DRAM 中, 存储活跃请求的权威完整 KV cache. 在 colocated 服务中, prefill 写入本地 host pool. 在 disaggregated 服务中, prefill 通过 prefill-decode 传输路径把 KV 状态发送到 decode instance 的 host pool.

解码 GPU 只存储 **GPU cache**, 它在 [第 1 节](#section-1) 引入, 并在 [图 2](#figure-02) 和 [图 3](#figure-03) 中标为 "hot device buffer". 从概念上看, 每个请求和每一层都有一个包含 $B$ 个 slot 的 cache, 每个 slot 保存一个逻辑位置在该层的 KV 记录. 这个 cache 不只是 top-$k$ 暂存区: 当前步骤选中的条目必须在注意力运行前就位, 其余 $B-k$ 个 slot 则保留近期使用、可能在后续选择中命中的记录. 因此, 每个请求和每层驻留的记录数不超过 $B$, cache 预热后通常大于 $k$; 必须满足 $B \ge k$, 以保证当前选择总能放入. **Page table** 把每个层内逻辑位置映射到 cache slot, 或映射到表示该记录仅在主机上的 sentinel. **LRU 元数据** 记录驻留 slot 的最近使用顺序, 并在选中条目未命中时直接决定替换对象. 与 KV tensor 相比, 元数据很小, 但每个稀疏层都必须在注意力 kernel 启动前查询并更新它; 这部分延迟会直接进入每个解码步骤, 因而 HiSparse 让元数据驻留 GPU, 并把更新融合到 [第 3.4 节](#section-3-4) 的 kernel 中.

选择状态则从不移动. [第 2.1 节](#section-2-1) 所述的紧凑索引器状态, 包括 DSA 的 per-token indexer key、NSA 的压缩 block key 和 Quest 的 page 摘要, 会在请求的整个生命周期内驻留 GPU; HiSparse 的 page table 与 LRU 元数据也是如此, HiSparse 不会把它们 page 到主机内存. 这类常驻状态仍随上下文长度增长, 但其单 token 成本比 KV 记录低两到三个数量级: indexer key 和 page-table entry 合计每个 token 最多占几百字节, KV 记录则约为 ${\sim}100\,\mathrm{KB}$ ([第 2.3 节](#section-2-3)). 在我们的部署中, 这些状态合计为数百 MB, 与它在 HBM 中替代的每请求数 GB KV cache 相比很小. 层级之间只移动注意力 KV 记录, 索引器自身的计算不变.

**替换策略.** 每个 request-layer cache 独立使用 LRU 管理, 并作一处有意调整: 在同一步内, *命中*条目在最近使用顺序中排在新获取的未命中条目之前. 因此, 跨步骤反复选中的记录会排在首次选中的记录之前; 出现淘汰压力时, 一次性选择会先离开, cache 会逐渐积累已表现出跨步骤复用的条目. [第 4.3 节](#section-4-3) 验证了这一选择: 最近使用顺序是未来稀疏选择的良好在线代理, 其趋势接近离线 Bélády 最优策略. 每层 cache 独立管理: KV 记录本来就属于特定层, 跨层协调驻留相比 LRU 已捕获的信息几乎没有额外收益 ([第 4.6 节](#section-4-6)); HiSparse 转而通过预取利用跨层选择结构 ([第 3.5 节](#section-3-5)).

**Cache 大小.** $B$ 在容量与延迟之间进行权衡. 解码 KV HBM 总量为 $N_{\mathrm{batch}} N_{\ell} B W_{\mathrm{KV}} s$, 较大的 $B$ 会提高命中率, 但也会占用本可准入更多并发请求的 HBM, 并延长 resolve kernel 的元数据扫描. 实际上, $B \in [2k, 4k]$ 是有效范围 ([第 4.4 节](#section-4-4)), 更快的 host-device 链路会把最优点推向 $B=2k$ ([第 4.5 节](#section-4-5)).

<span id="figure-02"></span>

![HiSparse 分层 KV cache 概览](./hisparse/figure-02.png)

**图 2.** HiSparse 概览. 主机内存保存每个活跃请求的权威完整 KV cache; GPU 只保留紧凑状态 (索引器状态、带 page-table 和 LRU 元数据的 per-request-per-layer GPU cache), 并执行所有计算. (1) Prefill 把每一层的 KV 记录写入 host pool. Decode 期间, (2) 稀疏索引器输出选中的逻辑位置; 融合 Resolve kernel 查询 GPU cache, (3) 在淘汰 LRU victim 的同时从 host pool 获取未命中记录, (4) 把物理 cache slot 交给稀疏注意力后端. (5) 新生成 token 的 KV 记录以 write-through 方式写入 host pool.

<span id="section-3-3"></span>

### 3.3 请求生命周期

一个请求会经历 [图 2](#figure-02) 所示的四个阶段. PD-colocated 与 PD-disaggregated 部署中的流程相同, 区别只在 prefill KV 如何到达 host pool, 即在本地写入, 或通过 prefill-decode 传输路径发送 ([第 3.2 节](#section-3-2)).

**(1) Prefill 和暂存.** Prefill engine 使用模型的常规 prefill 路径处理 prompt. 每一层产生 KV 状态后, HiSparse 将其写入 host KV pool. 每个解码步骤都要查询的紧凑索引器状态, 例如 DSA 的 lightning-indexer 表示, 仍留在设备上.

**(2) 准入.** 当请求的 host KV 状态可用, 且 HiSparse 已预留 per-layer GPU cache 和元数据后, 该请求就可以调度执行 decode. 每个请求预留的 KV 容量为 $N_{\ell} B W_{\mathrm{KV}} s$, 与配置的 cache 大小 $B$ 成正比, 而 $B$ 只是 $k$ 的一个小倍数; 它不再是 $N_{\ell} L_{\mathrm{ctx}} W_{\mathrm{KV}} s$, 因此长上下文请求的解码 HBM 不再随完整历史长度增长 (对于 $B=4096$ 的 GLM-5.1, 每个请求约 $0.4\,\mathrm{GB}$, 而 $128$K 上下文时原本为 $13.09\,\mathrm{GB}$, 减少约 ${\sim}30\times$).

**(3) 逐层解码.** 在解码步骤 $t$ 的第 $\ell$ 层, 索引器输出其选中集合. 未命中解析路径检查哪些选中逻辑位置已驻留, 从 host pool 获取缺失的层内 KV 记录, 更新 page table 和 LRU 状态, 并输出与选中位置对齐的密集物理设备 slot 向量. 解析完成后, 该层的稀疏注意力 kernel 才运行. 这是同步路径, [第 3.5 节](#section-3-5) 会把部分获取工作与前面层的计算重叠.

**(4) 新生成 KV 的 write-through.** 每个新生成 token 的 KV 记录会直接写入请求 GPU cache 的预留 slot, 因而最新位置始终驻留, 可供后续选择. 一个专用 backup stream 随后把记录写入 host pool, 该操作与下一步计算重叠, 并通过 event 排序, 确保后续任何获取引用该记录前, 后备副本已经完成.

<span id="section-3-4"></span>

### 3.4 融合未命中解析 kernel

未命中解析位于每个稀疏注意力层的关键路径上. 给定某层的选中集合, HiSparse 必须识别命中, 为未命中选择 victim, 获取缺失 KV 记录, 更新元数据, 并把物理设备 slot 返回给注意力后端. 如果把这些操作拆成多个 CUDA launch, 每一层都会反复把临时状态物化到 HBM, 并增加 launch 延迟. 中间状态也紧密耦合: hit/miss mark、victim assignment、LRU update 和输出 device-location vector 都依赖相同的选中集合与 resident-buffer 元数据. 因此, HiSparse 使用单个融合 CUDA kernel Resolve 完成未命中解析; 每个稀疏层启动一次, 由一个 CUDA block 处理一个请求的工作项 ([图 3](#figure-03)), 并捕获在 SGLang 的 steady-state decode CUDA graph [She24] 中, 我们的实现也集成于此. 由于解析只使用输出的逻辑位置和 HiSparse 自己的元数据, 同一个 kernel 可以处理 DSA、NSA 和 Quest, 输入逻辑索引, 输出物理 slot, 类似软件管理的 TLB.

<span id="figure-03"></span>

![HiSparse 融合未命中解析 kernel 的五个阶段](./hisparse/figure-03.png)

**图 3.** 融合未命中解析 kernel. 对一个请求和一层, Resolve 首先为选中的逻辑位置构建 shared-memory hash table. 随后, 它并行查询每个 GPU-cache slot, 把驻留条目标记为命中或淘汰候选, 对这些标记执行 parallel scan 以选择 victim 并更新 LRU 元数据, 从 pinned host memory 获取缺失 KV 记录, 最后按 top-$k$ 顺序输出供稀疏注意力后端使用的物理设备 slot.

**阶段 1: 暂存选中位置.** 线程协作把选中位置加载到 shared-memory hash table. 此后 kernel 可以快速执行选中集合成员测试, 无需反复从 HBM 读取 top-$k$ 向量.

**阶段 2: 标记 GPU-cache slot.** 如 [图 3](#figure-03) 所示, 线程针对每个 GPU-cache slot 当前存储的逻辑位置查询 hash table. Kernel 为每个 slot 写入紧凑标记: 驻留条目在当前选中集合中则为 hit, 否则为 evictable.

**阶段 3: 扫描标记并更新 LRU.** Kernel 随后对每个 slot 的标记执行 parallel scan. Scan 压紧可淘汰 slot, 为缺失的选中位置挑出足够的 victim, 并生成操作后 buffer 状态的新版 LRU 元数据. 命中 slot 会保留并提升到 most-recently-used 端, victim slot 分配给未命中, 新获取的未命中排在命中之后, 遵循 [第 3.2 节](#section-3-2) 的替换策略.

**阶段 4: 获取缺失 KV 记录.** 负责未命中的线程把对应的层内 KV 记录从 pinned host pool 复制到已占用的设备 slot. HiSparse 采用 Strata [Xie25a] 的 GPU-assisted IO 技术: GPU 线程不暂存 DMA copy, 而是直接对 pinned host memory 发出 vectorized non-coherent load (`ld.global.nc.v2.b64`), 以适应 cache miss 的分散地址, 并降低 PCIe 与 NVLink-C2C 系统上的 transaction 开销 [Nvi26a]. Per-thread transfer block size 经过调整, 使碎片化 miss read 仍能接近链路带宽.

**阶段 5: 发布注意力输入.** 最后, kernel 更新 page table, 并输出 `top_k_device_locs`, 即与选中逻辑位置对齐的密集物理设备 offset 向量. 下游稀疏注意力 gather 随后可以直接从 GPU cache 读取选中 KV 记录.

<span id="section-3-5"></span>

### 3.5 逐层预取

即使使用融合 resolve kernel, GPU cache 中未命中的选中条目仍可能暴露主机内存延迟. HiSparse 通过跨层预取隐藏这项成本, 目标是通过复用索引器输出来明确利用跨层选择局部性的模型. IndexCache [Bai26] 把层划分为我们所称的 *anchor* 层和 *shared* 层, 前者运行 top-$k$ 索引器, 后者复用此前 anchor 的选择结果; GLM-5.2 以 IndexShare 的形式原生采用该设计, 每四层共享一个索引器 [Glm26]. 对这类模型, HiSparse 无需猜测: anchor 层一旦输出选中集合, 同组每个 shared 层的选中位置便已知, 此时距其注意力运行还有数层.

**利用共享选择结果进行精确预取.** HiSparse 使用 *plan-then-IO* 方案: anchor 的 Resolve 还会记录 miss plan, 即哪些主机记录要移动到哪些 cache slot; side stream 上的 copy-only kernel 随后把该计划重放到每个 shared 层的 cache, 让传输与中间层的计算重叠. 每个 shared 层的 cache 都与 anchor 的 slot 布局同步变化, 因而 shared 层可以直接复用 anchor 的 slot table: 它等待 prefetch-completion event 后完全跳过解析, 不查询、不更新 LRU、不执行同步主机内存加载, 也不浪费推测流量. 复制过程复用 demand path 的 GPU-assisted IO ([第 3.4 节](#section-3-4)), 所以分散的 prefetch read 可以接近链路带宽; 这一点很重要, 因为预取只是把 IO 提前, 而没有消除 IO, 它仍会使用 demand miss 所在的同一条主机链路 ([第 4.5 节](#section-4-5)).

**一种推测式替代方案.** 对于不共享索引的模型, 我们也探索了推测式变体, 用第 $\ell$ 层的选中位置提示第 $\ell+1$ 层; 相邻层往往选择重叠位置, 错误提示只会浪费传输, 不会影响正确性. 该方案在评估中几乎没有带来端到端收益 ([第 4.6 节](#section-4-6)): LRU 管理的 GPU cache 已经捕获了大部分隐式跨层复用, 因此提示的记录通常已经驻留, 剩余未命中恰恰是提示未能预测的部分. 这一负面结果说明, 要隐藏未命中延迟, 正确方向是共享索引的模型协同设计, 而非更深入的推测.

<span id="section-4"></span>

## 4 评估

评估回答四个问题: (1) HiSparse 能否在不同上下文长度、模型、稀疏选择器和硬件平台上改善端到端服务? (2) GPU cache 能否利用足够的局部性, 将未命中次数保持在较低水平? (3) 什么因素决定未命中解析开销和 GPU-cache 大小的选择? (4) 更快的 host-device 链路和逐层预取如何与新兴硬件及模型设计趋势相互作用?

<span id="section-4-1"></span>

### 4.1 设置

**模型.** 我们评估三类稀疏注意力: DeepSeek-V4-Flash, 其混合注意力对压缩 KV 条目执行 NSA 风格的 top-$k$ 选择 (DeepSeek 称为 Compressed Sparse Attention) [Dee26, Yua25e]; 使用 DeepSeek Sparse Attention (DSA) 的 GLM-5.1-FP8 [Zen26, Dee25a, Xie26a]; Qwen3-30B-A3B-Thinking-2507 [Yan25g, Qwe25a], 并将 Quest 用作免训练稀疏选择器 [Tan24]. 逐层预取实验 ([第 4.6 节](#section-4-6)) 还使用 GLM-5.2-FP8, 该模型通过 IndexShare 在多个层组成的组内共享 DSA 索引器选择 [Glm26, Bai26]. 所有实验中, 每个查询都选择 $k=2048$ 个 token. 对 DeepSeek-V4-Flash, 选择粒度是 $4$-token 压缩 KV 条目, 因此 top-$512$ 选择覆盖 $2048$ 个 token; HiSparse 管理这些占模型 KV 主要部分的压缩 KV 条目, 其余分支状态仍驻留 GPU. 对其他模型, 选择粒度为 token, $k=2048$; Quest 不要求架构本身具有稀疏性, 它作为免训练选择器应用在每一层, 管理较小 Qwen3 模型的普通密集注意力 KV cache. 所有模型都使用 BF16 KV cache; 模型名称中的 FP8 表示权重精度.

**平台.** 端到端服务实验使用各图标明的平台: DeepSeek-V4-Flash 运行在 $2\times$B200 上, GLM-5.1-FP8 和 GLM-5.2-FP8 运行在 $8\times$H200 上, Qwen3+Quest 运行在 GH200 节点上. H200 节点的八张 GPU 搭配 $2\,\mathrm{TB}$ 主机 DRAM; 在最大工作点 ([第 4.6 节](#section-4-6), $32$K 输入 / $8$K 输出时的 $256$ 个并发请求), host KV pool 增长到约 $1\,\mathrm{TB}$ pinned host memory.

**Baseline.** 所有实验都以未修改的 SGLang v0.5.11 为 baseline, 它让完整 KV cache 驻留 HBM, 其余部署配置 (模型、并行方式、精度和调度器设置) 与 HiSparse 相同. 我们没有与其他 offloading 系统比较: 最接近的 ESS [Che25aa] 和 ECHO [Liu26] 是同期工作, 前者是通过模拟评估的原型, 后者专用于 NSA ([第 6 节](#section-6)).

**工作负载与指标.** 我们先使用 SGLang 标准 `bench_serving` 工具 [She24, Sgl26] 评估端到端服务. 对每个模型, 在固定输出长度和 closed-loop concurrency 下扫描对应图中所示的输入长度. 并发扫描 ([图 1](#figure-01)、[图 4](#figure-04) 和 [图 8](#figure-08)) 固定为 $32$K 输入 / $8$K 输出: 该长度代表长上下文服务, 也是最适合比较的区间; full-KV baseline 在 $32$K 时仍能准入多个并发水平, 因此两个系统都能形成完整曲线, 而在更长上下文下, baseline 只剩少数低并发点, 无法比较完整趋势. 因此, 最长 $200$K 的更长上下文以峰值吞吐量对比呈现 ([图 5](#figure-05)). 这些实验报告生成吞吐量 (生成 token/s)、TTFT 和 TPOT. 生成吞吐量衡量系统输出速率, TPOT 衡量首 token 之后每个输出 token 的延迟. 随后的 ablation 解释端到端收益来源: cache-policy 实验重放 LongBenchV2 [Bai25] 的稀疏选择 trace, kernel 实验报告未命中解析时间.

<span id="section-4-2"></span>

### 4.2 端到端基准

<span id="figure-04"></span>

![两张 B200 上的 DeepSeek-V4-Flash 端到端服务](./hisparse/figure-04.png)

**图 4.** $2\times$B200 上以 PD-colocated 模式运行 DeepSeek-V4-Flash (NSA) 的端到端服务, $32$K 输入 / $8$K 输出; 每步选择覆盖 $2048$ 个 token (对 $4$-token 压缩 KV 条目取 top-$512$). **(a)** 生成吞吐量与 closed-loop concurrency (实线: prefill+decode; 虚线: decode-only, 表示 PD-disaggregated decode-pool 吞吐量); **(b)** 平均 TTFT 与已达到的生成吞吐量; **(c)** 平均 TPOT 与已达到的生成吞吐量.

[图 4](#figure-04) 给出了 DeepSeek-V4-Flash 在 $32$K 输入 / $8$K 输出下的端到端基准. 低并发时, baseline 与 HiSparse 的吞吐量相近, 因为 baseline 仍能把所有活跃 KV cache 留在 HBM. 并发数上升后, baseline 饱和: 没有更多 KV 容量就无法准入额外请求, 吞吐量基本不再增长, 与 [图 1(a)](#figure-01) 中 GLM-5.1 的容量墙相同. HiSparse 降低每个请求的解码 HBM 占用并继续扩展, 在并发数 $64$ 时把生成吞吐量从 $600$ 提高到 $1257$ tokens/s ($2.1\times$), decode-only 吞吐量则从 $1511$ 提高到 $4308$ tokens/s ($2.9\times$). 收益完全来自 batch size: HiSparse 并没有加快单个解码步骤, 而是让相同 HBM 容纳更大的解码 batch. 在 $32$K 输入下, baseline 仍可容纳几十个请求, 所以这里的余量有限; 输入和输出更长时, baseline 可行的 batch 会进一步缩小, 相同作用也更明显 ([图 5](#figure-05)).

延迟图说明了这一点对 colocated 服务的意义. Baseline 中, decode 饱和会让 prefill chunk 和新请求排队, 因此即使单 token 解码延迟没有崩溃, TTFT 仍会快速增长: 平均 TTFT 从并发数 $8$ 时的 $26\,\mathrm{s}$ 增至并发数 $64$ 时的 $829\,\mathrm{s}$, 此时 HiSparse 为 $171\,\mathrm{s}$. HiSparse 留出更多 HBM 空间并更快排空 decode 工作, 在更高实际吞吐量下保持低得多的 TTFT. 随着 HiSparse 进入更高吞吐量工作点, TPOT 会上升; 但在重叠区间内仍与 baseline 相近 (并发数 $16$ 时为 $15.9$ 与 $16.0\,\mathrm{ms}$), 说明在长上下文区间, 未命中解析开销小于容量收益.

<span id="figure-05"></span>

![Quest 与 DSA 在不同输入长度下的峰值生成吞吐量](./hisparse/figure-05.png)

**图 5.** 另外两类稀疏注意力在不同输入长度下的峰值生成吞吐量, 二者都选择 $k=2048$ 个 token. *左:* GH200 上采用 Quest 的 Qwen3-30B-A3B. *右:* $8\times$H200 上采用 DSA 的 GLM-5.1-FP8.

[图 5](#figure-05) 同时沿两个方向扩展结果: 一方面扩展到其他模型、选择器和平台, 另一方面, 也是更重要的一方面, 扩展到长得多的上下文, 输入长度最高达到 $200$K (GH200 上的 Qwen3+Quest) 和 $160$K (H200 上的 GLM-5.1+DSA). 在 $4$K 时, full-KV baseline 已能在 HBM 中容纳有效 batch, HiSparse 几乎没有发挥空间: Qwen 从 $2430$ 变为 $2668$ tokens/s, GLM 则基本不变 ($2288$ 与 $2280$ tokens/s). 上下文变长后, baseline 受容量限制, HiSparse 的解码内存则与配置的 GPU-cache 大小成正比. 由此产生的收益很大, 且随上下文长度增长: Qwen 在 $32$K 和 $200$K 时分别提高 $3.6\times$ 和 $4.7\times$ ($511$ 到 $1824$、$111$ 到 $520$ tokens/s), GLM 在 $32$K 和 $160$K 时分别提高 $3.1\times$ 和 $2.9\times$ ($624$ 到 $1919$、$232$ 到 $680$ tokens/s). GLM 的 $32$K 点概括了 [图 1](#figure-01) 已展示的扫描, 后者各 panel 给出该工作点完整的并发与 TTFT 行为. 在这一区间, 将逻辑 KV 可用性与 GPU 驻留分离会转化为更高的服务吞吐量. Decode-only 曲线对 PD-disaggregated 服务给出同样结论: 排除 prefill 时间后, [图 4(a)](#figure-04) 的虚线估计了专用 decode pool 可达到的吞吐量. Full-KV decode pool 受 HBM 可准入的 batch 限制, HiSparse 提高的正是这一上限, 即前述 $2.9\times$ decode-only 收益. 受测试平台容量限制, 我们没有运行物理 disaggregated 部署, 而把 decode-only 速率作为其 proxy.

容量收益也可以换成其他资源. 不需要更多吞吐量的 operator 可以把它转化为硬件节省. 在 batch size 相同的情况下, HiSparse 会按原本提高并发数的相同倍数缩小解码 KV 预算; 对 GLM-5.1 在 $32$K 时约 ${\sim}60$ 个请求的 batch, HBM 占用从约 ${\sim}240\,\mathrm{GB}$ 降至 $B=4096$ 时的约 ${\sim}25\,\mathrm{GB}$, 而且差距随上下文变长而扩大 ($128$K 时每个请求为 $13.09$ 与 $0.4\,\mathrm{GB}$, [第 3.3 节](#section-3-3)). 在 KV 占主导的长上下文部署中, 相同工作负载因此可以放入更少的 GPU, 或更便宜、HBM 更少的部件, 代价是 [第 4.6 节](#section-4-6) 所述的单 token IO 开销.

<span id="section-4-3"></span>

### 4.3 GPU-cache 局部性与 LRU

<span id="figure-06"></span>

![七种 GPU-cache 配置的逐步 top-k 未命中率](./hisparse/figure-06.png)

**图 6.** 在七种 cache 配置下重放同一条 GLM-5.1 LongBenchV2 稀疏选择 trace ($k=2048$) 时的逐步 top-$k$ 未命中率 (跨层平均并平滑); $B$ 表示每个请求和每层的 KV-record slot 数. 仅暂存 top-$k$ 的 *Swap-vanilla* ($B=2048$) 不保留额外热条目, 平均每步有 $30\%$ 的选择未命中. 在相同 $B=4096$ 预算下, LRU (平均 $13.4\%$) 始终优于 FIFO ($17.2\%$) 和随机替换 ($16.1\%$), 趋势也接近离线 Bélády 最优策略 ($8.2\%$). 把 LRU cache 翻倍至 $B=8192$ 后, 未命中率再次减半至 $6.7\%$, 说明保留的局部性会直接减少主机内存加载.

HiSparse 有意把 GPU cache 用作热 KV cache, 而不只是临时 top-$k$ 暂存区. [图 6](#figure-06) 说明了这一选择为何重要, 以及 HiSparse 为何使用 LRU 管理 cache. Trace 来自一个 GLM-5.1 请求, 它处理 $100{,}384$-token LongBenchV2 prompt, 并在全部 $78$ 个稀疏层上解码 $1{,}799$ 步; 每种策略都重放相同的 per-layer top-$k$ 选择流 (图和下述平均值覆盖前 $1{,}000$ 个解码步骤), 因此差异只来自替换决策. 需要注意, $B$ 表示*每个请求和每层*的 KV-record slot 数 ([表 2](#table-02)), 所以 $B=4096$ 是 $k=2048$ 选择的两倍. 只保留当前 top-$k$ 条目 ($B=k$) 时, 平均每步有 $30\%$ 的选择未命中, 因为选中集合会随步骤变化. Cache 翻倍会立即减少未命中, 但替换策略决定保留多少局部性: 在 $B=4096$ 时, LRU 的平均未命中率为 $13.4\%$, 始终低于 FIFO ($17.2\%$) 和随机替换 ($16.1\%$), 且曲线形状接近离线最优的 Bélády 策略 [Bel66], 说明最近使用顺序是未来稀疏选择的良好在线代理. 因此, [第 3.4 节](#section-3-4) 的未命中解析 kernel 会保留驻留命中并原地更新 LRU, 而不是只根据当前 top-$k$ 集合重建 cache. 同一张图也显示了剩余空间: 把 LRU cache 翻倍至 $B=8192$ 后, 未命中率减半至 $6.7\%$, 而 Bélády 在 $B=4096$ 时便达到 $8.2\%$ (图中 $B=8192$ 时还会更低), 这给出了更智能的预测式替换策略在相同容量下仍可恢复的上限.

<span id="section-4-4"></span>

### 4.4 未命中解析成本与 cache 大小权衡

低未命中率只解决了一半问题: 剩余未命中的解析仍有时间成本, 而且该成本会随 cache 大小和 batch size 变化. 较大的 GPU cache 保留更多局部性, 但会延长元数据扫描, 并为每个请求消耗更多 HBM; batch size 较大时, 并发请求的未命中会争用主机内存带宽. 因此, 我们把融合 resolve kernel 的时间拆分到各阶段, 以确定实用的 cache 大小范围, 并找出未命中较少后出现的新瓶颈.

<span id="figure-07"></span>

![不同模型、cache 大小与平台上的未命中解析分解](./hisparse/figure-07.png)

**图 7.** 不同模型、GPU-cache 大小和平台上的未命中解析分解: H200 使用 PCIe Gen5 host-device 链路 (宽浅色实线, 实心 marker), GH200 使用 NVLink-C2C (深色虚线, 空心 marker). 每个 panel 标题给出模型以 KV-record 粒度表示的 top-$k$ ([第 4.1 节](#section-4-1)). *IO* 是主机内存获取阶段; *probe & scan* 包括元数据阶段, 几乎不受平台影响, GH200 虚线与 H200 区间相差不超过约 ${\sim}5\%$; 剩余阶段 (hash-table 构建和输出发布) 在所有情况下均为 $1$-$4\,\mu\mathrm{s}$, 图中省略. 上排: batch size 为 $16$ 时, 随 cache 比率 $B/k$ 变化的单次 kernel call 时间, 较大的 cache 通过减少未命中来降低 IO, 但会增加对驻留 slot 的 probe-and-scan 工作. 下排: $B=2k$ 时, 随 batch size 变化的单次 kernel call 时间, batch 较大时, PCIe 链路由 IO 主导, 更快的 NVLink-C2C 路径则压缩了该部分时间.

[图 7](#figure-07) 在三个端到端模型和 DeepSeek-V4-Pro (token-level 选择, $k=1024$) 上展示了这一调整权衡, 每个模型都使用其原生选择粒度. 较大的 GPU cache 会减少未命中数, 进而缩短 resolve kernel 的 IO 部分. 但它并非没有代价: kernel 必须 probe 和 scan 更多驻留 slot, 较大的 cache 还会为每个请求消耗更多 HBM, 而这些 HBM 原本可以容纳更多并发请求、更大的模型权重或更多驻留 MoE expert. 对图中各模型, 有效区间通常是选中集合大小的一个小倍数, 约为 $2k$-$4k$. 该范围能保留足够的热条目来利用局部性, 又不会让元数据扫描或 HBM 容量成为新瓶颈. 作为尺度参考, 在 H200 上的 GLM-5.2 decode profile 中, 稀疏注意力 kernel 本身每层约为 ${\sim}60\,\mu\mathrm{s}$ (per-GPU batch 为 $8$), 因此没有隐藏的 $100$-$200\,\mu\mathrm{s}$ resolve 会让一层注意力的关键路径延长一倍以上; 这说明既要减少未命中 ([第 4.3 节](#section-4-3)), 也要隐藏剩余部分 ([第 4.6 节](#section-4-6)).

下排在 $B=2k$ 时单独考察 batch-size 影响. Probe-and-scan 部分相对稳定, IO 部分则随 batch size 快速增长, 因为更多请求会并发发出 miss load. 这一结果说明后续 IO 侧设计的动机: 用 GPU-assisted IO 和调整后的 block size 提高每次获取的效率 ([第 3.4 节](#section-3-4), [第 3.5 节](#section-3-5)); 在平台支持时利用更快的 host-device 链路 ([第 4.5 节](#section-4-5)); 通过逐层预取把剩余延迟隐藏在计算后面 ([第 4.6 节](#section-4-6)).

<span id="section-4-5"></span>

### 4.5 带宽敏感性

上一项实验表明, batch size 较大时, 主机内存 IO 可以主导未命中解析成本. 这引出一个面向硬件的问题: 如果 CPU-GPU 路径快得多, HiSparse 是否仍应使用较大的 GPU cache 来避免未命中, 还是应接受更多未命中, 把内存留给并发请求? 我们在 GH200 级高带宽 host-device 路径上研究这一问题 [Nvi26a].

[图 7](#figure-07) 的 GH200 虚线只改变平台并重复 microbenchmark, 结果显示 IO 部分明显缩短; 对 GLM-5.1 的 $B=2k$、batch size $16$, 单次 call 从 $112$ 降至 $29\,\mu\mathrm{s}$. 因此, 超大 GPU cache 的收益减小: 获取一次未命中更便宜, 但扫描更大的 cache 仍需时间并消耗 HBM. 实际工作点会转向 $B=2k$ 等较小 cache, 为更大的解码 batch 保留更多设备内存. 这一结果再次说明系统层面的结论: HiSparse 把容量瓶颈转化为可调的延迟/带宽问题, host-device 链路更快的平台会让这项权衡更有利.

HiSparse 中的调整有意保持简单且静态: $B$ 是部署时固定的服务配置参数, 通过 [图 7](#figure-07) 一类面向目标平台的 profiling sweep 选择, $B=2k$ 是稳健默认值. 在实验中, 首选设置主要取决于平台的 host-link 带宽 (对比 [图 7](#figure-07) 中的实线与虚线 IO 曲线), 而不是工作负载. Cache 按请求分配, 因此在准入时或动态调整大小并不复杂, 但动态策略留待未来研究. 下一项实验考察互补的模型侧机会: 利用跨层选择局部性隐藏剩余未命中延迟.

<span id="section-4-6"></span>

### 4.6 逐层预取

即使 host-device 链路更快, 只要 KV 传输不能与有效计算重叠, 未命中仍会位于关键路径. 我们在 GLM-5.2-FP8 上评估 HiSparse 的精确预取路径 ([第 3.5 节](#section-3-5)); 该模型通过 IndexShare 跨层共享 DSA 索引器选择 [Glm26, Bai26]: $78$ 层中, $21$ 个 anchor 层运行索引器, 其余 $57$ 层复用前一个 anchor 的选择结果. Anchor 层输出选中集合后, HiSparse 会为后续 shared 层的 KV 记录发出 host-to-device load, 并让它们与中间层的计算重叠. 我们在 $8\times$H200、PD-colocated 模式、$32$K 输入 / $8$K 输出下比较四种设置, closed-loop concurrency 从 $8$ 扫描至 $256$: full-KV baseline; 使用同步未命中解析的 HiSparse (禁用预取); 使用精确预取的 HiSparse; *no-IO oracle*, 即跳过 resolve kernel 主机内存 IO 的 HiSparse. Oracle 在未命中时使用陈旧 KV 记录, 因此输出无效; 由于基准固定输出长度, 其时序仍可作为任何 IO 隐藏方案的有效上界, 比完全重叠的预取还强, 因为后者仍会把相同流量放到主机链路. 所有 HiSparse 配置都使用 $k=2048$ 和 $B=4096$ ($B=2k$), 位于 [第 4.4 节](#section-4-4) 确定的 $2k$-$4k$ 范围内.

<span id="figure-08"></span>

![使用 IndexCache 共享选择的精确逐层预取](./hisparse/figure-08.png)

**图 8.** 使用 IndexCache 共享选择的逐层精确预取: $8\times$H200 上以 PD-colocated 模式运行 GLM-5.2-FP8 (DSA), $32$K 输入, $8$K 输出, $k=2048$, $B=4096$. 灰色点线是 *no-IO oracle*, 它完全跳过主机内存 IO (输出无效; 作为所有 IO 隐藏方案的性能上界). **(a)** 生成吞吐量与 closed-loop concurrency (深色: prefill+decode; 浅色: decode-only, 表示 PD-disaggregated decode-pool 吞吐量). Full-KV baseline 在 KV cache 填满 HBM 后饱和, 所有 HiSparse 变体都继续扩展到 $256$ 个请求. **(b)** 平均 TTFT 与已达到的生成吞吐量, 点旁标签给出并发数. **(c)** 平均 TPOT 与已达到的生成吞吐量: 低并发时 oracle 与 baseline 的 TPOT 相同, 说明 HiSparse 的全部开销都是 IO; 在相同并发数下, 精确预取把 TPOT 降低 $13$-$15\%$, 消除了同步解析与 oracle 差距的约一半.

[图 8](#figure-08) 显示了与 [第 4.2 节](#section-4-2) 相同的容量趋势: full-context KV cache 填满 HBM 后, baseline 饱和, 平均 TTFT 从并发数 $16$ 时的 $16\,\mathrm{s}$ 跳至并发数 $32$ 时的 $91\,\mathrm{s}$ 和并发数 $64$ 时的 $275\,\mathrm{s}$, HiSparse 变体则扩展到 $256$ 个并发请求. Panel (b) 还表明 prefill-side staging 几乎没有成本: 低并发时 TTFT 不含排队, 此时把 prefill KV 以 write-through 方式写入 host pool 不会改变平均 TTFT (并发数 $8$ 时, baseline 为 $10.7\,\mathrm{s}$, 所有 HiSparse 变体为 $10.7$-$10.8\,\mathrm{s}$). 随后, 预取会恢复大部分剩余未命中解析成本. 在相同并发数下, 精确预取在整个 sweep 中把平均 TPOT 降低 $13$-$15\%$, 把生成吞吐量提高 $14$-$17\%$; 峰值生成吞吐量从 baseline 的 $618$ 和无预取时的 $1515$ 提高到 $1727$ tokens/s, 相比 full-KV baseline 端到端提高 $2.8\times$. No-IO oracle 给出的上界为 $2034$ tokens/s: 精确预取达到该上限的 $85\%$, 无预取时为 $74\%$. Decode-only 曲线为 PD-disaggregated 服务给出相同上界: oracle 的 decode 速率达到 $4671$ tokens/s, 精确预取为 $3410$ ($73\%$); 这一差距大于 colocated 模式, 因为后者的 prefill 时间稀释了解码侧 IO 的相对成本.

Panel (c) 针对这一明确上界量化开销. 低并发时, oracle 的 TPOT 与 full-KV baseline 相同 (并发数 $8$ 时为 $24.1$ 与 $24.8\,\mathrm{ms}$), 说明解析机制本身, 包括 hash probe、victim scan、LRU update 和 KV gather, 不增加可测量的单 token 成本: HiSparse 的全部 TPOT 开销来自主机内存 IO. 相比这一上界, 同步解析在并发数 $8$ 时暴露每 token $7.7\,\mathrm{ms}$ IO, 在并发数 $256$ 时暴露 $22.0\,\mathrm{ms}$; 精确预取分别把暴露时间降至 $3.0\,\mathrm{ms}$ 和 $11.2\,\mathrm{ms}$, 在整个 sweep 的相同并发数下隐藏约一半 IO 开销. 很多剩余开销来自结构本身: anchor 层自己的选择无法提前知道, 因此它们的未命中, 即 $78$ 层中的 $21$ 层, 按各层 IO 均匀计算约占 $27\%$, 必然保持同步. 扣除该下限后, 预取隐藏了其可处理 IO 的三分之二到五分之四; 剩余部分是重叠不足, 因为预取只是重新安排传输时间, 没有消除传输, 高 batch 下的流量仍会在同一条主机链路上与 demand miss 竞争.

我们还评估了 [第 3.5 节](#section-3-5) 的推测式变体, 它在不共享索引的模型中用第 $\ell$ 层的选择提示第 $\ell+1$ 层. 该方案只略微提高 GPU-cache 命中率, 没有产生可测量的端到端收益: 提示位置通常已驻留, 剩余未命中则恰好是新进入 top-$k$、提示无法预测的位置, 因此推测增加了 host-link 流量, 却没有移除关键路径加载. 结论取决于依赖关系: 要把未命中 IO 与计算重叠, 必须在某层运行前很早就知道其选择, 推测无法可靠提供这些信息. 模型协同设计可以做到这一点, 跨层共享选择会直接消除依赖并使预取保持精确, 这正是无可测量收益与上述 $13$-$15\%$ TPOT 降幅之间的区别.

<span id="section-5"></span>

## 5 讨论与限制

**暴露 IO 造成的 TPOT 开销.** HiSparse 用 HBM 容量换取主机内存流量, 这会表现为 TPOT 开销: 同样的低并发下, 同步解析未命中时每个 token 增加 $7$-$8\,\mathrm{ms}$ ([第 4.6 节](#section-4-6)). 设计通过多层措施降低该开销, 包括 LRU cache ([第 4.3 节](#section-4-3))、GPU-assisted IO ([第 3.4 节](#section-3-4)) 和更快的链路 ([第 4.5 节](#section-4-5)); 模型协同设计还能隐藏剩余开销的约一半: 使用共享选择时, 精确预取把低并发下暴露的 IO 降至每 token 约 $3\,\mathrm{ms}$, no-IO oracle 则表明解析机制本身没有成本 ([第 4.6 节](#section-4-6)). 当服务不受容量限制时, 例如短上下文或低并发, HiSparse 没有收益来抵消这项开销, 可以直接禁用.

**吞吐量以外: HBM 无法容纳的上下文.** 容量收益不只是 batch-size multiplier. 在 full-KV 服务中, KV cache 超过可用 HBM 的请求在*任何*并发数下都无法准入, 可服务上下文受设备内存限制, 而不是模型的上下文窗口限制 (一个 $1$M-token GLM-5.1 请求需要 $100\,\mathrm{GB}$ 以上 KV, [第 1 节](#section-1)). HiSparse 的解码侧 HBM 由 GPU-cache 大小限制, 与上下文长度无关; 因此, 在中等上下文上把容量转化为 batch size 的同一机制, 也会在极长上下文上把容量转化为可行性, 最大可服务上下文改由 host tier 容量决定. 这一点直接来自占用上界 ([第 3.3 节](#section-3-3)), 因此我们没有单独评估.

**主机内存容量.** 当前更根本的限制是第二层容量, 而非开销. HiSparse 假设 host DRAM 远大于 HBM, 这对 TB 级 PCIe H200 server 成立, 对基于 Grace 的 GB200/GB300 系统则不成立; 每个 Grace CPU 约有 ${\sim}480\,\mathrm{GB}$ LPDDR, 与配对 GPU 的 HBM 总量相近, 对 GB300 而言甚至更小. 第二层不比第一层大时, HiSparse 转化为 batch size 的容量倍数会缩小. NVMe 或 network-attached tier 可以恢复容量, 但延迟更高, 因而更依赖局部性和重叠.

**协同设计启示.** 我们希望这项探索能为未来协同设计提供参考. 为重叠预留空间的模型, 例如跨层共享选择或提前输出选择 [Bai26, Glm26], 会把 KV 放置从延迟问题转化为调度问题; 主机内存更大、CPU-GPU 互连更快的平台则能直接使用更小的 GPU cache 和更大的 batch ([第 4.5 节](#section-4-5)). 稀疏注意力让单步 KV 需求变小且可预测, 模型和硬件只需作适度调整, 就能叠加 HiSparse 当前已经利用的收益.

<span id="section-6"></span>

## 6 相关工作

**LLM 服务与 KV 内存管理.** PagedAttention 和 vLLM 以定长 page 管理 KV 内存, 减少碎片并提高服务吞吐量 [Kwo23]. SGLang 为结构化 LLM 程序提供高吞吐服务 runtime, 并优化 KV 复用 [She24]. 服务系统还会拆分或协调 prefill 与 decode, 因为两个阶段对资源的压力不同 [Pat23, Zho24]; IO-aware 注意力 kernel 可以降低计算侧内存流量, 但不会缩小驻留 KV 占用 [Dao22, Dao24a]. HiSparse 与这些工作互补: 它降低每个长上下文稀疏注意力请求的解码侧 HBM 占用, 对 disaggregated decode pool 和 colocated prefill/decode 部署都有帮助.

**KV 压缩与淘汰.** 大量工作通过丢弃或近似条目来缩小 KV cache: H2O、StreamingLLM 和 SnapKV 等淘汰策略会永久丢弃被判断为不重要的 token [Zha23g, Xia24a, Li24c], 量化则减少每个条目的字节数 [Liu24c]. 这些方法以输出保真度换取容量. HiSparse 与它们正交: 它移动 KV 条目, 而不是丢弃或近似条目, 稀疏注意力输出保持不变.

**分层与 offloaded KV cache.** FlexGen、HiCache、CachedAttention 和 Mooncake 等系统把模型或 KV 状态放在 GPU、主机和存储层之间, 面向吞吐量或跨轮服务 [Sto24, Xie25d, Gao24a, Qin25c]; HiSparse 的主机内存获取路径借用了 Strata 的 GPU-assisted IO 技术 [Xie25a]. 与 HiSparse 更接近的若干系统会把 KV 状态 offload 到主机内存, 并在 decode 期间有选择地获取或使用: InfiniGen 推测性地预取被预测为重要的条目 [Lee24c]; ShadowKV 通过保留在 GPU 上的 low-rank key 进行选择, 并 offload value [Sun25f]; MagicPIG 使用 CPU-resident LSH 对 KV 条目采样 [Che25ab]; ArkVale 淘汰冷 page, 但可通过 page 摘要将其召回 [Che24t]; PQCache 使用 product quantization 检索条目 [Zha25i]; NEO 和 FastDecode 则把注意力计算本身 offload 到 CPU [Jia25e, He24g]. 基于检索的系统会把自己的近似选择器接到密集注意力模型上, 因而输出偏离底层模型; CPU 计算系统保持精确, 但把注意力放到 CPU. HiSparse 改为服务模型*自己的* top-$k$ 选择, 从构造上保持精确, 所有计算仍在 GPU 上. 与 HiSparse 最接近的是两项同期工作. ESS 原型把 DeepSeek-V3.2 的 latent cache offload 到 GPU hot cache 后的主机内存, 其架构专用于该模型, 并通过模拟评估 [Che25aa]. ECHO 为 NSA 模型 offload KV 状态, 以预取为中心, 收益取决于能否准确预测后续选择 [Liu26]. HiSparse 通过与索引器无关的接口同时支持训练式和免训练选择器 (DSA、NSA、Quest), 优先利用局部性, LRU 管理的 cache 使大多数选择驻留, 直接减少 IO load, 而非只做重叠; 它用捕获在 decode CUDA graph 中的单个融合 kernel 解析未命中, 并只在模型共享选择结果、能够精确而非预测时使用预取.

**稀疏注意力.** 早期架构通过固定模式、content-based bucketing 或 low-rank projection 限制注意力 [Bel20, Kit20, Wan20a]. DSA/NSA 等训练式 query-dependent 稀疏注意力架构和混合注意力模型会降低注意力计算与活跃 KV 读取 [Dee25a, Zen26, Yua25e, Dee26], Quest 等免训练方法则在推理时选择随查询变化的 KV page [Tan24]. HiSparse 不是新的稀疏注意力算法, 而是服务系统层, 用来降低这些算法非活跃状态占用 HBM 的成本.

<span id="section-7"></span>

## 7 结论

Top-$k$ 稀疏注意力让服务系统继续为每个解码步骤几乎不读取的 KV 条目支付完整上下文的 HBM 成本. HiSparse 通过解耦逻辑可用性与物理驻留消除这种失配: 每个请求的完整 KV 历史保存在主机内存, 解码占用由小型 GPU cache 限制, 其 LRU 管理把选择局部性转化为命中; 一个融合 kernel 在 decode CUDA graph 内解析每层选择; 跨层共享选择的模型则让预取保持精确, 把大部分剩余 IO 与计算重叠. 由于改变的只有 KV 放置, 模型输出保持不变. 在 H200、B200 和 GH200 平台上的 DSA、NSA 和 Quest 实验中, HiSparse 在单 token 延迟相近的情况下, 将长上下文峰值生成吞吐量最高提高 $4.7\times$; no-IO oracle 表明解析机制本身没有可测量成本, 精确预取可达到所有 IO 隐藏方案吞吐量上限的 $85\%$.

除吞吐量外, 限制解码侧 HBM 还能让可服务上下文超过 HBM 单独可容纳的范围, 上限改由 host tier 容量决定. HiSparse 已随上游 SGLang 发布, 其更广泛的结论与协同设计有关: 稀疏注意力让单步 KV 需求变小且可预测, 模型与硬件只需适度配合, 例如共享或提前输出选择、提供更快的 host-device 链路, 就能叠加分层 KV 放置目前已经带来的收益.

## 致谢

我们感谢 Alibaba Cloud TairKVCache 团队、Ant Group SCT Inference 团队、Baidu Baige AI 团队和 Zhipu AI 对 HiSparse 的开源贡献. 我们感谢 Alibaba Cloud 的 Shangming Cai、Teng Ma 和 Xingyu Ling 提供建设性反馈. 本研究部分由 Stanford Platform Lab 及其附属机构支持. Zhiqiang Xie 获得 NVIDIA Graduate Fellowship 支持. 我们感谢 RadixArk 提供计算资源.

<span id="section-8"></span>

## 8 在 SGLang 中的实现

本附录概述 HiSparse 为 SGLang 添加的内容, 以及它与现有 engine 的连接位置. 该功能只需一个 flag 即可启用, 服务 stack 的其余部分保持不变; 它包括分布在六个 module 中约 $2{,}200$ 行新 Python 代码和一个 CUDA kernel header, 以及对 scheduler、model runner、attention backend 和 disaggregation path 的集成修改.

**新组件.** *Coordinator* (`managers/hisparse_coordinator.py`, 约 ${\sim}1{,}000$ 行) 负责 [第 3.3 节](#section-3-3) 的请求生命周期: 把 prefill KV 暂存到 host pool, 分配并扩展 per-request-per-layer GPU cache, 对新生成 KV 执行 write-through, 并协调 swap-in, 包括 [第 3.5 节](#section-3-5) 的 plan-then-IO prefetch group. *融合 kernel* (`jit_kernel/hisparse.py` 与 `csrc/hisparse.cuh`) 为 token-level 和 compressed KV layout 实现 Resolve, 可选择记录 miss plan, 还包括为 shared layer 重放已记录计划的 copy-only kernel. *内存层* (`mem_cache/allocator/hisparse.py`、`hisparse_memory_pool.py`、`pool_host/hisparse.py`) 提供 paged host-pool allocation、device-side cache pool, 以及基于 SGLang 现有 host KV cache class 的 mixin (见下文). 一个小型*配置 module* (`arg_groups/hisparse_hook.py`) 应用 backend 默认值, 并验证 attention-backend 与 KV-dtype 兼容性.

**Engine 集成.** 除 [第 3 节](#section-3) 的机制外, 工程工作集中在三个位置. Concurrency: staging、write-through 和 prefetch 分别运行在自己的 CUDA stream 上, 通过 event 与 compute stream 排序; scheduler 只在 staging acknowledgment 到达后准入请求, retract 和 pause 路径会与普通 KV 状态一起释放 HiSparse 状态. Graph capture: Resolve 和 prefetch fork 被捕获在 SGLang steady-state decode CUDA graph 内, 因而所有元数据更新和 IO 发出都必须可重放, 不能依赖 host-side branch. Compatibility: model runner 会为声明 selection-sharing group 的模型自动启用精确预取, 在 pipeline parallelism 或 speculative decoding 下回退到同步 swap-in, HIP kernel 变体则支持 AMD GPU; 在 disaggregated 模式下, prefill instance 通过现有 transfer backend 把 KV 直接写入 decode host 的 DRAM pool.

**与 HiCache 的关系.** SGLang 的 HiCache [Xie25d] 是分层*前缀* cache: host 和 storage tier 让已结束请求的 KV 可以被之后共享前缀的请求复用. HiSparse 通过 mixin 复用 HiCache 的 host-tier 基础设施, 即 pinned host KV pool 和 IO backend, 但管理的对象不同: 它管理由模型自身稀疏选择控制的 per-request decode working set. 在部署中, 两者组成 prefill-decode 双系统: prefill node 上的 HiCache 通过复用前缀减少 prefill 工作, decode node 上的 HiSparse 则限制每个请求的驻留量, 从而扩展 decode batch. GPU-assisted IO 路径沿用 Strata [Xie25a].

**配置.** 使用 `--enable-hisparse` 启用 HiSparse; JSON `--hisparse-config` 设置 `top_k`、`device_buffer_size` ($B$)、`host_to_device_ratio` (host-pool 容量与 device KV 预算之比), 以及 [第 3.4 节](#section-3-4) 的 swap-in transfer block size. 对 shared-index 模型, 精确预取会自动启用, ablation 可通过环境变量禁用; [第 4 节](#section-4) 的所有实验都使用这些 switch, 无需修改代码.
