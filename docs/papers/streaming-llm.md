---
title: 'StreamingLLM'
createTime: 2026/09/06 14:00:00
permalink: /papers/streaming-llm/
---

> [Guangxuan Xiao](https://guangxuanx.com/) [+1], [Yuandong Tian](https://yuandong-tian.com/), [Beidi Chen](https://www.andrew.cmu.edu/user/beidic/), [Song Han](https://www.rle.mit.edu/people/song-han/), [Mike Lewis](https://ai.meta.com/people/209431298931133/mike-lewis/). 2023 年 9 月 29 日首次提交至 arXiv; 发表于 ICLR 2024; 当前版本为 v4. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453v4). <a href="/paper/streaming-llm.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2309.17453). [TeX 源文件](https://export.arxiv.org/e-print/2309.17453v4). 准确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

在多轮对话等预期会有长期交互的流式应用中部署大语言模型 (LLM) 已是迫切需求, 但面临两项主要挑战. 第一, 在解码阶段缓存先前 token 的键和值状态 (KV) 会占用大量内存. 第二, 常用 LLM 无法泛化到比训练序列更长的文本. 窗口注意力只缓存最近的 KV, 是一种自然的方法, 但我们表明, 文本长度一旦超过缓存大小, 该方法就会失效. 我们观察到一个有趣的现象, 即"注意力汇聚": 保留初始 token 的 KV, 可以在很大程度上恢复窗口注意力的性能. 本文首先说明, "注意力汇聚"的出现源于模型把较强的注意力分数投向初始 token, 以其作为"汇聚点", 即使这些 token 在语义上并不重要. 基于上述分析, 我们提出 StreamingLLM, 一个高效框架, 使以*有限长度*注意力窗口训练的 LLM 无需任何微调即可泛化到*无限序列长度*. 我们表明, StreamingLLM 能让 Llama-2, MPT, Falcon 和 Pythia 在长达 400 万个乃至更多 token 上进行稳定, 高效的语言建模. 我们还发现, 在预训练时加入一个占位 token 作为专用注意力汇聚 token, 可以进一步改善流式部署. 在流式场景中, StreamingLLM 相比滑动窗口重计算基线最多可加速 22.2$\times$. 代码与数据集见此[链接](https://github.com/mit-han-lab/streaming-llm).

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) [Rad18, Bro20, Zha22, Ope23, Tou23, Tou23a] 日益普及, 为对话系统 [Sch22, Tao23, Chi23], 文档摘要 [Goy20, Zha23r], 代码补全 [Che21, Roz23] 和问答 [Kam23a] 等许多自然语言处理应用提供支持. 为充分发挥预训练 LLM 的潜力, 它们应当能够高效, 准确地生成长序列. 例如, 理想的 ChatBot 助手可以依据最近一整天的对话内容稳定工作. 然而, LLM 很难泛化到比预训练序列更长的序列, 例如 Llama-2 的预训练长度为 4K [Tou23a].

原因在于, LLM 受到预训练期间注意力窗口的限制. 尽管已有大量工作致力于扩大窗口大小 [Che23x, Kai23, Pen23], 提高长输入的训练 [Dao22, Dao23a] 和推理 [Pop22, Xia23, Ana23, Wan21f, Zha23g] 效率, 可接受的序列长度在本质上仍然是*有限*的, 因而无法持续部署.

本文首先引入 LLM 流式应用这一概念, 并提出问题:

> *能否在不牺牲效率和性能的情况下, 为无限长输入部署 LLM?*

把 LLM 用于无限输入流时, 会遇到两项主要挑战:

1. 在解码阶段, 基于 Transformer 的 LLM 会缓存所有先前 token 的键和值状态 (KV), 如[图 1(a)](#figure-01) 所示, 这可能导致内存用量过高, 解码延迟不断增加 [Pop22].
2. 现有模型的长度外推能力有限, 即序列长度一旦超出预训练时设定的注意力窗口大小, 性能就会下降 [Pre21, Che23x].

一种直观的方法是窗口注意力 [Bel20] ([图 1(b)](#figure-01)), 它只为最近 token 的 KV 状态维持一个固定大小的滑动窗口. 缓存首次填满后, 这种方法能保证内存用量和解码速度不变, 但当序列长度超过缓存大小时, 模型就会崩溃, 也就是说, *即使只逐出第一个 token 的 KV* 也会如此, 如[图 3](#figure-03) 所示. 另一种策略是滑动窗口重计算 ([图 1(c)](#figure-01)), 它会为每个生成的 token 重建最近 token 的 KV 状态. 这种方法性能很强, 但由于需要在窗口内计算二次复杂度的注意力, 速度明显更慢, 因而不适用于现实中的流式应用.

<span id="figure-01"></span>

![StreamingLLM 与现有方法的比较示意图](./streaming-llm/figure-01.png)

**图 1. StreamingLLM 与现有方法的比较示意图.** 语言模型在长度为 $L$ 的文本上完成预训练, 现要预测第 $T$ 个 token ($T\gg L$). (a) 稠密注意力的时间复杂度为 $O(T^2)$, 缓存大小持续增长. 当文本长度超过预训练文本长度时, 其性能下降. (b) 窗口注意力缓存最近 $L$ 个 token 的 KV. 它的推理效率较高, 但一旦初始 token 的键和值被逐出, 性能就会急剧下降. (c) 滑动窗口重计算会为每个新 token 从最近 $L$ 个 token 重建 KV 状态. 它在长文本上性能良好, 但由于上下文重计算中的注意力具有二次复杂度, 总复杂度为 $O(T L^2)$, 因此速度相当慢. (d) StreamingLLM 保留*注意力汇聚 token* (若干初始 token) 以稳定注意力计算, 并与最近 token 结合使用. 它效率较高, 在长文本上也能保持稳定性能. 困惑度使用 Llama-2-13B 在 PG-19 测试集第一本书 (65K token) 上测得.

为理解窗口注意力失效的原因, 我们发现了自回归 LLM 的一个有趣现象: 无论初始 token 是否与语言建模任务相关, 都会有大得惊人的注意力分数分配给它们, 如[图 2](#figure-02) 所示. 我们把这些 token 称为"**注意力汇聚 token**". 它们虽然缺少语义意义, 却会收集大量注意力分数. 我们认为原因在于 SoftMax 操作要求所有上下文 token 的注意力分数之和为 1. 因此, 即使当前查询在许多先前 token 中都没有很强的匹配项, 模型仍需把这些不必要的注意力值分配到某处, 使总和等于 1. *初始* token 会成为汇聚 token 的原因很直观: 自回归语言建模的性质使初始 token 对几乎所有后续 token 可见, 因此更容易被训练成注意力汇聚 token.

基于上述认识, 我们提出 StreamingLLM, 一个简单而高效的框架, 使以有限注意力窗口训练的 LLM 无需微调即可处理无限长文本. StreamingLLM 利用了注意力汇聚 token 具有较高注意力值这一事实; 保留这些 token 可以让注意力分数分布接近正常状态. 因此, StreamingLLM 只需把注意力汇聚 token 的 KV (仅 4 个初始 token 就已足够) 与滑动窗口的 KV 一同保留, 便可锚定注意力计算并稳定模型性能. 借助 StreamingLLM, Llama-2-[7, 13, 70]B, MPT-[7, 30]B, Falcon-[7, 40]B 和 Pythia-[2.9, 6.9, 12]B 等模型可以可靠地建模 400 万个 token, 而且可能处理更多. 与唯一可行的基线滑动窗口重计算相比, StreamingLLM 最多可加速 22.2$\times$, 实现 LLM 的流式使用.

我们进一步验证了注意力汇聚假说, 并说明语言模型经过预训练后, 在流式部署中可以只需要一个注意力汇聚 token. 具体而言, 我们建议在所有训练样本开头加入一个额外的可学习 token, 作为指定的注意力汇聚 token. 我们从头预训练了参数量为 1.6 亿的语言模型, 结果表明, 加入这一个汇聚 token 即可在流式场景中保持模型性能. 这与普通模型形成对照: 后者必须重新引入多个初始 token 作为注意力汇聚 token, 才能达到相同性能水平.

<span id="figure-02"></span>

![Llama-2-7B 各层与注意力头的平均注意力 logit](./streaming-llm/figure-02.png)

**图 2.** Llama-2-7B 在 256 个长度均为 16 的句子上的*平均*注意力 logit 可视化. 观察结果包括: (1) 前两层 (第 0 层和第 1 层) 的注意力图呈现"局部"模式, 最近的 token 获得更多注意力. (2) 超过最底部两层后, 模型的所有层和注意力头都会高度关注初始 token.

最后, 我们强调, StreamingLLM 并不扩展 LLM 的上下文长度, 而是从 KV 缓存中的 token 高效生成连贯文本. 它适用于要求持续运行, 内存用量少且对历史数据依赖较低的场景. StreamingLLM 还可以与上下文扩展方法结合, 增加可关注的近期上下文.

<span id="section-2"></span>

## 2 相关工作

围绕把 LLM 应用于长文本, 已有大量研究, 主要关注三个方向: **长度外推**, **上下文窗口扩展**和**改善 LLM 对长文本的利用**. 这些方向看似相关, 但一个方向上的进展未必会推动另一个方向. 例如, 扩大 LLM 的上下文大小并不会改善模型在该大小以外的性能, 而且两种方法都不能保证模型有效利用长上下文. 我们的 StreamingLLM 框架主要属于第一类, 即把 LLM 应用于显著超过预训练窗口大小, 甚至可能无限长的文本. 我们既不扩大 LLM 的注意力窗口, 也不增强模型对长文本的记忆与利用能力. 后两个方向与本文关注的问题正交, 可以与我们的技术结合.

**长度外推**旨在让以较短文本训练的语言模型能在测试时处理更长文本. 一条主要研究路线是为 Transformer 模型开发相对位置编码方法, 使其能够在训练窗口以外工作. 旋转位置嵌入 (RoPE) [Su21] 就是其中一种方法, 它变换每个注意力层中的查询和键, 以整合相对位置信息. 尽管这种方法很有潜力, 后续研究 [Pre21, Che23x] 表明, 它在超过训练窗口的文本上表现不佳. 另一种方法 ALiBi [Pre21] 根据查询与键的距离为注意力分数加入偏置, 从而引入相对位置信息. 这种方法的外推能力有所改善, 但我们对 MPT 模型的测试表明, 当文本长度远大于训练长度时, 模型仍会失效. 现有方法尚未实现无限长度外推, 因此没有任何现有 LLM 适合流式应用.

**上下文窗口扩展**旨在扩大 LLM 的上下文窗口, 使其能在一次前向传播中处理更多 token. 一条主要研究路线处理训练效率问题. 训练时的注意力计算具有二次复杂度, 因此开发长上下文 LLM 在计算和内存上都很困难. 解决方案包括 FlashAttention [Dao22, Dao23a] 等面向系统的优化, 它能加速注意力计算并减少内存占用; 也包括近似注意力方法 [Zah20, Bel20, Wan20a, Kit20], 它们以模型质量换取效率. 最近, 使用 RoPE 扩展预训练 LLM 的工作迅速增多 [Che23x, Kai23, Blo23a, Pen23], 涉及位置插值和微调. 不过, 上述技术都只能有限地扩大 LLM 的上下文窗口, 无法解决本文主要关注的无限输入处理问题.

**改善 LLM 对长文本的利用**旨在优化 LLM, 使其更好地捕捉和使用上下文中的内容, 而不只是把这些内容作为输入. Liu 等人 [Liu23z] 和 LongChat 团队 [Li23h] 指出, 前两个方向上的成功未必能转化为有效利用长上下文的能力. 如何让 LLM 有效使用长上下文仍是一个难题. 本文侧重稳定利用最近的 token, 使 LLM 能顺畅地用于流式应用.

<span id="section-3"></span>

## 3 StreamingLLM

<span id="figure-03"></span>

![不同 LLM 在 20K token 文本上的语言建模困惑度](./streaming-llm/figure-03.png)

**图 3.** 多种 LLM 在 20K token 文本上的语言建模困惑度. 观察结果呈现一致趋势: (1) 输入长度一旦超过预训练注意力窗口大小, 稠密注意力就会失效. (2) 输入长度一旦超过缓存大小, 也就是初始 token 被逐出时, 窗口注意力就会崩溃. (3) StreamingLLM 的性能保持稳定, 困惑度几乎与滑动窗口重计算基线相同.

<span id="section-3-1"></span>

### 3.1 窗口注意力的失效与注意力汇聚

窗口注意力技术虽然能提高推理效率, 却会产生极高的语言建模困惑度. 因而模型性能不适合部署到流式应用中. 本节用*注意力汇聚*概念解释窗口注意力为何失效, 这也是 StreamingLLM 的灵感来源.

**定位困惑度激增点.** [图 3](#figure-03) 给出了 20K token 文本上的语言建模困惑度. 可以清楚看到, 当文本长度超过缓存大小时, 排除初始 token 会导致困惑度激增. 这说明, 无论初始 token 与待预测 token 相距多远, 它们对维持 LLM 的稳定性都不可或缺.

**为什么移除*初始* token 的 KV 会使 LLM 失效?** 我们在[图 2](#figure-02) 中可视化了 Llama-2-7B 模型所有层与注意力头的注意力图. 我们发现, 除最底部两层外, 模型在所有层与注意力头中都会持续关注初始 token. 这意味着, 移除这些初始 token 的 KV, 会从注意力计算中 SoftMax 函数 ([公式 1](#equation-01)) 的分母中去掉很大一部分. 这一变化使注意力分数分布明显偏离正常推理时的预期分布.

<span id="equation-01"></span>

$$
\mathrm{SoftMax}(x)_i=\frac{e^{x_i}}{e^{x_1}+\sum_{j=2}^{N}e^{x_j}},\quad x_1\gg x_j,\ j\in 2,\dots,N
$$

初始 token 对语言建模的重要性有两种可能解释: (1) 它们的语义不可或缺; 或者 (2) 模型学习了对其绝对位置的偏置. 为区分这两种可能, 我们进行了实验 ([表 1](#table-01)), 把前 4 个 token 替换为换行 token `\n`. 观察结果表明, 模型仍然会高度关注这些位于开头的换行 token. 而且, 重新引入这些 token 后, 语言建模困惑度恢复到与保留原始初始 token 相近的水平. 这说明, 起始 token 的绝对位置比其语义价值更重要.

**LLM 把初始 token 视为注意力汇聚 token.** 为解释模型为何不顾语义上是否与语言建模相关, 仍不成比例地关注初始 token, 我们引入"*注意力汇聚*"这一概念. SoftMax 函数 ([公式 1](#equation-01)) 的性质决定了所有被关注 token 的值不能全为零. 即使当前嵌入自身包含的信息已经足以完成预测, 所有层的所有注意力头仍必须从其他 token 聚合一些信息. 因而, 模型往往把不必要的注意力值倾倒给特定 token. 量化离群值领域也观察到类似现象 [Xia23, Bon23], 并据此提出 SoftMax-Off-by-One [Mil23] 作为一种可能的补救方法.

<span id="table-01"></span>

![窗口注意力, 初始 token 与换行 token 的困惑度比较](./streaming-llm/table-01.png)

**表 1.** 窗口注意力在长文本上表现不佳. 把开头 4 个 token 与最近 1020 个 token 一同重新引入 (4+1020) 后, 困惑度得到恢复. 用换行 token `\n` 替换原始的 4 个初始 token (`4"\n"+1020`), 可实现相近的困惑度恢复效果. 缓存配置 $x$+$y$ 表示加入 $x$ 个初始 token 和 $y$ 个最近 token. 困惑度在 PG19 测试集第一本书 (65K token) 上测得.

<span id="table-02"></span>

![重新引入不同数量的初始 token 对 StreamingLLM 的影响](./streaming-llm/table-02.png)

**表 2.** 重新引入不同数量的初始 token 对 StreamingLLM 的影响. (1) 窗口注意力 ($0+y$) 的困惑度急剧增加. (2) 引入一个或两个初始 token 无法完全恢复模型困惑度, 说明模型并非只把第一个 token 用作注意力汇聚 token. (3) 引入 4 个初始 token 通常已经足够; 继续增加所带来的收益逐渐减小. 缓存配置 $x$+$y$ 表示把 $x$ 个初始 token 加入 $y$ 个最近 token. 困惑度在拼接后的 PG19 测试集的 400K token 上评估.

为什么 Llama-2, MPT, Falcon 和 Pythia 等各种自回归 LLM 始终关注*初始 token* 作为注意力汇聚 token, 而不是其他 token? 我们的解释很直接: 由于自回归语言建模具有顺序性, 初始 token 对所有后续 token 都可见, 而较晚的 token 只对有限数量的后续 token 可见. 因此, 初始 token 更容易被训练成注意力汇聚 token, 接收不必要的注意力.

我们注意到, LLM 通常经过训练后会使用多个初始 token, 而不只使用一个 token 作为注意力汇聚 token. 如[表 2](#table-02) 所示, 引入 4 个初始 token 作为注意力汇聚 token, 足以恢复 LLM 的性能. 相比之下, 只加入一个或两个 token 无法完全恢复. 我们认为, 出现这种模式的原因是这些模型在预训练时没有为所有输入样本设置一致的起始 token. Llama-2 虽然在每个段落前加上一个 ``<s>`` token, 但这一步发生在文本分块之前, 因而第 0 个位置大多会被随机 token 占据. 缺少统一的起始 token, 导致模型把多个初始 token 用作注意力汇聚 token. 我们假设, 在所有训练样本开头加入一个稳定的可学习 token, 就能让它单独作为固定的注意力汇聚 token, 不再需要多个初始 token 来保证流式处理的一致性. 我们将在[第 3.3 节](#section-3-3) 验证这一假设.

<span id="section-3-2"></span>

### 3.2 带注意力汇聚 token 的滚动 KV 缓存

<span id="figure-04"></span>

![StreamingLLM 的 KV 缓存](./streaming-llm/figure-04.png)

**图 4.** StreamingLLM 的 KV 缓存.

为使已经训练好的 LLM 支持流式处理, 我们提出一种简单的方法, 无需对模型微调即可恢复窗口注意力的困惑度. 除当前滑动窗口内的 token 外, 我们还会在注意力计算中重新引入少量起始 token 的 KV. StreamingLLM 的 KV 缓存在概念上可分为两个部分, 如[图 4](#figure-04) 所示: (1) 注意力汇聚 token (4 个初始 token) 稳定注意力计算; (2) 滚动 KV 缓存保留对语言建模至关重要的最近 token. StreamingLLM 的设计很通用, 可以顺畅集成到任何采用 RoPE [Su21] 和 ALiBi [Pre21] 等相对位置编码的自回归语言模型中.

在确定相对距离并向 token 添加位置信息时, StreamingLLM 关注的是*缓存内*的位置, 而不是*原始文本中*的位置. 这一区别对 StreamingLLM 的性能不可或缺. 例如, 假设当前缓存 ([图 4](#figure-04)) 中的 token 为 [0, 1, 2, 3, 6, 7, 8], 并正在解码第 9 个 token, 那么分配的位置是 [0, 1, 2, 3, 4, 5, 6, 7], 而不是原始文本中的位置 [0, 1, 2, 3, 6, 7, 8, 9].

对于 RoPE 这类编码, 我们会在引入旋转变换*之前*缓存 token 的键. 随后, 在每个解码阶段对滚动缓存中的键应用位置变换. 另一方面, 与 ALiBi 集成更加直接. 此时对注意力分数应用连续的线性偏置, 而不是"跳跃"的偏置. 这种在缓存内分配位置嵌入的方法对 StreamingLLM 的功能不可或缺, 它保证模型即使超出预训练注意力窗口大小也能高效运行.

<span id="section-3-3"></span>

### 3.3 使用注意力汇聚 token 预训练 LLM

<span id="table-03"></span>

![普通注意力, 零汇聚 token 与可学习汇聚 token 的流式困惑度比较](./streaming-llm/table-03.png)

**表 3.** 比较普通注意力与预训练时在开头加入零 token 和可学习汇聚 token 的方法. 为保证流式困惑度稳定, 普通模型需要多个初始 token. 零汇聚方法略有改善, 但仍需要其他初始 token. 相比之下, 使用可学习汇聚 token 训练的模型只加入该汇聚 token, 就能保持稳定的流式困惑度. 缓存配置 $x$+$y$ 表示加入 $x$ 个初始 token 和 $y$ 个最近 token. 困惑度在 PG19 测试集的第一个样本上评估.

正如[第 3.1 节](#section-3-1) 所述, 模型过度关注多个初始 token 的一个主要原因是, 缺少指定的汇聚 token 来承接过多的注意力分数. 因而, 模型在无意间把全局可见的 token, 主要是初始 token, 用作注意力汇聚 token. 一种可能的补救方法是有意加入一个全局可训练的注意力汇聚 token, 记作"Sink Token", 用它接收不必要的注意力分数. 另一种可能有效的方法是把常规 SoftMax 函数替换为 SoftMax-off-by-One [Mil23] 这样的变体:

<span id="equation-02"></span>

$$
\mathrm{SoftMax}_1(x)_i=\frac{e^{x_i}}{1+\sum_{j=1}^{N}e^{x_j}}
$$

它不要求所有上下文 token 上的注意力分数之和为 1. 注意, $\mathrm{SoftMax}_1$ 等价于在注意力计算中前置一个键和值特征均为全零的 token. 为使其符合我们的框架, 我们把这种方法记作"Zero Sink".

为验证这些方法, 我们从头预训练了 3 个参数量为 1.6 亿, 设置完全相同的语言模型. 第一个模型采用标准 SoftMax 注意力 (Vanilla), 第二个模型把常规注意力机制替换为 $\mathrm{SoftMax}_1$ (Zero Sink), 第三个模型在所有训练样本开头加入一个可学习的占位 token (Sink Token). 如[表 3](#table-03) 所示, 零汇聚方法虽然在一定程度上缓解了注意力汇聚问题, 模型却仍依赖其他初始 token 作为注意力汇聚 token. 引入汇聚 token 对稳定注意力机制非常有效. 只把该汇聚 token 与最近 token 配对使用, 就足以锚定模型性能, 最终的评估困惑度甚至还有小幅改善. 基于这些结果, 我们建议未来训练 LLM 时在所有样本中加入汇聚 token, 以优化流式部署.

<span id="section-4"></span>

## 4 实验

我们使用近期 4 个具有代表性的模型家族评估 StreamingLLM: Llama-2 [Tou23a], MPT [Mos23], PyThia [Bid23] 和 Falcon [Alm23]. 值得注意的是, Llama-2, Falcon 和 Pythia 采用 RoPE [Su21], MPT 则采用 ALiBi [Pre21], 二者是近期研究中最有影响力的两种位置编码技术. 多样化的模型选择保证了研究结果的有效性和稳健性. 我们把 StreamingLLM 与稠密注意力, 窗口注意力和滑动窗口重计算等成熟基线进行比较. 除非另有说明, 后续所有 StreamingLLM 实验都默认使用 4 个初始 token 作为注意力汇聚 token.

<span id="section-4-1"></span>

### 4.1 不同 LLM 家族与规模上的长文本语言建模

我们首先使用拼接后的 PG19 [Rae20] 测试集评估 StreamingLLM 的语言建模困惑度, 该测试集包含 100 本长篇书籍. 对 Llama-2 模型, 缓存大小设为 2048; 对 Falcon, Pythia 和 MPT 模型, 缓存大小设为 1024. 这是预训练窗口大小的一半, 如此选择是为了让可视化更清晰.

[图 3](#figure-03) 表明, StreamingLLM 在 20K token 文本上的困惑度可以与 oracle 基线 (滑动窗口重计算) 相当. 与此同时, 当输入长度超过预训练窗口时, 稠密注意力技术会失效; 当输入长度超过缓存大小并导致初始 token 被逐出时, 窗口注意力技术也难以工作. 在[图 5](#figure-05) 中, 我们进一步证实 StreamingLLM 可以在多个模型家族和规模上可靠处理超过 400 万个 token 的超长文本. 这些模型包括 Llama-2-[7, 13, 70]B, Falcon-[7, 40]B, Pythia-[2.8, 6.9, 12]B 和 MPT-[7, 30]B.

<span id="figure-05"></span>

![多种 LLM 家族和规模上 StreamingLLM 对 400 万 token 超长文本的语言建模困惑度](./streaming-llm/figure-05.png)

**图 5.** 多种 LLM 家族和规模上, StreamingLLM 对 400 万 token 超长文本的语言建模困惑度. 困惑度始终保持稳定. 我们使用拼接后的 PG19 测试集 (100 本书) 进行语言建模, 书籍切换会造成困惑度波动.

<span id="section-4-2"></span>

### 4.2 使用汇聚 token 预训练的结果

为验证在所有预训练样本中引入汇聚 token 可以改善流式 LLM 这一建议, 我们在完全相同的条件下训练了两个参数量均为 1.6 亿的语言模型. 一个模型沿用原始训练设置, 另一个模型在每个训练样本开头加入汇聚 token. 实验使用 Pythia-160M [Bid23] 代码库, 并遵循其训练方案. 我们在一台配备 8 块 NVIDIA A6000 GPU 的服务器上, 使用去重后的 Pile [Gao20] 数据集训练模型. 除了把训练批大小减小为 256 外, 我们保留了 Pythia 的全部训练配置, 包括学习率调度, 模型初始化和数据集排列. 两个模型均训练 143,000 步.

<span id="figure-06"></span>

![有无汇聚 token 的模型预训练损失曲线](./streaming-llm/figure-06.png)

**图 6.** 有汇聚 token 和无汇聚 token 的模型预训练损失曲线. 两个模型的收敛趋势相近.

<span id="table-04"></span>

![有无汇聚 token 的模型在 7 个 NLP 基准上的零样本准确率](./streaming-llm/table-04.png)

**表 4.** 7 个 NLP 基准上的零样本准确率 (%), 包括 ARC-[Challenge, Easy], HellaSwag, LAMBADA, OpenbookQA, PIQA 和 Winogrande. 预训练时加入汇聚 token 不会损害模型性能.

**收敛与常规模型性能.** 预训练期间加入汇聚 token 不会对模型收敛和随后在一系列 NLP 基准上的性能产生负面影响. 如[图 6](#figure-06) 所示, 使用汇聚 token 训练的模型与普通模型呈现相近的收敛动态. 我们在 7 个不同的 NLP 基准上评估这两个模型, 包括 ARC-[Challenge, Easy] [Cla18], HellaSwag [Zel19], LAMBADA [Pap16], OpenbookQA [Mih18], PIQA [Bis20] 和 Winogrande [Sak19]. 如[表 4](#table-04) 所示, 使用汇聚 token 预训练的模型与采用普通方法训练的模型表现相近.

**流式性能.** 如[表 3](#table-03) 所示, 采用传统方法训练的模型和加入汇聚 token 的模型在流式困惑度上有所不同. 普通模型需要加入多个 token 作为注意力汇聚 token, 才能保持稳定的流式困惑度. 相比之下, 使用汇聚 token 训练的模型仅用该汇聚 token 就能取得令人满意的流式性能.

<span id="figure-07"></span>

![有无汇聚 token 的预训练模型平均注意力 logit 对比](./streaming-llm/figure-07.png)

**图 7.** 对比有汇聚 token (右) 和无汇聚 token (左) 的预训练模型在 256 个句子上的平均注意力 logit, 每个句子长 16 个 token. 两幅图显示相同的层与注意力头. 主要观察结果为: (1) 没有汇聚 token 时, 模型在较低层呈现局部注意力, 在较深层对初始 token 的注意力增加. (2) 有汇聚 token 时, 所有层都会明显把注意力投向它, 从而有效收集多余的注意力. (3) 汇聚 token 出现后, 其他初始 token 获得的注意力减少, 说明指定汇聚 token 有助于改善流式性能.

<span id="figure-08"></span>

![StreamEval 中的第一个样本](./streaming-llm/figure-08.png)

**图 8.** StreamEval 中的第一个样本.

**注意力可视化.** [图 7](#figure-07) 比较了有无汇聚 token 的预训练模型的注意力图. 与 Llama-2-7B ([图 2](#figure-02)) 相似, 无汇聚 token 的模型在较早的层中呈现局部注意力, 较深的层则关注初始 token. 相比之下, 使用汇聚 token 训练的模型会在所有层和注意力头中持续聚焦汇聚 token, 表明注意力卸载机制有效. 模型高度关注汇聚 token, 而对其他初始 token 的注意力减少, 这解释了汇聚 token 为何能有效改善模型的流式性能.

<span id="table-05"></span>

![指令微调 Llama-2-Chat 模型在 ARC 数据集上的流式问答准确率](./streaming-llm/table-05.png)

**表 5.** ARC-[Easy, Challenge] 数据集上的准确率 (%). 问题被拼接起来并以流式方式回答, 以模拟现实中的聊天场景. 稠密基线因内存不足 (OOM) 而失败. 窗口注意力的准确率很低. StreamingLLM 的结果与单样本逐一处理的 one-shot 基线相当. 窗口注意力和 StreamingLLM 的缓存大小均为 1024.

<span id="section-4-3"></span>

### 4.3 指令微调模型上的流式问答结果

为说明 StreamingLLM 在现实场景中的适用性, 我们使用现实中常见的指令微调 LLM 模拟多轮问答.

首先, 我们拼接 ARC-[Challenge, Easy] 数据集中的所有问答对, 把连续的数据流输入 Llama-2-[7, 13, 70]B-Chat 模型, 并在每个答案位置使用精确匹配标准评估模型补全. 如[表 5](#table-05) 所示, 稠密注意力会触发内存不足 (OOM) 错误, 表明它不适合这一场景. 窗口注意力方法虽能高效运行, 但输入长度超过缓存大小后会随机输出, 因而准确率较低. 相比之下, StreamingLLM 能高效处理流式格式, 表现出色, 准确率与 one-shot 逐样本基线一致.

为突出更适合 StreamingLLM 的场景, 我们引入一个受 LongEval [Li23h] 基准启发的数据集 StreamEval. 如[图 8](#figure-08) 所示, LongEval 采用横跨长文本的单次查询设置, StreamEval 则每新增 10 行信息就查询一次模型. 每次查询的答案始终位于 20 行之前, 这反映了现实中的情形, 因为问题通常与近期信息有关. 如[图 9](#figure-09) 所示, 即使输入长度接近 120K token, 采用 StreamingLLM 的 LLM 仍能保持合理的准确率. 相比之下, 稠密注意力和窗口注意力分别在达到预训练文本长度和 KV 缓存大小时失效. 此外, 我们使用两个上下文扩展模型 LongChat-7b-v1.5-32k [Li23h] 和 Llama-2-7B-32K-Instruct [Tog23], 说明 StreamingLLM 可以与上下文扩展技术结合. 在 StreamingLLM 中, 上下文扩展意味着扩大流式 LLM 的最大缓存大小, 使其能够捕捉更广的局部信息.

<span id="figure-09"></span>

![StreamEval 基准上的性能](./streaming-llm/figure-09.png)

**图 9.** StreamEval 基准上的性能. 准确率在 100 个样本上取平均值.

<span id="section-4-4"></span>

### 4.4 消融研究

**初始 token 的数量.** 在[表 2](#table-02) 中, 我们消融了把不同数量的初始 token 与最近 token 一同加入后, 对流式困惑度的影响. 结果表明, 只引入一个或两个初始 token 并不足够, 4 个初始 token 则似乎已经达到足够的阈值, 再继续加入只会产生很小的影响. 这一结果支持我们在 StreamingLLM 中引入 4 个初始 token 作为注意力汇聚 token 的选择.

**缓存大小.** 在[表 6](#table-06) 中, 我们评估了缓存大小对 StreamingLLM 困惑度的影响. 与直觉相反, 增大缓存并不能始终降低语言建模困惑度. 这一不一致现象暴露出一种潜在限制: 这些模型可能没有最大限度地利用所接收的全部上下文. 未来研究应着力提高这些模型更好利用大范围上下文的能力.

<span id="table-06"></span>

![缓存大小对 StreamingLLM 性能的影响](./streaming-llm/table-06.png)

**表 6.** 缓存大小对 StreamingLLM 性能的影响. 增大 StreamingLLM 的缓存并不能始终降低困惑度, 说明这些模型可能没有充分利用提供的上下文. 缓存配置 $x$+$y$ 表示加入 $x$ 个初始 token 和 $y$ 个最近 token. 困惑度在拼接后的 PG19 测试集的 400K token 上评估.

<span id="section-4-5"></span>

### 4.5 效率结果

<span id="figure-10"></span>

![StreamingLLM 与滑动窗口重计算基线的逐 token 解码延迟和内存用量比较](./streaming-llm/figure-10.png)

**图 10.** 比较滑动窗口重计算基线与 StreamingLLM 的逐 token 解码延迟和内存用量, X 轴为缓存大小 (注意力窗口大小). StreamingLLM 每个 token 最多可加速 22.2$\times$, 同时保持与重计算基线相近的内存占用.

我们比较了 StreamingLLM 与滑动窗口重计算的解码延迟和内存用量, 后者是唯一质量可接受的基线. 两种方法均使用 Huggingface Transformers 库 [Wol20] 实现, 并在单块 NVIDIA A6000 GPU 上使用 Llama-2-7B 和 Llama-2-13B 模型测试. 如[图 10](#figure-10) 所示, 随着缓存大小增加, StreamingLLM 的解码速度线性增长. 滑动窗口重计算基线的解码延迟则呈二次增长. 因此, StreamingLLM 获得了相当可观的加速, 每个 token 最多达到 22.2$\times$. StreamingLLM 在降低延迟的同时, 仍保持与重计算基线一致的内存占用.

<span id="section-5"></span>

## 5 结论

在流式应用中部署 LLM 已是迫切需求, 但效率限制和长文本上的性能下降带来了挑战. 窗口注意力提供了部分解决方案, 可一旦排除初始 token, 其性能就会骤降. 认识到这些 token 具有"注意力汇聚 token"的作用后, 我们提出 StreamingLLM, 一个简单而高效的框架, 使 LLM 无需微调即可处理无限长文本. 通过把注意力汇聚 token 与最近 token 一同加入, StreamingLLM 可以高效建模长达 400 万个 token 的文本. 我们进一步表明, 使用专用汇聚 token 预训练模型可以改善流式性能. StreamingLLM 首次将 LLM 的预训练窗口大小与实际文本生成长度解耦, 为 LLM 的流式部署打开了道路.

**可复现性声明.** 本文呈现的所有结果均可复现. 我们已在这个 [GitHub 仓库](https://github.com/mit-han-lab/streaming-llm)中公开代码和数据集. 本文使用的模型均已公开, 我们也提供了访问这些模型的参考资料. 实验细节包括超参数, 训练协议和评估方法, 可在实验部分 ([第 4 节](#section-4)) 找到. 我们相信, 读者可以利用所提供的资源复现本文呈现的全部结果.

**影响声明.** StreamingLLM 已被多种 LLM 服务方案广泛采用, 包括 [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#run-llama-with-streamingllm), [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers), [HuggingFace Transformers](https://huggingface.co/docs/transformers/v4.39.3/en/internal/generation_utils#transformers.SinkCache) 和 [MLC LLM](https://github.com/mlc-ai/mlc-llm/pull/1459) 等.

## 致谢

本工作得到 MIT-IBM Watson AI Lab, Amazon and MIT Science Hub 和 National Science Foundation 的支持. 我们感谢 Angela Li 提供写作建议并制作演示, 感谢 Jingwei Zuo 校对, 也感谢 Xiuyu Li 对符号表示提出建议.

<span id="section-6"></span>

## 6 讨论

**应用.** StreamingLLM 尤其适合多轮对话等流式应用, 这类应用需要持续运行, 又不能过度依赖大量内存或历史数据. 例如, 在基于 LLM 的日常助手应用中, StreamingLLM 能使模型长时间顺畅运行. 它根据最近的交互生成回答, 因而无需频繁刷新缓存. 传统方法可能要在对话长度超过训练长度时重置缓存, 导致近期上下文丢失; 也可能需要根据最近的文本历史重新计算键值 (KV) 状态, 而这种做法效率较低.

**局限性.** StreamingLLM 虽能提高流式场景中 LLM 的效率, 却不会扩展模型的上下文窗口或增强其长期记忆能力. 如[第 8 节](#section-8) 所述, 模型只能在当前缓存的范围内运行. 因此, StreamingLLM 不适合长文档问答 (QA) 和摘要等需要长期记忆并依赖大量数据的任务. 不过, 它很适合日常对话和短文档 QA 等只需短期记忆的场景, 优势在于无需刷新缓存, 即可根据近期上下文生成连贯文本.

**更广泛的社会影响.** StreamingLLM 显著提高了 LLM 的效率和可用性, 让各领域都能更容易地使用这类模型. 在对话智能体等应用中, StreamingLLM 支持不间断的快速交互, 改善了用户体验, 尤其适用于需要固定长度模型的场景. 这一进展使对话更加顺畅, 也更能感知上下文, 因而可能惠及教育, 医疗和客户服务等领域. StreamingLLM 还能凭借处理效率降低计算负载, 符合发展环境可持续 AI 技术的需求. 这一点有助于让技术资源有限的地区更容易使用先进 AI 工具. 不过, StreamingLLM 潜在的负面影响与一般语言模型相同, 例如错误信息和生成偏见内容的风险. 必须用完善的伦理准则和安全措施应对这些风险. 总之, StreamingLLM 虽然与语言模型共有一些风险, 但它在改善用户体验, 普及 AI 使用和促进可持续发展方面也有显著的积极贡献. 这些益处表明, 负责任地部署和合乎伦理地使用该技术十分重要.

<span id="section-7"></span>

## 7 更多相关工作

**稀疏 Transformer.** 有关高效 Transformer 模型的文献主要关注降低自注意力机制的计算和内存复杂度. 一条相关研究路线通过把注意力范围限制在固定的预定义模式上, 例如局部窗口或步幅固定的分块模式, 对注意力矩阵进行稀疏化 [Tay22a]. Sparse Transformer [Chi19] 为注意力矩阵引入稀疏分解, 把注意力的计算复杂度降至 $O(n\sqrt{n})$. LongFormer [Bel20] 把膨胀局部窗口注意力与由任务驱动的全局注意力结合起来. Extended Transformer Construction (ETC) [Ain20] 提出一种新的全局-局部注意力机制, 包含 4 类注意力模式: 全局到全局, 局部到局部, 局部到全局和全局到局部. BigBird [Zah20] 以 ETC 为基础, 提出另一种线性复杂度注意力方案, 使用全局 token, 局部滑动窗口注意力和随机注意力. 不过, 这些方法存在若干限制. 第一, Sparse Transformer 和 ETC 需要为一种特定的块稀疏矩阵乘法变体编写自定义 GPU 内核. 第二, LongFormer, ETC 和 BigBird 都依赖全局注意力模式, 不适用于自回归语言模型. 第三, 这些方法与预训练模型不兼容, 必须从头重新训练. 相比之下, 我们的方法可以用标准 GPU 内核轻松实现, 并与采用稠密注意力的预训练自回归语言模型兼容, 而这类模型在 NLP 社区很常见. 这种兼容性带来显著优势, 无需任何微调即可利用现有预训练模型.

**同期工作.** 我们的研究与 Han 等人 [Han23x] 的工作同期开展, 后者从理论上研究语言模型长度泛化失效的问题, 并找出 3 个分布外因素. 受这一分析启发, 他们的方法采用"$\Lambda$"形注意力模式, 并重新配置位置编码距离, 以改善 LLM 的长度泛化. 这种方法与我们的方法相似. 不过, 我们的工作发现了"注意力汇聚"现象, 即 Transformer 模型倾向于给语义信息很少的初始 token 分配较高的注意力分数. 这一现象超出了长度泛化失效的范围, 表明 Transformer 模型中存在一个更普遍的问题. 我们不仅在自回归语言模型中观察到这种"注意力汇聚"行为, 也在 BERT 等编码器 Transformer ([第 13 节](#section-13)) 和 Vision Transformer (ViT) [Dar23a] 中观察到它, 说明该现象在 Transformer 架构中更为普遍. 为缓解"注意力汇聚"现象, 我们提出在预训练时引入一个可学习的汇聚 token, 并通过大量消融研究支持我们的结论.

Darcet 等人 [Dar23a] 同期观察到 Vision Transformer 中的注意力也会聚集到随机背景 patch token, 并把这些 token 称为"register". 这些 register 用来存储全局图像信息. 他们的解决方案是加入专用的"register" token, 以平衡注意力分布. 我们发现的"注意力汇聚"与这一概念相似. 在本文中, "注意力汇聚 token"是会不成比例地吸引后续 token 注意力的初始 token. 在预训练期间引入专用汇聚 token, 可以防止模型不当地把内容 token 用作注意力汇聚 token, 使注意力分布更有效. 不过二者有一项主要差异: Vision Transformer 中的"register"在中间层充当全局信息存储区, 我们的"注意力汇聚 token"则位于自回归模型的开头. 这种位置差异说明, 注意力计算中的 softmax 函数可能在注意力汇聚 token 的产生过程中发挥更根本的作用.

<span id="section-8"></span>

## 8 查询-答案行距增加时的 StreamEval 准确率

<span id="table-07"></span>

![查询-答案距离不断增加时 StreamEval 的准确率](./streaming-llm/table-07.png)

**表 7.** 查询-答案距离不断增加时 StreamEval 上的准确率 (%). StreamEval 每行包含 23 个 token. 准确率在 100 个样本上取平均值, 每个样本包含 100 个查询.

为评估 StreamingLLM 处理长输入的能力, 我们在 StreamEval 上评估了 Llama-2-7B-32K-Instruct 模型, 关注不同缓存配置下的不同查询-答案行距. StreamEval 每行包含 23 个 token, 因而行距对应的 token 距离为 $23\times\text{行距}$. 准确率在 100 个样本上取平均值, 每个样本包含 100 个查询. [表 7](#table-07) 表明, 当查询与答案的 token 距离在缓存大小以内时, StreamingLLM 能保持准确率. 不过, 距离增加时准确率会下降, 超过缓存容量后最终降至零.

这些结果表明, StreamingLLM 虽能依据近期上下文有效生成连贯文本, 却不能扩展语言模型的上下文长度. 这些结果还反映了当前语言模型所面临的一个更普遍的难题: 它们无法充分利用缓存内的上下文信息, 这与 Liu 等人 [Liu23z] 的观察一致.

<span id="section-9"></span>

## 9 长距离基准评估

<span id="table-08"></span>

![StreamingLLM 与 LongBench 默认截断基线的性能比较](./streaming-llm/table-08.png)

**表 8.** StreamingLLM 与 LongBench [Bai23] 默认截断基线的性能比较. 基线把输入截断为前 1750 个和后 1750 个 token. StreamingLLM 4+3496 使用 4 个注意力汇聚 token 和 3496 个最近 token, StreamingLLM 1750+1750 则在初始和最近两个部分各使用 1750 个 token.

我们使用 Llama-2-7B-chat 模型 (最大上下文长度为 4k) 在 LongBench [Bai23] 上评估 StreamingLLM, 该基准包含 3 类主要 NLP 任务: 单文档 QA (NarrativeQA [Koi18] 和 Qasper [Das21]), 多文档 QA (HotpotQA [Yan18a] 和 2WikiMQA [Ho20]) 以及摘要 (GovReport [Hua21], MultiNews [Fab19]). LongBench 为 Llama-2-7B-chat 模型设置的默认最大序列长度为 3,500 个 token, 从中间截断, 以保留开头和结尾的信息 (各 1,750 个 token). [表 8](#table-08) 表明, 缓存配置为 4+3496 的 StreamingLLM 不如截断基线, 可能是因为重要的输入提示开头信息丢失. 不过, 把注意力汇聚 token 数量调为 1750 后, 性能可以恢复到文本截断基线的水平. 这些结果印证了[第 8 节](#section-8) 的发现, 表明 StreamingLLM 的效果取决于缓存中的信息, 且缓存内性能与文本截断基线相当.

<span id="section-10"></span>

## 10 Llama-2-7B 在更长序列上的注意力可视化

<span id="figure-11"></span>

![Llama-2-7B 在 256 个长度为 128 的句子上的平均注意力 logit](./streaming-llm/figure-11.png)

**图 11.** Llama-2-7B 在 256 个长度均为 128 的句子上的*平均*注意力 logit 可视化.

为便于观察, [图 2](#figure-02) 使用短序列 (长度为 16) 可视化了 Llama-2-7B 的注意力图. 我们进一步在[图 11](#figure-11) 中可视化 Llama-2-7B 在较长序列 (长度为 128) 上的注意力. 我们发现, 短序列上的观察结果在长序列上同样成立: 无论初始 token 与序列中其余 token 相距多远, 在大多数层中, 初始 token 的注意力分数都远高于其他 token. 序列越长, 热力图中注意力汇聚 token 的分数看起来越细. 我们在[第 11 节](#section-11) 使用另一种方法进一步分析了更长序列 (长度为 4096) 上的注意力分布.

<span id="section-11"></span>

## 11 长输入中注意力汇聚的定量分析

<span id="figure-12"></span>

![Llama-2-7B 各层对第一个 token 的注意力分数](./streaming-llm/figure-12.png)

**图 12.** Llama-2-7B 各层对第一个 token 的注意力分数 (经过 SoftMax) 可视化. 注意力分数表示第 4096 个 token 在各层对第一个 token 的注意力. 误差条表示同一层内不同注意力头对第一个 token 的注意力分数的标准差. 结果在 256 个句子上取平均值, 每个句子长 4096 个 token.

为便于观察, [图 2](#figure-02) 和[图 13](#figure-13) 使用短序列展示了注意力汇聚现象. 在此基础上, [图 12](#figure-12) 给出了长输入 (序列长度为 4096) 中第一个 token 所获注意力分数 (经过 SoftMax) 的分布. 我们在 256 个序列上对注意力分数取平均值, 每个序列包含 4096 个 token. 图中数据表示第 4096 个 token 在每一层分配给初始 token 的注意力. 可以看到, 除最底部两层外, 第一个 token 的注意力分数非常高, 经常超过注意力总量的一半. 这一观察从实证上说明, 无论序列中其他 token 的距离如何, 大多数层和注意力头都会优先关注第一个 token. 这种趋势表明初始 token 在序列中发挥重要作用, 因为移除它们会从 SoftMax 函数的分母中去掉很大一部分, 从而对语言模型性能产生巨大影响.

<span id="section-12"></span>

## 12 Llama-2-70B 注意力可视化

<span id="figure-13"></span>

![Llama-2-70B 在 256 个长度为 16 的句子上的平均注意力 logit](./streaming-llm/figure-13.png)

**图 13.** Llama-2-70B 在 256 个长度均为 16 的句子上的*平均*注意力 logit 可视化.

[图 2](#figure-02) 给出了 Llama-2-7B 的注意力可视化, 我们又在[图 13](#figure-13) 中进一步可视化了 Llama-2-70B 的注意力. 我们发现, Llama-2-7B 上的观察结果在 Llama-2-70B 上同样成立: 在大多数层中, 初始 token 的注意力分数都远高于其余 token.

<span id="section-13"></span>

## 13 编码器 Transformer 中的注意力汇聚

<span id="figure-14"></span>

![BERT-base-uncased 中示例句子的注意力图](./streaming-llm/figure-14.png)

**图 14.** BERT-base-uncased 中句子 *"StreamingLLM can work on infinite-length texts without compromising efficiency and performance."* 的注意力图可视化.

本文主要研究 GPT 和 Llama 等自回归, 仅解码器语言模型中的注意力汇聚现象. 根据[第 3.1 节](#section-3-1) 的认识, 我们认为该现象很可能也存在于其他 Transformer 架构中, 包括 BERT [Dev19] 等编码器模型和 ViT [Dos20]. 这一假设来自这些模型共有的相似 Transformer 结构和 SoftMax 注意力机制. 为验证这一假设, 我们分析了 BERT-base-uncased 的注意力模式, 如[图 14](#figure-14) 所示. 结果表明, BERT-base-uncased 存在注意力汇聚现象, 特征是大多数层都会给 `[SEP]` token 分配不成比例的高注意力分数. 这说明, 模型始终依赖无处不在的 `[SEP]` token 作为注意力焦点. Darcet 等人 [Dar23a] 的同期研究还在 Vision Transformer 中发现了类似的注意力尖峰, 其原因是随机背景 patch token 充当全局图像信息的"register". 我们认为, 这些"register"与我们观察到的注意力汇聚现象相似, 说明这可能是所有 Transformer 模型共有的特征.

<span id="section-14"></span>

## 14 在预训练阶段使用更多汇聚 token

[第 3.3 节](#section-3-3) 表明, 在预训练阶段加入一个专用汇聚 token 不会影响模型性能, 却能把注意力汇聚集中到一个 token 上, 从而改善流式性能. 本节研究在预训练时加入更多汇聚 token 能否进一步优化预训练语言模型的性能.

如[图 15](#figure-15) 所示, 实验结果表明, 预训练时加入一个或两个汇聚 token, 所得到的预训练损失曲线都与基线 (普通) 模型非常相近. 不过, 如[表 9](#table-09) 所示, 引入第二个汇聚 token 并没有在大多数基准任务上带来实质性性能改善.

[表 10](#table-10) 中的进一步分析表明, 加入更多汇聚 token 不会改善流式性能. 有趣的是, 模型似乎需要同时依赖两个汇聚 token, 才能保持稳定的流式性能. 这些结果表明, 一个汇聚 token 足以改善流式性能, 加入更多汇聚 token 并不会进一步提高语言模型的整体性能. 这与 Vision Transformer (ViT) [Dar23a] 中的发现不同, 后者使用多个"register"会有益处.

<span id="figure-15"></span>

![使用 0 个, 1 个和 2 个汇聚 token 的模型预训练损失曲线](./streaming-llm/figure-15.png)

**图 15.** 使用 0 个, 1 个和 2 个汇聚 token 的模型预训练损失曲线.

<span id="table-09"></span>

![使用 0 个, 1 个和 2 个汇聚 token 的模型在 7 个 NLP 基准上的零样本准确率](./streaming-llm/table-09.png)

**表 9.** 7 个 NLP 基准上的零样本准确率 (%), 包括 ARC-[Challenge, Easy], HellaSwag, LAMBADA, OpenbookQA, PIQA 和 Winogrande.

<span id="table-10"></span>

![普通注意力与预训练时在开头加入一个或两个可学习汇聚 token 的比较](./streaming-llm/table-10.png)

**表 10.** 比较普通注意力与预训练时在开头加入零 token 和可学习汇聚 token 的方法. 缓存配置 $x$+$y$ 表示加入 $x$ 个初始 token 和 $y$ 个最近 token. 困惑度在 PG19 测试集的第一个样本上评估.

[+1]: 部分工作在 Meta AI 实习期间完成.
