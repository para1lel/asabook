---
title: 'FastMoE: A Fast Mixture-of-Expert Training System'
createTime: 2026/09/07 21:06:24
permalink: /papers/fastmoe/
---

> [Jiaao He](https://dblp.org/pid/249/2660), [Jiezhong Qiu](https://jiezhongqiu.com/), [Aohan Zeng](https://blog.sengxian.com/), [Zhilin Yang](https://kimiyoung.github.io/), [Jidong Zhai](https://pacman.cs.tsinghua.edu.cn/~zjd/) 和 [Jie Tang](https://keg.cs.tsinghua.edu.cn/persons/jietang/). 论文于 2021 年 3 月 24 日首次提交至 arXiv, 当前版本为 v1. [FastMoE: A Fast Mixture-of-Expert Training System](https://arxiv.org/abs/2103.13262). <a href="/paper/fastmoe.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2103.13262). [TeX 源文件](https://export.arxiv.org/e-print/2103.13262). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

专家混合 (MoE) 很有希望把语言模型的规模扩大到万亿参数. 但是, 训练万亿参数规模的 MoE 需要算法与系统协同设计, 从而得到经过良好调优的高性能分布式训练系统. 遗憾的是, 唯一满足这些要求的现有平台严重依赖 Google 的硬件 (TPU) 和软件 (Mesh Tensorflow) 栈, 并未向公众开放, 尤其没有对 GPU 和 PyTorch 社区开放.

本文介绍 *FastMoE*, 一个基于 PyTorch, 使用通用加速器的分布式 MoE 训练系统. 该系统提供分层接口, 既支持灵活的模型设计, 也便于适配 Transformer-XL 和 Megatron-LM 等不同应用. 与直接用 PyTorch 实现 MoE 模型不同, *FastMoE* 采用复杂的高性能加速技术, 大幅优化了训练速度. 该系统支持把不同专家放到跨多个节点的多块 GPU 上, 因此专家数量可以随 GPU 数量线性增加. *FastMoE* 的源代码以 Apache-2 许可证发布在 [https://github.com/laekov/fastmoe](https://github.com/laekov/fastmoe).

<span id="section-1"></span>

## 1 引言

以 BERT [Dev18], GPT-2/-3 [Rad19b, Bro20b], XLNet [Yan19], RoBERTa [Liu19a], T5 [Raf20b], GShard [Lep20] 和 Switch Transformer [Fed22] 为代表的大规模语言模型近来相继出现, 彻底改变了自然语言处理研究的面貌, 并在 GLUE [Wan18d], SuperGLUE [Wan19h] 等多项基准上重新建立了新的先进基线.

在许多可行方案中, 扩大模型规模已被证明是构建更强模型最简单且最有效的方法之一 [Kap20]. 从拥有 $340$ million 参数的 BERT [Dev18], 到拥有 $11$ billion 参数的 T5 [Raf20b], 再到拥有 $175$ billion 参数的 GPT-3 [Bro20b], 模型规模仅在两年内就扩大了 $500\times$. 更近一些, GShard [Lep20] 将参数量推至创纪录的 $600$ billion, 随后又被拥有 $1.6$ trillion 参数的 Switch Transformer [Fed22] 很快打破. GShard 与 Switch Transformer 能达到如此大的模型规模, 主要归功于一种称为专家混合 (MoE) [Sha17] 的新型神经网络架构.

<span id="figure-01"></span>

![MoE 层示意图](./fastmoe/figure-01.png)

**图 1.** MoE 层的示意图. 在此例中, 门控选择专家 1 和专家 3 参与计算.

一个 MoE 层 (示意见[图 1](#figure-01)) 由一个门控和一个专家池组成. 对每个输入, 门控只选择极少数专家参与计算. MoE 的特殊架构对大规模分布式训练而言是一把双刃剑. 一方面, 由于只稀疏激活专家, MoE 可以把模型规模扩大几个数量级, 而计算量 (FLOPs) 不会显著增加. 另一方面, 当规模扩大到数千个专家时, MoE 中不均衡的全对全通信模式给算法与系统的协同设计带来了新问题. 因此, PyTorch [Pas19] 和 TensorFlow [Aba16] 等传统深度学习库无法直接支持 MoE.

新模型架构带来的问题使研究界与产业界都需要一种支持大规模分布式训练的 MoE 实现. 但是, 尽管 PyTorch 中已经有一些朴素的单 GPU 实现 [Rau19], 目前唯一支持可扩展 MoE 训练的系统仍建立在 Google 的私有软硬件栈之上, 即 TPU [Jou17a] 和 Mesh TensorFlow [Sha18a]. 因而, 迫切需要在公开可用的硬件 (如 GPU) 和平台 (如 PyTorch [Pas19]) 上开发 MoE 系统.

为了给大规模 MoE 训练提供易用, 灵活, 高效, 可扩展的开源方案, 我们发布了 *FastMoE*, 设计目标如下:

- **易用:** 提供方便用户定义 MoE 层的接口, 并无缝支持流行的语言模型训练系统 Megatron-LM [Sho19].
- **灵活:** 让用户可以方便地定制门控网络和专家网络.
- **高效:** 为 Transformer 集成高度优化的前馈 (FFN) 层.
- **可扩展:** 支持跨多个节点的多块 GPU 训练, 以扩大 MoE 模型的规模.

与此前的单 GPU PyTorch 实现 [Rau19] 不同, *FastMoE* 专注于效率和可扩展性. *FastMoE* 包含专用 CUDA 内核以及面向高性能的特定优化. *FastMoE* 可以利用 NCCL [Jea17] 跨多个节点的多块 GPU 运行. *FastMoE* 对模型开发者隐藏了通信细节. *FastMoE* 的模型并行方法允许把专家分布到不同 GPU, 而模型的其他部分仍按 batch 维度 (数据并行) 或 tensor 维度 (模型并行) 进行并行化. 模型规模与专家数量成正比, 因而可能随训练所用 GPU 的数量一起增长, 这是训练万亿参数规模模型的关键.

实验表明, 在单块 GPU 上, *FastMoE* 快于只用 PyTorch API 实现的基线 [Rau19]. 在通过 Infiniband 网络连接的集群上跨节点运行时, *FastMoE* 也表现出合理的可扩展性. 我们使用分布式 *FastMoE* 训练了一个真实的 GPT 模型, 每层含 $96$ 个专家, 端到端训练速度可观. 相比计算量相同的非 MoE 模型, 该模型因 MoE 架构扩大了模型规模而受益.

本文其余部分安排如下. [第 2 节](#section-2) 介绍 MoE 的背景并比较现有系统. [第 3 节](#section-3) 详细介绍 *FastMoE* 系统. [第 4 节](#section-4) 说明实现高性能所面临的问题以及 *FastMoE* 的解决方案. [第 5 节](#section-5) 给出实验结果, 说明 *FastMoE* 的效率以及使用 *FastMoE* 训练的 MoE 模型所获得的性能提升. [第 6 节](#section-6) 总结全文并指出未来工作方向.

<span id="section-2"></span>

## 2 专家混合 (MoE)

本节回顾 MoE 架构以及当前用于训练 MoE 的系统.

<span id="section-2-1"></span>

### 2.1 MoE: 模型结构

Mixture-of-Expert 是 [Sha17] 提出的 Sparsely-Gated Mixture-of-Experts 层的简称. 一个 MoE 层由多个专家组成, 每个专家都可以是任意神经网络. 对专家唯一的约束是接收相同输入, 并在相同向量空间中给出输出. [图 1](#figure-01) 详细展示了一个 MoE 层. 系统引入一个称为*门控网络*的特殊神经网络, 为给定输入上的每个专家评分. 随后, 由一种因模型而异的策略根据分数选择专家. 被选中的专家, 如示例中的专家 $1$ 和 $3$, 接着被激活并处理输入样本. 最后使用某种算法将专家输出与分数组合为最终输出.

一种常见的专家选择方式是选择分数最高的前 $k$ 个专家. 在合成过程中, 分数作为专家输出的权重, 用于把各专家的输出相加为总输出. 由于梯度可以通过分数传播, 这种方法能够训练门控网络. [算法 1](#algorithm-01) 形式化描述了上述方法.

<span id="algorithm-01"></span>

**算法 1: 使用 top-$k$ 门控的 MoE 层前向计算.**

- **要求:** 含 $n$ 个专家的专家池: $\{E_1,E_2,\cdots,E_n\}$.
- **要求:** 门控 $G$.
- **要求:** 要选择的专家数量 $k$.
- **函数** $\operatorname{MoE}(x)$:
  - $\mathit{score}\leftarrow G(x)$.
  - $\mathit{indices}\leftarrow \operatorname{ArgMax}_k(\mathit{score})$.
  - $y\leftarrow$ 与 $x$ 相似的零张量.
  - **对于**每个索引 $i\in\mathit{indices}$:
    - $x_i\leftarrow E_i(x)$.
    - $y\leftarrow \mathit{score}_i*x_i+y$.
  - **返回** $y$.

<span id="section-2-2"></span>

### 2.2 当前的 MoE 训练系统

GShard 系统 [Lep20] 实现了分布式版本的 MoE 模型. 它最多使用 $2048$ 个 TPU 训练语言模型, 每层在每个 TPU 上放置 $1$ 个专家. 因此, MoE 层的参数量是非 MoE 层的 $2048\times$. 在 Switch Transformer [Fed22] 中, 模型规模进一步扩大到 $1.6$ trillion, 表明该系统有很强的大规模模型训练能力. 遗憾的是, 该系统尚未公开. 它与 TPU 集群紧密绑定, 因而很难在通用设备上复现实验. 此外, GShard 的设计缺少灵活性, 难以在不同复制策略下使用不同数量和规模的专家.

Tensor2tensor [Vas18c] 提供了一个 MoE Transformer 模型. 但这一实现使用 Mesh TensorFlow [Sha18a], 对 GPU 的支持并不好. 在 Transformer 中实现一个 FFN 需要编写超过 $100$ 行 TensorFlow 代码, 其中包含复杂的 `einsum` 运算符, 开发者很难理解其结构, 也难以在这些代码上探索其他模型结构.

PyTorch [Pas19] 是研究人员常用的深度学习框架, 与 TensorFlow 相比, 它的编码风格更直接, 也更灵活. 已经有人尝试用 PyTorch 训练 MoE 模型 [Rau19]. 但是, PyTorch 社区缺少多维并行训练工具, 所有基于 PyTorch 的实现都不支持多 GPU 训练. 采用 MoE 的最终目标是训练更大的模型, 因而这些基于 PyTorch 的实现无法成为可用方案.

<span id="section-3"></span>

## 3 FastMoE: 系统设计

本节介绍支持分布式训练的 *FastMoE* 设计.

<span id="section-3-1"></span>

### 3.1 面向不同模型探索者的灵活系统

**运行任意专家网络的骨架.** *FastMoE* 支持用任意网络作为专家. *FastMoE* 的 `FMoE` 接口以任意神经网络模块构造器为输入, 将该模块复制多次作为专家实例. 专家接收一批对齐且连续的输入特征, 输出应保持相同的 batch 顺序. 因此, 专家模块的实现与 MoE 架构相互解耦, 开发者可以专注于设计自己的专家网络.

为提供更强的灵活性, `FMoE` 类包含成员函数 `expert_fn`, 专家模块通过该函数执行前向计算. 可以重载此函数以进一步定制 MoE 行为. 例如, 在本节后文将提到的 `FMoETransformerMLP` 网络中. 专家列表会被一个特别优化的模块取代, 该模块并行运行专家, 从而大幅降低延迟.

同时, *FastMoE* 支持把多个专家放在同一个 worker 上, 为专家数量提供更灵活的配置空间 (即专家数量不必等于数据并行数), 这与 GShard 的设计不同.

**面向 Transformer 的高度优化 FFN.** 为了更好地支持使用 MoE 训练 Transformer, *FastMoE* 提供了标准的高性能 FFN 实现 (`FMoETransformerMLP`). 具体优化策略对开发者隐藏.

具体来说, 在同一个 worker 上放置多个专家时, 朴素实现会遍历这些专家并依次执行前向计算. 但是, 对于某些类型的专家网络, 可以利用并行执行带来的潜在加速. 在 *FastMoE* 中, 我们主要通过专用的 `FMoELinear` 模块优化全连接层的并行执行. 经过专门优化的专家模块不再依次计算各个专家, 而是维护一个可用硬件资源池, 并行执行专家计算.

**以插件方式支持 PyTorch 和 Megatron-LM.** *FastMoE* 的灵活性使其可以方便地适配现有训练应用. 以 Megatron-LM [Sho19] 为例, *FastMoE* 集成了一个插件式模块, 可以把原 Megatron-LM 模型中的 FFN 快速替换为 MoE 网络. 如代码清单 1 所示, 只需 2 行代码即可完成转换.

**代码清单 1: 在 Megatron-LM 中使用 FastMoE 的示例代码.**

```python
from fmoe.megatron import fmoefy
model = fmoefy(model, num_experts=<number of experts per worker>)
```

`fmoefy` 函数可以找到 Transformer 层中的 FFN. 随后, 系统用 *FastMoE* 创建 MoE 网络, 该模块封装 `FMoETransformerMLP` 模块以保持接口层兼容性.

<span id="section-3-2"></span>

### 3.2 以分布式方式扩大模型容量

**FastMoE 的模型并行方法.** 容纳大量专家并对其进行并行训练, 是许多 MoE 模型用于扩大模型容量的最有效方法之一. 模型开发者很难处理 GPU 之间乃至跨节点的复杂数据传输. 要获得较高的训练性能与硬件资源利用率, 需要计算机体系结构和并行编程方面的专业知识, 超出了普通模型开发者的技术栈.

*FastMoE* 支持把专家分布到多个节点的多个 worker 上, 这称为 *FastMoE 中的模型并行方法*. 输入数据交换的细节隐藏在 `FMoE` 接口内. 模型开发者只需为单个专家编写代码, *FastMoE* 会从所有 worker 收集全部输入数据并交给各专家. 因此, 模型开发者无须考虑跨 worker 通信的实现细节.

在 *FastMoE* 的设计中, 启用跨 worker 分布专家的功能后, 前向和反向计算都会加入额外的通信操作. 为了清楚地区分这些操作, 我们称其为全局数据交换操作, 与[第 4 节](#section-4) 将介绍的本地数据重排过程相对.

分布式环境中的一个主要问题是, 分配给某个 worker 上全部专家的输入样本总数可能差异很大. 在得到门控输出之前, 无法知道传入样本的数量. 但是, 分配用于存放输入样本的缓冲区又依赖这一数量. 因此, worker 在实际交换输入样本前, 要先交换数量信息, 再根据对专家计数信息的检查分配内存.

<span id="figure-02"></span>

![全局操作示例](./fastmoe/figure-02.png)

**图 2.** 全局操作的示例.

[图 2](#figure-02) 展示了 *FastMoE* 中的全局操作示例. 各 worker 首先统计本 worker 上每个专家被分配到的样本数. 随后, 它们交换专家输入的大小, 使所有 worker 得到传入样本的数量及其来源. 计算出各接收缓冲区的偏移后, worker 开始直接交换数据. 值得注意的是, 传入和传出样本的统计信息可以在整个训练迭代过程中重复使用.

**异构感知同步模块.** 网络的不同部分可能在不同的 worker 组之间复制, 由此产生异构性. 分布式模块需要判断某个参数的梯度是否应该同步, 以及应与谁同步. *FastMoE* 为每个参数引入*数据并行通信组*标签来处理这一问题.

标签可以取 `world`, `data parallel` 或 `none`, 分别表示梯度应该与 (1) 所有其他 worker, (2) 与模型并行组正交的数据并行组中的 worker 同步, 或 (3) 不与任何 worker 同步. 例如, 无论如何设置模型并行, 门控网络都会在所有 worker 上复制. 注意力层可能被拆分成模型并行子层, 因而其标签是 `data parallel`. 每个 worker 承载若干独有的专家网络, 其标签为 `none`. *FastMoE* 提供了定制的数据并行模块来替代 PyTorch 原有的分布式数据并行模块, 它可以识别这些标签并执行正确的同步.

<span id="section-4"></span>

## 4 实现高性能的优化

单节点上的 MoE 计算性能很重要, 因为它决定了系统扩展到任意规模时的理论上限.

计算 MoE 层最直观的方法是把输入 batch 切成样本, 再逐个样本计算. 随后, 按原始顺序堆叠输出特征. 但实际观察发现, 使用简单的 PyTorch 运算符实现 MoE 模型很难获得高性能. 所能达到的性能不到 GPU 峰值性能的 $5\%$.

<span id="figure-03"></span>

![不同问题规模下的 GeMM 性能](./fastmoe/figure-03.png)

**图 3.** 在 NVIDIA V100 上使用 `cuBLAS` 处理不同问题规模时的 `GeMM` 性能.

不失一般性, 我们假设专家网络是 FFN. 注意, FFN 中的主要运算来自全连接层, 它们由若干 `GeMM` 运算符组成. 当 batch 被拆成单个样本时, `GeMM` 运算会退化为 `GeMV`. [图 3](#figure-03) 给出了一个示例全连接层在不同 batch 大小下的浮点计算吞吐率. 现代异构计算设备上的矩阵乘法运算符会在所有维度应用复杂的分块技术进行精细调优, 因此只有 batch 足够大时, 吞吐率才能接近理论峰值. 由此可知, 为了让 MoE 计算获得高性能, 应将样本组成 batch, 以充分利用硬件资源.

*FastMoE* 把送往同一专家的所有输入样本放在一个 batch 中. 受数据表示的限制, *FastMoE* 使用专门开发的 CUDA 内核执行内存移动, 以减少开销. 给定每个样本将前往的门控索引, 把送往同一门控的全部输入样本放入连续内存空间的过程称为 `scatter`. 但是, 在神经网络的其他部分, batch 可能必须保持原始顺序, 例如 Transformer 的注意力层. 专家把结果输出到另一块连续内存空间后, 系统执行逆向操作, 即依据门控索引把重排后的特征向量放回原来的顺序. *FastMoE* 将这一过程称为 `gather`.

<span id="figure-04"></span>

![MoE 层重排计算示例](./fastmoe/figure-04.png)

**图 4.** MoE 层重排计算的示例

[图 4](#figure-04) 展示了重排计算过程. 当输入样本到专家的分配足够均衡时, 根据[图 3](#figure-03), 每个专家都有望获得较大的输入 batch, 从而达到令人满意的硬件利用率. 然而, 输入训练数据采用随机采样, 因而负载不均衡总会出现. 在数百万次训练迭代中, 某个专家很可能只接收到很少的输入样本. 此外, 在一个 worker 上放置多个专家时, 从统计上看, 专家的本地 batch 大小平均低于数据并行中的 batch 大小. *FastMoE* 使用定制的 stream manager 同时执行多个专家的计算, 以获取潜在的吞吐率提升.

<span id="section-5"></span>

## 5 评估

本节在单块 GPU 上将 *FastMoE* 的训练速度与另一个 PyTorch MoE 实现 [Rau19] 比较. 我们还报告 *FastMoE* 在分布式训练时的可扩展性. 据我们所知, *FastMoE* 是唯一可以跨不同节点和 GPU 运行的基于 PyTorch 的 MoE 系统. 我们还展示了使用 *FastMoE* 训练的 MoE Transformer 模型的端到端性能.

<span id="section-5-1"></span>

### 5.1 实验设置

我们用以下符号描述计算任务: 每块 GPU 放置 $n_e$ 个专家. 每个专家分别应用大小为 $d_m\times d_h$ 和 $d_h\times d_m$ 的两个线性层. 输入包含 $n_b$ 个样本. 门控模块为每个专家处理各样本的适合程度评分. 对每个输入样本, 选择分数最高的前 $k$ 个专家来处理该样本.

另外, 实验会执行若干轮预热, 其计算内容相同, 但不计入结果. 每项实验执行任务 $16$ 次, 并用平均时间计算性能. 我们也检查了执行时间的标准差. 它们均可忽略不计.

<span id="section-5-2"></span>

### 5.2 单 GPU 训练速度

我们在 NVIDIA TESLA V100 PCIe GPU 上测试 `FMoETransformerMLP` 的性能, 它完成的任务与基线 [Rau19] 中的 `MoE` 模块相似. 基线仅用 PyTorch API 实现, 模型结构采用硬编码. 为保证比较公平, 两个模块都使用随机初始化的矩阵作为门控网络的权重, 该门控网络由一个全连接层组成. 专家也执行相同的计算.

<span id="figure-05"></span>

![FastMoE 与基线的计算时间比较](./fastmoe/figure-05.png)

*延迟测试采用 $n_b=4096,d_m=1024,d_h=4096,k=2$.*

**图 5.** *FastMoE* 与基线实现的计算时间比较.

如[图 5](#figure-05) 所示, 基线实现始终比 *FastMoE* 慢. 随着专家数量增加, 基线在前向计算上花费的时间大幅增长, 而 *FastMoE* 的延迟保持稳定, 这得益于[第 4 节](#section-4) 提到的定制 stream manager. 考虑到 *FastMoE* 面向训练, 反向时间堆叠在前向时间之上. 我们观察到, *FastMoE* 在每次迭代的总耗时上优于基线.

<span id="section-5-3"></span>

### 5.3 跨 GPU 与跨节点可扩展性

为了考察 *FastMoE* 扩展到跨节点多块 GPU 时的性能, 我们在一个含 $8$ 个节点的集群上进行实验, 每个节点配备 $1$ 块 NVIDIA Tesla V100 GPU. 集群通过 Infiniband EDR 交换机和 $8$ 块 HCA 卡互连. 我们计算矩阵乘法运算的 FLOPs 来表示训练吞吐率.

<span id="figure-06"></span>

![FastMoE 跨多块 GPU 和多个节点的可扩展性](./fastmoe/figure-06.png)

*吞吐率测试采用 $n_e=4,n_b=4096,d_m=1024,d_h=4096,k=2$.*

**图 6.** *FastMoE* 跨多块 GPU 和多个节点的可扩展性

根据[图 6](#figure-06) 的结果, *FastMoE* 展现出跨节点可扩展性. GPU 数量从 $2$ 增加到 $8$ 时, 总吞吐率从 $10$ `TFLOPs` 增至 $25$ `TFLOPs`, 呈次线性扩展. 我们观察到, 扩展到 $2$ 块 GPU 后, 性能只有单块 GPU 的一半, 这表明 *FastMoE* 受通信限制. 使用更多 GPU 计算时会引入更多专家, 输入样本的交换粒度也会变小, 从而降低网络数据传输效率.

总之, *FastMoE* 的可扩展性能够以更高性能支持跨多个节点, 使用多块 GPU 训练大型 MoE 模型. 但是, 吞吐率仍有进一步优化的空间.

<span id="section-5-4"></span>

### 5.4 使用 FastMoE 获得端到端性能提升

我们用 Megatron-LM [Sho19] 在 $8$ 块 GPU 上训练一个 12 层 GPT 模型, 测试使用 *FastMoE* 带来的端到端性能提升. 如[第 3 节](#section-3) 所述, MoE 结构使用 *FastMoE* 的 Megatron adapter. 每层有 $96$ 个专家分布在这些 GPU 上, 即每块 GPU 放置 $12$ 个专家. 对每个输入 token, 选择分数最高的前 $2$ 个专家进行处理. 专家 MLP 层中的 $d_h$ 减半, 使模型的有效 FLOPs 几乎相同, 唯一差别是门控引入了额外 FLOPs, 但这部分可以忽略. 基线模型和 MoE 模型都训练 $70$ 小时. 训练中的 `lm loss` 指标反映模型的收敛趋势.

<span id="figure-07"></span>

![使用 FastMoE 训练 GPT 模型的 loss 曲线](./fastmoe/figure-07.png)

*较窄的深色线是对原始 loss 曲线以 $0.97$ 做指数平滑后的结果, 较亮的宽曲线分别表示原始曲线.*

**图 7.** 使用 *FastMoE* 训练 GPT 模型的 loss 曲线

从[图 7](#figure-07) 可见, 基线模型的训练速度约为 *FastMoE* 的 $3\times$. *FastMoE* 会执行更多计算和通信, 这样的减速是合理的. 所幸, 在相同训练迭代次数下, MoE 模型得到的 loss 低得多. 同时, 得益于 *FastMoE* 的效率, MoE 模型在相同训练时间内也得到了更低的 loss.

<span id="section-6"></span>

## 6 总结与未来工作

本文介绍 *FastMoE*, 一个用于训练专家混合模型的开源系统. 该系统基于常用的 PyTorch 框架, 目前支持在 GPU 上高效训练. 系统为不同用户提供多个层次的友好接口, 便于探索 MoE 架构的不同方面. *FastMoE* 在单块 GPU 上的性能经过充分优化, 可以发挥 GPU 的能力. *FastMoE* 还可以在多个节点之间跨 GPU 运行, 具备合理的可扩展性, 从而支持进一步扩大模型规模. 在使用 *FastMoE* 的端到端模型训练实验中, 我们观察到了真实的模型性能优势.

我们仍在继续开发 *FastMoE*, 加入更多功能并提高训练速度. 与 GShard 模型 [Lep20] 相比, *FastMoE* 缺少支持专家负载均衡的功能. 负载均衡监控器与负载均衡 loss 支持仍在开发中. 我们也在努力改进模型加载和保存等工具, 使系统更易使用. 多 GPU 性能需要高性能计算与机器学习双方共同努力. 我们欢迎对这个开源项目的任何贡献. 期待你的参与.
