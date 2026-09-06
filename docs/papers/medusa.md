---
title: 'Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads'
createTime: 2026/09/06 22:18:29
permalink: /papers/medusa/
pageClass: paper-reading medusa-paper
---

> [Tianle Cai](https://ctlllll.github.io/) [+author-equal] [+author-corresponding], [Yuhong Li](https://leeyeehoo.github.io/) [+author-equal] [+author-corresponding], [Zhengyang Geng](https://gsunshine.github.io/), [Hongwu Peng](https://harveyp123.github.io/), [Jason D. Lee](https://jasondlee88.github.io/), [Deming Chen](https://ece.illinois.edu/about/directory/faculty/dchen), [Tri Dao](https://tridao.me/). 论文于 2024 年 1 月 19 日首次提交至 arXiv; 当前版本为 v3, 修订于 2024 年 6 月 14 日. 论文发表于 *第 41 届国际机器学习大会论文集*, PMLR 235:5209-5235, 2024 年 7 月. [Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774). <a href="/paper/medusa.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [PMLR](https://proceedings.mlr.press/v235/cai24b.html). [DOI](https://doi.org/10.48550/arXiv.2401.10774). [TeX 源文件](https://export.arxiv.org/e-print/2401.10774v3). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

大型语言模型 (LLM) 采用自回归解码, 必须按顺序计算, 每一步都依赖上一步的输出. 这造成了瓶颈: 每一步都要把完整的模型参数从高带宽内存 (HBM) 搬到加速器缓存中. 推测解码等方法虽被用于解决这一问题, 但另行获取并维护草稿模型并不容易, 因而妨碍了它们的应用. 本文提出 Medusa: 在 LLM 上增加额外的解码头, 并行预测后续多个词元, 从而高效提升推理速度. Medusa 使用*树形注意力机制*构造多条候选续写, 并在每个解码步骤中同时验证它们. 借助并行处理, Medusa 大幅减少了所需的解码步数. 为适应不同使用场景, 我们给出两级微调流程: **Medusa-1** 直接在*冻结*的骨干 LLM 上微调 Medusa, 可实现无损推理加速; **Medusa-2** 将 Medusa 与骨干 LLM 一同微调, 能提高 Medusa 头的预测准确率并取得更高加速比, 但需要一套能够保留模型能力的专门训练方案. 此外, 我们还提出若干扩展: 在没有训练数据时使用*自蒸馏*, 并用*典型接受方案*提高接受率且维持生成质量. 我们在不同规模、不同训练流程的模型上评估了 Medusa. 实验表明, Medusa-1 在不损害生成质量的情况下可取得超过 $2.2\times$ 的加速, Medusa-2 则把加速比进一步提高到 2.3-$2.8\times$.

<span id="section-1"></span>

## 1 引言

大型语言模型 (LLM) 的最新进展表明, 随着模型规模扩大到数十亿参数, 语言生成质量会显著提升 [Bro20, Cho22b, Zha22, Hof22, Ope23, Pal23, Tou23a]. 但模型增长也带来了更高的*推理延迟*, 给实际应用造成了显著困难. 从系统角度看, LLM 推理主要受内存带宽限制 [Sha19, Kim23]; 延迟瓶颈来自加速器的内存带宽, 而不是算术计算. 这一瓶颈源于自回归解码的顺序性质: 每次前向传播都要把完整模型参数从高带宽内存 (HBM) 传到加速器缓存. 整个过程只生成一个词元, 没有充分利用现代加速器的算术计算能力, 因而效率低下.

为解决这一问题, 一种加快 LLM 推理的思路是*提高解码过程的算术强度* (总浮点运算量 (FLOP) 与总数据移动量之比), 同时*减少解码步数*. 推测解码正是沿着这条思路提出的 [Lev23, Che23, Xia23c, Mia23b]. 它先用较小的草稿模型生成一段词元序列, 再由原始的大模型筛选可接受的续写. 然而, 合适的草稿模型很难获得, 把它集成到分布式系统中更为困难 [Che23].

本文不再用独立草稿模型顺序生成候选输出, 而是重新审视并改进在骨干模型之上设置多个解码头以加快推理的思路 [Ste18]. 我们发现, 只要使用得当, 这项技术便能绕开推测解码的难题, 并顺畅地接入现有 LLM 系统. 具体而言, 我们提出 Medusa, 通过增加解码头来并发预测多个词元, 从而增强 LLM 推理. 这些头以*参数高效*的方式微调, 可以添加到任意现有模型上. Medusa 不需要草稿模型, 因此很容易集成进现有 LLM 系统, 包括分布式环境.

我们又用两点关键见解改进了 Medusa. 第一, 每个解码步骤只生成一条候选续写, 会造成计算资源利用不足. 为此, 我们让 Medusa 头生成多条候选续写, 只需简单调整注意力掩码便可并行验证. 第二, 我们可以沿用推测解码中的拒绝采样方案 [Lev23, Che23], 生成与原模型同分布的响应, 但这无法进一步提高加速率. 因此, 我们另行提出*典型接受*方案, 从 Medusa 头的输出中选择*合理*候选. 我们以温度作为阈值, 控制结果偏离原模型预测的程度, 提供一种比拒绝采样更高效的选择. 结果表明, 典型接受方案能够在生成质量相近的前提下进一步加快解码.

为了给 LLM 配备具有预测能力的 Medusa 头, 我们针对不同情形提出两套微调流程. 当计算资源有限, 或者希望把 Medusa 加入现有模型而不影响其性能时, 我们推荐 Medusa-1. 它占用的内存很少, 还可采用类似 QLoRA [Det24] 的量化技术进一步优化; 骨干模型保持不变, 因而生成质量不会受损. 不过, Medusa-1 没有充分发挥骨干模型的潜力. 进一步微调骨干模型可提高 Medusa 头的预测准确率, 直接带来更高加速比. 因此, 我们提出 Medusa-2, 适合计算资源充足, 或从基础模型直接进行监督微调 (SFT) 的场景. Medusa-2 的关键是一套训练协议: 联合训练 Medusa 头和骨干模型, 同时不损害模型的下一词元预测能力与输出质量. 我们根据模型训练方案和数据集可用性, 给出不同的数据获取策略. 如果模型在公开数据集上微调, 可直接把该数据集用于 Medusa. 如果数据集不可用, 或模型经历过基于人类反馈的强化学习 (RLHF) [Ouy22a], 我们建议通过自蒸馏为 Medusa 头生成训练数据集.

实验主要考察批大小为 1 的情形, 这代表了个人在本地托管 LLM 的典型用法. 我们测试了不同规模和训练设置的模型, 包括 Vicuna-7B、13B (使用公开数据集训练)、Vicuna-33B [Chi23a] (使用私有数据集训练 [+1]) 和 Zephyr-7B (同时经过监督微调与对齐). 在不损害生成质量的前提下, Medusa 面对不同提示类型都能取得 2.3 到 2.8 倍加速.

<span id="figure-01"></span>

![Figure 1. Medusa introduces multiple decoding heads, tree-based attention, candidate verification, and acceptance in one inference pipeline.](./medusa/figure-01.png)

**图 1.** Medusa 在 LLM 最后的隐藏状态之上增加*多个头*, 从而并行预测后续若干词元 ([第 2.1.1 节](#section-2-1-1)). 推理时, 每个头为各自对应的位置给出多个概率最高的预测. 这些预测被组合为候选, 再通过*树形注意力*机制并行处理 ([第 2.1.2 节](#section-2-1-2)). 最后一步验证候选并接受一段续写. 除标准拒绝采样外, 这里也可以用*典型接受*方案 ([第 2.3.1 节](#section-2-3-1)) 选择合理续写; *被接受前缀最长的候选*将进入下一轮解码.

<span id="section-2"></span>

## 2 方法

Medusa 沿用推测解码的整体框架, 每个解码步骤主要包含三个子步骤: (1) 生成候选, (2) 处理候选, (3) 接受候选. 在 Medusa 中, (1) 由 Medusa 头完成, (2) 由树形注意力完成; 由于 Medusa 头位于原模型之上, (2) 计算出的 logits 还可用于下一解码步骤的子步骤 (1). 最后的步骤 (3) 可用拒绝采样 [Lev23, Che23], 也可用典型接受 ([第 2.3.1 节](#section-2-3-1)). 整体流程见[图 1](#figure-01).

本节先介绍 Medusa 的关键组件, 包括 Medusa 头和树形注意力; 随后给出两级微调流程, 以适应不同使用场景; 最后介绍自蒸馏和典型接受两项扩展, 分别处理 Medusa 没有训练数据的情形, 并提高解码效率.

<span id="section-2-1"></span>

### 2.1 关键组件

<span id="section-2-1-1"></span>

#### 2.1.1 Medusa 头

在推测解码中, 后续词元由辅助草稿模型预测. 这个草稿模型既要足够小, 又要能有效生成原模型愿意接受的续写. 要同时满足这些要求并不容易, 现有方法 [Spe23, Mia23b] 往往需要单独*预训练*一个较小模型, 耗费大量额外计算资源. 例如, [Mia23b] 报告使用了 275 个 NVIDIA A100 GPU 小时. 单独预训练还可能导致草稿模型与原模型之间出现分布偏移, 生成原模型不偏好的续写. [Che23] 也指出了在分布式环境中服务多个模型的复杂性.

为了简化并普及 LLM 推理加速, 我们借鉴 [Ste18] 在机器翻译、图像超分辨率等任务中采用的并行解码. Medusa 头是附加在原模型最后隐藏状态上的解码头. 具体地, 给定原模型在位置 $t$ 的最后隐藏状态 $h_{t}$, 我们在 $h_{t}$ 上增加 $K$ 个解码头. 第 $k$ 个头预测后续词元中第 $(t+k+1)$ 个位置的词元 (原语言模型头预测第 $(t+1)$ 个位置). 第 $k$ 个头的预测记为 $p_{t}^{(k)}$, 它表示词表上的分布; 原模型的预测记为 $p_{t}^{(0)}$. 依照 [Ste18], 每个头都采用带残差连接的单层前馈网络. 我们发现, 这个简单设计已经足以取得令人满意的性能. 第 $k$ 个头定义如下:

$$
\begin{aligned}
p_{t}^{(k)}=\operatorname{softmax}\left(W_{2}^{(k)}\cdot\left(\operatorname{SiLU}(W_{1}^{(k)}\cdot h_{t})+h_{t}\right)\right), \\
\mathrm{where}\;W_{2}^{(k)}\in\mathbb{R}^{d\times V},W_{1}^{(k)}\in\mathbb{R}^{d\times d}.
\end{aligned}
$$

$d$ 是 LLM 最后隐藏层的输出维度, $V$ 是词表大小. 我们用与原语言模型头相同的参数初始化 $W_{2}^{(k)}$, 并把 $W_{1}^{(k)}$ 初始化为零. 这样, Medusa 头的初始预测便与原模型一致. 我们沿用 Llama 模型 [Tou23a] 所用的 SiLU 激活函数 [Elf17].

与草稿模型不同, Medusa 头和原骨干模型配套训练; 训练时骨干模型可以保持*冻结* (Medusa-1), 也可以一同训练 (Medusa-2). 借助强大的基础模型已经学到的表示, 这种方法即使在单张 GPU 上也能微调大模型. 它还使 Medusa 头的分布与原模型对齐, 缓解分布偏移问题. 此外, 新增的头与原语言模型头相似, 只有一层, 因此 Medusa 不会增加服务系统设计的复杂度, 对分布式环境也很友好. Medusa 头的训练方案将在[第 2.2 节](#section-2-2)讨论.

<span id="section-2-1-2"></span>

#### 2.1.2 树形注意力

通过 Medusa 头, 我们得到后续 $K+1$ 个词元的概率预测, 并据此构造长度为 $K+1$ 的候选续写. 推测解码研究 [Lev23, Che23] 通常只采样一条候选续写, 但在解码中使用多条候选, 可以提高单个解码步骤内的期望接受长度. 当然, 候选越多, 计算需求也越高. 为取得平衡, 我们采用树形注意力机制并发处理多条候选.

<span id="figure-02"></span>

![Figure 2. We demonstrates the use of tree attention to process multiple candidates concurrently. As exemplified, the top-2 predictions from the first Medusa head and the top-3 from the second result in a total of $2\times 3=6$ candidates. Each of these candidates corresponds to a distinct branch within the tree structure. To guarantee that each token only accesses its predecessors, we devise an attention mask that exclusively permits attention flow from the current token back to its antecedent tokens. The positional indices for positional encoding are adjusted in line with this structure.](./medusa/figure-02.png)

**图 2.** 图中展示了如何用树形注意力并发处理多条候选. 例子中, 第一个 Medusa 头取前 2 个预测, 第二个头取前 3 个预测, 共得到 $2\times 3=6$ 条候选. 每条候选对应树结构中的一条分支. 为保证每个词元只能访问自己的前驱, 我们设计了注意力掩码, 只允许注意力从当前词元流向其前驱词元. 位置编码所用的位置索引也随该结构调整.

这种注意力机制不同于传统的因果注意力: 只有同一条续写中的词元才被视为历史信息. 受图神经网络领域将图结构嵌入注意力这一思路 [Yin21] 启发, 我们把树结构纳入注意力掩码, 如[图 2](#figure-02) 所示. [Mia23b, Spe23] 等独立工作也探索了相近想法, 但它们采用自底向上的方式, 合并草稿模型生成的多条候选来构造树. 借助 Medusa 头生成的候选结构, 我们则自顶向下建树. 对第 $k$ 个头, 取其前 $s_{k}$ 个预测作为候选的基础, 其中 $s_{k}$ 是指定的超参数. 候选由各头前 $s_{k}$ 个预测的笛卡尔积确定. 例如在[图 2](#figure-02) 中, $s_{1}=2$ 且 $s_{2}=3$, 第一个头的每个预测后都可以接第二个头的任意预测. 由此得到一棵第 $k$ 层有 $s_{k}$ 条分支的树 (把虚拟根节点视为第 $0$ 层; 实际上, 第 $0$ 层对应原模型语言模型头的预测, 可独立采样). 在这棵树中, 一个词元只把自身的前驱视作历史上下文, 注意力掩码保证注意力只作用于这些前驱. 使用该掩码并正确设置位置编码的位置索引, 无需扩大批大小即可同时处理大量候选. 新词元总数为 $\sum_{k=1}^{K}\prod_{i=1}^{k}s_{i}$.

本节用笛卡尔积展示了最简单、最规则的建树方式. 但也可以采用更精细的树结构, 利用不同头中不同高排名预测准确率不均衡的特点. [第 2.3.3 节](#section-2-3-3)将继续讨论.

<span id="section-2-2"></span>

### 2.2 训练策略

最基本的做法是冻结骨干模型, 只微调 Medusa 头. 不过, 联合训练骨干模型与 Medusa 头可以显著提高后者的准确率. 根据计算资源和具体使用需求, 我们为 Medusa 头提出两级训练策略.

本节假设存在与目标模型输出分布一致的训练数据集, 例如目标模型监督微调 (SFT) 时使用的数据集. [第 2.3.2 节](#section-2-3-2)将讨论如何用自蒸馏摆脱对这类数据集的依赖.

<span id="section-2-2-1"></span>

#### 2.2.1 Medusa-1: 冻结骨干模型

在骨干模型冻结时训练 Medusa 头, 可以采用 Medusa 头预测与真实标签之间的交叉熵损失. 具体而言, 给定位置 $t+k+1$ 上的真实词元 $y_{t+k+1}$, 第 $k$ 个头的损失为 $\mathcal{L}_{k}=-\log p_{t}^{(k)}(y_{t+k+1})$, 其中 $p_{t}^{(k)}(y)$ 表示第 $k$ 个头预测词元 $y$ 的概率. 我们还观察到, $k$ 越大, $\mathcal{L}_{k}$ 也越大; 这是合理的, 因为第 $k$ 个头在 $k$ 较大时更难确定预测. 因此, 可以为 $\mathcal{L}_{k}$ 加上权重 $\lambda_{k}$, 平衡不同头的损失. Medusa 的总损失为:

<span id="equation-01"></span>

$$
\mathcal{L}_{\mathrm{Medusa}-1}=\sum_{k=1}^{K}-\lambda_{k}\log p_{t}^{(k)}(y_{t+k+1}).
$$

实际使用时, 我们把 $\lambda_{k}$ 设为 $0.8$ 之类常数的第 $k$ 次幂. 骨干模型只负责提供隐藏状态, 因而可以使用它的量化版本来降低内存占用. 这让 LLM 推理加速更易普及: 量化后, Medusa 与 QLoRA [Det24] 类似, 可以在单张消费级 GPU 上为大模型训练. 训练只需数小时 (例如在单张 NVIDIA A100 PCIe GPU 上, 用 60k 条 ShareGPT 样本为 Vicuna 7B 训练 Medusa-1 需要 5 小时).

<span id="section-2-2-2"></span>

#### 2.2.2 Medusa-2: 联合训练

为了进一步提高 Medusa 头的准确率, 可以把它们与骨干模型一同训练. 但这需要一套专门的训练方案, 以保留骨干模型的下一词元预测能力和输出质量. 为此, 我们提出三项策略:

- **组合损失**: 为保留骨干模型的下一词元预测能力, 需要把骨干模型的交叉熵损失 $\mathcal{L}_{\mathrm{LM}}=-\log p_{t}^{(0)}(y_{t+1})$ 加到 Medusa 损失上. 我们还加入权重 $\lambda_{0}$, 平衡骨干模型与 Medusa 头的损失. 因此, 总损失为:

  <span id="equation-02"></span>

  $$
  \mathcal{L}_{\mathrm{Medusa}-2}=\mathcal{L}_{\mathrm{LM}}+\lambda_{0}\mathcal{L}_{\mathrm{Medusa}-1}.
  $$
- **差分学习率**: 骨干模型已经训练充分, 而 Medusa 头需要更多训练, 因此可为两者设置不同的学习率, 让 Medusa 头更快收敛, 同时保留骨干模型的能力.
- **头部预热**: 训练刚开始时, Medusa 头的损失很大, 会产生较大梯度, 可能扭曲骨干模型的参数. 依照 [Kum22] 的思路, 我们采用两阶段训练. 第一阶段只像 Medusa-1 一样训练 Medusa 头. 第二阶段使用预热策略, 联合训练骨干模型与 Medusa 头. 具体做法是先训练骨干模型若干轮, 再将 Medusa 头与骨干模型一同训练. 除了这个简单方案, 也可逐渐提高骨干模型损失的权重 $\lambda_{0}$, 采用更精细的预热策略. 实践中, 两种策略都表现良好.

结合这些策略, 我们可以在不损害骨干模型能力的前提下, 联合训练 Medusa 头与骨干模型. 这套方案还能和监督微调 (SFT) 一起使用, 得到原生支持 Medusa 的模型.

<span id="section-2-2-3"></span>

#### 2.2.3 如何选择头的数量

经验上, 五个头已经足够. 因此, 我们建议训练五个头, 再按[第 2.3.3 节](#section-2-3-3)的策略确定树形注意力的最佳配置. 树形注意力优化后, 推理有时只需三个或四个头; 此时可以直接忽略多余的头, 不会产生开销.

<span id="section-2-3"></span>

### 2.3 扩展

<span id="section-2-3-1"></span>

#### 2.3.1 典型接受

推测解码论文 [Lev23, Che23] 使用拒绝采样, 生成与原模型分布一致的多样化输出. 但后续实现 [Gan23, Spe23] 表明, 采样温度升高时, 这种策略的效率会降低. 可以从草稿模型与原模型相同这一极端情形直观理解: 使用贪心解码时, 草稿模型的所有输出都会被接受, 效率最高; 拒绝采样则会产生额外开销, 因为草稿模型和原模型各自独立采样. 即使两者分布完全一致, 草稿模型的输出仍可能被拒绝.

然而在现实场景中, 从语言模型采样常用于生成多样化响应, 温度参数只是调节响应的“创造性”. 因此, 温度越高, 原模型接受草稿模型输出的机会理应越多. 我们认为, 通常没有必要与原模型的分布完全匹配. 为此, 我们不用拒绝采样, 而是提出*典型接受*方案来选择合理候选. 该方案受截断采样研究 [Hew22] 启发 (详见[第 6 节](#section-6)). 我们要选择的是*典型*候选, 即由原模型生成的概率不至于过低的候选. 我们自然地以*原模型*给出的预测概率作为度量, 并根据预测分布设置接受阈值. 具体而言, 给定上下文 $x_{1},x_{2},\cdots,x_{n}$, 在评估候选序列 $(x_{n+1},x_{n+2},\cdots,x_{n+K+1})$ (由原语言模型头和 Medusa 头的高排名预测组成) 时, 考察以下条件:

$$
\begin{aligned}
p_{\text{original}}(x_{n+k}|x_{1},x_{2},\cdots,x_{n+k-1})> \\
\min\left(\epsilon,\delta\exp\left(-H(p_{\text{original}}(\cdot|x_{1},x_{2},\cdots,x_{n+k-1}))\right)\right),
\end{aligned}
$$

其中 $H(\cdot)$ 表示熵函数, $\epsilon,\delta$ 分别是硬阈值和依赖熵的阈值. 这一准则改编自 [Hew22], 基于两点观察: (1) 概率相对较高的词元有意义; (2) 分布熵较高时, 多种续写都可能合理. 解码时, 按该准则评估每条候选; 如果候选的一个*前缀*满足条件, 就接受该前缀. 为保证每一步至少生成一个词元, 第一个词元采用*贪心解码*并*无条件*接受, 后续词元再使用典型接受. 当前步骤的最终预测取所有候选中*最长的已接受前缀*.

考察这个方案可以得到几点认识. 首先, 温度设为 $0$ 时, 只有概率最大的词元具有非零概率, 因而方案退化为贪心解码. 温度高于 $0$ 后, 只要 $\epsilon,\delta$ 设置恰当, 贪心解码的结果因为拥有最大概率, 仍会始终被接受, 从而取得最大加速比. 一般情形同样如此: 温度升高会带来更长的接受序列, 实验结果也印证了这一点.

实验表明, 典型接受能够在保持相近生成质量的同时取得更高加速比, 如[图 5](#figure-05) 所示.

<span id="section-2-3-2"></span>

#### 2.3.2 自蒸馏

在[第 2.2 节](#section-2-2)中, 我们假设存在与目标模型输出分布匹配的训练数据集, 但现实中并非总能满足. 例如, 模型所有者可能只发布模型而不发布训练数据; 模型也可能经历过基于人类反馈的强化学习 (RLHF), 使其输出分布不同于训练数据集. 为解决这一问题, 我们提出自动化自蒸馏流程, 让模型自身生成与其输出分布匹配的 Medusa 头训练数据.

数据集生成过程很直接. 首先从与目标模型领域相近的公开种子数据集出发, 例如对聊天模型使用 ShareGPT [Src23]. 然后只取数据集中的提示, 让模型作答. 为得到多轮对话样本, 可以依次把种子数据集中的提示输入模型. Zephyr 7B [Tun23] 一类模型同时在对话双方角色上训练, 具备自我对话能力; 对它们只需输入第一个提示, 便可让模型生成多轮对话.

对 Medusa-1 而言, 这个数据集足以训练 Medusa 头. 但对 Medusa-2, 我们观察到, 仅用该数据集训练骨干模型与 Medusa 头通常会降低生成质量. 事实上, 即使不训练 Medusa 头, 用该数据集训练骨干模型也会导致性能下降. 这说明, 与经典知识蒸馏工作 [Kim16c] 类似, 骨干模型也应使用原模型的概率预测, 而不是真实词元作为标签. 具体而言, 骨干模型的损失为:

$$
\mathcal{L}_{\mathrm{LM-distill}}=\operatorname{KL}(p_{\mathrm{original},t}^{(0)}\|p_{t}^{(0)}),
$$

其中 $p_{\mathrm{original},t}^{(0)}$ 表示原模型在位置 $t$ 上预测的概率分布.

不过, 直接获取原模型的概率预测, 需要在训练期间同时保存两个模型, 会增加内存需求. 为进一步缓解这一问题, 我们利用自蒸馏设置, 提出一种简单有效的方法: 使用 LoRA [Hu21] 等参数高效适配器微调骨干模型. 此时, 关闭适配器后的模型就是原模型, 蒸馏便不再消耗额外内存. 这套自蒸馏流程可以在不损害骨干模型能力、几乎不增加内存占用的情况下训练 Medusa-2. 最后有一点使用建议: 这种情况下最好使用不带量化的 LoRA, 否则教师模型会是量化模型, 可能降低生成质量.

<span id="section-2-3-3"></span>

#### 2.3.3 搜索优化的树结构

在[第 2.1.2 节](#section-2-1-2)中, 我们以笛卡尔积给出了最简单的树结构. 但当树的总节点数预算固定时, 规则树未必是最佳选择. 直观地说, 由不同头的不同高排名预测组成的候选, 准确率可能不同. 因此, 可以估计准确率, 并据此构造树结构.

具体而言, 可以用校准数据集计算不同头各个高排名预测的准确率. 以 $a_{k}^{(i)}$ 表示第 $k$ 个头的第 $i$ 高预测的准确率 [+2]. 假设各准确率相互独立, 则由不同头的第 $\left[i_{1},i_{2},\cdots,i_{k}\right]$ 高预测组成的候选序列, 其准确率可估为 $\prod_{j=1}^{k}a_{j}^{(i_{j})}$. 设 $I$ 为 $\left[i_{1},i_{2},\cdots,i_{k}\right]$ 的所有可能组合, $I$ 中每个元素都能映射到树的一个节点 (不仅包含叶节点, 而是包含所有节点). 候选序列接受长度的期望为:

$$
\sum_{\left[i_{1},i_{2},\cdots,i_{k}\right]\in I}\prod_{j=1}^{k}a_{j}^{(i_{j})}.
$$

设想逐个添加节点来构造树, 新节点对期望的贡献恰好就是与该节点关联的准确率. 因此, 可以贪心地添加与当前树相连且准确率最高的节点, 不断重复, 直到节点总数达到目标. 这样便能构造一棵使接受长度期望最大化的树. 更多细节见[第 8 节](#section-8).

<span id="figure-03"></span>

![Figure 3. Left: Speed comparison of baseline, Medusa-1 and Medusa-2 on Vicuna-7B/13B. Medusa-1 achieves more than $2\times$ wall-time speedup compared to the baseline implementation while Medusa-2 further improves the speedup by a significant margin. Right: Detailed speedup performance of Vicuna-7B with Medusa-2 on 8 categories from MT-Bench.](./medusa/figure-03.png)

**图 3.** 左: 在 Vicuna-7B/13B 上比较基线、Medusa-1 与 Medusa-2 的速度. Medusa-1 相比基线实现取得超过 $2\times$ 的实际时间加速, Medusa-2 又显著提高了加速比. 右: Medusa-2 Vicuna-7B 在 MT-Bench 8 个类别上的详细加速表现.

<span id="section-3"></span>

## 3 实验

本节通过实验展示 Medusa 在不同设置下的效果. 首先在 Vicuna-7B 和 13B 模型 [Chi23a] 上评估 Medusa-1 与 Medusa-2. 随后使用 Vicuna-33B 和 Zephyr-7B 评估本方法: 前者无法直接取得微调方案, 后者使用基于人类反馈的强化学习 (RLHF), 由此检验自蒸馏在这些场景中的可行性. 评估采用多轮对话形式的基准 MT-Bench [Sto23e]. 详细设置见[第 7 节](#section-7).

<span id="section-3-1"></span>

### 3.1 案例研究: Vicuna 7B 和 13B 上的 Medusa-1 与 Medusa-2

**实验设置.** 我们使用 Vicuna 模型系列 [Chi23a]. 该系列包含从 Llama 模型 [Tou23a] 微调得到的不同规模聊天模型 (7B、13B、33B). 其中, 7B 和 13B 在 ShareGPT [Src23] 数据集上训练, 33B 则是使用私有数据集训练的实验模型. 本节用 ShareGPT 数据集为 7B 和 13B 模型训练 Medusa 头, 共 $2$ 个 epoch. 我们采用 Vicuna v1.5, 它由上下文长度 4096 的 Llama-2 模型微调而来.

**结果.** 结果汇总于[图 3](#figure-03), 基线为 Huggingface 默认实现. 从[图 3(a)](#figure-03) 可见, 对 7B 模型, Medusa-1 和 Medusa-2 都显著提高了每秒处理词元数: Medusa-1 加速 $2.18\times$, Medusa-2 进一步达到 $2.83\times$. 在更大的 13B 模型上, Medusa-1 加速 $2.33\times$, Medusa-2 仍比基线快 $2.83\times$. 我们还绘制了 Medusa-2 Vicuna-7B 各类别的加速比. 编码类别获得 $3.29\times$ 加速, 说明 Medusa 对这类任务尤其有效, 也显示了优化广泛用于软件开发和其他编程任务的编码 LLM 的巨大潜力. “信息抽取”类别的加速比最高, 达到 $3.62\times$, 表明 Medusa 对这项任务的优化程度很高. 总体而言, Medusa 在不同模型规模和任务上都显著提升了推理速度.

<span id="section-3-2"></span>

### 3.2 案例研究: 在 Vicuna-33B 和 Zephyr-7B 上使用自蒸馏训练

**实验设置.** 本案例研究聚焦需要自蒸馏的情形, 以 Vicuna-33B [Chi23a] 和 Zephyr-7B [Tun23] 为例. 按照[第 2.3.2 节](#section-2-3-2)的流程, 我们先用一些种子提示生成数据集. ShareGPT [Src23] 和 UltraChat [Din23b] 用作种子数据集, 两种情形各收集约 $100k$ 个样本. 有趣的是, Zephyr 只需一个提示就能继续生成多轮对话, 因而很容易收集大型数据集. 对 Vicuna-33B, 我们依次输入每段多轮种子对话中的提示, 以温度 0.3 随机采样, 生成多轮对话. 两个模型都采用序列长度 $2048$、批大小 $128$ 训练.

<span id="table-01"></span>

![Table 1. Comparison of various Medusa-2 models. The first section reports the details of Medusa-2, including accelerate rate, overhead, and quality that denoted the average scores on the MT-Bench compared to the original models. The second section lists the speedup ($S$) of SpecDecoding and Medusa, respectively.](./medusa/table-01.png)

**表 1.** 不同 Medusa-2 模型的比较. 第一部分给出 Medusa-2 的详细信息, 包括加速率、开销, 以及相对原模型的质量; 质量以 MT-Bench 平均分表示. 第二部分分别列出 SpecDecoding 和 Medusa 的加速比 ($S$).

<span id="figure-04"></span>

![Figure 4. Effectiveness of numbers of candidate tokens for decoding introduced by trees (default number of candidate token for decoding is 1 when using KV cache). Left: The acceleration rate for randomly sampled dense tree settings (blue dots) and optimized sparse tree settings (red stars). Right: The speed (tokens/s) for both settings. The trend lines indicate that while the acceleration rate remains relatively stable for sparse trees, there is a notable decrease in speed as the candidate tokens increases.](./medusa/figure-04.png)

**图 4.** 树引入的解码候选词元数量有何影响 (使用 KV 缓存时, 默认候选词元数为 1). 左: 随机采样的稠密树设置 (蓝点) 与优化的稀疏树设置 (红星) 的加速率. 右: 两种设置的速度 (词元/秒). 趋势线表明, 稀疏树的加速率较为稳定, 但随着候选词元增加, 速度明显下降.

**结果.** [表 1](#table-01) 从加速率、开销和 MT-Bench 质量三方面比较了多种 Medusa-2 模型; GPT-4 担任评测者, 在 0 到 10 分之间评分. 我们报告 Medusa 相对原模型的质量差异. 值得注意的是, Medusa-2 Vicuna-33B 的加速率较低, 但质量相当. 我们推测, 原因是未公开的训练数据集与自蒸馏所用数据集不匹配. 因此, 自蒸馏可以很好地对齐模型的生成质量, Medusa 头学到的分布却来自自蒸馏数据, 可能偏离训练集. 我们还为 Vicuna 系列配合开源草稿模型应用了推测解码 [Che23, Lev23] (详见[第 9 节](#section-9)).

这些结果凸显了模型规模扩大并使用自蒸馏时, 速度与性能之间复杂的相互作用. 结果也表明, Medusa-2 有望在谨慎保持模型输出质量的同时提高处理效率, 为 LLM 与 Medusa 头的协同优化提供了一个有前景的方向.

<span id="section-3-3"></span>

### 3.3 消融研究

<span id="section-3-3-1"></span>

#### 3.3.1 树形注意力配置

我们在 MT-Bench 的写作和角色扮演类别上, 使用 Medusa-2 Vicuna-7B 研究树形注意力, 旨在说明其动机与性能.

[图 4(a)](#figure-04) 比较了随机采样的稠密树配置 ([第 2.1.2 节](#section-2-1-2), 蓝点) 与优化的稀疏树设置 ([第 2.3.3 节](#section-2-3-3), 红星) 的加速率. 64 个节点的稀疏树, 加速率优于 256 个节点的稠密树. [图 4(b)](#figure-04) 中速度下降, 是因为计算受限所引入的开销增加. 更复杂的树可以提高加速率, 但线性层和自注意力中的密集矩阵乘法也会拖慢速度. 如[图 4(a)](#figure-04) 所示, 加速率的增长呈对数趋势, 随树规模扩大而放缓; 不过初始收益很大, 足以让 Medusa 获得显著加速. 如果加速率的增幅小于开销, 整体性能反而会下降. 详细研究见[第 12 节](#section-12).

<span id="figure-05"></span>

![Figure 5. Performance comparison of Medusa using proposed typical sampling. The model is fully fine-tuned from Vicuna-7B. The plot illustrates the acceleration rate and average scores on the writing and roleplay (MT-Bench) with a fixed temperature of 0.7 for 3 different settings: greedy sampling and random sampling (RS) plotted as the star and the dot, and typical sampling curves under different thresholds.](./medusa/figure-05.png)

**图 5.** 使用所提典型采样的 Medusa 性能比较. 模型由 Vicuna-7B 全量微调. 在温度固定为 0.7 时, 图中展示三种设置在写作与角色扮演 (MT-Bench) 上的加速率和平均分: 星号表示贪心采样, 圆点表示随机采样 (RS), 曲线表示不同阈值下的典型采样.

<span id="section-3-3-2"></span>

#### 3.3.2 典型接受的阈值

我们使用 Medusa-2 Vicuna 7B, 在 MT-Bench [Sto23e] 的写作和角色扮演类别上研究典型接受阈值. 方法与 [Hew22] 一致, 设 $\alpha=\sqrt{\epsilon}$. [图 5](#figure-05) 比较了模型在不同采样设置下的表现: 阈值 $\epsilon$ 从 0.01 开始, 每次增加 0.01, 直至 0.25. 结果显示出明确的权衡: $\epsilon$ 增大时, 质量提高, 加速率则下降. 对需要创造力的任务, 默认随机采样优于贪心采样; 随 $\epsilon$ 增大, 所提典型采样可以达到与随机采样相当的表现.

<span id="table-02"></span>

![Table 2. Comparison of Different Settings of Vicuna-7B. Quality is obtained by evaluating models on MT-Bench using GPT-4 as the judge (higher the better).](./medusa/table-02.png)

**表 2.** Vicuna-7B 不同设置的比较. 质量由 GPT-4 担任裁判, 在 MT-Bench 上评估模型得到 (越高越好).

<span id="section-3-3-3"></span>

#### 3.3.3 两阶段微调的效果

[表 2](#table-02) 展示了 Vicuna-7B 不同微调策略的性能差异. Medusa-1 只微调 Medusa 头, 在不损害生成质量的情况下加速 2.18x. Medusa-2 采用两阶段微调 ([第 2.2.2 节](#section-2-2-2)), 保持生成质量的同时取得比 Medusa-1 更高的 2.83x 加速. 相比之下, 直接把模型与 Medusa 头一同微调会降低生成质量. 结果说明, 使用 Medusa-2 微调既能维持模型质量, 又能比 Medusa-1 获得更高加速比.

<span id="table-03"></span>

![Table 3. Impact of Techniques on Speedup](./medusa/table-03.png)

**表 3.** 各项技术对加速比的影响

<span id="section-4"></span>

## 4 讨论

总之, Medusa 为模型增加预测解码头, 同时生成多个词元, 绕开顺序解码的限制, 从而将 LLM 推理加快 2.3-2.8 倍. Medusa 的主要优点是简单、参数高效, 并且易于集成到现有系统中; 它不需要专门的草稿模型. 典型接受方案省去了拒绝采样的复杂环节, 同时给出合理输出. 我们的方法包含两套高效训练流程, 能在不同模型和提示类型上维持高质量输出. [表 3](#table-03) 总结了各项技术的发展过程及其对加速比的影响.

为简化问题, 本文主要考察批大小为 1 的设置. 不过需要强调, 本文思路可以推广到更大的批大小; 在本文发表后, TensorRT、Huggingface TGI 等库已经支持这种设置.

## 致谢

我们衷心感谢以下几位对本项目贡献良多的人士:

- Zhuohan Li 为 LLM 服务提供了宝贵见解. 如果你还没看过他的 vLLM 项目, 不妨去看看, 它确实令人赞叹.
- Shaojie Bai 参与了关键讨论, 帮助塑造了这项工作的早期阶段.
- Denny Zhou 向 Tianle 介绍了截断采样方案, 并鼓励他探索 LLM 服务领域.
- Yanping Huang 向 Tianle 指出了 LLM 服务受内存带宽限制的问题.
- Lianmin Zheng 澄清了不同规模 Vicuna 模型采用的不同训练方案.

Jason D. Lee 感谢 NSF CCF 2002272、NSF IIS 2107304 和 NSF CAREER Award 2144994 的支持. Deming Chen 感谢 UIUC AMD Center of Excellence 的支持.

<span id="section-5"></span>

## 5 影响声明

Medusa 是一种提高大型语言模型 (LLM) 推理速度的新方法, 会给社会、技术和伦理带来广泛影响. 本节详细讨论这些影响.

<span id="section-5-1"></span>

### 5.1 社会与技术影响

- **AI 的可及性与普及**: Medusa 显著提高 LLM 效率, 让更多用户和组织能够使用先进 AI 技术. 这种普及可能推动教育、医疗、娱乐等多个领域的创新, 并带来惠及整个社会的突破.
- **环境影响**: Medusa 对 LLM 推理的加速可能减少能源消耗和碳足迹. 这符合发展可持续 AI 实践的需要, 有助于环境保护.
- **经济影响**: Medusa 提高效率后, 部署先进 AI 模型的成本门槛可能下降, 中小企业也能利用先进 AI 能力. 这可能刺激经济增长、促进竞争并推动技术创新.

<span id="section-5-2"></span>

### 5.2 伦理考量

- **偏见与公平**: Medusa 虽以提高 LLM 效率为目标, 但也会继承骨干模型的伦理问题, 包括偏见与公平性. 它能够保持生成质量, 因此还需研究模型是否会延续或放大已有偏见.
- **透明度与问责**: Medusa 的树形注意力机制和多个解码头使其具有一定复杂度, 可能给模型可解释性带来挑战. 要建立对 AI 系统的信任, 必须确保决策过程透明, 并维持对这些决策的问责.
- **安全与隐私**: Medusa 增强后的 LLM 能力可能被恶意利用, 例如大规模生成虚假信息或自动发动网络攻击. 必须制定并执行伦理准则与安全措施, 防止滥用.

<span id="section-6"></span>

## 6 相关工作

<span id="section-6-1"></span>

### 6.1 LLM 推理加速

大型语言模型 (LLM) 推理效率低, 主要源于自回归解码受内存带宽限制. 已有多种方法尝试缓解这一问题, 改善推理延迟和吞吐量. 传统上, 批推理是提高算术强度、摆脱内存带宽限制的直接手段. 但在 LLM 中, 模型参数和键值 (KV) 缓存都会占用大量加速器内存, 阻碍大批量处理. 现有方法可在概念上分成两类: (1) 减少内存占用, 从而降低内存传输开销并支持更大的批大小; (2) 减少解码步数, 直接降低延迟.

**减少 KV 缓存.** 多查询注意力 [Sha19] 和分组查询注意力 [Ain23] 直接缩减 KV 缓存: 注意力模块使用的键头和值头少于查询头, 从而大幅降低 KV 内存占用, 支持更大批大小并提高加速器利用率 [Pop22]. 此外, [Zha23g] 选择性保留最关键的 KV 词元, 进一步缩减 KV 缓存. 从系统角度看, [Kwo23] 引入分页内存管理方案, 减少 KV 缓存碎片.

**量化.** 量化技术广泛用于降低 LLM 的内存占用. [Xia23] 在激活与参数之间重新缩放, 消除离群值并简化量化过程. [Det22] 把矩阵乘法拆为绝大多数 8 位运算和少量 16 位运算. [Fra22] 迭代地把权重列舍入到 3/4 位, [Lin23d] 则提出感知激活的量化方案, 保护显著权重, 把 LLM 压缩到 3/4 位. [Kim23] 还采用稀疏加低精度模式处理少量关键权重.

**推测解码.** 推测解码 [Lev23, Che23] 与上述方法正交, 旨在并行执行多个解码步骤, 减少所需的总步数. 它用较小的草稿模型猜测后续若干词, 再由 LLM 一并评估并接受合适结果. 这种方法与非自回归生成文献 [Xia23d] 相呼应, 但专为 LLM 的上述低效问题设计. 与以往工作不同, 我们让原模型自身预测, 不引入额外草稿模型, 因而更直接, 也能无缝集成进现有系统, 免去管理两个模型的复杂性. [Mia23b, Spe23] 分别独立提出用树形注意力并行生成多条候选: [Mia23b] 建议用模型集成提出候选, [Spe23] 则为草稿模型增加一层层级. 但草稿模型需要专门预训练, 并与目标模型对齐; 多个草稿模型更显笨重, 还涉及并行管理的复杂性. 我们只依赖解码头, 因而更简单. [Mia23b] 用多个草稿模型生成词元, 再以树形注意力合并; [Spe23] 用小型草稿模型批量处理树的每一层. 我们则直接取各 Medusa 头概率最高的词元, 构造静态稀疏树, 不需要自回归, 也不调整树结构. 这简化了流程并提高效率. 此外, 我们通过详细消融研究展示树节点如何影响解码速度.

<span id="section-6-2"></span>

### 6.2 采样方案

从大型语言模型 (LLM) 中采样文本的方式, 会显著影响生成结果的质量. 最新研究表明, 直接从语言模型采样可能得到不连贯或无意义的结果 [Pil21, Hol20]. 为解决这一问题, 研究者提出了*截断采样*方案 [Fan18, Bas21, Mei22, Hew22, Mei23]: 每个解码步骤都在特定*允许集合*上的截断分布中采样, 以得到高质量且多样的样本.

不同策略以不同方式定义允许集合. 例如, top-$k$ 采样 [Fan18] 保留概率最高的 $k$ 个词, top-$p$ 采样 [Hol20] 则取累计概率达到 $p$ 的最小词集. 典型解码 [Mei23] 用预测分布的熵确定纳入阈值. [Hew22] 提供了一个统一框架, 用于全面理解截断采样技术.

我们的典型接受方案受这些方法启发, 同样通过定义允许集合, 排除采样过程中概率很低的候选. 不同之处是, 我们不要求输出分布与语言模型分布精确一致. 这一偏离使输出更多样, 同时维持高质量, 并在不损害生成文本完整性的前提下提高效率.

<span id="section-7"></span>

## 7 实验设置

<span id="section-7-1"></span>

### 7.1 常用术语

我们说明三个常用术语:

- a) 加速率: 每个解码步骤平均解码的词元数. 标准自回归模型的加速率为 1.0.
- b) 开销: 相对经典解码, 每个解码步骤产生的开销; 计算方式是 Medusa 模型的平均单步延迟除以普通模型的平均单步延迟.
- c) 加速比: 实际时间上的加速倍数.

根据这些定义, 有如下关系: 加速比 = 加速率 / 开销.

<span id="section-7-2"></span>

### 7.2 共同设置

所有实验都使用 Axolotl [Axo23] 框架训练. 我们采用带预热的余弦学习率调度器和 8 位 AdamW [Det21] 优化器. 训练 $5$ 个单层 Medusa 头, 并把[公式 1](#equation-01) 中的 $\lambda_{k}$ 设为 $0.8^{k}$. Medusa-2 使用 LoRA [Hu21] 或 QLoRA [Det24] 微调, Medusa 头的学习率设为骨干模型的 $4$ 倍. LoRA 应用于骨干模型的所有线性层, 包括语言模型头. LoRA 适配器的秩为 $32$, $\alpha$ 为 $16$, 并加入 $0.05$ 的 dropout.

<span id="section-7-3"></span>

### 7.3 Vicuna 7B 和 13B 上的 Medusa-1 与 Medusa-2

全局批大小设为 $64$, 骨干模型的峰值学习率为 $5e^{-4}$, Medusa 头为 $2e^{-3}$, 预热 $40$ 步. 两个模型都使用 4 位量化骨干模型. 我们先用 Medusa-1 训练, 再以所得模型初始化 Medusa-2 的训练. Medusa-2 使用 QLoRA, [公式 2](#equation-02) 中的 $\lambda_{0}$ 设为 $0.2$.

<span id="section-7-4"></span>

### 7.4 在 Vicuna-33B 和 Zephyr-7B 上使用自蒸馏训练

两个模型都直接使用 Medusa-2, 不采用两阶段训练. 我们用正弦调度逐渐增大 $\theta_{0}$, 在训练结束时达到峰值; 实验发现, 这种方法同样有效. 自蒸馏损失相对较小, 因此骨干 LoRA 适配器的峰值学习率设为 $1e^{-4}$, 预热步数设为 $20$. [公式 2](#equation-02) 中的 $\lambda_{0}$ 设为 $0.01$.

<span id="section-8"></span>

## 8 优化树形注意力的可视化

[图 6](#figure-06) 展示了 Medusa-2 Vicuna-7B 的稀疏树结构. 这棵树有四层, 表明计算涉及四个 Medusa 头. 它先通过笛卡尔积形成, 随后根据各 Medusa 头在 Alpaca-eval 数据集 [Dub23] 上测得的 top-k 预测统计期望进行剪枝. 树在视觉上向左倾斜, 表示算法偏好各头中概率更高的节点.

<span id="figure-06"></span>

![Figure 6. Visualization of a sparse tree setting for Medusa-2 Vicuna-7B. The tree has 64 nodes representing candidate tokens and a depth of 4 which indicates 4 Medusa heads involved in calculation. Each node indicates a token from a top-k prediction of a Medusa head, and the edges show the connections between them. The red lines highlight the path that correctly predicts the future tokens.](./medusa/figure-06.png)

**图 6.** Medusa-2 Vicuna-7B 的稀疏树设置. 树有 64 个代表候选词元的节点, 深度为 4, 表示有 4 个 Medusa 头参与计算. 每个节点表示 Medusa 头的一项 top-k 预测, 边表示节点间的连接. 红线标出正确预测后续词元的路径.

<span id="section-9"></span>

## 9 推测解码结果

本研究把推测解码应用于不同规模的 Vicuna 模型 [Chi23a], 即 7B、13B 和 33B. 初步框架采用 Llama-68M、160M [Mia23b] 等开源模型, 以及 Tiny-Llama [Zha24ab] 和 Tiny-Vicuna [Pan23a]; Tiny-Vicuna 由 Tiny-Llama 按 Vicuna 风格的指令微调策略训练而来. 由于推测解码方法 [Che23, Lev23] 并未开源, 评估使用开源替代实现 [+3]. 此外, 我们用 `torch.compile()` 加快草稿模型的推理.

[图 7](#figure-07) 中的结果表明, 草稿模型的最佳设置随 Vicuna 模型规模而变. 对 Vicuna-7B, Llama-68M 在草稿词元数 $\gamma=4$ 时表现最佳; 对 Vicuna-13B, 同一草稿模型在 $\gamma=3$ 时效果最好; 对更大的 Vicuna-33B, Tiny-Vicuna (Vicuna-1B) 在 $\gamma=3$ 时带来最大加速. 这些结果说明, 草稿模型的选择与设置应适配 LLM 的规模, 这也是值得进一步探索的方向.

<span id="figure-07"></span>

![Figure 7. Inference speed of various models using speculative decoding on MT-Bench. Baseline model speeds are presented by grey dotted lines for comparison. $\gamma$ denotes the draft token number.](./medusa/figure-07.png)

**图 7.** 不同模型在 MT-Bench 上使用推测解码时的推理速度. 灰色虚线给出基线模型速度以供比较. $\gamma$ 表示草稿词元数.

<span id="section-10"></span>

## 10 所有模型的补充结果

不同模型上的加速比见[图 8](#figure-08).

<span id="figure-08"></span>

![Figure 8. Speedup of various models with Medusa-2. Medusa-2 shows significant speed improvement over all the models, while models trained with self-distillation (Zephyr-7B, Vicuna-13/33B) have weaker speedup due to the trade-off between preserving quality and boosting speed.](./medusa/figure-08.png)

**图 8.** 各模型使用 Medusa-2 时的加速比. Medusa-2 显著提高了所有模型的速度; 采用自蒸馏训练的模型 (Zephyr-7B、Vicuna-13/33B) 因为需要在保持质量与提高速度之间权衡, 加速比稍弱.

<span id="section-11"></span>

## 11 AlpacaEval 数据集上的补充结果

我们在 AlpacaEval [Li23z] 数据集上进行了更多实验. Medusa-2 取得了与 MT-Bench 结果相近且稳定的加速.

<span id="table-04"></span>

![Table 4. Speedup results on AlpacaEval Li23z dataset.](./medusa/table-04.png)

**表 4.** AlpacaEval [Li23z] 数据集上的加速结果.

<span id="section-12"></span>

## 12 硬件约束与 Medusa 的探索和建模

我们引入简化的 Llama 系列模型, 研究硬件约束, 特别是内存带宽限制, 对 Medusa 式并行解码的影响. 首先, 我们确认线性层和注意力矩阵乘法等矩阵乘法算子是主要开销来源. 随后在 A100-80GB-PCIe、A40 和 A6000 等多种 GPU 上分析 FLOP/s 与运算强度 (FLOP/s 与带宽 (字节/秒) 之比) 的关系. 接着考察使用 Medusa 后不同算子的 FLOP/s 与运算强度如何变化. 最后, 我们用简单的分析模型计算加速率, 并结合硬件基准测试, 由此了解模型规模、序列长度和批大小不同时的影响.

<span id="section-12-1"></span>

### 12.1 算子的屋顶线模型

我们分析了大型语言模型 (LLM) 各类算子的屋顶线模型, 重点考察 Llama-7B、Llama-13B 和 Llama-33B [Tou23a]. 这些模型在 A100-80GB-PCIe、A40 和 A6000 等不同 GPU 上测试. 我们研究三类矩阵乘法算子, 因为它们是这些模型计算开销的主要来源. 本研究沿用报告 [Che23g] 的方法; 该报告考察批大小的作用, 我们则更关注解码与并行解码.

[表 5](#table-05) 列出各算子在预填充、解码和 Medusa 解码阶段的计算复杂度与空间复杂度. 算子包括查询、键和值矩阵的线性层 ($X W_{Q}$、$X W_{K}$、$X W_{V}$), 注意力矩阵乘法 ($Q K^\top$、$P V$), 以及上投影/门控/下投影线性层 ($X W_{u}$、$X W_{g}$、$X W_{d}$). $b$ 表示批大小, $s$ 表示序列长度, $h$ 表示隐藏维度, $i$ 表示中间维度, $n$ 表示注意力头数, $d$ 表示头维度, $q$ 表示 Medusa 候选长度. 算子详情参见 [Tou23a, Che23g].

<span id="table-05"></span>

![Table 5. Computational and space complexity of the main operators in different phases. The table is based on the corresponding table in the report Che23g.](./medusa/table-05.png)

**表 5.** 不同阶段主要算子的计算复杂度与空间复杂度. 本表基于报告 [Che23g] 的对应表.

[图 9](#figure-09)-[17](#figure-17) 给出了三类算子在不同模型 (7/13/33B) 和多种设置下的基准测试. 为评估各算子的性能与吞吐量, 我们组合了取 2 的幂、从 1 到 64 的批大小和从 128 到 8192 的序列长度 (每个算子 49 种设置). 所有图中, 各算子在预填充与解码阶段的数据点, 无论 GPU 或模型规模如何, 都聚集在十分相近的位置.

预填充阶段, 增大批大小会改变注意力矩阵乘法的 FLOP/s (见 `‘qk/pv init‘`), 但不影响运算强度 (见[图 9](#figure-09) 中的竖直虚线箭头). 相比之下, 增大序列长度会同时影响预填充阶段的 FLOP/s 与运算强度 (见[图 9](#figure-09) 中的斜向虚线箭头). 解码阶段的注意力矩阵乘法显著受内存带宽限制. 批大小和序列长度变化虽使 FLOP/s 提高, 运算强度却几乎不变 (见 `‘qk/pv ar‘`). 这说明自注意力机制没有充分利用资源.

预填充阶段的线性层大多受计算限制 (见 `‘qkv mlp init‘` 和 `‘up/gate/down init‘`). 解码阶段, 线性层的数据点形成一条斜率与 GPU 内存带宽相同的直线 (见 `‘qkv mlp ar‘` 和 `‘up/gate/down ar‘`), 表明解码阶段的线性层同样受内存带宽限制. 在这一限制下, 增大批大小可以借助更好的并行性提高实际 FLOP/s 与运算强度. 注意, 线性层只处理新词元, 与序列长度无关 (见[表 5](#table-05) 的“Decoding”部分).

<span id="figure-09"></span>

![Figure 9. The figure shows the relationship between FLOP/s and Operational Intensity for all benchmarked datapoints of Llama-7B operators on A100-80GB-PCIe. The dashed lines represent the HBM bandwidth limit (1,935GB/s) and the peak performance limit (312 TFLOP/s) Nvi20. ‘`qkv mlp`’ stands for the linear layers projecting hidden features to query/key/value features. ‘`up/gate/down`’ stands for the linear layers following the attention block. ‘`qk/pv`’ stands for the two steps of attention matrix multiplications. ‘`ar`’ stands for the decoding (autoregressive) and ‘`init`’ stands for the prefill phase.](./medusa/figure-09.png)

**图 9.** 图中展示 A100-80GB-PCIe 上 Llama-7B 各算子全部基准数据点的 FLOP/s 与运算强度关系. 虚线表示 HBM 带宽上限 (1,935GB/s) 和峰值性能上限 (312 TFLOP/s) [Nvi20]. `qkv mlp` 表示把隐藏特征投影为查询/键/值特征的线性层; `up/gate/down` 表示注意力块之后的线性层; `qk/pv` 表示注意力矩阵乘法的两个步骤; `ar` 表示解码 (自回归), `init` 表示预填充阶段.

<span id="figure-10"></span>

![Figure 10. Llama-13B operators on A100-80GB-PCIe.](./medusa/figure-10.png)

**图 10.** A100-80GB-PCIe 上的 Llama-13B 算子.

<span id="figure-11"></span>

![Figure 11. Llama-33B operators on A100-80GB-PCIe.](./medusa/figure-11.png)

**图 11.** A100-80GB-PCIe 上的 Llama-33B 算子.

<span id="figure-12"></span>

![Figure 12. Llama-7B operators on A40.](./medusa/figure-12.png)

**图 12.** A40 上的 Llama-7B 算子.

<span id="figure-13"></span>

![Figure 13. Llama-13B operators on A40.](./medusa/figure-13.png)

**图 13.** A40 上的 Llama-13B 算子.

<span id="figure-14"></span>

![Figure 14. Llama-33B operators on A40.](./medusa/figure-14.png)

**图 14.** A40 上的 Llama-33B 算子.

<span id="figure-15"></span>

![Figure 15. Llama-7B operators on A6000.](./medusa/figure-15.png)

**图 15.** A6000 上的 Llama-7B 算子.

<span id="figure-16"></span>

![Figure 16. Llama-13B operators on A6000.](./medusa/figure-16.png)

**图 16.** A6000 上的 Llama-13B 算子.

<span id="figure-17"></span>

![Figure 17. Llama-33B operators on A6000.](./medusa/figure-17.png)

**图 17.** A6000 上的 Llama-33B 算子.

<span id="section-12-2"></span>

### 12.2 Medusa 中 FLOP/s 与运算强度的变化

我们研究 Medusa 如何改变运算强度并提高 FLOP/s, 设置为 A100-80GB-PCIe 上的 Llama 33B.

首先考察注意力矩阵乘法. [图 18](#figure-18) 和[表 6](#table-06) 展示批大小固定为 16 时 Medusa 的效果. 随候选词元增多, FLOP/s 与运算强度都提高 (原始解码结果以灰点表示), 说明 Medusa 可以利用额外候选词元提高计算吞吐量. 在批大小 16、序列长度 1024、候选词元 64 的设置下, Medusa 相对普通解码取得 $44\times$ FLOP/s 和 $41\times$ 运算强度. [图 19](#figure-19) 和[表 7](#table-07) 展示序列长度固定为 1024 时 Medusa 解码的效果. 此时增大批大小不会提高运算强度.

随后考察线性层, 重点是上投影/门控/下投影线性层, 结果见[图 20](#figure-20) 和[表 8](#table-08). 解码阶段的线性层只处理未来词元, 过去词元已被缓存, 因而与序列长度无关. 我们改变批大小来观察效果. 随 Medusa 增加候选词元数并增大批大小, 可以看到系统从内存带宽受限区转向计算受限区. 这表明 Medusa 能够改变线性层的性能特征, 使其从受内存带宽限制转为受计算能力限制.

<span id="figure-18"></span>

![Figure 18. FLOP/s vs. Operational Intensity of attention matrix multiplication with batch size 16.](./medusa/figure-18.png)

**图 18.** 批大小为 16 时, 注意力矩阵乘法的 FLOP/s 与运算强度.

<span id="figure-19"></span>

![Figure 19. FLOP/s vs. Operational Intensity of attention matrix multiplication with sequence length 1024.](./medusa/figure-19.png)

**图 19.** 序列长度为 1024 时, 注意力矩阵乘法的 FLOP/s 与运算强度.

<span id="figure-20"></span>

![Figure 20. FLOP/s vs. Operational Intensity of Linear layers.](./medusa/figure-20.png)

**图 20.** 线性层的 FLOP/s 与运算强度.

<span id="table-06"></span>

![Table 6. TFLOP/s & Operational Intensity of attention matrix multiplication with batch size 16 for Llama 33B on an A100 80GB PCIe.](./medusa/table-06.png)

**表 6.** A100 80GB PCIe 上 Llama 33B 在批大小 16 时, 注意力矩阵乘法的 TFLOP/s 与运算强度.

<span id="table-07"></span>

![Table 7. TFLOP/s & Operational Intensity of attention matrix multiplication with sequence length 1024 for Llama 33B on an A100 80GB PCIe.](./medusa/table-07.png)

**表 7.** A100 80GB PCIe 上 Llama 33B 在序列长度 1024 时, 注意力矩阵乘法的 TFLOP/s 与运算强度.

<span id="table-08"></span>

![Table 8. TFLOP/s & Operational Intensity of linear layers (up/gate/down) for Llama 33B on an A100 80GB PCIe.](./medusa/table-08.png)

**表 8.** A100 80GB PCIe 上 Llama 33B 的线性层 (up/gate/down) TFLOP/s 与运算强度.

<span id="section-12-3"></span>

### 12.3 预测 Medusa 性能

我们进一步用一个简单分析模型刻画加速率. [第 3.3.1 节](#section-3-3-1)的消融结果表明, 加速率可用简单对数函数近似. 根据[图 4(a)](#figure-04) 的结果, 将曲线建模为 $\texttt{acc_rate}=0.477\log(\texttt{num_candidate})$. 我们先把批大小固定为 1、序列长度固定为 1024, 模拟 Llama-7B 一个简化块的延迟 (依次处理 $X W_{Q}$、$X W_{K}$、$X W_{V}$、$Q K^\top$、$P V$、$X W_{u}$、$X W_{g}$、$X W_{d}$). 候选词元用[第 2.1.2 节](#section-2-1-2)所述树形注意力并行处理. Medusa 的验证、接受等后处理步骤开销很小, 因此略去其延迟. [图 21](#figure-21) 展示不同候选词元数下的模拟加速率和加速比. 候选词元增加时, 两者起初都提高; 超过 64 后, 加速比开始下降, 继续增加候选长度的收益递减. 这与[图 4(b)](#figure-04) 的实验结果一致, 表明候选词元数存在一个最佳区间, Medusa 在该区间内收益最大.

[图 22](#figure-22) 绘制了序列长度固定为 1024 时, 不同批大小下的模拟加速比. 批大小超过 32 后, 加速比下降, 甚至可能产生负面作用. 原因是线性层从受内存带宽限制转为受计算限制.

我们又在批大小为 4、序列长度不同的条件下进行实验. 如[图 23](#figure-23) 所示, 不同序列长度下的最佳候选词元数较为一致. 不过, 序列越长, 整体性能越低. 性能下降主要来自注意力矩阵乘法的开销; 线性层计算不受序列长度影响, 因而保持不变.

模拟结果表明, 使用 Medusa 扩展模型时, 最佳候选词元数至关重要, 超过一定范围后收益会下降. 起初, 增大批大小通过并行性改善性能; 但批大小过大, 会让线性层从受内存带宽限制转为受计算限制, 降低加速比. 更长的序列会增加注意力矩阵乘法开销, 使性能下降, 这也凸显了优化注意力机制的必要性. 有效扩展模型需要平衡候选词元数量, 调整批大小以免转为计算受限, 并改进长序列的注意力机制. 这些策略可以改善资源利用并提高性能, 也体现了模拟对性能预测与加速策略设计的价值.

<span id="figure-21"></span>

![Figure 21. Simulated acceleration rate, speedup, and normalized latency ablation using different numbers of candidate tokens under the setting of batch size 1 and sequence length 1024 for Llama-7B on an A100 80GB PCIe.](./medusa/figure-21.png)

**图 21.** A100 80GB PCIe 上 Llama-7B 在批大小 1、序列长度 1024 时, 使用不同候选词元数的模拟加速率、加速比与归一化延迟消融.

<span id="figure-22"></span>

![Figure 22. Simulated speedup with sequence length 1024 for Llama-7B.](./medusa/figure-22.png)

**图 22.** Llama-7B 在序列长度 1024 时的模拟加速比.

<span id="figure-23"></span>

![Figure 23. Simulated speedup with batch size 4 for Llama-7B.](./medusa/figure-23.png)

**图 23.** Llama-7B 在批大小 4 时的模拟加速比.

[+1]: 经作者确认, 这个版本属于实验版本, 使用的数据与 Vicuna 7B 和 13B 有所不同.

[+2]: 此处准确率针对单个第 $i$ 高的词元定义, 即该准确率等于 top-$i$ 准确率减去 top-$(i-1)$ 准确率.

[+3]: [https://github.com/feifeibear/LLMSpeculativeSampling](https://github.com/feifeibear/LLMSpeculativeSampling)

[+author-equal]: 同等贡献.

[+author-corresponding]: 通讯作者.
