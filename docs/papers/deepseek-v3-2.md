---
title: 'DeepSeek-V3.2: Open LLM Frontier'
createTime: 2026/09/06 16:16:22
permalink: /papers/deepseek-v3-2/
---

> [DeepSeek-AI](https://www.deepseek.com/). 论文于 2025 年 12 月 2 日首次提交至 arXiv. 本网页阅读版依据 arXiv v1 的 [DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models](https://arxiv.org/abs/2512.02556v1) 编排. 精确的印刷版式和参考文献以[原始 PDF](/paper/deepseek-v3-2.pdf) 为准. [DOI](https://doi.org/10.48550/arXiv.2512.02556). [TeX 源文件](https://arxiv.org/src/2512.02556v1).

## 摘要

我们提出 DeepSeek-V3.2, 这一模型兼顾了高计算效率与出色的推理和智能体性能. DeepSeek-V3.2 的关键技术突破如下: **(1) DeepSeek 稀疏注意力 (DSA)**: 我们提出 DSA, 一种高效的注意力机制, 在保持长上下文场景中模型性能的同时大幅降低计算复杂度. **(2) 可扩展的强化学习框架**: 通过采用稳健的强化学习方案并扩大后训练算力, DeepSeek-V3.2 的性能与 GPT-5 相当. 值得注意的是, 高算力变体 DeepSeek-V3.2-Speciale 超过了 GPT-5, 推理能力与 Gemini-3.0-Pro 相当, 并在 2025 年国际数学奥林匹克竞赛 (IMO) 和国际信息学奥林匹克竞赛 (IOI) 中达到金牌水平. **(3) 大规模智能体任务合成流程**: 为把推理融入工具使用场景, 我们开发了一套新颖的合成流程, 能系统地大规模生成训练数据. 这一方法支持可扩展的智能体后训练, 显著提升模型在复杂交互环境中的泛化能力和指令遵循稳健性.

<span id="figure-01"></span>

![图 1. DeepSeek-V3.2 与其他模型的基准测试. HMMT 2025 报告二月场次, 与基线保持一致. HLE 报告纯文本子集.](./deepseek-v3-2/figure-01.png)

**图 1.** DeepSeek-V3.2 与其他模型的基准测试. HMMT 2025 报告二月场次, 与基线保持一致. HLE 报告纯文本子集.

<span id="section-1"></span>

## 1 引言

推理模型 [Ope24, Guo25] 的发布是大语言模型 (LLM) 发展中的一个转折点, 推动模型在可验证领域的整体性能大幅跃升. 此后, LLM 的能力迅速进步. 但在过去几个月里, 两条发展路线出现了明显分化. 开源社区 [Yan25g, Glm25, Min25, Kim25f] 仍在不断取得进展, 闭源专有模型 [Ope25l, Ant25f, Com25a] 的性能提升速度却明显更快. 因而, 闭源与开源模型之间的性能差距似乎并未收敛, 反而正在扩大, 专有系统在复杂任务上的优势也越来越明显.

通过分析, 我们发现有三项主要缺陷限制了开源模型处理复杂任务的能力. 第一, 从架构看, 目前主要依赖普通注意力机制 [Vas17], 严重限制了长序列处理效率. 这种低效率对可扩展部署和有效后训练都构成了很大障碍. 第二, 从资源分配看, 开源模型在后训练阶段的算力投入不足, 限制了其处理困难任务的性能. 最后, 在 AI 智能体方面, 开源模型的泛化和指令遵循能力明显落后于专有模型 [Mcp25, Luo25j, Too26], 因而难以在真实部署中充分发挥作用.

为解决这些限制, 我们首先提出 DSA, 一种旨在大幅降低计算复杂度的高效注意力机制. 该架构解决了效率瓶颈, 即使在长上下文场景中也能保持模型性能. 其次, 我们开发了稳定且可扩展的 RL 方案, 可以在后训练阶段显著扩大算力投入. 该框架为后训练分配的算力预算超过预训练成本的 10%, 从而释放更强的能力. 第三, 我们提出一套新流程, 用于在工具使用场景中培养可泛化的推理能力. 我们先采用 DeepSeek-V3 [Dee24a] 的方法开展冷启动, 在单条轨迹中统一推理与工具使用. 随后进入大规模智能体任务合成阶段, 生成超过 1,800 个不同环境和 85,000 条复杂提示. 这些大规模合成数据驱动 RL 过程, 显著增强模型在智能体场景中的泛化和指令遵循能力.

在多项推理基准上, DeepSeek-V3.2 的性能与 Kimi-k2-thinking 和 GPT-5 相近. DeepSeek-V3.2 也显著推进了开放模型的智能体能力, 在 [Mcp25, Luo25j, Too26] 提出的长尾智能体任务上表现出色. 在智能体场景中, DeepSeek-V3.2 是一种成本效益很高的选择: 它以低得多的成本显著缩小了开放模型与前沿专有模型之间的性能差距. 为了进一步推进开放模型的推理能力, 我们放宽长度限制, 开发了 DeepSeek-V3.2-Speciale. 由此, DeepSeek-V3.2-Speciale 达到了领先闭源系统 Gemini-3.0-Pro [Dee25] 的性能水平. 它在 IOI 2025, ICPC World Final 2025, IMO 2025 和 CMO 2025 中均达到金牌水平.

<span id="section-2"></span>

## 2 DeepSeek-V3.2 架构

<span id="section-2-1"></span>

### 2.1 DeepSeek 稀疏注意力

DeepSeek-V3.2 的架构与 DeepSeek-V3.2-Exp 完全相同. 与 DeepSeek-V3.1 的最后一个版本 DeepSeek-V3.1-Terminus 相比, DeepSeek-V3.2 唯一的架构改动是通过持续训练引入 DeepSeek 稀疏注意力 (DSA).

**DSA 原型.** DSA 原型主要由两部分组成: 闪电索引器和细粒度 token 选择机制.

**闪电索引器**计算查询 token $\mathbf{h}_{t}\in\mathbb{R}^{d}$ 与前序 token $\mathbf{h}_{s}\in\mathbb{R}^{d}$ 之间的索引分数 $I_{t,s}$, 用于确定查询 token 应选择哪些 token:

<span id="equation-01"></span>

$$
I_{t,s}=\sum_{j=1}^{H^{I}}w_{t,j}^{I}\cdot\mathrm{ReLU}\left(\mathbf{q}^{I}_{t,j}\cdot\mathbf{k}^{I}_{s}\right),
$$

其中, $H^{I}$ 表示索引器头数; $\mathbf{q}^{I}_{t,j}\in\mathbb{R}^{d^{I}}$ 和 $w_{t,j}^{I}\in\mathbb{R}$ 由查询 token $\mathbf{h}_{t}$ 得到; $\mathbf{k}^{I}_{s}\in\mathbb{R}^{d^{I}}$ 由前序 token $\mathbf{h}_{s}$ 得到. 出于吞吐量考虑, 我们选择 ReLU 作为激活函数. 闪电索引器的头数很少, 并且可以用 FP8 实现, 因此计算效率很高.

给定每个查询 token $\mathbf{h}_{t}$ 的索引分数 $\{I_{t,s}\}$, **细粒度 token 选择机制**只检索索引分数排名前 k 的键值条目 $\{\mathbf{c}_{s}\}$. 随后, 在查询 token $\mathbf{h}_{t}$ 与稀疏选出的键值条目 $\{\mathbf{c}_{s}\}$ 之间应用注意力机制, 得到注意力输出 $\mathbf{u}_{t}$:

<span id="equation-02"></span>

$$
\mathbf{u}_{t}=\mathrm{Attn}\left(\mathbf{h}_t, \left\{\mathbf{c}_s \,\middle|\, I_{t,s}\in\mathrm{Top}\text{-}k\left(I_{t,:}\right)\right\}\right).
$$

<span id="figure-02"></span>

![图 2. DeepSeek-V3.2 的注意力架构, 其中 DSA 基于 MLA 实现. 绿色部分表示 DSA 如何依据索引器选出排名前 k 的键值条目.](./deepseek-v3-2/figure-02.png)

**图 2.** DeepSeek-V3.2 的注意力架构, 其中 DSA 基于 MLA 实现. 绿色部分表示 DSA 如何依据索引器选出排名前 k 的键值条目.

**在 MLA 下实现 DSA.** 考虑到要从 DeepSeek-V3.1-Terminus 继续训练, 我们在 DeepSeek-V3.2 中基于 MLA [Dee24d] 实现 DSA. 在算子层面, 为保证计算效率, 每个键值条目必须由多个查询共享 [Yua25e]. 因此, 我们基于 MLA 的 MQA [Sha19] 模式实现 DSA [+1], 每个潜在向量 (即 MLA 的键值条目) 由查询 token 的所有查询头共享. 基于 MLA 的 DSA 架构见[图 2](#figure-02). 我们还提供了 DeepSeek-V3.2 的开源实现 [+2], 以明确说明各项细节.

<span id="section-2-1-1"></span>

#### 2.1.1 持续预训练

我们从上下文长度已扩展至 128K 的 DeepSeek-V3.1-Terminus 基础检查点出发, 先进行持续预训练, 再进行后训练, 最终得到 DeepSeek-V3.2.

DeepSeek-V3.2 的持续预训练分为两个训练阶段. 两个阶段的训练数据分布都与 DeepSeek-V3.1-Terminus 扩展至 128K 长上下文时所用的数据完全一致.

**稠密预热阶段.** 我们先用一个很短的预热阶段初始化闪电索引器. 这一阶段保留稠密注意力, 并冻结闪电索引器之外的所有模型参数. 为使索引器输出与主注意力分布对齐, 对第 $t$ 个查询 token, 我们先把所有注意力头的主注意力分数相加. 随后沿序列维度对该总和做 L1 归一化, 得到目标分布 $p_{t,:}\in\mathbb{R}^{t}$. 基于 $p_{t,:}$, 我们将 KL 散度损失作为索引器的训练目标:

<span id="equation-03"></span>

$$
\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,:}\,\middle\|\,\mathrm{Softmax}\left({I}_{t,:}\right)\right).
$$

预热采用 $10^{-3}$ 的学习率. 索引器仅训练 1000 步, 每步包含 16 条 128K token 的序列, 共计 2.1B token.

**稀疏训练阶段.** 索引器预热完成后, 我们引入细粒度 token 选择机制, 并优化所有模型参数, 使模型适应 DSA 的稀疏模式. 这一阶段仍让索引器输出与主注意力分布对齐, 但只考虑选中的 token 集合 $\mathcal{S}_{t}=\left\{s \,\middle|\, I_{t,s}\in\mathrm{Top}\text{-}k\left(I_{t,:}\right)\right\}$:

<span id="equation-04"></span>

$$
\mathcal{L}^{I}=\sum_{t}\mathbb{D}_{\mathrm{KL}}\left(p_{t,\mathcal{S}_{t}}\,\middle\|\,\mathrm{Softmax}\left(I_{t,\mathcal{S}_t}\right)\right).
$$

需要说明的是, 为单独优化索引器, 我们将它的输入从计算图中分离. 索引器的训练信号只来自 $\mathcal{L}^{I}$, 主模型则只按语言建模损失优化. 稀疏训练阶段采用 $7.3\times 10^{-6}$ 的学习率, 每个查询 token 选择 2048 个键值 token. 主模型与索引器共同训练 $15000$ 步, 每步包含 480 条 128K token 的序列, 共计 943.7B token.

<span id="section-2-2"></span>

### 2.2 等效性评估

**标准基准.** 2025 年 9 月, 我们在一组覆盖多种能力的基准上评估 DeepSeek-V3.2-Exp, 并将其与 DeepSeek-V3.1-Terminus 比较, 二者性能相近. DeepSeek V3.2 Exp 显著提高了长序列上的计算效率, 同时在短上下文和长上下文任务上都没有出现相较 DeepSeek-V3.1-Terminus 明显的性能下降.

**人类偏好.** 直接评估人类偏好本身容易受到偏差影响, 因此我们用 ChatbotArena 作为间接评估框架, 近似衡量用户对新基础模型的偏好. DeepSeek-V3.1-Terminus 与 DeepSeek-V3.2-Exp 采用完全相同的后训练策略; 根据 2025 年 11 月 10 日进行的评估, 两者 Elo 分数十分接近. 结果说明, 新基础模型虽然加入了稀疏注意力机制, 性能仍与上一版本相当.

**长上下文评估.** DeepSeek-V3.2-Exp 发布后, 有多项独立长上下文评估采用此前未见的测试集进行测试. AA-LCR [+3] 是其中一个代表性基准, DeepSeek-V3.2-Exp 在推理模式下比 DeepSeek-V3.1-Terminus 高 4 分. 在 Fiction.liveBench 评估 [+4] 中, DeepSeek-V3.2-Exp 的多项指标也始终优于 DeepSeek-V3.1-Terminus. 这些证据表明, DeepSeek-V3.2-Exp 的基础检查点在长上下文任务上没有退步.

<span id="section-2-3"></span>

### 2.3 推理成本

DSA 将主模型的核心注意力复杂度从 $\mathcal{O}(L^2)$ 降至 $\mathcal{O}(L k)$, 其中 $k$ ($\ll L$) 是选中的 token 数. 闪电索引器的复杂度仍为 $\mathcal{O}(L^2)$, 但与 DeepSeek-V3.1-Terminus 的 MLA 相比, 所需计算量小得多. 结合优化后的实现, DSA 在长上下文场景中获得了显著的端到端加速. [图 3](#figure-03) 给出了 DeepSeek-V3.1-Terminus 与 DeepSeek-V3.2 的 token 成本如何随 token 在序列中的位置变化. 这些成本来自部署在 H800 GPU 上的真实服务基准测试, GPU 租用价格按每卡每小时 2 美元计算. 对于短序列预填充, 我们专门实现了一个带掩码的 MHA 模式来模拟 DSA, 可在短上下文条件下获得更高效率.

<span id="figure-03"></span>

![图 3. DeepSeek-V3.1-Terminus 与 DeepSeek-V3.2 在 H800 集群上的推理成本.](./deepseek-v3-2/figure-03.png)

**图 3.** DeepSeek-V3.1-Terminus 与 DeepSeek-V3.2 在 H800 集群上的推理成本.

<span id="section-3"></span>

## 3 后训练

持续预训练结束后, 我们进行后训练, 得到最终的 DeepSeek-V3.2. DeepSeek-V3.2 的后训练与稀疏持续预训练阶段一样采用稀疏注意力. DeepSeek-V3.2 保持与 DeepSeek-V3.2-Exp 相同的后训练流程, 包括专家蒸馏和混合 RL 训练.

**专家蒸馏.** 对每项任务, 我们先开发一个只用于该领域的专门模型, 所有专家模型都从同一个预训练 DeepSeek-V3.2 基础检查点微调而来. 除写作任务和通用问答外, 框架还包含六个专门领域: 数学, 编程, 通用逻辑推理, 通用智能体任务, 智能体编程和智能体搜索, 并且每个领域都支持思考模式与非思考模式. 每个专家都使用大规模强化学习 (RL) 算力进行训练. 我们还分别用不同模型生成长思维链推理 (思考模式) 和直接作答 (非思考模式) 的训练数据. 专家模型准备完成后, 再由它们为最终检查点生成各领域数据. 实验结果表明, 蒸馏数据训练的模型性能仅略低于各领域专家, 而随后的 RL 训练基本消除了这一差距.

**混合 RL 训练.** DeepSeek-V3.2 仍以组相对策略优化 (GRPO) [Sha24d, Guo25] 作为 RL 训练算法. 与 DeepSeek-V3.2-Exp 相同, 我们把推理, 智能体和人类对齐训练合并到一个 RL 阶段. 这种做法兼顾不同领域的性能, 同时避免多阶段训练方式中常见的灾难性遗忘. 对推理和智能体任务, 我们采用基于规则的结果奖励, 长度惩罚和语言一致性奖励. 对通用任务, 我们采用生成式奖励模型, 并为每条提示分别设置评估量规.

**DeepSeek-V3.2 与 DeepSeek-V3.2-Speciale.** DeepSeek-V3.2 融合了从专家模型蒸馏得到的推理, 智能体和人类对齐数据, 再经过数千步持续 RL 训练, 得到最终检查点. 为研究延长思考的潜力, 我们还开发了实验变体 DeepSeek-V3.2-Speciale. 该模型只用推理数据训练, 并在 RL 中采用较弱的长度惩罚. 我们还加入 DeepSeekMath-V2 [Sha25] 的数据集与奖励方法, 以增强数学证明能力.

下面重点介绍我们如何建立稳定方案来扩大 RL 算力 ([第 3.1 节](#section-3-1)), 以及如何把思考融入智能体任务 ([第 3.2 节](#section-3-2)).

<span id="section-3-1"></span>

### 3.1 扩展 GRPO

先回顾 GRPO 的目标. 对每个问题 $q$, 从旧策略 $\pi_{\mathrm{old}}$ 采样一组回答 $\{o_{1},\cdots,o_{G}\}$; GRPO 通过最大化以下目标来优化策略模型 $\pi_{\theta}$:

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\kern 5.0pt & \mathbb{E}_{q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|q)}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|} \\
\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)-\beta\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)\Bigg],
\end{aligned}
$$

其中

<span id="equation-06"></span>

$$
r_{i,t}(\theta)=\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}
$$

是当前策略与旧策略之间的重要性采样比. $\varepsilon$ 和 $\beta$ 分别是控制截断范围与 KL 惩罚强度的超参数. $\hat{A}_{i,t}$ 是 $o_{i,t}$ 的优势, 由每组内归一化后的结果奖励估计. 具体来说, 一组奖励模型会为组内每个输出 $o_{i}$ 评定结果奖励 $R_{i}$, 分别得到 $G$ 个奖励 $\boldsymbol{R}=\{R_{1},\cdots,R_{G}\}$. $o_{i,t}$ 的优势等于输出 $o_{i}$ 的奖励减去组内平均奖励, 即 $\hat{A}_{i,t}=R_{i}-\mathrm{mean}(\boldsymbol{R})$.

下面介绍直接建立在 GRPO 算法之上, 用于稳定 RL 扩展的附加策略.

**无偏 KL 估计.** 由于 $o_{i,t}$ 从旧策略 $\pi_{\mathrm{old}}(\cdot|q,o_{i,<t})$ 中采样, 我们修正 K3 估计器 [Sch20a], 利用当前策略 $\pi_{\theta}$ 与旧策略 $\pi_{\mathrm{old}}$ 之间的重要性采样比, 得到无偏 KL 估计.

<span id="equation-07"></span>

$$
\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)=\frac{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}\left(\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-\log\frac{\pi_{\mathrm{ref}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}-1\right).
$$

经过这一调整, KL 估计器的梯度变为无偏, 消除了系统性估计误差, 因而有助于稳定收敛. 这与原始 K3 估计器形成鲜明对比, 特别是当采样 token 在当前策略下的概率远低于参考策略时, 即 $\pi_{\theta}\ll\pi_{\mathrm{ref}}$. 此时, K3 估计器的梯度会赋予这些 token 过大且无上界的权重, 以提高它们的似然. 由此产生的噪声梯度更新会不断累积, 降低后续迭代中的样本质量, 并使训练不稳定. 实践中, 我们发现不同领域适合不同强度的 KL 正则化. 对数学等一些领域, 使用较弱的 KL 惩罚甚至完全省略 KL 惩罚, 反而可能取得更好的性能.

**离策略序列掩码.** 为提高 RL 系统效率, 我们通常生成一大批 rollout 数据, 再拆成多个小批次, 用于数次梯度更新. 这种做法天然会引入离策略行为. 用于高效生成数据的推理框架往往经过高度优化, 实现细节可能不同于训练框架. 这种训推不一致会进一步加重离策略程度. 为稳定训练并提高对离策略更新的容忍度, 我们根据数据采样策略 $\pi_{\mathrm{old}}$ 与当前策略 $\pi_{\theta}$ 之间的 KL 散度, 屏蔽会造成显著策略偏离的负样本序列. 更具体地说, 我们在 GRPO 损失中引入二值掩码 $M$:

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\kern 5.0pt & \mathbb{E}_{q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|q)}\Bigg[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|} \\
\min\left(r_{i,t}(\theta)\hat{A}_{i,t},\mathrm{clip}\left(r_{i,t}(\theta),1-\varepsilon,1+\varepsilon\right)\hat{A}_{i,t}\right)M_{i,t}-\beta\mathbb{D}_{\mathrm{KL}}\left(\pi_{\theta}(o_{i,t})\,\middle\|\,\pi_{\mathrm{ref}}(o_{i,t})\right)\Bigg],
\end{aligned}
$$

其中

<span id="equation-09"></span>

$$
M_{i,t}=\begin{cases}0&{\hat{A}_{i,t}<0,\frac{1}{|o_{i}|}\sum_{t=1}^{|o_{i}|}\log\frac{\pi_{\mathrm{old}}(o_{i,t}|q,o_{i,<t})}{\pi_{\theta}(o_{i,t}|q,o_{i,<t})}>\delta}\\[4.30554pt]
1&{\mathrm{otherwise},}\end{cases}
$$

$\delta$ 是控制策略偏离阈值的超参数. 这里的 $\pi_{\mathrm{old}}$ 表示推理框架直接返回的采样概率, 因此旧策略与当前策略之间的 KL 散度涵盖了上述两种离策略来源. 还需说明, 我们只屏蔽优势为负的序列.

直观上, 模型从自己的错误中学习时受益最大, 而高度离策略的负样本可能产生反作用, 误导优化过程或使其不稳定. 实验表明, 在一些原本会出现不稳定的训练场景中, 离策略序列掩码操作提高了稳定性.

**保持路由.** 混合专家 (MoE) 模型在推理时只激活部分专家模块, 从而提高计算效率. 但推理与训练框架之间的差异再叠加策略更新, 可能让相同输入在推理和训练时走过不同的专家路由. 这种不一致会使活跃参数子空间突然改变, 导致优化不稳定并加重离策略问题. 为缓解这一问题, 我们保留在推理框架采样时使用的专家路由, 并在训练时强制使用相同路由, 确保优化的是同一组专家参数. 研究发现, 保持路由操作对 MoE 模型的 RL 训练稳定性十分重要, 自 DeepSeek-V3-0324 起已用于我们的 RL 训练流程.

**保持采样掩码.** Top-p 和 top-k 采样是常用的采样策略, 用于提高 LLM 生成回答的质量. 在 RL 训练中使用这些策略也有好处, 因为它可以避免采样到原本会成为优化目标的极低概率 token. 截断虽能维持样本质量, 却会造成 $\pi_{\mathrm{old}}$ 与 $\pi_{\theta}$ 的动作空间不一致, 违背重要性采样原理并使训练不稳定. 为此, 我们保留从 $\pi_{\mathrm{old}}$ 采样时的截断掩码, 并在训练时将其应用于 $\pi_{\theta}$, 以保证两个策略拥有相同的动作子空间. 实验中, 我们发现 top-p 采样与保持采样掩码策略结合, 可以有效维持 RL 训练期间的语言一致性.

<span id="section-3-2"></span>

### 3.2 工具使用中的思考

<span id="section-3-2-1"></span>

#### 3.2.1 思考上下文管理

DeepSeek-R1 已经表明, 加入思考过程可以显著增强模型解决复杂问题的能力. 在此基础上, 我们希望把思考能力融入工具调用场景.

我们观察到, 如果照搬 DeepSeek-R1 的策略, 即第二轮消息到达时丢弃推理内容, 会造成严重的 token 浪费. 每次后续工具调用时, 这种做法都迫使模型重新推理整个问题. 为解决这一点, 我们为工具调用场景专门设计了[图 4](#figure-04) 所示的上下文管理方法:

- 只有对话中出现新的**用户消息**时, 才丢弃历史推理内容. 如果只是追加工具相关消息 (如工具输出), 整个交互过程中都**保留**推理内容.
- 移除推理轨迹时, **工具调用及其结果**的历史仍保留在上下文中.

需要注意的是, Roo Code 或 Terminus 等一些智能体框架通过用户消息模拟工具交互. 由于上述上下文管理规则, 这些框架可能无法充分受益于增强后的推理保留机制. 因此, 为在这类架构中获得最佳性能, 我们建议使用非思考模型.

<span id="figure-04"></span>

![图 4. 工具调用场景中的思考保留机制.](./deepseek-v3-2/figure-04.png)

**图 4.** 工具调用场景中的思考保留机制.

<span id="section-3-2-2"></span>

#### 3.2.2 冷启动

已有推理数据 (非智能体) 和非推理智能体数据时, 将两种能力结合起来的一种直接策略是精心设计提示. 我们认为, 模型已经具备足够的能力准确遵循明确指令, 因而可以把工具执行顺畅地融入推理过程.

为了说明冷启动机制的工作方式, 我们选取了附录[表 6](#table-06)-[表 8](#table-08) 所示的训练数据样本. 需要说明的是, 不同任务提示会对应不同的系统提示. [表 6](#table-06)-[表 8](#table-08) 给出了一个与竞赛编程提示对应的示例. [表 6](#table-06) 展示推理数据示例, 其中系统提示明确要求模型先推理再给出最终答案, 并用特殊标签 `<think></think>` 标记推理路径. [表 7](#table-07) 展示非推理智能体数据的提示, 其中系统提示包含工具调用指导. [表 8](#table-08) 展示我们设计的系统提示, 用于指示模型在推理过程中加入多次工具调用.

这样得到的工具使用推理模式虽然可能不够稳健, 但模型有时已经可以生成预期轨迹, 为后续强化学习阶段提供基础.

<span id="section-3-2-3"></span>

#### 3.2.3 大规模智能体任务

多样化的 RL 任务对增强模型稳健性十分重要. 对搜索, 代码工程和代码解释等任务, 我们使用真实世界的工具, 包括实际的 Web 搜索 API, 编程工具和 Jupyter Notebook. 这些 RL 环境是真实的, 但其中的提示来自互联网或合成生成, 而非真实用户交互. 对其他任务, 环境和提示均为合成构建. 我们使用的智能体任务见[表 1](#table-01).

<span id="table-01"></span>

![表 1. 不同智能体任务的说明, 包括任务数量, 环境类型 (真实或合成) 和提示来源 (提取或合成).](./deepseek-v3-2/table-01.png)

**表 1.** 不同智能体任务的说明, 包括任务数量, 环境类型 (真实或合成) 和提示来源 (提取或合成).

**搜索智能体.** 我们采用一个基于 DeepSeek-V3.2 的多智能体流程, 生成多样且高质量的训练数据. 首先从大规模 Web 语料中抽取多个领域内信息丰富的长尾实体. 随后, 问题构建智能体使用可配置深度和广度参数的搜索工具探索每个实体, 将发现的信息整合成问答对. 多个配置各异 (使用不同检查点, 系统提示等) 的答案生成智能体为每个问答对生成多样的候选回答. 一个具备搜索能力的验证智能体经过多轮验证所有答案, 只保留真实答案正确且所有候选回答都可验证为错误的样本. 这些数据覆盖多种语言, 领域和难度. 为补充这些可验证样本并更贴近真实使用方式, 我们还从已有的 helpful RL 数据集中筛选搜索工具能带来可测量收益的实例, 加入数据集. 随后, 我们围绕多个质量维度制定详细评估量规, 并用生成式奖励模型按量规为回答打分. 这种混合方法可以同时优化事实可靠性与实际帮助程度.

**代码智能体.** 我们从 GitHub 的数百万组 issue-Pull Request (PR) 对中挖掘数据, 为解决软件问题构建大规模可执行环境. 该数据集使用启发式规则和基于 LLM 的判断严格过滤, 以保证质量; 每个条目都必须包含合理的问题描述, 相关的正确补丁和用于验证的测试补丁. 我们使用由 DeepSeek-V3.2 驱动的自动环境搭建智能体为这些数据构建可执行环境. 该智能体负责安装包, 解析依赖并运行测试. 测试结果以标准 JUnit 格式输出, 确保不同编程语言和测试框架都能用一致方式解析. 只有应用正确补丁后, false-to-positive (F2P) 测试用例数非零 (表示问题已修复), 且 pass-to-fail (P2F) 测试用例数为零 (表示没有回归), 才认为环境成功构建. 通过这套流程, 我们成功构建了数万个可复现的问题解决环境, 覆盖 Python, Java, JavaScript, TypeScript, C, C++, Go 和 PHP 等多种编程语言.

**代码解释器智能体.** 我们使用 Jupyter Notebook 作为代码解释器来处理复杂推理任务. 为此, 我们整理了一组涵盖数学, 逻辑和数据科学的多样问题, 每个问题都要求模型借助代码执行能力得出解答.

**通用智能体.** 为扩大 RL 中智能体环境与任务的规模, 我们采用自动环境合成智能体, 合成 1,827 个面向任务的环境. 这些任务难以解决, 但容易验证. 合成流程主要包括环境与工具集构建, 任务合成和解答生成. 具体流程如下.

- 给定一个任务类别 (如规划旅行行程), 以及配有 bash 和搜索工具的沙箱, 智能体先用这些工具从互联网生成或检索相关数据, 并将其存入沙箱数据库.
- 随后, 智能体合成一组面向该任务的工具, 每个工具实现为一个函数.
- 为生成既有挑战性又能自动验证的任务, 智能体先根据当前数据库提出一个简单任务, 同时用 Python 实现对应的解答函数和验证函数. 解答函数只能调用工具函数或进行逻辑计算, 不能调用其他函数或直接访问数据库, 从而确保任务只能通过工具接口解决. 解答函数给出的结果还必须通过验证函数检查. 如果未通过, 智能体会修改解答函数或验证函数, 直至解答输出通过验证. 接着, 智能体逐步提高任务难度, 并更新相应的解答函数与验证函数. 在迭代过程中, 如果当前工具集不足以解决任务, 智能体会扩充工具集.

按照这套流程, 我们得到数千个 $\langle\mathrm{environment},\mathrm{tools},\mathrm{task},\mathrm{verifier}\rangle$ 四元组. 随后用 DeepSeek-V3.2 在该数据集上进行 RL, 并只保留 pass@100 非零的实例, 最终得到 1,827 个环境及对应任务 (共 4,417 个). 下面给出一个合成的旅行规划示例. 该示例说明, 在庞大的组合空间中搜索满足所有约束的旅行方案很难, 但检查一个候选方案是否满足约束相对直接.

![合成旅行规划任务示例](./deepseek-v3-2/example-01.png)

**合成任务示例: 旅行规划.**

![合成旅行规划任务的工具集](./deepseek-v3-2/example-02.png)

**旅行规划工具集.**

<span id="section-4"></span>

## 4 评估

<span id="section-4-1"></span>

### 4.1 主要结果

我们在 MMLU-Pro [Wan24c], GPQA Diamond [Rei23], Human Last Exam (HLE) 纯文本子集 [Pha25], LiveCodeBench (2024.08-2025.04), Codeforces, Aider-Polyglot, AIME 2025, HMMT Feb 2025, HMMT Nov 2025 [Bal25], IMOAnswerBench [Luo25], Terminal Bench 2.0, SWE-Verified [Ope24e], SWE Multilingual [Yan25b], BrowseComp [Wei25b], BrowseCompZh [Zho25], $\tau^{2}$-bench [Bar25c], MCP-Universe [Luo25j], MCP-Mark [Mcp25] 和 Tool-Decathlon [Too26] 上评估模型. 工具使用基准采用标准函数调用格式评估, 模型设为思考模式. 对 MCP-Universe [Luo25j] 和 MCP-Mark [Mcp25], 由于搜索与 Playwright 环境可能和官方设置略有不同, 所有模型都在我们的内部环境中评估. 温度设为 1.0, 上下文窗口设为 128K token. 对 AIME, HMMT, IMOAnswerBench 和 HLE 等数学任务, 我们使用以下模板评估: `"{question}\nPlease reason step by step, and put your final answer within \boxed{}."` 对 HLE, 我们还用官方模板评估了 DeepSeek-V3.2-Thinking, 得分为 $23.9$.

<span id="table-02"></span>

![表 2. DeepSeek-V3.2 与闭源和开放模型的比较. 对开放模型, 只比较支持工具使用中思考的模型. 粗体数字表示各模型类别 (开源与闭源) 中的最高分. τ²-Bench 结果取各类别的平均值. BrowseComp 中采用上下文管理技术的性能以 * 标记.](./deepseek-v3-2/table-02.png)

**表 2.** DeepSeek-V3.2 与闭源和开放模型的比较. 对开放模型, 只比较支持工具使用中思考的模型. 粗体数字表示各模型类别 (开源与闭源) 中的最高分. $\tau^{2}$-Bench 结果取各类别的平均值. BrowseComp 中采用上下文管理技术的性能以 * 标记.

在推理任务上, DeepSeek-V3.2 的性能与 GPT-5-high 相近, 但略低于 Gemini-3.0-Pro. 与 K2-Thinking 相比, DeepSeek-V3.2 使用的输出 token 少得多, 同时取得了相当的分数, 如[表 3](#table-03) 所示. 这些性能提升可以归因于 RL 训练获得了更多算力. 过去几个月中, 我们观察到性能随 RL 训练预算增加而持续提升, 目前该预算已经超过预训练成本的 10%. 我们推测, 增加算力预算还可进一步增强推理能力. 这里给出的 DeepSeek-V3.2 性能受到长度约束奖励模型的限制; 移除该限制后, 模型性能会进一步提高, 详见[第 4.2 节](#section-4-2).

在代码智能体评估中, DeepSeek-V3.2 在 SWE-bench Verified 和 Terminal Bench 2.0 上都明显优于开源 LLM, 表明它在真实编程工作流中具有潜力. 对于 Terminal Bench 2.0, 如前所述, 我们用于"思考模式"的上下文管理策略目前与 Terminus 不兼容, 因此报告的 46.4 分是在 Claude Code 框架下得到的. 我们也使用 Terminus 评估了 DeepSeek-V3.2 的非思考模式, 得分为 39.3. SWE-bench Verified 的主要分数来自内部框架. 在其他设置下开展的稳健性测试, 包括 Claude Code 和 RooCode 框架以及非思考模式, 得到了一致的结果, 分数介于 72 和 74 之间.

在搜索智能体评估中, 我们使用标准商业搜索 API 评估模型. DeepSeek-V3.2 的最大上下文长度只有 128K, 因而约有 20% 以上的测试用例会超过这一上限. 为此, 我们采用上下文管理方法计算最终得分. 作为参考, 不使用上下文管理时得分为 51.4. 更多细节见[第 4.4 节](#section-4-4).

在工具使用基准上, DeepSeek-V3.2 大幅缩小了开源与闭源 LLM 的性能差距, 但仍低于前沿模型. 对 $\tau^{2}$-bench, 我们让模型本身充当用户智能体, 最终各类别得分为 63.8 (Airline), 81.1 (Retail) 和 96.2 (Telecom). 对 MCP 基准, 我们采用函数调用格式, 并把工具输出放在角色指定为 'tool' 而非 'user' 的消息中. 测试时, 我们观察到 DeepSeek-V3.2 经常进行冗余的自我验证, 生成过长的轨迹. 这一倾向经常导致上下文长度超过 128K 限制, 在 MCP-Mark GitHub 和 Playwright 评估等任务中尤其明显. 因此, 这一现象影响了 DeepSeek-V3.2 的最终性能. 不过, 加入上下文管理策略还可进一步提升性能. 我们把它视作未来工作的一个方向, 也是用户实际使用时需要考虑的问题. 即使存在这一问题, DeepSeek-V3.2 仍显著优于现有开放模型. 这些基准所用的环境与工具集并未出现在 RL 训练中, 因此观察到的改进说明 DeepSeek-V3.2 可以把推理策略泛化到分布外的智能体场景. 智能体场景中的非思考模型评估见附录[表 9](#table-09).

<span id="section-4-2"></span>

### 4.2 DeepSeek-V3.2-Speciale 的结果

<span id="table-03"></span>

![表 3. 推理模型的基准性能与效率. 每项基准的单元格给出准确率和输出 token 数 (千). 每项基准的最高准确率用粗体表示, 第二高用下划线表示.](./deepseek-v3-2/table-03.png)

**表 3.** 推理模型的基准性能与效率. 每项基准的单元格给出准确率和输出 token 数 (千). 每项基准的最高准确率用粗体表示, 第二高用下划线表示.

[表 3](#table-03) 表明, DeepSeek-V3.2-Speciale 通过使用更多推理 token 获得更好的性能, 在多项基准上超过当前最先进的 Gemini-3.0-Pro. [表 4](#table-04) 还显示, 这个通用模型未经过针对性训练, 却在 2025 年国际信息学奥林匹克竞赛 (IOI) 和 ICPC 世界总决赛 (ICPC WF) 中达到金牌水平. 采用 [Sha25] 中的技术后, 模型在复杂证明任务上表现出色, 在 2025 年国际数学奥林匹克竞赛 (IMO) 和中国数学奥林匹克竞赛 (CMO) 中达到金牌分数线 [+5]. 详细评估方案见附录[第 9 节](#section-9).

不过, DeepSeek-V3.2-Speciale 的 token 效率仍明显低于 Gemini-3.0-Pro. 为降低部署成本和延迟, 我们在训练正式版 DeepSeek-V3.2 时采用更严格的 token 约束, 以优化性能与成本之间的权衡. 我们认为, token 效率仍是未来研究的重点领域.

<span id="table-04"></span>

![表 4. DeepSeek-V3.2-Speciale 在顶级数学与编程竞赛中的表现. ICPC WF 2025 报告每道成功解出问题的提交次数. DeepSeek-V3.2-Speciale 在 ICPC WF 2025 中排名第 2, 在 IOI 2025 中排名第 10.](./deepseek-v3-2/table-04.png)

**表 4.** DeepSeek-V3.2-Speciale 在顶级数学与编程竞赛中的表现. ICPC WF 2025 报告每道成功解出问题的提交次数. DeepSeek-V3.2-Speciale 在 ICPC WF 2025 中排名第 2, 在 IOI 2025 中排名第 10.

<span id="section-4-3"></span>

### 4.3 合成智能体任务

本节通过消融实验研究合成智能体任务的效果, 主要关注两个问题. 第一, 合成任务对强化学习而言是否足够困难? 第二, 这些合成任务的泛化能力如何, 即它们能否迁移到不同下游任务或真实环境?

为回答第一个问题, 我们从通用合成智能体任务中随机抽取 50 个实例, 同时评估用于合成的模型和前沿闭源 LLM. 如[表 5](#table-05) 所示, DeepSeek-V3.2-Exp 的准确率只有 12%, 前沿闭源模型最高也只有 62%. 这些结果说明, 合成数据中含有对 DeepSeek-V3.2-Exp 和前沿闭源模型都很困难的智能体任务.

<span id="table-05"></span>

![表 5. 不同模型在通用合成任务上的准确率.](./deepseek-v3-2/table-05.png)

**表 5.** 不同模型在通用合成任务上的准确率.

为研究在合成数据上进行 RL 能否泛化到其他任务或真实环境, 我们对 DeepSeek-V3.2 的 SFT 检查点 (记作 DeepSeek-V3.2-SFT) 应用 RL. 为排除长 CoT 和其他 RL 数据的影响, 我们只在非思考模式下使用合成智能体任务进行 RL. 随后, 将该模型与 DeepSeek-V3.2-SFT 和 DeepSeek-V3.2-Exp 比较, 其中 DeepSeek-V3.2-Exp 只在搜索和代码环境中进行 RL 训练. 如[图 5](#figure-05) 所示, 在大规模合成数据上进行 RL 后, 模型在 Tau2Bench, MCP-Mark 和 MCP-Universe 基准上相较 DeepSeek-V3.2-SFT 大幅提升. 相比之下, 将 RL 限制在代码和搜索场景并不能改善这些基准上的性能, 这进一步说明了合成数据的潜力.

<span id="figure-05"></span>

![图 5. DeepSeek-V3.2-SFT 只使用合成通用智能体数据进行 RL 训练的结果.](./deepseek-v3-2/figure-05.png)

**图 5.** DeepSeek-V3.2-SFT 只使用合成通用智能体数据进行 RL 训练的结果.

<span id="section-4-4"></span>

### 4.4 搜索智能体的上下文管理

<span id="figure-06"></span>

![图 6. 不同测试时算力扩展策略下 Browsecomp 的准确率.](./deepseek-v3-2/figure-06.png)

**图 6.** 不同测试时算力扩展策略下 Browsecomp 的准确率.

即使上下文窗口扩展到 128k, 智能体工作流仍经常遇到最大长度限制, 尤其是在搜索场景中, 推理过程会因此提前截断. 这一瓶颈使测试时算力的潜力无法充分发挥. 为解决这一问题, 当 token 用量超过上下文窗口长度的 80% 时, 我们采用几种简单策略管理上下文, 以扩展测试时 token 预算. 这些策略包括: (1) **Summary**, 总结溢出的轨迹并重新开始 rollout; (2) **Discard-75%**, 丢弃轨迹中最前面的 75% 工具调用历史以腾出空间; (3) **Discard-all**, 丢弃此前全部工具调用历史并重置上下文 (类似于新上下文工具 [Ant25a]). 作为比较, 我们还实现了并行扩展基线 **Parallel-fewest-step**, 它会采样 N 条彼此独立的轨迹, 再选出步骤最少的一条.

我们在 BrowseComp 基准 [Wei25b] 上评估这些策略. 如[图 6](#figure-06) 所示, 在不同算力预算下, 上下文管理让模型能够扩大测试时算力, 为更多执行步骤留出空间, 因而显著提升性能. 例如, Summary 将平均步数延长到 364, 性能最高提高到 60.2. 不过, 它的整体效率相对较低. Discard-all 虽然简单, 却在效率和可扩展性上都有良好表现, 用明显更少的步骤获得了 67.6 分, 与并行扩展相当.

总之, 测试时算力既可以通过上下文管理串行扩展, 也可以并行扩展, 两种方式都能有效增强模型解决问题的能力. 但不同策略的效率和可扩展性有所差异. 因而, 对模型性能进行基准测试时, 必须考虑实际计算成本. 如何找到串行与并行扩展的最佳组合, 同时最大化效率和可扩展性, 仍是未来工作的重要方向.

<span id="section-5"></span>

## 5 结论, 局限与未来工作

本文提出 DeepSeek-V3.2, 一个有效兼顾计算效率与高级推理能力的框架. DSA 在不牺牲长上下文性能的情况下解决了计算复杂度问题. 通过增加算力预算, DeepSeek-V3.2 在推理基准上取得了与 GPT-5 相当的性能. 最后, 大规模智能体任务合成流程的加入显著增强了工具使用能力, 为基于开放 LLM 构建稳健且可泛化的 AI 智能体带来新的可能. 高算力变体 DeepSeek-V3.2-Speciale 在 IMO 和 IOI 中达到金牌水平, 为开放 LLM 树立了一个里程碑.

虽然取得了这些成果, 但与 Gemini-3.0-Pro 等前沿闭源模型相比, DeepSeek-V3.2 仍有一些局限. 第一, 由于总训练 FLOP 较少, DeepSeek-V3.2 的世界知识广度仍落后于领先的专有模型. 我们计划在未来版本中扩大预训练算力, 缩小这一知识差距. 第二, token 效率仍是一个问题; DeepSeek-V3.2 通常需要更长的生成轨迹 (即更多 token), 才能达到 Gemini-3.0-Pro 等模型的输出质量. 未来工作将优化模型推理链的智能密度, 以提高效率. 第三, 它解决复杂任务的能力仍不及前沿模型, 因此我们还需进一步改进基础模型和后训练方案.

<span id="section-6"></span>

## 6 MLA 的 MHA 与 MQA 模式

<span id="figure-07"></span>

![图 7. MLA 的 MHA 与 MQA 模式示意图. DeepSeek-V3.1-Terminus 在训练和预填充时使用 MHA 模式, 解码时使用 MQA 模式.](./deepseek-v3-2/figure-07.png)

**图 7.** MLA 的 MHA 与 MQA 模式示意图. DeepSeek-V3.1-Terminus 在训练和预填充时使用 MHA 模式, 解码时使用 MQA 模式.

[图 7](#figure-07) 展示了 MLA 的两个方面, 即 MHA 与 MQA 模式, 以及两者之间的转换.

<span id="section-7"></span>

## 7 冷启动模板

<span id="table-06"></span>

![表 6. 推理数据系统提示示例](./deepseek-v3-2/table-06.png)

**表 6.** 推理数据系统提示示例. 系统提示要求模型在 `<think></think>` 标签中输出推理过程.

<span id="table-07"></span>

![表 7. 智能体系统提示与工具调用格式占位符](./deepseek-v3-2/table-07.png)

**表 7.** `{TOOL-DESCRIPTIONS}` 和 `{TOOLCALL-FORMAT}` 会被替换为具体工具及我们设计的工具调用格式.

<span id="table-08"></span>

![表 8. 要求推理的智能体系统提示](./deepseek-v3-2/table-08.png)

**表 8.** 模型在思考过程中执行工具调用.

<span id="section-8"></span>

## 8 DeepSeek-V3.2 非思考模式智能体评估

<span id="table-09"></span>

![表 9. DeepSeek-V3.2 非思考模式与思考模式的比较. 表中 Terminal Bench 分数使用 Claude Code 框架评估. 非思考模式使用 Terminus 框架时, Terminal Bench 2.0 得分为 39.3.](./deepseek-v3-2/table-09.png)

**表 9.** DeepSeek-V3.2 非思考模式与思考模式的比较. 表中 Terminal Bench 分数使用 Claude Code 框架评估. 非思考模式使用 Terminus 框架时, Terminal Bench 2.0 得分为 39.3.

非思考模式的性能略低于思考模式, 但仍有竞争力.

<span id="section-9"></span>

## 9 IOI, ICPC 世界总决赛, IMO 与 CMO 的评估方法

所有竞赛中, 模型的最大生成长度都设为 128k. 评估不使用工具或互联网访问, 并严格遵守竞赛的时间和尝试次数限制.

在 IOI 评估中, 我们按照官方竞赛规则设计提交策略. 规则允许每道题最多提交 50 次, 每次提交的得分取其所有子任务中的最高得分. 具体来说, 我们先为每道题采样 500 个候选解答, 再通过多阶段流程进行筛选. 第一阶段剔除未通过给定样例测试或超出长度限制的无效提交. 随后, 使用 DeepSeek-V32-Exp 模型识别并移除那些模型明确表示无法或拒绝解决问题的样本. 最后从余下的有效候选中选出思考轨迹最长的 50 个样本提交.

ICPC 评估采用相同的筛选方法, 但初始样本数较少. 每道题生成 32 个候选解答, 并用完全相同的筛选标准选择提交结果.

在 IMO 和 CMO 任务中, 我们采用生成-验证-改进循环. 模型不断改进解答, 直到自我评估达到满分或触及最大修订次数, 这一过程与 [Sha25] 完全相同.

<span id="section-10"></span>

## 10 作者名单

**研究与工程**: Aixin Liu, Aoxue Mei, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenhao Xu, Chong Ruan*, Damai Dai, Daya Guo, Dejian Yang, Deli Chen, Erhang Li, Fangqi Zhou*, Fangyun Lin, Fucong Dai, Guangbo Hao, Guanting Chen, Guowei Li, H. Zhang, Hanwei Xu, Hao Li, Haofen Liang, Haoran Wei, Haowei Zhang, Haowen Luo, Haozhe Ji, Honghui Ding, Hongxuan Tang, Huanqi Cao, Huazuo Gao, Hui Qu, Hui Zeng, Jialiang Huang, Jiashi Li, Jiaxin Xu, Jiewen Hu, Jingchang Chen, Jingting Xiang, Jingyang Yuan, Jingyuan Cheng, Jinhua Zhu, Jun Ran*, Junguang Jiang, Junjie Qiu, Junlong Li*, Junxiao Song, Kai Dong, Kaige Gao, Kang Guan, Kexin Huang*, Kexing Zhou, Kezhao Huang, Kuai Yu, Lean Wang, Lecong Zhang, Lei Wang, Liang Zhao, Liangsheng Yin*, Lihua Guo, Lingxiao Luo, Linwang Ma, Litong Wang, Liyue Zhang, M.S. Di, M.Y Xu, Mingchuan Zhang, Minghua Zhang, Minghui Tang, Mingxu Zhou, Panpan Huang, Peixin Cong, Peiyi Wang, Qiancheng Wang, Qihao Zhu, Qingyang Li, Qinyu Chen, Qiushi Du, Ruiling Xu, Ruiqi Ge, Ruisong Zhang, Ruizhe Pan, Runji Wang, Runqiu Yin, Runxin Xu, Ruomeng Shen, Ruoyu Zhang, S.H. Liu, Shanghao Lu, Shangyan Zhou, Shanhuang Chen, Shaofei Cai, Shaoyuan Chen, Shengding Hu, Shengyu Liu, Shiqiang Hu, Shirong Ma, Shiyu Wang, Shuiping Yu, Shunfeng Zhou, Shuting Pan, Songyang Zhou, Tao Ni, Tao Yun, Tian Pei, Tian Ye, Tianyuan Yue, Wangding Zeng, Wen Liu, Wenfeng Liang, Wenjie Pang, Wenjing Luo, Wenjun Gao, Wentao Zhang, Xi Gao, Xiangwen Wang, Xiao Bi, Xiaodong Liu, Xiaohan Wang, Xiaokang Chen, Xiaokang Zhang, Xiaotao Nie, Xin Cheng, Xin Liu, Xin Xie, Xingchao Liu, Xingkai Yu, Xingyou Li, Xinyu Yang, Xinyuan Li*, Xu Chen, Xuecheng Su, Xuehai Pan, Xuheng Lin, Xuwei Fu, Y.Q. Wang, Yang Zhang, Yanhong Xu, Yanru Ma, Yao Li, Yao Li, Yao Zhao, Yaofeng Sun, Yaohui Wang, Yi Qian, Yi Yu, Yichao Zhang, Yifan Ding, Yifan Shi, Yiliang Xiong, Ying He, Ying Zhou, Yinmin Zhong, Yishi Piao, Yisong Wang, Yixiao Chen, Yixuan Tan, Yixuan Wei, Yiyang Ma, Yiyuan Liu, Yonglun Yang, Yongqiang Guo, Yongtong Wu, Yu Wu, Yuan Cheng, Yuan Ou, Yuanfan Xu, Yuduan Wang, Yue Gong*, Yuhan Wu, Yuheng Zou, Yukun Li, Yunfan Xiong, Yuxiang Luo, Yuxiang You, Yuxuan Liu, Yuyang Zhou, Z.F. Wu, Z.Z. Ren, Zehua Zhao, Zehui Ren, Zhangli Sha, Zhe Fu, Zhean Xu, Zhenda Xie, Zhengyan Zhang, Zhewen Hao, Zhibin Gou, Zhicheng Ma, Zhigang Yan, Zhihong Shao, Zhixian Huang, Zhiyu Wu, Zhuoshu Li, Zhuping Zhang, Zian Xu, Zihao Wang, Zihui Gu, Zijia Zhu, Zilin Li, Zipeng Zhang, Ziwei Xie, Ziyi Gao, Zizheng Pan, Zongqing Yao

**数据标注:** Bei Feng, Hui Li, J.L. Cai, Jiaqi Ni, Lei Xu, Meng Li, Ning Tian, R.J. Chen, R.L. Jin, S.S. Li, Shuang Zhou, Tianyu Sun, X.Q. Li, Xiangyue Jin, Xiaojin Shen, Xiaosha Chen, Xinnan Song, Xinyi Zhou, Y.X. Zhu, Yanping Huang, Yaohui Li, Yi Zheng, Yuchen Zhu, Yunxian Ma, Zhen Huang, Zhipeng Xu, Zhongyu Zhang

**商业与合规:** Dongjie Ji, Jian Liang, Jianzhong Guo, Jin Chen, Leyi Xia, Miaojun Wang, Mingming Li, Peng Zhang, Ruyi Chen, Shangmian Sun, Shaoqing Wu, Shengfeng Ye, T.Wang, W.L. Xiao, Wei An, Xianzu Wang, Xiaowen Sun, Xiaoxiang Wang, Ying Tang, Yukun Zha, Zekai Zhang, Zhe Ju, Zhen Zhang, Zihua Qu

作者按名字的字母顺序排列. 标有 * 的人员已经离开团队.

[+1]: MLA 的 MQA 与 MHA 模式之差见附录[第 6 节](#section-6).

[+2]: [https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference)

[+3]: [https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning](https://artificialanalysis.ai/evaluations/artificial-analysis-long-context-reasoning)

[+4]: [https://fiction.live/stories/Fiction-liveBench-April-6-2025/oQdzQvKHw8JyXbN87](https://fiction.live/stories/Fiction-liveBench-April-6-2025/oQdzQvKHw8JyXbN87)

[+5]: 我们评估的是 CMO 2025 的英文版. IMO 2025 与 CMO 2025 的题目及推理代码见: [https://github.com/deepseek-ai/DeepSeek-Math-V2](https://github.com/deepseek-ai/DeepSeek-Math-V2).
