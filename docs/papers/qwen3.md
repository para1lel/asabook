---
title: 'Qwen3 Technical Report'
createTime: 2026/08/18 18:48:40
permalink: /papers/qwen3/
pageClass: paper-reading
---

> [An Yang](https://dblp.org/pid/63/10551), [Anfeng Li](https://dblp.org/pid/408/0604), [Baosong Yang](https://dblp.org/pid/203/8245), [Beichen Zhang](https://dblp.org/pid/71/9257), [Binyuan Hui](https://dblp.org/pid/246/4699), [Bo Zheng](https://dblp.org/pid/33/1610-7), [Bowen Yu](https://dblp.org/pid/95/10266-2), [Chang Gao](https://gao-xiao-bai.github.io/), [Chengen Huang](https://dblp.org/pid/330/8484), [Chenxu Lv](https://dblp.org/pid/297/4112), [Chujie Zheng](https://chujiezheng.github.io/), [Dayiheng Liu](https://dblp.org/pid/189/4488), [Fan Zhou](https://koalazf99.github.io/), [Fei Huang](https://dblp.org/pid/h/FeiHuang-5), [Feng Hu](https://dblp.org/pid/74/6975), [Hao Ge](https://dblp.org/pid/76/5849), [Haoran Wei](https://dblp.org/pid/183/9682), [Huan Lin](https://dblp.org/pid/45/3289), [Jialong Tang](https://tangjialong.github.io/), [Jian Yang](https://dblp.org/pid/y/JianYang3), [Jianhong Tu](https://dblp.org/pid/227/8305), [Jianwei Zhang](https://dblp.org/pid/144/1628-12), [Jianxin Yang](https://dblp.org/pid/242/4275), [Jiaxi Yang](https://dblp.org/pid/293/9901-4), [Jing Zhou](https://orcid.org/0009-0002-1701-9384), [Jingren Zhou](https://dblp.org/pid/84/2644-1), [Junyang Lin](https://dblp.org/pid/215/3823), [Kai Dang](https://dblp.org/pid/241/2644), [Keqin Bao](https://dblp.org/pid/331/5509), [Kexin Yang](https://dblp.org/pid/54/774-2), [Le Yu](https://dblp.org/pid/23/7122), [Lianghao Deng](https://dblp.org/pid/395/4128), [Mei Li](https://dblp.org/pid/06/1233), [Mingfeng Xue](https://dblp.org/pid/239/2887), [Mingze Li](https://dblp.org/pid/00/8348), [Pei Zhang](https://dblp.org/pid/78/5323-11), [Peng Wang](https://dblp.org/pid/95/4442-28), [Qin Zhu](https://dblp.org/pid/08/904), [Rui Men](https://dblp.org/pid/170/0093), [Ruize Gao](https://dblp.org/pid/180/4683), [Shixuan Liu](https://dblp.org/pid/152/3598), [Shuang Luo](https://scholar.google.com/citations?user=HvJiMJoAAAAJ), [Tianhao Li](https://dblp.org/pid/69/2238), [Tianyi Tang](https://steventang1998.github.io/), [Wenbiao Yin](https://dblp.org/pid/330/7482), [Xingzhang Ren](https://dblp.org/pid/218/6803), [Xinyu Wang](https://dblp.org/pid/68/1277-13), [Xinyu Zhang](https://dblp.org/pid/58/4582-17), [Xuancheng Ren](https://dblp.org/pid/202/2250), [Yang Fan](https://dblp.org/pid/81/5991), [Yang Su](https://dblp.org/pid/17/686), [Yichang Zhang](https://dblp.org/pid/165/9507), [Yinger Zhang](https://dblp.org/pid/293/6628), [Yu Wan](https://dblp.org/pid/06/6328-4), [Yuqiong Liu](https://dblp.org/pid/45/4771), [Zekun Wang](https://kugwzk.github.io/), [Zeyu Cui](https://dblp.org/pid/236/6347), [Zhenru Zhang](https://dblp.org/pid/311/4174), [Zhipeng Zhou](https://www.sciencedirect.com/science/article/pii/S0005109826001482), and [Zihan Qiu](https://dblp.org/pid/313/9471). 论文于 2025 年 5 月 14 日首次提交至 arXiv; 当前版本为 v1. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). [原始 PDF](/paper/qwen3.pdf). [DOI](https://doi.org/10.48550/arXiv.2505.09388). [TeX 源码](https://export.arxiv.org/e-print/2505.09388v1). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

本文介绍 Qwen 模型家族的最新版本 Qwen3. Qwen3 包含一系列大语言模型 (LLM), 旨在提升性能, 效率和多语言能力. Qwen3 系列同时包含稠密架构和混合专家 (MoE) 架构的模型, 参数规模从 6 亿到 2350 亿不等. Qwen3 的一项主要创新, 是将思考模式 (用于复杂的多步推理) 和非思考模式 (用于快速的上下文驱动式回答) 集成进统一框架. 因此, 用户无需再切换不同模型, 例如聊天优化模型 (如 GPT-4o) 和专用推理模型 (如 QwQ-32B), 并且可以根据用户查询或聊天模板动态切换模式. 同时, Qwen3 引入了思考预算机制, 允许用户在推理期间自适应分配计算资源, 根据任务复杂度平衡延迟和性能. 我们还利用旗舰模型的知识, 显著减少了构建较小模型所需的计算资源, 同时保证其性能极具竞争力. 实证评测表明, Qwen3 在代码生成, 数学推理和智能体任务等多种基准上取得了最先进的结果, 能够与更大的 MoE 模型和专有模型竞争. 与前代 Qwen2.5 相比, Qwen3 支持的语言和方言从 29 种扩展到 119 种, 通过改进跨语言理解与生成能力, 提高了全球用户的可用性. 为便于复现以及社区研究和开发, 所有 Qwen3 模型均以 Apache 2.0 许可证公开发布.

<span id="section-1"></span>

## 1 引言

追求通用人工智能 (AGI) 或超级人工智能 (ASI), 长期以来一直是人类的目标. GPT-4o [Gpt24], Claude 3.7 [Ant25b], Gemini 2.5 [Gem25], DeepSeek-V3 [Dee24a], Llama-4 [Lla25] 和 Qwen2.5 [Yang24] 等大型基础模型近期取得的进展, 表明我们正朝这一目标大步迈进. 这些模型在覆盖不同领域和任务的数万亿 token 数据集上训练, 将人类知识和能力有效压缩进参数. 通过强化学习优化的推理模型也在近期取得进展, 展现了基础模型改善推理时扩展并达到更高智能水平的潜力, 例如 o3 [Ope25] 和 DeepSeek-R1 [Guo25]. 尽管大多数最先进模型仍是专有模型, 开源社区的快速发展已经大幅缩小了开放权重模型与闭源模型之间的性能差距. 越来越多的顶级模型 [Lla25, Dee24a, Guo25, Yang24] 正以开源方式发布, 带动更广泛的人工智能研究和创新.

本文介绍 Qwen 基础模型家族的最新系列 Qwen3. Qwen3 是一组开放权重的大语言模型 (LLM), 在多种任务和领域中达到最先进的性能. 我们发布了稠密模型和混合专家 (MoE) 模型, 参数量从 6 亿到 2350 亿不等, 以满足不同下游应用的需求. 旗舰模型 Qwen3-235B-A22B 是一个 MoE 模型, 总参数量为 2350 亿, 每个 token 激活 220 亿参数. 这种设计兼顾高性能和高效推理.

Qwen3 引入了几项改进, 以增强功能和易用性. 首先, 它把思考模式和非思考模式这两种不同的运行模式集成到单个模型中. 用户因此无需更换模型便可切换模式, 例如不必在 Qwen2.5 和 QwQ [Qwq24] 之间切换. 开发者和用户可以借助这种灵活性, 高效调整模型行为以适应具体任务. Qwen3 还引入了思考预算, 让用户能够精细控制模型执行任务时投入的推理力度. 这项能力可以优化计算资源和性能, 根据现实应用的不同复杂度调整模型的思考行为. Qwen3 在覆盖多达 119 种语言和方言的 36 万亿 token 上完成预训练, 多语言能力也得到有效提升. 更广的语言覆盖范围, 扩大了模型在全球用例和国际应用中的部署空间. 这些改进使 Qwen3 成为一个先进的开源大语言模型家族, 能够处理不同领域和语言中的复杂任务.

Qwen3 的预训练使用约 36 万亿 token 的大规模数据集, 并在整理时兼顾语言和领域多样性. 为了高效扩充训练数据, 我们采用多模态方法: 对 Qwen2.5-VL [Bai25a] 进行微调, 从大量 PDF 文档中提取文本. 我们还用特定领域模型生成合成数据: Qwen2.5-Math [Yang24a] 负责数学内容, Qwen2.5-Coder [Hui24] 负责代码数据. 预训练采用三个阶段. 第一阶段使用约 30 万亿 token 训练模型, 建立扎实的通用知识基础. 第二阶段继续使用知识密集型数据训练, 增强模型在科学, 技术, 工程与数学 (STEM) 和代码等领域的推理能力. 第三阶段使用长上下文数据训练, 将最大上下文长度从 4,096 个 token 提高到 32,768 个 token.

为了让基础模型更好地符合人类偏好和下游应用需求, 我们采用多阶段后训练方法, 同时赋予模型思考 (推理) 和非思考模式. 前两个阶段通过长思维链 (CoT) 冷启动微调, 以及面向数学和代码任务的强化学习, 重点培养强推理能力. 后两个阶段先把带推理路径和不带推理路径的数据合并为统一数据集, 进一步微调模型, 让它有效处理两类输入; 随后应用通用领域强化学习, 改善模型在多种下游任务上的表现. 对于较小的模型, 我们采用强到弱蒸馏, 同时利用大模型的离策略和在策略知识迁移来增强其能力. 从先进教师模型进行蒸馏, 在性能和训练效率上都明显优于强化学习.

我们在覆盖多类任务和领域的一组完整基准上, 评测模型的预训练版本和后训练版本. 实验结果显示, 预训练基础模型达到了最先进的性能. 无论采用思考模式还是非思考模式, 后训练模型都能与领先的专有模型, 以及 o1, o3-mini 和 DeepSeek-V3 等大型混合专家 (MoE) 模型竞争. 这些模型在代码, 数学和智能体相关任务中表现突出. 例如, 旗舰模型 Qwen3-235B-A22B 在 AIME'24 和 AIME'25 [Aim25a] 上分别达到 85.7 和 81.5, 在 LiveCodeBench v5 [Jai24] 上达到 70.7, CodeForces 得分为 2,056, BFCL v3 [Yan24d] 得分为 70.8. Qwen3 系列中的其他模型, 相对于各自规模也表现出很强的性能. 我们还观察到, 增加思考 token 的思考预算, 会持续改善模型在不同任务上的性能.

下文将介绍模型架构设计和训练过程, 给出预训练模型与后训练模型的实验结果, 最后总结主要发现并列出未来可能的研究方向.

<span id="section-2"></span>

## 2 架构

Qwen3 系列包含 6 个稠密模型, 分别是 Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B 和 Qwen3-32B, 以及 Qwen3-30B-A3B 和 Qwen3-235B-A22B 两个 MoE 模型. 旗舰模型 Qwen3-235B-A22B 总共有 235B 参数, 其中激活 22B 参数. 下面详细介绍 Qwen3 模型的架构.

Qwen3 稠密模型的架构与 Qwen2.5 [Yang24] 相似, 使用分组查询注意力 (GQA, [Ain23]), SwiGLU [Yan17], 旋转位置嵌入 (RoPE, [Su24]), 以及采用预归一化的 RMSNorm [Jia23b]. 我们还移除了 Qwen2 [Yang24b] 使用的 QKV-bias, 并在注意力机制中引入 QK-Norm [Deh23], 以保证 Qwen3 训练稳定. [表 1](#table-01) 给出了模型架构的主要信息.

Qwen3 MoE 模型采用与 Qwen3 稠密模型相同的基础架构. [表 2](#table-02) 给出了模型架构的主要信息. 我们沿用 Qwen2.5-MoE [Yang24], 并实现细粒度专家划分 [Dai24]. Qwen3 MoE 模型共有 128 个专家, 每个 token 激活 8 个专家. 与 Qwen2.5-MoE 不同, Qwen3-MoE 的设计不包含共享专家. 我们还采用全局批次负载均衡损失 [Qiu25a], 鼓励专家分工. 这些架构和训练方面的改进, 大幅提高了模型在下游任务上的性能.

Qwen3 模型使用 Qwen 的 tokenizer [Bai23b], 它实现了词表大小为 151,669 的字节级字节对编码 (BBPE, [Bro20b, Wan20e, Sen15]).

<span id="table-01"></span>

![表 1: Qwen3 稠密模型的模型架构.](./qwen3/table-01.png)

**表 1.** Qwen3 稠密模型的模型架构.

<span id="table-02"></span>

![表 2: Qwen3 MoE 模型的模型架构.](./qwen3/table-02.png)

**表 2.** Qwen3 MoE 模型的模型架构.

<span id="section-3"></span>

## 3 预训练

本节介绍预训练数据的构建, 预训练方法的细节, 以及在标准基准上评测基础模型所得的实验结果.

<span id="section-3-1"></span>

### 3.1 预训练数据

与 Qwen2.5 [Yang24] 相比, 我们显著扩大了训练数据的规模和多样性. 具体而言, 收集的预训练 token 数量增加到两倍, 覆盖的语言数量增加到三倍. 所有 Qwen3 模型都在一个庞大而多样的数据集上训练, 数据集包含 **119 种语言和方言**, 总计 **36 万亿 token**. 其中包括代码, STEM (科学, 技术, 工程与数学), 推理任务, 书籍, 多语言文本和合成数据等不同领域的优质内容.

为了进一步扩大预训练语料库, 我们首先使用 Qwen2.5-VL [Bai25a] 识别大量类 PDF 文档中的文本. 随后用 Qwen2.5 [Yang24] 精炼识别出的文本, 改善其质量. 通过这两个步骤, 我们额外获得了数万亿优质文本 token. 我们还使用 Qwen2.5 [Yang24], Qwen2.5-Math [Yang24a] 和 Qwen2.5-Coder [Hui24] 合成不同格式的数万亿文本 token, 包括教材, 问答, 指令和代码片段, 覆盖数十个领域. 最后, 我们加入更多多语言数据并引入更多语言, 进一步扩展预训练语料库. 与 Qwen2.5 的预训练数据相比, 支持的语言数量从 29 种大幅增加到 119 种, 扩大了模型的语言覆盖范围和跨语言能力.

我们开发了一套多语言数据标注系统, 用于改善训练数据的质量和多样性. 该系统已用于大规模预训练数据集, 从教育价值, 学科, 领域和安全性等多个维度标注了超过 30 万亿 token. 这些细致的标注有助于更有效地筛选和组合数据. 以往研究 [Xie23, Fan23, Liu24q] 在数据源或领域层面优化数据混合, 我们的方法则利用细粒度数据标签, 在小型代理模型上开展大量消融实验, 从实例层面优化数据混合.

<span id="section-3-2"></span>

### 3.2 预训练阶段

Qwen3 模型的预训练分为三个阶段:

1. **通用阶段 (S1)**: 在第一个预训练阶段, 所有 Qwen3 模型使用超过 30 万亿 token 训练, 序列长度为 4,096 个 token. 在这一阶段, 模型已经就语言能力和通用世界知识完成充分的预训练, 训练数据覆盖 119 种语言和方言.

1. **推理阶段 (S2)**: 为进一步提高推理能力, 我们增加 STEM, 代码, 推理和合成数据的占比, 优化本阶段的预训练语料库. 模型继续使用约 5T 个质量更高的 token 预训练, 序列长度为 4,096 个 token. 本阶段还加快了学习率衰减.

1. **长上下文阶段**: 在最后一个预训练阶段, 我们收集优质长上下文语料, 扩展 Qwen3 模型的上下文长度. 所有模型都使用数千亿 token 预训练, 序列长度为 32,768 个 token. 长上下文语料中, 75% 的文本长度介于 16,384 到 32,768 个 token, 25% 介于 4,096 到 16,384 个 token. 按照 Qwen2.5 [Yang24] 的做法, 我们使用 ABF 技术 [Xio23], 将 RoPE 的基频从 10,000 提高到 1,000,000. 同时引入 YARN [Pen23] 和双块注意力 (DCA, [An24]), 使推理时的序列长度容量扩大四倍.

与 Qwen2.5 [Yang24] 类似, 我们针对上述三个预训练阶段建立缩放定律, 预测最优超参数, 例如学习率调度器和批次大小. 我们通过大量实验, 系统研究模型架构, 训练数据, 训练阶段与最优训练超参数之间的关系. 最后, 为每个稠密或 MoE 模型设定预测得到的最优学习率和批次大小策略.

<span id="section-3-3"></span>

### 3.3 预训练评测

我们对 Qwen3 系列的基础语言模型进行了全面评测. 基础模型评测主要考察通用知识, 推理, 数学, 科学知识, 代码和多语言能力. 预训练基础模型使用的评测数据集包含 15 个基准:

- **通用任务**: MMLU [Hen20] (5-shot), MMLU-Pro [Wan24c] (5-shot, CoT), MMLU-redux [Gem24a] (5-shot), BBH [Suz22] (3-shot, CoT), SuperGPQA [Du25a] (5-shot, CoT).

- **数学与 STEM 任务**: GPQA [Rei24] (5-shot, CoT), GSM8K [Cob21] (4-shot, CoT), MATH [Hen21] (4-shot, CoT).

- **代码任务**: EvalPlus [Liu24i] (0-shot) (HumanEval [Che21], MBPP [Aus21], Humaneval+ 和 MBPP+ 的平均值) [Liu24i], MultiPL-E [Cas23] (0-shot) (Python, C++, JAVA, PHP, TypeScript, C#, Bash, JavaScript), MBPP-3shot [Aus21], CRUXEval 的 CRUX-O (1-shot) [Gu24].

- **多语言任务**: MGSM [Shi23] (8-shot, CoT), MMMLU [Ope24c] (5-shot), INCLUDE [Rom24] (5-shot).

在基础模型的基线对比中, 我们按照参数规模, 将 Qwen3 系列基础模型与 Qwen2.5 基础模型 [Yang24], 以及其他领先的开源基础模型进行比较, 包括 DeepSeek-V3 Base [Dee24a], Gemma-3 [Gem25a], Llama-3 [Dub24] 和 Llama-4 [Lla25] 系列基础模型. 所有模型都使用相同的评测流水线和广泛采用的评测设置, 以保证比较公平.

**评测结果总结** 根据整体评测结果, 我们总结了 Qwen3 基础模型的几个主要结论.

1. 与此前最先进的开源稠密和 MoE 基础模型 (如 DeepSeek-V3 Base, Llama-4-Maverick Base 和 Qwen2.5-72B-Base) 相比, Qwen3-235B-A22B-Base 使用的总参数或激活参数明显更少, 却在大多数任务上超过了这些模型.

1. 对 Qwen3 MoE 基础模型的实验结果表明: (a) 使用相同的预训练数据时, Qwen3 MoE 基础模型只需 **1/5** 的激活参数, 就能达到与 Qwen3 稠密基础模型相近的性能. (b) 得益于 Qwen3 MoE 架构的改进, 训练 token 规模的扩大和更先进的训练策略, Qwen3 MoE 基础模型使用不到 **1/2** 的激活参数和更少的总参数, 就能超过 Qwen2.5 MoE 基础模型. (c) 即使激活参数只有 Qwen2.5 稠密基础模型的 **1/10**, Qwen3 MoE 基础模型仍能达到相近的性能, 因而在推理和训练成本上具有明显优势.

1. Qwen3 稠密基础模型的整体性能, 可与更高参数规模的 Qwen2.5 基础模型相比. 例如, Qwen3-1.7B/4B/8B/14B/32B-Base 分别达到了与 Qwen2.5-3B/7B/14B/32B/72B-Base 相近的性能. 特别是在 STEM, 代码和推理基准上, Qwen3 稠密基础模型甚至超过了参数规模更大的 Qwen2.5 基础模型.

详细结果如下.

<span id="table-03"></span>

![表 3: Qwen3-235B-A22B-Base 与其他有代表性的强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-03.png)

**表 3.** **Qwen3-235B-A22B-Base 与其他有代表性的强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

**Qwen3-235B-A22B-Base** 我们将 Qwen3-235B-A22B-Base 与此前规模相近的 MoE 模型 Qwen2.5-Plus-Base [Yang24], 以及其他领先的开源基础模型进行比较: Llama-4-Maverick [Lla25], Qwen2.5-72B-Base [Yang24] 和 DeepSeek-V3 Base [Dee24a]. [表 3](#table-03) 的结果表明, Qwen3-235B-A22B-Base 在大多数评测基准上取得了最高分. 我们还分别将 Qwen3-235B-A22B-Base 与其他基线比较, 进行详细分析.

1. 与最近开源的 Llama-4-Maverick-Base 相比, 后者的参数量约为其 **两倍**, 但 Qwen3-235B-A22B-Base 在大多数基准上依然更好.

1. 与此前最先进的开源模型 DeepSeek-V3-Base 相比, Qwen3-235B-A22B-Base 仅使用约 **1/3** 的总参数和 **2/3** 的激活参数, 就在 15 个评测基准中的 14 个超过了 DeepSeek-V3-Base, 体现了模型强大的性能和成本效益.

1. 与此前规模相近的 MoE 模型 Qwen2.5-Plus 相比, Qwen3-235B-A22B-Base 使用更少的总参数和激活参数, 性能却明显更强, 说明 Qwen3 在预训练数据, 训练策略和模型架构上具有显著优势.

1. 与此前的旗舰开源稠密模型 Qwen2.5-72B-Base 相比, Qwen3-235B-A22B-Base 在所有基准上都超过了前者, 激活参数还不到其 **1/3**. 同时, 得益于模型架构, Qwen3-235B-A22B-Base 每万亿 token 的推理和训练成本都远低于 Qwen2.5-72B-Base.

<span id="table-04"></span>

![表 4: Qwen3-32B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-04.png)

**表 4.** **Qwen3-32B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-05"></span>

![表 5: Qwen3-14B-Base, Qwen3-30B-A3B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-05.png)

**表 5.** **Qwen3-14B-Base, Qwen3-30B-A3B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-06"></span>

![表 6: Qwen8B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-06.png)

**表 6.** **Qwen8B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-07"></span>

![表 7: Qwen3-4B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-07.png)

**表 7.** **Qwen3-4B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-08"></span>

![表 8: Qwen3-1.7B-Base, Qwen3-0.6B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-08.png)

**表 8.** **Qwen3-1.7B-Base, Qwen3-0.6B-Base 与其他强开源基线的比较. 最高分和次高分分别以粗体和下划线标出.**

**Qwen3-32B-Base** Qwen3-32B-Base 是 Qwen3 系列中最大的稠密模型. 我们将它与规模相近的基线比较, 包括 Gemma-3-27B [Gem25a] 和 Qwen2.5-32B [Yang24]. 此外还引入了两个强基线: 最近开源的 MoE 模型 Llama-4-Scout, 其参数量是 Qwen3-32B-Base 的三倍, 但激活参数只有一半; 以及此前的旗舰开源稠密模型 Qwen2.5-72B-Base, 其参数量是 Qwen3-32B-Base 的两倍以上. [表 4](#table-04) 展示了结果, 并支持以下三个主要结论:

1. 与规模相近的模型相比, Qwen3-32B-Base 在大多数基准上超过 Qwen2.5-32B-Base 和 Gemma-3-27B Base. Qwen3-32B-Base 在 MMLU-Pro 和 SuperGPQA 上分别达到 65.54 和 39.78, 明显超过前代 Qwen2.5-32B-Base. 此外, Qwen3-32B-Base 在编码基准上的得分明显高于所有基线模型.

1. 出人意料的是, Qwen3-32B-Base 与 Qwen2.5-72B-Base 相比也取得了很有竞争力的结果. Qwen3-32B-Base 的参数量不到 Qwen2.5-72B-Base 的一半, 却在 15 个评测基准中的 10 个超过了后者. 在代码, 数学和推理基准上, Qwen3-32B-Base 优势明显.

1. 与 Llama-4-Scout-Base 相比, Qwen3-32B-Base 的参数量只有其三分之一, 激活参数量则是两倍, 并在全部 15 个基准上明显超过它.

**Qwen3-14B-Base 与 Qwen3-30B-A3B-Base** 我们将 Qwen3-14B-Base 和 Qwen3-30B-A3B-Base 与规模相近的基线比较, 包括 Gemma-3-12B Base 和 Qwen2.5-14B Base. 同样也引入两个强基线: (1) Qwen2.5-Turbo [Yang24], 它有 42B 参数, 激活 6B 参数. 请注意, 它的激活参数量是 Qwen3-30B-A3B-Base 的两倍. (2) Qwen2.5-32B-Base, 其激活参数量是 Qwen3-30B-A3B 的 11 倍, 总参数量是 Qwen3-14B 的两倍以上. [表 5](#table-05) 展示了结果, 可以得出以下结论.

1. 与规模相近的模型相比, Qwen3-14B-Base 在全部 15 个基准上都明显优于 Qwen2.5-14B-Base 和 Gemma-3-12B-Base.

1. 同样, Qwen3-14B-Base 使用不到一半的参数, 与 Qwen2.5-32B-Base 相比也取得了很有竞争力的结果.

1. Qwen3-30B-A3B 只使用 **1/5** 的激活非嵌入参数, 就在所有任务上明显超过 Qwen2.5-14B-Base, 并达到与 Qwen3-14B-Base 和 Qwen2.5-32B-Base 相近的性能, 从而在推理和训练成本上获得明显优势.

**Qwen3-8B / 4B / 1.7B / 0.6B-Base** 对于端侧模型, 我们以规模相近的 Qwen2.5, Llama-3 和 Gemma-3 基础模型为基线. 结果见 [表 6](#table-06), [表 7](#table-07) 和 [表 8](#table-08). Qwen3 8B / 4B / 1.7B / 0.6B-Base 模型在几乎所有基准上仍然保持强劲表现. Qwen3-8B / 4B / 1.7B-Base 甚至在半数以上的基准中超过了规模更大的 Qwen2.5-14B / 7B / 3B Base 模型, 尤其是在 STEM 相关基准和代码基准上, 反映出 Qwen3 模型的显著进步.

<span id="section-4"></span>

## 4 后训练

<span id="figure-01"></span>

![图 1: Qwen3 系列模型的后训练流水线.](./qwen3/figure-01.png)

**图 1.** Qwen3 系列模型的后训练流水线.

Qwen3 后训练流水线围绕两个主要目标设计:

1. **思考控制**: 将非思考模式和思考模式这两种不同的模式集成起来, 让用户可以选择模型是否进行推理, 并通过指定思考过程的 token 预算控制思考深度.

1. **强到弱蒸馏**: 简化并优化轻量模型的后训练过程. 我们利用大规模模型中的知识, 大幅降低构建小规模模型所需的计算成本和开发工作量.

如 [图 1](#figure-01) 所示, Qwen3 系列的旗舰模型采用一套精细的四阶段训练流程. 前两个阶段重点培养模型的思考能力. 后两个阶段旨在把强大的非思考功能集成进模型.

初步实验表明, 将教师模型的输出 logits 直接蒸馏进轻量学生模型, 可以有效增强性能, 同时精细控制其推理过程. 采用这种方法, 无需为每个小规模模型单独执行完整的四阶段训练流程. 从更高的 Pass@1 得分来看, 它能直接带来更好的性能; Pass@64 结果的改善也表明, 模型的探索能力有所提高. 此外, 该方法的训练效率高得多, 所需 GPU 小时只有四阶段训练方法的 1/10.

下文将介绍四阶段训练流程, 并详细说明强到弱蒸馏方法.

<span id="section-4-1"></span>

### 4.1 长 CoT 冷启动

我们首先整理一个综合数据集, 覆盖数学, 代码, 逻辑推理和一般 STEM 问题等多类内容. 数据集中的每个问题, 都配有经过验证的参考答案或基于代码的测试用例. 该数据集是长思维链 (long-CoT) 训练冷启动阶段的基础.

数据集构建采用严格的两阶段筛选过程: 查询筛选和回答筛选. 在查询筛选阶段, 我们用 Qwen2.5-72B-Instruct 识别并移除不易验证的查询. 其中包括含多个子问题的查询, 以及要求生成一般文本的查询. 如果 Qwen2.5-72B-Instruct 不使用 CoT 推理也能正确回答某个查询, 我们也会将其排除. 这样可以避免模型依赖表面猜测, 确保只纳入需要深入推理的复杂问题. 我们还使用 Qwen2.5-72B-Instruct 标注每个查询所属的领域, 保持数据集中的领域分布均衡.

预留验证查询集后, 我们使用 QwQ-32B [Qwq25] 为每个剩余查询生成 $N$ 个候选回答. 如果 QwQ-32B 始终无法生成正确解答, 则由人工标注员手动评估回答的准确性. 对于 Pass@$N$ 为正的查询, 我们应用更严格的筛选条件, 移除以下回答: (1) 最终答案错误; (2) 含有大量重复; (3) 明显是在没有充分推理的情况下猜测; (4) 思考内容与总结内容不一致; (5) 存在不当的语言混用或风格变化; (6) 疑似与潜在验证集条目过于相似. 随后, 从精炼后的数据集中仔细选择一个子集, 用于推理模式的初始冷启动训练. 此阶段的目标是在模型中建立基础推理模式, 而不是过分追求眼前的推理性能. 这种做法不会限制模型潜力, 为后续强化学习 (RL) 阶段保留更大的灵活性和改进空间. 为有效实现这一目标, 最好尽量减少准备阶段的训练样本和训练步数.

<span id="section-4-2"></span>

### 4.2 推理 RL

推理 RL 阶段使用的查询与验证器对, 必须满足以下四项标准: (1) 未在冷启动阶段使用; (2) 冷启动模型能够学会; (3) 尽可能有挑战性; (4) 覆盖广泛的子领域. 我们最终收集了 3,995 个查询与验证器对, 并使用 GRPO [Sha24] 更新模型参数. 我们观察到, 使用大批次和每个查询对应的大量 rollout, 再配合离策略训练提高样本效率, 有利于训练过程. 我们还通过控制模型熵稳定上升或保持稳定, 处理探索与利用之间的平衡; 这对于维持训练稳定十分重要. 因此, 在单次 RL 运行过程中, 训练奖励和验证性能都持续改善, 无需人工干预超参数. 例如, Qwen3-235B-A22B 模型的 AIME'24 得分, 在总计 170 个 RL 训练步骤中从 70.1 提高到 85.1.

<span id="section-4-3"></span>

### 4.3 思考模式融合

思考模式融合阶段的目标, 是把非思考能力集成进此前开发的思考模型. 这种方法让开发者能够管理和控制推理行为, 同时降低为思考任务和非思考任务分别部署模型的成本与复杂度. 为此, 我们在推理 RL 模型上进行持续监督微调 (SFT), 并设计聊天模板来融合两种模式. 我们还发现, 能熟练处理两种模式的模型, 在不同思考预算下都能保持稳定表现.

**SFT 数据的构建.** SFT 数据集同时包含思考数据和非思考数据. 为了不让额外 SFT 损害阶段 2 模型的性能, 思考数据由阶段 2 模型自身在阶段 1 查询上进行拒绝采样生成. 非思考数据则经过仔细整理, 覆盖代码, 数学, 指令遵循, 多语言任务, 创意写作, 问答和角色扮演等多类任务. 我们还使用自动生成的检查清单, 评估非思考数据的回答质量. 为提高低资源语言任务的性能, 特别增加了翻译任务的比例.

**聊天模板设计.** 为了更好地集成两种模式, 让用户动态切换模型的思考过程, 我们为 Qwen3 设计了 [表 9](#table-09) 所示的聊天模板. 具体而言, 对于思考模式和非思考模式的样本, 我们分别在用户查询或系统消息中引入 `/think` 和 `/no_think` 标记. 模型因此可以遵循用户输入, 选择合适的思考模式. 对于非思考模式样本, 我们在助手回答中保留空的思考块. 这种设计保证模型内部格式一致, 开发者也可以在聊天模板中拼接空思考块, 阻止模型进入思考行为. 模型默认运行在思考模式, 因此我们加入了一些用户查询中不含 `/think` 标记的思考模式训练样本. 对于更复杂的多轮对话, 我们在用户查询中随机插入多个 `/think` 和 `/no_think` 标记, 模型回答遵从最后遇到的标记.

**思考预算.** 思考模式融合还有一项优势: 模型学会以非思考和思考两种模式回答后, 会自然获得处理中间情况的能力, 即根据未完成的思考生成回答. 这项能力为控制模型的思考预算打下了基础. 具体而言, 当模型的思考长度达到用户设定的阈值时, 我们手动停止思考过程, 并插入停止思考指令: “`Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n`”. 插入这条指令后, 模型根据此前累积的推理生成最终回答. 需要说明的是, 这项能力没有经过显式训练, 而是在应用思考模式融合后自然出现的.

<span id="table-09"></span>

![表 9: 思考模式融合阶段, 思考模式和非思考模式的 SFT 数据示例. 在思考模式下, /think 标记可以省略, 因为它表示默认行为. 该功能已经在 Hugging Face tokenizer 支持的聊天模板中实现, 可以通过附加参数 enable_thinking=False 禁用思考模式.](./qwen3/table-09.png)

**表 9.** **思考模式融合阶段, 思考模式和非思考模式的 SFT 数据示例.** 在思考模式下, `/think` 标记可以省略, 因为它表示默认行为. 该功能已经在 Hugging Face tokenizer 支持的聊天模板 [+1] 中实现, 可以通过附加参数 `enable_thinking=False` 禁用思考模式.

<span id="section-4-4"></span>

### 4.4 通用 RL

通用 RL 阶段旨在广泛增强模型在不同场景下的能力和稳定性. 为此, 我们建立了一套精细的**奖励系统**, 覆盖**超过 20 个不同任务**, 每项任务都有定制的评分标准. 这些任务专门针对以下核心能力:

- **指令遵循**: 这项能力保证模型准确理解并遵循用户指令, 包括内容, 格式, 长度和结构化输出等要求, 使回答符合用户预期.

- **格式遵循**: 除了明确的指令, 我们希望模型遵守特定格式约定. 例如, 模型应根据 `/think` 和 `/no_think` 标记切换思考与非思考模式, 并始终使用指定 token (如 `<think>` 和 `</think>`) 分隔最终输出中的思考部分和回答部分.

- **偏好对齐**: 对于开放式查询, 偏好对齐侧重改善模型回答的帮助性, 吸引力和风格, 最终提供更自然, 更令人满意的用户体验.

- **智能体能力**: 训练模型通过指定接口正确调用工具. 在 RL rollout 期间, 模型可以根据真实环境的执行反馈, 完成完整的多轮交互循环, 从而提高长时域决策任务中的性能和稳定性.

- **特定场景能力**: 对于更为专门的场景, 我们针对具体上下文设计任务. 例如, 在检索增强生成 (RAG) 任务中, 我们加入奖励信号, 引导模型生成准确且符合上下文的回答, 尽量降低幻觉风险.

我们使用三类不同的奖励, 为上述任务提供反馈:

1. **基于规则的奖励**: 基于规则的奖励已广泛用于推理 RL 阶段, 对指令遵循 [Lam24] 和格式遵守等通用任务也很有用. 设计良好的规则奖励可以高精度评估模型输出是否正确, 避免奖励黑客等问题.

1. **带参考答案的模型奖励**: 在这种方法中, 我们为每个查询提供参考答案, 并提示 Qwen2.5-72B-Instruct 根据参考答案为模型回答评分. 该方法不要求严格格式, 可以更灵活地处理不同任务, 避免纯规则奖励可能产生的假阴性.

1. **不带参考答案的模型奖励**: 我们利用人类偏好数据训练奖励模型, 为模型回答给出标量分数. 这种方法不依赖参考答案, 可以处理范围更广的查询, 同时有效增强模型回答的吸引力和帮助性.

<span id="section-4-5"></span>

### 4.5 强到弱蒸馏

强到弱蒸馏流水线专为优化轻量模型而设计, 包括 5 个稠密模型 (Qwen3-0.6B, 1.7B, 4B, 8B 和 14B) 和一个 MoE 模型 (Qwen3-30B-A3B). 这种方法在增强模型性能的同时, 还能有效传授稳定的模式切换能力. 蒸馏过程分为两个主要阶段:

1. **离策略蒸馏**: 在初始阶段, 我们合并教师模型以 `/think` 和 `/no_think` 两种模式生成的输出, 进行回答蒸馏. 这有助于轻量学生模型掌握基本推理技能和在不同思考模式间切换的能力, 为后续在策略训练阶段打下坚实基础.

1. **在策略蒸馏**: 在这一阶段, 学生模型生成用于微调的在策略序列. 具体而言, 系统抽样提示, 再由学生模型以 `/think` 或 `/no_think` 模式生成回答. 随后, 将学生模型的 logits 与教师模型 (Qwen3-32B 或 Qwen3-235B-A22B) 的 logits 对齐, 通过最小化 KL 散度微调学生模型.

<span id="section-4-6"></span>

### 4.6 后训练评测

为了全面评估指令微调模型的质量, 我们采用自动基准, 分别评测模型在思考模式和非思考模式下的性能. 这些基准分为以下几个维度:

- **通用任务**: 使用 MMLU-Redux [Gem24a], GPQA-Diamond [Rei24], C-Eval [Hua23] 和 LiveBench (2024-11-25) [Whi24] 等基准. 对于 GPQA-Diamond, 每个查询抽样 10 次并报告平均准确率.

- **对齐任务**: 为评估模型与人类偏好的一致程度, 我们使用一组专门的基准. 指令遵循性能报告 IFEval [Zho23a] 的 strict-prompt 准确率. 对于一般主题上的人类偏好对齐, 使用 Arena-Hard [Li24j] 和 AlignBench v1.1 [Liu23m]. 写作任务则使用 Creative Writing V3 [Pae24] 和 WritingBench [Wu25b], 评估模型的熟练程度和创造力.

- **数学与文本推理**: 为评估数学和逻辑推理能力, 我们使用高难度数学基准 MATH-500 [Lig23], AIME'24 和 AIME'25 [Aim25a], 以及文本推理任务 ZebraLogic [Lin25] 和 AutoLogi [Zhu25b]. 每一年的 AIME 题目包括 Part I 和 Part II, 总计 30 题. 每道题抽样 64 次, 将平均准确率作为最终得分.

- **智能体与代码**: 为测试模型在代码和智能体任务上的熟练程度, 我们使用 BFCL v3 [Yan24d], LiveCodeBench (v5, 2024.10-2025.02) [Jai24], 以及 CodeElo [Qua25] 的 Codeforces Ratings. 对于 BFCL, 所有 Qwen3 模型都以 FC 格式评测, 并使用 yarn 将模型部署到 64k 上下文长度, 用于多轮评测. 部分基线来自 BFCL 排行榜, 取 FC 格式和 Prompt 格式中的较高分. 对于排行榜没有报告的模型, 使用 Prompt 格式评测. 对于 LiveCodeBench, 非思考模式使用官方推荐的提示; 思考模式则调整提示模板, 移除 `You will not return anything except for the program` 限制, 让模型更自由地思考. 为评估模型与竞赛编程专家之间的性能差距, 我们使用 CodeForces 计算 Elo 等级分. 在我们的基准中, 每道题最多生成 8 次独立推理尝试来求解.

- **多语言任务**: 在多语言能力方面, 我们评测四类任务: 指令遵循, 知识, 数学和逻辑推理. 指令遵循使用 Multi-IF [He24b] 评估, 该基准关注 8 种主要语言. 知识评测分为两类: INCLUDE [Rom24] 覆盖 44 种语言, 用于评测区域知识; MMMLU [Ope24c] 覆盖 14 种语言, 用于评测通用知识, 其中排除了尚未优化的约鲁巴语. 为提高评测效率, 这两个基准都只抽样原始数据的 10%. 数学任务使用 MT-AIME2024 [Son25] 和 PolyMath [Wan25g], 前者覆盖 55 种语言, 后者覆盖 18 种语言. 逻辑推理使用来自 [Zha24k] 的 MlogiQA 评测, 覆盖 10 种语言.

<span id="table-10"></span>

![表 10: 多语言基准及其包含的语言. 语言以 IETF 语言标签标识.](./qwen3/table-10.png)

**表 10.** **多语言基准及其包含的语言.** 语言以 IETF 语言标签标识.

<span id="table-11"></span>

![表 11: Qwen3-235B-A22B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-11.png)

**表 11.** **Qwen3-235B-A22B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-12"></span>

![表 12: Qwen3-235B-A22B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-12.png)

**表 12.** **Qwen3-235B-A22B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

对于所有采用思考模式的 Qwen3 模型, 抽样温度设为 0.6, top-p 设为 0.95, top-k 设为 20. 对于 Creative Writing v3 和 WritingBench, 还应用 1.5 的 presence penalty, 以鼓励生成更多样的内容. 对于采用非思考模式的 Qwen3 模型, 抽样超参数配置为 temperature = 0.7, top-p = 0.8, top-k = 20, presence penalty = 1.5. 两种模式的最大输出长度均设为 32,768 个 token, 只有 AIME'24 和 AIME'25 例外, 其长度扩展到 38,912 个 token, 以提供充足的思考空间.

**评测结果总结** 根据评测结果, 我们对最终版 Qwen3 模型总结出以下几个主要结论:

1. 旗舰模型 Qwen3-235B-A22B 在思考和非思考两种模式下, 都取得了开源模型中最先进的整体性能, 超过 DeepSeek-R1 和 DeepSeek-V3 等强基线. Qwen3-235B-A22B 与 OpenAI-o1, Gemini2.5-Pro 和 GPT-4o 等领先闭源模型相比也很有竞争力, 展现了强大的推理能力和全面的通用能力.

1. 旗舰稠密模型 Qwen3-32B 在大多数基准上超过了此前最强的推理模型 QwQ-32B, 与闭源 OpenAI-o3-mini 表现相近, 体现出很强的推理能力. Qwen3-32B 在非思考模式下同样表现出色, 超过了此前的旗舰非推理稠密模型 Qwen2.5-72B-Instruct.

1. Qwen3-30B-A3B, Qwen3-14B 和其他更小的稠密模型等轻量模型, 始终优于参数量相近或更大的开源模型, 证明强到弱蒸馏方法取得了成功.

详细结果如下.

**Qwen3-235B-A22B** 对于旗舰模型 Qwen3-235B-A22B, 我们将它与领先的推理模型和非推理模型进行比较. 在思考模式下, 以 OpenAI-o1 [Ope24], DeepSeek-R1 [Guo25], Grok-3-Beta (Think) [Gro25] 和 Gemini2.5-Pro [Gem25] 为推理基线. 在非思考模式下, 以 GPT-4o-2024-11-20 [Gpt24], DeepSeek-V3 [Dee24a], Qwen2.5-72B-Instruct [Yang24] 和 LLaMA-4-Maverick [Lla25] 为非推理基线. [表 11](#table-11) 和 [表 12](#table-12) 给出了评测结果.

1. 根据 [表 11](#table-11), Qwen3-235B-A22B (思考) 的激活参数和总参数分别只有 DeepSeek-R1 的 60% 和 35%, 却在 **17/23** 个基准上超过后者, 尤其是在数学, 智能体和代码等需要推理的任务上, 表明 Qwen3-235B-A22B 在开源模型中具有最先进的推理能力. Qwen3-235B-A22B (思考) 与闭源 OpenAI-o1, Grok-3-Beta (Think) 和 Gemini2.5-Pro 相比也很有竞争力, 大幅缩小了开源模型与闭源模型之间的推理能力差距.

1. 根据 [表 12](#table-12), Qwen3-235B-A22B (非思考) 超过了 DeepSeek-V3, LLaMA-4-Maverick 和此前旗舰模型 Qwen2.5-72B-Instruct 等其他领先开源模型, 还在 **18/23** 个基准上超过闭源 GPT-4o-2024-11-20, 说明即使没有审慎思考过程的增强, 它本身也具有很强的能力.

**Qwen3-32B** 对于旗舰稠密模型 Qwen3-32B, 思考模式下的基线包括 DeepSeek-R1-Distill-Llama-70B, OpenAI-o3-mini (medium), 以及此前最强的推理模型 QwQ-32B [Qwq25]. 非思考模式下的基线包括 GPT-4o-mini-2024-07-18, LLaMA-4-Scout, 以及此前的旗舰模型 Qwen2.5-72B-Instruct. [表 13](#table-13) 和 [表 14](#table-14) 给出了评测结果.

1. 根据 [表 13](#table-13), Qwen3-32B (思考) 在 **17/23** 个基准上超过 QwQ-32B, 成为 32B 这一理想规模上新的最先进推理模型. Qwen3-32B (思考) 还能与闭源 OpenAI-o3-mini (medium) 竞争, 并具有更好的对齐和多语言性能.

1. 根据 [表 14](#table-14), Qwen3-32B (非思考) 在几乎所有基准上都优于全部基线. 特别是, Qwen3-32B (非思考) 在通用任务上与 Qwen2.5-72B-Instruct 表现相当, 在对齐, 多语言和推理相关任务上明显占优, 再次证明 Qwen3 相对先前 Qwen2.5 系列模型有根本性的改进.

<span id="table-13"></span>

![表 13: Qwen3-32B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-13.png)

**表 13.** **Qwen3-32B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-14"></span>

![表 14: Qwen3-32B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-14.png)

**表 14.** **Qwen3-32B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-15"></span>

![表 15: Qwen3-30B-A3B / Qwen3-14B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-15.png)

**表 15.** **Qwen3-30B-A3B / Qwen3-14B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-16"></span>

![表 16: Qwen3-30B-A3B / Qwen3-14B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-16.png)

**表 16.** **Qwen3-30B-A3B / Qwen3-14B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-17"></span>

![表 17: Qwen3-8B / Qwen3-4B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-17.png)

**表 17.** **Qwen3-8B / Qwen3-4B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-18"></span>

![表 18: Qwen3-8B / Qwen3-4B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-18.png)

**表 18.** **Qwen3-8B / Qwen3-4B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-19"></span>

![表 19: Qwen3-1.7B / Qwen3-0.6B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-19.png)

**表 19.** **Qwen3-1.7B / Qwen3-0.6B (思考) 与其他推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

<span id="table-20"></span>

![表 20: Qwen3-1.7B / Qwen3-0.6B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-20.png)

**表 20.** **Qwen3-1.7B / Qwen3-0.6B (非思考) 与其他非推理基线的比较. 最高分和次高分分别以粗体和下划线标出.**

**Qwen3-30B-A3B 与 Qwen3-14B** 对于 Qwen3-30B-A3B 和 Qwen3-14B, 思考模式下分别与 DeepSeek-R1-Distill-Qwen-32B 和 QwQ-32B 比较, 非思考模式下分别与 Phi-4 [Abd24a], Gemma-3-27B-IT [Gem25a] 和 Qwen2.5-32B-Instruct 比较. [表 15](#table-15) 和 [表 16](#table-16) 给出了评测结果.

1. 根据 [表 15](#table-15), Qwen3-30B-A3B 和 Qwen3-14B (思考) 与 QwQ-32B 相比都很有竞争力, 尤其是在推理相关基准上. Qwen3-30B-A3B 使用更小的模型规模和不到 **1/10** 的激活参数, 就达到了与 QwQ-32B 相近的性能. 这说明强到弱蒸馏可以有效赋予轻量模型深厚的推理能力.

1. 根据 [表 16](#table-16), Qwen3-30B-A3B 和 Qwen3-14B (非思考) 在大多数基准上超过了非推理基线. 它们使用的激活参数和总参数明显更少, 却超过了先前的 Qwen2.5-32B-Instruct, 因而能够以更高的效率和更低的成本实现良好性能.

**Qwen3-8B / 4B / 1.7B / 0.6B** 对于 Qwen3-8B 和 Qwen3-4B, 思考模式下分别与 DeepSeek-R1-Distill-Qwen-14B 和 DeepSeek-R1-Distill-Qwen-32B 比较, 非思考模式下分别与 LLaMA-3.1-8B-Instruct [Dub24], Gemma-3-12B-IT [Gem25a], Qwen2.5-7B-Instruct 和 Qwen2.5-14B-Instruct 比较. 对于 Qwen3-1.7B 和 Qwen3-0.6B, 思考模式下分别与 DeepSeek-R1-Distill-Qwen-1.5B 和 DeepSeek-R1-Distill-Llama-8B 比较, 非思考模式下分别与 Gemma-3-1B-IT, Phi-4-mini, Qwen2.5-1.5B-Instruct 和 Qwen2.5-3B-Instruct 比较. [表 17](#table-17) 和 [表 18](#table-18) 给出了 Qwen3-8B 和 Qwen3-4B 的评测结果, [表 19](#table-19) 和 [表 20](#table-20) 则分别给出了 Qwen3-1.7B 和 Qwen3-0.6B 的结果. 总体来看, 这些端侧模型表现出色, 在思考或非思考模式下, 甚至能超过参数量更大的基线, 包括先前的 Qwen2.5 模型. 这些结果再次证明了强到弱蒸馏方法的有效性, 使我们能以显著降低的成本和工作量构建轻量 Qwen3 模型.

<span id="section-4-7"></span>

### 4.7 讨论

**思考预算的有效性** 为验证 Qwen3 能否利用更多思考预算提高智能水平, 我们在数学, 代码和 STEM 领域的 4 个基准上调整分配的思考预算. [图 2](#figure-02) 给出了所得的缩放曲线, Qwen3 的性能会随思考预算增加而平稳提升. 我们还观察到, 如果将输出长度进一步扩展到 32K 以上, 模型性能有望在未来继续提高. 我们把这项探索留作未来工作.

<span id="figure-02"></span>

![图 2: Qwen3-235B-A22B 在不同思考预算下的性能.](./qwen3/figure-02.png)

**图 2.** Qwen3-235B-A22B 在不同思考预算下的性能.

**在策略蒸馏的有效性和效率** 我们从同一个经过离策略蒸馏的 8B 检查点出发, 分别进行蒸馏和直接强化学习, 再比较性能与计算成本 (以 GPU 小时计), 以评估在策略蒸馏的有效性和效率. 为简化比较, 我们只关注数学和代码相关查询. [表 21](#table-21) 汇总的结果表明, 蒸馏的性能明显优于强化学习, 所需 GPU 小时约为后者的 $1/10$. 教师 logits 蒸馏还让学生模型能够扩大探索空间并增强推理潜力; 与初始检查点相比, 蒸馏后 AIME'24 和 AIME'25 基准上的 pass@64 得分有所提高, 证明了这一点. 相比之下, 强化学习没有改善 pass@64 得分. 这些观察结果说明, 用更强的教师模型指导学生模型学习具有优势.

<span id="table-21"></span>

![表 21: 在 Qwen3-8B 上比较强化学习与在策略蒸馏. 括号中的数字表示 pass@64 得分.](./qwen3/table-21.png)

**表 21.** 在 Qwen3-8B 上比较强化学习与在策略蒸馏. 括号中的数字表示 pass@64 得分.

**思考模式融合和通用 RL 的效果** 为评估后训练过程中思考模式融合和通用强化学习 (RL) 的有效性, 我们对 Qwen-32B 模型的不同阶段进行评测. 除了前文提到的数据集, 还引入几项内部基准来监测其他能力. 这些基准包括:

- **CounterFactQA**: 包含反事实问题, 模型需要识别这些问题不符合事实, 并避免生成幻觉回答.

- **LengthCtrl**: 包含带长度要求的创意写作任务; 最终得分依据生成内容长度与目标长度之间的差异计算.

- **ThinkFollow**: 包含随机插入 `/think` 和 `/no_think` 标记的多轮对话, 用于测试模型能否根据用户查询正确切换思考模式.

- **ToolUse**: 评估模型在单轮, 多轮和多步工具调用过程中的稳定性. 得分包括意图识别准确率, 格式准确率, 以及工具调用过程中的参数准确率.

<span id="table-22"></span>

![表 22: Qwen3-32B 经过推理 RL (阶段 2), 思考模式融合 (阶段 3) 和通用 RL (阶段 4) 后的性能. 带 * 的基准是内部数据集.](./qwen3/table-22.png)

**表 22.** Qwen3-32B 经过推理 RL (阶段 2), 思考模式融合 (阶段 3) 和通用 RL (阶段 4) 后的性能. 带 * 的基准是内部数据集.

[表 22](#table-22) 展示了结果, 可以得出以下结论:

1. 阶段 3 把非思考模式集成进模型, 该模型经过前两个训练阶段后, 已经具备思考能力. ThinkFollow 基准得分为 88.7, 说明模型初步获得了切换模式的能力, 但偶尔仍会出错. 阶段 3 还增强了模型在思考模式下的通用能力和指令遵循能力, CounterFactQA 和 LengthCtrl 分别提高了 10.9 分和 8.0 分.

1. 阶段 4 进一步增强模型在思考和非思考模式下的通用能力, 指令遵循能力和智能体能力. ThinkFollow 得分提高到 98.9, 可以保证准确切换模式.

1. 对于知识, STEM, 数学和代码任务, 思考模式融合和通用 RL 没有带来明显改善. 相反, 对 AIME'24 和 LiveCodeBench 等高难度任务, 思考模式下的性能在这两个训练阶段后反而下降. 我们推测, 性能下降是因为模型在范围更广的通用任务上训练, 可能损害了处理复杂问题的专门能力. 在开发 Qwen3 时, 我们选择接受这种性能权衡, 以增强模型的整体通用性.

<span id="section-5"></span>

## 5 结论

本技术报告介绍 Qwen 系列的最新版本 Qwen3. Qwen3 同时具有思考模式和非思考模式, 用户可以动态管理复杂思考任务使用的 token 数量. 模型在包含 36 万亿 token 的庞大数据集上预训练, 能够理解和生成 119 种语言和方言的文本. 一系列全面评测表明, 无论预训练模型还是后训练模型, Qwen3 在代码生成, 数学, 推理和智能体相关任务等多种标准基准上都表现强劲.

近期研究将集中在几个方面. 我们会继续扩大预训练规模, 使用质量更高, 内容更多样的数据. 同时, 为实现有效压缩和扩展到极长上下文等目标, 我们会改进模型架构和训练方法. 此外, 我们计划增加强化学习的计算资源, 尤其关注从环境反馈中学习的智能体 RL 系统. 这将帮助我们构建能够处理复杂任务的智能体, 这些任务需要进行推理时扩展.

<span id="section-6"></span>

## 6 作者

**核心贡献者:** An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu

**贡献者:** Bei Chen, Biao Sun, Bin Luo, Bin Zhang, Binghai Wang, Bowen Ping, Boyi Deng, Chang Si, Chaojie Yang, Chen Cheng, Chenfei Wu, Chengpeng Li, Chengyuan Li, Fan Hong, Guobin Zhao, Hang Zhang, Hangrui Hu, Hanyu Zhao, Hao Lin, Hao Xiang, Haoyan Huang, Hongkun Hao, Humen Zhong, Jialin Wang, Jiandong Jiang, Jianqiang Wan, Jianyuan Zeng, Jiawei Chen, Jie Zhang, Jin Xu, Jinkai Wang, Jinyang Zhang, Jinzheng He, Jun Tang, Kai Zhang, Ke Yi, Keming Lu, Keqin Chen, Langshi Chen, Le Jiang, Lei Zhang, Linjuan Wu, Man Yuan, Mingkun Yang, Minmin Sun, Mouxiang Chen, Na Ni, Nuo Chen, Peng Liu, Peng Wang, Peng Zhu, Pengcheng Zhang, Pengfei Wang, Qiaoyu Tang, Qing Fu, Qiuyue Wang, Rong Zhang, Rui Hu, Runji Lin, Shen Huang, Shuai Bai, Shutong Jiang, Sibo Song, Siqi Zhang, Song Chen, Tao He, Ting He, Tingfeng Hui, Wei Ding, Wei Liao, Wei Lin, Wei Zhang, Weijia Xu, Wenbin Ge, Wenmeng Zhou, Wenyuan Yu, Xianyan Jia, Xianzhong Shi, Xiaodong Deng, Xiaoming Huang, Xiaoyuan Li, Ximing Zhou, Xinyao Niu, Xipin Wei, Xuejing Liu, Yang Liu, Yang Yao, Yang Zhang, Yanpeng Li, Yantao Liu, Yidan Zhang, Yikai Zhu, Yiming Wang, Yiwen Hu, Yong Jiang, Yong Li, Yongan Yue, Yu Guan, Yuanzhi Zhu, Yunfei Chu, Yunlong Feng, Yuxin Zhou, Yuxuan Cai, Zeyao Ma, Zhaohai Li, Zheng Li, Zhengyang Tang, Zheren Fu, Zhi Li, Zhibo Yang, Zhifang Guo, Zhipeng Zhang, Zhiying Xu, Zhiyu Yin, Zhongshen Zeng, Zile Qiao, Ziye Meng, Zongmeng Zhang

<span id="section-a"></span>

## A 附录

<span id="section-a-1"></span>

### A.1 额外评测结果

<span id="section-a-1-1"></span>

#### A.1.1 长上下文能力

<span id="table-23"></span>

![表 23: Qwen3 模型在 RULER 基准上的性能.](./qwen3/table-23.png)

**表 23.** **Qwen3 模型在 RULER 基准上的性能.**

为评估长上下文处理能力, 我们在 [表 23](#table-23) 中报告 RULER 基准 [Hsi24a] 的结果. 为了实现长度外推, 我们使用 YARN [Pen23], 并设置 `scaling_factor=4`. 在思考模式下, 思考预算设为 8192 个 token, 以减少对极长输入的过度冗长推理.

结果表明:

1. 在非思考模式下, Qwen3 在长上下文处理任务上超过规模相近的 Qwen2.5 模型.

1. 在思考模式下, 模型性能略有下降. 我们推测, 思考内容没有为这些检索任务带来明显收益; 这些任务不依赖推理, 思考反而可能干扰检索过程. 我们会在未来版本中继续增强思考模式的长上下文能力.

<span id="section-a-1-2"></span>

#### A.1.2 多语言能力

[表 24](#table-24)-[表 35](#table-35) 给出了不同语言上的详细基准得分, 包括西班牙语, 法语, 葡萄牙语, 意大利语, 阿拉伯语, 日语, 韩语, 印度尼西亚语, 俄语, 越南语, 德语和泰语. 这些表格的结果表明, Qwen3 系列模型在所有评测基准上都具有竞争力, 展现出很强的多语言能力.

为了在范围更广的语言中评测 Qwen3 的性能, 我们使用自然语言理解基准 Belebele [Ban23]. 如 [表 36](#table-36) 所示, 我们在该基准支持的 80 种语言上评测, 排除了 42 种尚未优化的语言; 表中按语系组织语言. [表 37](#table-37) 给出了 Qwen3 与其他基线模型在 Belebele 基准上的性能比较. 结果表明, Qwen3 达到了与规模相近的 Gemma 模型相当的性能, 同时明显超过 Qwen2.5.

<span id="table-24"></span>

![表 24: 西班牙语 (es) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-24.png)

**表 24.** **西班牙语 (es) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-25"></span>

![表 25: 法语 (fr) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-25.png)

**表 25.** **法语 (fr) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-26"></span>

![表 26: 葡萄牙语 (pt) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-26.png)

**表 26.** **葡萄牙语 (pt) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-27"></span>

![表 27: 意大利语 (it) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-27.png)

**表 27.** **意大利语 (it) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-28"></span>

![表 28: 阿拉伯语 (ar) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-28.png)

**表 28.** **阿拉伯语 (ar) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-29"></span>

![表 29: 日语 (ja) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-29.png)

**表 29.** **日语 (ja) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-30"></span>

![表 30: 韩语 (ko) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-30.png)

**表 30.** **韩语 (ko) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-31"></span>

![表 31: 印度尼西亚语 (id) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-31.png)

**表 31.** **印度尼西亚语 (id) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-32"></span>

![表 32: 俄语 (ru) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-32.png)

**表 32.** **俄语 (ru) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-33"></span>

![表 33: 越南语 (vi) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-33.png)

**表 33.** **越南语 (vi) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-34"></span>

![表 34: 德语 (de) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-34.png)

**表 34.** **德语 (de) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-35"></span>

![表 35: 泰语 (th) 的基准得分. 最高分和次高分分别以粗体和下划线标出.](./qwen3/table-35.png)

**表 35.** **泰语 (th) 的基准得分.** 最高分和次高分分别以粗体和下划线标出.

<span id="table-36"></span>

![表 36: Qwen3 在 Belebele 基准上支持的语系和语言代码.](./qwen3/table-36.png)

**表 36.** Qwen3 在 Belebele 基准上支持的语系和语言代码.

<span id="table-37"></span>

![表 37: Qwen3 与其他基线模型在 Belebele 基准上的性能比较. 最高分以粗体标出, 次高分以下划线标出.](./qwen3/table-37.png)

**表 37.** **Qwen3 与其他基线模型在 Belebele 基准上的性能比较.** 最高分以粗体标出, 次高分以下划线标出.

[+1]: [https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json)
