---
title: 'AWQ'
createTime: 2026/09/07 12:00:00
permalink: /papers/awq/
---

> [Ji Lin](https://www.linji.me/), [Jiaming Tang](https://jiamingtang.me/), [Haotian Tang](https://www.mit.edu/~kentang/), [Shang Yang](https://ys-2020.github.io/), [Wei-Ming Chen](https://developer.nvidia.com/blog/author/weimingc/), [Wei-Chen Wang](https://weichenwang.me/), [Guangxuan Xiao](https://guangxuanx.com/), [Xingyu Dang](https://dangxingyu.github.io/), [Chuang Gan](https://people.csail.mit.edu/ganchuang/), [Song Han](https://songhan.mit.edu/) [+contrib]. 论文于 2023 年 6 月 1 日首次提交至 arXiv; 当前版本为 v6, 提交于 2026 年 4 月 25 日. 获 MLSys 2024 最佳论文奖. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). <a href="/paper/awq.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [TeX 源文件](https://export.arxiv.org/e-print/2306.00978). [项目代码](https://github.com/mit-han-lab/llm-awq). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

大语言模型 (LLM) 已经改变了许多 AI 应用. *端侧* LLM 变得越来越重要: 在边缘设备上本地运行 LLM 可以降低云计算成本, 也能保护用户隐私. 但模型规模极大, 硬件资源又有限, 部署面临不少困难. 我们提出 Activation-aware Weight Quantization (AWQ), 一种面向 LLM 低比特仅权重量化且对硬件友好的方法. AWQ 发现, LLM 中各个权重的重要性并不相同. 只保护 *1%* 的显著权重就能大幅降低量化误差. 识别显著权重通道时, 应参考激活分布, 而不是权重本身. 为避免硬件效率较低的混合精度量化, 我们从数学上推导出: 放大显著通道可以减小量化误差. AWQ 通过等价变换缩放显著权重通道, 从而保护这些通道. 缩放系数通过离线收集激活统计量确定. AWQ 不依赖反向传播或重建, 因而不会过拟合校准集, 并能泛化到不同领域和模态. 在多种语言建模及领域专项基准 (编程和数学) 上, AWQ 优于现有工作. 得益于更好的泛化能力, 它在*指令微调*语言模型上取得了出色的量化性能, 也首次实现了*多模态*语言模型的有效量化. 除 AWQ 外, 我们还实现了 TinyChat, 一个面向 4 比特端侧 LLM/VLM 的高效灵活推理框架. 借助算子融合和平台感知的权重打包, TinyChat 在桌面和移动 GPU 上都比 Huggingface FP16 实现快 3 倍以上. 它还让 70B 参数的 Llama-2 可以部署到移动 GPU 上.

<span id="section-1"></span>

## 1 引言

直接把大语言模型 (LLM) 部署到边缘设备上十分重要. 端侧使用无需把数据发送到云服务器, 既消除了由此产生的延迟, 又让 LLM 可以离线运行, 适合虚拟助手, 聊天机器人和自动驾驶汽车等实时应用. 维护和扩展集中式云基础设施的运营成本也能随之降低. 端侧 LLM 把敏感信息留在本地, 减少了数据泄露的可能, 因而还能提高数据安全性. LLM 建立在 Transformer 架构 [Vas17] 之上, 凭借在多种基准上的出色表现 [Bro20c, Zha22a, Tou23, Les23] 引起了广泛关注. 但庞大的模型规模导致服务成本很高. 例如, GPT-3 有 175B 参数, 以 FP16 存储时占用 350GB, 而最新的 B200 GPU 也只有 192GB 显存, 更不用说边缘设备.

<span id="figure-01"></span>

![AWQ 量化, TinyChat 推理与支持的边缘平台](./awq/figure-01.png)

**图 1.** 我们提出 **AWQ**, 一种通用的 LLM 权重量化方法. 为实现 AWQ, 我们开发了 **TinyChat**, 用于把 4 比特量化 LLM 部署到各种边缘平台, 性能比 FP16 提升 **3-4**$\times$. 特别是, 我们还制造了一台由 TinyChat 驱动的 **TinyChat computer**, 其中配有一块仅 8GB 内存, 功耗 15W 的 NVIDIA Jetson Orin Nano. 演示: [https://youtu.be/z91a8DrfgEw](https://youtu.be/z91a8DrfgEw).

LLM 的低比特权重量化可以显著减小端侧推理的内存占用, 但并不容易. 量化感知训练 (QAT) 的训练成本很高, 效率不足; 后训练量化 (PTQ) 在低比特设置下又会造成明显的精度下降. 最接近的工作是 GPTQ [Fra22], 它利用二阶信息补偿误差. 然而, GPTQ 在重建过程中可能过拟合校准集, 使分布外领域中学到的特征发生偏移 ([图 8](#figure-08)); LLM 是*通用*模型, 因而这一问题很严重.

本文提出 Activation-aware Weight Quantization (AWQ), 一种面向 LLM 的硬件友好型低比特仅权重量化方法. 我们的出发点是: LLM 中的*权重并非同等重要*. 只有一小部分 (0.1%-1%) *显著*权重; 跳过这些显著权重的量化, 可以显著降低量化损失 ([表 1](#table-01)). 为找到显著权重通道, 尽管这里做的是*仅权重*量化, 关键仍在于参考*激活*分布, 而非*权重*分布: 激活幅值较大的权重通道处理更重要的特征, 因而更显著. 为避免硬件效率较低的混合精度实现, 我们分析权重量化误差, 并推导出*放大显著通道可以减小其相对量化误差* ([公式 2](#equation-02)). 按照这一思路, 我们设计了逐通道缩放方法, 自动搜索最优缩放系数, 在量化全部权重的条件下尽量减小量化误差. AWQ 不依赖反向传播或重建, 因而不会过拟合校准集, 并能较好地保留 LLM 在各种领域和模态上的泛化能力.

为了实现 AWQ, 我们设计了 TinyChat, 一个高效推理框架, 用实际加速兑现 4 比特 LLM 在理论上的内存节省. 该框架通过即时反量化显著加速线性层. 我们还采用高效的 4 比特权重打包和算子融合, 尽量减小推理开销 (例如中间结果的 DRAM 访问和算子启动开销), 使权重量化到 4 比特后能在按字节对齐的计算机上真正带来加速.

实验表明, 在不同模型家族 (例如 LLaMA [Tou23] 和 OPT [Zha22a]), 不同模型规模及多种任务上, AWQ 都优于现有工作. 得益于更好的泛化能力, 它在*指令微调*语言模型 (例如 Vicuna) 上也有良好的量化性能, 并首次实现了*多模态*语言模型 (OpenFlamingo [Awa23]) 的有效量化. TinyChat 进一步把约 4$\times$ 的内存占用缩减转化为实际加速. 在桌面, 笔记本和移动 GPU 上, 对多种 LLM 的测试始终显示, 它平均比 Huggingface FP16 实现快 **3.2-3.3**$\times$. 它还让 Llama-2-70B 可以轻松部署到一块 64GB 内存的 NVIDIA Jetson Orin 上. 在只有 8GB 显存的 RTX 4070 笔记本 GPU 上, 13B 参数 LLM 也能以每秒 30 个 token 的交互速度运行. AWQ 已被业界和开源社区广泛采用: [HuggingFace Transformers](https://huggingface.co/docs/transformers/main_classes/quantization), [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/), [Microsfot DirectML](https://blogs.windows.com/windowsdeveloper/2024/05/24/quantization-with-directml-helps-you-scale-further-on-windows/), [Google Vertex AI](https://console.cloud.google.com/vertex-ai/publishers/meta/model-garden/llama-2-quantized), [Intel Neural Compressor](https://github.com/intel/neural-compressor), [Amazon Sagemaker](https://aws.amazon.com/blogs/machine-learning/boost-inference-performance-for-llms-with-new-amazon-sagemaker-containers/), [AMD](https://community.amd.com/t5/ai/reduce-memory-footprint-and-improve-performance-running-llms-on/ba-p/686157), [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/awq.md), [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/awq.py), [LMDeploy](https://github.com/InternLM/lmdeploy); 它还让 Falcon-180B 可以部署到[单块](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/blogs/Falcon180B-H200.md) H200 GPU 上.

<span id="section-2"></span>

## 2 相关工作

**模型量化方法.** 量化会降低深度学习模型的位精度 [Han15, Ben18, Nag19a, Wan19a, Nag20, Lin20b], 从而减小模型规模并加速推理. 量化技术大体分为两类: 量化感知训练 (QAT, 依赖反向传播更新量化权重) [Ben13, Gho21a, Nag21, Cho18a], 以及后训练量化 [Ben18, Nag19a, Nag20] (PTQ, 通常无需训练). QAT 方法很难扩展到 LLM 这样的大模型. 因此, LLM 量化通常采用 PTQ 方法.

**LLM 量化.** LLM 量化主要有两种设置: (1) W8A8 量化, 把激活和权重都量化为 INT8 [Det22, Xia23, Yao22a, Wei22b, Wei23]; (2) 低比特仅权重量化 (例如 W4A16), 只把权重量化为低比特整数 [Fra22, Det22c, She23a, Par22]. 本文关注第二种设置, 因为它既能降低硬件门槛 (所需内存更小), 又能加速 token 生成 (缓解访存受限的负载). 除直接舍入到最近值的基线 (RTN) 外, GPTQ [Fra22] 与本文最接近. 但 GPTQ 的重建过程会过拟合校准集, 可能无法保留 LLM 在其他模态和领域上的通用能力. 对某些模型 (例如 LLaMA-7B [Tou23] 和 OPT-66B [Zha22a]), 它还需要重排序技巧才能工作. 除面向通用硬件的量化方法外, SpAtten [Wan20b] 还设计了一种渐进方法, 逐步增加 softmax 计算所用的位数.

**低比特量化 LLM 的系统支持.** 低比特量化 LLM 已成为降低推理成本的常见设置, 也有一些系统可以把它转化为实际加速. GPTQ [Fra22] 为 OPT 模型提供 INT3 核函数, `GPTQ-for-LLaMA` 则借助 Triton [Til19] 把核函数支持扩展到经过重排序的 INT4 量化. FlexGen [She23a], `llama.cpp` [+llama-cpp] 和 `exllama` [+exllama] 使用逐组 INT4 量化来降低 I/O 与卸载开销. FasterTransformer 实现了仅权重逐张量量化的 FP16$\times$INT4 GEMM, 但不支持逐组量化. LUT-GEMM [Par22] 借助查找表在 GPU CUDA 核心上执行位运算. 我们同期的工作 MLC-LLM [Mlc23] 依托强大的 TVM [Che18e, Fen23] 后端, 在多种边缘 CPU 和 GPU 平台上取得了良好结果.

<span id="figure-02"></span>

![显著权重的三种量化方式](./awq/figure-02.png)

**图 2.** 我们观察到, 根据*激活分布*可以找到 LLM 中 1% 的显著权重 (中). 将显著权重保留为 FP16 能显著改善量化性能 (PPL 从左图的 43.2 降至中图的 13.0), 但混合精度格式的硬件效率不高. 我们遵循激活感知原则并提出 AWQ (右). AWQ 通过逐通道缩放保护显著权重并减小量化误差. 这里测量的是 OPT-6.7B 在 INT3-g128 量化下的困惑度.

<span id="section-3"></span>

## 3 AWQ: Activation-aware Weight Quantization

*量化*将浮点数映射为低比特整数. 它是减小 LLM 模型规模和推理成本的有效方法 [Det22, Fra22, Yao22a, Xia23]. 本节首先提出一种仅权重量化方法: 保护较"重要"的权重, 无需训练或回归即可提高精度. 随后再给出一种数据驱动方法, 搜索能减小量化误差的最优缩放系数 ([图 2](#figure-02)).

<span id="section-3-1"></span>

### 3.1 保留 1% 显著权重以改善 LLM 量化

<span id="table-01"></span>

![将不同比例的权重通道保留为 FP16 后的困惑度](./awq/table-01.png)

**表 1.** 将少量权重 (0.1%-1%) 保留为 FP16, 可使量化模型的性能明显优于直接舍入到最近值 (RTN). 只有根据*激活*分布而非*权重*分布选择重要权重并保留为 FP16 时, 这种做法才有效. 绿色标出了困惑度较好的结果. 实验采用组大小为 128 的 INT3 量化, 并测量 WikiText 困惑度 ($\downarrow$).

我们观察到, LLM 中的权重*并非同等重要*: 只有一小部分*显著*权重, 它们对 LLM 性能的影响远大于其他权重. 跳过这些显著权重的量化, 无需任何训练或回归即可弥补量化损失造成的性能下降 ([图 2b](#figure-02)). 为验证这一想法, [表 1](#table-01) 测试了跳过部分权重通道时量化 LLM 的性能. 我们测量了 INT3 量化模型的性能, 同时将一定比例的权重通道保留为 FP16. 判断权重重要性的一种常用方法是查看其幅值或 $L_2$ 范数 [Han15a, Fra18]. 但我们发现, 跳过范数较大的权重通道 (即 FP16% (based on W)) 并不能明显改善量化性能, 提升幅度与随机选择相近. 有意思的是, 根据*激活幅值*选择权重, 即使只将 0.1%-1% 的通道保留为 FP16, 也能显著提高性能. 我们推测, 幅值更大的输入特征通常更重要. 将相应权重保留为 FP16 可以保住这些特征, 从而改善模型性能.

**局限性:** 将 0.1% 的权重保留为 FP16, 可以在模型规模 (按总比特数衡量) 几乎不增加的情况下改善量化性能, 但这种混合精度数据类型会给系统实现带来困难. 我们需要一种无需实际保留 FP16 权重也能保护重要权重的方法.

<span id="section-3-2"></span>

### 3.2 通过激活感知缩放保护显著权重

我们提出另一种通过*逐通道缩放*来减小显著权重量化误差的方法, 它没有硬件效率问题.

**分析量化误差.** 首先分析仅权重量化产生的误差.

<span id="table-02"></span>

![将 1% 显著通道乘以不同缩放系数时的统计量](./awq/table-02.png)

**表 2.** 将 1% 的显著通道乘以 $s>1$ 时的统计量. 放大显著通道可将困惑度从 23.54 明显降至 11.92. 随 $s$ 增大, $\Delta$ 发生变化的比例上升, 显著通道的误差降低率也随之上升. 但最佳困惑度出现在 $s=2$; 继续增大 $s$ 会增加*非显著*通道的量化误差.

考虑一组或一块权重 $\mathbf{w}$; 线性运算可写为 $y=\mathbf{w}\mathbf{x}$, 其量化形式为 $y=Q(\mathbf{w})\mathbf{x}$. 量化函数具体定义为:

<span id="equation-01"></span>

$$
Q(\mathbf{w})=\Delta\cdot\mathrm{Round}\left(\frac{\mathbf{w}}{\Delta}\right),\quad \Delta=\frac{\max(|\mathbf{w}|)}{2^{N-1}},
$$

其中, $N$ 为量化位数, $\Delta$ 是由绝对最大值确定的量化缩放量. 现在考虑权重元素 $w\in\mathbf{w}$. 若将 $w$ 乘以 $s>1$, 再对 $x$ 进行反向缩放, 就会得到 $Q(w\cdot s)(x/s)$, 即:

<span id="equation-02"></span>

$$
Q(w\cdot s)\cdot\frac{x}{s}=\Delta'\cdot\mathrm{Round}\left(\frac{ws}{\Delta'}\right)\cdot x\cdot\frac{1}{s},
$$

其中, $\Delta'$ 是应用 $s$ 后的新量化缩放量. 我们通过实验发现: (1) $\mathrm{Round}(\cdot)$ 的期望误差 (记为 $\mathrm{RoundErr}(\cdot)$) 不变: 舍入函数把浮点数映射到整数, 误差大致在 $[0,0.5]$ 上均匀分布, 平均误差为 $0.25$; 即 $\mathrm{RoundErr}(\cdot)\sim0.25$. (2) 放大单个元素 $w$ 通常不会改变组 $\mathbf{w}$ 中的最大值, 因而 $\Delta'\approx\Delta$. (3) $\Delta$ 和 $x$ 以 FP16 表示, 本身没有量化误差. 因此, [公式 1](#equation-01) 和 [公式 2](#equation-02) 的量化误差可以写为:

<span id="equation-03"></span>

$$
\begin{aligned}
\mathrm{Err}(Q(w)x)&=\Delta\cdot\mathrm{RoundErr}\left(\frac{w}{\Delta}\right)\cdot x\\
\mathrm{Err}\left(Q(w\cdot s)\left(\frac{x}{s}\right)\right)&=\Delta'\cdot\mathrm{RoundErr}\left(\frac{ws}{\Delta'}\right)\cdot x\cdot\frac{1}{s}
\end{aligned}
$$

新误差与原误差之比为 $\frac{\Delta'}{\Delta}\cdot\frac{1}{s}$. 由于 $\Delta'\approx\Delta$ 且 $s>1$, 显著权重 $w$ 的相对误差更小.

为验证这一想法, 我们把 OPT-6.7B 模型中 1% 的显著通道乘以 $s>1$, 并在[表 2](#table-02) 中测量每一组 $\Delta$ 的变化. 放大显著通道非常有效: $s=1$ (即直接 RTN) 时的困惑度为 23.54, $s=2$ 时降至 11.92. 随 $s$ 增大, $\Delta$ 发生变化的比例总体上升, 但 $s<2$ 时仍然很小 (低于 5%); 显著通道的相对误差则继续减小. 尽管如此, 最佳 PPL 实际出现在 $s=2$. 这是因为 $s$ 过大会在 $\Delta$ 增大时提高*非显著*通道的相对误差 (非显著通道的误差会被 $\frac{\Delta'}{\Delta}$ 放大; $s=4$ 时, 21.2% 的通道上该比值大于 1), 从而损害模型的整体精度. 因此, 保护显著通道时还要考虑非显著通道的误差.

<span id="table-03"></span>

![FP16, RTN, 混合精度, 缩放与 AWQ 的困惑度对比](./awq/table-03.png)

**表 3.** AWQ 使用基于缩放的方法保护显著权重并减小量化误差. 它始终优于直接舍入到最近值 (RTN), 性能与混合精度 (1% FP16) 相当, 同时对硬件更友好. 这里采用组大小为 128 的 3 比特量化.

<span id="figure-03"></span>

![Llama-2-7B 的时延, 屋顶线与内存流量分析](./awq/figure-03.png)

**图 3.** NVIDIA RTX 4090 上 Llama-2-7B 的瓶颈分析. **左:** 在端侧 LLM 应用中, 生成阶段远慢于上下文阶段. **中:** 生成阶段受内存带宽限制, 算术强度较低. W4A16 量化可以把算术强度有效提高 4$\times$. **右:** 权重访问量比激活访问量高出多个数量级. 因此, 仅权重量化更适合端侧 LLM.

**搜索缩放系数.** 为同时考虑显著与非显著权重, 我们选择自动搜索逐输入通道的最优缩放系数, 使某一层在量化后的输出差异最小. 形式化地说, 需要优化以下目标:

<span id="equation-04"></span>

$$
\begin{aligned}
\mathbf{s}^{*}&=\mathop{\arg\min}_{\mathbf{s}}\mathcal{L}(\mathbf{s})\\
\mathcal{L}(\mathbf{s})&=\left\|Q\left(\mathbf{W}\cdot\mathrm{diag}(\mathbf{s})\right)\left(\mathrm{diag}(\mathbf{s})^{-1}\cdot\mathbf{X}\right)-\mathbf{W}\mathbf{X}\right\|
\end{aligned}
$$

这里, $Q$ 表示权重量化函数 (例如组大小为 128 的 INT3/INT4 量化), $\mathbf{W}$ 是原始 FP16 权重, $\mathbf{X}$ 是从小型校准集缓存的输入特征 (我们从预训练数据集中抽取小型校准集, 以免过拟合特定任务). $\mathbf{s}$ 是逐输入通道缩放系数; 对 $\mathbf{s}^{-1}\cdot\mathbf{X}$ 的运算通常可以融合到前一个算子中 [Wei22, Xia23]. 量化函数不可微, 所以无法直接用常规反向传播优化该问题. 有些技术采用近似梯度 [Ben13, Ess19], 但我们发现它们仍有收敛不稳定的问题.

为使过程更稳定, 我们分析影响缩放系数选择的因素, 为最优缩放系数定义一个*搜索空间*. 上一节表明, 权重通道的显著性实际由激活尺度决定 (因此称为"激活感知"). 于是, 我们采用一个非常简单的搜索空间:

<span id="equation-05"></span>

$$
\mathbf{s}=\mathbf{s}_{X}^{\alpha},\quad \alpha^{*}=\mathop{\arg\min}_{\alpha}\mathcal{L}(\mathbf{s}_{X}^{\alpha})
$$

$\mathbf{s}_{X}$ 是激活的逐通道平均幅值, 单个超参数 $\alpha$ 用于平衡对显著通道和非显著通道的保护. 在区间 $[0,1]$ 上快速网格搜索即可找到最佳 $\alpha$ ($0$ 表示不缩放; $1$ 对应搜索空间中最激进的缩放). 我们还通过权重裁剪来减小量化的 MSE 误差. [表 3](#table-03) 给出了 OPT 模型在 INT3-g128 量化下的消融实验; AWQ 始终优于直接舍入到最近值 (RTN), 性能与混合精度 (1% FP16) 相当, 同时对硬件更友好.

**优点.** 我们的方法不依赖回归 [Fra22] 或反向传播, 而许多量化感知训练方法需要这些过程. 它只测量每个通道的平均幅值, 对校准集的依赖很小, 因而不会过拟合 ([图 8](#figure-08)). 所以, 我们的方法在量化时需要的数据更少, 也能保留 LLM 在校准集分布之外的知识. 详情见[第 5.3 节](#section-5-3).

<span id="section-4"></span>

## 4 TinyChat: 将 AWQ 映射到边缘平台

AWQ 可以显著减小 LLM 的规模. 但要把 W4A16 (4 比特权重, 16 比特激活) 量化带来的理论内存节省转化为实际加速, 并不简单. SmoothQuant [Xia23] 等 W8A8 量化方法在存储和计算时保持*相同*的数据精度, 因而可以把反量化过程直接集成到计算核的尾声中. W4A16 量化则对内存访问和计算采用*不同*的数据类型. 为获得最佳性能, 反量化必须并入主计算循环, 实现难度更大. 为此, 我们提出 TinyChat, 一个轻量灵活的 AWQ 模型推理系统. 它提供 PyTorch 前端, 后端则利用设备专用指令集 (例如 CUDA/PTX, Neon 和 AVX).

<span id="section-4-1"></span>

### 4.1 AWQ 为何能加速端侧 LLM

<span id="figure-04"></span>

![4 比特权重的 SIMD 感知打包与解包](./awq/figure-04.png)

**图 4.** 面向 128 比特 SIMD 单元的 ARM NEON SIMD 感知权重打包. 原始权重先重排序, 再按位宽对齐打包; 运行时即可通过 AND 和移位运算, 使用 128 比特掩码把权重解包为字节.

为了理解边缘设备上量化 LLM 的加速机会, 我们首先分析 RTX 4090 GPU 上 LLaMA-7B [Tou23] 的时延构成. 实验采用适合边缘应用的批大小 1, 并用 NVIDIA FasterTransformer 以 FP16 实现模型.

**上下文阶段与生成阶段的时延.** 如[图 3a](#figure-03) 所示, 生成 20 个 token 需要 310ms, 而汇总包含 200 个 token 的提示只需 10ms. 因此, 生成阶段明显慢于上下文阶段, 在端侧交互应用中尤其如此.

**生成阶段受内存限制.** 为加速生成阶段, 我们在[图 3b](#figure-03) 中进行了屋顶线分析. 4090 GPU 的峰值计算吞吐为 165 TFLOPS, 内存带宽为 1TB/s. 因此, 算术强度 (FLOP 与内存访问量之比) 低于 165 的负载在 4090 GPU 上都会受到内存限制. 以 FP16 执行时, 端侧 LLM 生成阶段的算术强度约为 1. 这说明该负载确实受内存约束. 给定模型的 FLOP 数固定, 要提高峰值性能只能减少总内存流量. AWQ 将权重内存减少了 4 倍.

**权重访问主导内存流量.** 因此, 我们又在[图 3c](#figure-03) 中分别统计权重和激活的内存访问量. 显然, 端侧 LLM 的内存流量主要来自权重访问. 将模型权重量化为 4 比特整数后, 算术强度约可提高到 4 FLOPs/Byte, 对应[图 3b](#figure-03) 中 4TFLOPS 的峰值性能. 仅权重量化降低了权重位宽, 理论性能上限也随之提高, 所以 AWQ 自然采用这一设置来处理端侧 LLM 应用.

<span id="section-4-2"></span>

### 4.2 使用 TinyChat 部署 AWQ

以上分析表明, 4 比特权重量化在理论上可将峰值性能提高 4$\times$. 我们进一步设计 TinyChat 来实现这一加速. 在 GPU 上, 我们只实现注意力, 层归一化和线性投影核等必要组件. 灵活的前端便于定制, 也能快速支持新模型. 在不同 LLM 家族和 GPU 上, 采用 4 比特 AWQ 的 TinyChat 比 Huggingface FP16 实现快 3 倍以上. 在 CPU 上, 我们把整个计算图下沉为 C++, 以尽量降低开销.

**即时权重反量化.** 硬件不提供 INT4 与 FP16 之间的乘法指令, 所以量化层在矩阵计算前需要先把整数反量化为 FP16. 我们将反量化核与矩阵乘法核融合, 避免把反量化权重写入 DRAM. 矩阵-矩阵 (MM) 与矩阵-向量 (MV) 乘法核都采用这种融合.

**SIMD 感知权重打包.** 即时权重反量化减少了中间 DRAM 访问, 但成本仍然很高. 例如, 反量化*单个 4 比特权重*需要 1 次移位, 1 次按位 AND 和 1 次 FMA 缩放, 而反量化后的权重只执行 1 次 FMA 计算. 对偏好向量化指令的 SIMD CPU 来说, 这一过程尤其昂贵. 为缓解这一问题, 我们根据设备 SIMD 单元的位宽设计平台专用权重打包. [图 4](#figure-04) 展示了面向 128 比特 SIMD 寄存器的 ARM CPU 策略, 可加速最多 1.2$\times$. 每个寄存器保存 32 个 4 比特权重, 顺序为 $w_0,w_{16},w_1,w_{17},\ldots,w_{15},w_{31}$. 这种方式只需 3 条 SIMD 指令即可解包*全部 32 个权重*, 而常规打包 ($w_0,w_1,\ldots,w_{31}$) 需要为*每个权重*执行 3 条标量指令. 一般来说, 对 $2^n$ 比特 SIMD 寄存器, 相邻权重的索引相差 $1/8\times2^n$, 因为每个寄存器可容纳 $1/8\times2^n$ 个 8 比特整数. 在 GPU 上, 我们发现按照 $w_{\{0,2,4,6,1,3,5,7\}}$ 的顺序打包每 8 个权重更高效 [Kim22].

**算子融合.** 我们还广泛采用算子融合来优化端侧 LLM 推理. 对层归一化, 所有算子 (例如乘法, 除法和平方根) 都融合为一个核. 对注意力层, QKV 投影融合为一个核, 位置嵌入也即时计算. 我们还预先分配 KV 缓存, 并在注意力核内更新缓存. 对 Falcon [Pen23c] 和 StarCoder [Li23o] 这类前向传播实现效率较低的模型, 算子融合尤其有用. 4090 GPU 上每个 FP16 核的计算时间约为 0.01ms, 与 GPU 核启动开销相当. 因此, 通过算子融合减少核调用次数可以直接加速.

<span id="section-5"></span>

## 5 实验

<span id="table-04"></span>

![LLaMA 和 Llama-2 在 INT3 与 INT4 量化下的困惑度](./awq/table-04.png)

**表 4.** 在不同模型规模和不同位精度下, AWQ 都优于直接舍入到最近值 (RTN). 在 LLaMA 和 Llama-2 模型上, 它的困惑度始终优于 GPTQ (采用或不采用重排序).

<span id="table-05"></span>

![Mistral 与 Mixtral 在 4 比特和 3 比特 AWQ 下的困惑度](./awq/table-05.png)

**表 5.** Mistral-7B-Instruct-v0.2 [Jia23] 和 Mixtral-8x7B-Instruct-v0.1 [Jia24] 的 AWQ 量化结果. WikiText 上的 PPL 结果表明, AWQ 在不同模型架构上都能取得优秀的量化性能, 包括采用 GQA 的 LLM 和混合专家 (MoE) 模型.

<span id="section-5-1"></span>

### 5.1 设置

**量化.** 本文关注*仅权重逐组*量化. 以往工作 [Det22c, Fra22] 表明, 逐组量化总能改善性能与模型大小之间的权衡. 除非另有说明, 全文使用的组大小均为 128. 我们关注 INT4/INT3 量化, 因为它们基本能够保留 LLM 的性能 [Det22c]. 对 AWQ, 我们从 Pile [Gao20] 数据集中抽取一个小型校准集, 避免过拟合特定下游领域. 搜索[公式 5](#equation-05) 中的最优 $\alpha$ 时, 网格大小设为 20.

**模型.** 我们在 LLaMA [Tou23] 和 OPT [Zha22a] 家族上测试该方法. 还有 BLOOM [Les23] 等开放 LLM, 但其质量通常较差, 因而未纳入研究. 为验证方法的泛化能力, 我们还测试了指令微调模型 Vicuna [Chi23a] 和视觉语言模型 OpenFlamingo-9B [Awa23], LLaVA-13B [Liu23j].

**评估.** 依照已有文献 [Det22, Xia23, Fra22, Det22c, Yao22a], 我们主要在语言建模任务上评测量化模型 (使用 WikiText-2 [Mer16a] 评估困惑度), 因为困惑度可以稳定反映 LLM 的性能 [Det22c].

**基线.** 主要基线是直接舍入到最近值的量化 (RTN). 组大小较小 (如 128) 时, 它其实很强 [Fra22, Det22c]. 我们还与先进的 LLM 权重量化方法 GPTQ [Fra22] 比较. 对 GPTQ, 实验另加入采用"重排序"技巧的更新版本 (记作 GPTQ-Reorder 或 GPTQ-R). ZeroQuant [Yao22a], AdaRound [Nag20] 和 BRECQ [Li21a] 等技术依赖反向传播来更新量化权重, 很难扩展到大型模型; 它们也没有超过 GPTQ [Fra22], 因而未纳入研究.

<span id="section-5-2"></span>

### 5.2 评估

<span id="figure-05"></span>

![INT3 Vicuna 模型的 GPT-4 成对评估](./awq/figure-05.png)

**图 5.** 按 GPT-4 评估协议 [Chi23a] 比较 INT3-g128 量化 Vicuna 模型与其 FP16 版本. 获胜样例 (蓝色) 越多, 性能越好. AWQ 始终改善量化性能, 优于 RTN 和 GPTQ [Fra22], 说明它可以泛化到指令微调模型.

<span id="table-06"></span>

![OpenFlamingo-9B 在 COCO Captioning 上的结果](./awq/table-06.png)

**表 6.** 视觉语言模型 OpenFlamingo-9B [Awa23] 在 COCO Captioning 数据集上的量化结果. 在零样本和各种少样本设置下, Activation-aware Weight Quantization 都优于现有方法, 表明它可以泛化到不同模态和上下文学习负载. 在 INT4-g128 下, Activation-aware Weight Quantization 将 32-shot 量化性能下降从 4.57 减至 1.17, 模型大小缩小 4$\times$, 而性能损失可以忽略.

<span id="table-07"></span>

![VILA 在 11 个视觉语言基准上的结果](./awq/table-07.png)

**表 7.** VILA-7B 和 VILA-13B [Lin24c] 在 11 个视觉语言基准上的 INT4-g128 结果. AWQ 在所有基准上都保持无损性能. 由于空间有限, 基准名称采用缩写. VQA-v2 [Goy17]; GQA [Hud19]; VisWiz [Gur18]; SQA$^{\mathrm{I}}$: ScienceQA-IMG [Lu22a]; VQA$^{\mathrm{T}}$: TextVQA [Sin19]; POPE [Li23p]; MME [Fu23a]; MMB: MMBench [Liu23k]; MMB$^{\mathrm{CN}}$: MMBench-Chinese [Liu23k]; SEED: SEED-Bench [Li23n]; LLaVA$^{\mathrm{W}}$: LLaVA-Bench (In-the-Wild) [Liu23j]; MM-Vet [Yu23a].

**LLaMA 模型上的结果.** LLaMA 模型 (LLaMA [Tou23] 和 Llama-2 [Tou23a]) 的性能优于其他开放 LLM [Zha22a, Les23], 也是许多流行开源模型的基础 [Tao23, Chi23a], 因此我们以它们为主要研究对象. [表 4](#table-04) 评估了量化前后的困惑度. 在不同模型规模 (7B-70B) 和不同代际上, AWQ 始终优于直接舍入到最近值 (RTN) 和 GPTQ [Fra22] (采用或不采用重排序).

**Mistral / Mixtral 模型上的结果.** 我们还在 Mistral 和 Mixtral 上评估了 AWQ, 它们分别是很受欢迎的开源 LLM 和混合专家 (MoE) 模型 [Jia23, Jia24]. 结果表明, AWQ 在 Mistral 和 Mixtral 上都取得了更好的性能. 这说明 AWQ 对多种模型架构都有效.

<span id="figure-06"></span>

![采用 RTN 与 AWQ 的 LLaVA 视觉推理](./awq/figure-06.png)

**图 6.** LLaVA-13B [Liu23j] 的视觉推理样例. AWQ 优于直接舍入到最近值 (RTN) 的基线, 给出的答案更合理. 文本颜色表示正确或错误回答.

<span id="figure-07"></span>

![采用 RTN 与 AWQ 的 OpenFlamingo COCO 图像描述](./awq/figure-07.png)

**图 7.** 量化 OpenFlamingo-9B [Awa23] 在 COCO Captioning 数据集上的定性结果 (4-shot, INT4-g128 量化). 与直接舍入到最近值 (RTN) 的基线相比, 我们的方法显著提高了图像描述质量. 文本颜色表示正确或错误描述.

**指令微调模型的量化.** 指令微调可以显著改善模型的性能和易用性 [Wei22c, San22, Ouy22, Chu22]. 它已成为模型部署前的重要步骤. 我们在[图 5](#figure-05) 中进一步测试了流行指令微调模型 Vicuna [Chi23a]. 实验用 GPT-4 分数评估量化模型在 80 个样例问题上的表现, 并与 FP16 版本比较 [Chi23a]. 为消除顺序效应 (我们发现 GPT-4 倾向于提高第一个输入的评分), 两种顺序 (quantized-FP16, FP16-quantized) 都进行了比较, 共 160 次试验. 在 7B 和 13B 两种规模上, AWQ 都持续改善 INT3-g128 量化 Vicuna, 优于 RTN 与 GPTQ, 说明它可以泛化到指令微调模型.

<span id="table-08"></span>

![CodeLlama 与 Llama-2 在 MBPP 和 GSM8K 上的结果](./awq/table-08.png)

**表 8.** CodeLlama-7b-Instruct-hf 在 MBPP 数据集上, 以及 Llama-2 (7B/13B/70B) 在 GSM8K 数据集上的 INT4-g128 量化结果. AWQ 在编程和数学数据集上都优于现有方法, 表明它可以泛化到不同场景和评估设置. 特别是, AWQ 在 INT4-g128 配置下的性能与两个数据集上的原始 FP16 模型相当.

**多模态语言模型的量化.** 大型多模态模型 (LMM) 或视觉语言模型 (VLM) 是加入视觉输入的 LLM [Ala22, Li23m, Koh23, Dri23, Zha23e, Liu23j]. 这类模型可以根据图像或视频输入生成文本. 我们的方法不会过拟合校准集, 因而可以直接用于 VLM, 实现准确高效的量化. 我们用 OpenFlamingo-9B [Awa23] (对 [Ala22] 的开源复现) 在 COCO Captioning [Che15a] 数据集上进行实验 ([表 6](#table-06)). 实验测量不同少样本设置下 5k 个样例的平均性能. 由于语言部分占据绝大部分模型大小, 我们只量化模型的语言部分. 在零样本和各种少样本设置下, AWQ 都优于现有方法, 表明它可以泛化到不同模态和上下文学习负载. 在 INT4-g128 下, 它将 32-shot 的量化性能下降从 4.57 减至 1.17, 模型大小缩小 4$\times$, 性能损失可以忽略. 为进一步验证 AWQ 的泛化能力, 我们还在先进的多图像视觉语言模型 VILA 上评估 AWQ. [表 7](#table-07) 表明, AWQ 在 11 个视觉语言基准上实现了无损量化. [图 7](#figure-07) 还给出了一些定性图像描述结果, 用于展示我们相对 RTN 的优势. 我们的方法为 LMM/VLM 量化提供了一套开箱即用的方案. 据我们所知, 这是首项 VLM 低比特量化研究.

<span id="figure-08"></span>

![校准集大小和分布的消融实验](./awq/figure-08.png)

**图 8.** **左:** AWQ 只需小得多的校准集就能达到良好的量化性能. 与 GPTQ 相比, 它使用小 10$\times$ 的校准集也能取得更低的困惑度. **右:** 我们的方法对校准集分布更稳健. 总体而言, 校准分布与评估分布相同时效果最好 (PubMed-PubMed, Enron-Enron). 但二者分布不同时 (PubMed-Enron, Enron-PubMed), AWQ 的困惑度只增加 0.5-0.6, GPTQ 则恶化 2.3-4.9. 所有实验均采用 INT3-g128 量化的 OPT-6.7B 模型.

<span id="figure-09"></span>

![TinyChat 在桌面, 移动和笔记本 GPU 上的吞吐量](./awq/figure-09.png)

**图 9.** TinyChat 提供了一套开箱即用的方案, 把理论上的内存占用缩减转化为可测量的加速. 因此, 在 4090 (桌面 GPU) 和 Orin (移动 GPU) 上, TinyChat 最多分别比 Huggingface FP16 实现快 **3.9**$\times$ 和 **3.5**$\times$. AWQ 还让 Llama-2-13B 可以部署到只有 8GB 显存的笔记本 GPU (4070) 上.

<span id="figure-10"></span>

![TinyChat 在 Jetson Orin 与 Raspberry Pi 上的系统对比](./awq/figure-10.png)

**图 10.** 在 NVIDIA Jetson Orin 上运行 4 比特量化 Llama 模型时, TinyChat 比现有系统快 **1.2-3.0**$\times$. 它还支持多种通用和面向编程的 LLM; 相比同样支持全部这些负载的 AutoGPTQ, 加速至少为 **2.6**$\times$. TinyChat 还能在 Raspberry Pi 上直接运行, 使参数量不超过 7B 的 LLM 可以部署到资源极其有限的 IoT 设备上.

<span id="table-09"></span>

![结合 GPTQ 与 AWQ 的 INT2-g64 困惑度](./awq/table-09.png)

**表 9.** 我们的方法与 GPTQ 正交: 与 GPTQ 结合后, 它在极低比特量化 (INT2-g64) 下进一步缩小性能差距. 结果为 OPT 模型在 WikiText-2 上的困惑度.

**视觉推理结果.** [图 6](#figure-06) 进一步给出了 LLaVA-13B [Liu23j] 的一些定性视觉推理样例. 在 INT4-g128 量化下, AWQ 的回答比直接舍入到最近值 (RTN) 的基线更合理. 第一个例子中, AWQ 模型能理解这张梗图像从太空俯瞰的地球, RTN 则生成了错误描述 (红色标注).

**编程和数学任务上的结果.** 为进一步评估 AWQ 处理复杂生成任务的性能, 我们还在 MBPP [Aus21] 和 GSM8K [Cob21] 上测试 AWQ. MBPP [Aus21] 包含约 1,000 道 Python 编程题, 面向初级程序员, 涵盖编程基础和标准库功能等内容. GSM8K [Cob21] 用于支持基础数学问题问答任务, 这些问题需要多步推理. 我们把 CodeLlama-7b-Instruct-hf 和 Llama-2 量化为 INT4-g128, 并在编程和数学数据集上进行实验 ([表 8](#table-08)). AWQ 在两个数据集上都优于现有方法, 表明它可以泛化到复杂生成任务. 在 INT4-g128 配置下, AWQ 在两个数据集上的性能均与原始 FP16 模型相当.

**极低比特量化.** 为适应有限的设备内存, 我们进一步把 LLM 量化为 INT2 ([表 9](#table-09)). RTN 完全失效, AWQ 则在 GPTQ 的基础上显著改善困惑度. 我们的方法与 GPTQ 正交. 二者结合可以进一步改善 INT2 量化性能, 使这一设置更实用.

<span id="section-5-3"></span>

### 5.3 数据效率与泛化

**更高效地使用校准集数据.** 我们的方法不依赖回归或反向传播, 只测量校准集中的平均激活尺度, 因而所需校准集更小, 数据效率更高. 为验证这一点, [图 8a](#figure-08) 比较了采用 INT3-g128 量化的 OPT-6.7B 模型的困惑度. AWQ 只需小得多的校准集就能达到良好的量化性能; 与 GPTQ 相比, 它使用小 10$\times$ 的校准集也能取得更低的困惑度 (16 个序列 *v.s.* 192 个序列).

**对校准集分布稳健.** 我们的方法只测量校准集的平均激活尺度, 这一统计量在不同数据集分布间更容易泛化, 因而对校准集分布不那么敏感. [图 8b](#figure-08) 进一步测试了不同校准集分布的影响. 我们从 Pile 数据集 [Gao20] 中取出两个子集: PubMed Abstracts 和 Enron Emails [Kli04]. 每个子集都分别用作校准集, 然后在两套数据上评估量化模型 (校准集与评估集没有重叠; 评估使用 1k 个样例). 总体而言, 校准分布与评估分布相同时效果最好 (PubMed-PubMed, Enron-Enron). 但二者分布不同时 (PubMed-Enron, Enron-PubMed), AWQ 的困惑度只增加 0.5-0.6, GPTQ 则恶化 2.3-4.9. 这说明 AWQ 对校准集分布很稳健.

<span id="section-5-4"></span>

### 5.4 加速评估

<span id="table-10"></span>

![VILA 在 A100, RTX 4090 和 Jetson Orin 上的吞吐量](./awq/table-10.png)

**表 10.** TinyChat 还让先进视觉语言模型 VILA [Lin24c] 可以轻松部署到多种 GPU 平台. 借助 4 比特 AWQ 量化, TinyChat 对 VILA-7B 和 VILA-13B 的加速最多分别达到 **3.1**$\times$ 和 **2.9**$\times$.

**设置.** [图 9](#figure-09) 展示了 TinyChat 的系统加速结果. TinyChat 同时优化线性层和没有量化权重的层. 我们按照 exllama [+exllama] 的协议, 在 RTX 4090 和 Jetson Orin 上进行基准测试. 所有 LLM 的推理批大小均为 1, 提示长度固定为 4 个 token. 每次推理生成 200 个 token, 最终结果取时延中位数.

**结果.** 如[图 9a](#figure-09) 所示, 对 4090 上的三个 LLM 家族 (Llama-2, MPT 和 Falcon), TinyChat 比 Huggingface FP16 实现快 **2.7-3.9**$\times$. 对 Llama-2-7B, FP16 算子融合将推理速度从每秒 52 个 token 提高到 62 个 token. 在这个更强的 FP16 基线上, 快速量化线性核又带来 **3.1**$\times$ 的加速. Falcon-7B 的官方实现在推理时没有正确支持 KV 缓存, 因而明显慢于其他模型. 在这种情况下, 我们的 FP16 优化带来更大的 **1.6**$\times$ 加速. 在只有 8GB 显存的 4070 笔记本 GPU 上, Llama-2-13B 仍能以每秒 33 个 token 运行, 而 FP16 实现连 7B 模型也无法装入. [表 10](#table-10) 还展示了视觉语言模型 [Lin24c] 的加速结果. 在 NVIDIA Jetson Orin 上, TinyChat 对 VILA-7B 和 VILA-13B 都带来约 **3**$\times$ 的加速. 所有 AWQ 模型的前向传播都使用原生 PyTorch API 实现, 同一份代码可复用于不同 GPU 架构. 因此, TinyChat 很容易扩展.

**与其他系统比较.** [图 10](#figure-10) 将 TinyChat 与 AutoGPTQ, llama.cpp 和 exllama 等现有边缘 LLM 推理系统进行比较. 在 Orin 上, 本系统最多比 llama.cpp 快 1.7$\times$. llama.cpp 和 exllama 的适配范围较窄, 主要面向 LLaMA 和 Llama-2. TinyChat 则支持 StarCoder [Li23o], StableCode (GPT-NeoX) [Bla22], Mistral [Jia23] 和 Falcon [Pen23c] 等多种应用, 同时始终明显快于 AutoGPTQ. TinyChat 甚至让 LLM 可以部署到资源极其有限的 Raspberry Pi 4B 上, 7B 模型的速度可达每秒 0.7 个 token.

<span id="section-6"></span>

## 6 结论

本文提出 Activation-aware Weight Quantization (AWQ), 一种简单有效的低比特仅权重 LLM 压缩方法. AWQ 基于 LLM 中权重重要性不同这一观察, 通过逐通道缩放减小显著权重的量化损失. AWQ 不会过拟合校准集, 可以保留 LLM 在各种领域和模态上的通用能力. 它在语言建模上优于现有工作, 也适用于指令微调语言模型和多模态语言模型. TinyChat 系统进一步把 AWQ 的理论内存节省转化为实测加速: 在桌面和移动 GPU 上, 比 Huggingface FP16 实现快 **3.2-3.3**$\times$, 让 LLM 更容易部署到边缘设备.

## 致谢

感谢 MIT AI Hardware Program, National Science Foundation (CNS-2112562), MIT-IBM Watson AI Lab, Amazon and MIT Science Hub, Microsoft Turing Academic Program 和 Samsung 对本研究的支持.

[+contrib]: $*$: 算法共同负责人; $\dagger$: 系统共同负责人.

[+llama-cpp]: [https://github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp)

[+exllama]: [https://github.com/turboderp/exllama](https://github.com/turboderp/exllama)
