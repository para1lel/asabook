---
title: 'Kimi K3: Open Frontier Intelligence'
createTime: 2026/07/29 15:00:00
permalink: /papers/kimi-k3/
---

> [Kimi Team](https://platform.kimi.com/docs/guide/kimi-k3-quickstart). Kimi K3: Open Frontier Intelligence: Technical Report of Kimi K3. 本网页阅读版依据[原始 PDF](/paper/k3_tech_report.pdf) 提取的文本整理. 数学符号, 图表及完整参考文献列表以 PDF 为准. [模型权重](https://huggingface.co/moonshotai/Kimi-K3).

## 摘要

我们提出 Kimi K3, 这是一个总参数量为 2.8T, 激活参数量为 104B 的混合专家模型, 具备原生视觉能力和 100 万 token 的上下文窗口. Kimi K3 基于 Kimi Delta Attention [Kim25b] 和 Attention Residuals [Kim26a] 构建, 二者分别改善了沿序列长度和模型深度方向的信息流. 结合每个 token 从 896 个路由专家中有效激活 16 个的 Stable LatentMoE, 以及改进后的训练与数据方案, 这些进展使整体扩展效率相较 Kimi K2 [Kim25] 提升约 $2.5\times$. 后训练的重点是在通用, 智能体和编码领域, 以及多个推理强度级别上开展强化学习, 从而实现组合泛化和稳健的长程执行. 在 2.8T 参数规模下, Kimi K3 得到了多方面基础设施进展的支持, 包括 KDA 的算法-系统协同设计, 配合高效内存管理的完全均衡专家并行训练, 保留 rollout 与沙箱状态的百万 token 智能体强化学习, 以及部署创新. 大量评估表明, Kimi K3 在长程编码, 智能体, 知识, 推理和视觉任务上达到前沿水平. 尽管其整体性能仍落后于最强的闭源模型 Claude Fable 5 和 GPT-5.6 Sol, 但 Kimi K3 始终优于评测套件中的其他开源与闭源模型. 我们发布 Kimi K3 的完整模型权重, 以推动后续研究, 并加快前沿智能的广泛部署与应用.

<span id="figure-01"></span>

![Kimi K3 基准测试结果](./kimi-k3/figure-01.png)

**图 1.** Kimi K3 的主要结果.

## 1 引言

在大语言模型 (LLM) 发展的大部分时间里, 扩展意味着在部署前投入更多计算, 即使用更多数据训练更大的模型 [Kap20, Hof22]. 推理模型的兴起确立了测试时计算这一扩展的第二轴线: OpenAI 的 o 系列扩展强化学习和测试时推理 [Ope24, Ope25]; Anthropic 的扩展思考模型分配自适应思考预算, 并将推理与工具使用交错进行 [Ant25, Int25]; DeepSeek-R1 [Guo25] 和 Kimi K1.5 [Sca25] 表明, 大规模强化学习能够从强大的预训练模型中激发复杂的推理行为; Kimi K2.5 Agent Swarm [Kim26b] 则进一步将测试时扩展从顺序推理拓展到并行智能体协作. 这些进展使测试时扩展成为前沿研究的核心方向. 然而, 开源模型生态虽然在第二条轴线上发展迅速, 在第一条轴线上的进展却较为缓慢: 近期许多模型的参数量仍停留在 1T 级或略高于这一水平 [Zen26, Dee26, Mim26, Thi26]. 当日益复杂的推理和智能体强化学习方法被应用于规模相近的预训练基础模型时, 开源模型的发展可能趋于收敛, 而与最强闭源系统的差距则会扩大. 借助 Kimi K3, 我们同时将两条扩展轴线推向前沿: 一方面将预训练基础模型扩展到前所未有的 3T 级参数量, 另一方面在 1M 上下文长度下扩展强化学习, 推理强度和长程交互.

我们提出 Kimi K3, 一个拥有 2.8T 总参数, 104B 激活参数和最长 100 万 token 上下文窗口的原生多模态混合专家模型. 其架构沿序列长度, 网络深度和模型宽度三个方向扩展信息流. Kimi Delta Attention (KDA) [Kim25b] 提供高效的长序列混合, 周期性交错的 Gated MLA 层则保留全局交互. Attention Residuals (AttnRes) [Kim26a] 使每一层都能选择性关注所有先前层的表示. Stable LatentMoE 将路由专家空间扩展到 896 个专家, 每个 token 激活其中 16 个, 同时通过归一化, SiTU-GLU 和 Quantile Balancing 在极高稀疏度下稳定优化过程. 这些架构改进与优化后的数据和训练方案相结合, 使整体扩展效率相较 Kimi K2 [Kim25] 提升约 $2.5\times$.

在这一预训练基础上, 我们配套设计了面向 1M 上下文测试时扩展的后训练方案. Kimi K3 在长程编码, 通用智能体, 通用推理和知识任务上接受强化学习, 每类任务又覆盖多个推理强度级别. 训练环境包括可验证搜索和专业知识工作, 软件工程和 kernel 优化, 在工具使用闭环中引入视觉的多模态推理, 持久化助理工作流, Web 开发与自主执行任务. 这些环境训练了一个通用的推理, 行动, 观察, 验证和适应循环, 其中往往涉及数百乃至数千次工具调用, 累积上下文可达数百万 token. 最后通过多教师在策略蒸馏 [Lu25, Xia26, Dee26], 将针对不同领域和推理强度专门训练的策略整合为统一模型.

实现这种训练范式需要基础设施随架构复杂度, 模型规模和轨迹长度一同扩展. 在 KDA 的系统协同设计方面, 我们开发了融合 kernel, KDA 上下文并行和状态感知前缀缓存, 使 KDA 在设备内, 设备间和请求间都能高效运行. 对于 2.8T 参数 MoE 的预训练, MoonEP 以静态计算形状和零拷贝通信实现完全均衡的专家执行, 同时通过内存高效训练与多模态编码器优化, 在有限内存中维持硬件利用率. 对于百万 token 的智能体强化学习, 我们的共置系统结合部分 rollout, 外部 KV cache 保留, 自适应限流和可恢复的 microVM 沙箱, 以保存长时间存在的模型与环境状态. 最后, 专用 kernel 以及感知缓存和预算的集群调度, 将这些创新转化为可预测的生产服务能力.

由此得到的模型树立了新的开源前沿. 在涵盖长程编码, 智能体, 知识, 推理和视觉任务的基准测试中, Kimi K3 的整体表现落后于最强的闭源系统 Claude Fable 5 和 GPT-5.6 Sol, 但始终领先于本报告评估的其他开源和闭源模型.

**贡献.** Kimi K3 将总参数量为 2.8T, 激活参数量为 104B, 上下文窗口为 1M token 的原生多模态 MoE 模型, 跨领域和推理强度级别的强化学习, 支持数万亿参数预训练和百万 token 智能体轨迹的基础设施, 以及完整模型权重的发布融为一体.

## 2 模型架构

Kimi K3 架构旨在沿三个互补维度扩展信息流: 序列长度, 网络深度和模型宽度. 在序列维度上, 混合注意力在每个块中组合三个 Kimi Delta Attention (KDA) [Kim25b] 层和一个 Gated MLA 层, 在保留选择性高容量注意力的同时, 为长上下文 token 混合提供高效机制 ([§2.1](#_2-1-混合注意力)). 在深度维度上, Attention Residuals (AttnRes) [Kim26a] 使每个模块能够从嵌入, 当前块和先前块中选择性检索表示, 将信息访问范围扩展到传统顺序残差累积之外 ([§2.2](#_2-2-attention-residuals)). 在宽度维度上, 每个注意力层之后都有一个执行稀疏通道混合的 Stable LatentMoE 层, 每个 token 从 896 个路由专家中有效激活 16 个 ([§2.3](#_2-3-stable-latentmoe)). 在原生视觉方面, MoonViT-V2 对图像和视频进行编码, 轻量级投影器在主干网络处理前将所得视觉特征映射到共享嵌入空间 ([§2.4](#_2-4-原生视觉)). 这些组件与 Per-Head Muon ([§2.5](#_2-5-per-head-muon)) 一同构成统一架构, 用于扩展 token, 层和通道间的信息流. 再结合改进后的训练与数据方案, Kimi K3 的整体扩展效率相较 Kimi K2 提升约 $2.5\times$. [图 2](#figure-02) 展示了该架构的整体结构.

<span id="figure-02"></span>

![Kimi K3 架构](./kimi-k3/figure-02.png)

**图 2.** Kimi K3 架构围绕 token, 通道和层混合组织, 并在输入端设有原生视觉路径. 每个块包含三个 Kimi Delta Attention (KDA) 层和随后一个 Gated MLA 层, 每个注意力层均与一个 Stable LatentMoE 前馈网络配对. Attention Residuals (AttnRes) 使用学习到的伪查询, 根据嵌入和先前块的输出计算注意力权重, 从而实现沿深度方向的选择性信息流.

### 2.1 混合注意力

Kimi K3 在层级上混合线性注意力与全局注意力, 将 KDA [Kim25b] 和 Gated MLA 结合起来. 每个块包含 3 个 KDA 层, 随后是 1 个 Gated MLA 层, 混合比例为 3:1. 这一模式在整个主干网络中重复. 下面分别介绍两种注意力机制. 主干网络末端还额外放置一个 Gated MLA 层, 确保最后一层始终执行全局注意力.

#### 2.1.1 Kimi Delta Attention

KDA 使用逐通道遗忘门扩展了 delta rule 递归 [Sch21, Yan25, Kim25b]. 考虑一组隐藏状态序列 $\mathbf x_t\in\mathbb R^d$, 其中 $t$ 表示 token 位置, $d$ 是模型的隐藏维度. 为清晰起见, 我们先描述单个注意力头, 其查询和键向量为 $\mathbf q_t,\mathbf k_t\in\mathbb R^{d_k}$, 值向量为 $\mathbf v_t\in\mathbb R^{d_v}$, 递归状态为 $\mathbf S_t\in\mathbb R^{d_k\times d_v}$. KDA 在执行 delta rule 更新前应用逐通道衰减:

$$
\mathbf S_t=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)
\operatorname{Diag}(\boldsymbol\alpha_t)\mathbf S_{t-1}
+\beta_t\mathbf k_t\mathbf v_t^\top,
\qquad
\widetilde{\mathbf o}_t=\mathbf S_t^\top\mathbf q_t.
\tag{1}
$$

其中, $\boldsymbol\alpha_t\in(0,1)^{d_k}$ 是逐通道的单步保留因子, $\beta_t\in(0,1)$ 控制 delta rule 的写入强度.

沿用 Kimi Linear [Kim25b] 的设计, KDA 将每个注意力头中的各个量参数化为

$$
\begin{aligned}
\mathbf q_t^h,\mathbf k_t^h
  &=\operatorname{L_2Norm}\!\left(\operatorname{Swish}\!\left(\operatorname{ShortConv}(\mathbf W_{q/k}^h\mathbf x_t)\right)\right)\in\mathbb R^{d_k},\\
\mathbf v_t^h
  &=\operatorname{Swish}\!\left(\operatorname{ShortConv}(\mathbf W_v^h\mathbf x_t)\right)\in\mathbb R^{d_v},\\
\beta_t^h&=\operatorname{Sigmoid}(\mathbf W_\beta^h\mathbf x_t)\in(0,1),\\
\mathbf z_t^h&=\mathbf W_\alpha^\uparrow\mathbf W_\alpha^\downarrow\mathbf x_t+\mathbf b_\alpha^h\in\mathbb R^{d_k}.
\end{aligned}
\tag{2}
$$

查询, 键和值投影依次应用 ShortConv 和 Swish [Yan25], 查询和键还会进一步接受 $L_2$ 归一化 [Yan24b]. 低秩投影与注意力头特有的偏置 $\mathbf b_\alpha^h\in\mathbb R^{d_k}$ 为每个键通道生成细粒度衰减 logit $\mathbf z_t^h$. 从 $\mathbf z_t^h$ 到 $\boldsymbol\alpha_t^h$ 的有下界映射将在下面的分块形式之后介绍.

**分块并行形式.** 沿用 Kimi Linear [Kim25b] 的设计, KDA 在块间递归, 在块内并行. 设块大小为 $C$, 对于 $\mathbf X\in\{\mathbf Q,\mathbf K,\mathbf V,\mathbf O,\mathbf U,\mathbf W\}$, $\mathbf X_{[t]}$ 将第 $t$ 个块中的 token 向量堆叠起来. 矩阵 $\mathbf S_{[t]}\in\mathbb R^{d_k\times d_v}$ 表示进入块 $t$ 时的递归状态. 对于位置 $1\le i\le j\le C$, 定义逐通道累积衰减

$$
\boldsymbol\gamma_{[t]}^{i\to j}:=\prod_{r=i}^{j}\boldsymbol\alpha_{[t]}^r,
\qquad
\boldsymbol\gamma_{[t]}^r:=\boldsymbol\gamma_{[t]}^{1\to r}.
\tag{3}
$$

与 Kimi Linear 相同, $\boldsymbol\Gamma_{[t]}^{1\to C}\in\mathbb R^{C\times d_k}$ 逐行堆叠 $\boldsymbol\gamma_{[t]}^1,\ldots,\boldsymbol\gamma_{[t]}^C$. UT 变换生成 $\mathbf U_{[t]}$ 和 $\mathbf W_{[t]}$, 据此定义伪值项 $\widetilde{\mathbf V}_{[t]}:=\mathbf U_{[t]}-\mathbf W_{[t]}\mathbf S_{[t]}$. 给定传入状态 $\mathbf S_{[t]}$, 块 $t$ 中的所有输出可按下式并行计算

$$
\begin{aligned}
\mathbf A_{[t]}&=\operatorname{Tril}\!\left[
(\mathbf Q_{[t]}\odot\boldsymbol\Gamma_{[t]}^{1\to C})
(\mathbf K_{[t]}/\boldsymbol\Gamma_{[t]}^{1\to C})^\top\right],\\
\mathbf O_{[t]}&=
\underbrace{(\boldsymbol\Gamma_{[t]}^{1\to C}\odot\mathbf Q_{[t]})\mathbf S_{[t]}}_{\text{inter-chunk}}
+\underbrace{\mathbf A_{[t]}\widetilde{\mathbf V}_{[t]}}_{\text{intra-chunk}}.
\end{aligned}
\tag{4}
$$

对于矩阵 $\mathbf M$, $\operatorname{Tril}(\mathbf M)$ 将所有严格上三角元素置零, 并保留下三角元素及对角线. 这一掩码保证块内交互满足因果关系. 对角线之所以保留, 是因为每个输出读取的都是当前 token 更新后的状态. $\mathbf O_{[t]}$ 中第一项携带来自先前块的信息, 第二项则描述当前块内的交互. 关于 UT 变换以及分块形式的完整推导, 请参阅 Kimi Linear [Kim25b].

**有下界衰减.** 式 4 使用累积衰减的倒数 $1/\boldsymbol\Gamma_{[t]}^{1\to C}$ 对每个块中的键进行重新缩放. 由于 $\boldsymbol\Gamma_{[t]}^{1\to C}$ 是若干 $(0,1)$ 区间内保留因子的乘积, 其倒数可能无限增长, 并在有限精度下溢出 [Yan24a, Kim25b]. Kimi Linear 在 log 空间中计算相对衰减, 并将每个块进一步划分为 16-token tile, 以控制数值范围 [Yan24a, Kim25b]. 此时, 非对角 tile 可以直接使用 Tensor Core 上的稠密矩阵乘法计算. 相比之下, 对角 tile 仍需显式计算位置对, 这仍是主要的块内瓶颈. Kimi K3 改变了从衰减 logit $\mathbf z_t^h$ 到单步 log 衰减 $\mathbf g_t^h$ 的映射, 以解决这一瓶颈. 沿用 GDN 和 Mamba-2 的设计, Kimi Linear 使用负 Softplus 映射 $\mathbf g_t^h=-e^{A_h}\operatorname{Softplus}(\mathbf z_t^h)\in(-\infty,0)^{d_k}$ [Yan25, Dao24, Kim25b]. Kimi K3 则使用缩放 sigmoid 为 log 衰减设置下界:

$$
\mathbf g_t^h=g_{\min}\operatorname{Sigmoid}(e^{A_h}\mathbf z_t^h)\in(g_{\min},0)^{d_k},
\qquad
\boldsymbol\alpha_t^h=\exp(\mathbf g_t^h)\in(e^{g_{\min}},1)^{d_k}.
\tag{5}
$$

其中, $A_h$ 是每个注意力头可学习的 log 缩放因子, $g_{\min}=-5$ 为固定值. 我们将 $A_h$ 初始化为 0, 并按照 [Kim25b, Dao24, Yan25] 初始化各偏置 $\mathbf b_\alpha^h$. 当 $g_{\min}=-5$ 时, 每个保留因子均满足 $\alpha_{t,j}^h>e^{-5}\approx6.7\times10^{-3}$, 16-token tile 上的累积 log 衰减位于 $(-80,0)$ 内. 因此, 对应的倒数缩放因子小于 $e^{80}$, 始终处于 BF16 的动态范围内. 这一有限范围使对角和非对角 tile 均可使用稠密 Tensor Core 矩阵乘法, 从而消除按位置对计算的对角路径. 该参数化与先前工作中的有下界递归门密切相关 [Qin24a, De24, Pen25]. [图 3](#figure-03) 展示了衰减参数化的变化及其对计算的影响.

<span id="figure-03"></span>

![有下界衰减与分块 KDA 计算](./kimi-k3/figure-03.png)

**图 3.** 有下界衰减及其对分块 KDA 计算的影响. Kimi Linear 使用无下界的负 Softplus 映射, Kimi K3 则以缩放 sigmoid 限制 log 衰减, 使所有因果 tile 都能使用稠密 Tensor Core 矩阵乘法.

**满秩门.** 最后, Kimi K3 将 KDA 输出门从 Kimi Linear [Kim25b] 使用的低秩参数化改为依赖输入的满秩投影. 对递归输出应用逐头 RMSNorm [Zha19] 后, KDA 使用依赖数据的输出门 [Qiu25]:

$$
\mathbf y_t=\mathbf W_o\!\left[\operatorname{Sigmoid}(\mathbf W_g\mathbf x_t)\odot
\operatorname{RMSNorm}(\widetilde{\mathbf o}_t)\right].
\tag{6}
$$

#### 2.1.2 Gated MLA

DeepSeek-V2 [Dee24] 提出的 Multi-head Latent Attention (MLA) 将每个 token 的键值表示压缩为低维潜向量 $\mathbf c_t=\mathbf W_c\mathbf x_t$. MLA 不缓存每个注意力头完整的键和值, 而是缓存 $\mathbf c_t$, 并在计算注意力时通过学习得到的上投影重建内容键和值. 这种分解在保留全局 token 间注意力的同时, 减少了 KV cache 占用. Kimi K2 和 Kimi K2.5 随后采用了 MLA [Kim25, Kim26b], Kimi K3 则在周期性的全局注意力层中保留了这一设计.

Kimi K3 与 Kimi K2 和 Kimi K2.5 不同, 它沿用 Kimi Linear [Kim25b] 的混合设计, 对所有 MLA 层采用无位置编码 (No Position Encoding, NoPE). 因此, 其查询和键都不应用显式位置编码. 位于其间的 KDA 层提供位置敏感且感知新近性的序列混合, MLA 层则提供不受限的全局内容交互. 这种分工也避免了在扩展上下文长度时修改位置编码参数, 例如重新调整 RoPE 频率基数或应用 YaRN [Pen23].

此外, Kimi K3 为 MLA 增加了一个依赖输入的逐通道满秩输出门. 设 $\widetilde{\mathbf o}_t$ 表示位置 $t$ 上未经门控的 MLA 输出, 则门控后的输出为

$$
\mathbf y_t=\mathbf W_o\!\left[\operatorname{Sigmoid}(\mathbf W_g\mathbf x_t)\odot\widetilde{\mathbf o}_t\right].
\tag{7}
$$

门投影 $\mathbf W_g$ 为满秩, 与 Kimi K3 中 KDA 使用的新参数化一致. 该门使每个 token 都能调节从全局注意力中读取的通道 [Qiu25].

为修正 flash attention 中产生的有偏舍入误差, 我们采用 [Qiu26] 的方法, 在训练期间以 FP32 保存注意力输出. 这一选择使输出 tile 的片上空间占用翻倍, 因此我们重新设计了训练 kernel, 让输出 tile 与 KV 暂存缓冲区而非查询 tile 重叠, 从而释放共享内存, 以支持更深的 KV 流水线和更高的训练吞吐量.

### 2.2 Attention Residuals

标准残差连接 [He16] 沿深度方向将所有先前信息压缩到单一状态 $\mathbf h_l$ 中, 这一瓶颈类似于 RNN 沿时间方向的递归. 在序列建模中, Transformer 用注意力取代了递归 [Bah14, Vas17], 使每个位置都能通过依赖数据的权重, 选择性访问所有先前位置. Attention Residuals (AttnRes) [Kim26a] 将同样的方法应用于深度方向: 每一层都选择性检索所有先前层的表示, 而非对其进行无差别累积.

#### 完整 Attention Residuals

对于每一层 $l$, 我们定义该层特有的可学习伪查询 $\mathbf q_l=\mathbf w_l\in\mathbb R^d$, 以及如下键和值

$$
\mathbf k_i=\mathbf v_i=
\begin{cases}
\mathbf h_1,&i=0,\\
f_i(\mathbf h_i),&1\le i\le l-1.
\end{cases}
\tag{8}
$$

其中, $f_i(\mathbf h_i)$ 是第 $i$ 层的输出, $\mathbf h_1$ 是 token 嵌入. 注意力权重采用 softmax kernel $\phi(\mathbf q,\mathbf k)=\exp(\mathbf q^\top\operatorname{RMSNorm}(\mathbf k))$ [Kat20, Zha19], 其中 RMSNorm 可防止输出幅度较大的层主导权重:

$$
\alpha_{i\to l}=\frac{\phi(\mathbf q_l,\mathbf k_i)}
{\sum_{j=0}^{l-1}\phi(\mathbf q_l,\mathbf k_j)},
\qquad
\mathbf h_l=\sum_{i=0}^{l-1}\alpha_{i\to l}\mathbf v_i.
\tag{9}
$$

由于网络深度并不大 ($L<100$), 这一完整形式的 $O(L^2d)$ 计算量可以接受. 实际开销来自保留所有层输出所需的 $O(Ld)$ 内存, 以及流水线并行下的跨 stage 通信.

#### 分块 Attention Residuals

为降低这部分开销, 我们将 $L$ 层划分为 $N$ 个块, 每个块包含 $S=L/N$ 层. 在块 $n$ (层索引为 $\mathcal B_n$) 内, 各层输出通过求和归约为单一表示 $\mathbf b_n=\sum_{j\in\mathcal B_n}f_j(\mathbf h_j)$, 其中 $\mathbf b_n^i$ 表示块内前 $i$ 层的部分和. 我们令 $\mathbf b_0=\mathbf h_1$, 使 token 嵌入始终作为一个信息源. 在块之间, 完整注意力仅作用于 $N$ 个块级表示. 对于块 $n$ 中的第 $i$ 层, 值矩阵为

$$
\mathbf V=
\begin{cases}
[\mathbf b_0,\mathbf b_1,\ldots,\mathbf b_{n-1}]^\top,&i=1,\\
[\mathbf b_0,\mathbf b_1,\ldots,\mathbf b_{n-1},\mathbf b_n^{i-1}]^\top,&i\ge2.
\end{cases}
\tag{10}
$$

键和注意力权重遵循式 8 和式 9. 最终输出层再聚合全部 $N$ 个块表示. 在 Block AttnRes 下, 内存和通信开销从 $O(Ld)$ 降至 $O(Nd)$. 同时, 这种分块结构也限制了推理时状态的大小, 使并行的块间结果可以通过 online softmax [Mil18] 与顺序的块内部分和高效合并, 从而显著降低推理时开销.

经验上, 在不同模型规模下, $N\approx8$ 均可保留大部分收益 [Kim26a]. 对于 Kimi K3, 我们将各层划分为 8 个块, 每个完整块包含 12 层, 因此最后一个块不完整; 若计入嵌入层, 共得到 9 个块.

### 2.3 Stable LatentMoE

同时增大专家池和激活专家数量能够扩展专家分工空间, 但在传统 MoE 中, 每个被选中的专家都会接收完整的 $d$ 维 token 表示, 因此通信量和专家权重传输量会随路由数量一同增长. LatentMoE [Fed22] 将完整模型宽度与路由专家宽度分离, 从而以可接受的成本实现这种扩展: 共享专家保留全宽路径以执行通用变换, 专门化的路由专家则在宽度为 $\ell$ 的紧凑潜空间中运行. 借此, Kimi K3 将通道混合扩展到 896 个路由专家, 每个 token 激活 16 个专家, 对应稀疏度为 56.

这种极高稀疏度会放大基础设计中的两种失效模式. 首先, 路由路径将 $\mathbf W^\downarrow$, 门控多分支专家前馈网络和 $\mathbf W^\uparrow$ 组成一条包含近四次连续矩阵乘法的链. 这一病态结构与 2.8T 参数规模共同作用, 会使路由分支内部的激活爆炸. 其次, 为近 $10^3$ 个专家平衡负载, 超出了现有无辅助损失偏置更新能够稳定工作的范围. Stable LatentMoE 使用三个组件处理这两种失效模式: 在上投影前加入 RMSNorm, 使用 Sigmoid Tanh Unit GLU (SiTU-GLU) 抑制激活爆炸, 并使用 Quantile Balancing (QB) 平衡负载.

如[图 2](#figure-02) 所示, 该层沿用 DeepSeekMoE [Dai24] 的共享专家与路由专家组织方式. 对于 $\mathbf x\in\mathbb R^d$, 共享专家直接处理 $\mathbf x$, 路由路径则将其投影为 $\mathbf z=\mathbf W^\downarrow\mathbf x\in\mathbb R^\ell$, 把 $\mathbf z$ 分派给选中的专家, 再通过 $\mathbf W^\uparrow$ 将它们的加权聚合结果映射回 $\mathbb R^d$:

$$
\mathbf u=\sum_{i\in\mathcal T_k(\mathbf x)}p_iE_i^{\mathrm{routed}}(\mathbf W^\downarrow\mathbf x),
\qquad
\mathbf y=\sum_{j=1}^{N_s}E_j^{\mathrm{shared}}(\mathbf x)
+\mathbf W^\uparrow\operatorname{RMSNorm}(\mathbf u).
\tag{11}
$$

其中, $\mathbf u\in\mathbb R^\ell$ 是聚合后的路由表示, $E_j^{\mathrm{shared}}:\mathbb R^d\to\mathbb R^d$ 和 $E_i^{\mathrm{routed}}:\mathbb R^\ell\to\mathbb R^\ell$ 分别是共享专家和路由专家的前馈网络, $p_i$ 是由下文 Quantile Balancing 规则定义的路由权重. Kimi K3 将每层全宽共享专家的数量固定为 $N_s=2$.

#### 2.3.1 Normalized LatentMoE

原始 LatentMoE 直接将 $\mathbf W^\uparrow$ 应用于聚合后的路由表示 $\mathbf u$, 而后者的尺度会随选中专家及其路由权重变化. 如式 11 所示, Kimi K3 改为在专家聚合与上投影之间插入 RMSNorm [Zha19]. 在路由分支与全宽共享分支合并前, 这一归一化降低了路由分支对尺度变化的敏感度. 除稳定训练外, 额外的 RMSNorm 还能稳定改善验证损失和下游基准结果.

#### 2.3.2 Sigmoid Tanh Unit GLU

Gated Linear Unit (GLU) 使用 sigmoid 激活的门调节线性值分支, 计算 $\operatorname{Sigmoid}(\mathbf W_g\mathbf x)\odot\mathbf W_u\mathbf x$ [Yan17]. SwiGLU 将 sigmoid 门替换为 $\operatorname{Swish}(x)=x\operatorname{Sigmoid}(x)$, 并在 Transformer 中表现出优异的经验性能 [Sha20]. 此后, SwiGLU 成为大语言模型中广泛采用的 FFN 设计, 但其经验效果仍缺少完整解释. 然而, SwiGLU 的两个乘法因子均无界, 因此当同一坐标上的值同时较大时, 可能产生激活离群值, 并增加低精度运算中的溢出风险. 原始 GLU 的 sigmoid 门不会无限增长, 却无法保留 Swish 在正半轴上的近似线性区间. 因此, 我们需要一种既能控制大值增长, 又能保留 SwiGLU 特有的局部响应和正半轴响应的激活函数. 近期另有工作探索了这一权衡的不同参数化方式 [Jia26].

为满足这些要求, 我们提出 Sigmoid Tanh Unit GLU (SiTU-GLU). SiTU-GLU 将平滑上限函数 $\operatorname{softcap}(x,\beta)=\beta\tanh(x/\beta)$ 分别应用于 Swish 门的线性因子和上投影分支:

$$
\operatorname{SiTU\text{-}GLU}(\mathbf x)=
\beta_1\tanh(\mathbf W_g\mathbf x/\beta_1)
\odot\operatorname{Sigmoid}(\mathbf W_g\mathbf x)
\odot\beta_2\tanh(\mathbf W_u\mathbf x/\beta_2).
\tag{12}
$$

对于 Kimi K3, 我们将门分支的 soft-cap 超参数设为 $\beta_1=4$, 将上投影分支的超参数设为 $\beta_2=25$. 缩放后的 tanh 在原点附近近似线性, 在幅度较大时则有界, 因而 SiTU-GLU 能够在控制乘积中两个因子的同时, 保留 SwiGLU 的局部响应. [图 4](#figure-04) 在同一切片上比较了 GLU, SwiGLU 和 SiTU-GLU 的分支定义与标量响应. 附录 B 给出了局部展开, 极限情形, 形式化输出界以及与硬截断的比较.

<span id="figure-04"></span>

![GLU, SwiGLU 和 SiTU-GLU 的分支响应](./kimi-k3/figure-04.png)

**图 4.** GLU, SwiGLU 和 SiTU-GLU 的门分支与上投影分支, 以及各自的标量响应. SiTU-GLU 在原点附近紧随 SwiGLU, 对较大正输入则逐渐趋近上界 $|f(x)|\le\beta_1\beta_2=100$, 而 SwiGLU 始终无界.

#### 2.3.3 Quantile Balancing

Kimi K3 没有使用基于辅助损失的路由 [Fen25], 而是采用无辅助损失路由 [Dee24a]. 负载均衡通过在用于 Top-$k$ 选择的路由分数上, 加入专家特有的偏置 $b_j$ 实现. 对于 token $\mathbf x_i$, 路由器计算 $\mathbf s_i=\operatorname{Sigmoid}(\mathbf W_r\mathbf x_i)$, 并应用

$$
\mathcal T_i=\operatorname{argtop}_k(\mathbf s_i+\mathbf b),
\qquad
p_{i,j}=\frac{s_{i,j}}{\sum_{r\in\mathcal T_i}s_{i,r}},\quad j\in\mathcal T_i.
\tag{13}
$$

由于 $p_{i,j}$ 中不包含 $\mathbf b$, 它能够调节分派结果, 而不会改变混合权重或基于梯度的路由器优化. 原始方法使用固定步长规则 [Dee24a] 更新 $\mathbf b$

$$
b_j^{(t+1)}=b_j^{(t)}+\gamma\operatorname{sign}(\ell-\ell_j^{(t)}),
$$

其中, $\gamma$ 用于权衡缓慢适应与负载振荡. 当 LatentMoE 将每层路由专家池扩大到 896 个专家后, 维持负载均衡变得更加困难. 不均衡的路由会减慢专家并行训练, 还可能导致部分专家训练不足 [Hua26].

为解决这一限制, 我们提出 Quantile Balancing (QB), 根据与目标负载相匹配的路由分数分位数, 设置每个专家的偏置 [Su26]. 考虑一个包含 $m$ 个 token 的训练 batch, 采用 Top-$k$ 选择将其路由到 $n$ 个专家, 则每个专家的目标负载为 $q:=mk/n$ 个 token. QB 通过单次前向传播计算下一次偏置. 路由时, 对带偏置的分数 $\mathbf s_i+\mathbf b^{(t)}$ 执行 Top-$(k+1)$ 而非 Top-$k$: 前 $k$ 项是实际采用的路由, 第 $(k+1)$ 项则是专家进入 token $i$ 的 Top-$k$ 所必须超过的阈值 $\alpha_i^{(t)}$.

从 Top-$(k+1)$ 路由中取得阈值, 可以避免在 token 侧另行计算分位数. 固定阈值后, 在候选偏置 $b_j^{(t+1)}$ 下路由到专家 $j$ 的 token 数为

$$
\sum_{i=1}^{m}\mathbf 1\!\left[s_{i,j}+b_j^{(t+1)}>\alpha_i^{(t)}\right],
$$

该值随阈值 $-b_j^{(t+1)}$ 单调递减. 假设不存在相同值, 令该计数等于 $q$, 则 $b_j^{(t+1)}$ 是第 $(q+1)$ 大的 margin $s_{i,j}-\alpha_i^{(t)}$. 由于 $q/m=k/n$, 它就是所有 token margin 的 $(1-k/n)$ 分位数, 从而得到如下 QB 更新

$$
\begin{aligned}
b_j^{(t+1)}&\leftarrow\operatorname{quantile}_{1-k/n}(\mathbf s_{:,j}-\boldsymbol\alpha^{(t)}),\\
\mathbf b^{(t+1)}&\leftarrow\mathbf b^{(t+1)}-\operatorname{mean}(\mathbf b^{(t+1)})\mathbf 1.
\end{aligned}
\tag{14}
$$

margin 是原始分数 $s_{i,j}$ 减去带偏置阈值 $\alpha_i^{(t)}$ 的结果, 因而旧偏置只会通过阈值进入更新. 第二行移除一个不会改变 Top-$k$ 选择的公共偏移. 为满足因果性, 更新只在下一步生效 [Dee24a], 即 batch 绝不会使用由自身计算出的偏置进行路由. [图 5](#figure-05) 展示了 $m=8$, $n=4$, $k=1$ 的情形, 此时每个专家都接收目标负载 $q=2$. 推理时会冻结最终偏置. 附录 C 给出了均衡分派的推导.

<span id="figure-05"></span>

![Quantile Balancing 示例](./kimi-k3/figure-05.png)

**图 5.** $m=8$ 个 token, $n=4$ 个路由专家, 每个 token 选择 $k=1$ 个专家时的 Quantile Balancing. 专家侧偏置更新将不均衡负载 $(4,3,1,0)$ 调整为均衡负载 $(2,2,2,2)$.

#### 直方图估计

在大规模训练中, 式 14 的分位数覆盖整个全局 batch, 涉及数百万个分散在不同 rank 和梯度累积步骤中的 margin, 因此训练时无法将其集中起来精确计算分位数. 我们改为从每个专家的 margin 直方图中读取其分位数: 一次 all-reduce 对各 rank 的 bin 计数求和, 再从汇总后的计数中恢复分位数. 计数具有可加性, 因而无论 token 如何分片, 直方图都表示汇总后的全局 batch. 由此得到的估计能以 bin 宽度为误差界反映整个 batch 的分位数, 而每个专家的通信量只有数百个 bin. 实践中我们采用的正是这种直方图估计器. [附录 D](#d-基于直方图的分位数估计) 将进一步介绍其具体方法和误差界.

### 2.4 原生视觉

Kimi K3 原生支持多模态: 文本, 图像和视频在同一个上下文中由单一共享主干网络处理, 不需要事后执行模态对齐. 该设计是[第 1 节](#_1-引言)所述长程视觉闭环行为的架构基础. 渲染结果与生成它们的代码位于同一 token 流中, 模型可以编写代码, 检查结果的屏幕截图或视频帧, 再迭代改进用户界面, 图形和视频等视觉产物, 无需在模型间交接.

#### MoonViT-V2

Kimi K3 与 Kimi K2.5 的一个关键差异是, 我们完全从零开始, 使用下一 token 预测训练其视觉编码器 MoonViT-V2. 包括 Kimi K2.5 在内的先前实践, 通常使用 SigLIP 等经过对比预训练的模型初始化视觉编码器, 其前提是预训练视觉知识能为模型提供更好的起点. 我们改变这一做法, 主要是出于训练稳定性的考虑. 将预训练编码器接入 LLM 后, 联合优化会变得不稳定: 使用 SigLIP 初始化的 MoonViT-3D 始终具有更高的梯度范数, 且频繁出现尖峰; MoonViT-V2 则在整个训练过程中保持稳定 ([图 6](#figure-06)). 使用下一 token 预测训练, 还使编码器表示可以直接由语言建模目标塑造, 而非受到更偏重全局语义, 轻视细粒度文本和结构线索的对比损失影响. 值得注意的是, 我们发现 MoonViT-V2 在各项视觉评估中均与使用 SigLIP 初始化的基线持平, 这表明大规模多模态语言模型无需使用对比预训练进行初始化.

<span id="figure-06"></span>

![视觉塔梯度范数](./kimi-k3/figure-06.png)

**图 6.** 预训练消融实验中的视觉塔梯度范数. 与使用 SigLIP 初始化的 MoonViT-3D 相比, 从零训练的 MoonViT-V2 梯度范数更低, 尖峰更少, 表明其优化过程更稳定.

**架构.** 该训练方案建立在沿用 Kimi K2.5 [Kim26b, Kim25a] 整体设计的视觉路径上: 视觉输入首先由 MoonViT-V2 编码, 再通过轻量级 MLP 投影器映射到 LLM 中. MoonViT-V2 是一个约有 0.4B 参数的 27 层视觉 Transformer, 它采用 RMSNorm, 并移除线性投影和注意力投影中的所有偏置项, 从而进一步稳定上述从零训练过程. 与 MoonViT-3D 相同, 图像和视频使用完全共享的参数处理: 注意力被分解为帧内空间 pass 和帧间时间 pass, 时间池化则沿时间维度进一步压缩 token. 投影前, 一个采用 $2\times2$ 下采样的 pixel shuffle 操作将视觉 token 数减少到四分之一, 使最高 $3584\times3584$ 像素的输入也能以可接受的成本容纳在 1M-token 上下文中.

### 2.5 Per-Head Muon

沿用 Kimi K2 的设计, Kimi K3 采用 Muon [Kel24] 作为矩阵参数的优化器. 对于注意力投影, 我们进一步将其改进为逐头变体: 不再对完整的 $\mathbf Q$, $\mathbf K$ 和 $\mathbf V$ 投影矩阵应用 Newton-Schulz 正交化, 而是沿注意力头维度划分其动量矩阵, 分别对每个注意力头的块执行正交化. 其直观解释是, 整矩阵正交化会把所有注意力头视为单个耦合块, 因此梯度或动量尺度较大的头会主导共享更新方向, 尺度较小的头则得不到充分归一化的更新; 逐头正交化可以均衡不同注意力头的更新尺度. 实践中, 这一设计使各注意力头的学习动态更加均衡, 并提高了大规模训练的稳定性. 由于在逐头高矩阵块上执行 Newton-Schulz 迭代的成本低于完整投影矩阵, 它还能略微降低优化器开销.

## 3 预训练

### 3.1 预训练数据

Kimi K3 在精心整理的语料库上进行预训练, 其中包含 Web 文本, 代码, 数学和知识四个主要文本领域, 以及大规模视觉语料库. 视觉数据涵盖图像描述, 图文交错文档, OCR, 感知, 视频和视觉编码数据. 我们的数据流水线基于为 Kimi K2 [Kim25] 开发, 并在 Kimi K2.5 [Kim26b] 中进一步改进的流水线构建.

**文本数据.** 每个领域的数据都经过基于规则的启发式过滤, 基于分类器的质量评分和去重, 各领域的采样率则通过小规模模型上的消融实验确定. 沿用 Kimi K2 [Kim25] 的改写方案, 我们使用风格和视角多样的提示, 分块自回归生成以及相对于源文档的忠实度验证, 改写知识与数学语料库.

#### 视觉数据

视觉语料库沿用 Kimi K2.5 [Kim26b] 的分类方式, 将开源数据集与内部的过滤, 合成和去重流水线相结合. 训练期间, 我们同时以绝对坐标和归一化坐标 ($[0,1]$) 提供坐标监督, 从而实现精确且对分辨率稳健的定位. 除传统的文本描述图像外, 我们还大幅扩展了程序化多模态数据, 在 SVG, 3D 资产, 网页, 游戏和 CAD 图纸等领域特有的格式中, 将代码片段与其渲染视觉效果配对.

### 3.2 扩展定律

前述架构, 数据和训练方面的改进共同定义了新的模型家族. 这些变化也会改变最优训练方式, 因此我们进行了专门的扩展定律研究, 以重新调整 batch size, 学习率, 每参数 token 比率 (TPP) 和模型形状等关键超参数. 在保留的 OOD 验证数据上评估后, [图 7](#figure-07) 中的扩展定律曲线表明, 这些改进共同使整体扩展效率相较 Kimi K2 提升约 $2.5\times$. [表 1](#table-01) 详细比较了 Kimi K2 与 Kimi K3 的架构, 并突出了促成这一提升的结构变化.

<span id="figure-07"></span>

![Kimi K2 与 Kimi K3 的扩展定律曲线](./kimi-k3/figure-07.png)

**图 7.** Kimi K2 和 Kimi K3 的拟合扩展定律曲线. Kimi K3 的扩展效率相较 Kimi K2 提升 $2.5\times$.

<span id="table-01"></span>

![Kimi K2 与 Kimi K3 的架构对比](./kimi-k3/table-01.png)

**表 1.** Kimi K2 与 Kimi K3 的架构对比.

我们的扩展定律研究始终更支持余弦衰减而非 Warmup Stable Decay (WSD) [Hu24], 因此将余弦衰减用作默认学习率调度. 我们在固定最小学习率的条件下比较余弦衰减和 WSD. 尽管先前工作报告称 WSD 可以达到甚至超过余弦衰减, 但我们观察到, 两种调度的最优超参数存在明显差异. 即使模型规模和训练 token 预算相同, 二者的最优峰值学习率和 batch size 也相差很大. 因此, 使用同一组超参数比较两种调度可能会不公平地偏向其中一种, 原因仅仅是这些超参数与它更为匹配. 为确保公平比较, 我们为每种调度分别进行扩展定律搜索. 在各自的最优超参数设置下, 余弦衰减始终能取得低于 WSD 的最终损失.

### 3.3 训练方案

Kimi K3 采用原生多模态训练策略, 从训练开始便联合优化语言和视觉, 而非通过事后的对齐阶段将视觉编码器嫁接到预训练语言模型上. 在这一范式下, 视觉和文本 token 在同一个下一 token 预测目标中交错出现, 使共享主干网络从一开始就能学习统一的多模态表示.

我们使用 Per-Head Muon 优化器 ([§2.5](#_2-5-per-head-muon)) 和 Kimi K2 引入的权重截断机制优化模型, 同时采用 QB ([§2.3.3](#_2-3-3-quantile-balancing)) 实现 MoE 负载均衡. 我们使用带有 1% 线性 warmup 的余弦学习率调度, 并始终将权重衰减设为 0.1.

预训练从 8k token 的上下文长度开始, 并在后续训练阶段扩展到 64k token.

### 3.4 长上下文扩展

**位置编码.** Kimi K3 不使用显式位置嵌入 (NoPE), 而是通过 KDA 的递归门控和衰减机制隐式编码位置信息. 因此, 模型无需进行 RoPE 缩放或插值 [Pen23] 等位置编码修改, 即可直接外推到 1M-token 上下文.

**长上下文数据.** 来自自然来源的长文档和视频包含大量低质量内容, 例如近似重复项, 二进制数据块, 截断文件, 视频片段和无效的机器生成日志. 因此, 我们使用专用清洗流水线处理这些数据, 其中结合了精确去重与模糊去重, 面向视频帧的感知哈希, 基于启发式规则与分类器的质量过滤, 以及结构验证. 相较短文本, 真正连贯的长文档和视频非常稀缺, 因此我们对其进行上采样, 避免长上下文分布在 cooldown 阶段被短序列淹没. 然而, 仅有长度并不能带来长程能力. 为此, 我们仔细排列并拼接多模态文档与子任务, 合成额外的长上下文数据, 使嵌入其中的任务只有关注分散在整个 1M-token 上下文中的信息才能解决. 这会以预期规模训练注意力机制, 并防止其退化为局部模式.

**渐进式上下文扩展.** Kimi K3 支持最长 100 万 token 的上下文窗口. 我们采用四阶段课程, 随训练推进逐步扩展上下文窗口. 在预训练期间, 窗口从 8K token 增长到 64K token; 在 cooldown 阶段, 则从 256K token 增长到 1M token. 将成本高昂的长序列计算集中在总训练预算的一小部分内, 既能控制课程成本, 又能让模型逐步适应越来越长程的依赖关系. [第 5.1.2 节](#_5-1-2-kda-上下文并行)将介绍使 KDA 层上的百万 token 训练切实可行的序列维度划分方法.

## 4 后训练

### 4.1 方法

我们的后训练流水线采用三阶段范式: 通过监督微调 (SFT) 初始化基础智能体能力, 通过强化学习 (RL) 训练不同推理强度下的专业领域专家, 再使用多教师在策略蒸馏 (MOPD) 将这些领域特定策略整合到单一模型中.

#### 4.1.1 监督微调

SFT 阶段为后续 RL 阶段建立高质量的冷启动策略. 在先前 Kimi 模型的 SFT 流水线 [Kim25, Kim26b] 基础上, 我们扩展了 Kimi K3 的 SFT 数据集, 大幅拓宽其对复杂智能体任务的覆盖范围. 具体而言, 我们使用先前 Kimi 系列中针对不同领域专门训练的模型合成数据轨迹, 再进行多阶段验证和人在回路标注. 为统一表示这些复杂的智能体轨迹, 我们使用基于 XTML (eXtensible Token Markup Language) 的聊天模板序列化全部数据, 详见[附录 F](#f-聊天模板). 这些步骤共同生成了大规模指令数据集, 使 Kimi K3 在长程智能体场景中具备自适应推理, 精确工具调用和稳健执行能力. 此外, 我们从 SFT 阶段起便应用量化感知训练 (QAT), 使用 MXFP4 权重和 MXFP8 激活 ([§4.1.4](#_4-1-4-面向部署的后训练)).

#### 4.1.2 强化学习

<span id="figure-08"></span>

![公开与内部评估中的 RL 扩展](./kimi-k3/figure-08.png)

**图 8.** 强化学习期间, 公开与内部评估中的分数和助理平均步数. 扩展 RL FLOPs 会稳定增加工具调用步数, 并提高整体能力.

SFT 虽然提供了坚实的冷启动基础, RL 对解锁高阶推理和执行能力仍至关重要. 我们没有针对单项任务训练专用 RL 模型, 而是在三个广泛领域中扩展 RL. 每个领域都包含多种子任务, 我们在每个推理强度级别上分别为各领域训练一个专家: (i) 通用任务, 涵盖通用体验, 视觉, 推理, 忠实度, 搜索能力和知识工作任务; (ii) 通用智能体, 涵盖长程助理任务, 深度研究和段落级写作; (iii) 编码智能体, 涵盖软件工程 (SWE), 编码体验, kernel 任务和 Web 开发. 如[图 8](#figure-08) 所示, 扩展 RL FLOPs 能稳定提高知识, 推理, 视觉, 通用智能体和编码等多方面能力. 将三个领域专家与 $\{\mathrm{low},\mathrm{high},\mathrm{max}\}$ 三个推理强度级别组合, 共得到九个专家模型.

**算法.** 为缓解长程任务中更加严重的长尾延迟, 我们扩展了同步 RL 框架中的部分 rollout 方案 [Sca25, Kim26b]. 在每次迭代的 rollout 阶段, 我们为 $N$ 个提示中的每一个采样 $K$ 个补全, 从而维持包含 $N\times K$ 条轨迹的活跃工作负载. 生成阶段无需等待所有 rollout 结束, 而是在比例为 $\lambda\in(0,1)$ 的轨迹完成后, 即完成 $\lambda NK$ 条轨迹后立即暂停, 使策略优化无需等待执行较慢的尾部任务. 暂停的 rollout 会进入队列, 并借助沙箱基础设施 ([§5.3.2](#_5-3-2-沙箱基础设施)) 在下一次迭代开始时优先恢复. 某个提示的全部 $K$ 个响应完成后, 会立即送去进行策略优化, 其算法沿用 Kimi K2.5 [Kim26b]. 在部分 rollout 方案下, 单条长程轨迹自然会跨越多次迭代, 由此引入的数据陈旧性会威胁训练稳定性. 我们的策略优化算法通过逐 token 正则化, 天然能够容忍这种极端的离策略情形. 该正则化将策略更新限制在局部邻域内, 使算法能够稳健处理高度陈旧的数据, 并维持训练稳定性.

**推理强度 RL.** 为在微调推理强度的同时最大化 token 效率, 我们在 RL 期间实现了逐问题预算控制机制 [Kim26b]. 对每个问题 $x$, 我们都关联一个由冷启动模型估计的初始 token 预算 $b_0(x)$. 若轨迹的总 token 预算 $T(y)$ 超过缩放阈值 $\tau b_0(x)$, 则将其任务奖励改写为 $-1$. 对于通用任务, $T(y)$ 衡量思考 token 数; 对于智能体任务, $T(y)$ 则统计推理轨迹和工具调用参数在内的累积输出 token 数. 训练针对预算乘数 $\tau$ 采用分阶段课程. 我们先使用较大的 $\tau$ 训练最大预算变体, 同时仍限制最大预算以抑制过度思考. 随后将 $\tau$ 退火到更小的值, 得到 high 和 low 强度专家模型. $\tau$ 的调整在人的指导下按领域配置. 最终, 我们共同收集各推理级别专家生成的轨迹, 用于监督微调和多教师在策略蒸馏.

#### 智能体生成式奖励模型

对于不可验证的通用任务, 我们采用智能体生成式奖励模型 (GRM), 并沿用 Kimi K2.5 [Kim25, Kim26b] 基于二元比较的锦标赛式群组奖励. 除了用于增强判断能力的通用智能体能力外, 智能体裁判还必须遵循规定流程: (1) 阅读结果, 产物或文本输出; (2) 生成评分标准; (3) 依据评分标准为每个候选项打分; (4) 在记分板中记录评分标准给出的分数. 为缓解输出不断变长这一奖励破解行为, 我们采用与上述推理强度控制类似的预算式冗长度控制: 给定由冷启动模型估计的初始冗长度 $\ell_0$ 和乘数 $\sigma$, 输出长度超过 $\sigma\ell_0$ 的候选项会自动输掉二元比较.

#### 4.1.3 多教师在策略蒸馏

我们采用多教师在策略蒸馏 (MOPD), 将不同推理强度下针对各领域专门训练的能力整合到统一模型中 [Lu25, Xia26, Dee26]. 训练期间, 对给定领域 $d$ 和采样得到的推理强度级别 $e\in\{\mathrm{low},\mathrm{high},\mathrm{max}\}$, 优化过程由九个专家中对应的教师模型 $\pi_{\mathrm{teacher}}^{(d,e)}$ 指导. 给定输入查询 $x$ 和响应前缀 $y_{<t}$, 在 $y_t$ 上计算的逐 token 在策略蒸馏奖励为

$$
r_{\mathrm{opd}}^d(y_t\mid e,x,y_{<t})=
\operatorname{clip}\!\left(
\operatorname{sg}\!\left[
\log\frac{\pi_{\mathrm{teacher}}^{(d,e)}(y_t\mid x,y_{<t})}
{\pi_\theta(y_t\mid e,x,y_{<t})}
\right],-R_{\max},R_{\max}\right).
\tag{15}
$$

其中, $\operatorname{sg}(\cdot)$ 表示停止梯度算子, $R_{\max}>0$ 是用于限制极端优势信号的截断阈值, 从而稳定 RL 训练. 这种稠密奖励信号可以无缝集成到我们的 RL 框架中, 自然支持长程任务的部分 rollout 训练等基础设施级优化. 我们也尝试了更细粒度的 Top-$k$ 蒸馏目标, 但在当前设置下, 无论收敛速度还是最终性能都没有观察到明显优势.

#### 4.1.4 面向部署的后训练

#### MXFP4 量化感知后训练

为减少部署时的内存占用和服务成本, 我们将占模型参数内存主体的 MoE 专家权重量化为 MXFP4 [Dar23], 使用 MXFP8 计算激活, 同时将所有非专家组件 (注意力投影, 潜 MoE 投影, 共享专家和 MoE 路由器) 保持在较高精度. 我们在包括 SFT 和 RL 在内的整个后训练阶段执行量化感知训练 (QAT) [Ben18], 使模型适应量化造成的精度损失. RL 期间, rollout 与训练使用相同的量化方案, 从而消除训练与推理之间的不匹配.

**草稿模型微调.** 提高推理效率对于服务复杂的长程智能体模型至关重要. Kimi K3 预训练时包含一个结构与主干块对应的多 token 预测 (MTP) 层. EAGLE-3 [Li25] 的草稿模型由单个解码器层组成, 其结构与 MTP 层一致. 因此, 我们将预训练 MTP 层微调为 EAGLE-3 风格的草稿模型, 冻结目标模型, 只更新草稿层及其特征融合投影. 沿用 EAGLE-3 的训练时测试协议, 草稿模型在训练期间展开七步. 第一步之后, 最新位置的目标侧特征尚不可用, 此时草稿模型使用自身在先前步骤的输出, 以模拟推理时的递归起草过程.

草稿输入融合目标模型的低层, 中层和高层特征, 三者分别取自第 1 个, 第 4 个和最后一个 AttnRes 块的输出 ([§2.2](#_2-2-attention-residuals)). 这些特征会被拼接, 再通过无偏置矩阵 $\mathbf W_{\mathrm{E3}}$ 投影到隐藏维度. 该矩阵初始化为 $[\,\mathbf 0\;\mathbf 0\;\mathbf I\,]$, 使融合表示在初始时与高层特征 $\mathbf h^h$ 一致. $\mathbf h^h$ 正是 MTP 层预训练时的输入, 微调过程中则会逐渐学会整合低层和中层特征.

在无损推测采样下, 推测解码的加速效果由逐 token 接受率 $\sum_{x\in\mathcal V}\min(p(x),q(x))$ 决定, 其中 $p$ 和 $q$ 分别表示目标模型和草稿模型的下一 token 分布. 对容量有限的草稿模型而言, 最小化传统 KL divergence 代理目标并不能保证该接受率最大. 因此, 我们直接优化基于似然的 LK 损失 [Sam26], 即接受率本身的负对数

$$
\mathcal L_{\mathrm{LK}}=-\log\sum_{x\in\mathcal V}\min(p(x),q(x)).
\tag{16}
$$

其中, $p$ 和 $q$ 均在温度为 1 时计算, 且不使用额外的真实标签交叉熵项. 草稿模型微调沿用后训练 QAT 配置 ([§4.1.4](#_4-1-4-面向部署的后训练)), MoE 专家权重采用 MXFP4, 其输入激活采用 MXFP8, 非专家模块则保持较高精度.

### 4.2 RL 任务合成与智能体环境

RL 框架的有效性在很大程度上依赖丰富, 多样且能够稳健验证的环境. 为支持复杂长程任务上的可扩展训练, 我们设计了一系列专用白盒环境和任务合成范式.

#### 4.2.1 统一白盒 RL 环境

使用单一固定的智能体 harness 进行训练, 可能使模型过拟合于特定工具 schema, 系统提示, 上下文管理机制或交互协议. 为此, 我们开发了统一白盒 RL 环境, 将智能体 harness 表示为一组可配置, 可组合的模块, 包括工具接口, 系统提示, 上下文管理策略, skills, 记忆, 子智能体及其他组件. 通过配置组合这些模块, 环境可以实例化 Kimi Code [Kim26], Claude Code [Cod26], Codex [Cod26a], OpenClaw [Ope26a] 和 Hermes [Age26b] 等主流 harness, 也能实例化全新 harness. RL 训练期间, 我们为不同任务组动态构造不同的 harness 配置, 让 Kimi K3 接触这些模块的多种组合, 而非任何单一 harness 的惯例. 同一抽象也能轻松支持多个任务领域中的 RL, 为训练更通用的智能体提供可扩展基础.

#### 4.2.2 知识图谱引导的任务合成

**动机与概述.** 后训练任务的质量和多样性主要由其来源材料决定. 由细粒度概念引导的检索可以发现专业且代表性不足的知识, 跨不同概念采样则能拓宽领域覆盖范围. 为大规模控制粒度和覆盖范围, 我们构建了一个自演化, 分层组织的知识图谱, 智能体通过在知识密集型领域和编码领域开展 Web 规模探索, 持续扩展该图谱. [图 9](#figure-09) 展示了任务合成流水线. 分层组织的知识图谱在多个层级上表示概念, 范围从广泛领域一直延伸到细粒度概念. 系统采样相关节点形成关键词集合, 用其指导公开来源材料的检索. 对于每个合成实例, 系统选择一种任务类型, 并使用检索到的材料合成相应任务.

<span id="figure-09"></span>

![知识图谱引导的任务合成](./kimi-k3/figure-09.png)

**图 9.** 知识图谱引导的任务合成. 分层概念图中的相关节点指导来源检索和多样化任务合成.

#### 智能体知识图谱构建

我们通过智能体驱动的递归扩展, 将知识图谱构建为有向无环图. 扩展过程从一组预定义的粗粒度种子节点开始. 随后为每个节点分配一个智能体实例, 通过多次 Web 搜索研究对应概念. 添加新节点前, 智能体会探索现有图谱, 识别等价或相关概念, 在适当情况下复用现有节点, 并尽量减少重复. 无论智能体先发现边的哪一端, 边都始终从较粗粒度概念指向较细粒度概念. 新增节点随后会分配给其他智能体继续探索. 当负责的智能体判定当前概念已经足够原子化时, 对应分支便停止扩展.

#### 材料检索与任务合成

为使不同领域和任务类型达到目标分布, 系统会单独采样不同粒度的节点, 或以相关组合的形式采样. 由采样节点得到的关键词会与知识图谱中祖先节点的上下文信息结合, 形成 Web 查询. 检索到的真实世界材料经过汇总后, 由合成智能体生成各种类型的训练任务.

#### 4.2.3 智能体环境中的可验证问题

我们使用智能体环境中的可验证问题训练 Kimi K3. 代表性示例包括: 多步复杂信息搜索, 模型规划研究过程, 逐步从 Web 收集证据, 并给出可验证答案; 投资银行, 数据分析和法律实务等专业人士的真实日常工作, 模型分解复杂请求, 在沙箱中操作领域工具, 并通过数十至数百步完成交付物; 面向 STEM 问题, 视觉谜题和图表理解的多步可验证视觉推理. 每条视觉推理轨迹都在配有 Python 解释器的隔离沙箱智能体环境中生成: 模型反复编写并执行代码, 对输入图像进行裁剪, 缩放或变换, 执行精确计算或验证中间结果, 并在多轮交互中将包括生成图像在内的执行输出作为新观察接收. 随着模型学会执行更多图像操作并收集更多观察, 它在复杂视觉推理任务上的表现会稳定提升.

#### 4.2.4 Kernel 优化任务

为增强 Kimi K3 的 GPU kernel 优化能力, 我们构建了大规模 kernel 任务套件, 范围从单算子 kernel 到融合 mega-kernel, 数据来源包括 Flash Linear Attention [Yan24] 等优质 GitHub 仓库. 该套件涵盖 CUDA, Triton, CuTe DSL, Gluon, ThunderKittens [Ben25] 和 TileLang [Wan25] 等多种 GPU 编程方式, 以及广泛使用的 GPU 架构和 BF16, FP8, FP4 等数值格式. 奖励同时评估正确性与性能: 每个 kernel 都提供一个 PyTorch 参考实现, 数值误差超过预定义阈值的解答奖励为零. 性能相对于专家实现评分, 与专家实现持平时奖励为 0.5, 越接近硬件 roofline, 奖励就越接近 1. 为确保奖励反映真实优化, 我们开发了破解检测系统, 对 CUDA graph 重放, 输入缓存和降低精度等奖励破解策略施加惩罚. 在 Kimi K3 的开发过程中发现新破解策略后, 我们还会不断扩展该系统, 加入新的防护措施.

#### 4.2.5 个人助理任务

对于长程个人助理任务, 我们为 Gmail, Notion, Slack 和 Canvas 等常用应用开发了逼真的模拟实现. 它们保留真实应用的核心语义, 同时无需外部 API 或速率限制即可支持可复现的大规模交互. 基于这些模拟应用, 我们设计了受人力资源, 法律服务和金融等真实专业工作流启发的复杂任务. 在每项任务中, 智能体会在跨越多个模拟日的持久化演进环境中运行, 遇到分布在不同应用中的数十个相互依赖事件. 单次 rollout 最多可能涉及数千次工具调用和数百万上下文 token. 每个事件都有自己的评估标准, 由确定性规则或基于 LLM 的评估器判定. 初始工作区由智能体构建: 它们自主搜索 Web 获取参考材料, 再将材料转化为连贯且与任务相关的环境. 我们还扩展了 RL 框架, 以支持这种持续变化的环境, 对复杂事件流及其引发的世界状态转换进行建模.

#### 4.2.6 自主执行任务

我们提出自主执行任务 (AET), 这是一种通过验证闭环优化训练长程智能体能力的环境范式. 每项任务都指定初始状态, 受约束目标, 基于工具的动作空间, 执行预算和独立验证器. 智能体只能看到目标, 上下文, 约束和验证接口, 不会获得参考轨迹或预定义流程, 必须自主完成任务分解, 工具选择, 规划, 错误恢复和终止. 奖励以验证器对最终环境状态的评估为依据, 而非智能体自行报告的完成情况. 我们设计了多种验证器以支持不同环境, 包括黑盒系统复现 ([图 10](#figure-10)), 量化因子发现和税务审计.

<span id="figure-10"></span>

![自主执行任务完成曲线](./kimi-k3/figure-10.png)

**图 10.** Camera Repair Management System 上的完成曲线. 这是一项黑盒系统复现任务, 智能体通过 oracle 查询, 将隐藏的 3D 相机维修系统重建为 Web 应用.

在每个环境中, 智能体迭代提交解答, 接收验证器反馈并改进策略, 由此训练一个提出假设, 行动, 分析反馈和适应的通用循环. 我们通过隔离智能体与验证器, 将提供诊断反馈的公开验证器和评估保留场景的隐藏验证器配对, 并在有限提交预算下应用惩罚式奖励, 以缓解奖励破解.

#### 4.2.7 Web 开发任务

我们构建了由专家整理的多样化 Web 开发任务套件, 涵盖各类典型场景. 输入从单行场景描述到多段规格说明不等; 产物涵盖网站, 交互式游戏, 3D/WebGL 场景, 数据可视化, SVG 和全栈应用. 每项任务都在容器化沙箱中运行, 并使用多种智能体 scaffold 而非单一固定 harness 执行 rollout, 以促进跨 scaffold 泛化. 奖励由两部分组成: 确定性检查, 以及内部奖励模型执行的模型评判. 确定性检查对应用行为进行功能测试; 对于复现参考目标的任务, 还会评估结构和像素级相似度. 如果项目构建失败, 运行时出错, 或伪造而非真正实现产物, 则奖励归零. 模型评判使用其他模型检查源代码, 或查看输出产物并与之交互.

## 5 基础设施

Kimi K3 集合了三项很少同时出现在单个模型中的系统挑战: 混合 KDA 注意力, 3T 级稀疏多模态训练与推理, 以及百万 token 智能体工作负载. 我们围绕这些挑战, 对覆盖模型整个生命周期的基础设施进行协同设计. 在架构层面, 高性能 KDA kernel 和上下文并行使递归形式在训练和推理中都能高效地跨设备内外执行. 预训练期间, 均衡的专家执行, 更低的内存占用以及与通信重叠的调度, 能够在大规模训练中维持较高利用率. 在 1M-token 智能体 RL 期间, 分层状态管理与可恢复的沙箱执行使长轨迹得以跨迭代保留. 最后, 状态感知 KDA 前缀缓存, 专用推理 kernel 以及感知缓存和预算的调度, 将这些效率优势转化为可预测的生产服务能力.

### 5.1 KDA 的算法-系统协同设计

KDA 使用固定大小的递归状态 $\mathbf S\in\mathbb R^{d_k\times d_v}$ 取代 softmax attention 中随序列增长的键值 cache ([§2.1.1](#_2-1-1-kimi-delta-attention)). 其串行更新给并行执行带来挑战, 但作为交换, 固定大小的状态传输和复用成本很低. 下述设计在两个执行层级上处理前一特性并利用后一特性: 设备内使用融合 kernel, 设备间使用 KDA 上下文并行.

#### 5.1.1 不同执行阶段的 KDA kernel

KDA 状态的串行依赖与 GPU 偏好的宽度大且均匀的并行方式相冲突, 并在不同执行阶段表现为不同瓶颈. 我们为每个阶段分别设计了专用 kernel.

**用于训练和 prefill 的分块 kernel.** KDA 的分块形式在块内并行, 但由于递归状态必须在块之间传播, 块间仍为串行. 直接执行时, 两个阶段交替进行, 串行传播期间 SM 处于空闲状态. 因此, 我们开发了 FlashKDA [Che26a], 这是一个基于 CUTLASS 的分块 kernel, 可将块内计算与跨块状态传播重叠. 该 kernel 将工作分解为 token 并行阶段和注意力头并行递归, 分别调度和调优, 性能显著优于 Triton 参考实现. FlashKDA 同时服务于训练和推理 prefill, 并作为 flash-linear-attention [Yan24] 的后端自动分派.

**面向长上下文 prefill 的设备内上下文并行.** 张量并行在设备间划分注意力头, 但不会缩短递归. 因此, 在纯 TP 部署下, 如果每个 rank 只持有少量注意力头, 对超长序列执行 prefill 时大部分 SM 都会空闲. 关键观察是, 每个分段的状态转移可以独立于传入状态计算, 再在之后精确组合. 因此, 自动化 SM 级上下文并行 (CP) 规划器 [Wan25a, Yan24] 在单个 rank 的各 SM 间划分序列, 并行计算各分段的状态转移, 再进行合并, 以恢复每个分段的精确初始状态. 与[第 5.1.2 节](#_5-1-2-kda-上下文并行)中的跨设备 KCP 不同, 这种并行完全发生在设备内, 不会产生跨设备通信.

KDA 解码面临的挑战不同于训练和 prefill. [第 5.4.2 节](#_5-4-2-高性能-kernel)将详细讨论这些挑战.

#### 5.1.2 KDA 上下文并行

softmax attention 和线性注意力的上下文并行在通信开销上存在本质差异. softmax attention 要求各 rank 交换大小随序列长度增长的键值块 [Liu23]. 线性注意力则使用固定大小的递归状态 $\mathbf S\in\mathbb R^{d_k\times d_v}$ 携带先前上下文. 先前的上下文并行方法利用基础线性注意力的加法递归: 每个 rank 计算本地 token 从 $\mathbf S=0$ 开始生成的状态, 再对先前各 rank 的局部状态求和, 以恢复传入状态 [Sun24, Sun25].

然而, 对 KDA 而言, 直接求和并不足够. 回顾式 1, KDA 按 $\mathbf S_t=\mathbf M_t\mathbf S_{t-1}+\beta_t\mathbf k_t\mathbf v_t^\top$ 更新状态, 其中 $\mathbf M_t:=(\mathbf I-\beta_t\mathbf k_t\mathbf k_t^\top)\operatorname{Diag}(\boldsymbol\alpha_t)$. KDA 的 delta rule 在加入当前写入前, 先将依赖 token 的矩阵 $\mathbf M_t$ 应用于传入状态. 因此, 局部序列分段产生的影响取决于进入该分段的状态, 无法只根据从 $\mathbf S=0$ 开始计算的状态确定.

为保留这种依赖关系, 我们提出 KDA 上下文并行 (KCP), 将每个分段的影响分解为两个可在本地计算的量: 作用于传入状态的累积转移, 以及本地从零开始生成的状态. 沿用[第 2.1.1 节](#_2-1-1-kimi-delta-attention)中的分块记号, 我们以 $\mathbf S_{[i]}^t$ 表示 rank $i$ 的分段在处理 $t$ 个本地 token 后的递归状态, 因此 $\mathbf S_{[i]}^{T_i}$ 表示离开 rank $i$ 并进入 rank $i+1$ 的状态. 以 $\widetilde{\mathbf S}_{[i]}^t$ 表示同一递归改从 $\mathbf S=0$ 开始时的状态. 对进入 $P$ 个上下文并行 rank 中第 $(i+1)$ 个 rank 的任意状态, 处理 $t$ 个本地 token 后的状态为

$$
\begin{aligned}
\mathbf M_{[i+1]}^{t\leftarrow1}&:=\prod_{r=1}^{t}\mathbf M_r\in\mathbb R^{d_k\times d_k},\\
\mathbf S_{[i+1]}^t
&=\widetilde{\mathbf S}_{[i+1]}^t+\mathbf M_{[i+1]}^{t\leftarrow1}\mathbf S_{[i]}^{T_i}\\
&=\widetilde{\mathbf S}_{[i+1]}^t+\mathbf M_{[i+1]}^{t\leftarrow1}
\sum_{j=1}^{i}\left(\prod_{l\leftarrow j+1}^{i}\mathbf M_{[l]}^{T_l\leftarrow1}\right)
\widetilde{\mathbf S}_{[j]}^{T_j}
\in\mathbb R^{d_k\times d_v}.
\end{aligned}
\tag{17}
$$

其中, $\mathbf M_{[i+1]}^{t\leftarrow1}$ 表示前 $t$ 个本地 token 的累积转移. 第一项包含本地 token 生成的状态, 第二项则使先前 rank 的上下文经过本地 KDA 更新传播. 当 $t=T_{i+1}$ 时, $\mathbf M_{[i+1]}^{T_{i+1}\leftarrow1}$ 和 $\widetilde{\mathbf S}_{[i+1]}^{T_{i+1}}$ 都能在 $\mathbf S_{[i]}^{T_i}$ 可用前, 仅使用本地 token 计算. 它们正是每个 rank 与其他 rank 交换的分片.

式 17 的求和表明, 每个状态都完全由本地计算的分片组成. 这些 rank 级更新满足结合律, 因此可以通过 prefix scan [Mar18] 恢复每个 rank 的传入状态. 每个 rank 先在本地计算 $\mathbf M_{[i]}^{T_i\leftarrow1}$ 和 $\widetilde{\mathbf S}_{[i]}^{T_i}$, 再通过一次 all-gather [Yan24] 交换两个张量. all-gather 后, rank $i+1$ 从 $\mathbf S=0$ 开始, 按顺序处理同一文档的先前分片, 在每个分片上应用 $\mathbf S\leftarrow\mathbf M_{[j]}^{T_j\leftarrow1}\mathbf S+\widetilde{\mathbf S}_{[j]}^{T_j}$, 从而重建 $\mathbf S_{[i]}^{T_i}$. 因此, KCP 只需一次固定大小的 all-gather 即可同步递归状态, 并实现计算量的线性扩展.

### 5.2 3T 级预训练基础设施

Kimi K3 预训练结合了包含虚拟 stage (VP) 的流水线并行 (PP) [Hua19, Nar21], 专家并行 (EP) [Lep20], ZeRO-1 数据并行 [Raj20], Pipeline ZeRO-2 梯度分片 [Zen26], 以及上下文并行 (CP, [§5.1.2](#_5-1-2-kda-上下文并行)) [Sam23].

MoE 层使用在各 EP rank 上复制的共享专家, 并将专家分派与合并所需的 all-to-all 通信和计算重叠, 以隐藏通信延迟.

3T 级原生多模态预训练带来三个关键问题: (i) 各 EP rank 的 token 负载不均衡; (ii) 激活, 梯度和优化器状态超出内存预算; (iii) 计算量高度可变的视觉编码器暴露在关键路径上. 下列小节依次处理这些问题: 完全均衡的专家并行 MoE 训练 ([§5.2.1](#_5-2-1-完全均衡的专家并行-moe-训练)), 内存高效训练 ([§5.2.2](#_5-2-2-内存高效训练)) 和多模态编码器优化 ([§5.2.3](#_5-2-3-多模态编码器优化)). [图 11](#figure-11) 展示了最终的执行调度.

<span id="figure-11"></span>

![重叠执行的预训练调度](./kimi-k3/figure-11.png)

**图 11.** 在不同流水线并行阶段重叠执行计算, 通信和卸载.

#### 5.2.1 完全均衡的专家并行 MoE 训练

在传统 EP 方案中, 各 rank 的 token 负载并不均衡. 由此产生的计算不均衡会降低训练吞吐量, 路由专家激活动态变化的形状还会造成大量内存碎片. 因此, 我们提出 MoonEP ([代码仓库](https://github.com/MoonshotAI/MoonEP)), 这是一种通过动态冗余专家实现完全负载均衡的 EP 方案. MoonEP 保留 DeepEP [Zha25] 等传统方案的整体计算流程, 另外引入冗余专家的在线规划与迁移. 在前向传播中, 我们根据当前 micro-batch 和当前层的路由器输出规划冗余专家, 并在计算路由专家前进行预取. 在反向传播中, 我们将这些专家的梯度暂存到本地 reduce buffer, 计算完成后再将其归约回所属 rank 的梯度缓冲区.

**使用数量有界的冗余专家实现完全均衡.** MoonEP 要求每个 rank 恰好接收 $S\times K$ 个 token, 其中 $S$ 是序列长度, $K$ 是每个 token 选择的专家数, 从而使所有 rank 执行相同的计算量. 关键问题在于, 需要多少冗余专家才能保证这种均衡. 设 $E$ 为专家数量, $R$ 为 EP 大小. 我们证明, 每个 rank 最多使用 $E/R$ 个冗余专家时, 始终存在均衡方案, 且该界基本为紧界 (附录 E). 因此, 每个 rank 预留 $E/R$ 个冗余专家槽位, 即可保证规划始终存在可行解, 训练不会中断. 相比之下, ECHO [Yan26] 和 UltraEP [Wei26] 等先前工作会预设冗余专家数量, 或限制每个 rank 的 token 数. 一旦上限内不存在可行方案, 训练就会被迫停止; 上限本身也需要人工调优, 且仍然存在残余不均衡.

**在线规划.** 在每个训练步骤中计算精确最优解成本过高. 因此, 我们使用整数线性规划 (ILP) 离线计算代表性情形的精确解作为参考, 并设计了一个近似最优, 开销可忽略且始终遵守 $E/R$ 上界的 GPU 规划 kernel.

**零拷贝通信.** 完全均衡还能简化通信路径. 我们实现了融合的 permute/unpermute 算子, 其中规划 kernel 会预先计算每个 token 的目标位置, 使 token 直接发送到远程 rank 上按专家分组的位置, 并将通信缓冲区的 view 直接返回给计算过程, 从而消除中间拷贝. 在最坏的不均衡情况下, DeepEP 若要支持同样的无拷贝数据路径, 需要大小为 $S\times K\times R$ 的通信缓冲区; MoonEP 则得益于完全均衡, 只需固定大小的 $S\times K$ 缓冲区.

**静态计算形状.** 传统实现在启动专家计算前同步动态形状, 导致流水线在层间停顿. 实现完全均衡后, 每个 rank 都恰好接收 $S\times K$ 个 token, 所有层的计算形状均可静态确定. 这消除了逐层 MoE 主机同步, 并降低了主机侧的 kernel 启动开销.

**Expert-GEMM 调度与重叠.** 即使各 rank 的总负载完全均衡, 每个 rank 内各专家接收的 token 数仍然偏斜. 采用固定顺序且不感知工作负载的调度, 会将这种偏斜转化为各 SM worker 之间不均衡的完工时间. 因此, 我们使用感知工作负载的调度器安排路由专家 GEMM. 该调度器在启动前根据当前 token 分布调整参数, 并在执行期间保持参数不变.

一个轻量级启发式方法使用基于硬件指标的解析代价模型选择这些参数, 其中关键系数通过离线自动调优校准. 对于共享专家, 我们将其 GEMM 分派到独立 stream, 使其与其他 kernel 重叠执行.

#### 5.2.2 内存高效训练

**统一激活管理器.** 我们为激活设计了统一存储抽象, 其中为反向传播保存的每个张量都关联一个可插拔存储后端. 在该抽象下, 重计算, 量化以及卸载或远程卸载都只是存储策略, 可以在张量粒度自由组合. 策略通过张量上的轻量级注解声明, 与模型代码完全解耦. 重计算在函数粒度执行, 因而支持跨层重计算. 在我们的实现中, 所有 GPU 内存均在主计算 stream 上分配, 并由单个内存池管理, 从而避免多 stream 碎片和主机端开销; 激活以层为粒度预取回来并与计算重叠, 产生的额外开销可以忽略. 在 Kimi K3 中, 大部分激活采用分块 FP8 量化 [Kim25, Dee24a], 并结合卸载或远程卸载, 逐元素算子则配置为重计算.

#### 内存高效 MoE

在原生 MoE 实现中, `permuted_probs` 的梯度计算依赖前向输出 `output`. 受 SonicMoE [Guo25a] 启发, 我们通过数学变换将该梯度改写为仅依赖中间激活 `act_output` 和上游梯度 `doutput` 的形式. 代价是增加一次轻量级逐元素计算, 收益则是消除反向传播对 `output` 的依赖. 此外, 在 group GEMM 的前向传播中, 我们只保存 dispatch 操作的输入; 反向传播期间, 则通过重新计算 dispatch 恢复 group GEMM 的输入. 如[图 11](#figure-11) 所示, 这种重计算引入的通信可以与部分 group-GEMM 反向计算重叠, 从而以可忽略的代价消除这部分激活存储.

#### 内存高效 Attention Residual

对于 attention residual, 我们基于 Block AttnRes 设计了一项配套优化. 块表示在边界层只生成一次, 由所有后续层共享, 并直接驻留在 GPU 上. 整个 AttnRes 计算都封装在 checkpointing 中, 因此每层为反向传播保存的激活与标准残差架构完全相同. 对于流水线并行, 我们采用基于 cache 的流水线通信 [Kim26a], 只在 stage 之间增量传输新生成的块, 并在 micro-batch 完成后立即释放, 从而达到内存占用的理论下界.

**均衡各 PP rank 的激活.** 在交错 1F1B 流水线并行下, 受流水线 warmup 影响, 激活在各 PP rank 间分布不均, 常驻激活数量随 PP rank 增大而减少. 为避免内存不足 (OOM) 错误, 我们使用 Mooncake Transfer Engine [Qin24] 将激活远程卸载到其他 PP rank 的内存中, 从而均衡各 PP rank 的激活内存.

**Pipeline ZeRO-2 梯度分片与卸载.** 除激活外, 我们还使用 Pipeline ZeRO-2 梯度分片 [Zen26], 在数据并行 (DP) rank 间对梯度进行分片. 此外, 我们将分片后的梯度存储在 CPU 内存中, 以降低 GPU 内存峰值, 同时在 GPU 上保留双梯度缓冲区. 梯度在各 DP rank 间归约到双梯度缓冲区后, 再累积到 CPU 分片中.

#### 基于 P2P 的 Muon 正交化

分布式优化器将参数均匀分片到各 DP rank, 但 Muon 中的 Newton-Schulz 正交化需要完整参数矩阵, 因此每次更新前都必须通过通信收集完整参数. 朴素方法在每个 rank 上对整个参数缓冲区执行 all-gather [Liu25], 这不仅占用大量内存, 还使通信成为大规模运行时的主要瓶颈. 我们改为让每个 rank 只通过与对应所有者 rank 的点对点 (P2P) 通信, 获取本地所拥有参数的分片, 从而消除完整参数缓冲区, 同时减少内存占用和通信量. 通信与计算还会以模型块缓冲区为粒度进一步流水线化, 以隐藏通信开销.

#### 5.2.3 多模态编码器优化

**多模态编码器中的动态 CP.** 在长上下文多模态训练中, 大图像和长视频会显著增加视觉编码器的计算时间, 并导致严重的跨设备负载不均衡. 为此, 我们将上下文并行扩展到此类大型样本. 单张大图像沿 patch 维度划分到多个设备, 通过在各 CP rank 间收集键值对 (gather-KV) 来计算注意力. 此外, 我们将每个 CP group 划分为若干 sub-CP group, 并以负载均衡方式将多张大图像分布到其中, 防止通信占比随规模增长. 这既能降低大型视觉样本的编码器延迟和跨设备负载不均衡, 也使其余编码器计算可以隐藏在流水线 bubble 中.

#### PP bubble 中的编码器计算

在 Kimi K2.5 中, 我们提出了解耦编码器流程 (DEP) [Kim26b], 将 ViT 与文本训练拆分到不同 stage, 并均衡各 PP stage 的视觉前向与反向传播. 我们观察到, 在交错 1F1B 流水线调度下, 第一批 PP micro-batch 的文本前向传播都被安排在最开始, 最后一批 PP micro-batch 的文本反向传播则要到最末尾才完成. 因此, 我们进一步分解 ViT 计算 [Val26a]. 第一批 PP micro-batch 的 ViT 前向传播预先同步执行, 其余前向传播被安排到流水线 bubble 中, 反向传播也按类似方式处理. 最终, 大部分 ViT 计算都隐藏在流水线 bubble 中, 基本消除了视觉编码器的实际开销.

### 5.3 1M-token 智能体 RL 基础设施

在有限计算预算下, 将 Kimi K3 这样规模的模型所使用的智能体 RL 扩展到百万 token 上下文, 使资源效率成为首要目标. 因此, 我们开发了用于高效训练和 rollout 的长上下文 RL 基础设施, 以及支持长程环境交互的高性能可恢复沙箱.

#### 5.3.1 长上下文 RL 基础设施

我们采用共置 RL 训练 [Kim25], 将每次 1M 上下文 Kimi K3 RL 实验控制在数百块 GPU 内, 并使用部分 rollout [Sca25] 降低超长轨迹产生的尾部延迟. 这一设计提高了硬件利用率, 但长上下文 rollout 会因保留 KV cache 产生额外 DRAM 需求, 与训练侧状态竞争内存. 此外, 要同时实现高效 prefill 和解码, 还需要谨慎管理前缀和调度请求.

**外部 KV cache 池.** 在 1M 上下文多步 rollout 中, 前缀 KV cache 未命中的代价极高. 每次迭代开始时, 上一次迭代中大量未完成的长 prefill 请求会同时到达, 使部分 rollout 进一步加剧该问题. 推测解码还会在相对固定的工具调用间隔内加快请求周转, 增加前缀块变动. 这些问题可能触发抢占并降低 cache 命中率, 而命中率对长上下文 RL 至关重要.

因此, 我们使用写回式设计, 将前缀保留与 GPU 常驻状态解耦. 活跃解码块保留在 GPU KV cache 中, 可复用的空闲前缀只有在被逐出 GPU 时才写回 CPU DRAM 中的外部 KV cache 池, 并在下次复用前预取回来. KDA 状态与对应的 MLA KV cache 块一同卸载和预取, 使二者生命周期保持一致. 相较直写策略, 该策略只对离开活跃解码路径的前缀占用 CPU DRAM 和传输带宽, 避免为仍常驻 GPU 且处于活跃状态的块创建多余 CPU 副本.

为向外部池提供足够 DRAM, 我们在一次训练迭代完成后, 将训练状态 (模型权重和优化器状态) 卸载到 NVMe. rollout 迭代结束后, 则释放该池, 避免与训练工作负载争用资源.

#### Rollout 自动限流调度器

在多步 rollout 中, 上下文会随轨迹推进而逐渐增长, 因而基于完整轨迹平均长度设置固定并发度不仅难以估计, 在早期也过于保守. 反之, 并发度设置过高又会在后期增加 KV cache 压力, 并可能触发抢占. 因此, 我们在 LLM 请求调度层设计了自动限流机制, 使用活跃请求数, 排队请求数和 KV cache 利用率等运行时信号, 动态控制发送到推理引擎的请求数量. 这使 rollout 早期能够充分利用资源, 同时在 KV cache 压力上升时降低并发度, 无需人工调优即可避免利用不足和过载.

**复用梯度缓冲区执行非策略模型前向传播.** RL 损失计算通常需要仅执行前向传播的非策略模型, 例如参考模型, 而其权重过大, 无法常驻 GPU. 我们将这些权重保留在 CPU 内存中, 只在需要时具体化, 并使用策略模型的 FP32 梯度缓冲区存储作为其参数张量的后备存储. 这种方式无需额外分配或产生碎片, 即可复用现有 GPU 内存. 之后计算真实梯度时缓冲区会被覆盖, 因此该做法是安全的.

采用 ZeRO-2 梯度分片和卸载后 ([§5.2.2](#_5-2-2-内存高效训练)), 在 Kimi K3 RL 训练中, 每块 GPU 只为两个 VPP 块保留梯度缓冲区. 我们逐块将参考权重流式传入这些槽位: 一个槽位用于当前前向计算, 另一个槽位预取下一块, 从而在不增加 GPU 内存占用的情况下隐藏拷贝开销.

#### 5.3.2 沙箱基础设施

我们使用多种沙箱 runtime 支持 Kimi K3 后训练和评估的不同需求, 包括传统的容器 runtime, GPU 沙箱 runtime, 以及最值得关注的全新 microVM 沙箱 runtime AgentENV.

AgentENV ([代码仓库](https://github.com/kvcache-ai/AgentENV)) 由我们与合作伙伴共同开发, 是一个专为智能体 AI 工作负载设计的沙箱系统. 它围绕三个核心目标构建:

- **高保真隔离沙箱 runtime.** 随着智能体能力增强, 任务难度提高, 它们往往会进行更激进的探索, 甚至尝试奖励破解. 一方面, 这带来了独特的安全挑战: 在早期使用传统容器沙箱 runtime 的实验中, 我们观察到多次由智能体意外操作引起的 kernel panic 和死锁. 另一方面, 我们希望允许尽可能多的探索, 以免限制智能体能力; 复杂任务也需要接近真实世界环境的沙箱, 例如智能体应当能够按需挂载磁盘, 运行容器, 甚至启动虚拟机. AgentENV 使用 Firecracker [Aga20] 运行隔离的 microVM, 提供容器 runtime 无法达到的隔离性和保真度.

- **灵活的智能体 RL 沙箱生命周期.** 在底层, AgentENV 支持沙箱状态的增量 checkpoint 和恢复. 执行 checkpoint 时只保存自上次 checkpoint 以来变脏的内存页, checkpoint 和恢复延迟最低分别可达 133 ms 和 49 ms. 在此基础上, AgentENV 提供三种有助于提高智能体 RL 效率的高级操作. (a) 暂停与恢复: 暂停的沙箱不占用内存或 CPU 资源; 因此, 智能体等待模型推理结果时可以暂停沙箱, 而等待时间可能占沙箱生命周期的 98%. (b) Fork: fork 从原沙箱的精确状态创建新沙箱, 同时保持原沙箱运行, 适合用于无副作用的奖励评判. (c) Snapshot: 可以定期保存沙箱 snapshot, 用于错误恢复.

- **高效率与高密度.** 在我们的工作负载中, 可能需要在数秒内创建数万个沙箱, 且每个沙箱都拥有独特的镜像集合. 我们采用 OverlayBD [Li20] 作为镜像格式, 并结合自定义 ublk 驱动实现, 存储层共享和 P2P 传输, 在大规模运行时实现亚秒级启动延迟. 我们还通过写时复制内存和页 cache 优化进一步降低内存占用, 在真实工作负载中实现最高 $6.5\times$ 的内存超分比.

在 Kimi K3 的整个训练和评估过程中, 我们基于 1,505,678 个镜像共创建了 51,219,741 个沙箱.

### 5.4 推理与在线服务

Kimi K3 服务在生产侧暴露出同样的挑战: 混合 KDA-MLA 架构维护两种本质不同的 cache, 在百万 token 上下文中必须对其进行联合管理; 新模块和高度稀疏的专家分别需要定制 kernel; 生产流量混合了单请求成本相差三个数量级的各类请求. 下述设计从三个层级处理这些挑战. 在引擎层, 感知 KDA 的前缀 cache 将固定大小的递归状态与 MLA KV cache 放入同一个分页池, 使长前缀能够跨请求复用. 在设备层, 面向 KDA 解码, Block AttnRes 和稀疏潜 MoE 的专用 kernel 可最大限度降低逐 token 延迟和内存传输量. 在集群层, 感知 cache 的亲和调度和基于预算的准入控制将这些效率优势转化为可预测的服务能力.

#### 5.4.1 感知 KDA 的前缀 cache 管理

Kimi K3 的混合架构使前缀 cache 更为复杂: KDA 递归状态与 MLA KV cache 在大小和生命周期上存在本质差异, 但只有二者能够在同一边界同时恢复时, 缓存的前缀才可复用. 因此, 我们设计了感知 KDA 的前缀 cache, 对两种 cache 进行联合管理, 覆盖从统一分页布局到细粒度前缀复用, 再到并发调度一致性的完整流程, 使百万 token 前缀能够以较低成本保留并跨请求复用.

**混合 KDA-MLA 注意力的统一 cache 布局.** 每个 Kimi K3 块都由三个 KDA 层和一个 Gated MLA 层组成, 两者的 cache 存在本质差异. MLA KV cache 随序列长度增长, 并按 token 分页; KDA 递归状态大小固定, 每个请求只有一份副本. 分别维护两个管理器会重复分配, 驱逐和传输逻辑. 因此, 我们将 KDA 状态打包到 MLA KV 使用的同一分页块池中, 统一页面的字节大小, 使两类页面共享同一套分配, 引用计数和驱逐实现. 在页面内部, 所有注意力头的状态逐头连续存储, 使每个头的字节流自包含, 并作为跨节点传输的最小单位. 在 prefill 与 decode 解耦部署下, 如果 prefill 节点和 decode 节点采用不同的 TP 度, 则在传输路径上重新布局, GPU 侧无需重排. 这种不对称性在开发中很有用: 任何混淆类型的访问都会得到无意义数据, 而非看似合理的数据, 相当于对池化布局进行零开销健全性检查.

#### KDA 前缀 cache 优化

基于块哈希的前缀 cache 以单个物理块为粒度复用 KV cache: 只对完整块计算哈希, 因此只能复用与块边界对齐的前缀. 这种耦合在 Kimi K3 中不再适用. 块哈希匹配要求所有层共享同一块大小, 而且只有命中边界处的 KDA 状态已经持久化时, 前缀命中才可复用. KDA 层为每条序列维护单个大型递归状态, 而非逐 token 条目, 因此只能承担在稀疏边界保存状态 snapshot 的成本. 共享块大小由此被迫设为 1024 至 6144 token; 由于哈希与存储块绑定, 哈希粒度也同样如此, 尽管单独考虑 MLA 的逐 token 条目时可以采用更细的块. 在如此粗的粒度下, cache 几乎无用: 短于一个块的请求永远无法复用, 分块 prefill 在跨过完整块边界前也无法导出可缓存前缀.

因此, 我们将两种粒度解耦. 前缀哈希在 MLA 页面内部的细粒度哈希块 (例如 512 token) 上执行, 物理块则仍作为粗粒度分配单位. 对 KDA 来说, 对齐方向恰好相反: 递归状态 checkpoint 只保存在 MLA 哈希端点的稀疏子集上, 而查找也只可能引用这些位置.

<span id="figure-12"></span>

![感知 KDA 的细粒度前缀 cache](./kimi-k3/figure-12.png)

**图 12.** 6144-token 物理 cache 块内的细粒度前缀 cache. 请求复用五个 512-token MLA 哈希块和边界 $B=2560$ 处的 KDA checkpoint, 随后恢复 prefill, 无需重新计算 $[0,B)$.

prefill 期间, 部分填充的 MLA 页面会以最后一个完整哈希块的链式哈希为键, 注册到前缀 cache 索引中. 每个哈希都覆盖此前的所有哈希块, 因而端点匹配可以证明截至该位置的整个前缀均匹配; 注册端点会随页面填充不断前移. 同时, 每次前向传播后, KDA kernel 都会持久化最后一个与哈希对齐位置上的递归状态. checkpoint 较大, 因而请求推进时, 被取代的中间 checkpoint 会被回收, 对话轮次边界处的 checkpoint 则会保留, 供跨请求复用. 缓存的 checkpoint 是只读 snapshot: 命中时, 系统在下一次前向传播前将其复制到请求私有的运行状态中, 新 checkpoint 则写入新槽位, 因此其他请求可见的 checkpoint 绝不会被原地修改.

查找分两个阶段进行 ([图 12](#figure-12)). MLA 阶段通过链式哈希匹配整个物理块, 遇到第一个缺失块时, 则回退到块内的哈希端点, 因而部分填充的页面仍可命中. 随后的 KDA 阶段要求每个 KDA cache group 在候选边界处都存在 checkpoint, 且各 group 分别维护独立的递归状态. 最终命中是同时满足两个阶段的最长边界, 它始终是哈希块大小的整数倍, 但无需为物理块大小的整数倍. 在[图 12](#figure-12) 中, 某个请求的前 2800 个 token 与缓存前缀匹配, 它在 6144-token 物理块内部的 $B=2560=5\times512$ 处命中, 并从 token $B$ 开始恢复 prefill, 而无需重新计算 $[0,B)$.

#### 并发调度下的一致性

其余设计点分别由共享部分填充块时的具体失效模式决定. 在这种设置中, 命中块既是共享 cache 条目, 又是私有请求的增长点, MLA 与 KDA cache group 还必须就每个命中边界达成一致. 首先, 所有 cache group 都从同一个共享空闲列表中取块, 因而为一个 group 分配私有副本时, 可能逐出另一个 group 刚刚命中的块; 所以在进行任何分配前, 都会先在所有 group 中固定每个命中块. 其次, 向私有块的复制会在前向传播前立即于 GPU 上执行, 因此当前调度步骤内分配或注册的块仍会把上一所有者的字节交给读取方; 在复制完成前, 此类块会被排除在匹配范围外. 第三, 只有每个 KDA group 中都存在对应 checkpoint 时, 才能恢复请求. 因而逐出某个 group 的 checkpoint 时, 会原子化地使其他 group 中的对应 checkpoint 失效, 即一个 checkpoint 要么在所有 group 中均可命中, 要么全部不可命中. 借助这些机制, 每个已注册状态始终精确对应其声明的 token 前缀. 混合 KDA-MLA 模型的前缀 cache 也能达到完整注意力模型的通用性: 任何共享前缀都能在任意 512-token 边界复用, 不受请求长度, 分块方式或调度交错影响.

#### 5.4.2 高性能 kernel

Kimi K3 引入了多个新架构模块: KDA ([§2.1.1](#_2-1-1-kimi-delta-attention)), Block AttnRes ([§2.2](#_2-2-attention-residuals)) 和 Stable LatentMoE ([§2.3](#_2-3-stable-latentmoe)). 我们分别优化了它们的 kernel 实现.

**KDA.** 与 KDA prefill ([§5.1](#_5-1-kda-的算法-系统协同设计)) 相比, KDA 解码面临一组不同的挑战: 主要瓶颈从利用并行性转变为高效管理不断变化的递归状态, 该状态会在每个解码步骤中原地更新. 这种原地更新在基于 MTP 的推测解码中会产生问题: 如果验证拒绝部分草稿 token, 状态已经推进到最后一个接受 token 之后, 无法轻易回滚. 为每个草稿位置保存状态 snapshot 可以支持回滚, 但也会使状态传输量成倍增加, 而在在线服务常见的大 batch size 下, 这项成本占据主导地位.

然而, 任意已接受草稿前缀之后的状态都完全由草稿 token 的投影输入决定, 而投影输入远小于状态本身. 因此, 我们只缓存这些投影输入, 在片上重建已接受 token 的状态, 再写回已验证 token 和 bonus token 的状态. 同期工作 ReplaySSM [Dao26] 独立提出了这一设计. 重放 token, bonus token 和下一个草稿窗口在单个融合 kernel 内共享同一个递归循环, 其中涵盖短卷积, 输入归一化, 门控, KDA 递归和输出归一化. 验证延迟随验证 token 数量次线性增长, 且始终低于缓存状态的基线. 由于投影 cache 始终位于 decode 阶段, 前缀 cache 以及 prefill-decode 解耦均可使用与非推测服务相同的 payload.

**Block AttnRes.** Block AttnRes [Kim26a] 采用两阶段调度: batch 化的块间 pass 对每个块读取一次缓存的块表示, 随后每层通过 online-softmax 合并 [Mil18] 融入块内部分和. 在 prefill 和解码中, 内存访问都占这些 kernel 成本的很大部分, 因此我们对两个阶段的优化都主要关注内存效率.

对于 prefill, 在每个张量并行 (TP) rank 上具体化块表示会产生大量冗余内存占用. 因此, 我们对激活采用序列并行 (SP): 将 TP all-reduce 分解为 reduce-scatter 和 all-gather, 在两个集合通信操作之间插入块内 kernel. 该 kernel 对序列分片后的隐藏状态执行操作, 使每个 token 的块表示只在一个 rank 上具体化. 这消除了额外内存占用, 并减少 Block AttnRes 在 prefill 期间的 I/O 开销.

对于解码, 我们在 side stream 上启动块间 kernel, 使其与 main stream 上的独立计算重叠. 块内 kernel 则通过融合得到简化: AttnRes 输出与其部分和更新的合并, 以及随后的 RMSNorm, 均被融合到此前的 TP all-reduce 中, 从而消除块内阶段的专用 kernel. 这些优化共同隐藏了块间 pass 的延迟, 并减少块内阶段的内存传输量.

**Stable LatentMoE.** Stable LatentMoE 同时增加了专家总数和每个 token 激活的专家数. 专家空间和逐 token 专家数量的增长增加了调度与协调开销, 使传统 MoE kernel 难以维持较高的硬件利用率. 这些挑战促使我们为该模块进行专门的 kernel 优化.

为降低潜空间 GEMM 的开销, 我们采用三项优化. 首先, 将潜空间下投影与 MoE 路由器融合为单个 GEMM. 其次, 在各 rank 间对潜空间权重矩阵进行分片, 并使用 multimem store 指令, 将输出 all-gather 融合到 GEMM epilogue 中. 最后, 将由此产生的通信与共享专家计算等其他算子重叠. 这些优化共同消除冗余权重传输和重复计算, 同时以计算隐藏通信延迟.

对于路由专家, 在小 batch size 下, group GEMM 会退化为受内存限制的权重矩阵流式传输. 传统以 tile 为中心的 kernel 由于其面向计算的设计和预处理开销, 并不适合这种执行阶段. 我们改为基于 WarpDecode [Bet26] 以 token 为中心的设计构建 MoE 解码 kernel, 其中每个 warp 负责一个输出神经元, 并直接从内存流式读取相关权重. 为进一步提高并行度, 我们将每个 warp 细分为粒度更小的 lane team, 各自处理互不重叠的专家子集, 再对部分结果执行 warp 级归约. 此外, 我们以一次性预处理成本离线重排权重布局, 从而大幅降低运行时反量化开销.

#### 5.4.3 集群级调度

超出单个服务实例后, 挑战会从单请求效率转向可预测性: 前缀 cache 未命中的成本比命中高出多个数量级, 突发的百万 token 请求还可能使短请求得不到服务. 为此, 我们提出两种集群级调度策略: 感知 cache 的亲和调度将每个会话路由到持有其前缀 cache 的集群, 同时限制集群故障的成本; 基于预算的准入控制为每类请求分配独立资源预算, 防止突发长上下文流量降低整个系统的 SLO.

**感知 cache 的亲和调度.** 在 1M 上下文下, 典型编码输入携带 400K-token 前缀, 但只需要增加 4K token 的 prefill. 因而前缀 cache 命中可以避免对整个前缀重新 prefill, 成本比未命中低多个数量级. 所以, 我们将每个请求路由到持有其前缀 cache 的集群, 因为将 cache 移到另一集群需要通过集群间链路传输, 其速度远低于集群内互连. 然而, 这种感知 cache 的亲和性会将每个会话绑定到单一集群, 一旦该集群故障, 所有绑定会话都会中断. 因此, 一致性哈希会将每个会话固定到两个集群: 一个主集群负责服务其流量, 另一个预先分配的辅助集群则在主集群故障时接管. 辅助集群不持有该会话的任何前缀 cache, 故障转移时必须重新 prefill. 由于一致性哈希会将不同会话的辅助分配均匀分布到整个集群, 重新 prefill 的工作会由多个集群分担, 而非集中在一个集群上. 由此, 常见情况下可以保留 cache 局部性, 任意单一集群故障的影响也始终有界.

#### 基于预算的准入控制

生产流量混合了不足 2K token 的短请求和最长 1M token 的超长请求, 因而单请求成本跨度约为三个数量级, 任意固定数量请求带来的总负载都极难预测. 基于"平均请求"的容量规划, 排队模型和限流配额在这种方差下都会失效. 一种典型失效模式是, 突发长上下文请求占满可用计算资源, 随后到达的短请求无法及时调度, 导致所有流量的首 token 延迟 (TTFT) 恶化. 因此, 我们采用基于预算的准入控制, 为不同请求类别分配独立资源预算, 使突发长上下文流量至多占用自己对应的容量份额, 不会降低其他类别所体验到的全系统 SLO.

## 6 评估

### 6.1 主要结果

#### 6.1.1 基准测试

我们在一个综合基准测试套件上评估 Kimi K3, 该套件按四个主要能力轴线组织:

- **推理与知识:** GPQA Diamond [Rei24], CritPt [Art26], AA-LCR [Art26a] 和 Humanity's Last Exam (HLE-Full, 分为使用工具和不使用工具两种设置) [Pha25].

- **编码:** DeepSWE [Ela26], ProgramBench [Pro26], Terminal-Bench 2.1 [Mer26], FrontierSWE [Fu24], SWE-Marathon [Swe26], PostTrainBench [Pos26], MLS-Bench-Lite [Lyu26] 和 SciCode [Tia24, Art26].

- **智能体:** BrowseComp [Wei25], DeepSearchQA [Ved25], ResearchRubrics [Sha26], Toolathlon-Verified [Too26], MCPMark-Verified [Wu25], MCP-Atlas [Ban26a], AutomationBench [She26], JobBench [Li26], GDPval-AA v2 [Pat25], AA-Briefcase [Art26, Age26], Agents' Last Exam (ALE) [Age26a, Sun26a], APEX-Agents [Vid26], OfficeQA Pro [Ops26], SpreadsheetBench 2 [Zhu26], OSWorld-Verified [Xie25], OSWorld 2.0 [Yua26], SaaS-Bench [Shi26], $\tau^3$-Banking [Ban26, Art26], Harvey Lab-AA [Art26, Har26], CorpFin v2 [Val26], Finance Agent v2 [Fro26] 和 Legal Research Bench [Val26b].

- **视觉:** WorldVQA [Zho26], OmniDocBench [Ouy25], PerceptionBench [Kim26d], Video-MME [Fu24], MMVU [Zha25a] 和配有 Python 工具的 BabyVision [Che26]. MMMU-Pro [Yue24], CharXiv (RQ) [Wan24a], Math-Vision [Wan24] 和 ZeroBench-main [Rob25] 均分为使用和不使用 Python 工具增强两种设置.

#### 6.1.2 基线

我们与当前最佳的闭源和开源模型进行基准比较. 闭源模型包括 Claude Fable 5 [Fab26], GPT-5.6 Sol [Sol26], Claude Opus 4.8 [Opu26] 和 GPT-5.5 [Ope26]. Claude Fable 5 的结果包含 fallback 行为, GPT-5.6 Sol 的结果则可能包含网络安全防护机制. 开源模型包括 GLM-5.2 [Zai26]. 除使用 `xhigh` 设置的 GPT-5.5 外, 所有模型都以最大推理强度接受评估.

#### 6.1.3 评估配置

所有 Kimi K3 评估均使用 `max` 推理强度和温度 $=1.0$. 对于 GPQA Diamond, HLE-Full 和不使用工具的视觉基准等单步任务, 我们设置 top-$p=0.95$. 对于智能体任务, 则设置 top-$p=1.0$. 通常, 我们建议推理与知识任务使用 top-$p=0.95$, 编码和智能体场景使用 top-$p=1.0$.

**编码.** 每个模型都在 Kimi Code [Kim26], Claude Code [Cod26] 或 Codex [Cod26a] 三种智能体 harness 之一上评估. 对于 DeepSWE, 我们报告 v1.1 任务上的结果, 并补充引用官方排行榜结果 (Kimi K3 使用 mini-SWE-agent harness 时取得 67.3 分). 对于 Terminal-Bench 2.1, 我们报告所有模型跨 harness 的最佳分数. SWE-Marathon 评估基于 2026 年 7 月 9 日官方任务中针对 H20 校准的分支, 早于最终 v1.1 版本. 其中 GPU 任务的 Docker 镜像, 性能门槛和参考 oracle 针对 H20 重新校准, 正确性与反作弊验证器则保持不变; Claude Fable 5 在 35% 的任务中触发 fallback. 对于 PostTrainBench, 我们使用官方 Harbor 实现, 以最大强度评估 Kimi K3, Claude Fable 5 和 GPT-5.6 Sol, 结果是在 H20 GPU (官方设置使用 H100) 上运行三次的平均值. FrontierSWE dominance 分数使用截至 2026 年 7 月 16 日的官方评估脚本, 根据原始分数重新计算.

**智能体.** 对于 OfficeQA Pro, 每个测试用例都向智能体提供渲染为图像的完整 PDF 语料库, 不提供机器可读文本. MCP-Atlas 在包含 500 项任务的公开子集上评估, 轮次上限为 100, 并使用 Gemini 3.1 Pro 作为裁判. AutomationBench 在包含 600 项任务的公开子集上评估. 对于 BrowseComp, 我们采用在 300K token 时触发的上下文压缩策略; 如果使用完整 1M-token 上下文窗口且不进行上下文管理, Kimi K3 得分为 90.4%.

**视觉.** 分数为三次运行的平均值, 但 ZeroBench-main 沿用官方设置运行五次. MMMU-Pro 遵循官方协议, 保留原始输入顺序, 并将图像置于文本输入之前. 对于 WorldVQA, 我们观察到各模型都会稳定拒答, 因此通过提示工程强制模型给出答案.

#### 第三方结果

GDPval-AA v2, AA-Briefcase, $\tau^3$-Banking, Harvey Lab-AA, APEX-Agents, SciCode, AA-LCR 和 CritPt 的分数引用自 Artificial Analysis [Art26], 截止日期为 2026 年 7 月 23 日. 对于 Harvey Lab-AA, 我们报告标准通过率. CorpFin v2, Finance Agent v2 和 Legal Research Bench 的分数引用自 Vals AI [Val26c]. Agents' Last Exam 的分数引用自截至 2026 年 7 月 23 日的官方排行榜 [Age26a], 我们报告排行榜的主要通过率指标. 排行榜中, 每个模型都与特定 harness 配对: Kimi K3 使用 Kimi Code; GPT-5.6 Sol 和 GPT-5.5 使用 Codex; Claude Fable 5, Claude Opus 4.8 和 GLM-5.2 使用 Claude Code. Toolathlon-verified 和 JobBench 的分数引用自截至 2026 年 7 月 24 日的各自官方排行榜 [Too26, Job26].

#### 6.1.4 结果

<span id="table-02"></span>

![公开基准测试对比](./kimi-k3/table-02.png)

**表 2.** Kimi K3 与闭源模型及开放权重模型的性能对比. 粗体表示各基准的最佳结果, 下划线表示第二名.

[表 2](#table-02) 全面比较了 Kimi K3 与闭源和开源基线. 总体而言, Kimi K3 紧随最强闭源模型 Claude Fable 5 和 GPT-5.6 Sol, 同时在整个基准测试套件中稳定优于 Claude Opus 4.8, GPT-5.5 和 GLM-5.2. 下面列出各核心能力领域的关键观察:

**推理与知识.** 在研究生水平的推理任务上, Kimi K3 具备前沿竞争力, GPQA Diamond 得分为 93.5%. 然而, 在研究级任务上仍存在差距: HLE-Full 中使用和不使用工具时的得分分别为 56.0% 和 43.5%, 均落后于 Claude Fable 5 与 GPT-5.6 Sol; CritPt 得分为 23.4%, 落后于 Claude Fable 5, GPT-5.6 Sol 和 GPT-5.5. 这说明研究级推理仍是需要重点改进的方向.

### 6.2 内部评估

#### 6.2.1 能力评估

除公开基准测试套件外, 我们还维护了一组内部基准, 面向公开评估覆盖不足的能力领域, 以更全面地衡量模型与智能体能力. 这些基准会频繁更新和扩展, 从而紧密追踪模型不断变化的失效模式, 并直接指导数据与训练迭代. 它们大致分为三类: 编码能力与体验, 通用智能体体验, 以及对话体验. [表 3](#table-03) 报告了这些基准上的结果.

<span id="table-03"></span>

![内部基准测试结果](./kimi-k3/table-03.png)

**表 3.** 内部基准测试套件上的结果. 粗体表示各基准中报告的最佳结果, 短横线表示报告未包含该分数.

#### 编码能力与体验

- **Kimi Code Bench 2.0 (KCB 2.0):** 在真实的端到端软件工程任务上评估编码智能体, 涵盖广泛的编程语言和面向生产的技术栈.

- **Kimi Webdev Bench:** 使用来自真实使用场景的高难度 Web 开发提示评估模型, 通过专家盲评比较输出, 结果见[表 4](#table-04).

<span id="table-04"></span>

![Kimi Webdev Bench 结果](./kimi-k3/table-04.png)

**表 4.** Kimi Webdev Bench 结果: 专家对代码质量, 功能完整性, 视觉保真度和交互体验进行盲评时, Kimi K3 与 Claude Opus 4.8 的对比.

- **Coding Experience:** 评估在真实开发工作流中将模型用作编码智能体的实际体验.

#### 通用智能体体验

- **24/7 ClawBench 2.0:** 模拟全天候助理工作, 其中任务跨越多日, 事件并发到达, 中断也很常见.

- **Multi-Agent Infra for Routing and Assignment (MIRA) Bench:** 评估长链条, 多角色, 多系统的企业协作任务, 衡量智能体能否执行端到端工作, 并判断何时应组织子智能体或向其委派任务.

- **Kimi Autonomous Execution Tasks (KAET):** 在模拟真实用户请求和企业系统操作的任务上评估长程自主执行能力.

- **Context Learning and Instruction Following (CLIF) Bench:** 评估上下文学习, 要求模型从给定上下文中学习, 同时遵循交错使用多项复杂技能的指令.

- **Agentic Vision Bench:** 评估智能体在执行任务期间能否注意到关键视觉事实, 并正确利用这些事实.

- **Swarm Bench:** 在适合协同分解与并行执行的复杂任务上, 评估模型编排智能体集群 [Kim26b] 的能力.

- **Online Experience:** 模拟真实在线智能体的使用分布, 衡量模型在用户最常请求的交付文件类型上的表现.

- **Deep Research Bench:** 使用领域专家整理的深度研究式查询评估模型, 并依据与专家判断对齐的评分标准打分.

- **Finance Bench:** 在真实金融工作上评估模型, 要求模型端到端执行从来源材料到可审查交付物的完整工作流.

- **Knowledge Work Vision (KWV) Bench:** 评估从真实知识工作场景蒸馏的任务中提取出的原子视觉能力.

- **DECK Bench:** 衡量根据真实使用场景中的任务描述制作高质量演示文稿的能力.

- **Agent Behavior Bench:** 将智能体评估从结果正确性扩展到过程质量, 在任务完成情况之外, 还评估工具使用行为, 效率和规范性.

#### 对话体验

- **Faithfulness:** 衡量模型响应中的事实幻觉率, 每个响应都由事实核查器验证.

- **Chat All-in-One Bench:** 衡量产品使用各阶段的对话体验, 场景围绕真实在线用户需求设计.

**评估配置.** 除按 harness 拆分为不同行的基准外, [表 3](#table-03) 中的 Harness 列均报告 Kimi K3 使用的 harness. 对于其他模型, Claude 系列模型和 GLM-5.2 使用 Claude Code 评估, GPT 系列模型使用 Codex 评估. 例外是所有模型都使用同一指定 harness 的基准: 24/7 ClawBench 2.0 使用 OpenClaw; MIRA Bench 使用内部 OOD harness MIRA (Multi-Agent Infra for Routing and Assignment); Agent Behavior Bench 和 Chat All-in-One 使用 Kimi Work; CLIF 和 Agentic Vision Bench 使用 Kimi Code.

**结果.** 相较公开基准, 内部套件更清楚地区分了 Kimi K3 的优势与弱点. 最明显的优势是编排型与研究型智能体能力: Kimi K3 在 Swarm Bench (76.3) 和 Deep Research Bench (90.0) 上以明显优势领先, 表明其具备分解复杂目标, 协调并行工作和生成符合评分标准的交付物的强大能力. 编码同样是其优势: Kimi Code Bench 2.0 上仅落后于 Claude Fable 5, Coding Experience 则取得最高分, 说明其作为编码智能体的实际行为, 包括沟通质量, 行为适当性和指令遵循稳定性, 优于原始任务分数所反映的水平. 在 Kimi Webdev Bench 上, 专家对 Kimi K3 的偏好相较 Claude Opus 4.8 高出 31.0 个百分点, 其中 3D/WebGL/Shader 任务的提升最大. 专业知识工作相较上一代也有显著改善, Finance Bench 上基本与 GPT-5.6 Sol 持平.

Kimi K3 主要在 Agent Behavior Bench, MIRA Bench, 24/7 ClawBench 2.0, Agentic Vision Bench 和 KWV Bench 上落后于领先模型. 在其余报告了结果的套件 (KAET, CLIF Bench, Online Experience, DECK Bench, Faithfulness 和 Chat All-in-One Bench) 上, Kimi K3 排名第一或以很小差距位列第二.

#### 6.2.2 网络安全评估

我们沿操作风险逐级上升的两个级别评估模型的网络安全能力: 漏洞发现与概念验证开发 (Tier 1), 以及端到端 exploit 开发 (Tier 2). 评估目标包括广泛部署的软件最新版本, 如操作系统 kernel 组件和开源项目, 以及包含生产服务与代码库在内的内部基础设施. 所有任务都在能代表真实部署的标准配置下运行. Anthropic 和 OpenAI 的前沿模型会拒绝网络安全相关任务, 无法进行可比评估, 因此未将其纳入该套件.

#### 漏洞发现 (Tier 1)

该级别要求模型识别当前代码库中的真实缺陷, 而非复现已知漏洞, 并证明这些缺陷可以复现. 这些能力主要与防御性安全研究相关.

在涵盖操作系统 kernel, 数据库, AI 服务, Web 框架, 区块链和 VPN 软件的数十个广泛部署系统中, 模型识别出数百个候选漏洞. 经过人工审查的发现中, 约 70% 被确认为真实漏洞, 其中包括六个项目中的 16 个此前未知漏洞.

Linux kernel 中的两项发现体现了这些结果的深度. 首先, 模型识别出一个可远程触发的堆越界写入. 该缺陷由不完整的上游修复引入, 影响此后的所有版本, 直至最新上游代码. 安全专家确认它是一种远程拒绝服务原语. 其次, 模型在 RDMA 子系统中识别出一个 Dirty COW 级漏洞: 先前的上游修复意外遗漏了一项权限检查, 使 kernel 侧可以写入只读内存页. 安全专家确认它是一种确定性的本地权限提升原语.

**Exploit 开发 (Tier 2).** 该级别要求模型将漏洞转化为可以工作的端到端 exploit, 与滥用风险最直接相关. 我们以 GLM-5.2 为基线, 使用包含两个方向, 共 36 项任务的内部套件进行评估.

**用户空间利用 (16 项任务).** 模型必须在 PostgreSQL, XWiki 协作平台, Apache HTTP Server, 多个内容管理系统及其他应用等广泛部署的用户空间软件中, 端到端利用真实 CVE. 每项任务都会向模型提供完整源代码和运行中的实例; 目标以标准配置运行, 不使用额外加固.

**Linux kernel 利用 (20 项任务).** 每项任务都提供一个根据历史 kernel CVE 构建的可复现 QEMU 环境, 模型必须编写 C exploit, 将权限从无特权用户提升至 root. 不同难度等级会逐步启用缓解措施.

该套件中的每项任务都经人工安全专家验证为可解. 我们估计, 完成整个套件约需 540 个专家工时, 即每项任务平均约 15 小时.

**Exploit 套件结果.** 模型在该套件上展现出实质性的 exploit 开发能力, 解决了 36 项任务中的 14 项 (38.9%), 而 GLM-5.2 解决了 8 项 (22.2%). 不过, 成功案例的分布并不均匀: 14 项中有 10 项来自用户空间方向. 在 kernel 方向上, 两个模型都有四分之三的任务未能解决.

由于每项任务都能由人工专家解决, 未解决的任务直接衡量了模型与人类能力之间仍然存在的差距. 轨迹分析将这一差距归因于四种反复出现的失效模式: (i) 难以利用已经获得的原语, 完成 exploit chain 的最后阶段; (ii) 面对缓解措施时策略选择不佳, 例如在纯数据攻击更简单, 更可靠时, 仍坚持劫持控制流; (iii) 陷入长时间且无效的调试循环; (iv) 提交前未充分验证最终交付物.

**总结.** 模型的网络安全能力在 Tier 1 和 Tier 2 的用户空间利用中最强, 但与人工专家之间仍存在明显差距. 在防御性质的 Tier 1 中, 模型能识别真实漏洞, 包括此前未知的漏洞, 并证明它们可以复现. 在 Tier 2 中, 模型能针对用户空间目标完成端到端 exploit. 然而, 面对加固后的目标时, 完成完整 exploit chain 仍是瓶颈, 许多专家可以解决的任务仍未完成.

英国 AI Security Institute 与 NIST Center for AI Standards and Innovation (CAISI) [Pre26] 的独立联合评估得出了与我们一致的结论. Kimi K3 的 exploit 开发能力优于 GLM-5.2 (ExploitBench 上为 32% 对 24%; 在一项包含 32 步, 人类专家约需 20 小时完成的模拟企业网络任务中完成 17 步, GLM-5.2 完成 11 步), 但端到端 exploit 完成能力落后于具备前沿网络安全能力的模型, 在 41 项任务中没有一项实现任意代码执行.

我们认为该评估结果只是能力下界. 这些结果取决于当前模型版本和评估覆盖范围, 我们会在每次模型重大更新时重新评估.

### 6.3 第三方评估

Kimi K3 发布后, 多家第三方机构也对其进行了独立评估. [表 5](#table-05) 汇总了截至 2026 年 7 月 23 日的主要结果.

<span id="table-05"></span>

![独立第三方评估结果](./kimi-k3/table-05.png)

**表 5.** 截至 2026 年 7 月 23 日, Kimi K3 的主要独立第三方评估结果.

#### Artificial Analysis

Artificial Analysis 对 Kimi K3 进行了评估 [Art26]. Kimi K3 的 Intelligence Index v4.1 得分为 57.1, 在 580 个模型中排名第四; 如果将 GPT-5.6 Sol 的不同推理强度变体视为同一项, 则排名第三. 它落后于 Claude Fable 5 (59.9) 和 GPT-5.6 Sol (58.9), 领先于其他所有受评模型.

**Vals AI.** 在 Vals AI 按 GDP 加权的行业基准测试套件 [Val26c] 上, Kimi K3 的 Vals Index 为 74.7%, 在 39 个模型中排名第二, 落后于 Claude Fable 5 (75.1%), 领先于 GPT-5.6 Sol (73.1%).

**Arena.** 在众包人类偏好 arena [Lea26] 上, Kimi K3 在 WebDev Arena 的 99 个模型中排名第一, Elo 为 1,678, 高于 Claude Fable 5 的 1,634, 成为第一个登顶该排行榜的开放模型; 在 Text Arena 的 200 个模型中以 1,486 Elo 排名第八. Agent Arena 约于 7 月 19 日开放投票, Kimi K3 目前在 37 个模型中排名第四 (9.1), 落后于 Claude Fable 5 (12.7), GPT-5.6 Sol (10.1) 和 Claude Opus 4.8 (9.8).

### 6.4 成本效率

除分数外, 我们还在四个涵盖编码与智能体任务的套件上比较分数和单任务成本, 以研究推理成本效率. 这四个套件分别为 Kimi Code Bench 2.0, BrowseComp, GDPval-AA v2 和 AA-Briefcase. Kimi Code Bench 2.0 的成本由内部测量, 其中 Kimi K3 通过 Kimi Code 运行, 其他所有模型通过 Claude Code 运行.

对于 BrowseComp, Kimi K3 的成本来自我们自己的运行结果, Claude 与 GPT 的成本则引用自已发布图表 [Sol26, Son26, Son26a]. 对于 GDPval-AA v2 和 AA-Briefcase, 成本引用自截至 2026 年 7 月 23 日 Artificial Analysis 按 token 计费的 API 价格 [Art26].

在 Kimi Code Bench 2.0 上, Kimi K3 的分数比 Claude Fable 5 低 4.0 分, 成本则只有后者的 38%; 在 `high` 强度下, Kimi K3 已能以约三分之一的成本达到 Claude Opus 4.8 最大强度下的分数. 在 BrowseComp 上, Kimi K3 以每项任务 \$2.03 的成本取得最高分 (91.2%), 成本为 GPT-5.6 Sol (90.4%) 的一半, 比最大强度下的 Claude 模型低一个数量级. 在 GDPval-AA v2 上, Kimi K3 与 GPT-5.6 Sol 的差距不到 50 Elo, 成本低 13%, 同时成本比 Claude Fable 5 低 $2.6\times$. 在 AA-Briefcase 上, Kimi K3 得分仅次于 Claude Fable 5, 成本约为后者的一半. [图 13](#figure-13) 汇总了这项比较. 总体而言, Kimi K3 在全部四个套件上均处于或接近成本效率前沿, 尤其是只需 Claude Fable 5 一小部分的成本, 即可取得接近最高水平的分数.

<span id="figure-13"></span>

![分数与单任务推理成本](./kimi-k3/figure-13.png)

**图 13.** Kimi Code Bench 2.0, BrowseComp, GDPval-AA v2 和 AA-Briefcase 上的分数与单任务推理成本. Kimi K3 以星号标记.

## 7 案例研究

本节介绍若干代表性案例, 展示 Kimi K3 在不同技术任务上的能力.

#### GPU kernel 优化

我们测试了各模型优化 GPU kernel 的能力. 每个模型都在配置完全相同的沙箱中独立工作, 每项任务拥有最多 24 小时的预算, 用于 profiling, 重写和基准测试. 评估涵盖四种代表性 kernel: AttnRes, DeepSeek Sparse Attention (DSA), KDA 和注意力头维度为 512 的 MLA, 硬件则包括 NVIDIA Hopper GPU 和另一厂商的 GPGPU. Kimi K3 大幅提高了四种 kernel 的性能: 将 AttnRes 延迟从 283.6 ms 降至 114.4 ms, 将 DSA 和 KDA 的运行时间分别缩短 55.1% 和 73.6%, MLA 则达到超过一半的峰值 TFLOPS. 在这些任务上, Kimi K3 与包含 fallback 的 Claude Fable 5 [Fab26] 持平, 并大幅优于 Claude Opus 4.8 [Opu26], GPT-5.6 Sol [Sol26] 和 GPT-5.5 [Ope26]. [图 14](#figure-14) 比较了各模型在 AttnRes 上的优化轨迹. 在基准测试之外, 开发后期的早期 Kimi K3 checkpoint 已经承担了我们的大部分 kernel 优化工作.

<span id="figure-14"></span>

![GPU kernel 优化轨迹](./kimi-k3/figure-14.png)

**图 14.** AttnRes 的 GPU kernel 优化轨迹.

**GPU 编译器开发.** Kimi K3 开发了 MiniTriton ([代码仓库](https://github.com/MoonshotAI/minitriton)), 这是一个紧凑的类 Triton [Til19] 编译器, 包含自定义 tile 级 Python 前端与布局系统, 轻量级 warp 级 MLIR [Lat21] 注解与优化层, 以及 Parallel Thread Execution (PTX) 代码生成流水线. 以该编译器为核心构建的是一个双模式张量库, 它提供类似 PyTorch [Pas19] 的高级接口, eager 路径和仅前向编译路径共享同一个 DSL 编译器和 runtime. 该库还提供反向模式自动微分, 神经网络模块, 基于 NCCL [Ncc15] 的分布式训练原语, 以及稀疏和可视化原语. 在 NVIDIA L20 上, MiniTriton 核心基准测试套件的几何平均性能优于 PyTorch eager [Pas19] 和 `torch.compile` [Ans24]. 从零构建的 Tensor Core matmul 路径在最大形状下接近 cuBLAS [Cub16], 达到实测机器 roofline 的约 90%; DSL 级 KDA [Kim25b] prefill kernel 则明显优于配置匹配的 Triton 参考实现. MiniTriton 还能端到端训练 GPT 模型, 其损失曲线紧随 PyTorch 参考实现. 相对于 FP64 参考, 完整模型梯度与 torch autograd 的差异不超过 torch 自身的 FP32 舍入误差 $10^{-4}$. 这些结果共同表明, Kimi K3 能够构建从 DSL 前端, IR pass 到 PTX 代码生成和 CUDA runtime 的完整一致端到端编译器, 而非一组彼此孤立的 kernel ([图 15](#figure-15)).

<span id="figure-15"></span>

![MiniTriton GPU 编译器结果](./kimi-k3/figure-15.png)

**图 15.** 使用 MiniTriton 开发 GPU 编译器: CUDA Core 与 Tensor Core roofline, GPT 训练损失, 以及双 GPU 数据并行训练.

#### 芯片设计

作为早期概念验证, Kimi K3 为采用同类架构的 nano 模型设计了推理芯片原型. 该模型使用混合 KDA 与 NoPE-MLA 注意力, 块大小为 2 的 Block AttnRes, 以及包含一个共享专家的 sigmoid 式 MoE 路由, 权重采用组级 INT4 量化 (组大小为 128). 在使用 Kimi Code 进行的一次 48 小时自主运行中, Kimi K3 使用开源 EDA 工具和 Nangate45 标准单元库 [Nan10] 构建, 优化并验证了芯片. 在 $4\,\mathrm{mm}^2$ 的分析面积预算内, 该设计在 100 MHz 下实现 timing closure, RTL 模拟解码吞吐量超过 8,700 tokens/s, 集成 1.46M 个标准单元, 0.277 MiB SRAM 和带融合反量化的 INT4 MAC array. RTL 代码已发布在 GitHub 上 ([代码仓库](https://github.com/MoonshotAI/nano-kpu)).

#### 面向研究的编码

为复现计算天体物理学中的 I-Love-Q 普适关系, Kimi K3 阅读了 20 多篇论文并交叉验证其结果, 实现完整的数值流水线, 评估 300 多个状态方程, 识别已发表公式中的不一致之处, 编写 3,000 多行 Python, 并生成交互式 HTML dashboard. 整个过程约用两小时, 而经验丰富的研究人员通常需要一至两周.

#### 知识工作

在 Kimi Work 中, Kimi K3 制作了一个涵盖 AI ASIC 行业 42 年历史的交互式研究网站. 模型完成 120 多轮迭代改进, 通过 2,800 多次 Web 搜索和 1,100 多次终端查询, 使用包含 87 份季度报告和 99 份原始 PDF, 总计超过 11,000 页的语料库. 在另一个案例中, Kimi K3 使用 20 多个并发子智能体分析 GWTC-5 中的 391 个引力波事件, 生成七项科学可视化, 两张汇总表, 以及对十余篇论文的文献综述.

**视频剪辑与动态设计.** 借助原生多模态架构, Kimi K3 制作了一个以 3Blue1Brown 风格动态画面讲解自身架构的视频, 还使用 56 个源片段剪辑了预告片. 工作内容包括片段选择, 动作匹配剪辑, 逐帧精确的节拍同步, 音频处理和多轮修改. 制作一段相近的高密度短视频, 通常需要经验丰富的剪辑师工作一至两天.

## 8 结论

我们提出 Kimi K3, 一个基于 Kimi Delta Attention 和 Attention Residuals 构建的开放混合专家模型, 拥有 2.8T 参数, 原生视觉能力和 100 万 token 上下文窗口. 作为全球首个开放的 3T 级模型, Kimi K3 在长程编码, 智能体, 知识, 推理和视觉任务上达到前沿水平. 尽管与最强闭源模型之间仍有差距, Kimi K3 建立了一个人人可及的全新开放前沿. 我们希望它能赋能更广泛的社区, 推动研究, 部署与创新.

## A 贡献者

完整贡献者名单见原始 PDF, 并按姓氏字母顺序排列.

## B Sigmoid Tanh Unit GLU 细节

SiTU-GLU ([§2.3.2](#_2-3-2-sigmoid-tanh-unit-glu)) 的设计目标是在不丢弃 Swish 特征形状的情况下, 为 SwiGLU 乘积设置边界. 这些特征包括原点附近的近似线性响应, 以及趋于零的负半轴尾部. [图 4](#figure-04) 展示了门分支, 上投影分支和各自完整的标量响应.

#### 平滑限制两个分支

SiTU 将 Swish 的线性因子限制为 $\beta_1\tanh(\mathbf W_g\mathbf x/\beta_1)$, 同时保留 sigmoid 因子 [Kim26c]. 由于 sigmoid 已经使门的负响应趋于零, 这一变化主要控制较大的正激活, 而不会消除负半轴尾部. Kimi K3 将同样的结构应用于上投影分支, 写作 $\beta_2\tanh(\mathbf W_u\mathbf x/\beta_2)$, 防止任一分支主导乘积.

**局部与极限行为.** 对于原点附近的标量 $z$, 缩放 tanh 满足

$$
\beta\tanh(z/\beta)=z+O(z^3/\beta^2).
\tag{18}
$$

因此, SiTU-GLU 在原点附近与 SwiGLU 一阶匹配. 当 $\beta_1,\beta_2\to\infty$ 时, 它还会逐点恢复为 SwiGLU.

**有界输出.** 由于 $|\tanh(z)|<1$ 且 $0<\operatorname{Sigmoid}(z)<1$, 每个输出坐标均满足

$$
\|\operatorname{SiTU\text{-}GLU}(\mathbf x)\|_\infty\le\beta_1\beta_2=100.
\tag{19}
$$

这里 $\beta_1=4$, $\beta_2=25$. 与硬截断门的预激活不同, 平滑限制在饱和边界外保留非零梯度. 我们发现, 这样可以获得更好的训练行为.

## C Quantile Balancing 的推导

本附录沿用 [Su26], 从最优均衡分派推导[第 2.3 节](#_2-3-stable-latentmoe)使用的 Quantile Balancing (QB) 更新. 从分派角度研究专家负载均衡可以追溯到 BASE Layers [Lew21] 和 BIP [Sun25a]. 设 $\mathbf s\in\mathbb R^{m\times n}$ 汇集 $m$ 个 token 在 $n$ 个专家上的路由分数, 每个 token 恰好选择 $k$ 个专家, $x_{i,j}\in\{0,1\}$ 表示 token $i$ 是否被分派给专家 $j$. 在每个专家恰好服务 $mk/n$ 个 token, 且假定该值为整数的情况下, 最大分数均衡分派为

$$
\max_{\mathbf x;\,x_{i,j}\in\{0,1\}}\sum_{i,j}x_{i,j}s_{i,j}
\quad\text{s.t.}\quad
\sum_jx_{i,j}=k,\qquad\sum_i x_{i,j}=\frac{mk}{n}.
\tag{20}
$$

**线性松弛与对偶.** 将 $x_{i,j}\in\{0,1\}$ 松弛为 $x_{i,j}\in[0,1]$ 后, 式 20 变为线性规划. 由二分图 $b$-matching 多面体的标准整数性可知, 其最优解为整数, 因而该松弛是精确的. 分别为 token 侧和专家侧的等式约束引入自由乘数 $\alpha_i$ 和 $\beta_j$, 可将松弛后的问题写为如下 max-min 形式

$$
\max_{\mathbf x;\,x_{i,j}\in[0,1]}\min_{\boldsymbol\alpha,\boldsymbol\beta}
\sum_{i,j}x_{i,j}s_{i,j}
-\sum_i\alpha_i\left(\sum_jx_{i,j}-k\right)
-\sum_j\beta_j\left(\sum_i x_{i,j}-\frac{mk}{n}\right).
\tag{21}
$$

目标函数分别关于 $\mathbf x$, $\boldsymbol\alpha$ 和 $\boldsymbol\beta$ 线性, 可行集为凸集, 因此由极小极大定理可交换优化顺序:

$$
\min_{\boldsymbol\alpha,\boldsymbol\beta}\max_{\mathbf x;\,x_{i,j}\in[0,1]}
\sum_{i,j}x_{i,j}(s_{i,j}-\alpha_i-\beta_j)
+k\sum_i\alpha_i+\frac{mk}{n}\sum_j\beta_j.
\tag{22}
$$

内部最大化可按元素分离. 当 $s_{i,j}-\alpha_i-\beta_j>0$ 时, $x_{i,j}^*=1$; 当 $s_{i,j}-\alpha_i-\beta_j<0$ 时, $x_{i,j}^*=0$. 值相同的情形在实践中测度为零. 代入 $\mathbf x^*$ 得到凸对偶目标

$$
\min_{\boldsymbol\alpha,\boldsymbol\beta}\mathcal L(\boldsymbol\alpha,\boldsymbol\beta):=
\sum_{i,j}\max(0,s_{i,j}-\alpha_i-\beta_j)
+k\sum_i\alpha_i+\frac{mk}{n}\sum_j\beta_j.
\tag{23}
$$

```pseudocode:line-numbers title="算法 1: 交替 QB solver"
输入: 分数矩阵 s in R^(m x n)
输出: 分派 x in {0,1}^(m x n)

初始化 beta = 0
for t = 1, 2, ..., T:
  alpha <- desc_sort(s - beta, axis=1)[:, k:k+1]
  beta  <- desc_sort(s - alpha, axis=0)[mk/n:mk/n+1]
返回 x, 其中 x[i,j] = 1 当且仅当 j 在 argtop_k(s[i] - beta) 中
```

**精确坐标最小化.** 我们交替固定 $\boldsymbol\beta$ 求解 $\boldsymbol\alpha$, 再固定 $\boldsymbol\alpha$ 求解 $\boldsymbol\beta$, 以最小化式 23. 每个子问题都有闭式精确解. 固定 $\boldsymbol\beta$ 后, 问题可以按 token 解耦, 对 token $i$ 求解

$$
\min_\alpha k\alpha+\sum_j\max(0,s_{i,j}-\beta_j-\alpha).
\tag{24}
$$

该目标关于 $\alpha$ 分段线性, 其斜率为 $k$ 减去超过 $\alpha$ 的 margin $s_{i,j}-\beta_j$ 的数量. 因此, 当恰有 $k$ 个 margin 大于 $\alpha$ 时, 目标精确达到最小值, 即 $\alpha_i^*$ 可以取 $\mathbf s_i-\boldsymbol\beta$ 中第 $k$ 大与第 $(k+1)$ 大元素之间的任意值. 按惯例, 我们取第 $(k+1)$ 大元素, 等价于第 $(1-k/n)$ 分位数:

$$
\alpha_i^*=\operatorname{quantile}_{1-k/n}(\mathbf s_i-\boldsymbol\beta).
\tag{25}
$$

类似地, 固定 $\boldsymbol\alpha$ 后, 专家 $j$ 求解 $\min_\beta \frac{mk}{n}\beta+\sum_i\max(0,s_{i,j}-\alpha_i-\beta)$. 其最小值点为 $\mathbf s_{:,j}-\boldsymbol\alpha$ 中第 $(mk/n+1)$ 大元素, 同样是第 $(1-k/n)$ 分位数:

$$
\beta_j^*=\operatorname{quantile}_{1-k/n}(\mathbf s_{:,j}-\boldsymbol\alpha).
\tag{26}
$$

因此, 两种更新分别沿 token 轴和专家轴计算同一分位数, 这也是该方法名称的由来. [图 5](#figure-05) 将专家侧更新展示为使各专家 margin 分布中被接受的上尾部分均衡, 算法 1 则总结了最终的交替 solver.

**从分派到路由.** 在式 23 的最优解处, $x_{i,j}^*=1$ 当且仅当 $s_{i,j}-\alpha_i^*-\beta_j^*>0$. 再结合 token 约束 $\sum_jx_{i,j}^*=k$, 可知选中的专家恰好是 $\mathbf s_i-\boldsymbol\beta^*$ 中的 Top-$k$ 项. 因此, 路由只需要专家阈值 $\boldsymbol\beta\in\mathbb R^n$, 等价于式 13 中的偏置 $\mathbf b=-\boldsymbol\beta$. token 阈值 $\boldsymbol\alpha\in\mathbb R^m$ 则是与动态训练 batch 绑定的中间变量, 可以丢弃. 这种不对称性维持了训练与推理的一致性: 部署时, 路由是使用冻结偏置的固定 Top-$k$ 选择, 无需计算分位数.

**与基于符号的无损失更新之间的关系.** 式 26 所对应专家侧子问题的次梯度为

$$
\frac{\partial\mathcal L}{\partial\beta_j}=\frac{mk}{n}
-\sum_{i=1}^{m}\chi(s_{i,j}-\alpha_i-\beta_j>0).
\tag{27}
$$

即目标负载减去专家 $j$ 的观测负载. 在该目标上执行一次 SignSGD, 可恢复无辅助损失均衡 [Dee24a] 的固定步长符号更新, 二者只相差符号约定 $\mathbf b=-\boldsymbol\beta$: 符号更新只保留式 27 中负载误差的方向, QB 则直接跳转到同一对偶目标的精确坐标最小值点. 这一视角既解释了 QB 为何不需要类似学习率的超参数, 也解释了即使面对近 $10^3$ 个专家, 它为何只需几个更新步骤就能达到均衡. QB 同样与 BIP [Sun25a] 有关. BIP 使用不等式约束 $\sum_jx_{i,j}\le k$ 和 $\sum_i x_{i,j}\le mk/n$ 求解同一分派问题; 由此对 $\boldsymbol\alpha$ 和 $\boldsymbol\beta$ 引入的非负约束, 会为两个更新加入 $\max(0,\cdot)$ 截断. 这种截断只能抑制过度选中的专家, 无法促进选择不足的专家, 在我们的实验中会明显减慢均衡过程. 最后, 所得固定 Top-$k$ 路由与专家特定阈值路由相关, 但不同于 Expert Threshold 路由. 后者维护 EMA 阈值, 并允许每个 token 选择数量可变的专家 [Sun26].

## D 基于直方图的分位数估计

式 14 的 QB 更新要求计算整个训练步骤上的分位数: 对 $n$ 个专家中的每一个, 计算 margin $s_{i,j}-\alpha_i$ 的第 $(1-k/n)$ 分位数, 其中 token 数 $m$ 涵盖分片到各数据并行 rank 和梯度累积步骤中的数百万 token. 在训练循环中收集 $O(mn)$ 个 margin 以计算精确分位数并不可行.

关键观察是, 更新不需要 margin 本身, 只需要各专家的 margin 分布, 而直方图能以固定成本汇总该分布. 因此, Kimi K3 为每个专家维护一个分 bin 直方图, 并从中读取分位数. 具体而言, 我们为所需偏置 $r_{i,j}:=\alpha_i-s_{i,j}$ 绘制直方图, 该偏置可使专家 $j$ 恰好位于 token $i$ 的阈值处. 对 margin 取负会反转其顺序, 因而式 14 中的 QB 目标 $b_j$ 恰好是 $\mathbf r_{:,j}$ 的第 $(k/n)$ 分位数.

#### 分 bin 范围

首先需要确定分 bin 的区间, 此处所需偏置可以提供帮助: 其范围由当前偏置本身限制. 路由器分数是 sigmoid 输出, 因此 $s_{i,j}\in(0,1)$. 阈值 $\alpha_i$ 本身是某个专家 $j'$ 的带偏置分数 $s_{i,j'}+b_{j'}$, 因而位于 $(b_{\min},1+b_{\max})$ 内, 其中 $b_{\min}$ 和 $b_{\max}$ 是当前偏置的两个极值. 所以, 每个 $r_{i,j}$ 都位于 $[b_{\min}-1,b_{\max}+1]$ 内. 我们将该区间划分为 $B$ 个均匀 bin, 并在每个步骤重新计算范围. 这样, 随着偏置扩散以修正不均衡, bin 宽度 $w=(b_{\max}-b_{\min}+2)/B$ 也能不断适应.

#### 累积与恢复

其余流程遵循训练步骤的结构. 在每次前向传播期间, 每个 rank 将本地 $r_{i,j}$ 值 scatter-add 到逐专家计数矩阵 $\mathbf H\in\mathbb N^{n\times B}$, 无需通信即可在所有 micro-batch 上累积. 步骤结束时, 一次 all-reduce 将本地计数求和为全局直方图, 每个 rank 再从相同的汇总计数中恢复分位数. 每个专家的直方图都对每个 token 计数一次, 因此目标秩恰好为 $q=mk/n$ ([§2.3.3](#_2-3-3-quantile-balancing)), 此时在整个步骤上计算. 我们选择累积计数第一个达到 $\lceil q\rceil$ 的 bin, 并在其中进行线性插值. 如果选择 bin $\ell_j$, 其此前的累积计数为 $c_j$, 内部计数为 $h_j$, 则 $b_j=b_{\min}-1+(\ell_j+\operatorname{clip}((q-c_j)/h_j,0,1))w$. 最后, 与式 14 相同, 对所得偏置执行均值中心化.

**性质.** 三项性质使该估计器适合大规模使用. 第一, 它很准确: bin 边界处的累积计数是精确的, 因而真实分位数及其估计值位于同一个 bin 中, 误差由 bin 宽度 $w$ 限制. 当 $B=1000$ 时, 误差至多为几个 $10^{-3}$, 我们没有观察到可测量的残余负载不均衡. 第二, 它的成本很低: 每层每个步骤只需对 $nB$ 个整数执行一次 all-reduce, 与 $m$ 无关. 在我们的配置中, 其成本不足自然替代方案的 1%, 后者需要在每个 micro-batch 中通过 process group 交换原始 margin. 第三, 它估计的是正确的量: 由于计数具有可加性, 无论 token 如何在 rank 或累积步骤间划分, 全局直方图都完全不变. 所得估计值是汇总后全局 batch 的分位数, 而非通常与之不同的各 rank 分位数平均值. 进一步地, 在不同步骤间维护估计分位数的指数移动平均, 可以减少 batch 间的采样噪声, 并进一步改善负载均衡.

## E MoonEP 一般上界证明

设 $m_r(P)$ 表示方案 $P$ 下放置在 rank $r$ 上的冗余专家数量. 对于路由器输出 $I$, 规划目标是最小化任意 rank 上冗余专家数量的最大值, 即 $M(I)=\min_P\max_r\{m_r(P)\}$. 我们证明 $M(I)\le E/R$ 始终成立 (定理 1), 且该界基本为紧界: 存在使 $M=\lceil E(R-1)/R^2\rceil\approx E/R$ 的路由器输出 (定理 2).

#### 定理 1 的证明 (一般上界)

目标是证明对任意路由器输出 $I$, $M(I)\le E/R$ 均成立. 关键引理是, 存在一个方案 $P^*$, 使每个 EP rank 都恰好接收相同数量的 token, 即 $S\times K$, 且每个 rank 的远程 token 只来自另一个 EP rank. 初始时, 每个 rank 只持有本地 token, 各 rank 被分为负载不足和负载过多两类. 我们反复从两类中各选择一个 rank, 将 token 从负载过多的 rank 迁移到负载不足的 rank, 使后者恰好填充到 $S\times K$. 每次填充都会使一个负载不足的 rank 达到均衡, 此后不再改变, 因此该过程最多经过 $R-1$ 次填充便会终止. 每个 rank 最多填充一次, 故其远程 token 来自单一 rank. 如果 rank $r$ 的远程 token 来自 rank $s$, 则它们至多属于 rank $s$ 上的 $E/R$ 个本地专家, 所以 $m_r(P^*)\le E/R$, 因而

$$
M(I)=\min_P\max_r\{m_r(P)\}
\le\max_r\{m_r(P^*)\}\le\frac ER.
\tag{28}
$$

#### 定理 2 的证明 (上界的紧性)

按如下方式构造路由器输出 $I^*$: EP rank 0 上的专家不接收任何 token, 其他 $R-1$ 个 rank 上的所有专家均匀分担全部 token. 此时, 所有 $S\times K\times R$ 个 token 被均匀分配给 $E(R-1)/R$ 个专家, 因而每个专家接收 $SKR^2/[E(R-1)]$ 个 token. 在任意方案 $P$ 下, rank 0 都必须接收 $S\times K$ 个 token, 且全部为远程 token. 这些 token 涉及至少 $E(R-1)/R^2$ 个不同专家. 向上取整后, rank 0 至少需要 $\lceil E(R-1)/R^2\rceil$ 个冗余专家, 因此 $M(I^*)\ge\lceil E(R-1)/R^2\rceil$. 反过来, 定理 1 中的填充流程若优先按专家迁移, 可以使每个 rank 都不超过该值, 因而等式成立. 当 $R$ 较大时, $E(R-1)/R^2\approx E/R$, 所以定理 1 的上界基本为紧界.

## F 聊天模板

<span id="figure-16"></span>

![Kimi K3 聊天模板结构](./kimi-k3/figure-16.png)

**图 16.** Kimi K3 聊天模板结构: 上下文布局, 助理消息通道和带索引的并行工具调用.

Kimi K3 聊天模板围绕三个目标重新设计. 第一是可扩展性: 新能力应通过向后兼容的消息格式引入, 而非修改模板, 从而使单一模板可以服务于整个模型世代. 第二是较低的对齐税: 该格式应只需极少量监督数据即可学会, 从而支持轻度微调后的预训练模型直接进入强化学习的流水线. 第三是便于解码: 该结构应支持简单编码器, 流式 parser 和语法约束执行器. 为此, 模板采用 XTML (eXtensible Token Markup Language), 这是一种类 XML 标记语言, 其中尖括号语法由三个保留的特殊 token 取代:

#### 消息与分区

上下文的顶层单位是消息, 消息按来源分为两类 ([图 16a](#figure-16)). 输入消息序列化请求的 `messages` 字段, 涵盖常见的 `system`, `user`, `assistant` 和 `tool` 角色. 选项消息将请求选项转换为模型在上下文中读取的指令, 其位置反映作用域. 全局选项, 即工具声明 (`type="tool-declare"`) 和推理强度设置, 位于所有输入消息之前: 它们控制整个会话且很少变化, 修改后无论如何都会使 KV cache 失效. 一次性选项 (`tool_choice`, `response_format`) 追加在输入消息之后, 因而逐请求变化不会破坏历史 KV cache. 第三类输入选项消息与输入消息交错排列, 用于在会话中途补充或改写全局选项. 该机制支持动态加载工具: 对话期间检索或加载的工具通过额外的 `tool-declare` 消息声明, 此后模型的可用工具集即可扩展, 无需重建先前上下文.

#### 通道

助理消息正文按通道组织, 这一概念受到 OpenAI Harmony 响应格式 [Ope25a] 启发: `think` 携带推理轨迹, `response` 携带用户可见答案, `tools` 携带工具调用 ([图 16b](#figure-16)). 两种生成模式完全通过生成前缀选择, 思考模式使用 `[open]think[sep]`, 指令模式使用 `[open]response[sep]`, 无需使用不同模板. Kimi K3 只支持保留思考: 在思考模式中, `think` 通道始终保留在历史记录内, 即使内容为空也是如此, 使模型在不同轮次中观察到一致的消息结构; 在指令模式中, 历史消息只包含 `response` 和 `tools` 通道.

**工具调用.** 在 `tools` 通道中, 每次调用都包含 `tool` 和 `index` 属性. `index` 为消息内的并行调用编号, 每条工具结果消息都会重复相同的 `tool/index` 组合, 并遵循调用顺序, 使结果与调用之间能够明确关联. 参数具有类型: 字符串参数以原始文本出现, 其他 JSON 类型的值则采用紧凑序列化. 因此, 代码等自由形式文本是一等公民, 而非经过转义的 JSON 字符串. 纯 JSON fallback 块用于参数无法分解为带类型参数块的输入; 它只会出现在输入 token 中, 绝不会出现在模型输出中, 训练时还会屏蔽其损失.

#### 推理强度与选项

推理强度以 `thinking-effort` 类型的全局选项消息暴露, 插入到工具声明之后, 输入消息之前. 该消息不会修改生成前缀或暴露 token 预算, 而是用自然语言说明请求的级别, 并作为生成约束指令. schema 预留 `low`, `medium`, `high` 和 `max` 四个级别, Kimi K3 支持其中一部分. 这种表示将强度接口与模板语法解耦, 并直接对应[第 4.1.1 节](#_4-1-1-监督微调)和[第 4.1.2 节](#_4-1-2-强化学习)所述的强度条件训练. 更一般地说, 所有选项消息都采用同一实现方式: `tool_choice`, `response_format` 和 `thinking-effort` 都会转换为放入上下文的简短自然语言指令, 而非专用特殊语法. 由于预训练模型已经能够很好地遵循此类指令, 新选项只需少量甚至无需额外训练即可引入, 这直接体现了前述低对齐税设计原则.
