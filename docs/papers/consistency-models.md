---
title: Consistency Models
createTime: 2026/09/14 11:58:45
permalink: /papers/consistency-models/
pageClass: paper-reading
---

> [Yang Song](https://yang-song.net/), [Prafulla Dhariwal](https://dblp.org/pid/190/7235), [Mark Chen](https://dblp.org/pid/40/1660-3), [Ilya Sutskever](https://dblp.org/pid/60/5276). 2023 年 3 月 2 日首次提交至 arXiv, 当前版本为 v2. 发表于 [第 40 届国际机器学习大会论文集](https://proceedings.mlr.press/v202/song23a.html), PMLR 202:32211-32252, 2023. [Consistency Models](https://arxiv.org/abs/2303.01469). <a href="/paper/consistency-models.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2303.01469). [TeX 源文件](https://export.arxiv.org/e-print/2303.01469v2). 准确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

扩散模型显著推动了图像, 音频和视频生成的发展, 但它依赖迭代采样过程, 因而生成速度较慢. 为克服这一局限, 我们提出*一致性模型*, 这是一类通过将噪声直接映射为数据来生成高质量样本的新模型. 它在设计上支持快速的单步生成, 同时仍允许用多步采样在计算量与样本质量之间进行权衡. 它还支持图像修复, 着色和超分辨率等零样本数据编辑, 无需针对这些任务进行显式训练. 一致性模型既可以通过蒸馏预训练扩散模型来训练, 也可以完全作为独立的生成模型来训练. 大量实验表明, 在单步和少步采样中, 它优于现有的扩散模型蒸馏技术; 对于单步生成, 在 CIFAR-10 上取得了 3.55 的最新最佳 FID, 在 ImageNet $64\times 64$ 上取得了 6.20 的最新最佳 FID. 独立训练时, 一致性模型构成了一类新的生成模型, 在 CIFAR-10, ImageNet $64\times 64$ 和 LSUN $256\times 256$ 等标准基准上, 能够超越现有的单步, 非对抗式生成模型.

<span id="section-1"></span>

## 1 引言

<span id="figure-01"></span>

![图 1. 给定一条将数据平滑转换为噪声的概率流 (PF) ODE, 我们学习把 ODE 轨迹上的任意点 (例如 ${\mathbf{x}}_{t}$, ${\mathbf{x}}_{t^{\prime}}$ 和 ${\mathbf{x}}_{T}$) 映射到其起点 (例如 ${\mathbf{x}}_{0}$), 用于生成建模. 这类映射的模型称为一致性模型, 因为训练要求它对同一条轨迹上的各点给出一致的输出.](./consistency-models/figure-01.png)

**图 1.** 给定一条将数据平滑转换为噪声的概率流 (PF) ODE, 我们学习把 ODE 轨迹上的任意点 (例如 ${\mathbf{x}}_{t}$, ${\mathbf{x}}_{t^{\prime}}$ 和 ${\mathbf{x}}_{T}$) 映射到其起点 (例如 ${\mathbf{x}}_{0}$), 用于生成建模. 这类映射的模型称为一致性模型, 因为训练要求它对同一条轨迹上的各点给出一致的输出.

扩散模型 [Soh15, Son19a, Son20, Nic21, Son21] 也称为基于分数的生成模型, 已在图像生成 [Dha21, Nic22, Ram22, Sah22a, Rom22], 音频合成 [Kon21, Che21n, Pop21] 和视频生成 [Ho22, Ho22c] 等多个领域取得空前成功. 扩散模型的一个关键特征是迭代采样过程, 它从随机初始向量中逐步去除噪声. 这一迭代过程能够灵活权衡计算量与样本质量, 因为增加计算量并执行更多迭代通常会得到质量更高的样本. 它也是扩散模型多种零样本数据编辑能力的核心, 使模型能够解决从图像修复, 着色, 笔画引导的图像编辑, 到计算机断层扫描和磁共振成像等困难的逆问题 [Son19a, Son21, Son22, Son23a, Kaw21, Kaw22, Chu23, Men22]. 然而, 与 GAN [Goo14a], VAE [Kin14, Rez14] 或归一化流 [Din15, Din17, Kin18] 等单步生成模型相比, 扩散模型的迭代生成过程通常需要多出 10-2000 倍的样本生成计算量 [Son20, Nic21, Son21, Zha22i, Lu22c], 导致推理缓慢, 实时应用受限.

我们的目标是构建支持高效单步生成的生成模型, 同时不牺牲迭代采样的重要优势, 例如在需要时用计算量换取样本质量, 以及执行零样本数据编辑任务. 如[图 1](#figure-01) 所示, 我们以连续时间扩散模型 [Son21] 中的概率流 (PF) 常微分方程 (ODE) 为基础; 其轨迹将数据分布平滑地转换为易于处理的噪声分布. 我们提出学习一个模型, 将任意时间步上的任意点映射到轨迹起点. 该模型有一个显著性质, 即自一致性: *同一条轨迹上的点映射到同一个初始点*. 因此, 我们把这类模型称为**一致性模型**. 一致性模型只需一次网络求值, 就能把随机噪声向量 (ODE 轨迹的终点, 例如[图 1](#figure-01) 中的 ${\mathbf{x}}_{T}$) 转换为数据样本 (ODE 轨迹的起点, 例如[图 1](#figure-01) 中的 ${\mathbf{x}}_{0}$). 更重要的是, 在多个时间步串联一致性模型的输出, 可以用更多计算量改善样本质量并执行零样本数据编辑, 与扩散模型的迭代采样类似.

为了训练一致性模型, 我们给出两种以强制满足自一致性性质为基础的方法. 第一种方法使用数值 ODE 求解器和预训练扩散模型, 在 PF ODE 轨迹上生成相邻点对. 通过最小化模型在这些点对上的输出差异, 可以有效地把扩散模型蒸馏为一致性模型, 从而只用一次网络求值生成高质量样本. 相比之下, 第二种方法完全不需要预训练扩散模型, 可以独立训练一致性模型. 这种做法使一致性模型成为一个独立的生成模型族. 重要的是, 两种方法都不需要对抗训练, 对架构的限制也很少, 因而可以使用灵活的神经网络来参数化一致性模型.

我们在 CIFAR-10 [Kri09], ImageNet $64\times 64$ [Den09a] 和 LSUN $256\times 256$ [Yu15a] 等多个图像数据集上验证了一致性模型的有效性. 实验表明, 作为一种蒸馏方法, 一致性模型在多个数据集的少步生成中优于渐进式蒸馏 [Sal22] 等现有扩散蒸馏方法: 在 CIFAR-10 上, 一致性模型用一步和两步生成分别取得了 3.55 和 2.93 的最新最佳 FID; 在 ImageNet $64\times 64$ 上, 用一次和两次网络求值分别取得了创纪录的 6.20 和 4.70 FID. 作为独立生成模型训练时, 即使无法访问预训练扩散模型, 一致性模型也能达到或超过渐进式蒸馏的单步样本质量. 它还能够在多个数据集上超过许多 GAN 以及现有的非对抗式单步生成模型. 此外, 我们表明一致性模型可用于多种零样本数据编辑任务, 包括图像去噪, 插值, 修复, 着色, 超分辨率和笔画引导的图像编辑 (SDEdit, [Men22]).

<span id="section-2"></span>

## 2 扩散模型

一致性模型深受连续时间扩散模型理论 [Son21, Kar22] 的启发. 扩散模型先用高斯扰动逐步把数据扰动成噪声, 再通过连续的去噪步骤从噪声生成样本. 令 $p_{\text{data}}({\mathbf{x}})$ 表示数据分布. 扩散模型首先用随机微分方程 (SDE) [Son21] 对 $p_{\text{data}}({\mathbf{x}})$ 进行扩散

<span id="equation-01"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\bm{\mu}({\mathbf{x}}_{t},t)\mathop{}\!\mathrm{d}t+\sigma(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_{t},
$$

其中 $t\in[0,T]$, $T>0$ 是固定常数, $\bm{\mu}(\cdot,\cdot)$ 和 $\sigma(\cdot)$ 分别是漂移系数与扩散系数, $\{{\mathbf{w}}_{t}\}_{t\in[0,T]}$ 表示标准布朗运动. 我们将 ${\mathbf{x}}_{t}$ 的分布记为 $p_{t}({\mathbf{x}})$, 因而 $p_{0}({\mathbf{x}})\equiv p_{\text{data}}({\mathbf{x}})$. 这条 SDE 的一个重要性质是存在一条常微分方程 (ODE), [Son21] 将其称为*概率流 (PF) ODE*; 其解轨迹在 $t$ 时刻采样所得的分布为 $p_{t}({\mathbf{x}})$:

<span id="equation-02"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\left[\bm{\mu}({\mathbf{x}}_{t},t)-\frac{1}{2}\sigma(t)^{2}\nabla\log p_{t}({\mathbf{x}}_{t})\right]\mathop{}\!\mathrm{d}t.
$$

这里, $\nabla\log p_{t}({\mathbf{x}})$ 是 $p_{t}({\mathbf{x}})$ 的*分数函数*; 因此, 扩散模型也称为*基于分数的生成模型* [Son19a, Son20, Son21].

通常会把[公式 1](#equation-01) 中的 SDE 设计为使 $p_{T}({\mathbf{x}})$ 接近易于处理的高斯分布 $\pi({\mathbf{x}})$. 下文采用 [Kar22] 的设定, 即 $\bm{\mu}({\mathbf{x}},t)=\bm{0}$ 且 $\sigma(t)=\sqrt{2t}$. 在这种情况下, $p_{t}({\mathbf{x}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$, 其中 $\otimes$ 表示卷积运算, 且 $\pi({\mathbf{x}})=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$. 采样时, 我们先通过*分数匹配* [Hyv05, Vin11, Son19b, Son19a, Nic21] 训练*分数模型* ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\approx\nabla\log p_{t}({\mathbf{x}})$, 再将其代入[公式 2](#equation-02), 得到 PF ODE 的经验估计, 形式如下

<span id="equation-03"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t).
$$

我们把[公式 3](#equation-03) 称为*经验 PF ODE*. 随后, 从 $\hat{{\mathbf{x}}}_{T}\sim\pi=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ 采样来初始化经验 PF ODE, 再使用任意数值 ODE 求解器沿时间反向求解, 例如 Euler [Son21a, Son21] 和 Heun 求解器 [Kar22], 从而得到解轨迹 $\{\hat{{\mathbf{x}}}_{t}\}_{t\in[0,T]}$. 由此得到的 $\hat{{\mathbf{x}}}_{0}$ 可视为数据分布 $p_{\text{data}}({\mathbf{x}})$ 的近似样本. 为避免数值不稳定, 通常在 $t=\epsilon$ 时停止求解器, 其中 $\epsilon$ 是固定的小正数, 并接受 $\hat{{\mathbf{x}}}_{\epsilon}$ 作为近似样本. 按照 [Kar22], 我们把图像像素值缩放到 $[-1,1]$, 并设置 $T=80,\epsilon=0.002$.

扩散模型的瓶颈在于采样速度缓慢. 显然, 使用 ODE 求解器采样需要反复计算分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$, 计算代价很高. 现有的快速采样方法包括更快的数值 ODE 求解器 [Son21a, Zha22i, Lu22c, Doc22a] 和蒸馏技术 [Luh21, Sal22, Men22a, Zhe22g]. 然而, ODE 求解器仍然需要 10 次以上的求值才能生成有竞争力的样本. [Luh21] 和 [Zhe22g] 等大多数蒸馏方法, 都需要在蒸馏前从扩散模型收集大型样本数据集, 这个过程本身就有很高的计算成本. 据我们所知, 唯一没有这一缺点的蒸馏方法是渐进式蒸馏 (PD, [Sal22]); 我们在实验中对它与一致性模型进行了全面比较.

<span id="section-3"></span>

## 3 一致性模型

<span id="figure-02"></span>

![图 2. 一致性模型经过训练, 将 PF ODE 任意轨迹上的点映射到该轨迹的起点.](./consistency-models/figure-02.png)

**图 2.** 一致性模型经过训练, 将 PF ODE 任意轨迹上的点映射到该轨迹的起点.

我们提出一致性模型, 这是一类在设计核心上支持单步生成的新模型, 同时仍允许迭代生成, 以便权衡样本质量与计算量并执行零样本数据编辑. 一致性模型既可以采用蒸馏模式训练, 也可以采用独立模式训练. 在前一种情况下, 一致性模型把预训练扩散模型的知识蒸馏到单步采样器中, 在允许零样本图像编辑应用的同时, 显著提高其他蒸馏方法的样本质量. 在后一种情况下, 一致性模型独立训练, 不依赖预训练扩散模型. 这使它成为一类独立的新生成模型.

下面介绍一致性模型的定义, 参数化和采样, 并简要讨论它在零样本数据编辑中的应用.

**定义** 给定[公式 2](#equation-02) 中 PF ODE 的一条解轨迹 $\{{\mathbf{x}}_{t}\}_{t\in[\epsilon,T]}$, 我们将*一致性函数*定义为 ${\bm{f}}:({\mathbf{x}}_{t},t)\mapsto{\mathbf{x}}_{\epsilon}$. 一致性函数具有*自一致性*: 对属于同一条 PF ODE 轨迹的任意 $({\mathbf{x}}_{t},t)$ 对, 其输出都保持一致, 即对所有 $t,t^{\prime}\in[\epsilon,T]$, 都有 ${\bm{f}}({\mathbf{x}}_{t},t)={\bm{f}}({\mathbf{x}}_{t^{\prime}},t^{\prime})$. 如[图 2](#figure-02) 所示, *一致性模型*记为 ${\bm{f}}_{\bm{\theta}}$, 其目标是通过学习强制满足自一致性性质, 从数据估计一致性函数 ${\bm{f}}$ (详见[第 4 节](#section-4) 和[第 5 节](#section-5)). 请注意, 神经 ODE [Che18g] 语境中的神经流 [Bil21] 也使用了类似定义. 但与神经流相比, 我们不要求一致性模型可逆.

**参数化** 对任意一致性函数 ${\bm{f}}(\cdot,\cdot)$, 都有 ${\bm{f}}({\mathbf{x}}_{\epsilon},\epsilon)={\mathbf{x}}_{\epsilon}$, 即 ${\bm{f}}(\cdot,\epsilon)$ 是恒等函数. 我们将这一约束称为*边界条件*. 所有一致性模型都必须满足该边界条件, 因为它对成功训练一致性模型至关重要. 这个边界条件也是一致性模型在架构上最严格的约束. 对于基于深度神经网络的一致性模型, 我们讨论两种*几乎无需额外代价*即可实现该边界条件的方法. 假设有一个形式自由的深度神经网络 $F_{\bm{\theta}}({\mathbf{x}},t)$, 其输出维度与 ${\mathbf{x}}$ 相同. 第一种方法是直接将一致性模型参数化为

<span id="equation-04"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=\begin{cases}{\mathbf{x}}&\quad t=\epsilon\\
F_{\bm{\theta}}({\mathbf{x}},t)&\quad t\in(\epsilon,T]\end{cases}.
$$

第二种方法使用跳跃连接来参数化一致性模型, 即

<span id="equation-05"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t),
$$

其中 $c_{\text{skip}}(t)$ 和 $c_{\text{out}}(t)$ 是满足 $c_{\text{skip}}(\epsilon)=1$ 和 $c_{\text{out}}(\epsilon)=0$ 的可微函数. 这样, 如果 $F_{\bm{\theta}}({\mathbf{x}},t),c_{\text{skip}}(t),c_{\text{out}}(t)$ 均可微, 则一致性模型在 $t=\epsilon$ 处可微; 这一点对训练连续时间一致性模型至关重要 (见[第 9.1 节](#section-9-1) 和[第 9.2 节](#section-9-2)). [公式 5](#equation-05) 中的参数化与许多成功的扩散模型 [Kar22, Bal22] 高度相似, 因而更容易借用强大的扩散模型架构来构建一致性模型. 因此, 我们在所有实验中都采用第二种参数化.

**采样** 对于训练良好的一致性模型 ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, 可以先从初始分布 $\hat{{\mathbf{x}}}_{T}\sim\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ 采样, 再计算 $\hat{{\mathbf{x}}}_{\epsilon}={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$ 来生成样本. 这只涉及一次一致性模型的前向传播, 因而*能够单步生成样本*. 还可以交替执行去噪和噪声注入, 多次计算一致性模型来提高样本质量. [算法 1](#algorithm-01) 总结了这种*多步*采样过程, 它可以灵活权衡计算量与样本质量. 该过程在零样本数据编辑中也有重要用途. 实践中, 我们通过贪心算法寻找[算法 1](#algorithm-01) 中的时间点 $\{\tau_{1},\tau_{2},\cdots,\tau_{N-1}\}$, 使用三分搜索逐个确定时间点, 以优化[算法 1](#algorithm-01) 所得样本的 FID. 这假设给定先前的时间点后, FID 是下一个时间点的单峰函数. 实验中我们发现这一假设成立, 并把探索更好的策略留作未来工作.

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**算法 1: 多步一致性采样.**

- **输入:** 一致性模型 ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, 时间点序列 $\tau_{1}>\tau_{2}>\cdots>\tau_{N-1}$, 初始噪声 $\hat{{\mathbf{x}}}_{T}$.
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$.
- **对于** $n=1$ **到** $N-1$:
  - 采样 ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$.
  - $\hat{{\mathbf{x}}}_{\tau_{n}}\gets{\mathbf{x}}+\sqrt{\tau_{n}^{2}-\epsilon^{2}}{\mathbf{z}}$.
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{\tau_{n}},\tau_{n})$.
- **输出:** ${\mathbf{x}}$.

</div>

**零样本数据编辑** 与扩散模型类似, 一致性模型可以零样本地完成多种数据编辑和操作; 执行这些任务不需要显式训练. 例如, 一致性模型定义了从高斯噪声向量到数据样本的一一映射. 与 GAN, VAE 和归一化流等潜变量模型类似, 一致性模型可以沿潜在空间移动, 轻松地在样本之间进行插值 (见[图 11](#figure-11)). 由于一致性模型经过训练, 能够从 $t\in[\epsilon,T]$ 时的任意含噪输入 ${\mathbf{x}}_{t}$ 恢复 ${\mathbf{x}}_{\epsilon}$, 因此可以对多种噪声水平执行去噪 (见[图 12](#figure-12)). 此外, [算法 1](#algorithm-01) 的多步生成过程可以采用类似扩散模型的迭代替换过程 [Son19a, Son21, Ho22], 零样本地求解某些逆问题. 这使其能够用于多种图像编辑任务, 包括图像修复 (见[图 10](#figure-10)), 着色 (见[图 8](#figure-08)), 超分辨率 (见[图 6(b)](#figure-06)), 以及 SDEdit [Men22] 中的笔画引导图像编辑 (见[图 13](#figure-13)). 我们在[第 6.3 节](#section-6-3) 通过实验展示一致性模型在多种零样本图像编辑任务中的能力.

<span id="section-4"></span>

## 4 通过蒸馏训练一致性模型

我们给出的第一种一致性模型训练方法, 以蒸馏预训练分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 为基础. 讨论围绕[公式 3](#equation-03) 中的经验 PF ODE 展开, 该方程通过把分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 代入 PF ODE 得到. 考虑将时间范围 $[\epsilon,T]$ 离散为 $N-1$ 个子区间, 边界为 $t_{1}=\epsilon<t_{2}<\cdots<t_{N}=T$. 实践中, 我们遵循 [Kar22], 使用公式 $t_{i}=(\epsilon^{1/\rho}+\frac{i-1}{N-1}(T^{1/\rho}-\epsilon^{1/\rho}))^{\rho}$ 确定边界, 其中 $\rho=7$. 当 $N$ 足够大时, 对数值 ODE 求解器执行一次离散化步骤, 即可根据 ${\mathbf{x}}_{t_{n+1}}$ 准确估计 ${\mathbf{x}}_{t_{n}}$. 我们把这个估计记为 $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$, 定义如下

<span id="equation-06"></span>

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

其中 $\Phi(\cdots;{\bm{\phi}})$ 表示对经验 PF ODE 应用单步 ODE 求解器时的更新函数. 例如, 使用 Euler 求解器时, 有 $\Phi({\mathbf{x}},t;{\bm{\phi}})=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$, 对应以下更新规则

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}={\mathbf{x}}_{t_{n+1}}-(t_{n}-t_{n+1})t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1}).
$$

为简单起见, 本文只考虑单步 ODE 求解器. 将我们的框架推广到多步 ODE 求解器并不困难, 我们把它留作未来工作.

由于[公式 2](#equation-02) 中的 PF ODE 与[公式 1](#equation-01) 中的 SDE 存在联系 (见[第 2 节](#section-2)), 可以先采样 ${\mathbf{x}}\sim p_{\text{data}}$, 再向 ${\mathbf{x}}$ 添加高斯噪声, 由此沿 ODE 轨迹的分布采样. 具体而言, 给定数据点 ${\mathbf{x}}$, 我们先从数据集采样 ${\mathbf{x}}$, 再从 SDE 的转移密度 $\mathcal{N}({\mathbf{x}},t_{n+1}^{2}{\bm{I}})$ 采样 ${\mathbf{x}}_{t_{n+1}}$, 随后依据[公式 6](#equation-06) 执行数值 ODE 求解器的一次离散化步骤, 计算 $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$. 这样便能高效生成 PF ODE 轨迹上的相邻数据点对 $(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},{\mathbf{x}}_{t_{n+1}})$. 然后, 我们最小化模型在点对 $(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},{\mathbf{x}}_{t_{n+1}})$ 上的输出差异, 以训练一致性模型. 这引出了下面用于训练一致性模型的*一致性蒸馏*损失.

<span id="definition-01"></span>

**定义 1.** 一致性蒸馏损失定义为

<span id="equation-07"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\\
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))],
$$

其中, 期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$ 和 ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ 计算. 这里, $\mathcal{U}\llbracket 1,N-1\rrbracket$ 表示 $\{1,2,\cdots,N-1\}$ 上的均匀分布, $\lambda(\cdot)\in\mathbb{R}^{+}$ 是正权重函数, $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ 由[公式 6](#equation-06) 给出, ${\bm{\theta}}^{-}$ 表示优化过程中 ${\bm{\theta}}$ 历史值的滑动平均, $d(\cdot,\cdot)$ 是满足 $\forall{\mathbf{x}},{\mathbf{y}}:d({\mathbf{x}},{\mathbf{y}})\geq 0$ 且仅在 ${\mathbf{x}}={\mathbf{y}}$ 时 $d({\mathbf{x}},{\mathbf{y}})=0$ 的度量函数.

除非另有说明, 全文采用[定义 1](#definition-01) 中的记号, 并用 $\mathbb{E}[\cdot]$ 表示对所有随机变量的期望. 实验中, 我们考虑平方 $\ell_{2}$ 距离 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$, $\ell_{1}$ 距离 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$, 以及学习感知图像块相似度 (LPIPS, [Zha18d]). 我们发现, $\lambda(t_{n})\equiv 1$ 在所有任务和数据集上都表现良好. 实践中, 我们对模型参数 ${\bm{\theta}}$ 使用随机梯度下降来最小化目标, 同时用指数移动平均 (EMA) 更新 ${\bm{\theta}}^{-}$. 也就是说, 给定衰减率 $0\leq\mu<1$, 每个优化步骤之后执行如下更新:

<span id="equation-08"></span>

$$
{\bm{\theta}}^{-}\leftarrow\operatorname{stopgrad}(\mu{\bm{\theta}}^{-}+(1-\mu){\bm{\theta}}).
$$

完整训练过程总结于[算法 2](#algorithm-02). 按照深度强化学习 [Mni13, Mni15, Lil15] 和基于动量的对比学习 [Gri20, He20a] 中的惯例, 我们把 ${\bm{f}}_{{\bm{\theta}}^{-}}$ 称为“目标网络”, 把 ${\bm{f}}_{\bm{\theta}}$ 称为“在线网络”. 与直接设置 ${\bm{\theta}}^{-}={\bm{\theta}}$ 相比, 我们发现[公式 8](#equation-08) 中的 EMA 更新和“stopgrad”算子可以显著稳定训练过程, 并改善一致性模型的最终性能.

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**算法 2: 一致性蒸馏 (CD).**

- **输入:** 数据集 $\mathcal{D}$, 初始模型参数 $\bm{\theta}$, 学习率 $\eta$, ODE 求解器 $\Phi(\cdot,\cdot;\bm{\phi})$, $d(\cdot,\cdot)$, $\lambda(\cdot)$ 和 $\mu$.
- $\bm{\theta}^{-}\gets\bm{\theta}$.
- **重复直至**收敛:
  - 采样 ${\mathbf{x}}\sim\mathcal{D}$ 和 $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$.
  - 采样 ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$.
  - $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\gets{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};\bm{\phi})$.
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))$.
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})$.
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu\bm{\theta}^{-}+(1-\mu)\bm{\theta})$.

</div>

下面基于渐近分析, 从理论上说明一致性蒸馏的合理性.

<span id="theorem-01"></span>

**定理 1.** 令 $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$, 并令 ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ 为[公式 3](#equation-03) 中经验 PF ODE 的一致性函数. 假设 ${\bm{f}}_{\bm{\theta}}$ 满足 Lipschitz 条件: 存在 $L>0$, 使得对所有 $t\in[\epsilon,T]$, ${\mathbf{x}}$ 和 ${\mathbf{y}}$, 都有 $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$. 进一步假设, 对所有 $n\in\llbracket 1,N-1\rrbracket$, 在 $t_{n+1}$ 调用的 ODE 求解器局部误差由 $O((t_{n+1}-t_{n})^{p+1})$ 一致界定, 且 $p\geq 1$. 那么, 如果 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$, 则有

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details 证明
该证明基于归纳法, 与数值 ODE 求解器全局误差界的经典证明 [Sul03] 类似. 完整证明见[第 8.2 节](#section-8-2).
:::

由于 ${\bm{\theta}}^{-}$ 是 ${\bm{\theta}}$ 历史值的滑动平均, 当[算法 2](#algorithm-02) 的优化收敛时, 有 ${\bm{\theta}}^{-}={\bm{\theta}}$. 也就是说, 目标一致性模型最终会与在线一致性模型相同. 如果一致性模型还取得了零一致性蒸馏损失, 那么[定理 1](#theorem-01) 表明, 在某些正则性条件下, 只要 ODE 求解器的步长足够小, 估计得到的一致性模型就能达到任意精度. 重要的是, 边界条件 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},\epsilon)\equiv{\mathbf{x}}$ 排除了一致性模型训练中出现平凡解 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\equiv\bm{0}$ 的可能.

当 ${\bm{\theta}}^{-}={\bm{\theta}}$ 或 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ 时, 一致性蒸馏损失 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 可以推广到无限多个时间步 ($N\to\infty$). 所得连续时间损失函数不需要指定 $N$ 或时间步 $\{t_{1},t_{2},\cdots,t_{N}\}$. 不过, 它们涉及 Jacobian-向量积, 需要前向模式自动微分才能高效实现, 而某些深度学习框架可能没有完善支持. 我们在定理 3, 4 和 5 中给出这些连续时间蒸馏损失函数, 细节见[第 9.1 节](#section-9-1).

<span id="section-5"></span>

## 5 独立训练一致性模型

一致性模型可以不依赖任何预训练扩散模型进行训练. 这一点不同于现有的扩散蒸馏技术, 使一致性模型成为一个新的独立生成模型族.

<span id="algorithm-03"></span>

<div class="paper-algorithm">

**算法 3: 一致性训练 (CT).**

- **输入:** 数据集 $\mathcal{D}$, 初始模型参数 $\bm{\theta}$, 学习率 $\eta$, 步数调度 $N(\cdot)$, EMA 衰减率调度 $\mu(\cdot)$, $d(\cdot,\cdot)$ 和 $\lambda(\cdot)$.
- $\bm{\theta}^{-}\gets\bm{\theta}$ 且 $k\gets 0$.
- **重复直至**收敛:
  - 采样 ${\mathbf{x}}\sim\mathcal{D}$ 和 $n\sim\mathcal{U}\llbracket 1,N(k)-1\rrbracket$.
  - 采样 ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$.
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))$.
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-})$.
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu(k)\bm{\theta}^{-}+(1-\mu(k))\bm{\theta})$.
  - $k\gets k+1$.

</div>

回顾一致性蒸馏, 我们依靠预训练分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 逼近真实分数函数 $\nabla\log p_{t}({\mathbf{x}})$. 事实表明, 利用下面的无偏估计量 (见[第 8 节](#section-8) 中的[引理 1](#lemma-01)), 可以完全不使用这个预训练分数模型:

$$
\nabla\log p_{t}({\mathbf{x}}_{t})=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mathrel{\bigg|}{\mathbf{x}}_{t}\right],
$$

其中 ${\mathbf{x}}\sim p_{\text{data}}$, ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$. 也就是说, 给定 ${\mathbf{x}}$ 和 ${\mathbf{x}}_{t}$ 后, 可以用 $-({\mathbf{x}}_{t}-{\mathbf{x}})/t^{2}$ 估计 $\nabla\log p_{t}({\mathbf{x}}_{t})$.

如下述结果所示, 当 $N\to\infty$ 且使用 Euler 方法作为 ODE 求解器时, 这个无偏估计足以在一致性蒸馏中替代预训练扩散模型.

<span id="theorem-02"></span>

**定理 2.** 令 $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$. 假设 $d$ 和 ${\bm{f}}_{{\bm{\theta}}^{-}}$ 均二阶连续可微且二阶导数有界, 权重函数 $\lambda(\cdot)$ 有界, 并且 $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$. 进一步假设使用 Euler ODE 求解器, 且预训练分数模型与真实值一致, 即 $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$. 那么,

<span id="equation-09"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

其中, 期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$ 和 ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ 计算. 一致性训练目标记为 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, 定义如下

<span id="equation-10"></span>

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

其中 ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. 此外, 如果 $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$, 则 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$.

::: details 证明
该证明基于 Taylor 级数展开和分数函数的性质 (见[引理 1](#lemma-01)). 完整证明见[第 8.3 节](#section-8-3).
:::

我们把[公式 10](#equation-10) 称为*一致性训练* (CT) 损失. 关键在于, $\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})$ 只依赖在线网络 ${\bm{f}}_{\bm{\theta}}$ 和目标网络 ${\bm{f}}_{{\bm{\theta}}^{-}}$, 与扩散模型参数 ${\bm{\phi}}$ 完全无关. 损失函数 $\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ 的下降速度慢于余项 $o(\Delta t)$, 因而在 $N\to\infty$ 且 $\Delta t\to 0$ 时, 它会在[公式 9](#equation-09) 的损失中占主导地位.

为了改善实际性能, 我们提出在训练过程中按照调度函数 $N(\cdot)$ 逐渐增大 $N$. 直观上 (参见[图 3(d)](#figure-03)), $N$ 较小时 (即 $\Delta t$ 较大), 相对于底层的一致性蒸馏损失 (即[公式 9](#equation-09) 左侧), 一致性训练损失的“方差”较小而“偏差”较大, 有助于训练初期更快收敛. 相反, $N$ 较大时 (即 $\Delta t$ 较小), 其“方差”较大而“偏差”较小, 这更适合训练接近结束时的情况. 为获得最佳性能, 我们还发现 $\mu$ 应随 $N$ 一同变化, 变化方式由调度函数 $\mu(\cdot)$ 给出. 完整的一致性训练算法见[算法 3](#algorithm-03), 实验中使用的调度函数见[第 10 节](#section-10).

与一致性蒸馏类似, 当 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ 时, 一致性训练损失 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ 可以推广到连续时间 (即 $N\to\infty$), 如[定理 6](#theorem-06) 所示. 这个连续时间损失函数不需要 $N$ 或 $\mu$ 的调度函数, 但需要前向模式自动微分才能高效实现. 与离散时间 CT 损失不同, 连续时间目标不存在不理想的“偏差”, 因为在[定理 2](#theorem-02) 中相当于令 $\Delta t\to 0$. 更多细节见[第 9.2 节](#section-9-2).

<span id="section-6"></span>

## 6 实验

我们用一致性蒸馏和一致性训练, 在 CIFAR-10 [Kri09], ImageNet $64\times 64$ [Den09a], LSUN Bedroom $256\times 256$ 和 LSUN Cat $256\times 256$ [Yu15a] 等真实图像数据集上学习一致性模型. 比较结果采用 Fréchet Inception Distance (FID, [Heu17], 越低越好), Inception Score (IS, [Sal16], 越高越好), Precision (Prec., [Kyn19], 越高越好) 和 Recall (Rec., [Kyn19], 越高越好). 更多实验细节见[第 10 节](#section-10).

<span id="figure-03"></span>

![图 3. 在 CIFAR-10 上影响一致性蒸馏 (CD) 和一致性训练 (CT) 的多种因素. CD 的最佳配置为 LPIPS, Heun ODE 求解器和 $N=18$. 与在整个优化过程中固定 $N$ 和 $\mu$ 相比, 我们的自适应调度函数使 CT 的收敛速度显著加快.](./consistency-models/figure-03.png)

**图 3.** 在 CIFAR-10 上影响一致性蒸馏 (CD) 和一致性训练 (CT) 的多种因素. CD 的最佳配置为 LPIPS, Heun ODE 求解器和 $N=18$. 与在整个优化过程中固定 $N$ 和 $\mu$ 相比, 我们的自适应调度函数使 CT 的收敛速度显著加快.

<span id="figure-04"></span>

![图 4. 使用一致性蒸馏 (CD) 的多步图像生成. CD 在所有数据集和采样步数上都优于渐进式蒸馏 (PD), 唯一的例外是 Bedroom $256\times 256$ 上的单步生成.](./consistency-models/figure-04.png)

**图 4.** 使用一致性蒸馏 (CD) 的多步图像生成. CD 在所有数据集和采样步数上都优于渐进式蒸馏 (PD), 唯一的例外是 Bedroom $256\times 256$ 上的单步生成.

<span id="section-6-1"></span>

### 6.1 训练一致性模型

我们在 CIFAR-10 上开展一系列实验, 以了解各种超参数对通过一致性蒸馏 (CD) 和一致性训练 (CT) 所训练模型性能的影响. 首先考察 CD 中度量函数 $d(\cdot,\cdot)$, ODE 求解器和离散化步数 $N$ 的影响, 随后考察 CT 中调度函数 $N(\cdot)$ 和 $\mu(\cdot)$ 的影响.

在 CD 实验中, 我们以平方 $\ell_{2}$ 距离 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$, $\ell_{1}$ 距离 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$ 和学习感知图像块相似度 (LPIPS, [Zha18d]) 作为度量函数. 对 ODE 求解器, 我们比较 [Kar22] 详述的 Euler 前向方法和 Heun 二阶方法. 对离散化步数 $N$, 比较 $N\in\{9,12,18,36,50,60,80,120\}$. 实验中所有由 CD 训练的一致性模型均以相应的预训练扩散模型初始化, 由 CT 训练的模型则随机初始化.

如[图 3(a)](#figure-03) 所示, CD 的最佳度量是 LPIPS, 它在所有训练迭代中都大幅优于 $\ell_{1}$ 和 $\ell_{2}$. 这是符合预期的, 因为一致性模型在 CIFAR-10 上的输出是图像, 而 LPIPS 专门用于度量自然图像间的相似度. 接下来, 我们考察哪种 ODE 求解器和离散化步数 $N$ 最适合 CD. 如[图 3(b)](#figure-03) 和[图 3(c)](#figure-03) 所示, Heun ODE 求解器和 $N=18$ 是最佳选择. 尽管训练的是一致性模型而非扩散模型, 两者仍与 [Kar22] 的建议一致. 此外, [图 3(b)](#figure-03) 表明, $N$ 相同时, Heun 二阶求解器始终优于 Euler 一阶求解器. 这与[定理 1](#theorem-01) 一致: 在相同 $N$ 下, 由更高阶 ODE 求解器训练的最优一致性模型估计误差更小. [图 3(c)](#figure-03) 的结果还表明, 一旦 $N$ 足够大, CD 的性能便不再对 $N$ 敏感. 基于这些观察, 除非另有说明, 下文对 CD 使用 LPIPS 和 Heun ODE 求解器. CD 的 $N$ 在 CIFAR-10 和 ImageNet $64\times 64$ 上遵循 [Kar22] 的建议, 在其他数据集上则单独调优 (详见[第 10 节](#section-10)).

由于 CD 与 CT 联系紧密, 本文所有 CT 实验均采用 LPIPS. 与 CD 不同, CT 的损失函数不依赖任何特定的数值 ODE 求解器, 因而无需使用 Heun 二阶求解器. 如[图 3(d)](#figure-03) 所示, CT 的收敛对 $N$ 高度敏感——较小的 $N$ 收敛更快但样本更差, 较大的 $N$ 收敛更慢, 但收敛后的样本更好. 这与[第 5 节](#section-5) 的分析一致, 也促使我们在实践中逐渐增大 CT 的 $N$ 和 $\mu$, 以平衡收敛速度与样本质量. [图 3(d)](#figure-03) 表明, $N$ 和 $\mu$ 的自适应调度显著改善了 CT 的收敛速度和样本质量. 实验中, 我们针对不同分辨率的图像分别调优调度 $N(\cdot)$ 和 $\mu(\cdot)$, 更多细节见[第 10 节](#section-10).

<span id="table-01"></span>

![表 1. CIFAR-10 上的样本质量. $^{\ast}$需要构造合成数据进行蒸馏的方法.](./consistency-models/table-01.png)

**表 1.** CIFAR-10 上的样本质量. $^{\ast}$需要构造合成数据进行蒸馏的方法.

<span id="table-02"></span>

![表 2. ImageNet $64\times 64$ 以及 LSUN Bedroom 和 Cat $256\times 256$ 上的样本质量. $^{\dagger}$蒸馏技术.](./consistency-models/table-02.png)

**表 2.** ImageNet $64\times 64$ 以及 LSUN Bedroom 和 Cat $256\times 256$ 上的样本质量. $^{\dagger}$蒸馏技术.

<span id="section-6-2"></span>

### 6.2 少步图像生成

**蒸馏** 在现有文献中, 与我们的一致性蒸馏 (CD) 最具可比性的方法是渐进式蒸馏 (PD, [Sal22]); 到目前为止, 两者也是仅有的*不在蒸馏前构造合成数据*的蒸馏方法. 与此形成鲜明对比的是, 知识蒸馏 [Luh21] 和 DFNO [Zhe22g] 等其他蒸馏技术, 必须使用计算昂贵的数值 ODE/SDE 求解器从扩散模型生成大量样本, 以准备大型合成数据集. 我们在 CIFAR-10, ImageNet $64\times 64$ 和 LSUN $256\times 256$ 上全面比较 PD 与 CD, 所有结果见[图 4](#figure-04). 所有方法都从我们自行预训练的 EDM [Kar22] 模型进行蒸馏. 值得注意的是, 在所有采样迭代数下, *与 [Sal22] 原文所用的平方 $\ell_{2}$ 距离相比, 使用 LPIPS 度量始终能改善 PD*. 随着采样步数增加, PD 和 CD 都有所改善. 在所考虑的所有数据集, 采样步数和度量函数上, CD 始终优于 PD, 唯一例外是 Bedroom $256\times 256$ 的单步生成, 此时采用 $\ell_{2}$ 的 CD 略逊于采用 $\ell_{2}$ 的 PD. 如[表 1](#table-01) 所示, CD 甚至优于知识蒸馏 [Luh21] 和 DFNO [Zhe22g] 等需要构造合成数据集的蒸馏方法.

<span id="figure-05"></span>

![图 5. 由 EDM (*上*), CT + 单步生成 (*中*) 和 CT + 两步生成 (*下*) 生成的样本. 对应图像均由同一个初始噪声生成.](./consistency-models/figure-05.png)

**图 5.** 由 EDM (*上*), CT + 单步生成 (*中*) 和 CT + 两步生成 (*下*) 生成的样本. 对应图像均由同一个初始噪声生成.

<span id="figure-06"></span>

![图 6. 使用在 LSUN Bedroom $256\times 256$ 上通过一致性蒸馏训练的一致性模型进行零样本图像编辑.](./consistency-models/figure-06.png)

**图 6.** 使用在 LSUN Bedroom $256\times 256$ 上通过一致性蒸馏训练的一致性模型进行零样本图像编辑.

**直接生成** 在[表 1](#table-01) 和[表 2](#table-02) 中, 我们比较一致性训练 (CT) 与其他生成模型使用单步和两步生成时的样本质量, 并列出 PD 与 CD 的结果作为参考. 两张表报告的 PD 结果都使用 $\ell_{2}$ 度量函数, 因为这是 [Sal22] 原文采用的默认设置. 为公平比较, 我们确保 PD 与 CD 蒸馏相同的 EDM 模型. 从[表 1](#table-01) 和[表 2](#table-02) 可以看到, CT 在 CIFAR-10 上大幅优于现有的单步, 非对抗式生成模型, 即 VAE 和归一化流. 此外, *CT 无需依赖蒸馏, 便能达到与 PD 单步样本相当的质量*. [图 5](#figure-05) 给出了 EDM 样本 (上), 单步 CT 样本 (中) 和两步 CT 样本 (下). 在[第 12 节](#section-12) 的[图 14](#figure-14), [图 15](#figure-15), [图 16](#figure-16), [图 17](#figure-17), [图 18](#figure-18), [图 19](#figure-19), [图 20](#figure-20) 和[图 21](#figure-21) 中, 我们给出 CD 与 CT 的更多样本. 值得注意的是, *由同一初始噪声向量得到的所有样本在结构上都非常相似*, 尽管 CT 与 EDM 模型彼此独立训练. 由于 EDM 不会出现模式坍塌, 这说明 CT 也不太可能出现模式坍塌.

<span id="section-6-3"></span>

### 6.3 零样本图像编辑

与扩散模型类似, 一致性模型可以修改[算法 1](#algorithm-01) 中的多步采样过程, 进行零样本图像编辑. 我们用一个在 LSUN Bedroom 数据集上通过一致性蒸馏训练的模型展示这一能力. [图 6(a)](#figure-06) 表明, 尽管该模型从未针对着色任务进行训练, 它仍能在测试时为灰度卧室图像着色. [图 6(b)](#figure-06) 表明, 同一个模型可以从低分辨率输入生成高分辨率图像. [图 6(c)](#figure-06) 还表明, 它可以像扩散模型的 SDEdit [Men22] 一样, 根据人类创建的笔画输入生成图像. 这种编辑能力同样是零样本的, 因为模型未曾针对笔画输入进行训练. 在[第 11 节](#section-11), 我们还展示了一致性模型在图像修复 (见[图 10](#figure-10)), 插值 (见[图 11](#figure-11)) 和去噪 (见[图 12](#figure-12)) 上的零样本能力, 并给出更多着色 (见[图 8](#figure-08)), 超分辨率 (见[图 9](#figure-09)) 和笔画引导图像生成 (见[图 13](#figure-13)) 的示例.

<span id="section-7"></span>

## 7 结论

我们提出了一致性模型, 这类生成模型专为支持单步和少步生成而设计. 实验表明, 在多个图像基准和较少采样迭代下, 我们的一致性蒸馏方法优于现有的扩散模型蒸馏技术. 此外, 作为独立生成模型, 一致性模型生成的样本优于除 GAN 外的现有单步生成模型. 与扩散模型类似, 它也支持图像修复, 着色, 超分辨率, 去噪, 插值和笔画引导图像生成等零样本图像编辑应用.

此外, 一致性模型与深度 Q-learning [Mni15] 和基于动量的对比学习 [Gri20, He20a] 等其他领域使用的技术有显著相似之处. 这为不同领域之间相互借鉴思想和方法提供了可期的机会.

## 致谢

感谢 Alex Nichol 审阅本文并提供宝贵反馈, 感谢 Chenlin Meng 提供笔画引导图像生成实验所需的笔画输入, 也感谢 OpenAI Algorithms 团队.

<span id="section-8"></span>

## 8 证明

<span id="section-8-1"></span>

### 8.1 记号

我们用 ${\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t)$ 表示由 ${\bm{\theta}}$ 参数化的一致性模型, 用 ${\bm{f}}({\mathbf{x}},t;{\bm{\phi}})$ 表示[公式 3](#equation-03) 中经验 PF ODE 的一致性函数. 这里, ${\bm{\phi}}$ 表示该函数对预训练分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 的依赖. 对于[公式 2](#equation-02) 中 PF ODE 的一致性函数, 我们将其记为 ${\bm{f}}({\mathbf{x}},t)$. 给定多变量函数 ${\bm{h}}({\mathbf{x}},{\mathbf{y}})$, 令 $\partial_{1}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ 表示 ${\bm{h}}$ 关于 ${\mathbf{x}}$ 的 Jacobian, 类似地, $\partial_{2}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ 表示 ${\bm{h}}$ 关于 ${\mathbf{y}}$ 的 Jacobian. 除非另有说明, 假设 ${\mathbf{x}}$ 是从数据分布 $p_{\text{data}}({\mathbf{x}})$ 采样的随机变量, $n$ 从 $\llbracket 1,N-1\rrbracket$ 中均匀随机采样, ${\mathbf{x}}_{t_{n}}$ 从 $\mathcal{N}({\mathbf{x}};t_{n}^{2}{\bm{I}})$ 采样. 这里, $\llbracket 1,N-1\rrbracket$ 表示整数集合 $\{1,2,\cdots,N-1\}$. 还需回顾我们的定义

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

其中, $\Phi(\cdots;{\bm{\phi}})$ 表示由分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 定义的经验 PF ODE 的单步 ODE 求解器更新函数. 默认情况下, $\mathbb{E}[\cdot]$ 表示对表达式中所有相关随机变量的期望.

<span id="section-8-2"></span>

### 8.2 一致性蒸馏

<span id="theorem-01-appendix"></span>

**定理 1.** 令 $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$, 并令 ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ 为[公式 3](#equation-03) 中经验 PF ODE 的一致性函数. 假设 ${\bm{f}}_{\bm{\theta}}$ 满足 Lipschitz 条件: 存在 $L>0$, 使得对所有 $t\in[\epsilon,T]$, ${\mathbf{x}}$ 和 ${\mathbf{y}}$, 都有 $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$. 进一步假设, 对所有 $n\in\llbracket 1,N-1\rrbracket$, 在 $t_{n+1}$ 调用的 ODE 求解器局部误差由 $O((t_{n+1}-t_{n})^{p+1})$ 一致界定, 且 $p\geq 1$. 那么, 如果 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$, 则有

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details 证明

<span id="equation-11"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]=0.
$$

根据定义, 有 $p_{t_{n}}({\mathbf{x}}_{t_{n}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t_{n}^{2}{\bm{I}})$, 其中 $t_{n}\geq\epsilon>0$. 因此, 对每个 ${\mathbf{x}}_{t_{n}}$ 和 $1\leq n\leq N$, 都有 $p_{t_{n}}({\mathbf{x}}_{t_{n}})>0$. 所以, [公式 11](#equation-11) 蕴含

<span id="equation-12"></span>

$$
\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\equiv 0.
$$

由于 $\lambda(\cdot)>0$ 且 $d({\mathbf{x}},{\mathbf{y}})=0\Leftrightarrow{\mathbf{x}}={\mathbf{y}}$, 这进一步蕴含

<span id="equation-13"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\equiv{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}).
$$

现在令 ${\bm{e}}_{n}$ 表示 $t_{n}$ 处的误差向量, 定义为

$$
{\bm{e}}_{n}\coloneqq{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}).
$$

不难得到如下递推关系

<span id="equation-14"></span>

$$
\begin{aligned}
{\bm{e}}_{n+1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{e}}_{n},
\end{aligned}
$$

其中, (i) 来自[公式 13](#equation-13) 以及 ${\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}})={\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}})$. 由于 ${\bm{f}}_{\bm{\theta}}(\cdot,t_{n})$ 的 Lipschitz 常数为 $L$, 有

$$
\begin{aligned}
\|{\bm{e}}_{n+1}\|_{2} & \leq\|{\bm{e}}_{n}\|_{2}+L\|\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}-{\mathbf{x}}_{t_{n}}\|_{2} \\
\mathrel{{\mathop{=}\limits}}\|{\bm{e}}_{n}\|_{2}+L\cdot O((t_{n+1}-t_{n})^{p+1}) \\
=\|{\bm{e}}_{n}\|_{2}+O((t_{n+1}-t_{n})^{p+1}),
\end{aligned}
$$

其中, (i) 成立是因为 ODE 求解器的局部误差由 $O((t_{n+1}-t_{n})^{p+1})$ 界定. 此外, 我们注意到 ${\bm{e}}_{1}=\bm{0}$, 因为

$$
\begin{aligned}
{\bm{e}}_{1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{1}},t_{1})-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\mathbf{x}}_{t_{1}} \\
=\bm{0}.
\end{aligned}
$$

这里, (i) 成立是因为一致性模型的参数化满足 ${\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}})={\mathbf{x}}_{t_{1}}$, (ii) 则由 ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ 的定义推出. 因而可以对递推公式[公式 14](#equation-14) 使用归纳法, 得到

$$
\begin{aligned}
\|{\bm{e}}_{n}\|_{2} & \leq\|{\bm{e}}_{1}\|_{2}+\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((t_{k+1}-t_{k})^{p}) \\
\leq\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((\Delta t)^{p}) \\
=O((\Delta t)^{p})\sum_{k=1}^{n-1}(t_{k+1}-t_{k}) \\
=O((\Delta t)^{p})(t_{n}-t_{1}) \\
\leq O((\Delta t)^{p})(T-\epsilon) \\
=O((\Delta t)^{p}),
\end{aligned}
$$

从而完成证明.

:::

<span id="section-8-3"></span>

### 8.3 一致性训练

下面的引理给出分数函数的一个无偏估计量, 这对[定理 2](#theorem-02) 的证明至关重要.

<span id="lemma-01"></span>

**引理 1.** 令 ${\mathbf{x}}\sim p_{\text{data}}({\mathbf{x}})$, ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$, 且 $p_{t}({\mathbf{x}}_{t})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$. 则 $\nabla\log p_{t}({\mathbf{x}})=-\mathbb{E}[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}]$.

::: details 证明

$$
\begin{aligned}
\nabla\log p_{t}({\mathbf{x}}_{t}) & =\frac{\int p_{\text{data}}({\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{p_{t}({\mathbf{x}}_{t})} \\
=\int\frac{p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})}{p_{t}({\mathbf{x}}_{t})}\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
\mathrel{{\mathop{=}\limits}}\int p({\mathbf{x}}\mid{\mathbf{x}}_{t})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
=\mathbb{E}[\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mid{\mathbf{x}}_{t}] \\
=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}\right],
\end{aligned}
$$

其中, (i) 来自 Bayes 规则.

:::

<span id="theorem-02-appendix"></span>

**定理 2.** 令 $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$. 假设 $d$ 和 ${\bm{f}}_{{\bm{\theta}}^{-}}$ 均二阶连续可微且二阶导数有界, 权重函数 $\lambda(\cdot)$ 有界, 并且 $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$. 进一步假设使用 Euler ODE 求解器, 且预训练分数模型与真实值一致, 即 $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$. 那么,

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

其中, 期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$ 和 ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ 计算. 一致性训练目标记为 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, 定义如下

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

其中 ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. 此外, 如果 $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$, 则 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$.

::: details 证明

<span id="equation-15"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}}+(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}),t_{n}))] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})+\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}) \\
\qquad+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|))] \\
= & \mathbb{E}\{\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[ \\
\quad\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|)]\} \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})]\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)].
\end{aligned}
$$

然后, 对[公式 15](#equation-15) 应用[引理 1](#lemma-01), 并沿反方向使用 Taylor 展开, 得到

<span id="equation-16"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\mathbb{E}\left[\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\Big|{\mathbf{x}}_{t_{n+1}}\right]\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
\mathrel{{\mathop{=}\limits}} & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\bigg[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})) \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right] \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]+o(|t_{n+1}-t_{n}|)\bigg] \\
\qquad+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})t_{n+1}\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n+1}{\mathbf{z}}+(t_{n}-t_{n+1}){\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(\Delta t)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+o(\Delta t) \\
= & \mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
\end{aligned}
$$

其中, (i) 来自全期望公式, 且 ${\mathbf{z}}\coloneqq\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}}\sim\mathcal{N}(\bm{0},{\bm{I}})$. 这表明 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t)$, 从而完成[公式 9](#equation-09) 的证明. 此外, 只要 $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$, 就有 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$. 否则, $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})<O(\Delta t)$, 因而 $\lim_{\Delta t\to 0}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=0$, 这与 $\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$ 明显矛盾.

:::

<span id="remark-01"></span>

**备注 1.** 当 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ 这一条件不满足时, 例如 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ 的情况, 仍可根据[定理 6](#theorem-06) 的结果, 说明 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ 作为一致性模型训练目标的合理性.

<span id="section-9"></span>

## 9 连续时间推广

在适当条件下, 一致性蒸馏和一致性训练目标可以推广到无限多个时间步 ($N\to\infty$).

<span id="section-9-1"></span>

### 9.1 连续时间一致性蒸馏

根据 ${\bm{\theta}}^{-}={\bm{\theta}}$ 还是 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ (等同于设置 $\mu=0$), 一致性蒸馏目标 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 有两种可能的连续时间推广. 给定二阶连续可微的度量函数 $d({\mathbf{x}},{\mathbf{y}})$, 将 ${\bm{G}}({\mathbf{x}})$ 定义为矩阵, 其第 $(i,j)$ 个元素为

$$
[{\bm{G}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{x}},{\mathbf{y}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

类似地, 将 ${\bm{H}}({\mathbf{x}})$ 定义为

$$
[{\bm{H}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{y}},{\mathbf{x}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

矩阵 ${\bm{G}}$ 和 ${\bm{H}}$ 对构造一致性蒸馏的连续时间目标至关重要. 此外, 将 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ 关于 ${\mathbf{x}}$ 的 Jacobian 记为 $\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial{\mathbf{x}}}$.

当 ${\bm{\theta}}^{-}={\bm{\theta}}$ (不使用 stopgrad 算子) 时, 有以下理论结果.

<span id="theorem-03"></span>

**定理 3.** 令 $t_{n}=\tau(\frac{n-1}{N-1})$, 其中 $n\in\llbracket 1,N\rrbracket$, $\tau(\cdot)$ 是满足 $\tau(0)=\epsilon$ 和 $\tau(1)=T$ 的严格单调函数. 假设 $\tau$ 在 $[0,1]$ 上连续可微, $d$ 三阶连续可微且三阶导数有界, ${\bm{f}}_{{\bm{\theta}}}$ 二阶连续可微且一阶和二阶导数有界. 进一步假设权重函数 $\lambda(\cdot)$ 有界, 且 $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$. 那么, 在一致性蒸馏中使用 Euler 求解器时, 有

<span id="equation-17"></span>

$$
\lim_{N\to\infty}(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

其中, $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$ 定义为

<span id="equation-18"></span>

$$
\frac{1}{2}\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

上式的期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$ 和 ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$ 计算.

::: details 证明

<span id="equation-19"></span>

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})={\bm{f}}_{{\bm{\theta}}}({\mathbf{x}}_{t_{n+1}}+t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u,t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}) \\
= & t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u+O((\Delta u)^{2}),
\end{aligned}
$$

注意 $\tau^{\prime}(u_{n})=\frac{1}{\tau^{-1}(t_{n+1})}$. 随后对一致性蒸馏损失进行 Taylor 展开, 得到

$$
\begin{aligned}
(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2(\Delta u)^{2}}\bigg(\mathbb{E}\{\lambda(t_{n})\tau^{\prime}(u_{n})^{2}[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\lambda(t_{n})\tau^{\prime}(u_{n})^{2}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\frac{\lambda(t_{n})}{[(\tau^{-1})^{\prime}(t_{n})]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

其中, (i) 通过将 $d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),\cdot)$ 展开到二阶, 并注意到 $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ 和 $\nabla_{\mathbf{y}}d({\mathbf{x}},{\mathbf{y}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$ 而得到. (ii) 由[公式 19](#equation-19) 得到. 当 $\Delta u\to 0$, 或等价地 $N\to\infty$ 时, 对第 B.1 节两侧取极限, 得到[公式 17](#equation-17), 从而完成证明.

:::

<span id="remark-02"></span>

**备注 2.** 尽管[定理 3](#theorem-03) 为了技术上的简便而假设使用 Euler ODE 求解器, 但我们认为可以为更一般的求解器推导类似结果, 因为当 $N\to\infty$ 时, 所有 ODE 求解器的表现都应相近. 我们把[定理 3](#theorem-03) 更一般的版本留作未来工作.

<span id="remark-03"></span>

**备注 3.** [定理 3](#theorem-03) 表明, 可以通过最小化 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$ 来训练一致性模型. 特别地, 当 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$ 时, 有

<span id="equation-26"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\|\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\|^{2}_{2}\right].
$$

然而, 计算这个连续时间目标的损失函数时, 需要以 Jacobian-向量积作为子程序. 在不支持前向模式自动微分的深度学习框架中, 这可能速度缓慢且实现繁琐.

<span id="remark-04"></span>

**备注 4.** 如果 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ 与分数模型 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 对应的经验 PF ODE 真实一致性函数相同, 则

$$
\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial{\mathbf{x}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\equiv 0
$$

因而 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$. 可以注意到, 对所有 $t\in[\epsilon,T]$ 都有 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)\equiv{\mathbf{x}}_{\epsilon}$, 再对此恒等式求时间导数, 从而证明该结论:

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)\equiv{\mathbf{x}}_{\epsilon} \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}[-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)]+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\equiv 0.
\end{aligned}
$$

上述观察进一步说明了 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$ 的动机, 因为当且仅当一致性模型与真实一致性函数相同时, 它达到最小值.

对于 $\ell_{1}$ 范数等某些度量函数, Hessian ${\bm{G}}({\mathbf{x}})$ 为零, 因而[定理 3](#theorem-03) 没有实质内容. 下面表明, 只需稍微修改[定理 3](#theorem-03) 的证明, 即可得到适用于 $\ell_{1}$ 范数的非平凡陈述.

<span id="theorem-04"></span>

**定理 4.** 令 $t_{n}=\tau(\frac{n-1}{N-1})$, 其中 $n\in\llbracket 1,N\rrbracket$, $\tau(\cdot)$ 是满足 $\tau(0)=\epsilon$ 和 $\tau(1)=T$ 的严格单调函数. 假设 $\tau$ 在 $[0,1]$ 上连续可微, ${\bm{f}}_{{\bm{\theta}}}$ 二阶连续可微且一阶和二阶导数有界. 进一步假设权重函数 $\lambda(\cdot)$ 有界, 且 $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$. 假设使用 Euler ODE 求解器, 并在一致性蒸馏中设置 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$. 则有

<span id="equation-27"></span>

$$
\lim_{N\to\infty}(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

其中

$$
\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}\| t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\|_{1}\right]
$$

上式的期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$ 和 ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$ 计算.

::: details 证明

<span id="equation-28"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})\|_{1}] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\left[\lambda(t_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})+O((\Delta u)^{2})\|_{1}\right] \\
= & \mathbb{E}\left[\lambda(t_{n})\tau^{\prime}(u_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right] \\
= & \mathbb{E}\left[\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right]
\end{aligned}
$$

其中, (i) 通过把[公式 19](#equation-19) 代入前式得到. 当 $\Delta u\to 0$, 或等价地 $N\to\infty$ 时, 对[公式 28](#equation-28) 两侧取极限, 得到[公式 27](#equation-27), 从而完成证明.

:::

<span id="remark-05"></span>

**备注 5.** 根据[定理 4](#theorem-04), 可以通过最小化 $\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$ 来训练一致性模型. 此外, 使用[备注 4](#remark-04) 中相同的推理可知, 当且仅当对所有 ${\mathbf{x}}_{t}\in\mathbb{R}^{d}$ 和 $t\in[\epsilon,T]$ 都有 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)={\mathbf{x}}_{\epsilon}$ 时, $\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$.

在第二种情况 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ 下, 可以推导一个所谓的“伪目标”, 当 $N\to\infty$ 时, 它的梯度与 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 的梯度一致. 通过梯度下降最小化这个伪目标, 提供了另一种经蒸馏训练一致性模型的方法. 下面的定理给出该伪目标.

<span id="theorem-05"></span>

**定理 5.** 令 $t_{n}=\tau(\frac{n-1}{N-1})$, 其中 $n\in\llbracket 1,N\rrbracket$, $\tau(\cdot)$ 是满足 $\tau(0)=\epsilon$ 和 $\tau(1)=T$ 的严格单调函数. 假设 $\tau$ 在 $[0,1]$ 上连续可微, $d$ 三阶连续可微且三阶导数有界, ${\bm{f}}_{{\bm{\theta}}}$ 二阶连续可微且一阶和二阶导数有界. 进一步假设权重函数 $\lambda(\cdot)$ 有界, $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$, 且 $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\|_{2}<\infty$. 假设使用 Euler ODE 求解器, 且在一致性蒸馏中 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$. 那么,

<span id="equation-29"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}),
$$

其中

<span id="equation-30"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

上式的期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$ 和 ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$ 计算.

::: details 证明

<span id="equation-33"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]
\end{aligned}
$$

其中, (i) 通过将 $d(\cdot,{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))$ 展开到二阶, 并利用 $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ 和 $\nabla_{\mathbf{y}}d({\mathbf{y}},{\mathbf{x}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$ 得到. 接下来, 计算[公式 33](#equation-33) 关于 ${\bm{\theta}}$ 的梯度并化简, 得到

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

这里, (i) 来自链式法则; 由于 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, 有 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\equiv{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}},t)$, 因而 (ii) 来自[公式 19](#equation-19). 当 $\Delta u\to 0$ (或 $N\to\infty$) 时, 对第 B.1 节两侧取极限, 得到[公式 29](#equation-29), 从而完成证明.

:::

<span id="remark-06"></span>

**备注 6.** 当 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$ 时, 伪目标 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 可化简为

<span id="equation-42"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

<span id="remark-07"></span>

**备注 7.** [定理 5](#theorem-05) 定义的目标 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 只有其梯度有意义——不能通过跟踪 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 的值来衡量训练进度, 但仍可对此目标应用梯度下降, 从预训练扩散模型蒸馏一致性模型. 由于它不是通常意义上的损失函数, 我们称其为一致性蒸馏的“伪目标”.

<span id="remark-08"></span>

**备注 8.** 沿用[备注 4](#remark-04) 的推理不难得出: 如果 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ 与包含 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ 的经验 PF ODE 真实一致性函数相同, 则 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=0$, 且 $\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\bm{0}$. 但反命题一般不成立. 这使 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 区别于 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})$, 后者是真正的损失函数.

<span id="section-9-2"></span>

### 9.2 连续时间一致性训练

一个重要观察是, 无需任何预训练扩散模型也能估计[定理 5](#theorem-05) 中的伪目标, 因而可以直接对一致性模型进行一致性训练. 更准确地说, 有以下结果.

<span id="theorem-06"></span>

**定理 6.** 令 $t_{n}=\tau(\frac{n-1}{N-1})$, 其中 $n\in\llbracket 1,N\rrbracket$, $\tau(\cdot)$ 是满足 $\tau(0)=\epsilon$ 和 $\tau(1)=T$ 的严格单调函数. 假设 $\tau$ 在 $[0,1]$ 上连续可微, $d$ 三阶连续可微且三阶导数有界, ${\bm{f}}_{{\bm{\theta}}}$ 二阶连续可微且一阶和二阶导数有界. 进一步假设权重函数 $\lambda(\cdot)$ 有界, $\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$, $\sup_{{\mathbf{x}},t\in[\epsilon,T]}\|\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\|_{2}<\infty$, 并且 ${\bm{\phi}}$ 表示满足 ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$ 的扩散模型参数. 那么, 如果 ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$, 则有

<span id="equation-43"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-}),
$$

其中 $\mathcal{L}^{N}_{\text{CD}}$ 使用 Euler ODE 求解器, 并且

<span id="equation-44"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

上式的期望针对 ${\mathbf{x}}\sim p_{\text{data}}$, $u\sim\mathcal{U}[0,1]$, $t=\tau(u)$ 和 ${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$ 计算.

::: details 证明

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \begin{multlined}\frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

其中 ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$; (i) 先将 $d(\cdot,{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))$ 展开到二阶, 再注意到 $d({\mathbf{x}},{\mathbf{x}})\equiv 0$ 和 $\nabla_{\mathbf{y}}d({\mathbf{y}},{\mathbf{x}})|_{{\mathbf{y}}={\mathbf{x}}}\equiv\bm{0}$ 而得到. 接下来, 计算第 B.2 节关于 ${\bm{\theta}}$ 的梯度并化简, 得到

<span id="equation-59"></span>

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-}) \\
= & \begin{multlined}\frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\tau^{\prime}(u_{n})\Delta u\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)] \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]
\end{aligned}
$$

这里, (i) 来自链式法则, (ii) 来自 Taylor 展开. 当 $\Delta u\to 0$ 或 $N\to\infty$ 时, 对[公式 59](#equation-59) 两侧取极限, 得到[公式 43](#equation-43) 中的第二个等式.

现在证明第一个等式. 再次应用 Taylor 展开, 得到

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\partial_{1}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\bigg[\partial_{1}d({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
+{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))+O(|\Delta u|^{2})\bigg]\bigg\}\end{multlined} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

其中, (i) 成立是因为 ${\mathbf{x}}_{t_{n+1}}={\mathbf{x}}+t_{n+1}{\mathbf{z}}$, 且 $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}={\mathbf{x}}_{t_{n+1}}-(t_{n}-t_{n+1})t_{n+1}\frac{-({\mathbf{x}}_{t_{n+1}}-{\mathbf{x}})}{t_{n+1}^{2}}={\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1}){\mathbf{z}}={\mathbf{x}}+t_{n}{\mathbf{z}}$. 由于 (i) 与第 B.2 节相同, 可以沿用第 B.2 节到[公式 59](#equation-59) 的推理, 得到 $\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$, 从而完成证明.

:::

<span id="remark-09"></span>

**备注 9.** 注意, $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$ 不依赖扩散模型参数 ${\bm{\phi}}$, 因而无需任何预训练扩散模型即可优化.

<span id="remark-10"></span>

**备注 10.** 当 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{2}^{2}$ 时, 连续时间一致性训练目标变为

<span id="equation-60"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

<span id="remark-11"></span>

**备注 11.** 与[定理 5](#theorem-05) 中的 $\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ 类似, $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$ 是伪目标; 不能通过监测 $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})$ 的值来跟踪训练, 但仍可对此损失函数应用梯度下降, 直接从数据训练一致性模型 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$. 此外, [备注 8](#remark-08) 中的观察同样成立: 如果 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ 与 PF ODE 的真实一致性函数相同, 则 $\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=0$, 且 $\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=\bm{0}$.

<span id="section-9-3"></span>

### 9.3 实验验证

<span id="figure-07"></span>

![图 7. 离散一致性蒸馏/训练算法与对应连续时间算法的比较.](./consistency-models/figure-07.png)

**图 7.** 离散一致性蒸馏/训练算法与对应连续时间算法的比较.

为了通过实验验证连续时间 CD 和 CT 目标的有效性, 我们在 CIFAR-10 上使用多种损失函数训练一致性模型. 所有结果见[图 7](#figure-07). 所有连续时间实验均设置 $\lambda(t)=(\tau^{-1})^{\prime}(t)$. 其他超参数与[表 3](#table-03) 相同. 为改善性能, 我们偶尔会调整部分超参数. 对于蒸馏, 比较以下目标:

- CD $(\ell_{2})$: 使用 $N=18$ 和 $\ell_{2}$ 度量的一致性蒸馏 $\mathcal{L}^{N}_{\text{CD}}$.

我们没有在[定理 3](#theorem-03) 中考察 LPIPS 度量, 因为最小化所得目标需要通过 LPIPS 使用的 VGG 网络二阶导数进行反向传播, 这不仅计算成本很高, 也容易出现数值不稳定. 从[图 7(a)](#figure-07) 可以看到, 对 LPIPS 和 $\ell_{2}$ 两种度量, 连续时间蒸馏的 stopgrad 版本 ([定理 5](#theorem-05)) 都优于不使用 stopgrad 的版本 ([定理 3](#theorem-03)); 在所有蒸馏方法中, LPIPS 度量表现最佳. 此外, 离散时间一致性蒸馏优于连续时间一致性蒸馏, 原因可能是连续时间目标的方差更大, 而且离散时间目标可以使用有效的高阶 ODE 求解器.

对于一致性训练 (CT), 我们发现使用连续时间目标时, 以预训练 EDM 模型初始化一致性模型对稳定训练很重要. 我们推测这是连续时间损失函数方差较大所致. 因而为了公平比较, 即使离散时间 CT 使用随机初始化也能良好工作, 我们仍在离散时间和连续时间 CT 中, 用 CIFAR-10 上同一个预训练 EDM 模型初始化所有一致性模型. 连续时间 CT 的方差缩减技术留作未来研究.

我们通过实验比较以下目标:

- CT (LPIPS): 使用 $N=120$ 和 LPIPS 度量的一致性训练 $\mathcal{L}_{\text{CT}}^{N}$. 学习率设为 4e-4, 目标网络的 EMA 衰减率设为 0.99. 此处不使用 $N$ 和 $\mu$ 的调度函数, 因为当一致性模型以预训练 EDM 模型初始化时, 它们会减慢学习速度.

如[图 7(b)](#figure-07) 所示, LPIPS 度量改善了连续时间 CT 的性能. 我们还发现, 采用同一种 LPIPS 度量时, 连续时间 CT 优于离散时间 CT. 这可能源于离散时间 CT 的偏差: 对离散时间目标, [定理 2](#theorem-02) 中 $\Delta t>0$; 连续时间 CT 隐式地令 $\Delta t$ 趋近 $0$, 因而没有偏差.

<span id="section-10"></span>

## 10 更多实验细节

<span id="table-03"></span>

![表 3. 训练 CD 和 CT 模型所用的超参数](./consistency-models/table-03.png)

**表 3.** 训练 CD 和 CT 模型所用的超参数

**模型架构.** 模型架构遵循 [Son21, Dha21]. 具体而言, 所有 CIFAR-10 实验均使用 [Son21] 中的 NCSN++ 架构; 在 ImageNet $64\times 64$, LSUN Bedroom $256\times 256$ 和 LSUN Cat $256\times 256$ 上实验时, 则采用 [Dha21] 中相应的网络架构.

**一致性模型的参数化.** 一致性模型使用与 EDM 相同的架构. 唯一的区别是, 我们略微修改了 EDM 中的跳跃连接, 以确保一致性模型满足边界条件. 回顾[第 3 节](#section-3), 我们提出将一致性模型参数化为以下形式:

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t).
$$

EDM [Kar22] 采用

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{t^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}t}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

其中 $\sigma_{\text{data}}=0.5$. 然而, 当最小时间点 $\epsilon\neq 0$ 时, $c_{\text{skip}}$ 和 $c_{\text{out}}$ 的这一选择不满足边界条件. 为解决这个问题, 我们将其修改为

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{(t-\epsilon)^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}(t-\epsilon)}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

显然满足 $c_{\text{skip}}(\epsilon)=1$ 和 $c_{\text{out}}(\epsilon)=0$.

**一致性训练的调度函数.** 如[第 5 节](#section-5) 所述, 为获得最佳性能, 一致性生成需要指定调度函数 $N(\cdot)$ 和 $\mu(\cdot)$. 所有实验均采用以下形式的调度函数:

$$
\begin{aligned}
N(k) & =\left\lceil\sqrt{\frac{k}{K}((s_{1}+1)^{2}-s_{0}^{2})+s_{0}^{2}}-1\right\rceil+1 \\
\mu(k) & =\exp\left(\frac{s_{0}\log\mu_{0}}{N(k)}\right),
\end{aligned}
$$

其中, $K$ 表示训练迭代总数, $s_{0}$ 表示初始离散化步数, $s_{1}>s_{0}$ 表示训练结束时的目标离散化步数, $\mu_{0}>0$ 表示模型训练开始时的 EMA 衰减率.

**训练细节.** 在一致性蒸馏和渐进式蒸馏中, 我们都蒸馏 EDM [Kar22]. 这些 EDM 是我们按照 [Kar22] 给出的规格自行训练的. EDM 原文没有给出 LSUN Bedroom $256\times 256$ 和 Cat $256\times 256$ 数据集的超参数, 因而大体沿用 ImageNet $64\times 64$ 数据集的超参数. 不同之处在于, LSUN Bedroom 和 Cat 分别训练 600k 和 300k 次迭代, 并把批量大小从 4096 降至 2048.

LSUN $256\times 256$ 数据集采用与 ImageNet $64\times 64$ 数据集相同的 EMA 衰减率. 对渐进式蒸馏, 在 CIFAR-10 和 ImageNet $64\times 64$ 上使用 [Sal22] 所述的相同训练设置. 尽管原文没有在 LSUN $256\times 256$ 数据集上测试, 我们采用与 ImageNet $64\times 64$ 相同的设置, 并发现它们效果良好.

所有蒸馏实验都用预训练 EDM 权重初始化一致性模型. 对一致性训练, 则与训练 EDM 时一样随机初始化模型. 所有一致性模型均使用 Rectified Adam 优化器 [Liu19d] 训练, 不采用学习率衰减, warm-up 或权重衰减. 按照 [Kar22], 我们还对一致性蒸馏和一致性训练中的在线一致性模型权重应用 EMA, 并对训练中的在线一致性模型权重应用 EMA. LSUN $256\times 256$ 数据集所用的 EMA 衰减率与 ImageNet $64\times 64$ 相同; 唯一例外是 LSUN Bedroom $256\times 256$ 上的一致性蒸馏, 此时我们发现零 EMA 的效果更好.

在 CIFAR-10 和 ImageNet $64\times 64$ 上使用 LPIPS 度量时, 我们先用双线性上采样把图像缩放到 $224\times 224$, 再输入 LPIPS 网络. 在 LSUN $256\times 256$ 上, 则不缩放输入, 直接计算 LPIPS. 此外, 所有模型和数据集都采用水平翻转进行数据增强. 所有模型均在 Nvidia A100 GPU 集群上训练. 一致性训练和蒸馏的更多超参数见[表 3](#table-03).

<span id="section-11"></span>

## 11 更多零样本图像编辑结果

<span id="algorithm-04"></span>

<div class="paper-algorithm">

**算法 4: 零样本图像编辑.**

- **输入:** 一致性模型 ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$, 时间点序列 $t_{1}>t_{2}>\cdots>t_{N}$, 参考图像 ${\mathbf{y}}$, 可逆线性变换 ${\bm{A}}$, 二值图像掩码 $\bm{\Omega}$.
- ${\mathbf{y}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+\bm{0}\odot\bm{\Omega}]$.
- 采样 ${\mathbf{x}}\sim\mathcal{N}({\mathbf{y}},t_{1}^{2}{\bm{I}})$.
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t_{1})$.
- ${\mathbf{x}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+({\bm{A}}{\mathbf{x}})\odot\bm{\Omega}]$.
- **对于** $n=2$ **到** $N$:
  - 采样 ${\mathbf{x}}\sim\mathcal{N}({\mathbf{x}},(t_{n}^{2}-\epsilon^{2}){\bm{I}})$.
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t_{n})$.
  - ${\mathbf{x}}\gets{\bm{A}}^{-1}[({\bm{A}}{\mathbf{y}})\odot(1-\bm{\Omega})+({\bm{A}}{\mathbf{x}})\odot\bm{\Omega}]$.
- **输出:** ${\mathbf{x}}$.

</div>

一致性模型可以执行多种零样本图像编辑任务. 例如, 这里给出着色 (见[图 8](#figure-08)), 超分辨率 (见[图 9](#figure-09)), 图像修复 (见[图 10](#figure-10)), 插值 (见[图 11](#figure-11)), 去噪 (见[图 12](#figure-12)) 和笔画引导图像生成 (SDEdit, [Men22], 见[图 13](#figure-13)) 的更多结果. 此处使用的一致性模型, 在 LSUN Bedroom $256\times 256$ 上通过一致性蒸馏训练.

除图像插值和去噪外, 只需稍微修改[算法 1](#algorithm-01) 中的多步采样算法, 即可完成上述图像编辑任务. 所得伪代码见[算法 4](#algorithm-04). 其中, ${\mathbf{y}}$ 是引导样本生成的参考图像, $\bm{\Omega}$ 是二值掩码, $\odot$ 计算逐元素乘积, ${\bm{A}}$ 是可逆线性变换. 它把图像映射到潜在空间; 在这个空间中, ${\mathbf{y}}$ 的条件信息通过 $\bm{\Omega}$ 掩码注入迭代生成过程. 除非另有说明, 我们选择

$$
t_{i}=\left(T^{1/\rho}+\frac{i-1}{N-1}(\epsilon^{1/\rho}-T^{1/\rho})\right)^{\rho}
$$

其中, 在 LSUN Bedroom $256\times 256$ 实验中 $N=40$.

下面说明如何使用[算法 4](#algorithm-04) 完成各项任务.

**图像修复.** 使用[算法 4](#algorithm-04) 进行图像修复时, 令 ${\mathbf{y}}$ 为缺失像素被掩蔽的图像, $\bm{\Omega}$ 为二值掩码, 其中 1 表示缺失像素, ${\bm{A}}$ 为恒等变换.

**着色.** 图像着色算法与此类似, 因为把数据变换到解耦空间后, 着色就成为图像修复的一种特殊情况. 具体而言, 令 ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ 为待着色的灰度图像, 并假设 ${\mathbf{y}}$ 的所有通道均相同, 即用 NumPy 记号表示为 ${\mathbf{y}}[:,:,0]={\mathbf{y}}[:,:,1]={\mathbf{y}}[:,:,2]$. 实验中, 该灰度图像的每个通道都通过以下方式对彩色图像的 RGB 通道取加权平均而得到

$$
0.2989R+0.5870G+0.1140B.
$$

将 $\bm{\Omega}\in\{0,1\}^{h\times w\times 3}$ 定义为满足以下条件的二值掩码

$$
\bm{\Omega}[i,j,k]=\begin{cases}1,&\quad\text{$k=1$ 或 $2$}\\
0,&\quad\text{$k=0$}\end{cases}.
$$

令 ${\bm{Q}}\in\mathbb{R}^{3\times 3}$ 为正交矩阵, 其第一列与向量 $(0.2989,0.5870,0.1140)$ 成比例. 这个正交矩阵可以通过 QR 分解轻松得到; 实验中使用

$$
{\bm{Q}}=\begin{pmatrix}0.4471&-0.8204&0.3563\\
0.8780&0.4785&0\\
0.1705&-0.3129&-0.9343\end{pmatrix}.
$$

随后定义线性变换 ${\bm{A}}:{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}\mapsto{\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$, 其中

$$
{\mathbf{y}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{x}}[i,j,l]{\bm{Q}}[l,k].
$$

由于 ${\bm{Q}}$ 是正交矩阵, 逆变换 ${\bm{A}}^{-1}:{\mathbf{y}}\in\mathbb{R}^{h\times w}\mapsto{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}$ 很容易计算, 其中

$$
{\mathbf{x}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{y}}[i,j,l]{\bm{Q}}[k,l].
$$

按上述方式定义 ${\bm{A}}$ 和 $\bm{\Omega}$ 后, 便可以使用[算法 4](#algorithm-04) 进行图像着色.

**超分辨率.** 采用类似策略, 我们用[算法 4](#algorithm-04) 进行图像超分辨率. 为简单起见, 假设下采样图像通过对大小为 $p\times p$ 的非重叠图像块取平均得到. 假设全分辨率图像的形状为 $h\times w\times 3$. 令 ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ 表示直接上采样到全分辨率的低分辨率图像, 每个非重叠图像块中的像素取值相同. 另外, 令 $\bm{\Omega}\in\{0,1\}^{h/p\times w/p\times p^{2}\times 3}$ 为满足以下条件的二值掩码

$$
\bm{\Omega}[i,j,k,l]=\begin{cases}1,&\quad k\geq 1\\
0,&\quad k=0\end{cases}.
$$

与图像着色类似, 超分辨率需要正交矩阵 ${\bm{Q}}\in\mathbb{R}^{p^{2}\times p^{2}}$, 其第一列为 $(\frac{1}{p},\frac{1}{p},\cdots,\frac{1}{p})$. 这个正交矩阵可通过 QR 分解得到. 为执行超分辨率, 定义线性变换 ${\bm{A}}:{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}\mapsto{\mathbf{y}}\in\mathbb{R}^{h/p\times w/p\times p^{2}\times 3}$, 其中

$$
{\mathbf{y}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{x}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[m,k].
$$

逆变换 ${\bm{A}}^{-1}:{\mathbf{y}}\in\mathbb{R}^{h/p\times w/p\times p^{2}\times 3}\mapsto{\mathbf{x}}\in\mathbb{R}^{h\times w\times 3}$ 也很容易推导, 即

$$
{\mathbf{x}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{y}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[k,m].
$$

按上述方式定义 ${\bm{A}}$ 和 $\bm{\Omega}$ 后, 便可以使用[算法 4](#algorithm-04) 进行图像超分辨率.

**笔画引导图像生成.** 我们也可以使用[算法 4](#algorithm-04), 按照 SDEdit [Men22] 进行笔画引导图像生成. 具体而言, 令 ${\mathbf{y}}\in\mathbb{R}^{h\times w\times 3}$ 为笔画图. 设置 ${\bm{A}}={\bm{I}}$, 并将 $\bm{\Omega}\in\mathbb{R}^{h\times w\times 3}$ 定义为全 1 矩阵. 实验中设置 $t_{1}=5.38$, $t_{2}=2.24$ 和 $N=2$.

**去噪.** 单个一致性模型可以对受到不同尺度高斯噪声扰动的图像进行去噪. 假设输入图像 ${\mathbf{x}}$ 受到 $\mathcal{N}(\bm{0};\sigma^{2}{\bm{I}})$ 的扰动. 只要 $\sigma\in[\epsilon,T]$, 计算 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},\sigma)$ 即可得到去噪图像.

**插值.** 我们可以在一致性模型生成的两幅图像之间插值. 假设第一个样本 ${\mathbf{x}}_{1}$ 由噪声向量 ${\mathbf{z}}_{1}$ 生成, 第二个样本 ${\mathbf{x}}_{2}$ 由噪声向量 ${\mathbf{z}}_{2}$ 生成. 换言之, ${\mathbf{x}}_{1}={\bm{f}}_{\bm{\theta}}({\mathbf{z}}_{1},T)$ 且 ${\mathbf{x}}_{2}={\bm{f}}_{\bm{\theta}}({\mathbf{z}}_{2},T)$. 为在 ${\mathbf{x}}_{1}$ 与 ${\mathbf{x}}_{2}$ 之间插值, 首先使用球面线性插值得到

$$
{\mathbf{z}}=\frac{\sin[(1-\alpha)\psi]}{\sin(\psi)}{\mathbf{z}}_{1}+\frac{\sin(\alpha\psi)}{\sin(\psi)}{\mathbf{z}}_{2},
$$

其中 $\alpha\in[0,1]$, 且 $\psi=\arccos(\frac{{\mathbf{z}}_{1}^{\top}{\mathbf{z}}_{2}}{\|{\mathbf{z}}_{1}\|_{2}\|{\mathbf{z}}_{2}\|_{2}})$, 随后计算 ${\bm{f}}_{\bm{\theta}}({\mathbf{z}},T)$, 生成插值图像.

<span id="figure-08"></span>

![图 8. 灰度图像 (左), 一致性模型着色后的图像 (中) 和真实图像 (右).](./consistency-models/figure-08.png)

**图 8.** 灰度图像 (左), 一致性模型着色后的图像 (中) 和真实图像 (右).

<span id="figure-09"></span>

![图 9. 分辨率为 $32\times 32$ 的下采样图像 (左), 一致性模型生成的全分辨率 ($256\times 256$) 图像 (中), 以及分辨率为 $256\times 256$ 的真实图像 (右).](./consistency-models/figure-09.png)

**图 9.** 分辨率为 $32\times 32$ 的下采样图像 (左), 一致性模型生成的全分辨率 ($256\times 256$) 图像 (中), 以及分辨率为 $256\times 256$ 的真实图像 (右).

<span id="figure-10"></span>

![图 10. 掩蔽图像 (左), 一致性模型补全的图像 (中) 和真实图像 (右).](./consistency-models/figure-10.png)

**图 10.** 掩蔽图像 (左), 一致性模型补全的图像 (中) 和真实图像 (右).

<span id="figure-11"></span>

![图 11. 使用球面线性插值在最左和最右图像之间插值. 所有样本均由在 LSUN Bedroom $256\times 256$ 上训练的一致性模型生成.](./consistency-models/figure-11.png)

**图 11.** 使用球面线性插值在最左和最右图像之间插值. 所有样本均由在 LSUN Bedroom $256\times 256$ 上训练的一致性模型生成.

<span id="figure-12"></span>

![图 12. 使用一致性模型进行单步去噪. 最左侧图像为真实图像. 每两行中, 上行显示不同噪声水平的含噪图像, 下行给出去噪图像.](./consistency-models/figure-12.png)

**图 12.** 使用一致性模型进行单步去噪. 最左侧图像为真实图像. 每两行中, 上行显示不同噪声水平的含噪图像, 下行给出去噪图像.

<span id="figure-13"></span>

![图 13. 使用一致性模型执行 SDEdit. 最左侧图像为笔画输入, 右侧图像为笔画引导图像生成 (SDEdit) 的结果.](./consistency-models/figure-13.png)

**图 13.** 使用一致性模型执行 SDEdit. 最左侧图像为笔画输入, 右侧图像为笔画引导图像生成 (SDEdit) 的结果.

<span id="section-12"></span>

## 12 一致性模型的更多样本

我们给出一致性蒸馏 (CD) 和一致性训练 (CT) 在 CIFAR-10 ([图 14](#figure-14) 和[图 18](#figure-18)), ImageNet $64\times 64$ ([图 15](#figure-15) 和[图 19](#figure-19)), LSUN Bedroom $256\times 256$ ([图 16](#figure-16) 和[图 20](#figure-20)), 以及 LSUN Cat $256\times 256$ ([图 17](#figure-17) 和[图 21](#figure-21)) 上的更多样本.

<span id="figure-14"></span>

![图 14. CIFAR-10 $32\times 32$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-14.png)

**图 14.** CIFAR-10 $32\times 32$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-15"></span>

![图 15. ImageNet $64\times 64$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-15.png)

**图 15.** ImageNet $64\times 64$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-16"></span>

![图 16. LSUN Bedroom $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-16.png)

**图 16.** LSUN Bedroom $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-17"></span>

![图 17. LSUN Cat $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-17.png)

**图 17.** LSUN Cat $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-18"></span>

![图 18. CIFAR-10 $32\times 32$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-18.png)

**图 18.** CIFAR-10 $32\times 32$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-19"></span>

![图 19. ImageNet $64\times 64$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-19.png)

**图 19.** ImageNet $64\times 64$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-20"></span>

![图 20. LSUN Bedroom $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-20.png)

**图 20.** LSUN Bedroom $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.

<span id="figure-21"></span>

![图 21. LSUN Cat $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.](./consistency-models/figure-21.png)

**图 21.** LSUN Cat $256\times 256$ 上未经筛选的样本. 所有对应样本都使用相同的初始噪声.
