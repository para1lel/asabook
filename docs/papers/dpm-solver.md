---
title: 'DPM-Solver: Fast Diffusion Sampling'
createTime: 2026/09/12 00:00:00
permalink: /papers/dpm-solver/
pageClass: paper-reading
---

> [Cheng Lu](https://luchengthu.github.io/), [Yuhao Zhou](https://yuhaoz.com/), [Fan Bao](https://www.baofan.ai/), [Jianfei Chen](https://ml.cs.tsinghua.edu.cn/~jianfei/), [Chongxuan Li](https://ai.ruc.edu.cn/english/FACULTYn/ChongxuanLi/index.htm) 和 [Jun Zhu](https://ml.cs.tsinghua.edu.cn/~jun/index.shtml). 2022 年 6 月 2 日首次提交至 arXiv, 当前版本为 v3. 发表于 [NeurIPS 2022](https://proceedings.neurips.cc/paper_files/paper/2022/hash/260a14acce2a89dad36adc8eefe7c59e-Abstract-Conference.html), 第 5775–5787 页. [DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps](https://arxiv.org/abs/2206.00927). <a href="/paper/dpm-solver.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [DOI](https://doi.org/10.52202/068431-0418). [TeX 源文件](https://export.arxiv.org/e-print/2206.00927v3). 精确的印刷版式和参考文献以原始 PDF 为准.

## 摘要

扩散概率模型 (diffusion probabilistic model, DPM) 是一类新兴且能力很强的生成模型. 尽管生成质量很高, DPM 的采样速度仍然很慢: 生成一个样本通常要顺序执行数百乃至数千次大型神经网络的函数求值 (即采样步骤). 从另一个角度看, DPM 采样等价于求解对应的扩散常微分方程 (ordinary differential equation, ODE). 本文给出扩散 ODE 解的精确形式. 与以往工作把所有项都交给黑盒 ODE 求解器不同, 该形式解析地计算解的线性部分. 再作变量代换, 解可等价地化为神经网络的指数加权积分. 基于这一形式, 我们提出 DPM-Solver, 一种专用于扩散 ODE, 具有收敛阶保证的快速高阶求解器. DPM-Solver 无须额外训练, 同时适用于离散时间和连续时间 DPM. 实验表明, 在多个数据集上, DPM-Solver 只需 10 到 20 次函数求值便能生成高质量样本. 在 CIFAR-10 数据集上, 10 次和 20 次函数求值分别达到 4.70 和 2.87 的 FID; 与以往最先进的免训练采样器相比, 在多个数据集上取得 $4\sim 16\times$ 的加速. [+1]

<span id="section-1"></span>

## 1 引言

扩散概率模型 (DPM) [Soh15, Den20, Son21] 是一类新兴且表现优良的生成模型, 已用于图像生成 [Dha21, Men22], 视频生成 [Ho22], 文生图 [Ram22], 语音合成 [Che21n, Che21o] 和无损压缩 [Kin21] 等任务. DPM 由离散时间随机过程 [Soh15, Den20] 或连续时间随机微分方程 (stochastic differential equation, SDE) [Son21] 定义, 学习逐步去除添加到数据点上的噪声. 与广泛使用的生成对抗网络 (GAN) [Goo14] 和变分自编码器 (VAE) [Kin14] 相比, DPM 不仅能计算精确似然 [Son21], 在图像生成中还能得到更好的样本质量 [Dha21]. 但要得到高质量样本, DPM 往往要顺序执行数百或数千次大型神经网络求值, 因而采样远慢于单步的 GAN 或 VAE. 这种低效率正成为 DPM 用于下游任务的一项主要瓶颈, 因此亟需为 DPM 设计快速采样器.

现有 DPM 快速采样器可分为两类. 第一类包括知识蒸馏 [Sal22, Luh21] 以及噪声水平或样本轨迹学习 [San21a, Nic21, Lam21, Wat22]. 这些方法要先完成一段可能代价很高的训练, 才能高效采样, 其适用范围和灵活性也可能受限. 把它们迁移到不同模型, 数据集或采样步数时, 往往并不容易. 第二类是免训练采样器 [Son21a, Jol21, Bao22], 可以即插即用地作用于所有预训练 DPM. 此类方法包括采用隐式 [Son21a] 或解析 [Bao22] 生成过程, 先进的微分方程 (DE) 求解器 [Son21, Jol21, Liu22h, Pop22b, Tac21] 以及动态规划 [Wat22]. 不过, 这些方法仍需约 50 次函数求值 [Bao22] 才能生成高质量样本 (与普通采样器约 1000 次函数求值所得样本相当), 采样耗时依然不低.

本文把免训练采样器的效率推进到“少步采样”区间, 即在约 10 步顺序函数求值内生成高质量样本. 我们把 DPM 采样转化为求解对应的扩散 ODE, 并仔细考察这类方程的结构. 扩散 ODE 是半线性的: 它由数据变量的线性函数和神经网络参数化的非线性函数组成. 以往免训练采样器 [Son21, Jol21] 直接使用黑盒 DE 求解器, 没有利用这一结构. 为此, 我们解析计算解的线性部分, 推导出扩散 ODE 解的精确形式, 从而避开相应的离散化误差. 进一步作变量代换后, 解可等价地化为神经网络的指数加权积分. 这种积分形式很特殊, 可以用指数积分器的数值方法高效近似 [Hoc10].

基于上述解的形式, 我们通过近似该积分提出专用于扩散 ODE 的快速求解器 DPM-Solver. 具体而言, 本文给出一阶, 二阶和三阶 DPM-Solver, 并保证其收敛阶; 还为 DPM-Solver 提出自适应步长方案. 总体上, DPM-Solver 既适用于连续时间和离散时间 DPM, 也适用于带分类器引导的条件采样 [Dha21]. [图 1](#figure-01) 比较了去噪扩散隐式模型 (DDIM) [Son21a] 基线与 DPM-Solver 的加速效果. 在 ImageNet $256\times256$ 数据集 [Den09a] 上, DPM-Solver 只需 10 次函数求值便能生成高质量样本, 速度远快于 DDIM. 其他实验也表明, DPM-Solver 能显著加快离散时间和连续时间 DPM 的采样, 约 10 次函数求值即可得到优良的样本质量, 比此前所有免训练 DPM 采样器都快得多.

<span id="figure-01"></span>

![DDIM 与 DPM-Solver 在不同函数求值次数下生成的样本](./dpm-solver/figure-01.png)

**图 1.** 使用 ImageNet $256\times256$ 上预训练且带分类器引导的 DPM [Dha21], DDIM [Son21a] 在 10, 15, 20, 100 次函数求值 (NFE) 下生成的样本, 以及 DPM-Solver (本文方法) 仅用 10 NFE 生成的样本.

<span id="section-2"></span>

## 2 扩散概率模型

本节回顾扩散概率模型及其对应的微分方程.

<span id="section-2-1"></span>

### 2.1 前向过程与扩散 SDE

设有一个分布未知的 $D$ 维随机变量 $\bm{x}_{0}\in\mathbb{R}^{D}$, 其分布为 $q_{0}(\bm{x}_{0})$. 扩散概率模型 (DPM) [Soh15, Den20, Son21, Kin21] 从 $\bm{x}_{0}$ 出发定义一个 $T>0$ 的前向过程 $\{\bm{x}_{t}\}_{t\in[0,T]}$, 使任意 $t\in[0,T]$ 下, 给定 $\bm{x}_{0}$ 时 $\bm{x}_{t}$ 的条件分布满足

<span id="equation-2-1"></span>

$$
q_{0t}(\bm{x}_{t}|\bm{x}_{0})=\mathcal{N}(\bm{x}_{t}|\alpha(t)\bm{x}_{0},\sigma^{2}(t)\bm{I}),
$$

其中 $\alpha(t),\sigma(t)\in\mathbb{R}^{+}$ 是关于 $t$ 的可微函数, 导数有界; 为简洁起见, 记作 $\alpha_{t},\sigma_{t}$. $\alpha_{t}$ 和 $\sigma_{t}$ 的选择称为 DPM 的噪声调度. 令 $q_{t}(\bm{x}_{t})$ 表示 $\bm{x}_{t}$ 的边缘分布. DPM 选择噪声调度, 使某个 $\tilde{\sigma}>0$ 满足 $q_{T}(\bm{x}_{T})\approx\mathcal{N}(\bm{x}_{T}|\bm{0},\tilde{\sigma}^{2}\bm{I})$, 且信噪比 (SNR) $\alpha_{t}^{2}/\sigma_{t}^{2}$ 关于 $t$ 严格递减 [Kin21]. 另外, [Kin21] 证明, 对任意 $t\in[0,T]$, 下列随机微分方程 (SDE) 与[公式 2.1](#equation-2-1) 具有相同的转移分布 $q_{0t}(\bm{x}_{t}|\bm{x}_{0})$:

<span id="equation-2-2"></span>

$$
\mathrm{d}\bm{x}_{t}=f(t)\bm{x}_{t}\mathrm{d}t+g(t)\mathrm{d}\bm{w}_{t},\quad\bm{x}_{0}\sim q_{0}(\bm{x}_{0}),
$$

其中 $\bm{w}_{t}\in\mathbb{R}^{D}$ 为标准 Wiener 过程, 且

<span id="equation-2-3"></span>

$$
f(t)=\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t},\quad g^{2}(t)=\frac{\mathrm{d}\sigma_{t}^{2}}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\sigma_{t}^{2}.
$$

在一定正则条件下, [Son21] 证明, [公式 2.2](#equation-2-2) 的前向过程有一个从时刻 $T$ 到 $0$, 从边缘分布 $q_{T}(\bm{x}_{T})$ 出发的等价逆向过程:

<span id="equation-2-4"></span>

$$
\mathrm{d}\bm{x}_{t}=[f(t)\bm{x}_{t}-g^{2}(t)\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t})]\mathrm{d}t+g(t)\mathrm{d}\bar{\bm{w}}_{t},\quad\bm{x}_{T}\sim q_{T}(\bm{x}_{T}),
$$

其中 $\bar{\bm{w}}_{t}$ 是逆向时间中的标准 Wiener 过程. [公式 2.4](#equation-2-4) 唯一未知的项是各时刻 $t$ 的得分函数 $\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t})$. 实践中, DPM 使用参数为 $\theta$ 的神经网络 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 估计缩放后的得分函数 $-\sigma_{t}\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t})$. 参数 $\theta$ 通过最小化下列目标来优化 [Den20, Son21]:

$$
\begin{aligned}
\mathcal{L}(\theta;\omega(t)) & \coloneqq\frac{1}{2}\int_{0}^{T}\omega(t)\mathbb{E}_{q_{t}(\bm{x}_{t})}\Big[\|\bm{\epsilon}_{\theta}(\bm{x}_{t},t)+\sigma_{t}\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t})\|_{2}^{2}\Big]\mathrm{d}t \\
& =\frac{1}{2}\int_{0}^{T}\omega(t)\mathbb{E}_{q_{0}(\bm{x}_{0})}\mathbb{E}_{q(\bm{\epsilon})}\Big[\|\bm{\epsilon}_{\theta}(\bm{x}_{t},t)-\bm{\epsilon}\|_{2}^{2}\Big]\mathrm{d}t+C,
\end{aligned}
$$

其中 $\omega(t)$ 为权重函数, $\bm{\epsilon}\sim q(\bm{\epsilon})=\mathcal{N}(\bm{\epsilon}|\bm{0},\bm{I})$, $\bm{x}_{t}=\alpha_{t}\bm{x}_{0}+\sigma_{t}\bm{\epsilon}$, $C$ 是与 $\theta$ 无关的常数. $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 也可视为预测加入 $\bm{x}_{t}$ 的高斯噪声, 因而通常称为噪声预测模型. 由于 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 的真实值是 $-\sigma_{t}\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t})$, DPM 用 $-\bm{\epsilon}_{\theta}(\bm{x}_{t},t)/\sigma_{t}$ 代替[公式 2.4](#equation-2-4) 中的得分函数, 定义从时刻 $T$ 到 $0$, 从 $\bm{x}_{T}\sim\mathcal{N}(\bm{0},\tilde{\sigma}^{2}\bm{I})$ 出发的参数化逆向过程 (扩散 SDE):

<span id="equation-2-5"></span>

$$
\mathrm{d}\bm{x}_{t}=\left[f(t)\bm{x}_{t}+\frac{g^{2}(t)}{\sigma_{t}}\bm{\epsilon}_{\theta}(\bm{x}_{t},t)\right]\mathrm{d}t+g(t)\mathrm{d}\bar{\bm{w}}_{t},\quad\bm{x}_{T}\sim\mathcal{N}(\bm{0},\tilde{\sigma}^{2}\bm{I}).
$$

用数值求解器把 SDE 从 $T$ 离散化到 $0$, 求解[公式 2.5](#equation-2-5) 的扩散 SDE, 即可从 DPM 生成样本. [Son21] 证明, DPM 的传统祖先采样法 [Den20] 可视为[公式 2.5](#equation-2-5) 的一阶 SDE 求解器. 但这些一阶方法通常要经过数百乃至数千次函数求值才能收敛 [Son21], 因而采样极慢.

<span id="section-2-2"></span>

### 2.2 扩散 (概率流) ODE

离散化 SDE 时, 步长会受到 Wiener 过程随机性的限制 [Klo92]. 步长较大 (步数较少) 时往往不收敛, 在高维空间中尤其如此. 为加快采样, 可以考虑对应的概率流 ODE [Son21], 它在每个时刻 $t$ 都与 SDE 具有相同的边缘分布. 具体而言, 对 DPM, [Son21] 证明[公式 2.4](#equation-2-4) 的概率流 ODE 为

<span id="equation-2-6"></span>

$$
\frac{\mathrm{d}\bm{x}_{t}}{\mathrm{d}t}=f(t)\bm{x}_{t}-\frac{1}{2}g^{2}(t)\nabla_{\bm{x}}\log q_{t}(\bm{x}_{t}),\quad\bm{x}_{T}\sim q_{T}(\bm{x}_{T}),
$$

其中 $\bm{x}_{t}$ 的边缘分布仍为 $q_{t}(\bm{x}_{t})$. 用噪声预测模型代替得分函数后, [Son21] 定义了如下参数化 ODE (扩散 ODE):

<span id="equation-2-7"></span>

$$
\frac{\mathrm{d}\bm{x}_{t}}{\mathrm{d}t}=\bm{h}_{\theta}(\bm{x}_{t},t)\coloneqq f(t)\bm{x}_{t}+\frac{g^{2}(t)}{2\sigma_{t}}\bm{\epsilon}_{\theta}(\bm{x}_{t},t),\quad\bm{x}_{T}\sim\mathcal{N}(\bm{0},\tilde{\sigma}^{2}\bm{I}).
$$

从 $T$ 到 $0$ 求解该 ODE 即可抽取样本. 与 SDE 相比, ODE 没有随机性, 因而能采用更大的步长; 还可以借助高效的数值 ODE 求解器加速采样. [Son21] 用 RK45 ODE 求解器 [Dor80] 求解扩散 ODE. 在 CIFAR-10 数据集 [Kri09] 上, 约 60 次函数求值生成的样本便能达到[公式 2.5](#equation-2-5) 的 1000 步 SDE 求解器的质量. 但现有通用 ODE 求解器在少步 (约 10 步) 采样区间内仍无法生成令人满意的样本. 据我们所知, 少步采样区间尚无免训练 DPM 采样器, DPM 的采样速度仍是一项主要问题.

<span id="section-3"></span>

## 3 为扩散 ODE 定制快速求解器

如[第 2.2 节](#section-2-2)所述, 在高维空间中离散化 SDE 通常很困难 [Klo92], 很难在少量步骤内收敛. 相比之下, ODE 更易求解, 因而有望实现快速采样. 但[第 2.2 节](#section-2-2)也提到, 以往工作 [Son21] 所用的通用黑盒 ODE 求解器在实验中无法于少量步骤内收敛. 因此, 我们为扩散 ODE 设计专用求解器, 以实现快速, 高质量的少步采样. 首先详细考察扩散 ODE 的具体结构.

<span id="section-3-1"></span>

### 3.1 扩散 ODE 精确解的简化形式

本文的核心发现是: 给定时刻 $s>0$ 的初值 $\bm{x}_{s}$, [公式 2.7](#equation-2-7) 中扩散 ODE 在任意 $t<s$ 时刻的解 $\bm{x}_{t}$ 都可化为一种特殊的精确形式, 并能得到高效近似.

第一个发现来自扩散 ODE 的特殊结构: 解 $\bm{x}_{t}$ 有一部分可以精确计算. [公式 2.7](#equation-2-7) 中扩散 ODE 的右端由两部分组成. $f(t)\bm{x}_{t}$ 是 $\bm{x}_{t}$ 的线性函数; 另一项 $\frac{g^{2}(t)}{2\sigma_{t}}\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 由于神经网络 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 的存在, 通常是 $\bm{x}_{t}$ 的非线性函数. 这类 ODE 称为半线性 ODE. 以往工作 [Son21] 使用的黑盒 ODE 求解器把[公式 2.7](#equation-2-7) 中整个 $\bm{h}_{\theta}(\bm{x}_{t},t)$ 作为输入, 没有利用半线性结构, 因而线性项和非线性项都会产生离散化误差. 对半线性 ODE, “常数变易”公式 [Atk11] 能精确地写出时刻 $t$ 的解:

<span id="equation-3-1"></span>

$$
\bm{x}_{t}=\mathrm{e}^{\int_{s}^{t}f(\tau)\mathrm{d}\tau}\bm{x}_{s}+\int_{s}^{t}\left(\mathrm{e}^{\int_{\tau}^{t}f(r)\mathrm{d} r}\frac{g^{2}(\tau)}{2\sigma_{\tau}}\bm{\epsilon}_{\theta}(\bm{x}_{\tau},\tau)\right)\mathrm{d}\tau.
$$

这一形式将线性部分与非线性部分解耦. 不同于黑盒 ODE 求解器, 现在可以精确计算线性部分, 从而消除线性项的近似误差. 不过, 非线性部分的积分仍然复杂: 它把噪声调度的系数 (即 $f(\tau),g(\tau),\sigma_{\tau}$) 与复杂的神经网络 $\bm{\epsilon}_{\theta}$ 耦合在一起, 依然很难近似.

第二个发现是, 引入一个特殊变量可以大幅简化非线性部分的积分. 令 $\lambda_{t}\coloneqq\log(\alpha_t / \sigma_t)$ (即 log-SNR 的一半). 根据[第 2.1 节](#section-2-1) 中 DPM 的定义, $\lambda_{t}$ 关于 $t$ 严格递减. 我们可把[公式 2.3](#equation-2-3) 中的 $g(t)$ 改写为

<span id="equation-3-2"></span>

$$
g^{2}(t)=\frac{\mathrm{d}\sigma_{t}^{2}}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\sigma_{t}^{2}=2\sigma_{t}^{2}\left(\frac{\mathrm{d}\log\sigma_{t}}{\mathrm{d}t}-\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\right)=-2\sigma_{t}^{2}\frac{\mathrm{d}\lambda_{t}}{\mathrm{d}t}.
$$

再结合[公式 2.3](#equation-2-3) 中的 $f(t)=\mathrm{d}\log\alpha_{t}/\mathrm{d}t$, [公式 3.1](#equation-3-1) 可写为

<span id="equation-3-3"></span>

$$
\bm{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\alpha_{t}\int_{s}^{t}\left(\frac{\mathrm{d}\lambda_{\tau}}{\mathrm{d}\tau}\right)\frac{\sigma_{\tau}}{\alpha_{\tau}}\bm{\epsilon}_{\theta}(\bm{x}_{\tau},\tau)\mathrm{d}\tau.
$$

由于 $\lambda(t)=\lambda_{t}$ 关于 $t$ 严格递减, 它存在逆函数 $t_{\lambda}(\cdot)$, 满足 $t=t_{\lambda}(\lambda(t))$. 进一步把 $\bm{x}$ 与 $\bm{\epsilon}_{\theta}$ 的下标从 $t$ 换成 $\lambda$, 记 $\hat{\bm{x}}_{\lambda}\coloneqq\bm{x}_{t_{\lambda}(\lambda)}$, $\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\coloneqq\bm{\epsilon}_{\theta}(\bm{x}_{t_{\lambda}(\lambda)},t_{\lambda}(\lambda))$. 对 $\lambda$ 作变量代换, 改写[公式 3.3](#equation-3-3), 得到:

<span id="proposition-03-01"></span>

**命题 3.1 (扩散 ODE 的精确解).** 给定时刻 $s>0$ 的初值 $\bm{x}_{s}$, [公式 2.7](#equation-2-7) 中扩散 ODE 在时刻 $t\in[0,s]$ 的解为:

<span id="equation-3-4"></span>

$$
\bm{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\alpha_{t}\int_{\lambda_{s}}^{\lambda_{t}}\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\mathrm{d}\lambda.
$$

我们把积分 $\int \mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\mathrm{d}\lambda$ 称为 $\hat{\bm{\epsilon}}_{\theta}$ 的指数加权积分. 这种特殊形式与 ODE 求解器文献中的指数积分器密切相关 [Hoc10]. 据我们所知, 此前的扩散模型工作尚未揭示这一形式.

[公式 3.4](#equation-3-4) 为近似扩散 ODE 的解提供了新的视角. 具体而言, 给定时刻 $s$ 的 $\bm{x}_{s}$, 依照[公式 3.4](#equation-3-4), 近似时刻 $t$ 的解等价于直接近似 $\hat{\bm{\epsilon}}_{\theta}$ 从 $\lambda_{s}$ 到 $\lambda_{t}$ 的指数加权积分. 这样不会产生线性项误差, 而且指数积分器文献 [Hoc10, Hoc05] 已对该问题有充分研究. 下面据此构造扩散 ODE 的快速求解器.

<span id="section-3-2"></span>

### 3.2 扩散 ODE 的高阶求解器

本节利用[公式 3.4](#equation-3-4) 给出的解形式, 提出具有收敛阶保证的扩散 ODE 高阶求解器. 所用方法与分析主要受 ODE 文献中的指数积分器 [Hoc10, Hoc05] 启发.

具体而言, 给定时刻 $T$ 的初值 $\bm{x}_{T}$, 以及从 $t_{0}=T$ 递减到 $t_{M}=0$ 的 $M+1$ 个时间步 $\{t_{i}\}_{i=0}^{M}$. 令 $\tilde{\bm{x}}_{t_{0}}=\bm{x}_{T}$ 为初值. 本文求解器用 $M$ 步迭代计算序列 $\{\tilde{\bm{x}}_{t_{i}}\}_{i=0}^{M}$, 近似各时间步 $\{t_{i}\}_{i=0}^{M}$ 上的真实解; 最后一次迭代 $\tilde{\bm{x}}_{t_{M}}$ 近似时刻 $0$ 的真实解.

为减小 $\tilde{\bm{x}}_{t_{M}}$ 与时刻 $0$ 真实解之间的近似误差, 每一步都要减小 $\tilde{\bm{x}}_{t_{i}}$ 的近似误差 [Atk11]. 从时刻 $t_{i-1}$ 的前一步结果 $\tilde{\bm{x}}_{t_{i-1}}$ 出发, 根据[公式 3.4](#equation-3-4), 时刻 $t_{i}$ 的精确解 $\bm{x}_{t_{i-1}\to t_{i}}$ 为

<span id="equation-3-5"></span>

$$
\bm{x}_{t_{i-1}\to t_{i}}=\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\alpha_{t_{i}}\int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}}\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\mathrm{d}\lambda.
$$

因此, 要计算近似 $\bm{x}_{t_{i-1}\to t_{i}}$ 的 $\tilde{\bm{x}}_{t_{i}}$, 需要近似 $\hat{\bm{\epsilon}}_{\theta}$ 从 $\lambda_{t_{i-1}}$ 到 $\lambda_{t_{i}}$ 的指数加权积分. 记 $h_{i}\coloneqq\lambda_{t_{i}}-\lambda_{t_{i-1}}$, 并以 $\hat{\bm{\epsilon}}_{\theta}^{(n)}(\hat{\bm{x}}_{\lambda},\lambda)\coloneqq\frac{\mathrm{d}^{n}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)}{\mathrm{d}\lambda^{n}}$ 表示 $\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$ 关于 $\lambda$ 的 $n$ 阶全导数. 对 $k\geq 1$, $\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$ 在 $\lambda_{t_{i-1}}$ 处关于 $\lambda$ 的 $(k-1)$ 阶 Taylor 展开为

$$
\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)=\sum_{n=0}^{k-1}\frac{(\lambda-\lambda_{t_{i-1}})^{n}}{n!}\hat{\bm{\epsilon}}_{\theta}^{(n)}(\hat{\bm{x}}_{\lambda_{t_{i-1}}},\lambda_{t_{i-1}})+\mathcal{O}((\lambda-\lambda_{t_{i-1}})^{k}),
$$

把上述 Taylor 展开代入[公式 3.5](#equation-3-5), 得

<span id="equation-3-6"></span>

$$
\bm{x}_{t_{i-1}\to t_{i}}\!=\!\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\alpha_{t_{i}}\sum_{n=0}^{k-1}\hat{\bm{\epsilon}}_{\theta}^{(n)}(\hat{\bm{x}}_{\lambda_{t_{i-1}}},\lambda_{t_{i-1}})\!\int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}}\!\!\mathrm{e}^{-\lambda}\frac{(\lambda-\lambda_{t_{i-1}})^{n}}{n!}\mathrm{d}\lambda+\mathcal{O}(h_{i}^{k+1}),
$$

其中积分 $\int \mathrm{e}^{-\lambda}\frac{(\lambda-\lambda_{t_{i-1}})^{n}}{n!}\mathrm{d}\lambda$ 可以连续应用 $n$ 次分部积分解析计算 (见[第 8.2 节](#section-8-2)). 因此, 为近似 $\bm{x}_{t_{i-1}\to t_{i}}$, 只需近似 $n\leq k-1$ 时的 $n$ 阶全导数 $\hat{\bm{\epsilon}}_{\theta}^{(n)}(\hat{\bm{x}}_{\lambda},\lambda)$. 这在 ODE 文献中已有充分研究 [Hoc05, Lua21]. 略去 $\mathcal{O}(h_{i}^{k+1})$ 误差项, 再用“刚性阶条件” [Hoc05, Lua21] 近似前 $(k-1)$ 阶全导数, 就能得到扩散 ODE 的 $k$ 阶求解器. 我们把这类求解器统称为 DPM-Solver, 特定阶数 $k$ 的版本记作 DPM-Solver-$k$. 以 $k=1$ 为例, [公式 3.6](#equation-3-6) 变为

$$
\begin{aligned}
\bm{x}_{t_{i-1}\to t_{i}} & =\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\alpha_{t_{i}}\bm{\epsilon}_{\theta}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})\int_{\lambda_{t_{i-1}}}^{\lambda_{t_{i}}}\mathrm{e}^{-\lambda}\mathrm{d}\lambda+\mathcal{O}(h_{i}^{2}) \\
& =\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_{i}}(\mathrm{e}^{h_{i}}-1)\bm{\epsilon}_{\theta}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})+\mathcal{O}(h_{i}^{2}).
\end{aligned}
$$

略去高阶误差项 $\mathcal{O}(h_{i}^{2})$, 即得到 $\bm{x}_{t_{i-1}\to t_{i}}$ 的近似. 此处 $k=1$, 因而把该求解器称为 DPM-Solver-1. 详细算法如下.

**DPM-Solver-1.** 给定初值 $\bm{x}_{T}$ 和从 $t_{0}=T$ 递减到 $t_{M}=0$ 的 $M+1$ 个时间步 $\{t_{i}\}_{i=0}^{M}$. 从 $\tilde{\bm{x}}_{t_{0}}=\bm{x}_{T}$ 出发, 按下式迭代计算序列 $\{\tilde{\bm{x}}_{t_{i}}\}_{i=1}^{M}$:

<span id="equation-3-7"></span>

$$
\tilde{\bm{x}}_{t_{i}}=\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_{i}}(\mathrm{e}^{h_{i}}-1)\bm{\epsilon}_{\theta}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}),\ \ \ \ \text{where }h_{i}=\lambda_{t_{i}}-\lambda_{t_{i-1}}.
$$

$k\geq 2$ 时, 近似 Taylor 展开的前 $k$ 项还需要位于 $t$ 和 $s$ 之间的中间点 [Hoc05]. 推导较为技术化, 放在[第 8 节](#section-8). 下面给出 $k=2,3$ 的算法, 分别称为 DPM-Solver-2 和 DPM-Solver-3.

<span id="algorithm-01"></span>

**算法 1: DPM-Solver-2.**

- **输入:** 初值 $\bm{x}_T$, 时间步 $\{t_i\}_{i=0}^M$, 模型 $\bm{\epsilon}_\theta$.
- 令 $\tilde{\bm{x}}_{t_0}\leftarrow\bm{x}_T$.
- **对** $i\leftarrow1$ 到 $M$:
  - 令 $s_i\leftarrow t_\lambda\!\left(\frac{\lambda_{t_{i-1}}+\lambda_{t_i}}{2}\right)$.
  - 令 $\bm{u}_i\leftarrow\frac{\alpha_{s_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_i}\left(\mathrm{e}^{\frac{h_i}{2}}-1\right)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_i}(\mathrm{e}^{h_i}-1)\bm{\epsilon}_\theta(\bm{u}_i,s_i)$.
- **返回:** $\tilde{\bm{x}}_{t_M}$.

<span id="algorithm-02"></span>

**算法 2: DPM-Solver-3.**

- **输入:** 初值 $\bm{x}_T$, 时间步 $\{t_i\}_{i=0}^M$, 模型 $\bm{\epsilon}_\theta$.
- 令 $\tilde{\bm{x}}_{t_0}\leftarrow\bm{x}_T$, $r_1\leftarrow\frac{1}{3}$, $r_2\leftarrow\frac{2}{3}$.
- **对** $i\leftarrow1$ 到 $M$:
  - 令 $s_{2i-1}\leftarrow t_\lambda(\lambda_{t_{i-1}}+r_1h_i)$, $s_{2i}\leftarrow t_\lambda(\lambda_{t_{i-1}}+r_2h_i)$.
  - 令 $\bm{u}_{2i-1}\leftarrow\frac{\alpha_{s_{2i-1}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_{2i-1}}(\mathrm{e}^{r_1h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\bm{D}_{2i-1}\leftarrow\bm{\epsilon}_\theta(\bm{u}_{2i-1},s_{2i-1})-\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\bm{u}_{2i}\leftarrow\frac{\alpha_{s_{2i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_{2i}}(\mathrm{e}^{r_2h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})-\frac{\sigma_{s_{2i}}r_2}{r_1}\left(\frac{\mathrm{e}^{r_2h_i}-1}{r_2h_i}-1\right)\bm{D}_{2i-1}$.
  - 令 $\bm{D}_{2i}\leftarrow\bm{\epsilon}_\theta(\bm{u}_{2i},s_{2i})-\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_i}(\mathrm{e}^{h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})-\frac{\sigma_{t_i}}{r_2}\left(\frac{\mathrm{e}^{h_i}-1}{h}-1\right)\bm{D}_{2i}$.
- **返回:** $\tilde{\bm{x}}_{t_M}$.

这里, $t_{\lambda}(\cdot)$ 是 $\lambda(t)$ 的逆函数. 对 [Den20, Nic21] 实际采用的噪声调度, 它有解析形式, 见[第 10 节](#section-10). DPM-Solver-2 选取的中间点为 $(s_{i},\bm{u}_{i})$, DPM-Solver-3 则选取 $(s_{2i-1},\bm{u}_{2i-1})$ 和 $(s_{2i},\bm{u}_{2i})$. 由算法可见, $k=1,2,3$ 时, DPM-Solver-$k$ 每一步分别需要 $k$ 次函数求值. 高阶求解器 ($k=2,3$) 的单步代价更高, 但收敛阶也更高, 通常只需少得多的步骤便能收敛, 因而总体效率更好. 下面的定理说明 DPM-Solver-$k$ 是 $k$ 阶求解器, 证明见[第 8 节](#section-8).

<span id="theorem-03-02"></span>

**定理 3.2 (DPM-Solver-$k$ 是 $k$ 阶求解器).** 假设 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 满足[第 8.1 节](#section-8-1) 详述的正则条件, 则对 $k=1,2,3$, DPM-Solver-$k$ 是扩散 ODE 的 $k$ 阶求解器. 也就是说, 对 DPM-Solver-$k$ 计算得到的序列 $\{\tilde{\bm{x}}_{t_i}\}_{i=1}^M$, 时刻 $0$ 的近似误差满足 $\tilde{\bm{x}}_{t_M}-\bm{x}_0=\mathcal{O}(h_{\max}^k)$, 其中 $h_{\max}=\max_{1\leq i\leq M}(\lambda_{t_i}-\lambda_{t_{i-1}})$.

以往关于指数积分器的工作 [Hoc05, Lua21] 表明, $k\geq 4$ 的求解器需要更多中间点. 因此本文只考虑 $1$ 到 $3$ 阶, 更高阶的求解器留待以后研究.

<span id="section-3-3"></span>

### 3.3 步长方案

[第 3.2 节](#section-3-2) 提出的求解器需要预先指定时间步 $\{t_{i}\}_{i=0}^{M}$. 本文给出两种时间步方案. 第一种为人工设定, 即均匀划分区间 $[\lambda_{T}$, $\lambda_{0}$\]: $\lambda_{t_{i}}=\lambda_{T}+\frac{i}{M}(\lambda_{0}-\lambda_{T})$, $i=0,\dots,M$. 这不同于以往工作 [Den20, Son21] 对 $t_{i}$ 作均匀划分. 实验中, 采用均匀 $\lambda_{t_{i}}$ 时间步的 DPM-Solver 已能在少量步骤内生成质量不错的样本, 结果列于[第 11 节](#section-11). 第二种为自适应步长算法, 它组合不同阶数的 DPM-Solver, 动态调整步长. 该算法受 [Jol21] 启发, 实现细节见[第 9 节](#section-9).

少步采样需要用完给定的函数求值次数 (NFE). 当 NFE 不能被 $3$ 整除时, 我们先尽可能多地使用 DPM-Solver-3, 再根据 $K$ 除以 $3$ 的余数补一步 DPM-Solver-1 或 DPM-Solver-2, 详见[第 10 节](#section-10). 后续实验中, NFE $\leq 20$ 时采用这种求解器组合和均匀步长方案; 否则采用自适应步长方案.

<span id="section-3-4"></span>

### 3.4 从离散时间 DPM 采样

离散时间 DPM [Den20] 在 $N$ 个固定时间步 $\{t_{n}\}_{n=1}^{N}$ 上训练噪声预测模型. 该模型记为 $\tilde{\bm{\epsilon}}_{\theta}(\bm{x}_{n},n)$, 其中 $n=0,\dots,N-1$, 每个 $\bm{x}_{n}$ 对应时刻 $t_{n+1}$ 的取值. 对所有 $\bm{x}\in\mathbb{R}^{d},t\in[0,T]$, 令 $\bm{\epsilon}_{\theta}(\bm{x},t)\coloneqq\tilde{\bm{\epsilon}}_{\theta}(\bm{x},\frac{(N-1)t}{T})$, 就能把离散时间噪声预测模型转换为连续版本. $\tilde{\bm{\epsilon}}_{\theta}$ 的时间输入可能不是整数, 但实验表明噪声预测模型仍能正常工作; 我们推测这是因为时间嵌入较为平滑, 例如位置嵌入 [Den20]. 经过这种重参数化, 噪声预测模型可以接收连续时间步, 因而也能用 DPM-Solver 快速采样.

<span id="section-4"></span>

## 4 与现有快速采样方法的比较

本节讨论 DPM-Solver 与现有基于 ODE 的 DPM 快速采样方法之间的联系和区别, 并简要说明免训练采样器相较训练式采样器的优势.

<span id="section-4-1"></span>

### 4.1 DDIM 即 DPM-Solver-1

去噪扩散隐式模型 (DDIM) [Son21a] 为 DPM 设计了一种确定性快速采样方法. 对两个相邻时间步 $t_{i-1}$ 和 $t_{i}$, 设时刻 $t_{i-1}$ 已有解 $\tilde{\bm{x}}_{t_{i-1}}$, 则 DDIM 从 $t_{i-1}$ 到 $t_{i}$ 的一步为

<span id="equation-4-1"></span>

$$
\tilde{\bm{x}}_{t_{i}}=\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\alpha_{t_{i}}\left(\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}}-\frac{\sigma_{t_{i}}}{\alpha_{t_{i}}}\right)\bm{\epsilon}_{\theta}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}).
$$

尽管出发点完全不同, 我们证明 DPM-Solver-1 与 DDIM [Son21a] 的更新式相同. 由 $\lambda$ 的定义, $\frac{\sigma_{t_{i-1}}}{\alpha_{t_{i-1}}}=\mathrm{e}^{-\lambda_{t_{i-1}}}$, $\frac{\sigma_{t_{i}}}{\alpha_{t_{i}}}=\mathrm{e}^{-\lambda_{t_{i}}}$. 把这两式和 $h_{i}=\lambda_{t_{i}}-\lambda_{t_{i-1}}$ 代入[公式 4.1](#equation-4-1), 恰好得到[公式 3.7](#equation-3-7) 中 DPM-Solver-1 的一步更新. 不过, DPM-Solver 的半线性 ODE 形式还能系统地推广到高阶求解器, 并作收敛阶分析.

近期工作 [Sal22] 也通过对[公式 4.1](#equation-4-1) 两边求导, 说明 DDIM 是扩散 ODE 的一阶离散化, 但无法解释 DDIM 与扩散 ODE 的一阶 Euler 离散化有何区别. 本文指出 DDIM 是 DPM-Solver 的一个特例, 从而表明 DDIM 充分利用了扩散 ODE 的半线性结构, 这也解释了它为何优于传统 Euler 方法.

<span id="section-4-2"></span>

### 4.2 与传统 Runge-Kutta 方法的比较

直接把传统显式 Runge-Kutta (RK) 方法应用于[公式 2.7](#equation-2-7) 的扩散 ODE, 也能得到高阶求解器. 具体而言, RK 方法把[公式 2.7](#equation-2-7) 的解写成如下积分形式:

<span id="equation-4-2"></span>

$$
\bm{x}_{t}=\bm{x}_{s}+\int_{s}^{t}\bm{h}_{\theta}(\bm{x}_{\tau},\tau)\mathrm{d}\tau=\bm{x}_{s}+\int_{s}^{t}\left(f(\tau)\bm{x}_{\tau}+\frac{g^{2}(\tau)}{2\sigma_{\tau}}\bm{\epsilon}_{\theta}(\bm{x}_{\tau},\tau)\right)\mathrm{d}\tau,
$$

再选取 $[t,s]$ 之间的一些中间时间步, 组合这些时间步上 $\bm{h}_{\theta}$ 的求值来近似整个积分. 显式 RK 方法的近似误差取决于 $\bm{h}_{\theta}$, 包括线性项 $f(\tau)\bm{x}_{\tau}$ 和非线性噪声预测模型 $\bm{\epsilon}_{\theta}$ 对应的误差. 然而, 线性项的精确解含有指数系数 (见[公式 3.1](#equation-3-1)), 因而其误差可能指数增长. 大量实验结果 [Hoc10, Hoc05] 表明, 对半线性 ODE 直接采用显式 RK 方法, 在步长较大时可能出现数值不稳定. [第 5.1 节](#section-5-1) 也在实验中比较了 DPM-Solver 与传统显式 RK 方法; 相同阶数下, DPM-Solver 的离散化误差更小.

<span id="section-4-3"></span>

### 4.3 DPM 的训练式快速采样方法

需要额外训练或优化的采样器包括知识蒸馏 [Sal22, Luh21], 学习噪声水平或方差 [San21a, Nic21, Bao22a], 以及学习噪声调度或样本轨迹 [Lam21, Wat22]. 渐进蒸馏方法 [Sal22] 虽然能得到 4 步以内的快速采样器, 却要付出额外的训练成本, 并丢失原 DPM 的部分信息. 例如蒸馏后, 噪声预测模型无法预测 $[0,T]$ 之间每个时间步的噪声 (得分函数). 免训练采样器则保留原模型的全部信息, 因而可以直接把原模型与外部分类型组合, 扩展到条件采样 [Dha21] (例如[第 10 节](#section-10) 的分类器引导条件采样).

除直接设计 DPM 快速采样器外, 也有工作提出支持更快采样的新型 DPM, 如为 DPM 定义低维潜变量 [Vah21], 设计得分函数有界的特殊扩散过程 [Doc22], 或把 GAN 与 DPM 的逆向过程结合 [Xia22]. DPM-Solver 也可能加快这些 DPM 的采样, 留待以后研究.

<span id="table-01"></span>

![CIFAR-10 上 Runge-Kutta 方法与 DPM-Solver 的 FID](./dpm-solver/table-01.png)

**表 1.** 在 CIFAR-10 上改变函数求值次数 (NFE), 比较不同阶 Runge-Kutta (RK) 方法与 DPM-Solver 的 FID $\downarrow$. 对 RK 方法, 分别关于 $t$ ([公式 2.7](#equation-2-7)) 和 $\lambda$ ([公式 E.1](#equation-e-1)) 对扩散 ODE 求值. RK ($t$) 对 $t$ 采用均匀步长, RK ($\lambda$) 和 DPM-Solver 对 $\lambda$ 采用均匀步长.

<span id="figure-02"></span>

![连续时间和离散时间扩散模型在六个数据集上的 FID 曲线](./dpm-solver/figure-02.png)

**图 2.** 改变函数求值次数 (NFE), 比较不同 DPM 采样方法的样本质量, 指标为 FID $\downarrow$. 数据集包括同时采用连续时间和离散时间模型的 CIFAR-10, 以及采用离散时间模型的 CelebA $64\times64$, ImageNet $64\times64$, ImageNet $128\times128$ 和 LSUN bedroom $256\times256$. $^\dagger$GGDM [Wat22] 需要额外训练来优化样本轨迹, 其他方法均免训练. 为得到尽可能强的基线, CelebA 上的 DDIM 使用二次步长, 其 FID 优于原论文 [Son21a] 的均匀步长.

<span id="section-5"></span>

## 5 实验

本节说明, DPM-Solver 作为免训练采样器, 能大幅加速已有的预训练 DPM, 包括采用线性噪声调度 [Den20, Son21a] 或余弦噪声调度 [Nic21] 的连续时间和离散时间模型. 我们改变函数求值次数 (NFE), 即噪声预测模型 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 的调用次数, 比较 DPM-Solver 与其他方法的样本质量. 每项实验抽取 50K 个样本, 用广泛采用的 FID 分数 [Heu17] 评估样本质量; FID 越低, 通常表示样本质量越好.

除非另有说明, NFE 预算小于 20 时一律采用[第 3.3 节](#section-3-3) 中的求解器组合与均匀步长方案; 否则采用[第 3.3 节](#section-3-3) 的 DPM-Solver-3 自适应步长方案. DPM-Solver 的其他实现细节见[第 10 节](#section-10), 详细实验设置见[第 11 节](#section-11).

<span id="section-5-1"></span>

### 5.1 与连续时间采样方法的比较

首先比较 DPM-Solver 与其他连续时间 DPM 采样方法, 包括扩散 SDE 的 Euler-Maruyama 离散化 [Son21], 扩散 SDE 的自适应步长求解器 [Jol21], 以及[公式 2.7](#equation-2-7) 中扩散 ODE 的 RK 方法 [Son21, Dor80]. 实验在 CIFAR-10 数据集 [Kri09] 上, 从采用线性噪声调度的预训练连续时间“VP deep”模型 [Son21] 采样.

[图 2(a)](#figure-02) 展示了各求解器的效率. 采用 Euler 离散化的扩散 SDE 使用均匀时间步, NFE 分别为 50, 200, 1000; 对自适应步长 SDE 求解器 [Jol21] 和 RK45 ODE 求解器 [Dor80], 则改变容差超参数 [Son21, Jol21] 来控制 NFE. DPM-Solver 约 10 NFE 就能得到质量良好的样本, 其他求解器即使达到 50 NFE, 离散化误差仍然很大. 这说明 DPM-Solver 相比此前最好的求解器可加速约 $5$ 倍. 具体而言, 10, 12, 15, 20 NFE 分别达到 4.70, 3.75, 3.24 和 2.87 的 FID, 是 CIFAR-10 上最快的采样器.

作为消融实验, 我们还比较了二阶, 三阶 DPM-Solver 与 RK 方法, 结果见[表 1](#table-01). 对扩散 ODE, RK 方法分别关于[公式 2.7](#equation-2-7) 中的时间 $t$, 以及经变量代换得到的半 log-SNR $\lambda$ 求解 (具体形式见[第 11.1 节](#section-11-1)). 结果表明, NFE 相同时, DPM-Solver 的样本质量始终优于同阶 RK 方法. 在 15 NFE 以下的少步区间, RK 方法的离散化误差很大, DPM-Solver 的效率优势尤其明显. 主要原因是 DPM-Solver 解析计算线性项, 避开了相应的离散化误差. 三阶 DPM-Solver-3 的收敛速度也快于 DPM-Solver-2, 与[定理 3.2](#theorem-03-02) 的阶数分析一致.

<span id="section-5-2"></span>

### 5.2 与离散时间采样方法的比较

我们采用[第 3.4 节](#section-3-4) 的方法, 把 DPM-Solver 用于离散时间 DPM, 再与其他离散时间免训练采样器比较, 包括 DDPM [Den20], DDIM [Son21a], Analytic-DDPM [Bao22], Analytic-DDIM [Bao22], PNDM [Liu22h], FastDPM [Kon21a] 和 Itô-Taylor [Tac21]. 另与 GGDM [Wat22] 比较; 它使用相同的预训练模型, 但需要进一步训练采样轨迹. 比较时把 NFE 从 10 改变到 1000.

具体使用以下模型: CIFAR-10 上以 $L_{\text{simple}}$ 训练, 采用线性噪声调度的离散时间模型 [Den20]; CelebA $64\times64$ [Liu15a] 上采用线性噪声调度的离散时间模型 [Son21a]; ImageNet $64\times64$ [Den09a] 上以 $L_{\text{hybrid}}$ 训练, 采用余弦噪声调度的离散时间模型 [Nic21]; ImageNet $128\times128$ [Den09a] 上采用线性噪声调度和分类器引导的离散时间模型 [Dha21]; 以及 LSUN bedroom $256\times256$ [Yu15a] 上采用线性噪声调度的离散时间模型 [Dha21]. 对 ImageNet 上训练的模型, 只用其“均值”模型, 略去“方差”模型. 如[图 2](#figure-02) 所示, DPM-Solver 在所有数据集上都能于 12 步内得到合理的样本: CIFAR-10, CelebA $64\times64$, ImageNet $64\times64$ 和 ImageNet $128\times128$ 上的 FID 分别为 4.65, 3.71, 19.97 和 4.08. 它比此前最快的免训练采样器快 $4\sim 16\times$, 甚至优于需要额外训练的 GGDM.

<span id="section-6"></span>

## 6 结论

本文研究 DPM 的快速免训练采样, 提出专用于扩散 ODE 的快速免训练求解器 DPM-Solver, 约 10 步函数求值即可完成 DPM 的快速采样. DPM-Solver 利用扩散 ODE 的半线性结构, 直接近似其精确解的简化形式; 该形式由噪声预测模型的指数加权积分构成. 受指数积分器数值方法启发, 我们给出一阶, 二阶和三阶 DPM-Solver, 在理论上保证对噪声预测模型指数加权积分的近似收敛. 本文还提出人工设定和自适应两种步长方案, 并把 DPM-Solver 用于连续时间和离散时间 DPM. 实验表明, 在多个数据集上, DPM-Solver 约 10 次函数求值即可生成高质量样本; 与以往最先进的免训练采样器相比, 可加速 $4\sim 16\times$.

**局限性与更广泛的影响.** 尽管加速效果良好, DPM-Solver 的目标是快速采样, 可能不适合加快 DPM 的似然计算. 与常用 GAN 相比, 使用 DPM-Solver 的扩散模型仍不足以满足实时应用. 与其他深度生成模型一样, DPM 可能被用于生成有害的虚假内容; 本文求解器还可能放大深度生成模型用于恶意用途时的潜在负面影响.

## 致谢

本工作获得中国国家重点研发计划 (No. 2021ZD0110502), 中国国家自然科学基金项目 (Nos. 62061136001, 61620106010, 62076145, U19B2034, U1811461, U19A2081, 6197222, 62106120), 北京市自然科学基金项目 (No. JQ19016), 北京杰出青年科学家计划 (No. BJJWZYJH012019100020098), 清华大学国强研究院资助, 配备 GPU/DGX 加速的 NVIDIA NVAIL 计划, 清华大学高性能计算中心, 中央高校基本科研业务费专项资金, 以及中国人民大学科研基金 (22XNKJ13) 的支持. J.Z 还获得科学探索奖支持.

<span id="section-7"></span>

## 7 对噪声调度不变的采样

<span id="table-02"></span>

![对噪声调度选择不变的训练与采样形式](./dpm-solver/table-02.png)

**表 2.** 对噪声调度选择不变的形式. 关于 $\lambda$ 的最大似然训练损失等价于 [Kin21, Son21b] 中的目标, 扩散 ODE 的精确解则由[命题 3.1](#proposition-03-01) 给出.

本节进一步讨论[命题 3.1](#proposition-03-01) 的精确解, 并解释这一形式. 下面先关于 $\lambda$ (即半 log-SNR) 重述命题.

<span id="proposition-03-01-restated"></span>

**命题 3.1 (扩散 ODE 的精确解).** 给定时刻 $s$ 的初值 $\hat{\bm{x}}_{\lambda_s}$, 对应半 log-SNR 为 $\lambda_s$, 则[公式 2.7](#equation-2-7) 中扩散 ODE 在时刻 $t$, 对应半 log-SNR 为 $\lambda_t$ 时的解 $\hat{\bm{x}}_{\lambda_t}$ 为:

$$
\hat{\bm{x}}_{\lambda_t}=\frac{\alpha_t}{\alpha_s}\hat{\bm{x}}_{\lambda_s}-\alpha_t\int_{\lambda_s}^{\lambda_t}\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_\lambda,\lambda)\mathrm{d}\lambda.
$$

下面几节将说明, 这种形式把模型 $\bm{\epsilon}_{\theta}$ 与具体噪声调度解耦, 因而对噪声调度不变. 此外, [命题 3.1](#proposition-03-01) 中对 $\lambda$ 的变量代换与扩散模型的最大似然训练 [Kin21, Son21b] 密切相关. 我们将说明, 扩散模型的最大似然训练和采样都有独立于噪声调度的不变形式.

<span id="section-7-1"></span>

### 7.1 将采样解与噪声调度解耦

本节说明[命题 3.1](#proposition-03-01) 可以把扩散 ODE 的精确解与具体噪声调度 (即函数 $\alpha_{t}=\alpha(t)$ 和 $\sigma_{t}=\sigma(t)$ 的选择) 解耦. 也就是说, 给定起点 $\lambda_{s}$, 终点 $\lambda_{t}$, $\lambda_{s}$ 处的初值 $\hat{\bm{x}}_{\lambda_{s}}$ 和噪声预测模型 $\hat{\bm{\epsilon}}_{\theta}$, 解 $\hat{\bm{x}}_{\lambda_{t}}$ 不随 $\lambda_{s}$ 与 $\lambda_{t}$ 之间的噪声调度而变.

首先考虑与原始 DDPM [Den20, Son21] 等价的 VP 型扩散模型. VP 型扩散模型始终满足 $\alpha_{t}^{2}+\sigma_{t}^{2}=1$, 因而定义噪声调度等价于定义函数 $\alpha_{t}=\alpha(t)$. 例如, DDPM [Den20] 采用的噪声调度使 $\beta(t)=\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d} t}$ 成为 $t$ 的线性函数, i-DDPM [Nic21] 则使其成为 $t$ 的余弦函数. 由 $\lambda_{t}=\log\alpha_{t}-\log\sigma_{t}$, 可得 $\alpha_{t}=\sqrt{\frac{1}{1+\mathrm{e}^{-2\lambda_{t}}}}$, $\sigma_{t}=\sqrt{\frac{1}{1+\mathrm{e}^{2\lambda_{t}}}}$. 因此, 给定 $\lambda_{t}$ 即可直接计算 $\alpha_{t}$ 和 $\sigma_{t}$. 记 $\hat{\alpha}_{\lambda}\coloneqq\sqrt{\frac{1}{1+\mathrm{e}^{-2\lambda}}}$, 则

<span id="equation-a-2"></span>

$$
\hat{\bm{x}}_{\lambda_{t}}=\frac{\hat{\alpha}_{\lambda_{t}}}{\hat{\alpha}_{\lambda_{s}}}\hat{\bm{x}}_{\lambda_{s}}-\hat{\alpha}_{\lambda_{t}}\int_{\lambda_{s}}^{\lambda_{t}}\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\mathrm{d}\lambda.
$$

注意, 被积函数 $\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$ 是 $\lambda$ 的函数, 因而它从 $\lambda_{s}$ 到 $\lambda_{t}$ 的积分只取决于起点 $\lambda_{s}$, 终点 $\lambda_{t}$ 和函数 $\hat{\bm{\epsilon}}_{\theta}$, 与中间值无关. 其他系数 $\hat{\alpha}_{\lambda_{s}}$ 和 $\hat{\alpha}_{\lambda_{t}}$ 也只取决于起点和终点, 所以 $\hat{\bm{x}}_{\lambda_{t}}$ 不随具体噪声调度而变. 直观地说, 这是因为我们把[公式 3.1](#equation-3-1) 中关于时间 $t$ 的积分改成关于 $\lambda$ 的积分, 函数 $f(t)$ 和 $g(t)$ 随之化为解析形式 $\mathrm{e}^{-\lambda}$, 不再依赖 $f(t)$ 与 $g(t)$ 的具体选择. 对其他扩散模型类型, 如 VE 型和 subVP 型, [Kin21] 已证明, 适当缩放噪声预测模型即可使它们与 VP 型等价, 因而这些类型的解也有同样的性质.

综上, [命题 3.1](#proposition-03-01) 将扩散 ODE 的解与噪声调度解耦, 使我们得以为 DPM 专门设计采样器. 事实上, 如[第 3.2 节](#section-3-2)所示, DPM-Solver 唯一近似的是神经网络 $\hat{\bm{\epsilon}}_{\theta}$ 关于 $\lambda$ 的 Taylor 展开, 其他系数 (对应具体的噪声调度) 均作解析计算. 换言之, DPM-Solver 尽量保留已知信息, 只近似神经网络难以处理的积分, 因而能用少得多的步骤生成相当的样本.

<span id="section-7-2"></span>

### 7.2 为 $\lambda$ 选取时间步时对噪声调度不变

如[第 7.1 节](#section-7-1)所述, [命题 3.1](#proposition-03-01) 的形式将采样解与噪声调度解耦. 解取决于起点 $\lambda_{s}$ 和终点 $\lambda_{t}$, 不随中间噪声调度而变. 同样, DPM-Solver 算法的更新式也不随中间噪声调度而变. 因此, 一旦选定时间步 $\{\lambda_{i}\}_{i=0}^{M}$, DPM-Solver 的解便随之确定, 与中间噪声调度无关.

一种简单的 $\lambda$ 时间步选择方法是均匀划分 $[\lambda_{T},\lambda_{\epsilon}]$, 实验采用的正是这一设置. 不过, 我们认为还存在更精确的时间步选择方法, 留待以后研究.

<span id="section-7-3"></span>

### 7.3 与扩散模型最大似然训练的关系

有意思的是, 连续时间扩散 SDE 的最大似然训练也有这种不变性 [Kin21]. 下面简要回顾扩散 SDE 的最大似然训练损失, 再提出一个理解扩散模型的新视角.

记数据分布为 $q_{0}(\bm{x}_{0})$, 前向过程在各时刻 $t$ 的分布为 $q_{t}(\bm{x}_{t})$, 逆向过程在各时刻 $t$ 的分布为 $p_{t}(\bm{x}_{t})$, 且 $p_{T}=\mathcal{N}(\bm{0},\bm{I})$. [Son21] 证明, $q_{0}$ 与 $p_{0}$ 之间的 KL 散度可以由加权得分匹配损失界定:

<span id="equation-a-3"></span>

$$
D_{\mathrm{KL}}(q_{0}\;\|\;p_{0})\leq D_{\mathrm{KL}}(q_{T}\;\|\;p_{T})+\frac{1}{2}\int_{0}^{T}\frac{g^{2}(t)}{\sigma_{t}^{2}}\mathbb{E}_{q_{0}(\bm{x}_{0})}\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})}\Big[\|\bm{\epsilon}_{\theta}(\bm{x}_{t},t)-\bm{\epsilon}\|_{2}^{2}\Big]\mathrm{d}t+C,
$$

其中 $\bm{x}_{t}=\alpha_{t}\bm{x}_{0}+\sigma_{t}\bm{\epsilon}$, $C$ 是与 $\theta$ 无关的常数. 如[第 3.1 节](#section-3-1)所示,

<span id="equation-a-4"></span>

$$
g^{2}(t)=\frac{\mathrm{d}\sigma_{t}^{2}}{\mathrm{d}t}-2\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\sigma_{t}^{2}=2\sigma_{t}^{2}\left(\frac{\mathrm{d}\log\sigma_{t}}{\mathrm{d}t}-\frac{\mathrm{d}\log\alpha_{t}}{\mathrm{d}t}\right)=-2\sigma_{t}^{2}\frac{\mathrm{d}\lambda_{t}}{\mathrm{d}t},
$$

因此对 $\lambda$ 作变量代换, 得

<span id="equation-a-5"></span>

$$
D_{\mathrm{KL}}(q_{0}\;\|\;p_{0})\leq D_{\mathrm{KL}}(q_{T}\;\|\;p_{T})+\int_{\lambda_{T}}^{\lambda_{0}}\mathbb{E}_{q_{0}(\bm{x}_{0})}\mathbb{E}_{\bm{\epsilon}\sim\mathcal{N}(\bm{0},\bm{I})}\Big[\|\bm{\epsilon}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)-\bm{\epsilon}\|_{2}^{2}\Big]\mathrm{d}\lambda+C,
$$

它等价于 [Son21b] 的重要性采样技巧和 [Kin21] 的连续时间扩散损失. 与[命题 3.1](#proposition-03-01) 对照可见, 扩散模型的采样与最大似然训练都能化为关于 $\lambda$ 的积分, 从而不随具体噪声调度而变, [表 2](#table-02) 汇总了这两种形式. 训练与采样共有的不变性带来了理解扩散模型的新视角. 例如, 可以直接关于 (半) log-SNR $\lambda$ 定义噪声预测模型 $\bm{\epsilon}_{\theta}$, 而非关于时间 $t$ 定义; 此时无需另选特定噪声调度即可训练和采样. 这一发现或许能统一扩散模型不同的训练与推断方式, 留待以后研究.

<span id="section-8"></span>

## 8 定理 3.2 的证明

<span id="section-8-1"></span>

### 8.1 假设

本节始终用 $\bm{x}_{s}$ 表示从 $\bm{x}_{T}$ 出发的扩散 ODE [公式 2.7](#equation-2-7) 的解. 对 DPM-Solver-$k$, 作如下假设:

<span id="assumption-08-01"></span>

**假设 8.1.** 全导数 $\frac{\mathrm{d}^j\hat{\bm{\epsilon}}_\theta(\hat{\bm{x}}_\lambda,\lambda)}{\mathrm{d}\lambda^j}$ (视为 $\lambda$ 的函数) 对 $0\leq j\leq k+1$ 存在且连续.

<span id="assumption-08-02"></span>

**假设 8.2.** 函数 $\bm{\epsilon}_\theta(\bm{x},s)$ 关于第一个参数 $\bm{x}$ 满足 Lipschitz 条件.

<span id="assumption-08-03"></span>

**假设 8.3.** $h_{\max}=\mathcal{O}(1/M)$.

第一项假设是 Taylor 定理[公式 3.6](#equation-3-6) 所需的条件. 第二项用于把 $\epsilon_{\theta}(\tilde{\bm{x}}_{s},s)$ 替换为 $\epsilon_{\theta}(\bm{x}_{s},s)+\mathcal{O}(\bm{x}_{s}-\tilde{\bm{x}}_{s})$, 以便关于 $\lambda_{s}$ 作 Taylor 展开. 最后一项是排除过大步长的技术性假设.

<span id="section-8-2"></span>

### 8.2 指数加权积分的一般展开

先推导指数加权积分的 Taylor 展开. 设 $t<s$, 则 $\lambda_{t}>\lambda_{s}$. 记 $h\coloneqq\lambda_{t}-\lambda_{s}$, 并用 $\hat{\bm{\epsilon}}_{\theta}^{(k)}(\hat{\bm{x}}_{\lambda},\lambda)\coloneqq\frac{\mathrm{d}^{k}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)}{\mathrm{d}\lambda^{k}}$ 表示 $k$ 阶全导数. 对 $n\geq 0$, $\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$ 关于 $\lambda$ 的 $n$ 阶 Taylor 展开为

<span id="equation-b-1"></span>

$$
\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)=\sum_{k=0}^{n}\frac{(\lambda-\lambda_{s})^{k}}{k!}\hat{\bm{\epsilon}}_{\theta}^{(k)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{n+1}).
$$

为展开指数积分器, 再定义 [Hoc05]:

<span id="equation-b-2"></span>

$$
\varphi_{k}(z)\coloneqq\int_{0}^{1}\mathrm{e}^{(1-\delta)z}\frac{\delta^{k-1}}{(k-1)!}\mathrm{d}\delta,\quad\quad\varphi_{0}(z)=\mathrm{e}^{z}
$$

它满足 $\varphi_{k}(0)=\frac{1}{k!}$ 和递推关系 $\varphi_{k+1}(z)=\frac{\varphi_{k}(z)-\varphi_{k}(0)}{z}$. 对 $\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$ 作 Taylor 展开, 指数积分器可写为

<span id="equation-b-3"></span>

$$
\int_{\lambda_{s}}^{\lambda_{t}}\mathrm{e}^{-\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\mathrm{d}\lambda=\frac{\sigma_{t}}{\alpha_{t}}\sum_{k=0}^{n}h^{k+1}\varphi_{k+1}(h)\hat{\bm{\epsilon}}_{\theta}^{(k)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{n+2}).
$$

于是, [公式 3.4](#equation-3-4) 中 $\bm{x}_{t}$ 的解可展开为

<span id="equation-b-4"></span>

$$
\bm{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\sum_{k=0}^{n}h^{k+1}\varphi_{k+1}(h)\hat{\bm{\epsilon}}_{\theta}^{(k)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{n+2}).
$$

最后列出 $k=1,2,3$ 时 $\varphi_{k}$ 的闭式:

<span id="equation-b-5"></span>

<span id="equation-b-6"></span>

<span id="equation-b-7"></span>

$$
\begin{aligned}
\varphi_{1}(h) & =\frac{\mathrm{e}^{h}-1}{h}, \\
\varphi_{2}(h) & =\frac{\mathrm{e}^{h}-h-1}{h^{2}}, \\
\varphi_{3}(h) & =\frac{\mathrm{e}^{h}-h^{2}/2-h-1}{h^{3}}.
\end{aligned}
$$

<span id="section-8-3"></span>

### 8.3 $k=1$ 时定理 3.2 的证明

::: details 证明

在[公式 B.4](#equation-b-4) 中取 $n=0,t=t_{i},s=t_{i-1}$, 得

<span id="equation-b-8"></span>

$$
\bm{x}_{t_{i}}=\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\bm{x}_{t_{i-1}}-\sigma_{t}(\mathrm{e}^{h_{i}}-1)\bm{\epsilon}_{\theta}(\bm{x}_{t_{i-1}},{t_{i-1}})+\mathcal{O}(h_{i}^{2}).
$$

由[假设 8.2](#assumption-08-02) 和[公式 3.7](#equation-3-7), 有

$$
\begin{aligned}
\tilde{\bm{x}}_{t_{i}} & =\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_{i}}(\mathrm{e}^{h_{i}}-1)\bm{\epsilon}_{\theta}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}) \\
& =\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_{i}}(\mathrm{e}^{h_{i}}-1)\left(\bm{\epsilon}_{\theta}(\bm{x}_{t_{i-1}},t_{i-1})+\mathcal{O}(\tilde{\bm{x}}_{t_{i-1}}-\bm{x}_{t_{i-1}})\right) \\
& =\frac{\alpha_{t_{i}}}{\alpha_{t_{i-1}}}\bm{x}_{t_{i-1}}-\sigma_{t_{i}}(\mathrm{e}^{h_{i}}-1)\bm{\epsilon}_{\theta}(\bm{x}_{t_{i-1}},t_{i-1})+\mathcal{O}(\tilde{\bm{x}}_{t_{i-1}}-\bm{x}_{t_{i-1}}) \\
& =\bm{x}_{t_{i}}+\mathcal{O}(h_{\max}^{2})+\mathcal{O}(\tilde{\bm{x}}_{t_{i-1}}-\bm{x}_{t_{i-1}}).
\end{aligned}
$$

重复上述论证, 可得

$$
\tilde{\bm{x}}_{t_{M}}=\bm{x}_{t_{0}}+\mathcal{O}(Mh_{\max}^{2})=\bm{x}_{t_{0}}+\mathcal{O}(h_{\max}),
$$

证毕.

:::

<span id="section-8-4"></span>

### 8.4 $k=2$ 时定理 3.2 的证明

下面证明[算法 4](#algorithm-04) 中一般形式 DPM-Solver-2 的离散化误差.

::: details 证明

先考虑 $0<t<s<T,h:=\lambda_{t}-\lambda_{s}$ 时的如下更新.

<span id="equation-b-9a"></span>

<span id="equation-b-9b"></span>

<span id="equation-b-9c"></span>

$$
\begin{aligned}
 s_{1} & =t_{\lambda}\left(\lambda_{s}+r_{1}h\right), \\
\bar{\bm{u}} & =\frac{\alpha_{s_{1}}}{\alpha_{s}}\bm{x}_{s}-\sigma_{s_{1}}\left(\mathrm{e}^{r_{1}h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s), \\
\bar{\bm{x}}_{t} & =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\frac{\sigma_{t}}{2r_{1}}(\mathrm{e}^{h}-1)(\bm{\epsilon}_{\theta}(\bar{\bm{u}},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s)).
\end{aligned}
$$

除把 $\tilde{x}_{t_{i-1}}$ 换成精确解 $\bm{x}_{t_{i-1}}$ 外, 上述更新与取 $s=t_{i-1}$, $t=t_{i}$ 时 DPM-Solver-2 的一步相同. 一旦证明 $\bar{\bm{x}}_{t}=\bm{x}_{t}+\mathcal{O}(h^{3})$, 就能像[第 8.3 节](#section-8-3)一样证明 $\tilde{\bm{x}}_{t_{i}}=\bm{x}_{t_{i}}+\mathcal{O}(h_{\max}^{3})+\mathcal{O}(\tilde{\bm{x}}_{t_{i-1}}-\bm{x}_{t_{i-1}})$, 从而完成证明.

下面证明 $\bar{\bm{x}}_{t}=\bm{x}_{t}+\mathcal{O}(h^{3})$.

在[公式 B.4](#equation-b-4) 中取 $n=1$, 得

<span id="equation-b-10"></span>

$$
\bm{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}h\varphi_{1}(h)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\sigma_{t}h^{2}\varphi_{2}(h)\hat{\bm{\epsilon}}_{\theta}^{(1)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{3}).
$$

由[公式 B.1](#equation-b-1), 有

$$
\begin{aligned}
\bar{\bm{x}}_{t} & =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\frac{\sigma_{t}}{2r_{1}}(\mathrm{e}^{h}-1)(\bm{\epsilon}_{\theta}(\bar{\bm{u}},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s)) \\
& =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\frac{\sigma_{t}}{2r_{1}}\left(\mathrm{e}^{h}-1\right)\left[\bm{\epsilon}_{\theta}(\bar{\bm{u}},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s_{1}},s_{1})\right] \\
&\quad -\frac{\sigma_{t}}{2r_{1}}\left(\mathrm{e}^{h}-1\right)\left[(\lambda_{s_{1}}-\lambda_{s})\hat{\bm{\epsilon}}_{\theta}^{(1)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{2})\right].
\end{aligned}
$$

由 $\bm{\epsilon}_{\theta}$ 关于 $\bm{x}$ 的 Lipschitz 性质 ([假设 8.2](#assumption-08-02)),

$$
\|\bm{\epsilon}_{\theta}(\bar{\bm{u}},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s_{1}},s_{1})\|=\mathcal{O}(\|\bar{\bm{u}}-\bm{x}_{s_{1}}\|)=\mathcal{O}(h^{2}),
$$

最后一个等号可由 $k=1$ 的证明中相似的论证得到. 由于 $\mathrm{e}^{h}-1=\mathcal{O}(h)$, 上式中的第二项为 $\mathcal{O}(h^{3})$.

又因 $\lambda_{s_{1}}-\lambda_{s}=r_{1}h$, $\varphi_{1}(h)=(\mathrm{e}^{h}-1)/h$, $\varphi_{2}(h)=(\mathrm{e}^{h}-h-1)/h^{2}$, 可得

$$
\bm{x}_{t}-\bar{\bm{x}}_{t}=\sigma_{t}\left[h^{2}\varphi_{2}(h)-(\mathrm{e}^{h}-1)\frac{\lambda_{s_{1}}-\lambda_{s}}{2r_{1}}\right]\hat{\bm{\epsilon}}_{\theta}^{(1)}(\hat{\bm{x}}_{\lambda_{s}},\lambda_{s})+\mathcal{O}(h^{3}).
$$

注意到

$$
h^{2}\varphi_{2}(h)-(\mathrm{e}^{h}-1)\frac{\lambda_{s_{1}}-\lambda_{s}}{2r_{1}}=(2\mathrm{e}^{h}-h-2-h\mathrm{e}^{h})/2=\mathcal{O}(h^{3}).
$$

:::

<span id="section-8-5"></span>

### 8.5 $k=3$ 时定理 3.2 的证明

::: details 证明

与[第 8.4 节](#section-8-4)相同, 只需说明对 $0<t<s<T$ 和 $h=\lambda_{s}-\lambda_{t}$, 如下更新的误差为 $\bar{\bm{x}}_{t}=\bm{x}_{t}+\mathcal{O}(h^{4})$.

<span id="equation-b-11a"></span>

<span id="equation-b-11b"></span>

<span id="equation-b-11c"></span>

<span id="equation-b-11d"></span>

<span id="equation-b-11e"></span>

<span id="equation-b-11f"></span>

$$
\begin{aligned}
 s_{1} & =t_{\lambda}\left(\lambda_{s}+r_{1}h\right),\quad s_{2}=t_{\lambda}\left(\lambda_{s}+r_{2}h\right), \\
\bar{\bm{u}}_{1} & =\frac{\alpha_{s_{1}}}{\alpha_{s}}\bm{x}_{s}-\sigma_{s_{1}}\left(\mathrm{e}^{r_{1}h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s), \\
\bm{D}_{1} & =\bm{\epsilon}_{\theta}(\bar{\bm{u}}_{1},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s), \\
\bar{\bm{u}}_{2} & =\frac{\alpha_{s_{2}}}{\alpha_{s}}\bm{x}_{s}-\sigma_{s_{2}}\left(\mathrm{e}^{r_{2}h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\frac{\sigma_{s_{2}}r_{2}}{r_{1}}\left(\frac{\mathrm{e}^{r_{2}h}-1}{r_{2}h}-1\right)\bm{D}_{1}, \\
\bm{D}_{2} & =\bm{\epsilon}_{\theta}(\bar{\bm{u}}_{2},s_{2})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s), \\
\bar{\bm{x}}_{t} & =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\frac{\sigma_{t}}{r_{2}}\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\bm{D}_{2}.
\end{aligned}
$$

先证明

<span id="equation-b-12"></span>

$$
\bar{\bm{u}}_{2}=\bm{x}_{s_{2}}+\mathcal{O}(h^{3}).
$$

与[第 8.4 节](#section-8-4) 的证明类似, 由于 $\frac{\mathrm{e}^{r_{2}h-1}}{r_{2}h}-1=\mathcal{O}(h)$ 且 $\bar{\bm{u}}_{1}=\bm{x}_{s_{1}}+\mathcal{O}(h^{2})$, 有

$$
\begin{aligned}
\bar{\bm{u}}_{2} & =\frac{\alpha_{s_{2}}}{\alpha_{s}}\bm{x}_{s}-\sigma_{s_{2}}\left(\mathrm{e}^{r_{2}h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s) \\
&\quad -\sigma_{s_{2}}\frac{r_{2}}{r_{1}}\left(\frac{\mathrm{e}^{r_{2}h}-1}{r_{2}h}-1\right)\left(\bm{\epsilon}_{\theta}(\bm{x}_{s_{1}},s_{1})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s)\right)+\mathcal{O}(h^{3}) \\
& =\frac{\alpha_{s_{2}}}{\alpha_{s}}\bm{x}_{s}-\sigma_{s_{2}}\left(\mathrm{e}^{r_{2}h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s) \\
&\quad -\sigma_{s_{2}}\frac{r_{2}}{r_{1}}\left(\frac{\mathrm{e}^{r_{2}h}-1}{r_{2}h}-1\right)\bm{\epsilon}^{(1)}_{\theta}(\bm{x}_{s},s)(\lambda_{s_{1}}-\lambda_{s})+\mathcal{O}(h^{3}).
\end{aligned}
$$

令 $h_{2}=r_{2}h$, 沿用[第 8.4 节](#section-8-4) 的论证, 只需验证

$$
\begin{aligned}
\varphi_{1}(h_{2})h_{2} & =\mathrm{e}^{h_{2}}-1, \\
\varphi_{2}(h_{2})h_{2}^{2} & =\frac{r_{2}}{r_{1}}\left(\frac{\mathrm{e}^{h_{2}}-1}{h_{2}}-1\right)(\lambda_{s_{1}}-\lambda_{s})+\mathcal{O}(h^{3}),
\end{aligned}
$$

作 Taylor 展开即可验证两式成立.

利用 $\bar{\bm{u}}_{2}=\bm{x}_{s_{2}}+\mathcal{O}(h^{3})$ 和 $\lambda_{s_{2}}-\lambda_{s}=r_{2}h=\frac{2}{3}h$, 可得

$$
\begin{aligned}
\bar{\bm{x}}_{t} & =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\sigma_{t}\frac{1}{r_{2}}\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\big(\bm{\epsilon}_{\theta}(\bar{\bm{u}}_{2},s_{2})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s)\big) \\
& =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\sigma_{t}\frac{1}{r_{2}}\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\big(\bm{\epsilon}_{\theta}(\bm{x}_{s_{2}},s_{2})-\bm{\epsilon}_{\theta}(\bm{x}_{s},s)\big)+\mathcal{O}(h^{4}) \\
& =\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}\left(\mathrm{e}^{h}-1\right)\bm{\epsilon}_{\theta}(\bm{x}_{s},s) \\
&\quad -\sigma_{t}\frac{1}{r_{2}}\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\big(\bm{\epsilon}^{(1)}_{\theta}(\bm{x}_{s},s)r_{2}h+\frac{1}{2}\bm{\epsilon}^{(2)}_{\theta}(\bm{x}_{s},s)r_{2}^{2}h^{2}\big)+\mathcal{O}(h^{4}).
\end{aligned}
$$

与[公式 B.4](#equation-b-4) 在 $n=2$ 时的 Taylor 展开比较:

$$
\bm{x}_{t}=\frac{\alpha_{t}}{\alpha_{s}}\bm{x}_{s}-\sigma_{t}h\varphi_{1}(h)\bm{\epsilon}_{\theta}(\bm{x}_{s},s)-\sigma_{t}h^{2}\varphi_{2}(h)\bm{\epsilon}_{\theta}^{(1)}(\bm{x}_{s},s)-\sigma_{t}h^{3}\varphi_{3}(h)\bm{\epsilon}_{\theta}^{(2)}(\bm{x}_{s},s)+\mathcal{O}(h^{4}),
$$

需要验证下列条件:

$$
\begin{aligned}
 h\varphi_{1}(h) & =\mathrm{e}^{h}-1, \\
 h^{2}\varphi_{2}(h) & =\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)h, \\
 h^{3}\varphi_{3}(h) & =\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\frac{r_{2}h^{2}}{2}+\mathcal{O}(h^{4}).
\end{aligned}
$$

前两项显然成立. 最后一项可由下式得到:

$$
h^{3}\varphi_{3}(h)=\mathrm{e}^{h}-1-h-\frac{h^{2}}{2}=\frac{h^{3}}{6}+\mathcal{O}(h^{4})=\left(\frac{\mathrm{e}^{h}-1}{h}-1\right)\frac{r_{2}h^{2}}{2}.
$$

因此 $\bar{\bm{x}}_{t}=\bm{x}_{t}+\mathcal{O}(h^{4})$. 证毕.

:::

<span id="section-8-6"></span>

### 8.6 与显式指数 Runge-Kutta (expRK) 方法的联系

设有如下 ODE:

$$
\frac{\mathrm{d}\bm{x}_{t}}{\mathrm{d} t}=\alpha\bm{x}_{t}+\bm{N}(\bm{x}_{t},t),
$$

其中 $\alpha\in\mathbb{R}$, $\bm{N}(\bm{x}_{t},t)\in\mathbb{R}^{D}$ 是 $\bm{x}_{t}$ 的非线性函数. 给定时刻 $t$ 的初值 $\bm{x}_{t}$, 对 $h>0$, 时刻 $t+h$ 的真实解为

$$
\bm{x}_{t+h}=\mathrm{e}^{\alpha h}\bm{x}_{t}+\mathrm{e}^{\alpha h}\int_{0}^{h}\mathrm{e}^{-\alpha\tau}\bm{N}(\bm{x}_{t+\tau},t+\tau)\mathrm{d}\tau.
$$

指数 Runge-Kutta 方法 [Hoc10, Hoc05] 用若干中间点近似积分 $\int \mathrm{e}^{-\alpha\tau}\bm{N}(\bm{x}_{t+\tau},t+\tau)\mathrm{d}\tau$. DPM-Solver 受同一技术启发, 在 $\alpha=1$, $\bm{N}=\tilde{\bm{\epsilon}}_{\theta}$ 时近似相同的积分. 但 DPM-Solver 不同于 expRK 方法: 后者的线性项为 $\mathrm{e}^{\alpha h}\bm{x}_{t}$, 本文的线性项则是 $\frac{\alpha_{t+h}}{\alpha_{t}}\bm{x}_{t}$. 概括而言, DPM-Solver 借用了 expRK 推导指数加权积分高阶近似的技术, 但形式与 expRK 不同, 是针对扩散 ODE 具体形式定制的.

<span id="section-9"></span>

## 9 DPM-Solver 算法

先在[算法 3](#algorithm-03), [算法 4](#algorithm-04) 和[算法 5](#algorithm-05) 中列出 DPM-Solver-1, 2, 3 的细节. DPM-Solver-2 给出 $r_{1}\in(0,1)$ 的一般情形; 与[第 3 节](#section-3)一样, 通常取 $r_{1}=0.5$.

<span id="algorithm-03"></span>

**算法 3: DPM-Solver-1.**

- **输入:** 初值 $\bm{x}_T$, 时间步 $\{t_i\}_{i=0}^M$, 模型 $\bm{\epsilon}_\theta$.
- **定义** $\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}1(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i)$:
  - 令 $h_i\leftarrow\lambda_{t_i}-\lambda_{t_{i-1}}$.
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_i}(\mathrm{e}^{h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - **返回** $\tilde{\bm{x}}_{t_i}$.
- 令 $\tilde{\bm{x}}_{t_0}\leftarrow\bm{x}_T$.
- **对** $i\leftarrow1$ 到 $M$:
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}1(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i)$.
- **返回:** $\tilde{\bm{x}}_{t_M}$.

<span id="algorithm-04"></span>

**算法 4: DPM-Solver-2 (一般形式).**

- **输入:** 初值 $\bm{x}_T$, 时间步 $\{t_i\}_{i=0}^M$, 模型 $\bm{\epsilon}_\theta$, $r_1=0.5$.
- **定义** $\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}2(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i,r_1)$:
  - 令 $h_i\leftarrow\lambda_{t_i}-\lambda_{t_{i-1}}$.
  - 令 $s_i\leftarrow t_\lambda(\lambda_{t_{i-1}}+r_1h_i)$.
  - 令 $\bm{u}_i\leftarrow\frac{\alpha_{s_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_i}(\mathrm{e}^{r_1h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_i}(\mathrm{e}^{h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})-\frac{\sigma_{t_i}}{2r_1}(\mathrm{e}^{h_i}-1)(\bm{\epsilon}_\theta(\bm{u}_i,s_i)-\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}))$.
  - **返回** $\tilde{\bm{x}}_{t_i}$.
- 令 $\tilde{\bm{x}}_{t_0}\leftarrow\bm{x}_T$.
- **对** $i\leftarrow1$ 到 $M$:
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}2(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i,r_1)$.
- **返回:** $\tilde{\bm{x}}_{t_M}$.

<span id="algorithm-05"></span>

**算法 5: DPM-Solver-3.**

- **输入:** 初值 $\bm{x}_T$, 时间步 $\{t_i\}_{i=0}^M$, 模型 $\bm{\epsilon}_\theta$, $r_1=\frac{1}{3}$, $r_2=\frac{2}{3}$.
- **定义** $\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}3(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i,r_1,r_2)$:
  - 令 $h_i\leftarrow\lambda_{t_i}-\lambda_{t_{i-1}}$.
  - 令 $s_{2i-1}\leftarrow t_\lambda(\lambda_{t_{i-1}}+r_1h_i)$, $s_{2i}\leftarrow t_\lambda(\lambda_{t_{i-1}}+r_2h_i)$.
  - 令 $\bm{u}_{2i-1}\leftarrow\frac{\alpha_{s_{2i-1}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_{2i-1}}(\mathrm{e}^{r_1h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\bm{D}_{2i-1}\leftarrow\bm{\epsilon}_\theta(\bm{u}_{2i-1},s_{2i-1})-\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\bm{u}_{2i}\leftarrow\frac{\alpha_{s_{2i}}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{s_{2i}}(\mathrm{e}^{r_2h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})-\frac{\sigma_{s_{2i}}r_2}{r_1}\left(\frac{\mathrm{e}^{r_2h_i}-1}{r_2h_i}-1\right)\bm{D}_{2i-1}$.
  - 令 $\bm{D}_{2i}\leftarrow\bm{\epsilon}_\theta(\bm{u}_{2i},s_{2i})-\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})$.
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\frac{\alpha_{t_i}}{\alpha_{t_{i-1}}}\tilde{\bm{x}}_{t_{i-1}}-\sigma_{t_i}(\mathrm{e}^{h_i}-1)\bm{\epsilon}_\theta(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})-\frac{\sigma_{t_i}}{r_2}\left(\frac{\mathrm{e}^{h_i}-1}{h}-1\right)\bm{D}_{2i}$.
  - **返回** $\tilde{\bm{x}}_{t_i}$.
- 令 $\tilde{\bm{x}}_{t_0}\leftarrow\bm{x}_T$.
- **对** $i\leftarrow1$ 到 $M$:
  - 令 $\tilde{\bm{x}}_{t_i}\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}3(\tilde{\bm{x}}_{t_{i-1}},t_{i-1},t_i,r_1,r_2)$.
- **返回:** $\tilde{\bm{x}}_{t_M}$.

下面给出自适应步长算法 DPM-Solver-12 (组合一阶和二阶, [算法 6](#algorithm-06)) 与 DPM-Solver-23 (组合二阶和三阶, [算法 7](#algorithm-07)). 依照 [Jol21], 对图像数据令绝对容差 $\epsilon_{\text{atol}}=\frac{\bm{x}_{\max}-\bm{x}_{\min}}{256}$; 对 VP 型 DPM, 其值为 $0.0078$. 可以调节相对容差 $\epsilon_{\text{rtol}}$ 来平衡精度与 NFE. 实验发现 $\epsilon_{\text{rtol}}=0.05$ 已足以快速收敛.

实践中, 自适应步长求解器的输入是批量数据. 我们直接把 $E_{2}$ 和 $E_{3}$ 取为整批数据中的最大值. 为避免数值问题, 还用 $|s-\epsilon|>10^{-5}$ 实现比较 $s>\epsilon$.

<span id="algorithm-06"></span>

**算法 6: (DPM-Solver-12) 组合 DPM-Solver-1 和 2 的自适应步长算法.**

- **输入:** 起始时刻 $T$, 结束时刻 $\epsilon$, 初值 $\bm{x}_T$, 模型 $\bm{\epsilon}_\theta$, 数据维度 $D$, 超参数 $\epsilon_{\mathrm{rtol}}=0.05$, $\epsilon_{\mathrm{atol}}=0.0078$, $h_{\mathrm{init}}=0.05$, $\theta=0.9$.
- **输出:** 时刻 $\epsilon$ 的近似解 $\bm{x}_\epsilon$.
- 令 $s\leftarrow T$, $h\leftarrow h_{\mathrm{init}}$, $\bm{x}\leftarrow\bm{x}_T$, $\bm{x}_{\mathrm{prev}}\leftarrow\bm{x}_T$, $r_1\leftarrow\frac{1}{2}$, $\mathrm{NFE}\leftarrow0$.
- **当** $s>\epsilon$ 时:
  - 令 $t\leftarrow t_\lambda(\lambda_s+h)$.
  - 令 $\bm{x}_1\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}1(\bm{x},s,t)$.
  - 令 $\bm{x}_2\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}2(\bm{x},s,t,r_1)$ (与 DPM-Solver-1 共享函数值 $\bm{\epsilon}_\theta(\bm{x},s)$).
  - 令 $\bm{\delta}\leftarrow\max(\epsilon_{\mathrm{atol}},\epsilon_{\mathrm{rtol}}\max(|\bm{x}_1|,|\bm{x}_{\mathrm{prev}}|))$.
  - 令 $E_2\leftarrow\frac{1}{\sqrt{D}}\|\frac{\bm{x}_1-\bm{x}_2}{\bm{\delta}}\|_2$.
  - **若** $E_2\leq1$:
    - 令 $\bm{x}_{\mathrm{prev}}\leftarrow\bm{x}_1$, $\bm{x}\leftarrow\bm{x}_2$, $s\leftarrow t$.
  - 令 $h\leftarrow\min(\theta hE_2^{-\frac{1}{2}},\lambda_\epsilon-\lambda_s)$.
  - 令 $\mathrm{NFE}\leftarrow\mathrm{NFE}+2$.
- **返回:** $\bm{x}$, $\mathrm{NFE}$.

<span id="algorithm-07"></span>

**算法 7: (DPM-Solver-23) 组合 DPM-Solver-2 和 3 的自适应步长算法.**

- **输入:** 起始时刻 $T$, 结束时刻 $\epsilon$, 初值 $\bm{x}_T$, 模型 $\bm{\epsilon}_\theta$, 数据维度 $D$, 超参数 $\epsilon_{\mathrm{rtol}}=0.05$, $\epsilon_{\mathrm{atol}}=0.0078$, $h_{\mathrm{init}}=0.05$, $\theta=0.9$.
- **输出:** 时刻 $\epsilon$ 的近似解 $\bm{x}_\epsilon$.
- 令 $s\leftarrow T$, $h\leftarrow h_{\mathrm{init}}$, $\bm{x}\leftarrow\bm{x}_T$, $\bm{x}_{\mathrm{prev}}\leftarrow\bm{x}_T$, $r_1\leftarrow\frac{1}{3}$, $r_2\leftarrow\frac{2}{3}$, $\mathrm{NFE}\leftarrow0$.
- **当** $s>\epsilon$ 时:
  - 令 $t\leftarrow t_\lambda(\lambda_s+h)$.
  - 令 $\bm{x}_2\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}2(\bm{x},s,t,r_1)$.
  - 令 $\bm{x}_3\leftarrow\mathrm{DPM}\text{-}\mathrm{Solver}\text{-}3(\bm{x},s,t,r_1,r_2)$ (与 DPM-Solver-2 共享函数值).
  - 令 $\bm{\delta}\leftarrow\max(\epsilon_{\mathrm{atol}},\epsilon_{\mathrm{rtol}}\max(|\bm{x}_2|,|\bm{x}_{\mathrm{prev}}|))$.
  - 令 $E_3\leftarrow\frac{1}{\sqrt{D}}\|\frac{\bm{x}_2-\bm{x}_3}{\bm{\delta}}\|_2$.
  - **若** $E_3\leq1$:
    - 令 $\bm{x}_{\mathrm{prev}}\leftarrow\bm{x}_2$, $\bm{x}\leftarrow\bm{x}_3$, $s\leftarrow t$.
  - 令 $h\leftarrow\min(\theta hE_3^{-\frac{1}{3}},\lambda_\epsilon-\lambda_s)$.
  - 令 $\mathrm{NFE}\leftarrow\mathrm{NFE}+3$.
- **返回:** $\bm{x}$, $\mathrm{NFE}$.

<span id="section-10"></span>

## 10 DPM-Solver 的实现细节

<span id="section-10-1"></span>

### 10.1 采样结束时刻

理论上, 要生成样本, 需要从时刻 $T$ 到 $0$ 求解扩散 ODE. 实践中, 为避免 $t$ 接近 $0$ 时的数值问题, 噪声预测模型 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t)$ 的训练和评估通常从时刻 $T$ 进行到时刻 $\epsilon$, 其中 $\epsilon>0$ 是超参数 [Son21].

不同于基于扩散 SDE 的采样方法 [Den20, Son21], 我们不在最后的 $\epsilon$ 时刻加入“去噪”技巧 (即把噪声方差设为零), 而只用 DPM-Solver 从 $T$ 到 $\epsilon$ 求解扩散 ODE, 因为实验表明这样已经足够好.

对离散时间 DPM, 先把模型转换到连续时间 (见[第 10.2 节](#section-10-2)), 再从时刻 $T$ 求解到时刻 $t$.

<span id="section-10-2"></span>

### 10.2 从离散时间 DPM 采样

本节讨论离散时间 DPM 的一般情形, 包括 1000 步 DPM [Den20] 和 4000 步 DPM [Nic21], 并考虑采样结束时刻 $\epsilon$.

离散时间 DPM [Den20] 在 $N$ 个固定时间步 $\{t_{n}\}_{n=1}^{N}$ 上训练噪声预测模型. 实践中 $N=1000$ 或 $N=4000$, 4000 步 DPM 的实现 [Nic21] 会把时间步转换到 1000 步 DPM 的范围. 具体而言, 噪声预测模型记为 $\tilde{\bm{\epsilon}}_{\theta}(\bm{x}_{n},\frac{1000n}{N})$, 其中 $n=0,\dots,N-1$, 每个 $\bm{x}_{n}$ 对应时刻 $t_{n+1}$ 的取值. 实践中的离散时间 DPM 通常在 $[0,T]$ 之间取均匀时间步, 即 $t_{n}=\frac{nT}{N}$, $n=1,\dots,N$.

但离散时间噪声预测模型无法预测早于最小时间 $t_{1}$ 的噪声. 最小时间步为 $t_{1}=\frac{T}{N}$, 对应的离散时间噪声预测模型为 $\tilde{\bm{\epsilon}}_{\theta}(\bm{x}_{0},0)$, 因而需要把离散时间步 $[t_{1},t_{N}]=[\frac{T}{N},T]$ “缩放”到连续时间范围 $[\epsilon,T]$. 本文提出如下两种缩放方式.

**类型 1.** 把离散时间步 $[t_{1},t_{N}]=[\frac{T}{N},T]$ 缩放到连续时间范围 $[\frac{T}{N},T]$, 并对 $t\in[\epsilon,\frac{T}{N}]$ 令 $\bm{\epsilon}_{\theta}(\cdot,t)=\bm{\epsilon}_{\theta}(\cdot,\frac{T}{N})$. 此时可定义连续时间噪声预测模型

<span id="equation-d-1"></span>

$$
\bm{\epsilon}_{\theta}(\bm{x},t)=\tilde{\bm{\epsilon}}_{\theta}\left(\bm{x},1000\cdot\max\left(t-\frac{T}{N},0\right)\right),
$$

其中连续时间 $t\in[\epsilon,\frac{T}{N}]$ 映射到离散输入 $0$, 连续时间 $T$ 映射到离散输入 $\frac{1000(N-1)}{N}$.

**类型 2.** 把离散时间步 $[t_{1},t_{N}]=[\frac{T}{N},T]$ 缩放到连续时间范围 $[0,T]$. 此时可定义连续时间噪声预测模型

<span id="equation-d-2"></span>

$$
\bm{\epsilon}_{\theta}(\bm{x},t)=\tilde{\bm{\epsilon}}_{\theta}\left(\bm{x},1000\cdot\frac{(N-1)t}{N T}\right),
$$

其中连续时间 $0$ 映射到离散输入 $0$, 连续时间 $T$ 映射到离散输入 $\frac{1000(N-1)}{N}$.

$\tilde{\bm{\epsilon}}_{\theta}$ 的时间输入可能不是整数, 但实验表明噪声预测模型仍能正常工作; 我们推测这是因为时间嵌入较为平滑, 例如位置嵌入 [Den20]. 经过这种重参数化, 噪声预测模型可以接收连续时间步, 因而也能用 DPM-Solver 快速采样.

实践中取 $T=1$, 最小离散时间为 $t_{1}=10^{-3}$. 固定函数求值次数 $K$ 后, 实验发现 $K$ 较小时, 取 $\epsilon=10^{-3}$ 的类型 1 可能得到更好的样本质量; $K$ 较大时, 取 $\epsilon=10^{-4}$ 的类型 2 可能更好. 详细结果见[第 11 节](#section-11).

<span id="section-10-3"></span>

### 10.3 20 次函数求值以内的 DPM-Solver

给定函数求值次数的固定预算 $K\leq 20$, 把区间 $[\lambda_{T},\lambda_{\epsilon}]$ 均匀划分成 $M=(\lfloor K/3\rfloor+1)$ 段, 用 $M$ 步生成样本. 这 $M$ 步取决于 $K$ 模 $3$ 的余数 $R$, 以保证函数求值总次数恰好为 $K$.

- 若 $R=0$, 先执行 $M-2$ 步 DPM-Solver-3, 再分别执行 1 步 DPM-Solver-2 和 1 步 DPM-Solver-1. 函数求值总次数为 $3\cdot(\frac{K}{3}-1)+2+1=K$.

- 若 $R=1$, 先执行 $M-1$ 步 DPM-Solver-3, 再执行 1 步 DPM-Solver-1. 函数求值总次数为 $3\cdot(\frac{K-1}{3})+1=K$.

- 若 $R=2$, 先执行 $M-1$ 步 DPM-Solver-3, 再执行 1 步 DPM-Solver-2. 函数求值总次数为 $3\cdot(\frac{K-2}{3})+2=K$.

实验发现, 这种时间步设计能显著改善生成质量. DPM-Solver 用 10 步即可生成质量相当的样本, 用 20 步可生成高质量样本.

<span id="section-10-4"></span>

### 10.4 $t_\lambda(\cdot)$ ($\lambda(t)$ 的逆函数) 的解析形式

计算 $t_{\lambda}(\cdot)$ 的代价可以忽略不计. 以往 DPM 采用的 $\alpha_{t}$ 和 $\sigma_{t}$ 噪声调度, 即“线性”和“余弦”调度 [Den20, Nic21], 其 $\lambda(t)$ 与逆函数 $t_{\lambda}(\cdot)$ 均有解析形式. 这里主要考虑最常用的方差保持型; 其他类型 (方差爆炸型和次方差保持型) 的函数可用类似方法推导.

**线性噪声调度 [Den20].** 有

$$
\log\alpha_{t}=-\frac{(\beta_{1}-\beta_{0})}{4}t^{2}-\frac{\beta_{0}}{2}t,
$$

依照 [Son21], 其中 $\beta_{0}=0.1$, $\beta_{1}=20$. 由于 $\sigma_{t}=\sqrt{1-\alpha_{t}^{2}}$, 可以解析计算 $\lambda_{t}$. 其逆函数为

$$
t_{\lambda}(\lambda)=\frac{1}{\beta_{1}-\beta_{0}}\left(\sqrt{\beta_{0}^{2}+2(\beta_{1}-\beta_{0})\log\left(\mathrm{e}^{-2\lambda}+1\right)}-\beta_{0}\right).
$$

为减小数值问题的影响, 可用下列等价形式计算 $t_{\lambda}$:

$$
t_{\lambda}(\lambda)=\frac{2\log\left(\mathrm{e}^{-2\lambda}+1\right)}{\sqrt{\beta_{0}^{2}+2(\beta_{1}-\beta_{0})\log\left(\mathrm{e}^{-2\lambda}+1\right)}+\beta_{0}}.
$$

扩散 ODE 在 $[\epsilon,T]$ 之间求解, 其中 $T=1$.

**余弦噪声调度 [Nic21].** 记

$$
\log\alpha_{t}=\log\left(\cos\left(\frac{\pi}{2}\cdot\frac{t+s}{1+s}\right)\right)-\log\left(\cos\left(\frac{\pi}{2}\cdot\frac{s}{1+s}\right)\right),
$$

依照 [Nic21], 其中 $s=0.008$. [Nic21] 为保证数值稳定性而截断导数, 因此我们也把最大时间截为 $T=0.9946$. 由于 $\sigma_{t}=\sqrt{1-\alpha_{t}^{2}}$, 可以解析计算 $\lambda_{t}$. 给定固定的 $\lambda$, 令

$$
f(\lambda)=-\frac{1}{2}\log\left(\mathrm{e}^{-2\lambda}+1\right),
$$

它计算 $\lambda$ 对应的 $\log\alpha$. 逆函数为

$$
t_{\lambda}(\lambda)=\frac{2(1+s)}{\pi}\arccos\left(\mathrm{e}^{f(\lambda)+\log\cos\left(\frac{\pi s}{2(1+s)}\right)}\right)-s.
$$

扩散 ODE 在 $[\epsilon,T]$ 之间求解, 其中 $T=0.9946$.

<span id="section-10-5"></span>

### 10.5 用 DPM-Solver 作条件采样

稍作修改后, DPM-Solver 也可用于条件采样. 条件生成需要从包含条件噪声预测模型的条件扩散 ODE [Son21, Dha21] 采样. 按照分类器引导方法 [Dha21], 定义条件噪声预测模型 $\bm{\epsilon}_{\theta}(\bm{x}_{t},t,y)\coloneqq\bm{\epsilon}_{\theta}(\bm{x}_{t},t)-s\cdot\sigma_{t}\nabla_{\bm{x}}\log p_{t}(y|\bm{x}_{t};\theta)$, 其中 $p_{t}(y|\bm{x}_{t};\theta)$ 是预训练分类器, $s$ 是分类器引导尺度 (默认为 1.0). 由此可用 DPM-Solver 求解该扩散 ODE, 快速完成条件采样, 如[图 1](#figure-01) 所示.

<span id="section-10-6"></span>

### 10.6 数值稳定性

DPM-Solver 算法需要计算 $\mathrm{e}^{h_{i}}-1$. 依照 [Kin21], 我们用 expm1($h_{i}$) 代替 exp($h_{i}$)-1, 以提高数值稳定性.

<span id="section-11"></span>

## 11 实验细节

实验对最常用的方差保持 (VP) 型 DPM [Soh15, Den20] 进行采样. 此时对所有 $t\in[0,T]$, 都有 $\alpha_{t}^{2}+\sigma_{t}^{2}=1$ 且 $\tilde{\sigma}=1$. 尽管如此, 本文方法与理论结果是通用的, 不依赖于噪声调度 $\alpha_{t}$ 和 $\sigma_{t}$ 的选择.

所有实验均在 NVIDIA A40 GPU 上评估 DPM-Solver. 不过, 也可通过调整采样批量大小, 使用 NVIDIA GeForce RTX 2080Ti 等其他 GPU.

<span id="section-11-1"></span>

### 11.1 关于 $\lambda$ 的扩散 ODE

还可以把扩散 ODE 重参数化到 $\lambda$ 域. 本节给出 VP 型扩散 ODE 关于 $\lambda$ 的形式, 其他类型可用类似方法推导.

对给定 $\lambda$, 记 $\hat{\alpha}_{\lambda}\coloneqq\alpha_{t(\lambda)}$, $\hat{\sigma}_{\lambda}\coloneqq\sigma_{t(\lambda)}$. 由 $\hat{\alpha}_{\lambda}^{2}+\hat{\sigma}_{\lambda}^{2}=1$, 可证明 $\frac{\mathrm{d}\lambda}{\mathrm{d}\hat{\alpha}_{\lambda}}=\frac{1}{\hat{\alpha}_{\lambda}\hat{\sigma}^{2}_{\lambda}}$, 因此 $\frac{\mathrm{d}\log\hat{\alpha}_{\lambda}}{\mathrm{d}\lambda}=\hat{\sigma}^{2}_{\lambda}$. 对[公式 2.7](#equation-2-7) 作变量代换, 得

<span id="equation-e-1"></span>

$$
\frac{\mathrm{d}\hat{\bm{x}}_{\lambda}}{\mathrm{d}\lambda}=\hat{\bm{h}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)\coloneqq\hat{\sigma}_{\lambda}^{2}\hat{\bm{x}}_{\lambda}-\hat{\sigma}_{\lambda}\hat{\bm{\epsilon}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda).
$$

也可用 RK 方法直接求解 ODE [公式 E.1](#equation-e-1). [表 1](#table-01) 中 RK2 ($\lambda$) 和 RK3 ($\lambda$) 的实验采用了这一形式.

<span id="section-11-2"></span>

### 11.2 代码实现

代码同时使用 JAX (用于连续时间 DPM) 和 PyTorch (用于离散时间 DPM) 实现, 发布于 <https://github.com/LuChengTHU/dpm-solver>.

<span id="section-11-3"></span>

### 11.3 与连续时间采样方法的样本质量比较

<span id="table-03"></span>

![连续时间采样方法在 CIFAR-10 上的 FID](./dpm-solver/table-03.png)

**表 3.** 在 CIFAR-10 数据集上改变函数求值次数 (NFE), 以 FID $\downarrow$ 衡量连续时间方法的样本质量.

[表 3](#table-03) 给出了与[图 2(a)](#figure-02) 对应的详细 FID 结果. 实验使用 [Son21] 的官方代码与检查点, 代码采用 Apache License 2.0; 具体使用其发布的“VP deep”类型“checkpoint_8”. 我们比较 $\epsilon=10^{-3}$ 和 $\epsilon=10^{-4}$ 时的各方法. 实验发现, 基于扩散 SDE 的采样方法在 $\epsilon=10^{-3}$ 时样本质量更好, 基于扩散 ODE 的采样方法在 $\epsilon=10^{-4}$ 时更好. 对 DPM-Solver, NFE 小于 15 时, $\epsilon=10^{-3}$ 的 FID 优于 $\epsilon=10^{-4}$; NFE 大于 15 时则相反.

对采用 Euler 离散化的扩散 SDE, 使用 [Son21] 中预测器为“euler_maruyama”, 无校正器的 PC 采样器, 它在 $T$ 与 $\epsilon$ 之间采用均匀时间步. 最后一步加入“去噪”技巧, 可显著改善 $\epsilon=10^{-3}$ 时的 FID.

对采用 Improved Euler 离散化的扩散 SDE [Jol21], 直接沿用原论文只包含 $\epsilon=10^{-3}$ 的结果. 对应的相对容差 $\epsilon_{rel}$ 分别为 $0.50$, $0.10$ 和 $0.05$.

对采用 RK45 求解器的扩散 ODE, 使用 [Son21] 的代码, 并调节求解器的 atol 和 rtol. 随着 NFE 从小到大, $\epsilon=10^{-3}$ 的结果依次采用相同的 atol = rtol = $0.1$, $0.01$, $0.001$; $\epsilon=10^{-4}$ 的结果依次采用相同的 atol = rtol = $0.1$, $0.05$, $0.02$, $0.01$, $0.001$.

对采用 DPM-Solver 的扩散 ODE, NFE $\leq 20$ 时使用[第 10.3 节](#section-10-3) 的方法, 否则使用[第 9 节](#section-9) 的自适应步长求解器. $\epsilon=10^{-3}$ 时采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的 DPM-Solver-12; $\epsilon=10^{-4}$ 时采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的 DPM-Solver-23.

<span id="section-11-4"></span>

### 11.4 与 RK 方法的样本质量比较

[表 1](#table-01) 比较了 RK 方法与 DPM-Solver-2, 3 的表现. 本节列出详细设置.

设有 ODE

$$
\frac{\mathrm{d}\bm{x}_{t}}{\mathrm{d}t}=\bm{F}(\bm{x}_{t},t),
$$

从时刻 $t_{i-1}$ 的 $\tilde{\bm{x}}_{t_{i-1}}$ 出发, 用 RK2 按下式近似时刻 $t_{i}$ 的解 $\tilde{\bm{x}}_{t_{i}}$ (即显式中点法):

$$
\begin{aligned}
 h_{i} & =t_{i}-t_{i-1}, \\
 s_{i} & =t_{i-1}+\frac{1}{2}h_{i}, \\
\bm{u}_{i} & =\tilde{\bm{x}}_{t_{i-1}}+\frac{h_{i}}{2}\bm{F}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}), \\
\tilde{\bm{x}}_{t_{i}} & =\tilde{\bm{x}}_{t_{i-1}}+h_{i}\bm{F}(\bm{u}_{i},s_{i}).
\end{aligned}
$$

另用下列 RK3 近似时刻 $t_{i}$ 的解 $\tilde{\bm{x}}_{t_{i}}$ (即“Heun 三阶法”), 因为它与 DPM-Solver-3 很相似:

$$
\begin{aligned}
 h_{i} & =t_{i}-t_{i-1},\quad r_{1}=\frac{1}{3},\quad r_{2}=\frac{2}{3}, \\
 s_{2i-1} & =t_{i-1}+r_{1}h_{i},\quad s_{2i}=t_{i-1}+r_{2}h_{i}, \\
\bm{u}_{2i-1} & =\tilde{\bm{x}}_{t_{i-1}}+r_{1}h_{i}\bm{F}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1}), \\
\bm{u}_{2i} & =\tilde{\bm{x}}_{t_{i-1}}+r_{2}h_{i}\bm{F}(\bm{u}_{2i-1},s_{2i-1}), \\
\tilde{\bm{x}}_{t_{i}} & =\tilde{\bm{x}}_{t_{i-1}}+\frac{h_{i}}{4}\bm{F}(\tilde{\bm{x}}_{t_{i-1}},t_{i-1})+\frac{3h_{i}}{4}\bm{F}(\bm{u}_{2i},s_{2i}).
\end{aligned}
$$

对 RK2 ($t$) 和 RK3 ($t$), 取[公式 2.7](#equation-2-7) 中的 $\bm{F}(\bm{x}_{t},t)=\bm{h}_{\theta}(\bm{x}_{t},t)$; 对 RK2 ($\lambda$) 和 RK3 ($\lambda$), 取[公式 E.1](#equation-e-1) 中的 $\bm{F}(\hat{\bm{x}}_{\lambda},\lambda)=\hat{\bm{h}}_{\theta}(\hat{\bm{x}}_{\lambda},\lambda)$. 所有实验都关于 $t$ 或 $\lambda$ 采用均匀步长.

<span id="section-11-5"></span>

### 11.5 与离散时间采样方法的样本质量比较

<span id="table-04"></span>

![离散时间模型在 CIFAR-10, CelebA 和 ImageNet 上的 FID](./dpm-solver/table-04.png)

**表 4.** 改变函数求值次数 (NFE), 比较离散时间 DPM 在 CIFAR-10, CelebA $64\times64$ 和 ImageNet $64\times64$ 上的样本质量, 指标为 FID $\downarrow$. $^\dagger$GGDM 需要额外训练; 原论文缺失的结果以“$\backslash$”代替.

<span id="table-05"></span>

![ImageNet 128x128 与 LSUN bedroom 256x256 上的 FID](./dpm-solver/table-05.png)

**表 5.** 改变函数求值次数 (NFE), 比较带分类器引导的 ImageNet $128\times128$ 与 LSUN bedroom $256\times256$ 上的样本质量, 指标为 FID $\downarrow$. 除标有 $^\dagger$ 的实验使用 [Dha21] 微调的时间步外, DDIM 和 DDPM 的所有实验都采用均匀时间步. DPM-Solver 采用[第 10.3 节](#section-10-3) 所述的均匀 log-SNR 步.

如[表 4](#table-04) 和[表 5](#table-05) 所示, 我们把 DPM-Solver 与其他离散时间 DPM 采样方法作了比较. DDPM 和 DDIM 采样采用 [Son21a] 的代码, 代码为 MIT License. Analytic-DDPM 和 Analytic-DDIM 采样采用 [Bao22] 的代码, 许可证未知. GGDM [Wat22] 直接沿用原论文的最佳结果.

CIFAR-10 实验采用 [Den20] 的预训练检查点, [Son21a] 发布的代码也提供了该检查点. DDPM 和 DDIM 采用二次时间步, 实验中其 FID 优于均匀时间步 [Son21a]. Analytic-DDPM 和 Analytic-DDIM 采用均匀时间步. 对 DPM-Solver, 分别用类型 1 和类型 2 离散方法把离散时间模型转为连续时间模型. NFE $\leq 20$ 时采用[第 10.3 节](#section-10-3) 的方法, NFE $>20$ 时采用[第 9 节](#section-9) 的自适应步长求解器. 所有实验都采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的 DPM-Solver-12.

CelebA $64\times64$ 实验采用 [Son21a] 的预训练检查点. DDPM 和 DDIM 采用二次时间步, 实验中其 FID 优于均匀时间步 [Son21a]. Analytic-DDPM 和 Analytic-DDIM 采用均匀时间步. 对 DPM-Solver, 分别用类型 1 和类型 2 离散方法把离散时间模型转为连续时间模型. NFE $\leq 20$ 时采用[第 10.3 节](#section-10-3) 的方法, NFE $>20$ 时采用[第 9 节](#section-9) 的自适应步长求解器. 所有实验都采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的 DPM-Solver-12. CelebA $64\times64$ 上最好的 FID 结果甚至优于 1000 步 DDPM 和所有其他方法.

ImageNet $64\times64$ 实验采用 [Nic21] 的预训练检查点, 代码为 MIT License. 按照 [Son21a], DDPM 和 DDIM 采用均匀时间步. Analytic-DDPM 和 Analytic-DDIM 也采用均匀时间步. 对 DPM-Solver, 分别用类型 1 和类型 2 离散方法把离散时间模型转为连续时间模型. NFE $\leq 20$ 时采用[第 10.3 节](#section-10-3) 的方法, NFE $>20$ 时采用[第 9 节](#section-9) 的自适应步长求解器. 所有实验都采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的 DPM-Solver-23. ImageNet 数据集包含真实人物照片, 可能涉及隐私问题, 见 [Yan21e] 的讨论.

ImageNet $128\times128$ 实验采用 [Dha21] 的预训练检查点 (扩散模型与分类器模型均包括), 以分类器引导采样, 代码为 MIT License. 按照 [Son21a], DDPM 和 DDIM 采用均匀时间步. DPM-Solver 只用类型 1 离散方法把离散时间模型转为连续时间模型. NFE $\leq 20$ 时采用[第 10.3 节](#section-10-3) 的方法; NFE $>20$ 时采用相对容差 $\epsilon_{\text{rtol}}=0.05$ 的自适应步长 DPM-Solver-12, 详见[第 9 节](#section-9). 所有实验的分类器引导尺度均设为 $s=1.25$, 这是 [Dha21] 中 DDIM 的最佳设置 (详见其第 14 个表格).

LSUN bedroom $256\times256$ 实验采用 [Dha21] 的无条件预训练检查点, 代码为 MIT License. 按照 [Son21a], DDPM 和 DDIM 采用均匀时间步. DPM-Solver 只用类型 1 离散方法把离散时间模型转为连续时间模型, 并采用[第 10.3 节](#section-10-3) 的方法.

<span id="section-11-6"></span>

### 11.6 比较不同阶数的 DPM-Solver

我们还比较了不同阶数 DPM-Solver 的样本质量, 见[表 6](#table-06). DPM-Solver-1, 2, 3 关于 $\lambda$ 采用均匀时间步; NFE 小于 20 时采用[第 10.3 节](#section-10-3) 的快速版本, 称为 DPM-Solver-fast. 对离散时间模型只比较类型 2 离散方法, 类型 1 的结果相近.

DPM-Solver-2 的实际 NFE 为 $2\times\lfloor\text{NFE}/2\rfloor$, DPM-Solver-3 的实际 NFE 为 $3\times\lfloor\text{NFE}/3\rfloor$, 可能小于给定的 NFE, 因而用 ^†^ 标出实际 NFE 小于给定值的结果. 实验发现, NFE 小于 20 时, 提出的快速版本 DPM-Solver-fast 通常优于单一阶数的方法; NFE 较大时, DPM-Solver-3 优于 DPM-Solver-2, DPM-Solver-2 又优于 DPM-Solver-1, 与本文的收敛率分析一致.

<span id="table-06"></span>

![不同阶数 DPM-Solver 的 FID](./dpm-solver/table-06.png)

**表 6.** 改变函数求值次数 (NFE), 比较不同阶数 DPM-Solver 的样本质量, 指标为 FID $\downarrow$. 标有 $^\dagger$ 的结果表示给定 NFE 不能被 $2$ 或 $3$ 整除, 因而实际 NFE 小于给定值. DPM-Solver-fast 只在 NFE 小于 20 时评估, 因为 NFE 较大时它与 DPM-Solver-3 几乎相同.

<span id="section-11-7"></span>

### 11.7 DPM-Solver 与 DDIM 的运行时间比较

理论上, NFE 相同时, DPM-Solver 与 DDIM 的运行时间几乎相同, 都与 NFE 线性相关. 这是因为主要计算开销来自大型神经网络 $\bm{\epsilon}_{\theta}$ 的串行求值, 其他系数均解析计算, 开销可忽略.

[表 7](#table-07) 比较了在单张 NVIDIA A40 上, 不同数据集与 NFE 下 DPM-Solver 和 DDIM 的运行时间. 实验用 torch.cuda.Event 和 torch.cuda.synchronize 精确计算运行时间. 每个数据集都采用离散时间预训练扩散模型, 对 8 个批次进行评估并计算运行时间的均值与标准差. 受 GPU 显存限制, LSUN bedroom $256\times256$ 的批量大小为 64, 其他数据集为 128.

DDIM 采用官方实现 [+2]. DPM-Solver 的实现减少了一些系数的重复计算, 因此在相同 NFE 下略快于该 DDIM 实现. 尽管如此, 运行时间评估表明, 相同 NFE 下 DPM-Solver 与 DDIM 的运行时间几乎相同, 且运行时间近似与 NFE 成线性关系. 因此, NFE 的加速几乎就是实际运行时间的加速, DPM-Solver 确实能显著加快 DPM 采样.

<span id="table-07"></span>

![DDIM 与 DPM-Solver 在单张 NVIDIA A40 上的运行时间](./dpm-solver/table-07.png)

**表 7.** 改变函数求值次数 (NFE), 比较离散时间扩散模型采样时, DDIM 与 DPM-Solver 在单张 NVIDIA A40 上处理一个批次的运行时间 (秒/批次, $\pm$std).

<span id="section-11-8"></span>

### 11.8 ImageNet 256x256 上的条件采样

[图 1](#figure-01) 的条件采样采用 [Dha21] 带分类器引导的预训练检查点 (ADM-G), 分类器尺度为 $1.0$, 代码为 MIT License. DDIM 采用均匀时间步; DPM-Solver 采用[第 10.3 节](#section-10-3) 的快速版本 (DPM-Solver-fast), 步数分别为 10, 15, 20 和 100.

[图 3](#figure-03) 展示了 DDIM 与 DPM-Solver 的条件采样结果. DPM-Solver 用 15 NFE 生成的样本可与 DDIM 用 100 NFE 的结果相当.

<span id="figure-03"></span>

![DDIM 与 DPM-Solver 在 ImageNet 256x256 上生成的样本](./dpm-solver/figure-03.png)

**图 3.** 使用 ImageNet $256\times256$ 上预训练且带分类器引导的 DPM [Dha21], DDIM [Son21a] 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 15, 20, 100 次函数求值 (NFE) 生成的样本.

<span id="section-11-9"></span>

### 11.9 更多样本

CIFAR-10, CelebA $64\times64$, ImageNet $64\times64$, LSUN bedroom $256\times256$ [Yu15a] 和 ImageNet $256\times256$ 上的更多采样结果见[图 4](#figure-04), [图 5](#figure-05), [图 6](#figure-06), [图 7](#figure-07) 和[图 8](#figure-08).

<span id="figure-04"></span>

![DDIM 与 DPM-Solver 在 CIFAR-10 上生成的样本](./dpm-solver/figure-04.png)

**图 4.** 使用 CIFAR-10 上预训练的离散时间 DPM [Den20], DDIM [Son21a] (二次时间步) 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 12, 15, 20 次函数求值 (NFE) 生成的随机样本.

<span id="figure-05"></span>

![DDIM 与 DPM-Solver 在 CelebA 64x64 上生成的样本](./dpm-solver/figure-05.png)

**图 5.** 使用 CelebA $64\times64$ 上预训练的离散时间 DPM [Son21a], DDIM [Son21a] (二次时间步) 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 12, 15, 20 次函数求值 (NFE) 生成的随机样本.

<span id="figure-06"></span>

![DDIM 与 DPM-Solver 在 ImageNet 64x64 上生成的样本](./dpm-solver/figure-06.png)

**图 6.** 使用 ImageNet $64\times64$ 上预训练的离散时间 DPM [Nic21], DDIM [Son21a] (均匀时间步) 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 12, 15, 20 次函数求值 (NFE) 生成的随机样本.

<span id="figure-07"></span>

![DDIM 与 DPM-Solver 在 LSUN bedroom 256x256 上生成的样本](./dpm-solver/figure-07.png)

**图 7.** 使用 LSUN bedroom $256\times256$ 上预训练的离散时间 DPM [Dha21], DDIM [Son21a] (均匀时间步) 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 12, 15, 20 次函数求值 (NFE) 生成的随机样本.

<span id="figure-08"></span>

![DDIM 与 DPM-Solver 生成的 ImageNet 鹦鹉类别条件样本](./dpm-solver/figure-08.png)

**图 8.** 使用 ImageNet $256\times256$ 上预训练且带分类器引导的离散时间 DPM [Dha21] (分类器尺度: 1.0), DDIM [Son21a] (均匀时间步) 与 DPM-Solver (本文方法) 在相同随机种子下, 分别用 10, 12, 15, 20 次函数求值 (NFE) 生成的随机类别条件样本 (类别: 90, 鹦鹉).

[+1]: 代码发布于 [https://github.com/LuChengTHU/dpm-solver](https://github.com/LuChengTHU/dpm-solver).

[+2]: [https://github.com/ermongroup/ddim](https://github.com/ermongroup/ddim)
