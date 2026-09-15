---
title: 'Categorical Reparameterization with Gumbel-Softmax'
createTime: 2026/09/15 11:11:42
permalink: /papers/gumbel-softmax/
---

> [Eric Jang](https://evjang.com/), [Shixiang Gu](https://sites.google.com/view/gugurus/home) [+internship] 和 [Ben Poole](https://cs.stanford.edu/~poole/) [+internship]. 论文于 2016 年 11 月 3 日首次提交至 arXiv; 当前版本为 v5. 作为会议论文发表于 ICLR 2017. [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144v5). <a href="/paper/gumbel-softmax.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [TeX 源码](https://export.arxiv.org/e-print/1611.01144v5). 精确的印刷版式和参考文献以原论文 PDF 为准.

[+internship]: 这项工作在 Google Brain 实习期间完成.

## 摘要

类别变量很适合表示现实世界中的离散结构. 然而, 随机神经网络很少使用类别潜变量, 因为无法通过采样操作进行反向传播. 本文提出一种高效的梯度估计器: 它用新型 Gumbel-Softmax 分布中的可微样本, 代替类别分布中的不可微样本. 这种分布有一个重要性质, 即可以平滑地退火为类别分布. 我们证明, 在带类别潜变量的结构化输出预测和无监督生成建模任务中, Gumbel-Softmax 估计器优于已有的先进梯度估计器, 并且能显著加速半监督分类.

<span id="section-1"></span>

## 1 引言

含离散随机变量的随机神经网络是一种强大的技术, 可以表示无监督学习, 语言建模, 注意力机制和强化学习中遇到的分布. 例如, 离散变量已被用于学习概率潜表示, 分别对应不同的语义类别 [Kin14a], 图像区域 [Xu15] 和记忆位置 [Gra14a, Gra16]. 与连续表示相比, 离散表示通常更容易解释 [Che16k], 计算效率也更高 [Rae16].

然而, 含离散变量的随机网络很难训练, 因为反向传播算法虽然可以高效计算参数梯度, 却无法用于不可微层. 以往的随机梯度估计工作通常集中在两类方法上: 一类是加入 Monte Carlo 方差缩减技术的得分函数估计器 [Pai12, Mni14, Gu16, Gre13], 另一类是用于 Bernoulli 变量的有偏路径导数估计器 [Ben13]. 但是, 还没有专门为类别变量设计的梯度估计器. 本文有三项贡献:

1. 我们提出 Gumbel-Softmax, 这是一种定义在单纯形上的连续分布, 可以近似类别样本, 并且能够通过重参数化技巧方便地计算参数梯度.
2. 实验表明, 对 Bernoulli 变量和类别变量而言, Gumbel-Softmax 都优于所有单样本梯度估计器.
3. 我们证明, 这种估计器可以高效训练半监督模型 (例如 [Kin14a]), 无需对未观测的类别潜变量进行代价高昂的边缘化.

本文的实际成果是一种简单, 可微的类别变量近似采样机制, 可以集成到神经网络中, 并用标准反向传播训练.

<span id="section-2"></span>

## 2 Gumbel-Softmax 分布

先定义 Gumbel-Softmax 分布, 这是一种定义在单纯形上的连续分布, 可以近似类别分布中的样本. 设 $z$ 是类别变量, 各类别概率为 $\pi_1,\pi_2,...\pi_k$. 本文余下部分假设类别样本编码为 $k$ 维 one-hot 向量, 位于 $(k-1)$ 维单纯形 $\Delta^{k-1}$ 的顶点. 这样便可以定义这些向量的逐元素均值等量, 例如 $\mathbb{E}_p[z]=\left[\pi_1,...,\pi_k\right]$.

Gumbel-Max 技巧 [Gum54, Mad14] 提供了一种简单高效的方法, 可从类别概率为 $\pi$ 的类别分布中抽取样本 $z$:

<span id="equation-01"></span>

$$
z=\mathrm{one\_hot}\left(\argmax_i\left[g_i+\log\pi_i\right]\right)
$$

其中 $g_1...g_k$ 是从 $\mathrm{Gumbel}(0,1)$ 中抽取的 i.i.d 样本 [+1]. 我们用 softmax 函数作为 $\argmax$ 的连续可微近似, 生成 $k$ 维样本向量 $y\in\Delta^{k-1}$, 其中

<span id="equation-02"></span>

$$
y_i=\frac{\exp((\log(\pi_i)+g_i)/\tau)}{\sum_{j=1}^k\exp((\log(\pi_j)+g_j)/\tau)}\qquad\mathrm{for}\ i=1,...,k.
$$

[+1]: 可以用逆变换采样从 $\mathrm{Gumbel}(0,1)$ 分布中抽样: 先抽取 $u\sim\mathrm{Uniform}(0,1)$, 再计算 $g=-\log(-\log(u))$.

Gumbel-Softmax 分布的密度为 (推导见[第 7 节](#section-7)):

<span id="equation-03"></span>

$$
p_{\pi,\tau}(y_1,...,y_k)=\Gamma(k)\tau^{k-1}\left(\sum_{i=1}^k\pi_i/y_i^\tau\right)^{-k}\prod_{i=1}^k\left(\pi_i/y_i^{\tau+1}\right)
$$

[Mad16] 独立发现了这一分布, 并称之为 concrete 分布. 当 softmax 温度 $\tau$ 趋近于 $0$ 时, Gumbel-Softmax 分布的样本变成 one-hot 向量, Gumbel-Softmax 分布也变得与类别分布 $p(z)$ 相同.

<span id="figure-01"></span>

![不同温度下 Gumbel-Softmax 的期望与样本](./gumbel-softmax/figure-01.png)

**图 1.** Gumbel-Softmax 分布在离散的 one-hot 编码类别分布与连续类别密度之间插值. (a) 温度较低时 ($\tau=0.1,\tau=0.5$), Gumbel-Softmax 随机变量的期望趋近于具有相同 logits 的类别随机变量的期望. 随温度升高 ($\tau=1.0,\tau=10.0$), 期望收敛到类别上的均匀分布. (b) 当 $\tau\to0$ 时, Gumbel-Softmax 分布的样本与类别分布的样本相同. 温度较高时, Gumbel-Softmax 样本不再是 one-hot 向量; 当 $\tau\to\infty$ 时, 样本趋于均匀.

<span id="section-2-1"></span>

### 2.1 Gumbel-Softmax 估计器

当 $\tau>0$ 时, Gumbel-Softmax 分布是平滑的, 因此关于参数 $\pi$ 存在定义良好的梯度 $\partial y/\partial\pi$. 所以, 用 Gumbel-Softmax 样本代替类别样本后, 就能通过反向传播计算梯度 (见[第 3.1 节](#section-3-1)). 我们把训练期间用可微近似代替不可微类别样本的过程称为 Gumbel-Softmax 估计器.

Gumbel-Softmax 样本虽然可微, 但温度非零时并不等同于相应类别分布的样本. 在学习过程中, 小温度与大温度之间存在权衡: 温度较小时, 样本接近 one-hot, 但梯度方差较大; 温度较大时, 样本平滑, 但梯度方差较小 ([图 1](#figure-01)). 实践中, 我们从较高温度开始, 再退火到较小但非零的温度.

实验发现, softmax 温度 $\tau$ 按多种调度方式退火都能取得良好效果. 如果 $\tau$ 是学习得到的参数 (而不是按固定调度退火), 这一方案可解释为熵正则化 [Sze16, Per16]; 此时, Gumbel-Softmax 分布能够在训练过程中自适应调节候选样本的"置信度".

<span id="section-2-2"></span>

### 2.2 Straight-Through Gumbel-Softmax 估计器

one-hot 向量的连续松弛适用于学习隐藏表示, 序列建模等问题. 某些场景要求我们抽取离散值 (例如强化学习中的离散动作空间或量化压缩). 此时, 我们用 $\argmax$ 将 $y$ 离散化, 但在反向传播中使用连续近似, 即以 $\nabla_\theta z\approx\nabla_\theta y$ 近似梯度. 由于这种做法与 [Ben13] 中的有偏路径导数估计器相似, 我们称之为 Straight-Through (ST) Gumbel 估计器. 即使温度 $\tau$ 很高, ST Gumbel-Softmax 也能产生稀疏样本.

<span id="section-3"></span>

## 3 相关工作

本节回顾现有的离散变量随机梯度估计技术 (如[图 2](#figure-02) 所示). 考虑一个含离散随机变量 $z$ 的随机计算图 [Sch15a], 其分布依赖参数 $\theta$, 代价函数为 $f(z)$. 目标是通过梯度下降最小化期望代价 $L(\theta)=\mathbb{E}_{z\sim p_\theta(z)}[f(z)]$, 因而需要估计 $\nabla_\theta\mathbb{E}_{z\sim p_\theta(z)}[f(z)]$.

<span id="figure-02"></span>

![随机计算图中的五种梯度估计方法](./gumbel-softmax/figure-02.png)

**图 2.** 随机计算图中的梯度估计. (1) 如果 $x(\theta)$ 是确定且可微的, 就能通过反向传播计算 $\nabla_\theta f(x)$. (2) 随机节点 $z$ 的存在使反向传播无法进行, 因为采样器函数没有定义良好的梯度. (3) 得分函数估计器及其变体 (NVIL, DARN, MuProp, VIMCO) 沿代理损失 $\hat{f}\log p_\theta(z)$ 反向传播, 得到 $\nabla_\theta f(x)$ 的无偏估计, 其中 $\hat{f}=f(x)-b$, $b$ 是用于降低方差的基线. (4) Straight-Through 估计器主要针对 Bernoulli 变量提出, 它近似取 $\nabla_\theta z\approx1$. (5) Gumbel-Softmax 是连续分布 $y$ 的路径导数估计器, 而 $y$ 近似 $z$. 重参数化使梯度能够从 $f(y)$ 流向 $\theta$. 在训练过程中, $y$ 可以退火成 one-hot 类别变量.

<span id="section-3-1"></span>

### 3.1 路径导数梯度估计器

对于可以重参数化的分布, 样本 $z$ 可以写成参数 $\theta$ 和独立随机变量 $\epsilon$ 的确定函数 $g$, 即 $z=g(\theta,\epsilon)$. 这样便能计算从 $f$ 到 $\theta$ 的路径梯度, 而不会遇到随机节点:

<span id="equation-04"></span>

$$
\frac{\partial}{\partial\theta}\mathbb{E}_{z\sim p_\theta}\left[f(z))\right]=\frac{\partial}{\partial\theta}\mathbb{E}_\epsilon\left[f(g(\theta,\epsilon))\right]=\mathbb{E}_{\epsilon\sim p_\epsilon}\left[\frac{\partial f}{\partial g}\frac{\partial g}{\partial\theta}\right]
$$

例如, 正态分布 $z\sim\mathcal{N}(\mu,\sigma)$ 可以改写为 $\mu+\sigma\cdot\mathcal{N}(0,1)$, 从而很容易计算 $\partial z/\partial\mu$ 和 $\partial z/\partial\sigma$. 这种重参数化技巧常用于通过反向传播训练含连续潜变量的变分自编码器 [Kin14, Rez14]. 如[图 2](#figure-02) 所示, 我们在构造 Gumbel-Softmax 估计器时使用了这一技巧.

即使 $z$ 无法重参数化, 仍可使用有偏路径导数估计器. 一般而言, 可以用 $\nabla_\theta m(\theta)$ 近似 $\nabla_\theta z$, 其中 $m$ 是随机样本的可微代理. 对均值参数为 $\theta$ 的 Bernoulli 变量, Straight-Through (ST) 估计器 [Ben13] 以 $m=\mu_\theta(z)$ 近似, 因而 $\nabla_\theta m=1$. 当 $k=2$ (Bernoulli) 时, ST Gumbel-Softmax 与 [Chu16] 提出的斜率退火 Straight-Through 估计器相似, 但前者用 softmax 而非硬 sigmoid 确定斜率. [Rol16] 考虑了另一种方法, 让每个二值潜变量参数化一个连续混合模型. 通过连续变量进行反向传播, 并对二值变量边缘化, 即可得到重参数化梯度.

ST 估计器的一个局限是, 相对于与样本无关的均值进行反向传播, 可能导致前向传播与反向传播不一致, 产生更高的方差. Gumbel-Softmax 不会出现这一问题, 因为每个样本 $y$ 都是相应离散样本 $z$ 的可微代理.

<span id="section-3-2"></span>

### 3.2 基于得分函数的梯度估计器

得分函数估计器 (SF, 也称 REINFORCE [Wil92] 和似然比估计器 [Gly90]) 利用恒等式 $\nabla_\theta p_\theta(z)=p_\theta(z)\nabla_\theta\log p_\theta(z)$ 推导出以下无偏估计器:

<span id="equation-05"></span>

$$
\nabla_\theta\mathbb{E}_z\left[f(z)\right]=\mathbb{E}_z\left[f(z)\nabla_\theta\log p_\theta(z)\right]
$$

SF 只要求 $p_\theta(z)$ 关于 $\theta$ 连续, 不需要通过 $f$ 或样本 $z$ 进行反向传播. 然而, SF 方差很高, 因此收敛缓慢. 具体来说, SF 的方差随样本向量的维数线性增长 [Rez14], 所以特别难用于类别分布.

从学习信号 $f$ 中减去控制变量 $b(z)$, 再加回其解析期望 $\mu_b=\mathbb{E}_z\left[b(z)\nabla_\theta\log p_\theta(z)\right]$, 可以降低得分函数估计器的方差并保持无偏:

<span id="equation-06"></span>
<span id="equation-07"></span>

$$
\begin{aligned}
\nabla_\theta\mathbb{E}_z\left[f(z)\right]&=\mathbb{E}_z\left[f(z)\nabla_\theta\log p_\theta(z)+(b(z)\nabla_\theta\log p_\theta(z)-b(z)\nabla_\theta\log p_\theta(z))\right]\\
&=\mathbb{E}_z\left[(f(z)-b(z))\nabla_\theta\log p_\theta(z)\right]+\mu_b
\end{aligned}
$$

下面简要介绍几种使用控制变量的近期随机梯度估计器. 更详细的说明见 [Gu16].

- NVIL [Mni14] 使用两条基线: (1) 用 $f$ 的移动平均 $\bar{f}$ 对学习信号居中; (2) 用单层神经网络拟合 $f-\bar{f}$, 得到依赖输入的基线 (即居中学习信号本身的控制变量). 最后, 方差归一化用 $\max(1,\sigma_f)$ 除以学习信号, 其中 $\sigma_f^2$ 是 $\mathrm{Var}[f]$ 的移动平均.
- DARN [Gre13] 使用 $b=f(\bar{z})+f^\prime(\bar{z})(z-\bar{z})$, 其中基线是 $f(z)$ 在 $f(\bar{z})$ 处的一阶 Taylor 近似. 对 Bernoulli 变量, $z$ 取 $\frac{1}{2}$; 由于估计器表达式忽略了修正项 $\mu_b$, 当 $f$ 不是二次函数时, 该估计器存在偏差.
- MuProp [Gu16] 也把基线建模为一阶 Taylor 展开: $b=f(\bar{z})+f^\prime(\bar{z})(z-\bar{z})$, 且 $\mu_b=f^\prime(\bar{z})\nabla_\theta\mathbb{E}_z\left[z\right]$. 为了绕过离散采样的反向传播, 计算基线和相关梯度时, 用均值场近似 $f_{\mathrm{MF}}(\mu_\theta(z))$ 代替 $f(z)$.
- VIMCO [Mni16] 是用于多样本目标的梯度估计器, 它使用其他样本的均值 $b=\frac{1}{m}\sum_{j\neq i}f(z_j)$, 为每个样本 $z_i\in z_{1:m}$ 构造基线. 我们的实验比较单样本目标的估计器, 因此不包含 VIMCO; 不过, Gumbel-Softmax 很容易扩展到多样本目标.

<span id="section-3-3"></span>

### 3.3 半监督生成模型

半监督学习同时从标注数据 $(x,y)\sim\mathcal{D}_L$ 和未标注数据 $x\sim\mathcal{D}_U$ 中学习, 其中 $x$ 是观测 (即图像), $y$ 是相应标签 (例如语义类别). 对于半监督分类, [Kin14a] 提出一种变分自编码器 (VAE), 其潜状态是 Gaussian "风格"变量 $z$ 与类别"语义类别"变量 $y$ 的联合分布 ([图 6](#figure-06), [第 6 节](#section-6)). VAE 目标通过最大化生成模型下观测对数似然的变分下界, 端到端训练判别网络 $q_\phi(y|x)$, 推断网络 $q_\phi(z|x,y)$ 和生成网络 $p_\theta(x|y,z)$. 对标注数据, 类别 $y$ 已知, 因此只需对 $z\sim q(z|x,y)$ 进行推断. 标注数据的变分下界为:

<span id="equation-08"></span>

$$
\log p_\theta(x,y)\geq-\mathcal{L}(x,y)=\mathbb{E}_{z\sim q_\phi(z|x,y)}\left[\log p_\theta(x|y,z)\right]-\mathrm{KL}\left[q(z|x,y)\|p_\theta(y)p(z)\right]
$$

对未标注数据, 类别分布无法重参数化, 因而产生困难. [Kin14a] 的处理方法是对 $y$ 的所有类别边缘化; 这样对于每个 $y$, 未标注数据仍然在 $q_\phi(z|x,y)$ 上进行推断. 未标注数据的下界为:

<span id="equation-09"></span>
<span id="equation-10"></span>

$$
\begin{aligned}
\log p_\theta(x)\geq-\mathcal{U}(x)&=\mathbb{E}_{z\sim q_\phi(y,z|x)}\left[\log p_\theta(x|y,z)+\log p_\theta(y)+\log p(z)-q_\phi(y,z|x)\right]\\
&=\sum_y q_\phi(y|x)\left(-\mathcal{L}(x,y)+\mathcal{H}(q_\phi(y|x))\right)
\end{aligned}
$$

完整的最大化目标为:

<span id="equation-11"></span>

$$
\mathcal{J}=\mathbb{E}_{(x,y)\sim\mathcal{D}_L}\left[-\mathcal{L}(x,y)\right]+\mathbb{E}_{x\sim\mathcal{D}_U}\left[-\mathcal{U}(x)\right]+\alpha\cdot\mathbb{E}_{(x,y)\sim\mathcal{D}_L}\left[\log q_\phi(y|x)\right]
$$

其中 $\alpha$ 是生成目标与判别目标之间的标量权衡系数.

这种方法的一个局限是, 当模型的类别数很多时, 对全部 $k$ 个类别值进行边缘化的计算成本会高到无法承受. 如果 $D,I,G$ 分别是从 $q_\phi(y|x)$, $q_\phi(z|x,y)$ 和 $p_\theta(x|y,z)$ 采样的计算成本, 那么每个前向/反向步骤训练无监督目标都需要 $\mathcal{O}(D+k(I+G))$ 的成本. 相比之下, Gumbel-Softmax 能够通过 $y\sim q_\phi(y|x)$ 进行反向传播, 得到单样本梯度估计, 每个训练步骤的成本为 $\mathcal{O}(D+I+G)$. 训练速度的实验比较见[图 5](#figure-05).

<span id="section-4"></span>

## 4 实验结果

第一组实验将 Gumbel-Softmax 和 ST Gumbel-Softmax 与其他随机梯度估计器比较, 包括 Score-Function (SF), DARN, MuProp, Straight-Through (ST) 和 Slope-Annealed ST. 每种估计器在两项任务上评估: (1) 结构化输出预测; (2) 生成模型的变分训练. 训练和评估使用固定二值化的 MNIST 数据集, 这是评估随机梯度估计器的常见做法 [Sal08, Lar11].

学习率从 $\{3\mathrm{e}{-5},1\mathrm{e}{-5},3\mathrm{e}{-4},1\mathrm{e}{-4},3\mathrm{e}{-3},1\mathrm{e}{-3}\}$ 中选择; 我们在 MNIST 验证集上为每种估计器选择最佳学习率, 并报告测试集性能. Gumbel-Softmax 分布产生的样本在训练期间是连续的, 但评估时会离散化为 one-hot 向量. 我们还发现, SF, DARN 和 MuProp 必须使用方差归一化才能获得有竞争力的性能. 对二值 (Bernoulli) 神经网络使用 sigmoid 激活函数, 对类别变量使用 softmax 激活函数. 模型使用带 $0.9$ 动量的随机梯度下降训练.

<span id="section-4-1"></span>

### 4.1 随机二值网络的结构化输出预测

结构化输出预测的目标是: 给定 $28\times28$ MNIST 数字图像的上半部分 ($14\times28$), 预测其下半部分. 这是训练随机二值网络 (SBN) 的常用基准 [Rai14, Gu16, Mni16]. 该条件生成模型的最小化目标是似然目标的重要性采样估计, 即 $\mathbb{E}_{h\sim p_\theta(h_i|x_{\mathrm{upper}})}\left[\frac{1}{m}\sum_{i=1}^m\log p_\theta(x_{\mathrm{lower}}|h_i)\right]$, 其中训练时使用 $m=1$, 评估时使用 $m=1000$.

我们训练了一个含两个隐藏层的 SBN, 每层 200 个单元. 它对应 200 个 Bernoulli 变量 (记为 $392$-$200$-$200$-$392$), 或者 20 个类别变量 (每个变量含 10 个类别) 和二值化激活 (记为 $392$-$(20\times10)$-$(20\times10)$-$392$).

如[图 3](#figure-03) 所示, 对 Bernoulli 变量, ST Gumbel-Softmax 与其他估计器相当; 对类别变量, 它优于其他估计器. 同时, 对 Bernoulli 变量和类别变量, Gumbel-Softmax 都优于其他估计器. 我们发现这项任务无需对 softmax 温度退火, 因而使用固定的 $\tau=1$.

<span id="figure-03"></span>

![Bernoulli 和类别随机二值网络的测试损失](./gumbel-softmax/figure-03.png)

**图 3.** 使用随机二值网络在二值化 MNIST 结构化输出预测任务上的测试损失 (负对数似然), 其中 (a) 使用 Bernoulli 潜变量 ($392$-$200$-$200$-$392$), (b) 使用类别潜变量 ($392$-$(20\times10)$-$(20\times10)$-$392$).

<span id="section-4-2"></span>

### 4.2 用变分自编码器进行生成建模

我们训练变分自编码器 [Kin14], 目标是学习二值 MNIST 图像的生成模型. 实验中, 潜变量建模为一个隐藏层, 包含 200 个 Bernoulli 变量或 20 个类别变量 ($20\times10$). 训练目标使用学习得到的类别先验, 而不是 Gumbel-Softmax 先验. 因此, 如果样本不是离散的, 训练期间的最小化目标就不再是变分界. 实践中发现, 将这一目标与温度退火结合优化, 仍可最小化验证集和测试集上的实际变分界. 与结构化输出预测任务相同, 评估使用 $m=1000$ 的多样本界.

温度按照全局训练步数 $t$ 的调度 $\tau=\max(0.5,\exp(-rt))$ 退火, 每 $N$ 步更新一次 $\tau$. $N\in\{500,1000\}$ 和 $r\in\{1\mathrm{e}{-5},1\mathrm{e}{-4}\}$ 是超参数; 我们在验证集上选择表现最佳的估计器, 并报告其测试性能.

如[图 4](#figure-04) 所示, 对类别变量, ST Gumbel-Softmax 优于其他估计器; 对 Bernoulli 变量和类别变量, Gumbel-Softmax 都大幅优于其他估计器.

<span id="figure-04"></span>

![Bernoulli 和类别变分自编码器的测试损失](./gumbel-softmax/figure-04.png)

**图 4.** 二值化 MNIST VAE 上的测试损失 (负变分下界), 其中 (a) 使用 Bernoulli 潜变量 ($784$-$200$-$784$), (b) 使用类别潜变量 ($784$-$(20\times10)$-$200$).

<span id="table-01"></span>

![七种梯度估计器在 SBN 和 VAE 上的损失](./gumbel-softmax/table-01.png)

**表 1.** 对 Bernoulli 和类别潜变量, Gumbel-Softmax 估计器优于其他估计器. 在结构化输出预测 (SBN) 任务中, 数值表示输入图像的负对数似然 (nat, 越低越好). 在 VAE 任务中, 数值表示对数似然的负变分下界 (nat, 越低越好).

<span id="section-4-3"></span>

### 4.3 生成式半监督分类

我们将 Gumbel-Softmax 估计器用于二值 MNIST 数据集上的半监督分类. 实验比较原始的基于边缘化的推断方法 [Kin14a], 以及使用 Gumbel-Softmax 和 ST Gumbel-Softmax 的单样本推断.

训练数据集包含 100 个标注样本 (均匀分布在 10 个类别中) 和 50,000 个未标注样本; 每个 minibatch 都对未标注样本动态二值化. 判别模型 $q_\phi(y|x)$ 和推断模型 $q_\phi(z|x,y)$ 都实现为带 ReLU 激活函数的 3 层卷积神经网络. 生成模型 $p_\theta(x|y,z)$ 是带 ReLU 激活的 4 层转置卷积网络. 实验细节见[第 6 节](#section-6).

我们使用多个 $\alpha=\{0.1,0.2,0.3,0.8,1.0\}$ 值训练和评估各估计器, 并为每种估计器选择测试集上最佳的未标注分类结果, 报告于[表 2](#table-02). 退火调度为 $\tau=\max(0.5,\exp(-3\mathrm{e}{-5}\cdot t))$, 每 2000 步更新一次.

在 [Kin14a] 中, 对潜状态的推断通过边缘化 $y$, 并用重参数化技巧从 $q_\phi(z|x,y)$ 采样来完成. 然而, 这种方法的计算成本随类别数线性增长. Gumbel-Softmax 允许我们直接通过联合分布 $q_\phi(y,z|x)$ 的单个样本反向传播, 在不损害生成性能或分类性能的情况下大幅加快训练. (见[表 2](#table-02) 和[图 5](#figure-05)).

<span id="table-02"></span>

![边缘化与 Gumbel-Softmax 估计器的 ELBO 和准确率](./gumbel-softmax/table-02.png)

**表 2.** 在二值化 MNIST 数据集的图像分类任务上, 对 $y$ 边缘化与单样本变分推断表现相当 [Lar11]. 我们报告测试集中未标注数据的变分下界和图像分类准确率.

[图 5](#figure-05) 展示了 Gumbel-Softmax 与边缘化方法如何随类别数扩展. 这些实验使用带随机生成标签的 MNIST 图像. 类别数为 $10$ 时, 使用 Gumbel-Softmax 估计器训练模型的速度为原来的 $2\times$; 类别数为 $100$ 时, 速度为原来的 $9.9\times$.

<span id="figure-05"></span>

![半监督 VAE 的训练速度与 MNIST 类比结果](./gumbel-softmax/figure-05.png)

**图 5.** Gumbel-Softmax 允许我们通过后验 $q_\phi(y|x)$ 的样本进行反向传播, 为类别数很多的半监督学习任务提供了一种可扩展方法. (a) 在半监督 VAE 上比较 Gumbel-Softmax 与边缘化方法 [Kin14a] 的训练速度 (steps/sec). 评估在 GTX Titan X® GPU 上完成. (b) MNIST 类比的可视化结果, 每行改变风格变量 $z$, 每列改变类别变量 $y$.

<span id="section-5"></span>

## 5 讨论

本文的主要贡献是可重参数化的 Gumbel-Softmax 分布, 其相应估计器可以为类别分布提供低方差的路径导数梯度. 我们证明, Gumbel-Softmax 和 Straight-Through Gumbel-Softmax 在结构化输出预测与变分自编码器任务中都很有效, 对 Bernoulli 潜变量和类别潜变量的表现均优于现有随机梯度估计器. 最后, Gumbel-Softmax 能够大幅加快离散潜变量的推断.

## 致谢

衷心感谢 Luke Vilnis, Vincent Vanhoucke, Luke Metz, David Ha, Laurent Dinh, George Tucker 和 Subhaneil Lahiri 提供的宝贵讨论与反馈.

<span id="section-6"></span>

## 6 半监督分类模型

[图 6](#figure-06) 和[图 7](#figure-07) 描述了半监督分类实验使用的架构 ([第 4.3 节](#section-4-3)).

<span id="figure-06"></span>

![半监督生成与推断计算图](./gumbel-softmax/figure-06.png)

**图 6.** [Kin14a] 提出的半监督生成模型. (a) 生成模型 $p_\theta(x|y,z)$ 根据 Gaussian "风格"潜变量 $z$ 和类别变量 $y$ 合成图像. (b) 推断模型 $q_\phi(y,z|x)$ 在给定 $x$ 时抽取潜状态 $y,z$. Gaussian 变量 $z$ 可以对其参数求导, 因为它能够重参数化. 在以往工作中, 未观测到 $y$ 时, 训练 VAE 目标需要对 $y$ 的所有取值边缘化. (c) Gumbel-Softmax 对 $y$ 重参数化, 使反向传播也能通过 $y$, 而不会遇到随机节点.

<span id="figure-07"></span>

![用于分类, 推断与生成的卷积架构](./gumbel-softmax/figure-07.png)

**图 7.** 用于 (a) 分类 $q_\phi(y|x)$, (b) 推断 $q_\phi(z|x,y)$ 和 (c) 生成 $p_\theta(x|y,z)$ 的网络架构. 这些网络的输出分别参数化类别分布, Gaussian 分布和 Bernoulli 分布, 我们从这些分布中采样.

<span id="section-7"></span>

## 7 推导 Gumbel-Softmax 分布的密度

这里推导概率为 $\pi_1,...,\pi_k$, 温度为 $\tau$ 的 Gumbel-Softmax 分布的概率密度函数. 先定义 logits $x_i=\log\pi_i$ 和 Gumbel 样本 $g_1,...,g_k$, 其中 $g_i\sim\mathrm{Gumbel}(0,1)$. Gumbel-Softmax 样本可以按下式计算:

<span id="equation-12"></span>

$$
y_i=\frac{\exp((x_i+g_i)/\tau)}{\sum_{j=1}^k\exp((x_j+g_j)/\tau)}\qquad\mathrm{for}\ i=1,...,k
$$

<span id="section-7-1"></span>

### 7.1 中心化 Gumbel 密度

从 Gumbel 样本 $g$ 到 Gumbel-Softmax 样本 $y$ 的映射不可逆, 因为 softmax 操作的归一化会消去一个自由度. 为了补偿这一点, 我们定义一个等价采样过程: 在执行 softmax 之前减去最后一个元素 $(x_k+g_k)/\tau$:

<span id="equation-13"></span>

$$
y_i=\frac{\exp((x_i+g_i-(x_k+g_k))/\tau)}{\sum_{j=1}^k\exp((x_j+g_j-(x_k+g_k))/\tau)}\qquad\mathrm{for}\ i=1,...,k
$$

为了推导这个等价采样过程的密度, 先推导以下"中心化"多元 Gumbel 密度:

<span id="equation-14"></span>

$$
u_i=x_i+g_i-(x_k+g_k)\qquad\mathrm{for}\ i=1,...,k-1
$$

其中 $g_i\sim\mathrm{Gumbel}(0,1)$. 注意, 尺度参数 $\beta=1$, 均值为 $\mu$ 的 Gumbel 分布在 $z$ 处的概率密度为 $f(z,\mu)=e^{\mu-z-e^{\mu-z}}$. 现在可以对最后一个 Gumbel 样本 $g_k$ 边缘化, 计算该分布的密度:

$$
\begin{aligned}
p(u_1,...,u_{k-1})&=\int_{-\infty}^\infty dg_k\,p(u_1,...,u_k|g_k)p(g_k)\\
&=\int_{-\infty}^\infty dg_k\,p(g_k)\prod_{i=1}^{k-1}p(u_i|g_k)\\
&=\int_{-\infty}^\infty dg_k\,f(g_k,0)\prod_{i=1}^{k-1}f(x_k+g_k,x_i-u_i)\\
&=\int_{-\infty}^\infty dg_k\,e^{-g_k-e^{-g_k}}\prod_{i=1}^{k-1}e^{x_i-u_i-x_k-g_k-e^{x_i-u_i-x_k-g_k}}
\end{aligned}
$$

令 $v=e^{-g_k}$ 进行变量替换, 则 $dv=-e^{-g_k}dg_k$, 且 $dg_k=-dv\,e^{g_k}=dv/v$; 再定义 $u_k=0$ 以简化记号:

<span id="equation-15"></span>
<span id="equation-16"></span>
<span id="equation-17"></span>
<span id="equation-18"></span>

$$
\begin{aligned}
p(u_1,...,u_{k,-1})&=\delta(u_k=0)\int_0^\infty dv\,\frac{1}{v}ve^{x_k-v}\prod_{i=1}^{k-1}ve^{x_i-u_i-x_k-ve^{x_i-u_i-x_k}}\\
&=\exp\left(x_k+\sum_{i=1}^{k-1}(x_i-u_i)\right)\left(e^{x_k}+\sum_{i=1}^{k-1}e^{x_i-u_i}\right)^{-k}\Gamma(k)\\
&=\Gamma(k)\exp\left(\sum_{i=1}^k(x_i-u_i)\right)\left(\sum_{i=1}^ke^{x_i-u_i}\right)^{-k}\\
&=\Gamma(k)\left(\prod_{i=1}^k\exp(x_i-u_i)\right)\left(\sum_{i=1}^k\exp(x_i-u_i)\right)^{-k}
\end{aligned}
$$

<span id="section-7-2"></span>

### 7.2 变换为 Gumbel-Softmax

给定中心化 Gumbel 分布的样本 $u_1,...,u_{k,-1}$, 可以应用确定性变换 $h$, 得到 Gumbel-Softmax 样本的前 $k-1$ 个坐标:

<span id="equation-19"></span>

$$
y_{1:k-1}=h(u_{1:k-1}),\qquad h_i(u_{1:k-1})=\frac{\exp(u_i/\tau)}{1+\sum_{j=1}^{k-1}\exp(u_j/\tau)}\quad\forall i=1,...,k-1
$$

注意, 给定前 $k-1$ 个坐标后, 最后一个坐标的概率 $y_k$ 也随之确定, 因为 $\sum_{i=1}^k y_i=1$:

<span id="equation-20"></span>

$$
y_k=\left(1+\sum_{j=1}^{k-1}\exp(u_j/\tau)\right)^{-1}=1-\sum_{j=1}^{k-1}y_j
$$

因此, 只需对前 $k-1$ 个变量应用变量替换公式, 即可计算 Gumbel-Softmax 样本的概率:

<span id="equation-21"></span>

$$
p(y_{1:k})=p\left(h^{-1}(y_{1:k-1})\right)\det\left(\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}\right)
$$

所以还需要计算两部分: $h$ 的逆和它的 Jacobian 行列式. $h$ 的逆为:

<span id="equation-22"></span>

$$
h^{-1}(y_{1:k-1})=\tau\times\left(\log y_i-\log\left(1-\sum_{j=1}^{k-1}y_j\right)\right)=\tau\times(\log y_i-\log y_k)
$$

其 Jacobian 为

<span id="equation-23"></span>

$$
\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}=\tau\times\left(\mathrm{diag}\left(\frac{1}{y_{1:k-1}}\right)+\frac{1}{y_k}\right)=
\begin{bmatrix}
\frac{1}{y_1}+\frac{1}{y_k}&\frac{1}{y_k}&\dots&\frac{1}{y_k}\\
\frac{1}{y_k}&\frac{1}{y_2}+\frac{1}{y_k}&\dots&\frac{1}{y_k}\\
\vdots&\vdots&\ddots&\vdots\\
\frac{1}{y_k}&\frac{1}{y_k}&\dots&\frac{1}{y_{k-1}}+\frac{1}{y_k}
\end{bmatrix}
$$

接着计算 Jacobian 的行列式:

<span id="equation-24"></span>
<span id="equation-25"></span>
<span id="equation-26"></span>

$$
\begin{aligned}
\det\left(\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}\right)&=\tau^{k-1}\det\left(\left(I+\frac{1}{y_k}ee^\top\mathrm{diag}(y_{1:k-1})\right)\mathrm{diag}\left(\frac{1}{y_{1:k-1}}\right)\right)\\
&=\tau^{k-1}\left(1+\frac{1-y_k}{y_k}\right)\prod_{j=1}^{k-1}y_j^{-1}\\
&=\tau^{k-1}\prod_{j=1}^ky_j^{-1}
\end{aligned}
$$

其中 $e$ 是 $k-1$ 维全一向量, 我们使用了以下恒等式: $\det(A\,B)=\det(A)\det(B)$, $\det(\mathrm{diag}(x))=\prod_i x_i$, 以及 $\det(I+uv^\top)=1+u^\top v$.

将中心化 Gumbel 的密度 ([公式 15](#equation-15)), $h$ 的逆 ([公式 22](#equation-22)) 和它的 Jacobian 行列式 ([公式 26](#equation-26)) 代入变量替换公式 ([公式 21](#equation-21)):

<span id="equation-27"></span>
<span id="equation-28"></span>

$$
\begin{aligned}
p(y_1,..,y_k)&=\Gamma(k)\left(\prod_{i=1}^k\exp(x_i)\frac{y_k^\tau}{y_i^\tau}\right)\left(\sum_{i=1}^k\exp(x_i)\frac{y_k^\tau}{y_i^\tau}\right)^{-k}\tau^{k-1}\prod_{i=1}^ky_i^{-1}\\
&=\Gamma(k)\tau^{k-1}\left(\sum_{i=1}^k\exp(x_i)/y_i^\tau\right)^{-k}\prod_{i=1}^k\left(\exp(x_i)/y_i^{\tau+1}\right)
\end{aligned}
$$
