---
title: 'FlashAttention-3'
createTime: 2026/09/10 14:13:02
permalink: /papers/flashattention-3/
pageClass: paper-reading
---

> [Jay Shah](https://developer.nvidia.com/blog/author/jayshah/) [+equal], [Ganesh Bikshandi](https://dblp.org/pid/68/2188.html) [+equal], [Ying Zhang](https://x.com/ipiszy), [Vijay Thakkar](https://cse.gatech.edu/people/vijay-thakkar), [Pradeep Ramani](https://developer.nvidia.com/blog/author/prramani/) 和 [Tri Dao](https://tridao.me/). 论文于 2024 年 7 月 11 日首次提交至 arXiv, 当前版本为 v2, 修订于 2024 年 7 月 12 日. 发表于 [Advances in Neural Information Processing Systems 37 (NeurIPS 2024) 主会场](https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ede97c3e082c6df10a8d6103a2eebd2-Abstract-Conference.html). [FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608v2). <a href="/paper/flashattention-3.pdf" target="_blank" rel="noopener noreferrer">原始 PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2407.08608). [会议论文 DOI](https://doi.org/10.52202/079017-2193). [TeX 源码](https://export.arxiv.org/e-print/2407.08608v2). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

注意力是无处不在的 Transformer 架构中的核心层, 也是大语言模型和长上下文应用的瓶颈. FlashAttention 提出了一种通过尽量减少内存读写来加速 GPU 注意力计算的方法. 不过, 它尚未利用新近硬件具备的新能力, FlashAttention-2 在 H100 GPU 上的利用率仅为 35%. 我们提出三项主要技术来加速 Hopper GPU 上的注意力: 利用 Tensor Core 和 TMA 的异步性, (1) 通过 warp 特化重叠整体计算与数据移动, (2) 交错执行分块矩阵乘与 softmax 操作, 以及 (3) 借助硬件对 FP8 低精度的支持进行分块量化和非相干处理. 实验表明, 我们的方法 FlashAttention-3 在 H100 GPU 上可将 FP16 加速 1.5-$2.0\times$, 达到最高 740 TFLOPs/s (75% 利用率), FP8 则接近 1.2 PFLOPs/s. 我们还验证了 FP8 FlashAttention-3 的数值误差比基线 FP8 注意力低 $2.6\times$.

<span id="section-1"></span>

## 1 引言

在 Transformer 架构 [Vas17] 中, 注意力机制构成了主要的计算瓶颈, 因为计算查询与键的自注意力分数会随序列长度呈二次增长. 将注意力扩展到更长的上下文会带来新的能力, 包括对多篇长文档 [Guo21a, Sha22a, Pen23] 和大型代码库中的文件 [Roz23, Li23o] 建模并推理; 新的模态, 包括高分辨率图像 [Che22a]、音频 [Gul20] 和视频 [Ho22]; 以及新的应用, 包括具有长历史的用户交互 [Sun19d] 和长时域的智能体工作流 [Yao22b]. 因此, 如何在长上下文条件下加速注意力引起了广泛兴趣, 相关方向包括近似方法 [Kat20, Cho20a, Tay20a]、软件优化 [Rab21, Dao22, Kwo23], 乃至替代架构 [Pen23b, Sun23b, Gu23].

本文建立在 [Dao22] 的工作之上, 目标是开发精确注意力算法, 并在高层设计中融入对 GPU 执行模型和硬件特性的认识. Dao 等人在 [Dao22] 中提出 FlashAttention, 这是一种并行化注意力的新型分块策略: 它把所有注意力操作融合进单个 GPU kernel, 从而消除对低速全局内存的中间读写. [Dao23b] 将该算法重构为 FlashAttention-2, 进一步沿序列长度维度并行, 并在前向传播的内循环中按键矩阵和值矩阵的分块执行, 从而改善 GPU 的占用率和工作分配. 但我们观察到, 与优化后的矩阵乘 (GEMM) kernel 相比, FlashAttention-2 在较新的 GPU 上利用率仍然较低, 例如在 Hopper H100 GPU 上只有 35%, 而优化 GEMM 可达 80-90%. 这部分可能来自实现层面的差异, 例如以 Tensor Core 为目标时仍使用 Ampere 指令, 而没有换成 Hopper 专用指令. ThunderKittens [Res24] 和 cuDNN 9 [Nvi24g] 等工作已经表明, Hopper 专用指令与基于 tile 的抽象既能加快注意力计算, 也能简化实现.

更根本的问题在于, FlashAttention-2 的算法遵循简化的同步模型, 在设计中没有显式利用异步性和低精度. 异步性来自针对机器学习工作负载中最重要操作的硬件特化: 专用硬件单元负责矩阵乘法 (Tensor Core) 或内存加载 (Tensor Memory Accelerator, TMA), 与负责逻辑、整数和浮点计算的其余 CUDA core 分离. Hopper 中的 FP8 和 Blackwell 中的 FP4 延续了 FP16 (2017 年的 Pascal) 与 BF16 (2020 年的 Ampere) 的趋势; 在功耗和芯片面积相同的情况下, 低精度已经被证明能够带来两倍或四倍的吞吐量. 我们在[第 2.2 节](#section-2-2)回顾 Hopper 在这些方向提供的能力. 技术挑战是重新设计 FlashAttention-2 以使用这些硬件特性: 异步执行要求重叠矩阵乘与 softmax 的计算, 尽管后者依赖前者的输出; 低精度则要求尽可能减小量化误差, 尤其要处理 LLM 中的离群特征 [Det22, Sun24c].

为此, 我们提出 FlashAttention-3, 综合三项新思路来进一步提升新型 GPU 架构上的性能: [+1]

- **生产者-消费者异步执行:** 我们定义了一套 warp 特化的软件流水线方案, 把数据的生产者和消费者拆分到不同 warp, 利用数据移动与 Tensor Core 的异步执行, 从而进一步隐藏内存延迟和指令发射延迟.
- **把 softmax 隐藏在异步分块 GEMM 之下:** 我们将 softmax 中吞吐量相对较低的非 GEMM 操作, 例如浮点乘加与指数运算, 与执行 GEMM 的异步 WGMMA 指令重叠. 为此, 我们重写了 FlashAttention-2 算法, 绕开 softmax 和 GEMM 之间的一些顺序依赖. 以算法的两阶段版本为例, softmax 在一个分数矩阵分块上执行时, WGMMA 会在异步代理中计算下一个分块.
- **硬件加速的低精度 GEMM:** 我们调整前向传播算法, 使其能够以 FP8 Tensor Core 为 GEMM 目标, 实测 TFLOPs/s 几乎翻倍. 这要求弥合 WGMMA 不同的布局一致性要求, 即 FP32 累加器分块和 FP8 操作数矩阵分块在内存中的预期排列方式不同. 我们使用分块量化和非相干处理, 减轻切换到 FP8 精度造成的准确度损失.

为了通过实验验证方法, 我们在 H100 SXM5 GPU 上针对一系列参数测试 FlashAttention-3, 并表明: (1) FP16 前向传播比 FlashAttention-2 快 1.5-$2.0\times$ (最高达到 740 TFLOPs/s), 反向传播快 1.5-$1.75\times$; (2) FP8 接近 1.2 PFLOPs/s; (3) 对长序列而言, FP16 的性能超过 NVIDIA cuDNN 库中最先进的注意力实现, FP8 与之相当 [+2]. 我们还验证了 FP16 FlashAttention-3 与 FlashAttention-2 的数值误差相同, 并且优于标准注意力实现, 因为中间结果 (例如 softmax 重缩放) 保持为 FP32. 此外, 在存在离群特征时, 采用分块量化和非相干处理的 FP8 FlashAttention-3 比使用逐张量量化的标准注意力准确 $2.6\times$.

我们以宽松许可证开源 FlashAttention-3 [+3], 并计划将其集成到 PyTorch 和 Hugging Face 库中, 让尽可能多的研究人员与开发者受益.

<span id="section-2"></span>

## 2 背景: 多头注意力与 GPU 特性

<span id="section-2-1"></span>

### 2.1 多头注意力

令 $\mathbf{Q},\mathbf{K},\mathbf{V}\in\mathbb{R}^{N\times d}$ 为与单个头对应的查询、键和值输入序列, 其中 $N$ 是序列长度, $d$ 是头维度. 注意力输出 $\mathbf{O}$ 的计算方式为:

$$
\mathbf{S}=\alpha\mathbf{Q}\mathbf{K}^{\top}\in\mathbb{R}^{N\times N},\quad\mathbf{P}=\mathrm{softmax}(\mathbf{S})\in\mathbb{R}^{N\times N},\quad\mathbf{O}=\mathbf{P}\mathbf{V}\in\mathbb{R}^{N\times d},
$$

其中 $\mathrm{softmax}$ 按行应用, 通常把缩放因子设为 $\alpha=1/\sqrt{d}$. 实际上, 我们会从 $\mathbf{S}$ 中减去 $\mathrm{rowmax}(\mathbf{S})$, 防止指数函数出现数值不稳定. 对于多头注意力 (MHA), 每个头都有各自的查询、键和值投影, 这项计算会在多个头与批次之间并行, 生成完整的输出张量.

现在令 $\phi$ 为标量损失函数, 并以 $\mathbf{d}(-)=\partial\phi/\partial(-)$ 表示梯度. 给定输出梯度 $\mathbf{dO}\in\mathbb{R}^{N\times d}$, 我们按如下链式法则计算 $\mathbf{dQ}$、$\mathbf{dK}$ 和 $\mathbf{dV}$:

$$
\begin{aligned}
\mathbf{dV} & =\mathbf{P}^{\top}\mathbf{dO}\in\mathbb{R}^{N\times d} \\
\mathbf{dP} & =\mathbf{dO}\mathbf{V}^{\top}\in\mathbb{R}^{N\times N} \\
\mathbf{dS} & =\mathrm{dsoftmax}(\mathbf{dP})\in\mathbb{R}^{N\times N} \\
\mathbf{dQ} & =\alpha\mathbf{dS}\mathbf{K}\in\mathbb{R}^{N\times d} \\
\mathbf{dK} & =\alpha\mathbf{dS}^{\top}\mathbf{Q}\in\mathbb{R}^{N\times d},
\end{aligned}
$$

这里, 对作为向量 $s$ 之函数的 $p=\mathrm{softmax}(s)$, 有 $\mathbf{d}s=(\mathrm{diag}(p)-pp^{\top})\mathbf{d}p$; 我们用 $\mathrm{dsoftmax}(\mathbf{dP})$ 表示按行应用该公式. 最后, MHA 的反向传播同样会按头数和批次并行执行这项计算.

<span id="section-2-2"></span>

### 2.2 GPU 硬件特性与执行模型

我们介绍与 FlashAttention-3 有关的 GPU 执行模型, 并以 NVIDIA Hopper 架构作为这一模型的具体实例.

**内存层次结构:** GPU 内存按数据位置组织为层次结构, 容量与带宽成反比 ([表 1](#table-01)) [+4]. 全局内存 (GMEM) 也称 HBM, 是所有流式多处理器 (SM) 均可访问的片外 DRAM. 来自 GMEM 的数据会透明地缓存到片上 L2 cache. 再下一层, 每个 SM 包含一块较小的片上共享内存 (SMEM), 它由程序员管理, 并具有大量 bank. 最后一层是每个 SM 内的寄存器文件.

**线程层次结构:** GPU 编程模型围绕称为线程的逻辑执行单元组组织. 从最细到最粗, 线程层次由 thread、warp (32 个 thread)、warpgroup (4 个连续 warp)、threadblock (即 cooperative thread array, CTA)、threadblock cluster (Hopper 中) 和 grid 构成.

这两种层次结构紧密相连. 同一个 CTA 内的线程会共同调度到同一个 SM, 同一个 cluster 内的 CTA 会共同调度到同一个 GPC. CTA 内所有线程都能直接寻址 SMEM, 而每个线程最多拥有 256 个私有寄存器 (RMEM).

<span id="table-01"></span>

![表 1. NVIDIA Hopper H100 SXM5 GPU 的线程-内存层次结构.](./flashattention-3/table-01.png)

**表 1.** NVIDIA Hopper H100 SXM5 GPU 的线程-内存层次结构.

**异步性与 warp 特化:** GPU 是依赖并发和异步执行来隐藏内存与执行延迟的吞吐型处理器. 为了在 GMEM 和 SMEM 之间异步复制内存, Hopper 配备了专用硬件单元 Tensor Memory Accelerator (TMA) [Nvi24c]. 此外, Hopper 的 Tensor Core 与 Ampere 等先前架构不同, 它通过整个 warpgroup 范围的 WGMMA 指令 [Ptx24] 暴露, 同样采用异步执行, 并且可以直接从共享内存读取输入.

硬件对异步执行的支持使 warp 特化 kernel 成为可能: 一个 CTA 中的 warp 被划分为生产者或消费者角色, 分别只发射数据移动或计算操作. 一般来说, 这能增强编译器生成最优指令调度的能力 [Bau11]. 此外, Hopper 还支持通过 `setmaxnreg` 在 warpgroup 之间动态重新分配寄存器 [Ptx24], 因而执行 MMA 的 warp 可以获得比仅发射 TMA 的 warp 更大比例的 RMEM; TMA 只需要一个线程.

**低精度数值格式:** 现代 GPU 配有专门加速低精度计算的硬件单元. 例如, WGMMA 指令可以使用 Hopper 上的 FP8 Tensor Core, 与 FP16 或 BF16 相比, 每个 SM 的吞吐量可达到两倍.

不过, 要正确调用 FP8 WGMMA, 必须理解其操作数的布局约束. 给定 GEMM 调用 $A\times B^{\top}$, 其中 $A$ 为 $M\times K$ 矩阵, $B$ 为 $N\times K$ 矩阵, 如果操作数 $A$ 或 $B$ 在外层 $M$ 或 $N$ 维度连续, 我们称其为 *mn-major*; 如果改为在内层 $K$ 维度连续, 则称为 *k-major*. 对 FP16 WGMMA 而言, 位于 SMEM 的操作数既可采用 mn-major, 也可采用 k-major. FP8 WGMMA 则只支持 k-major 格式. 此外, 在注意力这类需要把前后相接的 GEMM 融合进单个 kernel 的场景中, FP32 累加器与 FP8 操作数的布局冲突, 会阻碍相互依赖的 FP8 WGMMA 调用.

在注意力场景中, 这些布局限制要求相应修改 FP8 算法设计, 详见[第 3.3 节](#section-3-3).

<span id="section-2-3"></span>

### 2.3 标准注意力与 FlashAttention

沿用 [Dao22] 的定义, 我们用**标准注意力**表示一种 GPU 注意力实现, 它会把中间矩阵 $\mathbf{S}$ 和 $\mathbf{P}$ 具体化到 HBM. FlashAttention 的核心思路是使用局部版本的 softmax 归约, 避免代价高昂的中间读写, 并把注意力融合进单个 kernel. 局部 softmax 对应[算法 1](#algorithm-01)消费者主循环的第 18-19 行, 以及对 $\mathbf{O}$ 分块的重缩放. [Dao23b] 给出了这套过程确实能够计算 $\mathbf{O}$ 的简洁推导.

<span id="section-3"></span>

## 3 FlashAttention-3 算法

本节介绍 FlashAttention-3 算法. 为简化说明, 我们重点讨论前向传播, 反向传播算法见[第 7.1 节](#section-7-1). 首先, 我们说明如何把 warp 特化和环形 SMEM 缓冲区集成到 FlashAttention-2 的基础算法中. 随后, 我们解释如何利用 WGMMA 的异步性, 定义一个重叠 GEMM-softmax 的两阶段流水线. 最后, 我们介绍 FP8 所需的修改, 包括满足布局要求, 以及通过分块量化和非相干处理提高准确度.

<span id="section-3-1"></span>

### 3.1 通过 warp 特化与乒乓调度实现生产者-消费者异步执行

**Warp 特化.** 与 FlashAttention-2 相同, FlashAttention-3 的前向传播可以直接在批大小、头数与查询序列长度三个维度上并行. 因此, 给出算法的 CTA 级视图就足够了: 它处理查询矩阵的一个 tile $\mathbf{Q}_{i}$, 计算对应的输出 tile $\mathbf{O}_{i}$. 为简化说明, 我们先给出带环形 SMEM 缓冲区、但还**没有**加入 GEMM-softmax 重叠的 warp 特化方案. 令 $d$ 为头维度, $N$ 为序列长度, 并固定查询分块大小 $B_{r}$, 将 $\mathbf{Q}$ 划分为 $T_{r}=\lceil\frac{N}{B_{r}}\rceil$ 个分块 $\mathbf{Q}_{1},..,\mathbf{Q}_{T_{r}}$.

<span id="algorithm-01"></span>

**算法 1: 不含消费者内部重叠的 FlashAttention-3 前向传播, CTA 视图.**

- **输入:** HBM 中的矩阵 $\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}$ 与 $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$, 键分块大小 $B_c$, 且 $T_c = \lceil \frac{N}{B_c} \rceil$.
- 初始化流水线对象, 用 $s$ 阶环形 SMEM 缓冲区管理屏障同步.
- **如果**位于生产者 warpgroup:
  - 释放预定数量的寄存器.
  - 发射从 HBM 到共享内存的 $\mathbf{Q}_i$ 加载.
  - 加载完成后, 提交通知消费者 $\mathbf{Q}_i$ 已加载.
  - **对于** $0 \le j < T_c$:
    - 等待缓冲区的第 $(j\,\%\,s)$ 阶被消费.
    - 发射从 HBM 到共享内存的 $\mathbf{K}_j, \mathbf{V}_j$ 加载, 目标为缓冲区的第 $(j\,\%\,s)$ 阶.
    - 加载完成后, 提交通知消费者 $\mathbf{K}_j, \mathbf{V}_j$ 已加载.
- **否则:**
  - 根据消费者 warp 数量重新分配预定数量的寄存器.
  - 在片上初始化 $\mathbf{O}_i = (0) \in \mathbb{R}^{B_r \times d}$ 和 $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
  - 等待 $\mathbf{Q}_i$ 加载到共享内存.
  - **对于** $0 \le j < T_c$:
    - 等待 $\mathbf{K}_j$ 加载到共享内存.
    - 计算 $\mathbf{S}_i^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top$ (SS-GEMM). 提交并等待.
    - 保存 $m_i^{\mathrm{old}} = m_i$, 并计算 $m_i = \max(m_i^{\mathrm{old}}, \mathrm{rowmax}(\mathbf{S}_i^{(j)}))$.
    - 计算 $\widetilde{\mathbf{P}}_i^{(j)} = \exp(\mathbf{S}_i^{(j)} - m_i)$ 和 $\ell_i = \exp(m_i^{\mathrm{old}} - m_i) \ell_i + \mathrm{rowsum}(\widetilde{\mathbf{P}}_i^{(j)})$.
    - 等待 $\mathbf{V}_j$ 加载到共享内存.
    - 计算 $\mathbf{O}_i = \mathrm{diag}(\exp(m_i^{\mathrm{old}} - m_i))^{-1} \mathbf{O}_i + \widetilde{\mathbf{P}}_i^{(j)} \mathbf{V}_j$ (RS-GEMM). 提交并等待.
    - 将缓冲区的第 $(j\,\%\,s)$ 阶释放给生产者.
  - 计算 $\mathbf{O}_i = \mathrm{diag}(\ell_i)^{-1} \mathbf{O}_i$ 和 $L_i = m_i + \log(\ell_i)$.
  - 把 $\mathbf{O}_i$ 和 $L_i$ 作为 $\mathbf{O}$ 和 $L$ 的第 $i$ 个分块写入 HBM.

在 Hopper 上实现[算法 1](#algorithm-01)时, 我们用 `setmaxnreg` 分配或释放寄存器, 用 TMA 加载 $\mathbf{Q}_{i}$ 与 $\{\mathbf{K}_{j},\mathbf{V}_{j}\}_{0\leq j<T_{c}}$, 并用 WGMMA 执行消费者主循环中的 GEMM. SS 或 RS 前缀表示第一个操作数来自共享内存还是寄存器文件. 解读[算法 1](#algorithm-01)的执行流程时需要注意, 由于采用异步执行, 发射 TMA 加载不会因等待其他加载完成而停顿. 此外, 在生产者主循环中, 缓冲区填满前的前 $s$ 次迭代不会发出等待.

**乒乓调度.** WGMMA 与 TMA 的异步性质配合 warp 特化, 使一个 warpgroup 的 softmax 计算可以和另一个 warpgroup 的 GEMM 重叠. 其依据是, 现代硬件加速器上非矩阵乘操作的吞吐量远低于矩阵乘操作. 例如, H100 SXM5 GPU 的 FP16 矩阵乘吞吐量为 989 TFLOPS, 但指数函数等特殊函数只有 3.9 TFLOPS [+5], 而 softmax 需要指数函数. 对头维度为 128 的 FP16 注意力前向传播, 矩阵乘 FLOPS 是指数运算次数的 512 倍, 但指数运算吞吐量低 256 倍, 因此指数运算消耗的周期可达到矩阵乘的 50%. FP8 的情况更差, 因为矩阵乘吞吐量翻倍, 而指数运算吞吐量不变.

由于指数运算由独立硬件单元 (multi-function unit) 执行, 理想情况是在 Tensor Core 执行矩阵乘时调度指数计算. 为此, 我们使用同步屏障 (`bar.sync` 指令), 强制 warpgroup 1 的 GEMM (某次迭代的 GEMM1, 即 $\mathbf{P}\mathbf{V}$, 以及下一次迭代的 GEMM0, 即 $\mathbf{Q}\mathbf{K}^{\top}$) 排在 warpgroup 2 的 GEMM 之前. 这样, warpgroup 1 的 softmax 会在 warpgroup 2 执行 GEMM 时调度. 随后角色交换, warpgroup 2 执行 softmax, warpgroup 1 执行 GEMM, 因而称为“乒乓”调度. [图 1](#figure-01) 给出了示意. 实际的乒乓调度没有图中那么规整, 但我们通常观察到性能有所提升, 例如头维度 128、序列长度 8192 的 FP16 前向传播从 570 TFLOPS 提升到 620-640 TFLOPS.

<span id="figure-01"></span>

![图 1. 两个 warpgroup 的乒乓调度, 用于重叠 softmax 和 GEMM: 一个 warpgroup 的 softmax 应在另一个 warpgroup 执行 GEMM 时调度. 相同颜色表示同一次迭代.](./flashattention-3/figure-01.png)

**图 1.** 两个 warpgroup 的乒乓调度, 用于重叠 softmax 和 GEMM: 一个 warpgroup 的 softmax 应在另一个 warpgroup 执行 GEMM 时调度. 相同颜色表示同一次迭代.

**注意力变体.** 对于多查询注意力 [Sha19] 和分组查询注意力 [Ain23a], 我们沿用 FlashAttention-2 的方法调整张量索引, 避免在 HBM 中复制 $\mathbf{K}$ 和 $\mathbf{V}$.

<span id="section-3-2"></span>

### 3.2 Warpgroup 内部的 GEMM-softmax 重叠

即使在单个 warpgroup 内部, 我们也可以让 softmax 的一部分指令与 GEMM 的一部分指令重叠. 下面介绍一种实现方法.

注意力算法内循环 (主循环) 中的操作存在顺序依赖, 妨碍了单次迭代内部的并行化. 例如, 局部 softmax (第 18-19 行) 依赖第一个 GEMM 的输出 $\mathbf{S}_{i}^{(j)}$, 而第二个 GEMM 以其结果 $\widetilde{\mathbf{P}}_{i}^{(j)}$ 作为操作数. 实际上, [算法 1](#algorithm-01)第 17 行和第 21 行的等待语句使 softmax 与 GEMM 串行执行. 不过, 借助额外的寄存器缓冲区, 我们可以跨迭代建立流水线, 从而打破这些依赖. 沿着这一思路, 我们提出下面的两阶段 [+6] GEMM-softmax 流水线算法:

<span id="figure-02"></span>

![图 2. 两阶段 WGMMA-softmax 流水线](./flashattention-3/figure-02.png)

**图 2.** 两阶段 WGMMA-softmax 流水线

<span id="algorithm-02"></span>

**算法 2: FlashAttention-3 消费者 warpgroup 前向传播.**

- **输入:** HBM 中的矩阵 $\mathbf{Q}_i \in \mathbb{R}^{B_r \times d}$ 与 $\mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$, 键分块大小 $B_c$, 且 $T_c = \lceil \frac{N}{B_c} \rceil$.
- 根据消费者 warp 数量重新分配预定数量的寄存器.
- 在片上初始化 $\mathbf{O}_i = (0) \in \mathbb{R}^{B_r \times d}$ 和 $\ell_i, m_i = (0), (-\infty) \in \mathbb{R}^{B_r}$.
- 等待 $\mathbf{Q}_i$ 与 $\mathbf{K}_0$ 加载到共享内存.
- 使用 WGMMA 计算 $\mathbf{S}_{\mathrm{cur}} = \mathbf{Q}_i \mathbf{K}_0^\top$. 提交并等待.
- 将缓冲区的第 $0$ 阶释放给 $\mathbf{K}$.
- 根据 $\mathbf{S}_{\mathrm{cur}}$ 计算 $m_i$、$\widetilde{\mathbf{P}}_{\mathrm{cur}}$ 和 $\ell_i$, 并重缩放 $\mathbf{O}_i$.
- **对于** $1 \le j < T_c - 1$:
  - 等待 $\mathbf{K}_j$ 加载到共享内存.
  - 使用 WGMMA 计算 $\mathbf{S}_{\mathrm{next}} = \mathbf{Q}_i \mathbf{K}_{j}^\top$. 提交但不等待.
  - 等待 $\mathbf{V}_{j-1}$ 加载到共享内存.
  - 使用 WGMMA 计算 $\mathbf{O}_{i} = \mathbf{O}_{i} + \widetilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$. 提交但不等待.
  - 等待 WGMMA $\mathbf{Q}_i \mathbf{K}_{j}^\top$.
  - 根据 $\mathbf{S}_{\mathrm{next}}$ 计算 $m_i$、$\widetilde{\mathbf{P}}_{\mathrm{next}}$ 和 $\ell_i$.
  - 等待 WGMMA $\widetilde{\mathbf{P}}_{\mathrm{cur}} \mathbf{V}_{j-1}$, 然后重缩放 $\mathbf{O}_i$.
  - 将缓冲区的第 $(j\,\%\,s)$ 阶、第 $(j-1\,\%\,s)$ 阶分别释放给 $\mathbf{K}$、$\mathbf{V}$.
  - 将 $\mathbf{S}_{\mathrm{next}}$ 复制到 $\mathbf{S}_{\mathrm{cur}}$.
- 等待 $\mathbf{V}_{T_c - 1}$ 加载到共享内存.
- 使用 WGMMA 计算 $\mathbf{O}_{i} = \mathbf{O}_{i} + \widetilde{\mathbf{P}}_{\mathrm{last}} \mathbf{V}_{T_c - 1}$. 提交并等待.
- **收尾:**
  - 根据 $m_i$ 重缩放 $\mathbf{O}_{i}$.
  - 根据 $m_i$ 和 $\ell_i$ 计算 $L_i$.
  - 把 $\mathbf{O}_{i}$ 和 $L_i$ 作为 $\mathbf{O}$ 和 $L$ 的第 $i$ 个分块写入 HBM.

[算法 2](#algorithm-02)会替换[算法 1](#algorithm-01)的消费者路径, 两者共同组成完整的 FP16 精度 FlashAttention-3 算法. 从高层看, 我们用 WGMMA 代指异步 GEMM. 在主循环 (第 8-16 行) 中, 迭代 $j$ 的第二次 WGMMA 操作 (第 11 行) 会与迭代 $j+1$ 的 softmax 操作 (第 13 行) 重叠.

上面所示的流水线结构在理论上能提升性能, 但实际使用时还要考虑几个方面:

**编译器重排.** 伪代码表示理想化的执行顺序, 但编译器 (NVCC) 常会为了优化而重新排列指令. 这可能打乱精心设计的 WGMMA 与非 WGMMA 操作流水线顺序, 导致意外行为或性能收益降低. 对 SASS 代码的分析表明, 编译器按预期生成了重叠代码 ([第 7.2 节](#section-7-2)).

**寄存器压力.** 为保持最佳性能, 应尽量减少寄存器溢出. 但是, 两阶段流水线需要额外寄存器来保存中间结果和维持各阶段间的上下文. 具体而言, 寄存器中必须多保留一个 $\mathbf{S}_{\mathrm{next}}$, 因此每个 threadblock 会额外使用大小为 $B_{r}\times B_{c}\times\text{sizeof}(\text{float})$ 的寄存器空间. 这种额外需求可能与增大分块尺寸这一常见优化冲突, 因为后者同样占用大量寄存器. 实践中应根据性能分析结果做取舍.

**三阶段流水线.** 在上述两阶段算法基础上, 我们提出三阶段变体, 进一步让第二次 WGMMA 与 softmax 重叠. 这种方法有望进一步提高 Tensor Core 利用率, 但新增流水线阶段会消耗更多寄存器, 使分块大小与流水线深度之间的平衡更难把握. 三阶段算法的详细说明和评估结果见[第 7.3 节](#section-7-3).

<span id="section-3-3"></span>

### 3.3 FP8 低精度

<span id="figure-03"></span>

![图 3. FP32 累加器寄存器的 WGMMA 布局, 第 0 行和第 8 行、线程 0-3、条目 0-7.](./flashattention-3/figure-03.png)

**图 3.** FP32 累加器寄存器的 WGMMA 布局, 第 0 行和第 8 行、线程 0-3、条目 0-7.

<span id="figure-04"></span>

![图 4. FP8 操作数 A 寄存器的 WGMMA 布局, 第 0 行和第 8 行、线程 0-3、条目 0-7.](./flashattention-3/figure-04.png)

**图 4.** FP8 操作数 A 寄存器的 WGMMA 布局, 第 0 行和第 8 行、线程 0-3、条目 0-7.

**效率: 布局变换.** 以 FP8 精度计算 FlashAttention-3 前向传播时, 在布局一致性方面会遇到 FP16 所没有的额外挑战.

首先, 输入张量 $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$ 通常在头维度上连续. 但为了满足第二次 GEMM 对 FP8 WGMMA 的 k-major 约束, $\mathbf{V}$, 更准确地说, 加载到 SMEM 中的 $\mathbf{V}$ tile, 必须在序列长度维度上连续. TMA 加载本身无法改变连续维度, 因而我们需要选择以下一种方式: (1) 把 $\mathbf{V}$ 在 GMEM 中转置, 作为预处理步骤; 或者 (2) 将 $\mathbf{V}$ 的 tile 加载到 SMEM 后, 在 kernel 内转置. 实现选项 (1) 时, 又可以 (1a) 把转置融合进旋转位置嵌入等前置步骤的收尾阶段, 或者 (1b) 调用独立的预处理转置 kernel [+7], 交换序列长度维度与头维度的 stride. 然而, (1a) 难以集成进标准库, (1b) 在推理这类受内存限制的场景中又过于浪费.

因此, FP8 FlashAttention-3 采用选项 (2). 在 kernel 内转置时, 我们利用 LDSM (`ldmatrix`) 和 STSM (`stmatrix`) 指令: 一个 warp 的线程以 128 byte 粒度协同把 SMEM 加载到 RMEM, 再把 RMEM 存回 SMEM. [+8] LDSM/STSM 指令对寄存器的使用都很高效, 可以在生产者 warpgroup 中执行, 而且在内存复制时能够转置布局. 此外, 从第二次迭代开始, 我们可以把下一个 $\mathbf{V}$ tile 的转置安排在两个 WGMMA 的执行阴影下; 这两个 WGMMA 分别涉及前一个 $\mathbf{V}$ tile 和当前 $\mathbf{K}$ tile.

其次, 我们观察到 FP8 WGMMA 的 FP32 累加器内存布局与寄存器中操作数 A 的预期布局不同, 这与 FP16 的情况不一样. [图 3](#figure-03) 和[图 4](#figure-04) 展示了两种布局的片段, 其中条目按所列顺序保存在每个线程的寄存器中. 借助 byte permute 指令, 我们可以把第一次 WGMMA 的累加器转换成适合第二次 WGMMA 的格式, 并与 kernel 内转置得到的 $\mathbf{V}$ tile 布局兼容. 具体来说, 参照[图 3](#figure-03), 我们把顺序改为

$$
\{\verb|d0 d1 d4 d5 d2 d3 d6 d7|\},
$$

随后对每 8 byte 重复这一寄存器置换. 从 $\mathbf{P}$ tile 的逻辑形状来看, 这一操作置换了它的列, 例如列 $0189$ 现在成为前四列. 为了让 WGMMA 计算正确的输出 tile, 我们可以相应安排 kernel 内转置, 写出匹配的 $\mathbf{V}$ tile 行置换. [+9]

**准确度: 分块量化与非相干处理.** FP8 (e4m3) 格式只用 3 bit 保存尾数, 4 bit 保存指数. 因此, 它的数值误差高于 FP16/BF16. 而且, 大模型中通常存在幅值远大于多数其他值的离群值 [Det22, Sun24c], 给量化带来困难. 常见做法是逐张量缩放 [Mic22], 即每个张量保留一个标量, 例如 $\mathbf{Q}$、$\mathbf{K}$ 和 $\mathbf{V}$ 各一个. 为降低 FP8 注意力的数值误差, 我们采用两项技术:

- **分块量化:** 每个分块保留一个标量. 对 $\mathbf{Q}$、$\mathbf{K}$、$\mathbf{V}$, 我们分别将张量拆成大小为 $B_{r}\times d$ 或 $B_{c}\times d$ 的分块, 再分别量化. 量化可以融合到注意力之前的操作中, 例如旋转位置嵌入, 不会带来额外减速, 因为旋转位置嵌入受内存带宽限制. FlashAttention-3 算法本来就按分块运行, 因此我们可以缩放 $\mathbf{S}$ 的每个分块来反映这种分块量化, 而不增加计算成本.
- **非相干处理:** 为了摊平离群值, 我们在量化到 FP8 之前, 将 $\mathbf{Q}$ 与 $\mathbf{K}$ 乘以随机正交矩阵 $\mathbf{M}$. 因为 $\mathbf{M}$ 正交, 有 $\mathbf{M}\mathbf{M}^{\top}=I$, 所以 $(\mathbf{Q}\mathbf{M})(\mathbf{K}\mathbf{M})^{\top}=\mathbf{Q}\mathbf{K}^{\top}$; 换言之, $\mathbf{Q}$ 和 $\mathbf{K}$ 同乘 $\mathbf{M}$ 不会改变注意力输出. 由于 $\mathbf{Q}\mathbf{M}$ 或 $\mathbf{K}\mathbf{M}$ 的每个条目都是 $\mathbf{Q}$ 或 $\mathbf{K}$ 条目的随机和, 这种方法会把离群值“摊开”, 从而降低量化误差. 实际上, 我们沿用 [Che24b] 和 [Tse24], 将 $\mathbf{M}$ 选为 $\pm 1$ 随机对角矩阵与 Hadamard 矩阵的乘积. 这样, 矩阵乘法的复杂度可以从 $O(d^{2})$ 降到 $O(d\log d)$, 还可以无额外计算成本地融合到旋转位置嵌入中.

我们在[第 4.3 节](#section-4-3)验证了这两项技术可将数值误差最多降低 $2.6\times$.

<span id="section-4"></span>

## 4 实证验证

我们使用 CUTLASS [Nvi24a] 提供的 WGMMA、TMA 抽象等基础组件实现 FlashAttention-3, 并评估其效率和准确度.

- **注意力基准测试.** 我们测量 FlashAttention-3 在不同序列长度下的运行时间, 并将其与 PyTorch 中的标准实现、FlashAttention-2、Triton 实现的 FlashAttention-2 (使用 H100 专用指令), 以及 cuDNN 中针对 H100 GPU 优化的厂商版 FlashAttention-2 实现进行比较. 结果确认, FlashAttention-3 最多比 FlashAttention-2 快 $2.0\times$, 比 Triton 版 FlashAttention-2 快 $1.5\times$. FlashAttention-3 最高达到 740 TFLOPs/s, 即 H100 GPU 理论最高 TFLOPs/s 的 75%.
- **消融研究.** 我们确认, warp 特化与 GEMM-softmax 流水线这两项算法改进都对 FlashAttention-3 的加速有所贡献.
- **FP8 注意力的准确度.** 我们验证了分块量化和非相干处理可将 FP8 FlashAttention-3 的数值误差降低 $2.6\times$.

<span id="section-4-1"></span>

### 4.1 注意力基准测试

我们在 H100 80GB SXM5 GPU 上针对不同设置测量多种注意力方法的运行时间: 无因果掩码或有因果掩码, 头维度为 64 或 128, 输入为 FP16. [图 5](#figure-05) 和[图 6](#figure-06) 给出了结果. FlashAttention-3 的前向传播约比 FlashAttention-2 快 1.5-$2.0\times$, 反向传播快 1.5-$1.75\times$. 与标准注意力实现相比, FlashAttention-3 最多可快 3-$16\times$. 对中长序列 (1k 及以上), FlashAttention-3 的速度甚至超过针对 H100 GPU 优化的厂商库, 即闭源 cuDNN.

**基准设置:** 序列长度取 512、1k、…、16k, 并设置批大小, 使 token 总数为 16k. 隐藏维度设为 2048, 头维度取 64、128 或 256, 即分别有 32、16 或 8 个头. 前向传播 FLOPs 按下式计算:

$$
\begin{aligned}
4\cdot\text{seqlen}^{2}\cdot\text{head dimension}\\
{}\cdot\text{number of heads}.
\end{aligned}
$$

使用因果掩码时, 由于实际只计算大约一半条目, 将该数除以 2. 反向传播 FLOPs 为前向传播 FLOPs 的 2.5 倍, 因为前向传播有 2 次矩阵乘, 而反向传播由于重新计算会执行 5 次矩阵乘.

<span id="figure-05"></span>

![图 5. H100 GPU 上的注意力前向传播速度 (FP16/BF16)](./flashattention-3/figure-05.png)

**图 5.** H100 GPU 上的注意力前向传播速度 (FP16/BF16)

<span id="figure-06"></span>

![图 6. H100 GPU 上的注意力反向传播速度 (FP16/BF16)](./flashattention-3/figure-06.png)

**图 6.** H100 GPU 上的注意力反向传播速度 (FP16/BF16)

我们还在相近设置下测量了 FP8 前向传播的运行时间. [图 7](#figure-07) 给出了头维度 256 的结果, 完整结果见[第 8.2 节](#section-8-2).

<span id="figure-07"></span>

![图 7. H100 GPU 上的注意力前向传播速度 (FP8)](./flashattention-3/figure-07.png)

**图 7.** H100 GPU 上的注意力前向传播速度 (FP8)

<span id="section-4-2"></span>

### 4.2 消融研究: 两阶段流水线实验

我们对无因果掩码的 FP16 FlashAttention-3 分别消融两阶段 WGMMA-softmax 流水线和 warp 特化, 固定参数 $\{\text{batch},\text{seqlen},\text{nheads},\text{hdim}\}=\{4,8448,16,128\}$. [表 2](#table-02) 的结果确认, 我们的算法改进, 即 warp 特化带来的异步执行以及 GEMM 与 softmax 的重叠, 带来了显著加速, 从 570 TFLOPs 提升到 661 TFLOPs.

<span id="table-02"></span>

![表 2. 流水线消融测量结果](./flashattention-3/table-02.png)

**表 2.** 流水线消融测量结果

<span id="section-4-3"></span>

### 4.3 数值误差验证

鉴于 FlashAttention 的数值误差 [Gol24] 受到关注, 我们以 FP64 参考实现为基准, 比较 FlashAttention-2、FlashAttention-3 与标准注意力实现. 为模拟 LLM 中的离群特征与激活 [Det22, Sun24c], 我们按以下分布生成 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的条目:

$$
\mathcal{N}(0,1)+\mathcal{N}(0,100)\cdot\mathrm{Bernoulli}(0.001).
$$

也就是说, 每个条目都服从均值为 0、标准差为 1 的正态分布, 但对其中 0.1% 的条目, 我们会再加上一个独立项, 它服从标准差为 10 的正态分布. 随后, 我们测量[表 3](#table-03) 所示的均方根误差 (RMSE). 在 FP16 下, FlashAttention-2 和 FlashAttention-3 的 RMSE 都比标准实现低 $1.7\times$, 因为中间结果 softmax 保持为 FP32. FP8 基线注意力采用逐张量缩放, 矩阵乘累加器为 FP32, 中间 softmax 结果保持为 FP16. 得益于分块量化和非相干处理, FP8 FlashAttention-3 比这项基线准确 $2.6\times$.

<span id="table-03"></span>

![表 3. FP16 与 FP8 (e4m3) 的数值误差比较.](./flashattention-3/table-03.png)

**表 3.** FP16 与 FP8 (e4m3) 的数值误差比较.

<span id="section-5"></span>

## 5 讨论、局限与结论

通过 FlashAttention-3, 我们证明了异步执行、低精度等新的编程技术和硬件特性能够显著影响注意力的效率与准确度. 与 FlashAttention-2 相比, 注意力速度提高了 1.5-$2.0\times$, 同时 FP8 数值误差相较标准逐张量量化降低 $2.6\times$. 我们希望在未来解决的局限包括: 针对 LLM 推理进行优化; 将持久化 kernel 设计集成进 FP8 kernel [+10]; 以及理解低精度注意力对大规模训练的影响. 本文虽然聚焦 Hopper GPU, 但我们预计这里提出的技术同样适用于其他硬件加速器. 我们希望, 注意力这种更快、更准确的基础操作能为长上下文任务打开新的应用空间.

## 致谢

我们感谢 NVIDIA CUTLASS 团队, 尤其是 Haicheng Wu、Aniket Shivam 和 Cris Cecka, 帮助我们理解 Hopper 的编程模型; 也感谢他们提供的库, 它为实现 FlashAttention-3 提供了简洁而强大的基础组件. 感谢 cuDNN 团队提出 FP8 kernel 内转置的思路. 重叠 GEMM 与 softmax 的想法受到我们与 Christopher Ré、Benjamin Spector、Aniket Shivam 和 Markus Hoehnerbach 深入讨论的启发. 乒乓调度改编自 CUTLASS 中采用 warp 特化的乒乓 GEMM 实现. 感谢 Driss Guessous 将 FlashAttention 集成到 PyTorch. FlashAttention-3 也受益于多次有益讨论: 与 Horace He 讨论不同注意力变体, 与 Hao Liu 和 Phil Wang 讨论分布式注意力, 与 Daniel Haziza 和 Chris De Sa 讨论量化. 感谢 Meta、Together AI 和 Princeton Language and Intelligence (PLI) 提供计算支持.

<span id="section-6"></span>

## 6 相关工作

**注意力变体与分布式注意力.** 自从注意力随 Transformer 架构 [Vas17] 普及以来, 大量工作通过近似注意力把它扩展到更长序列. 这些近似方法大体分为稀疏和低秩两类. 稀疏注意力只计算注意力矩阵 $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)$ 的一部分条目, 并假设其他条目为零. 不同方法选择零条目的方式各异, 可以使用固定模式 [Chi19]、滑动窗口 [Bel20a], 也可以通过哈希 [Kit20] 或路由 [Roy21] 动态选择. 低秩方法则假设注意力矩阵具有低秩结构, 对查询与键应用逐点非线性 [Kat20], 并使用随机投影 [Cho20a, Pen21, Xio21]. 稀疏近似和低秩近似还可以结合, 以获得更好的质量 [Zah20, Che21b]. 然而, 这些近似方法通常无法达到标准注意力的模型质量 [Tay20a], 因此大多数大规模模型并未采用.

还有一些注意力变体旨在缩小 KV cache, 提高推理效率. 多查询注意力 [Sha19] 与分组查询注意力 [Ain23a] 在不同的 $\mathbf{K}$ 和 $\mathbf{V}$ 头之间共享参数, 让多个查询头与同一个键头和值头交互. 多头潜在注意力 [Dee24] 将 $\mathbf{K}$ 和 $\mathbf{V}$ 参数化为共享矩阵的低秩投影, 进一步缩小 KV cache. 不过, 这些方法都不改变训练期间的核心计算 $\mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$, 只是改变 $\mathbf{Q},\mathbf{K},\mathbf{V}$ 的获取方式. 因此, 对标准注意力计算的任何效率或准确度改进都能让这些方法受益.

若要扩展到更长上下文, 可以把注意力计算分布到多个 GPU. Ring attention [Liu23, Liu24l] 及其变体 [Bra23] 等方法可将上下文长度扩展到 100 万. 它们使用 FlashAttention 或 FlashAttention-2 作为基础操作, 因此 FlashAttention-3 的改进同样会惠及这些分布式注意力方法.

**替代架构.** 受注意力局限的驱动, 研究者提出了多种替代架构. 这些方法建立在线性注意力 [Kat20] 与循环神经网络 (RNN) 的联系之上. RWKV [Pen23b]、H3 [Dao22g]、MEGA [Ma23b] 与 Retnet [Sun23b] 采用更复杂的递归, 增强线性注意力中简单累积和的表达能力. Mamba [Gu23] 与 xLSTM [Bec24] 使用可学习的递归权重, 在中小规模语言建模中能够达到 Transformer 的质量. 从 token mixing 矩阵的结构来看, 这些方法可以与线性注意力的推广联系起来 [Dao24]. 这些模型已开始得到一些应用, 出现在 Jamba [Jam24]、Zamba [Zam24]、Megalodon [Ma24e] 和 Mamba2-hybrid [Wal24] 等中大型模型中. 为了获得最高质量, 这些基于 SSM 与 RNN 的模型仍使用许多注意力层. 我们预计, 本文的注意力加速技术也能用于加速这些替代架构.

**低精度注意力.** 量化是加速注意力的一条很有前景的路径, 但相关工作大多聚焦缩小 KV cache, 以提高推理效率. QuIP [Che24b] 和 QuIP# [Tse24] 使用非相干处理降低量化误差, 我们将这项技术用于 FP8 FlashAttention-3. 最近的工作表明, 推理时的 KV cache 具有很高的可压缩性, 可以压缩到 4 bit、3 bit, 甚至 2 bit [Hoo24, Liu24c]. 不过, 训练期间的量化仍然困难, 因为稳定训练通常需要更高精度.

**硬件感知算法.** 本文工作聚焦面向微架构的具体调优, 以利用新指令集并采用原生异步编程模型. 硬件感知算法协同设计还有其他相互正交的方向. 最近的例子是 LeanAttention [San24a]: 它发现顺序 token 生成阶段的 GPU 占用率低、内存带宽需求高, 是推理的主要瓶颈, 因而采用类似 Stream-K 负载均衡 [Osa23] 的更智能策略进行优化, 达到接近峰值的占用率. 针对特定硬件优化 GEMM 的研究很多, 其中采用了大量相同技术. 例如, [Abd16] 为 K40c 图形处理器 (GPU) 提出了一款支持固定与可变尺寸的高性能批处理 GEMM kernel, 通过专用 GEMM 设计和全面的自动调优流程达到当时最先进的性能.

<span id="section-7"></span>

## 7 算法补充细节

<span id="section-7-1"></span>

### 7.1 通过 warp 特化实现反向传播的异步执行

与前向传播的[第 3.1 节](#section-3-1)相似, 我们使用 warp 特化处理异步执行. 前向传播只有简单的生产者-消费者模式, 而这里需要额外加入 $\mathbf{dQ}$ 写入者角色, 因为每个 threadblock 生成的 $\mathbf{dQ}$ 值都必须累加到全局 $\mathbf{dQ}$ 中. 这种 $\mathbf{dQ}$ 累加会引发内存争用, 因为许多 threadblock 会写入同一位置. 因此, 用单独一个 warp 处理这项工作和相应的异步执行, 可以避免阻塞 threadblock 中的其他 warp, 让它们继续执行下一项计算, 即矩阵乘.

[算法 3](#algorithm-03)给出了采用 warp 特化的反向传播.

<span id="algorithm-03"></span>

**算法 3: 采用 warp 特化的 FlashAttention-3 反向传播.**

- **输入:** HBM 中的矩阵 $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N \times d}$, HBM 中的 logsumexp 向量 $L \in \mathbb{R}^N$, 分块大小 $B_c$、$B_r$.
- 在预处理 kernel 中, 计算 $D = \mathrm{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^d$ (逐点相乘), 把 $D$ 写入 HBM, 并将其划分为 $T_r$ 个分块 $D_1, \dots, D_{T_r}$, 每个大小为 $B_r$.
- 将 $\mathbf{Q}$ 划分为 $T_r = \left\lceil\frac{N}{B_r} \right\rceil$ 个分块 $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_r}$, 每个大小为 $B_r \times d$; 将 $\mathbf{K}, \mathbf{V}$ 分别划分为 $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ 个分块 $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ 与 $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$, 每个大小为 $B_c \times d$.
- 将 $\mathbf{dO}$ 划分为 $T_r$ 个分块 $\mathbf{dO}_i, \dots, \mathbf{dO}_{T_r}$, 每个大小为 $B_r \times d$; 将 $L$ 划分为 $T_r$ 个分块 $L_i, \dots, L_{T_r}$, 每个大小为 $B_r$.
- 初始化流水线对象, 用 $s$ 阶环形 SMEM 缓冲区管理屏障同步.
- **如果**位于生产者 warpgroup:
  - 释放预定数量的寄存器.
  - 发射从 HBM 到共享内存的 $\mathbf{K}_j$ 和 $\mathbf{V}_j$ 加载.
  - 加载完成后, 提交通知消费者 $\mathbf{K}_j$ 和 $\mathbf{V}_j$ 已加载.
  - **对于** $1 \le i \leq T_r$:
    - 等待缓冲区的第 $(i\,\%\,s)$ 阶被消费.
    - 发射从 HBM 到共享内存的 $\mathbf{Q}_i, \mathbf{dO}_i$ 加载, 目标为缓冲区的第 $(i\,\%\,s)$ 阶.
    - 加载完成后, 提交通知消费者 $\mathbf{Q}_i, \mathbf{dO}_i$ 已加载.
- **否则如果**位于消费者 warpgroup:
  - 根据消费者 warp 数量重新分配预定数量的寄存器.
  - 在片上初始化 $\mathbf{dK}_j = (0)_{B_c \times d}, \mathbf{dV}_j = (0)_{B_c \times d}$.
  - 等待 $\mathbf{K}_j$ 和 $\mathbf{V}_j$ 加载到共享内存.
  - **对于** $1 \le i \leq T_r$:
    - 等待 $\mathbf{Q}_i$ 加载到共享内存.
    - 将 $L_i, D_i$ 从 HBM 加载到片上 SRAM.
    - 在片上计算 $\mathbf{S}_{i}^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). 提交.
    - 等待 $\mathbf{dO}_i$ 加载到共享内存.
    - 在片上计算 $\mathbf{dP}_{i}^{(j)} = \mathbf{dO}_{i} \mathbf{V}_j^\top \in \mathbb{R}^{B_r \times B_c}$ (SS-GEMM). 提交.
    - 在片上等待 $\mathbf{S}_{i}^{(j)}$, 随后计算 $\mathbf{P}_{i}^{(j)} = \exp(\mathbf{S}_{ij} - L_{i}) \in \mathbb{R}^{B_r \times B_c}$.
    - 在片上等待 $\mathbf{dP}_i^{(j)}$, 随后计算 $\mathbf{dS}_{i}^{(j)} = \mathbf{P}_{i}^{(j)} \circ (\mathbf{dP}_{i}^{(j)} - D_i) \in \mathbb{R}^{B_r \times B_c}$.
    - 在片上计算 $\mathbf{dV}_j \leftarrow \mathbf{dV}_j + (\mathbf{P}_{i}^{(j)})^\top \mathbf{dO}_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM). 提交.
    - 在片上计算 $\mathbf{dK}_{j} \leftarrow \mathbf{dK}_j + {\mathbf{dS}_{i}^{(j)}}^\top \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$ (RS-GEMM). 提交, 并等待 $\mathbf{dV}_j$ 与 $\mathbf{dK}_j$.
    - 在片上计算 $\mathbf{dQ}_{i}^{(\mathrm{local})} = \mathbf{dS}_{i}^{(j)} \mathbf{K}_j \in \mathbb{R}^{B_r \times d}$ (SS-GEMM), 并将 $\mathbf{dQ}_i^{(\mathrm{local})}$ 写入 SMEM. 通知 $\mathbf{dQ}$ 写入者.
- **否则如果**位于 $\mathbf{dQ}$ 写入者 warp:
  - **对于** $1 \le i \leq T_r$:
    - 等待 SMEM 中的 $\mathbf{dQ}_i^{(\mathrm{local})}$ 就绪.
    - 使用 semaphore, 将 $\mathbf{dQ}_i^{(\mathrm{local})}$ 原子加到全局内存中的 $\mathbf{dQ}_i$.

<span id="section-7-2"></span>

### 7.2 两阶段流水线的 SASS 分析

下面给出消费者 warpgroup 主循环内部的简化 SASS 代码.

```text
// Compute row_max
FMNMX.FTZ R0, R24, R6, !PT ;
SHFL.BFLY PT, R185, R2, 0x2, 0x1f ;
… FMNMX and SHFL.BFLY …

// Apply exp2 and row_sum. Rescale O.
FMUL.FTZ R2, R4, UR9 ;
MUFU.EX2 R185, R184 ;
FFMA.FTZ R24, R24, UR9, -R6.reuse ;
FADD.FTZ R24, R211, R24 ;
… FMUL, FFMA, FMUL, MUFU.EX2, FADD …

// FP32 -> FP16 conversion are interleaved with exp2, row_sum and O rescaling.
F2FP.F16.F32.PACK_AB R231, R25, R231 ;
… F2FP, FMUL, MUFU, FFMA, FADD ...

// Start the first WGMMA. Broken down into 8 HGMMAs.
// The first 7 HGMMAs are packed together.
WARPGROUP.ARRIVE ;
HGMMA.64x192x16.F32 R24, gdesc[UR44], RZ, !UPT ;
... HGMMA x 6 ...

// FP32->FP16, exp2, row_sum, O rescaling are interleaved with HGMMA.
F2FP.F16.F32.PACK_AB R214, R214, R187 ;
MUFU.EX2 R234, R5 ;
FADD.FTZ R237, R187, R2 ;
… F2FP, MUFU, FADD …

// The last HGMMA is issued here. No need to wait.
HGMMA.64x192x16.F32 R24, gdesc[UR44], R24, gsb0 ;

// Start the second WGMMA. Broken down into 12 HGMMAs.
// All 12 HGMMAs are packed together. Not interleaved with other instructions.
WARPGROUP.ARRIVE ;
HGMMA.64x128x16.F32 R120, R228, gdesc[UR8].tnspB, R120 ;
... HGMMA x 10 ...
HGMMA.64x128x16.F32 R120, R184, gdesc[UR8].tnspB, R120, gsb0 ;

// wgmma.wait_group at the end.
WARPGROUP.DEPBAR.LE gsb0, 0x0 ;
```

我们可以得到以下观察:

- Softmax 被重排到最开头, 甚至位于第一次 WGMMA 之前.
- 第一次 WGMMA 与 softmax 以及 $\mathbf{S}$ 的 FP32 $\rightarrow$ FP16 数据类型转换交错. 这说明 WGMMA 与非 WGMMA 操作在并行执行.
- `exp2`、`row\_sum`、O 重缩放与 FP32 $\rightarrow$ FP16 转换彼此交错.
- 第二次 WGMMA 按预期没有与其他指令重叠.

总体来看, SASS 表明两阶段流水线思路按预期工作.

<span id="section-7-3"></span>

### 7.3 三阶段流水线算法

我们尝试使用三阶段流水线算法, 并行执行迭代 $j+2$ 的第一次 WGMMA、迭代 $j+1$ 的 softmax 和迭代 $j$ 的第二次 WGMMA. [算法 4](#algorithm-04)给出了这套算法. 由于以下原因, 它的表现不如两阶段流水线算法:

<span id="figure-08"></span>

![图 8. 三阶段流水线](./flashattention-3/figure-08.png)

**图 8.** 三阶段流水线

<span id="algorithm-04"></span>

**算法 4: FlashAttention 三阶段流水线的消费者 warpgroup 前向传播.**

- **输入:** HBM 中的矩阵 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$, 分块大小 $B_c$、$B_r$. 每个 warpgroup 读取一个大小为 $B_r \times d$ 的 $\mathbf{Q}_i$ 分块、$T_c = \left\lceil \frac{N}{B_c} \right\rceil$ 个大小为 $B_c \times d$ 的分块 $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ 和 $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$. 每个 warpgroup 写入一个大小为 $B_r \times d$ 的输出分块 $\mathbf{O}_i$, 以及一个大小为 $B_r$ 的 logsumexp 分块 $L_i$.
- 初始化. 将 $\mathbf{Q}_i$ 从 HBM 加载到片上 SRAM. 初始化 $\mathbf{O}_i, \ell_i, m_i, \mathrm{scale}_o$.
- 等待生产者 warpgroup 将 $\mathbf{K}_0$ 从 HBM 加载到片上 SRAM.
- 使用 WGMMA 计算 $\mathbf{S} = \mathbf{Q}_i \mathbf{K}_0^\top$. 提交并等待.
- 根据 $\mathbf{S}$ 计算 $m_i$、$\widetilde{\mathbf{P}}_i$、$\ell_i$、$\mathrm{scale}_o$.
- 等待生产者 warpgroup 将 $\mathbf{K}_1$ 从 HBM 加载到片上 SRAM.
- 使用 WGMMA 计算 $\mathbf{S} = \mathbf{Q}_i \mathbf{K}_1^\top$. 提交并等待.
- **对于** $2 \le j < T_c - 2$:
  - 等待生产者 warpgroup 将 $\mathbf{K}_j$ 从 HBM 加载到片上 SRAM.
  - 使用 WGMMA 计算 $\mathbf{S}_{\mathrm{next}} = \mathbf{Q}_i \mathbf{K}_{j}^\top$. 提交但不等待.
  - 等待生产者 warpgroup 将 $\mathbf{V}_{j-2}$ 从 HBM 加载到片上 SRAM.
  - 根据 $\mathrm{scale}_o$ 重缩放 $\mathbf{O}_i$.
  - 使用 WGMMA 计算 $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{j-2}$. 提交但不等待.
  - 根据 $\mathbf{S}$ 计算 $m_i$、$\widetilde{\mathbf{P}}_{i,\mathrm{next}}$、$\ell_i$、$\mathrm{scale}_o$.
  - 等待此前所有 WGMMA.
  - 将 $\mathbf{S}_{\mathrm{next}}$ 复制到 $\mathbf{S}$.
  - 将 $\widetilde{\mathbf{P}}_{i,\mathrm{next}}$ 复制到 $\widetilde{\mathbf{P}}_i$.
- 等待生产者 warpgroup 将 $\mathbf{V}_{T_c-2}$ 从 HBM 加载到片上 SRAM.
- 根据 $\mathrm{scale}_o$ 重缩放 $\mathbf{O}_i$.
- 使用 WGMMA 计算 $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{T_c-2}$. 提交并等待.
- 根据 $\mathbf{S}$ 计算 $m_i$、$\widetilde{\mathbf{P}}_i$、$\ell_i$、$\mathrm{scale}_o$.
- 等待生产者 warpgroup 将 $\mathbf{V}_{T_c-1}$ 从 HBM 加载到片上 SRAM.
- 根据 $\mathrm{scale}_o$ 重缩放 $\mathbf{O}_i$.
- 使用 WGMMA 计算 $\mathbf{O}_i = \mathbf{O}_i + \widetilde{\mathbf{P}}_i \mathbf{V}_{T_c-1}$. 提交并等待.
- 收尾. 根据 $\ell_i$ 重缩放 $\mathbf{O}_i$. 根据 $\ell_i$ 和 $m_i$ 计算 $L_i$. 把 $\mathbf{O}_i$ 和 $L_i$ 作为 $\mathbf{O}$ 和 $L$ 的第 $i$ 个分块写入 HBM.

**重叠.** 我们原本预期 softmax 可以和“第一次 WGMMA + 第二次 WGMMA”重叠. 但编译器没有这样配合. SASS 代码显示, 只有第一次 WGMMA 与 softmax 重叠, 第二次没有. 尚不清楚编译器为何以这种方式重排指令.

**寄存器压力.** 与两阶段流水线算法相比, 该算法需要更多寄存器. 理论上, 它必须额外保存一个 $\tilde{\mathbf{P}}_{i}$ 和 $\mathrm{scale}_o$, 大小为 $B_{r}\times B_{c}\times\text{sizeof}(\text{input\_data\_type})+B_{r}\times\text{sizeof}(\text{float})$. 因此, 必须选用更小的分块尺寸.

<span id="section-8"></span>

## 8 实验与基准测试补充细节

<span id="section-8-1"></span>

### 8.1 系统与库

我们在 H100 80GB SXM5 (700W) 上测试速度. 所用库通常为写作时 (2024 年 5 月) 的最新版本. 具体包括:

- CUDA 12.3
- cuDNN 9.1.1.17
- CUTLASS 3.5
- FlashAttention 2.5.8
- Triton nightly 3.0.0.post20240424212437
- PyTorch 2.3.0

为减少波动, 我们把 GPU 时钟频率固定为 1830MHz, 计算 989 TFLOPS 的 FP16 理论最高吞吐量时也使用这一频率. 基准测试重复 100 次, 取平均时间.

<span id="section-8-2"></span>

### 8.2 FP8 注意力完整结果

我们使用以下序列长度: 512、1024、2048、4224、8448、16896. 当序列长度 $\geq$ 4k 时, 还会让它能被 132 整除, 这是 H100 SXM5 的 SM 数量, 从而避免 wave quantization.

<span id="figure-09"></span>

![图 9. H100 GPU 上的注意力前向传播速度 (FP8)](./flashattention-3/figure-09.png)

**图 9.** H100 GPU 上的注意力前向传播速度 (FP8)

[+1]: 我们在 NVIDIA Hopper 架构的背景下介绍实验结果. 不过, 只要 GPU 架构具备足够可靠的异步执行和低精度能力, 我们的算法就能工作.

[+2]: 更准确地说, 对头维度 64, FlashAttention-3 FP8 更快; 对头维度 128 和 256, 无因果掩码时二者相当, 有因果掩码时 FlashAttention-3 FP8 较慢.

[+3]: FlashAttention-3 可在 [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention) 获取.

[+4]: [Luo24b] 报告每个 SM、每时钟周期的共享内存带宽为 128 byte, 我们再乘以 132 个 SM 和 1830 MHz 的 boost clock.

[+5]: CUDA 编程指南规定, 每个流式多处理器 (SM) 每时钟周期可以执行 16 次特殊函数操作. 我们将 16 乘以 132 个 SM 和 1830 MHz 时钟频率, 得到 3.9 TFLOPS 的特殊函数吞吐量.

[+6]: 需要注意, 重叠方案的阶段数受环形 SMEM 缓冲区阶段数 $s$ 的上限约束, 但不必与 $s$ 相等.

[+7]: 优化后的转置 kernel 能达到接近设备带宽的速度 [Cut24].

[+8]: PTX 文档将 LDSM/STSM 描述为复制条目为 16 bit 的 $8\times 8$ 矩阵 [Ptx24], 但我们可以每次打包两个 8 bit 条目, 从而在 FP8 精度下使用 LDSM/STSM. 不过, LDSM/STSM 的转置版本无法拆分打包后的 8 bit 条目, 因此必须在 LDSM 与 STSM 之间移动某些寄存器, 才能真正完成逐 tile 转置; 细节从略.

[+9]: kernel 内转置提供的额外自由度无需再用 shuffle 指令改变线程间的寄存器所有权; 我们此前在 [Bik24] 中介绍过这种做法.

[+10]: 在我们的基准测试中, FP16 FlashAttention-3 采用持久化 kernel 和负载均衡策略, FP8 FlashAttention-3 则没有. 这在一定程度上解释了为什么对短序列和因果掩码, FP8 FlashAttention-3 的表现不如 FP8 cuDNN kernel.

[+equal]: 同等贡献.
