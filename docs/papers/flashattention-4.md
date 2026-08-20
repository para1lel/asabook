---
title: 'FlashAttention-4'
createTime: 2026/08/20 17:49:32
permalink: /papers/flashattention-4/
pageClass: paper-reading
---

> [Ted Zadouri](https://tedzadouri.com/) [+equal], [Markus Hoehnerbach](https://www.linkedin.com/in/markus-h%C3%B6hnerbach-0b6391166) [+equal], [Jay Shah](https://developer.nvidia.com/blog/author/jayshah/) [+equal], [Timmy Liu](https://www.linkedin.com/in/jian-timmy-liu), [Vijay Thakkar](https://cse.gatech.edu/people/vijay-thakkar), [Tri Dao](https://tridao.me/). 论文于 2026 年 3 月 5 日首次提交至 arXiv, 当前版本为 v1. 发表于 [Proceedings of Machine Learning and Systems 8 (MLSys 2026)](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html). [FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451v1). [原始 PDF](/paper/flashattention-4.pdf). [DOI](https://doi.org/10.48550/arXiv.2603.05451). [TeX 源码](https://export.arxiv.org/e-print/2603.05451v1). 精确的印刷版式与参考文献以原始 PDF 为准.

## 摘要

注意力是无处不在的 Transformer 架构的核心层, 也是大语言模型和长上下文应用的瓶颈. FlashAttention-3 通过异步执行和 warp 特化优化了 Hopper GPU 上的注意力, 但它主要针对 H100 架构. AI 行业已经迅速转向部署 B200 和 GB200 等基于 Blackwell 的系统; 由于硬件各单元扩展速度不对称, 这些系统的性能特征有本质差异: Tensor Core 吞吐量翻倍, 而其他功能单元 (共享内存带宽、指数单元) 扩展得更慢, 甚至保持不变. 我们提出几项技术, 以解决 Blackwell GPU 上随之变化的瓶颈: (1) 重新设计流水线, 利用完全异步的 MMA 操作和更大的 tile; (2) 用软件模拟指数函数, 并按条件执行 softmax 重缩放, 从而减少非矩阵乘操作; (3) 利用 Tensor Memory 和 2-CTA MMA 模式, 减少反向传播中的共享内存流量与原子加法. 实验表明, 我们的方法 FlashAttention-4 在 B200 GPU 上使用 BF16 时, 相比 cuDNN 9.13 最多加速 1.3$\times$, 相比 Triton 最多加速 2.7$\times$, 吞吐量最高达到 1613 TFLOPs/s (利用率 71%). 除了算法创新, 我们还完全使用嵌入 Python 的 CuTe-DSL 实现了 FlashAttention-4; 它保留了完整的表达能力, 编译速度则比传统的 C++ 模板方案快 20-30$\times$.

## 1 引言

Transformer 架构 [Vas17] 仍是几乎所有 AI 应用的主要骨干, 覆盖大语言模型 [Bro20]、视觉 [Dos20] 和多模态系统. 对 Transformer 而言, 注意力机制是主要计算瓶颈, 因为 query 与 key 之间的自注意力分数随序列长度二次增长. 将注意力扩展到更长的上下文后, 模型可以获得一些新能力, 例如跨多篇文档推理 [Guo21a, Sha22a]、建模整个代码库 [Roz23], 以及处理高分辨率视频 [Che22a, Ho22]. 与此同时, 加速器硬件仍在快速演进 [Nvi24d], 每一代都能提供高得多的峰值计算吞吐量. 但这种演进并不对称: 矩阵乘单元扩展很快, 内存带宽和专用计算单元等其他功能单元则扩展得较慢, 由此形成越来越不均衡的硬件流水线, 算法必须结合这些特征一同设计.

因此, 人们一直在研究如何把 GPU 的硬件特征深入融入算法创新, 以加快注意力计算. Dao 等人 [Dao22] 提出了 FlashAttention, 通过新的分块与 kernel 融合方式, 避免在较慢的全局内存中读写中间结果. Dao [Dao23b] 将其重构为 FlashAttention-2, 沿序列长度维度并行, 提高 GPU 占用率. Shah 等人 [Sha24b] 又将算法适配到 Hopper GPU, 得到 FlashAttention-3; 它用 warp 特化实现异步执行, 并加入 FP8 支持. 近期工作也研究了低精度注意力: SageAttention [Lin24d] 通过 INT8 量化实现加速, SageAttention2 [Lin24e] 将其扩展到 INT4/FP8 量化, SageAttention3 [Lin25e] 则在 Blackwell 消费级 GPU 上演示了 FP4 量化. 但这些方法主要面向消费级 GPU, 大部分 AI 计算部署在数据中心 GPU 上. 同时, FlashAttention-3 主要针对 NVIDIA Hopper H100 架构, 而 AI 行业已经迅速转向部署 B200 和 GB200 等基于 Blackwell 的数据中心系统 [Nvi24d]; 这一代 GPU 的性能特征有本质区别.

加速器演进中的一个明显趋势是, 不同硬件单元的扩展速度并不对称. Blackwell B200 的 Tensor Core 吞吐量是 Hopper H100 的两倍 (FP16/BF16 下为 2.25 PFLOPS, H100 为 1 PFLOPS), 但其他功能单元 (共享内存带宽、指数单元和整数/浮点 ALU) 扩展得更慢, 甚至没有变化. 结果是, 非 MMA 资源成了瓶颈. 我们的屋顶线分析 ([第 3.1 节](#section-03-01) 和[第 3.2 节](#section-03-02)) 表明, 对 Blackwell 上典型的注意力负载而言, 占主导的竟然是共享内存流量与指数运算, 二者所需时间比 MMA 计算高出 25-60%. Blackwell 还加入了新的架构特性: 每个 SM 有 256 KB Tensor Memory (TMEM), 用来存储 Tensor Core 的中间结果; MMA tile 为 $128 \times 128$ (面积是 Hopper $64 \times 128$ tile 的两倍); Tensor Core 操作完全异步, 结果直接写入 TMEM. 直接把现有注意力算法移植到新硬件上, 要么会损失大量性能, 要么会因为 Hopper MMA 指令不具备前向兼容性而根本无法运行.

为此, 我们提出 FlashAttention-4, 针对现代 GPU 架构中不断变化的瓶颈, 协同设计算法与 kernel 实现. 我们没有把硬件视为统一的计算资源, 而是明确找出非矩阵乘单元的瓶颈, 再用算法改动缓解这些问题:

1. **重新设计流水线以取得最大重叠:** 我们为前向与反向传播设计了新的软件流水线, 利用 Blackwell 的完全异步 MMA 操作和更大的 tile, 尽可能让 Tensor Core、softmax 计算和内存操作相互重叠.

2. **缓解指数单元瓶颈:** 在前向传播中, 我们用 FMA 单元上的多项式近似来模拟指数函数, 提高指数计算吞吐量. 我们还引入了条件 softmax 重缩放, 跳过不必要的重缩放操作.

3. **减少共享内存流量:** 在反向传播中, 我们用 Tensor Memory 存储更多中间结果, 从而减少共享内存流量. 我们还使用 Blackwell 的 2-CTA MMA 模式, 让每个 CTA 只暂存并加载操作数 B 的一半, 进一步减少共享内存流量; 在此基础上重构 dQ 步骤, 将原子归约的次数减半. 我们也实现了性能开销很小的确定性执行模式, 让强化学习应用中的训练可以复现.

4. **改进调度和资源分配:** 我们为 Blackwell 的资源限制与更大的 tile 设计了新的 CTA 调度策略和寄存器分配方案.

除了算法创新, 我们还完全使用嵌入 Python 的 CuTe-DSL 实现了 FlashAttention-4; 它保留了完整的表达能力, 编译速度则比传统的 C++ 模板方案快 20-30$\times$. 这个框架明显提高了开发效率并降低了使用门槛, 即使不熟悉 C++ 模板元编程, 研究人员也能快速构建和部署新的注意力变体.

为了用实验验证我们的方法, 我们在 B200 GPU 上测试 FlashAttention-4, 结果表明: (1) BF16 相比 cuDNN 最多加速 1.3$\times$, 相比 Triton 实现最多加速 2.7$\times$; (2) 在转移后的瓶颈资源上, 利用率接近峰值, 吞吐量最高约为 1600 TFLOPS (理论峰值的 71%); (3) 对长序列而言, FlashAttention-4 的性能优于其他注意力实现.

我们以宽松许可证开源 FlashAttention-4, 也在将它集成到常用库中, 方便更多研究人员与开发者使用. 代码位于 [https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute)

## 2 背景

### 2.1 多头注意力

设 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ 为单个注意力头对应的 query、key 和 value 输入序列, 其中 $N$ 是序列长度, $d$ 是 head dimension. 注意力输出 $\mathbf{O} \in \mathbb{R}^{N \times d}$ 的计算方式为:

$$
\begin{aligned}
\mathbf{S} &= \alpha \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N}, \\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \\
\mathbf{O} &= \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d},
\end{aligned}
$$

其中 $\mathrm{softmax}$ 按行计算, $\alpha = 1/\sqrt{d}$ 是缩放因子. 实际计算时, 为了保证数值稳定性, 我们会从 $\mathbf{S}$ 中减去 $\mathrm{rowmax}(\mathbf{S})$. 对多头注意力 (MHA), 每个注意力头都有一组独立的投影, 这些计算可以跨多个头和 batch 并行.

给定输出梯度 $\mathbf{dO} \in \mathbb{R}^{N \times d}$, 反向传播计算:

$$
\begin{aligned}
\mathbf{dV} &= \mathbf{P}^\top \mathbf{dO}, \quad \mathbf{dP} = \mathbf{dO} \mathbf{V}^\top, \\
\mathbf{dS} &= \mathrm{dsoftmax}(\mathbf{dP}), \\
\mathbf{dQ} &= \alpha \mathbf{dS} \mathbf{K}, \quad \mathbf{dK} = \alpha \mathbf{dS}^\top \mathbf{Q},
\end{aligned}
$$

其中 $\mathrm{dsoftmax}(\mathbf{dP})$ 表示按行计算的 softmax 梯度: 对 $p = \mathrm{softmax}(s)$, 有 $\mathbf{d}s = (\mathrm{diag}(p) - p p^\top)\mathbf{d}p$.

### 2.2 GPU 硬件特征与执行模型

这里介绍和 FlashAttention-4 有关的 GPU 执行模型, 重点是 NVIDIA Blackwell 架构 (B200 和 GB200). 我们会说明它与上一代 Hopper 架构的主要区别, 这些差异正是 FlashAttention-4 各项优化的出发点.

**内存层次:** GPU 内存由多个数据存储层级构成, 容量越大, 带宽通常越低. 全局内存 (GMEM), 也叫 HBM, 是芯片外 DRAM, 所有流式多处理器 (SM) 都可以访问. 来自 GMEM 的数据会自动缓存在芯片上的 L2 cache 中. 再往下, 每个 SM 内都有一块容量较小、由程序员管理且高度分 bank 的片上 cache, 称为共享内存 (SMEM). 最后是每个 SM 内的寄存器文件.

Blackwell 新增了一个叫作 *Tensor Memory* (TMEM) 的内存层级. 每个 SM 有 256 KB 片上 TMEM, 专门存储 Tensor Core 操作的中间结果. TMEM 与共享内存不同: 它按 warp 同步, 与 Tensor Core 紧密耦合, 矩阵乘加 (MMA) 单元可以把输出直接写入 TMEM, 无需占用寄存器. 这缓解了 Hopper kernel 中严重的寄存器压力, 也让更大的 tile 成为可能. TMEM 以 32 列 (16 KB) 为粒度分配, 程序员需要显式管理它的分配、释放和数据移动.

**线程层次:** GPU 编程模型以线程为基本执行单元, 再将执行单元组织成不同的逻辑层级. 从细到粗依次为线程、warp (32 个线程)、warpgroup (4 个连续的 warp)、threadblock (即协作线程数组, CTA)、threadblock cluster 和 grid. 同一 CTA 的线程会被共同调度到同一个 SM, 同一 cluster 内的 CTA 则会被共同调度到同一个 GPC. CTA 内的所有线程都能直接寻址 SMEM, 每个线程最多有 256 个私有寄存器 (RMEM).

**Tensor Core 与更强的异步性:** Blackwell 配备第五代 Tensor Core, 处理的 tile 比前几代明显更大. 每条 MMA Tensor Core 指令处理 $128 \times N$ tile (通常 $N =$ 128 或 256), Hopper 则处理 $64 \times N$. 更重要的是, Blackwell MMA 会异步地把输出直接写入 TMEM, Hopper MMA 则写入寄存器. 由于 MMA 单元不再等待寄存器回写, 这种完全异步的方式可以更好地重叠计算和其他操作.

硬件的异步支持使 warp 特化 kernel 成为可能: 一个 CTA 内的 warp 被分为 producer 或 consumer 角色, 各自只负责数据移动或计算 [Bau11].

**2-CTA Tensor Core:** Blackwell 支持 2-CTA Tensor Core MMA 模式. 在该模式下, 同一 thread block cluster 中的一对 CTA 协作执行一次 MMA, 操作可以读写两个 CTA 的 Tensor Memory. 这对 CTA 中由一个线程发起 MMA, 但配对 CTA 必须已启动, 并在操作执行期间保持活跃. 单 CTA MMA 把 M 维限制在 128, 配对模式则支持 M = 128 或 256: 它沿 M 维把 A tile 和累加器分给两个 CTA, 沿 N 维把 B tile 分给两个 CTA, 因而每个 CTA 只需在自己的共享内存中暂存 B 的一半, 硬件在矩阵乘时会使用组合后的完整 B tile. 这样可以减少重复的共享内存容量和带宽开销. 但这些操作会跨 CTA 对访问 Tensor Memory, 所以 kernel 必须以固定的二元组启动 CTA, 并且整个 kernel 中的 Tensor Memory 和 Tensor Core 操作都要使用一致的 2-CTA 模式.

**瓶颈转移:** Blackwell 体现了一个明显趋势: Tensor Core 吞吐量的扩展速度快于其他功能单元. Blackwell 的 FP16/BF16 Tensor Core 吞吐量是 Hopper 的两倍 (每块 GPU 为 2.25 PFLOPS [Nvi24d], Hopper 为 1 PFLOPS [Nvi22]), 但共享内存带宽和指数单元吞吐量没有变化, 或者扩展得更慢. 这种不均衡会把性能瓶颈从矩阵乘转移到共享内存流量和 softmax 等非矩阵乘操作. 正如[第 3.1 节](#section-03-01)和[第 3.2 节](#section-03-02)的屋顶线分析所示, kernel 需要精心设计, 尽量让 MMA 操作与这些瓶颈资源重叠.

B200 (以及 GB200) 上几个硬件组件的吞吐量如下.

1. Tensor Core: BF16 MMA 的吞吐量为每个 SM 每时钟周期 8192 次操作, 是 Hopper 每个 SM 每时钟周期 4096 次操作的两倍. 这个数值可以从理论最大 FLOPS 推出: 2.25 PFLOPS / 1850 Mhz 时钟频率 / 148 个 SM = 每个 SM 每时钟周期 8192 次操作.
2. 指数单元. B200 和 GB200 上的多功能单元 (MUFU) 每个 SM 每时钟周期可以执行 16 次操作, 与 Hopper 相同 [Nvi24c]. B300 和 GB300 GPU 已将指数吞吐量翻倍到每个 SM 每时钟周期 32 次操作, 不过写作本文时, 这些 GPU 尚未广泛部署.
3. SMEM: 读取吞吐量为每个 SM 每时钟周期 128 字节, 与 Hopper 相同; 该数值来自微基准测试 [Luo25h].

可以看到, Blackwell 的 MMA 吞吐量相比 Hopper 翻了一倍, 其他硬件单元却不一定按相同速度变快. 这反映了加速器设计中的一个普遍趋势: 在近似的功耗和芯片面积限制下, 提高最重要组件 (通常是矩阵乘单元) 的吞吐量, 以获得更高性能.

## 3 算法

<span id="section-03-01"></span>

### 3.1 注意力前向传播

我们先用屋顶线分析说明注意力前向传播的瓶颈, 由此引出新的流水线设计, 以及对 FlashAttention 算法的改动: 提高指数单元吞吐量, 并避免大部分 softmax 重缩放步骤.

#### 3.1.1 各单元吞吐分析

为了说明 kernel 设计和优化背后的直觉, 我们先根据矩阵乘单元 (Tensor Core)、共享内存 (SMEM) 和指数单元的吞吐量做屋顶线分析. 这是一个简化分析, 没有覆盖 GPU 的所有资源 (例如浮点运算、寄存器带宽和 L2 带宽). 尽管如此, 它仍能找出瓶颈.

设 $\mathbf{Q}$ 和 $\mathbf{K}$ 沿序列长度维度的 tile 形状为 $M \times N$, head dimension 为 $d$. 我们分析计算量与内存流量需求, 从而找出性能瓶颈.

**MMA 计算.** 前向传播的每次迭代执行两次矩阵乘加 (MMA): $\mathbf{Q}\mathbf{K}^\top$ (用 $M \times d$ 和 $d \times N$ 的输入计算 $M \times N$ 的输出), 以及 $\mathbf{P}\mathbf{V}$ (用 $M \times N$ 和 $N \times d$ 的输入计算 $M \times d$ 的输出). 每次 MMA 需要 $2MNd$ 次浮点运算. Tensor Core 每周期能执行 8192 FLOPs, 所以总计算时间为

<span id="equation-01"></span>

$$
T_{\mathrm{MMA}} = \frac{4MNd}{8192} \mathrm{\ cycles}.
$$

**共享内存流量.** 两次 MMA 中, 一次是 shared-shared (SS), 两个操作数都从共享内存读取 ($\mathbf{Q}\mathbf{K}^\top$); 另一次是 tensor-shared (TS), 操作数 $A$ 从 Tensor Memory 读取, 操作数 $B$ 从共享内存读取 ($\mathbf{P}\mathbf{V}$). 每条 MMA 指令处理 $128 \times 128$ tile, 所以计算 $M \times N$ 输出需要 $\lceil M/128 \rceil \times \lceil N/128 \rceil$ 条 MMA 指令. 关键在于, 需要多条 MMA 指令时, 共享内存中的操作数会被重复读取.

对于 $\mathbf{Q}\mathbf{K}^\top$ (SS), 计算 $M \times N$ 输出需要 $\lceil M/128 \rceil \times \lceil N/128 \rceil$ 条 MMA 指令; 每条指令从共享内存读取一块 $128 \times d$ 的 $\mathbf{Q}$ 和一块 $d \times 128$ 的 $\mathbf{K}^\top$. 共享内存总读取量为 $\lceil M/128 \rceil \times \lceil N/128 \rceil \times (128d + 128d) = \lceil M/128 \rceil \lceil N/128 \rceil \times 256d$ 个元素. 对于 $\mathbf{P}\mathbf{V}$ (TS), 计算 $M \times d$ 输出需要 $\lceil M/128 \rceil \times \lceil d/128 \rceil$ 条 MMA 指令; 每条指令从共享内存读取一块 $N \times 128$ 的 $\mathbf{V}$, 合计 $\lceil M/128 \rceil \times \lceil d/128 \rceil \times 128N$ 个元素. 每个元素占 2 字节 (bf16), 带宽为每周期 128 字节, 因而共享内存读取时间 ($T_{\mathrm{smem}}$) 为

<span id="equation-02"></span>

$$
= 2\Big\lceil\tfrac{M}{128}\Big\rceil\Big\lceil\tfrac{N}{128}\Big\rceil 256d
+ 2\Big\lceil\tfrac{M}{128}\Big\rceil\Big\lceil\tfrac{d}{128}\Big\rceil 128N
= \tfrac{3MNd}{8192}\ \mathrm{cycles}
$$

(假设 $M$、$N$、$d$ 都是 128 的倍数).

**指数单元.** 指数单元执行 softmax 所需的逐元素操作. 前向传播需要对 $M \times N$ 个值 (即注意力矩阵 $\mathbf{S}$) 计算指数. 指数单元每周期可以执行 16 次操作, 所需时间为

<span id="equation-03"></span>

$$
T_{\mathrm{exp}} = \frac{MN}{16} \mathrm{\ cycles}.
$$

[表 1](#table-01) 汇总了两种典型 tile 配置的分析结果. 当 $M = N = d = 128$ 时, 各资源比较均衡: 共享内存需要 768 个周期, 略低于 MMA 计算和指数单元的 1024 个周期. 对更大的 tile $M = 256, N = d = 128$, 由于 MMA 操作数会被重复读取, 共享内存流量增至 1536 个周期, MMA 计算和指数单元则翻倍到 2048 个周期. 这项分析说明 kernel 设计需要做到三点: (1) 使用大 tile, 尽量重叠 MMA 操作与 softmax 计算; (2) 使用其他硬件单元提高指数吞吐量; (3) 减少不必要的非矩阵乘操作时间.

<span id="figure-01"></span>

![FlashAttention-4 前向流水线](./flashattention-4/figure-01.png)

**图 1.** FlashAttention-4 前向流水线. 上标 $^H$ 表示与“高位” Q tile 对应的矩阵, 上标 $^L$ 表示与“低位” Q tile 对应的矩阵. 每个 Q tile 对应 128 个 query token.

<span id="table-01"></span>

![原论文表 1](./flashattention-4/table-01.png)

**表 1.** 注意力前向传播的屋顶线分析 (周期数). 对两种 tile 大小, 主要瓶颈都是 MMA 计算和指数单元.

#### 3.1.2 重叠矩阵乘和 softmax 的新流水线

Blackwell 的 Tensor Core FLOPS 再次翻倍, 因此, 与 Hopper 相比, 让 softmax 与 Tensor Core 操作重叠更加重要. 我们采用类似 FA-3 的乒乓调度, 每个 thread block 计算两个输出 tile. 一个 tile 执行 Tensor Core 操作时, 另一个 tile 计算 softmax. Hopper Tensor Core 将累加器保存在寄存器中, 每行由四个线程以交错模式处理; Blackwell Tensor Core 则把累加器保存在 Tensor Memory 中. 此外, Blackwell 的单个累加器 tile 大小为 128×128 个元素, Hopper 的 tile 大小是 64×128.

在这些 tile 之间分配工作的自然方式, 是使用两个各含 128 个线程的 warpgroup, 每个线程处理完整的一行. 这样就不再需要跨 warp shuffle 来归约行最大值, 每个线程也不必保存多份统计寄存器. 和 FA-3 一样, 我们显式同步两个 softmax warpgroup, 避免它们的关键区重叠; 这里的关键区指指数计算部分. 每个 softmax warpgroup 先把整行加载到寄存器, 再求最大值, 随后计算 softmax (减去最大值、重缩放、求指数、转换到输入精度), 最后计算行和.

FA-4 与 FA-3 的另一个区别是, $\mathbf{P}$ 通过 Tensor Memory 而不是寄存器文件传递, 所以我们可以把输出重缩放交给独立的“修正” warpgroup, 将其移出关键路径.

要让流水线重叠, Tensor Memory 可以采用几种不同的分区方式. 所有方案都要分配两个输出 tile 的空间; 当 head dimension 为 128 时, 剩余一半 Tensor Memory 用来存储 $\mathbf{S}$ 和 $\mathbf{P}$. 这部分内存可以存两份 $\mathbf{S}$ 或四份 $\mathbf{P}$ (假设 Tensor Core 的输入是 FP16 或 BF16). 因此, 剩余 Tensor Memory 大致有两种分法: 一块 $\mathbf{S}$ tile 和两块 $\mathbf{P}$ tile, 或者两块彼此与 $\mathbf{P}$ 重叠的 $\mathbf{S}$ tile. 我们选择后者, 因为这样一开始就能计算两块 $\mathbf{S}$ tile, 启动软件流水线. 此外还能留出一部分 Tensor Memory, 把重缩放统计信息传给修正 warpgroup.

Blackwell 的 tile 更大, 再加上所选的线程分工, 会带来一个问题: 除非从 Tensor Memory 重新加载, 否则寄存器中必须保存一整行 128 个元素. 我们用了两个 softmax warpgroup、一个修正 warpgroup, 以及一个负责驱动 Tensor Core 和 TMA 单元的 warpgroup, 因而必须给 softmax 分配足够的寄存器并避免寄存器 spill. 对 BF16 输入, 输入需要占 128 个寄存器, 输出可能还需要 64 个寄存器 (另有杂项和临时寄存器). 为了减轻寄存器压力, 我们分阶段存储 $\mathbf{P}$: 先存前三个四分之一块 (并触发对应的 MMA 操作), 再单独存最后一个四分之一块.

#### 3.1.3 指数函数模拟

**指数吞吐量瓶颈.** 现代 GPU 用多功能单元 (MUFU) 计算指数函数, 它的吞吐量远低于执行矩阵乘的 Tensor Core. 在 B200 和 GB200 GPU 上, MUFU 每个 SM 每时钟周期执行 16 次操作, 矩阵乘则为每个 SM 每时钟周期 8192 次操作. softmax 需要大量指数计算, 这种差距使指数函数成为注意力 kernel 的主要瓶颈.

**用多项式近似做软件模拟.** 为了提高指数吞吐量, 我们在浮点 FMA 单元上用软件模拟 $2^x$; FMA 单元可以与 MUFU 并行工作. 我们采用经典的区间缩减方法 (Cody-Waite), 再使用多项式近似 [Mul18]. 核心做法是把指数计算拆成:

<span id="equation-04"></span>

$$
2^x = 2^{\lfloor x \rfloor}\,2^{x-\lfloor x \rfloor}
$$

其中 $\lfloor x \rfloor$ 是整数部分, $x - \lfloor x \rfloor \in [0, 1)$ 是小数部分.

整数部分 $2^{\lfloor x \rfloor}$ 可以利用 IEEE 754 浮点表示的位结构高效计算. 指数字段直接表示 2 的幂, 所以计算 $2^{\lfloor x \rfloor}$ 等价于对指数位做移位和加法, 可以用整数 ALU 指令完成.

对于小数部分, 用多项式近似 $x_{\mathrm{frac}} \in [0, 1)$ 时的 $2^{x_{\mathrm{frac}}}$:

<span id="equation-05"></span>

$$
2^{x_{\mathrm{frac}}} \approx \sum_{i=0}^{n} p_i\, x_{\mathrm{frac}}^i
$$

其中 $p_0 = 1.0$, 其余系数用 Sollya 软件包 [Che10] 计算, 目标是在 $\lbrack 0, 1)$ 上最小化相对近似误差. 多项式用 Horner 法和 FMA 指令求值, 可以得到较高吞吐量.

完整算法如下:

- 将 $x$ 截断到不小于 $-127$, 避免下溢.
- 用向下舍入模式计算 $\lfloor x \rfloor$: 先给 $x$ 加上 $2^{23} + 2^{22}$ (把小数位压入尾数), 再以向下舍入模式减去该值.
- 计算小数部分: $x_{\mathrm{frac}} = x - \lfloor x \rfloor$.
- 计算多项式, 得到 $2^{x_{\mathrm{frac}}}$.
- 合并整数和小数部分: 把 $\lfloor x \rfloor$ 移入指数字段, 再加上 $2^{x_{\mathrm{frac}}}$ 的尾数位.

把指数计算分配到 MUFU 和 FMA 单元后, 指数吞吐量实际得到提高, 注意力计算中的这个瓶颈也随之缓解.

**部分模拟.** 多项式模拟可以提高指数吞吐量, 但也有代价: 中间值和系数需要额外寄存器, 寄存器带宽占用更高, 延迟也比 MUFU 指令长. 如果所有指数都用模拟计算, 寄存器压力会增大, 还可能发生 spill, 抵消吞吐量收益. 因此, 我们只对每个 softmax 行中的一部分元素 (10-25%) 使用模拟, 其余元素仍用硬件 `MUFU.EX2` 计算. 具体比例根据给定 tile 配置的 MMA 与指数吞吐量之比做经验调优.

**数值精度.** [表 2](#table-02) 比较了不同次数的多项式近似与硬件 `MUFU.EX2` 指令的精度, 测试使用 $[0, 1)$ 内 4M 个随机输入. 我们报告两个指标: FP32 级误差 (量化前) 和 BF16 级误差 (FP32 输出舍入到 BF16 后), 二者都以 FP64 结果为基准.

在 FP32 层面, 3 次多项式的最大相对误差为 $8.8 \times 10^{-5}$, 约为硬件的 $600\times$. 但舍入到 BF16 后, 两者误差几乎无法区分: BF16 的量化误差 (${\sim}3.9 \times 10^{-3}$) 大于所有次数 $\geq 3$ 的多项式近似误差. 对 99% 的输入, 3 次多项式与硬件结果相差不超过 1 BF16 ULP; softmax 输出以 BF16 精度使用, 所以这一精度足以用于注意力计算. 更高次多项式可以缩小 FP32 误差: 5 次多项式的最大相对误差在硬件的 $2\times$ 以内, 代价是每次求值多执行两条 FMA 指令.

<span id="table-02"></span>

![原论文表 2](./flashattention-4/table-02.png)

**表 2.** $[0, 1)$ 上 $2^x$ 的多项式模拟精度, 用 4M 个随机输入和 FP64 基准测量. FP32 列是多项式原始输出的误差; BF16 列是输出舍入到 BF16 后的误差. 对所有次数 $\geq 3$ 的多项式, BF16 量化误差都占主导.

#### 3.1.4 跳过在线 softmax 重缩放

**FlashAttention 在线 softmax.** FlashAttention 分块计算注意力 $\mathrm{softmax}(QK^\top)V$, 以减少内存流量. 为了保证数值稳定性, 算法在逐块处理时维护运行统计量. 计算第 $j$ 块时, 设 $S_j = Q K_j^\top$ 为该块的注意力分数. 在线 softmax 算法跟踪:

$$
\begin{aligned}
m_j &= \max(m_{j-1}, \mathrm{rowmax}(S_j)) \\
\ell_j &= e^{m_{j-1} - m_j} \ell_{j-1} + \mathrm{rowsum}(e^{S_j - m_j})
\end{aligned}
$$

其中 $m_j$ 是运行最大值, $\ell_j$ 是指数的运行和 (归一化因子). 中间输出 $O_j$ 更新为: $O_j = e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j.$ 重缩放因子 $e^{m_{j-1} - m_j}$ 会在遇到更大数值时重新归一化此前的结果, 从而保证数值稳定性.

**条件重缩放.** $e^{m_{j-1} - m_j} O_{j-1}$ 这一步需要一次向量乘法. 我们有两个简单观察:

1. 只有 $m_j > m_{j-1}$, 即找到新的更大值时, 才需要重缩放.
2. 重缩放可以容忍一定“余量”: 仅当 $m_j - m_{j-1} > \tau$ 时重缩放, 其中阈值 $\tau$ 通常设为 $\log_2(256) = 8.0$, 对应 256.0 的缩放因子. 只要跟踪统计量 (已经完成的总缩放量), 最后仍能得到真实分母和正确的最终输出.

在 FlashAttention-4 中, 我们把算法改为:

<span id="equation-06"></span>

$$
O_j = \begin{cases}
e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j & \mathrm{if}\ m_j - m_{j-1} > \tau \\
O_{j-1} + e^{S_j - m_{j-1}} V_j & \mathrm{otherwise}
\end{cases}
$$

当 $m_j - m_{j-1} \leq \tau$ 时, 我们不更新 $m$, 继续使用 $m_{j-1}$. 这种做法仍然正确, 因为计算结束时, 所有累加值都会用真实最大值 $m_{\mathrm{final}}$ 和最终归一化因子 $\ell_{\mathrm{final}}$ 重新归一化:

$$
\mathrm{Output} = \frac{1}{\ell_{\mathrm{final}}} O_{\mathrm{final}}
$$

这个改动明显减少了重缩放操作, 同时保持数值精度; 最后的归一化步骤会修正跳过中间重缩放产生的微小偏差.

实际实现中, 为了避免 warp 分歧, 只要 warp 内任一线程需要重缩放, 就对整个 warp 执行重缩放.

<span id="section-03-02"></span>

### 3.2 注意力反向传播

#### 3.2.1 各单元吞吐分析

与前向传播类似, 我们先根据矩阵乘单元 (Tensor Core)、共享内存 (SMEM) 和指数单元的吞吐量做屋顶线分析, 说明 kernel 设计和优化背后的直觉.

设 $\mathbf{Q}$ 和 $\mathbf{K}$ 沿序列长度维度的 tile 形状为 $M \times N$, head dimension 为 $d$. 我们分析计算量与内存流量需求, 从而找出性能瓶颈. 与前向传播不同, 为了简化 SMEM 周期数的公式, 我们假设 $M = N = d = 128$, 但仍保留变量名以便理解.

**MMA 计算.** 反向传播每次迭代执行五次矩阵乘加 (MMA). 每次 MMA 都涉及一个 $M \times N$ 矩阵、一个 $M \times d$ 矩阵和一个 $d \times N$ 矩阵 (三者中作为输出的矩阵不同), 需要 $2MNd$ 次浮点运算. Tensor Core 每周期可以执行 8192 FLOPs, 因而总计算时间为

<span id="equation-07"></span>

$$
T_{\mathrm{MMA}} = \frac{10MNd}{8192} \mathrm{\ cycles}.
$$

**共享内存流量.** 五次 MMA 中有三次是 shared-shared (SS) 操作, 两个操作数都从共享内存读取, 分别是 $\mathbf{S}^\top = \mathbf{K} \mathbf{Q}^\top$、$\mathbf{dP}^\top = \mathbf{V} \mathbf{dO}^\top$ 和 $\mathbf{dQ} = \mathbf{dS} \mathbf{K}$. 另外两次是 tensor-shared (TS) 操作, 操作数 $A$ 从 Tensor Memory 读取, 操作数 $B$ 从共享内存读取, 分别是 $\mathbf{dV} = \mathbf{P}^\top \mathbf{dO}$ 和 $\mathbf{dK} = \mathbf{dS}^\top \mathbf{Q}$. SS MMA 一共从共享内存读取 $2Md + 3Nd + MN$ 个元素, TS MMA 一共读取 $2Md$ 个元素. 共享内存带宽是每周期 128 字节, 每个元素占 2 字节 (bf16), 因而这部分开销为

<span id="equation-08"></span>

$$
T_{\mathrm{smem,MMA}} = \frac{4 M d + 3 N d + M N}{64} \mathrm{\ cycles}.
$$

此外, 算法会把大小为 $M \times N$ 的中间梯度 $\mathbf{dS}$ 以 bf16 写入共享内存, 需要 $2MN$ 字节, 即 $MN/64$ 个周期. 大小为 $M \times d$ 的梯度 $\mathbf{dQ}$ 会以 fp32 (每个元素 4 字节) 写入共享内存, 然后由 TMA 读回做归约, 共享内存总流量为 $8Md$ 字节, 即 $Md/16$ 个周期.

因此, 共享内存总访问时间 ($T_{\mathrm{smem}}$) 为

<span id="equation-09"></span>

$$
\frac{4 M d + 3 N d + M N}{64} + \frac{MN}{64} + \frac{Md}{16} \mathrm{\ cycles}.
$$

<span id="figure-02"></span>

![FlashAttention-4 反向传播计算图](./flashattention-4/figure-02.png)

**图 2.** FlashAttention-4 反向传播计算图 (5 次 MMA 操作 + 2 次逐元素操作), 图中给出了 1-CTA MMA 模式在 prologue、主循环和 tail 阶段的软件流水线顺序.

**指数单元.** 指数单元执行 softmax 及其梯度所需的逐元素操作 (指数、对数和相关非线性函数). 反向传播需要对 $M \times N$ 个值 (对应注意力矩阵 $\mathbf{S}$ 及相关项) 执行指数操作. 指数单元每周期可以执行 16 次操作, 所需时间为

<span id="equation-10"></span>

$$
T_{\mathrm{exp}} = \frac{MN}{16} \mathrm{\ cycles}.
$$

[表 3](#table-03) 汇总了典型 tile 配置 $M = N = d = 128$ 下的分析. 共享内存流量需要 3328 个周期, 高于 MMA 计算的 2560 个周期和指数单元的 1024 个周期. 这说明共享内存带宽是主要瓶颈, 不过问题没有全局内存流量主导时那么严重. 因此, kernel 设计需要尽量让 MMA 操作与其他计算重叠, 以隐藏共享内存延迟.

<span id="table-03"></span>

![原论文表 3](./flashattention-4/table-03.png)

**表 3.** $M = N = d = 128$ 时注意力反向传播的屋顶线分析. 共享内存流量是瓶颈, 所需时间比 MMA 计算约高 30%. 在 $M = 256$、$N = d = 128$ 的 2-CTA 配置中 ($\mathbf{dQ}$ MMA 是例外, 使用 $M = N = 128$、$d = 256$), 共享内存流量比 MMA 计算约高 5%.

#### 3.2.2 重叠矩阵乘和 softmax 的新流水线

FlashAttention 的反向传播执行五次 MMA: 一次用于重新计算 $\mathbf{S}$, 另外四次来自 $\mathbf{Q} \mathbf{K}$ 引出的两个梯度计算 ($\mathbf{dQ}$ 和 $\mathbf{dK}$), 以及 $\mathbf{P} \mathbf{V}$ 引出的两个梯度计算 ($\mathbf{dP}$ 和 $\mathbf{dV}$). 在 FA-3 中, 累加器存放在寄存器中, 而寄存器是一种有限资源. 这对操作顺序施加了很强的限制, 实际上把计算图串行化为 $\mathbf{S}, \mathbf{dP}, \mathbf{dV}, \mathbf{dQ}, \mathbf{dK}$; 只有 TMA load 可以明显偏离这个顺序执行. 除此之外, 算法基本相同: 它沿 KV 序列长度维度迭代, 并以相对于前向传播转置的形式计算数值, 因为 $\mathbf{dV}$ 和 $\mathbf{dK}$ 的梯度计算需要从 Tensor Memory 读取一个操作数. $\mathbf{dQ}$ 通过原子操作累加.

FA-4 中的 TMEM 提供了 FA-3 不具备的其他调度方案, 使 MMA 和非 MMA 操作可以明显重叠. 具体来说, 和前向传播一样, 我们想隐藏 softmax 计算延迟. FA-3 让 softmax 计算与 $\mathbf{dP}$ 的 MMA 重叠. 从上一节可知, 在 Blackwell 上至少要让两次 MMA 同时运行.

我们让上一轮迭代的 $\mathbf{dQ}$ 和 $\mathbf{dK}$ MMA 与当前计算重叠. 这要求仔细管理加载、MMA、计算和归约之间的共享内存与 Tensor Memory 资源. 特别需要注意, Tensor Memory 无法同时容纳五块累加器 tile. 它最多能容纳四块 128×128 元素的 tile, 而 $\mathbf{dV}$ 和 $\mathbf{dK}$ 都需要持续累加, 因而无法共用空间. 我们的实现让 $\mathbf{S}$ 和 $\mathbf{P}$ 共用一个 TMEM block (偏移 0), 让 $\mathbf{dP}$、$\mathbf{dS}$ 和 $\mathbf{dQ}$ 共用另一个 block. [图 2](#figure-02) 给出了 FA-4 反向传播的计算图.

<span id="figure-03"></span>

![2-CTA 反向传播 dQ 步骤](./flashattention-4/figure-03.png)

**图 3.** 在 2-CTA 反向传播的 $dQ$ 步骤中, CTA 对通过 DSMEM 交换 $dS$ tile 的一半, 每个 CTA 因此都能构造一个 $(\frac{M}{2} \times 2N)$ 操作数, 并用归约维度翻倍的 CTA-pair UMMA 完成计算.

#### 3.2.3 2-CTA 反向传播: 减少共享内存流量和全局原子加法

即使改进了流水线, 并把十个 GEMM 操作数中的两个保存在 Tensor Memory 中, 共享内存带宽仍主导反向传播. 在五次 GEMM 中, 余下八个 BF16 操作数都要从共享内存加载, 供 Tensor Core 使用; 这部分共享内存流量所需周期比 Tensor Core 计算约多 30%. 为了进一步缓解瓶颈, 我们采用 Blackwell 引入的 2-CTA MMA 模式, 沿 M 维切分输出累加器. 当 MMA tile 形状为 $M=256$、$N=K=128$ 时, 两个 CTA 共同组成一块更大的 tile: 每个 CTA 加载并暂存操作数 B 的一半, 只保留属于自己的累加器切片.

**共享内存流量.** 反向传播有五次 GEMM, 我们使用 $M=256$、$N=K=128$ 的 MMA tile 形状, 这样操作数 B 的共享内存流量大致可以减半. 在 FlashAttention 反向传播中, 每个 CTA 固定持有一块 KV tile (外层循环沿 $N$ 维分配给多个 CTA), 并在内层循环中流式处理 M tile. $\mathbf{dQ}$ 的累加是在外层循环中沿 KV 序列做归约, 但 2-CTA MMA 只切分输出 tile, 不切分归约轴; $\mathbf{dQ}$ MMA 的归约维度恰好是 $N$, 它自然地分布在 CTA 对上. 因此, 每个 CTA 仍需要为自己负责的行完成整个归约. 为解决归约轴上的冲突, 我们利用两个 CTA 位于同一 cluster 这一点, 通过分布式共享内存 (DSMEM) 交换一半 dS. 这会重新排列 $\mathbf{dS}$, 让它沿非归约轴切分: 每个 CTA 持有自己的 $\frac{M}{2}$ 行, 以及完整的 $2N$ 归约维度. 于是, 每个 CTA 的 $\mathbf{dQ}$ MMA tile 形状为 $(\frac{M}{2}, 2N)(2N, d)$, 在 Tensor Memory 中累加一块 $(\frac{M}{2}, d)$ tile. 在 2-CTA MMA 模式下, $\mathbf{S}$、$\mathbf{dP}$、$\mathbf{dV}$ 和 $\mathbf{dK}$ 的 MMA 使用 $M=256$ 的 tile; $\mathbf{dQ}$ 使用 $M=128$, 但归约维度翻倍为 $2N=256$. 随后, 我们相对于 1-CTA 变体重新排列软件流水线, 以隐藏 DSMEM 延迟. 当前 tile 的 $\mathbf{dP}$ 会在上一轮 tile 的 $\mathbf{dQ}$ 之前计算. $\mathbf{dQ}$ tile 足够小, 可以和 $\mathbf{P}$ 一同放入 TMEM, 复用 $\mathbf{S}$ 的 TMEM 区域; 因此, 我们不再像 1-CTA 模式那样让 $\mathbf{dP}$ 与 $\mathbf{dQ}$ 复用同一 TMEM 区域. 新的流水线顺序可以让当前 tile 的逐元素 $\mathbf{dS}$ 计算, 与上一轮 tile 的 $\mathbf{dQ}$ MMA 并行. [图 3](#figure-03) 展示了 $\mathbf{dQ}$ 步骤的分解方式.

**$\mathbf{dQ}$ 原子加法.** 这种 $\mathbf{dQ}$ 分解还有一个好处: 全局原子归约的次数减半. 原子更新会引入不确定性, 而且每次内层循环都要执行, 开销很大. 因此, 每个 CTA 只写入 $\mathbf{dQ}$ tile 的一半, 全局原子归约次数也只有 1-CTA 版本的一半.

#### 3.2.4 确定性反向传播

我们的反向 kernel 在全局内存中做跨 CTA 归约, 因此梯度计算会有不确定性 (通常影响 $\mathbf{dQ}$, 在 GQA 中还会影响 $\mathbf{dK}$/$\mathbf{dV}$). 为了让训练结果可复现, 并方便可靠地调试, 我们还提供了确定性执行模式. 常见做法是用信号量锁串行化全局归约, 我们也采用了这一方案. 具体来说, 向同一块 $\mathbf{dQ}$ tile 写入的每个 CTA 都要按预定顺序获取锁, 执行归约, 再递增信号量计数器以释放锁.

这种基于锁的方法会从两方面影响性能: (1) 为确保信号量写入在整个设备上可见, 必须发出内存 fence (正确的 acquire-release 语义需要这一步); (2) 每个 CTA 都要等待前一个归约同一块 $\mathbf{dQ}$ tile 的 CTA 完成, 因而产生停顿. 负载不均衡时, 随意选择 CTA 顺序会严重降低性能. 通常, 我们沿 head 和 batch 维度做 CTA swizzle, 减少停顿 (范围不超过 L2 cache 容量, 参见[第 3.3 节](#section-03-03)). 对因果掩码, 我们还会按降序启动 KV block, 从对角线开始按升序遍历 query block, 并按 query block 索引降序安排 $\mathbf{dQ}$ 归约. 这种“最短处理时间优先” (SPT) 调度保证 CTA 第一次写入 $\mathbf{dQ}$ 时无需等待.

<span id="section-03-03"></span>

### 3.3 调度

在因果掩码或可变序列长度 (varlen) 等许多场景中, 注意力 kernel 天然存在负载不均衡: SM 被分配到主循环长度不一的 worktile, 因为有些 worktile 需要执行更多 load 和 MMA. 此外, 我们还可以选择 SM 处理 tile 的顺序, 例如规定 grid 坐标的首选线性化方式. 抽去注意力的具体特征后, 相同并行处理器上的 makespan 最小化结论就可以应用到这里. FlashAttention-4 采用了经典的最长处理时间优先 (LPT) 调度 [Gra69]. 这里的应用方式适用于所有 GPU 架构, 在 Hopper GPU 上也已经验证可以改进 FlashAttention-3.

**因果掩码的 LPT.** 标准注意力 grid 表示为 (mblocks, heads, batches), 按从左到右的递增顺序计算. 但对角线上方的分数会被屏蔽, 所以对固定的 head 和 batch, SM 实际会低效地按 worktile 从短到长的顺序处理. 另一方面, 朴素 LPT 顺序也不是最优的: 不同 batch 的主循环 KV load 无法命中 L2 cache, 而先加载所有 KV head 时, 如果容量超出 L2 cache, 还会造成 cache thrashing. 因此, 我们始终把 batch 作为最外层维度, 并沿 head 做 swizzle. 具体做法是把 head 划成若干不会溢出 L2 cache 的区段; tile 调度器随后按区段内的 head、反向 mblock、区段、batch 的顺序遍历 grid. 对 MQA 或 GQA, 我们总会先遍历一个 KV head 对应的所有 query head, 再改变 mblock. 实验表明, 这个 LPT 顺序很有效: 例如在 H200 GPU 上, BF16、head dimension 128 时, MHA 的 FLOPS 提升 4-8%, MQA 8 提升 7-14%.

**可变序列长度的 LPT.** 对 varlen, batch 间的差异也会造成负载不均衡. 例如在 decode 负载中, 不同 batch 可能关注长度不同的上下文; 在混合或连续 batching 中, 一些 batch 可能执行 prefill, 另一些则执行 decode. 每个 batch 的 query 和 KV 序列长度列表通常作为注意力元数据保存在设备上; 标准 varlen 注意力 kernel 会在运行时读取这些整数, 并按 batch 递增顺序处理. 但给定的 batch 顺序对负载均衡而言可能很差, 例如先出现较短的方形 prefill, 后面才是长上下文 decode. 为了改善这一点, 我们可以先启动一个预处理 kernel, 按每个 worktile 的最大执行时间给 batch 排序, 从而强制采用 LPT 顺序; 该 kernel 还会写出从虚拟 batch 索引到实际 batch 索引的映射元数据, 后续注意力 kernel 读取映射后即可按排序后的顺序遍历 batch. 这份元数据可以缓存, 所以排序不会造成性能损失.

## 4 语言与框架

FlashAttention-4 完全使用嵌入 Python 的 CuTe-DSL [Nvi25d] 编写, 不包含任何 CUDA C++ 组件. CuTe-DSL 编译器接收 Python 源码, 将其 lower 到 PTX, 再由 PTX 编译器 (ptxas) 生成最终汇编代码 (SASS).

**简洁抽象下的完整表达能力.** CuTe-DSL 编程模型与 CUTLASS C++ 同构, 因此 FlashAttention-4 保留了底层 GPU 编程的完整表达能力, 同时可以用 Python 而不是 C++ 做元编程, 并获得快速 JIT 编译带来的效率. CuTe-DSL 也允许直接访问 PTX, 作为 escape hatch; 开发者可以实现所需的任意功能, 不受框架限制. 例如, 一些操作尚未完全暴露在 CuTe-DSL API 中 (后续版本会集成), 我们便使用自定义 PTX 指令序列. 这说明框架并没有把开发者限制在一小部分 GPU 功能内.

**通过 JIT 快速编译.** 以往的 FlashAttention 实现包含复杂的 C++ 模板元程序, 编译时间一直是瓶颈. CuTe-DSL 嵌入 Python 并采用即时 (JIT) 编译, 因而 FlashAttention-4 比传统 C++ 模板方案构建得更快. 如 [表 4](#table-04) 所示, 相比 FlashAttention-3, FlashAttention-4 将编译时间缩短了 20-30$\times$. 更短的迭代周期明显提高了开发效率, kernel 开发中的实验和调试也能更快完成.

<span id="table-04"></span>

![原论文表 4](./flashattention-4/table-04.png)

**表 4.** 单个 kernel 的编译时间: FA3 (C++ 模板) 与 FA4 (CuTe-DSL). FA2 和 FA3 通常需要为不同注意力变体预编译数百个 kernel.

**灵活性与易用性.** 这个基于 Python 的框架已经在实践中表现出灵活性: 开发者无需修改核心框架, 就在 FlashAttention-4 上构建了 FlexAttention 和 block-sparse 注意力变体. 由于使用门槛降低, 只具备几个月 GPU 编程经验的研究人员和工程师也能做出有价值的扩展, 不必深入掌握 C++ 模板元编程. 这种易用性加快了创新, 让注意力机制研究社区可以更快地探索新算法变体.

我们的目标是提供一个综合框架, 以一流性能构建各种注意力变体. FlashAttention-4 不要求从头实现每个注意力变体, 而是把通用功能拆成独立、可组合的 primitive. Block-sparse pattern、掩码策略、可变序列长度处理和工作调度等操作都作为相互正交的 primitive 暴露, 可以自由组合. 这种模块化设计让优化和新功能可以惠及框架上构建的所有注意力实现, 同时仍能编译为高效 GPU kernel, 达到最高性能.

## 5 实验评估

我们将 FlashAttention-4 的效率与多个开源和闭源 baseline 进行比较.

**注意力 benchmark.** 我们测量 FlashAttention-4 在不同序列长度和 head dimension 下的运行时间, 并与 PyTorch 标准实现、FlashAttention-2 [+fa3]、Triton (使用 B200 专用指令 [Til19])、Gluon (比 Triton 更底层、控制更细的 GPU 编程语言 [Tri24b]) 以及 cuDNN (针对 B200 GPU 优化的厂商库) 比较. 结果确认, FlashAttention-4 相比 cuDNN 9.13 最多加速 1.3$\times$, 相比 Triton 最多加速 2.7$\times$. FlashAttention-4 最高达到 1613 TFLOPs/s, 约为 B200 GPU 理论最大 TFLOPs/s 的 71%.

**Benchmark 设置.** 我们在 B200 GPU 上测量不同设置的运行时间: 是否使用因果掩码, head dimension 为 64、128 或 (192, 128), 输入为 BF16. 序列长度取 1k、2k、...、32k, batch size 则设为让 token 总数保持 32k. Hidden dimension 设为 2048, head dimension 为 64 或 128 (即 32 个 head 或 16 个 head). DeepSeek V3 [Dee24a] 使用 (192, 128) 配置, 对该配置, 我们使用 16 个 head, query dimension 为 192, key/value dimension 为 128. 前向传播 FLOPs 按 $4 \cdot \mathrm{seqlen}^2 \cdot \mathrm{head\ dimension} \cdot \mathrm{number\ of\ heads}$ 计算. 使用因果掩码时, 约有一半元素参与计算, 所以该数值除以 2. 反向传播 FLOPs 取前向传播的 2.5 倍 (前向传播有 2 次矩阵乘, 反向传播因重新计算而有 5 次矩阵乘).

### 5.1 前向传播

<span id="figure-04"></span>

![B200 上 head dimension 128 的前向传播 TFLOPS](./flashattention-4/figure-04.png)

**图 4.** B200 上 head dimension 128 的前向传播 TFLOPS (FP16/BF16). 左: 非因果注意力. 右: 因果注意力. 在不同序列长度下, FA4 相比 cuDNN 9.13.0 加速 1.1-1.3$\times$, 相比 Triton 加速 2.1-2.7$\times$. 自我们的实现首次发布以来, cuDNN 的新版本已吸收本文介绍的许多技术, 因而取得了接近 FA4 的性能.

[图 4](#figure-04) 和 [图 5](#figure-05) 给出了前向传播结果: FlashAttention-4 比 cuDNN 9.13 快 1.1-1.3$\times$, 比 Triton 快 2.1-2.7$\times$. 对中等和较长的序列 (4k 及以上), 在不同 head dimension 和因果掩码设置下, FlashAttention-4 的性能都持续高于所有 baseline. 因果场景的收益更大, 我们认为这是最长处理时间优先 (LPT) 调度器带来的效果.

<span id="figure-05"></span>

![因果注意力在 head dimension 192, 128 下的前向传播 TFLOPS](./flashattention-4/figure-05.png)

**图 5.** B200 上 cuDNN 与 FA4 的前向传播 TFLOPS 对比 (FP16/BF16), head dimension 为 (192, 128), 使用因果注意力 (DeepSeek V3 架构通常采用该配置).

### 5.2 反向传播

<span id="figure-06"></span>

![B200 上 head dimension 128 的反向传播 TFLOPS](./flashattention-4/figure-06.png)

**图 6.** B200 上 head dimension 128 的反向传播 TFLOPS (FP16/BF16). 左: 非因果注意力. 右: 因果注意力.

[图 6](#figure-06) 给出了反向传播结果. 在长序列和因果掩码设置下, FlashAttention-4 都能稳定加速, 说明 2-CTA 反向传播确实有效.

[图 7](#figure-07) 还给出了确定性反向传播的性能. 经过精心设计的 swizzle 和调度后, 确定性反向传播明显加快, 速度最高可以达到 1-CTA 非确定性反向传播的 75%.

<span id="figure-07"></span>

![因果注意力确定性反向传播的消融实验](./flashattention-4/figure-07.png)

**图 7.** B200 上 head dimension 128 的确定性反向传播消融实验 (FP16/BF16). 因果注意力: SPT、采用反向 mblock 顺序的 LPT、LPT, 以及不做 batch/head swizzle 的朴素方案.

## 6 讨论与结论

FlashAttention-4 处理的是硬件各单元扩展速度不对称的问题: Tensor Core 已经快到让主要瓶颈转移至共享内存流量和指数吞吐量, 因此需要协同设计算法与 kernel, 缓解这些限制. 我们围绕完全异步 MMA 重新设计流水线, 让 softmax 与采用更大 tile 的矩阵乘重叠; 又引入软件模拟指数函数和条件 softmax 重缩放, 减少非矩阵乘操作. 我们使用 Tensor Memory 和 2-CTA MMA 模式来减少共享内存流量. 此外, 2-CTA 还可以重构全局原子累加, 将全局原子加法次数减半. FlashAttention-4 完全用嵌入 Python 的 CuTe-DSL 实现, 在保留底层控制能力的同时, 编译速度比基于 C++ 模板的 kernel 快 20-30×. 虽然它针对 Blackwell GPU 优化, 但随着计算吞吐量继续比非矩阵乘单元增长得更快, 其中一些算法也可以扩展到其他加速器.

#### 致谢

我们感谢 Together AI、Meta、xAI 和 Princeton Language and Intelligence (PLI) 提供计算支持. 我们感谢 Schmidt Sciences AI2050 fellowship、Google ML and Systems Junior Faculty Awards 和 Google Research Scholar program 的支持. 我们还要感谢 Nvidia 的以下团队: CuDNN、TensorRT-LLM 和 Cutlass 团队一直参与讨论, 提供想法和反馈.

## 附录 A 实验与 benchmark 的补充细节

### A.1 系统与库

我们在 B100 180GB SXM6 (1000W) 上测试速度. 先预热 5 轮, 再重复 benchmark 10 次并取平均时间.

我们通常使用写作时 (2025 年 3 月) 的最新库版本. 具体版本如下:

- CUDA 13.1
- FlashAttention 2.8.3
- Triton 3.6
- PyTorch 2.10.0
- CuTe-DSL 4.4.1

对于 cuDNN, 正文比较了 cuDNN 9.13 和最新版本 cuDNN 9.19.1.2. 从 9.13 和 9.14 版开始 [Nvi25c], 我们与 cuDNN 团队合作, 将 FlashAttention-4 的一些技术纳入 cuDNN, 让这项工作惠及尽可能多的使用者.

### A.2 非因果确定性反向传播

为了完整起见, [图 8](#figure-08) 还给出了不使用因果掩码时确定性反向 kernel 的性能, 并与因果掩码结果并排展示.

<span id="figure-08"></span>

![使用和不使用因果掩码时确定性反向传播的消融实验](./flashattention-4/figure-08.png)

**图 8.** B200 上 head dimension 128 的确定性反向传播消融实验. 左: 采用 batch/head swizzle 与朴素方案的非因果注意力. 右: 因果注意力的 SPT、采用反向 mblock 顺序的 LPT、LPT, 以及不做 batch/head swizzle 的朴素方案.

[+equal]: 共同贡献

[+fa3]: FlashAttention-3 无法在 B200 上运行
