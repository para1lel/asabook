---
title: 'BitDecoding'
createTime: 2026/08/23 12:32:45
permalink: /papers/bitdecoding/
pageClass: paper-reading
---

> [Dayou Du](https://openalex.org/A5113196534) [+internship], [Shijie Cao](https://openalex.org/A5140976747) [+corresponding], [Jianyi Cheng](https://openalex.org/A5010206458), [Luo Mai](https://luomai.github.io/), [Ting Cao](https://openalex.org/A5061455084) 和 [Mao Yang](https://dblp.org/pid/89/1482-4.html). 论文于 2025 年 3 月 24 日首次提交至 arXiv; 当前版本为 v3. 本阅读版转录并翻译自 [BitDecoding: Unlocking Tensor Cores for Long-Context LLMs with Low-Bit KV Cache](https://arxiv.org/abs/2503.18773v3). <a href="/paper/bitdecoding.pdf" target="_blank">原始 PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2503.18773). [TeX 源码](https://export.arxiv.org/e-print/2503.18773v3). 精确的印刷版式和参考文献以原始 PDF 为准.

[+internship]: 部分工作在 Microsoft Research 实习期间完成.

[+corresponding]: 通讯作者.

## 摘要

长上下文大语言模型 (LLM) 的发展使不断扩大的键值 (KV) 缓存在自回归解码期间造成了明显的内存和带宽压力. 在保持精度的同时, KV 缓存量化 (例如 4-bit 或 2-bit) 可以减少内存占用, 但现有系统仅依赖 CUDA core 解码, 未能充分使用 GPU 上占主要算力的 Tensor Core, 因而效率不高.

我们提出 BitDecoding, 这是首个通过 CUDA core 与 Tensor Core 协同工作来高效解码低比特 KV 缓存的推理系统. BitDecoding 能自动构造适合 Tensor Core 的布局, 引入 warp 级反量化并行机制, 并通过查询变换、高性能 tensor-wise 与 channel-wise 量化以及软件流水化反量化 kernel 提供统一的系统支持, 从而执行混合精度计算. 针对不同架构的优化还利用了 Hopper 的 warpgroup tensor 指令和 Blackwell 的 NVFP4 (MXFP4) tensor 格式.

在 Blackwell、Hopper 和 Ampere GPU 上的评测表明, BitDecoding 相比 FP16 FlashDecoding-v2 平均加速 7.5$\times$, 使用 NVFP4 时在 Blackwell 上最高加速 8.6$\times$, 相比当前最佳方法最高加速 4.3$\times$. 在 128K 上下文的 LLaMA-3.1-8B 上, BitDecoding 将单 batch 解码延迟降低了 3$\times$. BitDecoding 已在 [https://github.com/OpenBitSys/BitDecoding](https://github.com/OpenBitSys/BitDecoding) 开源.

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) 处理**长上下文** [Pen23, Din24d, Tea24a] 的能力带来了图书摘要 [Cha23c]、多模态理解 [Yan23d] 和测试时扩展 [Dee25c, Ope25k] 等新能力. 但这些进展也带来了显著的内存与计算问题, 主要原因是长上下文场景中的键值 (KV) 缓存很大. 自回归解码期间, LLM 每生成一个 token 都要反复访问不断增长的缓存, 因而增加内存用量并拖慢解码. batch 越大, 问题越严重, 因为 KV 缓存会随并发查询数线性增长. 例如, 一个 7B 模型的参数大约需要 14GB GPU 内存, 但当上下文长度为 32K、batch size 为 8 时, 仅 KV 缓存就会占用 128GB GPU 内存 [Hoo24], 形成严重的内存瓶颈.

为缓解这一日益严重的瓶颈, **KV 缓存量化**成为一种可行方案. 降低 KV 缓存的 bit-width 可以减少内存开销并提高整体效率. 最近的量化算法表明, 低比特 KV 缓存可以保持较高精度.

QServe [Lin24a] 表明, 即使同时采用 4-bit 权重和 8-bit activation, 4-bit KV 缓存仍能提高 LLaMA-3、Qwen-1.5 等模型的吞吐量并保持较高精度.

后续研究 [Liu24c, Kan24, Su25c] 表明, 2-bit KV 缓存可以达到接近 fp16 的精度.

例如, Kivi [Liu24c] 在 LLaMA-2-7B-Chat 上使用 2-bit KV 缓存时, LongBench [Bai23] 精度仅下降 0.6%.

近期研究 [Zha24b, Tao24a] 还探索了 KV 缓存的 1-bit 量化, 并在特定条件下保持了可接受的精度.

这些结果说明, KV 缓存量化可以兼顾效率和精度, 适合部署长上下文 LLM.

*尽管节省了内存, 当前系统对低比特 KV 缓存的支持仍难以取得预期加速.* 现有实现 [Liu24c, Zha24e, Lin24a] 仍处于初步阶段, 且只适用于特定情况, 系统优化还有很大空间. 量化和反量化带来的开销是一项主要瓶颈. 尽管 KV 缓存是低比特的, 查询 (Q) 值和 attention score 仍保持高精度. 这会产生现有硬件无法原生支持的混合精度矩阵乘法 (mpGEMM), 因而必须在乘法前执行反量化. Ladder [Wan24e] 和 Marlin [Fra24] 等已有 mpGEMM kernel 面向低比特权重设计, 无法直接用于低比特 KV 缓存. 原因在于权重是*静态的, 可离线存储*, 而 KV 缓存是*动态的, 在线生成*. 在自回归解码中, 每个新生成的 token 都需要对低比特 KV 缓存执行量化、打包和反量化, 给 GPU kernel 设计带来明显的开销与复杂度, 如[图 1](#figure-01) 所示.

为解决这个问题, 我们的思路是让 Tensor Core 执行密集矩阵乘法, 同时让 CUDA core 高效完成 KV 缓存反量化. 以往工作要么使用彼此分离的 kernel, 要么在融合 attention 操作中仅使用 CUDA core, 因而没有充分利用 Tensor Core, 如[图 2](#figure-02) 所示. 我们的方法基于三个观察: 第一, 现代语言模型采用分组查询注意力 (GQA) 和多查询注意力 (MQA), 多个查询共享一组 key, 因而 Tensor Core 可以加速 self-attention 中的点积; 第二, 使用 Tensor Core 可以减轻 CUDA core 的计算压力, 让低比特操作执行得更高效; 最后, 较新的 GPU 架构提供了不同机制: Hopper 支持异步执行和 warp specialization, 使低比特操作能够与计算重叠 [Luo24b], 而 Blackwell 原生支持低精度格式 (例如 MXFP4), 减少了在线数据转换的需要和相关开销.

<span id="figure-01"></span>

![低比特权重与低比特 KV 缓存的混合精度矩阵乘法对比.](./bitdecoding/figure-01.png)

**图 1.** 低比特权重与低比特 KV 缓存的混合精度矩阵乘法对比. (a) 量化权重可以离线预处理. (b) 每生成一个新 token, KV 缓存都需要在线量化和打包.

使用 Tensor Core 解码低比特 KV 缓存时, 要想获得高效率并不容易. 第一, Tensor Core 要求反量化后的低比特数据与高精度格式对齐, 但在自回归解码中, KV 缓存动态增长且必须符合 Tensor Core 的特定布局, 因而很难对齐. 布局未经过优化时, Tensor Core 的利用率可能很低, 甚至会产生错误结果. 第二, 高昂的反量化成本会阻塞 Tensor Core 执行, CUDA core 与 Tensor Core 之间的 workload 不匹配还会降低 GPU occupancy. 第三, 不同 attention 机制和量化算法采用不同的 tensor-wise 与 channel-wise scaling, 因而需要一种通用且高度优化的实现来支持各类低比特 KV 缓存. 设计不当时, CUDA core 或 Tensor Core 会在长上下文生成期间成为性能瓶颈.

针对这些问题, 我们设计并实现了 **BitDecoding**, 一个使用低比特 KV 缓存的高性能长上下文 LLM 推理系统. BitDecoding 包含多项使用 Tensor Core 所需的设计: (i) 根据硬件指令构造低比特优化布局; (ii) 将 warp 与 residual buffer 对齐以跑满 Tensor Core; (iii) 重新映射布局以加速反量化; (iv) 协调量化与反量化 kernel. 我们还提出了新的 GPU warp 并行策略来降低低比特操作开销, 包括 (i) 高效的 warp 并行布局, 以及 (ii) 利用 GPU 内存层次改进 attention 算法, 实现快速 warp 同步.

我们还为 BitDecoding 的 LLM 推理实现了以下技术: (i) query transformation, 它能高效执行多种 attention 变体, 便于将 BitDecoding 接入现有 LLM; (ii) 同时支持 channel-wise 和 tensor-wise scaling 的高性能量化 kernel, 适用于不同量化算法; (iii) 采用软件定义流水线的反量化 kernel, 它协调 CUDA core 与 Tensor Core 完成 GEMM 和反量化, 并将额外低比特元数据等数据搬运过程重叠起来; 此外, BitDecoding 还包含架构专用优化, 使用 Hopper 的 warpgroup tensor 操作和 Blackwell 原生低精度 tensor 格式, 在新一代 GPU 上取得尽可能高的解码性能.

我们在 Blackwell、Hopper、Ada 和 Ampere GPU 架构上分别进行了 kernel 级与端到端评测. 在 kernel 层, BitDecoding 相比 FP16 FlashDecoding-v2 在 Blackwell 上最高加速 8.6$\times$ (例如使用原生 MXFP4 格式的 RTX 5090), 在 Hopper 上最高加速 8.0$\times$, 在 Ada 上最高加速 7.5$\times$, 在 Ampere 上最高加速 4.8$\times$; 相比 QServe 则最高加速 4.3$\times$. 在端到端模型层, 对于序列长度为 128K 的 LLaMA-3.1-8B, BitDecoding 将单 batch 解码延迟降低了 3$\times$, serving throughput 比 QServe 高出 4$\times$ 以上.

<span id="section-2"></span>

## 2 背景与动机

<span id="figure-02"></span>

![不同低比特 KV 缓存系统与半精度 FlashAttention 的对比.](./bitdecoding/figure-02.png)

**图 2.** 不同低比特 KV 缓存系统与半精度 FlashAttention 的对比. 每个系统都采用 attention 公式 $\mathrm{Out}=\mathrm{softmax}(Q\,\mathcal{D}(K'^\top))\,\mathcal{D}(V')$, 其中 $K'$ 和 $V'$ 是低比特量化后的 Key 与 Value tensor, $\mathcal{D}(\cdot)$ 表示反量化函数.

**LLM 推理与低比特 KV 缓存.** LLM 推理分为两个阶段: (i) *Prefill*, 处理 prompt 并计算用于缓存的 Key (K) 与 Value (V) tensor; (ii) *Decode*, 在自回归生成过程中逐 token 更新 KV 缓存. 对于一个有 $n$ 层、$h_{kv}$ 个 KV head、hidden size 为 $d$ 的模型, KV 缓存需要 $2 \cdot 16 \cdot n \cdot h_{kv} \cdot d \cdot b \cdot l$ bit (假设使用 FP16), 其中 $b$ 是 batch size, $l$ 是序列长度. 这项需求会随 $b$ 和 $l$ 线性增长, 因此 KV 缓存往往占据大部分内存, 长上下文与大 batch workload 尤其如此. 在 batch 推理中, 每个序列都有独立的历史上下文, 加载缓存的 Key 和 Value 时几乎无法利用 batch 级并行或复用; *因此, KV 缓存访问通常受内存带宽限制*. 为减少内存占用、提高吞吐量, 同时保持接近非量化 baseline 的精度, 研究界和工业界已广泛探索更低比特的 KV 缓存 [Liu24c, Hoo24, Zha24b].

**现代 GPU 上的 Tensor Core 与 CUDA core.** 在 GPU 上优化 LLM 推理和低比特 KV 缓存时, 必须同时利用 Tensor Core 和 CUDA core. Tensor Core 提供现代 GPU 的大部分计算 FLOPS, 但专用于矩阵操作 (例如 GEMM); CUDA core 则能灵活执行向量、标量和控制流操作, 峰值 FLOPS 明显较低. 例如, A100 的 Tensor Core 在 FP16/BF16 下最高可达 312 TFLOPS, 远高于 CUDA core 的 19.5 TFLOPS FP32 性能.

最近几代 GPU 进一步拉大了这一性能差距. Hopper 架构引入 Warpgroup Matrix Multiply-Accumulate (WGMMA) 指令和 warp-specialized pipeline, 以提高异步执行效率. Blackwell 架构原生支持 micro-scaling 格式 (例如 MXFP4、NVFP4), 最高可达 20 PFLOPS, 使这一差距更加明显.

为了加速 LLM 推理, 大量工作都在优化 attention 变体以使用 Tensor Core. 当前最佳的 LLM [Dee24a, Dub24, Yan25a] 越来越多地采用 MQA [Sha19] 和 GQA [Ain23], 通过让多个 query 复用 KV head 来降低内存带宽需求. 这种复用会提高 arithmetic intensity 和计算效率 [Sun25e], 很适合高吞吐、以矩阵计算为中心的 Tensor Core. 因此, 在长上下文和 grouped-attention LLM 中, 使用 Tensor Core 已成为高效推理的重要条件.

**现有低比特 KV 缓存系统的局限.** 为支持长上下文 LLM 推理中的低比特 KV 缓存, 已有多个系统被提出 [Liu24c, Lin24a, Zha24e]. 但它们通常没有充分利用 GPU, 性能不够理想. 主要原因如下.

- *采用独立低比特 KV 缓存 kernel 的 attention:* 最直接的方法以 Kivi [Liu24c] 为代表, 它把混合精度 attention 拆成多个独立 kernel, 并嵌入未融合的 attention 实现. 这种设计非常灵活, 很容易支持多种 attention 变体 [Ain23, Sha19]. 但各个 kernel 独立启动, 会反复读写中间数据, 增加 global memory traffic, 并破坏 on-chip data reuse. 结果是启动开销高、内存带宽压力大、实际吞吐量低.
- *仅在 CUDA core 上运行低比特 KV 缓存 kernel 的融合 attention:* CUDA core 可以通用地执行混合精度操作, 因而自然可以把 FlashAttention 风格的融合 [Dao24a] 扩展成仅使用 CUDA core 的低比特 KV 缓存实现. 这种实现比未融合设计更快, 但仍没有充分利用 Tensor Core. 在这些系统中, 反量化和矩阵操作 (GEMV/GEMM) 都通过 fused multiply-add (FMA) 指令在 CUDA core 上执行. 在混合精度下, CUDA core 必须处理代价很高的反量化 (例如 int4/8 $\rightarrow$ FP16/BF16)、scaling 和 element-wise 操作; 这些任务受内存限制, 会占用 instruction slot、register bandwidth 和 L1/L2 容量. 因此 occupancy 和 tile size 都会受到限制, 留给计算密集型矩阵乘法的资源也更少. 所以, 在 CUDA core 上同时运行反量化和矩阵乘法会带来很大开销, arithmetic intensity 较高的 attention 变体尤其明显.

<span id="section-3"></span>

## 3 方案与挑战

<span id="section-3-1"></span>

### 3.1 方案: 协同使用 Tensor Core 与 CUDA core

本文探索一种在长上下文 LLM 推理期间*协同*使用 Tensor Core 与 CUDA core 来支持低比特 KV 缓存的方案. 我们的新设计与实现会 (i) 在 Tensor Core 上构造并调度矩阵乘法, (ii) 在 CUDA core 上高效执行量化、打包和反量化等非矩阵乘法操作. 为了让这种协同有效, 我们平衡 Tensor Core 与 CUDA core 的 workload, 并细致安排数据搬运, 使反量化可以持续向 Tensor Core GEMM 供给数据而不产生 stall, 同时尽量减少 memory traffic, 提高端到端解码吞吐量.

为了便于广泛采用, 我们希望把这一协同设计实现成一个系统, 它 (i) 能够支持多种 attention 变体 (包括 MHA、MQA 和 GQA) 的低比特 KV 缓存, (ii) 能够跨越多代 GPU. 前一项要求系统提供易于接入现有 attention 实现的简洁接口; 后一项要求设计易于适配, 能快速面向不同 GPU backend 开发, 同时保持较高解码吞吐量.

我们预计这一方案能带来明显收益. 例如, 在 FlashAttention-3 (FA-3) [Sha24b] 上实现低比特解码后, 我们可以使用 SM90 专有的 warp-specialized pipeline 等功能, 相比以往实现最高加速 $6\times$, 并避免旧版 SM80 指令造成的 35% 吞吐损失. 这一设计也考虑了 Blackwell 的架构能力, 其原生低精度格式支持将进一步提高吞吐量.

<span id="section-3-2"></span>

### 3.2 尚待解决的挑战

这种方案很有潜力, 但要让 Tensor Core 和 CUDA core 在低比特 KV 缓存上*协同*工作, 实现起来仍有几个难点:

<span id="figure-03"></span>

![FP16 与 INT4 fragment 布局所产生的低比特布局不匹配.](./bitdecoding/figure-03.png)

**图 3.** (a) 矩阵 B 的 `mma.m16n8k16` fragment 布局. 指令定义的交错映射会把一组特定值分配给每个 thread ($T_i$). (b) 对于 INT4, 量化会按 thread 连续打包数值. 反量化后, 该布局与预期的交错模式不匹配.

**挑战 1: Tensor Core 经常遇到低比特布局不匹配.** 低比特数据布局很难满足 Tensor Core 的要求, 在 KV 缓存动态扩展的自回归生成中尤其如此.

运行时经过量化和打包后, 低比特 KV 缓存必须反量化为符合 Tensor Core 要求的半精度布局. 这种匹配之所以困难, 有三个原因.

第一, 不同指令和不同代 GPU 的 fragment 布局并不相同. 使用优化后的数据搬运指令 `ldmatrix` 后, register 中的 fragment 会采用严格的 value-to-thread 映射. [图 3a](#figure-03) 给出了 `mma.m16n8k16` 沿 $N$ 维重复 tiling 时, 每个 thread ($T$) 读取的 register. 但其他 Tensor Core 指令 (例如 `mma.m16n8k8`) 以及 Hopper 的 `wgmma` 指令族 (例如 `wgmma.m64n64k16`) 使用不同的映射.

第二, 更低的 bitwidth 会加剧对齐问题. Tensor Core 指令虽然要求特定的计算类型, 但其严格的交错 register 布局使低精度数据很难直接匹配. 如果不做布局变换, 低比特 register 布局就会因为不符合交错访问模式而成为 MMA 无法执行的**无效布局**. 如[图 3b](#figure-03) 所示, Thread 0 (T0) 最初计算的两个 FP16 值可能在 KV 缓存中被量化并打包成八个连续低比特值; 解包和反量化后, 它们不再对齐预期的 Tensor Core register 布局, 因而产生错误数值. 即使 Blackwell 原生支持低精度格式, 硬件支持仍然有限, KV 缓存尤其如此: 它仍需要持续量化和打包, 软件必须谨慎处理低精度数值和 micro-scaling factor [Nvi25e].

最后, 反量化可能成为执行瓶颈: 直接执行 low-bit$\rightarrow$FP16 cast 很慢 [Kim22], 要想高效运行还需要**适合的布局**. Ladder [Wan24e]、Marlin [Fra24] 等以往工作会为静态权重插入独立的布局变换 kernel 来缓解不匹配, 但这会带来很大开销, 并不适合动态解码. 实验细节见[表 2](#table-02).

<span id="figure-04"></span>

![原始 warp 设计以及有无反量化时的微观对比.](./bitdecoding/figure-04.png)

**图 4.** (a) 沿 $N$ 维使用单个 warp 执行 register 级操作时, 反量化 (DQ) 会造成 stall. (b) 有无反量化时的微观对比.

**挑战 2: 频繁 stall 限制 Tensor Core 利用率.** 我们发现, 高性能 attention kernel 通过经验调优得到的 warp 布局与划分方式, 往往会无意中降低低比特 KV 缓存的性能.

在 FlashAttention 的原始 warp 划分方式下, 额外的反量化 (DQ) 会显著降低吞吐量和 Tensor Core 利用率. 如[图 4a](#figure-04) 所示, FlashAttention 沿 $N$ 维分配单个 warp 来执行 register 级 softmax 和矩阵乘法 $P V$, $P$ 保存在符合 Tensor Core 布局的 register 中. 在矩阵乘法前插入 DQ 后, 这种策略效率很低: $K$ 或 $V$ 的小 warp tile 必须沿 $N$ 维依次遍历, 因此 DQ 经常使 warp stall. [图 4b](#figure-04) 中的 Nsight Compute profiling [Nvi25a] 证实, 新增的 DQ 开销会增加 memory-access stall, 降低计算吞吐量和 Tensor Core 利用率, 与以往观察 [Fan25e] 一致.

此外, 原生低精度格式虽然不需要反量化, 但也会引入自己的开销. 具体来说, 为了让第二次矩阵乘法 ($P V$) 使用低精度 Tensor Core, 必须在 softmax 后动态重新量化概率矩阵 $P$: $P_{f16}=\mathrm{softmax}(Q_{f4}K_{f4}^\top), \quad O_{f16}=\mathrm{Quant}(P_{f16})V_{f4}$. 这种在线量化会形成新的计算瓶颈, 同样可能阻塞 Tensor Core 执行.

**挑战 3: 缺少适用于不同低比特 KV 缓存方法的通用系统优化.** 常用 KV 缓存量化方法会为 Key tensor 采用不同的 scaling granularity, 包括 tensor-wise [Zha24e, Hoo24] 和 channel-wise [Liu24c, Kan24], 因而很难构建统一支持所有方法的系统. 在线量化和打包需要执行 reduction 与 element-wise transform, 带来不可忽略的运行开销. scale 与 zero-point 等辅助元数据还会增加 memory traffic; 若调度不当, 会打断 load-compute pipeline. 以往的混合精度 kernel 优化 [Wan24e, Fra24] 面向静态权重量化, 无法适用于 KV 缓存动态、逐步生成的特点. 目前仍然缺少可通用于高性能低比特 KV 缓存量化的系统级优化技术.

<span id="section-4"></span>

## 4 BitDecoding 设计

本节介绍 BitDecoding 系统的设计, 它通过 Tensor Core 和 CUDA core 的协同工作来支持低比特 KV 缓存. 设计主要包含 (i) 使用 Tensor Core 时优化低比特布局的新方法与原则, (ii) 并行和协调 GPU warp 的新策略, 用于减少反量化造成的 stall.

<span id="figure-05"></span>

![面向 Tensor Core 优化低比特布局的方法概览.](./bitdecoding/figure-05.png)

**图 5.** 面向 Tensor Core 优化低比特布局的方法概览. (1) 在 Tensor Core fragment 内融合计算和量化. (2) 低比特打包数据保留 FP16 数值. (3) 低比特布局与反量化后的半精度布局匹配. (4) 重新映射布局以加速反量化.

<span id="section-4-1"></span>

### 4.1 在 Tensor Core 上优化低比特布局的方法

本设计首先要让 BitDecoding 能够自动生成优化后的布局, 以便在不同 GPU 代际和不同低比特 KV 缓存配置下充分使用 Tensor Core. 为此, 我们采用以下原则与方法:

**(1) 利用硬件指令构造低比特优化布局.** 我们的设计来自一个新观察: `ldmatrix` 的 thread-to-register 映射会按照 Tensor Core 的交错 fragment 布局加载数据. 如[图 5](#figure-05)-(2) 所示, 如果每个 thread 随后在本地执行量化和打包, 产生的低比特打包会*隐式保留*半精度 (FP16) 交错布局. 解包和反量化后, 数值已经与 Tensor Core register 匹配, 不需要全局 reshape. 因此, 我们没有采用以往方法中通过手工实现 [Fra24] 或迭代搜索 [Wan24e] 进行的重量级全局变换, 而是在计算过程中利用硬件指令自动构造有效的低比特打包布局. 这种重新映射没有额外开销, 可以配合 Tensor Core 执行, 也不会产生额外数据搬运.

基于这一观察, 我们设计了专用 GPU *Residual Kernel*, 对新生成的 FP16 KV tensor 融合执行计算、量化和打包. 我们使用 `ldmatrix` 将高精度 KV tensor 加载到按 Tensor Core 方式组织的 register 中, 执行矩阵操作 (例如 $Q K^\top$ 或 $P V$), 随后让每个 thread 在 register 中量化并打包自己负责的部分 (见[图 5](#figure-05)-(1)). 产生的低比特数据采用交错且兼容该布局的形式, 直接写入 global memory 来更新低比特 KV 缓存.

为了读取该缓存, 我们引入了融合反量化与计算的 *Packing Kernel*. 为保证解包时 register 布局正确, 它会复用 Residual Kernel 的指令配置, 即 (i) 使用相同的 `ldmatrix` 变体, (ii) 使用相同的 `mma` 变体和 warp tiling 配置. 因此, Packing Kernel 通过 `ldmatrix` 加载打包后的低比特数据时, 解包后的值天然对齐 Tensor Core register, 可以立即参与矩阵乘法, 无需显式修正布局.

**(2) 将 warp 与 residual KV 缓存对齐以跑满 Tensor Core.** Tensor Core 以 warp tile 的形式执行矩阵操作, 只有输入 tile 填满时才能达到最佳吞吐量. 据此, *我们的思路*是分配一个大小与 Tensor Core tiling 容量相符的 residual buffer, 使低比特数据符合硬件的计算粒度, 从而充分利用计算单元.

为了实现这一点, 我们引入 residual block size 为 $N_r$ 的半精度 residual KV 缓存. 设 $X \in \mathbb{R}^{L \times d}$ 表示整个 KV 缓存. 我们将 $X$ 划分为:

$$
X=X_{\mathrm{pack}}\cup X_{\mathrm{res}},
$$

其中

$$
\begin{cases}
X_{\mathrm{pack}}=X[:L-N_r] \\
X_{\mathrm{res}}=X[L-N_r:]
\end{cases}
$$

我们用 $\beta$ 表示低比特量化的 bit-width (例如 $\beta=4$ 或 $2$), 用 $\omega$ 表示打包存储所用的 word size (例如 INT16 的 $\omega=16$). 对应的*打包比例*为 $R=\omega/\beta$. 设 $W_n$ 为沿 N 维的 warp 数, $P_n$ 为每个 warp tile 处理的元素数 (例如 `mma.m16n8k16` 下 $P_n=8$). 为了让每个 warp 对应的 Tensor Core fragment 都被填满, residual block size 按下式计算:

<span id="equation-01"></span>

$$
N_r=P_n\times W_n\times R
$$

这能保证低比特 KV 缓存 fragment 与 Tensor Core 操作的 warp 级 tiling 精确对齐, 以紧凑且兼容布局的形式打包, 并尽可能提高计算单元 occupancy.

**(3) 重新映射布局以加速反量化.** 该布局虽然兼容 Tensor Core, 但直接使用 `static_cast` 把低比特值转换成 FP16 会产生很大开销, 因而反量化效率仍然不高.

为降低这项开销, 我们参考 [Kim22] 的底层 bitwise operation 与指令, 进一步设计了更快的反量化映射方法. 使用 `ldmatrix` 把打包数据加载到 register 后, 我们先将其转换成 INT32, 再按照 75316420 模式映射到交错的 Tensor Core 布局. 该布局可以利用 `lop3` 指令执行 bitwise manipulation, 高效地将 INT4/INT2 数据转换成 FP16, 同时符合 Tensor Core 的计算模式.

**(4) 通过配置设置协调 Residual Kernel 与 Packing Kernel.** 该设计使用统一的指令配置来协调 Residual Kernel 和 Packing Kernel. 首先, 可根据 GPU 架构确定包括 `ldmatrix` 和 `mma` 变体在内的硬件指令配置. 根据该配置, 再按照低比特 KV 缓存的 bit-width 计算 residual block size $N_r$. 如[图 5](#figure-05) 所示, Residual Kernel 通过 `ldmatrix` 将高精度 KV 条目加载到 register, 使用 Tensor Core 计算, 然后融合量化和打包, 再把结果存入低比特 KV 缓存. Packing Kernel 使用相同的指令配置, 将打包数据加载到 register, 高效执行反量化, 随后继续进行 Tensor Core 计算.

<span id="section-4-2"></span>

### 4.2 warp 并行策略

第二项挑战是避免 BitDecoding 重蹈现有混合精度 attention warp 并行策略的覆辙; 这些策略会因频繁的 warp stall 而降低硬件利用率. 我们观察到, 低比特数据的搬运带宽远高于全精度数据, 因而瓶颈会从内存转向计算. 据此, 我们设计了一种利用 GPU 内存层次来高效并行低精度操作的 warp 布局, 它尽量减少数据搬运并显著提高 Tensor Core 利用率 ([表 3](#table-03) 显示其开销很小).

**(1) 提高低精度操作的 warp 并行度.** 我们引入一种新的 warp 布局, 使多个打包数据块可以并行操作. 以反量化为例, 我们调整 warp 划分策略来更好地利用并行性. 如[图 6](#figure-06) 所示, 原策略沿 $M$ 维分配多个 warp; 我们考虑到解码查询长度通常很小 ($<16$), 将分配限制为 $W_m=1$, 并重新分配资源以增加沿 $N$ 维的 warp 数 ($W_n$).

增加 $W_n$ 后, 多个 warp 会先并发反量化打包数据, 再执行基于 Tensor Core 的矩阵乘法, 因此 Streaming Multiprocessor (SM) 的 warp scheduler [San15] 可以有效缓解反量化 stall.

同样, 这种并行策略也能减少原生低精度 attention 在线量化造成的 stall, 避免量化或反量化成为串行瓶颈.

<span id="figure-06"></span>

![提高 Tensor Core 利用率的 warp 布局与协同 softmax 设计.](./bitdecoding/figure-06.png)

**图 6.** 通过以下方法提高并行度并高效使用 Tensor Core: (1) 新 warp 布局减少反量化 stall; (2) 协同 softmax 利用 GPU register 与 shared memory 间的数据搬运执行 cross-warp reduction, 开销很小.

**(2) 利用内存层次进行 warp 同步.** 但计算结果现在分散在不同的 register 和 warp 中, 原有的 register 级 softmax 无法继续使用. 而且, 新 warp 布局与 $P V$ 的 MMA 操作所需格式不兼容, 由此产生了*新的难题*.

为此, 我们利用由 register 和 shared memory 组成的多级内存层次, 为 softmax 计算执行 cross-warp reduction 与同步. 如算法 1 所示, 我们在 FlashAttention 等现有高性能 attention 算法上增加两个 shared memory buffer: $\mathit{sTMP} \in \mathbb{R}^{W_n}$ 和 $\mathit{sAcc} \in \mathbb{R}^{T_m \times T_n}$. $\mathit{sTMP}$ buffer 用于执行 cross-warp reduction, 计算 softmax 的 row-wise maximum. 具体做法是先在 register 中进行 intra-warp reduction, 再通过 shared memory 进行 inter-warp reduction. $\mathit{sAcc}$ buffer 暂存 Tensor Core register 中计算出的 attention score $P$, 随后通过 `ldmatrix` 重新加载, 为后续 Tensor Core `mma` 操作提供正确的对齐方式.

由于 $W_n$ 通常很小, 我们复用 $\mathit{sTMP}$ 的 shared memory pointer 来存放 $\mathit{sAcc}$, 从而减少内存开销. 在 Hopper Tensor Core 上, WGMMA 还支持直接访问 shared memory, 无需显式地把数据从 shared memory 搬到 register.

**算法 1: 多 warp 协同 Softmax**

- **输入:** SMEM 中的 $\mathit{sTMP} \in \mathbb{R}^{W_n}$ 和 $\mathit{sAcc} \in \mathbb{R}^{T_m \times T_n}$.
- **输入:** 将 $Q_i \in \mathbb{R}^{T_m \times d}$ 和 $K_i,V_i \in \mathbb{R}^{T_n \times d}$ 加载到 REG.
- $S_i=Q_i K_j^\top$, 其中 $S_i \in \mathbb{R}^{T_m \times T_n}$.
- $m_i^{\mathrm{new}}=\max(m_i,\mathrm{rowmax}(S_i,\mathit{sTMP}))$.
- $P_i=\exp(S_i-m_i^{\mathrm{new}})$, 其中 $P_i \in \mathbb{R}^{T_m \times T_n}$.
- $\mathit{sAcc}=\mathit{tiled\_copy\_r2s}(P_i)$.
- $P_i'=\mathit{tiled\_copy\_s2r}(\mathit{sAcc})$.
- $O_i^{\mathrm{new}}=P_i'V_j+\mathrm{diag}(e^{m_i-m_i^{\mathrm{new}}})O_i$.

<span id="figure-07"></span>

![BitDecoding 系统概览.](./bitdecoding/figure-07.png)

**图 7.** BitDecoding 系统概览. (1) **Query Transformation** 重构 query tensor 布局, 让多种 attention 变体能在 Tensor Core 上高效进行 warp 级执行. (2) **Residual Kernel** 以很小的开销执行量化和打包, 同时支持 tensor-wise 与 channel-wise scaling. (3) **Packing Kernel** 采用细粒度异步流水线执行反量化和矩阵乘法, 利用低比特参数尽可能提高 Tensor Core 与 CUDA core 的利用率.

<span id="section-5"></span>

## 5 系统实现

本节介绍 BitDecoding 的实现, 如[图 7](#figure-07) 所示. 我们的实现包含三个主要部分: (i) 支持 LLM 中不同 attention 变体的 *query transformation*; (ii) 以较低开销执行量化和打包, 并能通用于不同量化算法中 tensor-wise 与 channel-wise scaling 的 *Residual Kernel*; (iii) 采用细粒度流水线、充分利用 Tensor Core 与 CUDA core 的 *Packing Kernel*. 最后, 我们讨论针对特定架构的优化, 它们利用新一代 GPU (例如 Hopper 和 Blackwell) 的高级功能进一步提高解码吞吐量.

<span id="section-5-1"></span>

### 5.1 Query Transformation

现代 LLM 采用不同的 attention 变体 [Dub24, Yan25a, Dee24a], 它们使用不同的键值 (KV) 共享模式. BitDecoding 的目标是支持所有这些变体.

例如, 在 GQA 和 MQA 中, 多个 query head 共享一个 KV head, 从而减少 KV projection 和内存访问次数. 共享程度用 $g_q=h_q/h_{kv}$ 衡量, 其中 $h_q$ 和 $h_{kv}$ 分别为 query head 与 KV head 的数量: $g_q=1$ 对应 MHA, $g_q>1$ 表示 GQA, $h_{kv}=1$ (即 $g_q=h_q$) 则表示 MQA.

解码时会遇到一个问题: 因为 $Q\_len=1$ (每次处理一个 token), query tensor 的 batch 维很小, 直接计算 $Q\cdot K^\top$ 无法填满 Tensor Core, 导致 warp occupancy 和吞吐量都很低.

为此, 我们执行 *query transformation*, 重新组织 query 布局, 使其更好地匹配 Tensor Core tiling. 如[图 7](#figure-07) 左侧所示, 我们把 query tensor 从 $[1,(g_q,h_{kv})]$ reshape 成 $[g_q,h_{kv}]$, 在不改变 attention 语义和 KV 共享模式的前提下形成更大的 $Q$ tile. 随后, grouped query head 会作为更大的 GEMM block 并行处理, 充分填充 Tensor Core fragment, 提高 warp occupancy 和吞吐量.

<span id="section-5-2"></span>

### 5.2 Residual Kernel

低比特 KV 缓存设计的一项主要难点是支持多种量化算法, 特别是不同的 scaling granularity (例如 tensor-wise、channel-wise), 同时又不能牺牲性能. 为计算 scale 和 zero-point, 量化需要执行 reduction 和 element-wise operation, 随后进行 bit-packing; 在解码期间, 这些操作必须在线运行, 会增加运行开销, 并可能不符合 Tensor Core 要求的严格布局. 为此, 我们为 *Residual Kernel* 设计了两项优化:

**(1) 根据 residual block size 划分 KV 缓存.** 在上下文长度为 $L$ 的 prefill 阶段, 我们根据与 Tensor Core 对齐的 residual block size $N_r$ 划分 KV 缓存 (见[式 1](#equation-01)). 前 $N_p=L-(L\mod N_r)$ 个条目通过融合的量化和打包操作, 量化并打包到低比特 KV 缓存. 剩余 KV Tensor 的大小为 $\texttt{res\_len}=L\mod N_r$, 保存在半精度 residual KV 缓存中. 每个 decode step 都会把新生成的 $K,V$ tensor 添加到 residual cache, 并用于 attention 计算. 该缓存逐步增长, 直到达到 residual block size $N_r$. Residual Kernel 在每次生成 token 时用半精度 residual KV 缓存执行 attention; 当 $\texttt{res\_len}=N_r$ 时, 还可以选择将其量化成打包格式.

通过在解码期间这样划分 KV 缓存, 我们可以自然地沿 $seq\_len$ 执行 channel-wise 量化, 并在 residual block 内沿 hidden dimension 执行 tensor-wise 量化.

**(2) 使用 warp 级指令优化 reduction.** 如[图 7](#figure-07) 中部所示, 半精度 KV 数据计算完成后, 仍以 Tensor Core fragment 的形式保存在 register 中, 采用 `mma` 操作原生的交错布局. 为了高效计算量化参数 (scale 和 zero-point), 我们先进行 thread-level reduction, 得到每个 group 的局部 min/max 统计值.

随后通过 PTX 指令 `__shfl_xor_sync` 在 warp 内汇总这些局部结果, 无需 shared memory 即可高效执行 warp-level reduction. 当 warp repetition factor $W_n>1$ 时, 我们引入一个很小的 shared memory buffer 来协调 warp 之间的最终 reduction.

量化参数计算完成后, 每个 thread 都在 register 中执行量化, 并把低比特值打包成 INT16 格式. 这样无需额外搬运数据, 数据也能保持随时可用于计算的状态. 为降低开销, scale 和 zero-point 都以紧凑的 `half2` 格式存储, 从而在 decode 阶段的反量化过程中高效访问内存并使用 fused multiply-add.

<span id="section-5-3"></span>

### 5.3 Packing Kernel

另一个问题来自低比特辅助元数据 (scale 和 zero-point), 它会增加 memory traffic, 而反量化仍然需要在 CUDA core 上运行. 调度不当时, 这会打断 load-compute pipeline, 使 Tensor Core 操作无法与其他工作重叠. 因此, 我们设计了细粒度异步流水线: CUDA core 负责反量化, Tensor Core 执行矩阵乘法, 二者都与 GPU 内存层次中的数据传输重叠, 从而高效执行混合精度计算.

**(1) 优化异步数据搬运.** 在*从 Global Memory 到 Shared Memory* 的过程中, 我们沿用 FlashAttention [Dao24a] 的 block-wise tiling [Wan25] 和策略性重计算. 它使用 block size $T_m$ 和 $T_n$, 在 shared memory 中以 tile 形式处理输入矩阵 $Q \in \mathbb{R}^{T_m \times d}$、$K,V \in \mathbb{R}^{T_n \times d}$. Key-value tile 的数量为 $C_n=\lceil L/T_n\rceil$.

为了高效管理量化参数, 我们为 $K_{\mathrm{pack}}$ 参数 ($K_p$) 和 $V_{\mathrm{pack}}$ 参数 ($V_p$) 设置专用 shared memory buffer, 以 tile 形式高效复制内存. 这些 buffer 使用 `half2` 格式存储 `scale` 和 `zeros`, 一条指令即可完成加载.

$K_p$ 的形状由量化 granularity 设置决定, $V_p$ 采用 Tensor-wise 布局:

- **Channel-wise:** $(T_n/\mathit{group\_size},d)$.
- **Tensor-wise:** $(T_n,d/\mathit{group\_size})$.

为了尽可能让内存操作重叠, 所有 global-to-shared memory transfer 都通过 `cp.async` intrinsic 异步执行, 如[图 7](#figure-07) 右侧所示. 我们使用具有不同 caching strategy 的指令来优化内存事务:

- **`cp.async.cg`:** 用于 $Q$、$K_{\mathrm{pack}}$ 和 $V_{\mathrm{pack}}$; 这些数据不会在同一 kernel 内复用, 因而只在 global memory 中缓存.
- **`cp.async.ca`:** 用于 $K_p$ 和 $V_p$, 为细粒度内存访问提供更小的 byte-level alignment.

在 Hopper 架构上, 我们沿用 FA3, 使用 `tma.copy` 指令加载数据. 这样可以采用 warp-specialized scheduling, 改善 data locality, 并降低多个 warp 的内存延迟.

在*从 Shared Memory 到 Register* 的过程中, 我们使用 PTX 指令 `ldmatrix`, 按照 Tensor Core tiling 布局把 $K_{\mathrm{pack}}$、$V_{\mathrm{pack}}$ 和 $\mathit{sAcc}$ 从 shared memory 高效加载到 register. 为消除 bank conflict, 我们使用 [Nvi24a] 中定义的 sizzling scheme:

<span id="equation-02"></span>

$$
\mathrm{col}_{id}=\mathrm{row}_{id}\oplus\mathrm{col}_{id}
$$

实现无 bank conflict 的访问. 此外, 我们还重构 $K_p$ 和 $V_p$ 的 shared memory 布局, 进一步减少 bank conflict 并提高吞吐效率.

**(2) 让 CUDA Core 与 Tensor Core 重叠执行的异步流水线.** 为充分利用 CUDA core 和 Tensor Core, 我们实现了一个 register 级异步流水线, 让计算与内存操作重叠. 在该流水线中, 通过 `ldmatrix` 从 shared memory 加载数据和反量化 (`Dequant`), 会在 SM warp scheduler 的调度下与 Tensor Core 矩阵乘法 (`mma`) 并发运行.

如[图 7](#figure-07) 右侧所示, Tensor Core 通过 `mma` 处理第 $i$ 个 slice 时, 第 $(i+1)$ 个 slice 同时从 shared memory 加载 (`ldmatrix`) 并反量化. 这能维持连续的 producer-consumer flow, 提高 instruction throughput, 并尽可能利用 CUDA core 与 Tensor Core.

<span id="figure-08"></span>

![Blackwell 架构使用 MXFP4 时的 kernel 性能.](./bitdecoding/figure-08.png)

**图 8.** Blackwell 架构使用 mxfp4 时的 kernel 性能. (a) RTX 5090. (b) RTX PRO 6000.

<span id="section-5-4"></span>

### 5.4 新架构支持

目前的设计可以有效面向 Hopper 之前的架构 (例如 Ampere), 但新一代架构提供了不同的硬件功能, 需要针对性优化. 下文介绍我们如何调整方法, 利用 Hopper 和 Blackwell 的专用指令与原生数据格式.

**(1) 巧用 PTX 级指令, 发挥 Hopper 的 warpgroup 加速能力.** Hopper Tensor Core 引入了 Warpgroup Matrix Multiply-Accumulate (`wgmma`) 指令. 但该指令有一项重要限制: 在矩阵乘法 $C=A B$ 中, 只有 $A$ 和 $C$ 可以来自 register, $B$ 必须位于 shared memory. 这对低比特量化数据形成了挑战, 因为这些值通常要先在 register 中转换成 FP16, 再参与计算. 为此, 我们使用 Hopper 的 `STSM` PTX 指令, 把反量化后的 FP16 值高效存入 shared memory, 供 `wgmma_SS` 操作访问. WGMMA 的异步特性可以让存储与计算重叠, 从而提高性能.

**(2) 使用原生低精度格式加速 Blackwell.** Blackwell 架构原生支持低精度 tensor 操作, 无需显式反量化. 因此, 前述基于 `lop3` 的 register 重映射会被跳过, 改为直接执行. 我们面向 Blackwell 的低精度 `mma` 指令, 特别是支持 micro-scaling 格式 (例如 `mxfp4 / nvfp4`) 的指令, 直接在打包的 4-bit 数据上执行 GEMM. 尽管这些指令对打包值和 block scaling factor 都有严格的布局限制, 但[第 4.1 节](#section-4-1) 提出的布局变换策略并不依赖具体布局. 它能自动使打包后的 KV 数据符合硬件要求的格式, 接入 Blackwell 原生 tensor pipeline.

<span id="section-6"></span>

## 6 评测

本节将 BitDecoding 与当前最佳方法和系统进行全面比较. 主要结果如下:

1. BitDecoding 在多代 GPU 上都明显快于 FP16 FlashDecoding-v2: 使用原生 MXFP4 时, 在 Blackwell 上最高加速 8.6$\times$; 在 Hopper 上最高加速 8.0$\times$; 在 Ada 上最高加速 7.5$\times$; 相比当前最佳低比特系统 QServe, 最高加速 4.3$\times$ ([第 6.1 节](#section-6-1)).
2. 在端到端长上下文推理中, BitDecoding 把单 batch 延迟降低了 3x (LLaMA-3.1-8B, 128K 上下文), serving throughput 比 QServe 高出 4x 以上; 在以往纯 CUDA Core 方法性能下降的 GQA 设置下, 它仍能良好扩展 ([第 6.2 节](#section-6-2)).
3. BitDecoding 在接近 FP16 精度的同时, 各系统组件都能带来明显性能收益; 4-bit 量化的精度仅下降 0.2%, ablation study 也表明每个设计模块都为总体加速作出了贡献 ([第 6.3 节](#section-6-3)).

<span id="section-6-1"></span>

### 6.1 不同 GPU 架构上的 kernel 性能

**Kernel 设置.** 不同 LLM serving 场景需要不同的 workload 和 attention kernel 设计, 因此我们在以下三个代表性设置下评测性能:

- **Single:** $\mathit{batch\_size}=1$ 的场景, 代表有长上下文的边缘用户推理.
- **Batches:** 使用较大的 $\mathit{batch\_size}$, 保持输入长度相同并采用简单 padding.
- **Page:** 高吞吐场景, 使用 page management 技术 [Kwo23] 管理更大的 $\mathit{batch\_size}$.

**Baseline.** 我们将 BitDecoding 与几种有代表性的 attention kernel 实现进行比较. 对于 FP16 KV 缓存, 我们使用 FlashDecoding [Dao24a, Sha24b] 作为加速归一化的 baseline; 它是面向长上下文解码优化的 FlashAttention split-partitioned 变体. 对于低比特 KV 缓存, 我们评测了支持 4-bit 和 2-bit 量化的未融合 kernel Kivi [Liu24c], 以及 Atom [Zha24e] 和 QServe [Lin24a]; 后两者都是只使用 CUDA Core 的融合 kernel 实现, 支持带 page management 的 4-bit 缓存. Atom 不支持 GQA.

**量化设置.** 我们在多种量化配置下评测 BitDecoding, 对 4-bit 和 2-bit Key tensor 均支持 Channel-wise (KC) 与 Tensor-wise (KT) 方案.

<span id="figure-09"></span>

![Hopper H100 上的 kernel 性能.](./bitdecoding/figure-09.png)

**图 9.** Hopper (H100) 上的 kernel 性能.

**MXFP4 / NVFP4 上的结果 (RTX5090、RTX PRO 6000).** Blackwell 架构原生支持低精度数据格式, 在消除在线反量化开销的同时, 还能为低比特操作提供很高的计算吞吐量. 如[图 8a](#figure-08) 所示, BitDecoding 的性能明显更高: batch 场景下最高加速 8.6$\times$, 单 batch 长上下文解码 (128k) 时加速超过 4.3$\times$, 大幅领先未融合 attention baseline. 同样, [图 8b](#figure-08) 表明 RTX PRO 6000 也获得了很大提升, 大 batch size 下最高加速 6.5$\times$.

<span id="figure-10"></span>

![RTX4090 上的 kernel 性能.](./bitdecoding/figure-10.png)

**图 10.** RTX4090 上的 kernel 性能.

**高级 Tensor Core 加速的结果 (H100).** 新一代 GPU 架构通常会引入高级计算指令, 显著加快 kernel 执行. 如[图 9](#figure-09) 所示, 为 Hopper Tensor Core 优化的 FlashDecoding-v3 相比 v2 明显更快. BitDecoding-v2 最高加速 4.1$\times$, v3 实现则进一步提高到 8.0$\times$. 这是因为 BitDecoding 使用了 Hopper 的 `wgmma` 和异步内存指令, 即使在混合精度设置下也能保持较高的 Tensor Core 利用率.

<span id="figure-11"></span>

![A100 上的 kernel 性能.](./bitdecoding/figure-11.png)

**图 11.** A100 上的 kernel 性能.

**带宽受限 GPU 上的结果 (RTX 4090).** 对于带宽受限 GPU, 使用低精度数据是加速推理的重要手段. 如[图 10](#figure-10) 所示, 在 Single 和 Batches 设置下, BitDecoding 相比 FlashDecoding-v2 的 4-bit 加速约为 $4\times$, 2-bit 加速超过 $7\times$; 这些收益直接来自低比特 KV 缓存对 DRAM 瓶颈的缓解.

BitDecoding 在所有场景中都明显快于 baseline; 未融合的 KIVI 依赖独立 kernel, 在 GQA 中性能严重下降, 而 BitDecoding 的完全融合设计仍保持较高效率. 在 Page 设置下, 它也快于融合的 CUDA-core baseline: 对于 MHA, BitDecoding 加速超过 $6\times$, QServe 只有 $3.5\times$. 更重要的是, 在计算密集型 GQA 中, BitDecoding 仍有 $3\times$ 加速, QServe 则降至 $1.4\times$, 说明在纯 CUDA 方法失效的场景中, Tensor Core 仍能提供可靠加速.

**高带宽 GPU 上的结果 (A100).** 在 A100 这类高内存带宽架构上, 计算压力更明显; 如果 kernel 设计不能充分使用可用计算资源, 性能瓶颈会从内存访问转向计算利用率. 如[图 11](#figure-11) 所示, KIVI 和 QServe 的性能都不理想: KIVI 受未融合 kernel 设计影响, QServe 则未能充分使用 Tensor Core, 二者甚至慢于 FP16 baseline. BitDecoding 可以在各种 workload 下始终快于所有 baseline, 最高加速 $3\times$, 原因是它能高效使用 Tensor Core 和融合执行流水线. 另一个观察是, A100 上 4-bit 与 2-bit 变体的性能差距变小, 因为更高的 DRAM 带宽减轻了内存瓶颈, 性能平衡转向 compute-bound 执行.

<span id="section-6-2"></span>

### 6.2 LLM 推理系统中的性能

**模型设置.** 我们评测了多种 LLM, 包括 LLaMA-2-7B、LLaMA-3.1-8B、LLaMA-3.1-70B、Qwen3-8B 和 Qwen3-14B. 其中只有 LLaMA-2-7B 使用 MHA, 其他模型都使用 GQA. 除 LLaMA-3.1-70B 在 8$\times$A100 GPU 上评测外, 所有模型都在单张 A100 GPU 上运行.

**量化设置.** 我们为 LLM KV 缓存选择 channel-wise 量化, 因为它精度更高, 也与 Kivi 一致.

**与未融合 Attention 的对比.** 如[图 12](#figure-12) 所示, 在 Single 设置下, BitDecoding 在 128K 上下文长度时最高加速 3.3$\times$; 此时加载 KV 缓存是 LLM 推理的主要瓶颈. Kivi 的扩展能力有限, 由于不支持 block-tiling kernel, 在 128K 时会出现 out-of-memory (OOM). 在 Batches 设置下, BitDecoding 的吞吐量明显高于 KIVI: BitDecoding-KC-4 与 KC-2 分别最高达到 900 和 1200 tokens/s, KIVI-4 与 KIVI-2 的峰值则低于 700 tokens/s.

<span id="figure-12"></span>

![与 Kivi 比较端到端生成时间和解码吞吐量.](./bitdecoding/figure-12.png)

**图 12.** 与 Kivi 比较 (a) 端到端生成时间和 (b) 解码吞吐量.

**与纯 CUDA Core 融合 Attention 的对比.** Qserve 同时支持 MHA 和 GQA attention 结构, 因而我们在 page-setting 推理中比较 BitDecoding 与 Qserve. 最大吞吐量在 GPU 内存可容纳的最大 batch size 下评测. 如[图 13](#figure-13) 所示, Qserve 在 LLaMA-2-7B 上的吞吐量高于 FlashDecoding-v2, 但由于处理 GQA 的效率不高, 在其他所有模型上性能都会下降. BitDecoding 在单 GPU 和多 GPU 设置下的 LLaMA 与 Qwen 架构上都始终快于 QServe, 最大吞吐量是 QServe 的 2$\times$ 以上.

<span id="figure-13"></span>

![与 Qserve 比较解码吞吐量.](./bitdecoding/figure-13.png)

**图 13.** 与 Qserve 比较解码吞吐量.

<span id="section-6-3"></span>

### 6.3 精度、开销与性能分解

**精度分析.** 如[表 1](#table-01) 所示, 我们评测了不同 bit-width 下的吞吐量与精度. 2-bit 量化显著减少内存用量, 因而可以使用更大的 batch size, 吞吐量达到 FP16 的 $4.25\times$. 4-bit 量化在精度仅下降 $0.2\%$ 的情况下接近全精度, 同时加速 $2.98\times$. 这些结果反映出两者之间的取舍: 4-bit 量化较为均衡, 2-bit 则以少量精度损失换取最大吞吐量.

<span id="table-01"></span>

![低比特 KV 缓存的效率与精度取舍.](./bitdecoding/table-01.png)

**表 1.** 低比特 KV 缓存的效率与精度取舍. 我们使用 $seq\_len=32K$ 的 Llama-3.1-8B-Instruct, 并在 longbench [Bai23] 上评测平均精度.

<span id="table-02"></span>

![推理期间量化和打包的延迟对比.](./bitdecoding/table-02.png)

**表 2.** 推理期间量化和打包的延迟 (ms) 对比.

<span id="table-03"></span>

![协同 softmax 与 warp 对性能和有效性的影响.](./bitdecoding/table-03.png)

**表 3.** 协同 softmax 与 warp 对性能和有效性的影响.

**半精度 Residual Kernel 开销.** 因为 $seq\_len\gg N_r$, 且 $seq\_len$ 超过 32K, $N_r$ 始终小于 256, 半精度 residual KV Cache 只会带来很小的内存开销. 如[图 14](#figure-14) 所示, 半精度 residual KV 缓存因为多启动一个 kernel, 只会产生很小的运行开销. 随着序列变长, residual 部分在整个 KV 缓存中的占比下降, 这项开销会越来越小.

<span id="figure-14"></span>

![residual KV 缓存的运行开销.](./bitdecoding/figure-14.png)

**图 14.** residual KV 缓存的运行开销.

**量化和打包开销.** 我们在 $seq\_len=128K$ 时评测量化与打包延迟, 并将 BitDecoding 与 Marlin [Fra24]、Ladder [Wan24e] 比较. 如[表 2](#table-02) 所示, 以往混合精度计算方法的预变换与打包步骤会产生不可忽略的大量开销. 我们的 kernel 在 Prefill 阶段后开销很小, 主要来自 kernel 启动. 在解码期间, 这些操作完全融合到 kernel 计算中, 因此开销几乎可以忽略.

**反量化开销.** [图 15a](#figure-15) 显示, Atom 和 QServe 的反量化计算开销很高, 接近 kernel 执行时间的一半. BitDecoding 通过让 Tensor Core 更充分地重叠工作, 将这项开销降至 15% 以下 (4-bit) 和 35% 以下 (2-bit).

进一步比较 Atom 与 BitDecoding 的 microbenchmark ([图 15b](#figure-15)) 表明, BitDecoding 有效使用 Tensor Core, 因而 memory throughput 更高. Atom 则高度依赖 CUDA core, 加重了 FMA 和 ALU 操作的压力.

<span id="figure-15"></span>

![反量化开销与微观分析.](./bitdecoding/figure-15.png)

**图 15.** 反量化开销分析. (a) 反量化开销. (b) 微观分析.

**多 warp 协同 Softmax 开销.** [表 3](#table-03) 显示, 增加 $W_n$ 可以提高 Tensor Core 利用率并降低延迟, 但如果不采用协同 softmax, 结果会出错. 启用协同 softmax 后, 结果重新正确, 开销仅为 0.5%. 虽然它会访问 shared memory, 但低比特数据减轻了内存带宽压力, 使 kernel 从 memory-bound 转为 compute-bound, 因而开销很小.

**分解分析.** 为进一步分析 BitDecoding 的性能收益, 我们在[图 16](#figure-16) 中分解了各项优化. 参考 [Ash24], 我们使用 continuous-packing baseline, 每生成一步都量化并打包 KV 缓存; 它会产生大量开销, 还需要手动维护有效布局. 相比之下, 我们的布局设计会自动为任意低比特格式构造兼容 Tensor Core 的布局, 充分发挥 Tensor Core 的计算能力. 在此基础上, warp 并行策略进一步提供了明显加速, 流水线优化则继续改善端到端性能.

<span id="figure-16"></span>

![不同架构代际上的 BitDecoding 优化分解.](./bitdecoding/figure-16.png)

**图 16.** 不同架构代际上的 BitDecoding 优化分解.

<span id="section-7"></span>

## 7 相关工作

**KV 缓存量化算法.** KV 缓存量化可以在保持性能的同时, 减少长上下文 LLM 的内存用量. 最近的工作探索了 4-bit、2-bit, 甚至 1-bit KV 缓存量化, 希望进一步提高压缩率. KIVI [Liu24c]、Gear [Kan24] 和 KVQuant [Hoo24] 等方法使用 per-channel 量化处理 key-value outlier, RotateKV [Su25c] 则通过 rotation 平滑 channel-wise distribution. 这些方法在较高压缩率下效果很好, 但缺少高效的系统实现, 因而性能不够理想.

**混合精度矩阵乘法.** LLM 中的低比特权重和低比特 KV 缓存产生了一项特殊需求: 混合精度矩阵乘法 (mpGEMM) 的一个输入矩阵使用较低精度 (例如 INT4/2/1), 另一个输入矩阵保持较高精度 (例如 FP16/8). Ladder [Wan24e] 和 Marlin [Fra24] 等优化 kernel 通过布局变换和高效反量化提高性能. 但这些方法需要预先打包和变换权重, 因而难以用于自回归解码中的低比特 KV 缓存.

**低比特 KV 缓存的系统实现.** KIVI [Til19] 使用 Triton 和多个独立 kernel 实现低比特 KV Cache. Atom [Zha24e] 把量化集成到前一个 linear layer 中, QServe [Lin24a] 则直接把量化融合进 FlashAttention kernel. 但二者都依赖采用 fused multiply-add (FMA) 指令的 GEMV 操作, 没有使用 Tensor Core 加速.

<span id="section-8"></span>

## 8 结论

BitDecoding 证明, 通过系统化设计可以协调 CUDA core 与 Tensor Core, 高效执行低比特 KV 缓存解码, 为这类系统提供了新的基础. 它的布局构造与 warp 级协调技术可以适用于多种 attention 变体、量化方案和 GPU 代际, 也能自然扩展到 Blackwell 等新兴架构乃至后续架构. 我们预计 BitDecoding 将推动 KV 缓存量化的算法-系统协同设计、近乎无损的测试时扩展, 以及面向长上下文 LLM 推理、更强大的 GPU 执行模型.
