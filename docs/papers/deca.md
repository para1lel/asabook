---
title: DECA Accelerator
createTime: 2026-08-24
permalink: /papers/deca/
---

> [Gerasimos Gerogiannis](https://dblp.org/pid/314/4553), [Stijn Eyerman](https://dblp.org/pid/99/4678), [Evangelos Georganas](https://dblp.org/pid/121/2450), [Wim Heirman](https://heirman.net/), [Josep Torrellas](https://dblp.org/pid/t/JosepTorrellas). 首次提交 arXiv 的日期为 2025-05-25, 当前版本为 v2 (2025-08-08). 发表于 MICRO 2025, pages 184-200, online 2025-10-17, print 2025-10-18. [arXiv:2505.19349](https://arxiv.org/abs/2505.19349). [原始 PDF](/paper/deca.pdf). [DOI: 10.1145/3725843.3756073](https://doi.org/10.1145/3725843.3756073). [TeX 源码](https://arxiv.org/e-print/2505.19349). 原论文标题为 "DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model".

## 摘要

为缓解大语言模型 (LLM) 推理工作负载的内存带宽瓶颈, 权重矩阵以量化和稀疏格式存储在内存中. 因此, 这些矩阵的 tile 在交由核内通用矩阵乘法 (GeMM) 硬件引擎处理前, 必须先完成反量化和解稀疏. 目前这项工作由软件中的向量操作完成, 但性能提升有限. 此外, 由于 GeMM 的总体性能取决于内存资源, 向量单元和硬件矩阵引擎之间的相互作用, 系统的改进方向也难以判断.

为提高配备核内 GeMM 引擎和 HBM 的先进平台上的 LLM 推理性能, 本文作出三项主要贡献. 第一, 构建带三维可视化表示的分析性能模型, 用于说明内存资源, 向量单元和硬件矩阵引擎如何共同实现压缩 GeMM 性能. 第二, 提出新的近核机器学习模型解压缩加速器 DECA. DECA 从 CPU 卸载 tile 的解稀疏和反量化, 生成可直接供核内 GeMM 引擎使用的 tile. 第三, 引入新的 ISA 扩展, 支持以乱序方式调用近核加速器. 借助该扩展, 加速器和核心的计算可以高效交错并重叠执行. 评估显示, 在模拟的带 HBM 的 56 核 Xeon 4 服务器上, DECA 相比优化的 Intel 软件 kernel 可将压缩 GeMM 的执行速度最多提高 4 倍, 并将 Llama2-70B 和 OPT-66B 的下一个 token 生成时间缩短 1.6 倍至 2.6 倍.

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) 是最重要的机器学习 (ML) 工作负载之一, 擅长聊天机器人, 翻译, 文本摘要和内容创作等任务 [Zho24j, Zha23m, Kal23, Yao23d]. LLM 使用 Transformer [Vas17], 主要由多头注意力层和全连接 (FC) 层组成. 最大的模型在 FC 层中包含数万亿个参数 (权重) [Ope24a, Zha23d]. 在推理过程中, 这些权重的重复使用率很低 (例如在小批量场景中), 不仅给现代平台的内存容量带来压力, 也给内存带宽带来压力 [Yua24].

GPU 凭借高计算能力和内存带宽, 被视为 LLM 推理的标准平台. 然而, Intel Xeon 4 服务器 (代号 Sapphire Rapids, SPR) 的近期进展 [Bis21] 使 CPU 也成为有吸引力的 LLM 推理平台. 首先, 这类处理器配备了称为 TMUL 的核内通用矩阵乘法 (GeMM) 引擎 [Int24a]. TMUL 与 GPU Tensor Core [Mar18b] 作用相同. 它通过 AMX ISA 扩展 [Int24a] 编程, 在矩阵 tile 上执行 GeMM. 与完全依赖向量 SIMD 单元相比, 其 GeMM 计算吞吐量提高了一个数量级. 其次, SPR 服务器可以配备高带宽内存 (HBM), 使可用内存带宽比 DDR 平台提高 3-4$\times$.

我们观察到, 与 GPU 类似 [Yua24], SPR CPU 上的 LLM 推理受内存带宽限制. LLama2-70B 的 FC 层大规模 GeMM 占下一个 token 生成时间的 90% 以上 [Tou23a]. 这类 GeMM 的算术强度低, 需要从主内存加载大量权重. 在很大程度上, 加速 CPU 上的 LLM 推理就是加速这些大规模 GeMM.

深度神经网络 (DNN) 模型压缩技术 [Lia21a, Den20a], 例如低比特权重量化 [Gho21a] 和稀疏化/剪枝 [Hoe21, Xia23b, Zhu23b], 可以改善 GeMM 性能: 需要从内存加载的数据量减少, 从而显著加速受内存限制的 kernel. 遗憾的是, 与脉动阵列 [Jou23] 和 Tensor Core [Xia23b] 一样, TMUL 无法处理任意量化方案或稀疏模式. 因此, SPR 的 TMUL 引擎要求输入 tile 是格式正确的稠密数据 (即必须包含零值), 且格式为 BF16 [Kal19] 或 INT8.

为了同时利用模型压缩和 TMUL 的 GeMM 吞吐量, Intel 最近在 libxsmm 框架中引入了专用 kernel [Hei16]. libxsmm 使用一系列向量 (AVX) 指令从内存读取压缩 tile, 对其解稀疏和/或反量化, 再将结果送入 TMUL AMX 单元. 这种协同处理模式涉及两个不同的计算域 (向量和矩阵), 每个域都有自己的指令 (AVX 和 AMX) 及功能单元 (SIMD 单元和 TMUL).

我们对不同量化和稀疏工作负载下的 libxsmm kernel 进行了性能分析. 结果表明, 这些 kernel 在中等压缩的 GeMM 和带宽相对较低的 DDR 内存上非常有效, 但在 HBM 上性能会下降. 这种下降无法用传统的二维 (2D) Roofline 性能模型解释 [Zha15], 因为该模型只把内存带宽和 (矩阵) 计算吞吐量视为性能上界因素.

为指导性能优化, 我们首先构建一个捕获内存, 矩阵和向量资源交互的分析性能模型. 与 2D Roofline 不同, 该模型以三维表示呈现, 用一个表面分隔可实现和不可实现的性能. 因此, 我们将其称为 *Roof-Surface*. Roof-Surface 提供了有用的性能洞察, 并准确地将 libxsmm 的性能下降归因于 AVX 向量解压缩序列. 此外, 它揭示出要消除解压缩低效, 必须以难以接受的幅度扩展 CPU 核心资源.

针对这一问题, 本文提出 *DECA*, 一种新的近核 *机器学习模型解压缩加速器*. DECA 将 tile 的解稀疏和反量化从 CPU 卸载, 为 TMUL 生成可直接使用的 tile. DECA 可编程处理 1 到 8 任意比特数的量化数字格式, 支持任意程度的非结构化稀疏, 也支持分组量化 [Gho21a]. DECA 微架构利用带有高级向量操作的流水线执行解压缩. 重要的是, 我们使用 *Roof-Surface* 模型来 (1) 决定向量流水线微架构, 以及 (2) 执行设计空间探索并推导出均衡的 DECA 设计.

我们观察到, 如果 CPU 核心使用普通的内存映射 load/store 指令与 DECA 通信, 通信延迟会暴露出来并损害性能. 为此, 我们引入一种新的 ISA 扩展, 通过乱序调用加速器来隐藏 CPU 与 DECA 的通信延迟. 我们将该扩展称为 *Tile External Preprocess and Load* (TEPL).

针对两种不同的低比特量化格式 (BF8 和 MXFP4) 以及不同程度的非结构化稀疏进行的评估表明, DECA 非常有效. 在模拟的带 HBM 的 56 核 SPR 系统中, DECA 相比优化的 Intel libxsmm 软件 kernel, 可将压缩 GeMM 的执行速度提高最多 4x. 此外, 通过加速 FC 层, DECA 将 Llama2-70B 和 OPT-66B [Zha22] 的下一个 token 生成时间相比纯软件方案缩短 1.6$\times$—2.6$\times$, 相比未压缩基线模型缩短 2.5$\times$—5.0$\times$.

本文的贡献如下:

- 描述向量单元, 矩阵单元和内存之间交互的 *Roof-Surface* 性能模型.
- 用于加速压缩 ML 模型解稀疏和反量化的近核加速器 *DECA*.
- 支持乱序调用近核加速器的 Tile External Preprocess & Load (*TEPL*) 扩展.
- 基于模拟的评估, 用于分析 *DECA* 在 LLM 推理压缩 GeMM 中的性能.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 LLM 推理

大语言模型 (LLM) 由不同层组成, 例如嵌入层, 全连接 (FC) 层和注意力层 [Vas17]. LLM 推理分为两个阶段 [Pat23]. 第一阶段编码输入 token 并生成第一个 token (提示阶段), 第二阶段生成后续输出 token (生成阶段). 本文关注如何高效执行算术强度较低的生成阶段, 因为在许多实际用例中, 它占据端到端 LLM 推理时间的大部分 [Yua24].

GPU 凭借高计算能力和内存带宽, 被视为 LLM 推理的标准平台 [Pat23, Su23a]. 然而, HBM 和核内 GeMM 引擎等近期进展 [Bis21] 使 CPU 也成为有吸引力的 LLM 推理平台. 通过在 CPU 芯片上加入扩展或小型加速器来改善 CPU 的机器学习 (ML) 和科学计算能力, 已成为研究界和工业界日益关注的方向 [Jeo23, Gon20, Gon22, Sir23, Nas22, Ore22]. 因此, 本文聚焦于现代 CPU 服务器上的 LLM 推理.

<span id="section-2-2"></span>

### 2.2 模型压缩

对于算术强度较低的 LLM FC 层, 压缩权重矩阵可以减少数据移动, 从而直接改善 GPU 和 CPU 上的性能. ML 模型主要有两种压缩方式 [Han15, Zhu23b]:

- **量化** 是将权重以更低比特的格式存储, 例如用 FP8 或 FP4 替代 FP16. 已有多种量化方案 [Lin23d, Zha24e, Kim24, Wei23]. 其中一些方案还会将权重划分为多个组, 并为每组引入缩放因子 (*分组量化*) 以获得更高精度. 本文评估两种权重量化: BF8 (8 位 brain floating point) 和 MXFP4 [Bit23]. 后者使用 4 位浮点数, 对每 32 个权重采用共享缩放因子的分组量化 (8 位指数); 研究表明它不会降低 LLM 精度 [Bit23].
- **稀疏化** 是消除 (*剪枝*) 接近零或对模型精度贡献很小的权重 [Lec89, Bla20, Hoe21]. *非结构化稀疏* 不限制可删除权重的位置. 在相同稀疏程度下, 它比结构化稀疏具有更高精度 [Liu21d, Fra23a]. 本文采用非结构化稀疏和基于位掩码的稀疏格式, 以避免存储零值. 为了重建原始权重矩阵中非零值的位置, 使用一个位数等于原始矩阵元素数的位掩码. 位掩码中的 '1' 表示非零值的位置, 非零值连续存储在非零数组中. 近期提出的 SparseGPT 等 LLM 权重剪枝方法 [Fra23a] 已在几乎不损失精度的情况下实现最高 60-70% 的非结构化稀疏. 对于 ResNet50 等传统 ML 模型 [He16], 最高 95% 的非结构化稀疏很容易实现 [Pes21]. 我们认为 LLM 研究进展很快会支持更高的稀疏程度, 因此评估 50%-95% 的大范围稀疏率.

模型可以同时具有稀疏性和量化 [Har24]. 从稠密 BF16 模型出发, 密度因子为 $d$ 的 $Q$ 比特量化模型 (例如 $d=10\%$ 表示只有 10% 的权重非零) 会将模型大小缩小 $16/(Q\times d+1)$ 倍, 其中 '1' 来自位掩码中的比特. 我们假设激活的内存占用可以忽略. 本文将该缩小倍数称为 *压缩因子 (CF) *.

压缩过程在线下执行 (例如训练完成后). 该过程显示在图 [1](#figure-01) 的左侧. 本文假设模型已经压缩完成, 并在线用于推理.

<span id="section-2-3"></span>

### 2.3 矩阵扩展

已有多种矩阵扩展用于提高 CPU 上矩阵乘法的效率 [Int22, Wil22, Car22a, Bha21]. 本文使用 Intel 的 Advanced Matrix Extensions (AMX) [Int22]. AMX 在寄存器文件中增加 8 个矩阵寄存器, 称为 tile 寄存器. 每个寄存器最多容纳 16 行, 每行 64 字节, 可解释为 32 个 2 字节元素 (BF16) 或 64 个 1 字节元素 (INT8). 每个 tile 最多包含 1 KB 数据.

每个核心都有 tile 寄存器和用于相乘 tile 的矩阵乘法 TMUL 单元. AMX 包含 tload/tstore 指令, 用于在 tile 寄存器与内存之间加载/存储数据. 对于 batch size 为 $N\leq16$ 且使用 BF16 的 LLM 下一个 token 生成阶段, 权重 tile $W$ 包含 $M=16$ 行, 每行 $K=32$ 列. 激活 tile $A$ 包含 $N$ 行, 每行 $K=32$ 列. TMUL 执行 $A\times W^\top$, 生成 $N\times M$ 输出 tile. 无论 N 取何值, TMUL 操作都需要 16 个周期, 并执行总计 $N\times K\times M =N\times32\times16=512N$ 次融合乘加 (FMA), 即每周期 $32N$ 次 FMA. 当 N>16 时, 由于激活 tile 最多只能容纳 16 行, TMUL 吞吐量饱和于每周期 512 次 FMA. 本文提到的 FLOPs 均指 FMA.

<span id="section-2-4"></span>

### 2.4 GeMM 解压缩

与其他 GeMM 引擎类似 [Mar18b, Jou23], TMUL 只能处理非常特定的数据格式 (即 BF16 或 I8), 也无法处理非结构化稀疏. 如果 GeMM 含有压缩权重, 就需要解压缩以生成符合 TMUL 要求的 tile. 与压缩不同, 解压缩在线执行 (图 [1](#figure-01)), 因而会影响性能.

<span id="figure-01"></span>

![图 1. 离线压缩和在线解压缩.](./deca/figure-01.png)

**图 1.** 离线压缩和在线解压缩.

<span id="figure-02"></span>

```text
............
//Decompress Ti+1
for(r=0 to 16):
{
  //Decompress
  //row r of Ti+1
  VectorOps AVX
  ...
}

//GeMM Ti
MatrixOps AMX
TLoad Ti
TComp Tout, Ti

//Decompress Ti+2
//GeMM Ti+1
............
```

**图 2.** Libxsmm 压缩 GeMM kernel 的伪代码.

为了在压缩 GeMM kernel 中获得高性能并隐藏解压缩开销, Intel 最近在 Libxsmm 框架中集成了一种软件方案 [Hei16] (图 [2](#figure-02)). 解压缩序列使用 AVX 向量操作完成, 实际 GeMM 则使用 AMX 矩阵操作完成. Libxsmm 采用巧妙方法让两者重叠执行: 软件分配两个缓冲区, 并尽量将它们保留在 L1 缓存中. tile *Ti+1* 的 AVX 解压缩序列输出写入两个缓冲区之一. 与此同时, AMX 指令从另一个包含 *Ti* 的缓冲区加载数据, 而 *Ti* 之前已由 AVX 序列解压缩. AMX 与 AVX 的重叠由乱序执行实现, 依赖关系会自然得到遵守.

解压缩序列使用 permute 等向量操作进行解压缩, 并使用带掩码的向量扩展将零插入非零数组的相应位置. 受篇幅限制, 我们省略具体细节, 但第一点是, 解压缩通过 AVX 完成, 使用的计算 "域" (即独立的指令和功能单元) 与 AMX 不同. 第二点是, AVX 的动态指令数量远多于 AMX, 因为 AMX 使用 tile 大小的操作数 (1KB), 而 AVX 处理 cache line 大小的操作数 (64B, 即一行 tile).

<span id="section-3"></span>

## 3 研究动机

<span id="section-3-1"></span>

### 3.1 FC 层 GeMM 主导推理

表 [1](#table-01) 展示了在配备 DDR5 或 HBM 的 SPR 服务器上, Llama2-70B [Tou23a] 不同全连接 (FC) 层的 GeMM 所占下一个 token (即生成) 时间比例. 我们针对使用 BF16 权重的未压缩模型, 展示不同输入 token 数量和 batch size ($N$) 下的结果. 剩余时间花在注意力等不适用权重压缩的 kernel 上. 可以看到, 这类 GeMM 在 DDR5 上占用超过 95% 的时间, 在 HBM 上占用 85-90% 的时间. 因此, 加速这些 GeMM 可以大幅缩短下一个 token 的生成时间.

<span id="table-01"></span>

![表 1. FC 层 GeMM 对下一个 token 时间的贡献.](./deca/table-01.png){.paper-table-narrow}

**表 1.** FC 层 GeMM 对下一个 token 时间的贡献.

<span id="section-3-2"></span>

### 3.2 FC 层 GeMM 受带宽限制

图 [3](#figure-03) 展示了 LLama2-70B FC 层中一个大型 GeMM 在使用 DDR5 或 HBM 的 SPR 上的 Roofline 模型, 其中 N=4. 在计算受限区域, 我们使用 TMUL FLOPS 上限 ([第 2.3 节](#section-2-3)) 作为可实现的最大 GeMM FLOPS. 在本文中, 以每字节内存对应的 FLOPs 计算算术强度 (AI) 时, 假设权重矩阵的占用远大于激活的占用, 这对较小的 N 值成立. 两张图最左侧标记为 'BF16' 的圆点表示未压缩基线执行. 由于 AI 较低, 我们看到该执行在两种情况下都受内存带宽限制. 这促使我们采用模型压缩, 以减少从内存读取的数据量.

<span id="figure-03"></span>

![图 3. $N=4$ 时 GeMM 的传统 Roofline.](./deca/figure-03.png)

**图 3.** $N=4$ 时 GeMM 的传统 Roofline.

<span id="section-3-3"></span>

### 3.3 压缩 GeMM 可能引入低效

图 [3](#figure-03) 中的其他数据点表示采用 4 比特量化 (MXFP4) 或 8 比特量化 (BF8), 密度 (即非零值比例) 从 5% 到 100% 的压缩模型. 压缩减少从内存获取的数据量, 从而提高 AI; 随着压缩因子增加, 圆点向右移动. 对于每个设计点, 我们展示两个圆点: 一个表示 *Observed* 性能, 另一个位于具有相同 AI 的 Roofline 上, 后者称为 *Optimal* 性能.

我们看到, 随着压缩因子增加, Observed 点和 Optimal 点的差距越来越大. 在 DDR5 图中, 差距从密度为 5% 的 BF8 开始出现. 然而在 HBM 图中, 所有压缩模型都低于其 Optimal 性能; 对于密度为 5% 的 BF8, Optimal 与 Observed 性能之比为 4.94x. 这意味着性能受某种 Roofline 模型未捕获的低效因素限制. 通过手动分析, 我们发现根因是 AVX 解压缩指令序列的开销. 实际上, 核心中的 AVX SIMD 处理单元无法跟上内存带宽和/或 TMUL 的吞吐量.

考虑到 LLM 工作负载的重要性, 为解压缩开销提供某种硬件支持是合理的. 但 CPU 资源受限, 进行改动时必须谨慎. Roofline 模型无法告诉我们需要将向量吞吐量提高多少, 才能使 kernel 从受向量处理限制转为受内存或矩阵计算限制. 硬件方案可能设计得不够或过度, 存在这种风险. 为避免该风险, 下一节提出另一种分析模型. 该模型可以从理论上指导消除压缩 GeMM 解压缩开销所需的硬件支持.

<span id="section-4"></span>

## 4 Roof-Surface 模型

为了指导涉及矩阵, 向量和内存操作的 kernel 性能优化, 我们开发了一个捕获它们交互的性能模型. 该模型称为 *Roof-Surface*, 具有三维 (3D) 可视化表示. 我们还给出它的二维投影, 称为 *Bounding Region Diagram* (BORD).

<span id="section-4-1"></span>

### 4.1 三维 Roof-Surface 性能模型

当多个相互作用的因素影响性能时, 最慢的因素最终决定性能. 因此, 首先需要表达以下三者的处理速度: (1) 内存提供压缩 tile 的速度 (MEM), (2) 向量硬件处理压缩 tile 的速度 (VEC), 以及 (3) 矩阵硬件处理解压缩 tile 的速度 (MTX).

**内存.** 内存提供压缩 tile 的速率为 $\mathrm{MBW}/\mathrm{Bytes}_{\mathrm{tile}}$ tile/s, 其中 $MBW$ 是内存带宽, $\mathrm{Bytes}_{\mathrm{tile}}$ 是压缩 tile 的字节数. 由于一个压缩权重 tile 只用于一次 TMUL 矩阵操作, 我们将 $1/\mathrm{Bytes}_{\mathrm{tile}}$ 称为矩阵到内存算术强度或 $\mathrm{AI}_{\mathrm{XM}}$. 它表示每从内存加载一个字节可以执行多少次矩阵操作, 与图 [3](#figure-03) Roofline 中使用的传统算术强度非常相似. 区别在于, 它的单位是每字节的矩阵操作数, 而不是每字节的 FLOPs. 在本文设置中, 压缩因子 (CF) 更高的压缩方案 ([第 2.2 节](#section-2-2)) 具有更高的 $\mathrm{AI}_{\mathrm{XM}}$. 因此, 压缩 tile 的 MEM 速率为 $\mathrm{MBW}\cdot\mathrm{AI}_{\mathrm{XM}}$ tile/s.

**向量硬件.** 向量硬件以 $\mathrm{VOS}/\mathrm{VO}_{\mathrm{tile}}$ 的速率解压缩 tile, 其中 $\mathrm{VOS}$ 是架构每秒可执行的向量操作数, $\mathrm{VO}_{\mathrm{tile}}$ 是每个 tile 所需的向量操作数. $\mathrm{VOS}$ 是向量吞吐量, 取决于架构. 例如对于我们的 SPR 系统, 它等于处理器频率 ($f$), 核心数 ($c$) 和每核心 SIMD 单元数的乘积. $\mathrm{VO}_{\mathrm{tile}}$ 取决于 kernel. 由于 GeMM 中只有权重矩阵需要解压缩, $\mathrm{VO}_{\mathrm{tile}}$ 实际表示每次矩阵操作所需的向量操作数. 我们将 $1/\mathrm{VO}_{\mathrm{tile}}$ 称为矩阵到向量算术强度或 $\mathrm{AI}_{\mathrm{XV}}$, 因为它表示每次向量操作可以执行的矩阵操作数. 因此, VEC 速率为 $\mathrm{VOS}\cdot\mathrm{AI}_{\mathrm{XV}}$ tile/s.

**矩阵硬件.** 矩阵硬件每秒可执行 $\mathrm{MOS}$ 次矩阵操作. $\mathrm{MOS}$ 取决于架构而非 kernel. 例如在 SPR 系统中, 它等于 $fc/16$, 因为每个核心都有一个 TMUL, 执行一次 tile 乘法需要 16 个周期. 因此, MTX 的速率就是 $\mathrm{MOS}$ tile/s.

**最终性能.** 最终性能由上述三种速率中最低的 tile 处理速率决定. 具体而言, 架构每秒可处理的 tile 数 (*TPS*) 为:

<span id="equation-01"></span>

$$
\mathrm{TPS} = \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}
$$

根据 [第 2.3 节](#section-2-3), 一次 TMUL tile 操作对应 $512N$ 次 FMA, 因此可以直接得到每秒 FLOPs (*FLOPS*) 的速率:

<span id="equation-02"></span>

$$
\mathrm{FLOPS} = 512N\cdot \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}
$$

我们将该方程称为 *Roof-Surface* 方程. *min* 中的三个项都可能成为限制性能的因素. 对于给定架构 (即固定的 *MBW*, $\mathrm{VOS}$ 和 $\mathrm{MOS}$), *min* 中有 *两个取决于 kernel 的变量*: $\mathrm{AI}_{\mathrm{XM}}$ 和 $\mathrm{AI}_{\mathrm{XV}}$. 它们构成 kernel 的 "特征": 如果两个 kernel 具有相同特征, 则预测性能相同. 相比之下, Roofline 模型中的 kernel 特征只有一个变量, 即传统的 FLOP 到内存 AI. 现在, 性能模型不能再用图 [3](#figure-03) 的二维 (FLOP 到内存 AI 和 FLOPS) 表示, 而需要三个维度: $\mathrm{AI}_{\mathrm{XM}}$ 为 x 维, $\mathrm{AI}_{\mathrm{XV}}$ 为 y 维, FLOPS 为 z 维.

<span id="figure-04"></span>

![图 4. (a) 三维 Roof-Surface 模型. (b) 基于 Roofline (R-L), Roof-Surface (R-S) 和实际性能测量的最优性能, 单位为 TFLOPs.](./deca/figure-04.png)

**图 4.** (a) 三维 Roof-Surface 模型. (b) 基于 Roofline (R-L), Roof-Surface (R-S) 和实际性能测量的最优性能, 单位为 TFLOPs.

图 [4a](#figure-04) 展示将公式 [2](#equation-02) (N=4, HBM) 绘制在三维空间中形成的 *Roof-Surface* 图. Roof-Surface 图包含三个区域, 以不同颜色表示. 每个区域中, Roof-Surface 公式的不同项最小, 因而限制性能. 蓝色子表面下的操作点受 MTX 因素限制, 绿色子表面下的点受 MEM 因素限制, 橙色子表面下的点受 VEC 因素限制. Kernel 性能用三维空间中的点表示. 可实现性能由整体表面界定, 而不是像 Roofline 模型那样由一条线界定. 因此, 我们将该模型称为 Roof-Surface. 整体表面上方的点无法实现.

图 [4a](#figure-04) 还包含对应不同压缩方案观测性能的红点. 可以看到, VEC-bound 区域下的红点 (MXFP4, BF16_10%, BF8_5%) 非常接近相应切三角形的顶端 (即几乎正好位于 Roof-Surface 上), 直观地表明它们受向量操作限制. MEM-bound 区域中的红点 (BF16_30%) 略低于 Roof-Surface, 说明对该点而言, 内存延迟等未绘制因素带来了一点性能损失.

在图 [4b](#figure-04) 中, 我们展示 Roofline (R-L) 和 Roof-Surface (R-S) 模型预测的最优性能值, 以及实际观测值. 对于几乎所有 kernel, Roof-Surface 都能给出准确的性能上界, 而 Roofline 可能偏差很大. 如果将许多 Roofline 预测值绘制在三维空间中, 它们会漂浮在 Roof-Surface 之上. 需要注意的是, 对于 BF8, BF16_50% 和 BF16_30% kernel, R-L 和 R-S 的性能估计相同, 因为两个模型都将这些 kernel 归类为 MEM-bound.

<span id="section-4-2"></span>

### 4.2 二维边界区域图

我们引入一种更易可视化的 Roof-Surface 二维表示, 称为 Bounding Region Diagram (BORD). BORD 是 Roof-Surface 在 xy 平面上的投影. BORD 不展示 FLOPS 信息, 但能准确指出所绘制因素中哪一个限制给定 kernel 的性能.

<span id="figure-05"></span>

![图 5. 二维边界区域图 (BORD).](./deca/figure-05.png)

**图 5.** 二维边界区域图 (BORD).

图 [5a](#figure-05) 展示 HBM SPR 的 BORD. 图中给出分隔三个区域的直线方程: $y=(\mathrm{MBW}/\mathrm{VOS})x$, $x=\mathrm{MOS}/\mathrm{MBW}$ 和 $y=\mathrm{MOS}/\mathrm{VOS}$. 图中还展示了图 [3](#figure-03)b 中使用 BF8 和 MXFP4 的不同压缩 GeMM kernel 的位置, 以及使用不同密度 BF16 的其他 kernel 的位置. 我们观察到绝大多数 kernel 都是 VEC-bound. 要达到图 [3](#figure-03)b 中 Roofline 的性能, 必须将这些点推离 VEC-bound 区域.

图 [5b](#figure-05) 展示 DDR SPR 的 BORD, 其 *MBW* 值更小. 此时 MEM-bound 区域的面积增大. 在 BORD 绘制的 $\mathrm{AI}_{\mathrm{XM}}$ 和 $\mathrm{AI}_{\mathrm{XV}}$ 取值范围内, MTX-bound 区域不再可见, 其面积被 MEM 区域占据. BORD 还显示, 除了密度为 20% 及更低的 BF8 外, 我们所有 kernel 都位于 MEM-bound 区域或非常接近该区域. 这解释了为什么软件解压缩方案在图 [3](#figure-03)a 的大多数设计点上都能达到 Roofline.

<span id="figure-06"></span>

![图 6. VOS 提高 4 倍时 HBM 的二维 BORD.](./deca/figure-06.png){.paper-figure-half}

**图 6.** VOS 提高 4 倍时 HBM 的二维 BORD.

最后, 图 [6](#figure-06) 展示 HBM SPR 变体将 VOS 中的向量吞吐量提高 4x, 试图消除向量瓶颈时的 BORD. 与图 [5a](#figure-05) 相比, VEC-bound 区域面积变小, MEM-bound 区域覆盖了更多 kernel. 然而, 即使将 VOS 提高 4x, 仍不足以让所有 kernel 摆脱 VEC-bound.

我们发现, 在图 [5a](#figure-05) 的 HBM SPR 变体中, 核心通常将超过 95% 的动态指令用于 tile 解压缩, 且已经使用了 40-80% 的提交槽. 因此, 将 VOS 提高 4x 不仅需要把 SIMD AVX 单元数量提高 4x, 还需要以难以接受的幅度增加核心的超标量宽度. 我们在 [第 7 节](#section-7) 进一步讨论这一方案及其他传统方案 (例如在不增加 AVX 单元数量的情况下增加向量宽度) 的局限, 并在 [第 9 节](#section-9) 评估这些局限.

<span id="section-5"></span>

## 5 DECA 概览与乱序调用

前面的分析表明, 若要通过传统方案隐藏解压缩开销, 就必须大幅扩展通用核心资源, 成本非常高. 因此, 我们提出 *DECA*, 一种 *ML 模型近核解压缩加速器*. DECA 将解压缩的向量处理从核心卸载出去. 本节首先介绍 DECA 的集成方式, 然后引入一种新机制和 ISA 扩展, 用于高效重叠 CPU 核心与近核加速器的操作.

<span id="section-5-1"></span>

### 5.1 DECA 部署与系统集成

如图 [7](#figure-07) 所示, 我们设想每个核心都关联一个 DECA. DECA 具有内存映射接口, 核心可以通过该接口写入命令和读取数据. DECA 包含处理单元 (PE), 控制寄存器和 tile 输出 (*TOut*) 寄存器. 核心使用特权存储指令写入控制寄存器, 配置 PE 按指定量化方案, 在有或无稀疏的情况下对 tile 解压缩. 配置过程包括填充 DECA 用于高效反量化的查找表 (LUT) ([第 6 节](#section-6)).

<span id="figure-07"></span>

![图 7. 核心旁的 DECA 部署.](./deca/figure-07.png){.paper-figure-half}

**图 7.** 核心旁的 DECA 部署.

DECA PE 从内存读取压缩 tile, 处理后将解压缩 tile 写入 TOut 寄存器. 随后 CPU 核心读取 TOut 寄存器, 使用其中的数据通过 AMX 指令执行 GeMM. PE 通过 L2 访问内存, 发出普通 load (但从不 store) 以及由 PE 内置预取器生成的预取请求. 与先前工作类似 [Gon22, Ger23, Sir23], DECA 与核心共享 L2 TLB, 因而使用 CPU 核心的虚拟地址空间.

一个 DECA 可以被多个进程使用. 一种方案是在上下文切换时保存和恢复 DECA 状态. 另一种方案是让 DECA 在上下文切换期间保留状态; 当新进程尝试使用 DECA 时触发操作系统陷阱, 由操作系统保存状态并重新配置 DECA.

<span id="section-5-2"></span>

### 5.2 DECA 与核心协同处理 tile

为了高性能执行 GeMM, 我们引入一种机制, 使用硬件双缓冲让 DECA 中的向量操作与 CPU 核心中的 AMX 操作重叠. 设计如图 [8](#figure-08) 所示. 一个 DECA 有两个 Loader 模块和两个 TOut 寄存器. Loader 从内存系统读取压缩 tile, 其中包含三种数据结构: 数据, 位掩码和缩放因子. Loader 还可以发出预取请求, 提前加载 tile. 一个 tile 的处理过程是: DECA 将其加载到 Loader (图 [8](#figure-08) 中的 D1), 在 DECA 向量流水线中解压缩 (D2), 再存入 TOut 寄存器 (D3). 然后核心读取它 (C1), 使用它执行 AMX 操作 (C2), 并向 Loader 传递该 tile 三种数据结构的起始地址和长度, 促使 Loader (C3) 开始获取下一个 tile. 如图所示, 双缓冲可以重叠两个 tile 的操作. 当核心读取并处理 tile *i-1* 时, DECA 同时读取, 处理并写出 tile *i*. 核心完成 *i-1* 后, 触发 tile *i+1* 的获取.

<span id="figure-08"></span>

![图 8. DECA 与 CPU 核心协同处理 tile.](./deca/figure-08.png)

**图 8.** DECA 与 CPU 核心协同处理 tile.

我们研究 CPU 核心与 DECA 通信的两种方式. 第一种使用对内存映射 DECA 接口的普通 store; 第二种使用 [第 5.3 节](#section-5-3) 描述的 ISA 扩展. 采用第一种方式时, 图 [9](#figure-09) 展示核心按照图 [8](#figure-08) 处理 tile 的伪代码. 关键指令位于第 4-6 行. 核心使用 *TLoad* (AMX 指令) 将 tile $T_{i-1}$ 从 DECA 的 TOut 寄存器加载到 tile 寄存器 TReg$_1$ (第 4 行). 然后在 *TComp* 指令 (执行 GeMM 的 AMX 指令) 中使用该 tile, 并将输出保存到 tile 寄存器 TReg$_2$ (第 5 行). 最后, 核心使用普通 store 将 tile $T_{i+1}$ 的元数据 (记为 $M_{i+1}$) 写入 DECA Loader2 的内存映射寄存器. 该写操作促使 Loader2 开始获取 tile $T_{i+1}$. 在第 4-6 行执行的同时, DECA 正在解压缩 $T_i$.

<span id="figure-09"></span>

```text
............
DECA_ldr1 <- ST M_i
Fence
TReg_1 <- TLoad T_i-1
TReg_2 <- TComp TReg_1
DECA_ldr2 <- ST M_i+1
Fence
TReg_1 <- TLoad T_i
TReg_2 <- TComp TReg_1
............
```

**图 9.** 基于 store 调用 DECA 的 CPU 核心伪代码.

<span id="figure-10"></span>

```text
............
TReg_1 <- TEPL M_i-1
TReg_2 <- TComp TReg_1
TReg_1 <- TEPL M_i
TReg_2 <- TComp TReg_1
TReg_1 <- TEPL M_i+1
TReg_2 <- TComp TReg_1
............
```

**图 10.** 基于 TEPL 调用 DECA 的 CPU 伪代码. 每次迭代中, 架构 tile 寄存器 TReg$_1$ 和 TReg$_2$ 会重命名为不同的物理 tile 寄存器.

图 [9](#figure-09) 还展示了前一次迭代的一部分 (第 2 行) 和后一次迭代的一部分 (第 8-9 行). 为防止内存操作错误重排, 我们在每次迭代中加入一个内存栅栏. 具体而言, tile $T_i$ 的加载 (第 8 行) 不应早于将 $T_i$ 的元数据写入 DECA Loader1 的控制寄存器 (第 2 行), 后者会重置 TOut 寄存器 1 并从内存发起 tile 获取. 由于这两条指令相互没有依赖, 我们在第 3 行放置栅栏. 每次迭代都有一个栅栏.

遗憾的是, 这种方式的性能可能有限, 原因有二. 第一, 每次迭代都有一个阻止跨迭代重叠的栅栏. 第二, 迭代内部没有指令重叠: 第 4, 5 行的指令存在真实依赖, 第 6 行的 store 只有到达重排序缓冲区 (ROB) 头部时才能更新. 所有指令都被串行执行, 就像核心按序执行一样. 因此, 每次迭代中核心与 DECA 通信的延迟 (包括 load 和 store) 都会完全暴露.

<span id="section-5-3"></span>

### 5.3 乱序调用的 ISA 支持

为恢复乱序执行并隐藏核心与 DECA 的通信, 我们提出另一种依赖 CPU AMX ISA 扩展的方案. 我们将该扩展称为 *Tile External Preprocess and Load* (TEPL). 其主要思路是在硬件中将第 2 行和第 8 行的指令合并为一条指令, 从而消除图 [9](#figure-09) 中每次迭代的栅栏. 该指令使用元数据更新 Loader 的控制寄存器, 触发 tile 获取, 只有当 DECA 解压缩 tile 并将其存入核心 tile 寄存器 (例如 TReg$_1$) 后才返回核心.

TEPL 指令的参数包括一个存有 tile 元数据的源寄存器和一个目标核心 tile 寄存器. 元数据被传递给 DECA 以启动解压缩. 此外, 任意时刻可执行的 TEPL 指令最多等于 DECA Loader 的总数 (即两个). 结构冒险会阻止更多 TEPL 执行, 以避免覆盖加速器调用, 因为每个 DECA Loader 一次只能处理一个 tile.

采用该设计后, 图 [9](#figure-09) 中的代码被改写为图 [10](#figure-10). 栅栏被移除, 每次迭代只有两条指令 (例如第 4, 5 行). 由于 TReg$_1$ 和 TReg$_2$ 会重命名, 迭代之间不存在寄存器依赖. 不过, 结构冒险会使第 6 行的 TEPL 停顿, 直到前两个 TEPL 之一完成.

上下文切换只能发生在两条指令之间. 因此, 新进程尝试使用 DECA 时需要保存和恢复的状态只有 DECA 控制寄存器和 LUT, 不包括任何 tile 数据.

为了支持这些指令, 核心配备类似 load-store queue 的 *TEPL Queue*, 以及两个分别连接 DECA Loader 的 *TEPL 执行端口*. TEPL 指令 $i$ 进入 ROB 后会存入该队列. 当源寄存器可用且存在空闲 TEPL 执行端口时, $i$ 就会发射到 DECA.

为了获得高性能, TEPL 会尽早发射到 DECA, 不会等到到达 ROB 头部. 因此, 它们像 load 指令一样投机且乱序执行. 投机调用 DECA 始终安全, 因为 DECA 不更新内存状态. 如果 TEPL 指令尚未完成时核心需要刷新流水线 (例如发生分支预测错误或异常), 核心会向 DECA 发送 squash 信号. 此时 DECA 会中止正在进行的 tile 操作, 无论操作处于何种状态. 核心可以安全地重新发射同一条 TEPL.

总体而言, 该设计隐藏了核心与 DECA 之间的通信. 核心无需栅栏即可执行, 并能重叠多个 tile 的操作. TEPL 不只适用于 DECA, 核心还可以用它与其他类似 DECA 的近核 tile 预处理加速器通信.

<span id="section-6"></span>

## 6 DECA 微架构设计

下面介绍使 DECA 保持高解压缩性能, 同时支持丰富压缩方案的微架构. 为简化说明, 本文其余部分假设 DECA 的输出 tile 为 BF16 格式. DECA 也可以简单配置为生成 I8 输出 tile.

<span id="section-6-1"></span>

### 6.1 DECA 微架构

图 [11](#figure-11) 展示了 DECA PE 的微架构. 下面介绍其中的多个组件.

<span id="figure-11"></span>

![图 11. DECA PE 微架构.](./deca/figure-11.png)

**图 11.** DECA PE 微架构.

**内存访问.** DECA 有两个 Loader, 每个 Loader 由一个 *Load Queue* (LDQ) 和一个预取器 (PF) 组成. LDQ 访问内存, 读取压缩权重, 位掩码和缩放因子. DECA 调用时, CPU 提供这些结构的内存地址基址和长度, 作为元数据的一部分. 当请求的 cache line 从内存到达时, 根据其中包含的三种数据类型之一, 将其放入 *Sparse Quantized Queue* (SQQ), *Bitmask Queue* 或 *Scale Factor Queue*. PF 观察 tile 使用的地址基址和长度, 预测后续 tile 的地址和长度, 再生成预取请求将数据带入 L2 缓存. PF 的激进程度会动态调整, 以维持较高的 L2 MSHR 占用率.

**流水线阶段.** 流水线分为三个阶段, 分别负责反量化, 扩展 (即解稀疏) 和缩放. 每个阶段都有自己的输出寄存器以实现流水 (SD, DD 和 TOut). 反量化阶段从 SQQ 读取值, 使用 $L$ 个查找表组成的阵列 (*LUT Array*) 对其反量化, 并将反量化后的 BF16 值写入 *Sparse Dequantized* (SD) 寄存器. 这些值可能仍然稀疏, 即跳过零值后连续存储. 扩展阶段根据位掩码指示的位置插入零, 对数据进行解稀疏. 该操作使用由扩展索引控制的交叉开关 (*XBAR*) 完成, 扩展索引则由 *Parallel Prefix Sum* 电路根据位掩码生成. 结果写入 *Dense Dequantized* (DD) 寄存器, 其中包含稠密 (即显式包含零值) 的反量化数据. 最后, 如果使用分组量化, 缩放阶段将缩放因子与 BF16 值相乘, 应用相应缩放, 再将最终值写入 *TOut* 寄存器. 图中的红色箭头表示关键路径.

**复制的模块.** 一个 DECA PE 包含两个 Loader 和两个 TOut 寄存器, 用于重叠 DECA 与 CPU 的操作. 因此, 如图 [11](#figure-11) 所示, PE 复制了 LDQ, PF, 输入队列 (SQQ, Bitmask Queue 和 Scale Factor Queue) 以及 TOut. 一个 Loader 可以提供数据, 同时流水线处理另一个 Loader 提供的数据. 位掩码处理电路主要执行 1 位数据的加法, 也会复制该电路以隐藏其延迟. 流水线其余部分不复制, 每次供一个 Loader-TOut 对使用.

**向量操作 (vOp).** 生成一个始终包含 512 个 BF16 元素的解压缩 BF16 tile 需要多个周期. 这是因为流水线每次生成包含 *W* 个元素的输出块, 每个输出块使用一次 DECA 向量操作 (vOp). 没有流水线气泡时, 每个周期都会生成一个新块. vOp 从 SQQ 读取数据, 在各流水线阶段执行, 最后将 W 个元素写入 TOut. vOp 利用流水: 当一个 vOp 进入扩展阶段时, 下一个 vOp 可以进入反量化阶段. 一个 tile 的 vOp 按顺序处理, 只要 (1) 输入已从内存到达, 且 (2) 第一个流水线阶段空闲, 就可以进入流水线.

没有稀疏时, 一个 vOp 从 SQQ 读取 W 个元素. 有稀疏时, 由于 SQQ 不包含零值, 所需元素少于 W 个. 我们将某个 vOp 需要从 SQQ 读取的元素称为该 vOp 的窗口 (*Wnd*). 为确定 Wnd 的大小, POPCNT 电路统计位掩码中 "1" 的数量, 确定当前 Wnd 的结束位置和下一个 Wnd 的开始位置. 后者就是下一个从 SQQ 读取数据进入流水线的位置.

**LUT 阵列组织.** DECA 的反量化阶段支持最多 8 比特的量化数, 最多可表示 256 个不同值. 因此, LUT 阵列中的每个 $L$ 个 LUT 都存储 256 ($2^8$) 个 BF16 值. 对 8 比特值反量化, 就是以该 8 比特值作为 LUT 地址进行查找. DECA 包含 $L$ 个 LUT, 允许并行反量化多个值. 每个 LUT 内部又分为 4 个更小的子 LUT, 每个子 LUT 有一个读端口和 64 ($2^6$) 个条目. 如果量化数据位宽为 6 比特或更少, 可以独立使用 4 个子 LUT, 从一个含 256 个条目的 "大" LUT 中执行 4 次读取. 对于低于 6 比特的量化, 部分 LUT 条目是冗余的, 运行时不会使用.

**气泡与 Roof-Surface.** 为限制 DECA 面积, 我们将 "大" LUT 的数量设为 $L < W$. 如果 vOp 的 Wnd 大于 $L$ 个元素, vOp 会占用反量化阶段超过一个周期, 从而在流水线中产生一个或多个 *气泡*, 降低 vOp 吞吐量. 例如, 稠密 8 比特量化方案的 Wnd 为 W, 因此一个 vOp 总要花费 $W/L$ 个周期反量化. 虽然设置 $L < W$ 会限制稠密量化方案的 DECA 吞吐量, 但这不是主要问题, 因为 BF8_100% 和 MXFP4 等稠密方案只需较低向量吞吐量 (即 VOS) 就能离开向量 (VEC) 区域. 这一点可从图 [5](#figure-05) 的 BORD 中看到.

另一方面, 更稀疏的方案需要更高 VOS 才能离开 VEC-bound 区域. 幸运的是, DECA 流水线自然实现了这一点: 随着稀疏程度增加, vOp 的 Wnd 大于 $L$ 的概率降低. 因此, 稀疏方案产生的气泡更少, 在相同 $L$ 下自然比稠密方案获得更高吞吐量. 低位宽方案也有相同表现, 因为它们可以从 LUT 阵列并行读取超过 $L$ 个值.

**通用性与性能.** DECA 支持 8 比特及更低位宽的量化格式, 分组量化和非结构化稀疏, 覆盖当前及可能出现的大多数模型压缩方案. DECA 的设计很灵活: 通过改变 LUT 阵列中的值和/或使用不同缩放因子, 无需重新设计硬件即可支持丰富的格式. 此外, 不需要的阶段可以跳过 (例如只有量化而没有稀疏). 在性能方面, DECA 的主要优势是用一次 vOp 替代多个向量 (AVX) 指令, 完成整个解压缩过程, 包括反量化, 扩展和缩放. vOp 数量减少, *提高了 $\mathrm{AI}_{\mathrm{XV}}$* ([第 4 节](#section-4)), 使数据点远离 VEC 区域. 最后需要注意, DECA 只对非零值进行高效反量化; 对传统向量 ISA 的 CPU 而言, 扩展阶段存在数据依赖分支, 很难做到这一点.

<span id="section-6-2"></span>

### 6.2 微架构的定量设计

前文讨论了 Roof-Surface 模型如何 *定性地* 影响 DECA 设计. 例如, 模型建议通过优化 $\mathrm{AI}_{\mathrm{XV}}$ 设计更高性能的加速器, 而不是盲目扩展 CPU 宽度和 AVX 资源. 下面讨论如何 *定量地* 使用该模型确定 DECA 的 $W$ 和 $L$ 参数, 推导出均衡的设计.

考虑公式 [2](#equation-02). 我们需要表达方程中的参数如何依赖 $W$ 和 $L$. 实际上, 只有 $\mathrm{AI}_{\mathrm{XV}}$ 依赖 $W$ 和 $L$. $\mathrm{VOS}$ 为 $c\cdot1\cdot f$, 因为 $c$ 个 CPU 核心各有一个 DECA PE, 每周期最多完成一个 vOp, 且运行在核心频率下. 另一方面, 不同 kernel 的 $\mathrm{AI}_{\mathrm{XV}}$ 取决于 DECA 的 $W$ 和 $L$ 参数. 要计算它, 需要将每个 tile 所需的 vOp 数量和每个 tile 产生的气泡数量相加.

<span id="table-02"></span>

![表 2. DECA 与其他核内/近核加速器的比较.](./deca/table-02.png)

**表 2.** DECA 与其他核内/近核加速器的比较.

每个 tile 的 vOp 数量为 $\#\mathrm{vOps}=512/W$, 因为每个 tile 有 512 个元素, 一次 vOp 产生 W 个元素. 我们将每个 tile 的气泡数量表示为 $\#\mathrm{bbl}=\#\mathrm{vOps}\cdot\mathit{bpv}$, 其中 $\mathit{bpv}$ 是每个 vOp 的气泡数量. 由于气泡只能由反量化阶段资源不足产生, 我们用 $L_q$ 表示每周期最多可反量化的元素数. 对于 8 比特量化方案, $L_q=L$; 对于 7 比特, $L_q=2*L$; 对于 6 比特及以下, $L_q=4*L$. 没有稀疏时, $\mathit{bpv}=\lceil W/L_q\rceil-1$. 有稀疏时, 气泡生成不是确定的, 因为它取决于压缩 tile 中非零值的数量. 对于密度为 $d$ 的矩阵, 假设非零值均匀分布, 则连续 W 个矩阵元素中的非零值数量服从参数为 W 和 d 的二项分布. 气泡数量的期望值为:

$$
\begin{aligned}
\mathit{bpv} &=  \sum\nolimits_{k=0}^{\frac{W}{L_q}-1} k \cdot [F((k+1)L_q; W, d) - F(kL_q; W, d)]
\end{aligned}
$$

其中 $F(i;W,d)$ 是二项分布的累积分布函数. 最后, $\mathrm{AI}_{\mathrm{XV}}$ 为 $1/[\#\mathrm{vOps}\cdot(1+\mathit{bpv})]$.

现在已经具备使用 Roof-Surface 模型进行分析设计空间探索 (DSE) 所需的一切. 例如, 可以绘制不同 ($W$, $L$) 对应的 BORD, 选择以最低 DECA 硬件成本将所有 kernel 推出 VEC-bound 区域的参数对 (见 [第 9.2 节](#section-9-2)).

<span id="section-7"></span>

## 7 处理解压缩瓶颈的 DECA 替代方案

在 [第 5 节](#section-5) 和 [第 6 节](#section-6) 中, 我们讨论了 DECA 如何在支持丰富压缩方案的同时保持高解压缩性能. 下面讨论使用 DECA 的两种替代方案的缺点: 扩展 CPU 核心的向量资源, 或采用其他核内/近核加速器设计.

1. **传统的 CPU 向量资源扩展.** [第 4 节](#section-4) 的 Roof-Surface 分析表明, 要隐藏大部分解压缩开销, 需要将向量吞吐量 (VOS) 提高 4x 以上. 通过传统方式扩展核心向量资源来支持如此提升非常困难. 一种方法是将 SIMD AVX 向量单元数量增加 4x 以上. 然而, 如 [第 4 节](#section-4) 所述, 核心已经使用了 40-80% 的提交槽. 因此, 如此大幅增加向量单元数量, 就需要显著增加超标量核心宽度. 这并不理想, 因为核心面积会随超标量宽度呈二次方增长 [Pal97]. 另一种方法是增加 SIMD AVX 向量宽度. 这需要新的 AVX 指令, 处理至少 2048 位, 跨多个 cache line 的操作数. 然而, 支持 AVX2048 需要大幅修改 ISA 和流水线 (例如重新设计所有向量指令的更宽版本, 增加新的寄存器文件等). 此外, 向核心输入如此大的向量, 至少需要增加 L1 缓存的端口数, 这会损害 L1 访问延迟和核心周期时间, 影响核心在通用工作负载上的性能. 在 [第 9 节](#section-9) 中, 我们定量比较 DECA 与这些替代方案.
2. **使用矩阵操作的核内加速器.** TMUL, RASA [Jeo21] 等传统矩阵单元无法处理压缩 tile. 为避免 tile 解压缩, 一些核内加速器设计 [Jeo23, Pel24, Nvi24d] (例如 VEGETA [Jeo23]) 为矩阵单元增加了对特定结构化稀疏模式的支持. 这种方法会增加核心的硬件复杂度 (例如更大的矩阵单元, 更多架构寄存器以及寄存器重命名改动). 此外, 虽然跳过涉及零值的计算可以提高矩阵吞吐量 ($\mathrm{MOS}$), 但 [第 4 节](#section-4) 的 *Roof-Surface* 分析表明, 对我们的 kernel 而言这种提升并不必要: 大多数 kernel 摆脱向量限制区域后会转而受内存限制. 其他设计为矩阵单元增加对更高效低比特量化格式的原生支持 [Jan24, Nvi24d]. 然而, 对每种支持的格式, 这类设计都需要在矩阵单元中加入额外硬件. 此外, 如果出现此前未知的新量化格式, 就必须重新设计硬件. 相比之下, DECA 无需为每种格式增加硬件, 只需改变 LUT 阵列中的值和/或使用不同缩放因子, 就能支持非常丰富的量化格式. DECA 的灵活性使其无需重新设计硬件即可支持未来的量化格式. 原则上, 所有 DECA 硬件 (即 LUT 阵列, 扩展和缩放电路等) 都可以集成到矩阵乘法单元中. 然而, DECA 的解耦方式具有重要优势. 第一, 它更灵活: 解压缩器的输出还可以送入另一个加速器, 写回内存或用于其他场景. 第二, 将带有独立 Loader 的加速器连接到 L2 后, DECA 可以更有效地获取和预取数据. 最后, 所需的 CPU 核心 ISA 和流水线改动很少, 降低了影响核心通用工作负载性能的风险.
3. **使用向量操作的核内/近核加速器.** SPADE [Ger23] 和 SAVE [Gon20] 是面向稀疏应用, 设计为与 CPU 集成的加速器. 然而, 它们不依赖矩阵单元, 而是使用向量单元执行实际 GeMM. 对于高度稀疏的矩阵, 这种方法或许可行; 但机器学习模型中的矩阵通常只有中等稀疏度, 此时必须利用矩阵单元的高吞吐量 [Yan24m].

与其他最先进的核内/近核加速器相比, 表 [2](#table-02) 总结了 DECA 实现的独特特征组合. 第一, DECA 是首个同时支持丰富量化方案以及结构化或非结构化稀疏的设计, 并通过与 TMUL 矩阵单元协同实现高 GeMM 吞吐量. 第二, 借助投机调用, 它是首个支持与核心细粒度交错执行的近核加速器设计. 最后, 它只需对核心流水线做少量改动, 这些改动还可以复用于其他近核加速器 ([第 5.3 节](#section-5-3)).

<span id="section-8"></span>

## 8 方法

**模拟与系统参数.** 为评估本文方案, 我们使用基于 Sniper [Car14] 的内部模拟器模拟一个具有 SPR 类参数且完整支持 AMX 的 56 核服务器. 我们分别评估 DDR5 和 HBM 设计, 可实现的内存带宽约为 260GB/s 和 850GB/s. 我们向模拟器加入 (1) DECA PE, 以及 (2) 用于在核心流水线中支持 TEPL 的 TEPL 队列和端口. 核心和 DECA PE 均运行在 2.5GHz. 基线 PE 的参数为 W=32, L=8, 同时我们在 [第 9.2 节](#section-9-2) 评估其他选项.

**软件与 DECA 控制代码生成.** 我们使用 Intel Libxsmm 压缩 GeMM kernel ([第 2.4 节](#section-2-4)) 作为软件基线. 为调用 DECA, 我们修改 libxsmm JIT 编译器, 用 TEPL 指令替换 AVX 解压缩序列.

为单独评估 DECA 对压缩 GeMM 的有效性, 我们实现一个只包含全连接 (FC) 层的大型级联, 并使用 Parlooper [Geo23] 进行循环并行化. 这些层的权重矩阵包含 $\approx250$ million 个参数, 与 Llama-2-70B 的大型 FC 层相近. libxsmm 和 Parlooper 已集成到 Intel Tensor Processing Primitives (TPP) Framework [Geo21a] 中, 该框架支持 CPU 上端到端的 Llama-2 和 OPT 推理. 因此, 对于纯软件 LLM 推理, 我们直接使用 TPP; 对于使用 DECA 的推理, 则调用加入 TEPL 的 libxsmm kernel. 我们测试 batch size 1-16. 我们的模拟器兼容所有框架.

**压缩方案.** 在评估中, 我们将 BF16, BF8 和 MXFP4 分别称为 Q16, Q8 和 Q4. 评估范围限于这三种方案, 因为 libxsmm 已经支持它们. 对于 Q16 (仅稀疏) 和 Q8 (量化加稀疏), 我们还评估权重密度从 50% 到 5% 的非结构化稀疏. Q4 稀疏 kernel 当前尚未加入 libxsmm, 因此没有可与 DECA 性能比较的参考数据. 对于端到端 Llama-2-70B 和 OPT-66B 推理, 未压缩 Q16 基线, 密度为 50% 的 Q16 (Q16_50%) 以及 Q8_100% 无法装入 64GB HBM. 因此, 我们为这些方案模拟更大的 HBM 容量. 需要注意, Q4 的性能也代表带缩放因子的 INT4 压缩方案, 例如 AWQ [Lin23d].

**面积估算.** 我们估算 W=32, L=8 的 DECA 设计面积. 对于内存结构 (例如 LDQ 和 SQQ), 寄存器及 LUT 阵列, 使用 CACTI [Bal17]. 对于交叉开关和 BF16 乘法器, 分别使用 [Cak15] 和 [Zha19j] 的数据. 然后使用 [Sca17] 将数据缩放到 7nm. 我们估计 56 个 DECA PE 的总面积约为 2.51 $\mathrm{mm}^2$. Loader, SQQ, Bitmask Queue, Scale Factor Queue 和 TOut 寄存器约占 DECA 面积的 55%, LUT 阵列占 22%, 其余部分占 23%. 考虑到 56 核 SPR 的芯片总面积约为 1600 $\mathrm{mm}^2$ [Wik24], DECA 的面积开销小于 0.2%.

<span id="section-9"></span>

## 9 评估

<span id="section-9-1"></span>

### 9.1 用于压缩 GeMM 的 DECA

图 [12](#figure-12) 和 [13](#figure-13) 展示不同压缩方案下, libxsmm 软件方案 (*Software-only*) 和 DECA 相对于未压缩 BF16 基线的加速比. 我们还加入 *Roofline* 模型给出的 *Optimal* 加速比, 该模型假设所有 VEC 开销都被隐藏. 图中压缩方案按压缩因子递增排列. 结果对应 N=1.

<span id="figure-12"></span>

![图 12. DDR, $N=1$ 时压缩 GeMM 的加速比.](./deca/figure-12.png)

**图 12.** DDR, $N=1$ 时压缩 GeMM 的加速比.

<span id="figure-13"></span>

![图 13. HBM, $N=1$ 时压缩 GeMM 的加速比.](./deca/figure-13.png)

**图 13.** HBM, $N=1$ 时压缩 GeMM 的加速比.

在 DDR 设置下 (图 [12](#figure-12)), 只有压缩因子较高时 DECA 才比纯软件方案更快. 这符合预期, 因为根据图 [5b](#figure-05) 的 BORD, 只有高压缩因子方案受 VEC 限制. 加速比最高达到 1.7$\times$. 在 HBM 设置下 (图 [13](#figure-13)), DECA 几乎对所有压缩方案都能加速. 这是因为如图 [5a](#figure-05) 的 BORD 所示, 几乎所有方案都受 VEC 限制. 加速比最高达到 4.0$\times$. 在 DDR 和 HBM 两种设置中, DECA 性能都接近最优, 说明 VEC 开销已成功隐藏. 我们将该分析重复到 batch size 为 N=16 的情况, 观察到相似结果.

加入 DECA 的核心在向量处理方面远强于传统核心. 图 [14](#figure-14) 比较 DDR 设置, N=4 时两类核心在所有压缩方案上的平均性能. 图中比较了不同核心数: 8, 16,..., 56. 可以看到, 例如 16 个加入 DECA 的核心性能高于 56 个传统核心. 多出的核心可以释放给不消耗太多内存带宽的其他工作负载, 或通过断电来节能.

<span id="figure-14"></span>

![图 14. DDR, $N=4$ 时不同压缩方案的 TFLOPS.](./deca/figure-14.png)

**图 14.** DDR, $N=4$ 时不同压缩方案的 TFLOPS.

为进一步了解纯软件系统和 DECA 系统的性能, 表 [3](#table-03) 展示内存带宽, TMUL 以及 CPU AVX 单元或 DECA 的利用率百分比. 由于性能与 TMUL 利用率成正比, 表中显示 DECA 系统的性能远高于纯软件系统. 此外, 三个组件的操作会重叠, 因此利用率最高的组件最终成为瓶颈. 在纯软件系统中, 几乎所有密度下的瓶颈都是 AVX 向量单元. 这一观察验证了 Roof-Surface 的预测. 使用 DECA 后, 内存得到更充分利用, 直接带来性能提升. 需要注意, 虽然更稀疏的 kernel 执行时间更短, DECA 利用率仍大致不变. 如 [第 6 节](#section-6) 所述, DECA 会自然地为稀疏方案提供更高吞吐量.

<span id="table-03"></span>

![表 3. Q8, $N=1$, HBM 时的组件利用率.](./deca/table-03.png){.paper-table-narrow}

**表 3.** Q8, $N=1$, HBM 时的组件利用率.

图 [15](#figure-15) 将 DECA 与扩展 CPU 核心向量资源这一缓解解压缩开销的方案进行比较. 我们将加入 DECA 的核心与以下核心比较: (1) 向量 AVX 单元多 4$\times$ 的核心 (*More AVX Units*), 或 (2) AVX 单元宽度增加 4$\times$ 的核心 (*Wider AVX Units*). 对于更宽的 AVX2048 单元, 我们乐观地建模为从解压缩循环四次迭代中的三次删除动态指令. 由于不修改系统 cache line, 每次 AVX2048 内存操作会作为 4 次 cache line 大小的操作执行. 对于非 DECA 系统, 我们不扩展核心超标量宽度或 L1 端口数量, 因为如 [第 7 节](#section-7) 所述, 这些改动代价过高. 从图中可以看到, 传统向量扩展方法的性能远低于 DECA.

<span id="figure-15"></span>

![图 15. HBM, $N=1$ 时 DECA 与传统向量扩展的比较.](./deca/figure-15.png)

**图 15.** HBM, $N=1$ 时 DECA 与传统向量扩展的比较.

<span id="section-9-2"></span>

### 9.2 使用 Roof-Surface 进行设计空间探索

DECA 的 W 和 L 参数决定 DECA 的解压缩速度, 但过大的值可能增加面积而没有实际收益. 为此, 我们使用 *Roof-Surface* 检查不同 {W, L} 对的性能. 为了确定 DECA 的规模, 我们选择预测性能达到饱和的最小 {W, L} 对 (即预测所有 kernel 都不再受 VEC 限制). 根据模型, 该参数对为 {W=32, L=8}. 图 [16](#figure-16) 比较 HBM SPR 系统在无 DECA (a) 和有 DECA (b) 时不同 {W, L} 规模的 BORD: {W=8, L=4} (配置不足), {W=32, L=8} (最佳) 和 {W=64, L=64} (配置过度).

<span id="figure-16"></span>

![图 16. 无 DECA 及不同规模 DECA 时的 HBM BORD.](./deca/figure-16.png)

**图 16.** 无 DECA 及不同规模 DECA 时的 HBM BORD.

我们观察到, 与 CPU 相比, DECA 的每秒向量操作数 ($VOS$) 参数更小, 因为其 VEC-bound 区域更大. 然而, 如 [第 6 节](#section-6) 所述, DECA 减少了每次矩阵操作所需的向量操作数 (即提高 $\mathrm{AI}_{\mathrm{XV}}$). 配置不足的 {W=8, L=4} DECA 无法将 kernel 推出 VEC-bound 区域. 配置过度的 {W=64, L=64} 可以将 kernel 推出该区域, 但超出了所需程度. 我们模拟这些参数对的性能, 验证模型的准确性. 结果发现, DECA-best 系统比配置不足的 DECA 快 2$\times$. 配置过度的 DECA 系统只比 DECA-best 快不到 3%. 同时, DECA-best 的成本低得多: LUT 数量少 8$\times$, W 减半. 总体而言, Roof-Surface 模型准确捕获了矩阵-向量-内存交互的动态, 可以指导微架构决策.

<span id="section-9-3"></span>

### 9.3 DECA 集成与 TEPL 分析

下面评估 DECA 与核心集成时作出的不同设计决策. 我们从一个基础配置开始: DECA 从 LLC 读取压缩 tile (绕过 L2), 将解压缩 tile 写入 L2 供核心读取, 并通过普通 load, store 和栅栏调用. 随后逐步增强该配置: (1) 允许加速器从 L2 读取压缩权重并使用 L2 预取器 (*+Reads L2*), (2) 使用自己的预取器而不是 L2 预取器 (*+DECA prefetcher*), (3) 写入 TOut 寄存器而不是 L2 (*+TOut Regs*), 以及 (4) 使用 TEPL 指令代替 load, store 和栅栏 (*+TEPL (DECA) *).

<span id="figure-17"></span>

![图 17. HBM, $N=4$ 时的 DECA 集成功能.](./deca/figure-17.png)

**图 17.** HBM, $N=4$ 时的 DECA 集成功能.

图 [17](#figure-17) 展示逐步应用这些优化后, Q8 在不同密度下相对基础设计的加速比. 可以看到, *+Reads L2* 在所有密度下都能提升性能. 收益来自系统已有的 L2 硬件预取器, 它获取未来 tile, 隐藏内存和 LLC 访问延迟. *+DECA prefetcher* 通过使用 DECA 预取器而非默认 L2 预取器, 进一步改善性能. *+TOut Regs* 和 *+TEPL (DECA) * 减少或隐藏 DECA-核心通信延迟, 是实现乱序调用所必需的. 具体而言, *+TOut Regs* 允许核心直接从 DECA 获取数据, 不必绕行 L2 的更长路径. 此外, *+TEPL (DECA) * 将通信与计算重叠, 有效隐藏通信延迟. 随着密度降低, *+TOut Regs* 和 *+TEPL (DECA) * 的效果增强, 因为 DECA 处理低密度 tile 所需时间更少, 而与核心通信的开销保持不变. 因此, 在较低密度下通信成本更加明显. 需要注意, TEPL 对低密度模型非常有效: 密度为 5% 时, 性能翻倍.

<span id="section-9-4"></span>

### 9.4 用于 LLM 推理的 DECA

最后, 我们展示 DECA 对 LLM 下一个 token 生成 (包括非 GeMM 阶段) 的性能收益. 表 [4](#table-04) 展示 Llama2-70B 和 OPT-66B 模型在带 HBM 的 SPR 上, 输入 128 个 token, 输出 128 个 token, batch size 为 1 和 16 以及不同压缩方案下的下一个 token 延迟. 我们比较软件解压缩 (*SW*) 与本文方案 (*DECA*). 如前所述, 对未压缩 BF16 基线模型, 我们假设更大的 HBM 容量进行模拟. 可以看到, DECA 相比 *SW* 将下一个 token 的时间缩短 1.6$\times$-2.6$\times$. 这相当于相比未压缩基线模型获得 2.5$\times$-5.0x 的加速. 对于更短或更长的 token 序列, 我们也观察到相似结果.

<span id="table-04"></span>

![表 4. Llama2-70B/OPT-66B 下一个 token 的延迟 (ms).](./deca/table-04.png){.paper-table-narrow}

**表 4.** Llama2-70B/OPT-66B 下一个 token 的延迟 (ms).

<span id="section-10"></span>

## 10 其他相关工作

**解耦加速器.** 已有多种面向 ML 和科学应用稀疏性的独立解耦加速器 [Zha16c, Lu19, Ger24, Par17a, Che22b, Heg19, Sri20, Gon19a, Han16a, Adi23, Aan23]. 其他解耦加速器依赖量化 [Zhu24d, Ryu22, Jan24]. 近年来, 注意力加速器也逐渐流行 [Wan20b, Kac24, Lu21, Ham21, Ham20]. 解耦加速器需要较大的面积和功耗预算 [Jeo23], 且会遭受数据移动开销 [Ger23]. 因此, 研究者提出了与 CPU 集成的加速器 [Ger23, Jeo23, Gon20, Gon22, Jeo21, Nas22]. DECA 属于这一方向. 我们在 [第 7 节](#section-7) 讨论了其他核内/近核加速器的缺点.

**协同向量-矩阵处理.** 多种架构包含异构的矩阵和向量单元, 可以用 *Roof-Surface* 模型刻画它们的交互. 例如 Tandem 处理器 [Gho24], AWS Trainium [Bsh24, Fan24d], TPU [Nor21], 以及带有 Tensor Core 和 SIMT Core 的 GPU.

**受 DECA 启发的解压缩引擎对 GPU 的实用性.** 与 TMUL 类似, GPU Tensor Core 只支持有限的量化格式, 不支持非结构化稀疏. 因此, Flash-LLM 等 GPU kernel [Xia23b] 采用类似 libxsmm 的方法: 通过软件解压缩数据, 再将其送入 Tensor Core. 虽然有效, 但 Flash-LLM 会给 SM 的 L1/共享内存带来压力, 阻碍 Tensor Core 和 HBM 得到充分利用. 因此, 我们认为受 DECA 启发的解压缩引擎也可能适用于 GPU. NVIDIA 最近引入 TMA 加速器 [Luo24b], 用于将数据从内存供给 Tensor Core. 为 TMA 增加受 DECA 启发的解压缩能力是一个有趣的未来方向.

<span id="section-11"></span>

## 11 结论

为了改善配备核内 GeMM 引擎和 HBM 的先进 CPU 平台上的 LLM 推理, 本文作出三项贡献: *Roof-Surface* 性能模型, *DECA* 近核 ML 模型解压缩加速器, 以及支持乱序调用加速器的 TEPL ISA 扩展. 评估表明, DECA 能有效加速压缩 GeMM 和 LLM 推理.
