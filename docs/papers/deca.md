---
title: DECA Accelerator
createTime: 2026-08-23
permalink: /papers/deca/
---

> Gerasimos Gerogiannis, Stijn Eyerman, Evangelos Georganas, Wim Heirman, Josep Torrellas. 首次提交至 arXiv 的日期为 2025-05-25, 当前版本为 v2 (2025-08-08). Intel Corporation, Intel Labs, University of Illinois at Urbana-Champaign. [arXiv:2505.19349](https://arxiv.org/abs/2505.19349). [原始 PDF](/paper/deca.pdf). [TeX 源码](https://arxiv.org/e-print/2505.19349). 原论文标题为 "DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model".

## 摘要

为缓解大语言模型 (LLM) 推理工作负载的内存带宽瓶颈, 权重矩阵以量化和稀疏格式存储在内存中. 因此, 矩阵 tile 在交给核内通用矩阵乘法 (GeMM) 硬件引擎处理前, 需要完成反量化和解稀疏. 目前, 这些操作由软件通过向量指令执行, 性能提升有限. 此外, 由于 GeMM 的整体性能取决于内存资源, 向量单元和硬件矩阵引擎之间的相互作用, 系统的改进方向也难以判断.

为提高配备核内 GeMM 引擎和 HBM 的先进平台上的 LLM 推理性能, 本文做出三项主要贡献. 第一, 设计一种带三维可视化表示的分析性能模型, 用于说明内存资源, 向量单元和硬件矩阵引擎如何协同提供压缩 GeMM 性能. 第二, 提出新的近核机器学习模型解压缩加速器 DECA. DECA 从 CPU 卸载 tile 的解稀疏和反量化, 生成可直接供核内 GeMM 引擎使用的 tile. 第三, 引入一项新的 ISA 扩展, 允许以乱序方式调用近核加速器. 借助该扩展, 加速器计算和核心计算可以高效交错并重叠执行. 评估结果显示, 在模拟的 56 核 Xeon 4 HBM 服务器上, DECA 相比优化的 Intel 软件 kernel 可将压缩 GeMM 的执行速度最多提高 4 倍. DECA 还可将 Llama2-70B 和 OPT-66B 的下一个 token 生成时间缩短 1.6-2.6 倍.

<span id="section-1"></span>

## 1 引言

大语言模型 (LLM) 是聊天机器人, 翻译, 文本摘要和内容创作中的重要机器学习 (ML) 工作负载. LLM 使用 Transformer, 主要由多头注意力层和全连接 (FC) 层构成. 最大的模型在 FC 层中包含数万亿个参数. 推理时, 这些权重在小批量场景中的复用率较低, 会给内存容量和带宽带来压力.

GPU 凭借计算能力和内存带宽, 成为 LLM 推理的常用平台. 新一代 Intel Xeon 4 服务器增加了名为 TMUL 的核内通用矩阵乘法 (GeMM) 引擎, 并可配备高带宽内存 (HBM). TMUL 通过 AMX 指令编程, GeMM 吞吐量比向量 SIMD 单元高一个数量级; HBM 的带宽则是 DDR 系统的三至四倍.

Xeon 服务器上的 LLM 推理受内存带宽限制. FC 层中的大型 GeMM 占 Llama2-70B 下一个 token 生成时间的 90% 以上 [Lla23]. 因此, 加速这些 GeMM 是提高推理速度的关键.

低比特权重量化和稀疏化等模型压缩技术可以减少内存流量. 但 TMUL 需要密集的 BF16 或 INT8 tile, 无法直接处理任意量化方案或稀疏模式. 因此, Intel libxsmm 使用 AVX 向量指令读取压缩 tile, 执行解稀疏和反量化, 再把密集 tile 送入 AMX 单元. 这种协作模式结合了向量域和矩阵域, 两个域分别使用不同的指令和功能单元.

性能分析表明, libxsmm 在中等压缩率的 GeMM 和 DDR 内存上表现良好, 但使用 HBM 时性能会下降. 传统二维 Roofline 模型忽略了内存, 向量和矩阵资源之间的相互作用, 因而无法解释这一现象. 本文设计了一种名为 Roof-Surface 的三维模型, 用它揭示这种相互作用, 并确定性能下降源于 AVX 解压缩.

本文提出近核加速器 DECA, 从 CPU 核心卸载 tile 的解稀疏和反量化. DECA 支持 1 至 8 bit 的量化格式, 非结构化稀疏和分组量化. 本文还提出 Tile External Preprocess and Load (TEPL), 这项 ISA 扩展可以乱序调用 DECA, 并隐藏 CPU 与加速器之间的通信延迟.

对不同稀疏度的 BF8 和 MXFP4 进行评估后发现, 在模拟的 56 核 Xeon 4 HBM 服务器上, DECA 可将压缩 GeMM 最多加速 4 倍. 相比软件解压缩, DECA 将 Llama2-70B 和 OPT-66B 的下一个 token 生成时间缩短 1.6-2.6 倍; 相比未压缩基线, 则缩短 2.5-5.0 倍.

本文的贡献包括 Roof-Surface 性能模型, DECA 近核解压缩加速器, 支持乱序调用的 TEPL 扩展, 以及针对 LLM 推理中压缩 GeMM 的模拟评估.

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 LLM 推理

LLM 由嵌入层, 全连接层和注意力层构成. 推理先经历提示阶段, 对输入 token 编码并生成第一个 token; 随后进入生成阶段, 逐个生成输出 token. 本文关注算术强度较低的生成阶段, 因为在许多实际场景中, 该阶段占端到端推理时间的主要部分 [Yua24]. 带 HBM 和核内 GeMM 引擎的 CPU 是一种有吸引力的平台, 因此本文研究现代 CPU 服务器上的推理.

<span id="section-2-2"></span>

### 2.2 模型压缩

量化以 FP8 或 FP4 等低比特格式存储权重. 分组量化为每组权重增加一个缩放因子. 本文评估 BF8 和 MXFP4; MXFP4 使用 4 bit 数值, 每 32 个权重共享一个缩放因子. 稀疏化移除接近零的权重. 非结构化稀疏允许移除任意位置的权重, 再通过 bitmask 重建非零位置. 本文评估从 50% 到 5% 的密度.

对于密集 BF16 模型, 密度因子为 $d$ 的 Q-bit 模型可将模型大小缩小 $16/(Qd+1)$ 倍, 其中的 1 表示 bitmask. 本文将该比例称为压缩因子 (CF). 压缩离线执行, 解压缩在线执行, 后者会影响性能.

<span id="figure-01"></span>

![图 1. 离线压缩和在线解压缩.](./deca/figure-01.png)

**图 1.** 离线压缩和在线解压缩.

<span id="figure-02"></span>

```text
// 解压缩 Ti+1
对 Ti+1 的每一行 r:
  执行 AVX 向量操作
// GeMM Ti
使用 AMX 加载 Ti
使用 AMX 计算 Tout
// 解压缩 Ti+2 并计算 GeMM Ti+1
```

**图 2.** Libxsmm 压缩 GeMM kernel 的伪代码.

<span id="section-2-3"></span>

### 2.3 矩阵扩展

Intel Advanced Matrix Extensions (AMX) 增加了 8 个 tile 寄存器. 每个寄存器最多容纳 16 行, 每行 64 byte, 并解释为 BF16 或 INT8 元素. TMUL 将激活 tile $A$ 与权重 tile $W^\top$ 相乘. 当 batch size $N\leq16$ 时, 一次操作可在 16 个周期内完成 $512N$ 次融合乘加. 本文将一次融合乘加记为一次 FLOP.

<span id="section-2-4"></span>

### 2.4 GeMM 解压缩

TMUL 只接受特定的密集格式, 因此压缩权重必须先解压缩为 tile, 再执行乘法. Libxsmm 使用双缓冲, 将 tile $T_{i+1}$ 的 AVX 解压缩与 tile $T_i$ 的 AMX 处理重叠执行. 解压缩序列使用置换和带 mask 的向量扩展. AVX 按 cache line 大小的行处理数据, AMX 则处理整个 tile, 所以 AVX 指令数量远多于 AMX 指令.

<span id="section-3"></span>

## 3 研究动机

<span id="section-3-1"></span>

### 3.1 FC 层 GeMM 主导推理时间

<span id="table-01"></span>

![表 1. FC 层 GeMM 在下一个 token 生成时间中的占比.](./deca/table-01.png)

**表 1.** FC 层 GeMM 在下一个 token 生成时间中的占比.

下一个 token 的生成时间大多用于 FC 层 GeMM: DDR5 系统上的占比超过 95%, HBM 系统上为 85%-90%. 因此, 加速这些 GeMM 可以直接改善端到端性能.

<span id="section-3-2"></span>

### 3.2 FC 层 GeMM 受带宽限制

<span id="figure-03"></span>

![图 3. N=4 时 GeMM 的传统 Roofline.](./deca/figure-03.png)

**图 3.** $N=4$ 时 GeMM 的传统 Roofline.

未压缩的 BF16 点受内存带宽限制. 压缩会提高算术强度, 使数据点向右移动. 当压缩率较高时, 实测性能低于 Roofline, 因为 AVX 解压缩无法跟上内存带宽或 TMUL 吞吐量.

<span id="section-3-3"></span>

### 3.3 压缩 GeMM 可能降低效率

Roofline 分析无法说明消除解压缩瓶颈需要多少向量吞吐量. 扩展 CPU 核心可能造成向量单元, 超标量宽度和 cache 端口的过度配置. 下一节将建立一个模型, 用它指导这类硬件支持的设计.

<span id="section-4"></span>

## 4 Roof-Surface 模型

本文通过三维 Roof-Surface 及其二维投影 Bounding Region Diagram (BORD), 对矩阵, 向量和内存操作之间的相互作用建模.

<span id="section-4-1"></span>

### 4.1 三维 Roof-Surface 性能模型

内存以每秒 $\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}$ 个 tile 的速度提供压缩 tile. 向量硬件以 $\mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}$ 的速度执行解压缩. 矩阵硬件的处理速度为 $\mathrm{MOS}$. 可实现的 tile 速率为

<span id="equation-01"></span>

$$
\mathrm{TPS} = \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

一个 TMUL tile 执行 $512N$ 次 FMA, 因此 Roof-Surface 公式为

<span id="equation-02"></span>

$$
\mathrm{FLOPS} = 512N\cdot\min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

两个由 kernel 决定的变量是 $\mathrm{AI}_{\mathrm{XM}}$ 和 $\mathrm{AI}_{\mathrm{XV}}$. 它们的取值构成 kernel 特征, 因此具有相同特征的两个 kernel 也有相同的投影性能.

<span id="figure-04"></span>

![图 4. 三维 Roof-Surface 模型及其性能预测.](./deca/figure-04.png)

**图 4.** 三维 Roof-Surface 模型及其性能预测.

该曲面包含内存受限区, 向量受限区和矩阵受限区. 曲面下方的实测点可以揭示限制资源, 而曲面上方的点无法实现.

<span id="section-4-2"></span>

### 4.2 二维边界区域图

BORD 将 Roof-Surface 投影到 $\mathrm{AI}_{\mathrm{XM}}$-$\mathrm{AI}_{\mathrm{XV}}$ 平面. 其边界线分别为 $y=(\mathrm{MBW}/\mathrm{VOS})x$, $x=\mathrm{MOS}/\mathrm{MBW}$ 和 $y=\mathrm{MOS}/\mathrm{VOS}$.

<span id="figure-05"></span>

![图 5. HBM 和 DDR 系统的 BORD.](./deca/figure-05.png)

**图 5.** HBM 和 DDR 系统的 BORD.

在 HBM 系统中, 大多数 kernel 位于向量受限区; DDR 的内存带宽较小, 因而内存受限区更大.

<span id="figure-06"></span>

![图 6. 向量吞吐量提高 4 倍后的 HBM BORD.](./deca/figure-06.png)

**图 6.** 向量吞吐量提高 4 倍后的 HBM BORD.

即使向量吞吐量提高 4 倍, 也无法消除所有 kernel 的向量瓶颈. CPU 核心已经把大多数动态指令用于解压缩, 并占用 40%-80% 的提交槽位, 因此扩展传统核心的成本很高.

<span id="section-5"></span>

## 5 DECA 概览和乱序调用

<span id="section-5-1"></span>

### 5.1 DECA 的位置和系统集成

每个 CPU 核心都配有一个 DECA, 其中包含处理单元, 控制寄存器和 tile 输出寄存器. 核心通过特权 store 指令配置量化和稀疏参数. DECA 通过 L2 读取压缩 tile, 执行解压缩, 再把可直接使用的 tile 写入 tile 输出寄存器.

<span id="figure-07"></span>

![图 7. DECA 在核心旁的位置.](./deca/figure-07.png)

**图 7.** DECA 在核心旁的位置.

DECA 与 CPU 共享 L2 TLB 和虚拟地址空间. 上下文切换时可保存其状态; 另一个进程使用 DECA 时, 也可以陷入系统并重新配置.

<span id="section-5-2"></span>

### 5.2 DECA 与核心协同处理 tile

两个 loader 和两个 tile 输出寄存器实现硬件双缓冲. loader 读取数据, bitmask 和缩放因子. DECA 加载, 解压缩并写入 tile $T_i$ 时, 核心读取并计算 $T_{i-1}$, 随后开始读取 $T_{i+1}$.

<span id="figure-08"></span>

![图 8. DECA 与 CPU 核心协同处理 tile.](./deca/figure-08.png)

**图 8.** DECA 与 CPU 核心协同处理 tile.

<span id="figure-09"></span>

```text
TLoad TReg1, TOut1
TComp TReg2, TReg1
把 T(i+1) 的元数据存入 Loader2
fence
```

**图 9.** 通过内存映射调用 DECA 的 CPU 伪代码.

<span id="section-5-3"></span>

### 5.3 乱序调用的 ISA 支持

Tile External Preprocess and Load (TEPL) 将元数据提交和 tile 加载合并在一条指令中. 当 DECA 把 tile 解压缩到核心 tile 寄存器后, 该指令返回. 最多可以同时执行两条 TEPL 指令, 与两个 loader 对应.

<span id="figure-10"></span>

```text
TEPL TReg1, M(i+1)
TComp TReg2, TReg1
TEPL TReg2, M(i+2)
```

**图 10.** 使用 TEPL 指令的 CPU 伪代码.

TEPL 进入专用队列, 源寄存器和执行端口可用后立即执行. 由于 DECA 不会更新内存, 推测调用是安全的; 流水线 flush 时会发出取消信号. 这样可以移除 fence, 重叠多个 tile 的处理过程, 并隐藏核心与 DECA 之间的通信.

<span id="section-6"></span>

## 6 DECA 微架构设计

<span id="section-6-1"></span>

### 6.1 DECA 处理单元

<span id="figure-11"></span>

![图 11. DECA 处理单元微架构.](./deca/figure-11.png)

**图 11.** DECA 处理单元微架构.

每个处理单元包含两个加载队列 (LDQ) 和预取器, bitmask 队列, 缩放因子队列, 向量流水线, 查找表 (LUT) 阵列, 扩展阶段, 缩放阶段和 tile 输出寄存器. 流水线读取压缩数据, 根据 bitmask 扩展非零元素, 通过 LUT 执行反量化和缩放, 最后写出密集 BF16 tile.

<span id="section-6-2"></span>

### 6.2 定量微架构设计

向量流水线的宽度为 $W$, 延迟为 $L$. LUT 阵列支持 8 bit 量化, 并存储每个 8 bit 输入对应的值. bitmask 和缩放因子队列按一个 tile 的容量设置. 稀疏输入需要处理的元素较少, 因此流水线可以自然获得更高吞吐量. Roof-Surface 模型选择 $W=32$ 和 $L=8$ 作为均衡设计.

<span id="section-7"></span>

## 7 处理解压缩瓶颈的 DECA 替代方案

扩展传统向量资源需要更多 AVX 单元, 更宽的向量, 更大的超标量宽度和更多 cache 端口. 核内矩阵扩展可以支持特定的稀疏或低比特格式, 但需要针对格式设计专用硬件, 也无法适应未来的方案. 解耦向量加速器避免了部分核心改动, 却不能利用 TMUL 吞吐量处理稀疏度中等的机器学习模型. DECA 将解压缩单独处理, 通过 LUT 配置支持多种格式, 并以较小的 ISA 改动与矩阵单元协同工作.

<span id="table-02"></span>

![表 2. 与其他核内和近核加速器的比较.](./deca/table-02.png)

**表 2.** 与其他核内和近核加速器的比较.

<span id="section-8"></span>

## 8 方法

本文模拟一台类似 Xeon 4 的 56 核服务器, 主频为 2.5 GHz, 配备 260 GB/s DDR5 或 850 GB/s HBM. 模拟器以 Sniper 为基础, 增加了 DECA 处理单元和 TEPL 队列. 基线 DECA 使用 $W=32$ 和 $L=8$.

软件基线为 Intel libxsmm. 本文用 TEPL 指令替换其中的 AVX 解压缩序列, 并评估包含约 2.5 亿个参数的多级全连接层. 测试的 batch size 为 1 至 16, Llama2 和 OPT 推理使用 Intel Tensor Processing Primitives 框架.

本文评估 Q16, Q8 和 Q4 压缩; 对于 Q16 和 Q8, 密度范围为 50% 至 5%. 面积通过 CACTI 和已发表的电路模型估算. 56 个 DECA 处理单元的面积约为 $2.51\,\mathrm{mm}^2$, 不到 56 核 Xeon 4 die 面积的 0.2%.

<span id="section-9"></span>

## 9 评估

<span id="section-9-1"></span>

### 9.1 DECA 的压缩 GeMM 性能

<span id="figure-12"></span>

![图 12. DDR 系统中 N=1 时压缩 GeMM 的加速比.](./deca/figure-12.png)

**图 12.** DDR 系统中 $N=1$ 时压缩 GeMM 的加速比.

<span id="figure-13"></span>

![图 13. HBM 系统中 N=1 时压缩 GeMM 的加速比.](./deca/figure-13.png)

**图 13.** HBM 系统中 $N=1$ 时压缩 GeMM 的加速比.

DECA 在 DDR 上达到 1.7 倍加速, 在 HBM 上达到 4.0 倍加速. 由于向量开销被隐藏, 其性能接近最优值. batch size 增至 16 时也有类似结果.

<span id="figure-14"></span>

![图 14. DDR 系统中 N=4 时不同压缩方案的 TFLOPS.](./deca/figure-14.png)

**图 14.** DDR 系统中 $N=4$ 时不同压缩方案的 TFLOPS.

16 个配备 DECA 的核心性能超过 56 个传统核心, 因而可以将其他核心用于不同的工作负载或 power gating.

<span id="table-03"></span>

![表 3. Q8, N=1 和 HBM 条件下各组件的利用率.](./deca/table-03.png)

**表 3.** Q8, $N=1$ 和 HBM 条件下各组件的利用率.

DECA 提高了内存利用率, 同时避免向量解压缩单元成为瓶颈. 该结果验证了 Roof-Surface 的预测.

<span id="figure-15"></span>

![图 15. DECA 与传统向量扩展方案的比较.](./deca/figure-15.png)

**图 15.** HBM 和 $N=1$ 条件下 DECA 与传统向量扩展方案的比较.

增加 4 个 AVX 单元或把 AVX 单元加宽 4 倍, 性能仍远低于 DECA, 因为传统系统没有同时扩展超标量宽度或 L1 端口.

<span id="section-9-2"></span>

### 9.2 使用 Roof-Surface 探索设计空间

<span id="figure-16"></span>

![图 16. 未使用 DECA 和使用不同 DECA 尺寸时的 HBM BORD.](./deca/figure-16.png)

**图 16.** 未使用 DECA 和使用不同 DECA 尺寸时的 HBM BORD.

模型选择 $W=32,L=8$ 作为最小设计, 它可以使所有 kernel 离开向量受限区. 过度配置的 $W=64,L=64$ 设计速度提升不足 3%, LUT 数量却增加至 8 倍.

<span id="section-9-3"></span>

### 9.3 DECA 集成和 TEPL 分析

<span id="figure-17"></span>

![图 17. HBM 和 N=4 条件下 DECA 集成功能的效果.](./deca/figure-17.png)

**图 17.** HBM 和 $N=4$ 条件下 DECA 集成功能的效果.

从 L2 读取压缩权重, 使用 DECA 预取器, 写入 tile 输出寄存器以及使用 TEPL, 会依次提高性能. 当密度为 5% 时, TEPL 通过隐藏通信延迟使性能翻倍.

<span id="section-9-4"></span>

### 9.4 DECA 的 LLM 推理性能

<span id="table-04"></span>

![表 4. Llama2-70B 和 OPT-66B 的下一个 token 延迟.](./deca/table-04.png)

**表 4.** Llama2-70B 和 OPT-66B 的下一个 token 延迟.

输入和输出均为 128 个 token 时, DECA 相比软件解压缩将下一个 token 的生成时间缩短 1.6-2.6 倍, 相比未压缩基线则提供 2.5-5.0 倍加速.

<span id="section-10"></span>

## 10 其他相关工作

解耦加速器面向稀疏, 量化和注意力计算, 但通常会增加面积, 功耗和数据移动开销. CPU 集成加速器可以降低这些开销. 协同向量-矩阵处理器包括 Tandem, AWS Trainium, TPU, 以及带 Tensor Core 和 SIMT 核心的 GPU. GPU Tensor Core 也只接受有限的格式, 因此类似 DECA 的解压缩引擎可以扩展 TMA, 并减轻 shared memory 的压力.

<span id="section-11"></span>

## 11 结论

本文面向配备核内 GeMM 引擎和 HBM 的 CPU 平台, 提出 Roof-Surface 性能模型, DECA 近核机器学习模型解压缩加速器, 以及支持乱序调用加速器的 TEPL ISA 扩展. DECA 可加速压缩 GeMM 和 LLM 推理, 所需面积开销不到评估系统的 0.2%.
