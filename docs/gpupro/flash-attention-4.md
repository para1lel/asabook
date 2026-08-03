---
title: "Flash Attention 4"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/flash-attention-4/
pageClass: gpupro-page
---

::: info 概述
- Attention 运行两个 MMA, softmax 楔入它们之间, 因此它不能像 GEMM 那样重复一个 MMA.
- kernel 由第 I 部分中的硬件原语 (TMA, `tcgen05`, TMEM, barrier) 和第 III 部分中的 GEMM 技术以及 warp 角色, 在线 softmax 重新缩放, 因果屏蔽和 GQA 组成.
:::

Attention 是决定 Transformer 性能的关键 kernel, 也是前文所有技术最终汇合的地方. GEMM 中使用的各个组件都会在这里再次出现: TMA tile 搬运, `tcgen05` MMA, TMEM, warpgroup 寄存器 tile, 以及显式 barrier.

挑战在于, attention 并非简单地重复同一个 MMA. 它包含两个 MMA, 中间还穿插在线 softmax, causal masking, 以及把较早和较晚 block 维持在同一尺度上的重缩放操作.

新的难点正来自这个中间阶段. 普通 matmul 只需持续累加 accumulator; attention 则必须随着新的 key 和 value 流入, 重新检查并缩放已经计算的结果. softmax 本身运行在两个 Tensor Core MMA 之间的 CUDA Core 上, 因而指数计算和逐行 reduction 都直接位于关键路径上.

这就是为什么如此多的 attention 优化实际上是 softmax 优化: 重新制定 `exp`, 并将 softmax 与 MMA 重叠, 而不是停滞不前.

本章的目标不是从头推导 Flash Attention. 我们只介绍理解 kernel 所需的算法背景, 随后聚焦真正的新内容: 如何用 TIRx 实现这套算法.

最清晰的方法是跟踪单个 tile, 因为它流经 kernel. `Q`, `K` 和 `V` 作为输入 tile 进入, 从 GMEM 加载到 SMEM 中. score MMA 将 `Q` 和 `K` 相乘, 得到 TMEM 中的分数 tile `S`. Softmax 将 `S` 转换为分子 tile `P`, value MMA 结合 `P` 和 `V` 来更新输出 accumulator `O`.

到目前为止, 这看起来就像两个 matmul 粘在一起, 但有一个 GEMM 从未处理过的问题: 每当运行的 softmax 最大值发生变化时, 到目前为止积累的 `O` 突然处于错误的范围. 必须先对其进行重新缩放, 然后才能安全地添加下一个 value MMA. 下面的部分首先跟踪此路径, 然后才显示 TIRx 如何将每个阶段传递到 warpgroup 并将阶段连接在一起.

## 算法形态

在我们将 tile 放入内存之前, 我们需要 tile 所服务的算法. 对于一个查询块, Flash Attention 计算:

$$O = \text{softmax}(QK^{\top} / \sqrt{d})V$$

从字面上看, 公式说的是形成完整的分数矩阵 `S = QKᵀ`, 对其进行 softmax, 然后乘以 `V`. 这是我们无法使用的一种方法, 因为完整的 `S` 非常巨大. 在 seq=4096 时, 每个头大约容纳 16M 个元素, fp32 中大约有 64 MB, 这比 SMEM 或单个 128×512 TMEM 区域大几个数量级. 芯片上根本没有地方可以放置它. Flash Attention 的答案是根本不实现 `S`. 相反, 它以块的形式传输 `K/V` 并携带三个每行运行状态, 总结了迄今为止所看到的所有内容:

- `row_max`: 迄今为止看到的最高分数.
- `row_sum`: softmax 的运行分母.
- `O`: 运行输出 accumulator.

当新块到达时, 流式更新可以保持这些状态的正确性. 微妙之处在于, 每次我们处理一个块时, 运行最大值可能会上升, 一旦上升, 我们在旧最大值下计算的所有内容现在都处于错误的范围内. 因此, 在添加新贡献之前, 我们首先将旧状态拉回新规模:

```text
S = Q_block @ K_block.T
m_new = max(row_max, rowmax(S))
scale = exp((row_max - m_new) / sqrt(d))
P = exp((S - m_new) / sqrt(d))
row_sum = row_sum * scale + rowsum(P)
O = O * scale + P @ V_block
row_max = m_new
```

单个 `scale` 因子在这里发挥双重作用: 它重新调整运行分母和运行输出, 以便来自较早和较晚块的贡献最终以共同的比例进行测量.

上面的伪代码是用自然的 `exp` 和显式的 `/sqrt(d)` 编写的, 因为这样最容易阅读, 但 kernel 采用更便宜的路线. 它将 `1/sqrt(d)` 和 `log2(e)` 折叠为一个常数 `scale_log2 = log2(e)/sqrt(d)`, 并使用恒等式 `exp(x/sqrt(d)) = exp2(x · scale_log2)` 在原始分数上使用硬件 `exp2` 评估每个指数. 动机很简单, `exp2` 比该硬件上的自然 `exp` 更快.

在我们继续之前, 有一点值得确定: 这里的 `P` *不是* 最终的归一化 attention 矩阵. 它只是当前 K/V 块的 softmax 分子. 规范化是故意推迟的, 只有在最后一个块之后, kernel 才会写入 `O / row_sum`.

对于 TIRx, 了解算法计算的内容只是问题的一半. 另一半是在 kernel 运行时 *每个 tile 所在的位置*, 因为这就是决定 layout 和 barrier 代码的原因. `S`, `P` 和 `O` 都是 tile 值, 并且每一个都有一个 home:

- `S` 是分数 tile. score MMA 将其写入 TMEM.
- `P` 是 softmax 分子 tile. Softmax 将 `S` 从 TMEM 读取到寄存器, 计算 `P = exp((S - m_new) / sqrt(d))`, 并将 `P` 写回 TMEM.
- `O` 是输出 accumulator tile. value MMA 从 TMEM 中读取 `P`, 从 SMEM 中读取 `V`, 然后累加到 TMEM 中的 `O` 中.

我们之前标记的重新缩放也是 tile 操作, 而不是标量簿记: 当 `row_max` 更改时, 旧的 `O` 会从 TMEM 中读取, 在寄存器中相乘, 并在下一个更改之前写回 TMEM value MMA 累加进去. 后面的每个部分都遵循相同的结构: tile 放置, 硬件路径以及证明下一个 consumer 可以运行的 barrier.

## tile -原始图

有了运行状态及其所在位置, 我们就可以将算法布置为 tile 移动的具体序列. 对于一个 K/V 块, kernel 从上到下走这条 tile 路径:

```text
Q, K, V in GMEM
  -> Q, K, V in SMEM        by TMA load
  -> S in TMEM              by score MMA: QK^T
  -> P in TMEM              by softmax numerator: TMEM -> RF -> TMEM
  -> O in TMEM              by value MMA: P V
  -> O in GMEM              by normalization, SMEM staging, and TMA store
```

与 GEMM 的区别仅在于一行. GEMM 是一条重复的 MMA 链; FA4 有两个 MMA phase, softmax 位于链的中间. 接下来的几乎所有其他内容都是额外的阶段的结果.

如果我们将短路径扩展为显式的 producer - consumer 边, 我们将得到完整的图:

|阶段| tile 移动或计算 | TIRx 原语 | 硬件路径 |
|-------|--------------------------|----------------|---------------|
| 负载 Q/K/V | GMEM tile -> SMEM tile | `Tx.copy_async(..., dispatch="tma")` | TMA 负载 |
| score MMA | SMEM 中的 Q 和 SMEM 中的 K -> TMEM 中的分数 tile `S` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` |
| Softmax 读取 | TMEM 中的 `S` -> warpgroup 寄存器 tile | `Tx.wg.copy_async(reg, tmem)` | `tcgen05.ld` |
| Softmax 写入 |寄存器-> fp16 TMEM 视图中的分子 tile `P` | `Tx.copy_async(tmem_as_f16, reg)` | TMEM store, 其次是 `tcgen05.wait.st()` |
| value MMA | TMEM 中的 `P` 和 SMEM 中的 V -> 输出 accumulator TMEM 中的 `O` | `Tx.warp.gemm_async(..., dispatch="tcgen05")` | `tcgen05.mma` 带有 TMEM operand |
| 修正 | TMEM 中的 `O` -> TMEM 中的寄存器-> `O` | TMEM 读回,寄存器乘法, TMEM store | `tcgen05.ld` / TMEM store |
| epilogue | TMEM 中的最终 `O` ->寄存器-> SMEM -> GMEM | TMEM 读回, `Tx.copy`, TMA store | `tcgen05.ld` + TMA store |

新行是 softmax 和 Correction. 两者都会添加 TMEM ->寄存器-> TMEM 流量, 并且都会在 score MMA 和 value MMA 之间创建额外的切换.

尝试与你的代理一起: 要求它仅跟踪上面的短路径. 对于每个箭头, 命名 producer 阶段, consumer 阶段, 源 tile, 目标 tile 和硬件路径. 然后询问哪些箭头在 GEMM 章节中不存在.

## warp 角色和 scope

数据路径确定后, 自然的下一个问题是谁实际运行每个阶段. 这里的每个 CTA 总共有 4 个 warpgroup, 512 个 thread, 它们的划分不是根据它们接触的数据, 而是根据 warpgroup 所做的“什么样的工作”:

- WG3 驱动硬件引擎: TMA 负载, MMA 和 TMA store.
- WG0, WG1 和 WG2 执行这些引擎调用之间发生的寄存器繁重数学运算: softmax, 校正和 epilogue.

确切的角色表是:

| 业主 | 角色 | 它的作用 |
|-------|------|--------------|
| WG3, warp 1 | TMA 负载 | 将 Q, K 和 V tile 从 GMEM 加载到 SMEM |
| WG3, warp 0 | MMA | 同时出现 score MMA 和 value MMA 问题 |
| WG3, warp 2 | TMA store | 将最终的 O tile 从 SMEM 存储到 GMEM |
| WG0 | Q 阶段 0 的 Softmax | 从 TMEM 读取 S, 计算 P, 将 P 写入 TMEM |
| WG1 | Q 阶段 1 的 Softmax | 第二个 Q 流水线阶段的工作相同 |
| WG2 | 修正和 epilogue | 重新缩放 TMEM 中的 O, 标准化,阶段输出 |

人们很容易将“两个 Q 阶段”误读为两个 attention 头, 但事实并非如此. 它们只是 Q 流水线中的两个插槽, WG0 拥有一个, WG1 拥有另一个, 因此两个 Q tile 可以同时运行. 这就是 softmax 工作出现两次的原因, 一次在 WG0 上, 一次在 WG1 上.

代码用符号坐标挑选出这些角色:

```python
wg_id = T.warpgroup_id([4])
warp_id = T.warp_id_in_wg([4])
```

当你阅读 kernel 时, 首先找到角色分支. 它告诉你哪个团队拥有嵌套在其中的每个 tile 原语.

- WG3 warp 1 启动 TMA 加载命令. 选出的一个 lane 发出副本, TMA 引擎移动 tile.
- WG3 warp 0 发出 `tcgen05.mma` 指令.
- WG0 和 WG1 在完整的 warpgroup scope 下运行 softmax.
- WG2 运行校正, epilogue 在完整的 warpgroup scope 下工作.

一种不对称性最终塑造了整个 barrier 图: *每个* MMA, 无论是分数还是价值, 都来自 WG3 warp 0. WG0 和 WG1 根本不会发出 MMA. 他们只消耗分数 tile, 运行 softmax, 并将 `P` 写回 TMEM.

这种分离正是为什么 softmax 需要 barrier 的原因. `s_ready` 将分数 tile 从 MMA warp 传递到 softmax; `p_o_rescale` 带有 `P` 和 `O` 插槽, 该插槽对于 value MMA 来说是安全的, 要么已经重新缩放, 要么因为不需要重新缩放而被释放. 在本章的其余部分, 我们将继续讨论这两个名字.

## 阅读片段

本章中的片段摘自[`flash_attention4. py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py), 因此它们不可避免地引用了我们不复制的 kernel 部分中定义的名称. 自描述的 (`wg_id`, `warp_id`, `BLK_M` / `BLK_N`, `HEAD_DIM`, `kv_stage`, `SMEM_PIPE_DEPTH_*` / `TMEM_PIPE_DEPTH` 深度, `should_accumulate` 和 `CTA_GROUP` (此处为 1)) 我们在下面介绍它们首先重要的位置. 其余的在此处的表格中仅一行注释, 因此当片段将一个不熟悉的名称放在你面前时, 你可以在某个地方查看:

| 名称 | 含义 |
|------|---------|
| `q_stage`, `i_q` | Q 管线阶段, 0 或 1, 即哪个 Q tile 插槽 (`SMEM_PIPE_DEPTH_Q = 2`). 在 WG0/WG1 softmax 内部, warpgroup 自己的 `wg_id` (0 或 1)*是* 相同的阶段索引, 因此 `S_region[q_stage]`, `P_region[wg_id]` 和 `O_region[i_q]` 全部选择相同的 Q 阶段|
| `MMA_N` | TMEM 列中的得分/输出 tile 宽度 (128) |
| `MMA_K` | MMA `P` / `V` 列中的内 K 步 (16); `K_SPLIT = 6 * MMA_K = 96` |
| `K_SPLIT` | 分割点值- MMA 调度 (参见 *两个 MMA phase*); 第一个 value MMA 覆盖列 `0:K_SPLIT` (`6 * MMA_K = 96`) |
| `should_rescale` | WG2 每行标志: 旧的 `O` 是否需要在下一个 value MMA 之前重新缩放 (在 warpgroup 和 `any_sync` 之间减少) |
| `rescale_threshold` | 小行最大变化的跳过阈值; 当前的 kernel 使用 `8.0`, 并且跳过的重新缩放将 `acc_scale` 精确设置为 `1.0` |
| `scale_log2` | 以 log2 为单位的 softmax 比例 `log2(e)/√d`, 因此 `P = exp2((S - m) · scale_log2)` |
| `acc_scale` | 每行重新缩放因子 softmax 通过 SMEM 邮箱传递到 WG2 |
| `chunk_start` / `chunk_end`, `p_start` / `p_end` | 正在读取/写入的 32 宽 softmax 块的列范围 |

## 两个 MMA phase

对于每个流式 K/V tile, Flash Attention 运行两个 MMA phase, 并使用 softmax 桥接它们:

```text
Q, K -> score MMA -> S
S    -> softmax   -> P
P, V -> value MMA -> O
```

可以将其视为连续三个 producer 的流水线. 第一个 MMA 产生 attention 分数 `S`, softmax 将 `S` 转换为分子 `P`, 第二个 MMA 消耗 `P` 更新输出 accumulator `O`. 一旦每个 K/V tile 都有发言权, `row_sum` 的标准化就会被保留到 epilogue.

下面的每个 tile 操作都获得与我们用于 GEMM 步骤的相同的 scope / layout / dispatch 卡, 还有一个额外的行 Handoff, 用于命名传递的 barrier tile 到下一个角色.

计算代码从不使用原始 TMEM 列号. 相反, kernel 将其单个 TMEM 分配划分为每个阶段视图 (`S_region`, `P_region`, `O_region`), 并按流水线对它们进行索引阶段(`S_region[q_stage]`, `O_region[i_q]`, `P_region[i_q, 0:K_SPLIT]`). 这些视图在 [TMEM layout 和 Reuse](#tmem-layout-和重用) 部分中使用 `T.TMEMStages` 进行定义; 目前, 将每个区域视为同一物理 TMEM 的命名切片就足够了.

### score MMA

两个 phase 中的第一个是 score MMA, 它是打开每个 K/V 迭代的 matmul. 它计算:

$$S = Q_{\text{block}}K_{\text{block}}^{\top}$$

并将 `128 x 128` 分数 tile 写入 TMEM:

```python
Tx.warp.gemm_async(
  S_region[q_stage],
  Q_smem[q_stage, 0:BLK_M, 0:HEAD_DIM],
  K_smem[kv_stage, 0:BLK_N, 0:HEAD_DIM],
  dispatch="tcgen05",
  cta_group=CTA_GROUP,
)
if T.ptx.elect_sync():
  s_ready.arrive(q_stage)
```

我们可以问 GEMM 章节对每个 tile 操作提出的四个相同问题: 谁运行它, tile 位于何处, 它如何调度, 以及它如何移交:

> tile -原始读数: score MMA
> - scope: WG3 warp 0 发布; 选出的一个 lane 到达 `s_ready`.
> - layout: SMEM 中的 Q, K → TMEM 中的 `S` (`S_region[q_stage]`).
> - dispatch: `tcgen05`.
> - 切换: `s_ready` (→ softmax).

到达 `s_ready` 的单个选定的 thread 是整个切换. 它宣布这个分数 tile 已经完成, 并且 softmax warpgroup 现在可以免费阅读.

### Softmax 之间 MMA

在两个 MMA 之间坐落着 softmax, 即阶段, 它将分数 tile `S` 转换为分子 tile `P`. 其读出卡为:

> tile -原始读数: Softmax
> - scope: WG0 (Q 阶段 0) / WG1 (Q 阶段 1), 完整 warpgroup.
> - layout: TMEM 中的 `S` →寄存器→ fp16 TMEM 中的 `P` (`P_region[wg_id]`).
> - dispatch: `tcgen05.ld` 读取, TMEM store 写入; 它们之间的寄存器中的逐行数学.
> - 切换: 等待 `s_ready`; 到达 `p_o_rescale` (前 96 列) 和 `p_ready_2` (后 32 列).

这个阶段是根本没有 GEMM 对应物的. WG0/WG1 等待分数 tile 到达 `s_ready`, 然后从 TMEM 中一次读取寄存器大小的块:

```python
Tx.copy_async(
  s_chunk[:, chunk_start : chunk_end],
  S_region[wg_id, chunk_start : chunk_end],
)
```

这是在 warpgroup scope 下读取的 TMEM 到寄存器 tile. 现在分数位于寄存器, softmax warpgroup 按顺序执行三件事:

1. 计算行最大值和行总和,
2. 计算 softmax 分子 tile `P`,
3. 将 `P` 作为 fp16 写回 TMEM.

最后一步看起来像:

```python
Tx.copy_async(
  P_region[wg_id, p_start : p_end],
  p_chunk[:, p_start : p_end],
)
```

当我们刚刚在寄存器中完成计算时, 为什么要将 `P` 写回 TMEM? 因为 value MMA 需要 `P` 作为 *tile operand*, 并且 MMA 无法读取分散的 per-thread 标量寄存器作为矩阵. 此 kernel 中的 MMA 可读形式是 `P_region`, 这是 fp16 TMEM 别名 `tmem_as_f16` 的视图​​. 所以 writeback 不是多余的运动; 正是它使 `P` 成为下一个 MMA 可以实际使用的唯一形状.

### value MMA

第二个 phase 是 value MMA, 也是关闭每次 K/V 迭代的 value MMA. 它计算:

$$O = O + P_{\text{block}}V_{\text{block}}$$

当这个 MMA 运行时, `O` 已经进入当前 K/V 块的正确状态, 在第一个块上初始化, 在后面的块上重新缩放, 因此 MMA 要做的就是累积. 它与 GEMM 的不同之处在于 operand 所在的位置: A operand 在 TMEM 中是 `P`, B operand 在 SMEM 中是 `V`, 而 accumulator `O` 也在 TMEM 中:

```python
# First sub-MMA: columns 0:K_SPLIT (the first 96 of P / rows of V).
Tx.warp.gemm_async(
  O_region[i_q],
  P_region[i_q, 0:K_SPLIT],
  V_smem[kv_stage, 0:K_SPLIT, 0:HEAD_DIM],
  transB=True,
  accum=should_accumulate,
  dispatch="tcgen05",
  cta_group=CTA_GROUP,
)
# The second sub-MMA (same form, accum=True, gated on p_ready_2) covers the
# remaining columns K_SPLIT:BLK_N.
```

> tile -原始读数: value MMA
> - scope: WG3 warp 0.
> - layout: TMEM 中的 `P` + SMEM 中的 V → TMEM 中的 `O` (`O_region[i_q]`).
> - dispatch: `tcgen05` 与 TMEM operand.
> - 切换: 等待 `p_o_rescale`, `p_ready_2`, `kv_load.full`; 到达 `o_ready` (→ epilogue).

这个 operand 的摆放位置就是两个 MMA 的硬件区别:

- score MMA 从 SMEM 读取 operand: Q 和 K.
- value MMA 从 TMEM 读取一个 operand, `P`.
- value MMA 从 SMEM 读取另一个 operand, V.
- 结果累加到 TMEM 中的 `O` 中.

`accum=should_accumulate` 标志是实现算法中“初始化或添加”选择的标志: 它在查询块的第一个 K/V tile 上为 false, 而在之后的每个 tile 上为 true.

你可能还注意到, value MMA 不是一次性运行的, 而是分为 `96 + 32` 计划:

1. Softmax 将 `P` 写入四个 32 列的块中.
2. 一旦前三个块准备就绪, value MMA 就会从 `P` 的前 96 列和 `V` 的匹配行开始.
3. 最后 32 列等待 `p_ready_2`.
4. 第二个 MMA 消耗最后一个块并完成 tile.

拆分的原因是为了让 Tensor Core 保持忙碌. 将 value MMA 作为单个指令运行, 整个 phase 将停止, 直到所有四个 32 列 `P` 块都已求幂并存储. 通过立即触发前三个块, kernel 与最后一个块的 `exp` 和 TMEM 写入与已在运行中的 96 宽 MMA 重叠, 将原本的空闲时间转化为有用的工作.

## TMEM layout 和重用

所有 `S`, `P` 和 `O` 都必须共享一个 `128 x 512` TMEM 分配, 而它们的打包方式正是 barrier 和 layout 原来是离不开这个 kernel:

下图显示了直接打包: 分数槽, 分子槽和输出槽都共享一个 TMEM 分配, 因此 barrier 协议使重用合法.

![TMEM 布局](./images/tmem_layout_v3.png)

该图显示为一组 tile 插槽:

- 分数槽包含 `S = QK^T`.
- Softmax 求幂步骤之后, 分子槽保存 `P` tile.
- 输出插槽容纳 fp32 `O` accumulator.

这些不是独立的缓冲区. 它们是“相同”分配的区域, 共享不是一种风格选择, 而是一种强制选择. 当 Q 流水线深度为 2 时, 两个 `S` 插槽 (2 × MMA_N = 256 列) 和两个 `O` 插槽 (2 × MMA_N = 256 列) 已占所有 512 个 fp32 列. `P` 没有剩余任何内容, 因此 `P` 别无选择, 只能通过更窄的 fp16 视图来别名相同的字节. 这是安全的唯一原因是每个区域在其先前的 consumer 完成后严格重用, 并且该时间正是 barrier 所保证的. 所以在 FA4 中, barrier 不仅仅是调度; 它们首先使 layout 合法.

别名技巧是通过 `T.TMEMPool` 设置的. kernel 采用一个 fp32 视图 (`tmem`) 作为分数并输出 accumulator, 然后将池基数倒回到 0 并在 *相同的物理字节* 上采用第二个 fp16 视图 (`tmem_as_f16`):

```python
tmem_pool = T.TMEMPool(
  pool,
  total_cols=N_COLS_TMEM,
  cta_group=CTA_GROUP,
  tmem_addr=tmem_addr,
)
tmem = tmem_pool.alloc((128, N_COLS_TMEM), "float32")
tmem_pool.move_base_to(0)
tmem_as_f16 = tmem_pool.alloc((128, N_COLS_TMEM * 2), "float16")
tmem_pool.commit()
```

由于 fp16 元素的宽度只有一半, 因此 fp16 视图在相同字节上公开的可索引列数量是其两倍, 而这正是 `P` 所在的空间, 而 fp32 layout 没有空间容纳的空间. 有了这两个视图, kernel 就可以将 `S`, `P` 和 `O` 插槽作为 `T.TMEMStages` 的暂存区域进行划分, 这使得计算代码可以按流水线进行索引阶段而不是通过原始列:

```python
S_region = T.TMEMStages(
  tmem,
  col_start=0,
  width=MMA_N,
  stages=SMEM_PIPE_DEPTH_Q,
  stride=MMA_N,
)
O_region = T.TMEMStages(
  tmem,
  col_start=MMA_N * SMEM_PIPE_DEPTH_Q,
  width=MMA_N,
  stages=SMEM_PIPE_DEPTH_Q,
  stride=MMA_N,
)
P_region = T.TMEMStages(
  tmem_as_f16,
  col_start=MMA_N,
  width=BLK_N,
  stages=SMEM_PIPE_DEPTH_Q,
  stride=MMA_N * 2,
)
```

`P_region` 的步幅中的 `* 2` 是别名明显泄漏到代码中的地方. `S_region` 和 `O_region` 在 fp32 `tmem` 列中测量, 而 `P_region` 在 fp16 `tmem_as_f16` 列中测量, 宽度为 fp16 `tmem_as_f16` 列, 因此阶段到阶段的移动需要双倍的步幅才能落在相同的物理字节上. 不过, 一旦定义了区域, 计算代码就保持干净: 它写入 `S_region[q_stage]`, 读取 `S_region[wg_id,...]`, 写入 `P_region[wg_id,...]`, 并累积到 `O_region[i_q]`, 从未触及原始列索引.

与你的代理一起尝试: 要求它解释此 FA4 kernel 中的 fp32 (`tmem`) 和 fp16 (`tmem_as_f16`) 视图. 哪些物理 TMEM 区域包含 `S`, `P` 和 `O`, 为什么 `P_region` 的步幅使用 `MMA_N * 2`? 将重用问题保存到下一节: 在 barrier 表之后, 检查在每个区域可以重用之前必须完成哪个 consumer.

## barrier 如何连接角色

这是 kernel 中最难的部分, 所以循序渐进是值得的. 从沿着主计算路径移动数据的少数 barrier 开始, 并将其他所有内容视为簿记, 你可以稍后查找. 数据就绪交接是:

| 切换 | 含义 |
|---------|---------|
| TMA 负载->分数/ value MMA | Q, K 或 V 已到达 SMEM, 可以提供 MMA |
| score MMA -> softmax | `S` 已在 TMEM 中准备就绪 |
| softmax/校正 -> value MMA | `P` 在 TMEM 中已准备就绪, `O` 可以安全积累 |
| value MMA -> epilogue | 最终的 `O` 已在 TMEM 中准备就绪 |
| epilogue -> TMA store | `O_smem` 已准备好存储 |

不在该列表中的所有内容都是流水线簿记: barrier, 它释放 SMEM, TMEM 或暂存缓冲区, 以便另一个角色可以重用它. 有用的是, 每个 barrier, 无论是携带数据还是仅记账, 读取方式都相同, 就像 tile 切换一样. 你询问谁生成了数据, 谁使用了数据, 以及一旦数据都完成了, 哪个缓冲区就会空闲.

下图将这些切换折叠到两个 MMA phase 的确切就绪门中: score MMA 等待什么, 以及 value MMA 在累积之前必须等待什么.

![Flash Attention 4 MMA 输入门](./images/flash_attention_main_handoff.png)

将此图视为一组正确性门而不是时间表. 它回答了“在 MMA 可能触发之前必须满足的条件”, 并且没有提及时间. score MMA 在 SMEM 中等待 Q 和 K, 然后生成 `S`. value MMA 同时等待三件事: SMEM 中的 V, softmax 中的 `P` tile 以及 WG2 已发布或重新调整的 `O` 插槽. softmax-to-value 门被分割的原因是我们已经遇到的: 一旦 `P` 的前 96 列到位, value MMA 就可以开始, 并且 `p_ready_2` 释放最后 32 列.

有一种切换不适合 tile 就绪模型: softmax 到校正边缘. softmax 不是传递 tile, 而是通过单槽 SMEM 邮箱将单个标量 (K/V 循环期间的 `acc_scale`, 或 epilogue 中的最终 `row_sum`) 传递到 WG2. 由于该插槽在每次迭代中都会重复使用, 因此 `full` / `empty` barrier 对必须保护它:

下图放大了邮箱握手, 这就是为什么这一对 barrier 应被读取为标量 producer - consumer 通道, 而不是 tile 就绪门.

![Flash Attention 4 Softmax Scale-Slot 握手](./images/flash_attention_softmax_correction.png)

将 `softmax_corr.full` 和 `softmax_corr.empty` 读取为 producer - consumer 对:

1. Softmax 在重用缩放/总和槽之前等待 `softmax_corr.empty`.
2. Softmax 将 `acc_scale` 或最终的 `row_sum` 写入该插槽.
3. Softmax 抵达 `softmax_corr.full`.
4. WG2 等待 `softmax_corr.full`, 然后读取插槽.
5. WG2 抵达 `softmax_corr.empty`.
6. softmax warpgroup 可以在下一个 phase 中重用该槽.

值得注意的是 `softmax_corr.empty` 的含义和含义. 它仅表示 WG2 已消耗了缩放/求和槽. 它没有说明 `P` 是否准备好, 并且它强调“不是”让 value MMA 启动的门. 该门是 `p_o_rescale`, 当写入 `P` 的前 96 列并且 `O` 插槽可以安全累积时, 该门将触发. 混淆两者是错误结果错误的典型来源.

有了主路径, 完整的 barrier 列表可以作为参考:

| barrier | producer -> consumer | 什么变得安全 |
|---------|----------------------|-------------------|
| `q_load.full` | TMA 负载 -> score MMA | Q SMEM tile 可喂 MMA |
| `q_load.empty` | 此 Q 阶段-> TMA 负载的所有 score MMA | Q SMEM 阶段可重复用于下一个任务 |
| `kv_load.full` | TMA 负载->分数/ value MMA | K 或 V SMEM tile 可以喂 MMA |
| `kv_load.empty` | 分数/ value MMA -> TMA 负载 | K/V SMEM 阶段可重复使用 |
| `s_ready` | score MMA -> softmax | S TMEM tile 可读 |
| `p_o_rescale` | Softmax + WG2 -> value MMA | P 的前 96 列在 TMEM 中, O 槽对于 value MMA 是安全的 |
| `p_ready_2` | Softmax -> value MMA | P 的最后四分之一在 TMEM 中 |
| `o_ready` | value MMA -> epilogue | 最终 O accumulator 已准备就绪 |
| `softmax_corr.full` | Softmax -> WG2 | `acc_scale` 或最终的 `row_sum` 已在 SMEM 邮箱中准备就绪 |
| `softmax_corr.empty` | WG2 -> Softmax | WG2 读取后可以重复使用相同的 SMEM 邮箱槽 |
| `corr_epi.full` | epilogue -> TMA store | O_smem 已准备好存储 |
| `corr_epi.empty` | TMA store -> epilogue | O_smem 阶段可重复使用 |

就像在 GEMM 中一样, 你可以根据信号的产生者来预测 barrier 的类型:

- TMA 加载使用 `TMABar`, 因为 TMA 引擎会对其自己的完成情况进行字节计数.
- MMA 完成使用 `TCGen05Bar`, 因为 `tcgen05.commit` 发出完成组信号.
- 纯 thread 到 thread 切换使用 `MBarrier`, 其中参与的 thread 显式到达.

分离的 softmax 到值的切换值得仔细观察. 它使用两个门:

- 一旦 `P` 的前 96 列被写入, 并且 `O` tile 可以安全地累积, `p_o_rescale` 就会启动 value MMA.
- `p_ready_2` 发布 `P` 的最后 32 列, 与上一节中的 `96 + 32` 值 - MMA 计划相匹配.

第一个 K/V 块是简单的情况. WG2 提前到达 `p_o_rescale`, 因为还没有旧的 `O` tile 进行重新缩放.

后面的区块必须更加小心. 仅当 WG2 跳过不必要的重新缩放或完成旧 `O` 的重新缩放后, 才会到达 `p_o_rescale`. 跳过测试故意保守: softmax 计算 log2 缩放的增量 `(m_old - m_new) * scale_log2`; 如果该值仍然高于 `-rescale_threshold`, 则新的最大值尚未移动足够远以证明重新缩放是合理的, 因此 kernel 保留旧的最大值并将 `acc_scale` 设置为 1.0. 只有较大的最大跳跃才会采用 `exp2` 路径并要求 WG2 重新缩放 `O`.

然后, WG2 将 `should_rescale` 跨 warpgroup 减少为 `any_sync`. 如果没有行需要更新, 则仅保留 `O`. 该跳过很重要, 因为重新缩放 `O` 是整个 accumulator 上的完整 TMEM -> RF -> TMEM 读取-修改-写入, 当阈值逻辑已经保留时, 纯粹是浪费工作 `acc_scale` 为 1.0.

请注意, 所有新的 barrier cluster 都位于一处. `s_ready`, `p_o_rescale`, `p_ready_2`, 以及 softmax/校正对都是围绕 softmax 的 barrier. 它们存在的原因只有一个: score MMA 和 value MMA 不再相邻.寄存器数学, TMEM 重写和输出重新缩放现在位于它们之间, 并且每个步骤都需要自己进行切换.

尝试使用你的代理: 要求它通过 `s_ready`, `p_o_rescale`, `p_ready_2` 和 `o_ready` 跟踪一个 K/V 块. 对于每个 barrier, 询问谁在等待, 谁到达, 什么 tile 可以安全读取, 以及之后可以重用哪些存储.

## 流水线结构

barrier 告诉我们在角色消耗 tile 之前必须“准备好”什么. 他们没有告诉我们的是实际上“并发”运行的是什么, 这就是我们现在要解决的问题. 两者确实不同: 正确性门可以在 producer 运行之前或之后很久满足.

这里没有单一的流水线深度, 因为不同的 tile 流以不同的速率移动. 因此, kernel 为以下各项保留一个单独的环:

- Q 流水线深度 2: 一个 CTA 在两个 Q 阶段上工作. WG0 处理一个阶段, WG1 处理另一个.
- KV 流水线深度 3: K 和 V 块流经内部循环, 同时重复使用相同的 Q 阶段.
- TMEM 流水线深度 2: 每个 Q 阶段都有自己的 S/P/O TMEM 插槽, 这些插槽在匹配的 barrier 触发后会重复使用.

下图从正确性门切换到时间线视图, 显示一旦这些单独的环开始飞行, 哪些角色可以在大致相同的时间处于活动状态.

![Flash Attention 4 管线结构](./images/flash_attention_pipeline_v2.png)

将其视为时间线而不是 barrier 图表. 它显示哪些角色在大致相同的时刻处于活动状态, 而早期的 barrier 流图是你检查确切的 producer - consumer 等待的位置. 这两个人物回答了我们在本节开头提出的两个不同问题.

每行都与代码的角色分支之一匹配:

- WG3 warp 1 发出 TMA 负载.
- WG3 warp 0 同时发出 score MMA 和 value MMA.
- WG0 和 WG1 对两个 Q 阶段运行 softmax.
- WG2 释放或重新调整 `O`, 然后标准化最终输出.
- WG3 warp 2 发出 TMA store.

下图从左到右描绘了一个具有代表性的流水线波. 负载 warp 从 `Q0`, `K[n-1]`, `Q1`, `V[n-1]` 开始, 然后继续流式传输较低索引的 K/V 块. MMA warp 发出第一个 score MMA 来生成 `S0` 和 `S1`, WG0/WG1 将它们转换为 `P0` 和 `P1`.

重要的是, MMA warp “不”运行所有 score MMA, 然后运行所有 value MMA. 一旦两个 Q 阶段都已启动, 它就会交织两种类型: 当前 `V` 块的 value MMA, 然后下一个 `K` 块的 score MMA, 依此类推:

```text
score Q0*K[n-1]
score Q1*K[n-1]
value P0*V[n-1]
score Q0*K[n-2]
value P1*V[n-1]
score Q1*K[n-2]
value P0*V[n-2]
...
```

这种交错是分数, softmax, 校正和值行在图中全部重叠而不是整齐连续运行的原因.

WG2 行标记为 `release / rescale`, 这两半对应于我们看到的两种情况. 在第一个 K/V 块上, 还没有旧的 `O`, 因此 WG2 仅参与让 value MMA 继续进行的切换; 在后面的块上, 它可能会在 value MMA 累积到其中之前重新调整旧的 `O`. 归一化和 TMA store 在 attention 任务的最终 K/V 块之后只发生一次.

没有单一的 GEMM 式流水线可以描述 FA4, 因为 Q, K/V 和 TMEM 时隙都按独立的时间表推进. TIRx 使这些计划保持明确, 作为单独的 tile 缓冲区, `PipelineState` 游标和 barrier phase, 而不是将 kernel 隐藏在一个整体基元后面. 成本是更多的移动部件, 但好处是复杂性保持可见和可检查.

## 重新缩放和 writeback

重新调整是强制性的, 我们不能放弃优化. 在线 softmax 可以通过每个新分数 tile 来提高每行最大值, 并且无论何时, 从早期块累积的 `O` 都会按“旧”最大值进行缩放. 这使得前面的每个项都太大了 `exp(m_new - m_old)` 倍. 跳过校正, 这些块的权重过大, 最终的输出就是错误的. 修复方法是 TMEM →寄存器→ TMEM tile 操作:

$$O_{\text{old}} \leftarrow O_{\text{old}} \cdot e^{(m_{\text{old}} - m_{\text{new}}) / \sqrt{d}}$$

这项工作分为两个角色. Softmax 计算每行规模并将其放入 SMEM 邮箱中; WG2 等待 `softmax_corr.full`, 从 TMEM 中读取当前的 `O`, 乘以该比例, 然后写回 `O`:

```python
RESCALE_TILE = T.meta_var(16)
o_row = T.wg_reg_tile(RESCALE_TILE)
Tx.copy_async(o_row, O_region[i_q, d_start : d_start + RESCALE_TILE])
Tx.mul(o_row, o_row, acc_scale)
Tx.copy_async(O_region[i_q, d_start : d_start + RESCALE_TILE], o_row)
T.ptx.tcgen05.wait.st()
```

值得强调的是, 这是对整个 `O` accumulator 的完整 TMEM →寄存器→ TMEM tile 操作, 而不是一点标量簿记, 并且它带有与其他所有卡相同的读出卡阶段:

> tile -原始读数: 校正 (重新缩放)
> - scope: WG2, 完整 warpgroup.
> - layout: TMEM 中的 `O` →寄存器→ TMEM 中的 `O` (`O_region[i_q]`).
> - dispatch: `tcgen05.ld` 读取, TMEM store 写入;寄存器在它们之间相乘.
> - 切换: 等待 `softmax_corr.full`; 到达 `p_o_rescale` (→ value MMA) 和 `softmax_corr.empty` (→ softmax).

从头到尾跟踪同步:

1. Softmax 将比例值写入 SMEM.
2. WG2 等待 `softmax_corr.full`.
3. WG2 在 TMEM 中重新缩放 `O`.
4. WG2 抵达 `p_o_rescale`.
5. WG3 的 value MMA 现在可以消耗 `P` 并累积到重新缩放的 `O` tile 中.

当 `softmax_corr.empty` 在 WG2 读取 SMEM 槽后释放 SMEM 槽时, 循环将关闭, 这将释放 softmax 以在下一次迭代中重用邮箱.

一旦 K/V 循环结束, WG2 从校正切换到 epilogue. 它等待最终的 `row_sum` 和 `o_ready`, 从 TMEM 读取最终的 `O`, 乘以 `1 / row_sum` (我们一开始推迟的标准化), 转换为 fp16, 然后写入 `O_smem`. WG3 的 TMA store warp 然后将 `O_smem` 返回到 GMEM.

对于任何计划扩展此 kernel 的人来说, 有一个限制值得标记. 它仅计算前向输出, 而训练前向传递通常还会存储后向传递所需的对数和表达式 (LSE). 添加时需要记住一个缩放细节: 此 kernel 将 `row_max` 保留为 *原始*, 未缩放的 `QK^T` 分数的最大值, 而 `row_sum` 累积 `exp((S - row_max) / sqrt(d))`. 因此, 在形成自然对数 LSE 时, 必须将 `1/\sqrt{d}` 因子重新应用于 `row_max`:

$$\mathrm{LSE}_i = \log(\mathrm{row\_sum}_i) + \mathrm{row\_max}_i / \sqrt{d}$$

该实现仅是前向输出并且不写入 LSE.

## 因果掩蔽

因果 attention 增加了一个约束 (查询可能只关注其自身位置或之前的键), 并且 kernel 以两种互补的方式尊重它, 一种是廉价的, 一种是精确的.

最便宜的方法是完全跳过工作. 许多 K/V 块完全位于对角线上方, 对给定的 Q 块没有任何贡献, 因此 `get_n_block_max(...)` 计算该块可能需要的最后一个块, 并且循环根本不会加载或计算其余部分.

精确的方法处理跨越对角线的块, 其中某些列有效, 有些列无效. 这些块仍然运行 score MMA, 但 softmax 在求幂之前屏蔽掉无效列. 对于每一行, 它从行的查询位置和块偏移中派生出列限制, 将列保持在该限制或低于该限制, 并将超过该限制的每一列设置为寄存器中的 `-inf`, 因此这些列对行最大值或 `exp2` 分子没有任何贡献.

该实现不是逐个元素地分支, 而是使用 `mask_r2p(...)` 来应用限制, 这会将其转换为整个 32 宽分数块上的位掩码, 并一次性掩蔽该块. 完全位于对角线下方的块保留每一列, 并且根本不需要掩模.

从 tile -primitive 视图来看, 因果模式根本不重写数据路径. 它仅修剪 K/V 行程计数, 并在 score MMA 和 `P` writeback 之间的寄存器驻留 softmax 中插入屏蔽步骤.

## GQA 支持

分组查询 Attention 允许多个查询头共享单个 K/V 头. 这节省了内存带宽, 但提出了一个打包问题: 我们如何保留一个 K/V tile, 同时仍然通过它提供许多查询头? kernel 的答案是立即针对一个计划的 `kv_head_idx` 处理一整组查询头:

```python
GQA_RATIO = num_qo_heads // num_kv_heads
SEQ_Q_PER_TILE = BLK_M // GQA_RATIO
```

诀窍是重新解释 128 Q- tile 行. 对于 `GQA_RATIO=4`, 它们不再代表 128 个序列位置; 它们代表 32 个序列位置乘以 4 个查询头, 打包在一起, 以便所有四个头都使用相同的 K/V tile. 行解码为:

```text
seq_pos = row // GQA_RATIO
q_head  = row % GQA_RATIO
```

Q 载荷用 3D 视图表达这种堆积. 源是自然的 `Q[batch, seq, qo_head, dim]` layout, 而目标是完全相同的 SMEM tile, score MMA 稍后将读取为平面 `128 x HEAD_DIM` operand. 视图使两者协调一致, 并且无需任何复制即可实现:

```python
Q_smem_3d = Q_smem.view(
  SMEM_PIPE_DEPTH_Q,
  SEQ_Q_PER_TILE,
  GQA_RATIO,
  HEAD_DIM,
)
Tx.copy_async(
  Q_smem_3d[i_q, :, :, :],
  Q[batch_idx,
    m_start : m_start + SEQ_Q_PER_TILE,
    kv_head_idx * GQA_RATIO : (kv_head_idx + 1) * GQA_RATIO,
    :],
  **tma_copy_q,
)
```

K 和 V 永远不会在内存中扩展, 这就是 GQA 的全部要点: `kv_head_idx` 的单个 K/V tile 被打包到 Q 行中的所有 `GQA_RATIO` 查询头重用. 输出侧镜像输入, 并在 epilogue 之后使用匹配的 3D 视图将打包行存储回 `O[batch, seq, qo_head, dim]`.

结果是 GQA 完全处于 Q 负载和 O 存储边界. 在计算路径内, score MMA 仍然看到一个普通的 `128 x HEAD_DIM` Q tile, 而 tile 基元图的其余部分未受影响.

## tile 调度

调度器的工作是将每个 CTA 映射到 `(batch, kv_head, m_block)` attention 任务, 正确的策略取决于屏蔽是否使这些任务的成本相等:

- 非因果模式使用 `FlashAttentionLinearScheduler`. 每个任务执行相同的工作负载, 因此通过 `num_ctas` 推进的固定 CTA 池足以均匀地分布它们.
- 因果模式使用 `FlashAttentionLPTScheduler`, 因为因果屏蔽使工作变得非常不均匀: 靠近开头的 Q 块负责大约一个 K/V 块, 而靠近结尾的一个 Q 块负责所有这些块. 简单的分割会让一些 CTA 晚于其他 CTA 完成很长时间, 因此最长处理时间的调度器会预先加载重型块以平衡完成时间, 同时仍然将附近的批处理/头任务保持在一起以实现 L2 局部性.

尽管存在所有差异, 但这两个调度器公开了相同的循环接口:

```python
while scheduler.valid():
  m_block_idx = scheduler.m_block_idx
  batch_idx = scheduler.batch_idx
  kv_head_idx = scheduler.head_idx
  # process one Q block against its K/V block range
  scheduler.next_tile()
```

唯一的行为差异在于 `next_tile()` 的作用: 在非因果模式下, 它将 CTA 推进到另一任务, 而在因果模式下, 它会在当前任务之后结束循环. 无论哪种方式, 这纯粹是一个调度决策: 它选择 CTA 拥有的“哪个”attention tile, 而不是如何计算 tile. 在循环内部, 相同的本地原语运行, 无论: TMA 加载, score MMA, softmax, value MMA, 校正, TMA store.

## 编译并验证

上面的所有内容都是摘录, 因此为了将它们放在一起并实际运行 kernel, 我们从 `tirx-kernels` 导入真实的东西, 编译它, 并根据火炬参考检查它. 完整的 kernel (本章介绍的每一部分都组装成一个文件) 是 `tirx-kernels` 存储库中的 [`flash_attention4. py`](https://github.com/mlc-ai/tirx-kernels/blob/main/tirx_kernels/attention/flash_attention4.py). 与 GEMM 验证单元有两点不同: Flash Attention 具有更丰富的入口点 (`get_flash_attention4_kernel`), 并且其内置分析器需要额外的 `profiler_buf` 参数. 这是整章运行的一个单元格:

```python
import torch
import torch.nn.functional as F
import tvm
from tirx_kernels.attention.flash_attention4 import (
  get_flash_attention4_kernel, PROFILER_BUFFER_SIZE)

# GQA: 32 query heads share eight KV heads.
B, S, Hq, Hkv, D = 1, 1024, 32, 8, 128
Q = torch.randn(B, S, Hq, D, dtype=torch.float16, device="cuda")
K = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
V = torch.randn(B, S, Hkv, D, dtype=torch.float16, device="cuda")
O = torch.empty(B, S, Hq, D, dtype=torch.float16, device="cuda")
prof = torch.zeros(PROFILER_BUFFER_SIZE, dtype=torch.uint64, device="cuda")

kernel = get_flash_attention4_kernel(B, S, S, Hq, Hkv, D, is_causal=False)
target = tvm.target.Target("cuda")
with target:
  ex = tvm.compile(
    tvm.IRModule({"main": kernel}),
    target=target,
    tir_pipeline="tirx",
  )
# ex.mod takes torch tensors directly, as in every other chapter.
ex.mod(Q, K, V, O, prof)
torch.cuda.synchronize()

# torch reference; enable_gqa lets the 32 query heads share the 8 KV heads
qt, kt, vt = (x.transpose(1, 2).float() for x in (Q, K, V))
ref = F.scaled_dot_product_attention(
  qt,
  kt,
  vt,
  enable_gqa=True,
).transpose(1, 2).half()
torch.testing.assert_close(O, ref, rtol=1e-2, atol=1e-2)
print(f"FA4: B={B} S={S} Hq={Hq} Hkv={Hkv} D={D}, non-causal -> PASS")
```

预期输出: `... -> PASS`. kernel 以 fp32 累积在线 softmax, 但几个不同的近似值仍然将其结果与高精度参考分开. 有输入的 fp16 存储和舍入以及 operand; 基于 `exp2` 的 softmax 重构 (每个指数的 `scale_log2 = log2(e)/√d` 重构); 在线 softmax 重新排序和每行重新缩放, 按运行比例对块求和, 而不是一次性全部求和; 最后是 `O` 在 writeback 上的 fp16 演员表. 这里选择的 `rtol` / `atol` 与源 kernel 自己的测试使用的容差相同, 其大小可以根据火炬参考覆盖所有这些, 而不是单独进行 fp16 舍入. 因此, 如果你在这里看到真正的失败, 而不仅仅是接近失误, 请将其视为指向 softmax 路径的路标: 掉落的 `s_ready` / `p_o_rescale` / `p_ready_2` 等待, 或 `row_max` / `row_sum` 更新称重新调整步骤未能应用. 这些正是本章花费 barrier 进行的交接.

## 与 GEMM 的差异

下表比较了 FA4 与 GEMM 沿变化的轴:

| 方面 | GEMM | Flash Attention 4 |
|--------|------|-------------------|
| MMA phase | 一重复 MMA | score MMA 和 value MMA |
| MMA 之间工作 | 除了流水线交接之外没有其他任何操作 | 在线 softmax, 掩蔽和 O 重新缩放 |
| 运行状态 | 仅限 accumulator | 行最大值, 行总和, O accumulator |
| 主要中间体 | accumulator TMEM tile | S, P 和 O TMEM tile 区域 |
| warp 角色 | TMA producer, MMA consumer, writeback | TMA 负载, MMA, softmax, 校正, TMA store |
| barrier | 主要是加载/计算/ writeback 切换 | 额外的分数/softmax/值/校正切换 |
| 调度单元 | 输出矩阵 tile | 注意任务: `(batch, kv_head, m_block)` |

这些差异中的每一个都可以追溯到我们在本章开头的结构变化: 第二个 MMA, softmax 介于两者之间. 另一方面, 基础 TIRx 约定根本没有改变:

- tile 原语表示 tile 移动或计算什么,
- 周围的 scope 表示哪个 thread 配合,
- layout 表示 tile 所在的位置,
- barrier 表示下一个角色何时可以使用它.

所以 FA4 比 GEMM 更难, 不是因为它依赖于不同的硬件, 而是因为有更多的 tile 值以及它们之间更多的切换.

## 练习

1. 与 GEMM 相比, FA4 中的两个 MMA phase 之间出现了哪些新的 tile 切换? 将其命名为 producer, TMEM tile 和 consumer.
2. 为什么 softmax 将分子 tile `P` 写回 TMEM, 而不是仅将其保留在 value MMA 的寄存器中?
3. 选择 `p_o_rescale` 或 `p_ready_2`. barrier 到底证明了什么? 如果 value MMA 跳过这个等待, 可能会出现什么问题?

尝试使用你的代理: 选择一个未注释的 tile 基元, 例如 epilogue `Tx.copy_async`, fp32 -> fp16 `Tx.cast` 或第二个 `gemm_pv` 子 MMA. 询问其 scope / layout / dispatch / 切换卡, 然后根据源防护, 分配和等待检查答案.
