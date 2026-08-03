---
title: "使用 TMA 流水线化 GEMM"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/pipelined-gemm/
pageClass: gpupro-page
---

::: info 概述
- 基础 GEMM 让复制和计算轮流执行, 即使两者本可并行, 也会浪费大量时间.
- 第 4 步切换到 TMA 异步加载, 第 5 步双缓冲 SMEM 和预取 (PIPE_DEPTH=2); 完全加载/计算重叠在第 7 步中通过 warp specialization 到达, 第 6 步使 kernel 与 tile 调度器保持一致.
- 目标是在 Tensor Core 计算当前 tile 时, 同步加载下一个 tile.
:::

Tensor Core 是芯片上代价最高的计算单元, 但上一章中正确的 tiled GEMM 让它在大部分周期里处于空闲状态. kernel 按顺序执行: thread 先把 tile 复制到 shared memory, Tensor Core 再进行计算; 随后 thread 复制下一个 tile, Tensor Core 只能等待. 然而, 下一个 tile 的加载与当前 tile 的计算使用不同的硬件, 原本可以并行进行. 缩小这一差距不需要改变数据路径, tile, layout 和数学定义都已经正确; 真正需要改变的是工作在“何时”发生, 以及“由谁”调度. 本章保持 tile 数据路径不变, 直接消除这些空闲周期.

我们分三个渐进步骤完成这一目标. 第 4 步把批量 GMEM <-> SMEM 传输交给 TMA, 由专用复制硬件代替 thread 搬运 tile. 第 5 步加入双阶段软件流水线, 让下一个 K tile 在当前 tile 计算时提前就位. 第 6 步将 kernel 改为由 tile 调度器驱动的持久 kernel, 分摊每个 tile 的设置成本, 并通过调整 tile 顺序提高 operand 的缓存复用. 在整个过程中, SMEM, TMEM 和寄存器 layout 都保持不变. 唯一真正的新概念是硬件单元之间的异步交接: 各个引擎可以独立推进, 不必再严格同步执行.

## 第 4 步: TMA 异步加载

我们的第一步是让副本本身脱离关键路径. 想想 CTA 在第 1 步至第 3 步中所做的事情: 它的每一个 thread 都会计算地址并发出加载指令, 除了将 tile 传送到 SMEM 之外, 没有任何原因. 这是用于流水线而不是数学上的指令带宽. 第 4 步将同步 `Tx.copy` 替换为 TMA, 其中单个 thread 发出一个命令, TMA 引擎自行执行整个 tile 传输. 从这里开始, 示例以完整的 M=N=K=4096 大小运行, 而不是第 1 步至第 3 步的小大小, 并且它们的端到端时序出现在 [使用 Warp Specialization 和 Cluster 扩展 GEMM](/gpupro/warp-specialized-gemm/) 末尾的“端到端结果”表中.

> 此步骤更改的内容: dispatch
> - scope: 不变, 一个 warpgroup.
> - layout: 不变, 相同 SMEM/TMEM/寄存器 tile.
> - dispatch: GMEM → SMEM 负载从同步 `Tx.copy` 移至 TMA 引擎.

### TMA 发起模式

第 4 步的一个更改是用 TMA 负载替换同步 tile 副本, 因此有必要仔细查看该负载的发出方式. 对源代码的编辑只有几行, 但这些行背后的执行模型是不同的. 同步 `Tx.copy` 是 CTA thread 使用自己的指令自行完成的工作; TMA 复制是 thread 发出的命令, 之后 TMA 硬件执行所有移动. 两者并排观看是值得的.

之前 (第 3 步): 所有 128 个 thread 都参与复制, 然后 `cta_sync` 使共享内存写入可见:
```python
# All 128 threads participate.
Tx.cta.copy(
  Asmem[:, :],
  A[m_st : m_st + BLK_M, i * BLK_K : (i + 1) * BLK_K],
)
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

之后 (第 4 步): 一个 thread 发出 TMA 负载, 并且 mbarrier 跟踪硬件传输何时完成:
```python
tid = warp_id * 32 + lane_id                 # 0..127 within the warpgroup
if tid == 0:  # exactly one thread starts TMA
  Tx.copy_async(Asmem, A[...], dispatch="tma")
  Tx.copy_async(Bsmem, B[...], dispatch="tma")
  # Number of bytes expected from TMA.
  T.ptx.mbarrier.arrive.expect_tx(tma_bar, byte_count)
# Wait before MMA reads SMEM.
T.ptx.mbarrier.try_wait(tma_bar, phase)
```

请注意, 负载是在 `tid == 0` 上门控的, 而不是在 `elect_sync()` 上门控的, 而且这种区别比看起来更重要. `elect.sync` 选择一个活动的 lane *每个 warp*, 而 warpgroup 有四个 warp, 因此 `elect_sync()` 实际上会让四个 thread 进入加载协议. 问题在于协议向 mbarrier 宣告预期的字节数, 并且必须恰好宣告一次; 四个公告会破坏计数, 并且等待永远不会正确释放. 通过 warpgroup 范围的 id 精确选择一个 thread 是避免这种情况的干净方法.

诚实地了解加速的来源很重要. 第 4 步仍然在每次 TMA 加载后等待, 因此我们尚未将负载与计算重叠; 这就是第 5 步的工作. 这里的胜利纯粹来自于数据移动路径的改变:

- `Tx.copy` 使用 CTA thread 来计算地址并发出加载/存储指令.
- TMA 使用一个发出的命令来启动硬件 tile 传输. 地址生成, 合并和混合由 TMA 描述符描述并由 TMA 引擎执行.

因此, 尽管第 4 步在每次加载时仍然会阻塞, 但无论如何它最终都会更快. TMA 吸收了批量传输, 从而使 CTA thread 免于花费指令带宽来洗牌 tile, 仅此一项节省就足以取得进展.

### TMA 加载和存储同步

我们已经了解了 TMA 副本是如何发行的; 故事的另一半是知道它什么时候结束. 切换到 TMA 会同时改变两件事: 谁开始复制, 以及代码如何知道复制何时完成. 第一个从代码中显而易见; 第二个很容易被忽视, 如果出错的话会给你一个无声的正确性错误而不是崩溃. 使用 `Tx.cta.copy`, CTA thread 一起执行复制, 并且以下 `cta_sync()` 足以知道它已完成. 通过 TMA, 选定的一个 thread 发出 `Tx.copy_async(..., dispatch="tma")`, 引擎按自己的时间表执行传输, 并通过 mbarrier 发出完成信号.

这正是 `cta_sync()` 不再足够的原因. `cta_sync()` 仅等待 CTA 自己的 thread 并仅命令其共享内存写入; 它对飞行中的 TMA 传输一无所知, 因此当 tile 仍在到达时它会愉快地返回. 修复方法是使完成变得明确: 对于 TMA 加载, 选定的 thread 首先告诉 mbarrier 需要多少字节, 然后 CTA 在任何 MMA 接触 SMEM tile 之前等待 *该* mbarrier. 下图描绘了端到端的握手过程.

![TMA 异步加载: 同步流程](./images/tma_sync_flow.svg)

上图隔离了负载端握手: 选定的 thread 启动 TMA, mbarrier 计算预期字节, MMA 在读取 SMEM 之前等待释放. 其中“Elected thread”表示启动 TMA 的选定 thread, 在我们的代码中是 `tid == 0` thread, 而不是 `elect_sync()` lane.

将加载路径放在一起, 然后: 选定的 thread 发出两个 `copy_async` 调用, 并在它们之后执行 `arrive.expect_tx(total_bytes)`, 其中字节数正是 mbarrier 应保留的数据量. 一旦引擎移动了那么多字节, 匹配的 `mbarrier.try_wait(phase)` 就会释放, 只有这样 SMEM tile 才能安全地馈送到 MMA.

存储端通过相同的硬件传输, 但以不同的方式等待, 因此在你的头脑中将这两个协议清楚地分开是值得的: 加载使用 mbriers 和字节计数跟踪完成情况, 而存储则使用提交组和等待组跟踪它. thread 将其 fp16 结果写入 `Dsmem` 并同步后, 选定的一个 thread 启动 `Tx.copy_async(D[...], Dsmem, dispatch="tma")`, 然后是 `cp_async.bulk.commit_group()`, 然后是 `cp_async.bulk.wait_group(0)` 阻塞直到商店耗尽. 该等待不是可选的: 在前一个存储消失之前, `Dsmem` 不能重新用于下一个 tile.

尝试使用你的代理: 跟踪 1 K tile 的第 4 步加载和存储同步. 识别哪个 thread 启动每个 TMA 命令, 哪个 mbarrier 或提交组跟踪完成, 哪个等待保护 `Asmem` 和 `Bsmem` 的 MMA 读取, 哪个等待保护 `Dsmem` 的重用. 为什么此处的 TMA 加载协议选择 `elect_sync()` 是错误的 thread?

### 完成 kernel

完整的 kernel 将 TMA 加载和存储折叠到第 3 步结构中, 而该结构的其余部分保持不变. 导入与以前相同:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
```

它被包装在 `hgemm_v4(M, N, K)` 中, 这是我们始终遵循的模式: 包装器将形状相关常量和 layout 保留在使用它们的 kernel 旁边.

```python
def hgemm_v4(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K
  F16_SIZE = 2

  A_layout = tma_shared_layout(
    a_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, BLK_K),
  )
  B_layout = tma_shared_layout(
    b_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_N, BLK_K),
  )
  D_layout = tma_shared_layout(
    d_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, BLK_N),
  )

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- SMEM allocation (now includes Dsmem for TMA store) ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tma_bar = pool.alloc((1,), "uint64", align=8)
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
    pool.commit()

    # --- Barrier + TMEM init ---
    if warp_id == 0 and lane_id == 0:
      T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
      T.ptx.mbarrier.init(tma_bar.ptr_to([0]), 1)
    if warp_id == 0:
      T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
    )

    m_st = T.meta_var(bx * BLK_M)
    n_st = T.meta_var(by * BLK_N)
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0

    # --- Inline helpers ---
    @T.inline
    def tma_load(k_st):
      tma_config = T.meta_var({
          "dispatch": "tma", "cta_group": 1,
          "mbar": tma_bar.ptr_to([0])
      })
      Tx.copy_async(Asmem[:, :],
        A[m_st : m_st + BLK_M, k_st : k_st + BLK_K],
        **tma_config)
      Tx.copy_async(Bsmem[:, :],
        B[n_st : n_st + BLK_N, k_st : k_st + BLK_K],
        **tma_config)
      T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([0]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE
      )

    @T.inline
    def mma(accum):
      Tx.gemm_async(
        tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
        accum=accum, dispatch="tcgen05", cta_group=1
      )
      T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

    # --- K-loop with TMA async ---
    tid = T.meta_var(warp_id * 32 + lane_id)
    for k in range(K_TILES):
      k_st = T.meta_var(k * BLK_K)

      # Single thread issues TMA load
      if tid == 0:
        tma_load(k_st)

      # Wait for TMA to finish; the mbarrier release carries SMEM
      # visibility to the subsequent MMA, so no extra fence is needed.
      T.ptx.mbarrier.try_wait(tma_bar.ptr_to([0]), phase_tma)

      # Single thread issues MMA
      if tid == 0:
        mma(accum=k != 0)

      # Wait for MMA to finish
      T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
      phase_tma ^= 1
      phase_mma ^= 1

    # --- TMA Store Writeback ---
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

    # Read TMEM -> registers asynchronously. wait.ld and cta_sync
    # ensure the read completes.
    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()
    T.cuda.cta_sync()
    # Cast fp32 -> fp16
    Tx.cast(Dreg_f16[:], Dreg[:])
    # Write registers -> Dsmem, flush, then sync
    Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
    T.ptx.fence.proxy_async("shared::cta")
    T.cuda.warpgroup_sync(10)
    # TMA store: Dsmem -> GMEM. One selected thread starts the
    # store group before Dsmem is reused.
    if tid == 0:
      Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
        Dsmem[:, :], dispatch="tma")
      T.ptx.cp_async.bulk.commit_group()
      T.ptx.cp_async.bulk.wait_group(0)
    T.cuda.warpgroup_sync(10)

    # --- Deallocate TMEM ---
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

### kernel 中的 TMA 配置

kernel 中的几乎所有内容都是从第 3 步继承的. 只有五个配置点实际上携带 TMA 语义, 并且值得了解每个配置点的名称:

- TMA 配置: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` 告诉 `Tx.copy_async` 使用 TMA 并通过 `tma_bar` 报告加载完成情况.

- 字节数: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 是两个 fp16 operand tile 加载的字节数. `arrive.expect_tx(...)` 将此计数提供给 mbarrier.

- mbarrier 初始化: `init(tma_bar.ptr_to([0]), 1)` 创建 TMA 负载使用的完成 barrier.

- `@T.inline`: `tma_load(...)` 和 `mma(...)` 是辅助函数. 它们在编译时扩展为 kernel 主体, 并且可以使用周围 kernel 中的变量.

- TMA store 同步: epilogue 首先将 fp16 行写入 `Dsmem`. `fence.proxy_async` 和 `warpgroup_sync` 使这些 thread 写入的 SMEM 值准备好用于 TMA store 路径. 然后, 存储使用 `commit_group()` 和 `wait_group(0)` 等待 SMEM 到 GMEM 的传输完成.

此时我们的曲目是正确的, 但节奏是错误的. 第 4 步仍然在开始匹配的 MMA 之前完成每个加载, 因此加载和乘法实际上不会同时运行; 我们费了好大劲才分开的两个引擎仍然轮流运转. 下一步使 TMA 加载和存储路径保持原样, 而是重新安排计划, 以便加载一个 K tile 可以在计算在另一个上运行时继续进行.

## 第 5 步: 软件流水线(PIPE_DEPTH = 2)

当两个引擎明显独立时, 为什么第 4 步不能将负载与计算重叠? 事实证明, 障碍是存储. 只有一对 SMEM tile 时, 下一次加载无处可去: 直到当前 MMA 完成读取该对后才能开始, 因为提前开始会覆盖仍在使用的数据. 第 5 步通过双缓冲 shared memory 消除该存储冲突. 单 warpgroup 循环在启动下一个 TMA 加载之前仍然等待每个 MMA, 但它现在具有不同的阶段来预取和重用. 我们仍然处于完整的 M=N=K=4096 尺寸.

> 此步骤更改的内容: layout
> - scope: 不变, 一个 warpgroup.
> - layout: 单个 SMEM tile 对成为 `PIPE_DEPTH` -阶段环形缓冲区.
> - dispatch: 不变, TMA 负载和 `tcgen05` MMA; 此步骤添加了预取和阶段重用, 而完全加载/计算重叠在第 7 步中到达.

### 流水线演练

对于 `PIPE_DEPTH=2`, kernel 分配两个 SMEM 阶段, 为加载路径和 MMA 路径提供单独的插槽以进行工作.

将下图视为两个阶段缓冲区要启用的流水线结构, 而不是这个单个 warpgroup kernel 的精确执行跟踪. 第 5 步构建环形缓冲区并稍后预取阶段, 但主循环在发出下一个 TMA 加载之前仍然等待当前的 MMA. 当 warp specialization 为 TMA 和 MMA 提供单独的角色时, 完全加载/计算重叠在第 7 步中到达.

![*Pipeline PIPE_DEPTH=2, 目标进度; 此单个 warpgroup 步骤仅预取, 完全重叠与第 7 步*](./images/pipe_depth2.png) 中的 warp specialization 一起到达

一旦启动, 循环就会交替通过两个阶段. 两个 TMA 负载预先填充两个阶段; 之后, 循环等待当前的阶段, 在其上运行 MMA, 等待 MMA 完成读取阶段, 然后将 `k + PIPE_DEPTH` 的加载启动到阶段刚刚变得可重用. 这还不是并发 TMA/ MMA 调度, 但它建立了第 7 步将在 producer 和 consumer 角色之间分割的环形缓冲区结构.

具体来说, 该代码与第 4 步有四个地方不同:

1. `Asmem` 和 `Bsmem` 获得领先的 `PIPE_DEPTH` 维度, 因此每个阶段都有自己的 SMEM 存储.
2. `tma_bar` 成为一个阵列, 每个阶段具有一个 mbarrier.
3. 在主 K 循环之前, kernel 预取前两个阶段.
4. K 循环使用 `stage = k % PIPE_DEPTH`: 等待当前的阶段, 对其运行 MMA, 然后将阶段重用于 `k + PIPE_DEPTH`.

### 流水线力学

1. 预取: 在主循环运行之前, 我们加载第一个 `PIPE_DEPTH`阶段, 以便循环始终在第一次迭代时找到等待它的数据:
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
  tma_load(s, s * BLK_K)
```

2. 主循环: 对于每个 K tile, 我们等待其阶段准备好, 在其上运行 MMA, 然后通过启动 tile 的负载立即使现在空闲的阶段恢复工作 `PIPE_DEPTH` 前方:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

3. phase 管理: 这是让人绊倒的部分, 但规则比乍一看要简单. 每个 barrier 的 phase 翻转规则直接遵循 barrier 有多少个插槽, 这就是两个 barrier 以不同节奏翻转的原因. MMA accumulator 位于一个 TMEM 插槽中, 因此 `mma_bar` 是每次迭代都会重新访问的单个 barrier (`mma_bar.ptr_to([0])`), 并且 barrier 你重新访问每次迭代时必须将其 phase 翻转每次迭代. TMA barrier 讲述了一个不同的故事: 它们形成一个 `PIPE_DEPTH` 元素数组, 每个阶段一个 barrier, 以及任何给定的阶段 barrier 每次穿过环仅返回一次. 因此, 仅当阶段索引回滚到 0 时, `phase_tma` 才会翻转:
```python
if stage == PIPE_DEPTH - 1:
  phase_tma ^= 1
```

尝试使用你的代理: 使用 `PIPE_DEPTH=2` 和 `K_TILES=5`, 要求它跟踪主循环. 对于每个 `k`, 列出 `stage`, 传递给等待的 `phase_tma` 和 `phase_mma` 值, 以及是否发出新的预取. `phase_tma` 到底在哪里翻转, 为什么最后两次迭代没有预取?

### 完成 kernel

完整的 kernel 逐字保留第 4 步 TMA 加载和存储路径, 然后将其包装在我们刚刚描述的分段缓冲区和 phase 逻辑中. 进口量不变:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
```

它被包裹在 `hgemm_v5(M, N, K)` 中. `PIPE_DEPTH=2` 常量设置流水线阶段的数量 (这里有两个, 这正是双缓冲):

```python
PIPE_DEPTH = 2

def hgemm_v5(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")
  F16_SIZE = 2
  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K

  # Double-buffered layouts: first dimension is pipeline stage
  A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K))
  B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_N, BLK_K))
  D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, BLK_N))

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- SMEM allocation ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    # Double-buffered TMA barriers (one per stage), single MMA barrier
    tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
    pool.commit()

    # Initialize barriers: PIPE_DEPTH for TMA, 1 for MMA
    if warp_id == 0:
      if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
        for s in range(PIPE_DEPTH):
          T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
    if warp_id == 0:
      T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
    )

    m_st = T.meta_var(bx * BLK_M)
    n_st = T.meta_var(by * BLK_N)
    phase_tma: T.int32 = 0
    phase_mma: T.int32 = 0

    @T.inline
    def tma_load(stage, k_offset):
      tma_config = T.meta_var({
          "dispatch": "tma", "cta_group": 1,
          "mbar": tma_bar.ptr_to([stage])
      })
      Tx.copy_async(Asmem[stage, :, :],
        A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
        **tma_config)
      Tx.copy_async(Bsmem[stage, :, :],
        B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
        **tma_config)
      T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([stage]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

    @T.inline
    def mma(stage, accum):
      Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
        accum=accum, dispatch="tcgen05", cta_group=1)
      T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

    tid = T.meta_var(warp_id * 32 + lane_id)

    # === Prefetch: load first PIPE_DEPTH stages ===
    if tid == 0:
      for s in range(min(PIPE_DEPTH, K_TILES)):
        tma_load(s, s * BLK_K)

    # === Main loop ===
    for k in range(K_TILES):
      stage = k % PIPE_DEPTH

      # Wait for TMA to finish loading this stage
      T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

      # MMA on this stage's data
      if tid == 0:
        mma(stage, accum=(k != 0))

      T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
      phase_mma ^= 1

      # Issue next prefetch load (k + PIPE_DEPTH)
      next_k = k + PIPE_DEPTH
      if next_k < K_TILES:
        if tid == 0:
          tma_load(stage, next_k * BLK_K)

      # TMA phase flips when stage wraps around
      if stage == PIPE_DEPTH - 1:
        phase_tma ^= 1

    # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()
    T.cuda.cta_sync()
    Tx.cast(Dreg_f16[:], Dreg[:])
    Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
    T.ptx.fence.proxy_async("shared::cta")
    T.cuda.warpgroup_sync(10)
    if tid == 0:
      Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
        Dsmem[:, :], dispatch="tma")
      T.ptx.cp_async.bulk.commit_group()
      T.ptx.cp_async.bulk.wait_group(0)
    T.cuda.warpgroup_sync(10)

    # Deallocate TMEM
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

## 第 6 步: 持久化 kernel + tile 调度器

到目前为止, 一切都优化了单个 tile 内的工作. 第 6 步更改问题的规模并在 tile 上进行优化.

第 5 步为每个 128 x 128 输出 tile 启动一个 CTA. 对于 4096 x 4096 输出, 这意味着 1024 个独立的 CTA, 每个 CTA 都支付自己的设置成本, 然后在 tile 完成后消失.

第 6 步启动固定的 CTA 池, 然后要求每个 CTA 依次处理许多 tile. 这给我们带来了两件事: 设置工作分摊到多个 tile 上, 并且 tile 分配在 kernel 内部移动, 调度器可以在其中选择重用 operand 的顺序. 我们保持完整的 M=N=K=4096 大小.

> 此步骤更改的内容: scope
> - scope: 固定的持久 CTA 池, 每个 CTA 通过调度器循环多个输出 tile.
> - layout: 未更改, 与每个 tile SMEM/TMEM/寄存器路径相同.
> - dispatch: 不变.

### 持续调度

持久性 kernel 的定义思想是, 它根据硬件而不是问题调整 grid 的大小. 它会推出 `SM_COUNT` CTA, 大约每个 SM 一个, 无论有多少个输出 tile, 目的是保持每个 SM 持续占用. 我们故意说“大致”: 不能保证精确的 1:1 驻留, 因为它取决于占用情况以及硬件如何选择安排 CTA.

在 B200 上, 我们的目标是 `SM_COUNT=148`. 这 148 个 CTA 中的每一个都在 `ClusterPersistentScheduler2D` 传递给它的 tile 上循环.

第一个回报是摊销. TMEM 分配, barrier 初始化和调度器状态现在每个 CTA 发生一次, 并在 CTA 处理的大约 7 个 tile 中重复使用, 而不是在一次性 CTA 中重复 1024 次.

第二个回报来自调度器选择的顺序. 设置 `l2_group_size=8` 将附近的 tile 分组在一起, 因此共享行带的 tile 重用相同的 A 行- tile, 而共享列带的 tile 重用相同的 B tile. 连续运行这些 tile 可以使 operand 在 L2 中保持热状态, 而不是从 HBM 重新获取它们. 这正是第 3 步中留下的重用.

```python
bx = T.cta_id([SM_COUNT])  # 1D grid, one CTA per SM

tile_scheduler = ClusterPersistentScheduler2D(
  "ts",
  num_m_tiles=M // BLK_M,
  num_n_tiles=N // BLK_N,
  l2_group_size=8,       # Group 8 nearby tiles together
  num_clusters=SM_COUNT
)
tile_scheduler.init(bx)
```

循环 tile 会带来一个很容易被忽略的正确性结果. 每个 tile 都运行自己的新 K 循环, 这意味着它的 barrier phase 必须从已知状态开始. 在第 5 步中, CTA 恰好处理了一个 tile, 因此一次初始化 `phase_tma` 和 `phase_mma` 就完全没问题了. 在第 6 步中, 这些初始化器必须移动到 `while tile_scheduler.valid()` 循环的“内部”, 以便每个 tile 都以与其自己的 TMA 和 MMA 匹配的 phase 状态开始, 而不是继承之前的任何内容 tile 恰好留下了:

```python
while tile_scheduler.valid():
  phase_tma: T.int32 = 0
  phase_mma: T.int32 = 0
  ...
```

### 完成 kernel

从结构上来说, kernel 只不过是包裹在 tile-level 外循环中的第 5 步流水线. 唯一的新依赖项是调度器本身, 我们将其与其他依赖项一起导入:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D
```

grid 维度现在只是 `SM_COUNT` 而不是 `(M//BLK_M, N//BLK_N)`, 并且 `ClusterPersistentScheduler2D` 接管将其 tile 交给每个 CTA 的工作:

```python
SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
PIPE_DEPTH = 2

def hgemm_v6(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")
  F16_SIZE = 2
  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K

  A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K))
  B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_N, BLK_K))
  D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, BLK_N))

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    # 1D grid: one CTA per SM (not a 2D grid anymore!)
    bx = T.cta_id([SM_COUNT])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- SMEM allocation (same as Step 5) ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tma_bar = pool.alloc((PIPE_DEPTH,), "uint64", align=8)
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)
    pool.commit()

    # --- Barrier + TMEM init (same as Step 5) ---
    if warp_id == 0 and lane_id == 0:
      T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
      for s in range(PIPE_DEPTH):
        T.ptx.mbarrier.init(tma_bar.ptr_to([s]), 1)
    if warp_id == 0:
      T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)])
    )

    # Tile scheduler: assigns tiles to CTAs in L2-friendly order
    tile_scheduler = ClusterPersistentScheduler2D(
      "ts",
      num_m_tiles=M // BLK_M,
      num_n_tiles=N // BLK_N,
      l2_group_size=8,
      num_clusters=SM_COUNT
    )
    tile_scheduler.init(bx)

    tid = T.meta_var(warp_id * 32 + lane_id)

    @T.inline
    def tma_load(stage, k_offset, m_st, n_st):
      tma_config = T.meta_var({
          "dispatch": "tma", "cta_group": 1,
          "mbar": tma_bar.ptr_to([stage])
      })
      Tx.copy_async(Asmem[stage, :, :],
        A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
        **tma_config)
      Tx.copy_async(Bsmem[stage, :, :],
        B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
        **tma_config)
      T.ptx.mbarrier.arrive.expect_tx(
        tma_bar.ptr_to([stage]),
        (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)

    @T.inline
    def mma(stage, accum):
      Tx.gemm_async(tmem[:, :BLK_N], Asmem[stage, :, :], Bsmem[stage, :, :],
        accum=accum, dispatch="tcgen05", cta_group=1)
      T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

    # === Outer loop: iterate over tiles ===
    while tile_scheduler.valid():
      # Get current tile position from scheduler
      m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
      n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

      # === Inner loop: same pipeline as Step 5 ===
      phase_tma: T.int32 = 0
      phase_mma: T.int32 = 0

      # Prefetch first PIPE_DEPTH stages
      if tid == 0:
        for s in range(min(PIPE_DEPTH, K_TILES)):
          tma_load(s, s * BLK_K, m_st, n_st)

      # Main K-loop
      for k in range(K_TILES):
        stage = k % PIPE_DEPTH
        T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)
        if tid == 0:
          mma(stage, accum=(k != 0))
        T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
        phase_mma ^= 1
        next_k = k + PIPE_DEPTH
        if next_k < K_TILES:
          if tid == 0:
            tma_load(stage, next_k * BLK_K, m_st, n_st)
        if stage == PIPE_DEPTH - 1:
          phase_tma ^= 1

      # === TMA Store Writeback: TMEM -> RF -> Dsmem -> TMA -> GMEM ===
      Dreg = T.alloc_local((BLK_N,), acc_type)
      Dreg_f16 = T.alloc_local((BLK_N,), d_type)
      Dreg_wg = Dreg.view(
        128,
        BLK_N,
        layout=TileLayout(
          S[(128, BLK_N) : (1@tid_in_wg, 1)]
        ),
      )
      Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
      T.ptx.tcgen05.wait.ld()
      T.cuda.cta_sync()
      Tx.cast(Dreg_f16[:], Dreg[:])
      Tx.copy(Dsmem[warp_id * 32 + lane_id, 0:BLK_N], Dreg_f16[:])
      T.ptx.fence.proxy_async("shared::cta")
      T.cuda.warpgroup_sync(10)
      if tid == 0:
        Tx.copy_async(D[m_st : m_st + BLK_M, n_st : n_st + BLK_N],
          Dsmem[:, :], dispatch="tma")
        T.ptx.cp_async.bulk.commit_group()
        T.ptx.cp_async.bulk.wait_group(0)
      T.cuda.warpgroup_sync(10)

      T.cuda.cta_sync()
      tile_scheduler.next_tile()  # Move to next tile

    # Deallocate TMEM
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

## 练习

1. 在第 4 步中, `arrive.expect_tx` 使用 `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` 字节. 如果这个字节数太小或太大, mbarrier 会等待什么?
2. 在第 5 步中, 为什么每个 SMEM 阶段需要自己的 TMA barrier, 而不是为两个阶段共享一个 `tma_bar`?
3. 在第 6 步中, 带有 `BLK_M=BLK_N=128` 的 4096 x 4096 输出有多少个输出 tile? 使用 `SM_COUNT=148`, 每个持久 CTA 平均处理多少个 tile?
