---
title: "使用 Warp Specialization 和 Cluster 扩展 GEMM"
createTime: 2026/08/01 00:00:00
permalink: /gpupro/warp-specialized-gemm/
pageClass: gpupro-page
---

::: info 概述
- 流水线 GEMM 仍然有一个 warpgroup 执行负载, 依次是 MMA 和 writeback, 本章消除了瓶颈.
- 第 7 步将 warp 专门化为角色, 第 8 步添加 2-CTA cluster, 第 9 步添加多个 consumer.
- 每一步都消除了串行瓶颈, 最终达到接近最先进的吞吐量.
:::

上一章的流水线 GEMM ([使用 TMA 流水线化 GEMM](/gpupro/pipelined-gemm/)) 已经很快, 但仍要求同一个 warpgroup 完成所有工作: 发起加载, 执行 MMA, 再写回结果. 即使加入软件流水线, 同一组 thread 仍是三个硬件引擎共同的串行交汇点.

症状很直观: Tensor Core 工作时, TMA 单元闲置; 结果写回内存时, Tensor Core 又会闲置. 各个引擎都通过同一组 thread 相互等待. 解决办法是停止让一个 warpgroup 包办所有工作.

我们通过扩大合作的三个步骤来实现这一想法. 第 7 步 ([使用 Warp Specialization 和 Cluster 扩展 GEMM](#第-7-步-warp-specialization-流水线)) 将 warp 专门化为 producer, consumer, 以及 writeback 角色. 第 8 步 ([使用 Warp Specialization 和 Cluster 扩展 GEMM](#第-8-步-2-cta-cluster)) 将两个 CTA 加入到 cluster 中, 该 cluster 在其 shared memory 之间共享 operand. 第 9 步 ([使用 Warp Specialization 和 Cluster 扩展 GEMM](#第-9-步-多-consumer-warp-specialization)) 添加了第二个 MMA consumer, 因此一个上演 tile 提供两倍的数学运算.

这有助于将这三个步骤视为不同尺度的一种模式. 第 7 步将完整流水线保留在一个 CTA 内: TMA 和 MMA 共享一个 warpgroup, 而 writeback 在另一个中运行. 第 8 步扩大了 CTA 之间的合作, 生成了跨越两者的 256×256 tile. 第 9 步进一步推动计算密度: cluster 输出增长到 512×256, 每个阶段 B tile 都被 consumer 重用, 我们得到了教程中最密集的变体.

在这一切过程中, 有一件事始终保持不变. SMEM, TMEM 和寄存器 layout 仍然遵循我们在前两章中构建的约定; 改变的是“谁合作”, 而不是数据的布局方式. 第 8 步是协作 scope 第一次加宽超过单个 CTA, 因此其 operand tile 分为两个 CTA shared memory, 一个 layout 沿着 `cbx` cluster 轴.


## 第 7 步: warp specialization + 流水线

单 warpgroup kernel 浪费性能的原因很简单: 所有 thread 都沿同一条路径依次加载, 计算和写回. 加载时 Tensor Core 无事可做, 计算时 TMA 引擎又处于空闲. 解决方案就是 *warp specialization*. 我们不再让同一组 thread 轮流完成所有工作, 而是把不同任务交给专门的 warp, 让它们通过软件流水线连接并发执行. 这是 GEMM 优化路径中最大的架构变化, 本章其余内容都建立在它之上. 这里的基准规模为 M=N=K=4096.

> 此步骤更改的内容: scope
> - scope: 一个 warpgroup 步行负载→ MMA → writeback 依次成为三个并发角色 (TMA producer, MMA) consumer, writeback) 由满/空 barrier 连接.
> - layout: 不变, SMEM 阶段和 TMEM accumulator 与第 6 步相同.
> - dispatch: 不变, TMA 加载, `tcgen05` MMA.

主题.

- warp specialization: 将不同的 warp / warpgroup 专用于不同的任务

- 高级 barrier 抽象: `TMABar`, `TCGen05Bar`, `MBarrier`

- `PipelineState` 用于自动阶段/ phase 管理

- `warpgroup_sync` barrier 用于每个 warpgroup 同步的 ID

(多阶段 SMEM 流水线和持久 `ClusterPersistentScheduler2D` 与第 5 步至第 6 步一样重复使用; 这里只有 scope 拆分是新的.)

### 从顺序到并发

在介绍角色和 barrier 之前, 先隔离一下 warp specialization 消除的调度瓶颈. 下图使用 Step-4 风格的顺序时间线作为第 4 步至第 6 步中的预专业化 kernel 的紧凑参考, 然后将其放在第 7 步 warp 专业化时间表之上, 因此引擎利用率的差异一目了然.

![Warp 专业化时间线](./images/warp_specialization_timeline.png)

最上面的是预专业化单 warpgroup 模式: 相同的非专业化 thread 组同时拥有加载路径和 MMA 路径, 因此一个引擎可以轻松地闲置, 而另一个引擎则处于活动状态. 第 5 步和第 6 步通过双缓冲和持久调度改进了该基线, 但它们尚未将加载和计算拆分为独立的 producer 和 consumer 角色. 从底层来看, 专业化打破了这种轮流制. TMA producer 预取下一个 tile, 而 MMA consumer 正忙于计算, writeback 自行继续. producer warp 3 发出下一个负载, 而 consumer warp 0 仍在当前 MMA 中工作, 因此两个引擎都不必等待另一个引擎. load/ MMA 切换使用两个 barrier:

- `tma2mma` (TMA → MMA): 表示加载的 SMEM 数据已准备好供 MMA 使用.
- `mma2tma` (MMA → TMA): 表示 MMA 已完成读取缓冲区, 以便 TMA 可以将其重新用于下一次加载.

图中的一个细节一开始看起来像是一个错误: `mma2tma` 箭头向前跳过了阶段. 原因是环形缓冲区. 对于 `PIPE_DEPTH=2`, 有两个 SMEM 缓冲区: 阶段 0 和阶段 1; TMA Load k=0 填充缓冲区 0, TMA Load k=1 填充缓冲区 1. 当 MMA Compute k=0 完成读取缓冲区 0 时, 它会向 `mma2tma` 发出信号, 表示缓冲区是空闲的, 但实际想要返回缓冲区 0 的负载是 TMA Load k=2, 而不是 k=1 (正在使用缓冲区 1). 这就是为什么 `mma2tma` 箭头从 MMA 计算 k=0 一直到达 TMA 负载 k=2. 该版本跳跃到阶段仅仅是因为环有两个插槽.

### warp 角色

时间表显示了我们“为什么”分工; 下一个问题是“谁”负责每个部分. 专业化将三个作业 (加载, 计算, writeback) 分配给特定的 warp, 以便它们可以同时运行. 与 `WG_NUMBER=2` 相比, kernel 使用了两个 warpgroup (角色表中缩写为 WG):

| 演员 | 地点 | 职位 |
|-------|----------|-----|
| TMA producer | warpgroup 1, warp 3 | 通过 TMA 连续加载 A 和 B tile |
| MMA consumer | warpgroup 1, warp 0 | 数据准备好后立即运行 MMA |
| writeback | warpgroup 0 (全部 warp) | 读取 TMEM 结果, 写入 GMEM |

### 4 barrier

三个并发参与者需要四个 barrier, 并且这四个角色整齐地排列成两个相反的方向. 前向路径 (TMA → MMA → writeback) 发出数据 *就绪* 的信号; 它的消息是“你等待的 tile 就在这里.”后向路径 (writeback → MMA →TMA) 向缓冲区 *释放* 发出信号: “你想要的插槽再次空闲.”一旦你了解了命名约定, 名称就会自行读取: 每个都是 `source2destination`, 因此 `tma2mma` 就是 barrier, TMA 在其上发出 MMA 信号.

| barrier | 类型 | 方向 | 含义 |
|---------|------|-----------|---------|
| tma2mma | `TMABar` | TMA -> MMA | “SMEM 数据已准备好” |
| mma2tma | `TCGen05Bar` | MMA -> TMA | “SMEM 缓冲区可以重复使用” |
| mma2ld | `TCGen05Bar` | MMA -> writeback | “TMEM 结果已准备就绪” |
| ld2mma | `MBarrier` | writeback -> MMA | “下一个 tile TMEM 免费” |

为什么每个 barrier 都有它的 *类型*? 该类型遵循 producer 宣布其完成的方式. TMA 负载 使用 `TMABar`, 这是一个具有字节计数的 mbarrier: 一旦传输的字节到达, TMA 硬件本身就会到达 barrier, 因此 consumer 得知数据已准备好, 而无需任何 thread 进行轮询. TMA store 无法使用此功能 (商店没有人可以通知), 因此他们回退到 `cp_async.bulk.commit_group()` + `wait_group(0)`, 其中发出的 thread 只是等待自己的写入耗尽. MMA 操作 使用 `TCGen05Bar`, 其中 `tcgen05.commit()` 指令在 MMA 完成时向 barrier 发出信号.

这里的一个小细节将在第 8 步中得到回报. `arrive` 调用传递 `cta_mask=0`, 因为在单 CTA kernel 中没有其他 CTA 需要发出信号. 当第 8 步形成 cluster 时, 该参数将变为非零并成为唤醒协作 CTA 的机制.

### 流水线状态

四个 barrier 告诉角色“何时”缓冲区准备就绪; 当流水线循环时, 仍然需要跟踪每个角色所在的 *哪个* 缓冲区. 该簿记是 `PipelineState` 管理的. 环形缓冲区同时携带两部分记录: 我们当前所在的插槽, 以及我们正在等待该插槽的 barrier 的哪个“phase”. 在流水线循环中手动跟踪两者正是会产生逐一错误的情况, 并且这里的逐一错误会导致整个 kernel 陷入僵局. `PipelineState` 的存在是为了将两者保持在一起, 这样你就不必:

```python
# The producer starts ready at phase 1.
tma_ps = PipelineState(PIPE_DEPTH, phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

最初的 `phase` 决定了角色的第一个 `wait` 是让它运行还是让它阻塞, 而正确的答案在流水线的两端是相反的, 这就是让人绊倒的部分:
- `phase=1` (producer) -> 第一个 `wait(phase=1)` 看到 barrier 仍为 phase 0, 并且因为 0!= 1 它 立即通过. 这正是我们想要的, 因为缓冲区开始为空, 并且 producer 应该可以立即开始填充它们.

- `phase=0` (consumer) -> 第一个 `wait(phase=0)` 在 phase 0 处看到 barrier, 并且由于 0 == 0 它 块. 这又是我们想要的, 因为还没有数据, 并且在 producer 到达之前, consumer 没有任何内容可读取.

给两端相同的起始 phase, 你就会陷入僵局, 或者更糟糕的是, 静默错误, 所以这个选择值得正确选择.

### `warpgroup_sync` barrier ID

专业化引入了很容易陷入的同步危险. 一旦每个 warpgroup 运行不同的代码路径, 熟悉的 `cta_sync()` 将陷入死锁: 它使用硬件 barrier #0 并坚持 *每个* CTA thread 到达, 但在 warpgroup 分支内只有一些其中 thread 存在. 相反, 我们需要的是范围为单个 warpgroup 的 barrier. GPU 为我们提供了 16 个名为 barrier (ID 0–15) 的数据, 因此 kernel 达到了 `warpgroup_sync(10)`, 它仅同步一个 warpgroup 内的 thread. 当多个 warpgroup 各自需要自行同步时 (如多 consumer 第 9 步中发生的情况), 它们通过 `warpgroup_sync(wg_id + 10)` 获取不同的 ID, 这样它们就不会在同一硬件 barrier 上发生冲突.

实施.

我们在这里使用 `PIPE_DEPTH=2`, 这是仍然允许加载和计算重叠的最小深度. 越深入, 隐藏的内存延迟就越多, 直至达到 SMEM 预算的限制; 下面的“当第 7 步行为不当时”讨论详细讨论了这种权衡. 现在掌握了所有部分 (角色, 四个 barrier, `PipelineState` 和 warpgroup 范围同步), 我们可以将完整的 kernel 组合在一起:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
from tvm.tirx.lang.pipeline import TMABar, TCGen05Bar, MBarrier, PipelineState
from tvm.tirx.lang.tile_scheduler import ClusterPersistentScheduler2D

SM_COUNT = 148  # Number of SMs on NVIDIA B200 GPU
F16_SIZE = 2

def hgemm_v7(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K
  PIPE_DEPTH = 2
  WG_NUMBER = 2

  A_layout = tma_shared_layout(
    a_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K),
  )
  B_layout = tma_shared_layout(
    b_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_N, BLK_K),
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
    bx = T.cta_id([SM_COUNT])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- Allocation ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tma2mma = TMABar(pool, PIPE_DEPTH)
    mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
    mma2ld  = TCGen05Bar(pool, 1)
    ld2mma  = MBarrier(pool, 1)
    pool.move_base_to(1024)
    Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((BLK_M, BLK_N), d_type, layout=D_layout)

    # --- Barrier init ---
    tma2mma.init(1)
    mma2tma.init(1)
    mma2ld.init(1)
    ld2mma.init(128)   # all 128 Warpgroup 0 threads arrive
    pool.commit()

    # --- TMEM alloc + fence ---
    if wg_id == 0:
      if warp_id == 0:
        T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

    # --- Tile scheduler ---
    tile_scheduler = ClusterPersistentScheduler2D(
      "ts", num_m_tiles=M // BLK_M, num_n_tiles=N // BLK_N,
      l2_group_size=8, num_clusters=SM_COUNT)
    tile_scheduler.init(bx)
    m_st = T.meta_var(tile_scheduler.m_idx * BLK_M)
    n_st = T.meta_var(tile_scheduler.n_idx * BLK_N)

    # =============================================
    # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
    # =============================================
    if wg_id == 1:
      if warp_id == 3:
        # === TMA Producer ===
        tma_ps = PipelineState(PIPE_DEPTH, phase=1)

        @T.inline
        def tma_load(k_offset):
          Tx.copy_async(Asmem[tma_ps.stage, :, :],
            A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=1,
            mbar=tma2mma.ptr_to([tma_ps.stage]))
          Tx.copy_async(Bsmem[tma_ps.stage, :, :],
            B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=1,
            mbar=tma2mma.ptr_to([tma_ps.stage]))

        if T.filter(lane_id, T.ptx.elect_sync()):
          while tile_scheduler.valid():
            for k in range(K_TILES):
              mma2tma.wait(tma_ps.stage, tma_ps.phase)
              tma_load(k * BLK_K)
              tma2mma.arrive(tma_ps.stage,
                (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
              tma_ps.advance()
            tile_scheduler.next_tile()

      elif warp_id == 0:
        # === MMA Consumer ===
        mma_ps = PipelineState(PIPE_DEPTH, phase=0)
        ld_ps = PipelineState(1, phase=1)

        if T.filter(lane_id, T.ptx.elect_sync()):
          while tile_scheduler.valid():
            # Wait for TMEM to be free from previous tile's writeback
            ld2mma.wait(ld_ps.stage, ld_ps.phase)
            ld_ps.advance()

            for k in range(K_TILES):
              tma2mma.wait(mma_ps.stage, mma_ps.phase)
              Tx.gemm_async(
                tmem[:, :BLK_N],
                Asmem[mma_ps.stage, :, :],
                Bsmem[mma_ps.stage, :, :],
                accum=(k != 0), dispatch="tcgen05", cta_group=1)
              mma2tma.arrive(mma_ps.stage, cta_group=1, cta_mask=0)
              mma_ps.advance()

            # Signal results ready for writeback
            mma2ld.arrive(0, cta_group=1, cta_mask=0)
            tile_scheduler.next_tile()

    # =============================================
    # Warpgroup 0: Writeback
    # =============================================
    elif wg_id == 0:
      wb_ps = PipelineState(1, phase=0)
      reg_f16 = T.alloc_local((BLK_N,), d_type)

      while tile_scheduler.valid():
        # Wait for MMA results
        mma2ld.wait(wb_ps.stage, wb_ps.phase)
        wb_ps.advance()

        # Read TMEM -> registers (warpgroup scope)
        reg = T.alloc_local((BLK_N,), acc_type)
        reg_wg = reg.view(128, BLK_N,
          layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
        Tx.wg.copy_async(reg_wg[:], tmem[:, :BLK_N])
        T.ptx.tcgen05.wait.ld()

        # Signal TMEM free (all 128 threads arrive)
        ld2mma.arrive(0, cta_id=0, pred=True)

        # Cast fp32 -> fp16
        Tx.cast(reg_f16[:], reg[:])

        # Write to Dsmem + TMA store
        Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
        T.ptx.fence.proxy_async("shared::cta")
        T.cuda.warpgroup_sync(10)
        if warp_id == 0:
          if lane_id == 0:
            Tx.copy_async(D[m_st:m_st+BLK_M, n_st:n_st+BLK_N],
              Dsmem[:, :], dispatch="tma")
            T.ptx.cp_async.bulk.commit_group()
            T.ptx.cp_async.bulk.wait_group(0)
        T.cuda.warpgroup_sync(10)

        tile_scheduler.next_tile()

    # --- Cleanup ---
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

要运行其中任何一个 kernel, 请重复使用我们在第 1 步中展示的相同编译/运行/检查工具 ([构建 Tiled GEMM](/gpupro/tiled-gemm/)): 将 `hgemm_v1` 替换为 `hgemm_v7`, `hgemm_v8`, 或者 `hgemm_v9`, 并选择问题大小, 例如 `M=N=K=4096`. 请记住, 集群步骤需要 `M` 和 `N` 是其 cluster tile 的倍数 (第 8 步为 `256×256`, 第 9 步的 `512×256`), 因此很小的 `128×128` 大小根本不会产生 tile. 每个新的 Python 会话编译一个步骤, 在切换步骤之前重新启动 kernel, 因为 kernel 重用内部名称并且编译器保留每个会话的状态. 每个步骤的计时收集在下面的 *端到端结果* 中.

### epilogue (writeback) 详情

第 7 步可以负担得起简单的 epilogue. 仅使用 `BLK_N=128` 列, writeback warpgroup 将整个 TMEM tile 一次性读取到寄存器中, 然后发出一个 TMA store. 第 8 步和第 9 步不会有这种奢侈, 这正是他们引入我们稍后添加的分块的原因, 但现在的顺序是:

1. 等待 MMA: `mma2ld.wait(phase)`. 本教程中的第 8 步和第 9 步在此添加 `fence.after_thread_sync()` 作为保守的额外内容; MMA -completion mbarrier 已经涵盖了排序, 并且大多数 kernel (包括 CUTLASS) 省略了它, 因此第 7 步也省略了它.
2. 通过 `Tx.copy_async(reg_wg, tmem[:,:BLK_N])` 读取 TMEM ->寄存器(每个 thread, warpgroup scope 为 128 fp32, 然后是 `T.ptx.tcgen05.wait.ld()`).
3. 信号 MMA: `ld2mma.arrive(0, cta_id=0, pred=True)` (128 个 thread 全部到达); 下一个 tile 的 TMEM 现在免费. 两个 `arrive` kwargs 在集群步骤中重复出现: `cta_id` 命名 *哪个 CTA* 的 barrier 副本来发出信号 (`0` = 此 CTA, 本地 barrier; 在步骤中 8 合作社通过 `cta_mask` 到达目标 CTA-0), 而 `pred` 是一个 per-thread 谓词, 用于控制此 thread 是否实际到达 (此处为 `True`, 因此每个 writeback thread 计入到达总数).
4. 在寄存器中投射 fp32 -> fp16.
5. 写入寄存器-> Dsmem, 然后 `fence.proxy_async("shared::cta") + warpgroup_sync(10)` 进行刷新.
6. TMA store Dsmem -> GMEM 通过 `cp_async.bulk.commit_group() + wait_group(0)`.

第 8 步 (使用 `BLK_N=256`) 和第 9 步 (每个 consumer 使用 `MMA_N=256`) 无法保持这种一次性形式, 原因是寄存器压力. 每个 thread 读取 256 个 fp32 值意味着 256 × 4 = 1024 字节必须同时存在于每个 thread 的寄存器中, 这有溢出到本地内存的风险, 最重要的是, 会强制使用更大的 Dsmem 缓冲区. 因此, 这些步骤将 writeback 分解为 `EPI_N` 列块 (`EPI_N=64`): 每次迭代仅保留 `EPI_N` fp32 寄存器活动并发出相应较小的 TMA store, 用更多的商店说明换取保持舒适的寄存器预算.

实施说明.

- 持久 kernel: `bx = T.cta_id([SM_COUNT])` --- 每个 SM 一个 CTA, 在 tile 上循环

- L2 友好的调度: `ClusterPersistentScheduler2D` 为缓存局部性命令 tile

- 这种模式 --- warp specialization 加上软件流水线--- 在高性能 GEMM kernel 中很常见, 包括 CUTLASS 风格的设计.

### 当第 7 步出现问题时

第 7 步是第一个 GEMM kernel, 其中 TMA 加载, `tcgen05` MMA 和 writeback 都同时运行. 第 8 步和第 9 步中会出现相同的故障模式: barrier 计数不匹配, 角色防护位于错误位置, 缺少栅栏或在 TMA store 耗尽之前重复使用暂存缓冲区. 这些情况的调试清单收集在 [调试 Warp-Specialized Kernel](/gpupro/debugging-warp-specialized-kernels/) 中.

流水线深度调整. Step 7 kernel 运行在 `PIPE_DEPTH=2` (最小值). 将其推至 4 或 6 可以让 TMA producer 进一步领先于 MMA consumer 并隐藏更多内存延迟, 但它是通过花费更多 SMEM 来实现的, 而 SMEM 是有限的. B200 为每个 SM 提供 228 KB. 对于 `BLK_M=BLK_N=128, BLK_K=64, fp16`, 每个流水线阶段的 A 和 B 成本一起为 `(128*64 + 128*64) * 2 = 32 KB`, 而 `Dsmem` writeback 暂存缓冲区在顶部又增加了 32 KB. 这使得 `PIPE_DEPTH=4` 约为 160 KB, `PIPE_DEPTH=6` 约为 224 KB, 完全超出了预算. 要更深入地了解, 你必须重新考虑 writeback 暂存策略.

---

warp specialization 获得了一台 CTA 配合的 thread. 下一步将扩大这种合作, 跨越 CTA 本身的边界, 让其中两个人在一个更大的 tile 上工作.


## 第 8 步: 2-CTA cluster

第 7 步使引擎重叠, 但每个 CTA 仍然单独计算自己的 128×128 tile, 重新加载邻居无法借用的 operand. 第 8 步打破了这种隔离. 两个 CTA 加入 cluster 并获得访问彼此的 shared memory 的能力, 因此单个协作 `tcgen05` MMA 会生成一个 256×256 tile, 跨越两个 CTA 现在, B 的一负载可以提供两倍的 MMA 工作. 和之前一样, M=N=K=4096.

> 此步骤更改的内容: scope + layout + dispatch
> - scope: 协作的 scope 现在跨越 cluster 中的两个 CTA, 而不是一个.
> - layout: operand tile 分布在两个 CTA 的 SMEM 上; CTA 0 拥有共享完成 barrier (`remote_view`).
> - dispatch: MMA 获得 `cta_group` / `cta_mask`, 因此 `tcgen05` 作为 2-CTA 协作操作运行.

主题.

- CTA cluster: 多个 CTA 在更大的 tile 上协作

- 通过 `map_shared_rank` 进行跨 CTA SMEM 访问

- `cta_group=2` 用于在 256x256 cluster 上进行协作 MMA tile

- 使用 `cta_mask` 进行跨 CTA barrier 信令


### cluster tile 形状

整个优化依赖于单一硬件功能: 使用 `cta_group=2`, MMA 可以读取由 *两个* CTA 上演的 operand tile, 而不仅仅是它所在的那个. 每个 CTA 加载一个存储 B 的 128 行切片, 转置后变成 128 个逻辑输出列, 并且协作的 MMA 将两个切片重新缝合在一起, 形成一个 operand. 下图描绘了两个 CTA 的 A 和 B 切片如何组合成单个 256×256 cluster tile:

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo-zh/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>

*交互式: 每个 CTA 拥有一个 A 行片和一个存储 B 行片, 然后通过 cluster (DSMEM) 读取另一个 CTA 的存储 B 片. 在 `B.T` 之后, 两个存储的 B 切片覆盖整个输出列跨度, 因此这对生成一个 256×256 输出 tile.*

为什么 A 和 B 在 cluster 上分割: 要了解 256×256 tile 是如何分区的, 请回想一下教程将 GEMM 存储为 `D = A @ B.T`, 其中存储的 B 具有形状 `N x K`. 当 cluster 中有两个 CTA 时, 分割就完全消失了:

- A 垂直分割: CTA-0 保存 A0 (第 0-127 行), CTA-1 保存 A1 (第 128-255 行). 堆叠: `[A0; A1]` (256 行).
- 存储的 B 按行分割: CTA-0 加载 B 行 0-127, CTA-1 加载 B 行 128-255. 因为数学使用 `B.T`, 所以这两个存储的行切片成为逻辑右侧 operand 的两个 128 列切片.
- 使用 `cta_group=2`, MMA 硬件通过跨 CTA shared memory 访问从 两个 CTA 的 SMEM 读取 B, 因此它会看到完整的逻辑输出列范围.
- 结果: 两个 CTA 在一个 256x256 输出 tile 上进行协作. 每个 CTA 写入该 tile 的 128x256 行条带.

值得停下来看看为什么这是一次真正的胜利, 而不仅仅是工作的重新洗牌. 每个 CTA 仍然只加载 128×K 的 A 和 128×K 的 B, 因此 cluster 作为一个整体阶段大约是单个 CTA 的 operand 的 2 倍, 但它产生了 256×256 tile, 其输出 FLOP 约为 128×128 tile 的 4 倍. 因此, MMA 每个阶段 operand 字节的工作负载大约是两倍, 因为每个 CTA 的 B 切片都通过协作 MMA 与其他 CTA 的 A 切片重用. 换句话说, 算术强度大约翻倍, 而这正是仍然依赖内存的 kernel 所需的杠杆: 端到端表中约 2.2 倍的加速来自于将相同的字节提供给更多的数学运算.

### tile 地址计算

现在 cluster 是工作单元, tile 调度器也必须计入 cluster tile 中. 它返回的每个 `(m_idx, n_idx)` 都命名了一个完整的 256×256 区域, 并且 cluster 内的两个 CTA 将该区域分开. 将 cluster 坐标转换为每个实际加载的每个 CTA 切片, 如下所示:

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

两个 CTA 都在 *相同* 256×256 cluster tile 上工作, 并且单坐标 `cbx` (CTA 在 cluster 中的位置, 0 或 1) 是选择该 CTA 沿两个方向的贡献的因素轴. `m_st` 选择此 CTA 拥有的输出行条带, `n_st` 选择它馈送到协作 MMA 中的存储 B 切片, writeback 随后发出 256 列的两个 128 列一半输出跨度. 另请注意, `num_m_tiles = M // 256` 和 `num_n_tiles = N // 256` 计数为 cluster tile, 而不是单个 CTA tile.

乍一看, `cbx` 出现在 `m_st` 和 `n_st` 中, 就好像行偏移以某种方式泄漏到列中一样, 但两种用法都是正确的, 并且值得弄清楚原因. 在 writeback 路径上, `cbx` 仅属于 M 轴: 每个 CTA 拥有不同的 128 行条带 (`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`, 因此 CTA-0 写入 `m_idx*256.. +128` 行, CTA-1 写入接下来的 128 行), 但两者都 CTA 写入 cluster tile 的“完整”256 个输出列. 这正是商店从 cluster 的 `n_idx` (`n_st_epi = n_idx * 256 + no * 128`, 看不到 `cbx`) 而不是从每个 CTA `n_st` 派生其列的原因. `n_st` 携带 `cbx` 的原因是每个 CTA 将不同的存储 B 行切片加载到 MMA 中: 其中, `cbx` 是 *加载* 偏移量, 而不是 CTA 的输出列偏移量.

### 第 7 步的代码更改

与第 7 步的差异有六处编辑, 每处都编码我们刚刚描述的 cluster 约定的一个片段:

```python
# 1. Cluster launch
# cbx is the CTA index within the cluster (zero or one).
cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])

# 2. Cooperative MMA (was cta_group=1)
Tx.gemm_async(..., cta_group=2)

# 3. Cross-CTA shared memory access
B_remote = T.ptx.map_shared_rank(Bsmem, cta_id=1)

# 4. Cross-CTA barrier
tma2mma_cta0 = T.decl_buffer(
  [CTA_GROUP], "uint64",
  data=T.ptx.map_shared_rank(tma2mma.ptr_to([0]), 0),
  scope="shared"
)

# 5. mma2tma / mma2ld arrives go from cta_mask=0 (single CTA, Step 7)
#    to cta_mask=3 (signal both CTAs in the cluster)
mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)

# 6. Cluster sync replaces cta_sync at the end
T.cuda.cluster_sync()
```


### cluster - scope 更改

这六个编辑都源于同一个转变: 协作的 scope 现在是 cluster, 而不是单个 CTA. 以下几点说明了这种扩大在实践中意味着什么: 每个 CTA 如何找到自己的位置, cluster 与谁的 barrier 进行协调, 以及哪个 CTA 实际发布了合作 MMA.

- cluster CTA ID: `cbx` 告诉每个 CTA 它在 cluster 中的位置 (0 或 1). CTA-0 处理 A 行 0-127, CTA-1 处理 A 行 128-255.

- 远程 barrier 视图: 在 cluster 中, 每个 CTA 都有自己的 SMEM 和自己的 barrier, 这提出了一个明显的问题: 如果 CTA-1 需要等待 CTA-0 产生的东西, 它实际上接触谁的 barrier? 答案是指定 CTA-0 的 barrier 作为单个协调点, 并让 cluster 中的任何 CTA 到达它们. `map_shared_rank(tma2mma.ptr_to([0]), 0)` 使用 TIRx 包装器 `tma2mma.remote_view(0)` 返回一个指向 CTA-0 的 barrier 的 cluster 宽指针, 从那时起, 每个到达和等待目标都指向 CTA-0 的副本.

- 仅来自 CTA-0 的 MMA dispatch: 很容易将 `cta_group=2` 理解为并行启动两个引擎, 但事实并非如此. CTA-0 恰好发出一个 `tcgen05.mma`, 然后硬件驱动一个跨两个 CTA 的“单一协作” MMA, 从两个 SM 的 SMEM 读取 operand 并在两个 SM 的 TMEM 上写入 accumulator. CTA-1 根本不发出 MMA 问题. (每个 SM 只有一个 `tcgen05` 引擎, 所以 `cta_group=2` 是一个跨 SM MMA, 而不是两个并行运行的引擎.) 这就是代码用 `if cbx == 0:` 保护 MMA 的原因.

- 多播到达: `tcgen05.commit(..., cta_group=2, cta_mask=3)` 仅由 CTA-0 发出, 但向两个 CTA 发出 barrier 信号. `cta_mask=3` (二进制 `11`) 表示 CTA-0 和 CTA-1 都是目标.

- ld2mma 初始化计数: `init(128 * CTA_GROUP)` --- 两个 CTA 的 writeback warpgroup (各 128 个 thread) 均到达.


实施.

```python
def hgemm_v8(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  CTA_GROUP = 2
  BLK_M, BLK_N, BLK_K = 128, 128, 64
  MMA_M, MMA_N = 256, 256
  K_TILES = K // BLK_K
  PIPE_DEPTH = 4
  WG_NUMBER = 2
  F16_SIZE = 2  # fp16

  A_layout = tma_shared_layout(
    a_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_M, BLK_K),
  )
  B_layout = tma_shared_layout(
    b_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_N, BLK_K),
  )
  D_layout = tma_shared_layout(
    d_type,
    SwizzleMode.SWIZZLE_128B_ATOM,
    (BLK_M, 128),
  )

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    bx = T.cta_id([SM_COUNT])
    cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- Allocation ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tma2mma = TMABar(pool, PIPE_DEPTH)
    mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
    mma2ld  = TCGen05Bar(pool, 1)
    ld2mma  = MBarrier(pool, 1)
    pool.move_base_to(1024)
    Asmem = pool.alloc((PIPE_DEPTH, BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((BLK_M, 128), d_type, layout=D_layout)

    # --- Barrier init ---
    tma2mma.init(1)
    mma2tma.init(1)
    mma2ld.init(1)
    ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
    pool.commit()

    # --- TMEM alloc (cooperative) ---
    if wg_id == 0:
      if warp_id == 0:
        T.ptx.tcgen05.alloc(
          T.address_of(tmem_addr),
          n_cols=512,
          cta_group=CTA_GROUP,
        )
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

    # --- Tile scheduler (cluster tiles) ---
    tile_scheduler = ClusterPersistentScheduler2D(
      "ts", num_m_tiles=M // 256, num_n_tiles=N // 256,
      l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
    tile_scheduler.init(bx // CTA_GROUP)
    m_idx = T.meta_var(tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.n_idx)
    m_st = T.meta_var((m_idx * CTA_GROUP + cbx) * BLK_M)
    n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

    # --- Cross-CTA barrier view ---
    tma2mma_cta0 = tma2mma.remote_view(0)

    # =============================================
    # Warpgroup 1: TMA Producer (warp 3) + MMA Consumer (warp 0)
    # =============================================
    if wg_id == 1:
      if warp_id == 3:
        tma_ps = PipelineState(PIPE_DEPTH, phase=1)

        @T.inline
        def tma_load(k_offset):
          Tx.copy_async(Asmem[tma_ps.stage, :, :],
            A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=CTA_GROUP,
            mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
          Tx.copy_async(Bsmem[tma_ps.stage, :, :],
            B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=CTA_GROUP,
            mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

        if T.filter(lane_id, T.ptx.elect_sync()):
          while tile_scheduler.valid():
            for k in range(K_TILES):
              mma2tma.wait(tma_ps.stage, tma_ps.phase)
              tma_load(k * BLK_K)
              if cbx == 0:
                tma2mma_cta0.arrive(tma_ps.stage,
                  CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE)
              tma_ps.advance()
            tile_scheduler.next_tile()

      elif warp_id == 0:
        mma_ps = PipelineState(PIPE_DEPTH, phase=0)
        ld_ps = PipelineState(1, phase=1)

        if cbx == 0:
          if T.filter(lane_id, T.ptx.elect_sync()):
            while tile_scheduler.valid():
              ld2mma.wait(ld_ps.stage, ld_ps.phase)
              ld_ps.advance()

              for k in range(K_TILES):
                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                Tx.gemm_async(
                  tmem[:, :MMA_N],
                  Asmem[mma_ps.stage, :, :],
                  Bsmem[mma_ps.stage, :, :],
                  accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                mma_ps.advance()

              mma2ld.arrive(0, cta_group=CTA_GROUP, cta_mask=3)
              tile_scheduler.next_tile()

    # =============================================
    # Warpgroup 0: Writeback (256 columns in 2 x 128-column chunks)
    # =============================================
    elif wg_id == 0:
      wb_ps = PipelineState(1, phase=0)
      reg_f16 = T.alloc_local((128,), d_type)

      while tile_scheduler.valid():
        mma2ld.wait(wb_ps.stage, wb_ps.phase)
        wb_ps.advance()
        T.ptx.tcgen05.fence.after_thread_sync()

        for no in T.unroll(2):  # 2 chunks of 128 columns = 256 total
          reg = T.alloc_local((128,), acc_type)
          reg_wg = reg.view(128, 128,
            layout=TileLayout(S[(128, 128) : (1@tid_in_wg, 1)]))
          Tx.wg.copy_async(reg_wg[:], tmem[:, no * 128:(no + 1) * 128])
          T.ptx.tcgen05.wait.ld()
          Tx.cast(reg_f16[:], reg[:])
          Tx.copy(Dsmem[warp_id * 32 + lane_id, :], reg_f16[:])
          T.ptx.fence.proxy_async("shared::cta")
          T.cuda.warpgroup_sync(10)
          if warp_id == 0:
            if lane_id == 0:
              n_st_epi = T.meta_var(n_idx * 256 + no * 128)
              Tx.copy_async(D[m_st:m_st+BLK_M, n_st_epi:n_st_epi+128],
                Dsmem[:, :], dispatch="tma")
              T.ptx.cp_async.bulk.commit_group()
              T.ptx.cp_async.bulk.wait_group(0)
          T.cuda.warpgroup_sync(10)

        ld2mma.arrive(0, cta_id=0, pred=True)
        tile_scheduler.next_tile()

    # --- Cleanup ---
    T.cuda.cluster_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

  return kernel
```

2 个 CTA 有何变化.

- `CTA_GROUP = 2`, `MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` --- 两个 CTA 的 writeback 工作组均已到达

- TMA 到达字节计数包括两个 CTA: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` 和 `tcgen05.dealloc` 必须使用 `cta_group=2`

- writeback 将 256 个输出列拆分为两个 128 列块 --- 一次读取所有 256 个 TMEM 列超出了寄存器的容量. 第 9 步将块进一步缩小到 `EPI_N=64`

- `cluster_sync()` 最后替换 `cta_sync()` (确保所有 CTA 在 TMEM 释放之前完成)

所有额外的算术强度都直接显示在挂钟上: 第 8 步在 4096³ 时达到 0.104 ms, 大约是相同大小的 70 ms 第 1 步算法的 676 倍 (请参阅端到端表). kernel 现在倾向于计算限制, 这正是设置第 9 步的原因, 我们在其中添加第二个 MMA consumer 以保持更多的 Tensor Core 工作在进行中.

如果第 8 步的结果比第 7 步“慢”, 那么罪魁祸首几乎总是输入稍微错误的新 cluster 约定之一. 首先值得检查三件事: TMA 到达字节数为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`; 对于 256×256 cluster, 调度器尺寸为 `num_m_tiles=M//256, num_n_tiles=N//256` tile; writeback 发出两个 TMA store, 每 128 列块一个, 每个在 Dsmem 重用之前耗尽.

---

cluster 提高了 *跨* CTA 的重用性. 最后一步转向内部, 通过为 producer 提供第二个 MMA consumer 来保持供给, 从而提高每个 CTA 内的计算密度.


## 第 9 步: 多 consumer warp specialization

到第 8 步, MMA 确实很忙, 但是单个 consumer warp 只能如此快地咀嚼分阶段的 B tile, 而 B tile 就坐在 SMEM 中时间, 任何愿意阅读的人都可以阅读. 最终的优化利用了这一点: 它添加了第二个 MMA consumer, 将 *不同的* A 块与 *相同的* B tile 相乘. 每个 CTA 的计算密度翻倍, cluster 输出从 256×256 增长到 512×256. 和之前一样, M=N=K=4096.

> 此步骤更改的内容: scope + layout
> - scope: 1 个 MMA consumer 变成 2 个, 由 `warp_id` 选择.
> - layout: 一阶段 B tile 被 consumer 复用; A 获得 consumer 轴.
> - dispatch: 不变.

主题.

- 多个 MMA warp (consumer) 可实现更高的吞吐量

- 多个 writeback warpgroup, 具有独立的 barrier 插槽

- 本教程中最优化的 GEMM 变体使用的结构


### 多 consumer 结构

添加第二个 consumer 意味着 kernel 现在可以发挥更多不同的作用: 两个 MMA warp 而不是一个, 以及匹配的第二个 writeback warpgroup 耗尽额外的 accumulator. 通过 `NUM_CONSUMER=2` 和 `WG_NUMBER=3`, kernel 现在跨越三个 warpgroup (角色表中的缩写 WG):

| warpgroup | warp | 角色 |
|-----------|------|------|
| WG2 | warp 0 | MMA consumer 0: `Asmem[..., 0] x B` -> TMEM 列 `[0:256]` |
| WG2 | warp 1 | MMA consumer 1: `Asmem[..., 1] x B` -> TMEM 列 `[256:512]` |
| WG2 | warp 3 | TMA producer: 每个阶段加载 2x A 块 + 1x B 块 |
| WG0 | 全部 | writeback 为 consumer 0: 读取 TMEM `[0:256]` |
| WG1 | 全部 | writeback 为 consumer 1: 读取 TMEM `[256:512]` |

整个安排取决于一种不对称性. 每个 consumer 将其自己的 A 块与“相同”阶段的 B tile 相乘, 因此单个 B 负载现在可提供 2 倍的 MMA 工作, 并且 B 的每有用 FLOP 的负载成本实际上减半. 我们共享 B 而不是 A 的原因是两个 consumer 覆盖不同的 M 行条带: 它们的 A 块是真正不同的数据, 而 B 块对于两者来说是相同的. 练习 3 要求你说服自己这是唯一有效的分享.

### 第 8 步的更改

具体来说, 支持第二个 consumer 在几个地方触及了 kernel, 并且每一项更改都可以追溯到一个事实: 现在每个阶段有两个 A 块和两个 TMEM 范围来馈送和排出, 而 B 保持共享.阶段下面的编辑多了一个 A 块, 给每个 consumer 自己的 barrier 插槽, 并调整 tile 寻址以获得更高的 512×256 cluster tile.

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K),...)` --- 每个阶段 2 个 A 块, 每个 consumer 1 个

- TMA 同时加载 `Asmem[stage, 0]` 和 `Asmem[stage, 1]`, TMA 现在到达字节 `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE` (额外的 A 块)

- MMA warp `warp_id` 选择哪个 A 块和 TMEM 范围

- `mma2tma.init(NUM_CONSUMER)` --- consumer 信号 TMA 与阶段

- `mma2ld` 和 `ld2mma` 具有 `depth=NUM_CONSUMER` --- 每个 consumer 使用自己的 barrier 插槽 (`warp_id` 用于 MMA 侧, `wg_id` 用于 writeback 侧)

- tile 地址: `m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` --- M 方向具有额外的 `NUM_CONSUMER` 因子, 因为每个 cluster tile 现在跨越 `NUM_CONSUMER` M. tile 调度器中的 consumer 使用 `num_m_tiles = M // 256 // NUM_CONSUMER` (cluster tile 为 512x256)

- writeback 使用分块的 `EPI_N`, 因此每次迭代在寄存器中保留较少的 TMEM 读回值


实施.

```python
def hgemm_v9(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  CTA_GROUP = 2
  NUM_CONSUMER = 2
  BLK_M, BLK_N, BLK_K = 128, 128, 64
  MMA_N = BLK_N * CTA_GROUP   # 256
  K_TILES = K // BLK_K
  PIPE_DEPTH = 4
  EPI_N = 64
  WG_NUMBER = 3
  F16_SIZE = 2  # fp16

  A_layout = tma_shared_layout(a_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K))
  B_layout = tma_shared_layout(b_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (PIPE_DEPTH, BLK_N, BLK_K))
  D_layout = tma_shared_layout(d_type, SwizzleMode.SWIZZLE_128B_ATOM,
    (NUM_CONSUMER, BLK_M, EPI_N))

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    bx = T.cta_id([SM_COUNT])
    cbx, cby = T.cta_id_in_cluster([CTA_GROUP, 1])
    wg_id = T.warpgroup_id([WG_NUMBER])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- Allocation ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    tma2mma = TMABar(pool, PIPE_DEPTH)
    mma2tma = TCGen05Bar(pool, PIPE_DEPTH)
    # Depth 2, with one slot per consumer.
    mma2ld = TCGen05Bar(pool, NUM_CONSUMER)
    # Depth 2, with one slot per consumer.
    ld2mma = MBarrier(pool, NUM_CONSUMER)
    pool.move_base_to(1024)
    Asmem = pool.alloc(
      (PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K),
      a_type,
      layout=A_layout,
    )
    Bsmem = pool.alloc((PIPE_DEPTH, BLK_N, BLK_K), b_type, layout=B_layout)
    Dsmem = pool.alloc((NUM_CONSUMER, BLK_M, EPI_N), d_type, layout=D_layout)

    # --- Barrier init ---
    tma2mma.init(1)
    mma2tma.init(NUM_CONSUMER)  # each stage expects 2 arrivals
    mma2ld.init(1)              # each slot gets 1 arrival
    ld2mma.init(128 * CTA_GROUP)  # both CTAs' writeback threads
    pool.commit()

    # --- TMEM alloc (cooperative) ---
    if wg_id == 0:
      if warp_id == 0:
        T.ptx.tcgen05.alloc(
          T.address_of(tmem_addr),
          n_cols=512,
          cta_group=CTA_GROUP,
        )
    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), acc_type, scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

    # --- Tile scheduler (512x256 cluster tiles) ---
    tile_scheduler = ClusterPersistentScheduler2D(
      "ts", num_m_tiles=M // 256 // NUM_CONSUMER, num_n_tiles=N // 256,
      l2_group_size=8, num_clusters=SM_COUNT // CTA_GROUP)
    tile_scheduler.init(bx // CTA_GROUP)
    m_idx = T.meta_var(tile_scheduler.m_idx)
    n_idx = T.meta_var(tile_scheduler.n_idx)
    m_st = T.meta_var((m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M)
    n_st = T.meta_var((n_idx * CTA_GROUP + cbx) * BLK_N)

    tma2mma_cta0 = tma2mma.remote_view(0)

    # =============================================
    # Warpgroup 2: TMA Producer (warp 3) + 2 MMA Consumers (warp 0, 1)
    # =============================================
    if wg_id == 2:
      if warp_id == 3:
        # === TMA Producer: loads 2 A blocks + 1 B block per stage ===
        tma_ps = PipelineState(PIPE_DEPTH, phase=1)

        @T.inline
        def tma_load(k_offset):
          m_st_c1 = T.meta_var(m_st + CTA_GROUP * BLK_M)
          Tx.copy_async(Asmem[tma_ps.stage, 0, :, :],
            A[m_st:m_st+BLK_M, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=CTA_GROUP,
            mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
          Tx.copy_async(Asmem[tma_ps.stage, 1, :, :],
            A[m_st_c1:m_st_c1+BLK_M, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=CTA_GROUP,
            mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))
          Tx.copy_async(Bsmem[tma_ps.stage, :, :],
            B[n_st:n_st+BLK_N, k_offset:k_offset+BLK_K],
            dispatch="tma", cta_group=CTA_GROUP,
            mbar=tma2mma_cta0.ptr_to([tma_ps.stage]))

        if T.filter(lane_id, T.ptx.elect_sync()):
          while tile_scheduler.valid():
            for k in range(K_TILES):
              mma2tma.wait(tma_ps.stage, tma_ps.phase)
              tma_load(k * BLK_K)
              if cbx == 0:
                tma2mma_cta0.arrive(
                  tma_ps.stage,
                  CTA_GROUP
                  * (
                    NUM_CONSUMER * BLK_M * BLK_K
                    + BLK_N * BLK_K
                  )
                  * F16_SIZE,
                )
              tma_ps.advance()
            tile_scheduler.next_tile()

      elif warp_id < NUM_CONSUMER:
        # === MMA Consumer: warp_id selects A block and TMEM range ===
        mma_ps = PipelineState(PIPE_DEPTH, phase=0)
        ld_ps = PipelineState(1, phase=1)

        if cbx == 0:
          if T.filter(lane_id, T.ptx.elect_sync()):
            while tile_scheduler.valid():
              ld2mma.wait(warp_id, ld_ps.phase)
              ld_ps.advance()

              for k in range(K_TILES):
                tma2mma.wait(mma_ps.stage, mma_ps.phase)
                Tx.gemm_async(
                  tmem[:, warp_id * MMA_N:warp_id * MMA_N + MMA_N],
                  Asmem[mma_ps.stage, warp_id, :, :],
                  Bsmem[mma_ps.stage, :, :],
                  accum=(k != 0), dispatch="tcgen05", cta_group=CTA_GROUP)
                mma2tma.arrive(mma_ps.stage, cta_group=CTA_GROUP, cta_mask=3)
                mma_ps.advance()

              mma2ld.arrive(warp_id, cta_group=CTA_GROUP, cta_mask=3)
              tile_scheduler.next_tile()

    # =============================================
    # Warpgroup 0/1: Writeback (each reads its consumer's TMEM range)
    # =============================================
    elif wg_id < NUM_CONSUMER:
      wb_ps = PipelineState(1, phase=0)
      reg_f16 = T.alloc_local((EPI_N,), d_type)

      while tile_scheduler.valid():
        mma2ld.wait(wg_id, wb_ps.phase)  # wait for THIS consumer
        wb_ps.advance()
        T.ptx.tcgen05.fence.after_thread_sync()

        # Read TMEM in EPI_N=64 column chunks (4 iterations for 256 cols)
        for i in T.unroll(MMA_N // EPI_N):
          reg = T.alloc_local((EPI_N,), acc_type)
          reg_wg = reg.view(128, EPI_N,
            layout=TileLayout(S[(128, EPI_N) : (1@tid_in_wg, 1)]))
          col_st = T.meta_var(wg_id * MMA_N + i * EPI_N)
          col_end = T.meta_var(wg_id * MMA_N + i * EPI_N + EPI_N)
          Tx.wg.copy_async(reg_wg[:], tmem[:, col_st:col_end])
          T.ptx.tcgen05.wait.ld()
          Tx.cast(reg_f16[:], reg[:])
          Tx.copy(Dsmem[wg_id, warp_id * 32 + lane_id, :], reg_f16[:])
          T.ptx.fence.proxy_async("shared::cta")
          T.cuda.warpgroup_sync(wg_id + 10)
          if warp_id == 0:
            if lane_id == 0:
              m_st_epi = T.meta_var(
                (
                  m_idx * NUM_CONSUMER * CTA_GROUP
                  + wg_id * CTA_GROUP
                  + cbx
                )
                * BLK_M
              )
              n_st_epi = T.meta_var(n_idx * MMA_N + i * EPI_N)
              Tx.copy_async(
                D[m_st_epi:m_st_epi+BLK_M, n_st_epi:n_st_epi+EPI_N],
                Dsmem[wg_id, :, :], dispatch="tma")
              T.ptx.cp_async.bulk.commit_group()
              T.ptx.cp_async.bulk.wait_group(0)
          T.cuda.warpgroup_sync(wg_id + 10)

        ld2mma.arrive(wg_id, cta_id=0, pred=True)
        tile_scheduler.next_tile()

    # --- Cleanup ---
    T.cuda.cluster_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=CTA_GROUP)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=CTA_GROUP)

  return kernel
```

实施说明.

- 在此第 9 步设计中, `mma2ld` 和 `ld2mma` 都是与 `depth=NUM_CONSUMER` 共享的单个对象, 而不是单独的每个 consumer 对象. 插槽 0 连接 MMA warp 0 到 warpgroup 0, 插槽 1 连接 MMA warp 1 到 warpgroup 1; MMA 侧索引为 `warp_id`, writeback 侧索引为 `wg_id`.

## 端到端结果

下表报告了从初始基线到 warp 专用 cluster kernel 的测量里程碑, 以及 cuBLAS 参考. NVIDIA B200 上的参考数字, M=N=K=4096, fp16, 锁定时钟, 1000 次迭代定时基准:

| 步骤 | 技术 | 时间 | 加速比 |
|------|-----------|------|---------|
| 1 | 同步加载+ MMA | 70 毫秒 | 1× |
| 2 | K 环累积 | --- | 手柄 K 大于 1 tile |
| 3 | 空间平铺 | 53.6 毫秒 | ～1.3× |
| 4 | TMA 异步加载 | 0.49 毫秒 | ～142× |
| 5 |软件流水线| --- | 重叠负载+计算 |
| 6 | 持久 kernel | --- | L2 缓存局部性 |
| 7 | warp specialization | 0.23 毫秒 | ～309× |
| 8 | 2-CTA cluster | 0.104 毫秒 | ～676× |
| 9 | 多 consumer | 0.094 毫秒 | 〜744× |
| --- | cuBLAS (参考) | 0.094 毫秒 | 〜744× |

此表中的每次, 包括 70 ms 第 1 步基线, 都是在相同的 M=N=K=4096 大小下测量的, 这使得加速链具有端到端的可比性. 有必要准确说明 70 毫秒的实际含义, 因为它很容易被误读. 它 *不是* 来自 [构建 Tiled GEMM](/gpupro/tiled-gemm/) 的单 tile Step-1 kernel, 以 4096 立方运行; kernel 只能计算一个 128×128 tile 并且只能以小尺寸运行. 相反, 70 毫秒是一个简单的全尺寸基线, 它采用相同的顺序, 单 tile 方法, 并将其扩展到完整的 4096³ 问题. 第 1 步至第 3 步在 [构建 Tiled GEMM](/gpupro/tiled-gemm/) 中以小尺寸 (128×128 和 256³) 介绍, 以保持最初的演练简单; 这里的第 1 步和第 3 步行是它们的全尺寸基准对应行. 其余的破折号 (第 2 步、第 5 步和第 6 步) 标记了结构所示的步骤, 但没有单独计时.

将这些数字视为受控条件下的单个 B200 参考运行, 而不是作为排行榜条目. 每个步骤中嵌入的 `{.python.input}` 基准测试单元都是烟雾基准测试: 它们有利于发现趋势, 而不是声称峰值性能.

四种技术几乎占据了所有收益:

1. TMA 异步数据移动: 硬件复制引擎取代了软件复制 (从第 1 步 → 第 4 步约为 142 倍). 正确读取这个 142× 非常重要: 它反映了从单个 128×128- tile kernel (grid 1×1) 一直到带有 K 环, 空间平铺和许多 CTA, *连同* TMA; 这并不是 TMA 孤立的贡献. 隔离 TMA 意味着比较两个仅在复制机制上不同的全尺寸 kernel.
2. 软件流水线+ warp specialization: 通过赋予每个角色自己的专用角色来重叠加载和计算 (第 4 步 → 第 7 步中的约 2.2 倍).
3. CTA cluster: 2-SM 协作 MMA 改进了 CTA 之间的 B- tile 重用 (本基准测试中第 7 步 → 第 8 步的约 2.2 倍).
4. 多 consumer: 两个 MMA warp 可实现更高的计算密度 (第 8 步 → 第 9 步中约 10%).

在测量的里程碑上绘制的, 这四个相同的贡献追踪了从同步 tiled kernel 到 cuBLAS 参考的下降. 下图显示了所选的测量点:

![GEMM 优化之旅](./images/gemm_perf.png)

请注意, 随着我们沿着清单往下走, 收益会缩小, 这是结构性原因, 而不是努力减弱. 早期的步骤是为了解决“内存”瓶颈 (TMA 取代软件副本, cluster 提高算术强度), 而这正是 70 毫秒的大部分实际花费的地方, 因此这些步骤的回报最大. 到第 8 步, kernel 已经在 cuBLAS 的约 10% 范围内 (0.104 与 0.094 毫秒), 并且接近 *计算限制*, 这意味着几乎没有内存停顿可以隐藏; 第 9 步的多 consumer 重叠恢复了大部分剩余的内容. 大约 10% 的最终增益正是接近计算上限时所期望的: 它是接近解决的问题的收益递减, 而不是弱优化的标志.

我们在本章中构建的所有内容 (TMA 加载, `tcgen05` MMA, TMEM 读回和 warp 专用 barrier) 将直接延续到下一章. Flash Attention 重用了所有这些, 然后通过在两个 MMA phase 之间插入一个在线 softmax 步骤来提高难度, 而不是简单地重复单个步骤.


## 练习

1. 如果在第 7 步中将 TMA 和 MMA `PipelineState` 的初始 `phase` 设置为 `0`, 会发生什么情况? 画出死锁场景.
2. 对于第 8 步中的 `cta_group=2`, TMA 到达字节计数为 `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE`. 当每个 CTA 加载自己的数据时, 为什么要乘以 `CTA_GROUP`?
3. 第 9 步中, 每个 consumer 处理不同的 M 行, 但处理相同的 B tile. 为什么共享 B (而不是 A) 是正确的选择?

与你的代理一起尝试: 粘贴第 7 步 kernel 并要求其通过四个 barrier 跟踪一个 K- tile (`tma2mma`, `mma2tma`, `mma2ld`, `ld2mma`). 对于每个问题, 询问谁在等待, 谁到达, 什么 tile 可以安全读取, 以及哪个缓冲区随后可以重用.
