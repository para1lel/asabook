---
title: "Warp Specialization と Cluster による GEMM のスケーリング"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/warp-specialized-gemm/
pageClass: gpupro-page
---

::: info 概要
- パイプライン化された GEMM は依然として 1 つの warpgroup でロード、MMA、writeback を順番に行っており、この章でボトルネックが解消されます。
- ステップ 7 は warp を役割ごとに特化し、ステップ 8 では 2-CTA cluster を追加し、ステップ 9 では複数の consumer を追加します。
- 各ステップでシリアルボトルネックを除去し、ほぼ最先端のスループットを実現します。
:::

前章の [TMA による GEMM のパイプライン化](/ja/gpupro/pipelined-gemm/) で高速化した GEMM も、ロードの発行、MMA、結果の writeback を 1 つの warpgroup に任せています。ソフトウェアパイプラインを使っても、同じスレッド群が 3 つのエンジンを直列につなぐ合流点のままです。

症状は明確です。Tensor Core が動作している間は TMA ユニットが止まり、結果をメモリへ書き戻している間は Tensor Core が止まります。各エンジンが同じスレッド群を介して互いを待っているためです。解決策は、1 つのチームにすべての仕事を任せないことです。

私たちはこのアイデアを、協力拡大の三段階で追求しています。ステップ 7([Warp Specialization と Cluster による GEMM のスケーリング](#ステップ-7-warp-specialization-パイプライン))では、warp を producer、consumer、writeback の役割に特化させます。ステップ 8([Warp Specialization と Cluster による GEMM のスケーリング](#ステップ-8-2-cta-cluster))では、2 つの CTA をクラスタに結合し、それぞれの共有メモリ上で operand を共有します。ステップ 9([Warp Specialization と Cluster による GEMM のスケーリング](#ステップ-9-マルチ-consumer-warp-specialization))では 2 人目の MMA consumer が追加され、1 段階の tile で 2 倍の計算ができます。

3 つのステップを異なるスケールで一つのパターンとして見ると助けになります。ステップ 7 はパイプライン全体を 1 つの CTA 内に保持します: TMA と MMA は 1 つの warpgroup を共有し、writeback は別の CTA で実行されます。ステップ 8 では CTA 間の協力を広げ、両者にまたがる 256×256 tile が生成されます。ステップ 9 では計算密度がさらに高まります。cluster 出力は 512×256 に増加し、各段階化された B tile は両方の consumer で再利用され、チュートリアルで最も密度の高い barrier 構成に到達します。

このすべてを通して一つ変わらないことがあります。SMEM、TMEM、レジスタの layout は、前 2 章で作成した契約を引き続き尊重しています。変わるのはデータの配置ではなく、誰が協力するかです。ステップ 8 は、協調 scope が 1 つの CTA を超えて初めて拡大するため、その operand tile は 2 つの CTA の共有メモリに分割され、1 つの layout は `cbx` cluster 軸に沿って両方の CTA をまたいでいます。


## ステップ 7: warp specialization+パイプライン

単一 warpgroup の kernel が性能を引き出せない理由は単純です。全スレッドが同じ経路でロード、計算、writeback を順に行うため、ロード中は Tensor Core が、計算中は TMA エンジンが遊んでしまいます。解決策が *warp specialization* です。仕事ごとに専用の warp を割り当て、ソフトウェアパイプラインで接続しながら並行実行します。これは GEMM の最適化経路における最大のアーキテクチャ変更であり、本章の残りはこの構成を基礎にしています。ベンチマークのサイズは M=N=K=4096 です。

> このステップで変わるもの: scope
> - scope: 1 つの warpgroup ウォーキングロード→MMA→writeback を順に 3 つの並行ロール(TMA producer、MMA consumer、writeback)にし、フル/空の barrier でつながれます。
> - layout: 変更せず、Step 6 と同じ SMEM ステージと TMEM accumulator を使用します。
> - dispatch: 変更なし、TMA ロード、 `tcgen05` MMA。

トピック。

- warp specialization: 異なる warp や warpgroup を異なる任務に割り当てること

- 高レベルの barrier 抽象化: `TMABar`、 `TCGen05Bar`、 `MBarrier`

- `PipelineState` 自動ステージ/位相管理用

- `warpgroup_sync` warpgroup ごとの同期用 barrier ID

(多段階 SMEM パイプラインと永続 `ClusterPersistentScheduler2D` はステップ 5–6 から変更せずに再利用されており、ここでは scope 分割のみが新たに導入されています。)

### 連続から並行へ

役割や barrier を導入する前に、warp specialization によって除去されるスケジューリングのボトルネックを明確にしておくと役立ちます。下図はステップ 4 風の連続タイムラインをコンパクトな基準として使用し、ステップ 4 の warp specialization スケジュールの上に配置してエンジン利用率の違いを一目で確認できるようにしています。

![Warp Specialization Timeline](../../gpupro/images/warp_specialization_timeline.png)

その上にプレスペシャライゼーションのシングル warpgroup パターンがあります。同じ非専門化スレッドグループがロードパスと MMA パスの両方を所有するため、一方のエンジンがアクティブ中でも簡単にアイドル状態になります。ステップ 5 と 6 はダブルバッファリングと永続スケジューリングでベースラインを改善しますが、ロードと計算を独立した producer と consumer の役割に分割する段階はまだありません。下部では、専門化がその順番制を壊します。TMA producer は MMA の consumer が計算に忙しい間に次の tile をプリフェッチし、書き込みは自動的に進行します。producer warp 3 は次のロードを発行し、consumer warp 0 はまだ現在の MMA を処理中なので、どちらのエンジンも相手を待つ必要がありません。ロード/MMA のハンドオフは 2 つの barrier を使用します。

- `tma2mma` (TMA→MMA): ロードされた SMEM データが MMA が消費できる準備ができていることを示す信号です。
- `mma2tma` (MMA→TMA): MMA がバッファの読み込みを終えたことを知らせ、TMA が次の負荷のために再利用できるようにします。

図の一つの細部は最初は間違いに見えるかもしれません。 `mma2tma` 矢印が段階的に飛んでいくことです。その理由はリングバッファです。 `PIPE_DEPTH=2` では、ステージ 0 とステージ 1 の 2 つの SMEM バッファがあります。TMA ロード k=0 はバッファ 0 を埋め、TMA ロード k=1 はバッファ 1 を埋めます。MMA Compute k=0 がバッファ 0 の読み込みを終えると、バッファが空いていると `mma2tma` 信号を送りますが、実際にバッファ 0 を取り戻したいロードは TMA Load k=2 であり、バッファ 1 を使用している k=1 ではありません。だからこそ、MMA Compute の k=0 の `mma2tma` 矢印が TMA 負荷 k=2 まで伸びているのです。リリースは単にリングに 2 つのスロットがあるため、段階を飛ばします。

### warp の役割

タイムラインはなぜ作業を分担したのかを示していました。次の質問は、それぞれのパーツを誰が担当するかです。スペシャライゼーションは、3 つのジョブ(ロード、コンピュート、writeback)を特定の warp に割り当て、同時に実行できるようにします。 `WG_NUMBER=2` の場合、kernel は 2 つの warpgroup(役割表では WG と略される)を使用します。

| 俳優 | 所在地 | 職務 |
|-------|----------|-----|
|TMA producer| warpgroup 1、warp 3 | TMA を通じて A tile と B tile を継続的に読み込みます |
|MMA consumer| warpgroup 1、warp 0 | データが整い次第、MMA を実行 |
|Writeback| warpgroup 0(全 warp) | TMEM の結果を読み込み、GMEM に書き込みます |

### 4 つの barrier

3 人の同時アクターには 4 つの barrier が必要で、その 4 人はきれいに 2 つの反対方向に分類されます。フォワードパス(TMA → MMA → Writeback)はデータの *準備完了* を示します。メッセージは「あなたが待っていた tile がここにあります」です。後方パス(Writeback → MMA → TMA)はバッファの *リリース* を信号として「あなたが望んだスロットが再び空いています」と伝えます。命名規則が分かれば、名前は自然に読み取れるようになります。それぞれが「 `source2destination` 」であり、 `tma2mma` は TMA が MMA を示す barrier に過ぎません。

| barrier | 種類 | 演出 | 意味 |
|---------|------|-----------|---------|
|TMA2MMA| `TMABar` | TMA -> MMA | 「SMEM データが準備完了」 |
|MMA2TMA| `TCGen05Bar` | MMA -> TMA(ママ・マシック・ア・マシック・ア・マシオ)です | 「SMEM バッファは再利用可能」 |
|mma2ld| `TCGen05Bar` | MMA -> Writeback | 「TMEM の結果が出ました」 |
|LD2MMA| `MBarrier` | MMA ->書き返す | 「次の tile の TMEM は空いている」 |

なぜそれぞれの barrier はその *タイプ* を持っているのでしょうか?タイプは producer が完成を宣言する方法に由来します。TMA ロード `TMABar`、バイトカウント付きの mbarrier を使用します。転送のバイトが到達すると TMA ハードウェア自体が barrier に到達するため、consumer はスレッドポーリングなしでデータが準備完了であることを認識します。TMA ストアこれを利用できません(ストアには通知する人がいないため)、 `cp_async.bulk.commit_group()` + `wait_group(0)` にフォールバックし、発行スレッドは自分の書き込みがドレインされるのを待つだけです。MMA 操作 `TCGen05Bar` を用い、 `tcgen05.commit()` 命令が MMA 終了時に barrier を知らせます。

ここでの小さなディテールがステップ 8 で役立つでしょう。 `arrive` コールは、単一 CTA kernel では他の CTA が信号を送らないため、 `cta_mask=0` を通過します。ステップ 8 がクラスタを形成すると、この議論自体が非ゼロになり、協力する CTA を起動させる仕組みとなります。

### パイプライン州

4 つの barrier は、バッファが準備できたときに役割に伝えます。パイプラインのサイクル中、各ロールがどのバッファにいるかを追跡する何かがまだ必要です。その簿記こそが、 `PipelineState` が管理していることです。リングバッファは同時に 2 つの帳簿記を処理します: 現在どのスロットにいるか、そしてそのスロットの barrier のどの「phase」を待っているかです。両方をパイプラインループ上で手作業で追跡すると、1 つずつ誤りが生まれやすく、ここで 1 つだけ誤りが起きると kernel 全体がデッドロックされます。 `PipelineState` は両者をまとめるために存在し、以下のようなことをしなくて済むようにしています。

```python
# The producer starts ready at phase 1.
tma_ps = PipelineState(PIPE_DEPTH, phase=1)
# tma_ps.stage = current stage index
# tma_ps.phase = current phase (0 or 1)
tma_ps.advance()                          # Advance to next stage
```

最初の 1 `phase` が、その役割の最初の 1 `wait` を走らせるかブロックするかを決めるもので、正解はパイプの両端で反対の位置にある。これが人々をつまずかせる部分だ:
- `phase=1` (producer)->は最初の `wait(phase=1)` で barrier が相 0 のままであり、0 != 1 であるためすぐに通過。それこそが私たちの望むことです。なぜなら、バッファは空から始まり、producer はすぐにそれを埋め始められるはずだからです。

- `phase=0` (consumer)->最初の `wait(phase=0)` は相 0 で barrier を認識し、0 == 0 からをブロックします。これも私たちが求めているもので、まだデータがなく、consumer は producer が到着するまで読むものがないからです。

両端に同じ開始 phase を与えると膠着状態、あるいは最悪の場合はサイレント・コラプション(沈黙の汚染)になるので、この一つの選択を正しくする価値があります。

### `warpgroup_sync` barrier ID

専門化は同期の危険をもたらし、簡単に直面します。各 warpgroup が異なるコードパスを実行すると、馴染みのある `cta_sync()` はデッドロックします。ハードウェア barrier#0 を使い、*すべての* CTA スレッドが到達することを要求しますが、warpgroup ブランチ内ではそのうち一部のスレッドしか存在しません。代わりに必要なのは、単一の warpgroup に限定した barrier です。GPU は 16 個の名前付き barrier(ID 0–15)を与え、kernel は `warpgroup_sync(10)` に到達し、1 つの warpgroup 内のスレッドのみを同期させます。複数の warpgroup がそれぞれ独立して同期する必要がある場合、例えばマルチ consumer の Step 9 のように、 `warpgroup_sync(wg_id + 10)` を通じて異なる ID を取得し、同じハードウェア barrier で衝突しないようにします。

実装。

ここでは `PIPE_DEPTH=2` を使い、負荷と計算が重なり合う最小の深さです。より深く掘り下げると、SMEM 予算の限界までメモリ遅延が隠れます。以下の「ステップ 7 の誤った行動」についての議論では、そのトレードオフについて詳しく解説しています。すべての要素(役割、四つの barrier、 `PipelineState`、warpgroup による同期)が揃ったことで、完全な核を組み立てることができます:

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

これらの kernel のいずれかを実行するには、ステップ 1([Tiled GEMM の構築](/ja/gpupro/tiled-gemm/))で示した同じコンパイル/実行/チェックハーネスを再利用してください。 `hgemm_v1` を `hgemm_v7`、 `hgemm_v8`、または `hgemm_v9` に置き換え、問題サイズを `M=N=K=4096` のように選びます。cluster の `N` `M` ステップはクラスタ tile の倍数である必要があることを覚えておいてください(ステップ 8 は `256×256`、ステップ 9 は `512×256`)ので、小さな `128×128` サイズでは tile は全く生成されません。新しい Python セッションごとに 1 ステップコンパイルし、ステップを切り替える前に kernel を再起動してください。kernel は内部名を再利用し、コンパイラはセッションごとに状態を保持します。各ステップのタイミングは以下の「エンドツーエンド結果」に集計されています。

### epilogue(Writeback)詳細

ステップ 7 は心地よいシンプルな epilogue にできます。 `BLK_N=128` 列のみの場合、書き込み warpgroup は TMEM tile 全体を一度にレジスタに読み込み、その後 1 つの TMA ストアを発行します。ステップ 8 と 9 ではこの余裕がなく、まさに後で追加するチャンク化が導入される理由です。しかし現時点でのシーケンスは次の通りです:

1. MMA を待つ: `mma2ld.wait(phase)`。このチュートリアルのステップ 8 と 9 は、保守的な特典としてここに 1 `fence.after_thread_sync()` を追加します。MMA 完備 mbarrier はすでに順序をカバーしており、ほとんどの kernel(CUTLASS を含む)はこれを省略しているため、ステップ 7 も含まれます。
2. TMEM の->レジスタを読み取ります(スレッドあたり 128 fp32、warpgroupscope `Tx.copy_async(reg_wg, tmem[:, :BLK_N])` 経由、その後 `T.ptx.tcgen05.wait.ld()`)。
3. シグナル MMA: `ld2mma.arrive(0, cta_id=0, pred=True)` (128 スレッド全て到着);TMEM は次の tile で無料になりました。2 つのクワーグ `arrive` クラスタステップで繰り返し現れます。 `cta_id` はどの CTA の barrier のコピーを信号に割り当てるかを指定します(`0` = この CTA、ローカル barrier;ステップ 8 では協力者が `cta_mask` 経由でターゲット CTA-0 に到達します)、 `pred` はスレッドごとの述語で、このスレッドが実際に到着するかどうかをゲートします(`True`、すべての writeback スレッドが到着総数にカウントされます)。
4. fp32 -> f16 をレジスタでキャストします。
5. Dsmem ->レジスタを書き、フラッシュ `fence.proxy_async("shared::cta") + warpgroup_sync(10)` します。
6. TMA ストアの Dsmem -> `cp_async.bulk.commit_group() + wait_group(0)` 経由の GMEM です。

ステップ 8(`BLK_N=256` 時)とステップ 9(consumer あたり `MMA_N=256` 回)はこのワンパス形式を維持できず、その理由はレジスタ圧力です。スレッドあたり 256 FP32 値を読み取ると、各スレッドのレジスタに同時に 256 × 4 = 1024 バイトが存在しなければならず、これによりローカルメモリに流れ出すリスクがあり、さらに大きな Dsmem バッファを強制します。これらのステップは書き込みを `EPI_N` 列のチャンク(`EPI_N=64`)に分割します。各イテレーションは `EPI_N` つの fp32 レジスタだけを稼働させ、それに応じて小さな TMA ストアを発行し、いくつかのストア命令を多く交換して、快適なレジスタ予算を保ちます。

実装ノート。

- 永続 kernel: 1 つの SM につき 1 つの CTA `bx = T.cta_id([SM_COUNT])` ---、tile をループします

- L2 に適したスケジューリング: `ClusterPersistentScheduler2D` キャッシュ局所性のための tile 順序

- この warp specialization とソフトウェアパイプライン------パターンは、CUTLASS ス tile 設計を含む高性能 GEMM kernel で一般的です。

### ステップ 7 が悪さをするとき

ステップ 7 は、TMA ロード、 `tcgen05` MMA、書き込みが同時に進行する最初の GEMM kernel です。同じ故障パターンはステップ 8 と 9 でも現れます: barrier の数が不一致、役割ガードが間違った場所にいる、フェンスが欠けている、TMA ストアが空く前に再利用されたステージングバッファなどです。これらのケースのデバッグチェックリストは[Warp-Specialized Kernel のデバッグ](/ja/gpupro/debugging-warp-specialized-kernels/)にまとめられています。

パイプライン深度調整。Step 7 の kernel は最低 `PIPE_DEPTH=2` で動作します。4 や 6 に上げると、TMA producer は MMA consumer より先行し、より多くのメモリレイテンシを隠せますが、それはより多くの SMEM を費やすことで実現し、SMEM は有限です。B200 は 1 つの SM あたり 228KB を提供します。 `BLK_M=BLK_N=128, BLK_K=64, fp16` では、パイプライン各段階 A と B の合計で `(128*64 + 128*64) * 2 = 32 KB` コストがかかり、 `Dsmem` の writeback ステージングバッファはさらに 32KB を追加します。これにより `PIPE_DEPTH=4` は約 160KB、 `PIPE_DEPTH=6` は約 224KB で、予算にぴったりです。それ以上深く掘り下げるには、書き込みの段階戦略を再考する必要があります。

---

warp specialization は一つの CTA の協力関係を得た。次のステップでは、CTA 自体の境界を越えた協力を広げ、2 人を 1 つの大きな tile で作業させます。


## ステップ 8:2-CTA cluster

ステップ 7 ではエンジンが重なりましたが、各 CTA は依然として自分の 128×128 tile を単独で計算しておらず、隣接するものが借りられない operand を再読み込みしていました。ステップ 8 はその孤立を打破します。2 つの CTA がクラスタに結合し、互いの共有メモリにアクセスできるようになるため、1 つの協力 `tcgen05` MMA は 256×256 tile を生成し、両者をまたぐものを生成し、B の 1 回の負荷は MMA の作業量を 2 倍に供給します。前述と同様に、M=N=K=4096。

> このステップで変更されるもの: scope + layout + dispatch
> - scope: 協力 scope はクラスタ内の 2 つの CTA にまたがるものとなり、1 つではなく。
> - layout: operand tile は 2 つの CTA の SMEM に分割されます。CTA 0 は共有完結 barrier(`remote_view`)を所有しています。
> - dispatch: MMA は `cta_group` / `cta_mask` を得るため、 `tcgen05` 2-CTA 協同作戦として動作します。

トピック。

- CTA cluster: 大きな tile 上で複数の CTA が協力する

- `map_shared_rank` を経た CTA 間 SMEM アクセス

- `cta_group=2` 256x256 の cluster tile 上の協力型 MMA 用です

- クロス CTA barrier 信号 `cta_mask`


### cluster tile 形状

最適化は単一のハードウェア能力に依存しています。 `cta_group=2` を使えば、MMA は自分が存在する CTA だけでなく、両方の CTA によってステージ化された operand tile を読み取ることができます。各 CTA は保存された B の 128 行スライスを 1 つ読み込み、転置後は 128 の論理出力列となり、協調型 MMA は 2 つのスライスを 1 つの operand に再びつなぎ合わせます。下図は、2 つの CTA の A スライスと B スライスがどのように結合して単一の 256×256 cluster tile に形成されるかを示しています。

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo/cta_cluster.html" title="A 2-CTA cluster: cooperative MMA via cross-CTA SMEM read" loading="lazy"
        style="width:100%; min-width:720px; height:580px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>

*インタラクティブ: 各 CTA は A 行スライス 1 つと保存 B 行スライス 1 つを所有し、cluster 全体で他の CTA の保存 B スライスを読み込みます(DSMEM)。 `B.T` の後、2 つのストアド B スライスが出力カラム全体をカバーするため、ペアは 256×256 の出力 tile を 1 つ生成します。*

なぜ A と B がクラスタに分割されているのか: 256×256 tile がどのように分割されるかを見るには、チュートリアルでは GEMM が `D = A @ B.T` として保存され、B は形状 `N x K` で保存されていることを思い出してください。2 つの CTA が一つの cluster にいる場合、分割はきれいに抜け出します。

- A は縦に分割されています: CTA-0 は A0(0-127 行目)、CTA-1 は A1(128-255 行目)を保持します。スタック: `[A0; A1]` (256 行)。
- Stored B は行で分割されます: CTA-0 は B 行 0-127 を読み込み、CTA-1 は B 行 128-255 を読み込みます。数学では `B.T` が使われるため、その 2 つの行スライスは論理右 operand の 128 列の 2 つのスライスになります。
- `cta_group=2` では、MMA ハードウェアはクロス CTA 共有メモリアクセスを介して CTA の SMEM から B を読み込み、論理出力カラム全体を認識します。
- 結果として、2 つの CTA は 1 つの 256x256 出力 tile 上で協力します。各 CTA はその tile の 128x256 行のストライプを書き込みます。

なぜこれが単なる業務の再編成ではなく、本当の勝利なのかを考える価値があります。各 CTA は依然として A の 128×K と B の 128×K のみを読み込むため、クラスタ全体は単一の CTA の operand を約 2×段階化しますが、それでも 256×256 tile を生成し、これは 128×128 tile の約 4×の出力 FLOP を運びます。したがって、MMA は各 CTA の B スライスが協調型 MMA を通じて他の CTA の A スライスに対して再利用されるため、ステージド operand バイトあたり約 2 倍の作業量を行います。言い換えれば、算術強度はほぼ倍増し、これこそがメモリ重視の kernel に必要なレバーです。つまり、エンドツーエンドテーブルの~2.2×の高速化は、同じバイトをより多くの数学に入力することで得られます。

### tile アドレス計算

cluster が作業単位になった今、tile スケジューラーもクラスタ tile 数でカウントしなければなりません。各 `(m_idx, n_idx)` 返すには 256×256 の完全な領域名が付けられ、cluster 内の 2 つの CTA がその領域を分割しています。クラスタ座標を各キャラが実際に読み込む CTA スライスに変換するのは次のようになります。

```python
m_st = (m_idx * CTA_GROUP + cbx) * BLK_M
n_st = (n_idx * CTA_GROUP + cbx) * BLK_N
```

両方の CTA は同じ 256×256 のクラスタ tile で動作し、単一の座標 `cbx` (クラスタ内の CTA の位置、0 または 1 のいずれか)がこの CTA の両軸寄与を識別します。 `m_st` はこの CTA が所有する出力行ストライプを選択し `n_st`、協調型 MMA に入力する保存済み B スライスを選択し、その後、書き込みは 256 列の出力スパンのうち 128 列の半分の両方を出力します。また、 `num_m_tiles = M // 256` と `num_n_tiles = N // 256` は cluster tile をカウントしており、個々の CTA tile をカウントしないことに注意してください。

一見すると、 `m_st` と `n_st` の両方に `cbx` が現れ、行のずれが列に漏れ出したかのようですが、どちらの使い方も正しく、なぜそうなのかを解き明かす価値があります。書き込みパス上では、 `cbx` は M 軸のみに属します。各 CTA はそれぞれ 128 行のストライプ(`m_st = (m_idx * CTA_GROUP + cbx) * BLK_M`、CTA-0 は次の 128 行を書き込むので、CTA-0 は `m_idx*256 .. +128` 行、CTA-1 は次の 128 行を書き込みます)。しかし両方の CTA はクラスタ tile の *完全な* 出力列 256 列を書き込んでいます。だからこそストアはクラスタの `n_idx` 列(`n_st_epi = n_idx * 256 + no * 128`、 `cbx` は見えません)から列を導き出します。これは、CTA ごとのご `n_st` からではなく、 `n_st` がこの `cbx` を搭載している理由は、各 CTA が異なる保存 B 行スライスを MMA に読み込むためです。そこで `cbx` は *load* オフセットであり、CTA の出力カラムオフセットではありません。

### ステップ 7 からのコード変更

ステップ 7 に対する差分は 6 つの編集があり、それぞれが先ほど説明した cluster 契約の 1 つの部分を符号化しています:

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


### cluster・scope の変更

これら 6 つの編集はすべて同じ変化から生じています。協力範囲が単一の CTA ではなく cluster に変わったのです。以下のポイントは、その拡大が実際に何を意味するのかを明確に示しています。すなわち、各 CTA がどのように自分の居場所を見つけ、どの barrier がクラスタに位置づけられているか、そしてどの CTA が実際に協力的な MMA を発行しているかです。

- clusterCTA ID: `cbx` は各 CTA にクラスタ内での位置(0 または 1)を伝えます。CTA-0 は A 行 0-127、CTA-1 は 128-255 行を扱います。

- リモート barrier ビュー: cluster では、各 CTA が独自の SMEM と barrier を持ちます。ここで明らかな疑問が生じます。CTA-1 が CTA-0 の生成を待つ必要がある場合、実際にどの barrier に触れているのか?答えは、CTA-0 の barrier を単一の調整点として指定し、cluster 内の任意の CTA がそこに到達できるようにすることです。 `map_shared_rank(tma2mma.ptr_to([0]), 0)` は TIRx ラッパー `tma2mma.remote_view(0)` で CTA-0 の barrier へのクラスタ全体のポインタを返し、以降はすべての到達と待機が CTA-0 のコピーを対象とします。

- CTA-0 の MMA ディスパッス: `cta_group=2` を 2 つのエンジンを並列に点火していると読みたがりますが、実際はそうではありません。CTA-0 は正確に 1 `tcgen05.mma` を発行し、ハードウェアは両方の CTA にまたがる *単一の協調型* MMA を駆動し、両方の SM の SMEM から operand を読み込み、両方の SM の TMEM 上で accumulator を書き込みます。CTA-1 は MMA を全く使いません。(各 SM は `tcgen05` エンジンが 1 基のみなので、 `cta_group=2` 1 つのクロス SM MMA であり、2 つのエンジンが並んで走っているわけではありません。)だからこそコードは MMA を守り、 `if cbx == 0:`。

- マルチキャスト到着: `tcgen05.commit(..., cta_group=2, cta_mask=3)` は CTA-0 のみが発行しますが、両方の CTA の barrier に信号を送ります。 `cta_mask=3` (バイナリ `11`)は CTA-0 と CTA-1 の両方を標的にすることを意味します。

- ld2mma init count: `init(128 * CTA_GROUP)` --- 両方の CTA の writeback warpgroup(それぞれ 128 スレッド)が到着します。


実装。

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

2 つの CTA で何が変わるのか。

- `CTA_GROUP = 2`、 `MMA_N = BLK_N * CTA_GROUP = 256`

- `ld2mma.init(128 * CTA_GROUP)` ---両方の CTA の書き込み WG が到着します

- TMA の到着バイトカウントには両方の CTA が含まれます: `CTA_GROUP * (BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE`

- `tcgen05.alloc` と `tcgen05.dealloc` は `cta_group=2` を使わなければなりません。

- writeback は 256 列の出力を 128 列の 2 つのチャンクに分割---256 列すべての TMEM 列を同時に読み込むとレジスタ容量を超えます。ステップ 9 ではチャンクをさらに縮小 `EPI_N=64`

- `cluster_sync()` は最後に `cta_sync()` を置き換えます(すべての CTA が TMEM のディールロック前に完了していることを保証します)。

その余分な算術強度は壁時計に直接反映されます。ステップ 8 は 4096³で 0.104ms に達し、同じサイズの Step-1 アルゴリズムの 70ms より約 676×です(End-to-End 表を参照)。kernel は現在、計算に縛られる方向に傾いており、それがステップ 9 の準備を進めています。そこでは 2 つ目の MMA consumer を追加し、さらに多くの tensor コア作業を継続させています。

ステップ 8 がステップ 7 より *遅く* 出る場合、ほぼ間違いなく新しい cluster 契約のどれかが少し間違って入力されています。まず確認すべきことは 3 つあります。TMA の到達バイト数が `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE` であること。256×256 のクラスタ tile のスケジューラ寸法が `num_m_tiles=M//256, num_n_tiles=N//256` であること、そして、その書き込みは 128 列のチャンクごとに 1 つずつ 2 つの TMA ストアを発行し、それぞれが Dsmem を再利用する前に排水されます。

---

cluster は CTA を横断して再利用を上げました。最後のステップは内側に向かい、各 CTA 内での計算密度を上げます。これは、producer に 2 人目の MMA consumer を与え、供給し続けることです。


## ステップ 9: マルチ consumer warp specialization

ステップ 8 には MMA は本当に忙しいですが、1 つの consumer warp がステージ化された B tile を一つだけ速く処理でき、その B tile は SMEM にずっと置かれていて、読めば誰でも利用可能です。最終最適化はそれを活かし、異なる A ブロックを同じ B tile に掛ける 2 つ目の MMA consumer を追加します。CTA あたりの計算密度は倍増し、cluster 出力は 256×256 から 512×256 に増加します。前述と同様に、M=N=K=4096。

> このステップで変更される点: scope+layout
> - 範囲: 1 人の MMA consumer が 2 人になり、 `warp_id` によって選ばれます。
> - layout: 1 つの段階化された B tile が両方の consumer によって再利用されます。A は consumer 軸を獲得する。
> - dispatch: 変更なし。

トピック。

- 複数の MMA warp(consumer)による高スループット

- 独立した barrier スロットを持つ複数の書き込み warpgroup

- このチュートリアルで最も最適化された GEMM barrier 構成が使用している構造


### マルチ consumer 構造

2 つ目の consumer を追加することで、kernel はより明確な役割を割り当てることになります。例えば、MMA warp を 1 つから 2 つ、そして余分な accumulator を消耗させる 2 つ目の writeback warpgroup が対応するということです。 `NUM_CONSUMER=2` と `WG_NUMBER=3` を用いることで、kernel は 3 つの warpgroup(役割表では WG と略される)をまたいます。

| warpgroup | warp | 役割 |
|-----------|------|------|
|WG 2| warp 0 | MMA consumer 0: `Asmem[..., 0] x B` -> TMEM のコルズ `[0:256]` |
|WG 2| warp 1 | MMA consumer 1: `Asmem[..., 1] x B` -> TMEM のコルズ `[256:512]` |
|WG 2| warp 3 | TMA producer: 各ステージにつき 2 つの A ブロック+1 個の B ブロックをロードします |
|WG 0| 全員 | consumer 向け書き込み 0: TMEM を読み込む `[0:256]` |
|WG 1| 全員 | consumer1 の書き込み: TMEM を読み込む `[256:512]` |

全体の配置は一つの非対称性に依存しています。各 consumer は自分の A ブロックを同じ段階の B tile に掛け合わせるため、単一の B ロードが MMA の 2×を供給し、B の有用 FLOP あたりのロードコストは実質的に半分になります。B を共有して A を使わない理由は、両者の M 行のストライプが異なるためです。A ブロックは本当に異なるデータであり、B は両方で同一です。演習 3 は、これが唯一効果的な共有だと自分に納得させることを求めます。

### ステップ 8 からの変更点

具体的には、セカンド consumer のサポートは kernel のいくつかの箇所に関わり、すべての変更は一つの事実に結びつきます。つまり、各ステージごとに 2 つの A ブロックと 2 つの TMEM レンジを送り込み、2 つの TMEM レンジを供給・排水し、B は共有のままであるということです。以下の編集では、追加の A ブロックを 1 つ、各 consumer に専用の barrier スロットを与え、より高い 512×256 cluster tile の tile アドレスを調整します。

- `Asmem = pool.alloc((PIPE_DEPTH, NUM_CONSUMER, BLK_M, BLK_K), ...)` --- 各ステージにつき 2 つの A ブロック、consumer 1 つあたり 1 つ

- TMA は `Asmem[stage, 0]` と `Asmem[stage, 1]` の両方を読み込み、TMA の到達バイトは `CTA_GROUP * (NUM_CONSUMER * BLK_M * BLK_K + BLK_N * BLK_K) * F16_SIZE` (追加で A ブロック)になっています。

- MMA warp `warp_id` は A ブロックと TMEM 範囲を選択します

- `mma2tma.init(NUM_CONSUMER)` ---両 consumer が各段階ごとに TMA を信号します

- `mma2ld` と `ld2mma` は、各 consumer が自分の barrier スロット(MMA 側用 `warp_id`、writeback 側用 `wg_id`)を `depth=NUM_CONSUMER` ---しています。

- tile アドレス: `m_st = (m_idx * NUM_CONSUMER * CTA_GROUP + cbx) * BLK_M` ---M 方向は、各クラスタ tile が M の consumer にまた `NUM_CONSUMER` がるため追加の `NUM_CONSUMER` 要素があります。tile スケジューラは `num_m_tiles = M // 256 // NUM_CONSUMER` を使用します(クラスタ tile は 512x256 です)。

- writeback はチャンクされた `EPI_N` を使うため、各反復ごとにレジスタに残る TMEM リードバックの値を減らします


実装。

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

実装ノート。

- このステップ 9 設計では、 `mma2ld` と `ld2mma` それぞれが個別の consumer ごとのオブジェクトではなく、 `depth=NUM_CONSUMER` と共有された単一のオブジェクトとして扱われています。スロット 0 は MMA warp 0 と warpgroup 0 を接続し、スロット 1 は MMA warp 1 と warpgroup 1 を接続します。MMA 側は `warp_id` でインデックスされ、writeback 側は `wg_id` でインデックスされます。

## エンドツーエンドの結果

以下の表は、素朴なベースラインから warp 専用クラスタ kernel までの測定されたマイルストーンと、cuBLAS 参照を示しています。NVIDIA B200 の参照番号、M=N=K=4096、fp16、ロックされたクロック、1000 回のタイマーキングベンチマーク:

| ステップ | 技術 | 時間 | 加速 |
|------|-----------|------|---------|
| 1 | シンクロロード+MMA | 70 ミリ秒 | 1× |
| 2 | K ループの蓄積 | --- | ハンドル K は 1 tile より大きい |
| 3 | 空間 tile | 53.6 ミリ秒 | ~1.3× |
| 4 | TMA 非同期負荷 | 0.49 ミリ秒 | ~142× |
| 5 | ソフトウェアパイプライン | --- | オーバーラップ負荷+計算 |
| 6 | 永続 kernel | --- | L2 キャッシュ局所性 |
| 7 | warp の専門分野 | 0.23 ミリ秒 | ~309× |
| 8 | 2-CTA cluster | 0.104 ミリ秒 | ~676× |
| 9 | マルチ consumer | 0.094 ms | ~744× |
| --- | cuBLAS(参考文献) | 0.094 ms | ~744× |

この表では、ステップ 1 の基準を含めて 70ms のすべてが同じ M=N=K=4096 のサイズで測定されており、これがスピードアップチェーンを端から端まで比較可能な理由です。その 70 ミリ秒が実際に何を意味するのかを正確に把握する価値があります。誤解されやすいからです。これは 4096³で実行される[Tiled GEMM の構築](/ja/gpupro/tiled-gemm/)単一 tile のステップ 1 kernel ではありません。その kernel は常に 128×128 tile を 1 つしか計算せず、小さなサイズでしか動作しません。70ms は、同じ順序の単一 tile のアプローチを取り入れ、4096³³の全問題にスケールアップした素朴なフルサイズベースラインです。[Tiled GEMM の構築](/ja/gpupro/tiled-gemm/)では、最初のウォークスルーをシンプルにするために、128×128256³)でステップ 1 から 3 が導入されます。ここでの Step 1 と Step 3 の行は、実物大のベンチマーク版です。残りのダッシュ(ステップ 2、5、6)は構造上のステップを示していますが、それ自体は時間が計測されていません。

これらの数字は、リーダーボードのエントリーではなく、管理された条件下での単一の B200 リファレンスランとして読み取ってください。各ステップに埋め込まれた `{.python .input}` ベンチマークセルはスモークベンチマークであり、ピークパフォーマンスを主張するのではなく傾向を見抜くのに適しています。

4 つの技術がほぼすべての利益を担っています:

1. TMA 非同期データ移動: ハードウェアコピーエンジンがソフトウェアコピー(ステップ 1→ステップ 4 からの~142×に代わります。この 142×を正しく読むことが重要です。これは単一の 128×128 tile kernel(grid 1×1)から、K ループ、空間 tile、多数の CTA を持つ完全な tile 並列 kernel(TMA とともに)へと移行したことを反映しています。それは TMA の単独の貢献ではありません。TMA を分離するということは、コピー機構だけが異なる 2 つのフルサイズ kernel を比較することを意味します。
2. ソフトウェアパイプライン化 + Warp 専門化: 負荷と計算を重ね、それぞれに専用の役割を与えます(ステップ 4→ステップ 7 から~2.2×。
3. CTA cluster: 2-SM 協同型 MMA は、このベンチマークのステップ 7→ステップ 8 から CTA 間の B tile 再利用(~2.2×を向上させる。
4. マルチ consumer: より高い計算密度を実現するための 2 つの MMA warp(ステップ 8 からステップ 9 から~10%→増加)。

測定されたマイルストーンにプロットすると、これら 4 つの寄与は同期 tile 核から cuBLAS 基準への下降をたどります。下図は選ばれた測定点を示しています:

![GEMM Optimization Journey](../../gpupro/images/gemm_perf.png)

リストを進めるにつれて利益が縮小していることに注目してください。これは努力の弱さではなく構造的な理由によるものです。初期のステップは *メモリ* のボトルネック(TMA はソフトウェアコピーを置き換え、cluster は算術の強度を高める)を追い、実際に 70 ミリ秒の大部分がそこに費やされているため、そのステップが最も効果的です。ステップ 8 の時点で kernel はすでに cuBLAS の約 10% 以内(0.104 ms vs 0.094 ms)に入り、*計算バウンド* に近づいているため、隠すべきメモリストールはほとんど残っていません。Step 9 のマルチ consumer オーバーラップは、わずかに残ったもののほとんどを回復します。最終的な利得は約 10% で、計算天井付近で予想されるもので、ほぼ解決した問題の逓減であり、弱い最適化の兆候ではありません。

この章で築いたもの(TMA ロード、 `tcgen05` MMA、TMEM リードバック、warp 専用 barrier)は次の章に直接引き継がれます。Flash Attention はそれらをすべて使い回し、さらに単に 1 つの phase を繰り返すのではなく、2 つの MMA phase の間にオンライン softmax のステップを挟み込むことで難易度を上げています。


## 演習

1. ステップ 7 で TMA と MMA の両方の `PipelineState` で初期 `phase` を `0` に設定したらどうなりますか?膠着状態のシナリオを描く。
2. Step 8 の `cta_group=2` では、TMA の到達バイト数は `CTA_GROUP * (BLK_M*BLK_K + BLK_N*BLK_K) * F16_SIZE` になります。各 CTA が独自のデータを読み込むのに、なぜ 1 `CTA_GROUP` を掛けるのでしょうか?
3. ステップ 9 では、各 consumer が異なる M 行を扱いますが、同じ B tile を扱います。なぜ B(A ではなく)を共有するのが正しい選択なのでしょうか?

エージェントで試してみてください: Step 7 の kernel を貼り付け、4 つの barrier(`tma2mma`、 `mma2tma`、 `mma2ld`、 `ld2mma`)を 1K tile をトレースするよう指示します。それぞれについて、誰が待っているか、誰が到着するか、どの tile が安全に読めるか、どのバッファが再利用可能になるかを尋ねます。
