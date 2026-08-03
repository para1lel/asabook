---
title: "TMA による GEMM のパイプライン化"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/pipelined-gemm/
pageClass: gpupro-page
---

::: info 概要
- 基本的な GEMM はコピーと計算を順番に実行するため、両者を並行実行できる時間を無駄にしています。
- ステップ 4 は TMA 非同期ロードに切り替わり、ステップ 5 は SMEM とプリフェッチをダブルバッファで行います(PIPE_DEPTH=2)。ステップ 7 で warp specialization によりフルロード/コンピュートの重複が実現し、ステップ 6 では kernel を tile スケジューラで永続化します。
- 目標は、Tensor Core が現在の tile を処理している間に次の tile をロードすることです。
:::

Tensor Core はチップ上で最も高価な計算ユニットですが、前章の正しい tiled GEMM では大半の時間を待機に費やします。スレッドが tile を shared memory にコピーし、Tensor Core が計算し、次の tile をコピーする間はまた Tensor Core が待つ、という逐次実行だからです。ところが、次の tile のロードと現在の tile の計算は別のハードウェアを使うため、本来は並行して実行できます。tile、layout、計算内容はすでに正しいので、変更すべきなのは作業のタイミングと担当です。本章ではデータパスを保ったまま、この待ち時間を取り除きます。

この目標には 3 段階で到達します。ステップ 4 では GMEM <-> SMEM の一括転送を TMA に任せ、専用ハードウェアで tile を搬送します。ステップ 5 では 2 段のソフトウェアパイプラインを追加し、現在の K tile を計算している間に次の tile を準備します。ステップ 6 では tile スケジューラが駆動する persistent kernel に変更し、tile ごとの初期化コストを分散しながら、operand の局所性を保つ順序を選べるようにします。SMEM、TMEM、レジスタの layout は前章から変わりません。新しい要点は、各ハードウェアエンジンを同期的に足並みをそろえて進めるのではなく、非同期に引き継ぐことです。

## ステップ 4: TMA 非同期ロード

最初のステップは、コピー自体をクリティカルパスから外すことです。CTA がステップ 1〜3 で何をしていたか考えてみてください。すべてのスレッドが、tile を SMEM にシャトルする以外に目的もなくアドレスを計算し、ロード命令を発行しています。つまり、配管に使われる命令帯域幅が数学に使われていないということです。ステップ 4 では同期 `Tx.copy` を TMA に置き換え、単一のスレッドが 1 つのコマンドを出し、TMA エンジンが単独で tile 転送全体を実行します。ここから先は、ステップ 1-3 の小さなサイズではなく、M=N=K=4096 のフルサイズで実行され、その端から端までのタイミングは[Warp Specialization と Cluster による GEMM のスケーリング](/ja/gpupro/warp-specialized-gemm/)の末尾にある *端から端までの結果* 表に表示されます。

> このステップで変わるもの: dispatch
> - scope: 変更なし、warpgroup 1 つ。
> - layout: 変更せず、同じ SMEM/TMEM/レジスタ tile。
> - dispatch: GMEM→SMEM ロードが同期 `Tx.copy` から TMA エンジンへ移動します。

### TMA 発行パターン

ステップ 4 の変更点は、同期 tile コピーを TMA ロードに置き換えることなので、そのロードがどのように発行されているかをよく確認することが重要です。ソースへの編集は数行だけですが、その背後にある実行モデルは本質的に異なります。同期 `Tx.copy` は、CTA スレッド自身が独自の命令で行う作業のことです。TMA コピーとは、1 つのスレッドが出した後、TMA ハードウェアがすべての移動を行うコマンドのことです。両者を並べて見る価値は十分にあります。

前(ステップ 3): 128 スレッドすべてがコピーに参加し、その後 `cta_sync` 共有メモリ書き込みを可視化します:
```python
# All 128 threads participate.
Tx.cta.copy(
  Asmem[:, :],
  A[m_st : m_st + BLK_M, i * BLK_K : (i + 1) * BLK_K],
)
Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])
T.cuda.cta_sync()
```

(ステップ 4)後: 1 スレッドが TMA ロードを発行し、mbarrier はハードウェア転送完了時に追跡します:
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

負荷は `elect_sync()` ではなく `tid == 0` でゲートされており、その区別は見た目以上に重要です。 `elect.sync` warp ごとに 1 つのアクティブレーンを選び、warpgroup は 4 つの warp を持つので、実際には 4 つのスレッドがロードプロトコルに入ることになります `elect_sync()`。問題は、プロトコルが mbarrier に期待されるバイト数を通知し、それを正確に一度だけ発表しなければならないことです。4 回のアナウンスでカウントが歪み、待ち時間が正しくリリースされなくなります。warpgroup 全体の ID で正確に一つのスレッドを選ぶのが、それを回避するクリーンな方法です。

スピードアップの出所を正直に伝えることが大切です。ステップ 4 は TMA ロードのたびに待機するため、まだ負荷と計算が重なっていません。それがステップ 5 の仕事です。ここでの勝利は、データ移動の経路の変更にあくまで起因しています。

- `Tx.copy` CTA スレッドを使ってアドレスの計算やロード/ストア命令の発行を行います。
- TMA はハードウェア tile 転送を開始するために発行されたコマンドを 1 回使います。アドレス生成、凝集、スウィズリングは TMA ディスクリプタで記述され、TMA エンジンによって実行されます。

つまり、ステップ 4 は各ロードでブロックされますが、結局はより速く処理されます。TMA は大量転送を吸収し、CTA スレッドが tile の入れ替えに命令帯域幅を使う手間を省きます。その節約だけでも状況が進みます。

### TMA ロードおよびストア同期

TMA のコピーがどのように発行されるかは見てきました。物語のもう半分は、いつ終わったかを知ることです。TMA に切り替えると同時に 2 つのことが変わります。コピーを始めるのは誰か、そしてコードがいつ終了するかということです。一つ目はコードから明らかです。2 つ目は見落としやすく、間違えるとクラッシュではなく静かな正確性バグが発生します。 `Tx.cta.copy` では、CTA スレッドが一緒にコピーを行い、フォロー `cta_sync()` があれば完成が確認できます。TMA では、選択されたスレッド 1 `Tx.copy_async(..., dispatch="tma")` で発生し、エンジンは独自のスケジュールで転送を行い、mbarrier を通じて完了を信号します。

だからこそ、 `cta_sync()` はもはや十分ではないのです。 `cta_sync()` は CTA 自身のスレッドのみを待ち、共有メモリの書き込みのみを命令します。飛行中の TMA 転送については何も知らないので、tile が到着している間に喜んで戻ってきます。修正方法は完了を明示的にすることです: TMA ロードの場合、選択されたスレッドがまず mbarrier に期待されるバイト数を伝え、その後 CTA はその mbarrier で待ってから MMA が SMEM tile に触れます。下の図はその握手を端から端までたどっています。

![TMA Async Load: Synchronization Flow](../../gpupro/images/tma_sync_flow.svg)

上の図はロードサイドのハンドシェイクを分離しています。選択されたスレッドのうち 1 つが TMA を起動し、mbarrier が期待されるバイトをカウントし、MMA は SMEM を読み取る前にリリースを待ちます。「Elected Thread」と書かれている場合、それは TMA を開始する選択されたスレッドを意味し、私たちのコードでは `tid == 0` スレッドであり、 `elect_sync()` レーンではありません。

ロードパスをまとめると、選択されたスレッドは両方の `copy_async` 呼び出しを発行し、その後に `arrive.expect_tx(total_bytes)` を続けます。ここでバイト数は mbarrier が保持すべきデータ量を正確に示します。エンジンがその数のバイトを移動すると、マッチング `mbarrier.try_wait(phase)` が解放され、その時初めて SMEM tile は MMA に安全に送ることができます。

ストア側は同じハードウェア上で移動しますが待ち方が異なるため、両プロトコルを頭の中で明確に分けておくことが重要です。ロードは mbarriers やバイト数で完了を追跡し、ストアはコミットグループと待機グループで追跡します。スレッドが FP16 の結果を `Dsmem` に書き込み同期した後、選択されたスレッドが `Tx.copy_async(D[...], Dsmem, dispatch="tma")` 開始し、その後 `cp_async.bulk.commit_group()` ブロック `cp_async.bulk.wait_group(0)` ストアが消耗するまで続きます。その待ち時間は任意ではなく、前のストアが消えるまで次の tile に再利用できません `Dsmem`。

エージェントで試してみてください: ステップ 4 のロード&ストア同期を 1K tile ごとにトレースします。各 TMA コマンドを開始するスレッド、完了を追跡する mbarrier やコミットグループ、MMA の `Asmem` および `Bsmem` の読み取りを保護する待機、 `Dsmem` の再利用を保護する待機を特定します。なぜ `elect_sync()` ここで TMA ロードプロトコルのスレッド選択が間違っているのでしょうか?

### 完全 kernel

完全な kernel は TMA のロードとストアをステップ 3 構造に折りたたみ、残りの構造はそのままにします。インポートは以前と同じです:

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

`hgemm_v4(M, N, K)` でラップされており、このパターンを全編にわたって従います。ラッパーは形状依存の定数や layout を、それらを使う kernel のすぐ隣に保持します。

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

### kernel における TMA 構成

その kernel 内のほとんどすべてはステップ 3 から引き継がれます。実際に TMA の意味を持つ構成ポイントは 5 つだけで、それぞれの名前を知っておく価値があります。

- TMA 設定: `{"dispatch": "tma", "cta_group": 1, "mbar": tma_bar.ptr_to([0])}` は TMA を使い、ロード完了を `tma_bar` 経由で報告するよう指示 `Tx.copy_async`。

- バイトカウント: `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` は 2 つの FP16 operand tile によって読み込まれるバイト数です。 `arrive.expect_tx(...)` このカウントを mbarrier に与えます。

- mbarrier 初期化: `init(tma_bar.ptr_to([0]), 1)` TMA 負荷で使用される完了 barrier を作り出します。

- `@T.inline`: `tma_load(...)` と `mma(...)` は補助機能です。これらはコンパイル時に kernel ボディに展開され、周囲の kernel の変数を使用できます。

- TMA ストア同期: epilogue はまず fp16 行を `Dsmem` に書き込みます。 `fence.proxy_async` と `warpgroup_sync` は、スレッドで書かれた SMEM 値を TMA ストアパスに準備させます。ストアはその後、SMEM から GMEM への移行が完了するのを `commit_group()` と `wait_group(0)` で待ちます。

この時点で正しい曲は揃っていますが、リズムが間違っています。ステップ 4 は、マッチング MMA を開始する前に各ロードを完了するため、ロードと乗算が同時に実行されることはありません。私たちが必死に分離しようとした 2 台のエンジンは、今でも交代で動いています。次のステップでは TMA のロード&ストアパスはそのままにし、スケジュールを並べ替え、1K tile の読み込みを進めつつ別の K tile で計算が実行できるようにします。

## ステップ 5: ソフトウェアパイプライン(PIPE_DEPTH=2)

なぜステップ 4 は負荷と計算を重ねることができなかったのでしょうか?両エンジンは明らかに独立しているのに。障害は実は収納でした。SMEM tile ペアが 1 組だけの場合、次のロードは行き先がなく、現在の MMA がそのペアの読み込みを終えるまで開始できません。なぜなら、早めに始めるとまだ使用中のデータを上書きしてしまうからです。ステップ 5 では、共有メモリをダブルバッファリングすることでそのストレージの競合を除去します。シングル warpgroup ループは次の TMA ロードを起動する前に各 MMA を待ちますが、プリフェッチして再利用できる明確なステージが用意されています。まだ M=N=K=4096 の大きさです。

> このステップで変わること: layout
> - scope: 変更なし、warpgroup 1 つ。
> - layout: 単一の SMEM tile ペアが `PIPE_DEPTH` 段階のリングバッファになります。
> - dispatch: 変更なし、TMA ロード、 `tcgen05` MMA;このステップではプリフェッチとステージ再利用が追加され、フルロード/コンピュートのオーバーラップはステップ 7 で実現します。

### パイプラインのウォークスルー

`PIPE_DEPTH=2` では、kernel は 2 つの SMEM ステージを割り当て、ロードパスと MMA パスを別々のスロットで作業させます。

以下の図は、2 段階バッファが可能にすることを意図しているパイプライン構造として読み取れ、この単一 warpgroup kernel の正確な実行トレースとしてではありません。ステップ 5 はリングバッファを構築し、後の段階をプリフェッチしますが、メインループは次の TMA ロードを発行する前に現在の MMA を待ちます。完全な負荷と計算の重複はステップ 7 で訪れ、warp specialization により TMA と MMA は別の役割が与えられます。

![*Pipeline PIPE_DEPTH=2, the target schedule; this single-warpgroup step only prefetches, full overlap arrives with warp specialization in Step 7*](../../gpupro/images/pipe_depth2.png)

プライミングが完了すると、ループは 2 段階を交互に通過します。2 つの TMA 積載が両段を前方に埋めています。その後、ループは現在のステージを待ち、MMA を実行し、その MMA がステージの読み終えるのを待ち、再利用可能なステージに `k + PIPE_DEPTH` ロードを投げます。これはまだ同時進行の TMA/MMA スケジュールではありませんが、ステップ 7 で producer と consumer の役割に分けるリングバッファ構造が確立されます。

具体的には、コードはステップ 4 と 4 つの点で異なります。

1. `Asmem` と `Bsmem` は先頭 `PIPE_DEPTH` 次元を持つため、各ステージは独自の SMEM ストレージを持ちます。
2. `tma_bar` は各ステージにつき 1 つの mbarrier を持つ配列になります。
3. メインの K ループの前に、kernel は最初の 2 段階をプリフェッチします。
4. K ループは `stage = k % PIPE_DEPTH` を使います: 現在のステージを待ち、MMA を実行し、そのステージを `k + PIPE_DEPTH` に再利用します。

### パイプラインメカニクス

1. プリフェッチ: メインループが実行される前に、最初の `PIPE_DEPTH` 段階を読み込みます。これにより、ループが最初の反復で待機しているデータが必ず検出されます:
```python
for s in range(min(PIPE_DEPTH, K_TILES)):
  tma_load(s, s * BLK_K)
```

2. メインループ: 各 K tile のステージが準備完了するのを待ち、MMA を実行し、その後すぐに空いたステージを元の `PIPE_DEPTH` tile のロードを起動して再び作業に戻します:
```python
stage = k % PIPE_DEPTH
wait(tma_bar[stage], phase_tma)
mma(stage, accum)
wait(mma_bar[0], phase_mma)
phase_mma ^= 1
tma_load(stage, next_k * BLK_K)
```

3. phase 管理: ここが人をつまずかせる部分ですが、ルールは最初に見た目よりも単純です。各 barrier の位相反転ルールは、その barrier のスロット数に直接依存するため、2 つの barrier は異なるケデンスで反復するのです。MMA accumulator は 1 つの TMEM スロットに存在しているので、 `mma_bar` 1 つの barrier(`mma_bar.ptr_to([0])`)が 1 つで、毎回の反復で再訪し、その barrier は毎回反転しなければなりません。TMA の barrier は別の話をしています。各ステージに 1 つの barrier がある `PIPE_DEPTH` 要素の配列を形成し、各ステージの barrier はリングを通過するごとに一度だけ戻ってきます。したがって、ステージインデックスが 0 に戻ったときだけ `phase_tma` 反り返ります:
```python
if stage == PIPE_DEPTH - 1:
  phase_tma ^= 1
```

エージェントで試してみてください: `PIPE_DEPTH=2` と `K_TILES=5` で、メインループをトレースするように頼みます。各 `k` に対して、 `stage`、ウェイトに渡された `phase_tma` および `phase_mma` 値をリストし、新しいプリフェッチが発行されるかどうかを挙げます。 `phase_tma` はどこで切り替わるのか、そしてなぜ直近 2 回の反復にはプリフェッチがないのか?

### 完全 kernel

完全な kernel はステップ 4 の TMA ロードとストアパスをそのまま保持し、先ほど説明したステージバッファと位相ロジックでラップします。輸入は変更されていません:

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

`hgemm_v5(M, N, K)` で包まれています。 `PIPE_DEPTH=2` 定数はパイプラインステージの数を設定します(ここに 2 段階あり、これはちょうどダブルバッファリングです):

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

## ステップ 6: 永続 kernel+tile スケジューラ

これまでのすべては、単一の tile 内での作業を最適化してきました。ステップ 6 は問題のスケールを変更し、tile 全体で最適化します。

ステップ 5 は 128×128 の出力 tile ごとに 1 つの CTA を起動します。4096×4096 の出力の場合、1024 個の CTA がそれぞれ設定コストを支払い、tile が完成した瞬間に消えます。

ステップ 6 では、固定された CTA プールを起動し、各 CTA に順に多くの tile を処理するよう求めます。これにより、セットアップ作業が複数の tile に分散され、tile 割り当てが kernel 内で移動し、スケジューラが operand を再利用する順序を選択できるという点が 2 つ得られます。私たちは完全な M=N=K=4096 のままです。

> このステップで何が変わるか: 範囲
> - scope: スケジューラを通じて多くの出力 tile をループする、永続的な CTA の固定プール。
> - layout: 変更せず、同じ tile ごとの SMEM/TMEM/レジスタ経路。
> - dispatch: 変更なし。

### 永続スケジューリング

永続 kernel の定義的な考え方は、問題ではなくハードウェアに合わせて grid のサイズを調整することです。出力 tile の数に関わらず、1 つの SM ごとにおおよそ 1 つずつ `SM_COUNT` CTA を起動し、各 SM を継続的に占有させることを目指しています。「おおよそ」と言うのはあえて、正確な 1 対 1 の居住は保証されません。なぜなら、それは稼働状況やハードウェアが CTA をスケジュールする方法によって異なるからです。

B200 を狙っている `SM_COUNT=148`。その 148 の CTA は、 `ClusterPersistentScheduler2D` から渡された tile の上をループしています。

最初の利得は償却です。TMEM の割り当て、barrier 初期化、スケジューラの状態は、1 つの CTA ごとに 1 回ずつ行われ、CTA が扱う約 7 つの tile で再利用されます。使い捨て CTA 間で 1024 回繰り返されるのではなく。

2 つ目の報酬はスケジューラーが選ぶ順番から来ます。 `l2_group_size=8` 近くの tile をグループ化し、行帯を共有する tile は同じ A 行 tile を再利用し、列帯を共有する tile は同じ B tile を再利用します。これらの tile を連続して実行することで、HBM から再取得する代わりに L2 で operand をホットに保ちます。これはまさにステップ 3 が残した再利用のポイントです。

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

tile をループさせることで、見落としがちな正確性の結果が一つあります。各 tile は独自の新しい K ループを走っており、barrierphase は既知の状態から始まらなければなりません。ステップ 5 では CTA がちょうど 1 tile を処理していたので、 `phase_tma` と `phase_mma` を一度に初期化するのは問題ありませんでした。ステップ 6 では、初期化装置は `while tile_scheduler.valid()` ループの *内部* に移動し、各 tile が自身の TMA および MMA 処理にマッチした位相状態から始まるようにしなければなりません。前の tile が残したものを引き継ぐのではなく:

```python
while tile_scheduler.valid():
  phase_tma: T.int32 = 0
  phase_mma: T.int32 = 0
  ...
```

### 完全 kernel

構造的には、kernel はステップ 5 のパイプラインを tile レベルの外ループに巻き付けたに過ぎません。新しい依存関係はスケジューラ自体だけで、他のものと一緒にインポートしています:

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

grid 次元は単に `(M//BLK_M, N//BLK_N)` `SM_COUNT` となり、各 CTA に tile を渡す役割は `ClusterPersistentScheduler2D` が引き継ぎます。

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

## 演習

1. ステップ 4 では、 `arrive.expect_tx` `(BLK_M * BLK_K + BLK_N * BLK_K) * 2` バイトを使用します。このバイト数が少すぎたり大きすぎたりした場合、mbarrier は何を待っているのでしょうか?
2. ステップ 5 では、なぜ各 SMEM ステージごとに 1 つの `tma_bar` を共有しないのに、それぞれ独自の TMA barrier が必要なのでしょうか?
3. ステップ 6 では、4096 x 4096 の出力 `BLK_M=BLK_N=128` 出力 tile 数は何枚ですか? `SM_COUNT=148` の場合、各永続 CTA は平均して何 tile を処理するのでしょうか?
