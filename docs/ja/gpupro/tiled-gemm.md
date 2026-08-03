---
title: "Tiled GEMM の構築"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/tiled-gemm/
pageClass: gpupro-page
---

::: info 概要
- 1 つの出力 tile から始め、TIRx の tile プリミティブを使って正しい tiled GEMM を構築します。
- ステップ 1 で単一 tile の GEMM、ステップ 2 で K ループの累積、ステップ 3 で行列全体を覆う CTA の空間タイリングを実装します。
- 正確さが最優先です。パフォーマンスは次の二章の役割です。
:::

GEMM は本書全体を貫く中心的なワークロードです。線形層、attention projection、畳み込みの基礎にあり、GPU 実行時間の大部分を占めます。そのため、単に正しい GEMM と高速な GEMM の差は、チップの大半を遊ばせるか、十分に活用するかの差に直結します。

この差を一足飛びに埋めることはできません。最初から高性能 kernel を狙うと、データ転送、累積、タイリング、Tensor Core のスケジューリングを同時にデバッグすることになり、信頼できる比較基準も得られません。そこで、正しい結果を返す最小の kernel から始め、設計上の判断を 1 つずつ追加していきます。

この章では、まず正しい tiled GEMM を実装します。前章で概観した TIRx の scope / layout / dispatch モデルを実際の kernel に適用し、1 つの 128 x 128 出力 tile から出発して、フルサイズの行列、K 次元の累積、複数 CTA による空間タイリングへと段階的に拡張します。

本章は、1 本の GEMM 最適化経路を追う 3 章構成の第 1 章です。ここでは正しい tiled kernel を構築するところまで進みます。次章の [TMA による GEMM のパイプライン化](/ja/gpupro/pipelined-gemm/) ではスレッドによるコピーを TMA に置き換え、データ転送と計算を重ねます。続く [Warp Specialization と Cluster による GEMM のスケーリング](/ja/gpupro/warp-specialized-gemm/) では warp specialization と CTA cluster を導入します。kernel を毎回作り直すのではなく、前章の実装に機能を積み重ねていきます。

各ステップでは、同じ 3 項目の取り決めを編集すると考えると理解しやすくなります。どの scope が操作を実行するか、operand tile がどの layout を使うか、どの dispatch 経路で実行するか、の 3 点です。各ステップの冒頭では主要な変更点と、安全な再利用に必要な同期を短く整理します。ステップ 1 が、以降の変更の基準になります。

## GEMM

GEMM は線形層、attention 射影、多くの畳み込み実装の下に位置する高密度行列乗法であり、高速な GEMM kernel はほぼどこを見ても効果を発揮します。このチュートリアルの例は $D = A B^{\top}$ を使います:

- $A$ は形 $M \times K$。
- $B$ は形 $N \times K$。
- $D$ は形 $M \times N$。
- $D[m,n] = \sum_k A[m,k] \cdot B[n,k]$。

転置は追加で実行する演算ではなく、データの格納方法から生じます。この例では $B$ を、長さ $K$ の行を $N$ 本持つ行列として格納します。これは線形層の重みによく使われる layout であり、 $K$ に沿って縮約すると、データを並べ替えずに自然に $B^{\top}$ として読み取れます。

チュートリアル全体を通して、TFLOPS で kernel のスループットを測定し、掛け算 1 回あたりの浮動小数点演算を壁クロック時間に比べてカウントします。

$$\text{TFLOPS} = \frac{2 \times M \times N \times K}{t_{\text{seconds}} \times 10^{12}}$$

### GEMM データパス

このチュートリアルのすべての最適化は、データの位置と移動に帰着するため、コードを書く前にその点をマッピングする価値があります。本質的に、Blackwell GEMM kernel は 2 つの活動を中心に組織されています: メモリ間で tile を移動することと、その上で計算すること。下の図は、入力から出力までのすべての記憶を tile でなぞっています。

![*Memory Data Flow*](../../gpupro/images/memory_dataflow.png)

上の図は、後の最適化で修正されるが置き換えはしないベースラインパスを示しています。左から右へ読みます: operand tile はまず GMEM から SMEM へ移動します。 `tcgen05.mma` は SMEM の operand を消費し、TMEM に accumulator を書き込みます。最後に epilogue では TMEM をレジスタに読み戻し、結果を GMEM に保存します。この連鎖を覚えておいてください。なぜなら、以下のステップごとにこれらのホップの起こる様子が変わり、ホップ自体は変わりません。

## 最適化経路

上記のプレーンデータパスで正解が得られますが、ほとんどのハードウェアはアイドル状態のままになります。チュートリアルの残りは、ブラックウェルの特徴を一つずつ追加し、それぞれを TIRx tile プリミティブで表現することでそのギャップを埋めます。私たちがたどる道筋は、順にこれらの特徴を巡ります。

- TMA 非同期移動 GMEM <-> SMEM tile を Blackwell のハードウェアコピーパス内で移動させ、完了を追跡する barrier があります。
- ソフトウェアパイプラインは複数の SMEM ステージを利用し、次の K tile のデータ移動が現在の tile の tensor コア計算と重なるようにしています。
- 永続スケジューリングは、CTA のプールを固定し、各 tile が tile スケジューラを通じて多くの出力 tile を処理するため、1 tile ごとに 1 つの CTA を起動するのを防ぎます。
- warp 専門 producer、MMA consumer、書き込みの役割を別々の warpgroup に分けています。
- CTA cluster、2 つの CTA が 1 つの大きなブラックウェル MMA tile 上で協力できるようにします。
- マルチ consumer 実行は、複数の consumer warpgroup を用いて tile の異なる部分を同時に計算し、計算密度を高めます。

---

## ステップ 1: 逐次シングル tile GEMM

ハードウェアパス全体を処理する最も単純な GEMM は、単一の出力 tile を計算するものです。ここから始めましょう。ステップ 1 では、K = 64 の 128×128 出力 tile を計算します。ループをしなくて済み、データパスのすべての部分がちょうど一度ずつ現れます。繰り返しがなければ、ループについて推論する前に各ホップを個別に見ることができます。

> このステップで確立するもの: ベースライン
> - scope: 128 スレッドの warpgroup が順番に全ての道を歩き、次の段階を進みます。
> - layout: A tile と B tile は SMEM に、accumulator は TMEM に、結果はレジスタを通じて段階的に処理されます。
> - dispatch: 同期 `Tx.copy` が荷物を運び、MMA を `tcgen05` 走らせます。

### シングル tile データフロー

基準契約が固定されたら、次に決めるべきは、1 つの tile がその契約を通過する順番です。この最初の kernel はコアとなる GEMM データパスを正確に一度だけ歩きます。同じ GMEM -> SMEM -> TMEM ->がデータフロー図から GMEM チェーン->レジスタを登録し、ループは巻き込まれません。作業メモリを割り当て、operand を読み込み、積を計算し、結果を書き戻し、自分の処理を行います。

1. 割り当て: SMEM(プールアロケーター)、TMEM(`tcgen05.alloc`)、mbarrier
2. ロード: 全 128 スレッドが協力して GMEM から SMEM へ A tile と B tile をコピーします(同期 `Tx.copy`)
3. Compute: 単一選出スレッドは `Tx.gemm_async` + `tcgen05.commit` を発行します。すべてのスレッドは mbarrier で待機
4. Writeback: Warpgroup は TMEM→レジスタを読み込みます。各スレッドは FP32→FP16 をキャストし、GMEM に書き込みます
5. 解放: TMEM を解放

### 最初の kernel を構成する 4 つの部分

丸ごとは数十行ですが、部分的に消化しやすいです。メモリ割り当て、同期ロード、MMA dispatch、書き込みの 4 つの部分に分けて読み込み、その後に 1 つの kernel に組み立てます。途中で登場する API 名は、パート II で紹介された TIRx tile プリミティブ語彙([TIRx 入門](/ja/gpupro/tirx-introduction/)、[TIRx Layout API](/ja/gpupro/tirx-layout-api/))です。

メモリ割り当て。kernel はまず、operand 用の共有メモリと TMEM アドレス用のスロット、mbarrier を刻みます。

```python
pool = T.SMEMPool()
tmem_addr = pool.alloc((1,), "uint32")           # TMEM address (4 bytes)
mma_bar = pool.alloc((1,), "uint64", align=8)    # mbarrier (8 bytes)
pool.move_base_to(1024)                           # Skip to offset 1024
Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)  # 128×64 fp16
Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)  # 128×64 fp16
pool.commit()
```

ここで一時停止する価値のある二つの詳細があります。 `pool.move_base_to(1024)` は Asmem と Bsmem を 1024 のオフセットに割り当て、低いアドレスはその上の小さなメタデータの断片に予約し、かさばる operand tile がきれいな境界に落ちるようにしています。そして `layout=A_layout` は `tma_shared_layout`、TMA と `tcgen05.mma` の両方が直接読める簡易された SMEM 配置を求めており、まさにパート II で説明された layout と契約義務の一例です。

同期負荷。バッファが設置されても、operand は SMEM に到達しなければなりません。この最初のバージョンでは、CTA 自身のスレッドにコピーを任せました:

```python
Tx.cta.copy(Asmem[:, :], A[:, :])
Tx.cta.copy(Bsmem[:, :], B[:, :])
T.cuda.cta_sync()
```

ここには tile が 1 つだけ(M=N=128、K=64)ため、A と B の全体をコピーすることは全体の負荷となります。 `Tx.cta.copy(...)` CTA はそのコピーに対して協力し、各スレッドが自分のスライスデータを担当します。その後のこの `T.cuda.cta_sync()` は二重の役割を果たします。すべてのスレッドが完了するのを待ち、共有メモリの書き込みを公開するため、後で MMA が `Asmem` や `Bsmem` を読み取る際に、半分満たされたバッファではなく完全な tile を認識できるようにしています。このスレッド駆動のコピーは、私たちが最初に置き換えるものでもあります。次の章([TMA による GEMM のパイプライン化](/ja/gpupro/pipelined-gemm/)年)では TMA に置き換えられています。

MMA 通信。operand が SMEM に収まったことで、MMA を発行でき、単一の選出スレッドから行います:

```python
if warp_id == 0:
  if T.ptx.elect_sync():
    Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
      accum=False, dispatch="tcgen05", cta_group=1)
    T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)
```

2 つのネストされたガードは発行者を 2 段階で絞り込みます。外側 `if warp_id == 0` warp 群の warp 0 のみを保持し、内側 `if T.ptx.elect_sync():` はその warp 内で 1 つのアクティブなレーンを選択します。これらが合わせて、 `Tx.gemm_async` と `tcgen05.commit` を動かすためのスレッドが 1 本だけ残ります。

その単一のスレッドが何を意味し何を意味しないのかを明確にする価値があります。なぜなら、自然な読み方は誤解を招くからです。単一の発行スレッドが単一のスレッドの乗算を意味するわけではありません。計算は依然として完全な tile レベルの MMA であり、ハードウェアは SMEM operand 配置と TMEM accumulator 配置で記述される tile の協調乗算を行います。重要なのは、 `Tx.gemm_async` が 1 つの *tile 操作* であり、ハードウェア命令 1 つではないということです。K = 64 tile はハードウェアの MMA K-atom(`MMA_K = 16`)よりも幅が広いため、この 1 tile の操作は K に沿ってステップアップされた生の `tcgen05.mma` 命令の短いシーケンスに縮小され、warpgroup がそれらを協力的に駆動します。tile 操作を 1 スレッドだけが発行する理由は、各基礎の `tcgen05.mma` 自体が *単一命令* 協調操作であるためです。1 回の起動で tile MMA の K 原子を駆動します。もし 128 スレッドすべてがシーケンスを発行した場合、同じ作業は単に 128 回繰り返し起動されます。最後に、 `accum=False` フラグは MMA に TMEM 宛先を上書きするよう指示します。これは、拡張すべき事前部分和がないため、ここで求めているものです。

Writeback。製品は現在 TMEM に置かれていますが、発信者は GMEM に戻して FP16 として戻したいと言っています。したがって、epilogue は結果をレジスタを通して順番に描き、その過程でキャストしなければなりません。

```python
Dreg = T.alloc_local((BLK_N,), acc_type)        # per-thread fp32 register row
Dreg_f16 = T.alloc_local((BLK_N,), d_type)      # same row, cast to fp16
Dreg_wg = Dreg.view(
  128,
  BLK_N,
  layout=TileLayout(
    S[(128, BLK_N) : (1@tid_in_wg, 1)]
  ),
)
Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
T.ptx.tcgen05.wait.ld()
Tx.cast(Dreg_f16[:], Dreg[:])
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])
```

MMA は TMEM 内に 128×128 fp32 の accumulatorーtile を残します。fp32 は意図的です: GEMM は K に沿った多くの積を和とし、実行和を高精度に保つことで、積み重なる丸め誤差を抑えます。しかし `D` は fp16 なので、値はまっすぐ外に出せません。まずレジスタに入り、そこで FP16 に絞られ、その後 GMEM に到達します。

2 つのレジスタバッファはそれぞれ異なる役割を果たします。 `Dreg` は `BLK_N` 要素のスレッドごとのバッファであり、 `Dreg_wg` は選択した layout 下で同じレジスタを warpgroup 全体で *ビュー* したものです:

```python
TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)])
```

この layout は tile の第 1 次元を warpgroup のスレッドにマッピングします: スレッド 0 は行 0 を所有し、スレッド 1 は行 1 を所有し、以下から 127 行まで進みます。2 次元は各スレッドのレジスタバッファ内に留まるため、1 つのスレッドが 1 行のすべての列を保持します。warpgroup に 128 スレッド、tile に 128 行がある場合、128×128 の出力は 1 スレッドあたり 1 行にきれいに分割されます。

その視点で accumulatorーを読み取ることはまさに `Tx.wg.copy_async(Dreg_wg, tmem)` のやり方で、Blackwell TMEM の負荷経路に降 `tcgen05.ld` します。そのロードは非同期であるため、スレッドが `Dreg` に触れる前に完成 `T.ptx.tcgen05.wait.ld()` なければなりません。そうでなければ、スレッドはロードがまだ満たされていないレジスタを読み取ってしまいます。

待機が戻ると、各スレッドのプライベート `Dreg[:]` はその 1 つの論理出力行の fp32 値を保持します。スレッドはそれらを `Dreg_f16` で fp16 に絞り込み、どのグローバル行を担当するかを算出します。

```python
m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
```

そして、 `D[m_thr, n_st:n_st + BLK_N]` と書いています。行は 4 つの warp にきれいに分割されます。warp 0 は 0-31 行、warp 1 は 32-63 行、warp 2 は 64-95 行、warp 3 は 96-127 行を書きます。

### 完全 kernel

次に 4 つのパーツを再び 1 つのランナブル kernel(M=N=128, K=64)につなぎ合わせます。輸入が最優先です:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

kernel は後のステップで使われるのと同じ `hgemm_vX(M, N, K)` ス tile で巻き付けられます。ステップ 1 は `M=N=128, K=64` で実行されるため、起動はちょうど 1 つの出力 tile を含みます:

```python
def hgemm_v1(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
  # MMA_M/MMA_N/MMA_K document the underlying hardware MMA tile; they are not
  # passed to gemm_async (which derives the MMA shape from the operand and
  # accumulator tiles), so the later steps omit them.
  MMA_M, MMA_N, MMA_K = 128, 128, 16

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

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    # Step 1 is a single-tile kernel: M = BLK_M and N = BLK_N, so the grid
    # is 1x1. Starting with a 1x1 grid keeps the per-CTA tile offsets
    # (m_st, n_st) trivially zero; Steps 3+ generalise this to larger M / N.
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    # A single warpgroup makes wg_id zero and unused below.
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- SMEM allocation ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    pool.commit()

    # --- Barrier + TMEM init (warp 0 only) ---
    if warp_id == 0:
      if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
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
    phase_mma: T.int32 = 0

    # --- Load: all threads copy global -> shared (synchronous).
    # With M=BLK_M and N=BLK_N the slices below cover the full matrices;
    # the slice form is kept so the diff to Step 3 (multi-tile) is minimal.
    Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
    Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
    T.cuda.cta_sync()

    # --- Compute: single elected thread issues MMA ---
    if warp_id == 0:
      if T.ptx.elect_sync():
        Tx.gemm_async(
          tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
          accum=False, dispatch="tcgen05", cta_group=1
        )
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

    # --- Writeback: TMEM -> RF -> GMEM ---
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()
    Tx.cast(Dreg_f16[:], Dreg[:])
    m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

    # --- Deallocate TMEM ---
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

その後のすべての GEMM ステップは同じ方法でコンパイル、実行、自己検証を行うため、その足場を完全にここで一度だけ綴り、そこからは kernel のみを表示します。後のステップを実行する場合は、その `hgemm_vX` と対応する問題サイズを下のものの代わりに入れます。注意点として、新しい Python セッションごとに 1 ステップだけコンパイルし、次のステップを試す前に再起動してください。例は内部名を使い回し、コンパイラはセッションごとに状態を保持します。

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device('cuda')  # gpu(0)

M, N, K = 128, 128, 64
kernel = hgemm_v1(M, N, K)
with target:
  ex = tvm.compile(
    tvm.IRModule({"main": kernel}),
    target=target,
    tir_pipeline="tirx",
  )

torch.cuda.empty_cache()
torch.cuda.synchronize()
A_tensor = torch.randn(M, K, dtype=torch.float16, device=device)
B_tensor = torch.randn(N, K, dtype=torch.float16, device=device)
D_tensor = torch.zeros(M, N, dtype=torch.float16, device=device)

# ex.mod(...) takes torch tensors directly, using the same call form
# as every other chapter.
ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
# Relative tolerance, like the warp-specialization and Flash Attention cells:
# Output magnitude grows with K, so a fixed absolute bound would
# fail at larger K.
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")

# Optional timing for larger kernels.
ITERS = 10
for _ in range(3):
  ex.mod(A_tensor, B_tensor, D_tensor)
torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(ITERS):
  ex.mod(A_tensor, B_tensor, D_tensor)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / ITERS
tflops = 2 * M * N * K / ms / 1e9
print(f"Performance: {ms:.3f} ms, {tflops:.1f} TFLOPS")
```

ステップ 1 から 3 までは意図的に小さなサイズで進めています(こちらは 128×128、ステップ 3 では 256³)ため、最初のウォークスルーを分かりやすくしています。[Warp Specialization と Cluster による GEMM のスケーリング](/ja/gpupro/warp-specialized-gemm/)の終わりにあるクロスステップ *エンドツーエンド結果* テーブルは逆のアプローチを取ります。このステップ 1 アルゴリズムを含むすべてのステップを単一の M=N=K=4096 サイズで測定するため、速度アップ比率を直接比較可能です。

### 単一 tile kernel の限界

この核は正しいです。これがステップ 1 の目的でしたが、非常に狭い範囲でのみ正しいのです。4 つの制限は意図的に組み込まれており、残りの最適化パスではそれらを一つずつ解きほぐしていきました。

- 扱うのは 1 つの K tile だけなので、大きな K で収縮することはできません。
- 出力 tile は 1 つだけを処理するため、M と N は 128 にピン留めされています。
- TMA ではなく同期 GMEM -> SMEM コピーを使用します。
- データ移動と計算が重ならないため、両者は同時に動作しません。

---

## ステップ 2: K ループ蓄積

最初に取り除くべき制限は最も小さいものです。ステップ 1 は 64 幅の K tile 1 枚のみを扱いますが、実際の行列はそれよりもはるかに大きな収縮を扱います。ステップ 2 では単一の出力 tile を保持しつつ、K は幅 64 のチャンクを複数にわたってまたがらせます。

アイデアはシンプルです: MMA -> ->待機シーケンスをチャンクごとに 1 回繰り返し、各 MMA を同じ TMEM スロットに蓄積させるのです。本当の仕事は同期にあるのです。反復間で 1 つの mbarrier を再利用することで、本章で初めての真の正しさの危険が生じます。コードが誤った phase を追跡すると、待ち時間は MMA が実際に終了する前に戻り、結果が静かに破損することもあります。以下のメカニクスは、その失敗の原因と回避方法を正確に示しています。

> このステップで変わること: layout の再利用
> - scope: 変更なし、依然として単一の warpgroup。
> - layout/再利用: 同じ SMEM tile ペアと TMEM accumulator スロットを K ループ全体で再利用します。新たな保管場所は割り当てられません。operand tile は 1 つの固定されたバッファペアを通過し、accumulator 状態は 1 つの TMEM スロットに留まります。
> - 同期: 再利用された MMA barrier は各 K チャンクで正しい段階を通過しなければならず、そうでなければ後でより早い完了が確認されます。
> - dispatch: 変更なし。

### K ループ力学

ステップ 1 は幅 64 幅の K tile 1 枚で収縮しました。ここでは単一の出力 tile を保持しつつ、行列が要求する限り K を実行させます。64 より大きい K をカバーするには、K を `BLK_K=64` の塊に分けて歩きます。各イテレーションで次の A および B の K スライスを SMEM に読み込み、 `Tx.gemm_async` を発行します。 `accum` フラグはこれらのチャンクを 1 つのドット積につなぎ合わせる役割を果たします。最初のチャンクで `accum=False` が TMEM の accumulator を初期化し、その後のチャン `accum=True` クではそのチャンクの積をすでに TMEM にある計算合に加算します。

必要なのは同期です。MMA の完了ごとに 1 つの m barrier を再利用し、安全に再利用するにはどの barrierphase を待っているかを把握することが重要だ。mbarrier は 0 または 1 の位相を持ち、予想される到着が着地するたびに別の値に切り替わります。微妙な部分は待機条件自体です。 `try_wait(bar, phase)` は barrier の内部位相が `phase` の議論と *異なる* までブロックします。したがって、私たちが議論する際には、私たちが到達しようとしている段階ではなく、過去に置くと予想される段階を指さなければなりません。

| K 反復 | 待機前のローカル `phase_mma` | `try_wait` 何を待っているのか | 待機後のローカルアップデート |
|---|---:|---|---:|
| 0 | 0 | barrier が 1 に切り替わる | `phase_mma = 1` |
| 1 | 1 | barrier が 0 に反転します | `phase_mma = 0` |
| 2 | 0 | barrier が 1 に切り替わる | `phase_mma = 1` |

その一行の一 `phase_mma ^= 1` こそが、そのテーブルを正直に保っている。それを落としても 2 回目の反復は `try_wait(bar, 0)` を呼びますが、barrier は最初の MMA 後に phase 1 に切り替わっており、待つ間にミスマッチが見られ、2 回目の MMA が終わる前にすぐに戻ってきます。kernel は半分計算された accumulator を読み取り、誤りなく誤答を報告します。このバグはコンパイルして完璧に動作するので、位相切り替えがこれほど注目される理由です。

### 完全 kernel

以下の完全な核は、K ループと位相反転を折りたたみ入れたステップ 1 のものです。インポートは以前と同じです:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

`hgemm_v2(M, N, K)` で包まれています。grid は依然として `[1, 1]` です。なぜなら、私たちは依然として単一の出力 tile を計算しているからです。成長したのは K の範囲だけです。

```python
def hgemm_v2(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K

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

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    # This is still one output tile (M=N=128).
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    pool.commit()

    if warp_id == 0:
      if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
      T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

    phase_mma: T.int32 = 0
    m_st = T.meta_var(bx * BLK_M)
    n_st = T.meta_var(by * BLK_N)

    # === K-loop: iterate over K in chunks of BLK_K ===
    # The serial device loop keeps full-K A/B parameters shaped correctly.
    for i in T.serial(K_TILES):
      # Load the i-th K chunk
      Tx.cta.copy(Asmem[:, :], A[:, i*BLK_K:(i+1)*BLK_K])
      Tx.cta.copy(Bsmem[:, :], B[:, i*BLK_K:(i+1)*BLK_K])

      T.cuda.cta_sync()

      # MMA: accum=False for first tile, True for rest
      if warp_id == 0:
        if T.ptx.elect_sync():
          Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
            accum=(i != 0), dispatch="tcgen05", cta_group=1)
          T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

      # Wait for MMA, then flip phase
      T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
      phase_mma ^= 1

    # === Writeback (same as Step 1) ===
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()

    Tx.cast(Dreg_f16[:], Dreg[:])
    m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

---

## ステップ 3: 空間 tile(マルチ CTA)

K ループは収縮の次元を処理しましたが、M と N は依然として 128×128 の 1 枚の tile にピンで固定されています。実出力は 1 tile よりはるかに大きいため、基本的な kernel の最後の部分は M と N を複数の tile で同時にカバーすることです。ステップ 3 では、出力 tile ごとに 1 つずつ CTA の 2D grid を起動し、GPU がすべての tile を並列で計算できるようにします。この例では M=N=K=256 を使い、2×2 の tilegrid を得て、インデックスを埋めずに非自明にするのに十分な量を与えます。

> このステップで何が変わるか: 範囲
> - scope: CTA の 2D grid で、各 CTA は 128×128 の出力 tile を 1 つ所有します。
> - layout: 変更なし;CTA 内では、これは Step 2 と同じ SMEM/TMEM/レジスタ経路です。
> - dispatch: 変更なし。

### grid マッピング

grid 形状は tile から直接導かれます。128×128 の出力 tile ごとに 1 つの CTA を使うと、合計で `[M // BLK_M, N // BLK_N]` CTA が必要です。Step 2 と比べて本当に新しいのは、各 CTA に対してどの行列のスライスが *自分の* スライスであるかを教えることです。

CTA `(bx, by)` は以下の出力領域を所有しています:

```text
D[bx * BLK_M : (bx + 1) * BLK_M,
  by * BLK_N : (by + 1) * BLK_N]
```

そして、CTA の K ループは自身の行帯 A と列帯 B の一致する K スライスを繰り返し読み込みます。

```text
A[bx * BLK_M : (bx + 1) * BLK_M, k : k + BLK_K]
B[by * BLK_N : (by + 1) * BLK_N, k : k + BLK_K]
```

インデックス付けは `D = A @ B.T` の慣例から直接進みます。 `bx` は A と D の行を選択し、 `by` は B の行を選択し、転置を適用すると B の列となります。

CTA ごとに 1 tile だけ割り当てるのが一番シンプルなマッピングですが、同時に無駄でもあります。行内のすべての CTA は GMEM の同じ A tile を再ロードし、列内のすべての CTA は同じ B tile を再ロードするため、隣接する CTA がすでに取り込んだデータは再利用されません。その無駄は今はそのままにしておく;持続的なスケジューリング([TMA による GEMM のパイプライン化](/ja/gpupro/pipelined-gemm/)のステップ 6)が戻り、共有 operand を L2 でホットに保ちます。

エージェントに相談: `M=N=K=256`、 `BLK_M=BLK_N=128`、 `BLK_K=64` を使って CTA `(1, 0)` と CTA `(0, 1)` を追跡するよう依頼します。各 CTA に対して、リスト `m_st`、 `n_st`、各 K 回の反復でロードされる A および B スライス、そして D 領域を書き込みます。どの B 行が D 列になるのでしょうか?なぜなら kernel が `D = A @ B.T` を計算するからです。

### 完全 kernel

kernel は再びステップ 2 で、今回は grid 形状と CTA ごとのオフセットの 2 つだけ変更があります。内側の K ループと書き込みはそのままです。インポートは同じです:

```python
import tvm
from tvm.script import tirx as T
from tvm.script.tirx import tile as Tx
from tvm.tirx.cuda.operator.tile_primitive.tma_utils import (
  tma_shared_layout,
  SwizzleMode,
)
from tvm.tirx.layout import TileLayout, S, TLane, TCol, tid_in_wg
```

grid は `[1, 1]` から `[M // BLK_M, N // BLK_N]` となり、積載と貯蔵は CTA 自身の `m_st` と `n_st` によって相殺されます。

```python
def hgemm_v3(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
  K_TILES = K // BLK_K

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

  @T.prim_func
  def kernel(
    A: T.Buffer((M, K), a_type),
    B: T.Buffer((N, K), b_type),
    D: T.Buffer((M, N), d_type),
  ):
    T.device_entry()
    # 2D grid: one CTA per 128x128 output tile
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    pool.commit()

    if warp_id == 0:
      if lane_id == 0:
        T.ptx.mbarrier.init(mma_bar.ptr_to([0]), 1)
      T.ptx.tcgen05.alloc(T.address_of(tmem_addr), n_cols=512, cta_group=1)

    T.ptx.fence.proxy_async("shared::cta")
    T.ptx.fence.mbarrier_init()
    T.cuda.cta_sync()

    tmem = T.decl_buffer(
      (128, 512), "float32", scope="tmem", allocated_addr=tmem_addr[0],
      layout=TileLayout(S[(128, 512) : (1@TLane, 1@TCol)]))

    phase_mma: T.int32 = 0

    # Per-CTA tile offsets
    m_st = T.meta_var(bx * BLK_M)
    n_st = T.meta_var(by * BLK_N)

    # K-loop with offset A and B slices
    # The serial device loop keeps full-K A/B parameters shaped correctly.
    for i in T.serial(K_TILES):
      Tx.cta.copy(Asmem[:, :], A[m_st:m_st+BLK_M, i*BLK_K:(i+1)*BLK_K])
      Tx.cta.copy(Bsmem[:, :], B[n_st:n_st+BLK_N, i*BLK_K:(i+1)*BLK_K])

      T.cuda.cta_sync()

      if warp_id == 0:
        if T.ptx.elect_sync():
          Tx.gemm_async(tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
            accum=(i != 0), dispatch="tcgen05", cta_group=1)
          T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

      T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)
      phase_mma ^= 1

    # Writeback to the correct output tile
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))

    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()

    Tx.cast(Dreg_f16[:], Dreg[:])
    m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st:n_st+BLK_N], Dreg_f16[:])

    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

## 演習

1. ステップ 1〜3 では、 `Tx.copy` MMA の前に A と B の tile を SMEM に移動させます。なぜ kernel が `T.cuda.cta_sync()` を `Tx.gemm_async` が SMEM tile を読み取るのに?
2. ステップ 2 では、もし `phase_mma ^= 1` が K ループから外された場合はどうなりますか?kernel はすべての MMA を待つのですか?それとも遅い待機が早すぎることがありますか?
3. M=N=4096 で BLK_M=BLK_N=128 の場合、ステップ 3 で何件の CTA が起動されますか?どの operand tile が隣接する CTA 間で論理的に再利用されるのか、また Step 3 はその再利用を利用するのでしょうか?
