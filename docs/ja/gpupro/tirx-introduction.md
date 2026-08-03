---
title: "TIRx 入門"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/tirx-introduction/
pageClass: gpupro-page
---

::: info 概要
- TIRx は GPU kernel を書くための Python の DSL です。 プログラムは、構造化された中間表現を保持しつつ、スレッド、SMEM、TMEM、barrier、tensor コアなどのハードウェア概念を直接利用できます。
- TIRx の tile 操作は主に 3 つの情報によって決まります: どのスレッドが操作を行うか(scope)、データの layout、そして操作を実装するために使われるハードウェアパス(dispatch)です。
- 本章は実行可能な単一 tile の GEMM から始まり、TIRx kernel の作成、コンパイルおよび検証手法、そして scope、layout、dispatch が共同で kernel の挙動を決定する方法を紹介します。
:::

::: info 運用環境
この章の例には Blackwell GPU(`sm_100a`、例えば B200)、TIRx コンパイラ、CUDA をサポートする PyTorch が必要です。 TIRx は Apache TVM ホイールの `tvm.tirx` モジュールに搭載されています。 NVRTC で CUDA コードをコンパイルする際には、 `cuda-bindings` も必要で、これらは一緒にインストールできます:

```bash
pip install apache-tvm cuda-bindings
```

インストール後、TVM と TIRx が正しくインポート可能かどうかを確認するために以下のコマンドを実行します。

```bash
python -c "import tvm, tvm.tirx; print(tvm.__version__)"
```

その後の章で実行可能な例も同じ環境を使用しています。
:::

第 1 部では、現代 GPU の実行モデル、データ layout、TMA、tensor コア、TMEM、非同期などのハードウェア機構を紹介します。 次に、これらのメカニズムを真に実行可能な kernel として組織化する必要があります。

CUDA や PTX を直接使用すれば確かに可能ですが、低レベルのプログラムはしばしば、どのスレッドが操作を行うか、operand tile をどこに格納するか、どのハードウェア命令が最終的に使われるかなど、いくつかの重要な決定を内在パラメータ、アドレス計算、コード慣習の間に分散させてしまうことがあります。 これらの情報はすべてプログラム内に書かれていますが、コンパイラが全体としてチェックし変換するのは難しいです。

TIRx(Tensor IR next)は、これら 3 種類の意思決定を構造化された IR に明示的に書き込む Python の DSL です。

- scope: どのスレッドが操作を実行するか;
- layout: 論理 tile がメモリ、レーン、レジスタにどのようにマッピングされるか;
- dispatch: tile 操作が実装されるハードウェアです。

TIRx は依然としてスレッド、SMEM、TMEM、barrier、 `tcgen05.mma` といったハードウェアの概念を直接使っています。 違いは、この情報が明確な IR 構造を持つことで、コンパイラがプログラムをチェックして基礎コードを生成できるようになったことです。

この章では構文を個別に列挙するのではなく、完全な kernel から始まります。 まず最小限の単一 tile GEMM を実行し、その後 scope、layout、dispatch を分析し、最後にコンパイルの様子を確認します。

## 最初の TIRx kernel

以下の核計算:

```text
D = A × B^T
```

`A` と `B` の形状はどちらも `128×64` であり、出力 `D` の形状は `128×128` です。 この例は出力 tile を 1 つだけ扱う `128×128`、grid も CTA は 1 つだけです。 kernel のデータパスは次のように要約できます:

```text
A/B: GMEM -> SMEM -> tcgen05.mma
D:   tcgen05.mma -> TMEM -> registers -> GMEM
```

行列乗算は TIRx で `Tx.gemm_async` tile 操作として記述されます。 この操作は完全な `128×128×64` tile GEMM を記述します。 現在、 `tcgen05.mma` は一度に 16K の要素を処理するため、コンパイラは K 次元に沿って 4 つの MMA を生成します。 特定の命令列は、形状、layout、dispatch に基づいてコンパイラによって決定されます。

以下のコードを読むと、まず 4 つの段階を理解することができます。

1. SMEM および TMEM に応募してください。
2. A と B を GMEM から SMEM へ移動させる;
3. `Tx.gemm_async` を通じて MMA を始める;
4. TMEM の結果をレジスタに読み戻し、それを GMEM に書きます。

最も重要な tile 操作は、 `Tx.cta.copy`、 `Tx.gemm_async`、 `Tx.wg.copy_async` の 3 つです。 残りの PTX コールは TMEM の要求と解放、barrier の初期化、同期の確立に使われます。 この章では、まずこれらの段階を完成させるために必要な基礎的なステップとして扱います。

まず、この kernel で使用されるモジュールをインポートします:

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

関数 `hgemm_v1(M, N, K)` TIRx `PrimFunc` を構成し返します:

`@T.prim_func` この GPU 関数を定義するために、デバイスコードのエントリーポイントを示す `T.device_entry()` `T.cta_id` grid 内の CTA 座標を取得し、CTA 内で warpgroup 番号を取得 `T.warpgroup_id` `T.warp_id_in_wg`、warpgroup 内の warp 番号を取得し `T.lane_id`、warp 内でスレッドのレーン ID を取得します。これらの値に続く scope と条件付きチェックが使われます。

```python
def hgemm_v1(M, N, K):
  a_type = tvm.DataType("float16")
  b_type = tvm.DataType("float16")
  d_type = tvm.DataType("float16")
  acc_type = tvm.DataType("float32")

  BLK_M, BLK_N, BLK_K = 128, 128, 64
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
    # 本章调用时 M=BLK_M、N=BLK_N，
    # 所以 grid shape 为 1x1，m_st 和 n_st 都是 0。
    bx, by = T.cta_id([M // BLK_M, N // BLK_N])
    wg_id = T.warpgroup_id([1])
    warp_id = T.warp_id_in_wg([4])
    lane_id = T.lane_id([32])

    # --- 申请 SMEM ---
    pool = T.SMEMPool()
    tmem_addr = pool.alloc((1,), "uint32")
    mma_bar = pool.alloc((1,), "uint64", align=8)
    pool.move_base_to(1024)
    Asmem = pool.alloc((BLK_M, BLK_K), a_type, layout=A_layout)
    Bsmem = pool.alloc((BLK_N, BLK_K), b_type, layout=B_layout)
    pool.commit()

    # --- 由 warp 0 初始化 barrier 和 TMEM ---
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

    # --- Load：所有 threads 同步地将 A、B 从 GMEM 搬入 SMEM ---
    Tx.cta.copy(Asmem[:, :], A[m_st:m_st + BLK_M, :])
    Tx.cta.copy(Bsmem[:, :], B[n_st:n_st + BLK_N, :])
    T.cuda.cta_sync()

    # --- Compute：由一个被选中的 thread 发出 MMA ---
    if warp_id == 0:
      if T.ptx.elect_sync():
        Tx.gemm_async(
          tmem[:, :BLK_N], Asmem[:, :], Bsmem[:, :],
          accum=False, dispatch="tcgen05", cta_group=1
        )
        T.ptx.tcgen05.commit(mma_bar.ptr_to([0]), cta_group=1)

    T.ptx.mbarrier.try_wait(mma_bar.ptr_to([0]), phase_mma)

    # --- Writeback：TMEM -> registers -> GMEM ---
    Dreg = T.alloc_local((BLK_N,), acc_type)
    Dreg_f16 = T.alloc_local((BLK_N,), d_type)
    Dreg_wg = Dreg.view(128, BLK_N,
      layout=TileLayout(S[(128, BLK_N) : (1@tid_in_wg, 1)]))
    Tx.wg.copy_async(Dreg_wg[:, :], tmem[:, :BLK_N])
    T.ptx.tcgen05.wait.ld()
    Tx.cast(Dreg_f16[:], Dreg[:])
    m_thr = T.meta_var(m_st + warp_id * 32 + lane_id)
    Tx.copy(D[m_thr, n_st : n_st + BLK_N], Dreg_f16[:])

    # --- 释放 TMEM ---
    T.cuda.cta_sync()
    if warp_id == 0:
      T.ptx.tcgen05.relinquish_alloc_permit(cta_group=1)
      T.ptx.tcgen05.dealloc(tmem_addr[0], n_cols=512, cta_group=1)

  return kernel
```

以下の GEMM 章はこのバージョンから始まり、徐々に K ループ、出力 tile の増加、TMA、warp の専門化を加えていきます。

## 結果をまとめて検証する

まず kernel をコンパイルし、その後 PyTorch を使って参照結果と同じ行列の乗算を計算します。 目標は `"cuda"` と書けます; TVM は `sm_100a` などの現行デバイスからの特定のアーキテクチャを検査します。 パラメータ `tir_pipeline="tirx"` で TIRx 降下パイプラインを選択します。

コンパイルされた `ex.mod(...)` は手動のデータ変換なしで直接 PyTorch tensor を受け取ることができます:

```python
import torch

target = tvm.target.Target("cuda")
device = torch.device("cuda")

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

ex.mod(A_tensor, B_tensor, D_tensor)

D_ref = (A_tensor.float() @ B_tensor.float().T).half()
max_err = float((D_tensor - D_ref).abs().max())
print(f"Max error vs torch reference: {max_err:.6f}")
torch.testing.assert_close(D_tensor, D_ref, rtol=2e-2, atol=1e-2)
print("PASS")
```

最終出力が `PASS` であれば、コンパイルされた kernel と PyTorch の参照実装結果が許容される誤差内で整合していることを意味します。

## 範囲、layout、そして派遣

さて、この kernel セクションを振り返ってみると、 TIRx の各 tile 操作は、誰が実行するか、データがどこに保存されているか、そして実装に使われるハードウェアの 3 つの質問に答えます。 対応する 3 つの設計要素は、scope、layout、そして dispatch です。

以下の相互作用図は kernel からキーコードを抽出しています。 `Scope`、 `Layout`、または `Dispatch` をクリックすると、その情報で制御されるコードの行をハイライトしてください。

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo/tirx_dispatch.html?v=intro-tirx-wheel-20260723" title="TIRx 中的 Scope、Layout 与 Dispatch" loading="lazy"
        style="width:100%; min-width:960px; height:900px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>


scope はどのスレッドが操作を行うかを決定します。 `Tx.cta.copy(...)` は CTA 全体によって協調的に実行され、この kernel 内の 128 スレッドが GMEM から SMEM へのコピーに参加しています。 `Tx.gemm_async(...)` `warp_id == 0` および `elect_sync()` 条件に該当し、最終的にメッセージを送信するのは選ばれたスレッドのうち 1 つだけです。 MMA を終えた後、計算が終わるのを待 `mbarrier.try_wait`。 その後、 `Tx.wg.copy_async(...)` は warpgroup 全体が協働して TMEM accumulator に 128 スレッドのレジスタで配布しました。

layout は tile が物理的な場所にどのようにマッピングされるかを決定します。 `A_layout` と `B_layout` は、SMEM で A と B が 128 バイトのスウィズルを使うことを指定しています。 `tmem` `TileLayout` は accumulator を `TLane` と `TCol` にマッピングします。 `Dreg_wg` ビューは `tid_in_wg` を使って各スレッドがどの行を読み取るかを指定します。 MMA とコピーのエンドは、ハードウェアが正しいマトリックス tile と同じ要素バッチを解釈するために、マッチング layout を使用しなければなりません。

Dispatch は tile 操作に使用するハードウェアを決定します。 `Tx.gemm_async` 非同期 tile GEMM を表し、さらにコンパイラが Blackwell の `tcgen05.mma` パスを選択する必要があります `dispatch="tcgen05"`。 このバージョンでは、GMEM から SMEM へのコピーは通常のスレッドで処理されます。 後のバージョンでは、同じ種類の tile コピーを TMA に変更します。

コンパイラは scope、layout、dispatch を組み合わせて、特定のスレッドレベルの制御フロー、アドレス計算、ハードウェア命令を生成します。

## TIRx のコンパイル方法

kernel はすでに以下の 2 行のコードでコンパイルされています:

```python
target = tvm.target.Target("cuda")
ex = tvm.compile(
  tvm.IRModule({"main": kernel}),
  target=target,
  tir_pipeline="tirx",
)
```

`PrimFunc` はまず `IRModule` に置かれ、その後 `tvm.compile` に手渡されます。 `tir_pipeline="tirx"` TIRx の減速パイプラインを開始します。 コアパス `LowerTIRx` は各 tile プリミティブの範囲、layout、dispatch に基づいて実装され、 `Tx.gemm_async` や `Tx.cta.copy` などの高レベルの tile 操作を低レベルの TIR へと拡張します。

その後のパスでバッファのフラット化、ホスト/デバイス分割、デバイスコード生成が完了し、最終的に直接呼び出せる `Executable` が得られます。

コンパイラが下流前後に何を生成したか知りたい場合は、TIRx `PrimFunc` と最終的な CUDA C コードを別々に確認できます。

```python
kernel.show()
print(kernel.script())

print(ex.mod.imports[0].inspect_source())
```

これら 2 つのコード層を比較することで、tile 操作が最終的にどの命令を生成するか、layout やスレッド scope が特定のアドレス計算や制御フローにどのように変化するかを確認できます。

## 次

次の章では、 `TileLayout`、斧の名前付け、そしてスウィズルをさらに紹介します。 今後の GEMM 章ではこの kernel を拡張し、K ループ蓄積、空間 tile、TMA、warp specialization が追加されます。 言語参照は別途データ型、バッファ、制御ストリーム、スレッド同期構文を導入します。
