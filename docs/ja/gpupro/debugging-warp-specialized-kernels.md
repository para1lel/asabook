---
title: "Warp-Specialized Kernel のデバッグ"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/debugging-warp-specialized-kernels/
pageClass: gpupro-page
---

パート 3、GEMM ステップ 7-9 は、TMA ロード、 `tcgen05` MMA、TMEM/SMEM の書き込みを重複して実行します。 Flash Attention でのデータハンドオフのデバッグも同じ方法で行えます。まず各ロールを特定し、それぞれの役割が持つストレージ容量を決定し、最後に生成された CUDA がモデルと一致しているかを確認します。

kernel を最初から書き換えないでください。 まず、動作環境とテスト自体が有効であることを確認し、生成された CUDA を確認します。 環境およびコンパイルの問題を除外した後、このような kernel のランタイム障害は通常、データハンドオフに起因します: barrier が初期化されていない、到着カウントエラー、ロールブランチに集合が配置された、barrierphase が終了している、または producer 書き込みがまだ見えない状態でストレージ領域が先取り的に再利用されることがあります。

## 試運転前に環境を確認してください

まずは、運用環境の問題を解決しましょう。

```bash
python -c "import tvm, tvm.tirx; print(tvm.__file__, tvm.__version__)"
python - <<'PY'
import torch

print(torch.cuda.get_device_name(), torch.cuda.get_device_capability())
PY
```

これらの粒はブラックウェル(`sm_100a`)に向かっています。 Python が古い TVM チェックアウトをインポートしている場合や、現在の GPU が Blackwell アーキテクチャでない場合、kernel を変更する前にまず環境を修正すべきです。 次に最小正解度テストを実行します。例えば `run_correctness()`; 正しいテストに合格したら、次にパフォーマンスを見てください。

## 試運転開始

1. 問題を再現できる最小の形状で走りましょう。 不正なメモリアクセスが発生した場合は、次に Python を実行する前に再起動してください。
2. コンパイルが失敗した場合は、まずインストール済みの API、ターゲット、 `dispatch=`、バッファ scope を確認し、その後ランタイム同期コードを確認してください。
3. `inspect_source("cuda")` の出力を節約できます。 まずは role guard、 `mbarrier_init`、 `tcgen05`、 `cp.async.bulk.tensor`、 `cta_sync()` を検索し、その後 Python を読み直します。
4. 不具合のある kernel パスについては、ロール、ストレージ、ハンドオフ、ライフタイムテーブルを書きます。
5. この表で、生成された CUDA: barrier init が役割ブランチより前にあるか、TMA producer、MMA 発行者、書き込みグループが期待通りにいるか、CTA 全体の集合が誤って 1 つの warpgroup のみで実行されたブランチに配置されていないかを確認してください。
6. 問題をデッドロック、クラッシュ、誤った結果、または正しいが遅いものに分類し、下記の対応するセクションを確認してください。
7. 同時に変更されるハンドオーバーは 1 つだけです: init カウント、到着/待機 phase、ロールガード、フェンス、TMA ストアドレイン、TMEM のアロック/ディールロック、または tile スケジューラのアドバンスです。
8. 各修正の後、正確性が再確認され、その後性能が測定されます。

## データの引き渡し方法を明確に示してください

非同期 kernel をデバッグする前に、まず簡単な表を入力できます:

| プロジェクト | 記録すべきことは何でしょうか |
|---|---|
| 役割 | どのスレッド、warp、warpgroup、または CTA が各非同期操作を発信していますか? |
| 保管 | tile は各ステップに位置します: GMEM、SMEM、TMEM、またはレジスタ。 |
| ハンドオフ | producer、consumer、同期オブジェクト、到着カウント、位相、そしてフェンスまたはドレインを含んでデータを可視化します。 |
| 生涯 | 各ストレージ場所が再利用、読み取り、またはリリース可能な最も早い時間帯です。 |

次に、この表に基づいて生成された CUDA を確認してください:

- ロールガードはロールテーブルと一致します。
- barrier イニットはガードに守られるキャラクターブランチの前に位置します。
- レーン、warp、warpgroup のガードによって集団参加が誤って減少することはありませんでした。
- 到着/待機 phase はハンドオフテーブルと一致します。
- TMA ストアドレイン、TMEM のディールロック、SMEM の多重化は、ライフタイムテーブルが許す場合にのみ実行されます。

この表は、GEMM の TMA -> MMA ->writeback パイプライン、そして Flash Attention におけるスコア、softmax、バリュー、補正間のハンドオフの両方に適用されます。

## コンピレーションは失敗しました

まずコンパイルの問題を解決し、その後ランタイム同期のデバッグを行います:

| 現象 | 可能な場所 | まずはチェック |
|---|---|---|
| TIRx API 不明または属性エラーが発生しました | 取り付けたホイールがチュートリアルコードと一致しません | `tvm.__file__` と `tvm.__version__` を出力し、API 名を[TIRx 言語リファレンス](/ja/gpupro/tirx-language-reference/)と照合してください。 |
| 指定された `dispatch=` はサポートされていません | 現在、Target や Primitive はこのパスをサポートしていません | パラメータと目標能力 `dispatch` 確認; このチュートリアルの `tcgen05` ルートにはブラックウェルが必要です。 |
| バッファ scope の不一致 | バッファは誤ったハードウェア経路を通じて使用されます | 表のストレージを確認してください: TMEM は `tcgen05` 経由でアクセスしなければならず、TMA operand は互換性のある GMEM/SMEM layout を使用していなければなりません。 |
| コンパイルは成功しましたが、生成された CUDA には期待された経路が存在しません | 出動は予想通りに下がりませんでした | アルゴリズムを修正する前に、生成された CUDA で `tcgen05` と `cp.async.bulk.tensor` を検索してください。 |

## 生成されたコードを確認してください

任意のコンパイル済み kernel に対して、CUDA のソースコードを保存して簡単に検索・比較することができます:

```python
from pathlib import Path

cuda_source = ex.mod.imports[0].inspect_source("cuda")
Path("artifacts").mkdir(exist_ok=True)
Path("artifacts/my_kernel.cu").write_text(cuda_source, encoding="utf-8")
print(cuda_source)
```

TIRx 構成と CUDA 生成の対応関係は以下の通りです。

| TIRx | 生成 CUDA |
|------|---------------|
| `wg_id == 0` | `(warp_id_in_cta >> 2) == 0` |
| `wg_id == 1` | `(warp_id_in_cta >> 2) == 1` |
| `warp_id == 0` | `(warp_id_in_cta & 3) == 0` |
| `warp_id == 3` | `(warp_id_in_cta & 3) == 3` |
| `lane_id == 0` | `(((int)threadIdx.x) % 32) == 0` |
| `.init()` 内部警備隊 | `((int)threadIdx.x) < 1` (CTA スレッド 0 のみ許可) |
| `elect_sync()` | `tvm_builtin_elect_one_sync_op()` |

kernel 全体を読む前に、まず以下の文字列を検索してください。

| 生成された CUDA | 内容を確認してください |
|---|---|
| `if (threadIdx.x < 1)` | 単一の CTA スレッドのガードは通常、barrier の初期化に使用されます |
| `mbarrier_init` | barrier が存在し、キャラクターの分岐より先にあるかどうか |
| `tcgen05` | tensor コア経路が生成されたかどうか |
| `cp.async.bulk.tensor` | コピーは TMA に下がっているのでしょうか? |
| `cta_sync();` | CTA 全体の barrier; `wg_id` 支店内に設置することはできません |

## ステップ 7: 参照構造

ステップ 7 kernel をコンパイルするための正しいトップレベル構造は以下の通りです。 読みやすくするために、ここではキャラクター名の隣に「guard」と書きます。 生成された CUDA では、上記の表に対応する式を検索する必要があります。

```c
// (1) Barrier inits: top level, CTA thread 0 only
if (threadIdx.x < 1) {
  mbarrier_init(tma2mma[0..1], 1);
  mbarrier_init(mma2tma[0..1], 1);
  mbarrier_init(mma2ld, 1);
  mbarrier_init(ld2mma, 128);   // arrived by all 128 WG0 threads
}

// (2) TMEM alloc: WG0 warp 0, all lanes of the issuing warp
if (wg_id == 0 && warp_id == 0) tcgen05_alloc(..., 512);

// (3) Fences + cta_sync, then phase init: producer=1, consumer=0

// (4) Warp-specialized loop
if (wg_id == 1 && warp_id == 3 && elect_sync) {
  /* TMA */
  while (valid) {
    ...
    next_tile();
  }
}
if (wg_id == 1 && warp_id == 0 && elect_sync) {
  /* MMA */
  while (valid) {
    ...
    next_tile();
  }
}
if (wg_id == 0) {
  /* WB */
  while (valid) {
    ...
    next_tile();
  }
}

// (5) Cleanup: issuing warp, no lane guard
cta_sync();
if (warp_id == 0) {
  tcgen05_relinquish_alloc_permit();
  tcgen05_dealloc(..., 512);
}
```

アルゴリズムを修正する前に、まず次の点を確認してください:

- barrier イニットは `wg_id` ガードの中ではなく、最上階にあります。
- `tcgen05_alloc` と `tcgen05_dealloc` は warp ガードを持っていますが、レーンガードはありません。 warp 内のすべての航線が指揮を発します。
- TMA ループも MMA ループも `K_TILES` 回繰り返します。
- producer の初期段階は `1`、consumer の初期段階は `0` です。

## 現象に基づいて問題を特定しましょう

現象はあくまで手がかりとして機能すべきであり、最終的な診断として直接受け取るべきではありません。

| 手がかり | 可能な場所 | まずはチェック |
|---|---|---|
| kernel が詰まってしまい、ランタイムが不明な起動失敗を報告します | デッドロック | barrier の開始場所、到着数、 `cta_sync()` 場所、 `next_tile()` 関与範囲 |
| 不正メモリアクセス、XID、またはそれ以降の無関係な CUDA 呼び出しも失敗します | クラッシュ/中毒の状況 | Python を再起動して、ポインタの scope、ストレージの寿命、集合参加の scope を確認してください |
| エラーラインは 128 本の線または 1 tile 単位のストライプとして表示されます | 同期競争または tile インデックスの不一致 | producer/consumer phase、スケジューラーの進行、各行がどの warpgroup に属するか |
| 価値値が `NaN`、または明らかに無効なものを示す | ディスクリプタ、operand 設定、または accumulator は初期化されません | SMEM/TMEM 記述子、スウィズル/layout、accumulator の初期化 |
| 値は制限されていますが、誤差パターンは固定されています | 古いデータは読み取られるか、データの一部だけが完成します | フェンスや TMA の倉庫排水口が欠けているか、また保管が寿命が許される前に再利用されているか |
| 結果は正確でしたが、期待された加速とは違いました | 派遣や資源の問題 | 生成された CUDA 経路、パイプライン深度、占有率、レジスタのスプリル |

## Python の再起動時期

CUDA エラーは必ずしも自動的に回復するわけではありません。 不正メモリアクセス、XID、または「CUDA コンテキスト中毒」が発生した場合、その後の無関係な呼び出し(`torch.randn` など)も失敗し続けることがあります。 次の修正をテストする前に、Python プロセスを再起動してください。そうでなければ、前回のクラッシュ時の状態が残っている可能性があります。

## デッドロック

以下の順番で確認してください:

- 到着回数と初回数が一致しません。共通の状況は `MBarrier.init(128)` ですが、 `arrive` は `if warp_id == 0: if lane_id == 0:` に入っており、最終的には一つのスレッドだけが到達し、待ち時間は戻りません。

| barrier | init(カウント) | 誰が到着を送るのか | 到着者 |
 |---|---|---|---|
| `TMABar` (tma->mma) | 1 | TMA エンジンは `arrive(stage, bytes)` | 1 |
| `TCGen05Bar` (mma->tma, mma->ld) | 1 | MMA warp は `tcgen05.commit` | 1 |
| `MBarrier` (ld->mma) | 128 | WG0 のすべてのスレッドは `arrive` | 128 |

- barrier イニットはガード内部 `wg_id` 設置されています。 `.init()` `if threadIdx.x < 1:`、CTA スレッド 0 に降ります。 CTA スレッド 0 は WG0 に位置しているため、 `if wg_id == 1:` 内に置くとすべてのスレッドが init を実行しません。 INIT は最上階に違いない。 `inspect_source()` で `mbarrier_init` 認証を検索できます。

- `cta_sync()` warpgroup 支部内に位置しています。 `cta_sync` は `__syncthreads()` に対応し、CTA 内のすべてのスレッドが参加する必要があります。 `if wg_id == 0:` に置かれると、WG1 は決して到達できません。 1 つの warpgroup だけを同期させる場合は `T.cuda.warpgroup_sync(10)` を使いましょう。

- 一部の consumer warpgroup のスレッドは `tile_scheduler.next_tile()` を飛ばしました。 スケジューラスレッドごとの状態を保存します。 スキップされた呼び出しスレッドはループ内に永遠に残ることがあります。

- TMA は MMA の K tile の数と一致しません。MMA が `K_TILES` 回ではなく `K_TILES - 1` 回実行された場合、barrierphase は徐々にシフトし、2 枚目の外側 tile でデッドロックされます。

- `PipelineState` 初期位相誤差です。producer は `phase=1` から始まり、最初の待ち時間が直接過ぎます。 consumer は `phase=0` から始め、最初の待ち時間が詰まってしまいます。 両者が同じ phase から始まると、最初のハンドオーバーはすぐにデッドロックされることがあります。

## クラッシュとコンテキストポイズニング

一般的な原因には以下のようなものがあります:

- `pool.commit()` 後に `pool.alloc` に電話する。barrier ラッパーは内部で `alloc` を呼びます。 正しい順序は `tmem_addr -> barrier wrappers -> move_base_to(1024) -> Asmem / Bsmem / Dsmem -> commit()` です。
- `tcgen05.alloc` や `tcgen05.dealloc` をレーンガードで囲みます。コマンドを出す warp はすべてのレーンを巻き込む必要があります。 `if lane_id == 0:` 1 つのスレッドのみを実行することは未定義の動作と見なされます。
- `tcgen05.dealloc` `cta_sync()` が欠けていた。書き込みがまだ読み取られている間に、TMEM が解放されます。
- GMEM または SMEM の国境越えアクセス。問題を tile に絞り込み、スケジューラーの `m_idx` / `n_idx` を確認し、現在の形状が kerneltile やクラスタ tile の整数倍であることを確認しましょう。

## 結果は間違っていた

まず誤りパターンに基づいて分類し、その後原因を推測します。 全行エラーは行ストライプとして表示され、通常は producer と consumer の phase、tile インデックス、または役割所有権の不一致を示します。 `NaN` はしばしばディスクリプタ、operand 設定、または初期化されていない蓄積から生じます。 限定的だが固定された誤り値があり、通常は consumer が古い tile を読んだこと、部分的にしか書かれていない tile、またはまだ排水されていないストアを読んだことを示します。

- `tcgen05.commit` `elect_sync` の外にあります。32 スレッドがコミットグループを作成します。 そのうち 31 の空グループが即座に mbarrier に通知し、TMA が MMA が読み取る前に SMEM を上書きできるようにします。
- TMA ストアの前で欠 `fence.proxy_async("shared::cta")`。 TMA エンジン SMEM への書き込みスレッドを認識しない場合があります。
- TMA ストアには `cp_async.bulk.commit_group()` や `wait_group(0)` が欠けています。 ストアまだ排水されておらず、次の tile は Dsmem を再利用します。
- 永続 kernel は 1024x1024 のような小さな形状で時々失敗します。より大きな形状や長い K ループは競合を隠すことがあります。 tile 間の位相リセットと TMA ストアコミット/待機を再確認してください。
- `fence.after_thread_sync()` は通常、解決策にはなりません。MMA 完了 mbarrier はリリースと獲得の意味論を提供しました。 ステップ 8 と 9 は保守的に writeback の端に加算します: 位置は `mma2ld.wait` の後、最初の `tcgen05.ld` の前です。 TMA から MMA へのエッジを日常的に加えないでください。

## 結果は正しいですが、パフォーマンスはあまり良くありません

結果が正しくても性能が期待を大きく下回る場合は、同じ検査プロセスを継続できます。

| 手がかり | 可能な場所 | まずはチェック |
|---|---|---|
| 生成された CUDA には `cp.async.bulk.tensor` がありません | コピーは TMA に降格しません | `dispatch="tma"`、ターゲット機能、operandlayout の確認 |
| 生成された CUDA には `tcgen05` がありません | ブラックウェル・tensor・コア命令では MMA は下がりません | `dispatch="tcgen05"`、ターゲット機能、operandlayout の確認 |
| TMA と MMA は重なりません | パイプラインが浅すぎるか、位相が producer/consumer をシリアル実行に誘う | CUDA 生成の順番を確認してください。 |
| 小さな形状は正しいですが、大きな形状は性能が悪いです | レジスタの流出、占有またはステージング緩衝圧 | コンパイラのリソースレポートを確認してください。 tile の削減、書き込みブロック、パイプライン深度の縮小 |

## 有効な問題報告を提出してください

上記のチェックを完了しても問題が続く場合は、まず複製範囲を絞り込み、[A Pache TVM GitHub リポジトリ](https://github.com/apache/tvm/issues)問題を提出してください。 要件には以下が含まれます:

- `tvm.__file__`、 `tvm.__version__`、GPU 機能;
- 問題を再現できる最小の形状;
- 問題は、それがコンパイル失敗、デッドロック、クラッシュ、誤った結果、あるいは正しいが遅いかどうかです。
- 最小 kernel またはノートセルと対応する正確性テスト;
- `inspect_source("cuda")` 出力や、警備、barrier、または dispatch の経路が疑わしいものを示す最小の断片を保存します。
