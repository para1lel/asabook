---
title: "非同期協調: mbarrier"
createTime: 2026/08/01 00:00:00
permalink: /ja/gpupro/mbarrier/
pageClass: gpupro-page
---

::: info 概要
- TMA または tensor コア命令を発行することは、非同期操作が開始されたことを示します。 consumer は結果を読んだり関連リソースを再利用したりする前に、対応する完了信号を待たなければなりません。
- `mbarrier` producer の `arrive` と consumer の `wait` を分離し、通常のスレッドの到着や非同期ハードウェアの完了状況を同じ条件下で含めること; TMA では、まだ転送されていないバイト数も追跡します。
- 多段階パイプラインでは、各段階は通常、データが準備完了を通知するために `full` barrier を使用し、バッファを戻すために `empty` barrier を使用します。 barrier は phase を通じて繰り返し使用できます。kernel は前回のラウンドの完了状況と現在のラウンドの完了状態を混同しないように位相パリティを追跡する必要があります。
:::

TMA とブラックウェル tensor コアに関する前述の章では、2 種類の非同期操作が導入されました。 共通点は、スレッドは命令の発行のみを担当し、実際のデータ転送や行列乗算はハードウェアによって実行され続けることです。 命令を出すスレッドは、そのまま待機している必要はありません。

例えば TMA 負荷を挙げます。 プログラムはまず TMA 命令を発行し、その後 MMA を実行して SMEM tile を読み取ることができます。 しかし、このシーケンスは TMA が最初に開始したことを示すだけで、MMA がデータを読み込む時点ですでに終わっているを意味するわけではありません。 TMA がまだ書き込み中であれば、MMA は未完成 tile を読み取る可能性があります。 `tcgen05.mma` epilogue でも同じ問題があります。epilogue は tensor コアが TMEM accumulator を書き終えるまで結果を読み取るのを待たなければなりません。

この種の非同期データハンドオーバーには明確な完了信号が必要です。producer は作業終了時に通知を送り、consumer は通知を受け取った後にのみデータやリソースを再利用できます。 以下は、この信号を送信するために使われる `mbarrier` の紹介です。

## `mbarrier`

`mbarrier` は Memory Barrier の略で、共有メモリに保存されるハードウェア同期オブジェクトです。 内部符号化は不透明です。 その振る舞いを理解するには、まず到着カウンターと phase に注目してください。 到着カウンターは現在のラウンドに残っている到着数を示し、phase は barrier の現在のラウンドを示します。 待機のパリティ形式を使用する場合、kernel は `phase % 2` を追跡するだけでよい。0 または 1 の値は位相パリティと呼ばれます。 TMA ロードの場合、barrier は tx カウントを使って未完了の転送バイト数を追跡します。

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo/mbarrier_mechanism.html?v=review-20260720" title="mbarrier 的状态与操作" loading="lazy"
        style="width:100%; min-width:1320px; height:730px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>

*任意のフィールドをクリックすると、barrier 状態でのその意味を個別に確認できます。*

`mbarrier` まず、初期化が必要です。 `init` を実行する際、kernel は 1 ラウンドあたり barrier が到着を待つ回数を指定します。 phase 0 から始まる barrier では、到着予定数は期待される到着数に設定されます。 この瞬間から、関連する producer やリソース利用者が完了する報告を待ちます。

到着するたびに、現在障害を待つ作業量が軽減されます。 kernel 内の異なる参加者は異なる方法で到着を送ります。

TMA ロードの場合、一般的な経路は `mbarrier.arrive.expect_tx(bytes)` を実行することです。 この操作は同時に 2 つのことを達成します。まず、命令を開始したスレッドが 1 つの到着を完了し、保留中の到着数を 1 減らすこと。 次に、TMA エンジンが tx-count に転送することを期待するバイト数を記録します。

したがって、命令を開始したスレッドは到達を完了しますが、それが barrier が完了したことを意味するわけではありません。 TMA エンジンによって完了したすべての転送に対して、対応する tx カウントが complete-tx を通じて差し引かれます。 次の 2 つの条件が同時に満たされた場合にのみ、barrier は現在の相を完了し次の相に進みます。 対応する位相パリティもそれに応じて 0 から 1 の間で切り替わります。

```text
pending arrival count == 0
tx-count              == 0
```

したがって、 `expect_tx` 単に「また別の普通の到着」と理解することはできません。 また、非同期コピーを待つ転送バイト数も登録します。 barrier はすべての到着が完了するのを待ち、関連するすべてのデータが送信されるまで待たなければなりません。

tensor コアは別の到達経路を利用します。 単に `tcgen05.mma` を送るだけでは barrier は更新されません。 kernel はまた、以前に発行された非同期 tcgen05 操作に barrier 到着を関連付けるために `tcgen05.commit...mbarrier::arrive` を使う必要があります。 これらの作業が完了すると、ハードウェアは barrier 上に到着を報告します。 このステップが抜けていると、到着を待つ consumer はそのまま進むことができなくなります。

通常のスレッドも直接 `mbarrier.arrive` 実行可能です。 例えば、consumer が共有メモリバッファを読み取った後、到着時に producer にこのバッファを上書き・再利用可能に通知することができます。

`wait` は同じプロトコルの consumer 側です。consumer は barrier が現在の反復に対応する位相を完了するのを待ちます。 終了を待って初めてデータを読み取ったり、barrier で保護されたリソースを再利用したりできます。 元の PTX `mbarrier.try_wait.parity` は phase 完了前に `false` 戻ることがあるため、繰り返しのチェックが必要です。 本書で使われている `T.ptx.mbarrier.try_wait` はすでにこのループを内部的にカプセル化しており、指定された phase が完了するまで待ちます。 `arrive` と `wait` が分離された後、producer は進捗を報告し他の作業を続けられますが、consumer は本当にデータが必要な時だけ待ちます。

重要な点は、非同期ハードウェアがバックグラウンドで動作を続けるだけでなく、barrier レポートを通じてプロセスを完了するということです。 TMA は共有メモリ tile の準備完了を通知し、tensor コアは TMEM の結果が書き込まれたことを通知し、通常のスレッドはバッファブロックが使用されていないことを通知できます。 最終的にはすべて同じ producer と consumer の合意に従っています。つまり、producer は到着を報告し、consumer は実行して待つということです。

## phase: 異なるラウンドの区別方法

前述の通り、同じの `mbarrier` を繰り返し使用でき、各ラウンドを phase と呼びます。 一つの phase を終えると、barrier は自動的に次の phase に入り、次の到着を待ち続けます。 もし「この barrier が完了した」と言われれば、consumer は現在のデータが整ったため、前回のラウンドで残った完了状態を誤って想定するかもしれません。 位相パリティは、0 と 1 の隣接ラウンドを区別し、consumer がどのラウンドを待っているかを明確に理解できるようにします。

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo/phase_tracking.html?v=phase-order-20260720" title="复用 mbarrier 时的 phase tracking" loading="lazy"
        style="width:100%; min-width:1320px; height:640px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>

*異なる反復をクリックして、同じ barrier の各ラウンド後に位相パリティが 0 と 1 の間で交互に変化する様子を観察してください。*

上の図は barrier が相 0 と相 1 の間を切り替わり、繰り返し使用される様子を示しています。 2 段階 TMA パイプラインでは、ステージ 0 とステージ 1 にそれぞれ SMEM バッファと TMA barrier があります。 このようにして、consumer は各段階のデータを別々に待つことができます。

ステージ 0 とステージ 1 の順序で 2 段階にアクセスすると、現在のラウンドは単一の `phase_tma` で記録できます。 2 段階ごとにパリティを 1 回反転させます:

```text
stage = iteration % 2
T.ptx.mbarrier.try_wait(tma_bar.ptr_to([stage]), phase_tma)

if stage == 1:
    phase_tma ^= 1
```

初期化後、両方の barrier は phase 0 に入り、 `phase_tma` も 0 から始まります。 最初の 4 回のイテレーションにおける完全な状態変更は以下の通りです:

| 反復 | 舞台 | この待ち位相のパリティのラウンド | barrier 完成後の現在のパリティ | 反復終了後 `phase_tma` |
|---: |---: |---: |---: |---: |
| 0 | 0 | 0 | 1 | 0 |
| 1 | 1 | 0 | 1 | 1 |
| 2 | 0 | 1 | 0 | 1 |
| 3 | 1 | 1 | 0 | 0 |

イテレーション 0 はステージ 0 の phase 0 を待つ。 この段階が終わると、ステージ 0 の barrier は phase 1 に入りますが、円形バッファーはまだこのラウンドを通過していないため、 `phase_tma` は 0 のままです。 イテレーション 1 はステージ 1 の phase 0 を待ちます。 2 段階待った後、 `phase_tma` 1 に切り替わります。

イテレーション 2 はステージ 0 に戻り、今度は phase 1 を待ちます。 完了後、ステージ 0 の barrier は phase 0 に戻ります。 イテレーション 3 もステージ 1 の phase 1 を待っています。 第 2 ラウンドの両段階が終わると、 `phase_tma` もまた 0 に戻ります。

`phase_tma` は、ハードウェア上で 2 回の TMA 転送を最初に完了したかに関わらず、円形バッファへのソフトウェアアクセスのラウンドを説明しています。 したがって、深さが `S` の TMA パイプラインは通常、各段階に `full` barrier を用意して TMA 完了状態を記録し、同じ段階の 2 ラウンドを位相パリティで区別します。 完全なバッファ多重化プロトコルには、以下で紹介する `empty` barrier も必要です。

## 一般的な同期ルール

tensor コア kernel では、 `mbarrier` は主に以下の 3 種類のデータハンドオフを調整するために使われます。

非同期ハードウェアにデータを渡すスレッド。スレッドがまず共有メモリに書き込み、その後 TMA ストアや MMA でこのデータを読み取る場合、kernel はまず同期とシーケンスを確保しなければなりません。 そうでなければ、非同期操作が共有メモリバッファが完了する前に読み込みを開始することがあります。

TMA から MMA へのハンドオーバーデータ。TMA ロードは非同期で満たされた SMEM tile です。 producer は `mbarrier` 到着バイトと送信バイトの両方を同時に追跡できるようにします。 MMA を実行する consumer は、barrier が現在の位相を終え、対応する命令の順序要件を満たすまでこの tile を読み込みます。 下の相互作用図はこのタイムラインを示しています。

<div style="overflow-x:auto;">
<iframe src="/gpupro/demo/mbarrier_tma_timeline.html?v=review-20260720" title="通过 mbarrier 追踪 TMA load" loading="lazy"
        style="width:100%; min-width:1320px; height:500px; border:1px solid var(--vp-c-border); border-radius:6px;"></iframe>
</div>

*タイムラインに沿ってステップバイステップ進む。2 つの TMA コピーがそれぞれ 2048 バイトの転送を完成させることがわかります; 到着待ちと送信回数の両方がゼロにリセットされると、barrier は原子的に次の phase に移行し、その時初めて consumer の待ち時間が成功します。*

MMA の引き継ぎデータを epilogue に。 `tcgen05.mma` 非同期に TMEM accumulator を更新します。このスレッドは `tcgen05.commit` を通じて `mbarrier` への MMA リンク完了通知を開始する責任者です。 epilogue では barrier が完成するのを待ち、 `tcgen05` 順に対応するフェンスを実行してから TMEM の結果を読み上げます。

## ステージの再利用は barrier で管理されています

barrier とは「データが準備完了」だけでなく、「バッファが使い切られた」ことを意味する場合もあります。 したがって、パイプライン化された kernel は通常、各 SMEM 段階に対して 2 つの barrier を準備します。 `full[stage]` は TMA がその段階を満たし、 `empty[stage]` consumer がバッファを使い切ったことを意味します。 パイプラインが安定稼働している場合、ステージループは次のように要約できます:

![SMEM ステージ](../../gpupro/images/mbarrier_stage_reuse_v2_zh.svg)フル/エンプティ barrier 再利用プロトコル

`full` は producer から consumer へのデータ引き渡し、 `empty` consumer から producer へのバッファを返す責任を負っています。 各 barrier の予想到着数は、レポートを完了するスレッドの数に依存します。 パイプラインが同じ barrier のペアをループさせる場合、それぞれ `full` 位相パリティと `empty` の位相パリティを追跡する必要があります。

したがって、パイプライン化された kernel で待機と到達を読む際に、まず 3 つのオブジェクトを識別できます: 誰が producer か、誰が consumer か、そしてどのデータやリソースを渡しているかです。 これら 3 点が確認されると、各待ちと到着は特定のデータ準備、結果読み取り、またはバッファの多重化に対応します。
