---
title: 'Continuum: Multi-Turn LLM Agent Scheduling'
createTime: 2026/08/19 00:00:00
permalink: /ja/papers/continuum/
---

> [Hanchen Li](https://hanchenli.github.io/), [Runyuan He](https://runyuanhe.github.io/), [Qiuyang Mang](https://joyemang33.github.io/), [Qizheng Zhang](https://alex-q-z.github.io/), [Huanzhi Mao](https://huanzhimao.com/), [Xiaokun Chen](https://dblp.org/pid/252/1625.html), [Hangrui Zhou](https://hehezhou.github.io/), [Alvin Cheung](https://people.eecs.berkeley.edu/~akcheung/), [Joseph Gonzalez](https://dblp.org/pid/61/8262), and [Ion Stoica](https://dblp.org/pid/s/IonStoica.html). arXiv 初回投稿は 2025 年 11 月 4 日、現行版は v6 で、2026 年 5 月 25 日改訂。[Continuum: Efficient and Robust Multi-Turn LLM Agent Scheduling with KV Cache Time-to-Live](https://arxiv.org/abs/2511.02230v6)。[原論文 PDF](/paper/continuum.pdf)。[DOI](https://doi.org/10.48550/arXiv.2511.02230)。[TeX ソース](https://arxiv.org/src/2511.02230v6)。正確な印刷レイアウトと参考文献については、原論文 PDF を正本とする。

## 概要

KV キャッシュ管理は、効率的な LLM 推論に不可欠である。利用率を最大化するため、既存の推論エンジンは、新しいリクエストが待機している場合、完了したリクエストの KV キャッシュを退避する。この方針は、LLM 呼び出しとツール呼び出しを交互に実行し、ターンをまたぐ効果的な KV 再利用を妨げる休止が生じるエージェント型ワークロードでは機能しない。多くのツール呼び出しは、人間が応答するマルチターンチャットよりもはるかに短いため、ツールの実行中に KV キャッシュを保持することには可能性がある。しかし、課題は多い。第一に、再計算または再読み込み（オフロードが有効な場合）の潜在コストと、GPU から退避した後に増大するキューイング遅延の両方を考慮する必要がある。第二に、ツール呼び出し時間には内在的なばらつきがあるため、持続時間の予測可能性が限られる状況でも、手法は堅牢でなければならない。

本稿では、KV キャッシュ保持に time-to-live 機構を導入し、マルチターンのエージェントワークロードにおけるジョブ完了時間を最適化するサービングシステム Continuum を提示する。ツール呼び出しを生成するリクエストについて、Continuum は、再読み込みコストと退避によって生じうるキューイング遅延から time-to-live 値を決定し、KV キャッシュを選択的に GPU メモリへ固定する。TTL が期限切れになると、KV キャッシュは自動的に退避されて GPU メモリが解放され、エッジケースでも堅牢な性能が得られる。プログラムレベルの first-come-first-serve と組み合わせることで、Continuum はマルチターンの連続性を保ち、エージェントワークフローの遅延を削減する。Llama-3.1 8B/70B、Gemma-3 12B、GLM-4.5 355B を用いた実世界のエージェント（SWE-Bench、BFCL、OpenHand）での評価は、Continuum がスループットを改善しながら平均ジョブ完了時間を 8 倍以上改善することを示す。

<span id="figure-01"></span>

![従来のエージェントサービングシステムにおける二つの主な失敗モード](../../papers/continuum/figure-01.png)

**図 1。** 従来のエージェントサービングシステムにおける二つの主な失敗モード。赤いブロックは、最適でないスケジューリングと KV キャッシュ管理によるオーバーヘッドを表す。CPU オフロードを用いても、エージェントは KV キャッシュ退避後のキューイング遅延を被る。

<span id="section-01"></span>

## 1 はじめに

KV キャッシュ管理は大規模言語モデル推論の鍵であり、入力処理（prefill）と出力生成（decoding）の両段階に影響する [Kwo23, She24, Che25e]。KV キャッシュ管理の重要な構成要素は退避方針である。理想的には、近い将来に参照される token の退避を避けるべきである。従来のキャッシュシステムと同様に、既存の推論エンジンは、デコード終了後の KV キャッシュは重要性が低いと仮定する。そのため、利用率を最大化すべく、待機キューに別の新規リクエストがあれば破棄する。本稿では、この方針を**ターン終了時退避**と呼ぶ。

ターン終了時退避はマルチターンチャットには有効だが、現代のエージェント型ワークロード、特にツール呼び出しを伴うものの性能を大きく低下させうる。この種のエージェントアプリケーションは、ソフトウェア工学 [Pre24]、コンピュータ操作 [Ant24b]、科学研究 [Ren25a] などの領域で急速に普及している。これらのワークロードは、(a) 次のアクションを導出する推論ステップと、(b) エージェントが外部ツールを呼び出す実行ステップを交互に行う。ツールの出力はその後リクエストコンテキストへ追加され、推論エンジンで新しい推論ステップが開始される。ツール呼び出しは人間の入力速度よりはるかに高速（*すなわち* $\leq 2$s）になりうるため、この新しいワークロードにはターン終了時退避の変更が必要である。

中心的な問題は、エージェントが推論ステップからツール呼び出しへ移行するとき、リクエストの KV キャッシュが退避されることで生じる。このステップで KV キャッシュが退避されると、ツール実行が完了して次の推論ステップが始まる際、エンジンは prefix を再計算（prefill）するか、CPU オフロードが有効なら CPU から再読み込み [Che25e] しなければならない。反復する prefill は大きな遅延を生み、システム全体のスループットを低下させる。さらに重要なのは、KV キャッシュ再利用のために CPU オフロードを有効にしても、退避により**ターンごとのキューイング遅延**という別の問題が生じることである。次の推論ステップの KV キャッシュが GPU メモリから退避されていると、CPU から再読み込みできる場合でも、推論を開始する前に、ほかのリクエストが GPU メモリを解放するまで待機キューで待たなければならない。[図 1](#figure-01)に示すように、このターンごとのキューイング遅延は累積し、各エージェントプログラムの遅延を増大させうる。この遅延はオフラインプロファイリングでは測定できないため、その影響を含む新しいモデルが必要である。さらに、ツール呼び出しは本質的に変動しうるため、無限に待ち続けることを防ぐ最大 KV キャッシュ保持時間を設定する必要がある。しかし、この時間がツール呼び出しの直前に切れると、それまでの待機時間が無駄になる。したがって、ワークロードに最適に適応するよう KV キャッシュ保持時間を慎重に設定しなければならない。

従来研究はこれらの課題に対処できていない。InferCept [Abh24a] は、再読み込みコストだけに基づいて KV を preserve するか決定する。しかし、ターンをまたいで累積するキューイング遅延をモデル化せず、変動するツール呼び出し時間を扱う堅牢な機構も持たない。そのため、実環境への展開は現実的でない。[第 6 節](#section-06)で後述するように、InferCept はターンをまたいでキューイングペナルティを累積し、最適でない性能となる。Autellix [Luo25b] はターン終了時退避を用い、マルチターンのエージェントスケジューリングにおける KV キャッシュ保持の重要性を無視する。Pie [Gim25] はインターフェースを公開するが、KV キャッシュ保持を決める方針を提供しない。Ayo [Tan25]、Alto [San24]、Parrot [Qiu24] は静的ワークフローを仮定しており、動的エージェントには適用できない。

効率的かつ堅牢な解決策として、本稿では KV キャッシュの time-to-live 技術によりマルチターンのエージェントワークロードのジョブ完了時間を改善するサービングシステム Continuum を提示する。従来のキャッシュ研究に着想を得て、Continuum は KV キャッシュの TTL 機構を導入し、リクエスト完了後も KV キャッシュを GPU 内に保持して、従来のターン終了時退避を上書きする。推論ステップ中にツール呼び出しを生成する各 LLM リクエストについて、Continuum は prefill/再読み込みコストと、KV キャッシュ保持によるターンごとのキューイング遅延削減の両方をモデル化する。上記二つの要因とツール呼び出し分布に基づく潜在的なヒットの利益を求めた後、Continuum はそれを TTL 期間中の GPU メモリ占有コストと比較し、KV キャッシュが自動退避されるまで GPU メモリに残せる時間を決める。ツール呼び出しが TTL ウィンドウ内に戻れば、次のリクエストは即座に再開でき、prefill とキューイング遅延を削減する。ツール呼び出し予測が不正確で実行時間が予想より長い場合、Continuum は TTL 期限後に KV キャッシュを退避して誤りを堅牢に修正し、深刻なメモリ圧迫やデッドロックを防ぐ。さらに Continuum は TTL 機構をプログラムレベルの first-come-first-serve スケジューリングと組み合わせる。これによりリクエスト順序が改善され、複雑なエージェントワークフローのスケジューリングが単純になる。

Continuum は vLLM 上に実装し、保守やほかの推論エンジンへの統合が容易なモジュール設計とした。Continuum は、リクエストがサービングエンジンへ出入りするたびに呼ばれるツール呼び出しハンドラを実装する。このハンドラはツール呼び出しを識別し、持続時間を予測し、スループットとリクエスト順序の両方を考慮して KV キャッシュ固定のタイムアウトを決定する。このモジュール設計は、推論エンジンの元のスケジューリングロジックへの変更を最小限に抑え、将来のツール呼び出し対応スケジューリングへの拡張を可能にする。

Continuum の性能を評価するため、関数呼び出し [Mao24b] とコーディングエージェント [Lie25] の実際のエージェントワークロードで広範な実験を行った。三つのハードウェアおよびモデル構成を通じて、Continuum はマルチターンのエージェントワークロードで遅延を 1.12 倍から 3.66 倍削減し、スループットを 1.10 倍から 3.22 倍改善する。さらに、Tensormesh の内部テストベッドで Continuum を評価し、実際の SWE-agent ワークロードの遅延を最大 8.18 倍削減できることを示した。今後のエージェントサービング研究を促進するため、trace、コード、エージェントサービング用テストベッドを公開する予定である。

要約すると、本稿の貢献は以下のとおりである。

- エージェントサービングにおける重要な KV キャッシュ保持問題を特定し、より良い解決策の必要性を示す。
- ターン単位の退避コストとターンごとのキューイング遅延を削減する KV キャッシュ time-to-live 機構を備えた、効率的で堅牢なサービングシステム Continuum を設計する。
- 模擬ケースと実ケースの両方で、Continuum が従来手法に比べ遅延とスループットを最大 8.18 倍改善することを示す。
- 収集したエージェント推論 trace、コード、エージェントサービング用テストベッドを論文公開時にオープンソース化する。

<span id="section-02"></span>

## 2 背景

<span id="figure-02"></span>

![SWE-Agent の例](../../papers/continuum/figure-02.png)

**図 2。** SWE-Agent の例。エージェントは途中でツールを呼び出しながら、ソフトウェア工学上のバグを段階的に解決する。これらのツール呼び出しは持続時間が異なり、LLM 推論の連続性を断つ。

### 2.1 エージェントの ReAct パラダイム

現代のエージェント型ワークロードの多くは *ReAct* エージェントループ [Cao23] に従い、LLM がコンテキストを解釈して思考を出力する推論ステップと、外部ツールを呼び出すアクションステップを交互に行う。このパラダイムは事実上の標準となっている。Claude Code [Cod26] や Cursor [Cur25] などのコーディングエージェントは明快さと性能のために採用し、LangChain [Lan25] や LangGraph [Lan25a] などのフレームワークは広く利用可能にし、GPT-OSS [Ope25c] や Kimi-K2 [Kim25a] など最近のオープンウェイトモデルはツール呼び出し能力を基盤モデルへ直接組み込んでいる。

重要な傾向として、エージェントアプリケーションはこのループを*長期・マルチターン*の反復へ拡張し、数十から数百ターンにわたって思考、ツール呼び出し、コンテキスト更新を繰り返し交互に行うようになっている。この傾向は、ツール・エージェント・ユーザー間の対話を対象とする $\tau$-bench [Yao24]、マルチターンのツール拡張対話を対象とする MINT [Wan23a]、マルチターンの意思決定とツール利用シナリオを対象とする AgentBench [Liu23a] など、最近のベンチマークに現れている。

### 2.2 既存手法の限界

従来研究は、主に三つの理由からこの新しい複雑なワークロードを処理できない。

**固定ワークフロー。** 一連の研究は、**事前定義された静的な**計算グラフを持つエージェントワークフローのスケジューリングに注目した。Teola [Tan25] はアプリケーションを primitive レベルのデータフローグラフに分解し、グラフレベルの最適化を適用する。Alto [San24] は分散コンポーネント間のストリーミングとパイプライン実行に注目する。Parrot [Qiu24] は Semantic Variables を通じてアプリケーションレベルのコンテキストを LLM サービスへ公開し、連続する LLM リクエスト間のデータ依存関係をエンジンが推論できるようにする。Teola、Parrot、Alto の共通の制約は、静的または決定論的に定義された DAG を仮定し、依存グラフが実行時に変化する ReAct 型などの**動的エージェントワークロードには対応できない**ことである。この制約により、実際に存在する多様なエージェントを最適化できない [Any24, Lie25, Yan24d]。

**ツール呼び出しを考慮しない。** Autellix [Luo25b] は、エージェントプログラムの累積サービス時間が少ないリクエストを優先する Program-Level Attained Service（PLAS）スケジューリングを導入する。Tempo [Zha25j] は、チャット、エージェント、推論など異なる種類のリクエストに直面した際に SLO を満たすスケジューラを提案するが、本稿は多ターンで変動するツール呼び出しを持つエージェントワークロードに特に焦点を当てる。これらの研究は、持続時間の変動や KV キャッシュ管理への影響など、エージェントワークロードにおけるツール呼び出し固有の特性を考慮しない。[第 3.2 節](#section-03)で後述するように、この見落としは最適でないスケジューリング判断と遅延増加につながりうる。

**不十分な KV キャッシュ保持戦略。** 一部の従来研究は、エージェントワークロードにおける KV キャッシュ再利用の課題を認識していた。InferCept [Abh24a] は、ツール呼び出し間で KV キャッシュを固定する「preserve」操作を導入する。しかし、その方針はリクエストのマルチターン性を見落としている。KV キャッシュがターン間で退避されると、プログラムが戻るたびに追加のキューイング時間が生じる。マルチターンシナリオでは、キューイング時間が各ターンで累積する。この影響を無視すると、大きな利益がある場合でも KV キャッシュを GPU に保持しない。さらに preserve 操作は固定的であり、ツール利用へリアルタイムに適応できない。実際のツール呼び出し時間が予測より大幅に長い場合、KV キャッシュを盲目的に「preserve」すると著しく非効率になる。そのため、実環境への展開は現実的でない。Pie [Gim25] は、生成ループを細粒度のハンドラへ分解するプログラマブルサービングシステムを導入する。制御をユーザープログラムへ委譲し、独自のツール呼び出し処理を可能にする。しかし、開発者が各エージェントのスケジューリングを手作業で設計する必要があり、動的なツール呼び出し遅延やマルチターン依存関係へ適応する実際の方法を提供しない。

<span id="table-01"></span>

![Continuum と代表的なベースラインの比較](../../papers/continuum/table-01.png)

**表 1。** Continuum と代表的なベースラインの比較。

<span id="section-03"></span>

## 3 動機

<span id="figure-03"></span>

![エージェント型ワークロードの特性](../../papers/continuum/figure-03.png)

**図 3。** 評価で使用する SWE-Bench と BFCL のエージェント型ワークロードの特性。ステップ数が増えるほど、リクエストは完了へ近づく。

### 3.1 エージェント実行トレース

まず、現代のエージェント型ワークロードの特性を分析する。SWE-Bench [Nar24] 上で mini-swe-agent [Lie25] を実行した 100 trace と、BFCL V4 Web Search [Mao25] の 100 trace を収集・分析し、いずれも基盤モデルとして GPT-5 を用いた。[図 2](#figure-02)は SWE-Bench の短縮された代表 trace を示し、エージェントがソフトウェア工学タスクを段階的に解決する過程を説明する。

<span id="table-02"></span>

![収集した二つのデータセットの統計](../../papers/continuum/table-02.png)

**表 2。** 収集した二つのデータセットの統計。数値は（平均、標準偏差）の形式で報告する。

要点は三つある。第一に、これらの新しいエージェントプログラムには多くのターンがある。ターン数の増加はスケジューリングをさらに難しくする。第二に、ツール呼び出し時間はさまざまな分布を持つが、多くは短い。これらの短いツール呼び出しが生成されるとリクエストは完了扱いになるものの、ツール呼び出しの完了後すぐに次のリクエストが到着し、KV キャッシュを再利用する。

最後に、[図 3](#figure-03)に示すように、プログラムが完了へ近づくにつれ、両ワークロードで将来の token 数の期待値は全体として減少する。これは、後半のターンほど期待完了時間が短いことを示す。したがって、先に到着したリクエスト（プログラムレベル FCFS）またはより多くのターンを実行したリクエストを優先することは、理論上最適だが将来情報を必要とする shortest remaining time first（SRTF）方針の良い近似になりうる。しかし、ツール呼び出しがあるとこの順序の維持は容易でなく、[第 3.2 節](#section-03)で後述する。

<span id="figure-04"></span>

![CPU オフロード時のプログラムごとのキューイング遅延](../../papers/continuum/figure-04.png)

**図 4。** CPU オフロード時のプログラムごとのキューイング遅延。InferCept の preserve 判断はキューイングコストを無視するため、退避されたプログラムはターンをまたいで大きな待機時間を蓄積する。InferCept が再読み込みを削減しても、vanilla vLLM と同程度である。

<span id="figure-05"></span>

![ロングテールなツール呼び出し実行時間](../../papers/continuum/figure-05.png)

**図 5。** 関数の実行時間は極端なロングテールになりうる。fetch_url の最も遅い 10% が総遅延の 52.5% を占め、cd の最も遅い 10% は 94.1% を占める。

### 3.2 エージェントワークロードの課題

**ターン単位の退避。** ツール呼び出しが短い場合でも、推論エンジンは LLM リクエスト間の一様な空白として扱う。vLLM や SGLang は、デコードが終了するとリクエストの KV キャッシュを即座に退避し、暗黙にリクエスト完了と仮定する。しかし KV キャッシュが退避されていると、エンジンは完全な prefill を再実行するか、オフロードが有効なら DRAM から KV キャッシュを再読み込みする必要があり、追加遅延が生じる。大半のシステムはこのシナリオを効率的に処理できない。[図 1](#figure-01)はこの影響を示す。ツール呼び出しによる休止が KV キャッシュ退避を引き起こし、復帰時に prefill または KV 再読み込みが必要になる。したがって、このオーバーヘッドを避けるにはツール呼び出しを考慮する KV キャッシュ保持方針が重要である。

**ターンごとのキューイング遅延。** エージェントプログラムのマルチターン性は、従来研究が重大に見落としていた新しいスケジューラ上の課題ももたらす。現在のエージェントプログラムがツールを待つ間に、スケジューラがスループット最大化のため GPU メモリを別のリクエストへ割り当てると、現在のプログラムの KV キャッシュは GPU メモリから削除される。ツール呼び出しが戻り、後続の LLM リクエストがスケジューラへ送られると、GPU 空間が空くまで、ほかのリクエストの進行中の prefill/decoding の後ろで待たなければならない。

この待機期間は、KV キャッシュが CPU DRAM に保存されているかどうかにかかわらず、エージェントプログラムの実行に空白を生む。[図 1](#figure-01)に示すように、この空白も従来の prefill/読み込みコストに加えてツール呼び出しによる遅延へ寄与し、ターンをまたいで累積して各プログラムに大きな遅延をもたらす。さらに、プログラム実行の連続性を壊し、早く到着したリクエストを遅く到着したものより後に配置する。待機キュー内の新規リクエストに最高優先度を与えても、GPU 上ですでに進行中のほかのリクエストの計算に阻まれる点に注意されたい。

既存研究は、保持方針でターンごとのキューイング遅延を考慮しない。InferCept [Abh24a] の KV「preserve」操作は、CPU オフロードコストがツール呼び出し中の推定 GPU 占有コストを上回る場合にのみ呼び出される。重要なのは、この判断が直後のターンの*再読み込みコスト*だけを考慮し、退避されたプログラムが待機キューへ再度入った際、ほかのアクティブなリクエストの後ろで経験するキューイング遅延を完全に無視することである。LMCache [Che25e] のようなエンジンが高速な非同期 CPU オフロードを提供すると再読み込みコストが小さくなり、InferCept の preserve 操作はほとんど呼ばれない。しかし、オフロード速度に関係なくキューイング遅延は残る。KV を即座に再読み込みできても、復帰リクエストはほかのリクエストが占有する GPU メモリの解放を待たなければならない。このキューイングコストは*各*ターンで発生するため、累積遅延はプログラム当たりのターン数に比例して増える。これはまさにエージェント型ワークロードが動作する領域である。

[図 4](#figure-04)では、マルチターンスケジューリングを考慮しないことによる性能低下を示す。vanilla vLLM と InferCept アルゴリズムについて、各リクエストが経験する総退避オーバーヘッドをプロファイリングする。横軸は到着順の各エージェントプログラム、縦軸は各エージェントジョブの総 bubble time、すなわち実行前に待機キューで経験する総アイドル時間を表す。InferCept の KV 保持を用いても bubble は残り、vLLM に対してスループットが改善しているにもかかわらず、遅延増加を引き起こす。

**変動するツール呼び出し。** 現在の KV キャッシュ保持方針は、ツール呼び出しの変動が大きい場合にも失敗する。例えば InferCept は、ツール呼び出し後に次のリクエストが到着するまで KV キャッシュを GPU メモリへ固定する。ツール呼び出し遅延が安定していれば、この方法は機能する。しかし[図 5](#figure-05)に示すように、多くのツール呼び出しは実行時間が大きく変動する。ツール呼び出しが予想よりはるかに長いと、固定された KV キャッシュが長時間 GPU メモリを占有しうる。外部ツール呼び出しがより複雑なデータベースエージェントでも、同様のパターンが観測される。その結果、メモリ利用が非効率になり、保持した KV キャッシュが GPU を完全に占有すると潜在的なデッドロックさえ起こりうる。したがって、静的な保持方針は実用的なシナリオで堅牢性を欠く。

<span id="section-04"></span>

## 4 Continuum のスケジューリングアルゴリズム

従来研究の失敗を踏まえ、エージェント型ワークロードのサービングにおける中心的な問いを特定する。すなわち、マルチターンシナリオで KV キャッシュをいかに効率的かつ堅牢に保持するかである。

最適な KV キャッシュ保持方針には、以下の特性が必要である。

- ツール呼び出し後すぐに再利用されるリクエストの KV キャッシュを保持し、prefill/読み込みオーバーヘッドを最小化する。
- エージェントプログラムのマルチターン連続性を考慮し、待機を削減してプログラム順序を保つ。
- 変動するツール呼び出し遅延に対して堅牢である。

<span id="figure-06"></span>

![メモリ利用と遅延の間の time-to-live のトレードオフ](../../papers/continuum/figure-06.png)

**図 6。** メモリ利用と、prefill およびターンごとのキューイング遅延の間でバランスを取るため、time-to-live を適切に設定する必要がある。

堅牢性を保証するため、従来システムの Time-to-live（TTL）という考え方を採用する。各リクエストの KV キャッシュに TTL 値を与え、GPU メモリに残る最大時間を定義する。これにより KV キャッシュを保持しつつ、長時間実行または失敗したツール呼び出しが GPU リソースを無期限に阻害することを防ぐ。

しかし、静的な preserve 操作と比べ、各 KV キャッシュエントリに適切な TTL 値を設定することは難しい。第一に、TTL 値は大きすぎてはならない。[図 6](#figure-06)に示すようにタイムアウトが長すぎると、固定された KV キャッシュが不必要に GPU メモリを占有し、ほかのリクエストを阻害してシステム全体のスループットを低下させる。一方、特定の KV キャッシュの固定時間が短すぎると、ツール呼び出し完了前に KV キャッシュが退避され、GPU 占有時間を無駄にした上で、高価な再計算またはスケジューリング bubble がなお発生する。

これらのトレードオフを考えると、TTL 値は慎重に設定すべきである。ツール呼び出し時間、prefill/読み込みコスト、プログラム連続性の測定に基づき適切な TTL 値を設定して初めて、キャッシュ再利用の利益と、ほかのリクエストに対するシステムスループット維持の必要性を両立し、良好な性能を得られる。

**アルゴリズム 1。Continuum のスケジューリングアルゴリズム**

- **グローバル状態：** 待機キュー $Q$、TTL map $P$（固定されたプログラムと TTL を記録）、ツール呼び出し履歴 $S$。$S[f]$ はツール $f$ について記録されたツール呼び出し情報を表す
- **関数** $\mathrm{OnRequestArrive}(\mathrm{request}\ r)$：
  - $Q \leftarrow Q \cup \{r\}$、$id \leftarrow r$ の Program ID
  - **もし** $id$ が既知のプログラムなら：
    - $(f, t) \leftarrow r$ から得たツール呼び出し情報
    - $S[f] \leftarrow S[f] \cup \{t\}$
- **関数** $\mathrm{OnRequestFinish}(\mathrm{request}\ r)$：
  - **もし** $r$ がそのプログラムの最後のリクエストなら：
    - $r$ が使用する KV キャッシュを解放
  - **そうでなければ：**
    - $f \leftarrow r$ の完了後に呼び出す次のツール
    - $id \leftarrow r$ の Program ID
    - $P[id] \leftarrow \mathrm{CalcTTL}(r, S[f])$
- **関数** $\mathrm{Schedule}()$：
  - **間** $Q$ が空でない：
    - **各** $P.\mathrm{keys}$ の $id$ について：
      - **もし** 現在時刻 $> P[id]$ かつ $id \notin Q.\mathrm{programs}$ なら：
        - $id$ の直前のリクエストが使用する KV キャッシュを解放
        - $P \leftarrow P \setminus (id, P[id])$
    - $r \leftarrow \mathrm{argmax}_{r' \in Q}\ \mathrm{CalcPriority}(r', P)$
    - **もし** $r$ がメモリに収まらないなら：
      - **中断**
    - $Q \leftarrow Q \setminus \{r\}$
    - $r$ を実行へ発行
    - $id \leftarrow r$ の Program ID
    - **もし** $id \in P.\mathrm{keys}$ なら：
      - $P \leftarrow P \setminus (id, P[id])$

### 4.1 効用モデル

<span id="table-03"></span>

![Continuum のコストモデルで用いる主な記号](../../papers/continuum/table-03.png)

**表 3。** リクエスト $r$ と関連ツール呼び出し $f$ に対する Continuum のコストモデルで用いる主な記号。

リクエストの KV キャッシュを固定する有効な TTL 値（秒）を設定するため、Continuum は潜在的な再利用の利益とコストを最もよく釣り合わせる値を選ばなければならない。利益とコストはいずれも時間単位で測定する。最終的に全プログラムの総ジョブ完了遅延の変化へ換算されるためである。数学的には、リクエスト $r$ と TTL 値 $\tau$ が与えられると、Continuum はリクエスト $r$ の KV キャッシュを $\tau$ の間固定するための $\mathrm{Cost}(\tau,r)$ と $\mathrm{Benefit}(r)$ を推定する。

簡単化のため、$\mathrm{Benefit}(r)$ は次のリクエストが TTL ウィンドウ内に到着すると仮定する。ツール呼び出しが戻る前に TTL が期限切れになる場合は、[第 4.2 節](#section-04)で扱う。

**コスト推定。** リクエストの KV キャッシュを固定するコストは、ほかのリクエストのサービングに使えた GPU メモリを占有する機会コストから生じる。

$$\mathrm{Cost}(\tau, r) = \frac{\mathrm{MemUsage}(r)}{\mathcal{M}} \times \tau,$$

ここで $\mathrm{MemUsage}(r)$ はリクエスト $r$ の KV キャッシュが使用する GPU メモリ量、$\mathcal{M}$ はアクティブリクエストの平均 GPU メモリフットプリント、$\tau$ は TTL 値である。

比 $\frac{\mathrm{MemUsage}(r)}{\mathcal{M}}$ は、$r$ を固定したときに平均的なリクエスト何個が阻害されるかを表す。すなわち、$r$ の固定が $k$ 個のリクエストと同じメモリを占めるなら、$r$ の固定は約 $k$ 個の別リクエストへそれぞれ $\tau$ の遅延を加える。KV 保持が必要な場合、この阻害効果が起きるだけのリクエストが常に待機キューにあると仮定する。

**利益推定。** リクエストが TTL 期間内に再発行されると、リクエストの KV キャッシュを固定する利益が実現する。$r$ のプログラムの KV キャッシュを再読み込みまたは prefill するオーバーヘッドを避け、同時にターンごとのキューイング遅延を削減できる。

$$\mathrm{Benefit}(r) = \mathrm{CacheMissCost}(r) + \mathrm{OutofOrderCost}(r)$$

ここで $\mathrm{CacheMissCost}(r)$ はリクエスト $r$ の KV キャッシュを再読み込みまたは prefill するコスト、$\mathrm{OutofOrderCost}(r)$ は、ほかのリクエストが GPU メモリを解放するまで待つためにリクエストが被る期待キューイング遅延を測る。防止されるコストの和を利益とする。

$\mathrm{Cost}(\tau,r)$ と同様に、$\mathrm{CacheMissCost}(r)$ は、(1) コンテキスト再構築オーバーヘッド $\mathrm{Prefill\!-\!Reload}(r)$ と、(2) 追加遅延を受けるリクエスト数の近似 $\frac{\mathrm{MemUsage}(r)}{\mathcal{M}}$ によって測定できる。コストを形式的に次のように定義する。

$$\mathrm{CacheMissCost}(r) = \frac{\mathrm{MemUsage}(r)\times\mathrm{Prefill\!-\!Reload}(r)}{\mathcal{M}}$$

$\mathrm{Prefill\!-\!Reload}(r)$ は、CPU オフロードが有効かに応じた prefill または再読み込みの時間コストである。[第 5.3 節](#section-05)で述べる短いオフラインプロファイリングに基づく。

**期待キューイング遅延の測定。** [第 3.2 節](#section-03)で述べたように、KV キャッシュ保持は、CPU オフロードにより再読み込み自体が高速でも、退避されたプログラムが復帰するときのキューイング遅延も解消する。この $\mathrm{OutofOrderCost}$ 成分は、再読み込みコストしか考慮しない InferCept [Abh24a] など、従来の保持方針に欠ける重要な項である。この項をモデル化することで、キューイング遅延の削減が GPU メモリ占有コストを上回る限り、再読み込みが安価でも Continuum は KV キャッシュ保持を正当化できる。

キューイング遅延の利益は、ワークロードの memoryfulness、*すなわち* プログラムの進行に伴って残りステップ数が予測可能に減るかどうかと密接に関係する。例えば各プログラムのリクエスト数が幾何分布に従うなら、すでに処理した数にかかわらず残りリクエスト数の期待値は一定であり、順序を保っても短いジョブを先に完了できないため、固定によるキューイング遅延の利益はない。一方、各プログラムが固定数のリクエストを発行するなら、TTL は Shortest Job First を近似してキューイングコストを解消できる。

$N$ をプログラムの総リクエスト数、$k$ をすでに処理したリクエスト数とする。次の *memoryfulness factor* を定義する。

$$\eta = -\mathrm{Corr}(k, N - k).$$

この因子がワークロードの memoryfulness の程度をよくモデル化することが分かる。ワークロードが完全に memoryless なら $k$ は $N-k$ と独立であり、$\eta=0$ となる。逆に完全に memoryful、*すなわち* 全プログラムが同じ固定リクエスト数を持つなら、$\mathrm{Corr}(k,N-k)=\mathrm{Corr}(k,-k)=-1$ であり、$\eta=1$ となる。

場合によっては $\eta$ がゼロ未満（極端なロングテールのターン分布）になり、プログラムを進めるほど残り作業が増えて見える *anti-memoryful* パターンを示すことに注意されたい。このパターンは観測しなかったが、Continuum はそのような極端なワークロードも考慮して設計されている。ロングテールのターン分布へ適応するため、各プログラムを短時間だけ処理し、頻繁に切り替えることが望ましい。

以上の $\eta$ に基づき、$\mathrm{OutofOrderCost}(r)$ を定義できる。$\eta=1$ のとき、遅延は $r$ のプログラムが待機キューへ戻った際の待機時間そのものである。これに合わせ、過去のリクエストの単位コンテキストサイズ当たり平均待機時間を $\frac{\mathcal{T}}{\mathcal{M}}$ として記録する。ここで T は以前のリクエストの平均キューイング遅延である。この場合、遅延は $\frac{\mathcal{T}}{\mathcal{M}}\times\mathrm{MemUsage}(r)$ で適切に測定できる。大きなコンテキストのリクエストはスケジュールが難しいため（十分な連続メモリが解放されるまで待つ必要がある）、$\mathrm{MemUsage}(r)$ を考慮する。一般の場合、out-of-order コストを次のように定義する。

$$\mathrm{OutofOrderCost}(r) = \frac{\mathcal{T}}{\mathcal{M}}\times \mathrm{MemUsage}(r) \times \eta.$$

### 4.2 TTL 値の設定

本節では、上のコスト・利益モデルと過去のツール呼び出し情報に基づき、Continuum が KV キャッシュの TTL 値を設定する方法を述べる。アルゴリズム 1（`CalcTTL` 行）のとおり、Continuum は KV キャッシュ保持の期待純利益を最大化する最適 TTL 値 $\tau^{*}$ を決定する。

$$\tau^{*} = \mathrm{argmax}_{\tau}\ \mathcal{P}(\tau, f) \times \mathrm{Benefit}(r) - \mathrm{Cost}(\tau, r),$$

ここで $\mathcal{P}(\tau,f)$ はツール呼び出し $f$ が時間 $\tau$ 以内に完了する確率を推定する。この式は、$r$ の KV キャッシュを $\tau$ の間保持することの、総ジョブ遅延に関する期待純利益を表す。共通項 $\frac{\mathrm{MemUsage}(r)}{\mathcal{M}}$ を消去すると、上式は次に変形できる。

$$\mathrm{argmax}_{\tau}\ \mathcal{P}(\tau, f) \times \big(\mathcal{T}\cdot\eta + \mathrm{Prefill\!-\!Reload}(r)\big) - \tau,$$

したがって実装では $\mathcal{T}$ と $\mathcal{P}(\tau,f)$ だけを追加計算すればよい。$\mathcal{T}$ は、退避されたリクエストが経験したキューイング遅延の sliding-window average として推定できる。次のツール呼び出し時間を完全には予測できないため、過去のツール呼び出し記録 $S[f]$ から得た経験 CDF で $\mathcal{P}(\tau,f)$ を推定する。具体的には次のように計算する。

$$\mathcal{P}(\tau, f) = \frac{1}{|S[f]|} \cdot \sum_{t \in S[f]} \mathbb{I}[t \leq \tau],$$

ここで $\mathbb{I}[\cdot]$ は indicator function である。

最後に、$S[f]$ に記録された一意なツール呼び出し時間を候補（$\tau=0$ を含む）として列挙し、期待報酬が最大のものを選んで式 2 を解く。

**コールドスタート処理。** $S[f]$ の履歴が少ないと、経験 CDF 推定は信頼できない場合がある。この場合、まずグローバルなツール呼び出し情報を使って $\mathcal{P}(\tau,f_{\mathrm{any}})$ を推定する。これは $\sum_{t\in S}\mathbb{I}[t\leq\tau]/|S|$ と計算できる。

さらに、エンジンサービングの開始直後はグローバル記録さえ信頼できない場合がある。そこで Continuum の最小版を設計し、固定 TTL 閾値 $T_{\mathrm{default}}$ を使う。同じコストモデルから、ツール呼び出し時間が平均 1 の指数分布、*すなわち* $\mathrm{ToolCallDuration}\sim\mathrm{Exp}(1)$ に従い、ワークロードが完全に memoryful、*すなわち* $\eta=1$ と仮定して導出する。$T_{\mathrm{default}}$ は、このシナリオで最適な $\tau^{*}$ に設定する。

実際には閾値 $M$ を設け、$S[f]$ に基づき固定 TTL、グローバル記録、細粒度推定のいずれを使うか決める。すなわち $|S|\leq K$ なら $T_{\mathrm{default}}$ を使い、そうでなく $|S[f]|\leq K$ ならグローバル記録を使い、残りは細粒度 TTL 設定を使う。実装では $K=100$ とし、$\mathcal{T}$ をゼロで初期化する。

また、エージェントは通常、本番前にツールを用いて post-training されるため [Cao25, Che25b, Luo25a]、ユーザーは学習中にこれらのコストモデル統計を得ることもできる。

### 4.3 スケジューリング優先度

スケジューリングを TTL アルゴリズムと互換にするため、推論エンジンのリクエスト優先度を再定義する必要がある。Continuum は TTL-aware priority を導入し、プログラムレベル FCFS 順序を保ちながら、TTL 内の固定リクエストを昇格して連続性を維持する。具体的には、待機キュー $Q$ の各リクエスト $r$ に multi-key priority tuple を割り当て、以下の基準を順に用いて順位付けする。

- **preempted status：** 元のエンジンと同様、実行キューの競合により preempt されたリクエストを、されていないものより優先する。
- **TTL status：** そのほかのリクエストでは、TTL ウィンドウ内で保持されたものを固定されていないものより優先する。
- **プログラムレベル到着順：** 最後に、各カテゴリ内でプログラムレベルの到着時刻順に並べ、FCFS fairness を維持する。

<span id="section-05"></span>

## 5 Continuum のシステム設計

<span id="figure-07"></span>

![Continuum のシステム概要](../../papers/continuum/figure-07.png)

**図 7。** Continuum のシステム概要。

Continuum の設計目標は、推論エンジンのスケジューラ中核ループへの変更を最小限に抑えるモジュール型アーキテクチャである。クライアント側では、各推論リクエストへプログラム識別子（`program_id`）を付与し、システムがマルチターンのエージェントプログラムを認識し、ステップをまたいでツール呼び出しを推論できるようにする。

サービングエンジンへ到着すると、リクエストは既存のスケジューラループへ入る。Continuum は、リクエスト到着時と完了時に呼ばれる薄い Tool-Call Handler を追加する。ハンドラは LLM 出力からツール呼び出しを解析し、同じ `program_id` 内で観測したリクエスト間隔を用いてツールごとの遅延を追跡し、TTL をスケジューラへ返す。スケジューラはこのヒントを使って次ステップで再利用できるよう KV キャッシュを固定し、TTL 値が期限切れになるかプログラムが終了すると固定を解除する。

### 5.1 ツール呼び出しハンドラ

ツール呼び出しハンドラは、リクエスト到着後または完了時にメインスケジューラから呼ばれる独立クラスである。この分離構造により、ツール処理ロジックは中核スケジューリングループから隔離され、将来の parser や tool-aware policy へ拡張しやすい。

**ツール呼び出しの識別。** スケジューラがリクエストを完了すると、応答をツール呼び出しハンドラへ転送し、応答にツール invocation が含まれるかを判定する。LLM 出力は OpenAI schema のような標準ツール呼び出し構造をよく採用するため、ハンドラは function call schema に従ってメッセージを解析する。

```json
{
  "id": "fc_0",
  "call_id": "call_0",
  "type": "function_call",
  "name": "get_weather",
  "arguments": {"location": "Paris"}
}
```

この schema では、ハンドラは返された各 message block の `type` を確認し、function/tool call を示す場合は呼び出しの `name` を抽出してツール呼び出し種別として使う。SWE-Bench では、関数呼び出しを含む各 LLM 応答に `bash` 関数呼び出しがちょうど一つ含まれることが保証される。`bash` block 内の文字列を抽出し、その後の最初の単語をツール呼び出し名とする。

異なる LLM の関数呼び出し形式の例 [Lin25, Qwe24] は[付録 B](#section-appendix-b)に示す。Continuum は[付録 A](#section-appendix-a)と同様の parser によって容易に拡張できる。

**ツール完了時刻の記録。** プログラム ID $p$ で識別されるプログラム内の各 LLM リクエスト $i$ について、ツール呼び出し出力を持つ完了リクエストをスケジューラが記録した際、ハンドラはサーバー側完了 timestamp $t_{\mathrm{finish}}^{p,i}$ とツール呼び出し名を記録する。同じ $p$ を持つ次のリクエスト $i+1$ が到着すると、サーバー側到着 timestamp $t_{\mathrm{arrive}}^{p,i+1}$ を観測し、リクエスト間隔 $t_{\mathrm{arrive}}^{p,i+1}-t_{\mathrm{finish}}^{p,i}$ を計算する。この間隔を今回のツール呼び出し実行時間として記録し、将来の TTL 計算に保存する。

### 5.2 スケジューラにおける TTL による効率的な固定

ツール呼び出しハンドラが TTL 値を返した後、スケジューラは固定操作を実行する必要がある。

**リクエストの固定。** そのステップが最終ステップと示されていない場合（例えばツール呼び出しを含むと解析された場合）、スケジューラはツール呼び出しハンドラから TTL 値 $\tau^{*}$ を取得し、ゼロでなければ `pin_request(request, $\tau^{*}$)` を呼ぶ。これにより、リクエストと期限 `current_timestamp + $\tau^{*}$` の組を辞書 `pinned_requests` へ記録し、リクエストの KV block 解放を意図的に省略する。`pinned_requests` は待機キューにも渡され、同じプログラムの次リクエストを優先する。

**リクエストの固定解除。** 各スケジューリングステップの先頭で、スケジューラは `unpin_requests()` を実行する。`pinned_requests` を走査し、TTL が期限切れで、*かつ* `program_id` が現在待機キューに存在しないエントリの固定を解除する。後続リクエストがすでに推論エンジンへ到着しているものの、まだスケジュールできていない場合の早すぎる退避を防ぐ。また、プログラムの最後のステップが完了すると、近い将来の KV キャッシュ再利用はないため、同じ `program_id` を持つ残りの固定を能動的に解除する。

**デッドロック防止。** 固定リクエストは蓄積し、GPU メモリ全体が固定リクエストに占有されると潜在的なデッドロックが発生しうる。同じプログラムの次のリクエストが待機キューにあると固定リクエストが保持されるため、空き容量不足で新しいリクエストを実行できず、スケジューリングループ全体が停止しうる。

したがって、このようなデッドロックが起きたときに固定を解除する機構が必要である。Continuum では、空間競合で新しいリクエストをスケジュールできない場合、`pinned_requests` に固定リクエストがあるか確認する。存在すれば、最初のリクエストを実行可能になるまで、プログラム到着時刻が最も遅いものから victim を反復的に選び、固定を解除して空間を解放する。選択されたリクエストはキューから取り除かれ、KV キャッシュを解放し、必要に応じて再度キューへ入るため、後続割り当てを続行できる。これにより、多数の固定が存在してもデッドロックを防ぐ。

**オフラインプロファイル。** [第 4.1 節](#section-04)で必要な、コンテキストサイズに基づく prefill 時間と再読み込み時間（$\mathrm{Prefill\!-\!Reload}(r)$）を予測するため、各ハードウェア・モデル組についてオフラインプロファイルを行い、オンライン推定に用いる。目的は二つある。**(1)** CPU オフロード時の GPU-CPU 帯域幅。CPU オフロードの平均スループットを測定する。**(2)** prefill コスト推定用の prefill 対コンテキスト長曲線。chunk size $\{1000,2000,4000,... \mathrm{max\_context\_length}\}$ で prefill を実行し、データへ二次曲線を fitting する。リクエストの一部 page が GPU メモリに残り、再計算不要な場合があることは認める。しかし、メモリ競合時に残る page は通常少ないため、完全な prefill 時間で近似しても誤差は小さい。各ハードウェア・モデル組のプロファイリングは 10 分未満で完了する。

<span id="figure-08"></span>

![モデルとハードウェア構成別の Continuum の end-to-end 性能](../../papers/continuum/figure-08.png)

**図 8。** Continuum は、異なるモデル規模、ハードウェア構成、データセットを通じてベースラインスケジューラを上回る。

<span id="figure-09"></span>

![OpenHands における Continuum の性能](../../papers/continuum/figure-09.png)

**図 9。** H100 上で Llama-8B を用いた OpenHands において、Continuum は平均遅延と P95 遅延で最高性能を達成する。

### 5.3 実装

Continuum は約 1,000 行の Python により vLLM 上へ実装した。スケジューラクラスへ追加した上記の固定操作に加え、vLLM の元のスケジューラでツール呼び出しハンドラの三つの関数を使う。

- `func_call_finish(tool, timestamp):` リクエストが完了してツール呼び出しを含むと解析されたとき、ツール呼び出しハンドラへ開始時刻の記録を通知する。
- `update_tool_call_time(program_id, timestamp):` 新しいリクエストの到着は、前リクエストのツール呼び出し完了を意味するため、その時刻を記録する。
- `set_up_ttl(request, tool):` 過去のツール呼び出し情報とシステム設定に基づき、完了したリクエストに対してスケジューラが使う最適 TTL 値を返す。

<span id="section-06"></span>

## 6 評価

<span id="figure-10"></span>

![DRAM オフロードを有効にした end-to-end 評価](../../papers/continuum/figure-10.png)

**図 10。** DRAM オフロードが有効な場合も Continuum は一貫した改善を達成する。ツール呼び出しとマルチターンを併せて考慮し、InferCept のような高度な DRAM オフロードロジックを持つシステムを上回る。

評価から得た主な要点は次のとおりである。

- **遅延削減。** Continuum はインテリジェントな KV キャッシュ固定により、ベースラインスケジューラに対して大幅な遅延削減を達成する
- **堅牢な改善。** Continuum はターン数や異なるオフロードシナリオを通じてベースラインを上回る。
- **すぐに使える有用性。** Continuum は品質を低下させずに実際のエージェントを高速化できる。

### 6.1 実験設定

**モデルとハードウェア。** Llama-3.1-8B、Llama-3.1-70B、Gemma-3-12B を用いて Continuum を評価する。Runpod の A100-SXM GPU、AWS および Tensormesh の H100、オンプレミスサーバーの B200 GPU を使用する。

**データセット。** [図 12](#figure-12)の実 SWE-Bench 実験以外では、GPT-5 [+1] で実行して収集した二つのワークロードを評価し、エージェントプログラムの到着パターンには Poisson 分布を用いる。

- SWE-Bench [Jim23]：SWE-Bench 上で mini-swe-agent [Lie25] [+2] を実行する。リクエストはコンテキストウィンドウ内に収める。
- Berkeley Function Calling Leaderboard [Mao25]：最新版 BFCL V4（Web Search カテゴリ）を使用する。Web browsing tool で質問に回答するエージェントを含む。llama-3.1 のコンテキストウィンドウ（128k token）へ少なくとも 100 リクエストを収めるため、ワークロードを 0.4 倍に縮小した。
- OpenHand [Oth24]：OpenHands は人気のオープンソースコーディングエージェントである。公式リポジトリの Go 言語向け multi-SWE-bench [Zan25] 例を実行する。

**主なベースライン。**

- *Vanilla vLLM* vllm 0.10.2 の安定版を既定設定で用い、chunk size 2048 を有効にする。
- *CPU DRAM offloading* vllm 0.10.2 と LMCache 0.3.7 [Che25e] を用いる。A100 GPU ではオフロードに使う DRAM を 100GB、B200 と H100 GPU では GPU 当たり 200GB に設定する。以下のアルゴリズムにも適用する。
- *Autellix* Autellix [Luo25b] の PLAS アルゴリズムを vllm 上に実装する。LMCache を有効にし、CPU オフロードの場合へ拡張する（Autellix+）。
- *InferCept* InferCept [Abh24a] の selective preserve、swap、evict アルゴリズムを vllm + lmcache 上に実装する。LMCache の CPU オフロードは non-blocking（元の InferCept より優れる）なので、コスト推定を対応して更新する。
- *Distributed Inference* 実エージェント実験では、native cache-aware routing を備える SGLang 0.5.5.post3 [Sgl25a]、PD Disaggregation 用に 1P1D で構成した Nvidia Dynamo 0.7.0.post1 [Dyn25] などのオープンソース手法と比較する。

<span id="figure-11"></span>

![P90 と P95 の遅延比較](../../papers/continuum/figure-11.png)

**図 11。** Llama-8B モデルで SWE Bench trace を実行した場合、Continuum は P90 と P95 の遅延を改善する。

<span id="figure-12"></span>

![実 SWE-agent の比較](../../papers/continuum/figure-12.png)

**図 12。** 分散環境の実 SWE-agent において、Continuum は同じ pass rate で遅延を改善する。

### 6.2 エンドツーエンド実験

SWE-Bench、BFCL、OpenHands のワークロードで trace replay 実験を行う。[図 8](#figure-08)、[図 10](#figure-10)、[図 9](#figure-09)は Continuum の end-to-end 改善を示す。BFCL と SWE-Bench の両ワークロードで、平均応答時間とスループットが大幅に改善する。例えば Llama-3.1-8B モデルでは、vanilla vLLM ベースラインに対して平均応答時間を最大 2 倍削減する。性能向上は異なるモデル規模とハードウェア構成を通じて一貫し、多様なシナリオでの有効性を示す。Autellix は BFCL でベースラインを上回るが、実行時間が長いリクエストは期待終了時間も長いという誤った仮定のため、SWE-Bench では下回る。

1 秒当たりのジョブ数は、従来の LLM サービング論文で報告された値より少ないことに注意されたい。エージェント型ワークロードははるかに複雑で、10 回を超える LLM 推論リクエストを伴うことが多く、計算負荷が高いためである。

評価をほかの実用エージェントにも拡張した。[図 9](#figure-09)に示すように、AWS の H100 一基で Llama 8B を用いて OpenHands を実行すると、遅延を改善できる。平均ターン数が多いため、ベースラインが高いターン数で劣化し、本手法の改善はさらに顕著になる。

また、Continuum は CPU オフロードベースラインを一貫して上回る。一方、CPU オフロード上での PLAS の利得はベースラインに対して縮小する。これは、Continuum によるスケジューリング bubble 削減の堅牢な性能改善が DRAM オフロード技術と直交することを示す。

[図 11](#figure-11)では、ターンごとのキューイング遅延をベースラインより削減できるため、Continuum が P90 と P95 の遅延を改善することを示す。各点は、CPU オフロードを GPU 当たり 200GB に設定し、B200 一基で Llama-8B モデルを実行した構成である。

<span id="figure-13"></span>

![batch size と chunk size に対する感度](../../papers/continuum/figure-13.png)

**図 13。** Continuum は異なる最大 batch size と chunk size の構成を通じて遅延を改善する。

**分散環境における実 SWE-Agent。** 大規模な実展開シナリオで Continuum の性能を完全に評価する。Tensormesh の内部 H100 テストベッドで、SWE-Bench-Verified の 500 タスクに対して Continuum で実 SWE agent を実行する。Poisson 分布でエージェントを配布する job distributor を SWE-Bench プラットフォームへ追加し、エージェントクライアント環境を構築する。Continuum には単純な session-aware routing を用い、ほかの分散推論手法と比較する。ジョブごとの完了時間を測定し、生成後の結果について SWE-bench 上の各エージェントプログラムの pass rate を収集する。

[図 12](#figure-12)に示すように、pass rate が等しいとき、Continuum は平均遅延で一貫してベースラインを上回る。Continuum の pass rate は実際にはベースラインより高い。SWE-Bench が環境 docker の hang を防ぐ制限時間を持つためである。ベースラインの実行時間が 15 分を超えると preempt され、失敗として扱われる。この結果は、実際の本番環境における Continuum の有用性を示す。

### 6.3 感度分析

**推論エンジン構成の変更。** Continuum が推論エンジン構成の変化に堅牢であることを示すため、異なる構成で評価する。[図 13](#figure-13)では 1 秒当たりのジョブ数を 0.13 とし、最大 batch size を変えて Continuum と各ベースラインを比較する。Continuum の改善は異なる batch size でも安定している。さらに[図 13](#figure-13)では chunk size を 256 から 4096 まで変え、異なるサイズで同様の改善を観測する。これは、異なる推論エンジン構成に対する手法の堅牢性を示す。

**ターン数に対するスケーリング則。** [図 14](#figure-14)はマルチターンシナリオにおけるスケジューラの堅牢性を評価する。SWE-Bench の trace を反復（1$\times$ から 5$\times$）し、token length を反比例して縮小することで、総 token をコンテキストウィンドウ内に収めながらターン数が多いシナリオを模擬する。リクエスト率 0.13 JPS、DRAM オフロード 200 GB では、ターン数が増えるにつれベースライン手法が劣化する。ターン増加によりツール呼び出しが増え、総実行時間も長くなり、従来手法のスケジューリング課題が悪化するためである。対照的に、本手法は安定した低遅延性能を維持し、複雑で多ターンのエージェント対話に有効である。

**SSD オフロード。** CPU オフロードと同様に、SSD オフロードはより大きな空間を提供するが、読み込みは遅い。B200 上の llama-8B と SWE-bench ワークロードを用い、LMCache により CPU オフロードを超えて SSD storage layer を拡張した場合の Continuum を評価する。[図 15](#figure-15)に示すように、異なる容量の disk を併用しても、Continuum はベースラインより平均遅延を一貫して改善する。

<span id="figure-14"></span>

![ターン数増加に対する堅牢性](../../papers/continuum/figure-14.png)

**図 14。** ターン数が増えるほど Continuum の改善は大きくなり、遅延時間は安定したままである。

<span id="figure-15"></span>

![SSD オフロードの比較](../../papers/continuum/figure-15.png)

**図 15。** オフロード先を CPU から SSD へ拡張した場合も、Continuum は遅延を削減する。

### 6.4 アブレーション研究とマイクロベンチマーク

**アブレーション研究。** コストモデルが Continuum 全体の性能へ与える影響を分析する。[図 16](#figure-16)では、Continuum を一部の最適化だけを適用するベースラインと比較する。Program-Level FCFS は、vLLM の元の request-level FCFS を、プログラム到着に基づく優先度へ変更する。Static TTL は Program-Level FCFS 上に構築し、cold-start handling で推定した固定 TTL 閾値を使用する。結果のとおり、Continuum の各アイデアが段階的に性能を改善する。

**スケジューラオーバーヘッド。** [表 4](#table-04)に示すように、本手法はベースラインよりわずかなスケジューリングオーバーヘッドを追加する。しかし、このオーバーヘッドは 1 桁ミリ秒の範囲であり、LLM 推論の GPU 実行時間と比べ無視できる。スケジューリング戦略による大幅な end-to-end 性能向上は、この小さな遅延増加を大きく上回る。

**強化学習への応用。** Continuum の強化学習への利用可能性について micro-benchmark も行った。Multi-SWE bench [Zan25] 上で GLM-4.5-fp8 を学習する OpenHands Agent を rollout generation に用いた。ハードウェア構成は 8xH100 node である。原論文が報告する 1 分当たり推論ステップ数について、同時期の RL 研究 ThunderAgent [Kan26] と比較した。[表 5](#table-05)に示すように、Continuum は single-node rollout で高いスループットを達成する。

<span id="figure-16"></span>

![Continuum のスケジューリング要素のアブレーション](../../papers/continuum/figure-16.png)

**図 16。** Continuum の各アイデアの寄与。Program-level FCFS はリクエストではなく、プログラムの到着が早いリクエストを優先する。Static TTL は cold-start handling 機構から計算した固定 TTL 閾値を使用する。

<span id="table-04"></span>

![DRAM オフロード時のスケジューリング遅延オーバーヘッド](../../papers/continuum/table-04.png)

**表 4。** 異なる DRAM オフロード設定で、Continuum はわずかなスケジューリング遅延オーバーヘッドを追加する。

<span id="table-05"></span>

![OpenHands rollout のスループット](../../papers/continuum/table-05.png)

**表 5。** Continuum は OpenHands rollout で同時期の研究を上回る。

<span id="section-07"></span>

## 7 関連研究

**LLM 推論システム。** LLM 推論改善に関する研究は多い。vLLM [Kwo23] や SGLang [She24] などのサービングエンジンは、paged attention design と最適化 kernel を採用して最先端の推論を達成する。GPU 実行速度を改善する広範な kernel-level optimization [Ye25, Dao22, Zhu25a] に加え、resource management にも continuous batching [Yu22a]、chunked prefill [Ram24]、skip-join multi-level scheduling [Wu23a] など多くの最適化が提案されている。その多くが推論エンジンへ移植されている。

従来研究は CPU DRAM や disk への効率的なオフロードも検討している [Gao24a, Xie25, Che25e, Liu24d, Yao25]。分散推論では session-aware routing [Sri24, Vll25]、KV-cache-aware routing [Xia25]、prefill-decode disaggregation [Zho24] が採用されている。これらに基づき、Continuum は LLM 推論を長期のマルチターンエージェントワークロードへ拡張し、異なるリクエストがリソースを競合する場合の resource management を改善する。

**コンピュータシステムにおける Time-to-live 機構。** Time-to-live（TTL）はコンピュータシステム設計で長年使われる abstraction であり、DNS resolver、分散キャッシュ、CDN edge node、consistency protocol で、staleness を制限しリソースの無期限保持を防ぐために広く用いられる [Kri01, Jun03, Coh05, Nis13, Bas18a, Mou19, Law20, Yan21a, Her21, Hen24]。これらの環境で TTL は coarse-grained validity window として、予測できない更新または fetch 遅延下で freshness、load、robustness のバランスを取る。本研究はこの系譜を引き継ぎつつ、LLM 推論エンジン内の細粒度 resource management という新領域へ TTL を拡張する。entry が独立し、correctness constraint が性能上重要というより semantic である従来の TTL 利用と異なり、KV キャッシュは LLM サービングエンジンの GPU memory pressure、prefill cost、scheduling fairness と密接に相互作用する。著者らの知る限り、Continuum は予測ツール呼び出し時間、スケジューリング側の遅延伝播、ワークロードパターンに応じて TTL で LLM KV キャッシュを制御する初のシステムである。

**ReAct 型エージェントを超える一般性。** Continuum の現在の設計は、各 LLM ステップが明確なツール invocation を返し、次ステップまで空白がある ReAct 型 tool-interleaving agent 向けに最適化されている。順次の「reason -> tool -> reason」というリズムを保つため、Continuum は並列ツール呼び出しへ自然に拡張できる。しかし、新しいエージェントフレームワークには speculative branch、非同期 multi-agent coordination、context folding など非線形 control flow が含まれうる。このワークロードはほぼ実験段階で、実際の本番ワークロードではまだ試されていないが、推論パターンが逐次フローに反し、将来の変更を必要とする可能性がある。この種のワークロードへ Continuum を拡張することは重要な今後の課題である。詳細は[付録 C.1](#section-appendix-c1)で議論する。

<span id="section-08"></span>

## 8 結論

頻繁なツール呼び出し、ステップ間遅延の大きな変動、マルチターン連続性を保つ必要性により、エージェント型ワークロードは LLM サービングシステムへ新たなスケジューリング課題をもたらす。本稿では、time-to-live 機構によりキャッシュ再利用の利益と GPU メモリ阻害コストを両立する KV キャッシュ保持・スケジューリングシステム Continuum を提示した。TTL ベースの固定をプログラムレベル FCFS と統合し、Continuum は不要な prefill を減らし、ターンごとのキューイング遅延を緩和し、予測不能なツール呼び出し遅延へ堅牢に適応する。vLLM 上の実装は、モデル規模、ハードウェア構成、実世界のエージェントワークロードを通じて、end-to-end ジョブ完了時間を一貫して改善する。Continuum は、原則に基づく tool-aware KV management が効率的なマルチターンエージェントサービングに不可欠であることを示す。将来のシステムがエージェントワークロードを LLM 推論エンジンへ深く統合する基礎となることを期待する。

<span id="section-appendix-a"></span>

## 付録 A ツール呼び出しパーサの実装例

mini-SWE-agent 用ツール parser の実装を以下に示す。

```python
class ToolCallParser:
  """LLM 出力から関数呼び出しを抽出する parser。

  mini-swe-agent と同じ parsing logic を使って Markdown code block から bash command を抽出し、
  関数呼び出しを識別する。

  異なる parsing logic を持つ別の dataset にも拡張できる。
  """

  def parse(self, text: str) -> Optional[str]:
    """LLM 出力を解析し、関数呼び出し名を抽出する。

    引数：
      text：LLM の出力 text

    戻り値：
      関数呼び出し名（例："ls"、"cd"、"git"）。見つからなければ None
    """
    # mini-swe-agent と同じ正規表現 pattern：r"```bash\s*\n(.*?)\n```"
    actions = re.findall(r"```bash\s*\n(.*?)\n```", text, re.DOTALL)

    if len(actions) == 1:
      bash_action = actions[0].strip()
      # action の最初の単語（command）を抽出
      words = bash_action.split()
      if words:
        return words[0]

    return None
```

**リスト 1。** ツール呼び出し parser の例

<span id="section-appendix-b"></span>

## 付録 B その他の関数呼び出し例

内部では、モデルごとに chat template と生成におけるツール呼び出しの表現が異なる。例えば Llama-3 系は function-style string `func_name(param_1=val_1, param_2=val_2, ...)` を出力しうる一方、Qwen-3 系は `{"name": "func_name", "arguments": {...}}` を使う。形式にかかわらず、サービングエンジン（例えば vLLM、SGLang）はモデル固有で template-aware な parser を持ち、生成された長い文字列から関数名と引数を復元し、OpenAI-style schema に正規化して downstream で統一的に扱えるようにする。したがって、サービングエンジンが提供する一般的な function calling interface を使うなら、モデル固有の parsing を気にする必要はない。

アプリケーションが function calling interface を使わず、chat interface 経由で構造化 bash command の出力をモデルに求める場合も、関数名と引数は容易に解析できる。例えば SWE Bench で意図した tool invocation を抽出するには、唯一の bash code block を見つけ、command string を `&&` または `||` で分割し、各 sub-command を解析する。最初の token が executable/function name（pytest、git、…）で、残りが引数である。

```shell
pytest -q && git add -A && git commit -m "fix: handle None case in parser"
```

Terminal Bench では、structured format が command splitting をすでに処理しているため、さらに容易である。

```json
{
  "state_analysis": "The tests are failing with a NameError.",
  "explanation": "Open the file, fix the missing import and rerun tests.",
  "commands": [
    { "keystrokes": "vim src/app/main.py\n", "is_blocking": false, "timeout_sec": 2.0 },
    { "keystrokes": "pytest -q\n", "is_blocking": true, "timeout_sec": 30.0 }
  ],
  "is_task_complete": false
}
```

<span id="section-appendix-c"></span>

## 付録 C 関連研究に関する拡張議論

<span id="section-appendix-c1"></span>

### C.1 新しいツール呼び出し形式

**ツールを用いた思考。** このパターンは planning と execution を交互に行う。モデルは構造化された中間 plan を出力し、ツールを呼び出し、feedback を統合し、chain of thought を継続する [Ope25c, Gao24c, Wu25a, Che23a]。Continuum ではツール呼び出しが出力されると現在のリクエストを完了とみなし、ツール完了後、更新されたコンテキストを持つ後続リクエストを enqueue する。[付録 A](#section-appendix-a)に示すツール parser を実装すれば、このシナリオへ Continuum を拡張できる。

**並列ツール呼び出し。** sub-task が独立している場合（例えば「米国と英国の天気はどうか？」）、複数のツール呼び出しを並列に発行するとターン遅延を短縮できる [Kim24a, Ant25b, Ope23d, Mao24a, Yan24d, Pat25a]。設計上、これらの呼び出しは可換であり、任意の順序で実行し、応答が完了するたびコンテキストへ追加できる。クライアントからの function call predictor により Continuum を拡張できる。

**非同期ツール。** 非同期ツール呼び出しは実行を non-blocking にする。各呼び出しはモデルが後で await できる handle（*future*/promise）を返し、ツールが background で動く間も生成を続けられる [Gim24a, Gin24, Ope25d]。これは breadth-first または tree-search behavior（例えば複数の probe を並行して fan-out する deep-research や browsing agent）に特に有用である。このワークロードは Continuum に適している。モデルは await 間に active computation をほぼ行わないため、早すぎる退避を避ければ KV-cache reuse は高い。

### C.2 モデルアーキテクチャ

従来の decode-only transformer を超える新しい LLM architecture が提案されている。Mix-of-Experts（MoE）[Sha17, Fed22, Cho22b] は、入力 token ごとに一部の parameter だけを activate して sparsity を導入し、低い推論コストで大きなモデルを実現する。Sliding-window transformer [Bel20, Zah20] は attention scope を全コンテキストでなく local window に制限し、推論時の memory footprint を削減する。Hybrid Model は full attention と、linear attention [Cho20a, Kat20]、SSM [Gu23, Gu22, Gu20, Gu21]、low-rank attention [Wan20a] などの高効率 attention mechanism を組み合わせ、memory footprint を削減して推論速度を高める。これらの architecture は推論時の memory bottleneck を緩和してスループットを高めるが、本稿で議論したスケジューリング問題、特に異なるジョブが GPU 空間を継続的に競合することによる scheduling bubble は依然として発生する。

<span id="section-appendix-d"></span>

## 付録 D 制約と今後の課題

**TTL コストモデルの感度。** Continuum は、経験的なツール呼び出し CDF、memory usage estimate、「memoryfulness」factor を組み合わせて最適 TTL 値を導出する cost-benefit model に依存する。この設計は原則に基づくものの、過去 sample が予測に有効な程度にツール呼び出し分布とワークロード特性が安定していると仮定する。backend contention や外部 API の変動でツール遅延が急変するエージェントなど、高度に変動的または adversarial なワークロードでは、モデルが最適でない TTL を生成し、一時的にスケジューリング効率を低下させる場合がある。さらに memoryfulness factor $\eta$ や $\mathrm{CacheMissCost}()$ と $\mathrm{OutOfOrderCost}()$ の近似などの主要 parameter は、同じワークロードの過去ターンでの観測に依存し、未知のエージェント挙動へ generalize できない可能性がある。エージェントは通常事前に post-training されるため、Continuum は学習中の分布を cold start 処理に使って緩和できる。エージェントにおける突然の分布変化への対処は今後の課題とする。

[+1]: ワークフローがほぼ正しく生成されるよう、より高いモデル能力を持つ GPT-5 を使用する。小さな基盤モデルはしばしばタスクを完了できない。

[+2]: SWE-bench の公式エージェントで、4 月 13日時点の leaderboard で第 5 位である。
