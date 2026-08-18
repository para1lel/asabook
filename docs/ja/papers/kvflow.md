---
title: 'KVFlow: Efficient Prefix Caching'
createTime: 2026/08/19 00:00:00
permalink: /ja/papers/kvflow/
---

> [Zaifeng Pan](https://panzaifeng.github.io/)、[Ajjkumar Patel](https://dblp.org/pid/412/9471)、[Zhengding Hu](https://dblp.org/pid/359/5899) [+corresponding-author]、[Yipeng Shen](https://dblp.org/pid/04/4092)、[Yue Guan](https://dblp.org/pid/54/7820-3)、[Wan-Lu Li](https://dblp.org/pid/412/8586)、[Lianhui Qin](https://lianhui.ucsd.edu/)、[Yida Wang](https://yidawang.org/)、[Yufei Ding](https://yufeiding.ucsd.edu/)。arXiv 初回投稿は 2025 年 7 月 10 日、現行版は v1。[NeurIPS 2025](https://neurips.cc/virtual/2025/loc/san-diego/poster/119883) 採択。[KVFlow: Efficient Prefix Caching for Accelerating LLM-Based Multi-Agent Workflows](https://arxiv.org/abs/2507.07400v1)。[原論文 PDF](/paper/kvflow.pdf)。[DOI](https://doi.org/10.48550/arXiv.2507.07400)。[TeX ソース](https://arxiv.org/src/2507.07400v1)。正確な印刷レイアウトと参考文献については、原論文 PDF を正本とする。

## Abstract

大規模言語モデル（LLM）ベースのエージェントワークフローは、複雑なタスクを解くために複数の専門エージェントを連携させる一般的なパラダイムとなっている。サービング効率を高めるため、既存の LLM システムは prefix caching を用いてエージェントの固定 prompt に対応する key-value（KV）tensor を再利用し、反復呼び出しでの冗長な計算を避ける。しかし、現在のシステムは通常 Least Recently Used（LRU）方針で KV cache を退避するため、将来のエージェント利用を予測できず、再利用の直前に KV cache を破棄することが多い。その結果、cache miss が頻発し、再計算または swapping の大きなオーバーヘッドが生じる。

本稿では、エージェントワークロード向けのワークフロー対応 KV cache 管理フレームワーク KVFlow を提示する。KVFlow はエージェントの実行スケジュールを Agent Step Graph として抽象化し、各エージェントに将来の起動までの時間的な近さを推定する steps-to-execution 値を割り当てる。これらの値が KV node レベルの細粒度な退避方針を導くことで、KVFlow は再利用される可能性が高い entry を保持し、木構造 cache の共有 prefix を効率的に管理できる。さらに KVFlow は、次の step で実行予定のエージェントに必要な tensor を background thread で CPU から GPU へ先行して読み込む、完全に overlap した KV prefetching 機構を導入し、生成中の cache miss による停止を回避する。階層型 radix cache を備えた SGLang と比べ、KVFlow は大きな prompt を持つ単一ワークフローで最大 1.83$\times$、多数のワークフローが並行する状況で最大 2.19$\times$ の高速化を達成する。

<span id="section-01"></span>

## 1 Introduction

LLM ベースのエージェントワークフローは、固定 prompt で定義され特定の subtask を担う複数の専門エージェントを連携させ、複雑な問題をモジュール化され解釈可能な方法で解く [Cao23, Shi23b, Hon23, Li23s, Wan24l]。たとえば MetaGPT [Hon23] は、Product Manager や Engineer といった software engineering の役割を中心にエージェントの協調を構成する。この設計は再利用性と一貫性を高める一方、ワークフロー全体を通じて各エージェントの LLM を繰り返し呼び出す必要があるため、推論遅延も大きくなる。

このオーバーヘッドを軽減するため、既存のエージェントフレームワークとアプリケーション [Hon23, Wu23c, Zhu24a, Zha24l, Pan24c, He25a] は、システムレベルの最適化を備えた LLM サービングシステム [Kwo23, Zhe24, Ten23] に依存している。広く用いられる手法が prefix caching [Zhe24, Aut25] であり、静的 prompt token に対して self-attention layer が生成した key-value（KV）tensor を、decoding step や request をまたいで再利用する。各エージェントは名前、責務、行動特性を指定する固定 prompt で初期化されるため、この手法はエージェントワークフローで特に有効である。これらの prompt は iteration をまたいで変わらないため、prefix caching により静的内容の冗長な計算を避け、エージェントごとの推論遅延を大きく短縮できる。

しかし、GPU memory が限られる場合、prefix caching だけでは不十分である。既存システムは通常、直近でアクセスされていない KV cache を Least Recently Used（LRU）方針で退避する。この方針はエージェントワークフローで最適でない性能を招きうることが分かった。たとえば[図 1](#figure-01)に示すように、4 個のエージェントを反復的に呼び出す逐次実行 pipeline として構成したワークフロー [Wan24l] を考える。図の Executor エージェントが実行されている間、Expresser の KV cache は直近でアクセスされていないため、LRU 方針はこれを退避候補とみなす。すぐに再利用されるにもかかわらず、ワークフローが Expresser エージェントへ進むと cache miss が発生する。この退避動作は不要な再計算を生じさせ、エージェント実行全体の効率を低下させる。

エージェントワークフローにおける既存 LLM サービングシステムの限界に対処するため、本稿ではワークフロー対応 KV cache 管理フレームワーク KVFlow を提示する。まず、エージェント間の実行依存関係を捉え、条件分岐や同期 barrier を含む幅広いワークフロー構造に対応する柔軟な抽象化、*Agent Step Graph* を導入する。graph 内の各エージェント node には、そのエージェントが実行されるまでの近さを推定する、計算済みの *steps-to-execution* 値が関連付けられる。この値は graph 上を伝播する step aggregation function から導出され、KVFlow が動的で構造化された実行 pattern を推論できるようにする。

実行時、KVFlow はこの情報を二つの主要な方法で cache 動作の最適化に利用する。第一に、LRU の代わりに、steps-to-execution が大きいエージェントの KV cache を優先して退避するワークフロー対応方針を採用する。複数のエージェントが木構造 cache を介して共通 prefix を共有できるため、さらに cache node レベルで退避優先度を割り当て、細粒度かつ効率的に管理する。第二に、Agent Step Graph から次に呼び出されるエージェントを予測できることを利用し、必要な KV tensor を CPU から GPU へ事前に読み込む、完全に overlap した KV prefetching 機構を導入する。これにより、生成を停止させずに prefix cache miss を実質的に除去できる。これらの最適化により、cache 効率が向上し、エージェントワークフローの実行遅延が短縮される。

<span id="figure-01"></span>

![循環型エージェントワークフローと LRU prefix cache miss](../../papers/kvflow/figure-01.png)

**図 1。** Planner、Executor、Expresser、Reviewer の 4 エージェントからなる循環型エージェントワークフローの抽象化であり、[Wan24l] から改変した。timestamp 13 では Executor が active となり、その KV cache が更新されるため、LRU 方針により Expresser の cache が退避される。timestamp 14 で Expresser が再び active になると cache miss が発生し、prefill latency が増加する。

要約すると、本稿の貢献は次のとおりである。

- 既存の LLM サービングシステムでは、広く使われる LRU ベースの KV cache 退避方針がエージェントワークフローで最適でない性能をもたらすという、根本的な非効率を明らかにする。
- エージェントの実行順序に基づいて退避を優先付けし、完全に overlap した prefetching によって cache miss のオーバーヘッドを除去する、ワークフロー対応 KV cache 管理最適化 KVFlow を提案する。
- KVFlow を包括的に評価し、cache miss のオーバーヘッドを大きく削減できることを示す。階層型 radix cache を備えた SGLang と比べ、大きな prompt の単一ワークフローと多数の並行ワークフローで、それぞれ最大 1.83$\times$ と 2.19$\times$ の高速化を達成する。

<span id="section-02"></span>

## 2 Background

**LLM サービングシステムの prefix caching。** 細粒度な prefix 再利用を可能にし、冗長な保存をなくすため、現代の LLM サービングシステム [Kwo23, Zhe24] は GPU 上の KV cache を木構造に編成し、各 node に一連の token と対応する KV tensor を保存する。新しい request を受信すると、システムは木の root から prefix を照合し、照合した path 上の KV tensor を連結して、cache 済み prefix 全体を再構築する。GPU memory が不足すると、システムは LRU 方針に基づいて node を退避する。memory の枯渇には二つの原因がある。一般的な状況の一つは、大量の user request が並行してそれぞれ異なるエージェントワークフローを実行し、active な KV cache entry が多数生じる場合である。もう一つは、ハードウェア容量が限られる一方で、エージェントの prompt が非常に大きい場合である。[図 2(a)](#figure-02)に示すように、単一 request の KV cache size は prefix length とともに急速に増大し、context が長くなるほど memory pressure が高まる。さらに CPU memory を二次 cache layer として設定し、退避された KV tensor を backup することで、PCIe を介した cache swapping が可能になる。PCIe latency は生じるものの、swapping は KV tensor の再計算より大幅に高速である [Jin24a, Gao24a]。[図 2(b)](#figure-02)は、PCIe ベースの KV cache 転送に要する時間と prefill 計算時間を比較し、memory pressure 下では CPU memory への offload が効率的な方策であることを確認している。

<span id="figure-02"></span>

![context length に対する KV cache size、prefill latency、PCIe 転送時間](../../papers/kvflow/figure-02.png)

**図 2。** context length を変化させたときの KV cache の特性。（a）KV cache size は token 数に対して線形に増加し、memory 使用量が増える。（b）PCIe 経由の KV cache 転送時間は、再計算に必要な時間より常に大幅に短い。

**エージェントワークフロー。** エージェントワークフロー [Zha24l, He25a, Pan24c, Zhu24a] は、複雑なタスクを協調して解くため、複数のエージェントを実行 graph として構成する LLM アプリケーションパラダイムである。完全自律型エージェント [Par23, Wan24k, Neu24] と比べ、エージェントワークフローは人間の領域知識を活用し、多様なタスクでより一貫性と堅牢性の高い性能を得る [Hon23, Rid24, Wan23e, Zha24m, Qia23, Mor23a]。各エージェントの実行には通常、一回または複数回の LLM 呼び出しが含まれ、prompt は*固定*部分とタスク固有の*動的*部分からなる。固定部分は通常、エージェントの役割、行動指示、タスク記述、few-shot learning の例を encode し、非常に大きくなる場合がある。たとえば [Zha24m] の TestBench Agent と RTL Generator Agent の固定 prompt には長い few-shot learning の例が含まれ、それぞれ 3000 token と 1000 token を超える。そのため、固定部分に対応する KV を cache すれば prefill latency を大幅に短縮し、ワークフロー全体の実行効率を高められる。一方、動的部分には user が入力した質問や指示が含まれることが多く、cache の価値は低い。

<span id="section-03"></span>

## 3 Design of KVFlow

本節では、二つの主要な手法によってエージェントワークフローの prefix cache 管理を改善する KVFlow の設計を示す。第一に、将来の利用に基づいて KV node を優先付けし、既定の LRU 方針を改善するワークフロー対応退避方針を導入する。第二に、先行読み込みと状態対応 scheduling により CPU-GPU 転送 latency を隠す、overlap KV prefetching 機構を提案する。

<span id="section-03-01"></span>

### 3.1 Workflow-Aware Eviction Policy

既存の LLM サービングシステムは通常 LRU 退避方針を採用するが、これはエージェントワークフローでは最適でなくなる。具体的には、まもなく実行されるエージェントが長時間 idle だった一方、実行を終えたばかりのエージェントは近い将来に再び必要とされないことがある。さらに、直前に実行されたエージェントが動的に生成した suffix はタスクの進行に伴って急速に変化することが多く、再利用されにくいにもかかわらず、一時的に cache に保持される。ワークフロー情報があれば、今後のエージェント実行系列を予測でき、より適切な退避判断を行って LRU による非効率を避けられる。

<span id="figure-03"></span>

![Agent Step Graph と KV node 退避優先度の伝播](../../papers/kvflow/figure-03.png)

**図 3。** ワークフロー対応退避方針の概要。（a）各エージェントワークフローを Agent Step Graph として抽象化し、依存 edge 上の step aggregation function を用いて steps-to-execution 値を計算する。（b）これらの値を cache tree 上で伝播させ、KV node レベルの退避優先度を割り当てる。steps-to-execution が小さい node はより長く保持され、早すぎる退避の可能性が低下する。

**Agent Step Graph と steps-to-execution。** ワークフロー構造に基づいて退避を判断するには、まずエージェント間の依存関係を捉える必要がある。しかし、実世界のワークフローにおけるエージェント間の相互作用は非常に多様である。[図 3(a)](#figure-03)に示すように、二つのワークフローは大きく異なる。上の例では Expresser エージェントが Executor1 と Executor2 の両方に依存するのに対し、下のワークフローは条件分岐を含み、いずれかの executor が完了すれば Expresser を起動できる。control-flow graph（CFG）や DAG といった従来の抽象化では、このように多様な実行 semantics を一様に捉えられない。

そこで、各 node がエージェント呼び出しに対応し、edge が依存関係を encode する *Agent Step Graph* 抽象化を導入する。従来の graph と異なり、Agent Step Graph の各 node には、predecessor から steps-to-execution を導く方法を決める *step aggregation function* が関連付けられる。prefix cache 管理では、各エージェントが実行されうる最も早い step だけに注目し、具体的な依存関係の種類を捨象する。

たとえば[図 3(a)](#figure-03)上部のワークフローでは、Expresser エージェントは上流の両 executor が完了する必要があるため、その step 値は $\max(E1, E2) + 1$ と計算される。下部のワークフローでは、いずれかの path で十分なため、step 値は $\min(E1, E2) + 1$ となる。これらの aggregation function を再帰的に適用することで、Agent Step Graph は任意の multi-agent workflow における steps-to-execution を統一的に計算できる。

**ワークフロー対応退避優先度の割り当て。** Agent Step Graph の steps-to-execution に基づき、KV cache node に優先度を割り当てる細粒度な退避方針を設計する。[図 3(b)](#figure-03)に示すように、steps-to-execution が大きいエージェントほど退避されやすい。重要なのは、エージェントが木構造の cache layout で共通 prefix segment を共有しうるため、エージェントレベルではなく cache node レベルで退避優先度を割り当てる点である。

具体的には、各エージェントの固定 prompt 部分だけに優先度を割り当て、変化するすべての suffix には常に最高の退避優先度を与えて早期の退避を促す。各エージェントについて、その steps-to-execution 値を固定 prompt の末尾 node に割り当て、木を上向きに伝播させる。一つの node が複数エージェントからの入力を集約する場合、children のうち最小の優先度（すなわち、最も退避しにくい優先度）を割り当て、近い将来いずれかのエージェントにとって有用である限り共有 node が保持されるようにする。

この伝播方式は、ワークフローに基づく再利用可能性を動的に反映した priority map を cache tree 上に生成する。GPU memory が制約されると、KVFlow はまず変化する suffix を退避し、続いて割り当てられた優先度の降順で prefix KV node を段階的に退避して、近く再利用されそうにないものを優先する。この設計は複数の並行ワークフローにも自然に対応し、共有 node での競合はワークフロー間で最小の（最も保守的な）優先度を選んで解決する。

<span id="section-03-02"></span>

### 3.2 Overlapped KV Prefetching

ワークフロー対応退避方針は、まもなく実行されるエージェントの早すぎる退避を避けるが、KV cache が退避された後にエージェントを再実行する場合、cache miss はなお発生しうる。長い prompt では KV cache を一から再計算するオーバーヘッドが大きいため、このコストは特に高い。これを緩和するため、CPU memory を二次 cache として扱い、退避されたエージェントの固定 prompt KV を保存する。

CPU cache が利用できる場合、既存システムは通常、[図 4](#figure-04)上部の timeline に示す*事後的読み込み*方針を採用する。たとえば Executor 1 の prefix cache が CPU memory へ offload されている場合、システムは Executor 1 が schedule された時点で初めて事後的に読み戻し、再計算を避ける。しかし、CPU から GPU への data transfer には依然として目立つ latency が生じ、長い prefix では特に大きい。

**先行 prefetching。** この転送オーバーヘッドを減らすため、ワークフロー情報を使って必要な KV cache を事前に非同期読み込みする*先行 prefetching* 機構を提案する。[図 4](#figure-04)の二番目の timeline に示すように、Planner の実行中、システムは次に Executor 1 が呼ばれると予測し、その KV cache を CPU から GPU へ先行して prefetch する。エージェントの実行は主に GPU 上の model forward と next token sampling（model output を GPU から CPU へ転送）からなり、KV の読み込みは CPU から GPU への転送であるため、両者は異なる hardware resource を使い、干渉せず並行して進められる。特に PCIe は full-duplex 転送に対応し、競合なしで CPU と GPU の双方向通信が可能である。ワークフローに分岐がある場合、システムは同時 prefetch 数に事前設定した上限を設けつつ、Step Graph に基づいて次に実行されうるすべてのエージェントを保守的に prefetch する。

<span id="figure-04"></span>

![事後的読み込み、先行 prefetching、状態対応 scheduling の timeline](../../papers/kvflow/figure-04.png)

**図 4。** overlap KV prefetching の概要。事後的読み込みと比べ、KVFlow は今後のエージェントを事前に読み込む先行 prefetching と状態対応 scheduling を組み合わせ、CPU-GPU 転送のオーバーヘッドを最小化する。*GPU 内にあるエージェントは、同じワークフロー内のものでも、別の並行ワークフローのものでもよい。

しかし、prefetching だけでは必ずしも十分でない。現在のエージェントの実行時間が prefetch の所要時間より短い場合、未完了の KV 読み込みによって生成がなお block されうる。これは、複数のワークフローが CPU-GPU bandwidth を競合して queueing delay を生じる高並行設定で一般的である。[図 4](#figure-04)の二番目の timeline はこの状況を示しており、prefetching を行っても Executor 1 の生成が停止する。

**状態対応 scheduling。** GPU の idle 時間をさらに減らすため、request scheduling 方針に状態認識を加える。各 scheduling step で request の prefix cache がまだ読み込み中なら、scheduler は一時的にそれを飛ばし、[図 4](#figure-04)の Executor 2 や別の並行ワークフローの request など、準備済みの request を優先する。そのため、各 cache node に *GPU memory 内*、*CPU に backup 済み*、*読み込み中*、*offload 中* のいずれかを取る状態変数を関連付ける。scheduler は request に必要なすべての node を調べ、重複する読み込みを避けるため*読み込み中*のものを飛ばし、すべての依存対象が利用可能になってから request を dispatch する。完了すると background load thread が cache node の状態を更新し、scheduler に準備完了を伝える。同様に、memory 回収時の race condition を避けるため、*offload 中*の node は退避判断から除外する。

[図 4](#figure-04)の三番目の timeline に示すように、先行 prefetching と状態対応 scheduling を組み合わせることで、KVFlow は cache miss を実質的に除去し、GPU 計算と prefetching を完全に overlap させて CPU-GPU 転送 latency を隠す。

<span id="section-03-03"></span>

### 3.3 Implementation

KVFlow の prototype は、LLM 実行 backend と application development 用 frontend interface の両方を備えた効率的な LLM サービングシステム SGLang v0.4.4 [Zhe24] を基盤として実装した。SGLang の backend は radix tree を用いて prefix KV cache を管理する。この機構を拡張し、ワークフロー対応退避方針と完全に overlap した KV prefetching に対応させる。さらに、エージェントワークフロー情報を転送できるように、SGLang の frontend と backend の両方を変更する。現在の prototype は SGLang の frontend API に統合されているが、本手法は SGLang に限定されない。frontend が server へ送る HTTP request を変更すれば、ほかのエージェントワークフローフレームワークにも適用できる。

**Step 情報の取得。** Agent Step Graph から生成される steps-to-execution 情報の取得は、実行時に本手法の最適化を導くうえで不可欠である。本実装では、各 `sgl.function` が独立したエージェントに対応すると仮定する。実行中、LLM 呼び出しを just-in-time で置換し、ワークフロー metadata を HTTP request に埋め込む。この metadata には現在のエージェントの identity と Agent Step Graph 内の全エージェントの steps-to-execution が含まれ、後続の step でどのエージェントが呼ばれうるかを示す。この情報を受信すると、backend は KV cache tree の退避優先度を適宜更新し、退避可能な GPU memory が十分に大きければ prefetching を起動できる。

step graph topology の取得に加え、[図 3](#figure-03)に示す各エージェントの固定 prompt の末尾 KV node も追跡する必要がある。一つの request 内で prompt の固定部分と動的部分を区別することには課題がある。二つの代替策を提供する。第一に、固定部分の終了位置を user が明示的に指定できる primitive interface を導入する。第二に、エージェントの cache hit 履歴を追跡し、一貫して hit する prefix を固定部分とみなす heuristic を設計する。

**Client の追跡。** サービング環境では、複数のエージェントワークフローが同一 backend instance 上で並行して実行されうる。既存のサービングシステムは request の送信元を区別しないため、名前の衝突が生じる可能性がある。たとえば、異なる二つのワークフローがともに「Planner」という名前のエージェントを定義する場合がある。これに対処するため、各 application に一意の client ID を割り当てる。client ID を backend へ送るすべての request に付加することで、システムはエージェントの identity を区別し、異なる client から来たワークフロー間の干渉を避けられる。

<span id="section-04"></span>

## 4 Evaluation

異なる caching 条件と実行条件での性能を把握するため、さまざまな microbenchmark で KVFlow を評価する。実験は次の重要な問いに答えることを目的とする。（1）大きな prompt prefix と限られた GPU memory を持つ個々のワークフローで、KVFlow は end-to-end latency を短縮できるか。（2）複数のワークフローが並行して動作する高並行環境で、KVFlow はどのような性能を示すか。これらに答えるため、まず[第 4.1 節](#section-04-01)で単一ワークフローの latency を分析し、続いて[第 4.2 節](#section-04-02)で複数ワークフローの実行を調べる。

KVFlow は model weight、prompt、decoding logic に影響を与えずシステムレベルの cache 管理だけを変更するため、出力の意味的な正しさを維持することが保証される。したがって、評価は latency などのシステム性能指標だけに焦点を当てる。

<span id="section-04-01"></span>

### 4.1 Single-Workflow Latency

まず batch size = 1 で単一のエージェントワークフローを実行したときの latency を評価する。この単一 request の latency は、notebook や development tool などで user がワークフローを個別に起動する対話的な利用状況を反映する。throughput のため batching に依存する online serving system と異なり、このような状況では個々の request に対する応答性が重視される。

10 個のエージェントからなる逐次エージェントワークフローを benchmark とする。[第 2 節](#section-02)で述べたように、各エージェントの入力 prompt は固定 prefix（呼び出し間で共有）と動的 suffix（実行ごとに変化）からなる。両方の長さを制御して token 列を無作為に sampling し、合成入力 prompt を生成する。

**Model と testbed。** 二つの構成で実験する。（1）24GB memory と 2GB/s の PCIe Gen1 bandwidth を備えた NVIDIA A10G GPU 上の Llama-3.1-8B、（2）80GB memory と 64 GB/s の PCIe Gen5 bandwidth を備えた NVIDIA H100 GPU 上の Qwen2.5-32B である。Llama model は 32 個の attention head と 8 個の KV head を使用し、Qwen は 40 個の attention head と 8 個の KV head を使用する。latency 測定の一貫性を保つため、決定的 decoding（temperature = 0、greedy sampling）を採用する。この二つの設定は、GPU memory の制約が厳しい状況を表すものとして選んだ。

**Baseline。** KVFlow と二つの SGLang 構成を比較する。第一の構成は **SGLang** と表し、CPU backup を使わず、GPU memory 内に radix 構造の KV cache を維持する。GPU memory が不足すると prefix node は退避され、再利用時には一から再計算しなければならない。第二の構成は **SGLang w/ HiCache** と表し、SGLang の既定の CPU ベース cache 拡張である階層型 radix cache を有効にする。これは、頻繁に利用される cache node を host memory へ非同期に backup して SGLang の radix tree を拡張する。退避後に node がアクセスされると、再計算する代わりに CPU から読み戻される。CPU-GPU 転送コストをさらに減らすため、HiCache を備えた SGLang は layer $l$ の GPU 計算と layer $l{+}1$ の読み込みを overlap させ、単純な二段 pipeline を形成する。

<span id="figure-05"></span>

![A10G と H100 GPU における単一ワークフローの高速化](../../papers/kvflow/figure-05.png)

**図 5。** 10 エージェントの逐次ワークフローにおける SGLang（GPU-only cache）に対する高速化。横軸は*固定部分 token / 動的部分 token / 出力 token*。

**評価方法。** まず各エージェントの固定 prompt を複数回実行して cache を warm up し、prefix cache が構築され、CPU memory へ backup されていることを確認する（HiCache の場合）。続いて、毎回異なる動的 suffix を用いて 10 エージェントワークフローを 10 回実行する。latency は全実行の平均を取る。これは、ワークフローの反復呼び出しや loop のような動作を伴う現実的な利用 pattern を模擬する [Wan24l]。

**結果。** [図 5](#figure-05)は、異なる prompt 構成*固定 / 動的 / 出力*における SGLang（GPU-only cache）に対する高速化を示す。GPU memory を超えて退避を発生させるため、8192 token など意図的に大きな固定 prefix size を試す。

すべての設定で、KVFlow が一貫して最も高い高速化を達成する。たとえば A10G 上の 8192/32/32 では、KVFlow は SGLang w/ HiCache より 1.83$\times$、GPU-only の SGLang baseline より 2.91$\times$ 高速である。これは、ワークフロー対応退避と prefetch 方針が CPU-GPU 転送オーバーヘッドを効果的に隠すことを確認する。HiCache は再計算オーバーヘッドを減らすが、計算が転送より短い場合、pipeline の cold-start と不十分な overlap の影響を受ける。

SGLang w/ HiCache は一般に GPU-only の SGLang baseline より優れることも確認できる。CPU からの読み込みは通常、再計算より高速だからである。しかし、H100 上の一部の大規模 context 設定（たとえば 8192/32/32）では、HiCache の性能向上はわずかであるか、性能が低下する。これは、memory contention や転送量が多い場合に CPU-GPU 転送を効果的に overlap できないという、SGLang の pipelining logic における最適でない scheduling が原因かもしれない。

出力 token 数が増えると、KVFlow の相対的な利得は小さくなる。これらの設定では LLM decoding latency が総実行時間を支配し、cache 読み込みの影響を受ける時間の割合が減るためである。auto-regressive LLM の時間を要する decoding step を最適化する研究は多く、speculative decoding [Che23, Lev23, Sax23]、KV cache sparsity [Xia24a]、early exit [Fu24a] などがあり、これらは本研究と直交するため KVFlow と併用できる。

一方、KVFlow の高速化は固定 prompt length とともに増加する。固定 token を 8192 に設定すると、fixed = 4096 の 1.28$\times$ に対して平均高速化は 1.48$\times$ に達する。prefix length が長くなるほど cache miss のオーバーヘッドが増し、KVFlow の先行 cache 管理がより有効になるためである。

<span id="section-04-02"></span>

### 4.2 High-Concurrency Workflow Performance

単一の H100 GPU 上で複数の独立したワークフローを同時に起動し、高並行環境でのシステム性能も評価する。これらのワークフローは相互作用せず、共有もないと仮定する。[図 6](#figure-06)に示すように、各エージェントの固定 prompt length と並行ワークフロー数で表した四つの構成を benchmark とする。動的 token と出力 token の length は 256 に固定する。各設定では、prefix caching 用 memory を枯渇させずに GPU が収容できる適切な並行度を選ぶ。並行度が高すぎると、active request が利用可能な memory をすべて消費し、システムは再利用可能な prefix cache を維持できなくなるため、本最適化の対象外となる。

<span id="figure-06"></span>

![H100 上の高並行ワークフロー性能](../../papers/kvflow/figure-06.png)

**図 6。** H100 上で固定 prompt と並行度を変えたときの高並行ワークフロー性能の比較。

<span id="figure-07"></span>

![PEER 形式ワークフローの token length 分布](../../papers/kvflow/figure-07.png)

**図 7。** PEER 形式ワークフローにおける固定部分、動的部分、出力部分の token 分布。

<span id="figure-08"></span>

![PEER 形式 multi-agent application における KVFlow の高速化](../../papers/kvflow/figure-08.png)

**図 8。** PEER 形式 multi-agent application における、SGLang および HiCache に対する KVFlow の高速化。

**結果。** [図 6](#figure-06)によれば、すべての設定で KVFlow は SGLang と HiCache の両方を一貫して上回り、最大 1.25$\times$ の高速化を達成する。固定 prompt が 1024 token の場合、512 token より cache miss のオーバーヘッドが高いため、KVFlow の性能向上も大きい。特に HiCache は高並行環境で性能が著しく低く、複数の条件で SGLang にさえ劣る。たとえば、固定 prompt 1024 token、並行ワークフロー 64 の条件では、CPU ベース cache を使わない SGLang の 0.57$\times$ の性能しか得られない。これは、頻繁な cache miss が事後的な load-back 操作を引き起こし、SGLang の schedule-compute pipeline を乱すためだと考える。さらに SGLang の KV storage layout が断片化しているため、PCIe bandwidth を十分に利用できない。KVFlow も断片化の問題自体は解決しないが、より妥当な退避と先行 prefetching により、PCIe 転送と GPU 計算をはるかに適切に overlap させる。その結果、事後的読み込みを用いる素朴な LRU ベースの HiCache に対して最大 2.19$\times$ の性能向上を得る。

**現実的なワークフローの simulation。** 実世界の deployment scenario をより適切に反映するため、PEER [Wan24l] フレームワークに基づくエージェントワークフローを模擬する。本設定では、各ワークフローは PEER が提供するワークフロー template から instance 化した四つのエージェントで構成される。各エージェントについて役割と指示を sampling し、LLM にそのエージェントの prompt を生成させる。LLM sampling に本質的な無作為性があるため、役割と指示が似ていても、生成される prompt にはばらつきが生じる。一方、すべてのエージェントが同じ application context で動作するため、prompt は部分的に重複する prefix を共有することが多い。その結果、ワークフローは多様でありながら部分的に冗長となり、現実の multi-agent application に共通する特性を反映する。

PEER の Financial QA dataset をワークフロー入力として使用する。得られるワークロードは中規模で、エージェントの prompt は通常、数十 token から数百 token までに分布する。[図 7](#figure-07)は、全エージェントにおける固定部分、動的部分、出力部分の token length 分布を示す。[図 8](#figure-08)は、KVFlow、SGLang、HiCache を備えた SGLang の性能比較を示す。KVFlow は SGLang と HiCache の両方に対して明確な改善を示し、それぞれ最大 1.12$\times$ と 1.08$\times$ の高速化を得る。この結果は、現実的な deployment setting で複数 application をサービングする際の KVFlow の高い実用可能性を示している。

<span id="section-05"></span>

## 5 Related Work

**LLM サービング最適化。** 幅広い研究が request scheduling の最適化により online LLM serving を改善しており、continuous batching（iteration-level scheduling とも呼ばれる）[Yu22a]、head-of-line blocking を軽減する multi-level feedback queue [Wu23a]、streaming scenario 向けの quality-of-experience 対応 scheduler [Liu24r] などがある。別の一群の研究は KV cache 管理に焦点を当てる。vLLM は KV tensor の paged storage により memory fragmentation を減らす PagedAttention [Kwo23] を提案し、SGLang は prefix caching の冗長性をなくす RadixAttention [Zhe24] を導入する。chatbot scenario 向けの専用 prefix caching 方針を扱う研究もある [Gao24a, Yu25b]。InferCept [Abh24a] は tool calling duration を予測し、cost model を使って intercept された request の KV cache を保持、swap、破棄のいずれにするか決める。これらの最適化は、複数エージェントからなるワークフローに焦点を当てる KVFlow と直交する。Autellix [Luo25b] と ParrotServe [Qiu24] はエージェントワークフローの request scheduling を検討するが、prefix cache 管理を考慮しないため、目的は本研究と相補的である。

**エージェントワークフローフレームワーク。** 近年、多様な multi-agent framework [Hon23, Li23s, Wu23c, Zhu24a, Inc24, Ant24c, Gao24d] が提案され、エージェントを構造化された役割として編成し、複雑なタスクを協調して解く。これらのフレームワークは、エージェント間の message passing と依存関係構築の組み込み機構、エージェント action 内に tool usage と reasoning method を容易に統合する機能、一般的なエージェントの役割と行動に対する定義済み抽象化、エージェントの並行協調を支える効率的な multi-threaded execution を提供する。一部のシステムはエージェントワークフローを computation graph として抽象化し、node が LLM を呼び出すエージェントを表し、edge が control flow または message dependency を表す。この抽象化により、edge pruning [Zha24n]、operator insertion [Zha24l]、topology optimization [Zhu24a, He25a] などの graph-level transformation を適用し、application の correctness または quality を高められる。しかし、これらのフレームワークは application layer の構築に焦点を当てたままであり、生成処理を従来の LLM serving infrastructure に依存する。対照的に、本研究はエージェントワークフローの構造を利用してサービングシステム自体を最適化し、multi-agent execution workload における backend efficiency を対象とする。

<span id="section-06"></span>

## 6 Conclusion

本稿では、エージェントワークフローにおける LLM serving を最適化する、ワークフロー対応 KV cache 管理フレームワーク KVFlow を提示する。エージェント実行を Step Graph として抽象化し、各エージェントの steps-to-execution を計算することで、KVFlow は将来の利用を予測する原理的な退避方針を実現する。さらに、cache miss による停止を先行して除去する、完全に overlap した KV prefetching 機構を導入する。評価は、長い prompt または高い並行度を持つワークフローにおいて、KVFlow が既存システムよりサービング効率を大幅に改善することを示す。

multi-agent system に関する従来研究は主に frontend application logic と interaction protocol の設計に焦点を当ててきたが、KVFlow はシステムレベルの最適化を可能にするうえでワークフロー semantics が重要であることを示す。

[+corresponding-author]: 責任著者。
