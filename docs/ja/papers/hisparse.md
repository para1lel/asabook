---
title: 'HiSparse: Hierarchical KV Cache'
createTime: 2026/08/25 11:34:32
permalink: /ja/papers/hisparse/
---

> [Zhiqiang Xie](https://zhiqiangxie.com/) [+corresponding-author]、[Zhangheng Huang](https://github.com/hzh0425)、[Tingwei Huang](https://github.com/huangtingwei9988)、[Ziyi Xu](https://orcid.org/0009-0000-4411-9773)、[Ruiyang Ma](https://orcid.org/0009-0003-9067-9538)、[Christos Kozyrakis](https://kozyraki.github.io/)。2026 年 8 月 7 日に arXiv へ初回投稿。現行版は v1。[HiSparse: Scaling Sparse-Attention Decoding with Hierarchical KV Cache Management](https://arxiv.org/abs/2608.07009v1)。[原論文 PDF](/paper/hisparse.pdf)。[DOI](https://doi.org/10.48550/arXiv.2608.07009)。[TeX ソース](https://arxiv.org/src/2608.07009v1)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

[+corresponding-author]: *責任著者：`xiezhq@cs.stanford.edu`。*

## 概要

Top-$k$ 疎アテンションは、長いコンテキストの LLM デコードを低い計算コストで実行できるようにする。各ステップが読むのは、コンテキスト全体ではなく、選択された数千個の KV エントリだけである。しかしサービングシステムは通常、すべての位置を選択可能に保つため KV cache 全体を GPU HBM に置く。そのため、リクエストのメモリ使用量は依然として完全なコンテキスト長とともに増え、デコードは計算資源を使い切るよりはるか前に容量の壁へ達する。KV cache が HBM を超えるコンテキストは、そもそも処理できない。本稿では、疎アテンションサービング向けの、厳密かつ indexer-agnostic な階層型 KV cache である HiSparse を提案する。HiSparse は各リクエストの完全な KV 履歴をホストメモリに保持し、小さな固定長 GPU cache によってデコード時の占有量を制限する。融合 CUDA kernel は、ヒット検出、LRU 置換、host-to-device fetch を含む各レイヤーの選択解決を decode CUDA graph 内で実行する。レイヤー間で選択を共有するモデルでは、厳密な layer-wise prefetch が残るミスオーバーヘッドのおよそ半分を隠蔽する。変更するのは KV の配置だけなので、モデル出力は変化しない。HiSparse は upstream SGLang にマージされており、H200、B200、GH200 上で 3 種類の疎アテンション（DSA、NSA、Quest）を用いて評価した。長いコンテキストのワークロードでピーク生成スループットを最大 $4.7\times$ 向上させ、同程度のトークン当たりレイテンシを維持しながら、高負荷時の time-to-first-token を短縮する。また no-IO oracle は、解決機構自体に測定可能なトークン当たりコストがなく、有限の residency に必要な唯一の代価が host-device IO であることを示す。

<span id="section-1"></span>

## 1 はじめに

長いコンテキストの推論は標準的な LLM ワークロードになりつつある。コーディング agent はリポジトリ全体を調べ、アシスタントは長い文書を横断して情報を統合し、近年のモデルは数万から数百万トークンのコンテキストウィンドウを対象としている [Dee26, Zen26, Qwe25a]。これらのコンテキストを処理するコストは依然として高い。デコード中の各リクエストが、完全な履歴とともに増える KV cache を持つためである。

Top-$k$ 疎アテンションは、長いコンテキストをスケールさせる有力な方法である。各デコードステップは過去の全トークンにアテンションする代わりに、クエリに依存する小さな集合から選ばれた $k$ 個の KV エントリだけにアテンションする。通常は数千トークンであり、対象モデルのコンテキストより 1～2 桁少ない。DeepSeek-V3.2 は、学習された top-$k$ 選択がモデル品質を維持しながら、長いコンテキストのアテンションコストを大幅に減らすことを示した [Dee25a, Dee25d]。同じパターンは DeepSeek Sparse Attention（DSA）や Native Sparse Attention（NSA）[Yua25e] などの学習型アーキテクチャと、Quest [Tan24] のような学習不要のセレクターにも見られる。選択集合が決まれば、アテンション kernel は完全なコンテキストではなく $k$ 個の KV エントリだけを読むため、長いコンテキストのデコードは大幅に安く処理できるはずである。

しかし、疎アテンションは KV 容量のボトルネックを小さくしない。選択集合は生成トークンやレイヤーごとに変わり、現在スキップしたエントリが後で選ばれる可能性がある。そのため、すべての論理位置をアドレス可能に保つ目的で、サービングシステムは通常 KV cache 全体を GPU HBM に常駐させる。結果としてコストが不均衡になる。各デコードステップが読むのは $k$ エントリだけだが、完全なコンテキストのすべてのエントリが、*読むかもしれない*という理由で HBM を占有する。アテンションは桁違いに安くなっても、メモリ使用量は 1 バイトも減らない。圧縮 KV レイアウトでも負担は大きい。$128$K-token の GLM-5.1 リクエストは $13.09\,\mathrm{GB}$ の BF16 KV state を持ち、単一の $1$M-token リクエストは H200 のほぼ全体を占める。重みを数える前から $141\,\mathrm{GB}$ HBM のうち $100\,\mathrm{GB}$ 超を必要とし、重みが常駐していれば受け入れることもできず、処理可能なコンテキストはモデルのウィンドウを大きく下回る。実際の壁を決めるのは、コンテキスト長と並行数の*積*である。リクエスト当たり $32$K トークンでは数十個の並行リクエストが HBM を使い切り（[図 1](#figure-01)）、$128$K では同じ HBM に収まる数が $4\times$ 少なくなる。したがって、長いコンテキストの疎デコードでは、アテンション計算よりはるか前にメモリ容量が尽きる。

<span id="figure-01"></span>

![HiSparse 階層におけるデコードスループットと time-to-first-token](../../papers/hisparse/figure-01.png)

**図 1.** HiSparse は、長いコンテキストのワークロードにおけるデコードスループットを GPU メモリ容量から切り離す。**（a）** デコードスループットと並行数：HBM が飽和すると baseline は頭打ちになるが、HiSparse はスケールを続ける。長いコンテキストほど比例して低い並行数で同じ壁に達する。**（b）** PD-colocated モードにおける平均 TTFT とスループット：HiSparse は高いスループットでも低い TTFT を維持する。GLM-5.1-FP8（DSA、$k=2048$）、$8\times$H200、$32$K 入力、$8$K 出力。

[図 1](#figure-01) はサービング上の影響を示す。HBM が埋まるとデコードスループットが頭打ちになり、colocated serving では decode KV cache が prefill の作業領域を圧迫するため time-to-first-token（TTFT）が増える。しかし full-KV serving は無料で得られる oracle を無視している。疎セレクターは、各デコードステップの各レイヤーでアテンションが読む $k$ 個の位置を正確に通知する。これは KV 配置のためのステップごとの厳密な需要信号であり、密アテンションシステムには存在しなかった。需要には構造もある。連続するデコードステップは大きく重なる位置を再選択し [Che25aa]、近接レイヤーは相関した位置を選ぶ。近年のモデルは、レイヤー間で indexer 出力を共有してこの局所性を明示している [Bai26, Glm26]。疎な選択は強い時間局所性を持つメモリアクセスのように振る舞い、小さく高速な cache と、大きく低速な階層の組み合わせが機能する領域に該当する。この観察から、top-$k$ 疎アテンションサービング向けの厳密な階層型 KV cache システム HiSparse を着想した。HBM は、アテンションが読むものに使うべきであり、読む可能性があるすべてのものに使うべきではない。

HiSparse は各リクエストの完全な KV 履歴をホストメモリに保持し、HBM 内に小さな固定長の *GPU cache* を与える。各レイヤーの top-$k$ 選択は、アテンション kernel の起動前にこの cache に対して解決される。常駐エントリはその場で使い、少数のミスは 1 回の batched transfer でホストコピーから取得する。選択位置、アテンションスコア、出力は変わらない。HiSparse が消費するのは各レイヤーの出力位置だけなので、*indexer-agnostic* であり、再学習やモデル変更なしに DSA、NSA、Quest の下へ組み込める。したがって、リクエストのデコード時 HBM 消費量はコンテキスト長ではなく GPU-cache サイズに応じて増える。

HiSparse の要点は、この階層をデコードのクリティカルパスから外すことにあり、設計上の貢献は 3 つの課題に対応する。第 1 は、*何を常駐させるか*である。現在の top-$k$ 集合だけを staging すると局所性が失われ、LongBenchV2 の選択 trace では各ステップの選択の $30\%$ がミスする。そこで HiSparse は LRU で cache を管理し、top-$k$ の 2 倍のサイズで選択局所性を $87\%$ のヒット率へ変える（[第 4.3 節](#section-4-3)）。第 2 は、*残るミスを隠すこと*である。ミスのたびに、そのレイヤーのアテンションはホストメモリからの fetch を待ち、batch size が大きいと並行リクエストのミスが同じホストリンクを奪い合う（[第 4.4 節](#section-4-4)）。HiSparse は GPU-assisted IO で fetch の帯域効率を高め、モデルがレイヤー間で選択を共有する場合は、厳密な prefetch により残る転送を中間レイヤーの計算と重ねる（[第 3.5 節](#section-3-5)）。第 3 は、*解決自体を安くすること*である。ヒット検出、victim 選択、メタデータ更新、ホスト fetch は、各デコードステップのすべての疎レイヤーで発生するため、HiSparse はそれらを decode CUDA graph 内の単一 CUDA kernel に融合する（[第 3.4 節](#section-3-4)）。

HiSparse は研究用プロトタイプではない。実装は広く使われているオープンソースサービングフレームワーク upstream SGLang [She24] にマージされ、サポート対象のサービング機能として公開されている [Xie26a, Sgl26a]（統合の詳細は [第 8 節](#section-8)）。3 種類の疎アテンション（DSA、NSA、Quest）を 3 種類のハードウェア（H200、B200、GH200）で評価した。HiSparse は長いコンテキストのピーク生成スループットを最大 $4.7\times$ 向上させ、スループットが重なる範囲で同程度の time per output token（TPOT）を維持し、高負荷時の TTFT を短縮しながら、モデル出力を変えない。すべての利得は 1 つのレバーから得られる。有限の residency により、同じ HBM で大きな decode batch を実行できる。このレバーは反対方向にも使え、固定 batch を大幅に少ない HBM で処理するか（[第 4.2 節](#section-4-2)）、HBM だけでは保持できないコンテキストを処理できる（[第 5 節](#section-5)）。

本稿の貢献は次のとおりである。

- 長いコンテキストの top-$k$ 疎アテンションサービングにおける容量の壁を明らかにした。アクティブな KV 読み出しは $k$ に比例するが、HBM 常駐 KV の占有量は依然として完全なコンテキスト長に比例する（[第 2 節](#section-2)）。
- 完全な KV state をホストメモリで利用可能に保ちながら、固定長 GPU cache によりリクエスト当たりの decode HBM を制限する、厳密かつ indexer-agnostic な階層型 KV cache、HiSparse を提案した（[第 3 節](#section-3)）。
- 局所性を維持するミス解決経路を設計した。LRU 管理は選択局所性を cache hit に変え、layer-wise prefetch は残るミスレイテンシを隠し、融合 CUDA resolve kernel はデコードのクリティカルパス上で解決を安く保つ（[第 3.4 節](#section-3-4)、[第 3.5 節](#section-3-5)）。
- H200、B200、GH200 上の DSA、NSA、Quest ワークロードで HiSparse を評価し、長いコンテキストで大きなスループット向上を示すとともに、性能を決める cache policy、kernel、host-device 帯域のトレードオフを分析した（[第 4 節](#section-4)）。

<span id="section-2"></span>

## 2 背景と動機

<span id="section-2-1"></span>

### 2.1 Top-$k$ 疎アテンション

Top-$k$ 疎アテンションは、コンテキスト全体への full attention を、クエリに依存する少数の key 集合へのアテンションで置き換える。デコードステップ $t$ では、*indexer* が選択集合 $\mathcal{S}_t \subseteq \{1,\dots,L_{\mathrm{ctx}}\}$ を生成し、$|\mathcal{S}_t| = k$ とする。アテンション kernel は対応する key と value のエントリだけを読む。得られる疎アテンションは、モデルの疎アテンション規則に対して厳密である。$\mathcal{S}_t$ が決まれば、選択されなかったエントリは現在のアテンション計算に参加しない。

近年のシステムは主として $\mathcal{S}_t$ の生成方法が異なる（[表 1](#table-01)）。DeepSeek Sparse Attention（DSA）は backbone と共同学習する「lightning indexer」を導入し、DeepSeek-V3.2 と GLM-5.1 で使用している [Dee25a, Dee25d, Zen26, Xie26a]。トークンごとに 1 つのコンパクトな indexer key を保持し、軽量 kernel で履歴全体をスコアリングしてから token-level top-$k$ を取る。Native Sparse Attention（NSA）は、圧縮、選択、sliding-window の各 branch を学習可能なアーキテクチャで組み合わせる。選択 branch はトークンブロックごとに 1 つの圧縮 key を保持し、ブロックスコアの top-$k$ を取る [Yua25e]。Quest は、事前学習済み密モデル向けの学習不要な page-granular セレクターである。各ページの key の min/max summary を維持し、query-aware upper bound が最大のページを選ぶ [Tan24]。したがって、セレクターは維持する状態、実行するスコアリング、token、block、page という選択粒度が異なる。

<span id="table-01"></span>

![クエリ依存 top-k セレクターの比較](../../papers/hisparse/table-01.png)

**表 1.** クエリ依存の top-$k$ セレクター。各方式はコンパクトで HBM に常駐する選択状態を維持し、論理トークン位置を出力する。アテンションが読む KV record がメモリを支配し、HiSparse はそれらを階層的に管理する。

違いはあるが、各セレクターは共通のシステムインターフェースを定義する 3 つの性質を持つ。第 1 に、*選択*に必要な状態はコンパクトである。indexer key、block key、page summary は、アテンションが読む KV record よりはるかに小さく、長いコンテキストでも HBM に常駐できる。第 2 に、選択はアテンション kernel が KV record に触れる*前*に完了し、選択と読み出しの間に自然な介入点ができる。第 3 に、出力形式はすべて同じである。レイヤー $\ell$、ステップ $t$ の論理トークン位置集合を $\mathcal{S}_t^{(\ell)}$ と書く。各疎レイヤーは通常それぞれの集合を選ぶが、近年のモデルは連続レイヤー群で 1 つのレイヤーの選択を*共有*し、indexer cost を償却する [Bai26, Glm26]。HiSparse は後にこの設計を prefetch に利用する（[第 3.5 節](#section-3-5)）。HiSparse はまさにこれらの性質を利用する（[第 3 節](#section-3)）。GPU 上の選択状態と計算には手を加えず、位置インターフェースで介入し、大きな KV record だけを階層的に配置するため、indexer-agnostic になる。

<span id="section-2-2"></span>

### 2.2 KV cache と容量の壁

自己回帰デコード中は、過去のすべてのトークンの key と value が KV cache として保持される。Top-$k$ 疎アテンションが狭めるのは 1 ステップが*読む*範囲であり、*利用可能*に保つべき範囲ではない。$\mathcal{S}_t$ は完全な履歴から選ばれ、ステップごとに変化するため、現在スキップしたエントリが後で選ばれる可能性がある。そのため、indexer とアテンション backend からすべての位置をアドレス可能にする目的で、通常は cache 全体を HBM に常駐させる（[第 1 節](#section-1)）。結果として生じる admission constraint が容量の壁の定量的な形である。コンテキスト長 $L_{\mathrm{ctx}}$ の $N_{\mathrm{batch}}$ リクエストからなる decode batch は、モデル重みを置いた後に残る HBM に $N_{\mathrm{batch}} \times L_{\mathrm{ctx}}$ トークン分の KV state を収める必要がある一方、各デコードステップのアテンションが読むのはそのうち $N_{\mathrm{batch}} \times k$ トークンだけである。

この壁は一般的な 2 つのデプロイ方式の両方に現れる。並行リクエストを増やすと KV storage が HBM を満たすまではスループットが向上するが、その後はアテンション kernel に計算余力があっても scheduler が追加の decode work を受け入れられない。これが [図 1(a)](#figure-01) の plateau である。prefill と decode を別々の GPU pool で実行する *PD-disaggregated* serving では、HBM 容量が decode-pool throughput を直接制限する。*PD-colocated* serving では、prefill と decode が GPU と HBM を共有する。full-context decode KV cache は prefill chunk に必要なメモリを消費するため、新しいリクエストは最初のトークンを生成する前にメモリと decode slot の両方を待つ。平均 TTFT は queueing に支配され、負荷とともに急増する（[図 1(b)](#figure-01)）。

GLM-5.1 のデプロイ（[第 4.1 節](#section-4-1)：$8\times$H200、合計 $1.1\,\mathrm{TB}$ HBM）で具体的な値を示す。$32$K 入力／$8$K 出力では、各リクエストが最大 $4\,\mathrm{GB}$ の KV state を持ち、full-KV baseline は約 $60$ 並行リクエストで飽和する。KV は約 $240\,\mathrm{GB}$ であり、重み、activation、CUDA-graph state を常駐させた後に残る HBM に相当する（[図 1(a)](#figure-01)）。colocated ではこの飽和点から TTFT が増え始める。disaggregated では、同じハードウェアからなる decode pool の上限が、前段の prefill capacity にかかわらず対応する decode-only rate（測定では $777$ tokens/s）になる。$128$K では同じ計算により約 ${\sim}15$ リクエストしか入らない。

<span id="section-2-3"></span>

### 2.3 Availability と residency の分離

容量の壁は、同一である必要のない 2 つの要件を結び付けることで生じる。将来の top-$k$ 選択が履歴中の任意の位置を参照し得るため、モデルは過去のすべての KV エントリを*論理的に利用可能*にする必要がある。しかし GPU が必要とするのは、現在のステップで実際に読むエントリだけである。Full-HBM serving は「後で選択される可能性がある」を「今 HBM に置かなければならない」と扱う。単純だが、過度に強い制約である。サービングシステムは、indexer が $\mathcal{S}_t^{(\ell)}$ を出力した後、そのレイヤーのアテンションが実行される前に選択 KV エントリをデバイスへ置けばよい。$\mathcal{S}_t^{(\ell)}$ 外のエントリは、選択位置、アテンションスコア、出力を変えずに別の場所へ置ける。

ただし、分離だけでは offloading は実用にならない。CPU-GPU interconnect がただちにボトルネックになる。各ステップで各レイヤーの $k$ 個の選択をホストメモリから取得すると、GLM-5.1 の 1 リクエストは生成トークン当たり約 $200\,\mathrm{MB}$ の KV record を移動する（$k=2048$、全レイヤー合計でトークン当たり約 ${\sim}100\,\mathrm{KB}$ の KV）。TPOT が $30\,\mathrm{ms}$ の場合、リクエスト当たり約 $7\,\mathrm{GB/s}$ の持続的な host-to-device traffic となり、GPU 当たり十数リクエストだけで PCIe Gen5 $\times16$ link（方向当たり約 ${\sim}64\,\mathrm{GB/s}$）をミス転送が飽和させる。この差を埋めるのは選択自体の局所性、すなわち [第 1 節](#section-1) で述べた構造である。連続するデコードステップは大きく重なる位置を再選択し [Che25aa]、隣接レイヤーは相関した位置を選ぶ。新しいモデルは indexer 出力をレイヤー間で共有し、これを明示する [Bai26, Glm26]。LongBenchV2 の選択 trace では、控えめな LRU cache（top-$k$ の 2 倍）が選択の $87\%$ を device hit に変える（[第 4.3 節](#section-4-3)）。この局所性が設計全体を可能にする。最近選ばれた record の多くが再び選ばれるため caching が機能し、今後の選択が予測可能であるため prefetching が機能する。shared indexer では今後の選択が既知になる。

この分離を利得を失わずに実現するには、上記の局所性を保ちながら各リクエストの cache を制限し、残るミスレイテンシを計算の背後に隠し、出力位置だけを消費するインターフェースを通じて両方を行う必要がある。これらは [第 1 節](#section-1) で述べた課題である。次節でそれを満たす設計を示し、[第 4 節](#section-4) で各機構を定量化する。

<span id="section-3"></span>

## 3 HiSparse の設計

<span id="section-3-1"></span>

### 3.1 設計目標と不変条件

HiSparse は [第 2.3 節](#section-2-3) で導入した分離を実現する。すべての KV エントリは疎アテンションアルゴリズムから論理的に利用可能なままだが、GPU メモリに常駐する working set は有限である。以下の目標と不変条件が設計を導く。[表 2](#table-02) に本稿で用いる記号を示す。

<span id="table-02"></span>

![HiSparse の設計で使用する記号](../../papers/hisparse/table-02.png)

**表 2.** 記号。

**完全な KV availability。** HiSparse は、アクティブな各リクエストについて、decode GPU の HBM 外に完全な KV cache のコピーを維持する。疎 indexer が選択した任意の論理 KV 位置を、再計算なしで復元できる。

**有限の device footprint。** HiSparse は各リクエストとレイヤーについて、$B$ 個の論理 KV-record slot からなる固定長の *GPU cache* を HBM に予約し、そのリクエストとレイヤーで最近選択された KV record を保持する。$B$ は $B \ge k$ を満たすサービング設定値であり、$L_{\mathrm{ctx}}$ に依存しない。モデルがトークン、レイヤー当たり $W_{\mathrm{KV}}$ 個の要素を格納する場合、decode-side KV footprint は、メタデータを除いて $N_{\mathrm{batch}} N_{\ell} B W_{\mathrm{KV}} s$ として増え、$N_{\mathrm{batch}} N_{\ell} L_{\mathrm{ctx}} W_{\mathrm{KV}} s$ にはならない。

**厳密な疎アテンション出力。** あるレイヤーの疎アテンション kernel を実行する前に、そのレイヤーの選択集合 $\mathcal{S}_t^{(\ell)}$ に含まれるすべての KV エントリをデバイス上へ materialize しなければならない。HiSparse は非選択 KV エントリの配置を変えることがあるが、選択位置、アテンションスコア、アテンション出力は変えない。

**Indexer-agnostic interface。** HiSparse は選択集合の生成方法を仮定しない。DSA、NSA、Quest は異なる indexer を利用でき、HiSparse は各リクエストとレイヤーが出力する選択位置だけを消費する。

**ミスレイテンシをクリティカルパスから外す。** Residency の制限によってスループットをトークン当たりレイテンシと交換してはならない。選択集合のミスは各疎アテンションレイヤーに作業を加えるため、HiSparse はそのコストを設計目標として直接扱う。cache management は [第 2.3 節](#section-2-3) の選択局所性を保ち、大半の選択をヒットさせる（[第 3.2 節](#section-3-2)）。残りは単一の融合 kernel launch で解決し（[第 3.4 節](#section-3-4)）、prefetching は host-to-device transfer を先行レイヤーの計算と重ねる（[第 3.5 節](#section-3-5)）。

<span id="section-3-2"></span>

### 3.2 KV 階層とメタデータ

HiSparse は KV state を 2 階層に構成する（[図 2](#figure-02)）。pinned host DRAM に割り当てられた **host KV pool** は、アクティブなリクエストの authoritative な完全 KV cache を保持する。colocated serving では prefill がローカル host pool へ書き込む。disaggregated serving では、prefill が prefill-decode transfer path を通じて decode instance の host pool へ KV state を送る。

Decode GPU が格納するのは **GPU cache** だけである。これは [第 1 節](#section-1) で導入し、[図 2](#figure-02) と [図 3](#figure-03) では「hot device buffer」と表示している。概念的には、各リクエストとレイヤーに $B$-slot cache があり、各 slot が 1 つの論理位置の layer-local KV record を保持する。cache は top-$k$ の staging area だけではない。現在のステップで選ばれたエントリはアテンション実行前に存在する必要があり、残りの $B-k$ slot は後の選択でヒットする可能性がある最近の record を保持する。したがって、リクエスト、レイヤー当たりの常駐 record 数は最大 $B$ であり、cache の warm-up 後は通常 $k$ より多い。現在の選択を必ず収めるには $B \ge k$ が必要である。**Page table** は layer-local な各論理位置を cache slot、または host-only を示す sentinel に対応付ける。**LRU metadata** は常駐 slot の recency を記録し、選択エントリがミスしたときの置換を直接決める。メタデータは KV tensor より小さいが、各疎レイヤーがアテンション kernel の起動前に参照、更新するため、レイテンシは各デコードステップへ直接加わる。そこで HiSparse はメタデータを GPU に常駐させ、[第 3.4 節](#section-3-4) の融合 kernel に更新を組み込む。

これに対して選択状態は移動しない。[第 2.1 節](#section-2-1) で述べたコンパクトな indexer state、すなわち DSA の per-token indexer key、NSA の compressed block key、Quest の page summary は、リクエストの存続中 GPU に常駐する。HiSparse の page table と LRU metadata も同様であり、いずれもホストメモリへ page out しない。この常駐状態はコンテキスト長とともに増えるが、トークン当たりのコストは KV record より 2～3 桁小さい。indexer key と page-table entry の合計は最大でもトークン当たり数百バイトであり、KV record は約 ${\sim}100\,\mathrm{KB}$ である（[第 2.3 節](#section-2-3)）。デプロイ全体では数百 MB であり、HBM から置き換えるリクエスト当たり数 GB の KV cache と比べて小さい。階層間を移動するのはアテンション KV record だけであり、indexer 自体の計算は変わらない。

**置換ポリシー。** 各 request-layer cache は独立に LRU で管理し、1 つ意図的な変更を加える。同一ステップ内では、*ヒット*したエントリを新たに取得したミスより recency order の上位へ昇格させる。ステップをまたいで繰り返し選択される record は初回選択より優先されるため、eviction pressure 下では一度しか選ばれないエントリが先に除かれ、cache には複数ステップでの再利用が確認された位置が蓄積する。[第 4.3 節](#section-4-3) でこの選択を検証する。recency は将来の疎な選択に対する良い online proxy であり、offline Bélády optimum の傾向に従う。cache はレイヤーごとに独立管理する。KV record は本来 layer-local であり、レイヤー間で residency を調整しても LRU が既に捉えた情報以上の利得は小さい（[第 4.6 節](#section-4-6)）。HiSparse は代わりに prefetching でレイヤー間の選択構造を利用する（[第 3.5 節](#section-3-5)）。

**Cache sizing。** $B$ は容量とレイテンシをトレードオフする。Decode KV HBM の合計は $N_{\mathrm{batch}} N_{\ell} B W_{\mathrm{KV}} s$ である。大きな $B$ はヒット率を高めるが、追加の並行リクエストに使える HBM を消費し、resolve kernel の metadata scan を長くする。実用的な範囲は $B \in [2k, 4k]$ であり（[第 4.4 節](#section-4-4)）、高速な host-device link では最適値が $B=2k$ 側へ移る（[第 4.5 節](#section-4-5)）。

<span id="figure-02"></span>

![HiSparse の階層型 KV cache の概要](../../papers/hisparse/figure-02.png)

**図 2.** HiSparse の概要。ホストメモリは各アクティブリクエストの authoritative な完全 KV cache を保持する。GPU はコンパクトな状態（indexer state、page-table と LRU metadata を持つ per-request-per-layer GPU cache）だけを保持し、すべての計算を実行する。（1）Prefill は各レイヤーの KV record を host pool へ書き込む。Decode 中は、（2）疎 indexer が選択した論理位置を出力し、融合 Resolve kernel が GPU cache を probe する。（3）LRU victim を退避しながら host pool からミス record を取得し、（4）物理 cache slot を疎アテンション backend へ渡す。（5）新たに生成されたトークンの KV record は host pool へ write-through される。

<span id="section-3-3"></span>

### 3.3 リクエストのライフサイクル

リクエストは [図 2](#figure-02) に示す 4 段階を進む。PD-colocated と PD-disaggregated のフローは同じであり、prefill KV が host pool へ届く方法だけが異なる。ローカルで書くか、prefill-decode transfer path で送るかである（[第 3.2 節](#section-3-2)）。

**（1）Prefill と staging。** Prefill engine は通常の prefill path で prompt を処理する。各レイヤーの KV state が生成されるたびに、HiSparse はそれを host KV pool へ書き込む。DSA の lightning-indexer representation など、各デコードステップで参照する compact indexer state はデバイスに残る。

**（2）Admission。** Host KV state が利用可能になり、HiSparse が per-layer GPU cache と metadata を予約すると、リクエストは decode 用に schedule できる。リクエスト当たりの予約 KV capacity は $N_{\ell} B W_{\mathrm{KV}} s$ であり、$k$ の小さな倍数である設定値 $B$ に比例する。$N_{\ell} L_{\mathrm{ctx}} W_{\mathrm{KV}} s$ ではないため、長いコンテキストのリクエストが完全な履歴長に比例して decode HBM を消費しなくなる（GLM-5.1、$B=4096$ ではリクエスト当たり約 $0.4\,\mathrm{GB}$。$128$K context の $13.09\,\mathrm{GB}$ から約 ${\sim}30\times$ 減少）。

**（3）Layer decode。** デコードステップ $t$ のレイヤー $\ell$ で indexer が選択集合を出力する。ミス解決経路は、選択した論理位置の常駐状況を調べ、足りない layer-local KV record を host pool から取得し、page table と LRU state を更新して、選択位置と整列した密な physical device slot vector を出力する。そのレイヤーの疎アテンション kernel は解決完了後に実行される。これは同期経路であり、[第 3.5 節](#section-3-5) は fetch work の一部を先行レイヤーの計算と重ねる。

**（4）生成 KV の write-through。** 新たに生成した各トークンの KV record は、リクエストの GPU cache で予約された slot へ直接生成されるため、最新位置は後の選択に備えて常駐する。専用 backup stream が record を host pool へ write-through する。この処理は次ステップの計算と重なり、後の fetch が参照する前に backing copy が完成するよう event で順序付ける。

<span id="section-3-4"></span>

### 3.4 融合ミス解決 kernel

ミス解決は各疎アテンションレイヤーのクリティカルパスにある。HiSparse はレイヤーの選択集合を受け取り、ヒットを識別し、ミスの victim を選び、欠けた KV record を取得し、metadata を更新して、physical device slot を attention backend へ返す必要がある。操作を複数の CUDA launch に分けると、レイヤーごとに一時状態を HBM へ繰り返し materialize し、launch latency も加わる。中間状態も密接に結合している。hit/miss mark、victim assignment、LRU update、出力 device-location vector は、同じ選択集合と resident-buffer metadata に依存する。そこで HiSparse は単一の融合 CUDA kernel、Resolve でミスを解決する。各疎レイヤーで 1 回起動し、CUDA block 1 個が各リクエストの work item を処理し（[図 3](#figure-03)）、統合先である SGLang の steady-state decode CUDA graph [She24] 内へ capture される。解決が利用するのは出力論理位置と HiSparse 自身の metadata だけなので、同じ kernel が DSA、NSA、Quest を処理できる。ソフトウェア管理 TLB のように、論理 index を入力し、physical slot を出力する。

<span id="figure-03"></span>

![HiSparse の融合ミス解決 kernel の 5 段階](../../papers/hisparse/figure-03.png)

**図 3.** 融合ミス解決 kernel。1 リクエスト、1 レイヤーについて、Resolve はまず選択論理位置の shared-memory hash table を構築する。次に、すべての GPU-cache slot を hash table に対して並列に probe し、常駐エントリを hit または eviction candidate として mark する。その mark を parallel scan して victim を選び、LRU metadata を更新し、pinned host memory から欠けた KV record を取得して、疎アテンション backend 用の physical device slot を top-$k$ 順に出力する。

**Phase 1：選択位置の staging。** スレッドが協調して選択位置を shared-memory hash table へロードする。以後の kernel は、top-$k$ vector を HBM から繰り返し読み直さず、選択集合の membership test を高速に実行できる。

**Phase 2：GPU-cache slot の mark。** [図 3](#figure-03) のように、各 GPU-cache slot に現在格納されている論理位置をスレッドが hash table で probe する。kernel は slot ごとに compact mark を書く。常駐エントリが現在の選択集合に含まれれば hit、それ以外は evictable である。

**Phase 3：mark の scan と LRU update。** kernel は slot ごとの mark を parallel scan する。scan は evictable slot を compact にし、欠けた選択位置に必要な数の victim を選び、ステップ後の buffer state に対応する更新済み LRU metadata を生成する。hit slot は保持して most-recently-used 側へ昇格し、victim slot をミスへ割り当て、取得したミスを hit の直後に置く。[第 3.2 節](#section-3-2) の置換ポリシーに従う。

**Phase 4：欠けた KV record の fetch。** ミスを担当するスレッドは、対応する layer-local KV record を pinned host pool から取得して確保済み device slot へコピーする。HiSparse は Strata [Xie25a] の GPU-assisted IO を採用する。DMA copy を staging する代わりに、GPU thread が pinned host memory に対して vectorized non-coherent load（`ld.global.nc.v2.b64`）を直接発行する。cache miss の分散したアドレスを処理でき、PCIe と NVLink-C2C で transaction overhead を減らす [Nvi26a]。per-thread transfer block size を調整し、断片化した miss read でも link bandwidth に近づける。

**Phase 5：アテンション入力の publish。** 最後に kernel は page table を更新し、選択論理位置と整列した physical device offset の密 vector `top_k_device_locs` を出力する。下流の sparse-attention gather は選択 KV record を GPU cache から直接読める。

<span id="section-3-5"></span>

### 3.5 Layer-wise prefetch

融合 resolve kernel を用いても、GPU cache でミスした選択エントリはホストメモリレイテンシを露出させることがある。HiSparse はレイヤー間 prefetch によりコストを隠し、indexer output を再利用して cross-layer selection locality を明示するモデルを対象とする。IndexCache [Bai26] はレイヤーを、top-$k$ indexer を実行する *anchor* layer と、直前の anchor の選択を再利用する *shared* layer に分ける。GLM-5.2 は IndexShare としてこの設計を実装し、4 レイヤーの各 group が 1 つの indexer を共有する [Glm26]。この種のモデルで HiSparse に推測は不要である。anchor layer が選択集合を出力した時点で、group 内のすべての shared layer の選択位置が、アテンション実行より数レイヤー早く既知になる。

**共有選択による厳密な prefetch。** HiSparse は *plan-then-IO* scheme を用いる。anchor の Resolve は、どの host record をどの cache slot へ移すかという miss plan も記録する。side stream 上の copy-only kernel がその plan を各 shared layer の cache へ replay し、transfer を中間レイヤーの計算と重ねる。各 shared layer の cache は anchor の slot layout と lockstep で動くため、shared layer は anchor の slot table をそのまま再利用する。prefetch-completion event を待つだけで解決を完全に省略し、probe、LRU update、同期 host-memory load、無駄な speculative traffic はない。copy は demand path の GPU-assisted IO を再利用する（[第 3.4 節](#section-3-4)）。分散した prefetch read が link bandwidth に近づくことは重要である。prefetch は IO を消さずに早めるだけで、demand miss と同じ host link を使うからである（[第 4.5 節](#section-4-5)）。

**Speculative alternative。** Shared index を持たないモデルについて、レイヤー $\ell$ の選択位置を $\ell+1$ の hint に使う speculative variant も調べた。隣接レイヤーは重なる位置を選ぶことが多く、誤った hint は転送を無駄にするだけで correctness には影響しない。しかし評価では end-to-end gain がほとんどなかった（[第 4.6 節](#section-4-6)）。LRU-managed GPU cache が暗黙の cross-layer reuse を既に大半捉えるため、hint された record は通常常駐している。残るミスは、まさに hint が予測できない位置である。この否定的な結果から、ミスレイテンシを隠す正しい方向は、さらに深い speculation ではなく shared-index model co-design だと考える。

<span id="section-4"></span>

## 4 評価

評価では 4 つの問いに答える。（1）HiSparse はコンテキスト長、モデル、疎セレクター、ハードウェアプラットフォームをまたいで end-to-end serving を改善するか。（2）GPU cache はミス数を低く保つのに十分な局所性を利用できるか。（3）ミス解決オーバーヘッドと GPU-cache size の選択を決めるものは何か。（4）高速な host-device link と layer-wise prefetching は、新しいハードウェアとモデル設計の傾向にどう作用するか。

<span id="section-4-1"></span>

### 4.1 セットアップ

**モデル。** 3 種類の疎アテンションを評価する。DeepSeek-V4-Flash は圧縮 KV エントリに NSA-style top-$k$ 選択を適用する hybrid attention（DeepSeek は Compressed Sparse Attention と呼ぶ）を持つ [Dee26, Yua25e]。GLM-5.1-FP8 は DeepSeek Sparse Attention（DSA）を使用する [Zen26, Dee25a, Xie26a]。Qwen3-30B-A3B-Thinking-2507 [Yan25g, Qwe25a] には学習不要な疎セレクターとして Quest を適用する [Tan24]。layer-wise prefetch study（[第 4.6 節](#section-4-6)）では、IndexShare によりレイヤー群で DSA indexer の選択を共有する GLM-5.2-FP8 も使用する [Glm26, Bai26]。全実験でクエリ当たり $k=2048$ トークンを選択する。DeepSeek-V4-Flash は $4$-token compressed KV entry 単位で選択するため、top-$512$ が $2048$ トークンを覆う。HiSparse はモデルの KV footprint を支配する compressed KV entry を管理し、残る branch state は GPU に常駐する。他のモデルでは token-level、$k=2048$ である。Quest はアーキテクチャ上の sparsity を必要とせず、小さい Qwen3 モデルの通常の dense-attention KV cache に全レイヤーで適用する学習不要なセレクターである。全モデルの KV cache は BF16 であり、モデル名の FP8 は重み精度を表す。

**プラットフォーム。** End-to-end serving 実験は各図に示すプラットフォームを用いる。DeepSeek-V4-Flash は $2\times$B200、GLM-5.1-FP8 と GLM-5.2-FP8 は $8\times$H200、Qwen3+Quest は GH200 node である。H200 node は 8 GPU と $2\,\mathrm{TB}$ の host DRAM を組み合わせる。最大の動作点（[第 4.6 節](#section-4-6)、$32$K 入力／$8$K 出力、$256$ 並行リクエスト）では、host KV pool が約 $1\,\mathrm{TB}$ の pinned host memory になる。

**Baseline。** 全実験で、KV cache 全体を HBM に常駐させる未変更の SGLang v0.5.11 と比較する。モデル、parallelism、precision、scheduler setting は同一である。他の offloading system とは比較しない。最も近い ESS [Che25aa] と ECHO [Liu26] は同時期の研究であり、前者は simulation で評価した prototype、後者は NSA 専用である（[第 6 節](#section-6)）。

**ワークロードと指標。** まず SGLang の標準 `bench_serving` harness [She24, Sgl26] で end-to-end serving を評価する。各モデルについて、出力長と closed-loop concurrency を固定し、対応する図の入力長を sweep する。Concurrency sweep（[図 1](#figure-01)、[図 4](#figure-04)、[図 8](#figure-08)）は $32$K 入力／$8$K 出力に固定する。この長さは長いコンテキストのサービングを代表し、比較に最も情報量がある。full-KV baseline は $32$K なら複数の並行数を受け入れられ、両システムの完全な曲線を得られるが、はるかに長いコンテキストでは baseline が少数の低並行点に崩れ、傾向を比較しにくい。そこで最大 $200$K の長いコンテキストは peak-throughput comparison として報告する（[図 5](#figure-05)）。generation throughput（generated tokens/s）、TTFT、TPOT を測定する。Generation throughput はシステム出力レート、TPOT は最初のトークン以降の output-token latency を測る。次に ablation で end-to-end gain の由来を説明する。cache-policy 実験は LongBenchV2 [Bai25] の sparse-selection trace を replay し、kernel 実験は miss-resolution time を報告する。

<span id="section-4-2"></span>

### 4.2 End-to-end benchmark

<span id="figure-04"></span>

![2 基の B200 における DeepSeek-V4-Flash の end-to-end serving](../../papers/hisparse/figure-04.png)

**図 4.** $2\times$B200 の PD-colocated モードで DeepSeek-V4-Flash（NSA）を end-to-end serving。$32$K 入力／$8$K 出力。各ステップの選択は $2048$ トークンを覆う（$4$-token compressed KV entry の top-$512$）。**（a）** generation throughput と closed-loop concurrency（実線：prefill+decode、破線：decode-only、PD-disaggregated decode-pool throughput の目安）。**（b）** 平均 TTFT、**（c）** 平均 TPOT と達成した generation throughput。

[図 4](#figure-04) は DeepSeek-V4-Flash の $32$K 入力／$8$K 出力における end-to-end benchmark を示す。低い並行数では baseline が全 active KV cache を HBM に保持できるため、baseline と HiSparse のスループットは近い。並行数が増えると baseline は飽和する。KV capacity がなければリクエストを追加できず、スループットはほぼ一定になる。[図 1(a)](#figure-01) の GLM-5.1 と同じ壁である。HiSparse はリクエスト当たりの decode HBM footprint を減らしてスケールを続け、並行数 $64$ で generation throughput を $600$ から $1257$ tokens/s（$2.1\times$）へ、decode-only では $1511$ から $4308$ tokens/s（$2.9\times$）へ高める。利得はすべて batch-size effect である。HiSparse が個々の decode step を高速化するのではなく、同じ HBM に大きな decode batch を入れる。$32$K 入力では baseline も数十リクエストを保持できるため余地は中程度だが、入出力が長くなると実行可能 batch はさらに小さくなり、効果が大きくなる（[図 5](#figure-05)）。

レイテンシの panel は colocated serving における意味を示す。baseline では decode saturation が prefill chunk と新規リクエストを queue に入れるため、per-token decode latency が崩れていなくても TTFT が急増する。平均 TTFT は並行数 $8$ の $26\,\mathrm{s}$ から $64$ の $829\,\mathrm{s}$ へ増えるが、HiSparse は $171\,\mathrm{s}$ である。HiSparse は HBM の余裕を大きくして decode work を速く排出し、高い達成スループットでも TTFT を低く保つ。高スループット点では TPOT が増えるが、重なる範囲では baseline と同程度である（並行数 $16$ で $15.9$ と $16.0\,\mathrm{ms}$）。長いコンテキスト領域では、ミス解決オーバーヘッドより容量の利得が大きい。

<span id="figure-05"></span>

![Quest と DSA の入力長別ピーク生成スループット](../../papers/hisparse/figure-05.png)

**図 5.** 追加の 2 種類の疎アテンションについて、入力長別のピーク生成スループット。どちらも $k=2048$ トークンを選択する。*左：* GH200 上の Qwen3-30B-A3B と Quest。*右：* $8\times$H200 上の GLM-5.1-FP8 と DSA。

[図 5](#figure-05) は結果を 2 つの軸で拡張する。他のモデル、セレクター、プラットフォームと、さらに重要な、はるかに長いコンテキストである。入力長を GH200 上の Qwen3+Quest で $200$K、H200 上の GLM-5.1+DSA で $160$K まで sweep する。$4$K では full-KV baseline が有用な batch を HBM に収められるため、HiSparse の余地は小さい。Qwen は $2430$ から $2668$ tokens/s、GLM はほぼ不変（$2288$ と $2280$ tokens/s）である。長いコンテキストでは baseline が capacity-bound になる一方、HiSparse の decode memory は設定した GPU-cache size に比例する。利得は大きく、コンテキスト長とともに増える。Qwen は $32$K で $3.6\times$、$200$K で $4.7\times$（$511$ から $1824$、$111$ から $520$ tokens/s）、GLM は $32$K で $3.1\times$、$160$K で $2.9\times$（$624$ から $1919$、$232$ から $680$ tokens/s）向上する。GLM の $32$K 点は [図 1](#figure-01) の sweep を要約し、同図はその動作点での完全な concurrency と TTFT を示す。この領域では、論理 KV availability と GPU residency の分離が高い serving throughput に変わる。decode-only curve は PD-disaggregated serving でも同じことを示す。prefill time を除く [図 4(a)](#figure-04) の破線は dedicated decode pool の能力を見積もる。full-KV decode pool は HBM に入る batch で制限され、HiSparse はその上限を $2.9\times$ 引き上げる。testbed capacity のため物理的な disaggregated deployment は実行せず、decode-only rate を proxy とする。

容量の利得は別の用途にも変換できる。スループットを増やす必要がない operator は hardware saving として使える。batch size をそろえると、HiSparse は並行数を増やすのと同じ倍率で decode KV budget を減らす。GLM-5.1 の $32$K、約 ${\sim}60$-request batch では、$B=4096$ により HBM を約 ${\sim}240\,\mathrm{GB}$ から ${\sim}25\,\mathrm{GB}$ へ減らす。差はコンテキストとともに広がり、$128$K ではリクエスト当たり $13.09$ と $0.4\,\mathrm{GB}$ である（[第 3.3 節](#section-3-3)）。KV が支配する長いコンテキストでは、同じワークロードを少ない GPU または安価で HBM の少ない部品に収められる。代価は [第 4.6 節](#section-4-6) のトークン当たり IO overhead である。

<span id="section-4-3"></span>

### 4.3 GPU-cache locality と LRU

<span id="figure-06"></span>

![7 種類の GPU-cache 設定におけるステップ別 top-k ミス率](../../papers/hisparse/figure-06.png)

**図 6.** 同じ GLM-5.1 LongBenchV2 sparse-selection trace（$k=2048$）を 7 種類の cache setting で replay したときの per-step top-$k$ miss rate（レイヤー平均、平滑化）。$B$ はリクエスト、レイヤー当たりの KV-record slot 数。Top-$k$ だけを staging する *Swap-vanilla*（$B=2048$）は hot entry を余分に保持せず、各ステップの選択の平均 $30\%$ がミスする。同じ $B=4096$ では、LRU（平均 $13.4\%$）が FIFO（$17.2\%$）と random replacement（$16.1\%$）を常に上回り、offline Bélády optimum（$8.2\%$）の傾向に従う。LRU cache を $B=8192$ へ倍増するとミス率は $6.7\%$ へ再び半減し、保持した局所性が host-memory load を直接減らすことを示す。

HiSparse は GPU cache を単なる一時 top-$k$ staging area ではなく、hot KV cache として用いる。[図 6](#figure-06) は、その理由と LRU を使う理由を示す。trace は $100{,}384$-token LongBenchV2 prompt を処理し、全 $78$ sparse layer で $1{,}799$ step を decode する GLM-5.1 request から得た。すべての policy が同じ per-layer top-$k$ selection stream を replay し、図と平均値は最初の $1{,}000$ decode step を扱うため、差は置換判断だけによる。$B$ は*リクエスト、レイヤー当たり*の KV-record slot 数である（[表 2](#table-02)）。$B=4096$ は $k=2048$ の 2 倍である。現在の top-$k$ だけを常駐させる（$B=k$）と、選択集合がステップ間で変わるため平均 $30\%$ がミスする。cache を倍増するとミスは減るが、保持する局所性は置換 policy で決まる。$B=4096$ の LRU は平均 $13.4\%$ で、FIFO（$17.2\%$）と random（$16.1\%$）より常に低く、offline-optimal Bélády policy [Bel66] の形状に従う。recency が将来の疎な選択に対する良い online proxy である。このため [第 3.4 節](#section-3-4) の miss-resolution kernel は、現在の top-$k$ だけで cache を再構築せず、resident hit を保持して LRU を in-place update する。同図は残る余地も示す。LRU cache を $B=8192$ へ倍増すると $6.7\%$ へ半減する一方、Bélády は $B=4096$ で既に $8.2\%$ に達し、$B=8192$ ではさらに低い。同じ容量で predictive replacement policy が得られる上限を示す。

<span id="section-4-4"></span>

### 4.4 ミス解決コストと cache-size tradeoff

低いミス率だけでは不十分である。残るミスの解決には時間がかかり、そのコストは cache size と batch size で変わる。大きな GPU cache は局所性を多く保持するが、metadata scan を長くし、リクエスト当たりの HBM を増やす。batch size が大きいと並行リクエストのミスが host-memory bandwidth を競合する。そこで融合 resolve kernel の時間を phase 別に分解し、実用的な cache-size range と、ミスが少なくなった後の bottleneck を調べる。

<span id="figure-07"></span>

![モデル、cache size、platform 別のミス解決時間の内訳](../../papers/hisparse/figure-07.png)

**図 7.** モデル、GPU-cache size、platform 別のミス解決時間。H200 は PCIe Gen5 host-device link（太い明色線、塗りつぶし marker）、GH200 は NVLink-C2C（濃色破線、白抜き marker）。panel title は各モデルの KV-record granularity での top-$k$ を示す（[第 4.1 節](#section-4-1)）。*IO* は host-memory fetch phase、*probe & scan* は metadata phase であり、platform への依存は小さい。GH200 破線は H200 band の ${\sim}5\%$ 以内にある。残りの phase（hash-table build と output publication）は全条件で $1$-$4\,\mu\mathrm{s}$ なので省略する。上段：batch size $16$ で cache ratio $B/k$ を変えた per-kernel-call time。大きい cache はミスを減らして IO を短縮するが、resident slot の probe-and-scan work を増やす。下段：$B=2k$ で batch size を変えた per-kernel-call time。高 batch では PCIe link の IO が支配し、高速な NVLink-C2C path がそれを圧縮する。

[図 7](#figure-07) は 3 つの end-to-end model と DeepSeek-V4-Pro（token-level selection、$k=1024$）について、各 native selection granularity で tuning tradeoff を示す。大きな GPU cache はミス数を減らし、resolve kernel の IO 部分を短縮する。しかし無料ではない。kernel は多くの resident slot を probe、scan し、大きな cache はリクエスト当たりの HBM も増やす。その HBM は、並行リクエスト、モデル重み、常駐 MoE expert に使える。図のモデルでは有用な領域は選択集合の小さな倍数、通常 $2k$-$4k$ である。hot entry を局所性の利用に十分保持しつつ、metadata scan や HBM capacity を新たな bottleneck にしない。比較のため、H200 上の GLM-5.2 decode profile では sparse-attention kernel 自体がレイヤー当たり約 ${\sim}60\,\mu\mathrm{s}$（per-GPU batch $8$）である。隠蔽されない $100$-$200\,\mu\mathrm{s}$ resolve はレイヤーの attention critical path を倍以上にするため、ミスを少なくすること（[第 4.3 節](#section-4-3)）と、残りを隠すこと（[第 4.6 節](#section-4-6)）の両方が必要である。

下段は $B=2k$ で batch-size effect を分離する。probe-and-scan は比較的安定するが、多くのリクエストが miss load を同時発行するため、IO は batch size とともに急増する。この結果が IO-side design の動機になる。GPU-assisted IO と調整した block size で各 fetch を効率化し（[第 3.4 節](#section-3-4)、[第 3.5 節](#section-3-5)）、platform が備える高速な host-device link を利用し（[第 4.5 節](#section-4-5)）、layer-wise prefetch で残るレイテンシを計算の背後に隠す（[第 4.6 節](#section-4-6)）。

<span id="section-4-5"></span>

### 4.5 帯域感度

前の実験は、高 batch size で host-memory IO が miss-resolution cost を支配し得ることを示す。CPU-GPU path が大幅に高速化した場合、HiSparse は大きな GPU cache に HBM を使ってミスを避けるべきか、ミスを増やして並行数にメモリを残すべきか。この問いを GH200-class high-bandwidth host-device path [Nvi26a] で調べる。

[図 7](#figure-07) の GH200 破線は platform だけを変えて microbenchmark を繰り返す。IO は大きく短縮され、GLM-5.1、$B=2k$、batch size $16$ では call 当たり $112$ から $29\,\mu\mathrm{s}$ になる。大きな GPU cache の利得は小さくなる。ミスの fetch は安くなるが、大きな cache の scan は依然として時間と HBM を使う。実用的な動作点は $B=2k$ のような小さな cache へ移り、大きな decode batch に device memory を残せる。HiSparse は容量 bottleneck を調整可能な latency/bandwidth problem に変換し、高速な host-device link はこの tradeoff を有利にする。

HiSparse の tuning は意図的に単純かつ静的である。$B$ は deployment 時に固定する serving-configuration parameter であり、対象 platform で [図 7](#figure-07) のような profiling sweep を行って選ぶ。$B=2k$ は堅牢な default である。実験では、好ましい setting は workload ではなく host-link bandwidth に主として依存した（[図 7](#figure-07) の実線と破線の IO curve）。cache はリクエスト単位で割り当てるため、admission-time または dynamic resizing は機構上容易だが、dynamic policy は今後の課題とする。次に、cross-layer selection locality で残るミスレイテンシを隠す model-side opportunity を調べる。

<span id="section-4-6"></span>

### 4.6 Layer-wise prefetching

高速な host-device link でも、KV transfer を有用な計算と重ねない限りミスはクリティカルパスにある。HiSparse の厳密な prefetch path（[第 3.5 節](#section-3-5)）を、IndexShare で DSA indexer selection をレイヤー間共有する GLM-5.2-FP8 [Glm26, Bai26] で評価する。$78$ レイヤー中 $21$ anchor layer が indexer を実行し、残る $57$ layer は直前の anchor の選択を再利用する。anchor layer が選択集合を出力すると、HiSparse は後続 shared layer の KV record に host-to-device load を発行し、中間レイヤーの計算と重ねる。$8\times$H200、PD-colocated、$32$K 入力／$8$K 出力、closed-loop concurrency $8$～$256$ で 4 設定を比較する。full-KV baseline、同期 miss resolution の HiSparse（prefetch 無効）、厳密 prefetch の HiSparse、resolve kernel の host-memory IO を完全に省く *no-IO oracle* である。oracle はミスに古い KV record を使うため出力は無効だが、benchmark は出力長を固定するので、timing は IO-hiding scheme の有効な上界である。同じ host link traffic を残す完全重複 prefetch より強い。全 HiSparse setting は $k=2048$、$B=4096$（$B=2k$）であり、[第 4.4 節](#section-4-4) の $2k$-$4k$ 内にある。

<span id="figure-08"></span>

![IndexCache 共有選択を用いた厳密な layer-wise prefetch](../../papers/hisparse/figure-08.png)

**図 8.** IndexCache 共有選択による layer-wise exact prefetching。$8\times$H200 の PD-colocated で GLM-5.2-FP8（DSA）、$32$K 入力、$8$K 出力、$k=2048$、$B=4096$。灰色点線は host-memory IO を完全に省く *no-IO oracle*（出力は無効、任意の IO-hiding scheme の性能上界）。**（a）** generation throughput と closed-loop concurrency（濃色：prefill+decode、明色：decode-only、PD-disaggregated decode-pool throughput の目安）。full-KV baseline は KV cache が HBM を満たすと飽和し、HiSparse variant はすべて $256$ request までスケールする。**（b）** 平均 TTFT と達成した generation throughput。点の label は並行数。**（c）** 平均 TPOT と達成した generation throughput。低並行では oracle が baseline の TPOT と一致し、HiSparse の全 overhead が IO であることを示す。厳密 prefetch は同じ並行数で TPOT を $13$-$15\%$ 減らし、同期解決と oracle の差を約半分埋める。

[図 8](#figure-08) は [第 4.2 節](#section-4-2) と同じ容量傾向を示す。full-context KV cache が HBM を埋めると baseline は飽和し、平均 TTFT は並行数 $16$ の $16\,\mathrm{s}$ から $32$ の $91\,\mathrm{s}$、$64$ の $275\,\mathrm{s}$ へ跳ねる。一方 HiSparse variant は $256$ 並行リクエストまでスケールする。panel（b）は prefill-side staging が事実上無料であることも示す。queueing のない低並行では、prefill KV を host pool へ write-through しても平均 TTFT は変わらない（並行数 $8$ で baseline $10.7\,\mathrm{s}$、全 HiSparse variant $10.7$-$10.8\,\mathrm{s}$）。prefetch は残る miss-resolution cost の多くを回復する。同じ並行数で平均 TPOT を全 sweep にわたり $13$-$15\%$ 下げ、generation throughput を $14$-$17\%$ 上げる。ピークは baseline の $618$、prefetch 無効の $1515$ から $1727$ tokens/s になり、full-KV baseline 比 $2.8\times$ である。no-IO oracle は $2034$ tokens/s の上界を与え、厳密 prefetch は $85\%$、無効時は $74\%$ に達する。decode-only curve では oracle $4671$ tokens/s、厳密 prefetch $3410$（$73\%$）となり、PD-disaggregated の同じ上界を示す。colocated では prefill time が decode-side IO の相対コストを薄めるため、差が小さい。

Panel（c）は明示的な上界に対する overhead を定量化する。低並行で oracle は full-KV baseline の TPOT と一致する（並行数 $8$ で $24.1$ と $24.8\,\mathrm{ms}$）。hash probe、victim scan、LRU update、KV gather を含む resolve mechanism 自体には測定可能な per-token cost がなく、HiSparse の TPOT overhead 全体が host-memory IO である。同期解決は並行数 $8$ で $7.7\,\mathrm{ms}$、$256$ で $22.0\,\mathrm{ms}$ の IO をトークン当たり露出する。厳密 prefetch は $3.0\,\mathrm{ms}$ と $11.2\,\mathrm{ms}$ へ減らし、全 sweep の同じ並行数で約半分を隠す。残りの多くは構造的である。anchor layer 自身の選択は事前に未知なので、$78$ レイヤー中 $21$、均等な per-layer split なら IO の約 $27\%$ が同期のまま残る。この floor を除くと、prefetch は作用可能な IO の 3 分の 2～5 分の 4 を隠す。残りは overlap shortfall であり、prefetch は transfer を消さず前倒しするだけなので、高 batch では demand miss と同じ host link を競合する。

[第 3.5 節](#section-3-5) の speculative variant も評価した。shared index を持たないモデルでレイヤー $\ell$ の選択を $\ell+1$ の hint にする。GPU-cache hit rate はわずかに上がったが、測定可能な end-to-end gain はなかった。hint 位置は通常既に常駐し、残るミス、すなわち新たに top-$k$ へ入る位置は hint が予測できないため、host-link traffic だけを増やして critical-path load を除けない。依存関係が結論を決める。ミス IO を計算と重ねるには、レイヤー実行よりかなり前にその選択を知る必要があり、speculation は確実に供給できない。model co-design なら、レイヤー間で選択を共有して依存関係を直接なくし、prefetch を厳密にできる。測定可能な利得がない場合と、上記の $13$-$15\%$ TPOT reduction の差である。

<span id="section-5"></span>

## 5 議論と制約

**露出した IO による TPOT overhead。** HiSparse は HBM capacity を host-memory traffic と交換し、同期ミス解決では同じ低並行時にトークン当たり $7$-$8\,\mathrm{ms}$ の TPOT overhead が生じる（[第 4.6 節](#section-4-6)）。設計は LRU caching（[第 4.3 節](#section-4-3)）、GPU-assisted IO（[第 3.4 節](#section-3-4)）、高速 link（[第 4.5 節](#section-4-5)）を重ね、model co-design が残りの約半分を隠す。shared selection では厳密 prefetch により低並行時の露出 IO をトークン当たり約 $3\,\mathrm{ms}$ へ減らし、no-IO oracle は resolve mechanism 自体のコストがないことを示す（[第 4.6 節](#section-4-6)）。短いコンテキストや低並行など capacity-bound でない場合、HiSparse には overhead を相殺する利点がなく、無効化できる。

**Throughput 以外：HBM に収まらないコンテキスト。** 容量の利点は batch-size multiplier だけではない。full-KV serving では、KV cache が free HBM を超えるリクエストは*いかなる*並行数でも受け入れられず、モデルの context window ではなく device memory が処理可能なコンテキストを制限する（$1$M-token GLM-5.1 は $100\,\mathrm{GB}$ 超の KV を必要とする。[第 1 節](#section-1)）。HiSparse はコンテキスト長にかかわらず GPU-cache size で decode-side HBM を制限するため、中程度のコンテキストで容量を batch size に変える機構が、極端なコンテキストでは feasibility に変える。最大コンテキストは host-tier capacity で決まる。これは footprint bound（[第 3.3 節](#section-3-3)）から直接従うため、個別には評価しない。

**Host-memory capacity。** 現在のより根本的な制約は overhead ではなく second tier の大きさである。HiSparse は host DRAM が HBM より大幅に大きいことを仮定する。terabyte-class PCIe H200 server では成立するが、Grace-based GB200/GB300 では成立しない。各 Grace CPU の約 ${\sim}480\,\mathrm{GB}$ LPDDR は paired GPU の aggregate HBM と同程度で、GB300 では小さい。second tier が first tier より大きくなければ、HiSparse が batch size へ変える capacity multiplier は小さくなる。NVMe または network-attached tier は高いレイテンシと引き換えに容量を回復し、局所性と overlap の重要性を増す。

**Co-design implications。** この調査が将来の co-design に役立つことを期待する。レイヤー間で選択を共有する、または早期に出力するモデル [Bai26, Glm26] は overlap の余地を用意し、KV placement を latency problem から scheduling problem へ変える。host memory が大きく CPU-GPU interconnect が高速な platform は、直接小さい GPU cache と大きな batch へ変換できる（[第 4.5 節](#section-4-5)）。疎アテンションは per-step KV demand を小さく予測可能にする。モデルとハードウェアの小さな変更が、HiSparse が現在利用するものに利得を重ねる。

<span id="section-6"></span>

## 6 関連研究

**LLM serving と KV memory management。** PagedAttention と vLLM は固定長 page で KV memory を管理し、fragmentation を減らして serving throughput を高める [Kwo23]。SGLang は structured LLM program 向けに KV-reuse optimization を持つ high-throughput serving runtime を提供する [She24]。prefill と decode は異なる resource に負荷をかけるため、serving system は両者を分離または調整する [Pat23, Zho24]。IO-aware attention kernel は resident KV footprint を減らさずに compute-side memory traffic を減らす [Dao22, Dao24a]。HiSparse はこれらと相補的であり、長いコンテキストの疎アテンションリクエストごとに decode-side HBM footprint を減らし、disaggregated decode pool と colocated prefill/decode deployment の両方を助ける。

**KV compression と eviction。** 多くの研究が KV cache のエントリを廃棄または近似して小さくする。H2O、StreamingLLM、SnapKV などの eviction policy は重要でないと判断したトークンを恒久的に落とし [Zha23g, Xia24a, Li24c]、quantization はエントリ当たりの byte を減らす [Liu24c]。これらは output fidelity と capacity を交換する。HiSparse は直交し、KV entry を廃棄、近似せず移動するため、疎アテンション出力は変わらない。

**階層型および offloaded KV cache。** FlexGen、HiCache、CachedAttention、Mooncake は model または KV state を GPU、host、storage tier に配置し、throughput-oriented または cross-turn serving を行う [Sto24, Xie25d, Gao24a, Qin25c]。HiSparse の host-memory fetch path は Strata の GPU-assisted IO を借用する [Xie25a]。HiSparse に近い複数の system は KV state を host memory へ offload し、decode 中に選択的に取得または使用する。InfiniGen は重要と予測した entry を speculative prefetch し [Lee24c]、ShadowKV は GPU 上の low-rank key で選択して value を offload し [Sun25f]、MagicPIG は CPU-resident LSH で KV entry を sample する [Che25ab]。ArkVale は cold page を evict し、page summary で recall する [Che24t]。PQCache は product quantization で entry を retrieve する [Zha25i]。NEO と FastDecode は attention computation 自体を CPU へ offload する [Jia25e, He24g]。retrieval-based system は独自の approximate selector を dense-attention model に加えるため出力が元モデルからずれる。CPU-compute system は厳密だがアテンションを CPU に置く。HiSparse はモデル*自身*の top-$k$ selection を処理し、構成上厳密で、全計算を GPU に残す。最も近いものは 2 つの同時期の system である。ESS prototype は DeepSeek-V3.2 latent cache を GPU hot cache の後ろの host memory へ offload し、そのモデル専用の architecture を simulation で評価する [Che25aa]。ECHO は NSA model の KV state を offload し、今後の選択を正確に予測できるかに依存する prefetch を中心とする [Liu26]。HiSparse は indexer-agnostic interface により学習型と学習不要な selector（DSA、NSA、Quest）に対応し、局所性を優先する。LRU-managed cache が大半の選択を常駐させ、IO を重ねるのではなく load 自体を減らす。decode CUDA graph に capture した単一融合 kernel でミスを解決し、共有選択により予測ではなく厳密にできるモデルだけで prefetch を使う。

**疎アテンション。** 初期の architecture は fixed pattern、content-based bucketing、low-rank projection でアテンションを制限する [Bel20, Kit20, Wan20a]。DSA/NSA などの学習型 query-dependent sparse-attention architecture と hybrid attention model は、attention compute と active KV read を減らす [Dee25a, Zen26, Yua25e, Dee26]。Quest などの学習不要手法は推論時に query-dependent KV page を選ぶ [Tan24]。HiSparse は新しい sparse-attention algorithm ではなく、これらの algorithm の inactive state を HBM capacity 上で安くする serving-system layer である。

<span id="section-7"></span>

## 7 結論

Top-$k$ 疎アテンションを使っても、サービングシステムは各デコードステップでほとんど読まない KV entry に full-context HBM rent を払う。HiSparse は論理 availability と物理 residency を分離して不整合を解消する。各リクエストの完全な KV history を host memory に置き、decode footprint は小さな GPU cache で制限する。LRU management が selection locality を hit に変え、融合 kernel が decode CUDA graph 内で各レイヤーの選択を解決し、レイヤー間で選択を共有するモデルでは prefetch を厳密にして残る IO の多くを計算と重ねる。変えるのは KV placement だけなのでモデル出力は同じである。H200、B200、GH200 上の DSA、NSA、Quest で、HiSparse は同程度の per-token latency において peak long-context generation throughput を最大 $4.7\times$ 改善する。no-IO oracle は resolve mechanism 自体に測定可能な cost がなく、厳密 prefetch が任意の IO-hiding scheme の throughput 上限の $85\%$ に達することを示す。

さらに、decode-side HBM の制限により、HBM だけで保持できる範囲を超えて処理可能なコンテキストを拡張し、上限を host-tier capacity に移せる。HiSparse は upstream SGLang に含まれる。より広い教訓は co-design にある。疎アテンションは per-step KV demand を小さく予測可能にし、モデルとハードウェアの小さな対応、すなわち選択の共有または早期通知と高速な host-device link が、階層型 KV placement の利得を重ねる。

## 謝辞

HiSparse へのオープンソース貢献について、Alibaba Cloud TairKVCache team、Ant Group SCT Inference team、Baidu Baige AI team、Zhipu AI に感謝する。Alibaba Cloud の Shangming Cai、Teng Ma、Xingyu Ling から建設的な feedback をいただいた。本研究は Stanford Platform Lab とその affiliates の支援を一部受けた。Zhiqiang Xie は NVIDIA Graduate Fellowship の支援を受けた。計算資源を提供した RadixArk に感謝する。

<span id="section-8"></span>

## 8 SGLang での実装

この appendix は、HiSparse が SGLang に追加するものと、既存 engine への接続箇所をまとめる。単一 flag で有効化し、それ以外の serving stack は変えない。6 module にわたる約 $2{,}200$ 行の新しい Python、CUDA kernel header、scheduler、model runner、attention backend、disaggregation path への統合変更からなる。

**新規 component。** *Coordinator*（`managers/hisparse_coordinator.py`、約 ${\sim}1{,}000$ 行）は [第 3.3 節](#section-3-3) の request lifecycle を所有する。prefill KV を host pool へ staging し、per-request-per-layer GPU cache を割り当て、拡張し、生成 KV を write-through し、[第 3.5 節](#section-3-5) の plan-then-IO prefetch group を含む swap-in を調整する。*Fused kernel*（`jit_kernel/hisparse.py` と `csrc/hisparse.cuh`）は token-level、compressed KV layout 向けの Resolve を実装し、任意で miss plan を記録する。shared layer で plan を replay する copy-only kernel も含む。*Memory layer*（`mem_cache/allocator/hisparse.py`、`hisparse_memory_pool.py`、`pool_host/hisparse.py`）は paged host-pool allocation、device-side cache pool、SGLang の既存 host KV cache class に対する mixin を提供する。小さな *configuration module*（`arg_groups/hisparse_hook.py`）が backend default を適用し、attention-backend と KV-dtype の互換性を検証する。

**Engine integration。** [第 3 節](#section-3) の機構以外では、engineering work は 3 箇所に集中する。Concurrency：staging、write-through、prefetch は独立した CUDA stream で動き、event により compute stream と順序付ける。scheduler は staging acknowledgment 後にだけ request を admit し、retract と pause path は通常の KV state とともに HiSparse state を解放する。Graph capture：Resolve と prefetch fork は SGLang の steady-state decode CUDA graph 内へ capture され、すべての metadata update と IO issue が host-side branch なしで replay 可能でなければならない。Compatibility：model runner は selection-sharing group を宣言するモデルで厳密 prefetch を自動的に有効にし、pipeline parallelism または speculative decoding では同期 swap-in に戻る。HIP kernel variant は AMD GPU をサポートする。disaggregated mode では、prefill instance が既存 transfer backend を通じて decode host の DRAM pool へ KV を直接書く。

**HiCache との関係。** SGLang の HiCache [Xie25d] は階層型*prefix* cache である。host と storage tier により、終了済み request の KV を、prefix を共有する後続 request が再利用できる。HiSparse は mixin を通じて HiCache の host-tier infrastructure、すなわち pinned host KV pool と IO backend を再利用するが、管理対象は異なる。モデル自身の疎な選択に従う per-request decode working set である。deployment では 2 つが prefill-decode dual を構成する。prefill node の HiCache は prefix を再利用して prefill work を減らし、decode node の HiSparse は per-request residency を制限して decode batch を拡大する。GPU-assisted IO path は Strata [Xie25a] に従う。

**設定。** `--enable-hisparse` で HiSparse を有効にする。JSON `--hisparse-config` は `top_k`、`device_buffer_size`（$B$）、`host_to_device_ratio`（device KV budget に対する host-pool capacity）、[第 3.4 節](#section-3-4) の swap-in transfer block size を設定する。厳密 prefetch は shared-index model で自動有効になり、ablation では環境変数で無効にできる。[第 4 節](#section-4) の全実験はこれらの switch を使い、コード変更はない。
