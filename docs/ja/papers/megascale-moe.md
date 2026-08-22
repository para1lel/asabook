---
title: 'MegaScale-MoE'
createTime: 2026/08/22 13:02:12
permalink: /ja/papers/megascale-moe/
---

> [Chao Jin](https://dblp.org/pid/19/4764-7) [+equal]、[Ziheng Jiang](https://dblp.org/pid/14/8980) [+equal]、[Zhihao Bai](https://dblp.org/pid/234/8717)、[Zheng Zhong](https://dblp.org/pid/69/7279)、[Juncai Liu](https://dblp.org/pid/304/3355)、[Xiang Li](https://dblp.org/pid/40/1491-67)、[Ningxin Zheng](https://dblp.org/pid/234/5381)、[Xi Wang](https://dblp.org/pid/08/5760)、[Cong Xie](https://dblp.org/pid/130/0102)、[Qi Huang](https://dblp.org/pid/46/4397-1)、[Wen Heng](https://dblp.org/pid/201/7460)、[Yiyuan Ma](https://dblp.org/pid/234/3589)、[Wenlei Bao](https://dblp.org/pid/162/4919)、[Size Zheng](https://dblp.org/pid/254/6617-1)、[Yanghua Peng](https://dblp.org/pid/195/5934)、[Haibin Lin](https://dblp.org/pid/142/1829)、[Xuanzhe Liu](https://dblp.org/pid/08/2161)、[Xin Jin](https://dblp.org/pid/68/3340-8)、[Xin Liu](https://dblp.org/pid/76/1820-86)。2025 年 5 月 16 日に arXiv へ初回投稿、現行版は v3。EuroSys '26 採択；[DOI 10.1145/3767295.3769325](https://doi.org/10.1145/3767295.3769325)。[MegaScale-MoE: Large-Scale Communication-Efficient Training of Mixture-of-Experts Models in Production](https://arxiv.org/abs/2505.11432)。[原論文 PDF](/paper/megascale-moe.pdf)。[TeX ソース](https://export.arxiv.org/e-print/2505.11432v3)。正確な印刷レイアウトと参考文献については、原論文 PDF を正本とする。

[+equal]: 同等の貢献。

## 概要

大規模な Mixture-of-Experts（MoE）モデルを効率よく学習するために設計された本番システム MegaScale-MoE を提案する。MoE は、大規模言語モデル（LLM）をこれまでにない規模へ拡大し、モデル性能を高める有望なアーキテクチャとして台頭している。しかし、既存の MoE 学習システムでは学習効率が低下しており、この問題は MoE モデルの大規模化とハードウェアの継続的な進歩によってさらに深刻になっている。

効率的な通信が MoE 学習の高速化に不可欠であることを踏まえ、MegaScale-MoE は各 MoE 層の attention と FFN に通信効率のよい並列化戦略を個別に適用し、オペレータ間とオペレータ内の両方で通信と計算を重ね合わせる包括的な手法を採用する。さらに MegaScale-MoE は、通信パターンを調整した低精度の通信圧縮を適用し、学習効率を一段と高める。1,440 基の NVIDIA Hopper GPU 上で 352B MoE モデルを学習した場合、MegaScale-MoE は 1.41M tokens/s の学習スループットを達成し、Megatron-LM と比べて効率を $1.88\times$ 改善した。MoE 学習を高速化する運用経験を共有し、システム設計から得た知見を示すことで、本研究が今後の MoE システム研究を促すことを期待する。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）[Cho22b, Tou23a, Jia24a] のサイズが増すにつれ、その学習規模も拡大している。学習規模の拡大により、効率の改善は望ましいだけでなく不可欠なものになった [Jia24f]。数十億人のユーザーに向けた AI 製品を開発する企業として、私たちは数千基の GPU を用い、数千億パラメータの LLM を学習し続けている。したがって、学習効率がわずかに向上するだけでも、計算資源の消費量と学習時間を大きく削減でき、最先端 LLM の開発可能性と持続可能性に直接影響する。

LLM アーキテクチャの中で、Mixture-of-Experts（MoE）モデルは疎な活性化 [Cho22b, Jia24a, Fed22, Sha17] を特徴とし、入力トークンをすべてのパラメータへ送るのではなく、*expert* と呼ばれる専用ネットワーク構成要素のうち、選択された一群へ動的にルーティングする。この設計では、モデルサイズが増加しても必要な FLOPs は線形未満にしか増えないため、計算コストを大幅に削減できる。近年の産業界における進展 [Du22, Raj22, Dbr25, Xai24, Dee24a] は MoE モデルの可能性を示しており、同等のモデル品質を持つ dense モデルと比べて学習コストを 1 桁削減している。

MoE モデルは学習コストが低いにもかかわらず、システムの観点から見ると、学習時には通信が重大な性能ボトルネックになる。たとえば NVIDIA Hopper GPU 上で社内モデルを学習した場合、通信は forward pass の総時間の 43.6%、学習処理全体の 32% を占める。このボトルネックには、主に 2 つの要因がある。第一に、MoE モデルは本質的に通信オーバーヘッドが大きい。MoE モデルはパラメータ数が多いため、dense モデルの学習と比べて、モデル並列化のためにより多くの GPU へ分散させる必要がある。第二に、疎な計算を可能にするには、forward pass と backward pass の双方で、トークンの dispatch と集約を行う 2 回の追加 all-to-all 通信がそれぞれ必要となり、進行中の計算が妨げられる。

さらに、ハードウェアの進歩に伴って計算と通信の不均衡が広がり、通信オーバーヘッドの比重はますます高くなっている。モデルアーキテクチャの改良と並行してハードウェア性能も急速に向上し、GPU の処理速度は大幅に高まった（[図 1](#figure-01)）。同時に、効率的かつ費用対効果の高い学習のために、学習精度の低減も採用されている [Pen23e, Dee24a]。こうした傾向によって生の計算時間が短くなる一方、通信オーバーヘッドの相対的な影響がより重大なボトルネックとなる。たとえば、既存の tensor parallelism をそのまま複数ノードへ拡張すると、場合によっては通信オーバーヘッドが 50% を超えることが確認されている。したがって、MoE モデル学習のスケーラビリティを維持・改善するには通信の最適化が欠かせず、複数の GPU 間で頻繁なデータ同期を必要とする分散環境では特に重要となる。

<span id="figure-01"></span>

![図 1。NVIDIA GPU の進化。](../../papers/megascale-moe/figure-01.png)

**図 1。** NVIDIA GPU の進化。

本稿では、大規模 MoE 学習を効率化するよう最適化された本番システム MegaScale-MoE の設計、実装、運用経験を示す。MegaScale-MoE は通信ボトルネックに綿密に対処することで、MoE 学習の限界を押し広げ、性能と効率を大幅に改善することを目指す。MoE モデルと dense モデルのアーキテクチャ上の主な違いは層内にあり、それが通信オーバーヘッドの主因であるという知見に基づき、MegaScale-MoE は高帯域幅の NVLink を利用して、各 MoE 層を 1 ノード内に収める。私たちの分析（[第 3 節](#section-3)）と評価（[第 6 節](#section-6)）は、既存システム [Dee24a, Hwa23] で一般的なノード間 expert parallelism を使わなくても、本手法が数千基の GPU 上で数千億パラメータのモデルまで MoE 学習を効果的に拡張できることを示している。

具体的には、MegaScale-MoE は MoE 学習の通信問題を 3 つの側面から解決する。第一に、MegaScale-MoE は各 MoE 層の attention モジュールと FFN モジュールに並列化戦略を個別に適用し、通信量を削減する。既存の LLM 学習フレームワークにおける並列化戦略を比較し、通信量や通信を効果的にオーバーラップできるか（すなわち、クリティカルパス上にあるか）など、大規模学習への影響を包括的に検討する。この分析に基づき、MoE 学習に最適な並列化戦略の組み合わせを選択する。

第二に、MegaScale-MoE はオペレータ単位で通信と計算を完全にオーバーラップさせる。MegaScale-MoE は各 MoE 層の forward pass と backward pass を、個別の計算オペレータと通信オペレータへ分割する。オペレータ間のオーバーラップでは、MegaScale-MoE は forward propagation と backward propagation の双方で通信オペレータと計算オペレータを慎重に並べ替える包括的なスケジューリング戦略を採用し、独立した計算の中に通信を隠蔽する。この手法は GPU メモリ使用量も最適化する。MegaScale-MoE は selective activation rematerialization を用い、forward pass 中は一部の activation のみを GPU メモリに保持し、backward pass 中に必要な activation を再計算または再通信によって取得する。この包括的なスケジューリングにより、MegaScale-MoE は rematerialization のオーバーヘッドを効果的に隠蔽し、activation の半分だけを保存しながら同等の性能を実現する。

クリティカルパス上の通信をオーバーラップさせるため、MegaScale-MoE は通信を tile に分割して GPU の計算パターンに合わせ、その tile 単位の通信を計算カーネルへ融合する細粒度の手法を採用する。token dispatch を伴う MoE モデルでは、MegaScale-MoE は効率的なローカル scatter 操作をカーネルへ融合し、scatter された次元に沿って計算タスクを再編成することで、複数のデータソースに起因する通信ボトルネックを緩和する。この細粒度オーバーラップは各ノード内で行われ、GPU 間の高帯域幅接続を利用する。

第三に、MegaScale-MoE は通信圧縮を利用し、MoE 学習の効率をさらに高める。具体的には、広く用いられている BF16 mixed-precision 学習において、MegaScale-MoE はノード間パラメータ同期の精度を FP32 から BF16 へ落とし、それに伴うオーバーヘッドを半減する。FP8 学習では、MegaScale-MoE は BF16 reduce-scatter を FP8 通信に置き換え、専用の量子化戦略と FP32 reduction を組み込むことで、収束の安定性を保ちながら通信量を削減する。

MegaScale-MoE は、製品向け MoE モデルを学習するために私たちのデータセンターへ導入されている。最先端のオープンソース LLM 学習フレームワークである Megatron-LM [Sho19] と比較して、MegaScale-MoE は 1,440 基の NVIDIA Hopper GPU 上で 352B MoE モデルを学習した場合、MFU（Model FLOPs Utilization）を最大 $1.88\times$ 向上させる。包括的な通信最適化を備えた MegaScale-MoE は、私たちの本番環境で大規模学習を支え、数百万 GPU 時間を削減しながら、数兆パラメータと数千基の GPU へ効率よく拡張している。

<span id="section-2"></span>

## 2 背景

<span id="figure-02"></span>

![図 2。Mixture-of-Experts（MoE）層。](../../papers/megascale-moe/figure-02.png)

**図 2。** Mixture-of-Experts（MoE）層。

<span id="section-2-1"></span>

### 2.1 Transformer における Mixture-of-Experts

Mixture of Experts（MoE）機構は、Transformer [Vas17] モデルの性能を高めるために設計された高度な手法であり、これらのモデルは LLM [Jia24a, Cho22b, Dbr25, Dee24a] の領域でますます重要になっている。これは、feed-forward network（FFN）構成要素に複数の expert network を統合することで Transformer アーキテクチャを拡張する。[図 2](#figure-02) に示すように、MoE モデルは入力トークンの特性に基づいて、最も関連性の高い expert へ動的にルーティングする。このルーティングは学習可能な gating 機構によって管理され、各トークンに最も適した expert が選ばれる。このアーキテクチャ上の工夫により、各入力に対して活性化される expert は一部だけであるため、推論コストを比例して増やすことなく MoE モデルの容量を拡大できる。

<span id="section-2-2"></span>

### 2.2 大規模 LLM 学習

数万基の GPU 上で大規模言語モデルを大規模に学習することは、複数のシステム技術を必要とする複雑なシステムエンジニアリング上の課題である。学習ワークロードを分散するには、data parallelism、tensor parallelism、pipeline parallelism などを組み合わせる必要があり [Sho19, Ras20, Jia24f]、各手法には限界があるため、1 つの手法だけでは効果的にスケールできない。

**Data parallelism** は学習データをすべてのデバイスへ均等に分散し、各デバイスがモデルパラメータと optimizer state の複製を保持する。各学習 iteration の後にパラメータを同期するため、data parallelism は all-reduce 通信を実行する。Zero Redundancy Optimizer（ZeRO）[Raj20] は、参加するすべてのデバイスへ model state を分散することで data parallelism を改善する。ZeRO は段階的な 3 つの stage からなり、stage が進むほどメモリを節約できるよう設計されているが、その代償として通信量が増加する。

**Tensor parallelism** は計算負荷の高いテンソル演算を複数のデバイスへ分散し、並列計算によって学習処理を大幅に高速化する。具体的な分割戦略とモデル内のオペレータ間依存関係によって、tensor parallelism では分割された入力の収集（all-gather）や出力の統合（reduce-scatter）が必要になる場合がある。LLM 学習では、LayerNorm や Dropout のようなオペレータは計算負荷が比較的低い一方で、多くの activation memory を必要とする。この問題に対処するため、これらのオペレータを sequence length の次元に沿って分割する **sequence parallelism** [Kor22] という tensor parallelism の変種が提案されている。long-context 学習では、複数の研究 [Sho19, Sam23, Con25a] が self-attention 内の異なるオペレータへ sequence parallelism または tensor parallelism を適用している。[図 3](#figure-03) は、attention における主要な並列化戦略である tensor、sequence、context parallelism（TP、SP、CP）を示しており、これらは [第 3.1 節](#section-3-1) で分析する。

<span id="figure-03"></span>

![図 3。self-attention に対する各種並列化戦略。"TP" は hidden size の次元に沿った分割を表し、"SP" は sequence length の次元に沿った分割を表す。](../../papers/megascale-moe/figure-03.png)

**図 3。** self-attention に対する各種並列化戦略。"TP" は hidden size の次元に沿った分割を表し、"SP" は sequence length の次元に沿った分割を表す。

**Pipeline parallelism** はモデル層を stage に分割し、それぞれを異なるデバイスで処理して pipeline 実行を可能にすることで、効率を高める。そのために、各 batch は複数の micro-batch へ分割される。pipeline bubble を最小化するため、GPipe [Hua19]、PipeDream 1F1B [Nar19]、Interleaved 1F1B [Nar21] など、さまざまなスケジューリング戦略が開発されている。Megatron-LM は Interleaved 1F1B pipeline scheduling を採用し、1 つのデバイス上にある各 stage をさらに複数の virtual stage へ分割することで、pipeline bubble 率を低減している。

**Expert parallelism** は、expert を複数のデバイスへ分散してメモリ負荷を軽減し、並列処理を可能にする、MoE モデルの学習に特化した手法である。トークンを適切な expert へ効率よく割り当て、その出力を取得するため、通常は all-to-all 通信が用いられる。

<span id="section-3"></span>

## 3 通信効率のよい並列化

MoE モデルの普及とハードウェア計算能力の進歩に伴い、本番環境の MoE 学習では通信オーバーヘッドがますます重大な問題になっている。本節では、通信量を削減し、高い GEMM（General Matrix Multiplication）効率など、その他の学習要件も満たすために用いる並列化戦略を掘り下げる。

<span id="figure-04"></span>

![図 4。大規模 MoE 学習の設計空間。](../../papers/megascale-moe/figure-04.png)

**図 4。** 大規模 MoE 学習の設計空間。

[図 4](#figure-04) は、最外層の data parallelism を除いた、大規模 MoE 学習の並列化戦略の設計空間を示す。まずノード間並列化から検討する。Expert parallelism は expert を複数ノードへ分散して MoE モデルの大規模なパラメータによるメモリ負荷を軽減するが、層ごとにノード間通信が発生するため、学習効率が低下する。同様に、tensor parallelism は通信オーバーヘッドが大きいため、TP を 1 ノード内に限定するほうが効率的である。先行研究 [Jia24f] に従い、モデルパラメータを分散し、通信量を削減し、異なる micro-batch の通信をオーバーラップさせるために pipeline parallelism を採用する。

Megatron-LM [Sho19] や DeepSpeed-MoE [Raj22] など、従来の大規模 MoE 学習システムは、ノード内でモデルパラメータを分割し、学習を拡張するために tensor parallelism を組み込んでいる。しかし実運用では、この手法に 2 つの問題があることを確認した：（1）TP は expert dimension を分割するため GEMM 効率が低下する；（2）TP は大きな通信オーバーヘッドを生み、このオーバーヘッドは並列数を増やしても一定であるため、最新ハードウェアでは最終的に通信時間が計算時間を上回る。

これらの問題に対処するため、MoE モデルの構成要素ごとに並列化戦略を調整する。feed-forward network（すなわち expert）では tensor parallelism を expert parallelism に置き換え、top-k と expert size の違いに合わせて最適化した独自の通信モードを用いることで、通信オーバーヘッドを tensor parallelism より小さく保つ。その他の構成要素には sequence parallelism を適用し、batch dimension ではなく sequence dimension に沿って分割することで、global batch size を増やさずにスケールできるようにする。これにより、tensor parallelism と比べてクリティカルパス上の通信も削減される。構成要素間のパラメータの非対称性により、追加のメモリと DP 通信のオーバーヘッドは管理可能な範囲に収まる。以降では、このノード内並列化戦略の根拠と分析を詳述する。[表 1](#table-01) に主要な記号を示す。

<span id="section-3-1"></span>

### 3.1 Attention の sequence parallelism

MoE モデルの expert 構成要素は本質的に並列化できるため、MoE 学習に関する従来研究 [Raj22, Li23i] の多くは expert parallelism の最適化に注力し、attention など MoE 以外の構成要素には通常 data parallelism（DP）を適用している。しかし MoE 学習を大規模化すると、この手法では activation memory の消費量が $n\times$ になるため不十分である。この問題は、DP がノード間とノード内の両方で batch dimension を分割することから生じる。[図 4](#figure-04) に示す他のノード内並列化戦略と比べて、attention に DP を適用すると、ノード内の各 GPU が 1 つの micro-batch を同時に処理することになり、activation size が $8\times$ に増加して、しばしばメモリ不足が発生する。

<span id="table-01"></span>

![表 1。記号の説明。](../../papers/megascale-moe/table-01.png)

**表 1。** 記号の説明。

MoE 学習をスケール可能にするには、attention モジュールのノード内並列化が不可欠である。ノード内の attention 演算を並列化する手法として、tensor parallelism（TP）が一般に用いられる。しかし、クリティカルパス上で activation の all-gather と reduce-scatter を行うため、避けられない通信コストが発生する。計算 FLOPs と通信帯域幅の差が広がるにつれ、TP の通信オーバーヘッドが self-attention の計算時間さえ上回る場合があることがわかった。この通信支配のボトルネックは、通信と計算をオーバーラップさせる能力を制限し、最終的に学習効率を低下させる。

DeepSpeed-Ulysses [Sam23] で提案された sequence parallelism（SP）を採用し、MoE 学習を拡張するとともに、クリティカルパス上の通信を効果的に削減する。SP は一般に、長い入力に伴うメモリ問題へ対処する long-context 学習で用いられる。SP は大規模 MoE 学習にも適していることがわかった。第一に、特に grouped-query attention [Ain23a] を用いる場合、TP と比べて通信オーバーヘッドが大幅に減る。第二に、パラメータの重複とパラメータ同期時の通信オーバーヘッドは増えるものの、MoE モデル固有の特性によって、こうしたトレードオフは管理可能かつ許容可能である。

**通信効率。** TP を用いる場合、attention の通信量は次のとおりである。

<span id="equation-01"></span>

$$
2bsh(n-1)/n.
$$

SP を用いると、通信量は次の値まで減少する。

<span id="equation-02"></span>

$$
2bsh(n-1)/n\times(2+2/m)/n,
$$

ここで $m$ は query head 数と key-value head 数の比率を表す。モデルを、NVLink domain size が 8 の NVIDIA Hopper GPU ワークステーション上で学習すると仮定すると、sequence parallel attention の通信 latency は tensor-parallel attention で必要な値のおよそ 4 分の 1 まで削減できる。

<span id="figure-05"></span>

![図 5。SP attention におけるパラメータ同期の階層型通信。](../../papers/megascale-moe/figure-05.png)

**図 5。** SP attention におけるパラメータ同期の階層型通信。

**データ通信とメモリオーバーヘッド。** SP attention と TP attention の顕著な違いはパラメータをデバイス間へ分散する方法にあり、TP は attention weight を shard 化する一方、SP はそれを複製する。このため、gradient とパラメータの同期に伴う通信オーバーヘッドが増える可能性が懸念される。直感に反するが、ノード内・ノード間の帯域幅の非対称性と、[図 5](#figure-05) に示し [第 10.1 節](#section-10-1) で分析する、最新の通信ライブラリ [Ncc21] における階層型通信操作の採用を考慮すると、SP attention は TP attention と比べて $n\times$ 多くのパラメータを同期する必要があるにもかかわらず、実際のシナリオでは通信オーバーヘッドの差はわずかである。

一方、SP attention によって増える GPU メモリ消費量は、MoE 学習ではわずかである。expert が数十から数百ある大規模 MoE モデルでは、GPU メモリの大部分を expert parameter が消費する。[第 6.2 節](#section-6-2) で詳述する実験から、SP attention による追加のパラメータ同期とメモリオーバーヘッドが管理可能な範囲に収まることを確認した。

**均衡と不均衡。** Ulysses 形式の SP attention に加え、すべての activation を sequence dimension に沿って分割する context parallelism（CP）[Con25a] など、ほかの形式も検討した。しかし CP attention では、各トークンがそれより前のトークンだけを参照するため、attention の causal masking によってワークロードが不均衡になる。この問題を緩和するため、sequence の先頭側と末尾側の partition を同じ GPU 上で group 化する zigzag 戦略を試したが、完全な均衡を実現することは依然として難しい。そのため、大規模学習では、学習処理全体が最も不均衡な data batch によって制約されることが多い。さらに、この不均衡は training pipeline を乱し、学習効率全体を低下させる。

<span id="section-3-2"></span>

### 3.2 Feed-forward network の expert parallelism

<span id="figure-06"></span>

![図 6。通信効率のよい expert parallelism。$e$ は worker へルーティングされるトークン数を表す。](../../papers/megascale-moe/figure-06.png)

**図 6。** 通信効率のよい expert parallelism。$e$ は worker へルーティングされるトークン数を表す。

feed-forward network 構成要素の並列化戦略を選択する際、expert parallelism（EP）は一貫して tensor parallelism より優れている。TP は各 expert の hidden dimension を分割して GEMM 効率を低下させるのに対し、EP は各デバイス上で expert の計算を完全な形に保つ。理論上、EP の通信コストは次のとおりである。

<span id="equation-03"></span>

$$
2k/n\times bsh(n-1)/n,
$$

一方、TP では次のとおりである。

<span id="equation-04"></span>

$$
2bsh(n-1)/n.
$$

両者の相対効率は比率 $k/n$ に依存するが、EP の通信量を最小化するため、異なる top-$k$ 値に対応する適応的な通信戦略を設計した。

<span id="figure-07"></span>

![図 7。token dispatch における AG、RS、A2A の比較。](../../papers/megascale-moe/figure-07.png)

**図 7。** token dispatch における AG、RS、A2A の比較。

**効率的な通信パターン。** [図 6](#figure-06) は一般的な EP 実装と MegaScale-MoE の手法を比較している。標準的な EP 実装では、token dispatch と集約のために 2 回の all-to-all 通信が必要となる。さらに、同じ expert に割り当てられたトークンを連続したメモリ空間へ置くため、トークンの送信前と受信後に scatter 操作が必要になる場合がある。

top-$k$ の値が $n$ を超える場合、従来の all-to-all 通信を all-gather と reduce-scatter に置き換える。まず all-gather 操作で、すべての worker からトークンを収集する。次に、ローカル scatter 操作で不要なトークンを破棄し、現在の worker 上の expert が必要とするトークンだけを残す。expert の計算後、トークンを完全な tensor へ組み立てる。この手法では、通信前に gather 操作を実行し、その後 reduce-scatter で最終結果を生成できるため、EP の通信オーバーヘッドが TP 以下に保たれる。

実際の学習では、all-to-all は各 worker がほかのすべての worker と通信する必要があるのに対し、all-gather と reduce-scatter は隣接 worker のみと通信する ring-based 通信パターンに従うため、all-to-all 通信は all-gather や reduce-scatter より効率が悪い。[図 7](#figure-07) に示す Mixtral-$8\times$7B での 3 操作の通信時間から、top-$k$ > 6 の場合は all-gather ベースの EP 実装のほうが効率的であることがわかる。

**効率的なオペレータ。** Megatron-LM のように tensor の scatter と gather に `torch.scatter_add` と `torch.gather` を用いる代わりに、CUDA で効率的な scatter operator と gather operator を直接開発した。token routing の結果に基づき、入力 tensor の各 row（1 つのトークンを表す）から、出力 tensor 内の対応する row への mapping を事前計算する。scatter operator と gather operator は、この mapping に従ってデータ転送を効率よく実行する。

**負荷分散。** MoE モデル学習におけるよく知られた課題の 1 つは、expert 間の負荷分散である [Li23i, Dee24d]。これに対処するため、auxiliary loss と token dropping を用いて、各ノード内の GPU 間でワークロードを均衡させる。DeepSeek-V2 [Dee24d] と同様に、同じ GPU 上に配置された expert を 1 つの group として扱い、個々の expert ではなく各デバイスについて balance loss と computational capacity を計算する。

<span id="figure-08"></span>

![図 8。Selective activation rematerialization。](../../papers/megascale-moe/figure-08.png)

**図 8。** Selective activation rematerialization。

<span id="section-4"></span>

## 4 通信と計算のオーバーラップ

通信量を最小化するよう並列化戦略を最適化した後、通信と計算を包括的にオーバーラップさせる技術を用いて、通信オーバーヘッドをほぼゼロまで削減する。大規模モデルの学習ではさまざまな技術を統合するため、通信オーバーラップは複雑になる。たとえば、ある時点でデバイスが計算カーネルと通信カーネルを同時に処理し、PP 通信と DP 通信をオーバーラップさせ、デバイスとホスト間のデータ転送を管理する場合がある。Megatron-LM などの既存フレームワークは attention モジュールと FFN モジュールを MoE 層へ組み立て、backward propagation を `torch.autograd` パッケージに依存するため、通信をオーバーラップさせる柔軟性が制限される。これに対し MegaScale-MoE は、各 MoE 層の attention モジュールと FFN モジュールを GPU kernel として動作するオペレータへ分解し、柔軟なスケジューリングによる細粒度の通信オーバーラップを可能にする。

<span id="section-4-1"></span>

### 4.1 オペレータ間オーバーラップ

通信オペレータと独立した計算オペレータを異なる CUDA stream 上で非同期に実行し、両者をオーバーラップさせる。学習処理で最適な性能を実現するため、専用に手作業で調整した包括的なスケジューリング戦略を採用する。

**包括的なスケジューリング。** caller の観点からは、MoE 層全体の forward pass と backward pass を実行する統合 macro module を実装し、スケジューリングの柔軟性を広げる。たとえば backward pass では、効率を高めるため、さまざまな通信オペレータを activation recomputation など、依存関係のない計算とオーバーラップさせられる。runtime の観点では、blocking を防いで throughput を最大化するようリソース競合を解決し、同時実行される通信タスクを効率よく管理することが主要な課題となる。そのためには、干渉を最小化して全体の throughput を最適化するために各通信オペレータへ割り当てる SM 数を決めるなど、慎重な調整が必要である。

<span id="figure-09"></span>

![図 9。rematerialization における activation shape。](../../papers/megascale-moe/figure-09.png)

**図 9。** rematerialization における activation shape。

<span id="figure-10"></span>

![図 10。細粒度のオペレータ内通信・計算オーバーラップ。](../../papers/megascale-moe/figure-10.png)

**図 10。** 細粒度のオペレータ内通信・計算オーバーラップ。

**Selective activation rematerialization。** 包括的なスケジューリング戦略は、学習速度を損なわずにメモリ使用量を削減するうえでも役立つ。同等の計算量を必要とする dense モデルと比べて、MoE モデルはパラメータ数が数倍多いため、学習時のメモリ負荷が大幅に高い。DP group 間で重複する optimizer state を除去する ZeRO 最適化 [Raj20] に加え、selective activation rematerialization によってメモリ使用量をさらに最適化する。この手法は、ほかの必要なオペレータとオーバーラップできる計算オペレータと通信オペレータを再実行することで、activation memory の必要量を削減する。

[図 8a](#figure-08) は Mixtral [Jia24a] MoE 層の forward pass を示し、この処理で生成される主要な activation を強調している。MegaScale-MoE は、再計算の計算コストが高い activation を戦略的に保持し、メモリ負荷の高い操作または通信操作によって生成されるその他の activation は再計算する。これにより backward computation への依存を最小化し、rematerialization 操作をほかの計算や通信とオーバーラップさせて、クリティカルパスの遅延を回避できる。たとえば [図 8b](#figure-08) に示すように、FC2 の GroupedGEMM オペレータの backward pass は、入力として activation `fc2_in` と `fc2_out` の gradient（$\Delta$`fc2_out` と表記）を必要とする。MegaScale-MoE は `fc2_in` を再計算し、このオペレータを gradient communication（すなわち $\Delta$`ffn_out` の all-gather）とオーバーラップさせる。同様に、`ffn_in` は `RMSNorm` と all-gather を再実行して取得し、これらのオペレータはそれぞれ先行する通信と FC2 GroupedGEMM の中に隠蔽される。MegaScale-MoE は `ffn_out` を保存する必要をなくすため、SwiGLU [Sha20] activation function の直後に `ffn_out` の weighted sum も配置する。この並べ替えは、非線形境界をまたぐオペレータを避けることで計算の整合性を保証する。

[図 9](#figure-09) は forward propagation 中に生成される主要な activation の shape を示し、強調された activation は backward propagation 用に保持される。1 つの MoE 層内の model parallelism size を $n$、1 つの expert の intermediate hidden size を $fh$ とする。単一の MoE 層における総 activation は次のとおりである。

$$
(2n+2k+3kf+12+5/m)bsh/n,
$$

これを次の値まで削減した。

$$
(2kf+4+2/m)bsh/n.
$$

MegaScale-MoE は同じ学習速度を維持しながら、activation memory を $\sim 50\%$ 削減する。

<span id="section-4-2"></span>

### 4.2 オペレータ内オーバーラップ

オペレータ間オーバーラップは通信 latency を効果的に隠蔽するが、実行 timeline の bubble をすべて取り除くことは依然として容易ではなく、通信とオーバーラップできる rematerialization operator や gradient computation operator が存在しない forward pass では特に難しい。expert computation の token dispatch など、一部の forward operator は通信へ直接依存するため、別の micro-batch を導入しない限りオーバーラップできず、導入すればメモリ負荷が増加する。

広く採用されている解決策 [Jia24f, Tra25, Wan22b] は、オペレータをより小さな並列オペレータへ分解し、別々の CUDA stream 上で実行することで pipeline 化を可能にするものである。しかし、この手法では無視できないオーバーヘッドが生じる：$(i)$ 複雑な stream control には host の介入が伴い、CPU control の非決定的な性質によって不規則な bubble が発生する；$(ii)$ 末尾の計算が不完全になり、計算 latency 全体が増加する。

以上の問題に対処するため、直接依存関係にある通信オペレータと計算オペレータを並列化するオペレータ内オーバーラップを採用する。中心となる考え方は、これらのオペレータを融合し、ワークロードを tile へ分解することである。先行研究 [Jan22, Cha24c, Zha25e, Zhe25a] に従い、通信オペレータと計算オペレータの間に device memory 上の barrier を実装する。この barrier によって tile 単位の細粒度 notification が可能になり、host の介入が不要となるため、学習性能がさらに向上する。attention モジュールと FFN モジュール向けに、それぞれ GEMM とオーバーラップする kernel、および MoE GroupedGEMM とオーバーラップする kernel という 2 種類の kernel を実装する。

**GEMM とのオーバーラップ。** まず、GEMM kernel に対するオペレータ内の通信・計算オーバーラップを導入する。具体的には SP attention の Output Projection と QKV Projection に対して、それぞれ all-to-all（A2A）+GEMM kernel と GEMM+A2A kernel を実装し、ここで X+Y は Y が X の後に実行されることを意味する。[図 10](#figure-10) に A2A+GEMM の data flow とオーバーラップパターンを示す。ローカルデータに対する GEMM とリモートデータの通信は同時に開始される。データ転送には専用の GPU copy engine を利用し、すべての SM（streaming multiprocessor）が計算に完全に使用されるようにする。リモートの data tile がローカルメモリに到着すると、signal が GEMM kernel に通知し、到着した tile に対する計算を続行させる。GEMM+A2A では、all-to-all 操作を GEMM kernel へ融合する。GEMM 計算の各 tile の末尾でリモートデータ転送を行い、出力 data tile をリモート rank へ書き込む。tensor parallelism 向けに all-gather+GEMM kernel と GEMM+reduce-scatter kernel も実装しており、これらは A2A+GEMM と GEMM+A2A に類似している。

A2A+GEMM と GEMM+A2A では、all-to-all が all-gather や reduce-scatter より複雑なため、少数の SM を通信へ割り当てる。通信と計算の latency が同程度になるよう、通信用 SM の数を調整する。さらに、複数の rank が同じデバイスから同時に読み出したり、同じデバイスへ書き込んだりすると、NVLink で競合が発生する可能性がある。これを緩和するため swizzling [Cha24c, Zha25e, Zhe25a] を適用し、通信 tile の到着が計算 tile の進行速度と揃うよう、tile の通信と計算を並べ替える。

**GroupedGEMM とのオーバーラップ。** token dispatch と combine を伴う expert parallelism では、通信を GroupedGEMM とオーバーラップさせることを目指す。all-gather+scatter+GroupedGEMM と GroupedGEMM+gather+reduce-scatter という 2 種類のオーバーラップ kernel を実装する。GEMM kernel のオーバーラップ技術と異なり、MoE GroupedGEMM には token shuffling（scatter/gather）が必要である。そのため、各計算 tile が複数 rank のトークンに依存する場合がある。計算と通信を効果的にオーバーラップさせるため、各計算 tile が依存する rank 数を最小化するよう token order を sort する。さらに、各 tile は固有の依存関係を持つため、tile ごとの signal control は動的に決定される MoE routing に応じて変化する。

詳しく述べると、AG+scatter+GroupedGEMM では、トークンがルーティングされた expert index に基づき、sequence dimension に沿ってトークンを並べ替える。次に各 expert について、ルーティングされたトークンを source rank index に従って sort する。最後に、sort 済みの sequence を block へ分割し、一連の計算 tile を用いて GroupedGEMM を実行する。具体的には [図 10c](#figure-10) に示すように、index mapping に基づいて入力データの row を選択することで、ローカル scatter を kernel へ融合する。各 expert の GroupedGEMM 計算は tile へ分割され、各 tile は一部の source rank、場合によっては 1 つの source rank だけに依存する。これにより各計算 block の総待機時間が短縮され、expert parameter の重複ロードが避けられ、計算 tile と通信 tile のオーバーラップが改善される。

<span id="section-5"></span>

## 5 通信圧縮

通信圧縮を適用し、通信オーバーヘッドをさらに削減する。収束の安定性を保つため、mixed-precision 学習フレームワークでは通常、より正確に accumulation できるよう、reduction 待ちの tensor を FP32 などの高精度形式で転送する。data parallelism における gradient reduce-scatter は、その代表例である。

**DP 通信圧縮。** MoE モデルのパラメータが増えるにつれ、data parallelism におけるパラメータと gradient の同期に伴う通信オーバーヘッドも増加する。先行研究では、このコストを緩和する gradient compression が検討されてきた。私たちの BF16 mixed-precision 学習では、効率と収束の安定性のバランスを取りながら、gradient 同期に FP32 から BF16 への精度低減を慎重に適用する。

<span id="figure-11"></span>

![図 11。DP 通信圧縮。](../../papers/megascale-moe/figure-11.png)

**図 11。** DP 通信圧縮。

具体的には [図 11](#figure-11) に示すように、pipeline parallelism でローカルに gradient を accumulation する間、main gradient を FP32 のまま保持する。各 model stage で accumulation が完了した後、gradient 同期を reduce-scatter だけに頼る代わりに、gradient を BF16 へ cast し、data parallel group 内で all-to-all 通信を実行して必要な gradient shard を収集し、その後ローカルで FP32 により集約する。この手法では、FP32 で reduce-scatter を直接実行した場合と比べて精度低下が無視できるほど小さい一方、gradient 通信のオーバーヘッドを 50% 削減できることが結果からわかった。

この手法のリスクが小さい理由は主に 2 つある。第一に、通信時に accumulation 済み gradient を一度だけ BF16 へ変換し、ローカルの gradient accumulation は FP32 精度のまま維持する。第二に、BF16 gradient 通信に ring 形式の reduce を使わず all-to-all 通信を用い、最終 reduction を FP32 summation で計算する。この設計により、ring-based reduction で BF16 値を繰り返し accumulation することで生じ得る精度低下を防ぐ。

大きな gradient を cast して all-to-all 通信を実行すると、peak memory 消費量が増加し、メモリ不足エラーを引き起こす可能性があることを確認した。これを緩和するため、FP32 入力 buffer の半分に BF16 gradient を in-place で配置し、残り半分を BF16 all-to-all 通信の出力 buffer として用いる、メモリ効率のよいオペレータを開発し、peak memory の増加を防ぐ。

<span id="table-02"></span>

![表 2。評価に用いるモデル構成。](../../papers/megascale-moe/table-02.png)

**表 2。** 評価に用いるモデル構成。

**FP8 学習の通信圧縮。** 低精度の FP8 学習では計算時間が短くなるため、通信時間の割合が増加する。通信オーバーヘッドを緩和するため、適切な量子化技術を用い、FP8 精度によって通信量を圧縮する方法を検討する。現在は tensor parallelism を用いる FP8 MoE 学習へ通信圧縮を適用し、overflow や underflow が発生しやすい reduction scenario に注力している。たとえば、すべての tensor に E4M3 形式（4-bit exponent と 3-bit mantissa）を採用する。DP reduce-scatter 圧縮と同様に、forward propagation では BF16 TP reduce-scatter を FP8 all-to-all に置き換え、reduction を FP32 精度で実行する。対応する backward propagation では、gradient に FP8 all-gather を適用する。ただし、単に精度を落とすだけでは BF16 学習との loss のずれが生じる。これを緩和するため、forward 通信には token 単位の activation quantization、backward 通信には channel 単位の quantization を適用する。backward propagation ではさらに、小さな group size（例：128）を用い、token dimension に沿って quantization を group 化する。

<span id="section-6"></span>

## 6 評価

本節では、学習性能全体（[第 6.1 節](#section-6-1)）、MegaScale-MoE の主要な最適化に関する ablation study（[第 6.2 節](#section-6-2)）、precision-communication co-design の有効性（[第 6.3 節](#section-6-3)）を網羅する、MegaScale-MoE の包括的な評価を示す。[表 2](#table-02) は評価に用いた MoE モデルの構成を示しており、hidden size（$h$）、FFN intermediate size（$h_{\mathrm{ffn}}$）、expert 数、top-$k$ 値を詳述している。特記しない限り、評価は NVIDIA H800 GPU 上で行い、その仕様は [表 4](#table-04) に示す。

<span id="table-03"></span>

![表 3。NVIDIA H800 GPU を用いた 352B MoE モデルの strong-scaling 学習性能。throughput 列の括弧内の数値は、Megatron-LM に対する MegaScale-MoE の speedup を表す。](../../papers/megascale-moe/table-03.png)

**表 3。** NVIDIA H800 GPU を用いた 352B MoE モデルの strong-scaling 学習性能。throughput 列の括弧内の数値は、Megatron-LM に対する MegaScale-MoE の speedup を表す。

<span id="section-6-1"></span>

### 6.1 学習性能

MegaScale-MoE は、3D parallelism 戦略をサポートし、コミュニティの最新の最適化を継続的に取り込んでいる最先端のオープンソース LLM 学習システム Megatron-LM [Sho19] を基盤に構築されている。評価には、実験を開始した数か月前の時点で安定していたことから選択した、commit hash f1f03922 の GitHub 上の Megatron-LM [Meg25] を使用する。公平に比較するため、Megatron-LM と MegaScale-MoE で同じ global batch size を用い、それぞれのシステムに最適な並列化構成を選択する。具体的には、MegaScale-MoE は各ノード内で SP attention と EP を用いる一方、Megatron-LM は各ノード内で TP を採用し、両システムの PP size を 15 に設定する。すべての構成要素で TP size を統一するという要件を満たすよう、Megatron-LM の構成を調整する。[第 3.1 節](#section-3-1) で述べたように、Megatron-LM で TP size を 1 にすると許容できない $8\times$ の activation memory が必要となり（gradient checkpointing による低速な再計算でしか対処できない）、TP size を 8 にすると EP がノード間で動作することを強いられ、PP より多くの通信コストが発生する。なお、評価対象の両システムでは、data parallelism と pipeline parallelism に MegaScale [Jia24f] の通信・計算オーバーラップ技術を有効にしている。したがって、通信オーバーヘッドは主に TP、SP、EP などのノード内 model parallelism から生じる。sequence length は 8,192、vocabulary size は 65,536 である。

<span id="figure-12"></span>

![図 12。NVIDIA H800 GPU を用いた 352B MoE モデルの weak-scaling 学習性能。](../../papers/megascale-moe/figure-12.png)

**図 12。** NVIDIA H800 GPU を用いた 352B MoE モデルの weak-scaling 学習性能。

**スケーラビリティ。** [表 3](#table-03) は、352B MoE モデルにおける Megatron-LM と MegaScale-MoE の strong-scaling 学習性能を比較している。global batch size を 720 に固定したまま、GPU 数を増やす。すべての設定で、MegaScale-MoE は Megatron-LM に対して 1.65-$1.88\times$ の speedup を達成する。GPU 数の増加に伴い、MegaScale-MoE の MFU（Model FLOPs Utilization）は 32.48% から 27.89% へ低下する。これは、batch size が固定されており、GPU が増えるほど各 pipeline の micro-batch 数が減り、bubble が増えるため、予想どおりの結果である。

[図 12](#figure-12) は、同じモデルにおける Megatron-LM と MegaScale-MoE の weak-scaling 学習性能を示す。GPU 数（480 から 1,440）に比例して、global batch size を 360 から 1,080 へ増やす。MegaScale-MoE は Megatron-LM の 1.74-$1.79\times$ の学習 throughput を達成する。規模が大きくなると、通信オーバーヘッドの増加により Megatron-LM の throughput は 2.74% 低下する。これに対し MegaScale-MoE は、包括的な通信・計算オーバーラップの効果により、throughput の低下がわずか 0.2% にとどまり、ほぼ線形のスケーラビリティを示す。

<span id="figure-13"></span>

![図 13。異なる GPU で Mixtral-$8\times$7B を学習した場合の性能内訳。](../../papers/megascale-moe/figure-13.png)

**図 13。** 異なる GPU で Mixtral-$8\times$7B を学習した場合の性能内訳。

<span id="table-04"></span>

![表 4。各 NVIDIA GPU の仕様。](../../papers/megascale-moe/table-04.png)

**表 4。** 各 NVIDIA GPU の仕様。

**異なる GPU における性能内訳。** 本番環境で MoE モデルを学習する際の性能をさらに把握するため、MegaScale-MoE を詳しく分析する。32 基の NVIDIA H800、H20、A100 GPU 上で、それぞれ Mixtral-$8\times$7B を学習する。使用した GPU の仕様を [表 4](#table-04) に示す。DP size を 4、Megatron-LM の TP size を 8、MegaScale-MoE の SP size と EP size を 8 に設定する。[図 13b](#figure-13) に示すように、4 種類の GPU すべてで MegaScale-MoE は一貫して Megatron-LM を上回り、MFU は最大 $1.58\times$ 高い。[図 13a](#figure-13) は Megatron-LM と MegaScale-MoE の iteration time の内訳を示す。exposed communication time は、計算操作とオーバーラップしていない通信時間を表す。MFU の計算時には FlashAttention と GEMM を演算として数える。この性能向上は主に、MegaScale-MoE の通信効率のよい並列化戦略と、細粒度にオーバーラップした通信によるものである。

GPU の計算能力が高くなるほど MFU の値が低下することに注意されたい。dense モデルと異なり、MoE モデルには routing、ローカル scatter、gather などメモリ負荷の高い操作が多く含まれ、メモリ帯域幅は計算能力ほど速く向上しないため、これらの操作には依然として時間がかかるからである。さらに、GEMM もメモリ帯域幅に制約されるメモリロードへ依存するため、計算能力が高くなるほど GEMM 効率は低下する。

<span id="section-6-2"></span>

### 6.2 Ablation study

<span id="table-05"></span>

![表 5。240 基の NVIDIA H800 GPU と batch size 720 で 352B MoE モデルを学習した場合の throughput 改善の内訳。](../../papers/megascale-moe/table-05.png)

**表 5。** 240 基の NVIDIA H800 GPU と batch size 720 で 352B MoE モデルを学習した場合の throughput 改善の内訳。

MegaScale-MoE の最適化技術の有効性を評価する。まず、各技術を段階的に有効化し、性能全体への寄与を切り分ける体系的な内訳に関する実験を行う。[表 5](#table-05) は、240 基の GPU 上で global batch size 720 により 352B MoE モデルを学習した場合の、各種最適化による throughput 改善の内訳を示す。baseline は、attention と FFN の両方に TP を採用し、通信・計算オーバーラップを無効にした MegaScale-MoE である。まず、attention に SP、expert に EP という通信効率のよい戦略を適用することで、この baseline に対して throughput を 13% 改善する。次に、大規模 MoE 学習の主要なボトルネックである通信オーバーヘッドに取り組む。オペレータ間とオペレータ内のオーバーラップ手法は、これらのコストを効果的に隠蔽し、学習をそれぞれさらに 9% と 6% 高速化する。

<span id="figure-14"></span>

![図 14。各モデルの並列化効率。](../../papers/megascale-moe/figure-14.png)

**図 14。** 各モデルの並列化効率。

<span id="figure-15"></span>

![図 15。SP attention と TP attention におけるパラメータ同期時間。](../../papers/megascale-moe/figure-15.png)

**図 15。** SP attention と TP attention におけるパラメータ同期時間。

体系的な内訳に続き、動作をより深く理解するため、ほかの設定をすべて一定に保ちながら一度に 1 つの設定を変え、各構成要素の ablation study を行う。

<span id="figure-16"></span>

![図 16。各層におけるオーバーラップありの通信・計算時間とオーバーラップなしの時間の比較。M1-M6 は表 2 に上から順に記載した 6 モデルを表し、A2A、AG、RS はそれぞれ all-to-all、all-gather、reduce-scatter を指す。](../../papers/megascale-moe/figure-16.png)

**図 16。** 各層におけるオーバーラップありの通信・計算時間とオーバーラップなしの時間の比較。M1-M6 は [表 2](#table-02) に上から順に記載した 6 モデルを表し、A2A、AG、RS はそれぞれ all-to-all、all-gather、reduce-scatter を指す。

**並列化戦略。** 8 基の NVIDIA H800-SXM GPU を備えた単一ノードを用い、さまざまなノード内並列化戦略における学習効率を比較する。並列化戦略を X+Y と表記し、X は attention の並列化戦略、Y は expert の並列化戦略を表す。attention の並列化戦略には TP と本研究の SP があり、expert には TP と EP がある。最適化した並列化による性能上の効果だけを切り分けるため、ほかのシステム最適化は無効にする。

<span id="figure-17"></span>

![図 17。selective activation rematerialization（SAR）の ablation study。](../../papers/megascale-moe/figure-17.png)

**図 17。** selective activation rematerialization（SAR）の ablation study。

[表 2](#table-02) に示す多様なモデル構成を持つ 1 つの社内 MoE モデルと 5 つのオープンソース MoE モデルについて、学習時の MFU を測定する。global batch size を 32 に設定し、GPU メモリへ収まるよう各モデルの層数を調整する。[図 14](#figure-14) は、MegaScale-MoE の並列化戦略 SP+EP がほかの 3 つの並列化戦略を一貫して上回り、TP+TP と比べて MFU を 14.9%-32.9% 改善することを示している。性能向上には主に 2 つの要因がある。第一に、[第 3 節](#section-3) で述べたように、SP と EP は TP と比べて通信量を効果的に削減し、通信オーバーヘッドを低下させる。第二に、TP は FFN モジュールを intermediate size dimension に沿って分割するため、GEMM 効率が低下する。

並列化戦略をより包括的に評価するため、SP で attention parameter を複製することによる追加オーバーヘッドも報告する。メモリ使用量について、SP は TP よりメモリフットプリントが 1.2%-5.4% 大きく、7 モデル全体でパラメータ、gradient、optimizer state の保存に 1.7%-8.1% 多くのメモリを必要とする。SP による大幅な性能向上を考えれば、このオーバーヘッドは管理可能である。

パラメータ同期時間については、大規模学習の設定に従って TP または SP の size を 8 に設定し、各層を単一ノード内で実質的に並列化する。各 GPU 上の attention parameter size を 384 MB から 1536 MB まで変化させ、FFN parameter size は実際の典型的な学習設定を反映して GPU あたり 10 GB に固定する。SP attention と TP attention の MegaScale-MoE を 4 および 8 の DP group で実行し、それぞれ GPU 総数は 32 および 64 となる。[図 15](#figure-15) は、SP attention と TP attention の同期時間が一貫して同程度で、差はわずか 0.3%-3.1% であることを示す。これは、SP と TP の DP communication latency が同様の性能特性を示すという仮説と一致する。

**オペレータ内通信オーバーラップ。** 次に、[図 8](#figure-08) に示す forward pass の主要な 4 つの通信と、それに対応する計算オペレータの所要時間を測定する：$(i)$ QKV Projection と all-to-all、$(ii)$ all-to-all と Output Projection、$(iii)$ all-gather と scatter および GroupedGEMM、$(iv)$ GroupedGEMM と gather および reduce-scatter である。[図 16](#figure-16) は、6 モデルすべてにおいて、MegaScale-MoE が細粒度オーバーラップのない baseline と比べて、通信オペレータと計算オペレータの合計時間を 1.2-$4.7\times$ 短縮することを示している。また MegaScale-MoE は、オペレータ内の通信・計算オーバーラップによって、学習 iteration time を 7.1%-12.9% 削減する。

<span id="figure-18"></span>

![図 18。DP 通信圧縮を用いた MegaScale-MoE の学習 loss curve。](../../papers/megascale-moe/figure-18.png)

**図 18。** DP 通信圧縮を用いた MegaScale-MoE の学習 loss curve。

**Selective activation rematerialization。** MegaScale-MoE を、学習中にすべての activation を GPU メモリへ保存する、selective activation rematerialization を無効にした baseline（No SAR）と比較する。128 基の NVIDIA H800 GPU 上で Mixtral-$8\times$7B と Mixtral-$8\times$22B を学習し、両手法を評価する。[図 17](#figure-17) にメモリ使用量の内訳と学習 MFU を示す。No SAR と比べて、MegaScale-MoE は 2 モデルの activation memory 消費量をそれぞれ 45.5% と 57.2% 削減し、総メモリ使用量を 21.3% と 35% 削減する一方、学習性能の差を 0.5% 以内に保つ。

**Data parallelism の通信圧縮。** [第 5 節](#section-5) で説明した BF16 all-to-all DP 通信と FP32 reduce-scatter 通信を用いて 7B MoE モデルを学習し、通信圧縮技術の有効性を検証する。[図 18](#figure-18) に示す学習 loss curve は、ほぼ同一である。この最適化は batch の accumulation 済み gradient だけを圧縮し、通信時に限って BF16 と FP32 の間で変換するため、リスクは最小限である。

<span id="section-6-3"></span>

### 6.3 モデルの収束

MegaScale-MoE によるモデルの収束を評価する。[図 19](#figure-19) は、35B MoE モデルを scratch から学習した場合と、checkpoint から 176B MoE モデルの学習を継続した場合の loss curve を、BF16 精度と FP8 精度の両方について示している。MegaScale-MoE は BF16 形式と FP8 形式の双方で、安定した収束と一貫した training loss を保証する。

<span id="figure-19"></span>

![図 19。FP8 および BF16 における MegaScale-MoE の loss curve。](../../papers/megascale-moe/figure-19.png)

**図 19。** FP8 および BF16 における MegaScale-MoE の loss curve。

<span id="section-7"></span>

## 7 運用経験

本節では、MegaScale-MoE の導入と運用の経験について述べる。

**導入経験。** MegaScale-MoE は私たちの本番環境へ導入され、社内の大規模 MoE 学習タスクの大半を担っている。数兆パラメータのモデルを学習でき、単一の学習 job を 10,000 基を超える GPU まで拡張でき、個々の学習タスクは数か月にわたって実行される。前述の技術を組み合わせることで、MegaScale-MoE はモデル性能を損なわずに MoE 学習の通信 idle time を最小化し、メモリ使用量を最適化して、最終的に大規模 MoE 学習で数百万 GPU 時間を削減する。[図 20](#figure-20) は、各トークンに対して 20B パラメータを活性化する、200B パラメータの proprietary MoE モデルを学習した実際の本番 job におけるモデルの収束を示す。この job は 10,000 基を超える GPU を使用し、数か月間にわたって実行される。安定した学習処理のもとで loss は収束し続ける。

**FP8 学習。** FP8 学習の収束安定性を維持するため、広範な取り組みを行ってきた。たとえば、SwiGLU オペレータが数値範囲を大幅に広げることを確認した。これに対処するため、tensor 単位の quantization を、より高精度な token 単位の quantization（$1\times h$）へ置き換える。さらに、SwiGLU と gating weight の乗算によって動的な数値範囲がさらに広がるため、gating weight の乗算を FC2 出力の後へ戻し、quantization error を削減する。

学習の収束を保証するだけでなく、追加のエンジニアリング最適化も導入する。既存の FP8 学習実装 [Tra25, Lia24b] はモデルパラメータを BF16 で保存するため、GEMM 計算のために FP8 へ頻繁に変換する必要があり、cast と transpose のオーバーヘッドが加わる。これに対処するため、multi-precision optimizer を用いてモデルパラメータを直接 FP8 で保存する一方、main parameter は FP32 のまま、データ型ごとに別の buffer へ保持する。これによりメモリ消費量が減り、data parallelism におけるパラメータの all-gather 通信量が半減する。

**スケールアップ。** MoE モデルの学習では、計算負荷を増やさずにモデルパラメータを増加させることで、学習規模を無制限に拡張できるかという興味深いエンジニアリング上の問いが生じる。この手法は、モデルを大規模化すると追加のパラメータを収容するために TP degree を高める必要があるため、tensor parallelism では現実的でない。TP を増やすと GPU あたりの計算量は減るが、[式 1](#equation-01) と [式 4](#equation-04) に示すように通信オーバーヘッドは一定であり、通信時間が次第に長くなって学習効率が低下する。言い換えると、TP には本質的なスケーラビリティの限界があり、通信遅延を緩和するために高速なノード内 link へ依存することが多い。

<span id="figure-20"></span>

![図 20。10,000 基を超える GPU 上で数か月間にわたり実行された実際の本番 job の正規化学習 loss curve。数兆トークンを用い、20B の活性化パラメータと合計 200B パラメータを持つ MoE モデルを学習している。色の違いは学習の restart を表す。](../../papers/megascale-moe/figure-20.png)

**図 20。** 10,000 基を超える GPU 上で数か月間にわたり実行された実際の本番 job の正規化学習 loss curve。数兆トークンを用い、20B の活性化パラメータと合計 200B パラメータを持つ MoE モデルを学習している。色の違いは学習の restart を表す。

これに対し、SP と EP を用いて学習を拡張する場合、[式 2](#equation-02) と [式 3](#equation-03) に示すように、parallel size $n$ が増えるほど通信量は減少する。これは理論上、この並列化戦略が大幅に大きな規模まで拡張できることを意味する。しかし、実際の階層型インフラストラクチャでは、帯域幅が RDMA の水準まで低下する NVLink domain の外へ拡張した場合でも、この手法が学習効率を維持できるかという重大な課題が生じる。

形式的には、MoE 機構を組み込んだ SwiGLU 構造について、計算時間と通信時間の比率 $R$ を次のように定義する。

<span id="equation-05"></span>

$$
\mathrm{comm\_time}=\frac{2k\times bsh(n-1)/n/n}{\mathrm{bandwidth}},
$$

<span id="equation-06"></span>

$$
\mathrm{comp\_time}=\frac{3k\times bsh\times h_{\mathrm{ffn}}/n}{\mathrm{peak}}.
$$

<span id="equation-07"></span>

$$
R=\frac{\mathrm{comp\_time}}{\mathrm{comm\_time}}
$$

<span id="equation-08"></span>

$$
=3/2\times h_{\mathrm{ffn}}\times\frac{\mathrm{bandwidth}}{\mathrm{peak}}\times n/(n-1)
$$

<span id="equation-09"></span>

$$
\approx 3/2\times h_{\mathrm{ffn}}\times\frac{\mathrm{bandwidth}}{\mathrm{peak}}
$$

学習効率を維持するには FFN の計算時間が通信時間を上回り、通信オーバーヘッドが効果的にオーバーラップされなければならない。したがって、$R>1$ を保つことが目標となり、2 つの重要な知見が得られる。

- $R$ の値は expert 数、top-$k$、hidden dimension、parallelism size、input size のいずれにも依存しないため、algorithm parameter を柔軟に選択できる。
- $R$ は expert の intermediate dimension、computational peak、communication bandwidth のみによって決まる。したがって、固定されたハードウェア上では、expert dimension が十分に大きければ、エンジニアリングの観点から学習効率を維持しながら MoE モデルを拡張できる。

**包括的手法と自動化。** オペレータ間の通信・計算オーバーラップには、オペレータの実行順序、通信と計算の同時実行、通信用の SM 割り当ての決定など、多大なエンジニアリング作業を投入してきた。こうした手作業の介入によって学習動作をより深く理解し、的を絞った最適化が可能になる。学習が進み経験が蓄積されるにつれ、探索空間内のオペレータ scheduling を自動化し、学習処理を細粒度に最適化して最適な性能を得ることを目指している。自動最適化は今後の課題とする。

**MoE モデル学習と dense モデル学習。** MoE モデル学習の最適化を続ける中で、dense モデルの学習とは大きく異なる点がいくつか明らかになった。dense Transformer 層では、最適化の取り組みは self-attention と GEMM に集中する。前者は FlashAttention [Dao22] などの技術によって高速化されることが多く、後者は dense な計算であるため、通常は GPU の並列処理 unit で高い utilization を達成する。これに対して [図 13a](#figure-13) に示すように、attention と GroupedGEMM の合計 runtime は層の実行時間の約 3 分の 1 にすぎない。残りは通信とその他のオペレータによって消費される。MegaScale-MoE は通信オーバーヘッドに効果的に対処するが、dense モデルの対応するオペレータより本質的に複雑な MoE モデルの計算オペレータも、性能低下をもたらすことを確認している。具体的には、主に 3 つの理由から straggler の主因となる。

第一に、各 expert の intermediate dimension は dense モデルの FFN 層より小さい。複数 expert の計算を同時に効率よく処理するため、GroupedGEMM は多数の小さな行列乗算に単一の CUDA kernel を用いる。この kernel のリソース使用量（shared memory、L1 cache、thread 数を含む）は `cuFuncSetAttribute` によって細かく制御される。しかし、この細粒度の制御によって同期遅延が生じる場合がある。第二に、各 expert へルーティングされるトークン数が不均衡なため、GroupedGEMM の入力と出力は動的な shape の tensor になる。この tensor を頻繁に確保・解放することで、GPU memory fragmentation が悪化する。第三に、MoE gating 機構には routing score の計算や routing decision の通信などを行う多数の小さなオペレータが含まれる。CPU 性能の jitter によって、この kernel の launch が遅れ、launch latency が GPU 上での実際の実行時間を上回ることがあり、pipeline bubble が発生する。

<span id="section-8"></span>

## 8 関連研究

**大規模モデルの学習。** LLM 研究により、これらのモデルの膨大な計算需要へ対応する、スケーラブルで効率的かつ堅牢な学習技術 [Ras20, Sho19, Jia24f, Zha25ax] が開発されてきた。DeepSpeed [Ras20] は Zero Redundancy Optimizer（ZeRO）[Raj20, Raj21, Ren21] を備え、data parallelism に参加する GPU 間でモデルパラメータ、gradient、optimizer state を shard 化することで、メモリ消費量を管理可能な範囲に保ちながら LLM を拡張できる。Megatron-LM [Sho19] は層内 model parallelism 技術に重点を置き、各層のパラメータと計算を分割する。Pipeline parallelism は、連続する層の部分集合に属するパラメータと計算を各 GPU へ割り当て [Hua19, Nar19]、batch を micro-batch へ分割し、micro-batch を pipeline 方式で処理する。MegaScale [Jia24f] は、tensor、pipeline、data parallelism の組み合わせが、数十億パラメータの大規模モデルをこれまでにない規模で学習する効率的な戦略になり得ることを示している。

**Mixture-of-Expert の学習。** 高度な neural network の学習に伴う計算上の課題へ対処するため、機械学習分野では Mixture-of-Experts アーキテクチャの採用が進んでいる。その後、multi-GPU cluster 上で MoE の学習または推論を行うため、数多くの deep learning framework が提案された。DeepSpeed-MoE [Raj22] はモデルアーキテクチャの設計と圧縮技術によって学習コストを大幅に削減する。HetuMoE [Nie22] は階層型 all-to-all 通信戦略を利用して性能を高速化する。SE-MoE [She22] は、CPU memory や SSD などの異種リソースを用いたスケーラブルで効率的な学習へ重点を置く点で特徴的である。FasterMoE [He22] は dynamic shadowing、細粒度 scheduling、輻輳を回避する expert selection 戦略など、包括的な最適化群を導入する。Janus [Liu23r] は MoE モデルに data-centric な paradigm shift を提案し、通信需要の低減と学習効率の向上を目指す。Tutel [Hwa23] は adaptive parallelism と pipelining を採用し、MoE モデルに動的な解決策を提供する。しかし、その動的な parallelism switching と階層型 all-to-all は、数千億パラメータのモデルに大きなオーバーヘッドを生じさせる可能性がある。このようなオーバーヘッドを避けるため、最新の MoE 学習システム [Dee24d, Dee24a] は負荷分散に auxiliary loss または routing bias を用い、ノード間 token dispatch を制限する。MegaScale-MoE は各 MoE 層をノード内へ mapping することで、ノード間 token dispatch をなくす。

近年、DeepSeek-V3 [Dee24a] は本番規模の MoE モデルを学習するために 2 つの主要な最適化、すなわち高性能なノード間 all-to-all 通信を実現する DeepEP と、通信を計算とオーバーラップさせる DualPipe を導入した。ノード間 InfiniBand の帯域幅は比較的低いため、DeepEP はノード間通信量を一定に保つ目的で token dispatch を最大 4 ノードに制限し、routing の柔軟性を制約している。これに対し MegaScale-MoE は、任意の top-k expert へ効率よく routing できるよう各 MoE 層をノード内に配置する。DualPipe は異なる micro-batch 間で通信と計算をオーバーラップさせるために pipeline parallelism を利用し、モデルパラメータを $2\times$ 保存する必要がある。一方、MegaScale-MoE のオーバーラップは単一 micro-batch の forward pass または backward pass 内で行われるため、追加のメモリオーバーヘッドが発生せず、pipeline parallelism の有無を問わずシステムと互換性がある。

**Long-context 学習。** Megatron-LM [Sho19, Kor22] は特定の操作だけを sequence dimension に沿って分割するが、長い context を必要とするモデルの学習には、さまざまな sequence parallelism の手法 [Li24s, Liu23, Li23g, Gu24a] が検討されている。Blockwise Parallel Transformer [Liu24w] は、online softmax 計算に基づき、self-attention の block 単位の計算と FFN の融合を実装する。Ring Attention [Liu23, Li23g] は self-attention 計算と統合した ring 形式の通信機構を導入し、key chunk と value chunk の交換を可能にする。DeepSpeed Ulysses [Sam23] の all-to-all 形式の SP attention は通信量が少なく、計算パターンが均衡しているため、これを採用し、sequence length ではなく head によって attention を分割する。

**通信・計算オーバーラップ。** 複数のフレームワーク [Has19, Li20c, Mah23, Pen19, Zha23] は、単一の並列化戦略を用いる分散 deep learning 学習で、通信を計算とオーバーラップさせることに重点を置いている。一部の compiler 形式の研究 [Jan22, Wan22b, Pat24b] は kernel 間の細粒度オーバーラップを実現するが、GEMM kernel を過度に分割すると GPU utilization が低下する可能性がある。Centauri [Che24f] は通信の分割と階層型 scheduling により、3D parallelism を用いた LLM 学習の通信オーバーラップを強化する。Centauri と同様に、私たちのオペレータ間通信オーバーラップは、オペレータを並べ替えて独立した計算の中に通信を隠蔽する。さらに、GPU utilization を損なうことなく、オペレータ内オーバーラップによってクリティカルパス上の通信を隠蔽する。

<span id="section-9"></span>

## 9 結論

本稿では、MoE モデルを効率よく学習するために構築された本番品質のシステム MegaScale-MoE の設計、実装、導入を詳しく示した。MegaScale-MoE は、通信量の少ない並列化戦略、オペレータ間・オペレータ内の通信・計算オーバーラップ、通信パターンを調整した通信圧縮など、通信効率のよい手法を活用し、高性能 GPU の計算能力を引き出す。MegaScale-MoE は、1,440 基の NVIDIA Hopper GPU 上で 352B MoE モデルを学習した場合に 1.41M tokens/s の throughput を達成し、Megatron-LM より $1.88\times$ 改善する。大規模 MoE 学習の高速化に関する知見を共有することで、本研究が今後の研究を促すことを期待する。

## 謝辞

shepherd の Cheng Li と、貴重なフィードバックおよび提案を寄せてくださった匿名 reviewer に感謝する。本研究は一部、中国国家重点研究開発計画（Grant 2022YFB4500700）、若手教員科研創新能力支援プロジェクト（Grant ZYGXQNJSKYCXNLZCXM-I1）、北京大学中央高校基本科研業務費、中国国家自然科学基金（Grant 62172008 および Grant 62325201）の支援を受けた。Xin Jin と Xin Liu は corresponding author である。Chao Jin、Xuanzhe Liu、Xin Jin は、教育部高信頼ソフトウェア技術重点実験室（北京大学）にも所属する。

<span id="section-10"></span>

## 10 付録

<span id="section-10-1"></span>

### 10.1 パラメータ同期のための階層型通信

attention weight 全体の size を $P$、model parallelism（TP または SP）の dimension を $n$、data parallel size を $d$ とする。通常、model parallelism 用の GPU は同じノード上に配置されるためノード内通信が必要となる一方、data parallelism は複数ノードにまたがるためノード間通信が必要となる。同一のパラメータ partition をそれぞれ保持する $d$ 個のデバイスを含む data parallelism group を考える。

TP attention におけるパラメータ同期では、LLM 学習の主な 2 step で size $P/n$ のデータを $d$ 個のデバイス間で通信する。

- data size が $P/n$ で、$d$ 個のデバイス上で行われるノード間 `reduce-scatter` 操作。
- data size が $P/n$ で、$d$ 個のデバイス上で行われるノード間 `all-gather` 操作。

したがって主にノード間通信となり、通信量は $2P/n(d-1)/d$ である。

SP attention の場合、パラメータ同期には size $P$ のデータ全体が $n\times d$ 個のデバイス間で関わる。ノード内ネットワークとノード間ネットワークの帯域幅の差を考慮すると、この処理は、複製されたパラメータをまずノード内で reduction し、次にノード間で reduction した後、各デバイスへ再び分配する 4 step の階層型通信として実装できる。[図 5a](#figure-05) は $n=3$、$d=2$ の階層型通信の例を示す。詳細な手順は次のとおりである。

- data size が $P$ で、$n$ 個のデバイス上で行われるノード内 `reduce-scatter` 操作。
- data size が $P/n$ で、$d$ 個のデバイス上で行われるノード間 `reduce-scatter` 操作。
- data size が $P/n$ で、$d$ 個のデバイス上で行われるノード間 `all-gather` 操作。
- data size が $P$ で、$n$ 個のデバイス上で行われるノード内 `all-gather` 操作。

SP attention のノード間通信量は $2P/n(d-1)/d$ のままであり、ノード内通信量 $2P(n-1)/n$ が追加される。

さらに、ノード内通信とノード間通信では異なるリソースを用いるため、[図 5b](#figure-05) に示すように、これらの step を小さな chunk へ分割して pipeline 化し、相互に効率よく隠蔽できる。ノード間通信 latency とノード内通信 latency の比率は次のとおりである。

<span id="equation-10"></span>

$$
\frac{1}{n}\times\frac{\mathrm{intra\text{-}node\ bandwidth}}{\mathrm{inter\text{-}node\ bandwidth}}\times\frac{n(d-1)}{d(n-1)}
$$

NVLink 帯域幅が 450 GB/s、デバイス間 NIC 通信帯域幅が 50 GB/s である H100 SXM machine を用いる典型的な学習 scenario を考える。この状況では、ノード間通信の latency はノード内通信の latency を容易に上回り得る。これは、ノード内の通信がノード間の通信を覆い隠せることを意味する。したがって、このような scenario では、SP attention による gradient とパラメータの同期は、実際には TP attention と同等である。
