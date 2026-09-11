---
title: 'Rammer: Holistic DNN Compiler Scheduling'
createTime: 2026/09/11 13:04:22
permalink: /ja/papers/rammer/
---

> [Lingxiao Ma](https://xysmlx.github.io/) [+equal]、[Zhiqiang Xie](https://zhiqiangxie.com/) [+equal]、[Zhi Yang](https://yangzhihome.github.io/)、[Jilong Xue](https://dblp.org/pid/06/10336.html)、[Youshan Miao](https://youshan-miao.github.io/)、[Wei Cui](https://www.microsoft.com/en-us/research/people/weicu/)、[Wenxiang Hu](https://dblp.org/pid/141/4590.html)、[Fan Yang](https://fanyangcs.github.io/)、[Lintao Zhang](https://www.microsoft.com/en-us/research/video/lintao-zhang-showcases-tiger-bings-next-generation-index-serving-platform/)、[Lidong Zhou](https://www.microsoft.com/en-us/research/people/lidongz/)。第 14 回 USENIX Symposium on Operating Systems Design and Implementation (OSDI 20)、881-897 頁、2020 年 11 月 4-6 日。[会議ページ](https://www.usenix.org/conference/osdi20/presentation/ma)。<a href="/paper/rammer.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。本読解版は論文の実質的な本文、図、表、アルゴリズムを収録している。正確な印刷レイアウトと参考文献は原論文 PDF に従う。

[+equal]: 両著者の貢献は同等である。

## 概要

ハードウェア アクセラレータ上で Deep Neural Network (DNN) の計算を効率よく実行することは難しい。既存の DNN フレームワークやコンパイラは、データ フロー グラフ (DFG) 内の DNN 演算子を不透明なライブラリ関数として扱い、個別にアクセラレータへスケジュールして実行することが多い。演算子に含まれる並列性を活用するために、通常はハードウェアで実装された別層のスケジューラにも依存する。この 2 層方式は大きなスケジューリング オーバーヘッドを生み、利用可能なハードウェア資源を十分に活用できないことも多い。本論文では、大規模並列アクセラレータ上の DNN ワークロード実行を最適化する DNN コンパイラ設計 RAMMER を提案する。RAMMER はスケジューリング オーバーヘッドを最小化するため、コンパイル時に DNN の効率的な静的時空間スケジュールを生成する。演算子間と演算子内の並列性を包括的に協調スケジューリングし、ハードウェア利用率を最大化する。そのため RAMMER は、計算タスクとハードウェア アクセラレータに対して、ハードウェア中立で簡潔な新しい抽象化を複数導入する。これらの抽象化は RAMMER に豊かなスケジューリング空間を公開し、RAMMER は複数のヒューリスティクスで空間を探索して効率的なスケジュールを見つける。NVIDIA GPU、AMD GPU、Graphcore IPU など複数のハードウェア バックエンド向けに RAMMER を実装した。実験では、RAMMER は TensorFlow XLA や TVM などの最先端コンパイラを最大 20.1 倍上回った。NVIDIA がベンダー向けに最適化した独自 DNN 推論ライブラリ TensorRT も最大 3.1 倍上回った。

<span id="section-1"></span>

## 1 はじめに

Deep Neural Network (DNN) は、画像分類、自然言語処理、その他多くの AI タスクで広く採用されている。その重要性から、CPU、GPU、FPGA、DNN 専用アクセラレータなど多様な計算デバイスが DNN 計算に利用されてきた。これらのデバイスで効率よく DNN を計算することは、近年大きな注目を集めている研究課題である [Che18d, Guo19, Hol19, Liu19, Zha18c]。DNN 計算の効率を左右する重要な要因の 1 つがスケジューリング、すなわち計算の各部分を対象ハードウェア上で実行する順序の決定である。一般的なスケジューリングの重要性は広く知られ、十分に研究されている [Arp18, Leu04]。しかし、ハードウェア デバイス上の DNN 計算に特化したスケジューリングを扱う研究は少ない。

Deep Neural Network の計算パターンは通常、データ フロー グラフ (DFG) としてモデル化される。各ノードは行列乗算のような計算単位を表す演算子に対応し、辺は演算子間の依存関係を表す。この表現には 2 段階の並列性が自然に含まれる。第 1 は演算子間並列性で、DFG 上で依存関係を持たない演算子は並列に実行できる。第 2 は演算子内並列性で、行列乗算のような演算子が本質的に持つデータ並列性を利用し、GPU のような並列計算が可能なハードウェア アクセラレータを活用できる。

現在は、この 2 段階の並列性を利用するために 2 層スケジューリング方式が採用されている。演算子間の DFG 層スケジューラはデータ フロー グラフを受け取り、依存関係に基づいて実行可能になった演算子を発行する。演算子内スケジューラは演算子を受け取り、アクセラレータ内の並列実行ユニットへマッピングする。この階層設計は既存 DNN ツール群のシステム アーキテクチャに根本的な影響を与えている。たとえば DFG 層スケジューラは、通常 TensorFlow [Aba16] や ONNX Runtime [Onn18] などの Deep Learning フレームワークに実装される。一方、演算子層スケジューラは cuDNN [Cud23] や MKL-DNN [Mkl16] などの演算子ライブラリの背後に隠されることが多く、GPU の場合のようにハードウェアへ直接実装されることもある。

2 層スケジューリング方式は既存のフレームワークやアクセラレータで広く採用されているが、根本的な性能上の制約を持つ。この方式がうまく機能するのは、演算子の発行オーバーヘッドが演算子の実行時間に比べてほぼ無視でき、かつアクセラレータの全処理ユニットを飽和させるだけの演算子内並列性がある場合に限られる。実際には、そうでないことが多い。DNN アクセラレータの性能は CPU よりはるかに速く向上し続けているため、演算子発行のオーバーヘッドは一層目立つようになる。演算子内並列性が制限される小さなバッチ サイズの DNN 推論ワークロードでは、問題がさらに悪化する。また 2 層方式は上下層の微妙な相互作用を見落とす。全体性能を最適化するには、演算子内並列度を下げて演算子間並列度を高めることが有効な場合がある ([第 2 節](#section-2))。

これらの制約を緩和するため、DNN 計算に含まれる並列性を包括的に管理してスケジュールする Deep Learning コンパイラ RAMMER を提案する。RAMMER は rTask という新しい抽象化により、演算子間と演算子内のスケジューリングを統合する。rTask はスケジューラが演算子境界を越え、計算を細粒度にデバイスへ割り当てることを可能にする。スケジューリングをソフトウェアとハードウェアが別々に管理する 2 つの部分へ分割する既存設計とは異なり、RAMMER はソフトウェアのみの統一的な解である。底層ハードウェアへの依存が少ないため、多様な DNN アクセラレータへ適用できる。RAMMER では次の設計判断を行う。

第 1 に、ソフトウェア コンパイラから演算子内並列性を利用するため、RAMMER は DNN 演算子を rTask 演算子、すなわち rOperator として再定義する。rOperator は複数の独立した同種 rTask で構成される。各 rTask はアクセラレータの単一実行ユニット、たとえば GPU の Streaming Multiprocessor (SM) で動作する最小スケジューラブル単位である。これにより、細粒度の演算子内情報である rTask が RAMMER スケジューラへ公開される。RAMMER は DNN を rOperator ノードのデータ フロー グラフとして扱うため、粗粒度の演算子間 (DFG) 依存関係も把握できる。

しかし GPU など一部の現代的アクセラレータは、演算子内、すなわち rTask のスケジューリング インターフェースを公開しない。この課題に対処する第 2 の設計判断として、RAMMER はハードウェア アクセラレータを複数の仮想化実行ユニット (vEU) を含む仮想化並列デバイス (vDevice) として抽象化する。vDevice は、異なる演算子由来であっても複数の rTask を指定した vEU 上で所望の順序に実行できる。さらに vEU は、指定した rTask 群の完了を待つ barrier rTask を実行でき、依存演算子から来た rTask の正しい実行を保証する。vDevice は vEU をアクセラレータの物理実行ユニットへマッピングし、rTask の実計算を行う。

最後に、細粒度スケジューリングは、先に述べた演算子スケジューリングを上回るほど大きな実行時オーバーヘッドを生む可能性がある。そこで RAMMER はスケジューリング判断を実行時からコンパイル時へ移す。ほとんどの DNN の DFG はコンパイル時に利用でき、演算子は通常決定的な性能特性を示すため、コンパイル時プロファイリングで実行時性能を得られる [Siv19] という観察に基づく。不要な実行時オーバーヘッドを避けるだけでなく、より高コストなスケジューリング ポリシーを使って演算子間と演算子内の並列性をまとめて活用できる。

RAMMER は既存 DNN コンパイラで開発された最適化と互換性がある。TensorFlow などのフレームワークからデータ フロー グラフを取り込み、従来のグラフ オプティマイザが用いる技術 [Aba16] で最適化できる。rOperator も既存の kernel チューナ [Che18d] で最適化できる。既存最適化の上に RAMMER を適用すると、特に DNN 推論ワークロードで大きな追加性能向上が得られた。

RAMMER はハードウェア中立である。rTask、rOperator、vEU などの抽象化は、同種の実行ユニットを持つ任意の大規模並列計算デバイスに適用でき、DNN ワークロード向けに提案された計算デバイスのほぼすべてを含む。本論文では NVIDIA GPU 上の実装を詳しく説明し、さらに RAMMER を複数の代替計算デバイスへ再ターゲットした経験も述べる。

RAMMER を 52k 行の C++ コードで実装し、コードを公開した [+code]。6 つの DNN モデルを用いた評価では、NVIDIA と AMD の両 GPU 上で XLA や TVM などの最先端コンパイラを大幅に上回り、最大 20.1 倍高速化した。NVIDIA がベンダー向けに最適化した DNN 推論ライブラリ TensorRT [Ten17a] さえ最大 3.1 倍上回った。

[+code]: コードは <https://github.com/microsoft/nnfusion> で公開されている。

RAMMER で得た経験は、ベンダーが cuDNN や MKL-DNN のように高度に最適化した DNN 演算子実装をライブラリとして提供する現在の業界慣行が最適ではないことを強く示している。この慣行は DNN ワークロードに大きな効率上のコストをもたらす。現代的アクセラレータが利用可能なハードウェア並列性を増やし続ける一方、新しい DNN アーキテクチャは大きな演算子を多数の小さな演算子に置き換えて計算量を減らそうとしており [Xie16, Zop18]、今後状況はさらに悪化する。ハードウェア資源を十分に利用する包括的最適化を可能にするため、ベンダーには rOperator や vEU のような別形式で最適化済み実装を提供することを推奨する。

<span id="section-2"></span>

## 2 動機

本節では、既存 Deep Learning フレームワークの 2 層設計が持つ制約をいくつかの結果で示す。一般性を失うことなく、[第 5 節](#section-5) と同じ設定を用い、NVIDIA GPU 上で最先端 DNN フレームワーク TensorFlow [Aba16] を実験する。

**ハードウェア管理の演算子内スケジューリングは GPU 利用率を低下させる。** 2 層設計は演算子内スケジューリングを GPU などのアクセラレータにあるハードウェア スケジューラへ委ねる。[図 1](#figure-01) が示すように、この方式ではさまざまな DNN モデルで GPU 利用率が低くなりうる。バッチ サイズ 1 では Seq2Seq の GPU 利用率が 2% まで低下する。バッチ サイズを 16 に増やしても、6 モデルの平均 GPU 利用率は 40% にすぎない [+lstm-util]。スケジューリング効率を高めるため、現代的 GPU は独立した演算子を並行実行できるマルチストリーミング機構を備える。しかし[第 5 節](#section-5) の測定では、マルチストリーミングは全体性能を高めるどころか悪化させることが多い。

[+lstm-util]: バッチ サイズ 1 の LSTM の GPU 利用率がバッチ サイズ 16 よりわずかに高いのは、TensorFlow がバッチ サイズごとに異なる GEMM kernel 実装を用いるためである。

<span id="figure-01"></span>

![異なる DNN モデルとバッチ サイズにおける平均 GPU 利用率](../../papers/rammer/figure-01.png)

**図 1.** 異なる DNN モデルとバッチ サイズ (BS) における平均 GPU 利用率。利用率は kernel 実行のみを数え、演算子発行などの段階を除く。

**大きな演算子間スケジューリング オーバーヘッド。** 2 層方式は演算子間スケジューリングのオーバーヘッドも増やす。ここでは、GPU で実計算に使われない時間を演算子間スケジューリング オーバーヘッドとみなす。これには kernel 起動、コンテキスト初期化、ホストと GPU の通信など、演算子発行を支える各種操作が含まれる。[図 2](#figure-02) の各棒の上にある百分率は、DNN モデルが実際の GPU 計算に費やさなかった時間を示す。演算子間スケジューリングのオーバーヘッドが非常に大きいことがわかる。バッチ サイズ 1 では、6 モデルの平均オーバーヘッドは 55% である。バッチ サイズを 16 に増やすとやや改善するが、16% から 55% の範囲で依然として無視できない。TensorFlow のコンパイラを含む現代的 DNN コンパイラは、可能な場合に複数の DNN 演算子を 1 つへ統合する kernel fusion [Xla17, Che18d] を用いる。しかし[第 5 節](#section-5) の結果では、この技術でもオーバーヘッドは大幅に減らない。

<span id="figure-02"></span>

![異なる DNN モデルとバッチ サイズにおける平均 kernel 時間とエンドツーエンド実行時間](../../papers/rammer/figure-02.png)

**図 2.** 異なる DNN モデルとバッチ サイズ (BS) における平均 kernel 時間とエンドツーエンド実行時間。

**演算子間スケジューリングと演算子内スケジューリングの相互作用。** スケジューリングを 2 層へ分けると両者の微妙な相互作用が無視され、性能が準最適になる可能性がある。[図 3a](#figure-03) では、独立した 2 つの演算子が GPU にスケジュールされている。演算子 0 の性能を最大化するため、システムは並列度の高い最速実装を選びうる。すると演算子 0 はアクセラレータの全並列実行ユニット (EU)、ここでは GPU の Streaming Multiprocessor を貪欲に占有するが、各 EU は十分に利用されないこともある。全 EU を演算子 0 が占めるため、演算子 1 は資源が空くまで待つ。より良いスケジューラなら、演算子 0 の並列度を下げて演算子 1 を隣に配置し、演算子間並列度を高められる。[図 3b](#figure-03) にその例を示す。この問題は[第 3.3 節](#section-3-3) と[第 5 節](#section-5) で詳しく扱う。

<span id="figure-03"></span>

![非効率なスケジュールと最適化したスケジュール](../../papers/rammer/figure-03.png)

**図 3.** (a) 既存方式の非効率なスケジューリング、(b) 最適化されたスケジューリング計画。

**機会。** 上記の 2 層設計の根本的制約を踏まえると、演算子間と演算子内のスケジューリングをまとめて管理することが望ましい。しかし単純な実装では、すでに大きい演算子間スケジューリングをさらに上回るオーバーヘッドが生じうる。幸い、ほとんどの DNN の DFG はコンパイル時に利用でき、演算子は決定的な性能を示すことが多いため、コンパイル時プロファイリングで実行時間を得られる [Siv19]。[図 4](#figure-04) は ResNeXt [Xie16] の全演算子について GPU kernel 時間の平均と分散を示す。kernel 実行時間で重み付けした全演算子の標準偏差の平均はわずか 7% である。オフライン スケジュールを生成してスケジューリングを実行時からコンパイル時へ移し、実行時オーバーヘッドを減らせる。

<span id="figure-04"></span>

![ResNeXt の全演算子の kernel 時間プロファイル](../../papers/rammer/figure-04.png)

**図 4.** ResNeXt モデルの全演算子についてプロファイルした kernel 時間。各データ点を 1,000 回実行した。

<span id="section-3"></span>

## 3 RAMMER の設計

[第 2 節](#section-2) の観察に基づき、演算子間と演算子内のスケジューリングをまとめて管理する DNN コンパイラ フレームワーク RAMMER を設計した。[図 5](#figure-05) は既存 Deep Learning フレームワークと RAMMER の主な相違を示す。第 1 に、RAMMER の入力は従来の演算子ではなく rOperator をノードとするデータ フロー グラフである。rOperator は、アクセラレータの並列実行ユニット上で動作できる細粒度計算単位 rTask を明示的に公開する。rTask の詳細は[第 3.1 節](#section-3-1) で説明する。第 2 に、ソフトウェアとハードウェアに 2 層スケジューリングを分離せず、RAMMER は rTask 対応 DFG コンパイラを導入し、演算子間と演算子内のスケジューリングを 1 か所で管理する。このコンパイラは実行時のための静的実行計画を生成する。DNN 計算全体を 1 回のアクセラレータ呼び出しに収めるのは、非効率または不可能な場合が多い。そこで実行計画を複数の rProgram に分け、各 rProgram がハードウェアで行う計算の一部を含む。RAMMER はアクセラレータへ演算子を 1 つずつ発行せず、rProgram を 1 つずつ発行する。rTask 対応 DFG コンパイラの詳細は[第 3.3 節](#section-3-3) で述べる。実行計画を遂行するため、RAMMER はハードウェア アクセラレータを複数の仮想実行ユニット (vEU) を含む仮想化並列デバイス (vDevice) として抽象化する。vDevice は rTask 単位のスケジューリングと同期を提供し、コンパイル時に rProgram を対応する vEU へマッピングできる。実行時には vEU と vDevice がハードウェアへマッピングされる。仮想化デバイスは[第 3.2 節](#section-3-2) で説明する。

<span id="figure-05"></span>

![既存 DNN フレームワークと RAMMER のシステム概要](../../papers/rammer/figure-05.png)

**図 5.** (a) 既存 DNN フレームワークと (b) RAMMER の DNN 計算システム概要。RAMMER の DFG では、各ノードは rTask により演算子内並列性を明示する rOperator である。RAMMER は rOperator を動的に 1 つずつスケジュールせず、DFG を rTask で構成された静的実行計画 rProgram にコンパイルし、vDevice というソフトウェア デバイス抽象化でハードウェアへマッピングする。

<span id="section-3-1"></span>

### 3.1 rOperator

rOperator は独立した同種 rTask 群として定義される。rTask は RAMMER task の略で、アクセラレータ デバイスの処理要素で実行される演算子内の最小計算単位である。rTask は GPU の SIMD アーキテクチャなど、DNN アクセラレータの並列構造に自然に対応する。この種のアクセラレータで効率を最大化するには、計算を複数の並列な同種タスクへ分割する必要がある。各並列タスクを rTask として表すことで、演算子内並列性を底層ハードウェアだけでなく RAMMER コンパイラにも公開できる。rTask は論理的に並列タスクと同一なので、RAMMER は TVM [Che18d] など外部ツールを使って rOperator を rTask へ分割する。言い換えれば、適切な rTask 粒度の決定には外部ヒューリスティクスを用いる。

具体例として、行列乗算演算子を複数の同種 rTask に分割し、各 rTask が出力行列の 1 タイルを計算できる。タイル分割戦略は与えられているものとする。SeparableConv2D [Sep00] のように、複雑な DNN 演算子を独立した同種 rTask へ分割しにくい場合は、複数の依存 rOperator として表し、それぞれを rTask へ分割できる。

rTask は論理的な `rtask_id` で索引付けされ、rOperator 内で連続して番号が付く。rTask を実行する際、並列実行ユニットは `compute_rtask()` インターフェースを呼び出す ([図 6](#figure-06) 3 行目)。rProgram の生成には演算子内の rTask 総数が必要であり、`get_total_rtask_num()` で取得できる。対して従来の演算子には `compute()` インターフェースだけがある ([図 6](#figure-06) 1 行目)。rOperator の実装を rKernel と呼び、具体的な rTask 計算ロジックと rTask 総数を定める。1 つの rOperator は異なるタイル分割戦略に基づく複数の rKernel を持つことができ、たとえば資源効率と全体実行時間を交換する。

<span id="figure-06"></span>

![従来演算子と rOperator の実行インターフェース](../../papers/rammer/figure-06.png)

**図 6.** 従来演算子と rOperator の実行インターフェース。詳細は[第 4 節](#section-4)。

rOperator 抽象化により RAMMER は演算子間と演算子内の両並列性を公開でき、DNN 計算を包括的に最適化する新たな空間が開かれる。

<span id="section-3-2"></span>

### 3.2 仮想化並列デバイス

現代的アクセラレータは、rTask を所望の実行ユニットへ直接マッピングするインターフェースを提供しない。たとえば GPU は一度に 1 演算子を kernel として実行することだけを許す。この課題に対処するため、RAMMER はハードウェア アクセラレータをソフトウェア管理の仮想デバイス、仮想化並列デバイス (vDevice) として抽象化する。vDevice は複数の並列仮想実行ユニット (vEU) を提示し、それぞれが rTask を独立に実行できる。

RAMMER は vDevice を用い、rTask 対応 DFG の計算を vDevice 上の rProgram として編成する。rProgram は rTask の 2 次元配列 `prog[vEU_id][order]` で表される。`vEU_id` は rTask の割り当て先 vEU、`order` はその vEU 内での実行順序である。たとえば `prog[0][0]` は vEU 0 で最初に実行される rTask を示す。計画内の依存 rTask を正しく実行するため、RAMMER は barrier-rTask を導入する。barrier-rTask は `<vEU_id, order>` のペア リストを受け取り、各ペアで索引付けされた全 rTask の完了を待つ。この細粒度同期機構により rTask スケジュール計画を実行できる。

DNN 計算を実行するには、実行時に vDevice を物理アクセラレータへマッピングする必要がある。RAMMER が vDevice を各種ハードウェア アクセラレータへマッピングする方法は[第 4 節](#section-4) で述べる。

<span id="section-3-3"></span>

### 3.3 rTask 対応 DFG コンパイラ

rTask 抽象化と vDevice が公開する細粒度 rTask 実行機能により、大きな最適化空間が開かれる。RAMMER はこの空間で高品質なスケジュールを生成し、一連の rProgram として表す。そのため rTask 対応 DFG コンパイラは、スケジューリング機構とポリシーを分離する。機構側は、(1) ポリシーが実行計画を生成する 2 つのスケジューリング インターフェース、(2) ポリシーが要求するプロファイル情報を提供するプロファイラ、の 2 機能を提供する。

**スケジューリング インターフェース。** RAMMER の rTask 対応 DFG コンパイラは Append と Wait を導入する。`Append(task_uid, vEU_id)` は演算子の rTask を指定 vEU へ順番に割り当てる。`task_uid` は rTask のグローバル識別子で、本質的には演算子 id と演算子内の `rtask_id` の組み合わせである。第 2 の API `Wait(wtask_uid, list<task_uid>)` は、`wtask_uid` が示す rTask に `list<task_uid>` 内の rTask を待たせる。Wait は `wtask_uid` の直前に barrier-rTask ([第 3.2 節](#section-3-2)) を暗黙に Append する。同じ vEU へ順番に Append された連続 rTask $r_1, r_2, \ldots, r_n$ を待つ場合、最適化として待機リストには最後の $r_n$ だけを含めればよい。

**コンパイル時プロファイリング。** RAMMER のプロファイラは、1) vEU 上の個々の rTask 実行時間、2) ローカル メモリやレジスタなど rTask の資源使用量、3) rProgram の全体実行時間、の 3 種類を提供する。ポリシーはこの情報を使って効率的なスケジュールを生成できる。

**スケジューリング ポリシー。** アルゴリズム 1 は、上記インターフェースとプロファイラを使い、演算子間と演算子内の並列性を活用するポリシーを実装する方法を示す。このポリシーは rTask 対応 DFG を受け取り、演算子を wave 単位でスケジュールする [Lav06]。wave 内の演算子は DFG を幅優先探索したときの fringe node である。プロファイル結果 `time()` が総実行時間を短縮すると示す場合、その wave の演算子を現在の rProgram に含める。そうでなければ別の rProgram を作成する (2-10 行目)。

**アルゴリズム 1：Wavefront スケジューリング ポリシー**

- **データ：** $G$：rOperator の DFG、$D$：vDevice
- **結果：** `Plans`：rProgram
- **関数** `Schedule(G, D)`：
  - `P_curr = {}`
  - **各** `W = Wavefront(G)` **について**
    - `P1 = ScheduleWave(W, P_curr, D)`
    - `P2 = ScheduleWave(W, {}, D)`
    - **もし** `time(P1) <= time(P_curr) + time(P2)` **なら**
      - `P_curr = P1`
    - **それ以外は**
      - `Plans.push_back(P_curr)`
      - `P_curr = P2`
  - **返す** `Plans`
- **関数** `ScheduleWave(W, P, D)`：
  - `SelectRKernels(W, P)`
  - **各** `op in W` **について**
    - **各** `r in op.rTasks` **について**
      - `vEU = SelectvEU(op, P, D)`
      - `P.Wait(r, Predecessor(op).rTasks)`
      - `P.Append(r, vEU)`
  - **返す** `P`

まず、各 rOperator は rKernel と呼ぶ 1 つ以上の実装を持つと仮定する。各 rKernel は演算子を rTask へ分割する方法で、資源と実行時間のトレードオフが異なる。ある rOperator の rKernel 群のうち、実行時間が最短のものが最速で、実行時間と rTask 総数の積が最小のものが最も効率的である。

ポリシーは各 wave について `SelectRKernels()` で演算子実装を選ぶ (13 行目)。そのヒューリスティクスは次のとおりである。wave 内の全演算子に最速実装を使っても、全 rTask でアクセラレータの並列実行ユニットを占有できないなら、そのまま最速実装を選ぶ。そうでなければ最も効率的な rKernel を求めてプロファイルする。プロファイル結果がより短い実行時間を示せばそれらを選び、そうでなければ最速 rKernel を使う。このヒューリスティクスは個々にではなく、wave 内の rOperator と rTask をまとめて評価し、演算子間と演算子内のスケジューリングの相互作用を考慮する。rKernel 選択後、`SelectvEU()` で rTask の割り当て先 vEU を決める (16 行目)。現在の rProgram $P$ が与えられると、`SelectvEU()` は $P$ 内の各 rTask のプロファイル実行時間に基づき、その rTask を最も早く実行できる vEU を選ぶ。最後に `Wait()` で DFG に由来する rTask 単位の依存を保証し、`Append()` で選んだ vEU へ rTask を割り当てる (17-18 行目)。アルゴリズム 1 は RAMMER が機構とポリシーを分離する方法を示す。[第 5 節](#section-5) で示すように、この単純なポリシーでも最先端方式を、ときには大幅に上回る。提案した機構が今後さらに高度なポリシーの研究を可能にし、最適化空間をより深く探索できると期待する。

<span id="section-4"></span>

## 4 実装

RAMMER を 52k 行の C++ コードで実装し、そのうちコア コンパイラとスケジューリング機能は 3k 行である。入力は TensorFlow [Aba16] frozen graph、TorchScript [Dev18a]、ONNX [Onn18] のいずれかの DNN モデルである。まず入力モデルを rOperator の DFG へ変換する。入力モデルは未最適化であることが多いため、他のコンパイラと同様に定数畳み込み、共通部分式除去、パターンベースの kernel fusion など一般的なグラフ最適化も実装した。最適化 DFG の各 rOperator について、自動 kernel 生成器 [Che18d]、手動調整 kernel、他フレームワークの既存演算子から変換したものなど、複数のソースから 1 つ以上の rKernel 実装をロードする。RAMMER コンパイラは DFG をサブグラフへ分割し、たとえばアルゴリズム 1 のポリシーに従って各サブグラフを rProgram としてコンパイルする。出力時には各 rProgram をアクセラレータ上で動作する GPU kernel などのデバイス コードへ変換する。[図 7](#figure-07) に RAMMER の全体ワークフローを示す。

<span id="figure-07"></span>

![RAMMER の全体ワークフロー](../../papers/rammer/figure-07.png)

**図 7.** RAMMER の全体ワークフロー。

以下では CUDA GPU 向け RAMMER 実装の詳細を説明する。NVIDIA GPU と CUDA エコシステムは DNN アクセラレータとして最も広く使われているため、ここに焦点を当てる。vDevice 抽象化によって RAMMER コンパイラが異なるアクセラレータを統一インターフェースで支援できることを示すため、節末では AMD GPU と Graphcore IPU を含む他の DNN アクセラレータでの経験も簡潔に述べる。

<span id="section-4-1"></span>

### 4.1 NVIDIA CUDA GPU 上の RAMMER

NVIDIA GPU は通常、数十から数百の Streaming Multiprocessor (SM) を持ち、各 SM は数十のコアを含む。SM 上の計算は Single Instruction Multiple Thread (SIMT) モデルに従う。本論文では、NVIDIA が GPU プログラミングのために導入した CUDA [Nvi00a] の基本概念を読者が理解しているものとする。CUDA プログラムは CUDA kernel とも呼ばれ、複数のスレッドを block にまとめる。各 thread block は SM に割り当てられ、GPU ハードウェアがスケジュールする。RAMMER は各 vEU を SM へ自然にマッピングし、rTask を thread block として実装する。

<span id="section-4-1-1"></span>

#### 4.1.1 CUDA における rOperator

[図 8](#figure-08) は、$M \times K$ 行列 $A$ と $K \times N$ 行列 $B$ を乗算する単純な CUDA rOperator 実装を示す。簡単のため $M$ と $N$ は 32 で割り切れるものとする。各 rTask は出力行列 $C$ の $32 \times 32$ タイルを計算する。[図 8](#figure-08) 1-10 行目は 1 rTask 内の 1 スレッドの計算を示す。スレッドは RAMMER が割り当てた `rtask_id` で計算対象タイルを識別し (3-4 行目)、CUDA 組み込みスレッド索引 `threadIdx` でこのスレッドが計算するデータ要素を識別する (5-6 行目)。7-9 行目で要素を計算する。13 行目は rOperator が公開し、vEU の並列スレッドから呼ばれるインターフェースである。必要な rTask 総数は行列次元 $M$ と $N$ で決まり、`get_total_rtask_num` から取得できる (16 行目)。従来の CUDA コードとの重要な違いは、GPU ハードウェア スケジューラが制御する `blockIdx` ではなく、RAMMER が制御する論理索引 `rtask_id` を使うことである。適切な `rtask_id` で `compute_rtask()` を実行し、rTask を任意の vEU へマッピングできる。[図 8](#figure-08) のコードは説明用であり、[第 5 節](#section-5) の評価では、共有メモリやレジスタなど GPU メモリ階層を慎重に利用する、より複雑なタイル行列乗算 rOperator を用いる [Lai13, Nat10]。

<span id="figure-08"></span>

![rOperator 抽象化を用いた単純な行列乗算の CUDA 実装](../../papers/rammer/figure-08.png)

**図 8.** rOperator 抽象化を用いた単純な行列乗算の CUDA 実装。

<span id="section-4-1-2"></span>

#### 4.1.2 CUDA GPU 上の vDevice と vEU

CUDA GPU では、演算子内スケジューリングを通常 GPU 組み込みスケジューラが管理する。これを迂回するため、RAMMER は persistent thread-block (PTB) [Gup12] で vDevice 内の vEU を実装する。PTB は継続実行されるスレッド群を含む thread block で、RAMMER は PTB を任意の SM へ固定できる。rProgram が与えられると、PTB、したがって vEU の各スレッドは rProgram の順序に従って `compute_rtask()` を実行する。PTB で複数 rTask の `compute_rtask()` を連続実行するため、CUDA は `compute_rtask()` とその下位関数に `__device__` 修飾子を要求する ([図 8](#figure-08) 1、13 行目)。

[図 9](#figure-09) は 2 つの vEU、すなわち 2 つの PTB を持つ CUDA kernel 関数として vDevice を実装したコードである。Matmul、Relu、Conv の 3 rOperator を持つ DFG からコンパイルした rProgram を実行する。計画に従い、vEU 0 で Matmul の 2 rTask、並行して vEU 1 で Relu の 4 rTask を実行する。次に両 vEU へグローバル バリアを挿入し、それぞれが barrier-rTask を実行する。vEU 0 は vEU 1 の第 4 rTask、vEU 1 は vEU 0 の第 2 rTask を待つ。最後に Conv の 2 rTask を 2 vEU 上で実行する。各 vEU では rTask をコード分岐内で順番に実行し、現在の vEU Id が rProgram の生成した Id と一致するときだけ分岐が実行される。

<span id="figure-09"></span>

![2 つの vEU を持つ vDevice の CUDA コード](../../papers/rammer/figure-09.png)

**図 9.** 2 つの vEU を持つ vDevice の CUDA コード。

長い DNN 計算の前に、RAMMER は GPU スケジューラ [Wu15] を介して各 vEU、すなわち PTB を任意の SM へディスパッチする。利用率向上のため 1 SM で複数 vEU を並行実行できる。CUDA の SIMT モデルでは全 vEU は同種であり、1 SM が支援できる vEU 数は全 vEU 中で最も要求の大きい rTask、すなわちスレッド数、レジスタ数、共有メモリ量などが最大の rTask に依存する。実際には CUDA コンパイラ nvcc [Nvi00b] の最大 active PTB 数に従って、SM ごとの vEU 数を設定する。vDevice 抽象化により RAMMER の最適化はハードウェア非依存になる。

<span id="section-4-1-3"></span>

#### 4.1.3 CUDA の vEU 上で rTask を実行する

**異種 rTask の実行。** CUDA kernel では thread block のスレッド数が実行ライフサイクル全体で固定される。このため RAMMER は 1 vEU 上の全 rTask に同じ数の persistent thread での実行を要求する。実際には、異なる rOperator が並列度とスレッド当たり資源使用量のバランスを取るため、異なるスレッド数を用いることがある。RAMMER は vEU のスレッド数を、その vEU の rTask が使う最大スレッド数に設定する。より少ないスレッドを使う rTask には、余分なスレッドを不要かつ無効な実行から外す early-exit ロジックを挿入する。ただし early-exit は、スレッドがグローバル バリアを飛ばしてバリアが戻らなくなるデッドロックを起こしうる。RAMMER は CUDA cooperative group primitive [Nvi00d] で同期対象スレッドの範囲を明示的に制御し、この問題を避ける。

**barrier-rTask の実装。** 効率的な barrier-rTask のため、各要素が各 vEU の完了 rTask 数を追跡する整数 step 配列を導入する。rTask は完了時に最初のスレッドで対応要素を 1 増やす。$N$ 個の vEU 上の rTask 群を待つ barrier-rTask は最初の $N$ スレッドで対応要素をポーリングし、step が対象 rTask の order を上回るまで待つ。その後 `__syncthreads` を呼び出し、この vEU の全スレッドが次の rTask を実行できることを保証する。

<span id="section-4-1-4"></span>

#### 4.1.4 既存 CUDA 演算子の変換

DNN 演算子の多くはすでに CUDA kernel コードとして利用できる。開発負担を減らすため、RAMMER は既存 CUDA 演算子を rOperator へ変換する source-to-source 変換器を導入する。既存演算子も演算子内並列性を使うため thread block として実装されるが、`blockIdx` を使い、CUDA GPU ハードウェアに直接スケジューリングさせる。そこで `rtask_id` から所望の `blockIdx` を計算すれば、既存 kernel の計算ロジックを変えずに rOperator 化できる。

既存演算子の thread block は 1、2、3 次元形状で配置されうるが、vEU のスレッドは 1 次元形状であり、異なるスレッド形状の rTask を支援する必要がある。[図 10](#figure-10) は $[2 \times 2]$ と $[2 \times 3]$ の形状を持つ 2 rTask を 1 vEU で実行する例である。vEU は 1-D persistent thread 形状に固定し、スレッド索引を再マッピングして、vEU の 1-D `threadIdx` から既存 kernel の `threadIdx` を計算する。vEU のスレッド数は全 rTask の最大スレッド数なので、この再マッピングは常に可能である。[図 10](#figure-10) では vEU を $[1 \times 6]$ persistent thread とし、既存の $[2 \times 2]$ 形状を持つ rTask 0 を実行するとき、$[2 \times 2]$ を vEU の $[1 \times 6]$ スレッドへ再マッピングする。

<span id="figure-10"></span>

![1 つの vEU 上で 2 つの異種 rTask を実行する](../../papers/rammer/figure-10.png)

**図 10.** 1 つの vEU 上で 2 つの異種 rTask を実行する。

要約すると、既存 DNN 演算子を rOperator にするには、スレッドと block の索引を再マッピングし、early-exit ロジックを実装し、CUDA cooperative group primitive で active、すなわち early-exit していないスレッド上のローカル バリアを支援する。RAMMER は既存 kernel の入口にコンパイラ生成コードを挿入する。この変更で既存実装を保ったまま rTask 演算子として再利用できる。RAMMER では 70 rOperator に対して合計 150 rKernel を変換、実装した。

<span id="section-4-2"></span>

### 4.2 他のアクセラレータ上の RAMMER

RAMMER の設計は CUDA と NVIDIA GPU に限定されない。rTask、rOperator、vEU の抽象化は同種実行ユニットを持つ任意の大規模並列計算デバイスに適用でき、DNN 計算に使われるデバイスの大部分を含む。ここでは他デバイスを支援するための移植方法を述べる。

<span id="section-4-2-1"></span>

#### 4.2.1 AMD GPU 上の RAMMER

AMD GPU は NVIDIA GPU と同様に Compute Unit (CU) と呼ばれる多数の並列実行ユニットを持ち、CUDA に似た HIP プログラミング モデル [Amd00a] を備える。AMD の `hipify` は CUDA kernel を HIP kernel へ変換し、大部分の CUDA rOperator を HIP 版へ変換できる。アーキテクチャの細かな違いから、thread block ごとのスレッド数やローカル メモリ量など一部 CUDA kernel 設定は AMD GPU に最適でないため、性能向上のため 41 rKernel を再実装した。`hipify` は vDevice、すなわち PTB の CUDA 実装も HIP へ変換できる。唯一の例外は AMD GPU が cooperative group primitive を支援しない点である。rOperator に block 単位の同期回数 $S$、つまり `__syncthreads` の呼び出し数を返す API を追加し、early-exit スレッドには即座に終了する代わりに $S$ 回 `__syncthreads` を呼ばせる。

<span id="section-4-2-2"></span>

#### 4.2.2 Graphcore IPU 上の RAMMER

Graphcore IPU (Intelligence Processing Unit) [Gra20a] は GPU と大きく異なるアーキテクチャを持つ最先端 DNN アクセラレータである。IPU は Bulk-Synchronous-Parallel (BSP) 通信モデルを使う大規模並列 MIMD プロセッサで、1 基に tile と呼ばれる 1,216 個の並列処理ユニットを持つ。tile は hyper-threaded 計算コアと 256 KB のローカル メモリからなる。IPU 上の DNN 計算は明示的にデータ フロー グラフとしてプログラムされ、各 vertex が tile 上のコード、各 edge が vertex 間のデータ転送を表す。IPU コンパイラが vertex を tile へマッピングする。

RAMMER の rTask は IPU の MIMD モデルにも対応し、vEU を tile、vertex を rTask として扱える。したがって IPU の rOperator は vertex の集合として実装できる。特に IPU コンパイラはコンパイル時に vertex-to-tile マッピングを制御でき、vDevice 抽象化の中核機能を提供する。一方、ハードウェアの BSP モデルにより細粒度同期機構はなく、barrier-rTask をグローバル バリアで実装するため RAMMER のスケジューリング空間が狭まる可能性がある。それでも同じ計算ステップで異なる演算子の rTask をスケジュールし、利用率を高められる。評価用に合計 15 rOperator と 18 rKernel を実装した。

<span id="section-4-2-3"></span>

#### 4.2.3 x86 CPU 上の RAMMER

マルチコア x86 CPU にも RAMMER を実装したが、x86 プラットフォームでは抽象化による性能上の利点がほとんどなかった。数値計算における x86 コアの性能は相対的に低いため演算子実行時間が長く、コア数が少ないためほぼすべての DNN 演算子が全コアを占有できる。kernel 起動も通常の関数呼び出しにすぎず、スケジューリング オーバーヘッドは大きくない。したがって RAMMER は従来の 2 層方式に追加の利点を提供できない。

<span id="section-5"></span>

## 5 評価

本節では RAMMER を他の最先端フレームワークと比較し、その有効性を示す詳細な評価結果を述べる。

<span id="section-5-1"></span>

### 5.1 実験設定

**マシン環境。** 異なるアクセラレータを搭載した 3 台のサーバで RAMMER を評価した。CUDA GPU 評価には Intel Xeon E5-2690v4 CPU と NVIDIA Tesla V100 (16GB) GPU 4 基を備えた Azure NC24s_v3 VM を使い、Ubuntu 16.04、CUDA 10.0、cuDNN 7.6.5 を用いた。AMD ROCm GPU 評価には Intel Xeon E5-2640 v4 CPU と AMD Radeon Instinct MI50 (16GB) GPU 2 基を備えたサーバを使い、Ubuntu 18.04 と ROCm 3.1.1 [Amd00] を導入した。IPU 評価には Intel Xeon Platinum 8168 CPU と IPU 16 基を備えた Azure ND40s_v3 preview VM と Poplar-sdk 1.0 を用いた。

比較対象は、最先端 DNN フレームワークを代表する TensorFlow v1.15.2、最先端 DNN コンパイラを代表する TVM v0.7 [Che18d] と TensorFlow-XLA、NVIDIA GPU 向けベンダー固有推論ライブラリ TensorRT v7.0 の TensorFlow 統合版である。

**ベンチマークとデータセット。** CNN と RNN の典型的アーキテクチャ、および画像、NLP、音声の応用領域を網羅する代表的 DNN モデルを用いる。ResNeXt [Xie16] は ResNet [He16] の改良版、NASNet [Zop18] は Neural Architecture Search で得られた最先端 CNN、AlexNet [Kri12] は単純な構造を持つ古典的 CNN である。LSTM-TC [Hoc97] はテキスト分類用 RNN、DeepSpeech2 [Amo16] は代表的音声認識モデル、Seq2Seq [Sut14] は Neural Machine Translation 用モデルである。各モデルで使う rKernel を含む全実装は artifact evaluation リポジトリで公開している [+artifact]。

[+artifact]: Artifact は <https://github.com/microsoft/nnfusion/tree/osdi20_artifact/artifacts> で公開されている。

評価はモデル推論を中心とする。RAMMER は原理的に学習を制限しないが、学習支援にはより多くの演算子開発が必要である。CIFAR-10 [Kri00]、ImageNet [Den09]、LibriSpeech [Pan15]、合成データセット上で評価した。[表 1](#table-01) はモデル、ハイパーパラメータ、対応データセットを示す。全性能値は 1,000 回の平均で、いずれも変動は非常に小さかった。

<span id="table-01"></span>

![Deep Learning モデルとデータセット](../../papers/rammer/table-01.png)

**表 1.** Deep Learning モデルとデータセット。

<span id="section-5-2"></span>

### 5.2 CUDA GPU 上の評価

本節は次の問いに答える。1) 最先端 DNN フレームワークやコンパイラと比べた RAMMER の性能、2) GPU 並列資源の利用率、3) 実行時スケジューリング オーバーヘッドの削減量、4) 演算子内と演算子間の並列性を活用するスケジューリングによる性能向上、5) 全体性能に対する細粒度同期の効果。

<span id="section-5-2-1"></span>

#### 5.2.1 エンドツーエンド性能

まず TensorFlow (TF)、TensorFlow-XLA (TF-XLA)、TVM、TensorRT (TF-TRT) と比較する。抽象化の効果を示すため、既存コンパイラと同様の最適化だけを実装し、2 層方式を残したベースライン RAMMERBASE を作成した。同じコードベースで実装した通常の DNN コンパイラとみなせる。[図 11](#figure-11) にバッチ サイズ 1 の実行時間を示す。

<span id="figure-11"></span>

![NVIDIA V100 GPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間](../../papers/rammer/figure-11.png)

**図 11.** NVIDIA V100 GPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間。

RAMMER は TF を平均 14.29 倍、LSTM-TC で最大 33.94 倍上回る。TF が DFG レベルの重い実行時スケジューリング オーバーヘッドを受けるためで、小バッチ推論のように個々の演算子実行時間が短い場合に顕著である。DNN コンパイラである TF-XLA は演算子融合などの DFG 最適化と、カスタム kernel 生成などの演算子コード特殊化で TF を改善するが、オーバーヘッドを完全には避けられず、RAMMER との性能差は平均 11.25 倍、最大 20.12 倍である。ResNeXt や NASNet では TF よりオーバーヘッドが大きい場合もある。TVM は kernel tuning で演算子ごとの特殊化 kernel を生成する。1,000 ステップ調整して各演算子の最速 kernel を選ぶため TF と TF-XLA を大幅に改善するが、RAMMER はなお平均 3.48 倍、最大 6.46 倍上回る。TVM は個別演算子を高速化できても RAMMER のような細粒度並列性を利用できない。例外は AlexNet で、単純な逐次構造と少数の大きな演算子を持つ初期の現代的 DNN なので最適化しやすく、RAMMER と TVM は同等である。TensorRT は NVIDIA の高度に最適化された演算子を持つ DNN 推論ライブラリである。stand-alone 版がベンチマークを直接コンパイルできないため公式 TF 統合版を使ったが、DeepSpeech2、LSTM-TC、Seq2Seq-NMT は 50 時間以上コンパイルしても結果を出せなかった。そこで TensorRT native API で 3 モデルを再実装した。RAMMER は全ベンチマークで TensorRT を上回り、レイテンシを平均 2.18 倍、最大 3.09 倍短縮する。RAMMERBASE と比べてもエンドツーエンド性能を平均 2.59 倍、最大 6.29 倍改善する。

**異なるバッチ サイズ。** ResNeXt と LSTM-TC をバッチ サイズ 4、16 で評価した ([図 12](#figure-12))。効率的なオープンソース kernel を探すか、手動または自動で調整する必要があり、最適化 rOperator の開発コストが高いため 2 モデルに限定した。大きなバッチでは演算子実行時間が増え、既存フレームワークのオーバーヘッドが相対的に下がる。それでも RAMMER はバッチ サイズ 16 の ResNeXt における TensorRT を除く全システムを上回る。この場合、TensorRT はソース非公開の演算子を使い、RAMMER の実装はその性能にまだ届かない。閉源 kernel に匹敵する演算子実装は RAMMER の主要な課題である。バッチ サイズ 16 の ResNeXt では TF の 2.25 倍、TVM の 1.25 倍、LSTM-TC ではそれぞれ 20.08 倍と 9.0 倍である。

<span id="figure-12"></span>

![異なるバッチ サイズにおけるエンドツーエンド モデル推論時間](../../papers/rammer/figure-12.png)

**図 12.** 異なるバッチ サイズ (BS) におけるエンドツーエンド モデル推論時間。

**大きな入力サイズ。** 既定設定では ResNeXt と NASNet を CIFAR-10 の $32 \times 32$ 画像で評価する。より大きな画像での性能を見るため、原論文 [Xie16, Zop18] と同じハイパーパラメータで ImageNet を用いた。ResNeXt は 101 層、cardinality 64、bottleneck width 4d、NASNet は repeated cell 4、filter 1056 である。[図 13](#figure-13) の結果では、入力サイズを増やしても RAMMER の利得はほとんど変わらない。ImageNet の ResNeXt では TF の 18.91 倍、TVM の 4.96 倍、TF-TRT の 2.06 倍、NASNet ではそれぞれ 6.99 倍、1.33 倍、2.34 倍である。大規模データセット向けモデルには RAMMER が活用できる演算子間並列性が多いことが主因である。たとえば CIFAR-10 から ImageNet へ移ると ResNeXt の cardinality は 16 から 64 になる。

<span id="figure-13"></span>

![ImageNet 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間](../../papers/rammer/figure-13.png)

**図 13.** ImageNet データセット上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間 (画像サイズ：$224 \times 224$)。

以上では RAMMERBASE がすでに TF-XLA や TVM と同等以上の性能を持つ。以後は RAMMERBASE を最先端コンパイラのベースライン、TF-TRT を最先端 DNN 推論ライブラリとして RAMMER の利得を評価する。RAMMERBASE は実装差が性能比較に与える副作用も除ける。

<span id="section-5-2-2"></span>

#### 5.2.2 GPU 利用率

RAMMER は異なる演算子の rTask を並行実行して GPU 利用率を高める。TF、TF-TRT、RAMMERBASE と比較した。[図 14](#figure-14) はバッチ サイズ 1 の 6 モデルについて、実行時間全体の平均利用率を示す。kernel 実行だけを数え、演算子発行などを除く。NVIDIA profiler nvprof [Nvi00c] の SM-efficiency、すなわち少なくとも 1 warp が multiprocessor 上で active な時間の割合を使う。RAMMER は TF と TF-TRT に対して平均 4.32 倍と 2.45 倍利用率を高める。実行時オーバーヘッド低減と演算子協調スケジューリングの両方による。同じ kernel を使う RAMMERBASE と比べても、スケジューリングだけで平均 1.61 倍、LSTM-TC で最大 2.39 倍改善する。

<span id="figure-14"></span>

![GPU 利用率の比較](../../papers/rammer/figure-14.png)

**図 14.** GPU 利用率の比較。

[第 2 節](#section-2) のとおり、現代的 GPU は独立 kernel を並行スケジュールする multi-streaming を支援する。TF の stream 数を増やして評価した。[図 15](#figure-15) は各モデルで stream を 1、2、4 としたときの全実行時間と kernel 時間を示す。stream の増加は、他の研究 [Siv19] と同様、エンドツーエンド性能を悪化させる。4 stream は 1 stream より平均 2.72 倍遅い。kernel 時間もほとんど減らず、多数の kernel が逐次実行されたままで利用率向上が小さい。主因は[図 15](#figure-15) のように演算子スケジューリング オーバーヘッドが増えることである。

<span id="figure-15"></span>

![異なる stream 数における TensorFlow の性能](../../papers/rammer/figure-15.png)

**図 15.** 異なる stream (STM) 数における TF の性能。棒上の数字は kernel 時間を示す。

<span id="section-5-2-3"></span>

#### 5.2.3 スケジューリング オーバーヘッド

RAMMER はスケジューリング オーバーヘッドを効果的に減らす。TF、TF-TRT、RAMMERBASE と実行時オーバーヘッドを比較した。[図 16](#figure-16) は各モデルの総 kernel 時間と、実計算以外のスケジューリング時間を示す。TF に対して RAMMERBASE は全モデルの平均スケジューリング時間を 32.29 ms から 2.27 ms、割合を 55.41% から 18.43% へ減らす。TF-TRT と比べても 31.38% から 18.43% へ減る。実行コード パスの最適化と演算子融合による kernel 起動削減の効果であり、既存 DNN フレームワークの演算子スケジューリングが重いことを示す。RAMMER は RAMMERBASE の平均 2.27 ms をさらに 0.37 ms へ、6.14 倍削減する。静的なコンパイル時演算子スケジューリングにより、複数演算子を rProgram にまとめ、1 回の GPU kernel 起動で実行できるためである。

<span id="figure-16"></span>

![各モデルの GPU スケジューリング オーバーヘッド](../../papers/rammer/figure-16.png)

**図 16.** 各モデルの GPU スケジューリング オーバーヘッド。棒上の数字は百分率。TF：TensorFlow、TRT：TF-TRT、RB：RAMMERBASE、R：RAMMER。

<span id="section-5-2-4"></span>

#### 5.2.4 演算子内と演算子間スケジューリングの相互作用

RAMMER は個々の演算子だけを高速化せず、両スケジューリングの相互作用を最適化できる。[第 3.3 節](#section-3-3) のように各 rOperator に適切な rKernel を選ぶ。個別演算子の最速 kernel だけからなる集合と、RAMMER が選ぶ集合を比較した。[図 17](#figure-17) は ResNeXt と LSTM-TC で RAMMER と RAMMERBASE を示す。どちらの集合でも RAMMER は大幅に改善する。同じ最速 kernel を使う RAMMER-fast は RAMMERBASE-fast より平均 2.89 倍高速である。選択肢が増えた RAMMER-select は、単独では最速でない kernel を選ぶ場合でも RAMMER-fast より平均 1.44 倍、最大 2.28 倍改善する。同じ kernel を RAMMERBASE-select で使うと逆に平均 1.84 倍低下する。

<span id="figure-17"></span>

![異なる kernel 集合とバッチ サイズにおける性能](../../papers/rammer/figure-17.png)

**図 17.** 異なる kernel 集合とバッチ サイズ (BS) における性能。

バッチ サイズ 4 の LSTM-TC を詳しく分析した。Matmul の最速 kernel は 1,024 rTask で 4.28 マイクロ秒だが、RAMMER が選ぶ kernel は 16 rTask で、単独実行では 7.46 マイクロ秒と遅い。それでも、演算子内並列度を下げて個別 kernel を遅くし、演算子間並列度を上げて全体性能を改善するため後者を選ぶ。RAMMER の包括的スケジューリング能力による効果である。

<span id="section-5-2-5"></span>

#### 5.2.5 細粒度同期

barrier-rTask は、不規則構造を持つ DFG に追加の最適化空間を与える。この構造は Neural Architecture Search (NAS) のモデルで一般的である [Zop18]。NASBench [Yin19b] で最大 9 演算子、7 辺からなる 5,000 モジュールを無作為生成した。RAMMER は RAMMERBASE より平均 1.28 倍、最大 3.40 倍高速である。28.3% はアルゴリズム 1 が同一 wave に異種演算子を置くなど明らかな不規則構造を持つ。これらでは barrier-rTask はグローバル バリアより平均 1.11 倍、最大 1.89 倍高速である。[図 18](#figure-18) は各演算子の時間と rTask 数を含む例である。wave 間のグローバル バリアを除き、細粒度 rTask 同期を入れることで、wave 1 と wave 2 の STEM-Conv のように異なる wave の演算子実行を重ねられる。

<span id="figure-18"></span>

![NASBench が生成した不規則 DFG](../../papers/rammer/figure-18.png)

**図 18.** NASBench が生成した不規則 DFG。

<span id="section-5-3"></span>

### 5.3 他のアクセラレータ上の評価

<span id="section-5-3-1"></span>

#### 5.3.1 ROCm GPU 上のエンドツーエンド性能

AMD ROCm GPU 上で TF、TVM、RAMMERBASE と比較した。実験で有効化できなかった TF-XLA と NVIDIA 専用の TensorRT は除いた。[図 19](#figure-19) はバッチ サイズ 1 の 6 ベンチマークを示す。RAMMER は TF より平均 13.95 倍、LSTM-TC で最大 41.14 倍、TVM より平均 5.36 倍、最大 7.57 倍高速である。TVM auto tuning は ROCm で動作させられず、既定 kernel を用いた。RAMMERBASE に対してもスケジューリングにより平均 2.19 倍、最大 4.12 倍高速化する。図の RAMMERBASEK は RAMMERBASE と同一だが RAMMER の kernel を使う。RAMMER は単独最速の rKernel を常に選ぶわけではない。多くのモデルでは差が小さいが、ResNeXt では 3.02 倍低下し、スケジューリングと kernel 選択の相互作用の重要性がわかる。

<span id="figure-19"></span>

![AMD MI50 GPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間](../../papers/rammer/figure-19.png)

**図 19.** AMD MI50 GPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間。

<span id="section-5-3-2"></span>

#### 5.3.2 Graphcore IPU 上のエンドツーエンド性能

Graphcore IPU でも予備評価した。効率的な rOperator の実装に手間がかかるため 3 RNN に限定した。現在は 1 IPU のみを支援し、multi-IPU は今後の課題である。tile 当たり 256 KB のメモリ制限から各モデルを 4 層にして 1 IPU に収めた。[図 20](#figure-20) はバッチ サイズ 1 の性能で、予備実装でも RAMMERBASE より最大 5.37 倍高速になり、新しいアクセラレータ アーキテクチャに対する抽象化の適用可能性と有効性を示す。

<span id="figure-20"></span>

![Graphcore IPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間](../../papers/rammer/figure-20.png)

**図 20.** Graphcore IPU 上のバッチ サイズ 1 におけるエンドツーエンド モデル推論時間。

<span id="section-6"></span>

## 6 議論

利点を示した上で、RAMMER の制約と今後の課題を論じる。

**大きなバッチ サイズでの性能利得。** 演算子内並列性がハードウェアを飽和させられないとき RAMMER の利得は大きい。オンライン推論の小バッチが該当する。予備実験では、LSTM-TC のようなバッチ サイズ 256 の学習にも当てはまる。[図 21](#figure-21) に示すように、演算子内と演算子間の包括的最適化で RAMMER は RAMMERBASE より 2.28 倍、TF-XLA より 2.36 倍高速である。大バッチ学習の詳細分析と追加最適化は今後の課題とする。

<span id="figure-21"></span>

![NVIDIA V100 GPU 上のバッチ サイズ 256 における LSTM-TC のエンドツーエンド学習時間](../../papers/rammer/figure-21.png)

**図 21.** NVIDIA V100 GPU 上のバッチ サイズ 256 における LSTM-TC のエンドツーエンド モデル学習時間。TVM と TF-TRT は学習を支援しないためデータがない。

**動的グラフ。** 現在 RAMMER は静的グラフだけを支援する。動的制御フローを持つ DFG [Yu18c] では、条件分岐やループ本体など各静的サブグラフを個別 rProgram へコンパイルできる。実装は今後の課題である。

**ジョブ間スケジューリング。** RAMMER は単一 Deep Learning ジョブの最適化に焦点を当て、複数モデルのバッチ化や vDevice による各ジョブ資源の厳密な制御などジョブ間スケジューリングと直交する。同一アクセラレータで、異なる演算子だけでなく異なるジョブ由来の rTask を協調スケジュールする可能性は興味深い課題である。

<span id="section-7"></span>

## 7 関連研究

DNN コンパイラ最適化は 2 層表現に基づき 2 種類に分けられる。TensorFlow [Aba16]、PyTorch [Pyt17]、TVM [Che18d]、XLA [Xla17] などは演算子融合など DFG レベル最適化を使う。TASO [Jia19b] は自動グラフ置換を提案する。演算子レベルでは AutoTVM [Che18a]、Tensor Comprehensions [Vas18]、FlexTensor [Zhe20a]、Tiramisu [Bag19]、Halide [Rag13] などがハードウェア固有の効率的コードを調整、生成する。RAMMER は最適化済み DFG を入力し、これらの生成器で効率的 rKernel を作るため互換性がある。

DNN 推論最適化も注目されている。DeepCPU [Zha18c]、BatchMaker [Gao18]、GRNN [Hol19]、NeoCPU [Liu19] は CPU または GPU 上で特定 RNN/CNN を最適化する。Jain ら [Jai18a] は複数推論ジョブを時間的、空間的に多重化して GPU 利用率を高める。RAMMER は一般的モデルとアクセラレータに適用でき、コンパイラ最適化だけでなく新たな抽象化と広い空間を提供する点が異なる。Astra [Siv19] は DNN の予測可能性で学習をオンライン最適化し、RAMMER は同じ性質で個々の rTask スケジューリング オーバーヘッドを減らす。Nexus [She19]、PRETZEL [Lee18b]、Clipper [Cra17]、TF-serving [Ols17] は保証レイテンシ下の全スループットを最適化するが、RAMMER は単一モデルを最適化し、これらと直交する。

GPU 上のソフトウェア スケジューラで汎用ワークロードを扱う研究もある。Juggler [Bel18] はタスク DAG のジョブを動的実行し、Wu ら [Wu15] は SM 上のジョブ局所性を制御する。RAMMER は DNN の性質から rTask と rOperator という新しい表現を導入し、コンパイル時スケジューリングで実行時オーバーヘッドを体系的に避ける。

<span id="section-8"></span>

## 8 結論

既存 Deep Learning フレームワークは、演算子間スケジューリングをフレームワークで管理し、演算子内スケジューリングをハードウェアへ委譲する 2 層設計の根本的制約により不要なオーバーヘッドを受ける。RAMMER は包括的なコンパイラで、(1) 細粒度の演算子内並列性を公開する rTask 演算子抽象化、(2) 現代的アクセラレータを並列実行ユニットとして仮想化し細粒度スケジューリング能力を公開する仕組み、(3) DNN 計算の予測可能性を利用し、実行時スケジューリングをコンパイル時 rTask 実行計画の生成へ変換する仕組みを提供する。評価では、標準的 Deep Learning フレームワーク、コンパイル フレームワーク、GPU ベンダー固有推論エンジンに対して大幅に改善した。RAMMER は既存 DNN コンパイラ基盤のエコシステムを強化する新しい手法である。

## 謝辞

匿名査読者と shepherd の Jinyang Li 教授による多くの提案に感謝する。GPU を支援した Microsoft Grand Central Resources チームの Jim Jernigan と Kendall Martin に感謝する。Fan Yang は論文執筆中いつも寄り添った、今は亡き愛猫 Pearl に感謝する。本研究の一部は中国国家自然科学基金 Grant No. 61972004 の支援を受けた。
