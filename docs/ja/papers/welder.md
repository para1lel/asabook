---
title: 'Welder: Scheduling DNN Memory Access'
createTime: 2026/09/11 12:00:00
permalink: /ja/papers/welder/
pageClass: paper-reading
---

> [Yining Shi](https://dblp.org/pid/161/3927-1.html) [+internship]、[Zhi Yang](https://yangzhihome.github.io/)、[Jilong Xue](https://dblp.org/pid/06/10336.html)、[Lingxiao Ma](https://xysmlx.github.io/)、[Yuqing Xia](https://dblp.org/pid/211/8365.html)、[Ziming Miao](https://dblp.org/pid/216/9568.html)、[Yuxiao Guo](https://dblp.org/pid/22/329-1.html)、[Fan Yang](https://dblp.org/pid/29/3081-24.html)、[Lidong Zhou](https://www.microsoft.com/en-us/research/people/lidongz/)。2023 年 7 月、*17th USENIX Symposium on Operating Systems Design and Implementation（OSDI 23）*、pp. 701–718 に掲載。[Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/conference/osdi23/presentation/shi)。<a href="/paper/welder.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。arXiv 登録および TeX ソースは公開されていないため、正確な文言、印刷レイアウト、参考文献については出版 PDF を正本とする。

## 概要

より高忠実度なデータを処理する需要が増し、新しいハードウェアアクセラレータで高速な計算コアが使われるようになるにつれ、現代の深層ニューラルネットワーク（DNN）はますますメモリ集約的になっている。多くの一般的な DNN モデルで、計算コアの利用率が低い一方、メモリ帯域幅が飽和するという隔たりが観測されている。この非効率は、DNN を計算集約型ワークロードとして扱う従来の見方と、DNN モデルに包括的なメモリアクセス最適化がないことの両方に起因する。

本稿では、メモリアクセスを包括的に捉えて実行効率を最適化する深層学習コンパイラ WELDER を提案する。WELDER の中核は、tile レベルの細粒度なデータ管理を可能にする抽象化 tile-graph である。メモリ層間では最適化が独立するという観察を利用し、WELDER は組合せ的な DNN 最適化空間全体を複数の独立した空間へ分解し、tile トラフィックに基づくコストモデルによって演算子内と演算子間のデータ再利用を効果的に調整する。これにより、従来の個別的なメモリ最適化を単一空間に統合し、89 種類多い最適化パターンを含む効率的な実行計画を生成して、最先端手法を大幅に上回る。さらに WELDER は、既存のアクセラレータメモリとホストメモリを一つのシステムとして組み合わせ、任意に大きな入力を持つ DNN モデルも処理できる。

<span id="section-1"></span>

## 1 はじめに

深層ニューラルネットワーク（DNN）は、視覚や言語の分析・生成をはじめ、幅広いタスクに利用されている。従来、DNN は計算集約型ワークロードとみなされてきた。DNN モデルは通常データフローグラフ（DFG）として定義され、各ノードが行列乗算などの計算集約型演算子を表す。計算を高速化するため、これらの演算子は GPU や TPU [Jou21] のように多数の並列計算コアを備えた現代のアクセラレータへオフロードされる。アクセラレータを効率よく使うため、DNN フレームワークとコンパイラはコード特殊化 [Che18e, Zhe20, Zhu22] や演算子融合 [Che18e, Ma20a] など、さまざまな最適化技術を検討している。

計算中心の最適化は古典的な DNN モデルに有効だが、現代の DNN はますますメモリ集約的になっている。さまざまな最先端 DNN モデルをプロファイルすると、エンドツーエンド DNN 計算のボトルネックは大半が GPU メモリにある。メモリ帯域幅利用率は 96.7% に達する一方、計算コアの平均利用率は 51.6% にすぎない（[第 2 節](#section-2)）。さらに、ハードウェアと DNN モデルの進化に伴い、利用率の低いコアと飽和したメモリ帯域幅の隔たりは一層広がり得る。現代のモデルは、大きな画像、長い文、高精細グラフィックスなど忠実度の高いデータを処理するため、計算時により多くのメモリ帯域幅を消費する。また、TensorCore [Nvi17] のような高速な計算コアは、メモリへの圧力をさらに高める。

メモリ集約型 DNN ワークロードの最適化は難しい。GPU DRAM や共有メモリなど複数のメモリ層にまたがる複雑なデータアクセス・再利用パターンを改善する必要があるためである。メモリの観点では、DNN 計算は各演算子について、1）メモリ階層をまたいで入力テンソルをロードし、2）コアで計算し、3）結果テンソルをメモリ階層へ格納する処理を繰り返す。良いデータアクセスパターンを得るには、テンソルを分割した tile の各次元サイズを慎重に計算しなければならない。このようなタイリング戦略は、既存の実践でも得るのが難しい [Cut23, Zhe20, Zhu22]。さらに、アルゴリズム上の意味が異なるため、演算子ごとに異なるデータアクセスパターンが必要になり得る。この演算子間の多様性により、演算子間データ再利用は特に難しく、実現不能な場合も多い。あるメモリ層で導出された演算子の tile 形状が後続演算子と一致しなければ、その層で tile を再利用することは難しい。したがって既存手法は、演算子内最適化に集中して演算子間の中間テンソルをすべて GPU メモリなど最下位層に置くか、ルールベースの演算子融合で演算子間メモリのオーバーヘッドを軽減する。これらのルールは、要素単位演算子のレジスタ融合 [Pyt17, Aba16c, Che18e] や、限定された演算子型の共有メモリ融合 [Zhe22f] など、特定の組合せにしか適用できず、入力サイズやハードウェア構成が変わると最適でない場合がある。

本稿では、一般的な演算子からなるエンドツーエンド DNN モデルのメモリアクセスを包括的に最適化する深層学習コンパイラ WELDER を提案する。設計は三つの重要な観察に基づく。第一に、隣接する二演算子間の tile 形状競合を解決するため、各演算子の計算ロジックをテンソル式などで正確に保持できれば、出力 tile 形状を後方から前方へ伝播して、整合する tile 形状を自動推論できる。第二に、どの tile 形状が高性能かを決める際、計算パターンを TensorCore などのハードウェア特性に整合させれば、全メモリ層のデータトラフィックを最小化するだけでよい。tile 構成が整合した演算子のデータトラフィックは、入出力 tile サイズと入出力テンソル形状から容易にモデル化できる。最後に、メモリ階層全体を考えると、メモリトラフィックの最適化は各層間で本質的に独立している。すなわち層間独立性である。特に、このトラフィックモデルは対象メモリ層の tile 構成だけで決まる。これらの観察により、独立した各メモリ層で隣接演算子を整合し、トラフィックコストに従って適切な層の最適 tile サイズを決め、さらに多くの演算子へ最適化を広げるという効率的な手順で空間全体を最適化できる。

WELDER はこれらの知見を新しい DNN コンパイラ設計へ組み込む。第一に、細粒度データ管理のため、DNN 計算をモデル化する tile レベルのデータフローグラフ tile-graph を提案する。各ノードは一度にテンソルの一つのデータ tile を処理する。DNN 計算を多層メモリ階層へ対応付けるため、WELDER では各ノードのデータ tile サイズと、二ノード間で tile を再利用するメモリ層を制御できる。具体的には、SetConnect インターフェースで各辺の再利用層を設定し、Propagate インターフェースで接続されたノード群の tile 構成を推論する。第二に、tile レベルのデータフローを包括的かつ効率的に最適化するため、WELDER は層間独立性を利用して最適化空間を複数の部分空間へ分解する。その上で、各辺の異なるメモリ接続候補を列挙し、トラフィックコストモデルに従って各部分空間の効率的な tile 構成を決める二層のスケジューリング方針を提案する。最後に、最適化済み実行計画を、ハードウェア層で定義した Allocate、LoadTiles、ComputeTile、StoreTiles の四つの抽象計算インターフェースを介して、対象アクセラレータの実行可能コードへ対応付ける。

tile レベルの包括的データフロースケジューリングにより、WELDER はレジスタ上の要素単位融合や共有メモリ融合など、一般的な演算子融合を初めて一つの枠組みに統合する。この汎用性により、既存のルールベース手法ではほぼ未探索だった 89 種類の非典型的な演算子融合パターンを自動的に発見できる（[第 5.2 節](#section-5-2)）。また、単一演算子さえ GPU メモリに収まらない高解像度画像など、任意に大きな入力を持つ DNN モデルにも容易に対応できる。具体的には、ホストメモリなどの層を現在のメモリ階層へ追加することで、ホストとデバイスを合わせた階層全体の最適実行計画を生成する。

TVM [Che18e]、Rammer [Ma20a]、Roller [Zhu22] の上に WELDER を実装した。評価では、視覚、NLP、3D グラフィックスなどのタスクについて、古典的構造から最近の構造までを含む 10 個の最先端 DNN モデルを用いた。NVIDIA と AMD の両 GPU で、WELDER は PyTorch、ONNXRuntime、Ansor といった最先端 DNN フレームワーク・コンパイラを大幅に上回り、最大でそれぞれ 21.4×、8.7×、2.8× の高速化を達成した。自動最適化でありながら、高度に最適化された手作業の DNN 推論ライブラリ TensorRT [Ten17a] と NVIDIA のモデル専用実装 Faster Transformer [Fas21] も、最大 3.0×、1.7× 上回る。TensorCore のように計算コアが高速なハードウェアでは、改善幅がさらに大きく、将来の AI アクセラレータにおけるメモリ最適化の重要性を示している。

<span id="section-2"></span>

## 2 動機

現代の DNN はメモリ律速である。[図 1](#figure-01) は、ONNXRuntime [Onn21] で代表的な DNN ベンチマークを実行した際の、計算 FLOPS とグローバルメモリスループットを含む平均 GPU 利用率を示す。平均計算利用率は 51.6% にすぎない一方、メモリ利用率は 96.7% である。モデル別に見ると、畳み込みと行列乗算が中心で比較的高い計算利用率（80% 超）を得る ResNet と BERT は、古典的モデルの代表例である。一方、近年提案された他の一般的モデルは、計算集約型演算子以外にもメモリ集約型パターンを導入するため、計算効率が低い。また新しい DNN モデルは古典的モデルに比べ、メモリのロードに対するストアのトラフィック比が高い傾向にある。これらのモデルが高忠実度データを処理し、各層で大きな活性化を生成することが主な理由である。しかし ONNXRuntime など現行システムでは演算子間トラフィックを減らす最適化が限られ、大きな中間データをグローバルメモリ経由で頻繁に交換する。したがって、演算子をまたぐメモリアクセス効率の最適化が必要である。

<span id="figure-01"></span>

![図 1. モデルごとの計算・メモリ利用率。](../../papers/welder/figure-01.png)

**図 1.** NVIDIA V100 GPU 上の各モデルにおける計算 FLOPS とメモリ帯域幅利用率。

**競合する演算子内・演算子間データ再利用パターン。** 両者を同時に最適化するのは難しい。演算子は通常、全テンソル次元を走査する多層の入れ子ループとして実装される。演算子内では、複雑なループタイリングにより複数メモリ層のデータ再利用が暗黙に最適化される [Cut23, Zhe20, Zhu22]。Matmul と Softmax という連続する二演算子を考える。独立に最適化すると、共有メモリ上の最適 tile サイズは Matmul で $[32\times64]$、Softmax で $[4\times128]$ と異なる。そのため Softmax は Matmul の中間データを共有メモリで再利用できず、[図 2](#figure-02) のように合計遅延は 0.36ms になる。両者に演算子内・演算子間の再利用を同時に考慮させると、融合演算子の遅延は 0.29ms に下がり、1.26x 高速になる。整合後の tile サイズ $[16\times128]$ では、各演算子は単独実行時に 15% と 4% 性能が低下する、すなわち演算子内再利用には準最適な tile を使うことで、自身の効率を犠牲にして全体効率を高めている。これは、メモリアクセスを包括的に最適化するには、演算子内と演算子間の双方を扱う効率的な再利用手法が必要であることを示す。

<span id="figure-02"></span>

![図 2. 未融合・融合時の Matmul-Softmax の遅延。](../../papers/welder/figure-02.png)

**図 2.** 未融合、融合、および Matmul と Softmax の各カーネルの遅延。

**主要な観察。** [図 2](#figure-02) をさらに分析し、三つの観察を得た。第一に、出力 tile 形状から始める形状推論の連鎖により、演算子間で整合する tile 構成を導出できる。たとえば Softmax の [4×128] 出力 tile を計算するなら、その計算ロジックから依存する入力 tile も [4×128] と分かる。これを Matmul の出力 tile とすれば、Matmul の入力 tile は [4×k] と [k×128] になり、k は Matmul の reduction 次元を超えない任意の大きさに設定できる。このように共有メモリで中間 tile [4×128] を再利用し、二演算子を融合できる。

第二に、整合した tile 構成と元のテンソル形状があれば、総メモリトラフィックを解析的に容易に導出できる。この例で Matmul は形状 [98304×64] の A と [64×128] の B を入力し、[98304×128] の C を出力する。Softmax は C を入力し、同じ形状の D を生成する。A、B、D はグローバルメモリにある。まず D の一つの出力 tile [4×128] を計算するトラフィックを求める。Matmul は A から [4×k]、B から [k×128] の tile をロードし、中間 tile [4×128] を Softmax が共有メモリで消費した後、[4×128] の tile を D へ書く。入力形状から k=64 とできるため、一つの出力 tile のグローバルメモリトラフィックは 35KB、すなわち ((4*64+64*128+4*128)*4Bytes(FP32)) である。共有メモリでの再利用により中間 tile のトラフィックは省かれる。D 全体には 24,576 回、すなわち (98304*128)/(4*128) の計算が必要で、総トラフィックは 840MB、すなわち 24,576*35KB となる。同じ計算で出力 tile を [16×128] に変えると、264MB まで減る。

最後に、テンソル形状を指定すれば、トラフィックコストは共有メモリの [4×128] や [16×128] といった対象層の tile 構成だけで決まる。したがって、下位メモリ層からのトラフィックコストを最適化するため、各層の tile サイズを独立に選べる。

以上から、出力 tile 形状で隣接演算子群を整合し、メモリトラフィックに基づいて最適形状を決め、各メモリ層を独立に最適化するという有効な方法が得られる。WELDER は粗粒度の演算子間依存を細粒度の tile レベル依存へ変え、演算子間の偽の障壁を取り除いて並行性を高める。

<span id="section-3"></span>

## 3 WELDER の設計

[第 2 節](#section-2) の観察に基づき、WELDER は包括的なメモリアクセススケジューリング空間で現代 DNN の性能を高める。[図 3](#figure-03) に概要を示す。完全な DNN モデルを入力し、tile 単位の計算タスク、すなわち operator-tile のデータフローグラフ tile-graph へ変換する（[第 3.1 節](#section-3-1)）。tile-graph はデータ tile 構成とメモリ配置を細かく制御する。WELDER は「先に接続し、その後スケジュールする」方法で演算子内・演算子間の再利用競合を解決する。まず隣接演算子が特定メモリ層で tile を再利用できると仮定して接続し、共通する最良の形状を導出して総トラフィックが減るかを調べる。そのため SetConnect と、形状推論を連鎖させる Propagate の二インターフェースを提供する。次に、グラフ接続と部分グラフスケジューリングからなる二段階アルゴリズムで、複数メモリ層向けの効率的な実行計画、すなわち階層 tile-graph を再帰的に決定する（[第 3.2 節](#section-3-2)）。最後に Allocate、LoadTiles、ComputeTile、StoreTiles の四つの抽象インターフェースで特定アクセラレータの実行可能コードへ変換する（[第 3.3 節](#section-3-3)）。抽象アクセラレータのメモリ仕様は、tile-graph スケジューリング層で最適化の指針に使われる。

<span id="figure-03"></span>

![図 3. WELDER のシステム概要。](../../papers/welder/figure-03.png)

**図 3.** WELDER のシステム概要。

<span id="section-3-1"></span>

### 3.1 Operator-tile と Tile-graph

WELDER は DNN 計算を operator-tile という細粒度タスクで定義する。畳み込みなどの DNN 演算子は複数の同質な operator-tile として実装でき、ストリーミングまたは並列に実行して出力テンソルの全データ tile を計算する [Ma20a]。各 operator-tile は入力テンソルから切り出した tile を受け取り、出力テンソルの tile を計算する。そのロジックはインデックスベースのテンソル式で表す [Che18e]。[図 4a](#figure-04) と[図 4b](#figure-04) は Conv と MaxPool の例である。Conv は $[3\times3\times C]$ を入力して $[1\times1\times C]$ を計算し、MaxPool は $[2\times2\times F]$ を入力して $[1\times1\times F]$ を計算する。

<span id="figure-04"></span>

![図 4. Operator-tile と接続された tile-graph。](../../papers/welder/figure-04.png)

**図 4.** 二つの operator-tile、（a）Conv と（b）MaxPool、および（c）それらを tile-graph へ接続した例。簡潔にするため Conv の重みテンソルを省略する。

共有メモリなど階層メモリ資源の利用率を上げるため、WELDER は隣接 operator-tile を共通の中間 tile、すなわち reuse-tile で「接続」できる。二番目の operator-tile は、一番目の出力を完全な中間テンソルとして実体化せず直接利用する。[図 4](#figure-04)（c）は Conv と MaxPool を [2 × 2 × F] の reuse-tile で接続する例である。隣接する各辺に沿って複数の operator-tile を接続し、tile-graph と呼ぶデータフローグラフを構成できる。

**Tile 伝播。** 接続後、tile-graph 内の多くの tile は相関し、出力形状をグラフ全体へ伝播して自動推論できる。出力ノードから入力へ形状推論を連鎖させ、各 operator-tile のテンソル式と出力サイズから、依存する入力領域を正確に決める。Gather や stride 付き Convolution のように疎または非連続アクセスがある場合、式解析は保守的な入力形状上界を与える。出力ノードが複数なら、共通の祖先を持つため出力形状も相関し得る。この場合、最初の出力 tile を伝播した後、残りの出力ノードに別々の形状を伝播して最初のものと整合させる。二つの伝播で形状が一致しなければ、後者を現在のグラフへ接続しない。

**メモリトラフィックとフットプリント。** tile 伝播後、tile-graph のメモリトラフィックとフットプリントを決められる。個々の tile-graph のトラフィックは入出力 tile サイズの合計であり、これに出力テンソル全体に必要な tile-graph 数を掛けると総量になる。最小フットプリントは bestfit [Gar72] などのメモリ割当アルゴリズムで、全 tile をトポロジカル順に割り当てて求める。さらに reduction 軸を持つ入力 tile を小さく分け、順次ロード・消費して結果を出力 tile へ累積すれば、フットプリントを減らせる。具体的な方針は、伝播中に reduction 軸上の異なるサイズを自動的に試せる。

<span id="section-3-2"></span>

### 3.2 Tile-graph のスケジューリング

初期データフローグラフで表された DNN をアクセラレータへ対応付けるには、各演算子を再帰的に operator-tile へ分割して各メモリ層へ収め、上位層で接続して演算子間再利用を活用する。DNN 計算全体は二次元空間のデータストリーミングパイプラインとしてモデル化でき、tile は縦方向にメモリ階層を上下し、横方向には各層で後続演算子へ渡る。[図 5](#figure-05) は連続する Conv、ReLU、MaxPool を L2 から L0 の三層へ対応付ける例である。Conv の入力 tile は L2 から L1、L0 へ繰り返しロードされる。L0 で Conv と ReLU を接続すると Conv の出力を ReLU の入力として再利用し、二演算子が L0 の tile-graph を構成する。同時に L1 では仮想ノード Conv+ReLU にまとめられる。ReLU の出力は L1 の tile へ順次書き出され、L1 でさらに接続した MaxPool の入力として再利用される。三演算子は L1 で単一 tile-graph、L2 で仮想ノード Conv+ReLU+MaxPool となる。この再帰処理後、最下位層では全演算子が単一 tile-graph として接続される。

<span id="figure-05"></span>

![図 5. 三層メモリ階層へ対応付けた三演算子。](../../papers/welder/figure-05.png)

**図 5.** 連続する三演算子を三層メモリ階層へ対応付けた例。Conv の重みは省略する。

**最適化空間の分離。** DNN 計算は大半がメモリ律速なので、パイプラインの主目的をメモリトラフィックの最小化へ置き換えられる。各メモリ層のトラフィック最適化に固有の独立性を利用し、空間全体を複数の部分空間へ分解する。ある tile-graph が下位層からロード・ストアする総トラフィックは、全入出力形状を導出する出力 tile 形状だけで推定できる。同一または異なる層の各 tile-graph は、最適形状を探して独立にトラフィックを最適化できる。[図 5](#figure-05) では、L0 の Conv+ReLU を L1 の Conv+ReLU と MaxPool からなるグラフとは独立に最適化でき、これを層間独立性と呼ぶ。また L0 の Conv-Relu と MaxPool の最適構成も、各々が L1 のグラフと独立なので互いに独立であり、これを層内独立性と呼ぶ。実際の唯一の制約は、下位層の tile が上位層より大きいことである。通常、下位層の容量は上位層より大きいため成立する。以上の性質により、接続計画があれば各 tile-graph を独立にスケジュールできる。

<span id="figure-06"></span>

![図 6. WELDER のスケジューリングインターフェース。](../../papers/welder/figure-06.png)

**図 6.** WELDER のスケジューリングインターフェース。

**スケジューリングインターフェース。** [図 6](#figure-06) のように、グラフ接続と部分グラフのタイリングを制御する二インターフェースを提供する。SetConnect は tile-graph の辺へメモリ層を割り当て、既定では最下位層とする。接続後、Propagate に出力 tile の各次元サイズと、入力 tile の任意の reduction 軸を指定して形状を推論する。[図 5](#figure-05) では、SetConnect で Conv–Relu を L0、Relu–MaxPool を L1 で接続できる。Conv+Relu には出力形状 $[1,1]$ を指定し、Propagate で中間 reuse-tile $[1,1]$ を推論する。同様に Conv+Relu+MaxPool は出力 $[1,1,F]$ から中間形状 $[2,2,F]$ を推論できる。SetConnect は二ノードの接続を追加し、Propagate はノードの構成を設定するため、両者で tile-graph の辺と頂点を更新する完全なインターフェースになる。これらは WELDER の方針だけが使用し、エンドユーザーには透過的である。さらに MemFootprint と MemTraffic でフットプリントと総トラフィックを計算し、スケジューリングを導くコストモデルとする。

<span id="figure-07"></span>

![図 7. 二段階 tile-graph スケジューリングアルゴリズム。](../../papers/welder/figure-07.png)

**図 7.** 二段階 tile-graph スケジューリングアルゴリズム。

**スケジューリング方針。** WELDER は二段階アルゴリズムでデータフロー計算を最適化する。グラフ接続スケジューラが各辺の再利用層を変えて接続計画を列挙し、部分グラフスケジューラが分離された各部分グラフの効率的な tile 構成を素早く探す。[図 7](#figure-07) に示すように、DNN データフローグラフ $g$ とデバイス $d$ に対し、まず全ノードと出力辺をトポロジカル順に列挙する（1–3 行）。各辺で SetConnect などを使って接続層を試す（5 行）。全辺の接続層が 0 より大きい接続部分グラフを抽出する。0 は最下位層を表し、大きな数ほど上位である。ExtractSubgraph は 26–31 行に実装される。抽出した部分グラフに SubGraphTiling を適用して複数の効率的構成を得て、ハードウェア上のプロファイルで最良を選ぶ（7–10 行）。全接続層と比較後、現在の辺へ最良の層を設定する。

次に SubGraphTiling は部分グラフと直前の層の構成を入力し、現在の層を探索する。Roller [Zhu22] と同様の形状拡張で出力次元のサイズを列挙し（14 行）、1 などの初期形状から、総トラフィックを減らしてハードウェア特性に整合する形状へ広げる。出力形状から Propagate で完全な構成を推論し、MemFootprint で容量超過を調べる。超過しなければ MemTraffic などによるトラフィックをキーに整列済みリストへ加える（15–18 行）。最後に現在層でトラフィック最小の上位 $K$ 構成を選び、上位層の部分グラフを抽出して ExtractSubgraph と SubGraphTiling を再帰的に呼ぶ（20–24 行）。

WELDER は各層の容量を仮定しない。方針が中間データを置く最適な層とサイズを選び、総遅延を最小化する。ただし小さすぎる tile は演算子内再利用を悪化させるため、十分大きな中間 tile を保持できる大容量の高速上位メモリを持つハードウェアが有利である。結果の階層 tile-graph は最下位層の完全グラフから始まり、上位層で複数の部分グラフへ再帰的に分割される。

<span id="section-3-3"></span>

### 3.3 ハードウェアアクセラレータへの対応付け

階層 tile-graph は抽象実行計画であり、特定アクセラレータのコードへ対応付けられる。WELDER は階層メモリを持つ抽象デバイスを提供し、層数、容量、トランザクション幅を MemLevels で取得する（[図 7](#figure-07)）。ホストメモリや SSD などを追加層として既存デバイスへ容易に加え、単一デバイスに収まらないテンソルも扱える（[第 5.4 節](#section-5-4)）。性能向上は主に層間の帯域幅差から得られる。下位層がボトルネックで上位層が中間 tile を保持できれば、演算子間転送を高速な上位層で自動的にパイプライン化する。

<span id="table-01"></span>

![表 1. 抽象アクセラレータのデバイスインターフェース。](../../papers/welder/table-01.png)

**表 1.** 抽象ハードウェアアクセラレータのデバイスインターフェース。

<span id="figure-08"></span>

![図 8. 階層 tile-graph のコンパイル手順。](../../papers/welder/figure-08.png)

**図 8.** 階層 tile-graph のコンパイル手順。

階層 tile-graph を実行するため、Allocate、LoadTiles、ComputeTile、StoreTiles の四インターフェースを提供する（[表 1](#table-01)）。[図 8](#figure-08) の手順は最下位の完全 DNN グラフから始まる。各 tile-graph で対応層の作業領域を Allocate し、LoadTiles で入力をロードする。次に全ノードをトポロジカル順に実行する。最上位層なら ComputeTile でコア上に直接実行し、そうでなければ上位 tile-graph を再帰的に実行する。最後に StoreTiles で結果を下位層へ格納する。アクセラレータがインターフェースをコードエミッタと実行可能関数のどちらで実装するかにより、これはコード生成にもランタイム処理にもなる。WELDER ではコードエミッタとして実装し、アクセラレータ固有ロジックを生成する。この再帰手順で階層 tile-graph 全体を展開し、必要な最適化を含むモデル全体の計算プログラムを自動生成する。

<span id="section-4"></span>

## 4 実装

WELDER は TVM [Che18e]、Roller [Zhu22]、Rammer [Ma20a] に基づく。TVM でカーネルスケジュールを記述し、Roller で効率的構成を列挙し、Rammer でエンドツーエンドのグラフを最適化する。tile-graph、伝播、スケジューリング、コード生成などの中核は 5.2k 行で実装した。ONNX グラフを入力し、定数畳込みや単純な要素単位融合を行ってから tile-graph へ変換し、包括的なメモリ最適化を行う。統一デバイスインターフェースにより CUDA、ROCm GPU、GraphCore IPU を実装した（[表 1](#table-01)）。GPU ではグローバルメモリ、共有メモリ、レジスタの三層を扱う。大きな画像にはホスト層を追加して CUDA GPU と IPU のメモリを拡張する。

<span id="section-4-1"></span>

### 4.1 ハードウェアに整合した Tile 探索

**効率的なサイズの列挙。** データアクセス効率に影響するハードウェア要因を、トラフィックモデルのペナルティとして考慮する。非コアレッシングアクセスでは追加トランザクションを総量へ加える。CUDA GPU なら連続 128 byte（一トランザクション）のコアレッシングが望ましい。大きな tile で並行性が不足すれば、コア利用率に応じてトラフィックを増やす。総フットプリントが容量を超える構成には無限大のペナルティを課す。非効率な候補を避けるため、モデル上でトラフィックを最も減らす次元だけを列挙し、最小の上位 k 候補だけを取る。

**整合した計算並行性。** 同じ threadblock の最上位 operator-tile は統一 block サイズを使う。まずレジスタ層で十分な並列 tile 数を強制し、ハードウェア並行性へ整合させる。V100 では各 SM に 4 warp scheduler、各 warp に 32 thread があるため、tile 数は 128 より大きくする。全演算子の tile 数の最大公約数が 128 などの並行性より大きく 1024 などの上限より小さければ、それを共通 thread-block サイズとし、そうでなければ並行性と同じ値にする。決定後、レジスタ層の全 tile を thread へ bind する。一 thread が複数 tile を実行するなら TVM の virtual thread で bind し、全 memory bank への並行アクセスと bank conflict の回避を可能にする。

**TensorCore 対応。** CUDA GPU 上で GEMM、BatchMatmul、implicit GEMM [Li16c] による Convolution を高速化する。CUDA の Warp-Level Matrix Operations へ bind する軸を演算子へ注釈し、最上位 tile は thread でなく warp へ bind して MMA を実行する。列挙時には thread 数を warp サイズの整数倍にし、各 tile の M、N、K 軸を MMA fragment サイズの整数倍にする制約も加える。

<span id="section-4-2"></span>

### 4.2 コード生成とコンパイル

カーネル生成は TVM に基づく。レジスタ層の接続は compute_inline primitive で実装する。共有メモリ層では、接続された各部分について TVM で独立カーネルを生成し、追加 pass で単一の融合カーネルへ組み合わせる。

**Load/store の書換え。** TVM の独立カーネルはグローバルメモリを読み書きする。lowering に TIR [Ten20] pass を追加し、共有メモリアクセスへ書き換える。race condition 防止の memory fence と bank conflict 対策の padding も加える。元の global kernel は device function となり、最終融合カーネルに含まれる。

**Block/thread index の再対応付け。** 直接接続できない演算子では blockIdx と threadIdx を再対応付けする。Transpose などの BlockIdx 関係はテンソル式から導出する。ThreadIdx は 2D thread block を 1D へ接続する。thread 間 reduction や TensorCore が threadIdx.x と threadIdx.y を使う 2D block を要求し、他が threadIdx.x だけの 1D block を使う場合に必要である。総 thread 数が等しければ対応付けられる。

**メモリ管理。** 独立カーネル内の領域と演算子間 reuse buffer を含む全共有メモリを一元管理する。実行トポロジーから各 buffer の live range を解析し、allocate/free 列へ変換する。bestfit で各割当の offset を計算し、data type と TensorCore の alignment、たとえば misaligned address を避ける 32 byte alignment を考慮する。

**コンパイル高速化。** 並列コンパイルと部分グラフキャッシュを使う。構成間の独立性により、複数 process で各構成を並列に build・評価できる。また DNN で同じ部分グラフが繰り返されるため、signature で各パターンを cache する。12 層 BERT なら第一層の kernel code と profile latency を残り 11 層へ再利用する。

<span id="section-5"></span>

## 5 評価

<span id="section-5-1"></span>

### 5.1 実験設定

NVIDIA GPU、AMD GPU、Graphcore IPU を備えた三台のサーバーで評価する。

NVIDIA は二台で、Azure NC24s_v3 VM は Intel Xeon E5-2690v4、Tesla V100（16GB）、Ubuntu 16.04、CUDA 11.0、ローカル workstation は Intel Xeon E5-2678 v3、GeForce RTX 3090、Ubuntu 18.04、CUDA 11.3 を使う。AMD は Intel Xeon E5-2640 v4、Radeon Instinct MI50（16GB）、Ubuntu 18.04、ROCm 5.2.3 を使う。IPU は Azure ND40s_v3 VM、Intel Xeon Platinum 8168、16 IPU、Poplar-sdk 3.0 である。

**DNN ワークロード。** CNN、Transformer、CNN-Transformer、MLP の 10 モデルを評価し、多くは各タスクの最先端である。[表 2](#table-02) に型、タスク、発表年を示す。全モデルで公式 PyTorch 実装を無変更で使う。

<span id="table-02"></span>

![表 2. WELDER で評価した DNN モデル。](../../papers/welder/table-02.png)

**表 2.** WELDER で評価した DNN モデル。

**ベースライン。** PyTorch v1.12 [Pyt17]、ONNXRuntime v1.12 [Onn21]、Ansor v0.9 [Zhe20]、Rammer [Ma20a]、NVIDIA 向け TensorRT v8.4 [Ten17a] と比較する。Transformer では NVIDIA の手作業 C++ ライブラリ FasterTransformer v5.2 [Fas21] も使う。最新 AStitch [Zhe22f] を実装する BladeDISC v0.3.0 [Bla23] と、multi-stream scheduling の Nimble [Kwo20a] も NVIDIA 上のベースラインに含める。

各モデルを PyTorch で trace し ONNX へ export して、各フレームワークへ入力する。ONNXRuntime は CUDA execution provider と最適化レベル ALL、TensorRT は Python API で engine を build、Ansor は task 数の 800× trial とする。入出力は GPU memory に置き、転送コストを避ける。warm-up 後に各 workload を 5 秒以上反復し、変動が小さいため平均速度だけを報告する。モデル間平均は幾何平均である。

<span id="section-5-2"></span>

### 5.2 NVIDIA GPU での評価

本節では、1）最先端フレームワーク・コンパイラに対する性能、2）TensorCore による追加改善、3）専門家の融合ルールを超える新パターンの自動発見、4）メモリ効率と計算効率の改善、5）包括的最適化の探索効率、という五点を検証する。

<span id="figure-09"></span>

![図 9. NVIDIA V100 GPU（SIMT Core のみ）でのエンドツーエンド推論性能。](../../papers/welder/figure-09.png)

**図 9.** NVIDIA V100 GPU（SIMT Core のみ）でのエンドツーエンド推論性能。左：batch size 1、右：64。

**エンドツーエンド性能。** [図 9](#figure-09) は batch size 1 の性能を最良結果で正規化した高速化として示す。WELDER の幾何平均高速化は PyTorch に 4.29×、ONNXRuntime に 2.07× である。PyTorch は計算グラフの Python overhead が大きく、この設定で弱い。ONNXRuntime はそれを除き、pattern-based graph optimization を行う。Rammer は依存カーネルを共有メモリで融合できず、WELDER が 1.96× 上回る。BladeDISC は多くのモデルで unsupported operator により PyTorch へ fallback する。失敗しない BERT、MobileNet、BSRN、NeRF では WELDER が 2.70× 速い。Nimble が失敗するモデルを除く平均は 1.79× である。

Ansor は高性能 tensor program とレジスタ上の Matmul+BiasAdd、Conv2D+ReLU などのルール融合で性能を改善するが、追加再利用を活用できず、WELDER との平均差は 1.44× である。小チャネル畳み込み中心の NAFNet（1.70×）と BSRN（1.43×）で顕著である。attention block の LayerNorm や Softmax を融合できないため、BERT（1.71×）、Swin-Transformer（1.45×）、ViT（1.56×）でも差がある。CNN と Transformer の双方で機会を利用し、MobileViT、Conformer、Restormer では 1.64×、1.39×、1.29× となる。NeRF は計算集約的 MLP が支配的で追加機会がなく、1.09× にとどまる。

TensorRT は NVIDIA の高度に最適化された専用推論ライブラリである。BERT（1.02×）と Swin-T（0.97×）では WELDER と同等である。TensorRT が一般的な Transformer 向けに専門家のルールと社内カーネルを既に持ち、改善余地が少ないためである。WELDER は計算集約演算子で性能の低いカーネルを使いながら、パターンを自動発見して同等に達する。カーネル最適化は相補的であり、改善すれば WELDER にも恩恵がある。新しく多様な NAFNet では汎用性により最大 3.09×、全体平均で 1.47× TensorRT を上回る。

[図 9](#figure-09) は batch size 64 も示す。[表 2](#table-02) の最後の三モデルは入力が大きく PyTorch で trace できない。この設定でも PyTorch、ONNXRuntime、Rammer、BladeDISC、Nimble、Ansor、TensorRT を平均 1.83×、1.90×、2.1×、1.57×、1.49×、1.47×、1.21× 上回る。大きな batch では CUDA library を使うフレームワークが良くなり、最初の三者に対する差は縮むが、Ansor との差は batch 1 と同程度である。

**TensorCore での性能。** 高い計算 throughput はメモリアクセス圧力を高める。TensorCore は FP16 のみなので、PyTorch で重みと activation を FP16 に変換する。TensorRT は自身の converter の方が良いため例外とし、それ以外は onnxconverter_common [Onn20] を使う。

<span id="figure-10"></span>

![図 10. NVIDIA V100 GPU（TensorCore 有効）でのエンドツーエンド推論性能。](../../papers/welder/figure-10.png)

**図 10.** NVIDIA V100 GPU（TensorCore 有効）でのエンドツーエンド推論性能。左：batch size 1、右：64。

[図 10](#figure-10) の batch 1 の 10 ケースで全ベースラインを上回る。平均は PyTorch に 7.18×（MobileNet で 21.4×）、ONNXRuntime に 3.08×（Conformer で 8.72×）、BladeDISC に 5.29×（MobileNet で 16.9×）、Nimble に 2.72×（NeRF で 5.58×）、Rammer に 2.76×（NAFNet で 5.42×）、TensorRT に 1.53×（NAFNet で 2.98×）である。

<span id="table-03"></span>

![表 3. WELDER と FasterTransformer の性能。](../../papers/welder/table-03.png)

**表 3.** WELDER と FasterTransformer の性能。

batch 64 の残る 7 ケースでは、PyTorch、ONNXRuntime、BladeDISC、Nimble、Rammer、TensorRT をそれぞれ 1.98×、2.13×、1.97×、3.84×、3.45×、1.16× 上回る。

一部は SIMT より大きく、NeRF では TensorCore 上の TensorRT に 2.34×、SIMT では 1.16× である。TensorCore が計算集約部分を高速化し、残るメモリ集約部分の最適化が重要になるためである。

Ansor は TensorCore 非対応なので含めない。公平のため WELDER の TensorCore を無効にして FP16 を SIMT で比較した[図 11](#figure-11)では、FP32 よりやや高い平均 1.74×、最大 2.82× である。

<span id="figure-11"></span>

![図 11. FP16、TensorCore なしで Ansor と比較。](../../papers/welder/figure-11.png)

**図 11.** FP16、TensorCore なしで Ansor と比較。

**別の NVIDIA GPU。** Ampere の RTX-3090 でも評価した。V100 と異なる load、TensorCore instruction、SM 数を持つ。簡潔にするため常に最良の TensorRT だけと比較した。[図 12](#figure-12) の全 34 ケースの幾何平均で 1.40×、V100 の 1.36× と近く、多様なアーキテクチャへの適応を示す。

<span id="figure-12"></span>

![図 12. NVIDIA RTX-3090 で TensorRT と比較。](../../papers/welder/figure-12.png)

**図 12.** NVIDIA RTX-3090 で TensorRT と比較。

**自動発見したパターン。** 10 モデル、34 ケースの一意な演算子型で約 300 の融合部分グラフを発見した。うち 89 は reduction 演算子を二つ以上含み、Ansor の単純な要素単位ルールでは扱えない。多くは手動ルールや自動融合で未探索の非典型パターンである。[図 4](#figure-04) の二例は複数の Convolution または MatMul（Dot）とメモリ集約演算子を単一カーネルへ融合する。演算子数は 2–48、Ansor の基本融合に対して平均 1.87×、最大 5.4× である。最多パターンは 191 回使われた。

<span id="table-04"></span>

![表 4. WELDER が発見した融合パターン例。](../../papers/welder/table-04.png)

**表 4.** WELDER が発見した融合パターン例。

この汎用性は専門家によるモデル専用実装も上回る。FasterTransformer [Fas21] は BiasAdd+Transpose と Layernorm+Softmax のような要素単位・非要素単位の双方を手動最適化するが、WELDER はすべて自動融合する。さらに短い系列では Q*K と後続 Softmax も融合し、長さ 128 の BERT では融合、512 の Conformer では非融合と自動判断する。

[表 3](#table-03) の三モデルで平均 1.11×、ViT で最大 1.73× FasterTransformer を上回る。batch 1 の ViT では stride と kernel size がともに patch size 32 の非典型畳み込みが主因で、WELDER カーネルは 4.4x 速い。これは新しい形状・パターンへの適応性を示す。

7 層 MLP の NeRF では、専門家が fully-fused MLP [Mul21a] を一から作ることが多い。WELDER はこれを単一 GPU kernel へ自動融合し、前 6 層を TensorCore、出力層を SIMT として全中間結果を共有メモリに置く。自動結果は報告値と同程度の 5× 超である（コード [Mul21] は V100 非対応のため評価できない）。

NAFNet、BSRN、MobileNet でも異なる畳み込みを Pooling や PixelShuffle と融合する。NAFNet では連続 pointwise convolution と間の normalization を融合する。DWConv と PWConv からなる separable convolution では構成に応じて順序を決める。上層では feature map が大きく channel が少ないので DWConv+PWConv として完全な map を共有メモリに cache し、下層では map が小さいので PWConv+DWConv として DWConv 用の完全な channel を cache する。

<span id="figure-13"></span>

![図 13. 選択モデルの遅延、カーネル数、メモリトランザクション、IRS。](../../papers/welder/figure-13.png)

**図 13.** 3 モデルの遅延、GPU カーネル数、実行グローバルメモリトランザクション、中間結果サイズ（IRS）。FP32、batch size 64。

**アブレーションと感度。** WELDER-none は全演算子間接続を無効にし、WELDER-base はレジスタ層だけを有効にする。codegen 手法の Ansor も含める。[図 13](#figure-13) で base は none より遅延 52%（2.08×）、launch 67%、transaction 52%、IRS 66% を削減する。base は Ansor と近く、一般 tile 手法の効率を示す。共有メモリ接続を加えると base より遅延 29%（最大 1.82x）、launch 60%、transaction 25%、IRS 65% をさらに削減する。重みアクセスは融合で最適化できないため、transaction の削減は IRS より小さい。

BERT（text 128–512）、Conformer（audio frame 128–512）、NAFNet（image 256x256–1024x1024）で入力を変えた。[図 14](#figure-14) では大画像の NAFNet で融合利得が増え、他の Transformer では減る。系列長に対して計算が二次増加し、メモリ集約性が下がるためである。

<span id="figure-14"></span>

![図 14. 入力サイズを変えた感度分析。](../../papers/welder/figure-14.png)

**図 14.** 入力サイズを変え、WELDER-base と比較。

**コンパイル時間。** 多数の tuning・profiling trial を要する Ansor と[表 5](#table-05)で比較する。他は library kernel を直接呼び、追加時間がないので除外する。WELDER は一桁以上速い。Ansor は全演算子に大空間を作り、machine-learning tuning で暗黙に再利用を最適化するため、演算子当たり 800 trial と online cost model training を要する。WELDER は層別方針で空間を分解し解析モデルで traffic を推定するため、部分グラフ当たり 20 trial で済む。

<span id="table-05"></span>

![表 5. Ansor と WELDER のコンパイル時間。](../../papers/welder/table-05.png)

**表 5.** Ansor と WELDER のコンパイル時間。

**計算集約モデル。** ResNet [He16c]、VGG [Sim14a]、UNet [Ron15] は大きな convolution が支配的である。メモリ最適化中心でも TensorRT と同等になる。上位層接続が少なくても、Ansor [Zhe20] や Roller [Zhu22] と似た多層タイリングで高性能な単一演算子を生成できる。一方 cuDNN [Cud23] は winograd [Lav16] など、テンソル式から導出しにくい数値アルゴリズムを使う。この差を補うメモリ最適化がなければ TensorRT より遅い。[表 6](#table-06) の ResNet は DirectConv が良く両者対応なので同等だが、UNet と VGG16 は TensorRT の winograd が支配的で融合機会もなく TensorRT が良い。式の書換えによる winograd 対応は直交するため将来課題とする。

<span id="table-06"></span>

![表 6. 計算集約モデルの性能。](../../papers/welder/table-06.png)

**表 6.** 計算集約モデルの性能。

<span id="section-5-3"></span>

### 5.3 AMD ROCm GPU での評価

<span id="figure-15"></span>

![図 15. AMD ROCm MI50 GPU のエンドツーエンド推論性能。](../../papers/welder/figure-15.png)

**図 15.** AMD ROCm MI50 GPU のエンドツーエンド推論性能。左：batch size 1、右：64。

PyTorch、ONNXRuntime、Ansor と比較する。TensorRT と AStitch は NVIDIA 専用なので除く。[図 15](#figure-15) の 10 モデルで PyTorch、ONNXRuntime、Rammer を平均 2.62×、1.71×、2.14×、Ansor を 1.53× 上回る。batch 64 では順に 1.69×、1.23×、1.86×、1.47× である。ONNXRuntime で失敗する CNN は除いた。MI50 は V100 より FLOPS が低く帯域幅が高いため、公式仕様上ワークロードがより計算集約的で、メモリ最適化機会が少なく、高速化もわずかに小さい。

<span id="section-5-4"></span>

### 5.4 ホストメモリへの拡張

抽象デバイス層により、大規模 DNN 用に階層を拡張できる。UNet や VGG16 で高解像度医用画像 [Sid21] を扱うと、単一テンソルが GPU に収まらない。SwapAdvisor [Hua20] や Capuchin [Pen20] の tensor-based swapping は粒度が大きく効かない場合がある。WELDER は拡張階層上で包括的 traffic 最適化を行い、ホストから tile を load、デバイスで接続 tile を再利用して計算し、結果を store する。デバイス層の再利用だけを無効にした変種と比較する。

<span id="table-07"></span>

![表 7. 大規模 DNN をホストメモリへ拡張。](../../papers/welder/table-07.png)

**表 7.** 大規模 DNN をホストメモリへ拡張。

**GPU の拡張。** [表 7](#table-07) の UNet と VGG16 で、接続により平均 2.63×、1.89×、ホスト転送を 3.11×、2.90× 改善する。pinned memory と CUDA stream を伴う double buffering で転送と計算を重ねるため、traffic 削減率は実速度より高い。

**GraphCore IPU の拡張。** 異なる MIMD 構成で 300MB の小メモリを持つ IPU [Ipu23] でも初期評価した。二モデルを同じ方法で 2048*2048 入力とし、3.63×、3.09× を得た。限られた容量のため double buffer を無効にしたので GPU より改善率が高い。

<span id="section-6"></span>

## 6 議論

設計は主に静的モデル向けである。動的実行には二つの実用策がある。第一に PyTorch 2.0 で一般的な JIT compile により動的グラフを静的部分グラフへ変換し、計算支配部分を WELDER で最適化する。第二に、テンソル形状が動的でも演算子内部 tile は静的に決められるため、静的な tile レベル融合計画を生成し、並列 task 数だけを入力形状に任せられる。

<span id="section-7"></span>

## 7 関連研究

演算子融合は launch overhead を減らし高速メモリの locality を高める。TVM [Che18e]、Ansor [Zhe20]、XLA [Xla17]、DNNfusion [Niu21] はレジスタ融合を行う。AStitch [Zhe22f]、Apollo [Zha22h]、DeepCuts [Jun21] は既知型のルール、Bolt [Xin22] は限定的 template で共有メモリまで融合する。TensorRT [Ten17a] と ONNXRuntime [Onn21] も一般モデル向けの専門家ルールを持つ。WELDER は型を仮定せずテンソル式の一般演算子に使え、形状依存の resource behavior から最良層を自動決定する。Rammer [Ma20a]、HFuse [Li22e]、Nimble [Kwo20a] は horizontal fusion や multi-stream、CUDA graph で並行性と launch を改善する。WELDER は Rammer 上で相補的な vertical fusion による包括的メモリ最適化を行う。

Ansor [Zhe20] と Roller [Zhu22] は loop または tile による演算子内最適化を行う。Roller と Triton [Til19] も tile でカーネル性能を最適化する。WELDER は演算子内外を包括し、Roller の概念を tile-graph へ一般化して、明示的メモリ階層上の効率的方針を提案する。

NeRF の fully-fused MLP [Mul21a]、CNN の手動カーネル [Wan20h]、Transformer attention [Fas21, Fan21b] のように特定パターンを強く融合する研究もある。WELDER は大半を自動的に実現し、新パターンも生成する。

kernel fusion は画像処理 [Qia18, Qia19] や HPC [Wah14] にも使われるが、領域固有ルールに依存する。WELDER は DNN 向けだが一般のテンソル式演算子に適用でき、他領域にも役立ち得る。

<span id="section-8"></span>

## 8 結論

現代 DNN がメモリ集約的になるという観察から、新しい tile-graph 抽象で効率を最適化する WELDER を提案した。多層階層で演算子内外の再利用を包括的に最適化する。一般的融合を初めて一枠組みに統一し、最大 48 演算子を一 kernel にするものを含む 89 の非典型パターンを発見した。この汎用性で最先端を大幅に上回る。さらに、大きく接続性の高い on-chip memory など、将来の AI accelerator の傾向を利用する体系的方法を与える。

## 謝辞

匿名査読者と shepherd の Byung-Gon Chun 教授による広範な助言に感謝する。本研究は中国国家重点研究開発計画（No. 2021ZD0110202）から一部支援を受けた。

<span id="section-9"></span>

## 9 Artifact 付録

### 概要

WELDER は tile-graph 抽象によるエンドツーエンド DNN model compilation を提供する。本 artifact は NVIDIA V100 の主要評価結果を再現する。

<span id="section-9-1"></span>

### 9.1 対象

以下を検証する。

- [図 9](#figure-09)、[図 10](#figure-10)、[図 11](#figure-11)、[表 3](#table-03)、[表 6](#table-06) のエンドツーエンド性能。

- [図 1](#figure-01) と[図 2](#figure-02) の動機実験。

- [図 13](#figure-13) のアブレーション。

- [表 5](#table-05) のコンパイル時間。

- [表 7](#table-07) の GPU scale-out 実験。

<span id="section-9-2"></span>

### 9.2 内容

本 artifact は WELDER の全ソースコードを含む。環境構築用 Dockerfile を提供する。各図表に再現スクリプトがある。50 を超えるモデルケースの全コンパイル、特に Ansor は長時間を要するため、V100 向けの事前コンパイル済み log と model も提供する。詳細は repository の README.md を参照されたい。

<span id="section-9-3"></span>

### 9.3 公開場所

artifact は GitHub repository[+artifact] にある。git で clone し osdi2023welder branch を checkout する。

<span id="section-9-4"></span>

### 9.4 要件

CUDA runtime 11.0 以上をサポートする driver と NVIDIA V100 GPU が必要である。

[+internship]: 本研究は Microsoft Research での internship 中に行われた。

[+artifact]: <https://github.com/microsoft/nnfusion/tree/osdi2023welder>
