---
title: 'Ladder: Hardware-Aware Tensor Transformation'
createTime: 2026/09/10 00:31:20
permalink: /ja/papers/ladder/
---

> [Lei Wang](https://x.com/Lei_Wang_1999) [+intern], [Lingxiao Ma](https://xysmlx.github.io/), [Shijie Cao](https://caoshijie0501.github.io/), [Quanlu Zhang](https://dblp.org/pid/165/8284), [Jilong Xue](https://dblp.org/pid/06/10336.html), [Yining Shi](https://dblp.org/pid/161/3927-1.html) [+intern], [Ningxin Zheng](https://dblp.org/pid/234/5381), [Ziming Miao](https://dblp.org/pid/216/9568.html), [Fan Yang](https://fanyangcs.github.io/), [Ting Cao](https://www.microsoft.com/en-us/research/people/ticao/), [Yuqing Yang](https://dblp.org/pid/91/9064-1.html), [Mao Yang](https://www.microsoft.com/en-us/research/people/maoyang/). 第18回 USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), 2024年7月10–12日, 米国カリフォルニア州サンタクララ, 307–323頁. [Ladder: Enabling Efficient Low-Precision Deep Learning Computing through Hardware-aware Tensor Transformation](https://www.usenix.org/conference/osdi24/presentation/wang-lei). <a href="/paper/ladder.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>. arXiv 登録および TeX ソースは公開されていないため, 本閲覧版は正式出版 PDF に基づく. 厳密な紙面レイアウトと参考文献については原 PDF を正とする.

[+intern]: 本研究は Microsoft Research でのインターンシップ期間中に行われた。

## 概要

深層学習モデルの性能向上に対する需要の高まりを受け, 深層学習が誤差に対して持つ頑健性を活用する低精度計算へとパラダイムが移行している. 新しい低精度データ型や最適化手法が登場する一方, 既存のハードウェアとソフトウェアによる対応は不十分かつ非効率であり, 低精度計算から実際の性能向上を得ることは難しい.

本論文では, 進化し続けるカスタムデータ型と, 現行ハードウェアが固定的に対応する精度形式との隔たりを埋める新しいコンパイラ LADDER を提案する. LADDER は汎用型システム tType と拡張テンソル式を用い, カスタムデータ型を第一級の要素として深層ニューラルネットワーク (DNN) 計算を最適化済みの計算パイプラインへ変換する. これにより, データの格納, アクセス, 型変換を効率化する最適化空間が明示される. LADDER は新しいテンソルスケジューリングプリミティブ群とハードウェア認識最適化方針によって複雑な変換空間を探索し, メモリ階層や DNN 演算子が異なる場合にも最適な性能を確保する. 評価の結果, LADDER は多様な低ビット精度のカスタムデータ型を体系的に支援し, ハードウェアを変更せずに現代のアクセラレータ上で DNN 計算性能を大きく向上できることが分かった. この仕組みにより, モデル設計者はデータ型最適化を探究でき, ハードウェアベンダーは多様な精度形式への対応を柔軟に拡張できる.

<span id="section-1"></span>

## 1 はじめに

近年, 深層学習モデルの大規模化が進み [Bro20, Dev18, Kap20], GPU などのハードウェアアクセラレータには一層高い計算性能が求められている. 深層学習は本質的に誤差へ頑健であるため, float64 のような高精度を必要とする科学計算などの従来ワークロードと異なり, より低精度の算術を利用できる. この流れに沿って, 最新アクセラレータは 32-bit, 16-bit, さらには 8-bit 浮動小数点演算など, より多くの低精度計算ユニットを新世代へ組み込んでいる. 同時にモデル開発者は, モデル精度と学習効率の最適な均衡を目指し, 混合精度形式を含む各種カスタム低精度データ型を精力的に研究している. モデル配備時には, LLM の 2-bit 固定小数点精度 [Che24b] や, 複数の値が同じスケーリング係数を共有するグループ型 [Dar23] のように, 計算をさらに小さな表現へ変換して極限の効率を狙うこともできる.

しかしハードウェアアクセラレータが, 多様で急速に変化する精度形式, すなわちカスタムデータ型への要求に追随するのは難しい. チップ面積が限られハードウェアコストも高いため, 各アクセラレータが組み込める標準データ型向け計算ユニットは数種類に限られるからである. 16-bit 未満のような近年対応した低精度型についても, 細粒度の低ビットデータアクセスを粗粒度のメモリシステムへ整合させるのが複雑で, 既存ソフトウェアは一般に非効率である. 例えば NVIDIA GPU の共有メモリバンク幅は 4 bytes であり, 8-bit 要素をそのままロードまたはストアすると帯域が無駄になりやすい. 複数の値をまとめてパックし, 各メモリ階層の特性へ合わせるといった容易でない最適化が必要になる. その結果, 新しいデータ型をさまざまな演算子や形状と組み合わせてカーネルライブラリを最適化する作業は難しい. 例えば高度に最適化された NVIDIA GPU 向け CUTLASS でも, INT8 行列乗算は 422 TFlops, 利用率 68% にとどまる. 新しいカスタムデータ型への不十分で非効率な対応は, モデルとアクセラレータ双方の革新を大きく妨げている.

この課題について二つの点を観察した. 第一に, ハードウェアアクセラレータがカスタムデータ型の計算命令を持たなくても, 固定ビット幅の不透明なデータチャンクへキャストすれば, メモリシステムには任意のデータ型を格納できる. 第二に, ほとんどのカスタムデータ型は, 既存ハードウェアの計算ユニットが対応するより広い標準データ型へ損失なく変換できる. 例えば NF4 テンソルは, 型を変換すれば FP16 または FP32 演算で計算できる. そこでデータ格納と計算を分離し, あらゆるカスタムデータ型を支援する汎用手法を考えた. すなわちテンソルはカスタムデータ型のまま格納・転送し, 型変換を介して標準データ型で計算する. 現代の DNN モデルはメモリ集約的であり, 最新ハードウェアはメモリウォール問題に直面している [Shi23a]. この手法はメモリトラフィックと占有量を削減して低ビット型の性能上の利点を効果的に引き出せるため, 重要性が増している.

ただし, 既存アクセラレータ上で汎用カスタムデータ型の計算パイプラインを効率的に支援するのは容易ではない. 一般的なテンソル計算パイプラインでは, DRAM, L2 キャッシュ, 共有メモリ, レジスタなど, 複数のメモリ階層からデータをロードする. 第一に, どの階層でテンソル型を変換するかは, メモリ占有量, データアクセストラフィック, ハードウェアコストなどへ大きく影響し, 最適化が複雑である. 例えばレジスタ内で低ビットのデータチャンクを高ビット型へ変換するとレジスタスピルが発生し, 性能が急落することがある. 第二に, 異なるデータ型を含むパイプラインでは, データアクセススループットを最大化するため, メモリバンクなどのメモリシステムへ合わせた異なるレイアウト最適化が必要になる. メモリアクセスの swizzle [Nvi24a] など既存の最適化は少数の特定データ型向けに設計されており, 一般化しにくい.

これらの課題に対し, 汎用カスタムデータ型上の深層学習計算を効率化するコンパイラ LADDER を提案する. MXFP のようなブロック単位型を含め, 急速に進化するカスタムデータ型を容易に実装できるよう, LADDER はまず tType という汎用型システムを導入する. tType は本質的にタイル単位のデータ型であり, 型のビット幅, 要素形状, 型変換関数を明示することで一般的なカスタム型をすべて定義できる. LADDER は tType に基づき, DNN 演算子を表す既存のテンソル式を拡張し, 各テンソルへ tType を直接注釈できるようにする. これにより, カスタムデータ型を含む DNN 計算を標準的な計算パイプラインへ体系的に変換できる.

カスタムデータの格納, アクセス, 型変換を含む計算パイプラインを最適化するにあたり, パイプライン中のテンソル格納とアクセスは論理的に等価な複数の形式へ変換でき, 形式ごとに性能への影響が大きく異なることに着目した. 例えばサブテンソルは行優先, 列優先, ブロック単位, あるいは独自定義のレイアウトで格納できる. 計算命令へ合わせて所定の形状までパディングしたり, 上位メモリから異なる粒度, 例えば異なるタイル形状でアクセスしたりすることもできる.

これらはすべて全体性能へ大きく影響する. このような変換を行えるよう, LADDER は slice, map, pad, convert からなるテンソルスケジューリングプリミティブ群を導入し, 既定の計算パイプラインを最適化されたものへ変換する.

特定の計算パイプラインに最適なテンソル変換を導くには, メモリ階層間と演算子間の最適化を一体として考える必要がある. 例えば特定のデータレイアウトを隣接演算子へ伝播させれば, 明示的なレイアウト変換のコストを避けられる. また特定メモリ層のデータレイアウトは, そのメモリの特性と上位層からのアクセスパターンをともに考慮しなければならない. 階層横断と演算子横断の最適化は広大な探索空間を作る. LADDER は階層別のハードウェア認識方針でこの変換空間を最適化する. 下位メモリが望ましいデータアクセス粒度をヒントとして示し, 上位層はその粒度へ整合させて最適な計算粒度を決める. したがって LADDER は DNN 計算をまずタイルレベルのデータフローグラフとしてモデル化し, 次に粒度認識スケジューリング方針で変換スケジュールを最適化する.

LADDER は TVM [Che18], Roller [Zhu22], Welder [Shi23a] を基盤として実装され, オープンソースで公開されている [+source]. LADDER の DNN 演算コンパイル機能は BitBLAS [+bitblas] としても公開されている. これは既存の DNN/LLM フレームワークへ統合でき, 現在の深層学習エコシステムで効率的な低精度計算を実現するライブラリである. NVIDIA A100, V100, RTX A6000, AMD Instinct MI250 GPU 上の DNN 推論を評価したところ, LADDER はハードウェアが直接対応する型で最新 DNN コンパイラを上回り, GPU が未対応のカスタムデータ型も効率的に処理して最大 14.6× の高速化を達成した. LADDER は, 現代のハードウェアアクセラレータ上の DNN 計算で, カスタムデータ型として表現される汎用低ビット精度を体系的に支援する最初のシステムである. 実際の性能フィードバックを得ながらモデル設計者が柔軟なデータ型最適化を探究でき, ハードウェアベンダーもハードウェアを変更せず幅広い型へ対応できるようになる.

[+source]: <https://github.com/microsoft/BitBLAS/tree/osdi24_ladder_artifact>
[+bitblas]: <https://github.com/microsoft/BitBLAS>

<span id="section-2"></span>

## 2 背景と動機

<span id="section-2-1"></span>

### 2.1 深層学習における精度要件

大言語モデル (LLM) に代表される深層学習モデルの大規模化に伴い, 計算効率を高めメモリを節約する低ビット・混合精度計算への要求も強まっている. 本節では深層学習における新しいデータ型要件を紹介する.

**低ビット数値精度.** 深層学習モデルのデータ表現には長く FP32, すなわち 32-bit 浮動小数点数が使われてきた. しかし近年の実践から, FP32 の高精度が常に必要とは限らず, 低精度でも同等の効果を保ちながらコストを下げられることが分かっている. この変化を示す代表例が, 自動混合精度 (AMP) 学習における FP16/BF16 計算である [Mic18]. Transformer Engine [Mic22] や MS-AMP [Pen23e] はさらに進んで, 重み, 勾配, さらにはオプティマイザのテンソルにも FP8 を用い, 深層学習の精度低減を押し広げている. 推論時にはモデルを 8-bit や 4-bit まで大幅に量子化することが多い [Det22, Fra22, Xia23]. 最新研究では重みを 2-bit, さらには 1-bit まで量子化する試みも進む [Che24b, Wan23]. これは主に事前学習済み重みが冗長性を持ち, 計算の大半が順伝播だからである. [図 1](#figure-01) は深層学習モデルで用いられる各種データ形式を示し, 高精度形式から低ビット形式への明確な移行を表している.

<span id="figure-01"></span>

![深層学習の学習と推論で用いられる多様な狭精度データ型](../../papers/ladder/figure-01.png)

**図 1.** 深層学習の学習と推論で用いられる多様な狭精度データ型.

**グループ単位の精度スケーリング.** 低精度深層学習モデルの正確さと頑健性を高める一般的な方法として, スケーリング係数で値を再スケールし, データ分布をより正確に表現する手法がある. 従来はテンソル単位またはチャネル単位の係数がよく使われる. これに対しグループ単位のスケーリングは粒度が細かく, サブテンソルやグループの分布をよりよく捉えられるため, 性能を改善できる. 例えば学習後量子化 (PTQ) [Fra22] ではグループサイズ 128 または 64 がよく使われ, 各グループを FP16 でスケーリングする. OCP-MXFP [Dar23] では 32 要素のグループへ 8-bit の共有スケールを適用する.

**混合精度演算.** データ量子化では, テンソルごとに低ビット量子化への感度が異なるため混合精度演算が生じる. 例えば混合精度学習では FP32, FP16, FP8 など高ビットと低ビットのテンソルを組み合わせ, 計算効率と精度の均衡を取って性能を最適化する. LLM の量子化でも, 量子化しやすい重みは低ビットで表現できる一方, 量子化が難しい活性化には高ビット表現が必要である. この違いから W4A16, すなわち重みを 4-bit, 活性化を 16-bit で表す構成や, W2A16, W1A8 などの混合精度演算が生まれる [Che24b, Fra22, Wan23].

<span id="section-2-2"></span>

### 2.2 GPU における精度サポートの不足

GPU などのハードウェアアクセラレータは, 深層学習で変化するデータ型要件へ継続的に適応してきた. NVIDIA Fermi など初期世代の GPU は FP32 や FP64 といった標準型に対応していた. 深層学習ワークロードが重要になると, Pascal アーキテクチャで FP16 のような低精度形式が導入された. Turing は推論向けに INT4 と INT8 を加え, 対応範囲をさらに広げた. Ampere は BF16 を導入し, 機械学習用途における性能上の利点と数値範囲の均衡を取った. 最新の NVIDIA Hopper は FP8 に対応してこの流れを進め, 精度と性能のトレードオフを調整しながら効率を追求している. この進化は, 多様な計算ワークロードを処理する GPU の汎用性が高まっていることを示す. ただしハードウェアは通常, アルゴリズムやモデルの要求に遅れる. 未対応の型は, 対応済みの高精度型へ変換するか, その型で模擬しなければならず, 大きな性能問題や非効率を招き得る.

<span id="section-2-3"></span>

### 2.3 低精度計算の非効率性

低精度計算は, 細かなデータアクセス粒度と TensorCore のような特殊ハードウェアユニットのため, 特に最適化が難しい. [表 1](#table-01) のとおり, NVIDIA V100/A100 と AMD MI250 の三つの最新 GPU 上で, 最新のソフトウェアライブラリとコンパイラを使い, 複数精度の標準行列乗算ベンチマークを測定した. ここから三点が分かる. 第一に, 低精度計算のハードウェア利用率は概して低く, 平均 60% 未満である. 現在の深層学習で最も一般的な FP16 でさえ平均利用率は約 60% にとどまる. 第二に, ハードウェア対応済みの精度でもソフトウェアの支援が不十分な場合がある. A100 と MI250 はともに INT8 へ対応するが, 既存の深層学習コンパイラの大半は両 GPU 上の INT8 計算に対応しない. 第三に, 新しい精度要件へハードウェアが迅速に対応するのは難しい. 例えば FP8 は次世代の NVIDIA Hopper で初めて利用でき, F16 × NF4 のような混合精度計算は最新 GPU のどれにも対応していない.

<span id="table-01"></span>

![データ型, GPU, ライブラリ, コンパイラ別の行列乗算利用率](../../papers/ladder/table-01.png)

**表 1.** $M,N,K=16384$ における MatMul $[M,N]=[M,K]\times[N,K]$. 「X」は tensor core または matrix core で未対応であることを示す.

<span id="section-2-4"></span>

### 2.4 本研究の洞察

[図 2](#figure-02) に示す FP16×INT8 混合精度行列乗算を例に, 本研究の主要な洞察を説明する. DNN 演算はしばしば計算パイプラインとして実装される. 入力テンソルから小さなデータタイルを複数のメモリ階層を介して連続的にロードし, 最上位の計算コアで処理する. 各メモリ層には通常, L1 層の 8-byte トランザクション長のような望ましい最小アクセス粒度がある. 最新 GPU の一部には, 二次元データタイルを一度にロードする高効率な組み込み命令もある. 例えば `ldmatrix.2x2.f16` は 2×2 タイルをロードする. データタイルは通常ストライド付きメモリ空間へ格納されるため, アクセスがトランザクション長や命令形状と整合せず, 帯域利用率が下がりやすい. 左図では両テンソルとも, L1 からの各アクセスの利用率が半分にしか達していない. さらに FP16×INT8 の計算命令がないため, 対応データをレジスタへロードできても演算自体を実行できない. この問題は, データ型幅, メモリトランザクション長, 命令形状に基づいてテンソルレイアウトを適切に変換すれば回避できる. 右図では各 2×2 タイルを L1 層の連続メモリ空間へ格納し, 上位層のロード命令が帯域を完全に利用できるようにしている. 計算命令は FP16 形式にしか対応しないため, L2 から L1 へロードする途中で第二テンソルを INT8 から FP16 へ変換する. これにより L2 から L1 へのロードでは低ビット型による少ないトラフィックを活かし, L1 から L0 へのロードではトランザクション整合によってメモリ帯域を使い切り, 最後に型変換を通じてハードウェア計算ユニット上の計算を高速化できる. この例は, ハードウェア未対応のカスタムデータ型による DNN 計算でも, レイアウトとデータ型を適切に変換すればスケジュール・最適化できることを示す.

<span id="figure-02"></span>

![テンソル変換前後の混合精度行列乗算パイプライン](../../papers/ladder/figure-02.png)

**図 2.** MatMul: $C_{\mathrm{FP16}}[2,2]=A_{\mathrm{FP16}}[2,4]\times B_{\mathrm{INT8}}[2,4]$.

<span id="section-3"></span>

## 3 LADDER の設計

[第 2 節](#section-2) の観察に基づき, データ型を第一級の要素として扱い, テンソル変換によってカスタムデータ型上の効率的な DNN 計算を支援するコンパイラ LADDER を設計した. [図 3](#figure-03) にシステム構成を示す.

<span id="figure-03"></span>

![LADDER のシステム概要](../../papers/ladder/figure-03.png)

**図 3.** LADDER のシステム概要.

LADDER の中心は TypedTile (tTile) 抽象であり, タイルベースのテンソル抽象へデータ型, すなわち tType を加える ([第 3.1 節](#section-3-1)). アルゴリズム設計者は FP16 など一般的なデータ型を使うか, MXFP8 や NF4 などのカスタム型を tType として定義し, その型上の DNN 計算を記述できる. LADDER は DNN モデルを入力として tTile ベースのデータフローグラフ, すなわち tTile-graph へ変換する. そこでは各演算子を tTile ベースの計算タスク, すなわち tTile-operator として定義する ([第 3.1 節](#section-3-1)).

また LADDER はハードウェアアクセラレータを多層階層として抽象化し, 各層の要件を tTile で表す. これを tTile-device と呼ぶ ([第 3.1 節](#section-3-1)). tTile-device は対応データ型やトランザクションサイズなど, 各層の要件を明示する. tTile-graph 内の tTile を tTile-device へ整合させれば, tTile-graph が表す DNN 計算をハードウェアアクセラレータ上で実行できる.

初期 tTile-graph とハードウェア仕様を受け取ると, LADDER は DNN モデルをアクセラレータ上の効率的な実行計画へコンパイルする. tTile-graph を tTile-device へスケジュールしてハードウェア階層の要件を満たすため, LADDER はスケジューリング機構と方針を分離する. 機構として四つの tTile 変換プリミティブを提案し ([第 3.2 節](#section-3-2)), 初期 tTile-graph を, tTile の構成, 変換, ハードウェア階層上の配置を細かく制御した tTile-graph へスケジュールする. tTile 抽象は DNN 計算のスケジューリング空間を広げ, メモリ占有効率と遅延効率の新しいトレードオフを生む. 方針としては観察に基づくヒューリスティクスを用い, 遅延効率を最適化するハードウェア認識の階層別方針を提供する ([第 3.3 節](#section-3-3)). 最後に, スケジュール済み tTile-graph を実行用ハードウェア命令へ lowering する.

<span id="section-3-1"></span>

### 3.1 TypedTile: 型注釈付き Tensor Tile

LADDER は DNN 計算とハードウェア要件の両方を表すため, データ型注釈付きのタイルベーステンソル抽象 TypedTile (tTile) を提案する. [図 4](#figure-04) に tType, tTile, tTile-operator の定義を示す.

<span id="figure-04"></span>

![tType, tTile, tTile-operator の定義](../../papers/ladder/figure-04.png)

**図 4.** tType, tTile, tTile-operator の定義.

**tType.** tType はタイル単位のテンソルデータ型であり, 一つまたは複数要素の数値形式と精度を一般化する. tType は各要素のビット数 (`nElemBits`), 形状 (`shape`), 変換先 tType (`c_tTypes`) と変換関数 (`c_funcs`) を持つ. `shape` は一つの tType が表す要素数と配置を示す. `shape=[]` のとき tType はスカラーを表す. 例えば FP16 は `tType(nElemBits=16, shape=[])` と記述できる. tType はグループ単位データ型のように, 要素群も表現できる. 変換関数は, その tType を別の tType へ損失なく変換する方法を記述する. この定義により INT4, NF4, FP8, MXFP など一般的な低精度型を tType として表せる.

**tTile.** tTile は tType が注釈されたテンソルタイルで, `tTile(shape, dtype)` と表す. `shape` はタイルの論理形状, `dtype` はその tType である. テンソルは複数の tTile へ分割でき, 一つの tTile は一つの tType 粒度に複数要素を含み得る. この抽象はデータ形状と格納粒度をともに明示する.

**tTile-operator と tTile-graph.** tTile-operator は入出力が tTile である計算タスクである. LADDER は入力, 出力, 中間テンソルへ tType を注釈してテンソル式を拡張する. DNN モデルは tTile-operator のデータフローグラフとして表され, これを tTile-graph と呼ぶ.

ハードウェアアクセラレータは, DRAM やレジスタなどのメモリ層と計算ユニットからなる階層構造を持つ. 各層には望ましいデータアクセス形態がある. メモリ層は通常, ある粒度の連続データまたは所定形状のデータを単位とするトランザクションでアクセスする必要がある. 例えば NVIDIA GPU の共有メモリは, 32 個の 4-byte バンクからなるトランザクションを必要とする. 計算ユニットも通常, ある粒度の所定形状データを処理する. 例えば NVIDIA GPU の `hfma2` 命令は, 2 個の FP16 値を粒度として処理する.

これらの要件は tTile で記述できる. そこで LADDER はハードウェアアクセラレータを, tTile で記述された多層階層, すなわち tTile-device として抽象化する. 各層はメモリ層または計算ユニットであり, ある粒度上の形状として表される要件を tTile で, その粒度を tType で記述する.

<span id="figure-05"></span>

![tTile で表現・変換した FP16 と NF4 の混合行列乗算](../../papers/ladder/figure-05.png)

**図 5.** FP16 テンソル A と NF4 テンソル B の MatMul: (a) tType 注釈付きテンソル式, (b) NVIDIA A100 の tTile-device, (c) 計算パイプラインの疑似コード, (d) tTile 変換プリミティブを用いた Transform-Load, (e) テンソル B の変換.

[図 5](#figure-05)(b) は FP16 tensor core を備える NVIDIA A100 GPU の tTile-device を示す. FP16 tensor core の MMA 命令 [+mma] は, 二つの入力をそれぞれ $[16, 16]$ と $[8, 16]$ の粒度で処理する必要がある. これは形状 $[16, 16]$, `dtype=FP16` の tTile として表せる. FP16 tensor core のデータロード命令 [+ldmatrix] は, `half8`, すなわち 8 個の FP16 値を粒度として $[16, 2]$ のデータをロードする. これは形状 $[16, 2]$, `dtype=16B` の tTile として表せる. 共有メモリを完全に利用する要件は形状 $[32]$, `dtype=4B` の tTile, グローバルメモリの 32-byte トランザクション要件は形状 $[32]$, `dtype=1B` の tTile として表せる.

[+mma]: `mma.sync.aligned.m16n8k16.row.col.f16.f16.f32.f32`
[+ldmatrix]: `ldmatrix.sync.aligned.m8n8.x4.shared.b16`

<span id="section-3-2"></span>

### 3.2 tTile 変換

tTile は細粒度のテンソル格納とハードウェア階層の要件を明示する. 効率的に実行するには, tTile-graph 中の tTile で表された DNN 計算を tTile-device へ整合させる必要がある. [第 2 節](#section-2) の観察どおり, パイプライン中のテンソル格納とアクセスは, 論理的に等価でありながらハードウェア階層での性能影響が異なる形式へ変換できる. そこで LADDER は, tTile のレイアウトまたは tType を等価な tTile へ変換する機構を提案する. 具体的には tTile-operator の計算パイプラインを, ハードウェア階層上の Transform-Load, Compute, Transform-Store の三段階へ拡張する. Transform-Load は tTile 変換を伴って下位メモリ層から上位層へ tTile をロードする. Compute は計算ユニット上で tTile-operator の計算タスクを実行する. Transform-Store は tTile 変換を伴って上位層から下位層へ tTile を格納する.

<span id="figure-06"></span>

![四つの tTile 変換プリミティブ](../../papers/ladder/figure-06.png)

**図 6.** tTile 変換プリミティブ.

LADDER は[図 6](#figure-06) に示す四つのプリミティブで, tTile を等価な tTile へ変換する.

**Slice.** slice プリミティブは `tTile_input` のアドレス `index` から形状 `shape` の要素群を切り出し, 形状 `out_shape` の新しい tTile として返す. 通常はデータのタイリングを表すために使う.

**Map.** map プリミティブは tTile 内の要素レイアウトを変更する. `map_func` を与えると, 各要素のアドレスを期待するアドレスへ写像する. 例えば[図 5](#figure-05)(d) で L2 から L1 へ移す `TransformLoad_L1B` は, map プリミティブと `map_func` を使って要素アドレスを変更する.

**Pad.** pad プリミティブは `pad_shape` で指定した各境界を `pad_value` で埋め, `tTile_input` をパディングする. `pad_shape` の長さは `tTile_input` の形状次元数の 2 倍で, 各次元の左右境界をそれぞれ表す.

**Convert.** convert プリミティブは `tTile_input` の tType を指定した `new_tType` へ変換する. `new_tType` は入力 tType の `c_tTypes` に含まれていなければならない. convert は `tTile_input` の各要素へ `new_tType` に対応する `c_func` を適用し, 期待する `new_tType` の tTile を返す. 例えば[図 5](#figure-05)(d) の `TransformLoad_L1B` は, 計算コアが要求する FP16 tType を満たすため, convert で NF4 から FP16 へ変換する.

以上の四プリミティブにより, slice と pad で形状を変え, map で要素レイアウトを変更し, convert で tType を変換して, tTile を別の等価な tTile へ変換できる. これにより tTile-operator の各 tTile を tTile-device へ整合させ, ハードウェア階層内で効率的に処理できる.

[図 5](#figure-05) は, 四層の tTile-device, すなわち L2 から計算コアまでの階層 ([図 5](#figure-05)(b)) 上で, FP16 テンソル $A[32, 63]$ と NF4 テンソル $B[32, 63]$ を乗算し, FP32 で累積して FP16 テンソル $C[32, 32]$ を出力する例である ([図 5](#figure-05)(a)). [図 5](#figure-05)(c) に実行の疑似コードを示す. A と B の tTile は変換され, FP16 として L2 から L1 へロードされる. 次に `ldmatrix` で L0 へロードし, `mma` 命令で処理する. 中間値は L0 で FP32 として累積する. 最後に L0 の C の tTile を変換し, FP16 として L2 へ格納する. [図 5](#figure-05)(d)(e) は NF4 テンソル B を tTile-device へ整合させる詳細な変換を示す. A の変換も同様である. `mma` と `ldmatrix` は L1 に FP16 データを要求し, 各層には[図 5](#figure-05)(b) のトランザクション要件もある. そこで `TransformLoad_L1B` は $[16, 63]$ を切り出して $[16, 64]$ へパディングし, L2 の要件へ整合させる. 続いて FP16 へ変換し, L1 と L0 のトランザクション要件へ合う別の要素レイアウトへ写像する. これにより L1 で FP16 の `L1_B` $[16, 64]$ を得る. さらに `TransformLoad_L0B` が `ldmatrix` で `L1_B` を切り出し, L1, L0, `mma` コアの要件へ整合する FP16 の `L0_B` $[16, 16]$ を L0 上に得る.

<span id="section-3-3"></span>

### 3.3 ハードウェア認識 tTile-Graph スケジューリング

tTile-graph で表された DNN 計算を tTile-device へスケジュールするには, 各 tTile-operator の tTile 向け計算パイプライン, すなわち Transform-Load, Compute, Transform-Store を tTile-device へ写像する. 各 tTile-operator をメモリ層の容量に収まる複数の tTile へ分割し, tTile が各ハードウェア層の要件へ整合するよう変換をスケジュールし, 演算子間の tTile 構成と変換を協調させて全体を最適化できる. 最終的に tTile-graph 全体はデータパイプラインとしてスケジュールされる. tTile-operator ノードの tTile はハードウェア階層を上下し, 辺を越えて後続の tTile-operator ノードへ渡される.

tTile は DNN 計算スケジューリングへテンソル変換という次元を加えるため, tTile-graph のスケジューリング空間は大幅に広がる. tTile 変換はメモリ占有効率と遅延効率の新しいトレードオフも生み, スケジューリングをより複雑にする. NVIDIA GPU 上で FP16 テンソルと NF4 テンソルを MatMul する場合, ハードウェア対応の制約から NF4 を FP16 へ変換する必要がある. この変換は L1 から L0 への Transform-Load より前に終えればよく, L2 または L1 に配置できる. L2 で変換すると L2 と L1 のメモリを多く消費するが, その後 tTile を L2 から L1, L0 へ移す際に計算ユニットを占有しない. L1 で変換すると L2 のメモリと帯域を節約できる一方, 型変換が計算ユニットを占有する. 演算子が計算律速なら前者はメモリ消費と引き換えに低遅延となり, メモリ I/O 律速なら後者が遅延とメモリ占有の両面で優れる. また convert は L1 から L0 への Transform-Load より前に完了すればよいため, 前の演算子へ融合して端から端までの性能を高めることもできる.

この広大な空間に対し, LADDER は端から端までの遅延を最小化する方針を提供する. 下位メモリが望ましいアクセス粒度を tTile としてヒントに示し, 上位層は変換によってその粒度へ整合させて最適な計算粒度を決める, ハードウェア認識の階層別スケジューリング方針である. 探索空間を狭め, 妥当な時間で適切な計画を得るため, 観察に基づくヒューリスティクスを用いる.

**スケジューリング方針.** アルゴリズム 1 はヒントに基づく階層別スケジューリングを示す. tTile-graph で表された DNN モデル $g$ と, tTile-device で表されたハードウェア仕様 $D$ を受け取り, スケジュール済み tTile-graph $g_{ret}$ を返す. まずグラフをサブグラフへ分けてスケジュールする (33行目). 各サブグラフは, 最下位メモリから計算コアへ tTile をロードし, 結果を最下位メモリへ戻す計算パイプラインを表す. サブグラフは単一の tTile-operator, または融合可能な tTile-operator 群である. `ExtractConnectedGraph` には既存 DNN コンパイラの手法を利用できる [Che18, Shi23a].

**アルゴリズム 1. ヒントに基づく階層別スケジューリング**

- **入力:** $g$: tTile-graph; $D$: tTile-device
- **出力:** $g_{ret}$: scheduled tTile-graph
- **関数** `GetDeviceHint(g, D)`:
  - $D=$ `SelectDeviceConfig(g, D)`
  - `HintShape = None`, `HintGranularity = None`
  - **各要素について** `layer ∈ D.layers`:
    - `HintGranularity = LCM(HintGranularity, layer.tTile.type)`
  - **各要素について** `layer ∈ D.layers`:
    - `layer.tTile = convert(layer.tTile, HintGranularity)`
    - `HintShape = LCM(HintShape, layer.tTile.shape)`
  - **各要素について** `layer ∈ D.layers`:
    - `layer.tTile.shape = HintShape`
  - **返す** $D$
- **関数** `ScheduleTransform(op, D, l_id)`:
  - `tTile_h = op.tTile[l_id - 1]`
  - `tTile_l = op.tTile[l_id]`
  - `ScheduleSlice(tTile_l, tTile_h)`
  - **もし** `LCM(tTile_l.shape, tTile_h.shape) != tTile_l.shape`:
    - `SchedulePad(tTile_l, tTile_h, D)`
  - **もし** `tTile_l.type != tTile_h.type`:
    - `ScheduleConvert(tTile_l, tTile_h, D)`
  - **もし** `nBits(tTile_h.shape[-1]) != nBits(D.layers[l_id].shape[-1])`:
    - `ScheduleMap(tTile_l, tTile_h, D)`
  - **返す** `op.transform[l_id - 1]`
- **関数** `ScheduleConnectedGraph(g, D)`:
  - `D = GetDeviceHint(g, D)`
  - **各要素について** `l_id` in `length(D.layers)`:
    - **各要素について** `op ∈ g[l_id]`:
      - `op.tTile[l_id] = ScheduleTiling(op, D, l_id)`
      - **もし** `l_id > 0`:
        - `op.transform[l_id] = ScheduleTransform(op, D, l_id)`
  - `g = ProfileAndSelect(g)`
  - **返す** $g$
- **関数** `Schedule(g, D)`:
  - `g = ExtractConnectedGraph(g, D)`
  - **各要素について** $g_{\mathit{conn}}\in g$:
    - `g_conn = ScheduleConnectedGraph(g_conn, D)`
  - **返す** $g$

サブグラフを受け取ると, まずハードウェアからヒントを推論する. 最初に適切なハードウェア構成, 例えば計算コアを選ぶ (2行目). ここではハードウェアが対応するうち, 対象に最も近いビット幅の tType を優先する. ビット数の多い数値型は命令の実装に通常より多くのトランジスタを要し, 性能も低くなりやすいからである. NVIDIA A100 では NF4 を FP16 または FP32 へ変換して処理できるが, LADDER は FP32 コア (19.5 TFlops) ではなく FP16 コア (312 TFlops) を選ぶ. 次に各ハードウェア層について, ビット整合により共通の粒度と形状を求め, ヒントを設定する (1–11行目). NVIDIA A100 を例にすると ([図 5](#figure-05)(b)), `HintGranularity` は `ldmatrix` が要求する 16B, `HintShape` は $[4, 8]$ である. 内側次元は 128B となり, グローバルメモリの 32B トランザクションと共有メモリの 128B トランザクションへ整合する. 続いて計算コアから DRAM まで, 上位から下位へ層ごとにサブグラフをスケジュールする (25–29行目). 各層ではヒントを使って `ScheduleTiling` で tTile-operator のタイリングを決め (27行目), 次に tTile 変換を決める (29行目). `ScheduleTiling` が 16B 粒度で $[4, 8]$ の倍数となるよう演算子をタイル化すれば, 後続の `ScheduleTransform` が tTile-device へ整合させられる. `ScheduleTiling` には既存テンソルコンパイラも利用できる [Che18, Zhe20, Zhu22]. `ScheduleTransform` は形状と型の双方が tTile-device へ整合するか確認し, 必要な変換をスケジュールする (12–22行目). 複数の候補が残る場合は実測し, 最良のものを返す (30行目).

**ScheduleMap.** map 変換をスケジュールする際の `map_func` は自明ではない. LADDER は, tTile 内の要素を行優先順で必要なトランザクションサイズへ写像する `map_func` の推論法を提案する. [図 5](#figure-05)(e) では, 16B 粒度で L0 の形状 $[16, 2]$ を L1 が要求する形状 $[8]$ へ写像するため, 要素を行優先順で平坦化し, 形状 $[4, 8]$ を得る. map は別の `map_func` にも対応できる. この方針は最適性を保証しないが, [第 5 節](#section-5) のとおり既存最良手法を上回り, GPU 上の効率的な低精度 DNN 計算を実現できる. 今後, より高度な方針によってこのスケジューリング機構の最適化空間がさらに探究されることを期待する.

<span id="section-4"></span>

## 4 実装

LADDER はオープンソース DNN コンパイラ TVM [Che18], Welder [Shi23a], Roller [Zhu22] を基盤とし, Python と C++ を合わせて約 5K 行で実装した. TVM を変更してカーネルスケジュールとコード生成を実装し, Roller で効率的な tTile 構成を推論する. DNN モデルを全体として最適化する最新コンパイラ Welder は, 端から端までのグラフ最適化に用いる.

LADDER の入力は PyTorch プログラムである. PyTorch 組み込み型については DNN モデルプログラムを変更する必要がない. PyTorch が未対応の新しい型についてはカスタム演算子で PyTorch を拡張し, ユーザー定義型上のテンソル式を表す. PyTorch プログラムを受け取ると ONNX グラフへエクスポートする. ONNX も新しい型上の計算を表せるよう拡張し, tType 注釈付きテンソル式を ONNX グラフノードの属性へ保存する. エクスポートした ONNX グラフと対象アクセラレータの tTile ベース仕様ファイルから, LADDER は ONNX グラフを tTile-graph へ自動変換してスケジュールし, 対象アクセラレータのデバイスコードを生成する.

DNN アクセラレータとして広く使われる NVIDIA GPU と AMD GPU の双方へ LADDER を実装した. 本節の残りでは NVIDIA GPU 上の実装を詳しく述べ, AMD GPU 上の実装を簡潔に説明する. 最新 Hopper GPU の FP8 tensor core のような新命令や Graphcore IPU など別のアクセラレータも, tTile ベース抽象へ合致し, ハードウェア階層でデータをロード・ストアするプログラミングインターフェースを備えていれば移植できる.

<span id="section-4-1"></span>

### 4.1 NVIDIA CUDA GPU 上の LADDER

<span id="section-4-1-1"></span>

#### 4.1.1 tType と tTile

LADDER は FP32, FP16, INT8, FP8, MXFP, INT4, NF4, INT1 など一般的なデータ型の tType を実装している.

GPU は単一命令複数スレッド (SIMT) アーキテクチャであり, スレッド群が異なるデータへ同じ命令を実行する形を好む. そこで LADDER は tTile 内の要素と各メタデータを別々に格納する. [図 7](#figure-07) は NVIDIA GPU 上で形状 $\left[32, 32\right]$ の MXFP8 tTile を格納する方法を示す. 要素は一つの配列に, 共有スケーリング係数は別の配列に格納する. tTile へアクセスするときは連続スレッドが連続要素を処理し, coalesced access を実現する. 3-bit [Fra22] のように `nElemBits` が $2^{n}$ でない型もある. GPU の仕様上, LADDER はこのような型を 4B 粒度で格納する. 例えば 10 個の 3-bit 値を 4B (32-bit) 粒度へ格納できる.

<span id="figure-07"></span>

![NVIDIA GPU 上の MXFP8 tTile の格納](../../papers/ladder/figure-07.png)

**図 7.** NVIDIA GPU 上における形状 $\left[32, 32\right]$ の MXFP8 tTile の格納. E: 要素. S: メタデータ内の共有スケーリング係数.

<span id="section-4-1-2"></span>

#### 4.1.2 PTX 命令によるコード生成の最適化

NVIDIA はプログラミング用のアセンブリ命令を直接提供せず, NVIDIA GPU 向け低レベル仮想マシンとして Parallel-Thread-Execution (PTX) を導入している. PTX 仮想マシン上の ISA は NVIDIA GPU の命令レベル API とみなせる [Nvi25]. CUDA C++ コードはまず PTX へ, 次に実行用機械語へコンパイルされる. CUDA は一部ユニットについて C++ API と PTX API の両方を提供する. 例えば tensor core には WMMA C++ API と MMA PTX API があり, nvcc は一つの WMMA API を MMA 命令群へコンパイルする. MMA PTX API は WMMA C++ API より柔軟で性能も高い. LADDER は tensor core のコード生成に MMA PTX API を, Ampere GPU の新しい非同期メモリコピー機能に `cp.async` 命令を用いる [Nvi20]. また INT4 など低ビット整数から FP16 など浮動小数点への変換は大きなオーバーヘッドになり得る. LADDER は 4-bit 未満の整数変換を LOP3 命令で実装した [Nvi25]. これらの最適化のため TVM のコード生成モジュールを変更した.

<span id="section-4-2"></span>

### 4.2 AMD ROCm GPU 上の LADDER

AMD GPU も NVIDIA GPU と同様, 全 CU が共有するグローバルメモリ, 各 CU のローカルデータストア, レジスタ, 計算コアからなる階層を持つ. ローカルデータストアは共有メモリに相当する. したがって AMD GPU も, 層ごとに異なる tTile 構成を持つ四層 tTile-device として抽象化できる. ROCm は AMD GPU 向けに, CUDA と同様の機能を持ち大半の CUDA 文へ対応する HIP プログラミングモデルを提供する [Roc16]. AMD ROCm GPU に対応するため, TVM へ HIP 用の新しいコード生成バックエンドを実装した. また NVIDIA tensor core に相当する matrix core を利用するため, MFMA (Matrix Fused-Multiply Add) ISA レベル API を用いる.

<span id="section-5"></span>

## 5 評価

<span id="section-5-1"></span>

### 5.1 評価設定

**ハードウェアプラットフォーム.** 異なるハードウェアエコシステムで性能を総合的に評価するため, NVIDIA と AMD の多様な GPU 上で LADDER を測定した. NVIDIA は Tesla V100 (16GB), A100 (80GB), RTX A6000 (48GB) の三つで CUDA toolkit 12.1 を使用する. AMD は Instinct MI250 (128GB) で ROCm toolkit 5.7.0 を使用する. OS はすべて Ubuntu 20.04 で統一した.

**DNN モデル.** 複数の分野とアーキテクチャにまたがる最新 DNN モデル群で推論をベンチマークし, LADDER の有効性を評価する. 大言語モデル LLAMA-70B [Tou23a], BLOOM-176B [Les23], コンピュータビジョンモデル ResNet-50 [He16], ShuffleNet-V2 [Ma18], ViT-Base [Dos20], 音声モデル transducer Conformer-L [Gul20] を含む. データ型構成はいずれも最新研究に基づき, 深層学習コミュニティで評価済みである. LADDER はこれらをそのまま用い, モデル品質を追加で損なわない. 重みと活性化の型を $W_{\mathrm{type}}A_{\mathrm{type}}$ と表し, 以下の構成を評価する.

- **LLAMA-70B と BLOOM-176B:** $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Tou23a, Les23], $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ [Fra22, Lin23d], $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ [Det23a], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Mic22], $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$ [Dar23], $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ [Wan23].
- **ResNet-50:** $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [He16], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Mic22], $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$ [Dar23], $W_{\mathrm{INT1}}A_{\mathrm{INT4}}$ [Hua19d].
- **ShuffleNet-V2:** $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Ma18], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [She23f].
- **ViT-Base:** $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Dos20], $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ [Kuz22], $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ [Li22c].
- **Conformer-L:** $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ [Gul20], $W_{\mathrm{INT8}}A_{\mathrm{INT4}}$ [Din22], $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ [Din22].

多様な配備シナリオを覆うため, 複数のバッチサイズ (BS) と系列長 (SEQ) を設定する. LLAMA-70B と BLOOM-176B は (BS, SEQ) を (1, 1), (32, 1), (1, 4096) とし, オンライン/オフライン推論と pre-fill/decoding の各段階を代表させる. ResNet-50, ShuffleNet-V2, ViT-Base, Conformer-L はバッチサイズ 1 と 128 で評価する.

**ベースライン.** NVIDIA GPU では Welder [Shi23a], PyTorch-Inductor [Pas19], ONNXRuntime [Onn24], TensorRT [Ten24], AMOS [Zhe22d], TensorIR [Fen23], vLLM [Kwo23], および vLLM の 4-bit 量子化モデル対応である vLLM-$W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ [Kwo23] と比較する. AMD GPU では Welder [Shi23a], PyTorch-Inductor [Pas19], ONNXRuntime [Onn24], TensorIR [Fen23] と比較する.

ROCm デバイスの MatrixCore を利用するため Welder へ MIOpen と rocBLAS を統合し, TensorIR へ rocWMMA Auto Tensorize 対応を追加した. 演算子ベンチマークでは cuBLAS [Cub16], CUTLASS [Nvi24a], vLLM [Kwo23], cuDNN [Nvi25c], AMOS [Zhe22d], TensorIR [Fen23] と比較する.

<span id="section-5-2"></span>

### 5.2 NVIDIA GPU での評価

<span id="section-5-2-1"></span>

#### 5.2.1 エンドツーエンド性能

**推論遅延.** 前述の DNN モデルを Tesla A100, V100, RTX A6000 GPU 上で実行する. LLAMA-70B と BLOOM-176B は GPU メモリの制約から単一 decoder layer で測定する. 各層は同一で遅延が層数に比例するため, 単層の結果でモデル全体の性能を代表できる.

<span id="figure-08"></span>

![NVIDIA A100 GPU 上の端から端までの性能](../../papers/ladder/figure-08.png)

**図 8.** NVIDIA A100 GPU 上の端から端までの性能.

[図 8](#figure-08) に A100 GPU 上の推論遅延をまとめる. $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ では, Welder と比べて LLAMA, BLOOM, ResNet, ShuffleNet, Conformer, ViT を平均 1.0×, 1.2×, 2.0×, 1.2×, 1.1×, 1.4× 高速化した. Welder は Roller [Zhu22], cuBLAS [Cub16], CUTLASS [Nvi24a] でカーネルを生成するが, 共有メモリの bank conflict などに悩まされる. 不規則な形状の Conv2D が多い ResNet では特に顕著である. LADDER はテンソル変換スケジューリングでこの問題を解消し, ResNet の BS1 と BS128 で 1.1 ms と 7.6 ms を達成する. LLM で広く使われる $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ では vLLM より平均 2.3× 高速である. 他システムが扱わないカスタム型にも対応し, $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ の BLOOM-176B-BS1SEQ1 単層では 0.32 ms, Welder より最大 10× 高速となった.

<span id="figure-09"></span>

![NVIDIA V100 GPU 上の端から端までの性能](../../papers/ladder/figure-09.png)

**図 9.** NVIDIA V100 GPU 上の端から端までの性能.

<span id="figure-10"></span>

![NVIDIA RTX A6000 GPU 上の端から端までの性能](../../papers/ladder/figure-10.png)

**図 10.** NVIDIA RTX A6000 GPU 上の端から端までの性能.

Tesla V100 と RTX A6000 の結果を[図 9](#figure-09) と[図 10](#figure-10) に示す. A100 とよく一致している. 16GB メモリの V100 は BLOOM の decoder layer 一つでもメモリ不足となる. $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ では Welder より V100 で平均 1.1×, A6000 で 1.2× 高速である. $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ では A6000 上で vLLM より平均 2.0× 高速であり, V100 でも効率的な推論を可能にする. $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ では Welder より V100 で最大 13.3×, A6000 で 14.6× 高速である.

**メモリ使用量.** 低精度型は LLM の大きなメモリ要件を緩和する重要な手段である. A100 上の LLM 推論について複数の型構成でメモリ使用量を調べた. [図 11](#figure-11) のとおり, ビット幅の低下にほぼ比例して使用量が減る. 特に系列長 1 の decoding 段階で顕著であり, メモリ集約的な推論段階における精度スケーリングの利点を示す.

<span id="figure-11"></span>

![NVIDIA A100 GPU 上の LLM 推論のメモリ使用量](../../papers/ladder/figure-11.png)

**図 11.** NVIDIA A100 GPU 上の各データ型構成における LLM 推論のメモリ使用量.

最も極端な $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$, すなわち 1-bit 重みと 8-bit 活性化では大幅にメモリを節約できる. $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ と比べ, LLAMA の推論メモリは三つの BS/SEQ 構成でそれぞれ 74%, 74%, 24% 減り, BLOOM では 85%, 85%, 6% 減る.

**コンパイル時間.** [表 2](#table-02) で LADDER, AMOS, TensorIR, Welder のコンパイル時間を比較する. NVIDIA A100 上で ResNet と ShuffleNet をバッチサイズ 1 と 128 により端から端までコンパイルした. LADDER は平均して AMOS と TensorIR より大幅に短く, TensorIR より一桁, AMOS より二桁高速である. 一方, テンソル変換で低精度演算に対応するためスケジュール空間が広く, 低精度の性能上の利点と引き換えにコンパイル時の追加コストが生じる. そのため Welder よりはやや長い.

<span id="table-02"></span>

![NVIDIA A100 GPU 上のコンパイル時間比較](../../papers/ladder/table-02.png)

**表 2.** NVIDIA A100 GPU 上で端から端までモデルをコンパイルする時間の比較, 単位は分.

<span id="section-5-2-2"></span>

#### 5.2.2 オペレータベンチマーク

LADDER のカーネル性能を調べるため, LLAMA と ResNet の主要演算子からベンチマークを構成した. M0–M5 の六つの MatMul と C0–C7 の八つの Conv2d を, $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$, $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$, $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$, $W_{\mathrm{FP8}}A_{\mathrm{FP16}}$, $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$, $W_{\mathrm{MXFP8}}A_{\mathrm{MXFP8}}$ で測定した. 一貫性と信頼性のため全実験を NVIDIA A100 上で行った.

<span id="figure-12"></span>

![NVIDIA A100 GPU 上の演算子ベンチマーク](../../papers/ladder/figure-12.png)

**図 12.** NVIDIA A100 GPU 上の演算子ベンチマーク.

[図 12](#figure-12) のとおり, LADDER は $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ で最適な性能を示す. $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ へ移ると平均 1.8×, $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ では平均 4.5× 高速となる. Ada Lovelace, Hopper, Blackwell GPU は $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ tensor core を備える. CUDA 12.4 を用いた NVIDIA RTX 4090 でも演算子ベンチマークを実行し, ハードウェア対応の $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ 性能を評価した. 結果を[図 13](#figure-13) に示す.

<span id="figure-13"></span>

![NVIDIA RTX 4090 GPU 上の演算子ベンチマーク](../../papers/ladder/figure-13.png)

**図 13.** NVIDIA RTX 4090 GPU 上の演算子ベンチマーク.

$W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$ では cuBLAS を上回り CUTLASS と同等, $W_{\mathrm{FP8\_E5M2}}A_{\mathrm{FP8\_E5M2}}$ でも CUTLASS と同等である. cuBLAS は後者に対応しない. RTX 4090 が有効にするのは FP32 累積の $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ だけで, 理論性能は $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ と同じである. そのため M2 や M5 のような大きな行列では, cuBLAS, CUTLASS, LADDER の $W_{\mathrm{FP8}}A_{\mathrm{FP8}}$ は $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ と同程度となる. FP16 累積なら理論性能は倍になるが, NVIDIA は現在公開していない. RTX 4090 は型変換コアがより強力なため, LADDER は $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ や $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ で A100 より高い高速化率を得る.

<span id="section-5-2-3"></span>

#### 5.2.3 最適化の内訳

<span id="figure-14"></span>

![段階的な最適化の内訳](../../papers/ladder/figure-14.png)

**図 14.** 最適化の内訳.

[図 14](#figure-14) は, 異なるデータ形式について LLAMA-70B の単一系列 (BS1 SEQ1) と長い系列 (BS1 SEQ4096) のカーネルへ段階的に適用した最適化を示す. タイル認識カーネル変換によりデータ処理が滑らかになり, Roller ベースラインより 2.0× 高速化するとともに各種データ型へ対応した. PTX レベル最適化は GPU メモリ負荷を減らし, テンソル演算とレイアウトの高度な制御によってさらに最大 1.7× 高速化した. 包括的なスケジューリングは変換を最適化して最大 2.5× 高速化し, 特に MXFP8 のようなメモリ制約型へ有効である. 全体として LADDER は計算効率と適応性を高め, 複数の演算で明確な性能向上をもたらす.

<span id="section-5-2-4"></span>

#### 5.2.4 ビット幅のスケーリング

LADDER の汎用性により, 重みと活性化の双方で任意ビット幅の幅広い型へ対応できる. 精度スケーリングの性能影響を詳しく調べるため, ビット幅を段階的に下げた型設定で実験した. 二つの BS/SEQ 構成について, 端から端までの性能と個別演算子性能を評価する.

<span id="figure-15"></span>

![重みと活性化のビット幅を縮小したときの性能](../../papers/ladder/figure-15.png)

**図 15.** 重みと活性化のビット幅の縮小.

[図 15](#figure-15) に結果を示す. W と A のビット幅を下げるほど高速化率が上がり, 低精度演算の効率向上が現れる. メモリ律速となる系列長 1 の decoding では, W を $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ から $W_{\mathrm{INT2}}A_{\mathrm{INT4}}$, $W_{\mathrm{INT1}}A_{\mathrm{INT4}}$ へ下げるにつれて明確に高速化する. 一方, 計算律速となる系列長 4096 の encoding では, 混合精度演算が高精度計算へ依存するため高速化率は変わらない.

<span id="section-5-2-5"></span>

#### 5.2.5 低精度 LLM の効率と精度

低精度計算ではモデル品質と効率の両方が重要であり, 通常は効率と精度のトレードオフがある. LLAMA2-3B, 7B, 13B, 70B を例に最新手法を評価した. $W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$ の PTQ [Aut23, Mic22], $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ の GPTQ [Fra22], $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ の PTQ [Det22c], $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$ の BitDistiller [Du24], $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ の OneBit [Xu24h], $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$ の BitNet-b1.58 [Ma24] を用いる. PTQ と GPTQ は学習を含まない学習後量子化である. BitDistiller と OneBit は蒸留で 2-bit/1-bit 重み量子化を行う量子化認識学習である. BitNet-b1.58 は LLM を一から学習し, $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$ で表す三値重みを得る.

<span id="figure-16"></span>

![LLM 低精度手法の perplexity と遅延](../../papers/ladder/figure-16.png)

**図 16.** LLM の各低精度手法について, WikiText-2 上の PPL (↓) と A100 上で 1 token を decoding する遅延 (ms). $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$ の G64 は 64 要素単位の group-wise scaling を示す. LLAMA2-70B は pipeline parallelism を用いる.

[図 16](#figure-16) は WikiText-2 の perplexity (PPL) と A100 上の 1 token decoding 遅延を示す. PPL は低いほど品質がよい. $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ と $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$ の PPL は AFPQ [Zha23aa], $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ は OneBit [Xu24h], LLAMA2-3B の $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ は BitNet-b1.58 [Ma24] の報告値である. その他は公開チェックポイントと実装で評価した. $W_{\mathrm{FP8\_E4M3}}A_{\mathrm{FP8\_E4M3}}$, $W_{\mathrm{NF4}}A_{\mathrm{FP16}}$, $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ は PPL への影響が小さく, 平均 1.6×, 1.7×, 2.5× 高速化する. PTQ/GPTQ で重みを 2-bit にすると PPL が NaN になる [Du24, Xu24h] が, BitDistiller と OneBit は蒸留により 2-bit/1-bit 量子化で安定した結果を得る. ただし group-wise scaling は $W_{\mathrm{INT2}}A_{\mathrm{FP16}}$-G64 に追加計算を生じさせ, 高速化率は $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ と同程度になる.

BitNet-b1.58 は LLAMA2-3B で, 同じデータセットと token 数により学習した $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ モデルより良い PPL と 1.8× の高速化を同時に達成する [Ma24]. LLAMA2-3B は GPU を飽和させるには小さいため理論値には届かない. LLAMA2-70B で BitNet-b1.58 の $W_{\mathrm{INT2}}A_{\mathrm{INT8}}$ を評価すると $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ より 4.6× 高速であり, 精度と効率の両方で有望である.

モデル構成を比較すると規模が精度と効率の双方へ大きく影響する. $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ の LLAMA2-13B は $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ の LLAMA2-7B より両面で優れ, 量子化 LLAMA2-7B も $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ の LLAMA2-3B を両面で上回る. これは低精度計算の力を示す.

コミュニティでは低精度計算の研究が活発である. LADDER が効率のフィードバックを提供し, この方向の研究を支援することを期待する.

<span id="section-5-3"></span>

### 5.3 AMD GPU での評価

AMD Instinct MI250 上で LADDER を Welder, PyTorch-Inductor, ONNXRuntime と比較する. [図 17](#figure-17) に 6 モデルの端から端までの性能を示す.

<span id="figure-17"></span>

![AMD Instinct MI250 GPU 上の端から端までの性能](../../papers/ladder/figure-17.png)

**図 17.** AMD Instinct MI250 GPU 上の端から端までの性能.

$W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ では Welder より LLAMA, BLOOM, ResNet, ShuffleNet, Conformer, ViT で平均 2.1×, 2.35×, 1.5×, 10.5×, 1.6×, 1.5× 高速である. Welder は matrix core に rocBLAS と MIOpen を使って融合機会を失うため ShuffleNet で遅い. LADDER は効率的な matrix core カーネルを生成しながら多くの融合を可能にし, ShuffleNet-BS1 で 0.43 ms, Welder より 14.1× 高速となる. LLM の $W_{\mathrm{INT4}}A_{\mathrm{FP16}}$ では LLAMA-BS1SEQ1 が 0.73 ms で最大 3.8×, BLOOM-BS1SEQ1 が 1.75 ms で最大 4.5× 高速である.

<span id="section-6"></span>

## 6 議論

現在の LADDER 実装は主にモデル推論へ焦点を当てる. 本節では制約と今後の課題を述べる.

**Multi-GPU serving.** BLOOM-176B や LLAMA2-70B は単一 GPU に収まらず, 配備には複数 GPU が必要である. Multi-GPU 対応は LADDER と相補的である. LADDER は一つのアクセラレータ上の低精度計算を扱い, Multi-GPU フレームワーク [Kwo23, Sto23d, Lin24i, Zhe22] はモデル分割と GPU 間の並列計算を扱う. フレームワークがモデルを分割し, 各デバイスの計算を LADDER へ渡せば, 低精度モデルの Multi-GPU 並列計算を実現できる. 統合は今後の課題とする.

**低精度学習.** LADDER の設計は推論に限らない. 低精度モデルの学習と推論はともにシステムとハードウェアの対応を必要とし, 学習の逆伝播も順伝播に似ている. 低精度学習には二つの利点がある. 第一に高効率な低精度計算ユニットを使える. A100 の $W_{\mathrm{INT8}}A_{\mathrm{INT8}}$ tensor core は $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$ の 2×, $W_{\mathrm{INT4}}A_{\mathrm{INT4}}$ は 4× のスループットを持つ. 第二に低精度表現でメモリを減らし, より大きなバッチによってハードウェア利用率を高められる. 低精度学習は今後の課題とする.

<span id="section-7"></span>

## 7 関連研究

**深層学習コンパイラとフレームワーク.** 既存コンパイラ [Ans24, Che18, Ma20, Pas19, Shi23a, Zha23h, Zhe20, Zhe23d, Zhu22] の大半は FP16/FP32 など主流型の演算子・モデル計算を最適化し, 低精度型にはあまり注目しない. ただし多くの最適化は低精度計算と相補的である. LADDER は Roller [Zhu22] で効率的な tTile 構成を推論し, Welder [Shi23a] で端から端までのグラフ最適化を行う. SparTA [Zhe22e] は pruning と量子化をモデルの疎性として扱い, 疎モデルの推論と学習を一体に最適化する. LADDER の低精度カーネルでさらに性能を高められる. AMOS [Zhe22d] は FP16/INT8 の TensorCore 計算を最適化するが NVIDIA GPU 専用である. LADDER は異なる GPU 上の汎用カスタム型を支援し, 一般的な低精度計算を最適化する最初のコンパイラである. ONNXRuntime [Onn24] や TensorRT [Ten24] は推論用の一部低ビット演算子へ対応するが, 組合せごとの実装コストが大きく範囲は限られる. Triton [Til19] や TensorIR [Fen23] は DNN 演算子の計算パイプラインを直接記述でき, 各段階のスケジュールを柔軟に指定できる. しかし主眼は計算スケジューリングであり, LADDER が重視するカスタム型のデータスケジューリングにはほとんど対応しない.

**モデル固有の低精度最適化.** 既存コンパイラやフレームワークの対応不足から, ワークロード固有の最適化が多数提案されている. LLM 向け低精度量子化・学習 [Fra22, Kwo23, Lin23d, Ma24, Dar23, Tou23a, Wan23, Les23] や, ShuffleNet, Conformer などを FP8/FP16 へ最適化する研究 [Gul20, He16, Hua19d, Ma18, She23f] がある. LADDER はカスタム型と最適化方針を容易に実装する仕組みを提供するため, これらの手法とは相補的であり, LADDER 上へ実装するか自動最適化できる.

<span id="section-8"></span>

## 8 結論

本論文では GPU などのアクセラレータ上で汎用低精度計算を最適化する最初の深層学習コンパイラ LADDER を提案した. 汎用型システム tType と拡張テンソル式により, 深層学習の新しいデータ型を容易に実装・表現できる. 新しいテンソルスケジューリングプリミティブ群は, 計算パイプラインの格納, アクセス, 型変換の最適化を可能にする. 階層別ハードウェア認識方針は複雑な変換空間を探索し, 多様な低ビット精度カスタム型を体系的に支援する. これによりハードウェアを変更せず, 現代のアクセラレータ上の DNN 性能を高められる. モデル設計者はデータ型最適化を探究でき, ハードウェアベンダーは多様な精度形式への対応を柔軟に拡張できる.

## 謝辞

詳細な助言をくださった匿名査読者と匿名 shepherd に感謝する.


