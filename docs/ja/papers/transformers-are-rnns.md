---
title: 'Transformers are RNNs'
createTime: 2026/09/13 01:44:07
permalink: /ja/papers/transformers-are-rnns/
pageClass: paper-reading
---

> [Angelos Katharopoulos](https://angeloskath.github.io/)、[Apoorv Vyas](https://apoorv2904.github.io/)、[Nikolaos Pappas](https://nik0spapp.github.io/)、[François Fleuret](https://fleuret.org/francois/) [+affiliations]。2020 年 6 月 29 日に arXiv へ初投稿。現行版は v3 で、2020 年 8 月 31 日に改訂。*Proceedings of the 37th International Conference on Machine Learning*、PMLR 119:5156-5165、2020 年 7 月 13-18 日に掲載。[Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236)。<a href="/paper/transformers-are-rnns.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。[ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html)。[DOI](https://doi.org/10.48550/arXiv.2006.16236)。[TeX ソース](https://export.arxiv.org/e-print/2006.16236v3)。厳密な印刷レイアウトと参考文献については、原 PDF を正とする。

## 概要

Transformer はいくつかのタスクで顕著な性能を達成しているが、入力長に対する計算量が二次であるため、非常に長い系列では実用にならないほど遅い。この制約に対処するため、自己注意をカーネル特徴写像の線形な内積として表し、行列積の結合則を利用して、系列長を $N$ としたときの計算量を $\mathcal{O}\left(N^{2}\right)$ から $\mathcal{O}\left(N\right)$ へ削減する。この定式化によって、自己回帰 Transformer を大幅に高速化する反復実装が可能になり、再帰型ニューラルネットワークとの関係も明らかになることを示す。提案する *linear transformer* は通常の Transformer と同程度の性能を達成し、非常に長い系列の自己回帰予測では最大 4000 倍高速である。

<span id="section-1"></span>

## 1 はじめに

Transformer モデルは、もともと [Vas17] によってニューラル機械翻訳 [Sut14, Bah14] の文脈で導入され、自然言語 [Dev18]、音声 [Spe18]、画像 [Par19b] を扱う多様なタスクで優れた結果を示してきた。十分な教師情報があるタスクだけでなく、自己回帰 [Rad18, Rad19] またはマスク言語モデリングの目的関数 [Dev18, Yan20d, Son19, Liu19a] で事前学習した Transformer は、教師情報が限られる、あるいは存在しないタスクへの知識転移にも有効である。

しかし、こうした利点には非常に大きな計算コストとメモリコストが伴うことが多い。主なボトルネックは自己注意の大域的な受容野にあり、$N$ 個の入力からなるコンテキストの処理には、メモリと時間の両方で二次の計算量 $\mathcal{O}\left(N^{2}\right)$ が必要になる。その結果、実際には Transformer の学習は遅く、コンテキストも*制限される*。これは時間的な一貫性を損ない、長期依存関係の捕捉を妨げる。[Dai19] は以前のコンテキストのメモリへ注意を向けることで後者に対処したが、計算効率は犠牲になった。

近年、研究者は効率を損なわずにコンテキスト長を延ばす手法へ関心を移している。この目的のため、[Chi19] は注意行列の疎な因子分解を導入し、自己注意の計算量を $\mathcal{O}\left(N\sqrt{N}\right)$ へ削減した。[Kit20] は局所性鋭敏型ハッシュを用いて、計算量をさらに $\mathcal{O}\left(N\log N\right)$ まで削減した。これにより、長い系列へのスケーリングが可能になった。これらのモデルは大規模な系列で効率よく学習できるものの、自己回帰推論は高速化しない。

本論文では、メモリ使用量を大幅に削減し、コンテキスト長に対して線形にスケールする *linear transformer* モデルを導入する。これは、自己注意をカーネルに基づいて定式化し、行列積の結合則を使って自己注意の重みを計算することで実現する（[第 3.2 節](#section-3-2)）。この線形な定式化を用いると、因果マスクも線形時間かつ定数メモリで表せる（[第 3.3 節](#section-3-3)）。これによって Transformer と RNN の関係が明らかになり、自己回帰推論を桁違いに高速化できる（[第 3.4 節](#section-3-4)）。

画像生成と自動音声認識による評価から、*linear transformer* は Transformer と同等の性能水準に到達しながら、推論時には最大 3 桁高速であることを示す。

<span id="section-2"></span>

## 2 関連研究

本節では、Transformer の大きなメモリ要件と計算要件への対処を目指す、特に関連の深い研究を概観する。さらに、Transformer モデルの中核要素である自己注意を理論的に分析する手法を論じる。最後に、注意計算における softmax のボトルネックを緩和しようとする別系統の研究を紹介する。

<span id="section-2-1"></span>

### 2.1 効率的な Transformer

既存研究では、重み枝刈り [Mic19]、重み因子分解 [Lan20]、重み量子化 [Zaf19]、知識蒸留によって Transformer のメモリ効率を改善しようとしている。[Cla20] は、サンプル効率に優れ、総計算量を削減する replaced token detection という新しい事前学習目的を提案した。[Lam19] は product-key attention を用い、無視できるほど小さな計算オーバーヘッドで任意の層の容量を増やした。

これらの手法でメモリ要件や計算要件を減らすと学習時間または推論時間は短縮されるが、本質的に時間計算量は系列長に対して二次のままであり、長い系列へのスケーリングを妨げる。これに対し、提案手法は Transformer のメモリ計算量と時間計算量を、理論的にも（[第 3.2 節](#section-3-2)）、実験的にも（[第 4.1 節](#section-4-1)）削減することを示す。

別の研究系統は、Transformer における自己注意の「コンテキスト」を拡大することを目指している。コンテキストとは、自己注意の計算に使われる系列中の最大範囲を指す。[Dai19] は Transformer-XL を導入し、時間的な一貫性を損なうことなく固定長コンテキストを越える依存関係を学習することで、言語モデリングにおける最高水準の性能を達成した。しかし、以前のコンテキストをメモリに保持すると、大きな追加計算コストが生じる。一方、[Suk19] は、メモリ使用量と計算時間を制御しながら、注意ヘッドごとに最適な注意範囲を学習することで、コンテキスト長を大幅に拡大した。どちらの手法も通常のモデルと同じ漸近計算量を持つことに注意されたい。これに対し、提案手法は自己注意の漸近計算量を改善するため、はるかに大きなコンテキストを利用できる。

提案モデルとより密接に関係するのは [Chi19] と [Kit20] の研究である。前者の [Chi19] は注意行列の疎な因子分解を導入し、長い系列の生成モデリングにおける全体の計算量を二次から $\mathcal{O}\left(N\sqrt{N}\right)$ へ削減した。さらに最近、[Kit20] は Reformer を提案した。この手法は局所性鋭敏型ハッシュ（LSH）で内積の回数を減らし、計算量を $\mathcal{O}\left(N\log{N}\right)$ まで削減する。LSH を利用するため、Reformer は注意のキーをクエリと同一に制約している点に注意されたい。そのため、キーをクエリとは異なるものにする必要がある復号タスクには使えない。これに対し、*linear transformer* はクエリとキーに制約を課さず、系列長に対して線形にスケールする。さらに、検証 perplexity で同等の性能を達成しながら、自己回帰タスクの推論を 3 桁高速に実行できる。

<span id="section-2-2"></span>

### 2.2 自己注意の理解

理論的な観点から自己注意をより深く理解しようとする試みは少ない。[Tsa19] は Transformer の注意をカーネルに基づいて定式化し、入力間の類似度をカーネルスコアとして、入力にカーネル平滑化を適用するものと捉えた。この定式化は、注意の構成要素を理解し、位置埋め込みを統合するためのよりよい方法を与える。これに対し、提案手法はカーネルによる定式化を用いて自己注意の計算を高速化し、その計算量を削減する。また、類似度スコアが正となるカーネルをクエリとキーに適用すれば、線形注意が通常どおり収束することを観察した。

さらに最近、[Cor20] は、十分な数のヘッドを持つマルチヘッド自己注意が任意の畳み込み層を表現できることを、理論的証明と実験的証拠によって示した。一方、本研究では、自己回帰目的で学習した自己注意層を再帰型ニューラルネットワークと見なせることを示し、この観察を用いて自己回帰 Transformer モデルの推論時間を大幅に短縮できることを示す。

<span id="section-2-3"></span>

### 2.3 線形化した softmax

長年にわたり、softmax は多数のカテゴリを持つ分類モデルを学習する際のボトルネックであった [Goo01, Mor05, Mni09]。近年の研究 [Bla17, Raw19] は、特徴写像の線形な内積で softmax を近似し、サンプリングを通じて学習を高速化した。これらの研究に着想を得て、本研究では Transformer の softmax attention を線形化する。本研究と並行して、[She21] は画像の物体検出タスクに線形化注意を用いることを検討した。これに対し、本研究では注意計算を線形化するだけでなく、推論と学習の双方で線形時間かつ定数メモリとなる自己回帰 Transformer モデルを開発する。さらに、カーネルの観点から見れば、あらゆる Transformer を再帰型ニューラルネットワークと捉えられることを示す。

<span id="section-3"></span>

## 3 Linear Transformer

本節では、提案する *linear transformer* を形式化する。注意を従来の *softmax* attention から特徴写像に基づく内積注意へ変更すると、時間計算量とメモリ計算量が改善され、再帰型ニューラルネットワークと同様に線形時間で系列生成を行える因果モデルが得られることを示す。

まず [第 3.1 節](#section-3-1) で、[Vas17] が導入した Transformer アーキテクチャの定式化を示す。続いて [第 3.2 節](#section-3-2) と [第 3.3 節](#section-3-3) で提案する *linear transformer* を示し、最後に [第 3.4 節](#section-3-4) で Transformer を再帰型ニューラルネットワークとして書き直す。

<span id="section-3-1"></span>

### 3.1 Transformer

$x\in\mathbb{R}^{N\times F}$ を、次元 $F$ の特徴ベクトル $N$ 個からなる系列とする。Transformer は、次のように $L$ 個の Transformer 層 $T_{1}(\cdot),\dots,T_{L}(\cdot)$ の合成で定義される関数 $T:\mathbb{R}^{N\times F}\to\mathbb{R}^{N\times F}$ である。

<span id="equation-01"></span>

$$
T_{l}(x)=f_{l}(A_{l}(x)+x).
$$

関数 $f_{l}(\cdot)$ は各特徴をほかの特徴とは独立に変換し、通常は小規模な 2 層フィードフォワードネットワークで実装される。$A_{l}(\cdot)$ は自己注意関数であり、Transformer のうち系列をまたいで作用する唯一の部分である。

自己注意関数 $A_{l}(\cdot)$ は各位置について、ほかのすべての位置の特徴表現を、表現間の類似度スコアに比例する重みで加重平均する。形式的には、入力系列 $x$ を 3 つの行列 $W_{Q}\in\mathbb{R}^{F\times D}$、$W_{K}\in\mathbb{R}^{F\times D}$、$W_{V}\in\mathbb{R}^{F\times M}$ によって、それぞれ対応する表現 $Q$、$K$、$V$ へ射影する。全位置に対する出力 $A_{l}(x)=V^{\prime}$ は次のように計算される。

<span id="equation-02"></span>

$$
\begin{aligned}
Q & =xW_{Q}, \\
K & =xW_{K}, \\
V & =xW_{V}, \\
A_{l}(x)=V^{\prime} & =\mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right)V.
\end{aligned}
$$

前式では softmax 関数を $Q K^\top$ の各行に適用する。一般的な用語に従い、$Q$、$K$、$V$ をそれぞれ「クエリ」「キー」「値」と呼ぶ。

[式 2](#equation-02) は softmax attention と呼ばれる特定の自己注意を実装しており、類似度スコアはクエリとキーの内積の指数である。行列を $i$ で添字付けすると $i$ 行目がベクトルとして返ることから、任意の類似度関数に対する一般化注意の式を次のように書ける。

<span id="equation-03"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{N}\mathrm{sim}\left(Q_{i},K_{j}\right)V_{j}}{\sum_{j=1}^{N}\mathrm{sim}\left(Q_{i},K_{j}\right)}.
$$

類似度関数を $\mathrm{sim}\left(q,k\right)=\exp\left(\frac{q^\top k}{\sqrt{D}}\right)$ に置き換えると、[式 3](#equation-03) は [式 2](#equation-02) と等価になる。

<span id="section-3-2"></span>

### 3.2 線形化注意

[式 2](#equation-02) の注意の定義は一般的であり、多項式注意や RBF カーネル注意 [Tsa19] など、ほかのさまざまな注意実装の定義にも使える。[式 3](#equation-03) が注意関数を定義するために $\mathrm{sim}\left(\cdot\right)$ へ課す必要がある唯一の制約は、非負であることである。これには、すべてのカーネル $k(x,y):\mathbb{R}^{2\times F}\to\mathbb{R}_{+}$ が含まれる。

このようなカーネルが特徴表現 $\phi\left(x\right)$ を持つとき、[式 2](#equation-02) は次のように書き直せる。

<span id="equation-04"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{N}\phi\left(Q_{i}\right)^\top\phi\left(K_{j}\right)V_{j}}{\sum_{j=1}^{N}\phi\left(Q_{i}\right)^\top\phi\left(K_{j}\right)},
$$

さらに、行列積の結合則を使って次のように簡略化できる。

<span id="equation-05"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{N}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{N}\phi\left(K_{j}\right)}.
$$

分子を次のようなベクトル化形式で書くと、上式は理解しやすい。

<span id="equation-06"></span>

$$
\left(\phi\left(Q\right)\phi\left(K\right)^\top\right)V=\phi\left(Q\right)\left(\phi\left(K\right)^\top V\right).
$$

特徴写像 $\phi\left(\cdot\right)$ は行列 $Q$ と $K$ の各行に適用される。

[式 2](#equation-02) から、softmax attention の計算コストは、系列長を $N$ とすると $\mathcal{O}\left(N^{2}\right)$ でスケールすることが明らかである。クエリ、キー、値に関する勾配を計算するには注意行列全体を保存する必要があるため、メモリ要件も同様である。これに対し、[式 5](#equation-05) の提案する *linear transformer* は、$\sum_{j=1}^{N}\phi\left(K_{j}\right)V_{j}^\top$ と $\sum_{j=1}^{N}\phi\left(K_{j}\right)$ を一度計算して各クエリに再利用できるため、時間計算量とメモリ計算量はいずれも $\mathcal{O}\left(N\right)$ である。

<span id="section-3-2-1"></span>

#### 3.2.1 特徴写像と計算コスト

softmax attention では、乗算と加算に要する総コストは $\mathcal{O}\left(N^{2}\max\left(D,M\right)\right)$ でスケールする。ここで $D$ はクエリとキーの次元、$M$ は値の次元である。これに対し、線形注意では、まず次元 $C$ の特徴写像を計算する。その後、新しい値の計算には $\mathcal{O}\left(N C M\right)$ 回の加算と乗算が必要になる。

以上の分析では、カーネルと特徴関数の選択を考慮していない。指数カーネルに対応する特徴関数は無限次元であるため、厳密な softmax attention の線形化は実行不可能である。一方、たとえば多項式カーネルは厳密な有限次元特徴写像を持ち、指数カーネルや RBF カーネルと同程度に機能することが示されている [Tsa19]。2 次の線形化多項式 Transformer の計算コストは $\mathcal{O}\left(N D^{2} M\right)$ である。このため、$N>D^{2}$ のときに計算量は有利になる。実際に数万要素からなる系列を処理できることが望まれるため、この条件は実用上成り立つ。

より短い系列を扱う本研究の実験では、次のように定義される正の類似度関数を生じる特徴写像を用いる。

<span id="equation-07"></span>

$$
\phi\left(x\right)=\mathrm{elu}(x)+1,
$$

ここで $\mathrm{elu}(\cdot)$ は指数線形ユニット [Cle16] の活性化関数を表す。$x$ が負のときに勾配を 0 にしないため、$\mathrm{relu}(\cdot)$ よりも $\mathrm{elu}(\cdot)$ を用いる。この特徴写像による注意関数には、$\mathcal{O}\left(N D M\right)$ 回の乗算と加算が必要になる。実験節では、[式 7](#equation-07) の特徴写像が、計算要件とメモリ要件を大幅に削減しながら完全な Transformer と同等に機能することを示す。

<span id="section-3-3"></span>

### 3.3 因果マスク

Transformer アーキテクチャでは、位置 $i$ が位置 $j$ から影響を受けるのを $j\leq i$ の場合に限る、すなわち後続位置から影響を受けないよう注意計算をマスクすることで、自己回帰モデルを効率よく学習できる。形式的には、この因果マスクによって [式 3](#equation-03) は次のように変わる。

<span id="equation-08"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{i}\mathrm{sim}\left(Q_{i},K_{j}\right)V_{j}}{\sum_{j=1}^{i}\mathrm{sim}\left(Q_{i},K_{j}\right)}.
$$

[第 3.2 節](#section-3-2) の議論に従い、マスク付き注意を次のように線形化する。

<span id="equation-09"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)}.
$$

$S_{i}$ と $Z_{i}$ を次のように導入する。

<span id="equation-10"></span>

$$
S_{i}=\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top,
$$

<span id="equation-11"></span>

$$
Z_{i}=\sum_{j=1}^{i}\phi\left(K_{j}\right),
$$

これにより、[式 9](#equation-09) を次のように簡略化できる。

<span id="equation-12"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top S_{i}}{\phi\left(Q_{i}\right)^\top Z_{i}}.
$$

$S_{i}$ と $Z_{i}$ は $S_{i-1}$ と $Z_{i-1}$ から定数時間で計算できるため、因果マスクを備えた linear transformer の計算量は系列長に対して線形になる。

<span id="section-3-3-1"></span>

#### 3.3.1 勾配計算

任意の深層学習フレームワークで [式 12](#equation-12) を素朴に実装すると、勾配計算のためにすべての中間値 $S_{i}$ を保存する必要がある。これによりメモリ消費量は $\max\left(D,M\right)$ 倍に増え、因果線形注意をより長い系列やより深いモデルへ適用しにくくなる。そこで、[式 9](#equation-09) の分子の勾配を累積和として導出する。これにより、因果線形注意の順伝播と逆伝播をともに**線形時間**かつ**定数メモリ**で計算できる。詳しい導出は補足資料に示す。

分子 $\bar{V}_{i}$ と、分子に関するスカラー損失関数の勾配 $\nabla_{\bar{V}_{i}}\mathcal{L}$ が与えられたとき、$\nabla_{\phi\left(Q_{i}\right)}\mathcal{L}$、$\nabla_{\phi\left(K_{i}\right)}\mathcal{L}$、$\nabla_{V_{i}}\mathcal{L}$ を次のように導出する。

<span id="equation-13"></span>

$$
\nabla_{\phi\left(Q_{i}\right)}\mathcal{L}=\nabla_{\bar{V}_{i}}\mathcal{L}\left(\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top\right)^\top,
$$

<span id="equation-14"></span>

$$
\nabla_{\phi\left(K_{i}\right)}\mathcal{L}=\left(\sum_{j=i}^{N}\phi\left(Q_{j}\right)\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)V_{i},
$$

<span id="equation-15"></span>

$$
\nabla_{V_{i}}\mathcal{L}=\left(\sum_{j=i}^{N}\phi\left(Q_{j}\right)\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)^\top\phi\left(K_{i}\right).
$$

[式 9](#equation-09) と [式 13](#equation-13)-[15](#equation-15) に含まれる累積和の項は、系列長に対して線形時間かつ定数メモリで計算される。その結果、$C$ 次元の特徴写像に対し、計算量 $\mathcal{O}\left(N C M\right)$、メモリ計算量 $\mathcal{O}\left(N\max\left(C,M\right)\right)$ のアルゴリズムが得られる。分子の順伝播と逆伝播の擬似コード実装を [アルゴリズム 1](#algorithm-01) に示す。

<span id="section-3-3-2"></span>

#### 3.3.2 学習と推論

自己回帰 Transformer モデルの学習時には、正解系列全体を利用できる。このため、[式 1](#equation-01) の $f_{l}(\cdot)$ と注意計算の双方で、層ごとの並列化が可能である。その結果、Transformer は再帰型ニューラルネットワークよりも効率よく学習できる。一方、推論時には時刻 $i$ の出力が時刻 $i+1$ の入力になる。このため、自己回帰モデルは並列化できない。さらに、Transformer の時刻ごとのコストは一定ではなく、過去のすべての時刻について注意を計算する必要があるため、現在の系列長の二乗に比例して増加する。

提案する *linear transformer* モデルは*両者の長所を兼ね備える*。学習時には計算を並列化でき、GPU やその他のアクセラレータを十分に活用できる。推論時には、1 回の予測に要する時間とメモリのコストが一定である。つまり、$\phi\left(K_{j}\right)V_{j}^\top$ 行列を内部状態として保存し、再帰型ニューラルネットワークのように各時刻で更新するだけでよい。その結果、ほかの Transformer モデルより**数千倍高速**に推論できる。

<span id="section-3-4"></span>

### 3.4 Transformer は RNN である

文献では、Transformer モデルは再帰型ニューラルネットワークとは根本的に異なる手法と考えられている。しかし、[第 3.3 節](#section-3-3) の因果マスクの定式化と前節の議論から、因果マスクを持つ任意の Transformer 層は、入力を受け取って内部状態を変更し、その後に出力を予測するモデル、すなわち再帰型ニューラルネットワーク（RNN）として書けることが明らかになる。Universal Transformer [Deh18] とは異なり、ここでは深さではなく時間に関する再帰を考える。

以下の式では、[式 1](#equation-01) の Transformer 層を再帰型ニューラルネットワークとして形式化する。得られる RNN は、注意メモリ $s$ と正規化メモリ $z$ という 2 つの隠れ状態を持つ。添字は再帰における時刻を表す。

<span id="equation-16"></span>

$$
s_{0}=0,
$$

<span id="equation-17"></span>

$$
z_{0}=0,
$$

<span id="equation-18"></span>

$$
s_{i}=s_{i-1}+\phi\left(x_{i}W_{K}\right)\left(x_{i}W_{V}\right)^\top,
$$

<span id="equation-19"></span>

$$
z_{i}=z_{i-1}+\phi\left(x_{i}W_{K}\right),
$$

<span id="equation-20"></span>

$$
y_{i}=f_{l}\left(\frac{\phi\left(x_{i}W_{Q}\right)^\top s_{i}}{\phi\left(x_{i}W_{Q}\right)^\top z_{i}}+x_{i}\right).
$$

上式で、$x_{i}$ は特定の Transformer 層に対する $i$ 番目の入力、$y_{i}$ は $i$ 番目の出力を表す。この定式化は特徴関数に制約を課さず、理論上は softmax attention を使うものを含む*あらゆる Transformer* モデルを表現できる。この定式化は、Transformer と一般的な再帰型ネットワーク [Hoc97] の関係、および情報の保存と検索に使われる過程をよりよく理解するための第一歩である。

<span id="section-4"></span>

## 4 実験

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**アルゴリズム 1：因果マスクを備えた linear transformer。**

- **関数** $\mathrm{forward}(\phi(Q),\phi(K),V)$：
  - $V'\gets0$、$S\gets0$ と**設定**する。
  - $i=1,\dots,N$ の**各**値について：
    - $S\gets S+\phi(K_i)V_i^\top$ と**設定**する（[式 10](#equation-10)）。
    - $\bar V_i\gets\phi(Q_i)S$ と**設定**する。
  - $\bar V$ を**返す**。
- **関数** $\mathrm{backward}(\phi(Q),\phi(K),V,G)$：
  - $G$ は $\mathrm{forward}$ の出力に関する損失の勾配である。
  - $S\gets0$、$\nabla_{\phi(Q)}\mathcal L\gets0$ と**設定**する。
  - $i=1,\dots,N$ の**各**値について：
    - $S\gets S+\phi(K_i)V_i^\top$ と**設定**する。
    - $\nabla_{\phi(Q_i)}\mathcal L\gets G_iS^\top$ と**設定**する（[式 13](#equation-13)）。
  - $S\gets0$、$\nabla_{\phi(K)}\mathcal L\gets0$、$\nabla_V\mathcal L\gets0$ と**設定**する。
  - $i=N,\dots,1$ の**各**値について：
    - $S\gets S+\phi(Q_i)G_i^\top$ と**設定**する。
    - $\nabla_{V_i}\mathcal L\gets S^\top\phi(K_i)$ と**設定**する（[式 15](#equation-15)）。
    - $\nabla_{\phi(K_i)}\mathcal L\gets S V_i$ と**設定**する（[式 14](#equation-14)）。
  - $\nabla_{\phi(Q)}\mathcal L$、$\nabla_{\phi(K)}\mathcal L$、$\nabla_V\mathcal L$ を**返す**。

</div>

本節では、提案する *linear transformer* の性能を実験的に分析する。まず [第 4.1 節](#section-4-1) で、線形化注意を計算コスト、メモリ消費量、合成データ上での収束の観点から評価する。*linear transformer* の有効性をさらに示すため、[第 4.2 節](#section-4-2) の画像生成と [第 4.3 節](#section-4-3) の自動音声認識という 2 つの実世界の応用で提案モデルを評価する。提案モデルが、GPU メモリと計算量を大幅に抑えながら、最先端の Transformer アーキテクチャに匹敵する性能を達成することを示す。

すべての実験で、提案モデルを 2 つのベースライン、すなわち softmax attention を持つ完全な Transformer と Reformer [Kit20] と比較する。後者は最先端の高速 Transformer アーキテクチャである。Reformer には公開コードの PyTorch 再実装を使い、完全な Transformer には PyTorch の既定実装を使う。Reformer では可逆層を使わないが、自己注意層に関するメモリ消費量のみを測定するため、結果には影響しない。すべての実験で、標準的な Transformer アーキテクチャを **softmax** [Vas17]、提案する *linear transformer* を **linear**、Reformer [Kit20] を **lsh-X** と表記し、*X* はハッシュラウンド数を表す。

*linear transformer* の学習には [式 7](#equation-07) の特徴写像を用いる。ドキュメントと例を含む PyTorch [Pas19] コードは [https://linear-transformers.com/](https://linear-transformers.com/) で公開している。[式 13](#equation-13)-[15](#equation-15) の定数メモリ勾配計算は、およそ 200 行の CUDA コードで実装されている。

<span id="figure-01"></span>

![図 1。Reformer（lsh-X）、softmax attention、linear attention の順伝播／逆伝播に必要な計算資源の比較。linear モデルと Reformer モデルは系列長に対して線形にスケールするが、softmax はメモリと時間の双方で系列長の二乗に比例する。実験の詳細は[第 4.1 節](#section-4-1)を参照。](../../papers/transformers-are-rnns/figure-01.png)

**図 1。** Reformer（lsh-X）、softmax attention、linear attention の順伝播／逆伝播に必要な計算資源の比較。linear モデルと Reformer モデルは系列長に対して線形にスケールするが、softmax はメモリと時間の双方で系列長の二乗に比例する。実験の詳細は [第 4.1 節](#section-4-1) を参照。

<span id="figure-02"></span>

![図 2。系列複製タスクにおける *softmax*、*linear*、*reformer* attention の収束比較。*linear* は安定して収束し、softmax と同じ最終性能に達する。実験の詳細は[第 4.1 節](#section-4-1)を参照。](../../papers/transformers-are-rnns/figure-02.png)

**図 2。** 系列複製タスクにおける *softmax*、*linear*、*reformer* attention の収束比較。*linear* は安定して収束し、softmax と同じ最終性能に達する。実験の詳細は [第 4.1 節](#section-4-1) を参照。

<span id="section-4-1"></span>

### 4.1 合成タスク

<span id="section-4-1-1"></span>

#### 4.1.1 収束解析

*linear transformer* の収束特性を調べるため、因果マスクを用いた人工的なコピータスクで学習する。すなわち、Transformer は [Kit20] の系列複製タスクと同様に、一連の記号を複製しなければならない。専用の区切り記号で分けられた 10 種類の記号からなる、最大長 128 の系列を用いる。3 手法すべてについて、注意ヘッドを 8 個備えた 4 層 Transformer を、バッチサイズ 64、学習率 $10^{-3}$ の RAdam optimizer [Liu19d] で学習し、3000 回の更新後に学習率を $10^{-4}$ へ下げる。[図 2](#figure-02) は勾配ステップ数に対する損失を示す。linear は滑らかに収束し、ハッシュによるノイズがないため lsh より低い損失に達する。具体的には、softmax と同じ損失に達する。

<span id="section-4-1-2"></span>

#### 4.1.2 メモリ要件と計算要件

本項では、Transformer の計算要件とメモリ要件を比較する。系列長を $N\in\{2^{9},2^{10},\dots,2^{16}\}$ と変化させた合成入力について注意と勾配を計算し、Transformer の各変種で割り当てられた GPU メモリのピーク値と所要時間を測定する。バッチサイズは系列長に反比例させ、バッチ内のサンプルあたりの時間とメモリを報告する。

各手法は、GPU メモリに収まる最大系列長まで評価する。このベンチマークにはメモリ 11 GB の NVidia GTX 1080 Ti を用いる。その結果、最大系列長は softmax で 4,096 要素、lsh-4 と lsh-8 で 16,384 要素となる。予想どおり、softmax は系列長に対して二次でスケールする。[図 1](#figure-01) に示すように、すべての構成で提案手法はベースラインより高速で、必要なメモリも少ない。Reformer と linear attention はどちらも系列長に対して線形にスケールする。Reformer の漸近計算量は $\mathcal{O}\left(N\log N\right)$ だが、$\log N$ は十分に小さく、計算時間には影響しない。

<span id="section-4-2"></span>

### 4.2 画像生成

Transformer は条件付き・無条件の自己回帰生成タスクで優れた結果を示してきた [Rad19, Chi19]。しかし、タスクが本質的に逐次的であり、メモリが系列長の二乗に比例するため、Transformer からのサンプリングは遅い。本節では、因果マスクを備えた Transformer を学習し、画像をピクセル単位で予測する。提案手法は、次元あたりのビット数で *softmax* attention と同等の性能を達成しながら、最初のピクセルから最後のピクセルまで**画像あたり一定のメモリ**で、**1000 倍を超える速度**で画像を生成できる。学習の推移、生成画像の品質、1 枚の画像を生成する時間の比較については補足資料を参照されたい。さらに、PyTorch 実装とは異なり、推論中にキーと値をキャッシュする、より高速な softmax Transformer とも比較する。

<span id="section-4-2-1"></span>

#### 4.2.1 MNIST

<span id="table-01"></span>

![表 1。MNIST 画像の自己回帰生成の比較。linear transformer は完全な softmax attention とほぼ同じ bits/dim を達成しながら、画像生成のスループットは 300 倍を超える。実験の詳細は[第 4.2.1 節](#section-4-2-1)を参照。](../../papers/transformers-are-rnns/table-01.png)

**表 1。** MNIST 画像の自己回帰生成の比較。linear transformer は完全な softmax attention とほぼ同じ bits/dim を達成しながら、画像生成のスループットは 300 倍を超える。実験の詳細は [第 4.2.1 節](#section-4-2-1) を参照。

まず、広く使われている MNIST データセット [Lec00] を用いて、自己回帰 Transformer による画像生成で提案モデルを評価する。この実験のアーキテクチャは、各 8 個の注意ヘッドを持つ 8 層の注意層からなる。埋め込みサイズは 256、すなわちヘッドあたり 32 次元に設定する。フィードフォワードの次元は埋め込みサイズの 4 倍である。出力は [Sal17] が導入した 10 個の logistic 分布の混合でモデル化する。学習率 $10^{-4}$ の RAdam optimizer を用い、すべてのモデルを 250 epoch 学習する。Reformer ベースラインでは、ハッシュラウンド数を 1 と 4 にする。さらに [Kit20] の提案に従い、64 個の bucket と、約 32 要素からなる chunk を用いる。具体的には、長さ 783 の入力系列を、それぞれ 29 要素からなる 27 個の chunk に分割する。系列長は比較的短く、わずか 784 ピクセルであるため、バッチサイズの違いによる差を取り除く目的で、すべての手法にバッチサイズ 10 を用いる。

[表 1](#table-01) に結果をまとめる。linear transformer は、最終 perplexity に関して softmax Transformer とほぼ同じ性能を達成しながら、300 倍を超える速度で画像を生成できる。これは提案モデルのメモリ要件が小さいためであり、1 基の GPU で 10,000 枚の MNIST 画像を同時に生成できる。特に、ピクセル間で保存する必要があるのは [式 18](#equation-18) と [式 19](#equation-19) で述べた $s_{i}$ と $z_{i}$ の値だけなので、メモリは系列長に対して一定である。一方、softmax と Reformer はどちらも系列長とともに増えるメモリを必要とする。

提案する MNIST モデルによる画像補完と無条件サンプルを [図 3](#figure-03) に示す。linear transformer は、輪郭が鮮明でノイズのない、非常に説得力のあるサンプルを生成する。画像補完の場合、Transformer は元画像と同じストロークのスタイルと幅を使うことも学習しており、長い時間距離にわたって効果的に注意を向けている。達成した perplexity はすべてのモデルでほぼ同じなので、異なるモデルが生成したサンプルの間に質的な違いは見られない。

<span id="section-4-2-2"></span>

#### 4.2.2 CIFAR-10

<span id="table-02"></span>

![表 2。CIFAR-10 画像を生成する自己回帰 Transformer を、1 基の GPU で 1 週間学習する。linear transformer は softmax の 3 倍の epoch を完了し、その結果、より良い perplexity を達成する。提案モデルはベースラインより $4{,}000\times$ 高速に画像を生成する。実験の詳細は[第 4.2.2 節](#section-4-2-2)を参照。](../../papers/transformers-are-rnns/table-02.png)

**表 2。** CIFAR-10 画像を生成する自己回帰 Transformer を、1 基の GPU で 1 週間学習する。linear transformer は softmax の 3 倍の epoch を完了し、その結果、より良い perplexity を達成する。提案モデルはベースラインより $4{,}000\times$ 高速に画像を生成する。実験の詳細は [第 4.2.2 節](#section-4-2-2) を参照。

線形な定式化の利点は、系列長が増えるほど大きくなる。それを示すため、CIFAR-10 画像 [Kri09] を生成する 16 層 Transformer を学習する。各層には前の実験と同じ構成を用いる。Reformer では、論文の提案に従い、再び 64 個の bucket と、それぞれ 37 要素からなる 83 個の chunk を用いる。これはおよそ 32 に相当する。系列長が前の実験のほぼ 4 倍なので、完全な Transformer は、利用できる最大の GPU、すなわちメモリ 24 GB の NVidia P40 でも、バッチサイズ 1 でしか使用できない。linear transformer と Reformer にはバッチサイズ 4 を用いる。すべてのモデルを 7 日間学習する。次元あたりのビット数と画像生成スループットに関する結果を [表 2](#table-02) に示す。この実験の主眼は最終 perplexity ではないものの、系列長が増えるにつれて高速な Transformer モデルは GPU 時間あたりの効率がますます高まり、遅いモデルよりも良いスコアを達成することが明らかである。

Reformer と softmax attention はどちらも、1 ピクセルの生成に必要なメモリと時間がピクセル数の二乗に比例するため、linear transformer のスループット向上はさらに顕著である。具体的には、softmax Transformer が**画像を 1 枚生成する**間に、**提案手法は 4,460 枚の画像を生成できる**。提案モデルによる画像補完と無条件サンプルを [図 4](#figure-04) に示す。提案モデルは空間的な一貫性を持つ画像を生成し、画像カテゴリの認識を大きく妨げずに説得力のある補完を行う。たとえば [図 4(b)](#figure-04) では、すべての画像が犬の鼻（1 行目）またはトラックのフロントガラス（最終行）をうまく補完している。

<span id="figure-03"></span>

![図 3。提案手法が MNIST について生成した無条件サンプルと画像補完。（a）は遮蔽された元画像、（b）は補完、（c）は元画像を示す。提案モデルは softmax と同等の bits/dimension を達成しながら、スループットは **300 倍以上**で、**毎秒 142 枚**を生成する。詳細は[第 4.2.1 節](#section-4-2-1)を参照。](../../papers/transformers-are-rnns/figure-03.png)

**図 3。** 提案手法が MNIST について生成した無条件サンプルと画像補完。（a）は遮蔽された元画像、（b）は補完、（c）は元画像を示す。提案モデルは softmax と同等の bits/dimension を達成しながら、スループットは **300 倍以上**で、**毎秒 142 枚**を生成する。詳細は [第 4.2.1 節](#section-4-2-1) を参照。

<span id="figure-04"></span>

![図 4。提案手法が CIFAR-10 について生成した無条件サンプルと画像補完。（a）は遮蔽された元画像、（b）は補完、（c）は元画像を示す。系列長が増えるほど、linear transformer は softmax attention より効率的になる。提案モデルは **4,000 倍以上**のスループットを達成し、**毎秒 17.85 枚**を生成する。詳細は[第 4.2.2 節](#section-4-2-2)を参照。](../../papers/transformers-are-rnns/figure-04.png)

**図 4。** 提案手法が CIFAR-10 について生成した無条件サンプルと画像補完。（a）は遮蔽された元画像、（b）は補完、（c）は元画像を示す。系列長が増えるほど、linear transformer は softmax attention より効率的になる。提案モデルは **4,000 倍以上**のスループットを達成し、**毎秒 17.85 枚**を生成する。詳細は [第 4.2.2 節](#section-4-2-2) を参照。

<span id="section-4-3"></span>

### 4.3 自動音声認識

<span id="table-03"></span>

![表 3。WSJ データセットの自動音声認識における性能比較。結果は音素誤り率（PER）と epoch あたりの学習時間で示す。提案モデルは LSTM と Reformer を上回り、学習と評価も高速である。実験の詳細は[第 4.3 節](#section-4-3)を参照。](../../papers/transformers-are-rnns/table-03.png)

**表 3。** WSJ データセットの自動音声認識における性能比較。結果は音素誤り率（PER）と epoch あたりの学習時間で示す。提案モデルは LSTM と Reformer を上回り、学習と評価も高速である。実験の詳細は [第 4.3 節](#section-4-3) を参照。

提案手法が非自己回帰タスクにも使えることを示すため、Connectionist Temporal Classification（CTC）loss [Gra06] を用いた end-to-end 自動音声認識で linear transformer の性能を評価する。この設定では、各入力フレームについて音素上の分布を非自己回帰的に予測する。時間差分を含まない 40 次元 mel-scale filterbank を特徴量として、80 時間の WSJ データセット [Pau92] を用いる。データセットには平均 800 フレーム、最大系列長 2,400 フレームの系列が含まれる。このタスクでは、隠れサイズ 320 の 3 層双方向 LSTM [Hoc97] とも比較する。学習率 $10^{-3}$ の Adam optimizer [Kin15] を用い、検証誤差の減少が止まったときに学習率を下げる。Transformer モデルには、画像実験と同じ埋め込み次元を持つ 6 ヘッド 9 層を用いる。optimizer には、初期学習率 $10^{-4}$ の RAdam を用い、検証誤差の減少が止まったときに学習率を 2 で割る。

すべてのモデルを、音素誤り率（PER）と epoch あたりの学習時間で評価する。[表 3](#table-03) に示すように、linear は性能と速度の両面で再帰型ネットワークのベースラインと Reformer を大幅に上回る。softmax Transformer はすべてのベースラインより低い音素誤り率を達成するが、著しく遅い。具体的には、*linear transformer* は epoch あたりで 3 倍を超えて高速である。学習推移のプロットは補足資料に示す。

<span id="section-5"></span>

## 5 結論

本研究では、従来の Transformer のメモリコストと計算コストを大幅に削減する *linear transformer* モデルを提示した。特に、行列積の結合則を利用することで、自己注意を系列長に対して線形にスケールする時間とメモリで計算できる。提案モデルは因果マスクとともに使用しても、線形な漸近計算量を保つことを示した。最後に、Transformer モデルを再帰型ニューラルネットワークとして表し、自己回帰タスクの推論を数千倍高速に実行できるようにした。

この性質は、RNN と Transformer の双方における情報の保存と検索に関して、将来の研究へ多くの方向性を開く。検討すべきもう一つの研究方向は、線形注意に用いる特徴写像の選択に関するものである。たとえば、ランダム Fourier 特徴で RBF カーネルを近似すれば、softmax attention で事前学習したモデルを利用できる可能性がある。

## 謝辞

Angelos Katharopoulos は、助成番号 FNS-30209「ISUL」および FNS-30224「CORTI」により Swiss National Science Foundation の支援を受けた。Apoorv Vyas は、助成番号 FNS-30213「SHISSM」により Swiss National Science Foundation の支援を受けた。Nikolaos Pappas は、助成番号 P400P2_183911「UNISON」により Swiss National Science Foundation の支援を受けた。

<span id="section-6"></span>

## 6 勾配の導出

補足資料の最初の節では、因果マスク付き linear transformer の勾配を詳しく導出し、線形時間かつ定数メモリで計算できることを示す。具体的には、次式の分子に関するスカラー損失の勾配を導出する。

<span id="equation-21"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)}.
$$

分母および分数に関する勾配は autograd によって効率よく処理される。一般性を失うことなく、$Q$ と $K$ はすでに $\phi\left(\cdot\right)$ で写像されたベクトルを含むと仮定できる。したがって、分子を

<span id="equation-22"></span>

$$
\bar{V}_{i}=Q_{i}^\top\sum_{j=1}^{i}K_{j}V_{j}^\top,
$$

とし、$\nabla_{\bar{V}}\mathcal{L}$ が与えられたとき、$\nabla_{Q}\mathcal{L}$、$\nabla_{K}\mathcal{L}$、$\nabla_{V}\mathcal{L}$ を計算する。ここで、$Q\in\mathbb{R}^{N\times D}$、$K\in\mathbb{R}^{N\times D}$、$V\in\mathbb{R}^{N\times M}$ である。勾配を導出するため、まず上式をベクトル表記を使わず単一要素について表す。

<span id="equation-23"></span>

$$
\bar{V}_{ie}=\sum_{d=1}^{D}Q_{id}\sum_{j=1}^{i}K_{jd}V_{je}=\sum_{d=1}^{D}\sum_{j=1}^{i}Q_{id}K_{jd}V_{je}.
$$

続いて、任意の $Q_{lt}$ による偏微分を次のように取り、$Q$ の勾配を導出する。

<span id="equation-24"></span>

$$
\frac{\partial\mathcal{L}}{\partial Q_{lt}}=\sum_{e=1}^{M}\frac{\partial\mathcal{L}}{\partial\bar{V}_{le}}\frac{\partial\bar{V}_{le}}{\partial Q_{lt}}=\sum_{e=1}^{M}\frac{\partial\mathcal{L}}{\partial\bar{V}_{le}}\left(\sum_{j=1}^{l}K_{jt}V_{je}\right).
$$

上式を勾配の行列積として書くと、次のようになる。

<span id="equation-25"></span>

$$
\nabla_{Q_{i}}\mathcal{L}=\nabla_{\bar{V}_{i}}\mathcal{L}\left(\sum_{j=1}^{i}K_{j}V_{j}^\top\right)^\top,
$$

これにより、本文の [式 13](#equation-13) が証明される。[式 24](#equation-24) では、$Q_{lt}$ が $\bar{V}_{l}$ にのみ影響するという事実を利用したため、勾配計算で $i$ について総和を取る必要はない。しかし、$K$ と $V$ にはこれは当てはまらない。特に、$K_{j}$ は $i\geq j$ となるすべての $\bar{V}_{i}$ に影響する。したがって、$K_{lt}$ に関する損失の偏微分を次のように書ける。

<span id="equation-26"></span>

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial K_{lt}} & =\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}\frac{\partial\bar{V}_{ie}}{\partial K_{lt}}=\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}\frac{\partial\left(\sum_{d=1}^{D}\sum_{j=1}^{i}Q_{id}K_{jd}V_{je}\right)}{\partial K_{lt}} \\
& =\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}Q_{it}V_{le}.
\end{aligned}
$$

$Q$ と同様に、勾配をベクトル化形式で書ける。

<span id="equation-27"></span>

$$
\nabla_{K_{i}}\mathcal{L}=\left(\sum_{j=i}^{N}Q_{j}\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)V_{i},
$$

これにより、本文の [式 14](#equation-14) が証明される。同じ議論に従えば、$V_{lt}$ に関する損失の偏微分を計算し、式 15 を証明できる。$Q$ と $K$ に関する勾配の累積和行列は同じ大きさだが、一方は順伝播と同様に順方向（1 から $N$ までの総和）で計算され、もう一方は RNN の通時的逆伝播と同様に逆方向（$N$ から 1 までの総和）で計算される。

<span id="section-7"></span>

## 7 学習の推移

[図 5](#figure-05) に、すべての実験における Transformer モデルの学習推移を示す。MNIST の実験（[図 5(a)](#figure-05)）では、すべての手法を 250 epoch 学習する。系列長が十分に短いため、学習時間は手法間で大きく変わらない。提案手法は softmax attention と同程度に収束し、2 つの Reformer 変種を大幅に上回る。

一方、CIFAR-10（[図 5(b)](#figure-05)）では、すべての手法を一定時間、すなわち 7 日間学習する。*lsh-1* と *linear* は softmax および lsh-4 よりも大幅に多くの epoch を完了し、より良い性能を達成する。系列長をさらに増やせば、この差は拡大すると予想される。

最後に、自動音声認識の実験（[図 5(c)](#figure-05)）では、収束に関して softmax が Reformer と linear の双方を大幅に上回る。linear は epoch あたり $3\times$ 高速であるため、softmax と比べて約 4 倍多くの epoch を完了している。softmax attention はこのタスクで優れているものの、*linear transformer* は収束と最終性能の双方で Reformer を大幅に上回る。

<span id="figure-05"></span>

![図 5。すべての実験における Transformer の学習推移。*linear transformer* は一貫して Reformer より速く収束し、自己回帰実験では softmax と同程度に収束する。MNIST ではすべての手法を 250 epoch 学習し、CIFAR では 7 日間学習する。音声認識実験では、すべての手法を収束するまで学習する。実験の詳細は、本文の [第 4.2.1 節](#section-4-2-1)、[第 4.2.2 節](#section-4-2-2)、[第 4.3 節](#section-4-3) を参照。](../../papers/transformers-are-rnns/figure-05.png)

**図 5。** すべての実験における Transformer の学習推移。*linear transformer* は一貫して Reformer より速く収束し、自己回帰実験では softmax と同程度に収束する。MNIST ではすべての手法を 250 epoch 学習し、CIFAR では 7 日間学習する。音声認識実験では、すべての手法を収束するまで学習する。実験の詳細は、本文の [第 4.2.1 節](#section-4-2-1)、[第 4.2.2 節](#section-4-2-2)、[第 4.3 節](#section-4-3) を参照。

<span id="section-8"></span>

## 8 画像生成スループットの考察

<span id="section-8-1"></span>

### 8.1 Stateful softmax attention

本文の [第 4.2 節](#section-4-2) では、画像生成スループットを報告し、**softmax** Transformer および **lsh** と比較する。本節では、**stateful-softmax** と表記する別のベースラインを作成する。これは softmax 自己回帰 Transformer を再帰モデルとして実装したものである。つまり、すべてのキーと値を保存し、系列の次の要素を予測するときに再びモデルへ渡す。この再帰モデルの状態はキーと値の集合であり、その大きさは系列長に比例する。これは、固定次元の状態を持ち、直前の状態から $i$ 番目の状態を計算するコストが $i$ によらず一定である提案モデルとは、質的に異なる。

<span id="table-04"></span>

![表 4。MNIST および CIFAR-10 画像の自己回帰生成スループットの比較。実験は本文の [第 4.2 節](#section-4-2) に示す。stateful-softmax ではキーと値を保存し、次の要素の予測に再利用する。この追加ベースラインの詳しい説明は[第 8.1 節](#section-8-1)を参照。](../../papers/transformers-are-rnns/table-04.png)

**表 4。** MNIST および CIFAR-10 画像の自己回帰生成スループットの比較。実験は本文の [第 4.2 節](#section-4-2) に示す。stateful-softmax ではキーと値を保存し、次の要素の予測に再利用する。この追加ベースラインの詳しい説明は [第 8.1 節](#section-8-1) を参照。

[表 4](#table-04) に結果をまとめる。stateful-softmax は通常の Transformer より大幅に高速である。しかし、計算量は依然として系列長に対して二次であり、CIFAR-10 では提案する定式化の方が $50\times$ を超えて高速である。さらに、新しい入力が与えられるたびに並べ替えと chunk 分割の操作を行う必要があるため、Reformer に同様の stateful attention を実装するのは容易ではない。

<span id="section-8-2"></span>

### 8.2 バッチサイズの統一

前節まで、自己回帰画像生成タスクについて Transformer の各変種のスループットを評価した。しかし、考慮すべきもう一つの重要な要因は latency、すなわち 1 枚の画像を生成するために必要な総時間である。このため、バッチサイズを 1 とし、各手法が 1 枚の画像を生成するのに要する時間を測定する。GPU での推論に加え、CPU で必要な時間も評価する。結果を [表 5](#table-05) に示す。

<span id="table-05"></span>

![表 5。自己回帰 Transformer を用いて MNIST および CIFAR-10 の画像を 1 枚生成するために必要な時間の比較。すべての手法を CPU と GPU の双方でバッチサイズ 1 に設定して実行し、総時間を秒単位で報告する。表のすべての数値は、小さいほど良い。](../../papers/transformers-are-rnns/table-05.png)

**表 5。** 自己回帰 Transformer を用いて MNIST および CIFAR-10 の画像を 1 枚生成するために必要な時間の比較。すべての手法を CPU と GPU の双方でバッチサイズ 1 に設定して実行し、総時間を秒単位で報告する。表のすべての数値は、小さいほど良い。

すべての手法で GPU が十分に活用されず、[表 4](#table-04) に示した値より画像生成スループットが大幅に低くなる。提案する linear transformer はすべての手法より高速であり、特に CIFAR-10 で画像を生成する場合、softmax Transformer よりほぼ $6.6\times$ 高速である。linear 自己回帰 Transformer は、すべての場合で GPU より CPU 上の方が高速となる唯一の手法である。これは、RNN として注意を計算するコストが非常に小さく、系列に対する避けられない外側のループが主な計算ボトルネックになるためである。

<span id="section-9"></span>

## 9 画像生成の定性的結果

本節では、画像生成実験の定性的結果を示す。すべてのモデルの perplexity は予想どおりほぼ同じなので、質的な差は大きくない。しかし興味深い観察として、Reformer モデルの無条件サンプルは変化が大幅に少ない。さらに、すべてのモデルが大幅に優れた性能を示すため、画像補完は無条件生成よりかなり容易なタスクであることがわかる。

<span id="figure-06"></span>

![図 6。MNIST で学習した Transformer モデルの無条件サンプル。本文の [第 4.2.1 節](#section-4-2-1) を参照。](../../papers/transformers-are-rnns/figure-06.png)

**図 6。** MNIST で学習した Transformer モデルの無条件サンプル。本文の [第 4.2.1 節](#section-4-2-1) を参照。

<span id="figure-07"></span>

![図 7。学習したすべてのモデルによる MNIST 数字画像の補完。本文の [第 4.2.1 節](#section-4-2-1) を参照。](../../papers/transformers-are-rnns/figure-07.png)

**図 7。** 学習したすべてのモデルによる MNIST 数字画像の補完。本文の [第 4.2.1 節](#section-4-2-1) を参照。

<span id="figure-08"></span>

![図 8。CIFAR-10 で学習した Transformer モデルの無条件サンプル。本文の [第 4.2.2 節](#section-4-2-2) を参照。](../../papers/transformers-are-rnns/figure-08.png)

**図 8。** CIFAR-10 で学習した Transformer モデルの無条件サンプル。本文の [第 4.2.2 節](#section-4-2-2) を参照。

<span id="figure-09"></span>

![図 9。学習したすべての Transformer モデルによる CIFAR-10 画像の補完。本文の [第 4.2.2 節](#section-4-2-2) を参照。](../../papers/transformers-are-rnns/figure-09.png)

**図 9。** 学習したすべての Transformer モデルによる CIFAR-10 画像の補完。本文の [第 4.2.2 節](#section-4-2-2) を参照。

[+affiliations]: 所属：Idiap Research Institute（スイス）、EPFL（スイス）、University of Washington（米国シアトル）、University of Geneva（スイス）。本研究は Idiap で実施された。連絡先：Angelos Katharopoulos <firstname.lastname@idiap.ch>。
