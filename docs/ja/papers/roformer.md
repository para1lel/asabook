---
title: 'RoFormer: Rotary Position Embedding'
createTime: 2026/09/05 04:21:34
permalink: /ja/papers/roformer/
---

> [Jianlin Su](https://spaces.ac.cn/)、[Yu Lu](https://dblp.org/pid/09/2321.html)、[Shengfeng Pan](https://dblp.org/pid/249/7590.html)、[Ahmed Murtadha](https://dblp.org/pid/208/0019.html)、[Bo Wen](https://dblp.org/pid/00/2490.html)、[Yunfeng Liu](https://dblp.org/pid/56/5650.html)。2021 年 4 月 20 日に arXiv へ初回投稿され、現行版は v5 である。*Neurocomputing* 568（2024）、Article 127063 に掲載。[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)。[原論文 PDF](/paper/roformer.pdf)。[DOI](https://doi.org/10.1016/j.neucom.2023.127063)。[TeX ソース](https://export.arxiv.org/e-print/2104.09864v5)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 概要

近年、位置符号化は Transformer アーキテクチャにおいて有効性を示している。これは、系列内の異なる位置にある要素間の依存関係をモデル化するうえで有用な教師信号となる。本論文ではまず、位置情報を Transformer ベースの言語モデルの学習過程へ統合する各種手法を検討する。次に、位置情報を効果的に活用する新しい手法 Rotary Position Embedding（RoPE）を提案する。具体的には、RoPE は回転行列で絶対位置を符号化すると同時に、自己注意の定式化へ明示的な相対位置依存関係を組み込む。RoPE には、系列長に対する柔軟性、相対距離の増加に伴うトークン間依存関係の減衰、線形自己注意に相対位置符号化を導入できることなど、有用な性質がある。最後に、回転位置埋め込みで強化した Transformer、すなわち RoFormer を、複数の長文分類ベンチマークデータセットで評価する。実験では、RoFormer が一貫して代替手法を上回った。さらに、一部の実験結果を説明する理論解析も示す。RoFormer はすでに Huggingface に統合されている：[https://huggingface.co/docs/transformers/model_doc/roformer](https://huggingface.co/docs/transformers/model_doc/roformer)。

**キーワード：** 事前学習済み言語モデル、位置情報符号化、事前学習、自然言語処理。

<span id="section-1"></span>

## 1 はじめに

語の並び順は、自然言語理解において非常に重要である。再帰型ニューラルネットワーク（RRNs）ベースのモデルは、時間方向に沿って隠れ状態を再帰的に計算することでトークンの順序を符号化する。畳み込みニューラルネットワーク（CNNs）ベースのモデル [Geh17] は通常、位置に依存しないと考えられてきたが、近年の研究 [Isl20] は、一般的なパディング操作によって位置情報を暗黙的に学習できることを示した。近年、Transformer [Vas17] を基盤とする事前学習済み言語モデル（PLMs）は、文脈表現学習 [Dev19]、機械翻訳 [Vas17]、言語モデリング [Rad19] をはじめ、さまざまな自然言語処理（NLP）タスクで最高水準の性能を達成している。RNN や CNN ベースのモデルと異なり、PLM は自己注意機構を利用して、与えられたコーパスの文脈表現を意味的に捉える。その結果、PLM は RNN よりも並列化性能を大幅に向上させ、CNN と比べてより長いトークン間関係をモデル化する能力も高めている [+1]。

現在の PLM の自己注意アーキテクチャは位置に依存しないことが示されている [Yun20]。この知見を受け、位置情報を学習過程へ符号化するさまざまな手法が提案されてきた。一方では、事前定義した関数から生成した絶対位置符号化 [Vas17] を文脈表現に加える手法や、学習可能な絶対位置符号化 [Geh17, Dev19, Lan20, Cla20, Rad19, Rad18] が用いられる。もう一方では、既存研究 [Par16, Sha18d, Hua18a, Dai19, Yan19, Raf20, Ke20, He20, Hua20a] が相対位置符号化に注目し、通常は相対位置情報を注意機構へ符号化する。これらに加え、[Liu20] は Neural ODE [Che18g] の観点から位置符号化の依存関係をモデル化し、[Wan20f] は複素空間で位置情報をモデル化する手法を提案した。これらの手法は有効ではあるものの、いずれも位置情報を文脈表現へ加えるため、線形自己注意アーキテクチャには適さない。

本論文では、位置情報を PLM の学習過程で活用する新しい手法 Rotary Position Embedding（RoPE）を導入する。具体的には、RoPE は回転行列で絶対位置を符号化すると同時に、自己注意の定式化へ明示的な相対位置依存関係を組み込む。提案する RoPE は、系列長に対する柔軟性、相対距離の増加に伴うトークン間依存関係の減衰、線形自己注意に相対位置符号化を導入できることなど、有用な性質によって既存手法より優れている。複数の長文分類ベンチマークデータセットでの実験結果は、回転位置埋め込みで強化した Transformer、すなわち RoFormer が基準となる代替手法より高い性能を示すことを明らかにし、提案する RoPE の有効性を実証している。

本研究の貢献は、以下の 3 点に要約できる：

- 既存の相対位置符号化手法を調査し、その多くが、文脈表現へ位置符号化を加えた式を分解するという発想に基づくことを明らかにした。位置情報を PLM の学習過程で活用する新しい手法 Rotary Position Embedding（RoPE）を導入する。中心となる考え方は、明確な理論的解釈を持つ回転行列を文脈表現に乗じて相対位置を符号化することである。
- RoPE の性質を調べ、相対距離の増加に伴って減衰することを示す。これは自然言語の符号化に望ましい性質である。また、従来の相対位置符号化に基づく手法は線形自己注意と互換性がないと論じる。
- 提案する RoFormer を複数の長文ベンチマークデータセットで評価する。実験では、RoFormer が一貫して代替手法より高い性能を達成した。事前学習済み言語モデルを用いた一部の実験は GitHub で公開している：[https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer)。

本論文の以降の構成は次のとおりである。[第 2 節](#section-2)では、自己注意アーキテクチャにおける位置符号化問題を形式的に記述し、既存研究を振り返る。[第 3 節](#section-3)では回転位置符号化（RoPE）を説明し、その性質を調べる。[第 4 節](#section-4)では実験を報告する。最後に、[第 5 節](#section-5)で本論文をまとめる。

<span id="section-2"></span>

## 2 背景と関連研究

<span id="section-2-1"></span>

### 2.1 準備

$\mathbb{S}_{N}=\{w_{i}\}_{i=1}^{N}$ を $N$ 個の入力トークンからなる系列とし、$w_{i}$ を第 $i$ 要素とする。$\mathbb{S}_{N}$ に対応する単語埋め込みを $\mathbb{E}_{N}=\{{\boldsymbol{x}}_{i}\}_{i=1}^{N}$ と表す。ここで ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$ は、位置情報を含まないトークン $w_{i}$ の $d$ 次元単語埋め込みベクトルである。自己注意はまず単語埋め込みへ位置情報を組み込み、それらをクエリ、キー、バリュー表現へ変換する。

<span id="equation-01"></span>

$$
\begin{aligned}
{\boldsymbol{q}}_{m} & =f_{q}({\boldsymbol{x}}_{m},m) \\
{\boldsymbol{k}}_{n} & =f_{k}({\boldsymbol{x}}_{n},n) \\
{\boldsymbol{v}}_{n} & =f_{v}({\boldsymbol{x}}_{n},n),
\end{aligned}
$$

ここで ${\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}$、${\boldsymbol{v}}_{n}$ は、それぞれ $f_{q},f_{k}$、$f_{v}$ を通じて第 $m$ および第 $n$ 位置を組み込む。次にクエリとキーの値から注意重みを計算し、バリュー表現の重み付き和として出力を求める。

<span id="equation-02"></span>

$$
\begin{aligned}
a_{m,n} & =\frac{\exp(\frac{{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}}{\sqrt{d}})}{\sum_{j=1}^{N}\exp(\frac{{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{j}}{\sqrt{d}})} \\
\mathbf{o}_{m} & =\sum_{n=1}^{N}a_{m,n}{\boldsymbol{v}}_{n}
\end{aligned}
$$

既存の Transformer ベース位置符号化手法は、主として[式 1](#equation-01)を構成する適切な関数の選択に注目している。

<span id="section-2-2"></span>

### 2.2 絶対位置埋め込み

[式 1](#equation-01)の典型的な選択は、次のとおりである

<span id="equation-03"></span>

$$
f_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i},i):={\boldsymbol{W}}_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i}+{\boldsymbol{p}}_{i}),
$$

ここで ${\boldsymbol{p}}_{i}\in\mathbb{R}^{d}$ は、トークン ${\boldsymbol{x}}_{i}$ の位置に依存する $d$ 次元ベクトルである。既存研究 [Dev19, Lan20, Cla20, Rad19, Rad18] は、学習可能なベクトル集合 ${\boldsymbol{p}}_{i}\in\{{\boldsymbol{p}}_{t}\}_{t=1}^{L}$ を導入した。ここで $L$ は最大系列長である。[Vas17] は正弦関数によって ${\boldsymbol{p}}_{i}$ を生成することを提案した。

<span id="equation-04"></span>

$$
\begin{cases}{\boldsymbol{p}}_{i,2t}&=\sin(k/10000^{2t/d})\\
{\boldsymbol{p}}_{i,2t+1}&=\cos(k/10000^{2t/d})\end{cases}
$$

ここで ${\boldsymbol{p}}_{i,2t}$ は、$d$ 次元ベクトル ${\boldsymbol{p}}_{i}$ の第 $2t$ 要素である。次節では、正弦関数の観点から、提案する RoPE がこの直観と関連していることを示す。ただし RoPE は、文脈表現へ位置を直接加えるのではなく、正弦関数を乗じることで相対位置情報を組み込む。

<span id="section-2-3"></span>

### 2.3 相対位置埋め込み

[Sha18d] は[式 1](#equation-01)に次の異なる設定を適用した：

<span id="equation-05"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m}):={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m} \\
f_{k}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{k}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{k}_{r}) \\
f_{v}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{v}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{v}_{r})
\end{aligned}
$$

ここで $\tilde{{\boldsymbol{p}}}^{k}_{r},\tilde{{\boldsymbol{p}}}^{v}_{r}\in\mathbb{R}^{d}$ は学習可能な相対位置埋め込みである。$r=\mathrm{clip}(m-n,r_{\min},r_{\max})$ は位置 $m$ と $n$ の相対距離を表す。一定の距離を超えると正確な相対位置情報は有用でないという仮説に基づき、相対距離を切り詰めた。[式 3](#equation-03)の形を保ちながら、[Dai19] は[式 2](#equation-02)の ${\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}$ を次のように分解することを提案した

<span id="equation-06"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n}+{\boldsymbol{p}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n},
$$

中心となる考え方は、絶対位置埋め込み ${\boldsymbol{p}}_{n}$ を正弦符号化した相対位置ベクトル $\tilde{{\boldsymbol{p}}}_{m-n}$ へ置き換え、第 3 項と第 4 項の絶対位置 ${\boldsymbol{p}}_{m}$ を、クエリ位置に依存しない 2 つの学習可能なベクトル $\mathbf{u}$ と $\mathbf{v}$ へ置き換えることである。さらに、内容ベースのキーベクトル ${\boldsymbol{x}}_{n}$ と位置ベースのキーベクトル ${\boldsymbol{p}}_{n}$ に対して ${\boldsymbol{W}}_{k}$ を分け、それぞれ ${\boldsymbol{W}}_{k}$ と $\widetilde{{\boldsymbol{W}}}_{k}$ で表すと、次式を得る：

<span id="equation-07"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\mathbf{u}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+\mathbf{v}^\top{\boldsymbol{W}}_{q}^\top\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}
$$

$f_{v}({\boldsymbol{x}}_{j}):={\boldsymbol{W}}_{v}{\boldsymbol{x}}_{j}$ と置くことで、バリュー項から位置情報が除かれる点に注意されたい。後続研究 [Raf20, He20, Ke20, Hua20a] はこの設定に従い、相対位置情報を注意重みだけに符号化した。ただし [Raf20] は[式 6](#equation-06)を次のように変形した：

<span id="equation-08"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+b_{i,j}
$$

ここで $b_{i,j}$ は学習可能なバイアスである。[Ke20] は[式 6](#equation-06)の中央 2 項を調べ、絶対位置と単語の間にほとんど相関がないことを見いだした。[Raf20] は、単語または位置の対を異なる射影行列でモデル化することを提案した。

<span id="equation-09"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^\top\mathbf{U}_{q}^\top\mathbf{U}_{k}{\boldsymbol{p}}_{n}+b_{i,j}
$$

[He20] は、2 つのトークンの相対位置を完全にモデル化するには、[式 6](#equation-06)の中央 2 項を用いる必要があると論じた。その結果、絶対位置埋め込み ${\boldsymbol{p}}_{m}$ と ${\boldsymbol{p}}_{n}$ は、相対位置埋め込み $\tilde{{\boldsymbol{p}}}_{m-n}$ に単純に置き換えられた：

<span id="equation-10"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\tilde{{\boldsymbol{p}}}_{m-n}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$

相対位置埋め込みの 4 変種を比較した研究 [Rad18] は、[式 10](#equation-10)に似た変種が残りの 3 つより効率的であることを示した。概して、これらの手法はいずれも、[式 2](#equation-02)の自己注意設定のもとで[式 3](#equation-03)を分解し、[式 6](#equation-06)を変更しようとする。この考え方は [Vas17] で初めて提案された。これらは共通して、位置情報を文脈表現へ直接加える。これに対して本手法は、いくつかの制約のもとで[式 1](#equation-01)から相対位置符号化を導出することを目指す。次に、文脈表現の回転によって相対位置情報を組み込むことで、導出した手法の解釈可能性が高まることを示す。

<span id="section-3"></span>

## 3 提案手法

本節では、提案する回転位置埋め込み（RoPE）を説明する。まず[第 3.1 節](#section-3-1)で相対位置符号化問題を定式化し、続いて[第 3.2 節](#section-3-2)で RoPE を導出し、[第 3.3 節](#section-3-3)でその性質を調べる。

<span id="section-3-1"></span>

### 3.1 定式化

Transformer ベースの言語モデリングでは通常、自己注意機構を通じて各トークンの位置情報を利用する。[式 2](#equation-02)から分かるように、${\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}$ は一般に、異なる位置にあるトークン間の知識伝達を可能にする。相対位置情報を組み込むため、クエリ ${\boldsymbol{q}}_{m}$ とキー ${\boldsymbol{k}}_{n}$ の内積を関数 $g$ で表すことを要求する。この関数は、単語埋め込み ${\boldsymbol{x}}_{m}$、${\boldsymbol{x}}_{n}$ と、それらの相対位置 $m-n$ だけを入力変数とする。言い換えれば、内積が位置情報を相対形式だけで符号化することを望む：

<span id="equation-11"></span>

$$
\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n).
$$

最終目標は、上記の関係を満たす関数 $f_{q}({\boldsymbol{x}}_{m},m)$ と $f_{k}({\boldsymbol{x}}_{n},n)$ を求める、等価な符号化機構を見いだすことである。

<span id="section-3-2"></span>

### 3.2 回転位置埋め込み

<span id="section-3-2-1"></span>

#### 3.2.1 二次元の場合

まず、次元 $d=2$ の単純な場合を考える。この設定では、二次元平面上のベクトルの幾何学的性質とその複素数形式を利用し、[式 11](#equation-11)の定式化に対する解が次のようになることを証明できる（詳細は[第 3.4.1 節](#section-3-4-1)を参照）：

<span id="equation-12"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m},m) & =({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{i m\theta} \\
f_{k}({\boldsymbol{x}}_{n},n) & =({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{i n\theta} \\
g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n) & =\operatorname{Re}[({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}e^{i(m-n)\theta}]
\end{aligned}
$$

ここで $\operatorname{Re}[\cdot]$ は複素数の実部、$({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}$ は $({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})$ の共役複素数を表す。$\theta\in\mathbb{R}$ は、あらかじめ設定した非零定数である。さらに $f_{\{q,k\}}$ を行列の積として書ける：

<span id="equation-13"></span>

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)=\left(\begin{array}{cc}\cos{m\theta}&-\sin{m\theta}\\
\sin{m\theta}&\cos{m\theta}\end{array}\right)\left(\begin{array}{cc}W^{(11)}_{\{q,k\}}&W^{(12)}_{\{q,k\}}\\
W^{(21)}_{\{q,k\}}&W^{(22)}_{\{q,k\}}\end{array}\right)\left(\begin{array}{cc}x^{(1)}_{m}\\
x^{(2)}_{m}\end{array}\right)
$$

ここで $(x^{(1)}_{m},x^{(2)}_{m})$ は、${\boldsymbol{x}}_{m}$ を二次元座標で表したものである。同様に、$g$ も行列とみなせるため、二次元の場合に[第 3.1 節](#section-3-1)の定式化を解ける。具体的には、相対位置埋め込みの導入は単純である。アフィン変換後の単語埋め込みベクトルを、その位置インデックスの倍数に当たる角度だけ回転させればよく、これが*回転位置埋め込み*の直観を説明する。

<span id="section-3-2-2"></span>

#### 3.2.2 一般形

二次元での結果を、$d$ が偶数である任意の ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$ へ一般化するため、$d$ 次元空間を $d/2$ 個の部分空間に分割し、内積の線形性を利用して結合する。すると $f_{\{q,k\}}$ は次のようになる：

<span id="equation-14"></span>

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)={\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{\{q,k\}}{\boldsymbol{x}}_{m}
$$

ここで

<span id="equation-15"></span>

$$
{\boldsymbol{R}}^{d}_{\Theta,m}=\begin{pmatrix}\cos{m\theta_{1}}&-\sin{m\theta_{1}}&0&0&\cdots&0&0\\
\sin{m\theta_{1}}&\cos{m\theta_{1}}&0&0&\cdots&0&0\\
0&0&\cos{m\theta_{2}}&-\sin{m\theta_{2}}&\cdots&0&0\\
0&0&\sin{m\theta_{2}}&\cos{m\theta_{2}}&\cdots&0&0\\
\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&0&\cdots&\cos{m\theta_{d/2}}&-\sin{m\theta_{d/2}}\\
0&0&0&0&\cdots&\sin{m\theta_{d/2}}&\cos{m\theta_{d/2}}\end{pmatrix}
$$

これは、事前定義したパラメータ $\Theta=\{\theta_{i}=10000^{-2(i-1)/d},i\in[1,2,...,d/2]\}$ を持つ回転行列である。RoPE の図解を[図 1](#figure-01)に示す。RoPE を[式 2](#equation-02)の自己注意へ適用すると、次式を得る：

<span id="equation-16"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}=({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^\top({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})={\boldsymbol{x}}^\top{\boldsymbol{W}}_{q}R^{d}_{\Theta,n-m}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$

ここで ${\boldsymbol{R}}^{d}_{\Theta,n-m}=({\boldsymbol{R}}^{d}_{\Theta,m})^\top{\boldsymbol{R}}^{d}_{\Theta,n}$ である。${\boldsymbol{R}}^{d}_{\Theta}$ は直交行列であり、位置情報を符号化する過程の安定性を保証する点に注意されたい。さらに、$R^{d}_{\Theta}$ は疎であるため、[式 16](#equation-16)のように行列積を直接適用することは計算上効率的でない。理論的説明では別の実装を示す。

既存研究で採用された位置埋め込み手法、すなわち[式 3](#equation-03)、[4](#equation-04)、[5](#equation-05)、[6](#equation-06)、[7](#equation-07)、[8](#equation-08)、[9](#equation-09)、[10](#equation-10)が加法的であるのに対し、本手法は乗法的である。さらに RoPE は、自己注意へ適用するときに加法的位置符号化の展開式の項を変更するのではなく、回転行列の積によって相対位置情報を自然に組み込む。

<span id="figure-01"></span>

![回転位置埋め込みの実装](../../papers/roformer/figure-01.png)

**図 1.** 回転位置埋め込み（RoPE）の実装。

<span id="section-3-3"></span>

### 3.3 RoPE の性質

**長期的減衰：** [Vas17] に従い、$\theta_{i}=10000^{-2i/d}$ と設定する。この設定が長期的減衰の性質をもたらすことを証明できる（詳細は[第 3.4.3 節](#section-3-4-3)を参照）。すなわち、相対位置が増加すると内積が減衰する。この性質は、相対距離が長いトークン対ほど関連が弱いはずだという直観に合致する。

**線形注意を用いた RoPE：** 自己注意は、より一般的な形に書き換えられる。

<span id="equation-17"></span>

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})}.
$$

元の自己注意では、$\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})=\exp({\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}/\sqrt{d})$ を選択する。元の自己注意はトークンの各対についてクエリとキーの内積を計算する必要があり、$O(N^{2})$ の二次計算量を持つ点に注意されたい。[Kat20] に従い、線形注意は[式 17](#equation-17)を次のように変形する

<span id="equation-18"></span>

$$
\mathrm{Attention}({\boldsymbol{Q}},{\boldsymbol{K}},{\boldsymbol{V}})_{m}=\frac{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n})},
$$

ここで $\phi(\cdot),\varphi(\cdot)$ は通常、非負関数である。[Kat20] は $\phi(x)=\varphi(x)=\operatorname{elu}(x)+1$ を提案し、行列積の結合則を用いてキーとバリューの積を先に計算した。[She21] は内積の前に softmax 関数でクエリとキーを別々に正規化する。これは $\phi({\boldsymbol{q}}_{i})=\mathrm{softmax}({\boldsymbol{q}}_{i})$ および $\phi({\boldsymbol{k}}_{j})=\exp({\boldsymbol{k}}_{j})$ と等価である。線形注意の詳細については原論文を参照されたい。本節では、RoPE を[式 18](#equation-18)へ組み込むことに焦点を当てる。RoPE は回転によって位置情報を注入し、隠れ表現のノルムを変えないため、回転行列を非負関数の出力に乗じることで RoPE と線形注意を組み合わせられる。

<span id="equation-19"></span>

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\big({\boldsymbol{R}}^{d}_{\Theta,m}\phi({\boldsymbol{q}}_{m})\big)^\top\big({\boldsymbol{R}}^{d}_{\Theta,n}\varphi({\boldsymbol{k}}_{n})\big){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n})}.
$$

零除算の危険を避けるため分母は変更せず、分子の和には負の項が含まれ得る点に注意されたい。[式 19](#equation-19)における各バリュー ${\boldsymbol{v}}_{i}$ の重みは、確率として厳密に正規化されてはいないものの、この計算でもバリューの重要度をモデル化できると考える。

<span id="section-3-4"></span>

### 3.4 理論的説明

<span id="section-3-4-1"></span>

#### 3.4.1 二次元における RoPE の導出

$d=2$ の場合、クエリとキーに対応する 2 つの単語埋め込みベクトル ${\boldsymbol{x}}_{q}$、${\boldsymbol{x}}_{k}$ と、それぞれの位置 $m$、$n$ を考える。[式 1](#equation-01)によると、位置符号化後の対応するベクトルは次のとおりである：

<span id="equation-20"></span>

$$
\begin{aligned}
{\boldsymbol{q}}_{m} & =f_{q}({\boldsymbol{x}}_{q},m), \\
{\boldsymbol{k}}_{n} & =f_{k}({\boldsymbol{x}}_{k},n),
\end{aligned}
$$

ここで ${\boldsymbol{q}}_{m}$ と ${\boldsymbol{k}}_{n}$ の添字は、符号化された位置情報を示す。$f_{\{q,k\}}$ が生成するベクトル間の内積を定義する関数 $g$ が存在すると仮定する：

<span id="equation-21"></span>

$$
{\boldsymbol{q}}^\top_{m}{\boldsymbol{k}}_{n}=\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},n-m),
$$

さらに、次の初期条件を満たすことを要求する：

<span id="equation-22"></span>

$$
\begin{aligned}
{\boldsymbol{q}} & =f_{q}({\boldsymbol{x}}_{q},0), \\
{\boldsymbol{k}} & =f_{k}({\boldsymbol{x}}_{k},0),
\end{aligned}
$$

これらは、位置情報がまだ符号化されていないベクトルと解釈できる。この設定のもとで、$f_{q}$ と $f_{k}$ の解を求める。まず、二次元におけるベクトルの幾何学的意味と対応する複素数形式を利用し、[式 20](#equation-20)と[21](#equation-21)の関数を次のように分解する：

<span id="equation-23"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{q},m) & =R_{q}({\boldsymbol{x}}_{q},m)e^{i\Theta_{q}({\boldsymbol{x}}_{q},m)}, \\
f_{k}({\boldsymbol{x}}_{k},n) & =R_{k}({\boldsymbol{x}}_{k},n)e^{i\Theta_{k}({\boldsymbol{x}}_{k},n)}, \\
g({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)e^{i\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)},
\end{aligned}
$$

ここで $R_{f}$、$R_{g}$ と $\Theta_{f}$、$\Theta_{g}$ は、それぞれ $f_{\{q,k\}}$ と $g$ の動径成分および角度成分である。これらを[式 21](#equation-21)へ代入すると、次の関係を得る：

<span id="equation-24"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},n) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m), \\
\Theta_{k}({\boldsymbol{x}}_{k},n)-\Theta_{q}({\boldsymbol{x}}_{q},m) & =\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m),
\end{aligned}
$$

対応する初期条件は次のとおりである：

<span id="equation-25"></span>

$$
\begin{aligned}
{\boldsymbol{q}} & =\|{\boldsymbol{q}}\|e^{i\theta_{q}}=R_{q}({\boldsymbol{x}}_{q},0)e^{i\Theta_{q}({\boldsymbol{x}}_{q},0)}, \\
{\boldsymbol{k}} & =\|{\boldsymbol{k}}\|e^{i\theta_{k}}=R_{k}({\boldsymbol{x}}_{k},0)e^{i\Theta_{k}({\boldsymbol{x}}_{k},0)},
\end{aligned}
$$

ここで $\|{\boldsymbol{q}}\|$、$\|{\boldsymbol{k}}\|$ と $\theta_{q}$、$\theta_{k}$ は、それぞれ二次元平面上の ${\boldsymbol{q}}$ と ${\boldsymbol{k}}$ の動径成分および角度成分である。

次に、[式 24](#equation-24)で $m=n$ とし、[式 25](#equation-25)の初期条件を考慮する：

<span id="equation-26"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=R_{q}({\boldsymbol{x}}_{q},0)R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|, \\
\Theta_{k}({\boldsymbol{x}}_{k},m)-\Theta_{q}({\boldsymbol{x}}_{q},m) & =\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\Theta_{k}({\boldsymbol{x}}_{k},0)-\Theta_{q}({\boldsymbol{x}}_{q},0)=\theta_{k}-\theta_{q}.
\end{aligned}
$$

一方、[式 26a](#equation-26)から $R_{f}$ の直接的な解を構成できる：

<span id="equation-27"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m) & =R_{q}({\boldsymbol{x}}_{q},0)=\|{\boldsymbol{q}}\| \\
R_{k}({\boldsymbol{x}}_{k},n) & =R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{k}}\| \\
R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|
\end{aligned}
$$

これは、動径関数 $R_{q}$、$R_{k}$、$R_{g}$ が位置情報に依存しないことを意味する。他方、[式 26b](#equation-26)の $\Theta_{q}({\boldsymbol{x}}_{q},m)-\theta_{q}=\Theta_{k}({\boldsymbol{x}}_{k},m)-\theta_{k}$ は、角度関数がクエリとキーに依存しないことを示す。そこで $\Theta_{f}:=\Theta_{q}=\Theta_{k}$ とする。また、項 $\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)-\theta_{\{q,k\}}$ は位置 $m$ の関数であり、単語埋め込み ${\boldsymbol{x}}_{\{q,k\}}$ には依存しないため、これを $\phi(m)$ と表す。すると次式を得る：

<span id="equation-28"></span>

$$
\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)=\phi(m)+\theta_{\{q,k\}},
$$

さらに、[式 24](#equation-24)へ $n=m+1$ を代入し、上式を考慮すると、次式を得る：

<span id="equation-29"></span>

$$
\phi(m+1)-\phi(m)=\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},1)+\theta_{q}-\theta_{k},
$$

右辺は $m$ に依存しない定数なので、連続する整数を入力した $\phi(m)$ は等差数列をなす：

<span id="equation-30"></span>

$$
\phi(m)=m\theta+\gamma,
$$

ここで $\theta,\gamma\in\mathbb{R}$ は定数であり、$\theta$ は非零である。[式 27](#equation-27)、[28](#equation-28)、[29](#equation-29)、[30](#equation-30)の解をまとめると、次のようになる：

<span id="equation-31"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{q},m) & =\|{\boldsymbol{q}}\|e^{i\theta_{q}+m\theta+\gamma}={\boldsymbol{q}}e^{i(m\theta+\gamma)}, \\
f_{k}({\boldsymbol{x}}_{k},n) & =\|{\boldsymbol{k}}\|e^{i\theta_{k}+n\theta+\gamma}={\boldsymbol{k}}e^{i(n\theta+\gamma)}.
\end{aligned}
$$

[式 22](#equation-22)の $f_{q}$ と $f_{k}$ には制約を加えていないため、$f_{q}({\boldsymbol{x}}_{m},0)$ と $f_{k}({\boldsymbol{x}}_{n},0)$ は自由に選べる。[式 3](#equation-03)と比較できるように、次のように定義する：

<span id="equation-32"></span>

$$
\begin{aligned}
{\boldsymbol{q}}=f_{q}({\boldsymbol{x}}_{m},0) & ={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{n}, \\
{\boldsymbol{k}}=f_{k}({\boldsymbol{x}}_{n},0) & ={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}.
\end{aligned}
$$

続いて、[式 31](#equation-31)の最終解で単純に $\gamma=0$ と置く：

<span id="equation-33"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m},m) & =({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{i m\theta}, \\
f_{k}({\boldsymbol{x}}_{n},n) & =({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{i n\theta}.
\end{aligned}
$$

<span id="section-3-4-2"></span>

#### 3.4.2 回転行列積の計算効率に優れた実装

[式 15](#equation-15)における ${\boldsymbol{R}}^{d}_{\Theta,m}$ の疎性を利用すると、$R^{d}_{\Theta}$ と ${\boldsymbol{x}}\in\mathbb{R}^{d}$ の積は、より計算効率よく次のように実装できる：

<span id="equation-34"></span>

$$
{\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{x}}=\begin{pmatrix}x_{1}\\
x_{2}\\
x_{3}\\
x_{4}\\
\vdots\\
x_{d-1}\\
x_{d}\end{pmatrix}\otimes\begin{pmatrix}\cos{m\theta_{1}}\\
\cos{m\theta_{1}}\\
\cos{m\theta_{2}}\\
\cos{m\theta_{2}}\\
\vdots\\
\cos{m\theta_{d/2}}\\
\cos{m\theta_{d/2}}\end{pmatrix}+\begin{pmatrix}-x_{2}\\
x_{1}\\
-x_{4}\\
x_{3}\\
\vdots\\
-x_{d}\\
x_{d-1}\end{pmatrix}\otimes\begin{pmatrix}\sin{m\theta_{1}}\\
\sin{m\theta_{1}}\\
\sin{m\theta_{2}}\\
\sin{m\theta_{2}}\\
\vdots\\
\sin{m\theta_{d/2}}\\
\sin{m\theta_{d/2}}\end{pmatrix}
$$

<span id="section-3-4-3"></span>

#### 3.4.3 RoPE の長期的減衰

ベクトル ${\boldsymbol{q}}={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m}$ と ${\boldsymbol{k}}={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}$ の成分を 2 つずつまとめると、[式 16](#equation-16)の RoPE の内積を複素数の積として書ける。

<span id="equation-35"></span>

$$
({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^\top({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})=\operatorname{Re}\bigg[\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg]
$$

ここで ${\boldsymbol{q}}_{[2i:2i+1]}$ は、${\boldsymbol{q}}$ の第 $2i$ 成分から第 $(2i+1)$ 成分までを表す。$h_{i}={\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}$、$S_{j}=\sum_{i=0}^{j-1}e^{i(m-n)\theta_{i}}$ と置き、$h_{d/2}=0$、$S_{0}=0$ とすれば、Abel 変換を用いて和を次のように書き換えられる

<span id="equation-36"></span>

$$
\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}=\sum_{i=0}^{d/2-1}h_{i}(S_{i+1}-S_{i})=-\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i}).
$$

したがって、

<span id="equation-37"></span>

$$
\begin{aligned}
\bigg|\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg| & =\bigg|\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i})\bigg| \\
\leq\sum_{i=0}^{d/2-1}|S_{i+1}|\,|h_{i+1}-h_{i}| \\
\leq\big(\max_{i}|h_{i+1}-h_{i}|\big)\sum_{i=0}^{d/2-1}|S_{i+1}|
\end{aligned}
$$

$\theta_{i}=10000^{-2i/d}$ と設定すると、[図 2](#figure-02)に示すように、$\frac{1}{d/2}\sum_{i=1}^{d/2}|S_{i}|$ の値は相対距離 $m-n$ の増加に伴って減衰する。

<span id="figure-02"></span>

![RoPE の長期的減衰](../../papers/roformer/figure-02.png)

**図 2.** RoPE の長期的減衰。

<span id="section-4"></span>

## 4 実験と評価

提案する RoFormer を、次のようにさまざまな NLP タスクで評価する。[第 4.1 節](#section-4-1)では、機械翻訳タスクで提案手法の性能を検証する。次に[第 4.2 節](#section-4-2)では、事前学習段階で RoPE の実装と BERT [Dev19] を比較する。事前学習済みモデルを基に、[第 4.3 節](#section-4-3)では GLUE ベンチマーク [Wan18d] のさまざまな下流タスクで追加評価を行う。さらに[第 4.4 節](#section-4-4)では、提案する RoPE と PerFormer [Cho20a] の線形注意を組み合わせて実験する。最後に、[第 4.5 節](#section-4-5)で中国語データを用いた追加試験を示す。すべての実験は、それぞれ 4 基の V100 GPU を備えた 2 台のクラウドサーバーで実行した。

<span id="section-4-1"></span>

### 4.1 機械翻訳

まず、系列間言語翻訳タスクにおける RoFormer の性能を示す。

<span id="section-4-1-1"></span>

#### 4.1.1 実験設定

約 450 万文対からなる標準 WMT 2014 英独データセット [Boj14] を選択する。Transformer ベースの基準手法 [Vas17] と比較する。

<span id="section-4-1-2"></span>

#### 4.1.2 実装の詳細

RoPE を学習過程へ導入するため、基準モデル [Vas17] の自己注意層にいくつかの変更を加える。原言語と目的言語を合わせたバイト対符号化（BPE）[Sen15] に基づき、語彙数 37k の英独翻訳設定を再現する。評価時には、最後の 5 チェックポイントを平均して単一モデルを得る。結果には、ビーム幅 4、長さペナルティ 0.6 のビーム探索を用いる。実験は fairseq ツールキット（MIT License）[Ott19a] を用い、PyTorch で実装する。モデルは Adam オプティマイザで最適化し、$\beta_{1}=0.9$、$\beta_{2}=0.98$ とする。学習率は $1e-7$ から $5e-4$ まで線形に増加させた後、ステップ数の平方根の逆数に比例して減衰させる。0.1 のラベル平滑化も採用する。テストセット上の BLEU [Pap02] スコアを最終指標として報告する。

<span id="section-4-1-3"></span>

#### 4.1.3 結果

基準モデルと RoFormer を同じ設定で学習し、結果を[表 1](#table-01)に示す。本モデルは、基準 Transformer より高い BLEU スコアを得た。

<span id="table-01"></span>

![WMT 2014 英独翻訳タスクの BLEU スコア](../../papers/roformer/table-01.png)

**表 1.** WMT 2014 英独翻訳タスク [Boj14] において、提案する RoFormer は基準手法 [Vas17] より高い BLEU スコアを得た。

<span id="section-4-2"></span>

### 4.2 事前学習言語モデリング

第 2 の実験では、文脈表現の学習という観点から提案手法の性能を検証する。そのため、事前学習段階で BERT の元の正弦位置符号化を RoPE に置き換える。

<span id="section-4-2-1"></span>

#### 4.2.1 実験設定

事前学習には、Huggingface Datasets ライブラリ（Apache License 2.0）の BookCorpus [Zhu15] と Wikipedia Corpus [Wik21] を用いる。コーパスはさらに 8:2 の比率で学習セットと検証セットに分割する。学習過程のマスク言語モデリング（MLM）損失値を評価指標として用いる。よく知られた BERT [Dev19] を基準モデルとして採用する。実験では bert-base-uncased を使用する点に注意されたい。

<span id="section-4-2-2"></span>

#### 4.2.2 実装の詳細

RoFormer では、基準モデルの自己注意ブロックにある正弦位置符号化を提案する RoPE に置き換え、[式 16](#equation-16)に従って自己注意を実装する。BERT と RoFormer はどちらも、バッチサイズ 64、最大系列長 512 で 100k ステップ学習する。オプティマイザには AdamW [Los17] を用い、学習率は 1e-5 とする。

<span id="section-4-2-3"></span>

#### 4.2.3 結果

事前学習中の MLM 損失を[図 3](#figure-03)の左図に示す。通常の BERT と比べて、RoFormer はより速く収束する。

<span id="figure-03"></span>

![BERT、RoFormer、PerFormer 各変種の学習損失曲線](../../papers/roformer/figure-03.png)

**図 3.** 言語モデリングの事前学習における RoPE の評価。**左：** BERT と RoFormer の学習損失。**右：** RoPE を使用した場合と使用しない場合の PerFormer の学習損失。

<span id="section-4-3"></span>

### 4.3 GLUE タスクでのファインチューニング

前述の実験と同様に、事前学習済み RoFormer の重みをさまざまな GLUE タスクでファインチューニングし、下流 NLP タスクに対する汎化能力を評価する。

<span id="section-4-3-1"></span>

#### 4.3.1 実験設定

GLUE の複数のデータセット、すなわち MRPC [Dol05]、SST-2 [Soc13]、QNLI [Raj16]、STS-B [Aln17]、QQP [Che18h]、MNLI [Wil18] を対象とする。評価指標には、MRPC と QQP では F1 スコア、STS-B では Spearman 相関係数、それ以外では正解率を用いる。

<span id="section-4-3-2"></span>

#### 4.3.2 実装の詳細

Huggingface Transformers ライブラリ（Apache License 2.0）[Wol20] を使用し、前述の各下流タスクで 3 エポックのファインチューニングを行う。最大系列長は 512、バッチサイズは 32、学習率は {2,3,4,5}e-5 とする。[Dev19] に従い、検証セット上の最良平均結果を報告する。

<span id="table-02"></span>

![下流 GLEU タスクのファインチューニングによる RoFormer と BERT の比較](../../papers/roformer/table-02.png)

**表 2.** 下流 GLEU タスクのファインチューニングによる RoFormer と BERT の比較。

<span id="section-4-3-3"></span>

#### 4.3.3 結果

ファインチューニングタスクの評価結果を[表 2](#table-02)に示す。RoFormer は 6 データセット中 3 つで BERT を大幅に上回り、その改善幅も大きい。

<span id="section-4-4"></span>

### 4.4 RoPE を用いた Performer

Performer [Cho20a] は、別の注意機構である線形注意を導入する。これは、入力系列長に応じて増大する二次の計算コストを避けるよう設計されている。[第 3.3 節](#section-3-3)で述べたように、提案する RoPE は PerFormer モデルへ容易に実装でき、自己注意の計算量を線形に保ちながら相対位置符号化を実現できる。言語モデリングの事前学習タスクでその性能を示す。

<span id="section-4-4-1"></span>

#### 4.4.1 実装の詳細

Enwik8 データセット [Mah06] で試験を行う。このデータセットは英語版 Wikipedia に由来し、英語テキストに加えてマークアップ、特殊文字、他言語のテキストを含む。RoPE を、次元数 768、12 ヘッドを持つ文字ベースの 12 層 PerFormer へ組み込む [+2]。RoPE の有効性をより明確に示すため、学習率 1e-4、バッチサイズ 128、固定最大系列長 1024 などの同一設定において、RoPE を使用した場合と使用しない場合の事前学習過程の損失曲線を報告する。

<span id="section-4-4-2"></span>

#### 4.4.2 結果

[図 3](#figure-03)の右図に示すように、Performer へ RoPE を導入すると、同じ学習ステップ数でより速く収束し、損失も低くなる。線形計算量に加わるこれらの改善により、Performer はさらに魅力的になる。

<span id="section-4-5"></span>

### 4.5 中国語データでの評価

英語データでの実験に加え、中国語データでの結果も示す。RoFormer の長文に対する性能を検証するため、長さが 512 文字を超える長文書で実験する。

<span id="section-4-5-1"></span>

#### 4.5.1 実装

これらの実験では、WoBERT [Su20] の絶対位置埋め込みを提案する RoPE に置き換える変更を加えた。他の Transformer ベース中国語事前学習モデル、すなわち BERT [Dev19]、WoBERT [Su20]、NEZHA [Wei19] と横断的に比較するため、それぞれのトークン化粒度と位置埋め込み情報を[表 3](#table-03)に示す。

<span id="table-03"></span>

![RoFormer と他の中国語事前学習モデルの横断比較](../../papers/roformer/table-03.png)

**表 3.** RoFormer と他の中国語事前学習モデルの横断比較。「abs」と「rel」は、それぞれ絶対位置埋め込みと相対位置埋め込みを表す。

<span id="section-4-5-2"></span>

#### 4.5.2 事前学習

中国語版 Wikipedia、ニュース、フォーラムから収集した約 34GB のデータで RoFormer を事前学習する。さまざまな状況へモデルを適応させるため、バッチサイズと最大入力系列長を変えながら、複数段階で事前学習を行う。[表 4](#table-04)に示すように、系列長の上限を増やすと RoFormer の正解率が上昇し、長文を扱う能力が示される。これは、提案する RoPE の優れた汎化能力によるものだと考える。

<span id="table-04"></span>

![中国語データセットにおける RoFormer の事前学習戦略](../../papers/roformer/table-04.png)

**表 4.** 中国語データセットにおける RoFormer の事前学習戦略。学習手順を複数の連続する段階に分ける。各段階では、最大系列長とバッチサイズの特定の組み合わせでモデルを学習する。

<span id="section-4-5-3"></span>

#### 4.5.3 下流タスクとデータセット

RoFormer が長文、すなわち意味的テキスト照合を扱う能力を示すため、Chinese AI and Law 2019 Similar Case Matching（CAIL2019-SCM）データセット [Xia19a] を選択する。CAIL2019-SCM には、中国最高人民法院が公開した 8964 組の事件三つ組が含まれる。入力三つ組を（A、B、C）と表し、それぞれ 3 事件の事実記述とする。タスクは、事前定義した類似度尺度のもとで、対（A、B）が（A、C）より近いかどうかを予測することである。文書が長いこと、すなわち大半が 512 文字を超えることから、既存手法の多くは CAIL2019-SCM データセットで顕著な性能を示せない点に注意されたい。学習、検証、テストセットは、一般的な 6:2:2 の比率で分割する。

<span id="section-4-5-4"></span>

#### 4.5.4 結果

事前学習済み RoFormer モデルを、異なる入力長の CAIL2019-SCM へ適用する。[表 5](#table-05)に示すように、同じ事前学習データを用いた事前学習済み BERT および WoBERT モデルと比較する。短文の切り捨て長を 512 とした場合、RoFormer の結果は WoBERT と同程度で、BERT の実装をわずかに上回る。しかし最大入力テキスト長を 1024 まで増やすと、RoFormer は WoBERT を絶対値で 1.5% 上回る。

<span id="table-05"></span>

![CAIL2019-SCM タスクの実験結果](../../papers/roformer/table-05.png)

**表 5.** CAIL2019-SCM タスクの実験結果。第 1 列の数値は、最大切り捨て系列長を示す。結果は正解率の百分率で表す。

<span id="section-4-5-5"></span>

#### 4.5.5 本研究の限界

理論的根拠と有望な実験上の裏付けを示したものの、本手法には次の限界がある：

- 相対位置関係を二次元部分空間での回転として数学的に表したものの、他の位置符号化戦略を組み込んだ基準モデルより速く収束する理由について、十分な説明がない。
- 本モデルのトークン間積が長期的減衰という望ましい性質を持つことは証明したが（[第 3.3 節](#section-3-3)）、これは既存の位置符号化機構にも似た性質である。それにもかかわらず、本モデルは同種のモデルより長文で高い性能を示すが、その理由を十分には説明できていない。

提案する RoFormer は Transformer ベースの基盤上に構築されており、事前学習にはハードウェア資源を必要とする。

<span id="section-5"></span>

## 5 結論

本研究では、自己注意へ明示的な相対位置依存関係を組み込み、Transformer アーキテクチャの性能を高める新しい位置埋め込み手法を提案した。理論解析は、相対位置が自己注意のベクトル積によって自然に定式化でき、絶対位置情報は回転行列によって符号化されることを示す。さらに、提案手法を Transformer へ適用したときの有利な性質を数学的に説明した。最後に、英語と中国語の両ベンチマークデータセットでの実験により、本手法が事前学習の収束を速めることを示した。実験結果は、提案する RoFormer が長文タスクでより高い性能を達成できることも示している。

[+1]: 複数の CNN 層を積み重ねても、より長いトークン間関係を捉えられるが、ここでは単一層の設定のみを考える。

[+2]: この実験では、[https://github.com/lucidrains/performer-pytorch](https://github.com/lucidrains/performer-pytorch) のコード（MIT License）を採用した。
