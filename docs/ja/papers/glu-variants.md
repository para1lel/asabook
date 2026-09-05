---
title: 'GLU Variants Improve Transformer'
createTime: 2026/09/05 13:11:11
permalink: /ja/papers/glu-variants/
---

> [Noam Shazeer](https://www.noamshazeer.com/)。arXiv 初回投稿日：2020 年 2 月 12 日、現行版：v1。[GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1)。<a href="/paper/glu-variants.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[arXiv DOI](https://doi.org/10.48550/arXiv.2002.05202)。[TeX ソース](https://export.arxiv.org/e-print/2002.05202v1)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 概要

Gated Linear Units（GLU）[Dau16] は、2 つの線形射影の要素ごとの積からなり、そのうち一方は最初に sigmoid 関数を通過する。sigmoid の代わりに異なる非線形（あるいは線形）関数を用いることで、GLU の変種を構成できる。これらの変種を Transformer [Vas17] sequence-to-sequence モデルのフィードフォワードサブレイヤーで検証したところ、その一部は一般的に用いられる ReLU または GELU 活性化関数よりも品質を改善した。

<span id="section-1"></span>

## 1 はじめに

Transformer [Vas17] sequence-to-sequence モデルでは、multi-head attention と、同論文が「position-wise feed-forward networks」（FFN）と呼ぶものが交互に配置される。FFN はベクトル $x$（系列中の特定位置における隠れ表現）を受け取り、学習される 2 つの線形変換（行列 $W_1$、$W_2$ とバイアスベクトル $b_1$、$b_2$ で表される）を順に適用する。2 つの線形変換の間には rectified-linear（ReLU）[Glo11] 活性化関数が適用される。

<span id="equation-01"></span>

$$
\mathrm{FFN}(x,W_1,W_2,b_1,b_2)=\max(0,xW_1+b_1)W_2+b_2
$$

T5 コードベース [Raf19] に従い [+1]、本稿ではバイアスを持たない版を用いる：

<span id="equation-02"></span>

$$
\mathrm{FFN}_{\mathrm{ReLU}}(x,W_1,W_2)=\max(xW_1,0)W_2
$$

その後の研究では、ReLU を Gaussian Error Linear Units、$\mathrm{GELU}(x)=x\Phi(x)$ [Hen16] や、$\mathrm{Swish}_\beta(x)=x\sigma(\beta x)$ [Ram17] などの別の非線形活性化関数に置き換えることが提案されている。

<span id="equation-03"></span>

$$
\begin{aligned}
\mathrm{FFN}_{\mathrm{GELU}}(x,W_1,W_2)&=\mathrm{GELU}(xW_1)W_2 \\
\mathrm{FFN}_{\mathrm{Swish}}(x,W_1,W_2)&=\mathrm{Swish}_1(xW_1)W_2
\end{aligned}
$$

<span id="section-2"></span>

## 2 Gated Linear Units（GLU）とその変種

[Dau16] は Gated Linear Units（GLU）を導入した。これは入力に対する 2 つの線形変換を要素ごとに乗算し、その一方を sigmoid で活性化するニューラルネットワーク層である。同論文は活性化関数を省く方法も提案し、それを「bilinear」層と呼んで [Mni07] に帰している。

<span id="equation-04"></span>

$$
\begin{aligned}
\mathrm{GLU}(x,W,V,b,c)&=\sigma(xW+b)\otimes(xV+c) \\
\mathrm{Bilinear}(x,W,V,b,c)&=(xW+b)\otimes(xV+c)
\end{aligned}
$$

別の活性化関数を使った GLU の変種も定義できる：

<span id="equation-05"></span>

$$
\begin{aligned}
\mathrm{ReGLU}(x,W,V,b,c)&=\max(0,xW+b)\otimes(xV+c) \\
\mathrm{GEGLU}(x,W,V,b,c)&=\mathrm{GELU}(xW+b)\otimes(xV+c) \\
\mathrm{SwiGLU}(x,W,V,b,c,\beta)&=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)
\end{aligned}
$$

本稿では、第 1 の線形変換と活性化関数の代わりに GLU またはその変種の一つを用いる、Transformer FFN 層の新たな変種を提案する。ここでもバイアス項は省略する。

<span id="equation-06"></span>

$$
\begin{aligned}
\mathrm{FFN}_{\mathrm{GLU}}(x,W,V,W_2)&=(\sigma(xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{Bilinear}}(x,W,V,W_2)&=(xW\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{ReGLU}}(x,W,V,W_2)&=(\max(0,xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{GEGLU}}(x,W,V,W_2)&=(\mathrm{GELU}(xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)&=(\mathrm{Swish}_1(xW)\otimes xV)W_2
\end{aligned}
$$

これらの層はいずれも、元の FFN の 2 つに対して 3 つの重み行列を持つ。パラメータ数と計算量を一定に保つため、これらの層を元の 2 行列版と比較する際には、隠れユニット数 $d_{\mathrm{ff}}$（$W$ と $V$ の第 2 次元、および $W_2$ の第 1 次元）を $\frac{2}{3}$ 倍に減らす。

<span id="section-3"></span>

## 3 Text-to-Text Transfer Transformer（T5）での実験

ここまでに説明した FFN の変種を、[Raf19] の転移学習設定で検証する。encoder-decoder Transformer モデル [Vas17] を、欠落したテキスト区間を予測するノイズ除去目的で学習し、その後、さまざまな言語理解タスクで fine-tuning する。

<span id="section-3-1"></span>

### 3.1 モデルアーキテクチャ

[Raf19] の base モデルと同じコードベース、モデルアーキテクチャ、学習タスクを用いる。encoder と decoder はそれぞれ 12 層で構成され、$d_{\mathrm{model}}=768$ である。attention 層では $h=12$、$d_k=d_v=64$ である。FFN 層の隠れサイズは $d_{\mathrm{ff}}=3072$ である。前述のとおり、GLU 変種ベースの FFN 層は 2 つではなく 3 つの重み行列を持つため、base モデルと同じパラメータ数および演算数を維持するよう、隠れ層を $d_{\mathrm{ff}}=2048$ に縮小する。

<span id="table-01"></span>

![表 1。区間補完タスクにおける Transformer モデルのホールドアウト集合での対数パープレキシティ。](../../papers/glu-variants/table-01.png)

**表 1。** [Raf19] の区間補完タスクにおける Transformer モデルのホールドアウト集合での対数パープレキシティ。すべてのモデルでパラメータ数と計算量を一致させている。

<span id="section-3-2"></span>

### 3.2 事前学習とパープレキシティの結果

[Raf19] とまったく同様に、C4 データセットの span-filling 目的で 524,288 ステップの事前学習を行う。各学習バッチは 128 例で構成され、各例の入力は 512 token、出力は 114 token であり、出力には入力から削除された複数の token 区間が含まれる [+2]。[Raf19] と同様に、Adafactor optimizer [Sha18] と逆平方根 learning-rate schedule を用いる。学習の最後の 10％のステップでは、learning rate を線形に減衰させる。[Raf19] との主な違いは、事前学習中に dropout を使わないことである。これにより優れた結果が得られることを確認した。モデル品質のよい指標になると考え、C4 のホールドアウト shard 上で学習目的の対数パープレキシティを計算する。各モデルアーキテクチャについて、実行間のばらつきを測るため、より短い期間（65,536 ステップ）で 4 モデルも学習した。結果を[表 1](#table-01) に示す。GEGLU と SwiGLU の変種が最良のパープレキシティを示した。

<span id="section-3-3"></span>

### 3.3 Fine-Tuning

次に、学習済みの各モデルを一度ずつ、Stanford Question-Answering Dataset（SQuAD）[Raj16] と、GLUE [Wan18d] および SuperGlue [Wan19h] ベンチマーク内のすべての言語理解タスクを、例数に比例して混合したデータで fine-tuning する。 [+3] Fine-tuning は learning rate $10^{-3}$ で 131072 ステップ行う。学習時と同様に、各ステップの入力系列の合計長は約 65,536 token である。[Raf19] に従い、層の出力、フィードフォワード隠れ層、attention weight に dropout rate $0.1$ を用いる。embedding matrix は fine-tuning 中に固定する。

[表 2](#table-02)、[表 3](#table-03)、[表 4](#table-04) に開発集合での結果を示す。各タスクについて、fine-tuning 中に記録された checkpoint のうち最高のスコアを報告する。結果にはノイズがあるものの、新しい GLU 変種はほとんどのタスクで最良の性能を示す。比較のため、各表の末尾に [Raf19] の結果を掲載する。このモデルは、本稿の $\mathrm{FFN}_{\mathrm{ReLU}}$ モデルと同一である。その結果は著しく悪く、事前学習中に dropout を用いたことが原因だと考えている。[Raf19] が測定した実行間の標準偏差も掲載する。

<span id="table-02"></span>

![表 2。GLUE 言語理解ベンチマークの開発集合での結果。](../../papers/glu-variants/table-02.png)

**表 2。** GLUE 言語理解ベンチマーク [Wan18d]（開発集合）。

<span id="table-03"></span>

![表 3。SuperGLUE 言語理解ベンチマークの開発集合での結果。](../../papers/glu-variants/table-03.png)

**表 3。** SuperGLUE 言語理解ベンチマーク [Wan19h]（開発集合）。

<span id="table-04"></span>

![表 4。SQuAD v1.1 の開発集合での結果。](../../papers/glu-variants/table-04.png)

**表 4。** SQuAD [Raj16] v1.1（開発集合）。

<span id="section-4"></span>

## 4 結論

GLU 層のファミリーを拡張し、Transformer での使用を提案した。転移学習設定では、新しい変種は事前学習に用いるノイズ除去目的でより良いパープレキシティを示すほか、多くの下流言語理解タスクでもより良い結果をもたらすようである。これらのアーキテクチャは実装が容易で、明らかな計算上の欠点はない。これらのアーキテクチャがなぜ機能するかについて説明はできず、ほかのすべてと同じく、その成功を神の慈悲によるものと考える。

[+1]: ML の公平性のためでもある。

[+2]: 32-core TPUv2 cluster 上で、各学習ステップには約 0.15 秒を要した。

[+3]: これは、異なるタスクで別々に fine-tuning した [Raf19] とは異なる。簡単のため、fine-tuning は 1 回だけ行うことにした。
