---
title: 'Speculative Decoding'
createTime: 2026/08/05 00:26:41
permalink: /ja/papers/speculative-decoding/
---

> [Yaniv Leviathan](https://yanivle.github.io/)、[Matan Kalman](https://research.google/people/108191/)、[Yossi Matias](http://www.math.tau.ac.il/~matias/)。 arXiv 初回投稿日： 2022 年 11 月 30 日；現行版は v2。 ICML 2023 の oral paper として発表、 *Proceedings of Machine Learning Research* 202:19274-19286 に収録。[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)。[原 PDF](/paper/speculative-decoding.pdf)。[DOI](https://doi.org/10.48550/arXiv.2211.17192)。[TeX ソース](https://export.arxiv.org/e-print/2211.17192v2)。正確な印刷レイアウトと参考文献については原 PDF を正本とする。

## 要約

Transformer のような大規模自己回帰モデルの推論は遅い。 $K$ 個のトークンをデコードするには、モデルを $K$ 回直列に実行する必要がある。本論文では、複数のトークンを並列に計算することで、 *出力を一切変えずに*自己回帰モデルからのサンプリングを高速化するアルゴリズム、 *speculative decoding* を導入する。本手法の基礎となる観察は二つある。（1）難しい言語モデリングの問題には、より効率的なモデルで十分に近似できる簡単な部分問題が含まれることが多い。（2） speculative execution と新しいサンプリング手法を用いれば、大規模モデルを近似モデルの出力に対して並列に実行できるため、分布を変えずに正確なデコードを高速化し、複数のトークンを同時に生成できる可能性がある。本手法は再学習やアーキテクチャ変更なしに既存の off-the-shelf モデルを高速化できる。 T5-XXL で検証したところ、標準 T5X 実装に対して同一出力のまま 2X-3X の高速化を示した。

<span id="figure-01"></span>

![無条件言語モデリングにおける speculative decoding の例](../../papers/speculative-decoding/figure-01.png)

**図 1.** 無条件言語モデリングの場合における本手法。各行はアルゴリズムの 1 回の反復を表す。緑色のトークンは、近似モデルが提案し、ターゲットモデルが受理した候補である。ここでの近似モデルは lm1b 上で 8k トークンを用いて学習した、パラメータ数 6M の GPT 類似 Transformer デコーダである。ターゲットモデルは同じ設定で学習した、パラメータ数 97M の GPT 類似 Transformer デコーダである。赤色のトークンは拒否された候補、青色のトークンはその修正を表す。例えば最初の行では、ターゲットモデルを 1 回しか実行せずに 5 トークンを生成した。

## 1 はじめに

大規模自己回帰モデル、とりわけ大規模 Transformer [Vas17a] は小規模モデルよりはるかに高い能力を持つ。近年、テキストや画像の領域で GPT-3 [Bro20a]、 LaMDA [Tho22]、 Parti [Yu22]、 PaLM [Cho22] などがそのことを何度も示してきた。しかし、これら大規模モデルの 1 回のデコードは小規模モデルの 1 回のデコードより著しく遅い。さらに悪いことに、この処理は直列で行われる。 $K$ 個のトークンをデコードするには、モデルを $K$ 回直列に実行する必要がある。

大規模自己回帰モデル、特に大規模 Transformer の推論を高速化するために、いくつかの手法が開発されてきた。一部の手法は、 *すべての*入力について一様に推論コストを削減する [Hin15, Jas21, Hub16a, So21, Sha19a]。別の手法は、推論ステップには難易度の差があるという観察に基づく。あるステップには非常に大きなモデルが必要だが、別のステップはより効率的なモデルで十分に近似できる。これらの*適応計算*手法 [Han21, Suk19a, Sch21a, Sca20a, Bap20, Elb19, Sch20] は、簡単な推論ステップに使う計算資源を減らす。多くの手法は実用上きわめて有効だが、通常はモデルアーキテクチャと学習手順の変更、モデルの再学習を必要とし、同一出力を維持しない。

推論ステップには「難しい」ものと「簡単な」ものがある。この観察は本研究の主要な動機でもある。さらに、大規模モデルの推論は算術演算ではなく、メモリ帯域幅と通信がボトルネックであることが多い。そのため、追加の計算資源が利用できる場合がある。そこで、適応的に計算量を使う手法を補う方法として、並行性を高めることを提案する。具体的には、モデルアーキテクチャを変更せず、学習手順を変更またはモデルを再学習せず、モデル出力の分布も変えずに推論を高速化できる。これを実現するのが*speculative execution* である。

speculative execution [Bur85, Hen12] はプロセッサで一般的な最適化手法であり、タスクが実際に必要かどうかを検証するのと並行して、そのタスクを実行することで並行性を高める。よく知られた例は分岐予測である。 speculative execution を有効にするには、実行が必要となる可能性の高いタスクを提案する効率的な仕組みが必要である。本研究では、タスクがある確率で必要になる確率的設定へ speculative execution を一般化する。これを Transformer のような自己回帰モデルのデコードに適用すると、より効率的な*近似モデル*から生成結果をサンプリングし、遅い*ターゲットモデル*の speculative prefix として用いる。新しいサンプリング手法である*speculative sampling* により、これら speculative task が受理される確率を最大化しつつ、システムの出力がターゲットモデル単独の場合と同じ分布を持つことを保証する。例えば、[図 1](#figure-01) の 38 トークンからなる文は、より小さく効率的な近似モデル（6M パラメータ）を使うことで、より大きなターゲットモデル（97M パラメータ）の直列実行 9 回だけで生成されたが、その文が生成される確率は変わらない。

本手法をさまざまなタスクとモデル規模で分析する。 97M パラメータの lm1b 学習済み GPT 類似モデルによる無条件生成、 11B パラメータの T5-XXL モデルによる英独翻訳とニュース記事要約、 137B パラメータの LaMDA モデルによる対話タスクである。本手法を実装し、 T5-XXL の実際の walltime を堅牢な T5X 実装 [Rob22] と比較した。出力を変えることなく、すぐに使える 2X-3X のレイテンシ改善が得られた（[4 節](#section-04)）。

本手法は実際のプロダクション環境で使いやすく、新しいモデルの学習を必要とせず、出力も変えない。したがって、メモリ帯域幅がボトルネックで計算資源を利用できる一般的な状況では、 Transformer のような自己回帰モデルからのサンプリングを高速化するための有力なデフォルトになりうる。

主な貢献は二つである。（1）確率的設定への speculative execution の一般化と、 *speculative sampling* と呼ぶ新しいサンプリング手法。（2）モデルアーキテクチャ、学習方法、出力分布を変えずに自己回帰モデルのデコードを高速化できる、 *speculative decoding* と呼ぶデコード機構。

## 2 Speculative Decoding

### 2.1 概要

$M_{p}$ を高速化したいターゲットモデルとし、 $p(x_{t}|x_{<t})$ を接頭辞 $x_{<t}$ に対してモデルが与える分布とする。同じタスクのより効率的な近似モデルを $M_{q}$ とし、 $q(x_{t}|x_{<t})$ を接頭辞 $x_{<t}$ に対してそのモデルが与える分布と表す [+1]。中心となる考え方は次の三段階である。（1）より効率的なモデル $M_{q}$ で $\gamma\in\mathbb{Z}^{+}$ 個の続きを生成する（このパラメータの最適な選び方は [3.5 節](#section-03-05) を参照）。（2）ターゲットモデル $M_{p}$ で、 $M_{q}$ が出したすべての候補とそれぞれの確率を*並列に*評価し、同一の分布につながり*うる*候補をすべて受理する。（3）調整済み分布から追加のトークンをサンプリングして、最初に拒否された候補を修正する。すべての候補が受理された場合には、追加で 1 トークンを加える。このため、ターゲットモデル $M_{p}$ を 1 回並列実行するごとに少なくとも 1 個の新しいトークンが生成される。よって、最悪の場合でもターゲットモデルの直列実行回数は単純な自己回帰法より多くならない。一方で、 $M_{q}$ が $M_{p}$ をどれだけよく近似するかに応じて、最大 $\gamma+1$ 個の新しいトークンを生成できる。

### 2.2 Standardized Sampling

<span id="section-02-02"></span>

argmax、 top-k、 nucleus、 temperature の設定など、サンプリングには多くの方法とパラメータがある。一般的な実装は logits レベルでそれらを異なる形で扱うが、どれも調整済み確率分布からの標準サンプリングとして簡単に表せる。例えば argmax sampling は、分布の最大でない要素をゼロにして正規化することに等しい。したがって、確率分布からの標準サンプリングだけを扱い、ほかのサンプリング方式をすべてこの枠組みに変換できる。以下では、 $p(x)$ と $q(x)$ はそれぞれ $M_{p}$ と $M_{q}$ から得られ、サンプリング方法に合わせて調整された分布であると仮定する。

### 2.3 Speculative Sampling

<span id="section-02-03"></span>

$x\sim p(x)$ をサンプリングする代わりに、 $x\sim q(x)$ をサンプリングする。 $q(x)\leq p(x)$ ならそのまま残す。 $q(x)>p(x)$ なら、確率 $1-\frac{p(x)}{q(x)}$ でサンプルを拒否し、調整済み分布 $p^{\prime}(x)=\mathrm{norm}(\max(0,p(x)-q(x)))$ から $x$ をもう一度サンプリングする。任意の分布 $p(x)$ と $q(x)$ について、この方法でサンプリングした $x$ は確かに $x\sim p(x)$ であることを容易に示せる（[A.1 節](#section-a-01) を参照）。

条件付き $\mathrm{prefix}$ について $M_{q}$ を実行して分布 $q(x)$ を得たとする。そこからトークン $x_{1}\sim q(x)$ をサンプリングできる。続いて、 $\mathrm{prefix}$ 上で $M_{p}$ を実行して分布 $p(x)$ を計算すると同時に、 $\mathrm{prefix}+[x_{1}]$ 上で $M_{p}$ を実行して次のトークン $x_{2}$ の分布を speculative に計算する。両方の計算が終わったら上述のとおりに処理する。 $x_{1}$ が拒否されたら $x_{2}$ の計算を捨て、調整済み分布から $x_{1}$ を再サンプリングする。 $x_{1}$ が受理されたら両方のトークンを残す。[アルゴリズム 1](#algorithm-01) はこの考え方を一般化し、一度に 1 個から $\gamma+1$ 個のトークンをサンプリングする。

<span id="algorithm-01"></span>

**アルゴリズム 1：SpeculativeDecodingStep。**

- **入力：** $M_{p},M_{q},\mathrm{prefix}$。
- **サンプリング：** $M_{q}$ から自己回帰的に $\gamma$ 個の候補 $x_{1,\ldots,\gamma}$ をサンプリングする：
  - **反復：** $i=1$ から $\gamma$ まで：
    - $q_{i}(x)\leftarrow M_{q}(\mathrm{prefix}+[x_{1},\ldots,x_{i-1}])$。
    - $x_{i}\sim q_{i}(x)$。
- **並列実行：** $M_{p}$ を並列に実行する：
  - $p_{1}(x),\ldots,p_{\gamma+1}(x)\leftarrow M_{p}(\mathrm{prefix}),\ldots,M_{p}(\mathrm{prefix}+[x_{1},\ldots,x_{\gamma}])$。
- **決定：** 受理された候補数 $n$ を求める：
  - $r_{1}\sim U(0,1),\dots,r_{\gamma}\sim U(0,1)$。
  - $n\leftarrow\min(\{i-1\mid 1\leq i\leq\gamma,r_{i}>\frac{p_{i}(x)}{q_{i}(x)}\}\cup\{\gamma\})$。
- **調整：** 必要に応じて $M_{p}$ の分布を調整する：
  - $p^{\prime}(x)\leftarrow p_{n+1}(x)$。
  - **条件：** $n<\gamma$ の場合：
    - $p^{\prime}(x)\leftarrow\mathrm{norm}(\max(0,p_{n+1}(x)-q_{n+1}(x)))$。
- **返却：** $M_{p}$ の 1 トークンと $M_{q}$ の $n$ トークンを返す：
  - $t\sim p^{\prime}(x)$。
  - **返却：** $\mathrm{prefix}+[x_{1},\ldots,x_{n},t]$。

## 3 分析

### 3.1 生成トークン数

<span id="section-03-01"></span>

ターゲットモデルの直列呼び出し回数の削減率、言い換えれば [アルゴリズム 1](#algorithm-01) の 1 回の実行で生成される期待トークン数を分析する。

<span id="definition-03-01"></span>

**定義 3.1。** 接頭辞 $x_{<t}$ が与えられたとき、 *acceptance rate $\beta_{x_{<t}}$* とは [2.3 節](#section-02-03) に従う speculative sampling により $x_{t}\sim q(x_{t}|x_{<t})$ を受理する確率である [+2]。

$E(\beta)$ は $M_{q}$ が $M_{p}$ をどれだけよく近似するかの自然な尺度である。 $\beta$ が i.i.d. であるという単純化した仮定を置き、 $\alpha=E(\beta)$ と表す。このとき、[アルゴリズム 1](#algorithm-01) を 1 回実行して生成されるトークン数は、成功確率 $1-\alpha$、上限 $\gamma+1$ の打ち切り幾何変数である。[アルゴリズム 1](#algorithm-01) による期待生成トークン数は [式 1](#equation-01) を満たす。[図 2](#figure-02) を参照。

<span id="equation-01"></span>

$$
E(\#\ \mathrm{generated}\ \mathrm{tokens})=\frac{1-\alpha^{\gamma+1}}{1-\alpha}
$$

<span id="figure-02"></span>

![gamma ごとの 1 反復あたりの期待生成トークン数](../../papers/speculative-decoding/figure-02.png)

**図 2.** さまざまな $\gamma$ に対する、[アルゴリズム 1](#algorithm-01) が生成する期待トークン数の $\alpha$ による変化。

### 3.2 $\alpha$ の計算

接頭辞と二つのモデル $M_{p}$、$M_{q}$ が与えられたときの $\alpha$ を計算する簡単な式を導く。まず、自然な発散 $D_{\mathrm{LK}}$ を次のように定義する：

**定義 3.2。** $D_{\mathrm{LK}}(p,q)=\sum_{x}|p(x)-M(x)|=\sum_{x}|q(x)-M(x)|$、ただし $M(x)=\frac{p(x)+q(x)}{2}$。

<span id="lemma-03-03"></span>

**補題 3.3。** $D_{\mathrm{LK}}(p,q)=1-\sum_{x}\min(p(x),q(x))$

::: details 証明
$D_{\mathrm{LK}}(p,q)=\sum_{x}|p(x)-M(x)|=\sum_{x}\frac{|p-q|}{2}=1-\sum_{x}\frac{p+q-|p-q|}{2}=1-\sum_{x}\min(p(x),q(x))$
:::

[補題 3.3](#lemma-03-03) から直ちに次を得る：

**系 3.4。** $D_{\mathrm{LK}}(p,q)$ は $[0,1]$ 上の対称な発散である。<br>
$D_{\mathrm{LK}}(p,q)=0\iff p=q$。<br>
$D_{\mathrm{LK}}(p,q)=1\iff p$ と $q$ の台は互いに素である。

<span id="theorem-03-05"></span>

**定理 3.5。** $\beta=1-D_{\mathrm{LK}}(p,q)$

::: details 証明
$$
\beta=E_{x\sim q(x)}
\begin{cases}
1,&q(x)\leq p(x),\\
\dfrac{p(x)}{q(x)},&q(x)>p(x)
\end{cases}
=E_{x\sim q(x)}\min\left(1,\frac{p(x)}{q(x)}\right)=\sum_{x}\min(p(x),q(x))
$$
:::

最後に：

<span id="corollary-03-06"></span>

**系 3.6。** $\alpha=1-E(D_{\mathrm{LK}}(p,q))=E(\min(p,q))$

実験で観測された $\alpha$ の値は [表 3](#table-03) を参照。

### 3.3 Walltime 改善

<span id="section-03-03"></span>

i.i.d. の仮定の下で、本手法はターゲットモデルの呼び出し回数を $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 倍削減することを示した。一般の speculative execution、特に本アルゴリズムは、増加する並行性を支える十分な計算資源を仮定する（[3.4 節](#section-03-04)）。 walltime の分析では、 $M_{p}$ の $\gamma+1$ 回の評価を並列に実行しても walltime が増えないと仮定する。総 walltime の改善を求めるため、近似モデル $M_{q}$ を実行するコストを考える。

**定義 3.7。** *cost coefficient* $c$ を、 $M_{q}$ の 1 回の実行時間と $M_{p}$ の 1 回の実行時間の比とする。

$\alpha$ はモデルとタスクに内在する性質であるのに対し、 $c$ の値はハードウェア構成とソフトウェア実装の詳細に依存する。実験では $M_{q}$ は通常 $M_{p}$ より二、三桁小さく、 $c$ は常に $0.05$ 未満で、多くの場合 0 に無視できるほど近かった。

<span id="theorem-03-08"></span>

**定理 3.8。** [アルゴリズム 1](#algorithm-01) による総 walltime の期待改善率は $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)({\gamma}c+1)}$ である。

::: details 証明
$M_{p}$ の 1 ステップ実行のコストを $T$ とする。[アルゴリズム 1](#algorithm-01) の 1 回の実行は、近似モデル $M_{q}$ を $\gamma$ 回実行し、 $M_{p}$ を 1 回実行するため、 $Tc\gamma+T$ のコストがかかる。また [式 1](#equation-01) によれば、平均して $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 個のトークンを生成する。よって [アルゴリズム 1](#algorithm-01) で 1 トークンを生成する総期待コストは $\frac{(c\gamma+1)(1-\alpha)}{1-\alpha^{\gamma+1}}T$ となる。標準デコードで 1 トークンを生成するコストは $T$ なので、結論を得る。
:::

[定理 3.8](#theorem-03-08) は十分に長い生成を仮定している。例えば $M_{p}$ は少なくとも 1 回実行するため、改善率は生成トークン数で上限づけられる。

**系 3.9。** $\alpha>c$ なら、改善を得る $\gamma$ が存在し、改善率は少なくとも $\frac{1+\alpha}{1+c}$ である。

::: details 証明
ある $\gamma$ で改善するなら、任意の $0<\gamma^{*}<\gamma$ でも改善する。したがって、本手法が改善をもたらす条件は [定理 3.8](#theorem-03-08) で $\gamma=1$ として評価でき、 $\frac{1-\alpha^{2}}{(1-\alpha)(c+1)}=\frac{1+\alpha}{1+c}$ を得る。
:::

### 3.4 算術演算回数

<span id="section-03-04"></span>

[アルゴリズム 1](#algorithm-01) は $M_{p}$ を $\gamma+1$ 回並列に実行するため、 *同時に行われる*算術演算数は $\gamma+1$ 倍になる。[アルゴリズム 1](#algorithm-01) は 1 回の実行で最大 $\gamma+1$ 個のトークンを生成するので、 *総*算術演算数は標準デコードより多くなる可能性がある。 $M_{q}$ のサンプルを受理した場合、増加した並行性は「無料」であり、総演算数は増えない [+3]。しかし候補を拒否した場合には計算が無駄になる。本手法が総算術演算回数に与える影響を分析する。

**定義 3.10。** $\hat{c}$ を、近似モデル $M_{q}$ の 1 トークンあたりの算術演算数とターゲットモデル $M_{p}$ のそれとの比とする。

**定理 3.11。** [アルゴリズム 1](#algorithm-01) の総演算回数の期待増加係数は $\frac{(1-\alpha)({\gamma}\hat{c}+\gamma+1)}{1-\alpha^{\gamma+1}}$ である。

::: details 証明
$\hat{T}$ を標準デコードのベースラインが 1 トークンあたりに行う算術演算数、すなわち $M_{p}$ の 1 回の実行に要する演算数とする。[アルゴリズム 1](#algorithm-01) の 1 反復は、 $M_{q}$ の $\gamma$ 回の実行と $M_{p}$ の $\gamma+1$ 回の並列実行により、 $\hat{T}\hat{c}\gamma+\hat{T}(\gamma+1)$ 回の演算を要する。これを [アルゴリズム 1](#algorithm-01) が生成する期待トークン数、すなわち [式 1](#equation-01) と $\hat{T}$ で除けば、結論を得る。
:::

$\alpha$ が低いと算術演算数の増加は大きく、逆も成り立つ。 Transformer デコーダについては、[アルゴリズム 1](#algorithm-01) の総算術演算数（$M_{q}$ の実行を除く）は、 *同じ規模の Transformer エンコーダの 1 回の実行によって上から抑えられる*。

総算術演算数とは異なり、本手法では総メモリアクセス数を減らせる。具体的には、ターゲットモデルの重みと KV cache は [アルゴリズム 1](#algorithm-01) の各実行につき 1 回だけ読めばよい。そのため、[式 1](#equation-01) に従い、それらを読むメモリアクセス数は $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ 倍に縮小する。

<span id="figure-03"></span>

![cost coefficient ごとの最適な gamma](../../papers/speculative-decoding/figure-03.png)

**図 3.** さまざまな $c$ に対する、 $\alpha$ の関数としての最適な $\gamma$。

### 3.5 $\gamma$ の選択

<span id="section-03-05"></span>

$c$ と $\alpha$ が与えられ、十分な計算資源があると仮定する（[3.4 節](#section-03-04) を参照）。最適な $\gamma$ は、[定理 3.8](#theorem-03-08) の walltime 改善式 $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)({\gamma}c+1)}$ を最大化する値である。 $\gamma$ は整数なので、数値的に容易に求められる。[図 3](#figure-03) を参照。

[表 1](#table-01) と [図 4](#figure-04) は、 $c=\hat{c}=0$ と仮定したときの、さまざまな $\alpha$ と $\gamma$ に対する推論速度と総算術演算数のトレードオフを示す。[図 5](#figure-05) は簡略化したトレース図である。

<span id="table-01"></span>

![alpha と gamma ごとの総演算回数と推論速度](../../papers/speculative-decoding/table-01.png)

**表 1.** $c=\hat{c}=0$ を仮定した、さまざまな $\gamma$ と $\alpha$ に対する総算術演算回数およびベースラインに対する推論速度。

<span id="figure-04"></span>

![gamma ごとの速度向上係数と算術演算回数の増加](../../papers/speculative-decoding/figure-04.png)

**図 4.** さまざまな $\gamma$ に対する、 $\alpha$ の関数としての速度向上係数と算術演算回数の増加。

<span id="figure-05"></span>

![完全な encoder-decoder Transformer stack の簡略トレース](../../papers/speculative-decoding/figure-05.png)

**図 5.** 完全な encoder-decoder Transformer stack の簡略化されたトレース図。上段は $\gamma=7$ の speculative decoding を示す。各 $M_{p}$ 呼び出し（紫色ブロック）の前に $M_{q}$ を 7 回呼び出す（青色ブロック）。左側の黄色ブロックは $M_{p}$ の encoder 呼び出し、橙色ブロックは $M_{q}$ の encoder 呼び出しである。同様に中段は $\gamma=3$ の speculative decoding、下段は標準デコードを示す。

$\beta$ は一定ではないため、 $\alpha$ に基づいて $\gamma$ を一つに固定する代わりに、 $\beta$ の値を予測し、[アルゴリズム 1](#algorithm-01) の実行中に $\gamma$ を変化させれば、さらに改善できる可能性がある。追加改善係数の上限を得るため、 $\gamma$ に関する oracle があると仮定する。そのとき $E(\#\ \mathrm{generated}\ \mathrm{tokens})=\frac{1}{1-\alpha}$ となる。典型的な $c$ と $\alpha$ で計算資源が無限にあると仮定すると、強化された walltime 改善率は固定 $\gamma$ の場合より最大 $\sim$60% 高くなりうる。これは今後の課題とする [+4]。

### 3.6 近似モデル

<span id="section-03-06"></span>

speculative sampling、したがって speculative decoding は、近似モデル $M_{q}$ をどのように選んでも、制約なしに同一の出力分布を保証する（[A.1 節](#section-a-01) を参照）。実験では主に既存の off-the-shelf の小型 Transformer を近似モデルとしてテストした。さらに、ターゲットモデル $M_{p}$ と同じアーキテクチャで、同じ確率標準化を用いる近似モデルだけをテストした。この設定では、 $M_{q}$ を $M_{p}$ よりおよそ二桁小さくすると、 $\alpha$ と $c$ のバランスが取れ、通常は最良の性能を示した（[定理 3.8](#theorem-03-08)）。

別の近似モデルの種類は、 $c\approx 0$ である*negligible-cost model*、すなわちターゲットモデルに比べコストが無視できる近似モデルである。この場合、期待 walltime 改善率は $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ となり、上から $\frac{1}{1-\alpha}$ で抑えられる（$\gamma$ が大きいと等号に近づく）。興味深い negligible-cost approximation model の一つは n-gram model であり、評価はテーブル参照だけで済む。実験（[4.2 節](#section-04-02)）では、このような単純な n-gram model でも非ゼロの $\alpha$ が得られた。例えば英独翻訳で、 $M_{p}$ を T5-XXL 11B、 $M_{q}$ を単純な bigram model とすると、 $\alpha\approx 0.2$ が得られ、 $\gamma=3$ で $1.25$X の推論速度改善につながる。

ほかの単純な heuristic も negligible-cost approximation model として使える。例えば要約タスクや chat-like interface のように長い系列が繰り返されやすい場合 [+5]、一致する prefix が見つかったときに文脈からトークンをそのままコピーする近似モデルは高い $\alpha$ をもたらす可能性がある。これらのパラメータを持たない近似モデルには、プロダクション環境でより簡単に導入できるという利点もある。

speculative decoding に利用できる別の近似モデルは、[Ste18] のような非自己回帰モデルである。この場合、[アルゴリズム 1](#algorithm-01) の自己回帰ループの代わりに、非自己回帰モデルを 1 回呼び出すだけでよい。

理論的な観点で主に興味深い最後の例は、トークンをランダムに選ぶ近似モデルである。これはすべてのモデル $M_{p}$ に対して、ごく小さいとはいえ何らかの改善を保証する。

## 4 実験

<span id="section-04"></span>

### 4.1 経験的 Walltime 改善

<span id="section-04-01"></span>

本アルゴリズムを実装し、 T5-XXL を高速化する T5X codebase の実装と比較する。

**設定。** 標準的な encoder-decoder T5 version 1.1 model [Raf20] を、 T5 論文の二つのタスクでテストする。（1） WMT EnDe で fine-tune した英独翻訳、（2） CNN/DM で fine-tune したテキスト要約である。両方のタスクで $M_{p}$ に T5-XXL（11B）を用いる。近似モデル $M_{q}$ には既存の T5-large（800M）、 T5-base（250M）、 T5-small（77M） [Raf20] をテストする。すべてのモデルで既存の checkpoint を用いる。 1 台の TPU-v4 上で batch size 1 とし、 argmax sampling（temp=0）と standard sampling（temp=1）の両方について walltime 改善を測定する。

**結果。** [表 2](#table-02) に本手法の実験結果を示す。テストした近似モデルの中では、 $c$ と $\alpha$ のバランスがよい T5-small（77M）が最大の speedup を与える。予想どおり、 $\alpha$ は近似モデルの規模とともに増加する。興味深いことに、 argmax sampling（temp=0）では $\alpha$ と walltime 改善がより大きい。翻訳タスクでは 2.6X（temp=1）と 3.4X（temp=0）、要約タスクではやや低い 2.3X（temp=1）と 3.1X（temp=0）の speedup を観測した。実験結果は理論予測とよく一致し、実装の詳細に起因するばらつきがある（[A.3 節](#section-a-03) を参照）。

<span id="table-02"></span>

![T5-XXL の推論高速化に関する実験結果](../../papers/speculative-decoding/table-02.png)

**表 2.** T5-XXL 11B モデルの推論を高速化した実験結果。

### 4.2 経験的 $\alpha$ 値

<span id="section-04-02"></span>

本手法を実装したのは T5 だけだが、さまざまなタスク、 sampling method、 target model $M_{p}$、近似モデル $M_{q}$ について $\alpha$ の値を測定した。具体的には、以下の各設定について、 $M_{p}$ が生成した 10K トークン上で [系 3.6](#corollary-03-06) の期待値を評価した。

**GPT-like（97M params）。** lm1b 上で学習した decoder-only Transformer model を無条件言語生成でテストする [Che13a]。このモデルは Gelu activation [Hen16] を用いる GPT 類似 Transformer decoder である。 $M_{q}$ には、 dim 256、 dim feed-forward 1024、 2 layers、 4 attention heads の 6M パラメータ Transformer decoder model と、単純な unigram および bigram model を試した。 $M_{p}$ は 97M パラメータであり、 dim 768、 dim feed-forward 3072、 12 layers、 12 attention heads を持つ。すべてのモデルで 8k トークンの Bert tokenization [Dev19] を用いた。

**LaMDA（137B params）。** decoder-only LaMDA model を対話タスクでテストした [Tho22]。既存の LaMDA 137B checkpoint を $M_{p}$ とし、 LaMDA 8B、 LaMDA 2B、 LaMDA 100M を $M_{q}$ に用いた。

T5-XXL（11B params） model の設定については [4.1 節](#section-04-01) を参照。

[表 3](#table-03) はテストしたケースの $\alpha$ 値をまとめる。ターゲットモデルより二桁程度小さい近似モデルは、 0.5 から 0.9 の $\alpha$ 値を生む傾向がある。すべてのモデルで、調整済み分布が鋭いほど $\alpha$ 値も高い。単純な unigram や bigram による近似でも、無視できない $\alpha$ 値が得られる。例えば英独翻訳で bigram model の $\alpha$ は 0.2 であり、この場合 $c=0$ なので 1.25X の速度改善が得られる。この単純な近似モデルとしては驚くほど高いが、 T5-small を近似モデルに使う場合の speedup よりは低い。

<span id="table-03"></span>

![target model、近似モデル、sampling setting ごとの経験的 alpha](../../papers/speculative-decoding/table-03.png)

**表 3.** さまざまな target model $M_{p}$、近似モデル $M_{q}$、sampling setting に対する経験的 $\alpha$ 値。 T=0 と T=1 はそれぞれ argmax と standard sampling を表す [+6]。

## 5 関連研究

大規模モデルからの推論の効率は広く研究されている [Deh21]。多くの手法は、一般の大規模モデル、とりわけ Transformer のような自己回帰モデルからの推論を高速化することを目指す。蒸留 [Hin15]、 sparsification [Jas21]、 quantization [Hub16a]、 architecture modification [So21, Sha19a] など、すべてのトークンで推論を効率化しようとする手法は多い。本手法により近いのは、問題の難しさに応じて計算量を変える adaptive computation method [Han21] である。例には、入力の一部だけに attention を向けること [Suk19a] と early exit [Sch21a, Sca20a, Bap20, Elb19, Sch20] がある。特に Wisdom of Committees [Sch20] は off-the-shelf の小型モデルを利用するが、 adaptive computation approach であるため、停止する時点を heuristic で決める。その結果、ターゲットモデルと同一の出力を保証できない。一般に adaptive computation method は、モデル内部または補助モデルにより、計算を近道できる時点を学習する。これらの手法は通常、推論時間と算術演算の両方を節約するが、 architecture と training procedure の変更、 custom model の学習または既存モデルの再学習を必要とする。通常はモデル出力も変える。上記の多くの手法はメモリ対算術演算の比を改善するが、比が依然として高い場合には、これらの手法と speculative decoding を併用すると有効かもしれない。

二つの先行手法は、自己回帰モデルのデコードを高速化するために speculative execution を用いる。 Blockwise Parallel Decoding [Ste18] は本研究と同様に複数のトークンを並列にデコードする。ただし greedy decoding（temperature=0）だけをサポートし、一般の確率的設定を扱わない。また custom model の追加学習を必要とし、同一出力の保証ではなく downstream task quality の維持に焦点を当てる。 Shallow Aggressive Decoding（SAD） [Sun21] も本研究と同様に複数トークンを並列にデコードする。本研究とは異なり、 SAD は入力を出力にコピーすることだけをサポートし、一般の近似モデルをサポートしない。そのため、文法誤り訂正のように入力と出力が非常に似ている場合だけに適する。さらに Blockwise Parallel Decoding と同様に、 SAD は一般の確率的 sampling setting をサポートしない。

本研究を最初に公開した後、 speculative decoding の独立した実装 [Che23] が Chinchilla 70B で同様の 2X-2.5X 改善を示した。

## 6 議論

*speculative sampling* を提示した。これは効率的な*stochastic speculative execution*、すなわち確率的設定における speculative execution を可能にする。 Transformer のような自己回帰モデルからのデコードに与える影響を *speculative decoding* を通じて分析した。十分な計算資源があれば、一般的な最適化実装である T5X と比べ、実際に意味のある 2X-3X の speedup が得られることを示した。

speculative execution 全般、特に speculative decoding の制約の一つは、レイテンシを改善するために並行性を高める代償として、算術演算数が増えることである。したがって、追加の計算資源が利用できない構成では本手法は役に立たない。しかし追加の計算資源が利用できる一般的な場合、例えばメモリ帯域幅がボトルネックの場合、本手法は速度改善に加えて重要な利点を持つ。モデルアーキテクチャを変更せず、再学習を必要とせず、そして最も重要なことに、 *出力分布が同じままであることを保証する*。本手法は実装が容易で、 custom scheme を開発・評価せずに off-the-shelf model を使って推論を高速化できる。

今後の研究方向はいくつかある。特に、 speculative decoding と beam search の互換性をさらに調べることが重要である（[A.4 節](#section-a-04) を参照）。既存の off-the-shelf 近似モデルで本手法は大きな speedup をもたらすが、[3.6 節](#section-03-06) にある custom approximation model を用いると、さらに大きな改善を得られる可能性がある。例えば custom size、非自己回帰モデル、さまざまな heuristic などの custom architecture、または $M_{p}$ の soft target を使う標準的な distillation や $\alpha$ を直接最適化する $M_{q}$ などの custom training procedure である。近似モデル自体をさらに高速なモデルで加速する階層型アルゴリズムも興味深く、より能力の高い近似モデルを可能にするかもしれない。本研究では推論全体を通じて近似モデルと候補数 $\gamma$ を固定したが、推論中にそれらを変えれば追加の改善が得られる可能性がある（[3.5 節](#section-03-05)）。実験では、近似モデルが生成した分布に、ターゲットモデルで必要なものと同じ standardization を常に行った（[2.2 節](#section-02-02)）。異なる transformation を適用すればさらに改善できる可能性がある。 speculative decoding はテキスト modality だけでテストしたが、画像など他の domain でも有効に働く可能性があり、実験する価値がある。

最後に、 *stochastic speculative execution* と *speculative sampling* は、自己回帰モデルからの *speculative decoding* の範囲外でも役立つ場合がある。例えば遅い二つの関数 $f(x)$ と $g(y)$ があり、 $f(x)$ が $g$ の入力をサンプリングする分布を生成するとする。本手法を使えば $f$ と $g$ を並列に実行できる。この設定は、例えば物理シミュレーションや、 $f$ が行動の分布を出力する大規模モデルで $g$ が world simulation となる強化学習で生じうる。これはさらに調べる価値がある。

## 謝辞

LaMDA に関するあらゆる支援と、論文中の LaMDA 図の計算を手伝ってくれた YaGuang Li に特別な感謝を述べる。また、 XLA について有益な洞察と支援をくれた Blake Hechtman に感謝する。査読者の有益なコメント、 Asaf Aharoni、Reiner Pope、Sasha Goldshtein、Nadav Sherman、Eyal Segalis、Eyal Molad、Dani Valevski、Daniel Wasserman、Valerie Nygaard、Danny Vainstein、Google の LaMDA および Theta Labs チーム、そして家族にも感謝する。

## 付録 A

### A.1 Speculative Sampling の正しさ

<span id="section-a-01"></span>

::: details 証明
任意の分布 $p(x)$ と $q(x)$ に対し、 $p(x)$ と $q(x)$ から *speculative sampling* によりサンプリングされたトークンが、 $p(x)$ 単独からサンプリングしたトークンと同一に分布することを示す。 $\beta$ を acceptance probability とする（[定義 3.1](#definition-03-01)）。

$p^{\prime}(x)=\mathrm{norm}(\max(0,p(x)-q(x)))=\frac{p(x)-\min(q(x),p(x))}{\sum_{x^{\prime}}(p(x^{\prime})-\min(q(x^{\prime}),p(x^{\prime})))}=\frac{p(x)-\min(q(x),p(x))}{1-\beta}$ であることに注意する。調整済み分布 $p^{\prime}(x)$ の正規化定数は $1-\beta$ であり、最後の等式は [補題 3.3](#lemma-03-03) と [定理 3.5](#theorem-03-05) から直ちに従う。

ここで：

$$
P(x=x^{\prime})=P(\mathrm{guess}\ \mathrm{accepted},x=x^{\prime})+P(\mathrm{guess}\ \mathrm{rejected},x=x^{\prime})
$$

また：

$$
P(\mathrm{guess}\ \mathrm{accepted},x=x^{\prime})=q(x^{\prime})\min(1,\frac{p(x^{\prime})}{q(x^{\prime})})=\min(q(x^{\prime}),p(x^{\prime}))
$$

そして：

$$
P(\mathrm{guess}\ \mathrm{rejected},x=x^{\prime})=(1-\beta)p^{\prime}(x^{\prime})=p(x^{\prime})-\min(q(x^{\prime}),p(x^{\prime}))
$$

全体として：

$$
P(x=x^{\prime})=\min(p(x^{\prime}),q(x^{\prime}))+p(x^{\prime})-\min(p(x^{\prime}),q(x^{\prime}))=p(x^{\prime}).
$$

以上で示された。
:::

### A.2 Speculative Sampling と Rejection Sampling

<span id="section-a-02"></span>

Rejection sampling は、表面的には本手法に似た次の反復的サンプリング手順である：

1. $x\sim q(x)$ と $r\sim U(0,1)$ をサンプリングする。
2. $r<\frac{p(x)}{M q(x)}$ なら $x$ を返す。
3. 1 に戻る。

ここで $M=\max_{x}\frac{p(x)}{q(x)}$ である。 speculative sampling の代わりに非反復型の rejection sampling を使うこともできる。具体的には、上の 1 と 2 を行い、それ以外では*変更していない* $p(x)$ から直接サンプリングする。ただし、これは本手法よりはるかに効率が悪い。具体的には、期待受理確率 $E_{x\sim q(x)}\frac{p(x)}{M q(x)}=\sum_{x}p(x)\min_{x^{\prime}}\frac{q(x^{\prime})}{p(x^{\prime})}\leq\sum_{x}p(x)\min(1,\frac{q(x)}{p(x)})=\sum_{x}\min(p(x),q(x))=\alpha$ は、本手法における期待受理確率 $\alpha$ より（大幅に）低くなる可能性がある。

### A.3 理論予測と経験的ランタイム

<span id="section-a-03"></span>

[表 4](#table-04) は、[定理 3.8](#theorem-03-08) に基づく期待ランタイム改善と、[表 2](#table-02) の経験的に測定したランタイムを比較する。各モデルの $c$ は profiler trace に基づいて推定した。理論予測は大半の場合に測定ランタイムと一致する。大きな差異は、（1）本実装とベースライン間の最適化の違い、（2） $\beta$ が i.i.d. であるという単純化した仮定が近似にすぎないこと（[3.1 節](#section-03-01) を参照）による。

<span id="table-04"></span>

![理論上の改善係数と実測した改善係数の比較](../../papers/speculative-decoding/table-04.png)

**表 4.** 期待改善係数（Exp）と経験的に測定した改善係数（Emp）の比較。

### A.4 Beam Search への応用

<span id="section-a-04"></span>

本手法は、ある程度の性能ペナルティを伴うが beam search sampling に適用できる。元の beam width $w$ が与えられたとき、近似モデル $M_{q}$ と beam width $u\geq w$ を用いて $\gamma$ ステップの beam search を行える。次に、 $M_{p}$ を用いてすべての候補を並列にチェックする。計算予算は $M_{p}$ を $(w+u\gamma)$ 回実行することに相当する。最後に各ステップで $\mathrm{top}_{w}(M_{p})\subseteq \mathrm{top}_{u}(M_{q})$ なら、 $M_{q}$ の候補を受理し、 $M_{p}$ 単独で通常の beam search を行う場合と同一の結果を得られる。より精緻な手順なら、得られた候補が $M_{p}$ 単独の場合より高い確率を持つケースも受理できる。この設定における本手法の分析はより複雑であり、今後の課題とする。

### A.5 Lenience

[アルゴリズム 1](#algorithm-01) の強い性質は、出力分布が変わらないことを保証する点である。とはいえ、いくつかの変更を許容しつつよい保証を望むなら、さらに推論を高速化できる。同じアーキテクチャと同じ規模の二つのモデルを同じデータセットで学習しても、生成される確率分布は同一にならない。したがって、ある程度の lenience は理にかなうかもしれない。本論文では本節を除き、[アルゴリズム 1](#algorithm-01) の最も厳しい版を用い、いかなる lenience も許容していない。

lenience parameter $l\in[0,1]$ を導入し、[アルゴリズム 1](#algorithm-01) で $p(x)$ と比較する前に $q(x)$ に $l$ を掛けることができる。これでも、どのトークンも $\frac{p(x)}{l}$ より大きい確率でサンプリングされないというよい保証を保てる。例えば $l=\frac{1}{10}$ なら、どのトークンも真の確率の $10$X を超えてサンプリングされない。したがって極めてまれなトークンは極めてまれなままである。最小確率に関する保証はないため、 lenience はサンプルの多様性を損なう可能性がある。

具体的には、 lenience factor $l$ に対して次を得る：

$$
\begin{aligned}
\alpha
&=E_{x\sim q(x)}
\begin{cases}
1,&lq(x)\leq p(x),\\
\dfrac{p(x)}{lq(x)},&lq(x)>p(x)
\end{cases}\\
&=E_{x\sim q(x)}\frac{p(x)}{\max(p(x),lq(x))}
=\sum_{x}\frac{p(x)q(x)}{\max(p(x),lq(x))}\\
&=\frac{1}{l}\sum_{x}\min(p(x),lq(x))
=\sum_{x}\min\left(\frac{p(x)}{l},q(x)\right).
\end{aligned}
$$

[表 5](#table-05) は、 $M_{p}$ が T5-XXL（11B）、 $M_{q}$ が T5-small（77M）の場合の、異なる $l$ に対する $\alpha$ 値を示す。 $c=0.015$ とし、 lenience 値 1、0.5、0.3、0.1 を使うと、どのトークンも真の確率の 1X、2X、3X、10X を超えてサンプリングされないことを意味し、改善係数はそれぞれ 2.5X、3.1X、3.6X、5X となる。

<span id="table-05"></span>

![standard sampling における lenience parameter ごとの alpha](../../papers/speculative-decoding/table-05.png)

**表 5.** EnDe 翻訳タスクで $M_{p}$ が T5-XXL（11B）のとき、 standard sampling における各 $l$ の $\alpha$ 値。

temperature=0、すなわち argmax sampling を使う場合は、上のように lenience を使うことはできない。代わりに、分布を標準化する前にある程度の lenience を認められる。例えば $p(x)\leq l\cdot\max(p)$ のとき、 $M_{q}$ からサンプリングしたトークン $x$ を受理できる。この場合、 temperature=1 の場合と同様の $\alpha$ 値の経験的増加を測定する。例えば英独翻訳で、 $M_{p}$ が T5-XXL、$M_{q}$ が T5-small のとき、 lenience 値 1、0.5、0.3、0.1 に対する $\alpha$ 値は 0.75、0.75、0.8、0.87 となる。 $c=0.015$ と $\gamma=8$ を取ると、速度改善係数はそれぞれ 3.3X、3.3X、3.9X、4.9X となる [+7]。

[+1]: 文脈から接頭辞 $x_{<t}$ が明らかな場合、 $p(x_{t}|x_{<t})$ を $p(x)$ と表し、 $q(x)$ についても同様にする。

[+2]: 前と同様に、文脈から接頭辞が明らかな場合は下付き文字 $x_{<t}$ を省略する。

[+3]: $M_{q}$ のコストは無視する。

[+4]: この上界は、 oracle の予測を検証するために $M_{p}$ を実行し続けることを仮定する。検証を省略すれば上界は成り立たず、さらに大きな改善が得られる。

[+5]: 例えば、ユーザーと言語モデルがテキストやコードなどの内容を反復して編集する場合（「この物語を結末だけ変えて書き直せますか」、「この関数に X も実行させられますか」）。

[+6]: LaMDA モデルの出力は常に $\mathrm{Top}_{40}$ filter を通す。これは argmax には影響しないが、 standard sampling にはある程度影響する。

[+7]: この場合、[表 5](#table-05) に示す standard sampling の場合とは異なり、 lenience factor 0.5 は speed-up を改善しない。
