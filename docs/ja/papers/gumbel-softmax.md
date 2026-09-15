---
title: 'Categorical Reparameterization with Gumbel-Softmax'
createTime: 2026/09/15 11:11:42
permalink: /ja/papers/gumbel-softmax/
---

> [Eric Jang](https://evjang.com/)、[Shixiang Gu](https://sites.google.com/view/gugurus/home) [+internship]、[Ben Poole](https://cs.stanford.edu/~poole/) [+internship]。2016 年 11 月 3 日に arXiv へ初回投稿、現行版は v5。ICLR 2017 の会議論文として発表された。[Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144v5)。<a href="/paper/gumbel-softmax.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。[TeX ソース](https://export.arxiv.org/e-print/1611.01144v5)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

[+internship]: Google Brain でのインターンシップ中に行われた研究。

## 概要

カテゴリ変数は、世界の離散構造を表現するうえで自然な選択肢である。しかし、サンプルを通じて逆伝播できないため、確率的ニューラルネットワークでカテゴリ潜在変数が使われることは少ない。本研究では、カテゴリ分布からの微分不可能なサンプルを、新しい Gumbel-Softmax 分布からの微分可能なサンプルに置き換える効率的な勾配推定器を提案する。この分布には、カテゴリ分布へ滑らかにアニーリングできるという本質的な性質がある。カテゴリ潜在変数を用いた構造化出力予測および教師なし生成モデリングのタスクで、Gumbel-Softmax 推定器が最先端の勾配推定器を上回り、半教師あり分類を大幅に高速化できることを示す。

<span id="section-1"></span>

## 1 はじめに

離散確率変数を含む確率的ニューラルネットワークは、教師なし学習、言語モデリング、注意機構、強化学習で現れる分布を表現するための強力な手法である。たとえば離散変数は、互いに異なる意味クラス [Kin14a]、画像領域 [Xu15]、記憶位置 [Gra14a, Gra16] に対応する確率的潜在表現の学習に使われてきた。離散表現は連続表現より解釈しやすいことが多く [Che16k]、計算効率も高い [Rae16]。

しかし、離散変数を含む確率的ネットワークの学習は難しい。逆伝播アルゴリズムはパラメータ勾配を効率よく計算できる一方で、微分不可能な層には適用できないからである。従来の確率的勾配推定の研究では、Monte Carlo 分散削減法を加えたスコア関数推定器 [Pai12, Mni14, Gu16, Gre13]、または Bernoulli 変数向けのバイアス付き経路微分推定器 [Ben13] が主に扱われてきた。しかし、カテゴリ変数に特化して定式化された勾配推定器は存在しない。本研究の貢献は次の 3 点である。

1. 単体上の連続分布であり、カテゴリサンプルを近似でき、再パラメータ化トリックによってパラメータ勾配を容易に計算できる Gumbel-Softmax を導入する。
2. Bernoulli 変数とカテゴリ変数の両方で、Gumbel-Softmax がすべての単一サンプル勾配推定器を上回ることを実験で示す。
3. 未観測のカテゴリ潜在変数について高コストな周辺化を行わずに、この推定器を使って半教師ありモデル（たとえば [Kin14a]）を効率よく学習できることを示す。

本論文の実用的な成果は、ニューラルネットワークに組み込んで標準的な逆伝播で学習できる、カテゴリ変数向けの単純で微分可能な近似サンプリング機構である。

<span id="section-2"></span>

## 2 Gumbel-Softmax 分布

まず、カテゴリ分布からのサンプルを近似できる、単体上の連続分布である Gumbel-Softmax 分布を定義する。$z$ をクラス確率 $\pi_1,\pi_2,...\pi_k$ のカテゴリ変数とする。以下では、カテゴリサンプルを $(k-1)$ 次元単体 $\Delta^{k-1}$ の頂点上にある $k$ 次元 one-hot ベクトルとして符号化すると仮定する。これにより、これらのベクトルの要素ごとの平均 $\mathbb{E}_p[z]=\left[\pi_1,...,\pi_k\right]$ などを定義できる。

Gumbel-Max トリック [Gum54, Mad14] を用いると、クラス確率 $\pi$ のカテゴリ分布からサンプル $z$ を単純かつ効率よく抽出できる。

<span id="equation-01"></span>

$$
z=\mathrm{one\_hot}\left(\argmax_i\left[g_i+\log\pi_i\right]\right)
$$

ここで $g_1...g_k$ は $\mathrm{Gumbel}(0,1)$ から抽出した i.i.d サンプルである [+1]。$\argmax$ の連続かつ微分可能な近似として softmax 関数を使い、次の $k$ 次元サンプルベクトル $y\in\Delta^{k-1}$ を生成する。

<span id="equation-02"></span>

$$
y_i=\frac{\exp((\log(\pi_i)+g_i)/\tau)}{\sum_{j=1}^k\exp((\log(\pi_j)+g_j)/\tau)}\qquad\mathrm{for}\ i=1,...,k.
$$

[+1]: $\mathrm{Gumbel}(0,1)$ 分布からのサンプリングには逆変換サンプリングを利用できる。$u\sim\mathrm{Uniform}(0,1)$ を抽出し、$g=-\log(-\log(u))$ を計算する。

Gumbel-Softmax 分布の密度（導出は[第 7 節](#section-7)）は次のとおりである。

<span id="equation-03"></span>

$$
p_{\pi,\tau}(y_1,...,y_k)=\Gamma(k)\tau^{k-1}\left(\sum_{i=1}^k\pi_i/y_i^\tau\right)^{-k}\prod_{i=1}^k\left(\pi_i/y_i^{\tau+1}\right)
$$

この分布は [Mad16] によって独立に発見され、concrete 分布と呼ばれている。softmax 温度 $\tau$ が $0$ に近づくと、Gumbel-Softmax 分布のサンプルは one-hot になり、Gumbel-Softmax 分布はカテゴリ分布 $p(z)$ と同一になる。

<span id="figure-01"></span>

![温度ごとの Gumbel-Softmax の期待値とサンプル](../../papers/gumbel-softmax/figure-01.png)

**図 1。** Gumbel-Softmax 分布は、離散的な one-hot 符号化カテゴリ分布と連続カテゴリ密度の間を補間する。（a）低温（$\tau=0.1,\tau=0.5$）では、Gumbel-Softmax 確率変数の期待値は、同じ logits を持つカテゴリ確率変数の期待値に近づく。温度が上がると（$\tau=1.0,\tau=10.0$）、期待値はカテゴリ上の一様分布に収束する。（b）$\tau\to0$ のとき、Gumbel-Softmax 分布からのサンプルはカテゴリ分布からのサンプルと同一である。温度が高くなると Gumbel-Softmax サンプルは one-hot ではなくなり、$\tau\to\infty$ で一様になる。

<span id="section-2-1"></span>

### 2.1 Gumbel-Softmax 推定器

Gumbel-Softmax 分布は $\tau>0$ で滑らかであり、パラメータ $\pi$ に関して明確に定義された勾配 $\partial y/\partial\pi$ を持つ。したがって、カテゴリサンプルを Gumbel-Softmax サンプルで置き換えると、逆伝播によって勾配を計算できる（[第 3.1 節](#section-3-1)を参照）。学習時に微分不可能なカテゴリサンプルを微分可能な近似で置き換えるこの手続きを、Gumbel-Softmax 推定器と呼ぶ。

Gumbel-Softmax サンプルは微分可能だが、温度がゼロでない場合、対応するカテゴリ分布からのサンプルと同一ではない。学習には、小さな温度と大きな温度の間にトレードオフがある。温度が小さいとサンプルは one-hot に近いが勾配の分散は大きく、温度が大きいとサンプルは滑らかだが勾配の分散は小さい（[図 1](#figure-01)）。実際には、高い温度から始め、小さいがゼロではない温度までアニーリングする。

実験では、softmax 温度 $\tau$ はさまざまなスケジュールでアニーリングしても良好に機能した。$\tau$ が固定スケジュールでアニーリングされるのではなく学習されるパラメータなら、この方式はエントロピー正則化 [Sze16, Per16] と解釈できる。このとき Gumbel-Softmax 分布は、学習過程で提案サンプルの「確信度」を適応的に調節できる。

<span id="section-2-2"></span>

### 2.2 Straight-Through Gumbel-Softmax 推定器

one-hot ベクトルの連続緩和は、隠れ表現の学習や系列モデリングなどの問題に適している。離散値のサンプリングが必要な状況（たとえば強化学習の離散行動空間や量子化圧縮）では、$\argmax$ で $y$ を離散化する一方、逆向きパスでは $\nabla_\theta z\approx\nabla_\theta y$ として連続近似を使う。[Ben13] のバイアス付き経路微分推定器を思わせるため、これを Straight-Through（ST）Gumbel 推定器と呼ぶ。ST Gumbel-Softmax では、温度 $\tau$ が高い場合でもサンプルを疎にできる。

<span id="section-3"></span>

## 3 関連研究

本節では、離散変数に対する既存の確率的勾配推定法を概観する（[図 2](#figure-02)）。分布がパラメータ $\theta$ に依存する離散確率変数 $z$ と、コスト関数 $f(z)$ を含む確率的計算グラフ [Sch15a] を考える。目的は期待コスト $L(\theta)=\mathbb{E}_{z\sim p_\theta(z)}[f(z)]$ を勾配降下で最小化することであり、そのためには $\nabla_\theta\mathbb{E}_{z\sim p_\theta(z)}[f(z)]$ を推定する必要がある。

<span id="figure-02"></span>

![確率的計算グラフにおける 5 種類の勾配推定法](../../papers/gumbel-softmax/figure-02.png)

**図 2。** 確率的計算グラフにおける勾配推定。（1）$x(\theta)$ が決定論的かつ微分可能なら、$\nabla_\theta f(x)$ は逆伝播で計算できる。（2）サンプラー関数には明確に定義された勾配がないため、確率的ノード $z$ があると逆伝播できない。（3）スコア関数推定器とその変種（NVIL、DARN、MuProp、VIMCO）は、代理損失 $\hat{f}\log p_\theta(z)$ に沿って逆伝播することで $\nabla_\theta f(x)$ の不偏推定値を得る。ここで $\hat{f}=f(x)-b$、$b$ は分散削減用のベースラインである。（4）主に Bernoulli 変数向けに開発された Straight-Through 推定器は、$\nabla_\theta z\approx1$ と近似する。（5）Gumbel-Softmax は、$z$ を近似する連続分布 $y$ の経路微分推定器である。再パラメータ化によって勾配は $f(y)$ から $\theta$ へ流れる。学習の進行に伴って、$y$ を one-hot カテゴリ変数へアニーリングできる。

<span id="section-3-1"></span>

### 3.1 経路微分勾配推定器

再パラメータ化可能な分布では、サンプル $z$ をパラメータ $\theta$ と独立確率変数 $\epsilon$ の決定論的関数 $g$ として、$z=g(\theta,\epsilon)$ のように計算できる。すると、確率的ノードを経由せずに $f$ から $\theta$ への経路勾配を計算できる。

<span id="equation-04"></span>

$$
\frac{\partial}{\partial\theta}\mathbb{E}_{z\sim p_\theta}\left[f(z))\right]=\frac{\partial}{\partial\theta}\mathbb{E}_\epsilon\left[f(g(\theta,\epsilon))\right]=\mathbb{E}_{\epsilon\sim p_\epsilon}\left[\frac{\partial f}{\partial g}\frac{\partial g}{\partial\theta}\right]
$$

たとえば正規分布 $z\sim\mathcal{N}(\mu,\sigma)$ は $\mu+\sigma\cdot\mathcal{N}(0,1)$ と書き換えられるため、$\partial z/\partial\mu$ と $\partial z/\partial\sigma$ を容易に計算できる。この再パラメータ化トリックは、連続潜在変数を持つ変分オートエンコーダを逆伝播で学習する際によく使われる [Kin14, Rez14]。[図 2](#figure-02) に示すように、Gumbel-Softmax 推定器の構築でもこのトリックを利用する。

$z$ が再パラメータ化できない場合でも、バイアス付き経路微分推定器を利用できる。一般に、$m$ を確率的サンプルの微分可能な代理として、$\nabla_\theta z\approx\nabla_\theta m(\theta)$ と近似できる。平均パラメータが $\theta$ の Bernoulli 変数に対し、Straight-Through（ST）推定器 [Ben13] は $m=\mu_\theta(z)$ と近似し、$\nabla_\theta m=1$ とする。$k=2$（Bernoulli）の場合、ST Gumbel-Softmax は [Chu16] が提案した傾きアニーリング付き Straight-Through 推定器に似ているが、傾きの決定には hard sigmoid ではなく softmax を使う。[Rol16] は、各二値潜在変数が連続混合モデルをパラメータ化する別の方法を検討している。連続変数を通して逆伝播し、二値変数を周辺化することで再パラメータ化勾配を得る。

ST 推定器の制約の一つは、サンプルに依存しない平均について逆伝播すると、順向きパスと逆向きパスの間に不一致が生じ、分散が大きくなる可能性があることだ。各サンプル $y$ が対応する離散サンプル $z$ の微分可能な代理であるため、Gumbel-Softmax ではこの問題を回避できる。

<span id="section-3-2"></span>

### 3.2 スコア関数に基づく勾配推定器

スコア関数推定器（SF、REINFORCE [Wil92] および尤度比推定器 [Gly90] とも呼ばれる）は、恒等式 $\nabla_\theta p_\theta(z)=p_\theta(z)\nabla_\theta\log p_\theta(z)$ を使って、次の不偏推定器を導く。

<span id="equation-05"></span>

$$
\nabla_\theta\mathbb{E}_z\left[f(z)\right]=\mathbb{E}_z\left[f(z)\nabla_\theta\log p_\theta(z)\right]
$$

SF が要求するのは、$p_\theta(z)$ が $\theta$ について連続であることだけであり、$f$ やサンプル $z$ を通した逆伝播は必要ない。しかし、SF は分散が大きく、そのため収束が遅い。とくに SF の分散はサンプルベクトルの次元数に対して線形に増えるため [Rez14]、カテゴリ分布への利用は特に難しい。

学習信号 $f$ から制御変量 $b(z)$ を引き、その解析的期待値 $\mu_b=\mathbb{E}_z\left[b(z)\nabla_\theta\log p_\theta(z)\right]$ を加え戻すと、不偏性を保ったままスコア関数推定器の分散を削減できる。

<span id="equation-06"></span>
<span id="equation-07"></span>

$$
\begin{aligned}
\nabla_\theta\mathbb{E}_z\left[f(z)\right]&=\mathbb{E}_z\left[f(z)\nabla_\theta\log p_\theta(z)+(b(z)\nabla_\theta\log p_\theta(z)-b(z)\nabla_\theta\log p_\theta(z))\right]\\
&=\mathbb{E}_z\left[(f(z)-b(z))\nabla_\theta\log p_\theta(z)\right]+\mu_b
\end{aligned}
$$

制御変量を利用する近年の確率的勾配推定器を簡単にまとめる。これらの手法の詳細については [Gu16] を参照されたい。

- NVIL [Mni14] は、（1）$f$ の移動平均 $\bar{f}$ によって学習信号を中心化するベースラインと、（2）$f-\bar{f}$ に適合させた 1 層ニューラルネットワークで計算する入力依存のベースライン（中心化された学習信号そのものに対する制御変量）の 2 つを使う。最後に、分散正規化では学習信号を $\max(1,\sigma_f)$ で割る。ここで $\sigma_f^2$ は $\mathrm{Var}[f]$ の移動平均である。
- DARN [Gre13] は $b=f(\bar{z})+f^\prime(\bar{z})(z-\bar{z})$ を使い、このベースラインは $f(\bar{z})$ からの $f(z)$ の一次 Taylor 近似に対応する。Bernoulli 変数では $z$ を $\frac{1}{2}$ とするが、推定式の補正項 $\mu_b$ を無視するため、$f$ が二次関数でなければ推定器にはバイアスがある。
- MuProp [Gu16] もベースラインを一次 Taylor 展開 $b=f(\bar{z})+f^\prime(\bar{z})(z-\bar{z})$ としてモデル化し、$\mu_b=f^\prime(\bar{z})\nabla_\theta\mathbb{E}_z\left[z\right]$ とする。離散サンプリングを通した逆伝播を避けるため、ベースラインの計算と関連する勾配の導出では、$f(z)$ の代わりに平均場近似 $f_{\mathrm{MF}}(\mu_\theta(z))$ を使う。
- VIMCO [Mni16] は多重サンプル目的向けの勾配推定器であり、他のサンプルの平均 $b=\frac{1}{m}\sum_{j\neq i}f(z_j)$ を使って、各サンプル $z_i\in z_{1:m}$ のベースラインを構成する。ここでは単一サンプル目的の推定器を比較するため、実験から VIMCO を除外する。ただし Gumbel-Softmax は多重サンプル目的へ容易に拡張できる。

<span id="section-3-3"></span>

### 3.3 半教師あり生成モデル

半教師あり学習では、ラベル付きデータ $(x,y)\sim\mathcal{D}_L$ とラベルなしデータ $x\sim\mathcal{D}_U$ の両方から学習する。ここで $x$ は観測（画像など）、$y$ は対応するラベル（意味クラスなど）である。半教師あり分類に対し、[Kin14a] は潜在状態が Gaussian「スタイル」変数 $z$ とカテゴリ「意味クラス」変数 $y$ の同時分布である変分オートエンコーダ（VAE）を提案している（[図 6](#figure-06)、[第 6 節](#section-6)）。VAE 目的は、生成モデルの下での観測の対数尤度に対する変分下界を最大化し、識別ネットワーク $q_\phi(y|x)$、推論ネットワーク $q_\phi(z|x,y)$、生成ネットワーク $p_\theta(x|y,z)$ を end-to-end で学習する。ラベル付きデータではクラス $y$ が観測されるため、推論は $z\sim q(z|x,y)$ についてのみ行う。ラベル付きデータの変分下界は次式で与えられる。

<span id="equation-08"></span>

$$
\log p_\theta(x,y)\geq-\mathcal{L}(x,y)=\mathbb{E}_{z\sim q_\phi(z|x,y)}\left[\log p_\theta(x|y,z)\right]-\mathrm{KL}\left[q(z|x,y)\|p_\theta(y)p(z)\right]
$$

ラベルなしデータでは、カテゴリ分布を再パラメータ化できないため問題が生じる。[Kin14a] は全クラスについて $y$ を周辺化することで対処する。そのためラベルなしデータでも、各 $y$ に対して $q_\phi(z|x,y)$ 上で推論する。ラベルなしデータの下界は次のとおりである。

<span id="equation-09"></span>
<span id="equation-10"></span>

$$
\begin{aligned}
\log p_\theta(x)\geq-\mathcal{U}(x)&=\mathbb{E}_{z\sim q_\phi(y,z|x)}\left[\log p_\theta(x|y,z)+\log p_\theta(y)+\log p(z)-q_\phi(y,z|x)\right]\\
&=\sum_y q_\phi(y|x)\left(-\mathcal{L}(x,y)+\mathcal{H}(q_\phi(y|x))\right)
\end{aligned}
$$

完全な最大化目的は次のとおりである。

<span id="equation-11"></span>

$$
\mathcal{J}=\mathbb{E}_{(x,y)\sim\mathcal{D}_L}\left[-\mathcal{L}(x,y)\right]+\mathbb{E}_{x\sim\mathcal{D}_U}\left[-\mathcal{U}(x)\right]+\alpha\cdot\mathbb{E}_{(x,y)\sim\mathcal{D}_L}\left[\log q_\phi(y|x)\right]
$$

ここで $\alpha$ は、生成目的と識別目的の間のスカラーのトレードオフである。

この方法の制約の一つは、モデルのクラス数が多いと、$k$ 個の全クラス値についての周辺化が現実的でないほど高コストになることだ。$D,I,G$ をそれぞれ $q_\phi(y|x)$、$q_\phi(z|x,y)$、$p_\theta(x|y,z)$ からのサンプリングに要する計算コストとすると、教師なし目的の学習では順向き／逆向きステップごとに $\mathcal{O}(D+k(I+G))$ が必要になる。これに対し Gumbel-Softmax では、単一サンプル勾配推定のために $y\sim q_\phi(y|x)$ を通して逆伝播でき、学習ステップあたりのコストは $\mathcal{O}(D+I+G)$ となる。学習速度の実験的な比較は[図 5](#figure-05) に示す。

<span id="section-4"></span>

## 4 実験結果

最初の一連の実験では、Gumbel-Softmax と ST Gumbel-Softmax を、Score-Function（SF）、DARN、MuProp、Straight-Through（ST）、Slope-Annealed ST という他の確率的勾配推定器と比較する。各推定器を（1）構造化出力予測、（2）生成モデルの変分学習という 2 つのタスクで評価する。学習と評価には固定二値化した MNIST データセットを使う。これは確率的勾配推定器の評価で一般的な方法である [Sal08, Lar11]。

学習率は $\{3\mathrm{e}{-5},1\mathrm{e}{-5},3\mathrm{e}{-4},1\mathrm{e}{-4},3\mathrm{e}{-3},1\mathrm{e}{-3}\}$ から選ぶ。MNIST 検証セットを使って各推定器に最適な学習率を選び、テストセットでの性能を報告する。Gumbel-Softmax 分布から抽出するサンプルは学習中には連続だが、評価時には one-hot ベクトルへ離散化する。また、SF、DARN、MuProp で競争力のある性能を得るには分散正規化が必要だった。二値（Bernoulli）ニューラルネットワークには sigmoid 活性化関数を使い、カテゴリ変数には softmax 活性化を使った。モデルはモメンタム $0.9$ の確率的勾配降下法で学習した。

<span id="section-4-1"></span>

### 4.1 確率的二値ネットワークによる構造化出力予測

構造化出力予測の目的は、$28\times28$ の MNIST 数字の上半分（$14\times28$）から下半分を予測することである。これは確率的二値ネットワーク（SBN）の学習に広く使われるベンチマークである [Rai14, Gu16, Mni16]。この条件付き生成モデルの最小化目的は、尤度目的の重要度サンプリング推定 $\mathbb{E}_{h\sim p_\theta(h_i|x_{\mathrm{upper}})}\left[\frac{1}{m}\sum_{i=1}^m\log p_\theta(x_{\mathrm{lower}}|h_i)\right]$ である。学習時には $m=1$、評価時には $m=1000$ を使う。

各 200 ユニットの隠れ層を 2 層持つ SBN を学習した。これは、200 個の Bernoulli 変数（$392$-$200$-$200$-$392$ と表記）、または二値化された活性を持つ 20 個のカテゴリ変数（各 10 クラス、$392$-$(20\times10)$-$(20\times10)$-$392$ と表記）に対応する。

[図 3](#figure-03) に示すように、ST Gumbel-Softmax は Bernoulli 変数では他の推定器と同程度で、カテゴリ変数では他の推定器を上回る。一方、Gumbel-Softmax は Bernoulli 変数とカテゴリ変数の両方で他の推定器を上回る。このタスクでは softmax 温度をアニーリングする必要はなく、固定値 $\tau=1$ を使った。

<span id="figure-03"></span>

![Bernoulli およびカテゴリ SBN のテスト損失](../../papers/gumbel-softmax/figure-03.png)

**図 3。** 二値化 MNIST の構造化出力予測タスクにおける確率的二値ネットワークのテスト損失（負の対数尤度）。（a）Bernoulli 潜在変数（$392$-$200$-$200$-$392$）、（b）カテゴリ潜在変数（$392$-$(20\times10)$-$(20\times10)$-$392$）を用いる。

<span id="section-4-2"></span>

### 4.2 変分オートエンコーダによる生成モデリング

二値 MNIST 画像の生成モデルを学習することを目的として、変分オートエンコーダ [Kin14] を学習する。実験では、潜在変数を 200 個の Bernoulli 変数または 20 個のカテゴリ変数（$20\times10$）からなる単一隠れ層としてモデル化した。学習目的には Gumbel-Softmax 事前分布ではなく、学習されたカテゴリ事前分布を使う。そのため、サンプルが離散でなければ、学習時の最小化目的は変分下界ではなくなる。実際には、この目的を温度アニーリングと組み合わせて最適化しても、検証セットとテストセットにおける実際の変分下界は最小化された。構造化出力予測タスクと同じく、評価には $m=1000$ の多重サンプル下界を使う。

温度は大域学習ステップ $t$ のスケジュール $\tau=\max(0.5,\exp(-rt))$ でアニーリングし、$N$ ステップごとに $\tau$ を更新する。$N\in\{500,1000\}$ と $r\in\{1\mathrm{e}{-5},1\mathrm{e}{-4}\}$ はハイパーパラメータであり、検証セット上で最良の推定器を選び、そのテスト性能を報告する。

[図 4](#figure-04) に示すように、ST Gumbel-Softmax はカテゴリ変数で他の推定器を上回り、Gumbel-Softmax は Bernoulli 変数とカテゴリ変数の両方で他の推定器を大幅に上回る。

<span id="figure-04"></span>

![Bernoulli およびカテゴリ VAE のテスト損失](../../papers/gumbel-softmax/figure-04.png)

**図 4。** 二値化 MNIST VAE のテスト損失（負の変分下界）。（a）Bernoulli 潜在変数（$784$-$200$-$784$）、（b）カテゴリ潜在変数（$784$-$(20\times10)$-$200$）を用いる。

<span id="table-01"></span>

![7 種類の勾配推定器による SBN と VAE の損失](../../papers/gumbel-softmax/table-01.png)

**表 1。** Gumbel-Softmax 推定器は、Bernoulli およびカテゴリ潜在変数で他の推定器を上回る。構造化出力予測（SBN）タスクでは、数値は入力画像の負の対数尤度（nat、低いほど良い）に対応する。VAE タスクでは、数値は対数尤度に対する負の変分下界（nat、低いほど良い）に対応する。

<span id="section-4-3"></span>

### 4.3 生成的半教師あり分類

Gumbel-Softmax 推定器を二値 MNIST データセットの半教師あり分類に適用する。元の周辺化に基づく推論手法 [Kin14a] と、Gumbel-Softmax および ST Gumbel-Softmax による単一サンプル推論を比較する。

各 10 クラスへ均等に分配されたラベル付きサンプル 100 個と、ラベルなしサンプル 50,000 個からなるデータセットで学習した。ラベルなしサンプルは minibatch ごとに動的に二値化した。識別モデル $q_\phi(y|x)$ と推論モデル $q_\phi(z|x,y)$ は、それぞれ ReLU 活性化関数を持つ 3 層畳み込みニューラルネットワークとして実装した。生成モデル $p_\theta(x|y,z)$ は、ReLU 活性化を持つ 4 層の転置畳み込みネットワークである。実験の詳細は[第 6 節](#section-6)に示す。

推定器を複数の $\alpha=\{0.1,0.2,0.3,0.8,1.0\}$ について学習・評価し、各推定器についてテストセットで最良のラベルなし分類結果を選び、[表 2](#table-02) に報告した。$\tau=\max(0.5,\exp(-3\mathrm{e}{-5}\cdot t))$ のアニーリングスケジュールを使い、2000 ステップごとに更新した。

[Kin14a] では、$y$ を周辺化し、$q_\phi(z|x,y)$ からのサンプリングに再パラメータ化トリックを使って潜在状態を推論する。しかし、この方法の計算コストはクラス数に対して線形に増える。Gumbel-Softmax では同時分布 $q_\phi(y,z|x)$ からの単一サンプルを直接通して逆伝播できるため、生成性能や分類性能を損なわずに学習を大幅に高速化できる（[表 2](#table-02)、[図 5](#figure-05)）。

<span id="table-02"></span>

![周辺化および Gumbel-Softmax 推定器の ELBO と精度](../../papers/gumbel-softmax/table-02.png)

**表 2。** 二値化 MNIST データセットの画像分類に適用した場合、$y$ の周辺化と単一サンプル変分推論は同程度の性能を示す [Lar11]。テストセットのラベルなしデータについて、変分下界と画像分類精度を報告する。

[図 5](#figure-05) では、Gumbel-Softmax と周辺化のスケーリングをカテゴリのクラス数に対して比較する。これらの実験では、ランダムに生成したラベルを持つ MNIST 画像を使う。Gumbel-Softmax 推定器によるモデルの学習は、$10$ クラスで $2\times$、$100$ クラスで $9.9\times$ 高速である。

<span id="figure-05"></span>

![半教師あり VAE の学習速度と MNIST アナロジー](../../papers/gumbel-softmax/figure-05.png)

**図 5。** Gumbel-Softmax を使うと事後分布 $q_\phi(y|x)$ からのサンプルを通して逆伝播でき、多数のクラスを持つタスクに適用可能な半教師あり学習手法が得られる。（a）半教師あり VAE で、Gumbel-Softmax と周辺化 [Kin14a] の学習速度（steps/sec）を比較する。評価には GTX Titan X® GPU を使用した。（b）各行でスタイル変数 $z$、各列でクラス変数 $y$ を変えて生成した MNIST アナロジーの可視化。

<span id="section-5"></span>

## 5 考察

本研究の主な貢献は、再パラメータ化可能な Gumbel-Softmax 分布と、それに対応してカテゴリ分布の低分散な経路微分勾配を得られる推定器である。Gumbel-Softmax と Straight-Through Gumbel-Softmax は構造化出力予測および変分オートエンコーダのタスクで有効であり、Bernoulli 潜在変数とカテゴリ潜在変数の両方で既存の確率的勾配推定器を上回ることを示した。最後に、Gumbel-Softmax は離散潜在変数の推論を大幅に高速化する。

## 謝辞

有益な議論とフィードバックをいただいた Luke Vilnis、Vincent Vanhoucke、Luke Metz、David Ha、Laurent Dinh、George Tucker、Subhaneil Lahiri に心より感謝する。

<span id="section-6"></span>

## 6 半教師あり分類モデル

[図 6](#figure-06) と[図 7](#figure-07) に、半教師あり分類実験で用いたアーキテクチャを示す（[第 4.3 節](#section-4-3)）。

<span id="figure-06"></span>

![半教師あり生成モデルと推論モデルの計算グラフ](../../papers/gumbel-softmax/figure-06.png)

**図 6。** [Kin14a] が提案した半教師あり生成モデル。（a）生成モデル $p_\theta(x|y,z)$ は、潜在 Gaussian「スタイル」変数 $z$ とカテゴリクラス変数 $y$ から画像を合成する。（b）推論モデル $q_\phi(y,z|x)$ は、$x$ が与えられたときに潜在状態 $y,z$ をサンプリングする。Gaussian $z$ は再パラメータ化可能なので、そのパラメータについて微分できる。従来手法では $y$ が観測されない場合、VAE 目的の学習に $y$ の全値についての周辺化が必要だった。（c）Gumbel-Softmax は $y$ を再パラメータ化し、確率的ノードを経由せずに $y$ を通した逆伝播も可能にする。

<span id="figure-07"></span>

![分類、推論、生成の畳み込みアーキテクチャ](../../papers/gumbel-softmax/figure-07.png)

**図 7。** （a）分類 $q_\phi(y|x)$、（b）推論 $q_\phi(z|x,y)$、（c）生成 $p_\theta(x|y,z)$ モデルのネットワークアーキテクチャ。これらのネットワークの出力は、サンプリング元となるカテゴリ分布、Gaussian 分布、Bernoulli 分布をパラメータ化する。

<span id="section-7"></span>

## 7 Gumbel-Softmax 分布の密度の導出

確率 $\pi_1,...,\pi_k$、温度 $\tau$ の Gumbel-Softmax 分布の確率密度関数を導出する。まず logits $x_i=\log\pi_i$ と Gumbel サンプル $g_1,...,g_k$ を定義する。ここで $g_i\sim\mathrm{Gumbel}(0,1)$ である。Gumbel-Softmax からのサンプルは次のように計算できる。

<span id="equation-12"></span>

$$
y_i=\frac{\exp((x_i+g_i)/\tau)}{\sum_{j=1}^k\exp((x_j+g_j)/\tau)}\qquad\mathrm{for}\ i=1,...,k
$$

<span id="section-7-1"></span>

### 7.1 中心化 Gumbel 密度

softmax 演算の正規化によって自由度が一つ失われるため、Gumbel サンプル $g$ から Gumbel-Softmax サンプル $y$ への写像は可逆ではない。これを補うため、softmax の前に最後の要素 $(x_k+g_k)/\tau$ を引く等価なサンプリング過程を定義する。

<span id="equation-13"></span>

$$
y_i=\frac{\exp((x_i+g_i-(x_k+g_k))/\tau)}{\sum_{j=1}^k\exp((x_j+g_j-(x_k+g_k))/\tau)}\qquad\mathrm{for}\ i=1,...,k
$$

この等価なサンプリング過程の密度を導くため、まず次式に対応する「中心化」多変量 Gumbel 密度を導出する。

<span id="equation-14"></span>

$$
u_i=x_i+g_i-(x_k+g_k)\qquad\mathrm{for}\ i=1,...,k-1
$$

ここで $g_i\sim\mathrm{Gumbel}(0,1)$ である。尺度パラメータ $\beta=1$、平均 $\mu$ の Gumbel 分布が $z$ で取る確率密度は $f(z,\mu)=e^{\mu-z-e^{\mu-z}}$ である。最後の Gumbel サンプル $g_k$ を周辺化することで、この分布の密度を計算できる。

$$
\begin{aligned}
p(u_1,...,u_{k-1})&=\int_{-\infty}^\infty dg_k\,p(u_1,...,u_k|g_k)p(g_k)\\
&=\int_{-\infty}^\infty dg_k\,p(g_k)\prod_{i=1}^{k-1}p(u_i|g_k)\\
&=\int_{-\infty}^\infty dg_k\,f(g_k,0)\prod_{i=1}^{k-1}f(x_k+g_k,x_i-u_i)\\
&=\int_{-\infty}^\infty dg_k\,e^{-g_k-e^{-g_k}}\prod_{i=1}^{k-1}e^{x_i-u_i-x_k-g_k-e^{x_i-u_i-x_k-g_k}}
\end{aligned}
$$

$v=e^{-g_k}$ と変数変換すると $dv=-e^{-g_k}dg_k$、$dg_k=-dv\,e^{g_k}=dv/v$ となる。さらに表記を簡潔にするため $u_k=0$ と定義する。

<span id="equation-15"></span>
<span id="equation-16"></span>
<span id="equation-17"></span>
<span id="equation-18"></span>

$$
\begin{aligned}
p(u_1,...,u_{k,-1})&=\delta(u_k=0)\int_0^\infty dv\,\frac{1}{v}ve^{x_k-v}\prod_{i=1}^{k-1}ve^{x_i-u_i-x_k-ve^{x_i-u_i-x_k}}\\
&=\exp\left(x_k+\sum_{i=1}^{k-1}(x_i-u_i)\right)\left(e^{x_k}+\sum_{i=1}^{k-1}e^{x_i-u_i}\right)^{-k}\Gamma(k)\\
&=\Gamma(k)\exp\left(\sum_{i=1}^k(x_i-u_i)\right)\left(\sum_{i=1}^ke^{x_i-u_i}\right)^{-k}\\
&=\Gamma(k)\left(\prod_{i=1}^k\exp(x_i-u_i)\right)\left(\sum_{i=1}^k\exp(x_i-u_i)\right)^{-k}
\end{aligned}
$$

<span id="section-7-2"></span>

### 7.2 Gumbel-Softmax への変換

中心化 Gumbel 分布からのサンプル $u_1,...,u_{k,-1}$ が与えられたとき、決定論的変換 $h$ を適用し、Gumbel-Softmax からのサンプルの最初の $k-1$ 座標を得られる。

<span id="equation-19"></span>

$$
y_{1:k-1}=h(u_{1:k-1}),\qquad h_i(u_{1:k-1})=\frac{\exp(u_i/\tau)}{1+\sum_{j=1}^{k-1}\exp(u_j/\tau)}\quad\forall i=1,...,k-1
$$

$\sum_{i=1}^k y_i=1$ なので、最初の $k-1$ 座標が与えられると最後の座標の確率 $y_k$ は固定される。

<span id="equation-20"></span>

$$
y_k=\left(1+\sum_{j=1}^{k-1}\exp(u_j/\tau)\right)^{-1}=1-\sum_{j=1}^{k-1}y_j
$$

したがって、最初の $k-1$ 変数だけに変数変換公式を適用して、Gumbel-Softmax からのサンプルの確率を計算できる。

<span id="equation-21"></span>

$$
p(y_{1:k})=p\left(h^{-1}(y_{1:k-1})\right)\det\left(\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}\right)
$$

そのため、さらに $h$ の逆関数とその Jacobian 行列式という 2 つの要素を計算する必要がある。$h$ の逆関数は次式となる。

<span id="equation-22"></span>

$$
h^{-1}(y_{1:k-1})=\tau\times\left(\log y_i-\log\left(1-\sum_{j=1}^{k-1}y_j\right)\right)=\tau\times(\log y_i-\log y_k)
$$

その Jacobian は

<span id="equation-23"></span>

$$
\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}=\tau\times\left(\mathrm{diag}\left(\frac{1}{y_{1:k-1}}\right)+\frac{1}{y_k}\right)=
\begin{bmatrix}
\frac{1}{y_1}+\frac{1}{y_k}&\frac{1}{y_k}&\dots&\frac{1}{y_k}\\
\frac{1}{y_k}&\frac{1}{y_2}+\frac{1}{y_k}&\dots&\frac{1}{y_k}\\
\vdots&\vdots&\ddots&\vdots\\
\frac{1}{y_k}&\frac{1}{y_k}&\dots&\frac{1}{y_{k-1}}+\frac{1}{y_k}
\end{bmatrix}
$$

次に Jacobian の行列式を計算する。

<span id="equation-24"></span>
<span id="equation-25"></span>
<span id="equation-26"></span>

$$
\begin{aligned}
\det\left(\frac{\partial h^{-1}(y_{1:k-1})}{\partial y_{1:k-1}}\right)&=\tau^{k-1}\det\left(\left(I+\frac{1}{y_k}ee^\top\mathrm{diag}(y_{1:k-1})\right)\mathrm{diag}\left(\frac{1}{y_{1:k-1}}\right)\right)\\
&=\tau^{k-1}\left(1+\frac{1-y_k}{y_k}\right)\prod_{j=1}^{k-1}y_j^{-1}\\
&=\tau^{k-1}\prod_{j=1}^ky_j^{-1}
\end{aligned}
$$

ここで $e$ は $k-1$ 次元の全要素が 1 のベクトルであり、恒等式 $\det(A\,B)=\det(A)\det(B)$、$\det(\mathrm{diag}(x))=\prod_i x_i$、$\det(I+uv^\top)=1+u^\top v$ を使った。

中心化 Gumbel の密度（[式 15](#equation-15)）、$h$ の逆関数（[式 22](#equation-22)）、その Jacobian 行列式（[式 26](#equation-26)）を使い、変数変換公式（[式 21](#equation-21)）へ代入する。

<span id="equation-27"></span>
<span id="equation-28"></span>

$$
\begin{aligned}
p(y_1,..,y_k)&=\Gamma(k)\left(\prod_{i=1}^k\exp(x_i)\frac{y_k^\tau}{y_i^\tau}\right)\left(\sum_{i=1}^k\exp(x_i)\frac{y_k^\tau}{y_i^\tau}\right)^{-k}\tau^{k-1}\prod_{i=1}^ky_i^{-1}\\
&=\Gamma(k)\tau^{k-1}\left(\sum_{i=1}^k\exp(x_i)/y_i^\tau\right)^{-k}\prod_{i=1}^k\left(\exp(x_i)/y_i^{\tau+1}\right)
\end{aligned}
$$
