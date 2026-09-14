---
title: Consistency Models
createTime: 2026/09/14 11:58:45
permalink: /ja/papers/consistency-models/
pageClass: paper-reading
---

> [Yang Song](https://yang-song.net/)、[Prafulla Dhariwal](https://dblp.org/pid/190/7235)、[Mark Chen](https://dblp.org/pid/40/1660-3)、[Ilya Sutskever](https://dblp.org/pid/60/5276)。2023 年 3 月 2 日に arXiv へ初投稿、現行版は v2。[第 40 回 International Conference on Machine Learning 論文集](https://proceedings.mlr.press/v202/song23a.html)、PMLR 202:32211-32252、2023 年に掲載。[Consistency Models](https://arxiv.org/abs/2303.01469)。<a href="/paper/consistency-models.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2303.01469)。[TeX ソース](https://export.arxiv.org/e-print/2303.01469v2)。正確な印刷レイアウトと参考文献については原 PDF を正とする。

## 概要

拡散モデルは画像・音声・動画生成を大きく進歩させたが、反復的なサンプリング過程に依存するため生成が遅い。この制約を克服するため、ノイズをデータへ直接写像して高品質なサンプルを生成する新しいモデル群、*consistency models* を提案する。設計上、高速な 1 ステップ生成を実現しながら、マルチステップサンプリングによって計算量とサンプル品質を交換することもできる。また、画像のインペインティング、カラー化、超解像などのゼロショットデータ編集を、それらのタスクに対する明示的な訓練なしで行える。Consistency model は、事前学習済み拡散モデルを蒸留して訓練することも、完全に独立した生成モデルとして訓練することもできる。広範な実験により、1 ステップおよび少数ステップのサンプリングで既存の拡散モデル蒸留法を上回り、1 ステップ生成において CIFAR-10 で 3.55、ImageNet $64\times 64$ で 6.20 という新たな最高 FID を達成することを示す。単独で訓練した場合、consistency model は新しい生成モデル群となり、CIFAR-10、ImageNet $64\times 64$、LSUN $256\times 256$ などの標準ベンチマークで、既存の 1 ステップ非敵対的生成モデルを上回ることができる。

<span id="section-1"></span>

## 1 はじめに

<span id="figure-01"></span>

![図1. データをノイズへ滑らかに変換する Probability Flow（PF）ODE が与えられたとき、生成モデリングのため、ODE 軌道上の任意の点（たとえば ${\mathbf{x}}_{t}$、${\mathbf{x}}_{t^{\prime}}$、${\mathbf{x}}_{T}$）をその始点（たとえば ${\mathbf{x}}_{0}$）へ写像することを学習する。同じ軌道上の点に対して出力が一致するよう訓練されるため、これらの写像のモデルを consistency model と呼ぶ。](../../papers/consistency-models/figure-01.png)

**図 1.** データをノイズへ滑らかに変換する Probability Flow（PF）ODE が与えられたとき、生成モデリングのため、ODE 軌道上の任意の点（たとえば ${\mathbf{x}}_{t}$、${\mathbf{x}}_{t^{\prime}}$、${\mathbf{x}}_{T}$）をその始点（たとえば ${\mathbf{x}}_{0}$）へ写像することを学習する。同じ軌道上の点に対して出力が一致するよう訓練されるため、これらの写像のモデルを consistency model と呼ぶ。

拡散モデル [Soh15, Son19a, Son20, Nic21, Son21] は score-based generative model とも呼ばれ、画像生成 [Dha21, Nic22, Ram22, Sah22a, Rom22]、音声合成 [Kon21, Che21n, Pop21]、動画生成 [Ho22, Ho22c] など、多くの分野で前例のない成功を収めている。拡散モデルの重要な特徴は、ランダムな初期ベクトルからノイズを徐々に除去する反復的サンプリング過程である。この過程では、反復回数を増やすために計算量を追加すると通常はサンプル品質が向上するため、計算量と品質を柔軟に交換できる。また、拡散モデルの多くのゼロショットデータ編集能力の中核でもあり、画像のインペインティング、カラー化、ストローク誘導画像編集から、CT および MRI までの難しい逆問題を解ける [Son19a, Son21, Son22, Son23a, Kaw21, Kaw22, Chu23, Men22]。しかし GAN [Goo14a]、VAE [Kin14, Rez14]、normalizing flow [Din15, Din17, Kin18] のような 1 ステップ生成モデルと比べると、拡散モデルの反復生成は通常、サンプル生成に 10-2000 倍の計算を必要とし [Son20, Nic21, Son21, Zha22i, Lu22c]、推論が遅く、リアルタイム用途が制限される。

本研究の目的は、必要に応じて計算量とサンプル品質を交換できることや、ゼロショットデータ編集を行えることなど、反復サンプリングの重要な利点を損なわずに、効率的な 1 ステップ生成を可能にする生成モデルを作ることである。[図 1](#figure-01) に示すように、連続時間拡散モデル [Son21] の probability flow（PF）常微分方程式（ODE）を基礎とする。その軌道は、データ分布を扱いやすいノイズ分布へ滑らかに遷移させる。任意の時刻の任意の点を軌道の始点へ写像するモデルを学習する。注目すべき性質は自己整合性、すなわち*同じ軌道上の点は同じ初期点へ写像される*ことである。このため、この種のモデルを **consistency model** と呼ぶ。Consistency model は、ランダムノイズベクトル（ODE 軌道の終点、たとえば[図 1](#figure-01) の ${\mathbf{x}}_{T}$）を 1 回のネットワーク評価で変換し、データサンプル（ODE 軌道の始点、たとえば[図 1](#figure-01) の ${\mathbf{x}}_{0}$）を生成できる。さらに、複数の時刻で consistency model の出力を連鎖させると、拡散モデルの反復サンプリングと同様に、計算量を増やしてサンプル品質を改善し、ゼロショットデータ編集を行える。

Consistency model の訓練には、自己整合性を課すことに基づく 2 つの方法を用意する。第 1 の方法では、数値 ODE ソルバーと事前学習済み拡散モデルを用いて、PF ODE 軌道上の隣接点の組を生成する。これらの組に対するモデル出力の差を最小化すれば、拡散モデルを consistency model へ効率よく蒸留し、1 回のネットワーク評価で高品質なサンプルを生成できる。対照的に、第 2 の方法は事前学習済み拡散モデルをまったく必要とせず、consistency model を単独で訓練できる。この方法により、consistency model は独立した生成モデル群となる。どちらの方法も敵対的訓練を必要とせず、アーキテクチャ上の制約も少ないため、柔軟なニューラルネットワークで consistency model をパラメータ化できる。

CIFAR-10 [Kri09]、ImageNet $64\times 64$ [Den09a]、LSUN $256\times 256$ [Yu15a] など、複数の画像データセットで consistency model の有効性を示す。実験では、蒸留法としての consistency model は、多様なデータセットの少数ステップ生成で progressive distillation [Sal22] などの既存の拡散蒸留法を上回った。CIFAR-10 では 1 ステップと 2 ステップ生成でそれぞれ 3.55 と 2.93、ImageNet $64\times 64$ では 1 回と 2 回のネットワーク評価でそれぞれ 6.20 と 4.70 という新記録の FID を達成した。独立した生成モデルとして訓練した場合も、事前学習済み拡散モデルを利用できないにもかかわらず、progressive distillation の 1 ステップサンプルと同等以上の品質を得られる。また、複数のデータセットで多くの GAN や、既存の非敵対的 1 ステップ生成モデルを上回る。さらに、画像のデノイジング、補間、インペインティング、カラー化、超解像、ストローク誘導画像編集（SDEdit、[Men22]）など、幅広いゼロショットデータ編集に consistency model を利用できることを示す。

<span id="section-2"></span>

## 2 拡散モデル

Consistency model は連続時間拡散モデルの理論 [Son21, Kar22] に強く着想を得ている。拡散モデルは、ガウス摂動によってデータをノイズへ徐々に変換し、その後、逐次的なデノイジングによってノイズからサンプルを生成する。$p_{\text{data}}({\mathbf{x}})$ をデータ分布とする。拡散モデルは、確率微分方程式（SDE）[Son21] によって $p_{\text{data}}({\mathbf{x}})$ を拡散することから始める。

<span id="equation-01"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\bm{\mu}({\mathbf{x}}_{t},t)\mathop{}\!\mathrm{d}t+\sigma(t)\mathop{}\!\mathrm{d}{\mathbf{w}}_{t},
$$

ここで $t\in[0,T]$、$T>0$ は固定定数、$\bm{\mu}(\cdot,\cdot)$ と $\sigma(\cdot)$ はそれぞれドリフト係数と拡散係数であり、$\{{\mathbf{w}}_{t}\}_{t\in[0,T]}$ は標準ブラウン運動を表す。${\mathbf{x}}_{t}$ の分布を $p_{t}({\mathbf{x}})$ と表すため、$p_{0}({\mathbf{x}})\equiv p_{\text{data}}({\mathbf{x}})$ である。この SDE の顕著な性質は、[Son21] が *Probability Flow（PF）ODE* と名付けた常微分方程式（ODE）が存在することである。その解軌道を時刻 $t$ でサンプリングした分布は $p_{t}({\mathbf{x}})$ となる。

<span id="equation-02"></span>

$$
\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}=\left[\bm{\mu}({\mathbf{x}}_{t},t)-\frac{1}{2}\sigma(t)^{2}\nabla\log p_{t}({\mathbf{x}}_{t})\right]\mathop{}\!\mathrm{d}t.
$$

ここで $\nabla\log p_{t}({\mathbf{x}})$ は $p_{t}({\mathbf{x}})$ の*スコア関数*である。このため、拡散モデルは *score-based generative model* とも呼ばれる [Son19a, Son20, Son21]。

通常、[式 1](#equation-01) の SDE は、$p_{T}({\mathbf{x}})$ が扱いやすいガウス分布 $\pi({\mathbf{x}})$ に近くなるよう設計される。以下では [Kar22] の設定を採用し、$\bm{\mu}({\mathbf{x}},t)=\bm{0}$、$\sigma(t)=\sqrt{2t}$ とする。このとき $p_{t}({\mathbf{x}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$ であり、$\otimes$ は畳み込み、$\pi({\mathbf{x}})=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ である。サンプリングでは、まず*スコアマッチング* [Hyv05, Vin11, Son19b, Son19a, Nic21] によって*スコアモデル* ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\approx\nabla\log p_{t}({\mathbf{x}})$ を訓練する。次に、これを[式 2](#equation-02) へ代入して PF ODE の経験的な推定を得る。その形は次のとおりである。

<span id="equation-03"></span>

$$
\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t).
$$

[式 3](#equation-03) を*経験的 PF ODE* と呼ぶ。次に、$\hat{{\mathbf{x}}}_{T}\sim\pi=\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ をサンプリングして経験的 PF ODE を初期化し、Euler [Son21a, Son21] や Heun [Kar22] など任意の数値 ODE ソルバーで時間を逆向きに解き、解軌道 $\{\hat{{\mathbf{x}}}_{t}\}_{t\in[0,T]}$ を得る。得られた $\hat{{\mathbf{x}}}_{0}$ は、データ分布 $p_{\text{data}}({\mathbf{x}})$ からの近似サンプルとみなせる。数値的不安定性を避けるため、通常は固定された小さな正数 $\epsilon$ に対し $t=\epsilon$ でソルバーを停止し、$\hat{{\mathbf{x}}}_{\epsilon}$ を近似サンプルとして採用する。[Kar22] に従って画像の画素値を $[-1,1]$ に再スケーリングし、$T=80,\epsilon=0.002$ とする。

拡散モデルのボトルネックはサンプリングの遅さである。ODE ソルバーによるサンプリングでは、スコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ を反復評価するため、計算コストが高い。高速サンプリングの既存手法には、より高速な数値 ODE ソルバー [Son21a, Zha22i, Lu22c, Doc22a] と蒸留法 [Luh21, Sal22, Men22a, Zhe22g] がある。しかし ODE ソルバーは、競争力のあるサンプルを生成するために依然として 10 回を超える評価を必要とする。[Luh21] や [Zhe22g] など大半の蒸留法は、蒸留前に拡散モデルから大規模なサンプルデータセットを収集する必要があり、それ自体の計算コストも高い。われわれの知る限り、この欠点を持たない唯一の蒸留法は progressive distillation（PD、[Sal22]）であり、実験では consistency model と広範に比較する。

<span id="section-3"></span>

## 3 Consistency model

<span id="figure-02"></span>

![図2. Consistency model は、PF ODE の任意の軌道上の点をその軌道の始点へ写像するよう訓練される。](../../papers/consistency-models/figure-02.png)

**図 2.** Consistency model は、PF ODE の任意の軌道上の点をその軌道の始点へ写像するよう訓練される。

Consistency model は、設計の中心で 1 ステップ生成を支援しながら、サンプル品質と計算量を交換する反復生成、およびゼロショットデータ編集も可能にする新しいモデルである。蒸留モードでも単独モードでも訓練できる。前者では、事前学習済み拡散モデルの知識を 1 ステップサンプラーへ蒸留し、ゼロショット画像編集を可能にしつつ、他の蒸留法よりサンプル品質を大きく改善する。後者では、事前学習済み拡散モデルに依存せず単独で訓練する。これにより、独立した新しい生成モデル群となる。

以下では consistency model の定義、パラメータ化、サンプリングを導入し、ゼロショットデータ編集への応用を簡単に論じる。

**定義** [式 2](#equation-02) の PF ODE の解軌道 $\{{\mathbf{x}}_{t}\}_{t\in[\epsilon,T]}$ が与えられたとき、*consistency function* を ${\bm{f}}:({\mathbf{x}}_{t},t)\mapsto{\mathbf{x}}_{\epsilon}$ と定義する。Consistency function は*自己整合性*を持つ。すなわち、同じ PF ODE 軌道に属する任意の $({\mathbf{x}}_{t},t)$ の組に対して出力が一致し、すべての $t,t^{\prime}\in[\epsilon,T]$ について ${\bm{f}}({\mathbf{x}}_{t},t)={\bm{f}}({\mathbf{x}}_{t^{\prime}},t^{\prime})$ となる。[図 2](#figure-02) に示すように、${\bm{f}}_{\bm{\theta}}$ と表す *consistency model* の目的は、自己整合性を課すことを学習して、データから consistency function ${\bm{f}}$ を推定することである（詳細は[第 4 節](#section-4)と[第 5 節](#section-5)）。Neural ODE [Che18g] の文脈では neural flow [Bil21] に類似の定義が用いられる。ただし neural flow と異なり、consistency model に可逆性を課さない。

**パラメータ化** 任意の consistency function ${\bm{f}}(\cdot,\cdot)$ について、${\bm{f}}({\mathbf{x}}_{\epsilon},\epsilon)={\mathbf{x}}_{\epsilon}$、すなわち ${\bm{f}}(\cdot,\epsilon)$ は恒等関数である。この制約を*境界条件*と呼ぶ。境界条件は consistency model の訓練を成功させるうえで重要であり、すべての consistency model が満たさなければならない。また、アーキテクチャに対する最も強い制約でもある。深層ニューラルネットワークに基づく consistency model について、この境界条件を*ほぼ無償で*実装する 2 つの方法を論じる。${\mathbf{x}}$ と同じ次元の出力を持つ自由形式の深層ニューラルネットワーク $F_{\bm{\theta}}({\mathbf{x}},t)$ を仮定する。第 1 の方法は、consistency model を単純に次のようにパラメータ化することである。

<span id="equation-04"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=\begin{cases}{\mathbf{x}}&\quad t=\epsilon\\
F_{\bm{\theta}}({\mathbf{x}},t)&\quad t\in(\epsilon,T]\end{cases}.
$$

第 2 の方法は、スキップ接続を用いて consistency model を次のようにパラメータ化する。

<span id="equation-05"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t),
$$

ここで $c_{\text{skip}}(t)$ と $c_{\text{out}}(t)$ は、$c_{\text{skip}}(\epsilon)=1$、$c_{\text{out}}(\epsilon)=0$ を満たす微分可能な関数である。$F_{\bm{\theta}}({\mathbf{x}},t),c_{\text{skip}}(t),c_{\text{out}}(t)$ がすべて微分可能なら、consistency model は $t=\epsilon$ で微分可能となる。これは連続時間 consistency model の訓練に不可欠である（[第 9.1 節](#section-9-1)と[第 9.2 節](#section-9-2)）。[式 5](#equation-05) のパラメータ化は、多くの成功した拡散モデル [Kar22, Bal22] とよく似ており、強力な拡散モデルのアーキテクチャを consistency model の構築に流用しやすい。このため、すべての実験で第 2 のパラメータ化を用いる。

**サンプリング** 十分に訓練された consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$ があれば、初期分布 $\hat{{\mathbf{x}}}_{T}\sim\mathcal{N}(\bm{0},T^{2}{\bm{I}})$ からサンプリングし、$\hat{{\mathbf{x}}}_{\epsilon}={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$ を評価してサンプルを生成できる。必要な順伝播は 1 回だけであり、*1 ステップでサンプルを生成する*。デノイジングとノイズ注入を交互に行って consistency model を複数回評価すれば、サンプル品質も改善できる。[アルゴリズム 1](#algorithm-01) にまとめたこの*マルチステップ*サンプリングは、計算量とサンプル品質を柔軟に交換でき、ゼロショットデータ編集にも重要である。実際には、貪欲法で[アルゴリズム 1](#algorithm-01) の時刻 $\{\tau_{1},\tau_{2},\cdots,\tau_{N-1}\}$ を求め、三分探索によって時刻を 1 つずつ定め、得られるサンプルの FID を最適化する。これは、それまでの時刻が与えられたとき、FID が次の時刻の単峰関数であると仮定する。実験ではこの仮定が成立したため、より良い戦略の検討は今後の課題とする。

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**アルゴリズム 1：マルチステップ consistency sampling。**

- **入力：** Consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$、時刻列 $\tau_{1}>\tau_{2}>\cdots>\tau_{N-1}$、初期ノイズ $\hat{{\mathbf{x}}}_{T}$。
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{T},T)$。
- **各** $n=1$ **から** $N-1$ **について：**
  - ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$ をサンプリングする。
  - $\hat{{\mathbf{x}}}_{\tau_{n}}\gets{\mathbf{x}}+\sqrt{\tau_{n}^{2}-\epsilon^{2}}{\mathbf{z}}$。
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{\tau_{n}},\tau_{n})$。
- **出力：** ${\mathbf{x}}$。

</div>

**ゼロショットデータ編集** 拡散モデルと同様に、consistency model は明示的な訓練なしでさまざまなデータ編集や操作をゼロショットで行える。たとえば、ガウスノイズベクトルからデータサンプルへの 1 対 1 の写像を定義する。GAN、VAE、normalizing flow などの潜在変数モデルと同様に、潜在空間をたどることでサンプル間を容易に補間できる（[図 11](#figure-11)）。$t\in[\epsilon,T]$ の任意のノイズ付き入力 ${\mathbf{x}}_{t}$ から ${\mathbf{x}}_{\epsilon}$ を復元するよう訓練されるため、さまざまなノイズレベルでデノイジングも行える（[図 12](#figure-12)）。さらに、[アルゴリズム 1](#algorithm-01) のマルチステップ生成を、拡散モデルと同様の反復置換法 [Son19a, Son21, Ho22] と組み合わせれば、ある種の逆問題をゼロショットで解ける。これにより、インペインティング（[図 10](#figure-10)）、カラー化（[図 8](#figure-08)）、超解像（[図 6(b)](#figure-06)）、SDEdit [Men22] と同様のストローク誘導画像編集（[図 13](#figure-13)）が可能になる。[第 6.3 節](#section-6-3) では、多くのゼロショット画像編集タスクで consistency model の能力を実験的に示す。

<span id="section-4"></span>

## 4 蒸留による consistency model の訓練

第 1 の訓練法は、事前学習済みスコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ の蒸留に基づく。議論の中心は、スコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ を PF ODE へ代入して得られる[式 3](#equation-03) の経験的 PF ODE である。時間区間 $[\epsilon,T]$ を、境界 $t_{1}=\epsilon<t_{2}<\cdots<t_{N}=T$ を持つ $N-1$ 個の部分区間へ離散化する。実際には [Kar22] に従い、$t_{i}=(\epsilon^{1/\rho}+\frac{i-1}{N-1}(T^{1/\rho}-\epsilon^{1/\rho}))^{\rho}$ で境界を定め、$\rho=7$ とする。$N$ が十分大きければ、数値 ODE ソルバーを 1 離散化ステップ実行することで、${\mathbf{x}}_{t_{n+1}}$ から ${\mathbf{x}}_{t_{n}}$ を正確に推定できる。この推定を $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ と表し、次のように定義する。

<span id="equation-06"></span>

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

ここで $\Phi(\cdots;{\bm{\phi}})$ は、経験的 PF ODE に適用する 1 ステップ ODE ソルバーの更新関数である。たとえば Euler ソルバーでは $\Phi({\mathbf{x}},t;{\bm{\phi}})=-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ であり、次の更新則に対応する。

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}={\mathbf{x}}_{t_{n+1}}-(t_{n}-t_{n+1})t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1}).
$$

簡単のため、本研究では 1 ステップ ODE ソルバーのみを扱う。マルチステップ ODE ソルバーへの拡張は容易であり、今後の課題とする。

[式 2](#equation-02) の PF ODE と[式 1](#equation-01) の SDE には関係があるため（[第 2 節](#section-2)）、まず ${\mathbf{x}}\sim p_{\text{data}}$ をサンプリングし、${\mathbf{x}}$ にガウスノイズを加えることで、ODE 軌道の分布に沿ってサンプリングできる。具体的には、データ点 ${\mathbf{x}}$ に対し、データセットから ${\mathbf{x}}$ をサンプリングし、SDE の遷移密度 $\mathcal{N}({\mathbf{x}},t_{n+1}^{2}{\bm{I}})$ から ${\mathbf{x}}_{t_{n+1}}$ をサンプリングする。次に、[式 6](#equation-06) に従って数値 ODE ソルバーを 1 ステップ実行して $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ を計算すれば、PF ODE 軌道上の隣接点 $(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},{\mathbf{x}}_{t_{n+1}})$ を効率よく生成できる。その後、この組に対するモデル出力の差を最小化して consistency model を訓練する。これにより、次の *consistency distillation* 損失が導かれる。

<span id="definition-01"></span>

**定義 1。** Consistency distillation 損失を次のように定義する。

<span id="equation-07"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\\
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))],
$$

期待値は ${\mathbf{x}}\sim p_{\text{data}}$、$n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$、${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ について取る。$\mathcal{U}\llbracket 1,N-1\rrbracket$ は $\{1,2,\cdots,N-1\}$ 上の一様分布、$\lambda(\cdot)\in\mathbb{R}^{+}$ は正の重み関数、$\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}$ は[式 6](#equation-06) で与えられる。${\bm{\theta}}^{-}$ は最適化中の ${\bm{\theta}}$ の過去の値の移動平均を表し、$d(\cdot,\cdot)$ は $\forall{\mathbf{x}},{\mathbf{y}}:d({\mathbf{x}},{\mathbf{y}})\geq 0$ を満たし、${\mathbf{x}}={\mathbf{y}}$ の場合に限って $d({\mathbf{x}},{\mathbf{y}})=0$ となる距離関数である。

特に断らない限り、論文全体で[定義 1](#definition-01) の記法を用い、$\mathbb{E}[\cdot]$ はすべての確率変数についての期待値を表す。実験では、二乗 $\ell_{2}$ 距離 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$、$\ell_{1}$ 距離 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$、Learned Perceptual Image Patch Similarity（LPIPS、[Zha18d]）を検討する。$\lambda(t_{n})\equiv 1$ はすべてのタスクとデータセットで良好に機能した。実際には、モデルパラメータ ${\bm{\theta}}$ に対する確率的勾配降下で目的関数を最小化し、${\bm{\theta}}^{-}$ は指数移動平均（EMA）で更新する。すなわち、減衰率 $0\leq\mu<1$ に対し、各最適化ステップ後に次を更新する。

<span id="equation-08"></span>

$$
{\bm{\theta}}^{-}\leftarrow\operatorname{stopgrad}(\mu{\bm{\theta}}^{-}+(1-\mu){\bm{\theta}}).
$$

訓練手順全体を[アルゴリズム 2](#algorithm-02) にまとめる。深層強化学習 [Mni13, Mni15, Lil15] と momentum-based contrastive learning [Gri20, He20a] の慣例に従い、${\bm{f}}_{{\bm{\theta}}^{-}}$ を「ターゲットネットワーク」、${\bm{f}}_{\bm{\theta}}$ を「オンラインネットワーク」と呼ぶ。単に ${\bm{\theta}}^{-}={\bm{\theta}}$ とする場合と比べ、[式 8](#equation-08) の EMA 更新と「stopgrad」演算子は訓練を大幅に安定化し、最終性能を改善した。

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**アルゴリズム 2：Consistency Distillation（CD）。**

- **入力：** データセット $\mathcal{D}$、初期モデルパラメータ $\bm{\theta}$、学習率 $\eta$、ODE ソルバー $\Phi(\cdot,\cdot;\bm{\phi})$、$d(\cdot,\cdot)$、$\lambda(\cdot)$、$\mu$。
- $\bm{\theta}^{-}\gets\bm{\theta}$。
- 収束するまで**繰り返す：**
  - ${\mathbf{x}}\sim\mathcal{D}$ と $n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$ をサンプリングする。
  - ${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ をサンプリングする。
  - $\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\gets{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};\bm{\phi})$。
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))$。
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-};\bm{\phi})$。
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu\bm{\theta}^{-}+(1-\mu)\bm{\theta})$。

</div>

以下では、漸近解析に基づいて consistency distillation を理論的に正当化する。

<span id="theorem-01"></span>

**定理 1。** $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$ とし、${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ を[式 3](#equation-03) の経験的 PF ODE の consistency function とする。${\bm{f}}_{\bm{\theta}}$ が Lipschitz 条件を満たす、すなわち、ある $L>0$ が存在し、すべての $t\in[\epsilon,T]$、${\mathbf{x}}$、${\mathbf{y}}$ について $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$ とする。さらに、すべての $n\in\llbracket 1,N-1\rrbracket$ について、$t_{n+1}$ で呼び出す ODE ソルバーの局所誤差が $O((t_{n+1}-t_{n})^{p+1})$ で一様に上から抑えられ、$p\geq 1$ とする。このとき、$\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$ ならば、次が成り立つ。

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details 証明
証明は帰納法に基づき、数値 ODE ソルバーの大域誤差境界に対する古典的な証明 [Sul03] と同様である。完全な証明は[第 8.2 節](#section-8-2)に示す。
:::

${\bm{\theta}}^{-}$ は ${\bm{\theta}}$ の履歴の移動平均なので、[アルゴリズム 2](#algorithm-02) の最適化が収束すると ${\bm{\theta}}^{-}={\bm{\theta}}$ となる。つまり、ターゲットとオンラインの consistency model は最終的に一致する。さらに consistency distillation 損失がゼロなら、[定理 1](#theorem-01) より、いくつかの正則性条件のもとで ODE ソルバーのステップ幅を十分小さくすれば、推定した consistency model は任意に高い精度へ近づける。境界条件 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},\epsilon)\equiv{\mathbf{x}}$ は、訓練中に自明な解 ${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)\equiv\bm{0}$ が生じることを排除する。

${\bm{\theta}}^{-}={\bm{\theta}}$ または ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ なら、consistency distillation 損失 $\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})$ を無限個の時刻（$N\to\infty$）へ拡張できる。得られる連続時間損失関数では、$N$ も時刻 $\{t_{1},t_{2},\cdots,t_{N}\}$ も指定する必要がない。ただし Jacobian-vector product を含み、効率的な実装には forward-mode automatic differentiation が必要であるため、深層学習フレームワークによっては十分に対応していないことがある。これらの連続時間蒸留損失を定理 3、4、5 に示し、詳細は[第 9.1 節](#section-9-1)に譲る。

<span id="section-5"></span>

## 5 Consistency model の単独訓練

Consistency model は、事前学習済み拡散モデルに依存せず訓練できる。既存の拡散蒸留法とは異なり、独立した新しい生成モデル群となる。

<span id="algorithm-03"></span>

<div class="paper-algorithm">

**アルゴリズム 3：Consistency Training（CT）。**

- **入力：** データセット $\mathcal{D}$、初期モデルパラメータ $\bm{\theta}$、学習率 $\eta$、ステップスケジュール $N(\cdot)$、EMA 減衰率スケジュール $\mu(\cdot)$、$d(\cdot,\cdot)$、$\lambda(\cdot)$。
- $\bm{\theta}^{-}\gets\bm{\theta}$、$k\gets 0$。
- 収束するまで**繰り返す：**
  - ${\mathbf{x}}\sim\mathcal{D}$ と $n\sim\mathcal{U}\llbracket 1,N(k)-1\rrbracket$ をサンプリングする。
  - ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$ をサンプリングする。
  - $\mathcal{L}(\bm{\theta},\bm{\theta}^{-})\gets\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{\bm{\theta}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))$。
  - $\bm{\theta}\gets\bm{\theta}-\eta\nabla_{\bm{\theta}}\mathcal{L}(\bm{\theta},\bm{\theta}^{-})$。
  - $\bm{\theta}^{-}\gets\operatorname{stopgrad}(\mu(k)\bm{\theta}^{-}+(1-\mu(k))\bm{\theta})$。
  - $k\gets k+1$。

</div>

Consistency distillation では、事前学習済みスコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ で真のスコア関数 $\nabla\log p_{t}({\mathbf{x}})$ を近似した。次の不偏推定量（[第 8 節](#section-8)の[補題 1](#lemma-01)）を用いれば、この事前学習済みスコアモデルを完全に省ける。

$$
\nabla\log p_{t}({\mathbf{x}}_{t})=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mathrel{\bigg|}{\mathbf{x}}_{t}\right],
$$

ここで ${\mathbf{x}}\sim p_{\text{data}}$、${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$ である。すなわち、${\mathbf{x}}$ と ${\mathbf{x}}_{t}$ が与えられれば、$-({\mathbf{x}}_{t}-{\mathbf{x}})/t^{2}$ によって $\nabla\log p_{t}({\mathbf{x}}_{t})$ を推定できる。

次の結果が示すように、$N\to\infty$ の極限で Euler 法を ODE ソルバーに用いる場合、この不偏推定は consistency distillation において事前学習済み拡散モデルを置き換えるのに十分である。

<span id="theorem-02"></span>

**定理 2。** $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$ とする。$d$ と ${\bm{f}}_{{\bm{\theta}}^{-}}$ はともに 2 階連続微分可能で 2 階導関数が有界、重み関数 $\lambda(\cdot)$ は有界、$\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$ と仮定する。さらに Euler ODE ソルバーを用い、事前学習済みスコアモデルが真値に一致する、すなわち $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$ とする。このとき、

<span id="equation-09"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

期待値は ${\mathbf{x}}\sim p_{\text{data}}$、$n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$、${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ について取る。Consistency training 目的 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ を次のように定義する。

<span id="equation-10"></span>

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

ここで ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$ である。さらに、$\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$ なら $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ である。

::: details 証明
証明は Taylor 級数展開とスコア関数の性質（[補題 1](#lemma-01)）に基づく。完全な証明は[第 8.3 節](#section-8-3)に示す。
:::

[式 10](#equation-10) を *consistency training*（CT）損失と呼ぶ。$\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})$ はオンラインネットワーク ${\bm{f}}_{\bm{\theta}}$ とターゲットネットワーク ${\bm{f}}_{{\bm{\theta}}^{-}}$ のみに依存し、拡散モデルパラメータ ${\bm{\phi}}$ にはまったく依存しない。損失関数 $\mathcal{L}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ は剰余項 $o(\Delta t)$ より遅く減少するため、$N\to\infty$、$\Delta t\to 0$ では[式 9](#equation-09) の損失を支配する。

実用上の性能を改善するため、訓練中にスケジュール関数 $N(\cdot)$ に従って $N$ を徐々に増やす。直観的には（[図 3(d)](#figure-03)）、$N$ が小さい（$\Delta t$ が大きい）とき、基礎となる consistency distillation 損失（[式 9](#equation-09) の左辺）に対し、consistency training 損失は「分散」が小さく「バイアス」が大きいため、訓練初期の収束が速い。反対に、$N$ が大きい（$\Delta t$ が小さい）ときは「分散」が大きく「バイアス」が小さくなり、訓練終盤に適している。最高性能を得るには、$\mu$ もスケジュール関数 $\mu(\cdot)$ に従って $N$ とともに変化させるべきである。Consistency training の完全なアルゴリズムは[アルゴリズム 3](#algorithm-03)、実験で用いるスケジュール関数は[第 10 節](#section-10)に示す。

Consistency distillation と同様に、${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ なら、consistency training 損失 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ を連続時間（$N\to\infty$）へ拡張できる（[定理 6](#theorem-06)）。この連続時間損失関数は $N$ や $\mu$ のスケジュールを必要としないが、効率的な実装には forward-mode automatic differentiation が必要である。離散時間 CT 損失と異なり、連続時間目的には好ましくない「バイアス」がない。[定理 2](#theorem-02) で実質的に $\Delta t\to 0$ とするためである。詳細は[第 9.2 節](#section-9-2)に譲る。

<span id="section-6"></span>

## 6 実験

Consistency distillation と consistency training により、CIFAR-10 [Kri09]、ImageNet $64\times 64$ [Den09a]、LSUN Bedroom $256\times 256$、LSUN Cat $256\times 256$ [Yu15a] の実画像データセットで consistency model を学習する。Fréchet Inception Distance（FID、[Heu17]、低いほど良い）、Inception Score（IS、[Sal16]、高いほど良い）、Precision（Prec.、[Kyn19]、高いほど良い）、Recall（Rec.、[Kyn19]、高いほど良い）で比較する。追加の実験詳細は[第 10 節](#section-10)に示す。

<span id="figure-03"></span>

![図3. CIFAR-10 で consistency distillation（CD）と consistency training（CT）に影響する要因。CD の最良構成は LPIPS、Heun ODE ソルバー、$N=18$ である。$N$ と $\mu$ を固定する場合より、適応的スケジュールによって CT は大幅に速く収束する。](../../papers/consistency-models/figure-03.png)

**図 3.** CIFAR-10 で consistency distillation（CD）と consistency training（CT）に影響する要因。CD の最良構成は LPIPS、Heun ODE ソルバー、$N=18$ である。$N$ と $\mu$ を固定する場合より、適応的スケジュールによって CT は大幅に速く収束する。

<span id="figure-04"></span>

![図4. Consistency distillation（CD）によるマルチステップ画像生成。CD はすべてのデータセットとサンプリングステップで progressive distillation（PD）を上回る。唯一の例外は Bedroom $256\times 256$ の 1 ステップ生成である。](../../papers/consistency-models/figure-04.png)

**図 4.** Consistency distillation（CD）によるマルチステップ画像生成。CD はすべてのデータセットとサンプリングステップで progressive distillation（PD）を上回る。唯一の例外は Bedroom $256\times 256$ の 1 ステップ生成である。

<span id="section-6-1"></span>

### 6.1 Consistency model の訓練

CIFAR-10 で一連の実験を行い、consistency distillation（CD）と consistency training（CT）で訓練したモデルの性能に各種ハイパーパラメータが及ぼす影響を調べる。まず CD の距離関数 $d(\cdot,\cdot)$、ODE ソルバー、離散化ステップ数 $N$ を調べ、次に CT のスケジュール関数 $N(\cdot)$ と $\mu(\cdot)$ を調べる。

CD では、二乗 $\ell_{2}$ 距離 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|^{2}_{2}$、$\ell_{1}$ 距離 $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_{1}$、Learned Perceptual Image Patch Similarity（LPIPS、[Zha18d]）を距離関数として用いる。ODE ソルバーは [Kar22] の Euler 前進法と Heun 2 次法を比較する。離散化ステップ数は $N\in\{9,12,18,36,50,60,80,120\}$ を比較する。CD で訓練するモデルは対応する事前学習済み拡散モデルで初期化し、CT のモデルはランダムに初期化する。

[図 3(a)](#figure-03) に示すように、CD に最適な距離は LPIPS であり、すべての訓練反復で $\ell_{1}$ と $\ell_{2}$ を大きく上回る。CIFAR-10 上の出力は画像であり、LPIPS は自然画像の類似度測定用に設計されているため、予想どおりである。次に CD に最適な ODE ソルバーと $N$ を調べる。[図 3(b)](#figure-03)と[図 3(c)](#figure-03)より、Heun と $N=18$ が最良である。拡散モデルではなく consistency model を訓練しているにもかかわらず、どちらも [Kar22] の推奨と一致する。[図 3(b)](#figure-03)は、同じ $N$ なら Heun 2 次法が Euler 1 次法を一貫して上回ることも示す。これは、高次 ODE ソルバーで訓練した最適な consistency model は同じ $N$ で推定誤差が小さいとする[定理 1](#theorem-01)と整合する。[図 3(c)](#figure-03)から、$N$ が十分大きいと CD の性能は $N$ に鈍感になる。以後、特記しない限り CD に LPIPS と Heun を用いる。CIFAR-10 と ImageNet $64\times 64$ の $N$ は [Kar22] に従い、他のデータセットでは個別に調整する（[第 10 節](#section-10)）。

CD と CT は密接に関係するため、すべての CT 実験で LPIPS を用いる。CT の損失は特定の数値 ODE ソルバーに依存しないので、CD と異なり Heun 2 次法は不要である。[図 3(d)](#figure-03)に示すように、CT の収束は $N$ に非常に敏感である。小さい $N$ は速く収束するがサンプルが悪く、大きい $N$ は遅く収束するが収束後のサンプルは良い。[第 5 節](#section-5)の解析と一致し、収束速度と品質を均衡させるため $N$ と $\mu$ を徐々に増やす実用上の選択を裏づける。適応スケジュールは CT の収束速度とサンプル品質を大きく改善する。異なる解像度ごとに $N(\cdot)$ と $\mu(\cdot)$ を調整する。詳細は[第 10 節](#section-10)に示す。

<span id="table-01"></span>

![表1. CIFAR-10 のサンプル品質。$^{\ast}$蒸留用の合成データ構築を必要とする手法。](../../papers/consistency-models/table-01.png)

**表 1.** CIFAR-10 のサンプル品質。$^{\ast}$蒸留用の合成データ構築を必要とする手法。

<span id="table-02"></span>

![表2. ImageNet $64\times 64$、LSUN Bedroom および Cat $256\times 256$ のサンプル品質。$^{\dagger}$蒸留法。](../../papers/consistency-models/table-02.png)

**表 2.** ImageNet $64\times 64$、LSUN Bedroom および Cat $256\times 256$ のサンプル品質。$^{\dagger}$蒸留法。

<span id="section-6-2"></span>

### 6.2 少数ステップ画像生成

**蒸留** 現在の文献で consistency distillation（CD）と最も直接比較できるのは progressive distillation（PD、[Sal22]）である。現在まで、両者だけが*蒸留前に合成データを構築しない*蒸留法である。対照的に knowledge distillation [Luh21] や DFNO [Zhe22g] は、高価な数値 ODE/SDE ソルバーで拡散モデルから多数のサンプルを生成し、大規模な合成データセットを用意する必要がある。CIFAR-10、ImageNet $64\times 64$、LSUN $256\times 256$ で PD と CD を包括的に比較し、結果を[図 4](#figure-04)に示す。全手法は社内で事前学習した同じ EDM [Kar22] から蒸留する。すべてのサンプリング回数で、*[Sal22] の二乗 $\ell_{2}$ 距離より LPIPS を使うと PD が一貫して改善する*。PD と CD はステップ数を増やすほど改善する。Bedroom $256\times 256$ の 1 ステップ生成で $\ell_{2}$ の CD が $\ell_{2}$ の PD をわずかに下回る場合を除き、すべてのデータセット、ステップ数、距離関数で CD が PD を上回る。[表 1](#table-01)のとおり、CD は合成データを要する Knowledge Distillation [Luh21] や DFNO [Zhe22g] も上回る。

<span id="figure-05"></span>

![図5. EDM（*上*）、CT + 1 ステップ生成（*中*）、CT + 2 ステップ生成（*下*）のサンプル。対応する画像はすべて同じ初期ノイズから生成した。](../../papers/consistency-models/figure-05.png)

**図 5.** EDM（*上*）、CT + 1 ステップ生成（*中*）、CT + 2 ステップ生成（*下*）のサンプル。対応する画像はすべて同じ初期ノイズから生成した。

<span id="figure-06"></span>

![図6. LSUN Bedroom $256\times 256$ で consistency distillation により訓練した consistency model によるゼロショット画像編集。](../../papers/consistency-models/figure-06.png)

**図 6.** LSUN Bedroom $256\times 256$ で consistency distillation により訓練した consistency model によるゼロショット画像編集。

**直接生成** [表 1](#table-01)と[表 2](#table-02)で、consistency training（CT）と他の生成モデルを 1 ステップおよび 2 ステップ生成の品質で比較する。PD と CD も参考として含める。原論文 [Sal22] の既定設定なので、両表の PD は $\ell_{2}$ 距離による結果である。公平な比較のため、PD と CD は同じ EDM を蒸留する。CT は CIFAR-10 で VAE と normalizing flow という既存の 1 ステップ非敵対的モデルを大きく上回る。さらに、*CT は蒸留に依存せず、PD の 1 ステップサンプルに匹敵する品質を得る*。[図 5](#figure-05)に EDM（上）、1 ステップ CT（中）、2 ステップ CT（下）のサンプルを示す。[第 12 節](#section-12)の[図 14](#figure-14)から[図 21](#figure-21)には CD と CT の追加サンプルを示す。CT と EDM は独立に訓練されるが、*同じ初期ノイズから得たサンプルは構造的によく似る*。EDM は mode collapse を起こさないため、CT も起こしにくいことを示唆する。

<span id="section-6-3"></span>

### 6.3 ゼロショット画像編集

拡散モデルと同様に、[アルゴリズム 1](#algorithm-01)のマルチステップサンプリングを変更してゼロショット画像編集を行える。LSUN Bedroom で consistency distillation により訓練したモデルで実演する。[図 6(a)](#figure-06)では、カラー化で訓練していないにもかかわらず、テスト時にグレースケールの寝室画像をカラー化できる。[図 6(b)](#figure-06)では同じモデルが低解像度入力から高解像度画像を生成する。[図 6(c)](#figure-06)では拡散モデルの SDEdit [Men22] と同様に、人が作ったストローク入力から画像を生成する。ストローク入力で訓練していないため、これもゼロショットである。[第 11 節](#section-11)ではインペインティング（[図 10](#figure-10)）、補間（[図 11](#figure-11)）、デノイジング（[図 12](#figure-12)）に加え、カラー化（[図 8](#figure-08)）、超解像（[図 9](#figure-09)）、ストローク誘導生成（[図 13](#figure-13)）の追加例を示す。

<span id="section-7"></span>

## 7 結論

1 ステップおよび少数ステップ生成を支援するよう設計した生成モデル、consistency model を導入した。複数の画像ベンチマークと少ないサンプリング反復で、consistency distillation が既存の拡散モデル蒸留法を上回ることを実験的に示した。独立した生成モデルとしても、GAN を除く既存の 1 ステップ生成モデルより良いサンプルを生成する。拡散モデルと同様に、インペインティング、カラー化、超解像、デノイジング、補間、ストローク誘導生成などのゼロショット画像編集も可能である。

Consistency model は deep Q-learning [Mni15] や momentum-based contrastive learning [Gri20, He20a] など、他分野の技術ともよく似ている。異なる分野の着想や手法を相互に取り入れる余地がある。

## 謝辞

原稿を査読して有益なフィードバックを寄せた Alex Nichol、ストローク誘導画像生成実験に必要な入力を提供した Chenlin Meng、OpenAI Algorithms チームに感謝する。

<span id="section-8"></span>

## 8 証明

<span id="section-8-1"></span>

### 8.1 記法

${\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t)$ は ${\bm{\theta}}$ でパラメータ化した consistency model、${\bm{f}}({\mathbf{x}},t;{\bm{\phi}})$ は[式 3](#equation-03)の経験的 PF ODE の consistency function を表す。${\bm{\phi}}$ は事前学習済みスコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ への依存を示す。[式 2](#equation-02)の PF ODE の consistency function は ${\bm{f}}({\mathbf{x}},t)$ と表す。多変数関数 ${\bm{h}}({\mathbf{x}},{\mathbf{y}})$ に対し、$\partial_{1}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ は ${\mathbf{x}}$ に関する Jacobian、$\partial_{2}{\bm{h}}({\mathbf{x}},{\mathbf{y}})$ は ${\mathbf{y}}$ に関する Jacobian とする。特記しない限り、${\mathbf{x}}$ は $p_{\text{data}}({\mathbf{x}})$ から、$n$ は $\llbracket 1,N-1\rrbracket$ から一様に、${\mathbf{x}}_{t_{n}}$ は $\mathcal{N}({\mathbf{x}};t_{n}^{2}{\bm{I}})$ からサンプリングする。$\llbracket 1,N-1\rrbracket$ は整数集合 $\{1,2,\cdots,N-1\}$ を表す。さらに、次の定義を用いる。

$$
\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}\coloneqq{\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\Phi({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}),
$$

ここで $\Phi(\cdots;{\bm{\phi}})$ は、スコアモデル ${\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)$ が定める経験的 PF ODE の 1 ステップソルバーの更新関数である。既定では $\mathbb{E}[\cdot]$ は式中の関連する全確率変数についての期待値を表す。

<span id="section-8-2"></span>

### 8.2 Consistency distillation

<span id="theorem-01-appendix"></span>

**定理 1。** $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$ とし、${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ を[式 3](#equation-03)の経験的 PF ODE の consistency function とする。${\bm{f}}_{\bm{\theta}}$ が Lipschitz 条件を満たす、すなわち、ある $L>0$ が存在し、すべての $t\in[\epsilon,T]$、${\mathbf{x}}$、${\mathbf{y}}$ について $\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)-{\bm{f}}_{\bm{\theta}}({\mathbf{y}},t)\|_{2}\leq L\|{\mathbf{x}}-{\mathbf{y}}\|_{2}$ とする。さらに、すべての $n\in\llbracket 1,N-1\rrbracket$ について、$t_{n+1}$ で呼ぶソルバーの局所誤差が $O((t_{n+1}-t_{n})^{p+1})$ で一様に抑えられ、$p\geq1$ とする。$\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=0$ なら次が成り立つ。

$$
\sup_{n,{\mathbf{x}}}\|{\bm{f}}_{{\bm{\theta}}}({\mathbf{x}},t_{n})-{\bm{f}}({\mathbf{x}},t_{n};{\bm{\phi}})\|_{2}=O((\Delta t)^{p}).
$$

::: details 証明

<span id="equation-11"></span>

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]=0.
$$

定義より $p_{t_{n}}({\mathbf{x}}_{t_{n}})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t_{n}^{2}{\bm{I}})$ で、$t_{n}\geq\epsilon>0$ である。したがって全 ${\mathbf{x}}_{t_{n}}$ と $1\leq n\leq N$ について $p_{t_{n}}({\mathbf{x}}_{t_{n}})>0$ であり、[式 11](#equation-11)から次を得る。

<span id="equation-12"></span>

$$
\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({{\mathbf{x}}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\equiv 0.
$$

$\lambda(\cdot)>0$ かつ $d({\mathbf{x}},{\mathbf{y}})=0\Leftrightarrow{\mathbf{x}}={\mathbf{y}}$ なので、さらに次が従う。

<span id="equation-13"></span>

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\equiv{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}).
$$

時刻 $t_{n}$ の誤差ベクトル ${\bm{e}}_{n}$ を次のように定義する。

$$
{\bm{e}}_{n}\coloneqq{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}).
$$

次の漸化式が直ちに得られる。

<span id="equation-14"></span>

$$
\begin{aligned}
{\bm{e}}_{n+1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})-{\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}}) \\
={\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n}},t_{n})+{\bm{e}}_{n},
\end{aligned}
$$

ここで（i）は[式 13](#equation-13)と ${\bm{f}}({\mathbf{x}}_{t_{n+1}},t_{n+1};{\bm{\phi}})={\bm{f}}({\mathbf{x}}_{t_{n}},t_{n};{\bm{\phi}})$ による。${\bm{f}}_{\bm{\theta}}(\cdot,t_{n})$ の Lipschitz 定数は $L$ なので、

$$
\begin{aligned}
\|{\bm{e}}_{n+1}\|_{2} & \leq\|{\bm{e}}_{n}\|_{2}+L\|\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}}-{\mathbf{x}}_{t_{n}}\|_{2} \\
\mathrel{{\mathop{=}\limits}}\|{\bm{e}}_{n}\|_{2}+L\cdot O((t_{n+1}-t_{n})^{p+1}) \\
=\|{\bm{e}}_{n}\|_{2}+O((t_{n+1}-t_{n})^{p+1}),
\end{aligned}
$$

（i）はソルバーの局所誤差が $O((t_{n+1}-t_{n})^{p+1})$ で抑えられるため成り立つ。また ${\bm{e}}_{1}=\bm{0}$ である。なぜなら、

$$
\begin{aligned}
{\bm{e}}_{1} & ={\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{1}},t_{1})-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}}) \\
\mathrel{{\mathop{=}\limits}}{\mathbf{x}}_{t_{1}}-{\mathbf{x}}_{t_{1}} \\
=\bm{0}.
\end{aligned}
$$

（i）はパラメータ化により ${\bm{f}}({\mathbf{x}}_{t_{1}},t_{1};{\bm{\phi}})={\mathbf{x}}_{t_{1}}$ となるため、（ii）は ${\bm{f}}(\cdot,\cdot;{\bm{\phi}})$ の定義による。漸化式[式 14](#equation-14)に帰納法を適用すると、

$$
\begin{aligned}
\|{\bm{e}}_{n}\|_{2} & \leq\|{\bm{e}}_{1}\|_{2}+\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}O((t_{k+1}-t_{k})^{p+1}) \\
=\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((t_{k+1}-t_{k})^{p}) \\
\leq\sum_{k=1}^{n-1}(t_{k+1}-t_{k})O((\Delta t)^{p}) \\
=O((\Delta t)^{p})\sum_{k=1}^{n-1}(t_{k+1}-t_{k}) \\
=O((\Delta t)^{p})(t_{n}-t_{1}) \\
\leq O((\Delta t)^{p})(T-\epsilon) \\
=O((\Delta t)^{p}),
\end{aligned}
$$

を得て、証明が完了する。

:::

<span id="section-8-3"></span>

### 8.3 Consistency training

次の補題は、[定理 2](#theorem-02)の証明に必要なスコア関数の不偏推定量を与える。

<span id="lemma-01"></span>

**補題 1。** ${\mathbf{x}}\sim p_{\text{data}}({\mathbf{x}})$、${\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}};t^{2}{\bm{I}})$、$p_{t}({\mathbf{x}}_{t})=p_{\text{data}}({\mathbf{x}})\otimes\mathcal{N}(\bm{0},t^{2}{\bm{I}})$ とする。このとき $\nabla\log p_{t}({\mathbf{x}})=-\mathbb{E}[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}]$ である。

::: details 証明

$$
\begin{aligned}
\nabla\log p_{t}({\mathbf{x}}_{t}) & =\frac{\int p_{\text{data}}({\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}} \\
=\frac{\int p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}}}{p_{t}({\mathbf{x}}_{t})} \\
=\int\frac{p_{\text{data}}({\mathbf{x}})p({\mathbf{x}}_{t}\mid{\mathbf{x}})}{p_{t}({\mathbf{x}}_{t})}\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
\mathrel{{\mathop{=}\limits}}\int p({\mathbf{x}}\mid{\mathbf{x}}_{t})\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mathop{}\!\mathrm{d}{\mathbf{x}} \\
=\mathbb{E}[\nabla_{{\mathbf{x}}_{t}}\log p({\mathbf{x}}_{t}\mid{\mathbf{x}})\mid{\mathbf{x}}_{t}] \\
=-\mathbb{E}\left[\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t^{2}}\mid{\mathbf{x}}_{t}\right],
\end{aligned}
$$

（i）は Bayes の定理による。

:::

<span id="theorem-02-appendix"></span>

**定理 2。** $\Delta t\coloneqq\max_{n\in\llbracket 1,N-1\rrbracket}\{|t_{n+1}-t_{n}|\}$ とする。$d$ と ${\bm{f}}_{{\bm{\theta}}^{-}}$ は 2 階連続微分可能で 2 階導関数が有界、$\lambda(\cdot)$ は有界、$\mathbb{E}[\|\nabla\log p_{t_{n}}({\mathbf{x}}_{t_{n}})\|_{2}^{2}]<\infty$ とする。Euler ODE ソルバーを用い、事前学習済みスコアモデルが真値に一致する、すなわち $\forall t\in[\epsilon,T]:{\bm{s}}_{{\bm{\phi}}}({\mathbf{x}},t)\equiv\nabla\log p_{t}({\mathbf{x}})$ とする。このとき、

$$
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
$$

期待値は ${\mathbf{x}}\sim p_{\text{data}}$、$n\sim\mathcal{U}\llbracket 1,N-1\rrbracket$、${\mathbf{x}}_{t_{n+1}}\sim\mathcal{N}({\mathbf{x}};t_{n+1}^{2}{\bm{I}})$ について取る。Consistency training 目的 $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})$ は次のように定義する。

$$
\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))],
$$

ここで ${\mathbf{z}}\sim\mathcal{N}(\bm{0},{\bm{I}})$ である。さらに、$\inf_{N}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})>0$ なら $\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ である。

::: details 証明

<span id="equation-15"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}}+(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}),t_{n}))] \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})+\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}}) \\
\qquad+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|))] \\
= & \mathbb{E}\{\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[ \\
\quad\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})+o(|t_{n+1}-t_{n}|)]\} \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n+1}-t_{n})t_{n+1}\nabla\log p_{t_{n+1}}({\mathbf{x}}_{t_{n+1}})]\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)].
\end{aligned}
$$

次に[式 15](#equation-15)へ[補題 1](#lemma-01)を適用し、逆方向に Taylor 展開すると、

<span id="equation-16"></span>

$$
\begin{aligned}
\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\mathbb{E}\left[\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\Big|{\mathbf{x}}_{t_{n+1}}\right]\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
\mathrel{{\mathop{=}\limits}} & \mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))] \\
\quad+\mathbb{E}\left\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right]\right\} \\
\qquad+\mathbb{E}\{\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]\}+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\bigg[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})) \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\left[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})t_{n+1}\left(\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}}\right)\right] \\
\quad+\lambda(t_{n})\partial_{2}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))[\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})(t_{n}-t_{n+1})]+o(|t_{n+1}-t_{n}|)\bigg] \\
\qquad+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})t_{n+1}\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}^{2}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}_{t_{n+1}}+(t_{n}-t_{n+1})\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n+1}{\mathbf{z}}+(t_{n}-t_{n+1}){\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(|t_{n+1}-t_{n}|)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+\mathbb{E}[o(\Delta t)] \\
= & \mathbb{E}\left[\lambda(t_{n})d\left({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}\left({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}\right)\right)\right]+o(\Delta t) \\
= & \mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})+o(\Delta t),
\end{aligned}
$$

（i）は全期待値の法則による。また ${\mathbf{z}}\coloneqq\frac{{\mathbf{x}}_{t_{n+1}}-{\mathbf{x}}}{t_{n+1}}\sim\mathcal{N}(\bm{0},{\bm{I}})$ である。したがって $\mathcal{L}_{\text{CD}}^{N}=\mathcal{L}_{\text{CT}}^{N}+o(\Delta t)$ となり、[式 9](#equation-09)が示された。さらに $\inf_{N}\mathcal{L}_{\text{CD}}^{N}>0$ なら $\mathcal{L}_{\text{CT}}^{N}\geq O(\Delta t)$ である。そうでなければ $\mathcal{L}_{\text{CT}}^{N}<O(\Delta t)$ から $\lim_{\Delta t\to0}\mathcal{L}_{\text{CD}}^{N}=0$ となり、仮定に矛盾する。

:::

<span id="remark-01"></span>

**注 1。** ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$ の場合など、$\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})\geq O(\Delta t)$ が満たされない場合も、[定理 6](#theorem-06)により、$\mathcal{L}_{\text{CT}}^{N}$ を訓練目的とすることを正当化できる。

<span id="section-9"></span>

## 9 連続時間への拡張

適切な条件のもとで、consistency distillation と consistency training の目的関数を無限個の時刻（$N\to\infty$）へ一般化できる。

<span id="section-9-1"></span>

### 9.1 連続時間 consistency distillation

${\bm{\theta}}^{-}={\bm{\theta}}$ か ${\bm{\theta}}^{-}=\operatorname{stopgrad}({\bm{\theta}})$（$\mu=0$ と同じ）かに応じ、$\mathcal{L}_{\text{CD}}^{N}$ には 2 種類の連続時間拡張がある。2 階連続微分可能な距離関数 $d({\mathbf{x}},{\mathbf{y}})$ に対し、行列 ${\bm{G}}({\mathbf{x}})$ の $(i,j)$ 成分を次で定義する。

$$
[{\bm{G}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{x}},{\mathbf{y}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

同様に ${\bm{H}}({\mathbf{x}})$ を次で定義する。

$$
[{\bm{H}}({\mathbf{x}})]_{ij}\coloneqq\frac{\partial^{2}d({\mathbf{y}},{\mathbf{x}})}{\partial y_{i}\partial y_{j}}\bigg|_{{\mathbf{y}}={\mathbf{x}}}.
$$

${\bm{G}}$ と ${\bm{H}}$ は連続時間目的の構成に重要である。${\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)$ の ${\mathbf{x}}$ に関する Jacobian を $\frac{\partial{\bm{f}}_{\bm{\theta}}}{\partial{\mathbf{x}}}$ と表す。

${\bm{\theta}}^{-}={\bm{\theta}}$（stopgrad なし）なら、次の結果を得る。

<span id="theorem-03"></span>

**定理 3。** $t_{n}=\tau(\frac{n-1}{N-1})$、$n\in\llbracket1,N\rrbracket$ とし、$\tau$ は $\tau(0)=\epsilon,\tau(1)=T$ を満たす狭義単調関数とする。$\tau$ は $[0,1]$ で連続微分可能、$d$ は 3 階連続微分可能で 3 階導関数が有界、${\bm{f}}_{\bm{\theta}}$ は 2 階連続微分可能で 1・2 階導関数が有界とする。さらに $\lambda$ は有界、$\sup_{{\mathbf{x}},t}\|{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\|_{2}<\infty$ とする。Euler ソルバーを用いると、

<span id="equation-17"></span>

$$
\lim_{N\to\infty}(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

ここで $\mathcal{L}_{\text{CD}}^{\infty}$ は次で定義する。

<span id="equation-18"></span>

$$
\frac{1}{2}\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

期待値は ${\mathbf{x}}\sim p_{\text{data}},u\sim\mathcal{U}[0,1],t=\tau(u),{\mathbf{x}}_{t}\sim\mathcal{N}({\mathbf{x}},t^{2}{\bm{I}})$ について取る。

::: details 証明

<span id="equation-19"></span>

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})={\bm{f}}_{{\bm{\theta}}}({\mathbf{x}}_{t_{n+1}}+t_{n+1}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u,t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}) \\
= & t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u+O((\Delta u)^{2}),
\end{aligned}
$$

$\tau'(u_n)=\frac1{\tau^{-1}(t_{n+1})}$ に注意し、損失を Taylor 展開すると、

$$
\begin{aligned}
(N-1)^{2}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{(\Delta u)^{2}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2(\Delta u)^{2}}\bigg(\mathbb{E}\{\lambda(t_{n})\tau^{\prime}(u_{n})^{2}[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot[{\bm{f}}_{\bm{\theta}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})-{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\lambda(t_{n})\tau^{\prime}(u_{n})^{2}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\frac{1}{2}\mathbb{E}\bigg[\frac{\lambda(t_{n})}{[(\tau^{-1})^{\prime}(t_{n})]^{2}}\left(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\right)^{\top}{\bm{G}}({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}))\\
\cdot\bigg(\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}-t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\bigg)\bigg]+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

（i）は $d({\bm{f}}_{\bm{\theta}}(\cdot),\cdot)$ の 2 次展開と $d({\mathbf{x}},{\mathbf{x}})=0$、$\nabla_{\mathbf y}d|_{{\mathbf y}={\mathbf x}}=0$ による。（ii）は[式 19](#equation-19)による。$\Delta u\to0$（$N\to\infty$）で両辺の極限を取ると[式 17](#equation-17)を得る。

:::

<span id="remark-02"></span>

**注 2。** [定理 3](#theorem-03)は簡単のため Euler を仮定するが、$N\to\infty$ では全ソルバーが同様に振る舞うため、一般のソルバーにも類似結果が成り立つと考える。一般化は今後の課題とする。

<span id="remark-03"></span>

**注 3。** [定理 3](#theorem-03)より $\mathcal{L}_{\text{CD}}^{\infty}$ の最小化で consistency model を訓練できる。特に $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ なら、

<span id="equation-26"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathbb{E}\left[\frac{\lambda(t)}{[(\tau^{-1})^{\prime}(t)]^{2}}\|\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\|^{2}_{2}\right].
$$

ただし、この連続時間目的の評価は Jacobian-vector product を要し、forward-mode automatic differentiation のないフレームワークでは遅く、実装も煩雑である。

<span id="remark-04"></span>

**注 4。** ${\bm{f}}_{\bm{\theta}}$ が経験的 PF ODE の真の consistency function に一致するなら、

$$
\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)}{\partial{\mathbf{x}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}},t)\equiv 0
$$

したがって $\mathcal{L}_{\text{CD}}^{\infty}=0$ である。全 $t$ で ${\bm{f}}_{\bm{\theta}}({\mathbf{x}}_t,t)={\mathbf{x}}_\epsilon$ であることを用い、この恒等式を時間微分すれば示せる。

$$
\begin{aligned}
{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)\equiv{\mathbf{x}}_{\epsilon} \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\frac{\mathop{}\!\mathrm{d}{\mathbf{x}}_{t}}{\mathop{}\!\mathrm{d}t}+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}[-t{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)]+\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\equiv 0 \\
\Longleftrightarrow & \frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\equiv 0.
\end{aligned}
$$

これは、consistency model が真値に一致する場合に限って最小になるという、$\mathcal{L}_{\text{CD}}^{\infty}$ の別の動機を与える。

$\ell_1$ などでは Hessian ${\bm G}$ がゼロで[定理 3](#theorem-03)は空虚になる。証明を少し変えると非自明な結果を得る。

<span id="theorem-04"></span>

**定理 4。** [定理 3](#theorem-03)と同じ $t_n,\tau$ を用い、${\bm f}_{\bm\theta}$ は 2 階連続微分可能で導関数が有界、$\lambda$ は有界、$\sup\|{\bm s}_{\bm\phi}\|_2<\infty$ とする。Euler を用い、$d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_1$ とすれば、

<span id="equation-27"></span>

$$
\lim_{N\to\infty}(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}}),
$$

ここで、

$$
\mathcal{L}_{\text{CD, $\ell_{1}$}}^{\infty}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}\| t\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)}{\partial t}\|_{1}\right]
$$

期待値は ${\mathbf{x}},u,t,{\mathbf{x}}_t$ について上記と同様に取る。

::: details 証明

<span id="equation-28"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\|{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})\|_{1}] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\left[\lambda(t_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})+O((\Delta u)^{2})\|_{1}\right] \\
= & \mathbb{E}\left[\lambda(t_{n})\tau^{\prime}(u_{n})\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right] \\
= & \mathbb{E}\left[\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}\| t_{n+1}\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-\frac{\partial{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}+O(\Delta u)\|_{1}\right]
\end{aligned}
$$

（i）は[式 19](#equation-19)の代入による。[式 28](#equation-28)で $\Delta u\to0$ とすれば[式 27](#equation-27)を得る。

:::

<span id="remark-05"></span>

**注 5。** [定理 4](#theorem-04)より $\mathcal{L}_{\text{CD},\ell_1}^{\infty}$ を最小化して訓練できる。[注 4](#remark-04)と同様に、全 ${\mathbf{x}}_t,t$ で ${\bm f}_{\bm\theta}({\mathbf{x}}_t,t)={\mathbf{x}}_\epsilon$ である場合に限り、この損失はゼロとなる。

${\bm\theta}^{-}=\operatorname{stopgrad}({\bm\theta})$ の場合、$N\to\infty$ で $\mathcal L_{\text{CD}}^N$ と勾配が一致する「擬似目的」を導ける。これを勾配降下で最小化することも蒸留訓練法となる。

<span id="theorem-05"></span>

**定理 5。** [定理 3](#theorem-03)と同じ $t_n,\tau$ を用いる。$d$ は 3 階、${\bm f}_{\bm\theta}$ は 2 階連続微分可能で各導関数が有界、$\lambda$、${\bm s}_{\bm\phi}$、$\nabla_{\bm\theta}{\bm f}_{\bm\theta}$ は有界とする。Euler を用い、${\bm\theta}^{-}=\operatorname{stopgrad}({\bm\theta})$ なら、

<span id="equation-29"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}),
$$

ここで、

<span id="equation-30"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

期待値は ${\mathbf{x}},u,t,{\mathbf{x}}_t$ について上記と同様に取る。

::: details 証明

<span id="equation-33"></span>

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]
\end{aligned}
$$

（i）は $d$ の 2 次展開と $d({\mathbf{x}},{\mathbf{x}})=0$、対応する勾配がゼロであることによる。[式 33](#equation-33)を ${\bm\theta}$ で微分し簡約すると、

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}}) \\
= & \frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})] \\
\mathrel{{\mathop{=}\limits}} & \!\begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\Delta u\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\tau^{\prime}(u_{n})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\tau^{\prime}(u_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \!\begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\bigg[t_{n+1}\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial{\mathbf{x}}_{t_{n+1}}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t_{n+1}},t_{n+1})\\
-\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})}{\partial t_{n+1}}\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined}
\end{aligned}
$$

（i）は連鎖律、（ii）は[式 19](#equation-19)と ${\bm f}_{\bm\theta}={\bm f}_{{\bm\theta}^{-}}$ による。$\Delta u\to0$ で極限を取ると[式 29](#equation-29)を得る。

:::

<span id="remark-06"></span>

**注 6。** $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ なら、擬似目的は次の形に簡約できる。

<span id="equation-42"></span>

$$
\mathcal{L}_{\text{CD}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}-t\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}{\bm{s}}_{\bm{\phi}}({\mathbf{x}}_{t},t)\right)\right].
$$

<span id="remark-07"></span>

**注 7。** [定理 5](#theorem-05)の目的は勾配だけに意味がある。値から進捗は測れないが、勾配降下で蒸留できる。通常の損失ではないため「擬似目的」と呼ぶ。

<span id="remark-08"></span>

**注 8。** [注 4](#remark-04)と同様に、${\bm f}_{\bm\theta}$ が真の consistency function なら $\mathcal L_{\text{CD}}^{\infty}=0$ かつその勾配もゼロである。逆は一般に成り立たず、真の損失 $\mathcal L_{\text{CD}}^{\infty}({\bm\theta},{\bm\theta})$ と異なる。

<span id="section-9-2"></span>

### 9.2 連続時間 consistency training

[定理 5](#theorem-05)の擬似目的は事前学習済み拡散モデルなしで推定でき、直接 consistency training が可能になる。

<span id="theorem-06"></span>

**定理 6。** [定理 5](#theorem-05)と同じ正則性を仮定し、さらに $\mathbb E[\|\nabla\log p_{t_n}({\mathbf{x}}_{t_n})\|_2^2]<\infty$、${\bm s}_{\bm\phi}=\nabla\log p_t$ とする。${\bm\theta}^{-}=\operatorname{stopgrad}({\bm\theta})$ なら、

<span id="equation-43"></span>

$$
\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\lim_{N\to\infty}(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-}),
$$

ここで $\mathcal L_{\text{CD}}^N$ は Euler ソルバーを用い、

<span id="equation-44"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})\coloneqq\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t))\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

期待値は ${\mathbf{x}}\sim p_{\text{data}},u\sim\mathcal U[0,1],t=\tau(u),{\mathbf{x}}_t\sim\mathcal N({\mathbf{x}},t^2{\bm I})$ について取る。

::: details 証明

$$
\begin{aligned}
(N-1)\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))] \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{2\Delta u}\bigg(\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{3})]\bigg)\end{multlined} \\
= & \begin{multlined}\frac{1}{2\Delta u}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

${\mathbf z}\sim\mathcal N(\bm0,{\bm I})$ である。（i）は $d$ の 2 次展開と $d({\mathbf{x}},{\mathbf{x}})=0$、対応する勾配がゼロであることによる。${\bm\theta}$ で微分し簡約すると、

<span id="equation-59"></span>

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CT}}^{N}({\bm{\theta}},{\bm{\theta}}^{-}) \\
= & \begin{multlined}\frac{1}{2\Delta u}\nabla_{\bm{\theta}}\mathbb{E}\{\lambda(t_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\bigg\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\tau^{\prime}(u_{n})\Delta u\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\tau^{\prime}(u_{n})\Delta u\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \begin{multlined}\nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}){\mathbf{z}}\\
+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]\end{multlined} \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\lambda(t_{n})\tau^{\prime}(u_{n})[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)] \\
= & \nabla_{\bm{\theta}}\mathbb{E}\bigg\{\frac{\lambda(t_{n})}{(\tau^{-1})^{\prime}(t_{n})}[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n}))\bigg[\partial_{1}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\frac{{\mathbf{x}}_{t_{n}}-{\mathbf{x}}}{t_{n}}+\partial_{2}{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n}},t_{n})\bigg]\bigg\}+\mathbb{E}[O(|\Delta u|)]
\end{aligned}
$$

（i）は連鎖律、（ii）は Taylor 展開による。[式 59](#equation-59)で極限を取ると[式 43](#equation-43)の第 2 等式を得る。

次に第 1 等式を示す。再び Taylor 展開すると、

$$
\begin{aligned}
(N-1)\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathcal{L}_{\text{CD}}^{N}({\bm{\theta}},{\bm{\theta}}^{-};{\bm{\phi}})=\frac{1}{\Delta u}\nabla_{\bm{\theta}}\mathbb{E}[\lambda(t_{n})d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\mathbb{E}[\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\partial_{1}d({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))] \\
= & \frac{1}{\Delta u}\begin{multlined}\mathbb{E}\bigg\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}\bigg[\partial_{1}d({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}),{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))\\
+{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))+O(|\Delta u|^{2})\bigg]\bigg\}\end{multlined} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
= & \frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t_{n+1}},t_{n+1})^{\top}[{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t_{n+1}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}(\hat{{\mathbf{x}}}_{t_{n}}^{\bm{\phi}},t_{n}))]+O(|\Delta u|^{2})\} \\
\mathrel{{\mathop{=}\limits}} & \begin{multlined}\frac{1}{\Delta u}\mathbb{E}\{\lambda(t_{n})[\nabla_{\bm{\theta}}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})]^{\top}{\bm{H}}({\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n}))\\
\cdot[{\bm{f}}_{\bm{\theta}}({\mathbf{x}}+t_{n+1}{\mathbf{z}},t_{n+1})-{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}+t_{n}{\mathbf{z}},t_{n})]\}+\mathbb{E}[O(|\Delta u|^{2})]\end{multlined}
\end{aligned}
$$

（i）は ${\mathbf{x}}_{t_{n+1}}={\mathbf{x}}+t_{n+1}{\mathbf z}$ と $\hat{{\mathbf{x}}}_{t_n}^{\bm\phi}={\mathbf{x}}+t_n{\mathbf z}$ による。以後[式 59](#equation-59)まで同じ議論を適用すれば、CD と CT の勾配の極限が一致し、証明が完了する。

:::

<span id="remark-09"></span>

**注 9。** $\mathcal L_{\text{CT}}^{\infty}$ は拡散モデルパラメータ ${\bm\phi}$ に依存せず、事前学習済み拡散モデルなしで最適化できる。

<span id="remark-10"></span>

**注 10。** $d({\mathbf{x}},{\mathbf{y}})=\|{\mathbf{x}}-{\mathbf{y}}\|_2^2$ なら、連続時間 CT 目的は次の形になる。

<span id="equation-60"></span>

$$
\mathcal{L}_{\text{CT}}^{\infty}({\bm{\theta}},{\bm{\theta}}^{-})=2\mathbb{E}\left[\frac{\lambda(t)}{(\tau^{-1})^{\prime}(t)}{\bm{f}}_{\bm{\theta}}({\mathbf{x}}_{t},t)^{\top}\left(\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial t}+\frac{\partial{\bm{f}}_{{\bm{\theta}}^{-}}({\mathbf{x}}_{t},t)}{\partial{\mathbf{x}}_{t}}\cdot\frac{{\mathbf{x}}_{t}-{\mathbf{x}}}{t}\right)\right].
$$

<span id="remark-11"></span>

**注 11。** [定理 5](#theorem-05)の CD と同様、$\mathcal L_{\text{CT}}^{\infty}$ は擬似目的である。値から訓練進捗は追えないが、勾配降下でデータから直接 consistency model を訓練できる。[注 8](#remark-08)と同様に、真の consistency function に一致すれば目的と勾配はともにゼロとなる。

<span id="section-9-3"></span>

### 9.3 実験的検証

<span id="figure-07"></span>

![図7. 離散時間の consistency distillation/training と連続時間版の比較。](../../papers/consistency-models/figure-07.png)

**図 7.** 離散時間の consistency distillation/training と連続時間版の比較。

連続時間 CD・CT 目的を検証するため、CIFAR-10 で各種損失によりモデルを訓練する。結果は[図 7](#figure-07)に示す。連続時間実験では $\lambda(t)=(\tau^{-1})'(t)$ とし、他は[表 3](#table-03)と同じで、一部を性能向上のため調整した。蒸留では次を比較する。

- CD $(\ell_2)$：$N=18$、$\ell_2$。

[定理 3](#theorem-03)で LPIPS は調べなかった。VGG の 2 階導関数を逆伝播する必要があり、高コストで数値的に不安定だからである。[図 7(a)](#figure-07)では、LPIPS と $\ell_2$ の双方で stopgrad 版（[定理 5](#theorem-05)）が非 stopgrad 版（[定理 3](#theorem-03)）を上回り、全蒸留法で LPIPS が最良である。離散時間 CD は連続時間 CD を上回る。連続時間目的の分散が大きく、離散時間では有効な高次ソルバーを使えるためと考えられる。

連続時間 CT の安定化には、事前学習済み EDM での初期化が重要だった。連続時間損失の分散が大きいためと推測する。公平のため、ランダム初期化でも動く離散時間 CT を含め、全モデルを同じ CIFAR-10 EDM で初期化した。分散低減は今後の課題とする。

次の目的を比較する。

- CT（LPIPS）：$N=120$、学習率 4e-4、ターゲット EMA 0.99。EDM 初期化では学習が遅くなるため $N,\mu$ のスケジュールは使わない。

[図 7(b)](#figure-07)では LPIPS が連続時間 CT を改善し、同じ LPIPS の離散時間 CT も上回る。離散時間では[定理 2](#theorem-02)の $\Delta t>0$ によるバイアスがあるが、連続時間では暗黙に $\Delta t\to0$ とするためである。

<span id="section-10"></span>

## 10 追加の実験詳細

<span id="table-03"></span>

![表3. CD と CT の訓練に用いたハイパーパラメータ](../../papers/consistency-models/table-03.png)

**表 3.** CD と CT の訓練に用いたハイパーパラメータ

**モデルアーキテクチャ。** [Son21, Dha21] に従う。CIFAR-10 は NCSN++、ImageNet と LSUN は [Dha21] の対応する構成を用いる。

**Consistency model のパラメータ化。** EDM と同じ構成を用い、境界条件を満たすようスキップ接続だけを修正する。[第 3 節](#section-3)の形を用いる。

$$
{\bm{f}}_{\bm{\theta}}({\mathbf{x}},t)=c_{\text{skip}}(t){\mathbf{x}}+c_{\text{out}}(t)F_{\bm{\theta}}({\mathbf{x}},t).
$$

EDM [Kar22] では次を選ぶ。

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{t^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}t}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

$\sigma_{\text{data}}=0.5$ である。$\epsilon\neq0$ では境界条件を満たさないため、次へ修正する。

$$
c_{\text{skip}}(t)=\frac{\sigma_{\text{data}}^{2}}{(t-\epsilon)^{2}+\sigma_{\text{data}}^{2}},\quad c_{\text{out}}(t)=\frac{\sigma_{\text{data}}(t-\epsilon)}{\sqrt{\sigma_{\text{data}}^{2}+t^{2}}},
$$

これで $c_{\text{skip}}(\epsilon)=1,c_{\text{out}}(\epsilon)=0$ を満たす。

**CT のスケジュール関数。** 最高性能のため $N(\cdot),\mu(\cdot)$ を指定し、全実験で次の形を用いる。

$$
\begin{aligned}
N(k) & =\left\lceil\sqrt{\frac{k}{K}((s_{1}+1)^{2}-s_{0}^{2})+s_{0}^{2}}-1\right\rceil+1 \\
\mu(k) & =\exp\left(\frac{s_{0}\log\mu_{0}}{N(k)}\right),
\end{aligned}
$$

$K$ は総反復数、$s_0$ は初期ステップ数、$s_1$ は最終ステップ数、$\mu_0$ は初期 EMA 減衰率である。

**訓練詳細。** CD と PD は [Kar22] に従って自前で訓練した EDM を蒸留する。LSUN は ImageNet の設定をほぼ流用し、Bedroom/Cat を 600k/300k 回訓練し、バッチを 4096 から 2048 へ減らした。

LSUN の EMA は ImageNet と同じとした。PD は CIFAR-10 と ImageNet で [Sal22] の設定を用い、LSUN にも同じ設定を適用して良好に動作した。

蒸留は EDM 重み、CT はランダムに初期化した。Rectified Adam [Liu19d] を用い、学習率減衰、warm-up、weight decay は使わない。オンライン重みにも EMA を適用した。LSUN Bedroom の CD だけは EMA 0 が良かった。

LPIPS 入力は CIFAR-10 と ImageNet で $224\times224$ へ双線形拡大し、LSUN はそのまま評価した。全データで水平反転を用い、Nvidia A100 クラスタで訓練した。詳細は[表 3](#table-03)に示す。

<span id="section-11"></span>

## 11 ゼロショット画像編集の追加結果

<span id="algorithm-04"></span>

<div class="paper-algorithm">

**アルゴリズム 4：ゼロショット画像編集。**

- **入力：** Consistency model ${\bm{f}}_{\bm{\theta}}(\cdot,\cdot)$、時刻列 $t_{1}>t_{2}>\cdots>t_{N}$、参照画像 ${\mathbf{y}}$、可逆線形変換 ${\bm{A}}$、二値画像マスク $\bm{\Omega}$。
- ${\mathbf{y}}\gets{\bm{A}}^{-1}[(${\bm{A}}${\mathbf{y}})\odot(1-\bm{\Omega})+\bm{0}\odot\bm{\Omega}]$。
- ${\mathbf{x}}\sim\mathcal{N}(${\mathbf{y}},t_{1}^{2}{\bm{I}})$ をサンプリングする。
- ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(${\mathbf{x}},t_{1})$。
- ${\mathbf{x}}\gets{\bm{A}}^{-1}[(${\bm{A}}${\mathbf{y}})\odot(1-\bm{\Omega})+(${\bm{A}}${\mathbf{x}})\odot\bm{\Omega}]$。
- **各** $n=2$ **から** $N$ **について：**
  - ${\mathbf{x}}\sim\mathcal{N}(${\mathbf{x}},(t_{n}^{2}-\epsilon^{2}){\bm{I}})$ をサンプリングする。
  - ${\mathbf{x}}\gets{\bm{f}}_{\bm{\theta}}(${\mathbf{x}},t_{n})$。
  - ${\mathbf{x}}\gets{\bm{A}}^{-1}[(${\bm{A}}${\mathbf{y}})\odot(1-\bm{\Omega})+(${\bm{A}}${\mathbf{x}})\odot\bm{\Omega}]$。
- **出力：** ${\mathbf{x}}$。

</div>

カラー化（[図 8](#figure-08)）、超解像（[図 9](#figure-09)）、インペインティング（[図 10](#figure-10)）、補間（[図 11](#figure-11)）、デノイジング（[図 12](#figure-12)）、SDEdit（[図 13](#figure-13)）を追加検証する。モデルは LSUN Bedroom で CD 訓練した。

補間とデノイジング以外は[アルゴリズム 1](#algorithm-01)を少し変更した[アルゴリズム 4](#algorithm-04)で行う。${\mathbf y}$ は参照画像、$\bm\Omega$ はマスク、$\odot$ は要素積、${\bm A}$ は条件情報を注入する潜在空間への可逆変換である。

$$
t_{i}=\left(T^{1/\rho}+\frac{i-1}{N-1}(\epsilon^{1/\rho}-T^{1/\rho})\right)^{\rho}
$$

LSUN Bedroom では $N=40$ とする。

各タスクを以下に示す。

**インペインティング。** ${\mathbf y}$ は欠損をマスクした画像、$\bm\Omega$ の 1 は欠損画素、${\bm A}$ は恒等変換とする。

**カラー化。** 解結合空間ではインペインティングの特殊例になる。全チャネルが同じグレースケール ${\mathbf y}$ を RGB の加重平均で作る。

$$
0.2989R+0.5870G+0.1140B.
$$

$\bm\Omega$ を次の二値マスクとする。

$$
\bm{\Omega}[i,j,k]=\begin{cases}1,&\quad\text{$k=1$ or $2$}\\
0,&\quad\text{$k=0$}\end{cases}.
$$

第 1 列が $(0.2989,0.5870,0.1140)$ に比例する直交行列 ${\bm Q}$ を QR 分解で得る。

$$
{\bm{Q}}=\begin{pmatrix}0.4471&-0.8204&0.3563\\
0.8780&0.4785&0\\
0.1705&-0.3129&-0.9343\end{pmatrix}.
$$

線形変換 ${\bm A}$ を次で定義する。

$$
{\mathbf{y}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{x}}[i,j,l]{\bm{Q}}[l,k].
$$

${\bm Q}$ は直交行列なので逆変換も容易である。

$$
{\mathbf{x}}[i,j,k]=\sum_{l=0}^{2}{\mathbf{y}}[i,j,l]{\bm{Q}}[k,l].
$$

これらを[アルゴリズム 4](#algorithm-04)へ用いる。

**超解像。** $p\times p$ 非重複パッチ平均で低解像度化したと仮定し、直接拡大した ${\mathbf y}$ と次のマスクを用いる。

$$
\bm{\Omega}[i,j,k,l]=\begin{cases}1,&\quad k\geq 1\\
0,&\quad k=0\end{cases}.
$$

第 1 列が $(1/p,\ldots,1/p)$ の直交行列を QR 分解で得て、変換 ${\bm A}$ を定義する。

$$
{\mathbf{y}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{x}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[m,k].
$$

逆変換は次のとおりである。

$$
{\mathbf{x}}[i,j,k,l]=\sum_{m=0}^{p^{2}-1}{\mathbf{y}}[i\times p+(m-m\bmod p)/p,j\times p+m\bmod p,l]{\bm{Q}}[k,m].
$$

これらを[アルゴリズム 4](#algorithm-04)へ用いる。

**ストローク誘導生成。** SDEdit [Men22] と同様に、${\mathbf y}$ をストローク画、${\bm A}={\bm I}$、$\bm\Omega$ を全 1 とし、$t_1=5.38,t_2=2.24,N=2$ とする。

**デノイジング。** ${\mathbf x}$ が $\mathcal N(\bm0;\sigma^2{\bm I})$ で汚染され、$\sigma\in[\epsilon,T]$ なら ${\bm f}_{\bm\theta}({\mathbf x},\sigma)$ で復元する。

**補間。** ${\mathbf x}_1={\bm f}({\mathbf z}_1,T)$ と ${\mathbf x}_2={\bm f}({\mathbf z}_2,T)$ の間を球面線形補間する。

$$
{\mathbf{z}}=\frac{\sin[(1-\alpha)\psi]}{\sin(\psi)}{\mathbf{z}}_{1}+\frac{\sin(\alpha\psi)}{\sin(\psi)}{\mathbf{z}}_{2},
$$

$\psi$ を両ノイズの角度として ${\bm f}_{\bm\theta}({\mathbf z},T)$ を評価する。

<span id="figure-08"></span>

![図8. グレースケール（左）、カラー化結果（中央）、正解（右）。](../../papers/consistency-models/figure-08.png)

**図 8.** グレースケール（左）、カラー化結果（中央）、正解（右）。

<span id="figure-09"></span>

![図9. $32\times32$ 入力（左）、$256\times256$ 生成結果（中央）、正解（右）。](../../papers/consistency-models/figure-09.png)

**図 9.** $32\times32$ 入力（左）、$256\times256$ 生成結果（中央）、正解（右）。

<span id="figure-10"></span>

![図10. マスク画像（左）、補完結果（中央）、正解（右）。](../../papers/consistency-models/figure-10.png)

**図 10.** マスク画像（左）、補完結果（中央）、正解（右）。

<span id="figure-11"></span>

![図11. 左端と右端の間の球面線形補間。全サンプルは LSUN Bedroom の consistency model による。](../../papers/consistency-models/figure-11.png)

**図 11.** 左端と右端の間の球面線形補間。全サンプルは LSUN Bedroom の consistency model による。

<span id="figure-12"></span>

![図12. 1 ステップデノイジング。左端は正解、各 2 行の上段がノイズ付き、下段が復元画像。](../../papers/consistency-models/figure-12.png)

**図 12.** 1 ステップデノイジング。左端は正解、各 2 行の上段がノイズ付き、下段が復元画像。

<span id="figure-13"></span>

![図13. Consistency model による SDEdit。左端がストローク入力、右側が生成結果。](../../papers/consistency-models/figure-13.png)

**図 13.** Consistency model による SDEdit。左端がストローク入力、右側が生成結果。

<span id="section-12"></span>

## 12 追加サンプル

CIFAR-10（[図 14](#figure-14)、[図 18](#figure-18)）、ImageNet（[図 15](#figure-15)、[図 19](#figure-19)）、LSUN Bedroom（[図 16](#figure-16)、[図 20](#figure-20)）、Cat（[図 17](#figure-17)、[図 21](#figure-21)）の追加 CD・CT サンプルを示す。

<span id="figure-14"></span>

![図14. CIFAR-10 の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-14.png)

**図 14.** CIFAR-10 の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-15"></span>

![図15. ImageNet の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-15.png)

**図 15.** ImageNet の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-16"></span>

![図16. LSUN Bedroom の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-16.png)

**図 16.** LSUN Bedroom の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-17"></span>

![図17. LSUN Cat の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-17.png)

**図 17.** LSUN Cat の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-18"></span>

![図18. CIFAR-10 の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-18.png)

**図 18.** CIFAR-10 の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-19"></span>

![図19. ImageNet の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-19.png)

**図 19.** ImageNet の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-20"></span>

![図20. LSUN Bedroom の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-20.png)

**図 20.** LSUN Bedroom の未選別サンプル。同じ初期ノイズを使用。

<span id="figure-21"></span>

![図21. LSUN Cat の未選別サンプル。同じ初期ノイズを使用。](../../papers/consistency-models/figure-21.png)

**図 21.** LSUN Cat の未選別サンプル。同じ初期ノイズを使用。
