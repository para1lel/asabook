---
title: Elucidating Diffusion Model Design
createTime: 2026/09/13 20:30:00
permalink: /ja/papers/diffusion-design-space/
pageClass: paper-reading
---

> [Tero Karras](https://scholar.google.fi/citations?user=-50qJW8AAAAJ)、[Miika Aittala](https://people.csail.mit.edu/miika/)、[Timo Aila](https://users.aalto.fi/~ailat1/)、[Samuli Laine](https://users.aalto.fi/~laines9/)。[Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364)。2022 年 6 月 1 日に arXiv へ初投稿、現行の arXiv 版は v2（2022 年 10 月 11 日改訂）。*Advances in Neural Information Processing Systems 35*（NeurIPS 2022）、pp. 26565–26577 に掲載：[公式プロシーディングスページ](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a98846e9d9cc01cfb87eb694d946ce6b-Abstract-Conference.html)、[DOI](https://doi.org/10.52202/068431-1926)、[TeX ソース](https://arxiv.org/e-print/2206.00364v2)。<a href="/paper/diffusion-design-space.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。正確な印刷レイアウトと参考文献については、原論文 PDF を正とする。

## 概要

拡散ベース生成モデルの理論と実践は、現状では必要以上に複雑化していると私たちは考える。そこで、具体的な設計上の選択肢を明確に分離した設計空間を提示し、この状況の改善を図る。これにより、サンプリング過程と学習過程の双方、ならびにスコアネットワークの事前調整について、複数の変更点を特定できる。これらの改善を組み合わせることで、CIFAR-10 においてクラス条件付き設定で 1.79、無条件設定で 1.97 という新たな最高水準の FID を達成し、従来の設計よりも大幅に高速なサンプリング（画像 1 枚当たり 35 回のネットワーク評価）を実現する。さらに、そのモジュール性を示すため、私たちの設計変更によって、先行研究の事前学習済みスコアネットワークから得られる効率と品質の双方が劇的に向上することを示す。具体的には、学習済み ImageNet-64 モデルの FID を 2.07 から最高水準に近い 1.55 まで改善し、提案する改善を用いて再学習すると、新たな最高水準である 1.36 に到達する。

<span id="section-1"></span>

## 1 はじめに

拡散ベース生成モデル [Soh15] は、無条件 [Den20, Nic21, Son21] と条件付き [Ho22b, Nic22, Nic21, Pre22, Ram22, Rom22, Sah22, Son21] のいずれの設定でも、ニューラル画像合成の強力な新しい枠組みとして台頭しており、状況によっては GAN [Goo14] の品質さえ上回る [Dha21]。また、音声 [Kon21, Pop21] や動画 [Ho22] の生成、画像セグメンテーション [Bar22a, Wol22]、言語翻訳 [Nac21] といった他分野でも急速に利用が広がっている。そのため、これらのモデルを応用し、画像・分布の品質、学習コスト、生成速度の面でさらに改善することに大きな関心が寄せられている。

これらのモデルに関する文献は理論的内容が豊富であり、サンプリングスケジュール、学習ダイナミクス、ノイズレベルのパラメータ化などの導出は、可能な限り直接に理論的枠組みに基づく傾向があるため、モデルには確固たる理論的根拠が与えられる。しかし、このアプローチには、利用可能な設計空間を覆い隠してしまう危険がある。提案されたモデルが密結合した一式に見え、システム全体を壊さずに個々の構成要素を変更できないように思われる場合がある。

第 1 の貢献として、これらのモデルの背後にある理論を実践的な観点から捉え直し、導出元となり得る統計過程よりも、学習段階とサンプリング段階に現れる「具体的な」対象とアルゴリズムに重点を置く。目的は、これらの構成要素がどのように結び付いているか、またシステム全体の設計にどのような自由度があるかを、より深く理解することである。私たちは、ガウスノイズで汚染された学習データについて、ノイズレベルに依存する周辺分布のスコア [Hyv05] をニューラルネットワークでモデル化する、広範なモデル群に焦点を当てる。したがって、本研究は *denoising score matching* [Vin11] の文脈に位置付けられる。

第 2 の一連の貢献は、拡散モデルで画像を合成する際に用いるサンプリング過程に関するものである。サンプリングに最適な時間離散化を特定し、高次の Runge–Kutta 法をサンプリング過程に適用し、さまざまなサンプラースケジュールを評価するとともに、サンプリング過程における確率性の有用性を分析する。これらの改善により、合成時に必要なサンプリングステップ数が大幅に減少し、改善されたサンプラーは広く利用されている複数の拡散モデル [Nic21, Son21] にそのまま置き換えて使用できる。

第 3 の一連の貢献は、スコアをモデル化するニューラルネットワークの学習に焦点を当てる。一般に用いられるネットワークアーキテクチャ（DDPM [Den20]、NCSN [Son19a]）を引き続き利用しつつ、拡散モデルの設定におけるネットワークの入力、出力、損失関数の事前調整を初めて原理的に分析し、学習ダイナミクスを改善するためのベストプラクティスを導出する。また、学習中のノイズレベル分布の改善案を示し、通常は GAN で用いられる non-leaking augmentation [Kar20a] が拡散モデルにも有益であることを指摘する。

これらの貢献を総合すると、結果の品質を大幅に改善できる。たとえば、64$\times$64 解像度において CIFAR-10 [Kri09] で 1.79、ImageNet [Den09a] で 1.36 という記録的な FID を達成する。設計空間の主要な要素をすべて明示的に表へまとめることで、本アプローチは個々の構成要素に対する革新を容易にし、拡散モデルの設計空間をより広範かつ的を絞って探索できるようになると考えている。実装と事前学習済みモデルは <https://github.com/NVlabs/edm> で公開している。

<span id="section-2"></span>

## 2 共通の枠組みによる拡散モデルの表現

データ分布を $p_\text{data}(\boldsymbol{x})$、その標準偏差を $\sigma_\text{data}$ と表し、データへ標準偏差 $\sigma$ の i.i.d. ガウスノイズを加えて得られる平滑化分布の族 $p(\boldsymbol{x}; \sigma)$ を考える。$\sigma_{\max}\gg\sigma_\text{data}$ のとき、$p(\boldsymbol{x}; \sigma_{\max})$ は実質的に純粋なガウスノイズと区別できない。拡散モデルの発想は、ノイズ画像 $\boldsymbol{x}_0 \sim \mathcal{N}(\mathbf{0}, \sigma_{\max}^2 \mathbf{I})$ をランダムにサンプリングし、ノイズレベル $\sigma_0 = \sigma_{\max}> \sigma_1 > \dots > \sigma_N = 0$ をもつ画像 $\boldsymbol{x}_i$ へ順次ノイズ除去することで、各ノイズレベルにおいて $\boldsymbol{x}_i \sim p(\boldsymbol{x}_i; \sigma_i)$ を満たすようにすることである。したがって、この過程の終点 $\boldsymbol{x}_N$ はデータに従って分布する。

Song ら [Son21] は、サンプル $\boldsymbol{x}$ が時間とともに発展する間、所望の分布 $p$ を維持する確率微分方程式（SDE）を提示した。これにより、各反復でノイズの除去と追加をともに行う確率的ソルバーを用いて、上記の過程を実装できる。また、ランダム性の唯一の源が初期ノイズ画像 $\boldsymbol{x}_0$ である、対応する「確率流」常微分方程式（ODE）も示している。通常の説明順とは異なり、私たちはまず ODE を検討する。ODE は、サンプリング軌跡とその離散化を分析するうえで有用な設定を提供するためである。得られた知見は確率的サンプリングにも引き継がれ、[第 4 節](#section-4) で一般化として再導入する。

<span id="figure-01"></span>

![CIFAR-10 における denoising score matching。](../../papers/diffusion-design-space/figure-01.png)

**図 1.** CIFAR-10 における denoising score matching。**(a)** 強度の異なる加法ガウスノイズで汚染した学習セットの画像。ノイズレベルが高いと色が過飽和になるため、見やすく表示するために画像を正規化している。**(b)** [式 2](#equation-02) を解析的に最小化して得られる最適なノイズ除去結果（[第 8.3 節](#section-8-3) を参照）。ノイズレベルが上がるにつれて、結果はデータセットの平均へ近づく。

**ODE の定式化。** 確率流 ODE [Son21] は、時間を順方向または逆方向へ進める際に、画像のノイズレベルをそれぞれ連続的に増加または減少させる。ODE を規定するには、まず時刻 $t$ における所望のノイズレベルを定義するスケジュール $\sigma(t)$ を選ぶ必要がある。たとえば $\sigma(t)\propto\sqrt{t}$ とするのは、一定速度の熱拡散 [Fou22] に対応するため、数学的には自然である。しかし、[第 3 節](#section-3) で示すように、スケジュールの選択には実践上重大な影響があり、理論上の便宜だけを根拠に選ぶべきではない。

確率流 ODE を特徴付ける性質は、サンプル $\boldsymbol{x}_a \sim p \big( \boldsymbol{x}_a; \sigma(t_a) \big)$ を時刻 $t_a$ から $t_b$ まで（時間の順方向でも逆方向でも）発展させると、サンプル $\boldsymbol{x}_b \sim p \big( \boldsymbol{x}_b; \sigma(t_b) \big)$ が得られることである。先行研究 [Son21] に従うと、この要件は次式により満たされる（[第 8.1 節](#section-8-1) と [第 8.2 節](#section-8-2) を参照）。<span id="equation-01"></span>

$$
\mathrm{d}\boldsymbol{x}= -\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p \big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t\text{,}
$$
ここで、ドットは時間微分を表す。$\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma)$ は *score function* [Hyv05] であり、与えられたノイズレベルにおいてデータ密度の高い方向を指すベクトル場である。直観的には、この ODE の無限小の順方向ステップは、ノイズレベルの変化に応じた速さでサンプルをデータから遠ざける。同じことを逆向きに述べれば、逆方向ステップはサンプルをデータ分布へ近づける。

**Denoising score matching。** スコア関数には、一般には扱いにくい基礎密度関数 $p(\boldsymbol{x}; \sigma)$ の正規化定数に依存しないという顕著な性質がある [Hyv05]。そのため、評価ははるかに容易になり得る。具体的には、$D(\boldsymbol{x};\sigma)$ が、各 $\sigma$ について $p_\text{data}$ から抽出したサンプルの期待 $L_2$ ノイズ除去誤差を最小化する denoiser 関数、すなわち次式を満たすものとする。<span id="equation-02"></span>
<span id="equation-03"></span>

$$
\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} \mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} \| D(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\|^2_2
\text{,}\hspace*{2mm}\text{then}\hspace*{2mm}
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) = \big( D(\boldsymbol{x}; \sigma) - \boldsymbol{x}\big) / \sigma^2 \text{,}
$$
ここで、$\boldsymbol{y}$ は学習画像、$\boldsymbol{n}$ はノイズである。この見方では、スコア関数は $\boldsymbol{x}$ の信号からノイズ成分を分離し、[式 1](#equation-01) は時間とともにその成分を増幅（または減衰）させる。[図 1](#figure-01) は、理想的な $D$ の実際の振る舞いを示している。拡散モデルにおける重要な観察は、$D(\boldsymbol{x};\sigma)$ を [式 2](#equation-02) に従って学習したニューラルネットワーク $D_\theta(\boldsymbol{x};\sigma)$ として実装できることである。$D_\theta$ には、$\boldsymbol{x}$ を適切なダイナミックレンジへスケーリングする処理など、追加の前処理・後処理が含まれ得ることに注意されたい。このような *preconditioning* については [第 5 節](#section-5) で改めて扱う。

<span id="table-01"></span>

![各モデルファミリーで採用されている具体的な設計上の選択。$N$ は、サンプリング中に実行する ODE ソルバーの反復回数である。対応する時間ステップ列は $\{t_0, t_1, \dots, t_N\}$ で、$t_N = 0$ である。モデルが特定の $N$ と $\{t_i\}$ に対して学習されていた場合、その元の値をそれぞれ $M$ と $\{u_j\}$ で表す。denoiser は $D_\theta(\boldsymbol{x}; \sigma) = c_\mathrm{skip}(\sigma) \boldsymbol{x} + c_\mathrm{out}(\sigma) F_\theta(c_\mathrm{in}(\sigma) \boldsymbol{x}; c_\mathrm{noise}(\sigma))$ と定義し、$F_\theta$ は加工前のニューラルネットワーク層を表す。](../../papers/diffusion-design-space/table-01.png)

**表 1.** 各モデルファミリーで採用されている具体的な設計上の選択。$N$ は、サンプリング中に実行する ODE ソルバーの反復回数である。対応する時間ステップ列は $\{t_0, t_1, \dots, t_N\}$ で、$t_N = 0$ である。モデルが特定の $N$ と $\{t_i\}$ に対して学習されていた場合、その元の値をそれぞれ $M$ と $\{u_j\}$ で表す。denoiser は $D_\theta(\boldsymbol{x}; \sigma) = c_\mathrm{skip}(\sigma) \boldsymbol{x} + c_\mathrm{out}(\sigma) F_\theta(c_\mathrm{in}(\sigma) \boldsymbol{x}; c_\mathrm{noise}(\sigma))$ と定義し、$F_\theta$ は加工前のニューラルネットワーク層を表す。

**時間依存の信号スケーリング。** 一部の手法（[第 9.1 節](#section-9-1) を参照）は、追加のスケールスケジュール $s(t)$ を導入し、$\boldsymbol{x}= s(t) \hat\boldsymbol{x}$ を元のスケーリングされていない変数 $\hat\boldsymbol{x}$ のスケーリング版とみなす。これにより時間依存の確率密度が変わり、その結果として ODE の解軌跡も変化する。得られる ODE は [式 1](#equation-01) の一般化である。<span id="equation-04"></span>

$$
\mathrm{d}\boldsymbol{x}= \left[ \frac{\dot s(t)}{s(t)} ~\boldsymbol{x}-s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\left(\frac{\boldsymbol{x}}{s(t)}; \sigma(t)\right) \right] ~\mathrm{d}t\text{.}
$$
スコア関数を評価する際には、$p(\boldsymbol{x}; \sigma)$ の定義を $s(t)$ から独立させるため、$\boldsymbol{x}$ のスケーリングを明示的に元へ戻していることに注意されたい。

**離散化による求解。** 解くべき ODE は、[式 3](#equation-03) を [式 4](#equation-04) へ代入して点ごとの勾配を定義することで得られ、その解は数値積分、すなわち離散的な時間区間にわたって有限ステップを進めることで求められる。そのためには、積分方式（たとえば Euler 法や Runge–Kutta 法の変種）と、離散サンプリング時刻 $\{t_0, t_1, \dots, t_N\}$ の双方を選ぶ必要がある。多くの先行研究は Euler 法に依存しているが、[第 3 節](#section-3) では 2 次ソルバーの方が計算上のトレードオフに優れることを示す。簡潔さのため、ここでは私たちの ODE に Euler 法を適用した疑似コードを別途示さないが、[アルゴリズム 1](#algorithm-01) の 6～8 行目を省くことで得られる。

**全体像。** [表 1](#table-01) は、3 つの既存手法の決定論的変種を本枠組みで再現するための式を示している。これらの手法を選んだのは、広く利用され、最高水準の性能を達成しているためだけでなく、それぞれ異なる理論的基盤から導出されているためでもある。間接参照と再帰を取り除いたため、一部の式は原論文とかなり異なって見える。詳細は [第 9 節](#section-9) を参照されたい。この捉え直しの主な目的は、先行研究ではしばしば絡み合って見える独立した構成要素を、すべて明らかにすることである。本枠組みでは構成要素間に暗黙の依存関係はなく、個々の式に対する任意の妥当な選択は、原理上、動作するモデルにつながる。言い換えれば、たとえば極限でモデルがデータへ収束する性質を維持するために、ある構成要素の変更が別の箇所の変更を必須にすることはない。もちろん実際には、他より良く機能する選択や組み合わせもある。

<span id="section-3"></span>

## 3 決定論的サンプリングの改善

出力品質の向上および／またはサンプリングの計算コスト削減は、diffusion model 研究における一般的な主題である（たとえば [Doc22, Jol21, Liu22h, Lu22c, Luh21, Nic21, Sal22, Vah21, Wat22, Wat21, Zha22i]）。本研究の仮説は、サンプリング過程に関する選択が、ネットワークアーキテクチャや学習の詳細など、他の構成要素からほぼ独立しているというものである。言い換えれば、$D_\theta$ の学習手順が $\sigma(t)$、$s(t)$、および $\{t_i\}$ を規定すべきではなく、その逆も同様であり、サンプラーの観点から見れば、$D_\theta$ は単なる black box である [Wat22, Wat21]。本研究では、それぞれが異なる理論的枠組みとモデルファミリーを表す 3 つの*事前学習済み*モデル上で異なるサンプラーを評価し、この仮説を検証する。まず、これらのモデルについて元のサンプラー実装を用いてベースライン結果を測定し、次に[表 1](#table-01)の式を用いてこれらのサンプラーを本研究の統一された枠組みに取り込み、その後に本研究の改善を適用する。これにより、さまざまな実用上の選択肢を評価し、すべてのモデルに適用可能なサンプリング過程の一般的な改善を提案できる。

Song ら [Son21] が 32$\times$32 の無条件 CIFAR-10 [Kri09] で学習した「DDPM++ cont. (VP)」モデルと「NCSN++ cont. (VE)」モデルを評価する。これらは、それぞれ variance preserving（VP）定式化と variance exploding（VE）定式化 [Son21] に対応し、もともとは DDPM [Den20] と SMLD [Son19a] に着想を得ている。また、Dhariwal と Nichol [Dha21] が 64$\times$64 のクラス条件付き ImageNet [Den09a] で学習した「ADM (dropout)」モデルも評価する。これは improved DDPM（iDDPM）定式化 [Nic21] に対応する。このモデルは、$M=1000$ 個の離散的なノイズレベル集合を用いて学習された。詳細は[第 9 節](#section-9)に示す。

<span id="figure-02"></span>

![3 つの事前学習済みモデルを用いた決定論的サンプリング手法の比較。](../../papers/diffusion-design-space/figure-02.png)

**図 2。** 3 つの事前学習済みモデルを用いた決定論的サンプリング手法の比較。各曲線について、点は、観測された最小 FID の 3% 以内にある FID を達成する最小の NFE を示す。

50,000 枚の生成画像と利用可能なすべての実画像との間で計算した Fréchet inception distance（FID）[Heu17] により、結果の品質を評価する。[図 2](#figure-02)は、neural function evaluations（NFE）の関数として FID、すなわち 1 枚の画像を生成するために $D_\theta$ が評価される回数を示す。サンプリング過程のコストは全面的に $D_\theta$ のコストに支配されるため、NFE の改善はサンプリング速度の向上に直接つながる。元の決定論的サンプラーを青で示し、本研究の統一された枠組みにおけるこれらの手法の再実装（オレンジ）は、類似しているが一貫してより良い結果をもたらす。この差は、元の実装におけるいくつかの見落としと、DDIM の場合に離散ノイズレベルを本研究がより慎重に扱ったことによって説明される。[第 9 節](#section-9)を参照されたい。元のコードベースは互いに大きく異なる構造をもつものの、本研究の再実装は[アルゴリズム 1](#algorithm-01)と[表 1](#table-01)によって完全に規定されることに注意されたい。

**離散化と高次積分器。** ODE を数値的に解くことは、必然的に真の解軌道をたどることの近似となる。各ステップで、ソルバーは $N$ ステップの過程で累積する*打ち切り誤差*を生じさせる。局所誤差は一般にステップサイズに対して超線形にスケールするため、$N$ を増やすと解の精度が向上する。

一般に用いられる Euler 法は、$\mathcal{O}(h^2)$ の局所誤差をステップサイズ $h$ に対してもつ 1 次 ODE ソルバーである。高次の Runge-Kutta 法 [Sul03] はより有利にスケールするが、1 ステップ当たり $D_\theta$ を複数回評価する必要がある。diffusion model のサンプリングには、linear multistep 法も近年提案されている [Liu22h, Zha22i]。広範な試験を通じて、Heun の 2 次法 [Asc98]（別名 improved Euler、trapezoidal rule、Jolicoeur-Martineau ら [Jol21] が diffusion model の文脈で以前に検討）が、打ち切り誤差と NFE の間で優れたトレードオフを提供することを見いだした。[アルゴリズム 1](#algorithm-01)に示すように、$\boldsymbol{x}_{i+1}$ に追加の補正ステップを導入し、$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$ の $t_i$ と $t_{i+1}$ の間での変化を考慮する。この補正により $\mathcal{O}(h^3)$ の局所誤差が得られ、そのコストはステップごとに $D_\theta$ を 1 回追加評価することである。$\sigma=0$ へ進むとゼロ除算が生じるため、この場合は Euler 法へ戻すことに注意されたい。2 次ソルバーの一般的な族については、[第 10.2 節](#section-10-2)で論じる。

<span id="algorithm-01"></span>

<div class="paper-algorithm">

**アルゴリズム 1：任意の $\sigma(t)$ と $s(t)$ を用いる Heun の 2 次法による決定論的サンプリング。**

- **手順** $\operatorname{HeunSampler}(D_\theta(\boldsymbol{x};\sigma), \sigma(t), s(t), t_{i \in \{0, \dots, N\}})$
  - **サンプル** $\boldsymbol{x}_0 \sim \mathcal{N}(\mathbf{0}, \sigma^2(t_0)s^2(t_0)\mathbf{I})$。$t_0$ で初期サンプルを生成する。
  - **各** $i \in \{0, \dots, N-1\}$：$N$ 個の時刻ステップにわたって[式 4](#equation-04)を解く。
    - $\boldsymbol{d}_i \gets \left(\frac{\dot \sigma(t_i)}{\sigma(t_i)} + \frac{\dot s(t_i)}{s(t_i)}\right)\boldsymbol{x}_i - \frac{\dot\sigma(t_i)s(t_i)}{\sigma(t_i)}D_\theta\left(\frac{\boldsymbol{x}_i}{s(t_i)}; \sigma(t_i)\right)$。$\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ を $t_i$ で評価する。
    - $\boldsymbol{x}_{i+1} \gets \boldsymbol{x}_i + (t_{i+1}-t_i)\boldsymbol{d}_i$。$t_i$ から $t_{i+1}$ への Euler ステップを取る。
    - **もし** $\sigma(t_{i+1}) \ne 0$：$\sigma$ がゼロへ進む場合を除き、2 次補正を適用する。
      - $\boldsymbol{d}'_i \gets \left(\frac{\dot \sigma(t_{i+1})}{\sigma(t_{i+1})} + \frac{\dot s(t_{i+1})}{s(t_{i+1})}\right)\boldsymbol{x}_{i+1} - \frac{\dot\sigma(t_{i+1})s(t_{i+1})}{\sigma(t_{i+1})}D_\theta\left(\frac{\boldsymbol{x}_{i+1}}{s(t_{i+1})}; \sigma(t_{i+1})\right)$。$\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ を $t_{i+1}$ で評価する。
      - $\boldsymbol{x}_{i+1} \gets \boldsymbol{x}_i + (t_{i+1}-t_i)(\frac{1}{2}\boldsymbol{d}_i + \frac{1}{2}\boldsymbol{d}'_i)$。$t_{i+1}$ で explicit trapezoidal rule を適用する。
  - **返す** $\boldsymbol{x}_N$。$t_N$ におけるノイズのないサンプルを返す。

</div>

時刻ステップ $\{t_i\}$ は、ステップサイズ、したがって打ち切り誤差が異なるノイズレベル間でどのように分布するかを決定する。[第 10.1 節](#section-10-1)で詳細な解析を示し、ステップサイズは $\sigma$ の減少に伴って単調に減少すべきであり、サンプルごとに変化させる必要はないと結論づける。本研究では、時刻ステップをノイズレベル列 $\{\sigma_i\}$ に従って、すなわち $t_i=\sigma^{-1}(\sigma_i)$ と定義するパラメータ化された方式を採用する。$\sigma_{i<N} = (Ai + B)^\rho$ と置き、定数 $A$ と $B$ を、$\sigma_0 = \sigma_{\max}$ および $\sigma_{N-1} = \sigma_{\min}$ となるように選択すると、次を得る。<span id="equation-05"></span>

$$
\sigma_{i<N} = \big( {\sigma_{\max}}^\frac{1}{\rho} + {\textstyle\frac{i}{N-1}} ( {\sigma_{\min}}^\frac{1}{\rho} - {\sigma_{\max}}^\frac{1}{\rho} ) \big)^\rho \hspace*{3mm}\text{and}\hspace*{3mm}\sigma_N = 0 \text{.}
$$
ここで、$\rho$ は、$\sigma_{\min}$ 付近のステップを、$\sigma_{\max}$ 付近のステップが長くなることと引き換えに、どの程度短くするかを制御する。[第 10.1 節](#section-10-1)の解析は、$\rho=3$ と置くと各ステップの打ち切り誤差がほぼ均等になる一方、画像のサンプリングでは 5 から 10 の範囲の $\rho$ の方がはるかに良好に機能することを示す。これは、$\sigma_{\min}$ 付近の誤差が大きな影響を及ぼすことを示唆する。本稿の残りでは $\rho=7$ と置く。

Heun 法と[式 5](#equation-05)の結果を、[図 2](#figure-02)の緑の曲線で示す。すべての場合で一貫した改善が見られる。Heun 法は、Euler 法と同じ FID に、かなり低い NFE で到達する。

**軌道の曲率とノイズスケジュール。** ODE の解軌道の形状は、関数 $\sigma(t)$ と $s(t)$ によって定まる。これらの関数の選択は、上で論じた打ち切り誤差を低減する方法を提供する。その大きさは $\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$ の曲率に比例してスケールすると期待できるためである。これらの関数の最良の選択は $\sigma(t)=t$ と $s(t)=1$ であり、これは DDIM [Son21a] で行われた選択でもあると主張する。この選択により、[式 4](#equation-04)の ODE は $\mathrm{d}\boldsymbol{x}/ \mathrm{d}t= \big( \boldsymbol{x}- D(\boldsymbol{x}; t) \big) / t$ に簡約され、$\sigma$ と $t$ は交換可能になる。

その直接的な帰結として、任意の $\boldsymbol{x}$ と $t$ において、$t=0$ までの 1 回の Euler ステップは、denoised image $D_\theta(\boldsymbol{x}; t)$ を与える。したがって、解軌道の接線は常にデノイザーの出力を指す。これはノイズレベルに応じて緩やかにしか変化しないと期待でき、ほぼ線形の解軌道に対応する。[図 3c](#figure-03)の 1D ODE スケッチは、この直観を裏付ける。解軌道はノイズレベルが大きい場合と小さい場合の両方で線形に近づき、その間の狭い領域でのみ大きな曲率をもつ。同じ効果は[図 1b](#figure-01)の実データでも見られ、異なるデノイザーターゲット間の変化は比較的狭い $\sigma$ の範囲で生じる。提唱するスケジュールでは、これは ODE の高い曲率が同じ範囲に限定されることに対応する。

$\sigma(t)=t$ と $s(t)=1$ と置く効果を、[図 2](#figure-02)の赤い曲線で示す。DDIM はすでにこれらと同じ選択を用いているため、ImageNet-64 では赤い曲線と緑の曲線が同一である。しかし、VP と VE は元のスケジュールから切り替えることで大幅に改善する。

<span id="figure-03"></span>

![$p_\text{data}$ が $\boldsymbol{x}=\pm 1$ に 2 つの Dirac peak をもつ場合の、1D における ODE 曲率のスケッチ。](../../papers/diffusion-design-space/figure-03.png)

**図 3。** $p_\text{data}$ が $\boldsymbol{x}=\pm 1$ に 2 つの Dirac peak をもつ場合の、1D における ODE 曲率のスケッチ。水平 $t$ 軸は、各プロットで $\sigma\in[0,25]$ を示すように選び、挿入図はデータ付近の $\sigma\in[0,1]$ を示す。局所勾配の例を黒い矢印で示す。**（a）** Song ら [Son21] の variance preserving ODE は、大きな $\sigma$ で水平線へ平坦化する解軌道をもつ。局所勾配は、小さな $\sigma$ でのみデータを指し始める。**（b）** Variance exploding variant はデータ付近で極端な曲率をもち、解軌道は至る所で曲がっている。**（c）** DDIM [Son21a] と本研究が用いるスケジュールでは、$\sigma$ が増加するにつれて、解軌道はデータの平均を指す直線へ近づく。$\sigma\to 0$ になると、軌道は線形になり、データ多様体を指す。

**考察。** 決定論的サンプリングを改善するため本節で行った選択は、[表 1](#table-01)の*Sampling* 部分にまとめられている。これらを組み合わせると、高品質な結果に到達するために必要な NFE は大幅に削減される。すなわち VP では 7.3$\times$、VE では 300$\times$、DDIM では 3.2$\times$ であり、[図 2](#figure-02)で強調した NFE 値に対応する。実際、1 台の NVIDIA V100 で 1 秒当たり 26.3 枚の高品質な CIFAR-10 画像を生成できる。改善の一貫性は、サンプリング過程が各モデルの元の学習方法と直交するという本研究の仮説を裏付ける。さらなる検証として、本研究のスケジュールを用いた adaptive RK45 法 [Dor80] の結果を、[図 2](#figure-02)の黒い破線で示す。この高度な ODE ソルバーのコストは、その利点を上回る。

<span id="section-4"></span>

## 4 確率的サンプリング

決定論的サンプリングには、たとえば ODE を反転することで実画像を対応する潜在表現へ変換できるなど、多くの利点がある。しかし、各ステップで画像に新たなノイズを注入する確率的サンプリングよりも、出力品質が悪くなる傾向がある [Son21a, Son21]。ODE と SDE が理論上は同じ分布を復元するのであれば、確率性は正確にはどのような役割を果たすのだろうか。

**背景。** Song ら [Son21] の SDE は一般化でき [Hua21b, Zha21l]、[式 1](#equation-01)の確率流 ODE と、時間変化する *Langevin diffusion* SDE [Gre94] との和になる（[第 8.5 節](#section-8-5)を参照）。<span id="equation-06"></span>

$$
\mathrm{d}\boldsymbol{x}_{\pm} =
    \underbrace{-\dot\sigma(t) \sigma(t) \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t}_{\mathrm{probability\ flow\ ODE\ (Eq.\ 1)}}\,\pm\,
    \underbrace{
      \underbrace{\beta(t) \sigma(t)^2 \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t}_{\text{deterministic noise decay}} +
      \underbrace{\sqrt{2 \beta(t)} \sigma(t) \,\mathrm{d}\omega_t}_{\text{noise injection}}
    }_{\text{Langevin diffusion SDE}} ,
$$
ここで、$\omega_t$ は標準 Wiener 過程である。$\mathrm{d}\boldsymbol{x}_+$ と $\mathrm{d}\boldsymbol{x}_-$ は、時間を順方向と逆方向へ進むための別々の SDE となり、Anderson [And82] の時間反転公式によって関係づけられる。Langevin 項はさらに、決定論的な score-based denoising 項と確率的ノイズ注入項の組み合わせとみなせ、その正味のノイズレベルへの寄与は相殺される。したがって、$\beta(t)$ は既存のノイズを新たなノイズで置き換える相対的な率を実質的に表す。$\beta(t) = {\dot \sigma(t)}/{\sigma(t)}$ と選択すると Song ら [Son21] の SDE が復元され、このとき score は順方向 SDE から消える。

<span id="algorithm-02"></span>

<div class="paper-algorithm">

**アルゴリズム 2：$\sigma(t)=t$ と $s(t)=1$ を用いる本研究の確率的サンプラー。**

- **手順** $\operatorname{StochasticSampler}(D_\theta(\boldsymbol{x};\sigma), t_{i \in \{0, \dots, N\}}, \gamma_{i \in \{0, \dots, N-1\}}, S_\mathrm{noise})$
  - **サンプル** $\boldsymbol{x}_0 \sim \mathcal{N}(\mathbf{0}, t_0^2\mathbf{I})$。
  - **各** $i \in \{0, \dots, N-1\}$：
    - $\gamma_i = \begin{cases} \min(S_\mathrm{churn}/N, \sqrt{2}-1) & \text{if } t_i \in [S_\mathrm{tmin},S_\mathrm{tmax}], \\ 0 & \text{otherwise}. \end{cases}$
    - **サンプル** $\boldsymbol{\epsilon}_i \sim \mathcal{N}(\mathbf{0}, S_\mathrm{noise}^2\mathbf{I})$。
    - $\hat t_i \gets t_i + \gamma_i t_i$。一時的に増加させたノイズレベル $\hat t_i$ を選択する。
    - $\boldsymbol{\hat x}_i \gets \boldsymbol{x}_i + \sqrt{\hat t_i^2-t_i^2}\boldsymbol{\epsilon}_i$。$t_i$ から $\hat t_i$ へ移るため、新たなノイズを加える。
    - $\boldsymbol{d}_i \gets (\boldsymbol{\hat x}_i-D_\theta(\boldsymbol{\hat x}_i;\hat t_i))/\hat t_i$。$\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ を $\hat t_i$ で評価する。
    - $\boldsymbol{x}_{i+1} \gets \boldsymbol{\hat x}_i + (t_{i+1}-\hat t_i)\boldsymbol{d}_i$。$\hat t_i$ から $t_{i+1}$ への Euler ステップを取る。
    - **もし** $t_{i+1} \ne 0$：
      - $\boldsymbol{d}'_i \gets (\boldsymbol{x}_{i+1}-D_\theta(\boldsymbol{x}_{i+1};t_{i+1}))/t_{i+1}$。2 次補正を適用する。
      - $\boldsymbol{x}_{i+1} \gets \boldsymbol{\hat x}_i + (t_{i+1}-\hat t_i)(\frac{1}{2}\boldsymbol{d}_i + \frac{1}{2}\boldsymbol{d}'_i)$。
  - **返す** $\boldsymbol{x}_N$。

</div>

この観点は、確率性が実際に役立つ理由を明らかにする：暗黙の Langevin diffusion は、与えられた時刻においてサンプルを所望の周辺分布へ向けて駆動し、それ以前のサンプリングステップで生じた誤差を能動的に補正する。一方、離散的な SDE ソルバーステップで Langevin 項を近似すること自体も誤差を生じさせる。先行研究の結果 [Bao22, Jol21, Son21a, Son21] は、ゼロでない $\beta(t)$ が有用であることを示唆するが、本研究で確認できる限り、Song ら [Son21] における $\beta(t)$ の暗黙の選択には特別な性質がない。したがって、確率性の最適な量は経験的に決定すべきである。

**本研究の確率的サンプラー。** 本研究では、2 次の決定論的 ODE 積分器と、ノイズを加えて取り除く明示的な Langevin-like「churn」とを組み合わせた確率的サンプラーを提案する。擬似コードを[アルゴリズム 2](#algorithm-02)に示す。各ステップ $i$ では、サンプル $\boldsymbol{x}_i$ がノイズレベル $t_i$（$=\sigma(t_i)$）にあるものとして与えられたとき、2 つのサブステップを実行する。第一に、係数 $\gamma_i\ge0$ に従ってサンプルへノイズを加え、より高いノイズレベル $\hat t_i = t_i + \gamma_i t_i$ に到達させる。第二に、ノイズを増加させたサンプル $\boldsymbol{\hat x}_i$ から、$\hat t_i$ から $t_{i+1}$ まで ODE を 1 ステップで逆向きに解く。これにより、サンプル $\boldsymbol{x}_{i+1}$ がノイズレベル $t_{i+1}$ をもつ形で得られ、反復を続ける。これは汎用 SDE ソルバーではなく、この特定の問題に合わせたサンプリング手順であることを強調する。その正しさは、いずれも正しい分布を維持する 2 つのサブステップ（ODE ステップの打ち切り誤差を除く）を交互に行うことに由来する。Song ら [Son21] の predictor-corrector sampler は、概念的に本研究と類似した構造をもつ。

本研究の手法と Euler-Maruyama の主な違いを解析するため、まず後者で[式 6](#equation-06)を離散化する際の微妙な不一致に注目する。Euler-Maruyama は、最初にノイズを加え、その後 ODE ステップを行うものと解釈できるが、その ODE ステップはノイズ注入後の中間状態からではなく、反復ステップの開始時に $\boldsymbol{x}$ と $\sigma$ が初期状態のままであると仮定して行う。本研究の手法では、[アルゴリズム 2](#algorithm-02)の 7 行目で $D_\theta$ を評価するために用いるパラメータはノイズ注入後の状態に対応するが、Euler-Maruyama-like な手法では $\boldsymbol{x}_i;t_i$ を用い、$\boldsymbol{\hat x}_i;\hat t_i$ を用いない。$\Delta_t$ がゼロへ近づく極限では、これらの選択に差がない可能性があるが、大きなステップで低い NFE を追求すると、この違いは重要になるように見える。

**実用上の考慮事項。** 確率性の量を増やすことは、以前のサンプリングステップで生じた誤差の補正に有効だが、それ自体に欠点がある。すべてのデータセットとデノイザーネットワークにおいて、過剰な Langevin-like なノイズの追加と除去により、生成画像の細部が徐々に失われることを観測した（[第 11.1 節](#section-11-1)を参照）。また、ノイズレベルがきわめて低い場合と高い場合には、彩度が過剰な色へのドリフトも生じる。実用的なデノイザーが[式 3](#equation-03)においてわずかに非保存的なベクトル場を誘導し、Langevin diffusion の前提に反して、これらの有害な効果を引き起こすのではないかと推測する。注目すべきことに、解析的デノイザー（[図 1b](#figure-01)のものなど）を用いた本研究の実験では、そのような劣化は見られなかった。

劣化が $D_\theta(\boldsymbol{x}; \sigma)$ の欠陥によって引き起こされるのであれば、サンプリング中にヒューリスティックな手段を用いることでしか改善できない。ノイズレベルの特定の範囲 $t_i \in [S_\text{tmin}, S_\text{tmax}]$ 内でのみ確率性を有効にすることで、彩度が過剰な色へのドリフトに対処する。これらのノイズレベルについて、$\gamma_i = S_\text{churn}/ N$ と定義する。ここで、$S_\text{churn}$ は確率性の総量を制御する。さらに、画像にすでに存在する量を超える新たなノイズを決して導入しないよう、$\gamma_i$ をクランプする。最後に、新たに加えるノイズの標準偏差を増加させるため、$S_\text{noise}$ を $1$ よりわずかに大きく設定することで、細部の損失を部分的に相殺できることを見いだした。これは、$D_\theta(\boldsymbol{x};\sigma)$ に仮定される非保存性の主要な成分が、ノイズをわずかに除去しすぎる傾向であることを示唆する。これはおそらく、あらゆる $L_2$-trained デノイザーで生じると期待される平均への回帰によるものである [Leh18]。

<span id="figure-04"></span>

![本研究の確率的サンプラー（アルゴリズム 2）の評価。](../../papers/diffusion-design-space/figure-04.png)

**図 4。** 本研究の確率的サンプラー（[アルゴリズム 2](#algorithm-02)）の評価。紫の曲線は $\{S_\text{churn}, S_\text{tmin}, S_\text{tmax}, S_\text{noise}\}$ の最適な選択に対応し、オレンジ、青、緑は $S_\text{tmin,tmax}$ および／または $S_\text{noise}$ の効果を無効にした場合に対応する。赤い曲線は、本研究の決定論的サンプラー（[アルゴリズム 1](#algorithm-01)）の参照結果を示し、$S_\text{churn}= 0$ と置くことに等しい。黒い破線は先行研究の元の確率的サンプラー、すなわち VP では Euler–Maruyama [Son21]、VE では predictor-corrector [Son21]、ImageNet-64 では iDDPM [Nic21] に対応する。点は、観測された最小の FID を示す。

**評価。** [図 4](#figure-04)は、本研究の確率的サンプラーが先行するサンプラー [Jol21, Nic21, Son21] を大幅に上回り、特にステップ数が少ない場合に顕著であることを示す。Jolicoeur-Martineau ら [Jol21] は標準的な高次 adaptive SDE ソルバー [Rob12] を用いており、その性能はこの種のソルバー一般に対する良好なベースラインである。本研究のサンプラーは、たとえばノイズ注入と ODE ステップを順次実行するなど、このユースケースに合わせて設計されており、adaptive ではない。diffusion model のサンプリングにおいて、adaptive solver が十分に調整された固定スケジュールに対して正味の利点をもたらせるかは未解決の問題である。

サンプラーの改善だけで、もともと FID 2.07 [Dha21] を達成していた ImageNet-64 モデルを、state-of-the-art にきわめて近い 1.55 まで向上させることができる。従来、cascaded diffusion [Ho22b] では FID 1.48、classifier-free guidance [Ho21] では 1.55、StyleGAN-XL [Sau22] では 1.52 が報告されている。本研究の結果は、サンプラーの改善によって得られる潜在的な向上を示す一方、確率性の主な欠点も浮き彫りにする。最良の結果を得るには、特定のモデルに依存する複数のヒューリスティックな選択を、暗黙的または明示的に行わなければならない。実際、グリッドサーチ（[第 11.2 節](#section-11-2)）を用い、$\{S_\text{churn}, S_\text{tmin}, S_\text{tmax}, S_\text{noise}\}$ の最適値をケースごとに求める必要があった。このことは、モデル改善を評価する主な手段として確率的サンプリングを用いると、モデルアーキテクチャと学習に関する設計選択へ意図せず影響を及ぼす結果になりかねないという、一般的な懸念を生じさせる。

<span id="section-5"></span>

## 5 Preconditioning と学習

ニューラルネットワークを教師あり方式で学習するための、良いと知られている実践はさまざまに存在する。たとえば、入力信号と出力信号の大きさを単位分散などに固定し、サンプルごとの勾配の大きさに大きなばらつきが生じるのを避けることが望ましい [Bis95, Hua20b]。ニューラルネットワークに $D$ を直接モデル化させる学習は理想からほど遠く、たとえば、入力 $\boldsymbol{x}=\boldsymbol{y}+\boldsymbol{n}$ はクリーンな信号 $\boldsymbol{y}$ とノイズ $\boldsymbol{n}\sim\mathcal{N}(\mathbf{0},\sigma^2 \mathbf{I})$ の組み合わせであるため、その大きさはノイズレベル $\sigma$ に応じて著しく変化する。このため、一般的な実践では $D_\theta$ を直接ニューラルネットワークとして表現せず、代わりに別のネットワーク $F_\theta$ を学習し、そこから $D_\theta$ を導出する。

先行手法 [Nic21, Son21a, Son21] は、$\sigma$ に依存する正規化係数によって入力のスケーリングに対処し、$F_\theta$ を、単位分散にスケーリングした $\boldsymbol{n}$ を予測するよう学習することで出力を precondition しようとし、そこから信号を $D_\theta(\boldsymbol{x};\sigma) = \boldsymbol{x}-\sigma F_\theta(\cdot)$ によって再構成する。これには、$\sigma$ が大きい場合、既存のノイズ $\boldsymbol{n}$ を正確に打ち消して正しいスケールで出力するため、ネットワークが出力を注意深く微調整しなければならないという欠点がある。ネットワークが生じさせたあらゆる誤差は、係数 $\sigma$ によって増幅されることに注意されたい。この状況では、期待される出力 $D(\boldsymbol{x}; \sigma)$ を直接予測する方がはるかに容易だと思われる。信号とノイズを適応的に混合する先行するパラメータ化（たとえば [Doc22, Sal22, Vah21]）と同じ発想に基づき、$\sigma$ 依存の skip connection でニューラルネットワークを precondition し、$\boldsymbol{y}$ または $\boldsymbol{n}$、あるいはその中間を推定できるようにすることを提案する。したがって、$D_\theta$ を次の形で書く。<span id="equation-07"></span>

$$
D_\theta(\boldsymbol{x}; \sigma) = c_\text{skip}(\sigma) ~\boldsymbol{x}+ c_\text{out}(\sigma) ~F_\theta \big( c_\text{in}(\sigma) ~\boldsymbol{x}; ~c_\text{noise}(\sigma) \big) \text{,}
$$
ここで、$F_\theta$ は学習対象のニューラルネットワーク、$c_\text{skip}(\sigma)$ は skip connection を調整し、$c_\text{in}(\sigma)$ と $c_\text{out}(\sigma)$ は入力と出力の大きさをスケーリングし、$c_\text{noise}(\sigma)$ はノイズレベル $\sigma$ を $F_\theta$ の条件入力へ写像する。[式 2](#equation-02)のノイズレベルにわたる重み付き期待値を取ると、全体の学習損失 $\mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \left[ \lambda(\sigma) ~ \| D(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\|^2_2 \right]$ が得られる。ここで、$\sigma \sim p_\text{train}$、$\boldsymbol{y}\sim p_\text{data}$、$\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ である。与えられたノイズレベル $\sigma$ をサンプリングする確率は $p_\text{train}(\sigma)$ で与えられ、対応する重みは $\lambda(\sigma)$ で与えられる。[式 7](#equation-07)の生のネットワーク出力 $F_\theta$ に関して、この損失を等価に表せる。<span id="equation-08"></span>

$$
\mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[
    \underbrace{\lambda(\sigma) ~ c_\text{out}(\sigma)^2}_{\text{effective weight}}
    \big\|
      \underbrace{F_\theta \big( c_\text{in}(\sigma) \cdot (\boldsymbol{y}+ \boldsymbol{n}); c_\text{noise}(\sigma) \big)}_{\text{network output}} -
      \underbrace{\tfrac{1}{c_\text{out}(\sigma)} \big(\boldsymbol{y}- c_\text{skip}(\sigma) \cdot (\boldsymbol{y}+ \boldsymbol{n}) \big)}_{\text{effective training target}}
    \big\|^2_2 \Big] \text{.}
$$
この形により $F_\theta$ の実効学習ターゲットが明らかになり、preconditioning 関数の適切な選択を第一原理から決定できる。[第 8.6 節](#section-8-6)で詳述するように、ネットワーク入力と学習ターゲットが単位分散をもつこと（$c_\text{in}$、$c_\text{out}$）、および $F_\theta$ の誤差をできるだけ増幅しないこと（$c_\text{skip}$）を要求することで、[表 1](#table-01)に示した本研究の選択を導出する。$c_\text{noise}$ の式は経験的に選択する。

<span id="table-02"></span>

![本研究の学習改善の評価。開始点（config A）は、本研究の決定論的サンプラーを用いた VP と VE である。最終段階（config E、F）では、VP と VE の違いは $F_\theta$ のアーキテクチャだけである。](../../papers/diffusion-design-space/table-02.png)

**表 2。** 本研究の学習改善の評価。開始点（config A）は、本研究の**決定論的**サンプラーを用いた VP と VE である。最終段階（config E、F）では、VP と VE の違いは $F_\theta$ のアーキテクチャだけである。

[表 2](#table-02)は、[第 3 節](#section-3)の本研究の決定論的サンプラーを用いて評価した、一連の学習設定の FID を示す。Song ら [Son21] のベースライン学習設定から開始し、この設定は VP と VE の場合で大きく異なるため、それぞれについて別々の結果（config A）を示す。より意味のある比較点を得るため、基本ハイパーパラメータを再調整し（config B）、最低解像度の層を取り除いて、代わりに最高解像度の層の容量を 2 倍にすることで、モデルの表現力を改善する（config C）。詳細は[第 12.3 節](#section-12-3)を参照されたい。次に、元の $\{c_\text{in}, c_\text{out}, c_\text{noise}, c_\text{skip}\}$ の選択を本研究の preconditioning（config D）で置き換える。これにより、64$\times$64 解像度で大幅に改善する VE を除けば、結果はほぼ変わらないまま保たれる。FID 自体を改善する代わりに、本研究の preconditioning の主な利点は学習をより頑健にすることであり、悪影響を伴わずに損失関数の再設計へ焦点を移せるようになる。

**損失の重み付けとサンプリング。** [式 8](#equation-08)は、[式 7](#equation-07)のように precondition した $F_\theta$ を学習すると、実効的なサンプルごとの損失重み $\lambda(\sigma)c_\text{out}(\sigma)^2$ が生じることを示す。実効損失重みを均衡させるため、$\lambda(\sigma)=1/c_\text{out}(\sigma)^2$ と置く。これは[図 5a](#figure-05)（緑の曲線）に示すように、$\sigma$ の全範囲にわたって初期学習損失も均等にする。最後に、$p_\text{train}(\sigma)$、すなわち学習中にノイズレベルをどのように選択するかを決める必要がある。学習後の $\sigma$ ごとの損失（青とオレンジの曲線）を調べると、中間のノイズレベルでのみ大幅な低減が可能であることが分かる。きわめて低いレベルでは、消失するほど小さいノイズ成分を識別することは困難であるうえに重要でもない一方、高いレベルでは、学習ターゲットはデータセットの平均へ近づく正解とは常に異なる。したがって、[表 1](#table-01)で詳述し、[図 5a](#figure-05)（赤の曲線）で示すように、$p_\text{train}(\sigma)$ に単純な log-normal 分布を用い、関連する範囲へ学習の労力を集中させる。

[表 2](#table-02)は、提案する $p_\text{train}$ と $\lambda$（config E）が、本研究の preconditioning（config D）と組み合わせて用いると、すべての場合で FID の劇的な改善につながることを示す。同時期の研究で、Choi ら [Cho22c] は、画像の知覚的に認識可能な内容を形成するうえで最も関連性の高いノイズレベルを優先する、類似の方式を提案している。しかし、彼らは $\lambda$ の選択だけを単独で考慮しており、全体的な改善はより小さくなる。

**Augmentation regularization。** 小規模なデータセットで diffusion model を悩ませることの多い潜在的な過学習を防ぐため、GAN の文献 [Kar20a] から augmentation pipeline を借用する。この pipeline は、ノイズを加える前に学習画像へ適用する、さまざまな幾何変換（[第 12.2 節](#section-12-2)を参照）から成る。augmentation が生成画像へ漏れ出すのを防ぐため、augmentation パラメータを $F_\theta$ の条件入力として与え、推論時には augmentation されていない画像だけが生成されることを保証するため、それらをゼロに置く。[表 2](#table-02)は、data augmentation が一貫した改善（config F）をもたらし、条件付きおよび無条件 CIFAR-10 で 1.79 と 1.97 という新たな state-of-the-art FID を達成して、従来の記録である 1.85 [Sau22] と 2.10 [Vah21] を上回ることを示す。

<span id="figure-05"></span>

![（a）本稿で検討する 32$\times$32（青）および 64$\times$64（オレンジ）のモデルを代表する、ノイズレベルごとに観測された初期損失（緑）と最終損失。](../../papers/diffusion-design-space/figure-05.png)

**図 5。** **（a）** 本稿で検討する 32$\times$32（青）および 64$\times$64（オレンジ）のモデルを代表する、ノイズレベルごとに観測された初期損失（緑）と最終損失。網掛け領域は、10k 個のランダムサンプルにわたる標準偏差を表す。提案する学習サンプル密度を赤い破線で示す。**（b）** 無条件 CIFAR-10 に対する $S_\text{churn}$ の影響（256 ステップ、NFE $=$ 511）。Song ら [Son21] の元の学習設定では、確率的サンプリングが非常に有益である（青、緑）一方、決定論的サンプリング（$S_\text{churn}= 0$）は比較的悪い FID につながる。本研究の学習設定では状況が逆転し（オレンジ、赤）、確率的サンプリングは不要なだけでなく有害である。**（c）** クラス条件付き ImageNet-64 に対する $S_\text{churn}$ の影響（256 ステップ、NFE $=$ 511）。このより困難なシナリオでは、確率的サンプリングが再び有用であることが分かる。本研究の学習設定は、決定論的サンプリングと確率的サンプリングの両方で結果を改善する。

**確率的サンプリングの再検討。** 興味深いことに、[図 5b](#figure-05)、[図 5c](#figure-05)に示すように、モデル自体が改善するにつれて、確率的サンプリングの重要性は低下するように見える。CIFAR-10 で本研究の学習設定を用いた場合（[図 5b](#figure-05)）、最良の結果は決定論的サンプリングで得られ、いかなる量の確率的サンプリングも有害であった。

**ImageNet-64。** 最後の実験として、提案する学習改善を用い、クラス条件付き ImageNet-64 モデルをゼロから学習した。このモデルは、従来の記録である 1.48 [Ho22b] に対し、1.36 という新たな state-of-the-art FID を達成した。変更を加えずに ADM アーキテクチャ [Dha21] を使用し、最小限の調整を施した config E を用いて学習した。詳細は[第 12.3 節](#section-12-3)を参照されたい。過学習は懸念ではないと判断したため、augmentation regularization は採用しなかった。[図 5c](#figure-05)に示すように、確率的サンプリングの最適量は事前学習済みモデルの場合よりはるかに少なかったが、CIFAR-10 の場合とは異なり、確率的サンプリングは決定論的サンプリングより明らかに優れていた。これは、より多様なデータセットが引き続き確率的サンプリングの恩恵を受けることを示唆する。

<span id="section-6"></span>

## 6 結論

diffusion model を共通の枠組みに置く本研究のアプローチは、モジュール化された設計を明らかにする。これにより、個々の構成要素を対象とした調査が可能となり、実用可能な設計空間をより十分に網羅する助けとなる可能性がある。本研究の試験では、これによりさまざまな先行モデルのサンプラーを単純に置き換え、結果を劇的に改善できた。たとえば ImageNet-64 では、本研究のサンプラーにより、平均的なモデル（FID 2.07）を従来の SOTA モデル（1.48）[Ho22b] に対抗するモデル（1.55）へ変え、学習の改善と併用して SOTA FID 1.36 を達成した。また、35 回のモデル評価、決定論的サンプリング、小規模なネットワークだけを用いながら、CIFAR-10 で新たな state-of-the-art の結果も得た。現在の高解像度 diffusion model は、個別の super-resolution ステップ [Ho22b, Nic22, Ram22]、subspace projection [Jin22a]、非常に大規模なネットワーク [Dha21, Son21]、または hybrid approach [Pre22, Rom22, Vah21] のいずれかに依存しているが、本研究の貢献はこれらの拡張と直交すると考えている。とはいえ、本研究のパラメータ値の多くは、より高解像度のデータセットに対して再調整が必要となる可能性がある。さらに、確率的サンプリングと学習目的との正確な相互作用は、今後の研究にとって興味深い問いであり続けると考えている。

**社会的影響。** 本研究によるサンプル品質の進歩は、DALL$\cdot$E 2 のような大規模システムで使用された場合、偽情報の類型や、ステレオタイプと有害なバイアスの強調を含む、社会への悪影響を増幅する可能性がある [Mis22a]。diffusion model の学習とサンプリングには大量の電力が必要であり、本プロジェクトは NVIDIA V100 から成る社内クラスターで $\sim$250MWh を消費した。

## 謝辞

議論とコメントをいただいた Jaakko Lehtinen、Ming-Yu Liu、Tuomas Kynkäänniemi、Axel Sauer、Arash Vahdat、Janne Hellsten に、また計算インフラストラクチャを維持してくださった Tero Kuosmanen、Samuel Klenberg、Janne Hellsten に感謝する。

<span id="section-7"></span>

## 7 追加結果

<span id="table-03"></span>

![決定論的サンプリングに対する本研究の改善の評価。値は図 2 に示した曲線に対応する。「—」を付した値は、その上の値と同一である。これは、本研究のサンプラーが DDIM と同じ $\sigma(t)$ および $s(t)$ を使用するためである。](../../papers/diffusion-design-space/table-03.png)

**表 3。** 決定論的サンプリングに対する本研究の改善の評価。値は[図 2](#figure-02) に示した曲線に対応する。「—」を付した値は、その上の値と同一である。これは、本研究のサンプラーが DDIM と同じ $\sigma(t)$ および $s(t)$ を使用するためである。

<span id="table-04"></span>

![確率的サンプリングに対する本研究の改善の評価とアブレーション。値は図 4 に示した曲線に対応する。](../../papers/diffusion-design-space/table-04.png)

**表 4。** 確率的サンプリングに対する本研究の改善の評価とアブレーション。値は[図 4](#figure-04) に示した曲線に対応する。

<span id="figure-06"></span>

![Dhariwal と Nichol による事前学習済みモデルを用いた、64$\times$64 解像度のクラス条件付き ImageNet における各種サンプラーの結果。](../../papers/diffusion-design-space/figure-06.png)

**図 6。** Dhariwal と Nichol による事前学習済みモデル [Dha21] を用いた、64$\times$64 解像度のクラス条件付き ImageNet [Den09a] における各種サンプラーの結果。各ケースは[図 2c](#figure-02) および[図 4c](#figure-04) の点に対応する。

<span id="figure-07"></span>

![本研究の決定論的サンプラーと確率的サンプラーを用いた、64$\times$64 解像度のクラス条件付き ImageNet における本研究の学習構成の結果。](../../papers/diffusion-design-space/figure-07.png)

**図 7。** 本研究の決定論的サンプラーと確率的サンプラーを用いた、64$\times$64 解像度のクラス条件付き ImageNet [Den09a] における本研究の学習構成の結果。

<span id="figure-08"></span>

![Song らによる事前学習済みモデルを用いた、32$\times$32 解像度の無条件 CIFAR-10 における各種サンプラーの結果。](../../papers/diffusion-design-space/figure-08.png)

**図 8。** Song らによる事前学習済みモデル [Son21] を用いた、32$\times$32 解像度の無条件 CIFAR-10 [Kri09] における各種サンプラーの結果。各ケースは[図 2a](#figure-02)、[図 2b](#figure-02)、[図 4a](#figure-04)、[図 4b](#figure-04) の点に対応する。

<span id="figure-09"></span>

![32$\times$32 解像度の無条件 CIFAR-10 において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。](../../papers/diffusion-design-space/figure-09.png)

**図 9。** 32$\times$32 解像度の無条件 CIFAR-10 [Kri09] において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。

<span id="figure-10"></span>

![32$\times$32 解像度のクラス条件付き CIFAR-10 において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。](../../papers/diffusion-design-space/figure-10.png)

**図 10。** 32$\times$32 解像度のクラス条件付き CIFAR-10 [Kri09] において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。

<span id="figure-11"></span>

![64$\times$64 解像度の FFHQ および AFHQv2 において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。](../../papers/diffusion-design-space/figure-11.png)

**図 11。** 64$\times$64 解像度の FFHQ [Kar18] および AFHQv2 [Cho20c] において、各ケースで同じ潜在コード集合（$\boldsymbol{x}_0$）を用い、本研究の決定論的サンプラーを使用した、各種学習構成の結果。

<span id="figure-12"></span>

![本研究の決定論的サンプラーを用いた場合の、NFE の関数としての画像品質と FID。](../../papers/diffusion-design-space/figure-12.png)

**図 12。** 本研究の決定論的サンプラーを用いた場合の、NFE の関数としての画像品質と FID。32$\times$32 解像度では、NFE $=$ 13 前後で妥当な画像品質に達するが、FID は NFE $=$ 35 まで改善し続ける。64$\times$64 解像度では、NFE $=$ 19 前後で妥当な画像品質に達するが、FID は NFE $=$ 79 まで改善し続ける。

[図 6](#figure-06) は、Dhariwal と Nichol による事前学習済み ADM モデル [Dha21] を用いた、クラス条件付き ImageNet-64 [Den09a] の生成画像を示す。元の DDIM [Son21a] および iDDPM [Nic21] サンプラーを、決定論的設定と確率的設定の両方で本研究のサンプラーと比較している（[第 3 節](#section-3)および[第 4 節](#section-4)）。[図 7](#figure-07) は、本研究の改善された学習構成（[第 5 節](#section-5)）を用いてモデルをゼロから学習することで得られた、対応する結果を示す。

Song ら [Son21] による元のサンプラーおよび学習構成を、[図 8](#figure-08)と[図 9](#figure-09)（無条件 CIFAR-10 [Kri09]）、[図 10](#figure-10)（クラス条件付き CIFAR-10）、および[図 11](#figure-11)（FFHQ [Kar18] と AFHQv2 [Cho20c]）で本研究のものと比較している。比較を容易にするため、異なる学習構成および ODE の選択肢にわたって、各データセット／シナリオで同じ潜在コード $\boldsymbol{x}_0$ を使用する。[図 12](#figure-12) は、決定論的サンプリングを用いた場合のさまざまな NFE における生成画像の品質を示す。

[表 3](#table-03)と[表 4](#table-04)は、さまざまなデータセットにおける決定論的および確率的サンプリング手法の数値結果をまとめたものであり、これらは先に[図 2](#figure-02)と[図 4](#figure-04)で NFE の関数として示した。

<span id="section-8"></span>

## 8 式の導出

<span id="section-8-1"></span>

### 8.1 先行研究における元の ODE / SDE 定式化

Song ら [Son21] は、順方向 SDE（[Son21] の式 5）を
$$
\mathrm{d}\boldsymbol{x}= \boldsymbol{f}(\boldsymbol{x}, t) ~\mathrm{d}t + g(t) ~\mathrm{d}\omega_t
  \text{,}
$$
と定義する。ここで、$\omega_t$ は標準 Wiener 過程、$\boldsymbol{f}(\cdot, t): \mathbb{R}^d \rightarrow \mathbb{R}^d$ と $g(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ はそれぞれドリフト係数と拡散係数であり、$d$ はデータセットの次元数である。これらの係数は、variance preserving（VP）定式化と variance exploding（VE）定式化とで異なるように選択され、$\boldsymbol{f}(\cdot)$ は常に $\boldsymbol{f}(\boldsymbol{x}, t) = f(t) ~\boldsymbol{x}$ の形をとる。ここで、$f(\cdot): \mathbb{R} \rightarrow \mathbb{R}$ である。したがって、この SDE は等価に次のように書ける。<span id="equation-10"></span>

$$
\mathrm{d}\boldsymbol{x}= f(t) ~\boldsymbol{x}~\mathrm{d}t + g(t) ~\mathrm{d}\omega_t
  \text{.}
$$

この SDE の摂動カーネル（[Son21] の式 29）は一般に次の形をもつ。<span id="equation-11"></span>

$$
p_{0t}\big( \boldsymbol{x}(t) ~|~ \boldsymbol{x}(0) \big) = \mathcal{N} \big( \boldsymbol{x}(t); ~s(t) ~\boldsymbol{x}(0), ~s(t)^2 ~\sigma(t)^2 ~\mathbf{I}\big)
  \text{,}
$$
ここで、$\mathcal{N}(\boldsymbol{x}; \boldsymbol{\mu}, \boldsymbol{\Sigma})$ は、$\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ の確率密度関数を $\boldsymbol{x}$ において評価したものを表し、次が成り立つ。<span id="equation-12"></span>

$$
s(t) = \exp\left( \int_0^t f(\xi) ~\mathrm{d}\xi \right)
  \text{,}
  \hspace{4mm}\text{and}\hspace{4mm}
  \sigma(t) = \sqrt{\int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi}
  \text{.}
$$

周辺分布 $p_t(\boldsymbol{x})$ は、摂動カーネルを $\boldsymbol{x}(0)$ について積分することで得られる。<span id="equation-13"></span>

$$
p_t(\boldsymbol{x}) = \int_{\mathbb{R}^d} p_{0t}(\boldsymbol{x}~|~ \boldsymbol{x}_0) ~p_\text{data}(\boldsymbol{x}_0) ~\mathrm{d}\boldsymbol{x}_0
  \text{.}
$$

Song ら [Son21] は、これと同じ $p_t(\boldsymbol{x})$ に従うよう、確率流 ODE（[Son21] の式 13）を次のように定義する。<span id="equation-14"></span>

$$
\mathrm{d}\boldsymbol{x}= \left[ f(t) ~\boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) \right] ~\mathrm{d}t
  \text{.}
$$

<span id="section-8-2"></span>

### 8.2 本研究の ODE 定式化（[式 1](#equation-01)および[式 4](#equation-04)）

元の ODE 定式化（[式 14](#equation-14)）は、式中に現れる特定の項に直接対応する関数 $f$ と $g$ を中心に構築されており、周辺分布の性質（[式 12](#equation-12)）はこれらの関数に基づいて間接的にしか導出できない。しかし、$f$ と $g$ 自体は実用上ほとんど関心の対象ではない一方、そもそものモデル学習、サンプリング過程のブートストラップ、および ODE が実際にどのように振る舞うかの理解という点では、周辺分布がきわめて重要である。確率流 ODE の考え方が特定の周辺分布集合に一致させることである以上、周辺分布を第一級の対象として扱い、$\sigma(t)$ と $s(t)$ に基づいて ODE を直接定義して、$f(t)$ と $g(t)$ を不要にするのが妥当である。

まず、[式 13](#equation-13)の周辺分布を閉形式で表す。
$$
\begin{aligned}
  p_t(\boldsymbol{x}) &= \int_{\mathbb{R}^d} p_{0t}(\boldsymbol{x}~|~ \boldsymbol{x}_0) ~p_\text{data}(\boldsymbol{x}_0) ~\mathrm{d}\boldsymbol{x}_0 \\
  &= \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\Big[ \mathcal{N} \big( \boldsymbol{x}; ~s(t) ~\boldsymbol{x}_0, ~s(t)^2 ~\sigma(t)^2 ~\mathbf{I}\big) \Big] ~\mathrm{d}\boldsymbol{x}_0 \\
  &= \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\Big[ s(t)^{-d} ~\mathcal{N} \big( \boldsymbol{x}/ s(t); ~\boldsymbol{x}_0, ~\sigma(t)^2 ~\mathbf{I}\big) \Big] ~\mathrm{d}\boldsymbol{x}_0 \\
  &= s(t)^{-d} \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\mathcal{N} \big( \boldsymbol{x}/ s(t); ~\boldsymbol{x}_0, ~\sigma(t)^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &= s(t)^{-d} ~\Big[ p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \Big] \big( \boldsymbol{x}/ s(t) \big)
  \text{,}
\end{aligned}
$$

ここで、$p_a \ast p_b$ は確率密度関数 $p_a$ と $p_b$ の畳み込みを表す。角括弧内の式は、サンプルに i.i.d. Gaussian ノイズを加えることで得られる、$p_\text{data}$ を平滑化したものに対応する。この分布を $p(\boldsymbol{x}; \sigma)$ と表すことにする。<span id="equation-20"></span>

$$
p(\boldsymbol{x}; \sigma) = p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big)
  \hspace{5mm}\text{and}\hspace{5mm}
  p_t(\boldsymbol{x}) = s(t)^{-d} ~p\big( \boldsymbol{x}/ s(t); \sigma(t) \big)
  \text{.}
$$

これで、$p(\boldsymbol{x}; \sigma)$ を用いて、$p_t(\boldsymbol{x})$ による表現に代わる形で確率流 ODE（[式 14](#equation-14)）を表せる。<span id="equation-24"></span>

$$
\begin{aligned}
  \mathrm{d}\boldsymbol{x}&= \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \big[ p_t(\boldsymbol{x}) \big] \right] ~\mathrm{d}t \\
  &= \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \big[ s(t)^{-d} ~p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] \right] ~\mathrm{d}t \\
  &= \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\big[ \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log s(t)^{-d} + \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] \right] ~\mathrm{d}t \\
  &= \left[ f(t) \boldsymbol{x}- \tfrac{1}{2} ~g(t)^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t

  \text{.}
\end{aligned}
$$

次に、[式 12](#equation-12)に基づいて $f(t)$ を $s(t)$ で書き換える。<span id="equation-28"></span>

$$
\begin{aligned}
  \exp\left( \int_0^t f(\xi) ~\mathrm{d}\xi \right) &= s(t) \\
  \int_0^t f(\xi) ~\mathrm{d}\xi &= \log s(t) \\
  \mathrm{d}\bigg[ \int_0^t f(\xi) ~\mathrm{d}\xi \bigg] \big/ \mathrm{d}t &= \mathrm{d}\big[ \log s(t) \big] / \mathrm{d}t \\
  f(t) &= \dot s(t) / s(t)

  \text{.}
\end{aligned}
$$

同様に、$g(t)$ も $\sigma(t)$ で書き換えられる。<span id="equation-34"></span>

$$
\begin{aligned}
  \sqrt{\int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi} &= \sigma(t) \\
  \int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi &= \sigma(t)^2 \\
  \mathrm{d}\bigg[ \int_0^t \frac{g(\xi)^2}{s(\xi)^2} ~\mathrm{d}\xi \bigg] \big/ \mathrm{d}t &= \mathrm{d}\big[ \sigma(t)^2 \big] / \mathrm{d}t \\
  g(t)^2 / s(t)^2 &= 2 ~\dot\sigma(t) ~\sigma(t) \\
  g(t) / s(t) &= \sqrt{2 ~\dot\sigma(t) ~\sigma(t)} \\
  g(t) &= s(t) ~\sqrt{2 ~\dot\sigma(t) ~\sigma(t)}

  \text{.}
\end{aligned}
$$

最後に、$f$（[式 28](#equation-28)）と $g$（[式 34](#equation-34)）を[式 24](#equation-24)の ODE に代入する。
$$
\begin{aligned}
  \mathrm{d}\boldsymbol{x}&= \left[ [f(t)] ~\boldsymbol{x}- \tfrac{1}{2} ~[g(t)]^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t \\
  &= \bigg[ \big[ \dot s(t) / s(t) \big] ~\boldsymbol{x}- \tfrac{1}{2} ~\Big[ s(t) \sqrt{2 ~\dot\sigma(t) ~\sigma(t)} \Big]^2 ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \bigg] ~\mathrm{d}t \\
  &= \bigg[ \big[ \dot s(t) / s(t) \big] ~\boldsymbol{x}- \tfrac{1}{2} ~\Big[ 2 ~s(t)^2 ~\dot\sigma(t) ~\sigma(t) \Big] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \bigg] ~\mathrm{d}t \\
  %
  &= \left[ \frac{\dot s(t)}{s(t)} ~\boldsymbol{x}-s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\left(\frac{\boldsymbol{x}}{s(t)}; \sigma(t)\right) \right] ~\mathrm{d}t
  \text{.}
\end{aligned}
$$

こうして本文の[式 4](#equation-04)が得られ、$s(t) = 1$ と置けば[式 1](#equation-01)が復元される。
$$
\mathrm{d}\boldsymbol{x}= -\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t
  \text{.}
$$

本研究の定式化（[式 4](#equation-04)）は、確率流 ODE のあらゆる実現が、同じ正準 ODE の単なる再パラメータ化であることを明確に示している。$\sigma(t)$ の変更は $t$ の再パラメータ化に対応し、$s(t)$ の変更は $\boldsymbol{x}$ の再パラメータ化に対応する。

<span id="section-8-3"></span>

### 8.3 Denoising score matching（[式 2](#equation-02)および[式 3](#equation-03)）

完全を期すため、有限データセットに対する score matching と denoising の関係を導出する。この主題のより一般的な扱いとさらなる背景については、Hyvärinen [Hyv05] および Vincent [Vin11] を参照されたい。

学習集合が有限個のサンプル $\{\boldsymbol{y}_1, \dots, \boldsymbol{y}_Y\}$ から成ると仮定する。これは、$p_\text{data}(\boldsymbol{x})$ が Dirac delta 分布の混合で表されることを意味する。
$$
p_\text{data}(\boldsymbol{x}) = \frac{1}{Y} \sum_{i=1}^Y \delta \big( \boldsymbol{x}- \boldsymbol{y}_i \big)
  \text{,}
$$
これにより、[式 20](#equation-20)に基づいて $p(\boldsymbol{x}; \sigma)$ も閉形式で表せる。<span id="equation-45"></span>

$$
\begin{aligned}
  p(\boldsymbol{x}; \sigma) &= p_\text{data}\ast \mathcal{N} \big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \\
  &= \int_{\mathbb{R}^d} p_\text{data}(\boldsymbol{x}_0) ~\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &= \int_{\mathbb{R}^d} \Bigg[ \frac{1}{Y} \sum_{i=1}^Y \delta \big( \boldsymbol{x}_0 - \boldsymbol{y}_i \big) \Bigg] \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &= \frac{1}{Y} \sum_{i=1}^Y \int_{\mathbb{R}^d} \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{x}_0, ~\sigma^2 ~\mathbf{I}\big) ~\delta \big( \boldsymbol{x}_0 - \boldsymbol{y}_i \big) ~\mathrm{d}\boldsymbol{x}_0 \\
  &= \frac{1}{Y} \sum_{i=1}^Y \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big)

  \text{.}
\end{aligned}
$$

次に、[式 2](#equation-02)の denoising score matching 損失を考える。期待値を展開することで、この式をノイズを加えたサンプル $\boldsymbol{x}$ に関する積分として書き換えられる。<span id="equation-50"></span>

$$
\begin{aligned}
  \mathcal{L}(D; \sigma) &= \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\| D(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\|^2_2 \\
  &= \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{x}\sim \mathcal{N}(\boldsymbol{y}, \sigma^2 \mathbf{I})} ~\big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}\big\|^2_2 \\
  &= \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} \int_{\mathbb{R}^d} \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}, ~\sigma^2 ~\mathbf{I}) ~\big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}\big\|^2_2 ~\mathrm{d}\boldsymbol{x}\\
  &= \frac{1}{Y} \sum_{i=1}^Y \int_{\mathbb{R}^d} \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\|^2_2 ~\mathrm{d}\boldsymbol{x}\\
  &= \int_{\mathbb{R}^d} \underbrace{\frac{1}{Y} \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\|^2_2}_{=: ~ \mathcal{L}(D; \boldsymbol{x}, \sigma)}  ~\mathrm{d}\boldsymbol{x}
  %

  \text{.}
\end{aligned}
$$

[式 50](#equation-50)は、$\mathcal{L}(D; \sigma)$ を最小化するために、$\mathcal{L}(D; \boldsymbol{x}, \sigma)$ を各 $\boldsymbol{x}$ について独立に最小化できることを意味する。
$$
D(\boldsymbol{x}; \sigma) = \mathop{\mathrm{arg}\,\min}_{D(\boldsymbol{x}; \sigma)} \mathcal{L}(D; \boldsymbol{x}, \sigma)
  \text{.}
$$
これは凸最適化問題であり、$D(\boldsymbol{x}; \sigma)$ に関する勾配をゼロに置くことで、その解は一意に定まる。<span id="equation-57"></span>

$$
\begin{aligned}
  \mathbf{0}&= \nabla_{D(\boldsymbol{x}; \sigma)} \Big[ \mathcal{L}(D; \boldsymbol{x}, \sigma) \Big] \\
  \mathbf{0}&= \nabla_{D(\boldsymbol{x}; \sigma)} \Bigg[ \frac{1}{Y} \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\|^2_2 \Bigg] \\
  \mathbf{0}&= \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\nabla_{D(\boldsymbol{x}; \sigma)} \Big[ \big\| D(\boldsymbol{x}; \sigma) - \boldsymbol{y}_i \big\|^2_2 \Big] \\
  \mathbf{0}&= \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\Big[ 2 ~D(\boldsymbol{x}; \sigma) - 2~\boldsymbol{y}_i \Big] \\
  \mathbf{0}&= \Bigg[ \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) \Bigg] D(\boldsymbol{x}; \sigma) - \sum_{i=1}^Y \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\boldsymbol{y}_i \\
  D(\boldsymbol{x}; \sigma) &= \frac{ \sum_i \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) ~\boldsymbol{y}_i }{ \sum_i \mathcal{N}(\boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}) }

  \text{,}
\end{aligned}
$$
これは、理想的なデノイザー $D(\boldsymbol{x}; \sigma)$ の閉形式解を与える。[式 57](#equation-57)は小規模なデータセットでは実際に計算可能であることに注意されたい。CIFAR-10 について、その結果を[図 1b](#figure-01)に示す。

次に、[式 45](#equation-45)で定義した分布 $p(\boldsymbol{x}; \sigma)$ の score を考える。<span id="equation-60"></span>

$$
\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) &= \frac{\nabla_{\hspace{-0.5mm}\boldsymbol{x}}p(\boldsymbol{x}; \sigma)}{p(\boldsymbol{x}; \sigma)} \\
  &= \frac{ \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Big[ \frac{1}{Y} \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big] }{ \Big[ \frac{1}{Y} \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big] } \\
  &= \frac{ \sum_i \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }

  \text{.}
\end{aligned}
$$

[式 60](#equation-60)の分子はさらに簡約できる。
$$
\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) &= \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} ~\exp \frac{\| \boldsymbol{x}- \boldsymbol{y}_i \|_2^2}{-2 ~\sigma^2} \Bigg] \\
  &= \big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \exp \frac{\| \boldsymbol{x}- \boldsymbol{y}_i \|_2^2}{-2 ~\sigma^2} \Bigg] \\
  &= \Bigg[\big( 2 \pi \sigma^2 \big)^{-\frac{d}{2}} \exp \frac{\| \boldsymbol{x}- \boldsymbol{y}_i \|_2^2}{-2 ~\sigma^2} \Bigg] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \frac{\| \boldsymbol{x}- \boldsymbol{y}_i \|_2^2}{-2 ~\sigma^2} \Bigg] \\
  &= \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\Bigg[ \frac{\| \boldsymbol{x}- \boldsymbol{y}_i \|_2^2}{-2 ~\sigma^2} \Bigg] \\
  &= \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \bigg[ \frac{\boldsymbol{y}_i - \boldsymbol{x}}{\sigma^2} \bigg]
  \text{.}
\end{aligned}
$$

この結果を[式 60](#equation-60)に戻して代入する。<span id="equation-68"></span>

$$
\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) &= \frac{ \sum_i \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } \\
  &= \frac{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \Big[ \frac{\boldsymbol{y}_i - \boldsymbol{x}}{\sigma^2} \Big] }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } \\
  &= \Bigg( \frac{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) \boldsymbol{y}_i }{ \sum_i \mathcal{N} \big( \boldsymbol{x}; ~\boldsymbol{y}_i, ~\sigma^2 ~\mathbf{I}\big) } - \boldsymbol{x}\Bigg) \big/ \sigma^2

  \text{.}
\end{aligned}
$$

[式 68](#equation-68)の分数が[式 57](#equation-57)と同一であることに注目されたい。したがって、[式 68](#equation-68)は等価に次のように書ける。
$$
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p(\boldsymbol{x}; \sigma) = \big( D(\boldsymbol{x}; ~\sigma) - \boldsymbol{x}\big) / \sigma^2
  \text{,}
$$
これは本文の[式 3](#equation-03)と一致する。

<span id="section-8-4"></span>

### 8.4 本研究の ODE の実際の評価（[アルゴリズム 1](#algorithm-01)）

$\boldsymbol{x}$ を元のスケーリングされていない変数 $\hat\boldsymbol{x}$ のスケーリング版とみなし、$\boldsymbol{x}= s(t) ~\hat\boldsymbol{x}$ を、本研究のスケーリングされた ODE（[式 4](#equation-04)）に現れる score 項へ代入する。
$$
\begin{aligned}
  && \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \\
  &= \nabla_{[ s(t) \hat \boldsymbol{x}]} \log p\big( [s(t) ~\hat\boldsymbol{x}] / s(t); \sigma(t) \big) \\
  &= \nabla_{s(t) \hat \boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) \\
  &= \tfrac{1}{s(t)} \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big)
  \text{.}
\end{aligned}
$$

[式 3](#equation-03)を用いることで、これを $D(\cdot)$ に関してさらに書き換えられる。<span id="equation-74"></span>

$$
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) ~=~ \tfrac{1}{s(t) \sigma(t)^2} \Big( D\big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big)

  \text{.}
$$

ここで、理想的なデノイザー $D(\cdot)$ を学習済みモデル $D_\theta(\cdot)$ で近似し、[式 74](#equation-74)を[式 4](#equation-04)へ代入する。
$$
\begin{aligned}
  \mathrm{d}\boldsymbol{x}&= \left[ \dot s(t) ~\boldsymbol{x}/ s(t) - s(t)^2 ~\dot\sigma(t) ~\sigma(t) ~\Big[ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) \Big] \right] ~\mathrm{d}t \\
  &= \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) \right] ~\mathrm{d}t
  \text{.}
\end{aligned}
$$

最後に、$\hat\boldsymbol{x}= \boldsymbol{x}/ s(t)$ を逆代入する。<span id="equation-80"></span>

$$
\begin{aligned}
  \mathrm{d}\boldsymbol{x}&= \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( [\hat\boldsymbol{x}]; \sigma(t) \big) - [\hat\boldsymbol{x}] \Big) \right] ~\mathrm{d}t \\
  &= \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} \Big( D_\theta \big( [\boldsymbol{x}/ s(t)]; \sigma(t) \big) - [\boldsymbol{x}/ s(t)] \Big) \right] ~\mathrm{d}t \\
  &= \left[ \tfrac{\dot s(t)}{s(t)} ~\boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \big( \boldsymbol{x}/ s(t); \sigma(t) \big) + \tfrac{\dot\sigma(t)}{\sigma(t)} ~\boldsymbol{x}\right] ~\mathrm{d}t \\
  &= \left[ \left( \tfrac{\dot\sigma(t)}{\sigma(t)} + \tfrac{\dot s(t)}{s(t)} \right) \boldsymbol{x}- \tfrac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \big( \boldsymbol{x}/ s(t); \sigma(t) \big) \right] ~\mathrm{d}t

  \text{.}
\end{aligned}
$$

[式 80](#equation-80)は等価に次のように書ける。
$$
\mathrm{d}\boldsymbol{x}/ \mathrm{d}t = \bigg( \frac{\dot\sigma(t)}{\sigma(t)} + \frac{\dot s(t)}{s(t)} \bigg) \boldsymbol{x}- \frac{\dot\sigma(t) s(t)}{\sigma(t)} D_\theta \bigg( \frac{\boldsymbol{x}}{s(t)}; \sigma(t) \bigg)
  \text{,}
$$
これは[アルゴリズム 1](#algorithm-01)の 4 行目および 7 行目と一致する。

<span id="section-8-5"></span>

### 8.5 本研究の SDE 定式化（[式 6](#equation-06)）

本研究では、次の方針で[式 6](#equation-06)の SDE を導出する。

- 所望の周辺密度 $p\big( \boldsymbol{x}; \sigma(t) \big)$ は、データ密度 $p_\text{data}$ と、標準偏差 $\sigma(t)$ をもつ等方 Gaussian 密度との畳み込みである（[式 20](#equation-20)を参照）。したがって、密度を時間 $t$ の関数とみなすと、時間変化する拡散率をもつ熱拡散 PDE に従って発展する。第一段階として、この PDE を求める。

- 次に、Fokker-Planck 方程式を用いて、この PDE に従って密度が発展する SDE の族を復元する。[式 6](#equation-06)は、この族の適切なパラメータ化から得られる。

<span id="section-8-5-1"></span>

#### 8.5.1 熱拡散による周辺分布の生成

確率密度 $q(\boldsymbol{x}, t)$ の時間発展を考える。目標は、初期値 $q(\boldsymbol{x}, 0) := p_\text{data}(\boldsymbol{x})$ に対する解が $q(\boldsymbol{x}, t) = p\big( \boldsymbol{x}, \sigma(t) \big)$ となる PDE を求めることである。すなわち、この PDE は[式 20](#equation-20)で仮定した周辺分布を再現すべきである。

所望の周辺分布は、$p_\text{data}$ と、時間変化する標準偏差 $\sigma(t)$ をもつ等方正規分布との畳み込みであるため、時間変化する拡散率 $\kappa(t)$ をもつ熱方程式によって生成できる。この状況は Fourier 領域で最も簡便に解析できる。そこでは、周辺密度は Gaussian 関数と変換後のデータ密度との単なる点ごとの積になる。正しい標準偏差を生じさせる拡散率を求めるため、まず熱方程式の PDE を書き下す。<span id="equation-82"></span>

$$
\frac{\partial q(\boldsymbol{x}, t)}{\partial t} = \kappa(t) {\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)

  \text{.}
$$

$\boldsymbol{x}$ 次元に沿って変換した[式 82](#equation-82)の Fourier 変換版は、次で与えられる。<span id="equation-83"></span>

$$
\frac{\partial \hat q(\boldsymbol{\nu}, t)}{\partial t} = - \kappa(t) |\boldsymbol{\nu}|^2 \hat q(\boldsymbol{\nu}, t)

  \text{.}
$$

目標解 $q(\boldsymbol{x}, t)$ とその Fourier 変換 $\hat q(\boldsymbol{\nu}, t)$ は、[式 20](#equation-20)により次のように与えられる。
$$
\begin{aligned}
  q(\boldsymbol{x}, t) &= p\big( \boldsymbol{x}; \sigma(t) \big) = p_\text{data}(\boldsymbol{x}) \ast \mathcal{N}\big( \mathbf{0}, ~\sigma(t)^2 ~\mathbf{I}\big) \\
  \hat q(\boldsymbol{\nu}, t) &= \hat{p}_\text{data}(\boldsymbol{\nu}) ~\exp\Big( {-}\tfrac{1}{2} ~|\boldsymbol{\nu}|^2 ~\sigma(t)^2 \Big)
  \text{.}
\end{aligned}
$$

目標解を時間軸に沿って微分すると、次を得る。<span id="equation-87"></span>

$$
\begin{aligned}
  \frac{\partial \hat q(\boldsymbol{\nu}, t)}{\partial t} &= - \dot\sigma(t) \sigma(t) ~|\boldsymbol{\nu}|^2 ~ \hat{p}_\text{data}(\boldsymbol{\nu}) ~\exp\Big( {-}\tfrac{1}{2} ~|\boldsymbol{\nu}|^2 ~\sigma(t)^2 \Big) \\
  &= - \dot \sigma(t) \sigma(t) ~|\boldsymbol{\nu}|^2 ~\hat q(\boldsymbol{\nu},t)

  \text{.}
\end{aligned}
$$

[式 83](#equation-83)と[式 87](#equation-87)の左辺は同じである。両者を等置すると、所望の発展を生成する $\kappa(t)$ を解ける。
$$
\begin{aligned}
  - \kappa(t) |\boldsymbol{\nu}|^2 \hat q(\boldsymbol{\nu}, t) &= - \dot \sigma(t) \sigma(t) ~ |\boldsymbol{\nu}|^2 ~ \hat q(\boldsymbol{\nu},t) \\
  \kappa(t) &= \dot \sigma(t) \sigma(t)
  \text{.}
\end{aligned}
$$

まとめると、ノイズレベル $\sigma(t)$ に対応する所望の周辺密度は、次の PDE <span id="equation-90"></span>

$$
\frac{\partial q(\boldsymbol{x}, t)}{\partial t} = \dot \sigma(t) \sigma(t) {\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)
$$
によって、初期密度 $q(\boldsymbol{x}, 0) = p_\text{data}(\boldsymbol{x})$ から生成される。

<span id="section-8-5-2"></span>

#### 8.5.2 本研究の SDE の導出

SDE が次のように与えられるとする。<span id="equation-91"></span>

$$
\mathrm{d}\boldsymbol{x}= \boldsymbol{f}(\boldsymbol{x}, t) ~ \mathrm{d}t ~ + ~ \boldsymbol{g}(\boldsymbol{x}, t) ~ \mathrm{d}\omega_t

  \text{,}
$$
Fokker-Planck PDE は、その解の確率密度 $r(\boldsymbol{x}, t)$ の時間発展を次のように記述する。
$$
\frac{\partial r(\boldsymbol{x}, t)}{\partial t} = -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~r(\boldsymbol{x},t) \big) + \tfrac{1}{2} \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\nabla_{\hspace{-0.5mm}\boldsymbol{x}}: \big( \mathbf{D}(\boldsymbol{x}, t) ~r(\boldsymbol{x}, t) \big)
  \text{,}
$$
ここで、$\mathbf{D}_{ij} = \sum_k \boldsymbol{g}_{ik} \boldsymbol{g}_{jk}$ は*拡散テンソル*である。$\boldsymbol{g}(\boldsymbol{x}, t) = g(t) ~\mathbf{I}$、すなわち $\boldsymbol{x}$ に依存しない白色ノイズ付加という特殊な場合を考えると、方程式は次のように簡約される。<span id="equation-93"></span>

$$
\frac{\partial r(\boldsymbol{x}, t)}{\partial t} = -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~r(\boldsymbol{x},t) \big) + \tfrac{1}{2} ~g(t)^2 ~{\Delta_{\boldsymbol{x}}}r(\boldsymbol{x}, t)

  \text{.}
$$

解の密度が[式 90](#equation-90)の PDE で記述される SDE を求める。$r(\boldsymbol{x}, t) = q(\boldsymbol{x}, t)$ と置き、[式 93](#equation-93)と[式 90](#equation-90)を等置すると、SDE が満たすべき次の十分条件が得られる。
$$
\begin{aligned}
  -\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) \big) + \tfrac{1}{2} ~g(t)^2 ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) &= \dot\sigma(t) ~\sigma(t) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) \big) &= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t)
  \text{.}
\end{aligned}
$$

この方程式を満たす関数 $\boldsymbol{f}(\boldsymbol{x},t)$ と $g(t)$ の任意の選択が、求める SDE を構成する。ここで、そのような解の特定の族を求める。鍵となる考え方は、恒等式 $\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \nabla_{\hspace{-0.5mm}\boldsymbol{x}}= {\Delta_{\boldsymbol{x}}}$ によって与えられる。実際、$\boldsymbol{f}(\boldsymbol{x},t) ~q(\boldsymbol{x},t) = \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t)$ と置けば、$\upsilon(t)$ の選択を問わず、項 ${\Delta_{\boldsymbol{x}}}q(\boldsymbol{x},t)$ が両辺に現れて相殺される。
$$
\begin{aligned}
  \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\cdot \big( \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t) \big) &= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \upsilon(t) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x},t) &= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~{\Delta_{\boldsymbol{x}}}q(\boldsymbol{x}, t) \\
  \upsilon(t) &= \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t)
  \text{.}
\end{aligned}
$$

先に述べた $\boldsymbol{f}(\boldsymbol{x},t)$ は、実際には score 関数に比例する。これは、その式が密度の対数の勾配と一致するためである。
$$
\begin{aligned}
  \boldsymbol{f}(\boldsymbol{x}, t) &= \upsilon(t) ~\frac{\nabla_{\hspace{-0.5mm}\boldsymbol{x}}q(\boldsymbol{x},t)}{q(\boldsymbol{x},t)} \\
  &= \upsilon(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log q(\boldsymbol{x}, t) \\
  &= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log q(\boldsymbol{x}, t)
  \text{.}
\end{aligned}
$$

これを[式 91](#equation-91)に戻して代入し、$p(\boldsymbol{x}; \sigma(t))$ を $q(\boldsymbol{x},t)$ の代わりに書けば、解の密度がノイズレベル $\sigma(t)$ をもつ所望の周辺分布となる SDE の族が、$g(t)$ の任意の選択に対して得られる。
$$
\mathrm{d}\boldsymbol{x}= \Big( \tfrac{1}{2} ~g(t)^2 - \dot\sigma(t) ~\sigma(t) \Big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t ~+~ g(t) ~\mathrm{d}\omega_t
  \text{.}
$$

自由パラメータ $g(t)$ は、各時点におけるノイズ置換率を実質的に指定する。$g(t) = 0$ という特殊な選択は、確率流 ODE に対応する。しかし、$g(t)$ によるパラメータ化は特に直観的ではない。より解釈しやすいパラメータ化を得るため、$g(t) = \sqrt{2 ~\beta(t)} ~\sigma(t)$ と置くと、本文の[式 6](#equation-06)の（順方向）SDE が得られる。<span id="equation-103"></span>

$$
\mathrm{d}\boldsymbol{x}_{+} =
    -\dot\sigma(t) \sigma(t) \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t\, + \,
      \beta(t) \sigma(t)^2 \nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \,\mathrm{d}t+
      \sqrt{2 \beta(t)} \sigma(t) \,\mathrm{d}\omega_t

  \text{.}
$$

ここでノイズ置換は、ノイズの標準偏差 $\sigma(t)$ に比例し、その比例係数は $\beta(t)$ である。実際、[式 3](#equation-03)に従って中央の項の score 関数を展開すると、$\beta(t) ~\big[ D\big( \boldsymbol{x};\sigma(t) \big) - \boldsymbol{x}\big] ~\mathrm{d}t$ が得られ、これは負のノイズ成分に比例して $\boldsymbol{x}$ を変化させる。一方、確率項は同じ率で新たなノイズを注入する。直観的には、密度のぼかしによってデータ多様体が実質的にこの量だけ「広がる」ため、Langevin 探索の大きさを現在のノイズ標準偏差に応じてスケーリングすることは、妥当なベースラインである。

Denoising diffusion で用いる*逆方向* SDE は、[式 103](#equation-103)に Anderson [And82] の時間反転公式（Song ら [Son21] の式 6 に記載）を適用するだけで得られる。反転の効果は、中央の項の符号が変わることだけである。

SDE のスケーリングされた一般化は、先の ODE と同様の方法で導出できる。そのため、ここでは導出を省略する。

<span id="section-8-6"></span>

### 8.6 本研究の preconditioning と学習（[式 8](#equation-08)）

[式 2](#equation-02)に従うと、与えられたデノイザー $D_\theta$ とノイズレベル $\sigma$ に対する denoising score matching 損失は、次で与えられる。
$$
\mathcal{L}(D_\theta; \sigma) = \mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\| D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\|^2_2
  \text{.}
$$

$\mathcal{L}(D_\theta; \sigma)$ のノイズレベルにわたる重み付き期待値を取ることで、全体の学習損失を得る。<span id="equation-108"></span>

$$
\begin{aligned}
  \mathcal{L}(D_\theta) &= \mathbb{E}_{\sigma \sim p_\text{train}} \big[ \lambda(\sigma) ~\mathcal{L}(D_\theta; \sigma) \big] \\
  &= \mathbb{E}_{\sigma \sim p_\text{train}} ~\Big[ \lambda(\sigma) ~\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})} ~\big\| D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\|^2_2 \Big] \\
  &= \mathbb{E}_{\sigma \sim p_\text{train}} ~\mathbb{E}_{\boldsymbol{y}\sim p_\text{data}} ~\mathbb{E}_{\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})}  ~\Big[ \lambda(\sigma) ~\big\| D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\|^2_2 \Big] \\
  &= \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) ~\big\| D_\theta(\boldsymbol{y}+ \boldsymbol{n}; \sigma) - \boldsymbol{y}\big\|^2_2 \Big]

  \text{,}
\end{aligned}
$$
ここで、ノイズレベルは $\sigma \sim p_\text{train}$ に従って分布し、$\lambda(\sigma)$ によって重み付けされる。

[式 7](#equation-07)における本研究の $D_\theta(\cdot)$ の定義を用いると、$\mathcal{L}(D_\theta)$ はさらに次のように書き換えられる。<span id="equation-109"></span>

$$
\begin{aligned}
  && \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\| c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) + c_\text{out}(\sigma) F_\theta\big(c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)\big) - \boldsymbol{y}\big\|^2_2 \Big]  \\
  &= \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\| c_\text{out}(\sigma) F_\theta\big(c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)\big) - \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big) \big\|^2_2 \Big] \\
  &= \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) c_\text{out}(\sigma)^2 \big\| F_\theta\big(c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)\big) - \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) \big) \big\|^2_2 \Big] \\
  &= \mathbb{E}_{\sigma, \boldsymbol{y}, \boldsymbol{n}} \Big[ w(\sigma) ~\big\| F_\theta\big(c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)\big) - F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) \big\|^2_2 \Big]
  \text{,}
\end{aligned}
$$
これは[式 8](#equation-08)と一致し、$F_\theta$ を標準的な $L_2$ 損失で学習する従来型の教師あり学習に対応し、その実効重み $w(\cdot)$ およびターゲット $F_\text{target}(\cdot)$ は次で与えられる。
$$
w(\sigma) = \lambda(\sigma) ~c_\text{out}(\sigma)^2
  \hspace{4mm}\text{and}\hspace{4mm}
  F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) = \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big)
  \text{,}
$$

これで、[表 1](#table-01)の「Ours」列に示した $c_\text{in}(\sigma)$、$c_\text{out}(\sigma)$、$c_\text{skip}(\sigma)$、および $\lambda(\sigma)$ の式を第一原理から導出できる。

第一に、$F_\theta(\cdot)$ の学習入力が単位分散をもつことを要求する。
$$
\begin{aligned}
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ c_\text{in}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] &= 1 \\
  c_\text{in}(\sigma)^2 ~\mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}+ \boldsymbol{n}\big] &= 1 \\
  c_\text{in}(\sigma)^2 \big( \sigma_\text{data}^2 + \sigma^2 \big) &= 1 \\
  c_\text{in}(\sigma) &= 1 \big/ \sqrt{\sigma^2 + \sigma_\text{data}^2}
  \text{.}
\end{aligned}
$$

第二に、実効学習ターゲット $F_\text{target}$ が単位分散をもつことを要求する。<span id="equation-123"></span>

$$
\begin{aligned}
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ F_\text{target}(\boldsymbol{y}, \boldsymbol{n}; \sigma) \big] &= 1 \\
  \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{c_\text{out}(\sigma)} \big( \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big) \Big] &= 1 \\
  \tfrac{1}{c_\text{out}(\sigma)^2} \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] &= 1 \\
  c_\text{out}(\sigma)^2 &= \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \big[ \boldsymbol{y}- c_\text{skip}(\sigma) (\boldsymbol{y}+ \boldsymbol{n}) \big] \\
  c_\text{out}(\sigma)^2 &= \mathop{\mathrm{Var}}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \big( 1 - c_\text{skip}(\sigma) \big) ~\boldsymbol{y}+ c_\text{skip}(\sigma) ~\boldsymbol{n}\Big] \\
  c_\text{out}(\sigma)^2 &= \big( 1 - c_\text{skip}(\sigma) \big)^2 ~\sigma_\text{data}^2 + c_\text{skip}(\sigma)^2 ~\sigma^2

  \text{.}
\end{aligned}
$$

第三に、$c_\text{skip}(\sigma)$ は $c_\text{out}(\sigma)$ を最小化するように選択し、それによって $F_\theta$ の誤差ができるだけ増幅されないようにする。
$$
c_\text{skip}(\sigma) = \mathop{\mathrm{arg}\,\min}_{c_\text{skip}(\sigma)} c_\text{out}(\sigma)
  \text{.}
$$
$c_\text{out}(\sigma) \ge 0$ であるため、等価に次のように書ける。
$$
c_\text{skip}(\sigma) = \mathop{\mathrm{arg}\,\min}_{c_\text{skip}(\sigma)} c_\text{out}(\sigma)^2
  \text{.}
$$
これは凸最適化問題であり、$c_\text{skip}(\sigma)$ に関する導関数をゼロに置くことで、その解は一意に定まる。<span id="equation-131"></span>

$$
\begin{aligned}
  0 &= \mathrm{d}\big[ c_\text{out}(\sigma)^2 \big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &= \mathrm{d}\Big[ \big(1 - c_\text{skip}(\sigma)\big)^2 ~\sigma_\text{data}^2 + c_\text{skip}(\sigma)^2 ~\sigma^2 \Big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &= \sigma_\text{data}^2 ~\mathrm{d}\Big[ \big(1 - c_\text{skip}(\sigma)\big)^2 \Big] / \mathrm{d}c_\text{skip}(\sigma) + \sigma^2 ~\mathrm{d}\big[c_\text{skip}(\sigma)^2 \big] / \mathrm{d}c_\text{skip}(\sigma) \\
  0 &= \sigma_\text{data}^2 ~\big[ 2 ~c_\text{skip}(\sigma) - 2 \big] + \sigma^2 ~\big[ 2 ~c_\text{skip}(\sigma) \big] \\
  0 &= \big( \sigma^2 + \sigma_\text{data}^2 \big) ~c_\text{skip}(\sigma) - \sigma_\text{data}^2 \\
  c_\text{skip}(\sigma) &= \sigma_\text{data}^2 / \big( \sigma^2 + \sigma_\text{data}^2 \big)

  \text{.}
\end{aligned}
$$

ここで、[式 131](#equation-131)を[式 123](#equation-123)へ代入し、$c_\text{out}(\sigma)$ の式を完成させられる。
$$
\begin{aligned}
  c_\text{out}(\sigma)^2 &= \big( 1 - \big[ c_\text{skip}(\sigma) \big] \big)^2 ~\sigma_\text{data}^2 + \big[ c_\text{skip}(\sigma) \big]^2 ~\sigma^2 \\
  c_\text{out}(\sigma)^2 &= \bigg( 1 - \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} \bigg] \bigg)^2 ~\sigma_\text{data}^2 + \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 ~\sigma^2 \\
  c_\text{out}(\sigma)^2 &= \bigg[ \frac{\sigma^2 ~\sigma_\text{data}}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 + \bigg[ \frac{\sigma_\text{data}^2 ~\sigma}{\sigma^2 + \sigma_\text{data}^2} \bigg]^2 \\
  c_\text{out}(\sigma)^2 &= \frac{\big( \sigma^2 ~\sigma_\text{data}\big)^2 + \big( \sigma_\text{data}^2 ~\sigma \big)^2}{\big( \sigma^2 + \sigma_\text{data}^2 \big)^2} \\
  c_\text{out}(\sigma)^2 &= \frac{(\sigma \cdot \sigma_\text{data})^2 ~\big( \sigma^2 + \sigma_\text{data}^2 \big)}{\big( \sigma^2 + \sigma_\text{data}^2 \big)^2} \\
  c_\text{out}(\sigma)^2 &= \frac{(\sigma \cdot \sigma_\text{data})^2}{\sigma^2 + \sigma_\text{data}^2} \\
  c_\text{out}(\sigma) &= \sigma \cdot \sigma_\text{data}\big/ \sqrt{\sigma^2 + \sigma_\text{data}^2}
  \text{.}
\end{aligned}
$$

第四に、実効重み $w(\sigma)$ がノイズレベル間で一様になることを要求する。
$$
\begin{aligned}
  w(\sigma) &= 1 \\
  \lambda(\sigma) ~c_\text{out}(\sigma)^2 &= 1 \\
  \lambda(\sigma) &= 1 / c_\text{out}(\sigma)^2 \\
  \lambda(\sigma) &= 1 \big/ \bigg[ \frac{\sigma \cdot \sigma_\text{data}}{\sqrt{\sigma^2 + \sigma_\text{data}^2}} \bigg]^2 \\
  \lambda(\sigma) &= 1 \big/ \bigg[ \frac{(\sigma \cdot \sigma_\text{data})^2}{\sigma^2 + \sigma_\text{data}^2} \bigg] \\
  \lambda(\sigma) &= \big( \sigma^2 + \sigma_\text{data}^2 \big) / (\sigma \cdot \sigma_\text{data})^2
  \text{.}
\end{aligned}
$$

先行研究にならい、出力層の重みをゼロで初期化する。その結果、初期化時には $F_\theta(\cdot) = 0$ となり、各ノイズレベルにおける損失の期待値は $1$ となる。これは、$\lambda(\sigma)$ と $c_\text{skip}(\sigma)$ の選択を、固定した $\sigma$ について考えた[式 109](#equation-109)へ代入することで確認できる。
$$
\begin{aligned}
  && \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \lambda(\sigma) \big\| c_\text{skip}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}) + c_\text{out}(\sigma) F_\theta\big(c_\text{in}(\sigma) (\boldsymbol{y}{+} \boldsymbol{n}); c_\text{noise}(\sigma)\big) - \boldsymbol{y}\big\|^2_2 \Big] \\
  %
  &= \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma \cdot \sigma_\text{data})^2} \bigg\| \frac{\sigma_\text{data}^2}{\sigma^2 + \sigma_\text{data}^2} (\boldsymbol{y}{+} \boldsymbol{n}) - \boldsymbol{y}\bigg\|^2_2 \bigg] \\
  &= \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{\sigma^2 + \sigma_\text{data}^2}{(\sigma \cdot \sigma_\text{data})^2} \bigg\| \frac{\sigma_\text{data}^2 \boldsymbol{n}- \sigma^2 \boldsymbol{y}}{\sigma^2 + \sigma_\text{data}^2} \bigg\|^2_2 \bigg] \\
  &= \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[ \frac{1}{\sigma^2 + \sigma_\text{data}^2} \bigg\| \frac{\sigma_\text{data}}{\sigma} \boldsymbol{n}- \frac{\sigma}{\sigma_\text{data}} \boldsymbol{y}\bigg\|^2_2 \bigg] \\
  &= \frac{1}{\sigma^2 + \sigma_\text{data}^2} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \bigg[   \frac{\sigma_\text{data}^2}{\sigma^2} \langle \boldsymbol{n}, \boldsymbol{n}\rangle + \frac{\sigma^2}{\sigma_\text{data}^2} \langle \boldsymbol{y}, \boldsymbol{y}\rangle - 2 \langle \boldsymbol{y}, \boldsymbol{n}\rangle \bigg] \\
  &= \frac{1}{\sigma^2 + \sigma_\text{data}^2} \bigg[ \frac{\sigma_\text{data}^2}{\sigma^2} \underbrace{\mathop{\mathrm{Var}}(\boldsymbol{n})}_{= \sigma^2} + \frac{\sigma^2}{\sigma_\text{data}^2} \underbrace{\mathop{\mathrm{Var}}(\boldsymbol{y})}_{= \sigma_\text{data}^2} - 2 \underbrace{\mathop{\mathrm{Cov}}(\boldsymbol{y}, \boldsymbol{n})}_{=0} \bigg] \\
  &= 1
\end{aligned}
$$

<span id="section-9"></span>

## 9 先行手法の本研究の枠組みにおける再構成

本節では、先行手法について[表 1](#table-01)に示した式を導出し、対応する元のサンプラーと事前学習済みモデルを論じ、それらを本研究の枠組みで使用する際の実用上の考慮事項を詳述する。

実際には、これらの手法の元の実装は、モデルの入力と出力の定義、画像データのダイナミックレンジ、$\boldsymbol{x}$ のスケーリング、および $\sigma$ の解釈という点で大きく異なる。本研究では、モデルが常に本研究の $F_\theta$ の定義に一致し、画像データを常に連続範囲 $[-1, 1]$ で表現し、$\boldsymbol{x}$ と $\sigma$ の詳細が常に[式 4](#equation-04)と整合する統一設定に標準化することで、このばらつきを取り除く。

浮動小数点丸め誤差の累積を最小限に抑えるため、[アルゴリズム 1](#algorithm-01)と[アルゴリズム 2](#algorithm-02)は常に倍精度（`float64`）で実行する。ただし、実行時間を最小化し、ネットワークアーキテクチャの点で先行研究に忠実であるため、ネットワーク $F_\theta(\cdot)$ は引き続き単精度（`float32`）で実行する。

<span id="section-9-1"></span>

### 9.1 Variance preserving 定式化

<span id="section-9-1-1"></span>

#### 9.1.1 VP サンプリング

Song ら [Son21] は、VP SDE（[Son21] の式 32）を次のように定義する。
$$
\mathrm{d}\boldsymbol{x}= -\tfrac{1}{2} ~\Big( \beta_{\min}+ t ~\big( \beta_{\max}- \beta_{\min}\big) \Big) ~\boldsymbol{x}~\mathrm{d}t + \sqrt{ \beta_{\min}+ t ~\big( \beta_{\max}- \beta_{\min}\big) } ~\mathrm{d}\omega_t
  \text{,}
$$
これは、$f$ と $g$ を次のように選択した[式 10](#equation-10)と一致する。<span id="equation-153"></span>

$$
f(t) = -\tfrac{1}{2} ~\beta(t)
  \text{,}\hspace{4mm}
  g(t) = \sqrt{\beta(t)}
  \text{,}\hspace{4mm}\text{and}\hspace{4mm}
  \beta(t) = \big( \beta_{\max}- \beta_{\min}\big) ~t + \beta_{\min}
  \text{.}
$$

$\alpha(t)$ を $\beta(t)$ の積分とする。
$$
\begin{aligned}
  \alpha(t) &= \int_0^t \beta(\xi) ~\mathrm{d}\xi \\
  &= \int_0^t \Big[ \big( \beta_{\max}- \beta_{\min}\big) ~\xi + \beta_{\min}\Big] ~\mathrm{d}\xi \\
  &= \tfrac{1}{2} ~\big( \beta_{\max}- \beta_{\min}\big) ~t^2 + \beta_{\min}~t \\
  &= \tfrac{1}{2} ~\beta_\text{d}~t^2 + \beta_{\min}~t
  \text{,}
\end{aligned}
$$
ここで、$\beta_\text{d}= \beta_{\max}- \beta_{\min}$ である。[式 153](#equation-153)を[式 12](#equation-12)へ代入することで、$\sigma(t)$ の式を得られる。<span id="equation-163"></span>

$$
\begin{aligned}
  \sigma(t) &= \sqrt{\int_0^t \frac{\big[ g(\xi) \big]^2}{\big[ s(\xi) \big]^2} ~\mathrm{d}\xi} \\
  &= \sqrt{\int_0^t \frac{\big[ \sqrt{\beta(\xi)} \big]^2}{\big[ 1 / \sqrt{e^{\alpha(\xi)}} \big]^2} ~\mathrm{d}\xi} \\
  &= \sqrt{\int_0^t \frac{\beta(\xi)}{1 / e^{\alpha(\xi)}} ~\mathrm{d}\xi} \\
  &= \sqrt{\int_0^t \dot\alpha(\xi) ~e^{\alpha(\xi)} ~\mathrm{d}\xi} \\
  &= \sqrt{e^{\alpha(t)} - e^{\alpha(0)}} \\
  &= \sqrt{e^{\frac{1}{2} \beta_\text{d}t^2 + \beta_{\min}t} - 1}

  \text{,}
\end{aligned}
$$
これは[表 1](#table-01)の「Schedule」行と一致する。$s(t)$ についても同様に次を得る。<span id="equation-169"></span>

$$
\begin{aligned}
  s(t) &= \exp\left( \int_0^t \big[ f(\xi) \big] ~\mathrm{d}\xi \right) \\
  &= \exp\left( \int_0^t \big[ -\tfrac{1}{2} ~\beta(\xi) \big] ~\mathrm{d}\xi \right) \\
  &= \exp\left( -\tfrac{1}{2} \left[ \int_0^t \beta(\xi) ~\mathrm{d}\xi \right] \right) \\
  &= \exp\left( -\tfrac{1}{2} ~\alpha(t) \right) \\
  &= 1 / \sqrt{e^{\alpha(t)}} \\
  &= 1 / \sqrt{e^{\frac{1}{2} \beta_\text{d}t^2 + \beta_{\min}t}}

  \text{,}
\end{aligned}
$$
これは[表 1](#table-01)の「Scaling」行と一致する。[式 163](#equation-163)を用いると、[式 169](#equation-169)は等価に、やや簡単な次の形で書ける。<span id="equation-170"></span>

$$
s(t) = 1 / \sqrt{\sigma(t)^2 + 1}

  \text{.}
$$

Song ら [Son21] は、サンプリング時刻ステップ $\{t_0, \dots, t_{N-1}\}$ を $[\epsilon_\text{s}, 1]$ 内の等間隔に配置することを選択する。これは次の設定に対応する。
$$
t_{i<N} = 1 + \tfrac{i}{N-1}(\epsilon_\text{s} - 1)
  \text{,}
$$
これは[表 1](#table-01)の「Time steps」行と一致する。

最後に、Song ら [Son21] は $\beta_{\min}= 0.1$、$\beta_{\max}= 20$、$\epsilon_\text{s} = 10^{-3}$ と設定し（[Son21] の Appendix C）、画像を範囲 $[-1, 1]$ で表現することを選択する。これらの選択は本研究の定式化と容易に両立し、[表 1](#table-01)の「Parameters」セクションに反映されている。

<span id="section-9-1-2"></span>

#### 9.1.2 VP preconditioning

VP の場合、Song ら [Son21] は[式 13](#equation-13)の $p_t(\boldsymbol{x})$ の score を次のように近似する。[+1] <span id="equation-172"></span>

$$
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) ~\approx~ \underbrace{{-}\tfrac{1}{\bar\sigma(t)} ~F_\theta\big( \boldsymbol{x}; ~(M{-}1)t \big)}_{\mathop{\mathrm{score}}(\boldsymbol{x}; F_\theta, t)}

  \text{,}
$$
ここで、$M = 1000$、$F_\theta$ はネットワークを表し、$\bar\sigma(t)$ は[式 11](#equation-11)の摂動カーネルの標準偏差に対応する。

[式 20](#equation-20)と[式 11](#equation-11)から、それぞれ $p_t(\boldsymbol{x})$ と $\bar\sigma(t)$ の定義を展開し、$\boldsymbol{x}= s(t) \hat\boldsymbol{x}$ を代入して、スケーリングされていない変数 $\hat\boldsymbol{x}$ に関する対応する式を得る。
$$
\begin{aligned}
  \nabla_{\boldsymbol{x}} \log \big[ p\big( \boldsymbol{x}/ s(t); \sigma(t) \big) \big] &\approx& {-}\tfrac{1}{[s(t) \sigma(t)]} ~F_\theta\big( \boldsymbol{x}; ~(M{-}1)t \big) \\
  \nabla_{[s(t) \hat\boldsymbol{x}]} \log p\big( [s(t) ~\hat\boldsymbol{x}] / s(t); \sigma(t) \big) &\approx& {-}\tfrac{1}{s(t) \sigma(t)} ~F_\theta\big( [s(t) ~\hat\boldsymbol{x}]; ~(M{-}1)t \big) \\
  \tfrac{1}{s(t)} \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& {-}\tfrac{1}{s(t) \sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  \nabla_{\hat\boldsymbol{x}} \log p\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& {-}\tfrac{1}{\sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big)
  \text{.}
\end{aligned}
$$

ここで、左辺を[式 3](#equation-03)で置き換え、[式 170](#equation-170)から $s(t)$ の定義を展開する。
$$
\begin{aligned}
  \Big[ \Big( D\big( \hat\boldsymbol{x}; \sigma(t) \big) - \hat\boldsymbol{x}\Big) / \sigma(t)^2 \Big] &\approx& {-}\tfrac{1}{\sigma(t)} ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  D\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& \hat\boldsymbol{x}- \sigma(t) ~F_\theta\big( s(t) ~\hat\boldsymbol{x}; ~(M{-}1)t \big) \\
  D\big( \hat\boldsymbol{x}; \sigma(t) \big) &\approx& \hat\boldsymbol{x}- \sigma(t) ~F_\theta\bigg( \bigg[ \tfrac{1}{\sqrt{\sigma(t)^2 + 1}} \bigg] ~\hat\boldsymbol{x}; ~(M{-}1)t \bigg)
  \text{,}
\end{aligned}
$$
これは、$\sigma$ に関して、$\sigma(t) \rightarrow \sigma$ および $t \rightarrow \sigma^{-1}(\sigma)$ と置き換えることでさらに表せる。<span id="equation-180"></span>

$$
D(\hat\boldsymbol{x}; \sigma) ~\approx~ \hat\boldsymbol{x}- \sigma ~F_\theta\Big( \tfrac{1}{\sqrt{\sigma^2 + 1}} ~\hat\boldsymbol{x}; ~(M{-}1) ~\sigma^{-1}(\sigma) \Big)

  \text{.}
$$

[式 180](#equation-180)の右辺を $D_\theta$ の定義として採用し、次を得る。<span id="equation-181"></span>

$$
D_\theta(\hat\boldsymbol{x}; \sigma) = \underbrace{1~\cdot}_{c_\text{skip}}\hat\boldsymbol{x}~\underbrace{-~\sigma}_{c_\text{out}} \,\cdot ~F_\theta\Big( \underbrace{\tfrac{1}{\sqrt{\sigma^2 + 1}}}_{c_\text{in}} \,\cdot~\hat\boldsymbol{x}; ~\underbrace{(M{-}1)~\sigma^{-1}(\sigma)}_{c_\text{noise}} \Big)

  \text{,}
$$
ここで、$c_\text{skip}$、$c_\text{out}$、$c_\text{in}$、$c_\text{noise}$ は[表 1](#table-01)の「Network and preconditioning」セクションと一致する。

<span id="section-9-1-3"></span>

#### 9.1.3 VP 学習

Song ら [Son21] は、学習損失を次のように定義する。[+2]
$$
\mathbb{E}_{t \sim \mathcal{U}(\epsilon_\text{t}, 1), \boldsymbol{y}\sim p_\text{data}, \bar\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \mathbf{I})} \Big[ \big\| \bar\sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ \bar\sigma(t) ~\bar\boldsymbol{n}; ~F_\theta, t \big) + \bar\boldsymbol{n}\big\|^2_2 \Big]
  \text{,}
$$
ここで、$\mathop{\mathrm{score}}(\cdot)$ の定義は[式 172](#equation-172)と同じである。$\bar\sigma(t) = s(t) \sigma(t)$ および $\bar\boldsymbol{n}= \boldsymbol{n}/ \sigma(t)$ を代入して、この式を簡約する。ここで、$\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma(t)^2 \mathbf{I})$ である。<span id="equation-185"></span>

$$
\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \bar\boldsymbol{n}} \Big[ \big\| s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ [s(t)\sigma(t)] ~\bar\boldsymbol{n}; ~F_\theta, t \big) + \bar\boldsymbol{n}\big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{y}+ s(t)\sigma(t) ~[\boldsymbol{n}/ \sigma(t)]; ~F_\theta, t \big) + [\boldsymbol{n}/ \sigma(t)] \big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~(\boldsymbol{y}+ \boldsymbol{n}); ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\|^2_2 \Big]

  \text{.}
\end{aligned}
$$

[式 172](#equation-172)、[式 170](#equation-170)、および[式 74](#equation-74)を組み合わせることで、$\mathop{\mathrm{score}}(\cdot)$ を $D_\theta(\cdot)$ に関して表せる。
$$
\mathop{\mathrm{score}}\big( s(t) ~\boldsymbol{x}; F_\theta, t \big) ~=~ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \boldsymbol{x}; \sigma(t) \big) - \boldsymbol{x}\Big)
  \text{.}
$$

これを[式 185](#equation-185)に戻して代入すると、次を得る。
$$
\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| s(t) \sigma(t) ~\Big[ \tfrac{1}{s(t) \sigma(t)^2} \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) \Big] + \tfrac{1}{\sigma(t)} ~\boldsymbol{n}\big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| \tfrac{1}{\sigma(t)} \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) + \tfrac{1}{\sigma(t)} ~\boldsymbol{n}\big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{\sigma(t)^2} ~\big\| D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - \boldsymbol{y}\big\|^2_2 \Big]
  \text{.}
\end{aligned}
$$

これは $\sigma$ に関して、$\sigma(t) \rightarrow \sigma$ および $t \rightarrow \sigma^{-1}(\sigma)$ と置き換えることでさらに表せる。<span id="equation-190"></span>

$$
\underbrace{\mathbb{E}_{\sigma^{-1}(\sigma) \sim \mathcal{U}(\epsilon_\text{t}, 1)}}_{p_\text{train}} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \underbrace{\tfrac{1}{\sigma^2}}_{\lambda} \big\| D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma \big) - \boldsymbol{y}\big\|^2_2 \Big]

  \text{,}
$$
これは、[表 1](#table-01)の「Training」セクションに示した $p_\text{train}$ と $\lambda$ の選択を用いた[式 108](#equation-108)と一致する。

<span id="section-9-1-4"></span>

#### 9.1.4 VP の実用上の考慮事項

CIFAR-10 で用いる事前学習済み VP モデルは、Song ら [Son21] が提供する「DDPM++ cont. (VP)」チェックポイント [+3] に対応する。このモデルは合計 62 million 個の学習可能パラメータをもち、ノイズレベルの連続範囲 $\sigma \in \big[ \sigma(\epsilon_\text{t}), \sigma(1) \big] \approx [0.001, 152]$、すなわち本研究が選好するサンプリング範囲 $[0.002, 80]$ より広い範囲をサポートする。このモデルを $F_\theta(\cdot)$ として直接インポートし、[表 1](#table-01)の定義を用いて[アルゴリズム 1](#algorithm-01)と[アルゴリズム 2](#algorithm-02)を実行する。

[図 2a](#figure-02)における元のサンプラー（青）と本研究の再実装（オレンジ）の差は、Song ら [Son21] の実装上の見落としによって説明される。この点は Jolicoeur-Martineau ら [Jol21] も指摘している（[Jol21] の Appendix D）。第一に、元のサンプラーは Euler ステップで誤った乗数 [+4] を用いており、$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$ に乗じているのは $-1 / N$ であって、$(\epsilon_\text{s} - 1) / (N - 1)$ ではない。第二に、最後のステップで $t_{N-1} = \epsilon_\text{s}$ から $t_N = \epsilon_\text{s} - 1 / N$ へ進むため、オーバーシュートまたはアンダーシュートが生じる。ここで、$t_N < 0$ となるのは $N < 1000$ のときである。実際、これは生成画像に目立つノイズが含まれ、たとえば $N = 128$ ではきわめて深刻になることを意味する。本研究の定式化では、[アルゴリズム 1](#algorithm-01)のステップサイズが $\{t_i\}$ から一貫して計算され、$t_N = 0$ であるため、これらの問題を回避できる。

<span id="section-9-2"></span>

### 9.2 Variance exploding 定式化

<span id="section-9-2-1"></span>

#### 9.2.1 理論上の VE サンプリング

Song ら [Son21] は、VE SDE（[Son21] の式 30）を次のように定義する。
$$
\mathrm{d}\boldsymbol{x}= \sigma_{\min}\bigg( \frac{\sigma_{\max}}{\sigma_{\min}} \bigg)^t \sqrt{2 \log \frac{\sigma_{\max}}{\sigma_{\min}}} ~\mathrm{d}\omega_t
  \text{,}
$$
これは、次を用いた[式 10](#equation-10)と一致する。<span id="equation-192"></span>

$$
f(t) = 0
  \text{,}\hspace{4mm}
  g(t) = \sigma_{\min}\sqrt{2\log\sigma_\mathrm{d}} ~\sigma_\mathrm{d}^t
  \text{,}\hspace{4mm}\text{and}\hspace{4mm}
  \sigma_\mathrm{d}= \sigma_{\max}/ \sigma_{\min}
  \text{.}
$$

VE 定式化はスケーリングを用いない。これは[式 12](#equation-12)から容易に分かる。
$$
s(t) = \exp\left( \int_0^t \big[ f(\xi) \big] ~\mathrm{d}\xi \right) = \exp\left( \int_0^t \big[ 0 \big] ~\mathrm{d}\xi \right) = \exp(0) = 1
  \text{.}
$$

[式 192](#equation-192)を[式 12](#equation-12)へ代入すると、$\sigma(t)$ について次の形が示唆される。<span id="equation-199"></span>

$$
\begin{aligned}
  \sigma(t) &= \sqrt{\int_0^t \frac{\big[ g(\xi) \big]^2}{\big[ s(\xi) \big]^2} ~\mathrm{d}\xi} \\
  &= \sqrt{\int_0^t \frac{\big[ \sigma_{\min}\sqrt{2\log\sigma_\mathrm{d}} ~\sigma_\mathrm{d}^\xi \big]^2}{\big[ 1 \big]^2} ~\mathrm{d}\xi} \\
  &= \sqrt{\int_0^t \sigma_{\min}^2 ~\big[ 2\log\sigma_\mathrm{d}\big] ~\big[ \sigma_\mathrm{d}^{2\xi} \big] ~\mathrm{d}\xi} \\
  &= \sigma_{\min}\sqrt{\int_0^t \Big[ \log \big( \sigma_\mathrm{d}^2 \big) \Big] ~\Big[ \big( \sigma_\mathrm{d}^2 \big)^\xi \Big] ~\mathrm{d}\xi} \\
  &= \sigma_{\min}\sqrt{\big( \sigma_\mathrm{d}^2 \big)^t - \big( \sigma_\mathrm{d}^2 \big)^0} \\
  &= \sigma_{\min}\sqrt{\sigma_\mathrm{d}^{2t} - 1}

  \text{.}
\end{aligned}
$$

[式 199](#equation-199)は、Song らが報告した摂動カーネル（[Son21] の式 29）と整合する。しかし、これは彼らが意図した $\sigma(t) = \sigma_{\min}~\big( \tfrac{\sigma_{\max}}{\sigma_{\min}} \big)^t$ という定義（[Son21] の Appendix C）を満たさないことに注意する。

<span id="section-9-2-2"></span>

#### 9.2.2 実際の VE サンプリング

Song ら [Son21] の元の実装 [+5] は、discretized VE SDE [+8] の discretized reverse probability flow [+7] を積分するために reverse diffusion predictor [+6] を用いる。これらを組み合わせると、$\boldsymbol{x}_{i+1}$ について次の更新則が得られる。<span id="equation-200"></span>

$$
\boldsymbol{x}_{i+1} = \boldsymbol{x}_i + \tfrac{1}{2} ~\big( \bar\sigma_i^2 - \bar\sigma_{i+1}^2 \big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log \bar p_i (\boldsymbol{x})
  \text{,}
$$
ここで、
$$
\bar\sigma_{i<N} = \sigma_{\min}~\bigg( \frac{\sigma_{\max}}{\sigma_{\min}} \bigg)^{1 - i / (N-1)}
  \hspace{4mm}\text{and}\hspace{4mm}
  \bar\sigma_N = 0
  \text{.}
$$

興味深いことに、[式 200](#equation-200)は、次の選択を用いた本研究の ODE の Euler 反復と同一である。
$$
s(t) = 1
  \text{,}\hspace{4mm}
  \sigma(t) = \sqrt{t}
  \text{,}\hspace{4mm}\text{and}\hspace{4mm}
  t_i = \bar\sigma_i^2
  \text{.}
$$

これらの式は[表 1](#table-01)の「Sampling」セクションと一致し、その正しさは、これらを[アルゴリズム 1](#algorithm-01)の 5 行目へ代入することで検証できる。
$$
\begin{aligned}
  \boldsymbol{x}_{i+1} &= \boldsymbol{x}_i + (t_{i+1} - t_i) ~\boldsymbol{d}_i \\
  &= \boldsymbol{x}_i + (t_{i+1} - t_i) \bigg[ \bigg( \frac{\dot\sigma(t)}{\sigma(t)} + \frac{\dot s(t)}{s(t)} \bigg) \boldsymbol{x}- \frac{\dot\sigma(t) s(t)}{\sigma(t)} D \bigg( \frac{\boldsymbol{x}}{s(t)}; \sigma(t) \bigg) \bigg] \\
  &= \boldsymbol{x}_i + (t_{i+1} - t_i) \bigg[ \frac{\dot\sigma(t)}{\sigma(t)} ~\boldsymbol{x}- \frac{\dot\sigma(t)}{\sigma(t)} ~D \big( \boldsymbol{x}; \sigma(t) \big) \bigg] \\
  &= \boldsymbol{x}_i - (t_{i+1} - t_i) ~\dot\sigma(t) ~\sigma(t) \bigg[ \Big( D \big( \boldsymbol{x}; \sigma(t) \big) - \boldsymbol{x}\Big) \big/ \sigma(t)^2 \bigg] \\
  &= \boldsymbol{x}_i - (t_{i+1} - t_i) ~\dot\sigma(t) ~\sigma(t) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &= \boldsymbol{x}_i - (t_{i+1} - t_i) \Big[ \tfrac{1}{2\sqrt{t}} \Big] \Big[ \sqrt{t} \Big] ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &= \boldsymbol{x}_i + \tfrac{1}{2} ~(t_i - t_{i+1}) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) \\
  &= \boldsymbol{x}_i + \tfrac{1}{2} ~\big( \bar\sigma_i^2 - \bar\sigma_{i+1}^2 \big) ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big)
  \text{,}
\end{aligned}
$$
これは、$\bar p_i(\boldsymbol{x}) = p\big( \boldsymbol{x}; \sigma(t_i) \big)$ と選択することで[式 200](#equation-200)と同一になる。

最後に、Song ら [Son21] は CIFAR-10 に対して $\sigma_{\min}= 0.01$ と $\sigma_{\max}= 50$ を設定し（[Son21] の Appendix C）、先行する SMLD モデルに合わせて画像を範囲 $[0, 1]$ で表現することを選択する。本研究で標準化した範囲 $[-1, 1]$ は 2 倍広いため、補償のために $\sigma_{\min}$ と $\sigma_{\max}$ を 2$\times$ 倍しなければならない。[表 1](#table-01)の「Parameters」セクションは、これらの調整後の値を反映している。

<span id="section-9-2-3"></span>

#### 9.2.3 VE preconditioning

VE の場合、Song ら [Son21] は[式 13](#equation-13)の $p_t(\boldsymbol{x})$ の score を次のように直接近似する。[+9] <span id="equation-211"></span>

$$
\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p_t(\boldsymbol{x}) ~\approx~ \bar{F}_\theta\big( \boldsymbol{x}; \sigma(t) \big)

  \text{,}
$$
ここで、ネットワーク $\bar{F}_\theta$ は、追加の前処理 [+10] と後処理 [+11] [+12] のステップを含むよう設計されている。<span id="equation-212"></span>

$$
\bar{F}_\theta\big( \boldsymbol{x}; \sigma \big) ~=~ \tfrac{1}{\sigma} ~F_\theta\big( 2 \boldsymbol{x}{-} 1; \log(\sigma) \big)

  \text{.}
$$
一貫性のため、前処理と後処理はネットワーク自体に組み込むのではなく、$\{c_\text{skip}, c_\text{out}, c_\text{in}, c_\text{noise}\}$ を用いて処理する。

ただし、[式 211](#equation-211)と[式 212](#equation-212)は画像が範囲 $[0, 1]$ で表現されることを仮定しているため、本研究の枠組みでは直接使用できない。代わりに $[-1, 1]$ を使用するため、$p_t(\boldsymbol{x}) \rightarrow p_t(2 \boldsymbol{x}{-} 1)$、$\boldsymbol{x}\rightarrow \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2}$、および $\sigma \rightarrow \tfrac{1}{2} \sigma$ と置き換える。<span id="equation-215"></span>

$$
\begin{aligned}
  \nabla_{[\frac{1}{2} \boldsymbol{x}+ \frac{1}{2}]} \log p_t \big( 2 \big[ \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2} \big] {-} 1 \big) &\approx& \tfrac{1}{[\frac{1}{2} \sigma]} ~F_\theta\big( 2 \big[ \tfrac{1}{2} \boldsymbol{x}+ \tfrac{1}{2} \big] {-} 1; \log \big[ \tfrac{1}{2} \sigma \big] \big) \\
  2 ~\nabla_{\boldsymbol{x}} \log p_t(\boldsymbol{x}) &\approx& \tfrac{2}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log \big( \tfrac{1}{2} \sigma \big) \Big) \\
  \nabla_{\boldsymbol{x}} \log p(\boldsymbol{x}; \sigma) &\approx& \tfrac{1}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log \big( \tfrac{1}{2} \sigma \big) \Big)

  \text{.}
\end{aligned}
$$

ここで、[式 215](#equation-215)の左辺を[式 3](#equation-03)で置き換えることで、モデルを $D_\theta(\cdot)$ に関して表せる。<span id="equation-216"></span>
<span id="equation-217"></span>

$$
\begin{aligned}
  \Big( D_\theta \big( \boldsymbol{x}; \sigma \big) - \boldsymbol{x}\Big) / \sigma^2 &= \tfrac{1}{\sigma} ~F_\theta\Big( \boldsymbol{x}; \log\big( \tfrac{1}{2} \sigma \big) \Big)  \\
  D_\theta \big( \boldsymbol{x}; \sigma \big) &= \underbrace{1~\cdot}_{c_\text{skip}} \boldsymbol{x}+ \underbrace{\sigma~\cdot}_{c_\text{out}} F_\theta\Big(\underbrace{1~\cdot}_{c_\text{in}} \boldsymbol{x}; ~\underbrace{\log \big( \tfrac{1}{2} \sigma \big)}_{c_\text{noise}} \Big)

  \text{,}
\end{aligned}
$$
ここで、$c_\text{skip}$、$c_\text{out}$、$c_\text{in}$、$c_\text{noise}$ は[表 1](#table-01)の「Network and preconditioning」セクションと一致する。

<span id="section-9-2-4"></span>

#### 9.2.4 VE 学習

Song ら [Son21] は VP と VE に対する学習損失を同様に定義しているため、[式 216](#equation-216)から $\mathop{\mathrm{score}}(\cdot)$ の定義を借用し、[式 185](#equation-185)を再利用できる。<span id="equation-221"></span>

$$
\begin{aligned}
  && \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| s(t) \sigma(t) ~\mathop{\mathrm{score}}\big( s(t) ~(\boldsymbol{y}+ \boldsymbol{n}); ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| \sigma(t) ~\mathop{\mathrm{score}}\big( \boldsymbol{y}+ \boldsymbol{n}; ~F_\theta, t \big) + \boldsymbol{n}/ \sigma(t) \big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \big\| \sigma(t) ~\Big[ \Big( D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - (\boldsymbol{y}+ \boldsymbol{n}) \Big) / \sigma(t)^2 \Big] + \boldsymbol{n}/ \sigma(t) \big\|^2_2 \Big] \\
  &= \mathbb{E}_{t, \boldsymbol{y}, \boldsymbol{n}} \Big[ \tfrac{1}{\sigma(t)^2} ~\big\| D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma(t) \big) - \boldsymbol{y}\big\|^2_2 \Big]

  \text{.}
\end{aligned}
$$

VE 学習では、元の実装 [+13] は $\sigma(t) = \sigma_{\min}~\big( \tfrac{\sigma_{\max}}{\sigma_{\min}} \big)^t$ と定義する。したがって、[式 221](#equation-221)は次のように書き換えられる。
$$
\underbrace{\mathbb{E}_{\ln(\sigma) \sim \mathcal{U}( \ln(\sigma_{\min}), \ln(\sigma_{\max}))}}_{p_\text{train}} \mathbb{E}_{\boldsymbol{y}, \boldsymbol{n}} \Big[ \underbrace{\tfrac{1}{\sigma^2}}_{\lambda} \big\| D_\theta \big( \boldsymbol{y}+ \boldsymbol{n}; \sigma \big) - \boldsymbol{y}\big\|^2_2 \Big]
  \text{,}
$$
これは、[表 1](#table-01)の「Training」セクションに示した $p_\text{train}$ と $\lambda$ の選択を用いた[式 108](#equation-108)と一致する。

<span id="section-9-2-5"></span>

#### 9.2.5 VE の実用上の考慮事項

CIFAR-10 で用いる事前学習済み VE モデルは、Song ら [Son21] が提供する「NCSN++ cont. (VE)」チェックポイント [+14] に対応する。このモデルは合計 63 million 個の学習可能パラメータをもち、ノイズレベルの連続範囲 $\sigma \in \big[ \sigma(\epsilon_\text{t}), \sigma(1) \big] \approx [0.02, 100]$ をサポートする。これは本研究が選好するサンプリング範囲 $[0.002, 80]$ より狭いため、関連するすべての実験で $\sigma_{\min}= 0.02$ と設定する。この制約は config E における本研究の学習改善によって解消されるため、[表 2](#table-02)の config E および F では $\sigma_{\min}= 0.002$ の使用に戻すことに注意されたい。モデルをインポートする際は、[式 217](#equation-217)における $F_\theta(\cdot)$ の定義との一貫性を保つため、[式 212](#equation-212)に示した前処理と後処理のステップを取り除く。これらの変更により、[表 1](#table-01)の定義を用いて[アルゴリズム 1](#algorithm-01)と[アルゴリズム 2](#algorithm-02)を実行できる。

[図 2b](#figure-02)における元のサンプラー（青）と本研究の再実装（オレンジ）の差は、元の実装がステップ数の多い場合に被る浮動小数点丸め誤差によって説明される。[アルゴリズム 1](#algorithm-01)では $\boldsymbol{x}_i$ を倍精度で表現するため、これらの場合に本研究の結果の方が正確である。

<span id="section-9-3"></span>

### 9.3 Improved DDPM と DDIM

<span id="section-9-3-1"></span>

#### 9.3.1 DDIM の ODE 定式化

Song ら [Son21a] は、彼らの決定論的 DDIM サンプラーが次の ODE（[Son21a] の式 14）の Euler 積分として表せることを見いだした。<span id="equation-223"></span>

$$
\mathrm{d}\boldsymbol{x}(t) = \epsilon_\theta^{(t)} \left( \frac{\boldsymbol{x}(t)}{\sqrt{\sigma(t)^2 + 1}} \right) ~\mathrm{d}\sigma(t)
  \text{,}
$$
ここで、$\boldsymbol{x}(t)$ は彼らの離散更新式（[Son21a] の式 10）に現れる反復値をスケーリングしたものであり、$\epsilon_\theta$ は正規化されたノイズベクトルを予測するよう学習されたモデルである。すなわち、$\epsilon_\theta^{(t)}\big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) \approx \boldsymbol{n}(t) / \sigma(t)$ であり、$\boldsymbol{x}(t) = \boldsymbol{y}(t) + \boldsymbol{n}(t)$ である。本研究の定式化では、$D_\theta$ はクリーンな信号を近似するよう学習される。すなわち、$D_\theta\big( \boldsymbol{x}(t); \sigma(t) \big) \approx \boldsymbol{y}$ であるため、$\epsilon_\theta$ を $D_\theta$ に関して次のように再解釈できる。
$$
\begin{aligned}
  \boldsymbol{n}(t) &= \boldsymbol{x}(t) - \boldsymbol{y}(t) \\
  \big[ \boldsymbol{n}(t) / \sigma(t) \big] &= \big( \boldsymbol{x}(t) - \big[ \boldsymbol{y}(t) \big] \big) / \sigma(t) \\
  \epsilon_\theta^{(t)} \big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) &= \big( \boldsymbol{x}(t) - D_\theta \big( \boldsymbol{x}(t); \sigma(t) \big) \big) / \sigma(t)
  \text{.}
\end{aligned}
$$

$\epsilon(\cdot)$ と $D(\cdot)$ が $L_2$ の意味で理想的であると仮定すると、[式 3](#equation-03)を用いて上式をさらに簡約できる。<span id="equation-227"></span>
<span id="equation-229"></span>

$$
\begin{aligned}
  \epsilon^{(t)} \big( \boldsymbol{x}(t) / \sqrt{\sigma(t)^2 + 1} \big) &= \big( \boldsymbol{x}(t) - D \big( \boldsymbol{x}(t); \sigma(t) \big) \big) / \sigma(t)  \\
  &= -\sigma(t) ~\Big[ \Big( D \big( \boldsymbol{x}(t); \sigma(t) \big) - \boldsymbol{x}(t) \Big) / \sigma(t)^2 \Big] \\
  &= -\sigma(t) ~\nabla_{\boldsymbol{x}(t)} \log p\big( \boldsymbol{x}(t); \sigma(t) \big)

  \text{.}
\end{aligned}
$$

[式 229](#equation-229)を[式 223](#equation-223)へ戻して代入すると、次を得る。
$$
\mathrm{d}\boldsymbol{x}(t) = -\sigma(t) ~\nabla_{\boldsymbol{x}(t)} \log p\big( \boldsymbol{x}(t); \sigma(t) \big) ~\mathrm{d}\sigma(t)
  \text{,}
$$
これは $\sigma(t) = t$ と置くことで、さらに次のように簡約できる。
$$
\mathrm{d}\boldsymbol{x}= -t ~\nabla_{\hspace{-0.5mm}\boldsymbol{x}}\log p\big( \boldsymbol{x}; \sigma(t) \big) ~\mathrm{d}t
  \text{.}
$$
これは $s(t) = 1$、$\sigma(t) = t$ とした本研究の[式 4](#equation-04)と一致し、[表 1](#table-01)の「Sampling」セクションにも反映されている。

<span id="section-9-3-2"></span>

#### 9.3.2 iDDPM の時刻ステップ離散化

Ho ら [Den20] による元の DDPM 定式化では、順方向過程（[Den20] の式 2）を、$\bar\boldsymbol{x}_0 \sim p_\text{data}$ に対して離散分散スケジュール $\{\beta_1, \dots, \beta_T\}$ に従い Gaussian ノイズを徐々に加える Markov chain として定義する。
$$
q(\bar\boldsymbol{x}_t ~|~ \bar\boldsymbol{x}_{t-1}) = \mathcal{N}\big( \bar\boldsymbol{x}_t; ~\sqrt{1 - \beta_t} ~\bar\boldsymbol{x}_{t-1}, ~\beta_t ~\mathbf{I}\big)
  \text{.}
$$

$\bar\boldsymbol{x}_0$ から $\bar\boldsymbol{x}_t$ への対応する遷移確率（[Den20] の式 4）は、次で与えられる。<span id="equation-233"></span>

$$
q(\bar\boldsymbol{x}_t ~|~ \bar\boldsymbol{x}_0) = \mathcal{N}\big( \bar\boldsymbol{x}_t; ~\sqrt{\bar\alpha_t} ~\bar\boldsymbol{x}_0, ~(1 - \bar\alpha_t) ~\mathbf{I}\big)
  \text{,}\hspace{4mm}\text{where}\hspace{4mm}
  \bar\alpha_t = \prod_{s=1}^t ~(1 - \beta_s)
  \text{.}
$$

Ho ら [Den20] は線形スケジュールに基づいて $\{\beta_t\}$ を定義し、[式 233](#equation-233)から対応する $\{\bar\alpha_t\}$ を計算する。別の方法として、$\{\bar\alpha_t\}$ を先に定義し、その後 $\{\beta_t\}$ について解くこともできる。<span id="equation-234"></span>

$$
\begin{aligned}

  \bar\alpha_t &= \prod_{s=1}^t ~(1 - \beta_s) \\
  \bar\alpha_t &= \bar\alpha_{t-1} ~(1 - \beta_t) \\
  \beta_t &= 1 - \frac{\bar\alpha_t}{\bar\alpha_{t-1}}
  \text{.}
\end{aligned}
$$

Nichol と Dhariwal [Nic21] による improved DDPM 定式化では、$\bar\alpha_t$ に cosine schedule（[Nic21] の式 17）を用い、次のように定義する。
$$
\bar\alpha_t = \frac{f(t)}{f(0)}
  \text{,}\hspace{4mm}\text{where}\hspace{4mm}
  f(t) = \cos^2 \bigg( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \bigg)
  \text{,}
$$
ここで、$s = 0.008$ である。しかし、その実装 [+15] では、Nichol らは $f(0)$ による除算を省き、単純に次のように定義する。[+16] <span id="equation-238"></span>

$$
\bar\alpha_t = \cos^2 \bigg( \frac{t/T + s}{1 + s} \cdot \frac{\pi}{2} \bigg)
  \text{.}
$$

$t = T$ 付近での特異点を防ぐため、さらに $\beta_t$ を $0.999$ にクランプする。[式 233](#equation-233)と[式 234](#equation-234)を用いることで、このクランプを $\bar\alpha_t$ に関して表せる。<span id="equation-239"></span>

$$
\begin{aligned}

  \bar\alpha'_t &= \prod_{s=1}^t ~\big( 1 - [\beta'_s] \big) \\
  &= \prod_{s=1}^t ~\Big( 1 - \min\big( [\beta_s], ~0.999) \Big) \\
  &= \prod_{s=1}^t ~\bigg( 1 - \min\bigg( 1 - \frac{\bar\alpha_s}{\bar\alpha_{s-1}}, ~0.999 \bigg) \bigg) \\
  &= \prod_{s=1}^t ~\max\bigg( \frac{\bar\alpha_s}{\bar\alpha_{s-1}}, ~0.001 \bigg)
  \text{.}
\end{aligned}
$$

ここで、上の式を本研究の統一された枠組みで再解釈する。[表 1](#table-01)で、元の iDDPM のサンプリングステップを $\{u_j\}$ と表し、ノイズレベル $\sigma(u_j)$ の降順に並べたことを思い出されたい。ここで、$j \in \{0, \dots, M\}$ である。したがって、[式 233](#equation-233)、[式 238](#equation-238)、および[式 239](#equation-239)の記法を統一するには、$T \longrightarrow M$ および $t \longrightarrow M-j$ と置き換えなければならない。<span id="equation-243"></span>
<span id="equation-244"></span>
<span id="equation-245"></span>

$$
\begin{aligned}
  q(\bar\boldsymbol{x}_j ~|~ \bar\boldsymbol{x}_M) &= \mathcal{N}\big( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\bar\boldsymbol{x}_M, ~(1 - \bar\alpha'_j) ~\mathbf{I}\big)  \text{,} \\[2mm]
  \bar\alpha_j &= \cos^2 \bigg( \frac{(M - j) / M + C_2}{1 + C_2} \cdot \frac{\pi}{2} \bigg)  \text{,}\hspace{4mm}\text{and} \\
  \bar\alpha'_j &= \prod_{s=M-1}^j ~\max\bigg( \frac{\bar\alpha_j}{\bar\alpha_{j+1}}, ~C_1 \bigg) ~=~ \bar\alpha'_{j+1} ~\max\bigg( \frac{\bar\alpha_j}{\bar\alpha_{j+1}}, ~C_1 \bigg)
  \text{,}
\end{aligned}
$$
ここで、定数は $C_1 = 0.001$ および $C_2 = 0.008$ である。

[式 244](#equation-244)はさらに簡約できる。
$$
\begin{aligned}
  \bar\alpha_j &= \cos^2 \bigg( \frac{(M - j) / M + C_2}{1 + C_2} \cdot \frac{\pi}{2} \bigg) \\
  &= \cos^2 \bigg( \frac{\pi}{2} ~\frac{(1 + C_2) - j / M}{1 + C_2} \bigg)\\
  &= \cos^2 \bigg( \frac{\pi}{2} - \frac{\pi}{2} ~\frac{j}{M (1 + C_2)} \bigg)\\
  &= \sin^2 \bigg( \frac{\pi}{2} ~\frac{j}{M (1 + C_2)} \bigg)
  \text{,}
\end{aligned}
$$
これにより、[表 1](#table-01)の「Parameters」セクションに示した式が得られる。

$\boldsymbol{x}$ と $\bar\boldsymbol{x}$ の定義を統一するには、各時刻ステップ $t = u_j$ において、[式 11](#equation-11)の摂動カーネルと[式 243](#equation-243)の遷移確率を一致させなければならない。
$$
\begin{aligned}
  p_{0t}\big( \boldsymbol{x}(u_j) ~|~ \boldsymbol{x}(0) \big) &= q(\bar\boldsymbol{x}_j ~|~ \bar\boldsymbol{x}_M) \\
  \mathcal{N} \big( \boldsymbol{x}(u_j); ~s(t) ~\boldsymbol{x}(0), ~s(u_j)^2 ~\sigma(u_j)^2 ~\mathbf{I}\big) &= \mathcal{N}\left( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\bar\boldsymbol{x}_M, ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right)
  \text{.}
\end{aligned}
$$

[第 9.3.1 節](#section-9-3-1)の $s(t) = 1$ と $\sigma(t) = t$、および $\bar\boldsymbol{x}_M = \boldsymbol{x}(0)$ を代入する。
$$
\mathcal{N} \big( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~u_j^2 ~\mathbf{I}\big) = \mathcal{N}\left( \bar\boldsymbol{x}_j; ~\sqrt{\bar\alpha'_j} ~\boldsymbol{x}(0), ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right)
  \text{.}
$$

$\bar\boldsymbol{x}_j = \sqrt{\bar\alpha'_j} ~\boldsymbol{x}(u_j)$ と定義することで、これら 2 つの分布の平均を一致させられる。
$$
\begin{aligned}
  \mathcal{N} \big( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~u_j^2 ~\mathbf{I}\big) &= \mathcal{N}\left( \sqrt{\bar\alpha'_j} ~\boldsymbol{x}(u_j); ~\sqrt{\bar\alpha'_j} ~\boldsymbol{x}(0), ~\big( 1 - \bar\alpha'_j \big) ~\mathbf{I}\right) \\
  &= \mathcal{N}\bigg( \boldsymbol{x}(u_j); ~\boldsymbol{x}(0), ~\frac{1 - \bar\alpha'_j}{\bar\alpha'_j} ~\mathbf{I}\bigg)
  \text{.}
\end{aligned}
$$

分散を一致させて $\bar\alpha'_j$ について解くと、次を得る。
$$
\begin{aligned}
  u_j^2 &= (1 - \bar\alpha'_j) ~/~ \bar\alpha'_j \\
  u_j^2 ~\bar\alpha'_j &= 1 - \bar\alpha'_j \\
  u_j^2 ~\bar\alpha'_j + \bar\alpha'_j &= 1 \\
  (u_j^2 + 1) ~\bar\alpha'_j &= 1 \\
  \bar\alpha'_j &= 1 ~/~ (u_j^2 + 1)
  \text{.}
\end{aligned}
$$

最後に、[式 245](#equation-245)を用いて左辺を展開し、$u_{j-1}$ について解く。
$$
\begin{aligned}
  \bar\alpha'_{j+1} ~\max(\bar\alpha_j / \bar\alpha_{j+1}, ~C_1) &= 1 ~/~ (u_j^2 + 1) \\
  \bar\alpha'_j ~\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) &= 1 ~/~ (u_{j-1}^2 + 1) \\
  \big[ 1 ~/~ (u_j^2 + 1) \big] ~\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) &= 1 ~/~ (u_{j-1}^2 + 1) \\
  \max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) ~(u_{j-1}^2 + 1) &= u_j^2 + 1 \\
  u_{j-1}^2 + 1 &= (u_j^2 + 1) ~/~ \max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1) \\
  u_{j-1} &= \sqrt{\frac{u_j^2 + 1}{\max(\bar\alpha_{j-1} / \bar\alpha_j, ~C_1)} - 1}
  \text{,}
\end{aligned}
$$
これにより、$\{u_j\}$ の漸化式が得られ、これは $u_M = 0$ から開始し、[表 1](#table-01)の「Time steps」行と一致する。

<span id="section-9-3-3"></span>

#### 9.3.3 iDDPM の preconditioning と学習

[式 227](#equation-227)から $D_\theta(\cdot)$ を解くため、[第 9.3.1 節](#section-9-3-1)の $\sigma(t) = t$ を代入する。
$$
\begin{aligned}
  \epsilon_\theta^{(j)} \left( \boldsymbol{x}/ \sqrt{\sigma^2 + 1} \right) &= \big( \boldsymbol{x}- D_\theta(\boldsymbol{x}; \sigma) \big) / \sigma \\
  D_\theta(\boldsymbol{x}; \sigma) &= \boldsymbol{x}- \sigma ~\epsilon_\theta^{(j)} \left( \boldsymbol{x}/ \sqrt{\sigma^2 + 1} \right)
  \text{.}
\end{aligned}
$$

$F_\theta(\cdot; j) = \epsilon_\theta^{(j)}(\cdot)$ と定義し、$j$ は $\sigma$ に最も近い $u_j$ を見つけることで解くことを選択する。<span id="equation-268"></span>

$$
D_\theta(\boldsymbol{x}; \sigma) = \underbrace{1~\cdot}_{c_\text{skip}} \boldsymbol{x}~\underbrace{-~\sigma}_{c_\text{out}} \,\cdot ~F_\theta\Big( \underbrace{\tfrac{1}{\sqrt{\sigma^2 + 1}}}_{c_\text{in}} \,\cdot~\boldsymbol{x}; ~\underbrace{\mathop{\mathrm{arg}\,\min}_j |u_j - \sigma|}_{c_\text{noise}} \Big)

  \text{,}
$$
ここで、$c_\text{skip}$、$c_\text{out}$、$c_\text{in}$、$c_\text{noise}$ は[表 1](#table-01)の「Network and preconditioning」セクションと一致する。

[式 268](#equation-268)は[式 181](#equation-181)の VP preconditioning の式と同一であることに注意されたい。さらに、Nichol と Dhariwal [Nic21] は、主たる学習損失 $L_\text{simple}$（[Nic21] の式 14）を Song ら [Son21] と同じ方法で定義し、$\sigma$ は $\{u_j\}$ から一様に抽出される。したがって、$\sigma = u_j$、$j \sim \mathcal{U}(0, M-1)$、$\lambda(\sigma) = 1 / \sigma^2$ として[式 190](#equation-190)を再利用でき、これは[表 1](#table-01)の「Training」セクションと一致する。$L_\text{simple}$ に加えて、Nichol と Dhariwal [Nic21] は副次的な損失項 $L_\text{vlb}$ も用いる。詳細については、[Nic21] の対応するセクション（3.1）を参照されたい。

<span id="section-9-3-4"></span>

#### 9.3.4 iDDPM の実用上の考慮事項

ImageNet-64 で用いる事前学習済み iDDPM モデルは、Dhariwal と Nichol [Dha21] が提供する「ADM (dropout)」チェックポイント [+17] に対応する。このモデルは 296 million 個の学習可能パラメータをもち、$M = 1000$ 個の離散的なノイズレベル集合 $\sigma \in \{u_j\} \approx \{$20291, 642, 321, 214, 160, 128, 106, 92, 80, 71, $\dots$, 0.0064$\}$ をサポートする。$F_\theta$ をこのような特定の $\sigma$ の選択肢についてしか評価できないことから、実用上の課題が 3 つ生じる。

1.  DDIM の文脈では、$\{u_j\}$ をどのように再サンプリングして $\{t_i\}$ を得るかを、$N \ne M$ の場合について選ばなければならない。Song ら [Son21a] は、$t_i = u_{k \cdot i}$ とする単純な再サンプリング方式を用いる。ここで、再サンプリング係数は $k \in \mathbb{Z}^+$ である。しかし、この方式は $1000 \equiv 0 \pmod{N}$ を必要とするため、$N$ の可能な選択肢を大幅に制限する。一方、Nichol と Dhariwal [Nic21] は、$t_i = u_j$ とし、$j = \lfloor (M - 1) / (N - 1) \cdot i \rfloor$ とする、より柔軟な方式を用いる。しかし実際には、$u_{j<8}$ の値が、本研究が選好する $\sigma_{\max}= 80$ よりかなり大きいことに注意する。本研究では、$j = \lfloor j_0 + (M - 1 - j_0) / (N - 1) \cdot i \rfloor$ と定義し、$j_0 = 8$ とすることでこれらの値をスキップすることを選択し、これは[表 1](#table-01)の「Time steps」行と一致する。[図 2c](#figure-02)における元のサンプラー（青）と本研究の再実装（オレンジ）の差は、この選択によって説明される。

2.  本研究の時刻ステップ離散化（[式 5](#equation-05)）の文脈では、$\sigma_i \in \{u_j\}$ であることを保証しなければならない。各 $\sigma_i$ を最も近いサポート対象の値へ丸める、すなわち $\sigma_i \gets u_{\mathop{\mathrm{arg}\,\min}_j |u_j - \sigma_i|}$ とし、$\sigma_{\min}= 0.0064 ~\approx~ u_{N-1}$ と設定することでこれを実現する。[アルゴリズム 1](#algorithm-01)が $D_\theta(\cdot; \sigma)$ を $\sigma \in \{\sigma_{i<N}\}$ の場合にのみ評価するため、これで十分である。

3.  本研究の確率的サンプラーの文脈では、$\hat t_i \in \{u_j\}$ であることを保証しなければならない。[アルゴリズム 2](#algorithm-02)の 5 行目を $\hat t_i \gets u_{\mathop{\mathrm{arg}\,\min}_j |u_j - (t_i + \gamma_i t_i)|}$ に置き換えることで、これを実現する。

これらの変更により、事前学習済みモデルを $F_\theta(\cdot)$ として直接インポートし、[表 1](#table-01)の定義を用いて[アルゴリズム 1](#algorithm-01)と[アルゴリズム 2](#algorithm-02)を実行できる。モデルは、[Nic21] の対応するセクション（3.1）で述べられているように、$\epsilon_\theta(\cdot)$ と $\Sigma_\theta(\cdot)$ の両方を出力することに注意されたい。本研究では前者のみを用い、後者は無視する。

<span id="section-10"></span>

## 10 決定論的サンプリングの追加分析

<span id="section-10-1"></span>

### 10.1 打ち切り誤差の分析と離散化パラメータの選択

[第 3 節](#section-3)で論じたように、拡散モデルが多数のサンプリングステップを必要とする傾向がある根本的な理由は、どのような数値 ODE ソルバーも必然的に近似であることにある。ステップが大きいほど、各ステップで真の解からより遠く逸脱する。具体的には、与えられた $\boldsymbol{x}_{i-1}$ の値が時間ステップ $i-1$ におけるものであるとき、ソルバーは真の $\boldsymbol{x}^*_i$ を $\boldsymbol{x}_i$ として近似し、その結果、局所打ち切り誤差 $\boldsymbol{\tau}_i = \boldsymbol{x}^*_i - \boldsymbol{x}_i$ が生じる。局所誤差は $N$ ステップにわたって蓄積し、最終的に大域打ち切り誤差 $\boldsymbol{e}_N$ につながる。

Euler 法は1次の ODE ソルバーであり、$\boldsymbol{\tau}_i = \mathcal{O}\left(h_i^2\right)$ が任意の十分に滑らかな $\boldsymbol{x}(t)$ に対して成り立つことを意味する。ここで、$h_i = |t_i - t_{i-1}|$ は局所ステップ幅である [Sul03]。言い換えると、ある $C$ と $H$ が存在し、$\|\boldsymbol{\tau}_i\| < C h_i^2$ がすべての $h_i < H$ に対して成り立つ。すなわち、$h_i$ を半分にすると $\boldsymbol{\tau}_i$ は 4$\times$ 小さくなる。さらに、$D_\theta$ が Lipschitz 連続であると仮定すると—これは本論文で検討するすべてのネットワークアーキテクチャに当てはまる—大域打ち切り誤差は $\|\boldsymbol{e}_N\| \le E \max_i \|\boldsymbol{\tau}_i\|$ で上から抑えられる。ここで、$E$ の値は $N$、$t_0$、$t_N$、および Lipschitz 定数に依存する [Sul03]。したがって、与えられた $N$ に対する大域誤差を減らし、ひいては $N$ 自体を減らせるようにすることは、ソルバーと $\{t_i\}$ を、$\max_i \|\boldsymbol{\tau}_i\|$ が最小になるように選ぶことに帰着する。

<span id="figure-13"></span>

![(a) VE ベースの CIFAR-10 モデルで Euler 法を用いた場合の局所打ち切り誤差（$y$ 軸）と、異なるノイズレベル（$x$ 軸）。](../../papers/diffusion-design-space/figure-13.png)

**図 13。** **(a)** VE ベースの CIFAR-10 モデルで Euler 法を用いた場合の局所打ち切り誤差（$y$ 軸）と、異なるノイズレベル（$x$ 軸）。各曲線は、$N=64$ と多項式指数 $\rho$ の特定の選択によって定義される、異なる時間ステップ離散化に対応する。値は、1回の Euler 反復と、グラウンドトゥルースを表す複数のより小さな Euler 反復の系列との間の二乗平均平方根誤差（RMSE）を表す。低い $\sigma$ ではほとんど見えない網掛け領域は、異なる潜在変数 $\boldsymbol{x}_0$ にわたる標準偏差を表す。**(b)** Heun の2次法（[アルゴリズム 1](#algorithm-01)）に対応する誤差曲線。**(c)** 異なるモデルについて Heun の2次法で測定した FID（$y$ 軸）を、多項式指数（$x$ 軸）の関数として示す。網掛け領域は観測された FID の最小値と最大値の間の変動範囲を示し、点は他のすべての実験で使用する $\rho$ の値を示す。

局所打ち切り誤差が実際にどのように振る舞うかについて知見を得るため、VE ベースの CIFAR-10 モデルを用いて、異なるノイズレベルにわたる $\boldsymbol{\tau}_i$ の値を測定する。与えられたノイズレベルに対して、$t_i = \sigma^{-1}(\sigma_i)$ と設定し、場合に応じて何らかの $t_{i-1} > t_i$ を選ぶ。次に、$\boldsymbol{x}_{i-1}$ を $p(\boldsymbol{x}; \sigma_{i-1})$ からサンプリングし、真の $\boldsymbol{x}^*_i$ を、$t_{i-1}$ と $t$ の間で一様に選んだ部分区間にわたる200回の Euler ステップによって推定する。最後に、二乗平均平方根誤差（RMSE）、すなわち $\|\boldsymbol{\tau}_i\| / \scriptstyle\sqrt{\dim\boldsymbol{\tau}}$ の平均と標準偏差を、$\sigma_i$ の関数として、$\boldsymbol{x}_{i-1}$ の200個のランダムサンプルにわたり平均してプロットする。Euler 法の結果を[図 13a](#figure-13)に示す。青い曲線は一様なステップ幅 $h_\sigma = 1.25$ に対応し、これは $\sigma$ に関するもの、すなわち $\sigma_{i-1} = \sigma_i + h_\sigma$ および $t_{i-1} = \sigma^{-1}(\sigma_{i-1})$ である。低いノイズレベルでは誤差が非常に大きく（$\text{RMSE} \approx 0.56$、$\sigma_i \le 0.5$）、高いノイズレベルではかなり小さいことが分かる。これは、$\boldsymbol{e}_N$ を減らすには、$\sigma$ の低下に伴ってステップ幅を単調に小さくすべきだという一般的な直観と一致する。各曲線は標準偏差を示す網掛け領域で囲まれているが、$\sigma$ の値が低い場合にはほとんど見えない。これは、$\boldsymbol{\tau}_i$ が $\boldsymbol{x}_{i-1}$ に関してほぼ一定であり、したがってサンプルごとに $\{t_i\}$ のスケジュールを変えても利点がないことを示している。

ノイズレベルに応じて局所ステップ幅を変える便利な方法は、$\{\sigma_i\}$ を、単調増加する非有界なワープ関数 $w(z)$ の線形リサンプリングとして定義することである。言い換えると、$\sigma_{i<N} = w(A i + B)$ および $\sigma_N = 0$ とし、定数 $A$ と $B$ を、$\sigma_0 = \sigma_{\max}$ かつ $\sigma_{N-1} = \sigma_{\min}$ となるように選ぶ。実際には、$\sigma_{\min}= \max(\sigma_\text{lo}, 0.002)$ および $\sigma_{\max}= \min(\sigma_\text{hi}, 80)$ と設定する。ここで、$\sigma_\text{lo}$ と $\sigma_\text{hi}$ はそれぞれ、所与のモデルがサポートする最低および最高のノイズレベルであり、これらの選択が実際にかなり良好に機能することを確認した。ここで、低いノイズレベルと高いノイズレベルの間で $\boldsymbol{\tau}_i$ の釣り合いを取るために、例えば多項式ワープ関数 $w(z) = z^\rho$ を使用し、指数 $\rho$ でパラメータ化できる。この選択により、$\{\sigma_i\}$ について次の式が得られる。<span id="equation-269"></span>

$$
\sigma_{i<N} = \left( {\sigma_{\max}}^\frac{1}{\rho} + \frac{i}{N-1} \left( {\sigma_{\min}}^\frac{1}{\rho} - {\sigma_{\max}}^\frac{1}{\rho} \right) \right)^\rho, \sigma_N = 0,
$$
これは $\rho=1$ のとき一様離散化に帰着し、$\rho$ が増加するにつれて低いノイズレベルをますます重視する。[+18]

$\sigma_i$ の値に基づき、今や $\sigma_{i-1} = \big( \sigma_i^{1 / \rho} - A \big)^\rho$ を計算でき、これにより $\boldsymbol{\tau}_i$ を $\rho$ の異なる選択に対して[図 13a](#figure-13)で可視化できる。$\rho$ を増加させると、低いノイズレベル（$\sigma < 10$）では誤差が減少する一方、高いノイズレベル（$\sigma > 10$）では増加することが分かる。$\rho=2$ でおおよその均衡が得られるが、RMSE は依然として比較的高く（$\sim0.03$）、Euler 法が各ステップで正しい結果から数 ULP ずつ逸脱することを意味する。$N$ を増やせば誤差を減らせるものの、理想的にはステップ数が少ない場合でも RMSE を 0.01 より十分小さくしたい。

Heun 法は、$\boldsymbol{x}_{i+1}$ に追加の補正ステップを導入し、$\mathrm{d}\boldsymbol{x}/ \mathrm{d}t$ が $t_i$ と $t_{i+1}$ の間で変化し得るという事実を考慮する。Euler 法はこれを一定と仮定する。この補正により、局所打ち切り誤差は3次収束、すなわち $\boldsymbol{\tau}_i = \mathcal{O}\left(h_i^3\right)$ となるが、その代わりステップごとに $D_\theta$ の評価が1回追加される。Heun 法に類似する方式の一般族については、後の[第 10.2 節](#section-10-2)で論じる。[図 13b](#figure-13)は、[図 13a](#figure-13)と同じ設定を用いた Heun 法の局所打ち切り誤差を示す。$\|\boldsymbol{\tau}_i\|$ の差は全般により顕著であり、2次収束と3次収束の違いを考えれば予想どおりである。Euler 法で RMSE が低い場合は Heun 法ではさらに低くなる傾向があり、RMSE が高い場合にはその逆となる。最も注目すべきことに、赤い曲線はほぼ一定の $\text{RMSE} \in [0.0030, 0.0045]$ を示す。これは、[式 269](#equation-269)と Heun 法の組み合わせが、実際に $\rho=3$ でほぼ最適であることを意味する。

ここまでは、生の数値誤差、すなわち RGB 空間における真の結果からの成分ごとの偏差のみを検討してきた。生の数値誤差は、例えば ODE を最初に $t$ が増加する方向へ評価し、その後再び $t=0$ まで戻す画像操作など、特定のユースケースに関係する。この場合、$\|\boldsymbol{e}_N\|$ は処理中に元の画像がどの程度劣化するかを直接表し、$\rho=3$ を用いてそれを最小化できる。しかし、新しい画像をゼロから生成する場合を考えると、ノイズレベルが異なれば異なる種類の誤差が生じ、それらは知覚的重要性の点で必ずしも同等ではないと考えるのが妥当である。これを[図 13c](#figure-13)で調べ、$\rho$ の関数として FID を、異なるモデルと $N$ の異なる選択についてプロットする。ImageNet-64 モデルは離散的なノイズレベルの集合についてのみ学習されていることに注意されたい。[式 269](#equation-269)で使用するため、各 $t_i$ を最も近いサポート対象の値、すなわち $t'_i = u_{\mathop{\mathrm{arg}\,\min}_j |u_j - t_i|}$ に丸める。

プロットから、$\rho=3$ は比較的良好な FID をもたらすものの、$\rho > 3$ を選ぶことでさらに低減できることが分かる。これは、高いノイズレベルで意図的に誤差を導入し、低いノイズレベルでの誤差を減らすことに相当する。そもそも $\sigma_{\max}$ の値はいくぶん恣意的であるため、これは直観的にも納得できる。$\sigma_{\max}$ を増やすと $\|\boldsymbol{e}_N\|$ に大きな影響を与え得るが、結果として得られる画像分布にはそれほど大きな影響を与えない。一般に、$\rho=7$ はすべての場合でかなり良好に機能することを確認しており、他のすべての実験でこの値を使用する。

<span id="section-10-2"></span>

### 10.2 2次 Runge-Kutta 変種の一般族

[アルゴリズム 1](#algorithm-01)に示す Heun 法は、いずれも同じ計算コストを持つ陽的2段2次 Runge-Kutta 法の一族に属する。この一族の一般的なパラメータ化 [Sul03] は次のとおりである。
$$
\boldsymbol{d}_i = f(\boldsymbol{x}_i;t_i)\ \ \ \textrm{;} \ \ \ \boldsymbol{x}_{i+1} = \boldsymbol{x}_i + h\Big[\Big(1-{\tfrac{1}{2\alpha}}\Big)\boldsymbol{d}_i+{\tfrac{1}{2\alpha}}f(\boldsymbol{x}_i + \alpha h \boldsymbol{d}_i;t_i+\alpha h)\Big]\textrm{,}
$$
ここで、$h=t_{i+1}-t_i$ であり、$\alpha$ は追加の勾配を評価する位置と、それが実行するステップにどの程度影響するかを制御するパラメータである。$\alpha=1$ とすると Heun 法に対応し、$\alpha=\tfrac{1}{2}$ および $\alpha=\tfrac{2}{3}$ とすると、それぞれいわゆる midpoint 法および Ralston 法が得られる。これらの変種はすべて、基礎となる関数 $f$ の幾何形状に起因して生じる近似誤差の種類が異なる。

本研究のユースケースにおける最適な $\alpha$ を確定するため、別個の一連の実験を行った。結果によると、$\alpha=1$ は最適値に非常に近いようである。それでも、実験的に最良の選択は $\alpha=1.1$ であり、わずかに良い性能を示した。ただし、1より大きい値は目標の $t_{i+1}$ を行き過ぎるため、理論的には正当化が困難である。この観察に対する十分な説明がなく、一般に成り立つかどうかも判断できないため、$\alpha$ を新たなハイパーパラメータとはせず、代わりに Heun 法と正確に一致する $1$ に固定することにした。サンプリング中に $\alpha$ を変化させる可能性も含め、さらなる分析は今後の課題とする。

$\alpha=1$ と設定するもう1つの利点は、事前学習済みニューラルネットワーク $D_\theta(\boldsymbol{x};\sigma)$ のうち、$\sigma$ の特定の値についてのみ学習されたものを使用できるようになることである。これは、他の2次変種とは異なり、Heun ステップが追加の勾配をちょうど $t_{i+1}$ で評価するためである。したがって、各 $t_i$ がネットワークの学習対象であった $\sigma$ の値に対応することを保証すれば十分である。

<span id="algorithm-03"></span>

<div class="paper-algorithm">

**アルゴリズム 3：一般的な2次 Runge-Kutta 法、$\sigma(t)=t$ および $s(t)=1$ を用いた決定論的サンプリング。**

- **手続き** $\operatorname{AlphaSampler}(D_\theta(\boldsymbol{x};\sigma), t_{i \in \{0, \dots, N\}}, \alpha)$
  - **サンプル** $\boldsymbol{x}_0 \sim \mathcal{N}(\mathbf{0}, t_0^2\mathbf{I})$。
  - **各** $i \in \{0, \dots, N-1\}$ **について**：
    - $h_i \gets t_{i+1}-t_i$。ステップ長。
    - $\boldsymbol{d}_i \gets (\boldsymbol{x}_i-D_\theta(\boldsymbol{x}_i;t_i))/t_i$。$\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ を $(\boldsymbol{x},t_i)$ で評価する。
    - $(\boldsymbol{x}'_i,t'_i) \gets (\boldsymbol{x}_i+\alpha h\boldsymbol{d}_i,t_i+\alpha h)$。追加の評価点。
    - **もし** $t'_i \ne 0$ **ならば**：
      - $\boldsymbol{d}'_i \gets (\boldsymbol{x}'_i-D_\theta(\boldsymbol{x}'_i;t'_i))/t'_i$。$\mathrm{d}\boldsymbol{x}/\mathrm{d}t$ を $(\boldsymbol{x}'_i,t'_i)$ で評価する。
      - $\boldsymbol{x}_{i+1} \gets \boldsymbol{x}_i+h[(1-\frac{1}{2\alpha})\boldsymbol{d}_i+\frac{1}{2\alpha}\boldsymbol{d}'_i]$。$t_i$ から $t_{i+1}$ への2次ステップ。
    - **それ以外の場合**：
      - $\boldsymbol{x}_{i+1} \gets \boldsymbol{x}_i+h\boldsymbol{d}_i$。$t_i$ から $t_{i+1}$ への Euler ステップ。
  - **返す** $\boldsymbol{x}_N$。

</div>

[アルゴリズム 3](#algorithm-03)は、$\alpha$ でパラメータ化された一般的な2次ソルバーの擬似コードを示す。明確さのため、この擬似コードでは、[第 3 節](#section-3)で推奨する $\sigma(t)=t$ と $s(t)=1$ という特定の選択を仮定している。Euler ステップへのフォールバック（11行目）は、$\alpha \ge 1$ の場合にのみ起こり得ることに注意されたい。

<span id="section-11"></span>

## 11 確率的サンプリングに関する追加結果

<span id="section-11-1"></span>

### 11.1 過剰な確率的反復による画像劣化

<span id="figure-14"></span>

![ノイズの追加と除去を繰り返すことによる段階的な画像劣化。](../../papers/diffusion-design-space/figure-14.png)

**図 14。** ノイズの追加と除去を繰り返すことによる段階的な画像劣化。$p(\boldsymbol{x}; \sigma)$ から抽出したランダムな画像（第1列）から開始し、$\gamma_i = \sqrt{2}-1$ を固定して、一定数のステップ（残りの列）にわたり[アルゴリズム 2](#algorithm-02)を実行する。各行は、処理全体を通して固定する $\sigma$ の特定の選択（中央に表示）に対応する。デノイザーを通した後の結果、すなわち $D_\theta(\boldsymbol{x}_i; \sigma)$ を可視化する。

[図 14](#figure-14)は、過剰な Langevin 反復によって生じる画像劣化を示す（[第 4 節](#section-4)「実用上の考慮事項」）。これらの画像は、固定ノイズレベル $\sigma$ で指定回数の反復を行い、各反復で同量のノイズを追加して除去することにより生成される。理論上、Langevin dynamics は分布を理想分布 $p(\boldsymbol{x};\sigma)$ に近づけるはずだが、[第 4 節](#section-4)で述べたように、これはデノイザー $D_\theta(\boldsymbol{x};\sigma)$ が[式 3](#equation-03)で保存ベクトル場を誘導する場合にのみ成り立つ。

図から分かるように、正確な失敗モードはデータセットとノイズレベルに依存するものの、すべての場合で画像分布が反復の繰り返しによって損なわれることは明らかである。低いノイズレベル（およそ $0.2$ 未満）では、画像は2k回の反復から過飽和になり始め、その後は完全に破損する傾向がある。$S_\text{tmin}> 0$ と設定する本研究のヒューリスティックは、この影響を回避するため、非常に低いノイズレベルで確率的サンプリングを全面的に防ぐよう設計されている。

高いノイズレベルでは、標準偏差補正を行わずに反復した場合、すなわち $S_\text{noise}=1.000$ の場合、反復回数が多くなると画像がより抽象的になり、色彩を失う傾向が見られる。これは CIFAR-10 の10k列で特に顕著であり、画像はほぼ白黒となり、識別可能な背景がなくなる。$S_\text{noise}> 1$ と設定して標準偏差をヒューリスティックに膨張させると、図の右側にある対応画像に見られるように、この傾向を効率よく打ち消せる。注目すべきことに、それでも低いノイズレベルでの過飽和と破損は解消されず、過剰な反復の有害な影響には複数の原因があることを示唆している。観察されたこれらの影響の根本原因をより深く理解するには、さらなる研究が必要である。

<span id="figure-15"></span>

![Song らの事前学習済みネットワークを用いた、確率的サンプラー（アルゴリズム 2）のパラメータに関するアブレーション。](../../papers/diffusion-design-space/figure-15.png)

**図 15。** Song ら [Son21] および Dhariwal と Nichol [Dha21] の事前学習済みネットワークを用いた、本研究の確率的サンプラー（[アルゴリズム 2](#algorithm-02)）のパラメータに関するアブレーション。各曲線は FID（$y$ 軸）を $S_\text{churn}$（$x$ 軸）の関数として、$N = 256$ ステップ（NFE = 511）について示す。赤い破線は本研究の決定論的サンプラー（[アルゴリズム 1](#algorithm-01)）に対応し、$S_\text{churn}= 0$ と設定することに等しい。紫色の曲線は、各ケースについてグリッドサーチで個別に求めた $\{S_\text{tmin}, S_\text{tmax}, S_\text{noise}\}$ の最適な選択に対応する。オレンジ、青、緑は、$S_\text{tmin,tmax}$ および／または $S_\text{noise}$ の効果を無効にした場合に対応する。網掛け領域は、観測された FID の最小値と最大値の間の変動範囲を示す。

[図 15](#figure-15)は、Song ら [Son21] および Dhariwal と Nichol [Dha21] の事前学習済みネットワークを用い、NFE を固定したときの $S_\text{churn}$ の関数として、FID で表した本研究の確率的サンプラーの出力品質を示す。一般に、各ケースとヒューリスティック補正の各組み合わせには最適な確率性の量があり、それを超えると結果が劣化し始める。また、$S_\text{churn}$ の値にかかわらず、すべての補正を有効にしたときに最良の結果が得られることも分かる。ただし、$S_\text{noise}$ と $S_\text{tmin,tmax}$ のどちらがより重要かはケースによって異なる。

<span id="section-11-2"></span>

### 11.2 確率的サンプリングのパラメータ

<span id="table-05"></span>

![確率的サンプリング実験のパラメータ。](../../papers/diffusion-design-space/table-05.png)

**表 5。** [第 4 節](#section-4)の確率的サンプリング実験で使用したパラメータ。

[表 5](#table-05)に、本研究の確率的サンプリング実験で使用した $S_\text{churn}$、$S_\text{tmin}$、$S_\text{tmax}$、および $S_\text{noise}$ の値を示す。これらは、右端の列に列挙された組み合わせに対するグリッドサーチで決定した。最適なパラメータはケースに依存することが分かる。劣化現象をよりよく理解することで、将来この問題をより直接的に扱う方法が生まれることが期待される。

<span id="section-12"></span>

## 12 実装の詳細

本研究の手法は、Song ら [+19] [Son21]、Dhariwal と Nichol [+20] [Dha21]、および Karras ら [+21] [Kar21] の元の実装を大まかに基に、新たに作成したコードベースへ実装した。サンプラー、事前学習済みモデル、ネットワークアーキテクチャ、学習構成、評価を含め、本研究の実装が先行研究とまったく同じ結果を生成することを確認するため、広範なテストを実施した。すべての実験は、各マシンに8基の Tesla V100 GPU を搭載した NVIDIA DGX-1 上で、PyTorch 1.10.0、CUDA 11.4、および CuDNN 8.2.0 を使用して実行した。

本研究の実装と事前学習済みモデルは <https://github.com/NVlabs/edm> で入手できる。

<span id="section-12-1"></span>

### 12.1 FID の計算

50,000枚の生成画像と利用可能なすべての実画像との間で、$x$ 反転などのデータ拡張を一切行わずに FID [Heu17] を計算する。StyleGAN3 [+22] [Kar21] に付属する事前学習済み Inception-v3 モデルを使用するが、これはさらに、元の TensorFlow ベースのモデルを直接 PyTorch に移植したものである [+23]。本研究の FID 実装が、Dhariwal と Nichol [Dha21] および Karras ら [Kar21] と比較して同一の結果を生成することを確認した。通常 $\pm$2% 程度であるランダムな変動の影響を抑えるため、各実験で FID を3回計算し、最小値を報告する。また、[図 4](#figure-04)、[図 5b](#figure-05)、[図 13c](#figure-13)、および[図 15](#figure-15)では、得られた FID の最高値と最低値の差も強調している。

<span id="section-12-2"></span>

### 12.2 データ拡張正則化

[第 5 節](#section-5)では、条件付きデータ拡張を用いて $D_\theta$ の過学習に対処することを提案する。本研究のデータ拡張パイプラインは、もともと GAN の文脈で Karras ら [Kar20a] が提案したものと同じ概念に基づいて構築する。実際には6種類の幾何変換を使用する。色の破壊や画像空間フィルタリングなど、他の種類のデータ拡張は、拡散ベースのモデルに一貫して有害であることを確認した。

本研究のデータ拡張パイプラインの詳細を[表 6](#table-06)に示す。各学習画像 $\boldsymbol{y}\sim p_\text{data}$ にデータ拡張を独立に適用してから、ノイズ $\boldsymbol{n}\sim \mathcal{N}(\mathbf{0}, \sigma^2 \mathbf{I})$ を追加する。まず、重み付きコイントスに基づいて各データ拡張を有効化するか無効化するかを決定する。所与のデータ拡張を有効にする確率（「Prob.」列）は、常に有効にする $x$ 反転を除き、CIFAR-10 では12%、FFHQ と AFHQv2 では15%に固定する。次に、対応する分布（「Parameters」列）から8個のランダムパラメータを抽出する。所与のデータ拡張が無効な場合、関連するパラメータをゼロで上書きする。これらに基づき、パラメータ（「Transformation」列）に基づく同次2次元変換行列を構築する。この変換は、2$\times$ スーパーサンプリングされた高品質 Wavelet フィルタを用いる [Kar20a] の実装によって画像へ適用される。最後に、9次元の条件付け入力ベクトル（「Conditioning」列）を構築し、画像およびノイズレベルの入力に加えてデノイザーネットワークへ入力する。

<span id="table-06"></span>

![本研究のデータ拡張パイプライン。各学習画像には、一定の確率で非ゼロ値を取る8個のランダムパラメータに基づく複合幾何変換が施される。](../../papers/diffusion-design-space/table-06.png)

**表 6。** 本研究のデータ拡張パイプライン。各学習画像には、一定の確率で非ゼロ値を取る8個のランダムパラメータに基づく複合幾何変換が施される。

条件付け入力の役割は、ネットワークに一連の補助タスクを提示することである。$p(\boldsymbol{x}; \sigma)$ をモデル化する主タスクに加え、無限個の分布 $p(\boldsymbol{x}; \sigma, \boldsymbol{a})$ も、データ拡張パラメータ $\boldsymbol{a}$ の可能な各選択についてモデル化するようネットワークに実質的に求める。これらの補助タスクは、多種多様な固有の学習サンプルをネットワークに与え、個々のサンプルへの過学習を防ぐ。それでも、補助タスクは主タスクにも有益であるように見える。これは、デノイズ操作自体が $\boldsymbol{a}$ のどの選択に対しても類似しているためだと推測する。

条件付け入力は、ゼロがデータ拡張を適用しなかった場合に対応するよう設計した。サンプリング中は、主タスクと整合する結果を得るため、単に $\boldsymbol{a} = \mathbf{0}$ と設定する。補助タスクと主タスクの間に漏洩は観測されておらず、$A_\text{prob} = 100$% であっても、生成画像にはドメイン外の幾何変換の痕跡がまったく現れない。実際には、これは結果が改善する限り、定数 $\{A_\text{prob}, A_\text{scale}, A_\text{aniso}, A_\text{trans}\}$ を自由に選べることを意味する。水平反転は興味深い例である。先行研究の大半は、ランダムな $x$ 反転で学習セットを拡張しており、これはほとんどのデータセットに有益だが、生成画像中のテキストやロゴが鏡像として現れ得るという欠点がある。本研究の漏洩のないデータ拡張では、$x$ 反転のデータ拡張を100%の確率で実行することにより、欠点を伴わずに同じ利点を得られる。したがって、生成画像を元の分布に忠実に保つため、本研究のデータ拡張方式だけに依存し、データセットの $x$ 反転を無効化する。

<span id="section-12-3"></span>

### 12.3 学習構成

<span id="table-07"></span>

![学習ハイパーパラメータ。](../../papers/diffusion-design-space/table-07.png)

**表 7。** [第 5 節](#section-5)の学習実行で使用したハイパーパラメータ。

[表 7](#table-07)は、[第 5 節](#section-5)で報告した学習実験で使用したハイパーパラメータの正確な一式を示す。まず CIFAR-10、FFHQ、AFHQv2 で使用した構成を詳述し、次に改良した ImageNet モデルの学習について論じる。

[表 2](#table-02)の構成 A（「Baseline」）は2つのケース（VP と VE）に対する Song ら [Son21] の元の設定に対応し、構成 F（「Ours」）は本研究の改良設定に対応する。学習セットから合計2億枚の画像が抽出されるまで各モデルを学習した。これは[表 7](#table-07)で「200 Mimg」と略記しており、バッチサイズ512を用いた合計 $\sim$400,000回の学習反復に相当する。250万枚の画像ごとにモデルのスナップショットを保存し、解像度に応じて NFE $=$ 35 または NFE $=$ 79 の決定論的サンプラーで最小の FID を達成したスナップショットの結果を報告した。

構成 B では、より高速な学習を可能にし、より意味のある比較点を得るため、基本ハイパーパラメータを再調整する。具体的には、解像度に応じて、並列度を4基から8基の GPU へ、バッチサイズを128から512または256へ増やす。また、$\| \mathrm{d}\mathcal{L}(D_\theta) / \mathrm{d}\theta \|_2 \le 1$ を強制する勾配クリッピングを無効にする。これは実際には何の利点ももたらさないことを確認した。さらに、CIFAR-10 では学習率を0.0002から0.001へ引き上げ、最初の1000万枚の画像にわたってランプアップし、$\theta$ の指数移動平均の半減期を50万枚の画像に統一する。最後に、1%刻みの完全なグリッドサーチにより、[表 7](#table-07)に示すとおり各データセットの dropout 確率を調整する。総学習時間は、32$\times$32解像度の CIFAR-10 で約2日、64$\times$64解像度の FFHQ と AFHQv2 で4日である。

構成 C では、4$\times$4層を取り除き、代わりに16$\times$16層の容量を倍増させることで、モデルの表現力を改善する。前者は主に過学習に寄与する一方、後者は高品質な結果を得るうえで不可欠であることを確認した。Song ら [Son21] の元のモデルは、64$\times$64（該当する場合）および32$\times$32で128チャネル、16$\times$16、8$\times$8、および4$\times$4で256チャネルを使用する。これらを、64$\times$64（該当する場合）で128チャネル、32$\times$32、16$\times$16、および8$\times$8で256チャネルに変更する。[表 7](#table-07)では、これらのチャネル数を128の倍数として略記し、最高解像度から最低解像度の順に列挙する。実際には、この再配分により学習可能パラメータの総数がわずかに減少し、結果として、各モデルは $\sim$5600万個のパラメータを32$\times$32解像度で、$\sim$6200万個のパラメータを64$\times$64解像度で持つ。

構成 D では、元の事前条件付けを本研究の改良した式（[表 1](#table-01)の「Network and preconditioning」セクション）に置き換える。構成 E では、ノイズ分布と損失の重み付けについて同じ置き換えを行う（[表 1](#table-01)の「Training」セクション）。最後に、構成 F では、[第 12.2 節](#section-12-2)で論じたデータ拡張正則化を有効にする。他のハイパーパラメータは構成 C と同じままである。

ImageNet-64 では、最先端の結果に到達するため、他のデータセットと比べて大幅に長く学習する必要がある。学習時間を短縮するため、バッチサイズ4096（GPU 1基あたり128）で32基の NVIDIA Ampere GPU（4ノード）を使用し、混合精度 FP16/FP32 学習によって高性能 Tensor Core を利用した。実際には、学習可能パラメータを FP32 として保存するが、$F_\theta$ を評価するときは FP16 にキャストする。ただし、embedding 層と self-attention 層では、FP16 の限られた指数範囲が時折安定性の問題を引き起こすことを確認したため、この限りではない。モデルを2週間学習した。これは学習セットから抽出された $\sim$25億枚の画像と $\sim$600,000回の学習反復に相当し、学習率0.0001、5000万枚の画像に相当する指数移動平均、および Dhariwal と Nichol [Dha21] と同じモデルアーキテクチャと dropout 確率を使用した。過学習が問題になるとは認められなかったため、データ拡張正則化は使用しないことにした。

<span id="section-12-4"></span>

### 12.4 ネットワークアーキテクチャ

<span id="table-08"></span>

![本論文で使用したネットワークアーキテクチャの詳細。](../../papers/diffusion-design-space/table-08.png)

**表 8。** 本論文で使用したネットワークアーキテクチャの詳細。

学習上の改善の結果、構成 F における VP と VE のケースは、ネットワークアーキテクチャを除けば同一になる。VP は DDPM++ アーキテクチャを、VE は NCSN++ を使用し、いずれも元は Song ら [Son21] が提案したものである。[表 8](#table-08)に示すように、これらのアーキテクチャは、同じ U-net バックボーンを比較的単純に変形したものであり、3つの相違点がある。第1に、DDPM++ は upsampling 層と downsampling 層に box filter $[1, 1]$ を使用するのに対し、NCSN++ は bilinear filter $[1, 3, 3, 1]$ を使用する。第2に、DDPM++ はノイズレベルの positional encoding 方式を DDPM [Den20] から直接継承するのに対し、NCSN++ はこれを random Fourier features [Tan20a] に置き換える。第3に、NCSN++ は [Son21] の Appendix H（「progressive growing architectures」）で説明されているように、入力画像から encoder の各ブロックへの追加の residual skip connection を組み込む。

クラス条件付けとデータ拡張正則化のため、ノイズレベル入力と並行して2つのオプションの条件付け入力を導入し、元の DDPM++ および NCSN++ アーキテクチャを拡張する。クラスラベルは one-hot encoding したベクトルとして表し、まず $\sqrt{C}$ 倍する。ここで、$C$ はクラスの総数であり、その後ベクトルを全結合層に通す。データ拡張パラメータについては、[第 12.2 節](#section-12-2)の条件付け入力をそのまま全結合層に通す。次に、得られた特徴ベクトルを元のノイズレベル条件付けベクトルと要素ごとの加算によって結合する。

クラス条件付き ImageNet-64 には、Dhariwal と Nichol [Dha21] の ADM アーキテクチャを変更せずに使用する。このモデルには合計 $\sim$2億9600万個の学習可能パラメータがある。[表 7](#table-07)および[表 8](#table-08)で詳述するように、DDPM++ との最も顕著な違いには、チャネル数を大幅に増やした（例えば最低解像度で256ではなく768）やや浅いモデル（解像度ごとに4個ではなく3個の residual block）を使用すること、ネットワーク全体により多くの self-attention 層を散在させること（6層ではなく22層）、および multi-head attention を使用すること（例えば最低解像度で12 head）が含まれる。アーキテクチャ選択の正確な影響は、今後も興味深い研究課題であると考える。

<span id="section-12-5"></span>

### 12.5 ライセンス

データセット：

- CIFAR-10 [Kri09]：MIT ライセンス

- FFHQ [Kar18]：Creative Commons BY-NC-SA 4.0 ライセンス

- AFHQv2 [Cho20c]：Creative Commons BY-NC 4.0 ライセンス

- ImageNet [Den09a]：ライセンスの状態は不明

事前学習済みモデル：

- Song らの CIFAR-10 モデル [Son21]：Apache V2.0 ライセンス

- Dhariwal と Nichol の ImageNet-64 モデル [Dha21]：MIT ライセンス

- Szegedy らの Inception-v3 モデル [Sze16]：Apache V2.0 ライセンス

[+1]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/utils.py#L144>

[+2]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/losses.py#L73>

[+3]: `vp/cifar10_ddpmpp_continuous/checkpoint_8.pth`, <https://drive.google.com/drive/folders/1xYjVMx10N9ivQQBIsEoXEeu9nvSGTBrC>

[+4]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py#L182>

[+5]: <https://github.com/yang-song/score_sde_pytorch>

[+6]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py#L191>

[+7]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L102>

[+8]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L246>

[+9]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/utils.py#L163>

[+10]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L239>

[+11]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L261>

[+12]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/models/ncsnpp.py#L379>

[+13]: <https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py#L234>

[+14]: `ve/cifar10_ncsnpp_continuous/checkpoint_24.pth`, <https://drive.google.com/drive/folders/1b0gy_LLgO_DaQBgoWXwlVnL_rcAUgREh>

[+15]: <https://github.com/openai/improved-diffusion>

[+16]: <https://github.com/openai/improved-diffusion/blob/783b6740edb79fdb7d063250db2c51cc9545dcd1/improved_diffusion/gaussian_diffusion.py#L39>

[+17]: `https://openaipublic.blob.core.windows.net/diffusion/jul-2021/64x64_diffusion.pt`

[+18]: 極限では、$\rho \rightarrow \infty$ のとき[式 269](#equation-269)は元の VE ODE で使用されているものと同じ等比数列に帰着する。したがって、本研究の離散化は Song ら [Son21] が提案したもののパラメトリックな一般化と見なせる。

[+19]: <https://github.com/yang-song/score_sde_pytorch>

[+20]: <https://github.com/openai/guided-diffusion>

[+21]: <https://github.com/NVlabs/stylegan3>

[+22]: `https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl`

[+23]: `http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz`
