---
title: 'Retentive Network for Large Language Models'
createTime: 2026/09/05 14:30:00
permalink: /ja/papers/retentive-network/
pageClass: paper-reading
---

> [Yutao Sun](https://dblp.org/pid/01/9758) [+equal]、[Li Dong](https://dblp.org/pid/85/5090-4) [+equal]、[Shaohan Huang](https://www.microsoft.com/en-us/research/people/shaohanh/)、[Shuming Ma](https://dblp.org/pid/190/7739)、[Yuqing Xia](https://dblp.org/pid/211/8365)、[Jilong Xue](https://dblp.org/pid/06/10336)、[Jianyong Wang](https://dblp.org/pid/24/2006)、[Furu Wei](https://dblp.org/pid/72/5870) [+corresponding]。arXiv 初回投稿：2023 年 7 月 17 日。現行版：v4。[Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621)。<a href="/paper/retentive-network.pdf" target="_blank" rel="noopener noreferrer">原論文 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2307.08621)。[TeX ソース](https://export.arxiv.org/e-print/2307.08621v4)。正確な印刷レイアウトと参考文献については、原論文 PDF を参照されたい。

[+equal]: 同等の貢献。

[+corresponding]: 責任著者。

## 概要

本研究では、大規模言語モデルの基盤アーキテクチャとして **Retentive Network**（RetNet）を提案し、訓練の並列性、低コストな推論、良好な性能を同時に実現する。まず、再帰と attention の関係を理論的に導出する。次に、系列モデリングのための retention 機構を提案する。この機構は、並列、再帰、チャンク単位の再帰という 3 種類の計算パラダイムをサポートする。具体的には、並列表現によって訓練を並列化できる。再帰表現では低コストな $O(1)$ 推論が可能となり、性能を損なうことなく、デコードのスループット、レイテンシ、GPU メモリを改善する。チャンク単位の再帰表現は、線形計算量で効率的な長系列モデリングを実現する。各チャンクを並列に符号化しながら、チャンクを再帰的に要約する。言語モデリング実験の結果、RetNet は良好なスケーリング、並列訓練、低コストなデプロイ、効率的な推論を実現した。これらの興味深い特性により、RetNet は大規模言語モデルにおける Transformer の有力な後継となる。コードは [https://aka.ms/retnet](https://aka.ms/retnet) で公開予定である。

<span id="figure-01"></span>

![Retentive network の推論コストとスケーリング曲線](../../papers/retentive-network/figure-01.png)

**図 1。** Retentive network（RetNet）は Transformer と比べて、低コストな推論（GPU メモリ、スループット、レイテンシ）、訓練の並列性、良好なスケーリング曲線を実現する。推論コストの結果は入力長 8k で報告している。[図 6](#figure-06) に、異なる系列長での追加結果を示す。

> 可能性の限界を発見する唯一の方法は、その限界を越えて不可能の領域へ進むことである。
>
> —Arthur C. Clarke

<span id="section-1"></span>

## 1 はじめに

<span id="figure-02"></span>

![RetNet によって可能となる不可能な三角形](../../papers/retentive-network/figure-02.png)

**図 2。** RetNet は、訓練の並列性、良好な性能、低い推論コストを同時に実現し、「不可能な三角形」を可能にする。

Transformer [Vas17] は大規模言語モデル [Bro20] の事実上の標準アーキテクチャとなっており、元来は再帰モデル [Hoc97] における逐次的訓練の問題を克服するために提案された。しかし、Transformer の訓練並列性は非効率な推論と引き換えである。各ステップの計算量が $O(N)$ であり、メモリ帯域に制約される key-value cache [Sha19] を要するため、Transformer はデプロイに適さない。系列長が増えるほど GPU メモリ消費とレイテンシが増加し、推論速度が低下する。

Transformer と同等の訓練並列性と競争力のある性能を保ちつつ、効率的な $O(1)$ 推論を行う次世代アーキテクチャの開発が続けられてきた。これらの目標を同時に達成することは難しく、[図 2](#figure-02) に示す、いわゆる「不可能な三角形」となる。

研究には主に 3 つの潮流がある。第 1 に、線形化 attention [Kat20] は標準 attention score $\exp(\bm{q}\cdot\bm{k})$ を kernel $\phi(\bm{q})\cdot\phi(\bm{k})$ で近似し、自己回帰推論を再帰形式に書き換える。しかし、モデリング能力と性能は Transformer より劣り、この手法の普及を妨げている。第 2 の潮流は、訓練並列性を犠牲にして、効率的な推論のために再帰モデルへ回帰する。対策として element-wise operator [Pen23b] を用いて高速化するが、表現能力と性能が損なわれる。第 3 の潮流は、S4 [Gu22] やその変種 [Dao22g, Pol23a] など、attention を別の機構で置き換えることを検討する。従来研究はいずれも不可能な三角形を突破できず、Transformer に対する明確な勝者は現れていない。

本研究では、低コストな推論、効率的な長系列モデリング、Transformer と同等の性能、並列なモデル訓練を同時に実現する retentive network（RetNet）を提案する。具体的には、multi-head attention を置き換える multi-scale retention 機構を導入する。この機構は、並列、再帰、チャンク単位の再帰という 3 種類の計算パラダイムを持つ。第 1 に、並列表現は GPU を十分に活用する訓練並列性をもたらす。第 2 に、再帰表現はメモリと計算の両面で効率的な $O(1)$ 推論を可能にする。デプロイコストとレイテンシを大幅に削減できる。さらに、key-value cache の技巧を使わずに実装を大きく簡略化できる。第 3 に、チャンク単位の再帰表現により、効率的な長系列モデリングが可能となる。計算速度を高めるため各局所ブロックを並列に符号化し、GPU メモリを節約するため大域ブロックを再帰的に符号化する。

RetNet と Transformer およびその変種を比較するため、広範な実験を行う。言語モデリング実験の結果、RetNet はスケーリング曲線と in-context learning の両面で一貫して競争力を持つ。さらに、RetNet の推論コストは長さに依存しない。7B モデルと系列長 8k では、RetNet のデコードは key-value cache を持つ Transformer より 8.4$\times$ 高速で、メモリを 70% 削減する。訓練中も、RetNet は標準 Transformer よりメモリを 25-50% 削減し、7$\times$ 高速化する。また、高度に最適化された FlashAttention [Dao22] に対しても優位性を持つ。RetNet の推論レイテンシは batch size に影響されにくく、大きなスループットを実現できる。これらの興味深い特性により、RetNet は大規模言語モデルにおける Transformer の有力な後継となる。

<span id="section-2"></span>

## 2 Retentive Network

Retentive network（RetNet）は $L$ 個の同一 block を積層したもので、Transformer [Vas17] と同様のレイアウト（residual connection と pre-LayerNorm）に従う。各 RetNet block は、multi-scale retention（MSR）module と feed-forward network（FFN）module の 2 つを含む。以下の節で MSR module を説明する。入力系列 $x=x_1\cdots x_{|x|}$ が与えられると、RetNet は系列を自己回帰的に符号化する。入力 vector $\{\bm{x}_i\}_{i=1}^{|x|}$ はまず $X^0=[\bm{x}_1,\cdots,\bm{x}_{|x|}]\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$ にまとめられる。ここで $d_{\mathrm{model}}$ は hidden dimension である。次に、文脈化された vector 表現 $X^l=\mathrm{RetNet}_l(X^{l-1}), l\in[1,L]$ を計算する。

<span id="section-2-1"></span>

### 2.1 Retention

本節では、再帰と並列性の双対形式を持つ retention 機構を説明する。これにより、モデルを並列に訓練し、再帰的に推論できる。

入力 $X\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$ が与えられると、これを 1 次元関数 $v(n)=X_n\cdot\bm{w}_V$ に射影する。状態 $\bm{s}_n$ を介して $v(n)$ を $o(n)$ に写像する系列モデリング問題を考える。簡潔さのため、$v(n),o(n)$ を $v_n,o_n$ と表す。この写像を再帰形式で定式化する。

<span id="equation-01"></span>

$$
\begin{aligned}
\bm{s}_n &= A\bm{s}_{n-1}+K_n^\top v_n, &A\in\mathbb{R}^{d\times d}, K_n\in\mathbb{R}^{1\times d} \\
o_n &= Q_n\bm{s}_n=\sum_{m=1}^n Q_nA^{n-m}K_m^\top v_m, &Q_n\in\mathbb{R}^{1\times d}
\end{aligned}
$$

ここでは $v_n$ を状態 vector $\bm{s}_n$ に写像し、続いて線形変換を施して系列情報を再帰的に符号化する。

次に、射影 $Q_n,K_n$ を content-aware にする。

<span id="equation-02"></span>

$$
Q=X W_Q,\quad K=X W_K
$$

ここで、$W_Q,W_K\in\mathbb{R}^{d\times d}$ は学習可能な行列である。

行列を $A=\Lambda(\gamma e^{i\theta})\Lambda^{-1}$ と対角化する。ここで $\gamma,\theta\in\mathbb{R}^d$ である。すると $A^{n-m}=\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}$ を得る。$\Lambda$ を $W_Q$ と $W_K$ に吸収すると、[式 (1)](#equation-01) は次のように書き換えられる。

<span id="equation-03"></span>

$$
\begin{aligned}
o_n &= \sum_{m=1}^n Q_n(\gamma e^{i\theta})^{n-m}K_m^\top v_m \\
&=\sum_{m=1}^n(Q_n(\gamma e^{i\theta})^n)(K_m(\gamma e^{i\theta})^{-m})^\top v_m
\end{aligned}
$$

ここで $Q_n(\gamma e^{i\theta})^n,K_m(\gamma e^{i\theta})^{-m}$ は xPos [Sun22] として知られ、Transformer のために提案された相対位置埋め込みである。さらに $\gamma$ を scalar に簡略化すると、[式 (3)](#equation-03) は次のようになる。

<span id="equation-04"></span>

$$
o_n=\sum_{m=1}^n\gamma^{n-m}(Q_n e^{i n\theta})(K_m e^{i m\theta})^\dagger v_m
$$

ここで $^\dagger$ は共役転置である。この定式化は、訓練 instance 内で容易に並列化できる。

要約すると、[式 (1)](#equation-01) に示す再帰モデリングから出発し、[式 (4)](#equation-04) の並列定式化を導出する。元の写像 $v(n)\mapsto o(n)$ を vector として扱うと、次の retention 機構が得られる。

<span id="figure-03"></span>

![RetNet の並列表現と再帰表現](../../papers/retentive-network/figure-03.png)

**図 3。** RetNet の双対形式。「GN」は GroupNorm の略である。

**Retention の並列表現。** [図 3a](#figure-03) に示すように、retention layer は次のように定義される。

<span id="equation-05"></span>

$$
\begin{aligned}
Q=(X W_Q)\odot\Theta,&\quad K=(X W_K)\odot\overline{\Theta},\quad V=X W_V \\
\Theta_n=e^{i n\theta},&\quad
D_{nm}=\begin{cases}
\gamma^{n-m}, & n\ge m \\
0, & n<m
\end{cases} \\
\mathrm{Retention}(X)&=(Q K^\top\odot D)V
\end{aligned}
$$

ここで $\overline{\Theta}$ は $\Theta$ の複素共役であり、$D\in\mathbb{R}^{|x|\times|x|}$ は因果 mask と相対距離に沿う指数減衰を 1 つの行列に統合する。self-attention と同様に、並列表現によって GPU でモデルを効率的に訓練できる。

**Retention の再帰表現。** [図 3b](#figure-03) に示すように、提案機構は recurrent neural network（RNN）としても記述でき、推論に適している。第 $n$ time step では、出力を次のように再帰的に得る。

<span id="equation-06"></span>

$$
\begin{aligned}
S_n &= \gamma S_{n-1}+K_n^\top V_n \\
\mathrm{Retention}(X_n)&=Q_nS_n,\quad n=1,\cdots,|x|
\end{aligned}
$$

ここで $Q,K,V,\gamma$ は[式 (5)](#equation-05) と同じである。

**Retention のチャンク単位の再帰表現。** 並列表現と再帰表現の hybrid 形式を用いることで、特に長い系列の訓練を高速化できる。入力系列をチャンクに分割する。各チャンク内では並列表現（[式 (5)](#equation-05)）に従って計算する。一方、チャンク間の情報は再帰表現（[式 (6)](#equation-06)）に従って伝える。具体的に、$B$ をチャンク長とする。第 $i$ チャンクの retention 出力を次のように計算する。

<span id="equation-07"></span>

$$
\begin{aligned}
Q_{[i]}=Q_{Bi:B(i+1)}&,\quad K_{[i]}=K_{Bi:B(i+1)},\quad V_{[i]}=V_{Bi:B(i+1)} \\
R_i&=K_{[i]}^\top(V_{[i]}\odot\zeta)+\gamma^B R_{i-1},\quad\zeta_{ij}=\gamma^{B-i-1} \\
\mathrm{Retention}(X_{[i]})&=\underbrace{(Q_{[i]} K_{[i]}^\top\odot D)V_{[i]}}_{\mathrm{Inner}{-}\mathrm{Chunk}}+\underbrace{(Q_{[i]}R_{i-1})\odot\xi}_{\mathrm{Cross}{-}\mathrm{Chunk}},\quad\xi_{ij}=\gamma^{i+1}
\end{aligned}
$$

ここで ${[i]}$ は第 $i$ チャンクを表し、すなわち $x_{[i]}=[x_{(i-1)B+1},\cdots,x_{iB}]$ である。

<span id="section-2-2"></span>

### 2.2 Gated Multi-Scale Retention

各 layer で $h=\frac{d_{\mathrm{model}}}{d}$ 個の retention head を用いる。ここで $d$ は head dimension である。各 head は異なる parameter matrix $W_Q,W_K,W_V\in\mathbb{R}^{d\times d}$ を使用する。さらに、**m**ulti-**s**cale **r**etention（MSR）は head ごとに異なる $\gamma$ を割り当てる。簡潔さのため、異なる layer 間で $\gamma$ を同一かつ固定とする。また、retention layer の非線形性を高めるため、$\mathrm{swish}$ gate [Hen16, Ram17] を加える。形式的には、入力 $X$ に対して layer を次のように定義する。

<span id="equation-08"></span>

$$
\begin{aligned}
\bm{\gamma}&=1-2^{-5-\mathrm{arange}(0,h)}\in\mathbb{R}^h \\
\mathrm{head}_i&=\mathrm{Retention}(X,\gamma_i) \\
Y&=\mathrm{GroupNorm}_h(\mathrm{Concat}(\mathrm{head}_1,\cdots,\mathrm{head}_h)) \\
\mathrm{MSR}(X)&=(\mathrm{swish}(X W_G)\odot Y)W_O
\end{aligned}
$$

ここで $W_G,W_O\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{model}}}$ は学習可能な parameter であり、$\mathrm{GroupNorm}$ [Wu18c] は [Sho19] で提案された SubLN に従って各 head の出力を正規化する。head は複数の scale の $\gamma$ を用いるため、分散統計量が異なる。そこで、各 head の出力を個別に正規化する。

<span id="figure-04"></span>

![Retention の 3 種類の計算パラダイムの疑似コード](../../papers/retentive-network/figure-04.png)

**図 4。** Retention の 3 種類の計算パラダイムの疑似コード。

[図 4](#figure-04) に retention の疑似コードをまとめる。

**Retention score の正規化。** $\mathrm{GroupNorm}$ の scale 不変性を利用して、retention layer の数値精度を改善する。具体的には、$\mathrm{GroupNorm}$ 内で scalar 値を乗じても、出力と backward gradient は変化しない。すなわち、$\mathrm{GroupNorm}(\alpha*\mathrm{head}_i)=\mathrm{GroupNorm}(\mathrm{head}_i)$ である。[式 (5)](#equation-05) に 3 つの正規化因子を実装する。第 1 に、$Q K^\top$ を $\frac{Q K^\top}{\sqrt{d}}$ として正規化する。第 2 に、$D$ を $\tilde{D}_{nm}=\frac{D_{nm}}{\sqrt{\sum_{i=1}^nD_{ni}}}$ で置き換える。第 3 に、$R$ を retention score $R=Q K^\top\odot D$ とし、$\tilde{R}_{nm}=\frac{R_{nm}}{\max(|\sum_{i=1}^nR_{ni}|,1)}$ として正規化する。このとき retention 出力は $\mathrm{Retention}(X)=\tilde{R}V$ となる。scale 不変性により、これらの工夫は最終結果に影響せず、forward と backward の両方で数値の流れを安定化する。

<span id="section-2-3"></span>

### 2.3 Retentive Network の全体アーキテクチャ

$L$ layer の retentive network では、multi-scale retention（MSR）と feed-forward network（FFN）を積層してモデルを構築する。形式的には、入力系列 $\{x_i\}_{i=1}^{|x|}$ を word embedding layer によって vector に変換する。まとめた embedding $X^0=[\bm{x}_1,\cdots,\bm{x}_{|x|}]\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$ を入力とし、モデル出力 $X^L$ を計算する。

<span id="equation-09"></span>

$$
\begin{aligned}
Y^l&=\mathrm{MSR}(\mathrm{LN}(X^l))+X^l \\
X^{l+1}&=\mathrm{FFN}(\mathrm{LN}(Y^l))+Y^l
\end{aligned}
$$

ここで $\mathrm{LN}(\cdot)$ は LayerNorm [Ba16] である。FFN 部分は $\mathrm{FFN}(X)=\mathrm{gelu}(X W_1)W_2$ として計算する。ここで $W_1,W_2$ は parameter matrix である。

**訓練。** 訓練中は、並列表現（[式 (5)](#equation-05)）とチャンク単位の再帰表現（[式 (7)](#equation-07)）を用いる。系列またはチャンク内の並列化により、GPU を効率よく利用して計算を高速化できる。チャンク単位の再帰は長系列の訓練に特に有用であり、FLOPs とメモリ消費の両面で効率的である。

**推論。** 推論には再帰表現（[式 (6)](#equation-06)）を用いる。これは自己回帰デコードによく適合する。$O(1)$ 計算量によって、同等の結果を保ちながらメモリと推論レイテンシを削減する。

<span id="section-2-4"></span>

### 2.4 従来手法との関係と相違点

[表 1](#table-01) では、複数の観点から RetNet と従来手法を比較する。比較結果は[図 2](#figure-02) の「不可能な三角形」と対応している。さらに、チャンク単位の再帰表現により、RetNet は長系列に対して線形のメモリ計算量を持つ。個別の手法との比較も以下にまとめる。

**Transformer。** Retention の並列表現は Transformer [Vas17] と類似した考え方を持つ。最も関連する Transformer 変種は、xPos を位置埋め込みとして実装する Lex Transformer [Sun22] である。[式 (3)](#equation-03) で説明したように、retention の導出は xPos と整合する。attention と比べると、retention は $\mathrm{softmax}$ を除去して再帰的な定式化を可能にし、推論に大きな利点をもたらす。

**S4。** [式 (2)](#equation-02) と異なり、$Q_n$ と $K_n$ が content-unaware であれば、定式化は S4 [Gu22] に縮退する。ここで $O=(Q K^\top,Q A K^\top,\ldots,Q A^{|x|-1}K^\top)*V$ である。

**Linear Attention。** 変種は通常、$\mathrm{softmax}$ 関数を置き換えるため、さまざまな kernel $\frac{\phi(q_i)\phi(k_j)}{\sum_{n=1}^{|x|}\phi(q_i)\phi(k_n)}$ を使用する。しかし、linear attention は位置情報を効果的に符号化することが難しく、モデルの性能が低下する。また、本研究は $\mathrm{softmax}$ の近似を目指すのではなく、系列モデリングを根本から再検討する。

**AFT/RWKV。** Attention Free Transformer（AFT）は dot-product attention を element-wise operation に簡略化し、$\mathrm{softmax}$ を key vector に移す。RWKV は AFT の位置埋め込みを指数減衰に置き換え、訓練と推論の両方でモデルを再帰的に実行する。これに対して retention は、系列情報を符号化するための高次元状態を保持し、表現力と性能を高める。

**xPos/RoPE。** Transformer 用に提案された相対位置埋め込み手法と比べると、[式 (3)](#equation-03) は xPos [Sun22] および RoPE [Su24] と類似した定式化を示す。

**Sub-LayerNorm。** [式 (8)](#equation-08) に示すように、retention layer は Sub-LayerNorm [Wan22l] を用いて出力を正規化する。multi-scale modeling によって head ごとの分散が異なるため、元の LayerNorm を GroupNorm に置き換える。

<span id="table-01"></span>

![さまざまな観点からのモデル比較](../../papers/retentive-network/table-01.png)

**表 1。** さまざまな観点からのモデル比較。RetNet は訓練の並列化、一定の推論コスト、長系列に対する線形のメモリ計算量、良好な性能を実現する。

<span id="section-3"></span>

## 3 実験

RetNet を評価するため、言語モデリング実験を行う。言語モデリング性能と下流タスクにおける zero-shot/few-shot learning など、さまざまな benchmark で提案アーキテクチャを評価する。さらに、訓練と推論について、速度、メモリ消費、レイテンシを比較する。

<span id="section-3-1"></span>

### 3.1 設定

**Parameter allocation。** 公平な比較のため、MSR と FFN の parameter を再配分する。簡潔さのため、ここでは $d$ で $d_{\mathrm{model}}$ を表す。Transformer の self-attention には約 $4d^2$ 個の parameter があり、$W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}$ である。また FFN には $8d^2$ 個の parameter があり、中間 dimension は $4d$ である。これに対して RetNet の retention には $8d^2$ 個の parameter があり、$W_Q,W_K\in\mathbb{R}^{d\times d},W_G,W_V\in\mathbb{R}^{d\times2d},W_O\in\mathbb{R}^{2d\times d}$ である。$V$ の head dimension は $Q,K$ の 2 倍であることに注意されたい。拡張された dimension は $W_O$ によって $d$ に射影される。parameter 数を Transformer と同じに保つため、RetNet の FFN 中間 dimension を $2d$ とする。また実験では、head dimension を $256$、すなわち query と key を $256$、value を $512$ とする。公平な比較のため、異なるモデルサイズでも $\bm{\gamma}$ を同一に保つ。ここでは[式 (8)](#equation-08) の既定値ではなく、$\bm{\gamma}=1-e^{\mathrm{linspace}(\log\frac{1}{32},\log\frac{1}{512},h)}\in\mathbb{R}^h$ とする。

<span id="table-02"></span>

![モデルサイズと言語モデリングのハイパーパラメータ](../../papers/retentive-network/table-02.png)

**表 2。** 言語モデリング実験におけるモデルサイズと学習ハイパーパラメータ。

**言語モデルの訓練。** [表 2](#table-02) に示すように、1.3B、2.7B、6.7B という異なるサイズの言語モデルを一から訓練する。訓練 corpus は The Pile [Gao20]、C4 [Dod21]、The Stack [Koc22] を選別して組み合わせたものである。系列の開始を示すため `<bos>` token を追加する [+2]。訓練 batch size は 4M token、最大長は 2048 である。100B token、すなわち 25k step でモデルを訓練する。AdamW [Los17] optimizer を用い、$\beta_1=0.9,\beta_2=0.98$、weight decay は $0.05$ とする。warmup step 数は 375 で、learning rate を線形に減衰させる。訓練の安定性を保証するため、parameter は DeepNet [Wan22c] に従って初期化する。実装は TorchScale [Ma22] に基づく。512 基の AMD MI200 GPU でモデルを訓練する。

[+2]: 系列の先頭に `<bos>` token を追加すると、訓練の安定性と性能が向上することを確認した。

<span id="section-3-2"></span>

### 3.2 Transformer との比較

<span id="figure-05"></span>

![RetNet と Transformer の perplexity スケーリング曲線](../../papers/retentive-network/figure-05.png)

**図 5。** モデルサイズの拡大に伴って perplexity が低下する。モデルサイズが 2B を超えると、RetNet が Transformer を上回る傾向を実験的に観察した。

**言語モデリング。** [図 5](#figure-05) に示すように、Transformer と RetNet に基づく言語モデルについて、validation set 上の perplexity を報告する。1.3B、2.7B、6.7B という 3 種類のモデルサイズでスケーリング曲線を示す。RetNet は Transformer と同等の結果を達成する。より重要な点として、結果は RetNet がサイズのスケーリングに有利であることを示す。性能に加えて、実験における RetNet の訓練は非常に安定している。実験結果は、RetNet が大規模言語モデルにおける Transformer の有力な競合であることを示す。経験的には、モデルサイズが 2B を超えると RetNet が Transformer を上回り始める。[第 6 節](#section-6) では、異なる context length での言語モデリング結果もまとめる。

<span id="table-03"></span>

![Transformer と RetNet の zero-shot および few-shot 性能](../../papers/retentive-network/table-03.png)

**表 3。** Transformer と RetNet による zero-shot および few-shot learning。モデルサイズは 6.7B。

**下流タスクにおける zero-shot および few-shot 評価。** 幅広い下流タスクでも言語モデルを比較する。6.7B モデルを用いて zero-shot と 4-shot learning を評価する。[表 3](#table-03) に示すように、dataset は HellaSwag（HS）[Zel19]、BoolQ [Cla19]、COPA [Wan19h]、PIQA [Bis20]、Winograd、Winogrande [Lev12]、StoryCloze（SC）[Mos17] を含む。accuracy は[図 5](#figure-05) に示した言語モデリング perplexity と整合している。RetNet は zero-shot と in-context learning の設定で Transformer と同等の性能を達成する。

<span id="section-3-3"></span>

### 3.3 訓練コスト

<span id="table-04"></span>

![Transformer と RetNet の訓練メモリとスループット](../../papers/retentive-network/table-04.png)

**表 4。** Transformer（Trm）、FlashAttention を用いる Transformer（Trm+FlashAttn）、RetNet の訓練コスト。メモリ消費と訓練スループット（word per second、wps）を報告する。

[表 4](#table-04) に示すように、訓練系列長を 8192 とし、Transformer と RetNet の訓練速度とメモリ消費を比較する。再計算と kernel fusion によって速度を高め、GPU memory IO を削減する FlashAttention [Dao22] とも比較する。これに対して、RetNet は素の PyTorch code で実装し、kernel fusion や FlashAttention に似た高速化は今後の課題とする。[式 (7)](#equation-07) で説明したチャンク単位の再帰 retention 表現を用いる。chunk size は $512$ とする。FlashAttention は A100 向けに高度に最適化されているため、8 基の Nvidia A100-80GB GPU で結果を評価する。6.7B と 13B のモデルでは tensor parallelism を有効にする。

実験結果は、訓練時の RetNet が Transformer よりメモリ効率とスループットに優れることを示す。FlashAttention と比べても、RetNet は速度とメモリコストの面で競争力を保つ。さらに、特定の kernel に依存しないため、RetNet は他の platform でも容易に効率よく訓練できる。たとえば、AMD MI200 cluster 上で RetNet モデルを妥当なスループットで訓練している。RetNet は、kernel fusion などの高度な実装によってコストをさらに削減できる可能性がある。

<span id="section-3-4"></span>

### 3.4 推論コスト

<span id="figure-06"></span>

![Transformer と RetNet の推論メモリ、スループット、レイテンシ](../../papers/retentive-network/figure-06.png)

**図 6。** モデルサイズ 6.7B の Transformer と RetNet の推論コスト。RetNet はメモリ消費、スループット、レイテンシの点で Transformer を上回る。

[図 6](#figure-06) に示すように、推論時の Transformer と RetNet のメモリコスト、スループット、レイテンシを比較する。Transformer は以前にデコードした token の KV cache を再利用する。RetNet は[式 (6)](#equation-06) で説明した再帰表現を用いる。実験では A100-80GB GPU 上で 6.7B モデルを評価する。[図 6](#figure-06) は、RetNet の推論コストが Transformer より優れることを示す。

**メモリ。** [図 6a](#figure-06) に示すように、Transformer のメモリコストは KV cache によって線形に増加する。これに対して RetNet のメモリ消費は長い系列でも一定であり、RetNet の保持に必要な GPU メモリははるかに少ない。RetNet の追加メモリ消費はほぼ無視でき（約 3%）、モデル weight が 97% を占める。

**スループット。** [図 6b](#figure-06) に示すように、Transformer のスループットはデコード長の増加に伴って低下する。これに対して RetNet は retention の再帰表現を利用することで、より高く、長さに依存しないデコードスループットを持つ。

**レイテンシ。** レイテンシはデプロイにおける重要な指標であり、user experience に大きく影響する。[図 6c](#figure-06) にデコードレイテンシを示す。実験結果は、batch size の増加が Transformer のレイテンシを大きくすることを示す。さらに、入力が長いほど Transformer のレイテンシは急速に増加する。許容可能なレイテンシにするには batch size を制限する必要があり、Transformer の推論スループット全体が低下する。対照的に、RetNet のデコードレイテンシは Transformer より優れ、異なる batch size と入力長でもほぼ一定である。

<span id="section-3-5"></span>

### 3.5 Transformer 変種との比較

<span id="table-05"></span>

![効率的な Transformer 変種の言語モデリング perplexity](../../papers/retentive-network/table-05.png)

**表 5。** 言語モデリングの perplexity。RetNet は in-domain evaluation set と複数の out-of-domain corpus の両方で、他のアーキテクチャを上回る。

Transformer に加えて、Linear Transformer [Kat20]、RWKV [Pen23b]、H3 [Dao22g]、Hyena [Pol23a] など、さまざまな効率的 Transformer 変種と RetNet を比較する。すべてのモデルは 200M parameter、16 layer、hidden dimension 1024 である。H3 の head dimension は 8 とする。RWKV では、公平な比較のため FFN layer を他のモデルと同一に保ち、self-attention layer を TimeMix module で置き換える。batch size 0.5M token でモデルを 10k step 訓練する。ほとんどの hyperparameter と訓練 corpus は[第 3.1 節](#section-3-1) と同じである。

[表 5](#table-05) は in-domain validation set と、Project Gutenberg 2019-2022（PG22）[Sun22]、QMSum [Zho21b]、GovReport [Hua21]、SummScreen [Che21g, Sha22a] などの out-of-domain corpus における perplexity を示す。全体として、RetNet は異なる dataset で従来手法を上回る。RetNet は in-domain corpus でより良い評価結果を得るだけでなく、複数の out-of-domain dataset でもより低い perplexity を達成する。大幅なコスト削減（[第 3.3 節](#section-3-3)、[第 3.4 節](#section-3-4)）に加え、良好な性能により RetNet は Transformer の有力な後継となる。

さらに、比較手法の訓練効率と推論効率を議論する。$d$ を hidden dimension、$n$ を系列長とする。訓練時の RWKV の token-mixing 計算量は $O(dn)$ であり、Hyena は Fast Fourier Transform により $O(dn\log n)$ となる。この 2 手法は element-wise operator を使用し、モデリング能力との trade-off によって訓練 FLOPS を削減する。これに対して、retention のチャンク単位の再帰表現は $O(dn(b+h))$ である。ここで $b$ は chunk size、$h$ は head dimension であり、通常は $b=512,h=256$ とする。モデルサイズ（すなわち $d$）または系列長が大きい場合、追加の $b+h$ の影響は無視できる。したがって、RetNet の訓練はモデリング性能を犠牲にすることなく効率的である。推論時には、比較した効率的なアーキテクチャのうち、Hyena は Transformer と同じく各ステップ $O(n)$ の計算量を持つが、他の手法は $O(1)$ デコードが可能である。

<span id="section-3-6"></span>

### 3.6 Ablation Study

<span id="table-06"></span>

![In-domain と out-of-domain corpus における ablation 結果](../../papers/retentive-network/table-06.png)

**表 6。** In-domain と out-of-domain corpus における ablation 結果。

RetNet のさまざまな設計選択を ablate し、[表 6](#table-06) に言語モデリング結果を示す。評価設定と指標は[第 3.5 節](#section-3-5) と同じである。

**アーキテクチャ。** [式 (8)](#equation-08) で説明した $\mathrm{swish}$ gate と $\mathrm{GroupNorm}$ を ablate する。[表 6](#table-06) は、この 2 つの component が最終性能を改善することを示す。第 1 に、gating module は非線形性を高め、モデル能力を改善するために不可欠である。gate を除去した後は Transformer と同じ parameter allocation を使用することに注意されたい。第 2 に、retention の group normalization は multi-head 出力の分散を均衡させ、訓練安定性と言語モデリング結果を改善する。

**Multi-scale decay。** [式 (8)](#equation-08) に示すように、retention head ごとに異なる $\bm{\gamma}$ を減衰率として使用する。ablation study では、$\gamma$ decay を除去する設定（「$-\ \gamma$ decay」）と、全 head に同じ減衰率を適用する設定（「$-$ multi-scale decay」）を検討する。具体的には、$\gamma$ decay の除去は $\gamma=1$ と等価である。第 2 の設定では、すべての head について $\gamma=127/128$ とする。[表 6](#table-06) は、減衰機構と複数の減衰率の使用がともに言語モデリング性能を改善することを示す。

**Head dimension。** [式 (1)](#equation-01) の再帰的な観点では、head dimension は hidden state のメモリ容量を表す。ablation study では、既定の head dimension を $256$ から $64$ に減らす。すなわち、query と key を $64$、value を $128$ とする。hidden dimension $d_{\mathrm{model}}$ は同じに保つため、head 数が増加する。[表 6](#table-06) の実験結果は、より大きい head dimension がより良い性能を達成することを示す。

<span id="section-4"></span>

## 4 結論

本研究では、並列、再帰、チャンク単位の再帰という複数の表現を可能にする、系列モデリングのための retentive network（RetNet）を提案する。Transformer と比べて、RetNet は推論効率（メモリ、速度、レイテンシ）が大幅に高く、訓練並列性が良好で、競争力のある性能を持つ。特に $O(1)$ 推論計算量がデプロイにもたらす利点を考えると、これらの長所により RetNet は大規模言語モデルにおける Transformer の理想的な後継となる。今後は RetNet のモデルサイズ [Chi22a] と訓練 step 数を拡大したい。さらに、retention は長期メモリを圧縮することで、structured prompting [Hao22a] と効率的に連携できる。また、RetNet を backbone architecture として multimodal large language model [Hao22, Hua23c, Pen23f] を訓練する。加えて、携帯電話などの各種 edge device に RetNet モデルをデプロイすることにも関心がある。

## 謝辞

有益な議論をしてくださった Jiayu Ding、Songlin Yang、MSRA System Group の同僚に感謝する。

<span id="section-5"></span>

## 5 ハイパーパラメータ

<span id="table-07"></span>

![RetNet モデルで使用したハイパーパラメータ](../../papers/retentive-network/table-07.png)

**表 7。** [第 3 節](#section-3) のモデルで使用したハイパーパラメータ。

<span id="section-6"></span>

## 6 異なる Context Length のグループ別結果

[表 8](#table-08) に示すように、異なる context length での言語モデリング結果を報告する。数値を比較可能にするため、2048 個の text chunk を評価データとして使用し、最後の 128 token についてのみ perplexity を計算する。実験結果は、異なる context length で RetNet が Transformer を上回ることを示す。さらに、RetNet は長い context を利用して、より良い結果を得られる。

<span id="table-08"></span>

![異なる context length における言語モデリング perplexity](../../papers/retentive-network/table-08.png)

**表 8。** 異なる context length における RetNet と Transformer の言語モデリング perplexity。RetNet が系列長全体で一貫した優位性を持つことを結果が示す。
