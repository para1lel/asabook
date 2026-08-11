---
title: 'Gated Delta Networks'
createTime: 2026/08/05 00:34:32
permalink: /ja/papers/gated-delta-networks/
---

> [Songlin Yang](https://sustcsonglin.github.io/) [+author-note]、[Jan Kautz](https://www.jankautz.com/)、[Ali Hatamizadeh](https://ahatamiz.github.io/)。arXiv 初回投稿: 2024 年 12 月 9 日、現行版: v3。[ICLR 2025](https://openreview.net/forum?id=r8H7xhYPwz) 採択。[Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)。[原論文 PDF](/paper/gated-delta-networks.pdf)。[DOI](https://doi.org/10.48550/arXiv.2412.06464)。[TeX ソース](https://export.arxiv.org/e-print/2412.06464v3)。正確な印刷レイアウトと参考文献については、原論文 PDF を正本とする。

## 要約

線形 Transformer は標準的な Transformer の効率的な代替手段として注目されているが、検索タスクや長文脈タスクでの性能には限界があった。この問題に対し、近年の研究は二つの異なる仕組みを検討している。一つは適応的にメモリを制御するゲーティング、もう一つはメモリを正確に変更する delta 更新則である。本研究では、両者が相補的であることに着目する。ゲーティングはメモリをすばやく消去でき、delta 則は狙った箇所を更新できる。この知見に基づいて gated delta 則を導入し、現代のハードウェア向けに最適化した並列訓練アルゴリズムを開発する。提案アーキテクチャ Gated DeltaNet は、言語モデリング、常識推論、文脈内検索、長さ外挿、長文脈理解を含む複数のベンチマークで、Mamba2 や DeltaNet などの既存モデルを一貫して上回る。さらに、Gated DeltaNet 層をスライディングウィンドウ注意または Mamba2 層と組み合わせたハイブリッドアーキテクチャにより、訓練効率とタスク性能の両方を改善する。
コード: [https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

## 1 はじめに

Transformer アーキテクチャは、効果的な注意機構によって大規模言語モデル (LLM) の能力を大きく押し上げ、幅広いタスクで優れた性能を示してきた。この機構は系列を正確にモデル化でき、訓練時には現代の GPU の並列処理能力も活用できる。一方、自己注意の計算量は系列長に対して二次的に増えるため、訓練と推論の双方で大きな計算負荷が生じる。

この問題を緩和するため、線形 Transformer [ICMLa20] などの代替手法が研究されている。線形 Transformer は従来の softmax 注意をカーネル化した内積ベースの線形注意に置き換え、行列値状態を持つ線形 RNN として捉え直すことで、推論時のメモリ使用量を大幅に減らす。初期の線形 Transformer は言語モデリングで標準 Transformer に及ばなかったが、GLA [PMLRa24] や Mamba2 [Daoa24] に見られる LSTM 型のデータ依存ゲートを導入した近年の改良は、有望な成果を示している。ただし、長い系列の情報管理、とりわけ従来の Transformer が優位な文脈内検索では、依然として課題が残る [Arora23, Arora24, Jelass24, Wen24, Aky24]。

これは不思議ではない。線形 Transformer は、テンソル積表現 [Smolen90] を思わせる、外積ベースのキー・バリュー連想メモリを実装していると解釈できる。しかし、保存可能な直交キー・バリュー対の数はモデルの次元によって*制限*される。系列長がこの次元を超えると「メモリ衝突」が避けられず、正確な検索が難しくなる [ICMLa21]。

Mamba2 は単純なゲート付き更新則 ${\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}$ を導入し、各時刻ですべてのキー・バリュー対応を動的な比率 $\alpha_t\in(0,1)$ によって一様に減衰させる。しかし、この方法はキー・バリュー対応ごとの重要度の違いを考慮しないため、メモリを効率よく使えない場合がある。特定の対応だけを忘れたい場合でも、すべての対応が同じように忘却されるため、更新の対象を絞れず非効率である。

これに対し、delta 則 [Widrow60] を用いる線形 Transformer、すなわち DeltaNet [ICMLa21, NeurIP24] は、古いキー・バリュー対を到着した対で順次 (soft に) 置き換え、メモリを選択的に更新する。合成データによる文脈内検索ベンチマークでは高い性能を示す。一方、この処理で一度に変更できるのは一つのキー・バリュー対だけであり、古い情報や無関係な情報をすばやく消去できない。とくに、それまでのデータを消す必要がある文脈切り替え時に問題となる。そのため DeltaNet の実世界タスクでの性能は中程度にとどまっており [NeurIP24]、堅牢なメモリ消去機構の欠如が一因と考えられる。

ゲート付き更新則と delta 則はメモリ管理において相補的な利点を持つ。そこで本研究は、両者を組み合わせた単純で直感的な仕組みである*gated delta 則*を提案する。この統一的な更新則では、$\alpha_t\rightarrow 0$ とすればメモリをすばやく消去できる。一方、$\alpha_t\rightarrow 1$ とすれば、ほかの情報に影響を与えず特定の内容だけを選択的に更新できる。後者は純粋な delta 則への切り替えに相当する。

残る課題は、gated delta 則をハードウェア効率よく実装することである。[NeurIP24] は WY 表現 [Compua85] を用いて delta 則の計算を並列化した。本研究はこの効率的なアルゴリズムを拡張し、ゲーティング項を組み込む。この拡張はチャンク単位並列処理 [Huaa22, Suna23, PMLRa24] の利点を保ち、ハードウェア効率のよい訓練を可能にする。

得られたアーキテクチャ Gated DeltaNet は、言語モデリング、常識推論、文脈内検索、長さ外挿、長文脈理解を含む包括的なベンチマークで、Mamba2 と DeltaNet の双方を一貫して上回る。さらに、Gated DeltaNet 層をスライディングウィンドウ注意または Mamba2 層と組み合わせたハイブリッドアーキテクチャを構築し、訓練効率とモデル性能をいっそう改善する。

## 2 予備知識

### 2.1 Mamba2：減衰を伴う線形注意

正規化とクエリ／キーの活性化を除くと、線形 Transformer [ICMLa20] は次の線形漸化式として表せる。

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}+{\bm{v}}_t{\bm{k}}_t^\top\in\mathbb{R}^{d_v\times d_k},\qquad\qquad {\bm{o}}_t={\mathbf{S}}_t{\bm{q}}_t\in\mathbb{R}^{d_v}
$$

ここで $d_k$ と $d_v$ は、それぞれクエリ／キーと値の（ヘッド）次元である。漸化式を展開すると、次のベクトル形式（左）と行列形式（右）が得られる。

$$
{\bm{o}}_t=\sum_{i=1}^t({\bm{v}}_i{\bm{k}}_i^\top){\bm{q}}_t=\sum_{i=1}^t{\bm{v}}_i({\bm{k}}_i^\top{\bm{q}}_t)\in\mathbb{R}^{d_v},\qquad {\mathbf{O}}=({\mathbf{Q}}{\mathbf{K}}^\top\odot{\mathbf{M}}){\mathbf{V}}\in\mathbb{R}^{L\times d_v}
$$

ここで $L$ は系列長であり、${\mathbf{M}}\in\mathbb{R}^{L\times L}$ は、$i<j$ のとき ${\mathbf{M}}_{ij}=0$、それ以外では $1$ となる因果マスクである。

ただし、この素朴な線形注意は言語モデリングで Transformer に大きく劣る。そのため、過去の情報を忘却する減衰項を加えるのが一般的である。Mamba2 [Daoa24] を例に取ると、具体的なパラメータ化を除いて次の線形漸化式で表せる。

$$
{\mathbf{S}}_t={\color[\mathrm{rgb}]{0,0,1}\alpha_t}{\mathbf{S}}_{t-1}+{\bm{v}}_t{\bm{k}}_t^\top,\qquad {\bm{o}}_t={\mathbf{S}}_t{\bm{q}}_t
$$

ここで ${\color[\mathrm{rgb}]{0,0,1}\alpha_t\in(0,1)}$ は $t$ に応じて変化する、データ依存のスカラー減衰項である。累積減衰積 ${\color[\mathrm{rgb}]{0,0,1}\gamma_j=\prod_{i=1}^j\alpha_i}$ を定義して漸化式を展開すると、ベクトル形式（左）と行列並列形式（右）が得られる。

$$
{\bm{o}}_t=\sum_{i=1}^t\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{v}}_i{\bm{k}}_i^\top\right){\bm{q}}_t=\sum_{i=1}^t{\bm{v}}_i\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{k}}_i^\top{\bm{q}}_t\right),\qquad {\mathbf{O}}=\left(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma}\right){\mathbf{V}}
$$

${\color[\mathrm{rgb}]{0,0,1}\Gamma\in\mathbb{R}^{L\times L}}$ は減衰を考慮した因果マスクであり、$i\geq j$ のとき ${\color[\mathrm{rgb}]{0,0,1}\Gamma_{ij}=\frac{\gamma_i}{\gamma_j}}$、それ以外では ${\color[\mathrm{rgb}]{0,0,1}\Gamma_{ij}=0}$ である。この並列形式と再帰形式の等価性は、[Daoa24] では状態空間双対性（SSD）と呼ばれる。同様の再帰構造は Gated RFA [ICLRa21]、xLSTM [Beck24]、Gated RetNet [Sunb24] にも現れる。$\gamma_t$ がデータに依存しない場合は RetNet [Suna23] と Lightning-Attention [Lightn24] に帰着する。$\gamma_t$ をスカラーから行列へ拡張しても、外積構造でパラメータ化すれば効率的な訓練アルゴリズムを構成できる [PMLRa24, Peng24, Qin24a, Gated24, Systee24, Reprea25, Refini25]。

**チャンク単位の訓練。** 再帰形式と並列形式は、どちらも効率的な訓練には適していない [Huaa22, PMLRa24]。そこで、ハードウェア効率のよい線形時間訓練のためにチャンク並列形式 [Huaa22, Suna23] を用いる。入力と出力を長さ $C$ のチャンクに分け、各チャンクの出力を前チャンクの最終状態と現在チャンクのクエリ／キー／値ブロックから計算する。[Suna23, PMLRa24, NeurIP24] の記法に従い、クエリブロック ${\bm{q}}$ を例に取る。${\mathbf{Q}}_{[t]}:={\bm{q}}_{tC+1:(t+1)C+1}$ をチャンク $t$ のクエリブロック、${\bm{q}}_{[t]}^r:={\bm{q}}_{tC+r}$ をその $r$ 番目のクエリとする。初期状態は ${\mathbf{S}}_{[t]}:={\mathbf{S}}_{[t]}^0={\mathbf{S}}_{[t-1]}^C$ である。漸化式を部分的に展開すると、

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}+\sum_{i=1}^r{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad {\bm{o}}_{[t]}^r={\mathbf{S}}_{[t]}^r{\bm{q}}_{[t]}^r={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^r+\sum_{i=1}^r{\bm{v}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^r\right)\in\mathbb{R}^{d_v}
$$

等価な行列形式は次のとおりである。

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}{\mathbf{K}}_{[t]}^\top\in\mathbb{R}^{d_v\times d_k},\qquad {\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
$$

ここで ${\mathbf{M}}\in\mathbb{R}^{C\times C}$ は因果マスクである。上式は行列乗算を中心に構成されるため、Tensor Core による最適化が可能である。このチャンクアルゴリズムは、減衰を伴う線形注意にも容易に拡張できる。

<span id="equation-01"></span>

$$
{\mathbf{S}}_{[t+1]}={\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{S}}_{[t]}}}+{\mathbf{V}}_{[t]}^\top{\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{K}}_{[t]}}}\in\mathbb{R}^{d_v\times d_k},\qquad {\mathbf{O}}_{[t]}={\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\mathbf{Q}}_{[t]}}}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
\tag{1}
$$

ここで ${\color[\mathrm{rgb}]{0,0,1}(\Gamma_{[t]})_{ij}=\frac{\gamma_{[t]}^i}{\gamma_{[t]}^j},\ \gamma_{[t]}^j=\prod_{j=tC+1}^{tC+j}\alpha_j}$ である。[+1] 左矢印 ($\overleftarrow{\cdot}$) と右矢印 ($\overrightarrow{\cdot}$) は、それぞれ変数を各チャンクの先頭位置と末尾位置まで減衰させることを表す。

<span id="equation-02"></span>

$$
{\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\bm{q}}_{[t]}^r}}={\color[\mathrm{rgb}]{0,0,1}\gamma_{[t]}^r}{\bm{q}}_{[t]}^r\qquad\mathrm{decaying\ each\ vector\ to\ the\ first\ position\ of\ chunk}\ t
$$

$$
{\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\bm{k}}_{[t]}^r}}={\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}}{\bm{k}}_{[t]}^r\qquad\mathrm{decaying\ each\ vector\ to\ the\ last\ position\ of\ chunk}\ t
$$

$$
{\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{S}}_{[t]}}}={\color[\mathrm{rgb}]{0,0,1}\gamma_{[t]}^C}{\mathbf{S}}_{[t]}\qquad\mathrm{decaying\ the\ state\ matrix\ over\ the\ entire\ chunk}\ t
\tag{2}
$$

ほかの変数（例えば ${\color[\mathrm{rgb}]{0,0,1}\overrightarrow{\bm{v}}}$）も同様である。Mamba2 の SSD 分解アルゴリズムは、このチャンクアルゴリズムとほぼ等価である。[PMLRa24] はさらに、細粒度の減衰を組み込む一般化チャンクアルゴリズムを提案した。

### 2.2 Delta Networks：Delta 則を用いた線形注意

delta 更新則 [Widrow60, ICMLa21] は、現在の入力キー ${\bm{k}}_t$ に対応する値 ${\bm{v}}_t^{\mathrm{old}}$ を*動的に*消去し、新しい値 ${\bm{v}}_t^{\mathrm{new}}$ を書き込む。新しい値は現在の入力値と古い値の線形結合であり、「書き込み強度」$\beta_t\in(0,1)$ がその比率を決める。[+2]

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}-\underbrace{\left({\mathbf{S}}_{t-1}{\bm{k}}_t\right)}_{{\bm{v}}_t^{\mathrm{old}}}{\bm{k}}_t^\top+\underbrace{\left(\beta_t{\bm{v}}_t+(1-\beta_t){\mathbf{S}}_{t-1}{\bm{k}}_t\right)}_{{\bm{v}}_t^{\mathrm{new}}}{\bm{k}}_t^\top={\mathbf{S}}_{t-1}\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
$$

上式のとおり、DeltaNet は一般化 Householder 遷移行列 $({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top)$ を持つ一次線形漸化式である。連想記憶と言語モデリングの性能は高い [ICMLa21] が、計算効率が低かったため、[NeurIP24] が以下のハードウェア効率的なチャンク訓練法を導入するまで広く注目されなかった。

**チャンク並列形式。** 漸化式を部分的に展開すると、

<span id="equation-03"></span>

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^r{\mathbf{I}}-\beta_{[t]}^i{\bm{k}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\right)}_{:={\mathbf{P}}_{[t]}^r}+\underbrace{\sum_{i=1}^r\left(\beta_{[t]}^i{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^r\left({\mathbf{I}}-\beta_{[t]}^j{\bm{k}}_{[t]}^j{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{[t]}^r}
\tag{3}
$$

${\mathbf{P}}_{[t]}^j$ は一般化 Householder 行列の累積積を含み、古典的な WY 表現 [Compua85] で最適化できる。

<span id="equation-04"></span>

$$
{\mathbf{P}}_{[t]}^r={\mathbf{I}}-\sum_{i=1}^r{\bm{w}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_k\times d_k},\qquad {\bm{w}}_{[t]}^r=\beta_{[t]}^r\left({\bm{k}}_{[t]}^r-\sum_{i=1}^{r-1}{\bm{w}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_k}
\tag{4}
$$

同様に、${\mathbf{H}}_{[t]}^r$ は次のように表せる。

<span id="equation-05"></span>

$$
{\mathbf{H}}_{[t]}^r=\sum_{i=1}^r{\bm{u}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad {\bm{u}}_{[t]}^r=\beta_{[t]}^r\left({\bm{v}}_{[t]}^r-\sum_{i=1}^{r-1}{\bm{u}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_v}
\tag{5}
$$

行列形式では、 ${\mathbf{P}}_{[t]}={\mathbf{I}}-{\mathbf{W}}_{[t]}^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_k\times d_k}$, ${\mathbf{H}}_{[t]}={\mathbf{U}}_{[t]}^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_v\times d_k}$。UT 変換 [Joffra06] を用いると、 ${\mathbf{W}}$ と ${\mathbf{U}}$ はさらに次の行列形式になる。

<span id="equation-06"></span>

$$
{\mathbf{T}}_{[t]}=\left[{\mathbf{I}}+\mathrm{strictLower}\left(\mathrm{diag}(\beta_{[t]}){\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^\top\right)\right]^{-1}\mathrm{diag}(\beta_{[t]})\in\mathbb{R}^{C\times C}
\tag{6}
$$

<span id="equation-07"></span>

$$
{\mathbf{W}}_{[t]}={\mathbf{T}}_{[t]}{\mathbf{K}}_{[t]}\in\mathbb{R}^{C\times d_k},\qquad {\mathbf{U}}_{[t]}={\mathbf{T}}_{[t]}{\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
\tag{7}
$$

これらを[式 3](#equation-03)へ代入すると、行列乗算を利用して Tensor Core で最適化できる DeltaNet のチャンクアルゴリズムが得られる。

<span id="equation-08"></span>

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}{\mathbf{P}}_{[t]}+{\mathbf{H}}_{[t]}={\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^\top\right)^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_v\times d_k}
\tag{8}
$$

<span id="equation-09"></span>

$$
{\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right)\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^\top\right)\in\mathbb{R}^{C\times d_v}
\tag{9}
$$

## 3 ゲーテッドデルタネットワーク

### 3.1 定式化：ゲーテッド Delta 則

提案するゲーテッド delta 則は単純だが効果的である。

<span id="equation-10"></span>

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}\left({\color[\mathrm{rgb}]{0,0,1}\alpha_t}\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
\tag{10}
$$

データ依存のゲーティング項 ${\color[\mathrm{rgb}]{0,0,1}\alpha_t}\in(0,1)$ が状態の減衰を制御する。この形式では、ゲーティング項がメモリを適応的に管理し、delta 更新構造がキー・バリュー対応を学習する。

[Liua24] のオンライン学習フレームワークからゲーテッド delta 則を解析する。この枠組みでは、再帰状態の更新はオンライン学習問題の*閉形式*解として得られる（[表 1](#table-01)）。近年の線形 RNN は、状態が以前の値から離れすぎないようオンライン学習目的に正則化項を加え、メモリを保持する。しかし状態が情報で飽和すると、複数の情報が重ねて符号化され、正確な検索が難しくなる。Mamba2 と Gated DeltaNet は適応的な係数 $\alpha_t$ で正則化を緩め、${\mathbf{S}}_t$ と ${\mathbf{S}}_{t-1}$ の差を制御する。選択的な忘却によってメモリを動的に管理し、無関係な情報を除去できる（[§ 3.2](#32-ケーススタディsingle-needle-in-a-haystack-s-niah)）。

一方、Linear Attention（LA）と Mamba2 は単純な負の内積損失 $-\langle{\mathbf{S}}_t{\bm{k}}_t,{\bm{v}}_t\rangle$ を使う。Longhorn [Liua24] は、キー・バリュー対応をよりよくモデル化するため、表現力の高いオンライン回帰目的 $\|{\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t\|^2$ を用いる。Longhorn の更新則は delta 更新則によく似ており、[+3]（ゲーテッド）delta 則がインコンテキスト連想記憶で Mamba2 より優れる可能性を示す。

高速重みプログラミング [ICMLc22]、テスト時訓練 [Suna24]、テスト時回帰 [Testti25] の観点では、隠れ状態 ${\mathbf{S}}$ を（高速）重み行列と解釈できる。delta 則は*テスト時*確率的勾配降下法（SGD）でオンライン回帰目的 $\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\|{\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t\|^2$ を最適化する。

$$
{\mathbf{S}}_{t+1}={\mathbf{S}}_t-\beta_t\nabla\mathcal{L}({\mathbf{S}}_t)={\mathbf{S}}_t-\beta_t({\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t){\bm{k}}_t^\top={\mathbf{S}}_t\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
$$

ここで $\beta_t$ は（適応的な）学習率である。この見方では、ゲーテッド delta 則は SGD 更新に適応的な重み減衰項 $\alpha_t$ を加えたものと解釈できる。重み減衰は深層学習で広く使われる [Hertz91, Andriu23]。同時期の Titans [Learni24] も、RNN のテスト時 SGD 更新に重み減衰を組み込む有効性を示した。

<span id="table-01"></span>

![原論文の表 1](../../papers/gated-delta-networks/table-01.png)

**表 1.** [Liua24] の枠組みに基づく線形 RNN と対応するオンライン学習目的の比較。簡単のため、Longhorn のベクトル値 ${\bm{\beta}}$ をスカラー $\beta$ に置き換えた。

<span id="table-02"></span>

![原論文の表 2](../../papers/gated-delta-networks/table-02.png)

**表 2.** 1.3B モデルの S-NIAH ベンチマークにおけるゼロショット性能（設定は[§ 4](#4-実験)）。

### 3.2 ケーススタディ：Single Needle in a Haystack（S-NIAH）

delta 則とゲーテッド則の相補性を調べるため、RULER [Hsieh24] の Single Needle-In-A-Haystack（S-NIAH）でケーススタディを行う。キー・バリュー対が文脈という干し草の中の針となり、モデルはキーを与えられたときに値を想起しなければならない。[表 2](#table-02) から三つの点が分かる。

**減衰は記憶保持を損なう。** 最も単純な S-NIAH-1 は反復する合成文脈を使い、記憶すべき情報が少ないため長期保持を測る。DeltaNet はすべての系列長でほぼ完全な性能を示す。Mamba2 は履歴を急速に減衰させるため 2K を超えると大きく低下するが、Gated DeltaNet は delta 則により低下が小さい。

**ゲーティングはフィルタリングを助ける。** 実世界の文章を文脈に使う S-NIAH-2/3 では、モデルが関連しうる情報をすべて保存するため、効率的なメモリ管理が問われる。状態サイズが固定されていると、消去できない情報が重なって区別不能になり、メモリ衝突が起きる。DeltaNet は長い系列で大きく低下する。Mamba2 と Gated DeltaNet はゲートで不要な情報を除くため、性能を保ちやすい。

**Delta 則は記憶を助ける。** S-NIAH-3 では値を数字から UUID に変え、複雑なパターンの記憶を測る。Mamba2 は急速に低下するが Gated DeltaNet は良好であり、delta 則の記憶能力が高いことを確認できる。

### 3.3 アルゴリズム: ハードウェア効率のよいチャンク単位訓練

本節では、Gated DeltaNet を訓練するためのハードウェア効率のよいチャンク単位アルゴリズムを導出する。[式 10](#equation-10) の漸化式を部分的に展開すると、

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^r{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^i}\left({\mathbf{I}}-\beta_{[t]}^i{\bm{k}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\right)\right)}_{:={\mathbf{F}}_{[t]}^r}+\underbrace{\sum_{i=1}^r\left(\beta_{[t]}^i{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^r{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^j}\left({\mathbf{I}}-\beta_{[t]}^j{\bm{k}}_{[t]}^j{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{G}}_{[t]}^r}
$$

を得る。${\mathbf{F}}_{[t]}^r={\color[\mathrm{rgb}]{0,0,1}\gamma_{[t]}^r}{\mathbf{P}}_{[t]}^r={\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\mathbf{P}}_{[t]}^r}}$ であることは容易に分かる。${\mathbf{G}}_{[t]}^r$ については、[式 5](#equation-05) を次のように修正する。

$$
{\mathbf{G}}_{[t]}^r=\sum_{i=1}^r{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}}\widetilde{\bm{u}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad \widetilde{\bm{u}}_{[t]}^r=\beta_{[t]}^r\left({\bm{v}}_{[t]}^r-\sum_{i=1}^{r-1}\widetilde{\bm{u}}_{[t]}^i\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}}{\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_v}
$$

(証明は[付録 A](#付録-a-gated-delta-則の拡張-wy-表現)を参照)。UT 変換により、行列形式は次のようになる。

$$
\widetilde{\mathbf{U}}_{[t]}=\left[{\mathbf{I}}+\mathrm{strictLower}\left(\mathrm{diag}(\beta_{[t]})\left({\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\odot{\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^\top\right)\right)\right]^{-1}\mathrm{diag}(\beta_{[t]}){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
$$

Mamba2 が線形注意を拡張した方法 ([式 1](#equation-01)) と同様に、DeltaNet のチャンク単位アルゴリズム ([式 8](#equation-08)-[9](#equation-09)) を次のように修正すれば、Gated DeltaNet をハードウェア効率よく訓練できる。

$$
{\mathbf{S}}_{[t+1]}={\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{S}}_{[t]}}}+\left(\widetilde{\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\mathbf{W}}_{[t]}}}{\mathbf{S}}_{[t]}^\top\right)^\top{\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{K}}_{[t]}}}\in\mathbb{R}^{d_v\times d_k}
$$

$$
{\mathbf{O}}_{[t]}={\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\mathbf{Q}}_{[t]}}}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right)\left(\widetilde{\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\mathbf{W}}_{[t]}}}{\mathbf{S}}_{[t]}^\top\right)\in\mathbb{R}^{C\times d_v}
$$

ここで ${\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\bm{q}}_{[t]}^r}=\gamma_{[t]}^r}{\bm{q}}_{[t]}^r$、${\color[\mathrm{rgb}]{0,0,1}\overleftarrow{{\bm{w}}_{[t]}^r}=\gamma_{[t]}^r}{\bm{w}}_{[t]}^r$、${\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\bm{k}}_{[t]}^r}=\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}}}{\bm{k}}_{[t]}^r$、${\color[\mathrm{rgb}]{0,0,1}\overrightarrow{{\mathbf{S}}_{[t]}}=\gamma_{[t]}^C}{\mathbf{S}}_{[t]}$ であり、[式 2](#equation-02) の定義と同じである。

### 3.4 Gated Delta Networks とハイブリッドモデル

**Token mixer ブロック。** 基本の Gated DeltaNet は Llama のマクロアーキテクチャに従い、token mixer 層と SwiGLU MLP 層を積み重ねるが、自己注意を gated delta 則による token mixing に置き換える。[図 1](#figure-01) (右) にブロック構造を示す。gated delta 則 ([式 10](#equation-10)) では、クエリ、キー、値 $\{{\bm{q}},{\bm{k}},{\bm{v}}\}$ を線形射影、短い畳み込み、SiLU によって生成し、訓練を安定させるため ${\bm{q}},{\bm{k}}$ に L2 正規化を適用する。$\alpha,\beta$ には線形射影だけを用いる。[+4] [Suna23] に従い、出力射影の前に出力を正規化してゲート処理する。

<span id="figure-01"></span>

![図 1](../../papers/gated-delta-networks/figure-01.png)

**図 1.** Gated DeltaNet モデルの (ハイブリッド) アーキテクチャとブロック構造。Gated DeltaNet-H1 は Gated DeltaNet + SWA、H2 は Mamba2 + Gated DeltaNet + SWA の配置を用いる。ブロック内では、クエリ／キーパスが線形射影、短い畳み込み、SiLU、L2 正規化、値パスが線形射影、短い畳み込み、SiLU からなる。alpha/beta は線形射影を用い、出力ゲートは SiLU を伴う線形射影を適用する。

**ハイブリッドモデル。** 線形 Transformer は局所的なシフトや比較のモデル化に限界があり、状態サイズが固定されるため検索タスクも難しい [Arora24]。Griffin [De24] や Samba [Ren24] など近年のハイブリッドアーキテクチャに従い、線形再帰層とスライディングウィンドウ注意 (SWA) を組み合わせた GatedDeltaNet-H1 を構築する。また、Mamba2、GatedDeltaNet、SWA を順に積み重ねた GatedDeltaNet-H2 も構築する。

## 4 実験

**設定。** 純粋な Transformer、RNN ベースの手法、ハイブリッドアーキテクチャを含む近年の代表的なモデルを幅広く比較する。ベースラインは RetNet [Suna23]、HGRN2 [Qin24a]、Mamba [Daoc23]、Mamba2 [Daob24]、Samba [Ren24]、DeltaNet [NeurIP24] である。公平な比較のため、すべてのモデルを 1.3B パラメータとし、FineWeb-Edu [Penedo24] から抽出した 100B token を用いて同じ条件で訓練する。AdamW optimizer を使用し、最大学習率を 4e-4、weight decay を 0.1、gradient clipping を 1.0 とする。学習率には 1B token の warm-up を伴う cosine annealing schedule を用い、batch size は 0.5M token とする。すべてのモデルで、語彙数 32,000 の Llama2 tokenizer を使用する。系列モデリングの訓練長は 4K token とし、Samba と提案するハイブリッドモデルの sliding window size は 2K とする。評価設定は[§ B.1](#b1-評価)、ablation study は[§ B.2](#b2-アブレーション研究)を参照。

<span id="table-03"></span>

![原論文の表 3](../../papers/gated-delta-networks/table-03.png)

**表 3.** 言語モデリングと zero-shot 常識推論の性能比較。

**常識推論。** [表 3](#table-03) に、400M および 1.3B パラメータのモデルについて、言語モデリングの perplexity と常識推論ベンチマークの **zero-shot** accuracy を示す。どちらの規模でも、Gated DeltaNet は RetNet、HGRN2、Mamba、Mamba2、DeltaNet を含むほかの線形モデルを一貫して上回る。予想どおり、ハイブリッド版はさらに高い性能を示す。

<span id="table-04"></span>

![原論文の表 4](../../papers/gated-delta-networks/table-04.png)

**表 4.** 入力を 2K token に切り詰めた実世界の recall 型検索タスクにおける accuracy。SQD: SQUADE。TQA: Trivial QA。

**実世界データにおける文脈内検索。** [表 4](#table-04) に [Aroraa24] が使用した実世界の recall-intensive task の結果を示す。予想どおり、線形再帰モデルは Transformer に比べて大きく劣る。一方、線形再帰と注意を組み合わせたハイブリッドモデルは、検索タスクで純粋な注意モデルを上回る。

純粋な再帰モデルを見ると、DeltaNet は合成文脈内検索タスクで優れているにもかかわらず [NeurIP24]、実世界の検索では Mamba2 に及ばない。この結果は S-NIAH-2/3 での観察 ([表 2](#table-02)) と一致する。Gated DeltaNet は gated delta 則によって DeltaNet と Mamba2 の両方を上回るが、改善幅は[表 2](#table-02)より小さい。これは、instruction alignment を行っていない小規模言語モデルでは反復誤りが生じやすく、それが各タスクの主な誤りとなるためだと考えられる ([Aroraa24] 付録 E を参照)。この問題は更新則の選択にほぼ依存しないため、モデル間の性能差は[表 2](#table-02)ほど大きくならない。

<span id="figure-02"></span>

![図 2](../../papers/gated-delta-networks/figure-02.png)

**図 2.** 六つの長系列ベンチマークにおける長さ外挿。

**長系列における長さ外挿。** [図 2](#figure-02) に示すとおり、六つの長文脈ベンチマークで、最大 20K token の系列へ外挿する能力を評価する。RNN モデルの中では、Gated DeltaNet がタスク全体で最も低い perplexity を達成する。長さ外挿の結果にはばらつきがあるものの、Gated DeltaNet は比較的安定しており、メモリ管理が優れていることを示唆する。ハイブリッドモデルは、注意によって局所文脈をモデル化し、再帰部分のメモリ管理負担を軽くすることで、さらに性能を改善する。今後は、より長い系列に対する能力を検証する。

**長文脈理解。** [表 5](#table-05) に LongBench [Bai23] での結果を示す。再帰モデルの中では Gated DeltaNet が一貫して優れており、とくに single-document QA、few-shot 文脈内学習、コードタスクで大きな優位を示す。これらはそれぞれ、検索、文脈内学習、状態追跡の能力を表す。

<span id="table-05"></span>

![原論文の表 5](../../papers/gated-delta-networks/table-05.png)

**表 5.** LongBench [Bai23] の 14 タスクにおける accuracy。順に Narrative QA、QasperQA、MultiField QA、HotpotQA、2WikiMulti QA、Musique、GovReport、QMSum、MultiNews、TRec、Trivia QA、SamSum、LCC、RepoBench-P。

<span id="figure-03"></span>

![図 3](../../papers/gated-delta-networks/figure-03.png)

**図 3.** 単一の H100 GPU における 1.3B モデルの訓練 throughput 比較。

**Throughput 比較。** [図 3](#figure-03) に各モデルの訓練 throughput を示す。提案する gated delta 則が元の delta 則に加える overhead はわずかであり、Gated DeltaNet は DeltaNet とほぼ同じ throughput を達成する。両者は遷移行列の表現力が高いため、Mamba2 より毎秒 2-3K token ほど遅い。

2K context window では、高度に最適化された Flash-Attention-2 kernel [Daob23] により Transformer++ が最も高い性能を示す。このため、window size 2K の SWA とほかの token mixer を組み合わせたハイブリッド手法は、単独の mixer より高い throughput を示す。Samba は Mamba を、Gated DeltaNet-H1 と -H2 は Gated DeltaNet を上回る。Gated DeltaNet-H1 は、短い系列を含むすべての系列長で良好な訓練 throughput を維持する。

## 5 関連研究

**ゲート付き線形 RNN。** 大規模な線形再帰言語モデルは、訓練と推論の効率が高いため大きな注目を集めている。線形 RNN は、S4 [ICLRb22]、S5 [ICLRb23]、LRU [ICMLa23]、RWKV4/5 [EMNLP23]、RetNet [Suna23] に代表されるデータ非依存の減衰機構から、HGRN1/2 [Qin24a, Qina23]、Mamba1/2 [Daoc23, Daoa24]、RWKV6 [Peng24]、GSA [Gated24] など近年のアーキテクチャが採用するデータ依存の減衰機構へと急速に発展してきた。この変化は、ゲーティング／忘却機構 (Mamba では selective mechanism と呼ぶ) の利点が実証されてきたことによる。この古典的な概念はゲート付き RNN の研究 [Gersa00] に端を発し、その重要性は繰り返し確認されている [Greff15, Lasenb18, Qin24a, Qina23, Daoc23]。

現代の忘却ゲートは、LSTM などの従来設計と異なり、以前の隠れ状態への依存を除き、入力データだけに依存する。この変更により、系列長方向の効率的な並列処理が可能になる [Mar18, Qina23]。忘却ゲートを持たないことは DeltaNet の明確な弱点であり、本研究のゲート付き拡張はこの不足を自然かつ効果的で、ハードウェア効率のよい方法で補う。同時期の研究 RWKV-7 [+5] も同様の発想を用いるが、対角行列と低ランク行列の和による、より柔軟な遷移を採用する。すなわち ${\mathbf{S}}_t={\mathbf{S}}_{t-1}(\mathrm{diag}({\mathbf{d}}_t)-{\mathbf{a}}_t{\mathbf{b}}_t^\top)+{\bm{v}}_t{\bm{k}}_t^\top$、ただし ${\mathbf{d}}_t,{\mathbf{a}}_t,{\mathbf{b}}_t\in\mathbb{R}^{d_k}$ である。Flash Linear Attention [Yan24] の実装のように、チャンク単位アルゴリズムもこの場合に合わせて同様に修正できる。[+6]

**Delta 則。** Delta 学習則は Hebbian 学習より大きなメモリ容量を持つ [Gardne88, Kak89]。線形 Transformer が Hebbian 学習に似た規則を使うのに対し、DeltaNet はこの利点を活用する。メモリ容量の優位性は、合成文脈内学習タスクだけでなく、言語モデリング [Irie21, NeurIP24]、強化学習 [ICMLd22]、画像生成 [ICLRa23] にも及ぶ。[NeurIP24] は delta 則の計算を並列化し、DeltaNet のデータ依存な identity-plus-low-rank 構造 $({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top)$ が、Mamba2 のデータ依存な対角行列 $(\alpha_t{\mathbf{I}})$ より柔軟であることを示した。この構造上の利点は、正規言語の認識 [Bethar24, Grazzi24] や $\mathrm{TC}^0$ complexity class を超える状態追跡 [Merril24] など、コード生成と推論に重要な複雑な推論を可能にしうる。

こうした大きな利点がある一方、delta 則には理論上の限界があり [Bali23]、実世界のデータセットでは中程度の性能にとどまる [NeurIP24]。非線形漸化式で表現力を高める従来の試み [Irie21, ICMLd22] は一部の限界を克服したが、訓練時の並列性を犠牲にし、性能と効率の trade-off を生んだ。近年の研究は、並列性を損なわず状態追跡を改善するため、負の固有値を用いる方法 [Grazzi24] や、高ランク変換を可能にする複数の Householder 遷移行列の積 [Increa25] を提案している。これらの方法は Gated DeltaNet にもそのまま適用できる。

(オンライン) 学習目的の観点からは、別の定式化によって表現力をさらに拡張できる。TTT [Suna24] と Titans [Learni24] は非線形回帰 $\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\|f_{{\mathbf{S}}_t}({\bm{k}}_t)-{\bm{v}}_t\|^2$ を用いる。ここで $f_{\mathbf{S}}$ は ${\mathbf{S}}$ によってパラメータ化された非線形関数である。Mesa layer [Uncove24] は履歴全体を考慮する回帰 $\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\sum_{i=1}^t\|{\mathbf{S}}_t{\bm{k}}_i-{\bm{v}}_i\|^2$ を用いる。両者の違いは Least Mean Square と Recursive Least Square の違いに似ている。ただし、これらの表現力が高い変種は非線形漸化式を導入するため、回避策が必要になる。たとえば TTT と Titans のようにチャンク全体を処理した後だけ非線形更新を行うか、[Parall24, Systef24, Balanc25] のように非線形漸化式を近似する。

**ハイブリッドモデル。** 本研究では、層の間に注意層を挟むハイブリッド構成を検討する。この一般的な構成は MiniMax-01 [Scalin25] や Hybrid Mamba2-Attention [Catanz24] でも用いられている。一つの層の内部で線形注意と softmax 注意を組み合わせる方法 [Huaa22, Systeg24, ArXiv24, Combin24, Don25, Repreb25] も、興味深い研究対象である。

## 6 結論

本研究では Gated DeltaNet を提案した。Gated DeltaNet は Mamba2 より優れたキー・バリュー対応学習と、DeltaNet より適応的なメモリ消去を実現し、さまざまなタスクで一貫して高い実験性能を示す。[NeurIP24] の並列アルゴリズムを拡張し、Gated DeltaNet のハードウェア効率のよい訓練を可能にした。ハイブリッド Gated DeltaNet は訓練 throughput と総合性能をさらに改善し、実運用に適している。

## 謝辞

図の作成とモデル評価を支援してくれた Yu Zhang、原稿に有益な意見を寄せてくれた Kazuki Irie、長系列タスクの評価設定について議論してくれた Simeng Sun と Zhixuan Lin、DeltaNet のオンライン学習としての解釈について議論してくれた Eric Alcaide と Volodymyr Kyrylov に感謝する。

## 付録 A gated delta 則の拡張 WY 表現

記法を簡潔にするため、ここでは最初のチャンクだけを考える。

${\mathbf{S}}_t$ の拡張 WY 表現は

$$
{\mathbf{S}}_t=\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top,\qquad {\bm{u}}_t=\beta_t\left({\bm{v}}_t-\sum_{i=1}^{t-1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_t\right)
$$

である。これを数学的帰納法で証明する。

**証明。**

$$
{\mathbf{S}}_{t+1}={\mathbf{S}}_t\left({\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}\left({\mathbf{I}}-\beta_{t+1}{\bm{k}}_{t+1}{\bm{k}}_{t+1}^\top\right)\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
={\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}\left(\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top\right)-{\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}\beta_{t+1}\left(\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_i{\bm{k}}_{t+1}^\top\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top+\underbrace{\beta_{t+1}\left({\bm{v}}_{t+1}-\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_{t+1}\right)}_{{\bm{u}}_{t+1}}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^t{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top+\underbrace{{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{t+1}}}}_{1}{\bm{u}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^{t+1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top
$$

∎

## 付録 B 実験の続き

### B.1 評価

**常識推論。** [Daoc23] に従い、PIQA [AAAIc20]、HellaSwag [Italyb19]、WinoGrande [AAAId20]、ARC-easy (ARC-e) と ARC-challenge (ARC-c) [Clark18]、SIQA [Sap19]、BoolQ [Clark19]、Wikitext [ICLR17]、LAMBADA [August16] の各常識推論ベンチマークでモデルを評価する。

**文脈内検索。** 合成タスクと実世界タスクの両方を評価する。合成タスクには RULER [Hsieh24] の Needle-In-A-Haystack Single (NIAH-S) benchmark suite を用いる。これには難度が順に上がる S-NIAH-1 (passkey retrieval)、S-NIAH-2 (numerical needle in haystack)、S-NIAH-3 (word-based needle in haystack) の三タスクが含まれる。

実世界タスクでは [Aroraa24] に従い、構造化 HTML の関係抽出を行う SWDE [Lockar19]、PDF からキー・バリューを検索する FDA [Aroraa23]、さらに SQuAD [Austra18]、TriviaQA [Canada17]、Drop [Duaa19]、NQ [Kwiatk19] を含む複数の question-answering dataset で評価する。事前訓練済みモデルは instruction tuning を行っていないため、モデルの next-word prediction の訓練目的により近い [Aroraa24] の Cloze Completion Formatting prompt を使用する。

**長文脈理解。** LongBench [Bai23] の 14 タスクで評価する。内訳は、narrative comprehension (Narrative QA [Lingui18])、scientific understanding (QasperQA [Dasigi21])、multi-hop reasoning (MultiField QA、HotpotQA [Tsujib18]、2WikiMulti QA [Online20]、Musique [Lingui22])、document summarization (GovReport [Huang21]、QMSum [Zhong21]、MultiNews [Fabbri19])、各種の専門タスク (TRec [COLING02]、Trivia QA [Canada17]、SamSum [China19]、LCC [Guoa23]、RepoBench-P [Liub23]) である。

### B.2 アブレーション研究

<span id="table-06"></span>

![原論文の表 S.1](../../papers/gated-delta-networks/table-06.png)

**表 S.1.** Gated DeltaNet ブロックの ablation study。Avg-PPL と Avg-Acc は、それぞれ平均 perplexity と zero-shot 常識推論 accuracy ([表 3](#table-03)と同じ) を表す。すべてのモデルは 400M パラメータで、FineWeb-Edu [Penedo24] の同じ subset を用いて 15B token 訓練する。

<span id="table-07"></span>

![原論文の表 S.2](../../papers/gated-delta-networks/table-07.png)

**表 S.2.** Gated DeltaNet モデルの ablation study。すべての評価には `lm-evaluation-harness` [Gaob21] を用いる。すべてのモデルは Llama tokenizer を使用し、FineWeb-Edu [Penedo24] の同じ subset で訓練する。

[表 S.1](#table-06) は Gated DeltaNet ブロックの各構成要素に対する ablation study を示す。短い畳み込みと出力ゲートはいずれもモデル性能に不可欠であり、出力正規化による改善はわずかである。[NeurIP24] と同様に、最良の性能には L2 正規化が必要である一方、feature map の選択による影響は小さい。ただし、SiLU はほかの活性化関数を一貫して上回り、[Qin23] の観察と一致する。実験から、head dimension 128 が性能と計算効率の良好な trade-off になると分かった。また、[表 S.2](#table-07) は、複数のハイブリッドアーキテクチャのうち、Mamba2、Gated DeltaNet、SWA をこの順に組み合わせた構成が最も優れることを示す。

[+author-note]: 数式部分を担当。Songlin Yang が NVIDIA でインターンをしていた期間の研究。

[+1]: ここでは $\gamma$ の記号をやや広い意味で用い、系列全体ではなく、各チャンクの先頭位置から個別に計算した累積積を表す。

[+2]: $\beta_t\in(0,2)$ とすれば負の固有値を許容し、DeltaNet の状態追跡能力を引き出すこともできる [Grazzi24, Increa25]。

[+3]: 理論上の違いは最適化手法にある。Longhorn は implicit online learning [Bartle10] によって閉形式の大域最適更新を導出する一方、[Liua24] が指摘したように、DeltaNet は同じ目的を一段階の explicit gradient descent で最適化する。

[+4]: $\alpha$ には Mamba2 のパラメータ化を用いるが、簡潔にするため省略する。

[+5]: [https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)

[+6]: [https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule)
