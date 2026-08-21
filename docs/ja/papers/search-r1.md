---
title: 'Search-R1'
createTime: 2026/08/21 11:30:00
permalink: /ja/papers/search-r1/
pageClass: paper-reading
---

> [Bowen Jin](https://scholar.google.com/citations?user=dMwdOPkAAAAJ)、[Hansi Zeng](https://scholar.google.com/citations?user=a7O1D6oAAAAJ)、[Zhenrui Yue](https://scholar.google.com/citations?user=9Iy_KmsAAAAJ)、[Jinsung Yoon](https://scholar.google.com/citations?user=kiFd6A8AAAAJ)、[Sercan Ö. Arık](https://scholar.google.com/citations?user=-EZBCBAAAAAJ)、[Dong Wang](https://scholar.google.com/citations?user=-NfMhb0AAAAJ)、[Hamed Zamani](https://scholar.google.com/citations?user=d2uzDIAAAAAJ)、[Jiawei Han](https://hanj.cs.illinois.edu/)。2025 年 3 月 12 日に arXiv へ初回投稿、現行版は v5。[COLM 2025](https://openreview.net/forum?id=Rwhi91ideu) の会議論文として発表。[Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516)。[原論文 PDF](/paper/search-r1.pdf)。[DOI](https://doi.org/10.48550/arXiv.2503.09516)。[TeX ソース](https://export.arxiv.org/e-print/2503.09516v5)。正確な印刷レイアウトと参考文献については原論文 PDF を参照されたい。

## 概要

外部知識と最新情報を効率よく獲得することは、大規模言語モデル（LLM）による効果的な推論とテキスト生成に不可欠である。推論能力を備えた高度な LLM に、推論時に検索エンジンを使うようプロンプトで指示する方法は、多くの場合に最適ではない。LLM が検索エンジンと最適に対話する能力を十分に備えているとは限らないためである。本論文では、推論フレームワーク向けの強化学習（RL）を拡張した Search-R1 を提案する。LLM は、リアルタイム検索を伴う段階的推論の途中で、1 回または複数回の検索クエリを自律的に生成することを学習する。Search-R1 は、複数ターンの検索対話を含む LLM の推論軌跡を最適化し、安定した RL 学習のために検索 token のマスキングを用いるとともに、単純な結果ベースの報酬関数を採用する。7 つの質問応答データセットを用いた実験では、同一条件の各種 RAG ベースラインに対して、Search-R1 は Qwen2.5-7B で 24%、Qwen2.5-3B で 20% の性能向上を示した。さらに、RL の最適化手法、LLM の選択、検索拡張推論における応答長の変化について、実証的な知見を示す。コードとモデルチェックポイントは [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1) で公開している。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）は、自然言語理解と生成において顕著な能力を示している [Hen20, Cla18]。しかし、複雑な推論 [Wei22a] や、外部情報源からの最新情報の検索 [Jin25e] を求められると、LLM はしばしば困難に直面する。これらの制約を解消するには、高度な推論能力 [Hua22b] に加え、外部の最新情報を最大限に活用するため、検索エンジンと効果的に対話する能力 [Sch23] を統合する必要がある。

LLM と検索エンジンを統合する既存手法は、通常、（1）検索拡張生成（RAG）[Gao24e, Lew20] と、（2）検索エンジンをツールとして扱う手法 [Yao23b, Sch23] の 2 種類に分けられる。RAG モデルは一般に、LLM の入力をクエリとしてパッセージを検索し、それらを生成用の LLM コンテキストへ組み込む [Lew20]。これにより、LLM は質問への回答時に外部知識を利用できる。既存研究 [Tri22a] は、プロンプトによって LLM に複数ターン・複数クエリの検索を行わせるが、学習時に検索エンジンとの効果的な対話方法を最適化していないため、この方法は最適ではない。別の方法として、検索エンジンを含むツールを推論過程で利用するよう、LLM にプロンプトを与えたり学習させたりできる [Qu25, Tri22a]。ただし、プロンプトベースの手法は、LLM の事前学習で遭遇していないタスクに対して汎化しにくい。一方、学習ベースの手法は適応性が高いものの、大規模で高品質な注釈付き軌跡に依存するうえ、検索操作が本質的に微分不可能で、勾配降下法によるエンドツーエンド最適化を適用できないため、効果的なスケールアップが難しい [Sch23, Asa24a]。

強化学習（RL）[Sut99, Kae96] は、LLM の推論能力を高める有力な枠組みとなっている [Dee25c, Hou25a, Xie25c, Fau24]。OpenAI-o1 [Ope24h] や DeepSeek-R1 [Dee25c] などのモデルは、PPO [Sch17a] や GRPO [Sha24d] といった RL 手法を利用し、経験とフィードバックから学習することで論理的推論と問題解決能力を改善している。結果報酬だけで学習した場合でも、モデルは自己検証 [Wen22a] や自己修正 [Fau24] などの複雑な推論能力を獲得する。しかし、検索と推論を組み合わせた状況へ RL を適用するには、3 つの主要な課題がある。（1）**RL フレームワークと安定性** – 検索結果のコンテキストを取り込む場合に最適化の安定性を保ちながら、検索エンジンを LLM 向け RL 手法へ効果的に統合する方法は明確でない。（2）**複数ターンで交互に行う推論と検索** – 理想的には、LLM は反復的に推論して検索エンジンを呼び出し、問題の複雑さに応じて検索戦略を動的に調整できるべきである。（3）**報酬設計** – 単純な結果報酬だけで、有意義かつ一貫した検索行動を LLM に学習させられるかが明らかでないため、検索・推論タスクに有効な報酬関数の設計は根本的な課題である。

これらの課題に対処するため、LLM が自身の推論と検索エンジンとの対話を交互に行える新しい RL フレームワーク、Search-R1 を提案する。Search-R1 の主要な新規性は次のとおりである。（1）検索エンジンを環境の一部としてモデル化し、LLM の token 生成と検索エンジンによる検索が交互に現れる軌跡系列をサンプリングできるようにする。Search-R1 は PPO や GRPO を含む各種 RL アルゴリズムと互換性があり、安定した最適化のために検索 token マスキングを適用する。（2）Search-R1 は複数ターンの検索と推論をサポートし、`<search>` と `</search>` token が明示的に生成されると検索を呼び出す。検索内容は `<information>` と `</information>` token で囲み、LLM の推論ステップは `<think>` と `</think>` token で囲む。最終回答は `<answer>` と `</answer>` token で整形し、構造化された反復的意思決定を可能にする。（3）複雑なプロセス報酬を避け、単純な結果ベースの報酬関数を採用する。実験結果は、この最小限の報酬設計が検索・推論タスクで有効であることを示す。この意味で Search-R1 は、主としてパラメトリック推論に焦点を当てた DeepSeek-R1 Zero [Dee25c] を拡張し、検索拡張 RL 学習によって検索駆動の意思決定を改善するものとみなせる。

主な貢献は以下の 3 点である。

- 検索エンジンの結果を用いた LLM の推論を RL で改善する際の課題を分析し、実装に関する見解を示す。
- LLM の rollout と検索エンジンを用いた直接最適化をサポートする新しい RL フレームワーク Search-R1 を提案する。検索 token マスキングによって RL 学習を安定化し、複雑なタスクの解決に向けて複数ターンの推論と検索を交互に実行し、有効な結果報酬関数を用いる。
- Search-R1 の有効性を示す体系的な実験を行う。同一の実験設定（同一の検索モデル、学習データ、事前学習済み LLM など）の下で、2 つの LLM は RAG ベースラインに対し、それぞれ平均相対性能を **41% と 20% 改善**した。加えて、RL 手法の選択、異なる LLM の選択、応答長の検討を含め、推論・検索環境における RL について知見を示す。

<span id="section-2"></span>

## 2 関連研究

<span id="section-2-1"></span>

### 2.1 大規模言語モデルと検索

LLM [Zha23d, Tea24a, Ope23] は優れた推論 [Dee25c] とコーディング [Guo24c] の能力を示す一方、ドメイン固有の知識を欠くことが多く [Pen23d, Li23v]、ハルシネーションも起こしやすい [Zha23j]。この制約を緩和するため、外部情報を提供する検索エンジン [Zha24u] が広く統合されている。検索エンジンを LLM と統合する方法は、（1）検索拡張生成（RAG）[Gao24e] と、（2）検索エンジンをツールとして扱う方法 [Sch23] の 2 つが中心である。RAG [Lew20, Yue24d, Xio25a] は通常、1 回の検索とそれに続く生成からなるパイプラインを用いる。検索エンジンは入力クエリに基づいて関連情報を取得し、それをクエリと連結して LLM へ入力する。ただし、無関係な情報を検索したり [Jin25e]、十分に有用なコンテキストを得られなかったりする [Jia23c] 可能性がある。もう 1 つは search-as-a-tool であり、検索エンジンと対話するよう LLM にプロンプトを与えるか、微調整する。IRCoT [Tri22a] と ReAct [Yao23b] は、反復的な推論と検索エンジンの呼び出しをプロンプトで誘導し、Toolformer [Sch23] は教師あり微調整で検索能力を強化する。しかし、このような手法は、大規模に取得することが難しい高品質なラベル付き軌跡に依存する。近年の研究 [Dee25c] は、結果報酬だけを用いる RL によって LLM が高度な推論能力を獲得できることを示したが、検索エンジン呼び出しへの応用可能性は十分に検討されていない。

<span id="section-2-2"></span>

### 2.2 大規模言語モデルと強化学習

強化学習（RL）[Kae96] は、エージェントが環境と対話し、報酬としてフィードバックを受けながら逐次的な意思決定を学習し、時間とともに累積報酬を最大化する枠組みである [Sut99]。[Ouy22a] は、人間のフィードバックからの RL（RLHF）[Kau23] により、LLM の調整へ RL を導入した。この手法は、まず人間の選好データ [Lam24a] を用いて報酬モデルを学習し、その後、通常は Proximal Policy Optimization（PPO）を用いて、報酬モデルがポリシー LLM の RL 調整を導く。ただし、PPO では LLM を複数回最適化するため、実装が難しい。RL ベースの調整を単純化するため、Direct Preference Optimization（DPO）[Raf23] や SimPO [Men24] などの直接最適化手法が提案されている。LeRet [Hsu24] も同様の方法を採用し、情報検索の有効性を高めるため、多様なクエリを探索するよう LLM を学習する。これらの手法は計算効率に優れる一方、オフポリシーの問題 [Pan24d] があり、純粋な RL 手法の性能に常に匹敵するとは限らない。代替手法には、グループスコアからベースラインを推定して critic モデルを不要にする Group Relative Policy Optimization（GRPO）[Sha24d] と、単純化した REINFORCE 型 [Wil92] の最適化を行う RLOO [Ahm24] がある。これらの進歩にもかかわらず、LLM が検索エンジンと対話しながら推論する状況への RL の適用は、ほとんど研究されていない。

<span id="section-3"></span>

## 3 Search-R1

<span id="figure-01"></span>

![図 1。検索エンジンを用いる PPO と GRPO の学習。](../../papers/search-r1/figure-01.png)

**図 1。** 検索エンジンを用いる PPO と GRPO の学習（Search-R1）。Rollout 中、LLM は検索エンジンと複数ターンの対話を行える。

以下では Search-R1 の学習手法の詳細設計として、（1）検索エンジンを利用するための RL の拡張、（2）複数ターンの検索エンジン呼び出しを交互に含むテキスト生成、（3）学習テンプレート、（4）報酬モデル設計を説明する。

<span id="section-3-1"></span>

### 3.1 検索エンジンを用いた強化学習

検索エンジン $\mathcal{R}$ を利用する RL の目的関数を次のように定式化する。

<span id="equation-01"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x; \mathcal{R}) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x; \mathcal{R}) \right],
$$

ここで、$\pi_{\theta}$ はポリシー LLM、$\pi_{\mathrm{ref}}$ は参照 LLM、$r_{\phi}$ は報酬関数、$\mathbb{D}_{\mathrm{KL}}$ は KL ダイバージェンス尺度である。$x$ はデータセット $\mathcal{D}$ から抽出した入力サンプルを表し、$y$ は検索エンジン呼び出しの結果が交互に挿入された生成出力を表す。これは参照ポリシー $\pi_{\mathrm{ref}}(y \mid x)$ からサンプリングされ、検索エンジン $\mathcal{R}$ から取得される。主としてポリシー LLM $\pi_{\theta}(\cdot \mid x)$ に依存して rollout 系列を生成する従来の RL 手法 [Raf23, Ouy22a] とは異なり、本フレームワークは $\pi_{\theta}(\cdot \mid x; \mathcal{R})$ により、検索を交互に挟む推論を明示的に組み込む。これは $\pi_{\theta}(\cdot \mid x) \bigotimes \mathcal{R}$ とみなすことができ、$\bigotimes$ は検索と推論を交互に行うことを表す。これにより、外部情報の検索を必要とする推論集約的なタスクで、より効果的な意思決定が可能になる。rollout 過程の図解と[式 1](#equation-01)の説明は、[3.2 節](#section-3-2)と[付録 A](#appendix-a)に示す。

本手法は、確立された 2 つの方策勾配 RL 手法、Proximal Policy Optimization（PPO）[Sch17a] と Group Relative Policy Optimization（GRPO）[Sha24d, Dee25c] を基礎とし、それぞれの利点を活用して検索拡張推論を最適化する。

**検索 token の損失マスキング。** PPO と GRPO のいずれでも、token 単位の損失は rollout 系列全体に対して計算される。Search-R1 の rollout 系列は、LLM が生成した token と外部パッセージから取得した token の両方で構成される。LLM が生成した token を最適化すると、検索エンジンと対話して推論するモデルの能力は高まるが、取得した token に同じ最適化を適用すると、意図しない学習ダイナミクスが生じる可能性がある。そこで、検索 token に損失マスキングを導入し、取得内容を最適化過程から除外して、LLM が生成した token のみに対して方策勾配目的を計算する。これにより、検索拡張生成の柔軟性を維持しながら学習を安定させる。

**検索エンジンを用いた PPO。** Proximal Policy Optimization（PPO）[Sch17a] は、LLM で広く用いられる actor-critic 型 RL 手法である [Ouy22a]。検索エンジン呼び出しを含む本研究の推論状況では、次の目的関数を最大化して LLM を最適化する。

<span id="equation-02"></span>

$$
\mathcal{J}_{\mathrm{PPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\mathrm{old}}( \cdot\mid x; \mathcal{R})}
\left[ \frac{1}{\sum_{t=1}^{|y|} I(y_t)} \sum_{t=1: I(y_t)=1}^{|y|}
\min \left( \frac{\pi_{\theta}(y_t \mid x, y_{<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_t \mid x, y_{<t}; \mathcal{R})} A_t,
\mathrm{clip} \left( \frac{\pi_{\theta}(y_t \mid x, y_{<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_t \mid x, y_{<t}; \mathcal{R})}, 1 - \epsilon, 1 + \epsilon \right) A_t
\right) \right],
$$

ここで、$\pi_{\theta}$ と $\pi_{\mathrm{old}}$ は、それぞれ現在と以前のポリシーモデルを表す。$I(y_t)$ は token の損失マスキング演算であり、$y_t$ が LLM の生成 token なら $I(y_t)=1$、検索 token なら $I(y_t)=0$ となる。$\epsilon$ は学習を安定させるため PPO に導入されたクリッピング関連のハイパーパラメータである。アドバンテージ推定値 $A_t$ は、将来の報酬 $\{ r_{\geq t} \}$ と学習済み価値関数 $V_{\phi}$ に基づき、Generalized Advantage Estimation（GAE）[Sch15] で計算する。

**検索エンジンを用いた GRPO。** 方策最適化の安定性を高め、追加の価値関数近似を不要にするため、[Sha24d] では Group Relative Policy Optimization（GRPO）が導入された。GRPO は、学習済み価値関数に依存せず、複数のサンプル出力の平均報酬をベースラインとして用いる点で PPO と異なる。具体的には、各入力質問 $x$ に対し、GRPO は参照ポリシー $\pi_{\mathrm{ref}}$ から応答のグループ $\{ y_1, y_2, \dots, y_G \}$ をサンプリングする。その後、次の目的関数を最大化してポリシーモデルを最適化する。

<span id="equation-03"></span>

$$
\begin{aligned}
\mathcal{J}_{\mathrm{GRPO}}(\theta) =\;&
\mathbb{E}_{x \sim \mathcal{D}, \{ y_i \}_{i=1}^{G} \sim \pi_{\mathrm{old}}( \cdot\mid x; \mathcal{R})}
\Bigg[
\frac{1}{G} \sum_{i=1}^{G} \frac{1}{\sum_{t=1}^{|y_i|} I(y_{i,t})} \sum_{t=1: I(y_{i,t})=1}^{|y_i|}
\min \Bigg(
\frac{\pi_{\theta}(y_{i,t} \mid x, y_{i,<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_{i,t} \mid x, y_{i,<t}; \mathcal{R})} \hat{A}_{i,t},\\[8pt]
&\hspace{80pt} \mathrm{clip} \Bigg( \frac{\pi_{\theta}(y_{i,t} \mid x, y_{i,<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_{i,t} \mid x, y_{i,<t}; \mathcal{R})}, 1 - \epsilon, 1 + \epsilon \Bigg) \hat{A}_{i,t}
\Bigg)
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta} \| \pi_{\mathrm{ref}} \right]
\Bigg],
\end{aligned}
$$

ここで、$\epsilon$ と $\beta$ はハイパーパラメータであり、$\hat{A}_{i,t}$ は各グループ内の出力の相対報酬に基づいて計算したアドバンテージを表す。この手法では、$\hat{A}_{i,t}$ の計算に余分な複雑さを持ち込まずに済む。また、KL ダイバージェンスを報酬関数内のペナルティとして組み込む代わりに、学習ポリシーと参照ポリシーの KL ダイバージェンスを損失関数へ直接加えて正則化する。KL ダイバージェンス損失 $\mathbb{D}_{\mathrm{KL}}$ の計算時にも、検索 token のマスキングを適用する。

<span id="section-3-2"></span>

### 3.2 複数ターンの検索エンジン呼び出しを伴う生成

本節では、複数ターンの検索エンジン呼び出しを交互に挟んで LLM 応答を生成する rollout 過程を説明する。これは $y\sim \pi_{\theta}(\cdot \mid x; \mathcal{R}) = \pi_{\theta}(\cdot \mid x) \bigotimes \mathcal{R}$ と定式化される。

本手法は、LLM がテキスト生成と外部検索エンジンへの問い合わせを交互に行う反復的な枠組みに従う。具体的には、外部検索が必要なとき、システム命令によって、LLM は検索クエリを指定された 2 つの検索呼び出し token、`<search>` と `</search>` の間に入れる。生成系列でこれらの token を検出すると、システムは検索クエリを抽出し、検索エンジンへ問い合わせて関連する結果を取得する。取得情報は専用の検索 token、`<information>` と `</information>` で囲み、進行中の rollout 系列へ追加して、次の生成ステップの追加コンテキストとして用いる。この過程は、（1）action の最大回数に達するか、（2）モデルが指定された回答 token `<answer>` と `</answer>` で囲んだ最終応答を生成するまで反復する。ワークフロー全体をアルゴリズム 1 に示す。

**アルゴリズム 1：複数ターンの検索エンジン呼び出しを伴う LLM 応答の rollout**

- **入力：** 入力クエリ $x$、ポリシーモデル $\pi_{\theta}$、検索エンジン $\mathcal{R}$、action の最大予算 $B$。
- **出力：** 最終応答 $y$。
- rollout 系列を $y \gets \emptyset$ で初期化する。
- action 回数を $b \gets 0$ で初期化する。
- **While** $b < B$：
  - 現在の action における LLM rollout 系列を $y_b \gets \emptyset$ で初期化する。
  - **While** True：
    - 応答 token $y_t \sim \pi_{\theta}(\cdot \mid x, y + y_b)$ を生成する。
    - $y_t$ を rollout 系列へ追加し、$y_b \gets y_b + y_t$ とする。
    - **If** $y_t$ が [`</search>`, `</answer>`, `<eos>`] に含まれる：
      - break。
  - $y \gets y + y_b$。
  - **If** $y_b$ で `<search>` を検出した：
    - 検索クエリ $q \gets \mathrm{Parse}(y_b, \texttt{<search>}, \texttt{</search>})$ を抽出する。
    - 検索結果 $d = \mathcal{R}(q)$ を取得する。
    - $d$ を rollout へ挿入し、$y \gets y + \texttt{<information>}d\texttt{</information>}$ とする。
  - **Else if** $y_b$ で `<answer>` を検出した：
    - **Return** 最終生成応答 $y$。
  - **Else：**
    - 再考を促し、$y \gets y +$ 「私の action は正しくない。考え直そう。」とする。
  - action 回数を $b \gets b + 1$ とする。
- **Return** 最終生成応答 $y$。

<span id="section-3-3"></span>

### 3.3 学習テンプレート

Search-R1 の学習では、まず初期 LLM に所定の命令へ従うよう指示する単純なテンプレートを作成する。[表 1](#table-01)に示すように、このテンプレートはモデル出力を反復的に 3 つの部分、すなわち推論過程、検索エンジン呼び出し関数、回答の順に構成する。反省的推論や検索エンジン呼び出しを強制したり、特定の問題解決法を推奨したりするなど、内容に固有のバイアスを避けるため、制約は意図的にこの構造形式だけに限定する。これにより、RL 過程におけるモデル本来の学習ダイナミクスを観察でき、バイアスもかからない。

<span id="table-01"></span>

![表 1。Search-R1 のテンプレート。](../../papers/search-r1/table-01.png)

**表 1。** Search-R1 のテンプレート。`question` は学習時と推論時に具体的な質問へ置き換えられる。

<span id="section-3-4"></span>

### 3.4 報酬モデリング

報酬関数は主要な学習信号として、RL の最適化過程を導く。Search-R1 の学習には、モデル応答の正しさを評価する**最終結果報酬**だけで構成されたルールベースの報酬システムを採用する。たとえば、事実推論タスクでは、完全文字列一致のようなルールベースの基準で正しさを評価できる。

<span id="equation-04"></span>

$$
r_{\phi}(x, y) = \mathrm{EM}(a_{\mathrm{pred}}, a_{\mathrm{gold}}),
$$

ここで、$a_{\mathrm{pred}}$ は応答 $y$ から抽出した最終回答、$a_{\mathrm{gold}}$ は正解である。[Dee25c] とは異なり、学習済みモデルがすでに構造へ高い適合性を示すため、形式報酬は組み込まない。より複雑な形式報酬の検討は今後の課題とする。さらに、[Dee25c] に従い、ニューラル報酬モデルの学習も避ける。この判断は、大規模 RL において LLM が特定の報酬形式に敏感であることに加え、これらのモデルを再学習すると計算コストと複雑さが増すことに基づく。

<span id="section-4"></span>

## 4 主な結果

<span id="section-4-1"></span>

### 4.1 データセット

Search-R1 を 7 つのベンチマークデータセットで評価し、次のように分類する。（1）**一般質問応答：** NQ [Kwi19a]、TriviaQA [Jos17]、PopQA [Mal22]。（2）**マルチホップ質問応答：** HotpotQA [Yan18a]、2WikiMultiHopQA [Ho20]、Musique [Tri22]、Bamboogle [Lew23]。これらのデータセットは、推論を伴う多様な検索課題を含み、Search-R1 の包括的な評価を可能にする。

<span id="section-4-2"></span>

### 4.2 ベースライン

Search-R1 の有効性を評価するため、次のベースラインと比較する。（1）**検索を用いない推論：** 直接推論と Chain-of-Thought（CoT）推論 [Wei22a]。（2）**検索を用いる推論：** Retrieval-Augmented Generation（RAG）[Lew20]、IRCoT [Tri22a]、Search-o1 [Li25k]。（3）**微調整ベースの手法：** 教師あり微調整（SFT）[Chu22]、検索エンジンを用いない RL ベースの微調整（R1）[Dee25c]、検索エンジンを用いた棄却サンプリング [Ahn24]。R1 では Search-R1 と公平に比較するため、[Dee25c] が提案した RL 手法と本研究のデータを用いて LLM を学習する。これは検索エンジンを用いず、推論と回答のステップだけを含む。棄却サンプリングでは、指示された LLM に同じデータセットの各学習プロンプトから 5 つの候補応答を生成させ、正しい最終回答に至るものを選択する。その後、選択した軌跡から新しい学習セットを構築し、Search-R1 で提案したものと同じ複数ターンの LLM–検索エンジン対話 rollout 機構を保ったまま LLM を微調整する。

これらのベースラインは検索拡張手法と微調整手法を幅広く網羅し、ゼロショットと学習済み検索の両方の設定で Search-R1 を包括的に評価できる。各手法を公平に比較するため、同じ検索器、同じ検索文書数、同じ知識コーパス、同じ学習データ、同じ事前学習済み LLM を用いる。詳細は[付録 B](#appendix-b)に示す。

<span id="section-4-3"></span>

### 4.3 実験設定

実験には Qwen-2.5-3B（Base/Instruct）と Qwen-2.5-7B（Base/Instruct）[Yang24] の 2 種類のモデルを用いる。検索では、2018 年の Wikipedia dump [Kar20] を知識源、E5 [Wan22i] を検索器として用いる。公平に比較するため、[Lin23c] に従い、すべての検索ベース手法で取得するパッセージ数を 3 に設定する。検索パッセージ数の検討は[付録 G](#appendix-g)に示す。

学習では、NQ と HotpotQA の学習セットを統合し、Search-R1 とその他の微調整ベースのベースラインに共通するデータセットを作成する。7 つのデータセットのテストセットまたは検証セットで評価し、ドメイン内とドメイン外の両方の性能を測定する。[Yu24a] に従い、評価指標には Exact Match（EM）を用いる。base モデルは命令に従えないため、推論型ベースラインには instruct モデルを用いる。RL 調整手法の実験は、base と instruct の両モデルで行う。実験設定の詳細は[付録 B](#appendix-b)に示す。

特に断りがない限り、**デフォルトの RL 手法には PPO を用い**、PPO と GRPO の詳細な比較は[5.1 節](#section-5-1)に示す。

<span id="table-02"></span>

![表 2。主な結果。](../../papers/search-r1/table-02.png)

**表 2。** 主な結果。最高性能を太字で示す。$^\dagger/^\star$ はドメイン内／ドメイン外のデータセットを表す。

<span id="section-4-4"></span>

### 4.4 性能

7 つのデータセットにおける Search-R1 とベースライン手法の主な比較結果を[表 2](#table-02)に示す。結果から次の重要な知見を得た。**（1）Search-R1 は一貫して強力なベースライン手法を上回る。** Qwen2.5-7B と Qwen2.5-3B で、それぞれ平均相対性能が 24% と 20% 向上した。この向上は、分布内評価（*すなわち* NQ と HotpotQA）と分布外評価（*すなわち* TriviaQA、PopQA、2WikiMultiHopQA、Musique、Bamboogle）の両方で認められる。**（2）Search-R1 は検索を用いない LLM 推論の RL ベース学習（R1）を上回る。** 検索を LLM の推論へ組み込むと関連する外部知識を利用でき、全体的な性能が向上するため、これは予想どおりである。**（3）Search-R1 は base モデルと instruction-tuned モデルの両方で有効である。** これは、結果ベースの報酬を用いる DeepSeek-R1-Zero 型 RL [Dee25c] が検索を伴う推論にも適用でき、純粋な推論状況で従来実証されていた有効性を超えて拡張できることを示す。**（4）大きなモデルほど検索方法の学習に優れる。** 7B モデル上の Search-R1 は、3B モデルよりもはるかに大きな「性能差」を示す（*たとえば*第 2 位のモデルである RAG との差）。

<span id="section-5"></span>

## 5 分析

<span id="section-5-1"></span>

### 5.1 異なる RL 手法：PPO と GRPO

PPO と GRPO のそれぞれを基礎となる RL 手法として、Qwen2.5-3B/7B モデル上で Search-R1 を評価する。学習ダイナミクスの比較を[図 2(a)](#figure-02)、評価結果を[表 3](#table-03)に示す。そこから次の知見が得られる。**（1）すべての場合で GRPO は PPO より速く収束する。** これは、PPO が critic モデルに依存し、有効な学習を始めるまでに複数の warm-up ステップを必要とするためである。**（2）PPO は学習の安定性に優れる。** [図 2(a)](#figure-02)に示すように、GRPO は多数のステップを学習した後に報酬が崩壊する一方、PPO は安定している。**（3）PPO と GRPO の最終学習報酬は同程度である。** 収束速度と安定性は異なるものの、両手法の最終学習報酬と性能は近く、どちらも Search-R1 の最適化に利用できる。PPO は学習の安定性に優れるため、本設定ではより望ましい選択肢である。詳細な結果は[付録 F](#appendix-f)に示す。

<span id="table-03"></span>

![表 3。PPO と GRPO を用いた Search-R1 の 7 つのデータセットにおける性能。](../../papers/search-r1/table-03.png)

**表 3。** PPO と GRPO を用いた Search-R1 の 7 つのデータセットにおける性能。

<span id="figure-02"></span>

![図 2。PPO と GRPO、Base と Instruct、応答長、有効な検索。](../../papers/search-r1/figure-02.png)

**図 2。** (a) PPO と GRPO：GRPO は一般に速く収束するが、一定のステップを学習すると不安定になる場合がある。一方、PPO は最適化がより安定しているが、収束は遅い。(b) Base と Instruct LLM の検討：Instruction-tuned LLM は速く収束するが、両モデルの最終性能は非常に近い。(c) 応答長の検討：応答長は学習全体を通して減少–増加–安定という傾向を示し、LLM の全体的な性能推移と一致する。(d) 有効な検索回数の検討：学習が進むにつれ、LLM はより多く検索を呼び出すようになる。

<span id="section-5-2"></span>

### 5.2 Base LLM と Instruct LLM

base LLM と instruction-tuned LLM の両方について、Search-R1 の学習ダイナミクスを分析する。実験には Qwen2.5-3B と Qwen2.5-7B の 2 つのモデル変種を用いる。[図 2(b)](#figure-02)に示すように、instruction-tuned モデルは base モデルより速く収束し、初期性能も高い。しかし、学習後の両モデルの最終学習報酬は非常に近い。この知見は、一般的な post-training が推論と検索を組み合わせた状況で学習を加速する一方、RL は時間とともにその差を効果的に埋め、base モデルも同程度の性能へ到達できることを示唆する。詳細な結果は[付録 E](#appendix-e)に示す。

<span id="table-04"></span>

![表 4。検索 token の損失マスキングを用いる場合と用いない場合の Search-R1 の性能。](../../papers/search-r1/table-04.png)

**表 4。** 検索 token の損失マスキングを用いる場合と用いない場合の Search-R1 の性能。検索 token の損失マスキングを用いて学習した LLM は、一貫して高い性能を達成する。（LLM：Qwen2.5-7b-base、RL：PPO）

<span id="section-5-3"></span>

### 5.3 応答長と有効な検索の検討

Qwen2.5-7b-base モデルで Search-R1 の実験を行い、学習中の応答長と有効な検索エンジン呼び出し回数の推移を分析する。応答長の結果を[図 2(c)](#figure-02)に示す。主な傾向は次のとおりである。**（1）初期段階（最初の 100 ステップ）：** 応答長は急激に短くなり、学習報酬はわずかに増える。この段階で base モデルは冗長な埋め草を取り除き、タスク要件へ適応し始める。**（2）後期段階（100 ステップ以降）：** 応答長と学習報酬がともに大きく増える。この時点で、LLM は検索エンジンを頻繁に呼び出すことを学習し、取得したパッセージによって応答が長くなる。モデルが検索結果を効果的に利用できるようになるため、学習報酬は大幅に向上する。有効な検索の結果を[図 2(d)](#figure-02)に示す。学習が進むにつれて、LLM は検索エンジンをより多く呼び出すことを学習している。

<span id="section-5-4"></span>

### 5.4 検索 token の損失マスキングの検討

[3.1 節](#section-3-1)では、意図しない最適化挙動を防ぐため、検索 token の損失マスキングを導入した。ここでは Qwen2.5-7b-base モデルで実験し、検索 token の損失マスキングを用いる場合と用いない場合の学習ダイナミクスを比較する。[図 3](#figure-03)に示すように、検索 token のマスキングを適用すると LLM の改善が大きくなり、意図しない最適化の影響を抑えて学習の安定性を確保できる。[表 4](#table-04)に示す性能比較から、検索 token の損失マスキングを用いて学習した Search-R1 は、マスキングを用いない変種を一貫して上回ることが分かる。

検索 token の損失マスキング、base LLM と instruct LLM、PPO と GRPO の比較、Search-R1 学習時の検索パッセージ数、Search-R1（GRPO）のグループサイズ、ケーススタディに関する詳細な実験結果は、[付録 D](#appendix-d)、[付録 E](#appendix-e)、[付録 G](#appendix-g)、[付録 H](#appendix-h)、[付録 I](#appendix-i)、[付録 J](#appendix-j)に示す。

<span id="section-6"></span>

## 6 結論

本研究では、LLM が自己推論とリアルタイムの検索エンジン対話を交互に行える新しい RL フレームワーク、Search-R1 を提案した。複数ターンの検索に大量のプロンプトを必要とする既存の RAG 型手法や、大規模な教師あり学習データを必要とするツール利用手法とは異なり、Search-R1 は RL で LLM の rollout を最適化し、自律的なクエリ生成と取得情報の戦略的な利用を可能にする。7 つのデータセットを用いた広範な実験により、Search-R1 がリアルタイムの外部知識を必要とする複雑な推論タスクへ対処する LLM の能力を大幅に高めることを示した。また、本分析は検索拡張推論における RL の学習戦略について重要な知見を与える。今後は、より高度な報酬機構、不確実性に基づく検索の動的調整、多様なツールとの組み合わせ、検索以外の多様な情報源との統合など、Search-R1 をさらに広範な検索戦略へ拡張できる。マルチモーダル推論タスクへの適用可能性を検討することも有望である。

## 謝辞

本研究は一部、Apple PhD Fellowship、米国 DARPA の INCAS Program No. HR0011-21-C0165 と BRIES Program No. HR0011-24-3-0325、Office of Naval Research の契約番号 N000142412612、NSF の助成番号 IIS-19-56151 と 2402873、NSF Award No. 2019897 により支援された AI Research Institutes プログラムである Molecule Maker Lab Institute、NSF Award No. 2118329 による Institute for Geospatial Understanding through an Integrative Discovery Environment（I-GUIDE）、Cisco、Center for Intelligent Information Retrieval の支援を受けた。ここに示す意見、知見、結論または提言は著者のものであり、明示・黙示を問わず、必ずしも支援者または米国政府の見解を表すものではない。

<span id="appendix-a"></span>

## A 検索エンジンを用いた強化学習の定式化

大規模言語モデル（LLM）を学習するための古典的な強化学習（RL）の枠組みは、次のように定式化できる [Raf23, Ouy22a]。

<span id="equation-05"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x)}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x) \right],
$$

ここで、$x$ はデータセット $\mathcal{D}$ からサンプリングしたプロンプト、$y$ はポリシーモデル $\pi_\theta$ が生成した応答、$\pi_{\mathrm{ref}}$ は正則化の基準となる参照モデルを表す。報酬関数 $r_{\phi}(x, y)$ は生成応答の品質を定量化し、KL ダイバージェンス項は更新後のポリシーが参照モデルに近い状態を保つよう制約して、学習の安定性を高める。

ただし、この定式化は出力系列 $y$ の全体がポリシー LLM だけによって生成されると仮定している。モデルの挙動に内部推論と外部情報検索の両方が含まれる本研究の設定では、この仮定は成り立たない。そこで、RL の目的を外部検索エンジン $\mathcal{R}$ を組み込むよう拡張し、次の定式化を得る。

<span id="equation-06"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x; \mathcal{R}) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x; \mathcal{R}) \right],
$$

この修正した目的では、軌跡 $y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})$ が交互に現れる推論ステップと取得内容を含み、LLM と検索エンジンの複数ターンの対話を反映する。KL ダイバージェンスは、プロンプトと検索拡張コンテキストの両方を条件とする同時応答分布上で計算されるため、外部情報が存在する場合でも、学習ポリシーは参照モデルと整合する。

<span id="appendix-b"></span>

## B 実験設定

<span id="appendix-b-1"></span>

### B.1 ベースライン

近年の複数の研究は、Natural Questions（NQ）や HotpotQA などのベンチマークを中心に、より精緻な検索機構によって性能を高める RAG パイプラインを検討している。たとえば Re2G [Gla22] と RetroLLM [Li24r] は、強力な検索器と複雑な再ランキング戦略を用い、生成用の細粒度な根拠を選択する高度な retrieve-rerank-generate フレームワークを提案している。これらの手法は優れた結果を示す一方、タスク固有の設計や大規模なパイプラインに依存することが多く、汎化性とスケーラビリティが制限される。これに対し、本研究は検索拡張推論のための、より軽量で汎用的な手法に焦点を当てる。そのため、これらの手法は直接のベースラインには含めないが、検索拡張言語モデリングという広い領域における重要な方向性である。

<span id="appendix-b-2"></span>

### B.2 実験設定

実験には Qwen-2.5-3B（Base/Instruct）と Qwen-2.5-7B（Base/Instruct）[Yang24] の 2 種類のモデルを用いる。検索では、2018 年の Wikipedia dump [Kar20] を知識源、E5 [Wan22i] を検索器として用いる。公平に比較するため、[Lin23c] に従い、すべての検索ベース手法で取得するパッセージ数を 3 に設定する。

学習では、NQ と HotpotQA の学習セットを統合し、Search-R1 とその他の微調整ベースのベースラインに共通するデータセットを作成する。7 つのデータセットのテストセットまたは検証セットで評価し、ドメイン内とドメイン外の両方の性能を測定する。[Yu24a] に従い、評価指標には Exact Match（EM）を用いる。base モデルは命令に従えないため、推論型ベースラインには instruct モデルを用いる。RL 調整手法の実験は、base と instruct の両モデルで行う。実験設定の詳細は[付録 B](#appendix-b)に示す。

Search-R1 の PPO 変種では、ポリシー LLM と価値 LLM の学習率をそれぞれ 1e-6 と 1e-5 に設定する。500 ステップ学習し、ポリシーモデルと価値モデルの warm-up 比率は、それぞれ 0.285 と 0.015 とする。パラメータ $\lambda = 1$、$\gamma = 1$ の Generalized Advantage Estimation（GAE）を用いる。

学習は 8 基の H100 GPU を備えた単一ノードで行う。全体の batch size を 512、mini-batch size を 256、micro-batch size を 64 とする。最大系列長は 4,096 token、最大応答長は 500、取得内容の最大長も 500 token とする。GPU メモリ利用を最適化するため、gradient checkpointing を有効にし、CPU offloading を伴う Fully Sharded Data Parallel（FSDP）を用いる。

効率的な LLM rollout のため、tensor parallel size を 1、GPU memory utilization ratio を 0.6 として vLLM [+1] を採用する。rollout のサンプリングでは temperature を 1.0、top-p を 1.0 とする。KL ダイバージェンス正則化係数 $\beta$ と clip ratio $\epsilon$ は、それぞれ 0.001 と 0.2 に設定する。

GRPO の学習では、ポリシー LLM の学習率を 1e-6 とし、Verl の GRPO 実装 [She24a] [+2] に従ってプロンプトごとに 5 つの応答をサンプリングする。モデルを 500 ステップ学習し、学習率の warm-up 比率を 0.285 とする。PPO と同じ 8×H100 環境を用い、batch size と系列長の設定も同一とする。

gradient checkpointing、FSDP offloading、vLLM ベースの rollout も、上記と同じハイパーパラメータで用いる。rollout の temperature と top-p はともに 1.0、KL ダイバージェンス係数 $\beta$ と clip ratio $\epsilon$ はそれぞれ 0.001 と 0.2 に固定する。

両手法とも、モデルの checkpoint を 100 ステップごとに保存する。学習が発散した場合は、学習報酬曲線に基づく直近の安定した checkpoint で評価し、それ以外の場合は最終 checkpoint を評価に用いる。action の最大予算 $B$ を 4 とし、デフォルトでは上位 3 つのパッセージを取得する。

結果報酬は exact match（EM）で計算する。特に断りがない限り、**デフォルトの RL アルゴリズムには PPO を用い**、GRPO との詳細な比較は[5.1 節](#section-5-1)に示す。

[+1]: [https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)

[+2]: [https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_deepseek7b_llm.sh](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_deepseek7b_llm.sh)

<span id="appendix-c"></span>

## C 14B LLM における主な結果

Qwen2.5-14B モデルを用いて広範な実験を行い、その結果を[表 5](#table-05)に示す。Search-R1 は、評価したすべての指標でベースライン手法を一貫して上回る。また、モデルサイズを拡大すると Search-R1 の性能が一貫して向上することが分かり、本手法における LLM のスケーリングの利点が浮き彫りになった。

<span id="table-05"></span>

![表 5。14B LLM における主な結果。](../../papers/search-r1/table-05.png)

**表 5。** 主な結果。最高性能を太字で示す。$^\dagger/^\star$ はドメイン内／ドメイン外のデータセットを表す。

<span id="appendix-d"></span>

## D 検索 token の損失マスキングの検討

[3.1 節](#section-3-1)では、学習中の望ましくない最適化挙動を緩和するため、検索 token の損失マスキング戦略を導入した。その影響を評価するため、Qwen2.5-3b/7b-base モデルを用いて、検索 token の損失マスキングを用いる場合と用いない場合の学習ダイナミクスを比較する。[図 3](#figure-03)に示すように、マスキング機構を導入すると最適化が安定し、モデル性能が向上する。[表 6](#table-06)の定量的な結果からも、検索 token に損失マスキングを用いて学習した Search-R1 が、マスキングなしの変種を一貫して上回ることが確認できる。

<span id="figure-03"></span>

![図 3。検索 token の損失マスキングの検討。](../../papers/search-r1/figure-03.png)

**図 3。** 検索 token の損失マスキングの検討。

<span id="table-06"></span>

![表 6。検索 token の損失マスキングを用いる場合と用いない場合の Search-R1 の性能。](../../papers/search-r1/table-06.png)

**表 6。** 検索 token の損失マスキングを用いる場合と用いない場合の Search-R1 の性能。検索 token の損失マスキングを用いて学習した LLM は、一貫して高い性能を達成する。（RL：PPO）

<span id="appendix-e"></span>

## E Base LLM と Instruct LLM

Qwen2.5-3B と Qwen2.5-7B の 2 つのモデル規模を用い、base LLM と instruction-tuned LLM の両方について Search-R1 の学習ダイナミクスを調べる。[図 4](#figure-04)に示すように、instruction-tuned モデルは base モデルより速く収束し、初期性能も高い。初期の優位性にもかかわらず、学習後には両モデルの最終性能が同程度の水準へ収束する。これらの結果は、instruction tuning が推論と検索を組み合わせたタスクで初期学習を効率化する一方、強化学習は性能差を埋め、最終的に base モデルも同程度の結果へ到達できることを示す。

<span id="figure-04"></span>

![図 4。base LLM と instruct LLM における Search-R1 の検討。](../../papers/search-r1/figure-04.png)

**図 4。** base LLM と instruct LLM における Search-R1 の検討。instruction モデルは速く収束し、初期性能も高い。しかし、両モデルの最終性能は非常に近い。

<span id="appendix-f"></span>

## F Search-R1 における PPO と GRPO の比較

Qwen2.5-3B と Qwen2.5-7B を基礎モデルとして、PPO と GRPO の 2 つの強化学習アルゴリズムを用いた Search-R1 の有効性を評価する。[図 5](#figure-05)に学習ダイナミクスを示す。分析から次の重要な知見が得られる。**（1）すべての設定で GRPO は PPO より速く収束する。** これは PPO が別個の価値関数（critic）に依存し、有効なポリシー更新を始める前に初期 warm-up 段階を必要とするためである。**（2）PPO は学習挙動の安定性に優れる。** [図 5](#figure-05)に示すように、GRPO は学習ステップが長くなると報酬の崩壊に陥る一方、PPO は一貫して安定している。**（3）PPO と GRPO の最終報酬性能は同程度である。** これは、収束速度と安定性にトレードオフがあるものの、両手法とも Search-R1 を効果的に最適化できることを示す。

<span id="figure-05"></span>

![図 5。PPO と GRPO を基礎 RL 手法とした Search-R1 の学習ダイナミクス。](../../papers/search-r1/figure-05.png)

**図 5。** PPO と GRPO を基礎 RL 手法とした 4 つの LLM における Search-R1 の学習ダイナミクス。GRPO は一般に速く収束するが、一定のステップを学習すると不安定になる場合がある。一方、PPO は最適化がより安定しているが、収束は遅い。PPO と GRPO の最終報酬性能は同程度である。

<span id="appendix-g"></span>

## G Search-R1 学習時の検索パッセージ数の検討

検索パッセージ数（top-k）が Search-R1 の学習ダイナミクスに与える影響を調べる。主要実験では [Lin23c] に従って top-k = 3 としているが、その影響をより深く理解するため、top-k を 1、3、5 とする追加実験を行う。

[図 6](#figure-06)に各設定の学習報酬曲線を示す。3 つの設定はいずれも全体的な学習軌跡が近い。特に、top-k = 5 は初期収束が最も速く、最初の 200 ステップ以内に最高の学習報酬へ到達する。ただし、学習が進むにつれて報酬は徐々に低下し、不安定になる。これに対し、top-k = 1 と 3 は学習全体を通して一貫して改善し、最終的に top-k = 3 が 500 ステップ後の最高報酬を達成する。

500 ステップ時点の評価結果を[表 7](#table-07)にまとめる。top-k = 3 が全体として最高の性能を示す。これには 2 つの要因があると推測する。（1）top-k = 1 は検索の recall が低く、関連するコンテキスト情報の提供が制限される可能性が高い。（2）top-k = 5 ではノイズを含む、または無関係なパッセージ [Jin25e] が入るため precision が低下する。これは推論性能を低下させるだけでなく、RL の学習にも悪影響を与える可能性がある。追加コンテキストがしばしば役に立たず、誤解を招くとモデルが学習すると、検索内容の利用を避けるようになるためである。

<span id="figure-06"></span>

![図 6。異なる検索パッセージ数における Search-R1 の学習ダイナミクス。](../../papers/search-r1/figure-06.png)

**図 6。** 異なる検索パッセージ数における Search-R1 の学習ダイナミクス。（LLM：Qwen2.5-7b-base、RL：PPO）

<span id="table-07"></span>

![表 7。Search-R1 学習時の検索パッセージ数の検討。](../../papers/search-r1/table-07.png)

**表 7。** Search-R1 学習時の検索パッセージ数の検討。（LLM：Qwen2.5-7b-base、RL：PPO）

<span id="appendix-h"></span>

## H Search-R1（GRPO）学習時のグループサイズの検討

主要実験では、[She24a] の設定に従い、Search-R1（GRPO）のグループサイズを 5 に設定する。グループサイズが学習ダイナミクスに与える影響をさらに調べるため、グループサイズを 1、3、5 とした ablation study を行う。特に、グループサイズを 1 とすると、GRPO は標準的な REINFORCE アルゴリズム [Wil92] へ帰着する。

LLM を 500 ステップ学習し、モデルの checkpoint を 100 ステップごとに保存する。学習中にモデルが崩壊した場合は、最後の有効な checkpoint で評価し、それ以外の場合は 500 ステップ時点の checkpoint を評価する。

異なるグループサイズの設定における学習ダイナミクスを[図 7](#figure-07)に示す。一般に、グループサイズが大きいほど速く収束するが、強化学習に内在する不安定性により、崩壊のリスクも増える可能性がある。

各設定の評価結果を[表 8](#table-08)にまとめる。グループサイズを大きくすると収束が加速し、より高い学習報酬を達成できる一方、小さなグループサイズ（*たとえば* size = 1）では学習が安定し、汎化性能も高まる。このことは未知のタスクにおける優れた性能へ反映されており、GRPO の学習速度と安定性の間にあるトレードオフを示している。

<span id="figure-07"></span>

![図 7。異なるグループサイズにおける Search-R1（GRPO）の学習ダイナミクス。](../../papers/search-r1/figure-07.png)

**図 7。** 異なるグループサイズにおける Search-R1（GRPO）の学習ダイナミクス。（LLM：Qwen2.5-7b-base）

<span id="table-08"></span>

![表 8。Search-R1 のグループサイズの検討。](../../papers/search-r1/table-08.png)

**表 8。** Search-R1（GRPO）の 7 つのデータセットにおけるグループサイズの検討。（LLM：Qwen2.5-7b-base）

<span id="appendix-i"></span>

## I R1 と Search-R1 の比較：ケーススタディ

<span id="table-09"></span>

![表 9。R1 と Search-R1 のケーススタディ。](../../papers/search-r1/table-09.png)

**表 9。** R1 と Search-R1 のケーススタディ。

Search-R1 をより深く理解するため、Qwen2.5-7B-Base を用い、検索エンジンを用いない RL [Dee25c] と挙動を比較するケーススタディを行う。結果を[表 9](#table-09)に示す。そこから次の重要な知見が得られる。

**推論と検索を交互に行うと問題分析が強化される：** Search-R1 は LLM が複数ターンの検索を伴う詳細な推論を行えるようにする一方、検索を用いない RL はモデルの内部知識だけに依存する。検索したパッセージを組み込むことで、Search-R1 は推論を反復的に洗練し、より根拠のある正確な応答へ到達できる。

**反復検索による自己検証：** 2 回目の検索後、LLM はすでに質問へ回答するのに十分な情報を集めている。しかし Search-R1 は結論を自己検証するために追加の検索を行い、最終応答への確信をさらに強める。この現象は、検索を用いない LLM 推論 RL の知見 [Dee25c] と一致し、検索拡張設定でも RL が検証駆動の推論を促せることを示している。

<span id="appendix-j"></span>

## J Search-R1 の追加ケーススタディ

学習済み LLM の挙動と能力をさらに理解するため、Search-R1 の追加ケーススタディを行う。具体的には、Proximal Policy Optimization（PPO）を基礎となる強化学習アルゴリズムとして用い、Qwen2.5-7B-Base から微調整したモデルを分析する。結果を以下の表に示す。

<span id="table-10"></span>

![表 10。Search-R1 のケーススタディ 1。](../../papers/search-r1/table-10.png)

**表 10。** Search-R1 のケーススタディ 1（成功）：Search-R1 は複数ステップの推論と検索を行い、自己検証を経て最終的に質問へ回答する。

<span id="table-11"></span>

![表 11。Search-R1 のケーススタディ 2。](../../papers/search-r1/table-11.png)

**表 11。** Search-R1 のケーススタディ 2（失敗）：Search-R1 は複雑な問題を分解できない場合があり、無関係な検索パッセージに惑わされることもある。

<span id="table-12"></span>

![表 12。Search-R1 のケーススタディ 3。](../../papers/search-r1/table-12.png)

**表 12。** Search-R1 のケーススタディ 3（成功）：1 回の検索エンジン呼び出しで関連情報が見つかれば、Search-R1 は容易に質問へ回答できる。

<span id="table-13"></span>

![表 13。Search-R1 のケーススタディ 4。](../../papers/search-r1/table-13.png)

**表 13。** Search-R1 のケーススタディ 4（成功）：Search-R1 は適切なクエリを書き、それまでの検索エンジン呼び出しで得られなかった補助情報を検索できる。

<span id="table-14"></span>

![表 14。Search-R1 のケーススタディ 5。](../../papers/search-r1/table-14.png)

**表 14。** Search-R1 のケーススタディ 5（失敗）：取得情報が不十分、または誤解を招く場合、Search-R1 は質問に答えられない。

<span id="table-15"></span>

![表 15。Search-R1 のケーススタディ 6。](../../papers/search-r1/table-15.png)

**表 15。** Search-R1 のケーススタディ 6（成功）：十分かつ正確なコンテキストを取得できれば、Search-R1 はマルチホップ推論によって容易に質問へ回答できる。

<span id="table-16"></span>

![表 16。Search-R1 のケーススタディ 7。](../../papers/search-r1/table-16.png)

**表 16。** Search-R1 のケーススタディ 7（失敗）：Search-R1 は当初、複雑な問題を分解するための適切なクエリを書けなかった。モデルは十分な根拠を得ずに質問へ回答した。

<span id="table-17"></span>

![表 17。Search-R1 のケーススタディ 8。](../../papers/search-r1/table-17.png)

**表 17。** Search-R1 のケーススタディ 8（成功）：Search-R1 は不足している情報を検索するクエリを書ける。

<span id="table-18"></span>

![表 18。Search-R1 のケーススタディ 9。](../../papers/search-r1/table-18.png)

**表 18。** Search-R1 のケーススタディ 9（成功）：LLM が最初に書いたクエリはあまり意味をなさない。しかし、それを受けて LLM はクエリを書き直し、問題を段階的に解き始める。

<span id="table-19"></span>

![表 19。Search-R1 のケーススタディ 10。](../../papers/search-r1/table-19.png)

**表 19。** Search-R1 のケーススタディ 10（成功）：外部知識源だけでは質問へ回答できないと判断すると、Search-R1 は検索を止めることを学習する。

<span id="table-20"></span>

![表 20。Search-R1 のケーススタディ 11。](../../papers/search-r1/table-20.png)

**表 20。** Search-R1 のケーススタディ 11（失敗）：LLM は無関係な取得情報に惑わされ、誤った回答を返す場合がある。
