---
title: 'RAGEN: Multi-Turn Agent RL'
createTime: 2026/08/21 09:47:30
permalink: /ja/papers/ragen-multi-turn-agent-rl/
---

> [Zihan Wang](https://zihanwang314.github.io/), [Kangrui Wang](https://jameskrw.github.io/), [Qineng Wang](https://qinengwang-aiden.github.io/), [Pingyue Zhang](https://williamzhangsjtu.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Zhengyuan Yang](https://zyang-ur.github.io/), [Xing Jin](https://scholar.google.com/citations?user=vzp-yAgAAAAJ&hl=en), [Kefan Yu](https://huangtubaye233.github.io/), [Minh Nhat Nguyen](https://scholar.google.com/citations?user=lRG8dTEAAAAJ&hl=en), [Licheng Liu](https://lichengliu03.github.io/), [Eli Gottlieb](https://www.linkedin.com/in/eli-gottlieb1/), [Yiping Lu](https://2prime.github.io/), [Kyunghyun Cho](https://kyunghyuncho.me/), [Jiajun Wu](https://jiajunwu.com/), [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/), [Manling Li](https://limanling.github.io/)。論文は 2025 年 4 月 24 日に arXiv へ初回投稿され、現行版は v2 である。[RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073v2)。[原論文 PDF](/paper/ragen-multi-turn-agent-rl.pdf)。[DOI](https://doi.org/10.48550/arXiv.2504.20073)。[TeX ソース](https://arxiv.org/src/2504.20073v2)。厳密な印刷レイアウトと参考文献については原論文 PDF を正本とする。

## 要旨

大規模言語モデル（LLM）を対話型エージェントとして学習させる場合、長期的な意思決定や確率的な環境フィードバックとの相互作用など、固有の課題が生じる。強化学習（RL）は静的タスクを進展させてきたが、multi-turn agent RL の学習は十分に研究されていない。本稿では、trajectory-level agent RL の汎用フレームワーク **StarPO**（**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization）を提案し、LLM エージェントの学習と評価を行うモジュール型システム **RAGEN** を導入する。4 つの定型化した環境での研究から、3 つの主要な知見を得た。第 1 に、agent RL の学習では報酬の変動性が急落し、勾配がスパイクする **Echo Trap** が繰り返し現れる。この問題に対して、trajectory filtering、critic の導入、勾配安定化を組み込んだ安定版 **StarPO-S** を提案する。第 2 に、RL rollout の構成には、**多様な初期状態、中程度の相互作用粒度、より頻繁な sampling** が有効である。第 3 に、**きめ細かく reasoning-aware な報酬信号**がなければ、エージェントの推論は multi-turn RL からほとんど創発せず、浅い戦略や事実に反する思考を示す場合がある。

**キーワード**：LLM Agents、Multi-turn RL <br>**Web サイト**：[https://ragen-ai.github.io/](https://ragen-ai.github.io/) <br>**コード／環境**：[https://github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN)。

<span id="figure-01"></span>

![図 1。従来手法は、数学やコード生成などの非対話タスクを中心に扱う。RAGEN は、multi-turn の確率的相互作用を必要とするエージェントタスクにおいて、multi-turn rollout、trajectory-level の報酬割り当て、policy update を支援する汎用 agent RL フレームワーク StarPO を実装する。](../../papers/ragen-multi-turn-agent-rl/figure-01.png)

**図 1。** 従来手法は、数学やコード生成などの非対話タスクを中心に扱う。**RAGEN** は、multi-turn の確率的相互作用を必要とするエージェントタスクにおいて、multi-turn rollout、trajectory-level の報酬割り当て、policy update を支援する汎用 agent RL フレームワーク StarPO を実装する。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）を対話環境で自律エージェントとして機能させるための学習には、固有の課題がある。single-turn の数学問題 [Sha24d] やコード生成 [Dee24b] などの静的タスクとは異なり、エージェントは逐次的に意思決定し、turn をまたいで記憶を保持し、環境からの確率的フィードバックに適応しなければならない。planning assistant、robotics、tutoring agent の中核となるこれらの設定では、モデルは高い性能を示すだけでなく、経験を通じて自己改善する必要がある。

近年、rule-based reward を用いた LLM の reinforcement learning（RL）[Dee25c, Ope24i, Pan25c, Zen25d, Fau24, Gao24f] が研究されているが、rule-based RL によって推論と適応を自己進化させる対話型**エージェント**の学習は、依然として十分に検討されていない。特に LLM agent の学習では、学習の不安定性、複雑な報酬信号、環境変化に対する限られた汎化がしばしば見られ、確率的フィードバックを伴う multi-turn interaction では顕著になる。ここで重要な未解決問題は、*自己進化する LLM エージェントを効果的かつ安定して学習させる設計要因は何か*、という点である。

この問題を調べるため、汎用 RL フレームワーク **StarPO**（**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization）の下で、エージェント学習を体系的に研究する。StarPO は、推論、報酬割り当て、prompt-rollout 構造を柔軟に制御しつつ、**multi-turn、trajectory-level の agent training** を統一的に捉える。StarPO を基盤として、RL ベースの LLM agent training を研究するためのモジュール型学習・評価システム **RAGEN** を開発する。RAGEN は rollout generation、reward assignment、trajectory optimization を含む完全な学習ループを実装し、multi-turn かつ確率的な環境における LLM agent training dynamics を体系的に分析する研究基盤となる。

Web browsing のような実世界タスクで LLM agent を学習させる場合、pretraining prior と大規模な task-specific engineering に依存することが多い。本研究では、複雑さの異なる 4 つの環境で RAGEN を評価する。**Bandit**（single-turn、確率的）、**Sokoban**（multi-turn、決定論的）、**Frozen Lake**（multi-turn、確率的）、**WebShop**（multi-turn、open-domain）である。最初の 3 つの記号環境は**最小限で完全に制御可能**であり、WebShop は**実世界の理解と推論**を加える。これらを組み合わせることで、異なる意思決定課題に対する汎化を分析できる。

この設定を使って agent learning の 3 つの主要側面を分析し、安定した agent RL training の主要課題と設計原則を示す知見を以下にまとめる。

- **Multi-turn RL における勾配安定性が、安定した学習を左右する。** **Multi-turn RL training** は、局所的に報酬を得た推論パターンへエージェントが過適合する **Echo Trap** をしばしば引き起こす。これは reward variability collapse、entropy drop、gradient spike を伴う。この失敗を緩和するため、variability-based trajectory filtering、critic baseline、decoupled clipping により学習の頑健性を高めた **StarPO-S** を提案する。
- **Rollout の頻度と多様性が自己進化を形作る。** RL ベースの agent training では、LLM が自己生成した rollout trajectory が主要な学習材料となる。安定した agent RL training に必要な rollout の要因は、（1）**多様な初期状態**から rollout を生成し、**各初期状態に複数の response** を持たせること、（2）固定した turn 上限内で interaction horizon を伸ばすため、**各 turn で複数の action を実行**すること、（3）online feedback が現在の policy を反映するよう、**高い rollout frequency** を保つことである。
- **エージェント推論の創発には綿密な報酬信号が必要である。** Action format で推論を促すだけでは、推論行動は保証されない。StarPO による trajectory-level optimization と「**&lt;think&gt;**」token を用いた prompt があっても、推論に明確な報酬上の利点がなければ、モデルは直接 action selection に戻ることが多い。MDP の action space が単純で、浅い戦略でも十分なためだと考えられる。さらに、報酬がタスク成功だけを反映する場合、モデルは**事実に反する推論**を生成し、思考と環境状態にずれが生じる。Long-horizon agent training には、**きめ細かい reasoning-aware reward design** が必要である。

<span id="figure-02"></span>

![図 2。State-Thinking-Actions-Reward Policy Optimization（StarPO）フレームワーク。LLM は multi-turn の環境相互作用に対して推論に基づく action を生成し、trajectory-level reward を蓄積する。報酬を正規化し、LLM policy の更新に用いる。](../../papers/ragen-multi-turn-agent-rl/figure-02.png)

**図 2。** State-Thinking-Actions-Reward Policy Optimization（StarPO）フレームワーク。LLM は multi-turn の環境相互作用に対して推論に基づく action を生成し、trajectory-level reward を蓄積する。報酬を正規化し、LLM policy の更新に用いる。

<span id="section-2"></span>

## 2 フレームワーク

<span id="section-2-1"></span>

### 2.1 エージェント学習の MDP 定式化

従来の language model 向け reinforcement learning（RL）は single-turn setting を仮定することが多く、dataset $\mathcal{D}$ から sampling した prompt-response pair $(s,a)$ に対する expected reward $R(s,a)$ の最大化を目的とする。

<span id="equation-01"></span>

$$
J_{\mathrm{step}}(\theta)=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_{\theta}(\cdot|s)}[R(s,a)].
$$

一方、LLM-based agent は複数 turn にわたり展開し、確率的 feedback を示す対話環境で動作する必要がある。これらの dynamics を表すため、問題を Markov Decision Process（MDP）$\mathcal{M}=\{S,A,P\}$ として定式化する。$S$ は state（observation sequence や interaction history など）、$A$ は action（多くの場合 token sequence）、$P$ は transition dynamics と reward generation process を表す。時刻 $t$ で、agent policy $\pi_{\theta}$ は現在の state $s_{t}$ と interaction history に条件付けて action $a_{t}$ を生成する。環境は現在の transition dynamics に従い、reward $r_{t}$ と新しい state $s_{t+1}$ を返す。

$$
a_{t}\sim\pi_{\theta}(\cdot|s_{t},\tau_{<t}),\quad(r_{t},s_{t+1})\sim P(\cdot|s_{t},a_{t}),
$$

ここで $\tau_{<t}=\{s_{0},a_{0},r_{0},...,s_{t-1},a_{t-1},r_{t-1}\}$ は interaction history である。この対話過程は最大 horizon $K$ まで続き、agent の学習材料となる完全な trajectory $\tau=\{s_{0},a_{0},r_{0},...,s_{K}\}$ を生成する。

<span id="section-2-2"></span>

### 2.2 StarPO：Trajectory-Level Optimization による推論の強化

LLM agent の multi-turn interaction trajectory 全体を最適化する汎用 RL フレームワーク **StarPO**（State-Thinking-Action-Reward Policy Optimization）を導入する。各 action を独立に扱う静的タスク向け手法とは異なり、StarPO は observation、reasoning trace、action、feedback を含む**trajectory 全体**を、rollout と model optimization の一貫した単位として扱う。目的は expected trajectory reward の最大化である。

<span id="equation-02"></span>

$$
J_{\mathrm{StarPO}}(\theta)=\mathbb{E}_{\mathcal{M},\tau\sim\pi_{\theta}}\left[R(\tau)\right],
$$

ここで $\mathcal{M}$ は MDP、$\tau$ は推論を含む完全な interaction sequence、$R(\tau)$ は trajectory 全体の cumulative reward である。Policy probability $\pi_{\theta}(\tau)$ は token-level likelihood に分解できるため、StarPO は autoregressive LLM と直接互換性を持つ。[図 2](#figure-02) に StarPO の全過程を示し、以下で各要素を詳しく説明する。

##### StarPO と従来手法の Trajectory-Level Objective の比較

**従来手法（PPO [Sch17a]、GRPO [Sha24d] など）：**

$$
J_{\mathrm{step}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\left[R(x,y)\right]
\quad
\text{（入力 }x\text{ に対する single-turn output }y\text{ を最適化）}
$$

**StarPO（本手法）：**

$$
J_{\mathrm{StarPO}}(\theta)=\mathbb{E}_{\mathcal{M},\tau\sim\pi_\theta}\left[R(\tau)\right]
\quad
\text{（trajectory }\tau=\{s_0,a_0,r_0\dots,s_K\}\text{ 全体の報酬を最適化）}
$$

<span id="section-2-2-2"></span>

#### 2.2.2 最適化手順：推論・相互作用 trajectory からの学習

各 training iteration で、エージェントは initial state $s_{0}$ から始まり、$N$ 本の trajectory を生成する。各 step $t$ において、推論に基づく structured output を生成する。

<span id="equation-03"></span>

$$
a^T_{t}=\texttt{<think>}...\texttt{</think><answer>}\,a_{t}\,\texttt{</answer>},
$$

ここで $a^T_{t}$ は中間推論を含む完全な action output、$a_{t}$ は環境が実行可能な sub-action sequence である。環境は次の state $s_{t+1}$ と reward $r_{t}$ を返す。Rollout stage は完全な trajectory $\tau=\{s_{0},a^T_{0},r_{0},s_{1},...,a^T_{K-1},r_{K-1},s_{K}\}$ を生成する。*すべての要素は LLM によって生成されるか、環境によって生じる*ものであり、まとめて最適化される。

StarPO は rollout step と update step を交互に実行する。新しい rollout は $\pi_{\theta}$ を使って on-policy で生成するか、$\pi_{\text{old}}$ に基づく replay buffer から sampling できる。各 training loop は $P$ 個の initial state $s_{0}$ を持ち、それぞれが $N$ 本の trajectory を生成し、batch size $E$ で update する。この loop を合計 $L$ 回実行すると、gradient update の総数は $S=\frac{L\cdot P\cdot N}{E}$ となる。その他の training mechanism は[第 3 節](#section-3)で述べる。

<span id="section-2-2-3"></span>

#### 2.2.3 モジュール型最適化戦略

StarPO は、統一された trajectory-level abstraction の下で複数の policy optimization algorithm を支援する。Rollout trajectory $\tau_{i}=\{\tau_{i,(1)},\ldots,\tau_{i,(|\tau_{i}|)}\}$ が合計 $|\tau_{i}|$ token を持つとき、次の optimization strategy を用いて StarPO を具体化し、token-level update を行う。

- **PPO [Sch17a]。** PPO objective（詳細は[付録 A](#appendix-a)）を使い、critic を学習して token-level value と advantage $A_{i,t}$ を推定する。

<span id="equation-04"></span>

$$
J_{\mathrm{PPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
\min\left[
\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})}\cdot A_{i,t},\,
\mathrm{clip}\left(\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})},1-\varepsilon,1+\varepsilon\right)\cdot A_{i,t}
\right],
$$

ここで $G$ は batch 内の trajectory 数、$\tau_{i,(t)}$ は trajectory $\tau_i$ の第 $t$ token、$\tau_{i,<t}$ はその prefix である。

- **GRPO [Sha24d]。** Critic-free の GRPO training では、各 trajectory に scalar reward $R(\tau_i)$ を割り当て、$\tau_i$ 内の全 token に normalized advantage $\hat{A}_{i,t}$ を与える。

<span id="equation-05"></span>

$$
\hat{A}_{i,t}=\frac{R(\tau_i)-\mathrm{mean}(\{R(\tau_1),\ldots,R(\tau_G)\})}{\mathrm{std}(\{R(\tau_1),\ldots,R(\tau_G)\})}.
$$

GRPO objective は次のようになる。

<span id="equation-06"></span>

$$
J_{\mathrm{GRPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
\min\left[
\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})}\cdot\hat{A}_{i,t},\,
\mathrm{clip}\left(\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})},1-\varepsilon,1+\varepsilon\right)\cdot\hat{A}_{i,t}
\right].
$$

<span id="section-2-3"></span>

### 2.3 RAGEN システム

StarPO を実用化するため、制御された環境で LLM agent を学習させる完全なシステム **RAGEN** を構築する。RAGEN は structured rollout、customizable reward function、multi-turn かつ確率的な環境との統合を支援する。StarPO の execution backend であると同時に、reasoning agent training の stability、generalization、learning dynamics を研究する platform でもある。RAGEN は拡張可能に設計されており、新しい環境、reward scheme、rollout strategy を容易に追加できるため、RL-based agent training の基盤となる。

<span id="section-3"></span>

## 3 実験設定

<span id="section-3-1"></span>

### 3.1 環境とタスク

記号的な意思決定から現実的な意思決定までを含む 4 つの環境で LLM agent を評価する。**Bandit** は noisy feedback の下で risk-sensitive reasoning を試し、**Sokoban** は不可逆な symbolic planning を要求し、**Frozen Lake** は planning と probabilistic transition を組み合わせる。**WebShop** は natural language grounding と web environment interaction を扱う。最初の 3 つの記号環境は、明瞭な分析のため意図的に最小限かつ完全に制御可能にしている。WebShop は現実的な task structure と language input を導入する。環境の可視化は付録 C.1 に示す。

<span id="section-3-2"></span>

### 3.2 学習設定

主実験では、3 つの記号タスクに Qwen-2.5 Instruct 0.5B model、難しい WebShop には 3B variant を用いる。付録 [D](#appendix-d) では各種 model performance も報告する。Model は H100 GPU 上で StarPO variant を使い、100-200 回の rollout-update iteration にわたり学習する。各 batch は $P{=}8$ prompt を sampling し、各 prompt につき $N{=}16$ rollout、最大 5 turn、10 action とする。Policy update は GAE（$\gamma{=}1.0,\lambda{=}1.0$）付き GRPO または PPO、Adam optimizer、entropy bonus（$\beta{=}0.001$）、response-format penalty（$-0.1$）を用いる。詳細は付録 C.2 に示す。

<span id="section-3-3"></span>

### 3.3 評価指標

各環境につき固定した 256 prompt を temperature $T{=}0.5$ で評価し、episode は 5 turn 後に打ち切る。指標は、**（i）** success rate（task completion）、**（ii）** rollout entropy（exploration）、**（iii）** in-group reward variability（behavioral diversity）、**（iv）** response length（reasoning verbosity）、**（v）** gradient norm（training stability）である。すべて validation instance 上で計算する。詳細は付録 C.3 に示す。

<span id="section-4"></span>

## 4 実験結果と知見

<span id="section-4-1"></span>

### 4.1 Multi-turn Agent RL Training が新たな不安定性をもたらす

各 agent task で baseline StarPO を評価する（[図 3](#figure-03)）。Bandit や Sokoban などの記号環境では初期に改善するが、最終的には collapse する。これらの環境では PPO が GRPO より安定し、collapse が遅く、性能も高い。Critic がより滑らかな reward estimate を与えるためだと考えられる。興味深いことに Frozen Lake では GRPO の方が安定する。このタスクでは state value estimation が難しく、PPO が不安定になる可能性がある（付録 [I](#appendix-i)）。WebShop では両手法とも成功する。強い language prior と高い initial reward により、critic の必要性が小さいためだと考えられる。

<span id="figure-03"></span>

![図 3。各環境での baseline StarPO performance。Bandit や Sokoban などの記号タスクは collapse する一方、実世界の WebShop は高い値から始まり急速に改善する。PPO は token-level reward signal を安定させるため Bandit と Sokoban で強い。GRPO は、ランダム性により state value を推定しにくい Frozen Lake と、初期性能が高く critic による勾配安定化の必要性が低い WebShop で強い。](../../papers/ragen-multi-turn-agent-rl/figure-03.png)

**図 3。** **各環境での baseline StarPO performance。** Bandit や Sokoban などの記号タスクは collapse する一方、実世界の WebShop は高い値から始まり急速に改善する。PPO は token-level reward signal を安定させるため Bandit と Sokoban で強い。GRPO は、ランダム性により state value を推定しにくい Frozen Lake と、初期性能が高く critic による勾配安定化の必要性が低い WebShop で強い。

Collapse の原因を理解するため、初期と後期の trajectory を比較する。Bandit task では、初期の trajectory は記号的意味と expected reward について多様な推論を示すが、後期の response は反復的かつ決定論的になる。これは、**RL training がモデル固有の推論 shortcut を過度に増幅した**可能性を示す。局所的に報酬を得た template を強化し、exploration を抑制している。この failure mode を「**Echo Trap**」と呼ぶ。自己生成 trajectory で学習したモデルが記憶した reasoning path を繰り返し再利用し、多様性の collapse と長期性能の低下を引き起こす [Shu24] の知見に類似する。例は付録 [F](#appendix-f) に示す。

Collapse の検出には、2 つの主要指標を監視する。（1）**Average Reward** は plateau や低下が task performance の劣化を示し、（2）**Gradient Norm** の spike は不安定な update を示す。早期兆候を捉えるため、rollout-level signal も追跡する。（1）**Reward Standard Deviation** は policy が多様な outcome を生成しているか、反復行動に collapse しているかを示し、（2）**Output Entropy** は model prediction の決定性を示す。

<span id="figure-04"></span>

![図 4。Multi-turn RL における collapse indicator と early warning signal。Average reward と gradient norm（左側）は collapse を直接反映し、plateau と spike が性能および学習の不安定性を確認する。Reward standard deviation と entropy（右側）は reward が低下する前に不安定になることが多く、早期警告として機能する。](../../papers/ragen-multi-turn-agent-rl/figure-04.png)

**図 4。** **Multi-turn RL における collapse indicator と early warning signal。** Average reward と gradient norm（左側）は collapse を直接反映し、plateau と spike が性能および学習の不安定性を確認する。Reward standard deviation と entropy（右側）は reward が低下する前に不安定になることが多く、早期警告として機能する。

[図 4](#figure-04) は、task と optimization method ごとの dynamics をまとめている。結果から、**multi-turn agent RL で model collapse がどのように発生するか**について次の結論を得る。

- **Reward standard deviation は収束の早期指標である。** FrozenLake-PPO では std が step 40 で急低下し、性能がほぼ最適な step 90 で reward mean が collapse するよりかなり早い。Bandit-PPO では std が step 70 付近で底を打ち、step 120 の reward peak に先行する。Sokoban-PPO では std と mean が step 10 付近で同時に collapse し、早期の飽和を示す。
- **Gradient norm spike は不可逆な collapse を示す。** Bandit の step 170、Sokoban の 110、FrozenLake の 90 などで gradient norm spike が現れると、小さな parameter update でも loss が大きく変化し、その後の回復は難しくなる。
- **有効な学習中、entropy は安定した減衰傾向を示すべきである。** FrozenLake-GRPO にこの傾向が見られる。Bandit と Sokoban の GRPO のように、entropy が急上昇したり不規則に変化したりする場合、reasoning behavior の collapse と相関することが多い。

これらの pattern は、multi-turn RL が single-turn RL method では扱えない固有の課題をもたらすことを示す。そこで sampling quality、gradient stability、exploration regularization を対象として早期 collapse を防ぐ安定版 **StarPO-S** を導入する。

<span id="section-4-2"></span>

### 4.2 StarPO-S：Instance Filtering と Gradient Shaping による Multi-turn RL の安定化

Multi-turn reinforcement learning の不安定性に対処するため、学習の頑健性と効率を高める 3 つの変更を取り入れた StarPO の安定版 **StarPO-S** を導入する。Reward standard deviation の低下が collapse に先行するという知見から、次の問いを検討する。*行動がより不確実で reward variability が高い task instance を、より集中的に学習すべきだろうか*。

最も有効な training sample は、agent が**結果の不確実性を示す**もの、すなわち自明な instance と極端に難しい instance の両方を避けたものだと仮定する。この直観は、不確実な example が model にとって最も情報量が多いとする Active Learning [Set09] の原理に基づく。特定の agent task instance（MDP $\mathcal{M}=\{S,A,P\}$ の initial state $s_{0}$）に対する policy $\pi_{\theta}$ の trajectory-level outcome uncertainty $U$ を次のように定義する。

<span id="equation-07"></span>

$$
\mathrm{U}(\pi_{\theta},\mathcal{M},s_{0})=\mathrm{Std}_{\tau\sim\pi_{\theta}(\cdot|s_{0})}\left[R(\tau)\right].
$$

Training 中、repeated rollout から得た reward の standard deviation に基づいて prompt を並べ、各 training step で**不確実性が高い上位 $p$% の prompt だけを残す**。[図 5](#figure-05) は、StarPO-S の PPO と GRPO で $p$ を変えた効果を示す。Uncertainty-based filtering の効果は付録 [E](#appendix-e) でさらに検証する。

PPO run（[図 5](#figure-05) 左）では、low-variability rollout の filtering が collapse を大幅に遅らせる。Rollout の 75% を残すと FrozenLake の安定期間は 100 step から 140 step に延び、50% では collapse を完全に回避する。GRPO は critic-free design のため安定性が低いままだが、一定の改善がある。Filtering は効率も改善する（[図 5](#figure-05) 右）。StarPO-S の既定値には 25% を採用する。ただし、この積極的な値がすべての状況で最適とは限らない。Sokoban と FrozenLake は積極的 filtering によく反応する。比較的反復的な reasoning pattern を持ち、pretraining で十分に表現されていないため、類似 trajectory が batch を占めると collapse しやすい可能性がある。付録 [D](#appendix-d) では、model performance の位置付けを明確にするため、より大きな model（72B）と GPT-4o、Qwen-2.5-72B などの frontier model も示す。

Uncertainty-based filtering に加え、single-turn RL 向けの DAPO [Yu25g] から着想を得た 2 つの gradient shaping technique、**KL Term Removal** と **Clip-Higher**（Asymmetric Clipping）を採用する。Multi-turn agent setting に拡張して評価した結果、どちらも success rate を高め、安定した学習期間を延ばした。より柔軟な gradient shaping が multi-turn RL に有効である。設計詳細と performance ablation は付録 [D](#appendix-d) に示す。

<span id="figure-05"></span>

![図 5。Uncertainty-based filtering が multi-turn RL の安定性に与える影響。Low-variability trajectory の filtering は collapse risk を減らし、success rate を改善する。PPO variant では trajectory の半数以上を filtering すると collapse がほぼ抑制される。Training time も短縮される。](../../papers/ragen-multi-turn-agent-rl/figure-05.png)

**図 5。** **Uncertainty-based filtering が multi-turn RL の安定性に与える影響。Low-variability trajectory の filtering は collapse risk を減らし、success rate を改善する。PPO variant では trajectory の半数以上を filtering すると collapse がほぼ抑制される。Training time も短縮される。**

**全体比較。** [図 6](#figure-06) で 3 task における StarPO-S と vanilla StarPO を比較する。StarPO-S は一貫して collapse を遅らせ、最終 task performance を高める。この改善は、uncertainty filtering による選択的な training data と、KL removal および decoupled clipping による均衡した optimization signal が、reasoning mode の狭まりを減らすためだと考える。付録 [D](#appendix-d) では、selective response mask や Bi-level General Advantage Estimation（GAE）[Wan25ak] など、学習を安定化し性能を高める可能性のある variant も議論する。

<span id="figure-06"></span>

![図 6。StarPO-S は task 全体の安定性と最終性能を改善する。Vanilla StarPO と比べ、StarPO-S は 4 task すべてで collapse を緩和し、より高い success rate を達成する。](../../papers/ragen-multi-turn-agent-rl/figure-06.png)

**図 6。** **StarPO-S は task 全体の安定性と最終性能を改善する。Vanilla StarPO と比べ、StarPO-S は 4 task すべてで collapse を緩和し、より高い success rate を達成する。**

<span id="section-4-3"></span>

### 4.3 RL Training に有用な Trajectory の生成

RL training は trajectory quality に大きく依存する。Vanilla Sokoban で学習し、SokobanNewVocab、LargeSokoban、FrozenLake Task などで評価することで、rollout の 3 つの主要側面、*task diversity*、*interaction granularity*、*rollout frequency* を研究する。詳細は付録 [K](#appendix-k) に示す。

**Response comparison を伴う高い task diversity は汎化を改善する。** Task diversity とは、各 rollout-update cycle で使う異なる prompt の数である。Batch size が一定なら、prompt 当たりの response 数と trade-off になる。実験（[表 1](#table-01)）でこの trade-off を変えたところ、prompt 当たりの response を減らして task diversity を高める（例えば 4 response／prompt）と、一貫して汎化が改善した。ただし各 prompt に複数 rollout が含まれ、同様の条件下で agent が異なる outcome を比較できる場合に限られる。

**Action budget を増やすと planning が可能になるが、過度に long-horizon な rollout は noise を加える。** [表 2](#table-02) では turn 当たりの action 数を変更する。Turn 当たり最大 5 または 6 action で最も良い性能を得て、SokobanNewVocab や LargeSokoban など複雑な環境で特に顕著である。この設定は planning に十分な余地を与えながら、過度に長い rollout の混乱を避ける。Budget を 7 action に増やすと性能が低下する。Noisy transition と reward feedback の希薄化が原因と考えられる。

<span id="table-01"></span>

![表 1。Task diversity が generalization performance（%）に与える影響。複数 response を使う場合、diversity が高い設定（4 response／prompt）が最も良い。](../../papers/ragen-multi-turn-agent-rl/table-01.png)

**表 1。** **Task diversity が generalization performance（%）に与える影響。** 複数 response を使う場合、diversity が高い設定（4 response／prompt）が最も良い。

<span id="table-02"></span>

![表 2。Turn 当たりの action budget を変えたときの各環境の performance（%）。Turn 当たり 5-6 action が最も良く、効果的な multi-step planning との均衡が取れている。](../../papers/ragen-multi-turn-agent-rl/table-02.png)

**表 2。** **Turn 当たりの action budget を変えたときの各環境の performance（%）。** Turn 当たり 5-6 action が最も良く、効果的な multi-step planning との均衡が取れている。

<span id="table-03"></span>

![表 3。StarPO-S における reasoning 有り／無しの generalization performance（%）。Reasoning を無効にすると single-turn Bandit の汎化は大幅に低下するが、multi-turn Sokoban では効果が一様でないか小さい。](../../papers/ragen-multi-turn-agent-rl/table-03.png)

**表 3。** **StarPO-S における reasoning 有り／無しの generalization performance（%）。** Reasoning を無効にすると single-turn Bandit の汎化は大幅に低下するが、multi-turn Sokoban では効果が一様でないか小さい。

**頻繁な rollout update は optimization target と現在の policy behavior を整合させる。** Rollout freshness の効果を調べるため、1 組の rollout を $k$ 回連続の policy update で再利用する *Online-$k$* rollout strategy を採用する。$k$ が小さいほど rollout collection が頻繁になる。*Online-1* は完全な online setting に相当し、各 update iteration で新しい rollout を収集する。[図 7](#figure-07) に示すように、新しい rollout（*Online-1*）で学習した agent は、update を遅延させた場合（*Online-5* や *Online-10*）より速く収束し、task 全体で汎化も良い。これは multi-turn RL の中心的設計原則、すなわち trajectory が agent の最新 behavior を反映するとき学習が最も有効になることを支持する。頻繁な rollout は policy-data mismatch を減らし、optimization stability を高める。

<span id="section-4-4"></span>

### 4.4 推論は汎化を改善するが、Fine-Grained Reward がない Multi-turn Setting では消失する

Symbolic reasoning が agent generalization に与える影響を調べる。Reasoning は Bandit のような single-turn task では性能を高めるが、Sokoban のような複雑な multi-turn environment では成長せず、維持もされない。以下で順に分析する。

**Reasoning trace は single-turn Bandit task の汎化を改善する。** Symbolic Bandit environment で制御された generalization test を設計する。元の `Bandit` setting では `[Teacher, Engineer]` arm pair で model を学習し、`[Librarian, Trader]` で評価する。直感的な risk-reward alignment、すなわち `Engineer` と `Trader` が high-risk、high-reward である関係は保つ。`BanditRev` ではこの対応を反転し、職業に反直感的な reward profile を割り当て、推論を難しくする。

[表 3](#table-03) に示すように、reasoning trace で学習した model は `Bandit` でよりよく汎化し、反直感的な `BanditRev` でも同様である。Reasoning supervision が単なる記憶を越えて symbolic cue を内部化させることを示す。`BanditRev` の難しさが増しても、explicit reasoning を持つ model は持たない model を一貫して上回る（[表 3](#table-03)）。Semantic-reward misalignment の下でも、reasoning trace は agent が symbolic-reward association を内部化し、表面的な memorization を越えて汎化する助けになる。

**Multi-turn task では、学習とともに reasoning signal が薄れる。** Single-turn setting と異なり、Sokoban や FrozenLake などの multi-turn environment では reasoning の利点が限られる。Output format に明示的な `<think>` segment があっても、それを削除した no-think variant が同等か、より良い性能を示すことが多い。この退化を理解するため、学習中の average response length（[表 4](#table-04)、[図 14](#figure-14)）を分析すると、reasoning trace は時間とともに一貫して短くなり、model が自らの思考過程を抑制していることが分かる。Reasoning が不可欠な semantically misaligned `BanditRev` では trace が長く保たれ、context が難しいほど reasoning が維持されやすい。

Reasoning collapse は、**multi-turn task の sparse、delayed reward structure** に起因する可能性がある。この構造は coherent reasoning と trial-and-error による成功を区別できないことが多い。付録 [L](#appendix-l) の例では、model は incoherent または hallucinated reasoning を生成しても高い reward を得ており、この見方を支持する。ここから重要な問いが生じる。*報酬だけでは reasoning quality を反映できない場合、どのように有用な reasoning を一貫して強化できるか*。可能な方法の 1 つは、format-based penalty を使って action correctness と reasoning quality を分離することである。[Sha24d] と同様に、有効な `<think>`-`<answer>` 構造を欠く output に小さな penalty を課し、structured reasoning を促す。今後は partial correctness への reward など、より fine-grained な reward design によって long-horizon decision-making の reasoning を確実に強化できる可能性がある。

<span id="table-04"></span>

![表 4。学習 step ごとの reasoning length（`<think>` block length）。Token length は時間とともに概して減少するが、`ReverseBandit` のような文脈的に難しい問題は元の問題より多くの reasoning を必要とする。](../../papers/ragen-multi-turn-agent-rl/table-04.png)

**表 4。** **学習 step ごとの reasoning length（`<think>` block length）。** Token length は時間とともに概して減少するが、`ReverseBandit` のような文脈的に難しい問題は元の問題より多くの reasoning を必要とする。

<span id="figure-07"></span>

![図 7。異なる rollout frequency（Online-k）での performance。各 batch を k 回の policy update に再利用する rollout reuse factor k を変更する。k が小さい場合（Online-1 など）は rollout がより頻繁になる。新しい data は current policy と整合するため、convergence を改善する。](../../papers/ragen-multi-turn-agent-rl/figure-07.png)

**図 7。** **異なる rollout frequency（*Online-$k$*）での performance。各 batch を $k$ 回の policy update に再利用する rollout reuse factor $k$ を変更する。$k$ が小さい場合（*Online-1* など）は rollout がより頻繁になる。新しい data は current policy と整合するため、convergence を改善する。**

<span id="section-5"></span>

## 5 関連研究

近年、**reinforcement learning（RL）** による multi-step reasoning のための LLM fine-tuning と、decision-making task を構造化する agent framework の開発が進められている。Reasoning technique は、古典的 PPO [Sch17a] や actor-critic method [Haa18] から、meta token による structured prompting [Goy24, Her24] まで幅広い。RLOO [Koo19]、GRPO [Dee25c]、DAPO [Yu25g] などの policy variant は、training を安定させ sample efficiency を改善しうる。STaR [Zel22b] や MCTS-based reasoning [Hao23] などの研究は、少ない supervision で step-by-step reasoning を促進する。

**エージェント側。** System は初期の reactive planning [Yao23b, Xu23b] から、modular decision pipeline [Liu23q, Wu23c]、multi-agent cooperation [Li23t, Wan24y]、embodied interaction [Lin24g, Li25ad] へと発展してきた。Sokoban [Jun01]、FrozenLake [Del21]、WebShop [Yao22c] などの benchmark は、異なる dynamics の下で reasoning を評価する制御された testbed を提供する。本研究はこれらを踏まえ、symbolic task と language-centric task にまたがって RL-based reasoning と structured agent training を統合することを目指す。関連研究の詳細は付録 [B](#appendix-b) にまとめる。

<span id="section-6"></span>

## 6 結論と限界

Multi-turn かつ確率的な環境で、reinforcement learning により language agent を学習させる汎用システム RAGEN を提示した。StarPO framework を基盤とする RAGEN は reasoning-guided trajectory optimization を可能にし、gradient collapse、rollout drift、reasoning degradation など agent training 固有の新たな課題を明らかにする。広範な実験を通じ、rollout filtering、gradient shaping、reward-aware reasoning supervision など、training を安定させる主要な設計原則を特定した。これらの知見は、より頑健で汎化可能な LLM agent を構築する基盤となる。本 framework は、symbolic reasoning や web browsing などの領域で autonomous language agent を研究する scalable platform を提供する。本研究の限界は、比較的小規模な task を中心としたこと、replay buffer など確立された RL practice を用いていないこと、multimodal task を扱っていないことであり、今後の課題とする。

## 謝辞

DeepSeek-R1 model と初期の概念的着想を提供した DeepSeek team に感謝する。Infrastructure を支援した veRL team と、初期検討の参考となる発見を示した TinyZero team に感謝する。また、Han Liu、Xinyu Xing、Monica Lam、Li Erran Li、John Schulman、Akari Asai、Eiso Kant、Lu Lu、Runxin Xu、Zhihan Liu、Huajian Xin、Zijun Liu、Weiyi Liu、Weimin Wu、Yibo Wen、Jiarui Liu、Lorenzo Xiao、Ishan Mukherjee、Anabella Isaro、Haosen Sun、How-Yeh Wan、Lester Xue、Matthew Khoriaty、Haoxiang Sun、Jiajun Liu との有益な議論に感謝する。

<span id="appendix-a"></span>

## 付録 A 強化学習の背景

Reinforcement learning（RL）は、interaction と reward signal を通じて foundation model を学習させる。一般的な RL objective は次のとおりである。

<span id="equation-08"></span>

$$
J(\theta)=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_{\theta}(\cdot|s)}[R(s,a)],
$$

ここで $\pi_{\theta}$ は policy、$s$ は input prompt、$a$ は response、$R(s,a)$ は response quality を評価する reward function である。

一般的な RL approach は reward modeling と policy optimization を用いる。Proximal Policy Optimization（PPO）[Sch17a] は probability ratio clipping と advantage estimation により training を安定化する。Probability ratio を次のように定義する。

<span id="equation-09"></span>

$$
\rho_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}
$$

PPO objective は、この ratio を clipping とともに使う。

<span id="equation-10"></span>

$$
J_{\mathrm{PPO}}(\theta)=\mathbb{E}_{t}[\min(\rho_{i}A_{i},\hat{\rho_{i}}A_{i})-\beta D_{\mathrm{KL}}],
$$

Probability ratio は $\rho_{i}=\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}$、clipped ratio は $\hat{\rho_{i}}=\mathrm{clip}(\rho_{i},1-\varepsilon,1+\varepsilon)$ である。

Advantage estimation では、Generalized Advantage Estimation（GAE）[Sch15] が次を計算する。

<span id="equation-11"></span>

$$
A_{t}^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}
$$

ここで $\delta_{t}=r_{t}+\gamma V(s_{t+1})-V(s_{t})$ は TD error、$(\gamma,\lambda)$ は bias-variance trade-off を制御する。

近年、DeepSeek-R1-Zero [Dee24e] は Group Relative Policy Optimization（GRPO）によってこの paradigm を実装している。各 prompt に対して reasoning と action から成る $G$ 個の output $\{o_{i}\}$ を sampling し、次を最適化する。

<span id="equation-12"></span>

$$
J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{q,\{o_{i}\}}[J_{\mathrm{group}}(\theta)],
$$

ここで、

<span id="equation-13"></span>

$$
J_{\mathrm{group}}(\theta)=\frac{1}{G}\sum^{G}_{i=1}\min(\rho_{i}A_{i},\hat{\rho_{i}}A_{i})-\beta D_{\mathrm{KL}},
$$

GRPO は概ね Eq. 3 と同様だが、その advantage は neural model を必要とせず、次のように計算する。

<span id="equation-14"></span>

$$
A_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{j}\})}{\mathrm{std}(\{r_{j}\})}.
$$

Rule-based reward $r_{i}$ を使うこの pure RL approach は、emergent reasoning behavior を示す。

<span id="appendix-b"></span>

## 付録 B 関連研究の詳細

**LLM の推論に対する Reinforcement Learning。** LLM に対する reinforcement learning（RL）[Chr23, Ouy22b, Che21a, Hav24] は、その reasoning capability を大幅に改善してきた。主な approach には、policy update を clipping して性能を高めながら training stability を保つ Proximal Policy Optimization Algorithm（PPO）[Sch17a]、systematic problem-solving ability を高める Group Relative Policy Optimization（GRPO）[Dee25c]、critic を用いて robust exploration と stability を促す SAC [Haa18] や ArCHer [Zho24i] などの actor-critic method、structured thinking のための meta token [Goy24, Her24, Pfa24] がある。Process Reward Model（PRM）[Zha25av, Lig23a] と Monte Carlo Tree Search（MCTS）based approach [Hao23] も systematic problem-solving の重要な進展である。一方、近年の LLM reasoning 研究は、intermediate chain-of-thought rationale を生成させる technique を検討している。特に STaR [Zel22b] は、少数の rationale example と rationale のない大規模 dataset を反復利用する。SimpleRL-Zoo [Zen25e]、DAPO [Yu25g]、RLOO [Koo19]、Dr. GRPO [Liu25u]、Open Reasoner Zero [Hu25f] はいずれも、decoupled clipping、unbiased optimization、simple reward scheme を備えた最小限で再現可能な RL technique が LLM reasoning performance を大きく改善できることを示す。

**既存の agent framework。** LLM-based agent architecture は、初期の reasoning-action framework [Yao23b, Shi23c, Xu23b, Lin24h] から structured approach [Liu24u, Liu23q, Hao23, Zen25c] へ発展してきた。Multi-agent system [Du23a, Li23t, Che23f, Wan24y] は、より複雑な interaction を持つ task 向けに設計されている。OpenAI Gym [Bro16] などの広く使われる platform と、Sokoban [Jun01]、FrozenLake [Del21]、Webshop [Yao22c] などの specialized environment は、多様な agent evaluation testbed を提供する。さらに general-purpose system [She23c, Wu23c, Hao23a, Zhu23a, Xie23a] は、web navigation と search [Qi25a, Jin25b, Wei25g, Jin25f]、coding copilot [Jim24, Dee24b, Wan24z]、GUI [Qin25a, Yao22c]、game [Hu25g]、embodied task [Lin24g, Xi24a, Li25ad, Fen25b] まで幅広い応用を可能にした。Generative Agents と AgentSims [Par23a, Lin23b] は social interaction capability を進展させた。ただし architecture complexity と self-correction [He25c] には課題が残り、多様な multi-step reasoning task では特に顕著である [Wan25al, Ngu24a, Son24b]。

<span id="appendix-c"></span>

## 付録 C 実験設定の詳細

### C.1 環境とタスク

<span id="figure-08"></span>

![図 8。Bi-Arm Bandits environment。Agent は symbolic semantics に関連付けられた low-risk arm（Phoenix）と high-risk、high-reward arm（Dragon）から選択する。Agent は初期段階に安定した reward を選び、その後 reasoning により最大 expected reward を追求し、戦略的な risk-taking へ移行する。](../../papers/ragen-multi-turn-agent-rl/figure-08.png)

**図 8。** **Bi-Arm Bandits environment。Agent は symbolic semantics に関連付けられた low-risk arm（Phoenix）と high-risk、high-reward arm（Dragon）から選択する。Agent は初期段階に安定した reward を選び、その後 reasoning により最大 expected reward を追求し、戦略的な risk-taking へ移行する。**

<span id="figure-09"></span>

![図 9。Sokoban と Frozen Lake の環境。各環境について、左は agent が観測する text rendering、右は visual illustration を示す。（a）Sokoban は、agent が box を target へ押す deterministic な multi-turn puzzle である。（b）Frozen Lake は multi-turn reasoning と stochasticity を組み合わせ、agent は gift に到達すると成功する。](../../papers/ragen-multi-turn-agent-rl/figure-09.png)

**図 9。** Sokoban と Frozen Lake の環境。各環境について、左は agent が観測する text rendering、右は visual illustration を示す。（a）Sokoban は、agent が box を target へ押す deterministic な multi-turn puzzle である。（b）Frozen Lake は multi-turn reasoning と stochasticity を組み合わせ、agent は gift に到達すると成功する。

LLM agent を decision-making complexity の主要軸にわたって評価するため、**多様な 4 環境 testbed** を構築する。Bandit、Sokoban、Frozen Lake の 3 環境は symbolic、synthetic、fully controllable であり、ゼロからの RL learning を明瞭に分析できる。意図的に最小限で実世界 prior から切り離されており、GPT-4o のような large model でも未学習では性能が低いことから、grounded policy learning の必要性が分かる。これを補完する WebShop は、natural language grounding と semi-structured interface での web navigation を含む現実的な multi-turn task である。この 4 環境により、symbolic setting と open-domain setting にわたる agentic LLM の reasoning、training stability、generalization を体系的に研究できる。

各環境は異なる能力を要求する。Bandit は uncertainty 下での reasoning、Sokoban は不可逆な long-horizon planning、Frozen Lake は stochastic transition、WebShop は language understanding と goal-directed interaction を扱う。

**Bi-Arm Bandits。** Agent が**risk-sensitive hypothesis を形成し、学習に基づいて修正できるか**を評価するため、この環境を設計する。各 step で agent は「Dragon」と「Phoenix」のような 2 つの semantically symbolic option から選択する。各 option は固定 reward distribution と結び付く（[図 8](#figure-08)）。Low-risk arm は常に $0.15$ の reward を返し、high-risk arm は $\mathrm{Bernoulli}(0.25)$ から sampling する。後者は variance と expected return がともに高い。

High-risk arm の期待値が高くても、各 trial では low-risk arm の方が勝つ頻度が高い。これは reasoning を試すための設計である。Inductive bias がなければ、model は成功頻度の高い low-risk arm を好む可能性がある。Reasoning agent は symbolic cue（「Dragon」など）を基礎 reward statistics と関連付け、誤解を招く短期 signal を上書きし、long-term expected return に基づいて high-risk choice を「正当化」する必要がある。Symbolic label を反転させ、反対の reward system における agent reasoning も調べる。

**Sokoban。** Puzzle Sokoban（[図 9](#figure-09)）を使って multi-turn agent interaction を研究する。Agent は制約された step 内で grid 上の box を goal へ押さなければならない。標準 navigation と異なり Sokoban は不可逆であり、box は押すことしかできず引き戻せないため、dead-end を避ける先読みが必要である。Reward signal は効率と正確さを促す。Target 上の box ごとに $+1$、target 外の box に $-1$、task completion に $+10$、action ごとに $-0.1$ を与える。

**Frozen Lake。** この環境（[図 9](#figure-09)）は long-horizon decision-making と stochastic transition を組み合わせる。Agent は滑りやすい tile のある grid を移動し、各 action は確率 $1/3$ で成功し、確率 $2/3$ で垂直方向にずれる。穴に落ちず goal に到達する必要がある。Reward は sparse であり、成功した trial は $+1$、それ以外は $0$ である。

**WebShop。** Symbolic environment を補完するため、WebShop [Yao22c] を導入する。これは multi-turn web-based shopping task であり、natural language query の grounding、semi-structured interface の navigation、goal-relevant information の retrieval を評価する。Agent は search query を発行し、link を click し、product description を読み、user request に合う product を選ぶ。純粋な symbolic setting にはない現実的な language grounding と action space の課題を導入する。

### C.2 学習と評価の設定

Qwen2.5-0.5B-Instruct [Yang24] を使い、NVIDIA H100/A100 GPU 上で veRL repository [+verl] を利用して、StarPO variant による最大 200 回の rollout-update iteration を実行する。WebShop は long-context のため学習時間が極めて長く、100 step だけ学習する。各 rollout は environment group 当たり $K=16$ trajectory で、prompt size は $P=8$、episode 当たり最大 5 interaction turn である。Agent は turn 当たり最大 5 action、episode 当たり最大 10 action を実行できる。Update batch size は $E=32$、GPU 当たりの mini-batch size は 4 である。Policy optimization は $(\gamma=1.0,\lambda=1.0)$ の GAE と $(\beta_{1},\beta_{2})=(0.9,0.999)$ の Adam を使う。Entropy regularization（$\beta=0.001$）を使う。Vanilla StarPO experiment では $\mathrm{k1}$ estimation [+kl] による KL coefficient 0.001 を使い、[Yu25g] に従って training 中の KL loss term は除き、KL を post-hoc に追跡する。Agent が有効な structured response（`<think>` や `<answer>` tag など）を出力できない場合、$-0.1$ の format penalty を課して response convention を守らせる。Rollout generation を高速化するため `enforce_eager` を無効にし、vLLM の prefill と sampling の間で computation graph を保持する。Multi-GPU experiment には Fully Sharded Data Parallel（FSDP）training strategy を使う。Distributed training では Ray を multiprocessing backend とし、XFORMERS attention implementation を用いる。

[+verl]: https://github.com/volcengine/verl

[+kl]: http://joschu.net/blog/kl-approx.html

Evaluation では各環境から固定 256 input prompt を選び、temperature $T{=}0.5$ で stochastic sampling することで agent behavior の robustness を捉える。Episode は 5 turn または合計 10 action で打ち切る。

### C.3 評価指標

Agent learning dynamics を追跡し training instability を検出するため、training 全体で次の metric を監視する。固定 validation set で評価する success rate を除き、すべて validation instance 上で計算する。

- **Average Success Rate。** 固定した validation prompt set における task completion accuracy を測る。Agent が task を解いた場合、episode は成功とする（Bandit で high-reward arm を選ぶ、Sokoban ですべての box を target に置く、Frozen Lake で goal に到達する、WebShop で購入に成功する、など）。
- **Rollout Entropy。** Sampling した response の average token-level entropy を計算し、exploration level と policy uncertainty を捉える。Entropy の急低下は premature policy convergence または collapse を示す場合がある。
- **In-Group Reward Variance。** 同じ prompt group から sampling した rollout 間の reward standard deviation を測る。高い in-group variance は多様な behavior と learning potential を示し、急な collapse は reward homogenization と policy stagnation を示す。
- **Total Response Length。** Rollout 当たりの average generated token 数で、agent の verbosity と reasoning depth を測る。Length の変動は planning style や confidence の変化を示す場合がある。
- **Gradient Norm。** Policy gradient vector の $\ell_{2}$ norm で、training stability の代理指標として使う。Spike は policy behavior の phase transition や unstable reward signal と相関することが多い。

これらの metric は policy quality、update dynamics、reasoning behavior を補完的に捉え、agent training がいつ、なぜ成功または失敗するかを診断する。

<span id="appendix-d"></span>

## 付録 D より大きな Model と各種 Optimization Algorithm の結果

すべての評価を 3B/7B/72B model へ拡張し、KL removal、asymmetric clipping などの algorithm choice と、Generalized Advantage Estimation（GAE）および response masking を含む turn-aware optimization technique の効果を調べる。

**Scaling effect。** RL training の scaling effect を評価するため、学習 model を 3B／7B に拡張する。結果は[図 10](#figure-10) に示す。WebShop の context length は極めて長く、7B model は 4xH100 で OOM Error となるため、WebShop では 3B performance だけを報告する。**Bandit** と **WebShop** では larger model が smaller model より大幅に強い。一方、**Sokoban** と **FrozenLake** での改善は小さい。この差は環境の性質によると考える。Sokoban と FrozenLake は pretraining data との重複が少ない symbolic、grid-based task であり、model は language prior を利用しにくい。これに対し Bandit と WebShop は natural language interaction を含み、explicit environment dynamics がなくても pretrained model が linguistic pattern を policy learning に有効活用できる。[図 16](#figure-16)、[17](#figure-17)、[18](#figure-18)、[19](#figure-19) の case はこれをさらに裏付ける。Bandit と WebShop のような semantic-rich task は reasoning pattern が明らかに多様で、scale の恩恵も大きい。

<span id="figure-10"></span>

![図 10。環境ごとの scaling effect。Larger model は language prior を利用できる Bandit と WebShop で smaller model を上回るが、Sokoban や FrozenLake のような symbolic、grid-based environment では改善が限られる。](../../papers/ragen-multi-turn-agent-rl/figure-10.png)

**図 10。** **環境ごとの scaling effect。Larger model は language prior を利用できる Bandit と WebShop で smaller model を上回るが、Sokoban や FrozenLake のような symbolic、grid-based environment では改善が限られる。**

**Frontier model performance。** Small model の performance を位置付けるため、2 つの large foundation model、**GPT-4o** と **Qwen2.5-72B-Instruct** を `SimpleSokoban` と `FrozenLake` の zero-shot setting で評価する。両 model には task instruction と format example だけを与え、fine-tuning や in-context trajectory rollout は行わない。[表 5](#table-05) を参照。

<span id="table-05"></span>

![表 5。Zero-shot と学習済み performance の比較。0.5B model は prompt 当たり 4 response だけで学習しているが、fine-tuning なしの large foundation model と同等の performance を達成する。GPT-4o／Qwen の response length と effectiveness は環境別に報告する。](../../papers/ragen-multi-turn-agent-rl/table-05.png)

**表 5。** **Zero-shot と学習済み performance の比較。** 0.5B model は prompt 当たり 4 response だけで学習しているが、fine-tuning なしの large foundation model と同等の performance を達成する。GPT-4o／Qwen の response length と effectiveness は環境別に報告する。

GPT-4o と Qwen2.5-72B は task-specific adaptation なしで Sokoban と FrozenLake において 19-28% の success rate を達成する。一方、ゼロから学習した 0.5B model はそれぞれ **20.70%**、**21.48%** に達する。Model parameter が **100$\times$ 以上少ない**ことを考えると注目すべき結果である。厳しい resource constraint の下でも、rollout construction と policy optimization（[第 4.3 節](#section-4-3)）を慎重に設計すれば、はるかに大きい model の generalization ability に匹敵できる。

**Gradient shaping。** KL Term Removal と Clip-Higher [Yu25g] を single-turn static task から agent task へ単純に拡張した場合の有効性を評価する。

- **KL Term Removal：** PPO objective から KL divergence penalty を除き、policy loss と entropy bonus だけで gradient update する。Initial model distribution の近くに留まる制約を外し、model の exploration を促す。
- **Clip-Higher（Asymmetric Clipping）：** PPO clipping range を分離し、lower bound（$\varepsilon_{\mathrm{low}}=0.2$）より高い upper bound（$\varepsilon_{\mathrm{high}}=0.28$）を使う。Model は high-reward rollout からより積極的に学習できる。

[図 11](#figure-11) に示すように、両手法は success rate を高め、安定した training phase を延ばす。より柔軟な gradient shaping が multi-turn RL に有効である。

<span id="figure-11"></span>

![図 11。KL removal と asymmetric clipping が PPO stability に与える影響。2 つの設計はいずれも peak performance を高め、multi-turn RL の collapse を遅らせる。](../../papers/ragen-multi-turn-agent-rl/figure-11.png)

**図 11。** **KL removal と asymmetric clipping が PPO stability に与える影響。** 2 つの設計はいずれも peak performance を高め、multi-turn RL の collapse を遅らせる。

**Response Masking と Bi-Level GAE。** [Wan25ak] が提案した turn-aware optimization strategy に従い、0.5B model で response masking と bi-level GAE の効果を評価する。[図 12](#figure-12) に示すように、両 technique は multi-turn RL task の performance を改善し、turn-aware RL training algorithm が language agent training を安定化・強化できる可能性を示す。

<span id="figure-12"></span>

![図 12。Sokoban task における turn-aware optimization strategy の ablation。Response masking と bi-level GAE はともに multi-turn RL performance を改善する。](../../papers/ragen-multi-turn-agent-rl/figure-12.png)

**図 12。** **Sokoban task における turn-aware optimization strategy の ablation。Response masking と bi-level GAE はともに multi-turn RL performance を改善する。**

<span id="appendix-e"></span>

## 付録 E Uncertainty-Based Filtering はいつ有効か

StarPO-S の有効性は、各環境内の rollout reward variance に大きく依存すると仮定する。Task が簡単すぎる、または難しすぎる setting では、generated trajectory の intra-group variance が非常に低くなりやすい。これは model が sample 全体に過信しているか、一様に低性能であることを意味する。この場合、standard StarPO は誤解を招く gradient を伝播する可能性があるが、StarPO-S は low-confidence rollout を filtering して改善する。反対に、open-ended でより多様な環境（WebShop など）は rollout variance が自然に高く、StarPO-S filtering の marginal benefit は小さくなる。

<span id="figure-13"></span>

![図 13。Success rate（上）と rollout variance（下）の比較。StarPO-S は、Sokoban や Frozen Lake のように極端に簡単または難しい問題を持つ環境で training stability を概ね改善する。このような環境では rollout Std が小さい instance が生じ、StarPO-S で容易に除外して training を安定化できる。WebShop のような task は rollout Std が一貫して高く、StarPO 自体が良い性能を達成できる。](../../papers/ragen-multi-turn-agent-rl/figure-13.png)

**図 13。** **Success rate（上）と rollout variance（下）の比較。StarPO-S は、Sokoban や Frozen Lake のように極端に簡単または難しい問題を持つ環境で training stability を概ね改善する。このような環境では rollout Std が小さい instance が生じ、StarPO-S で容易に除外して training を安定化できる。WebShop のような task は rollout Std が一貫して高く、StarPO 自体が良い性能を達成できる。**

[図 13](#figure-13) はこの直観を支持する。上段は 4 環境における StarPO と StarPO-S の success rate、下段は training 中の `in_group_std` と `chosen_in_group_std` の変化を示す。Bandit、Sokoban、FrozenLake では StarPO-S が一貫して StarPO を上回り、rollout variance が下がるほど差が広がる。一方 WebShop では variance が高く安定しているため、generated response の多様性が高く、StarPO-S filtering の重要性は低い。このため performance gap が小さい。

以上から、StarPO-S は environment の rollout uncertainty が低い場合に最も有効であり、適用時期を判断する簡単な診断指標となる。

<span id="appendix-f"></span>

## 付録 F Case Study：RL における Echo Trap の発生

<span id="table-06"></span>

![表 6。Bandit task における reasoning pattern の例。上段は学習前の model による多様な reasoning、下段は RL training 後の反復的で collapse した reasoning を示す。](../../papers/ragen-multi-turn-agent-rl/table-06.png)

**表 6。** **Bandit task における reasoning pattern の例。上段は学習前の model による多様な reasoning、下段は RL training 後の反復的で collapse した reasoning を示す。**

<span id="figure-14"></span>

![図 14。異なる task での training iteration に伴う reasoning length。RL training 中の reasoning segment（`<think>` block）の average token count を追跡する。すべての環境で training とともに reasoning length が減少するが、`BanditRev` は長い trace を保つ。Semantic-reward conflict が大きく、より多くの熟考が必要なためと考えられる。](../../papers/ragen-multi-turn-agent-rl/figure-14.png)

**図 14。** **異なる task での training iteration に伴う reasoning length。RL training 中の reasoning segment（`<think>` block）の average token count を追跡する。すべての環境で training とともに reasoning length が減少するが、`BanditRev` は長い trace を保つ。Semantic-reward conflict が大きく、より多くの熟考が必要なためと考えられる。**

RL training の Echo Trap を示す case を提示する。[表 6](#table-06) の上段（Step 0）は Dragon と Phoenix について多様な hypothesis を示すが、下段（Step 150）は理由なしに「Dragon を選ぶ」ことへ集中した、ほぼ同一の表現に収束する。

<span id="appendix-g"></span>

## 付録 G Agent RL と Supervised Fine-Tuning の比較

RL training の StarPO に加え、別の agent training approach として Supervised Fine-tuning（SFT）を使い、Sokoban と Frozen Lake task で評価する。Rank 64、alpha 32 の LoRA を model の全 linear layer に適用する。SFT は learning rate 1e-4、training batch size 128 を使う。Breadth-first search（BFS）で ground-truth trajectory data を生成し、最大深度 100、training sample 1,000、test sample 100 とする。SFT では multi-turn interaction を conversation format に構成する。各 turn で model は ground-truth trajectory の次の action を生成し、response を `<answer> </answer>` tag で囲んで format consistency を保つ。

SFT と安定した RL baseline StarPO-S の性能を比較する。SFT は Sokoban と Frozen Lake でそれぞれ 74.6%、23% を達成し、StarPO-S は 20.3%、21.8% である。SFT は RL approach より高い性能を示す。Rule-based RL は agent task に有望だが、model self-evolution だけで human-comparable performance を達成するには、より scalable で有効な agent RL algorithm が必要である。

<span id="appendix-h"></span>

## 付録 H Low-Rank Adaptation（LoRA）による効率的な学習

**動機。** 本文は full-parameter fine-tuning の結果を報告しているが、より大きな model や long-horizon task へ拡張すると、この設定は実用上高コストになりうる。そこで Low-Rank Adaptation [Hu21] に基づく parameter-efficient な RAGEN variant を実装する [+lora]。

[+lora]: Rank を $r{=}64$、$\alpha{=}64$ とし、transformer block の全 linear projection に adapter を挿入する。Actor と critic の learning rate はともに $10\times$ に増やした。

**同等の性能。** Model parameter の一部だけを update しても、LoRA は SimpleSokoban task で network 全体を full fine-tuning した場合と同等の validation success rate に達し、validation set で約 $0.2\%$ を達成する。

**Resource saving。** LoRA と full fine-tuning の hardware footprint を比較する。80 分の training horizon で次を測定した。

- **GPU memory。** LoRA は device memory の $\mathbf{\approx 23\%}$ で安定し、full update は $\mathbf{\approx 48\%}$ である。Peak allocation を 50% 以上削減する。
- **GPU utilization。** Average GPU utilization は $\sim\!34\%$ から $\sim\!14\%$ に低下する。
- **Power consumption。** Mean power draw は $\sim\!22\%$ から $\sim\!12\%$ へ低下し、約 $45\%$ 削減される。

**要点。** Parameter-efficient fine-tuning は RAGEN の実用的な選択肢であり、同等の policy quality を保ちながら memory、compute、power demand を半分以下にできる。したがって、StarPO をより大きな backbone や長い context へ拡張する研究では、training loop を再設計せず LoRA（または別の adapter-based method）を既定の optimization strategy として採用できる。

<span id="appendix-i"></span>

## 付録 I Frozen Lake における PPO の Failure Mode

3 つの評価環境のうち、Frozen Lake では興味深い相違が見られる。PPO は GRPO より早く collapse するか、不安定に収束する傾向がある。PPO の方が良いという一般的傾向と対照的であり、追加分析を行う。

1 つの説明は、環境の long-horizon stochasticity にある。Frozen Lake では agent action が常に強い非決定的 transition を生み、中間 state が似ていても大きく異なる outcome に至る。このため value estimation が難しい。PPO は learned value function に依存するので、critic learning の不安定性が optimization noise を増幅し、early collapse に寄与する可能性がある。一方 GRPO は explicit value learning に依存しない。Reward-weighted update procedure はこの setting の uncertainty に耐性があり、他 task では効果が低くても Frozen Lake では比較的安定する可能性がある。総じて、stochasticity の高い environment は value-based method に大きな課題をもたらし、critic-free approach が有用な baseline になりうる。

<span id="appendix-j"></span>

## 付録 J Prompt Template

### J.1 Bi-Arm Bandit Environment Prompt

Bi-Arm bandit environment は、agent が exploration と exploitation の均衡を取る古典的 reinforcement learning problem を実装する。Prompt template を以下に示す。

**Model Template**

```text
<|im_start|>[system]:
{prompt}
あなたは親切な assistant です。回答は必ず <answer>...</answer> 内に書いてください。最大 response length：200 語（token）。
<|im_end|>
<|im_start|>[user]:
{prompt}
Bandit game をプレイしています。目標：引く arm を選び、total reward を最大化してください。
Game Rules：
1. {name_a} と {name_b} という 2 つの arm があります
2. 各 arm は、その名前に関連した独自の reward distribution を持ちます。
3. 各 arm の名前が持つ symbolic meaning を分析し、reward distribution がどのように振る舞うか推測してください。
4. 名前の symbolic meaning に基づき、平均して高い reward を得やすい arm はどちらだと思いますか？{name_a} と {name_b} から選び、<answer> {name_a} </answer> または <answer> {name_b} </answer> の形式で出力してください。
<|im_end|>
<|im_start|>assistant
<think>
```

### J.2 Sokoban Environment Prompt

Sokoban environment は、agent が box を target location へ押す古典的 puzzle game である。Language model との interface に使う prompt structure を以下に示す。

**Model Template**

```text
<|im_start|>system
{prompt}
あなたは親切な assistant です。回答ではまず思考を <think>...</think> 内に書き、その後に答えを <answer>...</answer> 内に書いてください。最大 response length：200 語（token）。
<|im_end|>
<|im_start|>user
{prompt}
Sokoban puzzle を解いています。あなたは player であり、すべての box を target へ押す必要があります。Box のすぐ隣にいるとき、同じ方向へ動いて押すことができます。Box を wall 越しに押したり、引いたりすることはできません。答えは <answer>Right || Right || Up</answer> のような action sequence にしてください。
<|im_end|>
<|im_start|>assistant
<think>
```

環境は grid-based representation を使い、要素ごとに記号を割り当てる。

**Grid Representation**

```text
State 内の各記号の意味：
#：wall、_：empty、O：target、✓：target 上の box、X：box、P：player、S：target 上の player
```

Instruction template は available action と restriction だけで構成される。

**Instruction Template**

```text
Available action：
Up, Down, Left, Right
最大 10 action を実行でき、action separator " || " で区切ります
```

### J.3 FrozenLake Environment Prompt

FrozenLake environment は、agent が滑りやすい凍った surface を移動して goal に到達する grid-world navigation task である。Prompt structure を以下に示す。

**Model Template**

```text
<|im_start|>system
{prompt}
あなたは親切な assistant です。回答ではまず思考を <think>...</think> 内に書き、その後に答えを <answer>...</answer> 内に書いてください。最大 response length：200 語（token）。
<|im_end|>
<|im_start|>user
{prompt}
FrozenLake puzzle を解いています。Hole を避けて target に到達してください。Ice が滑りやすいため、意図しない方向へ動く場合があります。回答例：<think>Hole を避けて target に着くには、左へ進み、その後上へ進みます。</think><answer>Left || Up</answer>
<|im_end|>
<|im_start|>assistant
<think>
```

環境は grid-based representation を使い、要素ごとに記号を割り当てる。

**Grid Representation**

```text
State 内の各記号の意味：
P：player、_：empty、O：hole、G：goal、X：hole 内の player、✓：goal 上の player
```

Instruction template は available action と restriction だけで構成される。

**Instruction Template**

```text
Available action：
Left, Down, Right, Up
最大 10 action を実行でき、action separator " || " で区切ります
```

<span id="figure-15"></span>

![図 15。Spurious reasoning を含む rollout。最終 outcome は成功するが、turn 間の reasoning trace は一貫せず、ときに事実と異なる。Model が coherent reasoning を回避して final reward だけを最適化し、RL training に noisy で誤解を招く可能性のある supervision を生む一般的 failure mode を示す。](../../papers/ragen-multi-turn-agent-rl/figure-15.png)

**図 15。** **Spurious reasoning を含む rollout。** 最終 outcome は成功するが、turn 間の reasoning trace は一貫せず、ときに事実と異なる。Model が coherent reasoning を回避して final reward だけを最適化し、RL training に noisy で誤解を招く可能性のある supervision を生む一般的 failure mode を示す。

<span id="appendix-k"></span>

## 付録 K 汎化評価環境

Training distribution を越えた汎化を評価するため、3 つの training environment に加え、異なる軸を変化させた 2 つの test environment を設計する。

- **SokobanDifferentGridVocab** は grid representation の visual vocabulary を変更する。標準記号（#、_、O、X など）ではなく、grid cell を `W`、`G`、`C` などの新しい vocabulary に写像する。Underlying spatial semantics を保ちながら、symbol variation に汎化できるかを試す。
- **LargerSokoban** は grid size を $6\times 6$ から $8\times 8$、box 数を 1 から 2 に増やし、spatial complexity と long-horizon planning demand を高める。小さい puzzle で学習した policy が複雑な configuration へ拡張できるかを評価する。

これらの環境は training 中に現れず、symbol shift、size scaling、environment shift の下で agent generalization capability を調べる。

<span id="appendix-l"></span>

## 付録 L Case Study：誤った推論による Spurious Reward

Sokoban の reasoning behavior を評価すると、model が誤った、または誤解を招く reasoning trace を示しても、非負または高い reward を受け取る場合がある。[図 15](#figure-15) は、model が box を target へ押すことには成功したものの、中間 decision が game dynamics について誤った仮定を反映する 3-turn rollout を示す。

Turn 1 と 2 で agent は「target を押す」「左の box へ移動する」など、もっともらしいが一貫しない plan を示す。これらは冗長または方向が誤っている。それでも最終 action sequence は goal に到達する。このような case は reward signal noise を増やし、RL training が真に有用な plan と偶然有効だった plan を区別しにくくする。

これは reasoning agent の multi-turn RL における主要課題を示す。*Outcome-based reward だけでは poor reasoning trace を十分に penalize できない*可能性があり、sparse または delayed feedback の環境では特に顕著である。

<span id="appendix-m"></span>

## 付録 M Case Study の拡張

Model scale と environment によって reasoning quality がどう変化するかを理解するため、Bandit（[図 16](#figure-16)）、Sokoban（[図 17](#figure-17)）、FrozenLake（[図 18](#figure-18)）、WebShop（[図 19](#figure-19)）の代表的 rollout case を、0.5B と 7B model scale で示す。**Larger model はより長く coherent な reasoning chain を生成する傾向があり、Bandit や WebShop のような semantic-rich decision task で特に顕著である**。一方、Sokoban のような grid-based environment や FrozenLake のような stochastic environment では、**small model と large model の両方が planning と alignment に苦しみ**、脆い heuristic や spurious correlation に頼ることが多い。これらの case は、Bandit や WebShop と異なり Sokoban と Frozen Lake では large model scale による明確な性能向上がないという付録 [D](#appendix-d) の実験と一致する。Reasoning quality と environment structure の関係を示し、stochastic または under-specified setting で reward-grounded reasoning を安定させる難しさを明らかにする。

<span id="figure-16"></span>

![図 16。Model scale ごとの Bandit task における reasoning-based arm selection。どちらも symbolic arm（Dragon と Phoenix）の reward tendency を prior knowledge から推定する。0.5B model は symbolic association に基づく短い理由を示す。7B model は stability と variance を比較する詳細な reasoning chain を生成し、強い prior knowledge と interpretive capacity を示す。両者とも Dragon を選ぶが、reasoning depth は異なる。](../../papers/ragen-multi-turn-agent-rl/figure-16.png)

**図 16。** **Model scale ごとの Bandit task における reasoning-based arm selection。どちらも symbolic arm（`Dragon` と `Phoenix`）の reward tendency を prior knowledge から推定する。0.5B model は symbolic association に基づく短い理由を示す。7B model は stability と variance を比較する詳細な reasoning chain を生成し、強い prior knowledge と interpretive capacity を示す。両者とも `Dragon` を選ぶが、reasoning depth は異なる。**

<span id="figure-17"></span>

![図 17。Model scale ごとの Sokoban rollout。0.5B model は reasoning が少なく、局所的には有効でも suboptimal な action を出すことが多い。7B model は turn 間でより structured な planning と symbolic alignment を示すが、long-horizon setting では非効率や heuristic move も残る。](../../papers/ragen-multi-turn-agent-rl/figure-17.png)

**図 17。** **Model scale ごとの Sokoban rollout。0.5B model は reasoning が少なく、局所的には有効でも suboptimal な action を出すことが多い。7B model は turn 間でより structured な planning と symbolic alignment を示すが、long-horizon setting では非効率や heuristic move も残る。**

<span id="figure-18"></span>

![図 18。Model scale ごとの FrozenLake rollout。0.5B agent は outcome にかかわらず固定 plan を繰り返し、adaptation と planning が限られる。7B agent は suboptimal command を出しても stochastic transition により high reward を受ける。Credit assignment の難しさと spurious pattern を強化する危険性を示す。](../../papers/ragen-multi-turn-agent-rl/figure-18.png)

**図 18。** **Model scale ごとの FrozenLake rollout。** 0.5B agent は outcome にかかわらず固定 plan を繰り返し、adaptation と planning が限られる。7B agent は suboptimal command を出しても stochastic transition により high reward を受ける。Credit assignment の難しさと spurious pattern を強化する危険性を示す。

<span id="figure-19"></span>

![図 19。WebShop rollout は model scale が long-context decision-making に与える影響を示す。0.5B agent は情報豊富な context があっても無関係な option を繰り返し選び、loop に陥る。Long-horizon memory と goal tracking が難しいことを示す。3B model は search query を絞り、product option を移動し、attribute を選び、購入を完了する multi-step reasoning chain に成功する。現実的な open-domain environment の compositional planning には scale が重要である。](../../papers/ragen-multi-turn-agent-rl/figure-19.png)

**図 19。** **WebShop rollout は model scale が long-context decision-making に与える影響を示す。** 0.5B agent は情報豊富な context があっても無関係な option を繰り返し選び、loop に陥る。Long-horizon memory と goal tracking が難しいことを示す。3B model は search query を絞り、product option を移動し、attribute を選び、購入を完了する multi-step reasoning chain に成功する。現実的な open-domain environment の compositional planning には scale が重要である。
