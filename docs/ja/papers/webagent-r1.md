---
title: 'WebAgent-R1'
createTime: 2026/08/21 14:01:26
permalink: /ja/papers/webagent-r1/
pageClass: paper-reading
---

> [Zhepei Wei](https://dblp.org/pid/247/2560.html) [+internship]、[Wenlin Yao](https://dblp.org/pid/203/8711.html)、[Yao Liu](https://dblp.org/pid/64/424.html)、[Weizhi Zhang](https://dblp.org/pid/205/0473.html)、[Qin Lu](https://dblp.org/pid/29/766.html)、[Liang Qiu](https://dblp.org/pid/01/1198.html)、[Changlong Yu](https://dblp.org/pid/76/238.html)、[Puyang Xu](https://dblp.org/pid/19/9239.html)、[Chao Zhang](https://dblp.org/pid/94/3019-14.html)、[Bing Yin](https://dblp.org/pid/98/8323.html)、[Hyokun Yun](https://dblp.org/pid/45/9671.html)、[Lihong Li](https://dblp.org/pid/l/LihongLi.html)。論文は 2025 年 5 月 22 日に arXiv へ初回投稿され、現行版は 2025 年 10 月 8 日付の v2 である。[EMNLP 2025 論文集](https://aclanthology.org/2025.emnlp-main.401/)に 7909-7928 ページとして収録。[WebAgent-R1: Training Web Agents via End-to-End Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2505.16421)。[原論文 PDF](/paper/webagent-r1.pdf)。[DOI](https://doi.org/10.18653/v1/2025.emnlp-main.401)。[TeX ソース](https://export.arxiv.org/e-print/2505.16421v2)。厳密な印刷レイアウトと参考文献については原論文 PDF を正本とする。

## 要旨

強化学習（RL）は大規模言語モデル（LLM）の強化において著しい成果を上げてきたが、主な対象は数学問題の求解などの single-turn task であった。動的な Web interface にまたがる long-horizon decision-making は複雑であり、multi-turn interaction を行う有効な Web agent の学習は依然として難しい。本稿では、Web agent を学習させるための簡潔でありながら有効な end-to-end multi-turn RL framework、WebAgent-R1 を提示する。この framework は Web environment との online interaction から直接学習し、多様な trajectory を並列に生成しながら、task の成否に応じた binary reward のみによって導かれる。WebArena-Lite benchmark での実験により WebAgent-R1 の有効性が示され、Qwen-2.5-3B の task success rate は 6.1% から 33.9% に、Llama-3.1-8B は 8.5% から 44.8% に向上し、既存の state-of-the-art 手法と OpenAI o3 などの強力な proprietary model を大幅に上回った。詳細な分析から、thinking-based prompting strategy と、Web task で interaction を増やす test-time scaling の有効性が明らかになった。さらに WebAgent-R1-Zero と WebAgent-R1-CoT という 2 つの variant を導入して異なる RL initialization policy を調べ、warm-up training stage（すなわち behavior cloning）の重要性を示すとともに、Web agent に long chain-of-thought（CoT）reasoning を組み込むための知見を得た。 [+code]

<span id="section-1"></span>

## 1 はじめに

強化学習（RL）は大規模言語モデル（LLM）を学習させる有望な方法として台頭しており、DeepSeek-R1 [Dee25c, Tea25h, Yan25g] などの近年の進展がその例である。しかし、既存研究は主として数学的推論 [Sha24d, Zen25b] などの single-turn かつ非対話的な task に焦点を当ててきた。Multi-turn の対話環境、特に Web browsing [Zho24f, He24e, Cha25d] のように long-horizon decision-making と domain-specific skill を必要とする複雑な状況での有効性は、依然として十分に研究されていない。

静的な環境とは異なり、Web task は動的で解決方法も多様であるため、LLM agent に固有の課題をもたらす。初期の Web agent 研究は主として prompting-based method [Wan24ab, Sod24, Fu24b, Zha25aw, Yan25o]、または supervised fine-tuning で demonstration trajectory を模倣する behavior cloning（BC）[Yin24, Hon24b, Lai24a, He24f, Put24] に依存していた。これらの方法は初期の成功を収めたものの、多様な strategy を探索したり trial and error から学んだりできず、Web agent の汎化性能を制限している。この問題に対処するため、近年の研究ではより良い policy training に RL を適用している。しかし、その大半は offline または iterative off-policy RL の解法 [Pen19a, Pan24f, Qi25a] に大きく依存しているため、Web agent と environment の end-to-end interaction が分断され、trajectory filtering [Bai24a]、outcome reward model training [Qi25a]、iterative optimization procedure [Zho24i] などの複雑さも加わる。こうした制約は、実環境への展開における実用性を損なう。

一方、いくつかの同時期の研究は、simulated game や coding environment [Wan25ag, Cao25b] などの multi-turn interactive scenario で LLM agent を学習させるため、on-policy update を用いた end-to-end RL を検討している。古い agent version が生成したデータで学習する off-policy RL と異なり、on-policy RL は agent の現在の行動から学習データを直接収集する。これにより、学習過程は agent の直近の行動により整合し、多くの場合、安定性と有効性が向上する [Sch15, Sch17a]。また、off-policy RL に伴う追加の overhead（たとえば replay buffer の維持や古い trajectory の filtering）が不要になり、agent は*自身の*過去の決定に基づいて適応的に行動できる。これは、初期の決定が後続の step に大きく影響しうる対話環境で重要な利点となる。

この利点は、environment の動的変化によって task 間に複雑な相互作用が生じる online Web environment で特に有用である。たとえば、agent が最初に user account から logout し、その後 user profile を編集するよう指示される状況を考える。これらの task は本質的に依存しており、agent が logout すると profile page に access できなくなる。もし agent が、logout したことのない旧 version から収集した off-policy data で学習されると、login behavior を学ぶ機会がないまま access が継続していると誤認し、無効な action を生成して task に失敗する可能性がある。End-to-end RL では environment state の変化に応じて適切な behavior をその場で学べるため、このような問題を回避できる。

これを踏まえ、Web agent を学習させる end-to-end multi-turn RL framework、WebAgent-R1 を提案する。具体的には、本設定における複数の主要課題に対処する設計を採用する。第 1 に、各 step の environment observation（たとえば HTML content）は数千 token に及ぶ場合があり、long horizon で蓄積される context は大きな memory overhead を生じさせる。これを緩和するため、turn をまたいで context を適応的に調整し、scalability を確保して out-of-memory 問題を防ぐ dynamic context compression mechanism を導入する。第 2 に、既存の LLM agent 向け RL 解法は multi-turn scenario に適していない。Group relative policy optimization（GRPO）[Sha24d] に着想を得て multi-turn setting（M-GRPO）へ拡張し、複数の trajectory を並列に生成する parallel trajectory rollout strategy によって training efficiency をさらに高める。これらの設計により効率的な RL training が可能になり、[図 1](#figure-01) に示すように WebArena-Lite benchmark で state-of-the-art の性能を達成した。広範な ablation によって主要な設計選択をさらに検証し、Web task に有効な test-time scaling strategy を明らかにするとともに、RL-based Web agent training における behavior cloning と long CoT reasoning の役割に関する知見を示す。

本稿の貢献は以下のとおりである。

- Dynamic context compression と parallel trajectory rollout mechanism により training efficiency を高めた、Web agent 向け end-to-end multi-turn RL framework を実装する。
- 提案する M-GRPO algorithm に基づき、Web agent の task success rate を大幅に高める。Qwen-2.5-3B は 6.1% から 33.9%、Llama-3.1-8B は 8.5% から 44.8% に向上し、WebArena-Lite benchmark における従来の state-of-the-art result を上回る。
- 広範な分析と ablation study により behavior cloning の重要な役割を示し、thinking-based prompting と test-time scaling strategy の有効性を検証するとともに、Web agent に long-CoT reasoning を組み込むための実用的な知見を提示する。

<span id="figure-01"></span>

![図 1。WebArena-Lite benchmark における既存手法と WebAgent-R1 の比較。](../../papers/webagent-r1/figure-01.png)

**図 1。** WebArena-Lite benchmark における既存手法と WebAgent-R1 の比較。本手法は強力な prompting-based baseline と finetuned baseline の両方を上回り、さまざまな model size でより優れた性能を達成する。

<span id="section-2"></span>

## 2 WebAgent-R1

<span id="figure-02"></span>

![図 2。WebAgent-R1 の概要と agent-Web interaction の例。](../../papers/webagent-r1/figure-02.png)

**図 2。** （**上**）：WebAgent-R1 で使用する end-to-end multi-turn RL training framework の概要。（**下**）：$k$ 番目の step における agent-Web interaction の input/output 例。最大 step 数に達するか、agent が task 完了を示す `exit()` action を生成するまで interaction は続く。

<span id="section-2-1"></span>

### 2.1 問題設定

Web task を、tuple $(\mathcal{S}, \mathcal{A}, \mathcal{T}, \mathcal{R})$ で定義される Partially Observable Markov Decision Process（POMDP）として定式化する。各時刻 $t$ で agent はまず environment $\mathcal{E}$ から state $s_t \in \mathcal{S}$ を観測する。これは現在の Web page に含まれる text-only HTML content として表される。次に、一般的な Web operation を含む predefined action space $\mathcal{A}$ から action $a_t$ を生成する。Environment dynamics $\mathcal{T}(s_{t+1}\mid s_t,a_t)$ は、action に応じて Web page がどのように変化するかを表す。Task が正常に完了するか最大 step 数に達するまで、agent は environment と interaction を行う。最後に、agent は reward function $\mathcal{R}$ から binary outcome reward $r_t \in \{0, 1\}$ を受け取る。

先行研究 [Qi25a] に従い、実用性を高めるため、WebShop [Yao22c] や Mind2Web [Den23] などの simulated または static environment ではなく WebArena [Zho24f] を Web environment として採用する。WebArena は Web agent 向けの現実的で self-host 可能な environment を提供し、最終 state にある成功の指標（たとえば確認 message や page 上の期待された content）を自動的に確認する rule-based rubric も備えている。先行研究の一部 [Liu25ab, He24e] は Web page screenshot を追加の visual input として組み込んでいるが、本稿では HTML による text-based decision-making のみを扱う。[Yan25o] などの研究では、model fine-tuning を行わずに action space や prompt design を最適化している。これらの方向性は本稿で検討する問題と直交しており、将来、本手法と概念的に統合できる。

<span id="section-2-2"></span>

### 2.2 Behavior Cloning

Web agent を初期化するため、まず expert demonstration の固定 dataset $\mathcal{D} = \{(h_t, a_t)\}$ を用いて behavior cloning（BC）を行う。ここで $h_t$ は時刻 $t$ までの完全な interaction history を表し、$h_t = (s_1, a_1, s_2, a_2, \ldots, s_t)$ と定義する。Policy $\pi_\theta$ は、この history を条件として expert action を模倣するよう supervised fine-tuning（SFT）で学習する。

$$
\mathcal{L}_{\mathrm{BC}} = - \mathbb{E}_{(h_t, a_t) \sim \mathcal{D}} \left[ \log \pi_\theta(a_t \mid h_t) \right]
$$

この warm-up stage により、agent は action space で定義された基本的な Web interaction skill を獲得する。Ablation study（[第 3.4 節](#section-3-4)）で示すように、この BC-trained policy は後続の reinforcement learning optimization に不可欠な基盤となる。

<span id="table-01"></span>

![表 1。Web agent の学習に用いる各手法の比較。](../../papers/webagent-r1/table-01.png)

**表 1。** Web agent の学習に用いる各手法の比較。*Trial-and-Error* は、environment との interaction を通じた学習（すなわち reinforcement learning）に対応するかを示す。*On-Policy* は、training data が現在の policy から収集されるかを示す。*Replay Buffer Free* は、off-policy RL で一般的に複雑さの原因となる replay buffer からの選択的な trajectory sampling が不要な手法を示す。*Self-Sufficient* は外部の training signal を必要としないことを意味する（たとえば WebRL は、GPT-4 が生成した新しい data を label 付けするため、追加の outcome reward model を学習する）。表に示すように、本手法だけが on-policy update による end-to-end RL を実現しながら、replay buffer の維持などの追加の複雑さを避け、外部の supervision も必要としない。

<span id="section-2-3"></span>

### 2.3 End-to-End Multi-Turn Reinforcement Learning

[図 2](#figure-02) に示すように、本稿の end-to-end multi-turn RL framework は rule-based outcome reward に導かれた online interaction を通じて Web agent を学習させる。効率的かつ scalable な学習を可能にするため、memory overhead を減らす *dynamic context compression* と、sampling efficiency を高める *parallel trajectory rollout* という 2 つの主要 mechanism を実装した。BC-trained policy を基に、multi-turn setting に拡張した GRPO [Qi25a] を用いて agent をさらに fine-tune し、これを *M-GRPO* と呼ぶ。本実装は、汎用性を保ちながら効率的な multi-turn RL training を支援し、将来の拡張（たとえば intermediate step に対する fine-grained reward shaping）も可能にする最小限の approach と見なせる。

**Dynamic Context Compression。** Web task では、各 observation $s_t$ が数千 token を含む場合が多い。Multi-turn interaction を重ねると蓄積 context が急速に増え、過剰な memory 使用量や out-of-memory 問題につながり、学習が現実的でなくなる。これに対処するため、dynamic context compression strategy を提案する。新しい observation が到着すると、完全な action history を保ちながら context length を減らすため、以前の observation を簡略化する。Step $t$ の interaction history を $h_t = (s'_1, a_1, s'_2, a_2, \ldots, s_t)$ とする。各 $s'_i$ は、過去の observation を表すわずかな token の template（たとえば $s'_i=\texttt{"Simplified HTML"}$）である。Agent が action $a_t$ を実行して新しい observation $s_{t+1}$ を受け取ると、更新された history は $h_{t+1} = (s'_1, a_1, s'_2, a_2, \ldots, s'_t, a_t, s_{t+1})$ となり、$s_t$ は簡略化版 $s'_t$ に置き換えられる。これにより、agent は過去の interaction に関する簡潔で情報量のある context を維持できる。Context は動的に変化するため、M-GRPO optimization 中に action token だけで loss が正しく計算されるよう、loss mask もそれに応じて更新する。

**Multi-turn GRPO。** GRPO に着想を得て標準形式を multi-turn RL setting へ拡張し、multi-turn group relative policy optimization（M-GRPO）を導入する。具体的には、各 task $q$ について、まず trajectory の group $\{\tau_1, \tau_2, \cdots, \tau_G\}$ を sample し、次の loss を最小化して policy model $\pi_\theta$ を最適化する。

$$
\mathcal{L}_{\text{M-GRPO}}(\theta) =
- \frac{1}{G} \sum_{i=1}^G
\frac{1}{|\tau_i|} \sum_{j=1}^{|\tau_i|} \left(
\frac{1}{|a_{i,j}|} \sum_{t=1}^{|a_{i,j}|} \left[
\tilde{A}_{i,j,t} - \beta\, \mathbb{D}_{\mathrm{KL}}(\theta)
\right] \right)
$$

ここで、$\tau_i=\{a_{i,1}, a_{i, 2}, \cdots, a_{i,|\tau_i|}\}$ は $i$ 番目の trajectory で生成された action sequence、$\tilde{A}_{i,j,t} = \min\{ r_{i,j,t}(\theta) A_{i,j}, \operatorname{clip}(r_{i,j,t}(\theta), 1-\epsilon, 1+\epsilon) A_{i,j}\}$ は trajectory $\tau_i$ の action $a_{i,j}$ における $t$ 番目の token の advantage、$r_{i,j,t}(\theta) = \frac{\pi_\theta(a_{i,j,t} \mid q, a_{i,j,<t})}{\pi_{\mathrm{old}}(a_{i,j,t} \mid q, a_{i,j,<t})}$ は importance sampling term を表す。$\epsilon$ と $\beta$ は hyper-parameter であり、$A_{i,j} = \frac{r_i - \mathrm{mean}(\bm{r})}{\mathrm{std}(\bm{r})}$ は rule-based reward function が生成した reward group $\bm{r}=\{r_1, r_2, \ldots, r_G\}$ を用いて計算する group relative advantage である。

<span id="table-02"></span>

![表 2。WebArena-Lite における task success rate の比較。](../../papers/webagent-r1/table-02.png)

**表 2。** WebArena-Lite の各 Web site における手法別の task success rate（SR）の比較 [Liu25ab, Qi25a, Zho24f]。Baseline performance には、再現結果と文献 [Qi25a] で報告された結果のうち高い方を記載する。最高 score は太字で示す。

**Parallel Trajectory Rollout。** Trajectory group の生成には environment との反復的な interaction が必要であり、時間がかかる場合がある。これに対処するため、複数の独立した browser instance $\{\mathcal{E}_1, \mathcal{E}_2, \cdots, \mathcal{E}_G\}$ を立ち上げ、それぞれが独自の context（たとえば cookie）を維持する parallel trajectory rollout strategy を導入する。各 task について、すべての instance は同じ開始 page で初期化されるが、agent はそれぞれと独立に interaction するため、多様な history と trajectory が得られる。この parallel design により、M-GRPO で trajectory を効率的に生成できる。

**Reward Design。** Web environment の既定の rule-based reward function を使用し、task-specific criterion（たとえば target page への到達）に基づいて binary reward（成功時は $r=1$、それ以外は $r=0$）を割り当てる。これにより outcome reward model [Qi25a] が不要になり、簡潔で汎化可能な training setting となる。

<span id="section-3"></span>

## 3 実験

<span id="section-3-1"></span>

### 3.1 実験設定

**Web Environment。** 先行研究 [Liu25ab, Qi25a] と同様に、実世界の状況における Web agent を対象とし、具体的には、多様な domain にまたがる実用的な task を支援する self-host 可能で現実的な Web environment、WebArena [Zho24f] を使用する。Domain には social forum（Reddit）、collaborative coding（GitLab）、e-commerce content management system（CMS）、open street map（Map）、online shopping（Shopping）が含まれる。

**Dataset and Evaluation Metrics。** [Qi25a] に従い、公開されている 9,460 本の trajectory を behavior cloning に使用し、より信頼性の高い評価のため、WebArena を人手で検証した version である WebArena-Lite を採用する。具体的には、検証済みの 165 task を評価に、残りの 647 task を RL training に使用する。Task success rate は組み込みの rule-based rubric で計算する。

**Baselines。** Prompting baseline では、general-purpose model（たとえば Qwen2.5、Llama3.1、GPT-4）と reasoning-specialized model（たとえば QwQ、OpenAI o3 [Ope25]）を含む open-source model と proprietary model を多様な model size にわたり包括的に比較する。Finetuning method では、Qwen2.5-3B と Llama3.1-8B を backbone model として使用する。WebArena-Lite benchmark におけるその他の baseline result については [Liu25ab] を参照されたい。

Environment と実装の詳細は[付録 A](#appendix-a) および[付録 B](#appendix-b) に示す。また、prompt template と qualitative example は[付録 D](#appendix-d) および[付録 E](#appendix-e) に示す。

<span id="section-3-2"></span>

### 3.2 主な結果

<span id="figure-03"></span>

![図 3。強化学習中の学習推移。](../../papers/webagent-r1/figure-03.png)

**図 3。** Reward、trajectory length、interaction 数を含む RL 中の training dynamics。図中の縦の破線が示すように、全過程は大きく 3 段階に分けられる。（1）初期 skill acquisition、（2）policy refinement のための exploration、（3）最終的な policy stabilization である。

**大半の LLM は prompting だけでは Web task に苦戦しており、Web agent における finetuning の重要性が示される。** [表 2](#table-02) に示すように、本実験から Web task における off-the-shelf model の限界が明らかになった。OpenAI o3 のような state-of-the-art model も、強力な汎用能力を持つにもかかわらず、success rate（SR）は 39.4% にとどまる。これに対し、単純な behavior cloning で学習した finetuned 3B model は success rate 20% を達成し、GPT-4o などの proprietary model を上回る。Off-the-shelf model の性能が低い原因は base model の size や能力ではなく、HTML structure と Web-specific behavior に対する理解の不足にあると推測する。Behavior cloning 後に 3B model と 8B model が同程度の性能を示すことも、この見方を裏付ける。これらの結果は、有効な LLM-based Web agent の開発には Web data を用いた domain-specific training が必要であることを示している。

**Reasoning model はより優れた Web agent である。** General-purpose LLM と比べ、明示的な thinking capability を備えた model は Web task で著しく優れた性能を示す。これは high-level goal を分解し、Web interface の動的変化を明示的に整理できるためだと考えられる。この差は、一般に multi-turn decision-making と動的な context 理解を必要とする Web environment において thinking が重要であることを示す。この観察に基づき、prompt design（[第 3.5 節](#section-3-5)）と training strategy（[第 3.4 節](#section-3-4)）を通じて thinking mechanism を Web agent に統合する方法をさらに調べ、その結果から Web agent に対する thinking ability の利点を改めて確認した。

**Reinforcement learning は Web agent の性能をさらに高める。** SFT による behavior cloning は Web agent としての LLM の性能を大幅に改善できる（たとえば Qwen2.5-3B を 6.1% から 20% に向上させる）が、SFT-trained policy に RL を適用すると、さらに大きな改善が得られる（たとえば Qwen2.5-3B を 20% から 33.9% へ向上させる）。この改善は、RL が long-horizon decision-making を最適化し、動的な Web interaction における trial-and-error を通じて SFT data にない新しい strategy を探索できるためだと考える。DigiRL や WebRL など、従来の Web agent 向け RL 解法も性能改善を示しているが、本手法はさらに優れた結果を達成しており、end-to-end multi-turn RL framework の有効性を示している。

<span id="section-3-3"></span>

### 3.3 学習推移

<span id="figure-04"></span>

![図 4。強化学習の initialization policy に関する ablation study。](../../papers/webagent-r1/figure-04.png)

**図 4。** WebAgent-R1（R1）と 2 つの variant を比較した RL initialization policy の ablation study。WebAgent-R1-Zero（R1-Zero）は SFT を行わず off-the-shelf model から初期化し、WebAgent-R1-CoT（R1-CoT）は behavior cloning 中に long chain-of-thought（CoT）data で学習した SFT model から初期化する。Task success rate、single-turn response length、interaction 数を、RL 適用前と適用後の両方で評価して比較する。

提案する end-to-end reinforcement learning が Web agent の behavior をどのように最適化するかを理解するため、reward、trajectory length（すなわち全 multi-turn interaction における model response の token 数）、interaction 数という 3 つの指標で training dynamics を分析する。[図 3](#figure-03) に示すように、学習過程は縦の破線で区切られた 3 つの異なる段階に大きく分けられる。

**Reward。** Phase 1 では reward が急速に増え、agent が基本 skill をすばやく学習し、より簡単な task に成功し始めたことが分かる。Phase 2 では reward の増加が plateau に達してわずかに変動し、agent が異なる strategy を探索しながら policy を改良していることを示す。Phase 3 では reward が再び徐々に改善し、exploitation が進み、安定性が高まったことを示す。

**Trajectory Length。** Trajectory length は Phase 1 で急増し、Phase 2 で安定する。Phase 3 では再び緩やかな増加が見られる。この傾向は、agent がまず詳細な output を生成することを学び、その後 consolidation の期間を経て、最後に verbosity と task effectiveness の均衡を取るためさらに調整することを示す。

**Number of Interactions。** Phase 1 では agent がより能動的になるにつれて interaction round 数が増え、Phase 2 では効率的な interaction を学ぶため減少する。Phase 3 では interaction 数が安定し、より一貫して有効な interaction strategy に収束したことを示す。

これらの傾向は、RL で一般に観察される 3 段階の learning dynamic を示している。（1）初期 skill acquisition、（2）policy refinement のための exploration、（3）最終的な policy stabilization である。興味深いことに、Qwen2.5-3B と Llama3.1-8B はいずれも同様の learning pattern を示しており、本稿の end-to-end multi-turn RL framework が model size をまたいで有効に scale し、安定した policy improvement を可能にすることが分かる。

<span id="section-3-4"></span>

### 3.4 Ablation Study

Framework の主要な設計選択を検証するため、Qwen2.5-3B を backbone model として一連の ablation study を行う。具体的には WebAgent-R1-Zero と WebAgent-R1-CoT という 2 つの variant を導入し、behavior cloning と long CoT が Web agent に及ぼす影響を調べる。結果を[図 4](#figure-04) に示す。

**Behavior cloning は RL による Web agent の学習に不可欠である。** WebAgent-R1-Zero は behavior cloning stage を省き、off-the-shelf model から直接 RL を開始するため、初期 success rate はわずか 6.1% である。意外にも、model の性能は RL 後にわずかに低下する。Model は Web task に関する知識を欠き、不完全または不正な action（たとえば必須 argument の欠落）を生成しがちで、RL 中に positive reward を得ることもほとんどないためだと推測する。これは有効な exploration と learning を大きく阻害し、Web agent の初期化とその後の RL を成功させるには behavior cloning が不可欠であることを示す。

**Behavior cloning に long-CoT data を組み込むと、より高性能な Web agent が得られる。** まず強力な reasoning model で long-CoT trace を生成して behavior cloning（BC）data を拡張し（詳細は[付録 C](#appendix-c) を参照）、SFT を適用して *long-CoT SFT* model（すなわち RL 前の WebAgent-R1-CoT variant）を得る。標準 BC data で学習した SFT model と比べ、long-CoT SFT model は大幅に高い task success rate（24.5% 対 20%）を達成し、Web agent における long-CoT reasoning の有効性を示す。

**Long-CoT SFT model に対する RL の改善は限定的である。** RL は vanilla SFT model と long-CoT SFT model の両方で有望な改善を示すものの、後者の gain が明らかに小さい点は興味深い。具体的には、WebAgent-R1 が 20% から 33.9% に向上する一方、WebAgent-R1-CoT は 24.5% から 30.3% への改善にとどまる。Long-CoT BC で学習した deterministic reasoning pattern が RL 中の model の exploration space を制約し、より柔軟な exploratory behavior を持つ標準 SFT model と比べて新しい strategy を発見する能力が制限されるためだと推測する。

<span id="section-3-5"></span>

### 3.5 分析

<span id="table-03"></span>

![表 3。Prompting design の分析。](../../papers/webagent-r1/table-03.png)

**表 3。** Prompting design の分析。平均 success rate（SR）、single-turn response length、interaction 数を報告する。この結果は、multi-turn interactive Web task に対し、interaction 数を増やす新しい test-time scaling paradigm を示す。

<span id="figure-05"></span>

![図 5。Interaction 数の増加による test-time scaling の分析。](../../papers/webagent-r1/figure-05.png)

**図 5。** 最大 interaction 数を増やした場合の test-time scaling の分析。Interaction を増やすことで Web agent はより長い trajectory を生成でき、success rate が一貫して向上する。

**Thinking format を用いた prompting は Web agent としての LLM の可能性を引き出す。** [表 3](#table-03) に示すように、thinking format を使用すると各 model の task success rate が大幅に向上し、特に強力な model で効果が大きい（たとえば o4-mini は 15.9% から 36.9% に向上する）。興味深いことに、平均 single-turn response length は同程度のまま（たとえば Qwen2.5-3B で 139 $\rightarrow$ 142 token）だが、thinking format を使うと interaction 数は大幅に増える（たとえば Qwen2.5-3B で 6 $\rightarrow$ 17）。これらの結果は、明示的な thinking instruction を含む prompting が、より頻繁な interaction を促して Web agent を強化することを示す。このため、single-turn response を長くするのではなく、より深い multi-turn interaction を行うことで Web agent を有効にできるという、Web task に対する新たな test-time scaling strategy をこの観察が示唆すると考える。

**Interaction を増やす test-time scaling は Web task の性能を高める。** 上記の知見に基づき、Web agent と environment の interaction 数を増やすと性能にどのような影響があるかをさらに調べる。[図 5](#figure-05) に示すように、interaction turn を増やすと、prompting-based、SFT（すなわち behavior cloning）、RL-based の各手法で success rate が一貫して向上する。この形式の test-time scaling は、より深い exploration を促して長い trajectory を生成し、interaction の拡張を通じて agent が action を反復的に改良し、より多くの情報に基づく decision を行えるようにする可能性があると推測する。

**WebAgent-R1 は out-of-distribution（OOD）task によく汎化する。** Qwen2.5-3B を baseline model として WebVoyager benchmark で追加評価を行った。この benchmark は WebArena environment で見られない多様な domain を含むため、本手法の OOD 評価に用いることができる。具体的には、WebVoyager の 5 domain からそれぞれ 25 task を無作為に sample して OOD evaluation set を構成し、本手法を prompting baseline および追加学習を行わない SFT variant と比較する。[表 4](#table-04) に示すように、WebAgent-R1 は全 domain で prompting baseline と SFT baseline の両方を一貫して上回り、本手法の有効性と汎化性能を確認できる。

<span id="table-04"></span>

![表 4。WebVoyager domain における out-of-distribution 評価。](../../papers/webagent-r1/table-04.png)

**表 4。** WebVoyager benchmark の 5 domain における out-of-distribution（OOD）評価。比較した各手法の domain 別 success rate を報告する。

<span id="section-4"></span>

## 4 関連研究

<span id="section-4-1"></span>

### 4.1 LLM-based Agent

LLM は、複雑な task を扱いやすい subgoal に分解し、long horizon にわたって reasoning するなど、有望な agent capability を示している [Zho22b, Hua22c, Mad22, Li23t, Li23w, Wu23c, Liu25ac, Chu25d]。こうした能力を基に、LLM-based agent は Web navigation [Nak21, Yao22c, Ma23a, Gur24, Abu24, Lut24, Pat24a, Put24]、general computer use [Li20d, Den23, Yan24l]、embodied environment [Pui18, Shr20, Toy21, Fan22, Hua22c] など、多様な実世界の interactive task に応用されている。具体的に、本稿では HTML content のみを基に browser-based environment で動作する text-based Web agent を対象とする。これには tool use、memory、partial observability 下での decision-making などの agent capability が必要である [Zho24f, Qi25a]。この研究系統と相補的に、GUI agent は screenshot などの追加の multimodal input を利用し、visual guidance に基づく environment との interaction を可能にする [Lee23a, Sha23d, Zhe24b, He24e, He24f, Koh24, Kil24, Lei25, Liu25ab]。包括的な概観については、近年の survey [Wan24k, Tse24b, Hu25h, Nin25] を参照されたい。

<span id="section-4-2"></span>

### 4.2 LLM のための Reinforcement Learning

DeepSeek-R1 [Dee25c] などの近年の進展は、LLM の強化における RL の大きな可能性を示している。しかし、従来研究の大半は数学問題 [Sha24d, Zhu25j, Sha25d, Ouy25c, Wei25h] などの single-turn task に焦点を当てており、multi-turn setting [Zho24i, Zho25g] の検討は限られている。最近では、LLM agent に search engine を繰り返し使用させる学習 [Jin25b, Sun25b, Che25z, Son25a, Wan25am] など、この方向で一定の進展が見られるが、通常は action を単純な API call に制限し、実際の environment interaction は行わない。RAGEN [Wan25ag] や SkyRL [Cao25b] など、少数の同時期の研究は simulated game や coding environment [Jim24] といったより動的な setting に RL を適用している。しかし、実世界の Web environment は依然として十分に研究されていない。本稿は実用的な framework を提供し、end-to-end RL で Web agent を学習させるための実行可能な知見を示すことで、この空白を埋める。

<span id="section-5"></span>

## 5 結論

本稿では、Web agent を学習させる end-to-end multi-turn RL framework、WebAgent-R1 を導入した。標準 GRPO を M-GRPO と呼ぶ multi-turn setting に拡張し、効率的な学習のため dynamic context compression と parallel trajectory rollout mechanism を実装した。実験では、WebAgent-R1 が WebArena-Lite benchmark で新たな state-of-the-art result を達成した。本稿の知見は、Web agent の初期化における behavior cloning の重要な役割を明らかにし、有効な RL の強固な基盤となることを示している。さらに training dynamics を分析し、thinking-based prompting と test-time scaling strategy の効果を調べ、interaction の深さを増やすことで Web agent が一貫して強化されることを示した。今後は multimodal input を検討し、本手法を general computer use など、Web environment を越えた広範な GUI-based task へ拡張する。

## 制約と潜在的リスク

WebAgent-R1 は有効であるものの、現在の approach には今後の研究方向につながる複数の制約がある。第 1 に、Web task への input は text のみを考慮している。Layout や color などの visual information は有効な navigation と decision-making に役立つため、screenshot などの visual input を追加すると性能が向上する可能性がある。第 2 に、本手法は rule-based outcome reward によって RL training を導く。この reward function は本設定では有効だが、open-ended travel planner agent のように task goal が曖昧で、明確な reference や検証可能な outcome がない他の interactive scenario では、容易に利用できない可能性がある。最後に、既存の Web agent と同様に、本 model は固定された predefined action（たとえば click、type）で学習されるため、未知の operation を必要とする interactive element に遭遇したときの柔軟性が制限される場合がある。新しい operation への動的な適応は、Web agent に残された未解決の課題である。

潜在的なリスクについては、実世界の environment、特に administrative privilege を伴う環境にこのような agent を展開する際、慎重に使用すべきである。たとえば production environment の content management system（CMS）と interaction する場合、agent が sensitive data の変更や削除など、destructive action を誤って実行する可能性がある。安全な展開のため、将来の研究では permission control、verification prompt、safeguard を組み込み、影響の大きい action や不可逆な action を防ぐ必要がある。

## 謝辞

著者らは、貴重な feedback と議論を提供した University of Virginia の Yu Meng と Shiyu Feng に感謝する。また、建設的で洞察に富む comment を寄せた匿名の reviewer にも感謝する。

<span id="appendix-a"></span>

## A Web Environment

**WebArena-Lite。** WebArena [Zho24f] は、LLM-based agent を開発するための現実的で self-host 可能な Web environment である。Social forum（Reddit）、collaborative coding（GitLab）、e-commerce content management system（CMS）、open street map（Map）、online shopping（OneStopShop）など、多様な domain にまたがる 812 の実世界 Web task で構成される。WebArena-Lite [Liu25ab] は、より信頼性の高い評価を目的に選定された WebArena の version である。代表的な 165 task を人手で検証して evaluation set とし、残りの 647 task を training に使用する。また、behavior cloning のため、program-based solver で自動的に annotation された 9,460 本の trajectory を提供する。各 Web site について、著者ら [Liu25ab] は主要機能と有効な item を整理して一連の task prototype を構築し、各 prototype に対する rule-based solver を Playwright script で手作業により実装する。対応する solver を Web site 上で実行し、ground-truth trajectory を収集する。最終的に、9,460 本の trajectory からなる 1,186 の有効な training sample が生成され、Apache License 2.0 で公開されている。

**Action Space。** Agent は、以下の predefined action を通じて environment と interaction する。

- **Click：** Web page element に対する mouse の左 click を再現する。
- **Right Click：** 指定した element を右 click する。
- **Type：** Input field に text string を入力する。
- **Search：** Search query を入力し、search operation を実行する。
- **Hover：** Cursor を特定の element 上へ移動し、tooltip や hidden menu を表示する。
- **Scroll Up / Scroll Down：** Page を縦方向に scroll する。
- **Press Enter：** 通常は typing 後に Enter key を押す操作を再現する。
- **Switch Tab：** 現在の browser tab を切り替える。
- **Select Dropdown Option：** Dropdown menu から option を選択する。
- **Wait：** Agent の interaction を短時間停止する。
- **Exit：** Final message とともに現在の session を終了する。
- **Go Backward / Go Forward：** Browser history を後方または前方へ移動する。

**Rule-based Metrics。** 実世界の Web task には通常 closed-form solution がなく、複数の trajectory が task の成功につながる場合もある。したがって、最終 goal を達成したかどうかだけに基づいて agent を評価し、次の rule-based evaluation metrics に従って task が正常に完了したかを示す Success Rate（SR）を計算する。

- **String Match：** Agent は期待される output と一致する answer string を提示しなければならない。
- **URL Match：** Agent は特定の Web page へ移動する必要がある。成功したかどうかは、最終 URL と reference URL の比較で判断する。
- **Program Execution：** Agent は Web page content または configuration を変更しなければならない。Rule-based script を実行して page の最終 state を抽出し、検証することで評価する。

WebArena の各 task には、これらの evaluation metric のいずれかが、該当する場合は対応する reference answer、target URL、validation script とともに関連付けられる。この多様な rule-based metric design は、異なる task objective と output format に対応しながら、幅広い Web task で一貫した評価を可能にする。

<span id="table-05"></span>

![表 5。WebAgent-R1 と WebAgent-R1-CoT の model output の比較。](../../papers/webagent-r1/table-05.png)

**表 5。** WebAgent-R1 と WebAgent-R1-CoT の model output の比較。同じ task（*「2023 年 1 月に最も売れた上位 3 製品は何か」*）に対する両 model の成功 trajectory を示す。明瞭さのため最初の 2 step のみを掲載する（追加の context として、完全な trajectory を[図 6](#figure-06) に示す）。WebAgent-R1 と比較して、long-CoT variant の WebAgent-R1-CoT はより詳細な thinking process を示す。

<span id="appendix-b"></span>

## B 実装の詳細

Qwen2.5-3B と Llama3.1-8B を backbone model として本手法を実装する。既定では、prompting method と fine-tuning method の両方で instruction-tuned version を使用する。Reinforcement learning（RL）の initialization policy は、behavior cloning で得られた supervised fine-tuning（SFT）checkpoint に基づく。WebRL は GPT-4 が生成した追加 data を利用して Llama3.1-8B を学習しているため、公平に比較できるよう、その公開 checkpoint で RL policy を初期化し、追加 data を導入せず元の 647 training task だけで本稿の end-to-end RL を適用する。

Model は 80GB memory の NVIDIA A100 GPU 8 基を搭載した単一 node で、full-parameter fine-tuning により学習する。GPU utilization を最適化するため、DeepSpeed [Raj20b] を使用して ZeRO-3 offload による distributed training を行い、gradient checkpointing、FlashAttention-2 [Dao24a]、bf16 mixed precision training も有効にして computation efficiency を高める。SFT では learning rate 5e-5、batch size 128 とし、5% の warmup step を含む cosine LR scheduler を使用する。RL training では constant learning rate 1e-6、batch size 16 を使用する。KL divergence regularization coefficient $\beta$ と clip ratio $\epsilon$ は、それぞれ 0.001 と 0.2 に設定する。最大 context length と new token 数は、それぞれ 16,384 と 1024 に設定する。M-GRPO 中に LLM rollout を効率的に実行するため、tensor parallel size 1、GPU memory utilization ratio 0.7 で vLLM [Kwo23] を使用する。Rollout sampling では temperature と top-p の両方を 1.0 に設定する。

<span id="appendix-c"></span>

## C Long-CoT Trace による Behavior Cloning の Data Augmentation

Ablation study（[第 3.4 節](#section-3-4)）で説明したように、強力な reasoning model である QwQ-32B を使用して long-CoT trace を生成し、元の behavior cloning data を拡張する。続いて SFT を適用して long-CoT SFT model を得た後、RL training を行って WebAgent-R1-CoT を得る。[表 5](#table-05) に示すように、WebAgent-R1-CoT は WebAgent-R1 と比較して、より詳細な thinking を生成できる。

<span id="appendix-d"></span>

## D Prompt Template

Data augmentation に使用する prompt を[表 6](#table-06) に示す。System prompt で action space を定義し、その内容を[表 7](#table-07) に示す。既定では thinking format を含む version を使用する。

<span id="appendix-e"></span>

## E 定性的な例

[図 6](#figure-06)-[10](#figure-10) では、WebArena の 5 つの Web site について、WebAgent-R1 が生成した実世界の成功 trajectory をそれぞれ示す。

<span id="table-06"></span>

![表 6。Long-CoT data augmentation に使用する prompt。](../../papers/webagent-r1/table-06.png)

**表 6。** Long-CoT data augmentation に使用する prompt。元の behavior cloning data で利用可能な *user intent*、*action history*、*current observation*、*next action*、*remarks* の各 field に、例を入力している。明瞭さのため、*current observation* に含まれる完全な HTML content は省略する。

<span id="table-07"></span>

![表 7。Web agent の system prompt。](../../papers/webagent-r1/table-07.png)

**表 7。** Web agent の system prompt。既定では thinking format を含む version（灰色で強調）を使用する。Thinking format を含まない variant（[第 3.5 節](#section-3-5) で説明）では、灰色の部分を取り除くだけでよい。

<span id="figure-06"></span>

![図 6。CMS task における WebAgent-R1 の成功 trajectory。](../../papers/webagent-r1/figure-06.png)

**図 6。** CMS task で WebAgent-R1 が生成した実世界の成功 trajectory の例：*「2023 年 1 月に最も売れた上位 3 製品は何か」*。

<span id="figure-07"></span>

![図 7。Map task における WebAgent-R1 の成功 trajectory。](../../papers/webagent-r1/figure-07.png)

**図 7。** Map task で WebAgent-R1 が生成した実世界の成功 trajectory の例：*「Homewood Suites Southpointe の滞在先から PPG Paints Arena まで、車での推定所要時間はどれくらいか」*。

<span id="figure-08"></span>

![図 8。Shopping task における WebAgent-R1 の成功 trajectory。](../../papers/webagent-r1/figure-08.png)

**図 8。** Shopping task で WebAgent-R1 が生成した実世界の成功 trajectory の例：*「Living room furniture category の製品を価格の降順で一覧表示する」*。

<span id="figure-09"></span>

![図 9。GitLab task における WebAgent-R1 の成功 trajectory。](../../papers/webagent-r1/figure-09.png)

**図 9。** GitLab task で WebAgent-R1 が生成した実世界の成功 trajectory の例：*「自分の RSS feed token を取得する」*。

<span id="figure-10"></span>

![図 10。Reddit task における WebAgent-R1 の成功 trajectory。](../../papers/webagent-r1/figure-10.png)

**図 10。** Reddit task で WebAgent-R1 が生成した実世界の成功 trajectory の例：*「Star Trek Starfleet Academy series に関する自分の post を編集し、body に 'Every watch makes me feel like a kid again' という 1 行を追加する」*。

[+internship]: Amazon での internship 中に行われた研究。

[+code]: Code と artifact は [https://github.com/weizhepei/WebAgent-R1](https://github.com/weizhepei/WebAgent-R1) で公開されている。
