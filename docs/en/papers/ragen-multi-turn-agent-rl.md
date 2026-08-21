---
title: 'RAGEN: Multi-Turn Agent RL'
createTime: 2026/08/21 09:47:30
permalink: /en/papers/ragen-multi-turn-agent-rl/
---

> [Zihan Wang](https://zihanwang314.github.io/), [Kangrui Wang](https://jameskrw.github.io/), [Qineng Wang](https://qinengwang-aiden.github.io/), [Pingyue Zhang](https://williamzhangsjtu.github.io/), [Linjie Li](https://scholar.google.com/citations?user=WR875gYAAAAJ&hl=en), [Zhengyuan Yang](https://zyang-ur.github.io/), [Xing Jin](https://scholar.google.com/citations?user=vzp-yAgAAAAJ&hl=en), [Kefan Yu](https://huangtubaye233.github.io/), [Minh Nhat Nguyen](https://scholar.google.com/citations?user=lRG8dTEAAAAJ&hl=en), [Licheng Liu](https://lichengliu03.github.io/), [Eli Gottlieb](https://www.linkedin.com/in/eli-gottlieb1/), [Yiping Lu](https://2prime.github.io/), [Kyunghyun Cho](https://kyunghyuncho.me/), [Jiajun Wu](https://jiajunwu.com/), [Li Fei-Fei](https://profiles.stanford.edu/fei-fei-li), [Lijuan Wang](https://www.microsoft.com/en-us/research/people/lijuanw/), [Yejin Choi](https://homes.cs.washington.edu/~yejin/), [Manling Li](https://limanling.github.io/). First submitted to arXiv on April 24, 2025; current version v2. [RAGEN: Understanding Self-Evolution in LLM Agents via Multi-Turn Reinforcement Learning](https://arxiv.org/abs/2504.20073v2). [Original PDF](/paper/ragen-multi-turn-agent-rl.pdf). [DOI](https://doi.org/10.48550/arXiv.2504.20073). [TeX source](https://arxiv.org/src/2504.20073v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Training large language models (LLMs) as interactive agents presents unique challenges including long-horizon decision making and interacting with stochastic environment feedback. While reinforcement learning (RL) has enabled progress in static tasks, multi-turn agent RL training remains underexplored. We propose **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization), a general framework for trajectory-level agent RL, and introduce **RAGEN**, a modular system for training and evaluating LLM agents. Our study on four stylized environments reveals three core findings. First, our agent RL training shows a recurring mode of **Echo Trap** where reward variability cliffs and gradient spikes; we address this with **StarPO-S**, a stabilized variant with trajectory filtering, critic incorporation, and gradient stabilization. Second, we find the shaping of RL rollouts would benefit from **diverse initial states, medium interaction granularity and more frequent sampling**. Third, we show that without **fine-grained, reasoning-aware reward signals**, agent reasoning hardly emerge through multi-turn RL and they may show shallow strategies or hallucinated thoughts.

**Keywords**: LLM Agents, Multi-turn RL <br>**Website**: [https://ragen-ai.github.io/](https://ragen-ai.github.io/) <br>**Code/Environments**: [https://github.com/RAGEN-AI/RAGEN](https://github.com/RAGEN-AI/RAGEN).

<span id="figure-01"></span>

![Figure 1. Previous methods focus on non-interactive tasks such as math or code generation. **RAGEN** implements StarPO, a general agent RL framework that supports multi-turn rollouts, trajectory-level reward assignment, and policy updates, on agent tasks requiring multi-turn stochastic interaction.](../../papers/ragen-multi-turn-agent-rl/figure-01.png)

**Figure 1.** Previous methods focus on non-interactive tasks such as math or code generation. **RAGEN** implements StarPO, a general agent RL framework that supports multi-turn rollouts, trajectory-level reward assignment, and policy updates, on agent tasks requiring multi-turn stochastic interaction.

<span id="section-1"></span>

## 1 Introduction

Training large language models (LLMs) to function as autonomous agents in interactive environments presents unique challenges. Unlike static tasks such as single-turn math problem solving [Sha24d] or coding [Dee24b], agent settings require models to make sequential decisions, maintain memory across turns, and adapt to stochastic feedback from their environment. These settings—central to planning assistants, robotics, and tutoring agents—demand that models not only perform well, but also self-improve through experience.

While recent work has explored reinforcement learning (RL) for LLMs [Dee25c, Ope24i, Pan25c, Zen25d, Fau24, Gao24f] using rule-based reward, it remains largely underexplored to train interactive **agents** that self-evolve to reason and adapt through rule-based RL. In particular, LLM agent training often exhibits training instability, complex reward signals, and limited generalization across environment changes, especially under multi-turn interaction with stochastic feedback. A key open question is: *what design factors make self-evolving LLM agents learn effectively and stably*?

We explore this question through a systematic study of agent learning under a general RL framework **StarPO** (**S**tate-**T**hinking-**A**ctions-**R**eward **P**olicy **O**ptimization). StarPO provides a unified view of **multi-turn, trajectory-level agent training** with flexible control over reasoning, reward assignment, and prompt-rollout structure. Built on top of StarPO, we develop **RAGEN**, a modular agent training and evaluation system designed to support the study of RL-based training in LLM Agents. RAGEN implements the full training loop—including rollout generation, reward assignment, and trajectory optimization—serving as a research infrastructure for systematic analysis of LLM agent training dynamics under multi-turn and stochastic environments.

Training LLM agents on real-world tasks like web browsing often depends on pretrained priors and heavy task-specific engineering. We evaluate RAGEN on four environments spanning different complexities: **Bandit** (single-turn, stochastic), **Sokoban** (multi-turn, deterministic), **Frozen Lake** (multi-turn, stochastic), and **WebShop** (multi-turn, open-domain). The first three symbolic environments are **minimalistic and fully controllable**, while WebShop adds **real-world understanding and reasoning**. Together, they enable analysis of generalization across varied decision-making challenges.

Using this setup, we analyze three key dimensions of agent learning, and summarize below findings that **reveal core challenges and design principles** for stable agent RL training:

- **Gradient Stability in Multi-turn RL is the Key to Stable Training.** We find that **multi-turn RL training** often leads to a recurring instability pattern, **Echo Trap**, where agents overfit to locally rewarded reasoning patterns, marked by reward variability collapse, entropy drop, and gradient spikes. To mitigate this failure mode, we propose **StarPO-S**, a stabilized variant of our framework that improves learning robustness through variability-based trajectory filtering, critic baselining, and decoupled clipping.
- **Rollout Frequency and Diversity Shape Self-Evolution.** In RL-based agent training, LLM self-generated rollout trajectories are served as core training material. We identify key rollout factors for stable agent RL training: (1) ensuring that rollouts come from **diverse initial states** with **multiple responses per initial state**, (2) **implementing multiple actions each turn** to improve interaction horizon within fixed turn limit, (3) maintaining a **high rollout frequency** to ensure online feedback reflects current policies.
- **Emerging *Agent* Reasoning Requires Meticulous Reward Signal.** We find that simply encouraging reasoning in the action format does not guarantee reasoning behavior. Even when models are prompted to reason (e.g., with ‘**&lt;think&gt;**’ tokens) with trajectory-level optimization via StarPO, they often regress to direct action selection if reasoning offers no distinct reward advantage. We assume this is due to the simple action spaces in MDP where shallow strategies suffice. Moreover, when rewards only reflect task success, models produce **hallucinated reasoning**, revealing a mismatch between thoughts and environment states. These issues underscore the need for **fine-grained, reasoning-aware reward design** for long-horizon agent training.

<span id="figure-02"></span>

![Figure 2. The State-Thinking-Actions-Reward Policy Optimization (StarPO) framework. LLM generates reasoning-guided actions for multi-turn interactions with environments and accumulates trajectory-level rewards, normalized and used to update the LLM policy.](../../papers/ragen-multi-turn-agent-rl/figure-02.png)

**Figure 2.** The State-Thinking-Actions-Reward Policy Optimization (StarPO) framework. LLM generates reasoning-guided actions for multi-turn interactions with environments and accumulates trajectory-level rewards, normalized and used to update the LLM policy.

<span id="section-2"></span>

## 2 Framework

<span id="section-2-1"></span>

### 2.1 The MDP Formulation for Agent Training

Previous reinforcement learning (RL) for language models often assumes a single-turn setting, where the goal is to maximize the expected reward $R(s,a)$ over prompt-response pairs $(s,a)$ sampled from a dataset $\mathcal{D}$:

<span id="equation-01"></span>

$$
J_{\mathrm{step}}(\theta)=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_{\theta}(\cdot|s)}[R(s,a)].
$$

However, LLM-based agents must operate in interactive environments that unfold over multiple turns and exhibit stochastic feedback. To capture these dynamics, we formulate the problem as a Markov Decision Process (MDP) $\mathcal{M}=\{S,A,P\}$, where $S$ represents states (e.g., observation sequences or interaction histories), $A$ represents actions (often token sequences), and $P$ denotes the transition dynamics and reward generation process. The agent policy $\pi_{\theta}$ generates an action $a_{t}$ at each time step $t$, conditioned on the current state $s_{t}$ and the interaction history. The environment returns a reward $r_{t}$ and a new state $s_{t+1}$ given the current transition dynamics:

$$
a_{t}\sim\pi_{\theta}(\cdot|s_{t},\tau_{<t}),\quad(r_{t},s_{t+1})\sim P(\cdot|s_{t},a_{t}),
$$

where $\tau_{<t}=\{s_{0},a_{0},r_{0},...,s_{t-1},a_{t-1},r_{t-1}\}$ denotes the interaction history. This interactive process continues for a maximum horizon $K$, yielding a full trajectory $\tau=\{s_{0},a_{0},r_{0},...,s_{K}\}$ that forms the learning material for the agent.

<span id="section-2-2"></span>

### 2.2 StarPO: Reinforcing Reasoning via Trajectory-Level Optimization

We introduce **StarPO** (State-Thinking-Action-Reward Policy Optimization), a general RL framework designed to optimize entire multi-turn interaction trajectories for LLM agents. Unlike previous methods for static tasks that treat each action independently, StarPO treats the **entire trajectory**—including observations, reasoning traces, actions, and feedback—as a coherent unit for rollout and model optimization. The objective is to maximize expected trajectory reward:

<span id="equation-02"></span>

$$
J_{\mathrm{StarPO}}(\theta)=\mathbb{E}_{\mathcal{M},\tau\sim\pi_{\theta}}\left[R(\tau)\right],
$$

where $\mathcal{M}$ is the MDP, $\tau$ is a full sequence of reasoning-augmented interactions, and $R(\tau)$ denotes the cumulative reward over the entire trajectory. The policy probability $\pi_{\theta}(\tau)$ is decomposed into token-level likelihoods, making StarPO directly compatible with autoregressive LLMs. [Figure 2](#figure-02) illustrates the full StarPO process, and we break them down in detail below.

##### Trajectory-Level Objective in StarPO vs. Previous Methods

**Previous methods (e.g., PPO [Sch17a], GRPO [Sha24d]):**

$$
J_{\mathrm{step}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\,y\sim\pi_\theta(\cdot|x)}\left[R(x,y)\right]
\quad
\text{(optimize single-turn output }y\text{ given input }x)
$$

**StarPO (ours):**

$$
J_{\mathrm{StarPO}}(\theta)=\mathbb{E}_{\mathcal{M},\tau\sim\pi_\theta}\left[R(\tau)\right]
\quad
\text{(optimize total reward over trajectory }\tau=\{s_0,a_0,r_0\dots,s_K\})
$$

<span id="section-2-2-2"></span>

#### 2.2.2 Optimization Procedure: Learning from Reasoning-Interaction Trajectories

At each training iteration, the agent begins from an initial state $s_{0}$ and generates $N$ trajectories. At each step $t$, the agent produces a reasoning-guided structured output:

<span id="equation-03"></span>

$$
a^\top_{t}=\texttt{<think>}...\texttt{</think><answer>}\,a_{t}\,\texttt{</answer>},
$$

where $a^\top_{t}$ is the full action output including intermediate reasoning, and $a_{t}$ is a sequence of environment-executable sub-action. The environment then returns the next state $s_{t+1}$ and reward $r_{t}$. The rollout stage produces complete trajectories $\tau=\{s_{0},a^\top_{0},r_{0},s_{1},...,a^\top_{K-1},r_{K-1},s_{K}\}$, where *every component is LLM-generated or environment-induced* and will be jointly optimized.

StarPO interleaves rollout and update steps. New rollouts can be generated on-policy using $\pi_{\theta}$, or sampled from a replay buffer under $\pi_{\text{old}}$. Each training loop consists of $P$ initial states $s_{0}$, each generating $N$ trajectories, and updates are performed with batch size $E$ for $L$ total loops. This yields $S=\frac{L\cdot P\cdot N}{E}$ total gradient update steps. Additional training mechanisms are discussed in §3.

<span id="section-2-2-3"></span>

#### 2.2.3 Modular Optimization Strategies

StarPO supports a variety of policy optimization algorithms under a unified trajectory-level abstraction. For each rollout trajectory $\tau_{i}=\{\tau_{i,(1)},\ldots,\tau_{i,(|\tau_{i}|)}\}$ of totally $|\tau_{i}|$ tokens, we instantiate StarPO with the following optimization strategies for token-level updates:

- **PPO [Sch17a].** We use the PPO objective (More details can be found in Appendix [A](#appendix-a)), where a critic is trained to estimate token-level value and advantages $A_{i,t}$:

<span id="equation-04"></span>

$$
J_{\mathrm{PPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
\min\left[
\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})}\cdot A_{i,t},\,
\mathrm{clip}\left(\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})},1-\varepsilon,1+\varepsilon\right)\cdot A_{i,t}
\right].
$$

where $G$ is the number of trajectories in the batch, $\tau_{i,(t)}$ denotes the $t$-th token in trajectory $\tau_i$, and $\tau_{i,<t}$ is its prefix.

- **GRPO [Sha24d].** For critic-free training leveraging GRPO, we assign a scalar reward $R(\tau_i)$ to each trajectory and normalized advantage $\hat{A}_{i,t}$ across all tokens in $\tau_i$:

<span id="equation-05"></span>

$$
\hat{A}_{i,t}=\frac{R(\tau_i)-\mathrm{mean}(\{R(\tau_1),\ldots,R(\tau_G)\})}
{\mathrm{std}(\{R(\tau_1),\ldots,R(\tau_G)\})}.
$$

The GRPO objective becomes:

<span id="equation-06"></span>

$$
J_{\mathrm{GRPO}}(\theta)=\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|\tau_i|}\sum_{t=1}^{|\tau_i|}
\min\left[
\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})}\cdot\hat{A}_{i,t},\,
\mathrm{clip}\left(\frac{\pi_\theta(\tau_{i,(t)}|\tau_{i,<t})}{\pi_{\text{old}}(\tau_{i,(t)}|\tau_{i,<t})},1-\varepsilon,1+\varepsilon\right)\cdot\hat{A}_{i,t}
\right].
$$

<span id="section-2-3"></span>

### 2.3 The RAGEN System

To implement StarPO in practice, we build **RAGEN**, a complete system for LLM agent training in controlled environments. RAGEN supports structured rollouts, customizable reward functions, and integration with multi-turn, stochastic environments. It serves both as the execution backend for StarPO and as a platform for studying stability, generalization, and learning dynamics in training reasoning agents. RAGEN is designed to be extensible: new environments, reward schemes, or rollout strategies can be easily plugged in, serving as a foundation for RL-based agent training.

<span id="section-3"></span>

## 3 Experiment Setup

<span id="section-3-1"></span>

### 3.1 Environments and Tasks

We evaluate LLM agents on four environments spanning symbolic and realistic decision-making: **Bandit** tests risk-sensitive reasoning under noisy feedback; **Sokoban** requires irreversible symbolic planning; **Frozen Lake** combines planning with probabilistic transitions; and **WebShop** involves natural language grounding and web environment interaction. The first three symbolic environments are deliberately minimal and fully controllable to support clean analysis, while WebShop introduces realistic task structure and language input. Environment visualizations are in Appendix C.1.

<span id="section-3-2"></span>

### 3.2 Training Settings

In our main experiments, we train Qwen-2.5 Instruct 0.5B models for three symbolic tasks and its 3B variant for the challenging WebShop. We also report various model performance in Appendix [D](#appendix-d). Models are trained with StarPO variants on H100 GPUs for 100-200 rollout–update iterations. Each batch samples $P{=}8$ prompts, with $N{=}16$ rollouts per prompt, up to 5 turns and 10 actions. Policy updates use GRPO or PPO with GAE ($\gamma{=}1.0,\lambda{=}1.0$), Adam optimizer, entropy bonus ($\beta{=}0.001$), and a response-format penalty ($-0.1$). More details can be found in Appendix C.2.

<span id="section-3-3"></span>

### 3.3 Evaluation Metrics

We evaluate on 256 fixed prompts per environment with temperature $T{=}0.5$, truncating episodes after 5 turns. Metrics include: **(i)** success rate (task completion), **(ii)** rollout entropy (exploration), **(iii)** in-group reward variability (behavioral diversity), **(iv)** response length (reasoning verbosity), and **(v)** gradient norm (training stability). All are computed over validation instances. More details can be found in Appendix C.3.

<span id="section-4"></span>

## 4 Experimental Results and Findings

<span id="section-4-1"></span>

### 4.1 Multi-turn Agent RL Training Introduces New Instability Pattern

We evaluate baseline StarPO across agent tasks ([Figure 3](#figure-03)). Symbolic environments like Bandit and Sokoban show early improvements but eventually collapse. PPO is more stable than GRPO in these settings, collapsing later and achieving higher performance, likely due to its critic providing smoother reward estimates. Interestingly, GRPO is more stable on Frozen Lake, likely due to the difficulty of estimating state values in this task, which may destabilize PPO (see Appendix [I](#appendix-i)). On WebShop, both methods succeed, likely due to strong language prior and high initial rewards reducing the need for a critic.

<span id="figure-03"></span>

![Figure 3. **Baseline StarPO performance across environments.** Symbolic tasks like Bandit and Sokoban lead to collapse while real-world WebShop starts high and improves fast. PPO is stronger in Bandit and Sokoban to provide more stable token-level reward signal, while GRPO is stronger in Frozen Lake whose randomness makes the state value hard to estimate, and WebShop whose strong initial performance lessens the need for a critic to stabilize gradients.](../../papers/ragen-multi-turn-agent-rl/figure-03.png)

**Figure 3.** **Baseline StarPO performance across environments.** Symbolic tasks like Bandit and Sokoban lead to collapse while real-world WebShop starts high and improves fast. PPO is stronger in Bandit and Sokoban to provide more stable token-level reward signal, while GRPO is stronger in Frozen Lake whose randomness makes the state value hard to estimate, and WebShop whose strong initial performance lessens the need for a critic to stabilize gradients.

To understand the causes of collapse, we compare early- and late-stage trajectories. In the Bandit task, early-stage trajectories exhibit diverse reasoning about symbolic meanings and expected rewards, while later-stage responses become repetitive and deterministic. This suggests that **RL training may have over-amplified inherent reasoning shortcuts**, reinforcing locally rewarded templates while suppressing exploration. We refer to this failure mode as an “**Echo Trap**”, which is similar to the findings in [Shu24], where the model repeatedly reuses memorized reasoning paths when trained on self-generated trajectories, leading to a collapse in diversity and long-term performance degradation. Examples are in Appendix [F](#appendix-f).

To detect collapse, we monitor two key metrics: (1) **Average Reward**, where a plateau or drop signals degraded task performance, and (2) **Gradient Norm**, where spikes suggest unstable updates. To identify early signs, we also track rollout-level signals: (1) **Reward Standard Deviation**, indicating whether the policy is producing diverse outcomes or collapsing into repetitive behavior, and (2) **Output Entropy**, which indicates how deterministic the model’s predictions are.

<span id="figure-04"></span>

![Figure 4. **Collapse indicators and early warning signals in multi-turn RL.** Average reward and gradient norm (left-side plots) reflect collapse directly, and their plateaus and spikes confirm performance and training instability. Reward standard deviation and entropy (right-side plots) often becomes unstable before reward degrades, serving as early warning signals.](../../papers/ragen-multi-turn-agent-rl/figure-04.png)

**Figure 4.** **Collapse indicators and early warning signals in multi-turn RL.** Average reward and gradient norm (left-side plots) reflect collapse directly, and their plateaus and spikes confirm performance and training instability. Reward standard deviation and entropy (right-side plots) often becomes unstable before reward degrades, serving as early warning signals.

[Figure 4](#figure-04) summarizes these dynamics across tasks and optimization methods. From the results, we draw the following conclusions regarding **how model collapse emerges in multi-turn agent RL**:

- **Reward standard deviation is an early indicator of convergence.** In FrozenLake-PPO, std drops sharply at step 40, well before reward mean collapses at step 90 while performance is near-optimal. In Bandit-PPO, std bottoms out around step 70, preceding the reward peak at step 120. In Sokoban-PPO, std and mean collapse together near step 10, suggesting early saturation.
- **Gradient norm spikes indicate irreversible collapse.** Once gradient norm spikes emerge, e.g., at step 170 (Bandit), 110 (Sokoban), and 90 (FrozenLake), even small parameter updates could induce drastic loss shifts, after which recovery becomes unlikely.
- **Entropy should follow a stable decay trend during effective learning**. This can be seen from FrozenLake-GRPO. Rapid entropy increases or erratic changes often correlate with collapsed reasoning behavior, such as GRPO on Bandit and Sokoban.

These patterns confirm that multi-turn RL introduces unique challenges that single-turn RL methods fail to handle. In response, we introduce **StarPO-S**, a stabilized variant that targets sampling quality, gradient stability, and exploration regularization to avoid premature collapse.

<span id="section-4-2"></span>

### 4.2 StarPO-S: Stabilize Multi-turn RL with Instance Filtering and Gradient Shaping

To address the instability of multi-turn reinforcement learning, we introduce **StarPO-S**, a stabilized variant of StarPO that incorporates three key modifications aimed at improving training robustness and efficiency. Building on the insight that declining reward standard deviation often precedes collapse, we investigate the following question: *should agents be trained more intensively on task instances where their behavior is more uncertain with higher reward variability?*

We hypothesize that the most effective training samples are those where the agent **exhibits outcome uncertainty**—avoiding both trivial task instances and overly difficult ones. This intuition is rooted in principles of Active Learning [Set09], where uncertain examples are the most informative ones models should learn from. We define trajectory-level outcome uncertainty $U$ for policy $\pi_{\theta}$ on a given agent task instance (initial state $s_{0}$ in an MDP $\mathcal{M}=\{S,A,P\}$) as:

<span id="equation-07"></span>

$$
\text{U}(\pi_{\theta},\mathcal{M},s_{0})=\text{Std}_{\tau\sim\pi_{\theta}(\cdot|s_{0})}\left[R(\tau)\right].
$$

During training, we sort prompts based on the standard deviation of reward obtained from repeated rollouts and **retain only the top $p$% highly-uncertain prompts** at each training step. [Figure 5](#figure-05) shows the effect of varying $p$ in PPO and GRPO under StarPO-S. We further validate the effect of uncertainty-based filtering in Appendix [E](#appendix-e).

In PPO runs ([Figure 5](#figure-05), left), filtering low-variability rollouts significantly delays collapse: retaining 75% of rollouts extends stability in FrozenLake from 100 to 140 steps, while 50% avoids collapse entirely. GRPO remains less stable due to its critic-free design but still benefits modestly. Filtering also improves efficiency ([Figure 5](#figure-05), right). We adopt 25% as the default for StarPO-S. However, we note that this aggressive value may not be optimal for all scenarios. Tasks like Sokoban and FrozenLake respond well to aggressive filtering, potentially due to their relatively repetitive reasoning patterns and under-representation in pretraining, which make them tend to collapse when similar trajectories dominate the batch. We further present larger model (72B), together with frontier model such as GPT-4o and Qwen-2.5-72B, in Appendix [D](#appendix-d), to better contextualize our model performance.

In addition to uncertainty-based filtering, we adopt two gradient shaping techniques inspired by DAPO [Yu25g] designed for single-turn RL: **KL Term Removal** and **Clip-Higher** (Asymmetric Clipping). We extend and evaluate them in the multi-turn agent setting, and find both methods boost the success rate and extend stable training phases, showing how multi-turn RL benefits from more flexible gradient shaping. Design details and performance ablation is in Appendix [D](#appendix-d).

<span id="figure-05"></span>

![Figure 5. **Effect of uncertainty-based filtering on multi-turn RL stability. Filtering out low-variability trajectories reduces collapse risk and improves success rate. On PPO variants, collapse is largely mitigated when more than half of the trajectories are filtered. Training time is reduced.**](../../papers/ragen-multi-turn-agent-rl/figure-05.png)

**Figure 5.** **Effect of uncertainty-based filtering on multi-turn RL stability. Filtering out low-variability trajectories reduces collapse risk and improves success rate. On PPO variants, collapse is largely mitigated when more than half of the trajectories are filtered. Training time is reduced.**

**Overall Comparison.** We compare StarPO-S with vanilla StarPO across three tasks in [Figure 6](#figure-06). StarPO-S consistently delays collapse and enhances final task performance. We attribute these gains to more selective training data (via uncertainty filtering), more balanced optimization signals (via KL removal and decoupled clipping), reducing narrowed reasoning modes. In the appendix [D](#appendix-d), we further discuss about other variants that may stablize training and enhance performance, such as selective response mask and Bi-level Gegenal Advantage Estimation (GAE) [Wan25ak].

<span id="figure-06"></span>

![Figure 6. **StarPO-S improves stability and final performance across tasks. Compared to vanilla StarPO, StarPO-S reliefs collapse in all four tasks and could achieves higher success rates. **](../../papers/ragen-multi-turn-agent-rl/figure-06.png)

**Figure 6.** **StarPO-S improves stability and final performance across tasks. Compared to vanilla StarPO, StarPO-S reliefs collapse in all four tasks and could achieves higher success rates. **

<span id="section-4-3"></span>

### 4.3 Generating Useful Trajectories for RL Training

Effective RL training depends heavily on trajectory quality. We study three key rollout dimensions—*task diversity*, *interaction granularity*, and *rollout frequency*—by training on vanilla Sokoban and evaluating on other tasks including SokobanNewVocab, LargeSokoban, and FrozenLake Task, which we detail in Appendix [K](#appendix-k).

**Higher task diversity with response comparison improves generalization.** Task diversity refers to the number of distinct prompts used for each rollout-update cycle. With a fixed batch size, it trades off against the number of responses per prompt. In our experiments ([Table 1](#table-01)), we vary this trade-off and find that higher task diversity—achieved by fewer responses per prompt (e.g., 4 per prompt)—consistently yields better generalization. This only holds when each prompt includes multiple rollouts, enabling the agent to contrast different outcomes under similar conditions.

**Allowing more action budgets enables planning, while overly long-horizon rollouts inject noise.** We vary the number of actions allowed per turn in [Table 2](#table-02). Allowing up to 5 or 6 actions per turn yields the best performance, especially on complex environments like SokobanNewVocab and LargeSokoban. This setting provides enough room for planning while avoiding the chaos of overly long rollouts. Increasing the budget to 7 actions degrades performance, likely due to noisy transitions and diluted reward feedback.

<span id="table-01"></span>

![Table 1. **Effect of Task Diversity on Generalization Performance (%).** Higher diversity with multiple responses yields the best performance (4 responses per prompt).](../../papers/ragen-multi-turn-agent-rl/table-01.png)

**Table 1.** **Effect of Task Diversity on Generalization Performance (%).** Higher diversity with multiple responses yields the best performance (4 responses per prompt).

<span id="table-02"></span>

![Table 2. **Performance across environments under different per-turn action budgets (%).** 5–6 actions per turn yields best performance, balancing well for effective multi-step planning.](../../papers/ragen-multi-turn-agent-rl/table-02.png)

**Table 2.** **Performance across environments under different per-turn action budgets (%).** 5–6 actions per turn yields best performance, balancing well for effective multi-step planning.

<span id="table-03"></span>

![Table 3. **Generalization performance (%) with and without reasoning under StarPO-S.** Disabling reasoning significantly reduces generalization in single-turn Bandit task, but has mixed or marginal effects in multi-turn Sokoban task.](../../papers/ragen-multi-turn-agent-rl/table-03.png)

**Table 3.** **Generalization performance (%) with and without reasoning under StarPO-S.** Disabling reasoning significantly reduces generalization in single-turn Bandit task, but has mixed or marginal effects in multi-turn Sokoban task.

**Frequent rollout updates ensure alignment between optimization targets and current policy behavior.** To investigate the effect of rollout freshness, we adopt an *Online-$k$* rollout strategy, where a single set of rollouts is reused for $k$ consecutive policy updates. A smaller $k$ implies more frequent rollout collection. Notably, *Online-1* corresponds to an fully online setting, with fresh rollouts collected every update iteration. As shown in [Figure 7](#figure-07), agents trained with fresher rollouts (*Online-1*) achieve faster convergence and better generalization across tasks compared to those with delayed updates (e.g., *Online-5* or *Online-10*). This supports a core design principle for multi-turn RL: learning is most effective when trajectories reflect the agent’s latest behavior. Frequent rollout reduces policy-data mismatch and improves optimization stability.

<span id="section-4-4"></span>

### 4.4 Reasoning Improves Generalization but Fades in Multi-Turn Settings Without Fine-Grained Rewards

We examine how symbolic reasoning impacts agent generalization. While reasoning enhances performance in single-turn tasks like Bandit, it fails to grow or sustain in complex multi-turn environments like Sokoban. Below, we analyze these effects step-by-step.

**Reasoning traces improve generalization in single-turn Bandit tasks.** We design a controlled generalization test in symbolic Bandit environments. In the original `Bandit` setting, the model is trained on the `[Teacher, Engineer]` arm pair and evaluated on `[Librarian, Trader]`, preserving intuitive risk-reward alignments (i.e., `Engineer` and `Trader` are high-risk, high-reward). In `BanditRev`, these associations are inverted, assigning counter-intuitive reward profiles to professions and making reasoning more challenging.

As shown in [Table 3](#table-03), models trained with reasoning traces generalize better in `Bandit` and even in the counterintuitive `BanditRev`, suggesting that reasoning supervision helps internalize symbolic cues beyond memorization. Despite the added difficulty in `BanditRev`, models with explicit reasoning consistently outperform those without, as shown in [Table 3](#table-03). This suggests that reasoning traces help the agent internalize symbolic-reward associations and generalize beyond surface-level memorization, even under semantic-reward misalignment.

**In multi-turn tasks, reasoning signals fade as training progresses.** In contrast to single-turn settings, we find that reasoning provides limited benefits in multi-turn environments like Sokoban and FrozenLake. Even when the output format includes explicit `<think>` segments, removing them (no-think variant) often yields comparable or even better performance. To understand this degradation, we analyze average response length during training ([Table 4](#table-04), [Figure 14](#figure-14)) and find that reasoning traces consistently shrink over time, suggesting the model is suppressing its own thought process. Interestingly, in the semantically misaligned `BanditRev` task—where reasoning is essential—traces remain longer, indicating that reasoning is better sustained when context is more challenging.

We presume that reasoning collapse may arise from the **sparse, delayed reward structure in multi-turn tasks**, which often fails to differentiate between coherent reasoning and trial-and-error success. This is supported by examples (Appendix [L](#appendix-l)) where models generate incoherent or hallucinated reasoning yet still receive high rewards. This raises an important challenge: *how can we consistently reinforce useful reasoning when the reward alone may not reflect its quality?* One possible approach is to decouple action correctness from reasoning quality using format-based penalties: similar to [Sha24d], we apply a small penalty to outputs lacking valid `<think>`–`<answer>` structure, encouraging structured reasoning. We believe future work may explore finer-grained reward designs such as rewarding partial correctness to reliably reinforce reasoning in long-horizon decision-making.

<span id="table-04"></span>

![Table 4. **Reasoning length (`<think>` block length) at different training steps.** Token length generally declines over time, while contextually challenging problems like `ReverseBandit` require more reasoning than their original counterparts.](../../papers/ragen-multi-turn-agent-rl/table-04.png)

**Table 4.** **Reasoning length (`<think>` block length) at different training steps.** Token length generally declines over time, while contextually challenging problems like `ReverseBandit` require more reasoning than their original counterparts.

<span id="figure-07"></span>

![Figure 7. **Performance under different rollout frequencies (*Online-$k$*). We vary the rollout reuse factor $k$, where each batch is reused for $k$ policy updates. Lower $k$ (e.g., *Online-1*) means more frequent rollouts. Fresher data improves convergence by staying aligned with the current policy.**](../../papers/ragen-multi-turn-agent-rl/figure-07.png)

**Figure 7.** **Performance under different rollout frequencies (*Online-$k$*). We vary the rollout reuse factor $k$, where each batch is reused for $k$ policy updates. Lower $k$ (e.g., *Online-1*) means more frequent rollouts. Fresher data improves convergence by staying aligned with the current policy.**

<span id="section-5"></span>

## 5 Related Work

Recent work has explored fine-tuning LLMs for multi-step reasoning through **reinforcement learning (RL)**, and developing agent frameworks to structure decision-making tasks. For reasoning, techniques range from classical PPO [Sch17a] and actor-critic methods [Haa18] to structured prompting via meta tokens [Goy24, Her24]. Policy variants such as RLOO [Koo19], GRPO [Dee25c] and DAPO [Yu25g] could stabilize training and improve sample efficiency. Parallel efforts like STaR [Zel22b] and MCTS-based reasoning [Hao23] promote step-by-step reasoning with minimal supervision.

**On the agent side**, systems have evolved from early reactive planning [Yao23b, Xu23b] to modularized decision pipelines [Liu23q, Wu23c], multi-agent cooperation [Li23t, Wan24y], and embodied interaction [Lin24g, Li25ad]. Benchmarks like Sokoban [Jun01], FrozenLake [Del21], and WebShop [Yao22c] provide controlled testbeds for evaluating reasoning under different dynamics. Our work builds on these advances, aiming to unify RL-based reasoning with structured agent training across symbolic and language-centric tasks. We summarize more related works in Appendix [B](#appendix-b).

<span id="section-6"></span>

## 6 Conclusions and Limitations

We present RAGEN, a general-purpose system for training language agents with reinforcement learning in multi-turn, stochastic environments. Built upon the StarPO framework, RAGEN enables reasoning-guided trajectory optimization and reveals new challenges unique to agent training—such as gradient collapse, rollout drift, and reasoning degradation. Through extensive experiments, we identify key design principles for stabilizing training, including rollout filtering, gradient shaping, and reward-aware reasoning supervision. These insights provide a foundation for building more robust and generalizable LLM agents. Our framework offers a scalable platform for studying autonomous language agents across domains such as symbolic reasoning and web browsing. Limitations of our work include the focus on relatively small-scale tasks, the omission of established RL practices like replay buffers, and the absence of multimodal tasks—which we leave for future work.

## Acknowledgements

We thank the DeepSeek team for providing the DeepSeek-R1 model and early conceptual inspirations. We are grateful to the veRL team for their infrastructure support, and to the TinyZero team for their discoveries that informed our initial exploration. We would like to appreciate insightful discussions with Han Liu, Xinyu Xing, Monica Lam, Li Erran Li, John Schulman, Akari Asai, Eiso Kant, Lu Lu, Runxin Xu, Zhihan Liu, Huajian Xin, Zijun Liu, Weiyi Liu, Weimin Wu, Yibo Wen, Jiarui Liu, Lorenzo Xiao, Ishan Mukherjee, Anabella Isaro, Haosen Sun, How-Yeh Wan, Lester Xue, Matthew Khoriaty, Haoxiang Sun, Jiajun Liu.

<span id="appendix-a"></span>

## Appendix A Background of Reinforcement Learning

Reinforcement learning (RL) enables foundation models to learn through interaction and reward signals. The general RL objective is:

<span id="equation-08"></span>

$$
J(\theta)=\mathbb{E}_{s\sim\mathcal{D},a\sim\pi_{\theta}(\cdot|s)}[R(s,a)],
$$

where $\pi_{\theta}$ is the policy, $s$ is the input prompt, $a$ is the response, and $R(s,a)$ is the reward function evaluating response quality.

Common approaches use reward modeling and policy optimization for RL. Proximal Policy Optimization (PPO) [Sch17a] stabilizes training through probability ratio clipping and advantage estimation. The probability ratio is defined as:

<span id="equation-09"></span>

$$
\rho_{t}(\theta)=\frac{\pi_{\theta}(a_{t}|s_{t})}{\pi_{\theta_{old}}(a_{t}|s_{t})}
$$

The PPO objective uses this ratio with clipping:

<span id="equation-10"></span>

$$
J_{\mathrm{PPO}}(\theta)=\mathbb{E}_{t}[\min(\rho_{i}A_{i},\hat{\rho_{i}}A_{i})-\beta D_{\mathrm{KL}}],
$$

with probability ratio $\rho_{i}=\frac{\pi_{\theta}(o_{i}|q)}{\pi_{\theta_{old}}(o_{i}|q)}$ and clipped ratio $\hat{\rho_{i}}=\text\mathrm{clip}(\rho_{i},1-\varepsilon,1+\varepsilon)$.

For advantage estimation, Generalized Advantage Estimation (GAE) [Sch15] computes:

<span id="equation-11"></span>

$$
A_{t}^{\mathrm{GAE}(\gamma,\lambda)}=\sum_{l=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}
$$

where $\delta_{t}=r_{t}+\gamma V(s_{t+1})-V(s_{t})$ is the TD error, and $(\gamma,\lambda)$ control the bias-variance tradeoff.

Recently, DeepSeek-R1-Zero [Dee24e] implements this paradigm through Group Relative Policy Optimization (GRPO), sampling $G$ outputs $\{o_{i}\}$ [consisting of reasoning and actions] for each prompt and optimizes:

<span id="equation-12"></span>

$$
J_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{q,\{o_{i}\}}[J_{\mathrm{group}}(\theta)],
$$

where:

<span id="equation-13"></span>

$$
J_{\mathrm{group}}(\theta)=\frac{1}{G}\sum^{G}_{i=1}\min(\rho_{i}A_{i},\hat{\rho_{i}}A_{i})-\beta D_{\mathrm{KL}},
$$

while mostly similar to Eq. 3, the GRPO advantage is neural-model free and calculated as:

<span id="equation-14"></span>

$$
A_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{j}\})}{\mathrm{std}(\{r_{j}\})}.
$$

Using rule-based rewards $r_{i}$, this pure RL approach demonstrates emergent reasoning behaviors.

<span id="appendix-b"></span>

## Appendix B Extended Related Work

**Reinforcement Learning for Reasoning in LLMs.** Reinforcement learning (RL) on LLMs [Chr23, Ouy22b, Che21a, Hav24] has significantly improved LLMs’ reasoning capabilities. Notable approaches include the use of Proximal Policy Optimization Algorithms (PPO) [Sch17a] which maintains training stability while enhancing performance by clipping policy updates, Group Relative Policy Optimization (GRPO) [Dee25c] for enhancing the ability of systematic problem-solving, actor-critic methods like SAC [Haa18] and ArCHer [Zho24i] that leverages an critic to promote robust exploration and stability, and meta tokens [Goy24, Her24, Pfa24] for structured thinking. Other significant developments include Process Reward Model (PRM) [Zha25av, Lig23a] and Monte Carlo Tree Search (MCTS) based approaches [Hao23] for systematic problem-solving. On the other hand, recent advances in LLM reasoning have explored techniques to enable models to generate intermediate chain-of-thought rationales. In particular, STaR [Zel22b] iteratively leverages a small set of rationale examples along with a large dataset without rationales. Recent work like SimpleRL-Zoo [Zen25e] DAPO [Yu25g], RLOO [Koo19], Dr. GRPO [Liu25u], and Open Reasoner Zero [Hu25f] all demonstrate that minimalist, reproducible RL techniques—featuring decoupled clipping, unbiased optimization, and simple reward schemes—can significantly enhance LLM reasoning performance.

**Existing agent frameworks.** LLM-based agent architectures have evolved from early reasoning-action frameworks [Yao23b, Shi23c, Xu23b, Lin24h] to structured approaches [Liu24u, Liu23q, Hao23, Zen25c]. Multi-agent systems [Du23a, Li23t, Che23f, Wan24y] are designed for tasks with more complex interactions. Widely used platforms such as OpenAI Gym [Bro16] and specialized environments including Sokoban [Jun01], FrozenLake [Del21], and Webshop [Yao22c] provide diverse testbeds for evaluating these agents. Moreover, general-purpose systems [She23c, Wu23c, Hao23a, Zhu23a, Xie23a] have enabled broad applications ranging from web navigation and search [Qi25a, Jin25b, Wei25g, Jin25f], coding copilot [Jim24, Dee24b, Wan24z] to GUI [Qin25a, Yao22c], Game [Hu25g] and embodied tasks [Lin24g, Xi24a, Li25ad, Fen25b]. Social interaction capabilities have been advanced through Generative Agents and AgentSims [Par23a, Lin23b]. Challenges persist in architectural complexity and self-correction [He25c], especially for diverse, multi-step reasoning tasks [Wan25al, Ngu24a, Son24b].

<span id="appendix-c"></span>

## Appendix C Detailed Experimental Settings

### C.1 Environments and Tasks

<span id="figure-08"></span>

![Figure 8. **Bi-Arm Bandits environment. The agent chooses between a low-risk arm (Phoenix) and a high-risk yet high-reward arm (Dragon), each linked to symbolic semantics. The agent learns to choose stable reward at early stages and reasons to pursue maximal expected reward and shift toward strategic risk-taking. **](../../papers/ragen-multi-turn-agent-rl/figure-08.png)

**Figure 8.** **Bi-Arm Bandits environment. The agent chooses between a low-risk arm (Phoenix) and a high-risk yet high-reward arm (Dragon), each linked to symbolic semantics. The agent learns to choose stable reward at early stages and reasons to pursue maximal expected reward and shift toward strategic risk-taking. **

<span id="figure-09"></span>

![Figure 9. Sokoban and Frozen Lake environments. For each environment, the left shows the agent-observed text rendering; the right is a visual illustration. (a) Sokoban is a deterministic multi-turn puzzle where the agent pushes boxes onto targets. (b) Frozen Lake combines multi-turn reasoning and stochasticity where the agent needs to reach the gift to succeed.](../../papers/ragen-multi-turn-agent-rl/figure-09.png)

**Figure 9.** Sokoban and Frozen Lake environments. For each environment, the left shows the agent-observed text rendering; the right is a visual illustration. (a) Sokoban is a deterministic multi-turn puzzle where the agent pushes boxes onto targets. (b) Frozen Lake combines multi-turn reasoning and stochasticity where the agent needs to reach the gift to succeed.

We construct a **diverse four-environment testbed** to evaluate LLM agents across key axes of decision-making complexity. Three environments—Bandit, Sokoban, and Frozen Lake—are symbolic, synthetic, and fully controllable, supporting clean analysis of RL learning from scratch. They are deliberately minimal and decoupled from real-world priors; even large models like GPT-4o perform poorly without training, highlighting the need for grounded policy learning. Complementarily, we include WebShop, a realistic multi-turn task involving natural language grounding and web navigation in a semi-structured interface. Together, the four environments enable systematic study of reasoning, training stability, and generalization in agentic LLMs across symbolic and open-domain settings.

Each environment stresses a distinct capability: Bandits tests reasoning under uncertainty, Sokoban emphasizes irreversible long-horizon planning, Frozen Lake involves stochastic transitions, and WebShop requires language understanding and goal-directed interaction.

**Bi-Arm Bandits.** We design this environment to evaluate whether agents can **form risk-sensitive hypotheses and revise them based on training**. At each step, the agent must choose between two semantically symbolic options—e.g., “Dragon” vs. “Phoenix”—each linked to a fixed reward distribution ([Figure 8](#figure-08)). The low-risk arm always returns a reward of $0.15$, while the high-risk arm samples from $\mathrm{Bernoulli}(0.25)$: higher variance, higher expected return.

Importantly, the low-risk arm wins more often per trial, even though the high-risk arm is better in expectation. This designed to test reasoning: without inductive bias, models may prefer the lo-arm due to its more frequent success, but a reasoning agent must learn to associate symbolic cues (e.g., "Dragon") with underlying reward statistics, override misleading short-term signals, and “justify” high-risk choices based on long-term expected return. We further test this by reversing the symbolic labels to probe agent’s reasoning under opposed reward systems.

**Sokoban.** We use the puzzle Sokoban ([Figure 9](#figure-09)) to study multi-turn agent interaction. The agent must push a box to the goal in a grid within constrained steps. Unlike standard navigation, Sokoban is irreversible: boxes can only be pushed, not pulled back, which requires the agent to reason ahead to avoid dead-ends. The reward signal encourages efficiency and accuracy: $+1$ for each box on target, $-1$ for off-target boxes, $+10$ upon task completion, and $-0.1$ per action.

**Frozen Lake.** This environment ([Figure 9](#figure-09)) combines long-horizon decision-making with stochastic transitions. The agent navigates a grid with slippery tiles; each action succeeds with probability $1/3$ and deviates perpendicularly with probability $2/3$. The agent should reach the goal without falling into holes. Rewards are sparse: successful trials receive a reward of $+1$, with all others $0$.

**WebShop.** To complement the symbolic environments, we include WebShop [Yao22c], a multi-turn web-based shopping task that tests agents’ ability to ground natural language queries, navigate semi-structured interfaces, and retrieve goal-relevant information. The agent must select a product that matches a user’s request by issuing search queries, clicking links, and reading product descriptions—introducing realistic language grounding and action space challenges absent in purely symbolic settings.

### C.2 Training and Evaluation Settings

We conduct our experiments using Qwen2.5-0.5B-Instruct [Yang24], trained via the StarPO variants with a maximum of 200 rollout–update iterations on NVIDIA H100/A100 GPUs leveraging the veRL repository [+verl]. Considering the extremely long training time of WebShop due to its long-context nature, we train on WebShop for 100 steps. Each rollout consists of $K=16$ trajectories per environment group, based on prompt size $P=8$ and maximum 5 interaction turns per episode. Agents are allowed up to 5 actions per turn and 10 actions per episode. The update batch size is $E=32$, with mini-batch size 4 per GPU. Policy optimization uses GAE with $(\gamma=1.0,\lambda=1.0)$ and Adam with $(\beta_{1},\beta_{2})=(0.9,0.999)$. We use entropy regularization ($\beta=0.001$) For experiments with vanilla StarPO we use a KL coefficient of 0.001, using the $\rm k1$ estimation [+kl]. without KL loss term during training, following [Yu25g], and track KL post-hoc. We impose a format penalty of $-0.1$ if the agent fails to output valid structured responses (e.g., missing `<think>` or `<answer>` tags), encouraging adherence to response conventions. To accelerate rollout generation, we disable enforce_eager and retain the computation graph across prefill and sampling in vLLM. We utilize Fully Sharded Data Parallel (FSDP) training strategy for multi-GPU experiments. For distributed training, we employ Ray as the multi-processing backend with XFORMERS attention implementation.

[+verl]: https://github.com/volcengine/verl

[+kl]: http://joschu.net/blog/kl-approx.html

For evaluation, we choose a fixed 256 input prompts per environment and decode using temperature $T{=}0.5$, sampling stochastically to better capture robustness in agent behaviors. Episode truncation occurs after 5 turns or 10 total actions.

### C.3 Evaluation Metrics

To track agent learning dynamics and detect training instabilities, we monitor the following metrics throughout training. Except for the success rate, which is evaluated on a fixed validation set, all metrics are computed over validation instances.

- **Average Success Rate.** Measures task completion accuracy on a fixed set of validation prompts. An episode is considered successful if the agent solves the task (e.g., pulling the high-reward arm in Bandit, pushing all boxes to targets in Sokoban, reaching the goal in Frozen Lake, and a successful purchase in WebShop).
- **Rollout Entropy.** Computes the average token-level entropy of sampled responses, capturing the exploration level and policy uncertainty. A sharp entropy drop may indicate premature policy convergence or collapse.
- **In-Group Reward Variance.** Measures reward standard deviation across rollouts sampled from the same prompt group. High in-group variance reflects diverse behaviors and learning potential; a sudden collapse indicates reward homogenization and policy stagnation.
- **Total Response Length.** Average number of tokens generated per rollout, measuring the verbosity and reasoning depth of the agent. Fluctuations in length may signal changes in planning style or confidence.
- **Gradient Norm.** $\ell_{2}$ norm of the policy gradient vector, used as a proxy for training stability. Spikes often correlate with phase transitions in policy behavior or unstable reward signals.

These metrics provide complementary views of policy quality, update dynamics, and reasoning behavior, helping diagnose when and why agent training succeeds or fails.

<span id="appendix-d"></span>

## Appendix D Results on Larger Models and Various Optimization Algorithms

We extend all our evaluation to 3B/7B/72B scale model and explore the effects of various algorithm choices like KL removal and asymmetric clipping, together with turn-aware optimization techniques including Generalized Advantage Estimation (GAE) and response masking.

**Scaling Effects.** We extend our trained model to 3B / 7B to evaluate scaling effects of RL training. Results can be found in [Figure 10](#figure-10). Note that due to the extremely long context length of WebShop could due to OOM Error within 4xH100 for 7B models, we only report 3B performance in WebShop task. On **Bandit** and **WebShop**, the larger model demonstrates significantly stronger performance compared to smaller models. However, the improvements are marginal on **Sokoban** and **FrozenLake**. We attribute this discrepancy to the nature of the environments: Sokoban and FrozenLake are symbolic, grid-based tasks with minimal overlap with pretraining data, limiting the model’s ability to leverage language priors. In contrast, Bandit and WebShop involve natural language interactions, allowing pretrained models to more effectively exploit linguistic patterns for policy learning, even in the absence of explicit environment dynamics. This can be further validated through cases in [Figure 16](#figure-16), [17](#figure-17), [18](#figure-18), [19](#figure-19), where semantic-rich tasks like Bandit and WebShop presents significantly more diverse reasoning patterns and better benefit from scale.

<span id="figure-10"></span>

![Figure 10. **Scaling effect across environments. The larger model outperforms smaller models on Bandit and WebShop, which benefit from language priors, but shows limited gains on symbolic, grid-based environments like Sokoban and FrozenLake.**](../../papers/ragen-multi-turn-agent-rl/figure-10.png)

**Figure 10.** **Scaling effect across environments. The larger model outperforms smaller models on Bandit and WebShop, which benefit from language priors, but shows limited gains on symbolic, grid-based environments like Sokoban and FrozenLake.**

**Frontier Model Performance.** To contextualize our small model’s performance, we evaluate two large foundation models—**GPT-4o** and **Qwen2.5-72B-Instruct**—on `SimpleSokoban` and `FrozenLake` in a zero-shot setting. Both models are prompted with task instructions and example formats, without any fine-tuning or in-context trajectory rollouts. See [Table 5](#table-05)

<span id="table-05"></span>

![Table 5. **Zero-shot vs. trained performance.** Our 0.5B model, trained with only 4 responses per prompt, achieves performance comparable to that of large foundation models without any fine-tuning. Response length and effectiveness values for GPT-4o / Qwen are reported per environment.](../../papers/ragen-multi-turn-agent-rl/table-05.png)

**Table 5.** **Zero-shot vs. trained performance.** Our 0.5B model, trained with only 4 responses per prompt, achieves performance comparable to that of large foundation models without any fine-tuning. Response length and effectiveness values for GPT-4o / Qwen are reported per environment.

While GPT-4o and Qwen2.5-72B achieve 19–28% success rates in Sokoban and FrozenLake without any task-specific adaptation, our 0.5B model reaches **20.70%** and **21.48%** respectively after training from scratch. This result is notable given that our model has over **100$\times$ fewer parameters**. It highlights that even under strict resource constraints, careful rollout construction and policy optimization (see [Section 4.3](#section-4)) can match the generalization ability of significantly larger models.

**Gradient Shaping.** We evaluate the effectiveness of KL Term Removal and Clip-Higher [Yu25g], finding them useful by simply extending from single-turn static tasks to agent tasks:

- **KL Term Removal:** We eliminate the KL divergence penalty from PPO’s objective, relying only on policy loss and entropy bonus for gradient updates. It removes the constraint to stay close to the initial model distribution and encourage the model to explore.
- **Clip-Higher (Asymmetric Clipping):** We decouple the PPO clipping range by using a higher upper bound ($\varepsilon_{\mathrm{high}}=0.28$) than the lower bound ($\varepsilon_{\mathrm{low}}=0.2$). It allows the model to learn more aggressively from high-reward rollouts for more effective training.

As shown in [Figure 11](#figure-11), both methods boost the success rate and extend stable training phases, showing how multi-turn RL benefits from more flexible gradient shaping.

<span id="figure-11"></span>

![Figure 11. **Effect of KL removal and asymmetric clipping on PPO stability.** The two designs both improve peak performance and delay collapse in multi-turn RL.](../../papers/ragen-multi-turn-agent-rl/figure-11.png)

**Figure 11.** **Effect of KL removal and asymmetric clipping on PPO stability.** The two designs both improve peak performance and delay collapse in multi-turn RL.

**Response Masking and Bi-Level GAE.** Following the turn-aware optimization strategy proposed in [Wan25ak], we evaluate the effect of response masking and bi-level GAE on a 0.5B model. As shown in [Figure 12](#figure-12), both techniques contribute to improved performance in multi-turn RL tasks, which highlight the promise of turn-aware RL training algorithms for stabilizing and enhancing language agent training.

<span id="figure-12"></span>

![Figure 12. **Ablation on turn-aware optimization strategies on Sokoban Task. Both response masking and bi-level GAE improve multi-turn RL performance.**](../../papers/ragen-multi-turn-agent-rl/figure-12.png)

**Figure 12.** **Ablation on turn-aware optimization strategies on Sokoban Task. Both response masking and bi-level GAE improve multi-turn RL performance.**

<span id="appendix-e"></span>

## Appendix E When Does Uncertainty-Based Filtering Help?

We hypothesize that the effectiveness of StarPO-S largely depends on the variance of rollout rewards within each environment. In settings where the task is either too easy or too difficult, the generated trajectories tend to have very low intra-group variance—meaning the model is overconfident or uniformly poor across samples. In such cases, standard StarPO may propagate misleading gradients, while StarPO-S helps by filtering out low-confidence rollouts. Conversely, in open-ended or more diverse environments (like WebShop), the rollout variance tends to be naturally high, which reduces the marginal benefit of StarPO-S filtering.

<span id="figure-13"></span>

![Figure 13. **Comparison of success rate (top) and rollout variance (bottom). StarPO-S basically improve training stability on the environments having extremely easy or difficult problems like Sokoban and Frozen Lake. This would ead to instances with small rollout Stds which can be easily filtered out through StarPO-S to make training more stable. On Tasks like WebShop, the rollout Std is already consistently high, and StarPO itself already can achieve good performance. **](../../papers/ragen-multi-turn-agent-rl/figure-13.png)

**Figure 13.** **Comparison of success rate (top) and rollout variance (bottom). StarPO-S basically improve training stability on the environments having extremely easy or difficult problems like Sokoban and Frozen Lake. This would ead to instances with small rollout Stds which can be easily filtered out through StarPO-S to make training more stable. On Tasks like WebShop, the rollout Std is already consistently high, and StarPO itself already can achieve good performance. **

[Figure 13](#figure-13) supports this intuition. The top row shows the success rates of StarPO and StarPO-S across four environments, while the bottom row shows the evolution of `in_group_std` and `chosen_in_group_std` over training. In Bandit, Sokoban, and FrozenLake, StarPO-S consistently outperforms StarPO, with the gap widening as the rollout variance drops. In WebShop, however, the high and stable variance suggests more diversity in generated responses, making the filtering effect of StarPO-S less critical—explaining the smaller performance gap.

These results indicate that StarPO-S is most beneficial when environments exhibit low rollout uncertainty, providing a simple diagnostic for when to apply it.

<span id="appendix-f"></span>

## Appendix F Case Study: The Emergence of Echo Trap with RL

<span id="table-06"></span>

![Table 6. **Examples of reasoning patterns in the Bandit task. Top rows show diverse reasoning from model before training; bottom rows show repetitive and collapsed reasoning after RL training.**](../../papers/ragen-multi-turn-agent-rl/table-06.png)

**Table 6.** **Examples of reasoning patterns in the Bandit task. Top rows show diverse reasoning from model before training; bottom rows show repetitive and collapsed reasoning after RL training.**

<span id="figure-14"></span>

![Figure 14. **Reasoning length over training iterations across different tasks. We track the average token count of reasoning segments (`<think>` blocks) during RL training. Across all environments, reasoning length declines as training progresses, with `BanditRev` maintaining longer traces—possibly due to greater semantic-reward conflict requiring more deliberation.**](../../papers/ragen-multi-turn-agent-rl/figure-14.png)

**Figure 14.** **Reasoning length over training iterations across different tasks. We track the average token count of reasoning segments (`<think>` blocks) during RL training. Across all environments, reasoning length declines as training progresses, with `BanditRev` maintaining longer traces—possibly due to greater semantic-reward conflict requiring more deliberation.**

We show cases that demonstrates the Echo Trap in RL training. As shown in [Table 6](#table-06), the top rows (Step 0) display varied hypotheses about Dragon and Phoenix, while the bottom rows (Step 150) converge to near-identical phrasing focused on “choosing Dragon” without justification.

<span id="appendix-g"></span>

## Appendix G Comparing Agent RL with Supervised Fine-Tuning

Apart from StarPO for RL training, we also employ Supervised Fine-tuning (SFT) as another agent training approach, evaluating it on the Sokoban and Frozen Lake task. We employ LoRA with a rank of 64 and an alpha value of 32, targeting all linear layers in the model. The SFT process uses a learning rate of 1e-4 with a training batch size of 128. We generate ground-truth trajectory data through breadth-first search (BFS), setting a maximum depth of 100 to create 1,000 training samples and 100 test samples. For SFT, we structure the multi-turn interaction as a conversational format. At each turn, the model must generate the next action from the ground-truth trajectory, encapsulating its response within `<answer> </answer>` tags to maintain format consistency.

We analyze the comparative performance of SFT against our stable RL baseline StarPO-S. SFT achieves 74.6% and 23% performance on Sokoban and Frozen Lake, respectively, Compared to the 20.3% and 21.8% performance with StarPO-S. The results indicate that SFT demonstrates superior performance to RL approaches. We draw conclusions from the results that although rule-based RL show promise for agent tasks, there is still a need to build more scalable and effective agent RL algorithms to achieve human-comparable performance with solely model self-evolution.

<span id="appendix-h"></span>

## Appendix H Efficient Training with Low-Rank Adaptation (LoRA)

**Motivation.** While the main body of the paper reports results obtained by full-parameter fine–tuning, in practice such a setting may be prohibitive when scaling to larger models or longer-horizon tasks. We therefore implement a parameter-efficient variant of RAGEN based on Low-Rank Adaptation [Hu21] [+lora].

[+lora]: We set rank $r{=}64$, $\alpha{=}64$, and inject adapters into all linear projections of the transformer blocks. We also increased learning rate by $10\times$ for both actor and critic.

**Performance parity.** Despite updating only a fraction of the model parameters, LoRA reaches a validation success rate comparable to that achieved by full fine-tuning of the entire network for the SimpleSokoban task, achieving approximately a $0.2\%$ success rate on the validation set.

**Resource savings.** We compare the hardware footprint of LoRA with full fine-tuning. Across an 80-minute training horizon we measure:

- **GPU memory.** LoRA stabilizes at $\mathbf{\approx 23\%}$ of device memory versus $\mathbf{\approx 48\%}$ for full updates, cutting the peak allocation by >50 %.
- **GPU utilization.** Average GPU utilization drops from $\sim\!34\%$ to $\sim\!14\%$.
- **Power consumption.** Mean power draw decreases from $\sim\!22\%$ to $\sim\!12\%$, a $\approx 45\%$ reduction.

**Take-aways.** Parameter-efficient fine-tuning provides a practically viable alternative for RAGEN: it attains comparable policy quality while more than halving memory, compute, and power demands. Consequently, future work that scales StarPO to larger backbones or longer contexts can adopt LoRA (or other adapter-based methods) as the default optimization strategy without re-engineering the training loop.

<span id="appendix-i"></span>

## Appendix I PPO Failure Mode in Frozen Lake

Among the three evaluated environments, we observe an interesting divergence on Frozen Lake: PPO tends to collapse earlier or converge less stably than GRPO. This contrasts with the general trend where PPO demonstrates better performance, prompting further analysis.

One possible explanation lies in the environment’s long-horizon stochasticity. In Frozen Lake, agent actions always lead to highly non-deterministic transitions, and intermediate states can appear similar while leading to very different outcomes. This makes value estimation challenging. As PPO relies on a learned value function, instability in critic learning may amplify optimization noise and contribute to early collapse. GRPO, by contrast, does not rely on explicit value learning. Its reward-weighted update procedure may be more tolerant to uncertainty in these settings, leading to comparatively more stable training on Frozen Lake—even if it remains less effective in other tasks. Overall, we summarize environments with high stochasticity may pose greater challenges for value-based methods, and that critic-free approaches can serve as a useful baseline in such cases.

<span id="appendix-j"></span>

## Appendix J Prompt Templates

### J.1 Bi-Arm Bandit Environment Prompts

The Bi-Arm bandit environment implements a classic reinforcement learning problem where an agent must balance exploration and exploitation. We present the prompt templates below.

**Model Templates**

```text
<|im_start|>[system]:
{prompt}
You're a helpful assistant. You always respond by giving your answer in <answer>...</answer>. Max response length: 200 words (tokens).
<|im_end|>
<|im_start|>[user]:
{prompt}
You are playing a bandit game. Goal: Maximize your total reward by choosing which arm to pull.
Game Rules:
1. There are 2 arms, named {name_a} and {name_b}
2. Each arm has its own reward distribution, related to their names.
3. Analyze the symbolic meaning of each arm's name to guess how their reward distribution might behave.
4. Based on the symbolic meaning of their names, which arm do you think is more likely to give higher rewards on average? Choose between {name_a} and {name_b}, and output like <answer> {name_a} </answer> or <answer> {name_b} </answer>.
<|im_end|>
<|im_start|>assistant
<think>
```

### J.2 Sokoban Environment Prompts

The Sokoban environment presents a classic puzzle game where an agent must push boxes to target locations. The following sections detail the prompt structure used to interface with language models.

**Model Templates**

```text
<|im_start|>system
{prompt}
You're a helpful assistant. You always respond by first wrapping your thoughts in <think>...</think>, then giving your answer in <answer>...</answer>. Max response length: 200 words (tokens).
<|im_end|>
<|im_start|>user
{prompt}
You are solving the Sokoban puzzle. You are the player and you need to push all boxes to targets. When you are right next to a box, you can push it by moving in the same direction. You cannot push a box through a wall, and you cannot pull a box. The answer should be a sequence of actions, like <answer>Right || Right || Up</answer>
<|im_end|>
<|im_start|>assistant
<think>
```

The environment uses a grid-based representation with specific symbols for different elements:

**Grid Representation**

```text
The meaning of each symbol in the state is:
#: wall, _: empty, O: target, ✓: box on target, X: box, P: player, S: player on target
```

The instruction template only consists of available actions and restrictions:

**Instruction Template**

```text
Your available actions are:
Up, Down, Left, Right
You can make up to 10 actions, separated by the action separator " || "
```

### J.3 FrozenLake Environment Prompts

The FrozenLake environment implements a grid-world navigation task where an agent must traverse a slippery frozen surface to reach a goal. Below we detail the prompt structure used for this environment.

**Model Templates**

```text
<|im_start|>system
{prompt}
You're a helpful assistant. You always respond by first wrapping your thoughts in <think>...</think>, then giving your answer in <answer>...</answer>. Max response length: 200 words (tokens).
<|im_end|>
<|im_start|>user
{prompt}
You are solving the FrozenLake puzzle. Forbid the whole and go to the target. You may move to the unintended direction due to the slippery ice. Example answer format: <think>To forbid the hole and go to the target, I should go left then go up.</think><answer>Left || Up</answer>
<|im_end|>
<|im_start|>assistant
<think>
```

The environment uses a grid-based representation with specific symbols for different elements:

**Grid Representation**

```text
The meaning of each symbol in the state is:
P: player, _: empty, O: hole, G: goal, X: player in hole, ✓: player on goal
```

The instruction template only consists of available actions and restrictions:

**Instruction Template**

```text
Your available actions are:
Left, Down, Right, Up
You can make up to 10 actions, separated by the action separator " || "
```

<span id="figure-15"></span>

![Figure 15. **A rollout with spurious reasoning.** The final outcome is successful, but the reasoning traces across turns are inconsistent and sometimes factually incorrect. This reflects a common failure mode where the model optimizes for final rewards while bypassing coherent reasoning, resulting in noisy and potentially misleading supervision during RL training.](../../papers/ragen-multi-turn-agent-rl/figure-15.png)

**Figure 15.** **A rollout with spurious reasoning.** The final outcome is successful, but the reasoning traces across turns are inconsistent and sometimes factually incorrect. This reflects a common failure mode where the model optimizes for final rewards while bypassing coherent reasoning, resulting in noisy and potentially misleading supervision during RL training.

<span id="appendix-k"></span>

## Appendix K Generalization Evaluation Environments

To evaluate generalization beyond the training distribution, we design two new test environments besides the three training environments that vary along different axes:

- **SokobanDifferentGridVocab** modifies the visual vocabulary used to represent the grid. Instead of using the standard symbols (#, _, O, X, etc.), it maps grid cells to a new vocabulary such as `W`, `G`, `C`, etc. This tests whether the model generalizes across symbol variations while retaining underlying spatial semantics.
- **LargerSokoban** increases the grid size from $6\times 6$ to $8\times 8$ and the number of boxes from 1 to 2, introducing greater spatial complexity and longer-horizon planning demands. This setting evaluates whether the policy trained on small puzzles can scale up to more complex configurations.

These environments are not seen during training and serve to probe the agent’s generalization capability under symbol shift, size scaling, and environment shift, respectively.

<span id="appendix-l"></span>

## Appendix L Case Study: Spurious Reward from Incorrect Reasoning

While evaluating reasoning behavior in Sokoban, we observe that the model may occasionally receive non-negative or even high rewards despite exhibiting flawed or misleading reasoning traces. [Figure 15](#figure-15) presents a 3-turn rollout where the model successfully pushes the box onto the target, yet its intermediate decisions reflect incorrect assumptions about the game dynamics.

In Turn 1 and 2, the agent provides plausible but ultimately incoherent plans, such as "pushing the target" or “moving toward the box on the left,” which are either redundant or directionally incorrect. Despite these errors, the final action sequence still manages to reach the goal. Such cases increase reward signal noise, making it harder for RL training to distinguish between genuinely useful plans and coincidentally effective ones.

This highlights a key challenge in multi-turn RL with reasoning agents: *outcome-based reward alone may not sufficiently penalize poor reasoning traces*, especially in environments with sparse or delayed feedback.

<span id="appendix-m"></span>

## Appendix M Extended Case Studies

To better understand how reasoning quality evolves across model scales and environments, we present representative rollout cases across six settings: Bandit ([Figure 16](#figure-16)), Sokoban ([Figure 17](#figure-17)), FrozenLake ([Figure 18](#figure-18)), and WebShop ([Figure 19](#figure-19)), each at 0.5B and 7B model scales. We observe that **larger models tend to generate longer and more coherent reasoning chains, especially in semantic-rich decision tasks like Bandit and WebShop**. However, in grid-based environments like Sokoban and stochastic ones like FrozenLake, **both small and large models struggle with planning and alignment**, often resorting to brittle heuristics or spurious correlations. These cases align with the experiments in Section [D](#appendix-d) that Sokoban and Frozen Lake does not have a significant performance gain from larger model scales unlike Bandit and WebShop, illustrating how reasoning quality interacts with environment structure, and highlight the challenges of stabilizing reward-grounded reasoning in stochastic or under-specified settings.

<span id="figure-16"></span>

![Figure 16. **Reasoning-based arm selection in Bandit tasks across model scales. In both cases, the model must infer reward tendencies of symbolic arms (`Dragon` vs. `Phoenix`) based on prior knowledge. The 0.5B model offers a short justification rooted in symbolic association. The 7B model generates a more elaborate reasoning chain comparing stability and variance, reflecting its stronger prior knowledge and interpretive capacity. Both ultimately select `Dragon`, but through different levels of reasoning depth.**](../../papers/ragen-multi-turn-agent-rl/figure-16.png)

**Figure 16.** **Reasoning-based arm selection in Bandit tasks across model scales. In both cases, the model must infer reward tendencies of symbolic arms (`Dragon` vs. `Phoenix`) based on prior knowledge. The 0.5B model offers a short justification rooted in symbolic association. The 7B model generates a more elaborate reasoning chain comparing stability and variance, reflecting its stronger prior knowledge and interpretive capacity. Both ultimately select `Dragon`, but through different levels of reasoning depth.**

<span id="figure-17"></span>

![Figure 17. **Sokoban rollouts across model scales. At 0.5B, the model performs minimal reasoning and often issues locally valid but suboptimal actions. The 7B model demonstrates more structured planning and symbolic alignment across turns, though still exhibits inefficiencies and heuristic moves in long-horizon settings.**](../../papers/ragen-multi-turn-agent-rl/figure-17.png)

**Figure 17.** **Sokoban rollouts across model scales. At 0.5B, the model performs minimal reasoning and often issues locally valid but suboptimal actions. The 7B model demonstrates more structured planning and symbolic alignment across turns, though still exhibits inefficiencies and heuristic moves in long-horizon settings.**

<span id="figure-18"></span>

![Figure 18. **FrozenLake rollouts across model scales.** At 0.5B, the agent repeats a fixed plan regardless of outcome, suggesting limited adaptation or planning. The 7B agent receives high reward despite issuing a suboptimal command, due to stochastic transitions—highlighting the difficulty of credit assignment in such environments and the risk of reinforcing spurious patterns.](../../papers/ragen-multi-turn-agent-rl/figure-18.png)

**Figure 18.** **FrozenLake rollouts across model scales.** At 0.5B, the agent repeats a fixed plan regardless of outcome, suggesting limited adaptation or planning. The 7B agent receives high reward despite issuing a suboptimal command, due to stochastic transitions—highlighting the difficulty of credit assignment in such environments and the risk of reinforcing spurious patterns.

<span id="figure-19"></span>

![Figure 19. **WebShop rollouts illustrate the impact of model scale on long-context decision making.** At 0.5B, the agent becomes stuck in a loop, repeatedly selecting irrelevant options despite rich contextual information— indicating difficulty in long-horizon memory and goal tracking. In contrast, the 3B model executes a successful multi-step reasoning chain: narrowing search queries, navigating product options, selecting attributes, and finalizing purchase. This highlights the importance of scale for compositional planning in realistic, open-domain environments.](../../papers/ragen-multi-turn-agent-rl/figure-19.png)

**Figure 19.** **WebShop rollouts illustrate the impact of model scale on long-context decision making.** At 0.5B, the agent becomes stuck in a loop, repeatedly selecting irrelevant options despite rich contextual information— indicating difficulty in long-horizon memory and goal tracking. In contrast, the 3B model executes a successful multi-step reasoning chain: narrowing search queries, navigating product options, selecting attributes, and finalizing purchase. This highlights the importance of scale for compositional planning in realistic, open-domain environments.
