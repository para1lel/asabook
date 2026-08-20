---
title: 'RL-based Agentic Search'
createTime: 2026/08/20 19:35:00
permalink: /en/papers/rl-based-agentic-search/
pageClass: paper-reading
---

> [Minhua Lin](https://dblp.org/pid/274/1711), [Zongyu Wu](https://dblp.org/pid/322/4801-1), [Zhichao Xu](https://dblp.org/pid/146/0697-1), [Hui Liu](https://dblp.org/pid/93/4010-33), [Xianfeng Tang](https://dblp.org/pid/33/7694), [Qi He](https://dblp.org/pid/51/6972-2), [Charu C. Aggarwal](https://www.charuaggarwal.net/), [Hui Liu](https://dblp.org/pid/93/4010-31), [Xiang Zhang](https://dblp.org/pid/91/4353-1), and [Suhang Wang](https://dblp.org/pid/136/9440). First submitted to arXiv on October 19, 2025; current version v2. [A Comprehensive Survey on Reinforcement Learning-based Agentic Search: Foundations, Roles, Optimizations, Evaluations, and Applications](https://arxiv.org/abs/2510.16724). [Original PDF](/paper/rl-based-agentic-search.pdf). [DOI](https://doi.org/10.48550/arXiv.2510.16724). [TeX source](https://export.arxiv.org/e-print/2510.16724v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

The advent of large language models (LLMs) has transformed information access and reasoning through open-ended natural language interaction. However, LLMs remain limited by static knowledge, factual hallucinations, and the inability to retrieve real-time or domain-specific information. Retrieval-Augmented Generation (RAG) mitigates these issues by grounding model outputs in external evidence, but traditional RAG pipelines are often single turn and heuristic, lacking adaptive control over retrieval and reasoning. Recent advances in *agentic search* address these limitations by enabling LLMs to plan, retrieve, and reflect through multi-step interaction with search environments. Within this paradigm, reinforcement learning (RL) offers a powerful mechanism for adaptive and self-improving search behavior. This survey provides the first comprehensive overview of *RL-based agentic search*, organizing the emerging field along three complementary dimensions: (i) *What RL is for* (functional roles), (ii) *How RL is used* (optimization strategies), and (iii) *Where RL is applied* (scope of optimization). We summarize representative methods, evaluation protocols, and applications, and discuss open challenges and future directions toward building reliable and scalable RL driven agentic search systems. We hope this survey will inspire future research on the integration of RL and agentic search. Our repository is available at [https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers](https://github.com/ventr1c/Awesome-RL-based-Agentic-Search-Papers).

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) [Ouy22a, Tou23a, Zha25ak] have shown unprecedented capabilities in natural language understanding, reasoning, and generation, fundamentally reshaping how users access and interact with information. Despite these advantages, LLMs still suffer from several limitations: they are constrained by static knowledge cutoffs [Che24r], prone to factual hallucinations [Sah24], and unable to access real-time or domain-specific information. To address these challenges, the paradigm of *Retrieval-Augmented Generation (RAG)* [Lew20, Gao24e] has emerged as a popular solution. RAG combines the reasoning power of LLMs with the precision of classical information retrieval (IR) techniques such as TF–IDF [Spa72, Aiz03], BM25 [Rob95, Rob09], and link-analysis models like PageRank [Bri98, Pag99, Bia05]. By retrieving evidence from external knowledge bases and conditioning responses on this context, RAG enables LLMs to generate more accurate and factually grounded outputs, particularly in knowledge-intensive tasks [Asa24, Bor22, Fan24c].

However, traditional RAG systems [Che17] are typically single-turn and heuristic-driven: they retrieve once and generate once, lacking the ability to iteratively refine queries or adapt retrieval strategies based on intermediate feedback. Retrieved documents may be irrelevant or noisy, hindering downstream reasoning [Jia23c, Cha25b, Jin25e, Jin25b]. Moreover, LLMs often struggle to fully utilize retrieved evidence, limiting the overall effectiveness of the pipeline. These limitations motivates the development of more *agentic search systems*, where LLMs act as autonomous decision-makers that dynamically plan, retrieve, reason, and reflect over multiple steps.

To this end, researchers have proposed *search agents* i.e., LLM-based systems capable of multi-step interaction with search environments [Jia24c, Zhe25f]. Unlike traditional RAG, search agents can iteratively issue and refine queries, assess the quality of retrieved results, and dynamically adapt their strategies to solve complex, multi-hop tasks. This shift from passive retrieval to active agency represents a paradigm change in information-seeking. However, early search agents often heavily rely on handcrafted prompting [Li25k] or supervised fine-tuning [Qin23b, Asa24a], limiting their ability to autonomously discover optimal strategies.

Recently, reinforcement learning (RL) [Sut98] has emerged as a promising paradigm for developing adaptive and autonomous search agents [Jin25b, Wan25ah]. We define *RL-based agentic search* as training an LLM as a decision-making agent that interacts with a search environment, receives external feedback, and iteratively improves its strategy to maximize rewards. This formulation highlights three key aspects: (i) *autonomy*, where the agent determines its search actions; (ii) *learning*, where strategies are acquired through reinforcement rather than manual design; and (iii) *interaction*, where the agent engages in multi-turn exchanges with search environments to refine reasoning and retrieval.

Despite rapid progress, a systematic understanding of RL–based agentic search remains limited. As summarized in [Table 1](#table-01), recent surveys [Xi25a, Li25q, Gao25e] have examined agentic search from various perspectives. However, they either pay less attention to RL [Xi25a] or focus on specific sub-domains such as Deep Research [Li25q] and RAG [Gao25e]. The role of RL in enabling adaptive and autonomous search behaviors remains underexplored. In contrast, this paper presents the *first* comprehensive survey dedicated to RL-based agentic search, aiming to clarify how RL benefits agentic search across three complementary dimensions: (i) *What RL is for*, describing its functional roles in guiding retrieval, reasoning, and decision making; (ii) *How RL is used*, covering optimization strategies such as reward design, policy learning, and advanced training methods; and (iii) *Where RL is applied*, examining the scope of RL intervention from the agent level to the step and module levels. For each dimension, we review representative methods and summarize emerging trends. The overview structure of our paper is shown in [Figure 1](#figure-01).

This paper is organized as follows: [Section 2](#section-2) introduces the foundations of agentic search and RL. From [Sections 3](#section-3) to [5](#section-5), we examine RL for agentic search from the three perspectives outlined above. [Section 6](#section-6) reviews evaluation metrics and representative applications, and [Section 7](#section-7) concludes with open challenges and future directions.

<span id="figure-01"></span>

![Figure 1. Overview of RL-based Agentic Search.](../../papers/rl-based-agentic-search/figure-01.png)

**Figure 1.** Overview of RL-based Agentic Search.

<span id="table-01"></span>

![Table 1. Comparison of representative surveys and this work. ✓ indicates the topic is a primary focus; ✗ indicates limited or no coverage. Unlike prior surveys that focus on non-RL agentic RAG or general search agents, or on RL methods limited to building deep-research systems, our work uniquely unifies **RL foundations** with **agentic search behavior**, analyzing how RL benefits agentic search, how it optimizes search agents, and how such systems can be effectively evaluated.](../../papers/rl-based-agentic-search/table-01.png)

**Table 1.** Comparison of representative surveys and this work. ✓ indicates the topic is a primary focus; ✗ indicates limited or no coverage. Unlike prior surveys that focus on non-RL agentic RAG or general search agents, or on RL methods limited to building deep-research systems, our work uniquely unifies **RL foundations** with **agentic search behavior**, analyzing how RL benefits agentic search, how it optimizes search agents, and how such systems can be effectively evaluated.

<span id="section-2"></span>

## 2 Background and Preliminary

<span id="section-2-1"></span>

### 2.1 Large Language Models as Agents

LLMs [Ouy22a, Tou23a, Yan25g, Wan25n, Lin25f, Zha24t] have demonstrated remarkable capabilities in text understanding, reasoning, and generation, fundamentally reshaping how humans access and interact with information. Their success has enabled natural language interfaces to diverse knowledge resources. However, these models remain limited by static training corpora, hallucinations, and their inability to access real-time or domain-specific knowledge directly [Ji23a]. To overcome these, researchers have increasingly augmented LLMs with external information sources and decision-making capabilities. A prominent direction is *Retrieval-Augmented Generation (RAG)* [Lew20, Gao24e, Liu25r], where LLMs query external knowledge bases to ground responses in retrieved evidence. Building on this paradigm, recent advances [Qin23b, Zhe25f] further position LLMs as *agentic systems*, capable of invoking external tools such as *search engines, code interpreters, knowledge-base query APIs, and web browsers* to interact with dynamic environments and perform multi-step reasoning.

<span id="section-2-2"></span>

### 2.2 From Traditional IR to Agentic Search

<span id="section-2-2-1"></span>

#### 2.2.1 Traditional IR

In classical information retrieval (IR), the primary objective is to return a ranked list of documents that best match a user query, relying on statistical models such as TF–IDF [Spa72] and BM25 [Rob95], as well as link analysis methods like PageRank [Bri98, Pag99] that incorporates metadata beyond pure texts. Retrieval itself is the endpoint of the process, leaving users to interpret and synthesize the results. In addition, while effective for many tasks, traditional IR methods are fundamentally limited in their ability to capture complex user intent or perform multi-step reasoning [Sha25c].

<span id="section-2-2-2"></span>

#### 2.2.2 RAG

Retrieval-Augmented Generation (RAG) [Lew20] integrates retrieval into the generation process by conditioning LLM responses on retrieved documents. In its standard pipeline, the model issues a query, retrieves relevant evidence, and generates an answer based on this input. While this retrieve–then–read architecture improves factual grounding, it remains limited: RAG is typically single-turn, lacks mechanisms for adaptive query refinement, and is vulnerable to irrelevant or noisy retrievals [Jia23c, Jin25e]. Iterative extensions [Tri22a, Asa24a] allow multiple rounds of retrieval, but these approaches still position the LLM as a largely passive consumer of evidence rather than an active search agent.

<span id="section-2-2-3"></span>

#### 2.2.3 Agentic Search

Agentic search moves beyond RAG by framing the LLM as an autonomous decision-making agent. Rather than passively consuming retrieved documents, the model determines *when*, *where*, and *how* to search, and integrates retrieved evidence into its ongoing reasoning and actions. This paradigm, often instantiated as *deep research agents* [Xu25f], represents a shift from retrieval as static evidence injection to retrieval as dynamic tool use for problem solving. Formally, deep research agents are LLM-powered systems that integrate dynamic reasoning, adaptive planning, multi-turn data retrieval, tool use, and evidence synthesis to support complex informational research tasks.

<span id="section-2-3"></span>

### 2.3 Basics of Reinforcement Learning

<span id="figure-02"></span>

![Figure 2. Overview of RL components](../../papers/rl-based-agentic-search/figure-02.png)

**Figure 2.** Overview of RL components

Reinforcement Learning (RL) is a fundamental paradigm in machine learning that studies how an agent interacts with its environment to maximize cumulative rewards through trial and error [Sut98]. As illustrated in [Figure 2](#figure-02), the agent observes a state $s_{t}$ from the environment at each time step $t$, selects an action $a_{t}$ according to a policy $\pi(a_{t}|s_{t})$, and then receives a reward $r_{t}$ as the environment transitions to a new state $s_{t+1}$. The agent continuously updates its policy $\pi$ to maximize the cumulative reward over time. Formally, such an optimization problem is modeled as a Markov Decision Process (MDP), represented by a tuple $(\mathcal{S},\mathcal{A},\mathcal{T},\mathcal{R})$, where $\mathcal{S}$ is the set of possible states, $\mathcal{A}$ is the action space, $\mathcal{T}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow[0,1]$ denotes the state transition probability function, and $\mathcal{R}:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\rightarrow\mathbb{R}$ defines the reward function. The optimization objective is to learn a policy $\pi$ that maximizes the expected discounted cumulative reward $\sum_{k=0}^{\infty}\gamma^{k}r_{t+k+1}$, where $\gamma\in(0,1]$ is the discount factor.

Policy gradient methods [Sch17a, Liu25u, Fen25b] are widely used in RL-based agentic search, as they directly optimize stochastic policies over large discrete action spaces. Generally, they can be grouped into (i) *on-policy optimization*, which updates the policy from fresh rollouts (e.g., PPO [Sch17a] and GRPO [Sha24d]); and (ii) *off-policy or preference-based optimization*, which leverages offline trajectories or preference data without requiring online sampling (e.g., DPO [Raf23] and ReMix [Lia25g]).

<span id="section-2-3-1"></span>

#### 2.3.1 On-policy Optimization

On-policy algorithms interact with the environment using the current policy to collect rollouts, estimate advantages, and update the same policy that generated those samples. They are favored in large-scale LLM and agentic search training due to their ability to directly optimize behavioral policies under accurate reward signals. Within this family, two subgroups can be distinguished:

- **Critic-based algorithms.** These methods rely on an explicit *value function* or *critic* model to estimate the expected return for each state or token. The critic provides token-level feedback that reduces the variance of policy gradients and stabilizes training, but it also introduces additional computational cost and memory overhead. PPO [Sch17a] is the most widely used example of this paradigm.
- **Critic-free algorithms.** In contrast, critic-free approaches remove the value network entirely and estimate the advantage directly from relative reward statistics. Instead of relying on learned value predictions, these algorithms sample multiple responses for each input and compute a *group-based advantage* by normalizing rewards within the group. This strategy significantly reduces training complexity and GPU memory consumption while maintaining stable optimization. Representative examples include GRPO [Sha24d], Dr.GRPO [Liu25u], DAPO [Yu25g], and GiGPO [Fen25b].

**Proximal Policy Optimization (PPO)**. PPO [Sch17a] is one of the most widely used methods for training RL agents. It aims to maximize the following objective function:

<span id="equation-01"></span>

$$
\mathcal{J}_{\mathrm{PPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_{\mathrm{old}}(\cdot|x)}\left[\min\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)}A,\right.\right.\left.\left.\mathrm{clip}_{\epsilon}\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)}\right)A\right)-\beta\mathbb{D}_{\mathrm{KL}}(\pi_{\theta}\mid\mid\pi_{\mathrm{ref}})\right],
$$

where $\pi_{\theta}$ and $\pi_{\mathrm{old}}$ denote the current and previous policy models, respectively. $\pi_{\mathrm{ref}}$ is the reference model that regularizes the policy update via a KL-divergence penalty, measured and weighted by $\mathbb{D}_{\mathrm{KL}}$ and $\beta$, respectively. $x$ denotes the input samples drawn from the distribution $D$. $\mathrm{clip}_{\epsilon}$ is the clipping function with hyperparameter $\epsilon$ for stabilizing training. The advantage estimate $A$ is computed using Generalized Advantage Estimation (GAE) [Sch15], based on the reward $r$ and a learned value function $V_{\psi}$.

**Group Relative Policy Optimization (GRPO)**. GRPO [Sha24d] extends PPO by eliminating the need for a separate value function model, which often doubles memory usage. Instead, it estimates relative advantages within groups of sampled responses from the same input, leading to improved training efficiency. Specifically, for each input $x\in D$, GRPO samples a group of outputs $\{y_{1},y_{2},\cdots,y_{G}\}$ from the old policy $\pi_{\mathrm{old}}$ and optimizes the new policy $\pi_{\theta}$ by maximizing:

<span id="equation-02"></span>

$$
\mathcal{J}_{\mathrm{GRPO}}(\theta)=\mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|x)}\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)}A_{i},\mathrm{clip}_{\epsilon}\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)}\right)A_{i}\right)-\beta\mathbb{D}_{\mathrm{KL}}(\pi_{\theta}\mid\mid\pi_{\mathrm{ref}})\right],
$$

where $A_{i}$ is the advantage computed using rewards $\{r_{1},r_{2},\ldots,r_{G}\}$ corresponding to the outputs within each group:

<span id="equation-03"></span>

$$
A_{i}=\frac{r_{i}-\mathrm{mean}(\{r_{1},r_{2},\ldots,r_{G}\})}{\mathrm{std}(\{r_{1},r_{2},\ldots,r_{G}\})}.
$$

**Decoupled Clip and Dynamic Sampling Policy Optimization (DAPO)**. DAPO [Yu25g] is an emerging RL approach for training long chain-of-thought (CoT) reasoning models. Specifically, DAPO addresses several limitations of GRPO, including entropy collapse, reward noise, and training instability. It introduces four key techniques to improve RL performance in long CoT scenarios: clip-higher, dynamic sampling, token-level policy gradient loss, and overlong reward shaping. Formally, the objective function for DAPO aims to maximize the following:

<span id="equation-04"></span>

$$
\begin{aligned}
\mathcal{J}_{\mathrm{DAPO}}(\theta)= & \mathbb{E}_{x\sim\mathcal{D},\{y_{i}\}_{i=1}^{G}\sim\pi_{\mathrm{old}}(\cdot|x)}\frac{1}{G}\sum_{i=1}^{G}\left[\min\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)}A_{i},\mathrm{clip}\left(\frac{\pi_{\theta}(y|x)}{\pi_{\mathrm{old}}(y|x)},1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}\right)A_{i}\right)\right] \\
& \mathrm{s.t.},~0<|\{y_{i}\mid\mathit{it\_equivalent}(x,y_{i})\}|<G,
\end{aligned}
$$

where $A_{i}$ is the advantage estimate defined in [Equation 3](#equation-03). $\epsilon_{\mathrm{high}}$ is typically larger than $\epsilon_{\mathrm{low}}$ to provide more flexibility for increasing low-probability tokens, and $\mathit{it\_equivalent}$ is the dynamic sampling function that over-samples and filters out prompts with accuracy equal to 1 or 0. Note that the KL term is excluded in DAPO because the model distribution can diverge significantly from the initial model during the training of long CoT models.

<span id="section-2-3-2"></span>

#### 2.3.2 Off-policy Optimization

Off-policy and preference-based algorithms, in contrast, do not require new rollouts from the current policy. Instead, they learn from previously collected trajectories or explicit preference annotations, which greatly improves data efficiency and stability. These methods are particularly useful in large-scale LLM alignment and agentic search scenarios, where collecting online feedback is costly or impractical.

**Direct Preference Optimization (DPO)**. DPO [Raf23] is a representative *RL-free* approach for aligning LLMs with human preferences. Unlike conventional Reinforcement Learning from Human Feedback (RLHF) [Chr17, Sti20, Ouy22a, Zha25v], which trains a separate reward model and performs iterative policy optimization (e.g., via PPO), DPO formulates alignment as a direct probabilistic classification problem. It bypasses the explicit reward modeling and RL loop by learning directly from preference-labeled response pairs. Formally, Given a dataset $\mathcal{D}$ containing triplets $(x,y_{w},y_{l})$, where $x$ is a prompt, and $y_{w}$ and $y_{l}$ denote the *preferred (winning)* and *dispreferred (losing)* responses respectively, the preferences are assumed to be generated by an underlying latent reward function $r^{*}(y,x)$ such that $r^{*}(y_{w},x)>r^{*}(y_{l},x)$. DPO optimizes the policy $\pi_{\theta}$ to increase the relative likelihood of $y_{w}$ over $y_{l}$ with respect to a reference model $\pi_{\text{ref}}$ as:

<span id="equation-05"></span>

$$
\mathcal{J}_{\mathrm{DPO}}(\theta)=\mathbb{E}_{(x,y_{w},y_{l})\sim\mathcal{D}}\left[\log{\sigma}\left(\beta\frac{\pi_{\theta}(y_{w}|x)}{\pi_{\mathrm{ref}}(y_{w}|x)}-\beta\frac{\pi_{\theta}(y_{l}|x)}{\pi_{\mathrm{ref}}(y_{l}|x)}\right)\right],
$$

where $\pi_{\mathrm{ref}}$ is the reference model, and $\beta$ is a hyperparameter that controls the strength of this regularization. The $\sigma$ function is the sigmoid, which helps to optimize the relative probability of the two responses. By using this objective, DPO directly optimizes the policy to reflect human preferences without needing an intermediate reward model.

<span id="figure-03"></span>

![Figure 3. Illustrative framework of RL-based agentic search. RL intervenes at multiple decision points—controlling when to retrieve (retrieval control), how to formulate queries (query optimization), how to integrate evidence into reasoning (reasoning-retrieval integration), and which tools or knowledge sources to use (tool and knowledge integration).](../../papers/rl-based-agentic-search/figure-03.png)

**Figure 3.** Illustrative framework of RL-based agentic search. RL intervenes at multiple decision points—controlling when to retrieve (retrieval control), how to formulate queries (query optimization), how to integrate evidence into reasoning (reasoning-retrieval integration), and which tools or knowledge sources to use (tool and knowledge integration).

<span id="section-2-4"></span>

### 2.4 RL-based Agentic Search

In agentic search, retrieval and reasoning are embedded in a *sequential decision process* rather than executed as fixed, one-shot steps. The agent must decide *when* to search, *how* to formulate or refine queries, and *how* to incorporate retrieved evidence into multi-step reasoning. [Figure 3](#figure-03) sketches this pipeline and highlights the decision points where RL can intervene: (i) **search control** (whether/when to retrieve), (ii) **query optimization** (how to retrieve), and (iii) **reasoning integration** (how to use retrieved information).

<span id="section-2-4-1"></span>

#### 2.4.1 Comparison with Pre-RL Agentic Search

Before the introduction of RL into agentic search, most systems relied on either *structured prompting* [Zho24h, Xu24e, Wan23h, Lee24e, Che24s] or *supervised fine-tuning (SFT)* [Sch23, Asa23, Aks23, Zha25ac] to guide retrieval and reasoning behaviors.

**Prompting-based Methods.** These methods primarily depend on human-designed heuristics and pre-defined reasoning workflows. For instance, PlanRAG [Lee24e] and MetaRAG [Zho24h] employ an iterative loop in which the agent alternates between searching, generating an answer, and reflecting on its quality before deciding whether to conduct further searches. This process repeats until a satisfactory response is achieved. Similarly, Knowledge-driven CoT [Wan23h] follows a reflection chain that encourages the model to re-evaluate intermediate reasoning and adjust its strategy dynamically based on retrieved evidence. While effective, these prompting-based systems rely on fixed symbolic templates or handcrafted prompt structures that cannot adapt to unseen task distributions or dynamic retrieval environments.

**SFT-based Methods.** These methods train models on datasets of high-quality trajectories that include search, reflection, and generation actions, allowing the model to internalize these behaviors into its parameters. For example, Toolformer [Sch23] fine-tunes an LM on self-labeled data where API calls are automatically inserted into text generation. It learns to decide when and how to use external tools such as calculators or Wikipedia search engines, improving factuality without additional human supervision. Similarly, SelfRAG [Asa23] introduces *self-reflective retrieval-augmented generation*, where the model is supervised to generate both normal tokens and special *reflection tokens* (e.g., `<Retrieve>`, `<Relevant>`, `<Supported>`) that indicate when to retrieve new evidence and how well each generation is supported by retrieved passages. Despite these advances, SFT-based approaches remain fundamentally imitation-driven. They can capture correlations between context and actions but lack mechanisms for long-horizon credit assignment or outcome-driven optimization.

**Limitations and Why RL.** Despite their progress, both prompting- and SFT-based agents face inherent limitations:

- *Poor adaptivity*: Their behaviors are largely predefined or imitated from static datasets. They cannot dynamically adjust retrieval frequency or reformulate queries when facing unseen tasks or API behaviors.
- *Supervision bottleneck*: High-quality reasoning and search trajectories are costly to collect and difficult to scale across tasks, which constrains generalization and makes further improvement beyond demonstrations challenging.

RL provides a principled way to overcome these issues by optimizing the agent as a policy $\pi_{\theta}$ that interacts with an environment, receives feedback, and adapts through trial and error. Unlike SFT-based imitation, RL directly optimizes task-level rewards that integrate correctness, cost, and latency, enabling the discovery of *adaptive and efficient* retrieval policies. This paradigm allows the agent to reason about the *long-term consequences* of each search decision, moving beyond static imitation toward outcome-driven learning.

<span id="section-2-4-2"></span>

#### 2.4.2 Formalization.

Formally, RL-based agentic search can be modeled as a MDP. The goal is to train a policy $\pi_{\theta}$ that maximizes cumulative reward by taking a sequence of actions in an environment. The key components are: (i) **Agent**: The LLM policy $\pi_{\theta}$, parameterized by $\theta$, which generates actions conditioned on the current state; (ii) **Environment**: External resources the agent can interact with, such as search engine APIs, retrievers, knowledge graphs, or tool interfaces; (iii) **State ($s_{t}$)**: The current context, including the original query, intermediate reasoning traces, retrieved evidence, and action history; (iv) **Action ($a_{t}$)**: A discrete decision, such as issuing a query, reformulating an existing query, selecting documents, invoking tools (e.g., search APIs, retrievers), or terminating with a final answer; (v) **Action ($a_{t}$)**: A discrete decision, such as issuing a query, reformulating an existing query, selecting documents, invoking tools (e.g., search APIs, retrievers), or terminating with a final answer; (vi) **Reward ($r_{t}$)**: A scalar feedback signal capturing task success (e.g., answer correctness, factual consistency), process quality (e.g., query efficiency, reasoning coherence), or resource costs (e.g., API calls, latency); and (vii) **Transition ($\mathcal{T}$)**: The dynamics induced by both the environment (e.g., a search engine returning documents) and the agent’s internal updates.

<span id="table-02"></span>

![Table 2. The categorization of RL-based search agents from functional roles’ perspective.](../../papers/rl-based-agentic-search/table-02.png)

**Table 2.** The categorization of RL-based search agents from functional roles’ perspective.

<span id="section-3"></span>

## 3 What RL is for: Functional Roles in Agentic Search

RL plays a wide range of functional roles within agentic search, extending well beyond basic retrieval. In this section, we categorize these roles into five major dimensions to illustrate how RL enables agents to decide not only *when* to search, but also *how* to formulate queries, *how* to interleave reasoning with evidence, and *how* to coordinate across multiple agents and tools. [Table 2](#table-02) summarizes representative works of each RL’s role.

<span id="section-3-1"></span>

### 3.1 Retrieval Control

A core role of RL in agentic search is to control *whether, when, and how* an agent retrieves external information. Rather than being a fixed design principle, this perspective synthesizes recent trends observed across RL-based retrieval systems [Hua25e, Jin25b, Wu25q, Wan25ah], where retrieval control emerges as a central optimization target. Effective retrieval control is crucial, since excessive or unnecessary queries increase cost and latency, while insufficient retrieval risks missing critical evidence. RL enables agents to balance this trade-off by learning adaptive retrieval policies that respond to task context and uncertainty. Methods in this category address three key aspects: (i) *adaptive search decisions*—whether to retrieve or rely on parametric knowledge, (ii) *search intensity and persistence*—how often and how deeply to retrieve, and (iii) *search efficiency*—minimizing redundancy, cost, and latency while preserving task performance.

<span id="section-3-1-1"></span>

#### 3.1.1 Adaptive Search Decisions

RL enables agents to decide whether a question can be answered using internal parametric knowledge or requires external retrieval. Search-R1 [Jin25b], ReSearch [Che25i], and R1-Searcher [Son25a] are early examples that teach LLMs to invoke search engines only when necessary. Specifically, as shown in [Table 3](#table-03), these methods encourage LLMs to call a search engine to access external information when the internal knowledge is insufficient to produce an accurate answer. Building on this idea, DeepRAG [Gua25c] formulates RAG as a MDP, where complex queries are *iteratively decomposed into atomic subqueries*, each representing a focused information need. At each reasoning step, the agent decides whether to answer the subquery using its parametric knowledge or to retrieve external evidence, guided by a reward that jointly optimizes answer correctness and retrieval cost.

<span id="section-3-1-2"></span>

#### 3.1.2 Search Intensity

For complex or ambiguous queries, a single retrieval attempt may be insufficient. RL has been used to optimize the depth and persistence of the search process. Pangu DeepDiver [Shi25d] introduces *Search Intensity Scaling*, rewarding agents for intensifying retrieval when ambiguity is detected. ReZero [Dao25] rewards retry attempts after failed searches, encouraging persistence and robustness. StepSearch [Wan25ah] introduces step-wise rewards based on information gain and redundancy penalties to guide retrieval step by step.

<span id="section-3-1-3"></span>

#### 3.1.3 Search Efficiency.

Efficiency concerns both the *cost* of retrieval (e.g., number of API calls, training rollouts) and the *time* required to complete searches. R1-Searcher++ [Son25b] extends R1-Searcher by introducing a *group reward* that measures retrieval thriftiness through the variance of retrieval counts across responses, rewarding the correct answer that requires the fewest retrieval calls while penalizing redundant searches. IKEA [Hua25e] introduces knowledge-boundary–aware rewards that favor internal reasoning unless external retrieval is necessary. Search Wisely [Wu25q] improves cost efficiency by filtering low-confidence queries that are likely to yield poor results. StepSearch [Wan25ah] penalizes redundant queries with step-wise rewards, encouraging more concise retrieval strategies. ZeroSearch [Sun25b] reduces API overhead by simulating retrieval in latent space, enabling curriculum-style training without reliance on real search engines. Beyond reducing retrieval calls, ParallelSearch [Zha25af] decomposes complex questions into parallel sub-queries to maintain coverage while significantly lowering response time, and RAG-R1 [Tan25c] similarly incentivizes multi-query parallelism to enhance inference efficiency. In addition, WebThinker [Li25l] extends the notion of efficiency from search cost to reasoning behavior, applying preference optimization to align query strategies with long-horizon reasoning objectives such as correctness, tool efficiency, and thinking conciseness, thereby refining retrieval decisions through reasoning-driven feedback rather than retrieval accuracy alone.

<span id="section-3-2"></span>

### 3.2 Query Optimization

Even when retrieval is triggered, the quality of queries strongly influences outcomes. Poorly posed queries yield irrelevant or noisy results. RL is then used to refine query generation based on feedback, moving beyond static heuristics. Existing works can be categorized into (i) *conversational reformulation* and (ii) *retriever-aware optimization*.

<span id="section-3-2-1"></span>

#### 3.2.1 Conversational Reformulation

In interactive settings, user queries are often ambiguous or context-dependent, making direct retrieval unreliable. RL enables agents to reformulate such inputs into self-contained queries by framing reformulation as a sequential decision-making process. ConvSearch-R1 [Zhu25i] optimizes a rewriter policy with retrieval-based rewards, where higher rewards are assigned when reformulated queries retrieve gold passages at higher ranks. Its rewriter is first fine-tuned through SFT on data generated via retrieval-guided self-distillation, and then refined through RL using a *Rank-Incentive Reward Shaping* function that encourages ranking gold passages higher while mitigating reward sparsity. This two-stage design aligns the query rewriter with retriever preferences and improves retrieval precision in multi-turn search. MaskSearch [Wu25r] extends this paradigm by incorporating a *Rewriter Agent* to refine search queries for more comprehensive retrieval, whose outputs are further used in the reasoning traces for the SFT of the LLM. Instead of optimizing a separate rewriter policy, RAG-R1 [Tan25c] encourages the LLM itself to generate multiple parallel queries within a single prompt to improve inference efficiency and retrieval diversity. Similarly, ParallelSearch [Zha25af] trains LLMs to decompose complex or multi-hop questions into parallel sub-queries within a single reasoning turn. During RL fine-tuning, a *decomposition reward* encourages effective query breakdown, while a *search-count reward* penalizes excessive search actions, balancing reformulation granularity and retrieval efficiency.

<span id="section-3-2-2"></span>

#### 3.2.2 Retriever-Aware Optimization

While conversational reformulation focuses on resolving user-side ambiguity, retriever-aware optimization instead targets the system side of query generation. It trains agents to adapt their queries to the characteristics, biases, and feedback signals of specific retrievers. The objective is to bridge the semantic gap between LLM-generated queries and the retriever’s actual ranking behavior, thereby improving retrieval accuracy and robustness across different search infrastructures. DeepRetrieval [Jia25c] exemplifies this idea by training LLMs to produce queries that align with the biases of black-box search engines, effectively exploiting retriever behavior to maximize recall. WebThinker [Li25l] applies preference optimization to align query strategies with long-horizon reasoning objectives such as correctness, tool efficiency, and thinking conciseness, enabling the agent to refine its search behavior using reasoning-driven feedback instead of retrieval accuracy alone. ZeroSearch [Sun25b] further extends this approach by simulating retrieval environments, allowing agents to learn robust query behaviors that generalize across different retrievers while avoiding the cost and instability of real API calls. Similarly, s3 [Jia25d] introduces a lightweight RL-based searcher module decoupled from the LLM generator, enabling scalable and model-agnostic query optimization. Together, these approaches highlight the broader goal of designing retriever-aware query policies that remain effective across heterogeneous search environments.

<span id="section-3-3"></span>

### 3.3 Reasoning–Retrieval Integration

Beyond deciding *when* and *how* to search effectively, knowledge-intensive tasks often require tight coupling between reasoning and retrieval. Evidence is only valuable if it improves reasoning, and reasoning should guide what to retrieve next. RL optimizes how LLMs interleave these processes, manage context, and refine reasoning based on feedback.

<span id="section-3-3-1"></span>

#### 3.3.1 Reasoning–Search Interleaving

Beyond simply allowing retrieval during reasoning [Jin25b, Che25i], RL optimizes retrieval to enhance reasoning quality. R-Search [Zha25at] introduces an *evidence reward* to encourage high-quality query generation yielding more informative evidences. AutoRefine [Shi25e] extends the standard “search-and-think” paradigm to “search-and-refine-during-think,” rewarding intermediate refinement steps to reinforce faithful and targeted knowledge extraction. EvolveSearch [Zha25au] further strengthens reasoning–retrieval interplay through iterative cycles of SFT and RL to enhance the data efficiency during training, enabling agents to progressively refine both their reasoning paths and retrieval strategies. In contrast, MaskSearch [Wu25r] focuses on enhancing the model’s retrieval-aware reasoning ability *before* RL optimization. It introduces a *Retrieval-Augmented Mask Prediction (RAMP) pretraining task*, which teaches the model to leverage external search tools to fill masked spans with retrieved knowledge in the SFT stage. This pre-RL objective establishes a retrieval-aware prior that aligns reasoning and retrieval behaviors, enhancing the universal search capabilities across various downstream tasks.

<span id="section-3-3-2"></span>

#### 3.3.2 Context and Memory Management

While existing agentic search systems [Zha25at, Jin25b, Wan25ah] are effective for short-horizon tasks such as single-turn retrieval or step-level reasoning, they often struggle in long-horizon or multi-session settings, where agents must manage extended interaction histories within limited context windows. To operate efficiently under such constraints, agents need to *actively manage memory*—deciding what information to retain, summarize, or discard as a search episode unfolds. Recent studies [Gao25d, Xu25g, Wu25l, Che25u, Li25h, Li25y] apply RL to optimize this process, framing memory control as a sequential decision problem balancing *information fidelity* and *context efficiency*. Specifically, two complementary strategies have emerged:

- **Internal management:** The agent itself performs memory operations such as summarizing, refreshing, or pruning its working context under RL guidance. For instance, ReSum [Wu25l] trains agents with RL to generate concise summaries of past reasoning and interactions, enabling long-context reasoning without exceeding token limits. SFR-DeepResearch [Ngu25a] further introduces explicit memory actions (e.g., `clean_memory`, `store_snippet`), using RL signals to decide when to retain or discard past information, thus preventing memory overflow and redundancy.
- **External management:** Other frameworks use auxiliary summarization modules to compress historical context before reinjection into the agent’s reasoning stream. In such cases, RL or policy learning is used to determine when and how to invoke these summarizers. For example, **WebSailor** [Li25h] employs an external summarizer to condense browsing traces for multi-page search; **ASearcher** [Gao25d] dynamically summarizes multi-turn research sessions to preserve key findings; and **RECON** [Xu25g] integrates a frozen, pretrained summarizer into an RL-based search agent (e.g., Search-R1); the summarizer, trained via supervised relevance pretraining and multi-aspect distillation, enables the agent to reason over concise, factual evidence while substantially reducing context length and cost.

<span id="section-3-4"></span>

### 3.4 Multi-Agent Collaboration

Beyond relying on a single LLM to handle both reasoning and retrieval, advanced agentic search systems [Che25t, Wan25t] decompose the process into multiple specialized modules, such as query rewriting [Ma23], document selection [Ke24], and reasoning control. RL is then used to align the objectives of distinct agents, ensuring that local decisions, such as when to reformulate, which evidence to retain, and how to schedule retrieval steps, contribute to globally coherent and efficient search. Existing approaches can be broadly categorized into (i) *planner–executor architectures* and (ii) *cooperative multi-agent systems*.

<span id="figure-04"></span>

![Figure 4. Overview of RL for multi-agent collaboration. (a) *Planner–executor architecture*: a central planner coordinates specialized executor agents for task decomposition and dynamic subtask allocation. (b) *Cooperative multi-agent system*: multiple agents jointly optimize shared objectives through communication, coordination, and reward sharing.](../../papers/rl-based-agentic-search/figure-04.png)

**Figure 4.** Overview of RL for multi-agent collaboration. (a) *Planner–executor architecture*: a central planner coordinates specialized executor agents for task decomposition and dynamic subtask allocation. (b) *Cooperative multi-agent system*: multiple agents jointly optimize shared objectives through communication, coordination, and reward sharing.

<span id="section-3-4-1"></span>

#### 3.4.1 Planner–Executor Architectures

A representative paradigm is the *planner–executor architecture*, where a high-level planner orchestrates specialized executors responsible for distinct retrieval or reasoning operations. As shown in [Figure 4](#figure-04)(a), the planner acts as a meta-policy that decides which executor to invoke, when to switch subtasks, and how to allocate search or computational budgets, thus achieving *adaptive orchestration* across heterogeneous RAG modules.

**MAO-ARAG** [Che25t] exemplifies this design. It models multi-agent RAG as a *multi-agent semi-Markov decision process (MSMDP)*, where the planner coordinates executors such as query rewriters, document selectors, retrievers, and generators. Specifically, a planner agent intelligently se- lects and integrates the appropriate agents from these execu- tors into a suitable workflow tailored for each query, striving for high-quality answers while maintaining reasonable costs. During each turn, the planner agent is trained using PPO, optimizing by the following reward:

<span id="equation-06"></span>

$$
r_{t}=r_{\mathrm{F1}}-\alpha\cdot r_{\mathrm{CP}}-r_{\mathrm{FP}},
$$

where $r_{\mathrm{F1}}$ is the outcome-based reward based on F1 score, and $r_{\mathrm{CP}}$ and $r_{\mathrm{FP}}$ are the cost penalty and format penalty, respectively. These rewards together improve answer quality while keeping costs within a reasonable range.

OPERA [Liu25x] extends this idea to multi-hop retrieval and reasoning. It adopts a hierarchical RL framework composed of a high-level planning module and low-level execution agents. Three role-specific agents, including *Plan*, *Analysis–Answer*, and *Rewrite*, are optimized with Multi-Agents Progressive GRPO (MAPGRPO), a GRPO-based algorithm that provides fine-grained, role-specific credit assignment. Each agent is trained with tailored reward signals: the Plan Agent for decomposition validity, the Analysis–Answer Agent for reasoning and factual correctness, and the Rewrite Agent for retrieval relevance and formatting. This hierarchical optimization yields stable convergence and interpretable reasoning trajectories, enabling OPERA to learn cost-efficient, verifiable retrieval–reasoning workflows.

<span id="section-3-4-2"></span>

#### 3.4.2 Cooperative Multi-Agent Systems

Another workflow models the agentic search as a cooperative muti-agent games, where each module is treated as an RL agent whose actions influence retrieval outcomes, with a shared global reward aligning their behaviors toward better performance. The overall framework is illustrated in [Figure 3](#figure-03)(b). For example, SIRAG [Wan25t] trains a Decision-Maker to decide when to retrieve and a Knowledge-Selector to filter which documents should be passed downstream, with RL rewards aligning their decisions toward high-quality evidence integration. MMOA-RAG [Che25q] generalizes this setting to larger agent pools, where RL optimizes how agents share responsibilities for query reformulation, evidence selection, and verification. In addition, some works such as AgentGym-RL [Xi25b] and Chain-of-Agents [Li25j] provide general infrastructures for training multi-agent systems, where agentic search is a core evaluation setting.

<span id="section-3-5"></span>

### 3.5 Tool and Knowledge Integration

Finally, rather than relying solely on text retrieval, agentic search increasingly requires integration with heterogeneous external resources, including APIs [Jia25a], multi-modal tools [Wan25ai, Gen25a], and structured knowledge bases [Cha25c, Hao25b], to extend the scope of tasks agents can solve, where RL is a natural solution to enable them. Research in this category can be grouped into two directions: (i) *multi-tool and multi-modality reasoning*, where agents learn to coordinate across diverse toolkits such as search engines, code interpreters, and vision models, and (ii) *structured knowledge exploration*, where RL trains agents to navigate symbolic environments like knowledge graphs or tables in a goal-directed way.

<span id="section-3-5-1"></span>

#### 3.5.1 Multi-tool and Multi-modality Reasoning

Many tasks require more than text-based retrieval, demanding agents to combine computation, web search, and multimodal understanding. RL has been used to optimize tool selection and sequencing by providing feedback on whether tool calls lead to accurate reasoning or task completion. Tool-Star [Don25e] integrates six tools, including search engines and code generators, using a self-critic RL setup that rewards correct intermediate outputs. VerlTool [Jia25a] generalizes this with a unified RL framework that manages heterogeneous APIs and multi-modal LLMs (MLLMs). In multi-modal contexts, MMSearch-R1 [Wu25s], Visual-ARFT [Liu25z], and VRAG-RL [Wan25ai] extend Search-R1 paradigms to visual question answering by rewarding policies that align retrieved text and visual evidence. WebWatcher [Gen25a] further trains agents with RL to coordinate multiple tools simultaneously, handling both textual and visual inputs.

<span id="section-3-5-2"></span>

#### 3.5.2 Structured Knowledge Navigation

In many domains, critical information is stored in structured resources such as knowledge graphs (KG) or databases [Bol08, Zho17b, Lin24f, Lin25g]. RL is applied by defining traversal as a sequential decision-making process: each step selects which entity or relation to follow, with rewards reflecting correctness, coverage, or efficiency. For instance, GRAIL [Cha25c] applies RL to learn KG traversal policies that reach correct answers efficiently. DynaSearcher [Hao25b] extends this with multi-reward RL, jointly optimizing for accuracy, efficiency, and balanced exploration of KG.

<span id="section-4"></span>

## 4 How RL is Used: Optimization Strategies

This section examines how RL is applied in agentic search systems, covering training pipelines, algorithmic design, and reward mechanisms. [Table 7](#table-07) summarizes representative works with corresponding optimization strategies.

<span id="section-4-1"></span>

### 4.1 Training Regime

The training regime defines how RL is integrated into agentic search, encompassing initialization strategies, environment design, and optimization workflows. It determines how agents acquire, refine, and stabilize their decision-making policies throughout interaction-based learning.

<span id="section-4-1-1"></span>

#### 4.1.1 Standard Agentic Search Pipeline

A typical RL training pipeline for agentic search, exemplified by Search-R1 [Jin25b], comprises two stages: a *cold-start* initialization and subsequent RL fine-tuning. The cold-start phase ensures interface compliance (e.g., API calls, tool schemas) and stabilizes early rollouts. During RL training, the policy LLM receives complex queries and generates interleaved reasoning and tool-use actions within simulated or real search environments. The overall training pipeline and prompt template are summarized in [Table 3](#table-03).

<span id="table-03"></span>

![Table 3. Standard agentic search prompt template using Search-R1 as an example.](../../papers/rl-based-agentic-search/table-03.png)

**Table 3.** Standard agentic search prompt template. We use the prompt template of Search-R1 [Jin25b] as an example.

<span id="section-4-1-2"></span>

#### 4.1.2 Cold Start

A dominant paradigm initializes agents via supervised fine-tuning (SFT) before RL optimization [Li25h, Wu25d, Don25e, Son25b]. This stage equips models with baseline task competence and mitigates early instability caused by sparse rewards in long-horizon environments. For instance, Webagent-R1 [Wei25f] shows that SFT provides crucial web-interaction knowledge for downstream RL, while WebSailor [Li25h] finds that SFT accelerates convergence and stabilizes multi-step tool use. EvolveSearch [Zha25au] further introduces a self-improving SFT–RL loop, where RL-refined policies generate new demonstrations for iterative SFT retraining. Conversely, several works [Sun25b, Xi25b] question the necessity of SFT. ZeroSearch [Sun25b] replaces it with latent-space retrieval simulation, enabling pure RL training without external supervision, while AgentGym-RL [Xi25b] employs curriculum-based horizon scaling to stabilize RL-only training.

<span id="section-4-1-3"></span>

#### 4.1.3 Simulation-Based Training

Training RL agents in real-world search environments can be prohibitively expensive, slow, and non-reproducible. Simulation environments provide a controlled, accelerated, and cost-effective alternative. For example, ZeroSearch [Sun25b] proposes a novel RL framework that *simulates* search by transforming an LLM into a retrieval module, avoiding the cost and noise of real search engines during training. It employs a curriculum that incrementally degrades the quality of simulated documents, forcing the agent to become more robust. O<sup>2</sup>-Searcher [Mei25a] also leverages an efficient, locally simulated search environment for training, focusing on open-domain open-ended question answering scenarios. WebSailor-V2 [Li25y] proposes a dual-environment RL framework, utilizing a high-fidelity simulator for rapid algorithm iteration and a robust, managed real-world environment for stable final policy training. This hybrid approach addresses the challenges of both scalability and realism.

<span id="section-4-1-4"></span>

#### 4.1.4 RL Algorithms

Most RL-based search agents employ policy-gradient algorithms, particularly PPO [Sch17a], GRPO [Sha24d], and Reinforce++ [Hu25d]. Recent variants adapt these methods to the search context: Search Wisely [Wu25q] introduces $\beta$-GRPO for uncertainty-aware calibration, StepSearch [Wan25ah] implements step-wise PPO aligned with information gain, and ReinforceRAG [Zen25c] augments policy gradients with retrieval-aware baselines to mitigate variance under sparse rewards. The details of the RL algorithms applied in RL-based search agents are in [Table 7](#table-07).

<span id="section-4-1-5"></span>

#### 4.1.5 Curriculum Learning and Horizon Scaling

RL training for long-horizon search tasks remains challenging due to sparse rewards and unstable credit assignment. Curriculum learning alleviates these issues by gradually expanding task complexity or interaction length. AgentGym-RL [Xi25b] proposes *ScalingInter-RL*, which progressively extends the interaction horizon—starting from short, focused tasks and gradually scaling to multi-step reasoning—balancing exploration and exploitation. ZeroSearch [Sun25b] employs a curriculum that systematically increases retrieval noise, compelling agents to develop more resilient strategies. InfoSeek [Xia25d] similarly generates progressively harder research tasks to facilitate structured capability growth. These strategies jointly improve convergence stability and support continual capability scaling.

<span id="section-4-1-6"></span>

#### 4.1.6 Iterative and Self-Evolving Frameworks

Beyond static curricula, some frameworks close the loop between data generation and policy learning. EvolveSearch [Zha25au] epitomizes this approach: RL-trained models generate higher-quality search trajectories that are distilled back into SFT data, creating a self-reinforcing cycle of improvement. Such iterative frameworks demonstrate how RL can act not only as a training objective but as a data generator, continuously refining both model behavior and supervision quality.

<span id="table-04"></span>

![Table 4. Comparison of representative reward functions in RL-based agentic search. $a_{\mathrm{pred}}$ and $a_{\text{gt}}$ denote the predicted and ground-truth answers, respectively. $r_{\text{ans}}$ is the answer-level reward; $\mathrm{RT}$ is the number of retrieval steps; $\mathrm{RT}_{\max}$ is the maximum retrieval budget; $r_{\text{kb}+}$ and $r_{\text{kb}-}$ denote the maximal knowledge-boundary reward and a small penalty, respectively. $\mathbb{I}(\cdot)$ is the indicator function, $\gamma$ the discount factor, $v(\cdot)$ the rollout value, and $\alpha$ a decay coefficient. $r_{\text{sim}}(\cdot,\cdot)$ is the reward function based on the semantic similarity between the model-generated search query and the ground-truth query using a Sentence Transformer.](../../papers/rl-based-agentic-search/table-04.png)

**Table 4.** Comparison of representative reward functions in RL-based agentic search. $a_{\mathrm{pred}}$ and $a_{\text{gt}}$ denote the predicted and ground-truth answers, respectively. $r_{\text{ans}}$ is the answer-level reward; $\mathrm{RT}$ is the number of retrieval steps; $\mathrm{RT}_{\max}$ is the maximum retrieval budget; $r_{\text{kb}+}$ and $r_{\text{kb}-}$ denote the maximal knowledge-boundary reward and a small penalty, respectively. $\mathbb{I}(\cdot)$ is the indicator function, $\gamma$ the discount factor, $v(\cdot)$ the rollout value, and $\alpha$ a decay coefficient. $r_{\text{sim}}(\cdot,\cdot)$ is the reward function based on the semantic similarity between the model-generated search query and the ground-truth query using a Sentence Transformer.

<span id="section-4-2"></span>

### 4.2 Reward Design

Reward design is paramount in RL training for agentic search, determining which behaviors are reinforced and how credit is allocated across complex trajectories. Modern agentic search employs *multi-faceted, multi-turn reward mechanisms* that optimize not only accuracy of final outcomes and intermediate reasoning, but also diverse desiderata such as clarity, truthfulness, conciseness, efficiency, and reduced hallucination tendencies. These sophisticated reward structures can be categorized along two complementary dimensions: temporal scope (outcome vs. process-level) and objective diversity (single vs. multi-faceted optimization). [Table 4](#table-04) summarizes representative reward functions adopted in recent RL-based agentic search frameworks [Jin25b, Che25i, Dao25, Shi25d, Shi25e, Zha25at, Zha25ae, Liu25z], illustrating how different designs balance final-answer accuracy, intermediate reasoning quality, and resource-efficient retrieval.

<span id="section-4-2-1"></span>

#### 4.2.1 Outcome-level Rewards

Outcome-level rewards evaluate final task completion but increasingly incorporate multiple quality dimensions beyond simple correctness. Early approaches like Search-R1 [Jin25b] and ReSearch [Che25i] rely on basic exact match (EM) and format reward for correctness and style consistency. Subsequent **multi-faceted** extensions enhance these metrics: R-Search [Zha25at] introduces *cross-model evidence utility*, rewarding evidence quality and interpretability alongside correctness. IKEA [Hua25e] designs *knowledge-boundary shaping* to optimize both accuracy and efficiency by discouraging redundant retrieval. R1-Searcher++ [Son25b] measure *group-relative efficiency* through retriever call variance, balancing task success with resource conservation. O<sup>2</sup>-Searcher [Mei25a] introduces a *diversity reward* to encourage *query diversity* to mitigate duplication under budget constraints.

<span id="section-4-2-2"></span>

#### 4.2.2 Process-level Rewards

While outcome signals are simple and effective for general tasks, they often prove too sparse to guide learning in long-horizon, multi-step search settings [Den25b]. Process-level rewards address this limitation by providing dense, fine-grained feedback throughout the reasoning–retrieval trajectory, enabling *multi-turn, multi-faceted* optimization of intermediate behaviors, such as faithfulness [Shi25e] and efficiency [Wan25ah]. ReasonRAG [Zha25ae] introduces *shortest-path reward estimation* (SPRE), which simultaneously optimizes reasoning quality and conciseness by simulating its possible outcomes and penalizing unnecessarily long trajectories. StepSearch [Wan25ah] evaluates the utility of each retrieval step across multiple dimensions, including information gain and redundancy penalties. AutoRefine [Shi25e] reinforces faithful and targeted knowledge extraction through iterative step-level rewards. In addition to these verifiable rule-based rewards, some works [Wan25t, Den25b] also sample rewards from LLMs for providing step-level rewards to address the sparse reward and training stability or enable faithful search [Xu25h].

<span id="section-5"></span>

## 5 Where RL is Applied: The Scope of Optimization

The application of RL in agentic search can be categorized by the *architectural level* at which optimization occurs. This perspective clarifies whether RL refines specific sub-skills, optimizes the policy of a single agent, or orchestrates behavior across multi-agent or system-wide search infrastructures. We summarize representative works across these three levels of scope in [Table 5](#table-05).

<span id="table-05"></span>

![Table 5. The categorization of RL-based search agents from the optimization scope’s perspective.](../../papers/rl-based-agentic-search/table-05.png)

**Table 5.** The categorization of RL-based search agents from the optimization scope’s perspective.

<span id="section-5-1"></span>

### 5.1 Agent-level Scpoe

At the agent level, RL optimizes end-to-end search policies, either for single autonomous search agents or coordinated multi-agent search systems. This scope captures how RL shapes the core search decision-making processes that define effective information-seeking behavior.

<span id="section-5-1-1"></span>

#### 5.1.1 Single-agent Optimization

This is the most prevalent paradigm, where RL directly optimizes a unified policy governing the agent’s entire search workflow. The agent learns when to retrieve, how to formulate queries, how to interpret evidence, and when to terminate its search. Search-R1 [Jin25b] exemplifies this approach, training an LLM to autonomously decide when and how to invoke external search engines during reasoning. R1-Searcher++ [Son25b] extends this by balancing internal knowledge use with external search reliance. Web-based agents such as WebSailor [Li25h] and WebDancer [Wu25d] demonstrate RL’s potential to train robust, long-horizon search policies for complex web environments.

<span id="section-5-1-2"></span>

#### 5.1.2 Multi-agent Coordination

For more complex search pipelines, distinct agents specialize in search-related functions such as query reformulation, document selection, and evidence synthesis. RL coordinates these specialized search agents to achieve coherent information-seeking behavior. SIRAG [Wan25t] jointly trains a *Decision Maker* to control search timing and a *Knowledge Selector* to filter retrieved documents under a shared reward function. MAO-ARAG [Che25t] orchestrates multiple search specialists (e.g., query reformulators, document selectors, answer generators) using RL to optimize their collaborative search performance.

<span id="section-5-2"></span>

### 5.2 Module-Level & Step-level Scope

This scope focuses on optimizing specific search components or decision steps within broader agentic search workflows. Instead of training the entire agent policy end-to-end, RL refines localized behaviors, making it valuable for improving specific aspects of the search pipeline.

<span id="section-5-2-1"></span>

#### 5.2.1 Module-level Optimization

RL can enhance specialized modules that operate alongside frozen LLMs. This modular approach isolates search-specific capabilities for targeted improvement without full-model retraining. The s3 [Jia25d] exemplifies this strategy by training a lightweight searcher module while keeping the generator frozen, ensuring efficiency and model-agnostic adaptability. AI-SearchPlanner [Mei25b] follows a similar design, training a retrieval-planning module to decide when and how to query while leveraging a frozen QA model for final answer generation.

<span id="section-5-2-2"></span>

#### 5.2.2 Step-level Optimization

RL can also provide fine-grained feedback on individual search actions, such as query generation, document selection, or refinement. StepSearch [Wan25ah] provides step-wise rewards based on information gain and redundancy penalties to encourage concise, effective search. AutoRefine [Shi25e] reinforces iterative “search-and-refine” behaviors, encouraging agents to iteratively improve their information gathering. Search Wisely [Wu25q] applies RL to control retrieval confidence, discouraging low-confidence searches that waste resources.

<span id="section-5-3"></span>

### 5.3 System-level Scope

At the system level, RL orchestrates comprehensive search infrastructures and multi-agent search ecosystems. Rather than optimizing individual search agents, this scope addresses how RL can improve entire search system architectures, resource allocation, and search workflow management across complex information-seeking platforms.

<span id="section-5-3-1"></span>

#### 5.3.1 Unified RL-based Framework for Search

Several recent works build general-purpose platforms for developing, training, and evaluating RL-based search agents. AgentGym-RL [Xi25b] provides a modular benchmark suite that supports diverse RL algorithms across multiple information environments. RAG-Gym [Xio25a] offers structured environments for optimizing retrieval-augmented agents and systematically comparing reward and policy designs. VerlTool [Jia25a] extends this trend to tool-augmented systems, offering unified APIs and environments for training agents that operate over heterogeneous information sources and modalities.

<span id="section-6"></span>

## 6 Evaluation and Application

Evaluating RL-based agentic search systems requires multi-dimensional assessment across search effectiveness, reasoning quality, efficiency, and generalization. This section reviews the datasets, evaluation metrics, and application domains that currently define the landscape of RL-based agentic search evaluation and deployment.

<span id="table-06"></span>

![Table 6. The categorization of commonly used datasets in RL-based agentic search.](../../papers/rl-based-agentic-search/table-06.png)

**Table 6.** The categorization of commonly used datasets in RL-based agentic search.

<span id="section-6-1"></span>

### 6.1 Datasets

RL-based agentic search is evaluated across diverse benchmarks that test retrieval effectiveness and reasoning ability in open-domain, web-based, and domain-specific settings. [Table 6](#table-06) summarizes these representative datasets and the corresponding studies that adopt them. Next, we give the details.

<span id="section-6-1-1"></span>

#### 6.1.1 Knowledge-Intensive QA Benchmarks

A primary evaluation setting for agentic search is *knowledge-intensive question answering (QA)*, where answering a question requires retrieving external evidence beyond the model’s parametric knowledge. These benchmarks jointly evaluate the agent’s ability to (i) retrieve relevant information and (ii) synthesize evidence into correct, verifiable answers. Natural Questions (NQ) [Kwi19a] and TriviaQA [Jos17] serve as foundational single-hop QA datasets, widely used in works such as Search-R1 [Jin25b] and R-Search [Zha25at], to test when and how agents invoke retrieval. For multi-hop reasoning, HotpotQA [Man18] is employed in ReSearch [Che25i] and AutoRefine [Shi25e], requiring iterative retrieval and reasoning over multiple evidence chains. Fact-checking tasks such as FEVER [Tho18] further test retrieval faithfulness and evidence verification. HARIS [Hu25e], for instance, uses FEVER to train agents that assess the credibility of retrieved claims under RL signals.

<span id="section-6-1-2"></span>

#### 6.1.2 Web-based Search Benchmarks

Web environments provide more realistic and dynamic evaluation settings. WebQA [Cha22b] offers large-scale web-based QA tasks used in WebThinker [Li25l]. GAIA (General AI Assistant) defines multi-step, interactive web tasks requiring reasoning and tool coordination, serving as a key benchmark for AgentGym-RL [Xi25b] and WebSailor-V2 [Li25y]. Mind2Web [Gou25] and related web navigation datasets evaluate the ability of web agents such as WebDancer [Wu25d] to handle multi-hop web browsing and action planning.

<span id="section-6-1-3"></span>

#### 6.1.3 Knowledge Sources

Most open-domain and web-based agents rely on large-scale text corpora as retrieval backends. Common choices include the English Wikipedia dump [Wik25], widely used in benchmarks such as NQ, TriviaQA, and HotpotQA; web-scale resources such as Common Crawl [Com25] and KILT [Pet20]; and domain-specific knowledge bases such as PubMed [Nat25] and arXiv [Arx25], which support research-oriented agents [Zhe25f, Yu25i]. Some systems, including DeepResearcher [Zhe25f] and WebThinker [Li25l], further augment these static corpora with dynamic web-search APIs to access up-to-date or domain-targeted information.

<span id="section-6-1-4"></span>

#### 6.1.4 Multi-modal Search

Recent advances in agentic search [Wu25s, Liu25z] extend beyond text-only retrieval to incorporate visual and structured modalities, motivating new benchmarks for *multi-modal search*. Early datasets, e.g., **InfoSeek** [Che23e] and **SlideVQA** [Tan23], established vision–language question answering over slides and figures, bridging perception and reasoning. Building on this foundation, [Liu25z] introduce *MAT-Search* and *MAT-Coding* to evaluate agentic retrieval and tool-use abilities under verifiable reward signals. **MFC-Bench** [Wan24x] benchmarks multimodal fact-checking with $35k$ image–text samples across manipulation, out-of-context, and veracity subtasks, providing a large-scale testbed for factual grounding. Meanwhile, **MMLongBench-Doc** [Ma24c] focuses on long-context multimodal document understanding, covering $135$ lengthy documents that combine text, layout, tables, and charts. Together, these benchmarks advance RL-based agentic search toward unified, perception-grounded multi-modal retrieval and reasoning.

<span id="section-6-1-5"></span>

#### 6.1.5 Conversational and Multi-turn Search

CoQA [Red19] and QuAC [Cho18] benchmark the ability of agents to maintain context across multi-turn interactions, as explored in ConvSearch-R1 [Zhu25i]. MSMarco [Baj16] evaluates large-scale passage retrieval and ranking, assessing an agent’s ability to locate relevant information efficiently, as applied in DeepRetrieval [Jia25c] and RAG-Gym [Xio25a].

<span id="section-6-1-6"></span>

#### 6.1.6 Domain-specific Search Tasks

Some specialized datasets [Wel17, Cla18, Tal19, Hen20] target specific reasoning domains. For instance, SciQ [Wel17] and ARC [Cla18] focus on scientific reasoning, relevant to agents like DeepResearcher [Zhe25f]. CommonsenseQA [Tal19] tests the integration of factual retrieval and commonsense reasoning, used in IKEA [Hua25e]. MMLU [Hen20] evaluates general knowledge breadth, serving as a multi-domain benchmark for tool-augmented systems such as Tool-Star [Don25e].

<span id="section-6-2"></span>

### 6.2 Metrics

Evaluating RL-based agentic search requires metrics that capture multiple dimensions of performance, including answer quality, retrieval effectiveness, efficiency, and process-level behavior.

<span id="section-6-2-1"></span>

#### 6.2.1 Answer Quality

Exact Match (EM) and F1 score are two of the most commonly used metrics, which provide direct measures of task success, serving as primary evaluation metrics in many works [Jin25b, Dao25]. To evaluate the generated answer quality against reference responses, ROUGE and BLEU scores evaluate generated answer quality against reference responses. To handle the case that answers may be correct but phrased differently from gold standards, BERTScore [Zha19i] is applied in RAG-Gym [Xio25a].

<span id="section-6-2-2"></span>

#### 6.2.2 Search Effectiveness

To measure the quality of the retrieved information, several traditional information retrieval metrics remain fundamental. Specifically, *Precision*, *Recall*, and *F1* measure the quality of retrieved information. Mean Reciprocal Rank (MRR) and Normalized Discounted Cumulative Gain (NDCG) evaluate ranking quality when systems need to prioritize multiple search results. For example, DeepRetrieval [Jia25c] trains LLMs to generate queries that maximize the retrieval performance of black-box search engines in terms of retrieval metrics like Recall and NDCG.

<span id="section-6-2-3"></span>

#### 6.2.3 Search Efficiency

It aims to measure search agents’ efficiency from both resource and latency cost perspectives. *Number of Search Queries* [Shi25d] measures how many queries an agent issues, while *API Call Cost* [Che25t] quantifies the expense of invoking external services. *Response Time* assesses end-to-end latency, important for interactive settings. *Search Redundancy* [Son25b] captures repeated or semantically similar queries that waste resources.

<span id="section-6-2-4"></span>

#### 6.2.4 Process Metrics

Beyond end-task accuracy, several works assess intermediate behaviors. StepSearch [Wan25ah] defines *Information Gain* per retrieval step to quantify the utility of each search action. SIRAG [Wan25t] measures *Query Quality Score* via LLM-as-Judge to evaluate whether generated queries are likely to yield relevant evidence. R-Search [Zha25at] introduces *Evidence Utilization Rate* to measure how effectively agents leverage retrieved information in final reasoning.

<span id="section-6-3"></span>

### 6.3 Applications

The progress in RL-based agentic search has led to broad practical applications spanning scientific research, software development, multi-modal reasoning, and conversational AI.

<span id="section-6-3-1"></span>

#### 6.3.1 Deep Research

Scientific and academic research represents a major application domain for RL-based search agents. DeepResearcher [Zhe25f] demonstrates automated literature review and hypothesis generation through RL-optimized search strategies across academic databases. MedResearcher-R1 [Yu25i] specializes in medical research, using RL to navigate complex biomedical knowledge bases and synthesize clinical evidence. WebResearcher [Qia25c] extends research capabilities to general web-based investigation with unbounded reasoning horizons. SFR-DeepResearch [Ngu25a] focuses on autonomous reasoning for research tasks, while Atom-Searcher [Den25b] enhances deep research through fine-grained atomic thought rewards. WebThinker [Li25l] is a deep research agent empowered with comprehensive research capabilities across diverse domains through iterative online DPO.

<span id="section-6-3-2"></span>

#### 6.3.2 Multi-modal Search

In addition to text-only search, there are several recent efforts [Wu25s, Wan25ai] exploring multi-modality search agents, combining both text and visual information. VRAG-RL [Wan25ai] enables vision-perception-based RAG for visually rich information understanding, using RL to iteratively reason across both textual and visual content. Visual-ARFT [Liu25z] demonstrates visual agentic reinforcement fine-tuning for tasks requiring integrated visual and textual search. WebWatcher [Gen25a] breaks new ground in vision-language deep research agents, combining web search with visual analysis capabilities. These applications are particularly valuable in domains like e-commerce, where product search requires understanding both descriptions and images, and in scientific research involving visual data analysis.

<span id="section-6-3-3"></span>

#### 6.3.3 Code Agents

Beyond typical search-related applications, RL-powered search agents are being integrated into programming and software development workflows. Tool-Star [Don25e] demonstrates multi-tool reasoning capabilities that include code execution and debugging, using RL to coordinate between search engines, code interpreters, and other development tools. VerlTool [Jia25a] provides a unified framework for agentic RL with tool use that specifically supports code interpreters alongside other APIs, enabling agents to search for code solutions, execute them, and iteratively refine implementations. These systems learn to balance web search for coding solutions with direct code experimentation, optimizing both information gathering and implementation efficiency.

<span id="section-6-3-4"></span>

#### 6.3.4 AI Assistants

Conversational AI is a growing deployment area for RL-based search agents, which is far beyond a naive chatbot but like a personal assistant with the capability to handle various realistic tasks. For instance, ConvSearch-R1 [Zhu25i] specifically addresses conversational search scenarios, using RL to enhance query reformulation and maintain context across multi-turn interactions. Lucy [Dao25a] demonstrates edge-running agentic web search on mobile devices with machine-generated task vectors, showcasing practical deployment in resource-constrained environments. MAO-ARAG [Che25t] provides adaptive retrieval-augmented generation through multi-agent orchestration, suitable for intelligent assistant applications that need to balance response quality with computational efficiency. These systems use RL to learn to understand user intent, search for relevant information, and provide contextually appropriate responses while maintaining conversation flow.

<span id="section-6-3-5"></span>

#### 6.3.5 Domain-specific Applications

In addition to the aforementioned general applications, RL-based search agents are also applied in specialized domains tailored to specific knowledge areas and user needs. For instance, HierSearch [Tan25d] presents enterprise search frameworks that integrate local knowledge bases with web search, addressing corporate information management needs. KunLunBaizeRAG [Li25ab] focuses on inference performance optimization for large language models in domain-specific RAG scenarios. DynaSearcher [Hao25b] demonstrates dynamic knowledge graph (KG) augmented search for structured information retrieval, particularly valuable in domains with rich relational data. GRAIL [Cha25c] enables interactive KG exploration for retrieval-augmented reasoning through RL.

<span id="section-6-3-6"></span>

#### 6.3.6 Takeaways

The diversity of applications demonstrates the broad applicability and practical value of RL-based agentic search systems. From code development [Don25e] to scientific research [Zhe25f], multi-modal understanding [Wu25s], conversational AI [Zhu25i], and specialized domains [Tan25d], these systems address real-world information-seeking challenges across multiple sectors. The success of these applications highlights the importance of domain-specific adaptation, multi-modal capabilities, and efficient resource management in practical deployments. Future applications will likely see increased integration across modalities and domains, with RL enabling agents to adapt their search strategies dynamically based on task requirements and user contexts.

<span id="section-7"></span>

## 7 Challenges and Future Directions

Despite the remarkable strides of RL-based agentic search, many fundamental challenges and opportunities lie ahead. In this section, we discuss key future directions that will shape the evolution of intelligent search agents, addressing both technical limitations and emerging requirements for real-world deployment.

**Multi-modal Agentic Search**. Real-world information exists across multiple modalities, including text, images, videos, audio, and structured data. Current RL-based search agents primarily focus on textual information, limiting their applicability to complex, multi-modal information-seeking tasks that require understanding and reasoning across diverse content types. While initial efforts [Wu25s, Wan25ai, Gen25a] enable search engines to facilitate reasoning in vision-language models [Bor24, Gao25f, Wu25t], several fundamental limitations persist: (i) how to ensure consistency between textual descriptions and visual content during search-integrated reasoning; (ii) how to determine which modality contributes most to successful outcomes in multi-modal search tasks; and (iii) how to design reward functions that jointly capture relevance, coherence, and cross-modal alignment. Addressing these challenges is essential for moving toward robust multi-modal agentic search, where agents can adaptively select, integrate, and reason over heterogeneous sources to solve open-ended real-world queries.

**Memory-augmented and Long-horizon Search**. Real-world information-seeking often spans multiple sessions, where agents must remember past queries, retrieved evidence, or user feedback. Current RL-based search agents [Jin25b, Zha25at] typically operate within limited context windows and lack sophisticated memory mechanisms for long-term information retention and retrieval. While some initial efforts [Ngu25a, Wu25l] consider simple memory management techniques such as summarization and cleanup operations, they still struggle with more complex tasks requiring long-term interactions and cross-session continuity. To advance agentic search in long-horizon scenarios, future research should explore developing sophisticated memory architectures that can selectively store, organize, and retrieve search-related knowledge over time. Promising directions include: (i) *hierarchical memory systems* that differentiate between short-term working memory, episodic memory across sessions, and long-term semantic knowledge; (ii) *selective memory* mechanisms that use RL signals to decide what retrieved information to retain, compress, or discard based on long-term utility; and (iii) *temporal reasoning integration* that allows agents to model information decay, relevance shifts, and evolving user intents;

**Trustworthy Agentic Search**. Search agents operating in open environments face pressing security, ethical, and reliability challenges that directly affect user trust. These agents may encounter adversarial content, misinformation, or malicious actors attempting to manipulate their behavior for harmful purposes. Existing studies have revealed significant vulnerabilities in search-augmented systems. For instance, PoisonedRAG [Zou25a] demonstrates that RAG can be misled by injected malicious knowledge, resulting in incorrect or unsafe outputs. While Search Wisely [Wu25q] explores uncertainty-aware search to mitigate overconfidence, it remains unclear how search agents perform under adversarial conditions and how to guarantee robustness in real-world deployments. Moreover, these agents frequently interact with sensitive information, raising concerns about privacy protection, ethical information use, and compliance with data governance regulations. Future research should investigate how to develop reliable, privacy-preserving and ethically aligned search agents. Promising directions include: (i) *adversarially robust RL training*, where agents are exposed to poisoned or noisy retrieval environments to learn resilient policies; (ii) *privacy-preserving agentic search*, such as federated or encrypted search agents, to safeguard sensitive user information; (iv) *value-aligned reward design*, ensuring that optimization objectives incorporate fairness, transparency, and safety constraints; and (v) *auditing and verification tools* that allow both developers and end users to interpret, monitor, and evaluate agent behavior. In conclusion, these approaches would move RL-based agentic search toward systems that are not only effective but also secure, ethical, and trustworthy for real-world applications.

**Cross-domain Generalization**. Current RL-based search agents are often trained for specific domains or tasks, limiting their generalizability. Real-world deployment requires agents that can adapt their search strategies across diverse domains and contexts. To solve this challenge and expand agentic search to broader applications, future works can focus on learning generalizable search principles that can be applied across diverse contexts. For example, one potential solution is to develop meta learning approach to to create universal search strategies that can transfer across different information spaces, or to build agents that can automatically identify and adapt to domain-specific search requirements.

**Human–AI Co-search**. Traditional IR systems were designed for humans as the primary end users [Mar06, Whi09]. The integration of retrieval into large-scale AI systems has reshaped this paradigm, particularly with the rise of LLMs. Retrieval is no longer performed solely for human consumption but increasingly serves to enhance models’ reasoning and generation capabilities [Xu25f]. This shift raises fundamental questions about *how humans and AI agents will collaboratively engage in exploratory search*. RL–based agentic search systems provide a natural foundation for this shift. Through interaction and feedback, RL enables agents to learn adaptive retrieval policies that align with evolving user intents and contextual cues, fostering *human–AI co-search* where agents act as copilots that assist users in locating, interpreting, and synthesizing information. Future research may explore: (i) *Adaptive interaction modeling*, where RL agents learn user preferences and search behaviors to personalize strategies and result presentation; (ii) *Explainable search reasoning*, allowing agents to justify retrieval choices and promote transparency; (iii) *Collaborative query refinement*, enabling iterative reformulation of search goals through natural-language interaction.

<span id="section-8"></span>

## 8 Conclusion

The integration of RL into agentic search marks a fundamental shift in how LLMs interact with external knowledge. Unlike naive RAG, RL enables agents to dynamically decide *when*, *what*, and *how* to search, transforming search into an adaptive and interactive process. This survey provides the first systematic overview of RL-based agentic search, synthesizing research across three perspectives: (i) *What RL is for*; (ii) *How RL is used*; and (iii) *Where RL is applied*. We further examine evaluation metrics, system benchmarks, and representative applications, offering a comparative view of current progress. Looking ahead, RL-based agentic search holds the potential to redefine information retrieval and reasoning. We hope this survey provides a foundation for advancing research in this emerging field and inspires new directions toward practical, robust, and intelligent agentic search systems.

<span id="table-07"></span>

![Table 7. Overview of RL-based agentic search from the perspective of reinforcement learning optimization strategies. ORM and PRM denote the *Outcome Reward Model* and the *Process Reward Model*, respectively. “Rule-based” indicates that the reward function is entirely computed from predefined rules; otherwise, an LLM is involved as a reward judge.](../../papers/rl-based-agentic-search/table-07.png)

**Table 7.** Overview of RL-based agentic search from the perspective of reinforcement learning optimization strategies. ORM and PRM denote the *Outcome Reward Model* and the *Process Reward Model*, respectively. “Rule-based” indicates that the reward function is entirely computed from predefined rules; otherwise, an LLM is involved as a reward judge.
