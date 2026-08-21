---
title: 'Search-R1'
createTime: 2026/08/21 11:30:00
permalink: /en/papers/search-r1/
pageClass: paper-reading
---

> [Bowen Jin](https://scholar.google.com/citations?user=dMwdOPkAAAAJ), [Hansi Zeng](https://scholar.google.com/citations?user=a7O1D6oAAAAJ), [Zhenrui Yue](https://scholar.google.com/citations?user=9Iy_KmsAAAAJ), [Jinsung Yoon](https://scholar.google.com/citations?user=kiFd6A8AAAAJ), [Sercan Ö. Arık](https://scholar.google.com/citations?user=-EZBCBAAAAAJ), [Dong Wang](https://scholar.google.com/citations?user=-NfMhb0AAAAJ), [Hamed Zamani](https://scholar.google.com/citations?user=d2uzDIAAAAAJ), and [Jiawei Han](https://hanj.cs.illinois.edu/). First submitted to arXiv on March 12, 2025; current version v5. Published as a conference paper at [COLM 2025](https://openreview.net/forum?id=Rwhi91ideu). [Search-R1: Training LLMs to Reason and Leverage Search Engines with Reinforcement Learning](https://arxiv.org/abs/2503.09516). [Original PDF](/paper/search-r1.pdf). [DOI](https://doi.org/10.48550/arXiv.2503.09516). [TeX source](https://export.arxiv.org/e-print/2503.09516v5). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Efficiently acquiring external knowledge and up-to-date information is essential for effective reasoning and text generation in large language models (LLMs). Prompting advanced LLMs with reasoning capabilities to use search engines during inference is often suboptimal, as the LLM might not fully possess the capability on how to interact optimally with the search engine. This paper introduces Search-R1, an extension of reinforcement learning (RL) for reasoning frameworks where the LLM learns to autonomously generate (multiple) search queries during step-by-step reasoning with real-time retrieval. Search-R1 optimizes LLM reasoning trajectories with multi-turn search interactions, leveraging retrieved token masking for stable RL training and a simple outcome-based reward function. Experiments on seven question-answering datasets show that Search-R1 improves performance by 24% (Qwen2.5-7B) and 20% (Qwen2.5-3B) over various RAG baselines under the same setting. This paper further provides empirical insights into RL optimization methods, LLM choices, and response length dynamics in retrieval-augmented reasoning. The code and model checkpoints are available at [https://github.com/PeterGriffinJin/Search-R1](https://github.com/PeterGriffinJin/Search-R1).

<span id="section-1"></span>

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in natural language understanding and generation [Hen20, Cla18]. Despite these achievements, LLMs often encounter challenges when tasked with complex reasoning [Wei22a] and retrieving up-to-date information from external sources [Jin25e]. Addressing these limitations necessitates integrating advanced reasoning abilities [Hua22b] and the capability to interact effectively with search engines to best utilize external up-to-date information [Sch23].

Existing approaches for integrating LLMs with search engines typically fall into two categories: (1) retrieval-augmented generation (RAG) [Gao24e, Lew20] and (2) treating the search engine as a tool [Yao23b, Sch23]. RAG models often retrieve passages based on the LLM input as query and incorporate them into the LLM’s context for generation [Lew20]. This allows the LLM to leverage external knowledge when answering questions. Although existing work [Tri22a] prompts LLM for multi-turn, multi-query retrieval, this approach is suboptimal because the LLM is not optimized to learn how to interact effectively with search engines during training. Alternatively, LLMs can be prompted or trained to utilize tools, including search engines, as part of their reasoning process [Qu25, Tri22a]. However, prompting-based approaches often struggle to generalize, as certain tasks may not have been encountered during LLM pretraining. On the other hand, training-based approaches offer greater adaptability but are difficult to scale effectively due to their reliance on large-scale, high-quality annotated trajectories and the inherent non-differentiability of the search operation, which renders end-to-end gradient descent-based optimization inapplicable [Sch23, Asa24a].

Reinforcement Learning (RL) [Sut99, Kae96] has emerged as a potent paradigm for enhancing the reasoning capabilities of LLMs [Dee25c, Hou25a, Xie25c, Fau24]. Notably, models like OpenAI-o1 [Ope24h] and DeepSeek-R1 [Dee25c] have leveraged RL techniques (*e.g.*, PPO [Sch17a] and GRPO [Sha24d]) to improve logical inference and problem-solving skills by learning from experience and feedback. With RL, even when trained solely on the outcome rewards, the models learn complex reasoning capabilities, including self-verification [Wen22a] and self-correction [Fau24]. However, applying RL to search-and-reasoning scenarios presents three key challenges: (1) **RL Framework and Stability** – It remains unclear how to effectively integrate the search engine into the RL approaches for LLMs while ensuring stable optimization, particularly when incorporating retrieved context. (2) **Multi-Turn Interleaved Reasoning and Search** – Ideally, the LLM should be capable of iterative reasoning and search engine calls, dynamically adjusting the retrieval strategy based on the complexity of the problem. (3) **Reward Design** – Designing an effective reward function for search and reasoning tasks remains a fundamental challenge, as it is unclear whether simple outcome-based rewards are sufficient to guide the LLM to learn meaningful and consistent search behaviors.

To address aforementioned challenges, we introduce Search-R1, a novel RL framework that enables LLMs to interact with search engines in an interleaved manner with their own reasoning. Specifically, Search-R1 introduces the following key innovations: (1) We model the search engine as part of the environment, enabling sampled trajectory sequences that interleave LLM token generation with search engine retrievals. Search-R1 is compatible with various RL algorithms, including PPO and GRPO, and we apply retrieved token masking to ensure stable optimization. (2) Search-R1 supports multi-turn retrieval and reasoning, invoking search calls when explicitly triggered by `<search>` and `</search>` tokens. Retrieved content is enclosed within `<information>` and `</information>` tokens, while LLM reasoning steps are wrapped within `<think>` and `</think>` tokens. The final answer is formatted using `<answer>` and `</answer>` tokens, allowing for structured, iterative decision-making. (3) We adopt a straightforward outcome-based reward function, avoiding the complexity of process-based rewards. Our results demonstrate that this minimal reward design is effective in search-and-reasoning scenarios. As such, Search-R1 can be viewed as an extension of DeepSeek-R1 Zero [Dee25c], which primarily focuses on parametric reasoning by introducing search-augmented RL training for enhanced retrieval-driven decision-making.

In summary, our key contributions are threefold:

- Our work analyzes the challenges and provides perspectives on implementing RL to improve how LLMs reason using search engine results.
- We propose Search-R1, a novel RL framework that supports LLM rollouts and direct optimization with a search engine, including retrieved token masking to stabilize RL training, multi-turn interleaved reasoning and search to support complex task-solving and an effective outcome reward function.
- We conduct systematic experiments to demonstrate the effectiveness of Search-R1, with two LLMs achieving respective average relative **improvements of 41% and 20% over RAG baselines** under the same experimental setup (*e.g.*, same retrieval model, training data, and pre-trained LLMs). In addition, we provide insights on RL for reasoning and search settings, including RL method selection, different LLM choices, and response length study.

<span id="section-2"></span>

## 2 Related Works

<span id="section-2-1"></span>

### 2.1 Large Language Models and Retrieval

Despite demonstrating remarkable reasoning [Dee25c] and coding [Guo24c] capabilities, LLMs [Zha23d, Tea24a, Ope23] often lack domain-specific knowledge [Pen23d, Li23v] and are prone to hallucinations [Zha23j]. To mitigate these limitations, search engines [Zha24u] are widely integrated to supply external information. There are two primary ways to integrate search engines with LLMs: (1) retrieval-augmented generation (RAG) [Gao24e] and (2) treating the search engines as tools [Sch23]. RAG [Lew20, Yue24d, Xio25a] typically follows a round of retrieval and sequential generation pipelines, where a search engine fetches relevant information based on the input query, which is then concatenated with the query and fed into the LLM. However, this could face challenges of retrieving irrelevant information [Jin25e] and failing to provide sufficiently useful context [Jia23c]. An alternative approach is search-as-a-tool, where LLMs are prompted or fine-tuned to interact with search engines. IRCoT [Tri22a] and ReAct [Yao23b] use prompting to guide iterative reasoning and search engine calls, while Toolformer [Sch23] leverages supervised fine-tuning to enhance search capabilities. However, such methods rely on high-quality labeled trajectories, which are difficult to obtain at scale. Recent work [Dee25c] suggests that RL can enable LLMs to develop advanced reasoning skills using only outcome rewards, yet its potential in search engine calling scenarios remains under-explored.

<span id="section-2-2"></span>

### 2.2 Large Language Models and Reinforcement Learning

Reinforcement learning (RL) [Kae96] is a learning paradigm where an agent learns to make sequential decisions by interacting with an environment and receiving feedback in the form of rewards, aiming to maximize cumulative reward over time [Sut99]. RL was introduced to LLM tuning by [Ouy22a] through RL from human feedback (RLHF) [Kau23]. This approach first trains a reward model using human preference data [Lam24a], which then guides RL-based tuning of the policy LLM, typically via Proximal Policy Optimization (PPO). However, PPO involves multiple rounds of LLM optimization, making it challenging to implement. To simplify RL-based tuning, direct optimization methods such as Direct Preference Optimization (DPO) [Raf23] and SimPO [Men24] have been proposed. A similar approach is employed in LeRet [Hsu24], where LLMs are trained to explore diverse queries to enhance the effectiveness of information retrieval. While these methods offer computational efficiency, they suffer from off-policy issues [Pan24d] and do not consistently match the performance of pure RL approaches. Alternative solutions include Group Relative Policy Optimization (GRPO) [Sha24d], which eliminates the need for a critic model by estimating baselines from group scores, and RLOO [Ahm24], which introduces a simplified REINFORCE-style [Wil92] optimization framework. Despite these advances, the application of RL to LLM-driven search engine interactions and reasoning remains largely unexplored.

<span id="section-3"></span>

## 3 Search-R1

<span id="figure-01"></span>

![Figure 1. Demonstration of PPO and GRPO training with the search engine.](../../papers/search-r1/figure-01.png)

**Figure 1.** Demonstration of PPO and GRPO training with the search engine (Search-R1). During the rollout, LLMs can conduct multi-turn interactions with the search engine.

In the following sections, we present the detailed design for training methods of Search-R1, covering (1) extending RL to utilize search engines; (2) text generation with an interleaved multi-turn search engine call; (3) the training template; and (4) reward model design.

<span id="section-3-1"></span>

### 3.1 Reinforcement Learning with a Search Engine

We formulate the RL objective function utilizing a search engine $\mathcal{R}$ as follows:

<span id="equation-01"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x; \mathcal{R}) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x; \mathcal{R}) \right],
$$

where $\pi_{\theta}$ is the policy LLM, $\pi_{\mathrm{ref}}$ is the reference LLM, $r_{\phi}$ is the reward function and $\mathbb{D}_{\mathrm{KL}}$ is KL-divergence measure. $x$ denote input samples drawn from the dataset $\mathcal{D}$, and $y$ represent the generated outputs interleaved with search engine calling results, sampled from the reference policy $\pi_{\mathrm{ref}}(y \mid x)$ and retrieved from the search engine $\mathcal{R}$. Unlike prior RL approaches that primarily rely on the policy LLM $\pi_{\theta}(\cdot \mid x)$ to generate rollout sequences [Raf23, Ouy22a], our framework explicitly incorporates retrieval interleaved reasoning via $\pi_{\theta}(\cdot \mid x; \mathcal{R})$, which can be seen as $\pi_{\theta}(\cdot \mid x) \bigotimes \mathcal{R}$, where $\bigotimes$ denotes interleaved retrieval-and-reasoning. This enables more effective decision-making in reasoning-intensive tasks that require external information retrieval. An illustration of the rollout process and an explanation of [Equation 1](#equation-01) are provided in [Section 3.2](#section-3-2) and [Appendix A](#appendix-a).

Our approach builds upon two well-established policy gradient RL methods: Proximal Policy Optimization (PPO) [Sch17a] and Group Relative Policy Optimization (GRPO) [Sha24d, Dee25c], leveraging their respective advantages to optimize retrieval-augmented reasoning.

**Loss Masking for Retrieved Tokens.** In both PPO and GRPO, the token-level losses are computed over the entire rollout sequence. In Search-R1, the rollout sequence consists of both LLM-generated tokens and retrieved tokens from external passages. While optimizing LLM-generated tokens enhances the model’s ability to interact with the search engine and perform reasoning, applying the same optimization to retrieved tokens can lead to unintended learning dynamics. To address this, we introduce loss masking for retrieved tokens, ensuring the policy gradient objective is computed only over LLM-generated tokens, excluding retrieved content from the optimization process. This approach stabilizes training while preserving the flexibility of search-augmented generation.

**PPO with Search Engine.** Proximal Policy Optimization (PPO) [Sch17a] is a popular actor-critic RL approach commonly used for LLMs [Ouy22a]. For our reasoning scenarios that involve search engine calling, it optimizes LLMs by maximizing the following objective:

<span id="equation-02"></span>

$$
\mathcal{J}_{\mathrm{PPO}}(\theta) = \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\mathrm{old}}( \cdot\mid x; \mathcal{R})}
\left[ \frac{1}{\sum_{t=1}^{|y|} I(y_t)} \sum_{t=1: I(y_t)=1}^{|y|}
\min \left( \frac{\pi_{\theta}(y_t \mid x, y_{<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_t \mid x, y_{<t}; \mathcal{R})} A_t,
\mathrm{clip} \left( \frac{\pi_{\theta}(y_t \mid x, y_{<t}; \mathcal{R})}{\pi_{\mathrm{old}}(y_t \mid x, y_{<t}; \mathcal{R})}, 1 - \epsilon, 1 + \epsilon \right) A_t
\right) \right],
$$

where $\pi_{\theta}$ and $\pi_{\mathrm{old}}$ represent the current and previous policy models, respectively. $I(y_t)$ is the token loss masking operation such that $I(y_t)=1$ if $y_t$ is a LLM generated token and $I(y_t)=0$ if $y_t$ is a retrieved token. The term $\epsilon$ is a clipping-related hyperparameter introduced in PPO to stabilize training. The advantage estimate $A_t$ is computed using Generalized Advantage Estimation (GAE) [Sch15], based on future rewards $\{ r_{\geq t} \}$ and a learned value function $V_{\phi}$.

**GRPO with Search Engine.** To improve policy optimization stability and avoid the need for an additional value function approximation, Group Relative Policy Optimization (GRPO) is introduced in [Sha24d]. GRPO differs from PPO by leveraging the average reward of multiple sampled outputs as a baseline rather than relying on a learned value function. Specifically, for each input question $x$, GRPO samples a group of responses $\{ y_1, y_2, \dots, y_G \}$ from the reference policy $\pi_{\mathrm{ref}}$. The policy model is then optimized by maximizing the following objective function:

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

where $\epsilon$ and $\beta$ are hyperparameters, and $\hat{A}_{i,t}$ represent the advantage, computed based on the relative rewards of outputs within each group. This approach avoids introducing additional complexity in the computation of $\hat{A}_{i,t}$. Additionally, instead of incorporating KL divergence as a penalty within the reward function, GRPO regularizes by directly adding the KL divergence between the trained policy and the reference policy to the loss function. The retrieved token masking is also applied when calculating the KL divergence loss $\mathbb{D}_{\mathrm{KL}}$.

<span id="section-3-2"></span>

### 3.2 Generation with Multi-turn Search Engine Calling

In this section, we describe the rollout process for LLM response generation with interleaved multi-turn search engine calls, formulated as: $y\sim \pi_{\theta}(\cdot \mid x; \mathcal{R}) = \pi_{\theta}(\cdot \mid x) \bigotimes \mathcal{R}$.

Our approach follows an iterative framework where the LLM alternates between text generation and external search engine queries. Specifically, the system instruction guides the LLM to encapsulate its search query between two designated search call tokens, `<search>` and `</search>`, whenever an external retrieval is needed. Upon detecting these tokens in the generated sequence, the system extracts the search query, queries the search engine, and retrieves relevant results. The retrieved information is then enclosed within special retrieval tokens, `<information>` and `</information>`, and appended to the ongoing rollout sequence, serving as additional context for the next generation step. This process continues iteratively until one of the following conditions is met: (1) the maximum number of action is reached, or (2) the model generates a final response, which is enclosed between designated answer tokens, `<answer>` and `</answer>`. The complete workflow is outlined in Algorithm 1.

**Algorithm 1: LLM Response Rollout with Multi-Turn Search Engine Calls**

- **Require:** Input query $x$, policy model $\pi_{\theta}$, search engine $\mathcal{R}$, maximum action budget $B$.
- **Ensure:** Final response $y$.
- Initialize rollout sequence $y \gets \emptyset$.
- Initialize action count $b \gets 0$.
- **While** $b < B$:
  - Initialize current action LLM rollout sequence $y_b \gets \emptyset$.
  - **While** True:
    - Generate response token $y_t \sim \pi_{\theta}(\cdot \mid x, y + y_b)$.
    - Append $y_t$ to rollout sequence $y_b \gets y_b + y_t$.
    - **If** $y_t$ in [`</search>`, `</answer>`, `<eos>`]:
      - break.
  - $y \gets y + y_b$.
  - **If** `<search>` detected in $y_b$:
    - Extract search query $q \gets \mathrm{Parse}(y_b, \texttt{<search>}, \texttt{</search>})$.
    - Retrieve search results $d = \mathcal{R}(q)$.
    - Insert $d$ into rollout $y \gets y + \texttt{<information>}d\texttt{</information>}$.
  - **Else if** `<answer>` detected in $y_b$:
    - **Return** final generated response $y$.
  - **Else:**
    - Ask for rethink $y \gets y +$ “My action is not correct. Let me rethink.”
  - Increment action count $b \gets b + 1$.
- **Return** final generated response $y$.

<span id="section-3-3"></span>

### 3.3 Training Template

To train Search-R1, we start by crafting a simple template that directs the initial LLM to follow our predefined instructions. As shown in [Table 1](#table-01), this template structures the model’s output into three parts in an iterative fashion: first, a reasoning process, then a search engine calling function, and finally, the answer. We deliberately limit our constraints to this structural format, avoiding any content-specific biases, such as enforcing reflective reasoning and search engine calling or endorsing specific problem-solving approaches. This ensures that the model’s natural learning dynamics during the RL process remain observable and unbiased.

<span id="table-01"></span>

![Table 1. Template for Search-R1.](../../papers/search-r1/table-01.png)

**Table 1.** Template for Search-R1. `question` will be replaced with the specific question during training and inference.

<span id="section-3-4"></span>

### 3.4 Reward Modeling

The reward function serves as the primary training signal, guiding the optimization process in RL. To train Search-R1, we adopt a rule-based reward system that consists solely of **final outcome rewards**, which assess the correctness of the model’s response. For instance, in factual reasoning tasks, correctness can be evaluated using rule-based criteria such as exact string matching:

<span id="equation-04"></span>

$$
r_{\phi}(x, y) = \mathrm{EM}(a_{\mathrm{pred}}, a_{\mathrm{gold}}),
$$

where $a_{\mathrm{pred}}$ is the extracted final answer from response $y$ and $a_{\mathrm{gold}}$ is the ground truth answer. Unlike [Dee25c], we do not incorporate format rewards, as our learned model already demonstrates strong structural adherence. We leave the exploration of more complex format rewards for future work. Furthermore, we avoid training neural reward models, following [Dee25c]. This decision is motivated by the sensitivity of LLMs to specific forms of rewards in large-scale RL, as well as the additional computational cost and complexity introduced by retraining these models.

<span id="section-4"></span>

## 4 Main Results

<span id="section-4-1"></span>

### 4.1 Datasets

We evaluate Search-R1 on seven benchmark datasets, categorized as follows: (1) **General Question Answering**: NQ [Kwi19a], TriviaQA [Jos17], and PopQA [Mal22]. (2) **Multi-Hop Question Answering**: HotpotQA [Yan18a], 2WikiMultiHopQA [Ho20], Musique [Tri22], and Bamboogle [Lew23]. These datasets encompass a diverse range of search with reasoning challenges, enabling a comprehensive evaluation of Search-R1.

<span id="section-4-2"></span>

### 4.2 Baselines

To evaluate the effectiveness of Search-R1, we compare it against the following baselines: (1) **Inference without Retrieval**: Direct inference and Chain-of-Thought (CoT) reasoning [Wei22a]. (2) **Inference with Retrieval**: Retrieval-Augmented Generation (RAG) [Lew20], IRCoT [Tri22a], and Search-o1 [Li25k]. (3) **Fine-Tuning-Based Methods**: Supervised fine-tuning (SFT) [Chu22], RL-based fine-tuning without a search engine (R1) [Dee25c] and rejection sampling [Ahn24] with a search engine. For R1, we train the LLMs with the RL methods proposed in [Dee25c] with our data to have a fair comparison with Search-R1. It only contains reasoning and answer steps without a search engine. For rejection sampling, we generate five candidate responses per training prompt from the same dataset with the instructed LLMs and select those that lead to correct final answers. These selected trajectories are then used to construct a new training set that retains the same multi-turn LLM–search engine interaction rollout mechanism proposed in Search-R1 to fine-tune the LLMs.

These baselines cover a broad spectrum of retrieval-augmented and fine-tuning approaches, allowing for a comprehensive assessment of Search-R1 in both zero-shot and learned retrieval settings. To make a fair comparison between different methods, we use the same retriever, same number of retrieved documents, same knowledge corpus, same training data and same pre-trained LLMs. Details can be found in [Appendix B](#appendix-b).

<span id="section-4-3"></span>

### 4.3 Experimental Setup

We conduct experiments using two types of models: Qwen-2.5-3B (Base/Instruct) and Qwen-2.5-7B (Base/Instruct) [Yang24]. For retrieval, we use the 2018 Wikipedia dump [Kar20] as the knowledge source and E5 [Wan22i] as the retriever. To ensure fair comparison, we follow [Lin23c] and set the number of retrieved passages to 3 across all retrieval-based methods. A study of the number of retrieved passages can be found in [Appendix G](#appendix-g).

For training, we merge the training sets of NQ and HotpotQA to form a unified dataset for Search-R1 and other fine-tuning based baselines. Evaluation is conducted on the test or validation sets of seven datasets to assess both in-domain and out-of-domain performance. Exact Match (EM) is used as the evaluation metric, following [Yu24a]. For inference-style baselines, we use instruct models, as base models fail to follow instructions. For RL tuning methods, experiments are conducted on both base and instruct models. More details on experimental settings can be found in [Appendix B](#appendix-b).

Unless stated otherwise, **PPO is used as the default RL method**, and a detailed comparison between PPO and GRPO is provided in [Section 5.1](#section-5-1).

<span id="table-02"></span>

![Table 2. Main results.](../../papers/search-r1/table-02.png)

**Table 2.** Main results. The best performance is set in bold. $^\dagger/^\star$ represents in-domain/out-domain datasets.

<span id="section-4-4"></span>

### 4.4 Performance

The main results comparing Search-R1 with baseline methods across the seven datasets are presented in [Table 2](#table-02). From the results, we make the following key observations: **(1) Search-R1 consistently outperforms strong baseline methods.** We achieve 24% and 20% average relative improvement with Qwen2.5-7B and Qwen2.5-3B, respectively. These gains hold across both in-distribution evaluation (*i.e.*, NQ and HotpotQA) and out-of-distribution evaluation (*i.e.*, TriviaQA, PopQA, 2WikiMultiHopQA, Musique, and Bamboogle). **(2) Search-R1 surpasses RL-based training for LLM reasoning without retrieval (R1).** This aligns with expectations, as incorporating search into LLM reasoning provides access to relevant external knowledge, improving overall performance. **(3) Search-R1 is effective for both base and instruction-tuned models.** This demonstrates that DeepSeek-R1-Zero-style RL with outcome-based rewards [Dee25c] can be successfully applied to reasoning with search, extending beyond its previously established effectiveness in pure reasoning scenarios. **(4) Larger models are better on learning how to do search.** Search-R1 on 7B model shows much larger “performance gap” compared with 3B model (*e.g.*, compared with second best model - RAG).

<span id="section-5"></span>

## 5 Analysis

<span id="section-5-1"></span>

### 5.1 Different RL methods: PPO vs. GRPO

We evaluate Search-R1 using both PPO and GRPO as the base RL method, conducting experiments on Qwen2.5-3B/7B models. The training dynamics comparison is presented in [Figure 2(a)](#figure-02) and the evaluation results are presented in [Table 3](#table-03), revealing the following insights: **(1) GRPO converges faster than PPO across all cases.** This is because PPO relies on a critic model, which requires several warm-up steps before effective training begins. **(2) PPO demonstrates greater training stability.** As shown in [Figure 2(a)](#figure-02), GRPO leads to reward collapse after training for many steps, whereas PPO remains stable. **(3) The final training rewards of PPO and GRPO are comparable.** Despite differences in convergence speed and stability, both methods achieve similar final train reward and performance, indicating that both are viable for optimizing Search-R1. PPO exhibits greater training stability, making it a preferable choice in this setting. More results are in [Appendix F](#appendix-f).

<span id="table-03"></span>

![Table 3. The performance results of Search-R1 with PPO and GRPO on seven datasets.](../../papers/search-r1/table-03.png)

**Table 3.** The performance results of Search-R1 with PPO and GRPO on seven datasets.

<span id="figure-02"></span>

![Figure 2. PPO vs. GRPO, Base vs. Instruct, response length, and valid search.](../../papers/search-r1/figure-02.png)

**Figure 2.** (a) PPO vs. GRPO: GRPO generally converges faster but may exhibit instability after trained for a number of steps, whereas PPO provides more stable optimization but converges at a slower rate. (b) Base vs. Instruct LLM study: Instruction-tuned LLMs converge faster, but the final performance of both modles remains highly similar. (c) Response length study: The response length exhibits a decrease-increase-stabilize trend throughout training, aligning with the overall performance trajectory of the LLM. (d) # Valid search study: As the training proceeds, the LLM learns to call search more.

<span id="section-5-2"></span>

### 5.2 Base vs. Instruct LLMs

We analyze the training dynamics of Search-R1 across both base LLMs and instruction-tuned LLMs. Experiments are conducted on two model variants: Qwen2.5-3B, and Qwen2.5-7B. As shown in [Figure 2(b)](#figure-02), we observe that instruction-tuned models converge faster and start from a higher initial performance compared to base models. However, the final training reward of both model types remains highly similar after training. This finding suggests that while general post-training accelerates learning in reasoning-plus-search scenarios, RL can effectively bridge the gap over time, enabling base models to achieve comparable performance. More results can be found in [Appendix E](#appendix-e).

<span id="table-04"></span>

![Table 4. The performance of Search-R1 with and without retrieved token loss masking.](../../papers/search-r1/table-04.png)

**Table 4.** The performance of Search-R1 with and without retrieved token loss masking. The LLM trained with retrieved token loss masking achieves consistently better performance. (LLM: Qwen2.5-7b-base; RL: PPO)

<span id="section-5-3"></span>

### 5.3 Response Length and Valid Search Study

We conduct an experiment using Search-R1 with the Qwen2.5-7b-base model to analyze the dynamics of response length and number of valid search engine calls over the course of training. The response length result is presented in [Figure 2(c)](#figure-02), revealing the following key trends: **(1) Early Stage (First 100 Steps)**: The response length sharply decreases, while the training reward exhibits a slight increase. During this phase, the base model learns to eliminate excessive filler words and begins adapting to the task requirements. **(2) Later Stage (After 100 Steps)**: Both response length and training reward increase significantly. At this point, the LLM learns to call the search engine frequently, resulting in longer responses due to retrieved passages. The training reward improves substantially, as the model becomes more effective at leveraging search results. The valid search result is presented in [Figure 2(d)](#figure-02), showing that the LLMs learn to call the search engine more times as the training proceeds.

<span id="section-5-4"></span>

### 5.4 Study of Retrieved Tokens Loss Masking

In [Section 3.1](#section-3-1), we introduced loss masking for retrieved tokens to prevent unintended optimization behaviors. Here, we conduct experiments on the Qwen2.5-7b-base model, comparing training dynamics with and without retrieved token loss masking. As shown in [Figure 3](#figure-03), applying retrieved token masking results in greater LLM improvements, mitigating unintended optimization effects and ensuring more stable training. The performance comparison is provided in [Table 4](#table-04), demonstrating that Search-R1 trained with retrieved token loss masking consistently outperforms the variant without masking.

More experimental results on retrieved token loss mask, base vs. instruct LLMs, comparison between PPO/GRPO, the number of retrieved passages in Search-R1 training, group size study in Search-R1 (GRPO), case studies can be found in [Appendix D](#appendix-d), [Appendix E](#appendix-e), [Appendix G](#appendix-g), [Appendix H](#appendix-h), [Appendix I](#appendix-i) and [Appendix J](#appendix-j).

<span id="section-6"></span>

## 6 Conclusions

In this work, we introduced Search-R1, a novel RL framework that enables LLMs to interleave self-reasoning with real-time search engine interactions. Unlike existing RAG-like approaches, which relies on extensive prompting for multi-turn retrieval, or tool-use methods that require large-scale supervised training data, Search-R1 optimizes LLM rollouts through RL, allowing autonomous query generation and strategic utilization of retrieved information. Through extensive experiments on seven datasets, we demonstrated that Search-R1 significantly enhances LLMs' ability to tackle complex reasoning tasks requiring real-time external knowledge. Our analysis also provides key insights into RL training strategies for search-augmented reasoning. Looking ahead, future work can explore expanding Search-R1 to support broader search strategies, including more sophisticated reward mechanisms, dynamic retrieval adjustments based on uncertainty, combining with diverse set of tools and integration with diverse information sources beyond search. It is also promising to investigate its applicability to multimodal reasoning tasks.

## Acknowledgments

This research was supported in part by Apple PhD Fellowship, in part by US DARPA INCAS Program No. HR0011-21-C0165 and BRIES Program No. HR0011-24-3-0325, in part by the Office of Naval Research contract number N000142412612, in part by NSF grant numbers IIS-19-56151 and 2402873, in part by the Molecule Maker Lab Institute: An AI Research Institutes program supported by NSF under Award No. 2019897 and the Institute for Geospatial Understanding through an Integrative Discovery Environment (I-GUIDE) by NSF under Award No. 2118329, in part by Cisco, and in part by the Center for Intelligent Information Retrieval. Any opinions, findings, and conclusions or recommendations expressed herein are those of the authors and do not necessarily represent the views, either expressed or implied, of the sponsors or the U.S. Government.

<span id="appendix-a"></span>

## A Formulation of Reinforcement Learning with a Search Engine

The classical reinforcement learning (RL) framework for training large language models (LLMs) can be formulated as follows [Raf23, Ouy22a]:

<span id="equation-05"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x)}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x) \right],
$$

where $x$ denotes a prompt sampled from a dataset $\mathcal{D}$, $y$ is a response generated by the policy model $\pi_\theta$, and $\pi_{\mathrm{ref}}$ represents a reference model that serves as a regularization anchor. The reward function $r_{\phi}(x, y)$ quantifies the quality of the generated response, while the KL divergence term constrains the updated policy to remain close to the reference model, thereby promoting training stability.

However, this formulation assumes that the entire output sequence $y$ is generated solely by the policy LLM. This assumption does not hold in our setting, where model behavior incorporates both internal reasoning and external information retrieval. To accommodate this, we extend the RL objective to incorporate an external search engine $\mathcal{R}$, yielding the following formulation:

<span id="equation-06"></span>

$$
\max_{\pi_\theta} \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})}
\left[ r_{\phi}(x, y) \right]
- \beta \mathbb{D}_{\mathrm{KL}} \left[ \pi_{\theta}(y \mid x; \mathcal{R}) \,\|\|\, \pi_{\mathrm{ref}}(y \mid x; \mathcal{R}) \right],
$$

In this revised objective, the trajectory $y \sim \pi_{\theta}(\cdot \mid x; \mathcal{R})$ includes interleaved reasoning steps and retrieved content, reflecting a multi-turn interaction between the LLM and the search engine. The KL divergence is computed over the joint response distribution conditioned on both the prompt and the retrieval-augmented context, ensuring the learned policy remains aligned with the reference model even in the presence of external information.

<span id="appendix-b"></span>

## B Experimental Setups

<span id="appendix-b-1"></span>

### B.1 Baselines

Several recent works have explored RAG pipelines, particularly in benchmarks such as Natural Questions (NQ) or HotpotQA, aiming to improve performance through more elaborate retrieval mechanisms. For instance, Re2G [Gla22] and RetroLLM [Li24r] propose sophisticated retrieve-rerank-generate frameworks that employ strong retrievers and complex reranking strategies to select fine-grained evidence for generation. While these approaches demonstrate impressive results, they often rely on task-specific engineering or heavyweight pipelines that limit generalizability and scalability. In contrast, our focus is on a more lightweight and general approach to retrieval-augmented reasoning. As such, we do not include these methods as direct baselines, though they represent valuable directions in the broader space of retrieval-enhanced language modeling.

<span id="appendix-b-2"></span>

### B.2 Experimental Settings

We conduct experiments using two types of models: Qwen-2.5-3B (Base/Instruct) and Qwen-2.5-7B (Base/Instruct) [Yang24]. For retrieval, we use the 2018 Wikipedia dump [Kar20] as the knowledge source and E5 [Wan22i] as the retriever. To ensure fair comparison, we follow [Lin23c] and set the number of retrieved passages to 3 across all retrieval-based methods.

For training, we merge the training sets of NQ and HotpotQA to form a unified dataset for Search-R1 and other fine-tuning based baselines. Evaluation is conducted on the test or validation sets of seven datasets to assess both in-domain and out-of-domain performance. Exact Match (EM) is used as the evaluation metric, following [Yu24a]. For inference-style baselines, we use instruct models, as base models fail to follow instructions. For RL tuning methods, experiments are conducted on both base and instruct models. More details on experimental settings can be found in [Appendix B](#appendix-b).

For the PPO variant of Search-R1, we set the learning rate of the policy LLM to 1e-6 and that of the value LLM to 1e-5. Training is conducted for 500 steps, with warm-up ratios of 0.285 and 0.015 for the policy and value models, respectively. We use Generalized Advantage Estimation (GAE) with parameters $\lambda = 1$ and $\gamma = 1$.

Training is performed on a single node with 8 H100 GPUs. We use a total batch size of 512, with a mini-batch size of 256 and a micro-batch size of 64. The maximum sequence length is set to 4,096 tokens, with a maximum response length of 500 and a maximum length of 500 tokens for retrieved content. To optimize GPU memory usage, we enable gradient checkpointing and use Fully Sharded Data Parallel (FSDP) with CPU offloading.

For efficient LLM rollouts, we adopt vLLM [+1] with a tensor parallel size of 1 and GPU memory utilization ratio of 0.6. The rollout sampling uses a temperature of 1.0 and a top-p value of 1.0. The KL divergence regularization coefficient $\beta$ and clip ratio $\epsilon$ are set to 0.001 and 0.2.

For GRPO training, we set the policy LLM learning rate to 1e-6 and sample 5 responses per prompt, following the GRPO implementation in Verl [She24a] [+2]. The model is trained for 500 steps with a learning rate warm-up ratio of 0.285. Training is conducted on the same 8×H100 setup with identical batch sizes and sequence length configurations as in PPO.

We also use gradient checkpointing, FSDP offloading, and vLLM-based rollouts with the same hyperparameters as above. The rollout temperature and top-p values are both set to 1.0, and the KL divergence coefficient $\beta$ and clip ratio $\epsilon$ are fixed at 0.001 and 0.2.

For both methods, model checkpoints are saved every 100 steps. In cases where training diverges, we evaluate at the most recent stable checkpoint according to the training reward curve; otherwise, the final checkpoint is used for evaluation. The maximum action budget $B$ is set to 4, and we retrieve the top 3 passages by default.

We compute outcome rewards using exact match (EM). Unless otherwise noted, **PPO is used as the default RL algorithm**, and a detailed comparison with GRPO is provided in [Section 5.1](#section-5-1).

[+1]: [https://docs.vllm.ai/en/latest/](https://docs.vllm.ai/en/latest/)

[+2]: [https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_deepseek7b_llm.sh](https://github.com/volcengine/verl/blob/main/examples/grpo_trainer/run_deepseek7b_llm.sh)

<span id="appendix-c"></span>

## C Main Results on 14B LLM

We conduct extensive experiments using the Qwen2.5-14B models, and the results are presented in [Table 5](#table-05). As shown, Search-R1 consistently outperforms all baseline methods across the evaluated metrics. Furthermore, we observe that increasing the model size leads to consistent performance gains with Search-R1, highlighting the benefits of LLM size scaling in our approach.

<span id="table-05"></span>

![Table 5. Main results on 14B LLM.](../../papers/search-r1/table-05.png)

**Table 5.** Main results. The best performance is set in bold. $^\dagger/^\star$ represents in-domain/out-domain datasets.

<span id="appendix-d"></span>

## D Retrieved Token Loss Masking Study

In [Section 3.1](#section-3-1), we introduced a loss masking strategy for retrieved tokens to mitigate undesirable optimization behaviors during training. To evaluate its impact, we conduct experiments using the Qwen2.5-3b/7b-base model, comparing training dynamics with and without retrieved token loss masking. As illustrated in [Figure 3](#figure-03), incorporating the masking mechanism leads to more stable optimization and improved model performance. Quantitative results in [Table 6](#table-06) further confirm that Search-R1, when trained with loss masking on retrieved tokens, consistently outperforms its unmasked counterpart.

<span id="figure-03"></span>

![Figure 3. Retrieved Token Loss Masking Study.](../../papers/search-r1/figure-03.png)

**Figure 3.** Retrieved Token Loss Masking Study

<span id="table-06"></span>

![Table 6. The performance of Search-R1 with and without retrieved token loss masking.](../../papers/search-r1/table-06.png)

**Table 6.** The performance of Search-R1 with and without retrieved token loss masking. The LLM trained with retrieved token loss masking achieves consistently better performance. (RL: PPO)

<span id="appendix-e"></span>

## E Base vs. Instruct LLMs

We investigate the training dynamics of Search-R1 across both base and instruction-tuned LLMs, using two model scales: Qwen2.5-3B and Qwen2.5-7B. As depicted in [Figure 4](#figure-04), instruction-tuned models exhibit faster convergence and benefit from higher initial performance relative to their base counterparts. Despite this early advantage, the final performance of both model types converges to a similar level after training. These results indicate that while instruction tuning facilitates more efficient early-stage learning in reasoning-plus-search tasks, reinforcement learning is capable of closing the performance gap, ultimately enabling base models to reach comparable outcomes.

<span id="figure-04"></span>

![Figure 4. Study of Search-R1 on base and instruct LLMs.](../../papers/search-r1/figure-04.png)

**Figure 4.** Study of Search-R1 on base and instruct LLMs. The instruction model converges faster and starts from a better initial performance. However, the final performance of both models is very similar.

<span id="appendix-f"></span>

## F Comparison of PPO and GRPO in Search-R1

We assess the effectiveness of Search-R1 under two reinforcement learning algorithms: PPO and GRPO, using both Qwen2.5-3B and Qwen2.5-7B as the underlying models. [Figure 5](#figure-05) illustrates the training dynamics. Our analysis yields the following key observations: **(1) GRPO exhibits faster convergence than PPO across all settings**, attributed to the fact that PPO relies on a separate value function (critic), which requires an initial warm-up phase before effective policy updates can be made. **(2) PPO provides more stable training behavior**, as evidenced in [Figure 5](#figure-05), where GRPO encounters reward collapse over extended training steps, whereas PPO maintains stability throughout. **(3) PPO and GRPO achieve comparable final reward performance**, suggesting that despite trade-offs in convergence speed and stability, both methods are effective for optimizing Search-R1.

<span id="figure-05"></span>

![Figure 5. Training dynamics of Search-R1 with PPO and GRPO.](../../papers/search-r1/figure-05.png)

**Figure 5.** Training dynamics of Search-R1 with PPO and GRPO as the base RL method across four LLMs. GRPO generally converges faster but may exhibit instability after trained for a number of steps, whereas PPO provides more stable optimization but converges at a slower rate. PPO and GRPO achieve comparable final reward performance.

<span id="appendix-g"></span>

## G Number of Retrieved Passages Study in Search-R1 Training

We investigate the impact of the number of retrieved passages (top-k) on the training dynamics of Search-R1. While our main experiments adopt top-k = 3 following [Lin23c], we conduct additional studies with top-k set to 1, 3, and 5 to better understand its influence.

[Figure 6](#figure-06) presents the training reward curves under these settings. We observe that all three configurations exhibit similar overall training trajectories. Notably, top-k = 5 achieves the fastest initial convergence, reaching the highest training reward within the first 200 steps. However, its reward gradually declines and becomes more unstable as training progresses. In contrast, top-k = 1 and 3 demonstrate more consistent improvements throughout training, with top-k = 3 ultimately achieving the highest reward after 500 steps.

Evaluation results at step 500 are summarized in [Table 7](#table-07), where top-k = 3 yields the best overall performance. We hypothesize two contributing factors: (1) top-k = 1 likely suffers from low retrieval recall, limiting the ability to provide relevant contextual information; (2) top-k = 5 introduces lower precision due to the inclusion of noisy or irrelevant passages [Jin25e], which not only degrades inference performance but may also adversely affect RL training—discouraging the model from leveraging retrieved content when it learns that the additional context is often unhelpful or misleading.

<span id="figure-06"></span>

![Figure 6. Training dynamics with a different number of retrieved passages.](../../papers/search-r1/figure-06.png)

**Figure 6.** The training dynamics of Search-R1 with a different number of retrieved passages. (LLM: Qwen2.5-7b-base, RL: PPO)

<span id="table-07"></span>

![Table 7. The number of retrieved passages study in Search-R1 training.](../../papers/search-r1/table-07.png)

**Table 7.** The number of retrieved passages study in Search-R1 training. (LLM: Qwen2.5-7b-base; RL: PPO)

<span id="appendix-h"></span>

## H Group Size Study in Search-R1 (GRPO) Training

In our main experiment, we set the group size for Search-R1 (GRPO) to 5, following the setting in [She24a]. To further investigate the impact of group size on training dynamics, we conduct an ablation study with group sizes of 1, 3, and 5. Notably, when the group size is set to 1, GRPO reduces to the standard REINFORCE algorithm [Wil92].

We train the LLMs for 500 steps, saving model checkpoints every 100 steps. If the model collapses during training, we use the last valid checkpoint for evaluation; otherwise, we evaluate the checkpoint at step 500.

The training dynamics under different group size configurations are illustrated in [Figure 7](#figure-07). We observe that a larger group size generally leads to faster convergence but may also increase the risk of collapse due to the inherent instability of reinforcement learning.

Evaluation results across different settings are summarized in [Table 8](#table-08). While larger group sizes can accelerate convergence and achieve higher training rewards, smaller group sizes (*e.g.*, size = 1) enable more stable training and better generalization. This is reflected in superior performance on unseen tasks, highlighting a trade-off between learning speed and stability in GRPO training.

<span id="figure-07"></span>

![Figure 7. Training dynamics with different group size.](../../papers/search-r1/figure-07.png)

**Figure 7.** The training dynamics of Search-R1 (GRPO) with different group size. (LLM: Qwen2.5-7b-base)

<span id="table-08"></span>

![Table 8. The group size study of Search-R1.](../../papers/search-r1/table-08.png)

**Table 8.** The group size study of Search-R1 (GRPO) on seven datasets. (LLM: Qwen2.5-7b-base)

<span id="appendix-i"></span>

## I Comparison between R1 and Search-R1: A Case Study

<span id="table-09"></span>

![Table 9. A case study of R1 and Search-R1.](../../papers/search-r1/table-09.png)

**Table 9.** A case study of R1 and Search-R1.

To gain deeper insights into Search-R1, we conduct a case study using Qwen2.5-7B-Base, comparing its behavior with RL without a search engine [Dee25c]. The results are presented in [Table 9](#table-09), revealing the following key observations:

**Interleaved Reasoning and Retrieval Enhances Problem Analysis**: Search-R1 enables the LLM to perform in-depth reasoning with multi-turn retrieval, whereas RL without search relies solely on the models' internal knowledge. By incorporating retrieved passages, Search-R1 allows the LLM to iteratively refine its reasoning, leading to more informed and accurate responses.

**Self-Verification through Iterative Retrieval**: We observe that after the second retrieval round, the LLM has already gathered sufficient information to answer the question. However, Search-R1 performs an additional retrieval step to self-verify its conclusion, further reinforcing its confidence in the final response. This phenomenon aligns with findings from LLM reasoning RL without retrieval [Dee25c], highlighting how RL can encourage verification-driven reasoning even in search-augmented settings.

<span id="appendix-j"></span>

## J More Case Studies of Search-R1

To gain a deeper understanding of the behavior and capabilities of the trained LLM, we conduct additional case studies on Search-R1. Specifically, we analyze the model fine-tuned from Qwen2.5-7B-Base using Proximal Policy Optimization (PPO) as the underlying reinforcement learning algorithm. The results are shown in the following tables.

<span id="table-10"></span>

![Table 10. Search-R1 case study 1.](../../papers/search-r1/table-10.png)

**Table 10.** Search-R1 case study 1 (successful): Search-R1 conduct multi-step reasoning, search, with self-verification and finally answer the question.

<span id="table-11"></span>

![Table 11. Search-R1 case study 2.](../../papers/search-r1/table-11.png)

**Table 11.** Search-R1 case study 2 (failed): Search-R1 sometimes fail to decompose the complex problem and can be mislead by irrelevent searched passages.

<span id="table-12"></span>

![Table 12. Search-R1 case study 3.](../../papers/search-r1/table-12.png)

**Table 12.** Search-R1 case study 3 (successful): Search-R1 can easily answer the question if the relevant information can be found with one search engine call.

<span id="table-13"></span>

![Table 13. Search-R1 case study 4.](../../papers/search-r1/table-13.png)

**Table 13.** Search-R1 case study 4 (successful): Search-R1 can write the right query to search for auxiliary information not provided in the previous search engine calls.

<span id="table-14"></span>

![Table 14. Search-R1 case study 5.](../../papers/search-r1/table-14.png)

**Table 14.** Search-R1 case study 5 (failed): Search-R1 fails to answer the question with insufficient or misleading retrieved information.

<span id="table-15"></span>

![Table 15. Search-R1 case study 6.](../../papers/search-r1/table-15.png)

**Table 15.** Search-R1 case study 6 (successful): Search-R1 can easily answer the question with multi-hop reasoning when sufficient and accurate context is retrieved.

<span id="table-16"></span>

![Table 16. Search-R1 case study 7.](../../papers/search-r1/table-16.png)

**Table 16.** Search-R1 case study 7 (failed): Search-R1 failed to write the right queries to decompose a complex problem at the beginning. The model answer the question without obtaining enough evidence.

<span id="table-17"></span>

![Table 17. Search-R1 case study 8.](../../papers/search-r1/table-17.png)

**Table 17.** Search-R1 case study 8 (successful): Search-R1 can write query to search for insufficient information.

<span id="table-18"></span>

![Table 18. Search-R1 case study 9.](../../papers/search-r1/table-18.png)

**Table 18.** Search-R1 case study 9 (successful): The first query written by the LLM is not very meaningful. However, upon that, LLM starts to write the query and solve the problem step by step.

<span id="table-19"></span>

![Table 19. Search-R1 case study 10.](../../papers/search-r1/table-19.png)

**Table 19.** Search-R1 case study 10 (successful): Search-R1 learns to stop searching when it finds out the external knowledge source is not sufficient to answer the question.

<span id="table-20"></span>

![Table 20. Search-R1 case study 11.](../../papers/search-r1/table-20.png)

**Table 20.** Search-R1 case study 11 (failed): The LLM can be misled by irrelevant retrieved information and provide a wrong answer.
