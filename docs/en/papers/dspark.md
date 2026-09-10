---
title: 'DSpark: Confidence-Scheduled Speculative Decoding'
createTime: 2026/09/10 23:14:39
permalink: /en/papers/dspark/
---

> [Xin Cheng](https://dblp.org/pid/96/4269-2.html) [+equal], [Xingkai Yu](https://dblp.org/pid/257/4432.html) [+equal], [Chenze Shao](https://dblp.org/pid/227/3123.html) [+equal], [Jiashi Li](https://dblp.org/pid/241/9364.html) [+equal], [Yunfan Xiong](https://dblp.org/pid/343/8320.html) [+equal], [Yi Qian](https://dblp.org/pid/84/4625.html), [Jiaqi Zhu](https://dblp.org/pid/23/5224.html), [Shirong Ma](https://dblp.org/pid/289/1472.html), [Xiaokang Zhang](https://dblp.org/pid/60/7487.html), [Jiasheng Ye](https://dblp.org/pid/298/0158.html), [Qinyu Chen](https://dblp.org/pid/91/5007.html), [Chengqi Deng](https://dblp.org/pid/255/4939.html), [Jiping Yu](https://dblp.org/pid/226/4099.html), [Damai Dai](https://dblp.org/pid/199/2097.html), [Zhengyan Zhang](https://dblp.org/pid/23/10446.html), [Yixuan Wei](https://dblp.org/pid/238/0096.html), [Yixuan Tan](https://dblp.org/pid/166/3526.html), [Wenkai Yang](https://dblp.org/pid/250/3934.html), [Runxin Xu](https://dblp.org/pid/267/5291.html), [Yu Wu](https://dblp.org/pid/22/0-24.html), [Zhean Xu](https://dblp.org/pid/355/2237.html), [Xuanyu Wang](https://dblp.org/pid/195/6495.html), [Muyang Chen](https://dblp.org/pid/335/2649.html), [Rui Tian](https://dblp.org/pid/15/9722.html), [Xiao Bi](https://dblp.org/pid/237/3479.html), [Zhewen Hao](https://dblp.org/pid/313/3144.html), [Shaoyuan Chen](https://dblp.org/pid/126/1575.html), [Huanqi Cao](https://dblp.org/pid/214/8159.html), [Wentao Zhang](https://dblp.org/pid/41/3249.html), [Anyi Xu](https://dblp.org/pid/319/6065.html), [Huishuai Zhang](https://dblp.org/pid/144/7537.html), [Dongyan Zhao](https://dblp.org/pid/63/1870.html), and [Wenfeng Liang](https://dblp.org/pid/59/9456.html). Affiliations: Peking University and DeepSeek-AI. First submitted to arXiv on July 6, 2026; current version v1. [DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation](https://arxiv.org/abs/2607.05147v1). <a href="/paper/dspark.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2607.05147). [TeX source](https://arxiv.org/src/2607.05147v1). [Code and checkpoints](https://github.com/deepseek-ai/DeepSpec). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Speculative decoding accelerates Large Language Model (LLM) inference by decoupling draft generation from target verification. While recent parallel drafters efficiently propose long token sequences in a single forward pass, they suffer from rapid acceptance decay due to a lack of inter-token dependencies. Furthermore, indiscriminately verifying these extended blocks wastes critical batch capacity on tokens with high rejection risks, severely degrading throughput in high-concurrency serving systems. We introduce DSpark, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. To maintain draft quality, DSpark utilizes a semi-autoregressive architecture—coupling a parallel backbone with a lightweight sequential module—to introduce intra-block dependency modeling and mitigate suffix decay. To optimize system efficiency, DSpark employs confidence-scheduled verification, dynamically tailoring the verification length for each request based on estimated prefix survival probabilities and engine-specific throughput profiles. On offline benchmarks across diverse domains, DSpark substantially improves the accepted length over state-of-the-art autoregressive and parallel drafters. When deployed within the DeepSeek-V4 serving system under live user traffic, DSpark successfully mitigates verification waste. Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60%-85% at matched throughput levels. More importantly, by preventing severe throughput degradation under strict interactivity constraints, it enables performance tiers that were previously unattainable, shifting the Pareto frontier of our serving system. To facilitate community progress, we open-source the [DSpark checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/main) alongside [DeepSpec](https://github.com/deepseek-ai/DeepSpec), an algorithm-driven training repository for speculative decoding.

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) generate text autoregressively: each new token requires a full forward pass conditioned on all preceding tokens, making inference latency proportional to the output length. The resulting low GPU utilization and high user-perceived waiting time constitute a primary bottleneck in production LLM serving, particularly for latency-sensitive scenarios such as real-time conversational assistants and multi-turn agentic workflows. Speculative decoding [Lev23, Che23] offers a principled solution: a lightweight *draft model* proposes a block of candidate tokens, and the full-size *target model* verifies the entire block in a single forward pass via rejection sampling, accepting the longest prefix consistent with the target distribution and appending one bonus token. Because verification is parallel and the acceptance rule preserves the target distribution exactly, speculative decoding accelerates generation without any quality loss.

The design of the draft model governs the trade-off between drafting latency and acceptance rate. Early drafters are autoregressive [Li24v, Che24w], conditioning each position on previously sampled tokens. However, their drafting latency grows linearly with the block size, forcing these methods to use short blocks and shallow architectures. To break this sequential bottleneck, parallel drafters [Cai24d, Liu26b, Che26d] have emerged as a compelling alternative: all draft positions are produced in a single forward pass, making drafting latency nearly independent of block size. This structural advantage theoretically allows parallel drafters to efficiently generate substantially longer draft blocks.

However, fully unlocking the potential of large parallel draft blocks introduces two critical bottlenecks—one in generation quality, and the other in system efficiency. First, because parallel drafters predict each position independently, they cannot model inter-token dependencies within a block. This independence leads to multi-modal collisions and rapid acceptance decay at later positions [Gu18, Hua22d]. Second, determining the optimal verification length remains a challenge. While parallel generation easily produces long draft blocks, indiscriminately verifying all proposed tokens degrades system throughput, particularly under high-concurrency workloads [Liu24ab, Hu26]. The ideal verification length varies along two axes. On the data side, structured requests like code naturally sustain higher acceptance rates than open-ended chat [Xia24j, Abr26]. On the system side, verifying extra tokens is nearly free under light loads. Under heavy loads, however, verifying tokens with a high rejection risk occupies critical batch capacity that could otherwise serve other active requests [Liu24ac, Wu25u].

To address these bottlenecks, we introduce **DSpark**, a speculative decoding framework that unifies high-throughput parallel generation with adaptive, load-aware verification. At its core, DSpark is designed to resolve the inherent trade-offs in draft generation and verification through two complementary mechanisms.

- First, to overcome the lack of inter-token dependencies, DSpark adopts a semi-autoregressive architecture. It keeps the computationally expensive draft backbone fully parallel, appending only a lightweight serial output head to inject local transition information. This design preserves the drafting speed of parallel models while significantly mitigating suffix decay.
- Second, to resolve the system-level bottleneck, DSpark employs confidence-scheduled verification. By coupling a confidence head—which estimates per-position prefix survival probabilities—with a hardware-aware scheduler, DSpark dynamically tailors the verification length for each request. This scheduler leverages real-time engine throughput profiles to route target verification budget only toward tokens with the highest expected return.

We extensively evaluate DSpark across both controlled offline benchmarks and production-scale online deployments. On controlled offline benchmarks—spanning mathematical reasoning, code generation, and daily chat—DSpark consistently outperforms strong baselines. Specifically, across the Qwen3-4B, 8B, and 14B target models [Yan25g], it improves the macro-average accepted length over the autoregressive Eagle3 [Li25] by 30.9%, 26.7%, and 30.0%, and over the parallel DFlash [Che26d] by 16.3%, 18.4%, and 18.3%, respectively. Beyond top-line metrics, our fine-grained position-wise analysis reveals the distinct generation characteristics of different drafters, empirically demonstrating how DSpark successfully combines the high initial-token capacity of parallel models with the suffix coherence of autoregressive models.

Beyond offline evaluation, we deployed DSpark within the DeepSeek-V4 [Int26] serving system to assess its performance under live user traffic. Compared to the prior MTP-1 production baseline [Dee24a], DSpark significantly broadens the system’s operational envelope. Specifically, it consistently accelerates per-user generation speeds by 60%-85% (V4-Flash) and 57%-78% (V4-Pro) at matched aggregate throughput capacities. Furthermore, under strict Service Level Agreements (SLAs) where the baseline’s capacity deteriorates severely—such as 120 TPS for Flash and 50 TPS for Pro—DSpark mitigates verification overhead to maintain robust throughput. By overcoming this performance cliff, DSpark unlocks strict interactivity tiers that were previously unattainable, effectively shifting the Pareto frontier of LLM serving.

To foster collective advancement within the open-source community, we are making our artifacts publicly available. Specifically, we release the trained [DSpark checkpoints](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-DSpark/tree/main) for both the DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview) models. Furthermore, we open-source [DeepSpec](https://github.com/deepseek-ai/DeepSpec), an algorithm-driven training repository, including Eagle3, DFlash and DSpark. These artifacts are intended to support further research on efficient LLM serving.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 Speculative Decoding

Autoregressive language models generate one token per forward pass, making inference latency proportional to output length. Speculative decoding [Ge22, Lev23, Che23] accelerates the inference of a target model $M_{t}$ using a lightweight draft model $M_{d}$. At each decoding cycle, the draft model proposes $\gamma$ candidate tokens $x_{1},\ldots,x_{\gamma}$. The target model verifies all candidates in a single forward pass, accepting the longest prefix consistent with its own distribution.

Concretely, at each draft position $k$, the target model computes its own distribution $p_{k}^{t}$ and compares it against the draft distribution $p_{k}^{d}$. The token $x_{k}$ is accepted with probability $\min(1,\,p_{k}^{t}(x_{k})/p_{k}^{d}(x_{k}))$. Verification proceeds left to right: the first rejection at position $k$ discards all subsequent tokens $x_{k+1},\ldots,x_{\gamma}$, regardless of their quality.

Let $\tau$ denote the number of accepted tokens per cycle, and let $T_{\mathrm{draft}}$ and $T_{\mathrm{verify}}$ be the wall-clock times of the drafting and verification passes, respectively. The average latency per generated token is:

<span id="equation-01"></span>

$$
L=\frac{T_{\mathrm{draft}}+T_{\mathrm{verify}}}{\tau}.
$$

Improving speedup therefore reduces to three levers: lowering $T_{\mathrm{draft}}$ (draft faster), raising $\tau$ (draft better), or reducing the effective $T_{\mathrm{verify}}$ (verify smarter).

<span id="section-2-2"></span>

### 2.2 Drafter Architectures

The design of the draft model determines how $T_{\mathrm{draft}}$ and $\tau$ trade off. Existing approaches fall into two categories.

**Autoregressive drafters.** Autoregressive drafters generate draft tokens sequentially, conditioning each position on previously sampled tokens [Dee24a, Li24g, Li24v, Li25, Zha25az]. This explicit dependency gives strong modeling capacity, but the drafting cost grows linearly with block size: $T_{\mathrm{draft}}\propto\gamma$, which forces autoregressive drafters to use small $\gamma$ and shallow architectures to keep $T_{\mathrm{draft}}$ low. To compensate for the short block, tree-based verification [Mia24a] expands candidates into a tree and verifies multiple paths via tree attention, but the large number of verification tokens reduces overall serving throughput.

**Parallel drafters.** Parallel drafters produce all $\gamma$ draft tokens in a single forward pass, making $T_{\mathrm{draft}}$ nearly independent of the block size [Cai24d, Che26d, Liu26b, Li25ae, San26]. This allows substantially larger blocks (e.g., $\gamma{=}16$) without proportionally increasing latency.

Among them, DFlash [Che26d] is a state-of-the-art parallel drafter, which conditions its draft model on rich context features extracted from the target model (KV injection). During prefill, hidden states from a set of target layers $\{l_{1},\ldots,l_{m}\}$ are concatenated and projected into the draft hidden space:

<span id="equation-02"></span>

$$
H_{\mathrm{ctx}}=\mathrm{RMSNorm}\bigl(W_{c}\,[H^{(l_{1})};\,\ldots;\,H^{(l_{m})}]\bigr),
$$

where $W_{c}\in\mathbb{R}^{d\times md}$ is a shared projection. These context features are injected into every draft layer by concatenating them with the draft block representations along the sequence dimension of keys and values:

<span id="equation-03"></span>

$$
K_{i}=[W_{i}^{K}H_{\mathrm{ctx}};\;W_{i}^{K}H_{d}],\quad V_{i}=[W_{i}^{V}H_{\mathrm{ctx}};\;W_{i}^{V}H_{d}].
$$

All positions within a block attend bidirectionally to each other and to the injected target context.

The draft model shares the target model’s embedding layer and language modeling head (both frozen). It takes as input the embedding of an anchor token [+1] followed by $\gamma$ mask token embeddings, and produces logits for all mask positions in a single forward pass. Since drafting requires only a single forward pass regardless of block size, DFlash can afford deeper architectures and larger blocks than autoregressive drafters under the same latency budget.

<span id="section-3"></span>

## 3 Architecture

The overview of DSpark is shown in [Figure 1](#figure-01). Recall from [Equation 1](#equation-01) that the per-token latency of speculative decoding is $L=(T_{\mathrm{draft}}+T_{\mathrm{verify}})/\tau$. Autoregressive drafters achieve high $\tau$ but pay $T_{\mathrm{draft}}\propto\gamma$; parallel drafters collapse $T_{\mathrm{draft}}$ to a single pass but sacrifice $\tau$ because each position is predicted independently. Meanwhile, fixed-length verification wastes $T_{\mathrm{verify}}$ on low-confidence suffix tokens that are almost certain to be rejected. DSpark addresses these limitations with two complementary components:

- **Semi-autoregressive generation** ([Section 3.1](#section-3-1)). A parallel backbone handles the bulk of draft computation, which keeps $T_{\mathrm{draft}}$ nearly independent of $\gamma$. A lightweight sequential block then injects dependency among draft tokens, improving $\tau$ at minimal additional latency.
- **Confidence-scheduled verification** ([Section 3.2](#section-3-2)). A confidence head estimates per-position acceptance probabilities, and a hardware-aware scheduler uses these estimates to prune low-confidence suffix tokens, cutting unnecessary verification compute.

<span id="figure-01"></span>

![Figure 1. **The DSpark architecture and decoding cycle.** Given prompt tokens ABC, the target model executes one step to generate the next token D, which serves as the anchor for the drafting phase. Using D as the input, DSpark employs a heavy parallel backbone and a lightweight sequential head to generate draft tokens EFGH along with their corresponding confidence scores $c_{1}$-$c_{4}$. The Hardware-Aware Prefix Scheduler then evaluates these scores to retain the prefix EFG and drop the low-confidence token H. Finally, the target model verifies the scheduled prefix in parallel. As illustrated, E and F are accepted while G is rejected, prompting the model to generate a corrected token G<sup>∗</sup> to complete the current round.](../../papers/dspark/figure-01.png)

**Figure 1.** **The DSpark architecture and decoding cycle.** Given prompt tokens ABC, the target model executes one step to generate the next token D, which serves as the anchor for the drafting phase. Using D as the input, DSpark employs a heavy parallel backbone and a lightweight sequential head to generate draft tokens EFGH along with their corresponding confidence scores $c_{1}$-$c_{4}$. The Hardware-Aware Prefix Scheduler then evaluates these scores to retain the prefix EFG and drop the low-confidence token H. Finally, the target model verifies the scheduled prefix in parallel. As illustrated, E and F are accepted while G is rejected, prompting the model to generate a corrected token G<sup>∗</sup> to complete the current round.

<span id="section-3-1"></span>

### 3.1 Semi-Autoregressive Generation

A parallel drafter produces all $\gamma$ draft logits in one forward pass, so each prediction cannot condition on tokens sampled elsewhere in the block. When the context admits multiple plausible continuations, e.g., “of course” and “no problem”, a parallel drafter may produce incoherent combinations such as “of problem” or “no course”, because each position marginalizes over all possible predecessors rather than conditioning on the one actually sampled [Gu18, Hua22e]. Acceptance rate thus decays rapidly along the block, wasting both draft and verification compute. We therefore adopt a **semi-autoregressive** structure that splits draft generation into two stages:

**Parallel stage.** A parallel backbone (in our instantiation, DFlash [Che26d]) runs a single forward pass over the entire block, producing hidden states $h_{1},\ldots,h_{\gamma}$ and base logits $U_{1},\ldots,U_{\gamma}$. We make only a minor modification to the original DFlash backbone: instead of feeding an anchor token plus $\gamma$ mask tokens and predicting only the mask positions, we treat the anchor itself as the first prediction position, so $\gamma$ input tokens (anchor $+$ $\gamma{-}1$ masks) yield $\gamma$ draft logits. This reduces draft computation while maintaining similar draft quality.

**Sequential stage.** The sequential stage supplements the base logits with a prefix-dependent transition bias $B_{k}(x_{0},x_{<k},x_{k})$, allowing each draft position to condition on previously sampled tokens within the block. Rather than defining a globally normalized energy model, the sequential stage induces a causal block distribution through an autoregressive factorization:

<span id="equation-04"></span>

$$
P(X\mid x_{0})=\prod_{k=1}^{\gamma}p_{k}(x_{k}\mid x_{0},x_{<k}),\qquad p_{k}(v\mid x_{0},x_{<k})=\frac{\exp\!\left(U_{k}(v)+B_{k}(x_{0},x_{<k},v)\right)}{\sum_{u\in\mathcal{V}}\exp\!\left(U_{k}(u)+B_{k}(x_{0},x_{<k},u)\right)}.
$$

Here, $x_{0}$ denotes the anchor token from the previous verification cycle, $U_{k}$ is the base logit vector produced by the parallel backbone at position $k$, and $\mathcal{V}$ is the vocabulary. At inference time, the sequential block samples left to right according to $p_{k}(\cdot\mid x_{0},x_{<k})$. Because this sampling process is inherently sequential, the block must be computationally lightweight ($T_{\mathrm{sequential}}\ll T_{\mathrm{parallel}}$) so that the overall draft latency remains dominated by the parallel stage. We describe two instantiations of the sequential block below.

- **Markov head.** The simplest instantiation restricts $B_{k}$ to depend only on the immediately preceding token, reducing it to a first-order transition $B(x_{k-1},x_{k})$. In principle this is a full $V\times V$ matrix $B$; we approximate it with a low-rank factorization $B=W_{1}W_{2}$, where $W_{1}\in\mathbb{R}^{V\times r}$ and $W_{2}\in\mathbb{R}^{r\times V}$. Given the preceding token $x_{k-1}$, the transition bias for position $k$ is:

  <span id="equation-05"></span>

  $$
  B(x_{k-1},\,\cdot\,)=W_{1}[x_{k-1}]\,W_{2}\;\in\;\mathbb{R}^{V},
  $$

  where $W_{1}$ serves as an embedding lookup table and $W_{2}$ as a logit projection. The low-rank factorization ($r{=}256$ by default) keeps both storage and per-step compute small, making the sequential loop efficient even for large vocabularies. Returning to the earlier example: once position 1 samples “of”, the Markov head boosts “course” and suppresses “problem” at position 2, which mitigates the cross-mode collision.
- **RNN head.** The Markov head is memoryless beyond one step—position $k$ cannot access tokens before $x_{k-1}$. The RNN head relaxes this by maintaining a recurrent state $s_{k}$ that accumulates the full prefix history within a block. At each step, the module concatenates the current state $s_{k-1}\in\mathbb{R}^{r}$, the previous token embedding $W_{1}[x_{k-1}]\in\mathbb{R}^{r}$, and the backbone hidden $h_{k}\in\mathbb{R}^{d}$ into an input vector $z_{k}=[s_{k-1};\,W_{1}[x_{k-1}];\,h_{k}]\in\mathbb{R}^{2r+d}$, then applies a single gated update:

  <span id="equation-06"></span>

  $$
  \begin{aligned}
  s_{k}=\sigma(W_{g}\,z_{k}) & \odot s_{k-1}+\bigl(1-\sigma(W_{g}\,z_{k})\bigr)\odot\tanh(W_{c}\,z_{k}), \\
  & B_{k}(x_{<k},\,\cdot\,)=W_{2}^{\top}\,\tanh(W_{o}\,z_{k}),
  \end{aligned}
  $$

  where $W_{g},W_{c},W_{o}\in\mathbb{R}^{r\times(2r+d)}$ are jointly parameterized by a single linear projection that is split into gate, candidate, and output components. The state $s_{0}$ is initialized to zero.

<span id="section-3-2"></span>

### 3.2 Confidence-Scheduled Verification

The semi-autoregressive architecture enables DSpark to generate large draft blocks efficiently. However, producing more draft tokens does not automatically translate to higher end-to-end speedups. Indiscriminately verifying the full draft block can actually degrade overall system throughput, especially in high-concurrency scenarios [Liu24ab, Hu26].

This performance bottleneck stems from two interacting factors. First, on the data side, draft acceptance rates inherently vary across domains: structured text like code naturally yields high acceptance, whereas open-ended chat has significantly lower acceptance [Xia24j, Abr26]. Second, on the system side, the actual cost of verifying an extra token depends strictly on the engine load. Under light system load, an extra verification incurs minimal penalty even if rejected. However, under high-concurrency deployments, every unnecessary verification occupies target model batch capacity that could otherwise serve other active requests [Liu24ac, Wu25u].

Therefore, fully unlocking the potential of large draft blocks requires a unified mechanism that routes target model compute only toward tokens with a positive expected return. DSpark achieves this by coupling a **confidence head** ([Section 3.2.1](#section-3-2-1)) that predicts prefix survival probabilities, with a **hardware-aware prefix scheduler** ([Section 3.2.2](#section-3-2-2)) that dynamically determines the optimal verification lengths based on current system load.

<span id="section-3-2-1"></span>

#### 3.2.1 Confidence Head

Drawing inspiration from [Hua24f, Wan26c], the confidence head outputs a scalar $c_{k}\in(0,1)$ for each draft position $k$. Crucially, $c_{k}$ models the *conditional* probability that the draft token at position $k$ will survive target verification, given that all preceding tokens in the block have been accepted. The architecture features a lightweight linear projection followed by a sigmoid function:

<span id="equation-07"></span>

$$
c_{k}=\sigma\bigl(w^{\top}[h_{k};\,W_{1}[x_{k-1}]]\bigr),
$$

where $h_{k}$ is the hidden state of the backbone and $W_{1}[x_{k-1}]$ is the Markov Embedding from the previous draft token. We supervise $c_{k}$ using the analytical acceptance rate per-step $c_{k}^{*}$. This rate is determined by the total variation distance between the draft distribution $p_{k}^{d}$ and the target distribution $p_{k}^{t}$:

<span id="equation-08"></span>

$$
c_{k}^{*}=1-\tfrac{1}{2}\|p_{k}^{d}-p_{k}^{t}\|_{1}.
$$

**Post-hoc Calibration.** Unlike threshold-based verification heuristics [Hua24f, Li24v, Zha26c], which only require confidence scores to correctly rank draft token qualities, our hardware-aware scheduling approach (detailed in [Section 3.2.2](#section-3-2-2)) precisely requires the absolute magnitudes of the cumulative acceptance probabilities to compute the expected acceptance length $\tau$. Because neural confidence estimates are often overconfident [Guo17a, Ova19], using the raw confidence scores directly would distort the throughput estimation, leading to suboptimal scheduling.

To address this, we introduce **Sequential Temperature Scaling (STS)**. Because each $c_{i}$ models a conditional probability, the chain rule dictates that the joint probability of a draft prefix being accepted factorizes into the cumulative product $\prod_{i\leqslant k}c_{i}$. Using a held-out validation set, STS calibrates this joint probability consecutively from left to right. Specifically, at each position $k\in\{1,\dots,\gamma\}$, we perform a simple 1D grid search to find the optimal temperature scalar that minimizes the Expected Calibration Error (ECE) [Nae15] of the cumulative product, keeping the already-calibrated scores of all preceding positions fixed. Crucially, temperature scaling is an order-preserving transformation: it rectifies the predicted probabilities to match empirical acceptance rates without disrupting the relative draft token rankings learned by the confidence head.

<span id="section-3-2-2"></span>

#### 3.2.2 Hardware-Aware Prefix Scheduler

<span id="algorithm-01"></span>

**Algorithm 1: Hardware-Aware Prefix Scheduler.**

- **Require:** Active requests $r\in\{1,\dots,R\}$; confidence sequence $c_{r,1},\dots,c_{r,\gamma}$ per request; profiled step curve $\mathrm{SPS}(B)$.
- **Ensure:** Selected per-request prefix lengths $\ell_{1}^{*},\dots,\ell_{R}^{*}$.
- **For** $r=1$ **to** $R$:
  - Compute prefix survival probabilities: $a_{r,j}\gets\prod_{i\leq j}c_{r,i}$ for $j=1,\dots,\gamma$.
- Construct candidate space $\mathcal{E}\gets\{(r,j)\mid a_{r,j}>0\}$ and sort descending by $a_{r,j}$.
- Initialize states: $\ell_{r}\gets0$ for all $r$; Batch size $B\gets R$; Expected accepts $\tau^{*}\gets R$.
- Initialize tracking: $\Theta_{\mathrm{best}}\gets R\cdot\mathrm{SPS}(R)$; Selected lengths $\ell_{r}^{*}\gets0$ for all $r$.
- **For each** $(r,j)\in\mathcal{E}$ in sorted order:
  - $\ell_{r}\gets j$; $B\gets B+1$; $\tau^{*}\gets\tau^{*}+a_{r,j}$.
  - Current throughput $\Theta\gets\tau^{*}\cdot\mathrm{SPS}(B)$.
  - **If** $\Theta>\Theta_{\mathrm{best}}$:
    - $\Theta_{\mathrm{best}}\gets\Theta$; Update selected lengths $\ell_{r}^{*}\gets\ell_{r}$.
  - **Else:**
    - **break**
- **return** $(\ell_{1}^{*},\dots,\ell_{R}^{*})$ achieving $\Theta_{\mathrm{best}}$.

Prior methods [Hua24f, Li24v] typically apply a static threshold to confidence scores to determine verification length. While effective under isolated, single-request assumptions, static thresholds can be suboptimal in high-concurrency production systems, where the utility of verifying a draft token depends heavily on the current system load.

To address this, we formulate verification length selection as a global throughput maximization problem ([Algorithm 1](#algorithm-01)). Consider a batch of $R$ active requests. For request $r$, let $c_{r,1},\dots,c_{r,\gamma}$ be the per-position confidence estimates, and let $\ell_{r}\in\{0,\dots,\gamma\}$ denote the scheduled verification length. Because speculative decoding dynamically accepts draft tokens only as a continuous prefix, the survival probability of a token at position $j$ is the cumulative product $a_{r,j}=\prod_{i\leqslant j}c_{r,i}$.

In a single verification step, the total batch size (measured in tokens) sent to the target model is $B=\sum_{r=1}^{R}(1+\ell_{r})$, and the expected number of successfully accepted tokens is $\tau=\sum_{r=1}^{R}\bigl(1+\sum_{j=1}^{\ell_{r}}a_{r,j}\bigr)$. Under a simplifying assumption [+2], let $\mathrm{SPS}(B)$ denote the engine throughput, measured in steps per second, for a given forward-pass batch size $B$. Crucially, this capacity curve is profiled once during engine initialization and stored as a lightweight cost table. Our scheduler then aims to maximize the expected system-wide token throughput $\Theta=\tau\cdot\mathrm{SPS}(B)$ by dynamically selecting verification lengths $\ell_{1},\dots,\ell_{R}$.

Although finding the global maximum of $\Theta$ appears to be a combinatorial search, the objective structure allows for an efficient greedy solution. Because $a_{r,j}$ is monotonically non-increasing with respect to $j$ (i.e., $a_{r,j}\leq a_{r,j-1}$), the marginal gain in expected accepted tokens for extending request $r$’s verification length from $j-1$ to $j$ is exactly $a_{r,j}$. This monotonicity ensures that sorting candidate tokens globally by $a_{r,j}$ naturally respects intra-block prefix dependencies. Consequently, if the total verification batch size $B$ were fixed, the optimal allocation $\{\ell_{r}\}$ would be determined by greedily selecting the draft tokens with the highest survival probabilities from the global pool of all $\{a_{r,j}\}$.

Building on this insight, the optimization can be evaluated along this greedy admission path. We first globally sort all valid prefix extensions in descending order of survival probability. To dynamically determine the optimal target batch size $B$, we incrementally admit tokens from this sorted pool, updating the expected throughput $\Theta$ via an lookup from the cost table.

Lossless speculative decoding strictly requires the *non-anticipating property*: admission decisions must not depend on future candidate tokens [Lev23, Che23]. Because our confidence head relies on the Markov feature of the previously sampled token, computing the next survival probability $a_{r,k+1}$ explicitly requires the instantiated candidate $x_{r,k}$. A retrospective global search would thus inadvertently leak $x_{r,k}$ into the admission decision for step $k$, introducing selection bias (we provide a concrete counterexample demonstrating this theoretical violation in [Section 8](#section-8)).

To enforce strict causality, the scheduler ([Algorithm 1](#algorithm-01)) employs an early-stopping mechanism. By breaking the greedy search immediately when the throughput drops ($\Theta\leq\Theta_{\mathrm{best}}$), the truncation decision relies solely on the prefix processed up to that exact step. This isolates the admission event from future tokens, ensuring exact target-distribution recovery. Note that this stepwise early-stopping yields the global maximum throughput if and only if the objective $\Theta$ is unimodal, which implicitly assumes a smoothly decaying hardware capacity curve. We address the engineering adaptations required for real-world, non-smooth SPS characteristics and asynchronous system pipelines in [Section 5.2](#section-5-2).

<span id="section-3-3"></span>

### 3.3 Training

During training, we randomly sample multiple anchor positions from each target sequence to form $\gamma$-token blocks as training data. The target model is frozen throughout training; the draft model shares its embedding layer and language modeling head and keeps them frozen, updating only the backbone drafter, sequential block, and confidence head.

The training objective consists of three terms: a cross-entropy loss $\mathcal{L}_{\mathrm{ce}}$, a distribution-matching loss $\mathcal{L}_{\mathrm{tv}}$, and a confidence loss $\mathcal{L}_{\mathrm{conf}}$. All three are position-weighted by $w_{k}=\exp(-(k{-}1)/\gamma)$ [Che26d], which emphasizes earlier block positions that contribute more to the expected acceptance length under prefix-based verification. The cross-entropy loss $\mathcal{L}_{\mathrm{ce}}$ trains the drafter to predict the correct next token:

<span id="equation-09"></span>

$$
\mathcal{L}_{\mathrm{ce}}=-\sum_{k=1}^{\gamma}w_{k}\log p^{d}_{k}(x_{k}^{*}),
$$

where $x_{k}^{*}$ is the ground-truth token and $p^{d}_{k}$ is the draft distribution. The distribution-matching loss $\mathcal{L}_{\mathrm{tv}}$ penalizes the total variation distance between the draft and target distributions:

<span id="equation-10"></span>

$$
\mathcal{L}_{\mathrm{tv}}=\sum_{k=1}^{\gamma}w_{k}\|p_{k}^{d}-p_{k}^{t}\|_{1}.
$$

Since the total variation distance is a direct proxy for the acceptance rate: the per-step acceptance probability equals $1-\frac{1}{2}\|p^{d}-p^{t}\|_{1}$ [Lev23], minimizing $\mathcal{L}_{\mathrm{tv}}$ directly maximizes the expected acceptance rate.

The confidence loss $\mathcal{L}_{\mathrm{conf}}$ is a binary cross-entropy that trains the confidence head to predict the soft acceptance label $c_{k}^{*}$ from [Equation 8](#equation-08):

<span id="equation-11"></span>

$$
\mathcal{L}_{\mathrm{conf}}=-\sum_{k=1}^{\gamma}w_{k}\bigl[c_{k}^{*}\log c_{k}+(1-c_{k}^{*})\log(1-c_{k})\bigr].
$$

The overall objective is a weighted combination of the three terms (with default weights $\alpha_{\mathrm{ce}}=0.1$, $\alpha_{\mathrm{tv}}=0.9$, $\alpha_{\mathrm{conf}}=1.0$):

<span id="equation-12"></span>

$$
\mathcal{L}=\alpha_{\mathrm{ce}}\,\mathcal{L}_{\mathrm{ce}}+\alpha_{\mathrm{tv}}\,\mathcal{L}_{\mathrm{tv}}+\alpha_{\mathrm{conf}}\,\mathcal{L}_{\mathrm{conf}}
$$

<span id="section-4"></span>

## 4 Experiments

In this section, we validate the draft quality of DSpark using offline benchmarks and report the effectiveness of confidence scheduler under online production traffic in [Section 5](#section-5). The experimental setup is described in [Section 4.1](#section-4-1), main results in [Section 4.2](#section-4-2), and additional analyses are included in [Section 4.3](#section-4-3).

<span id="section-4-1"></span>

### 4.1 Experimental Setup

**Target and draft models.** We evaluate DSpark on four target models spanning different scales and model families: Qwen3-{4B, 8B, 14B} [Yan25g], and Gemma4-12B [Gem26]. For draft models, we compare DSpark with two representative drafters: DFlash [Che26d], a state-of-the-art parallel drafter, and Eagle3 [Li25], an autoregressive drafter based on Training-Time Test (TTT). For strict and fair comparison, we retrain all drafters in the same [training framework](https://github.com/deepseek-ai/DeepSpec) and on the same data [+3]. We align Eagle3’s TTT horizon (7) with the block size (7) used by DFlash and DSpark, and we use the same target-model feature layers for all drafters. For the number of draft model layers, we set 1 for Eagle3 and 5 for DSpark and DFlash [Che26d]. Unless otherwise stated, DSpark denotes the Markov-head variant; we study the RNN-head variant in [Section 4.3.2](#section-4-3-2).

**Training data.** We use [Open-PerfectBlend](https://huggingface.co/datasets/mlabonne/open-perfectblend), an open-sourced version of PerfectBlend [Xu24k] consisting of 1.3 million samples. It is a general-purpose instruction dataset containing chat (17.6%), math (39.4%), code (38.9%), and instruction-following data (4.1%). We only use the prompts from Open-PerfectBlend; responses are regenerated by each target model with recommended sampling parameters. Each drafter is trained for 10 epochs to ensure full convergence. For data generation and evaluation, we adopt the non-thinking mode.

**Evaluation protocol.** We evaluate the performance of different algorithms on three domains:

- **Mathematical Reasoning**, including GSM8K [Cob21], MATH500 [Lig23a] and AIME25 [Zha25ba].
- **Code Generation**, including MBPP [Aus21b], HumanEval [Che21h] and Live-CodeBench [Jai25a].
- **Daily Chat**, including MT-Bench [Sto23e], Alpaca [Tao23] and Arena-Hard [Li24w, Li24j].

For all benchmarks, we use standard speculative decoding [Lev23, Che23] with the sampling temperature set to $1.0$. We report the accepted length ($\tau$) per decoding round [+4]. For all drafters, we use chain-based drafting.

<span id="table-01"></span>

![Table 1. Main speculative decoding results. We report accepted length ($\tau$) per decoding round (higher is better) for different target models and domains. Bold marks the best results.](../../papers/dspark/table-01.png)

**Table 1.** **Main speculative decoding results.** We report accepted length ($\tau$) per decoding round (higher is better) for different target models and domains. **Bold** marks the best results.

<span id="section-4-2"></span>

### 4.2 Experimental Results

To isolate the raw draft quality from system-level scheduling policies, our offline evaluation disables the confidence scheduler, forcing all drafters to propose a fixed block of tokens. The main results, measured by the average accepted length ($\tau$) per round, are reported in [Table 1](#table-01).

DSpark consistently outperforms both the autoregressive baseline (Eagle3) and the parallel baseline (DFlash) across all evaluated target models and benchmark domains. Specifically, across the Qwen3-4B, 8B, and 14B models, DSpark improves the macro-average accepted length over Eagle3 by 30.9%, 26.7%, and 30.0%, respectively. Similarly, compared to DFlash, DSpark yields relative improvements of 16.3%, 18.4%, and 18.3% across the three scales. Crucially, this advantage generalizes across model families, as demonstrated by the consistent performance gains on the Gemma4-12B target.

Beyond the average improvements, [Table 1](#table-01) reveals a strong domain effect: the accepted length is naturally higher on structured tasks (e.g., 5.57 on math and 5.12 on code for Qwen3-4B) than on open-ended chat (3.49). This inherent variance in data predictability means a static verification length often wastes compute on trailing tokens that are highly likely to be rejected. This directly motivates our confidence-scheduled verification, which dynamically prunes the draft block based on expected acceptance.

<span id="section-4-3"></span>

### 4.3 Experimental Analysis

<span id="section-4-3-1"></span>

#### 4.3.1 Why Can Parallel Generation Outperform Autoregression?

<span id="figure-02"></span>

![Figure 2. Position-wise conditional acceptance. We report the empirical conditional acceptance rate for each draft position, averaged across benchmarks within each domain using the Qwen3-4B target model. Unlike standard prefix survival, this metric isolates the baseline predictive quality at position $k$ by removing the penalty of previous rejections. Notice that the autoregressive drafter (Eagle3) remains stable or trends upward, while the parallel drafter (DFlash) suffers suffix decay.](../../papers/dspark/figure-02.png)

**Figure 2.** **Position-wise conditional acceptance.** We report the empirical conditional acceptance rate for each draft position, averaged across benchmarks within each domain using the Qwen3-4B target model. Unlike standard prefix survival, this metric isolates the baseline predictive quality at position $k$ by removing the penalty of previous rejections. Notice that the autoregressive drafter (Eagle3) remains stable or trends upward, while the parallel drafter (DFlash) suffers suffix decay.

[Table 1](#table-01) presents a counter-intuitive observation: the parallel drafter (DFlash) and the semi-autoregressive drafter (DSpark) often yield longer accepted lengths than the fully autoregressive drafter (Eagle3). This finding contrasts with the standard expectation that step-by-step autoregression produces higher-quality sequences than parallel models [Ren20, Isr26, Zhe25b].

To analyze this behavior, we examine performance beyond the macro-level accepted length. Using the Qwen3-4B target model and the benchmark sets described in [Section 4.1](#section-4-1), we introduce *position-wise conditional acceptance* tracked during actual speculative decoding rollouts. Specifically, for a given draft position $k$, the evaluation denominator counts only the instances where the target model successfully verifies and accepts all preceding draft tokens from $1$ to $k-1$. The metric then calculates the proportion of these valid instances where the token at position $k$ is also accepted. This approach ensures that the evaluation of position $k$ is not penalized by earlier prefix errors, revealing the underlying predictive quality at each specific step. [Figure 2](#figure-02) details these measurements, demonstrating clear behavioral differences across the architectures.

**The Capacity Advantage at Position 1.** At the first draft position, both architectures predict the next token based solely on the target context. The performance divergence here stems strictly from architectural capacity: autoregressive models like Eagle3 are constrained to shallow networks due to their $O(\gamma)$ latency, whereas $O(1)$ parallel drafters can afford much deeper networks. This structural gap yields a substantial accuracy margin at position 1, with DFlash starting noticeably higher than Eagle3 (e.g., 0.88 vs. 0.81 on Math, and 0.72 vs. 0.53 on Chat). Because speculative decoding operates as a strict prefix-matching survival process, the first token carries the highest leverage—a rejection here immediately invalidates the entire block. Consequently, this initial capacity advantage disproportionately boosts the final accepted length, explaining why parallel drafters ultimately outperform autoregressive ones globally despite rapid acceptance decay at later positions.

**The Limitation of Independence at Later Positions.** Examining the tail of the curves (positions 2 through 7) exposes the inherent limitation of independent parallel generation. As earlier tokens lock in a specific semantic path, subsequent tokens naturally become more predictable. Autoregressive models like Eagle3 effectively leverage this conditional certainty, maintaining or even increasing conditional acceptance deeper into the block (e.g., from 0.53 to 0.74 on Chat). In contrast, DFlash suffers from rapid acceptance decay, dropping from 0.87 to 0.78 on Code and 0.72 to 0.63 on Chat. Because each parallel position marginalizes over all possible prior tokens rather than conditioning on an exact sampled prefix, the model frequently proposes inconsistent suffix combinations—a mode known as multi-modal collision [Gu18, Ste18].

**Mitigating Suffix Decay with Semi-Autoregression.** The preceding analysis highlights a clear architectural objective: combining the high capacity of a parallel backbone for the initial token with the dependency modeling of an autoregressive model for subsequent tokens. This directly motivates DSpark’s semi-autoregressive design. As shown in [Figure 2](#figure-02), DSpark inherits the high initial acceptance of the deep parallel drafter (e.g., starting at 0.93 on Math). Simultaneously, its lightweight sequential head mitigates the rapid acceptance decay typical of parallel generation. By resolving this trade-off, DSpark maintains a high and stable conditional acceptance rate throughout the entire draft block.

<span id="section-4-3-2"></span>

#### 4.3.2 A Little Autoregression Goes a Long Way

Building on the insights from [Section 4.3.1](#section-4-3-1), we explore the architectural design space of DSpark along two dimensions: drafter depth (number of transformer layers) and proposal length (block size $\gamma$). Unless otherwise stated, all experiments in this section use Qwen3-4B as the target model and follow the evaluation protocol detailed in [Section 4.1](#section-4-1).

<span id="figure-03"></span>

![Figure 3. Effect of drafter depth. With proposal length fixed, DSpark’s performance improves as drafter layers are added. Notably, a shallow 2-layer DSpark outperforms a deeper 5-layer DFlash baseline, highlighting the parameter efficiency of sequential modeling.](../../papers/dspark/figure-03.png)

**Figure 3.** **Effect of drafter depth.** With proposal length fixed, DSpark’s performance improves as drafter layers are added. Notably, a shallow 2-layer DSpark outperforms a deeper 5-layer DFlash baseline, highlighting the parameter efficiency of sequential modeling.

<span id="figure-04"></span>

![Figure 4. Effect of proposal length and latency overhead. DSpark consistently outperforms DFlash across various block sizes (left three panels). The rightmost panel demonstrates that the sequential head introduces minimal latency overhead during serving.](../../papers/dspark/figure-04.png)

**Figure 4.** **Effect of proposal length and latency overhead.** DSpark consistently outperforms DFlash across various block sizes (left three panels). The rightmost panel demonstrates that the sequential head introduces minimal latency overhead during serving.

**Drafter Depth.** Increasing the number of transformer layers naturally expands a draft model’s predictive capacity. To isolate this effect, we fix the block size to 7 and vary the number of DSpark layers from 1 to 5, comparing it against a 5-layer DFlash baseline. [Figure 3](#figure-03) aggregates the accepted lengths across the math, code, and chat domains. As expected, DSpark’s performance improves monotonically with depth, with the steepest marginal gain occurring from one to two layers. Notably, a 2-layer DSpark outperforms the 5-layer DFlash baseline across all domains. This demonstrates that injecting local auto-regression via a lightweight sequential head offers a highly favorable accuracy-parameter trade-off, achieving better sequence coherence than simply stacking deeper parallel layers.

**Proposal Length.** Next, we fix the drafter depth to 5 layers and scale the draft length (proposal length $\gamma$ plus one anchor token) across $\{4,8,12,16\}$ to evaluate performance on longer draft blocks. For DSpark, we evaluate both the default Markov head and the RNN head. The first three panels of [Figure 4](#figure-04) show that DSpark consistently outperforms DFlash at every proposal length. More importantly, the performance gap steadily widens as $\gamma$ increases. Because pure parallel generation (DFlash) suffers from rapid acceptance decay ([Figure 2](#figure-02)), its marginal utility diminishes for long blocks. DSpark mitigates this decay, causing its relative gain over DFlash to grow. For instance, at $\gamma=7$, DSpark improves the accepted length by 16% on math, 15% on code, and 18% on chat; at $\gamma=15$, these gains expand to 30%, 26%, and 22%, respectively. Also, RNN head provides only marginal additional gains over the Markov head, mainly at longer proposal lengths. Given its higher implementation complexity and less favorable deployment properties, we use the Markov head as the default.

**Latency Overhead.** We quantify the overhead of the sequential generation loop in DSpark. The rightmost panel of [Figure 4](#figure-04) reports the per-round engine latency—comprising one target verification pass, the parallel draft block forward, and the serial sampling loop—measured at a batch size of 128. To prevent sequence-length bias, the reported latency represents the arithmetic mean across varying context lengths ($\{512,1024,2048,4096\}\ \mathrm{tokens}$). Since the target model dominates the verification compute time at this batch size, the sequential block’s latency overhead is negligible. Consequently, scaling the draft length from 4 to 16 adds a marginal 0.2% to 1.3% to the full-round latency over the DFlash baseline, despite delivering up to a 30% improvement in accepted length.

<span id="section-4-3-3"></span>

#### 4.3.3 Verify Smarter, Not Longer: The Role of Confidence Head

While DSpark sustains high acceptance over long draft blocks, verifying the entire proposal remains inefficient [Hua24f, Hu26]. Due to the inherent domain variance noted in [Section 4.2](#section-4-2), trailing tokens in open-ended chat still face high rejection risks, making blind verification a waste of target compute. To evaluate whether the confidence head can effectively prune these unpromising suffixes, we conduct an offline threshold sweep using Qwen3-4B. We validate the estimator in isolation here, reserving the hardware-aware prefix scheduler ([Section 3.2.2](#section-3-2-2)) for live production evaluation in [Section 5](#section-5).

<span id="figure-05"></span>

![Figure 5. Confidence threshold sweep. A threshold of 0 corresponds to standard fixed-length verification. As the threshold increases, the overall acceptance rate steadily rises because the confidence head effectively prunes tokens that would ultimately be rejected (hashed bars).](../../papers/dspark/figure-05.png)

**Figure 5.** **Confidence threshold sweep.** A threshold of 0 corresponds to standard fixed-length verification. As the threshold increases, the overall acceptance rate steadily rises because the confidence head effectively prunes tokens that would ultimately be rejected (hashed bars).

<span id="figure-06"></span>

![Figure 6. The Reliability Diagram on Alpaca Dataset. While the raw confidence estimator achieves strong discrimination, its predictions are inherently overconfident. Applying post-hoc calibration helps to align the prefix survival probabilities with empirical acceptance rates. The shaded background histogram represents the frequency distribution of sample counts across different confidence bins.](../../papers/dspark/figure-06.png)

**Figure 6.** **The Reliability Diagram on Alpaca Dataset.** While the raw confidence estimator achieves strong discrimination, its predictions are inherently overconfident. Applying post-hoc calibration helps to align the prefix survival probabilities with empirical acceptance rates. The shaded background histogram represents the frequency distribution of sample counts across different confidence bins.

**Diagnostic: Static Threshold Sweep.** [Figure 5](#figure-05) plots the average tokens per step (bars) and the overall acceptance rate (line) across confidence thresholds. As the threshold increases, the acceptance rate steadily rises because the estimator filters out tokens that would ultimately be rejected (hashed bars). This suggests that the confidence head can identify lower-value suffix tokens and this pruning is most pronounced on chat workloads, where higher-entropy token distributions limit the efficiency of fixed-length verification. In the Chat subplot, raising the threshold significantly reduces rejected tokens, increasing the acceptance rate from 45.7% to 95.7%. In contrast, structured tasks (Math and Code) experience milder pruning and retain more draft tokens, with acceptance rates rising from 76.9% to 92.5% and 67.6% to 92.0%, respectively.

**From Static Thresholds to Calibrated Scheduling.** While useful for diagnostics, a static threshold is sub-optimal in dynamic serving environments because it ignores system load: verifying low-confidence tokens incurs minimal opportunity cost under low concurrency, but wastes critical batch capacity under high concurrency. This load dependency motivates the hardware-aware prefix scheduler. As formulated in [Section 3.2](#section-3-2), maximizing system-level throughput requires the confidence model to exhibit both strong predictive discrimination and precise calibration to accurately estimate cumulative survival probabilities. The reliability diagram ([Figure 6](#figure-06)) demonstrates that while the raw model achieves strong discrimination (ROC-AUC [Han82] ranging from 0.81 to 0.90), it is overly confident (ECE 3%-8%). Applying post-hoc STS ([Section 3.2.1](#section-3-2-1)) mitigates this overconfidence, reducing the average ECE to $\sim$1% and yielding reliable survival estimates.

<span id="section-5"></span>

## 5 Real-World Deployment of DSpark

While [Section 4](#section-4) establishes the algorithmic gains of DSpark on offline benchmarks, deploying it alongside large-scale models like DeepSeek-V4 [Int26] introduces additional system-level challenges across both training and inference. In this section, we present the end-to-end production pipeline of DSpark. We detail our scalable training mechanisms, the system-level optimizations necessary to deploy the hardware-aware prefix scheduler ([Section 3.2.2](#section-3-2-2)), and the framework’s end-to-end performance under live user traffic.

<span id="section-5-1"></span>

### 5.1 Scalable and Flexible Training

The DSpark draft models are co-deployed with the preview versions of DeepSeek-V4-Flash and DeepSeek-V4-Pro [Int26]. The parallel backbone comprises three MoE layers [Dai24] with mHC [Xie26] and a sliding window attention of 128. We configure the maximum block size to $\gamma=5$ and utilize the Markov head for sequential modeling. Furthermore, the confidence head is trained end-to-end alongside the draft model and subsequently calibrated via STS to provide reliable scheduling signals.

Training the draft model requires the target model’s output distributions for supervision. Evaluating both models over the full document context incurs substantial memory footprints and inter-worker communication overhead. To address these bottlenecks, we implement two system-level optimizations within our internal training framework (HAI-LLM) [+5]:

- **Hidden state communication.** Transferring the target model’s full-vocabulary logits ($V\approx 10^{5}$) across parallel workers creates a significant bandwidth bottleneck. Instead, we temporarily cache the target model’s forward-pass activations and communicate only the hidden states immediately preceding the language modeling (LM) head. The LM head projection is then executed locally on the draft model’s workers only for the sampled target positions. This reduces the per-token communication complexity to $O(d)$, where $d$ is the hidden dimension.
- **Anchor-bounded sequence packing.** To decouple the draft model’s computational cost from the target model’s context length, we sample a fixed number of draft anchors from the training sequence and pack these isolated prediction blocks into dense training batches. We manage this packing via token-level attention indices rather than standard 2D masks. This maintains exact causal masking across multiple independent sequences and anchors, avoiding the computational and memory overhead associated with standard padding.

<span id="section-5-2"></span>

### 5.2 Hardware-Aware Prefix Scheduler in Practice

In [Section 3.2.2](#section-3-2-2), [Algorithm 1](#algorithm-01) provides a theoretically sound and lossless scheduling mechanism. However, directly deploying this algorithm into a production environment exposes two fundamental conflicts with real-world infrastructure. First, the algorithm assumes a smooth, unimodal capacity curve, whereas the true hardware capacity $\mathrm{SPS}(B)$ is inherently discrete, exhibiting a jagged, step-wise degradation [Yan20e]. Second, the algorithm requires scheduling of dynamic draft tokens per step, which clashes with continuous CUDA graph replay [Spe23a] and Zero-Overhead Scheduling (ZOS) [Zhu25c, Zhe24].

To navigate the trade-offs among system compatibility, throughput, and algorithmic correctness, we adapt the scheduler to operate asynchronously. Because ZOS requires the batch size for the next step to be known before the current step completes, synchronous scheduling would inevitably stall the GPU pipeline. Instead, we approximate the upcoming verification capacity using the confidence head outputs from two steps prior. Mechanically, the candidate tokens in the current step are still strictly sorted by their actual, up-to-date cumulative confidence scores; the historical prediction from two steps prior is used solely to determine the dynamic truncation length (i.e., the batch capacity limit $K$). This effectively casts the admission process as a dynamic top-$K$ selection. While approximating the capacity $K$ introduces a slight temporal offset, the selection mechanism is fundamentally rank-preserving: the most confident draft tokens are always prioritized for verification. This adaptation fully hides scheduling latency and ensures seamless ZOS integration.

Building on this asynchronous pipeline, we resolve the hardware utilization bottleneck. To prevent the scheduler from being trapped in local minima by jagged SPS cliffs, we remove the early-stopping `break`, enabling an unconstrained global search. Ordinarily, this retrospective search would leak future token information and violate the lossless guarantee ([Section 8](#section-8)). However, our ZOS-driven adaptation naturally prevents this. Because the unconstrained search evaluates only historical predictions from two steps prior, the admission decision is isolated from the realization of the current token $x_{r,k}$. The truncation length inherently depends only on information available from two steps prior. Thus, asynchronous design forms a causal barrier, maximizing physical throughput across hardware cliffs while preserving the exact target distribution.

<span id="section-5-3"></span>

### 5.3 High-Throughput and Low-Latency Inference

During decoding, production serving systems must simultaneously optimize two competing objectives: per-request latency and aggregate throughput [Kwo23, Zho24, Zha25c]. The former governs the quality of service for individual users—a factor increasingly critical in agent-based workloads [Tiw26]—while the latter determines the total number of concurrently served users. Because speculative decoding inevitably incurs wasted verification compute, it inherently navigates this trade-off, trading extra system compute for faster per-request generation.

In our deployment setting, however, the number of requests processed per step is frequently constrained by resource limits (e.g., fixed KV-cache capacity per request) and the pool of available user traffic (e.g., RL long-tail loads). Consequently, the effective batch size persistently remains well below the GPU’s compute-saturating threshold. Under this regime, the traditional trade-off simplifies: given a fixed concurrency limit, maximizing per-GPU total token throughput and maximizing the generation speed per user (*tok/s/user*) become highly correlated objectives rather than competing ones.

To achieve this maximum throughput, the asynchronous scheduler ([Section 5.2](#section-5-2)) actively routes idle compute toward the most promising draft tokens. However, executing this dynamic routing introduces a severe challenge at the physical execution layer: the inference framework must efficiently support variable-length queries within a single batch. Standard decode kernels are heavily optimized for fixed query lengths; naively processing variable-length verified prefixes leads to severe GPU under-utilization due to padding and uneven workload distribution. We resolve this by decoupling physical execution from logical sequence tracking. In our compute kernels, all tokens across different requests are flattened and processed identically as independent elements. The complex intra-sequence dependencies are then strictly conveyed via a marker tensor integrated into our sparse attention implementation. Specifically on the DeepSeek-V4 architecture, only the index-attention and compress kernels require modification to support this variable-length routing, allowing the dynamic scheduler to operate seamlessly without introducing low-level execution overhead.

<span id="section-5-4"></span>

### 5.4 Performance under Live User Traffic

<span id="figure-07"></span>

![Figure 7. **Throughput vs. TPS.** Aggregate output token throughput against per-request generation speed (tok/s/user) under live traffic. In our production deployment, DSpark improves the observed throughput-interactivity frontier relative to the MTP-1 baseline under the measured traffic and engine configurations.](../../papers/dspark/figure-07.png)

**Figure 7.** **Throughput vs. TPS.** Aggregate output token throughput against per-request generation speed (tok/s/user) under live traffic. In our production deployment, DSpark improves the observed throughput-interactivity frontier relative to the MTP-1 baseline under the measured traffic and engine configurations.

We evaluate DSpark-5 (configured with a maximum draft length of $\gamma=5$) against the MTP-1 [Dee24a] baseline within the production serving engines of DeepSeek-V4-Flash (preview) and DeepSeek-V4-Pro (preview). MTP-1 represents the former production setup, having been superseded by DSpark two weeks following the DeepSeek-V4-preview release. This single-token setup was historically maintained in production because deploying a static multi-token drafter (e.g., MTP-3/5) strictly degrades aggregate throughput under high concurrency due to excessive verification overhead. Therefore, comparing DSpark against this established baseline directly demonstrates its ability to safely unlock the performance potential of larger draft blocks in dynamic serving environments. In all figures, the scatter points represent raw telemetry data sampled directly from live user traffic, capturing complex, real-world request distributions, while the solid lines represent the fitted performance frontiers.

**The Serving Pareto Frontier.** [Figure 7](#figure-07) illustrates the trade-off between aggregate system throughput and per-user generation speed (interactivity). To quantify DSpark’s behavior under practical deployment constraints, we evaluate the system at several interactivity SLA anchors. Here, an SLA (Service Level Agreement) specifies the minimum per-user generation speed (in tokens per second) that the system must guarantee.

For the V4-Flash engine, we evaluate the system at SLA anchors of 80 and 120 tok/s/user. At the moderate 80 tok/s/user SLA, DSpark improves aggregate throughput by 51% over the MTP-1 baseline. The stricter 120 tok/s/user SLA represents a qualitatively different regime: under this constraint, the single-token MTP-1 baseline approaches its operational boundary and can sustain only a very small concurrent batch. Consequently, the relative throughput ratio at this point is numerically large, with DSpark achieving a nominal 661% higher aggregate throughput. We therefore interpret this high-SLA point primarily as evidence that DSpark extends the feasible interactivity frontier, rather than as a representative multiplicative speedup over a well-utilized baseline. At matched practical throughput levels, which provide a more stable comparison, DSpark accelerates per-user generation speeds by 60% to 85%.

The V4-Pro deployment shows the same pattern. At the moderate 35 tok/s/user SLA, DSpark improves aggregate throughput by 52%. At the stricter 50 tok/s/user SLA, MTP-1 again enters a low-concurrency regime, yielding a nominal 406% relative throughput advantage for DSpark. As with V4-Flash, we treat this point as an indication that DSpark sustains useful throughput under an interactivity target that the baseline cannot efficiently support. At matched system capacities, DSpark delivers 57% to 78% faster per-user generation. Overall, these results show that DSpark shifts the observed throughput-interactivity frontier outward: it improves throughput in moderate-SLA regimes and, more importantly, preserves non-degenerate serving capacity under strict interactivity constraints.

<span id="figure-08"></span>

![Figure 8. Load-adaptive throughput and verification budgets. Top row (a, b): Aggregate output throughput across varying levels of system concurrency. Bottom row (c, d): The average target verification budget allocated per request. As concurrent load increases, the dynamic scheduler automatically restricts the per-request verification length to prevent resource contention.](../../papers/dspark/figure-08.png)

**Figure 8.** **Load-adaptive throughput and verification budgets.** Top row (a, b): Aggregate output throughput across varying levels of system concurrency. Bottom row (c, d): The average target verification budget allocated per request. As concurrent load increases, the dynamic scheduler automatically restricts the per-request verification length to prevent resource contention.

**Throughput Dynamics under Load.** [Figure 8](#figure-08) analyzes the underlying mechanism driving these gains by plotting aggregate throughput (top row) and the dynamic verification budget (bottom row) against system concurrency.

- Under the moderate concurrency regimes typical of our production deployment (fewer than 200 concurrent requests for V4-Flash and 150 for V4-Pro), the hardware-aware scheduler leverages available target compute capacity by allocating longer verification budgets, expanding from MTP-1’s static 2 tokens to roughly 4-6 tokens per request. This extended verification yields more accepted tokens per forward pass, directly contributing to the throughput gains observed on the Pareto frontier.
- As system concurrency scales and target capacity saturates, the scheduler dynamically restricts this budget. The average verification length decreases smoothly with load, ensuring that low-confidence draft tokens are pruned before they consume critical batch capacity. This load-aware behavior stabilizes production deployment: DSpark maximizes the utility of idle compute under light traffic, while effectively preserving critical batch capacity under heavy traffic.

**Limitations.** Although the prefix scheduler minimizes wasted target-model verification, DSpark still incurs a fixed draft-side cost to generate the initial $\gamma$-token block via the parallel backbone. For complex queries with inherently low acceptance rates, this upfront drafting compute is unrecoverable. Future optimizations could introduce difficulty-aware early exiting within the draft model, enabling such requests to bypass full-block generation.

<span id="section-6"></span>

## 6 Related Work

**Speculative Decoding Algorithms.** Speculative decoding accelerates autoregressive generation by decoupling token proposal from verification. Building on early blockwise methods [Ste18, Sun21, Ge22, Xia23e], modern approaches employ rejection sampling to exactly preserve the target model’s distribution [Che23, Lev23]. Because inference speedup directly depends on the drafter’s efficiency and accuracy, extensive research has focused on optimizing its architecture. Beyond using standalone small language models [Che23, Lev23], subsequent work integrates multi-token heads or feature extrapolators directly into the target model [Cai24d, Ank24, Li24g, Li24v, Li25, Zha25az, Glo24, Dee24a, Cai25, Eld26]. Other strategies include self-speculation via early exits [Zha24ae, Elh24, Liu24ad, Xia25e], dynamic vocabulary compression [Zha25bb, Wil26], prompt lookup [Sax23, Som25], suffix automata [Hu25k], and retrieval [He23, She26a]. To remove the sequential bottleneck of the drafting itself, a line of research proposes parallel or blockwise generation, including Medusa [Cai24d], P-EAGLE [Hui26], PARD [An26, An26a], DART [Liu26b] and DFlash [Che26d]. DDTree, TAPS and JetSpec then extend the draft chain to verifiable trees [Rin26, Wan26d, Hu26a]. Concurrent efforts include: Domino [Hua26a] introduces a CausalEncoder conceptually similar to our RNN Head; DFlare [Zha26d] addresses conditioning bottlenecks via layer-wise fusion.

**System-Aware Scheduling for Speculative Decoding.** Beyond drafter architecture, another line of work focuses on determining the optimal number of speculative tokens to generate or verify in each round. To this end, various approaches adapt draft lengths on the fly using confidence heuristics [Mam24, Li24v, Du24b, Liu26c, Wen26], learned acceptance predictors [Hua24f, Vll26], or bandit-style policies [Liu26d]. Furthermore, recognizing speculative decoding as inherently a system-level scheduling problem, recent works optimize overall goodput and latency by adjusting speculation budgets according to real-time system load and request priority [Dec26, Liu24ab, Hua26b, Li26a, Wu25u, Hu26, Mia24a, Sad25].

**Parallel Generation.** Models that generate tokens in parallel offer a decoding latency nearly independent of output length, making them an attractive alternative to autoregressive decoding. Non-Autoregressive Transformers [Gu18] pioneered this direction by predicting all positions independently in a single pass. However, this forces the model to average over all plausible modes, often producing outputs that mix fragments from different valid sequences. Two broad lines of work have emerged to address this limitation. One direction retains the single-pass architecture but changes what the model sees or how it is trained: introducing latent variables as conditioning input to steer all positions toward a consistent output [Gu18, Kai18, Ma19], or relaxing the training objective so that the model focuses on producing a single coherent output rather than modeling the full distribution over all valid alternatives [Sha21, Du21b, Qia21a, Sha23e]. The other direction reintroduces limited sequential dependency through iterative re-prediction [Gha19, Aus21c, Li22d], block-level autoregression [Wan18g, Arr25], or structured output layers such as CRF [Sun19e], CTC [Lib18, Sah20], HMM [Hua22d], and PCFG [Gui23].

Speculative decoding places a further demand that the drafter must provide exact per-token probabilities for the rejection sampling rule. Most techniques above cannot readily provide such probabilities due to iterative refinement, latent marginalization, or global normalization. For instance, in a design closely related to ours, CRF-NAT [Sun19e] also places a sequential module over parallel hidden states, but its globally normalized partition function prevents exact per-token probability computation. Similarly, when adapting the CTC output layer to parallel speculative decoding, CTC-drafter [Wen24b] is restricted to greedy verification due to the latent marginalization of alignment paths. DSpark circumvents these limitations by keeping the sequential correction local, so per-token probabilities remain exact softmax evaluations.

<span id="section-7"></span>

## 7 Conclusion

In this paper, we present DSpark, a speculative decoding framework designed to overcome the structural and system-level bottlenecks of large language model inference in high-concurrency production environments. Algorithmically, DSpark introduces a semi-autoregressive generation paradigm—coupling a computationally heavy parallel backbone with a lightweight sequential head—to mitigate the rapid suffix decay of independent parallel drafters. At the system level, we formulate verification length selection as a global throughput maximization problem, employing a hardware-aware prefix scheduler that dynamically tailors the target model’s verification budget based on calibrated survival probabilities and real-time engine load. Extensive offline evaluations demonstrate that DSpark substantially outperforms state-of-the-art autoregressive and parallel baselines across diverse domains. Furthermore, its real-world deployment within the DeepSeek-V4 validates its practical value in production serving: by intelligently managing verification overhead, DSpark sustains robust concurrency under heavy load, consistently accelerates per-user generation speeds, and shifts the Pareto frontier of LLM serving outward.

<span id="section-8"></span>

## 8 Counterexample: Selection Bias Without Early-Stopping

We provide a simple counterexample to illustrate how an offline global search, i.e., operating without the `break` condition in [Algorithm 1](#algorithm-01), violates the non-anticipating property required by lossless speculative decoding. Formally, the admission event for the $k$-th draft token, $\ell_{r}\geq k$, must be determined by scheduler-visible information available before the token $x_{r,k}$ is sampled. It must not depend on the realization of $x_{r,k}$ itself. Consider a scenario with a single request ($R=1$) and maximum draft length ($\gamma=2$). Suppose the pre-token confidence for the first position is $a_{1}=0.8$, and the profiled capacity curve is

$$
\mathrm{SPS}(1)=1.0,\qquad\mathrm{SPS}(2)=0.5,\qquad\mathrm{SPS}(3)=0.45.
$$

The expected throughputs for verifying $0$ and $1$ draft tokens are

$$
\begin{aligned}
\Theta_{0} & =1\cdot\mathrm{SPS}(1)=1.0, \\
\Theta_{1} & =(1+0.8)\cdot\mathrm{SPS}(2)=0.9.
\end{aligned}
$$

Without early-stopping, the scheduler proceeds to evaluate $\Theta_{2}$ before committing any admission decisions. Because the Markov confidence head uses the previously sampled token, the next confidence score $c_{2}$ explicitly depends on the realization of $x_{1}$. Consequently, the second-prefix survival probability

$$
a_{2}=a_{1}c_{2}
$$

also depends on $x_{1}$. Consider two possible realizations of $x_{1}$:

- **Case 1 ($x_{1}$ yields a high $c_{2}$):** Suppose $x_{1}$ results in $c_{2}=0.9$. Then

  $$
  a_{2}=0.8\times 0.9=0.72.
  $$

  The expected throughput for length $2$ is

  $$
  \Theta_{2}=(1+0.8+0.72)\times 0.45=1.134.
  $$

  Since $\Theta_{2}$ is the global maximum among $\{1.0,0.9,1.134\}$, the scheduler returns $\ell=2$. The first token $x_{1}$ is admitted into the verification prefix.
- **Case 2 ($x_{1}$ yields a low $c_{2}$):** Suppose $x_{1}$ results in $c_{2}=0$. Then

  $$
  a_{2}=0.
  $$

  The expected throughput for length $2$ is

  $$
  \Theta_{2}=(1+0.8+0)\times 0.45=0.81.
  $$

  Here, the global maximum remains $\Theta_{0}=1.0$, so the scheduler returns $\ell=0$. The first token $x_{1}$ is not admitted into the verification prefix.

Thus, the admission of the first draft token dynamically depends on the value of the first draft token itself. This retrospective dependence introduces selection bias: the scheduler favors tokens that lead to highly confident continuations, even though the admission decision for $x_{1}$ should have been made before observing $x_{1}$. We now make the distributional bias explicit. Let the vocabulary be $\{A,B\}$, and consider the target and draft distributions at the first position:

$$
p_{\mathrm{t}}(A)=0.7,\qquad p_{\mathrm{t}}(B)=0.3,
$$

$$
p_{\mathrm{d}}(A)=0.5,\qquad p_{\mathrm{d}}(B)=0.5.
$$

The standard speculative acceptance probability at the first position is

$$
\sum_{x\in\{A,B\}}\min\bigl(p_{\mathrm{t}}(x),p_{\mathrm{d}}(x)\bigr)=\min(0.7,0.5)+\min(0.3,0.5)=0.8,
$$

matching the assumed value ($a_{1}=0.8$). Suppose the retrospective scheduler behaves as above: $x_{1}=A$ yields a high continuation confidence and hence $\ell=2$, while $x_{1}=B$ yields a low continuation confidence and hence $\ell=0$. Then the first output token is distributed as follows. If $x_{1}=A$, the draft token is admitted and accepted with probability

$$
\min\left(1,\frac{p_{\mathrm{t}}(A)}{p_{\mathrm{d}}(A)}\right)=\min\left(1,\frac{0.7}{0.5}\right)=1,
$$

so the output token is $A$. If $x_{1}=B$, the draft token is not admitted; the target model instead generates a fresh token from $p_{\mathrm{t}}$. Therefore,

$$
\Pr(Y=A)=\Pr(x_{1}=A)\cdot 1+\Pr(x_{1}=B)\cdot p_{\mathrm{t}}(A)=0.5+0.5\times 0.7=0.85,
$$

and hence

$$
\Pr(Y=B)=0.15.
$$

This output distribution ($(0.85,0.15)$) differs from the target distribution ($(0.7,0.3)$), proving that the retrospective scheduler is not lossless. The early-stopping mechanism prevents this issue in the causal greedy scheduler. Since $\Theta_{1}<\Theta_{0}$, the scheduler halts immediately and returns $\ell=0$ before evaluating any continuation-dependent quantity such as $c_{2}$. The admission decision for the first position therefore depends only on pre-token information and cannot be biased by the realization of $x_{1}$. This restores the non-anticipating property required by the standard losslessness argument.

[+equal]: Equal contribution.

[+1]: We use the terms *anchor token* and *bonus token* interchangeably in this paper to denote the final token generated by the target model in the previous decoding round.

[+2]: In practical serving scenarios, average context lengths remain well below extremes (e.g., 1M tokens), making their impact on decode latency marginal for highly optimized architectures like DeepSeek-V4. Moreover, in prefill-decode disaggregated deployments, decode load balancers keep both request counts and total context lengths roughly balanced across data-parallel (DP) ranks. This effectively amortizes sequence-length variance, allowing us to safely assume that engine throughput depends predominantly on the verification batch size $B$.

[+3]: To facilitate future research, we release all [checkpoints](https://huggingface.co/collections/deepseek-ai/deepspec) we trained, including Eagle3, DFlash and DSpark.

[+4]: For clarity, unless otherwise stated, all reported metrics for accepted length and acceptance rate include the target-generated bonus token.

[+5]: [https://www.high-flyer.cn/en/blog/hai-llm/](https://www.high-flyer.cn/en/blog/hai-llm/)
