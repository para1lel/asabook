---
title: 'Speculative Decoding'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/speculative-decoding/
---

> [Yaniv Leviathan](https://yanivle.github.io/), [Matan Kalman](https://research.google/people/108191/), and [Yossi Matias](http://www.math.tau.ac.il/~matias/). First submitted to arXiv on November 30, 2022; current version v2. Published as an oral paper at ICML 2023, *Proceedings of Machine Learning Research* 202:19274-19286. [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192). [Original PDF](/paper/speculative-decoding.pdf). [DOI](https://doi.org/10.48550/arXiv.2211.17192). [TeX source](https://export.arxiv.org/e-print/2211.17192v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Inference from large autoregressive models like Transformers is slow - decoding $K$ tokens takes $K$ serial runs of the model. In this work we introduce *speculative decoding* - an algorithm to sample from autoregressive models faster *without any changes to the outputs*, by computing several tokens in parallel. At the heart of our approach lie the observations that (1) hard language-modeling tasks often include easier subtasks that can be approximated well by more efficient models, and (2) using speculative execution and a novel sampling method, we can make exact decoding from the large models faster, by running them in parallel on the outputs of the approximation models, potentially generating several tokens concurrently, and without changing the distribution. Our method can accelerate existing off-the-shelf models without retraining or architecture changes. We demonstrate it on T5-XXL and show a 2X-3X acceleration compared to the standard T5X implementation, with identical outputs.

<span id="figure-01"></span>

![Refer to caption](../../papers/speculative-decoding/figure-01.png)

**Figure 1.** Our technique illustrated in the case of unconditional language modeling. Each line represents one iteration of the algorithm. The green tokens are the suggestions made by the approximation model (here, a GPT-like Transformer decoder with 6M parameters trained on lm1b with 8k tokens) that the target model (here, a GPT-like Transformer decoder with 97M parameters in the same setting) accepted, while the red and blue tokens are the rejected suggestions and their corrections, respectively. For example, in the first line the target model was run only once, and 5 tokens were generated.

## 1 Introduction

Large autoregressive models, notably large Transformers [Vas17a], are much more capable than smaller models, as is evidenced countless times in recent years e.g., in the text or image domains, like GPT-3 [Bro20a], LaMDA [Tho22], Parti [Yu22], and PaLM [Cho22]. Unfortunately, a single decode step from these larger models is significantly slower than a step from their smaller counterparts, and making things worse, these steps are done serially - decoding $K$ tokens takes $K$ serial runs of the model.

Given the importance of large autoregressive models and specifically large Transformers, several approaches were developed to make inference from them faster. Some approaches aim to reduce the inference cost for *all* inputs equally [Hin15, Jas21, Hub16a, So21, Sha19a]. Other approaches stem from the observation that not all inference steps are born alike - some require a very large model, while others can be approximated well by more efficient models. These *adaptive computation* methods [Han21, Suk19a, Sch21a, Sca20a, Bap20, Elb19, Sch20] aim to use less compute resources for easier inference steps. While many of these solutions have proven extremely effective in practice, they usually require changing the model architecture, changing the training-procedure and re-training the models, and don’t maintain identical outputs.

The key observation above, that some inference steps are “harder” and some are “easier”, is also a key motivator for our work. We additionally observe that inference from large models is often not bottlenecked on arithmetic operations, but rather on memory bandwidth and communication, so additional computation resources might be available. Therefore we suggest increasing concurrency as a complementary approach to using an adaptive amount of computation. Specifically, we are able to accelerate inference without changing the model architectures, without changing the training-procedures or needing to re-train the models, and without changing the model output distribution. This is accomplished via *speculative execution*.

Speculative execution [Bur85, Hen12] is an optimization technique, common in processors, where a task is performed in parallel to verifying if it’s actually needed - the payoff being increased concurrency. A well-known example of speculative execution is branch prediction. For speculative execution to be effective, we need an efficient mechanism to suggest tasks to execute that are likely to be needed. In this work, we generalize speculative execution to the stochastic setting - where a task *might be* needed with some probability. Applying this to decoding from autoregressive models like Transformers, we sample generations from more efficient *approximation models* as speculative prefixes for the slower *target models*. With a novel sampling method, *speculative sampling*, we maximize the probability of these speculative tasks to be accepted, while guaranteeing that the outputs from our system have the same distribution as those from the target model alone. For example, the sentence in [Figure 1](#figure-01), consisting of 38 tokens, was generated by our method with only 9 serial runs of a larger target model (97M parameters) thanks to a smaller and more efficient approximation model (6M parameters), while the probability of generating it is unchanged.

We analyze our method in a variety of tasks and model sizes: unconditional generation from a 97M parameter GPT-like model trained on lm1b, English to German translation and news article summarization with an 11B parameters T5-XXL model, and a dialog task with a 137B parameter LaMDA model. We implement our method and compare actual walltimes for T5-XXL to those of the robust T5X implementation [Rob22], showing an out-of-the-box latency improvement of 2X-3X, without any change to the outputs ([Section 4](#section-04)).

Our method is easy to employ in actual production settings, doesn’t require training new models, and doesn’t change the outputs. Therefore, in common situations where memory bandwidth is the bottleneck, and compute resources are available, it may be a good default to accelerate sampling from autoregressive models like Transformers.

To summarize, our main contributions are: (1) A generalization of speculative execution to the stochastic setting, with a novel sampling method we call *speculative sampling*, and (2) A decoding mechanism we call *speculative decoding* that can accelerate decoding from autoregressive models, without any change to the model architectures, training regimes and output distributions.

## 2 Speculative Decoding

### 2.1 Overview

Let $M_{p}$ be the target model, inference from which we’re trying to accelerate, and $p(x_{t}|x_{<t})$ the distribution we get from the model for a prefix $x_{<t}$. Let $M_{q}$ be a more efficient approximation model for the same task, and denote by $q(x_{t}|x_{<t})$ the distribution we get from the model for a prefix $x_{<t}$ [+1]. The core idea is to (1) use the more efficient model $M_{q}$ to generate $\gamma\in\mathbb{Z}^{+}$ completions (see [Section 3.5](#section-03-05) for how to optimally choose this parameter), then (2) use the target model $M_{p}$ to evaluate all of the guesses and their respective probabilities from $M_{q}$ *in parallel*, accepting all those that *can* lead to an identical distribution, and (3) sampling an additional token from an adjusted distribution to fix the first one that was rejected, or to add an additional one if they are all accepted. That way, each parallel run of the target model $M_{p}$ will produce at least one new token (so the number of serial runs of the target model can never, even in the worst case, be larger than the simple autoregressive method), but it can potentially generate many new tokens, up to $\gamma+1$, depending on how well $M_{q}$ approximates $M_{p}$.

### 2.2 Standardized Sampling

<span id="section-02-02"></span>

First, note that while there are many methods and parameters of sampling, like argmax, top-k, nucleus, and setting a temperature, and popular implementations usually treat them differently at the logits level, they can all easily be cast into standard sampling from an adjusted probability distribution. For example, argmax sampling is equivalent to zeroing out non-max elements of the distribution and normalizing. We can therefore only deal with standard sampling from a probability distribution, and cast all of the other types of sampling into that framework. Going forward we’ll assume that $p(x)$ and $q(x)$ are the distributions from $M_{p}$ and $M_{q}$ respectively, adjusted for the sampling method.

### 2.3 Speculative Sampling

<span id="section-02-03"></span>

To sample $x\sim p(x)$, we instead sample $x\sim q(x)$, keeping it if $q(x)\leq p(x)$, and in case $q(x)>p(x)$ we reject the sample with probability $1-\frac{p(x)}{q(x)}$ and sample $x$ again from an adjusted distribution $p^{\prime}(x)=\mathrm{norm}(\max(0,p(x)-q(x)))$ instead. It’s easy to show (see [Section A.1](#section-a-01)) that for any distributions $p(x)$ and $q(x)$, and $x$ sampled in this way, indeed $x\sim p(x)$.

Given the distribution $q(x)$ obtained from running $M_{q}$ on a conditioning $\mathrm{prefix}$, we can sample a token $x_{1}\sim q(x)$. We then calculate the distribution $p(x)$ by running $M_{p}$ on $\mathrm{prefix}$ while in parallel speculatively calculating the distribution of the next token $x_{2}$ by running $M_{p}$ on $\mathrm{prefix}+[x_{1}]$. Once both computations complete, we proceed as per above: If $x_{1}$ is rejected, we discard the computation of $x_{2}$ and re-sample $x_{1}$ from an adjusted distribution, and if $x_{1}$ is accepted, we keep both tokens. [Algorithm 1](#algorithm-01) generalizes this idea to sample between 1 and $\gamma+1$ tokens at once.

<span id="algorithm-01"></span>

**Algorithm 1: SpeculativeDecodingStep.**

- **Input:** $M_{p},M_{q},\mathrm{prefix}$.
- **Sample** $\gamma$ guesses $x_{1,\ldots,\gamma}$ from $M_{q}$ autoregressively:
  - **For** $i=1$ to $\gamma$:
    - $q_{i}(x)\leftarrow M_{q}(\mathrm{prefix}+[x_{1},\ldots,x_{i-1}])$.
    - $x_{i}\sim q_{i}(x)$.
- **Run** $M_{p}$ in parallel:
  - $p_{1}(x),\ldots,p_{\gamma+1}(x)\leftarrow M_{p}(\mathrm{prefix}),\ldots,M_{p}(\mathrm{prefix}+[x_{1},\ldots,x_{\gamma}])$.
- **Determine** the number of accepted guesses $n$:
  - $r_{1}\sim U(0,1),\dots,r_{\gamma}\sim U(0,1)$.
  - $n\leftarrow\min(\{i-1\mid 1\leq i\leq\gamma,r_{i}>\frac{p_{i}(x)}{q_{i}(x)}\}\cup\{\gamma\})$.
- **Adjust** the distribution from $M_{p}$ if needed:
  - $p^{\prime}(x)\leftarrow p_{n+1}(x)$.
  - **If** $n<\gamma$:
    - $p^{\prime}(x)\leftarrow\mathrm{norm}(\max(0,p_{n+1}(x)-q_{n+1}(x)))$.
- **Return** one token from $M_{p}$ and $n$ tokens from $M_{q}$:
  - $t\sim p^{\prime}(x)$.
  - **Return:** $\mathrm{prefix}+[x_{1},\ldots,x_{n},t]$.

## 3 Analysis

### 3.1 Number of Generated Tokens

<span id="section-03-01"></span>

Let’s analyze the reduction factor in the number of serial calls to the target model, or equivalently, the expected number of tokens produced by a single run of [Algorithm 1](#algorithm-01).

<span id="definition-03-01"></span>

###### Definition 3.1.

The *acceptance rate $\beta_{x_{<t}}$*, given a prefix $x_{<t}$, is the probability of accepting $x_{t}\sim q(x_{t}|x_{<t})$ by speculative sampling, as per [Section 2.3](#section-02-03) [+2].

$E(\beta)$ is then a natural measure of how well $M_{q}$ approximates $M_{p}$. If we make the simplifying assumption that the $\beta$s are i.i.d., and denote $\alpha=E(\beta)$, then the number of tokens produced by a single run of [Algorithm 1](#algorithm-01) is a capped geometric variable, with success probability $1-\alpha$ and cap $\gamma+1$, and the expected number of tokens generated by [Algorithm 1](#algorithm-01) satisfies [Equation 1](#equation-01). See [Figure 2](#figure-02).

<span id="equation-01"></span>

$$
E(\#\ \mathrm{generated}\ \mathrm{tokens})=\frac{1-\alpha^{\gamma+1}}{1-\alpha}\tag{1}
$$

<span id="figure-02"></span>

![Refer to caption](../../papers/speculative-decoding/figure-02.png)

**Figure 2.** The expected number of tokens generated by [Algorithm 1](#algorithm-01) as a function of $\alpha$ for various values of $\gamma$.

### 3.2 Calculating $\alpha$

We’ll now derive a simple formula for calculating $\alpha$ given a prefix and the two models $M_{p}$ and $M_{q}$. We start by defining a natural divergence $D_{\mathrm{LK}}$:

###### Definition 3.2.

$D_{\mathrm{LK}}(p,q)=\sum_{x}|p(x)-M(x)|=\sum_{x}|q(x)-M(x)|$ where $M(x)=\frac{p(x)+q(x)}{2}$.

###### Lemma 3.3.

<span id="lemma-03-03"></span>

$D_{\mathrm{LK}}(p,q)=1-\sum_{x}\min(p(x),q(x))$

###### Proof.

$D_{\mathrm{LK}}(p,q)=\sum_{x}|p(x)-M(x)|=\sum_{x}\frac{|p-q|}{2}=1-\sum_{x}\frac{p+q-|p-q|}{2}=1-\sum_{x}\min(p(x),q(x))$ ∎

From [Lemma 3.3](#lemma-03-03) we immediately get the following results:

###### Corollary 3.4.

$$
\begin{aligned}
D_{\mathrm{LK}}(p,q)&\ \mathrm{is\ a\ symmetric\ divergence\ in}\ [0,1].\\
D_{\mathrm{LK}}(p,q)=0&\iff p=q.\\
D_{\mathrm{LK}}(p,q)=1&\iff p\ \mathrm{and}\ q\ \mathrm{have\ disjoint\ support}.
\end{aligned}
$$

###### Theorem 3.5.

<span id="theorem-03-05"></span>

$\beta=1-D_{\mathrm{LK}}(p,q)$

###### Proof.

$$
\beta=E_{x\sim q(x)}
\begin{cases}
1,&q(x)\leq p(x),\\
\dfrac{p(x)}{q(x)},&q(x)>p(x)
\end{cases}
=E_{x\sim q(x)}\min\left(1,\frac{p(x)}{q(x)}\right)=\sum_{x}\min(p(x),q(x))
$$

∎

Finally we get:

###### Corollary 3.6.

<span id="corollary-03-06"></span>

$\alpha=1-E(D_{\mathrm{LK}}(p,q))=E(\min(p,q))$

See [Table 3](#table-03) for empirically observed $\alpha$ values in our experiments.

### 3.3 Walltime Improvement

<span id="section-03-03"></span>

We’ve shown that with the i.i.d. assumption our algorithm reduces the number of calls to the target model by a factor of $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$. Note that speculative execution in general, and our algorithm in particular, assume that we have enough compute resources to support the increased concurrency ([Section 3.4](#section-03-04)). For the walltime anaylsis, we’ll assume that we can run $\gamma+1$ concurrent evaluations of $M_{p}$ in parallel without increasing the walltime. To get the total walltime improvement, we now consider the cost of running the approximation model $M_{q}$.

###### Definition 3.7.

Let $c$, the *cost coefficient*, be the ratio between the time for a single run of $M_{q}$ and the time for a single run of $M_{p}$.

Note that unlike $\alpha$ which is an intrinsic property of the models and the task, the value of $c$ depends on the hardware configuration and software implementation details. In our experiments where $M_{q}$ is typically a couple of orders of magnitude smaller than $M_{p}$, $c$ was always less than $0.05$ and often negligibly close to 0.

###### Theorem 3.8.

<span id="theorem-03-08"></span>

The expected improvement factor in total walltime by [Algorithm 1](#algorithm-01) is $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)({\gamma}c+1)}$.

###### Proof.

Denote the cost of running a single step of $M_{p}$ by $T$. Now, each run of [Algorithm 1](#algorithm-01) costs $Tc\gamma+T$ (for running the approximation model $M_{q}$ $\gamma$ times and running $M_{p}$ once) and according to [Equation 1](#equation-01) produces $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$ tokens on average. So the overall expected cost for producing a token with [Algorithm 1](#algorithm-01) is $\frac{(c\gamma+1)(1-\alpha)}{1-\alpha^{\gamma+1}}T$. Since the cost of producing a single token with the standard decoding algorithm is $T$, we get the desired result. ∎

Note that [Theorem 3.8](#theorem-03-08) assumes long enough generations (for example, since we run $M_{p}$ at least once, the improvement factor is capped by the number of generated tokens).

###### Corollary 3.9.

If $\alpha>c$, there exists $\gamma$ for which we’ll get an improvement, and the improvement factor will be at least $\frac{1+\alpha}{1+c}$.

###### Proof.

If we get an improvement for $\gamma$, we’d also get an improvement for any $0<\gamma^{*}<\gamma$, so for our method to yield an improvement, we can evaluate [Theorem 3.8](#theorem-03-08) for $\gamma=1$, yielding $\frac{1-\alpha^{2}}{(1-\alpha)(c+1)}=\frac{1+\alpha}{1+c}$. ∎

### 3.4 Number of Arithmetic Operations

<span id="section-03-04"></span>

[Algorithm 1](#algorithm-01) does $\gamma+1$ runs of $M_{p}$ in parallel, so the number of *concurrent* arithmetic operations grows by a factor of $\gamma+1$. Now, since [Algorithm 1](#algorithm-01) produces at most $\gamma+1$ tokens per run, the *total* number of arithmetic operations might be higher than that of the standard decoding algorithm. When we accept the sample from $M_{q}$ the increased concurrency is “free” and the total number of operations isn’t increased [+3]. When we reject a guess though, computation is wasted. Let’s now analyze the effect of our method on the total number of arithmetic operations.

###### Definition 3.10.

Let $\hat{c}$ be the ratio of arithmetic operations per token of the approximation model $M_{q}$ to that of the target model $M_{p}$.

###### Theorem 3.11.

The expected factor of increase in the number of total operations of [Algorithm 1](#algorithm-01) is $\frac{(1-\alpha)({\gamma}\hat{c}+\gamma+1)}{1-\alpha^{\gamma+1}}$.

###### Proof.

Denote by $\hat{T}$ the number of arithmetic operations done by a standard decoding baseline per token, i.e. the number of operations of a single run of $M_{p}$. Then a single iteration of [Algorithm 1](#algorithm-01) costs $\hat{T}\hat{c}\gamma+\hat{T}(\gamma+1)$ operations (for $\gamma$ runs of $M_{q}$ and $\gamma+1$ parallel runs of $M_{p}$). Dividing by the expected number of tokens produced by [Algorithm 1](#algorithm-01), i.e. [Equation 1](#equation-01), and by $\hat{T}$, we get the desired result. ∎

If $\alpha$ is low, the increase in the number of arithmetic operations is high, and vice-versa. Note that for Transformer decoders, the total number of arithmetic operations by [Algorithm 1](#algorithm-01) (not counting runs of $M_{q}$) *can be bounded from above by a single run of the same-size Transformer encoder*.

Unlike the total number of arithmetic operations, the total number of memory accesses can go down with our method. Specifically, the target model’s weights and KV cache can be read once per execution of [Algorithm 1](#algorithm-01), so the number of memory accesses for reading them shrinks by a factor of $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, according to [Equation 1](#equation-01).

<span id="figure-03"></span>

![Refer to caption](../../papers/speculative-decoding/figure-03.png)

**Figure 3.** The optimal $\gamma$ as a function of $\alpha$ for various values of $c$.

### 3.5 Choosing $\gamma$

<span id="section-03-05"></span>

Given $c$ and $\alpha$ and assuming enough compute resources (see [Section 3.4](#section-03-04)), the optimal $\gamma$ is the one maximizing the walltime improvement equation ([Theorem 3.8](#theorem-03-08)): $\frac{1-\alpha^{\gamma+1}}{(1-\alpha)({\gamma}c+1)}$. Since $\gamma$ is an integer, it can be easily found numerically, see [Figure 3](#figure-03).

[Table 1](#table-01) and [Figure 4](#figure-04) illustrate the trade-off between inference speed and the total number of arithmetic operations for various values of $\alpha$ and $\gamma$, assuming $c=\hat{c}=0$. [Figure 5](#figure-05) shows a simplified trace diagram.

<span id="table-01"></span>

![Original paper Table 1](../../papers/speculative-decoding/table-01.png)

**Table 1.** The total number of arithmetic operations and the inference speed vs the baseline, for various values of $\gamma$ and $\alpha$, assuming $c=\hat{c}=0$.

<span id="figure-04"></span>

![Refer to caption](../../papers/speculative-decoding/figure-04.png)

**Figure 4.** The speedup factor and the increase in number of arithmetic operations as a function of $\alpha$ for various values of $\gamma$.

<span id="figure-05"></span>

![Refer to caption](../../papers/speculative-decoding/figure-05.png)

**Figure 5.** A simplified trace diagram for a full encoder-decoder Transformer stack. The top row shows speculative decoding with $\gamma=7$ so each of the calls to $M_{p}$ (the purple blocks) is preceded by 7 calls to $M_{q}$ (the blue blocks). The yellow block on the left is the call to the encoder for $M_{p}$ and the orange block is the call to the encoder for $M_{q}$. Likewise the middle row shows speculative decoding with $\gamma=3$, and the bottom row shows standard decoding.

Instead of picking a single value for $\gamma$ based on $\alpha$, since the $\beta$s aren’t constant, we could get further improvement by predicting the value of $\beta$ and accordingly varying the value of $\gamma$ during the run of [Algorithm 1](#algorithm-01). To get an upper bound on the additional improvement factor, assume we had an oracle for $\gamma$. We would then have $E(\#\ \mathrm{generated}\ \mathrm{tokens})=\frac{1}{1-\alpha}$. For typical values of $c$ and $\alpha$, and assuming unbounded compute resources, the enhanced walltime improvement factor can be up to $\sim$60% higher than the improvement factor with a fixed $\gamma$. We leave exploring this for future work [+4].

### 3.6 Approximation Models

<span id="section-03-06"></span>

Speculative sampling, and therefore speculative decoding, guarantee an identical output distribution for any choice of approximation model $M_{q}$ without restriction (see [Section A.1](#section-a-01)). In our experiments, we mostly tested existing off-the-shelf smaller Transformers as the approximation models. Further, we only tested approximation models of the same architecture as the target models $M_{p}$ and using the same probability standardization. In this setup, choosing $M_{q}$ to be around two orders of magnitude smaller than $M_{p}$ usually performed best, balancing $\alpha$ and $c$ ([Theorem 3.8](#theorem-03-08)).

Another type of approximation models, *negligible-cost models*, are those for which $c\approx 0$, i.e. approximation models with a negligible cost relative to the target model. In this case, we get an expected walltime improvement of $\frac{1-\alpha^{\gamma+1}}{1-\alpha}$, which is bounded from above by $\frac{1}{1-\alpha}$ (we approach equality if $\gamma$ is large). One interesting type of negligible-cost approximation models are n-gram models, where the evaluation amounts to a table lookup. Interestingly, in empirical tests ([Section 4.2](#section-04-02)) we get non zero $\alpha$s even for these trivial n-gram models. For example, for the English-German translation task, with $M_{p}$ being T5-XXL 11B and $M_{q}$ being a trivial bigram model, we get $\alpha\approx 0.2$ which leads to an inference speed improvement factor of $1.25$X with $\gamma=3$.

Other simple heuristics can be used as negligible-cost approximation models. For example, in cases where long sequences are likely to repeat, such as for summarization tasks or chat-like interfaces [+5], an approximation model that simply copies tokens from the context in case we find a matching prefix, might yield high values of $\alpha$. These parameter-less approximation models, have the additional advantage of being even simpler to deploy from a production standpoint.

Another type of approximation models that can be used by speculative decoding are non-autoregressive models, like those from [Ste18]. Then, instead of the autogreressive loop in [Algorithm 1](#algorithm-01) we’d just call the non-autoregressive model once.

A final example, interesting mostly from a theoretical perspective, is an approximation model which chooses tokens at random, which guarantees some improvement (although very small) for all models $M_{p}$.

## 4 Experiments

<span id="section-04"></span>

### 4.1 Empirical Walltime Improvement

<span id="section-04-01"></span>

We implement our algorithm and compare it to the implementation in the T5X codebase for accelerating T5-XXL.

**Setup.** We test a standard encoder-decoder T5 version 1.1 model [Raf20] on two tasks from the T5 paper: (1) English to German translation fine tuned on WMT EnDe, and (2) Text summarization fine tuned on CCN/DM. For both tasks, we use T5-XXL (11B) for $M_{p}$. For the approximation model $M_{q}$ we test several existing configurations, namely T5-large (800M), T5-base (250M), and T5-small (77M) [Raf20]. We use existing checkpoints for all models. We measure walltime improvements with a batch size of 1 on a single TPU-v4 for both argmax sampling (temp=0) and standard sampling (temp=1).

**Results.** [Table 2](#table-02) shows the empirical results from our method. We see that T5-small (77M), with a good balance of $c$ and $\alpha$, provides the highest speedup out of the tested approximation models. As expected we see that $\alpha$ increases with the size of the approximation model. Interestingly, $\alpha$ and walltime improvement are higher for argmax sampling (temp=0). We observe speedups of 2.6X (temp=1) and 3.4X (temp=0) on the translation task and slightly lower speedups of 2.3X (temp=1) and 3.1X (temp=0) for the summarization task. These empirical results match well with the theoretical predictions, with some variance due to implementation details (see [Section A.3](#section-a-03)).

<span id="table-02"></span>

![Original paper Table 2](../../papers/speculative-decoding/table-02.png)

**Table 2.** Empirical results for speeding up inference from a T5-XXL 11B model.

### 4.2 Empirical $\alpha$ Values

<span id="section-04-02"></span>

While we only implemented our method for T5, we measured $\alpha$ values for various tasks, sampling methods, target models $M_{p}$, and approximation models $M_{q}$. Specifically, we evaluated the expectation from [Corollary 3.6](#corollary-03-06) on 10K tokens generated by $M_{p}$, for each of the settings below.

**GPT-like (97M params).** We test a decoder-only Transformer model on unconditional language generation, trained on lm1b [Che13a]. The model here is a GPT-like Transformer decoder with Gelu activations [Hen16]. For $M_{q}$ we experimented with a Transformer decoder model with 6M parameters: dim 256, dim feed-forward 1024, 2 layers, 4 attention heads, as well as simple unigram and bigram models. $M_{p}$ has 97M parameters: dim 768, dim feed-forward 3072, 12 layers, 12 attention heads. We used Bert tokenization [Dev19] with 8k tokens for all models.

**LaMDA (137B params).** We tested a decoder only LaMDA model on a dialog task [Tho22]. We used existing checkpoints from LaMDA 137B as $M_{p}$ and LaMDA 8B, LaMDA 2B, and LaMDA 100M for $M_{q}$.

See [Section 4.1](#section-04-01) for the setup of the T5-XXL (11B params) model.

[Table 3](#table-03) summarizes the $\alpha$ values for the tested cases. We observe that approximation models that are a couple of orders of magnitude smaller than the target model tend to produce $\alpha$ values between 0.5 and 0.9. Interestingly, we also note that for all models, the sharper the adjusted distribution, the higher the $\alpha$ values. Finally, we note that even trivial unigram and bigram approximations yield non negligible $\alpha$ values. For example, for the case of English to German translation, the bigram model has an $\alpha$ value of 0.2, and since $c=0$ in this case, yields a 1.25X speed improvement, which is surprisingly high for this trivial approximation model (but is still lower than the speedup we get from using T5-small as the approximation model).

<span id="table-03"></span>

![Original paper Table 3](../../papers/speculative-decoding/table-03.png)

**Table 3.** Empirical $\alpha$ values for various target models $M_{p}$, approximation models $M_{q}$, and sampling settings. T=0 and T=1 denote argmax and standard sampling respectively [+6].

## 5 Related work

The efficiency of inference from large models was studied extensively [Deh21]. Many approaches aim to speed up inference from large models in general, and autoregressive models like Transformers in particular. Numerous techniques try to make inference more efficient for all tokens, e.g. distillation [Hin15], sparcification [Jas21], quantization [Hub16a], and architecture modification [So21, Sha19a]. Closer to our approach are adaptive computation methods which adapt the amount of computation to problem difficulty [Han21]. Examples include attending to a subset of the inputs [Suk19a], and early exits [Sch21a, Sca20a, Bap20, Elb19, Sch20]. Notably, Wisdom of Committees [Sch20] leverages off-the-shelf smaller models, but is an adaptive computation approach, and so it uses a heuristic to determine when to stop, losing the guarantee of identical outputs to those of the target models. In general, adaptive computation methods usually learn, either within the model itself or with an auxiliary model, when a computation shortcut can be taken. Usually, these methods save on both inference time and arithmetic operations, but require a change of architecture, a change of training procedure and training custom models or re-training of existing models. They usually also change the outputs of the model. We note that while many of the methods above improve the memory to arithmetic-operations ratio, in cases where the ratio remains high, these methods and our speculative decoding method might be effective in tandem.

Two prior methods leverage speculative execution for speeding up decoding from autoregressive models. Blockwise Parallel Decoding [Ste18] decodes several tokens in parallel, similarly to our work. However, it only supports greedy decoding (temperature=0) and not the general stochastic setting, it requires additional training of a custom model, and focuses on preserving down-stream task quality, instead of guaranteeing identical outputs. Shallow Aggressive Decoding (SAD) [Sun21] also decodes several tokens in parallel, similarly to our work. Unlike our work, SAD only supports copying the input to the output, and not general approximation models, making it only suitable for the cases where the inputs and outputs are very similar like grammatical error correction. In addition, similarly to Blockwise Parallel Decoding, SAD does not support the general stochastic sampling setting.

After we initially published our work, an independent implementation of speculative decoding [Che23] showed similar 2X-2.5X improvements on Chinchilla 70B.

## 6 Discussion

We presented *speculative sampling* which enables efficient *stochastic speculative execution* - i.e. speculative execution in the stochastic setting. We analyzed its impact on decoding from autoregressive models like Transformers via *speculative decoding* and have shown that given enough compute resources, we get meaningful 2X-3X speedups in practice vs T5X, a popular optimized implementation.

One limitation of speculative execution in general, and of speculative decoding in particular, is that latency is improved through increased concurrency at the cost of an increased number of arithmetic operations. Thus, our method is not helpful for configurations where additional computation resources are not available. However, in common cases where additional computation resources are available (e.g. when memory bandwidth is the bottleneck) our method provides the speedup with significant benefits: the model architecture doesn’t change, retraining isn’t required, and most importantly, *the output distribution is guaranteed to stay the same*. Our method is easy to implement, and can be used to speedup inference using out-of-the-box models without developing and evaluating custom schemes.

There are several directions for follow up research, importantly, further investigating the compatibility of speculative decoding with beam search (see [Section A.4](#section-a-04)). Also, while our method yields substantial speedups with existing off-the-shelf approximation models, greater improvements might be obtained via custom approximation models ([Section 3.6](#section-03-06)), such as those with custom architectures (e.g. custom sizes, non-autoregressive models, or various heuristics) or with custom training procedures (e.g. standard distillation with soft targets from $M_{p}$, or optimizing $M_{q}$ for $\alpha$ directly). It could also be interesting to explore a hierarchical version of the algorithm, where the approximation model is itself accelerated by an even faster model, which could allow for more capable approximation models. In this work we fixed the approximation model and the number of guesses $\gamma$ throughout inference, but varying them during inference could yield additional improvements ([Section 3.5](#section-03-05)). In our experiments we always performed the same standardization on the distributions generated by the approximation model as the desired one for the target model ([Section 2.2](#section-02-02)), but further improvements might be obtained by applying different transformations. We tested speculative decoding only in the text modality, but it might work well in other domains (e.g. images) which would be interesting to experiment with.

Finally, we note that *stochastic speculative execution* and *speculative sampling* can be helpful outside the scope of *speculative decoding* from autoregressive models. For example, given two slow functions, $f(x)$ and $g(y)$ such that $f(x)$ generates a distribution from which $g$’s input is sampled, we could use our method to run $f$ and $g$ in parallel. This setup might arise e.g. in physics simulations, or in reinforcement learning where $f$ is a large model that produces a distribution on actions, and $g$ is the world simulation, which would be interesting to explore.

## Acknowledgments

We would like to extend a special thank you to YaGuang Li for help with everything LaMDA related and for calculating the LaMDA figures in the paper, and to Blake Hechtman for great insights and help with XLA. We would also like to thank the reviewers for insightful comments, as well as Asaf Aharoni, Reiner Pope, Sasha Goldshtein, Nadav Sherman, Eyal Segalis, Eyal Molad, Dani Valevski, Daniel Wasserman, Valerie Nygaard, Danny Vainstein, the LaMDA and Theta Labs teams at Google, and our families.

## Appendix A

### A.1 Correctness of Speculative Sampling

<span id="section-a-01"></span>

We will now show that for any distributions $p(x)$ and $q(x)$, the tokens sampled via *speculative sampling* from $p(x)$ and $q(x)$ are distributed identically to those sampled from $p(x)$ alone. Let $\beta$ be the acceptance probability ([Definition 3.1](#definition-03-01)).

Note that as $p^{\prime}(x)=\mathrm{norm}(\max(0,p(x)-q(x)))=\frac{p(x)-\min(q(x),p(x))}{\sum_{x^{\prime}}(p(x^{\prime})-\min(q(x^{\prime}),p(x^{\prime})))}=\frac{p(x)-\min(q(x),p(x))}{1-\beta}$, the normalizing constant for the adjusted distribution $p^{\prime}(x)$ is $1-\beta$, where the last equation follows immediately from [Lemma 3.3](#lemma-03-03) and [Theorem 3.5](#theorem-03-05).

Now:

$$
P(x=x^{\prime})=P(\mathrm{guess}\ \mathrm{accepted},x=x^{\prime})+P(\mathrm{guess}\ \mathrm{rejected},x=x^{\prime})
$$

Where:

$$
P(\mathrm{guess}\ \mathrm{accepted},x=x^{\prime})=q(x^{\prime})\min(1,\frac{p(x^{\prime})}{q(x^{\prime})})=\min(q(x^{\prime}),p(x^{\prime}))
$$

And:

$$
P(\mathrm{guess}\ \mathrm{rejected},x=x^{\prime})=(1-\beta)p^{\prime}(x^{\prime})=p(x^{\prime})-\min(q(x^{\prime}),p(x^{\prime}))
$$

Overall:

$$
P(x=x^{\prime})=\min(p(x^{\prime}),q(x^{\prime}))+p(x^{\prime})-\min(p(x^{\prime}),q(x^{\prime}))=p(x^{\prime}).
$$

As desired. $\square$

### A.2 Speculative Sampling vs. Rejection Sampling

<span id="section-a-02"></span>

Rejection sampling is the following iterative sampling procedure that looks superficially similar to ours:

1. Sample $x\sim q(x)$ and $r\sim U(0,1)$.
2. If $r<\frac{p(x)}{M q(x)}$ return $x$.
3. Go to 1.

Where $M=\max_{x}\frac{p(x)}{q(x)}$. We could employ a non-iterative version of rejection sampling instead of speculative sampling - specifically go through steps 1 and 2 above, and otherwise sample from an *unmodified* $p(x)$ directly. That would be much less efficient than our method though. Specifically, the expected accept probability here is $E_{x\sim q(x)}\frac{p(x)}{M q(x)}=\sum_{x}p(x)\min_{x^{\prime}}\frac{q(x^{\prime})}{p(x^{\prime})}\leq\sum_{x}p(x)\min(1,\frac{q(x)}{p(x)})=\sum_{x}\min(p(x),q(x))=\alpha$ is (potentially much) lower than the expected accept probability in our method $\alpha$.

### A.3 Theoretical Predictions vs. Empirical Runtimes

<span id="section-a-03"></span>

[Table 4](#table-04) compares the expected runtime improvements based on [Theorem 3.8](#theorem-03-08) to the empirically measured runtimes from [Table 2](#table-02). We estimated the values of $c$ for the various models based on profiler traces. We can see that the theoretical predictions mostly match the measured runtimes. The larger differences are due to: (1) optimization differences between our implementation and the baseline, and (2) the simplifying assumption that the $\beta$s are i.i.d. being only an approximation (see [Section 3.1](#section-03-01)).

<span id="table-04"></span>

![Original paper Table 4](../../papers/speculative-decoding/table-04.png)

**Table 4.** Expected improvement factor (Exp) vs. empirically measured improvement factor (Emp).

### A.4 Application to Beam Search

<span id="section-a-04"></span>

Our method can be applied, with some performance penalty, to beam search sampling. Given the original beam width $w$, we can perform beam search with the approximation model $M_{q}$ and beam width $u\geq w$ for $\gamma$ steps. Then, we can use $M_{p}$ to check all of the candidates in parallel (costing a compute budget of $(w+u\gamma)$ runs of $M_{p}$). Finally, for each step, we can accept the guesses of $M_{q}$ as long as $\mathrm{top}_{w}(M_{p})\subseteq \mathrm{top}_{u}(M_{q})$ to get identical results to regular beam search with $M_{p}$ alone (with a more elaborate procedure we could also accept cases where the candidates we got happen to have higher probabilities than those of $M_{p}$ alone). The analysis of our method in this setting is more involved and we leave it for future work.

### A.5 Lenience

A strong property of [Algorithm 1](#algorithm-01) is that the output distribution is guaranteed to remain unchanged. That said, if we’re willing to allow some changes, with nice guarantees, we can get further inference speed improvements. To further motivate this, note that when we train two models with identical architectures and sizes on the same dataset, the generated probability distributions will not be identical, so some lenience might make sense. Note that the results in this paper except for this section use the strictest version of [Algorithm 1](#algorithm-01) and don’t allow lenience of any kind.

We could include a lenience parameter $l\in[0,1]$ and multiply $q(x)$ by $l$ before comparing with $p(x)$ in [Algorithm 1](#algorithm-01). This still maintains the nice guarantee that no token can be sampled with probability greater than $\frac{p(x)}{l}$. This means for example, that with $l=\frac{1}{10}$ no token can be sampled with more than $10$X its ground truth probability, so we can guarantee that extremely rare tokens will remain extremely rare (there is no guarantee on the minimum probability, so lenience could hurt the diversity of the samples).

Specifically, with a lenience factor $l$ we have

$$
\begin{aligned}
\alpha
&=E_{x\sim q(x)}
\begin{cases}
1,&lq(x)\leq p(x),\\
\dfrac{p(x)}{lq(x)},&lq(x)>p(x)
\end{cases}\\
&=E_{x\sim q(x)}\frac{p(x)}{\max(p(x),lq(x))}
=\sum_{x}\frac{p(x)q(x)}{\max(p(x),lq(x))}\\
&=\frac{1}{l}\sum_{x}\min(p(x),lq(x))
=\sum_{x}\min\left(\frac{p(x)}{l},q(x)\right).
\end{aligned}
$$

[Table 5](#table-05) shows $\alpha$ values for different values of $l$ when $M_{p}$ is T5-XXL (11B) and $M_{q}$ is T5-small (77M). With $c=0.015$, using lenience values of 1, 0.5, 0.3, and 0.1 (meaning that no token can be sampled with probability greater than 1X, 2X, 3X and 10X of the ground truth) we get improvement factors of 2.5X, 3.1X, 3.6X, and 5X respectively.

<span id="table-05"></span>

![Original paper Table 5](../../papers/speculative-decoding/table-05.png)

**Table 5.** $\alpha$ values for various values of $l$ with standard sampling where $M_{p}$ is T5-XXL (11B) on the EnDe translation task.

Note that when using temperature = 0 (i.e. argmax sampling), we can no longer use lenience as above. Instead, we could allow some lenience before standardizing the distributions. For example, we could accept the token $x$ sampled from $M_{q}$ in case $p(x)\leq l\cdot \max(p)$. In this case, we measure similar empirical increases in $\alpha$ values to those with temperature = 1. For example, when using lenience values of 1, 0.5, 0.3, and 0.1 for $M_{p}$ T5-XXL $M_{q}$ T5-small for English-German translation, we get $\alpha$ values of 0.75, 0.75, 0.8, 0.87. Taking for example $c=0.015$ and $\gamma=8$ we get speed improvement factors of 3.3X, 3.3X, 3.9X, and 4.9X respectively [+7].

[+1]: We’ll use $p(x)$ to denote $p(x_{t}|x_{<t})$ whenever the prefix $x_{<t}$ is clear from the context, and similarly for $q(x)$.

[+2]: As before, we’ll omit the $x_{<t}$ subscript whenever the prefix is clear from the context.

[+3]: Neglecting the cost of $M_{q}$.

[+4]: The above bound assumes that we still run $M_{p}$ to verify the oracle’s predictions. If we skip those verifications the bound doesn’t hold and we would get a substantial additional improvement.

[+5]: E.g. where a user and a language model iterate on content, like text or code (“can you rewrite this story but change the ending”, “can you make this function also do X”).

[+6]: Note that the outputs from the LaMDA model always go through a $\mathrm{Top}_{40}$ filter. This has no effect on argmax, but does have some effect on standard sampling.

[+7]: In this case, unlike in the standard sampling case shown in [Table 5](#table-05), a lenience factor of 0.5 doesn’t improve the speed-up.
