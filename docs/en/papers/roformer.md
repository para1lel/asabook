---
title: 'RoFormer: Rotary Position Embedding'
createTime: 2026/09/05 04:21:34
permalink: /en/papers/roformer/
---

> [Jianlin Su](https://spaces.ac.cn/), [Yu Lu](https://dblp.org/pid/09/2321.html), [Shengfeng Pan](https://dblp.org/pid/249/7590.html), [Ahmed Murtadha](https://dblp.org/pid/208/0019.html), [Bo Wen](https://dblp.org/pid/00/2490.html), and [Yunfeng Liu](https://dblp.org/pid/56/5650.html). First submitted to arXiv on April 20, 2021; current version v5. Published in *Neurocomputing* 568 (2024), Article 127063. [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5). [Original PDF](/paper/roformer.pdf). [DOI](https://doi.org/10.1016/j.neucom.2023.127063). [TeX source](https://export.arxiv.org/e-print/2104.09864v5). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Position encoding recently has shown effective in the transformer architecture. It enables valuable supervision for dependency modeling between elements at different positions of the sequence. In this paper, we first investigate various methods to integrate positional information into the learning process of transformer-based language models. Then, we propose a novel method named Rotary Position Embedding(RoPE) to effectively leverage the positional information. Specifically, the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Notably, RoPE enables valuable properties, including the flexibility of sequence length, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Finally, we evaluate the enhanced transformer with rotary position embedding, also called RoFormer, on various long text classification benchmark datasets. Our experiments show that it consistently overcomes its alternatives. Furthermore, we provide a theoretical analysis to explain some experimental results. RoFormer is already integrated into Huggingface: [https://huggingface.co/docs/transformers/model_doc/roformer](https://huggingface.co/docs/transformers/model_doc/roformer).

**Keywords:** Pre-trained Language Models; Position Information Encoding; Pre-training; Natural Language Processing.

<span id="section-1"></span>

## 1 Introduction

The sequential order of words is of great value to natural language understanding. Recurrent neural networks (RRNs) based models encode tokens’ order by recursively computing a hidden state along the time dimension. Convolution neural networks (CNNs) based models (CNNs) [Geh17] were typically considered position-agnostic, but recent work [Isl20] has shown that the commonly used padding operation can implicitly learn position information. Recently, the pre-trained language models (PLMs), which were built upon the transformer [Vas17], have achieved the state-of-the-art performance of various natural language processing (NLP) tasks, including context representation learning [Dev19], machine translation [Vas17], and language modeling [Rad19], to name a few. Unlike, RRNs and CNNs-based models, PLMs utilize the self-attention mechanism to semantically capture the contextual representation of a given corpus. As a consequence, PLMs achieve a significant improvement in terms of parallelization over RNNs and improve the modeling ability of longer intra-token relations compared to CNNs [+1].

It is noteworthy that the self-attention architecture of the current PLMs has shown to be position-agnostic [Yun20]. Following this claim, various approaches have been proposed to encode the position information into the learning process. On one side, generated absolute position encoding through a pre-defined function [Vas17] was added to the contextual representations, while a trainable absolute position encoding [Geh17, Dev19, Lan20, Cla20, Rad19, Rad18]. On the other side, the previous work [Par16, Sha18d, Hua18a, Dai19, Yan19, Raf20, Ke20, He20, Hua20a] focuses on relative position encoding, which typically encodes the relative position information into the attention mechanism. In addition to these approaches, the authors of [Liu20] have proposed to model the dependency of position encoding from the perspective of Neural ODE [Che18g], and the authors of [Wan20f] have proposed to model the position information in complex space. Despite the effectiveness of these approaches, they commonly add the position information to the context representation and thus render them unsuitable for the linear self-attention architecture.

In this paper, we introduce a novel method, namely Rotary Position Embedding(RoPE), to leverage the positional information into the learning process of PLMS. Specifically, RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation. Note that the proposed RoPE is prioritized over the existing methods through valuable properties, including the sequence length flexibility, decaying inter-token dependency with increasing relative distances, and the capability of equipping the linear self-attention with relative position encoding. Experimental results on various long text classification benchmark datasets show that the enhanced transformer with rotary position embedding, namely RoFormer, can give better performance compared to baseline alternatives and thus demonstrates the efficacy of the proposed RoPE.

In brief, our contributions are three-folds as follows:

- We investigated the existing approaches to the relative position encoding and found that they are mostly built based on the idea of the decomposition of adding position encoding to the context representations. We introduce a novel method, namely Rotary Position Embedding(RoPE), to leverage the positional information into the learning process of PLMS. The key idea is to encode relative position by multiplying the context representations with a rotation matrix with a clear theoretical interpretation.
- We study the properties of RoPE and show that it decays with the relative distance increased, which is desired for natural language encoding. We kindly argue that previous relative position encoding-based approaches are not compatible with linear self-attention.
- We evaluate the proposed RoFormer on various long text benchmark datasets. Our experiments show that it consistently achieves better performance compared to its alternatives. Some experiments with pre-trained language models are available on GitHub: [https://github.com/ZhuiyiTechnology/roformer](https://github.com/ZhuiyiTechnology/roformer).

The remaining of the paper is organized as follows. We establish a formal description of the position encoding problem in self-attention architecture and revisit previous works in [Section 2](#section-2). We then describe the rotary position encoding (RoPE) and study its properties in [Section 3](#section-3). We report experiments in [Section 4](#section-4). Finally, we conclude this paper in [Section 5](#section-5).

<span id="section-2"></span>

## 2 Background and Related Work

<span id="section-2-1"></span>

### 2.1 Preliminary

Let $\mathbb{S}_{N}=\{w_{i}\}_{i=1}^{N}$ be a sequence of $N$ input tokens with $w_{i}$ being the $i^{th}$ element. The corresponding word embedding of $\mathbb{S}_{N}$ is denoted as $\mathbb{E}_{N}=\{{\boldsymbol{x}}_{i}\}_{i=1}^{N}$, where ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$ is the d-dimensional word embedding vector of token $w_{i}$ without position information. The self-attention first incorporates position information to the word embeddings and transforms them into queries, keys, and value representations.

<span id="equation-01"></span>

$$
\begin{aligned}
{\boldsymbol{q}}_{m} & =f_{q}({\boldsymbol{x}}_{m},m) \\
{\boldsymbol{k}}_{n} & =f_{k}({\boldsymbol{x}}_{n},n) \\
{\boldsymbol{v}}_{n} & =f_{v}({\boldsymbol{x}}_{n},n),
\end{aligned}
$$

where ${\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}$ and ${\boldsymbol{v}}_{n}$ incorporate the $m^{th}$ and $n^{th}$ positions through $f_{q},f_{k}$ and $f_{v}$, respectively. The query and key values are then used to compute the attention weights, while the output is computed as the weighted sum over the value representation.

<span id="equation-02"></span>

$$
\begin{aligned}
a_{m,n} & =\frac{\exp(\frac{{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}}{\sqrt{d}})}{\sum_{j=1}^{N}\exp(\frac{{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{j}}{\sqrt{d}})} \\
\mathbf{o}_{m} & =\sum_{n=1}^{N}a_{m,n}{\boldsymbol{v}}_{n}
\end{aligned}
$$

The existing approaches of transformer-based position encoding mainly focus on choosing a suitable function to form [Equation 1](#equation-01).

<span id="section-2-2"></span>

### 2.2 Absolute position embedding

A typical choice of [Equation 1](#equation-01) is

<span id="equation-03"></span>

$$
f_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i},i):={\boldsymbol{W}}_{t:t\in\{q,k,v\}}({\boldsymbol{x}}_{i}+{\boldsymbol{p}}_{i}),
$$

where ${\boldsymbol{p}}_{i}\in\mathbb{R}^{d}$ is a d-dimensional vector depending of the position of token ${\boldsymbol{x}}_{i}$. Previous work [Dev19, Lan20, Cla20, Rad19, Rad18] introduced the use of a set of trainable vectors ${\boldsymbol{p}}_{i}\in\{{\boldsymbol{p}}_{t}\}_{t=1}^{L}$, where $L$ is the maximum sequence length. The authors of [Vas17] have proposed to generate ${\boldsymbol{p}}_{i}$ using the sinusoidal function.

<span id="equation-04"></span>

$$
\begin{cases}{\boldsymbol{p}}_{i,2t}&=\sin(k/10000^{2t/d})\\
{\boldsymbol{p}}_{i,2t+1}&=\cos(k/10000^{2t/d})\end{cases}
$$

in which ${\boldsymbol{p}}_{i,2t}$ is the $2t^{th}$ element of the d-dimensional vector ${\boldsymbol{p}}_{i}$. In the next section, we show that our proposed RoPE is related to this intuition from the sinusoidal function perspective. However, instead of directly adding the position to the context representation, RoPE proposes to incorporate the relative position information by multiplying with the sinusoidal functions.

<span id="section-2-3"></span>

### 2.3 Relative position embedding

The authors of [Sha18d] applied different settings of [Equation 1](#equation-01) as following:

<span id="equation-05"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m}):={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m} \\
f_{k}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{k}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{k}_{r}) \\
f_{v}({\boldsymbol{x}}_{n},n):={\boldsymbol{W}}_{v}({\boldsymbol{x}}_{n}+\tilde{{\boldsymbol{p}}}^{v}_{r})
\end{aligned}
$$

where $\tilde{{\boldsymbol{p}}}^{k}_{r},\tilde{{\boldsymbol{p}}}^{v}_{r}\in\mathbb{R}^{d}$ are trainable relative position embeddings. Note that $r=\mathrm{clip}(m-n,r_{\min},r_{\max})$ represents the relative distance between position $m$ and $n$. They clipped the relative distance with the hypothesis that precise relative position information is not useful beyond a certain distance. Keeping the form of [Equation 3](#equation-03), the authors [Dai19] have proposed to decompose ${\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}$ of [Equation 2](#equation-02) as

<span id="equation-06"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n}+{\boldsymbol{p}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{p}}_{n},
$$

the key idea is to replace the absolute position embedding ${\boldsymbol{p}}_{n}$ with its sinusoid-encoded relative counterpart $\tilde{{\boldsymbol{p}}}_{m-n}$, while the absolute position ${\boldsymbol{p}}_{m}$ in the third and fourth term with two trainable vectors $\mathbf{u}$ and $\mathbf{v}$ independent of the query positions. Further, ${\boldsymbol{W}}_{k}$ is distinguished for the content-based and location-based key vectors ${\boldsymbol{x}}_{n}$ and ${\boldsymbol{p}}_{n}$, denoted as ${\boldsymbol{W}}_{k}$ and $\widetilde{{\boldsymbol{W}}}_{k}$, resulting in:

<span id="equation-07"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\mathbf{u}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+\mathbf{v}^\top{\boldsymbol{W}}_{q}^\top\widetilde{{\boldsymbol{W}}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}
$$

It is noteworthy that the position information in the value term is removed by setting $f_{v}({\boldsymbol{x}}_{j}):={\boldsymbol{W}}_{v}{\boldsymbol{x}}_{j}$. Later work [Raf20, He20, Ke20, Hua20a] followed these settings by only encoding the relative position information into the attention weights. However, the authors of [Raf20] reformed [Equation 6](#equation-06) as:

<span id="equation-08"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+b_{i,j}
$$

where $b_{i,j}$ is a trainable bias. The authors of [Ke20] investigated the middle two terms of [Equation 6](#equation-06) and found little correlations between absolute positions and words. The authors of [Raf20] proposed to model a pair of words or positions using different projection matrices.

<span id="equation-09"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{p}}_{m}^\top\mathbf{U}_{q}^\top\mathbf{U}_{k}{\boldsymbol{p}}_{n}+b_{i,j}
$$

The authors of [He20] argued that the relative positions of two tokens could only be fully modeled using the middle two terms of [Equation 6](#equation-06). As a consequence, the absolute position embeddings ${\boldsymbol{p}}_{m}$ and ${\boldsymbol{p}}_{n}$ were simply replaced with the relative position embeddings $\tilde{{\boldsymbol{p}}}_{m-n}$:

<span id="equation-10"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}={\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}+{\boldsymbol{x}}_{m}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}\tilde{{\boldsymbol{p}}}_{m-n}+\tilde{{\boldsymbol{p}}}_{m-n}^\top{\boldsymbol{W}}_{q}^\top{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$

A comparison of the four variants of the relative position embeddings [Rad18] has shown that the variant similar to [Equation 10](#equation-10) is the most efficient among the other three. Generally speaking, all these approaches attempt to modify [Equation 6](#equation-06) based on the decomposition of [Equation 3](#equation-03) under the self-attention settings in [Equation 2](#equation-02), which was originally proposed in [Vas17]. They commonly introduced to directly add the position information to the context representations. Unlikely, our approach aims to derive the relative position encoding from [Equation 1](#equation-01) under some constraints. Next, we show that the derived approach is more interpretable by incorporating relative position information with the rotation of context representations.

<span id="section-3"></span>

## 3 Proposed approach

In this section, we discuss the proposed rotary position embedding (RoPE). We first formulate the relative position encoding problem in [Section 3.1](#section-3-1), we then derive the RoPE in [Section 3.2](#section-3-2) and investigate its properties in [Section 3.3](#section-3-3).

<span id="section-3-1"></span>

### 3.1 Formulation

Transformer-based language modeling usually leverages the position information of individual tokens through a self-attention mechanism. As can be observed in [Equation 2](#equation-02), ${\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}$ typically enables knowledge conveyance between tokens at different positions. In order to incorporate relative position information, we require the inner product of query ${\boldsymbol{q}}_{m}$ and key ${\boldsymbol{k}}_{n}$ to be formulated by a function $g$, which takes only the word embeddings ${\boldsymbol{x}}_{m}$, ${\boldsymbol{x}}_{n}$, and their relative position $m-n$ as input variables. In other words, we hope that the inner product encodes position information only in the relative form:

<span id="equation-11"></span>

$$
\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n).
$$

The ultimate goal is to find an equivalent encoding mechanism to solve the functions $f_{q}({\boldsymbol{x}}_{m},m)$ and $f_{k}({\boldsymbol{x}}_{n},n)$ to conform the aforementioned relation.

<span id="section-3-2"></span>

### 3.2 Rotary position embedding

<span id="section-3-2-1"></span>

#### 3.2.1 A 2D case

We begin with a simple case with a dimension $d=2$. Under these settings, we make use of the geometric property of vectors on a 2D plane and its complex form to prove (refer [Section 3.4.1](#section-3-4-1) for more details) that a solution to our formulation [Equation 11](#equation-11) is:

<span id="equation-12"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m},m) & =({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{i m\theta} \\
f_{k}({\boldsymbol{x}}_{n},n) & =({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{i n\theta} \\
g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},m-n) & =\operatorname{Re}[({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}e^{i(m-n)\theta}]
\end{aligned}
$$

where $\operatorname{Re}[\cdot]$ is the real part of a complex number and $({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})^{*}$ represents the conjugate complex number of $({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})$. $\theta\in\mathbb{R}$ is a preset non-zero constant. We can further write $f_{\{q,k\}}$ in a multiplication matrix:

<span id="equation-13"></span>

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)=\left(\begin{array}{cc}\cos{m\theta}&-\sin{m\theta}\\
\sin{m\theta}&\cos{m\theta}\end{array}\right)\left(\begin{array}{cc}W^{(11)}_{\{q,k\}}&W^{(12)}_{\{q,k\}}\\
W^{(21)}_{\{q,k\}}&W^{(22)}_{\{q,k\}}\end{array}\right)\left(\begin{array}{cc}x^{(1)}_{m}\\
x^{(2)}_{m}\end{array}\right)
$$

where $(x^{(1)}_{m},x^{(2)}_{m})$ is ${\boldsymbol{x}}_{m}$ expressed in the 2D coordinates. Similarly, $g$ can be viewed as a matrix and thus enables the solution of formulation in [Section 3.1](#section-3-1) under the 2D case. Specifically, incorporating the relative position embedding is straightforward: simply rotate the affine-transformed word embedding vector by amount of angle multiples of its position index and thus interprets the intuition behind *Rotary Position Embedding*.

<span id="section-3-2-2"></span>

#### 3.2.2 General form

In order to generalize our results in 2D to any ${\boldsymbol{x}}_{i}\in\mathbb{R}^{d}$ where $d$ is even, we divide the d-dimension space into $d/2$ sub-spaces and combine them in the merit of the linearity of the inner product, turning $f_{\{q,k\}}$ into:

<span id="equation-14"></span>

$$
f_{\{q,k\}}({\boldsymbol{x}}_{m},m)={\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{\{q,k\}}{\boldsymbol{x}}_{m}
$$

where

<span id="equation-15"></span>

$$
{\boldsymbol{R}}^{d}_{\Theta,m}=\begin{pmatrix}\cos{m\theta_{1}}&-\sin{m\theta_{1}}&0&0&\cdots&0&0\\
\sin{m\theta_{1}}&\cos{m\theta_{1}}&0&0&\cdots&0&0\\
0&0&\cos{m\theta_{2}}&-\sin{m\theta_{2}}&\cdots&0&0\\
0&0&\sin{m\theta_{2}}&\cos{m\theta_{2}}&\cdots&0&0\\
\vdots&\vdots&\vdots&\vdots&\ddots&\vdots&\vdots\\
0&0&0&0&\cdots&\cos{m\theta_{d/2}}&-\sin{m\theta_{d/2}}\\
0&0&0&0&\cdots&\sin{m\theta_{d/2}}&\cos{m\theta_{d/2}}\end{pmatrix}
$$

is the rotary matrix with pre-defined parameters $\Theta=\{\theta_{i}=10000^{-2(i-1)/d},i\in[1,2,...,d/2]\}$. A graphic illustration of RoPE is shown in [Figure 1](#figure-01). Applying our RoPE to self-attention in [Equation 2](#equation-02), we obtain:

<span id="equation-16"></span>

$$
{\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}=({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^\top({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})={\boldsymbol{x}}^\top{\boldsymbol{W}}_{q}R^{d}_{\Theta,n-m}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}
$$

where ${\boldsymbol{R}}^{d}_{\Theta,n-m}=({\boldsymbol{R}}^{d}_{\Theta,m})^\top{\boldsymbol{R}}^{d}_{\Theta,n}$. Note that ${\boldsymbol{R}}^{d}_{\Theta}$ is an orthogonal matrix, which ensures stability during the process of encoding position information. In addition, due to the sparsity of $R^{d}_{\Theta}$, applying matrix multiplication directly as in [Equation 16](#equation-16) is not computationally efficient; we provide another realization in theoretical explanation.

In contrast to the additive nature of position embedding method adopted in the previous works, i.e., [Equations 3](#equation-03), [4](#equation-04), [5](#equation-05), [6](#equation-06), [7](#equation-07), [8](#equation-08), [9](#equation-09) and [10](#equation-10), our approach is multiplicative. Moreover, RoPE naturally incorporates relative position information through rotation matrix product instead of altering terms in the expanded formulation of additive position encoding when applied with self-attention.

<span id="figure-01"></span>

![Figure 1. Implementation of Rotary Position Embedding(RoPE).](../../papers/roformer/figure-01.png)

**Figure 1.** Implementation of Rotary Position Embedding(RoPE).

<span id="section-3-3"></span>

### 3.3 Properties of RoPE

**Long-term decay:** Following [Vas17], we set $\theta_{i}=10000^{-2i/d}$. One can prove that this setting provides a long-term decay property (refer to [Section 3.4.3](#section-3-4-3) for more details), which means the inner-product will decay when the relative position increase. This property coincides with the intuition that a pair of tokens with a long relative distance should have less connection.

**RoPE with linear attention:** The self-attention can be rewritten in a more general form.

<span id="equation-17"></span>

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})}.
$$

The original self-attention chooses $\operatorname{sim}({\boldsymbol{q}}_{m},{\boldsymbol{k}}_{n})=\exp({\boldsymbol{q}}_{m}^\top{\boldsymbol{k}}_{n}/\sqrt{d})$. Note that the original self-attention should compute the inner product of query and key for every pair of tokens, which has a quadratic complexity $O(N^{2})$. Follow [Kat20], the linear attentions reformulate [Equation 17](#equation-17) as

<span id="equation-18"></span>

$$
\mathrm{Attention}({\boldsymbol{Q}},{\boldsymbol{K}},{\boldsymbol{V}})_{m}=\frac{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n}){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n})},
$$

where $\phi(\cdot),\varphi(\cdot)$ are usually non-negative functions. The authors of [Kat20] have proposed $\phi(x)=\varphi(x)=\operatorname{elu}(x)+1$ and first computed the multiplication between keys and values using the associative property of matrix multiplication. A softmax function is used in [She21] to normalize queries and keys separately before the inner product, which is equivalent to $\phi({\boldsymbol{q}}_{i})=\mathrm{softmax}({\boldsymbol{q}}_{i})$ and $\phi({\boldsymbol{k}}_{j})=\exp({\boldsymbol{k}}_{j})$. For more details about linear attention, we encourage readers to refer to original papers. In this section, we focus on discussing incorporating RoPE with [Equation 18](#equation-18). Since RoPE injects position information by rotation, which keeps the norm of hidden representations unchanged, we can combine RoPE with linear attention by multiplying the rotation matrix with the outputs of the non-negative functions.

<span id="equation-19"></span>

$$
\mathrm{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})_{m}=\frac{\sum_{n=1}^{N}\big({\boldsymbol{R}}^{d}_{\Theta,m}\phi({\boldsymbol{q}}_{m})\big)^\top\big({\boldsymbol{R}}^{d}_{\Theta,n}\varphi({\boldsymbol{k}}_{n})\big){\boldsymbol{v}}_{n}}{\sum_{n=1}^{N}\phi({\boldsymbol{q}}_{m})^\top\varphi({\boldsymbol{k}}_{n})}.
$$

It is noteworthy that we keep the denominator unchanged to avoid the risk of dividing zero, and the summation in the numerator could contain negative terms. Although the weights for each value ${\boldsymbol{v}}_{i}$ in [Equation 19](#equation-19) are not strictly probabilistic normalized, we kindly argue that the computation can still model the importance of values.

<span id="section-3-4"></span>

### 3.4 Theoretical Explanation

<span id="section-3-4-1"></span>

#### 3.4.1 Derivation of RoPE under 2D

Under the case of $d=2$, we consider two-word embedding vectors ${\boldsymbol{x}}_{q}$, ${\boldsymbol{x}}_{k}$ corresponds to query and key and their position $m$ and $n$, respectively. According to [Equation 1](#equation-01), their position-encoded counterparts are:

<span id="equation-20"></span>

$$
\begin{aligned}
{\boldsymbol{q}}_{m} & =f_{q}({\boldsymbol{x}}_{q},m), \\
{\boldsymbol{k}}_{n} & =f_{k}({\boldsymbol{x}}_{k},n),
\end{aligned}
$$

where the subscripts of ${\boldsymbol{q}}_{m}$ and ${\boldsymbol{k}}_{n}$ indicate the encoded positions information. Assume that there exists a function $g$ that defines the inner product between vectors produced by $f_{\{q,k\}}$:

<span id="equation-21"></span>

$$
{\boldsymbol{q}}^\top_{m}{\boldsymbol{k}}_{n}=\langle f_{q}({\boldsymbol{x}}_{m},m),f_{k}({\boldsymbol{x}}_{n},n)\rangle=g({\boldsymbol{x}}_{m},{\boldsymbol{x}}_{n},n-m),
$$

we further require below initial condition to be satisfied:

<span id="equation-22"></span>

$$
\begin{aligned}
{\boldsymbol{q}} & =f_{q}({\boldsymbol{x}}_{q},0), \\
{\boldsymbol{k}} & =f_{k}({\boldsymbol{x}}_{k},0),
\end{aligned}
$$

which can be read as the vectors with empty position information encoded. Given these settings, we attempt to find a solution of $f_{q}$, $f_{k}$. First, we take advantage of the geometric meaning of vector in 2D and its complex counter part, decompose functions in [Equations 20](#equation-20) and [21](#equation-21) into:

<span id="equation-23"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{q},m) & =R_{q}({\boldsymbol{x}}_{q},m)e^{i\Theta_{q}({\boldsymbol{x}}_{q},m)}, \\
f_{k}({\boldsymbol{x}}_{k},n) & =R_{k}({\boldsymbol{x}}_{k},n)e^{i\Theta_{k}({\boldsymbol{x}}_{k},n)}, \\
g({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)e^{i\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m)},
\end{aligned}
$$

where $R_{f}$, $R_{g}$ and $\Theta_{f}$, $\Theta_{g}$ are the radical and angular components for $f_{\{q,k\}}$ and $g$, respectively. Plug them into [Equation 21](#equation-21), we get the relation:

<span id="equation-24"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},n) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m), \\
\Theta_{k}({\boldsymbol{x}}_{k},n)-\Theta_{q}({\boldsymbol{x}}_{q},m) & =\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m),
\end{aligned}
$$

with the corresponding initial condition as:

<span id="equation-25"></span>

$$
\begin{aligned}
{\boldsymbol{q}} & =\|{\boldsymbol{q}}\|e^{i\theta_{q}}=R_{q}({\boldsymbol{x}}_{q},0)e^{i\Theta_{q}({\boldsymbol{x}}_{q},0)}, \\
{\boldsymbol{k}} & =\|{\boldsymbol{k}}\|e^{i\theta_{k}}=R_{k}({\boldsymbol{x}}_{k},0)e^{i\Theta_{k}({\boldsymbol{x}}_{k},0)},
\end{aligned}
$$

where $\|{\boldsymbol{q}}\|$, $\|{\boldsymbol{k}}\|$ and $\theta_{q}$, $\theta_{k}$ are the radial and angular part of ${\boldsymbol{q}}$ and ${\boldsymbol{k}}$ on the 2D plane.

Next, we set $m=n$ in [Equation 24](#equation-24) and take into account initial conditions in [Equation 25](#equation-25):

<span id="equation-26"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m)R_{k}({\boldsymbol{x}}_{k},m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=R_{q}({\boldsymbol{x}}_{q},0)R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|, \\
\Theta_{k}({\boldsymbol{x}}_{k},m)-\Theta_{q}({\boldsymbol{x}}_{q},m) & =\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\Theta_{k}({\boldsymbol{x}}_{k},0)-\Theta_{q}({\boldsymbol{x}}_{q},0)=\theta_{k}-\theta_{q}.
\end{aligned}
$$

On one hand, from, a straightforward solution of $R_{f}$ could be formed from [Equation 26a](#equation-26):

<span id="equation-27"></span>

$$
\begin{aligned}
R_{q}({\boldsymbol{x}}_{q},m) & =R_{q}({\boldsymbol{x}}_{q},0)=\|{\boldsymbol{q}}\| \\
R_{k}({\boldsymbol{x}}_{k},n) & =R_{k}({\boldsymbol{x}}_{k},0)=\|{\boldsymbol{k}}\| \\
R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},n-m) & =R_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},0)=\|{\boldsymbol{q}}\|\|{\boldsymbol{k}}\|
\end{aligned}
$$

which interprets the radial functions $R_{q}$, $R_{k}$ and $R_{g}$ are independent from the position information. On the other hand, as can be noticed in [Equation 26b](#equation-26), $\Theta_{q}({\boldsymbol{x}}_{q},m)-\theta_{q}=\Theta_{k}({\boldsymbol{x}}_{k},m)-\theta_{k}$ indicates that the angular functions does not dependent on query and key, we set them to $\Theta_{f}:=\Theta_{q}=\Theta_{k}$ and term $\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)-\theta_{\{q,k\}}$ is a function of position $m$ and is independent of word embedding ${\boldsymbol{x}}_{\{q,k\}}$, we denote it as $\phi(m)$, yielding:

<span id="equation-28"></span>

$$
\Theta_{f}({\boldsymbol{x}}_{\{q,k\}},m)=\phi(m)+\theta_{\{q,k\}},
$$

Further, by plugging $n=m+1$ to [Equation 24](#equation-24) and consider the above equation, we can get:

<span id="equation-29"></span>

$$
\phi(m+1)-\phi(m)=\Theta_{g}({\boldsymbol{x}}_{q},{\boldsymbol{x}}_{k},1)+\theta_{q}-\theta_{k},
$$

Since RHS is a constant irrelevant to $m$, $\phi(m)$ with continuous integer inputs produce an arithmetic progression:

<span id="equation-30"></span>

$$
\phi(m)=m\theta+\gamma,
$$

where $\theta,\gamma\in\mathbb{R}$ are constants and $\theta$ is non-zero. To summarize our solutions from [Equations 27](#equation-27), [28](#equation-28), [29](#equation-29) and [30](#equation-30):

<span id="equation-31"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{q},m) & =\|{\boldsymbol{q}}\|e^{i\theta_{q}+m\theta+\gamma}={\boldsymbol{q}}e^{i(m\theta+\gamma)}, \\
f_{k}({\boldsymbol{x}}_{k},n) & =\|{\boldsymbol{k}}\|e^{i\theta_{k}+n\theta+\gamma}={\boldsymbol{k}}e^{i(n\theta+\gamma)}.
\end{aligned}
$$

Note that we do not apply any constrains to $f_{q}$ and $f_{k}$ of [Equation 22](#equation-22), thus $f_{q}({\boldsymbol{x}}_{m},0)$ and $f_{k}({\boldsymbol{x}}_{n},0)$ are left to choose freely. To make our results comparable to [Equation 3](#equation-03), we define:

<span id="equation-32"></span>

$$
\begin{aligned}
{\boldsymbol{q}}=f_{q}({\boldsymbol{x}}_{m},0) & ={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{n}, \\
{\boldsymbol{k}}=f_{k}({\boldsymbol{x}}_{n},0) & ={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}.
\end{aligned}
$$

Then, we simply set $\gamma=0$ in [Equation 31](#equation-31) of the final solution:

<span id="equation-33"></span>

$$
\begin{aligned}
f_{q}({\boldsymbol{x}}_{m},m) & =({\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})e^{i m\theta}, \\
f_{k}({\boldsymbol{x}}_{n},n) & =({\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})e^{i n\theta}.
\end{aligned}
$$

<span id="section-3-4-2"></span>

#### 3.4.2 Computational efficient realization of rotary matrix multiplication

Taking the advantage of the sparsity of ${\boldsymbol{R}}^{d}_{\Theta,m}$ in [Equation 15](#equation-15), a more computational efficient realization of a multiplication of $R^{d}_{\Theta}$ and ${\boldsymbol{x}}\in\mathbb{R}^{d}$ is:

<span id="equation-34"></span>

$$
{\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{x}}=\begin{pmatrix}x_{1}\\
x_{2}\\
x_{3}\\
x_{4}\\
\vdots\\
x_{d-1}\\
x_{d}\end{pmatrix}\otimes\begin{pmatrix}\cos{m\theta_{1}}\\
\cos{m\theta_{1}}\\
\cos{m\theta_{2}}\\
\cos{m\theta_{2}}\\
\vdots\\
\cos{m\theta_{d/2}}\\
\cos{m\theta_{d/2}}\end{pmatrix}+\begin{pmatrix}-x_{2}\\
x_{1}\\
-x_{4}\\
x_{3}\\
\vdots\\
-x_{d}\\
x_{d-1}\end{pmatrix}\otimes\begin{pmatrix}\sin{m\theta_{1}}\\
\sin{m\theta_{1}}\\
\sin{m\theta_{2}}\\
\sin{m\theta_{2}}\\
\vdots\\
\sin{m\theta_{d/2}}\\
\sin{m\theta_{d/2}}\end{pmatrix}
$$

<span id="section-3-4-3"></span>

#### 3.4.3 Long-term decay of RoPE

We can group entries of vectors ${\boldsymbol{q}}={\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m}$ and ${\boldsymbol{k}}={\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n}$ in pairs, and the inner product of RoPE in [Equation 16](#equation-16) can be written as a complex number multiplication.

<span id="equation-35"></span>

$$
({\boldsymbol{R}}^{d}_{\Theta,m}{\boldsymbol{W}}_{q}{\boldsymbol{x}}_{m})^\top({\boldsymbol{R}}^{d}_{\Theta,n}{\boldsymbol{W}}_{k}{\boldsymbol{x}}_{n})=\operatorname{Re}\bigg[\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg]
$$

where ${\boldsymbol{q}}_{[2i:2i+1]}$ represents the $2i^{th}$ to $(2i+1)^{th}$ entries of ${\boldsymbol{q}}$. Denote $h_{i}={\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}$ and $S_{j}=\sum_{i=0}^{j-1}e^{i(m-n)\theta_{i}}$, and let $h_{d/2}=0$ and $S_{0}=0$, we can rewrite the summation using Abel transformation

<span id="equation-36"></span>

$$
\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}=\sum_{i=0}^{d/2-1}h_{i}(S_{i+1}-S_{i})=-\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i}).
$$

Thus,

<span id="equation-37"></span>

$$
\begin{aligned}
\bigg|\sum_{i=0}^{d/2-1}{\boldsymbol{q}}_{[2i:2i+1]}{\boldsymbol{k}}_{[2i:2i+1]}^{*}e^{i(m-n)\theta_{i}}\bigg| & =\bigg|\sum_{i=0}^{d/2-1}S_{i+1}(h_{i+1}-h_{i})\bigg| \\
\leq\sum_{i=0}^{d/2-1}|S_{i+1}|\,|h_{i+1}-h_{i}| \\
\leq\big(\max_{i}|h_{i+1}-h_{i}|\big)\sum_{i=0}^{d/2-1}|S_{i+1}|
\end{aligned}
$$

Note that the value of $\frac{1}{d/2}\sum_{i=1}^{d/2}|S_{i}|$ decay with the relative distance $m-n$ increases by setting $\theta_{i}=10000^{-2i/d}$, as shown in [Figure 2](#figure-02).

<span id="figure-02"></span>

![Figure 2. Long-term decay of RoPE.](../../papers/roformer/figure-02.png)

**Figure 2.** Long-term decay of RoPE.

<span id="section-4"></span>

## 4 Experiments and Evaluation

We evaluate the proposed RoFormer on various NLP tasks as follows. We validate the performance of the proposed solution on machine translation task [Section 4.1](#section-4-1). Then, we compare our RoPE implementation with BERT[Dev19] during the pre-training stage in [Section 4.2](#section-4-2). Based on the pre-trained model, in [Section 4.3](#section-4-3), we further carry out evaluations across different downstream tasks from GLUE benchmarks[Wan18d]. In Addition, we conduct experiments using the proposed RoPE with the linear attention of PerFormer [Cho20a] in [Section 4.4](#section-4-4). By the end, additional tests on Chinese data are included in [Section 4.5](#section-4-5). All the experiments were run on two cloud severs with 4 x V100 GPUs.

<span id="section-4-1"></span>

### 4.1 Machine Translation

We first demonstrate the performance of RoFormer on sequence-to-sequence language translation tasks.

<span id="section-4-1-1"></span>

#### 4.1.1 Experimental Settings

We choose the standard WMT 2014 English-German dataset[Boj14], which consists of approximately 4.5 million sentence pairs. We compare to the transformer-based baseline alternative [Vas17].

<span id="section-4-1-2"></span>

#### 4.1.2 Implementation details

We carry out some modifications on self-attention layer of the baseline model [Vas17] to enable RoPE to its learning process. We replicate the setup for English-to-German translation with a vocabulary of 37k based on a joint source and target byte pair encoding(BPE)[Sen15]. During the evaluation, a single model is obtained by averaging the last 5 checkpoints. The result uses beam search with a beam size of 4 and length penalty 0.6. We implement the experiment in PyTorch in the fairseq toolkit (MIT License)[Ott19a]. Our model is optimized with the Adam optimizer using $\beta_{1}=0.9$, $\beta_{2}=0.98$, learning rate is increased linearly from $1e-7$ to $5e-4$ and then decayed proportionally to the inverse square root of the step number. Label smoothing with 0.1 is also adopted. We report the BLEU[Pap02] score on the test set as the final metric.

<span id="section-4-1-3"></span>

#### 4.1.3 Results

We train the baseline model and our RoFormer under the same settings and report the results in [Table 1](#table-01). As can be seen, our model gives better BLEU scores compared to the baseline Transformer.

<span id="table-01"></span>

![Table 1. The proposed RoFormer gives better BLEU scores compared to its baseline alternative Vas17 on the WMT 2014 English-to-German translation taskBoj14.](../../papers/roformer/table-01.png)

**Table 1.** The proposed RoFormer gives better BLEU scores compared to its baseline alternative [Vas17] on the WMT 2014 English-to-German translation task[Boj14].

<span id="section-4-2"></span>

### 4.2 Pre-training Language Modeling

The second experiment is to validate the performance of our proposal in terms of learning contextual representations. To achieve this, we replace the original sinusoidal position encoding of BERT with our RoPE during the pre-training step.

<span id="section-4-2-1"></span>

#### 4.2.1 Experimental Settings

We use the BookCorpus [Zhu15] and the Wikipedia Corpus [Wik21] from Huggingface Datasets library (Apache License 2.0) for pre-training. The corpus is further split into train and validation sets at 8:2 ratio. We use the masked language-modeling (MLM) loss values of the training process as an evaluation metric. The well-known BERT [Dev19] is adopted as our baseline model. Note that we use bert-base-uncased in our experiments.

<span id="section-4-2-2"></span>

#### 4.2.2 Implementation details

For RoFormer, we replace the sinusoidal position encoding in the self-attention block of the baseline model with our proposed RoPE and realizes self-attention according to [Equation 16](#equation-16). We train both BERT and RoFormer with batch size 64 and maximum sequence length of 512 for 100k steps. AdamW [Los17] is used as the optimizer with learning rate 1e-5.

<span id="section-4-2-3"></span>

#### 4.2.3 Results

The MLM loss during pre-training is shown on the left plot of [Figure 3](#figure-03). Compare to the vanilla BERT, RoFormer experiences faster convergence.

<span id="figure-03"></span>

![Training loss curves for BERT, RoFormer, and PerFormer variants](../../papers/roformer/figure-03.png)

**Figure 3.** Evaluation of RoPE in language modeling pre-training. **Left**: training loss for BERT and RoFormer. **Right**: training loss for PerFormer with and without RoPE.

<span id="section-4-3"></span>

### 4.3 Fine-tuning on GLUE tasks

Consistent with the previous experiments, we fine-tune the weights of our pre-trained RoFormer across various GLUE tasks in order to evaluate its generalization ability on the downstream NLP tasks.

<span id="section-4-3-1"></span>

#### 4.3.1 Experimental Settings

We look at several datasets from GLUE, i.e. MRPC [Dol05], SST-2 [Soc13], QNLI [Raj16], STS-B [Aln17], QQP [Che18h] and MNLI [Wil18]. We use F1-score for MRPC and QQP dataset, spearman correlation for STS-B, and accuracy for the remaining as the evaluation metrics.

<span id="section-4-3-2"></span>

#### 4.3.2 Implementation details

We use Huggingface Transformers library (Apache License 2.0)[Wol20] to fine-tune each of the aforementioned downstream tasks for 3 epochs, with a maximum sequence length of 512, batch size of 32 and learning rates {2,3,4,5}e-5. Following [Dev19], we report the best-averaged results on the validation set.

<span id="table-02"></span>

![Table 2. Comparing RoFormer and BERT by fine tuning on downstream GLEU tasks.](../../papers/roformer/table-02.png)

**Table 2.** Comparing RoFormer and BERT by fine tuning on downstream GLEU tasks.

<span id="section-4-3-3"></span>

#### 4.3.3 Results

The evaluation results of the fine-tuning tasks are reported in [Table 2](#table-02). As can be seen, RoFormer can significantly outperform BERT in three out of six datasets, and the improvements are considerable.

<span id="section-4-4"></span>

### 4.4 Performer with RoPE

Performer [Cho20a] introduces an alternative attention mechanism, linear attention, which is designed to avoid quadratic computation cost that scales with input sequence length. As discussed in [Section 3.3](#section-3-3), the proposed RoPE can be easily implemented in the PerFormer model to realize the relative position encoding while keeping its linearly scaled complexity in self-attention. We demonstrate its performance with the pre-training task of language modeling.

<span id="section-4-4-1"></span>

#### 4.4.1 Implementation details

We carry out tests on the Enwik8 dataset [Mah06], which is from English Wikipedia that includes markup, special characters and text in other languages in addition to English text. We incorporate RoPE into the 12 layer char-based PerFormer with 768 dimensions and 12 heads [+2]. To better illustrate the efficacy of RoPE, we report the loss curves of the pre-training process with and without RoPE under the same settings, i.e., learning rate 1e-4, batch size 128 and a fixed maximum sequence length of 1024, etc.

<span id="section-4-4-2"></span>

#### 4.4.2 Results

As shown on the right plot of [Figure 3](#figure-03), substituting RoPE into Performer leads to rapid convergence and lower loss under the same amount of training steps. These improvements, in addition to the linear complexity, make Performer more attractive.

<span id="section-4-5"></span>

### 4.5 Evaluation on Chinese Data

In addition to experiments on English data, we show additional results on Chinese data. To validate the performance of RoFormer on long texts, we conduct experiments on long documents whose length exceeds 512 characters.

<span id="section-4-5-1"></span>

#### 4.5.1 Implementation

In these experiments, we carried out some modifications on WoBERT [Su20] by replacing the absolute position embedding with our proposed RoPE. As a cross-comparison with other pre-trained Transformer-based models in Chinese, i.e. BERT [Dev19], WoBERT [Su20], and NEZHA [Wei19], we tabulate their tokenization level and position embedding information in [Table 3](#table-03).

<span id="table-03"></span>

![Table 3. Cross-comparison between our RoFormer and other pre-trained models on Chinese data. ’abs’ and ’rel’ annotates absolute position embedding and relative position embedding, respectively.](../../papers/roformer/table-03.png)

**Table 3.** Cross-comparison between our RoFormer and other pre-trained models on Chinese data. ’abs’ and ’rel’ annotates absolute position embedding and relative position embedding, respectively.

<span id="section-4-5-2"></span>

#### 4.5.2 Pre-training

We pre-train RoFormer on approximately 34GB of data collected from Chinese Wikipedia, news and forums. The pre-training is carried out in multiple stages with changing batch size and maximum input sequence length in order to adapt the model to various scenarios. As shown in [Table 4](#table-04), the accuracy of RoFormer elevates with an increasing upper bound of sequence length, which demonstrates the ability of RoFormer in dealing with long texts. We claim that this is the attribute to the excellent generalizability of the proposed RoPE.

<span id="table-04"></span>

![Table 4. Pre-training strategy of RoFormer on Chinese dataset. The training procedure is divided into various consecutive stages. In each stage, we train the model with a specific combination of maximum sequence length and batch size.](../../papers/roformer/table-04.png)

**Table 4.** Pre-training strategy of RoFormer on Chinese dataset. The training procedure is divided into various consecutive stages. In each stage, we train the model with a specific combination of maximum sequence length and batch size.

<span id="section-4-5-3"></span>

#### 4.5.3 Downstream Tasks & Dataset

We choose Chinese AI and Law 2019 Similar Case Matching (CAIL2019-SCM)[Xia19a] dataset to illustrate the ability of RoFormer in dealing with long texts, i.e., semantic text matching. CAIL2019-SCM contains 8964 triplets of cases published by the Supreme People’s Court of China. The input triplet, denoted as (A, B and C), are fact descriptions of three cases. The task is to predict whether the pair (A, B) is closer than (A, C) under a predefined similarity measure. Note that existing methods mostly cannot perform significantly on CAIL2019-SCM dataset due to the length of documents (i.e., mostly more than 512 characters). We split train, validation and test sets based on the well-known ratio 6:2:2.

<span id="section-4-5-4"></span>

#### 4.5.4 Results

We apply the pre-trained RoFormer model to CAIL2019-SCM with different input lengths. The model is compared with the pre-trained BERT and WoBERT model on the same pre-training data, as shown in [Table 5](#table-05). With short text cut-offs, i.e., 512, the result from RoFormer is comparable to WoBERT and is slightly better than the BERT implementation. However, when increasing the maximum input text length to 1024, RoFormer outperforms WoBERT by an absolute improvement of 1.5%.

<span id="table-05"></span>

![Table 5. Experiment results on CAIL2019-SCM task. Numbers in the first column denote the maximum cut-off sequence length. The results are presented in terms of percent accuracy.](../../papers/roformer/table-05.png)

**Table 5.** Experiment results on CAIL2019-SCM task. Numbers in the first column denote the maximum cut-off sequence length. The results are presented in terms of percent accuracy.

<span id="section-4-5-5"></span>

#### 4.5.5 Limitations of the work

Although we provide theoretical groundings as well as promising experimental justifications, our method is limited by following facts:

- Despite the fact that we mathematically format the relative position relations as rotations under 2D sub-spaces, there lacks of thorough explanations on why it converges faster than baseline models that incorporates other position encoding strategies.
- Although we have proved that our model has favourable property of long-term decay for intern-token products, [Section 3.3](#section-3-3), which is similar to the existing position encoding mechanisms, our model shows superior performance on long texts than peer models, we have not come up with a faithful explanation.

Our proposed RoFormer is built upon the Transformer-based infrastructure, which requires hardware resources for pre-training purpose.

<span id="section-5"></span>

## 5 Conclusions

In this work, we proposed a new position embedding method that incorporates explicit relative position dependency in self-attention to enhance the performance of transformer architectures. Our theoretical analysis indicates that relative position can be naturally formulated using vector production in self-attention, with absolution position information being encoded through a rotation matrix. In addition, we mathematically illustrated the advantageous properties of the proposed method when applied to the Transformer. Finally, experiments on both English and Chinese benchmark datasets demonstrate that our method encourages faster convergence in pre-training. The experimental results also show that our proposed RoFormer can achieve better performance on long texts task.

[+1]: A stack of multiple CNN layers can also capture longer intra-token relation, here we only consider single layer setting.

[+2]: For this experiment, we adopt code (MIT License) from [https://github.com/lucidrains/performer-pytorch](https://github.com/lucidrains/performer-pytorch)
