---
title: 'Mamba: Selective State Space Models'
createTime: 2026/09/06 23:30:00
permalink: /en/papers/mamba/
pageClass: paper-reading
---

> [Albert Gu](https://dblp.org/pid/130/0612.html) [+author-order] and [Tri Dao](https://tridao.me/). First submitted to arXiv on December 1, 2023; current version v2, May 31, 2024. Published at COLM 2024 and selected as an Outstanding Paper. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752). <a href="/paper/mamba.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [COLM 2024 / OpenReview](https://openreview.net/forum?id=tEYskw1VY2). [DOI](https://doi.org/10.48550/arXiv.2312.00752). [TeX source](https://export.arxiv.org/e-print/2312.00752v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Foundation models, now powering most of the exciting applications in deep learning, are almost universally based on the Transformer architecture and its core attention module. Many subquadratic-time architectures such as linear attention, gated convolution and recurrent models, and structured state space models (SSMs) have been developed to address Transformers' computational inefficiency on long sequences, but they have not performed as well as attention on important modalities such as language. We identify that a key weakness of such models is their inability to perform content-based reasoning, and make several improvements. First, simply letting the SSM parameters be functions of the input addresses their weakness with discrete modalities, allowing the model to *selectively* propagate or forget information along the sequence length dimension depending on the current token. Second, even though this change prevents the use of efficient convolutions, we design a hardware-aware parallel algorithm in recurrent mode. We integrate these selective SSMs into a simplified end-to-end neural network architecture without attention or even MLP blocks (**Mamba**). Mamba enjoys fast inference (5$\times$ higher throughput than Transformers) and linear scaling in sequence length, and its performance improves on real data up to million-length sequences. As a general sequence model backbone, Mamba achieves state-of-the-art performance across several modalities such as language, audio, and genomics. On language modeling, our Mamba-3B model outperforms Transformers of the same size and matches Transformers twice its size, both in pretraining and downstream evaluation.

<span id="section-1"></span>

## 1 Introduction

Foundation models (FMs), or large models pretrained on massive data then adapted for downstream tasks, have emerged as an effective paradigm in modern machine learning. The backbone of these FMs are often *sequence models*, operating on arbitrary sequences of inputs from a wide variety of domains such as language, images, speech, audio, time series, and genomics [Sut14, Dos20, Oor16, Bro20, Ism19, Pol23a]. While this concept is agnostic to a particular choice of model architecture, modern FMs are predominantly based on a single type of sequence model: the Transformer [Vas17] and its core attention layer [Bah15]. The efficacy of self-attention is attributed to its ability to route information densely within a context window, allowing it to model complex data. However, this property brings fundamental drawbacks: an inability to model anything outside of a finite window, and quadratic scaling with respect to the window length. An enormous body of research has appeared on more efficient variants of attention to overcome these drawbacks [Tay22a], but often at the expense of the very properties that makes it effective. As of yet, none of these variants have been shown to be empirically effective at scale across domains.

Recently, structured state space sequence models (SSMs) [Gu21a, Gu22a] have emerged as a promising class of architectures for sequence modeling. These models can be interpreted as a combination of recurrent neural networks (RNNs) and convolutional neural networks (CNNs), with inspiration from classical state space models [Kal60]. This class of models can be computed very efficiently as either a recurrence or convolution, with linear or near-linear scaling in sequence length. Additionally, they have principled mechanisms for modeling long-range dependencies [Gu20a] in certain data modalities, and have dominated benchmarks such as the Long Range Arena [Tay21]. Many flavors of SSMs  [Gu22a, Gup22, Gu22b, Li23y, Ma23b, Smi23, Orv23] have been successful in domains involving continuous signal data such as audio and vision [Goe22a, Sao23, Ngu22]. However, they have been less effective at modeling discrete and information-dense data such as text.

We propose a new class of **selective state space models**, that improves on prior work on several axes to achieve the modeling power of Transformers while scaling linearly in sequence length.

**Selection Mechanism.** First, we identify a key limitation of prior models: the ability to efficiently *select* data in an input-dependent manner (i.e. focus on or ignore particular inputs). Building on intuition based on important synthetic tasks such as selective copy and induction heads, we design a simple selection mechanism by parameterizing the SSM parameters based on the input. This allows the model to filter out irrelevant information and remember relevant information indefinitely.

**Hardware-aware Algorithm.** This simple change poses a technical challenge for the computation of the model; in fact, all prior SSMs models must be time- and input-invariant in order to be computationally efficient. We overcome this with a hardware-aware algorithm that computes the model recurrently with a scan instead of convolution, but does not materialize the expanded state in order to avoid IO access between different levels of the GPU memory hierarchy. The resulting implementation is faster than previous methods both in theory (scaling linearly in sequence length, compared to pseudo-linear for all convolution-based SSMs) and on modern hardware (up to 3$\times$ faster on A100 GPUs).

**Architecture.** We simplify prior deep sequence model architectures by combining the design of prior SSM architectures [Dao23d] with the MLP block of Transformers into a single block, leading to a simple and homogenous architecture design (**Mamba**) incorporating selective state spaces.

Selective SSMs, and by extension the Mamba architecture, are fully recurrent models with key properties that make them suitable as the backbone of general foundation models operating on sequences.

1. High quality: selectivity brings strong performance on dense modalities such as language and genomics.
1. Fast training and inference: computation and memory scales linearly in sequence length during training, and unrolling the model autoregressively during inference requires only constant time per step since it does not require a cache of previous elements.
1. Long context: the quality and efficiency together yield performance improvements on real data up to sequence length 1M.

We empirically validate Mamba's potential as a general sequence FM backbone, in both pretraining quality and domain-specific task performance, on several types of modalities and settings:

- **Synthetics.** On important synthetic tasks such as copying and induction heads that have been proposed as being key to large language models, Mamba not only solves them easily but can *extrapolate solutions indefinitely long* ($>$1M tokens).

- **Audio and Genomics.** Mamba out-performs prior state-of-the-art models such as SaShiMi, Hyena, and Transformers on modeling audio waveforms and DNA sequences, both in pretraining quality and downstream metrics (e.g. reducing FID on a challenging speech generation dataset by more than half). In both settings, its *performance improves with longer context up to million-length sequences*.

- **Language Modeling.** Mamba is the first *linear-time sequence model that truly achieves Transformer-quality performance*, both in pretraining perplexity and downstream evaluations. With scaling laws up to 1B parameters, we show that Mamba exceeds the performance of a large range of baselines, including very strong modern Transformer training recipes based on LLaMa [Tou23]. Our Mamba language model has 5$\times$ generation throughput compared to Transformers of similar size, and Mamba-3B's quality matches that of Transformers twice its size (e.g. 4 points higher avg. on common sense reasoning compared to Pythia-3B and even exceeding Pythia-7B).

Model code and pre-trained checkpoints are open-sourced at <https://github.com/state-spaces/mamba>.

<span id="figure-01"></span>

![Overview of selective state space models](../../papers/mamba/figure-01.png)

**Figure 1.** (**Overview**.) Structured SSMs independently map each channel (e.g. $D=5$) of an input $x$ to output $y$ through a higher dimensional latent state $h$ (e.g. $N=4$). Prior SSMs avoid materializing this large effective state ($D N$, times batch size $B$ and sequence length $L$) through clever alternate computation paths requiring time-invariance: the $(\Delta, \bm{A}, \bm{B}, \bm{C})$ parameters are constant across time. Our selection mechanism adds back input-dependent dynamics, which also requires a careful hardware-aware algorithm to only materialize the expanded states in more efficient levels of the GPU memory hierarchy.

<span id="section-2"></span>

## 2 State Space Models

Structured state space sequence models (S4) are a recent class of sequence models for deep learning that are broadly related to RNNs, and CNNs, and classical state space models. They are inspired by a particular continuous system [Equation 1](#equation-01) that maps a 1-dimensional function or sequence $x(t) \in \mathbb{R}\mapsto y(t) \in \mathbb{R}$ through an implicit latent state $h(t) \in \mathbb{R}^N$.

Concretely, S4 models are defined with four parameters $(\Delta, \bm{A}, \bm{B}, \bm{C})$, which define a sequence-to-sequence transformation in two stages.

<span id="equation-01"></span>

$$
\begin{aligned}
      h'(t) &= \bm{A}h(t) + \bm{B}x(t) \\
      y(t) &= \bm{C}h(t)
    \end{aligned}
$$

<span id="equation-02"></span>
<span id="equation-02-a"></span>
<span id="equation-02-b"></span>

$$
\begin{aligned}
      h_{t} &= \overline{\bm{A}}h_{t-1} + \overline{\bm{B}}x_t \\
      y_t &= \bm{C}h_t
    \end{aligned}
$$

<span id="equation-03"></span>
<span id="equation-03-a"></span>
<span id="equation-03-b"></span>

$$
\begin{aligned}
      \bm{\overline{K}} &= (\bm{C}\bm{\overline{B}}, \bm{C}\bm{\overline{A}}\bm{\overline{B}}, \dots, \bm{C}\bm{\overline{A}}^{k}\bm{\overline{B}}, \dots) \\
      y &= x \ast \bm{\overline{K}}
    \end{aligned}
$$

**Discretization.** The first stage transforms the "continuous parameters" $(\Delta, \bm{A}, \bm{B})$ to "discrete parameters" $(\overline{\bm{A}}, \overline{\bm{B}})$ through fixed formulas $\overline{\bm{A}}= f_A(\Delta, \bm{A})$ and $\overline{\bm{B}}= f_B(\Delta, \bm{A}, \bm{B})$, where the pair $(f_A, f_B)$ is called a *discretization rule*.

Various rules can be used such as the zero-order hold (ZOH) defined in equation [Equation 4](#equation-04).

<span id="equation-04"></span>

$$
\overline{\bm{A}}= \exp(\Delta\bm{A})
    \qquad
    \overline{\bm{B}}= (\Delta\bm{A})^{-1} (\exp(\Delta\bm{A}) - \bm{I}) \cdot \Delta\bm{B}
$$

Discretization has deep connections to continuous-time systems which can endow them with additional properties such as resolution invariance [Ngu22] and automatically ensuring that the model is properly normalized [Gu23a, Orv23]. It also has connections to gating mechanisms of RNNs [Tal18a, Gu20b] which we will revisit in [Section 3.5](#section-3-5). However, from a mechanical point of view discretization can simply be viewed as the first step of the computation graph in the forward pass of an SSM.

Alternate flavors of SSMs can bypass the discretization step and parameterize $(\overline{\bm{A}}, \overline{\bm{B}})$ directly instead [Zha23q], which may be easier to reason about.

**Computation.** After the parameters have been transformed from $(\Delta, \bm{A}, \bm{B}, \bm{C}) \mapsto (\overline{\bm{A}}, \overline{\bm{B}}, \bm{C})$, the model can be computed in two ways, either as a **linear recurrence** [Equation 2](#equation-02) or a **global convolution** [Equation 3](#equation-03).

Commonly, the model uses the convolutional mode [Equation 3](#equation-03) for efficient parallelizable training (where the whole input sequence is seen ahead of time), and switched into recurrent mode [Equation 2](#equation-02) for efficient autoregressive inference (where the inputs are seen one timestep at a time).

**Linear Time Invariance (LTI).** An important property of equations [Equation 1](#equation-01) to [Equation 3](#equation-03) is that the model's dynamics are constant through time. In other words $(\Delta, \bm{A}, \bm{B}, \bm{C})$, and consequently $(\overline{\bm{A}}, \overline{\bm{B}})$ as well, are fixed for all time-steps. This property is called *linear time invariance (LTI)*, which is deeply connected to recurrence and convolutions. Informally, we think of LTI SSMs as being equivalent to any linear recurrence [Equation 2a](#equation-02-a) or convolution [Equation 3b](#equation-03-b), and use LTI as an umbrella term for these classes of models.

Thus far, all structured SSMs have been LTI (e.g. computed as convolutions) because of fundamental efficiency constraints, discussed in [Section 3.3](#section-3-3). However, a core insight of this work is that LTI models have fundamental limitations in modeling certain types of data, and our technical contributions involve removing the LTI constraint while overcoming the efficiency bottlenecks.

**Structure and Dimensions.** Finally, we note that structured SSMs are so named because computing them efficiently also requires imposing structure on the $\bm{A}$ matrix. The most popular form of structure is diagonal [Gup22, Gu22b, Smi23], which we also use.

In this case, the $\bm{A}\in \mathbb{R}^{N \times N}, \bm{B}\in \mathbb{R}^{N \times 1}, \bm{C}\in \mathbb{R}^{1 \times N}$ matrices can all be represented by $N$ numbers. To operate over an input sequence $x$ of batch size $B$ and length $L$ with $D$ channels, the SSM is applied independently to each channel. Note that in this case, the total hidden state has dimension $D N$ per input, and computing it over the sequence length requires $O(B L D N)$ time and memory; this is the root of the fundamental efficiency bottleneck addressed in [Section 3.3](#section-3-3).

**General State Space Models.** We note that the term *state space model* has a very broad meaning which simply represents the notion of any recurrent process with a latent state. It has been used to refer to many disparate concepts in different disciplines, including Markov decision processes (MDP) (reinforcement learning [Haf20]), dynamic causal modeling (DCM) (computational neuroscience [Fri03]), Kalman filters (controls [Kal60]), hidden Markov models (HMM) and linear dynamical systems (LDS) (machine learning), and recurrent (and sometimes convolutional) models at large (deep learning).

Throughout this entire paper we use the term "SSM" to refer exclusively to the class of structured SSMs or S4 models [Gu22a, Gup22, Gu22b, Ma23b, Smi23, Has23] and use these terms interchangeably. For convenience we may also include derivatives of such models, such as those focusing on either the linear-recurrence or global-convolution viewpoints [Orv23, Li23y, Pol23a], and clarify nuances when necessary.

**SSM Architectures.** SSMs are standalone sequence transformations that can be incorporated into end-to-end neural network architectures.

(We also sometimes call SSM architectures SSNNs, which are to SSM layers as CNNs are to linear convolution layers.)

We discuss some of the most well-known SSM architectures, many of which will also serve as our primary baselines.

- Linear attention [Kat20] is an approximation of self-attention involving a recurrence which can be viewed as a degenerate linear SSM.

- H3 [Dao23d] generalized this recurrence to use S4; it can be viewed as an architecture with an SSM sandwiched by two gated connections ([Figure 3](#figure-03)). H3 also inserts a standard local convolution, which they frame as a shift-SSM, before the main SSM layer.

- Hyena [Pol23a] uses the same architecture as H3 but replaces the S4 layer with an MLP-parameterized global convolution [Rom21].

- RetNet [Sun23a] adds an additional gate to the architecture and uses a simpler SSM, allowing an alternative parallelizable computation path, using a variant of multi-head attention (MHA) instead of convolutions.

- RWKV [Pen23g] is a recent RNN designed for language modeling based on another linear attention approximation, the attention-free Transformer [Zha21e]. Its main "WKV" mechanism involves LTI recurrences and can be viewed as the ratio of two SSMs.

Other closely related SSMs and architectures are discussed further in an extended related work ([Section 8](#section-8)). We highlight in particular S5 [Smi23], QRNN [Bra16], and SRU [Lei17], which we view as the most closely related methods to our core selective SSM.

<span id="section-3"></span>

## 3 Selective State Space Models

We motivate our selection mechanism using intuition from synthetic tasks ([Section 3.1](#section-3-1)), then explain how to incorporate this mechanism into state space models ([Section 3.2](#section-3-2)). The resulting time-varying SSMs cannot use convolutions, presenting a technical challenge of how to compute them efficiently. We overcome this with a hardware-aware algorithm that exploits the memory hierarchy on modern hardware ([Section 3.3](#section-3-3)). We then describe a simple SSM architecture without attention or even MLP blocks ([Section 3.4](#section-3-4)). Finally, we discuss some additional properties of selection mechanisms ([Section 3.5](#section-3-5)).

<span id="section-3-1"></span>

### 3.1 Motivation: Selection as a Means of Compression

We argue that a fundamental problem of sequence modeling is *compressing context into a smaller state*. In fact, we can view the tradeoffs of popular sequence models from this point of view. For example, attention is both effective and inefficient because it explicitly does not compress context at all. This can be seen from the fact that autoregressive inference requires explicitly storing the entire context (i.e. the KV cache), which directly causes the slow linear-time inference and quadratic-time training of Transformers. On the other hand, recurrent models are efficient because they have a finite state, implying constant-time inference and linear-time training. However, their effectiveness is limited by how well this state has compressed the context.

To understand this principle, we focus on two running examples of synthetic tasks ([Figure 2](#figure-02)).

- The **Selective Copying** task modifies the popular Copying task [Arj16] by varying the position of the tokens to memorize. It requires *content-aware* reasoning to be able to memorize the relevant tokens (*colored*) and filter out the irrelevant ones (*white*).

- The **Induction Heads** task is a well-known mechanism hypothesized to explain the majority of in-context learning abilities of LLMs [Ols22]. It requires *context-aware* reasoning to know when to produce the correct output in the appropriate context (*black*).

These tasks reveal the failure mode of LTI models. From the recurrent view, their constant dynamics (e.g. the $(\overline{\bm{A}}, \overline{\bm{B}})$ transitions in [Equation 2](#equation-02)) cannot let them select the correct information from their context, or affect the hidden state passed along the sequence in an input-dependent way. From the convolutional view, it is known that global convolutions can solve the vanilla Copying task [Rom21] because it only requires time-awareness, but that they have difficulty with the Selective Copying task because of lack of content-awareness ([Figure 2](#figure-02)). More concretely, the spacing between inputs-to-outputs is varying and cannot be modeled by static convolution kernels.

In summary, the efficiency vs. effectiveness tradeoff of sequence models is characterized by how well they compress their state: efficient models must have a small state, while effective models must have a state that contains all necessary information from the context. In turn, we propose that a fundamental principle for building sequence models is **selectivity**: or the context-aware ability to focus on or filter out inputs into a sequential state. In particular, a selection mechanism controls how information propagates or interacts along the sequence dimension (see [Section 3.5](#section-3-5) for more discussion).

<span id="figure-02"></span>

![Copying, selective copying, and induction-head tasks](../../papers/mamba/figure-02.png)

**Figure 2.** (*Left*) The standard version of the Copying task involves constant spacing between input and output elements and is easily solved by time-invariant models such as linear recurrences and global convolutions. (*Right Top*) The Selective Copying task has random spacing in between inputs and requires time-varying models that can *selectively* remember or ignore inputs depending on their content. (*Right Bottom*) The Induction Heads task is an example of associative recall that requires retrieving an answer based on context, a key ability for LLMs.

<span id="section-3-2"></span>

### 3.2 Improving SSMs with Selection

One method of incorporating a selection mechanism into models is by letting their parameters that affect interactions along the sequence (e.g. the recurrent dynamics of an RNN or the convolution kernel of a CNN) be input-dependent.

[Algorithm 1](#algorithm-01) and [Algorithm 2](#algorithm-02) illustrates the main selection mechanism that we use. The main difference is simply making several parameters $\Delta, \bm{B}, \bm{C}$ functions of the input, along with the associated changes to tensor shapes throughout. In particular, we highlight that these parameters now have a length dimension $L$, meaning that the model has changed from time-invariant to time-varying. (Note that shape annotations were described in [Section 2](#section-2).) This loses the equivalence to convolutions [Equation 3](#equation-03) with implications for its efficiency, discussed next.

We specifically choose $s_B(x) = \mathrm{Linear}_N(x)$, $s_C(x) = \mathrm{Linear}_N(x)$, $s_\Delta(x) = \mathrm{Broadcast}_D(\mathrm{Linear}_1(x))$, and $\tau_\Delta= \mathrm{softplus}$, where $\mathrm{Linear}_d$ is a parameterized projection to dimension $d$. The choice of $s_\Delta$ and $\tau_\Delta$ is due to a connection to RNN gating mechanisms explained in [Section 3.5](#section-3-5).

<span id="algorithm-01"></span>

**Algorithm 1. SSM (S4)**

- **Input:** $x : (B, L, D)$
- **Output:** $y : (B, L, D)$
- $\bm{A} : (D, N) \leftarrow \mathrm{Parameter}$
  - Represents structured $N \times N$ matrix
- $\bm{B} : (D, N) \leftarrow \mathrm{Parameter}$
- $\bm{C} : (D, N) \leftarrow \mathrm{Parameter}$
- $\Delta : (D) \leftarrow \tau_\Delta(\mathrm{Parameter})$
- $\overline{\bm{A}}, \overline{\bm{B}} : (D, N) \leftarrow \mathrm{discretize}(\Delta, \bm{A}, \bm{B})$
- $y \leftarrow \mathrm{SSM}(\overline{\bm{A}}, \overline{\bm{B}}, \bm{C})(x)$
  - Time-invariant: recurrence or convolution
- **Return** $y$

<span id="algorithm-02"></span>

**Algorithm 2. SSM + Selection (S6)**

- **Input:** $x : (B, L, D)$
- **Output:** $y : (B, L, D)$
- $\bm{A} : (D, N) \leftarrow \mathrm{Parameter}$
  - Represents structured $N \times N$ matrix
- $\bm{B} : (B, L, N) \leftarrow s_B(x)$
- $\bm{C} : (B, L, N) \leftarrow s_C(x)$
- $\Delta : (B, L, D) \leftarrow \tau_\Delta(\mathrm{Parameter} + s_\Delta(x))$
- $\overline{\bm{A}}, \overline{\bm{B}} : (B, L, D, N) \leftarrow \mathrm{discretize}(\Delta, \bm{A}, \bm{B})$
- $y \leftarrow \mathrm{SSM}(\overline{\bm{A}}, \overline{\bm{B}}, \bm{C})(x)$
  - Time-varying: recurrence (scan) only
- **Return** $y$

<span id="section-3-3"></span>

### 3.3 Efficient Implementation of Selective SSMs

Hardware-friendly primitives such as convolutions [Kri12] and attention [Bah15, Vas17] enjoy widespread application. Here we aim to make selective SSMs efficient on modern hardware (GPUs) as well. The selection mechanism is quite natural, and earlier works attempted to incorporate special cases of selection, such as letting $\Delta$ vary over time in recurrent SSMs [Gu20a]. However,

as previously mentioned a core limitation in the usage of SSMs is their computational efficiency,

which was why S4 and all derivatives used LTI (non-selective) models, most commonly in the form of global convolutions.

<span id="section-3-3-1"></span>

#### 3.3.1 Motivation of Prior Models

We first revisit this motivation and overview our approach to overcome limitations of prior methods.

- At a high level, recurrent models such as SSMs always balance a tradeoff between expressivity and speed: as discussed in [Section 3.1](#section-3-1), models with larger hidden state dimension should be more effective but slower. Thus we want to *maximize hidden state dimension without paying speed and memory costs*.

- Note that the recurrent mode is more flexible than the convolution mode, since the latter [Equation 3](#equation-03) is derived from expanding the former [Equation 2](#equation-02) [Gu21a, Gu22a]. However, this would require computing and materializing the latent state $h$ with shape $\mathtt{(B,L,D,N)}$, which is much larger (by a factor of $N$, the SSM state dimension) than the input $x$ and output $y$ of shape $\mathtt{(B,L,D)}$. Thus the more efficient convolution mode was introduced which could bypass the state computation and materializes a convolution kernel [Equation 3a](#equation-03-a) of size only $\mathtt{(B,L,D)}$.

- Prior LTI state space models leverage the dual recurrent-convolutional forms to increase the effective state dimension by a factor of $N$ ($\approx 10-100$), much larger than traditional RNNs, without efficiency penalties.

<span id="section-3-3-2"></span>

#### 3.3.2 Overview of Selective Scan: Hardware-Aware State Expansion

The selection mechanism is designed to overcome the limitations of LTI models; at the same time, we therefore need to revisit the computation problem of SSMs. We address this with three classical techniques: kernel fusion, parallel scan, and recomputation. We make two main observations:

- The naive recurrent computation uses $O(B L D N)$ FLOPs while the convolutional computation uses $O(B L D \log(L))$ FLOPs, and the former has a lower constant factor. Thus for long sequences and not-too-large state dimension $N$, the recurrent mode can actually use fewer FLOPs.

- The two challenges are the sequential nature of recurrence, and the large memory usage. To address the latter, just like the convolutional mode, we can attempt to not actually materialize the full state $h$.

The main idea is to leverage properties of modern accelerators (GPUs) to materialize the state $h$ only in more efficient levels of the memory hierarchy. In particular, most operations (except matrix multiplication) are bounded by memory bandwidth [Wil09, Iva21, Dao22]. This includes our scan operation, and we use kernel fusion to reduce the amount of memory IOs, leading to a significant speedup compared to a standard implementation.

Concretely, instead of preparing the scan input $(\overline{\bm{A}}, \overline{\bm{B}})$ of size $\mathtt{(B,L,D,N)}$ in GPU HBM (high-bandwidth memory), we load the SSM parameters $(\Delta, \bm{A}, \bm{B}, \bm{C})$ directly from slow HBM to fast SRAM, perform the discretization and recurrence in SRAM, and then write the final outputs of size $(\mathtt{B,L,D})$ back to HBM.

To avoid the sequential recurrence, we observe that despite not being linear it can still be parallelized with a work-efficient parallel scan algorithm [Ble90, Mar18, Smi23].

Finally, we must also avoid saving the intermediate states, which are necessary for backpropagation. We carefully apply the classic technique of recomputation to reduce the memory requirements: the intermediate states are not stored but recomputed in the backward pass when the inputs are loaded from HBM to SRAM. As a result, the fused selective scan layer has the same memory requirements as an optimized transformer implementation with FlashAttention.

Details of the fused kernel and recomputation are in [Section 10](#section-10).

The full Selective SSM layer and algorithm is illustrated in [Figure 1](#figure-01).

<span id="section-3-4"></span>

### 3.4 A Simplified SSM Architecture

As with structured SSMs, selective SSMs are standalone sequence transformations that can be flexibly incorporated into neural networks. The H3 architecture is the basis for the most well-known SSM architectures ([Section 2](#section-2)), which are generally comprised of a block inspired by linear attention interleaved with an MLP (multi-layer perceptron) block. We simplify this architecture by combining these two components into one, which is stacked homogenously ([Figure 3](#figure-03)). This is inspired by the gated attention unit (GAU) [Hua22], which did something similar for attention.

This architecture involves expanding the model dimension $D$ by a controllable expansion factor $E$. For each block, most of the parameters ($3 E D^2$) are in the linear projections ($2 E D^2$ for input projections, $E D^2$ for output projection) while the inner SSM contributes less.

The number of SSM parameters (projections for $\Delta, \bm{B}, \bm{C}$, and the matrix $\bm{A}$) are much smaller in comparison.

We repeat this block, interleaved with standard normalization and residual connections, to form the Mamba architecture. We always fix to $E=2$ in our experiments and use two stacks of the block to match the $12D^2$ parameters of a Transformer's interleaved MHA (multi-head attention) and MLP blocks.

We use the SiLU / Swish activation function [Hen16a, Ram17], motivated so that the Gated MLP becomes the popular "SwiGLU" variant [Dau17, Sha20, Cho23a, Tou23].

Finally, we additionally use an optional normalization layer (we choose LayerNorm [Ba16]), motivated by RetNet's usage of a normalization layer in a similar location [Sun23a].

<span id="figure-03"></span>

![Mamba architecture](../../papers/mamba/figure-03.png)

**Figure 3.** (**Architecture**.) Our simplified block design combines the H3 block, which is the basis of most SSM architectures, with the ubiquitous MLP block of modern neural networks. Instead of interleaving these two blocks, we simply repeat the Mamba block homogenously. Compared to the H3 block, Mamba replaces the first multiplicative gate with an activation function. Compared to the MLP block, Mamba adds an SSM to the main branch. For $\sigma$ we use the SiLU / Swish activation [Hen16a, Ram17].

<span id="section-3-5"></span>

### 3.5 Properties of Selection Mechanisms

The selection mechanism is a broader concept that can be applied in different ways, such as to more traditional RNNs or CNNs, to different parameters (e.g. $\bm{A}$ in [Algorithm 2](#algorithm-02)), or using different transformations $s(x)$.

<span id="section-3-5-1"></span>

#### 3.5.1 Connection to Gating Mechanisms

We highlight the most important connection: the classical gating mechanism of RNNs is an instance of our selection mechanism for SSMs. We note that the connection between RNN gating and the discretization of continuous-time systems is well established [Fun93, Tal18a]. In fact, [Theorem 1](#theorem-01) is an improvement of [Gu21a] (Lemma 3.1), generalizing to the ZOH discretization and input-dependent gates (proof in [Section 9](#section-9)). More broadly, $\Delta$ in SSMs can be seen to play a generalized role of the RNN gating mechanism. In line with prior work, we adopt the view that *discretization of SSMs is the principled foundation of heuristic gating mechanisms*.

<span id="theorem-01"></span>

**Theorem 1.** When $N=1, \bm{A}=-1, \bm{B}=1, s_\Delta=\mathrm{Linear}(x)$, and $\tau_\Delta=\mathrm{softplus}$, then the selective SSM recurrence ([Algorithm 2](#algorithm-02)) takes the form

<span id="equation-05"></span>

$$
\begin{aligned}
        g_t &= \sigma(\mathrm{Linear}(x_t)) \\
        h_{t} &= (1-g_t) h_{t-1} + g_t x_t
        .
      \end{aligned}
$$

As mentioned in [Section 3.2](#section-3-2), our specific choices of $s_\Delta, \tau_\Delta$ is from this connection. In particular, note that if a given input $x_t$ should be completely ignored (as necessary in the synthetic tasks), all $D$ channels should ignore it, and so we project the input down to $1$ dimension before repeating/broadcasting with $\Delta$.

<span id="section-3-5-2"></span>

#### 3.5.2 Interpretation of Selection Mechanisms

We elaborate on three particular mechanistic effects of selection.

**Variable Spacing.** Selectivity allows filtering out irrelevant noise tokens that may occur between inputs of interest. This is exemplified by the Selective Copying task, but occurs ubiquitously in common data modalities, particularly for discrete data—for example the presence of language fillers such as "um". This property arises because the model can mechanistically filter out any particular input $x_t$, for example in the gated RNN case ([Theorem 1](#theorem-01)) when $g_t \to 0$.

**Filtering Context.** It has been empirically observed that many sequence models do not improve with longer context [Shi23d], despite the principle that more context should lead to strictly better performance. An explanation is that many sequence models cannot effectively ignore irrelevant context when necessary; an intuitive example are global convolutions (and general LTI models). On the other hand, selective models can simply reset their state at any time to remove extraneous history, and thus their performance in principle improves monotonicly with context length (e.g. [Section 4.3.2](#section-4-3-2)).

**Boundary Resetting.** In settings where multiple independent sequences are stitched together, Transformers can keep them separate by instantiating a particular attention mask, while LTI models will bleed information between the sequences. Selective SSMs can also reset their state at boundaries (e.g. $\Delta_t \to \infty$, or [Theorem 1](#theorem-01) when $g_t \to 1$). These settings may occur artificially (e.g. packing documents together to improve hardware utilization) or naturally (e.g. episode boundaries in reinforcement learning [Lu23a]).

Additionally, we elaborate on effects of each selective parameter.

**Interpretation of $\Delta$.** In general, $\Delta$ controls the balance between how much to focus or ignore the current input $x_t$. It generalizes RNN gates (e.g. $g_t$ in [Theorem 1](#theorem-01)): mechanically, a large $\Delta$ resets the state $h$ and focuses on the current input $x$, while a small $\Delta$ persists the state and ignores the current input. SSMs [Equation 1](#equation-01)-[Equation 2](#equation-02) can be interpreted as a continuous system discretized by a timestep $\Delta$, and in this context the intuition is that large $\Delta\to \infty$ represents the system focusing on the current input for longer (thus "selecting" it and forgetting its current state) while a small $\Delta\to 0$ represents a transient input that is ignored.

**Interpretation of $\bm{A}$.** We remark that while the $\bm{A}$ parameter could also be selective, it ultimately affects the model only through its interaction with $\Delta$ via $\overline{\bm{A}}= \exp(\Delta\bm{A})$ (the discretization [Equation 4](#equation-04)). Thus selectivity in $\Delta$ is enough to ensure selectivity in $(\overline{\bm{A}}, \overline{\bm{B}})$, and is the main source of improvement. We hypothesize that making $\bm{A}$ selective in addition to (or instead of) $\Delta$ would have similar performance, and leave it out for simplicity.

**Interpretation of $\bm{B}$ and $\bm{C}$.** As discussed in [Section 3.1](#section-3-1), the most important property of selectivity is filtering out irrelevant information so that a sequence model's context can be compressed into an efficient state. In an SSM, modifying $\bm{B}$ and $\bm{C}$ to be selective allows finer-grained control over whether to let an input $x_t$ into the state $h_t$, or the state into the output $y_t$. These can be interpreted as allowing the model to modulate the recurrent dynamics based on content (input) and context (hidden states) respectively.

<span id="section-3-6"></span>

### 3.6 Additional Model Details

**Real vs. Complex.** Most prior SSMs use complex numbers in their state $h$, which is necessary for strong performance on many tasks in perceptual modalities [Gu22a]. However, it has been empirically observed that completely real-valued SSMs seem to work fine, and possibly even better, in some settings [Ma23b]. We use real values as the default, which work well for all but one of our tasks; we hypothesize that the complex-real tradeoff is related to the continuous-discrete spectrum in data modalities, where complex numbers are helpful for continuous modalities (e.g. audio, video) but not discrete (e.g. text, DNA).

**Initialization.** Most prior SSMs also suggest special initializations, particularly in the complex-valued case, which can help in several settings such as low-data regimes. Our default initialization for the complex case is S4D-Lin and for the real case is S4D-Real [Gu22b], which is based on the HIPPO theory [Gu20a]. These define the $n$-th element of $\bm{A}$ as $-1/2 + n i$ and $-(n+1)$ respectively. However, we expect many initializations to work fine, particularly in the large-data and real-valued SSM regimes; some ablations are considered in [Section 4.6](#section-4-6).

**Parameterization of $\Delta$.** We defined the selective adjustment to $\Delta$ as $s_\Delta(x) = \mathrm{Broadcast}_D(\mathrm{Linear}_1(x))$, which was motivated by the mechanics of $\Delta$ ([Section 3.5](#section-3-5)). We observe that it can be generalized from dimension $1$ to a larger dimension $\mathtt{R}$. We set this to be a small fraction of $\mathtt{D}$, which uses a negligible number of parameters compared to the main Linear projections in the block. We additionally note that the broadcasting operation can instead be viewed as another Linear projection, initialized to a specific pattern of $1$'s and $0$'s; if this projection is trainable, this leads to the alternative $s_\Delta(x) = \mathrm{Linear}_D(\mathrm{Linear}_R(x))$, which can be viewed as a low-rank projection.

In our experiments, the $\Delta$ parameter (which can be viewed as a bias term) is initialized to $\tau_\Delta^{-1}(\mathrm{Uniform}([0.001, 0.1]))$, following prior work on SSMs [Gu23a].

**Remark.** For brevity in our experimental results, we sometimes abbreviate selective SSMs as *S6 models*, because they are S4 models with a *selection* mechanism and computed with a *scan*.
<span id="section-4"></span>

## 4 Empirical Evaluation

In [Section 4.1](#section-4-1) we test Mamba's ability to solve the two synthetic tasks motivated in [Section 3.1](#section-3-1). We then evaluate on three domains, each evaluated on autoregressive pretraining as well as downstream tasks.

- [Section 4.2](#section-4-2): language model pretraining (scaling laws), and zero-shot downstream evaluation.

- [Section 4.3](#section-4-3): DNA sequence pretraining, and fine-tuning on a long-sequence classification task.

- [Section 4.4](#section-4-4): audio waveform pretraining, and the quality of autoregressively generated speech clips.

Finally, [Section 4.5](#section-4-5) shows Mamba's computational efficiency at both training and inference time, and [Section 4.6](#section-4-6) ablates various components of the architecture and selective SSMs.

<span id="section-4-1"></span>

### 4.1 Synthetic Tasks

Full experiment details for these tasks including task details and training protocol are in [Section 11.1](#section-11-1).

<span id="section-4-1-1"></span>

#### 4.1.1 Selective Copying

The Copying task is one of the most well-studied synthetic tasks for sequence modeling, originally designed to test the memorization abilities of recurrent models. As discussed in [Section 3.1](#section-3-1), LTI SSMs (linear recurrences and global convolutions) can easily solve this task by only keeping track of time instead of reasoning about the data; for example, by constructing a convolution kernel of exactly the right length ([Figure 2](#figure-02)). This was explicitly validated in earlier work on global convolutions [Rom21]. The Selective Copying task prevents this shortcut by randomizing the spacing between tokens. Note that this task has been introduced before as the Denoising task [Jin19].

Note that many previous works argue that adding architecture gating (multiplicative interactions) can endow models with "data-dependence" and solve related tasks [Dao23d, Pol23a]. However, we find this explanation insufficient intuitively because such gating does not interact along the sequence axis, and cannot affect the spacing between tokens. In particular architecture gating is not an instance of a selection mechanism ([Section 7](#section-7)).

[Table 1](#table-01) confirms that gated architectures such as H3 and Mamba only partially improve performance, while the selection mechanism (modifying S4 to S6) easily solves this task, particularly when combined with these more powerful architectures.

<span id="section-4-1-2"></span>

#### 4.1.2 Induction Heads

Induction heads [Ols22] is a simple task from the mechanistic interpretability lens [Elh21] that is surprisingly predictive of the in-context learning ability of LLMs. It requires models to perform associative recall and copy: for example, if the model has seen a bigram such as "Harry Potter" in the sequence, then the next time "Harry" appears in the same sequence, the model should be able to predict "Potter" by copying from history.

**Dataset.** We train a 2-layer model on the induction heads task at sequence length $256$, with a vocab size of $16$, which is comparable to prior work on this task [Dao23d] but with longer sequences. We additionally investigate generalization and extrapolation abilities by evaluating on a range of sequence lengths from $2^6 = 64$ up to $2^{20} = 1048576$ at test time.

**Models.** Following established work on induction heads, we use 2 layer models, which allows attention to mechanistically solve the induction heads task [Ols22]. We test both multi-head attention (8 heads, with various positional encodings) and SSM variants. We use a model dimension $D$ of $64$ for Mamba and $128$ for the other models.

**Results.** [Table 2](#table-02) shows that Mamba—or more precisely, its selective SSM layer—has the ability to solve the task perfectly because of its ability to selectively remember the relevant token while ignoring everything else in between. **It generalizes perfectly to million-length sequences, or $4000\times$ longer than it saw during training**, while no other method goes beyond $2\times$.

Out of positional encoding variants for attention models, xPos (which was designed for length extrapolation) is slightly better than the others; also note that all attention models were only tested up to sequence length $2^{14}=16384$ due to memory limitations. Out of other SSMs, H3 and Hyena are similar, contrary to the findings in [Pol23a].

<span id="table-01"></span>

![Selective copying accuracy table](../../papers/mamba/table-01.png)

**Table 1.** (**Selective Copying**.)
Accuracy for combinations of architectures and inner sequence layers.

<span id="table-02"></span>

![Induction-head length generalization table](../../papers/mamba/table-02.png)

**Table 2.** (**Induction Heads**.) Models are trained on sequence length $2^8=256$, and tested on increasing sequence lengths of $2^6=64$ up to $2^{20}=1048576$. Full numbers in [Table 11](#table-11).

<span id="section-4-2"></span>

### 4.2 Language Modeling

We evaluate the Mamba architecture on standard autoregressive language modeling against other architectures, on both pretraining metrics (perplexity) and zero-shot evaluations. We set the model sizes (depth and width) to mirror GPT3 specifications. We use the Pile dataset [Gao20], and follow the training recipe described in [Bro20]. All training details are in [Section 11.2](#section-11-2).

<span id="section-4-2-1"></span>

#### 4.2.1 Scaling Laws

For baselines, we compare against the standard Transformer architecture (GPT3 architecture), as well as the strongest Transformer recipe we know of (here referred to as Transformer++), based on the PaLM and LLaMa architectures (e.g. rotary embedding, SwiGLU MLP, RMSNorm instead of LayerNorm, no linear bias, and higher learning rates). We also compare against other recent subquadratic architectures ([Figure 4](#figure-04)). All model details are in [Section 11.2](#section-11-2).

[Figure 4](#figure-04) shows scaling laws under the standard Chinchilla [Hof22b] protocol, on models from $\approx 125M$ to $\approx 1.3B$ parameters. **Mamba is the first attention-free model to match the performance of a very strong Transformer recipe (Transformer++) that has now become standard, particularly as the sequence length grows.** (We note that full results on context length 8k are missing for the RWKV and RetNet baselines, prior strong recurrent models that can also be interpreted as SSMs, because of a lack of efficient implementations leading to out-of-memory or unrealistic computation requirements.)

<span id="figure-04"></span>

![Language-model scaling laws](../../papers/mamba/figure-04.png)

**Figure 4.** (**Scaling Laws**.) Models of size $\approx 125M$ to $\approx 1.3B$ parameters, trained on the Pile. Mamba scales better than all other attention-free models and is the first to match the performance of a very strong "Transformer++" recipe that has now become standard, particularly as the sequence length grows.

<span id="section-4-2-2"></span>

#### 4.2.2 Downstream Evaluations

[Table 3](#table-03) shows the performance of Mamba on a range of popular downstream zero-shot evaluation tasks. We compare against the most well-known open source models at these sizes, most importantly Pythia [Bid23] and RWKV [Pen23g] which were trained with the same tokenizer, dataset, and training length (300B tokens) as our models. (Note that Mamba and Pythia are trained with context length 2048, while RWKV was trained with context length 1024.)

<span id="table-03"></span>

![Zero-shot language evaluation table](../../papers/mamba/table-03.png)

**Table 3.** (**Zero-shot Evaluations**.) Best results for each size in bold. We compare against open source LMs with various tokenizers, trained for up to 300B tokens. Pile refers to the validation split, comparing only against models trained on the same dataset and tokenizer (GPT-NeoX-20B). For each model size, Mamba is best-in-class on every single evaluation result, and generally matches baselines at twice the model size.

<span id="section-4-3"></span>

### 4.3 DNA Modeling

Motivated by the success of large language models, there has been recent exploration into using the foundation model paradigm for genomics. DNA has been likened to language in that it consists of sequences of discrete tokens with a finite vocabulary. It is also known for requiring long-range dependencies to model [Avs21]. We investigate Mamba as a FM backbone for pretraining and fine-tuning in the same setting as recent works on long-sequence models for DNA [Ngu23a]. In particular, we focus on two explorations of scaling laws across model size and sequence length ([Figure 5](#figure-05)), and a difficult downstream synthetic classification task requiring long context ([Figure 6](#figure-06)).

For pretraining, we largely follow a standard causal language modeling (next token prediction) setup for the training and model details (see also [Section 11.2](#section-11-2)). For the dataset, we largely follow the setup of HyenaDNA [Ngu23a], which uses the HG38 dataset for pretraining consisting of a single human genome with about 4.5 billion tokens (DNA base pairs) in the training split.

<span id="section-4-3-1"></span>

#### 4.3.1 Scaling: Model Size

In this experiment, we investigate the scaling properties of genomics foundation models with various model backbones ([Figure 5](#figure-05) *Left*).

**Training.** To advantage the baselines, we train on a short sequence length of $1024$; as shown in [Section 4.3.2](#section-4-3-2), we expect results to favor Mamba even more at longer sequence lengths. We fix a global batch size of $1024$, for a total of $2^{20} \approx 1M$ tokens per batch. Models were trained for $10K$ gradient steps for a total of $10B$ tokens.

**Results.** [Figure 5](#figure-05) (*Left*) shows that Mamba's pretraining perplexity improves smoothly with model size, and that Mamba scales better than both HyenaDNA and Transformer++. For example, at the largest model size of $\approx 40M$ parameters, the curve shows that **Mamba can match the Transformer++ and HyenaDNA models with roughly $3\times$ to $4\times$ fewer parameters**.

<span id="section-4-3-2"></span>

#### 4.3.2 Scaling: Context Length

In the next DNA experiment, we investigate the scaling properties of models with respect to sequence length. We only compare the HyenaDNA and Mamba models, as quadratic attention becomes prohibitively expensive at longer sequence lengths. We pretrain models on sequence lengths $2^{10}=1024$, $2^{12}=4096$, $2^{14}=16384$, $2^{16}=65536$, $2^{18}=262144$, $2^{20}=1048576$. We fix a model size of 6 layers by width $128$ (about 1.3M-1.4M parameters). Models were trained for $20K$ gradient steps for a total of $\approx 330B$ tokens. The longer sequence lengths used sequence length warmup similar to [Ngu23a].

**Results.** [Figure 5](#figure-05) (*Right*) shows that **Mamba is able to make use of longer context even up to extremely long sequences of length 1M**, and its pretraining perplexity improves as the context increases. On the other hand, the HyenaDNA model gets worse with sequence length. This is intuitive from the discussion in [Section 3.5](#section-3-5) on properties of the selection mechanism. In particular, LTI models cannot selectively ignore information; from a convolutional perspective, a very long convolution kernel is aggregating all information across a long sequence which may be very noisy. Note that while HyenaDNA claims to improve with longer context, their results do not control for computation time.

<span id="section-4-3-3"></span>

#### 4.3.3 Synthetic Species Classification

We evaluate models on a downstream task of classifying between 5 different species by randomly sampling a contiguous segment of their DNA. This task is adapted from HyenaDNA, which used the species $\{ \texttt{human}, \texttt{lemur}, \texttt{mouse}, \texttt{pig}, \texttt{hippo} \}$. We modify the task to be significantly more challenging by classifying between the five *great apes* species\
$\{ \texttt{human}, \texttt{chimpanzee}, \texttt{gorilla}, \texttt{orangutan}, \texttt{bonobo} \}$, which are known to share 99% of their DNA.

<span id="figure-05"></span>

![DNA scaling laws](../../papers/mamba/figure-05.png)

**Figure 5.** (**DNA Scaling Laws**.) Pretraining on the HG38 (human genome) dataset. (*Left*) Fixing short context length $2^{10}=1024$ and increasing size from $\approx200K$ to $\approx 40M$ parameters, Mamba scales better than baselines. (*Right*) Fixing model size and increasing sequence lengths while keeping tokens/batch and total training tokens fixed. Unlike baselines, the selection mechanism of Mamba facilitates better performance with increasing context length.

<span id="figure-06"></span>

![Great-apes DNA classification accuracy](../../papers/mamba/figure-06.png)

**Figure 6.** (**Great Apes DNA Classification**.) Accuracy after fine-tuning on sequences of length $2^{10}=1024$ up to $2^{20}=1048576$ using pretrained models of the same context length. Numerical results in [Table 13](#table-13).

<span id="figure-07"></span>

![Audio pretraining results](../../papers/mamba/figure-07.png)

**Figure 7.** (**Audio Pretraining**.) Mamba improves performance over prior state-of-the-art (Sashimi) in autoregressive audio modeling, while improving up to minute-long context or million-length sequences (controlling for computation).

<span id="section-4-4"></span>

### 4.4 Audio Modeling and Generation

For the audio waveform modality, we compare primarily to the SaShiMi architecture and training protocols [Goe22a].

This model comprises:

1.  a U-Net backbone with two stages of pooling by a factor $p$ that doubles the model dimension $D$ per stage,

2.  alternating S4 and MLP blocks in each stage.

We consider replacing the S4+MLP blocks with Mamba blocks.

Experiment details are in [Section 11.4](#section-11-4).

<span id="section-4-4-1"></span>

#### 4.4.1 Long-Context Autoregressive Pretraining

We evaluate pretraining quality (autoregressive next-sample prediction) on YouTubeMix [Dee17], a standard piano music dataset used by prior work consisting of $4$ hours of solo piano music, sampled at a rate of 16000 Hz. Pretraining details largely follow the standard language modeling setup ([Section 4.2](#section-4-2)). [Figure 7](#figure-07) evaluates the effect of increasing training sequence lengths from $2^{13}=8192$ to $2^{20}\approx 10^6$, while keeping computation fixed.

(There are some slight edge cases to the way the data is curated, which may lead to kinks in the scaling curves. For example, only minute-long clips were available so the maximum sequence length is actually bounded by $60s \cdot 16000Hz = 960000$.)

**Both Mamba and the SaShiMi (S4+MLP) baseline improve consistently with longer context lengths; Mamba is better throughout, and the gap widens at longer lengths.** The main metric is bits per byte (BPB), which is a constant factor $\log(2)$ of the standard negative log-likelihood (NLL) loss for pretraining other modalities.

We note one important detail: this is the only experiment in this paper in which we switched from the real parameterization to complex ([Section 3.6](#section-3-6)). We show additional ablations in [Section 11.4](#section-11-4).

<span id="section-4-4-2"></span>

#### 4.4.2 Autoregressive Speech Generation

SC09 is a benchmark speech generation dataset [War18, Don19b], consisting of $1$-second clips sampled at 16000 Hz of the digits "zero" through "nine" with highly variable characteristics. We largely follow the autoregressive training setup and generation protocol of [Goe22a].

[Table 4](#table-04) shows automated metrics of the Mamba-UNet model compared to a variety of baselines from [Goe22a]: WaveNet [Oor16], SampleRNN [Meh17], WaveGAN [Don19b], DiffWave [Kon21], and SaShiMi. **A small Mamba model outperforms the state-of-the-art (and much larger) GAN- and diffusion- based models.** A larger model parameter-matched to the baselines further improves on fidelity metrics dramatically.

[Table 5](#table-05) takes the small Mamba model and investigates combinations of different architectures for the outer stages and center stage. It shows that Mamba is consistently better than S4+MLP in the outer blocks, and Mamba $>$ S4+MLP $>$ MHA+MLP in the center blocks.

<span id="table-04"></span>

![SC09 generation metrics table](../../papers/mamba/table-04.png)

**Table 4.** (**SC09**) Automated metrics for unconditional generation on a challenging dataset of fixed-length speech clips. (*Top to Bottom*) Autoregressive baselines, non-autoregressive baselines, Mamba, and dataset metrics.

<span id="table-05"></span>

![SC09 model ablations table](../../papers/mamba/table-05.png)

**Table 5.** (**SC09 Model Ablations**) Models with 6M parameters. In SaShiMi's U-Net backbone, there are 8 center blocks operating on sequence length $1000$, sandwiched on each side by 8 outer blocks on sequence length $4000$, sandwiched by 8 outer blocks on sequence length $16000$ (40 blocks total). The architecture of the 8 center blocks are ablated independently of the rest. Note that Transformers (MHA+MLP) were not tested in the more important outer blocks because of efficiency constraints.

<span id="section-4-5"></span>

### 4.5 Speed and Memory Benchmarks

We benchmark the speed of the SSM scan operation (state expansion $N=16$), as well as the end-to-end inference throughput of Mamba, in [Figure 8](#figure-08). Our efficient SSM scan is faster than the best attention implementation that we know of (FlashAttention-2 [Dao24a]) beyond sequence length 2K, and up to 20-40$\times$ faster than a standard scan implementation in PyTorch. Mamba achieves 4-5$\times$ higher inference throughput than a Transformer of similar size, since without the KV cache it can use much higher batch sizes. For example, a Mamba-6.9B (untrained) would have higher inference throughput than a $5\times$ smaller Transformer-1.3B. Details in [Section 11.5](#section-11-5), which additionally includes a benchmark of memory consumption.

<span id="figure-08"></span>

![Selective-scan efficiency benchmarks](../../papers/mamba/figure-08.png)

**Figure 8.** (**Efficiency Benchmarks**.) (*Left*) Training: our efficient scan is $40\times$ faster than a standard implementation. (*Right*) Inference: as a recurrent model, Mamba can achieve $5\times$ higher throughput than Transformers.

<span id="section-4-6"></span>

### 4.6 Model Ablations

We perform a series of detailed ablations on components of our model, focusing on the setting of language modeling with size $\approx 350$M models at Chinchilla token counts (same setting as [Figure 4](#figure-04)).

<span id="section-4-6-1"></span>

#### 4.6.1 Architecture

[Table 6](#table-06) investigates the effects of the architecture (block) and its inner SSM layer ([Figure 3](#figure-03)). We find that

- Among previous non-selective (LTI) SSMs, which are equivalent to global convolutions, performance is very similar.

- Replacing the complex-valued S4 variant from previous work with a real-valued one does not affect performance much, suggesting that (at least for LM) real-valued SSMs may be a better choice when accounting for hardware efficiency.

- Replacing any of these with a selective SSM (S6) significantly improves performance, validating the motivation of [Section 3](#section-3).

- The Mamba architecture performs similarly to the H3 architecture (and seems slightly better when using a selective layer).

We also investigate interleaving the Mamba block with other blocks such as MLP (a traditional architecture) MHA (a hybrid attention architecture) in [Section 11.2.2](#section-11-2-2).

<span id="section-4-6-2"></span>

#### 4.6.2 Selective SSM

[Table 7](#table-07) ablates the selective SSM layer by considering different combinations of selective $\Delta$, $\bm{B}$, and $\bm{C}$ parameters ([Algorithm 2](#algorithm-02)), showing that $\Delta$ is the most important parameter due to its connection to RNN gating ([Theorem 1](#theorem-01)).

[Table 8](#table-08) considers different initializations of the SSM, which have been shown to make a large difference in some data modalities and settings [Gu22a, Gu22b]. On language modeling, we find that simpler real-valued diagonal initializations (S4D-Real, row 3) instead of more standard complex-valued parameterizations (S4D-Lin, row 1) perform better. Random initializations also work well, consistent with findings from prior work [Meh23].

[Table 9](#table-09) and [Table 10](#table-10) consider varying the dimension of the $\Delta$ and $(\bm{B}, \bm{C})$ projections respectively. Changing them from static to selective provides the most benefit, while increasing the dimensions further generally improves performance modestly with a small increase in parameter count.

<span id="table-06"></span>

![Architecture and SSM layer ablations table](../../papers/mamba/table-06.png)

**Table 6.** (**Ablations: Architecture and SSM layer**.) The Mamba block performs similarly to H3 while being simpler. In the inner layer, there is little difference among different parameterizations of LTI models, while selective SSMs (S6) provide a large improvement. More specifically, the S4 (real) variant is S4D-Real and the S4 (complex) variant is S4D-Lin.

<span id="table-07"></span>

![Selective-parameter ablations table](../../papers/mamba/table-07.png)

**Table 7.** (**Ablations: Selective parameters**.) $\Delta$ is the most important parameter ([Theorem 1](#theorem-01)), but using multiple selective parameters together synergizes.

<span id="table-08"></span>

![State-matrix parameterization ablations table](../../papers/mamba/table-08.png)

**Table 8.** (**Ablations: Parameterization of $\bm{A}$**.) The more standard initializations based on S4D-Lin [Gu22b] perform worse than S4D-Real or a random initialization, when the SSM is selective.

<span id="table-09"></span>

![Delta expressivity ablations table](../../papers/mamba/table-09.png)

**Table 9.** (**Ablations: Expressivity of $\Delta$**.) The selection mechanism of $\Delta$ constructs it with a projection of the input. Projecting it even to dim. $1$ provides a large increase in performance; increasing it further provides further improvements at the cost of a modest increase in parameters. State size fixed to $N=16$.

<span id="table-10"></span>

![SSM state-dimension ablations table](../../papers/mamba/table-10.png)

**Table 10.** (**Ablations: SSM state dimension**.) (*Top*) Constant $\bm{B}$ and $\bm{C}$ (*Bottom*) Selective $\bm{B}$ and $\bm{C}$. Increasing the SSM state dimension $N$, which can be viewed as an expansion factor on the dimension of the recurrent state, can significantly improve performance for a negligible cost in parameters/FLOPs, but only when $\bm{B}$ and $\bm{C}$ are also selective. Size of $\Delta$ projection fixed to $64$.

Of particular note is the dramatic improvement of the selective SSM when the state size $N$ is increased, with over a 1.0 perplexity improvement for a cost of only 1% additional parameters. This validates our core motivation in [Section 3.1](#section-3-1) and [Section 3.3](#section-3-3).

<span id="section-5"></span>

## 5 Discussion

We discuss related work, limitations, and some future directions.

**Related Work.** [Section 7](#section-7) discusses how the selection mechanism relates to similar concepts. [Section 8](#section-8) has an extended related work of SSMs and other related models.

**No Free Lunch: Continuous-Discrete Spectrum.** Structured SSMs were originally defined as discretizations of continuous systems [Equation 1](#equation-01), and have had a strong inductive bias toward continuous-time data modalities such as perceptual signals (e.g. audio, video). As discussed in [Section 3.1](#section-3-1) and [Section 3.5](#section-3-5), the selection mechanism overcomes their weaknesses on discrete modalities such as text and DNA; but this conversely can impede their performance on data that LTI SSMs excel on. Our ablations on audio waveforms examine this tradeoff in more detail.

**Downstream Affordances.** Transformer-based foundation models (particularly LLMs) have a rich ecosystem of properties and modes of interaction with pretrained models, such as fine-tuning, adaptation, prompting, in-context learning, instruction tuning, RLHF, quantization, and so on. We are particularly interested in whether Transformer alternatives such as SSMs have similar properties and affordances.

**Scaling.** Our empirical evaluation is limited to small model sizes, below the threshold of most strong open source LLMs (e.g. Llama [Tou23]) as well as other recurrent models such as RWKV [Pen23g] and RetNet [Sun23a], which have been evaluated at the 7B parameter scale and beyond. It remains to assess whether Mamba still compares favorably at these larger sizes. We also note that scaling SSMs may involve further engineering challenges and adjustments to the model that are not discussed in this paper.

<span id="section-6"></span>

## 6 Conclusion

We introduce a selection mechanism to structured state space models, allowing them to perform context-dependent reasoning while scaling linearly in sequence length. When incorporated into a simple attention-free architecture, Mamba achieves state-of-the-art results on a diverse set of domains, where it matches or exceeds the performance of strong Transformer models. We are excited about the broad applications of selective state space models to build foundation models for different domains, especially in emerging modalities requiring long context such as genomics, audio, and video. Our results suggest that Mamba is a strong candidate to be a general sequence model backbone.

## Acknowledgments

We thank Karan Goel, Arjun Desai, and Kush Bhatia for helpful feedback on the draft.

<span id="section-7"></span>

## 7 Discussion: Selection Mechanism

Our selection mechanism is inspired by and related to concepts such as gating, hypernetworks, and data-dependence. It can also be viewed as related to "fast weights" [Sch92, Ba16a], which connects classical RNNs with the mechanism of linear attention [Sch21]. However, we believe that it is a distinct concept that is worth clarifying.

**Gating.** Gating originally referred to the gating mechanisms of RNNs such as the LSTM [Hoc97] and GRU [Chu14], or the gated equation [Equation 5](#equation-05) in [Theorem 1](#theorem-01). This was interpreted as a particular mechanism for controlling whether to let an input into the hidden state of an RNN. In particular, this affects the propagation of signal through time and causes inputs to interact along the sequence length dimension.

However, the concept of gating has since been relaxed in popular usage to simply mean any multiplicative interaction (often with an activation function). For example, *elementwise* multiplicative components of neural network architectures (that do not interact along sequence length) are now commonly referred to as gated architectures [Hua22, Meh23], despite a very different meaning than the original RNN sense. Thus we believe the original concept of *RNN gating* versus the popular usage of *multiplicative gating* actually have a very different semantic meaning.

**Hypernetworks.** Hypernetworks refer to neural networks whose parameters are themselves generated by smaller neural networks. The original idea [Ha17] used it in a narrow sense to define a large RNN whose recurrent parameters are generated by a smaller RNN, and other variants have been around for a long time [Sch92].

**Data-dependence.** Similar to hypernetworks, data-dependence can refer to any notion where some parameters of the model depend on the data [Pol23a].

**Example: GLU Activation.** To illustrate the issues with these concepts, consider a simple diagonal linear layer $y = \bm{D}x$, where $\bm{D}$ is a diagonal weight parameter. Now suppose that $\bm{D}$ is itself generated from a linear transformation of $x$, with an optional nonlinearity: $\bm{D} = \sigma(\bm{W} x)$. Since it is diagonal, the multiplication becomes an elementwise product: $y = \sigma(\bm{W} x) \circ x$.

This is a rather trivial transformation, yet it technically satisfies the common meanings of gating (since it has a multiplicative "branch"), hypernetworks (since the parameter $\bm{D}$ is generated by another layer), and data-dependent (since $\bm{D}$ depends on the data $x$). However, this in fact simply defines a GLU function, which is so simple that it is often considered just an activation function [Dau17, Sha20] instead of a meaningful layer.

**Selection.** Thus, while selection mechanisms could be considered a special case of ideas such as architectural gating, hypernetworks, or data-dependence, so can an enormous range of other constructions—essentially anything with a multiplication, including standard attention mechanisms [Bah15, Vas17] as well—and we find it uninformative to think of them as such.

Instead, we view it as most closely related to the gating mechanism of traditional RNNs, which is a special case ([Theorem 1](#theorem-01)) and also has a deeper history of connections to SSMs through variable (input-dependent) discretization of $\Delta$ [Fun93, Tal18a, Gu20a]. We also eschew the term "gating" in favor of *selection* to clarify the overloaded use of former. More narrowly, we use selection to refer to the *mechanistic* action of a model to select or ignore inputs and facilitate data interaction along the sequence length ([Section 3.1](#section-3-1)). Beyond selective SSMs and gated RNNs, other examples may include input-dependent convolutions [Yan19c, Lio20, Kos23, Lut23] and even attention.

<span id="section-8"></span>

## 8 Related Work

We overview several prior works related to our methods. We mention that some of the most closely related models include recurrent layers such as S4, S5, and quasi-RNNs; as well as end-to-end architectures such as H3, RetNet, and RWKV.

<span id="section-8-1"></span>

### 8.1 S4 Variants and Derivatives

We describe a brief overview of some structured SSMs from past work, particularly those that have a relation to our method.

- S4 [Gu21a, Gu22a] introduced the first structured SSM, describing diagonal structure and diagonal plus low-rank (DPLR). It focused on efficient convolutional algorithms for DPLR SSMs due to a connection to continuous-time online memorization (HIPPO) [Gu20a].

- DSS [Gup22] first discovered the empirical effectiveness of diagonal structured SSMs by approximating the HIPPO initialization. This was expanded on theoretically in S4D [Gu22b].

- S5 [Smi23] independently discovered the diagonal SSM approximation, and is the first S4 model to be computed recurrently with the parallel scan. However, this required lowering the effective state dimension, which they accomplished by switching the SSM dimensions from a SISO (single-input single-output) to MIMO (multi-input multi-output) formulation. Our proposed S6 shares the scan, but differs by (i) keeping the SISO dimensions, which provides a larger effective recurrent state, (ii) using a hardware-aware algorithm to overcome the computation issue, (iii) adding the selection mechanism.

  [Lu23a] applied S5 to meta-RL in order to handle resetting the SSM state between episode trajectories. Their mechanism can be viewed as a particular hard-coded instance of a selection mechanism, where $\overline{\bm{A}}$ is manually set to $0$, instead of our learnable mechanism that depends on the input. It would be interesting to apply selective SSMs generically to this setting and probe if the model has learned to automatically reset its state on episode boundaries.

- Mega [Ma23b] introduced a simplification of S4 to be real- instead of complex- valued, giving it an interpretation of being an exponential moving average (EMA). They additionally make an interesting connection of the discretization step of SSMs to an EMA *damping* term. Contrary to findings in the original S4 papers, this was the first model to show that real-valued SSMs are empirically effective in certain settings or when combined with different architectural components.

- Liquid S4 [Has23] is also motivated by augmenting S4 with an input-dependent state transition. From this perspective it shares similarity to selection mechanisms, although in a limited form which is still computed convolutionally and close to LTI.

- SGConv [Li23y], Hyena [Pol23a], LongConv [Fu23b], MultiresConv [Shi23e], and Toeplitz Neural Network [Qin23d] all focus on the convolutional representation of S4 and create global or long convolution kernels with different parameterizations. However, these methods cannot do fast autoregressive inference directly.

Notably, all of these methods, and all other structured SSMs that we are aware of, have been non-selective and usually strictly LTI (linear time invariant).

<span id="section-8-2"></span>

### 8.2 SSM Architectures

We use SSM architectures or state space neural networks (SSNN) to refer to deep neural network architectures incorporating one of the previous SSMs as a black box layer.

- GSS [Meh23] was the first gated neural network architecture incorporating SSMs. It is motivated by the gated attention unit (GAU) of [Hua22] and looks quite similar to our block, except with additional projections. Most importantly, its projection *contracts* the model dimension to reduce the state size of the SSM, while ours *expands* the model dimension in order to increase the state size, based on the motivation in [Section 3.1](#section-3-1).

- Mega [Ma23b] combined the EMA simplification of S4 described above into a hybrid architecture using an efficient attention approximation.

- H3 [Dao23d] is motivated by combining S4 with linear attention [Kat20]. It is the first to generalize this formulation of linear attention to more general recurrences, which is also the basis of later architectures.

- Selective S4 [Wan23l] incorporates S4 as a black box to generate a binary mask which is multiplied on the input. While sharing the "selection" name, we consider this an architectural modification that is closer to architectural gating than a selection mechanism ([Section 7](#section-7)). For example, we hypothesize that it would not solve the Selective Copying task because simply masking out the irrelevant inputs does not affect the spacing between the relevant ones (indeed, the Selective Copying task can even be viewed as coming pre-masked if the noise tokens are embedded to 0).

- RetNet [Sun23a] is also based on Linear Attention and very similar to H3, but reduces the inner S4 layer to a special case where the state dimension is $N=1$. Although not framed as such, its recurrence can be viewed as a special case of a linear SSM.

  Its primary source of improvement is using a linear attention with large *head dimension*, which can be viewed as another method to perform input-dependent state expansion. Using a larger head dimension in the context of linear attention variants was first done by H3, but not extensively used since this requires a proportional amount of extra computation. RetNet avoids this with an alternate way to parallelize the computation with a variant of standard multi-head attention instead of convolutions, made feasible by their particular special case of SSMs which acts as a simple EMA.

- RWKV [Pen23g] is another recent RNN designed for language modeling. It is based on AFT (attention-free Transformer [Zha21e]), another variant of linear attention. Its main "WKV" mechanism involves LTI recurrences and can be seen as the ratio of two SSMs.

We also highlight the gated attention unit (GAU) from [Hua22], which was motivated by combining the Transformer's MHA and MLP blocks together and was an inspiration for our architecture ([Section 3.4](#section-3-4)) combining the H3 and MLP blocks.

<span id="section-8-3"></span>

### 8.3 Relationship to RNNs

RNNs and SSMs are broadly related, as they both involve the concepts of *recurrence* on a latent *state*.

Several older RNNs such as the strongly typed RNN [Bal16], quasi-RNN (QRNN) [Bra16], and simple recurrent unit (SRU) [Lei17, Lei21] involve forms of gated RNNs without time-wise nonlinearities. Because of the connections of gating mechanisms and selection mechanisms, these can be viewed as cases of selective SSMs, and are thus more powerful in a sense than the family of LTI structured SSMs above. The main differences are:

- They do not use state expansion ($N=1$) or selective $\bm{B}, \bm{C}$ parameters, both of which are important for performance ([Section 4.6](#section-4-6)).

- They use a heuristic gating mechanism, which we generalize as a consequence of the selection mechanism + discretization ([Theorem 1](#theorem-01)). The connections to principled SSM theory provides better parameterizations and initializations ([Section 3.6](#section-3-6)).

Additionally, older RNNs famously suffered from efficiency issues and the vanishing gradients problem [Hoc91, Hoc01, Pas13], both caused by their sequential nature. The former could be solved for some of the above RNNs by leveraging the parallel scan [Mar18], but the latter was difficult without theory later developed for SSMs. For example, modern structured SSMs differ in more careful parameterization of the recurrent dynamics inspired by classical SSM theory (e.g. through discretization [Gu21a, Gu23a]), or direct analysis [Orv23, Kau20, Gup22a]).

We also note that there is a long line of work on orthogonal RNNs [Arj16, Hen16c, Mha17, Vor17, Lez19] which are motivated by constraining the $\overline{\bm{A}}$ transition matrix to be orthogonal or unitary, in order to control its eigenvalues and prevent the vanishing gradient problem. However, these had other limitations; we believe that these stem from the fact that orthogonal/unitary RNNs are also LTI. For example, they are almost always evaluated on the Copying task which they can solve perfectly, but observed to struggle on the Selective Copying task [Jin19].

<span id="section-8-4"></span>

### 8.4 Linear Attention

The Linear Attention (LA) [Kat20] framework is an important result popularizing kernel attention and showing how it relates to recurrent autoregressive models. Many variants have proposed alternative kernels and other modifications. Random Feature Attention (RFA) [Pen21] chooses the kernel feature map to approximate softmax attention (i.e. the $\exp$ feature map) using the random Fourier feature approximation of Gaussian kernels [Rah07]. Performer [Cho21] finds an approximation to the exponential kernel involving only positive features, which also allows the softmax normalization term. TransNormer [Qin22a] showed that the LA denominator term can be unstable and proposed replacing it with a LayerNorm. cosFormer [Qin22b] augments RFA with a cosine reweighting mechanism that incorporates positional information to emphasize locality. Linear Randomized Attention [Zhe22b] generalize RFA from the perspective of importance sampling, and generalize it to provide better estimates of the full softmax kernel (rather than just the $\exp$-transformed numerator).

Aside from kernel attention, many other variants of efficient attention exist; the survey [Tay22a] offers an extensive categorization of many of these.

<span id="section-8-5"></span>

### 8.5 Long Context Models

Long context has become a popular subject, and several recent models have claimed to scale to longer and longer sequences. However, these are often from a computational standpoint and have not been extensively validated. These include:

- Recurrent Memory Transformer [Bul23], a lightweight wrapper around a Transformer backbone. It showed ability to generalize up to 1M sequences but only on synthetic memorization tasks; their main result is similar to our Induction Heads extrapolation experiment ([Table 2](#table-02)).

- LongNet [Din23a], which claimed to scale to 1B length but only evaluated on length $<100K$ for actual tasks.

- Hyena and HyenaDNA [Pol23a, Ngu23a], which claimed to leverage up to 1M context. However, their experiments trained on proportionally more data at longer contexts, making it hard to conclude if quality improvements at 1M context are due to context length or due to more data and computation.

- Sparse Transformer [Chi19] showed a proof-of-concept of using a strided sparse attention Transformer to model audio waveforms of length $2^{20}=1048576$, although did not discuss performance tradeoffs when controlling for computation and model size.

In contrast, we believe this work presents one of the first approaches to meaningfully demonstrate increasing performance with longer context.

<span id="section-9"></span>

## 9 Mechanics of Selective SSMs

::: details Proof

Consider a selective SSM ([Algorithm 2](#algorithm-02)) with $N=1, \bm{A}=-1, \bm{B}=1, s_\Delta=\mathrm{Linear}(x), \tau_\Delta=\mathrm{softplus}$. The corresponding continuous-time SSM [Equation 1](#equation-01) is

$$
\begin{aligned}

  h(t) = -h(t) + x(t)
\end{aligned}
$$
which is also called a *leaky integrator*.

The discretization step size is

$$
\begin{aligned}

  \Delta_t &= \tau_\Delta(\mathrm{Parameter} + s_\Delta(x_t)) \\
      &= \mathrm{softplus}(\mathrm{Parameter} + \mathrm{Linear}(x_t)) \\
      &= \mathrm{softplus}(\mathrm{Linear}(x_t))
\end{aligned}
$$
where we observe that the parameter can be viewed as a learnable bias and folded into the linear projection.

Now applying the zero-order hold (ZOH) discretization formulas:

$$
\begin{aligned}

  \overline{\bm{A}}_t &= \exp(\Delta\bm{A}) = \frac{1}{1 + \exp(\mathrm{Linear}(x_t))} = \sigma(-\mathrm{Linear}(x_t))
    \\&= 1 - \sigma(\mathrm{Linear}(x_t))
    \\
  \overline{\bm{B}}_t &= (\Delta\bm{A})^{-1} (\exp(\Delta\bm{A}) - \bm{I}) \cdot \Delta\bm{B} = -(\exp(\Delta\bm{A}) - \bm{I}) = 1 - \overline{\bm{A}}
    \\&= \sigma(\mathrm{Linear}(x_t))
    .
\end{aligned}
$$

Thus the final discrete recurrence [Equation 2a](#equation-02-a) is

$$
\begin{aligned}

  g_t &= \sigma(\mathrm{Linear}(x_t)) \\
  h_{t} &= (1-g_t) h_{t-1} + g_t x_t
\end{aligned}
$$
as desired.

:::
<span id="section-10"></span>

## 10 Hardware-aware Algorithm For Selective SSMs

Without input-dependent selectivity, SSMs can be efficiently implemented as a convolution [Gu22a, Dao23d], which leverages the fast Fourier transform (FFT) as primitive. With selectivity, SSMs are no-longer equivalent to convolution, but we leverage the parallel associative scan. While SSM scans are theoretically efficient ($O(B L D N)$ FLOPs, scaling linear in $L$), training foundation models with selective SSMs requires them to be efficient on modern hardware (GPUs) as well. We describe how we use *kernel fusion* and *recomputation* to make SSM scan fast and memory-efficient. We evaluate the speed of our scan implementation compared to convolution and attention in [Section 4.5](#section-4-5), showing that it is up to 7$\times$ times faster than attention at sequence length 32K, and is as memory-efficient as the best attention implementation (FlashAttention).

**Speed.** On modern hardware accelerators (GPUs) most operations (except matrix multiply) are bounded by memory-bandwidth [Wil09, Iva21, Dao22]. This the case with our scan operation, and we use kernel fusion to reduce the amount of memory IOs, leading to significant speedup compared to a standard implementation.

The standard way to implement the scan algorithm in [Section 3.2](#section-3-2) is to prepare the scan input $\overline{\bm{A}}, \overline{\bm{B}}$ of size $(B, L, D, N)$ in GPU HBM (high-bandwidth memory, commonly referred to as GPU memory), call a parallel associative scan implementation to write the scan output of size $(B, L, D, N)$ to GPU HBM, then multiply that scan output with $\bm{C}$ to produce an output of size $(B, L, D)$. However, this requires the number of memory reads/writes on the order of $O(B L D N)$. We can instead fuse the discretization step, the scan, and the multiplication with $\bm{C}$ into one kernel:

1.  We read in $O(B L D + D N)$ bytes of memory ($\Delta, \bm{A}, \bm{B}, \bm{C}$) from slow HBM to fast SRAM.

2.  We discretize to produce $\overline{\bm{A}}, \overline{\bm{B}}$ of size $(B, L, D, N)$ in SRAM.

3.  We perform a parallel associative scan, yielding intermediate states of size $(B, L, D, N)$ in SRAM.

4.  We multiply and sum with $\bm{C}$, producing outputs of size $(B, L, D)$ and write it to HBM.

This way, we reduce IOs by a factor of $O(N)$ (the state dimension), which in practice speeds up the operation by 20-40 times ([Section 4.5](#section-4-5)).

For sequence length $L$ too long where we cannot fit the sequence in SRAM (which is much smaller than HBM), we split the sequences into chunks and perform the fused scan on each chunk. As long as we have the intermediate scan states, we can continue the scan with the next chunk.

**Memory.** We describe how we use the classical technique of *recomputation* to reduce the total amount of memory required to train selective SSM layers.

From the way we fuse the forward pass, we do not save the intermediate states of size $(B, L, D, N)$ to avoid memory blowup. However, these intermediate states are necessary for the backward pass to compute gradients. We instead recompute those intermediate states in the backward pass. Since the inputs $\Delta, \bm{A}, \bm{B}, \bm{C}$ and output gradient read from HBM to SRAM are of size $O(B L N + D N)$, and the input gradients are also of size $O(B L N + D N)$, recomputation avoids the cost of reading $O(B L N D)$ elements from HBM. This means that recomputation of the SSM states in the backward pass speeds up the computation compared to storing them and reading them from HBM.

Beyond optimizing for the memory requirement of just the scan operation, we also use recomputation to optimize the memory requirement of the entire selective SSM block (input projection, convolution, activation, scan, output projection). In particular, we do not save intermediate activations that take a lot of memory but are fast to recompute (e.g. output of activation function or short convolution). As a result, the selective SSM layer has the same memory requirement as an optimized Transformer implementation with FlashAttention. In particular, each attention layer (FlashAttention) stores around 12 bytes of activations per token, an each MLP layer stores around 20 bytes of activations per token, for a total of 32 bytes ((assuming mixed-precision training in FP16 or BF16)). Each selective SSM stores around 16 bytes of activations per token. Hence two layers of selective SSMs have around the same activation memory as an attention layer and an MLP layer.

<span id="section-11"></span>

## 11 Experimental Details and Additional Results

<span id="section-11-1"></span>

### 11.1 Synthetic Tasks

**Selective Copying.** Our setting is on sequences of length 4096, with a vocab size of 16 possible tokens (including the white "noise" token from [Figure 2](#figure-02)) and requiring models to memorize 16 "data" tokens. We use 2 layer models with a model dimension of $D = 64$.

Models are trained for 400K steps at a constant learning rate of $0.0001$ with a batch size of $64$.

**Induction Heads.**

<span id="table-11"></span>

![Induction-head extrapolation results table](../../papers/mamba/table-11.png)

**Table 11.** (**Induction heads**.) Models are trained on sequence length $2^8=256$, and tested on various sequence lengths of $2^6=64$ up to $2^{20}=1048576$. ✓ denotes perfect generalization accuracy, while ✗ denotes out of memory.

Training consists of randomly generating data every step, with a batch size of $8$. We choose an "epoch" size of 8192 steps, and track the accuracy on fixed validation sets (also randomly generated) of each target sequence length. For the MHA-Abs and Mamba models, results are reported after the 25th epoch ($8192 \times 25 = 204800$ steps). For the MHA-RoPE and MHA-xPos models, results are reported after the 50th epoch ($8192 \times 50 = 409600$ steps). For the LTI H3 and Hyena models, results are reported after the 10th epoch ($81920$ steps) because they had converged by then and failed to improve further.

We use the Adam optimizer with no weight decay. All models are trained at constant learning rates $2e-4$ and $1e-3$, and the better results are reported for each model ($2e-4$ for all models except Mamba). The attention and Hyena models did not learn at LR $1e-3$. H3 learned at both LRs, but interestingly generalized better to shorter sequences at the smaller LR of $2e-4$. Mamba learned at both LRs, but extrapolated better at the larger LR of $1e-3$.

<span id="section-11-2"></span>

### 11.2 Language Modeling

<span id="section-11-2-1"></span>

#### 11.2.1 Scaling Law Details

Scaling law experiments generally followed the GPT3 recipe. All models were trained on the Pile with the GPT2 tokenizer.

**Model Sizes.** [Table 12](#table-12) specifies the model sizes we use for scaling laws. This is taken directly from the GPT3 specifications [Bro20], with very minor modifications. First, we changed the batch size of the 1.3B model from 1M tokens to 0.5M tokens, since we did not use enough parallelization to require the larger batch size. Second, we changed the number of training steps and total tokens to roughly match Chinchilla scaling laws [Hof22b], which specify that training tokens should increase proportionally to model size.

<span id="table-12"></span>

![Scaling-law model sizes table](../../papers/mamba/table-12.png)

**Table 12.** (**Scaling Law Model Sizes**.) Our model sizes and hyperparameters for scaling experiments. (Model dimension and number of heads applies only to Transformer models.)

**Training Recipes.** All models used the AdamW optimizer with

- gradient clip value $1.0$

- weight decay $0.1$

- no dropout

- linear learning rate warmup with cosine decay

By default, the peak learning rate is the GPT3 specification.

We give several models an "improved recipe", inspired by changes adopted by popular large language models such as PaLM [Cho23a] and LLaMa [Tou23]. These include:

- linear learning rate warmup with cosine decay to $1e-5$, with a peak value of $5\times$ the GPT3 value

- no linear bias terms

- RMSNorm instead of LayerNorm

- AdamW hyperparameter $\beta=(.9, .95)$ (the GPT3 value) instead of the PyTorch default of $\beta=(.9, .999)$

**Architecture and Training Details.** Our models are:

- **Transformer**: The standard Transformer based on GPT3 ([Table 12](#table-12)).

- **Transformer++**: A Transformer with an improved architecture, namely rotary positional encodings [Su21] and SwiGLU MLP [Sha20], and the improved training recipe above.

- **Hyena**: Interleaving a Hyena block (the H3 block with S4 replaced by a global convolution parameterized by an MLP) with standard MLP blocks. The MLP blocks have expansion factor $2$ instead of $4$ and the number of layers is correspondingly increased by $1.5\times$ to preserve parameter count.

- **H3++**: The H3 architecture with a few modifications, including (i) using the same "thin" Hyena dimensions above (ii) the improved training recipe above (iii) a linear attention *head dimension* of 8.

- **RWKV**: The default RWKV model from [Pen23g], including its modified MLP block. We also used as much of its specified training recipe as possible, such as increasing the learning rates by $2\times$ or $3\times$ on certain parameters.

- **RetNet**: The default RetNet model from [Sun23a]. We also gave it the improved training recipe above.

- **Mamba**: The standard Mamba architecture, with the improved training recipe.

<span id="section-11-2-2"></span>

#### 11.2.2 Additional Scaling Law Ablations

We perform additional ablations on the architecture using the same protocol as the 2k context length scaling laws in [Figure 4](#figure-04) (*Left*).

**Mamba Architecture: Interleaving Blocks.** We test the effect of different architectural blocks combined with the Mamba block. We focus on the viewpoint that the Mamba block is simply the standard SwiGLU block with an extra $\mathrm{conv} \to \mathrm{SSM}$ path added. This leads to two natural ablations:

- What if the Mamba block is interleaved with a standard MLP block, instead of stacked homogenously? This can also be interpreted as taking Mamba and removing half of the SSMs.

- What if the Mamba block is interleaved with MHA (multi-head attention) blocks? This can also be interpreted as taking a Transformer with SwiGLU MLPs (i.e. what we call Transformer++) and simply adding SSMs to the MLP blocks.

[Figure 9](#figure-09) (*Right*) shows these variants compared to the original (homogenous) Mamba architecture. Interestingly, neither change matters too much. The Mamba-MLP architecture is only slightly worse, and still better than all models except Transformer++. The Mamba-MHA architecture is only slightly better, which is somewhat surprising in light of the fact that many recent works have found that combining (LTI) SSMs with Attention can lead to substantial improvements [Dao23d, Fat23a, Sao23, Zuo22, Fat23].

**H3 Architecture: Training Recipes.** Next we ablate differences between the Hyena and H3++ models, our weakest and strongest models outside of Transformer++ and Mamba, particularly to isolate the effect of training recipes.

- **Hyena**: The Hyena block with its original architecture and GPT3 training recipe (same as [Figure 4](#figure-04)).

- **Hyena+**: The same architecture but with the improved training recipe described above.

- **H3+**: The same architecture as Hyena+ but with the Hyena convolution kernel swapped out for S4D convolution kernel.

- **H3++**: The same as H3+, but with a linear attention *head dimension* of 8. This increases computation inside the SSM recurrence but does not increase parameters.

Our general convention is that "Model+" represents the base model with the improved training recipe, and "Model++" also allows for architectural changes.

[Figure 9](#figure-09) (*Right*) shows that

- A large improvement is achieved by the improved training recipe, which was used for many of the models in the main [Figure 4](#figure-04) (RetNet, H3++, Transformer++, Mamba).

- The choice of the inner LTI SSM does not matter (e.g. Hyena vs. S4), consistent with findings throughout this paper.

- The head dimension expansion improves performance, consistent with one of our main themes that expanded state dimension improves performance for SSMs ([Section 3](#section-3)).

<span id="figure-09"></span>

![Additional language-model scaling ablations](../../papers/mamba/figure-09.png)

**Figure 9.** (**Scaling laws: extra ablations**.) (*Left*) Instead of (*Right*) Instead of

<span id="section-11-2-3"></span>

#### 11.2.3 Downstream Evaluation Details

This pretraining procedure is the same as the scaling law protocol, but extended to 300B tokens and with the GPT-NeoX tokenizer [Bla22] instead of GPT2 tokenizer. For the 1.3B model, we use a batch size of 1M tokens to be consistent with the GPT3 specifications. We report the perplexity on the Pile validation set, and for this metric only compare to models trained on the same dataset and with the same tokenizer, in particular Pythia and RWKV.

For downstream evaluation, we use the LM evaluation harness from EleutherAI [Gao21], as done by most work in this area. We evaluate on the following tasks/datasets that measure common sense reasoning:

- LAMBADA [Pap16a]

- HellaSwag [Zel19]

- PIQA [Bis20]

- ARC-challenge [Cla18]

- ARC-easy: an easy subset of ARC-challenge

- WinoGrande [Sak21]

We report accuracy for LAMBADA, WinoGrande, PIQA, and ARC-easy, and accuracy normalized by sequence length for HellaSwag and ARC-challenge (since normalized accuracy is higher for almost all models for these task).

<span id="section-11-3"></span>

### 11.3 DNA Modeling

<span id="section-11-3-1"></span>

#### 11.3.1 Pretraining Details

We describe the dataset and training procedure of the HG38 pretraining task in more detail.

The dataset follows the splits from the prior Enformer work on genomics [Avs21]; the training split contains a total of $S=34021$ segments of length $2^{17}=131072$ that cover the genome, for a total of approximately 4.5 billion tokens (DNA base pairs). These segments are pairs of (chromosome number, starting index, ending index), and can be extended if necessary (e.g. to get longer segments).

We deviate from HyenaDNA when the training sequence length is not $2^{17}$. HyenaDNA always takes a fixed sub-segment (e.g. the beginning or middle of the prescribed segment), and thus for any training sequence length each epoch is fixed to $34021$ samples and doesn't necessarily go through the whole genome. On the other hand, we use the entire training data:

- When the context length $L$ is less than (or equal to) $2^{17}$, we divide up each segment into non-overlapping sub-segments of length $L$, so that there are $S \times \frac{2^{17}}{L}$ total samples and $S \times 2^{17} \approx 4.5B$ tokens per epoch.

- When the context length $L$ is greater than $2^{17}$, we turn each segment into two samples, one that begins with the prescribed segment and one that ends with the prescribed segment. Thus each epoch has $2S$ items and $2 S L$ tokens per epoch. For example, at sequence length $2^{18}=262144$ there are $4\times$ as many tokens as the default, and at sequence length $2^{20}$ there are $16\times$ as many tokens.

Other training details generally follow the same protocol as our language modeling experiments ([Section 11.2](#section-11-2)). For example, we use the AdamW with $(\beta_1, \beta_2) = (0.9, 0.95)$, no dropout, weight decay $0.1$. We use a cosine learning rate scheduler with linear warmup for 10% of total steps.

<span id="section-11-3-2"></span>

#### 11.3.2 Scaling: Model Size Details

**Models.** The models we consider are:

- Transformer++: a Transformer with improved architecture, notably the usage of RoPE positional encodings [Su21]. Informally, we found these to be noticeably better than vanilla positional encodings from [Vas17].

- HyenaDNA: the Hyena model from [Pol23a, Ngu23a], which is roughly a Transformer with the MHA block replaced by an H3 block using a global convolution parameterized by an MLP.

- Mamba: the standard Mamba architecture.

**Model Sizes.** We use the following model sizes.

![DNA scaling model sizes](../../papers/mamba/model-sizes.png)

**Model sizes.**
Note that the number of blocks for Mamba is doubled, because one Transformer "layer" includes both the MHA and MLP blocks (and similarly for Hyena), which requires two Mamba blocks to match parameters ([Section 3.4](#section-3-4)).

**Training.** For each model (Transformer++, HyenaDNA, Mamba), we swept the learning rate across $\{1e-3, 2e-3, 4e-3, 8e-3\}$. The optimal Transformer and HyenaDNA learning rates were 2e-3 across all sizes. The optimal Mamba learning rate was 8e-3; note that Mamba performed better than baselines with matched learning rates (2e-3), but was more stable and improved even more at higher learning rates. (Furthermore, as this LR is on the upper range of the sweep, it is possible that our results are still suboptimal.)

Note that, in contrast to standard LM scaling laws ([Table 12](#table-12)), our LR held constant across model sizes for simplicity. The optimal LR should go down for larger models, but we didn't find a noticeable effect at the small model sizes (at most a few million parameters) we considered.

<span id="section-11-3-3"></span>

#### 11.3.3 Scaling: Context Length Details

We use a total batch size of $2^{24}\approx 16M$ tokens per training step, for every sequence length (e.g. at length $2^{20}$ there are $16$ segments per batch and at length $2^{10}$ there are $16384$ segments per batch). This is a large batch size relative to the model size by usual LM standards, but note that a batch size of $2^{23}$ is the minimum possible on a machine with 8 GPUs and sequence length of $2^20$, and that HyenaDNA used much larger batches of $2^{28}$.

The learning rate used was $0.008$ for Mamba and 0.001 for HyenaDNA; we initially attempted to use the same learning rate of $0.002$ from the previous section for HyenaDNA, but found that it was unstable at the longest context length.

**Sequence Length Warmup.** Following [Ngu23a], we use sequence length warmup (SLW) during pretraining. We choose a simple schedule of 2 epochs at each power-of-two sequence length starting from $2^{10}=1024$. (Note that because of how data is curated, at the longest sequence lengths more steps and tokens are spent proportionally. In particular, each stage up to length $2^{17}$ processes the same number of tokens, but $4\times$ as many tokens are processed at length $2^{18}$, $8\times$ as many at length $2^{19}$, and $16\times$ as many at length $2^{20}$.)

Unlike HyenaDNA, we always control for the number of tokens per gradient update, so the batch size is successively halved as the sequence lengths are doubled in each stage.

**Remark.** We also note that the schedule was not tuned, and we never experimented with turning off sequence length warmup for these pretraining experiments. We later found that SLW did not help noticeably for audio pretraining at similar lengths ([Section 4.4](#section-4-4)), and it is possible that it is not necessary for DNA pretraining either.
<span id="section-11-3-4"></span>

#### 11.3.4 Species (Great Apes) Classification

Models are causal and therefore only the last element (across the sequence length) of the model's output is used for the classification head. Note that we control for the total number of elements in the loss function per gradient step. The pretraining objective includes all positions across the sequence length, so that $\mathrm{batch\_size} \times \mathrm{sequence\_length}$ is held constant; in other words, the batch size decreases as the sequence length increases. However, for a classification task, since only the last position enters the loss, the batch size itself is held constant. Note that this also means that fine-tuning models with longer sequence lengths is more computationally expensive.

Training consists of 10 epochs, each of which has 1024 gradient steps. Each gradient step uses batch size 64, which are all independently randomly drawn by uniformly picking a species, uniformly picking a chromosome, and then uniformly picking a contiguous segment of DNA.

Following [Ngu23a], models with a maximum context length greater than $2^{14} = 16384$ use sequence length warmup with 1 epoch at length $2^{14}=16384$, 1 epoch at length $2^{15}=32768$, 1 epoch at length $2^{16}=65536$, and so on up to the maximum sequence length. For example, the model with $2^{20}=1048576$ context undergoes $6$ epochs of sequence length warmup before $4$ more epochs at its maximum sequence length.

The learning rate for all Hyena models is $\mathtt{4e-5}$, while the learning rate for all Mamba models is $\mathtt{1e-4}$. These were found by performing learning rate sweeps for each model among $\{1e-5, 2e-5, 4e-5, 1e-4, 2e-4\}$ for the smaller sequence lengths $(2^{10}, 2^{12}, 2^{14}, 2^{16})$, and these values were consistently found to be the best for each model. An abridged learning rate sweep was done at length $2^{18}$, which agreed with these values, and a single run at length $2^{20}$ was performed (as described above, the computational cost of these experiments is proportional to the sequence length). The learning rate followed a cosine decay schedule with warmup with 5 epochs of linear warmup to the maximum learning rate, and 5 epochs of cosine decay down to $1e-6$. The unusually long learning rate warmup schedule was chosen because the sequence length warmup was also long (e.g. comprising 6 out of 10 epochs for the model with context length $2^{20}$); we did not experiment with this choice.

Results for the Species classification task are in [Table 13](#table-13).

<span id="table-13"></span>

![Great-apes DNA classification table](../../papers/mamba/table-13.png)

**Table 13.** (**Great Apes DNA Classification**.) Accuracy after fine-tuning on sequences of length $2^{10}=1024$ up to $2^{20}=1048576$ using pretrained models of the same context length. Random guessing is 20%.

<span id="section-11-4"></span>

### 11.4 Audio Details

<span id="section-11-4-1"></span>

#### 11.4.1 YouTubeMix Audio Pretraining

**Model.** We use a model with 3 blocks per stage ($3\times5=15$ total Mamba blocks), pooling factor $p=16$, and outer dimension $D=64$, for about 3.5M parameters.

**Dataset.** The data is mu-law encoded at 8 bits, so the model is modeling discrete tokens with a vocab size of $256$.

The dataset consists of clips of up to 1 minute long, or length $960000$, which is subsampled and divided into segments of any desired sequence length. Since the architecture involves two stages of pooling by a factor of $16$, and we want the resulting sequence length to be a a multiple of $8$ for hardware efficiency, the longest possible sequence is $468 \times 2048 = 958464$. The rest of our sequence lengths are defined by successively halving this and rounding up to the nearest multiple of $2048$.

[Table 14](#table-14) lists the specifications used in [Figure 7](#figure-07). Beyond the varying batch sizes, the number of valid segments in the training set varied between different sequence lengths (e.g. the number of training steps per epoch was not constant for different points in the graph), which may have contributed to kinks in the scaling curves.

<span id="table-14"></span>

![YouTubeMix length-scaling setup table](../../papers/mamba/table-14.png)

**Table 14.** YouTubeMix length scaling sequence lengths and batch sizes.

**Training.** Models were trained for $200K$ training steps with a maximum learning rate of $0.002$, $20K$ (10%) warmup steps, and weight decay $0.1$ (similar to our general pretraining recipe across domains).

**Additional Ablations: SSM Parameterizations.** We investigate SSM parameterizations on long-form audio waveform pretraining in the setting of [Figure 7](#figure-07). The setting is modified slightly to use larger models ($8$ layers and $D=64$ for 6M params, the SaShiMi default), shorter sequences ($2^{11}=2048$ to $2^{18}=262144$ instead of $2^{13}$ to $2^{20}$), lower LR ($0.001$ from $0.002$), and shorter training cycles (100K instead of 200K steps).

[Figure 10](#figure-10) shows that the change from S4 $\to$ S6 (i.e. the selection mechanism) is not always beneficial. On long-form audio waveforms, it in fact significantly hampers performance, which may be intuitive from the point of view that audio is uniformly sampled and very smooth, and therefore benefits from continuous linear time-invariant (LTI) methods. After ablating away the selection mechanism, note that the resulting model is the S4 layer inside the Mamba block. To disambiguate, we call this Mamba-S4 as opposed the default Mamba architecture Mamba-S6.

However, on the right side, we keep the outer layers of the U-Net Mamba-S4 and ablate only the inner layers. The performance differences shrink dramatically; this reinforces the hypothesis that layers closer to the *raw* audio signal should be LTI, but once they are "tokenized" and compressed by the outer layers, the inner layers no longer need to be LTI. In this setting however, the real-valued SSM still underperforms the complex-valued one.

<span id="figure-10"></span>

![YouTubeMix audio ablations](../../papers/mamba/figure-10.png)

**Figure 10.** (**Audio Pretraining (YouTubeMix) Ablations**.) As a uniformly-sampled "continuous" signal modality, audio waveforms actually benefit from LTI models which have matching inductive bias. (*Left*) Homogenous models (all blocks have the same parameterization) (*Right*) Only the center U-Net blocks are ablated; the outer blocks are Mamba-S4. Purple line is same as figure on left.

<span id="section-11-4-2"></span>

#### 11.4.2 SC09 Speech Generation

Autoregressive training largely followed the autoregressive language modeling protocol, such as

- Weight decay $0.1$

- Learning rate warmup for 10% of total steps

- AdamW optimizer with $\beta=(0.9, 0.95)$

- Gradient clip value $0.1$

We used a learning rate of $0.002$ and $200000$ training steps at a batch size of $16$.

The large Mamba model in [Table 4](#table-04) has 15 layers per stage with an outer dimension of $D=96$ and pooling factor $4$. We note that this dataset is small (training went through 100 epochs) and for this large model, there was significant overfitting of the BPB or NLL. However, automated metrics of generated samples continually improving throughout training.

The models in the architecture ablations in [Table 5](#table-05) all have 8 layers per stage with an outer dimension of $\mathtt{D}=64$ and pooling factor $4$. The S4+MLP block has roughly $2D^2 + 4D^2$ parameters (expansion factor $2$ in the MLP). The Transformer block has $4D^2 + 2D^2$ parameters (expansion factor $1$ in the MLP). The Mamba block has the usual $\approx 6D^2$ parameters. All models have roughly 6M total parameters.

<span id="section-11-5"></span>

### 11.5 Efficiency Benchmark

**Scan Operation.** We compare the core operation of selective SSMs, which is the parallel scan ([Section 3.3](#section-3-3)), against convolution and attention, measured on an A100 80GB PCIe GPU. Note that these do not include the cost of other operations outside of this core operation, such as computing the convolutional kernel in global-convolution models, or computing the QKV projections in attention.

As a baseline, we implement a standard parallel scan in PyTorch with no kernel fusion. This requires materializing the parameters $\overline{\bm{A}}, \overline{\bm{B}}, \bm{C}$ in HBM.

Our scan implementation fuses the discretization step and the parallel scan, avoiding the cost of materializing all the large parameters in HBM.

For convolution, we use the standard implementation in PyTorch, which separately performs FFTs on the inputs and the filters, multiply them in frequency domain, then performs an inverse FFT to obtain the result. The theoretical complexity is $O(L \log (L))$ for sequence length $L$.

For attention, we compare against the fastest implementation that we are aware of (FlashAttention-2 [Dao24a]), with causal mask. Note that FlashAttention-2 with causal mask is about 1.7$\times$ faster than without causal mask, since approximately only half of the attention entries are computed.

We use batch size of 1 and increase the sequence length from $2^9=512$, $2^{10}\approx 1K$, $2^{11}\approx 2K$, up to $2^{19} \approx 500K$ (some of the baselines run out of memory before reaching 500K). We use a model dimension of $D = 1024$ and state dimension $N = 16$. We measure with BF16 inputs, which is the data type most commonly used for large scale training.

**End-to-end Inference.** We measure the inference throughput of a Mamba 1.4B model and an untrained Mamba 6.9B model, against a standard Transformer (GPT3 architecture) at 1.3B and 6.7B size. We use the standard Transformer implementation in the Huggingface `transformers` library.

We set the prompt length to be 2048 and the generation length to be 128. We vary the batch size from 1, 2, 4, 8, 16, 32, 64, to 128, and measure time time taken to generate 128 tokens. We then calculate the throughput (tokens/s) as $\mathrm{batch\ size} \times 128 / \mathrm{time\ taken}$. We repeat the measurements 3 times and take the average. Measurements are done on an A100 80GB PCIe GPU.

**Memory Benchmark.** The memory usage simply scales proportionally to the size of the activation tensors, as with most deep sequence models. We report measurements of the training memory requirements of 125M models on 1 A100 80GB GPU. Each batch consists of sequences of length 2048. We compare to the most memory-efficient Transformer implementation we are aware of (with kernel fusion from `torch.compile` and with FlashAttention-2). [Table 15](#table-15) shows that Mamba's memory requirement is comparable to a similar-sized Transformer with an extremely optimized implementation, and we expect further improvement in Mamba's memory footprint in the future.

<span id="table-15"></span>

![Memory benchmark table](../../papers/mamba/table-15.png)

**Table 15.** (**Memory benchmark**.) Mamba's memory footprint is comparable to the most optimized Transformer. Results for 125M models.

[+author-order]: The authors are listed alphabetically by first name.
