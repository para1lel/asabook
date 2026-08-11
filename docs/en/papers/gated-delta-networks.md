---
title: 'Gated Delta Networks'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/gated-delta-networks/
pageClass: paper-reading
---

> [Songlin Yang](https://sustcsonglin.github.io/) [+author-note], [Jan Kautz](https://www.jankautz.com/), and [Ali Hatamizadeh](https://ahatamiz.github.io/). First submitted to arXiv on December 9, 2024; current version v3. Published at [ICLR 2025](https://openreview.net/forum?id=r8H7xhYPwz). [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). [Original PDF](/paper/gated-delta-networks.pdf). [DOI](https://doi.org/10.48550/arXiv.2412.06464). [TeX source](https://export.arxiv.org/e-print/2412.06464v3). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.
Code: [https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

## 1 Introduction

The Transformer architecture has significantly advanced the capabilities of Large Language Models (LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism. This mechanism excels in precise sequence modeling and leverages the parallel processing capabilities of modern GPUs during training. However, the self-attention component scales quadratically with sequence length, leading to substantial computational demands that pose challenges for both training and inference.

To mitigate these issues, researchers have explored alternatives such as linear Transformers [ICMLa20], which replace traditional softmax-based attention with kernelized dot-product-based linear attention, substantially reducing memory requirements during inference by reframing as a linear RNN with matrix-valued states. While early versions of linear Transformers underperformed in language modeling tasks compared to standard Transformers, recent enhancements—such as incorporating data-dependent gating mechanisms akin to those in LSTMs, exemplified by models like GLA [PMLRa24] and Mamba2 [Daoa24]—have shown promising improvements. However, challenges persist in managing information over long sequences, particularly for in-context retrieval tasks where traditional Transformers maintain their advantage [Arora23, Arora24, Jelass24, Wen24, Aky24].

This phenomenon is not surprising: linear Transformers can be interpreted as implementing an outer-product-based key-value association memory, reminiscent of tensor product representation [Smolen90]. However, the number of orthogonal key-value pairs they can store is *bounded* by the model’s dimensionality. When the sequence length exceeds this dimension, “memory collisions” become inevitable, hindering exact retrieval [ICMLa21].

Mamba2 addresses this limitation by introducing a simple gated update rule, ${\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}$, which uniformly decays all key-value associations at each time step by a dynamic ratio, $\alpha_{t}\in(0,1)$. However, this approach does not account for the varying importance of different key-value associations, potentially leading to inefficient memory utilization. If the model needs to forget a specific key-value association, all key-value associations are equally forgotten, making the process less targeted and efficient.

In contrast, the linear Transformer with the delta rule [Widrow60], known as DeltaNet [ICMLa21, NeurIP24], selectively updates memory by (softly) replacing an old key-value pair with the incoming one in a sequential manner. This method has demonstrated impressive performance in synthetic benchmarks for in-context retrieval. However, since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased. Consequently, DeltaNet has been found to perform moderately on real-world tasks [NeurIP24], likely due to the absence of a robust memory-clearing mechanism.

Recognizing the complementary advantages of the gated update rule and the delta rule in memory management, we propose the *gated delta rule*, a simple and intuitive mechanism that combines both approaches. This unified rule enables flexible memory control: it can promptly clear memory by setting $\alpha_{t}\rightarrow 0$, while selectively updating specific content without affecting other information by setting $\alpha_{t}\rightarrow 1$ (effectively switching to the pure delta rule).

The remaining challenge lies in implementing the gated delta rule in a hardware-efficient manner. Building upon [NeurIP24]’s efficient algorithm that parallelizes the delta rule computation using the WY representation [Compua85], we carefully extend their approach to incorporate the gating terms. Our extension preserves the benefits of chunkwise parallelism [Huaa22, Suna23, PMLRa24], enabling hardware-efficient training.

Our resulting architecture, Gated DeltaNet, consistently outperforms both Mamba2 and DeltaNet across a comprehensive suite of benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. Building on these results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, further enhancing both training efficiency and model performance.

## 2 Preliminary

### 2.1 Mamba2: Linear Attention with decay

It is known that the linear transformer [ICMLa20] can be formulated as the following linear recurrence when excluding normalization and query/key activations:

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}+{\bm{v}}_t{\bm{k}}_t^\top\in\mathbb{R}^{d_v\times d_k},\qquad\qquad {\bm{o}}_t={\mathbf{S}}_t{\bm{q}}_t\in\mathbb{R}^{d_v}
$$

where $d_k$ and $d_v$ represent the (head) dimensions for query/key and value, respectively. By expanding the recurrence, we can express it in both vector form (left) and matrix form (right) as follows:

$$
{\bm{o}}_t=\sum_{i=1}^t({\bm{v}}_i{\bm{k}}_i^\top){\bm{q}}_t=\sum_{i=1}^t{\bm{v}}_i({\bm{k}}_i^\top{\bm{q}}_t)\in\mathbb{R}^{d_v},\qquad {\mathbf{O}}=({\mathbf{Q}}{\mathbf{K}}^\top\odot{\mathbf{M}}){\mathbf{V}}\in\mathbb{R}^{L\times d_v}
$$

where $L$ is the sequence length, and ${\mathbf{M}}\in\mathbb{R}^{L\times L}$ is the causal mask defined by ${\mathbf{M}}_{ij}=0$ when $i<j$, and $1$ otherwise.

However, this vanilla linear attention underperforms Transformers in language modeling by a large margin. To address this, it is common to add a decay term to forget historical information. Here we take Mamba2 [Daoa24] as an example, which can be represented by the following linear recurrence (up to specific parameterization):

$$
{\mathbf{S}}_t={\color{#000099}\alpha_t}{\mathbf{S}}_{t-1}+{\bm{v}}_t{\bm{k}}_t^\top,\qquad {\bm{o}}_t={\mathbf{S}}_t{\bm{q}}_t
$$

where ${\color{#000099}\alpha_t\in(0,1)}$ is a data-dependent scalar-valued decay term that varies with $t$. Define the cumulative decay product ${\color{#000099}\gamma_j=\prod_{i=1}^j\alpha_i}$, and by expanding the recurrence, we can express the result in both a vector form (left) and a matrix parallel form (right):

$$
{\bm{o}}_t=\sum_{i=1}^t\left({\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{v}}_i{\bm{k}}_i^\top\right){\bm{q}}_t=\sum_{i=1}^t{\bm{v}}_i\left({\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{k}}_i^\top{\bm{q}}_t\right),\qquad {\mathbf{O}}=\left(({\mathbf{Q}}{\mathbf{K}}^\top)\odot{\color{#000099}\Gamma}\right){\mathbf{V}}
$$

Here, ${\color{#000099}\Gamma\in\mathbb{R}^{L\times L}}$ is a decay-aware causal mask where ${\color{#000099}\Gamma_{ij}=\frac{\gamma_i}{\gamma_j}}$ if $i\geq j$ and ${\color{#000099}\Gamma_{ij}=0}$ otherwise. The equivalence between these parallel and recurrent forms is also referred to as the state space duality (SSD) described in [Daoa24]. This recurrence structure appears in several other architectures including Gated RFA [ICLRa21], xLSTM [Beck24], and Gated RetNet [Sunb24]. When $\gamma_t$ is data-independent, the formulation reduces to RetNet [Suna23] and Lightning-Attention [Lightn24]. Furthermore, if $\gamma_t$ is extended to be matrix-valued rather than scalar-valued, efficient training algorithms remain possible when parameterized with an outer-product structure, as demonstrated by [PMLRa24] and used by [PMLRa24, Peng24, Qin24a, Gated24, Systee24, Reprea25, Refini25].

**Chunkwise training.** However, both the recurrent and parallel forms are not ideal for efficient training [Huaa22, PMLRa24], which motivates the use of the chunkwise parallel form [Huaa22, Suna23] for hardware-efficient, linear-time training, as introduced below. To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of size $C$, and computes outputs for each chunk based on the final state of the previous chunk and the query/key/value blocks of the current chunk. Following the notation of [Suna23, PMLRa24, NeurIP24], we take the query block, ${\bm{q}}$, as an example. We denote ${\mathbf{Q}}_{[t]}:={\bm{q}}_{tC+1:(t+1)C+1}$ as the query block for chunk $t$, and ${\bm{q}}_{[t]}^r:={\bm{q}}_{tC+r}$ as the $r$-th query within chunk $t$. The initial state of chunk $t$ is defined as ${\mathbf{S}}_{[t]}:={\mathbf{S}}_{[t]}^0={\mathbf{S}}_{[t-1]}^C$. By partially expanding the recurrence, we have

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}+\sum_{i=1}^r{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad {\bm{o}}_{[t]}^r={\mathbf{S}}_{[t]}^r{\bm{q}}_{[t]}^r={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^r+\sum_{i=1}^r{\bm{v}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^r\right)\in\mathbb{R}^{d_v}
$$

Equivalently, in matrix form:

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}{\mathbf{K}}_{[t]}^\top\in\mathbb{R}^{d_v\times d_k},\qquad {\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
$$

where ${\mathbf{M}}\in\mathbb{R}^{C\times C}$ is the causal mask. The above equations are rich in matrix multiplications (matmuls), allowing for tensor-core-based hardware optimization. This chunkwise algorithm could be easily extended to linear attention with decay:

<span id="equation-01"></span>

$$
{\mathbf{S}}_{[t+1]}={\color{#000099}\overrightarrow{{\mathbf{S}}_{[t]}}}+{\mathbf{V}}_{[t]}^\top{\color{#000099}\overrightarrow{{\mathbf{K}}_{[t]}}}\in\mathbb{R}^{d_v\times d_k},\qquad {\mathbf{O}}_{[t]}={\color{#000099}\overleftarrow{{\mathbf{Q}}_{[t]}}}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\color{#000099}\Gamma_{[t]}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
\tag{1}
$$

where ${\color{#000099}(\Gamma_{[t]})_{ij}=\frac{\gamma_{[t]}^i}{\gamma_{[t]}^j},\ \gamma_{[t]}^j=\prod_{j=tC+1}^{tC+j}\alpha_j}$. [+1] Here we use the left arrow ($\overleftarrow{\cdot}$) or the right arrow ($\overrightarrow{\cdot}$) to denote a variable decaying to the first position and the last position of each chunk, respectively,

<span id="equation-02"></span>

$$
{\color{#000099}\overleftarrow{{\bm{q}}_{[t]}^r}}={\color{#000099}\gamma_{[t]}^r}{\bm{q}}_{[t]}^r\qquad\mathrm{decaying\ each\ vector\ to\ the\ first\ position\ of\ chunk}\ t
$$

$$
{\color{#000099}\overrightarrow{{\bm{k}}_{[t]}^r}}={\color{#000099}\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}}{\bm{k}}_{[t]}^r\qquad\mathrm{decaying\ each\ vector\ to\ the\ last\ position\ of\ chunk}\ t
$$

$$
{\color{#000099}\overrightarrow{{\mathbf{S}}_{[t]}}}={\color{#000099}\gamma_{[t]}^C}{\mathbf{S}}_{[t]}\qquad\mathrm{decaying\ the\ state\ matrix\ over\ the\ entire\ chunk}\ t
\tag{2}
$$

and likewise for other variables (e.g., ${\color{#000099}\overrightarrow{\bm{v}}}$). The SSD decomposition algorithm introduced in Mamba2 is largely equivalent to this chunkwise algorithm. For a more generalized approach, [PMLRa24] proposed an extended chunkwise algorithm for linear attention that incorporates fine-grained decay mechanisms.

### 2.2 Delta Networks: Linear Attention with Delta Rule

The delta update rule [Widrow60, ICMLa21] *dynamically* erases the value (${\bm{v}}_t^{\mathrm{old}}$) associated with the current input key (${\bm{k}}_t$) and writes a new value (${\bm{v}}_t^{\mathrm{new}}$), which is a linear combination of the current input value and the old value based on the "writing strength" $\beta_t\in(0,1)$. [+2]

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}-\underbrace{\left({\mathbf{S}}_{t-1}{\bm{k}}_t\right)}_{{\bm{v}}_t^{\mathrm{old}}}{\bm{k}}_t^\top+\underbrace{\left(\beta_t{\bm{v}}_t+(1-\beta_t){\mathbf{S}}_{t-1}{\bm{k}}_t\right)}_{{\bm{v}}_t^{\mathrm{new}}}{\bm{k}}_t^\top={\mathbf{S}}_{t-1}\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
$$

As shown above, DeltaNet implements a first-order linear recurrence with generalized Householder transition matrices $({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top)$. Despite demonstrating superior associative recall and language modeling performance [ICMLa21], DeltaNet received limited attention due to computational inefficiency until [NeurIP24] introduced a hardware-efficient chunkwise training algorithm, as detailed below.

**Chunkwise parallel form.** By partially expanding the recurrence, we have

<span id="equation-03"></span>

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^r{\mathbf{I}}-\beta_{[t]}^i{\bm{k}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\right)}_{:={\mathbf{P}}_{[t]}^r}+\underbrace{\sum_{i=1}^r\left(\beta_{[t]}^i{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^r\left({\mathbf{I}}-\beta_{[t]}^j{\bm{k}}_{[t]}^j{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{[t]}^r}
\tag{3}
$$

where ${\mathbf{P}}_{[t]}^j$ involves cumulative products of generalized Householder matrices, which could be optimized by the classical WY representation [Compua85]:

<span id="equation-04"></span>

$$
{\mathbf{P}}_{[t]}^r={\mathbf{I}}-\sum_{i=1}^r{\bm{w}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_k\times d_k},\qquad {\bm{w}}_{[t]}^r=\beta_{[t]}^r\left({\bm{k}}_{[t]}^r-\sum_{i=1}^{r-1}{\bm{w}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_k}
\tag{4}
$$

Likewise, ${\mathbf{H}}_{[t]}^r$ could be represented as:

<span id="equation-05"></span>

$$
{\mathbf{H}}_{[t]}^r=\sum_{i=1}^r{\bm{u}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad {\bm{u}}_{[t]}^r=\beta_{[t]}^r\left({\bm{v}}_{[t]}^r-\sum_{i=1}^{r-1}{\bm{u}}_{[t]}^i\left({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_v}
\tag{5}
$$

and in matrix form: ${\mathbf{P}}_{[t]}={\mathbf{I}}-{\mathbf{W}}_{[t]}^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_k\times d_k}$, ${\mathbf{H}}_{[t]}={\mathbf{U}}_{[t]}^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_v\times d_k}$. By using the UT transform [Joffra06], we can further write ${\mathbf{W}}$ and ${\mathbf{U}}$ in matrix form:

<span id="equation-06"></span>

$$
{\mathbf{T}}_{[t]}=\left[{\mathbf{I}}+\mathrm{strictLower}\left(\mathrm{diag}(\beta_{[t]}){\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^\top\right)\right]^{-1}\mathrm{diag}(\beta_{[t]})\in\mathbb{R}^{C\times C}
\tag{6}
$$

<span id="equation-07"></span>

$$
{\mathbf{W}}_{[t]}={\mathbf{T}}_{[t]}{\mathbf{K}}_{[t]}\in\mathbb{R}^{C\times d_k},\qquad {\mathbf{U}}_{[t]}={\mathbf{T}}_{[t]}{\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
\tag{7}
$$

Substituting these back into [Eq. 3](#equation-03) yields a hardware-efficient chunkwise algorithm for DeltaNet that leverages matmuls, enabling tensor core based GPU optimization:

<span id="equation-08"></span>

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}{\mathbf{P}}_{[t]}+{\mathbf{H}}_{[t]}={\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^\top\right)^\top{\mathbf{K}}_{[t]}\in\mathbb{R}^{d_v\times d_k}
\tag{8}
$$

<span id="equation-09"></span>

$$
{\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right)\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^\top\right)\in\mathbb{R}^{C\times d_v}
\tag{9}
$$

## 3 Gated Delta Networks

### 3.1 Formulation: Gated Delta Rule

The proposed gated delta rule is simple yet effective:

<span id="equation-10"></span>

$$
{\mathbf{S}}_t={\mathbf{S}}_{t-1}\left({\color{#000099}\alpha_t}\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
\tag{10}
$$

where the data-dependent gating term ${\color{#000099}\alpha_t}\in(0,1)$ controls state decay. This formulation unifies the advantages of both gating mechanisms and the delta rule: the gating term enables adaptive memory management, while the delta update structure facilitates effective key-value association learning.

We present a formal analysis of the gated delta rule through the lens of the online learning framework introduced by [Liua24]. In this framework, recurrent state updates emerge as *closed-form* solutions to an online learning problem, as shown in [Table 1](#table-01). Recent linear RNN architectures typically incorporate a regularization term in their online learning objective to prevent state divergence from previous values, thereby enabling memory retention. However, this retention mechanism becomes problematic when the state becomes saturated with information. In such cases, each state would encode a superposition of multiple information pieces, making precise retrieval challenging. To address this limitation, Mamba2 and Gated DeltaNet introduce an adaptive scaling factor $\alpha_t$ that relaxes the regularization term, allowing controlled deviations between ${\mathbf{S}}_t$ and ${\mathbf{S}}_{t-1}$. This modification enables dynamic memory management through selective forgetting, which could be useful in filtering out irrelevant information (see [§ 3.2](#32-case-study-single-needle-in-a-haystack-s-niah)).

On the other hand, Linear Attention (LA) and Mamba2 use a simple negative inner-product loss $-\langle{\mathbf{S}}_t{\bm{k}}_t,{\bm{v}}_t\rangle$, while Longhorn [Liua24] uses a more expressive online regression objective $\|{\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t\|^2$ for better modeling of key-value associations. The resulting Longhorn's update rule closely resembles the delta update rule, [+3] suggesting the superiority of the (gated) delta rule over Mamba2 in in-context associative recall.

From the perspective of fast weight programming [ICMLc22] and test-time training [Suna24] and regression [Testti25], the hidden state ${\mathbf{S}}$ can be interpreted as a (fast) weight matrix, with the delta rule optimizing the online regression objective $\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\|{\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t\|^2$ via *test-time* stochastic gradient descent (SGD):

$$
{\mathbf{S}}_{t+1}={\mathbf{S}}_t-\beta_t\nabla\mathcal{L}({\mathbf{S}}_t)={\mathbf{S}}_t-\beta_t({\mathbf{S}}_t{\bm{k}}_t-{\bm{v}}_t){\bm{k}}_t^\top={\mathbf{S}}_t\left({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top\right)+\beta_t{\bm{v}}_t{\bm{k}}_t^\top
$$

where $\beta_t$ represents the (adaptive) learning rate. From this perspective, the gated delta rule can be viewed as incorporating an adaptive weight decay term $\alpha_t$ into the SGD update, a technique widely used in deep learning [Hertz91, Andriu23]. Concurrently, Titans [Learni24] demonstrated the effectiveness of incorporating weight decay mechanisms in RNN test-time SGD updates.

<span id="table-01"></span>

![Original paper Table 1](../../papers/gated-delta-networks/table-01.png)

**Table 1.** Comparison of different linear RNN models and their corresponding online learning objectives using the framework from [Liua24]. For convenience, we simplify Longhorn's vector-valued ${\bm{\beta}}$ to scalar $\beta$.

<span id="table-02"></span>

![Original paper Table 2](../../papers/gated-delta-networks/table-02.png)

**Table 2.** Zero-shot performance comparison on S-NIAH benchmark suite for 1.3B models (see [§ 4](#4-experiments) for setups).

### 3.2 Case study: Single Needle in a Haystack (S-NIAH)

To better understand the complementary strength between the delta rule and the gated rule, we present a case study on the Single Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER [Hsieh24], where a key-value pair acts as a needle in the haystack (context) and the model must recall the value when given the key. [Table 2](#table-02) presents the results and we draw three main observations:

**Decay hurts memory retention.** In the simplest S-NIAH-1 setting with repeated synthetic context, models memorize minimal information, testing long-term retention. DeltaNet achieves near-perfect performance across all sequence lengths. Mamba2 degrades significantly beyond 2K sequences since it decays historical information too quickly, while Gated DeltaNet's degradation is less severe thanks to the use of delta rule.

**Gating facilitates filtering.** In S-NIAH-2/3 with real-world-essay context, models store all potentially relevant information, testing efficient memory management. With fixed state size, lack of clearance causes memory collision—information becomes superimposed and indistinguishable. DeltaNet's performance drops significantly at longer sequences due to poor memory clearance. Mamba2 and Gated DeltaNet maintain better performance through gating mechanisms that filter irrelevant information.

**Delta rule helps memorization.** In S-NIAH-3, values change from numbers to UUIDs, testing complex pattern memorization. Mamba2's performance drops quickly, while Gated DeltaNet performs better, verifying that the delta rule indeed has better memorization ability.

### 3.3 Algorithm: Hardware-efficient Chunkwise training

In this subsection, we derive a hardware-efficient chunkwise algorithm for training Gated DeltaNet. By partially expanding the recurrence in [Eq. 10](#equation-10), we have

$$
{\mathbf{S}}_{[t]}^r={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^r{\color{#000099}\alpha_{[t]}^i}\left({\mathbf{I}}-\beta_{[t]}^i{\bm{k}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\right)\right)}_{:={\mathbf{F}}_{[t]}^r}+\underbrace{\sum_{i=1}^r\left(\beta_{[t]}^i{\bm{v}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^r{\color{#000099}\alpha_{[t]}^j}\left({\mathbf{I}}-\beta_{[t]}^j{\bm{k}}_{[t]}^j{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{G}}_{[t]}^r}
$$

It is easy to see that ${\mathbf{F}}_{[t]}^r={\color{#000099}\gamma_{[t]}^r}{\mathbf{P}}_{[t]}^r={\color{#000099}\overleftarrow{{\mathbf{P}}_{[t]}^r}}$. As for ${\mathbf{G}}_{[t]}^r$, we adapt [Eq. 5](#equation-05) as follows,

$$
{\mathbf{G}}_{[t]}^r=\sum_{i=1}^r{\color{#000099}\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}}\widetilde{\bm{u}}_{[t]}^i{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_v\times d_k},\qquad \widetilde{\bm{u}}_{[t]}^r=\beta_{[t]}^r\left({\bm{v}}_{[t]}^r-\sum_{i=1}^{r-1}\widetilde{\bm{u}}_{[t]}^i\left({\color{#000099}\frac{\gamma_{[t]}^r}{\gamma_{[t]}^i}}{\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^r\right)\right)\in\mathbb{R}^{d_v}
$$

(see [§ A](#appendix-a-extended-wy-representation-for-gated-delta-rule) for a proof). By UT transform, we have the matrix form:

$$
\widetilde{\mathbf{U}}_{[t]}=\left[{\mathbf{I}}+\mathrm{strictLower}\left(\mathrm{diag}(\beta_{[t]})\left({\color{#000099}\Gamma_{[t]}}\odot{\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^\top\right)\right)\right]^{-1}\mathrm{diag}(\beta_{[t]}){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_v}
$$

Similar to how Mamba2 extends linear attention ([Eq. 1](#equation-01)), we can adapt DeltaNet's chunkwise algorithm ([Eq. 8](#equation-08)-[9](#equation-09)) for Gated DeltaNet to enable hardware-efficient training as follows:

$$
{\mathbf{S}}_{[t+1]}={\color{#000099}\overrightarrow{{\mathbf{S}}_{[t]}}}+\left(\widetilde{\mathbf{U}}_{[t]}-{\color{#000099}\overleftarrow{{\mathbf{W}}_{[t]}}}{\mathbf{S}}_{[t]}^\top\right)^\top{\color{#000099}\overrightarrow{{\mathbf{K}}_{[t]}}}\in\mathbb{R}^{d_v\times d_k}
$$

$$
{\mathbf{O}}_{[t]}={\color{#000099}\overleftarrow{{\mathbf{Q}}_{[t]}}}{\mathbf{S}}_{[t]}^\top+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^\top\odot{\mathbf{M}}\right)\left(\widetilde{\mathbf{U}}_{[t]}-{\color{#000099}\overleftarrow{{\mathbf{W}}_{[t]}}}{\mathbf{S}}_{[t]}^\top\right)\in\mathbb{R}^{C\times d_v}
$$

where ${\color{#000099}\overleftarrow{{\bm{q}}_{[t]}^r}=\gamma_{[t]}^r}{\bm{q}}_{[t]}^r$, ${\color{#000099}\overleftarrow{{\bm{w}}_{[t]}^r}=\gamma_{[t]}^r}{\bm{w}}_{[t]}^r$, ${\color{#000099}\overrightarrow{{\bm{k}}_{[t]}^r}=\frac{\gamma_{[t]}^C}{\gamma_{[t]}^r}}{\bm{k}}_{[t]}^r$, and ${\color{#000099}\overrightarrow{{\mathbf{S}}_{[t]}}=\gamma_{[t]}^C}{\mathbf{S}}_{[t]}$ like we defined in [Eq. 2](#equation-02).

### 3.4 Gated Delta Networks and Hybrid Models

**Token mixer block.** The basic Gated DeltaNet follows Llama's macro architecture, stacking token mixer layers with SwiGLU MLP layers, but replaces self-attention with gated delta rule token mixing. [Fig. 1](#figure-01) (right) shows its block design. For the gated delta rule ([Eq. 10](#equation-10)), queries, keys and values $\{{\bm{q}},{\bm{k}},{\bm{v}}\}$ are generated through linear projection, short convolution and SiLU, with L2 normalization applied to ${\bm{q}},{\bm{k}}$ for training stability. $\alpha,\beta$ use linear projection only. [+4] Following [Suna23], the output is processed through normalization and gating before applying output projection.

<span id="figure-01"></span>

![Figure 1](../../papers/gated-delta-networks/figure-01.png)

**Figure 1.** Visualization of the (hybrid) architecture and block design of Gated DeltaNet models. Gated DeltaNet-H1 and H2 use Gated DeltaNet + SWA and Mamba2 + Gated DeltaNet + SWA patterns, respectively. In the block design, query/key paths consist of linear proj., shortconv., SiLU and L2 norm; value path includes linear proj., shortconv. and SiLU; alpha/beta use linear proj.; and output gate applies linear proj. with SiLU.

**Hybrid models.** Linear transformers have limitations in modeling local shifts and comparisons, and their fixed state size makes it hard for retrieval tasks [Arora24]. Following recent hybrid architectures like Griffin [De24] and Samba [Ren24], we combine linear recurrent layers with sliding window attention (SWA), resulting in GatedDeltaNet-H1. We also stack Mamba2, GatedDeltaNet and SWA, resulting in GatedDeltaNet-H2.

## 4 Experiments

**Setup.** Our experiments encompass a comprehensive comparison of recent state-of-the-art architectures, including pure Transformer models, RNN-based approaches, and hybrid architectures. We evaluate against the following baselines: RetNet [Suna23], HGRN2 [Qin24a], Mamba [Daoc23], Mamba2 [Daob24], Samba [Ren24], and DeltaNet [NeurIP24]. For fair comparison, all models are trained under identical conditions with 1.3B parameters on 100B tokens sampled from the FineWeb-Edu dataset [Penedo24]. We use the AdamW optimizer with a peak learning rate of 4e-4, weight decay of 0.1, and gradient clipping of 1.0. The learning rate follows a cosine annealing schedule with a 1B token warm-up period and batch size of 0.5M tokens. All models employ the Llama2 tokenizer with a vocabulary size of 32,000. For sequence modeling, we set the training length to 4K tokens, with Samba and our hybrid models using a sliding window size of 2K. See [§ B.1](#b1-evaluation) for evaluation settings and [§ B.2](#b2-ablation-study) for ablation studies.

<span id="table-03"></span>

![Original paper Table 3](../../papers/gated-delta-networks/table-03.png)

**Table 3.** Performance comparison on language modeling and zero-shot common-sense reasoning.

**Common-sense reasoning.** In [Table 3](#table-03), we present the language modeling perplexity and **zero-shot** accuracy on commonsense reasoning benchmarks for models with 400M and 1.3B parameters. Gated DeltaNet consistently outperforms other linear models, including RetNet, HGRN2, Mamba, Mamba2, and DeltaNet, at both scales. As expected, the hybrid variant further enhances performance.

<span id="table-04"></span>

![Original paper Table 4](../../papers/gated-delta-networks/table-04.png)

**Table 4.** Accuracy on recall-world retrieval tasks with input truncated to 2K tokens. SQD: SQUADE. TQA: Trivial QA.

**In-context retrieval on real-world data.** [Table 4](#table-04) presents results on real-world recall-intensive tasks used by [Aroraa24]. As expected, linear recurrent models show a significant performance gap compared to Transformers, while hybrid models combining linear recurrence and attention outperform pure attention models in retrieval tasks.

For pure recurrent models, despite DeltaNet's superior performance on synthetic in-context retrieval tasks [NeurIP24], its real-world retrieval performance lags behind Mamba2, consistent with our observations in S-NIAH-2 and S-NIAH-3 ([Table 2](#table-02)). Gated DeltaNet outperforms both DeltaNet and Mamba2 thanks to its gated delta rule, though the improvement margin is smaller than in [Table 2](#table-02). We attribute this reduced performance gap to instruction-unaligned small language models being prone to repetition errors, which are the primary source of errors in these tasks (cf. [Aroraa24], Appendix E). Since this issue is largely independent of the update rule choice, the performance differences between models are less pronounced compared to [Table 2](#table-02).

<span id="figure-02"></span>

![Figure 2](../../papers/gated-delta-networks/figure-02.png)

**Figure 2.** Length extrapolation on six long benchmarks.

**Length extrapolation on long sequences.** As shown in [Fig. 2](#figure-02), we evaluate the models' capacity to extrapolate to sequences of up to 20K tokens across six long-context benchmarks. Gated DeltaNet achieves the lowest overall perplexity across tasks among RNN models. While we observe mixed results in length extrapolation, Gated DeltaNet exhibits relatively more robust performance, suggesting better memory management. The hybrid models further improve upon this by leveraging attention for local context modeling, which reduces the memory management burden on their recurrent components. Future work will explore these models' capabilities on even longer sequences.

**Long context understanding.** As demonstrated in [Table 5](#table-05), we evaluated the models' performance on LongBench [Bai23]. In recurrent models, Gated DeltaNet shows consistent advantages, especially in single-doc QA, few-shot in-context learning, and Code tasks, demonstrating its superior capabilities in retrieval, in-context learning, and state tracking, respectively.

<span id="table-05"></span>

![Original paper Table 5](../../papers/gated-delta-networks/table-05.png)

**Table 5.** Accuracy on 14 tasks from LongBench [Bai23]: Narrative QA, QasperQA, MultiField QA, HotpotQA, 2WikiMulti QA, Musique, GovReport, QMSum, MultiNews, TRec, Trivia QA, SamSum, LCC, and RepoBench-P by order.

<span id="figure-03"></span>

![Figure 3](../../papers/gated-delta-networks/figure-03.png)

**Figure 3.** Training throughput comparison of 1.3B models on a single H100 GPU.

**Throughput Comparison.** The training throughput comparison across different models is presented in [Fig. 3](#figure-03). As our analysis shows, the proposed gated delta rule introduces only marginal overhead compared to the original delta rule, with Gated DeltaNet achieving essentially the same throughput as DeltaNet. Both are slightly slower than Mamba2 (2-3K tokens/sec) due to their more expressive transition matrices.

The Transformer++ achieves the best performance in the 2K context window domain, thanks to the highly optimized Flash-Attention-2 kernel [Daob23]. Consequently, hybrid approaches combining 2K window-size SWA attention with other token mixers demonstrate higher throughput than standalone mixers: Samba outperforms Mamba, while Gated DeltaNet-H1 and -H2 outperform Gated DeltaNet. Notably, Gated DeltaNet-H1 maintains compelling training throughput across all sequence lengths, even on short sequences.

## 5 Related Work

**Gated linear RNN.** Large linear recurrent language models have attracted significant attention due to their training and inference efficiency. The field of linear RNNs has rapidly evolved from using data-independent decay mechanisms, as exemplified by models like S4 [ICLRb22], S5 [ICLRb23], LRU [ICMLa23], RWKV4/5 [EMNLP23], and RetNet [Suna23], to incorporating data-dependent decay mechanisms in more recent architectures such as HGRN1/2 [Qin24a, Qina23], Mamba1/2 [Daoc23, Daoa24], RWKV6 [Peng24], GSA [Gated24]. This transition stems from the proven advantages of gating/forgetting mechanisms (termed selective mechanisms in Mamba)—a classical concept originating in the gated RNN literature [Gersa00] whose significance has been consistently reaffirmed [Greff15, Lasenb18, Qin24a, Qina23, Daoc23].

Modern forget gates differ from traditional designs like those in LSTM by removing the dependency on the previous hidden state, relying solely on input data. This modification enables efficient parallelism across sequence lengths [Mar18, Qina23]. The absence of a forget gate has been a notable limitation in DeltaNet, and our gated extension addresses this gap in a natural, effective, and hardware-efficient way. We also note a recent concurrent work RWKV-7 [+5] using a similar idea, but with a more relaxable formalism using diagonal-plus-low-rank transitions: ${\mathbf{S}}_t={\mathbf{S}}_{t-1}(\mathrm{diag}({\mathbf{d}}_t)-{\mathbf{a}}_t{\mathbf{b}}_t^\top)+{\bm{v}}_t{\bm{k}}_t^\top$ where ${\mathbf{d}}_t,{\mathbf{a}}_t,{\mathbf{b}}_t\in\mathbb{R}^{d_k}$. The chunkwise algorithm could be similarly adapted to this case, as implemented in Flash Linear Attention [Yan24]. [+6]

**Delta rule.** The delta learning rule demonstrates superior memory capacity compared to Hebbian learning [Gardne88, Kak89], an advantage DeltaNet leverages while linear transformers rely on Hebbian-like rules. This memory capacity advantage is evident in synthetic in-context learning tasks and extends to language modeling [Irie21, NeurIP24], reinforcement learning [ICMLd22], and image generation [ICLRa23]. [NeurIP24] parallelized delta rule computation and demonstrated how DeltaNet's data-dependent identity-plus-low-rank structure $({\mathbf{I}}-\beta_t{\bm{k}}_t{\bm{k}}_t^\top)$ offers greater flexibility than Mamba2's data-dependent diagonal matrices $(\alpha_t{\mathbf{I}})$. This structural advantage could enable complex reasoning, including regular language recognition [Bethar24, Grazzi24] and state-tracking beyond $\mathrm{TC}^0$ complexity [Merril24]—crucial for coding and reasoning applications.

Despite these significant advantages, the delta rule faces theoretical limitations [Bali23] and shows only moderate performance on real-world datasets [NeurIP24], suggesting room for improvement. Previous attempts to enhance expressiveness through nonlinear recurrence [Irie21, ICMLd22] addressed some limitations but sacrificed training parallelism, creating a performance-efficiency tradeoff. Recent work proposes some enhancements without compromising parallelism for better state tracking performance, including using negative eigenvalues [Grazzi24] and multiple products of householder transition matrices [Increa25] which enable high-rank transformations. These methods could be applied to Gated DeltaNet seamlessly.

From a (online) learning objective perspective, alternative formulations could further extend expressiveness: nonlinear regression ($\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\|f_{{\mathbf{S}}_t}({\bm{k}}_t)-{\bm{v}}_t\|^2$) as in TTT [Suna24] and Titans [Learni24], where $f_{\mathbf{S}}$ is a nonlinear function parameterized by ${\mathbf{S}}$; or regression considering the entire history ($\mathcal{L}({\mathbf{S}}_t)=\frac{1}{2}\sum_{i=1}^t\|{\mathbf{S}}_t{\bm{k}}_i-{\bm{v}}_i\|^2$) as in Mesa layer [Uncove24]—analogous to the difference between Least Mean Square and Recursive Least Square algorithms. However, these more expressive variants introduce nonlinear recurrence and require workarounds, such as performing nonlinear updates only after processing entire chunks (as in TTT and Titans); or approximating nonlinear recurrence methods like [Parall24, Systef24, Balanc25].

**Hybrid models.** In this work, we explore interleaving hybrid attention layers across layers, which is commonly used such as in MiniMax-01 [Scalin25] and Hybrid Mamba2-Attention [Catanz24]. It is also interesting to investigate hybrid linear/softmax attention within a single layer [Huaa22, Systeg24, ArXiv24, Combin24, Don25, Repreb25].

## 6 Conclusion

In this work, we introduced Gated DeltaNet, which enables better key-value association learning compared to Mamba2 and more adaptive memory clearance than DeltaNet, leading to consistently better empirical results across various tasks. We extended the parallel algorithm from [NeurIP24] to enable hardware-efficient training of Gated DeltaNet. Our hybrid Gated DeltaNet model achieves even higher training throughput and overall performance, making it well-suited for practical deployment.

## Acknowledgment

We thank Yu Zhang for assistance with figure creation and model evaluation; Kazuki Irie for providing valuable feedback on the draft; Simeng Sun and Zhixuan Lin for insightful discussions on long-sequence task evaluation settings; and Eric Alcaide and Volodymyr Kyrylov for their helpful discussions on the online learning perspective of DeltaNet.

## Appendix A Extended WY representation for gated delta rule

To reduce notation clutter, we only consider the first chunk here.

For ${\mathbf{S}}_t$, the extended WY representation is

$$
{\mathbf{S}}_t=\sum_{i=1}^t{\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top,\qquad {\bm{u}}_t=\beta_t\left({\bm{v}}_t-\sum_{i=1}^{t-1}{\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_t\right)
$$

We proof this by mathmetical induction.

**Proof.**

$$
{\mathbf{S}}_{t+1}={\mathbf{S}}_t\left({\color{#000099}\alpha_{t+1}}\left({\mathbf{I}}-\beta_{t+1}{\bm{k}}_{t+1}{\bm{k}}_{t+1}^\top\right)\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
={\color{#000099}\alpha_{t+1}}\left(\sum_{i=1}^t{\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top\right)-{\color{#000099}\alpha_{t+1}}\beta_{t+1}\left(\sum_{i=1}^t{\color{#000099}\frac{\gamma_t}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_i{\bm{k}}_{t+1}^\top\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^t{\color{#000099}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top+\underbrace{\beta_{t+1}\left({\bm{v}}_{t+1}-\sum_{i=1}^t{\color{#000099}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top{\bm{k}}_{t+1}\right)}_{{\bm{u}}_{t+1}}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^t{\color{#000099}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top+\underbrace{{\color{#000099}\frac{\gamma_{t+1}}{\gamma_{t+1}}}}_{1}{\bm{u}}_{t+1}{\bm{k}}_{t+1}^\top
$$

$$
=\sum_{i=1}^{t+1}{\color{#000099}\frac{\gamma_{t+1}}{\gamma_i}}{\bm{u}}_i{\bm{k}}_i^\top
$$

∎

## Appendix B Experiment Contunued

### B.1 Evaluation

**Commonsense reasoning.** Following [Daoc23], we evaluate our model on multiple commonsense reasoning benchmarks: PIQA [AAAIc20], HellaSwag [Italyb19], WinoGrande [AAAId20], ARC-easy (ARC-e) and ARC-challenge (ARC-c) [Clark18], SIQA [Sap19], BoolQ [Clark19], Wikitext [ICLR17], and LAMBADA [August16].

**In-context retrieval.** Our evaluation comprises both synthetic and real-world tasks. For synthetic tasks, we utilize the Needle-In-A-Haystack Single (NIAH-S) benchmark suite from RULER [Hsieh24], which includes three increasingly complex tasks: S-NIAH-1 (passkey retrieval), S-NIAH-2 (numerical needle in haystack), and S-NIAH-3 (word-based needle in haystack).

For real-world tasks, following [Aroraa24], we evaluate on diverse datasets: SWDE [Lockar19] for structured HTML relation extraction, FDA [Aroraa23] for PDF key-value retrieval, and several question-answering datasets including SQuAD [Austra18], TriviaQA [Canada17], Drop [Duaa19], and NQ [Kwiatk19]. Since our pretrained models lack instruction tuning, we employ the Cloze Completion Formatting prompts provided by [Aroraa24], which better align with our models' next-word-prediction training objective.

**Long context understanding.** We evaluate on 14 tasks from Longbench [Bai23], encompassing: narrative comprehension (Narrative QA [Lingui18]), scientific understanding (QasperQA [Dasigi21]), multi-hop reasoning (MultiField QA, HotpotQA [Tsujib18], 2WikiMulti QA [Online20], Musique [Lingui22]), document summarization (GovReport [Huang21], QMSum [Zhong21], MultiNews [Fabbri19]), and various specialized tasks (TRec [COLING02], Trivia QA [Canada17], SamSum [China19], LCC [Guoa23], and RepoBench-P [Liub23]).

### B.2 Ablation Study

<span id="table-06"></span>

![Original paper Table S.1](../../papers/gated-delta-networks/table-06.png)

**Table S.1.** Ablation study on the Gated DeltaNet block. Avg-PPL and Avg-Acc denote average perplexity and zero-shot commonsense reasoning accuracy (as in [Table 3](#table-03)), respectively. All models have 400M parameters and are trained for 15B tokens on the same subset of FineWeb-Edu dataset [Penedo24].

<span id="table-07"></span>

![Original paper Table S.2](../../papers/gated-delta-networks/table-07.png)

**Table S.2.** Ablation studies of Gated DeltaNet models. All evaluations are performed by using `lm-evaluation-harness` [Gaob21]. All models use the Llama tokenizer and are trained on the same subset of the FineWeb-Edu dataset [Penedo24].

[Table S.1](#table-06) presents ablation studies on the Gated DeltaNet block's components. Our experiments demonstrate that both the short convolution and output gate are crucial for model performance, while output normalization yields marginal improvements. Consistent with [NeurIP24], we found L2 normalization to be essential for optimal performance, though the choice of feature map was less influential. Nevertheless, SiLU consistently outperformed other activation functions, aligning with observations from [Qin23]. Through empirical analysis, we determined that a head dimension of 128 provides an optimal trade-off between performance and computational efficiency. Additionally, [Table S.2](#table-07) demonstrates that among various hybrid architectures, the combination of Mamba2, Gated DeltaNet, and SWA in this specific order produces superior results.

[+author-note]: Equation contribution. Work done during SY's internship at NVIDIA.

[+1]: Here we slightly abuse the notation of $\gamma$ to denote the cumulative product for each chunk (starting with the first position of each chunk separately) instead of the entire sequence.

[+2]: It is possible to set $\beta_t\in(0,2)$ to allow negative eigenvalue to unlock the state tracking abilities of DeltaNet [Grazzi24, Increa25].

[+3]: The theoretical distinction lies in the optimization approach: Longhorn uses implicit online learning [Bartle10] to derive closed-form globally optimal updates, while DeltaNet optimizes the same objective through one-step explicit gradient descent, as noted by [Liua24].

[+4]: We use Mamba2's parameterization for $\alpha$ but omit it for brevity.

[+5]: [https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7](https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v7)

[+6]: [https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/generalized_delta_rule)
