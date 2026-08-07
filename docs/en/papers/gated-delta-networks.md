---
title: 'Gated Delta Networks'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/gated-delta-networks/
---

> [Songlin Yang](https://sustcsonglin.github.io/), [Jan Kautz](https://www.jankautz.com/), and [Ali Hatamizadeh](https://ahatamiz.github.io/). First submitted to arXiv on December 9, 2024; current version v3. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464). [Original PDF](/paper/gated-delta-networks.pdf). [TeX source](https://export.arxiv.org/e-print/2412.06464). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Linear Transformers have gained attention as efficient alternatives to standard Transformers, but their performance in retrieval and long-context tasks has been limited. To address these limitations, recent work has explored two distinct mechanisms: gating for adaptive memory control and the delta update rule for precise memory modifications. We observe that these mechanisms are complementary—gating enables rapid memory erasure while the delta rule facilitates targeted updates. Building on this insight, we introduce the gated delta rule and develop a parallel training algorithm optimized for modern hardware. Our proposed architecture, Gated DeltaNet, consistently surpasses existing models like Mamba2 and DeltaNet across multiple benchmarks, including language modeling, common-sense reasoning, in-context retrieval, length extrapolation, and long-context understanding. We further enhance performance by developing hybrid architectures that combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, achieving both improved training efficiency and superior task performance.
Code: [https://github.com/NVlabs/GatedDeltaNet](https://github.com/NVlabs/GatedDeltaNet)

## 1 Introduction

The Transformer architecture has significantly advanced the capabilities of Large Language Models (LLMs), showcasing exceptional performance across a wide range of tasks due to its effective attention mechanism. This mechanism excels in precise sequence modeling and leverages the parallel processing capabilities of modern GPUs during training. However, the self-attention component scales quadratically with sequence length, leading to substantial computational demands that pose challenges for both training and inference.

To mitigate these issues, researchers have explored alternatives such as linear Transformers [ICMLa20], which replace traditional softmax-based attention with kernelized dot-product-based linear attention, substantially reducing memory requirements during inference by reframing as a linear RNN with matrix-valued states. While early versions of linear Transformers underperformed in language modeling tasks compared to standard Transformers, recent enhancements—such as incorporating data-dependent gating mechanisms akin to those in LSTMs, exemplified by models like GLA [PMLRa24] and Mamba2 [Daoa24]—have shown promising improvements. However, challenges persist in managing information over long sequences, particularly for in-context retrieval tasks where traditional Transformers maintain their advantage [Arora23, Arora24, Jelass24, Wen24, Aky24].

This phenomenon is not surprising: linear Transformers can be interpreted as implementing an outer-product-based key-value association memory, reminiscent of tensor product representation [Smolen90]. However, the number of orthogonal key-value pairs they can store is *bounded* by the model’s dimensionality. When the sequence length exceeds this dimension, “memory collisions” become inevitable, hindering exact retrieval [ICMLa21].

Mamba2 addresses this limitation by introducing a simple gated update rule, ${\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}$, which uniformly decays all key-value associations at each time step by a dynamic ratio, $\alpha_{t}$. However, this approach does not account for the varying importance of different key-value associations, potentially leading to inefficient memory utilization. If the model needs to forget a specific key-value association, all key-value associations are equally forgotten, making the process less targeted and efficient.

In contrast, the linear Transformer with the delta rule [Widrow60], known as DeltaNet [ICMLa21, NeurIP24], selectively updates memory by (softly) replacing an old key-value pair with the incoming one in a sequential manner. This method has demonstrated impressive performance in synthetic benchmarks for in-context retrieval. However, since this process only modifies a single key-value pair at a time, the model lacks the ability to rapidly clear outdated or irrelevant information, especially during context switches where previous data needs to be erased. Consequently, DeltaNet has been found to perform moderately on real-world tasks [NeurIP24], likely due to the absence of a robust memory-clearing mechanism.

Recognizing the complementary advantages of the gated update rule and the delta rule in memory management, we propose the *gated delta rule*, a simple and intuitive mechanism that combines both approaches. This unified rule enables flexible memory control: it can promptly clear memory by setting $\alpha_{t}\rightarrow 0$, while selectively updating specific content without affecting other information by setting $\alpha_{t}\rightarrow 1$ (effectively switching to the pure delta rule).

The remaining challenge lies in implementing the gated delta rule in a hardware-efficient manner. Building upon [NeurIP24]’s efficient algorithm that parallelizes the delta rule computation using the WY representation [Compua85], we carefully extend their approach to incorporate the gating terms. Our extension preserves the benefits of chunkwise parallelism [Huaa22, Suna23, PMLRa24], enabling hardware-efficient training.

Our resulting architecture, Gated DeltaNet, consistently outperforms both Mamba2 and DeltaNet across a comprehensive suite of benchmarks, including language modeling, commonsense reasoning, in-context retrieval, length extrapolation, and long-context understanding. Building on these results, we also develop hybrid architectures that strategically combine Gated DeltaNet layers with sliding window attention or Mamba2 layers, further enhancing both training efficiency and model performance.

## 2 Preliminary

### 2.1 Linear Attention with Chunkwise Parallel Form

It is known that the linear transformer [ICMLa20] can be formulated as the following linear recurrence when excluding normalization and query/key activations:

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad\qquad{\bm{o}}_{t}={\mathbf{S}}_{t}{\bm{q}}_{t}\in\mathbb{R}^{d_{v}}
$$

where $d_{k}$ and $d_{v}$ represent the (head) dimensions for query/key and value, respectively. By expanding the recurrence, we can express it in both vector form (left) and matrix form (right) as follows:

$$
{\bm{o}}_{t}=\sum_{i=1}^{t}({\bm{v}}_{i}{\bm{k}}_{i}^{\top}){\bm{q}}_{t}=\sum_{i=1}^{t}{\bm{v}}_{i}({\bm{k}}_{i}^{\top}{\bm{q}}_{t})\in\mathbb{R}^{d_{v}},\qquad{\mathbf{O}}=({\mathbf{Q}}{\mathbf{K}}^{\top}\odot{\mathbf{M}}){\mathbf{V}}\in\mathbb{R}^{L\times d_{v}}
$$

where $L$ is the sequence length, and ${\mathbf{M}}\in\mathbb{R}^{L\times L}$ is the causal mask defined by ${\mathbf{M}}_{\mathrm{ij}}=0$ when $i<j$, and $1$ otherwise.

This formulation makes it clear that linear attention eliminates the softmax operation used in traditional attention mechanisms and instead leverages the linearity and associativity of matrix multiplications, leading to linear complexity. However, both the recurrent and parallel forms are not ideal for efficient training [PMLRa24], which motivates the use of the chunkwise parallel form [Huaa22, Suna23, PMLRa24] for hardware-efficient, linear-time training, as introduced below.

#### Chunkwise parallel form.

To summarize, the chunkwise parallel form splits inputs and outputs into several chunks of size $C$, and computes outputs for each chunk based on the final state of the previous chunk and the query/key/value blocks of the current chunk. Following the notation of [Suna23, PMLRa24, NeurIP24], let’s take the query block, ${\bm{q}}$, as an example. We denote ${\mathbf{Q}}_{[t]}:={\bm{q}}_{\mathrm{tC}+1:(t+1)C+1}$ as the query block for chunk $t$, and ${\bm{q}}_{[t]}^{r}:={\bm{q}}_{\mathrm{tC}+r}$ as the $r$-th query within chunk $t$. The initial state of chunk $t$ is defined as ${\mathbf{S}}_{[t]}:={\mathbf{S}}_{[t]}^{0}={\mathbf{S}}_{[t-1]}^{C}$. By partially expanding the recurrence, we have

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad{\bm{o}}_{[t]}^{r}={\mathbf{S}}_{[t]}^{r}{\bm{q}}_{[t]}^{r}={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^{r}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}\left({\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^{r}\right)\in\mathbb{R}^{d_{v}}
$$

Equivalently, in matrix form:

$$
{\mathbf{S}}_{[t+1]}={\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\in\mathbb{R}^{d_{v}\times d_{k}},\qquad{\mathbf{O}}_{[t]}={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\mathbf{M}}\right){\mathbf{V}}_{[t]}\in\mathbb{R}^{C\times d_{v}}
$$

where ${\mathbf{M}}\in\mathbb{R}^{C\times C}$ is the causal mask. The above equations are rich in matrix multiplications (matmuls), and by setting $C$ to a multiple of 16, one can take advantage of tensor cores—specialized GPU units for efficient half-precision matmul operations—for hardware-efficient training. Typically, $C$ is set to a small constant (e.g., 64 as implemented in FLA [Zhang24]), ensuring that the overall computational complexity remains linear with respect to sequence length, enabling efficient modeling of extremely long sequences.

### 2.2 Mamba2: Linear attention with scalar-valued data-dependent decay

Mamba2 [Daoa24] can be represented by the following linear recurrence (up to specific parameterization):

$$
{\mathbf{S}}_{t}=\alpha_{t}{\mathbf{S}}_{t-1}+{\bm{v}}_{t}{\bm{k}}_{t}^{\top},\qquad{\bm{o}}_{t}={\mathbf{S}}_{t}{\bm{q}}_{t}
$$

where $\alpha_{t}\in(0,1)$ is a data-dependent scalar-valued decay term. In the following, we will highlight the decay terms in blue to facilitate a clearer comparison with vanilla linear attention. Define the cumulative decay product $\gamma_{j}=\prod_{i=1}^{j}\alpha_{i}$, and by expanding the recurrence, we can express the result in both a vector form (left) and a matrix parallel form (right):

$$
{\bm{o}}_{t}=\sum_{i=1}^{t}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}{\bm{v}}_{i}{\bm{k}}_{i}^{\top}\right){\bm{q}}_{t}=\sum_{i=1}^{t}{\bm{v}}_{i}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}{\bm{k}}_{i}^{\top}{\bm{q}}_{t}\right),\qquad{\mathbf{O}}=\left(\left({\mathbf{Q}}{\mathbf{K}}^{\top}\right)\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma}\right){\mathbf{V}}
$$

Here, $\Gamma\in\mathbb{R}^{L\times L}$ is a decay-aware causal mask where $\Gamma_{\mathrm{ij}}=\frac{\gamma_{i}}{\gamma_{j}}$ if $i\geq j$ and $\Gamma_{\mathrm{ij}}=0$ otherwise.

This parallel and recurrent formulation is referred to as state space duality (SSD) in [Daoa24]. Notably, this recurrence structure has also been employed in Gated RFA [ICLRa21], xLSTM [Beck24], and Gated RetNet [Sunb24].

#### Chunkwise parallel form.

Slightly abusing the notation, we define the local cumulative product of decays within the chunk as $\gamma_{[t]}^{j}=\prod_{i=\mathrm{tC}+1}^{\mathrm{tC}+j}\alpha_{i}$. Additionally, we define $(\Gamma_{[t]})_{\mathrm{ij}}=\frac{\gamma_{[t]}^{j}}{\gamma_{[t]}^{i}}$ for $i\geq j$ and 0 otherwise. By partially expanding the recurrence, we obtain the following equations:

$$
{\mathbf{S}}_{[t]}^{r}={\color[\mathrm{rgb}]{0,0,1}\gamma^{r}_{[t]}}{\mathbf{S}}_{[t]}+\sum_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}}}{\bm{v}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top},\qquad{\bm{o}}_{[t]}^{r}={\color[\mathrm{rgb}]{0,0,1}\gamma^{r}_{[t]}}{\mathbf{S}}_{[t]}^{r}{\bm{q}}_{[t]}^{r}={\mathbf{S}}_{[t]}{\bm{q}}_{[t]}^{r}+\sum_{i=1}^{r}{\bm{v}}_{[t]}^{i}\left({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma^{r}_{[t]}}{\gamma^{i}_{[t]}}}{\bm{k}}_{[t]}^{i\top}{\bm{q}}_{[t]}^{r}\right)
$$

This can be equivalently expressed in matrix form as:

$$
{\mathbf{S}}_{[t+1]}\qquad =\color[\mathrm{rgb}]{0,0,1}{\gamma_{[t]}^{C}}\color[\mathrm{rgb}]{0,0,0}{\mathbf{S}}_{[t]}+{\mathbf{V}}_{[t]}^{\top}{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\frac{\gamma_{[t]}^{C}}{\gamma_{[t]}}\right)}{\mathbf{K}}_{[t]}\tag{1}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+\left({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\right){\mathbf{V}}_{[t]}\tag{2}
$$

We observe that the (cumulative) decay term can be seamlessly integrated into the matmuls with minimal computational overhead. This ensures that the chunkwise parallel form remains efficient and compatible with high-performance tensor core-based acceleration.

### 2.3 Delta Networks: Linear Attention with Delta Rule

The delta update rule [Widrow60, ICMLa21] *dynamically* erases the value (${\bm{v}}_{t}^{\mathrm{old}}$) associated with the current input key (${\bm{k}}_{t}$) and writes a new value (${\bm{v}}_{t}^{\mathrm{new}}$), which is a linear combination of the current input value and the old value. This process updates a key-value association pair at each time step, where the scalar $\beta_{t}\in(0,1)$ determines the extent to which the old association is replaced by the new one, as shown below.

$$
{\mathbf{S}}_{t}\qquad ={\mathbf{S}}_{t-1}-\underbrace{\left({\mathbf{S}}_{t-1}{\bm{k}}_{t}\right)}_{ {\bm{v}}_{t}^{\mathrm{old}}}{\bm{k}}_{t}^{\top}+\underbrace{\left(\beta_{t}{\bm{v}}_{t}+(1-\beta_{t}){\mathbf{S}}_{t-1}{\bm{k}}_{t})\right)}_{ {\bm{v}}_{t}^{\mathrm{new}}}{\bm{k}}_{t}^{\top}={\mathbf{S}}_{t-1}\left({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}
$$

#### Chunkwise parallel form.

By partially expanding the recurrence, we have

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^{r}{\mathbf{I}}-\beta_{[t]}^{i}{\bm{k}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)}_{:={\mathbf{P}}_{[t]}^{r}}+\underbrace{\sum_{i=1}^{r}\left(\beta^{i}_{[t]}{\bm{v}}^{i}_{[t]}{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^{r}\left({\mathbf{I}}-\beta_{[t]}^{j}{\bm{k}}^{j}_{[t]}{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{t}^{r}}\tag{3}
$$

where ${\mathbf{P}}_{[t]}^{j}$ involves cumulative products of transition matrices. [NeurIP24] show these take the form of (generalized) Householder matrices, allowing memory-efficient computation through classical WY representation [Compua85]. Based on this, they introduce two compact representations to optimize the process:

$$
{\mathbf{P}}_{[t]}^{r}\qquad ={\mathbf{I}}-\sum_{i=1}^{r}\mathbf{w}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{k}\times d_{k}}\qquad{\mathbf{H}}_{[t]}^{r}=\sum_{i=1}^{r}\mathbf{u}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\in\mathbb{R}^{d_{v}\times d_{k}}\tag{4}
$$

$$
\mathbf{w}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{k}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{w}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\quad\mathbf{u}_{[t]}^{r}=\beta_{[t]}^{r}\left({\bm{v}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{u}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\tag{5}
$$

where $\mathbf{w}_{[t]}^{r}\in\mathbb{R}^{d_{k}}$ and $\mathbf{u}_{[t]}^{r}\in\mathbb{R}^{d_{v}}$. Substituting these back into Eq. [3](#S2.E3 "In Chunkwise parallel form. ‣ 2.3 Delta Networks: Linear Attention with Delta Rule ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule") and in matrix form, we have:

$$
{\mathbf{S}}_{[t+1]}\qquad ={\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)^{\top}{\mathbf{K}}_{[t]}\tag{6}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\mathbf{M}})\left({\mathbf{U}}_{[t]}-{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)\tag{7}
$$

## 3 Gated Delta Networks

### 3.1 Formulation: Gated Delta Rule

The proposed gated delta rule is simple yet effective:

$$
{\mathbf{S}}_{t}={\mathbf{S}}_{t-1}\left(\color[\mathrm{rgb}]{0,0,1}{\alpha_{t}}\color[\mathrm{rgb}]{0,0,0}({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top})\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}\tag{8}
$$

where the data-dependent gating term $\color[\mathrm{rgb}]{0,0,1}{\alpha_{t}}\color[\mathrm{rgb}]{0,0,0}\in(0,1)$ controls state decay. This formulation unifies the advantages of both gating mechanisms and the delta rule: the gating term enables adaptive memory management, while the delta update structure facilitates effective key-value association learning.

We present a formal analysis of the gated delta rule through the lens of the online learning framework introduced by [Liua24]. In this framework, recurrent state updates emerge as solutions to an online learning problem with objective function $\bm{L_{t}}(S)$. As shown in [Table 1](#table-01), recent linear RNN architectures typically incorporate a regularization term in their online learning objective to prevent state divergence from previous values, thereby enabling memory retention. However, this retention mechanism becomes problematic when the state becomes saturated with information. In such cases, each state must encode a superposition of multiple information pieces, making precise retrieval challenging. To address this limitation, Mamba2 and Gated DeltaNet introduce an adaptive scaling factor $\alpha_{t}$ that relaxes the regularization term, allowing controlled deviations between $S_{t}$ and $S_{t-1}$. This modification enables dynamic memory management through selective forgetting.

On the other hand, Linear Attention (LA) and Mamba2 use a simple linear prediction loss $\langle{\mathbf{S}}_{t}{\bm{k}}_{t},{\bm{v}}_{t}\rangle$, while Longhorn [Liua24] uses a more expressive online regression objective $\|{\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}\|^{2}$ for better modeling of key-value associations. The resulting Longhorn’s update rule closely resembles the delta update rule [+1], suggesting the superiority of the (gated) delta rule over Mamba2 in in-context associative recall.

From the perspective of fast weight programming [ICMLc22] and test-time training [Suna24], the hidden state ${\mathbf{S}}$ can be interpreted as a weight matrix, with the delta rule optimizing the objective $L({\mathbf{S}}_{t})=\frac{1}{2}\|{\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}\|^{2}$ via *online* stochastic gradient descent (SGD):

$$
{\mathbf{S}}_{t+1}\qquad ={\mathbf{S}}_{t}-\beta_{t}\nabla_{S}L({\mathbf{S}}_{t})={\mathbf{S}}_{t}-\beta_{t}({\mathbf{S}}_{t}{\bm{k}}_{t}-{\bm{v}}_{t}){\bm{k}}_{t}^{\top}={\mathbf{S}}_{t}\left({\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}\right)+\beta_{t}{\bm{v}}_{t}{\bm{k}}_{t}^{\top}
$$

where $\beta_{t}$ represents the (adaptive) learning rate. From this perspective, the gated delta rule can be viewed as incorporating an adaptive weight decay term $\alpha_{t}$ into the SGD update, a technique widely used in deep learning [Hertz91, Andriu23].

<span id="table-01"></span>

![Original paper Table 1](../../papers/gated-delta-networks/table-01.png)

**Table 1.** Comparison of different linear RNN models and their corresponding online learning objectives using the framework from [Liua24]. For convenience, we simplify Longhorn’s vector-valued ${\bm{\beta}}$ to scalar $\beta$.

### 3.2 Algorithm: Hardware-efficient Chunkwise training

In this subsection, we describe an efficient chunkwise algorithm for gated delta rule.

#### Chunkwise parallel form.

By partially expanding the recurrence, we have

$$
{\mathbf{S}}_{[t]}^{r}={\mathbf{S}}_{[t]}\underbrace{\left(\prod_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^{i}}\left({\mathbf{I}}-\beta_{[t]}^{i}{\bm{k}}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)\right)}_{:={\mathbf{P}}_{[t]}^{r}}+\underbrace{\sum_{i=1}^{r}\left(\beta^{i}_{[t]}{\bm{v}}^{i}_{[t]}{\bm{k}}_{[t]}^{i\top}\prod_{j=i+1}^{r}{\color[\mathrm{rgb}]{0,0,1}\alpha_{[t]}^{j}}\left({\mathbf{I}}-\beta_{[t]}^{j}{\bm{k}}^{j}_{[t]}{\bm{k}}_{[t]}^{j\top}\right)\right)}_{:={\mathbf{H}}_{[t]}^{r}}
$$

We adapt the WY representation in Eq. [4](#S2.E4 "In Chunkwise parallel form. ‣ 2.3 Delta Networks: Linear Attention with Delta Rule ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")-[5](#S2.E5 "In Chunkwise parallel form. ‣ 2.3 Delta Networks: Linear Attention with Delta Rule ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule") to incorporate the decay term as below,

$$
{\mathbf{P}}_{[t]}^{r}\qquad =\color[\mathrm{rgb}]{0,0,1}{\gamma_{[t]}^{r}}\color[\mathrm{rgb}]{0,0,0}\left({\mathbf{I}}-\sum_{i=1}^{r}\mathbf{w}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\right)\qquad {\mathbf{H}}_{[t]}^{r}\qquad =\sum_{i=1}^{r}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}^{r}}{\gamma_{t}^{i}}}\mathbf{u}_{[t]}^{i}{\bm{k}}_{[t]}^{i\top}\tag{9}
$$

$$
\mathbf{w}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{k}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{w}_{[t]}^{i}({\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\qquad \mathbf{u}_{[t]}^{r}\qquad =\beta_{[t]}^{r}\left({\bm{v}}_{[t]}^{r}-\sum_{i=1}^{r-1}\left(\mathbf{u}_{[t]}^{i}({\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{[t]}^{r}}{\gamma_{[t]}^{i}}}{\bm{k}}_{[t]}^{i\top}{\bm{k}}_{[t]}^{r})\right)\right)\tag{10}
$$

and the proof of correctness can be found at Appendix. Equivalently, in matrix form:

$$
{\mathbf{S}}_{[t+1]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\gamma_{[t]}^{C}}{\mathbf{S}}_{[t]}+\left({\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)^{\top}{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\frac{\gamma_{[t]}^{C}}{\gamma_{[t]}}\right)}{\mathbf{K}}_{[t]}\tag{11}
$$

$$
{\mathbf{O}}_{[t]}\qquad ={\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{Q}}_{[t]}{\mathbf{S}}_{[t]}^{\top}+({\mathbf{Q}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\odot{\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}})\left({\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}\right)\tag{12}
$$

#### Comparison to Eq.[1](#S2.E1 "In Chunkwise parallel form. ‣ 2.2 Mamba2: Linear attention with scalar-valued data-dependent decay ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")-[2](#S2.E2 "In Chunkwise parallel form. ‣ 2.2 Mamba2: Linear attention with scalar-valued data-dependent decay ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")

We can see that the key distinction lies in the replacement of the value block ${\mathbf{V}}_{[t]}$ with the “pseudo”-value term ${\mathbf{U}}_{[t]}-{\color[\mathrm{rgb}]{0,0,1}\mathrm{Diag}\left(\gamma_{[t]}\right)}{\mathbf{W}}_{[t]}{\mathbf{S}}_{[t]}^{\top}$. This modification resembles Eq.[6](#S2.E6 "In Chunkwise parallel form. ‣ 2.3 Delta Networks: Linear Attention with Delta Rule ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")-[7](#S2.E7 "In Chunkwise parallel form. ‣ 2.3 Delta Networks: Linear Attention with Delta Rule ‣ 2 Preliminary ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule"), but notably incorporates decay-awareness.

#### UT transform.

To maximize hardware efficiency, we apply the UT transform [Joffra06] to Eq. [10](#S3.E10 "In Chunkwise parallel form. ‣ 3.2 Algorithm: Hardware-efficient Chunkwise training ‣ 3 Gated Delta Networks ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule"). This technique reformulates operations into matmul form, reducing non-matmul FLOPs, which is crucial to enable better hardware utilization during training [Daob23, Refc23, PMLRa24].

$$
{\mathbf{W}}_{[t]}\qquad ={\mathbf{A}}^{W}_{[t]}\mathrm{Diag}(\beta_{[t]}){\mathbf{K}}_{[t]},\qquad {\mathbf{A}}^{W}_{[t]}=\left({\mathbf{I}}-\mathrm{lower}(\mathrm{Diag}(\beta_{[t]}){\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^{\top})\right)^{-1}
$$

$$
{\mathbf{U}}_{[t]}\qquad ={\mathbf{A}}^{U}_{[t]}\mathrm{Diag}\left(\beta_{[t]}\right){\mathbf{V}}_{[t]},\qquad {\mathbf{A}}^{U}_{[t]}=\left({\mathbf{I}}-\mathrm{lower}\left(\mathrm{Diag}(\beta_{[t]})\left({\color[\mathrm{rgb}]{0,0,1}\Gamma_{[t]}}\odot{\mathbf{K}}_{[t]}{\mathbf{K}}_{[t]}^{\top}\right)\right)\right)^{-1}
$$

where $\mathrm{lower}(\cdot):=\mathrm{tril}(\cdot,-1)$; and the inverse of a lower triangle matrix can be calculated efficiently by forward substitution.

#### Remark on speed.

Similar to Mamba2, the gating term (colored in blue) only performs elementwise multiplication with (intermediate) variables without affecting matrix multiply structures, enabling tensor core GPU optimization. As shown in [Fig. 3](#figure-03), Gated DeltaNet maintains the same speed as DeltaNet, with only a small performance gap to Mamba2 despite having a more complex and expressive transition matrix.

### 3.3 Gated Delta Networks and Hybrid Models

<span id="figure-01"></span>

![figure 1](../../papers/gated-delta-networks/figure-01.png)

**Figure 1.** Visualization of the (hybrid) architecture and block design of Gated DeltaNet models. Gated DeltaNet-H1 and H2 use Gated DeltaNet + SWA and Mamba2 + Gated DeltaNet + SWA patterns, respectively. In the block design, query/key paths consist of linear proj., shortconv., SiLU and L2 norm; value path includes linear proj., shortconv. and SiLU; alpha/beta use linear proj.; and output gate applies linear proj. with SiLU.

#### Token mixer block.

The basic Gated DeltaNet follows Llama’s macro architecture, stacking token mixer layers with SwiGLU MLP layers, but replaces self-attention with gated delta rule token mixing. [Fig. 1](#figure-01) (right) shows its block design. For the gated delta rule (Eq. [8](#S3.E8 "In 3.1 Formulation: Gated Delta Rule ‣ 3 Gated Delta Networks ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")), queries, keys and values $\{ {\bm{q}},{\bm{k}},{\bm{v}}\}$ are generated through linear projection, short convolution and SiLU, with L2 normalization applied to ${\bm{q}},{\bm{k}}$ for training stability. $\alpha,\beta$ use linear projection only. [+2] Following [Suna23], the output is processed through normalization and gating before applying output projection.

#### Hybrid models.

Linear transformers have limitations in modeling local shifts and comparisons, and their fixed state size makes it hard for retrieval tasks [Arora24]. Following recent hybrid architectures like Griffin [De24] and Samba [Ren24], we combine linear recurrent layers with sliding window attention (SWA), resulting in GatedDeltaNet-H1. We also stack Mamba2, GatedDeltaNet and SWA, resulting in GatedDeltaNet-H2.

## 4 Experiments

#### Setup

Our experiments encompass a comprehensive comparison of recent state-of-the-art architectures, including pure Transformer models, RNN-based approaches, and hybrid architectures. We evaluate against the following baselines: RetNet [Suna23], HGRN2 [Qin24a], Mamba [Daoc23], Mamba2 [Daob24], Samba [Ren24], and DeltaNet [NeurIP24]. For fair comparison, all models are trained under identical conditions with 1.3B parameters on 100B tokens sampled from the FineWeb-Edu dataset [Penedo24]. We use the AdamW optimizer with a peak learning rate of 4e-4, weight decay of 0.1, and gradient clipping of 1.0. The learning rate follows a cosine annealing schedule with a 1B token warm-up period and batch size of 0.5M tokens. All models employ the LLaMA 2 tokenizer with a vocabulary size of 32,000. For sequence modeling, we set the training length to 4K tokens, with Samba and our hybrid models using a sliding window size of 2K. See appendix for evaluation settings.

<span id="table-02"></span>

![Original paper Table 2](../../papers/gated-delta-networks/table-02.png)

**Table 2.** Performance comparison on language modeling and zero-shot common-sense reasoning.

#### Common-sense reasoning

In [Table 2](#table-02), we present the language modeling perplexity and zero-shot accuracy on commonsense reasoning benchmarks for models with 400M and 1.3B parameters. Gated DeltaNet consistently outperforms other linear models, including RetNet, HGRN2, Mamba, Mamba2, and DeltaNet, at both scales. As expected, the hybrid variant further enhances performance.

<span id="table-03"></span>

![Original paper Table 3](../../papers/gated-delta-networks/table-03.png)

**Table 3.** Performance comparison on S-NIAH benchmark suite.

#### In-context retrieval on synthetic data

[Table 3](#table-03) shows the results on Single Needle-In-A-Haystack (S-NIAH) benchmark suite from RULER [Hsieh24]. In the simplest S-NIAH-1 setting with synthetic inputs, DeltaNet achieves near-perfect performance across all sequence lengths, benefiting from its delta update rule which is specifically advantageous for in-context recall (§[3.1](#S3.SS1 "3.1 Formulation: Gated Delta Rule ‣ 3 Gated Delta Networks ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")). In contrast, Gated DeltaNet shows slightly lower retrieval accuracy since its gating mechanism discards information, compromising perfect memory retention, while Mamba2’s performance degrades significantly beyond 2K sequences.

However, retrieval from memory depends on not only retention but also the ability to "forget": given fixed state size, lack of memory clearance leads to memory collision when the state becomes saturated - multiple pieces of information become superimposed, making them indistinguishable. This becomes evident in NIAH-2 and NIAH-3 where needles are grounded in real-world text data: DeltaNet’s performance degrades significantly, while Gated DeltaNet’s adaptive memory management demonstrates clear advantages over both Mamba2 and DeltaNet.

#### In-context retrieval on real-world data

<span id="table-04"></span>

![Original paper Table 4](../../papers/gated-delta-networks/table-04.png)

**Table 4.** Accuracy on recall-world retrieval tasks with input truncated to 2K tokens. SQD: SQUADE. TQA: Trivial QA.

[Table 4](#table-04) presents results on real-world recall-intensive tasks used by [Aroraa24]. As expected, linear recurrent models show a significant performance gap compared to Transformers, while hybrid models combining linear recurrence and attention outperform pure attention models in retrieval tasks.

For pure recurrent models, despite DeltaNet’s superior performance on synthetic in-context retrieval tasks [NeurIP24], its real-world retrieval performance lags behind Mamba2, consistent with our observations in S-NIAH-2 and S-NIAH-3 ([Table 3](#table-03)). Gated DeltaNet outperforms both DeltaNet and Mamba2 thanks to its gated delta rule, though the improvement margin is smaller than in [Table 3](#table-03). We attribute this reduced performance gap to instruction-unaligned small language models being prone to repetition errors, which are the primary source of errors in these tasks (cf. [Aroraa24]). Since this issue is largely independent of the update rule choice, the performance differences between models are less pronounced compared to [Table 3](#table-03).

<span id="figure-02"></span>

![figure 2](../../papers/gated-delta-networks/figure-02.png)

**Figure 2.** Length extrapolation on six long benchmarks.

#### Length extrapolation on long sequences.

As shown in Fig.[2](#figure-02), we evaluate the models’ capacity to extrapolate to sequences of up to 20K tokens across six long-context benchmarks. Gated DeltaNet achieves the lowest overall perplexity across tasks among RNN models. While we observe mixed results in length extrapolation, Gated DeltaNet exhibits relatively more robust performance, suggesting better memory management. The hybrid models further improve upon this by leveraging attention for local context modeling, which reduces the memory management burden on their recurrent components. Future work will explore these models’ capabilities on even longer sequences.

#### Long context understanding

As demonstrated in [Table 5](#table-05), we evaluated the models’ performance on LongBench [Bai23]. In recurrent models, Gated DeltaNet shows consistent advantages, especially in single-doc QA, few-shot in-context learning, and Code tasks, demonstrating its superior capabilities in retrieval, in-context learning, and state tracking, respectively.

<span id="table-05"></span>

![Original paper Table 5](../../papers/gated-delta-networks/table-05.png)

**Table 5.** Accuracy on 14 tasks from LongBench [Bai23]: Narrative QA, QasperQA, MultiField QA, HotpotQA, 2WikiMulti QA, Musique, GovReport, QMSum, MultiNews, TRec, Trivia QA, SamSum, LCC, and RepoBench-P by order.

<span id="figure-03"></span>

![figure 3](../../papers/gated-delta-networks/figure-03.png)

**Figure 3.** Training throughput comparison of 1.3B models on a single H100 GPU.

#### Throughput Comparison.

The training throughput comparison across different models is presented in [Fig. 3](#figure-03). As our analysis shows, the proposed gated delta rule introduces only marginal overhead compared to the original delta rule, with Gated DeltaNet achieving essentially the same throughput as DeltaNet. Both are slightly slower than Mamba2 (2-3K tokens/sec) due to their more expressive transition matrices.

The Transformer++ achieves the best performance in the 2K context window domain, thanks to the highly optimized Flash-Attention-2 kernel [Daob23]. Consequently, hybrid approaches combining 2K window-size SWA attention with other token mixers demonstrate higher throughput than standalone mixers: Samba outperforms Mamba, while Gated DeltaNet-H1 and -H2 outperform Gated DeltaNet. Notably, Gated DeltaNet-H1 maintains compelling training throughput across all sequence lengths, even on short sequences.

## 5 Related Work

#### Gated linear RNN.

Large linear recurrent language models have attracted significant attention due to their training and inference efficiency. The field of linear RNNs has rapidly evolved from using data-independent decay mechanisms, as exemplified by models like S4 [ICLRb22], S5 [ICLRb23], LRU [ICMLa23], RWKV4/5 [EMNLP23], and RetNet [Suna23], to incorporating data-dependent decay mechanisms in more recent architectures such as HGRN1/2 [Qin24a, Qina23], Mamba1/2 [Daoc23, Daoa24], RWKV6 [Peng24], and GSA [Gated24]. This transition stems from the proven advantages of gating/forgetting mechanisms (termed selective mechanisms in Mamba)—a classical concept originating in the gated RNN literature [Gersa00] whose significance has been consistently reaffirmed [Greff15, Lasenb18, Qin24a, Qina23, Daoc23].

Modern forget gates differ from traditional designs like those in LSTM by removing the dependency on the previous hidden state, relying solely on input data. This modification enables efficient parallelism across sequence lengths [Mar18, Qina23]. The absence of a forget gate has been a notable limitation in DeltaNet, and our gated extension addresses this gap in a natural and effective way.

#### Delta rule.

The delta learning rule has been shown to offer superior memory capacity compared to the Hebbian learning rule [Gardne88, Kak89]. While linear transformers rely on a Hebbian-like learning rule, DeltaNet utilizes the delta rule, and this advantage in memory capacity is empirically evident in synthetic in-context learning tasks. Moreover, this superiority extends across various applications, including language modeling [Irie21, NeurIP24], reinforcement learning [ICMLd22], and image generation [ICLRa23]. [NeurIP24] further parallelized delta rule computations across sequence lengths and demonstrated the enhanced expressiveness of DeltaNet’s transition matrix. Specifically, DeltaNet’s data-dependent identity-plus-low-rank structure (${\mathbf{I}}-\beta_{t}{\bm{k}}_{t}{\bm{k}}_{t}^{\top}$) offers greater flexibility compared to Mamba2’s data-dependent diagonal matrices ($\alpha_{t}{\mathbf{I}}$). This architectural shift from diagonal to structured dense matrices substantially improves the model’s capabilities in complex reasoning tasks, including regular language recognition [Bethar24, Grazzi24] and state-tracking tasks beyond the TC0 complexity class [Merril24]—capabilities that are particularly crucial for applications like coding. Recent work by [Grazzi24] suggests that allowing negative eigenvalues in DeltaNet could further enhance its state tracking capabilities, which could be directly incorporated into Gated DeltaNet as well.

The delta rule shares an intriguing connection with online (meta) learning via gradient descent [Munkhd19, ICMLc22]. Recent architectures like Longhorn [Liua24] and TTT [Suna24] revisit this relationship by reformulating state space learning as a gradient-based online learning problem (see also §[3.1](#S3.SS1 "3.1 Formulation: Gated Delta Rule ‣ 3 Gated Delta Networks ‣ Gated Delta Networks: Improving Mamba2 with Delta Rule")). While Longhorn offers a more theoretically rigorous formulation, its reliance on diagonal approximation significantly compromises expressiveness. TTT presents an interesting case: its linear variant without Layernorm is equivalent to DeltaNet, but adding Layernorm transforms it into a non-linear RNN model. This transformation necessitates a hybrid training approach where a "delta-like-rule" is applied at the chunk level every N tokens (where N is the chunk size).

Despite its advantages, the delta rule has theoretical limitations [Bali23] and shows moderate performance on real-world datasets [NeurIP24]. Previous extensions enhance expressiveness through strict *nonlinear* recurrence [Irie21, ICMLd22], but sacrifice training parallelism. Our Gated DeltaNet maintains a linear RNN, enabling efficient training while improving expressiveness through gating, leading to consistent improvement across tasks. Future work could adopt GLA-like diagonal gating [PMLRa24] to further relax gating restrictions.

## 6 Conclusion

In this work, we introduced Gated DeltaNet, which enables better key-value association learning compared to Mamba2 and more adaptive memory clearance than DeltaNet, leading to consistently better empirical results across various tasks. We extended the parallel algorithm from  [NeurIP24] to enable hardware-efficient training of Gated DeltaNet. Our hybrid Gated DeltaNet model achieves even higher training throughput and overall performance, making it well-suited for practical deployment.

## Acknowledgment

We thank Yu Zhang for assistance with figure drawing, Simeng Sun and Zhixuan Lin for valuable discussions on the evaluation, and Eric Alcaide for insightful feedback on the online learning perspective of DeltaNet.

## Appendix A

### A.1 Extended WY representation for gated delta rule

To reduce notation clutter, we only consider the first chunk here.

For ${\mathbf{S}}_{t}$, the extended WY representation is

$$
{\mathbf{S}}_{t}=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top},\qquad\mathbf{u}_{t}=\beta_{t}\left({\bm{v}}_{t}-\sum_{i=1}^{t-1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^\top{\bm{k}}_{t}\right)
$$

We proof this by mathmetical induction.

###### Proof.

$$
\centering{\mathbf{S}}_{t+1}\@\mathrm{add}@\mathrm{centering}\qquad ={\mathbf{S}}_{t}\left({\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}({\mathbf{I}}-\beta_{t+1}{\bm{k}}_{t+1}{\bm{k}}_{t+1}^{\top})\right)+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
={\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}(\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top})-{\color[\mathrm{rgb}]{0,0,1}\alpha_{t+1}}\beta_{t+1}(\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}{\bm{k}}_{i}{\bm{k}}_{t+1}^{\top})+\beta_{t+1}{\bm{v}}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}+\underbrace{\beta_{t+1}\left({\bm{v}}_{t+1}-\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^\top{\bm{k}}_{t+1}\right)}_{\mathbf{u}_{t+1}}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}+\underbrace{ {\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{t+1}}}}_{1}\mathbf{u}_{t+1}{\bm{k}}_{t+1}^{\top}
$$

$$
=\sum_{i=1}^{t+1}{\color[\mathrm{rgb}]{0,0,1}\frac{\gamma_{t+1}}{\gamma_{i}}}\mathbf{u}_{i}{\bm{k}}_{i}^{\top}
$$

∎

For ${\mathbf{P}}_{t}$,

$$
{\mathbf{P}}_{t}\qquad =\prod_{i=1}^{t}{\color[\mathrm{rgb}]{0,0,1}\alpha_{t}}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)
$$

$$
={\color[\mathrm{rgb}]{0,0,1}\underbrace{\left(\prod_{i=1}^{t}\alpha_{t}\right)}_{\gamma_{t}}}\underbrace{\left(\prod_{i=1}^{t}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)\right)}_{ {\mathbf{I}}-\sum_{i=1}^{t}\mathbf{w}_{i}{\bm{k}}_{i}^{\top}}
$$

and

$$
\prod_{i=1}^{t}\left({\mathbf{I}}-\beta_{i}{\bm{k}}_{i}{\bm{k}}_{i}^{\top}\right)={\mathbf{I}}-\sum_{i=1}^{t}\mathbf{w}_{i}{\bm{k}}_{i}^{\top},\quad\mathbf{w}_{n}=\beta_{n}{\bm{k}}_{n}-\beta_{n}\sum_{t=1}^{n-1}\left(\mathbf{w}_{t}({\bm{k}}_{t}^{\top}{\bm{k}}_{n}\right)
$$

has already been proved in [NeurIP24].

### A.2 Ablation Study

<span id="table-06"></span>

![Original paper Table 6](../../papers/gated-delta-networks/table-06.png)

**Table S.1.** Ablation study on the Gated DeltaNet block. Avg-PPL and Avg-Acc denote average perplexity and zero-shot commonsense reasoning accuracy (as in [Table 2](#table-02)), respectively. All models have 400M parameters and are trained for 15B tokens on the same subset of FineWeb-Edu dataset [Penedo24].

<span id="table-07"></span>

![Original paper Table 7](../../papers/gated-delta-networks/table-07.png)

**Table S.2.** Ablation studies of Gated DeltaNet models. All evaluations are performed by using lm-evaluation-harness [Gaob21]. All models use the Llama tokenizer and are trained on the same subset of the FineWeb-Edu dataset [Penedo24].

[Table S.1](#table-06) presents ablation studies on the Gated DeltaNet block’s components. Our experiments demonstrate that both the short convolution and output gate are crucial for model performance, while output normalization yields marginal improvements. Consistent with [NeurIP24], we found L2 normalization to be essential for optimal performance, though the choice of feature map was less influential. Nevertheless, SiLU consistently outperformed other activation functions, aligning with observations from [Qin23]. Through empirical analysis, we determined that a head dimension of 128 provides an optimal trade-off between performance and computational efficiency. Additionally, [Table S.2](#table-07) demonstrates that among various hybrid architectures, the combination of Mamba2, Gated DeltaNet, and SWA in this specific order produces superior results.

## Appendix B Experimental settings

### B.1 Evaluation

#### Commonsense reasoning

Following [Daoc23], we evaluate our model on multiple commonsense reasoning benchmarks: PIQA [AAAIc20], HellaSwag [Italyb19], WinoGrande [AAAId20], ARC-easy (ARC-e) and ARC-challenge (ARC-c) [Clark18], SIQA [Sap19], BoolQ [Clark19], Wikitext [ICLR17], and LAMBADA [August16].

#### In-context retrieval

Our evaluation comprises both synthetic and real-world tasks. For synthetic tasks, we utilize the Needle-In-A-Haystack Single (NIAH-S) benchmark suite from RULER [Hsieh24], which includes three increasingly complex tasks: S-NIAH-1 (passkey retrieval), S-NIAH-2 (numerical needle in haystack), and S-NIAH-3 (word-based needle in haystack). For real-world tasks, following [Aroraa24], we evaluate on diverse datasets: SWDE [Lockar19] for structured HTML relation extraction, FDA [Aroraa23] for PDF key-value retrieval, and several question-answering datasets including SQuAD [Austra18], TriviaQA [Canada17], Drop [Duaa19], and NQ [Kwiatk19]. Since our pretrained models lack instruction tuning, we employ the Cloze Completion Formatting prompts provided by [Aroraa24], which better align with our models’ next-word-prediction training objective.

#### Long context understanding

We evaluate on 14 tasks from Longbench [Bai23], encompassing: narrative comprehension (Narrative QA [Lingui18]), scientific understanding (QasperQA [Dasigi21]), multi-hop reasoning (MultiField QA, HotpotQA [Tsujib18], 2WikiMulti QA [Online20], Musique [Lingui22]), document summarization (GovReport [Huang21], QMSum [Zhong21], MultiNews [Fabbri19]), and various specialized tasks (TRec [COLING02], Trivia QA [Canada17], SamSum [China19], LCC [Guoa23], and RepoBench-P [Liub23]).

[+1]: The theoretical distinction lies in the optimization approach: Longhorn uses implicit online learning [Bartle10] to derive closed-form globally optimal updates, while DeltaNet optimizes the same objective through one-step explicit gradient descent, as noted by [Liua24]. Despite Longhorn’s stronger theoretical grounding, we found no significant empirical performance differences between these approaches and thus maintain DeltaNet’s original formulations.

[+2]: We use Mamba2’s parameterization for $\alpha$ but omit it for brevity.
