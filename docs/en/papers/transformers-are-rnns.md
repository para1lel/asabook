---
title: 'Transformers are RNNs'
createTime: 2026/09/13 01:44:07
permalink: /en/papers/transformers-are-rnns/
pageClass: paper-reading
---

> [Angelos Katharopoulos](https://angeloskath.github.io/), [Apoorv Vyas](https://apoorv2904.github.io/), [Nikolaos Pappas](https://nik0spapp.github.io/), [François Fleuret](https://fleuret.org/francois/) [+affiliations]. First submitted to arXiv on June 29, 2020; current version v3, revised August 31, 2020. Published in the *Proceedings of the 37th International Conference on Machine Learning*, PMLR 119:5156-5165, July 13-18, 2020. [Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention](https://arxiv.org/abs/2006.16236). <a href="/paper/transformers-are-rnns.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [ICML 2020](https://proceedings.mlr.press/v119/katharopoulos20a.html). [DOI](https://doi.org/10.48550/arXiv.2006.16236). [TeX source](https://export.arxiv.org/e-print/2006.16236v3). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Transformers achieve remarkable performance in several tasks but due to their quadratic complexity, with respect to the input’s length, they are prohibitively slow for very long sequences. To address this limitation, we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from $\mathcal{O}\left(N^{2}\right)$ to $\mathcal{O}\left(N\right)$, where $N$ is the sequence length. We show that this formulation permits an iterative implementation that dramatically accelerates autoregressive transformers and reveals their relationship to recurrent neural networks. Our *linear transformers* achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences.

<span id="section-1"></span>

## 1 Introduction

Transformer models were originally introduced by [Vas17] in the context of neural machine translation [Sut14, Bah14] and have demonstrated impressive results on a variety of tasks dealing with natural language [Dev18], audio [Spe18], and images [Par19b]. Apart from tasks with ample supervision, transformers are also effective in transferring knowledge to tasks with limited or no supervision when they are pretrained with autoregressive [Rad18, Rad19] or masked language modeling objectives [Dev18, Yan20d, Son19, Liu19a].

However, these benefits often come with a very high computational and memory cost. The bottleneck is mainly caused by the global receptive field of self-attention, which processes contexts of $N$ inputs with a quadratic memory and time complexity $\mathcal{O}\left(N^{2}\right)$. As a result, in practice transformers are slow to train and their context is *limited*. This disrupts temporal coherence and hinders the capturing of long-term dependencies. [Dai19] addressed the latter by attending to memories from previous contexts albeit at the expense of computational efficiency.

Lately, researchers shifted their attention to approaches that increase the context length without sacrificing efficiency. Towards this end, [Chi19] introduced sparse factorizations of the attention matrix to reduce the self-attention complexity to $\mathcal{O}\left(N\sqrt{N}\right)$. [Kit20] further reduced the complexity to $\mathcal{O}\left(N\log N\right)$ using locality-sensitive hashing. This made scaling to long sequences possible. Even though the aforementioned models can be efficiently trained on large sequences, they do not speed-up autoregressive inference.

In this paper, we introduce the *linear transformer* model that significantly reduces the memory footprint and scales linearly with respect to the context length. We achieve this by using a kernel-based formulation of self-attention and the associative property of matrix products to calculate the self-attention weights ([Section 3.2](#section-3-2)). Using our linear formulation, we also express causal masking with linear complexity and constant memory ([Section 3.3](#section-3-3)). This reveals the relation between transformers and RNNs, which enables us to perform autoregressive inference orders of magnitude faster ([Section 3.4](#section-3-4)).

Our evaluation on image generation and automatic speech recognition demonstrates that *linear transformer* can reach the performance levels of transformer, while being up to three orders of magnitude faster during inference.

<span id="section-2"></span>

## 2 Related Work

In this section, we provide an overview of the most relevant works that seek to address the large memory and computational requirements of transformers. Furthermore, we discuss methods that theoretically analyze the core component of the transformer model, namely self-attention. Finally, we present another line of work that seeks to alleviate the softmax bottleneck in the attention computation.

<span id="section-2-1"></span>

### 2.1 Efficient Transformers

Existing works seek to improve memory efficiency in transformers through weight pruning [Mic19], weight factorization [Lan20], weight quantization [Zaf19] or knowledge distillation. [Cla20] proposed a new pretraining objective called replaced token detection that is more sample efficient and reduces the overall computation. [Lam19] used product-key attention to increase the capacity of any layer with negligible computational overhead.

Reducing the memory or computational requirements with these methods leads to training or inference time speedups, but, fundamentally, the time complexity is still quadratic with respect to the sequence length which hinders scaling to long sequences. In contrast, we show that our method reduces both memory and time complexity of transformers both theoretically ([Section 3.2](#section-3-2)) and empirically ([Section 4.1](#section-4-1)).

Another line of research aims at increasing the “context” of self-attention in transformers. Context refers to the maximum part of the sequence that is used for computing self-attention. [Dai19] introduced Transformer-XL which achieves state-of-the-art in language modeling by learning dependencies beyond a fixed length context without disrupting the temporal coherence. However, maintaining previous contexts in memory introduces significant additional computational cost. In contrast, [Suk19] extended the context length significantly by learning the optimal attention span per attention head, while maintaining control over the memory footprint and computation time. Note that both approaches have the same asymptotic complexity as the vanilla model. In contrast, we improve the asymptotic complexity of the self-attention, which allows us to use significantly larger context.

More related to our model are the works of [Chi19] and [Kit20]. The former [Chi19] introduced sparse factorizations of the attention matrix reducing the overall complexity from quadratic to $\mathcal{O}\left(N\sqrt{N}\right)$ for generative modeling of long sequences. More recently, [Kit20] proposed Reformer. This method further reduces complexity to $\mathcal{O}\left(N\log{N}\right)$ by using locality-sensitive hashing (LSH) to perform fewer dot products. Note that in order to be able to use LSH, Reformer constrains the keys, for the attention, to be identical to the queries. As a result this method cannot be used for decoding tasks where the keys need to be different from the queries. In comparison, *linear transformers* impose no constraints on the queries and keys and scale linearly with respect to the sequence length. Furthermore, they can be used to perform inference in autoregressive tasks three orders of magnitude faster, achieving comparable performance in terms of validation perplexity.

<span id="section-2-2"></span>

### 2.2 Understanding Self-Attention

There have been few efforts to better understand self-attention from a theoretical perspective. [Tsa19] proposed a kernel-based formulation of attention in transformers which considers attention as applying a kernel smoother over the inputs with the kernel scores being the similarity between inputs. This formulation provides a better way to understand attention components and integrate the positional embedding. In contrast, we use the kernel formulation to speed up the calculation of self-attention and lower its computational complexity. Also, we observe that if a kernel with positive similarity scores is applied on the queries and keys, linear attention converges normally.

More recently, [Cor20] provided theoretical proofs and empirical evidence that a multi-head self-attention with sufficient number of heads can express any convolutional layer. Here, we instead show that a self-attention layer trained with an autoregressive objective can be seen as a recurrent neural network and this observation can be used to significantly speed up inference time of autoregressive transformer models.

<span id="section-2-3"></span>

### 2.3 Linearized softmax

For many years, softmax has been the bottleneck for training classification models with a large number of categories [Goo01, Mor05, Mni09]. Recent works [Bla17, Raw19], have approximated softmax with a linear dot product of feature maps to speed up the training through sampling. Inspired from these works, we linearize the softmax attention in transformers. Concurrently with this work, [She21] explored the use of linearized attention for the task of object detection in images. In comparison, we do not only linearize the attention computation, but also develop an autoregressive transformer model with linear complexity and constant memory for both inference and training. Moreover, we show that through the lens of kernels, every transformer can be seen as a recurrent neural network.

<span id="section-3"></span>

## 3 Linear Transformers

In this section, we formalize our proposed *linear transformer*. We present that changing the attention from the traditional *softmax* attention to a feature map based dot product attention results in better time and memory complexity as well as a causal model that can perform sequence generation in linear time, similar to a recurrent neural network.

Initially, in [Section 3.1](#section-3-1), we introduce a formulation for the transformer architecture introduced in [Vas17]. Subsequently, in [Section 3.2](#section-3-2) and [Section 3.3](#section-3-3) we present our proposed *linear transformer* and finally, in [Section 3.4](#section-3-4) we rewrite the transformer as a recurrent neural network.

<span id="section-3-1"></span>

### 3.1 Transformers

Let $x\in\mathbb{R}^{N\times F}$ denote a sequence of $N$ feature vectors of dimensions $F$. A transformer is a function $T:\mathbb{R}^{N\times F}\to\mathbb{R}^{N\times F}$ defined by the composition of $L$ transformer layers $T_{1}(\cdot),\dots,T_{L}(\cdot)$ as follows,

<span id="equation-01"></span>

$$
T_{l}(x)=f_{l}(A_{l}(x)+x).
$$

The function $f_{l}(\cdot)$ transforms each feature independently of the others and is usually implemented with a small two-layer feedforward network. $A_{l}(\cdot)$ is the self attention function and is the only part of the transformer that acts across sequences.

The self attention function $A_{l}(\cdot)$ computes, for every position, a weighted average of the feature representations of all other positions with a weight proportional to a similarity score between the representations. Formally, the input sequence $x$ is projected by three matrices $W_{Q}\in\mathbb{R}^{F\times D}$, $W_{K}\in\mathbb{R}^{F\times D}$ and $W_{V}\in\mathbb{R}^{F\times M}$ to corresponding representations $Q$, $K$ and $V$. The output for all positions, $A_{l}(x)=V^{\prime}$, is computed as follows,

<span id="equation-02"></span>

$$
\begin{aligned}
Q & =xW_{Q}, \\
K & =xW_{K}, \\
V & =xW_{V}, \\
A_{l}(x)=V^{\prime} & =\mathrm{softmax}\left(\frac{Q K^\top}{\sqrt{D}}\right)V.
\end{aligned}
$$

Note that in the previous equation, the softmax function is applied rowwise to $Q K^\top$. Following common terminology, the $Q$, $K$ and $V$ are referred to as the “queries”, “keys” and “values” respectively.

[Equation 2](#equation-02) implements a specific form of self-attention called softmax attention where the similarity score is the exponential of the dot product between a query and a key. Given that subscripting a matrix with $i$ returns the $i$-th row as a vector, we can write a generalized attention equation for any similarity function as follows,

<span id="equation-03"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{N}\mathrm{sim}\left(Q_{i},K_{j}\right)V_{j}}{\sum_{j=1}^{N}\mathrm{sim}\left(Q_{i},K_{j}\right)}.
$$

[Equation 3](#equation-03) is equivalent to [Equation 2](#equation-02) if we substitute the similarity function with $\mathrm{sim}\left(q,k\right)=\exp\left(\frac{q^\top k}{\sqrt{D}}\right)$.

<span id="section-3-2"></span>

### 3.2 Linearized Attention

The definition of attention in [Equation 2](#equation-02) is generic and can be used to define several other attention implementations such as polynomial attention or RBF kernel attention [Tsa19]. Note that the only constraint we need to impose to $\mathrm{sim}\left(\cdot\right)$, in order for [Equation 3](#equation-03) to define an attention function, is to be non-negative. This includes all kernels $k(x,y):\mathbb{R}^{2\times F}\to\mathbb{R}_{+}$.

Given such a kernel with a feature representation $\phi\left(x\right)$ we can rewrite [Equation 2](#equation-02) as follows,

<span id="equation-04"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{N}\phi\left(Q_{i}\right)^\top\phi\left(K_{j}\right)V_{j}}{\sum_{j=1}^{N}\phi\left(Q_{i}\right)^\top\phi\left(K_{j}\right)},
$$

and then further simplify it by making use of the associative property of matrix multiplication to

<span id="equation-05"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{N}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{N}\phi\left(K_{j}\right)}.
$$

The above equation is simpler to follow when the numerator is written in vectorized form as follows,

<span id="equation-06"></span>

$$
\left(\phi\left(Q\right)\phi\left(K\right)^\top\right)V=\phi\left(Q\right)\left(\phi\left(K\right)^\top V\right).
$$

Note that the feature map $\phi\left(\cdot\right)$ is applied rowwise to the matrices $Q$ and $K$.

From [Equation 2](#equation-02), it is evident that the computational cost of softmax attention scales with $\mathcal{O}\left(N^{2}\right)$, where $N$ represents the sequence length. The same is true for the memory requirements because the full attention matrix must be stored to compute the gradients with respect to the queries, keys and values. In contrast, our proposed *linear transformer* from [Equation 5](#equation-05) has time and memory complexity $\mathcal{O}\left(N\right)$ because we can compute $\sum_{j=1}^{N}\phi\left(K_{j}\right)V_{j}^\top$ and $\sum_{j=1}^{N}\phi\left(K_{j}\right)$ once and reuse them for every query.

<span id="section-3-2-1"></span>

#### 3.2.1 Feature Maps and Computational Cost

For softmax attention, the total cost in terms of multiplications and additions scales as $\mathcal{O}\left(N^{2}\max\left(D,M\right)\right)$, where $D$ is the dimensionality of the queries and keys and $M$ is the dimensionality of the values. On the contrary, for linear attention, we first compute the feature maps of dimensionality $C$. Subsequently, computing the new values requires $\mathcal{O}\left(N C M\right)$ additions and multiplications.

The previous analysis does not take into account the choice of kernel and feature function. Note that the feature function that corresponds to the exponential kernel is infinite dimensional, which makes the linearization of exact softmax attention infeasible. On the other hand, the polynomial kernel, for example, has an exact finite dimensional feature map and has been shown to work equally well with the exponential or RBF kernel [Tsa19]. The computational cost for a linearized polynomial transformer of degree 2 is $\mathcal{O}\left(N D^{2} M\right)$. This makes the computational complexity favorable when $N>D^{2}$. Note that this is true in practice since we want to be able to process sequences with tens of thousands of elements.

For our experiments, that deal with smaller sequences, we employ a feature map that results in a positive similarity function as defined below,

<span id="equation-07"></span>

$$
\phi\left(x\right)=\mathrm{elu}(x)+1,
$$

where $\mathrm{elu}(\cdot)$ denotes the exponential linear unit [Cle16] activation function. We prefer $\mathrm{elu}(\cdot)$ over $\mathrm{relu}(\cdot)$ to avoid setting the gradients to 0 when $x$ is negative. This feature map results in an attention function that requires $\mathcal{O}\left(N D M\right)$ multiplications and additions. In our experimental section, we show that the feature map of [Equation 7](#equation-07) performs on par to the full transformer, while significantly reducing the computational and memory requirements.

<span id="section-3-3"></span>

### 3.3 Causal Masking

The transformer architecture can be used to efficiently train autoregressive models by masking the attention computation such that the $i$-th position can only be influenced by a position $j$ if and only if $j\leq i$, namely a position cannot be influenced by the subsequent positions. Formally, this causal masking changes [Equation 3](#equation-03) as follows,

<span id="equation-08"></span>

$$
V^{\prime}_{i}=\frac{\sum_{j=1}^{i}\mathrm{sim}\left(Q_{i},K_{j}\right)V_{j}}{\sum_{j=1}^{i}\mathrm{sim}\left(Q_{i},K_{j}\right)}.
$$

Following the reasoning of [Section 3.2](#section-3-2), we linearize the masked attention as described below,

<span id="equation-09"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)}.
$$

By introducing $S_{i}$ and $Z_{i}$ as follows,

<span id="equation-10"></span>

$$
S_{i}=\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top,
$$

<span id="equation-11"></span>

$$
Z_{i}=\sum_{j=1}^{i}\phi\left(K_{j}\right),
$$

we can simplify [Equation 9](#equation-09) to

<span id="equation-12"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top S_{i}}{\phi\left(Q_{i}\right)^\top Z_{i}}.
$$

Note that, $S_{i}$ and $Z_{i}$ can be computed from $S_{i-1}$ and $Z_{i-1}$ in constant time hence making the computational complexity of linear transformers with causal masking linear with respect to the sequence length.

<span id="section-3-3-1"></span>

#### 3.3.1 Gradient Computation

A naive implementation of [Equation 12](#equation-12), in any deep learning framework, requires storing all intermediate values $S_{i}$ in order to compute the gradients. This increases the memory consumption by $\max\left(D,M\right)$ times; thus hindering the applicability of causal linear attention to longer sequences or deeper models. To address this, we derive the gradients of the numerator in [Equation 9](#equation-09) as cumulative sums. This allows us to compute both the forward and backward pass of causal linear attention in **linear time** and **constant memory**. A detailed derivation is provided in the supplementary material.

Given the numerator $\bar{V}_{i}$ and the gradient of a scalar loss function with respect to the numerator $\nabla_{\bar{V}_{i}}\mathcal{L}$, we derive $\nabla_{\phi\left(Q_{i}\right)}\mathcal{L}$, $\nabla_{\phi\left(K_{i}\right)}\mathcal{L}$ and $\nabla_{V_{i}}\mathcal{L}$ as follows,

<span id="equation-13"></span>

$$
\nabla_{\phi\left(Q_{i}\right)}\mathcal{L}=\nabla_{\bar{V}_{i}}\mathcal{L}\left(\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top\right)^\top,
$$

<span id="equation-14"></span>

$$
\nabla_{\phi\left(K_{i}\right)}\mathcal{L}=\left(\sum_{j=i}^{N}\phi\left(Q_{j}\right)\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)V_{i},
$$

<span id="equation-15"></span>

$$
\nabla_{V_{i}}\mathcal{L}=\left(\sum_{j=i}^{N}\phi\left(Q_{j}\right)\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)^\top\phi\left(K_{i}\right).
$$

The cumulative sum terms in [Equation 9](#equation-09), [Equations 13](#equation-13)-[15](#equation-15) are computed in linear time and require constant memory with respect to the sequence length. This results in an algorithm with computational complexity $\mathcal{O}\left(N C M\right)$ and memory $\mathcal{O}\left(N\max\left(C,M\right)\right)$ for a given feature map of $C$ dimensions. A pseudocode implementation of the forward and backward pass of the numerator is given in [Algorithm 1](#algorithm-01).

<span id="section-3-3-2"></span>

#### 3.3.2 Training and Inference

When training an autoregressive transformer model the full ground truth sequence is available. This makes layerwise parallelism possible both for $f_{l}(\cdot)$ of [Equation 1](#equation-01) and the attention computation. As a result, transformers are more efficient to train than recurrent neural networks. On the other hand, during inference the output for timestep $i$ is the input for timestep $i+1$. This makes autoregressive models impossible to parallelize. Moreover, the cost per timestep for transformers is not constant; instead, it scales with the square of the current sequence length because attention must be computed for all previous timesteps.

Our proposed *linear transformer* model *combines the best of both worlds*. When it comes to training, the computations can be parallelized and take full advantage of GPUs or other accelerators. When it comes to inference, the cost per time and memory for one prediction is constant for our model. This means we can simply store the $\phi\left(K_{j}\right)V_{j}^\top$ matrix as an internal state and update it at every time step like a recurrent neural network. This results in inference **thousands of times faster** than other transformer models.

<span id="section-3-4"></span>

### 3.4 Transformers are RNNs

In literature, transformer models are considered to be a fundamentally different approach to recurrent neural networks. However, from the causal masking formulation in [Section 3.3](#section-3-3) and the discussion in the previous section, it becomes evident that any transformer layer with causal masking can be written as a model that, given an input, modifies an internal state and then predicts an output, namely a Recurrent Neural Network (RNN). Note that, in contrast to Universal Transformers [Deh18], we consider the recurrence with respect to time and not depth.

In the following equations, we formalize the transformer layer of [Equation 1](#equation-01) as a recurrent neural network. The resulting RNN has two hidden states, namely the attention memory $s$ and the normalizer memory $z$. We use subscripts to denote the timestep in the recurrence.

<span id="equation-16"></span>

$$
s_{0}=0,
$$

<span id="equation-17"></span>

$$
z_{0}=0,
$$

<span id="equation-18"></span>

$$
s_{i}=s_{i-1}+\phi\left(x_{i}W_{K}\right)\left(x_{i}W_{V}\right)^\top,
$$

<span id="equation-19"></span>

$$
z_{i}=z_{i-1}+\phi\left(x_{i}W_{K}\right),
$$

<span id="equation-20"></span>

$$
y_{i}=f_{l}\left(\frac{\phi\left(x_{i}W_{Q}\right)^\top s_{i}}{\phi\left(x_{i}W_{Q}\right)^\top z_{i}}+x_{i}\right).
$$

In the above equations, $x_{i}$ denotes the $i$-th input and $y_{i}$ the $i$-th output for a specific transformer layer. Note that our formulation does not impose any constraint on the feature function and it can be used for representing *any transformer* model, in theory even the ones using softmax attention. This formulation is a first step towards better understanding the relationship between transformers and popular recurrent networks [Hoc97] and the processes used for storing and retrieving information.

<span id="section-4"></span>

## 4 Experiments

<span id="algorithm-01"></span>

**Algorithm 1: Linear transformers with causal masking.**

<div class="paper-algorithm">

- **Function** $\mathrm{forward}(\phi(Q),\phi(K),V)$:
  - Set $V'\gets0$ and $S\gets0$.
  - **For** $i=1,\dots,N$:
    - Set $S\gets S+\phi(K_i)V_i^\top$ ([Equation 10](#equation-10)).
    - Set $\bar V_i\gets\phi(Q_i)S$.
  - **Return** $\bar V$.
- **Function** $\mathrm{backward}(\phi(Q),\phi(K),V,G)$:
  - $G$ is the gradient of the loss with respect to the output of $\mathrm{forward}$.
  - Set $S\gets0$ and $\nabla_{\phi(Q)}\mathcal L\gets0$.
  - **For** $i=1,\dots,N$:
    - Set $S\gets S+\phi(K_i)V_i^\top$.
    - Set $\nabla_{\phi(Q_i)}\mathcal L\gets G_iS^\top$ ([Equation 13](#equation-13)).
  - Set $S\gets0$, $\nabla_{\phi(K)}\mathcal L\gets0$ and $\nabla_V\mathcal L\gets0$.
  - **For** $i=N,\dots,1$:
    - Set $S\gets S+\phi(Q_i)G_i^\top$.
    - Set $\nabla_{V_i}\mathcal L\gets S^\top\phi(K_i)$ ([Equation 15](#equation-15)).
    - Set $\nabla_{\phi(K_i)}\mathcal L\gets S V_i$ ([Equation 14](#equation-14)).
  - **Return** $\nabla_{\phi(Q)}\mathcal L$, $\nabla_{\phi(K)}\mathcal L$ and $\nabla_V\mathcal L$.

</div>

In this section, we analyze experimentally the performance of the proposed *linear transformer*. Initially, in [Section 4.1](#section-4-1), we evaluate the linearized attention in terms of computational cost, memory consumption and convergence on synthetic data. To further showcase the effectiveness of *linear transformers*, we evaluate our model on two real-world applications, image generation in [Section 4.2](#section-4-2) and automatic speech recognition in [Section 4.3](#section-4-3). We show that our model achieves competitive performance with respect to the state-of-the-art transformer architectures, while requiring significantly less GPU memory and computation.

Throughout our experiments, we compare our model with two baselines, the full transformer with softmax attention and the Reformer [Kit20], the latter being a state-of-the-art accelerated transformer architecture. For the Reformer, we use a PyTorch reimplementation of the published code and for the full transformer we use the default PyTorch implementation. Note that for Reformer, we do not use the reversible layers, however, this does not affect the results as we only measure the memory consumption with respect to the self attention layer. In all experiments, we use **softmax** [Vas17] to refer to the standard transformer architecture, **linear** for our proposed *linear transformers* and **lsh-X** for Reformer [Kit20], where *X* denotes the hashing rounds.

For training the *linear transformers*, we use the feature map of [Equation 7](#equation-07). Our PyTorch [Pas19] code with documentation and examples can be found at [https://linear-transformers.com/](https://linear-transformers.com/). The constant memory gradient computation of [Equations 13](#equation-13)-[15](#equation-15) is implemented in approximately 200 lines of CUDA code.

<span id="figure-01"></span>

![Figure 1. Comparison of the computational requirements for a forward/backward pass for Reformer (lsh-X), softmax attention and linear attention. Linear and Reformer models scale linearly with the sequence length unlike softmax which scales with the square of the sequence length both in memory and time. Full details of the experiment can be found in [Section 4.1](#section-4-1).](../../papers/transformers-are-rnns/figure-01.png)

**Figure 1.** Comparison of the computational requirements for a forward/backward pass for Reformer (lsh-X), softmax attention and linear attention. Linear and Reformer models scale linearly with the sequence length unlike softmax which scales with the square of the sequence length both in memory and time. Full details of the experiment can be found in [Section 4.1](#section-4-1).

<span id="figure-02"></span>

![Figure 2. Convergence comparison of *softmax*, *linear* and *reformer* attention on a sequence duplication task. *linear* converges stably and reaches the same final performance as softmax. The details of the experiment are in [Section 4.1](#section-4-1).](../../papers/transformers-are-rnns/figure-02.png)

**Figure 2.** Convergence comparison of *softmax*, *linear* and *reformer* attention on a sequence duplication task. *linear* converges stably and reaches the same final performance as softmax. The details of the experiment are in [Section 4.1](#section-4-1).

<span id="section-4-1"></span>

### 4.1 Synthetic Tasks

<span id="section-4-1-1"></span>

#### 4.1.1 Convergence Analysis

To examine the convergence properties of *linear transformers* we train on an artifical copy task with causal masking. Namely, the transformers have to copy a series of symbols similar to the sequence duplication task of [Kit20]. We use a sequence of maximum length 128 with 10 different symbols separated by a dedicated separator symbol. For all three methods, we train a 4 layer transformer with 8 attention heads using a batch size of 64 and the RAdam optimizer [Liu19d] with a learning rate of $10^{-3}$ which is reduced to $10^{-4}$ after 3000 updates. [Figure 2](#figure-02) depicts the loss with respect to the number of gradient steps. We observe that linear converges smoothly and reaches a lower loss than lsh due to the lack of noise introduced by hashing. In particular, it reaches the same loss as softmax.

<span id="section-4-1-2"></span>

#### 4.1.2 Memory and Computational Requirements

In this subsection, we compare transformers with respect to their computational and memory requirements. We compute the attention and the gradients for a synthetic input with varying sequence lengths $N\in\{2^{9},2^{10},\dots,2^{16}\}$ and measure the peak allocated GPU memory and required time for each variation of transformer. We scale the batch size inversely with the sequence length and report the time and memory per sample in the batch.

Every method is evaluated up to the maximum sequence length that fits the GPU memory. For this benchmark we use an NVidia GTX 1080 Ti with 11GB of memory. This results in a maximum sequence length of 4,096 elements for softmax and 16,384 for lsh-4 and lsh-8. As expected, softmax scales quadratically with respect to the sequence length. Our method is faster and requires less memory than the baselines for every configuration, as seen in [Figure 1](#figure-01). We observe that both Reformer and linear attention scale linearly with the sequence length. Note that although the asymptotic complexity for Reformer is $\mathcal{O}\left(N\log N\right)$, $\log N$ is small enough and does not affect the computation time.

<span id="section-4-2"></span>

### 4.2 Image Generation

Transformers have shown great results on the task of conditional or unconditional autoregressive generation [Rad19, Chi19], however, sampling from transformers is slow due to the task being inherently sequential and the memory scaling with the square of the sequence length. In this section, we train causally masked transformers to predict images pixel by pixel. Our achieved performance in terms of bits per dimension is on par with *softmax* attention while being able to generate images **more than 1,000 times faster** and with **constant memory per image** from the first to the last pixel. We refer the reader to our supplementary for comparisons in terms of training evolution, quality of generated images and time to generate a single image. In addition, we also compare with a faster softmax transformer that caches the keys and values during inference, in contrast to the PyTorch implementation.

<span id="section-4-2-1"></span>

#### 4.2.1 MNIST

<span id="table-01"></span>

![Table 1. Comparison of autoregressive image generation of MNIST images. Our linear transformers achieve almost the same bits/dim as the full softmax attention but more than 300 times higher throughput in image generation. The full details of the experiment are in [Section 4.2.1](#section-4-2-1).](../../papers/transformers-are-rnns/table-01.png)

**Table 1.** Comparison of autoregressive image generation of MNIST images. Our linear transformers achieve almost the same bits/dim as the full softmax attention but more than 300 times higher throughput in image generation. The full details of the experiment are in [Section 4.2.1](#section-4-2-1).

First, we evaluate our model on image generation with autoregressive transformers on the widely used MNIST dataset [Lec00]. The architecture for this experiment comprises 8 attention layers with 8 attention heads each. We set the embedding size to 256 which is 32 dimensions per head. Our feed forward dimensions are 4 times larger than our embedding size. We model the output with a mixture of 10 logistics as introduced by [Sal17]. We use the RAdam optimizer with a learning rate of $10^{-4}$ and train all models for 250 epochs. For the reformer baseline, we use 1 and 4 hashing rounds. Furthermore, as suggested in [Kit20], we use 64 buckets and chunks with approximately 32 elements. In particular, we divide the 783 long input sequence to 27 chunks of 29 elements each. Since the sequence length is realtively small, namely only 784 pixels, to remove differences due to different batch sizes we use a batch size of 10 for all methods.

[Table 1](#table-01) summarizes the results. We observe that linear transformers achieve almost the same performance, in terms of final perplexity, as softmax transformers while being able to generate images more than 300 times faster. This is achieved due to the low memory requirements of our model, which is able to simultaneously generate 10,000 MNIST images with a single GPU. In particular, the memory is constant with respect to the sequence length because the only thing that needs to be stored between pixels are the $s_{i}$ and $z_{i}$ values as described in [Equations 18](#equation-18) and [19](#equation-19). On the other hand, both softmax and Reformer require memory that increases with the length of the sequence.

Image completions and unconditional samples from our MNIST model can be seen in [Figure 3](#figure-03). We observe that our linear transformer generates very convincing samples with sharp boundaries and no noise. In the case of image completion, we also observe that the transformer learns to use the same stroke style and width as the original image effectively attending over long temporal distances. Note that as the achieved perplexity is more or less the same for all models, we do not observe qualitative differences between the generated samples from different models.

<span id="section-4-2-2"></span>

#### 4.2.2 CIFAR-10

<span id="table-02"></span>

![Table 2. We train autoregressive transformers for 1 week on a single GPU to generate CIFAR-10 images. Our linear transformer completes 3 times more epochs than softmax, which results in better perplexity. Our model generates images $4{,}000\times$ faster than the baselines. The full details of the experiment are in [Section 4.2.2](#section-4-2-2).](../../papers/transformers-are-rnns/table-02.png)

**Table 2.** We train autoregressive transformers for 1 week on a single GPU to generate CIFAR-10 images. Our linear transformer completes 3 times more epochs than softmax, which results in better perplexity. Our model generates images $4{,}000\times$ faster than the baselines. The full details of the experiment are in [Section 4.2.2](#section-4-2-2).

The benefits of our linear formulation increase as the sequence length increases. To showcase that, we train 16 layer transformers to generate CIFAR-10 images [Kri09]. For each layer we use the same configuration as in the previous experiment. For Reformer, we use again 64 buckets and 83 chunks of 37 elements, which is approximately 32, as suggested in the paper. Since the sequence length is almost 4 times larger than for the previous experiment, the full transformer can only be used with a batch size of 1 in the largest GPU that is available to us, namely an NVidia P40 with 24GB of memory. For both the linear transformer and reformer, we use a batch size of 4. All models are trained for 7 days. We report results in terms of bits per dimension and image generation throughput in [Table 2](#table-02). Note that although the main point of this experiment is not the final perplexity, it is evident that as the sequence length grows, the fast transformer models become increasingly more efficient per GPU hour, achieving better scores than their slower counterparts.

As the memory and time to generate a single pixel scales quadratically with the number of pixels for both Reformer and softmax attention, the increase in throughput for our linear transformer is even more pronounced. In particular, **for every image generated** by the softmax transformer, **our method can generate 4,460 images**. Image completions and unconditional samples from our model can be seen in [Figure 4](#figure-04). We observe that our model generates images with spatial consistency and can complete images convincigly without significantly hindering the recognition of the image category. For instance, in [Figure 4(b)](#figure-04), all images have successfully completed the dog’s nose (first row) or the windshield of the truck (last row).

<span id="figure-03"></span>

![Figure 3. Unconditional samples and image completions generated by our method for MNIST. (a) depicts the occluded orignal images, (b) the completions and (c) the original. Our model achieves comparable bits/dimension to softmax, while having more than **300 times** higher throughput, generating **142 images/second**. For details see [Section 4.2.1](#section-4-2-1).](../../papers/transformers-are-rnns/figure-03.png)

**Figure 3.** Unconditional samples and image completions generated by our method for MNIST. (a) depicts the occluded orignal images, (b) the completions and (c) the original. Our model achieves comparable bits/dimension to softmax, while having more than **300 times** higher throughput, generating **142 images/second**. For details see [Section 4.2.1](#section-4-2-1).

<span id="figure-04"></span>

![Figure 4. Unconditional samples and image completions generated by our method for CIFAR-10. (a) depicts the occluded orignal images, (b) the completions and (c) the original. As the sequence length grows linear transformers become more efficient compared to softmax attention. Our model achieves more than **4,000 times** higher throughput and generates **17.85 images/second**. For details see [Section 4.2.2](#section-4-2-2).](../../papers/transformers-are-rnns/figure-04.png)

**Figure 4.** Unconditional samples and image completions generated by our method for CIFAR-10. (a) depicts the occluded orignal images, (b) the completions and (c) the original. As the sequence length grows linear transformers become more efficient compared to softmax attention. Our model achieves more than **4,000 times** higher throughput and generates **17.85 images/second**. For details see [Section 4.2.2](#section-4-2-2).

<span id="section-4-3"></span>

### 4.3 Automatic Speech Recognition

<span id="table-03"></span>

![Table 3. Performance comparison in automatic speech recognition on the WSJ dataset. The results are given in the form of phoneme error rate (PER) and training time per epoch. Our model outperforms the LSTM and Reformer while being faster to train and evaluate. Details of the experiment can be found in [Section 4.3](#section-4-3).](../../papers/transformers-are-rnns/table-03.png)

**Table 3.** Performance comparison in automatic speech recognition on the WSJ dataset. The results are given in the form of phoneme error rate (PER) and training time per epoch. Our model outperforms the LSTM and Reformer while being faster to train and evaluate. Details of the experiment can be found in [Section 4.3](#section-4-3).

To show that our method can also be used for non-autoregressive tasks, we evaluate the performance of linear transformers in end-to-end automatic speech recognition using Connectionist Temporal Classification (CTC) loss [Gra06]. In this setup, we predict a distribution over phonemes for each input frame in a non autoregressive fashion. We use the 80 hour WSJ dataset [Pau92] with 40-dimensional mel-scale filterbanks without temporal differences as features. The dataset contains sequences with 800 frames on average and a maximum sequence length of 2,400 frames. For this task, we also compare with a bidirectional LSTM [Hoc97] with 3 layers of hidden size 320. We use the Adam optimizer [Kin15] with a learning rate of $10^{-3}$ which is reduced when the validation error stops decreasing. For the transformer models, we use 9 layers with 6 heads with the same embedding dimensions as for the image experiments. As an optimizer, we use RAdam with an initial learning rate of $10^{-4}$ that is divided by 2 when the validation error stops decreasing.

All models are evaluated in terms of phoneme error rate (PER) and training time per epoch. We observe that linear outperforms the recurrent network baseline and Reformer both in terms of performance and speed by a large margin, as seen in [Table 3](#table-03). Note that the softmax transformer, achieves lower phone error rate in comparison to all baselines, but is significantly slower. In particular, *linear transformer* is more than $3\times$ faster per epoch. We provide training evolution plots in the supplementary.

<span id="section-5"></span>

## 5 Conclusions

In this work, we presented *linear transformer*, a model that significantly reduces the memory and computational cost of the original transformers. In particular, by exploiting the associativity property of matrix products we are able to compute the self-attention in time and memory that scales linearly with respect to the sequence length. We show that our model can be used with causal masking and still retain its linear asymptotic complexities. Finally, we express the transformer model as a recurrent neural network, which allows us to perform inference on autoregressive tasks thousands of time faster.

This property opens a multitude of directions for future research regarding the storage and retrieval of information in both RNNs and transformers. Another line of research to be explored is related to the choice of feature map for linear attention. For instance, approximating the RBF kernel with random Fourier features could allow us to use models pretrained with softmax attention.

## Acknowledgements

Angelos Katharopoulos was supported by the Swiss National Science Foundation under grant numbers FNS-30209 ”ISUL” and FNS-30224 ”CORTI”. Apoorv Vyas was supported by the Swiss National Science Foundation under grant number FNS-30213 ”SHISSM”. Nikolaos Pappas was supported by the Swiss National Science Foundation under grant number P400P2_183911 ”UNISON”.

<span id="section-6"></span>

## 6 Gradient Derivation

In the first section of our supplementary material, we derive in detail the gradients for causally masked linear transformers and show that they can be computed in linear time and constant memory. In particular, we derive the gradients of a scalar loss with respect to the numerator of the following equation,

<span id="equation-21"></span>

$$
V^{\prime}_{i}=\frac{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)V_{j}^\top}{\phi\left(Q_{i}\right)^\top\sum_{j=1}^{i}\phi\left(K_{j}\right)}.
$$

The gradient with respect to the denominator and the fraction are efficiently handled by autograd. Without loss of generality, we can assume that $Q$ and $K$ already contain the vectors mapped by $\phi\left(\cdot\right)$, hence given the numerator

<span id="equation-22"></span>

$$
\bar{V}_{i}=Q_{i}^\top\sum_{j=1}^{i}K_{j}V_{j}^\top,
$$

and $\nabla_{\bar{V}}\mathcal{L}$ we seek to compute $\nabla_{Q}\mathcal{L}$, $\nabla_{K}\mathcal{L}$ and $\nabla_{V}\mathcal{L}$. Note that $Q\in\mathbb{R}^{N\times D}$, $K\in\mathbb{R}^{N\times D}$ and $V\in\mathbb{R}^{N\times M}$. To derive the gradients, we first express the above equation for a single element without using vector notation,

<span id="equation-23"></span>

$$
\bar{V}_{ie}=\sum_{d=1}^{D}Q_{id}\sum_{j=1}^{i}K_{jd}V_{je}=\sum_{d=1}^{D}\sum_{j=1}^{i}Q_{id}K_{jd}V_{je}.
$$

Subsequently we can start deriving the gradients for $Q$ by taking the partial derivative for any $Q_{lt}$, as follows

<span id="equation-24"></span>

$$
\frac{\partial\mathcal{L}}{\partial Q_{lt}}=\sum_{e=1}^{M}\frac{\partial\mathcal{L}}{\partial\bar{V}_{le}}\frac{\partial\bar{V}_{le}}{\partial Q_{lt}}=\sum_{e=1}^{M}\frac{\partial\mathcal{L}}{\partial\bar{V}_{le}}\left(\sum_{j=1}^{l}K_{jt}V_{je}\right).
$$

If we write the above equation as a matrix product of gradients it becomes,

<span id="equation-25"></span>

$$
\nabla_{Q_{i}}\mathcal{L}=\nabla_{\bar{V}_{i}}\mathcal{L}\left(\sum_{j=1}^{i}K_{j}V_{j}^\top\right)^\top,
$$

proving [Equation 13](#equation-13) from the main paper. In [Equation 24](#equation-24) we made use of the fact that $Q_{lt}$ only affects $\bar{V}_{l}$ hence we do not need to sum over $i$ to compute the gradients. However, for $K$ and $V$ this is not the case. In particular, $K_{j}$ affects all $\bar{V}_{i}$ where $i\geq j$. Consequently, we can write the partial derivative of the loss with respect to $K_{lt}$ as follows,

<span id="equation-26"></span>

$$
\begin{aligned}
\frac{\partial\mathcal{L}}{\partial K_{lt}} & =\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}\frac{\partial\bar{V}_{ie}}{\partial K_{lt}}=\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}\frac{\partial\left(\sum_{d=1}^{D}\sum_{j=1}^{i}Q_{id}K_{jd}V_{je}\right)}{\partial K_{lt}} \\
=\sum_{e=1}^{M}\sum_{i=l}^{N}\frac{\partial\mathcal{L}}{\partial\bar{V}_{ie}}Q_{it}V_{le}.
\end{aligned}
$$

As for $Q$ we can now write the gradient in vectorized form,

<span id="equation-27"></span>

$$
\nabla_{K_{i}}\mathcal{L}=\left(\sum_{j=i}^{N}Q_{j}\left(\nabla_{\bar{V}_{j}}\mathcal{L}\right)^\top\right)V_{i},
$$

proving [Equation 14](#equation-14) from the paper. Following the same reasoning, we can compute the partial derivative of the loss with respect to $V_{lt}$ and prove equation 15. Note that the cumulative sum matrices for the gradient with respect to $Q$ and $K$ have the same size, however one is computed in the forward direction (summing from 1 to $N$) similarly to the forward pass and the other is computed in the backwards direction (summing from $N$ to 1) similar to backpropagation through time done in RNNs.

<span id="section-7"></span>

## 7 Training Evolution

In [Figure 5](#figure-05) we present the training evolution of all transformer models in our experiments. For the MNIST experiment ([Figure 5(a)](#figure-05)) we train all methods for 250 epochs. The sequence length is small enough so that the training time does not vary significantly for all methods. We observe that our method converges on par with softmax attention outperforming significantly both reformer variants.

On the other hand, for CIFAR-10 ([Figure 5(b)](#figure-05)) we train all methods for a fixed amount of time, namely 7 days. We observe that *lsh-1* and *linear* complete significantly more epochs than softmax and lsh-4 and achieve better performance. This gap is expected to increase with a further increase in sequence length.

Finally, in our last experiment on automatic speech recognition ([Figure 5(c)](#figure-05)), softmax outperforms significantly both Reformer and linear in terms of convergence. Note that linear is $3\times$ faster per epoch which means it has completed approximately 4 times more epochs in comparison to softmax. Even though softmax attention is better in this task, we observe that *linear transformers* significantly outperform Reformer both in terms of convergence and final performance.

<span id="figure-05"></span>

![Figure 5. Training evolution of transformers for all our experiments. It can be observed that *linear transformers* converge consistently faster than Reformer and in the autoregressive experiments on par with softmax. For MNIST all methods are trained for 250 epochs while for CIFAR we train for 7 days. In the speech recognition experiments all methods are trained to convergence. The details of the experiments can be found in [Section 4.2.1](#section-4-2-1), [Section 4.2.2](#section-4-2-2) and [Section 4.3](#section-4-3) in the main paper.](../../papers/transformers-are-rnns/figure-05.png)

**Figure 5.** Training evolution of transformers for all our experiments. It can be observed that *linear transformers* converge consistently faster than Reformer and in the autoregressive experiments on par with softmax. For MNIST all methods are trained for 250 epochs while for CIFAR we train for 7 days. In the speech recognition experiments all methods are trained to convergence. The details of the experiments can be found in [Section 4.2.1](#section-4-2-1), [Section 4.2.2](#section-4-2-2) and [Section 4.3](#section-4-3) in the main paper.

<span id="section-8"></span>

## 8 Image Generation Throughput Discussion

<span id="section-8-1"></span>

### 8.1 Stateful softmax attention

In [Section 4.2](#section-4-2) of the main paper, we report the image generation throughput and we compare with **softmax** transformer and **lsh**. In this section we create another baseline, denoted as **stateful-softmax**, that implements a softmax autoregressive transformer as a recurrent model. Namely, all the keys and values are saved and then passed to the model again when predicting the next element of the sequence. The state of this recurrent model is the set of keys and values which has size proportional to the sequence length. This is qualitatively different to our proposed model that has a state with fixed dimensions and computing the $i$-th state given the previous one has fixed computational cost regardless of $i$.

<span id="table-04"></span>

![Table 4. Comparison of autoregressive image generation throughput of MNIST and CIFAR-10 images. The experiment can be found in [Section 4.2](#section-4-2) in the main paper. For stateful-softmax we save the keys and values and reuse them for predicting the next element. A detailed description of this extra baseline can be found in [Section 8.1](#section-8-1).](../../papers/transformers-are-rnns/table-04.png)

**Table 4.** Comparison of autoregressive image generation throughput of MNIST and CIFAR-10 images. The experiment can be found in [Section 4.2](#section-4-2) in the main paper. For stateful-softmax we save the keys and values and reuse them for predicting the next element. A detailed description of this extra baseline can be found in [Section 8.1](#section-8-1).

[Table 4](#table-04) summarizes the results. We observe that stateful-softmax is significantly faster than vanilla transformers. However, its complexity is still quadratic with respect to the sequence length and our forumlation is more than $50\times$ faster for CIFAR-10. Moreover, we would like to point out that implementing a similar stateful attention for Reformer is not a trivial task as the sorting and chunking operations need to be performed each time a new input is provided.

<span id="section-8-2"></span>

### 8.2 Equalizing the batch size

In the previous sections we evaluate the throughput of all transformer variants for the task of autoregressive image generation. However, another important factor to consider is latency, namely the total time required to produce a single image. To this end, we use a batch size of 1 and measure the time required by all methods to generate a single image. In addition to running the inference on the GPU, we also evaluate the time required on CPU. The results are reported in [Table 5](#table-05).

<span id="table-05"></span>

![Table 5. Comparison of the time required to generate a single image with autoregressive transformers on MNIST and CIFAR-10. We run all methods with a batch size of 1 both on CPU and GPU and report the total time in seconds. For all numbers in the table, lower is better.](../../papers/transformers-are-rnns/table-05.png)

**Table 5.** Comparison of the time required to generate a single image with autoregressive transformers on MNIST and CIFAR-10. We run all methods with a batch size of 1 both on CPU and GPU and report the total time in seconds. For all numbers in the table, lower is better.

We observe that all methods underutilize the GPU and achieve significantly smaller image generation throughput than the one shown in [Table 4](#table-04). The proposed linear transformer is faster than all the methods and in particular it is almost $6.6\times$ faster than softmax transformers for generating an image on CIFAR-10. Note that our linear autoregressive transformer is the only method that is faster on the CPU than on the GPU in every case. This is due to the fact that computing the attention as an RNN has such a low cost that the main computational bottleneck becomes the inevitable outer loop over the sequence.

<span id="section-9"></span>

## 9 Qualitative Results on Image Generation

In this section we provide qualitative results for our image generation experiments. Since the perplexity of all models is approximately the same, as expected, the qualitative differences are not significant. A rather interesting observation however is that the Reformer models provide significantly fewer variations in their unconditional samples. Moreover, we observe that image completion is a significantly easier task than unconditional generation as all models perform significantly better.

<span id="figure-06"></span>

![Figure 6. Unconditional samples from the transformer models trained with MNIST. See [Section 4.2.1](#section-4-2-1) in the main paper.](../../papers/transformers-are-rnns/figure-06.png)

**Figure 6.** Unconditional samples from the transformer models trained with MNIST. See [Section 4.2.1](#section-4-2-1) in the main paper.

<span id="figure-07"></span>

![Figure 7. MNIST digit completion from all trained models. See [Section 4.2.1](#section-4-2-1) in the main paper.](../../papers/transformers-are-rnns/figure-07.png)

**Figure 7.** MNIST digit completion from all trained models. See [Section 4.2.1](#section-4-2-1) in the main paper.

<span id="figure-08"></span>

![Figure 8. Unconditional samples from the transformer models trained with CIFAR-10. See [Section 4.2.2](#section-4-2-2) in the main paper.](../../papers/transformers-are-rnns/figure-08.png)

**Figure 8.** Unconditional samples from the transformer models trained with CIFAR-10. See [Section 4.2.2](#section-4-2-2) in the main paper.

<span id="figure-09"></span>

![Figure 9. CIFAR-10 image completions from all trained transformer models. See [Section 4.2.2](#section-4-2-2) in the main paper.](../../papers/transformers-are-rnns/figure-09.png)

**Figure 9.** CIFAR-10 image completions from all trained transformer models. See [Section 4.2.2](#section-4-2-2) in the main paper.

[+affiliations]: Affiliations: Idiap Research Institute, Switzerland; EPFL, Switzerland; University of Washington, Seattle, USA; and University of Geneva, Switzerland. Work done at Idiap. Correspondence to: Angelos Katharopoulos <firstname.lastname@idiap.ch>.
