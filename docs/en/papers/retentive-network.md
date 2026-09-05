---
title: 'Retentive Network for Large Language Models'
createTime: 2026/09/05 14:30:00
permalink: /en/papers/retentive-network/
pageClass: paper-reading
---

> [Yutao Sun](https://dblp.org/pid/01/9758) [+equal], [Li Dong](https://dblp.org/pid/85/5090-4) [+equal], [Shaohan Huang](https://www.microsoft.com/en-us/research/people/shaohanh/), [Shuming Ma](https://dblp.org/pid/190/7739), [Yuqing Xia](https://dblp.org/pid/211/8365), [Jilong Xue](https://dblp.org/pid/06/10336), [Jianyong Wang](https://dblp.org/pid/24/2006), and [Furu Wei](https://dblp.org/pid/72/5870) [+corresponding]. First submitted to arXiv on July 17, 2023; current version v4. [Retentive Network: A Successor to Transformer for Large Language Models](https://arxiv.org/abs/2307.08621). <a href="/paper/retentive-network.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2307.08621). [TeX source](https://export.arxiv.org/e-print/2307.08621v4). The original PDF remains authoritative for the exact print layout and bibliography.

[+equal]: Equal contribution.

[+corresponding]: Corresponding author.

## Abstract

In this work, we propose **Retentive Network** (RetNet) as a foundation architecture for large language models, simultaneously achieving training parallelism, low-cost inference, and good performance. We theoretically derive the connection between recurrence and attention. Then we propose the retention mechanism for sequence modeling, which supports three computation paradigms, i.e., parallel, recurrent, and chunkwise recurrent. Specifically, the parallel representation allows for training parallelism. The recurrent representation enables low-cost $O(1)$ inference, which improves decoding throughput, latency, and GPU memory without sacrificing performance. The chunkwise recurrent representation facilitates efficient long-sequence modeling with linear complexity, where each chunk is encoded parallelly while recurrently summarizing the chunks. Experimental results on language modeling show that RetNet achieves favorable scaling results, parallel training, low-cost deployment, and efficient inference. The intriguing properties make RetNet a strong successor to Transformer for large language models. Code will be available at [https://aka.ms/retnet](https://aka.ms/retnet).

<span id="figure-01"></span>

![Retentive network inference cost and scaling curve](../../papers/retentive-network/figure-01.png)

**Figure 1.** Retentive network (RetNet) achieves low-cost inference (i.e., GPU memory, throughput, and latency), training parallelism, and favorable scaling curves compared with Transformer. Results of inference cost are reported with 8k as input length. [Figure 6](#figure-06) shows more results on different sequence lengths.

> The only way to discover the limits of the possible is to go beyond them into the impossible.
>
> —Arthur C. Clarke

<span id="section-1"></span>

## 1 Introduction

<span id="figure-02"></span>

![The impossible triangle made possible by RetNet](../../papers/retentive-network/figure-02.png)

**Figure 2.** RetNet makes the "impossible triangle" possible, which achieves training parallelism, good performance, and low inference cost simultaneously.

Transformer [Vas17] has become the de facto architecture for large language models [Bro20], which was initially proposed to overcome the sequential training issue of recurrent models [Hoc97]. However, training parallelism of Transformers is at the cost of inefficient inference, because of the $O(N)$ complexity per step and memory-bound key-value cache [Sha19], which renders Transformers unfriendly to deployment. The growing sequence length increases GPU memory consumption as well as latency and reduces inference speed.

Numerous efforts have continued to develop the next-generation architecture, aiming at retaining training parallelism and competitive performance as Transformers while having efficient $O(1)$ inference. It is challenging to achieve the above goals simultaneously, i.e., the so-called "impossible triangle" as shown in [Figure 2](#figure-02).

There have been three main strands of research. First, linearized attention [Kat20] approximates standard attention scores $\exp(\bm{q}\cdot\bm{k})$ with kernels $\phi(\bm{q})\cdot\phi(\bm{k})$, so that autoregressive inference can be rewritten in a recurrent form. However, the modeling capability and performance are worse than Transformers, which hinders the method's popularity. The second strand returns to recurrent models for efficient inference while sacrificing training parallelism. As a remedy, element-wise operators [Pen23b] are used for acceleration, however, representation capacity and performance are harmed. The third line of research explores replacing attention with other mechanisms, such as S4 [Gu22], and its variants [Dao22g, Pol23a]. None of the previous work can break through the impossible triangle, resulting in no clear winner compared with Transformers.

In this work, we propose retentive networks (RetNet), achieving low-cost inference, efficient long-sequence modeling, Transformer-comparable performance, and parallel model training simultaneously. Specifically, we introduce a multi-scale retention mechanism to substitute multi-head attention, which has three computation paradigms, i.e., parallel, recurrent, and chunkwise recurrent representations. First, the parallel representation empowers training parallelism to utilize GPU devices fully. Second, the recurrent representation enables efficient $O(1)$ inference in terms of memory and computation. The deployment cost and latency can be significantly reduced. Moreover, the implementation is greatly simplified without key-value cache tricks. Third, the chunkwise recurrent representation can perform efficient long-sequence modeling. We parallelly encode each local block for computation speed while recurrently encoding the global blocks to save GPU memory.

We conduct extensive experiments to compare RetNet with Transformer and its variants. Experimental results on language modeling show that RetNet is consistently competitive in terms of both scaling curves and in-context learning. Moreover, the inference cost of RetNet is length-invariant. For a 7B model and 8k sequence length, RetNet decodes 8.4$\times$ faster and saves 70% of memory than Transformers with key-value caches. During training, RetNet also achieves 25-50% memory saving and 7$\times$ acceleration than standard Transformer and an advantage towards highly-optimized FlashAttention [Dao22]. Besides, RetNet's inference latency is insensitive to batch size, allowing enormous throughput. The intriguing properties make RetNet a strong successor to Transformer for large language models.

<span id="section-2"></span>

## 2 Retentive Networks

Retentive network (RetNet) is stacked with $L$ identical blocks, which follows a similar layout (i.e., residual connection, and pre-LayerNorm) as in Transformer [Vas17]. Each RetNet block contains two modules: a multi-scale retention (MSR) module, and a feed-forward network (FFN) module. We introduce the MSR module in the following sections. Given an input sequence $x=x_1\cdots x_{|x|}$, RetNet encodes the sequence in an autoregressive way. The input vectors $\{\bm{x}_i\}_{i=1}^{|x|}$ is first packed into $X^0=[\bm{x}_1,\cdots,\bm{x}_{|x|}]\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$, where $d_{\mathrm{model}}$ is hidden dimension. Then we compute contextualized vector representations $X^l=\mathrm{RetNet}_l(X^{l-1}), l\in[1,L]$.

<span id="section-2-1"></span>

### 2.1 Retention

In this section, we introduce the retention mechanism that has a dual form of recurrence and parallelism. So we can train the models in a parallel way while recurrently conducting inference.

Given input $X\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$, we project it to one-dimensional function $v(n)=X_n\cdot\bm{w}_V$. Consider a sequence modeling problem that maps $v(n)\mapsto o(n)$ through states $\bm{s}_n$. Let $v_n,o_n$ denote $v(n),o(n)$ for simplicity. We formulate the mapping in a recurrent manner:

<span id="equation-01"></span>

$$
\begin{aligned}
\bm{s}_n &= A\bm{s}_{n-1}+K_n^\top v_n, &A\in\mathbb{R}^{d\times d}, K_n\in\mathbb{R}^{1\times d} \\
o_n &= Q_n\bm{s}_n=\sum_{m=1}^n Q_nA^{n-m}K_m^\top v_m, &Q_n\in\mathbb{R}^{1\times d}
\end{aligned}
$$

where we map $v_n$ to the state vector $\bm{s}_n$, and then implement a linear transform to encode sequence information recurrently.

Next, we make the projection $Q_n,K_n$ content-aware:

<span id="equation-02"></span>

$$
Q=X W_Q,\quad K=X W_K
$$

where $W_Q,W_K\in\mathbb{R}^{d\times d}$ are learnable matrices.

We diagonalize the matrix $A=\Lambda(\gamma e^{i\theta})\Lambda^{-1}$, where $\gamma,\theta\in\mathbb{R}^d$. Then we obtain $A^{n-m}=\Lambda(\gamma e^{i\theta})^{n-m}\Lambda^{-1}$. By absorbing $\Lambda$ into $W_Q$ and $W_K$, we can rewrite [Equation (1)](#equation-01) as:

<span id="equation-03"></span>

$$
\begin{aligned}
o_n &= \sum_{m=1}^n Q_n(\gamma e^{i\theta})^{n-m}K_m^\top v_m \\
&=\sum_{m=1}^n(Q_n(\gamma e^{i\theta})^n)(K_m(\gamma e^{i\theta})^{-m})^\top v_m
\end{aligned}
$$

where $Q_n(\gamma e^{i\theta})^n,K_m(\gamma e^{i\theta})^{-m}$ is known as xPos [Sun22], i.e., a relative position embedding proposed for Transformer. We further simplify $\gamma$ as a scalar, [Equation (3)](#equation-03) becomes:

<span id="equation-04"></span>

$$
o_n=\sum_{m=1}^n\gamma^{n-m}(Q_n e^{i n\theta})(K_m e^{i m\theta})^\dagger v_m
$$

where $^\dagger$ is the conjugate transpose. The formulation is easily parallelizable within training instances.

In summary, we start with recurrent modeling as shown in [Equation (1)](#equation-01), and then derive its parallel formulation in [Equation (4)](#equation-04). We consider the original mapping $v(n)\mapsto o(n)$ as vectors and obtain the retention mechanism as follows.

<span id="figure-03"></span>

![Parallel and recurrent representations of RetNet](../../papers/retentive-network/figure-03.png)

**Figure 3.** Dual form of RetNet. "GN" is short for GroupNorm.

**The Parallel Representation of Retention.** As shown in [Figure 3a](#figure-03), the retention layer is defined as:

<span id="equation-05"></span>

$$
\begin{aligned}
Q=(X W_Q)\odot\Theta,&\quad K=(X W_K)\odot\overline{\Theta},\quad V=X W_V \\
\Theta_n=e^{i n\theta},&\quad
D_{nm}=\begin{cases}
\gamma^{n-m}, & n\ge m \\
0, & n<m
\end{cases} \\
\mathrm{Retention}(X)&=(Q K^\top\odot D)V
\end{aligned}
$$

where $\overline{\Theta}$ is the complex conjugate of $\Theta$, and $D\in\mathbb{R}^{|x|\times|x|}$ combines causal masking and exponential decay along relative distance as one matrix. Similar to self-attention, the parallel representation enables us to train the models with GPUs efficiently.

**The Recurrent Representation of Retention.** As shown in [Figure 3b](#figure-03), the proposed mechanism can also be written as recurrent neural networks (RNNs), which is favorable for inference. For the $n$-th timestep, we recurrently obtain the output as:

<span id="equation-06"></span>

$$
\begin{aligned}
S_n &= \gamma S_{n-1}+K_n^\top V_n \\
\mathrm{Retention}(X_n)&=Q_nS_n,\quad n=1,\cdots,|x|
\end{aligned}
$$

where $Q,K,V,\gamma$ are the same as in [Equation (5)](#equation-05).

**The Chunkwise Recurrent Representation of Retention.** A hybrid form of parallel representation and recurrent representation is available to accelerate training, especially for long sequences. We divide the input sequences into chunks. Within each chunk, we follow the parallel representation ([Equation (5)](#equation-05)) to conduct computation. In contrast, cross-chunk information is passed following the recurrent representation ([Equation (6)](#equation-06)). Specifically, let $B$ denote the chunk length. We compute the retention output of the $i$-th chunk via:

<span id="equation-07"></span>

$$
\begin{aligned}
Q_{[i]}=Q_{Bi:B(i+1)}&,\quad K_{[i]}=K_{Bi:B(i+1)},\quad V_{[i]}=V_{Bi:B(i+1)} \\
R_i&=K_{[i]}^\top(V_{[i]}\odot\zeta)+\gamma^B R_{i-1},\quad\zeta_{ij}=\gamma^{B-i-1} \\
\mathrm{Retention}(X_{[i]})&=\underbrace{(Q_{[i]} K_{[i]}^\top\odot D)V_{[i]}}_{\mathrm{Inner}{-}\mathrm{Chunk}}+\underbrace{(Q_{[i]}R_{i-1})\odot\xi}_{\mathrm{Cross}{-}\mathrm{Chunk}},\quad\xi_{ij}=\gamma^{i+1}
\end{aligned}
$$

where ${[i]}$ indicates the $i$-th chunk, i.e., $x_{[i]}=[x_{(i-1)B+1},\cdots,x_{iB}]$.

<span id="section-2-2"></span>

### 2.2 Gated Multi-Scale Retention

We use $h=\frac{d_{\mathrm{model}}}{d}$ retention heads in each layer, where $d$ is the head dimension. The heads use different parameter matrices $W_Q,W_K,W_V\in\mathbb{R}^{d\times d}$. Moreover, **m**ulti-**s**cale **r**etention (MSR) assigns different $\gamma$ for each head. For simplicity, we set $\gamma$ identical among different layers and keep them fixed. In addition, we add a $\mathrm{swish}$ gate [Hen16, Ram17] to increase the non-linearity of retention layers. Formally, given input $X$, we define the layer as:

<span id="equation-08"></span>

$$
\begin{aligned}
\bm{\gamma}&=1-2^{-5-\mathrm{arange}(0,h)}\in\mathbb{R}^h \\
\mathrm{head}_i&=\mathrm{Retention}(X,\gamma_i) \\
Y&=\mathrm{GroupNorm}_h(\mathrm{Concat}(\mathrm{head}_1,\cdots,\mathrm{head}_h)) \\
\mathrm{MSR}(X)&=(\mathrm{swish}(X W_G)\odot Y)W_O
\end{aligned}
$$

where $W_G,W_O\in\mathbb{R}^{d_{\mathrm{model}}\times d_{\mathrm{model}}}$ are learnable parameters, and $\mathrm{GroupNorm}$ [Wu18c] normalizes the output of each head, following SubLN proposed in [Sho19]. Notice that the heads use multiple $\gamma$ scales, which results in different variance statistics. So we normalize the head outputs separately.

<span id="figure-04"></span>

![Pseudocode for the three retention computation paradigms](../../papers/retentive-network/figure-04.png)

**Figure 4.** Pseudocode for the three computation paradigms of retention.

The pseudocode of retention is summarized in [Figure 4](#figure-04).

**Retention Score Normalization.** We utilize the scale-invariant nature of $\mathrm{GroupNorm}$ to improve the numerical precision of retention layers. Specifically, multiplying a scalar value within $\mathrm{GroupNorm}$ does not affect outputs and backward gradients, i.e., $\mathrm{GroupNorm}(\alpha*\mathrm{head}_i)=\mathrm{GroupNorm}(\mathrm{head}_i)$. We implement three normalization factors in [Equation (5)](#equation-05). First, we normalize $Q K^\top$ as $\frac{Q K^\top}{\sqrt{d}}$. Second, we replace $D$ with $\tilde{D}_{nm}=\frac{D_{nm}}{\sqrt{\sum_{i=1}^nD_{ni}}}$. Third, let $R$ denote the retention scores $R=Q K^\top\odot D$, we normalize it as $\tilde{R}_{nm}=\frac{R_{nm}}{\max(|\sum_{i=1}^nR_{ni}|,1)}$. Then the retention output becomes $\mathrm{Retention}(X)=\tilde{R}V$. The above tricks do not affect the final results while stabilizing the numerical flow of both forward and backward passes, because of the scale-invariant property.

<span id="section-2-3"></span>

### 2.3 Overall Architecture of Retention Networks

For an $L$-layer retention network, we stack multi-scale retention (MSR) and feed-forward network (FFN) to build the model. Formally, the input sequence $\{x_i\}_{i=1}^{|x|}$ is transformed to vectors by a word embedding layer. We use the packed embeddings $X^0=[\bm{x}_1,\cdots,\bm{x}_{|x|}]\in\mathbb{R}^{|x|\times d_{\mathrm{model}}}$ as the input and compute the model output $X^L$:

<span id="equation-09"></span>

$$
\begin{aligned}
Y^l&=\mathrm{MSR}(\mathrm{LN}(X^l))+X^l \\
X^{l+1}&=\mathrm{FFN}(\mathrm{LN}(Y^l))+Y^l
\end{aligned}
$$

where $\mathrm{LN}(\cdot)$ is LayerNorm [Ba16]. The FFN part is computed as $\mathrm{FFN}(X)=\mathrm{gelu}(X W_1)W_2$, where $W_1,W_2$ are parameter matrices.

**Training.** We use the parallel ([Equation (5)](#equation-05)) and chunkwise recurrent ([Equation (7)](#equation-07)) representations during the training process. The parallelization within sequences or chunks efficiently utilizes GPUs to accelerate computation. More favorably, chunkwise recurrence is especially useful for long-sequence training, which is efficient in terms of both FLOPs and memory consumption.

**Inference.** The recurrent representation ([Equation (6)](#equation-06)) is employed during the inference, which nicely fits autoregressive decoding. The $O(1)$ complexity reduces memory and inference latency while achieving equivalent results.

<span id="section-2-4"></span>

### 2.4 Relation to and Differences from Previous Methods

[Table 1](#table-01) compares RetNet with previous methods from various perspectives. The comparison results echo the "impossible triangle" presented in [Figure 2](#figure-02). Moreover, RetNet has linear memory complexity for long sequences due to the chunkwise recurrent representation. We also summarize the comparisons with specific methods as follows.

**Transformer.** The parallel representation of retention shares similar spirits as Transformers [Vas17]. The most related Transformer variant is Lex Transformer [Sun22] which implements xPos as position embeddings. As described in [Equation (3)](#equation-03), the derivation of retention aligns with xPos. In comparison with attention, retention removes $\mathrm{softmax}$ and enables recurrent formulation, which significantly benefits inference.

**S4.** Unlike [Equation (2)](#equation-02), if $Q_n$ and $K_n$ are content-unaware, the formulation can be degenerated to S4 [Gu22], where $O=(Q K^\top,Q A K^\top,\ldots,Q A^{|x|-1}K^\top)*V$.

**Linear Attention.** The variants typically use various kernels $\frac{\phi(q_i)\phi(k_j)}{\sum_{n=1}^{|x|}\phi(q_i)\phi(k_n)}$ to replace the $\mathrm{softmax}$ function. However, linear attention struggles to effectively encode position information, rendering the models less performant. Besides, we reexamine sequence modeling from scratch, rather than aiming at approximating $\mathrm{softmax}$.

**AFT/RWKV.** Attention Free Transformer (AFT) simplifies dot-product attention to element-wise operations and moves $\mathrm{softmax}$ to key vectors. RWKV replaces AFT's position embeddings with exponential decay and runs the models recurrently for training and inference. In comparison, retention preserves high-dimensional states to encode sequence information, which contributes to expressive ability and better performance.

**xPos/RoPE.** Compared with relative position embedding methods proposed for Transformers, [Equation (3)](#equation-03) presents a similar formulation as xPos [Sun22] and RoPE [Su24].

**Sub-LayerNorm.** As shown in [Equation (8)](#equation-08), the retention layer uses Sub-LayerNorm [Wan22l] to normalize outputs. Because the multi-scale modeling leads to different variances for the heads, we replace the original LayerNorm with GroupNorm.

<span id="table-01"></span>

![Model comparison from various perspectives](../../papers/retentive-network/table-01.png)

**Table 1.** Model comparison from various perspectives. RetNet achieves training parallelization, constant inference cost, linear long-sequence memory complexity, and good performance.

<span id="section-3"></span>

## 3 Experiments

We conduct experiments on language modeling to evaluate RetNet. We evaluate the proposed architecture with various benchmarks, i.e., language modeling performance, and zero-/few-shot learning on downstream tasks. Moreover, for training and inference, we compare speed, memory consumption, and latency.

<span id="section-3-1"></span>

### 3.1 Setup

**Parameter Allocation.** We re-allocate the parameters in MSR and FFN for fair comparisons. Let $d$ denote $d_{\mathrm{model}}$ for simplicity here. In Transformers, there are about $4d^2$ parameters in self-attention where $W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}$, and $8d^2$ parameters in FFN where the intermediate dimension is $4d$. In comparison, RetNet has $8d^2$ parameters in retention, where $W_Q,W_K\in\mathbb{R}^{d\times d},W_G,W_V\in\mathbb{R}^{d\times2d},W_O\in\mathbb{R}^{2d\times d}$. Notice that the head dimension of $V$ is twice $Q,K$. The widened dimension is projected back to $d$ by $W_O$. In order to keep the parameter number the same as Transformer, the FFN intermediate dimension in RetNet is $2d$. Meanwhile, we set the head dimension to $256$ in our experiments, i.e., $256$ for queries and keys, and $512$ for values. For fair comparison, we keep $\bm{\gamma}$ identical among different model sizes, where $\bm{\gamma}=1-e^{\mathrm{linspace}(\log\frac{1}{32},\log\frac{1}{512},h)}\in\mathbb{R}^h$ instead of the default value in [Equation (8)](#equation-08).

<span id="table-02"></span>

![Model sizes and language-modeling hyperparameters](../../papers/retentive-network/table-02.png)

**Table 2.** Sizes, and learning hyper-parameters of the models in language modeling experiments.

**Language Model Training.** As shown in [Table 2](#table-02), we train language models with various sizes (i.e., 1.3B, 2.7B, and 6.7B) from scratch. The training corpus is a curated compilation of The Pile [Gao20], C4 [Dod21], and The Stack [Koc22]. We append the `<bos>` token to indicate the start of a sequence [+2]. The training batch size is 4M tokens with 2048 maximal length. We train the models with 100B tokens, i.e., 25k steps. We use the AdamW [Los17] optimizer with $\beta_1=0.9,\beta_2=0.98$, and weight decay is set to $0.05$. The number of warmup steps is 375 with linear learning rate decay. The parameters are initialized following DeepNet [Wan22c] to guarantee training stability. The implementation is based on TorchScale [Ma22]. We train the models with 512 AMD MI200 GPUs.

[+2]: We find that appending the `<bos>` token at the beginning benefits training stability and performance.

<span id="section-3-2"></span>

### 3.2 Comparisons with Transformer

<span id="figure-05"></span>

![Perplexity scaling curves of RetNet and Transformer](../../papers/retentive-network/figure-05.png)

**Figure 5.** Perplexity decreases along with scaling up the model size. We empirically observe that RetNet tends to outperform Transformer when the model size is larger than 2B.

**Language Modeling.** As shown in [Figure 5](#figure-05), we report perplexity on the validation set for the language models based on Transformer and RetNet. We present the scaling curves with three model sizes, i.e., 1.3B, 2.7B, and 6.7B. RetNet achieves comparable results with Transformers. More importantly, the results indicate that RetNet is favorable regarding size scaling. Besides performance, the RetNet training is quite stable in our experiments. Experimental results show that RetNet is a strong competitor to Transformer for large language models. Empirically, we find that RetNet starts to outperform Transformer when the model size is larger than 2B. We also summarize the language modeling results with different context lengths in [Section 6](#section-6).

<span id="table-03"></span>

![Zero-shot and few-shot performance of Transformer and RetNet](../../papers/retentive-network/table-03.png)

**Table 3.** Zero-shot and few-shot learning with Transformer and RetNet. The model size is 6.7B.

**Zero-Shot and Few-Shot Evaluation on Downstream Tasks.** We also compare the language models on a wide range of downstream tasks. We evaluate zero-shot and 4-shot learning with the 6.7B models. As shown in [Table 3](#table-03), the datasets include HellaSwag (HS) [Zel19], BoolQ [Cla19], COPA [Wan19h], PIQA [Bis20], Winograd, Winogrande [Lev12], and StoryCloze (SC) [Mos17]. The accuracy numbers are consistent with language modeling perplexity presented in [Figure 5](#figure-05). RetNet achieves comparable performance with Transformer on zero-shot and in-context learning settings.

<span id="section-3-3"></span>

### 3.3 Training Cost

<span id="table-04"></span>

![Training memory and throughput of Transformer and RetNet](../../papers/retentive-network/table-04.png)

**Table 4.** Training cost of Transformer (Trm), Transformer with FlashAttention (Trm+FlashAttn), and RetNet. We report memory consumption and training throughput (word per second; wps).

As shown in [Table 4](#table-04), we compare the training speed and memory consumption of Transformer and RetNet, where the training sequence length is 8192. We also compare with FlashAttention [Dao22], which improves speed and reduces GPU memory IO by recomputation and kernel fusion. In comparison, we implement RetNet using vanilla PyTorch code, and leave kernel fusion or FlashAttention-like acceleration for future work. We use chunkwise recurrent representation of retention as described in [Equation (7)](#equation-07). The chunk size is set to $512$. We evaluate the results with eight Nvidia A100-80GB GPUs, because FlashAttention is highly optimized for A100. Tensor parallelism is enabled for 6.7B and 13B models.

Experimental results show that RetNet is more memory-efficient and has higher throughput than Transformers during training. Even compared with FlashAttention, RetNet is still competitive in terms of speed and memory cost. Moreover, without relying on specific kernels, it is easy to train RetNet on other platforms efficiently. For example, we train the RetNet models on an AMD MI200 cluster with decent throughput. It is notable that RetNet has the potential to further reduce cost via advanced implementation, such as kernel fusion.

<span id="section-3-4"></span>

### 3.4 Inference Cost

<span id="figure-06"></span>

![Inference memory, throughput, and latency of Transformer and RetNet](../../papers/retentive-network/figure-06.png)

**Figure 6.** Inference cost of Transformer and RetNet with a model size of 6.7B. RetNet outperforms Transformers in terms of memory consumption, throughput, and latency.

As shown in [Figure 6](#figure-06), we compare memory cost, throughput, and latency of Transformer and RetNet during inference. Transformers reuse KV caches of previously decoded tokens. RetNet uses the recurrent representation as described in [Equation (6)](#equation-06). We evaluate the 6.7B model on the A100-80GB GPU in our experiments. [Figure 6](#figure-06) shows that RetNet outperforms Transformer in terms of inference cost.

**Memory.** As shown in [Figure 6a](#figure-06), the memory cost of Transformer increases linearly due to KV caches. In contrast, the memory consumption of RetNet remains consistent even for long sequences, requiring much less GPU memory to host RetNet. The additional memory consumption of RetNet is almost negligible (i.e., about 3%) while the model weights occupy 97%.

**Throughput.** As presented in [Figure 6b](#figure-06), the throughput of Transformer drops along with the decoding length increases. In comparison, RetNet has higher and length-invariant throughput during decoding, by utilizing the recurrent representation of retention.

**Latency.** Latency is an important metric in deployment, which greatly affects user experience. We report decoding latency in [Figure 6c](#figure-06). Experimental results show that increasing batch size renders Transformer's latency larger. Moreover, the latency of Transformers grows faster with longer input. In order to make latency acceptable, we have to restrict the batch size, which harms the overall inference throughput of Transformers. By contrast, RetNet's decoding latency outperforms Transformers and keeps almost the same across different batch sizes and input lengths.

<span id="section-3-5"></span>

### 3.5 Comparison with Transformer Variants

<span id="table-05"></span>

![Language-modeling perplexity of efficient Transformer variants](../../papers/retentive-network/table-05.png)

**Table 5.** Perplexity results on language modeling. RetNet outperforms other architectures on both the in-domain evaluation set and various out-of-domain corpora.

Apart from Transformer, we compare RetNet with various efficient Transformer variants, including Linear Transformer [Kat20], RWKV [Pen23b], H3 [Dao22g], and Hyena [Pol23a]. All models have 200M parameters with 16 layers and a hidden dimension of 1024. For H3, we set the head dimension as 8. For RWKV, we use the TimeMix module to substitute self-attention layers while keeping FFN layers consistent with other models for fair comparisons. We train the models with 10k steps with a batch size of 0.5M tokens. Most hyperparameters and training corpora are kept the same as in [Section 3.1](#section-3-1).

[Table 5](#table-05) reports the perplexity numbers on the in-domain validation set and other out-of-domain corpora, e.g., Project Gutenberg 2019-2022 (PG22) [Sun22], QMSum [Zho21b], GovReport [Hua21], SummScreen [Che21g, Sha22a]. Overall, RetNet outperforms previous methods across different datasets. RetNet not only achieves better evaluation results on the in-domain corpus but also obtains lower perplexity on several out-of-domain datasets. The favorable performance makes RetNet a strong successor to Transformer, besides the benefits of significant cost reduction ([Section 3.3](#section-3-3), [Section 3.4](#section-3-4)).

In addition, we discuss the training and inference efficiency of the compared methods. Let $d$ denote the hidden dimension, and $n$ the sequence length. For training, RWKV's token-mixing complexity is $O(dn)$ while Hyena's is $O(dn\log n)$ with Fast Fourier Transform acceleration. The above two methods reduce training FLOPS via employing element-wise operators to trade-off modeling capacity. In comparison with retention, the chunk-wise recurrent representation is $O(dn(b+h))$, where $b$ is the chunk size, $h$ is the head dimension, and we usually set $b=512,h=256$. For either large model size (i.e., larger $d$) or sequence length, the additional $b+h$ has negligible effects. So the RetNet training is quite efficient without sacrificing the modeling performance. For inference, among the compared efficient architectures, Hyena has the same complexity (i.e., $O(n)$ per step) as Transformer while the others can perform $O(1)$ decoding.

<span id="section-3-6"></span>

### 3.6 Ablation Studies

<span id="table-06"></span>

![Ablation results on in-domain and out-of-domain corpora](../../papers/retentive-network/table-06.png)

**Table 6.** Ablation results on in-domain and out-of-domain corpora.

We ablate various design choices of RetNet and report the language modeling results in [Table 6](#table-06). The evaluation settings and metrics are the same as in [Section 3.5](#section-3-5).

**Architecture.** We ablate the $\mathrm{swish}$ gate and $\mathrm{GroupNorm}$ as described in [Equation (8)](#equation-08). [Table 6](#table-06) shows that the above two components improve the final performance. Firstly, the gating module is essential for enhancing non-linearity and improving model capability. Notice that we use the same parameter allocation as Transformers after removing the gate. Secondly, group normalization in retention balances the variances of multi-head outputs, which improves training stability and language modeling results.

**Multi-Scale Decay.** [Equation (8)](#equation-08) shows that we use different $\bm{\gamma}$ as the decay rates for the retention heads. In the ablation studies, we examine removing $\gamma$ decay (i.e., "$-\ \gamma$ decay") and applying the same decay rate across heads (i.e., "$-$ multi-scale decay"). Specifically, ablating $\gamma$ decay is equivalent to $\gamma=1$. In the second setting, we set $\gamma=127/128$ for all heads. [Table 6](#table-06) indicates that both the decay mechanism and using multiple decay rates can improve the language modeling performance.

**Head Dimension.** From the recurrent perspective of [Equation (1)](#equation-01), the head dimension implies the memory capacity of hidden states. In the ablation study, we reduce the default head dimension from $256$ to $64$, i.e., $64$ for queries and keys, and $128$ for values. We keep the hidden dimension $d_{\mathrm{model}}$ the same so the number of heads increases. Experimental results in [Table 6](#table-06) show that the larger head dimension achieves better performance.

<span id="section-4"></span>

## 4 Conclusion

In this work, we propose retentive networks (RetNet) for sequence modeling, which enables various representations, i.e., parallel, recurrent, and chunkwise recurrent. RetNet achieves significantly better inference efficiency (in terms of memory, speed, and latency), favorable training parallelization, and competitive performance compared with Transformers. The above advantages make RetNet an ideal successor to Transformers for large language models, especially considering the deployment benefits brought by the $O(1)$ inference complexity. In the future, we would like to scale up RetNet in terms of model size [Chi22a] and training steps. Moreover, retention can efficiently work with structured prompting [Hao22a] by compressing long-term memory. We will also use RetNet as the backbone architecture to train multimodal large language models [Hao22, Hua23c, Pen23f]. In addition, we are interested in deploying RetNet models on various edge devices, such as mobile phones.

## Acknowledgement

We would like to acknowledge Jiayu Ding, Songlin Yang, and colleagues from MSRA System Group for the helpful discussions.

<span id="section-5"></span>

## 5 Hyperparameters

<span id="table-07"></span>

![Hyperparameters used for the RetNet models](../../papers/retentive-network/table-07.png)

**Table 7.** Hyperparamters used for the models in [Section 3](#section-3).

<span id="section-6"></span>

## 6 Grouped Results of Different Context Lengths

As shown in [Table 8](#table-08), we report language modeling results with different context lengths. In order to make the numbers comparable, we use 2048 text chunks as evaluation data and only compute perplexity for the last 128 tokens. Experimental results show that RetNet outperforms Transformer across different context lengths. Besides, RetNet can utilize longer context for better results.

<span id="table-08"></span>

![Language-modeling perplexity at different context lengths](../../papers/retentive-network/table-08.png)

**Table 8.** Language modeling perplexity of RetNet and Transformer with different context length. The results show that RetNet has a consistent advantage across sequence length.
