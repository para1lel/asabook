---
title: 'GLU Variants Improve Transformer'
createTime: 2026/09/05 13:11:11
permalink: /en/papers/glu-variants/
---

> [Noam Shazeer](https://www.noamshazeer.com/). First submitted to arXiv on February 12, 2020; current version v1. [GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202v1). <a href="/paper/glu-variants.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [arXiv DOI](https://doi.org/10.48550/arXiv.2002.05202). [TeX source](https://export.arxiv.org/e-print/2002.05202v1). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Gated Linear Units [Dau16] consist of the component-wise product of two linear projections, one of which is first passed through a sigmoid function. Variations on GLU are possible, using different nonlinear (or even linear) functions in place of sigmoid. We test these variants in the feed-forward sublayers of the Transformer [Vas17] sequence-to-sequence model, and find that some of them yield quality improvements over the typically-used ReLU or GELU activations.

<span id="section-1"></span>

## 1 Introduction

The Transformer [Vas17] sequence-to-sequence model alternates between multi-head attention, and what it calls "position-wise feed-forward networks" (FFN). The FFN takes a vector $x$ (the hidden representation at a particular position in the sequence) and passes it through two learned linear transformations, (represented by the matrices $W_1$ and $W_2$ and bias vectors $b_1$ and $b_2$). A rectified-linear (ReLU) [Glo11] activation function applied between the two linear transformations.

<span id="equation-01"></span>

$$
\mathrm{FFN}(x,W_1,W_2,b_1,b_2)=\max(0,xW_1+b_1)W_2+b_2
$$

Following the T5 codebase [Raf19] [+1], we use a version with no bias:

<span id="equation-02"></span>

$$
\mathrm{FFN}_{\mathrm{ReLU}}(x,W_1,W_2)=\max(xW_1,0)W_2
$$

Subsequent work has proposed replacing the ReLU with other nonlinear activation functions such as Gaussian Error Linear Units, $\mathrm{GELU}(x)=x\Phi(x)$ [Hen16], and $\mathrm{Swish}_\beta(x)=x\sigma(\beta x)$ [Ram17].

<span id="equation-03"></span>

$$
\begin{aligned}
\mathrm{FFN}_{\mathrm{GELU}}(x,W_1,W_2)&=\mathrm{GELU}(xW_1)W_2 \\
\mathrm{FFN}_{\mathrm{Swish}}(x,W_1,W_2)&=\mathrm{Swish}_1(xW_1)W_2
\end{aligned}
$$

<span id="section-2"></span>

## 2 Gated Linear Units (GLU) and Variants

[Dau16] introduced Gated Linear Units (GLU), a neural network layer defined as the component-wise product of two linear transformations of the input, one of which is sigmoid-activated. They also suggest omitting the activation, which they call a "bilinear" layer and attribute to [Mni07].

<span id="equation-04"></span>

$$
\begin{aligned}
\mathrm{GLU}(x,W,V,b,c)&=\sigma(xW+b)\otimes(xV+c) \\
\mathrm{Bilinear}(x,W,V,b,c)&=(xW+b)\otimes(xV+c)
\end{aligned}
$$

We can also define GLU variants using other activation functions:

<span id="equation-05"></span>

$$
\begin{aligned}
\mathrm{ReGLU}(x,W,V,b,c)&=\max(0,xW+b)\otimes(xV+c) \\
\mathrm{GEGLU}(x,W,V,b,c)&=\mathrm{GELU}(xW+b)\otimes(xV+c) \\
\mathrm{SwiGLU}(x,W,V,b,c,\beta)&=\mathrm{Swish}_\beta(xW+b)\otimes(xV+c)
\end{aligned}
$$

In this paper, we propose additional variations on the Transformer FFN layer which use GLU or one of its variants in place of the first linear transformation and the activation function. Again, we omit the bias terms.

<span id="equation-06"></span>

$$
\begin{aligned}
\mathrm{FFN}_{\mathrm{GLU}}(x,W,V,W_2)&=(\sigma(xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{Bilinear}}(x,W,V,W_2)&=(xW\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{ReGLU}}(x,W,V,W_2)&=(\max(0,xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{GEGLU}}(x,W,V,W_2)&=(\mathrm{GELU}(xW)\otimes xV)W_2 \\
\mathrm{FFN}_{\mathrm{SwiGLU}}(x,W,V,W_2)&=(\mathrm{Swish}_1(xW)\otimes xV)W_2
\end{aligned}
$$

All of these layers have three weight matrices, as opposed to two for the original FFN. To keep the number of parameters and the amount of computation constant, we reduce the number of hidden units $d_{\mathrm{ff}}$ (the second dimension of $W$ and $V$ and the first dimension of $W_2$) by a factor of $\frac{2}{3}$ when comparing these layers to the original two-matrix version.

<span id="section-3"></span>

## 3 Experiments on Text-to-Text Transfer Transformer (T5)

We test the FFN variants we have described on the transfer-learning setup from [Raf19]. An encoder-decoder transformer model [Vas17] is trained on a denoising objective of predicting missing text segments, and subsequently fine-tuned on various language understanding tasks.

<span id="section-3-1"></span>

### 3.1 Model Architecture

We use the same code base, model architecture, and training task as the base model from [Raf19]. The encoder and decoder each consist of 12 layers, with $d_{\mathrm{model}}=768$. For the attention layers, $h=12$ and $d_k=d_v=64$. The FFN layers have hidden size $d_{\mathrm{ff}}=3072$. As we describe above, for the GLU-variant-based FFN layers, which have thee weight matrices instead of two, we reduce the hidden layer to $d_{\mathrm{ff}}=2048$, so as to maintain the same parameter and operation counts as the base model.

<span id="table-01"></span>

![Table 1. Heldout-set log-perplexity for Transformer models on the segment-filling task.](../../papers/glu-variants/table-01.png)

**Table 1.** Heldout-set log-perplexity for Transformer models on the segment-filling task from [Raf19]. All models are matched for parameters and computation.

<span id="section-3-2"></span>

### 3.2 Pre-Training and Perplexity Results

Identically to [Raf19], we pre-train for 524,288 steps on the span-filling objective on the C4 dataset. Each training batch consists of 128 examples, each of which has an input of 512 tokens and an output of 114 tokens, the output containing multiple spans of tokens which were deleted from the input [+2]. Similarly to [Raf19], we use the Adafactor optimizer [Sha18] and an inverse-square-root learning-rate schedule. We also decay the learning rate linearly for the final 10 percent of the training steps. Our main departure from [Raf19] is that we use no dropout during pre-training. We find this to produce superior results. We compute the log-perplexity on the training objective on a heldout shard of C4, which we believe to be a good indicator of model quality. For each model architecture, we also trained four models for a shorter period (65,536 steps) to measure inter-run variability. The results are listed in [table 1](#table-01). The GEGLU and SwiGLU variants produce the best perplexities.

<span id="section-3-3"></span>

### 3.3 Fine-Tuning

We then fine-tune each fully-trained model once on an examples-proportional mixture of the Stanford Question-Answering Dataset (SQuAD) [Raj16] and all the language understanding tasks in the GLUE [Wan18d] and SuperGlue [Wan19h] benchmarks. [+3] Fine-tuning consists of 131072 steps with a learning rate of $10^{-3}$. As in training, the input sequences for each step have a combined length of approximately 65,536 tokens. Following [Raf19], we use a dropout rate of $0.1$ on the layer outputs, feed-forward hidden-layers and attention weights. The embedding matrices are fixed during fine-tuning.

[Tables 2](#table-02), [3](#table-03) and [4](#table-04) show results on the development sets. For each task, we report the best score of any of the checkpoints recorded during fine-tuning. While the results are noisy, the new GLU-variants perform best on most of the tasks. For comparison, at the bottom of each of the tables we list the reuslts from [Raf19]. The model is identical to our $\mathrm{FFN}_{\mathrm{ReLU}}$ model. Their results are notably worse, which we believe was caused by their use of dropout during pre-training. Also listed are the inter-run standard deviations measured by [Raf19].

<span id="table-02"></span>

![Table 2. GLUE Language-Understanding Benchmark development-set results.](../../papers/glu-variants/table-02.png)

**Table 2.** GLUE Language-Understanding Benchmark [Wan18d] (dev).

<span id="table-03"></span>

![Table 3. SuperGLUE Language-Understanding Benchmark development-set results.](../../papers/glu-variants/table-03.png)

**Table 3.** SuperGLUE Language-Understanding Benchmark [Wan19h] (dev).

<span id="table-04"></span>

![Table 4. SQuAD version 1.1 development-set results.](../../papers/glu-variants/table-04.png)

**Table 4.** SQuAD [Raj16] v1.1 (dev).

<span id="section-4"></span>

## 4 Conclusions

We have extended the GLU family of layers and proposed their use in Transformer. In a transfer-learning setup, the new variants seem to produce better perplexities for the de-noising objective used in pre-training, as well as better results on many downstream language-understanding tasks. These architectures are simple to implement, and have no apparent computational drawbacks. We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.

[+1]: Also in the interest of ML fairness.

[+2]: Each training step took approximately 0.15 seconds on a 32-core TPUv2 cluster.

[+3]: This departs from [Raf19], who fine-tuned separately on the different tasks. We chose one fine-tuning run for simplicity.
