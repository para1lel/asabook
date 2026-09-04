---
title: 'Attention Is All You Need'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/attention-is-all-you-need/
---

> [Ashish Vaswani](https://x.com/ashvaswani) [+equal], [Noam Shazeer](https://www.noamshazeer.com/) [+equal], [Niki Parmar](https://x.com/nikiparmar09) [+equal], [Jakob Uszkoreit](http://jakob.uszkoreit.net/) [+equal], [Llion Jones](https://x.com/LlionJ) [+equal], [Aidan N. Gomez](https://aidangomez.ca/) [+equal] [+google-brain], [Lukasz Kaiser](http://liafa.jussieu.fr/~kaiser/) [+equal], and [Illia Polosukhin](https://ilblackdragon.com/) [+equal] [+google-research]. First submitted to arXiv on June 12, 2017; current version v7. Published in the proceedings of the 31st Conference on Neural Information Processing Systems (NIPS 2017). [Attention Is All You Need](https://arxiv.org/abs/1706.03762v7). [Original PDF](/paper/attention-is-all-you-need.pdf). [TeX source](https://export.arxiv.org/e-print/1706.03762v7). The original PDF remains authoritative for the exact print layout and bibliography.

Provided proper attribution is provided, Google hereby grants permission to reproduce the tables and figures in this paper solely for use in journalistic or scholarly works.

## Abstract

The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.

<span id="section-1"></span>

## 1 Introduction

Recurrent neural networks, long short-term memory [Hoc97] and gated recurrent [Chu14] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [Sut14, Bah14, Cho14a]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [Wu17, Luo15a, Joz16].

Recurrent models typically factor computation along the symbol positions of the input and output sequences. Aligning the positions to steps in computation time, they generate a sequence of hidden states $h_{t}$, as a function of the previous hidden state $h_{t-1}$ and the input for position $t$. This inherently sequential nature precludes parallelization within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples. Recent work has achieved significant improvements in computational efficiency through factorization tricks [Kuc17] and conditional computation [Sha17], while also improving model performance in case of the latter. The fundamental constraint of sequential computation, however, remains.

Attention mechanisms have become an integral part of compelling sequence modeling and transduction models in various tasks, allowing modeling of dependencies without regard to their distance in the input or output sequences [Bah14, Kim17]. In all but a few cases [Par16], however, such attention mechanisms are used in conjunction with a recurrent network.

In this work we propose the Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output. The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

<span id="section-2"></span>

## 2 Background

The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [Kai16a], ByteNet [Kal17] and ConvS2S [Geh17], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [Hoc01]. In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to averaging attention-weighted positions, an effect we counteract with Multi-Head Attention as described in [Section 3.2](#section-3-2).

Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence. Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [Che16e, Par16, Pau17, Lin17b].

End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [Suk15].

To the best of our knowledge, however, the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution. In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [Kai16, Kal17] and [Geh17].

<span id="section-3"></span>

## 3 Model Architecture

<span id="figure-01"></span>

![Refer to caption](../../papers/attention-is-all-you-need/figure-01.png)

**Figure 1.** The Transformer - model architecture.

Most competitive neural sequence transduction models have an encoder-decoder structure [Cho14a, Bah14, Sut14]. Here, the encoder maps an input sequence of symbol representations $(x_{1},\dots,x_{n})$ to a sequence of continuous representations $\mathbf{z}=(z_{1},\dots,z_{n})$. Given $\mathbf{z}$, the decoder then generates an output sequence $(y_{1},\dots,y_{m})$ of symbols one element at a time. At each step the model is auto-regressive [Gra13], consuming the previously generated symbols as additional input when generating the next.

The Transformer follows this overall architecture using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder, shown in the left and right halves of [Figure 1](#figure-01), respectively.

<span id="section-3-1"></span>

### 3.1 Encoder and Decoder Stacks

**Encoder:** The encoder is composed of a stack of $N=6$ identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network. We employ a residual connection [He16] around each of the two sub-layers, followed by layer normalization [Ba16]. That is, the output of each sub-layer is $\mathrm{LayerNorm}(x+\mathrm{Sublayer}(x))$, where $\mathrm{Sublayer}(x)$ is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension $d_{\mathrm{model}}=512$.

**Decoder:** The decoder is also composed of a stack of $N=6$ identical layers. In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This masking, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position $i$ can depend only on the known outputs at positions less than $i$.

<span id="section-3-2"></span>

### 3.2 Attention

An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

<span id="section-3-2-1"></span>

#### 3.2.1 Scaled Dot-Product Attention

We call our particular attention "Scaled Dot-Product Attention" ([Figure 2](#figure-02)). The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$. We compute the dot products of the query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix $Q$. The keys and values are also packed together into matrices $K$ and $V$. We compute the matrix of outputs as:

<span id="equation-01"></span>

$$
\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_{k}}}\right)V
$$

The two most commonly used attention functions are additive attention [Bah14], and dot-product (multiplicative) attention. Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_{k}}}$. Additive attention computes the compatibility function using a feed-forward network with a single hidden layer. While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

While for small values of $d_{k}$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_{k}$ [Bri17]. We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients [+dot-product]. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_{k}}}$.

<span id="section-3-2-2"></span>

#### 3.2.2 Multi-Head Attention

<span id="figure-02"></span>

![Refer to caption](../../papers/attention-is-all-you-need/figure-02.png)

**Figure 2.** (left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel.

Instead of performing a single attention function with $d_{\mathrm{model}}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_{k}$, $d_{k}$ and $d_{v}$ dimensions, respectively. On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_{v}$-dimensional output values. These are concatenated and once again projected, resulting in the final values, as depicted in [Figure 2](#figure-02).

Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

$$
\begin{aligned}
\mathrm{MultiHead}(Q,K,V)&=\mathrm{Concat}(\mathrm{head}_{1},\dots,\mathrm{head}_{h})W^{O}\\
\text{where}\quad \mathrm{head}_{i}&=\mathrm{Attention}(QW^{Q}_{i},KW^{K}_{i},VW^{V}_{i})
\end{aligned}
$$

Where the projections are parameter matrices $W^{Q}_{i}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{k}}$, $W^{K}_{i}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{k}}$, $W^{V}_{i}\in\mathbb{R}^{d_{\mathrm{model}}\times d_{v}}$ and $W^{O}\in\mathbb{R}^{h d_{v}\times d_{\mathrm{model}}}$.

In this work we employ $h=8$ parallel attention layers, or heads. For each of these we use $d_{k}=d_{v}=d_{\mathrm{model}}/h=64$. Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

<span id="section-3-2-3"></span>

#### 3.2.3 Applications of Attention in our Model

The Transformer uses multi-head attention in three different ways:

- In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [Wu17, Bah14, Geh17].
- The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- Similarly, self-attention layers in the decoder allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. See [Figure 2](#figure-02).

<span id="section-3-3"></span>

### 3.3 Position-wise Feed-Forward Networks

In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

<span id="equation-02"></span>

$$
\mathrm{FFN}(x)=\max(0,xW_{1}+b_{1})W_{2}+b_{2}
$$

While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1. The dimensionality of input and output is $d_{\mathrm{model}}=512$, and the inner-layer has dimensionality $d_{\mathrm{ff}}=2048$.

<span id="section-3-4"></span>

### 3.4 Embeddings and Softmax

Similarly to other sequence transduction models, we use learned embeddings to convert the input tokens and output tokens to vectors of dimension $d_{\mathrm{model}}$. We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities. In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [Pre16]. In the embedding layers, we multiply those weights by $\sqrt{d_{\mathrm{model}}}$.

<span id="section-3-5"></span>

### 3.5 Positional Encoding

Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. The positional encodings have the same dimension $d_{\mathrm{model}}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [Geh17].

In this work, we use sine and cosine functions of different frequencies:

$$
\mathrm{PE}_{(\mathrm{pos},2i)}=\sin(\mathrm{pos}/10000^{2i/d_{\mathrm{model}}})
$$

$$
\mathrm{PE}_{(\mathrm{pos},2i+1)}=\cos(\mathrm{pos}/10000^{2i/d_{\mathrm{model}}})
$$

where $\mathrm{pos}$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $10000\cdot 2\pi$. We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $\mathrm{PE}_{\mathrm{pos}+k}$ can be represented as a linear function of $\mathrm{PE}_{\mathrm{pos}}$.

We also experimented with using learned positional embeddings [Geh17] instead, and found that the two versions produced nearly identical results (see [Table 3](#table-03) row (E)). We chose the sinusoidal version because it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training.

<span id="section-4"></span>

## 4 Why Self-Attention

In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $(x_{1},\dots,x_{n})$ to another sequence of equal length $(z_{1},\dots,z_{n})$, with $x_{i},z_{i}\in\mathbb{R}^{d}$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [Hoc01]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

<span id="table-01"></span>

![Original paper Table 1](../../papers/attention-is-all-you-need/table-01.png)

**Table 1.** Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.

As noted in [Table 1](#table-01), a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $O(n)$ sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length $n$ is smaller than the representation dimensionality $d$, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [Wu17] and byte-pair [Sen15] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size $r$ in the input sequence centered around the respective output position. This would increase the maximum path length to $O(n/r)$. We plan to investigate this approach further in future work.

A single convolutional layer with kernel width $k<n$ does not connect all pairs of input and output positions. Doing so requires a stack of $O(n/k)$ convolutional layers in the case of contiguous kernels, or $O(\log_{k}(n))$ in the case of dilated convolutions [Kal17], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of $k$. Separable convolutions [Cho16a], however, decrease the complexity considerably, to $O(k\cdot n\cdot d+n\cdot d^{2})$. Even with $k=n$, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

<span id="section-5"></span>

## 5 Training

This section describes the training regime for our models.

<span id="section-5-1"></span>

### 5.1 Training Data and Batching

We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding [Bri17], which has a shared source-target vocabulary of about 37000 tokens. For English-French, we used the significantly larger WMT 2014 English-French dataset consisting of 36M sentences and split tokens into a 32000 word-piece vocabulary [Wu17]. Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens.

<span id="section-5-2"></span>

### 5.2 Hardware and Schedule

We trained our models on one machine with 8 NVIDIA P100 GPUs. For our base models using the hyperparameters described throughout the paper, each training step took about 0.4 seconds. We trained the base models for a total of 100,000 steps or 12 hours. For our big models,(described on the bottom line of [table 3](#table-03)), step time was 1.0 seconds. The big models were trained for 300,000 steps (3.5 days).

<span id="section-5-3"></span>

### 5.3 Optimizer

We used the Adam optimizer [Kin15] with $\beta_{1}=0.9$, $\beta_{2}=0.98$ and $\epsilon=10^{-9}$. We varied the learning rate over the course of training, according to the formula:

<span id="equation-03"></span>

$$
\mathrm{lrate}=d_{\mathrm{model}}^{-0.5}\cdot\min\left(\mathrm{step\_num}^{-0.5},\mathrm{step\_num}\cdot\mathrm{warmup\_steps}^{-1.5}\right)
$$

This corresponds to increasing the learning rate linearly for the first $\mathrm{warmup\_steps}$ training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used $\mathrm{warmup\_steps}=4000$.

<span id="section-5-4"></span>

### 5.4 Regularization

We employ three types of regularization during training:

**Residual Dropout** We apply dropout [Sri14] to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of $P_{\mathrm{drop}}=0.1$.

**Label Smoothing** During training, we employed label smoothing of value $\epsilon_{\mathrm{ls}}=0.1$ [Sze16]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.

<span id="section-6"></span>

## 6 Results

<span id="section-6-1"></span>

### 6.1 Machine Translation

<span id="table-02"></span>

![Original paper Table 2](../../papers/attention-is-all-you-need/table-02.png)

**Table 2.** The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost.

On the WMT 2014 English-to-German translation task, the big transformer model (Transformer (big) in [Table 2](#table-02)) outperforms the best previously reported models (including ensembles) by more than $2.0$ BLEU, establishing a new state-of-the-art BLEU score of $28.4$. The configuration of this model is listed in the bottom line of [Table 3](#table-03). Training took $3.5$ days on $8$ P100 GPUs. Even our base model surpasses all previously published models and ensembles, at a fraction of the training cost of any of the competitive models.

On the WMT 2014 English-to-French translation task, our big model achieves a BLEU score of $41.0$, outperforming all of the previously published single models, at less than $1/4$ the training cost of the previous state-of-the-art model. The Transformer (big) model trained for English-to-French used dropout rate $P_{\mathrm{drop}}=0.1$, instead of $0.3$.

For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals. For the big models, we averaged the last 20 checkpoints. We used beam search with a beam size of $4$ and length penalty $\alpha=0.6$ [Wu17]. These hyperparameters were chosen after experimentation on the development set. We set the maximum output length during inference to input length + $50$, but terminate early when possible [Wu17].

[Table 2](#table-02) summarizes our results and compares our translation quality and training costs to other model architectures from the literature. We estimate the number of floating point operations used to train a model by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU [+flops].

<span id="section-6-2"></span>

### 6.2 Model Variations

<span id="table-03"></span>

![Original paper Table 3](../../papers/attention-is-all-you-need/table-03.png)

**Table 3.** Variations on the Transformer architecture. Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013. Listed perplexities are per-wordpiece, according to our byte-pair encoding, and should not be compared to per-word perplexities.

To evaluate the importance of different components of the Transformer, we varied our base model in different ways, measuring the change in performance on English-to-German translation on the development set, newstest2013. We used beam search as described in the previous section, but no checkpoint averaging. We present these results in [Table 3](#table-03).

In [Table 3](#table-03) rows (A), we vary the number of attention heads and the attention key and value dimensions, keeping the amount of computation constant, as described in [Section 3.2.2](#section-3-2-2). While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads.

In [Table 3](#table-03) rows (B), we observe that reducing the attention key size $d_{k}$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial. We further observe in rows (C) and (D) that, as expected, bigger models are better, and dropout is very helpful in avoiding over-fitting. In row (E) we replace our sinusoidal positional encoding with learned positional embeddings [Geh17], and observe nearly identical results to the base model.

<span id="section-6-3"></span>

### 6.3 English Constituency Parsing

<span id="table-04"></span>

![Original paper Table 4](../../papers/attention-is-all-you-need/table-04.png)

**Table 4.** The Transformer generalizes well to English constituency parsing (Results are on Section&nbsp;23 of WSJ)

To evaluate if the Transformer can generalize to other tasks we performed experiments on English constituency parsing. This task presents specific challenges: the output is subject to strong structural constraints and is significantly longer than the input. Furthermore, RNN sequence-to-sequence models have not been able to attain state-of-the-art results in small-data regimes [Vin15].

We trained a 4-layer transformer with $d_{\mathrm{model}}=1024$ on the Wall Street Journal (WSJ) portion of the Penn Treebank [Mar93], about 40K training sentences. We also trained it in a semi-supervised setting, using the larger high-confidence and BerkleyParser corpora from with approximately 17M sentences [Vin15]. We used a vocabulary of 16K tokens for the WSJ only setting and a vocabulary of 32K tokens for the semi-supervised setting.

We performed only a small number of experiments to select the dropout, both attention and residual ([Section 5.4](#section-5-4)), learning rates and beam size on the Section&nbsp;22 development set, all other parameters remained unchanged from the English-to-German base translation model. During inference, we increased the maximum output length to input length + $300$. We used a beam size of $21$ and $\alpha=0.3$ for both WSJ only and the semi-supervised setting.

Our results in [Table 4](#table-04) show that despite the lack of task-specific tuning our model performs surprisingly well, yielding better results than all previously reported models with the exception of the Recurrent Neural Network Grammar [Dye16].

In contrast to RNN sequence-to-sequence models [Vin15], the Transformer outperforms the BerkeleyParser [Pet06] even when training only on the WSJ training set of 40K sentences.

<span id="section-7"></span>

## 7 Conclusion

In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

The code we used to train and evaluate our models is available at [https://github.com/tensorflow/tensor2tensor](https://github.com/tensorflow/tensor2tensor).

**Acknowledgements** We are grateful to Nal Kalchbrenner and Stephan Gouws for their fruitful comments, corrections and inspiration.

<span id="section-8"></span>

## 8 Attention Visualizations

<span id="figure-03"></span>

![Refer to caption](../../papers/attention-is-all-you-need/figure-03.png)

**Figure 3.** An example of the attention mechanism following long-distance dependencies in the encoder self-attention in layer 5 of 6. Many of the attention heads attend to a distant dependency of the verb ‘making’, completing the phrase ‘making…more difficult’. Attentions here shown only for the word ‘making’. Different colors represent different heads. Best viewed in color.

<span id="figure-04"></span>

![Refer to caption](../../papers/attention-is-all-you-need/figure-04.png)

**Figure 4.** Two attention heads, also in layer 5 of 6, apparently involved in anaphora resolution. Top: Full attentions for head 5. Bottom: Isolated attentions from just the word ‘its’ for attention heads 5 and 6. Note that the attentions are very sharp for this word.

<span id="figure-05"></span>

![Refer to caption](../../papers/attention-is-all-you-need/figure-05.png)

**Figure 5.** Many of the attention heads exhibit behaviour that seems related to the structure of the sentence. We give two such examples above, from two different heads from the encoder self-attention at layer 5 of 6. The heads clearly learned to perform different tasks.

[+equal]: Equal contribution. Listing order is random. Jakob proposed replacing RNNs with self-attention and started the effort to evaluate this idea. Ashish, with Illia, designed and implemented the first Transformer models and has been crucially involved in every aspect of this work. Noam proposed scaled dot-product attention, multi-head attention and the parameter-free position representation and became the other person involved in nearly every detail. Niki designed, implemented, tuned and evaluated countless model variants in our original codebase and tensor2tensor. Llion also experimented with novel model variants, was responsible for our initial codebase, and efficient inference and visualizations. Lukasz and Aidan spent countless long days designing various parts of and implementing tensor2tensor, replacing our earlier codebase, greatly improving results and massively accelerating our research.

[+google-brain]: Work performed while at Google Brain.

[+google-research]: Work performed while at Google Research.

[+dot-product]: To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean $0$ and variance $1$. Then their dot product, $q\cdot k=\sum_{i=1}^{d_{k}}q_{i}k_{i}$, has mean $0$ and variance $d_{k}$.

[+flops]: We used values of 2.8, 3.7, 6.0 and 9.5 TFLOPS for K80, K40, M40 and P100, respectively.
