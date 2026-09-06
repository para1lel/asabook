---
title: StreamingLLM
createTime: 2026/09/06 14:00:00
permalink: /en/papers/streaming-llm/
---

> [Guangxuan Xiao](https://guangxuanx.com/) [+1], [Yuandong Tian](https://yuandong-tian.com/), [Beidi Chen](https://www.andrew.cmu.edu/user/beidic/), [Song Han](https://www.rle.mit.edu/people/song-han/), and [Mike Lewis](https://ai.meta.com/people/209431298931133/mike-lewis/). First submitted to arXiv on September 29, 2023; published at ICLR 2024; current version v4. [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453v4). <a href="/paper/streaming-llm.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2309.17453). [TeX source](https://export.arxiv.org/e-print/2309.17453v4). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Deploying Large Language Models (LLMs) in streaming applications such as multi-round dialogue, where long interactions are expected, is urgently needed but poses two major challenges. Firstly, during the decoding stage, caching previous tokens' Key and Value states (KV) consumes extensive memory. Secondly, popular LLMs cannot generalize to longer texts than the training sequence length. Window attention, where only the most recent KVs are cached, is a natural approach — but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely *attention sink*, that keeping the KV of initial tokens will largely recover the performance of window attention. In this paper, we first demonstrate that the emergence of *attention sink* is due to the strong attention scores towards initial tokens as a “sink” even if they are not semantically important. Based on the above analysis, we introduce StreamingLLM, an efficient framework that enables LLMs trained with a *finite length* attention window to generalize to *infinite sequence length* without any fine-tuning. We show that StreamingLLM can enable Llama-2, MPT, Falcon, and Pythia to perform stable and efficient language modeling with up to 4 million tokens and more. In addition, we discover that adding a placeholder token as a dedicated attention sink during pre-training can further improve streaming deployment. In streaming settings, StreamingLLM outperforms the sliding window recomputation baseline by up to 22.2$\times$ speedup. Code and datasets are provided in the [link](https://github.com/mit-han-lab/streaming-llm).

<span id="section-1"></span>

## 1 Introduction

Large Language Models (LLMs) [Rad18, Bro20, Zha22, Ope23, Tou23, Tou23a] are becoming ubiquitous, powering many natural language processing applications such as dialog systems [Sch22, Tao23, Chi23], document summarization [Goy20, Zha23r], code completion [Che21, Roz23] and question answering [Kam23a]. To unleash the full potential of pretrained LLMs, they should be able to efficiently and accurately perform long sequence generation. For example, an ideal ChatBot assistant can stably work over the content of recent day-long conversations. However, it is very challenging for LLM to generalize to longer sequence lengths than they have been pretrained on, e.g., 4K for Llama-2 [Tou23a].

The reason is that LLMs are constrained by the attention window during pre-training. Despite substantial efforts to expand this window size [Che23x, Kai23, Pen23] and improve training [Dao22, Dao23a] and inference [Pop22, Xia23, Ana23, Wan21f, Zha23g] efficiency for lengthy inputs, the acceptable sequence length remains intrinsically *finite*, which doesn't allow persistent deployments.

In this paper, we first introduce the concept of LLM streaming applications and ask the question:

> *Can we deploy an LLM for infinite-length inputs without sacrificing efficiency and performance?*

When applying LLMs for infinite input streams, two primary challenges arise:

1. During the decoding stage, Transformer-based LLMs cache the Key and Value states (KV) of all previous tokens, as illustrated in [Figure 1](#figure-01) (a), which can lead to excessive memory usage and increasing decoding latency [Pop22].
2. Existing models have limited length extrapolation abilities, i.e., their performance degrades [Pre21, Che23x] when the sequence length goes beyond the attention window size set during pre-training.

An intuitive approach, known as window attention [Bel20] ([Figure 1](#figure-01) b), maintains only a fixed-size sliding window on the KV states of most recent tokens. Although it ensures constant memory usage and decoding speed after the cache is initially filled, the model collapses once the sequence length exceeds the cache size, i.e., *even just evicting the KV of the first token*, as illustrated in [Figure 3](#figure-03). Another strategy is the sliding window with re-computation (shown in [Figure 1](#figure-01) c), which rebuilds the KV states of recent tokens for each generated token. While it offers strong performance, this approach is significantly slower due to the computation of quadratic attention within its window, making this method impractical for real-world streaming applications.

<span id="figure-01"></span>

![Illustration of StreamingLLM versus existing methods](../../papers/streaming-llm/figure-01.png)

**Figure 1.** **Illustration of StreamingLLM *vs.* existing methods.** The language model, pre-trained on texts of length $L$, predicts the $T$th token ($T\gg L$). (a) Dense Attention has $O(T^2)$ time complexity and an increasing cache size. Its performance decreases when the text length exceeds the pre-training text length. (b) Window Attention caches the most recent $L$ tokens' KV. While efficient in inference, performance declines sharply once the starting tokens' keys and values are evicted. (c) Sliding Window with Re-computation rebuilds the KV states from the $L$ recent tokens for each new token. While it performs well on long texts, its $O(T L^2)$ complexity, stemming from quadratic attention in context re-computation, makes it considerably slow. (d) StreamingLLM keeps the *attention sink* (several initial tokens) for stable attention computation, combined with the recent tokens. It's efficient and offers stable performance on extended texts. Perplexities are measured using the Llama-2-13B model on the first book (65K tokens) in the PG-19 test set.

To understand the failure of window attention, we find an interesting phenomenon of autoregressive LLMs: a surprisingly large amount of attention score is allocated to the initial tokens, irrespective of their relevance to the language modeling task, as visualized in [Figure 2](#figure-02). We term these tokens “**attention sinks**”. Despite their lack of semantic significance, they collect significant attention scores. We attribute the reason to the Softmax operation, which requires attention scores to sum up to one for all contextual tokens. Thus, even when the current query does not have a strong match in many previous tokens, the model still needs to allocate these unneeded attention values somewhere so it sums up to one. The reason behind *initial* tokens as sink tokens is intuitive: initial tokens are visible to almost all subsequent tokens because of the autoregressive language modeling nature, making them more readily trained to serve as attention sinks.

Based on the above insights, we propose StreamingLLM, a simple and efficient framework that enables LLMs trained with a finite attention window to work on text of infinite length without fine-tuning. StreamingLLM exploits the fact that attention sinks have high attention values, and preserving them can maintain the attention score distribution close to normal. Therefore, StreamingLLM simply keeps the attention sink tokens' KV (with just 4 initial tokens sufficing) together with the sliding window's KV to anchor the attention computation and stabilize the model's performance. With StreamingLLM, models including Llama-2-[7, 13, 70]B, MPT-[7, 30]B, Falcon-[7, 40]B, and Pythia-[2.9,6.9,12]B can reliably model 4 million tokens, and potentially even more. Compared with the only viable baseline, sliding window with recomputation, StreamingLLM achieves up to 22.2$\times$ speedup, realizing the streaming use of LLMs.

Furthermore, we confirm our attention sink hypothesis and demonstrate that language models can be pre-trained to require only a single attention sink token for streaming deployment. Specifically, we suggest that an extra learnable token at the beginning of all training samples can serve as a designated attention sink. By pre-training 160-million parameter language models from scratch, we demonstrate that adding this single sink token preserves the model's performance in streaming cases. This stands in contrast to vanilla models, which necessitate the reintroduction of multiple initial tokens as attention sinks to achieve the same performance level.

<span id="figure-02"></span>

![Average attention logits in Llama-2-7B](../../papers/streaming-llm/figure-02.png)

**Figure 2.** Visualization of the *average* attention logits in Llama-2-7B over 256 sentences, each with a length of 16. Observations include: (1) The attention maps in the first two layers (layers 0 and 1) exhibit the “local” pattern, with recent tokens receiving more attention. (2) Beyond the bottom two layers, the model heavily attends to the initial token across all layers and heads.

Finally, we emphasize that StreamingLLM efficiently generates coherent text from tokens within the KV cache without extending the LLMs' context length. It suits continuous operation needs with minimal memory use and past data reliance. Additionally, StreamingLLM can complement context extension methods to increase the attendable recent context.

<span id="section-2"></span>

## 2 Related Work

Extensive research has been done on applying LLMs to lengthy texts, with three main areas of focus: **Length Extrapolation**, **Context Window Extension**, and **Improving LLMs' Utilization of Long Text**. While seemingly related, it's worth noting that progress in one direction doesn't necessarily lead to progress in the other. For example, extending the context size of LLMs doesn't improve the model's performance beyond the context size, and neither approach ensures effective use of the long context. Our StreamingLLM framework primarily lies in the first category, where LLMs are applied to text significantly exceeding the pre-training window size, potentially even of infinite length. We do not expand the attention window size of LLMs or enhance the model's memory and usage on long texts. The last two categories are orthogonal to our focus and could be integrated with our techniques.

**Length extrapolation** aims to enable language models trained on shorter texts to handle longer ones during testing. A predominant avenue of research targets the development of relative position encoding methods for Transformer models, enabling them to function beyond their training window. One such initiative is Rotary Position Embeddings (RoPE) [Su21], which transforms the queries and keys in every attention layer for relative position integration. Despite its promise, subsequent research [Pre21, Che23x] indicated its underperformance on text that exceeds the training window. Another approach, ALiBi [Pre21], biases the query-key attention scores based on their distance, thereby introducing relative positional information. While this exhibited improved extrapolation, our tests on MPT models highlighted a breakdown when the text length was vastly greater than the training length. Current methodologies, however, have yet to achieve infinite length extrapolation, causing no existing LLMs to fit for streaming applications.

**Context Window Extension** centers on expanding the LLMs' context window, enabling the processing of more tokens in one forward pass. A primary line of work addresses the training efficiency problem. Given the attention to computation's quadratic complexity during training, developing a long-context LLM is both a computational and memory challenge. Solutions have ranged from system-focused optimizations like FlashAttention [Dao22, Dao23a], which accelerates attention computation and reduces memory footprint, to approximate attention methods [Zah20, Bel20, Wan20a, Kit20] that trade model quality for efficiency. Recently, there has been a surge of work on extending pre-trained LLMs with RoPE [Che23x, Kai23, Blo23a, Pen23], involving position interpolation and fine-tuning. However, all the aforementioned techniques only extend LLMs' context window to a limited extent, which falls short of our paper's primary concern of handling limitless inputs.

**Improving LLMs' Utilization of Long Text** optimizes LLMs to better capture and employ the content within the context rather than merely taking them as inputs. As highlighted by Liu et al. [Liu23z] and Li et al. [Li23h], success in the previously mentioned two directions does not necessarily translate to competent utilization of lengthy contexts. Addressing this effective usage of prolonged contexts within LLMs is still a challenge. Our work concentrates on stably harnessing the most recent tokens, enabling the seamless streaming application of LLMs.

<span id="section-3"></span>

## 3 StreamingLLM

<span id="figure-03"></span>

![Language modeling perplexity on 20K-token texts](../../papers/streaming-llm/figure-03.png)

**Figure 3.** Language modeling perplexity on texts with 20K tokens across various LLM. Observations reveal consistent trends: (1) Dense attention fails once the input length surpasses the pre-training attention window size. (2) Window attention collapses once the input length exceeds the cache size, i.e., the initial tokens are evicted. (3) StreamingLLM demonstrates stable performance, with its perplexity nearly matching that of the sliding window with re-computation baseline.

<span id="section-3-1"></span>

### 3.1 The Failure of Window Attention and Attention Sinks

While the window attention technique offers efficiency during inference, it results in an exceedingly high language modeling perplexity. Consequently, the model's performance is unsuitable for deployment in streaming applications. In this section, we use the concept of *attention sink* to explain the failure of window attention, serving as the inspiration behind StreamingLLM.

**Identifying the Point of Perplexity Surge.** [Figure 3](#figure-03) shows the perplexity of language modeling on a 20K token text. It is evident that perplexity spikes when the text length surpasses the cache size, led by the exclusion of initial tokens. This suggests that the initial tokens, regardless of their distance from the predicted tokens, are crucial for maintaining the stability of LLMs.

**Why do LLMs break when removing *initial* tokens' KV?** We visualize attention maps from all layers and heads of the Llama-2-7B and models in [Figure 2](#figure-02). We find that, beyond the bottom two layers, the model consistently focuses on the initial tokens across all layers and heads. The implication is clear: removing these initial tokens' KV will remove a considerable portion of the denominator in the SoftMax function ([Equation 1](#equation-01)) in attention computation. This alteration leads to a significant shift in the distribution of attention scores away from what would be expected in normal inference settings.

<span id="equation-01"></span>

$$
\mathrm{SoftMax}(x)_i = \frac{e^{x_i}}{e^{x_1} + \sum_{j=2}^{N} e^{x_j}}, \quad x_1 \gg x_j, j \in 2, \dots, N
$$

There are two possible explanations for the importance of the initial tokens in language modeling: (1) Either their semantics are crucial, or (2) the model learns a bias towards their absolute position. To distinguish between these possibilities, we conduct experiments ([Table 1](#table-01)), wherein the first four tokens are substituted with the linebreak token “\n”. The observations indicate that the model still significantly emphasizes these initial linebreak tokens. Furthermore, reintroducing them restores the language modeling perplexity to levels comparable to having the original initial tokens. This suggests that the absolute position of the starting tokens, rather than their semantic value, holds greater significance.

**LLMs attend to Initial Tokens as Attention Sinks.** To explain why the model disproportionately focuses on initial tokens—regardless of their semantic relevance to language modeling, we introduce the concept of “*attention sink*”. The nature of the SoftMax function ([Equation 1](#equation-01)) prevents all attended tokens from having zero values. This requires aggregating some information from other tokens across all heads in all layers, even if the current embedding has sufficient self-contained information for its prediction. Consequently, the model tends to dump unnecessary attention values to specific tokens. A similar observation has been made in the realm of quantization outliers [Xia23, Bon23], leading to the proposal of SoftMax-Off-by-One [Mil23] as a potential remedy.

<span id="table-01"></span>

![Window attention and linebreak-token perplexity](../../papers/streaming-llm/table-01.png)

**Table 1.** Window attention has poor performance on long text. The perplexity is restored when we reintroduce the initial four tokens alongside the recent 1020 tokens (4+1020). Substituting the original four initial tokens with linebreak tokens “\n” (4“\n”+1020) achieves comparable perplexity restoration. Cache config x+y denotes adding x initial tokens with y recent tokens. Perplexities are measured on the first book (65K tokens) in the PG19 test set.

<span id="table-02"></span>

![Effects of reintroduced initial token numbers on StreamingLLM](../../papers/streaming-llm/table-02.png)

**Table 2.** Effects of reintroduced initial token numbers on StreamingLLM. (1) Window attention (0+y) has a drastic increase in perplexity. (2) Introducing one or two initial tokens doesn't fully restore model perplexity, showing that the model doesn't solely use the first token as the attention sink. (3) Introducing four initial tokens generally suffices; further additions have diminishing returns. Cache config x+y denotes adding x initial tokens to y recent tokens. Perplexities are evaluated on 400K tokens in the concatenated PG19 test set.

Why do various autoregressive LLMs, such as Llama-2, MPT, Falcon, and Pythia, consistently focus on *initial tokens* as their attention sinks, rather than other tokens? Our explanation is straightforward: Due to the sequential nature of autoregressive language modeling, initial tokens are visible to all subsequent tokens, while later tokens are only visible to a limited set of subsequent tokens. As a result, initial tokens are more easily trained to serve as attention sinks, capturing unnecessary attention.

We've noted that LLMs are typically trained to utilize multiple initial tokens as attention sinks rather than just one. As illustrated in [Table 2](#table-02), the introduction of four initial tokens, as attention sinks, suffices to restore the LLM's performance. In contrast, adding just one or two doesn't achieve full recovery. We believe this pattern emerges because these models didn't include a consistent starting token across all input samples during pre-training. Although Llama-2 does prefix each paragraph with a “`<s>`” token, it's applied before text chunking, resulting in a mostly random token occupying the zeroth position. This lack of a uniform starting token leads the model to use several initial tokens as attention sinks. We hypothesize that by incorporating a stable learnable token at the start of all training samples, it could singularly act as a committed attention sink, eliminating the need for multiple initial tokens to ensure consistent streaming. We will validate this hypothesis in [Section 3.3](#section-3-3).

<span id="section-3-2"></span>

### 3.2 Rolling KV Cache with Attention Sinks

<span id="figure-04"></span>

![The KV cache of StreamingLLM](../../papers/streaming-llm/figure-04.png)

**Figure 4.** The KV cache of StreamingLLM.

To enable LLM streaming in already trained LLMs, we propose a straightforward method that can recover window attention's perplexity without any model finetuning. Alongside the current sliding window tokens, we reintroduce a few starting tokens' KV in the attention computation. The KV cache in StreamingLLM can be conceptually divided into two parts, as illustrated in [Figure 4](#figure-04): (1) Attention sinks (four initial tokens) stabilize the attention computation; 2) Rolling KV Cache retains the most recent tokens, crucial for language modeling. StreamingLLM' design is versatile and can be seamlessly incorporated into any autoregressive language model that employs relative positional encoding, such as RoPE [Su21] and ALiBi [Pre21].

When determining the relative distance and adding positional information to tokens, StreamingLLM focuses on positions *within the cache* rather than those *in the original text*. This distinction is crucial for StreamingLLM's performance. For instance, if the current cache ([Figure 4](#figure-04)) has tokens [0, 1, 2, 3, 6, 7, 8] and is in the process of decoding the 9th token, the positions assigned are [0, 1, 2, 3, 4, 5, 6, 7], rather than the positions in the original text, which would be [0, 1, 2, 3, 6, 7, 8, 9].

For encoding like RoPE, we cache the Keys of tokens *prior to* introducing the rotary transformation. Then, we apply position transformation to the keys in the rolling cache at each decoding phase. On the other hand, integrating with ALiBi is more direct. Here, the contiguous linear bias is applied instead of a 'jumping' bias to the attention scores. This method of assigning positional embedding within the cache is crucial to StreamingLLM's functionality, ensuring that the model operates efficiently even beyond its pre-training attention window size.

<span id="section-3-3"></span>

### 3.3 Pre-Training LLMs with Attention Sinks

<span id="table-03"></span>

![Comparison of vanilla attention, Zero Sink, and a learnable Sink Token](../../papers/streaming-llm/table-03.png)

**Table 3.** Comparison of vanilla attention with prepending a zero token and a learnable sink token during pre-training. To ensure stable streaming perplexity, the vanilla model requires several initial tokens. While Zero Sink shows a slight improvement, it still needs other initial tokens. Conversely, the model trained with a learnable Sink Token shows stable streaming perplexity with only the sink token added. Cache config $x$+$y$ denotes adding $x$ initial tokens with $y$ recent tokens. Perplexity is evaluated on the first sample in the PG19 test set.

As elaborated in [Section 3.1](#section-3-1), a significant reason for the model's excessive attention to multiple initial tokens is the absence of a designated sink token to offload excessive attention scores. Due to this, the model inadvertently uses globally visible tokens, primarily the initial ones, as attention sinks. A potential remedy can be the intentional inclusion of a global trainable attention sink token, denoted as a “Sink Token”, which would serve as a repository for unnecessary attention scores. Alternatively, replacing the conventional SoftMax function with a variant like SoftMax-off-by-One [Mil23],

<span id="equation-02"></span>

$$
\mathrm{SoftMax}_1(x)_i = \frac{e^{x_i}}{1+\sum_{j=1}^{N} e^{x_j}},
$$

which does not require the attention scores on all contextual tokens to sum up to one, may also be effective. Note that $\mathrm{SoftMax}_1$ is equivalent to prepending a token with an all-zero Key and Value features in the attention computation. We denote this method as “Zero Sink” to fit our framework.

For validation, we pre-train three language models with 160 million parameters from scratch under identical settings. The first model utilizes the standard SoftMax attention (Vanilla), the second replaced the regular attention mechanism with $\mathrm{SoftMax}_1$ (Zero Sink), and one prepending a learnable placeholder token (Sink Token) in all training samples. As shown in [Table 3](#table-03), while the zero sink alleviates the attention sink problem to some extent, the model still relies on other initial tokens as attention sinks. Introducing a sink token is highly effective in stabilizing the attention mechanism. Simply pairing this sink token with recent tokens sufficiently anchors the model's performance, and the resulting evaluation perplexity is even marginally improved. Given these findings, we recommend training future LLMs with a sink token in all samples to optimize streaming deployment.

<span id="section-4"></span>

## 4 Experiments

We evaluate StreamingLLM using four prominent recent model families: Llama-2 [Tou23a], MPT [Mos23], PyThia [Bid23], and Falcon [Alm23]. Notably, Llama-2, Falcon, and Pythia incorporate RoPE [Su21], whereas MPT employs ALiBi [Pre21] — two of the most influential position encoding techniques in recent research. Our diverse model selection ensures the validity and robustness of our findings. We benchmark StreamingLLM against established baselines such as dense attention, window attention, and the sliding window approach with re-computation. In all subsequent experiments with StreamingLLM, we default to using four initial tokens as attention sinks unless stated otherwise.

<span id="section-4-1"></span>

### 4.1 Language Modeling on Long Texts Across LLM Families and Scales

We firstly evaluate StreamingLLM's language modeling perplexity using the concatenated PG19 [Rae20] test set, which contains 100 long books. For Llama-2 models, the cache size is set at 2048, while for Falcon, Pythia, and MPT models, it's set at 1024. This is half the pre-training window size chosen to enhance visualization clarity.

[Figure 3](#figure-03) illustrates that StreamingLLM can match the oracle baseline (sliding window with re-computation) in terms of perplexity on texts spanning 20K tokens. Meanwhile, the dense attention technique fails when the input length exceeds its pre-training window, and the window attention technique struggles when the input length surpasses the cache size, leading to the eviction of the initial tokens. In [Figure 5](#figure-05), we further substantiate that StreamingLLM can reliably handle exceptionally extended texts, encompassing more than 4 million tokens, across a spectrum of model families and scales. This includes Llama-2-[7,13,70]B, Falcon-[7,40]B, Pythia-[2.8,6.9,12]B, and MPT-[7,30]B.

<span id="figure-05"></span>

![StreamingLLM perplexity on four-million-token texts](../../papers/streaming-llm/figure-05.png)

**Figure 5.** Language modeling perplexity of StreamingLLM on super long texts with 4 million tokens across various LLM families and scales. The perplexity remains stable throughout. We use the concatenated test set of PG19 (100 books) to perform language modeling, with perplexity fluctuations due to book transitions.

<span id="section-4-2"></span>

### 4.2 Results of Pre-Training with a Sink Token

To validate our suggestion that introducing a sink token to all pre-training samples improves streaming LLMs, we trained two language models, each with 160 million parameters, under identical conditions. While one model adhered to the original training settings, the other incorporated a sink token at the start of every training sample. Our experiments employed the Pythia-160M [Bid23] codebase and followed its training recipe. We train the models on an 8xA6000 NVIDIA GPU server using the deduplicated Pile [Gao20] dataset. Apart from reducing the training batch size to 256, we retained all Pythia training configurations, including learning rate schedules, model initialization, and dataset permutations. Both models were trained for 143,000 steps.

<span id="figure-06"></span>

![Pre-training loss curves with and without sink tokens](../../papers/streaming-llm/figure-06.png)

**Figure 6.** Pre-training loss curves of models w/ and w/o sink tokens. Two models have a similar convergence trend.

<span id="table-04"></span>

![Zero-shot accuracy across seven NLP benchmarks](../../papers/streaming-llm/table-04.png)

**Table 4.** Zero-shot accuracy (in %) across 7 NLP benchmarks, including ARC-[Challenge, Easy], HellaSwag, LAMBADA, OpenbookQA, PIQA, and Winogrande. The inclusion of a sink token during pre-training doesn't harm the model performance.

**Convergence and Normal Model Performance.** Including a sink token during pre-training has no negative impact on model convergence and subsequent performance on a range of NLP benchmarks. As depicted in [Figure 6](#figure-06), models trained with a sink token exhibit similar convergence dynamics compared to their vanilla counterparts. We evaluate the two models on seven diverse NLP benchmarks, including ARC-[Challenge, Easy] [Cla18], HellaSwag [Zel19], LAMBADA [Pap16], OpenbookQA [Mih18], PIQA [Bis20], and Winogrande [Sak19]. As shown in [Table 4](#table-04), the model pre-trained with a sink token performs similarly to that trained using the vanilla approach.

**Streaming Performance.** As illustrated in [Table 3](#table-03), the streaming perplexities differ between models trained using traditional methods and those augmented with a sink token. Remarkably, the vanilla model requires the addition of multiple tokens as attention sinks to maintain stable streaming perplexity. In contrast, the model trained with a sink token achieves satisfactory streaming performance using just the sink token.

<span id="figure-07"></span>

![Attention maps for models pre-trained without and with a sink token](../../papers/streaming-llm/figure-07.png)

**Figure 7.** Visualization of average attention logits over 256 sentences, each 16 tokens long, comparing models pre-trained without (left) and with (right) a sink token. Both maps show the same layers and heads. Key observations: (1) Without a sink token, models show local attention in lower layers and increased attention to initial tokens in deeper layers. (2) With a sink token, there is clear attention directed at it across all layers, effectively collecting redundant attention. (3) With the presence of the sink token, less attention is given to other initial tokens, supporting the benefit of designating the sink token to enhance the streaming performance.

<span id="figure-08"></span>

![The first sample in StreamEval](../../papers/streaming-llm/figure-08.png)

**Figure 8.** The first sample in StreamEval.

**Attention Visualization.** [Figure 7](#figure-07) contrasts attention maps for models pre-trained with and without a sink token. The model without the sink token, similar to Llama-2-7B ([Figure 2](#figure-02)), shows early-layer local attention and deeper-layer focus on initial tokens. In contrast, models trained with a sink token consistently concentrate on the sink across layers and heads, indicating an effective attention offloading mechanism. This strong focus on the sink, with reduced attention to other initial tokens, explains the sink token's efficacy in enhancing model's streaming performance.

<span id="table-05"></span>

![Accuracy on the ARC Easy and Challenge datasets](../../papers/streaming-llm/table-05.png)

**Table 5.** Accuracy (in %) on the ARC-[Easy, Challenge] datasets. Questions were concatenated and answered in a streaming manner to mimic a real-world chat setting. The dense baseline fails due to Out-of-Memory (OOM) errors. Window attention has poor accuracy. StreamingLLM has comparable results with the one-shot sample-by-sample baseline. Window attention and StreamingLLM use cache sizes of 1024.

<span id="section-4-3"></span>

### 4.3 Results on Streaming Question Answering with Instruction-tuned Models

To show StreamingLLM's real-world applicability, we emulate multi-round question-answering using instruction-tuned LLMs, commonly used in real-world scenarios.

We first concatenate all question-answer pairs from the ARC-[Challenge, Easy] datasets, feed the continuous stream to Llama-2-[7,13,70]B-Chat models, and assess model completions at each answer position using an exact match criterion. As [table 5](#table-05) indicates, dense attention results in Out-of-Memory (OOM) errors, showing it unsuitable for this setting. While the window attention method works efficiently, it exhibits low accuracy due to random outputs when the input length exceeds the cache size. Conversely, StreamingLLM excels by efficiently handling the streaming format, aligning with the one-shot, sample-by-sample baseline accuracy.

Highlighting a more fitting scenario for StreamingLLM, we introduce a dataset, StreamEval, inspired by the LongEval [Li23h] benchmark. As depicted in [Figure 8](#figure-08), diverging from LongEval's single query over a long-span setup, we query the model every 10 lines of new information. Each query's answer is consistently 20 lines prior, reflecting real-world instances where questions typically pertain to recent information. As illustrated in [Figure 9](#figure-09), LLMs employing StreamingLLM maintain reasonable accuracy even as input lengths approach 120K tokens. In contrast, both dense and window attention fail at the pre-training text length and the KV cache size, respectively. Additionally, we utilize two context-extended models, LongChat-7b-v1.5-32k [Li23h] and Llama-2-7B-32K-Instruct [Tog23], to show that StreamingLLM can complement context extension techniques. Within StreamingLLM, context extension means broadening the maximum cache size of streaming LLMs, enabling the capture of broader local information.

<span id="figure-09"></span>

![Performance on the StreamEval benchmark](../../papers/streaming-llm/figure-09.png)

**Figure 9.** Performance on the StreamEval benchmark. Accuracies are averaged over 100 samples.

<span id="section-4-4"></span>

### 4.4 Ablation Studies

**Numbers of Initial Tokens.** In [Table 2](#table-02), we ablate the effect of adding varying numbers of initial tokens with recent tokens on the streaming perplexity. The results show the insufficiency of introducing merely one or two initial tokens, whereas a threshold of four initial tokens appears enough, with subsequent additions contributing marginal effects. This result justifies our choice of introducing 4 initial tokens as attention sinks in StreamingLLM.

**Cache Sizes.** In [Table 6](#table-06), we evaluate cache size's impact on StreamingLLM's perplexity. Contrary to intuition, increasing the cache size doesn't consistently lower the language modeling perplexity. This inconsistency shows a potential limitation where these models might not maximize the utility of the entire context they receive. Future research efforts should target enhancing these models' capabilities to utilize extensive contexts better.

<span id="table-06"></span>

![Effects of cache size on StreamingLLM performance](../../papers/streaming-llm/table-06.png)

**Table 6.** Effects of cache size on StreamingLLM's performance. Increasing the cache size in StreamingLLM doesn't consistently yield a decrease in perplexity, showing these models may not fully utilize the provided context. Cache config $x$+$y$ denotes adding $x$ initial tokens with $y$ recent tokens. Perplexity is evaluated on 400K tokens in the concatenated PG19 test set.

<span id="section-4-5"></span>

### 4.5 Efficency Results

<span id="figure-10"></span>

![Per-token decoding latency and memory usage](../../papers/streaming-llm/figure-10.png)

**Figure 10.** Comparison of per-token decoding latency and memory usage between the sliding window approach with re-computation baseline and StreamingLLM, plotted against the cache size (attention window size) on the X-axis. StreamingLLM delivers a remarkable speedup of up to 22.2$\times$ per token and retains a memory footprint similar to the re-computation baseline.

We benchmark StreamingLLM's decoding latency and memory usage against the sliding window with re-computation, which is the only baseline with acceptable quality. Both methods are implemented using the Huggingface Transformers library [Wol20] and tested on a single NVIDIA A6000 GPU using the Llama-2-7B and Llama-2-13B models. As shown in [Figure 10](#figure-10), as the cache size increases, StreamingLLM's decoding speed has a linear growth. The sliding window with re-computation baseline has a quadratic rise in decoding latency. Thus, StreamingLLM achieves an impressive speedup, reaching up to 22.2$\times$ per token. Despite its reduced latency, StreamingLLM sustains a memory footprint consistent with the re-computation baseline.

<span id="section-5"></span>

## 5 Conclusion

Deploying LLMs in streaming applications is urgently needed but comes with challenges due to efficiency limitations and reduced performance with longer texts. Window attention provides a partial solution, but its performance plummets when initial tokens are excluded. Recognizing the role of these tokens as “attention sinks”, we introduced StreamingLLM—a simple and efficient framework that enables LLMs to handle unlimited texts without fine-tuning. By adding attention sinks with recent tokens, StreamingLLM can efficiently model texts of up to 4 million tokens. We further show that pre-training models with a dedicated sink token can improve the streaming performance. StreamingLLM firstly decouples the LLM's pre-training window size and its actual text generation length, paving the way for the streaming deployment of LLMs.

**Reproducibility Statement.** All findings presented in this paper can be reproduced. We have made our code and datasets available in this [github repo](https://github.com/mit-han-lab/streaming-llm). The models used in this paper are all openly available, and we provide references to access them. Details regarding our experiments, including hyperparameters, training protocols, and evaluation methods, can be found in the Experiments section ([Section 4](#section-4)). We are confident that with the provided resources, readers can reproduce the entirety of our presented results.

**Impact Statement.** StreamingLLM has been widely adopted by various LLM serving solutions including [NVIDIA TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#run-llama-with-streamingllm), [Intel Extension for Transformers](https://github.com/intel/intel-extension-for-transformers), [HuggingFace Transformers](https://huggingface.co/docs/transformers/v4.39.3/en/internal/generation_utils#transformers.SinkCache), [MLC LLM](https://github.com/mlc-ai/mlc-llm/pull/1459), etc.

## Acknowledgements

This work is supported by MIT-IBM Watson AI Lab, Amazon and MIT Science Hub, National Science Foundation. We thank Angela Li for writing suggestions and demo making, Jingwei Zuo for proofreading, and Xiuyu Li for the suggestion on notations.

<span id="section-6"></span>

## 6 Discussions

**Applications.** StreamingLLM is particularly suited for streaming applications, such as multi-round dialogues, where continuous operation without heavy reliance on extensive memory or historical data is crucial. For instance, in a daily assistant application based on LLMs, StreamingLLM enables the model to function seamlessly over extended periods. It bases its responses on recent interactions, thus avoiding the need for frequent cache refreshes. Traditional methods might require resetting the cache when the conversation length surpasses the training length, leading to a loss of recent context, or they might need to recompute key-value (KV) states from recent text history, which can be inefficient.

**Limitations.** While StreamingLLM improves the efficiency of LLMs in streaming contexts, it does not extend the models' context window or enhance their long-term memory capabilities. As detailed in [Section 8](#section-8), the model is limited to operating within the confines of its current cache. Consequently, StreamingLLM is not suitable for tasks that demand long-term memory and extensive data dependency, such as long document question-answering (QA) and summarization. However, it excels in scenarios only requiring short-term memory, like daily conversations and short document QA, where its strength lies in generating coherent text from recent context without the need for cache refreshment.

**Broader Societal Impacts.** StreamingLLM significantly enhances the efficiency and accessibility of LLMs, democratizing their use across various sectors. By enabling nonstop and rapid interactions in applications like conversational agents, StreamingLLM improves user experiences, especially in scenarios requiring fixed-length models. This advancement allows for more seamless and contextually aware dialogues, potentially benefiting sectors like education, healthcare, and customer service. Additionally, StreamingLLM's efficiency in processing reduces the computational load, aligning with the need for environmentally sustainable AI technologies. This aspect is crucial in making advanced AI tools more accessible in regions with limited technological resources. However, the potential negative impacts of StreamingLLM mirror those associated with general language models, such as misinformation and biased content generation risks. It's essential to address these risks with robust ethical guidelines and safeguards. In summary, while StreamingLLM shares some risks common to language models, its positive contributions towards enhancing user experience, democratizing AI access, and promoting sustainability are noteworthy. These benefits underscore the importance of responsible deployment and ethical use of this technology.

<span id="section-7"></span>

## 7 Additional Related Works

**Sparse Transformers.** The literature on efficient Transformer models primarily focuses on reducing the computational and memory complexity of the self-attention mechanism. A relevant line of work involves sparsifying the attention matrix by restricting the field of view to fixed, predefined patterns, such as local windows or block patterns with fixed strides [Tay22a]. Sparse Transformer [Chi19] introduces sparse factorizations of the attention matrix, reducing the computational complexity of attention to $O(n\sqrt{n})$. LongFormer [Bel20] combines dilated local windowed attention with task-motivated global attention. Extended Transformer Construction (ETC) [Ain20] presents a novel global-local attention mechanism, incorporating four types of attention patterns: global-to-global, local-to-local, local-to-global, and global-to-local. Building on ETC, BigBird [Zah20] proposes another linear complexity attention alternative, utilizing global tokens, local sliding window attentions, and random attention. However, these methods have several limitations. First, Sparse Transformer and ETC require custom GPU kernels for a specific block-sparse variant of matrix-matrix multiplication. Second, LongFormer, ETC, and BigBird all rely on a global attention pattern, which is unsuitable for autoregressive language models. Third, these methods are incompatible with pre-trained models, necessitating retraining from scratch. In contrast, our method offers ease of implementation using standard GPU kernels and is compatible with pre-trained autoregressive language models using dense attention, which are prevalent in the NLP community. This compatibility provides a significant advantage, allowing for the leveraging of existing pre-trained models without any fine-tuning.

**Concurrent Works.** Our research coincides with the work of Han et al. [Han23x], who conducted a theoretical study on the length generalization failure of language models, identifying three out-of-distribution factors. Their approach, inspired by this analysis, involves employing a “$\Lambda$”-shaped attention pattern and reconfiguring position encoding distances to enhance length generalization in LLMs. This approach bears a resemblance to our methodology. However, our work uncovers the “attention sink” phenomenon, wherein Transformer models tend to assign high attention scores to initial tokens with small semantics. This phenomenon extends beyond the scope of length generalization failure, indicating a more pervasive issue in Transformer models. We observe this “attention sink” behavior not only in auto-regressive language models but also in encoder Transformers such as BERT (see [Section 13](#section-13)), and Vision Transformers (ViTs) [Dar23a], suggesting its broader prevalence in Transformer architectures. To mitigate the “attention sink” phenomenon, we propose the introduction of a learnable sink token during pre-training, and we support our findings with extensive ablation studies.

In parallel, Darcet et al. [Dar23a] observed similar attention concentration on random background patch tokens in Vision Transformers, termed as “registers.” These registers act as repositories for global image information. Their solution, adding dedicated “register” tokens, aims to balance attention distribution. Our finding of “attention sinks” parallels this concept. In our paper, the “attention sinks” are initial tokens that disproportionately attract attention from subsequent tokens. Introducing a dedicated sink token during pre-training prevents the model from inappropriately using content tokens as attention sinks, leading to more effective attention distribution. However, a key difference exists: “registers” in Vision Transformers function as global information holders within intermediate layers, whereas our “attention sinks” are positioned as initial tokens in autoregressive models. This positional variance suggests that the softmax function in attention computation might play a more fundamental role in the emergence of attention sinks.

<span id="section-8"></span>

## 8 Accuracy on StreamEval with Increasing Query-Answer Line Distance

<span id="table-07"></span>

![StreamEval accuracy with increasing query-answer distance](../../papers/streaming-llm/table-07.png)

**Table 7.** Accuracy (in %) on StreamEval with increasing query-answer distance. Each line in StreamEval contains 23 tokens. Accuracies are averaged over 100 samples, and each sample contains 100 queries.

To assess StreamingLLM's handling of extended inputs, we evaluated the Llama-2-7B-32K-Instruct model on StreamEval, focusing on different query-answer line distances under various cache configurations. In StreamEval, each line consists of 23 tokens, making the line distances equivalent to token distances of $23 \times \mathrm{line\ distances}$. Accuracy was calculated by averaging results over 100 samples, with each sample comprising 100 queries. [Table 7](#table-07) illustrates that StreamingLLM retains accuracy when the token distance between the query and answer is within the cache size. However, accuracy diminishes as this distance increases and eventually drops to zero when it surpasses the cache capacity.

These results demonstrate that while StreamingLLM is effective in generating coherent text based on recent context, it cannot extend the context length of language models. These results also emphasize a broader challenge in current language models: their inability to fully utilize context information within the cache, a finding that aligns with the observations made by Liu et al. [Liu23z].

<span id="section-9"></span>

## 9 Long-Range Benchmark Evaluation

<span id="table-08"></span>

![StreamingLLM and truncation performance on LongBench](../../papers/streaming-llm/table-08.png)

**Table 8.** Performance comparison of StreamingLLM against the default truncation baseline in LongBench [Bai23]. The baseline truncates inputs to 1750 initial and 1750 final tokens. StreamingLLM 4+3496 uses 4 attention sink tokens and 3496 recent tokens, while StreamingLLM 1750+1750 uses 1750 tokens for both initial and recent segments.

We evaluated StreamingLLM using the Llama-2-7B-chat model (max context length 4k) on LongBench [Bai23], which encompasses three key NLP tasks: single-document QA (NarrativeQA [Koi18] and Qasper [Das21]), multi-document QA (HotpotQA [Yan18a] and 2WikiMQA [Ho20]), and summarization (GovReport [Hua21], MultiNews [Fab19]). LongBench sets a default max sequence length of 3,500 tokens for the Llama-2-7B-chat model, truncating from the middle to preserve beginning and end information (1,750 tokens each). [Table 8](#table-08) shows that StreamingLLM with a 4+3496 cache configuration underperforms compared to the truncation baseline, likely due to the loss of crucial initial input prompt information. However, aligning the attention sink number to 1750 restores performance to the level of the text truncation baseline. These results corroborate the findings in [Section 8](#section-8), demonstrating that StreamingLLM's effectiveness is contingent on the information within its cache, with in-cache performance comparable to the text truncation baseline.

<span id="section-10"></span>

## 10 Llama-2-7B Attention Visualization on Longer Sequences

<span id="figure-11"></span>

![Average Llama-2-7B attention logits on 128-token sequences](../../papers/streaming-llm/figure-11.png)

**Figure 11.** Visualization of the *average* attention logits in Llama-2-7B over 256 sentences, each with a length of 128.

[Figure 2](#figure-02) visualizes the attention map of Llama-2-7B using short sequences (length of 16) for clarity. We further visualize the attention of Llama-2-7B on longer sequences (length of 128) in [Figure 11](#figure-11). We find the observations on short sequences also hold on longer sequences, where the attention scores of the initial tokens are much higher than the rest of the tokens in most layers, regardless of the distance between the initial tokens and the tokens in the rest of the sequence. Because the longer the sequence, the thinner the attention sinks' scores are visualized on the heatmap. We further analyze the attention distribution on longer sequences (length of 4096) using a different method in [Section 11](#section-11).

<span id="section-11"></span>

## 11 Quatitative Analysis of Attention Sinks in Long Inputs

<span id="figure-12"></span>

![Attention scores on the first token across Llama-2-7B layers](../../papers/streaming-llm/figure-12.png)

**Figure 12.** Visualization of attention scores (after SoftMax) on the first token across layers in Llama-2-7B. Attention Scores are the 4096th token's attention towards the first token in each layer. The error bars are the standard deviation of the first token's attention scores across different heads in one layer. Results are averaged over 256 sentences, each having a length of 4096 tokens.

[Figures 2](#figure-02) and [13](#figure-13) illustrate the attention sink phenomenon using short sequences for clarity. Extending this analysis, [Figure 12](#figure-12) demonstrates the distribution of attention scores (after SoftMax) towards the first token in lengthy inputs (sequence length of 4096). We average attention scores across 256 sequences, with each sequence comprising 4096 tokens. The plotted data represent the attention allocated by the 4096th token to the initial token in every layer. Notably, the attention scores for the first token are significantly high, often exceeding half of the total attention, except for the two bottom layers. This observation empirically substantiates the preferential focus on the first token by the majority of layers and heads, irrespective of other tokens' distances within the sequence. Such a trend underscores the critical role of the initial tokens in a sequence, as their removal has a huge impact on language model performance due to a large portion of the denominator in the SoftMax function being removed.

<span id="section-12"></span>

## 12 Llama-2-70B Attention Visualization

<span id="figure-13"></span>

![Average attention logits in Llama-2-70B](../../papers/streaming-llm/figure-13.png)

**Figure 13.** Visualization of the *average* attention logits in Llama-2-70B over 256 sentences, each with a length of 16.

[Figure 2](#figure-02) shows the attention visualization of Llama-2-7B, we further visualize the attention of Llama-2-70B in [Figure 13](#figure-13). We find the observation on Llama-2-7B also holds on Llama-2-70B, where the attention scores of the initial tokens are much higher than the rest of the tokens in most layers.

<span id="section-13"></span>

## 13 Attention Sinks in Encoder Transformers

<span id="figure-14"></span>

![Attention maps in BERT-base-uncased](../../papers/streaming-llm/figure-14.png)

**Figure 14.** Visualization of attention maps for sentence *“StreamingLLM can work on infinite-length texts without compromising efficiency and performance.”* in BERT-base-uncased.

In this paper, we mainly explore the attention sink phenomenon observed in autoregressive, decoder-only language models like GPT and Llama. Building upon the insights from [Section 3.1](#section-3-1), we propose that this phenomenon likely extends to other Transformer architectures, including encoder models such as BERT [Dev19] and ViT [Dos20]. This assumption stems from the fact that these models share a similar Transformer structure and utilize SoftMax attention mechanisms. To substantiate our hypothesis, we analyze the attention patterns of BERT-base-uncased, as depicted in [Figure 14](#figure-14). Our findings reveal that BERT-base-uncased exhibits the attention sink phenomenon, characterized by disproportionately high attention scores assigned to the `[SEP]` token in most layers. This indicates that the model consistently relies on the omnipresent `[SEP]` token as a focal point for attention. Furthermore, concurrent research by Darcet et al. [Dar23a] identifies similar attention spikes in Vision Transformers, attributed to random background patch tokens acting as “registers” for global image information. We contend that these “registers” are analogous to the attention sink phenomenon we observed, suggesting that this is a universal characteristic across all Transformer models.

<span id="section-14"></span>

## 14 Using More Sink Tokens in the Pre-Training Stage

[Section 3.3](#section-3-3) illustrated that incorporating a single dedicated sink token in the pre-training stage doesn't affect model performance but enhances streaming performance by centralizing attention sinks to one token. This section delves into whether adding additional sink tokens during pre-training could further optimize the performance of pre-trained language models.

As depicted in [Figure 15](#figure-15), our experiments show that incorporating either one or two sink tokens during pre-training results in pre-training loss curves that closely resemble those of the baseline (vanilla) model. However, as detailed in [Table 9](#table-09), the introduction of a second sink token does not yield substantial improvements in performance across most benchmark tasks.

Further analysis, as shown in [Table 10](#table-10), reveals that the inclusion of additional sink tokens does not enhance streaming performance. Interestingly, the model appears to rely on both sink tokens to maintain stable streaming performance. These findings suggest that while a single sink token is adequate for improving streaming performance, adding more sink tokens does not lead to further enhancements in overall language model performance. This contrasts with findings in Vision Transformers (ViT) [Dar23a], where multiple “registers” have been found to be beneficial.

<span id="figure-15"></span>

![Pre-training loss curves with zero, one, and two sink tokens](../../papers/streaming-llm/figure-15.png)

**Figure 15.** Pre-training loss curves of models with 0, 1, and 2 sink tokens.

<span id="table-09"></span>

![Zero-shot accuracy with zero, one, and two sink tokens](../../papers/streaming-llm/table-09.png)

**Table 9.** Zero-shot accuracy (in %) across 7 NLP benchmarks, including ARC-[Challenge, Easy], HellaSwag, LAMBADA, OpenbookQA, PIQA, and Winogrande.

<span id="table-10"></span>

![Streaming perplexity with zero, one, and two sink tokens](../../papers/streaming-llm/table-10.png)

**Table 10.** Comparison of vanilla attention with prepending a zero token and a learnable sink token during pre-training. Cache config $x$+$y$ denotes adding $x$ initial tokens with $y$ recent tokens. Perplexity is evaluated on the first sample in the PG19 test set.

[+1]: Part of the work done during an internship at Meta AI.
