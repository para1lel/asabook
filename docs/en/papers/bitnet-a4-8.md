---
title: 'BitNet a4.8: 4-bit Activations for 1-bit LLMs'
createTime: 2026/09/08 16:34:10
permalink: /en/papers/bitnet-a4-8/
pageClass: paper-reading
---

> [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Shuming Ma](https://shumingma.com/) [+author-note], and [Furu Wei](https://thegenerality.com/)◇. [GeneralAI](https://aka.ms/GeneralAI). First submitted to arXiv on November 7, 2024; current version v1; work in progress. [BitNet a4.8: 4-bit Activations for 1-bit LLMs](https://arxiv.org/abs/2411.04965). <a href="/paper/bitnet-a4-8.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2411.04965). [TeX source](https://export.arxiv.org/e-print/2411.04965v1). The original PDF remains authoritative for the exact print layout and bibliography.

[+author-note]: Equal contribution. ◇ Corresponding author. S. Ma and F. Wei are with Microsoft Research. H. Wang is with University of Chinese Academy of Sciences.

## Abstract

Recent research on the 1-bit Large Language Models (LLMs), such as BitNet b1.58 [Ma24], presents a promising direction for reducing the inference cost of LLMs while maintaining their performance. In this work, we introduce **BitNet a4.8**, enabling 4-bit activations for 1-bit LLMs. BitNet a4.8 employs a hybrid quantization and sparsification strategy to mitigate the quantization errors introduced by the outlier channels. Specifically, we utilize 4-bit activations for inputs to the attention and feed-forward network layers, while sparsifying intermediate states followed with 8-bit quantization. Extensive experiments demonstrate that BitNet a4.8 achieves performance comparable to BitNet b1.58 with equivalent training costs, while being faster in inference with enabling 4-bit (INT4/FP4) kernels. Additionally, BitNet a4.8 activates only 55% of parameters and supports 3-bit KV cache, further enhancing the efficiency of large-scale LLM deployment and inference.

<span id="figure-01"></span>

![Overview of BitNet a4.8 weight and activation quantization](../../papers/bitnet-a4-8/figure-01.png)

**Figure 1.** The overview of BitNet a4.8 with both weight and activation quantization. All the parameters are ternery (i.e., 1.58-bit as in BitNet b1.58 [Ma24]). We use a hybrid quantization and sparsification strategy to deal with outlier activations in certain Transformer sub-layers.

<span id="section-1"></span>

## 1 Introduction

Recent works [Ma24] have shown that 1-bit LLMs can match the full-precision models given the same amount of parameters and training tokens while being significantly cost-effective in terms of latency, memory, throughput, and energy consumption. With model weights represented in 1.58-bit (i.e., $\{-1, 0, 1\}$), the bottleneck of inference has shifted from the limited memory bandwidth to high computational cost. Low-bit or sparse activations in LLMs served as a promising approach to further reduce the computational budget while maintaining performance on downstream tasks.

One common approach is to utilize activation sparsity [Liu23t, Son24c, Liu24aa], which reduces the inference FLOPs and the I/O of weight by pruning the activation entries with smaller magnitudes. Sparsification is particularly well-suited for handling activations that exhibit highly imbalanced long-tailed distributions. Recent works [Wan24ag] have demonstrated that LLMs with fully sparsely-activated activations can achieve results comparable to dense models while having much less active parameters.

In addition to sparsification, activation quantization is another approach to accelerate the matrix multiplication. However, the optimization of neural networks with low-bit activations is challenging due to the emergence of outlier dimensions as the training progresses and the model size grows. Despite these outliers only account for a very small portion of the activations [Det22, Xia23], they have much larger magnitude, which leads to significant quantization errors and performance degradation on downstream tasks. Previous works [Xi23, Ash24, Liu24b, Lin24b] mostly utilize Hadamard or learnable rotary transformation to amortize the outlier features into other entries. However, they are mostly designed for the LLMs of higher precision (e.g., 4-bit). For 1-bit LLMs, the extremely low bit-width of the weights makes it challenging to absorb these transformation matrices directly into the weights, while leaving them as online transformations introduces additional computational overhead and limits overall inference performance.

In this work, we introduce **BitNet a4.8**, a hybrid quantization and sparsification strategy that enables 4-bit activations for 1-bit LLMs. By carefully analyzing the activation distribution of 1-bit LLMs, we selectively apply 4-bit quantization or sparsification based on the distribution patterns of these activations. Specifically, as shown in [Figure 1](#figure-01), BitNet a4.8 employs 4-bit activations for the inputs to attention and FFN, while utilizing sparsification with 8 bits for intermediate states. To improve the training efficiency, BitNet a4.8 is trained from 8-bit to 4-bit activations with a two-stage recipe, which requires only a few training tokens to adapt BitNet b1.58 to the low-bit activations at the end of training. Extensive experiments demonstrate that BitNet a4.8 achieves competitive performance to BitNet b1.58 with the same training cost while being significantly more efficient at inference time. Furthermore, BitNet a4.8 has only 55% activated parameters and supports 3-bit KV cache, which further enhances the efficiency of LLM deployment.

<span id="section-2"></span>

## 2 BitNet a4.8

<span id="section-2-1"></span>

### 2.1 Architecture

As shown in [Figure 1](#figure-01), BitNet a4.8 adopts the same layout as BitNet b1.58. Following [Wan23, Ma24], we replace the linear projections in both attention and feed-forward network (FFN) with BitLinear to learn 1.58-bit weights from the scratch. For activations, we adopt a hybrid quantization and sparsification strategy to mitigate the error introduced by outlier dimensions.

[Figure 2](#figure-02) illustrates the distribution of each component's inputs of a BitNet b1.58 model with 7B model size. Inputs to the attention and FFN layers typically follow a Gaussian-like distribution, while activations before the FFN down projections and the output projections in attention have more outlier channels and massive amount of entries around zero. [Liu24aa] also reported similar observations for full-precision LLMs. As shown in [Figure 3](#figure-03), directly applying low-bit quantization to these intermediate states introduces substantial quantization errors.

<span id="figure-02"></span>

![Activation distributions at the inputs to each projection](../../papers/bitnet-a4-8/figure-02.png)

**Figure 2.** The distribution of the inputs to each projection. The visualization is conducted with a 7B BitNet b1.58 model on a subset of the valid set of C4. For the layers that exhibit Gaussian-like distributions, we employ 4-bit activation quantization. For the layers which distributions are sharp, we adopt Q-Sparse [Wan24ag] to perform sparsification on the activations.

Therefore, we use the sparsification method from Q-Sparse [Wan24ag] to retain these intermediate states at 8 bits while removing the computation bottleneck. For output projection of self-attention layers, we use a sparsify-then-quantize function:

<span id="equation-01"></span>

$$
\mathbf{Y} = \left(\mathrm{Q}_{\mathrm{INT}8}(\mathbf{X}) \odot \mathbf{M}\right) \cdot \mathrm{Q}_{w}(\mathbf{W})^\top, \quad \mathbf{M} = \mathrm{Top}_k\left(|\mathbf{X}|\right)
$$

where $\mathrm{Q}_{w}(\cdot)$ and $\mathrm{Q}_{\mathrm{INT}8}(\cdot)$ denote the quantization function for weight $\mathbf{W}$ and activations $\mathbf{X}$, respectively. $\mathbf{M}$ is the mask tensor that indicates the maximum top-K elements in terms of the absolute values of the activations $\mathbf{X}$, $\odot$ is the element-wise multiplication operation.

Specifically, the functions of weight quantization and activation quantization can be formulated as:

<span id="equation-02"></span>

$$
\mathrm{Q}_{w}(\mathbf{W}) = \alpha\mathrm{RoundClip}\left(\frac{\mathbf{W}}{\alpha+\epsilon}, -1, 1\right), \quad \alpha = \mathrm{mean}(|\mathbf{W}|)
$$

<span id="equation-03"></span>

$$
\mathrm{Q}_{\mathrm{INT}8}(\mathbf{X}) = \frac{\gamma}{127}\mathrm{RoundClip}\left(\frac{127}{\gamma+\epsilon}\mathbf{X}, -128, 127\right), \quad \gamma = \max(|\mathbf{X}|)
$$

<span id="equation-04"></span>

$$
\mathrm{RoundClip}(X, a, b) = \min\left(\max(\mathrm{round}(X), a), b\right)
$$

For FFN, we adopt squared ReLU [So21, Wan24ag] and gated linear unit (GLU) to further boost the activation sparsity. It is defined as follows:

<span id="equation-05"></span>

$$
\mathrm{ReLU}^2\mathrm{GLU}(\mathbf{X}) = \mathbf{X}\mathbf{W}_{\mathrm{up}}^\top \odot \mathrm{ReLU}^2\left(\mathbf{X}\mathbf{W}_{\mathrm{gate}}^\top\right)
$$

According to our preliminary experiments, with squared ReLU, the inputs to the down projection achieve over 80% sparsity with minimal impact on performance. Additionally, we observe that the outputs of gate projection $\mathrm{ReLU}^2(\mathbf{X}\mathbf{W}_{\mathrm{gate}}^\top)$ exhibit high activation sparsity as well (e.g., 67.5% for 7B models). This characteristic enables further reduction in inference FLOPs for the up projection by first computing the gate projection and then performing the up projection only on the non-zero channels of the gates.

For the input to attention and FFN, since they have much less outlier features, we use absmean function to quantize the activations to 4-bit integers:

<span id="equation-06"></span>

$$
\mathbf{Y} = \mathrm{Q}_{\mathrm{INT}4}(\mathbf{X}) \cdot \mathrm{Q}_{w}(\mathbf{W})^\top
$$

<span id="equation-07"></span>

$$
\mathrm{Q}_{\mathrm{INT}4}(\mathbf{X}) = \frac{\beta}{\sqrt{7}}\mathrm{RoundClip}\left(\frac{\sqrt{7}}{\beta+\epsilon}\mathbf{X}, -8, 7\right), \quad \beta = \mathrm{mean}(|\mathbf{X}|)
$$

<span id="section-2-2"></span>

### 2.2 Training

**Continue-training from BitNet b1.58.** BitNet a4.8 is trained with a two-stage recipe from W1.58A8 to W1.58A4. For the first stage, we train the model with 8-bit activations and $\mathrm{ReLU}^2\mathrm{GLU}$. For the second stage, we adopt the hybrid quantization and sparsification as shown in [Section 2.1](#section-2-1). BitNet a4.8 quickly adapts to 4-bit and sparse activations with only a few training tokens while having negligible loss on performance.

**Gradient approximation.** Following [Wan23, Wan24ag], we use straight-through estimator (STE) [Ben13] to conduct the gradient approximation for BitNet a4.8, as well as mixed precision training to update the parameters. We directly bypass the non-differentiable functions, including the quantization function and top-K sparsification function during the backward propagation. For mixed precision training, we maintain a full-precision latent weight to accumulate parameter updates. During the forward, we quantize the latent weight into 1.58-bit on the fly.

<span id="figure-03"></span>

![Output-projection input distributions under quantization and sparsification](../../papers/bitnet-a4-8/figure-03.png)

**Figure 3.** The distribution of the inputs to the output projection of attention with different quantization and sparsification. The visualization is conducted with a 7B BitNet b1.58 model on a subset of the valid set of C4.

<span id="section-2-3"></span>

### 2.3 Floating-point quantization

Floating-point quantization offers a broader dynamic range than the integer-based quantization, which is crucial for handling the long-tailed distribution of the activations. For floating-point precision, we only leave the inputs to down projection of FFN at 8-bit integers, and quantize the other activations to FP4 using MinMax quantizer [Liu23c]. It is defined as follows:

<span id="equation-08"></span>

$$
\mathrm{Q}_{\mathrm{FP}4}(\mathbf{X}) = \frac{\gamma}{2^{M+b}}\mathrm{Round}\left(\frac{2^{M+b}}{\gamma}\mathbf{X}\right), \quad \gamma = 2^{\max\left(\left\lfloor\left\lfloor\log_2|\mathbf{X}|\right\rfloor+b\right\rfloor,1\right)}
$$

<span id="equation-09"></span>

$$
b = \log_2\left(\frac{2-2^{-M}}{|\mathbf{X}|_{\max}}\right) + 2^E - 1
$$

where $E$ and $M$ denote the bit-width of the exponent and mantissa component, respectively. We adopt the E2M1 format due to its larger dynamic range. As shown in [Table 1](#table-01), BitNet a4.8 with FP4 quantization has the similar performance as it with the hybrid quantization and sparsification strategy based on integers.

<span id="table-01"></span>

![Perplexity and end-task results for BitNet a4.8, BitNet b1.58, and LLaMA LLM](../../papers/bitnet-a4-8/table-01.png)

**Table 1.** Perplexity and results of BitNet a4.8, BitNet b1.58 and LLaMA LLM on the end tasks. The standard variance of error for average scores is 1.06%.

<span id="section-3"></span>

## 3 Experiments

We compared BitNet a4.8 to BitNet b1.58 and our reproduced FP16 LLaMA LLM of various sizes. For 1.58-bit models, we adopted the two-stage weight decay and learning rate scheduling following the training recipe of BitNet b1.58 [Ma24]. More details can be found in [Section 5](#section-5). All models were trained with 100B tokens from the RedPajama dataset [Tog23a] to ensure a fair comparison. For BitNet a4.8, we first train the model with 8-bit activations for 95B tokens. Then we reuse the optimizer states and continue-train the model with the proposed hybrid quantization and sparsification for 5B tokens. We set topK as 50% for the output projection of attention.

We evaluated the zero-shot accuracy for these models on a range of language tasks using the *lm-evaluation-harness* toolkit [Gao24h], including ARC-Easy (ARCe) [Yad19], ARC-Challenge (ARCc) [Yad19], Hellaswag (HS) [Zel19], Winogrande (WGe) [Sak20] and PIQA (PQ) [Bis20]. We also reported the perplexity on the validation set of C4 [Raf19] dataset.

<span id="section-3-1"></span>

### 3.1 Main Results

[Table 1](#table-01) summarizes the detailed results of BitNet a4.8, BitNet b1.58 and FP16 LLaMA LLM. The performance gap between full-precision (i.e., FP16) LLaMA LLM and BitNet b1.58 narrows as the model size grows. For 7B models, BitNet b1.58 matches LLaMA LLM in terms of both language model perplexity and average accuracy on the end tasks. Furthermore, BitNet a4.8 achieves performance comparable to BitNet b1.58, with almost no loss in average accuracy.

<span id="table-02"></span>

![Detailed component sparsity for BitNet a4.8, BitNet b1.58, and LLaMA LLM](../../papers/bitnet-a4-8/table-02.png)

**Table 2.** Detailed sparsity of BitNet a4.8, BitNet b1.58 and LLaMA LLM on the valid set of C4.

**Sparsity.** [Table 2](#table-02) demonstrates detailed sparsity of each component for BitNet a4.8, BitNet b1.58 and FP16 LLaMA LLM across various sizes. The sparsity is calculated with non-embedding parameters on the valid set of C4. Notably, BitNet a4.8 achieves significantly higher sparsity than both BitNet b1.58 and LLaMA LLM. For example, in the 7B model, BitNet a4.8 reaches an overall sparsity of 44.5%, with only 3.4B active parameters. The inputs to the down projection demonstrate particularly high sparsity, consistent with our observation that intermediate state distributions are sharply centered around zero. Additionally, we also observe that the outputs of gate projection are very sparse. It leads to a high sparsity for up projection, since we only need to perform projection on the non-zero channels selected from the gates. Specifically, for the 7B BitNet a4.8, the sparsity of the gates and the inputs to up projection is 67.5% and 12.0%, respectively. Consequently, the sparsity of up projection can be estimated as $1 - (1 - 12.0\%)\times(1 - 67.5\%)$, that is 71.4%.

<span id="table-03"></span>

![Low-bit QKV results for BitNet a4.8 at 3B and 7B sizes](../../papers/bitnet-a4-8/table-03.png)

**Table 3.** Detailed results of BitNet a4.8 with QKV states varying bit-widths on the end tasks. We reported the zero-shot accuracy of all models.

**Low-bit Attention.** [Table 3](#table-03) presented detailed results of BitNet a4.8 with low-bit attention in 3B and 7B model size. Low-bit attention is essential for efficient long sequence modeling, as it reduces the memory footprint and IO of KV cache and accelerates the attention computation. In our experiments, we adopted post-RoPE quantization. The QKV heads were directly quantized to unsigned integers using the absmax function, without the need of any calibration dataset. For 3-bit KV quantization, we retain the heads of the bos token at 4-bit, as it contains more outlier features. As shown in [Table 3](#table-03), BitNet a4.8 achieves negligible accuracy loss with 4-bit KV or QKV heads in 3B and 7B models. Furthermore, the KV cache of BitNet a4.8 can be quantized to 3-bit integers, resulting in almost no degradation on average accuracy.

<span id="section-3-2"></span>

### 3.2 Ablation Study

<span id="figure-04"></span>

![Hybrid quantization and sparsification ablation loss curves](../../papers/bitnet-a4-8/figure-04.png)

**Figure 4.** Ablation study on the hybrid quantization and sparsification.

<span id="figure-05"></span>

![FFN down-projection quantization and activation-function ablation loss curves](../../papers/bitnet-a4-8/figure-05.png)

**Figure 5.** Ablation study on different quantization or activation function for the inputs to down projection of FFN.

<span id="table-04"></span>

![TopK sparsification ablation for attention output-projection inputs](../../papers/bitnet-a4-8/table-04.png)

**Table 4.** Ablations on the TopK sparsification for the inputs to the output projection of attention.

**Hybrid architecture.** [Figure 4](#figure-04) presented the training loss curve of 700M BitNet a4.8 with the full INT4/FP4 quantization, and the hybrid quantization and sparsification. We train these models with the first-stage scheduling for 25B tokens from the RedPajama dataset. We adopt absmean and MinMax quantizer for full INT4 and FP4 quantization, respectively. Besides, for full INT4 quantization, we use absmean quantizer with $\beta = 2\mathrm{mean}(|X|)$ for down projection in FFN, as its inputs have larger outliers. As shown in [Figure 4](#figure-04), the full INT4 quantization leads to divergence. Furthermore, the hybrid architecture significantly outperforms the full FP4 architecture in terms of training perplexity.

**Down projection of FFN.** We compared 1.3B BitNet a4.8 with different quantization or activation function for the down projection of FFN. All models were trained with the first-stage scheduling for 50B tokens from the RedPajama dataset. To ensure a fair comparison, we leave the other activations at 8-bits. We adopt the absmax quantizer for INT8 quantization and MinMax quantizer for FP4 quantization. The $\beta$ of absmean quantizer is set as $2\mathrm{mean}(|X|)$. [Figure 5](#figure-05) shows the training loss curves of these models. Squared ReLU achieves slightly better training perplexity than Swish while enabling higher sparsity. Furthermore, applying FP4 quantization for the inputs to the down projection leads to a significant performance degradation, while using INT4 activations with STE causes divergence.

<span id="figure-06"></span>

![Loss curves for four-bit quantizers at attention and FFN inputs](../../papers/bitnet-a4-8/figure-06.png)

**Figure 6.** Ablations on 4-bit quantizers for the inputs to attention and FFN.

**Output projection of attention.** [Table 4](#table-04) demonstrates detailed results of 3B BitNet a4.8 with and without Top-K sparsification for the inputs to the output projection of attention. Both models were trained with the same two stage recipe from 8-bit to 4-bit activations. We set $K$ as 50% for sparsification. The baseline utilized the INT8 absmax quantizer for the output projection's inputs. The results show that TopK sparsification brings negligible perplexity and accuracy loss.

**4-bit quantization.** We presented the loss curves of 3B BitNet a4.8 with different 4-bit quantizers for the inputs to the attention and FFN. We compared the performance of BitNet a4.8 with floating-point quantization with E2M1 and E1M2 formats using MinMax quantizer, integer quantization with absmax and absmean quantizer. As shown in [Figure 6](#figure-06), FP4 with E2M1 format and INT4 with absmean quantizer achieve slightly better training perplexity, as they are well-suited for handling small-magnitude activation entries.

<span id="table-05"></span>

![Two-trillion-token results for BitNet a4.8 and BitNet b1.58](../../papers/bitnet-a4-8/table-05.png)

**Table 5.** Results of BitNet a4.8 and BitNet b1.58 with 2B parameters and 2T training tokens.

<span id="section-3-3"></span>

### 3.3 More Training Tokens

Prior research [Det22] demonstrates a positive correlation between the number of training tokens and the prevalence of activation outliers in language models. To rigorously evaluate the scalability characteristics of BitNet a4.8, we conducted extensive experiments using a model configuration with 2 billion parameters trained on 2 trillion tokens. We performed a controlled comparison against BitNet b1.58 using identical training data and configurations. The empirical results, presented in [Table 5](#table-05), demonstrate that BitNet a4.8 maintains performance parity with negligible degradation in accuracy metrics while achieving 4-bit activation compression. These findings provide strong evidence for the efficacy of our proposed approach at scale.

<span id="section-4"></span>

## 4 Conclusion

In this paper, we present BitNet a4.8 which enables 4-bit activations for 1-bit LLMs. BitNet a4.8 uses a novel hybrid quantization and sparsification architecture to reduce the quantization errors introduce by outlier channels of activations. Specifically, we employ 4-bit quantization for inputs to the attention and FFN layers, while sparsifying the intermediate states with 8-bit integers. BitNet a4.8 is continue-trained from W1.58A8 to W1.58A4. Experimental results demonstrate that BitNet a4.8 achieves results comparable to BitNet b1.58 with the same training cost, while significantly enhancing inference efficiency.

## Acknowledgements

We would like to acknowledge Lei Wang for the discussion on the inference efficiency.

<span id="section-5"></span>

## 5 Hyper-parameters

<span id="table-06"></span>

![Model configurations for BitNet a4.8, BitNet b1.58, and LLaMA LLM](../../papers/bitnet-a4-8/table-06.png)

**Table 6.** Model configurations for both BitNet a4.8, BitNet b1.58 and LLaMA LLM.

<span id="table-07"></span>

![Training hyper-parameters for BitNet a4.8 and LLaMA LLM](../../papers/bitnet-a4-8/table-07.png)

**Table 7.** Hyper-parameters for both BitNet a4.8 and LLaMA LLM training.
