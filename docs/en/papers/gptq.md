---
title: 'GPTQ'
createTime: 2026/09/07 00:18:21
permalink: /en/papers/gptq/
pageClass: paper-reading
---

> [Elias Frantar](https://efrantar.github.io/) [+author-note], [Saleh Ashkboos](https://sashkboos.github.io/), [Torsten Hoefler](https://htor.inf.ethz.ch/), and [Dan Alistarh](https://daslab.ista.ac.at/). First submitted to arXiv on October 31, 2022; revised as v2 on March 22, 2023; published at [ICLR 2023](https://openreview.net/forum?id=tcbBPnfwxS). [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323). <a href="/paper/gptq.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [TeX source](https://export.arxiv.org/e-print/2210.17323). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Generative Pre-trained Transformer models, known as GPT or OPT, set themselves apart through breakthrough performance across complex language modelling tasks, but also by their extremely high computational and storage costs. Specifically, due to their massive size, even inference for large, highly-accurate GPT models may require multiple performant GPUs, which limits the usability of such models. While there is emerging work on relieving this pressure via model compression, the applicability and performance of existing compression techniques is limited by the scale and complexity of GPT models. In this paper, we address this challenge, and propose GPTQ, a new one-shot weight quantization method based on approximate second-order information, that is both highly-accurate and highly-efficient. Specifically, GPTQ can quantize GPT models with 175 billion parameters in approximately four GPU hours, reducing the bitwidth down to 3 or 4 bits per weight, with negligible accuracy degradation relative to the uncompressed baseline. Our method more than doubles the compression gains relative to previously-proposed one-shot quantization methods, preserving accuracy, allowing us for the first time to execute an 175 billion-parameter model inside a single GPU for generative inference. Moreover, we also show that our method can still provide reasonable accuracy in the *extreme quantization* regime, in which weights are quantized to 2-bit or even *ternary* quantization levels. We show experimentally that these improvements can be leveraged for end-to-end inference speedups over FP16, of around 3.25x when using high-end GPUs (NVIDIA A100) and 4.5x when using more cost-effective ones (NVIDIA A6000). The implementation is available at [https://github.com/IST-DASLab/gptq](https://github.com/IST-DASLab/gptq).

<span id="section-1"></span>

## 1 Introduction

Pre-trained generative models from the Transformer [Vas17] family, commonly known as GPT or OPT [Rad19, Bro20, Zha22], have shown breakthrough performance for complex language modelling tasks, leading to massive academic and practical interest. One major obstacle to their usability is computational and storage cost, which ranks among the highest for known models. For instance, the best-performing model variants, e.g. GPT3-175B, have in the order of 175 billion parameters and require tens-to-hundreds of GPU years to train [Zha22]. Even the simpler task of inferencing over a pre-trained model, which is our focus in this paper, is highly challenging: for instance, the parameters of GPT3-175B occupy 326GB (counting in multiples of 1024) of memory when stored in a compact float16 format. This exceeds the capacity of even the highest-end single GPUs, and thus inference must be performed using more complex and expensive setups, such as multi-GPU deployments.

Although a standard approach to eliminating these overheads is *model compression*, e.g. [Hoe21, Gho21a], surprisingly little is known about compressing such models for inference. One reason is that more complex methods for low-bitwidth quantization or model pruning usually require *model retraining*, which is extremely expensive for billion-parameter models. Alternatively, *post-training* methods [Nag20, Wan20g, Hub20, Nah21], which compress the model in one shot, without retraining, would be very appealing. Unfortunately, the more accurate variants of such methods [Li21a, Hub21, Fra22a] are complex and challenging to scale to billions of parameters [Yao22]. To date, only basic variants of round-to-nearest quantization [Yao22, Det22] have been applied at the scale of GPT-175B; while this works well for low compression targets, e.g., 8-bit weights, they fail to preserve accuracy at higher rates. It therefore remains open whether one-shot *post-training quantization* to higher compression rates is generally-feasible.

<span id="figure-01"></span>

![Original paper Figure 1](../../papers/gptq/figure-01.png)

**Figure 1.** Quantizing OPT models to 4 and BLOOM models to 3 bit precision, comparing GPTQ with the FP16 baseline and round-to-nearest (RTN) [Yao22, Det22].

**Contribution.** In this paper, we present a new post-training quantization method, called GPTQ, [+gptq-name] which is efficient enough to execute on models with hundreds of billions of parameters in at most a few hours, and precise enough to compress such models to 3 or 4 bits per parameter without significant loss of accuracy. For illustration, GPTQ can quantize the largest publicly-available models, OPT-175B and BLOOM-176B, in approximately four GPU hours, with minimal increase in perplexity, known to be a very stringent accuracy metric.

Further, we show that our model can also provide robust results in the *extreme quantization* regime, in which models are quantized to 2 bits per component, or even *ternary values*. On the practical side, we develop an execution harness which allows us to execute the resulting compressed models efficiently for generative tasks. Specifically, we are able to run the compressed OPT-175B model for the first time on a single NVIDIA A100 GPU, or using only two more cost-effective NVIDIA A6000 GPUs. We also implement bespoke GPU kernels which are able to leverage compression for faster memory loading, resulting in speedups of $\approx 3.25 \times$ when using A100 GPUs, and $4.5\times$ when using A6000 GPUs.

To our knowledge, we are the first to show that extremely accurate language models with hundreds of billions of parameters can be quantized to 3-4 bits/component: prior *post-training methods* only remain accurate at 8 bits [Yao22, Det22], while prior *training-based* techniques have only tackled models that are smaller by one to two orders of magnitude [Wu22b]. This high degree of compression may appear natural, as these networks are overparametrized; yet, as we discuss in our detailed analysis of results, compression induces non-trivial tradeoffs between the accuracy of the language modeling (perplexity), bit-width, and the size of the original model.

We hope that our work will stimulate further research in this area, and can be a further step towards making these models available to a wider audience. In terms of limitations, our method currently does not provide speedups for the actual multiplications, due to the lack of hardware support for mixed-precision operands (e.g. FP16 x INT4) on mainstream architectures. Moreover, our current results do not include activation quantization, as they are not a significant bottleneck in our target scenarios; however, this can be supported using orthogonal techniques [Yao22].

<span id="section-2"></span>

## 2 Related Work

Quantization methods fall broadly into two categories: quantization during training, and post-training methods. The former quantize models during typically extensive retraining and/or finetuning, using some approximate differentiation mechanism for the rounding operation [Gho21a, Nag21]. By contrast, post-training ("one-shot") methods quantize a pretrained model using modest resources, typically a few thousand data samples and a few hours of computation. Post-training approaches are particularly interesting for massive models, for which full model training or even finetuning can be expensive. We focus on this scenario here.

**Post-training Quantization.** Most post-training methods have focused on vision models. Usually, accurate methods operate by quantizing either individual layers, or small blocks of consecutive layers. (See [Section 3](#section-3) for more details.) The AdaRound method [Nag20] computes a data-dependent rounding by annealing a penalty term, which encourages weights to move towards grid points corresponding to quantization levels. BitSplit [Wan20g] constructs quantized values bit-by-bit using a squared error objective on the residual error, while AdaQuant [Hub21] performs direct optimization based on straight-through estimates. BRECQ [Li21a] introduces Fisher information into the objective, and optimizes layers within a single residual block jointly. Finally, Optimal Brain Quantization (OBQ) [Fra22a] generalizes the classic Optimal Brain Surgeon (OBS) second-order weight pruning framework [Has93, Sin20, Fra21] to apply to quantization. OBQ quantizes weights one-by-one, in order of quantization error, always adjusting the remaining weights. While these approaches can produce good results for models up to $\approx 100$ million parameters in a few GPU hours, scaling them to networks orders of magnitude larger is challenging.

**Large-model Quantization.** With the recent open-source releases of language models like BLOOM [Lau22] or OPT-175B [Zha22], researchers have started to develop affordable methods for compressing such giant networks for inference. While all existing works—ZeroQuant [Yao22], LLM.int8() [Det22], and nuQmm [Par22]— carefully select quantization granularity, e.g., vector-wise, they ultimately just round weights to the nearest (RTN) quantization level, in order to maintain acceptable runtimes for very large models. ZeroQuant further proposes layer-wise knowledge distillation, similar to AdaQuant, but the largest model it can apply this approach to has only 1.3 billion parameters. At this scale, ZeroQuant already takes $\approx 3$ hours of compute; GPTQ quantizes models 100$\times$ larger in $\approx 4$ hours. LLM.int8() observes that *activation outliers* in a few feature dimensions break the quantization of larger models, and proposes to fix this problem by keeping those dimensions in higher precision. Lastly, nuQmm develops efficient GPU kernels for a specific binary-coding based quantization scheme.

Relative to this line of work, we show that a significantly more complex and accurate quantizer can be implemented efficiently at large model scale. Specifically, GPTQ more than doubles the amount of compression relative to these prior techniques, at similar accuracy.

<span id="section-3"></span>

## 3 Background

**Layer-Wise Quantization.** At a high level, our method follows the structure of state-of-the-art post-training quantization methods [Nag20, Wan20g, Hub21, Fra22a], by performing quantization layer-by-layer, solving a corresponding reconstruction problem for each layer. Concretely, let $\mathbf{W_\ell}$ be the weights corresponding to a linear layer $\ell$ and let $\mathbf{X}_\ell$ denote the layer input corresponding to a small set of $m$ data points running through the network. Then, the objective is to find a matrix of quantized weights $\mathbf{\widehat{W}}$ which minimizes the squared error, relative to the full precision layer output. Formally, this can be restated as

<span id="equation-01"></span>

$$
\operatorname*{argmin}_{\mathbf{\widehat{W}}} \, \|\mathbf{W} \mathbf{X} - \mathbf{\widehat{W}} \mathbf{X}\|_2^2.
$$

Further, similar to [Nag20, Li21a, Fra22a], we assume that the quantization grid for $\mathbf{\widehat{W}}$ is fixed before the process, and that individual weights can move freely as in [Hub21, Fra22a].

**Optimal Brain Quantization.** Our approach builds on the recently-proposed Optimal Brain Quanization (OBQ) method [Fra22a] for solving the layer-wise quantization problem defined above, to which we perform a series of major modifications, which allow it to scale to large language models, providing more than *three orders of magnitude* computational speedup. To aid understanding, we first briefly summarize the original OBQ method.

The OBQ method starts from the observation that [Equation 1](#equation-01) can be written as the sum of the squared errors, over each row of $\mathbf{W}$. Then, OBQ handles each row $\mathbf{w}$ independently, quantizing one weight at a time while always updating all not-yet-quantized weights, in order to compensate for the error incurred by quantizing a single weight. Since the corresponding objective is a quadratic, whose Hessian is $\mathbf{H}_F = 2\mathbf{X}_F\mathbf{X}_F^\top$, where $F$ denotes the set of remaining full-precision weights, the greedy-optimal weight to quantize next, which we denote by $w_q$, and the corresponding optimal update of all weights in $F$, denoted by $\boldsymbol{\delta}_F$, are given by the following formulas, where $\mathrm{quant}(w)$ rounds $w$ to the nearest value on the quantization grid:

<span id="equation-02"></span>

$$
w_q = \operatorname*{argmin}_{w_q} \, \frac{(\mathrm{quant}(w_q) - w_q)^2}{[\mathbf{H}_F^{-1}]_{qq}}, \quad \boldsymbol{\delta}_F = - \frac{w_q - \mathrm{quant}(w_q)}{[\mathbf{H}_F^{-1}]_{qq}} \cdot (\mathbf{H}_F^{-1})_{:, q}.
$$

OBQ quantizes weights iteratively using these two equations, until all the weights of $\mathbf{w}$ are quantized. This is done efficiently, avoiding expensive full recomputations of $\mathbf{H}^{-1}$, by removing the $q$th row and column of $\mathbf{H}$, which is necessary after quantizing $w_q$, directly in the inverse via one step of Gaussian elimination. Namely, the updated inverse is given by the formula

<span id="equation-03"></span>

$$
\mathbf{H}_{-q}^{-1} = \Big(\mathbf{H}^{-1} - \frac{1}{[\mathbf{H}^{-1}]_{qq}} \mathbf{H}^{-1}_{:, q} \mathbf{H}^{-1}_{q, :} \Big)_{-p}.
$$

This method comes with a vectorized implementation, handling multiple rows of $\mathbf{W}$ in parallel. Eventually, the algorithm can achieve reasonable runtimes on medium-sized models: for instance, it can fully quantize the ResNet-50 model (25M parameters) in $\approx 1$ hour on a single GPU, which is roughly in line with other post-training methods achieving state-of-the-art accuracy [Fra22a]. However, the fact that OBQ's runtime for a $d_{\mathrm{row}} \times d_{\mathrm{col}}$ matrix $\mathbf{W}$ has *cubic* input dependency $O(d_{\mathrm{row}} \cdot d_{\mathrm{col}}^3)$ means that applying it to models with billions of parameters is extremely expensive.

<span id="section-4"></span>

## 4 The GPTQ Algorithm

**Step 1: Arbitrary Order Insight.** As explained in the previous section, OBQ quantizes weights in greedy order, i.e. it always picks the weight which currently incurs the least additional quantization error. Interestingly, we find that, while this quite natural strategy does indeed seem to perform very well, its improvement over quantizing the weights in arbitrary order is generally small, in particular on large, heavily-parametrized layers. Most likely, this is because the slightly lower number of quantized weights with large individual error is balanced out by those weights being quantized towards the end of the process, when only few other unquantized weights that can be adjusted for compensation remain. As we will now discuss, this insight that *any fixed order may perform well*, especially on large models, has interesting ramifications.

<span id="figure-02"></span>

![Original paper Figure 2](../../papers/gptq/figure-02.png)

**Figure 2.** GPTQ quantization procedure. Blocks of consecutive *columns* (bolded) are quantized at a given step, using the inverse Hessian information stored in the Cholesky decomposition, and the remaining weights (blue) are updated at the end of the step. The quantization procedure is applied recursively inside each block: the white middle column is currently being quantized.

The original OBQ method quantizes rows of $\mathbf{W}$ independently, in a specific order defined by the corresponding errors. By contrast, we will aim to quantize the weights of *all rows in the same order*, and will show that this typically yields results with a final squared error that is similar to the original solutions. As a consequence, the set of unquantized weights $F$ and similarly $\mathbf{H}_F^{-1}$ is always the same for all rows (see [Figure 2](#figure-02) for an illustration). In more detail, the latter is due to the fact that $\mathbf{H}_F$ depends only on the layer inputs $\mathbf{X}_F$, which are the same for all rows, and not on any weights. Therefore, we have to perform the update of $\mathbf{H}_F^{-1}$ given by [Equation 3](#equation-03) only $d_{\mathrm{col}}$ times, once per column, rather than $d_{\mathrm{row}} \cdot d_{\mathrm{col}}$ times, once per weight. This reduces the overall runtime from $O(d_{\mathrm{row}} \cdot d_{\mathrm{col}}^3)$ to $O(\max \, \{d_{\mathrm{row}} \cdot d_{\mathrm{col}}^2, d_{\mathrm{col}}^3\})$, i.e., by a factor of $\min \, \{d_{\mathrm{row}}, d_{\mathrm{col}}\}$. For larger models, this difference consists of several orders of magnitude. However, before this algorithm can actually be applied to very large models in practice, two additional major problems need to be addressed.

**Step 2: Lazy Batch-Updates.** First, a direct implementation of the scheme described previously will not be fast in practice, because the algorithm has a relatively low compute-to-memory-access ratio. For example, [Equation 3](#equation-03) needs to update all elements of a potentially huge matrix using just a few FLOPs for each entry. Such operations cannot properly utilize the massive compute capabilities of modern GPUs, and will be bottlenecked by the significantly lower memory bandwidth.

Fortunately, this problem can be resolved by the following observation: The final rounding decisions for column $i$ are only affected by updates performed on this very column, and so updates to later columns are irrelevant at this point in the process. This makes it possible to "lazily batch" updates together, thus achieving much better GPU utilization. Concretely, we apply the algorithm to $B = 128$ columns at a time, keeping updates contained to those columns and the corresponding $B \times B$ block of $\mathbf{H}^{-1}$ (see also [Figure 2](#figure-02)). Only once a block has been fully processed, we perform global updates of the entire $\mathbf{H}^{-1}$ and $\mathbf{W}$ matrices using the multi-weight versions of [Equation 2](#equation-02) and [Equation 3](#equation-03) given below, with $Q$ denoting a set of indices, and $\mathbf{H}_{-Q}^{-1}$ denoting the inverse matrix with the corresponding rows and columns removed:

<span id="equation-05"></span>

$$
\begin{aligned}
\boldsymbol{\delta}_F &= -(\mathbf{w}_Q - \mathrm{quant}(\mathbf{w}_Q))([\mathbf{H}_F^{-1}]_{Q,Q})^{-1} (\mathbf{H}_F^{-1})_{:, Q}, \\
    \mathbf{H}_{-Q}^{-1} &= \Big(\mathbf{H}^{-1} - \mathbf{H}^{-1}_{:, Q} ([\mathbf{H}^{-1}]_{Q,Q})^{-1} \mathbf{H}^{-1}_{Q, :} \Big)_{-Q}.
\end{aligned}
$$

Although this strategy does not reduce the theoretical amount of compute, it effectively addresses the memory-throughput bottleneck. This provides an order of magnitude speedup for very large models in practice, making it a critical component of our algorithm.

**Step 3: Cholesky Reformulation.** The final technical issue we have to address is given by numerical inaccuracies, which can become a major problem at the scale of existing models, especially when combined with the block updates discussed in the previous step. Specifically, it can occur that the matrix $\mathbf{H}_F^{-1}$ becomes indefinite, which we notice can cause the algorithm to aggressively update the remaining weights in incorrect directions, resulting in an arbitrarily-bad quantization of the corresponding layer. In practice, we observed that the probability of this happening increases with model size: concretely, it almost certainly occurs for at least a few layers on models that are larger than a few billion parameters. The main issue appears to be the repeated applications of [Equation 5](#equation-05), which accumulate various numerical errors, especially through the additional matrix inversion.

For smaller models, applying dampening, that is adding a small constant $\lambda$ (we always choose 1% of the average diagonal value) to the diagonal elements of $\mathbf{H}$ appears to be sufficient to avoid numerical issues. However, larger models require a more robust and general approach.

To address this, we begin by noting that the only information required from $\mathbf{H}_{F_q}^{-1}$, where $F_q$ denotes the set of unquantized weights when quantizing weight $q$, is row $q$, or more precisely, the elements in this row starting with the diagonal. The consequence is that we could precompute all of these rows using a more numerically-stable method without any significant increase in memory consumption. Indeed, the row removal via [Equation 3](#equation-03) for our symmetric $\mathbf{H}^{-1}$ essentially corresponds to taking a Cholesky decomposition, except for the minor difference that the latter divides row $q$ by $([\mathbf{H}^{-1}_{F_q}]_{qq})^{1/2}$. Hence, we can leverage state-of-the-art Cholesky kernels to compute all information we will need from $\mathbf{H}^{-1}$ upfront. In combination with mild dampening, the resulting method is robust enough to execute on huge models without issues. As a bonus, using a well-optimized Cholesky kernel also yields further speedup. We detail all small changes necessary for the Cholesky version of the algorithm next.

**The Full Algorithm.** Finally, we present the full pseudocode for GPTQ in [Algorithm 1](#algorithm-01), including the optimizations discussed above.

<span id="algorithm-01"></span>

**Algorithm 1: Quantize $\mathbf{W}$ given inverse Hessian $\mathbf{H}^{-1} = (2 \mathbf{X} \mathbf{X}^\top + \lambda \mathbf{I})^{-1}$ and blocksize $B$.**

- Set $\mathbf{Q} \gets \mathbf{0}_{d_{\mathrm{row}} \times d_{\mathrm{col}}}$ — quantized output.
- Set $\mathbf{E} \gets \mathbf{0}_{d_{\mathrm{row}} \times B}$ — block quantization errors.
- Set $\mathbf{H}^{-1} \gets \mathrm{Cholesky}(\mathbf{H}^{-1})^\top$ — Hessian inverse information.
- **For** $i = 0, B, 2B, \dots$:
  - **For** $j = i, \dots, i + B - 1$:
    - Set $\mathbf{Q}_{:,j} \gets \mathrm{quant}(\mathbf{W}_{:,j})$ — quantize column.
    - Set $\mathbf{E}_{:,j-i} \gets (\mathbf{W}_{:,j} - \mathbf{Q}_{:,j}) / [\mathbf{H}^{-1}]_{jj}$ — quantization error.
    - Set $\mathbf{W}_{:,j:(i+B)} \gets \mathbf{W}_{:,j:(i+B)} - \mathbf{E}_{:,j-i} \cdot \mathbf{H}^{-1}_{j,j:(i+B)}$ — update weights in block.
  - Set $\mathbf{W}_{:,(i+B):} \gets \mathbf{W}_{:,(i+B):} - \mathbf{E} \cdot \mathbf{H}^{-1}_{i:(i+B),(i+B):}$ — update all remaining weights.

<span id="section-5"></span>

## 5 Experimental Validation

**Overview.** We begin our experiments by validating the accuracy of GPTQ relative to other accurate-but-expensive quantizers, on smaller models, for which these methods provide reasonable runtimes. Next, we examine GPTQ's runtime scaling for very large models. Then, we present 3- and 4-bit quantization results for the entire BLOOM and OPT model families, evaluated via perplexity on challenging language generation tasks. In addition, we show that our method is also stable for 2-bit quantization when the granularity is reduced to small blocks of consecutive weights. To complement this perplexity analysis, we also evaluate the resulting quantized models on a series of standard zero-shot tasks. Finally, we focus on the two largest (and interesting) openly-available models, Bloom-176B and OPT-175B, where we perform a detailed evaluation on several tasks. For these models, we also present practical improvements, namely reducing the number of GPUs required for inference as well as end-to-end speedups for generative tasks.

**Setup.** We implemented GPTQ in PyTorch [Pas19] and worked with the HuggingFace integrations of the BLOOM [Lau22] and OPT [Zha22] model families. We quantized all models (including the 175 billion parameter variants) *using a single NVIDIA A100 GPU* with 80GB of memory. Our entire GPTQ calibration data consists of 128 random 2048 token segments from the C4 dataset [Raf20], i.e., excerpts from randomly crawled websites, which represents generic text data. We emphasize that this means that GPTQ does not see any task-specific data, and our results thus remain actually "zero-shot". We perform standard uniform per-row asymmetric quantization on the min-max grid, similar to [Det22]. Additional evaluation details can be found in [Section 9.2.1](#section-9-2-1).

To ensure that the entire compression procedure can be performed with significantly less GPU memory than what would be required to run the full precision model, some care must be taken. Specifically, we always load one Transformer block, consisting of 6 layers, at a time into GPU memory and then accumulate the layer-Hessians and perform quantization. Finally, the current block inputs are sent through the fully quantized block again to produce the new inputs for the quantization of the next block. Hence, the quantization process operates not on the layer inputs in the full precision model but on the actual layer inputs in the already partially quantized one. We find that this brings noticeable improvements at negligible extra cost.

**Baselines.** Our primary baseline, denoted by RTN, consists of rounding all weights to the nearest quantized value on exactly the same asymmetric per-row grid that is also used for GPTQ, meaning that it corresponds precisely to the state-of-the-art weight quantization of LLM.int8(). This is currently the method of choice in all works on quantization of very large language models [Det22, Yao22, Par22]: its runtime scales well to networks with many billions of parameters, as it simply performs direct rounding. As we will also discuss further, more accurate methods, such as AdaRound [Nag20] or BRECQ [Li21a], are currently too slow for models with many billions of parameters, the main focus of this work. Nevertheless, we also show that GPTQ is competitive with such methods for small models, while scaling to huge ones like OPT-175B as well.

**Quantizing Small Models.** As a first ablation study, we compare GPTQ's performance relative to state-of-the-art post-training quantization (PTQ) methods, on ResNet18 and ResNet50, which are standard PTQ benchmarks, in the same setup as [Fra22a]. As can be seen in [Table 1](#table-01), GPTQ performs on par at 4-bit, and slightly worse than the most accurate methods at 3-bit. At the same time, it significantly outperforms AdaQuant, the fastest amongst prior PTQ methods. Further, we compare against the full greedy OBQ method on two smaller language models: BERT-base [Dev19] and OPT-125M. The results are shown in Appendix [Table 8](#table-08). At 4 bits, both methods perform similarly, and for 3 bits, GPTQ surprisingly performs slightly better. We suspect that this is because some of the additional heuristics used by OBQ, such as early outlier rounding, might require careful adjustments for optimal performance on non-vision models. Overall, GPTQ appears to be competitive with state-of-the-art post-training methods for smaller models, while taking only $< 1$ minute rather than $\approx 1$ hour. This enables scaling to much larger models.

<span id="table-01"></span>

![Original paper Table 1](../../papers/gptq/table-01.png)

**Table 1.** Comparison with state-of-the-art post-training methods for vision models.

<span id="table-02"></span>

![Original paper Table 2](../../papers/gptq/table-02.png)

**Table 2.** GPTQ runtime for full quantization of the 4 largest OPT and BLOOM models.

**Runtime.** Next we measure the full model quantization time (on a single NVIDIA A100 GPU) via GPTQ; the results are shown in [Table 2](#table-02). As can be seen, GPTQ quantizes 1-3 billion parameter models in a matter of minutes and 175B ones in a few hours. For reference, the straight-through based method ZeroQuant-LKD [Yao22] reports a 3 hour runtime (on the same hardware) for a 1.3B model, which would linearly extrapolate to several hundred hours (a few weeks) for 175B models. Adaptive rounding-based methods typically employ a lot more SGD steps and would thus be even more expensive [Nag20, Li21a].

**Language Generation.** We begin our large-scale study by compressing the entire OPT and BLOOM model families to 3- and 4-bit. We then evaluate those models on several language tasks including WikiText2 [Mer16] (see [Figure 1](#figure-01) as well as [Table 3](#table-03) and [Table 4](#table-04)), Penn Treebank (PTB) [Mar94] and C4 [Raf20] (both in [Section 9.3](#section-9-3)). We focus on these perplexity-based tasks, as they are known to be particularly sensitive to model quantization [Yao22]. On OPT models, GPTQ clearly outperforms RTN, by significant margins. For example, GPTQ loses only 0.03 perplexity at 4-bit on the 175B model, while RTN drops 2.2 points, performing worse than the $10\times$ smaller full-precision 13B model. At 3-bit, RTN collapses completely, while GPTQ can still maintain reasonable perplexity, in particular for larger models. BLOOM shows a similar pattern: the gaps between methods are however usually a bit smaller, indicating that this model family might be easier to quantize. One interesting trend (see also [Figure 1](#figure-01)) is that larger models generally (with the exception of OPT-66B [+opt-66]) appear easier to quantize. This is good news for practical applications, as these are the cases where compression is also the most necessary.

<span id="table-03"></span>

![Original paper Table 3](../../papers/gptq/table-03.png)

**Table 3.** OPT perplexity results on WikiText2.

<span id="table-04"></span>

![Original paper Table 4](../../papers/gptq/table-04.png)

**Table 4.** BLOOM perplexity results for WikiText2.

**175 Billion Parameter Models.** We now examine BLOOM-176B and OPT-175B, the largest dense openly-available models. [Table 5](#table-05) summarizes results across Wikitext-2, PTB, C4. We observe that, at 4 bits, GPTQ models reach only $\leq 0.25$ lower perplexity than the full-precision versions, with a large gap to RTN results on OPT-175B. At 3-bit, RTN collapses, while GPTQ is still able to maintain good performance on most tasks, losing only $0.3 - 0.6$ points for more than $5\times$ compression. We note that GPTQ's accuracy can be further improved via finer-granularity grouping [Par22]: group-size 1024 ($\approx$ 0.02 extra bits) improves perplexities by about $0.2$ on average and group-size 128 ($\approx$ 0.15 extra bits) by another $0.1$, which is only $0.1 - 0.3$ off from the uncompressed accuracy. We note that grouping interacts very well with GPTQ, as the group parameters can be determined during the quantization process of each layer, always using the most current updated weights.

<span id="table-05"></span>

![Original paper Table 5](../../papers/gptq/table-05.png)

**Table 5.** Results summary for OPT-175B and BLOOM-176B. “g1024” and “g128” denote results with groupings of size 1024 and 128, respectively.

**Practical Speedups.** Finally, we study practical applications. As an interesting use-case, we focus on the OPT-175B model: quantized to 3 bits, this model takes approximately 63GB of memory, including the embeddings and the output layer, which are kept in full FP16 precision. Additionally, storing the complete history of keys and values for all layers, a common optimization for generation tasks, consumes another $\approx 9$GB for the maximum of 2048 tokens. Hence, we can actually fit the entire quantized model into a single 80GB A100 GPU, which can be executed by dynamically dequantizing layers as they are required during inference (the model would not fully fit using 4 bits). For reference, standard FP16 execution requires 5x80GB GPUs, and the state-of-the-art 8bit LLM.int8() quantizer [Det22] requires 3 such GPUs.

Next, we consider language generation, one of the most appealing applications of these models, with the goal of latency reduction. Unlike LLM.int8(), which reduces memory costs but has the same runtime as the FP16 baseline, we show that our quantized models can achieve significant speedups for this application. For language generation, the model processes and outputs one token at-a-time, which for OPT-175B can easily take a few 100s of milliseconds per token. Increasing the speed at which the user receives generated results is challenging, as compute is dominated by matrix-vector products. Unlike matrix-matrix products, these are primarily limited by memory bandwidth. We address this problem by developing a quantized-matrix full-precision-vector product kernel which performs a matrix vector product by dynamically dequantizing weights when needed. Most notably, this does *not* require any activation quantization. While dequantization consumes extra compute, the kernel has to access a lot less memory, leading to significant speedups, as shown in [Table 6](#table-06). We note that almost all of the speedup is due to our kernels, as communication costs are negligible in our standard HuggingFace-accelerate-like setting (see [Section 9.2.2](#section-9-2-2) for details).

<span id="table-06"></span>

![Original paper Table 6](../../papers/gptq/table-06.png)

**Table 6.** Average per-token latency (batch size 1) when generating sequences of length 128.

For example, using our kernels, the 3-bit OPT-175B model obtained via GPTQ running on a single A100 is about $\mathbf{3.25\boldsymbol{\times}}$ faster than the FP16 version (running on 5 GPUs) in terms of average time per token. More accessible GPUs, such as the NVIDIA A6000, have much lower memory bandwidth, so this strategy is even more effective: executing the 3-bit OPT-175B model on 2x A6000 GPUs reduces latency from 589 milliseconds for FP16 inference (on 8 GPUs) to 130 milliseconds, a $\mathbf{4.5\boldsymbol{\times}}$ latency reduction.

<span id="figure-03"></span>

![Original paper Figure 3](../../papers/gptq/figure-03.png)

**Figure 3.** The accuracy of OPT and BLOOM models post-GPTQ, measured on LAMBADA.

**Zero-Shot Tasks.** While our focus is on language generation, we also evaluate the performance of quantized models on some popular zero-shot tasks, namely LAMBADA [Pap16], ARC (Easy and Challenge) [Bor18] and PIQA [Tat03]. [Figure 3](#figure-03) visualizes model performance on LAMBADA (and see also "Lamb." results in [Table 5](#table-05)). We observe similar behavior as before: the outliers are that 1) quantization appears "easier" across the whole spectrum of models at 4-bit, where even RTN performs relatively well, and 2) at 3-bit, RTN breaks down, while GPTQ still provides good accuracy. We provide additional results in [Section 9.4](#section-9-4).

**Additional Tricks.** While our experiments so far have focused exclusively on vanilla row-wise quantization, we want to emphasize that GPTQ is *compatible with essentially any choice of quantization grid*. For example, it is easily combined with standard *grouping* [Ali17, Par22], i.e. applying independent quantization to groups of $g$ consecutive weights. As shown in the last rows of [Table 5](#table-05), this can bring noticeable extra accuracy for the largest models at 3-bit. Further, as visualized in [Figure 4](#figure-04), it significantly reduces the accuracy losses for medium sized models at 4-bit precision.

<span id="table-07"></span>

![Original paper Table 7](../../papers/gptq/table-07.png)

**Table 7.** 2-bit GPTQ quantization results with varying group-sizes; perplexity on WikiText2.

<span id="figure-04"></span>

![Original paper Figure 4](../../papers/gptq/figure-04.png)

**Figure 4.** GPTQ at 4-bit with different group-sizes on medium sized OPT models.

**Extreme Quantization.** Lastly, grouping also makes it possible to achieve reasonable performance for extreme quantization, to around 2-bits per component on average. [Table 7](#table-07) shows results on WikiText2 when quantizing the biggest models to 2-bit with varying group-sizes. At $\approx 2.2$ bit (group-size 128; using FP16 scale and 2-bit zero point per group) the perplexity increase is already less than 1.5 points, while dropping to 0.6 - 0.7 at $\approx 2.6$ bit (group-size 32), which is only slightly worse than vanilla 3-bit and might be interesting for practical kernel implementations. Further, if we reduce group size to 8, we can apply *ternary* (-1, 0, +1) quantization, which achieves 9.20 WikiText2 PPL on OPT-175B, a less than 1 point drop. While this leads to worse compression on average relative to the 2-bit numbers above, this pattern could be efficiently implemented on custom hardware such as FPGAs. In summary, these results are an encouraging first step towards pushing highly-accurate *one-shot* compression of very large language models, even lower than 3 bits per value on average.

<span id="section-6"></span>

## 6 Summary and Limitations

We have presented GPTQ, an approximate second-order method for quantizing truly large language models. GPTQ can accurately compress some of the largest publicly-available models down to 3 and 4 bits, which leads to significant usability improvements, and to end-to-end speedups, at low accuracy loss. We hope that our method will make these models accessible to more researchers and practitioners. At the same time, we emphasize some significant limitations: On the technical side, our method obtains speedups from reduced memory movement, and does not lead to computational reductions. In addition, our study focuses on generative tasks, and does not consider activation quantization. These are natural directions for future work, and we believe this can be achieved with carefully-designed GPU kernels and existing techniques [Yao22, Wu22b].

## Acknowledgments

Elias Frantar and Dan Alistarh gratefully acknowledge funding from the European Research Council (ERC) under the European Union's Horizon 2020 programme (grant agreement No. 805223 ScaleML), as well as experimental support from Eldar Kurtic, and from the IST Austria IT department, in particular Stefano Elefante, Andrei Hornoiu, and Alois Schloegl. The work of Saleh Ashkboos and Torsten Hoefler was supported by the PASC DaCeMI project, received EuroHPC-JU funding under grant MAELSTROM, No. 955513. We thank the Swiss National Supercomputing Center (CSCS) for supporting us with compute infrastructure.

<span id="section-7"></span>

## 7 Ethics Statement

Our work introduces a general method for compressing large language models (LLMs) via quantization, with little-to-no accuracy loss in terms of standard accuracy metrics such as perplexity. Our method is task-agnostic, as it only uses a tiny amount of randomly-chosen data for calibration. We therefore do not foresee any significant ethical implications arising directly from the technical details of our method. However, one possible consideration is that our study focused on "leading accuracy" metrics that are standard in the literature, such as perplexity, which is essentially standard in the literature [Det22, Yao22]. We believe a thorough study of the impact of compression upon secondary measures, and in particular bias effects [Ben21] is warranted, and may be rendered easier through our work. At the same time, our work makes inference on extremely large language models more accessible, for better or for worse. We believe that, in time, such tools will become much easier to use and deploy, making the need to understand their power and limitations even more stringent.

<span id="section-8"></span>

## 8 Reproducibility Statement

In the Supplementary Materials, we provide code to reproduce all experiments in this paper. More specifically, this includes:

- Compressing all models from the OPT and BLOOM model families to 2/3/4 bits.

- Evaluating perplexity of the quantized models.

- Our 3-bit CUDA kernel together with compressed inference benchmarking features.

- Code for the ZeroShot experiments.

- A README file providing sample commands and information on how to run all scripts.

<span id="section-9"></span>

## 9 Appendix

<span id="section-9-1"></span>

### 9.1 Additional Comparison with OBQ

We now provide an additional comparison between GPTQ and OBQ on BERT-base/SQuAD [Raj16] and OPT-125M/WikiText2, which is one of the largest models to which OBQ can be reasonably applied.

<span id="table-08"></span>

![Original paper Table 8](../../papers/gptq/table-08.png)

**Table 8.** Comparison of GPTQ relative to OBQ on BERT-base/SQuAD and OPT-125M/WikiText2.

<span id="section-9-2"></span>

### 9.2 Experiment Details

This section provides additional details about our experiment setup, in particular regarding the model evaluation and the setup of our timing experiments.

<span id="section-9-2-1"></span>

#### 9.2.1 Evaluation

For language generation experiments, we calculate the perplexity, in standard fashion like [Rad19], as follows: First, the entire validation set is concatenated using two linebreaks as separators and encoded using the default HuggingFace tokenizer of each model. Next, the sequence is split into non-overlapping segments of width 2048, the full context size of our models. These are sent through the model to collect the log-probabilities corresponding to the next token each. Their exponentiated average is the final perplexity we report.

For zero-shot tasks we follow the EleutherAI evaluation harness [+evaluation-harness] in terms of data preprocessing and final score calculation. We note that we evaluate all individual samples separately and thus do not apply any padding.

<span id="section-9-2-2"></span>

#### 9.2.2 Timing Experiment Setup

Our timing experiments are performed following the standard HuggingFace/accelerate [+accelerate] setup also used by the recent work LLM.int8() [Det22]. In this setting, the model is split by distributing chunks of consecutive layers across GPUs. Importantly, in this setup the communication costs are minimal, $< 5\%$ of the total runtime even when working with 8 GPUs. This means almost all of the reported speedups are due to our quantized-matrix full-precision vector product kernels. We emphasize that the only difference between the FP16 baseline and our quantized models are the kernels used to perform the underlying matrix-vector products.

This means all overheads due to HuggingFace, attention or non-quantized operations like residuals or LayerNorms are exactly the same. Consequently, our quantized models should benefit from more advanced distribution strategies [Zhe22] or more efficient attention kernels [Dao22] just as much as our baseline.

In general, our kernels target generative inference in the low batch-size setting (for simplicity, we consider only batchsize 1) where the underlying (close to) matrix-vector products are memory-bound. For non-generative and large-batch applications, operations may be compute- rather than memory-bound and our kernels thus not directly applicable. Instead, one could simply decompress the matrix before performing the corresponding matrix-matrix calculations: this takes $<$ 1.5ms on an A100 and $<$ 3ms on an A6000 compared to 76ms/365ms for the subsequent OPT-175B FC2 layer computation with batchsize $16 \times 1024$ tokens. Hence, for such applications our methods significantly reduce the required number of GPUs at very little computational overhead. This is similar to recent work [Det22], but we achieve a $2.5\times$ higher compression rate.

<span id="section-9-3"></span>

### 9.3 Additional Language Generation Results

[Table 9](#table-09), [Table 10](#table-10), [Table 11](#table-11) and [Table 12](#table-12) show additional results for language generation tasks.

<span id="table-09"></span>

![Original paper Table 9](../../papers/gptq/table-09.png)

**Table 9.** OPT perplexity results on PTB.

<span id="table-10"></span>

![Original paper Table 10](../../papers/gptq/table-10.png)

**Table 10.** BLOOM perplexity results for PTB.

<span id="table-11"></span>

![Original paper Table 11](../../papers/gptq/table-11.png)

**Table 11.** OPT perplexity results on C4. We note that the calibration data used by GPTQ is sampled from the C4 training set, this task is thus not fully zero-shot.

<span id="table-12"></span>

![Original paper Table 12](../../papers/gptq/table-12.png)

**Table 12.** BLOOM perplexity results for C4. We note that the calibration data used by GPTQ is sampled from the C4 training set, this task is thus not fully zero-shot.

<span id="section-9-4"></span>

### 9.4 Additional ZeroShot Results

This section contains additional results for zero-shot tasks.

<span id="table-13"></span>

![Original paper Table 13](../../papers/gptq/table-13.png)

**Table 13.** OPT accuracy on LAMBADA.

<span id="table-14"></span>

![Original paper Table 14](../../papers/gptq/table-14.png)

**Table 14.** BLOOM accuracy on LAMBADA.

<span id="table-15"></span>

![Original paper Table 15](../../papers/gptq/table-15.png)

**Table 15.** OPT accuracy on PIQA.

<span id="table-16"></span>

![Original paper Table 16](../../papers/gptq/table-16.png)

**Table 16.** BLOOM accuracy on PIQA.

<span id="table-17"></span>

![Original paper Table 17](../../papers/gptq/table-17.png)

**Table 17.** OPT accuracy on ARC-easy.

<span id="table-18"></span>

![Original paper Table 18](../../papers/gptq/table-18.png)

**Table 18.** BLOOM accuracy on ARC-easy.

<span id="table-19"></span>

![Original paper Table 19](../../papers/gptq/table-19.png)

**Table 19.** OPT accuracy on ARC-challenge.

<span id="table-20"></span>

![Original paper Table 20](../../papers/gptq/table-20.png)

**Table 20.** BLOOM accuracy on ARC-challenge.

<span id="table-21"></span>

![Original paper Table 21](../../papers/gptq/table-21.png)

**Table 21.** OPT accuracy on StoryCloze.

<span id="table-22"></span>

![Original paper Table 22](../../papers/gptq/table-22.png)

**Table 22.** BLOOM accuracy on StoryCloze.

[+author-note]: Corresponding author: `elias.frantar@ist.ac.at`.

[+gptq-name]: This merges the name of the OPT model family with the abbreviation for post-training quantization (PTQ).

[+opt-66]: Upon closer inspection of the OPT-66B model, it appears that this is correlated with the fact that this trained model has a significant fraction of dead units in the early layers, which may make it harder to compress.

[+evaluation-harness]: [https://github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)

[+accelerate]: [https://huggingface.co/docs/accelerate/index](https://huggingface.co/docs/accelerate/index)
