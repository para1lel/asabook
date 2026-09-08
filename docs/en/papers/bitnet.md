---
title: 'BitNet: Scaling 1-bit Transformers'
createTime: 2026/09/08 15:00:00
permalink: /en/papers/bitnet/
pageClass: paper-reading
---

> [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Shuming Ma](https://shumingma.com/) [+author-note], [Li Dong](https://dong.li/), [Shaohan Huang](https://buaahsh.github.io/), [Huaijie Wang](https://dblp.dagstuhl.de/pid/346/1061.html), [Lingxiao Ma](https://xysmlx.github.io/), [Fan Yang](https://fanyangcs.github.io/), [Ruiping Wang](https://www.jdl.link/user/rpwang/index.htm), [Yi Wu](https://jxwuyi.weebly.com/), and [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]. First submitted to arXiv on October 17, 2023; current version v1; work in progress. [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453). <a href="/paper/bitnet.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2310.11453). [TeX source](https://export.arxiv.org/e-print/2310.11453v1). The original PDF remains authoritative for the exact print layout and bibliography.

[+author-note]: Hongyu Wang and Shuming Ma contributed equally. Furu Wei is the corresponding author. Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Lingxiao Ma, Fan Yang, and Furu Wei are with Microsoft Research. Hongyu Wang and Ruiping Wang are with University of Chinese Academy of Sciences. Huaijie Wang and Yi Wu are with Tsinghua University. [GeneralAI](https://aka.ms/GeneralAI).

## Abstract

The increasing size of large language models has posed challenges for deployment and raised concerns about environmental impact due to high energy consumption. In this work, we introduce BitNet, a scalable and stable 1-bit Transformer architecture designed for large language models. Specifically, we introduce `BitLinear` as a drop-in replacement of the `nn.Linear` layer in order to train 1-bit weights from scratch. Experimental results on language modeling show that BitNet achieves competitive performance while substantially reducing memory footprint and energy consumption, compared to state-of-the-art 8-bit quantization methods and FP16 Transformer baselines. Furthermore, BitNet exhibits a scaling law akin to full-precision Transformers, suggesting its potential for effective scaling to even larger language models while maintaining efficiency and performance benefits.

<span id="figure-01"></span>

![BitNet performance, post-training quantization comparison, energy reduction, and scaling curves](../../papers/bitnet/figure-01.png)

**Figure 1.** BitNet trains 1-bit Transformers from scratch, obtaining competitive results in an energy-efficient way. BitNet significantly outperforms state-of-the-art quantization methods. As the model size scales up, the cost savings become more significant while achieving competitive performance with the models trained with FP16.

> I don't think there's anything unique about human intelligence. All the neurons in the brain that make up perceptions and emotions operate in a binary fashion.
>
> — William Henry Gates III

<span id="section-1"></span>

## 1 Introduction

The rapid growth of large language models [Bro20, Ope23, Cho22, Ani23, Tou23, Tou23a] has led to significant improvements in various tasks. However, it is expensive to host large language models due to the high inference costs and energy consumption. As the size of these models grows, the memory bandwidth required for accessing and processing the model parameters becomes a major bottleneck, limiting the overall inference performance. Moreover, when deploying these models on distributed systems or multi-device platforms, the inter-device communication overhead can significantly impact the inference latency and energy consumption. Model quantization [Fra23, Che24b, Xia23] has emerged as a promising solution, as it can significantly reduce the memory footprint and computational cost of large-scale models while maintaining competitive performance.

Most existing quantization approaches for large language models are post-training. They are simple and easy to apply since it does not require any changes to the training pipeline or retraining the model. However, it will result in a more significant loss of accuracy especially when the precision goes lower, because the model is not optimized for the quantized representation during training.

Another strand of quantizing deep neural networks is quantization-aware training. Compared to post-training, it typically results in better accuracy, as the model is trained to account for the reduced precision from the beginning. Moreover, it allows the model to continue-train or do fine-tuning, which is essential for large language models. The challenge of quantization-aware training mainly lies in optimization, i.e., the model becomes more difficult to converge as the precision goes lower. Besides, it is unknown whether quantization-aware training follows the scaling law of neural language models.

In this work, we focus on binarization (i.e., 1-bit), which is the extreme case of quantization, applied to large language models. Previous studies on binarized neural networks [Ras16, Bul19a] have mostly revolved around convolutional neural networks. Recently, there has been some research on binarized Transformers. However, these studies have focused on machine translation or BERT pretraining, which is quite different from large language models. For example, machine translation employs an encoder-decoder architecture, BERT pretraining utilizes a bidirectional encoder, and large language models use a unidirectional decoder. Furthermore, large language models are typically scaled up to a much larger model size, while BERT and machine translation models do not undergo such extensive scaling.

To the best of our knowledge, this work is the first to investigate quantization-aware training for 1-bit large language models. We propose BitNet, a 1-bit Transformer architecture for large language models, which aims to scale efficiently in terms of both memory and computation. BitNet employs low-precision binary weights and quantized activations, while maintaining high precision for the optimizer states and gradients during training. Our approach is designed to be scalable and stable, with the ability to handle large language models efficiently. The implementation of the BitNet architecture is quite simple, requiring only the replacement of linear projections (i.e., *nn.Linear* in PyTorch) in the Transformer. Furthermore, it complements other acceleration methods for large language models, such as PagedAttention [Kwo23], FlashAttention [Dao22, Dao24a], and speculative decoding [Lev23].

We evaluate BitNet on a range of language modeling benchmarks, comparing with state-of-the-art quantization methods and FP16 Transformers. Experimental results demonstrate that BitNet achieves competitive performance in terms of both perplexity and downstream task accuracy. More importantly, BitNet significantly reduces memory footprint and energy consumption compared to the baselines. Furthermore, we show that BitNet follows a scaling law similar to that of full-precision Transformers, indicating that it can be effectively scaled to even larger language models with potential benefits in terms of performance and efficiency.

<span id="section-2"></span>

## 2 BitNet

As shown in [Figure 2](#figure-02), BitNet uses the same layout as Transformers, stacking blocks of self-attention and feed-forward networks. Compared with vanilla Transformer, BitNet uses `BitLinear` ([Equation 11](#equation-11)) instead of conventional matrix multiplication, which employs binarized (i.e., 1-bit) model weights. We leave the other components high-precision, e.g., 8-bit in our experiments. We summarized the reasons as follows. First, the residual connections and the layer normalization contribute negligible computation costs to large language models. Second, the computation cost of QKV transformation is much smaller than the parametric projection as the model grows larger. Third, we preserve the precision for the input/output embedding because the language models have to use high-precision probabilities to perform sampling.

<span id="figure-02"></span>

![BitLinear computation flow and BitNet architecture](../../papers/bitnet/figure-02.png)

**Figure 2.** (a) The computation flow of `BitLinear`. (b) The architecture of BitNet, consisting of the stacks of attentions and FFNs, where matrix multiplication is implemented as `BitLinear`.

<span id="section-2-1"></span>

### 2.1 BitLinear

We first binarize the weights to either $+1$ or $-1$ with the signum function. Following [Liu22], we centralize the weights to be zero-mean before binarization to increase the capacity within a limited numerical range. A scaling factor $\beta$ is used after binarization to reduce the $l2$ error between the real-valued and the binarized weights. The binarization of a weight $W \in \mathcal{R}^{n \times m}$ can be formulated as:

<span id="equation-01"></span>

$$
\widetilde{W} = \mathrm{Sign}(W - \alpha),
$$

<span id="equation-02"></span>

$$
\mathrm{Sign}(W_{ij}) = \left\{
\begin{aligned}
&+1, \quad &&\text{if } W_{ij} > 0, \\
&-1, \quad &&\text{if } W_{ij} \leq 0,
\end{aligned}
\right.
$$

<span id="equation-03"></span>

$$
\alpha = \frac{1}{nm}\sum_{ij} W_{ij}
$$

We further quantize the activations to $b$-bit precision. Following [Det22], we use absmax quantization, which scales activations into the range $[-Q_b, Q_b]$ ($Q_b=2^{b-1}$) by multiplying with $Q_b$ and dividing by the absolute maximum of the input matrix:

<span id="equation-04"></span>

$$
\widetilde{x} = \mathrm{Quant}(x) = \mathrm{Clip}\left(x \times \frac{Q_b}{\gamma}, -Q_b+\epsilon, Q_b-\epsilon\right),
$$

<span id="equation-05"></span>

$$
\mathrm{Clip}(x, a, b) = \max(a, \min(b, x)), \quad \gamma = \|x\|_{\infty},
$$

where $\epsilon$ is a small floating-point number that prevents overflow when performing the clipping.

For the activations before the non-linear functions (e.g., ReLU), we scale them into the range $[0, Q_b]$ by subtracting the minimum of the inputs so that all values are non-negative:

<span id="equation-06"></span>

$$
\widetilde{x} = \mathrm{Quant}(x) = \mathrm{Clip}\left((x-\eta) \times \frac{Q_b}{\gamma}, \epsilon, Q_b-\epsilon\right), \quad \eta = \min_{ij} x_{ij}.
$$

In this work, we quantize the activation to 8-bit and leave lower precision in future work. Moreover, the quantization is performed per tensor during training while per token during inference for both stability and efficiency.

With the above quantization equations, the matrix multiplication can be written as:

<span id="equation-07"></span>

$$
y = \widetilde{W} \widetilde{x}
$$

We assume that the elements in $W$ and $x$ are mutually independent and share the same distribution, and $W$ and $x$ are independent of each other. Then the variance of the output $y$ is estimated as:

<span id="equation-08"></span>

$$
\mathrm{Var}(y) = n \mathrm{Var}(\widetilde{w} \widetilde{x})
$$

<span id="equation-09"></span>

$$
=n E[\widetilde{w}^2] E[\widetilde{x}^2]
$$

<span id="equation-10"></span>

$$
=n \beta^2 E[\widetilde{x}^2] \approx E[\widetilde{x}^2]
$$

For the full-precision computation, the variance of the output $\mathrm{Var}(y)$ is at the scale of $1$ with the standard initialization methods (e.g., Kaiming initialization or Xavier initialization), which has a great benefit to the training stability. To preserve the variance after quantization, we introduce a LayerNorm [Ba16] function before the activation quantization. In this way, the variance of the output $y$ is then estimated as $\mathrm{Var}(y) \approx E[\mathrm{LN}(\widetilde{x})^2] = 1$, which has the same magnitude as the full-precision counterpart $\mathrm{Var}(y)$. In the context of Transformers, it has the exact implementation as `SubLN` [Wan22l]. With `SubLN` and the quantization methods above, we have `BitLinear`, which is formulated as:

<span id="equation-11"></span>

$$
y = \widetilde{W} \widetilde{x} = \widetilde{W}\,\mathrm{Quant}(\mathrm{LN}(x)) \times \frac{\beta\gamma}{Q_b}
$$

<span id="equation-12"></span>

$$
\mathrm{LN}(x) = \frac{x-E(x)}{\sqrt{\mathrm{Var}(x)+\epsilon}}, \quad \beta = \frac{1}{nm}\|W\|_1
$$

[Figure 2](#figure-02) provides an illustration of the computation flow of `BitLinear`. After the SubLN operation, the activations are quantized with the absmax function. The matrix multiplication is performed between the 1-bit weights and the quantized activations. The output activations are rescaled with $\{\beta, \gamma\}$ to dequantize them to the original precision.

**Model parallelism with Group Quantization and Normalization.** One essential technique to scale up large language models is model parallelism [Sho19], which partitions the matrix multiplication on multiple devices. A prerequisite for the existing model parallelism approaches is that the tensors are independent along the partition dimension. However, all of the parameters $\alpha$, $\beta$, $\gamma$, and $\eta$ are calculated from the whole tensors, breaking the independent prerequisite. One solution is to introduce one *all-reduce* operation for each parameter. However, even though the communication for each parameter is small, the amount of synchronization is growing as the model becomes deeper, which significantly slows the forward pass. The problem also exists in `SubLN`, where the mean and the variance should be estimated across the partition dimension.

To this end, we propose a simple approach that makes the model parallelism more efficient. We divide the weights and activations into groups and then independently estimate each group's parameters. This way, the parameters can be calculated locally without requiring additional communication. This approach, called Group Quantization, is formulated as follows:

For a weight matrix $W \in \mathcal{R}^{n \times m}$, we divide it into $G$ groups along the partition dimension, and each group has a size of $\frac{n}{G} \times m$. We then estimate the parameters for each group independently:

<span id="equation-13"></span>

$$
\alpha_g = \frac{G}{nm}\sum_{ij} W_{ij}^{(g)}, \quad \beta_g = \frac{G}{nm}\|W^{(g)}\|_1,
$$

where $W^{(g)}$ denotes the $g$-th group of the weight matrix. Similarly, for the activations, we can divide the input matrix $x \in \mathcal{R}^{n \times m}$ into $G$ groups and calculate the parameters for each group:

<span id="equation-14"></span>

$$
\gamma_g = \|x^{(g)}\|_{\infty}, \quad \eta_g = \min_{ij} x_{ij}^{(g)}
$$

For LN, we can apply the group normalization technique [Wu20a] to compute the mean and variance for each group independently:

<span id="equation-15"></span>

$$
\mathrm{LN}(x^{(g)}) = \frac{x^{(g)}-E(x^{(g)})}{\sqrt{\mathrm{Var}(x^{(g)})+\epsilon}}
$$

In this way, we can efficiently implement model parallelism with Group Quantization and Normalization, which requires no additional communication and can scale to large language models.

<span id="section-2-2"></span>

### 2.2 Model Training

**Straight-through estimator.** To train our 1-bit model, we employ the straight-through estimator (STE) [Ben13] to approximate the gradient during backpropagation. This method bypasses the non-differentiable functions, such as the Sign ([Equation 2](#equation-02)) and Clip ([Equation 5](#equation-05)) functions, during the backward pass. STE allows gradients to flow through the network without being affected by these non-differentiable functions, making it possible to train our quantized model.

**Mixed precision training.** While the weights and the activations are quantized to low precision, the gradients and the optimizer states are stored in high precision to ensure training stability and accuracy. Following the previous work [Liu21f], we maintain a latent weight in a high-precision format for the learnable parameters to accumulate the parameter updates. The latent weights are binarized on the fly during the forward pass and never used for the inference process.

**Large learning rate.** One challenge for the optimization is that a small update on the latent weights often makes no difference in the 1-bit weights. This results in a biased gradient and update which are estimated based on the 1-bit weights. This problem is even worse at the beginning of the training, where the models are supposed to converge as fast as possible. To address this challenge, we explore various methods, concluding that increasing the learning rate is the simplest and best way to accelerate the optimization. Our experiments show that BitNet benefits from a large learning rate in terms of convergence, while the FP16 Transformer diverges at the beginning of training with the same learning rate. More details can be found in [Section 3](#section-3).

<span id="table-01"></span>

![Energy consumption of BitNet and Transformer at three model sizes](../../papers/bitnet/table-01.png)

**Table 1.** Energy consumption of BitNet and Transformer varying different model size. Results are reported with 512 as input length.

<span id="section-2-3"></span>

### 2.3 Computational Efficiency

We estimate the computational efficiency of BitNet in terms of both arithmetic operations energy and memory footprint. We mainly focus on the calculation for the matrix multiplication, since it contributes the most to the cost of large language models.

**Arithmetic operations energy.** According to the energy model in [Hor14, Zha22g], the energy consumption for different arithmetic operations can be estimated as follows:

<span id="table-02"></span>

![ADD and MUL energy consumption for FP32, FP16, and INT8 at 45nm and 7nm](../../papers/bitnet/table-02.png)

**Table 2.** ADD and MUL energy consumption [Hor14, Zha22g] for different bit representations at 45nm and 7nm process nodes.

In vanilla Transformers, for matrix multiplication with dimensions $m \times n$ and $n \times p$, the energy consumption can be calculated as follows:

<span id="equation-16"></span>

$$
E_{\mathrm{add}} = m \times (n-1) \times p \times \hat{E}_{\mathrm{add}}
$$

<span id="equation-17"></span>

$$
E_{\mathrm{mul}} = m \times n \times p \times \hat{E}_{\mathrm{mul}}
$$

For BitNet, the energy consumption of the matrix multiplication is dominated by the addition operations, as the weights are 1-bit. The multiplication operations are only applied to scale the output with the scalars $\beta$ and $\frac{\gamma}{Q_b}$, so the energy consumption for multiplication can be computed as:

<span id="equation-18"></span>

$$
E_{\mathrm{mul}} = (m \times p + m \times n) \times \hat{E}_{\mathrm{mul}}
$$

which is significantly smaller than that in Transformers. The energy savings of W1A8 BitNet compared to a full-precision (32-32) and half-precision (16-16) Transformer are shown in [Table 1](#table-01). As can be seen, BitNet provides significant energy savings, especially for the multiplication operations, which are the major component of the matrix multiplication energy consumption.

<span id="section-3"></span>

## 3 Comparison with FP16 Transformers

<span id="section-3-1"></span>

### 3.1 Setup

We train a series of autoregressive language models with BitNet of various scales, ranging from 125M to 30B. The models are trained on an English-language corpus, which consists of the Pile dataset, Common Crawl snapshots, RealNews, and CC-Stories datasets. We use the Sentencpiece tokenizer to preprocess data and the vocabulary size is 16K. Besides BitNet, we also train the Transformer baselines with the same datasets and settings for a fair comparison. More details can be found in the appendix.

<span id="section-3-2"></span>

### 3.2 Inference-Optimal Scaling Law

Neural language models have proven to scale predictably [Kap20] with vanilla Transformer architecture. The loss scales as the power law with the amount of computation used for training. This allows us to determine the optimal allocation of a computation budget as well as predict the performance of large language models from smaller models.

To study the scaling law of binarized Transformer, we start by plotting the scaling curve of both BitNet and the FP16 Transformer baseline against the parameter count. We fix the number of training tokens and vary the model sizes. [Figure 3](#figure-03) shows that the loss scaling of BitNet is similar to the FP16 Transformer, which follows a power-law. We then fit the scaling law with an irreducible loss term:

<span id="equation-19"></span>

$$
L(N)=aN^b+c
$$

To evaluate whether the scaling law can accurately predict the loss, we choose the models from 125M to 6.7B to fit the parameters in the power-law and use the law to predict the loss of 13B and 30B. It shows that the fitted scaling law predicted BitNet's loss with high accuracy. Besides, the gap between BitNet and FP16 Transformer becomes smaller as the model size grows.

While the power-law above measures the trend of the scaling of BitNet, it does not properly model the relationship between the loss and the actual compute. Previous work [Kap20, Hen20a, Hof22] estimates the compute by calculating the FLOPs. However, it does not apply to 1-bit models whose cost is dominated by integer computation. Moreover, it mainly measures the training computation rather than the inference. To have a better understanding of the scaling efficiency of neural language models, we introduce Inference-Optimal Scaling Law. It predicts the loss against the energy consumption. We focus on the inference energy cost as it scales with the usage of the model, while the training cost is only once. We estimate the energy consumption as in [Section 2.3](#section-2-3). [Figure 3](#figure-03) shows the scaling curve against the inference energy cost at 7nm process nodes. It proves that BitNet has much higher scaling efficiency. Given a fixed computation budget, BitNet achieves a significantly better loss. Meanwhile, the inference cost is much smaller to get the same performance as the FP16 models.

<span id="figure-03"></span>

![Scaling curves of BitNet and FP16 Transformers against energy and model size](../../papers/bitnet/figure-03.png)

**Figure 3.** Scaling curves of BitNet and FP16 Transformers.

<span id="section-3-3"></span>

### 3.3 Results on Downstream Tasks

In addition to the loss, we are also concerned about the capabilities with the scaling of BitNet. Compared with the loss, the capacity is more difficult to predict due to the emergent nature of neural language models. To evaluate the capabilities with the interpretable metrics, we test both the 0-shot and 4-shot results on four downstream tasks, including Hellaswag [Zel19], Winogrande [Sak20], Winograd [Lev12], and Storycloze [Mos16]. [Figure 4](#figure-04) reports the average results of BitNet and FP16 Transformer with various scales. Similar to the loss scaling curve, the performance on the downstream tasks can scale as the computation budget grows. Besides, the scaling efficiency of capabilities is much higher than the FP16 Transformer baseline, in terms of both zero-shot and few-shot performance.

<span id="figure-04"></span>

![Zero-shot and few-shot downstream performance of BitNet and FP16 Transformer](../../papers/bitnet/figure-04.png)

**Figure 4.** Zero-shot (Left) and few-shot (Right) performance of BitNet and FP16 Transformer against the inference cost.

<span id="section-3-4"></span>

### 3.4 Stability Test

The major challenge for training low-bit Transformers is the stability in optimization. Therefore, we perform stability tests for both BitNet and the FP16 baseline by training a series of models with varying peak learning rates. [Figure 5a](#figure-05) illustrates the results of the stability test. It shows that BitNet can converge with a large learning rate while FP16 Transformer can not, demonstrating better training stability of BitNet. This advantage in optimization enables the training with larger learning rates. [Figure 5b](#figure-05) shows that BitNet can benefit from the increase in learning rate, achieving better convergence in terms of PPL.

<span id="figure-05"></span>

![Training stability and learning-rate convergence of BitNet and FP16 Transformer](../../papers/bitnet/figure-05.png)

**Figure 5.** BitNet is more stable than FP16 Transformer with a same learning rate (Left). The training stability enables BitNet a larger learning rate, resulting in better convergence (Right).

<span id="section-4"></span>

## 4 Comparison with Post-training Quantization

<span id="section-4-1"></span>

### 4.1 Setup

We train BitNet with the same setup as described in [Section 3.1](#section-3-1). We compare BitNet with state-of-the-art quantization methods, including Absmax [Det22], SmoothQuant [Xia23], GPTQ [Fra23], and QuIP [Che24b]. These methods are post-training quantization over an FP16 Transformer model, which follows the same training setting and data as BitNet. Among them, Absmax and SmoothQuant quantize both the weights and the activations, while GPTQ and QuIP only reduce the precision of weights. We apply the methods to various quantization levels. For the weight-only quantization (i.e., GPTQ and QuIP), we experiment with W4A16 and W2A16. For weight-and-activation quantization (i.e., Absmax and SmoothQuant), we use them to quantize the FP16 Transformers to W8A8, W4A4, and W1A8. Our implementation of BitNet is binary weight 8-bit activation (W1A8), which has lower or equal bits than the baselines.

<span id="section-4-2"></span>

### 4.2 Results

[Table 3](#table-03) presents a detailed comparative analysis of the zero-shot performance of our proposed method, BitNet, against various baseline approaches on four benchmark datasets, namely Winogrande, Winograd, Storycloze, and Hellaswag. All models have the model sizes of 6.7B for a fair comparison. The methods are evaluated across several weight bit levels, spanning from 16 down to 1. Besides the zero-shot accuracy on the downstream tasks, the evaluation metrics include language model perplexity on the validation set, which provides a comprehensive understanding of each method's performance.

The results demonstrate the effectiveness of BitNet in achieving competitive performance levels compared to the baseline approaches, particularly for lower bit levels. The zero-shot scores of BitNet are comparable with the 8-bit models, while the inference cost is much lower. For the 4-bit models, the weight-only quantization methods outperform the weight-and-activation quantizers, mainly because the activation is more difficult to quantify. BitNet, as a 1-bit model, significantly achieves better results than both the weight-and-activation quantization methods and the weight-only methods. As for the lower-bit models, BitNet has consistently superior scores over all baselines. This proves the advantages of the quantization-aware training approaches over the post-training quantization methods. [Figure 6](#figure-06) summarizes both the zero-shot accuracy and few-shot accuracy of our method and the baselines while scaling up the model size from 1.3B to 6.7B. It proves that the advantage is consistent across different scales.

<span id="figure-06"></span>

![Zero-shot and few-shot results for BitNet and post-training quantization baselines](../../papers/bitnet/figure-06.png)

**Figure 6.** Zero-shot (Left) and few-shot (Right) results for BitNet and the post-training quantization baselines on downstream tasks.

<span id="table-03"></span>

![Zero-shot results for BitNet and post-training quantization baselines](../../papers/bitnet/table-03.png)

**Table 3.** Zero-shot results for BitNet and the baselines (`PTQ`: Post-training quantization, `WGe`: Winogrande, `WG`: Winograd, `SC`: Storycloze, and `HS`: Hellaswag dataset).

<span id="section-5"></span>

## 5 Ablation Studies

In [Table 4](#table-04), we present an ablation study of our compared with several alternative approaches. We ablate the effect of our choices in activation quantization approaches as well as the techniques to stabilize the model training. BitNet implement absmax to quantize the activation and use `SubLN` for training stability. One quantization alternative is the elastic function [Liu22], which dynamically adjusts the scales with learnable parameters. In our experiments, we find that absmax has better performance than the elastic function. Besides, the absmax function leads to more stable training, which enables a larger learning rate for BitNet. We further compare `SubLN` with the Pre-LN and the BMT architecture [Zha23s]. Pre-LN is the default architecture for GPT pertaining, while BMT has proven to improve the stability of binarized models. Our experiments show that `SubLN` outperforms both Pre-LN and BMT. Therefore, we choose absmax and `SubLN` as the implementation in BitNet.

<span id="table-04"></span>

![Ablation of BitNet quantization and normalization choices](../../papers/bitnet/table-04.png)

**Table 4.** Ablation of BitNet (`WGe`: Winogrande, `WG`: Winograd, `SC`: Storycloze, and `HS`: Hellaswag dataset). Elastic is an activation quantization method from [Liu22], while BMT is the architecture from [Zha23s] to stabilize the training of low-bit models.

<span id="section-6"></span>

## 6 Conclusion and Future Work

We present BitNet, a novel 1-bit Transformer architecture for large language models. Our approach is designed to be scalable and stable, with the ability to handle large language models efficiently. The experimental results demonstrate that BitNet achieves competitive performance in terms of both perplexity and downstream task performance, while significantly reducing memory footprint and energy consumption compared to the baselines. Moreover, BitNet follows a scaling law similar to that of full-precision Transformers, indicating that it can be effectively scaled to even larger language models with potential benefits in terms of performance and efficiency. In the future, we would like to scale up BitNet in terms of model size and training steps. We are also interested in applying BitNet in other architectures (e.g., RetNet [Sun23a]) for training large language models.

<span id="section-7"></span>

## 7 Hyperparameters

<span id="table-05"></span>

![Model configurations for BitNet scaling experiments](../../papers/bitnet/table-05.png)

**Table 5.** Model configuration for BitNet in the scaling experiments.

<span id="table-06"></span>

![Hyperparameters for BitNet and FP16 Transformer scaling experiments](../../papers/bitnet/table-06.png)

**Table 6.** Hyperparameters for BitNet and the FP16 Transformers in the scaling experiments. For 13B and 30B model, we set weight decay to 0.05 for training stability.

<span id="table-07"></span>

![Hyperparameters for the BitNet and FP16 Transformer stability test](../../papers/bitnet/table-07.png)

**Table 7.** Hyperparameters for the stability test of BitNet and FP16 Transformer.

<span id="table-08"></span>

![Hyperparameters for BitNet ablations](../../papers/bitnet/table-08.png)

**Table 8.** Hyperparameters for the ablations of BitNet.
