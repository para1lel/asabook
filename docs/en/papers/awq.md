---
title: 'AWQ'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/awq/
---

> Ji Lin, [Jiaming Tang](https://dblp.org/pid/277/8890), [Haotian Tang](https://dblp.org/pid/245/0058), [Shang Yang](https://dblp.org/pid/79/9960), Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, [Xingyu Dang](https://dblp.org/pid/348/8880), Chuang Gan, and Song Han. First submitted to arXiv on June 1, 2023; current version v6. [AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978). [Original PDF](/paper/awq.pdf). [TeX source](https://export.arxiv.org/e-print/2306.00978). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Large language models (LLMs) have shown excellent performance on various tasks, but the astronomical model size raises the hardware barrier for serving (memory size) and slows down token generation (memory bandwidth). In this paper, we propose Activation-aware Weight Quantization (AWQ), a hardware-friendly approach for LLM low-bit weight-only quantization. Our method is based on the observation that weights are not equally important: protecting *only 1%* of salient weights can greatly reduce quantization error. We then propose to search for the optimal per-channel scaling that protects the salient weights by observing the *activation, not weights*. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs’ generalization ability on different domains and modalities, without overfitting to the calibration set. AWQ outperforms existing work on various language modeling and domain-specific benchmarks. Thanks to better generalization, it achieves excellent quantization performance for *instruction-tuned* LMs and, for the first time, *multi-modal* LMs. Alongside AWQ, we implement an efficient and flexible inference framework tailored for LLMs on the edge, offering more than 3$\times$ speedup over the Huggingface FP16 implementation on both desktop and mobile GPUs. It also democratizes the deployment of the 70B Llama-2 model on mobile GPU (NVIDIA Jetson Orin 64GB).

 [+1]

## 1 Introduction

Large language models (LLMs) based on transformers [Advanf17] have shown excellent performance on various benchmarks [Systea01, Open22, Xive23, Xivr22]. However, the large model size leads to the high serving costs. For example, GPT-3 has 175B parameters, which is 350GB in FP16, while the latest H100 GPU only has 96GB memory, let alone edge devices.

Low-bit weight quantization for LLMs can save memory but is hard. Quantization-aware training (QAT) is not practical due to the high training cost, while post-training quantization (PTQ) suffers from large accuracy degradation under a low-bit setting. The closest work is GPTQ [Xivo22], which uses second-order information to perform error compensation. It may over-fit the calibration set during reconstruction, distorting the learned features on out-of-distribution domains ([Figure 6](#figure-06)), which could be problematic since LLMs are *generalist* models.

In this paper, we propose Activation-aware Weight Quantization (AWQ), a hardware-friendly low-bit weight-only quantization method for LLMs. Our method is based on the observation that *weights are not equally important* for LLMs’ performance. There is a small fraction (0.1%-1%) of *salient* weights; skipping the quantization of these salient weights will significantly reduce the quantization loss ([Table 1](#table-01)). To find the salient weight channels, the insight is that we should refer to the *activation* distribution instead of the *weight* distribution, despite we are doing *weight-only* quantization: weight channels corresponding to larger activation magnitudes are more salient since they process more important features. To avoid the hardware-inefficient mixed-precision implementation, we analyze the error from weight quantization and derive that *scaling up the salient channels can reduce their relative quantization error* (Equation [2](#S2.E2 "In 2.2 Protecting Salient Weights by Activation-aware Scaling ‣ 2 AWQ: Activation-aware Weight Quantization ‣ AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration")). Following the intuition, we designed a per-channel scaling method to automatically search for the optimal scaling that minimizes the quantization error under full-weight quantization. AWQ does not rely on any backpropagation or reconstruction, so it can well preserve LLMs’ generalization ability on various domains and modalities without overfitting to the calibration set. Furthermore, we implemented an efficient serving framework to convert theoretical memory savings from AWQ to practical speedup. Our framework takes advantage of kernel fusion to minimize the inference overhead (*e.g*., intermediate DRAM access and kernel launch overhead), so that we can better realize the speed up from quantizing linear layers (AWQ is applied to linear layers which consist most of the parameters).

Experiments show that AWQ outperforms existing work on various tasks for different model families (*e.g*., LLaMA [Xive23], OPT [Open22]) and model sizes. Thanks to better generalization, it also achieves good quantization performance for *instruction-tuned* LMs (*e.g*., Vicuna) and, for the first time, *multi-modal* LMs (OpenFlamingo [March23]). With our efficient system implementation, we consistently observe a 3.2-3.3$\times$ average speedup compared to the FP16 implementation by Huggingface across a diverse spectrum of LLMs. Furthermore, it facilitates effortless deployment of the Llama-2-70B model on a single NVIDIA Jetson Orin with 64GB of memory. It also democratizes LLMs with up to 13 billion parameters at an interactive pace of 30 tokens per second on a laptop RTX 4070 GPU with only 8GB of memory.

AWQ has been widely adopted by various open-source LLM serving solutions including [FastChat](https://github.com/lm-sys/FastChat/blob/main/docs/awq.md), [vLLM](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/quantization_utils/awq.py), [HuggingFace TGI](https://github.com/huggingface/text-generation-inference/pull/1054), [LMDeploy](https://github.com/InternLM/lmdeploy), etc.

## 2 AWQ: Activation-aware Weight Quantization

*Quantization* maps a floating-point number into lower-bit integers. It is an effective method to reduce the model size and inference costs of LLMs [Xivm22, Xivo22, Effici22, Xivt22]. In this section, we first propose a weight-only quantization method to improve accuracy *without training/regression* by protecting more "important" weights. And then develop a data-driven method to search for the optimal scaling that reduces quantization errors ([Figure 1](#figure-01)).

<span id="figure-01"></span>

![Refer to caption](../../papers/awq/figure-01.png)

**Figure 1.** We observe that we can find 1% of the salient weights in LLMs by observing the *activation distribution* (middle). Keeping the salient weights in FP16 can significantly improve the quantized performance (PPL from 43.2 (left) to 13.0 (middle)), but the mixed-precision format is not hardware-efficient. We follow the activation-awareness principle and propose AWQ (right). AWQ performs per-channel scaling to protect the salient weights, leading to reduced quantized error. PPL is measured with OPT-6.7B under INT3-g128 quantization.

### 2.1 Improving LLM Quantization by Preserving 1% Salient Weights

<span id="table-01"></span>

| PPL $\downarrow$ | FP16 | RTN | FP16% (based on act.) |  |  | FP16% (based on W) |  |  | FP16% (random) |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| (w3-g128) | 0.1% | 1% | 3% | 0.1% | 1% | 3% | 0.1% | 1% | 3% |  |  |
| OPT-1.3B | 14.62 | 119.00 | 25.03 | 16.91 | 16.68 | 108.71 | 98.55 | 98.08 | 119.76 | 109.38 | 61.49 |
| OPT-6.7B | 10.86 | 23.54 | 11.58 | 11.39 | 11.36 | 23.41 | 22.37 | 22.45 | 23.54 | 24.23 | 24.22 |
| OPT-13B | 10.13 | 46.04 | 10.51 | 10.43 | 10.42 | 46.07 | 48.96 | 54.49 | 44.87 | 42.00 | 39.71 |

**Table 1.** Keeping a small fraction of weights (0.1%-1%) in FP16 significantly improves the performance of the quantized models over round-to-nearest (RTN). It is only effective when we select the important weights in FP16 by looking at *activation* distribution instead of *weight* distribution. We highlight results with a decent perplexity in green. We used INT3 quantization with a group size of 128 and measured the WikiText perplexity ($\downarrow$).

We observe that the weights of LLMs are *not equally important*: there is a small fraction of *salient* weights that are much more important for LLMs’ performance compared to others. Skipping the quantization of these salient weights can help bridge the performance degradation due to the quantization loss *without* any training or regression ([Figure 1](#figure-01)(b)). To verify the idea, we benchmark the performance of quantized LLMs when skipping part of the weight channels in [Table 1](#table-01). We measured the performance of INT3 quantized models while keeping some ratios of weight channels in FP16. A widely used method to determine the importance of weights is to look at its magnitude or $L_{2}$-norm [Advanb15, Xivbj18]. But we find skipping the weight channels with large norm (*i.e*., FP16% (based on W)) does not significantly improve the quantized performance, leading to a similar marginal improvement as random selection. Interestingly, selecting weights based on *activation magnitude* can significantly improve the performance: keeping only 0.1%-1% of the channels corresponding to larger activation significantly improves the quantized performance, even matching a strong reconstruction-based method GPTQ [Xivo22]. We hypothesize that the input features with larger magnitudes are generally more important. Keeping the corresponding weights in FP16 can preserve those features, which contributes to better model performance.

Limitations: Despite keeping 0.1% of weights in FP16 can improve the quantized performance without a noticeable increase in model size (measured in total bits), such a mixed-precision data type will make the system implementation difficult. We need to come up with a method to protect the important weights without actually keeping them as FP16.

### 2.2 Protecting Salient Weights by Activation-aware Scaling

We propose an alternative method to reduce the quantization error of the salient weight by *per-channel scaling*, which does not suffer from the hardware inefficiency issue.

Analyzing the quantization error. We start by analyzing the error from weight-only quantization. Consider a group/block of weight $\mathbf{w}$; the linear operation can be written as $y=\mathbf{w}\mathbf{x}$, and the quantized counterpart is $y=Q(\mathbf{w})\mathbf{x}$. Specifically, the quantization function is defined as:

$$
Q(\mathbf{w})=\Delta\cdot\mathrm{Round}(\frac{\mathbf{w}}{\Delta}),\quad\Delta=\frac{\max(|\mathbf{w}|)}{2^{N-1}},\tag{1}
$$

where $N$ is the number of quantization bits, and $\Delta$ is the quantization scaler determined by the absolute maximum value. Now consider a weight element $w\in\mathbf{w}$, if we multiply $w$ with $s>1$ and the inversely scale $x$, we will have $Q(w\cdot s)(x/s)$, which is:

$$
Q(w\cdot s)\cdot\frac{x}{s}=\Delta^{ {}^{\prime}}\cdot\mathrm{Round}(\frac{\mathrm{ws}}{\Delta})\cdot x\cdot\frac{1}{s},\tag{2}
$$

where $\Delta^{ {}^{\prime}}$ is the new quantization scaler after applying $s$. We empirically find that: (1) The expected error from $\mathrm{Round}(\cdot)$ (denoted as $\mathrm{RoundErr}$) does not vary: since the round function maps a floating-point number to an integer, the error is roughly uniformly distributed from 0-0.5, resulting in an average error of  0.25; (2) Scaling up a single element $w$ usually does not change the extreme value from the group $\mathbf{w}$. Therefore we have $\Delta^{ {}^{\prime}}\approx\Delta$; (3) The error from equation [2](#S2.E2 "In 2.2 Protecting Salient Weights by Activation-aware Scaling ‣ 2 AWQ: Activation-aware Weight Quantization ‣ AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration") can be expressed as $\mathrm{Err}^{ {}^{\prime}}=\Delta^{ {}^{\prime}}\cdot \mathrm{RoundErr}\cdot\frac{1}{s}$, the ratio compared to the original error $\mathrm{RoundErr}$ is $\frac{\Delta^{ {}^{\prime}}}{\Delta}\cdot\frac{1}{s}$. Given $\Delta^{ {}^{\prime}}\approx\Delta$ and $s>1$, the relative error is smaller for the salient weight $w$.

To verify the idea, we multiply the 1% salient channels with $s>1$ for the OPT-6.7B model, and measure the change in $\Delta$ for each group in [Table 2](#table-02). We find that scaling up the salient channels is quite effective: the perplexity improves from 23.54 for $s=1$ (simply RTN) to 11.92 for $s=2$. As $s$ goes larger, the percentage of changed $\Delta$ generally gets larger, but the proportion is still quite small for $s<2$; the relative error for the salient channels continues to go smaller as $s$ increases. Nonetheless, the best PPL actually appears at $s=2$. This is because if we use a very large $s$, it will increase the relative error for the *non-salient* channels when $\Delta$ increases (the error of non-salient channels will be amplified by $\frac{\Delta^{ {}^{\prime}}}{\Delta}$, and the ratio is larger than 1 for 21.2% of the channels under $s=4$), which can damage the model’s overall accuracy. Therefore, we need to also consider the error from the non-salient channels when protecting salient ones.

<span id="table-02"></span>

| OPT-6.7B | $s=1$ | $s=1.25$ | $s=1.5$ | $s=2$ | $s=4$ |
| --- | --- | --- | --- | --- | --- |
| proportion of $\Delta^{ {}^{\prime}}\neq\Delta$ | 0% | 2.8% | 4.4% | 8.2% | 21.2% |
| average $\Delta^{ {}^{\prime}}/\Delta$ | 1 | 1.005 | 1.013 | 1.038 | 1.213 |
| average $\frac{\Delta^{ {}^{\prime}}}{\Delta}\cdot\frac{1}{s}$ (error reduction rate) | 1 | 0.804 | 0.676 | 0.519 | 0.303 |
| Wiki-2 PPL | 23.54 | 12.87 | 12.48 | 11.92 | 12.36 |

**Table 2.** Statistics when multiplying the 1% salient channels by $s>1$. Scaling up the salient channels significantly improves the perplexity (23.54 to 11.92). As $s$ goes larger, the percentage of changed $\Delta$ increases, and the error reduction rate for salient channels also increases. However, the best perplexity is achieved at $s=2$, since further increasing $s$ will increase the quantization error for *non-salient* channels.

<span id="table-03"></span>

| OPT / PPL$\downarrow$ |  | 1.3B | 2.7B | 6.7B | 13B | 30B |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 | - | 14.62 | 12.47 | 10.86 | 10.13 | 9.56 |
| INT3 g128 | RTN | 119.47 | 298.00 | 23.54 | 46.04 | 18.80 |
| 1% FP16 | 16.91 | 13.69 | 11.39 | 10.43 | 9.85 |  |
| $s=2$ | 18.63 | 14.94 | 11.92 | 10.80 | 10.32 |  |
| AWQ | 16.32 | 13.58 | 11.39 | 10.56 | 9.77 |  |

**Table 3.** AWQ protects salient weights and reduces quantization error by using a scaling-based method. It consistently outperforms Round-to-nearest quantization (RTN) and achieves comparable performance as mixed-precision (1% FP16) while being more hardware-friendly.

Searching to scale. To consider both salient and non-salient weights, we choose to automatically search for an optimal (per input channel) scaling factor that minimizes the output difference after quantization for a certain layer. Formally, we want to optimize the following objective:

$$
\mathbf{s}^{*}=\mathrm{arg\,min}_{\mathbf{s}}\mathcal{L}(\mathbf{s}),\quad\mathcal{L}(\mathbf{s})=\| Q(\mathbf{W}\cdot\mathbf{s})(\mathbf{s^{-1}}\cdot\mathbf{X})-\mathbf{W}\mathbf{X}\|\tag{3}
$$

Here $Q$ means the weight quantization function (*e.g*., INT3/INT4 quantization with group size 128), $\mathbf{W}$ is the original weights in FP16, and $\mathbf{X}$ is the input features cached from a small calibration set (we take a small calibration set from he pre-training dataset in order not to overfit to a specific task). $\mathbf{s}$ is a per-(input) channel scaling factor; for $\mathbf{s^{-1}}\cdot\mathbf{X}$, it can usually be fused into the previous operator [Xivs22, Xivt22]. Since the quantization function is not differentiable, we are not able to directly optimize the problem with vanilla backpropagation. There are some techniques relying on approximated gradients [Xivf13, Xivn02], which we found still suffers from unstable convergence.

To make the process more stable, we define a *search space* for the optimal scale by analyzing the factors that will affect the choice of scaling factor. As shown in the last section, the saliency of weight channels is actually determined by the activation scale (thus “activation-awareness”). Therefore, we simply use a very simple search space:

$$
\mathbf{s}=\mathbf{s_{X}}^{\alpha},\quad\alpha^{*}=\mathrm{arg\,min}_{\alpha}\mathcal{L}(\mathbf{s_{X}}^{\alpha})\tag{4}
$$

$\mathbf{s}$ is only related to the magnitude of activation $\mathbf{s_{X}}$, and we use a single hyper-parameter $\alpha$ to balance between the protection of salient and non-salient channels. We can find the best $\alpha$ by a fast grid search over the interval of $[0,1]$ ($0$ means we do not scale; $1$ corresponds to the most aggressive scaling). We further apply weight clipping also by minimizing the MSE error, since clipping the weights can further help to reduce $\Delta^{ {}^{\prime}}$ in Equation [2](#S2.E2 "In 2.2 Protecting Salient Weights by Activation-aware Scaling ‣ 2 AWQ: Activation-aware Weight Quantization ‣ AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration"); thus reducing quantization error. We provide an ablation study on OPT models under INT3-g128 quantization in [Table 3](#table-03); AWQ consistently outperforms round-to-nearest quantization (RTN) and achieves comparable performance as mixed-precision (1% FP16) while being more hardware-friendly.

Advantages. Our method does not rely on any regression [Xivo22] or backpropagation, which is required by many quantization-aware training methods. It has minimal reliance on the calibration set since we only measure the average magnitude per channel, thus preventing over-fitting ([Figure 6](#figure-06)). Therefore, our method requires fewer data for the quantization process and can preserve LLMs’ knowledge outside of the calibration set’s distribution. See Section [3.3](#S3.SS3 "3.3 Analysis ‣ 3 Experiments ‣ AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration") for more details.

## 3 Experiments

### 3.1 Settings

#### Quantization.

We focus on *weight-only grouped* quantization in this work. As shown in previous work [Xivn22, Xivo22], grouped quantization is always helpful for improving performance/model size trade-off. We used a group size of 128 throughout the work, except otherwise specified. We focus on INT4/INT3 quantization since they are able to mostly preserve the LLMs’ performance [Xivn22]. For AWQ, we used a small calibration set from the Pile [Xiva20] dataset in order not to overfit to a specific downstream domain. We used a grid size of 20 to search for the optimal $\alpha$ in Equation [4](#S2.E4 "In 2.2 Protecting Salient Weights by Activation-aware Scaling ‣ 2 AWQ: Activation-aware Weight Quantization ‣ AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration").

#### Models.

We benchmarked our method on LLaMA [Xive23] and OPT [Open22] families. There are other open LLMs like BLOOM [Xivr22], but they are generally worse in quality, so we do not include them in our study. We further benchmark an instruction-tuned model Vicuna [Marcha23] and visual language models OpenFlamingo-9B [March23] and LLaVA-13B [Visual23] to demonstrate the generability of our method.

#### Evaluations.

Following previous literature [Xivm22, Xivt22, Xivo22, Xivn22, Effici22], we mainly profiled the quantized models on language modeling tasks (perplexity evaluation on WikiText-2 [Pointa16]) since perplexity can stably reflect the LLM’s performance [Xivn22].

#### Baselines.

Our primary baseline is vanilla round-to-nearest quantization (RTN). It is actually quite strong when using a small group size like 128 [Xivo22, Xivn22]. We also compare with a state-of-the-art method GPTQ [Xivo22] for LLM weight quantization. For GPTQ, we also compare with an updated version that uses a “reorder” trick (denoted as GPTQ-Reorder or GPTQ-R). Other techniques like ZeroQuant [Effici22], AdaRound [PMLRf20], and BRECQ [Xivaw21] rely on backpropagation to update the quantized weights, which may not easily scale up to large model sizes; they also do not outperform GPTQ [Xivo22], thus not included for study.

### 3.2 Evaluation

#### Results on LLaMA models.

We focus our study on LLaMA models (LLaMA [Xive23] and Llama-2 [Xivf23]) due to their superior performance compared to other open-source LLMs [Open22, Xivr22]; it is also the foundation of many popular open-source models [Stanfo23, Marcha23]. We evaluate the perplexity before and after quantization in [Table 4](#table-04). We can see that AWQ consistently outperforms round-to-nearest (RTN) and GPTQ [Xivo22] (w/ and w/o reordering) across different model scales (7B-70B) and generations.

<span id="table-04"></span>

| PPL$\downarrow$ |  | Llama-2 |  |  | LLaMA |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | 7B | 13B | 70B | 7B | 13B | 30B | 65B |  |
| FP16 | - | 5.47 | 4.88 | 3.32 | 5.68 | 5.09 | 4.10 | 3.53 |
| INT3 g128 | RTN | 6.66 | 5.52 | 3.98 | 7.01 | 5.88 | 4.88 | 4.24 |
| GPTQ | 6.43 | 5.48 | 3.88 | 8.81 | 5.66 | 4.88 | 4.17 |  |
| GPTQ-R | 6.42 | 5.41 | 3.86 | 6.53 | 5.64 | 4.74 | 4.21 |  |
| AWQ | 6.24 | 5.32 | 3.74 | 6.35 | 5.52 | 4.61 | 3.95 |  |
| INT4 g128 | RTN | 5.73 | 4.98 | 3.46 | 5.96 | 5.25 | 4.23 | 3.67 |
| GPTQ | 5.69 | 4.98 | 3.42 | 6.22 | 5.23 | 4.24 | 3.66 |  |
| GPTQ-R | 5.63 | 4.99 | 3.43 | 5.83 | 5.20 | 4.22 | 3.66 |  |
| AWQ | 5.60 | 4.97 | 3.41 | 5.78 | 5.19 | 4.21 | 3.62 |  |

**Table 4.** AWQ improves over round-to-nearest quantization (RTN) for different model sizes and different bit-precisions. It consistently achieves better perplexity than GPTQ (w/ and w/o reordering) on LLaMA & Llama-2 models.

<span id="figure-02"></span>

![Refer to caption](../../papers/awq/figure-02.png)

**Figure 2.** Comparing INT3-g128 quantized Vicuna models with FP16 counterparts under GPT-4 evaluation protocol [Marcha23]. More winning cases (in blue) indicate better performance. AWQ consistently improves the quantized performance compared to RTN and GPTQ [Xivo22], showing generalization to instruction-tuned models.

#### Quantization of instruction-tuned models.

Instruction tuning can significantly improve the models’ performance and usability  [Xivaz21, Xivay21, Systee22, Xivl22]. It has become an essential procedure before model deployment. We further benchmark our method’s performance on a popular instruction-tuned model Vicuna [Marcha23] in [Figure 2](#figure-02). We used the GPT-4 score to evaluate the quantized models’ performance against the FP16 counterpart on 80 sample questions [Marcha23]. We compare the responses with both orders (quantized-FP16, FP16-quantized) to get rid of the ordering effect (we found GPT-4 tends to increase the rating of the first input), leading to 160 trials. AWQ consistently improves the INT3-g128 quantized Vicuna models over RTN and GPTQ under both scales (7B and 13B), demonstrating the generability to instruction-tuned models.

<span id="table-05"></span>

| COCO (CIDEr $\uparrow$) |  | 0-shot | 4-shot | 8-shot | 16-shot | 32-shot | *$\Delta$(32-shot)* |
| --- | --- | --- | --- | --- | --- | --- | --- |
| FP16 | - | 63.73 | 72.18 | 76.95 | 79.74 | 81.70 | - |
| INT4 g128 | RTN | 60.24 | 68.07 | 72.46 | 74.09 | 77.13 | -4.57 |
| GPTQ | 59.72 | 67.68 | 72.53 | 74.98 | 74.98 | -6.72 |  |
| AWQ | 62.57 | 71.02 | 74.75 | 78.23 | 80.53 | -1.17 |  |
| INT3 g128 | RTN | 46.07 | 55.13 | 60.46 | 63.21 | 64.79 | -16.91 |
| GPTQ | 29.84 | 50.77 | 56.55 | 60.54 | 64.77 | -16.93 |  |
| AWQ | 56.33 | 64.73 | 68.79 | 72.86 | 74.47 | -7.23 |  |

**Table 5.** Quantization results of a visual language model OpenFlamingo-9B [March23] on COCO Captioning datasets. AWQ outperforms existing methods under zero-shot and various few-shot settings, demonstrating the generability to different modalities and in-context learning workloads. AWQ reduces the quantization degradation (32-shot) from 4.57 to 1.17 under INT4-g128, providing 4$\times$ model size reduction with negligible performance loss.

<span id="figure-03"></span>

![Refer to caption](../../papers/awq/figure-03.png)

**Figure 3.** Qualitative results of quantized OpenFlamingo-9B [March23] on COCO captioning dataset (4-shot, INT4-g128 quantization). Our method significantly improves the captioning quality compared to the round-to-nearest (RTN) baseline. We color the text to show the correct or wrong captions.

#### Quantization of multi-modal language models.

Large multi-modal models (LMMs) or visual language models (VLMs) are LLMs augmented with vision inputs [Systed22, Xivc23, Xivb23, Xiva23, Xivh23, Visual23]. Such models are able to perform text generation conditioned on image/video inputs. Since our method does not have the overfitting issue to the calibration set, it can be directly applied to VLMs to provide accurate and efficient quantization. We perform experiments with the OpenFlamingo-9B model [March23] (an open-source reproduction of [Systed22]) on COCO captioning [Xivaa15] dataset ([Table 5](#table-05)). We measured the average performance of 5k samples under different few-shot settings. We only quantize the language part of the model since it dominates the model size. AWQ outperforms existing methods under zero-shot and various few-shot settings, demonstrating the generability to different modalities and in-context learning workloads. It reduces the quantization degradation (32-shot) from 4.57 to 1.17 under INT4-g128, providing 4$\times$ model size reduction with negligible performance loss. We further provide some qualitative captioning results in [Figure 3](#figure-03) to show our advantage over RTN. Our method provides a push-the-button solution for LMM/VLM quantization. It is the *first* study of VLM low-bit quantization to the best of our knowledge.

<span id="figure-04"></span>

![Refer to caption](../../papers/awq/figure-04.png)

**Figure 4.** Visual reasoning examples from LLaVA-13B model [Visual23]. AWQ improves over the round-to-nearest (RTN) baseline, providing more reasonable answers. We color the text to show the correct or wrong responses.

#### Visual reasoning results.

We further provide some qualitative visual reasoning examples of the LLaVA-13B [Visual23] model in [Figure 4](#figure-04). AWQ improves the responses compared to the round-to-nearest (RTN) baseline for INT4-g128 quantization, leading to more reasonable answers. In this first example, the AWQ model can understand the meme as it resembles the Earth when looking from space, while RTN produces wrong descriptions (marked in red). In the second example, AWQ correctly answers the question (the artist of the painting), while RTN does not provide any information about the artist. In the last example, RTN falsely points out a bird in the picture, while AWQ provides more information by noticing the image is taken in a mountain area. AWQ improves the visual reasoning ability of VLMs by reducing factual errors in the responses; RTN is not good enough even for 4 bits.

<span id="table-06"></span>

| OPT / Wiki PPL$\downarrow$ |  | 1.3B | 2.7B | 6.7B | 13B | 30B |
| --- | --- | --- | --- | --- | --- | --- |
| FP16 | - | 14.62 | 12.47 | 10.86 | 10.13 | 9.56 |
| INT2 g64 | RTN | 10476 | 193210 | 7622 | 17564 | 8170 |
| GPTQ | 46.67 | 28.15 | 16.65 | 16.74 | 11.75 |  |
| AWQ +GPTQ | 35.71 | 25.70 | 15.71 | 13.25 | 11.38 |  |

**Table 6.** Our method is orthogonal to GPTQ: it further closes the performance gap under extreme low-bit quantization (INT2-g64) when combined with GPTQ. Results are WikiText-2 perplexity of OPT models.

#### Extreme low-bit quantization.

We further quantize LLM to INT2 to accommodate limited device memory ([Table 6](#table-06)). RTN completely fails, and AWQ brings significant perplexity improvement on top of GPTQ, though there is still a performance gap compared to FP16. Our method is orthogonal to GPTQ. We can combine our method with GPTQ to further improve the INT2 quantization performance, making it a more practical setting.

<span id="figure-05"></span>

![Refer to caption](../../papers/awq/figure-05.png)

**Figure 5.** AWQ provides a turn-key solution to transform the theoretical memory footprint reduction into a quantifiable speedup. As a result, AWQ is up to 3.9$\times$ and 3.5$\times$ faster than the FP16 implementation from Huggingface on 4090 (desktop GPU) and Orin (mobile GPU), respectively. AWQ also democratizes Llama-2-13B deployment on laptop GPUs (4070) with merely 8GB memory.

#### Speedup Evaluation.

In [Figure 5](#figure-05), we demonstrate the system acceleration results for AWQ. We optimize both linear layers and layers that do not have quantized weights. We conduct benchmarking experiments on RTX 4090 (desktop GPU), RTX 4070 (laptop GPU) and Jetson Orin (mobile GPU). We perform batch size = 1 inference for all LLMs using a fixed prompt length of 4 tokens. We generate 200 tokens for each inference run and calculate the median latency as the final result. As in [Figure 5](#figure-05)(a), our system brings 2.7-3.9$\times$ speedup to three families of LLMs (Llama-2, MPT and Falcon) on 4090 compared with the Huggingface FP16 implementation. Notably, on the laptop 4070 GPU with only 8GB memory, we are still able to run Llama-2-13B models at 33 tokens / second, while the FP16 implementation cannot fit 7B models.

Our system also exhibits promising performance on the NVIDIA Jetson Orin (32GB). As shown in [Figure 5](#figure-05)(b), our system achieves an interactive processing rate of 33 tokens per second when running Llama-2 models. Thanks to AWQ, even larger models such as MPT-30B can operate smoothly on this resource-constrained edge device, delivering a processing speed of 7.8 tokens per second. It’s worth noting that we implement the forward pass for all AWQ models using native PyTorch APIs, and this code is reused across various GPU architectures. Consequently, our system provides the best of both worlds: state-of-the-art inference speed and exceptional extensibility.

### 3.3 Analysis

<span id="figure-06"></span>

![Refer to caption](../../papers/awq/figure-06.png)

**Figure 6.** Left: AWQ needs a much smaller calibration set to reach a good quantized performance. It can achieve better perplexity using 10$\times$ smaller calibration set compared to GPTQ. Right: Our method is more robust to the calibration set distribution. Overall, using the same calibration and evaluation distribution works the best (PubMed-PubMed, Enron-Enron). But when using a different calibration distribution (PubMed-Enron, Enron-PubMed), AWQ only increases the perplexity by 0.5-0.6, while GPTQ has 2.3-4.9 worse perplexity. All experiments are done with the OPT-6.7B model under INT3-g128 quantization.

#### Better data-efficiency for the calibration set.

Our method requires a smaller calibration set since we do not rely on regression/backpropagation; we only measure the average activation scale from the calibration set, which is data-efficient. To demonstrate the idea, we compare the perplexity of the OPT-6.7B model with INT3-g128 quantization in [Figure 6](#figure-06) (a). AWQ needs a much smaller calibration to reach a good quantized performance; it can achieve better perplexity using 10$\times$ smaller calibration set compared to GPTQ (16 sequences *v.s.* 192 sequences).

#### Robust to the calibration set distributions.

Our method is less sensitive to the calibration set distribution since we only measure the average activation scale from the calibration set, which is more generalizable across different dataset distributions. We further benchmarked the effect of the different calibration set distributions in [Figure 6](#figure-06)(b). We took two subsets from the Pile dataset [Xiva20]: PubMed Abstracts and Enron Emails [ECML04]. We use each of the subsets as the calibration set and evaluate the quantized model on both sets (the calibration and evaluation sets are split with no overlapping; we used 1k samples for evaluation). Overall, using the same calibration and evaluation distribution works the best (PubMed-PubMed, Enron-Enron). But when using a different calibration distribution (PubMed-Enron, Enron-PubMed), AWQ only increases the perplexity by 0.5-0.6, while GPTQ has 2.3-4.9 worse perplexity. This demonstrates the robustness of AWQ to the calibration set distribution.

## 4 Related Work

#### Model quantization methods.

Quantization reduces the bit-precision of deep learning models [ICLRb16, Recogo18, Visiom19, CVPRc19, PMLRf20, Systeq20], which helps to reduce the model size and accelerate inference. Quantization techniques generally fall into two categories: quantization-aware training (QAT, which relies on backpropagation to update the quantized weights) [Xivf13, Xivav21, Xivax21, Xivbi18] and post-training quantization [Recogo18, Visiom19, PMLRf20] (PTQ, usually training-free). The QAT methods cannot easily scale up to large models like LLMs. Therefore, people usually use PTQ methods to quantize LLMs.

#### Quantization of LLMs.

People study two settings for LLM quantization: (1) W8A8 quantization, where both activation and weights are quantized to INT8 [Xivm22, Xivt22, Effici22, Pushin22, Xivg23]; (2) Low-bit weight-only quantization (*e.g*., W4A16), where only weights are quantized into low-bit integers [Xivo22, Xivn22, Xivd23, Xivq22]. We focus on the second setting in this work since it not only reduces the hardware barrier (requiring a smaller memory size) but also speeds up the token generation (remedies memory-bound workload). Apart from the vanilla round-to-nearest baseline (RTN), GPTQ [Xivo22] is the closest to our work. However, the reconstruction process of GPTQ leads to an over-fitting issue to the calibration set and may not preserve the generalist abilities of LLMs for other modalities and domains. It also requires a reordering trick to work for some models (*e.g*., LLaMA-7B [Xive23] and OPT-66B [Open22]).

#### System support for low-bit quantized LLMs.

Low-bit quantized LLMs have been a popular setting to reduce inference costs. There are some system supports to achieve a practical speed-up. GPTQ [Xivo22] provides INT3 kernels for OPT models and GPTQ-for-LLaMA extends kernel support for INT4 reordered quantization with the help of Triton [Languf19]. FlexGen [Xivd23] and llama.cpp [+2] perform group-wise INT4 quantization to reduce I/O costs and offloading. FasterTransformer [+3] implements FP16$\times$INT4 GEMM for weight-only per-tensor quantization but does not support group quantization. LUT-GEMM [Xivq22] performs bitwise computation on GPU CUDA cores with the help of lookup tables. AWQ kernels are adaptively executed on both tensor cores and CUDA cores, suitable for both context and generation phases in LLM inference. Consequently, we run state-of-the-art LLaMA models with 3.2-3.3$\times$ speedup over the FP16 implementation from Huggingface.

## 5 Conclusion

In this work, we propose Activation-aware Weight Quantization (AWQ), a simple yet effective method for low-bit weight-only LLM compression AWQ is based on the observation that weights are not equally important in LLMs and performs per-channel scaling to reduce the quantization loss of salient weights. AWQ does not over-fit the calibration set and preserves the generalist abilities of LLMs in various domains and modalities. It outperforms existing work on language modeling and can be applicable to instruction-tuned LMs and multi-modal LMs. Our system implementation further translates the theoretical memory savings achieved by AWQ into 3.2-3.3$\times$ measured speedups over the FP16 implementations from Huggingface on desktop and mobile GPUs, democratizing LLM deployment on the edge.

## Acknowledgements

We thank MIT AI Hardware Program, National Science Foundation, NVIDIA Academic Partnership Award, MIT-IBM Watson AI Lab, Amazon and MIT Science Hub, Qualcomm Innovation Fellowship, Microsoft Turing Academic Program for supporting this research.

## Appendix A Broader Impacts and Limitations

Broader impacts. In this paper, we propose a general technique to enable accurate and efficient low-bit weight-only quantization of large language models (LLMs). It makes LLMs more efficient and accessible and thus may inherit the impacts of LLMs. On the positive side, quantization helps to democratize LLMs, which helps to benefit more people (especially those with lower income). It reduces the costs and hardware barrier of deploying LLMs and facilitates edge inference of these models, addressing the data privacy issue (since we no longer need to send data to the cloud). On the negative side, LLMs may be exploited by malicious users to produce misinformation and manipulation. Quantization can not prevent such negative effects but it does not make it worse.

Limitations. In this paper, we follow previous work [Xivm22, Xivo22, Xivt22, Effici22, Xivn22] to mostly benchmark the quantized models on standard accuracy metrics like perplexity and accuracy. However, besides accuracy, there are other important metrics for LLM benchmark like robustness, fairness, bias, toxicity, helpfulness, calibration, *etc*. [Xivp22]. We think it would be helpful to perform a more holistic evaluation of quantized LLMs covering these aspects, which we leave to future work. Furthermore, we only study low-bit integer quantization of LLMs due to easier data type casting on hardware. There might be a further improvement from changing data types (*e.g*., FP4 [Xivn22]), which we do not include in the study.

## Appendix B Amount of Computation

We study the post-training quantization (PTQ) of LLMs in this work. The computation requirement is generally modest since we do not rely on any backpropagation. We used one NVIDIA A100 GPU for smaller models (<40B parameters) and 2-4 A100 GPUs for larger models due to memory limits.

The quantization process is generally fast, requiring a few GPU hours (ranging from 0.1 to 3, depending on the model size). The accuracy measurement time depends on the model and dataset sizes: testing LLaMA-65B (the biggest model we tested on multiple datasets) on 4 common sense QA tasks requires 3 GPU hours; testing it on MMLU (consisting of 57 sub-datasets) requires 5 GPU hours. The GPU hours would be smaller for smaller models and datasets (*e.g*., WikiText-2).

## Appendix C Limitation with No-group Quantization

Our method searches for good scaling to protect the salient weight channels. It works pretty well under grouped quantization, matching the same accuracy as keeping salient weights in FP16 ([Figure 1](#figure-01)). However, such a scaling-based method can only protect *one* salient channel for *each group*. It is not a problem for grouped quantization (we only need to protect 0.1%-1% of salient channels, the group size is usually small, like 128, so we need to protect fewer than 1 channel in each group on average). But for no-group quantization, we can only protect one input channel for the *entire weight*, which may not be enough to bridge the performance degradation. As shown in [Table 7](#table-07), under INT3-g128 quantization, AWQ achieves similar performance compared to keeping 1% salient weights in FP16. While under INT3 no-group quantization, there is still a noticeable gap. Nonetheless, we want to stress that the performance of no-group quantization is still far behind grouped quantization at a similar cost. Therefore, grouped quantization is a *more practical solution* for LLM compression for edge deployment and AWQ can effectively improve the quantized performance under this setting.

<span id="table-07"></span>

| PPL $\downarrow$ | FP16 | INT3 (group 128) |  |  | INT3 (no group) |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RTN | 1% FP16 | AWQ | RTN | 1% FP16 | AWQ |  |  |
| OPT-6.7B | 12.29 | 43.16 | 13.02 | 12.99 | 21160 | 14.67 | 18.11 |
| LLaMA-7B | 9.49 | 12.10 | 10.77 | 10.82 | 50.45 | 14.06 | 20.52 |

**Table 7.** AWQ can match the performance of keeping 1% salient weights in FP16 under grouped quantization without introducing mixed-precisions, but not for no-group quantization. Nonetheless, grouped quantization has a far better performance compared to no-group, making it a far more practical setting for weight-only quantization of LLMs, while AWQ performs quite well under this setting. Results are perplexity on the WikiText-2 dataset.

[+1]: †footnotetext: $*$ indicates equal contributions.

[+2]: \*\*https://github.com/ggerganov/llama.cpp

[+3]: ††https://github.com/NVIDIA/FasterTransformer
