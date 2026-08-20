---
title: 'A Survey of Low-bit Large Language Models'
createTime: 2026/08/03 11:20:41
permalink: /en/papers/low-bit-llms/
---

> [Ruihao Gong](https://xhplus.github.io/), [Yifu Ding](https://yifu-ding.github.io/), [Zining Wang](https://dblp.org/pid/181/4125), [Chengtao Lv](https://scholar.google.com/citations?user=r8vseSUAAAAJ), [Xingyu Zheng](https://xingyu-zheng.github.io/), [Jinyang Du](https://dblp.org/pid/18/8964), [Jinyang Guo](https://jinyangguo.github.io/), [Xianglong Liu](https://xlliu-beihang.github.io/), [Haotong Qin](https://htqin.github.io/), [Michele Magno](https://pbl.ee.ethz.ch/the-institute/people/person-detail.michele-magno.html), [Yang Yong](https://dblp.org/pid/24/167), [Shiqiao Gu](https://dblp.org/pid/305/3723), and [Dahua Lin](https://dahualin.org/). First submitted to arXiv on September 25, 2024; published in *Neural Networks* 192 (2025). This web reading edition follows arXiv v3 of [A Survey of Low-bit Large Language Models: Basics, Systems, and Algorithms](https://arxiv.org/abs/2409.16694). The [original PDF](/paper/low-bit-llms.pdf) remains authoritative for mathematical notation, figures, tables, and the complete reference list. [DOI](https://doi.org/10.1016/j.neunet.2025.107856). [TeX source](https://arxiv.org/src/2409.16694).

## Abstract

Large language models (LLMs) have achieved remarkable advancements in natural language processing, showcasing exceptional performance across various tasks. However, the expensive memory and computational requirements present significant challenges for their practical deployment. Low-bit quantization has emerged as a critical approach to mitigate these challenges by reducing the bit-width of model parameters, activations, and gradients, thus decreasing memory usage and computational demands. This paper presents a comprehensive survey of low-bit quantization methods tailored for LLMs, covering the fundamental principles, system implementations, and algorithmic strategies. An overview of basic concepts and new data formats specific to low-bit LLMs is first introduced, followed by a review of frameworks and systems that facilitate low-bit LLMs across various hardware platforms. Then, we categorize and analyze techniques and toolkits for efficient low-bit training and inference of LLMs. Finally, we conclude with a discussion of future trends and potential advancements of low-bit LLMs.
Our systematic overview from basic, system, and algorithm perspectives can offer valuable insights and guidelines for future works to enhance the efficiency and applicability of LLMs through low-bit quantization.

## 1 Introduction

Large language models (LLMs) [Ope24a, Tou23, Tou23a, Dub24, Loz24, Dee24] have revolutionized natural language processing by delivering unprecedented performance across a range of tasks, from text generation to language understanding. However, their remarkable capabilities come with significant computational and memory demands. This has raised considerable challenges when deploying these models in scenarios with limited resources or high concurrency. To address these challenges, low-bit quantization has emerged as a pivotal approach for enhancing the efficiency and deployability of LLMs.

Low-bit quantization involves the process of reducing the bit-width of tensors, which effectively decreases the memory footprint and computational requirements of LLMs.
By compressing weights, activations, and gradients of LLMs with low-bit integer/binary representation, quantization can significantly accelerate inference and training and reduce storage requirements with acceptable accuracy.
This efficiency is crucial for enabling advanced LLMs to be accessible on devices with constrained resources, thereby broadening their applicability.

In this paper, we aim to provide a survey with a comprehensive overview of low-bit quantization for large language models (LLMs), encompassing the fundamental concepts, system implementations, and algorithmic approaches related to low-bit LLMs.
Compared with the traditional models, LLMs, as the representative paradigm of the foundation model,
always feature a vast number of parameters, which presents unique challenges for effective quantization. As depicted in [Figure 1](#figure-01), Section 2 introduces the fundamentals of low-bit quantization of LLMs, including new low-bit data formats and quantization granularities specific to LLMs. Section 3 reviews the systems and frameworks supporting low-bit LLMs across various hardware platforms. We then categorize low-bit quantization techniques for efficient training and inference in Sections 4 and 5, respectively. For training, we discuss methods for low-bit training and fine-tuning of LLMs. For inference, we differentiate LLM quantization methods by quantization-aware training and post-training quantization. Quantization-aware training is often used for low-bit settings (such as binary quantization). Post-training quantization is more commonly applied in existing research since it is a resource-efficient pipeline. For a clear understanding, we first cover the widely used techniques of equivalent transformation for reducing outlier influence and weight compensation for mitigating quantization errors. Then the mixed precision, techniques that combine quantization with other compression methods, as well as methods for new quantization forms are discussed. Additionally, we summarize toolkits that integrate these algorithms to support the development of accurate low-bit LLMs. Finally, Section 6 explores future trends and directions, discussing emerging research areas, potential breakthroughs, and the impact of new technologies on LLM quantization.
Our survey provides a detailed description of the fundamentals of low-bit LLMs and gives a comprehensive view of the system implementations for accelerating training and inference through low-bit quantization and algorithms and strategies to maintain and enhance quantized accuracy. We believe this survey can provide valuable insights and advance the development of LLM quantization.

<span id="figure-01"></span>

![The skeleton of the LLM Quantization methods. The diagram illustrates the main areas in the survey.](../../papers/low-bit-llms/figure-01.png)

**Figure 1.** The skeleton of the LLM Quantization methods. The diagram illustrates the main areas in the survey.

## 2 Basics of Low-bit LLMs

In this section, we introduce the basic fundaments of quantization and low-bit LLMs from three aspects: (1) Low-bit number formats. To deal with the outliers in LLMs, low-bit floating-points are first used in quantization. And lots of custom data formats are designed to tackle the outliers. However, integers are still the mainstream. (2) Quantization granularity. To improve the performance of quantized LLMs, finer-grained quantization retains more information and generates better results. But course-grained ones occupy less storage and are more efficient in inference. (3) Dynamic or static quantization. Dynamic quantization does not require calibration, as the quantization parameters are calculated on the fly, making the preparation of a quantized model simpler. In contrast, static quantization requires pre-calibration of quantization parameters, but it offers faster inference performance.

### 2.1 Low-bit Number Formats

We start with the low-bit number formats at the beginning of the introduction. First, we demonstrate the standard formats that are well-recognized, but focus on the differences in LLMs. Second, we introduce some typical custom formats that are designed for LLMs.

#### 2.1.1 Standard Formats

**Floating-point Numbers**.
The floating-point data type is comprehensively defined in the IEEE 754 [Iee19] standard, which is also the most prevailing number format in computer systems. Let us denote them as $\mathrm{FP}k$, where $k$ represents the number of bits that the value occupies in memory, usually $32,16,8$, etc. A floating-point number can be uniformly expressed as:

$$
\begin{aligned}
X_{\mathrm{FP}k} = {(-1)}^{s}2^{p-\mathrm{bias}}(1.\mathrm{mantissa})={(-1)}^{s}2^{p-\mathrm{bias}} \\
\left(1+\frac{d_{1}}{2}+\frac{d_{2}}{2^{2}}+\ldots+\frac{d_{m}}{2^{m}}\right),
\end{aligned}
$$

where $s$ is the sign bit, $p$ is the exponent integer, $\mathrm{bias}$ is applied to the exponent, $m$ is the total number of mantissa bits in the significand, and $d_{1},d_{2},\ldots,d_{m}$ represent the digits of the mantissa part in the binary format.
The bits of $s$, $p$ and $m$ should be accumulated to $k$ for an $\mathrm{FP}k$ value.

Since LLMs occupy more memory, lower-bit formats become popularly adopted in both training and inference. We omit the 32-bit number format here since the 16 and lower bitwidth has become the mainstream practice in application.
We can further categorize each $\mathrm{FP}k$ according to its bit allocations for the exponent (E) and mantissa (M) parts.
We use $\mathrm{E}e\mathrm{M}m$ to denote the subcategories.
As for $\mathrm{FP}16$, IEEE 754 defines float16 (also known as half-precision or FP16) and bfloat16 (brain floating point or BF16), which can be represented as E5M10 and E8M7, respectively. Therefore, bfloat16 can represent larger magnitudes with more exponent bits (identical to that of FP32) while more sparse than float16 with less mantissa in the significand, which may exert unprecedented potential in LLMs [Hen19].
As well as E4M3 and E5M2 for $\mathrm{FP}8$, both are standard formats that are already supported by several mainstream deep learning inference engines, such as MLC-LLM, Quanto, and so on (see Section 3.1.2 for details).

**NormalFloat (NF)** [Det24] is a fixed floating-point method used in weight-only quantization strategies for LLMs. The data representing format follows the floating-points, but the $2^{k}$ values $X^{\mathrm{NF}}_{i},i\in[0,2^{k}-1]$ are estimated to be:

$$
\begin{aligned}
X^{\mathrm{NF}}_{i} =\frac{1}{2}\Bigl(\mathrm{quantile}\!\left(N(0,1),\frac{i}{2^{k}+1}\right) \\
\quad+\,\mathrm{quantile}\!\left(N(0,1),\frac{i+1}{2^{k}+1}\right)\Bigr),
\end{aligned}
$$

where
$\mathrm{quantile}(\cdot,q)$ is the $q$-th quantiles of the input. $N(0,1)$ means the standard normal distribution. For a tensor that does not fall within the range of -1 to 1, we must first scale it using its maximum absolute value. To ensure the exact representation for zero, it asymmetrically divides the data into the positive and negative parts by estimating $2^{k-1}$ of $X^{\mathrm{NF}}_{i}$ for the negative and $2^{k-1}-1$ for the positive, then removes one of the zeros in both sets. NF is estimated to have an almost equal expected number of values in each quantization bin to keep the most information in the quantized formats.

**Micro Scaling FP** [Dar23].
It was proposed and developed in collaboration with industry alliance members, including AMD, Arm, Intel, Meta, Microsoft, NVIDIA, and Qualcomm,
which aims to establish a unified standard for fine-grained sub-blocks of tensor format. It applies E8M0 scaling factors on a block of data with various original formats (i.e., FP8, FP6, FP4, INT8). The scaling block size indicates the number of elements that each scaling applies. It keeps high precision for the value representation but is significantly efficient on hardware due to the shared scalings.

**Integer Numbers**. Integer quantization is the most widely studied quantized data format since the quantization technique has emerged. It divides the floating-points into $2^{k}$ equally spaced discrete integers.
The formula is:

$$
X_{\mathrm{INT}_{k}}=(-1)^{s}(d_{1}2^{m}+d_{2}2^{m-1}+\cdots+d_{m}2^{0}),\quad x\in\mathbb{N}^{+},
$$

where $m=k-1$ and $s\in\{0,1\}$ for signed integers. $m=k$ for unsigned integers while we regard $s=0$. Therefore, the signed integers range from $[-2^{k-1},2^{k-1}-1]$, and the unsigned one $[0,2^{k}-1]$. Before the advent of LLMs, integer quantization had been applied in BERT-based language models, as demonstrated by [She20].

**Binarized Numbers**. Binarization is the most aggressive quantization technique, which directly abstracts the sign of value [Liu18, Qin22, Li24]. It will lose most information, but bring significant acceleration and parameter compression in inference. The hardware takes $0,1$ for each bit originally, but developers define different logic rules and accumulation algorithms to achieve various binarized computations.
Therefore, floating-point numbers can be binarized to $\{-1,1\}$ or $\{0,1\}$, depending on what value we expect the single bit to represent in our algorithms. Some studies further extended binarization to ternary quantization. Before the emergence of LLMs, works such as [Bai21, Zha20, Liu22, Liu23a] explored binary or ternary quantization formats.

[Table 1](#table-01) shows the representation ranges of various standard formats. It shows that even with the same bit-width, different numerical representation formats can have significantly different value ranges. The floating-point numbers with larger $E$ have larger representation ranges but sparser points. Therefore, there is a tradeoff between finer data intervals or larger data ranges when determining data formats for a specific model and task.

<span id="table-01"></span>

![Min and Max values for different number formats [Iee19].](../../papers/low-bit-llms/table-01.png)

**Table 1.** Min and Max values for different number formats [Iee19].

#### 2.1.2 Custom Formats

For faster computation and better fitting the numerical distributions of LLMs, many studies propose custom number formats besides the standard formats described above. Here we introduce three typical customized formats.
We omit the works before the advent of LLMs [Tam19] because their performance has not been validated on LLMs.

**Floating-point Integer (Flint)** [Guo22] combines the advantages of floating-point and integer representations, which is $X_{\mathrm{Flint}}=2^{p-\mathrm{bias}}\times(1.\mathrm{mantissa})$. We take the 4-bit Flint on float-based MAC units as an example:

$$
\begin{aligned}
p&=\begin{cases}
3-\mathrm{LZD}(b_{2}b_{1}b_{0}),&b_{3}=0,\\
4+\mathrm{LZD}(b_{2}b_{1}b_{0}),&b_{3}=1,
\end{cases}\\
\mathrm{mantissa}&=b_{2}b_{1}b_{0}\mathbin{\texttt{<<}}(\mathrm{LZD}(b_{2}b_{1}b_{0})+1),
\end{aligned}
$$

where the `LZD` denotes the `L`eading `Z`ero `D`etector [Okl94] which accumulates the leading zeros on the left of the bitstring, `<<` is the left shift operation, and $\mathrm{bias}=1$ for float-based Flint4. It expands the range by integrating exponents into the integers, therefore. Compared to pure integers, Flint can represent a larger range with a limited number of bits, which better fits the distribution of LLM parameters.

**Adaptive Biased Float (Abfloat)** is first proposed in Outlier-Victim Pair Quantization (OVP) [Guo23] to deal with outliers. The difference to Flint is that Abfloat applies a bigger $\mathrm{bias}$ to the exponent, and left shifts $m$-bit to enlarge the $1$ before `mantissa`, making the magnitude even larger to cover the outliers. The $\mathrm{E}e\mathrm{M}m$ Abfloat value can be expressed as:

$$
X_{\mathrm{Abfloat}}=(-1)^{s}\times 2^{p+\mathrm{bias}}\times(2^{m}+\mathrm{mantissa}).
$$

When $\mathrm{bias}=0$, the range is similar to $\mathrm{Flint}4$. With $\mathrm{bias}=2$ for E2M1, the range changes to $\{12,\dots,96\}$. With $\mathrm{bias}=3$, the range further extends to $\{24,\dots,192\}$.
The other difference to Flint is that Abfloat is only adopted on outliers, but the normal values are stored in INT4/8 or Flint4. Both data formats require custom system support to define the behavior of the base operations (such as addition, multiplication, and so on).

**Student Float (SF)** [Dot24] follows the floating-point format but has specific fixed points for quantization, which is different from the above two types. SF is an improvement of NF in Section 2.1.1 and holds the view that the parameters obey Student’s t-distribution $S(t;\nu)$, of which the probability density function is:

$$
S(t;\nu)=\frac{\Gamma\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\Gamma\left(\frac{\nu}{2}\right)}\left(1+\frac{t^{2}}{\nu}\right)^{-\frac{\nu+1}{2}},
$$

where $t$ and $\nu$ are the independent variable and degrees of freedom, respectively, and $\Gamma$ is generalized factorial.

$$
\tilde{X}^{\mathrm{SF}}_{i}=\mathrm{quantile}\left(S(t;\nu),q_{i}\right),\quad q_{i}=\begin{cases}\omega+(\frac{1}{2}-\omega)\frac{i-1}{7}&i\in\{1,\dots,8\}\\ \frac{1}{2}+(\frac{1}{2}-\omega)\frac{i-8}{8}&i\in\{9,\dots,16\}\end{cases},
$$

where $\omega=\frac{1}{2}(\frac{1}{32}+\frac{1}{30})$, $\{q_{1},\dots,q_{8}\}$ and $\{q_{9},\dots,q_{16}\}$ are two groups of evenly spaced quantiles. Then we normalize $\tilde{X}^{\mathrm{SF}}$ to $[-1,1]$ by ${X}^{\mathrm{SF}}_{i}=\frac{\tilde{X}^{\mathrm{SF}}_{i}}{\max_{i}|\tilde{X}^{\mathrm{SF}}_{i}|}$.
As $\nu$ increases, the peaks of the t-distribution become shorter and wider, and SF4 spreads out more. It converges to the standard normal distribution (NF) as $\nu\to\infty$. Same as NF, SF is used in weight-only quantization (which we introduce in Section 3.2.1). Therefore, it does not need the low-level definition of base operations but requires a custom dequantization from SF to standard formats.

### 2.2 Quantization Granularity

<span id="figure-02"></span>

![Illustrations for different quantization granularity.](../../papers/low-bit-llms/figure-02.png)

**Figure 2.** Illustrations for different quantization granularity.

Quantization granularity refers to the different weight/activation partitions corresponding to each element of the scaling factor and zeropoint. It determines how finely the scale recovers and the zero point shifts. [Figure 2](#figure-02) showcases five fundamental types of quantization granularity: tensor-wise, token-wise, channel-wise, group-wise, and element-wise.

**Tensor-wise** is the simplest and coarsest granularity, which takes a single scaling factor and zero point to the entire tensor [Zha24]. It can be the fastest but may lead to the most performance degradation because it is incapable of handling the values with a wide variation. Therefore, it is unsuitable for cases where accuracy is important or the task/model is sensitive to quantization.

**Token-wise** is used in LLMs only, which means that each token (word or subword) has a scaling [Yao22]. It captures the fine-grained variations in different tokens. Usually, we adopt dynamic token-wise quantization for activation to reduce the quantization error and ensure diversity in generative models.

**Channel-wise** means each channel in weight within a tensor uses one scale and can be merged into quantized weight [Kim24]. Token-wise activation and channel-wise weight are usually used together. Because for $i$-th token in activation and $j$-th channel in weight, the corresponding $s_{\textbf{x}_{i}}\in s_{\textbf{x}}\in\mathbb{R}^{T\times 1}$ and $s_{\textbf{w}_{j}}\in s_{\textbf{w}}\in\mathbb{R}^{1\times C}$ can be calculated first as $s\in\mathbb{R}^{[1]}$ and multiplied to the coordinate $[i,j]$ in output matrix $\textbf{X}_{O}$. In this way, we preserve the generation performance with little computation overheads.

**Group-wise** balances the computational complexity and the quantization error by grouping tensors or channels with the same scaling factor. It also reduces the storage of scaling factors by $g$ if there are $g$ tokens/channels per group [Heo23, Yao22].

**Element-wise** is only applied while training the weight, which is always used together with another quantization granularity, such as tensor-wise (see [Figure 2(e)](#figure-02)). Before inference, the element-wise scaling is merged into the quantized weight. Therefore, only the tensor-wise scale needs to be computed in inference [Lee23] to recover the value magnitude.

Different quantization granularity are always combined and adopted together. For example, [Lee23] uses a channel-wise scale for the Key matrix but a token-wise scale for the Value matrix based on the distribution of the data. More algorithms can be found in Section 5.2.3.

### 2.3 Dynamic and Static Quantization

<span id="figure-03"></span>

![Dynamic and static quantization. Operations in the green block mean the inference process, while outside the block is the production and preparation process.](../../papers/low-bit-llms/figure-03.png)

**Figure 3.** Dynamic and static quantization. Operations in the green block mean the inference process, while outside the block is the production and preparation process.

Dynamic and static quantization mainly refers to the strategies in PTQ, which are illustrated in [Figure 3](#figure-03). We take integer quantization as an example, and other low-bit quantization methods have a similar process.

**Dynamic Quantization** [Kri18, Liu22a] calibrates and stores quantized weight. Usually, it does not need input data, but searches for the optimal scaling factors $s_{\mathbf{w}}$ and zero-points $Z_{\mathbf{w}}$ by minimizing the quantization error for each tensor of weight. During inference, the activation will be input into the quantization module to compute the optimal scaling factors $s_{\mathbf{x}}$ and $Z_{\mathbf{x}}$, and then quantized to INT8 by the dynamically computed factors before conducting integer GEMM with quantized weight. The scaling and zero point of activation are obtained in real time based on the current batch of input data. Therefore, the scaling factor flexibly adapts the input data distribution, bringing the smallest quantization error. While it takes extra computational complexity to get the scale during inference. It is suitable for scenarios that require rapid deployment because it does not require calibration.

**Static Quantization** [Bai21a] takes calibration data consisting of a small fraction of the training dataset. By inputting the samples into the model, we find the optimal scaling factors for both weight and activation (the middle one in [Figure 3](#figure-03)) or weight only (the right one), and are fixed during inference. It allows for the evaluation of the quantized model during preparation, ensuring that quantization does not significantly harm the model’s performance.
As for inference, the middle one in [Figure 3](#figure-03) quantizes the activation to low-bit and computes low-bit GEMM [Det22] with quantized weight. For the right one in [Figure 3](#figure-03), the weight will be dequantized to floating-point numbers, and the activation will not be quantized before conducting floating-point GEMM [Lin24], thus we name it weight-only quantization.

<span id="table-02"></span>

![Inference frameworks for quantized large language models.](../../papers/low-bit-llms/table-02.png)

**Table 2.** Inference frameworks for quantized large language models.

## 3 Frameworks and System Support

In the few short years since the large language model emerged, there have arisen many frameworks to support the easy usage of LLMs.
We have selected some well-known representative frameworks and tools related to quantization, summarized and introduced them in this section according to the following categories:

1. **Inference framework for quantization**, which provides comprehensive libraries and APIs for the rapid development and deployment of LLM applications,

2. **System support for quantization**, which supports the underlying core functionality for quantization methods.

In the following, our emphasis is on the quantization of LLMs across various frameworks and libraries.

### 3.1 Inference Framework for Quantization

We list the representative inference frameworks in [Table 2](#table-02). The inference process of Large Language Models (LLMs) consists of two key stages: Prefill and Decode. During the Prefill stage, the input prompt is tokenized and processed through the model’s Transformer layers to generate contextual embeddings, leveraging self-attention mechanisms to capture dependencies between tokens. This stage establishes a rich contextual representation of the input, which is stored for subsequent text generation. In the Decode stage, the model generates text autoregressively, predicting one token at a time by iteratively considering the sequence of previously generated tokens. This involves embedding lookup, attention computation, and token selection based on probability distributions. While the prefill stage processes the entire input at once, making it computationally intensive, the decode stage operates incrementally, building the output sequentially. Together, these stages enable LLMs to produce coherent and contextually relevant text, forming the foundation for optimization techniques like quantization, which aim to enhance efficiency without compromising performance.
Currently, no single inference framework dominates in terms of performance or usage. However, some classic deep learning frameworks, such as TensorRT-LLM ([Link](https://github.com/NVIDIA/TensorRT-LLM)), ONNX-runtime ([Link](https://github.com/microsoft/onnxruntime)), Transformers ([Link](https://huggingface.co/docs/transformers/en/index)) (Huggingface), OpenVINO ([Link](https://github.com/openvinotoolkit/nncf)), PowerInfer ([Link](https://github.com/SJTU-IPADS/PowerInfer)), PPLNN ([Link](https://github.com/openppl-public/ppl.nn)), and Xorbits Inference ([Link](https://github.com/xorbitsai/inference)) have integrated the support for efficient inference of large models. In addition, other inference frameworks emerged after the advent of large models that are specifically proposed for LLMs, such as bitsandbytes ([Link](https://github.com/bitsandbytes-foundation/bitsandbytes)), ctransformers ([Link](https://github.com/marella/ctransformers)), MLC-LLM ([Link](https://github.com/mlc-ai/mlc-llm)), DeepSpeed-MII ([Link](https://github.com/microsoft/DeepSpeed-MII)), vLLM ([Link](https://github.com/vllm-project/vllm)),
LMDeploy ([Link](https://github.com/InternLM/lmdeploy)),
LightLLM ([Link](https://github.com/ModelTC/lightllm)), QServe ([Link](https://github.com/mit-han-lab/qserve)), llama.cpp ([Link](https://github.com/ggerganov/llama.cpp)), llama2.c ([Link](https://github.com/karpathy/llama2.c)), inferflow ([Link](https://github.com/inferflow/inferflow)), ScaleLLM ([Link](https://github.com/vectorch-ai/ScaleLLM)), SGLang ([Link](https://github.com/sgl-project/sglang)), gpt-fast ([Link](https://github.com/pytorch-labs/gpt-fast)), FastChat ([Link](https://github.com/lm-sys/FastChat)), OpenLLM ([Link](https://github.com/bentoml/OpenLLM)) and so on. These frameworks are lightweight and have integrated many specialized optimization techniques for large models.

#### 3.1.1 Ready-to-use Algorithms

With the emergence of quantization algorithms for LLMs, some typical methods have already been integrated into most frameworks, while some methods may be developed and published originally on a specific framework. We list the most ready-to-use algorithms in each mainstream framework in [Table 2](#table-02). Some methods are included by most frameworks, such as GPTQ [Fra22], AWQ [Lin24], SmoothQuant [Xia23], and so on. These methods share several advantages: high accuracy and efficient performance after quantization, seamless integration into existing implementation procedures, and user-friendliness.

In addition, some algorithms are supported by several frameworks. For example, LLM.int8() [Det22] was well supported by bitsandbytes (in HuggingFace), which allows to store and load 8-bit weights directly from the HuggingFace Hub and quantize weight in linear layers to 8-bit.
FP6-LLM [Xia24] is integrated in DeepSpeed-FastGen ([Link](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)) [Hol24] to implement the runtime quantization for 6-bit floating-point weight-only quantization. It allows efficient quantization and dequantization of 6-bit weight LLMs through a unified configuration option.
It is noteworthy that Transformers (by HuggingFace) and QServe (by MIT EECS [Lin24a]) integrate most algorithms with comprehensive user manuals and detailed examples, enabling a quick start for deep learning researchers and developers.

#### 3.1.2 Bitwidth Support

The support for bitwidth always reflects how comprehensive the quantization system implementation is for an inference framework or engine. It can be categorized into three types according to its position and function in accelerating LLMs:

**Weight-only<sub>bit</sub>** means only quantizing the weight while keeping FP16 activation [Lin24]. The quantized weight will be dequantized back to FP16 using pre-obtained scaling factors, and then conduct FP16 `mma` with FP16 activation. Therefore, it theoretically supports non-uniform quantization with arbitrary bitwidth. The speedup is achieved by reducing the latency of data transmission between the computing device and storage host with smaller amounts of weight data, but the dequantizing of weight costs extra time. The detailed speedup will be discussed in Section 3.2.1.

**W<sub>bit</sub>&A<sub>bit</sub>** means that the algorithm quantizes both the weight and activation, and conducts low-bit matrix multiplication (MatMul) in low-level (for example, in PTX ISA 8.5 ([Link](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)) for NVIDIA GPUs, instruction `mma.sync.aligned.shape.row.col .s32.u4.u4.s32` means the data type of the multipliers is the 4-bit unsigned integer). All the frameworks support the INT8 and FP16 MatMul. However, limited by the computing capabilities of the hardware and the supported operations in the instruction set, only part of them have INT4 and FP8 MatMul. Few supports different bitwidth of weight and activation (like $W_{\mathrm{INT4}}A_{\mathrm{INT8}}$), which requires customized computation kernels with assembled GEMV instructions ([Link](https://huggingface.co/docs/transformers/main/en/quantization/eetq)) [Egi24]. It should be mentioned that if you want to use low-bit MatMul, your hardware architecture must support the specific low-bit computing, and it is necessary to upgrade/downgrade the driver to the corresponding version to reproduce the real low-bit computation and get the desired speedup ratio.

**KV Cache<sub>bit</sub>** lists the bitwidth of Key-Value Cache. As a caching technology, memory consumption of the KV cache increases rapidly as batch size and sequence length continue to grow, potentially surpassing the model size. Therefore, quantizing the KV cache significantly reduces memory usage during model inference. There are several works devoted to quantizing the KV cache [Hoo24, Yue24a, Liu24a]. Similar to weight-only algorithms, the quantized key-value pairs usually need to be dequantized to floating-point before MatMul, otherwise, the specific system support of multiplying low-bit to floating-point is required. Except for the listed bitwidth, all frameworks support the FP16 KV cache, which means directly storing the activation.

We also list the quantization granularity. Users should refer to the manual to make sure that the quantization granularity is used for weight, activation, or KV cache. We sort out the granularity supports in each framework as a reference to help choose a suitable framework that implements the desired computation kernels.

#### 3.1.3 Target Platforms

Numerous vendors are competing fiercely in the deep learning hardware. As one of the pioneers in the field of deep learning GPUs today, NVIDIA GPUs are supported by most frameworks. Meanwhile, vLLM, bitsandbytes, llama.cpp, ctransformers, MLC-LLM, and PowerInfer also have the support for AMD GPUs.
For some other processing units, such as TPU, XPU, Metal, and other hardware, the system support is relatively limited. Some frameworks that are devoted to generalizing LLMs to edge devices are more likely to extend the support for those platforms, such as MLC-LLM, ONNX-Runtime, and llama.cpp.
However, it should be noted that the frameworks with support for both low-bit quantization and hardware deployment in [Table 5](#table-05) cannot guarantee the deployment of any quantized model on each listed hardware. Users should carefully refer to the manual for guidance. However, the table we compiled may help reduce the time it takes to find a suitable framework that may meet your deployment desire.

#### 3.1.4 Model Family

All the frameworks support custom model definition and seamlessly integrate external model zoos, such as HuggingFace Hub. To help users quickly get started, the frameworks provide predefined specification files for commonly used models. We can roughly classify the large models into three categories: Transformer-like LLMs (e.g., Llama, Orion, Baichuan, ChatGLM, Falcon), Mixture-of-Expert(e.g., Mixtral, Mistral, DeepSeek), Multi-modal LLMs (e.g., LLaVA).
However, not all large models included in external model zoos can be smoothly supported, because the frameworks integrate new algorithms with a lag. Therefore, users should refer to the model zoo provided by the framework, and make sure that the target model has no additional underlying system requirements before importing a new model from the external model zoo that is beyond the supported model list.

### 3.2 System Support for Quantization

In practical implementations, it is perplexing that some quantization algorithms, although reducing the bitwidth of weight or activation, do not lead to a faster inference. Therefore, a critical question comes into mind: *How does quantization actually achieve real acceleration and storage saving?* To answer this question, we must first clarify the data transmission process involved in model inference.

<span id="figure-04"></span>

![Data transmission of weight and activation in the caching system during inference. The bandwidth and latency are officially reported by NVIDIA A100 as an example. `PCIe` is a high-speed interface standard used for connecting various hardware components, such as GPUs, SSDs. `Async_Copy` means asynchronous data copy using cp.async intrinsic. `ldmatrix` and `lds` are data loading instructions that load matrix from shared memory to registers with a strict layout requirement or in a fine-grained and flexible manner, respectively [Nvi25].](../../papers/low-bit-llms/figure-04.png)

**Figure 4.** Data transmission of weight and activation in the caching system during inference. The bandwidth and latency are officially reported by NVIDIA A100 as an example. `PCIe` is a high-speed interface standard used for connecting various hardware components, such as GPUs, SSDs. `Async_Copy` means asynchronous data copy using cp.async intrinsic. `ldmatrix` and `lds` are data loading instructions that load matrix from shared memory to registers with a strict layout requirement or in a fine-grained and flexible manner, respectively [Nvi25].

The data transmission process of weight and activation in the multi-level caching system is outlined in [Figure 4](#figure-04), which shows the general dataflow of quantized LLMs. GPUs typically use a hierarchical cache structure with multiple levels, each with different sizes and IO speeds. On-chip caches (L2 cache, shared memory, and registers) provide faster access but have limited capacity, while off-chip caches (device memory or global memory, host memory) offer more capacity but have slower access speed. Therefore, in today’s LLMs inference frameworks, we need to load and compute data in segments with highly parallel single instruction, multiple threads (SIMT) paradigm to ensure an acceptable inference speed.

*Host memory $\rightarrow$ Device memory.* For weight, we load one layer’s weight from the host memory to the device’s global memory. The bandwidth is relatively low, which is 25 GB/s per direction (taking NVIDIA A100 as an example [Smi20]). If quantized, it is always in a compact format, thus the time can be saved. The activation is originally generated on the device during inference, which does not need to be copied from the host.

*Off-chip memory $\rightarrow$ On-chip memory.* We copy a chunk of weight and activation ready to compute matrix multiplication from the off-chip global memory to the on-chip L2 cache and shared memory. The amount of data copied at a time is basically determined by the design of matrix multiplication (MatMul) kernels, which is always multiple of the number of elements computed in one kernel execution by SIMT. The bandwidth is 1555 GB/s in A100.

*Shared memory $\rightarrow$ Registers.* For faster computation, the quantize/dequantize operations and MatMul are always conducted in registers. Therefore, we need to copy the weight and activation from the shared memory to the registers with small pieces. The bandwidth is 19400 GB/s, which requires more than 10 times threads and 1/780 compute intensity of `PCIe`.

*Offloading (Registers $\rightarrow$ shared memory $\rightarrow$ off-chip memory).* The computation results are copied or accumulated to the corresponding elements on shared memory. After finishing the computation for the chunk of data, the results on shared memory are offloaded to the off-chip memory. The memory that stores the weight and activation of the last chunk can be freed before moving to the next.

Above, we have clarified the data transmission process by taking the MatMul of a linear layer as an example.
Only after then can we answer the question: **How do quantization reduce the latency and storage?**
To achieve the actual inference acceleration and storage saving, we need comprehensive system support for quantization from the bottom up.

In the following sections, we demonstrate the system supports for quantization according to the action scopes: **Weight-only, Weight & Activation, KV Cache, and Quantization & Dequantization**.
We first provide the common and general practices in most frameworks. While these practices may not be the most efficient, they offer high scalability and generalization, allowing new algorithms and implementations to be quickly and easily integrated. Then, we introduce several custom designs. These studies investigate the speedup and generation quality bottlenecks and propose faster solutions for a certain scope.
[Figure 5](#figure-05) shows how the quantization of weight or activation reduces inference time (4-bit integer quantization is taken as an example, which can also be any other low-bit data format). [Figure 6](#figure-06) illustrates how quantized KV Cache affects the inference.
Speedup Timelines in both figures clearly divide the whole process into three types based on the time consumption compared to the FP16 counterpart: *Speedup* (green line), *Slow down* (dark grey line), and *Not affect* (light grey line).

<span id="figure-05"></span>

![The data transmission process of quantization for (a) Quantized weight preparation (weight pack), (b) Weight-only quantization, and (c) Weight & Activation quantization.](../../papers/low-bit-llms/figure-05.png)

**Figure 5.** The data transmission process of quantization for (a) Quantized weight preparation (weight pack), (b) Weight-only quantization, and (c) Weight & Activation quantization.

#### 3.2.1 Weight-Only Quantization

The fundamental bottleneck in model inference before and after the advent of large models is the data transmission and storage costs, which are always neglected in ordinary small models. Due to the large amount of data, the transmission latency can not be overlooked, which even surpasses the computation latency and becomes the major challenge in LLM inference. Therefore, weight-only quantization emerges, which compacts the weight and reduces the data copy burden among levels of caches [Lin24, Fra22].

The processes related to weight-only quantization are illustrated in [Figure 5](#figure-05) (a) and (b). Both weight-only and weight & activation quantization require packing weight to lower bitwidth beforehand. The weight packing is only conducted once before inference, and it costs little computation resources and time. The weight data are distributed to multi-threads, with each thread tiles a chunk of data according to the following steps: (1) quantizing the weight to lower bitwidth by pre-obtained scaling factors, (2) densely packing them into `uINT32` units without idle bits, (3) offloading and storing into host memory. Therefore, the packed weight has a significant reduction in storage compared to the floating-point one.

See speedup timeline in (b), weight-only quantization alleviates the burden of data transmission from host memory to on-chip memory by reducing the data amounts. However, it introduces additional dequantization of weight before conducting the MatMul because the general kernels only receive the same datatype of inputs.
As long as the time spent on dequantization is shorter than the time saved on data transmission, the weight-only quantization brings benefits in acceleration, which indeed is the case. It is the overload of parameter transmission in LLMs that makes weight-only quantization valuable in practice. Therefore, even using floating-point MatMul kernels, weight-only quantization can still accelerate the inference of LLMs.

As for custom designs, since weight-only quantization dequantizes the weight back to FP16, it is possible to pack the weight with arbitrary bitwidths during quantized weight preparation. Many works propose 3-bit, 5-bit, 6-bit weight quantization [Shi24, Fra22, Xia24].
Furthermore, since the quantized weight must be dequantized to higher bitwidths before MatMul, it is not necessary to design a linear surjection from low bitwidth numbers to real values. In other words, we can map the integers to arbitrary floating-point numbers, and adopt lookup tables for dequantization [Dot24, Det24].
To make full use of storage and reduce the time of dequantizing weight during inference, researchers design customized backends on specific platforms to support efficient inference. For example, FP6-LLM [Xia24] designs a complete GPU kernel to support faster FP6$\rightarrow$FP16 dequantization and the dense storage of weight. SpQR [Det23] has an efficient decoding backend based on GPUs to deal with the outliers by sparse quantization and achieves load balancing.

#### 3.2.2 Weight & Activation Quantization

Following the traditional practice of quantization, both weight and activation are quantized to low bitwidth, and the MatMul kernels are also implemented by low-bit instructions.
We illustrate the speedup timeline in [Figure 5(c)](#figure-05) that the accelerated processes are weight transmission in the caching system as well as the low-bit MatMul. The extra operations are the quantization for activation from FP16 to low-bit integer before MatMul, and the datatype casting for the results from INT32 to FP16 after MatMul.
Weight & activation quantization yields greater acceleration compared to the weight-only quantization because the computationally intensive MatMul usually can be accelerated by lower bitwidth kernels, which use more efficient instructions and a better degree of parallelism. Meanwhile, it is recommended to simplify the complexity of activation quantization to minimize the time spent on runtime quantization.
However, the actual speedup ratio highly depends on the hardware design, such as the number of floating-point and integer processing units.

As for custom designs, there are two categories of techniques:

1. Faster Quantization and Dequantization (or datatype conversion). For example, QQQ [Zha24a] proposes faster FP16$\rightarrow$INT8 for quantizing activation, INT4$\rightarrow$INT8 for dequantizing weight, and INT32$\rightarrow$FP16 for casting the MatMul results to accelerate the data format conversion during inference. This work is based on [Kim22] which firstly introduces a faster INT4$\rightarrow$FP16 datatype conversion. Besides speeding up, other approaches turn to remove the process. Tender [Lee24] proposes a decomposed quantization technique to eliminate runtime dequantization/quantization during inference.

2. Faster MatMul Kernel. GEMV can be more flexible and efficient in fitting various bitwidths than GEMM, and even receives input matrices with two bitwidths, such as INT1*INT8 and INT3*INT8 [Wan23]. By assembling several products of a matrix and a vector, we can get the desired results without padding or idle bits. For example, EETQ ([Link](https://github.com/NetEase-FuXi/EETQ)) introduces GEMV operators which are 13-27% faster than GEMM kernel. SqueezeLLM [Kim23] proposes LUT-based MatMul by GEMV, which supports highly efficient 4-bit MatMul kernel on hardware architectures that do not support integer MatMul instruction. AQLM [Egi24] designs W1A16 and W2A8 MatMul kernels to receive input matrices with extremely low bitwidth and calculate them directly without dequantizing or datatype conversion.

<span id="figure-06"></span>

![Illustration of KV Cache quantization.](../../papers/low-bit-llms/figure-06.png)

**Figure 6.** Illustration of KV Cache quantization.

#### 3.2.3 KV Cache Quantization

KV Cache, or key-value cache, is to optimize the generative models that predict text token by token. Although the model generates only one token at a time, each token depends on the previous context. To avoid repeated calculation, the KV cache acts as a memory bank storing previous key-value results to reuse in the following generations. However, the storage highly depends on the sequence length, hidden size, attention head numbers, and so on. Quantization is an efficient approach to compressing storage. The overall process is illustrated in [Figure 6](#figure-06).

The KV cache is generated and updated in runtime along with the serialized input data. During inference, the $\mathbf{K}_{\mathrm{new}}$ and $\mathbf{V}_{\mathrm{new}}$ from linear layers are first quantized, then concatenated to the end of the stored key and value lists, which are also quantized, to form new lists. When the cache size exceeds its limit, the earliest key-value pairs will be dropped. Then we dequantize the matrices to FP16 before conducting multi-head attention forward propagation with the newly generated query $\mathbf{Q}_{\mathrm{new}}$. We illustrate how KV cache quantization affects the inference in the Speedup Timeline. Compared to the FP KV cache, the quantized one occupies less storage in device memory and spares less time in KV data transmission in the caching system due to the smaller data bytes.

There are mainly four techniques for quantizing KV cache:

1. Quantizing to lower bitwidth. QoQ [Lin24a] compresses KV to 4-bit and proposes SmoothAttention to prevent the accuracy drop due to the lower bitwidth. KIVI [Liu24c] even developed a tuning-free 2-bit KV cache quantization algorithm. [Yan24c] proposes a mixed-precision strategy that quantizes the earliest KV to lower bitwidth, while keeping the new KV with more bits.

2. Quantizing window. Many studies [Zha24b, Dua24] postpone the quantization of KV pairs, but only quantize them in a batch when the length of the full-precision KV list exceeds the window size. For example, SKVQ [Dua24] employs a sliding-window mechanism, determining the quantization parameters within the window.

3. Skipping the dequantization of $\mathbf{K}_{\mathrm{new}}$. Methods like WKVQuant [Yue24a] concatenates the FP $\mathbf{K}_{\mathrm{new}}$ and $\mathbf{V}_{\mathrm{new}}$ to the dequantized $\mathbf{K}_{\mathrm{prev}}$ and $\mathbf{V}_{\mathrm{prev}}$, which preserves more information of current token in $\mathbf{K}$ and $\mathbf{V}$ matrices, then quantizes the $\mathbf{K}_{\mathrm{new}}$ and $\mathbf{V}_{\mathrm{new}}$ and stores them into the KV cache (when meets the condition).

4. Optimizing outliers. There are token-wise outliers in KV matrices, so methods such as storing the outliers with higher bitwidth or mitigating the outlier magnitudes can improve the performance [Don24, Liu24a, Kan24, Lin24a]. We omit the details of this category, as the general practices are similar to the quantization methods used for the entire model.

#### 3.2.4 Quantization and Dequantization

In this section, we roughly categorize the quantization into three types: (1) *Floating-point Quantization*, casting the high-bit floating-points into low-bit ones. (2) *Integer Quantization*, which mainly refers to dividing the floating-points into evenly spaced integers. We omit requantizing higher bitwidth integers to lower bitwidth ones here, because it is seldom used in real practices, and few studies propose faster implementations to convert integers. (3) *Binarization*, including `sign` and `bool` functions.

##### Floating-point Quantization

Quantizing higher bitwidth floating-point to lower is actually the clip of mantissa bits. That is because the source value with higher bitwidth usually has more or equal bits for both exponent and mantissa parts compared to the target value with lower bitwidth. Algorithm 1 provides an example of quantizing FP32 to FP8. And we follow [Mic22] to summarize the general process as follows:

1. *Scale.* Since the target value occupies less bitwidth, the representation range may shrink drastically, and not be able to convey most of the data. Scaling the source value to a suitable range can best preserve the information after quantized to FP8. The scaling is pre-obtained by learning or calibration.

2. *Check Overflow/Underflow.* Check whether the source value overflows the FP8 range, either from the upper or lower bound. If so, return the maximum or minimum directly.
   If it is not overflow, check if the exponent part underflows from the smallest positive normal number that the FP8 format can present. If so, we divide the value by the smallest subnormal number in FP$x$, round to the nearest integer, and then multiply the smallest subnormal number. The integer determines the value of mantissa bits and the exponent bits are all set to zero.

3. *Copy and Round.* If the value is neither overflowing nor underflowing of FP8, we copy the lower $e$ bits from the source FP32 value to the target FP8 value. Then we clip the mantissa to $m$ bits by rounding to the nearest.
   It is notable that rounding and overflow/underflow handling are both crucial for maintaining numerical stability and precision in real applications. However, since the reduction of mantissa bits, precision degradation is inevitable while converting to lower bitwidth.

**Algorithm 1: Quantization to lower-bit floating-point values.**

- **Input:** $X_{\mathrm{FP}32}$, $s\in\mathbb{R}^{+}$, $X_{0}\in\mathbb{R}$, $e,m\in\mathbb{Z}^{+}$, $\mathrm{clip}^{\min}$, and $\mathrm{clip}^{\max}$.
- **Output:** $X_{\mathrm{FP}8}$.
- Set $X_{\mathrm{FP32}}^{\mathrm{unscaled}}=X_{\mathrm{FP32}}/s$.
- Set $e^{\min}$ to `-(1 << (e - 1)) + 1` and $e^{\max}$ to `(1 << (e - 1))`.
- Set $m=x-e-1$, the theoretical maximum of the exponent part for FP8.
- Set $X_{\mathrm{FP}8}^{e}=e^{\max}+2^{8-1}\mathbin{\texttt{<<}}23$, the theoretical maximum of the mantissa part for FP8.
- Set ${X_{\mathrm{FP}8}^{m}}$ to `~(0x007FFFFF >> m) & 0x007FFFFF`.
- Set $X_{\mathrm{FP}8}^{\mathrm{theomax}}=X_{\mathrm{FP}8}^{e}+X_{\mathrm{FP}8}^{m}$.
- **Check exponent overflow:**
  - **If** $X_{\mathrm{FP32}}^{\mathrm{unscaled}}>\min(\mathrm{clip}^{\max},X_{\mathrm{FP}8}^{\mathrm{theomax}})$:
    - Set $X_{\mathrm{FP}8}=\min(\mathrm{clip}^{\max},X_{\mathrm{FP}8}^{\mathrm{theomax}})$.
  - **Else if** $X_{\mathrm{FP}8}^{\mathrm{theomax}}<\max(\mathrm{clip}^{\min},-X_{\mathrm{FP}8}^{\mathrm{theomax}})$:
    - Set $X_{\mathrm{FP}8}=\max(\mathrm{clip}^{\min},-X_{\mathrm{FP}8}^{\mathrm{theomax}})$.
  - **Else:**
    - Set $X_{\mathrm{FP}8}^{\mathrm{sign}}=X_{\mathrm{FP32}}^{\mathrm{unscaled}}$ `& 0x80000000`.
    - Set $X_{\mathrm{FP}8}^{e}=X_{\mathrm{FP32}}^{\mathrm{unscaled}}$ `& 0x7F800000`.
    - Set $X_{\mathrm{FP}8}^{m}=X_{\mathrm{FP32}}^{\mathrm{unscaled}}$ `& 0x007FFFFF`.
    - **Check exponent underflow:**
      - **If** $(X_{\mathrm{FPx}}^{e}\mathbin{\texttt{>>}}23)-2^{x-1}<e^{\min}+1$:
        - Set ${X_{\mathrm{FP}8}^{\min}}_{\mathrm{subnorm}}$ to `1 / (1 << ((1 << (e - 1)) + m - 2))`.
        - Set $X_{\mathrm{FPx}}=\mathrm{round2int}(X_{\mathrm{FP32}}^{\mathrm{unscaled}}/{X_{\mathrm{FP}8}^{\min}}_{\mathrm{subnorm}}){X_{\mathrm{FP}8}^{\min}}_{\mathrm{subnorm}}$.
    - **Round the mantissa:**
      - Set $R_{m}=(X_{\mathrm{FP}8}^{m}\mathbin{\texttt{<<}}m)$ `& 0x007FFFFF + 0x3F800000`.
      - Set $R_{m}=\mathrm{round2int}(R_{m}-1)$.
    - **Process the mantissa:**
      - Set $X_{\mathrm{FP}8}^{m}=(X_{\mathrm{FP}8}^{m}\mathbin{\texttt{>>}}(23-m)+R_{m})\mathbin{\texttt{<<}}(23-m)$.
      - Set $X_{\mathrm{FP8}}=X_{\mathrm{FP}8}^{\mathrm{sign}}+X_{\mathrm{FP}8}^{e}+X_{\mathrm{FP}8}^{m}$.
- **Return:** $X_{\mathrm{FP8}}$.

##### Floating-point Dequantization

Dequantizing floating-point numbers to higher bitwidth is straightforward. In the FP format system, the bitwidth of both the exponent and mantissa bits in lower bitwidth values will not exceed that in higher bitwidth. Therefore, we can directly extract and copy the sign bit, exponent and mantissa from the original value (with fewer bitwidth) to the most significant bits in the corresponding parts of the target value (with more bitwidth). And then we conduct zero filling on the rest bits for the exponent and mantissa parts ([Link](https://github.com/pytorch/pytorch/blob/main/c10/util/Float8_fnuz_cvt.h)).

##### Integer Quantization

We first scale the floating-point numbers to the representation span of $\mathrm{INT}k$ by dividing the scaling factor $s\in\mathbb{R}^{+}$, and adding a zero-point $z\in\mathbb{Z}$ to shift the clamped range [Wu20]. $\mathrm{round}(\cdot)$ is the round-to-the-nearest function, and $\mathrm{clamp}(\cdot,q^{\min},q^{\max})$ restricts values to be within the representation span of $k$-bit with $q^{\min}=-2^{k-1},q^{\max}=2^{k-1}-1$ in symmetric quantization and $q^{\min}=0,q^{\max}=2^{k}-1$ in asymmetric quantization. Therefore, the overall quantization formulation can be written as:

$$
X_{\mathrm{INT}_{k}}=\mathrm{clamp}\left(\mathrm{round}\left(\frac{X_{\mathrm{FP}}}{s}\right)+z,q^{\min},q^{\max}\right),
$$

where the scaling factor $s$ can be initialized as $s_{0}=({X_{\mathrm{FP}}^{\max}-X_{\mathrm{FP}}^{\min}})/$ $({q^{\max}-q^{\min}}),$
where $X_{\mathrm{FP}}^{\max}$
and $X_{\mathrm{FP}}^{\min}$ are the maximum and minimum values.

For system support, many frameworks apply the Marlin quantization ([Link](https://github.com/IST-DASLab/marlin)) [Fra24] as the standard process.
Algorithm 2 outlines the steps involved in Marlin quantization, and uses 4-bit integer quantization as an example. The values are quantized and stored as unsigned integers with the desired bitwidth. Extra pre/post-shift will be conducted to get the signed values. Therefore, we first scale the $X_{\mathrm{FP32}}$ values by $s$ and round it to integers. Then, adding $2^{k-1}$ to shift the values to non-negative integers within the span of uINT4 (4-bit unsigned integer). We omit the detail of C++ built-in datatype casting function `float2uint`. For clarity, we first explain the packing process with nested **for** loops, followed by its equivalent simplified form.
In 4-bit quantization, every 8 values are packed as a single uINT32, and the quantized matrix size is a quarter of the original. By using $i$`::`8, we abstract every 8 values along the dimension $C$ starting with $i$, incrementing by 8, and ending by default (till the end of dimension $C$). And then left shift the values by $4*i$
to place the 4-bit value to the corresponding bit range, and leave $4*i$ zeros on the right, allowing previously-stored quantized values to be preserved after `OR` operation.

**Algorithm 2: Marlin quantization from FP32 to INT4.**

- **Input:** $X_{\mathrm{FP32}}\in\mathbb{R}^{T,C}$ and $s\in\mathbb{R}^{+}$.
- **Output:** $X_{\mathrm{uINT}4}$.
- Set $X_{\mathrm{FP32}}^{\mathrm{round}}\leftarrow\mathrm{round}(X_{\mathrm{FP32}}/\mathrm{scale})$.
- **Shift to the span of uINT4:**
  - Set $X_{\mathrm{FP32}}^{\mathrm{clamp}}\leftarrow\mathrm{clamp}(X_{\mathrm{FP32}}^{\mathrm{round}}+2^{3},0,2^{4}-1)$.
  - Set $X_{\mathrm{uINT32}}\leftarrow\mathrm{float2uint}(X_{\mathrm{FP32}}^{\mathrm{clamp}})$.
- **Pack every eight $X_{\mathrm{uINT32}}$ values into a single uINT32:**
  - **For** $k\leftarrow 0$ **to** `C//8`:
    - **For** $i\leftarrow 0$ **to** $7$:
      - Set $X_{\mathrm{INT4}}[:,k]_{(4i+3:4i)}\leftarrow X_{\mathrm{uINT32}}[:,i+8k]\mathbin{\texttt{<<}}(4i)$.
- **Equivalent simplified form:**
  - `i::8` creates a sequence that starts at $i$, increments by $8$, and ends by default.
  - **For** $i\leftarrow 0$ **to** $7$:
    - Set $X_{\mathrm{uINT4}}[:,:]_{(:4i)}\leftarrow X_{\mathrm{uINT32}}[:,i\mathbin{\texttt{::}}8]\mathbin{\texttt{<<}}(4i)$.
- **Return:** $X_{\mathrm{uINT4}}$.

There are several custom algorithms that introduce faster data type conversions.
QQQ [Zha24a] designs a faster FP16 to INT8 conversion, named `FastFP16toINT8`. It starts by shifting the FP16 values to the representation span of uINT8 by adding 128. Next, adding an additional 1024 which effectively converts and places the 8 bits of uINT8 into the lower segment of the FP16 mantissa. Finally, the lower 8 bits from FP16 are extracted and applied with an `XOR` operation with `0x80` to obtain the desired INT8 format. The overall process can be further simplified to `FMA`, `PRMT`, and `XOR` operations in practice.

##### Integer Dequantization

It means projecting the integers back to the real numbers by multiplying the scaling factors, which can be expressed as:

$$
\hat{X}_{\mathrm{FP}}=s\cdot(X_{\mathrm{INT}x}-z)\approx X_{\mathrm{FP}}.
$$

Therefore, in many works $s$ can also be initialized by searching from candidates to find an optimal [Wei23]:

$$
s_{\mathrm{candidate}} =\frac{i}{\mathrm{num}_{i}}s_{0},\qquad i\in\mathbb{Z}^{+},i\in(0,\mathrm{num}_{i}).
$$

$$
\mathrm{s.t.}\;\min\|{X}_{\mathrm{FP}}-\hat{X}_{\mathrm{FP}}\|_{p}.
$$

where $\mathrm{num}_{i}$ means the number of candidates, which is always set as 50, 100 and so on [Yua24a, Wei23]. $s$ can also be a learnable parameter [Wei23, Sha23]. The way to find a better $s$ has been widely studied before LLMs emerged [Din24, Wei23a, Tia24a].

For system support, we first unpack the elements according to the way we pack them, and then multiply them to the corresponding scaling factor, which can be tensor-wise, channel-wise, token-wise, and other granularities described in Section 2.2. Custom implementations are also proposed, `SINT4toS8` [Li23a] designs a faster conversion from INT4 to INT8 by multiplying by 16.

##### Binarization

It takes the `sign` or `bool` function to abstract the sign:

$$
X_{\mathrm{sign}}=\begin{cases}1,&X_{\mathrm{FP}}\geq 0,\\ -1,&X_{\mathrm{FP}}<0,\end{cases}\quad X_{\mathrm{bool}}=\begin{cases}1,&X_{\mathrm{FP}}\geq 0,\\ 0,&X_{\mathrm{FP}}<0.\end{cases}
$$

Using `sign` or `bool` depends on the algorithm design, i.e. what value we expect the bits to represent. For example, binarized transformers always use `bool` function on attention scores and the post-ReLU activation. While the weight and activation in linear functions take `sign` function. Since the hardware always regards the bits as 0 or 1, we can assemble instructions to achieve any desired matrix multiplication ([Link](https://github.com/yifu-ding/BGEMM-CUDA)).
For example, on NVIDIA GPUs, the `mma` instruction takes 0/1 bit matrices and regards them as 0s and 1s while conducting bitwise accumulation operation `popcount`.
Therefore, to obtain the correct accumulation, the `popcount` function is designed with different arithmetic rules, i.e., if it substracts 1 for each 0, we can have the result of `sign` function.
It has many accelerated implementations, such as lookup table ([Link](https://github.com/WojciechMula)), nifty popcnt [Wil58], hacker popcnt [War12], hakmem popcnt ([Link](https://en.wikipedia.org/w/index.php?title=HAKMEM&oldid=1228234783)) and so on.

##### Binarization Dequantization

It is simply by multiplying a scaling factor $s$, i.e. $\hat{X}_{\mathrm{FP}}=s\cdot X_{\mathrm{sign}/\mathrm{bool}}$ to preserve the magnitude of the original values. It is easy to understand that large amounts of information will be lost in binarization. Therefore, few studies are devoted to binarizing LLMs due to the sharp performance degradation.
Due to the significant speedup and storage reduction, it is valuable to dig deeper into binarizing LLMs, but may require a new formulation beyond `sign` and `bool` functions. DB-LLM [Che24] proposes 2-bit weight quantization by decomposing to two 1-bit weight matrices, which can be efficient in MatMul theoretically.

## 4 Quantization Strategies for Efficient LLM Training

### 4.1 Low-bit Training

There are different strategies to accelerate the training of Large Language Models (LLMs) using low-bit. The common-used techniques contain BF16, FP16, FP8, and INT8 training.

**FP16 training:** Among all the data formats, BF16 training is widely used for LLMs since they are usually stable during training. However, they require hardware (e.g., A100, 4090, H100) support with Ampere or Hopper architectures. For some older hardware like Volta or Turing architectures (e.g., V100, T4), the data format is not available. In these cases, FP16 is often adopted to speed up the training, even for some small computer vision models. Since they have smaller exponent bits, FP16 faces a higher risk of encountering underflow or overflow issues. Therefore, a loss scaling strategy is proposed to preserve small or large gradient magnitudes. A detailed process is illustrated in Algorithm 3.

**Algorithm 3: Algorithm for Weight Update with FP16 Precision.**

- Maintain a primary copy of the weights in FP32.
- **While** not converged:
  - Make an FP16 copy of the weights.
  - Perform forward propagation with FP16 weights and activations.
  - Multiply the resulting loss by the scaling factor $S$.
  - Perform backward propagation with FP16 weights, activations, and their gradients.
  - Multiply the weight gradient by $1/S$.
  - Complete the weight update, including gradient clipping.

**FP8 training.** Since some hardware vendors like NVIDIA or AMD have designated new architectures supporting FP8 or FP4 formats. To achieve satisfactory acceleration with little modification, we can utilize the library Transformer Engine provided by vendors. While the dynamic range provided by the FP8 types is sufficient to store any particular activation or gradient, it is not sufficient for all of them at the same time. This makes the single loss scaling factor strategy, which worked for FP16, infeasible for FP8 training and instead requires using distinct scaling factors for each FP8 tensor. The scaling process can be formulated as:

$$
\mathrm{FP8\_MAX}=\mathrm{maximum\_representable\_value}(\mathrm{fp8\_format}),
$$

$$
\mathrm{exp}=\mathrm{get\_exponent}(\mathrm{FP8\_MAX}/\mathrm{amax}),
$$

$$
\mathrm{new\_scaling\_factor}=2.0^{\mathrm{exp}}.
$$

$\mathrm{fp8\_format}$ indicates the formats like E4M3 or E5M2. $\mathrm{FP8\_MAX}$ is the relevant max value under that format. $\mathrm{amax}$ is the maximal absolute value of the tensor. Then we can calculate the $\mathrm{new\_scaling\_factor}$ with $\mathrm{exp}$. However, the calculation of $\mathrm{new\_scaling\_factor}$ can not be online since it will introduce much more memory access. The best practice is to employ delayed scaling. This strategy chooses the scaling factor based on the maximums of absolute values seen in some number of previous iterations. This enables the full performance of FP8 computation but requires storing the history of maximums as additional parameters of the FP8 operators. Deepseek V3 [Dee24a], one of the state-of-the-art models, introduces fine-grained block-wise FP8 quantization, enabling highly accurate FP8 training. In [Table 3](#table-03), we list the prevalent frameworks and engines that support low-bit floating-point training, including the Deepspeed ([Link](https://github.com/microsoft/DeepSpeed)) from Microsoft, Megatron-LM ([Link](https://github.com/NVIDIA/Megatron-LM)) from NVIDIA, and UnitScaling ([Link](https://github.com/graphcore-research/unit-scaling)) from GraphCore.

<span id="table-03"></span>

![Systems for low-bit training.](../../papers/low-bit-llms/table-03.png)

**Table 3.** Systems for low-bit training.

**INT8 training:** During training, in addition to the model’s weight parameters, it is also necessary to save the gradients required by the optimizer and the backup information of the weights or gradients.

This makes the massive parameter scale of LLMs a more pronounced memory bottleneck during fine-tuning, hindering their deployment in broader application scenarios. INT8 Training [Zhu20] is considered a direct method to reduce the memory usage of gradients during training. However, the instability of quantization in backpropagation makes the training of LLMs more unstable and can even lead to crashes.
QST [Zha24c] proposes optimizing three key sources of memory usage simultaneously: model weights, optimizer states, and intermediate activations. In addition to quantizing the LLM model weights to 4 bits and introducing a separate side network that uses the LLM’s hidden states for task-specific predictions, QST also uses several low-rank adapters and gradient-free downsampling modules to significantly reduce the number of trainable parameters, thereby saving memory on optimizer states. Q-GaLore [Zha24d] points out that GaLore’s memory-saving strategy of projecting gradients using SVD incurs significant time costs. To address this, Q-GaLore adaptively updates the gradient subspace based on gradient convergence statistics and keeps the projection matrix in INT4 format and the weights in INT8 format, allowing Llama-7b to be trained from scratch on a single 16GB GPU.
Jetfire [Xi24] features an INT8 data flow to optimize memory access and a per-block quantization method to maintain the accuracy of pretrained transformers. 4-bit Optimizer [Li24a] uses a smaller block size and proposes to utilize both row-wise and column-wise information for better quantization, and further identifies a zero point problem of quantizing the second moment, solving it with a linear quantizer.

**Takeaways of subsection 4.1** BF16 and FP16 training have become widely adopted techniques to accelerate the training process, with relatively lower accuracy risks. FP8, while effective for fine-grained quantization in specific modules like linear layers, carries higher precision risks compared to BF16/FP16. INT8, which has been explored in some research but not yet widely adopted in practice, poses the highest accuracy risks. To mitigate these risks, techniques such as dynamic scaling are often introduced to adjust and stabilize the precision during training.

### 4.2 Quantization Strategies for PEFT

The well-pretrained LLMs possess excellent generalization and exhibit good transferability and adaptability during fine-tuning, making them potentially useful for a variety of downstream tasks. However, the massive parameter scale of LLMs creates a significant memory bottleneck during fine-tuning, hindering their broader application. Thus, the concept of parameter-efficient fine-tuning (PEFT) is introduced to address the issue of LLM fine-tuning under resource constraints [Din23, Han24].

As the demand for fine-tuning LLMs arises, it has been discovered that quantization can reduce memory usage during the fine-tuning process; some improve traditional QAT training, significantly reducing the parameter load during each update, while other class of methods combines quantization with the Low-Rank Adaptation (LoRA) fine-tuning approach.

#### 4.2.1 Partial Parameter Fine-Tuning with Quantization

Previous QAT methods require almost the same resources as full parameter training, making them infeasible in resource-constrained fine-tuning scenarios. Therefore, partial parameter fine-tuning strategies have been proposed. PEQA [Kim24] follows the naive QAT training approach. But after quantizing the weights $W$, it obtains scaling factors $s_{0}$ and fixed-point numbers $\overline{W}_{0}$, then it keeps $W$ and only trains $s_{0}$. OWQ [Lee24a] only updates the high-precision “weak columns" after mixed-precision quantization.

#### 4.2.2 Low-bit Low-Rank Adaptation

Low-Rank Adaptation (LoRA) [Hu21] freezes the pre-trained weights and only trains low-rank matrices. Although it reduces the trainable parameters by 10,000 times, it does not decrease the size of the pre-trained model weight itself, thus only reducing the memory requirements for fine-tuning by 3 times.

Methods like QLoRA [Det24] utilize low-bit quantization to further reduce the memory occupation by fine-tuning the LoRA for quantized LLMs. They first quantize the pre-trained LLM to low bits using PTQ methods:

$$
\textbf{W}_{q}\leftarrow \mathrm{quant}(\textbf{W}),
$$

Where **W** is the weight of each layer.

Then, they freeze all weight parameters and update only the LoRA during fine-tuning, with the forward pass as follows:

$$
\textbf{Y}=\textbf{X}\cdot \mathrm{dequant}(\textbf{W}_{q})+\textbf{X}\cdot\textbf{A}\textbf{B},
$$

Where **X** is the input of each layer.

In these methods, matrix **A** is typically initialized with random Gaussian values, while **B** is initialized to all zeros. This approach not only significantly reduces the memory footprint of the model’s weight parameters but also ensures that the optimizer only needs to store the gradients of LoRA during fine-tuning, greatly decreasing memory usage. QLoRA [Det24] introduces the use of Normal Float for double quantization of **W**, achieving both good accuracy retention and memory savings, allowing fine-tuning of a 65B pre-trained model using a single 48GB GPU. IR-QLoRA [Qin24b] incorporates information theory into the QLoRA paradigm, enhancing fine-tuning performance through information calibration and connection. LoRA+ [Hay24] demonstrates that setting different learning rates for matrices A and B in LoRA enables efficient feature learning. QDyLoRA [Raj24] and Bayesian-LoRA [Meo24] employs more flexible rank allocation within LoRA.

Moreover, some methods aim to obtain a deployable quantized and merged model after the LoRA fine-tuning. QA-LoRA [Xu23] uses INT format to quantize **W** and adjusts $\textbf{X}\cdot\textbf{A}^{i\times r}\textbf{B}^{r\times o}$ to $\mathrm{mean}(\textbf{X})\cdot\textbf{A}^{\frac{i}{L}\times r}\textbf{B}^{r\times o}$, allowing the fine-tuned **A****B** to be losslessly merged into the INT format $\textbf{W}_{q}$, without extra computation when deployment. L4Q [Jeo24], on the other hand, maintains the dimension $\textbf{A}\in\mathbb{R}^{i\times r}$ and directly uses the full QAT forward propagation method, simultaneously updating the quantizer parameters $s$ and $b$ for $\textbf{A},\textbf{B}$, and $\textbf{W}+\textbf{A}\textbf{B}$. While L4Q does not reduce the memory footprint of the weights through quantization during pre-training, the optimizer still does not need to retain the gradients of the weights, resulting in a fine-tuned quantized model that can be deployed directly with higher accuracy.

Many methods have recognized that the initialization of LoRA significantly impacts the effectiveness of these quantization-based parameter-efficient fine-tuning methods. As a result, they aim to minimize $\left\|\textbf{W}-(\textbf{W}_{q}+\textbf{A}\textbf{B})\right\|_{F}$ before fine-tuning. LoftQ [Li23b] and LQ-LoRA [Guo23a] both achieve this through iterative computation: $Q_{t}\leftarrow \mathrm{quant}(\textbf{W}-\textbf{A}_{t-1}\textbf{B}_{t-1}^{\top})$ and $\textbf{A}_{t},\textbf{B}_{t}\leftarrow \mathrm{SVD}(\textbf{W}-Q_{t})$. LQ-LoRA also suggests incorporating calibration data, adjusting the minimization objective to $\left\|\sqrt{F}\odot(\textbf{W}-(\textbf{W}_{q}+\textbf{A}\textbf{B}))\right\|_{F}^{2}$, where $F$ is the Fisher information matrix for **W**, and $\odot$ represents the Hadamard product. Additionally, LQ-LoRA introduces dynamic quantization configurations to better adapt to resource constraints.

<span id="figure-07"></span>

![Illustrations for different LoRA structures.](../../papers/low-bit-llms/figure-07.png)

**Figure 7.** Illustrations for different LoRA structures.

[Figure 7](#figure-07) is an illustration of different LoRA structures. [Figure 7(a)](#figure-07) represents methods like QLoRA that do not alter any part of the LLM during the fine-tuning stage and keep the complete original LoRA structure [Det24, Qin24b, Hay24, Li23b]. [Figure 7(b)](#figure-07) represents methods like QA-LoRA that also do not change any part of the LLM during fine-tuning but modify the original LoRA structure [Xu23]. [Figure 7(c)](#figure-07) represents methods like L4Q that modify the original LoRA structure and use a training process similar to QAT [Jeo24].
Both (a) and (b) require only the quantized LLM weights $W_{q}$ during fine-tuning, while (c) needs to store the pre-trained full-precision weights $W_{\mathrm{fp}}$. (a) is solely intended to reduce training costs and cannot directly produce a quantized model after fine-tuning, while both (b) and (c) can directly integrate the LoRA module after fine-tuning to produce a deployable quantized model. Unlike the weight-only quantization in these methods, RoLoRA [Hua24] incorporates rotations with LoRA for effective weight-activation quantization. Although there are existing LoRA works on MoE [Li24b, Luo24, Gao24], they have not yet focused on quantization. In the context of quantization, it is crucial to assess whether reducing bit precision exacerbates the expert imbalance problem. Additionally, it is important to explore which position should use the LoRA-MoE method for quantization-aware training (including the router and load balancing) and to examine whether allocating more bits to deeper layers is necessary [Gao24].

**Takeaways of subsection 4.2** To reduce memory usage during quantization-aware training, a common strategy is to employ partial weight updates, such as updating only a subset of weight columns. To reduce memory usage during normal fine-tuning, quantization can be combined with low-rank approximation techniques, enabling fixed weights to be quantized to lower bit-widths for further memory reduction.

## 5 Quantization Algorithms for Efficient LLM Inference

This section navigates through the algorithms of LLM quantization. Quantization algorithms can be broadly divided into two primary approaches: Quantization-Aware Training (QAT) and Post-Training Quantization (PTQ). QAT integrates quantization into the training/fine-tuning process, enabling the model to learn and adapt to the quantization constraints, thereby minimizing the accuracy loss associated with lower precision. In contrast, in the scenario of PTQ, we are given a pre-trained floating-point model along with a small amount of calibration data, aiming to generate an accurate quantized model without an end-to-end training process. We will delve into these quantization algorithms in detail. By the end of this section, we hope our survey can serve as a thorough and systematic collection of the various quantization algorithms applicable to LLMs, their implementation strategies, and their implications for model performance and efficiency.

<span id="table-04"></span>

![Comparison of different QAT methods.](../../papers/low-bit-llms/table-04.png)

**Table 4.** Comparison of different QAT methods.

### 5.1 Quantization-Aware Training

[Table 4](#table-04) summarizes the different QAT methods for LLMs. LLM-QAT [Liu23b] is the pioneering work that investigates the QAT for LLMs. To overcome the training data limits, it proposes data-free knowledge distillation which aligns the teacher logits of full-precision models and student logits of quantized models. Following LLM-QAT, BitDistiller [Du24] employs the asymmetric clipping strategy for asymmetric quantization during the self-distillation stage. EfficientQAT [Che24a] significantly reduces the training cost by splitting the QAT into two consecutive phases. The first phase optimizes all parameters for each block and then the second phase merely optimizes quantization parameters for the entire network. To pave the way for a new era of extreme quantization level, BitNet [Wan23] replaces the BitLinears with original Linears and trains from scratch. Its variant, BitNet b1.58 [Ma24], leverages ternary weight for each parameter which achieves near-lossless performance.

**Takeaways of subsection 5.1** Quantization-aware training (QAT) is particularly beneficial in extremely low-bit scenarios, despite its more complex training process. If your goal involves ultra-low-bit configurations and sufficient computational resources are available, QAT can be an effective solution. However, starting QAT from scratch can be challenging; it is generally more practical and efficient to fine-tune a pre-trained model using QAT. Additionally, it is crucial to select training data that generalizes well across diverse domains to mitigate the risk of overfitting.

### 5.2 Post-Training Quantization

<span id="figure-08"></span>

![An overview of the PTQ algorithms.](../../papers/low-bit-llms/figure-08.png)

**Figure 8.** An overview of the PTQ algorithms.

Post-Training Quantization (PTQ) is a technique that applies quantization to a pre-trained model. Unlike QAT, PTQ does not require the model to be trained with quantization modules. This makes PTQ a highly practical approach for deploying models that were originally trained with high precision. PTQ is particularly useful when access to the training data is limited or when retraining is computationally expensive. Therefore, with the development of the LLMs, the past few years have witnessed a remarkable surge in PTQ algorithms because of their small training cost.

To have a better introduction, we systematically divide PTQ algorithms into several categories, as described in [Figure 8](#figure-08).

#### 5.2.1 Equivalent Transformation

Many studies [Luo20, Bon21, Wei23, Xia23] have highlighted the presence of significant outliers in LLMs. These outliers pose substantial challenges for quantization, as they force a large number of normal values to be represented with a limited number of bits, which leads to large quantization errors and accuracy degradation. Therefore, a multitude of algorithms have emerged in recent years, aiming to mitigate the issue of outliers in LLMs.

Among all the algorithms addressing the outlier problem, equivalent transformation is one of the most representative and effective methods. One of the pioneering works in applying the equivalent transformation to language models is the Outlier Suppression (OS) [Wei22]. OS splits the LayerNorm function and migrates $\gamma$, which is a parameter of LayerNorm, to avoid the outlier.

$$
\textbf{X}_{j}=\textbf{X}^{\prime}_{j}\cdot\gamma_{j}
$$

Then the LayerNorm becomes the non-scaling one, and the weight of the next layer can absorb the $\gamma$:

$$
\textbf{W}(x\odot\begin{bmatrix}\gamma_{1}\\ \gamma_{2}\\ \cdots\\ \gamma_{n}\end{bmatrix})=(\textbf{W}\odot\begin{bmatrix}\gamma_{1}&\gamma_{2}&\cdots&\gamma_{n}\\ \gamma_{1}&\gamma_{2}&\cdots&\gamma_{n}\\ \cdots\\ \gamma_{1}&\gamma_{2}&\cdots&\gamma_{n}\end{bmatrix})x
$$

By doing so, OS can suppress the outliers. Starting from the OS method, numerous subsequent equivalent transformation techniques have emerged. Most equivalent transformation methods alleviate the impact of outliers on quantization by making the outliers in weights or activations more symmetrical and smooth, which can be formulated as follows:

$$
\begin{aligned}
\textbf{Y}&=\textbf{X}\textbf{W}+\textbf{B} \\
&=[(\textbf{X}-\Delta)\cdot\textbf{M}^{-1}]\cdot[\textbf{M}\cdot\textbf{W}]+(\textbf{B}+\Delta\cdot\textbf{W}),
\end{aligned}
$$

where $\Delta$ is a shifting factor used to make the distribution of outliers in the input symmetrical, and **M** is a matrix used to make the distribution smoother. By adopting the aforementioned equivalent transformation, many existing quantization methods have achieved state-of-the-art (SOTA) performance under various quantization settings and scenarios.

Based on the implementation, equivalent transformation can be further subdivided into three types: shifting transformation, scaling transformation, and rotation transformation. We then independently provide a detailed introduction for each type.

<span id="figure-09"></span>

![Overall diagram of shifting transformation. $\Delta_{1}$ and $\Delta_{2}$ represent two types of shifting operation. $\Delta_{1}$ can be merged into the parameter $\beta$ in Layernorm and the weight metrics. Specifically, $\Delta_{2}$ can not be merged into the weight matrix. Therefore, the shift transformation between value projection $W_{v}$ and out projection $W_{o}$ can only be conducted online, which may raise extra computation burden.](../../papers/low-bit-llms/figure-09.png)

**Figure 9.** Overall diagram of shifting transformation. $\Delta_{1}$ and $\Delta_{2}$ represent two types of shifting operation. $\Delta_{1}$ can be merged into the parameter $\beta$ in Layernorm and the weight metrics. Specifically, $\Delta_{2}$ can not be merged into the weight matrix. Therefore, the shift transformation between value projection $W_{v}$ and out projection $W_{o}$ can only be conducted online, which may raise extra computation burden.

##### Shifting Transformation

Outliers in LLMs are asymmetrically distributed across different channels. This asymmetrical representation can cause a tensor composed of channels with small ranges to exhibit a very large overall range, resulting in difficulty in the quantization process. To address this issue, OS+ [Wei23] first proposes the channel-wise shifting transformation, which adjusts activations across channels to mitigate the impact of asymmetry as the following equation:

$$
\hat{X}=X-\Delta,
$$

where $\Delta\in\mathbb{R}^{c\times 1}$ serves as a row vector and shifts each channel of the activations. Note that this operation is not the conventional shifting operation used in symmetric quantization. Instead, it operates on a channel-wise basis and provides a better distribution for per-tensor quantization. In detail, OS+ defined $\Delta$ in a handicraft-way:

$$
\Delta_{j}=\frac{\max(\textbf{X}_{:,j})+\min(\textbf{X}_{:,j})}{2}.
$$

With the channel-wise shifting in place, the tensor range is reduced to the largest channel range, eliminating the influence of asymmetric outliers. However, handcrafting the equivalent parameters leads to sub-optimal results. Hence OmniQuant [Sha23] is proposed to determine the optimal shifting parameters in a differentiable way by including the block-wise quantization error minimization:

$$
\underset{\Delta}{\mathrm{arg}\,\min}\,\|\mathcal{O}(\textbf{W},\textbf{X})-\mathcal{O}\big(Q_{w}\left(\textbf{W};\Delta\right),Q_{a}\left(\textbf{X};\Delta\right)\big)\|,
$$

where $\mathcal{O}$ represents the mapping function for a transformer block in the LLM, $Q_{w}(\cdot)$ and $Q_{a}(\cdot)$ denote the weight and activation quantizer respectively, $\Delta$ is the shifting parameter. Block-wise minimization is easy to optimize with minimal resource requirements.
Therefore, by optimizing the objective function block by block, a more effective shifting vector can be obtained compared to the direct computation used in OS+ in an efficient and resource-saving way. However, OmniQuant requires fine-tuning of the learnable parameters; otherwise, issues such as gradient explosion can easily occur. Similar to OmniQuant, AffineQuant [Ma24a] also adopts a learning-based shifting operation.

We illustrate the diagram of shifting transformation as shown in [Figure 9](#figure-09). The shifting factor $\Delta$ can be fused in LayerNorm and weight matrix so that no further overhead is needed.

##### Scaling Transformation

<span id="figure-10"></span>

![Overall diagram of scaling transformation. $\Phi$ can be merged into the parameter $\gamma$ in Layernorm and the weight metrices.](../../papers/low-bit-llms/figure-10.png)

**Figure 10.** Overall diagram of scaling transformation. $\Phi$ can be merged into the parameter $\gamma$ in Layernorm and the weight metrices.

Shifting transformation effectively addresses the issue of asymmetrical distribution of outliers in activations, reducing the large range caused by the asymmetry. However, this only aids per-tensor quantization and does not reduce the difficulty of per-channel quantization, as it does not fundamentally eliminate the outliers distributed across channels in the activations. To further reduce the impact of outliers on quantization, SmoothQuant [Xia23] initially proposes to use a scaling transformation. It relies on a key observation: although activations are much more difficult to quantize than weights due to the presence of outliers, different tokens exhibit similar variations across their channels [Det22]. Based on this observation, SmoothQuant migrates the quantization difficulty from activations to weights offline by introducing a mathematically equivalent per-channel scaling transformation that significantly smooths the magnitudes across channels:

$$
\textbf{Y}=(\textbf{X}\mathrm{diag}(\Phi)^{-1})\cdot(\mathrm{diag}(\Phi)\textbf{W})=\hat{\textbf{X}}\hat{\textbf{W}},
$$

where $s$ is a smoothing factor. Note that $\mathrm{diag}(\Phi)$ corresponds to the matrix $M$ in Equation 22, but it is a diagonal matrix used to achieve per-channel smoothing. SmoothQuant introduces a hyper-parameter $\alpha$ as the migration strength to control how much difficulty to migrate from activation to weights, using the following equation:

$$
\Phi_{j}=\frac{\max(|\textbf{X}_{j}|)^{\alpha}}{\max(|\textbf{W}_{j}|)^{1-\alpha}}.
$$

However, this method requires multiple trials to determine the optimal migration strength for different models, i.e., $\alpha=0.5$ is a well-balanced point for all OPT [Zha22] and BLOOM [Les23] models.

Inspired by SmoothQuant, FPTQ [Li23c] argues that it is unnecessary to consider weights for computing the activation smoothing scale while it is crucial to retain all the activation values with a non-linear lossless mapping. This mapping needs to fit two criteria: (1) touching gently with the inliers and (2) harshly suppressing the outliers. Based on this, they adopt a logarithmic
function to improve the calculation of the smooth matrix $s$:

$$
\Phi_{j}=\frac{\max(|\textbf{X}_{j}|)}{\log_{2}(2+\max(\textbf{X}_{j}))}.
$$

In addition to FPTQ, many other works have followed the approach of SmoothQuant. Both OS+ and AWQ [Lin24] use searching-based methods to find the smooth scale. However, the optimization objectives and search spaces of the two methods differ. The optimization objective of OS+ is:

$$
\begin{aligned}
\Phi^{*} = \underset{\Phi}{\mathrm{arg}\,\min}\,\mathbb{E}\|Q\big((\textbf{X}-\Delta)\cdot \mathrm{diag}(\Phi)^{-1}\big)Q\big(\mathrm{diag}(\Phi)\cdot\textbf{W}^{\mathsf{T}}\big) \\
+\hat{\textbf{b}}-(\textbf{X}\textbf{W}^{\mathsf{T}}+\textbf{b})\|^{2}_{F}.
\end{aligned}
$$

To simplify the search space, OS+ optimizes the outlier threshold $t$, compressing channels with an activation range over $t$ into $(-t,t)$ and leaving others unchanged. This reduces the problem to a single variable. A grid search is then used for $t$ to minimize the objective. After finding the optimal $t$, the scaling vector is calculated as follows:

$$
\Phi_{j}=\max(1.0,\frac{\max(\textbf{X}_{:,j}-\Delta_{j})}{t}).
$$

AWQ finds that the saliency of weight channels is actually determined by the activation scale. To this end, it adopts an activation-awareness optimization objective and uses a very simple search space:

$$
\begin{aligned}
\Phi ={\Phi_{x}}^{\alpha}, \\
\alpha^{*} =\underset{\alpha}{\mathrm{arg}\,\min}\,\left\|Q\!\left(\textbf{W}\cdot\mathrm{diag}({\Phi_{x}}^{\alpha})\right)(\mathrm{diag}({\Phi_{x}}^{\alpha}))^{-1}\textbf{X}-\textbf{W}\textbf{X}\right\|,
\end{aligned}
$$

where ${\Phi_{x}}$ is the average magnitude of activation (per channel), and use a single hyper-parameter $\alpha$ to balance between the protection of salient and non-salient channels.

In addition to searching-based methods, some approaches use learning-based techniques to find the optimal scaling matrix. OmniQuant and AffineQuant also learn the scaling matrix. In Equation 25, OmniQuant learns both the shifting factor $\Delta$ and the scaling matrix $\mathrm{diag}(\Phi)$. However, OmniQuant optimizes only within the range of a diagonal matrix. AffineQuant [Ma24a] argues that this limited search range can lead to significant quantization errors, reducing the generalizability of the quantization method in low-bit scenarios. It proposes learning a general invertible matrix to perform equivalent affine transformations on weights and activations, achieving better results.

We also illustrate the diagram of scaling transformation in [Figure 10](#figure-10). The same as shifting transformation, scaling factor $\Phi$ can be merged into layernorm and weight matrix.

##### Rotation Transformation

Rotation transformation was first introduced by QuIP [Che24b]. QuIP is based on the insight that quantization works better when the weight and Hessian matrices are incoherent. This means that the weights should have similar magnitudes and the directions that require precise rounding should not align with the coordinate axes. To make it straight, a weight matrix is $\mu$-incoherent if:

$$
\max(\textbf{W})\leq\mu\|\textbf{W}\|_{F}/\sqrt{mn},
$$

where $mn$ is the number of the matrix elements and $\|\cdot\|_{F}$ is the Frobenius norm. QuIP shows that multiplying a weight matrix on the left and right by an orthogonal matrix can reduce incoherence, which is equal to performing a rotation transformation on the weight matrix. QuIP utilizes Kronecker-structured orthogonal matrices, allowing for rapid additional computations. Building on this, QuIP# [Tse24] replaces these with Hadamard matrices, enhancing quantization through better incoherence and speeding up the forward pass, as the Hadamard transform can be computed in $\mathcal{O}(n\log n)$ addition operations.

Both of these two methods target weight-only quantization. Following these approaches, QuaRot [Ash24] introduces a weight&activation quantization method that also quantizes the KV cache. QuaRot operates in two stages. First, the model weights are manipulated in full precision, and two Hadamard operations are added to the model’s forward pass. In the second stage, the weights are quantized using an existing method, and quantization operations are integrated into the forward pass for online activation quantization.

<span id="figure-11"></span>

![Overall diagram of the rotation transformation. The rotated activations exhibit fewer outliers and are easier to quantize. $R_{1}$ and $R_{2}$ are randomized matrices which can be merged into the weights matrices. $R_{3}$ and $R_{4}$ can not be merged and are usually Hadamard matrices.](../../papers/low-bit-llms/figure-11.png)

**Figure 11.** Overall diagram of the rotation transformation. The rotated activations exhibit fewer outliers and are easier to quantize. $R_{1}$ and $R_{2}$ are randomized matrices which can be merged into the weights matrices. $R_{3}$ and $R_{4}$ can not be merged and are usually Hadamard matrices.

However, both the orthogonal matrices in QuIP and the Hadamard matrices in QuIP# and QuaRot are randomly generated. Although these works have shown that these randomly generated matrices can alleviate the outlier problem to some extent, they are not optimal.
SpinQuant [Liu24b] finds that the performance of a quantized network can vary significantly with different rotation matrices. For example, the average accuracy on downstream zero-shot reasoning tasks can fluctuate by up to 13 points depending on the rotation used on the MMLU benchmark. Therefore SpinQuant proposes a learning-based rotation transformation. The rotation matrix is learned using the Cayley SGD method, with the following optimization objective:

$$
\textbf{R}^{*}=\underset{\textbf{R}\in\mathcal{M}}{\mathrm{arg}\,\min}\,\mathcal{L}_{Q}(\textbf{R}|\textbf{W},\textbf{X}).
$$

Here, $\mathcal{M}$ presents the Stiefel manifold, i.e., the set of all orthogonal matrices. $\mathcal{L}_{Q}(\cdot)$ denotes the task loss. By employing the learned matrix, the performance is improved significantly and the variance becomes much smaller compared with randomized matrices. The diagram in SpinQuant [Liu24b] effectively illustrates the overall process of the rotation transformation, so we have borrowed it for our purposes as shown in [Figure 11](#figure-11). Specifically, for Quarot [Ash24], since it employs a head-wise rotation transformation at $R_{2}$, an online Hadamard matrix needs to be inserted before quantizing the attention output to achieve an equivalent transformation. DuQuant [Lin24b] identifies the limitations of these methods in smoothing massive outliers and therefore utilizes rotation and permutation transformations based on prior knowledge. Meanwhile, unlike SpinQuant, a greedy search strategy is employed to optimize the rotation matrix. PrefixQuant [Che24c] discovers the token-wise outliers, especially appearing in initial tokens and low-semantic tokens. Since these tokens remain unchanged across all inputs, PrefixQuant stores their KV cache through offline prefilling.

We can observe that scaling transformation and rotation transformation can be utilized for the different parts of LLM quantization. QServe [Lin24a] is a co-designed quantization system for efficient LLM serving, combining scaling and rotation transformations. For where additional overhead is required for the online computation of rotation matrices, QServe uses scale transformation as a substitute for rotation operations, thereby avoiding the extra overhead.

#### 5.2.2 Compensation

The weight compensation technique, originally stemming from Optimal Brain Damage (OBD) [Lec89], involves a Taylor series expansion of the objective function. This method assumes that upon the removal of any given parameter, the influence of the remaining parameters on the objective function remains unchanged. Based on OBD, OBS [Has93] and OBQ [Fra22a] calculate the impact of each parameter weight on the objective function by solving the inverse Hessian matrix. Concurrently, they compute a compensation term applied to the remaining weights to offset the error introduced by each weight adjustment.

Although one-by-one weight quantization methods have achieved satisfactory performance on smaller models, the computational overhead becomes prohibitive when scaling to larger models. To accelerate quantization, GPTQ [Fra22] quantizes the weights column-by-column, and the rounding errors are compensated using second-order information. Specifically, this algorithm compensates for the quantization error induced by the quantized weights $\mathrm{Quant}(\mathbf{W}_{i})$ by adjusting the subset R of full-precision weights $R$ with an update $\boldsymbol{\delta}_{R}$:

$$
\mathbf{W}_{i}=\underset{\mathbf{W}_{i}}{\mathrm{arg}\,\min}\,\frac{(\mathrm{Quant}(\mathbf{W}_{i})-\mathbf{W}_{i})^{2}}{[\mathbf{H}_{R}^{-1}]_{ii}}.
$$

$$
\boldsymbol{\delta}_{R}=-\frac{\mathbf{W}_{i}-\mathrm{Quant}(\mathbf{W}_{i})}{[\mathbf{H}_{R}^{-1}]_{ii}}\cdot(\mathbf{H}_{R}^{-1})_{:,i}.
$$

where the Hessian matrix is $\mathbf{H}_{R}=2\mathbf{X}_{R}\mathbf{X}_{R}^{\top}$. Based on GPTQ, several works have been successively proposed. QuantEase [Beh23] utilizes the Coordinate Descent to compute more precise compensation for the unquantized weights. QQQ [Zha24a] adopts the GPTQ for the transferred weights by OS+ [Wei23].

#### 5.2.3 Mixed-precision

As aforementioned, the presence of outliers is widely found in the activations and weights of large language models, which poses a significant challenge for quantization. Consequently, the motivation of numerous mixed-precision methods for LLMs is to represent a small number of outlier values in higher precision and other values in lower precision separately. Similarly, depending on the granularity of mixed precision, methods can be categorized into element-wise, channel-wise, token-wise and tensor-wise approaches, as described in Section 2.2.

**Element-wise.**
SpQR [Det23] was the first to demonstrate that outliers also exist in weights. It identifies and isolates these outlier weights based on their sensitivity, saving them as a highly sparse, higher-precision matrix. SqueezeLLM [Kim23] adopts non-uniform quantization for non-salient weights, which achieves near-lossless performance. Similarly, CherryQ [Cui24] defines heterogeneity to identify the critical cherry parameters. To explore extreme compression rate, PB-LLM [Sha23a] is the first to binarize the non-salient weights in LLMs. Since PB-LLM still allocates high precision to 10%-30% of salient weights, BiLLM [Hua24a] employs residual approximation for salient weights and group quantization for non-salient weights, reducing the quantization bit-width of LLM weights to 1.08 bits. GEAR [Kan24] extends the concept of mixed precision to the KV cache compression and utilizes low-rank matrices to approximate the quantization residuals.

**Channel-wise.**
LLM.int8() [Det22] splits the weights and activations into two independent parts according to the outlier channels to minimize output quantization errors in activations, which effectively reduces the GPU memory usage during inference. OWQ [Lee24a] proposes a sensitivity-aware mixed-precision scheme to identify the weak columns by Hessian metric. Furthermore, OWQ also provides weak column tuning (WCT) to enable accurate parameter-efficient fine-tuning for task-specific adaptation. RPTQ [Yua23] observes the varying ranges across channels in activations pose challenges for quantization. Therefore, RPTQ reorders the channels into different clusters with respective quantization. Atom [Zha24e] employs dynamic reorder for activations and static reorder for weights to remain aligned with the corresponding activation channels. Atom further quantizes the KV cache to 4-bit which significantly boosts serving throughput. Inspired by information theory, CQ [Zha24b] couples multiple key/value channels together and jointly quantize them.

**Token-wise.** Some KV cache quantization studies, such as KVQuant [Hoo24], IntactKV [Liu24a] and SKVQ [Dua24] discover token-wise outliers caused by special tokens (first token or low-semantic-value tokens) significantly influence the performance. So they store these token-wise outliers with higher precision in advance. KIVI [Liu24c] and WKVQuant [Yue24a] keep the most recent KV cache in full-precision and quantize the past KV cache. MiKV [Yan24c], Zipcache [He24a], and Snapkv [Li24c] retain the important KV pairs in high precision based on distinct metrics. QAQ [Don24] dynamically allocates the adaptive bits for the different tokens.

**Tensor-wise.**
LLM-MQ [Li23d] assigns higher bit-widths to more sensitive layers based on first-order information and quantization error. CacheGen [Liu24d] identifies LLM is more sensitive to losses in the KV cache values of the early layers than to losses in those
of the deeper layers. It assigns higher-bit precision in sensitive early layers. The QuantMoE-Bench [Li24d] investigates the weight bits among different blocks, experts, and linear layers, revealing that the varying numbers of weight bits are effective.

#### 5.2.4 Combination

Although current quantization methods for large models have achieved relatively good results, their performance under extremely high compression rates is still unsatisfactory due to the limited representation capacity of low-bit quantization. Currently, commonly used compression methods including low-rank decomposition, model sparsification, and model distillation are explored to combine with quantization.

##### Low-rank

Although QAT is generally considered to offer the best accuracy, its high memory cost makes it difficult to apply to LLMs. Therefore, some methods consider introducing LoRA or other matrix decomposition methods as a trade-off between PTQ and QAT. Unlike PEFT discussed in Section 3.3, these methods aim to reduce quantization error using techniques like LoRA or SVD to achieve a quantized model closer to the full-precision model, rather than enhancing learning ability on fine-tuning datasets.
Some works have used LoRA to achieve parameter-efficient QAT. LR-QAT [Bon24] computes $s\cdot\mathrm{clamp}(\textbf{W}_{q}+\textbf{A}^{i\times r}\textbf{B}^{r\times o})$ during the forward pass and does not update **W** during the backward pass, allowing a 7B LLM to be trained on a single consumer-grade GPU with 24GB of memory. This approach results in a quantization-friendly model after fine-tuning. LLM-QFA aims to produce models with various bit widths through a single supernet training, significantly reducing the resource overhead of this production method by leveraging the low resource cost of LoRA. INT2.1 [Cha23] utilizes LoRA to shift the optimization target from minimizing per-layer or per-block quantization error to minimizing the overall output error of the model. Through end-to-end fine-tuning, it reduces the distance between the output distribution and its corresponding original full-precision counterpart.
Other works have reduced quantization errors through matrix decomposition. LQER [Zha24f] applies SVD to quantization errors and uses an activation-induced scaling matrix to guide the singular value distribution toward the desired pattern. Delta-CoMe [Pin24] discovers that the singular values of delta weights exhibit a long-tailed distribution after applying SVD, and proposes a mixed-precision delta quantization method that uses high-bit representations for the singular vectors corresponding to these singular values. ZeroQuant-V2 [Yao23] introduced an optimized low-rank compensation method that enhances model quality recovery by leveraging a low-rank matrix obtained through SVD of the quantization errors. LCQ [Cai24] uses low-rank codebooks with a rank greater than one for quantization, addressing the issue of accuracy loss when using rank-one codebooks under high compression ratios.

##### Sparsification

Model sparsification aims to remove unimportant weights to accelerate the model, while quantization further reduces the remaining weights using lower-bit representations. Therefore, the two methods can be effectively used in a complementary manner. SDQ [Jeo24a] first sparsifies the weights of LLMs based on the magnitude as much as possible until the quality of the LLM is significantly impacted (e.g., a 1% increase in perplexity). Then it utilizes a mixed-precision quantization method to deal with the outliers. However, this method does not take into account the coupling of the two approaches.
Sparsification and quantization often conflict with each other. Sparsification tends to preserve parameters with large absolute values in LLMs [Han15, Sun23], while quantization prefers a smaller range of parameter values [Wei23]. As a result, the parameters preserved during sparsification may degrade the performance of quantization. JSQ [Guo24] design a new sparsity metric to address this issue:

$$
\begin{aligned}
\textbf{I}_{ij} =\|\textbf{X}\|_{2}\cdot\|\textbf{W}\|, \\
\textbf{A}_{ij} =\max(\hat{\textbf{Y}}_{:i})-\min(\hat{\textbf{Y}}_{:i}), \\
\mathrm{where}\qquad\hat{\textbf{Y}}=\textbf{X}\cdot(\Theta(\textbf{W};i;j))^{\mathsf{T}}, \\
\textbf{S}_{ij} =\textbf{I}_{ij}+\lambda\textbf{A}_{ij}.
\end{aligned}
$$

Here, $\Theta(\textbf{W};i;j)$ denotes an auxiliary weight matrix when setting the element at $i$th row and $j$th column as 0 in **W**. $\lambda$ is a trade-off factor. By using this metric, a better trade-off between preserving outliers for more information and minimizing the activation range for better quantization can be achieved.

##### Quantization

In addition to combining quantization with other compression methods, different quantization techniques can also be integrated to achieve better results. A recent work [Sha24a] combines the SmoothQuant and GPTQ together. Actually, most of the equivalent transformation methods and the compensation quantization methods are orthogonal which can be merged for further exploration.

#### 5.2.5 More LLM-Based Architectures

Besides traditional dense LLMs, the quantization methods tailored for multimodal large language models (MLLMs) and mixture-of-expert (MoE) models have also garnered widespread attention. Q-VLM [Wan24b] offers the first post-training quantization framework for MLLMs by mining cross-layer dependency to achieve satisfying trade-offs
between discretization errors and the search cost. MQuant [Yu25a] proposes a static solution, utilizing seperate quantization parameters for visual and language modality. Furthermore, it relieves weight outliers arising from online Hadamard rotations. MBQ [Li24e] also considers the sensitivity between language and vision modality, which adjusts the reconstruction loss for the optimal channel-wise equalization factors. QuantMoE-Bench [Li24d] explores the structure-aware mix-precision quantization schemes for MOE models, indicating different MoE structures require varying numbers of bits. MC-MOE [Hua24b] converts the bit allocation problem into a Linear Programming (LP) problem and balances the importance between each expert.

#### 5.2.6 More Quantization Forms

Beyond integer quantization, more forms of quantization are being introduced for LLMs, as they can also compress the average bit-width of a 32-bit or 16-bit model down to 4 or lower bits.
While these methods do not always offer significant acceleration benefits when saving memory, they generally lead to improvements in precision.

##### More Quantization Datatypes

Integer quantization typically assigns a single scaling factor to an entire block and quantizes each element individually into an integer number. This reduces memory usage while also enabling the acceleration of fixed-point operations after quantizing both weights and activations. However, as higher precision is demanded for LLM quantization, formats that better match the original distribution of values have been proposed.
Normal Float [Det21, Det24], proposed alongside Quantile Quantization, is based on the assumption that the weight distribution follows a normal distribution. It is considered an information-theoretically optimal data type that ensures each quantization bin has an equal number of values assigned from the input tensor. However, Dotzel et al. [Dot24] conducted a statistical analysis and found that the distributions of most LLM weights and activations follow a Student’s t-distribution. Based on this, they derived a new theoretically optimal format, Student Float (SF4).
Floating-Point (FP) quantization offers better hardware support compared to NF/SF and is more flexible than integer quantization, allowing it to more effectively handle long-tail or bell-shaped distributions. Since FP can support flexible allocation of exponent and mantissa bits, several allocation schemes have been proposed. FPQ [Liu23c] determines FP quantizers through a joint format and max value search combined with a pre-shifted exponent bias. FP8 quantization [Kuz22] tests various allocation schemes by evaluating metrics like quantization error and proposes FP8 quantization simulation for learnable allocation and quantization.

##### Vector Quantization

Vector Quantization (VQ) quantizes multiple vector dimensions jointly. It achieves this by learning codebooks $C_{1},...,C_{M}$, each containing $2^{B}$ vectors (for B-bit codes). To encode a given database vector, VQ splits it into sub-groups of entries, and then encodes every group by choosing a vector from the learned codebook. A part of the weights of $i$-th layer is encoded by choosing a single code from each codebook and summing them up:

$$
\widehat{W}_{i,j}=\sum_{m=1}^{M}{C_{m}d_{ijm}}
$$

where $d_{ijm}\in\mathbb{R}^{2^{B}}$ represents a one-hot code for the $i$-th output unit, $j$-th group of input dimensions and $m$-th codebook.

To represent the full weights of $i$-th layer, simply concatenate:

$$
\widehat{W}_{i}=\widehat{W}_{1}\oplus...\oplus\widehat{W}_{d_{\mathrm{in}}/g}
$$

where $\oplus$ denotes concatenation.

Transformer-VQ [Lin23] applies vector quantization (VQ) to the key vector sequence of Attention, reducing the complexity of Attention to linear. Most other VQ works focus on optimizing the codebooks $C_{m}\in\mathbb{R}^{2^{B}}$, and the discrete codes represented by one-hot $d$. AQLM [Egi24] learns additive quantization of weight matrices in an input-adaptive fashion and jointly optimizes codebook parameters across each transformer block. QuIP# [Tse24] uses vector quantization to exploit the spherical sub-Gaussian distribution inherent in incoherent weights by introducing a hardware-efficient codebook based on the highly symmetrical E8 lattice. GPTVQ [Van24] interleaves the quantization of one or more columns with updates to the remaining unquantized weights, using information from the Hessian of the per-layer output reconstruction MSE, and further compresses the codebooks by using integer quantization and SVD-based compression. PV-Tuning [Mal24] notes that using straight-through estimators (STE) leads to suboptimal results and proposes an alternating iterative optimization strategy for scales, codebooks, zeros (continuous parameters), and assignments (discrete codes) during fine-tuning. QTIP [Tse24a] uses a stateful decoder that separates the codebook size from the bitrate and effective dimension to achieve ultra-high-dimensional quantization.

**Takeaways of subsection 5.2** For standard Post-Training Quantization (PTQ) of LLMs, equivalent transformation techniques such as shifting, scaling, and rotation can be employed to mitigate the impact of outliers. Quantization error can be further minimized through advanced compensation methods like GPTQ. For scenarios prioritizing high accuracy, mixed-precision quantization can be applied to recover performance loss. Conversely, if a high compression rate is the goal, combining low-rank approximation and sparsity-based methods can be effective. Furthermore, there are unique opportunities to explore emerging data formats, novel quantization functions, and cutting-edge model architectures, such as Multimodal Large Language Models (MLLMs) and Mixture of Expert (MOE) models.

### 5.3 Quantization Toolkit and Benchmark

#### 5.3.1 Toolkits

To quantize the LLMs, there are always three basic strategies, quantization aware-training (QAT), post-training quantization (PTQ), and parameter-efficient fine-tuning (PEFT).

<span id="table-05"></span>

![Quantization toolkits and benchmarks for large language models.](../../papers/low-bit-llms/table-05.png)

**Table 5.** Quantization toolkits and benchmarks for large language models.

The quantization toolkits that are devoted to providing comprehensive comparisons have good support for the prevailing models and quantization algorithms in various aspects. Most toolkits include well-known models like the Llama series, Mixtral, Vicuna, and so on. Those who pay more attention to the model diversity, such as QLLM-Eval, have further support for various models.
As well as the algorithms, LLMC, LMQuant, and MI-optimize focus on the performance of different quantization algorithms, and provide uniform, fair, comprehensive benchmarks for comparisons.
All the benchmarks are based on one or several inference frameworks as backends, and leave interfaces for users to define and evaluate any custom models and algorithms easily.

#### 5.3.2 Evaluation

The evaluation in the benchmarks showcases the most interesting aspects of quantization LLMs, i.e., efficiency and generation quality. We list the detailed tracks in [Table 5](#table-05).
**For efficiency**, the inference efficiency is measured by deployability and throughput, which are the most crucial features in LLMs compression [Lin24a, Gon24]. Typically, reducing the storage of the parameters can speed up the inference theoretically, but it depends on the actual system implementation. The benchmark provides us with a fair and convenient probe to distinguish the algorithms and implementations that have practical acceleration and storage saving.
The production efficiency is measured by calibration time, which indicates the time and computational resources cost of the PTQ algorithms [Gon24]. Methods that spare lots of resources usually have better generation quality, while those that require less time may have worse generation performance. It is a trade-off in producing quantized LLMs.
**For generation quality**, it has many aspects, such as perplexity, accuracy, logic, completion, trustworthiness and so on [Lin24a, Gon24, Li24f, Liu24e]. Most benchmarks evaluate emergent capability, which is the key feature of LLMs. Specifically, models and algorithms are tested under diverse scenarios, like dialogue, long-context, or multi-task [Li24f]. And some benchmarks are aware of the safety of generative contents, and estimate the trustworthiness and robustness of LLMs [Li24f, Liu24e].

**Takeaways of subsection 5.3** If your goal is to reproduce a variety of quantization algorithms, LLMC, MI-optimize, and LMQuant are recommended, as they provide a comprehensive suite of quantization methods.
If your focus is to deploy across inference frameworks, LLMC stands out as an ideal choice, providing flexible quantization settings and seamless compatibility with multiple backends.

## 6 Future Trends and Directions

As the field of large language model quantization continues to evolve, several emerging trends and research directions are poised to shape its future. This section explores the anticipated advancements in quantization techniques, model architectures, and hardware design that will drive improvements in the efficiency, performance, and application of quantized models.

**Quantization Techniques.** Despite progress, several challenges remain in quantization techniques. Firstly, one major issue is the unclear low-semantic-valueorigin of outliers in large language models (LLMs), which presents a significant barrier to further reducing quantization bit widths. Research aimed at uncovering the internal mechanisms behind these outliers is crucial and will provide valuable insights for the community, potentially advancing the state of quantization and enabling more efficient models. Secondly, pushing the boundaries of minimal bit representation with acceptable accuracy is highly valuable. Achieving the lowest possible bit width while maintaining performance can fully leverage hardware capabilities and maximize its potential. Thirdly, exploring unified strategies for mixed-bit quantization, including both bit selection and intra-layer/inter-layer bit allocation, is essential for optimizing model performance and efficiency. Current methods primarily emphasize intra-layer mixed precision, often overlooking the potential benefits of inter-layer mixed precision. Last but not least, developing semantic-guided strategies for achieving even lower-bit quantization and compression of key-value (KV) caches will be a major focus. During inference with long context lengths, the primary bottleneck often lies in the substantial memory usage of KV caches. Therefore, identifying effective methods for compressing KV caches is crucial for overcoming this limitation and enhancing model efficiency.

**Model Architecture.** Innovations in model architecture will also play a pivotal role. Firstly, quantizing models that handle multiple modalities will be explored to ensure efficiency across diverse data types and applications. Secondly, research will expand to include quantization strategies for new and emerging model structures such as the Mixture of Experts (MOE) and other large-scale architectures. Third, exploring the relationship between quantization and model size will provide insights into optimizing smaller models for performance while managing quantization trade-offs.

**Hardware Design.** Advancements in hardware and quantization co-design will be essential for unlocking new potential. The first area of focus is the development of systems for new types of extremely low-bit quantization. Innovative formats for low-bit representation and efficient system implementations may offer new solutions to the challenges posed by Moore’s Law. The second area involves accelerating training with lower-bit precision, such as FP4. Research into hardware that supports training with such low-bit precision will be essential for speeding up model training while preserving performance.

## 7 Conclusions

In this survey, we have presented an in-depth exploration of low-bit quantization techniques for large language models (LLMs), highlighting their significance in addressing the computational and memory challenges associated with deploying these models in constrained environments. We began by elucidating the fundamentals of low-bit quantization, including the novel data formats and granularities that cater specifically to LLMs. Our review of systems and frameworks has illustrated the diverse approaches and tools available for supporting low-bit LLMs across different hardware platforms. We have also categorized and discussed various techniques for optimizing training and inference, providing a comprehensive understanding of current methodologies. Lastly, we have explored future directions and emerging trends in the field, emphasizing potential research areas and technological advancements that could further enhance the efficiency and effectiveness of LLM quantization. As the landscape of LLM research continues to evolve, this survey aims to serve as a valuable resource for advancing the development of low-bit quantization techniques.
