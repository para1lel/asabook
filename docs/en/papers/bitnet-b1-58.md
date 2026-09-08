---
title: 'The Era of 1-bit LLMs'
createTime: 2026/09/08 13:02:00
permalink: /en/papers/bitnet-b1-58/
pageClass: paper-reading
---

> [Shuming Ma](https://shumingma.com/) [+author-note], [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Lingxiao Ma](https://xysmlx.github.io/), [Lei Wang](https://dblp.org/pid/181/2817-222), [Wenhui Wang](https://www.microsoft.com/en-us/research/people/wenwan/), [Shaohan Huang](https://buaahsh.github.io/), [Li Dong](https://dong.li/), [Ruiping Wang](https://www.jdl.link/user/rpwang/index.htm), [Jilong Xue](https://dblp.org/pid/06/10336), and [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]. First submitted to arXiv on February 27, 2024; current version v1; work in progress. [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764). <a href="/paper/bitnet-b1-58.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2402.17764). [TeX source](https://export.arxiv.org/e-print/2402.17764v1). The original PDF remains authoritative for the exact print layout and bibliography.

[+author-note]: Shuming Ma and Hongyu Wang contributed equally. Furu Wei is the corresponding author. Shuming Ma, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Jilong Xue, and Furu Wei are with Microsoft Research. Hongyu Wang and Ruiping Wang are with University of Chinese Academy of Sciences. [GeneralAI](https://aka.ms/GeneralAI).

## Abstract

Recent research, such as BitNet [Wan23], is paving the way for a new era of 1-bit Large Language Models (LLMs). In this work, we introduce a 1-bit LLM variant, namely **BitNet b1.58**, in which every single parameter (or weight) of the LLM is ternary $\{-1, 0, 1\}$. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption. More profoundly, the 1.58-bit LLM defines a new scaling law and recipe for training new generations of LLMs that are both high-performance and cost-effective. Furthermore, it enables a new computation paradigm and opens the door for designing specific hardware optimized for 1-bit LLMs.

<span id="figure-01"></span>

![Pareto comparison and computation paradigms for BitNet b1.58 and full-precision Transformer LLMs](../../papers/bitnet-b1-58/figure-01.png)

**Figure 1.** 1-bit LLMs (e.g., BitNet b1.58) provide a Pareto solution to reduce inference cost (latency, throughput, and energy) of LLMs while maintaining model performance. The new computation paradigm of BitNet b1.58 calls for actions to design new hardware optimized for 1-bit LLMs.

<span id="section-1"></span>

## 1 The Era of 1-bit LLMs

In recent years, the field of AI has seen a rapid growth in the size and capabilities of Large Language Models (LLMs). These models have demonstrated remarkable performance in a wide range of natural language processing tasks, but their increasing size has posed challenges for deployment and raised concerns about their environmental and economic impact due to high energy consumption.

One approach to address these challenges is to use post-training quantization to create low-bit models for inference [Xia23, Fra23, Che24b, Tse24]. This technique reduces the precision of weights and activations, significantly reducing the memory and computational requirements of LLMs. The trend has been to move from 16 bits to lower bits, such as 4-bit variants [Fra23, Lin24]. However, post-training quantization is sub-optimal, even though it is widely used in industry LLMs.

Recent work on 1-bit model architectures, such as BitNet [Wan23], presents a promising direction for reducing the cost of LLMs while maintaining their performance. Vanilla LLMs are in 16-bit floating values (i.e., FP16 or BF16), and the bulk of any LLMs is matrix multiplication. Therefore, the major computation cost comes from the floating-point addition and multiplication operations. In contrast, the matrix multiplication of BitNet only involves integer addition, which saves orders of energy cost for LLMs. As the fundamental limit to compute performance in many chips is power, the energy savings can also be translated into faster computation.

In addition to computation, the process of transferring model parameters from DRAM to the memory of an on-chip accelerator (e.g., SRAM) can be expensive during inference. There have been attempts to enlarge SRAM to improve throughput, but this introduces significantly higher costs than DRAM. Compared to full-precision models, 1-bit LLMs have a much lower memory footprint from both a capacity and bandwidth standpoint. This can significantly reduce the cost and time of loading weights from DRAM, leading to faster and more efficient inference.

In this work, we introduce a significant 1-bit LLM variant called **BitNet b1.58**, where every parameter is ternary, taking on values of $\{-1, 0, 1\}$. We have added an additional value of 0 to the original 1-bit BitNet, resulting in 1.58 bits in the binary system. BitNet b1.58 retains all the benefits of the original 1-bit BitNet, including its new computation paradigm, which requires almost no multiplication operations for matrix multiplication and can be highly optimized. Additionally, it has the same energy consumption as the original 1-bit BitNet and is much more efficient in terms of memory consumption, throughput and latency compared to FP16 LLM baselines. Furthermore, BitNet b1.58 offers two additional advantages. Firstly, its modeling capability is stronger due to its explicit support for feature filtering, made possible by the inclusion of 0 in the model weights, which can significantly improve the performance of 1-bit LLMs. Secondly, our experiments show that BitNet b1.58 can match full precision (i.e., FP16) baselines in terms of both perplexity and end-task performance, starting from a 3B size, when using the same configuration (e.g., model size, training tokens, etc.).

<span id="section-2"></span>

## 2 BitNet b1.58

BitNet b1.58 is based on the BitNet architecture, which is a Transformer that replaces *nn.Linear* with *BitLinear*. It is trained from scratch, with 1.58-bit weights and 8-bit activations. Compared to the original BitNet, it introduces some modifications that we summarize below.

**Quantization Function.** To constrain the weights to -1, 0, or +1, we adopt an *absmean* quantization function. It first scales the weight matrix by its average absolute value, and then round each value to the nearest integer among $\{-1, 0, +1\}$:

<span id="equation-01"></span>

$$
\widetilde{W}=\mathrm{RoundClip}\left(\frac{W}{\gamma+\epsilon},-1,1\right),
$$

<span id="equation-02"></span>

$$
\mathrm{RoundClip}(x,a,b)=\max(a,\min(b,\mathrm{round}(x))),
$$

<span id="equation-03"></span>

$$
\gamma=\frac{1}{nm}\sum_{ij}|W_{ij}|.
$$

The quantization function for activations follows the same implementation in BitNet, except that we do not scale the activations before the non-linear functions to the range $[0,Q_b]$. Instead, the activations are all scaled to $[-Q_b,Q_b]$ per token to get rid of the zero-point quantization. This is more convenient and simple for both implementation and system-level optimization, while introduces negligible effects to the performance in our experiments.

**LLaMA-alike Components.** The architecture of LLaMA [Tou23, Tou23a] has been the de-facto backbone for open-source LLMs. To embrace the open-source community, our design of BitNet b1.58 adopts the LLaMA-alike components. Specifically, it uses RMSNorm [Zha19], SwiGLU [Sha20], rotary embedding [Su24], and removes all biases. In this way, BitNet b1.58 can be integrated into the popular open-source software (e.g., Huggingface, vLLM [Kwo23], and llama.cpp [+llama-cpp]) with minimal efforts.

[+llama-cpp]: [llama.cpp](https://github.com/ggerganov/llama.cpp).

<span id="table-01"></span>

![Perplexity, memory, and latency results for BitNet b1.58 and LLaMA LLM](../../papers/bitnet-b1-58/table-01.png)

**Table 1.** Perplexity as well as the cost of BitNet b1.58 and LLaMA LLM.

<span id="table-02"></span>

![Zero-shot accuracy of BitNet b1.58 and LLaMA LLM across seven end tasks](../../papers/bitnet-b1-58/table-02.png)

**Table 2.** Zero-shot accuracy of BitNet b1.58 and LLaMA LLM on the end tasks.

<span id="section-3"></span>

## 3 Results

We compared BitNet b1.58 to our reproduced FP16 LLaMA LLM in various sizes. To ensure a fair comparison, we pre-trained the models on the RedPajama dataset [Tog23a] for 100 billion tokens. We evaluated the zero-shot performance on a range of language tasks, including ARC-Easy [Yad19], ARC-Challenge [Yad19], Hellaswag [Zel19], Winogrande [Sak19], PIQA [Bis20], OpenbookQA [Mih18b], and BoolQ [Cla19]. We also reported the validation perplexity on the WikiText2 [Mer16] and C4 [Raf19] datasets.

We compared the runtime GPU memory and latency of both LLaMA LLM and BitNet b1.58. The results were measured using the FasterTransformer [+fastertransformer] codebase, which is well-optimized for LLM inference latency on GPU devices. The 2-bit kernel from Ladder [Wan24e] is also integrated for BitNet b1.58. We reported the time per output token, as it is the major cost for inference.

[+fastertransformer]: [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer).

[Table 1](#table-01) summarizes the perplexity and the cost for BitNet b1.58 and LLaMA LLM. It shows that BitNet b1.58 starts to match full precision LLaMA LLM at 3B model size in terms of perplexity, while being 2.71 times faster and using 3.55 times less GPU memory. In particular, BitNet b1.58 with a 3.9B model size is 2.4 times faster, consumes 3.32 times less memory, but performs significantly better than LLaMA LLM 3B.

[Table 2](#table-02) reports the detailed results of the zero-shot accuracy on the end tasks. We followed the pipeline from *lm-evaluation-harness* [+lm-evaluation-harness] to perform the evaluation. The results show that the performance gap between BitNet b1.58 and LLaMA LLM narrows as the model size increases. More importantly, BitNet b1.58 can match the performance of the full precision baseline starting from a 3B size. Similar to the observation of the perplexity, the end-task results reveal that BitNet b1.58 3.9B outperforms LLaMA LLM 3B with lower memory and latency cost. This demonstrates that BitNet b1.58 is a Pareto improvement over the state-of-the-art LLM models.

[+lm-evaluation-harness]: [EleutherAI lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

**Memory and Latency** We further scaled up the model size to 7B, 13B, and 70B and evaluated the cost. [Figure 2](#figure-02) illustrates the trends of latency and memory, showing that the speed-up increases as the model size scales. In particular, BitNet b1.58 70B is 4.1 times faster than the LLaMA LLM baseline. This is because the time cost for *nn.Linear* grows with the model size. The memory consumption follows a similar trend, as the embedding remains full precision and its memory proportion is smaller for larger models. Both latency and memory were measured with a 2-bit kernel, so there is still room for optimization to further reduce the cost.

<span id="figure-02"></span>

![Decoding latency and memory consumption of BitNet b1.58 across model sizes](../../papers/bitnet-b1-58/figure-02.png)

**Figure 2.** Decoding latency (Left) and memory consumption (Right) of BitNet b1.58 varying the model size.

**Energy** We also estimate the arithmetic operations energy consumption of both BitNet b1.58 and LLaMA LLM. We focus mainly on the calculation for matrix multiplication, since it contributes the most to the cost of LLMs. [Figure 3](#figure-03) illustrates the composition of the energy cost. The majority of BitNet b1.58 is INT8 addition calculation, while LLaMA LLM consists of both FP16 addition and FP16 multiplication. According to the energy model in [Hor14, Zha22g], BitNet b1.58 saves 71.4 times arithmetic operations energy consumption for matrix multiplication on 7nm chips. We further reported the end-to-end energy cost for models with 512 tokens. Our results show that as the model size scales, BitNet b1.58 becomes increasingly more efficient in terms of energy consumption compared to the FP16 LLaMA LLM baseline. This is due to the fact that the percentage of *nn.Linear* grows with the model size, while the cost from other components is smaller for larger models.

<span id="figure-03"></span>

![Arithmetic-operation and end-to-end energy consumption of BitNet b1.58 and LLaMA LLM](../../papers/bitnet-b1-58/figure-03.png)

**Figure 3.** Energy consumption of BitNet b1.58 compared to LLaMA LLM at 7nm process nodes. On the left is the components of arithmetic operations energy. On the right is the end-to-end energy cost across different model sizes.

**Throughput** We compare the throughput of BitNet b1.58 and LLaMA LLM with 70B parameters on two 80GB A100 cards, using pipeline parallelism [Hua19c] so that LLaMA LLM 70B could be run on the devices. We increased the batch size until the GPU memory limit was reached, with a sequence length of 512. [Table 3](#table-03) shows that BitNet b1.58 70B can support up to 11 times the batch size of LLaMA LLM, resulting an 8.9 times higher throughput.

<span id="table-03"></span>

![Maximum batch size and throughput of BitNet b1.58 and LLaMA LLM at 70B parameters](../../papers/bitnet-b1-58/table-03.png)

**Table 3.** Comparison of the throughput between BitNet b1.58 70B and LLaMA LLM 70B.

**BitNet b1.58 is enabling a new scaling law with respect to model performance and inference cost**. As a reference, we can have the following equivalence between different model sizes in 1.58-bit and 16-bit based on the results in [Figure 2](#figure-02) and [Figure 3](#figure-03).

- 13B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 3B FP16 LLM.
- 30B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 7B FP16 LLM.
- 70B BitNet b1.58 is more efficient, in terms of latency, memory usage and energy consumption, than 13B FP16 LLM.

**Training with 2T Tokens** The number of training tokens is a crucial factor for LLMs. To test the scalability of BitNet b1.58 in terms of tokens, we trained a BitNet b1.58 model with 2T tokens following the data recipe of StableLM-3B [Tow23], which is the state-of-the-art open-source 3B model. Both models were evaluated on a benchmark that consists of Winogrande [Sak19], PIQA [Bis20], SciQ [Wel17], LAMBADA [Pap16], and ARC-easy [Yad19]. We reported the zero-shot accuracy in [Table 4](#table-04). For tasks measured with accuracy and normalized accuracy, we take the average of the two. The results of StableLM 3b at 2T tokens are taken directly from its technical report. Our findings shows that BitNet b1.58 achieves a superior performance on all end tasks, indicating that 1.58-bit LLMs also have strong generalization capabilities.

<span id="table-04"></span>

![Zero-shot results for BitNet b1.58 and StableLM-3B after training with 2T tokens](../../papers/bitnet-b1-58/table-04.png)

**Table 4.** Comparison of BitNet b1.58 with StableLM-3B with 2T tokens.

<span id="section-4"></span>

## 4 Discussion and Future Work

**1-bit Mixture-of-Experts (MoE) LLMs** Mixture-of-Experts (MoE) have proven to be a cost-effective approach for LLMs. While it significantly reduces the computation FLOPs, the high memory consumption and inter-chip communication overhead limit its deployment and application. These challenges can be addressed by 1.58-bit LLMs. Firstly, the reduced memory footprint reduces the number of devices required to deploy MoE models. Moreover, it significantly reduces the overhead of transferring activations across networks. Ultimately, there would be no overhead if the entire models could be placed on a single chip.

**Native Support of Long Sequence in LLMs** In the era of LLMs, the ability to handle long sequence has become a critical demand. One major challenge for long sequence inference is the memory consumption introduced by the KV caches. BitNet b1.58 represents a significant step towards native support for long sequences, as it reduces the activations from 16 bits to 8 bits, allowing the context length to be doubled given the same resources. This can be further losslessly compressed to 4 bits or even lower for 1.58-bit LLMs, which we leave as future work.

**LLMs on Edge and Mobile** The use of 1.58-bit LLMs has the potential to greatly improve the performance of language models on edge and mobile devices. These devices are often limited by their memory and computational power, which can restrict the performance and the scale of LLMs. However, the reduced memory and energy consumption of 1.58-bit LLMs allows them to be deployed on these devices, enabling a wide range of applications that were previously not possible. This can greatly enhance the capabilities of edge and mobile devices and enable new and exciting applications of LLMs. Moreover, 1.58-bit LLMs are more friendly to CPU devices, which are the main processors used in edge and mobile devices. This means that BitNet b1.58 can be efficiently executed on these devices, further improving their performance and capabilities.

**New Hardware for 1-bit LLMs** Recent work like Groq [+groq] has demonstrated promising results and great potential for building specific hardware (e.g., LPUs) for LLMs. Going one step further, we envision and call for actions to design new hardware and system specifically optimized for 1-bit LLMs, given the new computation paradigm enabled in BitNet [Wan23].

[+groq]: [Groq](https://groq.com/).
