---
title: 'BitNet b1.58 2B4T Technical Report'
createTime: 2026/09/08 15:00:00
permalink: /en/papers/bitnet-b1-58-2b4t/
pageClass: paper-reading
---

> [Shuming Ma](https://shumingma.com/) [+author-note], [Hongyu Wang](https://ustcwhy.github.io/) [+author-note], [Shaohan Huang](https://buaahsh.github.io/), [Xingxing Zhang](https://xingxingzhang.github.io/), [Ying Hu](https://dblp.org/pid/92/4882.html), [Ting Song](https://aclanthology.org/people/ting-song/), [Yan Xia](https://www.microsoft.com/en-us/research/people/yanxia/), and [Furu Wei](https://www.microsoft.com/en-us/research/people/fuwei/) [+author-note]. First submitted to arXiv on April 16, 2025; current version v2; work in progress. [BitNet b1.58 2B4T Technical Report](https://arxiv.org/abs/2504.12285). <a href="/paper/bitnet-b1-58-2b4t.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2504.12285). [TeX source](https://export.arxiv.org/e-print/2504.12285v2). The original PDF remains authoritative for the exact print layout and bibliography.

[+author-note]: Shuming Ma and Hongyu Wang contributed equally. Furu Wei is the corresponding author. Shuming Ma, Shaohan Huang, Xingxing Zhang, Ting Song, Yan Xia, and Furu Wei are with Microsoft Research. Hongyu Wang is with University of Chinese Academy of Sciences. Ying Hu is with Tsinghua University. [GeneralAI](https://aka.ms/GeneralAI).

## Abstract

We introduce BitNet b1.58 2B4T, the first open-source, native 1-bit Large Language Model (LLM) at the 2-billion parameter scale. Trained on a corpus of 4 trillion tokens, the model has been rigorously evaluated across benchmarks covering language understanding, mathematical reasoning, coding proficiency, and conversational ability. Our results demonstrate that BitNet b1.58 2B4T achieves performance on par with leading open-weight, full-precision LLMs of similar size, while offering significant advantages in computational efficiency, including substantially reduced memory footprint, energy consumption, and decoding latency. To facilitate further research and adoption, the model weights are released via Hugging Face along with open-source inference implementations for both GPU and CPU architectures.

- **BitNet b1.58 2B4T (1.58-bit):** [`bitnet-b1.58-2B-4T`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)<br>*The packed weight of BitNet b1.58 2B4T, used for inference only*
- **BitNet b1.58 2B4T (bf16):** [`bitnet-b1.58-2B-4T-bf16`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-bf16)<br>*The master weight of BitNet b1.58 2B4T, used for training only*
- **BitNet b1.58 2B4T (gguf):** [`bitnet-b1.58-2B-4T-gguf`](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T-gguf)<br>*The GGUF format of BitNet b1.58 2B4T, used for bitnet.cpp*
- **BitNet b1.58 2B4T Code:** [`bitnet.cpp`](https://github.com/microsoft/BitNet); **Demo:** [`aka.ms/bitnet-demo`](https://aka.ms/bitnet-demo)

<span id="figure-01"></span>

![Figure 1. BitNet b1.58 2B4T advances the Pareto frontier defined by leading open-weight LLMs under 3B parameters in terms of performance versus memory, demonstrating superior efficiency.](../../papers/bitnet-b1-58-2b4t/figure-01.png)

**Figure 1.** BitNet b1.58 2B4T advances the Pareto frontier defined by leading open-weight LLMs under 3B parameters in terms of performance versus memory, demonstrating superior efficiency.

<span id="section-1"></span>

## 1 Introduction

Open-source large language models (LLMs) have become pivotal in democratizing access to advanced AI capabilities, fostering innovation, and enabling research across diverse fields such as natural language processing, code generation, and vision computing [Dub24, Qwe25, Bai25a]. Their public availability allows for widespread experimentation and adaptation. However, a significant barrier hinders their broader adoption: the substantial computational resources required for deployment and inference. State-of-the-art open LLMs typically require large memory footprints, consume considerable energy, and exhibit notable inference latency, rendering them impractical for many edge devices, resource-constrained environments, and real-time applications.

1-bit LLMs, representing an extreme yet promising form of model quantization where weights and potentially activations are constrained to binary $\{-1,+1\}$ or ternary $\{-1,0,+1\}$, offer a compelling solution to the efficiency challenges. By drastically reducing the memory required to store weights and enabling highly efficient bitwise computations, they have the potential to significantly lower deployment costs, reduce energy consumption, and accelerate inference speeds. While prior work has explored 1-bit models, existing open efforts often fall into two categories: 1) post-training quantization (PTQ) methods applied to pre-trained full-precision models, which can lead to significant performance degradation [Xu24h, Tea24b], or 2) native 1-bit models (trained from scratch with 1-bit weights) that have been developed at relatively smaller scales (e.g., OLMo-Bitnet-1B [+1]) and may not yet match the capabilities of larger, full-precision counterparts. This performance gap has limited the practical impact of 1-bit LLMs thus far.

To bridge this gap between efficiency and performance, we introduce BitNet b1.58 2B4T, the first open-source, native 1-bit LLM trained at scale. This model, comprising 2 billion parameters, was trained from scratch on a substantial dataset of 4 trillion tokens, leveraging architectural and training innovations specific to the 1-bit paradigm. **The core contribution of this work is to demonstrate that a native 1-bit LLM, when trained effectively at scale, can achieve performance comparable to leading open-weight, full-precision models of similar size across a wide range of tasks.**

This technical report details the development and evaluation of BitNet b1.58 2B4T. We describe the architecture and training methodology, and then present comprehensive evaluation results on standard benchmarks assessing language understanding, mathematical reasoning, coding proficiency, and multi-turn conversational abilities. Our findings confirm its strong performance relative to established full-precision baselines, coupled with significant advantages in efficiency. Finally, we announce the public release of the BitNet b1.58 2B4T model weights via Hugging Face and provide open-source inference code optimized for both GPU and CPU execution, aiming to facilitate further research and the practical deployment of highly efficient LLMs.

<span id="section-2"></span>

## 2 Architecture

The architecture of BitNet b1.58 2B4T is derived from the standard Transformer model [Vas17], incorporating significant modifications based on the BitNet framework [Wan23, Ma24]. The model is trained entirely from scratch.

The core architectural innovation lies in replacing the standard full-precision linear layers (*torch.nn.Linear*) with custom *BitLinear* layers. This constitutes the foundation of the BitNet approach. Within these *BitLinear* layers:

- **Weight Quantization:** Model weights are quantized to 1.58 bits during the forward pass. This is achieved using an absolute mean (absmean) quantization scheme, which maps weights to ternary values $\{-1,0,+1\}$. This drastically reduces the model size and enables efficient mathematical operations.
- **Activation Quantization:** Activations flowing through the linear projection are quantized to 8-bit integers. This employs an absolute maximum (absmax) quantization strategy, applied per-token.
- **Normalization:** We incorporate `subln` normalization [Wan22l] to further enhance training stability, which can be particularly beneficial in quantized training regimes.

Beyond the *BitLinear* layers, several established LLM techniques are integrated to enhance performance and stability:

- **Activation Function (FFN):** Within the feed-forward network (FFN) sub-layers, instead of the commonly used SwiGLU activation [Sha20], BitNet b1.58 2B4T employs squared ReLU ($\mathrm{ReLU}^{2}$). This choice is motivated by its potential to improve model sparsity and computational characteristics within the 1-bit context [Wan24af, Wan24ag].
- **Positional Embeddings:** Rotary Position Embeddings (RoPE) [Su24] are used to inject positional information, a standard practice in modern high-performance LLMs.
- **Bias Removal:** Consistent with architectures like LLaMA, all bias terms are removed from the linear layers and normalization layers throughout the network, reducing parameter count and potentially simplifying quantization.

For tokenization, we adopt the tokenizer developed for LLaMA 3 [Dub24]. This tokenizer implements a byte-level Byte-Pair Encoding (BPE) scheme with a vocabulary size of 128,256 tokens. This choice ensures robust handling of diverse text and code, and its widespread adoption facilitates straightforward integration with existing open-source tooling and ecosystems.

<span id="section-3"></span>

## 3 Training

The training process for BitNet b1.58 2B4T involved three distinct phases: large-scale pre-training followed by supervised fine-tuning (SFT) and direct preference optimization (DPO). While advanced techniques like Proximal Policy Optimization (PPO) or Group Relative Policy Optimization (GRPO) can further enhance capabilities such as mathematics and chain-of-thought reasoning [Sch17a, Sha24d], the current version of BitNet b1.58 2B4T relies solely on pre-training, SFT, and DPO. The exploration of reinforcement learning methods remains a direction for future work.

<span id="section-3-1"></span>

### 3.1 Pre-training

The pre-training phase aimed to imbue the model with broad world knowledge and foundational language capabilities. We adapted general training strategies from established LLM practices [Dub24], with specific adjustments tailored for the 1-bit architecture.

<span id="section-3-1-1"></span>

#### 3.1.1 Learning Rate Schedule

A two-stage learning rate schedule was employed.

- **Stage 1 (High Learning Rate):** The initial phase utilized a standard cosine decay schedule but commenced with a relatively high peak learning rate. This decision was informed by the observation that 1-bit models often exhibit greater training stability compared to their full-precision counterparts, allowing for more aggressive initial learning steps.
- **Stage 2 (Cooldown):** Approximately midway through the planned training token count, the learning rate was abruptly decayed and subsequently maintained via a cosine schedule with a significantly lower peak value. This "cooldown" phase allows the model to refine its representations on higher-quality data (see [Section 3.1.3](#section-3-1-3)).

<span id="section-3-1-2"></span>

#### 3.1.2 Weight Decay Schedule

Complementing the learning rate adjustments, a two-stage weight decay strategy was implemented.

- **Stage 1:** During the first training stage, weight decay followed a cosine schedule, reaching a peak value of $0.1$. This regularization helps prevent overfitting during the initial high-learning-rate phase.
- **Stage 2:** In the second stage, weight decay was effectively disabled (set to zero). This allows the model parameters to settle into finer-grained optima guided by the lower learning rate and curated data.

<span id="section-3-1-3"></span>

#### 3.1.3 Pre-training Data

The pre-training corpus comprised a mixture of publicly available text and code datasets, including large web crawls like DCLM [Li25a] and educational web pages like FineWeb-EDU [Pen24]. To enhance mathematical reasoning abilities, we also incorporated synthetically generated mathematical data. The data presentation strategy aligned with the two-stage training: the bulk of general web data was processed during Stage 1, while higher-quality curated datasets were emphasized during the Stage 2 cooldown phase, coinciding with the reduced learning rate.

<span id="section-3-2"></span>

### 3.2 Supervised Fine-tuning (SFT)

Following pre-training, the model underwent supervised fine-tuning (SFT) to enhance its instruction-following capabilities and improve its performance in conversational interaction formats.

<span id="section-3-2-1"></span>

#### 3.2.1 SFT Data

The SFT phase utilized a diverse collection of publicly available instruction-following and conversational datasets. These included, but were not limited to, WildChat [Zha24ac], LMSYS-Chat-1M [Zhe23c], WizardLM Evol-Instruct [Xu24i], and SlimOrca [Lia23]. To further bolster specific capabilities, particularly in reasoning and complex instruction adherence, we supplemented these with synthetic datasets generated using methodologies like GLAN [Li24u] and MathScale [Tan24c].

<span id="section-3-2-2"></span>

#### 3.2.2 Chat Template

For conversational tasks during SFT and inference, the following chat template structure was employed:

```text
<|begin_of_text|>System: {system_message}<|eot_id|>
User: {user_message_1}<|eot_id|>
Assistant: {assistant_message_1}<|eot_id|>
User: {user_message_2}<|eot_id|>
Assistant: {assistant_message_2}<|eot_id|>...
```

<span id="section-3-2-3"></span>

#### 3.2.3 Optimization Details

Several optimization choices were key during SFT:

- **Loss Aggregation:** Instead of averaging the cross-entropy loss across tokens within a batch (mean reduction), we employed summation. Empirically, we observed that summing the losses led to improved convergence and better final performance for this model.
- **Hyperparameter Tuning:** Careful tuning of the learning rate and the number of training epochs was performed. Consistent with our pre-training findings, the 1-bit model benefited from a relatively larger learning rate during SFT compared to typical full-precision model fine-tuning. Furthermore, achieving optimal convergence required extending the fine-tuning duration over a larger number of epochs than full-precision models of similar size.

<span id="section-3-3"></span>

### 3.3 Direct Preference Optimization (DPO)

To further align the model’s behavior with human preferences regarding helpfulness and safety, we applied Direct Preference Optimization (DPO) [Raf23] following the SFT phase. DPO offers an efficient alternative to traditional RLHF by directly optimizing the language model using preference data, thereby circumventing the need to train a separate reward model. This DPO stage served to refine the model’s conversational prowess and overall alignment with desired interaction patterns in practical use cases.

<span id="section-3-3-1"></span>

#### 3.3.1 Training Data

The preference dataset used for DPO training was constructed from a combination of publicly available resources recognized for capturing diverse human judgments on model outputs. Specifically, we utilized UltraFeedback [Cui24a] and MagPie [Xu24j]. The aggregation of these datasets provided a robust and multifaceted preference signal, guiding the model towards generating responses more aligned with human expectations.

<span id="section-3-3-2"></span>

#### 3.3.2 Training Details

The DPO training phase was conducted for 2 epochs. We employed a learning rate of $2\times 10^{-7}$ and set the DPO beta parameter, which controls the divergence from the reference policy, to 0.1. To enhance training efficiency during this phase, we integrated optimized kernels from the *Liger Kernel* library [Hsu24a]. Qualitatively, our observations indicate that the DPO process effectively steered the model towards preferred response styles without inducing significant degradation in the core capabilities established during pre-training and SFT.

<span id="table-01"></span>

![Table 1. Comparison of BitNet b1.58 2B4T with leading open-weight full-precision LLMs of similar size (1B-2B parameters) on efficiency metrics and performance across a wide range of benchmarks. All models compared are instruction-tuned versions.](../../papers/bitnet-b1-58-2b4t/table-01.png)

**Table 1.** Comparison of BitNet b1.58 2B4T with leading open-weight full-precision LLMs of similar size (1B-2B parameters) on efficiency metrics and performance across a wide range of benchmarks. All models compared are instruction-tuned versions.

<span id="section-4"></span>

## 4 Evaluation

<span id="table-02"></span>

![Table 2. Comparison of BitNet b1.58 (2B) against Qwen2.5 1.5B in its original bf16 precision and after INT4 post-training quantization (GPTQ and AWQ). All models shown are based on instruction-tuned checkpoints.](../../papers/bitnet-b1-58-2b4t/table-02.png)

**Table 2.** Comparison of BitNet b1.58 (2B) against Qwen2.5 1.5B in its original bf16 precision and after INT4 post-training quantization (GPTQ and AWQ). All models shown are based on instruction-tuned checkpoints.

<span id="table-03"></span>

![Table 3. Performance comparison of BitNet b1.58 2B4T against other open-weight 1-bit models. This includes natively trained 1-bit models (Bonsai-0.5B, OLMo-Bitnet-1B) and larger models post-training quantized to 1.58-bit (Falcon3-1.58bit-7B, Llama3-8B-1.58).](../../papers/bitnet-b1-58-2b4t/table-03.png)

**Table 3.** Performance comparison of BitNet b1.58 2B4T against other open-weight 1-bit models. This includes natively trained 1-bit models (Bonsai-0.5B, OLMo-Bitnet-1B) and larger models post-training quantized to 1.58-bit (Falcon3-1.58bit-7B, Llama3-8B-1.58).

We measure performance on a wide variety of benchmarks classified as follows:

- **Language understanding and reasoning:** ARC-Easy [Yad19], ARC-Challenge [Yad19], HellaSwag [Zel19], WinoGrande [Sak19], PIQA [Bis20], OpenbookQA [Mih18b], and CommonsenseQA [Tal19]
- **World knowledge:** TruthfulQA [Lin22] and MMLU [Hen20]
- **Reading comprehension:** TriviaQA [Jos17] and BoolQ [Cla19]
- **Math and code:** GSM8K [Cob21], MATH-500 [Hen21] and HumanEval+ [Liu24i]
- **Instruction following and conversation:** IFEval [Zho23a] and MT-bench [Sto23e]

We compare BitNet b1.58 2B4T with leading open-weight full precision LLMs of similar size, including LLaMA 3.2 1B [Dub24], Gemma-3 1B [Gem25a], Qwen2.5 1.5B [Qwe25], SmolLM2 1.7B [All25] and MiniCPM 2B [Hu24]. All models are instruction-tuned versions. We re-run all benchmarks with a public evaluation pipeline for a fair comparison. More evaluation details are available at [Section 9](#section-9). The main results are presented in [Table 1](#table-01).

<span id="section-4-1"></span>

### 4.1 Main Results

As shown in [Table 1](#table-01), BitNet b1.58 2B4T demonstrates remarkable resource efficiency. Its non-embedding memory footprint and estimated energy consumption [Hor14, Zha22g] during decoding are substantially lower compared to all the full-precision models evaluated, highlighting a significant advantage in operational cost and deployability on resource-constrained devices.

In terms of task performance, BitNet b1.58 2B4T proves highly competitive. It achieves the best results among the compared models on several benchmarks spanning reasoning, knowledge, and math capabilities. On other benchmarks, its performance is closely comparable to the top-performing full-precision models. While some full-precision models show slight advantages on specific tasks or the overall average, BitNet b1.58 2B4T delivers strong performance across the board. The results indicate that BitNet b1.58 2B4T achieves capabilities nearly on par with leading models in its size class while offering dramatically improved efficiency.

<span id="section-4-2"></span>

### 4.2 Comparison with Post-training Quantized Models

We further investigate the efficiency-performance trade-off by comparing BitNet b1.58 2B4T against post-training quantized (PTQ) versions of a leading competitor, Qwen2.5 1.5B, using standard INT4 methods (GPTQ and AWQ). The results are summarized in [Table 2](#table-02).

While INT4 quantization successfully reduces the memory footprint of the full-precision model, BitNet b1.58 2B4T achieves an even lower memory requirement due to its native 1-bit architecture. More importantly, this superior memory efficiency does not compromise performance relative to the quantized models. Standard PTQ techniques lead to a noticeable degradation in performance compared to the original full-precision model. In contrast, BitNet b1.58 2B4T maintains stronger overall performance than the INT4 quantized versions of Qwen2.5-1.5B on the evaluated benchmarks. This comparison suggests that BitNet b1.58 2B4T represents a more favorable point on the efficiency-performance curve than applying conventional INT4 PTQ to existing architectures, offering better performance with lower resource usage.

<span id="section-4-3"></span>

### 4.3 Comparison with Open-weight 1-bit Models

Finally, we situate BitNet b1.58 2B4T within the landscape of other models designed for or quantized to near 1-bit precision. We compare it against natively trained 1-bit models of smaller scale and significantly larger models post-training quantized to 1.58-bit precision. The comparative results are presented in [Table 3](#table-03).

The evaluation clearly positions BitNet b1.58 2B4T as the leading model in this category. It demonstrates significantly stronger overall performance than all other compared 1-bit models, achieving the highest scores on the vast majority of benchmarks. Notably, BitNet b1.58 2B4T substantially outperforms not only the smaller, natively trained 1-bit models but also the much larger models (in terms of parameter count) that were quantized to 1-bit. This highlights the effectiveness of the native training approach employed by BitNet b1.58 2B4T, allowing it to set a new state-of-the-art performance level for models operating at this extreme level of quantization, even surpassing larger models subjected to post-training quantization.

<span id="section-5"></span>

## 5 Inference Implementation

Efficient inference is crucial for deploying Large Language Models, particularly for resource-constrained environments. The unique quantization scheme of BitNet b1.58 2B4T, employing 1.58-bit weights and 8-bit activations (W1.58A8), necessitates specialized implementations, as standard deep learning libraries often lack optimized kernels for such mixed-precision, low-bit formats. To address this, we developed and open-sourced dedicated inference libraries for both GPU and CPU platforms. The code is publicly available at [https://aka.ms/bitnet](https://aka.ms/bitnet).

<span id="section-5-1"></span>

### 5.1 GPU Inference

Current GPU architectures and their associated software libraries (e.g., cuBLAS, PyTorch kernels) are primarily optimized for operations involving standard data types like FP16, BF16, and INT8/INT4. Native, high-performance support for the specific W1.58A8 matrix multiplication required by BitNet b1.58 2B4T is generally unavailable. This limitation can hinder the realization of the theoretical efficiency gains offered by 1-bit models on existing hardware.

To enable efficient GPU inference, we developed a custom CUDA kernel specifically designed for the W1.58A8 matrix multiplication. Since ternary weights ($\{-1,0,+1\}$, representing 1.58 bits) cannot be stored efficiently using standard data types, we pack multiple weight values into a single 8-bit integer (‘int8’) for storage in High Bandwidth Memory (HBM). Specifically, four ternary values are encoded into one ‘int8’ value. During computation, the CUDA kernel loads the packed ‘int8’ weights from HBM into the GPU’s faster on-chip Shared Memory (SRAM). It then unpacks these values back into a representation suitable for efficient ternary computation (e.g., reconstructing the -1, 0, +1 values) immediately before performing the matrix multiplication with the 8-bit activations. This ‘pack-store-load-unpack-compute’ strategy minimizes memory bandwidth usage while leveraging custom compute instructions. Further implementation details and optimization strategies are elaborated in the Ladder framework [Wan24e].

While our custom kernel significantly improves performance compared to naive implementations, we note that current commodity GPU architectures are not optimally designed for the 1-bit models. We believe that future hardware innovations, potentially incorporating dedicated logic for low-bit operations, will be essential to fully unlock the performance and energy efficiency potential of models like BitNet b1.58.

<span id="section-5-2"></span>

### 5.2 CPU Inference

To ensure broad accessibility and enable deployment on devices lacking powerful GPUs (e.g., edge devices, laptops, standard servers), we developed *bitnet.cpp*. This C++ library serves as an official reference implementation for CPU inference of 1-bit LLMs, including BitNet b1.58.

*bitnet.cpp* provides optimized kernels tailored for efficient execution on standard CPU architectures. The kernels are designed to operate efficiently with the model’s specific quantization scheme, avoiding the overhead of generic quantization libraries or intricate low-level bit manipulation where possible. It processes the weight elements in a manner consistent with the BitNet b1.58 training methodology, ensuring numerical accuracy (lossless inference relative to the training procedure).

This approach delivers fast and accurate inference of 1.58-bit models directly on CPUs. More technical details and usage instructions can be found in the *bitnet.cpp* repository and associated technical report [Wan25ao].

<span id="section-6"></span>

## 6 Conclusion

This technical report introduced BitNet b1.58 2B4T, a significant step towards highly efficient yet capable Large Language Models. As the first open-source, native 1-bit LLM trained at the 2-billion parameter scale on 4 trillion tokens, our work demonstrates the viability of extreme quantization directly within the training process.

Comprehensive evaluations across benchmarks assessing language understanding, reasoning, mathematics, coding, and dialogue revealed that BitNet b1.58 2B4T achieves performance comparable to state-of-the-art open-weight, full-precision models of similar size. Crucially, this performance parity is achieved with dramatically reduced computational requirements, offering substantial savings in memory footprint, energy consumption, and inference latency. To facilitate practical use and further research, we developed and released optimized inference implementations for both GPU (via custom CUDA kernels) and CPU (via the ‘bitnet.cpp’ library), alongside the model weights available on Hugging Face.

BitNet b1.58 2B4T represents a compelling proof-of-concept that challenges the necessity of full-precision weights for achieving high performance in LLMs at scale. It opens avenues for deploying powerful language models in resource-constrained environments where previous models were prohibitive, potentially democratizing access to advanced AI capabilities.

<span id="section-7"></span>

## 7 Future Directions

While BitNet b1.58 2B4T demonstrates promising results, several exciting research directions remain:

- **Scaling Laws and Larger Models:** Investigating the scaling properties of native 1-bit LLMs is crucial. Future work will explore training larger models (e.g., 7B, 13B parameters and beyond) and training on even larger datasets to understand if the performance parity with full-precision models holds.
- **Hardware Co-Design and Optimization:** The full potential of 1-bit models is likely hindered by current hardware limitations. Continued development of highly optimized kernels for existing hardware (GPUs, CPUs, NPUs) is needed. Furthermore, co-designing future hardware accelerators specifically optimized for 1-bit computations and data movement could unlock orders-of-magnitude improvements in speed and energy efficiency.
- **Extended Sequence Length:** Extending the maximum sequence length of BitNet b1.58 2B4T can process is crucial. This enhancement is vital for tasks demanding long-context understanding, such as summarizing lengthy documents or engaging in complex problem-solving, and is particularly critical for improving performance on **long chain-of-thought reasoning** tasks. Investigating efficient attention mechanisms suitable for low-bit models at longer sequence lengths will be key.
- **Multilingual Capabilities:** The current model is primarily trained on English-centric data. Extending the pre-training corpus and potentially adapting the architecture to effectively support **multiple languages** is a key direction for broader applicability.
- **Multimodal Integration:** Exploring the integration of 1-bit principles into **multimodal architectures** is another promising frontier. Developing efficient ways to process and fuse information from different modalities (e.g., text and images) within a low-bit framework could enable new applications.
- **Theoretical Understanding:** Delving deeper into the theoretical underpinnings of why 1-bit training at scale is effective remains an open area. Analyzing the learning dynamics, loss landscapes, and representational properties of these models could yield valuable insights for future development.

By pursuing these directions, we aim to further advance the capability and efficiency of 1-bit LLMs, paving the way for more sustainable and accessible artificial intelligence. The open-source release of BitNet b1.58 2B4T and its associated tools provides a foundation for the community to build upon these efforts.

<span id="section-8"></span>

## 8 Open-weight Baselines

We summarize the links to the open-weight LLMs evaluated in this work as below:

- **LLaMA 3.2 1B**: [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
- **Gemma-3 1B**: [google/gemma-3-1b-it](https://huggingface.co/google/gemma-3-1b-it)
- **Qwen2.5 0.5B**: [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- **Qwen2.5 1.5B**: [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
- **Qwen2.5 3B**: [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **SmolLM2 1.7B**: [HuggingFaceTB/SmolLM2-1.7B-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- **MiniCPM 2B**: [openbmb/MiniCPM-2B-dpo-bf16](https://huggingface.co/openbmb/MiniCPM-2B-dpo-bf16)
- **Qwen2.5 1.5B-GPTQ-int4**: [Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GPTQ-Int4)
- **Qwen2.5 1.5B-AWQ-int4**: [Qwen/Qwen2.5-1.5B-Instruct-AWQ](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-AWQ)
- **Bonsai 0.5B**: [deepgrove/Bonsai](https://huggingface.co/deepgrove/Bonsai)
- **OLMo-Bitnet 1B**: [NousResearch/OLMo-Bitnet-1B](https://huggingface.co/NousResearch/OLMo-Bitnet-1B)
- **Falcon3-1.58bit 7B**: [tiiuae/Falcon3-7B-Instruct-1.58bit](https://huggingface.co/tiiuae/Falcon3-7B-Instruct-1.58bit)
- **Llama3-8B-1.58 8B**: [HF1BitLLM/Llama3-8B-1.58-100B-tokens](https://huggingface.co/HF1BitLLM/Llama3-8B-1.58-100B-tokens)

<span id="section-9"></span>

## 9 Evaluation Pipeline Details

To ensure standardized evaluation, we employed established toolkits for different benchmark categories. Specifically:

- For the HumanEval+ coding benchmark, we utilized the [evalplus](https://github.com/evalplus/evalplus) toolkit.
- For the MATH-500 mathematical reasoning benchmark, we used a customized version of the [math-evaluation-harness](https://github.com/ZubinGou/math-evaluation-harness) toolkit.
- For the MT-Bench conversational benchmark, evaluation was performed using the official [LLM Judge](https://github.com/lm-sys/FastChat/blob/main/fastchat/llm_judge/README.md) open-source codebase.
- For all other benchmarks assessing language understanding, reasoning, knowledge, and comprehension, we used the standard [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) framework.

Models were prompted using a chat format for generative tasks (e.g., GSM8K, IFEval, and MT-Bench), while default settings from the respective toolkits were used for other tasks.

<span id="table-04"></span>

![Table 4. ADD and MUL energy consumption (in pJ) of different precision at 7nm process nodes.](../../papers/bitnet-b1-58-2b4t/table-04.png)

**Table 4.** ADD and MUL energy consumption (in pJ) of different precision at 7nm process nodes.

For energy consumption, we utilize the energy model in [Hor14, Zha22g] to estimate the arithmetic operations energy (AOE) of matrix multiplication. The sequence length is set as 512 tokens. We present the energy consumption for ADD and MUL operation at 7nm process nodes in [Table 4](#table-04).

To assess CPU decoding performance, latency measurements were conducted on a Surface Laptop Studio 2 system powered by a 13th Gen Intel Core i7-13800H processor. The benchmarking process utilized 8 CPU threads. Specifically, the BitNet b1.58 2B4T model was tested using its *bitnet.cpp* implementation, whereas other models were evaluated using the *llama.cpp* framework. For each model, we generated 128 tokens and report the average latency per token for this task.

[+1]: [https://huggingface.co/NousResearch/OLMo-Bitnet-1B](https://huggingface.co/NousResearch/OLMo-Bitnet-1B)
