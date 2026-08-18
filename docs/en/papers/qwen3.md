---
title: 'Qwen3 Technical Report'
createTime: 2026/08/18 18:48:40
permalink: /en/papers/qwen3/
pageClass: paper-reading
---

> [An Yang](https://dblp.org/pid/63/10551), [Anfeng Li](https://dblp.org/pid/408/0604), [Baosong Yang](https://dblp.org/pid/203/8245), [Beichen Zhang](https://dblp.org/pid/71/9257), [Binyuan Hui](https://dblp.org/pid/246/4699), [Bo Zheng](https://dblp.org/pid/33/1610-7), [Bowen Yu](https://dblp.org/pid/95/10266-2), [Chang Gao](https://gao-xiao-bai.github.io/), [Chengen Huang](https://dblp.org/pid/330/8484), [Chenxu Lv](https://dblp.org/pid/297/4112), [Chujie Zheng](https://chujiezheng.github.io/), [Dayiheng Liu](https://dblp.org/pid/189/4488), [Fan Zhou](https://koalazf99.github.io/), [Fei Huang](https://dblp.org/pid/h/FeiHuang-5), [Feng Hu](https://dblp.org/pid/74/6975), [Hao Ge](https://dblp.org/pid/76/5849), [Haoran Wei](https://dblp.org/pid/183/9682), [Huan Lin](https://dblp.org/pid/45/3289), [Jialong Tang](https://tangjialong.github.io/), [Jian Yang](https://dblp.org/pid/y/JianYang3), [Jianhong Tu](https://dblp.org/pid/227/8305), [Jianwei Zhang](https://dblp.org/pid/144/1628-12), [Jianxin Yang](https://dblp.org/pid/242/4275), [Jiaxi Yang](https://dblp.org/pid/293/9901-4), [Jing Zhou](https://orcid.org/0009-0002-1701-9384), [Jingren Zhou](https://dblp.org/pid/84/2644-1), [Junyang Lin](https://dblp.org/pid/215/3823), [Kai Dang](https://dblp.org/pid/241/2644), [Keqin Bao](https://dblp.org/pid/331/5509), [Kexin Yang](https://dblp.org/pid/54/774-2), [Le Yu](https://dblp.org/pid/23/7122), [Lianghao Deng](https://dblp.org/pid/395/4128), [Mei Li](https://dblp.org/pid/06/1233), [Mingfeng Xue](https://dblp.org/pid/239/2887), [Mingze Li](https://dblp.org/pid/00/8348), [Pei Zhang](https://dblp.org/pid/78/5323-11), [Peng Wang](https://dblp.org/pid/95/4442-28), [Qin Zhu](https://dblp.org/pid/08/904), [Rui Men](https://dblp.org/pid/170/0093), [Ruize Gao](https://dblp.org/pid/180/4683), [Shixuan Liu](https://dblp.org/pid/152/3598), [Shuang Luo](https://scholar.google.com/citations?user=HvJiMJoAAAAJ), [Tianhao Li](https://dblp.org/pid/69/2238), [Tianyi Tang](https://steventang1998.github.io/), [Wenbiao Yin](https://dblp.org/pid/330/7482), [Xingzhang Ren](https://dblp.org/pid/218/6803), [Xinyu Wang](https://dblp.org/pid/68/1277-13), [Xinyu Zhang](https://dblp.org/pid/58/4582-17), [Xuancheng Ren](https://dblp.org/pid/202/2250), [Yang Fan](https://dblp.org/pid/81/5991), [Yang Su](https://dblp.org/pid/17/686), [Yichang Zhang](https://dblp.org/pid/165/9507), [Yinger Zhang](https://dblp.org/pid/293/6628), [Yu Wan](https://dblp.org/pid/06/6328-4), [Yuqiong Liu](https://dblp.org/pid/45/4771), [Zekun Wang](https://kugwzk.github.io/), [Zeyu Cui](https://dblp.org/pid/236/6347), [Zhenru Zhang](https://dblp.org/pid/311/4174), [Zhipeng Zhou](https://www.sciencedirect.com/science/article/pii/S0005109826001482), and [Zihan Qiu](https://dblp.org/pid/313/9471). First submitted to arXiv on May 14, 2025; current version v1. [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388). [Original PDF](/paper/qwen3.pdf). [DOI](https://doi.org/10.48550/arXiv.2505.09388). [TeX source](https://export.arxiv.org/e-print/2505.09388v1). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models—such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)—and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0.

<span id="section-1"></span>

## 1 Introduction

The pursuit of artificial general intelligence (AGI) or artificial super intelligence (ASI) has long been a goal for humanity. Recent advancements in large foundation models, e.g., GPT-4o [Gpt24], Claude 3.7 [Ant25b], Gemini 2.5 [Gem25], DeepSeek-V3 [Dee24a], Llama-4 [Lla25], and Qwen2.5 [Yang24], have demonstrated significant progress toward this objective. These models are trained on vast datasets spanning trillions of tokens across diverse domains and tasks, effectively distilling human knowledge and capabilities into their parameters. Furthermore, recent developments in reasoning models, optimized through reinforcement learning, highlight the potential for foundation models to enhance inference-time scaling and achieve higher levels of intelligence, e.g., o3 [Ope25], DeepSeek-R1 [Guo25]. While most state-of-the-art models remain proprietary, the rapid growth of open-source communities has substantially reduced the performance gap between open-weight and closed-source models. Notably, an increasing number of top-tier models [Lla25, Dee24a, Guo25, Yang24] are now being released as open-source, fostering broader research and innovation in artificial intelligence.

In this work, we introduce Qwen3, the latest series in our foundation model family, Qwen. Qwen3 is a collection of open-weight large language models (LLMs) that achieve state-of-the-art performance across a wide variety of tasks and domains. We release both dense and Mixture-of-Experts (MoE) models, with the number of parameters ranging from 0.6 billion to 235 billion, to meet the needs of different downstream applications. Notably, the flagship model, Qwen3-235B-A22B, is an MoE model with a total of 235 billion parameters and 22 billion activated ones per token. This design ensures both high performance and efficient inference.

Qwen3 introduces several key advancements to enhance its functionality and usability. First, it integrates two distinct operating modes, thinking mode and non-thinking mode, into a single model. This allows users to switch between these modes without alternating between different models, e.g., switching from Qwen2.5 to QwQ [Qwq24]. This flexibility ensures that developers and users can adapt the model's behavior to suit specific tasks efficiently. Additionally, Qwen3 incorporates thinking budgets, providing users with fine-grained control over the level of reasoning effort applied by the model during task execution. This capability is crucial to the optimization of computational resources and performance, tailoring the model's thinking behavior to meet varying complexity in real-world applications. Furthermore, Qwen3 has been pre-trained on 36 trillion tokens covering up to 119 languages and dialects, effectively enhancing its multilingual capabilities. This broadened language support amplifies its potential for deployment in global use cases and international applications. These advancements together establish Qwen3 as a cutting-edge open-source large language model family, capable of effectively addressing complex tasks across various domains and languages.

The pre-training process for Qwen3 utilizes a large-scale dataset consisting of approximately 36 trillion tokens, curated to ensure linguistic and domain diversity. To efficiently expand the training data, we employ a multi-modal approach: Qwen2.5-VL [Bai25a] is finetuned to extract text from extensive PDF documents. We also generate synthetic data using domain-specific models: Qwen2.5-Math [Yang24a] for mathematical content and Qwen2.5-Coder [Hui24] for code-related data. The pre-training process follows a three-stage strategy. In the first stage, the model is trained on about 30 trillion tokens to build a strong foundation of general knowledge. In the second stage, it is further trained on knowledge-intensive data to enhance reasoning abilities in areas like science, technology, engineering, and mathematics (STEM) and coding. Finally, in the third stage, the model is trained on long-context data to increase its maximum context length from 4,096 to 32,768 tokens.

To better align foundation models with human preferences and downstream applications, we employ a multi-stage post-training approach that empowers both thinking (reasoning) and non-thinking modes. In the first two stages, we focus on developing strong reasoning abilities through long chain-of-thought (CoT) cold-start finetuning and reinforcement learning focusing on mathematics and coding tasks. In the final two stages, we combine data with and without reasoning paths into a unified dataset for further fine-tuning, enabling the model to handle both types of input effectively, and we then apply general-domain reinforcement learning to improve performance across a wide range of downstream tasks. For smaller models, we use strong-to-weak distillation, leveraging both off-policy and on-policy knowledge transfer from larger models to enhance their capabilities. Distillation from advanced teacher models significantly outperforms reinforcement learning in performance and training efficiency.

We evaluate both pre-trained and post-trained versions of our models across a comprehensive set of benchmarks spanning multiple tasks and domains. Experimental results show that our base pre-trained models achieve state-of-the-art performance. The post-trained models, whether in thinking or non-thinking mode, perform competitively against leading proprietary models and large mixture-of-experts (MoE) models such as o1, o3-mini, and DeepSeek-V3. Notably, our models excel in coding, mathematics, and agent-related tasks. For example, the flagship model Qwen3-235B-A22B achieves 85.7 on AIME'24 and 81.5 on AIME'25 [Aim25a], 70.7 on LiveCodeBench v5 [Jai24], 2,056 on CodeForces, and 70.8 on BFCL v3 [Yan24d]. In addition, other models in the Qwen3 series also show strong performance relative to their size. Furthermore, we observe that increasing the thinking budget for thinking tokens leads to a consistent improvement in the model's performance across various tasks.

In the following sections, we describe the design of the model architecture, provide details on its training procedures, present the experimental results of pre-trained and post-trained models, and finally, conclude this technical report by summarizing the key findings and outlining potential directions for future research.

<span id="section-2"></span>

## 2 Architecture

The Qwen3 series includes 6 dense models, namely Qwen3-0.6B, Qwen3-1.7B, Qwen3-4B, Qwen3-8B, Qwen3-14B, and Qwen3-32B, and 2 MoE models, Qwen3-30B-A3B and Qwen3-235B-A22B. The flagship model, Qwen3-235B-A22B, has a total of 235B parameters with 22B activated ones. Below, we elaborate on the architecture of the Qwen3 models.

The architecture of the Qwen3 dense models is similar to Qwen2.5 [Yang24], including using Grouped Query Attention (GQA, [Ain23]), SwiGLU [Yan17], Rotary Positional Embeddings (RoPE, [Su24]), and RMSNorm [Jia23b] with pre-normalization. Besides, we remove QKV-bias used in Qwen2 [Yang24b] and introduce QK-Norm [Deh23] to the attention mechanism to ensure stable training for Qwen3. Key information on model architecture is provided in [Table 1](#table-01).

The Qwen3 MoE models share the same fundamental architecture as the Qwen3 dense models. Key information on model architecture is provided in [Table 2](#table-02). We follow Qwen2.5-MoE [Yang24] and implement fine-grained expert segmentation [Dai24]. The Qwen3 MoE models have 128 total experts with 8 activated experts per token. Unlike Qwen2.5-MoE, the Qwen3-MoE design excludes shared experts. Furthermore, we adopt the global-batch load balancing loss [Qiu25a] to encourage expert specialization. These architectural and training innovations have yielded substantial improvements in model performance across downstream tasks.

Qwen3 models utilize Qwen's tokenizer [Bai23b], which implements byte-level byte-pair encoding (BBPE, [Bro20b, Wan20e, Sen15]) with a vocabulary size of 151,669.

<span id="table-01"></span>

![Table 1: Model architecture of Qwen3 dense models.](../../papers/qwen3/table-01.png)

**Table 1.** Model architecture of Qwen3 dense models.

<span id="table-02"></span>

![Table 2: Model architecture of Qwen3 MoE models.](../../papers/qwen3/table-02.png)

**Table 2.** Model architecture of Qwen3 MoE models.

<span id="section-3"></span>

## 3 Pre-training

In this section, we describe the construction of our pretraining data, the details of our pretraining approach, and present experimental results from evaluating the base models on standard benchmarks.

<span id="section-3-1"></span>

### 3.1 Pre-training Data
Compared with Qwen2.5 [Yang24], we have significantly expanded the scale and diversity of our training data. Specifically, we collected twice as many pre-training tokens—covering three times more languages. All Qwen3 models are trained on a large and diverse dataset consisting of **119 languages and dialects**, with a total of **36 trillion tokens**. This dataset includes high-quality content in various domains such as coding, STEM (Science, Technology, Engineering, and Mathematics), reasoning tasks, books, multilingual texts, and synthetic data.

To further expand the pre-training data corpus, we first employ the Qwen2.5-VL model [Bai25a] to perform text recognition on a large volume of PDF-like documents. The recognized text is then refined using the Qwen2.5 model [Yang24], which helps improve its quality. Through this two-step process, we are able to obtain an additional set of high-quality text tokens, amounting to trillions in total. Besides, we employ Qwen2.5 [Yang24], Qwen2.5-Math [Yang24a], and Qwen2.5-Coder [Hui24] models to synthesize trillions of text tokens in different formats, including textbooks, question-answering, instructions, and code snippets, covering dozens of domains. Finally, we further expand the pre-training corpus by incorporating additional multilingual data and introducing more languages. Compared to the pre-training data used in Qwen2.5, the number of supported languages has been significantly increased from 29 to 119, enhancing the model's linguistic coverage and cross-lingual capabilities.

We have developed a multilingual data annotation system designed to enhance both the quality and diversity of training data. This system has been applied to our large-scale pre-training datasets, annotating over 30 trillion tokens across multiple dimensions such as educational value, fields, domains, and safety. These detailed annotations support more effective data filtering and combination. Unlike previous studies [Xie23, Fan23, Liu24q] that optimize the data mixture at the data source or domain level, our method optimizes the data mixture at the instance-level through extensive ablation experiments on small proxy models with the fine-grained data labels.

<span id="section-3-2"></span>

### 3.2 Pre-training Stage
The Qwen3 models are pre-trained through a three-stage process:


1. **General Stage (S1)**: At the first pre-training stage, all Qwen3 models are trained on over 30 trillion tokens using a sequence length of 4,096 tokens. At this stage, the models have been fully pre-trained on language proficiency and general world knowledge, with training data covering 119 languages and dialects.


1. **Reasoning Stage (S2)**: To further improve the reasoning ability, we optimize the pre-training corpus of this stage by increasing the proportion of STEM, coding, reasoning, and synthetic data. The models are further pre-trained with about 5T higher-quality tokens at a sequence length of 4,096 tokens. We also accelerate the learning rate decay during this stage.


1. **Long Context Stage**: In the final pre-training stage, we collect high-quality long context corpora to extend the context length of Qwen3 models. All models are pre-trained on hundreds of billions of tokens with a sequence length of 32,768 tokens. The long context corpus includes 75% of text between 16,384 to 32,768 tokens in length, and 25% of text between 4,096 to 16,384 in length. Following Qwen2.5 [Yang24], we increase the base frequency of RoPE from 10,000 to 1,000,000 using the ABF technique [Xio23]. Meanwhile, we introduce YARN [Pen23] and Dual Chunk Attention (DCA, [An24]) to achieve a four-fold increase in sequence length capacity during inference.

Similar to Qwen2.5 [Yang24], we develop scaling laws for optimal hyper-parameters (e.g., learning rate scheduler, and batch size) predictions based on three pre-training stages mentioned above. Through extensive experiments, we systematically study the relationship between model architecture, training data, training stage, and optimal training hyper-parameters. Finally, we set the predicted optimal learning rate and batch size strategy for each dense or MoE model.

<span id="section-3-3"></span>

### 3.3 Pre-training Evaluation

We conduct comprehensive evaluations of the base language models of the Qwen3 series. The evaluation of base models mainly focuses on their performance in general knowledge, reasoning, mathematics, scientific knowledge, coding, and multilingual capabilities. The evaluation datasets for pre-trained base models include 15 benchmarks:


- **General Tasks**: MMLU [Hen20] (5-shot), MMLU-Pro [Wan24c] (5-shot, CoT), MMLU-redux [Gem24a] (5-shot), BBH [Suz22] (3-shot, CoT), SuperGPQA [Du25a](5-shot, CoT).


- **Math & STEM Tasks**: GPQA [Rei24] (5-shot, CoT), GSM8K [Cob21] (4-shot, CoT), MATH [Hen21] (4-shot, CoT).


- **Coding Tasks**: EvalPlus [Liu24i] (0-shot) (Average of HumanEval [Che21], MBPP [Aus21], Humaneval+, MBPP+) [Liu24i], MultiPL-E [Cas23] (0-shot) (Python, C++, JAVA, PHP, TypeScript, C#, Bash, JavaScript), MBPP-3shot [Aus21], CRUX-O of CRUXEval (1-shot) [Gu24].


- **Multilingual Tasks**: MGSM [Shi23] (8-shot, CoT), MMMLU [Ope24c] (5-shot), INCLUDE [Rom24] (5-shot).

For the base model baselines, we compare the Qwen3 series base models with the Qwen2.5 base models [Yang24] and other leading open-source base models, including DeepSeek-V3 Base [Dee24a], Gemma-3 [Gem25a], Llama-3 [Dub24], and Llama-4 [Lla25] series base models, in terms of scale of parameters. All models are evaluated using the same evaluation pipeline and the widely-used evaluation settings to ensure fair comparison.

**Summary of Evaluation Results** Based on the overall evaluation results, we highlight some key conclusions of Qwen3 base models.


1. Compared with the previously open-source SOTA dense and MoE base models (such as DeepSeek-V3 Base, Llama-4-Maverick Base, and Qwen2.5-72B-Base), Qwen3-235B-A22B-Base outperforms these models in most tasks with significantly fewer total parameters or activated parameters.


1. For the Qwen3 MoE base models, our experimental results indicate that: (a) Using the same pre-training data, Qwen3 MoE base models can achieve similar performance to Qwen3 dense base models with only **1/5** activated parameters. (b) Due to the improvements of the Qwen3 MoE architecture, the scale-up of the training tokens, and more advanced training strategies, the Qwen3 MoE base models can outperform the Qwen2.5 MoE base models with less than **1/2** activated parameters and fewer total parameters. (c) Even with **1/10** of the activated parameters of the Qwen2.5 dense base model, the Qwen3 MoE base model can achieve comparable performance, which brings us significant advantages in inference and training costs.


1. The overall performance of the Qwen3 dense base models is comparable to the Qwen2.5 base models at higher parameter scales. For example, Qwen3-1.7B/4B/8B/14B/32B-Base achieve comparable performance to Qwen2.5-3B/7B/14B/32B/72B-Base, respectively. Especially in STEM, coding, and reasoning benchmarks, the performance of Qwen3 dense base models even surpasses Qwen2.5 base models at higher parameter scales.

The detailed results are as follows.

<span id="table-03"></span>

![Table 3: Comparison among Qwen3-235B-A22B-Base and other representative strong open-source baselines. The highest, the second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-03.png)

**Table 3.** **Comparison among Qwen3-235B-A22B-Base and other representative strong open-source baselines. The highest, the second-best scores are shown in bold and underlined, respectively.**

**Qwen3-235B-A22B-Base** We compare Qwen3-235B-A22B-Base to our previous similar-sized MoE Qwen2.5-Plus-Base [Yang24] and other leading open-source base models: Llama-4-Maverick [Lla25], Qwen2.5-72B-Base [Yang24], DeepSeek-V3 Base [Dee24a]. From the results in [Table 3](#table-03), the Qwen3-235B-A22B-Base model attains the highest performance scores across most of the evaluated benchmarks. We further compare Qwen3-235B-A22B-Base with other baselines separately for the detailed analysis.


1. Compared with the recently open-source model Llama-4-Maverick-Base, which has about **twice** the number of parameters, Qwen3-235B-A22B-Base still performs better on most benchmarks.


1. Compared with the previously state-of-the-art open-source model DeepSeek-V3-Base, Qwen3-235B-A22B-Base outperforms DeepSeek-V3-Base on 14 out of 15 evaluation benchmarks with only about **1/3** the total number of parameters and **2/3** activated parameters, demonstrating the powerful and cost-effectiveness of our models.


1. Compared with our previous MoE Qwen2.5-Plus of similar size, Qwen3-235B-A22B-Base significantly outperforms it with fewer parameters and activated parameters, which shows the remarkable advantages of Qwen3 in pre-training data, training strategy, and model architecture.


1. Compared with our previous flagship open-source dense model Qwen2.5-72B-Base, Qwen3-235B-A22B-Base surpasses the latter in all benchmarks and uses fewer than **1/3** of the activated parameters. Meanwhile, due to the advantage of the model architecture, the inference costs and training costs on each trillion tokens of Qwen3-235B-A22B-Base are much cheaper than those of Qwen2.5-72B-Base.

<span id="table-04"></span>

![Table 4: Comparison among Qwen3-32B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-04.png)

**Table 4.** **Comparison among Qwen3-32B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-05"></span>

![Table 5: Comparison among Qwen3-14B-Base, Qwen3-30B-A3B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-05.png)

**Table 5.** **Comparison among Qwen3-14B-Base, Qwen3-30B-A3B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-06"></span>

![Table 6: Comparison among Qwen8B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-06.png)

**Table 6.** **Comparison among Qwen8B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-07"></span>

![Table 7: Comparison among Qwen3-4B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-07.png)

**Table 7.** **Comparison among Qwen3-4B-Base and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-08"></span>

![Table 8: Comparison among Qwen3-1.7B-Base, Qwen3-0.6B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-08.png)

**Table 8.** **Comparison among Qwen3-1.7B-Base, Qwen3-0.6B-Base, and other strong open-source baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

**Qwen3-32B-Base** Qwen3-32B-Base is our largest dense model among the Qwen3 series. We compare it to the baselines of similar sizes, including Gemma-3-27B [Gem25a] and Qwen2.5-32B [Yang24]. In addition, we introduce two strong baselines: the recently open-source MoE model Llama-4-Scout, which has three times the parameters of Qwen3-32B-Base but half the activated parameters; and our previous flagship open-source dense model Qwen2.5-72B-Base, which has more than twice the number of parameters compared to Qwen3-32B-Base. The results are shown in [Table 4](#table-04), which support three key conclusions:


1. Compared with the similar-sized models, Qwen3-32B-Base outperforms Qwen2.5-32B-Base and Gemma-3-27B Base on most benchmarks. Notably, Qwen3-32B-Base achieves 65.54 on MMLU-Pro and 39.78 on SuperGPQA, significantly outperforming its predecessor Qwen2.5-32B-Base. In addition, Qwen3-32B-Base achieves significantly higher encoding benchmark scores than all baseline models.


1. Surprisingly, we find that Qwen3-32B-Base achieves competitive results compared to Qwen2.5-72B-Base. Although Qwen3-32B-Base has less than half the number of parameters of Qwen2.5-72B-Base, it outperforms Qwen2.5-72B-Base in 10 of the 15 evaluation benchmarks. On coding, mathematics, and reasoning benchmarks, Qwen3-32B-Base has remarkable advantages.


1. Compared to Llama-4-Scout-Base, Qwen3-32B-Base significantly outperforms it on all 15 benchmarks, with only one-third of the number of parameters of Llama-4-Scout-Base, but twice the number of activated parameters.

**Qwen3-14B-Base & Qwen3-30B-A3B-Base** The evaluation of the Qwen3-14B-Base and Qwen3-30B-A3B-Base is compared against baselines of similar sizes, including Gemma-3-12B Base, Qwen2.5-14B Base. Similarly, we also introduce two strong baselines: (1) Qwen2.5-Turbo [Yang24], which has 42B parameters and 6B activated parameters. Note that its activated parameters are twice those of Qwen3-30B-A3B-Base. (2) Qwen2.5-32B-Base, which has 11 times the activated parameters of Qwen3-30B-A3B and more than twice that of Qwen3-14B. The results are shown in [Table 5](#table-05), where we can draw the following conclusions.


1. Compared with the similar-sized models, Qwen3-14B-Base significantly performs better than Qwen2.5-14B-Base and Gemma-3-12B-Base on all 15 benchmarks.


1. Similarly, Qwen3-14B-Base also achieves very competitive results compared to Qwen2.5-32B-Base with less than half of the parameters.


1. With only 1/5 activated non-embedding parameters, Qwen3-30B-A3B significantly outperforms Qwen2.5-14B-Base on all tasks, and achieves comparable performance to Qwen3-14B-Base and Qwen2.5-32B-Base, which brings us significant advantages in inference and training costs.

**Qwen3-8B / 4B / 1.7B / 0.6B-Base** For edge-side models, we take similar-sized Qwen2.5, Llama-3, and Gemma-3 base models as the baselines. The results can be seen in [Table 6](#table-06), [Table 7](#table-07), and [Table 8](#table-08). All Qwen3 8B / 4B / 1.7B / 0.6B-Base models continue to maintain strong performance across nearly all benchmarks. Notably, Qwen3-8B / 4B / 1.7B-Base models even outperform larger size Qwen2.5-14B / 7B / 3B Base models on over half of the benchmarks, especially on STEM-related and coding benchmarks, reflecting the significant improvement of the Qwen3 models.

<span id="section-4"></span>

## 4 Post-training

<span id="figure-01"></span>

![Figure 1: Post-training pipeline of the Qwen3 series models.](../../papers/qwen3/figure-01.png)

**Figure 1.** Post-training pipeline of the Qwen3 series models.

The post-training pipeline of Qwen3 is strategically designed with two core objectives:


1. **Thinking Control**: This involves the integration of two distinct modes, namely the “non-thinking” and “thinking” modes, providing users with the flexibility to choose whether the model should engage in reasoning or not, and to control the depth of thinking by specifying a token budget for the thinking process.


1. **Strong-to-Weak Distillation**: This aims to streamline and optimize the post-training process for lightweight models. By leveraging the knowledge from large-scale models, we substantially reduce both the computational costs and the development efforts required for building smaller-scale models.

As illustrated in [Figure 1](#figure-01), the flagship models in the Qwen3 series follow a sophisticated four-stage training process. The first two stages focus on developing the models' “thinking” abilities. The next two stages aim to integrate strong “non-thinking” functionalities into the models.

Preliminary experiments suggest that directly distilling the output logits from teacher models into lightweight student models can effectively enhance their performance while maintaining fine-grained control over their reasoning processes. This approach eliminates the necessity of performing an exhaustive four-stage training process individually for every small-scale model. It leads to better immediate performance, as indicated by higher Pass@1 scores, and also improves the model's ability of exploration, as reflected in improved Pass@64 results. In addition, it achieves these gains with much greater training efficiency, requiring only 1/10 of the GPU hours compared to the four-stage training method.

In the following sections, we present the four-stage training process and provide a detailed explanation of the Strong-to-Weak Distillation approach.

<span id="section-4-1"></span>

### 4.1 Long-CoT Cold Start

We begin by curating a comprehensive dataset that spans a wide range of categories, including math, code, logical reasoning, and general STEM problems. Each problem in the dataset is paired with verified reference answers or code-based test cases. This dataset serves as the foundation for the “cold start” phase of long Chain-of-Thought (long-CoT) training.

The dataset construction involves a rigorous two-phase filtering process: query filtering and response filtering. In the query filtering phase, we use Qwen2.5-72B-Instruct to identify and remove queries that are not easily verifiable. This includes queries containing multiple sub-questions or those asking for general text generation. Furthermore, we exclude queries that Qwen2.5-72B-Instruct can answer correctly without using CoT reasoning. This helps prevent the model from relying on superficial guessing and ensures that only complex problems requiring deeper reasoning are included. Additionally, we annotate each query's domain using Qwen2.5-72B-Instruct to maintain balanced domain representation across the dataset.

After reserving a validation query set, we generate $N$ candidate responses for each remaining query using QwQ-32B [Qwq25]. When QwQ-32B consistently fails to generate correct solutions, human annotators manually assess the accuracy of the responses. For queries with positive Pass@$N$, further stringent filtering criteria are applied to remove responses that (1) yield incorrect final answers, (2) contain substantial repetition, (3) clearly indicate guesswork without adequate reasoning, (4) exhibit inconsistencies between the thinking and summary contents, (5) involve inappropriate language mixing or stylistic shifts, or (6) are suspected of being overly similar to potential validation set items. Subsequently, a carefully selected subset of the refined dataset is used for the initial cold-start training of the reasoning patterns. The objective at this stage is to instill foundational reasoning patterns in the model without overly emphasizing immediate reasoning performance. This approach ensures that the model's potential is not limited, allowing for greater flexibility and improvement during the subsequent reinforcement learning (RL) phase. To achieve this objective effectively, it is preferable to minimize both the number of training samples and the training steps during this preparatory phase.

<span id="section-4-2"></span>

### 4.2 Reasoning RL

The query-verifier pairs used in the Reasoning RL stage must satisfy the following four criteria: (1) They were not used during the cold-start phase. (2) They are learnable for the cold-start model. (3) They are as challenging as possible. (4) They cover a broad range of sub-domains. We ultimately collect a total of 3,995 query-verifier pairs, and employed GRPO [Sha24] to update the model parameters. We observe that using a large batch size and a high number of rollouts per query, along with off-policy training to improve sample efficiency, is beneficial to the training process. We have also addressed how to balance exploration and exploitation by controlling the model’s entropy to increase steadily or remain stable, which is crucial for maintaining stable training. As a result, we achieve consistent improvements in both training reward and validation performance over the course of a single RL run, without any manual intervention on hyperparameters. For instance, the AIME'24 score of the Qwen3-235B-A22B model increases from 70.1 to 85.1 over a total of 170 RL training steps.

<span id="section-4-3"></span>

### 4.3 Thinking Mode Fusion

The goal of the Thinking Mode Fusion stage is to integrate the “non-thinking” capabilities into the previously developed “thinking” model. This approach allows developers to manage and control reasoning behaviors, while also reducing the cost and complexity of deploying separate models for thinking and non-thinking tasks. To achieve this, we conduct continual supervised fine-tuning (SFT) on the Reasoning RL model and design a chat template to fuse the two modes. Moreover, we find that models capable of handling both modes proficiently perform consistently well under different thinking budgets.

**Construction of SFT data.** The SFT dataset combines both the “thinking” and “non-thinking” data. To ensure that the performance of the Stage 2 model is not compromised by the additional SFT, the “thinking” data is generated via rejection sampling on Stage 1 queries using the Stage 2 model itself. The “non-thinking” data, on the other hand, is carefully curated to cover a diverse range of tasks, including coding, mathematics, instruction-following, multilingual tasks, creative writing, question answering, and role-playing. Additionally, we employ automatically generated checklists for assessing the response quality of “non-thinking” data. To enhance the performance on tasks with low-resource languages, we particularly increase the proportion of translation tasks.

**Chat Template Design.** To better integrate the two modes and enable users to dynamically switch the model's thinking process, we design chat templates for Qwen3, as shown in [Table 9](#table-09). Specifically, for samples in thinking mode and non-thinking mode, we introduce `/think` and `/no_think` flags in the user query or system message, respectively. This allows the model to follow the user's input and select the appropriate thinking mode accordingly. For non-thinking mode samples, we retain an empty thinking block in the assistant's response. This design ensures internal format consistency within the model and allows developers to prevent the model from engaging in thinking behavior by concatenating an empty think block in the chat template. By default, the model operates in thinking mode; therefore, we add some thinking mode training samples where the user queries do not include `/think` flags. For more complex multi-turn dialogs, we randomly insert multiple `/think` and `/no_think` flags into users' queries, with the model response adhering to the last flag encountered.

**Thinking Budget.** An additional advantage of Thinking Mode Fusion is that, once the model learns to respond in both non-thinking and thinking modes, it naturally develops the ability to handle intermediate cases—generating responses based on incomplete thinking. This capability lays the foundation for implementing budget control over the model's thinking process. Specifically, when the length of the model's thinking reaches a user-defined threshold, we manually halt the thinking process and insert the stop-thinking instruction: “`Considering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>.\n\n`”. After this instruction is inserted, the model proceeds to generate a final response based on its accumulated reasoning up to that point. It is worth noting that this ability is not explicitly trained but emerges naturally as a result of applying Thinking Mode Fusion.

<span id="table-09"></span>

![Table 9: Examples of SFT data for thinking and non-thinking modes during the thinking mode fusion stage. For the thinking mode, the /think flag can be omitted since it represents the default behavior. This feature has been implemented in the chat template supported by the Hugging Face's tokenizer, where the thinking mode can be disabled using an additional parameter enable_thinking=False.](../../papers/qwen3/table-09.png)

**Table 9.** **Examples of SFT data for thinking and non-thinking modes during the thinking mode fusion stage.** For the thinking mode, the `/think` flag can be omitted since it represents the default behavior. This feature has been implemented in the chat template [+1] supported by the Hugging Face's tokenizer, where the thinking mode can be disabled using an additional parameter `enable_thinking=False`.

<span id="section-4-4"></span>

### 4.4 General RL

The General RL stage aims to broadly enhance the models' capabilities and stability across diverse scenarios. To facilitate this, we have established a sophisticated **reward system** covering **over 20 distinct tasks**, each with customized scoring criteria. These tasks specifically target enhancements in the following core capabilities:


- **Instruction Following**: This capability ensures that models accurately interpret and follow user instructions, including requirements related to content, format, length, and the use of structured output, delivering responses that align with user expectations.


- **Format Following**: In addition to explicit instructions, we expect the model to adhere to specific formatting conventions. For instance, it should respond appropriately to the `/think` and `/no_think` flags by switching between thinking and non-thinking modes, and consistently use designated tokens (e.g., `<think>` and `</think>`) to separate the thinking and response parts in the final output.


- **Preference Alignment**: For open-ended queries, preference alignment focuses on improving the model’s helpfulness, engagement, and style, ultimately delivering a more natural and satisfying user experience.


- **Agent Ability**: This involves training the model to correctly invoke tools via designated interfaces. During the RL rollout, the model is allowed to perform complete multi-turn interaction cycles with real environment execution feedback, thereby improving its performance and stability in long-horizon decision-making tasks.


- **Abilities for Specialized Scenarios**: In more specialized scenarios, we design tasks tailored to the specific context. For example, in Retrieval-Augmented Generation (RAG) tasks, we incorporate reward signals to guide the model toward generating accurate and contextually appropriate responses, thereby minimizing the risk of hallucination.

To provide feedback for the aforementioned tasks, we utilized three distinct types of rewards:


1. **Rule-based Reward**: The rule-based reward has been widely used in the reasoning RL stage, and is also useful for general tasks such as instruction following [Lam24] and format adherence. Well-designed rule-based rewards can assess the correctness of model outputs with high precision, preventing issues like reward hacking.


1. **Model-based Reward with Reference Answer**: In this approach, we provide a reference answer for each query and prompt Qwen2.5-72B-Instruct to score the model's response based on this reference. This method allows for more flexible handling of diverse tasks without requiring strict formatting, avoiding false negatives that can occur with purely rule-based rewards.


1. **Model-based Reward without Reference Answer**: Leveraging human preference data, we train a reward model to assign scalar scores to model responses. This approach, which does not depend on a reference answer, can handle a broader range of queries while effectively enhancing the model's engagement and helpfulness.

<span id="section-4-5"></span>

### 4.5 Strong-to-Weak Distillation

The Strong-to-Weak Distillation pipeline is specifically designed to optimize lightweight models, encompassing 5 dense models (Qwen3-0.6B, 1.7B, 4B, 8B, and 14B) and one MoE model (Qwen3-30B-A3B). This approach enhances model performance while effectively imparting robust mode-switching capabilities. The distillation process is divided into two primary phases:


1. **Off-policy Distillation**: At this initial phase, we combine the outputs of teacher models generated with both `/think` and `/no_think` modes for response distillation. This helps lightweight student models develop basic reasoning skills and the ability to switch between different modes of thinking, laying a solid foundation for the next on-policy training phase.


1. **On-policy Distillation**: In this phase, the student model generates on-policy sequences for fine-tuning. Specifically, prompts are sampled, and the student model produces responses in either `/think` or `/no_think` mode. The student model is then fine-tuned by aligning its logits with those of a teacher model (Qwen3-32B or Qwen3-235B-A22B) to minimize the KL divergence.

<span id="section-4-6"></span>

### 4.6 Post-training Evaluation

To comprehensively evaluate the quality of instruction-tuned models, we adopted automatic benchmarks to assess model performance under both thinking and non-thinking modes. These benchmarks are categorized into several dimensions:


- **General Tasks**: We utilize benchmarks including MMLU-Redux [Gem24a], GPQA-Diamond [Rei24], C-Eval [Hua23], and LiveBench (2024-11-25) [Whi24]. For GPQA-Diamond, we sample 10 times for each query and report the averaged accuracy.


- **Alignment Tasks**: To evaluate how well the model aligns with human preferences, we employ a suite of specialized benchmarks. For instruction-following performance, we report the strict-prompt accuracy of IFEval [Zho23a]. To assess alignment with human preferences on general topics, we utilize Arena-Hard [Li24j] and AlignBench v1.1 [Liu23m]. For writing tasks, we rely on Creative Writing V3 [Pae24] and WritingBench [Wu25b] to evaluate the model's proficiency and creativity.


- **Math & Text Reasoning**: For evaluating mathematical and logical reasoning skills, we employ high-level math benchmarks including MATH-500 [Lig23], AIME'24 and AIME'25 [Aim25a], and text reasoning tasks including ZebraLogic [Lin25] and AutoLogi [Zhu25b]. For AIME problems, each year's questions include Part I and Part II, totaling 30 questions. For each question, we sample 64 times and take the average accuracy as the final score.


- **Agent & Coding**: To test the model's proficiency in coding and agent-based tasks, we use BFCL v3 [Yan24d], LiveCodeBench (v5, 2024.10-2025.02) [Jai24], and Codeforces Ratings from CodeElo [Qua25]. For BFCL, all Qwen3 models are evaluated using the FC format, and yarn was used to deploy the models to a context length of 64k for Multi-Turn evaluation. Some baselines are derived from the BFCL leaderboard, taking the higher scores between FC and Prompt formats. For models not reported on the leaderboard, the Prompt formats are evaluated. For LiveCodeBench, for the non-thinking mode, we use the officially recommended prompt, while for the thinking mode, we adjust the prompt template to allow the model to think more freely, by removing the restriction `You will not return anything except for the program`. To evaluate the performance gap between models and competitive programming experts, we use CodeForces to calculate Elo ratings. In our benchmark, each problem is solved by generating up to eight independent reasoning attempts.


- **Multilingual Tasks**: For multilingual capabilities, we evaluate four kinds of tasks: instruction following, knowledge, mathematics, and logical reasoning. Instruction following is assessed using Multi-IF [He24b], which focuses on 8 key languages. Knowledge assessment consisted of two types: regional knowledge evaluated through INCLUDE [Rom24], covering 44 languages, and general knowledge assessed with MMMLU [Ope24c] across 14 languages, excluding the unoptimized Yoruba language; for these two benchmarks, we sample only 10% of the original data to improve evaluation efficiency. The mathematics task employ MT-AIME2024 [Son25], encompassing 55 languages, and PolyMath [Wan25g], which includes 18 languages. Logical reasoning is evaluated using MlogiQA, covering 10 languages, sourced from [Zha24k].

<span id="table-10"></span>

![Table 10: Multilingual benchmarks and the included languages. The languages are identified in IETF language tags.](../../papers/qwen3/table-10.png)

**Table 10.** **Multilingual benchmarks and the included languages.** The languages are identified in IETF language tags.

<span id="table-11"></span>

![Table 11: Comparison among Qwen3-235B-A22B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-11.png)

**Table 11.** **Comparison among Qwen3-235B-A22B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-12"></span>

![Table 12: Comparison among Qwen3-235B-A22B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-12.png)

**Table 12.** **Comparison among Qwen3-235B-A22B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

For all Qwen3 models in the thinking mode, we utilize a sampling temperature of 0.6, a top-p value of 0.95, and a top-k value of 20. Additionally, for Creative Writing v3 and WritingBench, we apply a presence penalty of 1.5 to encourage the generation of more diverse content. For Qwen3 models in the non-thinking mode, we configure the sampling hyperparameters with temperature = 0.7, top-p = 0.8, top-k = 20, and presence penalty = 1.5. For both the thinking and non-thinking modes, we set the max output length to 32,768 tokens, except AIME'24 and AIME'25 where we extend this length to 38,912 tokens to provide sufficient thinking space.

**Summary of Evaluation Results** From the evaluation results, we summarize several key conclusions of the finalized Qwen3 models as follows:


1. Our flagship model, Qwen3-235B-A22B, demonstrates the state-of-the-art overall performance among open-source models in both the thinking and non-thinking modes, surpassing strong baselines such as DeepSeek-R1 and DeepSeek-V3. Qwen3-235B-A22B is also highly competitive to closed-source leading models, such as OpenAI-o1, Gemini2.5-Pro, and GPT-4o, showcasing its profound reasoning capabilities and comprehensive general abilities.


1. Our flagship dense model, Qwen3-32B, outperforms our previous strongest reasoning model, QwQ-32B, in most of the benchmarks, and performs comparably to the closed-source OpenAI-o3-mini, indicating its compelling reasoning capabilities. Qwen3-32B is also remarkably performant in the non-thinking mode and surpasses our previous flagship non-reasoning dense model, Qwen2.5-72B-Instruct.


1. Our lightweight models, including Qwen3-30B-A3B, Qwen3-14B, and other smaller dense ones, possess consistently superior performance to the open-source models with a close or larger amount of parameters, proving the success of our Strong-to-Weak Distillation approach.

The detailed results are as follows.

**Qwen3-235B-A22B** For our flagship model Qwen3-235B-A22B, we compare it with the leading reasoning and non-reasoning models. For the thinking mode, we take OpenAI-o1 [Ope24], DeepSeek-R1 [Guo25], Grok-3-Beta (Think) [Gro25], and Gemini2.5-Pro [Gem25] as the reasoning baselines. For the non-thinking mode, we take GPT-4o-2024-11-20 [Gpt24], DeepSeek-V3 [Dee24a], Qwen2.5-72B-Instruct [Yang24], and LLaMA-4-Maverick [Lla25] as the non-reasoning baselines. We present the evaluation results in [Table 11](#table-11) and [Table 12](#table-12).


1. From [Table 11](#table-11), with only 60% activated and 35% total parameters, Qwen3-235B-A22B (Thinking) outperforms DeepSeek-R1 on **17/23** the benchmarks, particularly on the reasoning-demanded tasks (e.g., mathematics, agent, and coding), demonstrating the state-of-the-art reasoning capabilities of Qwen3-235B-A22B among open-source models. Moreover, Qwen3-235B-A22B (Thinking) is also highly competitive to the closed-source OpenAI-o1, Grok-3-Beta (Think), and Gemini2.5-Pro, substantially narrowing the gap in the reasoning capabilities between open-source and close-source models.


1. From [Table 12](#table-12), Qwen3-235B-A22B (Non-thinking) exceeds the other leading open-source models, including DeepSeek-V3, LLaMA-4-Maverick, and our previous flagship model Qwen2.5-72B-Instruct, and also surpasses the closed-source GPT-4o-2024-11-20 in **18/23** the benchmarks, indicating its inherent strong capabilities even when not enhanced with the deliberate thinking process.

**Qwen3-32B** For our flagship dense model, Qwen3-32B, we take DeepSeek-R1-Distill-Llama-70B, OpenAI-o3-mini (medium), and our previous strongest reasoning model, QwQ-32B [Qwq25], as the baselines in the thinking mode. We also take GPT-4o-mini-2024-07-18, LLaMA-4-Scout, and our previous flagship model, Qwen2.5-72B-Instruct, as the baselines in the non-thinking mode. We present the evaluation results in [Table 13](#table-13) and [Table 14](#table-14).


1. From [Table 13](#table-13), Qwen3-32B (Thinking) outperforms QwQ-32B on **17/23** the benchmarks, making it the new state-of-the-art reasoning model at the sweet size of 32B. Moreover, Qwen3-32B (Thinking) also competes with the closed-source OpenAI-o3-mini (medium) with better alignment and multilingual performance.


1. From [Table 14](#table-14), Qwen3-32B (Non-thinking) exhibits superior performance to all the baselines on almost all the benchmarks. Particularly, Qwen3-32B (Non-thinking) performs on par with Qwen2.5-72B-Instruct on the general tasks with significant advantages on the alignment, multilingual, and reasoning-related tasks, again proving the fundamental improvements of Qwen3 over our previous Qwen2.5 series models.

<span id="table-13"></span>

![Table 13: Comparison among Qwen3-32B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-13.png)

**Table 13.** **Comparison among Qwen3-32B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-14"></span>

![Table 14: Comparison among Qwen3-32B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-14.png)

**Table 14.** **Comparison among Qwen3-32B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-15"></span>

![Table 15: Comparison among Qwen3-30B-A3B / Qwen3-14B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-15.png)

**Table 15.** **Comparison among Qwen3-30B-A3B / Qwen3-14B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-16"></span>

![Table 16: Comparison among Qwen3-30B-A3B / Qwen3-14B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-16.png)

**Table 16.** **Comparison among Qwen3-30B-A3B / Qwen3-14B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-17"></span>

![Table 17: Comparison among Qwen3-8B / Qwen3-4B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-17.png)

**Table 17.** **Comparison among Qwen3-8B / Qwen3-4B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-18"></span>

![Table 18: Comparison among Qwen3-8B / Qwen3-4B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-18.png)

**Table 18.** **Comparison among Qwen3-8B / Qwen3-4B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-19"></span>

![Table 19: Comparison among Qwen3-1.7B / Qwen3-0.6B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-19.png)

**Table 19.** **Comparison among Qwen3-1.7B / Qwen3-0.6B (Thinking) and other reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

<span id="table-20"></span>

![Table 20: Comparison among Qwen3-1.7B / Qwen3-0.6B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-20.png)

**Table 20.** **Comparison among Qwen3-1.7B / Qwen3-0.6B (Non-thinking) and other non-reasoning baselines. The highest and second-best scores are shown in bold and underlined, respectively.**

**Qwen3-30B-A3B & Qwen3-14B** For Qwen3-30B-A3B and Qwen3-14B, we compare them with DeepSeek-R1-Distill-Qwen-32B and QwQ-32B in the thinking mode, and Phi-4 [Abd24a], Gemma-3-27B-IT [Gem25a], and Qwen2.5-32B-Instruct in the non-thinking mode, respectively. We present the evaluation results in [Table 15](#table-15) and [Table 16](#table-16).


1. From [Table 15](#table-15), Qwen3-30B-A3B and Qwen3-14B (Thinking) are both highly competitive to QwQ-32B, especially on the reasoning-related benchmarks. It is noteworthy that Qwen3-30B-A3B achieves comparable performance to QwQ-32B with a smaller model size and less than **1/10** activated parameters, demonstrating the effectiveness of our Strong-to-Weak Distillation approach in endowing lightweight models with profound reasoning capabilities.


1. From [Table 16](#table-16), Qwen3-30B-A3B and Qwen3-14B (Non-thinking) surpass the non-reasoning baselines in most of the benchmarks. They exceed our previous Qwen2.5-32B-Instruct model with significantly fewer activated and total parameters, allowing for more efficient and cost-effective performance.

**Qwen3-8B / 4B / 1.7B / 0.6B** For Qwen3-8B and Qwen3-4B, we compare them with DeepSeek-R1-Distill-Qwen-14B and DeepSeek-R1-Distill-Qwen-32B in the thinking mode, and LLaMA-3.1-8B-Instruct [Dub24], Gemma-3-12B-IT [Gem25a], Qwen2.5-7B-Instruct, and Qwen2.5-14B-Instruct in the non-thinking mode, respectively. For Qwen3-1.7B and Qwen3-0.6B, we compare them with DeepSeek-R1-Distill-Qwen-1.5B and DeepSeek-R1-Distill-Llama-8B in the thinking mode, and Gemma-3-1B-IT, Phi-4-mini, Qwen2.5-1.5B-Instruct, and Qwen2.5-3B-Instruct in the non-thinking mode, respectively. We present the evaluation results of Qwen3-8B and Qwen3-4B in [Table 17](#table-17) and [Table 18](#table-18) and those of Qwen3-1.7B and Qwen3-0.6B in [Table 19](#table-19) and [Table 20](#table-20), respectively. Overall, these edge-side models exhibit impressive performance and outperform baselines even with more parameters, including our previous Qwen2.5 models, in either the thinking or the non-thinking mode. These results, once again, demonstrate the efficacy of our Strong-to-Weak Distillation approach, making it possible for us to build the lightweight Qwen3 models with remarkably reduced costs and efforts.

<span id="section-4-7"></span>

### 4.7 Discussion

**The Effectiveness of Thinking Budget** To verify that Qwen3 can enhance its intelligence level by leveraging an increased thinking budget, we adjust the allocated thinking budget on four benchmarks across Mathematics, Coding, and STEM domains. The resulting scaling curves are presented in [Figure 2](#figure-02), Qwen3 demonstrates scalable and smooth performance improvements correlated to the allocated thinking budget. Moreover, we observe that if we further extend the output length beyond 32K, the model's performance is expected to improve further in the future. We leave this exploration as future work.

<span id="figure-02"></span>

![Figure 2: Performance of Qwen3-235B-A22B with respect to the thinking budget.](../../papers/qwen3/figure-02.png)

**Figure 2.** Performance of Qwen3-235B-A22B with respect to the thinking budget.

**The Effectiveness and Efficiency of On-Policy Distillation** We evaluate the effectiveness and efficiency of on-policy distillation by comparing the performance and computational cost—measured in GPU hours—after undergoing distillation versus direct reinforcement learning, both starting from the same off-policy distilled 8B checkpoint. For simplicity, we focus solely on math and code-related queries in this comparison. The results, summarized in [Table 21](#table-21), show that distillation achieves significantly better performance than reinforcement learning while requiring approximately only $1/10$ of the GPU hours. Furthermore, distillation from teacher logits enables the student model to expand its exploration space and enhance its reasoning potential, as evidenced by the improved pass@64 scores on the AIME'24 and AIME'25 benchmarks after distillation, compared to the initial checkpoint. In contrast, reinforcement learning does not lead to any improvement in pass@64 scores. These observations highlight the advantages of leveraging a stronger teacher model in guiding student model learning.

<span id="table-21"></span>

![Table 21: Comparison of reinforcement learning and on-policy distillation on Qwen3-8B. Numbers in parentheses indicate pass@64 scores.](../../papers/qwen3/table-21.png)

**Table 21.** Comparison of reinforcement learning and on-policy distillation on Qwen3-8B. Numbers in parentheses indicate pass@64 scores.

**The Effects of Thinking Mode Fusion and General RL** To evaluate the effectiveness of Thinking Mode Fusion and General Reinforcement Learning (RL) during the post-training, we conduct evaluations on various stages of the Qwen-32B model. In addition to the datasets mentioned earlier, we introduce several in-house benchmarks to monitor other capabilities. These benchmarks include:


- **CounterFactQA**: Contains counterfactual questions where the model needs to identify that the questions are not factual and avoid generating hallucinatory answers.


- **LengthCtrl**: Includes creative writing tasks with length requirements; the final score is based on the difference between the generated content length and the target length.


- **ThinkFollow**: Involves multi-turn dialogues with randomly inserted `/think` and `/no_think` flags to test whether the model can correctly switch thinking modes based on user queries.


- **ToolUse**: Evaluates the stability of the model in single-turn, multi-turn, and multi-step tool calling processes. The score includes accuracy in intent recognition, format accuracy, and parameter accuracy during the tool calling process.

<span id="table-22"></span>

![Table 22: Performance of Qwen3-32B after Reasoning RL (Stage 2), Thinking Mode Fusion (Stage 3), and General RL (Stage 4). Benchmarks with * are in-house datasets.](../../papers/qwen3/table-22.png)

**Table 22.** Performance of Qwen3-32B after Reasoning RL (Stage 2), Thinking Mode Fusion (Stage 3), and General RL (Stage 4). Benchmarks with * are in-house datasets.

The results are shown in [Table 22](#table-22), where we can draw the following conclusions:


1. Stage 3 integrates the non-thinking mode into the model, which already possesses thinking capabilities after the first two stages of training. The ThinkFollow benchmark score of 88.7 indicates that the model has developed an initial ability to switch between modes, though it still occasionally makes errors. Stage 3 also enhances the model's general and instruction-following capabilities in thinking mode, with CounterFactQA improving by 10.9 points and LengthCtrl by 8.0 points.


1. Stage 4 further strengthens the model's general, instruction-following, and agent capabilities in both thinking and non-thinking modes. Notably, the ThinkFollow score improves to 98.9, ensuring accurate mode switching.


1. For Knowledge, STEM, Math, and Coding tasks, Thinking Mode Fusion and General RL do not bring significant improvements. In contrast, for challenging tasks like AIME'24 and LiveCodeBench, the performance in thinking mode actually decreases after these two training stages. We conjecture this degradation is due to the model being trained on a broader range of general tasks, which may compromise its specialized capabilities in handling complex problems. During the development of Qwen3, we choose to accept this performance trade-off to enhance the model's overall versatility.

<span id="section-5"></span>

## 5 Conclusion

In this technical report, we introduce Qwen3, the latest version of the Qwen series. Qwen3 features both thinking mode and non-thinking mode, allowing users to dynamically manage the number of tokens used for complex thinking tasks. The model was pre-trained on an extensive dataset containing 36 trillion tokens, enabling it to understand and generate text in 119 languages and dialects. Through a series of comprehensive evaluations, Qwen3 has shown strong performance across a range of standard benchmarks for both pre-trained and post-trained models, including tasks related to code generation, mathematics, reasoning, and agents.

In the near future, our research will focus on several key areas. We will continue to scale up pretraining by using data that is both higher in quality and more diverse in content. At the same time, we will work on improving model architecture and training methods for the purposes of effective compression, scaling to extremely long contexts, etc. In addition, we plan to increase computational resources for reinforcement learning, with a particular emphasis on agent-based RL systems that learn from environmental feedback. This will allow us to build agents capable of tackling complex tasks that require inference time scaling.

<span id="section-6"></span>

## 6 Authors

**Core Contributors:** An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu

**Contributors:** Bei Chen, Biao Sun, Bin Luo, Bin Zhang, Binghai Wang, Bowen Ping, Boyi Deng, Chang Si, Chaojie Yang, Chen Cheng, Chenfei Wu, Chengpeng Li, Chengyuan Li, Fan Hong, Guobin Zhao, Hang Zhang, Hangrui Hu, Hanyu Zhao, Hao Lin, Hao Xiang, Haoyan Huang, Hongkun Hao, Humen Zhong, Jialin Wang, Jiandong Jiang, Jianqiang Wan, Jianyuan Zeng, Jiawei Chen, Jie Zhang, Jin Xu, Jinkai Wang, Jinyang Zhang, Jinzheng He, Jun Tang, Kai Zhang, Ke Yi, Keming Lu, Keqin Chen, Langshi Chen, Le Jiang, Lei Zhang, Linjuan Wu, Man Yuan, Mingkun Yang, Minmin Sun, Mouxiang Chen, Na Ni, Nuo Chen, Peng Liu, Peng Wang, Peng Zhu, Pengcheng Zhang, Pengfei Wang, Qiaoyu Tang, Qing Fu, Qiuyue Wang, Rong Zhang, Rui Hu, Runji Lin, Shen Huang, Shuai Bai, Shutong Jiang, Sibo Song, Siqi Zhang, Song Chen, Tao He, Ting He, Tingfeng Hui, Wei Ding, Wei Liao, Wei Lin, Wei Zhang, Weijia Xu, Wenbin Ge, Wenmeng Zhou, Wenyuan Yu, Xianyan Jia, Xianzhong Shi, Xiaodong Deng, Xiaoming Huang, Xiaoyuan Li, Ximing Zhou, Xinyao Niu, Xipin Wei, Xuejing Liu, Yang Liu, Yang Yao, Yang Zhang, Yanpeng Li, Yantao Liu, Yidan Zhang, Yikai Zhu, Yiming Wang, Yiwen Hu, Yong Jiang, Yong Li, Yongan Yue, Yu Guan, Yuanzhi Zhu, Yunfei Chu, Yunlong Feng, Yuxin Zhou, Yuxuan Cai, Zeyao Ma, Zhaohai Li, Zheng Li, Zhengyang Tang, Zheren Fu, Zhi Li, Zhibo Yang, Zhifang Guo, Zhipeng Zhang, Zhiying Xu, Zhiyu Yin, Zhongshen Zeng, Zile Qiao, Ziye Meng, Zongmeng Zhang

<span id="section-a"></span>

## A Appendix

<span id="section-a-1"></span>

### A.1 Additional Evaluation Results
<span id="section-a-1-1"></span>

#### A.1.1 Long-Context Ability

<span id="table-23"></span>

![Table 23: Performance of Qwen3 Models on the RULER benchmark.](../../papers/qwen3/table-23.png)

**Table 23.** **Performance of Qwen3 Models on the RULER benchmark.**

For evaluating long-context processing capabilities, we report the results on the RULER benchmark [Hsi24a] in [Table 23](#table-23). To enable length extrapolation, we utilize YARN [Pen23] with a `scaling_factor=4`. In thinking mode, we set the thinking budget to 8192 tokens to mitigate overly verbose reasoning on the extremely long inputs.

The results show that:


1. In non-thinking mode, Qwen3 outperforms Qwen2.5 models of a similar size in long-context processing tasks.


1. In thinking mode, the model's performance slightly degrades. We hypothesize that the thinking content does not provide significant benefits for these retrieval tasks, which do not rely on reasoning and may instead interfere with the retrieval process. We are committed to enhancing the long-context capability in the thinking mode in future versions.

<span id="section-a-1-2"></span>

#### A.1.2 Multilingual Ability
[Table 24](#table-24)-[Table 35](#table-35) presents the detailed benchmark scores across various languages, including Spanish, French, Portuguese, Italian, Arabic, Japanese, Korean, Indonesian, Russian, Vietnamese, German, and Thai. The results of these tables demonstrate that the Qwen3 series models achieve competitive performance across all evaluated benchmarks, showcasing their strong multilingual capabilities.

To evaluate the performance of Qwen3 across a broader range of languages, we utilize Belebele [Ban23], a benchmark for natural language understanding. We conduct evaluations on 80 supported languages from the benchmark, excluding 42 unoptimized languages, as shown in [Table 36](#table-36) (organized by language family). The performance comparison between Qwen3 and other baseline models on the Belebele benchmark is presented in [Table 37](#table-37). The results show that Qwen3 achieves comparable performance to similarly-sized Gemma models while outperforming Qwen2.5 significantly.

<span id="table-24"></span>

![Table 24: Benchmark scores for language: Spanish (es). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-24.png)

**Table 24.** **Benchmark scores for language: Spanish (es)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-25"></span>

![Table 25: Benchmark scores for language: French (fr). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-25.png)

**Table 25.** **Benchmark scores for language: French (fr)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-26"></span>

![Table 26: Benchmark scores for language: Portuguese (pt). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-26.png)

**Table 26.** **Benchmark scores for language: Portuguese (pt)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-27"></span>

![Table 27: Benchmark scores for language: Italian (it). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-27.png)

**Table 27.** **Benchmark scores for language: Italian (it)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-28"></span>

![Table 28: Benchmark scores for language: Arabic (ar). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-28.png)

**Table 28.** **Benchmark scores for language: Arabic (ar)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-29"></span>

![Table 29: Benchmark scores for language: Japanese (ja). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-29.png)

**Table 29.** **Benchmark scores for language: Japanese (ja)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-30"></span>

![Table 30: Benchmark scores for language: Korean (ko). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-30.png)

**Table 30.** **Benchmark scores for language: Korean (ko)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-31"></span>

![Table 31: Benchmark scores for language: Indonesian (id). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-31.png)

**Table 31.** **Benchmark scores for language: Indonesian (id)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-32"></span>

![Table 32: Benchmark scores for language: Russian (ru). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-32.png)

**Table 32.** **Benchmark scores for language: Russian (ru)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-33"></span>

![Table 33: Benchmark scores for language: Vietnamese (vi). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-33.png)

**Table 33.** **Benchmark scores for language: Vietnamese (vi)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-34"></span>

![Table 34: Benchmark scores for language: German (de). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-34.png)

**Table 34.** **Benchmark scores for language: German (de)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-35"></span>

![Table 35: Benchmark scores for language: Thai (th). The highest and second-best scores are shown in bold and underlined, respectively.](../../papers/qwen3/table-35.png)

**Table 35.** **Benchmark scores for language: Thai (th)**. The highest and second-best scores are shown in **bold** and underlined, respectively.

<span id="table-36"></span>

![Table 36: Language families and language codes supported by Qwen3 in Belebele Benchmark](../../papers/qwen3/table-36.png)

**Table 36.** Language families and language codes supported by Qwen3 in Belebele Benchmark

<span id="table-37"></span>

![Table 37: Comparison of Belebele Benchmark performance between Qwen3 and other baseline models. Scores are highlighted with the highest in bold and the second-best underlined.](../../papers/qwen3/table-37.png)

**Table 37.** **Comparison of Belebele Benchmark performance between Qwen3 and other baseline models.** Scores are highlighted with the highest in **bold** and the second-best underlined.

[+1]: [https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json](https://huggingface.co/Qwen/Qwen3-32B/blob/main/tokenizer_config.json)
