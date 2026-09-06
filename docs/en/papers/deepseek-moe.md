---
title: 'DeepSeekMoE'
createTime: 2026/09/06 19:41:12
permalink: /en/papers/deepseek-moe/
---

> [Damai Dai](https://dblp.org/pid/199/2097.html) [+internship], [Chengqi Deng](https://dblp.org/pid/255/4939.html), [Chenggang Zhao](https://dblp.org/pid/254/2607.html) [+internship], [R.X. Xu](https://dblp.org/pid/267/5291.html), [Huazuo Gao](https://dblp.org/pid/366/3356.html), [Deli Chen](https://dblp.org/pid/50/2637.html), [Jiashi Li](https://dblp.org/pid/241/9364.html), [Wangding Zeng](https://dblp.org/pid/315/5319.html), [Xingkai Yu](https://dblp.org/pid/257/4432.html) [+internship], [Y. Wu](https://dblp.org/pid/22/0-24.html), [Zhenda Xie](https://dblp.org/pid/239/8676.html), [Y.K. Li](https://dblp.org/pid/16/8783.html), [Panpan Huang](https://dblp.org/pid/19/6338.html), [Fuli Luo](https://dblp.org/pid/220/4216.html), [Chong Ruan](https://dblp.org/pid/159/9956.html), [Zhifang Sui](https://dblp.org/pid/22/5834.html), and [Wenfeng Liang](https://dblp.org/pid/59/9456.html). First submitted to arXiv on January 11, 2024; current version v1. Later published in the *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1280–1297, August 2024. [DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066v1). <a href="/paper/deepseek-moe.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [ACL 2024](https://aclanthology.org/2024.acl-long.70/). [DOI](https://doi.org/10.18653/v1/2024.acl-long.70). [TeX source](https://export.arxiv.org/e-print/2401.06066v1). [Code and models](https://github.com/deepseek-ai/DeepSeek-MoE). The original PDF remains authoritative for the exact print layout and bibliography.

[+internship]: Contribution during internship at DeepSeek-AI.

## Abstract

In the era of large language models, Mixture-of-Experts (MoE) is a promising architecture for managing computational costs when scaling up model parameters. However, conventional MoE architectures like GShard, which activate the top-$K$ out of $N$ experts, face challenges in ensuring expert specialization, i.e. each expert acquires non-overlapping and focused knowledge. In response, we propose the **DeepSeekMoE** architecture towards ultimate expert specialization. It involves two principal strategies: (1) finely segmenting the experts into $mN$ ones and activating $mK$ from them, allowing for a more flexible combination of activated experts; (2) isolating $K_s$ experts as shared ones, aiming at capturing common knowledge and mitigating redundancy in routed experts. Starting from a modest scale with 2B parameters, we demonstrate that DeepSeekMoE 2B achieves comparable performance with GShard 2.9B, which has 1.5$\times$ expert parameters and computation. In addition, DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with the same number of total parameters, which set the upper bound of MoE models. Subsequently, we scale up DeepSeekMoE to 16B parameters and show that it achieves comparable performance with LLaMA2 7B, with only about 40% of computations. Further, our preliminary efforts to scale up DeepSeekMoE to 145B parameters consistently validate its substantial advantages over the GShard architecture, and show its performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

<span id="figure-01"></span>

![Figure 1. Comparison between DeepSeekMoE 16B and open source models on the Open LLM Leaderboard. The red dashed line is linearly fitted from data points of all models except DeepSeekMoE 16B. DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B, which has approximately 2.5 times the activated parameters.](../../papers/deepseek-moe/figure-01.png)

**Figure 1.** Comparison between DeepSeekMoE 16B and open source models on the Open LLM Leaderboard. The red dashed line is linearly fitted from data points of all models except DeepSeekMoE 16B. DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B, which has approximately 2.5 times the activated parameters.

<span id="section-1"></span>

## 1 Introduction

Recent research and practices have empirically demonstrated that, with sufficient training data available, scaling language models with increased parameters and computational budgets can yield remarkably stronger models [Bro20, Ope23, Tou23, Hof22]. It is imperative to acknowledge, however, that the endeavor to scale models to an extremely large scale is also associated with exceedingly high computational costs. Considering the substantial costs, the Mixture-of-Experts (MoE) architecture [Jac91, Jor94, Sha17] has emerged as a popular solution. It can enable parameter scaling, while concurrently keeping computational costs at a modest level. Recent applications of MoE architectures in Transformers [Vas17d] have yielded successful attempts at scaling language models to a substantial size [Fed22, Lep20, Du22, Zop22], accompanied with remarkable performance. These achievements underscore the considerable potential and promise of MoE language models.

Despite the promising potential of MoE architectures, existing MoE architectures potentially suffer from issues of knowledge hybridity and knowledge redundancy, which limit the expert specialization, i.e., each expert acquires non-overlapping and focused knowledge. Conventional MoE architectures substitute the Feed-Forward Networks (FFNs) in a Transformer with MoE layers. Each MoE layer consists of multiple experts, with each structurally identical to a standard FFN, and each token is assigned to one [Fed22] or two [Lep20] experts. This architecture manifests two potential issues: (1) **Knowledge Hybridity**: existing MoE practices often employ a limited number of experts (e.g., 8 or 16), and thus tokens assigned to a specific expert will be likely to cover diverse knowledge. Consequently, the designated expert will intend to assemble vastly different types of knowledge in its parameters, which are hard to utilize simultaneously. (2) **Knowledge Redundancy**: tokens assigned to different experts may require common knowledge. As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby leading to redundancy in expert parameters. These issues collectively hinder the expert specialization in existing MoE practices, preventing them from reaching the theoretical upper-bound performance of MoE models.

In response to the aforementioned issues, we introduce **DeepSeekMoE**, an innovative MoE architecture specifically designed towards ultimate expert specialization. Our architecture involves two principal strategies: (1) **Fine-Grained Expert Segmentation:** while maintaining the number of parameters constant, we segment the experts into a finer grain by splitting the FFN intermediate hidden dimension. Correspondingly, keeping a constant computational cost, we also activate more fine-grained experts to enable a more flexible and adaptable combination of activated experts. Fine-grained expert segmentation allows diverse knowledge to be decomposed more finely and be learned more precisely into different experts, where each expert will retain a higher level of specialization. In addition, the increased flexibility in combining activated experts also contributes to a more accurate and targeted knowledge acquisition. (2) **Shared Expert Isolation:** we isolate certain experts to serve as shared experts that are always activated, aiming at capturing and consolidating common knowledge across varying contexts. Through compressing common knowledge into these shared experts, redundancy among other routed experts will be mitigated. This can enhance the parameter efficiency and ensure that each routed expert retains specialized by focusing on distinctive aspects. These architectural innovations in DeepSeekMoE offer opportunities to train a parameter-efficient MoE language model where each expert is highly specialized.

Starting from a modest scale with 2B parameters, we validate the advantages of the DeepSeekMoE architecture. We conduct evaluations on 12 zero-shot or few-shot benchmarks spanning diverse tasks. Empirical results indicate that DeepSeekMoE 2B surpasses GShard 2B [Lep20] by a substantial margin, and even matches GShard 2.9B, a larger MoE model with 1.5$\times$ expert parameters and computation. Remarkably, we find that DeepSeekMoE 2B nearly approaches the performance of its dense counterpart with an equivalent number of parameters, which sets the strict upper bound of MoE language models. In pursuit of deeper insights, we conduct elaborate ablation studies and analysis on the expert specialization for DeepSeekMoE. These studies validate the effectiveness of fine-grained expert segmentation and shared expert isolation, and provide empirical evidence supporting the assertion that DeepSeekMoE can achieve a high level of expert specialization.

Leveraging our architecture, we subsequently scale up the model parameters to 16B and train DeepSeekMoE 16B on a large-scale corpus with 2T tokens. Evaluation results reveal that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B [Dee24e], a dense model trained on the same 2T corpus. We also compare DeepSeekMoE with open source models and the evaluations demonstrate that DeepSeekMoE 16B consistently outperforms models with a similar number of activated parameters by a large margin, and achieves comparable performance with LLaMA2 7B [Tou23a], which has approximately 2.5 times the activated parameters. [Figure 1](#figure-01) demonstrates the evaluation results on the Open LLM Leaderboard [+1]. Additionally, we conduct supervised fine-tuning (SFT) for alignment, transforming the model into a chat model. Evaluation results show that DeepSeekMoE Chat 16B also achieves comparable performance with DeepSeek Chat 7B and LLaMA2 SFT 7B in the chat setting. Encouraged by these results, we further undertake a preliminary endeavor to scale up DeepSeekMoE to 145B. The experimental results still validate its substantial advantages over the GShard architecture consistently. In addition, it shows performance comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

Our contributions are summarized as follows:

- **Architectural Innovation.** We introduce DeepSeekMoE, an innovative MoE architecture aiming at achieving ultimate expert specialization, which employs two principal strategies of fine-grained expert segmentation and shared expert isolation.

- **Empirical Validation.** We conduct extensive experiments to empirically validate the effectiveness of the DeepSeekMoE architecture. Experimental results validate the high level of expert specialization in DeepSeekMoE 2B, and indicate that DeepSeekMoE 2B can nearly approach the upper bound performance for MoE models

- **Scalability.** We scale up DeepSeekMoE to train a 16B model and show that with only about 40% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B and LLaMA2 7B. We also undertake a preliminary endeavor to scale up DeepSeekMoE to 145B, highlighting its consistent advantages over the GShard architecture and showing a comparable performance with DeepSeek 67B.

- **Alignment for MoE.** We successfully perform supervised fine-tuning on DeepSeekMoE 16B to create an aligned chat model, showcasing the adaptability and versatility of DeepSeekMoE 16B.

- **Public Release.** In the spirit of open research, we release the model checkpoint of DeepSeekMoE 16B to the public. Notably, this model can be deployed on a single GPU with 40GB of memory without the need for quantization.

<span id="section-2"></span>

## 2 Preliminaries: Mixture-of-Experts for Transformers

We first introduce a generic MoE architecture commonly used in Transformer language models. A standard Transformer language model is constructed by stacking $L$ layers of standard Transformer blocks, where each block can be represented as follows:

$$
\begin{aligned}
    \mathbf{u}_{1:T}^{l} &= \operatorname{Self-Att}\left( \mathbf{h}_{1:T}^{l-1} \right) + \mathbf{h}_{1:T}^{l-1}, \\
    \mathbf{h}_{t}^{l} &= \operatorname{FFN}\left( \mathbf{u}_{t}^{l} \right) + \mathbf{u}_{t}^{l},
\end{aligned}
$$

where $T$ denotes the sequence length, $\operatorname{Self-Att}(\cdot)$ denotes the self-attention module, $\operatorname{FFN}(\cdot)$ denotes the Feed-Forward Network (FFN), $\mathbf{u}_{1:T}^{l} \in \mathbb{R}^{T \times d}$ are the hidden states of all tokens after the $l$-th attention module, and $\mathbf{h}_{t}^{l} \in \mathbb{R}^{d}$ is the output hidden state of the $t$-th token after the $l$-th Transformer block. For brevity, we omit the layer normalization in the above formulations.

A typical practice to construct an MoE language model usually substitutes FFNs in a Transformer with MoE layers at specified intervals [Fed22, Lep20, Du22, Zop22]. An MoE layer is composed of multiple experts, where each expert is structurally identical to a standard FFN. Then, each token will be assigned to one [Fed22] or two [Lep20] experts. If the $l$-th FFN is substituted with an MoE layer, the computation for its output hidden state $\mathbf{h}_{t}^{l}$ is expressed as:

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{N} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leqslant j \leqslant N \}, K), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right),
\end{aligned}
$$

where $N$ denotes the total number of experts, $\operatorname{FFN}_{i}(\cdot)$ is the $i$-th expert FFN, $g_{i,t}$ denotes the gate value for the $i$-th expert, $s_{i,t}$ denotes the token-to-expert affinity, $\operatorname{Topk}(\cdot, K)$ denotes the set comprising $K$ highest affinity scores among those calculated for the $t$-th token and all $N$ experts, and $\mathbf{e}_{i}^{l}$ is the centroid of the $i$-th expert in the $l$-th layer. Note that $g_{i,t}$ is sparse, indicating that only $K$ out of $N$ gate values are nonzero. This sparsity property ensures computational efficiency within an MoE layer, i.e., each token will be assigned to and computed in only $K$ experts. Also, in the above formulations, we omit the layer normalization operation for brevity.

<span id="figure-02"></span>

![Figure 2. Illustration of DeepSeekMoE. Subfigure (a) showcases an MoE layer with the conventional top-2 routing strategy. Subfigure (b) illustrates the fine-grained expert segmentation strategy. Subsequently, subfigure (c) demonstrates the integration of the shared expert isolation strategy, constituting the complete DeepSeekMoE architecture. It is noteworthy that across these three architectures, the number of expert parameters and computational costs remain constant.](../../papers/deepseek-moe/figure-02.png)

**Figure 2.** Illustration of DeepSeekMoE. Subfigure (a) showcases an MoE layer with the conventional top-2 routing strategy. Subfigure (b) illustrates the fine-grained expert segmentation strategy. Subsequently, subfigure (c) demonstrates the integration of the shared expert isolation strategy, constituting the complete DeepSeekMoE architecture. It is noteworthy that across these three architectures, the number of expert parameters and computational costs remain constant.

<span id="section-3"></span>

## 3 DeepSeekMoE Architecture

On top of the generic MoE architecture outlined in [Section 2](#section-2), we introduce DeepSeekMoE, which is specifically designed to exploit the potential of expert specialization. As illustrated in [Figure 2](#figure-02), our architecture incorporates two principal strategies: fine-grained expert segmentation and shared expert isolation. Both of these strategies are designed to elevate the level of expert specialization.

<span id="section-3-1"></span>

### 3.1 Fine-Grained Expert Segmentation

In scenarios where the number of experts is limited, tokens assigned to a particular expert will be more likely to cover diverse types of knowledge. As a consequence, the designated expert will intend to learn vastly different types of knowledge in its parameters, and they are hard to be simultaneously utilized. However, if each token can be routed to more experts, diverse knowledge will gain the potential to be decomposed and learned in different experts respectively. In this context, each expert can still retain a high level of expert specialization, contributing to a more focused knowledge distribution across experts.

In pursuit of the goal, while maintaining a consistent number of expert parameters and computational cost, we segment the experts with a finer grain. The finer expert segmentation enables a more flexible and adaptable combination of activated experts. To be specific, on top of a typical MoE architecture shown in [Figure 2](#figure-02)(a), we segment each expert FFN into $m$ smaller experts by reducing the FFN intermediate hidden dimension to $\frac{1}{m}$ times its original size. Since each expert becomes smaller, in response, we also increase the number of activated experts to $m$ times to keep the same computation cost, as illustrated in [Figure 2](#figure-02)(b). With the fine-grained expert segmentation, the output of an MoE layer can be expressed as:

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | 1 \leqslant j \leqslant mN \}, mK), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right),
\end{aligned}
$$

where the total number of expert parameters is equal to $N$ times the number of parameters in a standard FFN, and $mN$ denotes the total number of fine-grained experts. With the fine-grained expert segmentation strategy, the number of nonzero gates will also increases to $mK$.

From a combinatorial perspective, the fine-grained expert segmentation strategy substantially enhances the combinatorial flexibility of activated experts. As an illustrative example, we consider the case where $N=16$. A typical top-2 routing strategy can yield $\binom{16}{2}=120$ possible combinations. By contrast, if each expert is split into $4$ smaller experts, the fine-grained routing strategy can yield $\binom{64}{8}=4,426,165,368$ potential combinations. The surge in combinatorial flexibility enhances the potential for achieving more accurate and targeted knowledge acquisition.

<span id="section-3-2"></span>

### 3.2 Shared Expert Isolation

With a conventional routing strategy, tokens assigned to different experts may necessitate some common knowledge or information. As a result, multiple experts may converge in acquiring shared knowledge in their respective parameters, thereby resulting in redundancy in expert parameters. However, if there are shared experts dedicated to capturing and consolidating common knowledge across varying contexts, the parameter redundancy among other routed experts will be alleviated. This alleviation of redundancy will contribute to a more parameter-efficient model with more specialized experts.

Towards this objective, in addition to the fine-grained expert segmentation strategy, we further isolate $K_{s}$ experts to serve as shared experts. Regardless of the router module, each token will be deterministically assigned to these shared experts. In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by $K_{s}$, as depicted in [Figure 2](#figure-02)(c). With the shared expert isolation strategy integrated, an MoE layer in the complete DeepSeekMoE architecture is formulated as follows:

$$
\begin{aligned}
\mathbf{h}_{t}^{l} & = \sum_{i=1}^{K_{s}} {\operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} + \sum_{i=K_{s} + 1}^{mN} \left( {g_{i,t} \operatorname{FFN}_{i}\left( \mathbf{u}_{t}^{l} \right)} \right) + \mathbf{u}_{t}^{l}, \\
g_{i,t} & = \begin{cases}
s_{i,t}, & s_{i,t} \in \operatorname{Topk} (\{ s_{j, t} | K_{s} + 1 \leqslant j \leqslant mN \}, mK - K_{s}), \\
0, & \text{otherwise},
\end{cases} \\
s_{i,t} & = \operatorname{Softmax}_i \left( {\mathbf{u}_{t}^{l}}^\top \mathbf{e}_{i}^{l} \right).
\end{aligned}
$$

Finally, in DeepSeekMoE, the number of shared expert is $K_{s}$, the total number of routed experts is $mN - K_{s}$, and the number of nonzero gates is $mK - K_{s}$.

It is worth noting that the prototype of shared expert isolation can be credited to [Raj22]. The key distinction lies in the fact that they derive this strategy from an engineering perspective, while we approach it from an algorithmic standpoint.

<span id="section-3-3"></span>

### 3.3 Load Balance Consideration

Automatically learned routing strategies may encounter the issue of load imbalance, which manifests two notable defects. Firstly, there is a risk of routing collapse [Sha17], i.e., the model always selects only a few experts, preventing other experts from sufficient training. Secondly, if experts are distributed across multiple devices, load imbalance can exacerbate computation bottlenecks.

**Expert-Level Balance Loss.** In order to mitigate the risk of routing collapse, we also employ an expert-level balance loss. The computation of the balance loss is as follows:

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{ExpBal}} & = \alpha_1 \sum_{i=1}^{N^{\prime}}{f_i P_i}, \\
    f_i & = \frac{N^{\prime}}{K^{\prime}T} \sum_{t=1}^{T}{ \mathbb{1}( \text{Token } t \text{ selects Expert } i )}, \\
    P_i & = \frac{1}{T} \sum_{t=1}^{T}{s_{i,t}},
\end{aligned}
$$

where $\alpha_1$ is a hyper-parameter called expert-level balance factor, $N^{\prime}$ is equal to $(mN - K_s)$ and $K^{\prime}$ is equal to $(mK - K_s)$ for brevity. $\mathbb{1}(\cdot)$ denotes the indicator function.

**Device-Level Balance Loss.** In addition to the expert-level balance loss, we introduce a device-level balance loss. When aiming to alleviate computation bottlenecks, it becomes unnecessary to enforce strict balance constraints at the expert level, because excessive constraints on load balance will compromise model performance. Instead, our primary objective is to ensure balanced computation across the devices. If we partition all routed experts into $D$ groups $\{\mathcal{E}_1, \mathcal{E}_2, ..., \mathcal{E}_D \}$, and deploy each group on a single device, the device-level balance loss is computed as follows:

$$
\begin{aligned}
    \mathcal{L}_{\mathrm{DevBal}} & = \alpha_{2} \sum_{i=1}^{D}{f_i^{\prime} P_i^{\prime}}, \\
    f_i^{\prime} & = \frac{1}{|\mathcal{E}_i|} \sum_{j \in \mathcal{E}_i}{ f_j }, \\
    P_i^{\prime} & = \sum_{j \in \mathcal{E}_i}{ P_j },
\end{aligned}
$$

where $\alpha_{2}$ is a hyper-parameter called device-level balance factor. In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor to promote balanced computation across the devices.

<span id="section-4"></span>

## 4 Validation Experiments

<span id="section-4-1"></span>

### 4.1 Experimental Setup

<span id="section-4-1-1"></span>

#### 4.1.1 Training Data and Tokenization

Our training data is sampled from a large-scale multilingual corpus created by DeepSeek-AI. The corpus primarily focuses on English and Chinese but also encompasses other languages. It is derived from diverse sources, including web text, mathematical material, coding scripts, published literature, and various other textual materials. For the purpose of validation experiments, we sample a subset containing 100B tokens from the corpus to train our models. For tokenization, we utilize the HuggingFace Tokenizer [+2] tools to train byte pair encoding (BPE) [Sen16] tokenizers on a smaller subset of the training corpus. In the validation experiments, we prepare a tokenizer with a vocabulary size of 8K, and the vocabulary size will be scaled up when training larger models.

<span id="section-4-1-2"></span>

#### 4.1.2 Infrastructures

We conduct experiments based on HAI-LLM [Hig23], an efficient and light-weight training framework which integrates multiple parallelism strategies, including tensor parallelism [Sho19, Nar21, Kor22], ZeRO data parallelism [Raj20], PipeDream pipeline parallelism [Har18], and more specifically, expert parallelism [Lep20] by combining data and tensor parallelism. In order to optimize performance, we develop GPU kernels with CUDA and Triton [Til19] for gating algorithms and fusing computations across linear layers in different experts.

All experiments are carried out on clusters equipped with NVIDIA A100 or H800 GPUs. Each node in the A100 cluster contains 8 GPUs connected pairwise via the NVLink bridge. The H800 cluster also features 8 GPUs per node, interconnected using NVLink and NVSwitch within nodes. For both A100 and H800 clusters, InfiniBand interconnects are utilized to facilitate communication across nodes.

<span id="section-4-1-3"></span>

#### 4.1.3 Hyper-Parameters

**Model Settings.** In the validation experiments, we set the number of Transformer layers to 9 and the hidden dimension to 1280. We employ the multi-head attention mechanism with a total of 10 attention heads, where each head has a dimension of 128. For initialization, all learnable parameters are randomly initialized with a standard deviation of 0.006. We substitute all FFNs with MoE layers, and ensure that the total number of expert parameters equals 16 times that of a standard FFN. Additionally, we keep the activated expert parameters, including shared expert parameters and activated routed expert parameters, as 2 times that of a standard FFN. Under this configuration, each MoE model has approximately 2B total parameters, with the number of activated parameters around 0.3B.

**Training Settings.** We employ the AdamW optimizer [Los17] with hyper-parameters set to $\beta_1=0.9$, $\beta_2=0.95$, and $\mathrm{weight\_decay}=0.1$. The learning rate is scheduled using a warmup-and-step-decay strategy. Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate is multiplied by 0.316 at 80% of the training steps, and again by 0.316 at 90% of the training steps. The maximum learning rate for validation experiments is set to $1.08 \times 10^{-3}$, and the gradient clipping norm is set to 1.0. The batch size is set to 2K, and with a maximum sequence length of 2K, each training batch contains 4M tokens. Correspondingly, the total number of training steps is set to 25,000 to achieve 100B training tokens. Due to the abundance of training data, we do not use dropout during training. Given the relatively small model size, all parameters, including expert parameters, are deployed on a single GPU device to avoid unbalanced computation. Correspondingly, we do not drop any tokens during training and do not employ the device-level balance loss. In order to prevent routing collapse, we set an expert-level balance factor of 0.01.

For readability, we also present an overview table of hyper-parameters for DeepSeekMoE across different sizes in [Section 10](#section-10).

<span id="section-4-1-4"></span>

#### 4.1.4 Evaluation Benchmarks

We conduct evaluations on a wide range of benchmarks covering various types of tasks. We list the benchmarks as follows.

**Language Modeling.** For language modeling, we evaluate the models on the test set of Pile [Gao20], and the evaluation metric is the cross-entropy loss.

**Language Understanding and Reasoning.** For language understanding and reasoning, we consider HellaSwag [Zel19], PIQA [Bis20], ARC-challenge and ARC-easy [Cla18]. The evaluation metric for these tasks is accuracy.

**Reading Comprehension.** For reading comprehension, we use RACE-high and RACE-middle [Lai17], and the evaluation metric is accuracy.

**Code Generation.** For code generation, we evaluate the models on HumanEval [Che21e] and MBPP [Aus21b]. The evaluation metric is Pass@1, which represents the pass rate for only one generation attempt.

**Closed-Book Question Answering.** For closed-book question answering, we consider TriviaQA [Jos17] and NaturalQuestions [Kwi19a]. The evaluation metric is the Exactly Matching (EM) rate.

<span id="table-01"></span>

![Table 1. Evaluation results for validation experiments. Bold font indicates the best. Compared with other MoE architectures, DeepSeekMoE exhibits a substantial performance advantage.](../../papers/deepseek-moe/table-01.png)

**Table 1.** Evaluation results for validation experiments. **Bold** font indicates the best. Compared with other MoE architectures, DeepSeekMoE exhibits a substantial performance advantage.

<span id="section-4-2"></span>

### 4.2 Evaluations

**Baselines.** Including DeepSeekMoE, we compare five models for validation experiments. **Dense** denotes a standard dense Transformer language model with 0.2B total parameters. **Hash Layer** [Rol21] is an MoE architecture based on top-1 hash routing, with 2.0B total parameters and 0.2B activated parameters, aligned with the dense baseline. **Switch Transformer** [Fed22] is another well-known MoE architecture based on top-1 learnable routing, with total parameters and activated parameters the same as Hash Layer. **GShard** [Lep20] employs a top-2 learnable routing strategy, with 2.0B total parameters and 0.3B activated parameters since one more expert is activated compared to top-1 routing methods. **DeepSeekMoE** has 1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN. Including DeepSeekMoE, all compared models share the same training corpus and training hyper-parameters. All compared MoE models have the same number of total parameters, and GShard has the same number of activated parameters as DeepSeekMoE.

**Results.** We present the evaluation results in [Table 1](#table-01). For all demonstrated models, we report the final evaluation results after training on 100B tokens. From the table, we make the following observations: (1) With sparse architectures and more total parameters, Hash Layer and Switch Transformer achieve significantly stronger performance than the dense baseline with the same number of activated parameters. (2) Compared with Hash Layer and Switch Transformer, GShard has more activated parameters and achieves slightly better performance than Switch Transformer. (3) With the same number of total parameters and activated parameters, DeepSeekMoE demonstrates overwhelming advantages over GShard. These results showcase the superiority of our DeepSeekMoE architecture within the existing landscape of MoE architectures.

<span id="table-02"></span>

![Table 2. Comparisons among DeepSeekMoE, larger GShard models, and larger dense models. In the line of '# Experts', $a$ + $b$ denotes $a$ shared experts and $b$ routed experts. In the line of '# Activated Experts', $a$ + $b$ denotes $a$ activated shared experts and $b$ activated routed experts. DeepSeekMoE achieves comparable performance with a GShard model containing 1.5 times expert parameters and computation. In addition, DeepSeekMoE nearly approaches the performance of a dense model with 16 times FFN parameters, which sets the upper bound for MoE models in terms of the model capacity.](../../papers/deepseek-moe/table-02.png)

**Table 2.** Comparisons among DeepSeekMoE, larger GShard models, and larger dense models. In the line of "# Experts", $a$ + $b$ denotes $a$ shared experts and $b$ routed experts. In the line of "# Activated Experts", $a$ + $b$ denotes $a$ activated shared experts and $b$ activated routed experts. DeepSeekMoE achieves comparable performance with a GShard model containing 1.5 times expert parameters and computation. In addition, DeepSeekMoE nearly approaches the performance of a dense model with 16 times FFN parameters, which sets the upper bound for MoE models in terms of the model capacity.

<span id="section-4-3"></span>

### 4.3 DeepSeekMoE Aligns Closely with the upper bound of MoE Models

We have demonstrated that DeepSeekMoE outperforms the dense baseline and other MoE architectures. In order to provide a more precise understanding of the performance of DeepSeekMoE, we compare it with larger baselines with more total parameters or activated parameters. The comparisons enable us to estimate the required model size of GShard or dense baselines to achieve equivalent performance to DeepSeekMoE.

**Comparison with GShard$\times 1.5$.** [Table 2](#table-02) shows the comparison between DeepSeekMoE and a larger GShard model with 1.5 times the expert size, which results in 1.5 times both expert parameters and expert computation. Overall, we observe that DeepSeekMoE achieves comparable performance with GShard$\times 1.5$, underscoring the significant advantage inherent in the DeepSeekMoE architecture. In addition to the comparison with GShard$\times 1.5$, we also show the comparison with GShard$\times 1.2$ in [Section 11](#section-11).

Furthermore, we increase the number of total parameters of DeepSeekMoE to 13.3B and compare it with GShard$\times 1.2$ and GShard$\times 1.5$ with 15.9B and 19.8B total parameters, respectively. We find that at a larger scale, DeepSeekMoE can even outperform GShard$\times 1.5$ distinctly. These results are also provided in [Section 11](#section-11).

**Comparison with Dense$\times 16$.** [Table 2](#table-02) also shows the comparison between DeepSeekMoE and larger dense models. For a fair comparison, we do not use the widely used ratio (1:2) between the attention and FFN parameters. Instead, we configure 16 shared experts where each expert has the same number of parameters as a standard FFN. This architecture mimics a dense model with 16 times standard FFN parameters. From the table, we find that DeepSeekMoE nearly approaches the performance of Dense$\times 16$, which sets the strict upper bound of MoE models in terms of the model capacity. These results suggest that, ***at least at the scale of about 2B parameters and 100B training tokens, the performance of DeepSeekMoE aligns closely with the theoretical upper bound of MoE models***. Also, we provide additional comparisons with Dense$\times 4$ in [Section 11](#section-11).

<span id="figure-03"></span>

![Figure 3. Ablation studies for DeepSeekMoE. The performance is normalized by the best performance for clarity in presentation. All compared models have the same number of parameters and activated parameters. We can find that fine-grained expert segmentation and shared expert isolation both contribute to stronger overall performance.](../../papers/deepseek-moe/figure-03.png)

**Figure 3.** Ablation studies for DeepSeekMoE. The performance is normalized by the best performance for clarity in presentation. All compared models have the same number of parameters and activated parameters. We can find that fine-grained expert segmentation and shared expert isolation both contribute to stronger overall performance.

<span id="section-4-4"></span>

### 4.4 Ablation Studies

In order to substantiate the effectiveness of the fine-grained expert segmentation and shared expert isolation strategies, we conduct ablation studies for DeepSeekMoE and present the results in [Figure 3](#figure-03). For a fair comparison, we ensure all models included in the comparison have the same number of total parameters and activated parameters.

**Shared Expert Isolation.** In order to evaluate the influence of the shared expert isolation strategy, we isolate one expert as the shared one based on GShard. From [Figure 3](#figure-03), we observe that compared with GShard, the intentional isolation of a shared expert yields improved performance across a majority of benchmarks. These results support the proposition that the shared expert isolation strategy contributes to a stronger model performance.

**Fine-Grained Expert Segmentation.** In order to assess the effectiveness of the fine-grained expert segmentation strategy, we conduct a more detailed comparison by further segmenting the experts into a finer grain. To be specific, we segment each expert into 2 or 4 smaller experts, resulting in a total of 32 (1 shared + 31 routed) or 64 (1 shared + 63 routed) experts. [Figure 3](#figure-03) reveals a consistent trend that the continuous refinement of expert segmentation granularity corresponds to a continuous enhancement in overall model performance. These findings provide empirical substantiation for the effectiveness of the fine-grained expert segmentation strategy.

**Ratios Between Shared and Routed Experts.** In addition, we investigate the best ratio of shared experts and routed experts. Based on the finest granularity with 64 total experts and keeping the number of total experts and activated experts constant, we attempt to isolate 1, 2, and 4 experts as shared ones. We find that different ratios of the shared experts and routed experts do not significantly impact the performance, and 1, 2, and 4 shared experts achieve a Pile loss of 1.808, 1.806, and 1.811, respectively. Considering that the ratio of 1:3 yields a marginally better Pile loss, when scaling up DeepSeekMoE, we keep the ratio between shared experts and activated routed experts as 1:3.

<span id="section-4-5"></span>

### 4.5 Analysis on Expert Specialization

In this section, we conduct an empirical analysis on the expert specialization of DeepSeekMoE 2B. DeepSeekMoE 2B in this section refers to the model reported in [Table 1](#table-01), i.e., comprising 2.0B total parameters, with 1 shared expert and 7 out of 63 routed experts being activated.

<span id="figure-04"></span>

![Figure 4. Pile loss with regard to different ratios of disabled top routed experts. Notably, DeepSeekMoE exhibits greater sensitivity to the ratio of disabled top routed experts, indicating lower redundancy among routed experts in DeepSeekMoE.](../../papers/deepseek-moe/figure-04.png)

**Figure 4.** Pile loss with regard to different ratios of disabled top routed experts. Notably, DeepSeekMoE exhibits greater sensitivity to the ratio of disabled top routed experts, indicating lower redundancy among routed experts in DeepSeekMoE.

**DeepSeekMoE Exhibits Lower Redundancy Among Routed Experts.** In order to assess the redundancy among routed experts, we disable varying ratios of top routed experts and evaluate the Pile loss. To be specific, for each token, we mask a certain ratio of experts with the highest routing probability, and then select top-K experts from the remaining routed experts. For fairness, we compare DeepSeekMoE with GShard$\times 1.5$ since they have the same Pile loss when no experts are disabled. As shown in [Figure 4](#figure-04), compared with GShard$\times 1.5$, DeepSeekMoE is more sensitive to the disabling of top routed experts. This sensitivity suggests a lower level of parameter redundancy in DeepSeekMoE, since each routed expert is more irreplaceable. In contrast, GShard$\times 1.5$ exhibits greater redundancy among its expert parameters, so it can buffer the performance drop when top routed experts are disabled.

**Shared Experts Are Irreplaceable by Routed Experts.** In order to investigate the role of the shared expert in DeepSeekMoE, we disable it and activate one more routed expert. The evaluation on Pile shows a significant increase in the Pile loss, rising from 1.808 to 2.414, even though we maintain the same computational cost. This result highlights the crucial function of the shared expert and indicates that the shared expert captures fundamental and essential knowledge not shared with routed experts, making it irreplaceable by routed ones.

<span id="figure-05"></span>

![Figure 5. Pile loss with regard to different numbers of activated routed experts in DeepSeekMoE. With only 4 routed experts activated, DeepSeekMoE achieves a Pile loss comparable with GShard.](../../papers/deepseek-moe/figure-05.png)

**Figure 5.** Pile loss with regard to different numbers of activated routed experts in DeepSeekMoE. With only 4 routed experts activated, DeepSeekMoE achieves a Pile loss comparable with GShard.

<span id="figure-06"></span>

![Figure 6. Comparison between GShard and DeepSeekMoE with half the activated experts (trained from scratch). With the same total expert parameters and only half of the activated expert parameters, DeepSeekMoE still outperforms GShard.](../../papers/deepseek-moe/figure-06.png)

**Figure 6.** Comparison between GShard and DeepSeekMoE with half the activated experts (trained from scratch). With the same total expert parameters and only half of the activated expert parameters, DeepSeekMoE still outperforms GShard.

**DeepSeekMoE Acquires Knowledge More Accurately.** In order to validate our claim that higher flexibility in combining activated experts contributes to a more accurate and targeted knowledge acquisition, we investigate whether DeepSeekMoE can acquire requisite knowledge with fewer activated experts. To be specific, we vary the number of activated routed experts from 3 to 7 and evaluate the resulting Pile loss. As demonstrated in [Figure 5](#figure-05), even with only 4 routed experts activated, DeepSeekMoE achieves a Pile loss comparable with GShard. This observation supports the proposition that DeepSeekMoE can acquire requisite knowledge more accurately and efficiently.

Encouraged by these findings, in order to validate the expert specialization and accurate knowledge acquisition of DeepSeekMoE more rigorously, we train a new model from scratch. This model comprises 1 shared expert and 63 routed experts, where only 3 routed experts are activated. The evaluation results shown in [Figure 6](#figure-06) demonstrate that, even with the same total expert parameters and only half of the activated expert parameters, DeepSeekMoE still outperforms GShard. This highlights the ability of DeepSeekMoE to leverage expert parameters more efficiently, i.e., the proportion of effective parameters in the activated experts is much higher than that of GShard.

<span id="section-5"></span>

## 5 Scaling up to DeepSeekMoE 16B

With the DeepSeekMoE architecture, we scale up our MoE model to a larger scale with 16B total parameters and train it on 2T tokens. Our results demonstrate that compared with LLaMA2 7B, DeepSeekMoE 16B achieves superior performance with only about 40% of computations.

<span id="section-5-1"></span>

### 5.1 Experimental Setup

<span id="section-5-1-1"></span>

#### 5.1.1 Training Data and Tokenization

We sample the training data from the same corpus as described in [Section 4.1.1](#section-4-1-1). Different from the validation experiments, we sample a larger amount of data with 2T tokens, aligning with the number of training tokens of LLaMA2 7B. We also use the HuggingFace Tokenizer tools to train a BPE tokenizer, but the vocabulary size is set to 100K for DeepSeekMoE 16B.

<span id="section-5-1-2"></span>

#### 5.1.2 Hyper-Parameters

**Model Settings.** For DeepSeekMoE 16B, we set the number of Transformer layers to 28 and the hidden dimension to 2048. We employ the multi-head attention mechanism with a total of 16 attention heads, where each head has a dimension of 128. As for initialization, all learnable parameters are randomly initialized with a standard deviation of 0.006. We substitute all FFNs except for the first layer with MoE layers, since we observe that the load balance status converges especially slower for the first layer. Each MoE layer consists of 2 shared experts and 64 routed experts, where each expert is 0.25 times the size of a standard FFN. Each token will be routed to these 2 shared experts and 6 out of 64 routed experts. An even finer expert segmentation granularity is not employed due to the potential reduction in computational efficiency associated with excessively small expert sizes. At a larger scale over 16B, a finer granularity can still be employed. Under our configuration, DeepSeekMoE 16B has approximately 16.4B total parameters, with the number of activated parameters around 2.8B.

**Training Settings.** We employ the AdamW optimizer [Los17] with hyper-parameters set to $\beta_1=0.9$, $\beta_2=0.95$, and $\mathrm{weight\_decay}=0.1$. The learning rate is also scheduled using a warmup-and-step-decay strategy. Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate is multiplied by 0.316 at 80% of the training steps, and again by 0.316 at 90% of the training steps. The maximum learning rate for DeepSeekMoE 16B is set to $4.2 \times 10^{-4}$, and the gradient clipping norm is set to 1.0. The batch size is set to 4.5K, and with a maximum sequence length of 4K, each training batch contains 18M tokens. Correspondingly, the total number of training steps is set to 106,449 to achieve 2T training tokens. Due to the abundance of training data, we do not use dropout during training. We leverage pipeline parallelism to deploy different layers of a model on different devices, and for each layer, all the experts will be deployed on the same device. Therefore, we also do not drop any tokens during training and do not employ the device-level balance loss. In order to prevent routing collapse, we set a quite small expert-level balance factor of 0.001 because we find that under our parallelization strategy, a higher expert-level balance factor cannot increase the computation efficiency, but instead, it will compromise the model performance.

<span id="section-5-1-3"></span>

#### 5.1.3 Evaluation Benchmarks

In addition to the benchmarks used in the validation experiments, we incorporate additional benchmarks for a more comprehensive evaluation. We introduce the distinctions from the benchmarks used in validation experiments as follows.

**Language Modeling.** For language modeling, we also evaluate the models on the test set of Pile [Gao20]. Since the tokenizer used in DeepSeekMoE 16B is different from that used in LLaMA2 7B. For a fair comparison, we use bits per byte (BPB) as the evaluation metric.

**Reading Comprehension.** For reading comprehension, we additionally consider DROP [Dua19]. The evaluation metric is the Exactly Matching (EM) rate.

**Math Reasoning.** For math reasoning, we additionally incorporate GSM8K [Cob21] and MATH [Hen21], using EM as the evaluation metric.

**Multi-Subject Multiple-Choice.** For multi-subject multiple-choice, we additionally evaluate the models on MMLU [Hen20]. The evaluation metric is accuracy.

**Disambiguation.** For disambiguation, we additionally consider WinoGrande [Sak19] and the evaluation metric is accuracy.

**Chinese Benchmarks.** Since DeepSeekMoE 16B is pretrained on a bilingual corpus, we also evaluate it on four Chinese benchmarks. CLUEWSC [Xu20] is a Chinese disambiguation benchmark. CEval [Hua23] and CMMLU [Li23e] are two Chinese multi-subject multiple-choice benchmarks with a similar form to MMLU. CHID [Zhe19] is a Chinese idiom completion benchmark, aiming to evaluate the understanding of Chinese culture. The evaluation metrics for the aforementioned Chinese benchmarks are accuracy or EM.

**Open LLM Leaderboard.** We evaluate all of the aforementioned benchmarks based on our internal evaluation framework. In order to compare DeepSeekMoE 16B with open source models fairly and conveniently, we additionally evaluate DeepSeekMoE 16B on the Open LLM Leaderboard. The Open LLM Leaderboard is a public leaderboard supported by HuggingFace, it consists of six tasks: ARC [Cla18], HellaSwag [Zel19], MMLU [Hen20], TruthfulQA [Lin22], Winogrande [Sak19], and GSM8K [Cob21].

<span id="section-5-2"></span>

### 5.2 Evaluations

<span id="table-03"></span>

![Table 3. Comparison between DeepSeek 7B and DeepSeekMoE 16B. Bold font indicates the best or near the best. With only 40.5% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B.](../../papers/deepseek-moe/table-03.png)

**Table 3.** Comparison between DeepSeek 7B and DeepSeekMoE 16B. **Bold** font indicates the best or near the best. With only 40.5% of computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B.

<span id="section-5-2-1"></span>

#### 5.2.1 Internal Comparison with DeepSeek 7B

We first conduct an internal comparison between DeepSeekMoE 16B and DeepSeek 7B [Dee24e], a dense language model with 6.9B parameters. Ensuring fairness, both models are trained on the same corpus with 2T tokens. This enables an accurate assessment of the effectiveness of our MoE architecture, independent of the influence of the training data.

The evaluation results are presented in [Table 3](#table-03), yielding the following observations: (1) On the whole, with about only 40% of the computations, DeepSeekMoE 16B achieves comparable performance with DeepSeek 7B. (2) DeepSeekMoE 16B exhibits notable strengths in language modeling and knowledge-intensive tasks such as Pile, HellaSwag, TriviaQA, and NaturalQuestions. Given that in an MoE model, FFN parameters are much heavier than attention parameters, these outcomes align with the proposition that FFNs in Transformers exhibit the capability for knowledge memorization [Dai22a]. (3) Compared with the excellent performance on other tasks, DeepSeekMoE exhibits limitations in addressing multiple-choice tasks. This inadequacy stems from the limited attention parameters in DeepSeekMoE 16B (DeepSeekMoE 16B has only about 0.5B attention parameters, while DeepSeek 7B has 2.5B attention parameters). Our earlier investigation on DeepSeek 7B reveals a positive correlation between the attention capacity and performance on multiple-choice tasks. For example, DeepSeek 7B MQA, which is equipped with the multi-query attention mechanism [Sha19], also struggled in MMLU-like tasks. In addition, for a more comprehensive understanding of the training process of DeepSeekMoE 16B, we also provide the benchmark curves of DeepSeekMoE 16B and DeepSeek 7B (Dense) during training in [Section 12](#section-12) for reference.

Critically, due to the modest number of parameters in DeepSeekMoE 16B, it enables single-device deployment on a GPU with 40GB of memory. With appropriate operator optimizations, it can achieve nearly 2.5 times the inference speed of a 7B dense model.

<span id="table-04"></span>

![Table 4. Comparison between LLaMA2 7B and DeepSeekMoE 16B. With only 39.6% of computations, DeepSeekMoE 16B outperforms LLaMA2 7B on the majority of benchmarks.](../../papers/deepseek-moe/table-04.png)

**Table 4.** Comparison between LLaMA2 7B and DeepSeekMoE 16B. With only 39.6% of computations, DeepSeekMoE 16B outperforms LLaMA2 7B on the majority of benchmarks.

<span id="section-5-2-2"></span>

#### 5.2.2 Comparison with Open Source Models

**Internal Comparison with LLaMA2 7B.** In the realm of open source models, we mainly compare DeepSeekMoE 16B with LLaMA2 7B [Tou23a], a well-known and strong open source language model with 6.7B parameters. Both DeepSeekMoE 16B and LLaMA2 7B are pretrained on 2T tokens. Compared with LLaMA2 7B, DeepSeekMoE has 245% of total parameters but only needs 39.6% of computations. The results on our internal benchmarks are presented in [Table 4](#table-04), leading to the following observations. (1) Among the evaluated benchmarks, with only about 40% of computations, DeepSeekMoE 16B outperforms LLaMA2 7B on the majority of benchmarks. (2) The math reasoning and code generation capabilities of DeepSeekMoE 16B are stronger than LLaMA2 7B, attributed to the enriched presence of mathematical and code-related text in our pretraining corpus. (3) Given the presence of Chinese texts in our pretraining corpus, DeepSeekMoE 16B exhibits a substantial performance advantage over LLaMA2 7B on Chinese benchmarks. (4) Despite being trained on fewer English texts, DeepSeekMoE 16B achieves comparable or better performance compared with LLaMA2 7B on English understanding or knowledge-intensive benchmarks, which demonstrates the exceptional capabilities of DeepSeekMoE 16B.

**Evaluation on Open LLM Leaderboard.** Beyond our internal evaluations, we also evaluate DeepSeekMoE 16B on the Open LLM Leaderboard and compare it with other open source models. In addition to LLaMA2 7B, we take a broader set of open source models into consideration, including LLaMA 7B [Tou23], Falcon 7B [Alm23b], GPT-J 6B [Wan21g], RedPajama-INCITE 7B and 3B [Tog23a], Open LLaMA 7B and 3B [Gen23a], OPT 2.7B [Zha22], Pythia 2.8B [Bid23], GPT-neo 2.7B [Bla21], and BLOOM 3B [Les23]. The evaluation results, as presented in [Figure 1](#figure-01), show that DeepSeekMoE 16B consistently outperforms models with similar activated parameters by a large margin. Moreover, it achieves comparable performance with LLaMA2 7B, which has approximately 2.5 times the activated parameters.

<span id="section-6"></span>

## 6 Alignment for DeepSeekMoE 16B

Previous research indicates that MoE models typically do not emerge significant gains from fine-tuning [Fed22, Art22]. However, [She23a] present findings suggesting that MoE models can indeed benefit from instruction tuning. In order to assess whether DeepSeekMoE 16B can benefit from fine-tuning, we conduct supervised fine-tuning to construct a chat model based on DeepSeekMoE 16B. The experimental results reveal that DeepSeekMoE Chat 16B also achieves comparable performance with LLaMA2 SFT 7B and DeepSeek Chat 7B.

<span id="section-6-1"></span>

### 6.1 Experimental Setup

**Training Data.** For training the chat model, we conduct supervised fine-tuning (SFT) on our in-house curated data, comprising 1.4M training examples. This dataset spans a broad range of categories including math, code, writing, question answering, reasoning, summarization, and more. The majority of our SFT training data is in English and Chinese, rendering the chat model versatile and applicable in bilingual scenarios.

**Hyper-Parameters.** During supervised fine-tuning, we set the batch size to 1024 examples and conduct training over 8 epochs using the AdamW optimizer [Los17]. We employ a maximum sequence length of 4K, and pack the training examples as densely as possible until reaching the sequence length limit. We do not use dropout for supervised fine-tuning, and simply set a constant learning rate of $10^{-5}$ without incorporating any learning rate scheduling strategy.

**Evaluation Benchmarks.** For the evaluation of the chat models, we employ benchmarks similar to those used in [Section 5.1.3](#section-5-1-3), with the following adjustments: (1) We exclude Pile [Gao20] since chat models are seldom employed for pure language modeling. (2) We exclude CHID [Zhe19] due to the observed instability of results, hindering the derivation of solid conclusions. (3) We additionally include BBH [Suz22] to provide a more comprehensive assessment of the reasoning ability of the chat models.

<span id="table-05"></span>

![Table 5. Comparison among LLaMA2 SFT 7B, DeepSeek Chat 7B and DeepSeekMoE Chat 16B, with all of these three models fine-tuned on the same SFT data. Compared with both 7B dense models, DeepSeekMoE Chat 16B still achieves comparable or better performance on the majority of benchmarks with only 40% of computations.](../../papers/deepseek-moe/table-05.png)

**Table 5.** Comparison among LLaMA2 SFT 7B, DeepSeek Chat 7B and DeepSeekMoE Chat 16B, with all of these three models fine-tuned on the same SFT data. Compared with both 7B dense models, DeepSeekMoE Chat 16B still achieves comparable or better performance on the majority of benchmarks with only 40% of computations.

<span id="section-6-2"></span>

### 6.2 Evaluations

**Baselines.** In order to validate the potential of DeepSeekMoE 16B after alignment, we conduct supervised fine-tuning for LLaMA2 7B, DeepSeek 7B, and DeepSeekMoE 16B, where we utilize totally the same fine-tuning data to ensure fairness. Correspondingly, we construct three chat models, including LLaMA2 SFT 7B [+3], DeepSeek Chat 7B, and DeepSeekMoE Chat 16B. Subsequently, we compare DeepSeekMoE Chat 16B with the other two dense chat models (with about 2.5 times the FLOPs) across a wide range of downstream tasks.

**Results.** The evaluation results are presented in [Table 5](#table-05). Our key observations include: (1) DeepSeekMoE Chat 16B, while consuming nearly 40% of computations, achieves comparable performance with 7B dense models across language understanding and reasoning (PIQA, ARC, BBH), machine reading comprehension (RACE), mathematical (GSM8K, MATH), and knowledge-intensive tasks (TriviaQA, NaturalQuestions). (2) On code generation tasks, DeepSeekMoE Chat 16B significantly outperforms LLaMA2 SFT 7B, demonstrating notable improvements on HumanEval and MBPP. In addition, it also surpasses DeepSeek Chat 7B. (3) On multiple-choice question answering benchmarks including MMLU, CEval, and CMMLU, DeepSeekMoE Chat 16B still falls behind DeepSeek Chat 7B, consistent with the observations for the base model ([Section 5.2.1](#section-5-2-1)). However, it is worth noting that, after supervised fine-tuning, the performance gap between DeepSeekMoE 16B and DeepSeek 7B is narrowed. (4) Benefiting from the pretraining on a bilingual corpus, DeepSeekMoE Chat 16B notably outperforms LLaMA2 SFT 7B on all Chinese benchmarks. These results demonstrate the balanced capabilities of DeepSeekMoE 16B in both Chinese and English, enhancing its versatility and applicability in diverse scenarios. In conclusion, the evaluation for the chat models highlights the potential of DeepSeekMoE 16B in benefiting from alignment, and validates its consistent advantages in achieving comparable performance with dense models while using only about 40% of computations.

<span id="section-7"></span>

## 7 DeepSeekMoE 145B Ongoing

Encouraged by the outstanding performance of DeepSeekMoE 16B, we further undertake a preliminary endeavor to scale up DeepSeekMoE to 145B. In this initial study, DeepSeekMoE 145B is trained on 245B tokens, but it has demonstrated consistent advantages over the GShard architecture and shown promise to match or exceed the performance of DeepSeek 67B (Dense). Furthermore, upon the completion of the final version and full training of DeepSeekMoE 145B, we also plan to make it publicly available.

<span id="section-7-1"></span>

### 7.1 Experimental Setup

**Training Data and Tokenization.** For DeepSeekMoE 145B, we employ exactly the same training corpus and tokenizer as DeepSeekMoE 16B, with the only difference being that DeepSeekMoE 145B is trained on 245B tokens for an initial study.

**Model Settings.** For DeepSeekMoE 145B, we set the number of Transformer layers to 62 and the hidden dimension to 4096. We employ the multi-head attention mechanism with a total of 32 attention heads, where each head has a dimension of 128. As for initialization, all learnable parameters are randomly initialized with a standard deviation of 0.006. As in DeepSeekMoE 16B, we also substitute all FFNs except for the first layer with MoE layers. Each MoE layer consists of 4 shared experts and 128 routed experts, where each expert is 0.125 times the size of a standard FFN. Each token will be routed to these 4 shared experts and 12 out of 128 routed experts. Under this configuration, DeepSeekMoE 145 has approximately 144.6B total parameters, with the number of activated parameters around 22.2B.

**Training Settings.** We employ the AdamW optimizer [Los17] with hyper-parameters set to $\beta_1=0.9$, $\beta_2=0.95$, and $\mathrm{weight\_decay}=0.1$. For the preliminary study of DeepSeekMoE 145B, we employ a warmup-and-constant learning rate scheduler. Initially, the learning rate linearly increases from 0 to the maximum value during the first 2K steps. Subsequently, the learning rate keeps constant during the remaining training process. The maximum learning rate for DeepSeekMoE 145B is set to $3.0 \times 10^{-4}$, and the gradient clipping norm is set to 1.0. The batch size is set to 4.5K, and with a maximum sequence length of 4K, each training batch contains 18M tokens. We train DeepSeekMoE 145B for 13,000 steps, achieving 245B training tokens. Also, we do not use dropout during training. We leverage pipeline parallelism to deploy different layers of a model on different devices, and for each layer, all the routed experts will be uniformly deployed on 4 devices (i.e., expert parallelism combined with data parallelism). Since we employ expert parallelism for DeepSeekMoE 145B, the device-level load balance should be considered to reduce the computational bottleneck. In response, we set the device-level balance factor to 0.05 to encourage balanced computation across devices. Also, we still set a small expert-level balance factor of 0.003 to prevent routing collapse.

**Evaluation Benchmarks.** We evaluate DeepSeekMoE 145B on exactly the same internal benchmarks as used for DeepSeekMoE 16B (see [Section 5.1.3](#section-5-1-3)).

<span id="table-06"></span>

![Table 6. Comparison among DeepSeek 67B (Dense) and MoE models at the scale of about 140B total parameters. In the lines of '# Experts' and '# Activated Experts', $a$ + $b$ denotes $a$ shared experts and $b$ routed experts, respectively. Bold font indicates the best or near the best performance excluding the last column. DeepSeekMoE 145B, and even DeepSeekMoE 142B (Half Activated) that has only a half of activated expert parameters, outperform GShard 137B by a large margin. Moreover, with 28.5% of computations, DeepSeekMoE 145B achieves comparable performance with DeepSeek 67B.](../../papers/deepseek-moe/table-06.png)

**Table 6.** Comparison among DeepSeek 67B (Dense) and MoE models at the scale of about 140B total parameters. In the lines of "# Experts" and "# Activated Experts", $a$ + $b$ denotes $a$ shared experts and $b$ routed experts, respectively. **Bold** font indicates the best or near the best performance excluding the last column. DeepSeekMoE 145B, and even DeepSeekMoE 142B (Half Activated) that has only a half of activated expert parameters, outperform GShard 137B by a large margin. Moreover, with 28.5% of computations, DeepSeekMoE 145B achieves comparable performance with DeepSeek 67B.

<span id="section-7-2"></span>

### 7.2 Evaluations

**Baselines.** Apart from **DeepSeekMoE 145B**, we consider three additional models for comparison. **DeepSeek 67B (Dense)** is a dense model with 67.4B total parameters (refer to [Dee24e] for the model and training details). **GShard 137B** shares the same hidden dimension and number of layers as DeepSeekMoE 145B, but follows the GShard architecture. Note that DeepSeekMoE 145B aligns the intermediate hidden dimension in each expert to a multiple of 64 for computation efficiency, so its model size is 6% larger than GShard 137B. **DeepSeekMoE 142B (Half Activated)** has a similar architecture to DeepSeekMoE 145B, but it contains only 2 shared experts, and only 6 out of 128 routed experts are activated. It is noteworthy that all compared models, including DeepSeekMoE 145B, share the same training corpus. In addition, all MoE models in the comparison are trained from scratch and share the same training hyper-parameters.

**Results.** From the evaluation results presented in [Table 6](#table-06), we have the following observations: (1) Despite having comparable total parameters and computations, DeepSeekMoE 145B significantly outperforms GShard 137B, highlighting the advantages of the DeepSeekMoE architecture again. (2) On the whole, with only 28.5% of computations, DeepSeekMoE 145B achieves comparable performance with DeepSeek 67B (Dense). Consistent with the findings from DeepSeekMoE 16B, DeepSeekMoE 145B exhibits remarkable strengths in language modeling and knowledge-intensive tasks, but with limitations in multiple-choice tasks. (3) At a larger scale, the performance of DeepSeekMoE 142B (Half Activated) does not lag behind too much from DeepSeekMoE 145B. In addition, despite having only a half of activated expert parameters, DeepSeekMoE 142B (Half Activated) still match the performance of DeepSeek 67B (Dense), with only 18.2% of computations. It also outperforms GShard 137B, which aligns with the conclusion from [Section 4.5](#section-4-5).

<span id="section-8"></span>

## 8 Related Work

The Mixture of Experts (MoE) technique is first proposed by [Jac91] [Jor94] to deal with different samples with independent expert modules. [Sha17] introduce MoE into language model training and build a large-scale LSTM-based [Hoc97] MoE models. As Transformer become the most popular architecture for NLP, many attempts extend FFNs in a Transformer as MoE layers to build MoE language models. GShard [Lep20] and Switch Transformer [Fed22] are pioneers which employ learnable top-2 or top-1 routing strategies to scale the MoE language models to an extremely large scale. Hash Layer [Rol21] and StableMoE [Dai22b] use fixed routing strategies for more stable routing and training. [Zho22a] propose an expert-choice routing strategy, where each token can be assigned to different numbers of experts. [Zop22] focus on the issues of training instability and fine-tuning difficulty in MoE models, and propose ST-MoE to overcome these challenges. In addition to research on MoE architectures and training strategies, recent years have also witnessed the emergence of numerous large-scale language or multimodal models [Lin21a, Du22, Ren23a, Xue23] based on existing MoE architectures. By and large, most of the previous MoE models are based on conventional top-1 or top-2 routing strategies, leaving large room for improving expert specialization. In response, our DeepSeekMoE architecture aims to improve the expert specialization to the utmost extent.

<span id="section-9"></span>

## 9 Conclusion

In this paper, we introduce the DeepSeekMoE architecture for MoE language models, with the objective of achieving ultimate expert specialization. Through fine-grained expert segmentation and shared expert isolation, DeepSeekMoE achieves significantly higher expert specialization and performance compared with prevailing MoE architectures. Starting with a modest scale of 2B parameters, we validate the advantages of DeepSeekMoE, demonstrating its capability to approach the upper bound performance for MoE models. Furthermore, we provide empirical evidence to show that DeepSeekMoE has a higher level of expert specialization than GShard.

Scaling up to a larger scale of 16B total parameters, we train DeepSeekMoE 16B on 2T tokens and demonstrate its outstanding performance comparable with DeepSeek 7B and LLaMA2 7B, with only about 40% of computations. Additionally, supervised fine-tuning is conducted for alignment to construct an MoE chat model based on DeepSeekMoE 16B, further showing its adaptability and versatility. Further, we perform a preliminary exploration to scale DeepSeekMoE to 145B parameters. We find that DeepSeekMoE 145B still keeps substantial advantages over the GShard architecture, and demonstrates comparable performance with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations.

For research purposes, we release the model checkpoint of DeepSeekMoE 16B to the public, which can be deployed on a single GPU with 40GB of memory. We aspire for this work to provide valuable insights for both academia and industry, and contribute to the accelerated advancement of large-scale language models.

<span id="section-10"></span>

## 10 Overview of Hyper-Parameters

We present the overview of hyper-parameters for DeepSeekMoE across various sizes in [Table 7](#table-07).

<span id="table-07"></span>

![Table 7. Overview of hyper-parameters for DeepSeekMoE across various sizes. The relative expert size is in comparison to a standard FFN.](../../papers/deepseek-moe/table-07.png)

**Table 7.** Overview of hyper-parameters for DeepSeekMoE across various sizes. The relative expert size is in comparison to a standard FFN.

<span id="section-11"></span>

## 11 Comparing DeepSeekMoE with Larger Models

Comparisons among DeepSeekMoE, GShard$\times 1.2$, and GShard$\times 1.5$ are shown in [Table 8](#table-08). Comparisons among DeepSeekMoE, Dense$\times 4$, and Dense$\times 16$ are shown in [Table 9](#table-09).

<span id="table-08"></span>

![Table 8. Comparison between DeepSeekMoE and larger GShard models.](../../papers/deepseek-moe/table-08.png)

**Table 8.** Comparison between DeepSeekMoE and larger GShard models.

<span id="table-09"></span>

![Table 9. Comparison between DeepSeekMoE and larger dense baselines.](../../papers/deepseek-moe/table-09.png)

**Table 9.** Comparison between DeepSeekMoE and larger dense baselines.

At a larger scale of 13B total parameters, we also compare DeepSeekMoE with GShard$\times 1.2$ and GShard$\times 1.5$, and show results in [Table 10](#table-10). At a larger scale, DeepSeekMoE even outperforms GShard$\times 1.5$ distinctly.

<span id="table-10"></span>

![Table 10. Comparison between DeepSeekMoE and larger GShard models at a larger scale.](../../papers/deepseek-moe/table-10.png)

**Table 10.** Comparison between DeepSeekMoE and larger GShard models at a larger scale.

<span id="section-12"></span>

## 12 Training Benchmark Curves of DeepSeekMoE 16B

We present the benchmark curves during training of DeepSeekMoE 16B and DeepSeek 7B (Dense) in [Figure 7](#figure-07) for reference.

<span id="figure-07"></span>

![Figure 7. Benchmark curves during training of DeepSeekMoE 16B and DeepSeek 7B (Dense).](../../papers/deepseek-moe/figure-07.png)

**Figure 7.** Benchmark curves during training of DeepSeekMoE 16B and DeepSeek 7B (Dense).

[+1]: [Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
[+2]: [HuggingFace Tokenizers](https://github.com/huggingface/tokenizers)
[+3]: We use LLaMA2 SFT to distinguish from the official LLaMA2 Chat [Tou23a] model.
