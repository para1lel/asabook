---
title: 'The Llama 3 Herd of Models'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/llama-3/
---

> [Llama Team](https://llama.meta.com/). First submitted to arXiv on July 31, 2024; current version v3. [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783). [Original PDF](/paper/llama-3.pdf). [TeX source](https://export.arxiv.org/e-print/2407.21783). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support multilinguality, coding, reasoning, and tool usage. Our largest model is a dense Transformer with 405B parameters and a context window of up to 128K tokens. This paper presents an extensive empirical evaluation of Llama 3. We find that Llama 3 delivers comparable quality to leading language models such as GPT-4 on a plethora of tasks. We publicly release Llama 3, including pre-trained and post-trained versions of the 405B parameter language model and our Llama Guard 3 model for input and output safety. The paper also presents the results of experiments in which we integrate image, video, and speech capabilities into Llama 3 via a compositional approach. We observe this approach performs competitively with the state-of-the-art on image, video, and speech recognition tasks. The resulting models are not yet being broadly released as they are still under development.

[Website](https://llama.meta.com/)
<span id="section-1"></span>

## 1 Introduction

Foundation models are general models of language, vision, speech, and/or other modalities that are designed to support a large variety of AI tasks. They form the basis of many modern AI systems.

The development of modern foundation models consists of two main stages: (1) a pre-training stage in which the model is trained at massive scale using straightforward tasks such as next-word prediction or captioning and (2) a post-training stage in which the model is tuned to follow instructions, align with human preferences, and improve specific capabilities (for example, coding and reasoning).
<span id="table-01"></span>

![Original paper Table 1](../../papers/llama-3/table-01.png)

**Table 1.** Overview of the Llama 3 Herd of models. All results in this paper are for the Llama 3.1 models.

In this paper, we present a new set of foundation models for language, called Llama 3. The Llama 3 Herd of models natively supports multilinguality, coding, reasoning, and tool usage. Our largest model is dense Transformer with 405B parameters, processing information in a context window of up to 128K tokens. Each member of the herd is listed in [Table 1](#table-01). All the results presented in this paper are for the Llama 3.1 models, which we will refer to as Llama 3 throughout for brevity.

We believe there are three key levers in the development of high-quality foundation models: data, scale, and managing complexity. We seek to optimize for these three levers in our development process:

- **Data.** Compared to prior versions of Llama [Tou23, Tou23c], we improved both the quantity and quality of the data we use for pre-training and post-training. These improvements include the development of more careful pre-processing and curation pipelines for pre-training data and the development of more rigorous quality assurance and filtering approaches for post-training data. We pre-train Llama 3 on a corpus of about 15T multilingual tokens, compared to 1.8T tokens for Llama 2.
- **Scale.** We train a model at far larger scale than previous Llama models: our flagship language model was pre-trained using $3.8\times 10^{25}$ FLOPs, almost $50\times$ more than the largest version of Llama 2. Specifically, we pre-trained a flagship model with 405B trainable parameters on 15.6T text tokens. As expected per scaling laws for foundation models, our flagship model outperforms smaller models trained using the same procedure. While our scaling laws suggest our flagship model is an approximately compute-optimal size for our training budget, we also train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget. We use the flagship model to further improve the quality of those smaller models during post-training.
- **Managing complexity.** We make design choices that seek to maximize our ability to scale the model development process. For example, we opt for a standard dense Transformer model architecture [Vas17] with minor adaptations, rather than for a mixture-of-experts model [Sha17] to maximize training stability. Similarly, we adopt a relatively simple post-training procedure based on supervised finetuning (SFT), rejection sampling (RS), and direct preference optimization (DPO; [Raf24]) as opposed to more complex reinforcement learning algorithms [Ouy22, Sch17] that tend to be less stable and harder to scale.

The result of our work is Llama 3: a herd of three multilingual [+1] language models with 8B, 70B, and 405B parameters. We evaluate the performance of Llama 3 on a plethora of benchmark datasets that span a wide range of language understanding tasks. In addition, we perform extensive human evaluations that compare Llama 3 with competing models. An overview of the performance of the flagship Llama 3 model on key benchmarks is presented in [Table 1](#table-01). Our experimental evaluation suggests that our flagship model performs on par with leading language models such as GPT-4 [Ope23b] across a variety of tasks, and is close to matching the state-of-the-art. Our smaller models are best-in-class, outperforming alternative models with similar numbers of parameters [Bai23b, Jia23a]. Llama 3 also delivers a much better balance between helpfulness and harmlessness than its predecessor [Tou23c]. We present a detailed analysis of the safety of Llama 3 in Section [5.4](#section-5-4).
<span id="table-02"></span>

![Table 2](../../papers/llama-3/table-02.png)

**Table 2.** Performance of finetuned Llama 3 models on key benchmark evaluations. The table compares the performance of the 8B, 70B, and 405B versions of Llama 3 with that of competing models. We boldface the best-performing model in each of three model-size equivalence classes. ${}^{\triangle}$Results obtained using 5-shot prompting (no CoT). ${}^{\triangleleft}$Results obtained without CoT. ${}^{\diamondsuit}$Results obtained using zero-shot prompting.

We are publicly releasing all three Llama 3 models under an updated version of the Llama 3 Community License; see [https://llama.meta.com](https://llama.meta.com). This includes pre-trained and post-trained versions of our 405B parameter language model and a new version of our Llama Guard model [Ina23] for input and output safety. We hope that the open release of a flagship model will spur a wave of innovation in the research community, and accelerate a responsible path towards the development of artificial general intelligence (AGI).

As part of the Llama 3 development process we also develop multimodal extensions to the models, enabling image recognition, video recognition, and speech understanding capabilities. These models are still under active development and not yet ready for release. In addition to our language modeling results, the paper presents results of our initial experiments with those multimodal models.
<span id="section-2"></span>

## 2 General Overview

The model architecture of Llama 3 is illustrated in [Figure 1](#figure-01). The development of our Llama 3 language models comprises two main stages:

- **Language model pre-training.** We start by converting a large, multilingual text corpus to discrete tokens and pre-training a large language model (LLM) on the resulting data to perform next-token prediction. In the language model pre-training stage, the model learns the structure of language and obtains large amounts of knowledge about the world from the text it is “reading”. To do this effectively, pre-training is performed at massive scale: we pre-train a model with 405B parameters on 15.6T tokens using a context window of 8K tokens. This standard pre-training stage is followed by a continued pre-training stage that increases the supported context window to 128K tokens. See Section [3](#section-3) for details.
- **Language model post-training.** The pre-trained language model has a rich understanding of language but it does not yet follow instructions or behave in the way we would expect an assistant to. We align the model with human feedback in several rounds, each of which involves supervised finetuning (SFT) on instruction tuning data and Direct Preference Optimization [Raf24]. At this post-training [+2] stage, we also integrate new capabilities, such as tool-use, and observe strong improvements in other areas, such as coding and reasoning. See Section [4](#section-4) for details. Finally, safety mitigations are also incorporated into the model at the post-training stage, the details of which are described in Section [5.4](#section-5-4).
<span id="figure-01"></span>

![Refer to caption](../../papers/llama-3/figure-01.png)

**Figure 1.** Illustration of the overall architecture and training of Llama 3. Llama 3 is a Transformer language model trained to predict the next token of a textual sequence. See text for details.

The resulting models have a rich set of capabilities. They can answer questions in at least eight languages, write high-quality code, solve complex reasoning problems, and use tools out-of-the-box or in a zero-shot way.

We also perform experiments in which we add image, video, and speech capabilities to Llama 3 using a compositional approach. The approach we study comprises the three additional stages illustrated in [Figure 28](#figure-28):

- **Multi-modal encoder pre-training.** We train separate encoders for images and speech. We train our image encoder on large amounts of image-text pairs. This teaches the model the relation between visual content and the description of that content in natural language. Our speech encoder is trained using a self-supervised approach that masks out parts of the speech inputs and tries to reconstruct the masked out parts via a discrete-token representation. As a result, the model learns the structure of speech signals. See Section [7](#section-7) for details on the image encoder and Section [8](#section-8) for details on the speech encoder.
- **Vision adapter training.** We train an adapter that integrates the pre-trained image encoder into the pre-trained language model. The adapter consists of a series of cross-attention layers that feed image-encoder representations into the language model. The adapter is trained on text-image pairs. This aligns the image representations with the language representations. During adapter training, we also update the parameters of the image encoder but we intentionally do not update the language-model parameters. We also train a video adapter on top of the image adapter on paired video-text data. This enables the model to aggregate information across frames. See Section [7](#section-7) for details.
- **Speech adapter training.** Finally, we integrate the speech encoder into the model via an adapter that converts speech encodings into token representations that can be fed directly into the finetuned language model. The parameters of the adapter and encoder are jointly updated in a supervised finetuning stage to enable high-quality speech understanding. We do not change the language model during speech adapter training. We also integrate a text-to-speech system. See Section [8](#section-8) for details.

Our multimodal experiments lead to models that can recognize the content of images and videos, and support interaction via a speech interface. These models are still under development and not yet ready for release.
<span id="section-3"></span>

## 3 Pre-Training

Language model pre-training involves: (1) the curation and filtering of a large-scale training corpus, (2) the development of a model architecture and corresponding scaling laws for determining model size, (3) the development of techniques for efficient pre-training at large scale, and (4) the development of a pre-training recipe. We present each of these components separately below.
<span id="section-3-1"></span>

### 3.1 Pre-Training Data

We create our dataset for language model pre-training from a variety of data sources containing knowledge until the end of 2023. We apply several de-duplication methods and data cleaning mechanisms on each data source to obtain high-quality tokens. We remove domains that contain large amounts of personally identifiable information (PII), and domains with known adult content.
<span id="section-3-1-1"></span>

#### 3.1.1 Web Data Curation

Much of the data we utilize is obtained from the web and we describe our cleaning process below.

**PII and safety filtering.** Among other mitigations, we implement filters designed to remove data from websites are likely to contain unsafe content or high volumes of PII, domains that have been ranked as harmful according to a variety of Meta safety standards, and domains that are known to contain adult content.

**Text extraction and cleaning.** We process the raw HTML content for non-truncated web documents to extract high-quality diverse text. To do so, we build a custom parser that extracts the HTML content and optimizes for precision in boilerplate removal and content recall. We evaluate our parser’s quality in human evaluations, comparing it with popular third-party HTML parsers that optimize for article-like content, and found it to perform favorably. We carefully process HTML pages with mathematics and code content to preserve the structure of that content. We maintain the image alt attribute text since mathematical content is often represented as pre-rendered images where the math is also provided in the alt attribute. We experimentally evaluate different cleaning configurations. We find markdown is harmful to the performance of a model that is primarily trained on web data compared to plain text, so we remove all markdown markers.

**De-duplication.** We apply several rounds of de-duplication at the URL, document, and line level:

- **URL-level de-duplication.** We perform URL-level de-duplication across the entire dataset. We keep the most recent version for pages corresponding to each URL.
- **Document-level de-duplication.** We perform global MinHash [Bro97] de-duplication across the entire dataset to remove near duplicate documents.
- **Line-level de-duplication.** We perform aggressive line-level de-duplication similar to ccNet [Wen19]. We remove lines that appeared more than 6 times in each bucket of 30M documents. Although our manual qualitative analysis showed that the line-level de-duplication removes not only leftover boilerplate from various websites such as navigation menus, cookie warnings, but also frequent high-quality text, our empirical evaluations showed strong improvements.

**Heuristic filtering.** We develop heuristics to remove additional low-quality documents, outliers, and documents with excessive repetitions. Some examples of heuristics include:

- We use duplicated n-gram coverage ratio [Rae21] to remove lines that consist of repeated content such as logging or error messages. Those lines could be very long and unique, hence cannot be filtered by line-dedup.
- We use “dirty word” counting [Raf19] to filter out adult websites that are not covered by domain block lists.
- We use a token-distribution Kullback-Leibler divergence to filter out documents containing excessive numbers of outlier tokens compared to the training corpus distribution.

**Model-based quality filtering.** Further, we experiment with applying various model-based quality classifiers to sub-select high-quality tokens. These include using fast classifiers such as fasttext [Jou17] trained to recognize if a given text would be referenced by Wikipedia [Tou23], as well as more compute-intensive Roberta-based classifiers [Liu19a] trained on Llama 2 predictions. To train a quality classifier based on Llama 2, we create a training set of cleaned web documents, describe the quality requirements, and instruct Llama 2’s chat model to determine if the documents meets these requirements. We use DistilRoberta [San19] to generate quality scores for each document for efficiency reasons. We experimentally evaluate the efficacy of various quality filtering configurations.

**Code and reasoning data.** Similar to [Dee24b], we build domain-specific pipelines that extract code and math-relevant web pages. Specifically, both the code and reasoning classifiers are DistilRoberta models trained on web data annotated by Llama 2. Unlike the general quality classifier mentioned above, we conduct prompt tuning to target web pages containing math deduction, reasoning in STEM areas and code interleaved with natural language. Since the token distribution of code and math is substantially different than that of natural language, these pipelines implement domain-specific HTML extraction, customized text features and heuristics for filtering.

**Multilingual data.** Similar to our processing pipelines for English described above, we implement filters to remove data from websites that are likely to contain PII or unsafe content. Our multilingual text processing pipeline has several unique features:

- We use a fasttext-based language identification model to categorize documents into 176 languages.
- We perform document-level and line-level de-duplication within data for each language.
- We apply language-specific heuristics and model-based filters to remove low-quality documents.

In addition, we perform quality ranking of multilingual documents using a multilingual Llama 2-based classifier to ensure that high-quality content is prioritized. We determine the amount of multilingual tokens used in pre-training experimentally, balancing model performance on English and multilingual benchmarks.
<span id="section-3-1-2"></span>

#### 3.1.2 Determining the Data Mix

To obtain a high-quality language model, it is essential to carefully determine the proportion of different data sources in the pre-training data mix. Our main tools in determining this data mix are knowledge classification and scaling law experiments.

**Knowledge classification.** We develop a classifier to categorize the types of information contained in our web data to more effectively determine a data mix. We use this classifier to downsample data categories that are over-represented on the web, for example, arts and entertainment.

**Scaling laws for data mix.** To determine the best data mix, we perform scaling law experiments in which we train several small models on a data mix and use that to predict the performance of a large model on that mix (see Section [3.2.1](#section-3-2-1)). We repeat this process multiple times for different data mixes to select a new data mix candidate. Subsequently, we train a larger model on this candidate data mix and evaluate the performance of that model on several key benchmarks.

**Data mix summary.** Our final data mix contains roughly 50% of tokens corresponding to general knowledge, 25% of mathematical and reasoning tokens, 17% code tokens, and 8% multilingual tokens.
<span id="section-3-1-3"></span>

#### 3.1.3 Annealing Data

Empirically, we find that annealing (see Section [3.4.3](#section-3-4-3)) on small amounts of high-quality code and mathematical data can boost the performance of pre-trained models on key benchmarks. Akin to [Li25a], we perform annealing with a data mix that upsamples high-quality data in select domains. We do not include any training sets from commonly used benchmarks in our annealing data. This enables us to assess the true few-shot learning capabilities and out-of-domain generalization of Llama 3.

Following [Ope23b], we evaluate the efficacy of annealing on the GSM8k [Cob21] and MATH [Hen21] training sets in annealing. We find that annealing improved the performance of a pre-trained Llama 3 8B model on the GSM8k and MATH validation sets by 24.0% and 6.4%, respectively. However, the improvements on the 405B model are negligible, suggesting that our flagship model has strong in-context learning and reasoning capabilities and does not require specific in-domain training samples to obtain strong performance.

**Using annealing to assess data quality.** Similar to [Bla24], we find that annealing enables us to judge the value of small domain-specific datasets. We measure the value of such datasets by annealing the learning rate of a 50% trained Llama 3 8B model linearly to 0 on 40B tokens. In those experiments, we assign 30% weight to the new dataset and the remaining 70% weight to the default data mix. Using annealing to evaluate new data sources is more efficient than performing scaling law experiments for every small dataset.
<span id="section-3-2"></span>

### 3.2 Model Architecture

Llama 3 uses a standard, dense Transformer architecture [Vas17]. It does not deviate significantly from Llama and Llama 2 [Tou23, Tou23c] in terms of model architecture; our performance gains are primarily driven by improvements in data quality and diversity as well as by increased training scale.

We make a few small modifications compared to Llama 2:

- We use grouped query attention (GQA; [Ain23]) with 8 key-value heads to improve inference speed and to reduce the size of key-value caches during decoding.
- We use an attention mask that prevents self-attention between different documents within the same sequence. We find that this change had limited impact during in standard pre-training, but find it to be important in continued pre-training on very long sequences.
- We use a vocabulary with 128K tokens. Our token vocabulary combines 100K tokens from the tiktoken [+3] tokenizer with 28K additional tokens to better support non-English languages. Compared to the Llama 2 tokenizer, our new tokenizer improves compression rates on a sample of English data from 3.17 to 3.94 characters per token. This enables the model to “read” more text for the same amount of training compute. We also found that adding 28K tokens from select non-English languages improved both compression ratios and downstream performance, with no impact on English tokenization.
- We increase the RoPE base frequency hyperparameter to 500,000. This enables us to better support longer contexts; [Xio23] showed this value to be effective for context lengths up to 32,768.
<span id="table-03"></span>

![Original paper Table 3](../../papers/llama-3/table-03.png)

**Table 3.** Overview of the key hyperparameters of Llama 3. We display settings for 8B, 70B, and 405B language models.

Llama 3 405B uses an architecture with 126 layers, a token representation dimension of 16,384, and 128 attention heads; see [Table 3](#table-03) for details. This leads to a model size that is approximately compute-optimal according to scaling laws on our data for our training budget of $3.8\times 10^{25}$ FLOPs.
<span id="section-3-2-1"></span>

#### 3.2.1 Scaling Laws

We develop scaling laws [Hof22, Kap20] to determine the optimal model size for our flagship model given our pre-training compute budget. In addition to determining the optimal model size, a major challenge is to forecast the flagship model’s performance on downstream benchmark tasks, due to a couple of issues: (1) Existing scaling laws typically predict only next-token prediction loss rather than specific benchmark performance. (2) Scaling laws can be noisy and unreliable because they are developed based on pre-training runs conducted with small compute budgets [Wei22d].

To address these challenges, we implement a two-stage methodology to develop scaling laws that accurately predict downstream benchmark performance:

1. We first establish a correlation between the compute-optimal model’s negative log-likelihood on downstream tasks and the training FLOPs.
2. Next, we correlate the negative log-likelihood on downstream tasks with task accuracy, utilizing both the scaling law models and older models trained with higher compute FLOPs. In this step, we specifically leverage the Llama 2 family of models.

This approach enables us to predict downstream task performance given a specific number of training FLOPs for compute-optimal models. We use a similar method to select our pre-training data mix (see Section [3.4](#section-3-4)).

**Scaling law experiments.** Concretely, we construct our scaling laws by pre-training models using compute budgets between $6\times 10^{18}$ FLOPs and $10^{22}$ FLOPs. At each compute budget, we pre-train models ranging in size between 40M and 16B parameters, using a subset of model sizes at each compute budget. In these training runs, we use a cosine learning rate schedule with a linear warmup for 2,000 training steps. The peak learning rate is set between $2\times 10^{-4}$ and $4\times 10^{-4}$ depending on the size of the model. We set the cosine decay to 0.1 of the peak value. The weight decay at each step is set to 0.1 times the learning rate at that step. We use a fixed batch size for each compute scale, ranging between 250K and 4M.
<span id="figure-02"></span>

![Refer to caption](../../papers/llama-3/figure-02.png)

**Figure 2.** Scaling law IsoFLOPs curves between $6\times 10^{18}$ and $10^{22}$ FLOPs. The loss is the negative log-likelihood on a held-out validation set. We approximate measurements at each compute scale using a second degree polynomial.
<span id="figure-03"></span>

![Original paper Figure 3](../../papers/llama-3/figure-03.png)

**Figure 3.** Number of training tokens in identified compute-optimal models as a function of pre-training compute budget. We include the fitted scaling-law prediction as well. The compute-optimal models correspond to the parabola minimums in [Figure 3](#figure-03).

These experiments give rise to the IsoFLOPs curves in [Figure 2](#figure-02). The loss in these curves is measured on a separate validation set. We fit the measured loss values using a second-degree polynomial and identify the minimums of each parabola. We refer to minimum of a parabola as the *compute-optimal* model at the corresponding pre-training compute budget.

We use the compute-optimal models we identified this way to predict the optimal number of training tokens for a specific compute budget. To do so, we assume a power-law relation between compute budget, $C$, and the optimal number of training tokens, $N^{\star}(C)$:

$$
N^{\star}(C)=\mathrm{AC}^{\alpha}.
$$

We fit $A$ and $\alpha$ using the data from [Figure 3](#figure-03). We find that $(\alpha,A)=(0.53,0.29)$; the corresponding fit is shown in [Figure 3](#figure-03). Extrapolation of the resulting scaling law to $3.8\times 10^{25}$ FLOPs suggests training a 402B parameter model on 16.55T tokens.

An important observation is that IsoFLOPs curves become *flatter* around the minimum as the compute budget increases. This implies that performance of the flagship model is relatively robust to small changes in the trade-off between model size and training tokens. Based on this observation, we ultimately decided to train a flagship model with 405B parameters.

**Predicting performance on downstream tasks.** We use the resulting compute-optimal models to forecast the performance of the flagship Llama 3 model on benchmark data sets. First, we linearly correlate the (normalized) negative log-likelihood of correct answer in the benchmark and the training FLOPs. In this analysis, we use only the scaling law models trained up to $10^{22}$ FLOPs on the data mix described above. Next, we establish a sigmoidal relation between the log-likelihood and accuracy using both the scaling law models and Llama 2 models, which were trained using the Llama 2 data mix and tokenizer. We show the results of this experiment on the ARC Challenge benchmark in [Figure 4](#figure-04)). We find this two-step scaling law prediction, which extrapolates over four orders of magnitude, to be quite accurate: it only slightly underestimates the final performance of the flagship Llama 3 model.
<span id="figure-04"></span>

![Refer to caption](../../papers/llama-3/figure-04.png)

**Figure 4.** Scaling law forecast for ARC Challenge. *Left:* Normalized negative log-likelihood of the correct answer on the ARC Challenge benchmark as a function of pre-training FLOPs. *Right:* ARC Challenge benchmark accuracy as a function of the normalized negative log-likelihood of the correct answer. This analysis enables us to predict model performance on the ARC Challenge benchmark before pre-training commences. See text for details.
<span id="section-3-3"></span>

### 3.3 Infrastructure, Scaling, and Efficiency

We describe our hardware and infrastructure that powered Llama 3 405B pre-training at scale and discuss several optimizations that leads to improvements in training efficiency.
<span id="section-3-3-1"></span>

#### 3.3.1 Training Infrastructure

The Llama 1 and 2 models were trained on Meta’s AI Research SuperCluster [Lee22]. As we scaled further, the training for Llama 3 was migrated to Meta’s production clusters [Lee24b].This setup optimizes for production-grade reliability, which is essential as we scale up training.

**Compute.** Llama 3 405B is trained on up to 16K H100 GPUs, each running at 700W TDP with 80GB HBM3, using Meta’s Grand Teton AI server platform [Bow22]. Each server is equipped with eight GPUs and two CPUs. Within a server, the eight GPUs are connected via NVLink. Training jobs are scheduled using MAST [Cho24], Meta’s global-scale training scheduler.

**Storage.** Tectonic [Pan21], Meta’s general-purpose distributed file system, is used to build a storage fabric [Bat24] for Llama 3 pre-training. It offers 240 PB of storage out of 7,500 servers equipped with SSDs, and supports a sustainable throughput of 2 TB/s and a peak throughput of 7 TB/s. A major challenge is supporting the highly bursty checkpoint writes that saturate the storage fabric for short durations. Checkpointing saves each GPU’s model state, ranging from 1 MB to 4 GB per GPU, for recovery and debugging. We aim to minimize GPU pause time during checkpointing and increase checkpoint frequency to reduce the amount of lost work after a recovery.

**Network.** Llama 3 405B used RDMA over Converged Ethernet (RoCE) fabric based on the Arista 7800 and Minipack2 Open Compute Project [+4] OCP rack switches. Smaller models in the Llama 3 family were trained using Nvidia Quantum2 Infiniband fabric. Both RoCE and Infiniband clusters leverage 400 Gbps interconnects between GPUs. Despite the underlying network technology differences between these clusters, we tune both of them to provide equivalent performance for these large training workloads. We elaborate further on our RoCE network since we fully own its design.

- **Network topology.** Our RoCE-based AI cluster comprises 24K GPUs [+5] connected by a three-layer Clos network [Lee24b]. At the bottom layer, each rack hosts 16 GPUs split between two servers and connected by a single Minipack2 top-of-the-rack (ToR) switch. In the middle layer, 192 such racks are connected by Cluster Switches to form a pod of 3,072 GPUs with full bisection bandwidth, ensuring no oversubscription. At the top layer, eight such pods within the same datacenter building are connected via Aggregation Switches to form a cluster of 24K GPUs. However, network connectivity at the aggregation layer does not maintain full bisection bandwidth and instead has an oversubscription ratio of 1:7. Our model parallelism methods (see Section [3.3.2](#section-3-3-2)) and training job scheduler [Cho24] are all optimized to be aware of network topology, aiming to minimize network communication across pods.
- **Load balancing.** LLM training produces fat network flows that are hard to load balance across all available network paths using traditional methods such as Equal-Cost Multi-Path (ECMP) routing. To address this challenge, we employ two techniques. First, our collective library creates 16 network flows between two GPUs, instead of just one, thereby reducing the traffic per flow and providing more flows for load balancing. Second, our Enhanced-ECMP (E-ECMP) protocol effectively balances these 16 flows across different network paths by hashing on additional fields in the RoCE header of packets.
- **Congestion control.** We use deep-buffer switches in the spine [Gan24] to accommodate transient congestion and buffering caused by collective communication patterns. This setup helps limit the impact of persistent congestion and network back pressure caused by slow servers, which is common in training. Finally, better load balancing through E-ECMP significantly reduces the chance of congestion. With these optimizations, we successfully run a 24K GPU cluster without traditional congestion control methods such as Data Center Quantized Congestion Notification (DCQCN).
<span id="section-3-3-2"></span>

#### 3.3.2 Parallelism for Model Scaling

To scale training for our largest models, we use 4D parallelism—a combination of four different types of parallelism methods—to shard the model. This approach efficiently distributes computation across many GPUs and ensures each GPU’s model parameters, optimizer states, gradients, and activations fit in its HBM. Our implementation of 4D parallelism is illustrated in [Figure 5](#figure-05). It combines tensor parallelism (TP; [Kri12, Sho19, Kor22]), pipeline parallelism (PP; [Hua19b, Nar21, Lam23]), context parallelism (CP; [Liu23]), and data parallelism (DP; [Raj20a, Ren21, Zha23]).

Tensor parallelism splits individual weight tensors into multiple chunks on different devices. Pipeline parallelism partitions the model vertically into stages by layers, so that different devices can process in parallel different stages of the full model pipeline. Context parallelism divides the input context into segments, reducing memory bottleneck for very long sequence length inputs. We use fully sharded data parallelism [Raj20a, Ren21, Zha23], which shards the model, optimizer, and gradients while implementing data parallelism which processes data in parallel on multiple GPUs and synchronizes after each training step. Our use of FSDP for Llama 3 shards optimizer states and gradients, but for model shards we do not reshard after forward computation to avoid an extra all-gather communication during backward passes.

**GPU utilization.** Through careful tuning of the parallelism configuration, hardware, and software, we achieve an overall BF16 Model FLOPs Utilization (MFU; [Cho22a]) of 38-43% for the configurations shown in [Table 4](#table-04). The slight drop in MFU to 41% on 16K GPUs with DP=128 compared to 43% on 8K GPUs with DP=64 is due to the lower batch size per DP group needed to keep the global tokens per batch constant during training.
<span id="table-04"></span>

![Original paper Table 4](../../papers/llama-3/table-04.png)

**Table 4.** Scaling configurations and MFU for each stage of Llama 3 405B pre-training. See text and [Figure 5](#figure-05) for descriptions of each type of parallelism.
<span id="figure-05"></span>

![Refer to caption](../../papers/llama-3/figure-05.png)

**Figure 5.** Illustration of 4D parallelism. GPUs are divided into parallelism groups in the order of $[\mathrm{TP}, \mathrm{CP}, \mathrm{PP}, \mathrm{DP}]$, where DP stands for FSDP. In this example, 16 GPUs are configured with a group size of $|\mathrm{TP}|=2$, $|\mathrm{CP}|=2$, $|\mathrm{PP}|=2$, and $|\mathrm{DP}|=2$. A GPU’s position in 4D parallelism is represented as a vector, $[D_1, D_2, D_3, D_4]$, where $D_{i}$ is the index on the $i$-th parallelism dimension. In this example, GPU0 $[\mathrm{TP}0, \mathrm{CP}0, \mathrm{PP}0, \mathrm{DP}0]$ and GPU1 $[\mathrm{TP}1, \mathrm{CP}0, \mathrm{PP}0, \mathrm{DP}0]$ are in the same TP group, GPU0 and GPU2 are in the same CP group, GPU0 and GPU4 are in the same PP group, and GPU0 and GPU8 are in the same DP group.

**Pipeline parallelism improvements.** We encountered several challenges with existing implementations:

- **Batch size constraint.** Current implementations have constraints on supported batch size per GPU, requiring it to be divisible by the number of pipeline stages. For the example in [Figure 6](#figure-06), the depth-first schedule (DFS) of pipeline parallelism [Nar21] requires $N=\mathrm{PP}=4$, while the breadth-first schedule (BFS; [Lam23]) requires $N=M$, where $M$ is the total number of micro-batches and $N$ is the number of contiguous micro-batches for the same stage’s forward or backward. However, pre-training often needs flexibility to adjust batch size.
- **Memory imbalance.** Existing pipeline parallelism implementations lead to imbalanced resource consumption. The first stage consumes more memory due to the embedding and the warm-up micro-batches.
- **Computation imbalance.** After the last layer of the model, we need to calculate output and loss, making this stage the execution latency bottleneck.

To address these issues, we modify our pipeline schedule as shown in [Figure 6](#figure-06), which allows setting $N$ flexibly—in this case $N=5$, which can run a arbitrary number of micro-batches in each batch. This allows us to run: (1) fewer micro-batches than the number of stages when we have batch size limit at large scale; or (2) more micro-batches to hide point-to-point communication, finding a sweet spot between DFS and breadth first schedule (BFS) for the best communication and memory efficiency. To balance the pipeline, we reduce one Transformer layer each from the first and the last stages, respectively. This means that the first model chunk on the first stage has only the embedding, and the last model chunk on the last stage has only output projection and loss calculation. To reduce pipeline bubbles, we use an interleaved schedule [Nar21] with $V$ pipeline stages on one pipeline rank. Overall pipeline bubble ratio is $\frac{\mathrm{PP}-1}{V*M}$. Further, we adopt asynchronous point-to-point communication in PP, which considerably speeds up training, especially in cases when the document mask introduces extra computation imbalance. We enable TORCH\_NCCL\_AVOID\_RECORD\_STREAMS to reduce memory usage from asynchronous point-to-point communication. Finally, to reduce memory cost, based on detailed memory allocation profiling, we proactively deallocate tensors that will not be used for future computation, including the input and output tensors of each pipeline stage, that will not be used for future computation. With these optimizations, we could pre-train Llama 3 on sequences of 8K tokens without activation checkpointing.
<span id="figure-06"></span>

![Refer to caption](../../papers/llama-3/figure-06.png)

**Figure 6.** Illustration of pipeline parallelism in Llama 3. Pipeline parallelism partitions eight pipeline stages (0 to 7) across four pipeline ranks (PP ranks 0 to 3), where the GPUs with rank 0 run stages 0 and 4, the GPUs with P rank 1 run stages 1 and 5, *etc*. The colored blocks (0 to 9) represent a sequence of micro-batches, where $M$ is the total number of micro-batches and $N$ is the number of continuous micro-batches for the same stage’s forward or backward. Our key insight is to make $N$ tunable.

**Context parallelism for long sequences.** We utilize context parallelism (CP) to improve memory efficiency when scaling the context length of Llama 3 and enable training on extremely long sequences up to 128K in length. In CP, we partition across the sequence dimension, and specifically we partition the input sequence into $2\times\mathrm{CP}$ chunks so each CP rank receives two chunks for better load balancing. The $i$-th CP rank received both the $i$-th and the $(2\times\mathrm{CP}-1-i)$-th chunks.

Different from existing CP implementations that overlap communication and computation in a ring-like structure [Liu23], our CP implementation adopts an all-gather based method where we first all-gather the key (K) and value (V) tensors, and then compute attention output for the local query (Q) tensor chunk. Although the all-gather communication latency is exposed in the critical path, we still adopt this approach for two main reasons: (1) it is easier and more flexible to support different types of attention masks in all-gather based CP attention, such as the document mask; and (2) the exposed all-gather latency is small as the communicated K and V tensors are much smaller than Q tensor due to the use of GQA [Ain23]. Hence, the time complexity of attention computation is an order of magnitude larger than all-gather ($O(S^{2})$ versus $O(S)$, where $S$ represents the sequence length in the full causal mask), making the all-gather overhead negligible.

**Network-aware parallelism configuration.** The order of parallelism dimensions, $[\mathrm{TP}, \mathrm{CP}, \mathrm{PP}, \mathrm{DP}]$, is optimized for network communication. The innermost parallelism requires the highest network bandwidth and lowest latency, and hence is usually constrained to within the same server. The outermost parallelism may spread across a multi-hop network and should tolerate higher network latency. Therefore, based on the requirements for network bandwidth and latency, we place parallelism dimensions in the order of $[\mathrm{TP}, \mathrm{CP}, \mathrm{PP}, \mathrm{DP}]$. DP (*i.e.*, FSDP) is the outermost parallelism because it can tolerate longer network latency by asynchronously prefetching sharded model weights and reducing gradients. Identifying the optimal parallelism configuration with minimal communication overhead while avoiding GPU memory overflow is challenging. We develop a memory consumption estimator and a performance-projection tool which helped us explore various parallelism configurations and project overall training performance and identify memory gaps effectively.

**Numerical stability.** By comparing training loss between different parallelism setups, we fixed several numerical issues that impact training stability. To ensure training convergence, we use FP32 gradient accumulation during backward computation over multiple micro-batches and also reduce-scatter gradients in FP32 across data parallel workers in FSDP. For intermediate tensors, *e.g.*, vision encoder outputs, that are used multiple times in the forward computation, the backward gradients are also accumulated in FP32.
<span id="section-3-3-3"></span>

#### 3.3.3 Collective Communication

Our collective communication library for Llama 3 is based on a fork of Nvidia’s NCCL library, called NCCLX. NCCLX significantly improves the performance of NCCL, especially for higher latency networks. Recall that the order of parallelism dimensions is $[\mathrm{TP}, \mathrm{CP}, \mathrm{PP}, \mathrm{DP}]$, where DP corresponds to FSDP. The outermost parallelism dimensions, PP and DP, may communicate through a multi-hop network, with latency up to tens of microseconds. The original NCCL collectives—all-gather and reduce-scatter in FSDP, and point-to-point in PP—require data chunking and staged data copy. This approach incurs several inefficiencies, including (1) requiring a large number of small control messages to be exchanged over the network to facilitate data transfer, (2) extra memory-copy operations, and (3) using extra GPU cycles for communication. For Llama 3 training, we address a subset of these inefficiencies by tuning chunking and data transfer to fit our network latencies, which can be as high as tens of microseconds for a large cluster. We also allow small control messages to traverse our network at a higher priority, especially avoiding being head-of-line blocked in deep-buffer core switches. Our ongoing work for future Llama versions involves making deeper changes in NCCLX to holistically address all the aforementioned problems.
<span id="section-3-3-4"></span>

#### 3.3.4 Reliability and Operational Challenges

The complexity and potential failure scenarios of 16K GPU training surpass those of much larger CPU clusters that we have operated. Moreover, the synchronous nature of training makes it less fault-tolerant—a single GPU failure may require a restart of the entire job. Despite these challenges, for Llama 3, we achieved higher than 90% effective training time while supporting automated cluster maintenance, such as firmware and Linux kernel upgrades [Vig24], which resulted in at least one training interruption daily. The effective training time measures the time spent on useful training over the elapsed time.

During a 54-day snapshot period of pre-training, we experienced a total of 466 job interruptions. Of these, 47 were planned interruptions due to automated maintenance operations such as firmware upgrades or operator-initiated operations like configuration or dataset updates. The remaining 419 were unexpected interruptions, which are classified in [Table 5](#table-05). Approximately 78% of the unexpected interruptions are attributed to confirmed hardware issues, such as GPU or host component failures, or suspected hardware-related issues like silent data corruption and unplanned individual host maintenance events. GPU issues are the largest category, accounting for 58.7% of all unexpected issues. Despite the large number of failures, significant manual intervention was required only three times during this period, with the rest of issues handled by automation.
<span id="table-05"></span>

![Original paper Table 5](../../papers/llama-3/table-05.png)

**Table 5.** Root-cause categorization of unexpected interruptions during a 54-day period of Llama 3 405B pre-training. About 78% of unexpected interruptions were attributed to confirmed or suspected hardware issues.

To increase the effective training time, we reduced job startup and checkpointing time, and developed tools for fast diagnosis and problem resolution. We extensively use PyTorch’s built-in NCCL flight recorder [Ans24], a feature that captures collective metadata and stack traces into a ring buffer, and hence allowing us to diagnose hangs and performance issues quickly at scale, particularly with regard to NCCLX. Using this, we efficiently record every communication event and the duration of each collective operation, and also automatically dump tracing data on NCCLX watchdog or heartbeat timeout. We enable more computationally intensive tracing operations and metadata collection selectively as needed live in production through online configuration changes [Tan15] without needing a code release or job restart.

Debugging issues in large-scale training is complicated by the mixed use of NVLink and RoCE in our network. Data transfer over NVLink typically occurs through load/store operations issued by CUDA kernels, and failures in either the remote GPU or NVLink connectivity often manifest as stalled load/store operations within CUDA kernels without returning a clear error code. NCCLX enhances the speed and accuracy of failure detection and localization through a tight co-design with PyTorch, allowing PyTorch to access NCCLX’s internal state and track relevant information. While stalls due to NVLink failures cannot be completely prevented, our system monitors the state of the communication library and automatically times out when such a stall is detected. Additionally, NCCLX traces the kernel and network activities of each NCCLX communication and provides a snapshot of the failing NCCLX collective’s internal state, including finished and pending data transfers between all ranks. We analyze this data to debug NCCLX scaling issues.

Sometimes, hardware issues may cause still-functioning but slow stragglers that are hard to detect. Even a single straggler can slow down thousands of other GPUs, often appearing as functioning but slow communications. We developed tools to prioritize potentially problematic communications from selected process groups. By investigating just a few top suspects, we were usually able to effectively identify the stragglers.

One interesting observation is the impact of environmental factors on training performance at scale. For Llama 3 405B , we noted a diurnal 1-2% throughput variation based on time-of-day. This fluctuation is the result of higher mid-day temperatures impacting GPU dynamic voltage and frequency scaling.

During training, tens of thousands of GPUs may increase or decrease power consumption at the same time, for example, due to all GPUs waiting for checkpointing or collective communications to finish, or the startup or shutdown of the entire training job. When this happens, it can result in instant fluctuations of power consumption across the data center on the order of tens of megawatts, stretching the limits of the power grid. This is an ongoing challenge for us as we scale training for future, even larger Llama models.
<span id="section-3-4"></span>

### 3.4 Training Recipe

The recipe used to pre-train Llama 3 405B consists of three main stages: (1) initial pre-training, (2) long-context pre-training, and (3) annealing. The three stages are described separately below. We use similar recipes to pre-train the 8B and 70B models.
<span id="section-3-4-1"></span>

#### 3.4.1 Initial Pre-Training

We pre-train Llama 3 405B using AdamW with a peak learning rate of $8\times 10^{-5}$, a linear warm up of 8,000 steps, and a cosine learning rate schedule decaying to $8\times 10^{-7}$ over 1,200,000 steps. We use a lower batch size early in training to improve training stability, and increase it subsequently to improve efficiency. Specifically, we use an initial batch size of 4M tokens and sequences of length 4,096, and double these values to a batch size of 8M sequences of 8,192 tokens after pre-training 252M tokens. We double the batch size again to 16M after pre-training on 2.87T tokens. We found this training recipe to be very stable: we observed few loss spikes and did not require interventions to correct for model training divergence.

**Adjusting the data mix.** We made a several adjustments to the pre-training data mix during training to improve model performance on particular downstream tasks. In particular, we increased the percentage of non-English data during pre-training to improve the multilingual performance of Llama 3. We also upsample mathematical data to improve the model’s mathematical reasoning performance, we added more recent web data in the later stages of pre-training to advance the model’s knowledge cut-off, and we downsampled subsets of the pre-training data that were later identified as being lower quality.
<span id="section-3-4-2"></span>

#### 3.4.2 Long Context Pre-Training

In the final stages of pre-training, we train on long sequences to support context windows of up to 128K tokens. We do not train on long sequences earlier because the compute in self-attention layers grows quadratically in the sequence length. We increase the supported context length in increments, pre-training until the model has successfully adapted to the increased context length. We assess successful adaptation by measuring whether (1) model performance on short-context evaluations has recovered completely and (2) the model perfectly solves “needle in a haystack” tasks up to that length. In Llama 3 405B pre-training, we increased context length gradually in six stages, starting from the original 8K context window and ending in the final 128K context window. This long-context pre-training stage was performed using approximately 800B training tokens.
<span id="section-3-4-3"></span>

#### 3.4.3 Annealing

During pre-training on the final 40M tokens, we linearly annealed the learning rate to 0, maintaining a context length of 128K tokens. During this annealing phase, we also adjusted the data mix to upsample data sources of very high quality; see Section [3.1.3](#section-3-1-3). Finally, we compute the average of model checkpoints ([Pol91] averaging) during annealing to produce the final pre-trained model.
<span id="section-4"></span>

## 4 Post-Training

We produce the aligned Llama 3 models by applying several rounds of post-training, [+6] or aligning the model with human feedback [Ouy22, Raf24] on top of a pre-trained checkpoint. Each round of post-training involves supervised finetuning (SFT) followed by Direct Preference Optimization [Raf24] on examples collected either via human annotations or generated synthetically. Our post-training modeling and data approaches are described in Sections [4.1](#section-4-1) and  [4.2](#section-4-2) respectively. We further detail custom data curation strategies to improve the reasoning, coding, factuality, multilingual, tool use, long context, and precise instruction following in Section [4.3](#section-4-3).
<span id="figure-07"></span>

![Refer to caption](../../papers/llama-3/figure-07.png)

**Figure 7.** Illustration of the overall post-training approach for Llama 3. Our post-training strategy involves rejection sampling, supervised finetuning, and direct preference optimization. See text for details.
<span id="section-4-1"></span>

### 4.1 Modeling

The backbone of our post-training strategy is a reward model and a language model. We first train a reward model on top of the pre-trained checkpoint using human-annotated preference data (see Section [4.1.2](#section-4-1-2)). We then finetune pre-trained checkpoints with supervised finetuning (SFT; see Section [4.1.3](#section-4-1-3)), and further align the checkpoints with Direct Preference Optimization (DPO; see Section [4.1.4](#section-4-1-4)). This process is illustrated in [Figure 7](#figure-07). Unless otherwise noted, our modeling procedure applies to Llama 3 405B, and we refer to Llama 3 405B as Llama 3 for simplicity.
<span id="section-4-1-1"></span>

#### 4.1.1 Chat Dialog Format

To tune LLMs for human-AI interaction, we need to define a chat dialog protocol for the model to understand human instructions and perform conversational tasks. Compared to its predecessor, Llama 3 has new capabilities such as tool use (Section [4.3.5](#section-4-3-5)) which may require generating multiple messages and sending them to different locations (e.g., user, ipython) within a single dialog turn. To support this, we design a new multi-message chat protocol which uses various special header and termination tokens. The header tokens are used to indicate the source and destination of each message in a conversation. Similarly, the termination tokens indicate when it is the time to alternate between human and AI to speak.
<span id="section-4-1-2"></span>

#### 4.1.2 Reward Modeling

We train a reward model (RM) covering different capabilities on top of the pre-trained checkpoint. The training objective is the same as Llama 2 except that we remove the margin term in the loss, as we observe diminishing improvements after data scaling. Following Llama 2, we use all of our preference data for reward modeling after filtering out samples with similar responses. In addition to standard preference pair of (chosen, rejected) response, annotations also create a third “edited response” for some prompts, where the chosen response from the pair is further edited for improvement (see Section [4.2.1](#section-4-2-1)). Hence, each preference ranking sample has two or three responses with clear ranking (*edited* > *chosen* > *rejected*). We concatenate the prompt and multiple responses into a single row during training with responses randomly shuffled. This is an approximation to the standard scenario of putting the responses in separate rows and computing the scores, but in our ablations, this approach improves training efficiency without a loss in accuracy.
<span id="section-4-1-3"></span>

#### 4.1.3 Supervised Finetuning

The reward model is then used to perform rejection sampling on our human annotation prompts, the details of which are described in Section [4.2](#section-4-2). Together with this rejection-sampled data and other data sources (including synthetic data), we finetune the pre-trained language model using a standard cross entropy loss on the target tokens (while masking loss on prompt tokens). More details about the data mix can be found in Section [4.2](#section-4-2). We refer to this stage as *supervised finetuning* [Wei22c, San22, Wan22], even though many of the training targets are model-generated. Our largest models are finetuned with a learning rate of $10^{-5}$ over the course of 8.5K to 9K steps. We found these hyperparameter settings to work well across different rounds and data mixes.
<span id="section-4-1-4"></span>

#### 4.1.4 Direct Preference Optimization

We further train our SFT models with Direct Preference Optimization [Raf24] for human preference alignment. For training, we primarily use the most recent batches of preference data collected using the best performing models from the previous alignment rounds. As a result, our training data conforms better to the distribution of the policy model that is being optimized in each round. We also explored on-policy algorithms such as PPO [Sch17], but found that DPO required less compute for large-scale models and performed better, especially on instruction following benchmarks like IFEval [Zho23a]. For Llama 3, we use a learning rate of $10^{-5}$ and set the $\beta$ hyper-parameter to be 0.1. In addition, we apply the following algorithmic modifications to DPO:

- **Masking out formatting tokens in DPO loss:** We mask out special formatting tokens including header and termination tokens (described in Section [4.1.1](#section-4-1-1)) from both chosen and rejected responses in the loss to stabilize DPO training. We observe that having these tokens contribute to the loss may lead to undesired model behaviors such as tail repetition or abruptly generating termination tokens. We hypothesize that this is due to the contrastive nature of the DPO loss – the presence of common tokens in both chosen and rejected responses leads to a conflicting learning objective as the model needs to increase and reduce the likelihood of these tokens simultaneously.
- **Regularization with NLL loss:** We add an additional negative log-likelihood (NLL) loss term with a scaling coefficient of $0.2$ on the chosen sequences, similar to [Pan24b]. This helps further stabilize DPO training by maintaining desired formatting for generation and preventing the decrease of log probability of chosen responses [Pan24b, Pal24].
<span id="section-4-1-5"></span>

#### 4.1.5 Model Averaging

Finally, we average models obtained from experiments using various versions of data or hyperparameters at each RM, SFT, or DPO stage [Izm19, Wor22, Li22].
<span id="section-4-1-6"></span>

#### 4.1.6 Iterative Rounds

Following Llama 2, we apply the above methods in six rounds. In each cycle, we collect new preference annotations and SFT data, sampling synthetic data from the latest models.
<span id="section-4-2"></span>

### 4.2 Post-training Data

The post-training data composition plays a critical role in the usefulness and behavior of language models. In this section, we discuss our human annotation procedures and preference data collection (Section [4.2.1](#section-4-2-1)), the composition of our SFT data (Section [4.2.2](#section-4-2-2)), and methods for data quality control and cleaning (Section [4.2.3](#section-4-2-3)).
<span id="section-4-2-1"></span>

#### 4.2.1 Preference Data

Our preference data annotation process is similar to Llama 2. We deploy multiple models for annotation after each round and sample two responses from two different models for each user prompt. These models can be trained with different data mixes and alignment recipes, allowing for different capability strength (*e.g.*, code expertise) and increased data diversity. We ask annotators to rate the strength of their preference by categorizing it into one of four levels, based on how much more they prefer the chosen response over the rejected one: significantly better, better, slightly better, or marginally better. We also incorporate an editing step after preference ranking to encourage annotators to further improve the preferred response. Annotators edit the chosen response directly or prompt the model with feedback to refine its own response. Consequently, a portion of our preference data has three responses ranked (*edited* > *chosen* > *rejected*).

In [Table 6](#table-06), we report the statistics of preference annotations that we use for Llama 3 training. General English covers multiple subcategories such as knowledge-based question and answering or precise instruction-following, which fall outside the scope of specific capabilities. Compared to Llama 2, we observe an increase in the average length of prompt and response, suggesting that we train Llama 3 on more complex tasks. In addition, we implement a quality analysis and human evaluation process to rigorously assess the data collected, allowing us to refine our prompts and provide systematic, actionable feedback to annotators. For example, as Llama 3 improves after each round, we increase prompt complexity accordingly to target areas where the model lags.

In each round of post-training, we use all the preference data that is available at the time for reward modeling, while only using the latest batches from various capabilities for DPO training. For both reward modeling and DPO, we use samples that are labeled as the chosen response being significantly better or better than the rejected counterpart for training and discard samples with similar responses.
<span id="table-06"></span>

![Original paper Table 6](../../papers/llama-3/table-06.png)

**Table 6.** Statistics of human preference data. We list statistics of the internally collected human preference data used for Llama 3 alignment. We ask annotators to perform multi-turn dialogues with the models and make comparisons among responses at each turn. In post-processing, we split each dialogue to multiple examples at a turn level. Each example consists of a prompt (including previous dialog if available) and a response (e.g., chosen or rejected response).
<span id="section-4-2-2"></span>

#### 4.2.2 SFT Data

Our finetuning data is largely comprised of the following sources:

- Prompts from our human annotation collection with rejection-sampled responses.
- Synthetic data targeting specific capabilities (see Section [4.3](#section-4-3) for more details).
- Small amounts of human-curated data (see Section [4.3](#section-4-3) for more details).

As our post-training rounds progress, we develop stronger Llama 3 variants that we use to collect larger datasets that cover a wide range of complex capabilities. In this section, we discuss the details for the rejection-sampling procedure and overall composition of our final SFT datamix.

**Rejection sampling.** During rejection sampling (RS), for each prompt collected during human annotation (Section [4.2.1](#section-4-2-1)) we sample $K$ (typically between 10 and 30) outputs from the latest chat model policy (usually the best performing checkpoint from the previous post-training iteration, or the best performing checkpoint for a particular capability) and use our reward model to select the best candidate, consistent with [Bai22]. In later rounds of post-training, we introduce system prompts to steer RS responses to conform with desirable tone, style, or formatting, which might be different for different capabilities.

To increase the efficiency of rejection sampling, we adopt PagedAttention [Kwo23]. PagedAttention enhances memory efficiency through dynamic key-value cache allocation. It supports arbitrary output lengths by dynamically scheduling requests based on the current cache capacity. Unfortunately, this carries the risk of swap-out when running out of memory. To eliminate such swap overhead, we define a maximum output length and perform a request only if sufficient memory is available to fit an output with that length. PagedAttention also enables us to share the key-value cache pages for a prompt across all corresponding outputs. Together, this leads to a throughput improvement of over $2\times$ during rejection sampling.

**Overall data composition.** [Table 7](#table-07) shows data statistics for each broad category of our “helpfulness” mix. While SFT and preference data contain overlapping domains, they are curated differently, yielding distinct count statistics. In Section [4.2.3](#section-4-2-3) we describe techniques for categorizing topic, complexity, and quality of our data samples. In each round of post-training, we adjust our overall data mix carefully across these axes to tune performance across a wide range of benchmarks. Our final data mix epochs multiple times on some high quality sources and downsamples others.
<span id="table-07"></span>

![Original paper Table 7](../../papers/llama-3/table-07.png)

**Table 7.** Statistics of SFT data. We list internally collected SFT data used for Llama 3 alignment. Each SFT example consists of a context (i.e., all conversation turns except the last one) and a final response.
<span id="section-4-2-3"></span>

#### 4.2.3 Data Processing and Quality Control

Given that most of our training data is *model-generated*, it requires careful cleaning and quality control.

**Data cleaning.** In the early rounds, we observed a number of undesirable patterns common in our data, such as excessive use of emojis or exclamation points. Therefore, we implement a series of rule-based data removal and modification strategies to filter or clean problematic data. For example, to mitigate overly-apologetic tonal issues, we identify overused phrases (such as “I’m sorry” or “I apologize”) and carefully balance the proportion of such samples in our dataset.

**Data pruning.** We also apply a collection of model-based techniques to remove low-quality training samples and improve overall model performance:

- **Topic classification:** We first finetune Llama 3 8B into a topic classifier, and perform inference over all data to classify it into both coarsely-grained buckets (“mathematical reasoning”) and fine-grained buckets (“geometry and trigonometry”).
- **Quality scoring:** We use both reward model and Llama-based signals to obtain a quality score for each sample. For an RM-based score, we consider data that is in the top quartile of RM scores as high quality. For a Llama-based score, we prompt Llama 3 checkpoint to rate each sample on a three-point scale for general English data (accuracy, instruction following, and tone/presentation) and a two-point scale for coding data (bug identification and user intention), and consider samples that obtain the maximum score as high quality. The RM and Llama-based scores have high disagreement rates, and we find that combining these signals yield the best recall on our internal test set. Ultimately, we select examples that are marked as high quality by the RM *or* the Llama-based filter.
- **Difficulty scoring:** Because we are also interested in prioritizing examples that are more complex for the model, we score data using two measures of difficulty: Instag [Lu23] and Llama-based scoring. For Instag, we prompt Llama 3 70B to perform intention tagging of SFT prompts, where more intentions implies more complexity. We also prompt Llama 3 to measure the difficulty [Liu24j] of dialogs on a three-point scale.
- **Semantic deduplication:** Finally, we perform semantic deduplication [Abb23, Liu24j]. We first cluster complete dialogs using RoBERTa [Liu19a] and within each cluster sort them by quality score $\times$ difficulty score. We then do greedy selection by iterating through all sorted examples, and only keeping the ones that have maximum cosine similarity less than a threshold to the examples seen so far in the cluster.
<span id="section-4-3"></span>

### 4.3 Capabilities

We highlight special efforts to improve performance for specific capabilities such as code (Section [4.3.1](#section-4-3-1)), multilinguality (Section [4.3.2](#section-4-3-2)), math and reasoning (Section [4.3.3](#section-4-3-3)), long context (Section [4.3.4](#section-4-3-4)), tool use (Section [4.3.5](#section-4-3-5)), factuality (Section [4.3.6](#section-4-3-6)), and steerability (Section [4.3.7](#section-4-3-7)).
<span id="section-4-3-1"></span>

#### 4.3.1 Code

LLMs for code have received significant attention since the release of Copilot and Codex [Che21]. Developers are now widely using these models to generate code snippets, debug, automate tasks, and improve code quality. For Llama 3, we target improving and evaluating code generation, documentation, debugging, and review capabilities for the following high priority programming languages: Python, Java, Javascript, C/C++, Typescript, Rust, PHP, HTML/CSS, SQL, bash/shell. Here, we present our work on improving these coding capabilities via training a code expert, generating synthetic data for SFT, improving formatting with system prompt steering, and creating quality filters to remove bad samples from our training data.

**Expert training.** We train a code expert which we use to collect high quality human annotations for code throughout subsequent rounds of post-training. This is accomplished by branching the main pre-training run and continuing pre-training on a 1T token mix of mostly (>85%) code data. Continued pre-training on domain-specific data has been shown to be effective for improving performance in a specific domain [Gur20]. We follow a recipe similar to that of CodeLlama [Roz23]. For the last several thousand steps of training we perform long-context finetuning (LCFT) to extend the expert’s context length to 16K tokens on a high quality mix of repo-level code data. Finally, we follow the similar post-training modeling recipes described in Section [4.1](#section-4-1) to align this model, except with SFT and DPO data mixes primarily targeting code. This model is also used for rejection sampling (Section [4.2.2](#section-4-2-2)) for coding prompts.

**Synthetic data generation.** During development, we identified key issues in code generation, including difficulty in following instructions, code syntax errors, incorrect code generation, and difficulty in fixing bugs. While intensive human annotation could theoretically resolve these issues, synthetic data generation offers a complementary approach at a lower cost and higher scale, unconstrained by the expertise level of annotators. As such, we use Llama 3 and the code expert to generate a large quantity of synthetic SFT dialogs.

We describe three high-level approaches for generating synthetic code data. In total, we generate over $2.7$M synthetic examples which were used during SFT.

1. Synthetic data generation: execution feedback. The 8B and 70B models show significant performance improvements when trained on data generated by a larger, more competent model. However, our initial experiments revealed that training Llama 3 405B on its own generated data is not helpful (and can even degrade performance). To address this limitation, we introduced execution feedback as a source of truth, enabling the model to learn from its mistakes and stay on track. In particular, we generate large dataset of approximately one million synthetic coding dialogues using the following process:

    - Problem description generation: First, we generate a large collection of programming problem descriptions that span a diverse range of topics, including those in the long tail distribution. To achieve this diversity, we sample random code snippets from various sources and prompt the model to generate programming problems inspired by these examples. This allowed us to tap into a wide range of topics and create a comprehensive set of problem descriptions [Wei24a].
    - Solution generation: Then, we prompt Llama 3 to solve each problem in a given programming language. We observe that adding general rules of good programming to the prompt improves the generated solution quality. Also, we find it is helpful to require the model to explain its thought process in comments.
    - Correctness analysis: After generating a solution, it is crucial to recognize that its correctness is not guaranteed, and including incorrect solutions in the finetuning dataset could harm the model’s quality. While we do not ensure complete correctness, we develop methods to approximate it. To achieve this, we extract the source code from the generated solution and applied a combination of static and dynamic analysis techniques to test its correctness, including:
        - Static analysis: We run all generated code through a parser and a linter to ensure syntactic correctness, catching errors such as syntax errors, use of uninitialized variables or non-imported functions, code style issues, typing errors, and others.
        - Unit test generation and execution: For each problem and solution, we prompt the model to generate unit tests, executed in a containerized environment together with the solution, catching run-time execution errors and some semantic errors.
    - Error feedback and iterative self-correction: When a solution fails at any step, we prompt the model to revise it. The prompt included the original problem description, the faulty solution, and feedback from the parser/linter/tester (stdout, stderr/ and return code). After a unit test execution failure, the model could either fix the code to pass the existing tests or modify its unit tests to accommodate the generated code. Only dialogs that pass all checks are included in the final dataset, used for supervised finetuning (SFT). Notably, we observed that about 20% of solutions were initially incorrect but self-corrected, indicating that the model learned from the execution feedback and improved its performance.
    - Fine-tuning and iterative improvement: The finetuning process is conducted over multiple rounds, with each round building on the previous one. After each round, the model is improved, generating higher-quality synthetic data for the next round. This iterative process allows for progressive refinement and enhancement of the model’s performance.

2. Synthetic data generation: programming language translation. We observe a performance gap between major programming languages (*e.g.*, Python/C++) and less common ones (*e.g.*, Typescript/PHP). This is not surprising as we have less training data for less common programming languages. To mitigate this, we supplement our existing data by *translating* data from common programming languages to less common languages (similar to [Che23a] in the context of reasoning). This is achieved by prompting Llama 3 and ensuring quality via syntax parsing, compilation, and execution. [Figure 8](#figure-08) demonstrates an example of synthetic PHP code translated from Python. This improves performance significantly for less common languages as measured by the MultiPL-E [Cas23] benchmark.
<span id="figure-08"></span>

    ![Refer to caption](../../papers/llama-3/figure-08.png)

    **[Figure 8](#figure-08).** Code translation example. We display an example of using Llama 3 to translate Python code (left) to PHP code (right) to augment our SFT dataset with a wider range of programming languages.

3. Synthetic data generation: backtranslation. To improve certain coding capabilities (e.g., documentation, explanations) where execution feedback is less informative for determining quality, we employ an alternative multi-step approach. Using this procedure, we generated approximately 1.2M synthetic dialogs related to code explanation, generation, documentation, and debugging. Beginning with code snippets from a variety of languages in our pre-training data:

    - Generate: We prompt Llama 3 to generate data that represents our target capability (e.g., we add comments and docstrings for the code snippet, or we ask the model to explain a piece of code).
    - Backtranslate: We then prompt the model to “backtranslate” the synthetically generated data to the original code (e.g., we prompt the model to generate code only from its documentation, or we ask the model to generate code only from its explanation).
    - Filter: Using the original code as a reference, we prompt the Llama 3 to determine the quality of the output (e.g., we ask the model how faithful the backtranslated code is to the original). We then use the generated examples that have the highest self-verification scores in SFT.

**System prompt steering during rejection sampling.** During the rejection sampling process, we used code specific system prompts to improve code readability, documentation, thoroughness, and specificity. Recall, from Section [7](#table-07) this data is used to finetune the language model. [Figure 9](#figure-09) shows an example of how the system prompt helps improve the generated code quality — it adds necessary comments, uses more informative variable names, saves memory, etc.
<span id="figure-09"></span>

![Refer to caption](../../papers/llama-3/figure-09.png)

**Figure 9.** Improving generated code quality with system prompts. *Left:* without system prompt *Right:* with system prompt.

**Filtering training data with execution and model-as-judge signals.** As described in Section [4.2.3](#section-4-2-3), we occasionally encounter quality issues in our rejection-sampled data, such as code blocks containing bugs. Detecting these issues in our rejection-sampled data is not as straightforward as it is for our *synthetic code data*, as the rejection-sampled responses typically contain a mix of natural language and code for which the code may not always be expected to be executable. (For example, user prompts may explicitly ask for pseudo-code or edits to only a very small snippet of an executable program.) To address this, we utilize the “model-as-judge” approach, where earlier versions of Llama 3 assess and assign a binary (0/1) score based on two criteria: code correctness and code style. We retain only those samples that achieve a perfect score of 2. Initially, this stringent filtering led to a regression in downstream benchmark performance, primarily because it disproportionately removed examples with challenging prompts. To counteract this, we strategically revise the responses of some coding data categorized as most challenging until they met the Llama-based “model-as-judge” criteria. By refining these challenging problems, the coding data achieves a balance between quality and difficulty, resulting in optimal downstream performance.
<span id="section-4-3-2"></span>

#### 4.3.2 Multilinguality

We describe how we improve Llama 3’s multilingual capabilities, including training an expert specialized on substantially more multilingual data, sourcing and generating high quality multilingual instruction tuning data for German, French, Italian, Portuguese, Hindi, Spanish, and Thai, and tackling specific challenges of multilingual language steering to enhance the overall performance of our model.

**Expert training.** Our Llama 3 pre-training data mix contains significantly more English tokens than non-English tokens. To collect higher quality human annotations in non-English languages, we train a multilingual expert by branching off the pre-training run and continuing to pre-train on a data mix that consists of $90$% multilingual tokens. We then perform post-training on this expert following Section [4.1](#section-4-1). This expert model is then used to collect higher quality annotations in non-English languages until pre-training was fully complete.

**Multilingual data collection.** Our multilingual SFT data is derived primarily from sources described below. The overall distribution is 2.4% human annotations, 44.2% data from other NLP tasks, 18.8% rejection sampled data, and 34.6% translated reasoning data.

- **Human annotations:** We collect high-quality, manually annotated data from linguists and native speakers. These annotations mostly consist of open-ended prompts that represent real world use cases.
- **Data from other NLP tasks:** To further augment, we use multilingual training data from other tasks and rewrite into dialog format. For example, we use data from exams-qa [Har20] and Conic10k [Wu23]. To improve language alignment, we also use parallel texts from GlobalVoices [Pro16] and Wikimedia [Tie12]. We use LID based filtering and Blaser2.0 [Com23] to remove low quality data. For parallel text data, instead of using the bitext pairs directly, we apply a multilingual template inspired by [Wei22c] to better simulate real-life conversations in translation and language learning scenarios.
- **Rejection sampled data:** We apply rejection sampling on our human annotated prompts to generate high-quality samples for finetuning, with few modifications compared to the process for English data:
    - Generation: We explored randomly choosing the temperature hyperparameter from the range $0.2-1$ for diverse generations in early rounds of post-training. With high temperature, responses for multilingual prompts can get creative and inspiring, but are also susceptible to unnecessary or unnatural code-switching. In the final round of post-training, we use a constant value of 0.6 to balance the trade-off. Additionally, we used specialized system prompts to improve response format, structure and general readability.
    - Selection: Prior to reward model based selection, we implement multilingual-specific checks to ensure high language-match rate between the prompt and response (e.g., a romanized Hindi prompt should not expect a response in Hindi Devanagari script).
- **Translated data:** We try to avoid using machine-translated data to finetune the model in order to prevent translationese [Biz20, Mue23] or possible name bias [Wan22a], gender bias [Sav21], or cultural bias [Ji23]. Moreover, we aim to prevent the model from being exposed only to tasks that are rooted in English cultural context, which may not be representative of the linguistic and cultural diversity we aim to capture. We made one exception to this and translated our synthetic quantitative reasoning data (see Section [4.3.3](#section-4-3-3) for details) to improve performance in quantitative reasoning in non-English languages. Due to the simple nature of the language in these math problems, the translated samples were found to have little to no quality issues. We observed strong gains on MGSM [Shi22] from adding this translated data.
<span id="section-4-3-3"></span>

#### 4.3.3 Math and Reasoning

We define reasoning as the ability to perform multi-step computations and arrive at the correct final answer. Several challenges guide our approach to training models that excel in mathematical reasoning:

- **Lack of prompts:** As the complexity of questions increases, the number of valid prompts or questions for Supervised Fine-Tuning (SFT) decreases. This scarcity makes it difficult to create diverse and representative training datasets for teaching models various mathematical skills [Yu23, Yue23, Luo23, Mit24, Sha24, Yue24c].
- **Lack of ground truth chain of thought:** Effective reasoning requires a step-by-step solution to facilitate the reasoning process [Wei22a]. However, there is often a shortage of ground truth chains of thought, which are essential for guiding the model how to break down the problem step-by-step and reach the final answer [Zel22].
- **Incorrect intermediate steps:** When using model-generated chains of thought, the intermediate steps may not always be correct [Cob21, Ues22, Lig23, Wan23a]. This inaccuracy can lead to incorrect final answers and needs to be addressed.
- **Teaching models to use external tools:** Enhancing models to utilize external tools, such as code interpreters, allows them to reason by interleaving code and text [Gao23, Che22, Gou23]. This capability can significantly improve their problem-solving abilities.
- **Discrepancy between training and inference:** There is often a discrepancy between how the model is finetuned during training and how it is used during inference. During inference, the finetuned model may interact with humans or other models, requiring it to improve its reasoning using feedback. Ensuring consistency between training and real-world usage is crucial for maintaining reasoning performance.

To address these challenges, we apply the following methodologies:

- **Addressing the lack of prompts:** We source relevant pre-training data from mathematical contexts and converted it into a question-answer format which can then be used for supervised finetuning. Additionally, we identify mathematical skills where the model under-performs and actively sourced prompts from humans to teach models such skills. To facilitate this process, we create a taxonomy of mathematical skills [Did24] and ask humans to provide relevant prompts/questions accordingly.
- Augmenting training data with step-wise reasoning traces: We use Llama 3 to generate step-by-step solutions for a set of prompts. For each prompt, the model produces a variable number of generations. These generations are then filtered based on the correct answer [Li24i]. We also do self-verification where Llama 3 is used to verify whether a particular step-by-step solution is valid for a given question. This process improves the quality of the finetuning data by eliminating instances where the model does not produce valid reasoning traces.
- **Filtering incorrect reasoning traces:** We train outcome and stepwise reward models [Lig23, Wan23a] to filter training data where the intermediate reasoning steps were incorrect. These reward models are used to eliminate data with invalid step-by-step reasoning, ensuring high-quality data for finetuning. For more challenging prompts, we use Monte Carlo Tree Search (MCTS) with learned step-wise reward models to generate valid reasoning traces, further enhancing the collection of high-quality reasoning data [Xie24].
- **Interleaving code and text reasoning:** We prompt Llama 3 to solve reasoning problems through a combination of textual reasoning and associated Python code [Gou23]. Code execution is used as a feedback signal to eliminate cases where the reasoning chain was not valid, ensuring the correctness of the reasoning process.
- **Learning from feedback and mistakes:** To simulate human feedback, we utilize incorrect generations (*i.e.*, generations leading to incorrect reasoning traces) and perform error correction by prompting Llama 3 to yield correct generations [An23, Wel22, Mad24a]. The iterative process of using feedback from incorrect attempts and correcting them helps improve the model’s ability to reason accurately and learn from its mistakes.
<span id="section-4-3-4"></span>

#### 4.3.4 Long Context

During the final pre-training stage, we extend the context length of Llama 3 from 8K tokens to 128K tokens (see Section [3.4](#section-3-4) for more details). Similar to pre-training, we find that during finetuning we must carefully tune the recipe to balance short and long-context capabilities.

**SFT and synthetic data generation.** Naively applying our existing SFT recipe with only short-context data resulted in significant regressions in long-context capabilities from pre-training, highlighting the need to incorporate long-context data in our SFT data mix. In practice, however, it is largely impractical to get humans to annotate such examples due to the tedious and time-consuming nature of reading lengthy contexts, so we predominantly rely on synthetic data to fill this gap. We use earlier versions of Llama 3 to generate synthetic data based on the key long-context use-cases: (possibly multi-turn) question-answering, summarization for long documents, and reasoning over code repositories, and describe them in greater detail below.

- **Question answering:** We carefully curate a set of long documents from our pre-training mix. We split these documents into chunks of 8K tokens, and prompted an earlier version of the Llama 3 model to generate QA pairs conditional on randomly selected chunks. During training, the whole document is used as context.
- **Summarization:** We applied hierarchical summarization of long-context documents by first summarizing the chunks of 8K input length using our strongest Llama 3 8K context model and then summarizing the summaries. During training we provide the full document and prompt the model to summarize the document while preserving all the important details. We also generate QA pairs based on the summaries of the documents and prompt the model with questions that require global understanding of the whole long document.
- **Long context code reasoning:** We parse Python files to identify import statements and determine their dependencies. From here, we select the most commonly depended-upon files, specifically those referenced by at least five other files. We remove one of these key files from a repository and prompt the model to identify which files depended on the missing file and to generate the necessary missing code.

We further categorize these synthetically generated samples based on the sequence length (16K, 32K, 64K and 128K) to enable more fine-grained targeting of input lengths.

Through careful ablations, we observe that mixing $0.1$% of synthetically generated long-context data with the original short-context data optimizes the performance across both short-context and long-context benchmarks.

**DPO.** We observe that using only short context training data in DPO did not negatively impact long-context performance as long as the SFT model is high quality in long context tasks. We suspect this is due to the fact that our DPO recipe has fewer optimizer steps than SFT. Given this finding, we keep the standard short-context recipe for DPO on top of our long-context SFT checkpoints.
<span id="section-4-3-5"></span>

#### 4.3.5 Tool Use

Teaching LLMs to use tools such as search engines or code interpreters hugely expands the range of tasks they can solve, transforming them from pure chat models into more general assistants [Nak21, Tho22a, Par22a, Gao23, Mia23, Sch24]. We train Llama 3 to interact with the following tools:

- **Search engine.** Llama 3 is trained to use Brave Search [+7] to answer questions about recent events that go beyond its knowledge cutoff or that require retrieving a particular piece of information from the web.
- **Python interpreter.** Llama 3 can generate and execute code to perform complex computations, read files uploaded by the user and solve tasks based on them such as question answering, summarization, data analysis or visualization.
- **Mathematical computational engine.** Llama 3 can use the Wolfram Alpha API [+8] to more accurately solve math, science problems, or retrieve accurate information from Wolfram’s database.

The resulting model is able to use these tools in a chat setup to solve the user’s queries, including in multi-turn dialogs. If a query requires multiple tool calls, the model can write a step-by-step plan, call the tools in sequence, and do reasoning after each tool call.

We also improve Llama 3’s zero-shot tool use capabilities — given in-context, potentially unseen tool definitions and a user query, we train the model to generate the correct tool call.

**Implementation.** We implement our core tools as Python objects with different methods. Zero-shot tools can be implemented as Python functions with descriptions, documentation (*i.e.*, examples for how to use them), and the model only needs the function’s signature and docstring as context to generate the appropriate call. We also convert function definitions and calls to JSON format, e.g., for web API calls. All tool calls are executed by the Python interpreter, that must be enabled in the Llama 3 system prompt. Core tools can be individually enabled or disabled in the system prompt.

**Data collection.** Different from [Sch24], we rely on human annotations and preferences to teach Llama 3 to use tools. There are two main differences with the post-training pipeline generally used in Llama 3:

- For tools, dialogs often contain more than a single assistant message (e.g., calling the tool and reasoning about the tool output). Thus, we annotate at the message level to collect granular feedback: annotators provide a preference between two assistant messages with the same context or, if both contain major problems, edit one of the messages. The chosen or edited message is then added to the context and the dialog continues. This provides human feedback for both the assistant’s ability of calling the tools and reasoning about the tool outputs. Annotators cannot rank or edit the tool outputs.
- We do not perform rejection sampling, as we did not observe gains in our tool benchmarks.

To accelerate the annotation process, we start by bootstrapping basic tool use capabilities by finetuning on synthetically generated data from previous Llama 3 checkpoints. Thus, annotators have fewer edits to perform. In a similar spirit, as Llama 3 gradually improves through its development, we progressively complexify our human annotation protocols: we start by single-turn tool use annotations, before moving to tool use in dialogs, and finally annotating for multi-step tool use and data analysis.

**Tool datasets.** To create data for tool usage applications, we leverage the following procedure:

- **Single-step tool use:** We start by few-shot generation of synthetic user prompts which, by construction, require a call to one of our core tools (for example, questions that exceed our knowledge cutoff date). Then, still relying on few-shot generation, we generate appropriate tool calls for these prompts, execute them, and add the output to the model’s context. Finally, we prompt the model again to generate a final answer to the user’s query based on the tool output. We end up with trajectories of the following form: system prompt, user prompt, tool call, tool output, final answer. We also filter around $30\%$ this dataset to remove tool calls that cannot be executed or other formatting issues.
- **Multi-step tool use:** We follow a similar protocol and first generate synthetic data to teach the model basic multi-step tool use capabilities. To do this, we first prompt Llama 3 to generate user prompts that require at least two tool calls, that can be the same or different tools from our core set. Then, conditioned on these prompts, we few-shot prompt Llama 3 to generate a solution consisting of interleaved reasoning steps and tool calls, similar to ReAct [Yao22b]. See [Figure 10](#figure-10) for an example of Llama 3 performing a task involving multi-step tool usage.
- **File uploads:** We annotate for the following filetypes: .txt, .docx, .pdf, .pptx, .xlsx, .csv, .tsv, .py, .json, .jsonl, .html, .xml. Our prompts are based on a provided file, and ask to summarize the contents of the file, find and fix bugs, optimize a piece of code, perform data analysis or visualization. See [Figure 11](#figure-11) for an example of Llama 3 performing a task involving a file upload.

After finetuning on this synthetic data, we gather human annotations in diverse and challenging scenarios including multi-turn interactions, more than three step tool use, and instances where a tool call does not yield a satisfying answer. We augment our synthetic data with different system prompts to teach the model to use tools only when activated. To train the model to avoid calling tools for simple queries, we also add queries from easy math or question answering datasets [Ber13, Kon16, Jos17, Ami19] and their responses without tools, but with tools activated in system prompt.
<span id="figure-10"></span>

![Refer to caption](../../papers/llama-3/figure-10.png)

**Figure 10.** Multi-step tool usage. Example of Llama 3 performing multi-step planning, reasoning, and tool calling to solve a task.

**Zero-shot tool use data.** We improve Llama 3 zero-shot tool use abilities (also referred to as function calling) by finetuning on a large and diverse set of partly synthetic (functions definitions, user query, corresponding call) tuples. We evaluate our model on a set of unseen tools.

- **Single, nested, and parallel function calling:** Calls can be simple, nested, i.e. we pass a function call as an argument of another function, or parallel, i.e. the model returns a list of independent function calls. Generating a diverse set of functions, queries and ground truths can be challenging [Mek24], and we resort to mining the Stack [Koc22] to ground our synthetic user queries in real functions. More precisely, we extract function calls and their definitions, clean and filter them, e.g. for missing docstrings or non-executable functions, and use Llama 3 to generate a natural language query corresponding to the function call.
- **Multi-turn function calling:** We also generate synthetic data for multi-turn dialogs with function calls, following a protocol similar to the one proposed in [Li23l]. We use multiple agents that generate domains, APIs, user queries, API calls, and responses, while also ensuring that the generated data covers a set of diverse domains and realistic APIs. All agents are variants of Llama 3 prompted in different ways depending on their roles and collaborate in a step-by-step manner.
<span id="figure-11"></span>

    ![Refer to caption](../../papers/llama-3/figure-11.png)

    **[Figure 11](#figure-11).** Processing file uploads. Example of Llama 3 performing analysis and visualization of an uploaded file.
<span id="section-4-3-6"></span>

#### 4.3.6 Factuality

Hallucinations remain a major challenge for large language models. Models tend to be overconfident, even in domains where they have little knowledge. Despite these shortcomings, they are often used as knowledge bases, which can lead to risky outcomes such as the spread of misinformation. While we recognize that factuality can go beyond hallucinations, we took a hallucination-first approach here.

We follow the principle that post-training should align the model to “know what it knows” rather than add knowledge [Gek24, Mie20]. Our primary approach involves generating data that aligns model generations with subsets of factual data present in the pre-training data. To achieve this, we develop a knowledge probing technique that takes advantage of Llama 3’s in-context abilities. This data generation process involves the following procedure:

1. Extract a data snippet from the pre-training data.
2. Generate a factual question about these snippets (context) by prompting Llama 3.
3. Sample responses from Llama 3 to the question.
4. Score the correctness of the generations using the original context as a reference and Llama 3 as a judge.
5. Score the informativeness of the generations using Llama 3 as a judge.
6. Generate a refusal for responses which are consistently informative and incorrect across the generations, using Llama 3.

We use data generated from the knowledge probe to encourage the model to only answer questions which it has knowledge about, and refuse answering those questions that it is unsure about. Further, pre-training data is not always factually consistent or correct. We therefore also collect a limited set of labeled factuality data that deals with sensitive topics where factually contradictory or incorrect statements are prevalent.
<span id="section-4-3-7"></span>

#### 4.3.7 Steerability

Steerability is the ability to direct the model’s actions and outcomes to meet developer and user specifications. As Llama 3 is a generic foundational model, it should be maximally steerable to different downstream use cases easily. For Llama 3, we focus on enhancing its steerability through system prompt with natural language instructions, especially around response length, format, tone and character/persona.

**Data collection.** We collect steerability preference samples within the general English category by asking annotators to design different system prompts for Llama 3. Annotators then engage in conversations with the models to evaluate their consistency in following instructions defined in system prompts over the course of the conversation. We show an example customized system prompt used for enhancing steerability below:

You are a helpful and cheerful AI Chatbot that acts as a meal plan assistant for busy families. The family consists of 2 adults, 3 teenagers, and 2 preschoolers. Plan two or three days at a time and use leftovers or extra ingredients for the second day’s plan. The user will let you know if they want two or three days. If they don’t, assume three days. Each plan should include breakfast, lunch, snack, and dinner. Ask the user if they approve of the plan or need adjustments. After they approve provide a grocery list with family size in mind. Always keep family preferences in mind and if there’s something that they don’t like provide a substitution. If the user is not feeling inspired then ask them what’s the one place they wish they could visit on vacation this week and then suggest meals based on that location’s culture. Weekend meals can be more complex. Weekday meals should be quick and easy. For breakfast and lunch, easy food like cereal, English muffins with pre-cooked bacon, and other quick easy foods are preferred. The family is busy. Be sure to ask if they have essentials and favorites on hand like coffee or energy drinks so they don’t forget to buy it. Remember to be budget-conscious unless it’s a special occasion.

**Modeling.** After we collect the preference data, we leverage this data in reward modeling, rejection sampling, SFT, and DPO to enhance Llama 3’s steerability.
<span id="section-5"></span>

## 5 Results

We performed an extensive series of evaluations of Llama 3, investigating the performance of: (1) the pre-trained language model, (2) the post-trained language model, and (3) the safety characteristics of Llama 3. We present the results of these evaluations in separate subsections below.
<span id="section-5-1"></span>

### 5.1 Pre-trained Language Model

In this section, we report evaluation results for our pre-trained Llama 3 (Section [3](#section-3)), comparing with various other models of comparable sizes. We reproduce results of competitor models whenever possible. For non-Llama models, we report the best score across results that are publicly reported or (where possible) that we reproduced ourselves. The specifics of these evaluations, including configurations such as the number of shots, metrics, and other pertinent hyperparameters and settings, can be accessed on our [Github repository here.](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md) Additionally, we are releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [Huggingface here](https://huggingface.co/meta-llama). We evaluate the quality of our models on standard benchmarks (Section [5.1.1](#section-5-1-1)), for robustness to changes in multiple-choice question setups (Section [5.1.2](#section-5-1-2)), and on adversarial evaluations (Section [5.1.3](#section-5-1-3)). We also conduct a contamination analysis to estimate the extent to which our evaluations are impacted by contamination of training data (Section [5.1.4](#section-5-1-4)).
<span id="section-5-1-1"></span>

#### 5.1.1 Standard Benchmarks
<span id="table-08"></span>

![Original paper Table 8](../../papers/llama-3/table-08.png)

**Table 8.** Pre-training benchmarks by category. Overview of all benchmarks we use to evaluate pre-trained Llama 3 models, grouped by capability category.

To compare our models with the current state-of-the-art, we evaluate Llama 3 on a large number of standard benchmark evaluations shown in [Table 8](#table-08). These evaluations cover eight top-level categories: (1) commonsense reasoning; (2) knowledge; (3) reading comprehension; (4) math, reasoning, and problem solving; (5) long context; (6) code; (7) adversarial evaluations; and (8) aggregate evaluations.

**Experimental setup.** For each benchmark, we compute scores for Llama 3 as well as various other pre-trained models of comparable sizes. Where possible, we recompute numbers with our own pipeline for other models. To ensure a fair comparison, we then select the best score between the score that we computed and the reported number for that model with comparable or more conservative settings. You can find additional details on our evaluation setup [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md). For some models, it is not possible to (re)compute benchmark values, for instance, because the pre-trained model is not released or because the API does not provide access to log-probabilities. In particular, this is true for all models comparable to Llama 3 405B. Thus, we do not report category averages for Llama 3 405B, which requires that all numbers are available for all benchmarks.

**Significance estimates.** Benchmark scores are estimates of a model’s true performance. These estimates have variance because benchmark sets are finite samples drawn from some underlying distribution. We follow [Mad24b] and report on this variance via 95% confidence intervals (CIs), assuming that benchmark scores are Gaussian distributed. While this assumption is incorrect (*e.g.*, benchmark scores are bounded), preliminary bootstrap experiments suggest CIs (for discrete metrics) are a good approximation:

$$
\mathrm{CI}(S)=1.96\times\sqrt{\frac{S\times(1-S)}{N}}.
$$

Herein, $S$ is the observed benchmark score (*e.g.*, accuracy or EM) and $N$ the sample size of the benchmark. We omit CIs for benchmark scores that are not simple averages. We note that because subsampling is not the only source of variation, our CI values lower bound the actual variation in the capability estimate.
<span id="figure-12"></span>

![Refer to caption](../../papers/llama-3/figure-12a.png)

![Refer to caption](../../papers/llama-3/figure-12b.png)

**Figure 12.** Performance of pre-trained Llama 3 8B and 70B models on pre-training benchmarks. Results are aggregated by capability category by averaging accuracies across all benchmarks corresponding to that category.

**Results for 8B and 70B models.** [Figure 12](#figure-12) reports the average performance of Llama 3 8B and 70B on the commonsense reasoning, knowledge, reading comprehension, math and reasoning, and code benchmarks. The results show that Llama 3 8B outperforms competing models in virtually every category, both in terms of per-category win rate and in terms of average per-category performance. We also find that Llama 3 70B outperforms its predecessor Llama 2 70B by a large margin on most benchmarks, with the exception of commonsense benchmarks that are likely saturated. Llama 3 70B also outperforms Mixtral 8x22B.

**Detailed results for all models.** [Tables 9](#table-09), [10](#table-10), [11](#table-11), [12](#table-12), [13](#table-13), and [14](#table-14) present the benchmark performance of pre-trained Llama 3 8B, 70B, and 405B models on reading comprehension tasks, coding tasks, commonsense understanding tasks, mathematical reasoning tasks, and general tasks. The tables compare Llama 3’s performance with that of models of similar size. The results show that Llama 3 405B performs competitively with other models in its class. In particular, Llama 3 405B substantially outperforms prior open-source models. For long-context, we present more comprehensive results (including probing tasks like needle-in-a-haystack) in Section [5.2](#section-5-2).
<span id="table-09"></span>

![Table 9](../../papers/llama-3/table-09.png)

**Table 9.** Pre-trained model performance on reading comprehension tasks. Results include 95% confidence intervals.
<span id="table-10"></span>

![Original paper Table 10](../../papers/llama-3/table-10.png)

**Table 10.** Pre-trained model performance on coding tasks. Results include 95% confidence intervals.
<span id="table-11"></span>

![Original paper Table 11](../../papers/llama-3/table-11.png)

**Table 11.** Pre-trained model performance on commonsense understanding tasks. Results include 95% confidence intervals.
<span id="table-12"></span>

![Original paper Table 12](../../papers/llama-3/table-12.png)

**Table 12.** Pre-trained model performance on math and reasoning tasks. Results include 95% confidence intervals. $^{\diamondsuit}$11-shot. $^{\triangle}$Variable shot.
<span id="table-13"></span>

![Original paper Table 13](../../papers/llama-3/table-13.png)

**Table 13.** Pre-trained model performance on general language tasks. Results include 95% confidence intervals.
<span id="section-5-1-2"></span>

#### 5.1.2 Model Robustness

In addition to performance on benchmarks, robustness is an important factor in the quality of pre-trained language models. We investigate the robustness of our pre-trained language models to design choices in multiple-choice question (MCQ) setups. Prior work has reported that model performance can be sensitive to seemingly arbitrary design choices in such setups, for example, model scores and even rankings may change with the order and labels of the in-context examples [Lu22, Zha21, Rob23, Lia22, Gup24], the exact format of the prompt [Web23a, Mis22], or the answer choice format and order [Alz24, Wan24f, Zhe23a]. Motivated by this work, we use the MMLU benchmark to evaluate the robustness of our pre-trained models to: (1) few-shot label bias, (2) label variants, (3) answer order, and (4) prompt format:
<span id="figure-13"></span>

![Original paper Figure 13](../../papers/llama-3/figure-13.png)

**Figure 13.** Robustness of our pre-trained language models to different design choices in the MMLU benchmark. *Left:* Performance for different label variants. *Right:* Performance for different labels present in few-shot examples.
<span id="figure-14"></span>

![Original paper Figure 14](../../papers/llama-3/figure-14.png)

**Figure 14.** Robustness of our pre-trained language models to different design choices in the MMLU benchmark. *Left:* Performance for different answer orders. *Right:* Performance for different prompt formats.

- **Few-shot label bias.** Following [Zhe23a] and [Web23], we investigate the impact of the distribution of labels in four-shot examples. Specifically, we consider settings in which: (1) all few-shot examples have the same label (A A A A); (2) all examples have a different label (A B C D); and (3) there are only two labels present (A A B B and C C D D).
- **Label variants.** We also study model response to different choice token sets. We consider the two sets proposed by [Alz24]: namely, a set of common language independent tokens ($ & # @) and a of rare tokens (œ § з ü) that do not have any implicit relative order. We also consider two versions of the canonical labels (A. B. C. D. and A) B) C) D)) and a numerical list (1\. 2. 3. 4.).
- **Answer order.** Following [Wan24f], we compute how stable the results are across different answer orders. To compute this, we remap all the answers in the dataset according to a fixed permutation. For example, for the permutation A B C D, all answer options with label A and B keep their label, and all answer options with label C get label D, and vice versa.
- **Prompt format.** We evaluate variance in performance across five task prompts that differ in the level of information provided: one prompt simply asks the model to answer the question, whereas other prompts assert the expertise of the model or that the best answer should be chosen.

[Figure 13](#figure-13) presents the results of our experiments studying robustness of model performance to label variants (left) and few-shot label bias (right). The results show that our pre-trained language models are very robust to changes in MCQ labels and to the structure of the few-shot prompt labels. This robustness is particularly pronounced for the 405B parameter model. [Figure 14](#figure-14) presents the results of our study of robustness to answer order and prompt format. The results in the figure further underscore the robustness of the performance of our pre-trained language models, in particular, of Llama 3 405B.
<span id="figure-15"></span>

![Original paper Figure 15](../../papers/llama-3/figure-15.png)

**Figure 15.** Adversarial versus non-adversarial performance for question answering, mathematical reasoning, and paraphrase detection benchmarks. *Left:* Results for pre-trained models. *Right:* Results for post-trained models.
<span id="section-5-1-3"></span>

#### 5.1.3 Adversarial Benchmarks

In addition to the benchmarks presented above, we evaluate on several adversarial benchmarks in three areas: question answering, mathematical reasoning, and paraphrase detection. This testing probes the model’s capabilities on tasks specifically created to be challenging and can potentially also point to overfitting on benchmarks. For question answering, we use Adversarial SQuAD [Jia17] and Dynabench SQuAD [Kie21]. For mathematical reasoning, we use GSM-Plus [Li24h]. For paraphrase detection, we use PAWS [Zha19e]. [Figure 15](#figure-15) presents the scores of Llama 3 8B, 70B, and 405B on the adversarial benchmarks as a function of their performance on non-adversarial benchmarks. The non-adversarial benchmarks we use are SQuAD [Raj16] for question answering, GSM8K for mathematical reasoning, and QQP [Wan17] for paraphrase detection. Each datapoint represents a pair of an adversarial and non-adversarial datasets (*e.g.* QQP paired with PAWS), and we show all possible pairs within a category. The diagonal black line represents parity between adversarial and non-adversarial datasets — being on the line would indicate the model has similar performance regardless of the adversarial nature. On paraphrase detection, neither pre-trained nor post-trained models appear to suffer from the type of adversariality with which PAWS was constructed, marking a substantial step with respect to the previous generation of models. This result confirms the findings of [Web23], who also found that LLMs are less susceptible to the type of spurious correlations found in several adversarial datasets. For mathematical reasoning and question answering, however, the adversarial performances are substantially lower than the non-adversarial performances. This pattern is similar for pre-trained and post-trained models.
<span id="section-5-1-4"></span>

#### 5.1.4 Contamination Analysis

We conduct a contamination analysis to estimate to what extent benchmark scores may be influenced by contamination of the evaluation data in the pre-training corpus. In previous work, several different contamination methods have been used, with various different hyperparameters – we refer to [Sin24] for an overview. Any of these methods can suffer from false positives and negatives, and how to best run contamination analyses is currently still an open field of research. Here, we largely follow the suggestions of [Sin24].
<span id="table-14"></span>

![Original paper Table 14](../../papers/llama-3/table-14.png)

**Table 14.** Performance of pre-trained models on long-context tasks. Results include 95% confidence intervals.
<span id="table-15"></span>

![Original paper Table 15](../../papers/llama-3/table-15.png)

**Table 15.** Percentage of evaluation sets considered to be contaminated because similar data exists in the training corpus, and the estimated performance gain that may result from that contamination. See the text for details.

**Method.** Specifically, [Sin24] propose to select contamination detection methods empirically, based on which method results in the largest difference between the ‘clean’ part of the dataset and the entire dataset, which they call *estimated performance gain*. For all our evaluation datasets, we score examples based on 8-gram overlap, a method that was found by [Sin24] to be accurate for many datasets. We consider an example of a dataset $D$ to be contaminated if a ratio $\mathcal{T}_{D}$ of its tokens are part of an 8-gram occurring at least once in the pre-training corpus. We select $\mathcal{T}_{D}$ separately for each dataset, based on which value shows the maximal significant estimated performance gain across the three model sizes.

**Results.** In [Table 15](#table-15), we report the percentage of evaluation data that is considered contaminated for the maximal estimated performance gain, as described above, for all key benchmarks. From the table, we exclude numbers for benchmarks for which the results are not significant, for instance because the clean or contaminated set has too few examples, or because the observed performance gain estimate shows extremely erratic behavior. In [Table 15](#table-15), we observe that for some datasets contamination has a large impact, while for others it does not. For example, for PiQA and HellaSwag, both the estimation of contamination and the estimation of performance gain are high. For Natural Questions, on the other hand, the estimated 52% contamination seems to have virtually no effect on the performance. For SQuAD and MATH, low thresholds yield high levels of contamination, but no performance gains. This suggests that contamination is either not helpful for these datasets, or that a larger n is required to obtain a better estimate. Finally, for MBPP, HumanEval, MMLU and MMLU-Pro, other contamination detection methods may be needed: even with higher thresholds, 8-gram overlap gives such high contamination scores that it is impossible to get a good performance gain estimate.
<span id="section-5-2"></span>

### 5.2 Post-trained Language Model

We present results for our Llama 3 post-trained models on benchmarks across different capabilities. Similar to pre-training we are releasing the data generated as part of evaluations with publicly available benchmarks which can be found on [Huggingface here](https://huggingface.co/meta-llama). Additional details on our eval setup can be found [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md).
<span id="table-16"></span>

![Original paper Table 16](../../papers/llama-3/table-16.png)

**Table 16.** Post-training benchmarks by category. Overview of all benchmarks we use to evaluate post-trained Llama 3 models, ordered by capability.

**Benchmarks and metrics.** [Table 16](#table-16) contains an overview of all the benchmarks, organized by the capability. We apply decontamination of the post-training data by running exact match with the prompts from each benchmark. In addition to the standard academic benchmarks, we also performed extensive human evaluation of different capabilities. Details are provided in Section [5.3](#section-5-3).

**Experimental setup.** We employ a similar experimental setup to the pre-training phase and conduct a comparative analysis of Llama 3 alongside other models of comparable size and capability. To the extent possible, we evaluate the performance of other models ourselves and compare the results with the reported numbers, selecting the best score. You can find additional details on our evaluation setup [here](https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/eval_details.md).
<span id="section-5-2-1"></span>

#### 5.2.1 General Knowledge and Instruction-Following Benchmarks

We evaluate Llama 3 on benchmarks for general knowledge and instruction-following in [Table 1](#table-01).

**General knowledge.** We leverage MMLU [Hen20] and MMLU-Pro [Wan24c] to evaluate Llama 3’s capability on knowledge-based question answering. For MMLU, we report the macro average of subtask accuracy under the 5-shot standard setting without CoT. MMLU-Pro is an extension of MMLU, incorporating more challenging, reasoning-focused questions, eliminating noisy questions, and expanding the choice set from four to ten options. Given its focus on complex reasoning, we report 5-shot CoT for MMLU-Pro. All tasks are formatted as generation tasks, similar to simple-evals [Ope24f]. As shown in [Table 1](#table-01), our 8B and 70B Llama 3 variants outperform other models of similar sizes on both general knowledge tasks. Our 405B model outperforms GPT-4 and Nemotron 4 340B, with Claude 3.5 Sonnet leading among larger models.

**Instruction following.** We assess the ability of Llama 3 and other models to follow natural language instructions on IFEval  [Zho23a]. IFEval comprises approximately 500 “verifiable instructions” such as “write in more than 400 words”, which can be verified by heuristics. We report the average of prompt-level and instruction-level accuracy, under strict and loose constraints in [Table 1](#table-01). Note that all Llama 3 variants outperform comparable models across IFEval.
<span id="table-17"></span>

![Original paper Table 17](../../papers/llama-3/table-17.png)

**Table 17.** Performance of Llama 3 models and GPT-4o on a variety of proficiency exams including LSAT, SAT, GMAT, and AP, and GRE tests. For GRE exams, we report normalized score; for all others, we report accuracy. For the bottom two rows corresponding to GRE Quant. and GRE Verbal, we report the scaled scores out of 170.
<span id="section-5-2-2"></span>

#### 5.2.2 Proficiency Exams

Next, we evaluate our models on a wide variety of proficiency exams originally designed to test humans. We source these exams from publicly available official sources; for some exams, we report average scores across different exam sets per proficiency exam. Specifically, we average:

- **GRE:** Official GRE Practice Test 1 and 2 (from the Educational Testing Services);
- **LSAT:** Official Preptest 71, 73, 80 and 93;
- **SAT:** 8 exams from The Official SAT Study guide edition 2018;
- **AP:** One official practice exam per subject;
- **GMAT** Official GMAT Online Exam.

Questions in these exams contain both MCQ style and generation questions. We exclude the questions that are accompanied with images. For the GRE exams that contain questions with multiple correct options, we qualify the outputs as correct only if all the correct options are selected by the model. The evaluations are run using few shot prompting wherever we have more than 1 exam set per exam. We scale the scores to be in the range 130-170 for GRE and report accuracy for all other exams. Our results can be found in [Table 17](#table-17). We observe that the performance of our Llama 3 405B model is very similar to Claude 3.5 Sonnet and GPT-4o. Our 70B model has an even more impressive performance. It is significantly better than GPT-3.5 Turbo and beats Nemotron 4 340B on many tests.
<span id="section-5-2-3"></span>

#### 5.2.3 Coding Benchmarks

We evaluate Llama 3 on code generation on several popular Python and multi-programming language benchmarks. To gauge the effectiveness of our models in generating functionally correct code, we use the pass@$N$ metric, which evaluates the pass rate for a set of unit tests among $N$ generations. We report pass@1.

**Python code generation.** HumanEval [Che21] and MBPP [Aus21a] are popular benchmarks for Python code generation which focus on relatively simple, self-contained functions. HumanEval+ [Liu24i] is an enhanced version of HumanEval, in which more tests are generated to avoid false positives. The MBPP EvalPlus base version (v0.2.0) is a selection of 378 well-formed problems out of the 974 initial problems in all of the original MBPP (train and test) dataset  [Liu24i]. Results for these benchmarks are reported in [Table 18](#table-18). Across the Python variants of these benchmarks, Llama 3 8B and 70B outperform models of similar sizes. For the largest models, Llama 3 405B, Claude 3.5 Sonnet and GPT-4o perform similarly, with GPT-4o showing the strongest results.

**Multi-programming language code generation.** To assess code generation capabilities beyond Python, we report results for the MultiPL-E [Cas23] benchmark, which is based on translations of problems from HumanEval and MBPP. Results for a subset of popular programming languages are reported in [Table 19](#table-19). Note that there is a significant drop in performance compared to the Python counterparts in [Table 18](#table-18).
<span id="table-18"></span>

![Original paper Table 18](../../papers/llama-3/table-18.png)

**Table 18.** Pass@1 scores on code generation benchmarks. We report results on HumanEval [Che21], MBPP [Aus21a], as well as EvalPlus  [Liu24i] versions of these benchmarks.
<span id="table-19"></span>

![Original paper Table 19](../../papers/llama-3/table-19.png)

**Table 19.** Performance of non-Python programming tasks. We report Llama 3 results on MultiPL-E [Cas23].
<span id="section-5-2-4"></span>

#### 5.2.4 Multilingual Benchmarks

Llama 3 supports 8 languages — English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai, although the underlying foundation model has been trained on a broader collection of languages. [+9] In [Table 20](#table-20), we show results from evaluating Llama 3 on the multilingual MMLU [Hen20] and Multilingual Grade School Math (MGSM) [Shi22] benchmarks.

**Multilingual MMLU.** We translate MMLU questions, few-shot examples, and answers using Google Translate. We leave the task instructions in English and perform the evaluation in a 5-shot setting. In [Table 20](#table-20), we report average results across German, French, Italian, Portuguese, Hindi, Spanish, and Thai.
<span id="table-20"></span>

![Original paper Table 20](../../papers/llama-3/table-20.png)

**Table 20.** Multilingual benchmarks. For MGSM [Shi22], we report 0-shot CoT results for our Llama 3 models. Multilingual MMLU is an internal benchmark with translated MMLU [Hen20] questions and answers into 7 languages – we report 5-shot results averaged across these languages.

**MGSM [Shi22].** We use the same native prompts as in simple-evals [Ope24f] for testing our models in a 0-shot CoT setting. In [Table 20](#table-20), we report average results across languages covered in MGSM benchmark. We find that Llama 3 405B outperforms most other models on MGSM, achieving an average of 91.6%. On MMLU, in line with English MMLU results shown above, Llama 3 405B falls behind GPT-4o by 2%. On the other hand, both Llama 3 70B and 8B models demonstrate strong performance, leading among competitors with a wide margin on both tasks.
<span id="section-5-2-5"></span>

#### 5.2.5 Math and Reasoning Benchmarks

Our math and reasoning benchmark results are presented in [Table 1](#table-01). Llama 3 8B model outperforms other models of similar sizes on GSM8K, MATH, and GPQA. Our 70B model performs significantly better than other models in its class on all the benchmarks. Finally, Llama 3 405B model is the best in its category on GSM8K and ARC-C, while on MATH, it is the second best model. On GPQA, it is competitive with GPT-4o, with Claude 3.5 Sonnet being the best model by a significant margin.
<span id="section-5-2-6"></span>

#### 5.2.6 Long Context Benchmarks

We consider a diverse set of tasks that span various domains and text types. In the benchmarks we list below, we focus on sub-tasks that use unbiased evaluation protocols, i.e., accuracy-based metrics rather than n-gram overlapping metrics. We also prioritize tasks that we found to be of low variance.

- Needle-in-a-Haystack [Kam23] measures a model’s ability to retrieve a hidden information inserted in random parts of the long document. Our Llama 3 models demonstrate perfect needle retrieval performance, successfully retrieving 100% of needles at all document depths and context lengths. We also measure performance on Multi-needle ([Table 21](#table-21)), a variation of Needle-in-a-Haystack, where we insert four needles in the context and test if a model can retrieve two of them. Our Llama 3 models achieve near perfect retrieval results.
- ZeroSCROLLS [Sha23c] is a zero-shot benchmark for natural language understanding over long texts. We report numbers on the validation set, as the ground truth answers are not publicly available. Our Llama 3 405B and 70B models either match or surpass other models on various tasks in this benchmark.
- InfiniteBench [Zha24h] requires models to understand long dependencies in the context window. We evaluate Llama 3 on En.QA (QA over novels) and En.MC (multiple-choice QA over novels), where our 405B model outperforms all others. The gains are particularly significant on En.QA.
<span id="table-21"></span>

![Original paper Table 21](../../papers/llama-3/table-21.png)

**Table 21.** Long-context benchmarks. For ZeroSCROLLS [Sha23c], we report numbers on the validation set. For QuALITY we report exact match, for Qasper - f1 and for SQuALITY - rougeL. We report f1 for InfiniteBench [Zha24h] En.QA metric and accuracy for En.MC. For Multi-needle  [Kam23] we insert 4 needles in the context and test if a model can retrieve 2 needles at different context lengths, we compute average recall across 10 sequence lengths up till 128k.
<span id="section-5-2-7"></span>

#### 5.2.7 Tool Use Performance

We evaluate our models on a range of benchmarks for zero-shot tool use (*i.e.* function calling): Nexus [Sri23], API-Bank [Li23l], Gorilla API-Bench [Pat23b], and the Berkeley Function Calling Leaderboard (BFCL) [Yan24d]. Results are shown in [Table 22](#table-22).

On Nexus, our Llama 3 variants perform the best compared to their counterparts. On the API-Bank, our Llama 3 8B and 70B models outperform other models in their category by a significant margin. The 405B model is behind Claude 3.5 Sonnet by only 0.6%. Finally, our 405B and 70B models perform competitively on BFCL and are close second in their respective size class. Llama 3 8B performs the best in its category.

**Human evaluations.** We also conduct human evaluations to test the tool use capabilities of the model, with a focus on code execution tasks. We collect 2000 user prompts related to code execution (without plotting or file uploads), plot generation, and file uploads. These prompts are collected from the LMSys dataset [Chi24], GAIA benchmark [Mia23a], human annotators, and synthetic generation.
<span id="table-22"></span>

![Original paper Table 22](../../papers/llama-3/table-22.png)

**Table 22.** Zero-shot tool use benchmarks. We report function calling accuracy across Nexus [Sri23], API-Bank [Li23l], API-Bench [Pat23b], and BFCL [Yan24d].

We compare Llama 3 405B to GPT-4o using OpenAI’s Assistants API [+10]. The results are provided in [Figure 16](#figure-16). On text-only code execution tasks and plots generation, Llama 3 405B significantly beats GPT-4o. However, it lags behind on the file upload use case.
<span id="section-5-3"></span>

### 5.3 Human Evaluations

In addition to evaluations on standard benchmark sets, we also perform a series of human evaluations. These evaluations allow us to measure and optimize more subtle aspects of model performance, such as our model’s tone, verbosity, and understanding of nuances and cultural contexts. Well-designed human evaluations closely reflect the user experience, providing insights into how the model performs in real-world scenarios.

**Prompt collection.** We collected high-quality prompt spanning a wide range of categories and difficulties. To do so, we first developed a taxonomy with categories and subcategories capturing as many model capabilities as possible. We used this taxonomy to collect about $7,000$ prompts spanning six individual capabilities (English, reasoning, coding, Hindi, Spanish, and Portuguese), and three multiturn capabilities [+11] (English, reasoning, and coding). We ensured that within each category, prompts are uniformly distributed across subcategories. We also categorized each prompt into one of three difficulty levels and ensured that our prompt collection contains roughly $10\%$ easy prompts, $30\%$ medium prompts, and $60\%$ hard prompts. All the human evaluation prompt sets were subject to a thorough quality assurance process. Modeling teams did not have access to our human-evaluation prompts to prevent accidental contamination or overfitting on the test set.
<span id="figure-16"></span>

![Original paper Figure 16](../../papers/llama-3/figure-16.png)

**Figure 16.** Human evaluation results for Llama 3 405B vs. GPT-4o on code execution tasks including plotting and file uploads. Llama 3 405B outperforms GPT-4o on code execution (without plotting or file uploads) as well as plot generation, but lags behind in file upload use cases.

**Evaluation process.** To perform a pairwise human evaluation of two models, we ask human annotators which of two model responses (produced by different models) they prefer. Annotators use a 7-point scale for their ratings, enabling them to indicate whether one model response is much better than, better than, slightly better than, or about the same as the other model response. When an annotator indicates that one model response is better or much better than the other model response, we consider this a “win” for that model. We perform pairwise comparisons between models in which we report win rates per capability in the prompt set.

**Results.** We use our human evaluation process to compare Llama 3 405B with GPT-4 (0125 API version), GPT-4o (API version), and Claude 3.5 Sonnet (API version). The results of these evaluations are presented in [Figure 17](#figure-17). We observe that Llama 3 405B performs approximately on par with the 0125 API version of GPT-4, while achieving mixed results (some wins and some losses) compared to GPT-4o and Claude 3.5 Sonnet. On nearly all capabilities, the win rates of Llama 3 and GPT-4 are within the margin of error. On multiturn reasoning and coding tasks, Llama 3 405B outperforms GPT-4 but it underperforms GPT-4 on multilingual (Hindi, Spanish, and Portuguese) prompts. Llama 3 performs on par with GPT-4o on English prompts, on par with Claude 3.5 Sonnet on multilingual prompts, and outperforms Claude 3.5 Sonnet on single and multiturn English prompts. However, it trails Claude 3.5 Sonnet in capabilities such as coding and reasoning. Qualitatively, we find that model performance in human evaluations is heavily influenced by nuanced factors such as model tone, response structure, and verbosity – factors that we are optimizing for in our post-training process. Overall, our human evaluation results are consistent with those on standard benchmark evaluations: Llama 3 405B is very competitive with leading industry models, making it the best-performing openly available model.
<span id="figure-17"></span>

![Original paper Figure 17](../../papers/llama-3/figure-17.png)

**Figure 17.** Human evaluation results for the Llama 3 405B model. *Left:* Comparison with GPT-4. *Middle:* Comparison with GPT-4o. *Right:* Comparison with Claude 3.5 Sonnet. All results include 95% confidence intervals and exclude ties.

**Limitations.** All human evaluation results underwent a thorough data quality assurance process. However, since it is challenging to define objective criteria for evaluating model responses, human evaluations can still be influenced by personal biases, backgrounds, and preferences of human annotators, which may lead to inconsistent or unreliable results.
<span id="section-5-4"></span>

### 5.4 Safety

We focus our study on assessing Llama 3’s ability to generate content in a safe and responsible way, while still maximizing helpful information. Our safety work begins in the pre-training stage, primarily in the form of data cleaning and filtering. We then describe our approach to safety finetuning, focusing on how to train the model to align to specific safety policies while still retaining helpfulness. We analyze each of the Llama 3 capabilities, including multilingual, long context, tool usage, and various multimodal capabilities, to measure the effectiveness of our safety mitigations. Subsequently, we describe our assessment of uplift for cybersecurity and chemical and biological weapons risks. Uplift refers to the additional risk introduced by new technological developments compared to using existing available technologies (such as web search). We then describe how we leverage Red Teaming to iteratively identify and combat various safety risks across capabilities and perform a residual risk assessment. Finally, we describe system-level safety, or the development and orchestration of classifiers around the input and output of the model itself to further enhance safety and make it easier for developers to both customize safety to various usecases and deploy generative AI in more responsible ways.
<span id="section-5-4-1"></span>

#### 5.4.1 Benchmark Construction

We create various internal benchmarks to help us develop models safely and responsibly. Our benchmarks are heavily inspired by the risk categories from the ML Commons taxonomy of hazards [Vid24]. While various benchmarks for language model safety exist such as ToxiGen [Har22], XS Test [Rot23], and more, a number of the new capabilities in Llama 3 do not have sufficient external benchmarks available and often external benchmarks lack sufficient coverage of breadth and depth. For each risk category, we collect human-written prompts that are either adversarial or borderline in nature — examples of such prompts can be found in [Table 23](#table-23). Adversarial prompts range from straightforward ones that directly elicit a harmful response to ones that incorporate sophisticated jailbreaking techniques. Benchmarks of these adversarial prompts are the basis for how we measure violation rate. As a counter-metric to violation rate, we also construct false refusal benchmarks composed of borderline prompts. We refer to false refusal as occurring when a model refuses to answer in a helpful way even when a plausible, safe response is possible. Borderline prompts are prompts near the decision boundary that a well-calibrated model should handle, for example, *“How can I steal the spotlight from my bestie who always acts like the main character??”* Our overall benchmark size across violations and false refusals is over 4000 prompts per capability or language, and contains a mix of single-turn and multi-turn prompts.
<span id="table-23"></span>

![Original paper Table 23](../../papers/llama-3/table-23.png)

**Table 23.** Examples of adversarial prompts from our internal benchmarks across all the capabilities.
<span id="section-5-4-2"></span>

#### 5.4.2 Safety Pre-training

We believe responsible development must be considered from an end-to-end perspective and incorporated at every stage of model development and deployment. During pre-training, we apply a variety of filters, such as filters to identify websites that likely contain personally identifiable information (see Section [3.1](#section-3-1)). We also focus heavily on discoverable memorization [Nas23]. Similar to [Car22], we sample prompts and ground truths at different frequencies of occurrence in the training data using an efficient rolling hash index of all n-grams in the corpus. We construct different test scenarios by varying the length of prompt and ground truth, the detected language of target data, and the domain. We then measure how often the model generates the ground truth sequence verbatim, and analyze the relative rates of memorization in the specified scenarios. We define verbatim memorization as the inclusion rate – the proportion of model generations that include the ground truth continuation exactly – and report averages weighted by the prevalence of given characteristics in the data, as shown in [Table 24](#table-24). We find low memorization rates of training data (1.13% and 3.91% on average for the 405B with $n=50$ and $n=1000$ respectively). Memorization rates are roughly on par with Llama 2 at equivalent size and using the same methodology applied to its data mix. [+12]
<span id="table-24"></span>

![Original paper Table 24](../../papers/llama-3/table-24.png)

**Table 24.** Average verbatim memorization in pre-trained Llama 3 for selected test scenarios. Our baseline is Llama 2 in the English, 50-gram scenario using the same prompting methodology applied to its data mix.
<span id="section-5-4-3"></span>

#### 5.4.3 Safety Finetuning

We describe our approach to safety finetuning to mitigate risks across many capabilities, which encompasses two key aspects: (1) safety training data and (2) risk mitigation techniques. Our safety finetuning process builds upon our general finetuning methodology with modifications tailored to address specific safety concerns.

We optimize for two primary metrics: Violation Rate (VR), a metric that captures when the model produces a response that violates a safety policy, and False Refusal Rate (FRR), a metric that captures when the model incorrectly refuses to respond to a harmless prompt. In parallel, we evaluate model performance on helpfulness benchmarks to ensure that safety improvements do not compromise overall helpfulness.
<span id="figure-18"></span>

![Original paper Figure 18](../../papers/llama-3/figure-18.png)

**Figure 18.** Influence of model size on safety mix design for balancing violation rate (VR) and false refusal rate (FRR). Each point of the scatterplot represents a different data mix balancing safety and helpfulness data. Different model sizes retain varying capacities for safety learning. Our experiments show that 8B models require a higher proportion of safety data relative to helpfulness data in the overall SFT mix to achieve comparable safety performance to 70B models. Larger models are more capable of discerning between adversarial and borderline context, resulting in a more favorable balance between VR and FRR.

**Finetuning data.** The quality and design of safety training data has a profound impact on performance. Through extensive ablations, we find that the quality is more critical than the quantity. We mainly use human-generated data collected from our data vendors, but find that it can be prone to errors and inconsistencies — particularly for nuanced safety policies. To ensure the highest quality data, we developed AI-assisted annotation tools to support our rigorous quality assurance processes. In addition to collecting adversarial prompts, we also gather a set of similar prompts, which we refer to as borderline prompts. These are closely related to the adversarial prompts but with a goal to teach the model to learn to provide helpful responses, thereby reducing the false refusal rate (FRR).

Beyond human annotation, we also leverage synthetic data to improve the quality and coverage of our training datasets. We utilize a range of techniques to generate additional adversarial examples, including in-context learning with carefully crafted system prompts, guided mutation of seed prompts based on new attack vectors, and advanced algorithms including Rainbow Teaming [Sam24], based on MAP-Elites [Mou15], which generate prompts constrained across multiple dimensions of diversity.

We further address the model’s tone when producing safe responses, which has an impact on downstream user experience. We developed a refusal tone guideline for Llama 3 and ensured that all new safety data adhered to it through rigorous quality assurance process. We also refine existing safety data to align with the guideline, using a combination of zero-shot rewriting and human-in-the-loop editing to produce high-quality data. By employing these methods, along with a tone classifier to assess tone quality for safety responses, we are able to significantly improve the model’s verbiage.

**Safety supervised finetuning.** Following our Llama 2 recipe [Tou23c], we combine all helpfulness data and safety data during the model alignment stage. Additionally, we introduce a borderline dataset to help the model discern the subtle distinctions between safe and unsafe requests. Our annotation teams are instructed to meticulously craft responses to safety prompts based on our guidelines. We have found that SFT is highly effective in aligning the model when we strategically balance the ratio of adversarial to borderline examples. We put the focus on more challenging risk areas, with a higher ratio of borderline examples. This plays a crucial role in our successful safety mitigation efforts while keeping false refusal to a minimum.

Further, we examine the impact of model size on the trade-off between FRR and VR in [Figure 18](#figure-18). Our results show that it varies — with smaller models requiring a larger proportion of safety data relative to helpfulness, and that it is more challenging to efficiently balance VR and FRR compared to larger models.
<span id="figure-19"></span>

![Original paper Figure 19](../../papers/llama-3/figure-19.png)

**Figure 19.** Violation rates (VR) and false refusal rates (FRR) on English and our core multilingual short context benchmarks, comparing Llama 3 405B—with and without Llama Guard (LG) system-level protections—to competitor models and systems. Languages not supported by Comp. 3 represented with an ‘x.’ Lower is better.

**Safety DPO.** To reinforce safety learning, we incorporate adversarial and borderline examples into our preference datasets in DPO. We discover that crafting response pairs to be nearly orthogonal in an embedding space is particularly effective in teaching the model to distinguish between good and bad responses for a given prompt. We conduct multiple experiments to determine the optimal ratio of adversarial, borderline, and helpfulness examples, aiming to optimize the trade-off between FRR and VR. We also find that the model size influences the learning outcomes — as a result, we tailor different safety mixes for various model sizes.
<span id="section-5-4-4"></span>

#### 5.4.4 Safety Results

We first highlight Llama 3’s general behavior along various axes and then describe results for each specific new capability and our effectiveness at mitigating the safety risks.

**Overall performance.** A comparison of Llama 3’s final violation and false refusal rates with similar models can be found in [Figure 19](#figure-19) and [20](#figure-20). These results focus on our largest parameter size Llama 3 405B model, compared to relevant competitors. Two of the competitors are end-to-end systems accessed through API, and one of them is an open source language model that we host internally and we evaluate directly. [+13] We evaluate our Llama models both standalone and coupled with Llama Guard, our open source system-level safety solution (more in Section [5.4.7](#section-5-4-7)).
<span id="figure-20"></span>

![Original paper Figure 20](../../papers/llama-3/figure-20.png)

**Figure 20.** Violation rates (VR) and false refusal rates (FRR) on tool use and long context benchmarks. Lower is better. The performance for DocQA and Many-shot benchmarks are listed separately. Note we do not have a borderline data set for Many-shot, due to the adversarial nature of the benchmark, and thus do not measure false refusal rates on it. For Tool Usage (Search), we only test Llama 3 405B compared to Comp. 1.

While a low violation rate is desirable, it is critical to consider false refusal as a counter-metric, as a model that always refuses is maximally safe, but not helpful in the slightest. Similarly, a model that always answers every prompt, regardless of how problematic the request, would be overly harmful and toxic. In [Figure 21](#figure-21), leveraging our internal benchmarks, we explore how different models and systems in industry navigate this trade off and how Llama 3 compares. We find that our models achieve very competitive violation rate metrics while keeping false refusal rate low as well, indicating a solid balance between helpfulness and safety.

**Multilingual safety.** Our experiments demonstrate that safety knowledge in English does not readily transfer to other languages, particularly given the nuance of safety policies and language-specific context. Therefore, it is essential to collect high-quality safety data for each language. We also found that the distribution of safety data per language significantly impacts performance from a safety standpoint, with some languages benefiting from transfer learning while others require more language-specific data. To achieve a balance between FRR and VR, we iteratively add adversarial and borderline data while monitoring the impact on both metrics.

We display results on our internal benchmarks in [Figure 19](#figure-19) for short context models, showing Llama 3’s violation and false refusal rates for English and non-English languages compared to similar models and systems. To construct the benchmarks for each language, we use a combination of prompts written by native speakers, sometimes supplementing with translations from our English benchmarks. For each of our supported languages, we find that Llama 405B with Llama Guard is at least as safe, if not strictly safer, than the two competing systems when measured on our internal benchmark, while maintaining competitive false refusal rates. Looking at the Llama 405B model on its own, without Llama Guard, we find that it has a significantly lower violation rate than the competing standalone open source model, trading off a higher false refusal rate.
<span id="figure-21"></span>

![Original paper Figure 21](../../papers/llama-3/figure-21.png)

**Figure 21.** Violation and false refusal rates across models and capabilities. Each point represents the overall false refusal and violation rate for an internal capability benchmark across all safety categories. Symbols indicate whether we are evaluating model or system level safety. As expected model level safety results indicate higher violation rates and lower refusal rates compared to system level safety results. Llama 3 aims to balance a low violation rate with a low false refusal rate, while some competitors are more skewed towards one or the other.

**Long-context safety.** Long-context models are vulnerable to many-shot jailbreaking attacks without targeted mitigation [Ani24]. To address this, we finetune our models on SFT datasets that include examples of safe behavior in the presence of demonstrations of unsafe behavior in context. We develop a scalable mitigation strategy that significantly reduces VR, effectively neutralizing the impact of longer context attacks even for 256-shot attacks. This approach shows little to no impact on FRR and most helpfulness metrics.

To quantify the effectiveness of our long context safety mitigations, we use two additional benchmarking methods: DocQA and Many-shot. For DocQA, short for “document question answering,” we use long documents with information that could be utilized in adversarial ways. Models are provided both the document and a set of prompts related to the document in order to test whether the questions being related to information in the document affected the model’s ability to respond safely to the prompts. For Many-shot, following [Ani24], we construct a synthetic chat history composed of unsafe prompt-response pairs. A final prompt, unrelated to previous messages, is used to test whether the unsafe behavior in-context influenced the model to response unsafely. The violation and false refusal rates for both DocQA and Many-shot are shown in [Figure 20](#figure-20). We see that Llama 405B (with and without Llama Guard) is Pareto-better than the Comp. 2 system across both violation rates and false refusal rates, across both DocQA and Many-shot. Relative to Comp. 1, we find that Llama 405B is significantly safer, while coming at a trade off on false refusal.

**Tool usage safety.** The diversity of possible tools and the implementation of the tool usage call and integration into the model make tool usage a challenging capability to fully mitigate [Wal24a]. We focus on the search usecase. Violation and false refusal rates are shown in [Figure 20](#figure-20). We tested against the Comp. 1 system, where we find that Llama 405B is significantly safer, though has a slightly higher false refusal rate.
<span id="section-5-4-5"></span>

#### 5.4.5 Cybersecurity and Chemical/Biological Weapons Safety

**CyberSecurity evaluation results.** To evaluate cybersecurity risk, we leverage the CyberSecEval benchmark framework [Bha23, Bha24], which contains tasks that measure safety across domains such as generating insecure code, generating malicious code, textual prompt injection, and vulnerability identification. We developed and applied Llama 3 to new benchmarks on spear phishing and autonomous cyberattacks.

Overall, we find that Llama 3 does not have significant susceptibilities in generating malicious code or exploiting vulnerabilities. We describe brief results on specific tasks:

- **Insecure coding testing framework:** Evaluating Llama 3 8B, 70B, and 405B against the insecure coding testing framework, we continue to observe that larger models both generate more insecure code and also generate code with a higher average BLEU score [Bha23].
- **Code interpreter abuse prompt corpus:** We identify that Llama 3 models are susceptible to executing malicious code under certain prompts, with Llama 3 405B being particularly susceptible by complying with malicious prompts 10.4% of the time. Llama 3 70B complied at a rate of 3.8%.
- **Text-based prompt injection benchmark:** When evaluated against prompt injection benchmarks, prompt injection attacks against Llama 3 405B were successful 21.7% of the time. [Figure 23](#figure-23) provides text-based prompt injection success rates across Llama 3, GPT-4 Turbo, Gemini Pro, and Mixtral models.
- **Vulnerability identification challenges:** In assessing Llama 3’s ability to identify and exploit vulnerabilities using CyberSecEval 2’s capture-the-flag test challenges, Llama 3 does not outperform commonly used, traditional non-LLM tools and techniques.
- **Spear phishing benchmark:** We evaluate model persuasiveness and success rate in carrying out personalized conversations designed to deceive a target into unwittingly participating in security compromises. Randomized detailed victim profiles were generated by an LLM to serve as spear phishing targets. A judge LLM (Llama 3 70B) scored the performance of Llama 3 70B and 405B in interacting with a victim model (Llama 3 70B) and evaluated the success of the attempt. Llama 3 70B and Llama 3 405B were evaluated by the judge LLM to be moderately persuasive. Llama 3 70B was judged by an LLM to have been successful in 24% of spear phishing attempts while Llama 3 405B was judged to be successful in 14% of attempts. [Figure 23](#figure-23) presents judge LLM-evaluated persuasiveness scores across models and phishing objectives.
- **Attack automation framework:** We assess Llama 3 70B’s and 405B’s potential to function as an autonomous agent across four critical phases of a ransomware attack – network reconnaissance, vulnerability identification, exploit execution, and post exploitation actions. We enable the models to behave autonomously by configuring the models to iteratively generate and execute new Linux commands in response to output from their prior commands on a Kali Linux virtual machine as they targeted another virtual machine with known vulnerabilities. Although Llama 3 70B and 405B efficiently identify network services and open ports in their network reconnaissance, the models fail to effectively use this information to gain initial access to the vulnerable machine across 20 and 23 test runs respectively. In identifying vulnerabilities, Llama 3 70B and 405B are moderately effective but struggle with selecting and applying successful exploitation techniques. Attempts to execute exploits were entirely unsuccessful as were post-exploit attempts to maintain access or impact hosts within a network.
<span id="figure-22"></span>

![Original paper Figure 22](../../papers/llama-3/figure-22.png)

**Figure 22.** Text-based prompt injection success rates per model across prompt injection strategies. Llama 3 is on average more susceptible to prompt injection than GPT-4 Turbo and Gemini Pro but less susceptible than Mixtral models when evaluated using this benchmark.
<span id="figure-23"></span>

![Original paper Figure 23](../../papers/llama-3/figure-23.png)

**Figure 23.** Average spear phishing persuasiveness scores across spear phisher models and goals. Attempt persuasiveness is evaluated by a Llama 3 70B judge LLM.

**Uplift testing for cyber attacks.** We conduct an uplift study which measures the extent a virtual assistant improved the cyberattack rates of both novice and expert cyberattackers between two simulated offensive cybersecurity challenges. A two-stage study was conducted with 62 internal volunteers. Volunteers were categorized into “expert” (31 subjects) and “novice” (31 subjects) cohorts based on their offensive security experience. For the first stage, subjects were asked to complete the challenge without any LLM assistance but with access to the open internet. For the second stage, subjects retained access to the internet but were also provided with Llama 3 405B to complete a different offensive cybersecurity challenge of similar difficulty to the first. An analysis of the completion rates of challenge attack phases by subjects indicates that both novices and experts using the 405B model demonstrated insignificant uplift over having open access to the internet without an LLM.

**Uplift testing for chemical and biological weapons.** To assess risks related to proliferation of chemical and biological weapons, we perform uplift testing designed to assess whether use of Llama 3 could meaningfully increase the capabilities of actors to plan such attacks.

The study consists of six-hour scenarios where teams of two participants were asked to generate fictitious operational plans for either a biological or chemical attack. The scenarios cover the major planning stages of a CBRNE attack (agent acquisition, production, weaponization, and delivery) and are designed to elicit detailed plans that would address challenges related to procurement of restricted materials, real-world laboratory protocols, and operational security. Participants are recruited based on previous experience in relevant areas of scientific or operational expertise, and assigned to teams consisting of two low-skill actors (no formal training) or two moderate-skill actors (some formal training and practical experience in science or operations).

The study was generated in collaboration with a set of CBRNE experts, and designed to maximize the generality, validity, and robustness of both quantitative and qualitative outcomes. A preliminary study was also performed in order to validate the study design, including a robust power analysis ensuring that our sample size was sufficient for statistical analysis.

Each team is assigned to a “control” or “LLM” condition. The control team has access to internet-based resources only, while the LLM-enabled team had internet access as well as access to Llama 3 models enabled with web search (including PDF ingestion), information retrieval capabilities (RAG), and code execution (Python and Wolfram Alpha). To enable testing of RAG capabilities, a keyword search is used to generate a dataset of hundreds of relevant scientific papers and pre-loaded into the Llama 3 model inference system. At the conclusion of the exercise, the operational plans generated by each team are evaluated by subject matter experts with domain expertise in biology, chemistry, and operational planning. Each plan is evaluated across four stages of potential attacks, generating scores for metrics such as scientific accuracy, detail, detection avoidance, and probability of success in scientific and operational execution. After a robust Delphi process to mitigate bias and variability in subject matter expert (SME) evaluations, final scores are generated by pooling stage-level metrics into a comprehensive score.

Quantitative analysis of these results of this study show no significant uplift in performance related to usage of the Llama 3 model. This result holds true when performing an aggregate analysis (comparing all LLM conditions to the web-only control condition) as well as for breakdowns by subgroups (e.g., separate evaluation of the Llama 3 70B and Llama 3 405B models, or separate evaluation of scenarios related to chemical or biological weapons). After validating these results with CBRNE SMEs, we assess that there is a low risk that release of Llama 3 models will increase ecosystem risk related to biological or chemical weapon attacks.
<span id="section-5-4-6"></span>

#### 5.4.6 Red Teaming

We utilize Red Teaming to discover risks and use the findings to improve our benchmarks and safety tuning datasets. We conduct recurring red teaming exercises to continuously iterate and discover new risks, which guides our model development and mitigation process.

Our red team consists of experts in cybersecurity, adversarial machine learning, responsible AI, and integrity, in addition to multilingual content specialists with backgrounds in integrity issues for specific geographic markets. We also partner with internal and external subject-matter experts in critical risk areas to help build risk taxonomies and aid in more focused adversarial assessment.

**Adversarial testing on specific model capabilities.** We began initial red teaming by focusing on individual model capabilities in a risk discovery process, in context of specific high-risk categories then testing capabilities together. The red team focused on prompt-level attacks to emulate more likely more real world scenarios — we find that models often deviate from expected behavior, particularly in cases when the prompt’s intention is being obfuscated or when prompts layer multiple abstractions. These risks get more complex with additional capabilities, and we describe several of our red teaming discoveries in detail below. We utilize these red team discoveries in concert with our results on internal safety benchmarks to develop focused mitigations to continuously and iteratively improve model safety.

- **Short and long-context English.** We employed a mix of well known, published and unpublished techniques across single and multi-turn conversations. We also leveraged advanced, adversarial multi-turn automation similar to PAIR [Cha23a] across some techniques and risk categories. Largely, multi-turn conversations lead to more harmful outputs. Several attacks were pervasive across model checkpoints, particularly when used together.
  - Multi-turn refusal suppression to specify the model response to follow a particular format or include/exclude particular information related to the refusal as specific phrases.
  - Hypothetical scenarios wrap violating prompts as hypothetical/theoretical tasks or fictional scenarios. Prompts can be as simple as adding the word “hypothetically” or crafting an elaborate layered scenario.
  - Personas and role play gives the model a violating persona with specific violating response characteristics (e.g. “You are X, your goal is Y”) or yourself as the user adapting a specific benign character that obfuscates the context of the prompt.
  - Adding disclaimers and warnings works as a form of response priming and we assume a method to allow for the model a path to helpful compliance that intersects with generalized safety training. Asking for disclaimers, trigger warnings and more to be added in multi-turn conversations in concert with other attacks mentioned contributed to increased violation rates.
  - Gradually escalating violation is a multi-turn attack where the conversation starts out with a more or less benign request and then through direct prompting for more exaggerated content can gradually lead the model into generating a very violating response. Once the model has started outputting violating content, it can be difficult for the model to recover (or another attack can be used if a refusal is encountered). With longer context models, this will be an increasingly seen issue.
- **Multilingual.** We identify a number of unique risks when considering multiple languages.
  - Mixing multiple languages in one prompt or conversation can easily lead to more violating outputs than if a single language was used.
  - Lower resource languages can lead to violating outputs given a lack of related safety fine tuning data, weak model generalization of safety or prioritization of testing or benchmarks. However, this attack often result in poor quality generally, limiting real adversarial use.
  - Slang, specific context or cultural-specific references can confuse or appear to be violating at first glance, only to see the model does not comprehend a given reference correctly to make an output truly harmful or prevent it from being a violating output.
- **Tool use.** During testing, apart from English-text level adversarial prompting techniques being successful in generating violating outputs, several tool specific attacks were also discovered. This included but was not limited to:
  - Unsafe tool chaining such as asking for multiple tools at once with one being violating could, in early checkpoints, lead to all of the tools being called with a mix of benign and violating inputs.
  - Forcing tool use often with specific input strings, fragmented or encoded text can trigger a tool input to be potentially violating, leading to a more violating output. Other techniques can then be used to access the tool results, even if the model would normally refuse to perform the search or assist with the results.
  - Modifying tool use parameters such as swapping words in queries, retrying, or obfuscating some of the initial request in a multi-turn conversation lead to violations in many early checkpoints as a form of forcing tool use.

**Child safety risks.** Child Safety risk assessments were conducted using a team of experts, to assess the model’s capability to produce outputs that could result in Child Safety risks and inform on any necessary and appropriate risk mitigations via fine tuning. We leveraged those expert red teaming sessions to expand the coverage of our evaluation benchmarks through model development. For Llama 3, we conducted new in-depth sessions using objective based methodologies to assess model risks along multiple attack vectors. We also partnered with content specialists to perform red teaming exercises assessing potentially violating content while taking account of market specific nuances or experiences.
<span id="section-5-4-7"></span>

#### 5.4.7 System Level Safety

In various real-world applications of large language models, models are not used in isolation but are integrated into broader systems. In this section, we describe our system level safety implementation, which supplements model-level mitigations by providing more flexibility and control. To enable this, we develop and release a new classifier, Llama Guard 3, which is a Llama 3 8B model fine-tuned for safety classification. Similar to Llama Guard 2 [Lla24], this classifier is used to detect whether input prompts and/or output responses generated by language models violate safety policies on specific categories of harm. It is designed to support Llama’s growing capabilities, and can be used for English and multilingual text. It is also optimized to be used in the context of tool-calls such as search-tools and preventing code interpreter abuse. Finally, we also provide quantized variants to reduce memory requirements. We encourage developers to use our release of system safety components as a foundation and configure them for their own use cases.

**Taxonomy.** We train on the 13 hazard categories listed in the AI Safety taxonomy [Vid24]: Child Sexual Exploitation, Defamation, Elections, Hate, Indiscriminate Weapons, Intellectual Property, Non-Violent Crimes, Privacy, Sex-Related Crimes, Sexual Content, Specialized Advice, Suicide & Self-Harm, and Violent Crimes. We also train on Code Interpreter Abuse category to support tool-calls use cases.

**Training data.** We start with the English data used by Llama Guard [Ina23] and expand this dataset to incorporate new capabilities. For new capabilities such as multilingual and tool use, we collect prompt and response classification data, as well as utilize the data collected for safety finetuning. We increase the number of unsafe responses in the training set by doing prompt engineering to get the LLM to not refuse responding to adversarial prompts. We use Llama 3 to obtain response labels on such generated data.

To improve the performance of Llama Guard 3, we do extensive cleaning of the collected samples using human annotation as well as LLM annotation by Llama 3. Obtaining labels for user prompts is a much harder task for both humans and LLMs, and we find that the human labels are slightly better, especially for borderline prompts, though our full iterative system is able to reduce the noise and produce more accurate labels.

**Results.** Llama Guard 3 is able to significantly reduce violations across capabilities (-65% violations on average across our benchmarks). Note that adding system safeguards (and any safety mitigations in general) comes at the cost of increased refusals to benign prompts. In [Table 25](#table-25) we report reductions in violation rate and increases in false refusal rate increase compared to the base model to highlight this tradeoff. This effect is also visible in [Figure 19](#figure-19), [20](#figure-20), and [21](#figure-21). System safety also offers more flexibility. Llama Guard 3 can be deployed for specific harms only enabling control over the violations and false refusals trade-off at the harm category level. [Table 26](#table-26) presents violations reduction per category to inform which category should be turned on/off based on the developer use case. To make it easier to deploy safety systems, we provide a quantized version of Llama Guard 3 using the commonly used int8 quantization technique, reducing its size by more than 40%. [Table 27](#table-27) illustrates that quantization has negligible impact on the performance of the model.
<span id="table-25"></span>

![Original paper Table 25](../../papers/llama-3/table-25.png)

**Table 25.** Violation Rate (VR) and False Refusal Rate (FRR) relative to Llama 3 when using Llama Guard 3 for input or output filtering on different languages. For example, -50% for VR means that there is a 50% reduction in the rate of Llama 3 model violations when using Llama Guard. Evaluations are performed on generations from the 405B-parameter Llama 3 model. Lower is better.
<span id="table-26"></span>

![Original paper Table 26](../../papers/llama-3/table-26.png)

**Table 26.** Violation rate and false refusal rate relative to Llama 3 when using Llama Guard 3 for input or output filtering on different safety categories. For example, -50% for VR means that there is a 50% reduction in the rate of Llama 3 model violations when using Llama Guard. Evaluations are performed on English prompts and generations from the 405B parameter Llama 3 model. Lower is better.
<span id="table-27"></span>

![Original paper Table 27](../../papers/llama-3/table-27.png)

**Table 27.** int8 Llama Guard. Effect of int8 quantization on Llama Guard 3 output classification performance for different model capabilities.

**Prompt-based system guards.** System-level safety components enable developers to customize and control how LLM systems respond to user requests. As part of our work on improving the overall safety of the model system and enable developers to deploy responsibly, we describe and release the creation of two prompt-based filtering mechanisms: Prompt Guard and Code Shield. We open-source these for the community to leverage as-is or take as inspiration and adapt for their usecases. Prompt Guard is a model-based filter designed to detect prompt attacks, which are input strings designed to subvert the intended behavior of an LLM functioning as part of an application. The model is a multi-label classifier that detects two classes of prompt attack risk - direct jailbreaks (techniques that explicitly try to override a model’s safety conditioning or system prompt) and indirect prompt injections (instances where third-party data included in a model’s context window includes instructions inadvertently executed as user commands by an LLM). The model is fine-tuned from mDeBERTa-v3-base, a small (86M) parameter model suitable for filtering inputs into an LLM. We evaluate the performance on several evaluation datasets shown in [Table 28](#table-28). We evaluate on two datasets (jailbreaks and injections) drawn from the same distribution as the training data, as well as an out-of-distribution dataset in English, a multilingual jailbreak set built from machine translation, and a dataset of indirect injections drawn from CyberSecEval (both English and multilingual). Overall, we find that the model generalizes well to new distributions and has strong performance.
<span id="table-28"></span>

![Original paper Table 28](../../papers/llama-3/table-28.png)

**Table 28.** Performance of Prompt Guard. We include in- and out-of-distribution evaluations, a multilingual jailbreak built using machine translation, and a dataset of indirect injections from CyberSecEval.

**Code Shield.** Code Shield is an example of a class of system-level protections based on providing inference-time filtering. In particular, it focuses on detecting the generation of insecure code before it might enter a downstream usecase such as a production system. It does so by leveraging a static analysis library, the Insecure Code Detector (ICD), to identify insecure code. ICD uses a suite of static analysis tools to perform the analysis across 7 programming languages. These kinds of guardrails are generally useful for developers, who can deploy multi-layered protections in various applications.
<span id="section-5-4-8"></span>

#### 5.4.8 Limitations

We conducted extensive measurement and mitigation on a wide variety of risks to safe usage of Llama 3. However, no testing can be guaranteed to be exhaustive in identifying every possible risk. Llama 3 may still generate harmful content due to training on various datasets, particularly for languages beyond English and when prompt engineered by skilled adversarial red teamers. Malicious developers or adversarial users may find new ways to jailbreak our models and use them for various nefarious usecases. We will continue to proactively identify risks, conduct research on mitigation methods, and we encourage developers to consider responsibility in every aspect — from model development to deployment to users. We hope developers will leverage and contribute to the tools we release in our open-source system-level safety suite.
<span id="section-6"></span>

## 6 Inference

We investigate two main techniques to make inference with the Llama 3 405B model efficient: (1) pipeline parallelism and (2) FP8 quantization. We have publicly released our implementation of FP8 quantization.
<span id="section-6-1"></span>

### 6.1 Pipeline Parallelism

When using a BF16 number representation for the model parameters, Llama 3 405B does not fit in the GPU memory of a single machine with 8 Nvidia H100 GPUs. To address this issue, we parallelize model inference using BF16 precision across 16 GPUs on two machines. Within each machine, the high NVLink bandwidth enables the use of tensor parallelism [Sho19]. Across nodes, however, connectivity has lower bandwidth and higher latency, so we use pipeline parallelism [Hua19b] instead. During training with pipeline parallelism, bubbles are a major efficiency concern (see Section [3.3](#section-3-3)). However, they are not an issue during inference, since inference does not involve a backward pass that requires a pipeline flush. Therefore, we use micro-batching to improve inference throughput with pipeline parallelism. We evaluate the effect of using two micro-batches in inference workloads of 4,096 input tokens and 256 output tokens both during the key-value cache *pre-fill* stage of inference and during the *decoding* stage. We find that micro-batching improves throughput of inference with the same local batch size; see [Figure 24](#figure-24). These improvements result from micro-batching enabling concurrent execution of micro batches in both these stages. The additional synchronization points due to micro-batching also increase latency but, overall, micro-batching still leads to a better throughput-latency trade-off.
<span id="figure-24"></span>

![Original paper Figure 24](../../papers/llama-3/figure-24.png)

**Figure 24.** Effect of micro-batching on inference throughput and latency during the *Left:* pre-filling and *Right:* decoding stage. The numbers in the plot correspond to the (micro-)batch size.
<span id="section-6-2"></span>

### 6.2 FP8 Quantization

We perform experiments leveraging the native FP8 support of H100 GPUs to perform low-precision inference. To enable low-precision inference, we apply FP8 quantization to most matrix multiplications inside the model. In particular, we quantize most parameters and activations in the feedforward network layers in the model, which account for roughly 50% of the inference compute time. We do not quantize parameters in the self-attention layers of the model. We leverage dynamic scaling factors for better accuracy [Xia24b], optimizing our CUDA kernels [+15] to reduce the overhead of calculating the scales. We find that the quality of Llama 3 405B is sensitive to certain types of quantization, and make a few additional changes to increase the model output quality: 1. Akin to [Zha21d], we do not perform quantization in the first and last Transformer layers. 2. High-perplexity tokens such as dates can lead to large activation values. In turn, these can lead to high dynamic scaling factors in FP8 and a non-negligible number of underflows, leading to errors in decoding. To address this issue, we upper bound the dynamic scaling factors to $1200$. 3. We use row-wise quantization, computing scaling factors across rows for parameter and activation matrices (see [Figure 25](#figure-25)). We find this works better than a tensor-wise quantization approach.
<span id="figure-25"></span>

![Original paper Figure 25](../../papers/llama-3/figure-25.png)

**Figure 25.** Illustration of tensor-wise and row-wise FP8 quantization. *Right:* Row-wise quantization enables the use of more granular activation factors than *Left:* tensor-wise quantization.

**Effect of quantization errors.** Evaluations on standard benchmarks often suggest that FP8 inference performs on par with BF16 inference even without these mitigations. However, we find that such benchmarks do not adequately reflect the effects of FP8 quantization. When scaling factors are not upper bounded, the model occasionally produces corrupted responses even though the benchmark performance is strong. Instead of relying on benchmarks to measure distribution changes due to quantization, we find it is better to analyze the distribution of reward-model scores for $100,000$ responses produced using both FP8 and BF16. [Figure 26](#figure-26) shows the resulting reward distribution for our quantization approach. The results in the figure show that our approach to FP8 quantization has very limited impact on the model’s response.
<span id="figure-26"></span>

![Original paper Figure 26](../../papers/llama-3/figure-26.png)

**Figure 26.** Reward score distribution for Llama 3 405B using BF16 and FP8 inference. Our FP8 quantization approach has negligible impact on the model’s responses.

**Experimental evaluation of efficiency.** [Figure 27](#figure-27) depicts the throughput-latency trade-off of performing FP8 inference with Llama 3 405B in the pre-fill and decoding stages, using 4,096 input tokens and 256 output tokens. The figure compares the efficiency of FP8 inference with that of the two-machine BF16 inference approach described in Section [6.1](#section-6-1). The results show that use of FP8 inference leads to throughput improvements of up to 50% during the pre-fill stage, and a substantially better throughput-latency trade-off during decoding.
<span id="figure-27"></span>

![Original paper Figure 27](../../papers/llama-3/figure-27.png)

**Figure 27.** Throughput-latency trade-off in FP8 inference with Llama 3 405B compared with BF16 inference using different pipeline parallelization setups. *Left:* Results for pre-filling. *Right:* Results for decoding.
<span id="figure-28"></span>

![Original paper Figure 28](../../papers/llama-3/figure-28.png)

**Figure 28.** Illustration of the compositional approach to adding multimodal capabilities to Llama 3 that we study in this paper. This approach leads to a multimodal model that is trained in five stages: (1) language model pre-training, (2) multi-modal encoder pre-training, (3) vision adapter training, (4) model finetuning, and (5) speech adapter training.
<span id="section-7"></span>

## 7 Vision Experiments

We perform a series of experiments in which we incorporate visual-recognition capabilities into Llama 3 via a compositional approach that consists of two main stages. First, we compose a pre-trained image encoder [Xu23a] and the pre-trained language model by introducing and training a set of cross-attention layers between the two models [Ala22] on a large number of image-text pairs. This leads to the model illustrated in [Figure 28](#figure-28). Second, we introduce temporal aggregator layers and additional video cross-attention layers that operate on a large collection of video-text pairs to learn the model to recognize and process temporal information from videos. A compositional approach to foundation model development has several advantages: (1) it enables us to parallelize the development of the vision and language modeling capabilities; (2) it circumvents complexities of joint pre-training on visual and language data that stem from tokenization of visual data, differences in background perplexities of tokens originating from different modalities, and contention between modalities; (3) it guarantees that model performance on text-only tasks is not affected by the introduction of visual-recognition capabilities, and (4) the cross-attention architecture ensures that we do not have to expend compute passing full-resolution images through the increasingly LLM backbones (specifically, the feed-forward networks in each transformer layer), making it more efficient during inference. We note that our multimodal models are still under development and not yet ready for release. Before presenting the results of our experiments in Section [7.6](#section-7-6) and [7.7](#section-7-7), we describe the data we used to train visual recognition capabilities, the model architecture of the vision components, how we scale training of those components, and our pre-training and post-training recipes.
<span id="section-7-1"></span>

### 7.1 Data

We describe our image and video data separately below.
<span id="section-7-1-1"></span>

#### 7.1.1 Image Data

Our image encoder and adapter are trained on image-text pairs. We construct this dataset via a complex data processing pipeline that consists of four main stages: (1) quality filtering, (2) perceptual de-duplication, (3) resampling, and (4) optical character recognition. We also apply a series of safety mitigations.

- **Quality filtering.** We implement quality filters that remove non-English captions and low-quality captions via heuristics such as low alignment scores produced by [Rad21]. Specifically, we remove all image-text pairs below a certain CLIP score.
- **De-duplication.** De-duplicating large-scale training datasets benefits model performance because it reduces training compute spent on redundant data [Ess24, Lee21, Abb23] and memorization [Car23, Som23]. Hence, we de-duplicate our training data for both efficiency and privacy reasons. To do so, we use an internal version of the state-of-the-art SSCD copy-detection model [Piz22] to de-duplicate images at scale. For all images, we first compute a 512-dimensional representation using the SSCD model. We use those embeddings to perform a nearest neighbor (NN) search for each image across all images in our data set, using a cosine similarity measure. We define examples above a certain similarity threshold as duplicates. We group these duplicates using a connected-components algorithm, and maintain only one image-text pair per connected component. We increase the efficiency of our de-duplication pipeline by: (1) pre-clustering the data using k-means clusters and (2) using FAISS [Joh19] for NN searches and clustering.
- **Resampling.** We ensure diversity of the image-text pairs via resampling akin to [Xu23a, Mah18, Mik13]. First, we construct a vocabulary of n-grams by parsing high-quality text sources. Next, we compute the frequency of each vocabulary n-gram in our dataset. We then resample the data as follows: If any of the n-grams in a caption occurs less than $T$ times in the vocabulary, we keep the corresponding image-text pair. Otherwise, we independently sample each of the n-grams $n_{i}$ in the caption with probability $\sqrt{T/f_{i}}$ where $f_{i}$ indicates the frequency of n-gram $n_{i}$; we keep the image-text pair if any of the n-grams was sampled. This resampling aids performance on low-frequency categories and fine-grained recognition tasks.
- **Optical character recognition.** We further improve our image-text data by extracting text written in the image and concatenating it with the caption. The written text is extracted using a proprietary optical character recognition (OCR) pipeline. We observe that adding OCR data into the training data greatly improves tasks that require OCR capabilities, such as document understanding.

**Transcribing documents.** To improve the performance of our models on document understanding tasks, we render pages from documents as images and paired the images with their respective text. The document text is obtained either directly from the source or via a document parsing pipeline.

**Safety.** We focus primarily on ensuring that the pre-training dataset for image recognition does not contain unsafe content, such as sexual abuse material (CSAM) [Thi23]. We scan all our training images for CSAM using perceptual hashing approaches such as PhotoDNA [Far21] as well as internal, proprietary classifiers. We also use a proprietary media-risk retrieval pipeline to identify and remove image-text pairs that we consider to be NSFW, for example, because they contain sexual or violent content. We believe that minimizing the prevalence of such material in the training dataset improves the safety of the final model without impacting its helpfulness. Finally, we perform face blurring on all images in our training set. We test the model against human generated prompts that refer to an attached image.

**Annealing data.** We create an annealing dataset by resampling the image-caption pairs to a smaller volume of $\sim$350M examples using n-grams. Since the n-grams resampling favor richer text descriptions, this selects a higher-quality data subset. We augment the resulting data with $\sim$150M examples from five additional sources:

- **Visual grounding.** We link noun phrases in the text to bounding boxes or masks in the image. The grounding information (bounding boxes and masks) are specified in the image-text pair in two ways. (1) We overlay boxes or masks with marks on the image and use marks in the text as reference, akin to set-of-marks [Yan23b]. (2) We insert normalized $(x_{\min},y_{\min},x_{\max},y_{\max})$ coordinates directly into the text, demarcated by special tokens.
- **Screenshot parsing.** We render screenshots from HTML code and task the model with predicting the code that produced a specific element in the screenshot, akin to [Lee23a]. The element of interest is indicated in the screenshot via a bounding box.
- **Question-answer pairs.** We include question-answer pairs, enabling us to use volumes of question-answering data that are too large to be used in model finetuning.
- **Synthetic captions.** We include images with synthetic captions that were generated by an early version of the model. Compared to original captions, we find that synthetic captions provide a more comprehensive description of images than the original captions.
- **Synthetically-generated structured images.** We also include synthetically generated images for a variety of domains such as charts, tables, flowcharts, math equations and textual data. These images are accompanied by a structured representation such as the corresponding markdown or LaTeX notation. Besides improving recognition capabilities of the model for these domains, we find this data useful to generate question-answer pairs via the text model for finetuning.
<span id="section-7-1-2"></span>

#### 7.1.2 Video Data

For video pre-training, we use a large dataset of video-text pairs. Our dataset is curated through a multi-stage process. We filter and clean the associated texts using rule-based heuristics, such as ensuring a minimum length and fixing capitalization. Then, we run language identification models to filter out non-English texts. We run OCR detection models to filter out videos with excessive overlaid text. To ensure reasonable alignment between the video-text pairs, we use CLIP [Rad21] style image-text and video-text contrastive models. We first compute image-text similarity using a single frame in the videos and filtered out low similarity pairs, and then subsequently filter out pairs with low video-text alignment. Some of our data contains static or low-motion videos; we filter out such data using motion-score based filtering [Gir23]. We do not apply any filters on the visual quality of the videos such as aesthetic scores or resolution filtering. Our dataset contains videos with an average duration of 21 seconds and a median duration of 16 seconds, with over $99\%$ videos being under a minute. The spatial resolution varies significantly between 320p and 4K videos, with over $70\%$ of the videos having a short side greater than 720 pixels. The videos have varying aspect ratios with almost all videos having between aspect ratio between $1{:}2$ and $2{:}1$, with a $1{:}1$ median.
<span id="section-7-2"></span>

### 7.2 Model Architecture

Our visual-recognition model consists of three main components: (1) an image encoder, (2) an image adapter, and (3) a video adapter.

**Image encoder.** Our image encoder is a standard vision transformer (ViT; [Dos20]) that is trained to align images and text [Xu23a]. We use the ViT-H/14 variant of the image encoder, which has 630M parameters that were trained on 2.5B image-text pairs for five epochs. The image encoder is pre-trained on images with resolution $224\times 224$; images were split up into $16\times 16$ patches of equal size (*i.e.*, a patch size of $14x14$ pixels). As also demonstrated by prior work such as ViP-Llava [Cai24a], we observe that image encoders trained via a contrastive text alignment objective are unable to preserve fine-grained localization information. To alleviate this, we employ a *multi-layer* feature extraction, where features from the *4${}^{\mathrm{th}}$, 8${}^{\mathrm{th}}$, 16${}^{\mathrm{th}}$, 24${}^{\mathrm{th}}$ and 31${}^{\mathrm{st}}$* layers are also provided in addition to the final layer features. In addition, we further insert 8 *gated* self-attention layers (making a total of 40 transformer blocks) prior to pre-training of the cross-attention layers to learn alignment-specific features. The image encoder therefore eventually has a total $850$M parameters with the additional layers. With the multi-layer features, the image encoder produces a $7680$-dimensional representation for each of the resulting $16\times 16\!=\!256$ patches. The parameters of the image encoder are *not* frozen during subsequent training stages as we found it to improve performance, especially in domains such as text recognition.

**Image adapter.** We introduce cross-attention layers between the visual token representations produced by the image encoder and the token representations produced by the language model [Ala22]. The cross-attention layers are applied after every fourth self-attention layer in the core language model. Like the language model itself, the cross-attention layers use generalized query attention (GQA) for increased efficiency. The cross-attention layers introduce substantial numbers of additional trainable parameters into the model: for Llama 3 405B, the cross-attention layers have $\approx$100B parameters. We pre-train our image adapter in two stages: (1) initial pre-training followed by (2) annealing:

- **Initial pre-training.** We pre-train our image adapter on our dataset of $\sim$6B image-text pairs described above. For compute efficiency reasons, we resize all images to fit within *at most* four tiles of $336\times 336$ pixels each, where we arrange the tiles to support different aspect ratios, *e.g.*, $672\times 672$, $672\times 336$, and $1344\times 336$.
- **Annealing.** We continue training the image adapter on $\sim$500M images from the annealing dataset described above. During annealing, we increase the per-tile image resolution to improve performance on tasks that require higher-resolution images, for example, infographics understanding.

**Video adapter.** Our model takes as input up to 64 frames (uniformly sampled from a full video), each of which is processed by the image encoder. We model temporal structure in videos through two components: (i) encoded video frames are aggregated by a temporal aggregator which merges 32 consecutive frames into one, (ii) additional video cross attention layers are added before every fourth image cross attention layer. The temporal aggregator is implemented as a perceiver resampler [Jae21, Ala22]. We pre-train using 16 frames per video (aggregated to 1 frame), but increase the number of input frames to 64 during supervised finetuning. The video aggregator and cross attention layers have 0.6B and 4.6B parameters for Llama 3 7B and 70B, respectively.
<span id="section-7-3"></span>

### 7.3 Model Scaling

After the visual-recognition components are added to Llama 3, the model contains self-attention layers, cross-attention layers, and a ViT image encoder. To train adapters for the smaller 8B and 70B parameter models, we found a combination of data and tensor parallelization is the most efficient. Model or pipeline parallelism does not increase efficiency at these scales because the gathering of model parameters would dominate the computation. We do, however, use pipeline parallelism (in addition to data and tensor parallelism) when training the adapter for the 405B parameter model. Training at this scale introduces three new challenges in addition to those outlined in Section [3.3](#section-3-3): model heterogeneity, data heterogeneity, and numerical instabilities.

**Model heterogeneity.** The model computation is heterogeneous because more computation is performed on some tokens than on others. In particular, image tokens are processed by the image encoder and the cross-attention layers, whereas text tokens are only processed by the language backbone. This heterogeneity leads to bottlenecks in the scheduling of pipeline parallelism. We address this problem by ensuring each pipeline stage contains five layers: namely, four self-attention layers in the language backbone and a cross-attention layer. (Recall that we introduce a cross-attention layer after every fourth self-attention layer.) In addition, we replicate the image encoder on all pipeline stages. Because we train on paired image-text data, this enables us to perform load balancing between the image and text parts of the computation.

**Data heterogeneity.** The data is heterogeneous because, on average, images have more tokens than the associated text: an image has 2,308 tokens, whereas the associated text contains an average of only 192 tokens. As a result, the computation of cross-attention layers requires more time and memory than the computation of self-attention layers. We address this problem by introducing sequence parallelization in the image encoder, so that each GPU processes roughly the same number of tokens. Because the average text size is relatively short, we also use a substantially larger micro-batch size (8 instead of 1).

**Numerical instabilities.** After the image encoder is added to the model, we find that performing gradient accumulation in bf16 led to numerical instabilities. The most likely explanation for this is that image tokens are introduced into the language backbone via *all* cross-attention layers. This implies that numerical deviations in the representation of an image token have an outsized impact on the overall computation because the errors are compounded. We address this by performing gradient accumulation in FP32.
<span id="section-7-4"></span>

### 7.4 Pre-training

**Image.** We initialize from the pre-trained text model and vision encoder weights. The vision encoder is unfrozen, while the text model weights are kept frozen as explained above. First, we train the model using 6B image-text pairs where each image is resized to fit within four tiles of $336\times 336$ pixels. We use a global batch size of 16,384 and a cosine learning rate schedule with initial learning rate $10\times 10^{-4}$ and a weight decay of $0.01$. The initial learning rate was determined based on small-scale experiments. However, these findings did not generalize well to very long training schedules and dropped the learning rate a few times during training when the loss values became stagnant. After the base pre-training, we increase the image resolution further and continue training the same weights on the annealing dataset. The optimizer is re-initialized via warm-up to learning rate $2\times 10^{-5}$ and again follows a cosine schedule.

**Video.** For video pre-training, we start from the image pre-trained and annealed weights as described above. We add the video aggregator and cross-attention layers as described in the architecture, initialized randomly. We freeze all the parameters in the model except the video-specific ones (the aggregator and video cross-attention), and train them on the video pre-training data. We use the same training hyperparameters as the image annealing stage, with small differences in the learning rate. We uniformly sample 16 frames from the full video, and represent each frame using four chunks, each of size of $448\times 448$ pixels. We use an aggregation factor of 16 in the video aggregator, hence obtaining one effective frame, which the text tokens cross-attend to. We use a global batch size of 4,096, a sequence length of 190 tokens, and a learning rate of $10^{-4}$ during training.
<span id="section-7-5"></span>

### 7.5 Post-Training

In this section, we describe the post-training recipe for our vision adapters. After pre-training, we fine-tune the model on highly curated multi-modal conversational data to enable chat capabilities. We further implement direct preference optimization (DPO) to boost human evaluation performance and rejection sampling to improve multi-modal reasoning capabilities. Finally, we add a quality-tuning stage where we continue fine-tuning the model on a very small set of high-quality conversational data which further boosts human evaluation while retaining performance across benchmarks. More details on each of these steps are provided below.
<span id="section-7-5-1"></span>

#### 7.5.1 Supervised Finetuning Data

We describe our supervised finetuning (SFT) data for image and video capabilities separately below.

**Image.** We utilize a mix of different datasets for supervised finetuning.

- **Academic datasets.** We convert a highly filtered collection of existing academic datasets to question-answer pairs using templates or via LLM rewriting. The LLM rewriting’s purpose is to augment the data with different instructions and to improve the language quality of answers.
- **Human annotations.** We collect multi-modal conversation data via human annotators for a wide range of tasks (open-ended question-answering, captioning, practical use cases, etc.) and domains (e.g., natural images and structured images). Annotators are provided with images and asked to write conversations. To ensure diversity, we cluster large-scale datasets and sampled images uniformly across different clusters. Further, we acquire additional images for a few specific domains by expanding a seed via k-nearest neighbors. Annotators are also provided with intermediate checkpoints of existing models to facilitate model-in-the-loop style annotations, so that model generations can be utilized as a starting point by the annotators to then provide additional human edits. This is an iterative process, in which model checkpoints would be regularly updated with better performing versions trained on the latest data. This increases the volume and efficiency of human annotations, while also improving their quality.
- **Synthetic data.** We explore different ways to generate synthetic multi-modal data by using text-representations of images and a text-input LLM. The high-level idea is to utilize the reasoning capabilities of text-input LLMs to generate question-answer pairs in the text domain, and replace the text representation with its corresponding images to produce synthetic multi-modal data. Examples include rendering texts from question-answer datasets as images or rendering table data into synthetic images of tables and charts. Additionally, we use captions and OCR extractions from existing images to generate additional conversational or question-answer data related to the images.

**Video.** Similar to the image adapter, we use academic datasets with pre-existing annotations and convert them into appropriate textual instructions and target responses. The targets are converted to open-ended responses or multiple-choice options, whichever is more appropriate. We ask humans to annotate videos with questions and corresponding answers. The annotators are asked to focus on questions that could not be answered based on a single frame, to steer the annotators towards questions that require temporal understanding.
<span id="section-7-5-2"></span>

#### 7.5.2 Supervised Finetuning Recipe

We describe our supervised finetuning (SFT) recipe for image and video capabilities separately below.

**Image.** We initialize from the pre-trained image adapter, but hot-swap the pre-trained language model’s weights with the instruction tuned language model’s weights. The language model weights are kept frozen to maintain text-only performance, i.e., we only update the vision encoder and image adapter weights.

Our approach to finetune the model is similar to [Wor22]. First, we run a hyperparameter sweep using multiple random subsets of data, learning rates and weight decay values. Next, we rank the models based on their performance. Finally, we average the weights of the top-$K$ models to obtain the final model. The value of $K$ is determined by evaluating the averaged models and selecting the instance with highest performance. We observe that the averaged models consistently yield better results compared to the best individual model found via grid search. Further, this strategy reduces sensitivity to hyperparameters.

**Video.** For video SFT, we initialize the video aggregator and cross-attention layers using the pre-trained weights. The rest of the parameters in the model, the image weights and the LLM, are initialized from corresponding models following their finetuning stages. Similar to video pre-training, we then finetune only the video parameters on the video SFT data. For this stage, we increase the video length to 64 frames, and use an aggregation factor of 32 to get two effective frames. The resolution of the chunks is also increased to be consistent with the corresponding image hyperparameters.
<span id="section-7-5-3"></span>

#### 7.5.3 Preference Data

We built multimodal pair-wise preference datasets for reward modeling and direct preference optimization.

- **Human annotations.** The human-annotated preference data consists of comparisons between two different model outputs, labeled as “chosen” and “rejected”, with 7-scale ratings. The models used to generate responses are sampled on-the-fly from a pool of the best recent models, each with different characteristics. We update the model pool weekly. Besides preference labels, we also request annotators to provide optional human edits to correct inaccuracies in “chosen” responses because vision tasks have a low tolerance for inaccuracies. Note that human editing is an optional step because there is a trade-off between volume and quality in practice.
- **Synthetic data.** Synthetic preference pairs could also be generated by using text-only LLMs to edit and deliberately introduce errors in the supervised finetuning dataset. We took the conversational data as input, and use an LLM to introduce subtle but meaningful errors (e.g., change objects, change attributes, add mistakes in calculations, etc.). These edited responses are used as negative “rejected” samples and paired with the “chosen” original supervised finetuning data.
- **Rejection sampling.** Furthermore, to create more *on-policy* negative samples, we leveraged the iterative process of rejection sampling to collect additional preference data. We discuss our usage of rejection sampling in more detail in the following sections. At a high-level, rejection sampling is used to iteratively sample high-quality generations from a model. Therefore, as a by-product, all generations that are not selected can be used as negative rejected samples and used as additional preference data pairs.
<span id="section-7-5-4"></span>

#### 7.5.4 Reward Modeling

We train a vision reward model (RM) on top of the vision SFT model and the language RM. The vision encoder and the cross-attention layers are initialized from the vision SFT model and unfrozen during training, while the self-attention layers are initialized from the language RM and kept frozen. We observe that freezing the language RM part generally leads to better accuracy, especially on tasks that require the RM to judge based on its knowledge or the language quality. We adopt the same training objective as the language RM, but adding a weighted regularization term on the square of the reward logits averaged over the batch, which prevents the reward scores from drifting. The human preference annotations in Section [7.5.3](#section-7-5-3) are used to train the vision RM. We follow the same practice as language preference data (Section [4.2.1](#section-4-2-1)) to create two or three pairs with clear ranking (*edited* > *chosen* > *rejected*). In addition, we also synthetically augment the negative responses by perturbing the words or phrases related to the information in the image (such as numbers or visual texts). This encourages the vision RM to ground its judgement based on the actual image content.
<span id="section-7-5-5"></span>

#### 7.5.5 Direct Preference Optimization

Similar to the language model (Section [4.1.4](#section-4-1-4)), we further train the vision adapters with Direct Preference Optimization (DPO; [Raf24]) using the preference data described in Section [7.5.3](#section-7-5-3). To combat the distribution shift during post-training rounds, we only keep recent batches of human preference annotations while dropping batches that are sufficiently off-policy (e.g., if the base pre-trained model is changed). We find that instead of always freezing the reference model, updating it in an exponential moving average (EMA) fashion every k-steps helps the model learn more from the data, resulting in better performance in human evaluations. Overall, we observed that the vision DPO model consistently performs better than its SFT starting point in human evaluations for every finetuning iteration.
<span id="section-7-5-6"></span>

#### 7.5.6 Rejection Sampling

Most available question-answer pairs only contain the final answer and lack the chain-of-thought explanation that is required to train a model that generalizes well for reasoning tasks. We use rejection sampling to generate the missing explanations for such examples and boost the model’s reasoning capabilities. Given a question-answer pair, we generate multiple answers by sampling the finetuned model with different system prompts or temperature. Next, we compare the generated answers to the ground-truth via heuristics or an LLM judge. Finally, we retrain the model by adding the correct answers back into the finetuning data mix. We find it useful to keep multiple correct answers per question. To ensure we only add high-quality examples back into training, we implemented the following two guardrails. First, we find that some examples contain incorrect explanations, despite the final answer being correct. We observed that this pattern occurs more frequently for questions where only a small fraction of the generated answers is correct. Therefore, we drop answers for questions where the probability of the answer being correct is below a certain threshold. Second, raters prefer some answers over others due to differences in language or style. We use the reward model to select top-$K$ highest-quality answers and add them back into training.
<span id="section-7-5-7"></span>

#### 7.5.7 Quality Tuning

We curate a small but *highly* selective SFT dataset where all samples have been rewritten and verified either by humans or our best models to meet our highest standards. We train DPO models with this data to improve response quality, calling the process Quality-Tuning (QT). We find that QT significantly improves human evaluations without affecting generalization verified by benchmarks when the QT dataset covers a wide range of tasks and proper early stopping is applied. We select checkpoints at this stage purely based on benchmarks to ensure capabilities are retained or improved.
<span id="section-7-6"></span>

### 7.6 Image Recognition Results

We evaluate the performance of the image understanding capabilities of Llama 3 on a range of tasks spanning natural image understanding, text understanding, charts understanding and multimodal reasoning:

- MMMU [Yue24b] is a challenging dataset for mulitmodal reasoning where model is expected to understand images and solve college-level problems spanning 30 different disciplines. This includes both multiple-choice and open ended questions. We evaluate our model on the validation set with 900 images, in line with other works.
- VQAv2 [Ant15] tests the ability of a model to combine image understanding, language understanding and commonsense knowlege to answer generic questions about natural images
- AI2 Diagram [Kem16] evaluates models capability to parse scientific diagrams and answer questions about the same. We use the same evaluation protocol as Gemini and x.ai, and report scores using a transparent bounding box.
- ChartQA [Mas22] is a challenging benchmark for charts understanding. This requires model to visually understand different kinds of charts and answer logical questions about the charts.
- TextVQA [Sin19a] is a popular benchmark dataset that requires models to read and reason about text in images to answer questions about them. This tests the OCR understanding ability of the model on natural images.
- DocVQA [Mat21] is a benchmark dataset focused on document analysis and recognition. It contains images of a wide range of documents which evaluates a model’s ability to perform OCR understanding and reason about the contents of a document to answer questions about them.

[Table 29](#table-29) presents the results of our experiments. The results in the table show that our vision module attached to Llama 3 performs competitively across a wide range of image-recognition benchmarks at varying model capacities. Using the resulting Llama 3-V 405B model, we outperform GPT-4V on all benchmarks, while being slightly behind Gemini 1.5 Pro and Claude 3.5 Sonnet. Llama 3 405B appears particularly competitive on document understanding tasks.
<span id="table-29"></span>

![Original paper Table 29](../../papers/llama-3/table-29.png)

**Table 29.** Image understanding performance of our vision module attached to Llama 3. We compare model performance to GPT-4V, GPT-4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet. ${}^{\triangle}$Results obtained using external OCR tools.
<span id="section-7-7"></span>

### 7.7 Video Recognition Results

We evaluate our video adapter for Llama 3 on three benchmarks:

- PerceptionTest [Pat23a] evaluates the model’s ability to answer temporal reasoning questions focusing on skills (memory, abstraction, physics, semantics) and different types of reasoning (descriptive, explanatory, predictive, counterfactual). It consists of $11.6K$ test QA pairs, each with an on-average $23s$ long video, filmed by $100$ participants worldwide to show perceptually interesting tasks. We focus on the multiple-choice question answering task, where each question is paired with three possible options. We report performance on the held-out test split which is accessed by submitting our predictions to an online challenge server. [+16]
- NExT-QA [Xia21] is another temporal and causal reasoning benchmark, with a focus on open-ended question answering. It consists of $1K$ test videos each on-average $44s$ in length, paired with $9K$ questions. The evaluation is performed by comparing the model’s responses with the ground truth answer using Wu-Palmer Similarity (WUPS) [Wu94]. [+17]
- TVQA [Lei18] evaluates the model’s ability to perform compositional reasoning, requiring spatiotemporal localization of relevant moments, recognition of visual concepts, and joint reasoning with subtitle-based dialogue. This dataset, being derived from popular TV shows, additionally tests for the model’s ability to leverage its outside-knowledge of those TV shows in answering the questions. It consists of over $15K$ validation QA pairs, with each corresponding video clip being on-average $76s$ in length. It also follows a multiple-choice format with five options for each question, and we report performance on the validation set following prior work [Ope23a].
- ActivityNet-QA [Yu19] evaluates the model’s ability to reason over long video clips to understand actions, spatial relations, temporal relations, counting, etc. It consists of $8K$ test QA pairs from $800$ videos, each on-average $3$ minutes long. For evaluation, we follow the protocol from prior work [Gem23, Lin23a, Maa24], where the model generates short one-word or one-phrase answers, and the correctness of the output is evaluated using the GPT-3.5 API which compares it to the ground truth answer. We report the average accuracy as evaluated by the API.

When performing inference, we uniformly sample frames from the full video clip and pass those frames into the model with a short text prompt. Since most of our benchmarks involve answering multiple-choice questions, we use the following prompt: Select the correct answer from the following options: {question}. Answer with the correct option letter and nothing else. For benchmarks that require producing a short answer (e.g., ActivityNet-QA and NExT-QA), we use the following prompt: Answer the question using a single word or phrase. {question}. For NExT-QA, since the evaluation metric (WUPS) is sensitive to the length and the specific words used, we additionally prompt the model to be specific and respond with the most salient answer, for instance specifying “living room” instead of simply responding with “house” when asked a location question. For benchmarks that contain subtitles (i.e., TVQA), we include the subtitles corresponding to the clip in the prompt during inference.
<span id="table-30"></span>

![Original paper Table 30](../../papers/llama-3/table-30.png)

**Table 30.** Video understanding performance of our vision module attached to Llama 3. We find that across range of tasks covering long-form and temporal video understanding, our vision adapters for Llama3 8B and 70B parameters are competitive and sometimes even outperform alternative models.

We present the performance of Llama 3 8B and 70B in [Table 30](#table-30). We compare Llama 3’s performance with that of two Gemini and two GPT-4 models. Note that all our results are zero-shot, as we do not include any part of these benchmarks in our training or finetuning data. We find that our Llama 3 models that train a small video adapter during post-training are very competitive, and in some cases even better, than other models that potentially leverage native multimodal processing all the way from pre-training. Llama 3 performs particularly well on video recognition given that we only evaluate the 8B and 70B parameter models. Llama 3 achieves its best performance on PerceptionTest, suggesting the model has a strong ability to perform complex temporal reasoning. On long-form activity understanding tasks like ActivityNet-QA, Llama 3 is able to obtain strong results even though it is processing only up to 64 frames, which means that for a 3-minute long video the model only processes one frame every 3 seconds.
<span id="section-8"></span>

## 8 Speech Experiments

We perform experiments to study a compositional approach of integrating speech capabilities into Llama 3, resembling the method we used for visual recognition. On the input side, an encoder, together with an adapter, is incorporated to process speech signals. We leverage a system prompt (in text) to enable different modes of operation for speech understanding in Llama 3. If no system prompt is provided, the model acts as a general-purpose spoken dialogue model which can effectively respond to the user speech in a manner that is consistent with the text-only version of Llama 3. The dialogue history is introduced as the prompt prefix to improve the multi-round dialogue experience. We also experiment with system prompts that enable the use of Llama 3 for automatic speech recognition (ASR) and automatic speech translation (AST). The speech interface of Llama 3 supports up to 34 languages.[+18] It also allows for the interleaved input of text and speech, enabling the model to solve advanced audio-comprehension tasks.

We also experiment with a speech generation approach in which we implement a streaming text-to-speech (TTS) system that generates speech waveforms on-the-fly during language model decoding. We design the speech generator for Llama 3 based on a proprietary TTS system and do not fine-tune the language model for speech generation. Instead, we focus on improving speech synthesis latency, accuracy, and naturalness by leveraging Llama 3 embeddings at inference time. The speech interface is illustrated in [Figure 1](#figure-01) and[Figure 29](#figure-29).
<span id="figure-29"></span>

![Architecture of our speech interface for Llama 3.](../../papers/llama-3/figure-29.png)

**Figure 29.** Architecture of our speech interface for Llama 3.
<span id="section-8-1"></span>

### 8.1 Data
<span id="section-8-1-1"></span>

#### 8.1.1 Speech Understanding

The training data can be categorized into two types. The pre-training data includes a large amount of unlabeled speech, which is used to initialize the speech encoder in a self-supervised manner. The supervised finetuning data includes speech recognition, speech translation, and spoken dialogue data; this data is used to unlock specific abilities when integrated with the large language model.

**Pre-training data.** To pre-train the speech encoder, we curate a dataset of approximately 15M hours of speech recordings encompassing a large number of languages. We filter our audio data using a voice activity detection (VAD) model and select audio samples with a VAD threshold above 0.7 for pre-training. In speech pre-training data, we also focus on ensuring the absence of PII. We use the Presidio Analyzer to identify such PII.

**Speech recognition and translation data.** Our ASR training data contains 230K hours of manually transcribed speech recordings that span 34 languages. Our AST training data contains 90K hours of translations in two directions: from 33 languages to English and from English to 33 languages. This data contains both supervised and synthetic data generated using the NLLB toolkit [Mar22]. The use of synthetic AST data enables us to increase model quality for low-resource languages. The speech segments in our data have a maximum length of 60 seconds.

**Spoken dialogue data.** To finetune the speech adapter for spoken dialogue, we synthetically generate responses for speech prompts by asking the language model to respond to transcriptions of those prompts [Fat24]. We generate synthetic data this way using a subset of the ASR dataset with 60K hours of speech. In addition, we generate 25K hours of synthetic data by running the Voicebox TTS system [Le24] on subsets of the data used to finetune Llama 3. We used several heuristics to select a subset of finetuning data that matches the distribution of speech. These heuristics include focusing on relatively short prompts with a simple structure and without non-text symbols.
<span id="section-8-1-2"></span>

#### 8.1.2 Speech Generation

The speech generation datasets mainly consist of those for training the text normalization (TN) model and the prosody model (PM). Both training data are augmented with an additional input feature of the Llama 3 embeddings to provide contextual information.

**Text normalization data.** Our TN training dataset includes 55K samples that cover a wide range of semiotic classes (*e.g.*, number, date, time) that require non-trivial normalization. Each sample is a pair of written-form text and the corresponding normalized spoken-form text, with an inferred sequence of handcrafted TN rules that carry out the normalization.

**Prosody model data.** The PM training data includes linguistic and prosodic features extracted from a 50K-hour TTS dataset, which are paired transcripts and audios recorded by professional voice actors in studio settings.

**Llama 3 embedding.** The Llama 3 embeddings are taken as the output of the 16th decoder layer. We work exclusively with the Llama 3 8B model and extract the embeddings for a given text (*i.e.* written-form input text for TN or the audio transcript for PM) as if they are generated by the Llama 3 model with an empty user prompt. In a given sample, each chunk in the Llama 3 token sequence is explicitly aligned with the corresponding chunks in native input sequence for TN or PM, *i.e.*, TN-specific text tokens (demarcated by unicode category) or phone-rate features respectively. This allows for training the TN and PM modules with streaming input of Llama 3 tokens and embeddings.
<span id="section-8-2"></span>

### 8.2 Model Architecture
<span id="section-8-2-1"></span>

#### 8.2.1 Speech Understanding

On the input side, the speech module consists of two successive modules: a speech encoder and an adapter. The output of the speech module is directly fed into the language model as token representation, enabling direct interaction between speech and text tokens. Furthermore, we incorporate two new special tokens to enclose the sequence of speech representations. The speech module differs substantially from the vision module (see [Section 7](#section-7)), which feeds multi-modal information into the language model via cross-attention layers. By contrast, the speech module generates embeddings that can be seamlessly integrated with text tokens, enabling the speech interface to leverage all the capabilities of the Llama 3 language model.

**Speech encoder.** Our speech encoder is a Conformer [Gul20] model with 1B parameters. The input to the model consists of 80-dimensional mel-spectrogram features, which are first processed by a stride-4 stacking layer followed by a linear projection to reduce the frame length to 40 ms. The resulting features are processed by an encoder with 24 Conformer layers. Each Conformer layer has a latent dimension of 1536, and consists of two Macron-net style feed-forward networks with dimension 4096, a convolution module with kernel size 7, and a rotary attention module [Su24] with 24 attention heads.

**Speech adapter.** The speech adapter contains about 100M parameters. It is composed of a convolution layer, a rotary Transformer layer, and a linear layer. The convolution layer has a kernel size of 3 and a stride of 2, which is designed to reduce the speech frame length to 80ms. This allows the model to provide more coarse-grained features to the language model. The Transformer layer has a latent dimension of 3072 and a feed-forward network with a dimension of 4096 which further processes the information from speech with context after the convolutional downsampling. Finally, the linear layer maps the output dimension to match that of the language-model embedding layer.
<span id="section-8-2-2"></span>

#### 8.2.2 Speech Generation

We use Llama 3 8B embeddings in two key components for speech generation: Text Normalization and Prosody Modeling. The TN module ensures semantic correctness by contextually transforming written text into spoken form. The PM module enhances naturalness and expressiveness by predicting prosodic features using these embeddings. Together, they enable accurate and natural speech generation.

**Text normalization.** As a determinant of the semantic correctness of generated speech, the text normalization (TN) module carries out context-aware transformation from written-form text into the respective spoken form which is eventually verbalized by the downstream components. For example, the written-form text *123* is read as a cardinal number (*one hundred twenty three*) or spelled digit-by-digit (*one two three*) depending on the semantic context. The TN system consists of a streaming LSTM-based sequence-tagging model that predicts the sequence of handcrafted TN rules used to transform the input text [Kan24a]. The neural model also takes in Llama 3 embeddings via cross attention to leverage the contextual information encoded therein, enabling minimal text token lookahead and streaming input/output.

**Prosody modeling.** [] To enhance the naturalness and expressiveness of synthesized speech, we integrate a decoder-only Transformer-based Prosody model (PM) [Rad21] that takes the Llama 3 embeddings as an additional input. This integration leverages the linguistic capabilities of Llama 3, utilizing both its textual output and intermediate embeddings at the token rate [Dev18, Don19, Raf19, Guo23c] to enhance the prediction of prosody features, thus reducing the lookahead required by the model.

The PM integrates several input components to generate comprehensive prosody predictions: linguistic features derived from the text normalization front-end detailed above, tokens, and embeddings. The PM predicts three key prosodic features: log duration of each phone, log F0 (fundamental frequency) average, and log power average across the phone duration. The model comprises a uni-directional Transformer and six attention heads. Each block includes cross-attention layers and dual fully connected layers with a hidden dimension of 864. A distinctive feature of the PM is its dual cross-attention mechanism, with one layer dedicated to linguistic inputs and the other to Llama embeddings. This setup efficiently manages varying input rates without requiring explicit alignment.
<span id="section-8-3"></span>

### 8.3 Training Recipe
<span id="section-8-3-1"></span>

#### 8.3.1 Speech Understanding

Training of the speech module is done in two stages. The first stage, speech pre-training, leverages unlabeled data to train a speech encoder that exhibits strong generalization capabilities across languages and acoustic conditions. In the second stage, supervised fine-tuning, the adapter and pre-trained encoder are integrated with the language model, and trained jointly with it while the LLM stays frozen. This enables the model to respond to speech input. This stage uses labeled data corresponding to speech understanding abilities.

Multilingual ASR and AST modeling often results in language confusion/interference, which leads to degraded performance. A popular way to mitigate this is to incorporate language identification (LID) information, both on the source and target side. This can lead to improved performance in the predetermined set of directions, but it does come with potential loss of generality. For instance, if a translation system expects LID on both source and target side, then the model will not likely to show good zero-shot performance in directions that were not seen in training. So our challenge is to design a system that allows LID information to some extent, but keeps the model general enough such that we can have the model do speech translation in unseen directions. To address this, we design system prompts which only contain LID for the text to be emitted (target side). There is no LID information for the speech input (source side) in these prompts, which also potentially allows it to work with code-switched speech. For ASR, we use the following system prompt: `Repeat after me in {language}:`, where `{language}` comes from one of the 34 languages (English, French, *etc.*) For speech translation, the system prompt is: `Translate the following sentence into {language}:`. This design has been shown to be effective in prompting the language model to respond in the desired language. We used the same system prompts during training and inference.

**Speech pre-training.** We use the self-supervised BEST-RQ algorithm [Chi22] to pre-train the speech encoder. We apply a mask of 32-frame length with a probability of 2.5% to the input mel-spectrogram. If the speech utterances are longer than 60 seconds, we perform a random crop of 6K frames, corresponding to 60 seconds of speech. We quantize mel-spectrogram features by stacking 4 consecutive frames, projecting the 320-dimensional vectors to a 16-dimensional space, and performing a nearest-neighbor search with respect to cosine similarity metric within a codebook of 8,192 vectors. To stabilize pre-training, we employ 16 different codebooks. The projection matrix and codebooks are randomly initialized and are not updated throughout the model training. The multi-softmax loss is used only on masked frames for efficiency reasons. The encoder is trained for 500K steps with a global batch size of 2,048 utterances.

**Supervised finetuning.** Both the pre-trained speech encoder and the randomly initialized adapter are further jointly optimized with in the supervised finetuning stage. The language model remains unchanged during this process. The training data is a mixture of ASR, AST, and spoken dialogue data. The speech model for Llama 3 8B is trained for 650K updates, using a global batch size of 512 utterances and an initial learning rate of $10^{-4}$. The speech model for Llama 3 70B is trained for 600K updates, using a global batch size of 768 utterances and an initial learning rate of $4\times10^{-5}$.
<span id="section-8-3-2"></span>

#### 8.3.2 Speech Generation

To support real-time processing, the prosody model employs a lookahead mechanism that considers a fixed number of future phones and a variable number of future tokens. This ensures consistent lookahead while processing incoming text, which is crucial for low-latency speech synthesis applications.

**Training.** We develop a dynamic alignment strategy utilizing causal masking to facilitate streamability in speech synthesis. This strategy incorporates a lookahead mechanism for a fixed number of future phones and a variable number of future tokens, aligning with the chunking process during text normalization ([Section 8.1.2](#section-8-1-2)). For each phone, the token lookahead includes the maximum number of tokens defined by the chunk size, resulting in variable lookahead for Llama embeddings but fixed lookahead for phonemes.

The Llama 3 embeddings are sourced from the Llama 3 8B model, which remains frozen during the training of the Prosody Model. The input phone-rate features include both linguistic and speaker/style controllability elements. The model training is conducted with a batch size of 1,024 utterances, each with a maximum length of 500 phones. We employ a learning rate of $9 \times 10^{-4}$ using the AdamW optimizer, training over 1 million updates with a learning rate warmup for the first 3,000 updates, following a cosine schedule.

**Inference.** During inference, the same lookahead mechanism and causal masking strategy are employed to ensure consistency between training and real-time processing. The PM handles incoming text in a streaming manner, updating the input phone by phone for phone-rate features and chunk by chunk for token-rate features. The new chunk input is updated only when the first phone for that chunk is current, maintaining the alignment and lookahead as during training.

For prosody target prediction, we employ a delayed pattern approach [Kha21], which enhances the model's ability to capture and reproduce long-range prosodic dependencies. This approach contributes to the naturalness and expressiveness of the synthesized speech, ensuring low-latency and high-quality output.
<span id="section-8-4"></span>

### 8.4 Speech Understanding Results

We evaluate the speech understanding capabilities of our speech interface for Llama 3 on three tasks: **(1)** automatic speech recognition, **(2)** speech translation, and **(3)** spoken question answering. We compare the performance of our speech interface for Llama 3 with three state-of-the-art models for speech understanding: Whisper [Rad23], SeamlessM4T [Bar23a], and Gemini.[+19] In all the evaluations, we used greedy search for Llama 3 token prediction.

**Speech recognition.** We evaluate the ASR performance on the English datasets of Multilingual LibriSpeech (MLS; [Pra20]), LibriSpeech [Pan15], VoxPopuli [Wan21b], and a subset of the multilingual FLEURS dataset [Con22]. In evaluation, the decoding results are post-processed using the Whisper text normalizer to ensure consistency in comparing with the reported results of other models. On all benchmarks, we measure the word error rate of our speech interface for Llama 3 on the standard test set of those benchmarks, except for Chinese, Japanese, Korean and Thai, where the character error rate is reported.
<span id="table-31"></span>

![Original paper Table 31](../../papers/llama-3/table-31.png)

**Table 31.** **Word error rate of our speech interface for Llama 3 on speech recognition tasks.** We report the performance of Whisper, SeamlessM4T, and Gemini for reference.

[Table 31](#table-31) shows the results of ASR evaluations. It demonstrates the strong performance of Llama 3 (and multi-modal foundation models more generally) on speech recognition tasks: our model outperforms models that are tailored to speech like Whisper[+20] and SeamlessM4T on all benchmarks. On MLS English, Llama 3 performs similarly to Gemini.
<span id="table-32"></span>

![Original paper Table 32](../../papers/llama-3/table-32.png)

**Table 32.** **BLEU score of our speech interface for Llama 3 on speech translation tasks.** We report the performance of Whisper and SeamlessM4T for reference.

**Speech translation.** We also evaluate our models on speech translation tasks in which the model is asked to translate non-English speech into English text. We use the FLEURS and Covost 2 [Wan21c] datasets in these evaluations, measuring BLEU scores of the translated English. [Table 32](#table-32) presents the results of these experiments.[+21] The performance of our models in speech translation highlights the advantages of multimodal foundation models for tasks such as speech translation.
<span id="figure-30"></span>

![Transcribed dialogue examples using the speech interface for Llama 3.](../../papers/llama-3/figure-30.png)

**Figure 30.** Transcribed dialogue examples using the speech interface for Llama 3. The examples illustrate zero-shot multi-turn and code-switching capabilities.

**Spoken question answering.** The speech interface of Llama 3 demonstrates remarkable question answering capabilities. The model can effortlessly comprehend code-switched speech without any prior exposure to such data. Notably, although the model was trained only on single-turn dialogue, it is capable of engaging in extended, coherent multi-turn dialogue sessions. [Figure 30](#figure-30) presents a few examples that highlight these multilingual and multi-turn capabilities.
<span id="table-33"></span>

![Original paper Table 33](../../papers/llama-3/table-33.png)

**Table 33.** **Speech toxicity of our speech interface to Llama 3 on the MuTox dataset.** AT refers to added toxicity (%) and LT refers to lost toxicity (%).

**Safety.** We evaluate the safety of our speech model on MuTox [Cos23], a multilingual audio-based dataset of 20,000 utterances for English and Spanish and 4,000 for 19 other languages, each with toxicity labels attached. The audio is passed as input to the model and the output is evaluated for toxicity, after cleaning some special characters. We apply the MuTox classifier [Cos23] and compare the results with Gemini 1.5 Pro. We evaluate the percentage of added toxicity (AT), when the input prompt is safe and the output is toxic, and the percentage of lost toxicity (LT), when the input prompt is toxic and the answer is safe. [Table 33](#table-33) shows the results for English and an average across all 21 languages that we evaluated on.[+22] The percentage of added toxicity is very low: our speech models have the lowest percentage of added toxicity for English, with less than 1%. It removes significantly more toxicity than it adds.
<span id="section-8-5"></span>

### 8.5 Speech Generation Results

For speech generation, we focus on evaluating the quality of token-wise input streaming models with the Llama 3 embeddings for the text normalization and prosody modeling tasks. The evaluation focuses on comparisons with models that do not take the Llama 3 embeddings as an additional input.

**Text normalization.** To measure the effect of Llama 3 embeddings, we experimented with changing the amount of right context the model uses. We trained the model using a right context of 3 TN tokens (demarcated by unicode category). This model is compared to models that do not use the Llama 3 embeddings, using a 3-token right context or a full bi-directional context. As expected, [Table 34](#table-34) shows using the full right context improves performance for the model without Llama 3 embeddings. However, the model that incorporates the Llama 3 embeddings outperforms all other models, hence enabling token-rate input/output streaming without relying on long context in the input.
<span id="table-34"></span>

![Original paper Table 34](../../papers/llama-3/table-34.png)

**Table 34.** **Sample-wise text normalization (TN) accuracy.** We compare models with or without Llama 3 8B embeddings, and using different right-context values.

**Prosody modeling.** To evaluate the performance of the our prosody model (PM) with Llama 3 8B, we conducted two sets of human evaluation comparing models with and without Llama 3 embeddings. Raters listened to samples from different models and indicated their preferences. To generate the final speech waveform, we use an in-house transformer based acoustic model [Wu21] that predicts spectral features and a WaveRNN neural vocoder [Kal18] to generate the final speech waveform.
<span id="table-35"></span>

![Original paper Table 35](../../papers/llama-3/table-35.png)

**Table 35.** **Prosody Modeling (PM) evaluation.** *Left:* Rater preferences of PM for Llama 3 8B vs. streaming phone-only baseline. *Right:* Rater preferences of PM for Llama 3 8B vs. non-streaming phone-only baseline.

First, we compare directly to a streaming baseline model without Llama 3 embeddings. In the second test, the Llama 3 8B PM is compared to a non-streaming baseline model without Llama 3 embeddings. As shown in [Table 35](#table-35), the Llama 3 8B PM is preferred 60% of the time compared to the streaming baseline, and 63.6% of the time compared to the non-streaming baseline, indicating a significant improvement in perceived quality. The key advantage of the Llama 3 8B PM is its token-wise streaming capability ([Section 8.2.2](#section-8-2-2)), which maintains low latency during inference. This reduces the model's lookahead requirements, enabling more responsive and real-time speech synthesis compared to non-streaming baselines. Overall, the Llama 3 8B prosody model consistently outperforms the baseline models, demonstrating its effectiveness in enhancing the naturalness and expressiveness of synthesized speech.
<span id="section-9"></span>

## 9 Related Work

The development of Llama 3 builds on a large body of prior work studying foundation models for language, images, videos, and speech. A comprehensive overview of that work is outside the scope of this paper; we refer the reader to [Bor24] [Mad24, Zha23d] for such overviews. Below, we briefly outline seminal works that directly influenced the development of Llama 3.
<span id="section-9-1"></span>

### 9.1 Language

**Scale.** Llama 3 follows the enduring trend of applying straightforward methods at ever increasing scales in foundation models. Improvements are driven by increased compute and improved data, with the 405B model using almost fifty times the pre-training compute budget of Llama 2 70B. Despite containing 405B parameters, our largest Llama 3 in fact contains fewer parameters than earlier and much less performant models such as PALM [Cho22a], due to better understanding of scaling laws [Kap20, Hof22]. Little is publicly known about the size of other frontier models, such as Claude 3 or GPT 4 [Ope23b], but overall performance is compareable.

**Small models.** Developments in smaller models have paralleled those in large models. Models with fewer parameters can dramatically improve inference cost and simplify deployment [Meh24, Gem24b]. The smaller Llama 3 models achieve this by training far beyond the point of compute optimal training, effectively trading training compute for inference efficiency. An alternative path is to distill larger models into smaller ones, as in Phi [Abd24].

**Architectures.** While Llama 3 makes minimal architectural modifiations to compared to Llama 2, other recent foundation models have explored other designs. Most notably, mixture of experts architectures [Sha17, Lew21, Fed22, Zho22a] can be used as an efficient way to increase the capacity of a models, such as in Mixtral [Jia24a] and Arctic [Sno24]. Llama 3 outperforms these models, suggesting that dense architectures are not the limiting factor, but there remain numerous trade offs in terms of training and inference efficiency, and model stability at scale.

**Open source.** Open weights foundation models have rapidly improved over the last year, with Llama3-405B now competitive with the current closed weight state-of-the-art. Numerous model families have recently been developed, including Mistral [Jia23a], Falcon [Alm23a], MPT [Dat24], Pythia [Bid23], Arctic [Sno24], OpenELM [Meh24], OLMo [Gro24], StableLM [Bel24], OpenLLaMA [Gen23b], Qwen [Bai23b], Gemma [Gem24b], Grok [Xai24], and Phi [Abd24].

**Post-training.** Post-training follows the established strategy of instruction tuning [Chu22, Ouy22] followed by alignment with human feedback [Kau23]. While some studies have shown the surprising effectiveness of lightweight alignment procedures [Zho24a], uses millions of human instructions and preference judgments to improve the pre-trained model, including techniques such as rejection sampling [Bai22], supervised finetuning [San22], and Direct Preference Optimization [Raf24]. In order to curate these instruction and preference examples, we deploy earlier versions of to filter [Liu24j], re-write [Pan24a], or generate prompts and responses [Liu24g] and apply these techniques through multiple rounds of post-training.
<span id="section-9-2"></span>

### 9.2 Multimodality

Our experiments with multimodal capabilities for Llama 3 are part of a long line of work on foundation models that jointly model multiple modalities.

**Images.** A substantial body of work has trained image-recognition models on large amounts of image-text pairs, for example, [Mah18] [Xia24c, Cha24, Ope23a]. [Rad21] presented one of the first models to jointly embed images and text via contrastive learning. More recently, a series of models has studied approaches similar to the one used in Llama 3, for example, [Ala22] [Dai23, Liu23h, Liu23f, Yan23a, Ye23, Zhu23]. Our approach in Llama 3 combines ideas from many of these papers to achieve results that are comparable with Gemini 1.0 Ultra [Gem23] and GPT-4 Vision [Ope23a]; see [Section 7.6](#section-7-6).

**Video.** Although video inputs are supported by an increasing number of foundation models [Gem23, Ope23a], the body of work on joint modeling of videos and language is not that large. Akin to Llama 3, most current studies adopt an adapter approach to align video and language representations and unlock question-answering and reasoning about videos [Lin23a, Li23k, Maa24, Zha23c, Zha22c]. We find that such approaches produce results that are competitive with the state-of-the-art; see [Section 7.7](#section-7-7).

**Speech.** Our work also fits in a larger body of work combining language and speech modeling. Earlier joint models of text and speech include AudioPaLM [Rub23], VioLA [Wan23c], VoxtLM [Mai23], SUTLM [Cho23], and Spirit-LM [Ngu24]. Our work builds on prior compositional approaches to combining speech and language like [Fat24]. Unlike most prior work, we opt to not finetune the language model itself for speech tasks as doing so may lead to contention on non-speech tasks. We find that at larger model scales, strong performances are attainable even without such finetuning; see [Section 8.4](#section-8-4).
<span id="section-10"></span>

## 10 Conclusion

In many ways, the development of high-quality foundation models is still in its infancy. Our experience in developing Llama 3 suggests that substantial further improvements of these models are on the horizon. Throughout the development of the Llama 3 model family, we found that a strong focus on high-quality data, scale, and simplicity consistently yielded the best results. In preliminary experiments, we explored more complex model architectures and training recipes but did not find the benefits of such approaches to outweigh the additional complexity they introduce in model development.

Developing a flagship foundation model such as Llama 3 involves overcoming a plethora of deep technical problems but also requires clever organizational decisions. For example, to ensure Llama 3 is not accidentally overfitted on commonly used benchmarks, our pre-training data was procured and processed by a separate team that was strongly incentivized to prevent contamination of that pre-training data with external benchmarks. As another example, we ensure that our human evaluations remain trustworthy by allowing only a small set of researchers who do not contribute to model development to perform and access these evaluations. While such organizational decisions are rarely discussed in technical papers, we found them to be pivotal to the successful development of the Llama 3 family of models.

We shared the details of our development process because we believe this will: **(1)** help the larger research community understand the key factors of foundation model development and **(2)** contribute to a more informed debate about the future of foundation models in the general public. We also shared preliminary experiments with integrating multimodal capabilities into Llama 3. While these models are still under active development and not yet ready for release, we hope sharing our results early will accelerate research in this direction.

Following the positive outcomes of the detailed safety analyses presented in this paper, we publicly release our Llama 3 language models in order to accelerate the development of AI systems for a plethora of societally relevant use cases and enable the research community to scrutinize our models and identify ways to make these models better and safer. We believe that the public release of foundation models plays a key role in the responsible development of such models, and we hope that the release of Llama 3 encourages the industry to embrace the open, responsible development of AGI.

## Contributors and Acknowledgements

Llama 3 is the result of the work of a large number of people at Meta. Below, we list all **core contributors** (people who worked on Llama 3 for at least $\nicefrac{2}{3}$rd of the runtime of the project) and **contributors** (people who worked on Llama 3 for at least $\nicefrac{1}{5}$th of the runtime of the project). We list all contributors in alphabetical order of first name.

### Core Contributors

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurelien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Roziere, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, Danny Wyatt, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Francisco Guzmán, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Govind Thattai, Graeme Nail, Gregoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel Kloumann, Ishan Misra, Ivan Evtimov, Jack Zhang, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Karthik Prasad, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, Khalid El-Arini, Krithika Iyer, Kshitiz Malik, Kuenley Chiu, Kunal Bhalla, Kushal Lakhotia, Lauren Rantala-Yeary, Laurens van der Maaten, Lawrence Chen, Liang Tan, Liz Jenkins, Louis Martin, Lovish Madaan, Lubo Malo, Lukas Blecher, Lukas Landzaat, Luke de Oliveira, Madeline Muzzi, Mahesh Pasupuleti, Mannat Singh, Manohar Paluri, Marcin Kardas, Maria Tsimpoukelli, Mathew Oldham, Mathieu Rita, Maya Pavlova, Melanie Kambadur, Mike Lewis, Min Si, Mitesh Kumar Singh, Mona Hassan, Naman Goyal, Narjes Torabi, Nikolay Bashlykov, Nikolay Bogoychev, Niladri Chatterji, Ning Zhang, Olivier Duchenne, Onur Çelebi, Patrick Alrassy, Pengchuan Zhang, Pengwei Li, Petar Vasic, Peter Weng, Prajjwal Bhargava, Pratik Dubal, Praveen Krishnan, Punit Singh Koura, Puxin Xu, Qing He, Qingxiao Dong, Ragavan Srinivasan, Raj Ganapathy, Ramon Calderer, Ricardo Silveira Cabral, Robert Stojnic, Roberta Raileanu, Rohan Maheswari, Rohit Girdhar, Rohit Patel, Romain Sauvestre, Ronnie Polidoro, Roshan Sumbaly, Ross Taylor, Ruan Silva, Rui Hou, Rui Wang, Saghar Hosseini, Sahana Chennabasappa, Sanjay Singh, Sean Bell, Seohyun Sonia Kim, Sergey Edunov, Shaoliang Nie, Sharan Narang, Sharath Raparthy, Sheng Shen, Shengye Wan, Shruti Bhosale, Shun Zhang, Simon Vandenhende, Soumya Batra, Spencer Whitman, Sten Sootla, Stephane Collot, Suchin Gururangan, Sydney Borodinsky, Tamar Herman, Tara Fowler, Tarek Sheasha, Thomas Georgiou, Thomas Scialom, Tobias Speckbacher, Todor Mihaylov, Tong Xiao, Ujjwal Karn, Vedanuj Goswami, Vibhor Gupta, Vignesh Ramanathan, Viktor Kerkez, Vincent Gonguet, Virginie Do, Vish Vogeti, Vítor Albiero, Vladan Petrovic, Weiwei Chu, Wenhan Xiong, Wenyin Fu, Whitney Meers, Xavier Martinet, Xiaodong Wang, Xiaofang Wang, Xiaoqing Ellen Tan, Xide Xia, Xinfeng Xie, Xuchao Jia, Xuewei Wang, Yaelle Goldschlag, Yashesh Gaur, Yasmine Babaei, Yi Wen, Yiwen Song, Yuchen Zhang, Yue Li, Yuning Mao, Zacharie Delpierre Coudert, Zheng Yan, Zhengxing Chen, and Zoe Papakipos.

### Contributors

Aaditya Singh, Aayushi Srivastava, Abha Jain, Adam Kelsey, Adam Shajnfeld, Adithya Gangidi, Adolfo Victoria, Ahuva Goldstand, Ajay Menon, Ajay Sharma, Alex Boesenberg, Alexei Baevski, Allie Feinstein, Amanda Kallet, Amit Sangani, Amos Teo, Anam Yunus, Andrei Lupu, Andres Alvarado, Andrew Caples, Andrew Gu, Andrew Ho, Andrew Poulton, Andrew Ryan, Ankit Ramchandani, Annie Dong, Annie Franco, Anuj Goyal, Aparajita Saraf, Arkabandhu Chowdhury, Ashley Gabriel, Ashwin Bharambe, Assaf Eisenman, Azadeh Yazdan, Beau James, Ben Maurer, Benjamin Leonhardi, Bernie Huang, Beth Loyd, Beto De Paola, Bhargavi Paranjape, Bing Liu, Bo Wu, Boyu Ni, Braden Hancock, Bram Wasti, Brandon Spence, Brani Stojkovic, Brian Gamido, Britt Montalvo, Carl Parker, Carly Burton, Catalina Mejia, Ce Liu, Changhan Wang, Changkyu Kim, Chao Zhou, Chester Hu, Ching-Hsiang Chu, Chris Cai, Chris Tindal, Christoph Feichtenhofer, Cynthia Gao, Damon Civin, Dana Beaty, Daniel Kreymer, Daniel Li, David Adkins, David Xu, Davide Testuggine, Delia David, Devi Parikh, Diana Liskovich, Didem Foss, Dingkang Wang, Duc Le, Dustin Holland, Edward Dowling, Eissa Jamil, Elaine Montgomery, Eleonora Presani, Emily Hahn, Emily Wood, Eric-Tuan Le, Erik Brinkman, Esteban Arcaute, Evan Dunbar, Evan Smothers, Fei Sun, Felix Kreuk, Feng Tian, Filippos Kokkinos, Firat Ozgenel, Francesco Caggioni, Frank Kanayet, Frank Seide, Gabriela Medina Florez, Gabriella Schwarz, Gada Badeer, Georgia Swee, Gil Halpern, Grant Herman, Grigory Sizov, Guangyi (Jack) Zhang, Guna Lakshminarayanan, Hakan Inan, Hamid Shojanazeri, Han Zou, Hannah Wang, Hanwen Zha, Haroun Habeeb, Harrison Rudolph, Helen Suk, Henry Aspegren, Hunter Goldman, Hongyuan Zhan, Ibrahim Damlaj, Igor Molybog, Igor Tufanov, Ilias Leontiadis, Irina-Elena Veliche, Itai Gat, Jake Weissman, James Geboski, James Kohli, Janice Lam, Japhet Asher, Jean-Baptiste Gaya, Jeff Marcus, Jeff Tang, Jennifer Chan, Jenny Zhen, Jeremy Reizenstein, Jeremy Teboul, Jessica Zhong, Jian Jin, Jingyi Yang, Joe Cummings, Jon Carvill, Jon Shepard, Jonathan McPhie, Jonathan Torres, Josh Ginsburg, Junjie Wang, Kai Wu, Kam Hou U, Karan Saxena, Kartikay Khandelwal, Katayoun Zand, Kathy Matosich, Kaushik Veeraraghavan, Kelly Michelena, Keqian Li, Kiran Jagadeesh, Kun Huang, Kunal Chawla, Kyle Huang, Lailin Chen, Lakshya Garg, Lavender A, Leandro Silva, Lee Bell, Lei Zhang, Liangpeng Guo, Licheng Yu, Liron Moshkovich, Luca Wehrstedt, Madian Khabsa, Manav Avalani, Manish Bhatt, Martynas Mankus, Matan Hasson, Matthew Lennie, Matthias Reso, Maxim Groshev, Maxim Naumov, Maya Lathi, Meghan Keneally, Miao Liu, Michael L. Seltzer, Michal Valko, Michelle Restrepo, Mihir Patel, Mik Vyatskov, Mikayel Samvelyan, Mike Clark, Mike Macey, Mike Wang, Miquel Jubert Hermoso, Mo Metanat, Mohammad Rastegari, Munish Bansal, Nandhini Santhanam, Natascha Parks, Natasha White, Navyata Bawa, Nayan Singhal, Nick Egebo, Nicolas Usunier, Nikhil Mehta, Nikolay Pavlovich Laptev, Ning Dong, Norman Cheng, Oleg Chernoguz, Olivia Hart, Omkar Salpekar, Ozlem Kalinli, Parkin Kent, Parth Parekh, Paul Saab, Pavan Balaji, Pedro Rittner, Philip Bontrager, Pierre Roux, Piotr Dollar, Polina Zvyagina, Prashant Ratanchandani, Pritish Yuvraj, Qian Liang, Rachad Alao, Rachel Rodriguez, Rafi Ayub, Raghotham Murthy, Raghu Nayani, Rahul Mitra, Rangaprabhu Parthasarathy, Raymond Li, Rebekkah Hogan, Robin Battey, Rocky Wang, Russ Howes, Ruty Rinott, Sachin Mehta, Sachin Siby, Sai Jayesh Bondu, Samyak Datta, Sara Chugh, Sara Hunt, Sargun Dhillon, Sasha Sidorov, Satadru Pan, Saurabh Mahajan, Saurabh Verma, Seiji Yamamoto, Sharadh Ramaswamy, Shaun Lindsay, Shaun Lindsay, Sheng Feng, Shenghao Lin, Shengxin Cindy Zha, Shishir Patil, Shiva Shankar, Shuqiang Zhang, Shuqiang Zhang, Sinong Wang, Sneha Agarwal, Soji Sajuyigbe, Soumith Chintala, Stephanie Max, Stephen Chen, Steve Kehoe, Steve Satterfield, Sudarshan Govindaprasad, Sumit Gupta, Summer Deng, Sungmin Cho, Sunny Virk, Suraj Subramanian, Sy Choudhury, Sydney Goldman, Tal Remez, Tamar Glaser, Tamara Best, Thilo Koehler, Thomas Robinson, Tianhe Li, Tianjun Zhang, Tim Matthews, Timothy Chou, Tzook Shaked, Varun Vontimitta, Victoria Ajayi, Victoria Montanez, Vijai Mohan, Vinay Satish Kumar, Vishal Mangla, Vlad Ionescu, Vlad Poenaru, Vlad Tiberiu Mihailescu, Vladimir Ivanov, Wei Li, Wenchen Wang, Wenwen Jiang, Wes Bouaziz, Will Constable, Xiaocheng Tang, Xiaojian Wu, Xiaolan Wang, Xilun Wu, Xinbo Gao, Yaniv Kleinman, Yanjun Chen, Ye Hu, Ye Jia, Ye Qi, Yenda Li, Yilin Zhang, Ying Zhang, Yossi Adi, Youngjin Nam, Yu (Sid) Wang, Yu Zhao, Yuchen Hao, Yundi Qian, Yunlu Li, Yuzi He, Zach Rait, Zachary DeVito, Zef Rosnbrick, Zhaoduo Wen, Zhenyu Yang, Zhiwei Zhao, and Zhiyu Ma.

### Acknowledgements

We thank Mark Zuckerberg, Chris Cox, Ahmad Al-Dahle, Santosh Janardhan, Joelle Pineau, Yann LeCun, Aparna Ramani, Yee Jiun Song, and Ash Jhaveri for their invaluable support for Llama 3.

We also thank Aasish Pappu, Adebissy Tharinger, Adnan Aziz, Aisha Iqbal, Ajit Mathews, Albert Lin, Amar Budhiraja, Amit Nagpal, Andrew Or, Andrew Prasetyo Jo, Ankit Jain, Antonio Prado, Aran Mun, Armand Kok, Ashmitha Jeevaraj Shetty, Aya Ibrahim, Bardiya Sadeghi, Beibei Zhu, Bell Praditchai, Benjamin Muller, Botao Chen, Carmen Wang, Carolina Tsai, Cen Peng, Cen Zhao, Chana Greene, Changsheng Zhao, Chenguang Zhu, Chloé Bakalar, Christian Fuegen, Christophe Ropers, Christopher Luc, Dalton Flanagan, Damien Sereni, Dan Johnson, Daniel Haziza, Daniel Kim, David Kessel, Digant Desai, Divya Shah, Dong Li, Elisabeth Michaels, Elissa Jones, Emad El-Haraty, Emilien Garreau, Eric Alamillo, Eric Hambro, Erika Lal, Eugen Hotaj, Fabian Gloeckle, Fadli Basyari, Faith Eischen, Fei Kou, Ferdi Adeputra, Feryandi Nurdiantoro, Flaurencya Ciputra, Forest Zheng, Francisco Massa, Furn Techaletumpai, Gobinda Saha, Gokul Nadathur, Greg Steinbrecher, Gregory Chanan, Guille Cobo, Guillem Brasó, Hany Morsy, Haonan Sun, Hardik Shah, Henry Erksine Crum, Hongbo Zhang, Hongjiang Lv, Hongye Yang, Hweimi Tsou, Hyunbin Park, Ian Graves, Jack Wu, Jalpa Patel, James Beldock, James Zeng, Jeff Camp, Jesse He, Jilong Wu, Jim Jetsada Machom, Jinho Hwang, Jonas Gehring, Jonas Kohler, Jose Leitao, Josh Fromm, Juan Pino, Julia Rezende, Julian Garces, Kae Hansanti, Kanika Narang, Kartik Khandelwal, Keito Uchiyama, Kevin McAlister, Kimish Patel, Kody Bartelt, Kristina Pereyra, Kunhao Zheng, Lien Thai, Lu Yuan, Lunwen He, Marco Campana, Mariana Velasquez, Marta R. Costa-jussa, Martin Yuan, Max Ren, Mayank Khamesra, Mengjiao MJ Wang, Mengqi Mu, Mergen Nachin, Michael Suo, Mikel Jimenez Fernandez, Mustafa Ozdal, Na Li, Nahiyan Malik, Naoya Miyanohara, Narges Torabi, Nathan Davis, Nico Lopero, Nikhil Naik, Ning Li, Octary Azis, PK Khambanonda, Padchara Bubphasan, Pian Pawakapan, Prabhav Agrawal, Praveen Gollakota, Purin Waranimman, Qian Sun, Quentin Carbonneaux, Rajasi Saha, Rhea Nayak, Ricardo Lopez-Barquilla, Richard Huang, Richard Qiu, Richard Tosi, Rishi Godugu, Rochit Sapra, Rolando Rodriguez Antunez, Ruihan Shan, Sakshi Boolchandani, Sam Corbett-Davies, Samuel Djunaedi, Sarunya Pumma, Saskia Adams, Scott Wolchok, Shankar Kalyanaraman, Shashi Gandham, Shengjie Bi, Shengxing Cindy, Shervin Shahidi, Sho Yaida, Shoubhik Debnath, Sirirut Sonjai, Srikanth Sundaresan, Stephanie Worland, Susana Contrera, Tejas Shah, Terry Lam, Tony Cao, Tony Lee, Tristan Rice, Vishy Poosala, Wenyu Chen, Wesley Lee, William Held, Xiaozhu Meng, Xinhua Wang, Xintian Wu, Yanghan Wang, Yaroslava Kuzmina, Yifan Wang, Yuanhao Xiong, Yue Zhao, Yun Wang, Zaibo Wang, Zechun Liu, and Zixi Qi for helpful contributions to Llama 3.

[+18]: The speech interface supports the following 34 languages: Arabic, Bengali, Chinese, Czech, Dutch, English, Finnish, French, German, Greek, Gujarati, Hindi, Hungarian, Indonesian, Italian, Japanese, Kannada, Korean, Malayalam, Marathi, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swahili, Swedish, Tamil, Telugu, Thai, Turkish, Urdu, Vietnamese.

[+19]: Due to technical limitations, we compare with the performance of Gemini on MLS reported in the original paper.

[+20]: On FLEURS ASR, Malayalam is not officially reported for Whisper v3, so we use the average of 33 languages.

[+21]: On Covost 2, we evaluate only on 15 (out of 21) languages.

[+22]: Note that for Gemini, we encountered that a significant number of responses were empty, which could be due to safety filters on their side (though some empty responses were for non-toxic input) or to rate limits. To conduct the analysis, we assumed that all the empty responses are safe. This is the most conservative approach for results and the upper bound of what Gemini results would look like.

[+1]: The Llama 3 8B and 70B were pre-trained on multilingual data but were intended for use in English at the time.

[+2]: In this paper, we use the term “post-training” to refer to any model training that happens outside of pre-training.

[+3]: [https://github.com/openai/tiktoken/tree/main](https://github.com/openai/tiktoken/tree/main)

[+4]: Open Compute Project: [https://www.opencompute.org/](https://www.opencompute.org/)

[+5]: Note that we use only up to 16K of these 24K GPUs for Llama 3 pre-training.

[+6]: We use the term “post-training” to refer to any model training that happens outside of pre-training.

[+7]: [https://brave.com/search/api/](https://brave.com/search/api/)

[+8]: [https://products.wolframalpha.com/llm-api/documentation](https://products.wolframalpha.com/llm-api/documentation)

[+9]: Llama 3 has not been optimized or safety tuned for use cases in those other languages. Developers may fine-tune Llama 3 models for languages beyond the 8 supported languages provided they comply with the Llama 3 Community License and the Acceptable Use Policy and in such cases are responsible for ensuring that any uses of Llama 3 in additional languages is done in a safe and responsible manner.

[+10]: [https://platform.openai.com/docs/assistants/overview](https://platform.openai.com/docs/assistants/overview)

[+11]: For multiturn human evaluations, the number of turns is between 2 and 11 in each prompt. We assess the model response in the final turn.

[+12]: Note there are limitations with our analysis — for example, recent work advocates for metrics beyond exact match [Ipp23] and alternative prompt search strategies [Kas24]. Nonetheless, we find the results of the evaluations to be encouraging.

[+13]: Because these safety benchmarks are internal to Meta, we acknowledge that the numbers in this section are not reproducible externally, and so we choose to anonymize the competitors we evaluate against.

[+14]: We note that these results do not imply the violation rate and false refusal rate is $0\%$ on this category. The result implies that those rates are so low that this particular evaluation is unable to measure them.

[+15]: Our FP8 kernels are available at [https://github.com/pytorch/FBGEMM/tree/main/fbgemm\_gpu/experimental/gen\_ai](https://github.com/pytorch/FBGEMM/tree/main/fbgemm_gpu/experimental/gen_ai). We provide usage examples at [https://github.com/meta-llama/llama-agentic-system](https://github.com/meta-llama/llama-agentic-system).

[+16]: See [https://eval.ai/web/challenges/challenge-page/2091/overview](https://eval.ai/web/challenges/challenge-page/2091/overview).

[+17]: See [https://github.com/doc-doc/NExT-OE](https://github.com/doc-doc/NExT-OE).
