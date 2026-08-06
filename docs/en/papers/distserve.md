---
title: 'DistServe'
createTime: 2026/08/04 23:48:22
permalink: /en/papers/distserve/
---

> Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. First submitted to arXiv on January 18, 2024; current version v3. [DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving](https://arxiv.org/abs/2401.09670). [Original PDF](/paper/distserve.pdf). [TeX source](https://export.arxiv.org/e-print/2401.09670). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.

DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application’s TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy *tailored* for each phase. DistServe also places the two phases according to the serving cluster’s bandwidth to minimize the communication caused by disaggregation. As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve 4.48$\times$ more requests or 10.2$\times$ tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for $>90\%$ of requests.

## 1 Introduction

Large language models (LLMs), such as GPT-4 [Gpta23], Bard [Bard23], and LLaMA [Openb23], represent a groundbreaking shift in generative AI. They start to reshape existing Internet services, ranging from search engines to personal assistants [Inflec23], and enable fundamentally new applications, like universal chatbots [Introd22, Vicuna23] and programming assistants [Xivbg21, Xivab23]. Yet, these advances come with a significant challenge: processing an end-to-end LLM query can be substantially slower than a standard search query [Reutea23]. In order to meet the stringent latency requirements of various applications, service providers need to over-provision compute resources, particularly many GPUs, leading to a shortfall in cost efficiency. Therefore, optimizing the cost per LLM query while adhering to high SLO attainment (the proportion of requests that meet the SLOs) is becoming increasingly essential for all LLM services.

<span id="figure-01"></span>

![Refer to caption](../../papers/distserve/figure-01.png)

**Figure 1.** Performance when serving an LLM with 13B parameters under a synthetic workload with input length = 512 and output length = 64 on one NVIDIA 80GB A100. Upper: The P90 time-to-first-token (TTFT) latency comparing existing systems vs. a system serving only the prefill phase. Down: The P90 time-per-output-token (TPOT) latency comparing existing systems vs. a system serving only the decoding phase.

An LLM service responds to a user query in two phases. The *prefill phase* processes a user’s prompt, composed of a sequence of tokens, to generate the first token of the response *in one step*. Following it, the *decoding phase* sequentially generates subsequent tokens *in multiple steps*; each decoding step generates a new token based on tokens generated in previous steps, until reaching a termination token. This dual-phase process distinguishes LLM services from traditional services – an LLM service’s latency is uniquely measured by two key metrics: the *time to first token* (TTFT), which is the duration of the prefill phase, and the *time per output token* (TPOT), which represents the average time taken to generate a token for each request (except for the first token) [+1]. Different applications place varying demands on each metric. For example, real-time chatbots [Introd22] prioritize low TTFT for response promptness, while TPOT only remains important until it is faster than human reading speed (i.e., 250 words/min). Conversely, document summarization emphasizes low TPOT for faster generation of the summary.

Hence, given the application’s TTFT and TPOT requirements, an effective LLM serving system should balance these needs and maximize *per-GPU goodput*, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned – higher per-GPU goodput directly translates into lower cost per query.

As the prefill and decoding phases share the LLM weights and working memory, existing LLM serving systems typically colocate both phases on GPUs and maximize the overall system throughput – tokens generated per second across all users and requests – by batching the prefill and decoding steps across requests [OSDIa22, Princa23]. However, to meet latency requirements, we find these systems must over-provision compute resources. To see this, [Figure 1](#figure-01) illustrates how the P90 TTFT and TPOT shift with increasing request rates when serving a 13B LLM using existing systems [Efficc23], with workload pattern and two latency constraints set to emulate using LLM to generate a short summary for an article. Under the SLO attainment of 90%, the maximum achievable goodput on a single A100 GPU, which is constrained by the more stringent one of TTFT and TPOT requirements, is about 1.6 requests per second (rps). The performance contrasts sharply when each phase is served independently on a separate GPU, shown by the orange and green curves, which achieve per-GPU goodput of 5.6 rps for the prefill phase and 10 rps for decoding. Ideally, by allocating 2 GPUs for prefill and 1 GPU for decoding, we can effectively serve the model with an overall goodput of 10 rps, or equally 3.3 rps per GPU, which is 2.1x higher than existing systems. The gap in goodput primarily stems from the colocation of the prefill and decoding – two phases with very distinct computational characteristics and latency requirements (§[2.1](#S2.SS1 "2.1 LLM Inference ‣ 2 Background and Motivation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")).

First, colocation leads to strong *prefill-decoding interference*. A prefill step often takes much longer than a decoding step. When batched together, decoding steps in the batch are delayed by the prefill steps, significantly elongating their TPOT; similarly, the inclusion of decoding steps contributes to a non-trivial increase in TTFT, as evidenced in [Figure 2](#figure-02). Even if we schedule them separately, issues persist as they begin to compete for resources. Decoding tasks awaiting GPU execution are subject to increased queuing delays due to ongoing prefill tasks, and vice versa. Prioritized scheduling of one phase risks failing the latency requirements of the other.

Second, the prefill and decoding computation differ in latency requirements and preference for different forms of parallelism (§[3](#S3 "3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). Colocating prefill and decoding, however, couples their resource allocation, and prevents implementing different parallelism strategies more suited to meeting the specific latency requirements of each phase.

To overcome these challenges, we propose to disaggregate the prefill and decoding phases of LLM inference, assigning them to separate GPUs. Our approach has two benefits. First, operating each phase independently on different GPUs eliminates prefill-decoding interference. Second, it allows to scale each phase independently with tailored resource allocation and model parallelism strategies to meet their specific latency requirements. Although disaggregation causes communication of intermediate states between GPUs, we show that the communication overhead is insubstantial (§[3.3](#S3.SS3 "3.3 Practical Problems ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) in modern GPU clusters, and when managed appropriately, disaggregation significantly improves per-GPU goodput.

Based on the above insights, in this work, we build DistServe, a goodput-optimized LLM serving system by disaggregating the prefill and decoding phases. Given TTFT and TPOT requirements, DistServe first scales each phase independently by co-optimizing the GPU allocation and parallelism strategies of the prefill and decoding phase assuming serving a single model replica. The optimization ensures maximizing the per-GPU goodput and may assign different numbers of GPUs and parallelism strategies to each phase depending on their respective latency requirements. DistServe then scales this allocation to multiple instances via replication until meeting the user-required traffic rate (§[4](#S4 "4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). DistServe also features an algorithm to place the prefill and decoding computation according to their allocation schemes and the cluster’s bandwidth to minimize the overhead of communicating intermediate states between phases.

We implement DistServe as an orchestration layer on top of the LLM inference engine. We evaluate DistServe on various LLMs, varying the workloads based on three important real-world LLM applications: chatbots, programming assistant, and document summary. Compared to state-of-the-art solutions, DistServe can serve up to $4.48\times$ more requests under latency constraints. Our contributions are:

-   •

    Identify the problems of prefill-decoding interference and resource coupling in existing LLM serving systems and propose to disaggregate the prefill and decoding phases.

-   •

    Design a novel placement algorithm to automatically choose the goodput-optimal schema for prefill and decoding instances.

-   •

    Conduct a comprehensive evaluation of DistServe with realistic workloads.

## 2 Background and Motivation

An LLM service follows a client-server architecture: the client submits a sequence of text as a request to the server; the server hosts the LLM on GPUs, runs inference over the request, and responds (or streams) the generation back to the client. As explained in §[1](#S1 "1 Introduction ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), due to the unique prefill-decoding process, LLM service may impose aggressive service-level objectives (SLOs) on both TTFT and TPOT, varying with the application’s needs. The serving system must meet both SLOs while minimizing the cost associated with expensive GPUs. In other words, we want the serving system to maximize the requests served per second adhering to the SLO attainment goal for each GPU provisioned – *maximizing per-GPU goodput*. Next, we detail the LLM inference computation (§[2.1](#S2.SS1 "2.1 LLM Inference ‣ 2 Background and Motivation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) and discuss existing optimizations for LLM serving (§[2.2](#S2.SS2 "2.2 LLM Serving Optimization ‣ 2 Background and Motivation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")).

### 2.1 LLM Inference

Modern LLMs [Gpta23, Openb23] predict the next token given an input sequence. This prediction involves computing a hidden representation for each token within the sequence. An LLM can take a variable number of input tokens and compute their hidden representations in parallel, and its computation workload increases superlinearly with the number of tokens processed in parallel. Regardless of the input token count, the computation demands substantial I/O to move LLM weights and intermediate states from the GPU’s HBM to SRAM. This process is consistent across varying input sizes.

The prefill step deals with a new sequence, often comprising many tokens, and processes these tokens concurrently. Unlike prefill, each decoding step only processes one new token generated by the previous step. This leads to significant computational differences between the two phases. When dealing with user prompts that are not brief, the prefill step tends to be computation-bound. For instance, for a 13B LLM, computing the prefill of a 512-token sequence makes an A100 compute-bound. The larger the model, the shorter sequence is needed to turn the prefill step compute-bound (see §[3.1](#S3.SS1 "3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). In contrast, the decoding phase, despite processing only one new token per step, incurs a similar level of I/O to the prefill phase, making it constrained by the GPU’s memory bandwidth.

<span id="figure-02"></span>

![Refer to caption](../../papers/distserve/figure-02.png)

**Figure 2.** Execution time for one batch when serving an LLM with 13B parameters as batch size increases. Compared between a decoding-only batch and a batch adding just one prefill request.

During both phases, intermediate states, known as KV caches [Efficc23], are generated at each token position, which are needed again in later decoding steps. To avoid recomputing them, they are saved in GPU memory. Because of the shared use of LLM weights and KV caches in memory, most LLM inference engines opt to colocate the prefill and decoding phases on GPUs, despite their distinct computational characteristics.

### 2.2 LLM Serving Optimization

In real-time online serving, multiple requests come and must be served within SLOs. Batching and parallelizing their computation is key for achieving low latency, high throughput, and high utilization of GPUs.

Batching. Current serving systems [OSDIa22, Efficc23, Xivz23] utilize a batching technique known as *continuous batching*. This method batches the prefill of new requests with the decoding of ongoing ones. It boosts the GPU utilization and maximizes the overall system throughput – tokens generated per second across all users and requests. However, as mentioned in §[1](#S1 "1 Introduction ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") and elaborated later in §[2.3](#S2.SS3 "2.3 Problems and Opportunities ‣ 2 Background and Motivation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), this approach leads to trade-offs between TTFT and TPOT. An advanced variant of continuous batching [Xivz23] attempts to balance TTFT and TPOT by segmenting prefill and attaching decoding jobs in a manner that avoids exceeding GPU performance limits – but essentially, it trades TTFT for TPOT. In summary, batching prefill and decoding invariably leads to compromises in either TTFT or TPOT.

Model parallelism. In LLM serving, model parallelism is generally divided as intra- and inter-operator parallelisms [Xivaa23, OSDIb22, Traini20]. Both can be used to support larger models but may impact serving performance differently. Intra-operator parallelism partitions computationally intensive operators, such as matrix multiplications, across multiple GPUs, accelerating computation but causing substantial communication. It reduces the execution time [+2], hence latency, particularly for TTFT of the prefill phase, but requires high bandwidth connectivity between GPUs (e.g., NVLink). Inter-operator parallelism organizes LLM layers into stages, each running on a GPU to form pipelines. It moderately increases execution time due to inter-stage communication, but linearly scales the system’s rate capacity with each added GPU. In this paper, we reveal an additional benefit of model parallelism: reduced queuing delay of both prefill and decoding phases, steaming from shorter execution time. We delve into this further in §[3](#S3 "3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"). Besides model parallelism, replicating a model instance, irrespective of its model parallelism configurations, linearly scales the system’s rate capacity.

These parallelism strategies create a complex space of optimization that requires careful trade-offs based on the application’s latency requirements.

### 2.3 Problems and Opportunities

Colocating and batching the prefill and decoding computation to maximize the overall system throughput, as in existing systems, is cost-effective for service providers. However, in the presence of SLOs, present approaches struggle to maintain both high service quality and low cost due to the issues discussed below.

Prefill-decoding interference. As [Figure 2](#figure-02) shows, adding a single prefill job to a batch of decoding requests significantly slows down both processes, leading to a marked increase in TTFT and TPOT. Specifically, the decoding tasks in the batch must wait for lengthier prefill jobs to complete, thus extending TPOT; the slowdown intensifies with a longer prefill, shown in [Figure 2](#figure-02)(b). Adding decoding jobs to prefill also increases the time to complete the prefill task, particularly when the GPU is already at capacity ([Figure 2](#figure-02) blue curves).

Ineffective scheduling. Unbatching prefill and decoding jobs and scheduling them sequentially does not mitigate the interference. Decoding jobs may experience longer queuing delays due to waiting for ongoing prefill jobs on GPUs. Moreover, batches dedicated to decoding often lead to GPU underutilization. Prioritizing tasks in either phase adversely affects the latency of the other, rendering priority scheduling ineffective.

Resource and parallelism coupling. Colocating prefill and decoding phases on the same GPUs unavoidably share their resource and parallelism settings. However, each phase has its unique computational characteristic and latency requirement that calls for more heterogeneous resource allocation. For example, the prefill phase benefits from more GPUs and intra-op parallelism to reduce execution time to meet the tight SLO on TTFT. The decoding phase can handle a much higher rate using fewer GPUs than prefill, and its optimal parallelism configuration depends on the running batch size. In existing systems, due to coupling, resource allocation and parallelism plans are tailored to satisfy the *more demanding* of TTFT and TPOT, which may not be ideal for the other. This often leads to resource over-provisioning to meet both SLOs.

Opportunities. To address these issues, we propose to disaggregate the prefill and decoding phases. We use the term instance to denote a unit of resources that manages exactly one complete copy of model weights. One instance can correspond to many GPUs when model parallelism is applied. Note that when we disaggregate the two phases to different GPUs, each phase manages its copy of the model weights, resulting in *prefill instances* and *decoding instances*. A prefill instance, upon receiving a request, performs only the prefill computation for this request to generate the first output token. It then sends the intermediate results (mainly KV caches) to a decoding instance, which is responsible for subsequent decoding steps. Because decoding computation often has low GPU utilization, we may allocate multiple prefill instances per decoding instance. This allows batching more decoding jobs to achieve higher GPU utilization.

Disaggregating prefill and decoding naturally resolves the interference between the two phases and enables each to focus on its optimization target – TTFT or TPOT. Each type of instance can employ different resources and parallelism strategies to meet a variety of latency requirements. By adjusting the number of GPUs and parallelisms provided to the two types of instances, we can maximize the per-device goodput of the overall system, avoiding over-provisioning, eventually translating to reduced cost-per-query adhering to service quality. Next, we develop ways to find out the best resource allocation and parallelism plan for each phase.

## 3 Tradeoff Analysis

Disaggregation uncouples the two phases and allows a distinct analysis of the characteristics of each phase, providing valuable insights into the algorithm design. It also expands the design space: now each phase needs to be scaled and scheduled independently based on their latency requirements.

In this section, we analyze the computational pattern of prefill (§[3.1](#S3.SS1 "3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) and decoding instances (§[3.2](#S3.SS2 "3.2 Analysis for Decoding Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) *post disaggregation*. We aim to identify key parameters and derive guidelines for batching and parallelism in each phase. We then highlight several practical deployment considerations (§[3.3](#S3.SS3 "3.3 Practical Problems ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). This section lays the foundation for per-gpu goodput optimization.

### 3.1 Analysis for Prefill Instance

<span id="figure-03"></span>

![Refer to caption](../../papers/distserve/figure-03.png)

**Figure 3.** Throughput for prefill phase and decoding phase with different batch size and input length when serving an LLM with 13B parameters.

After disaggregation, the prefill phase generates the first token by processing all tokens of the user prompt in parallel. Assuming a given arrival rate, our goal is to fulfill the service’s latency requirement on TTFT using the least resources.

Batching strategy. The prefill step is typically compute-intensive. [Figure 3](#figure-03)(a) shows how the throughput of the prefill phase changes with the input length and the batch size. For a 13B parameter LLM, processing a single sequence of 512 tokens can fully engage an A100 GPU; larger models require shorter sequences to reach GPU saturation. Once the GPU becomes compute-bound, adding more requests to the batch no longer improves GPU efficiency. Instead, it proportionally extends the total processing time for the batch, inadvertently delaying all included requests. Hence, for prefill instances, it is necessary to profile the specific LLM and GPUs in advance to identify a critical input length threshold, denoted as $L_{m}$, beyond which the prefill phase becomes compute-bound. Batching more requests should only be considered when the input length of the scheduled request is below $L_{m}$. In practice, user prompts typically average over hundreds of tokens [Shareg23]. Batch sizes for the prefill instance are generally kept small.

<span id="figure-04"></span>

![Refer to caption](../../papers/distserve/figure-04.png)

**Figure 4.** Average TTFT when serving an LLM with 66B parameters using different parallelism on two A100 GPUs.

Parallelism plan. To study the parallelism preferences for prefill-only instances, we serve a 66B LLM on two A100 GPUs with inter-op or intra-op parallelism strategy. To simplify the problem, we assume uniform requests input lengths of 512 tokens and a Poisson arrival process. We compare the resulting average TTFT at various arrival rates in [Figure 4](#figure-04)(a): intra-op parallelism is more efficient at lower arrival rates, while inter-op parallelism gains superiority as the rate increases. Disaggregation enables the prefill phase to function analogously to an M/D/1 queue, so we can use queuing theory to verify the observation.

We start by developing notations using the single-device case without parallelism: each request’s execution time, denoted as $D$, remains constant due to uniform prefill length. Since one request saturates the GPU, we schedule requests via First-Come-First-Served (FCFS) without batching. Suppose the Poisson arrival rate is $R$ and the utilization condition of $\mathrm{RD}<1$, the average TTFT ($\mathrm{Avg}\_\mathrm{TTFT}$) can be modeled by the M/D/1 queue [Sons18] in close form:

$$
\small \mathrm{Avg}\_\mathrm{TTFT}=D+\frac{\mathrm{RD}^{2}}{2(1-\mathrm{RD})},\tag{1}
$$

where the first term represents the execution time and the second corresponds to the queuing delay. Based on Eq. [1](#S3.E1 "Equation 1 ‣ 3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), we incorporate parallelism.

With 2-way inter-op parallelism, we assume the request-level latency becomes $D_{s}$, and the slowest stage takes $D_{m}$ to finish. We have $D\approx D_{s}\approx 2\times D_{m}$, due to negligible inter-layer activation communication [OSDIb22, Xivaa23]. The average TTFT with 2-way inter-op parallelism is derived as:

$$
\small \mathrm{Avg}\_\mathrm{TTFT}_{\mathrm{inter}}=D_{s}+\frac{\mathrm{RD}_{m}^{2}}{2(1-\mathrm{RD}_{m})}=D+\frac{\mathrm{RD}^{2}}{4(2-\mathrm{RD})}.\tag{2}
$$

For intra-op parallelism, we introduce a speedup coefficient $K$, where $1<K<2$, reflecting the imperfect speedup caused by high communication overheads of intra-op parallelism. With the execution time $D_{s}=\frac{D}{K}$, the average TTFT for 2-degree intra-op parallelism is:

$$
\small \mathrm{Avg}\_\mathrm{TTFT}_{\mathrm{intra}}=\frac{D}{K}+\frac{\mathrm{RD}^{2}}{2K(K-\mathrm{RD})}.\tag{3}
$$

Comparing Eq. [2](#S3.E2 "Equation 2 ‣ 3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") and Eq. [3](#S3.E3 "Equation 3 ‣ 3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"): at lower rates, where execution time (first term) is the primary factor, intra-op parallelism’s reduction in execution time makes it more efficient. As the rate increases and the queuing delay (second term) becomes more significant, inter-op parallelism becomes advantageous, concurred with [Figure 4](#figure-04)(a).

The prefill phase’s preference for parallelism is also influenced by TTFT SLO and the speedup coefficient $K$. Seen from [Figure 4](#figure-04)(a): A more stringent SLO will make intra-op parallelism more advantageous, due to its ability to support higher request rates while adhering to SLOs. The value of K depends on factors such as the input length, model architecture, communication bandwidth, and placement [Traini20, OSDIb22]. As shown in [Figure 4](#figure-04)(b), a decrease in K notably reduces the efficacy of intra-op parallelism. §[4](#S4 "4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") develops algorithms that optimize the resource and parallelism configurations taking into consideration these knobs.

### 3.2 Analysis for Decoding Instance

Unlike the prefill instance, a decoding instance follows a distinct computational pattern: it receives the intermediate states (KV caches) and the first token from the prefill instance and generates subsequent tokens one at a time. For decoding instances, our optimization goal is to satisfy the application’s TPOT requirement using minimal computing resources.

Batching strategy. Since a single decoding job is heavily bandwidth-bound, batching is key to avoiding low GPU utilization (hence high per-gpu goodput). In existing systems where the prefill and decoding phases are colocated, increasing the decoding batch size is difficult because it conflicts with meeting latency goals, particularly in scenarios with high request rates. This is because sharing GPUs cause competition between prefill and decoding jobs, leading to a trade-off between TTFT and TPOT. For example, a higher arrival rate generates more prefill jobs, demanding greater GPU time to meet TTFT requirements if prioritizing prefill jobs, which in turn adversely affects TPOT.

On the contrary, disaggregation offers a solution by enabling the allocation of multiple prefill instances to a single decoding instance. This approach allows for accumulating a larger batch size on dedicated GPUs for the decoding phase without sacrificing TPOT.

<span id="figure-05"></span>

![Refer to caption](../../papers/distserve/figure-05.png)

**Figure 5.** Decoding phase latency and throughput when serving a 13B LLM with batch size = 128 and input length = 256 under different parallel degrees.

Parallelism plan. Post-disaggregation, the batch size for decoding may be constrained by GPU memory capacity, as it is necessary to maintain the KV caches for all active requests. Scaling the decoding instance with model parallelism or leveraging advanced memory management techniques for LLM KV caches, such as Paged-Attention [Efficc23] and GQA [Traini23], enable further scaling the decoding batch size to nearly compute-bound. As the decoding batch size continue to increase to approach the compute bound, the decoding computation begins to resemble the prefill phase. With this observation, we investigate how the latency and throughput change under different parallelism degrees under large batch conditions in [Figure 5](#figure-05): intra-op parallelism reduces latency with diminishing returns, caused by communication and reduced utilization after partitioning. Inter-op parallelism can almost linearly scale the throughput. Hence, when the TPOT SLO is stringent, intra-op parallelism is essential to reduce TPOT to meet latency goals. Beyond this, inter-op parallelism is preferable to enhance throughput linearly.

It is worth noting that when the model can fit into the memory of a single GPU, replication is a competitive option in addition to model parallelism for both prefill and decoding instances, to linearly scale the system’s rate capacity. It may also reduce the queuing delay – as indicated by Eq. [1](#S3.E1 "Equation 1 ‣ 3.1 Analysis for Prefill Instance ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") – by substituting $R$ with $R/N$ assuming requests are equally dispatched to $N$ replicas, at the cost of maintaining additional replicas of the model weights in GPU memory.

### 3.3 Practical Problems

We have developed foundational principles for selecting batching and parallelisms for each phase. In this section, we discuss and address several challenges encountered during the practical deployment of disaggregated prefill and decoding phases.

Variable prefill length. §[3](#S3 "3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") has assumed uniform prompt length across requests. In real deployments, depending on the LLM application, the lengths of requests are non-uniform. The non-uniformity can cause pipeline bubbles [Effici19, SOSPb19] for prefill instances applying inter-op parallelism, because the execution time of pipeline stages across requests of different lengths will vary. This results in slight deviations from the conclusions indicated by using M/D/1 queue model. To address the problem, §[4](#S4 "4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") develops algorithms that search for parallelisms based on workloads, and resort to scheduling to minimize the bubbles (§[4.3](#S4.SS3 "4.3 Online scheduling ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")).

Communication overhead. Transferring KV caches from prefill to decoding instances incurs notable overheads. For example, the KV cache size of a single 512-token request on OPT-66B is approximately 1.13GB. Assuming an average arrival rate of 10 requests per second, we need to transfer $1.13\times 10=11.3$ GB data – or equivalently 90Gbps bandwidth to render the overhead invisible. The size of the KV caches increases with average input length and arrival rate. While many modern GPU clusters for LLMs are equipped with Infiniband (e.g., 800 Gbps), in cases where cross-node bandwidth is limited, disaggregation relies on the commonly available intra-node NVLINK, where the peak bandwidth between A100 GPUs is 600 GB/s, again rendering the transmission overhead negligible (see §[6.3](#S6.SS3 "6.3 Latency Breakdown ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). However, this requirement imposes additional constraints on the placement of preill and decoding instances that we take into consideration in the next section.

Through the analysis in this section, we identify the workload pattern, placement constraints, SLO requirements, parallelism strategies, and resource allocation as key parameters that create a web of considerations in designing the disaggregated serving system. How to automatically navigate the search space to find the configuration that achieves optimal per-gpu goodput is challenging, and addressed next.

## 4 Method

We built DistServe to solve the above challenges. Given the model, workload characteristic, latency requirements, and SLO attainment target, DistServe will determine (a) the parallelism strategies for prefill and decoding instances, (b) the number of each instance type to deploy, as well as (c) how to place them onto the physical cluster. We call the solution a placement. Our goal is to find a placement that maximizes the per-gpu goodput.

As explained in §[3.3](#S3.SS3 "3.3 Practical Problems ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), a key design consideration is to manage communications between disaggregated prefill and decoding phases, given varying cluster setups. In this section, we first present two placement algorithms: one for clusters with high-speed cross-node networks (§[4.1](#S4.SS1 "4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) and the other for environments lacking such infrastructure (§[4.2](#S4.SS2 "4.2 Placement for Low Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")); the latter introduces additional constraints. We then develop online scheduling optimizations that adapt to the nuances of real-world workloads (§[4.3](#S4.SS3 "4.3 Online scheduling ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")).

### 4.1 Placement for High Node-Affinity Cluster

**Algorithm 1: High Node-Affinity Placement Algorithm.**

- **Input:** LLM $G$, #node limit per-instance $N$, #GPU per-node $M$, GPU memory capacity $C$, workload $W$, and traffic rate $R$.
- **Output:** The placement $\mathit{\mathrm{best}\_\mathrm{plm}}$.
- Initialize $\mathit{\mathrm{prefill}\_\mathrm{config}}\leftarrow\emptyset$ and $\mathit{\mathrm{decode}\_\mathrm{config}}\leftarrow\emptyset$.
- **For** $\mathit{\mathrm{intra}\_\mathrm{op}}\in\{1,2,...,M\}$:
  - **For** $\mathit{\mathrm{inter}\_\mathrm{op}}\in\{1,2,...,\frac{N\times M}{\mathit{\mathrm{intra}\_\mathrm{op}}}\}$:
    - **If** $\frac{G.\mathrm{size}}{\mathit{\mathrm{inter}\_\mathrm{op}}\times\mathit{\mathrm{intra}\_\mathrm{op}}}<C$:
      - $\hat{G}\leftarrow\mathrm{parallel}(G,\mathit{\mathrm{inter}\_\mathrm{op}},\mathit{\mathrm{intra}\_\mathrm{op}})$.
      - $\mathit{\mathrm{prefill}\_\mathrm{goodput}}\leftarrow\mathrm{simu\_prefill}(\hat{G},W)$.
      - $\mathit{\mathrm{decode}\_\mathrm{goodput}}\leftarrow\mathrm{simu\_decode}(\hat{G},W)$.
      - **If** $\frac{\mathit{\mathrm{prefill}\_\mathrm{config}.\mathrm{goodput}}}{\mathrm{prefill}\_\mathrm{config}.\mathrm{num}\_\mathrm{gpus}}<\frac{\mathit{\mathrm{prefill}\_\mathrm{goodput}}}{\mathrm{config}.\mathrm{num}\_\mathrm{gpus}}$:
        - $\mathit{\mathrm{prefill}\_\mathrm{config}}\leftarrow\mathit{\mathrm{config}}$.
      - **If** $\frac{\mathit{\mathrm{decode}\_\mathrm{config}.\mathrm{goodput}}}{\mathrm{decode}\_\mathrm{config}.\mathrm{num}\_\mathrm{gpus}}<\frac{\mathit{\mathrm{decode}\_\mathrm{goodput}}}{\mathrm{config}.\mathrm{num}\_\mathrm{gpus}}$:
        - $\mathit{\mathrm{decode}\_\mathrm{config}}\leftarrow\mathit{\mathrm{config}}$.
- $n\leftarrow\lceil\frac{R}{\mathit{\mathrm{prefill}\_\mathrm{config}.\mathrm{goodput}}}\rceil$ and $m\leftarrow\lceil\frac{R}{\mathit{\mathrm{decode}\_\mathrm{config}.\mathrm{goodput}}}\rceil$.
- $\mathit{\mathrm{best}\_\mathrm{plm}}\leftarrow(\mathit{\mathrm{prefill}\_\mathrm{config}},\mathit{\mathrm{decode}\_\mathrm{config}},n,m)$.
- **Return:** $\mathit{\mathrm{best}\_\mathrm{plm}}$.

On high node-affinity clusters equipped with Infiniband, KV caches transmission overhead across nodes is negligible, DistServe can efficiently deploy prefill and decoding instances across any two nodes without constraints. We propose a two-level placement algorithm for such scenarios: we first optimize the parallelism configurations for prefill and decoding instances separately to attain phase-level optimal per-gpu goodput; then, we use replication to match the overall traffic rate.

However, finding the optimal parallel configuration for a single instance type, such as for the prefill instance, is still challenging, due to the lack of a simple analytical formula to calculate the SLO attainment (a.k.a., percentage of requests that meet TTFT requirement), given that the workload has diverse input, output lengths, and irregular arrival patterns. Gauging the SLO via real-testbed profiling is time-prohibitive. We thus resort to building a simulator to estimate the SLO attainment, assuming prior knowledge of the workload’s arrival process and input and output length distributions. Although short-term interval is impossible to predict, the workload pattern over longer timescales (e.g., hours or days) is often predictable [Xivaa23, Servin23]. DistServe fits a distribution from the history request traces and resamples new traces from the distribution as the input workload to the simulator to compute the SLO attainment. Next, DistServe simply enumerates the placements via binary search and finds the maximum rate that meets the SLO attainment target with simulation trials.

Algorithm [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") outlines the process. We enumerate all feasible parallel configurations, subject to cluster capacity limit, for both prefill and decoding instances. For example, for a specific prefill phase configuration, we use `simu_prefill` to simulate and find their maximum goodput (similarly for using `simu_decode` for decoding). After determining the optimal parallel configurations for both prefill and decoding instances, we replicate them to achieve the user-required overall traffic rate according to their goodput.

The complexity of Algorithm [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") is $O(\mathrm{NM}^{2})$, with $N$ as the node limit per instance and $M$ representing the typical number of GPUs per node in modern clusters (e.g., 8). The search space is manageable and the solving time is under 1.3 minutes in our largest setting, as demonstrated in §[6.5](#S6.SS5 "6.5 Algorithm Running Time ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving").

Simulator building. Algorithm [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") relies on a simulator to estimate the goodput under various SLOs and SLO attainment goals given the workload and the parallelism plan. To build an accurate simulator, we analyze the FLOPs and the number of memory accesses for prefill and decoding phases respectively, and use a latency model to approximate the inference execution time. See details in Appendix [A](#A1 "Appendix A Latency Model for LLM Inference ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"). The simulator aligns well with real profiling results, thanks to the high predictability of DNN workloads [Novemb20, Xivaa23], verified in §[6.4](#S6.SS4 "6.4 Ablation Studies ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving").

By far, we have developed Algorithm [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") assuming we can place the prefill and decoding between any two nodes of the cluster, and the KV cache transmission utilizes high bandwidth. In many real clusters, GPUs inside a node access to high-bandwidth NVLINK while GPUs distributed across nodes have limited bandwidth. We next develop an algorithm to address this constraint.

**Algorithm 2: Low Node-Affinity Placement Algorithm.**

- **Input:** LLM $G$, #node limit per-instance $N$, #GPU per-node $M$, GPU memory capacity $C$, workload $W$, and traffic rate $R$.
- **Output:** The placement $\mathit{\mathrm{best}\_\mathrm{plm}}$.
- Initialize $\mathit{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}}\leftarrow\emptyset$.
- **For** $\mathit{\mathrm{inter}\_\mathrm{op}}\in\{1,2,...,N\}$:
  - $\hat{G}\leftarrow\mathrm{parallel}(G,\mathit{\mathrm{inter}\_\mathrm{op}})$.
  - $\mathcal{P}\leftarrow\mathrm{get\_intra\_node\_configs}(\hat{G},M,C)$.
  - **For** $P\in\mathcal{P}$:
    - $\mathit{P.\mathrm{goodput}}\leftarrow\mathrm{simulate}(\hat{G},P,W)$.
    - **If** $\frac{\mathit{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}.\mathrm{goodput}}}{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}.\mathrm{num}\_\mathrm{gpus}}<\frac{\mathit{P.\mathrm{goodput}}}{P.\mathrm{num}\_\mathrm{gpus}}$:
      - $\mathit{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}}\leftarrow P$.
- $n\leftarrow\lceil\frac{R}{\mathit{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}.\mathrm{goodput}}}\rceil$.
- $\mathit{\mathrm{best}\_\mathrm{plm}}\leftarrow(\mathit{\mathrm{inter}\_\mathrm{op}},\mathit{\mathrm{intra}\_\mathrm{node}\_\mathrm{config}},n)$.
- **Return:** $\mathit{\mathrm{best}\_\mathrm{plm}}$.

### 4.2 Placement for Low Node-Affinity Cluster

A straightforward solution is to always colocate prefill and decoding instances on the same node, utilizing the NVLINK, which is commonly available inside a GPU node. For large models, e.g. with 175B parameters (350GB), we may be unable to even host a single pair of prefill and decoding instances in an 8-GPU node ($80G\times 8=640G<350\times 2\mathrm{GB}$). We incorporate this as additional placement constraints and co-optimize it with model parallelism, presented in Algorithm [2](#alg2 "Algorithm 2 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving").

The key insight is that intermediate states transfers occur exclusively between corresponding layers of prefill and decoding instances. Leveraging inter-op parallelism, we group layers into stages and divide each instance into segments, termed as instance segments, with each segment maintaining one specific inter-op stage. By colocating prefill and decoding segments of the same stage within a single node, we force the transfer of intermediate states to occur only via NVLINK. Inside a node, we set the same parallelism and resource allocation for segments of the same instance. Given the typical limitation of GPUs per node (usually 8), we can enumerate possible configurations inside one node and use the simulator to identify the configurations that yield the best goodput.

As outlined in Algorithm [2](#alg2 "Algorithm 2 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), we begin by enumerating inter-op parallelism degrees to get all the possible instance segments. For each segment, we get all possible intra-node configurations by calling `get_intra_node_configs`. Then we use simulation to find the optimal one and replicate it to satisfy the target traffic rate.

### 4.3 Online scheduling

<span id="figure-06"></span>

![Refer to caption](../../papers/distserve/figure-06.png)

**Figure 6.** DistServe Runtime System Architecture

The runtime architecture of DistServe is shown in [Figure 6](#figure-06). DistServe operates with a simple FCFS scheduling policy. All incoming requests arrive at a centralized controller, then dispatched to the prefill instance with the shortest queue for prefill processing, followed by dispatch to the least loaded decoding instance for decoding steps. This setup, while simple, is optimized with several key enhancements tailored to the nuances of real-world workloads.

Reducing pipeline bubbles. To mitigate the pipeline bubbles caused by non-uniform prompt lengths (§[3.3](#S3.SS3 "3.3 Practical Problems ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")), we schedule the requests in a way that balances the execution time across all batches in the pipeline. This is achieved by noting that, for both prefill and decoding instances, the number of new tokens in the batch is a reliable indicator of the batch’s real execution time. For prefill instances, we profile the target model and GPU to figure out the shortest prompt length $L_{m}$ needed to saturate the GPU. We schedule prefill batches with a total sequence length close to $L_{m}$, by either batching multiple requests shorter than $L_{m}$ or individually scheduling requests longer than $L_{m}$. For decoding instances, we set $L_{m}$ as the largest batch size.

Combat busrtiness. Burstiness in workloads can cause a deluge of KV caches to transfer from prefill to decoding instances, risking memory overload on decoding instances. To circumvent this, DistServe employs a “pull” method for KV cache transmission rather than a “push” approach – decoding instances fetch KV cache from prefill instances *as needed*, using the GPU memory of prefill instances as a queuing buffer. Hence, each type of instance operates at its own pace without complex coordination.

Replaning. The resource and parallelism plan in DistServe is optimized for a specific workload pattern, which may become suboptimal if the workload pattern changes over time. DistServe implement periodic replanning. A workload profiler monitors key parameters such as the average input and output length of the requests, the average arrival rate, etc. If a significant pattern shift is detected, DistServe will trigger a rerun of the placement algorithm based on recent historical data. This process is expedient – the proposed algorithm runs in seconds (§[6.5](#S6.SS5 "6.5 Algorithm Running Time ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) and reloading LLM weights can be completed within minutes – far shorter than the hourly scale at which real-world workload variations tend to occur.

DistServe does not implement advanced runtime policies like preemption [Julyb22] and fault tolerance [Systen21], which are complementary to disaggregation. Nevertheless, we discuss how they fit into DistServe. In DistServe, the FCFS policy can lead to a “convoy effect”, where longer requests block shorter ones in the prefill stage. Incorporating preemptive strategies, as suggested in existing literature [Xivac23], could enhance efficiency and is feasible within our system’s architecture. While not a primary focus in the current DistServe, fault tolerance is a critical aspect for consideration. In traditional colocation- and replication-based systems, a fault in one instance typically does not disrupt other replica instances. However, in DistServe, the dependency between prefill and decoding instances introduces the risk of fault propagation. For example, a fault in a single decoding instance mapped to multiple prefill instances could potentially cripple the entire service and cluster. We leave both as future work.

## 5 Implementation

DistLLM is an end-to-end distributed serving system for LLMs with a placement algorithm module, a RESTful API frontend, an orchestration layer, and a parallel execution engine. The algorithm module, frontend, and orchestration layer are implemented with 6.5K lines of Python code. The parallel execution engine is implemented with 8.1K lines of C++/CUDA code.

The placement algorithm module implements the algorithm and the simulator mentioned in §[4](#S4 "4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") which gives the placement decision for a specific model and cluster setting. The frontend supports OpenAI API compatible interface where clients can specify the sampling parameters like maximum output length and temperature. The orchestration layer manages the prefill and decoding instances, responsible for request dispatching, KV caches transmission, and results delivery. It utilizes NCCL [Nvidia23] for cross-node GPU communication and asynchronous cudaMemcpy for intra-node communication, which avoids blocking the GPU during transmission. Each instance is powered by a parallel execution engine, which uses Ray [OSDIg18] actor to implement GPU workers that execute the LLM inference and manage the KV Cache in a distributed manner. It integrates many recent LLM optimizations like continuous batching [OSDIa22], FlashAttention [Fastb22], PagedAttention [Efficc23] and supports popular open-source LLMs such as OPT [Opena22] and LLaMA [Openb23].

## 6 Evaluation

In this section, we evaluate DistServe under different sizes of LLMs ranging from 13B to 175B and various application datasets including chatbot, code-completion, and summarization. The evaluation shows that DistServe consistently outperforms the current state-of-the-art system across all the settings (§[6.2](#S6.SS2 "6.2 End-to-end Experiments ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")). Specifically, DistServe can handle up to $4.48\times$ higher rates and $10.2\times$ more stringent SLO while meeting the latency requirements for over 90% requests. Additionally, we analyze the latency breakdown in DistServe to show the communication overhead is insubstantial thanks to our bandwidth-aware placement algorithm (§[6.3](#S6.SS3 "6.3 Latency Breakdown ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) and do ablation studies of our techniques (§[6.4](#S6.SS4 "6.4 Ablation Studies ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")).

<span id="table-01"></span>

| Application | Model Size | TTFT | TPOT | Dataset |
| --- | --- | --- | --- | --- |
| Chatbot OPT-13B | 26GB | 0.2s | 0.1s | ShareGPT [Shareg23] |
| Chatbot OPT-66B | 132GB | 0.4s | 0.1s | ShareGPT [Shareg23] |
| Chatbot OPT-175B | 350GB | 4.0s | 0.2s | ShareGPT [Shareg23] |
| Code Completion OPT-66B | 132GB | 0.125s | 0.2s | HumanEval [Evalua21] |
| Summarization OPT-66B | 132GB | 15s | 0.15s | LongBench [Longbe23] |

**Table 1.** Workloads in evaluation and latency requirements.

<span id="figure-07"></span>

![Refer to caption](../../papers/distserve/figure-07.png)

**Figure 7.** The input and output length distributions of (a) ShareGPT, (b) HumanEval, and (c) LongBench datasets.

<span id="figure-08"></span>

![Refer to caption](../../papers/distserve/figure-08a.png)

![Refer to caption](../../papers/distserve/figure-08b.png)

**Figure 8.** Chatbot application with OPT models on the ShareGPT dataset.

### 6.1 Experiments Setup

Cluster testbed. We deploy DistServe on a cluster with 4 nodes and 32 GPUs. Each node has 8 NVIDIA SXM A100-80GB GPUs connected with NVLINK. The cross-node bandwidth is 25Gbps. Due to the limited cross-node bandwidth, we use the low node-affinity placement algorithm (§[2](#alg2 "Algorithm 2 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) for DistServe in most of the experiments except for the ablation study (§[6.4](#S6.SS4 "6.4 Ablation Studies ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving")) which uses simulation.

Model and workloads setup. Similar to prior work on LLM serving [Efficc23], we choose the OPT [Opena22] series, which is a representative LLM family widely used in academia and industry. We use FP16 precision in all experiments. For workloads, as shown in [Table 1](#table-01), We choose three typical LLM applications and set the SLOs empirically based on their service target because there exists no available SLO settings for these applications as far as we know. For each application, we select a suitable dataset and sample requests from it for evaluation. Since all the datasets do not include timestamps, we generate request arrival times using Poisson distribution with different request rates. Due to the space limit, we test the chatbot workload on all three OPT models and the other two workloads on OPT-66B, which matches the largest size in the recent open-source LLM series [Openb23].

-   •

    Chatbot [Introd22]: We use the ShareGPT dataset [Shareg23] for the chatbot application, which is a collection of user-shared conversations with ChatGPT. For OPT-13B, the TTFT SLO is set to 0.2s for responsiveness and the TPOT SLO is set to 0.1s which is higher than the normal human read speed. For OPT-66B and OPT-175B, we slightly relax the two SLOs due to the increase of model execution latency.

-   •

    Code completion [Evalua21]: We use the HumanEval [Evalua21] dataset for the code completion task. It includes 164 programming problems with a function signature or docstring which is commonly used in academia to evaluate code completion models. Since the code completion tool is used as a personal real-time coding assistant, we set both SLOs to be stringent.

-   •

    Summarization [Summar23]: It is a popular LLM task to generate a concise summary for a long article, essay, or even an academic paper. We use LongBench [Longbe23] dataset which contains the summarization task. As shown in [Figure 7](#figure-07), LongBench has much longer input lengths than the other two datasets. So we set a loose TTFT SLO but require a stringent TPOT.

Metrics. We use SLO attainment as the major evaluation metric. Under a specific SLO attainment goal (say, 90%), we are concerned with two things: the maximum per-GPU goodput and the minimal SLO the system can handle. We are particularly interested in an SLO attainment of 90% (indicated by the vertical lines in all curve plots), but will also vary the rate and latency requirements to observe how the SLO attainment changes. To accurately understand the respective impacts of the two latency requirements on the system, we also present the proportion of requests that only meet one of these SLOs.

Baseline. We compare DistServe to the state-of-the-art serving system vLLM [Efficc23]. It supports iteration-level scheduling proposed by Orca [OSDIa22] and PagedAttention to reduce memory fragmentation caused by KV cache allocation. However, it colocates and batches the prefill and decoding computation to maximize the overall system throughput and struggles to meet the latency requirements in a cost-efficient way. Since vLLM only supports intra-op parallelism, we follow previous work [Efficc23] to set intra-op equals 1, 4, and 8 for the three OPT models, respectively.

<span id="figure-09"></span>

![Refer to caption](../../papers/distserve/figure-09.png)

**Figure 9.** Code completion and summarization tasks with OPT-66B on HumanEval and LongBench datasets, respectively.

### 6.2 End-to-end Experiments

In this Section, we compare the end-to-end performance of DistServe against vLLM on real application datasets.

Chatbot. We evaluate the performance of DistServe on the chatbot application for all three OPT models. The first row of [Figure 8](#figure-08) illustrates that when we gradually increase the rate, more requests will violate the latency requirements and the SLO attainment decreases. The vertical line shows the maximum per-GPU rate the system can handle to meet latency requirements for over 90% of the requests. The dotted and dashed lines show the achieved SLO attainment for only TTFT or TPOT requirements, respectively.

On the ShareGPT dataset, DistServe can sustain $2.0\times$–$3.41\times$ higher request rate compared to vLLM. This is because DistLLM eliminates the prefill-decoding interference through disaggregation. Two phases can optimize their own objectives by allocating different resources and employing tailored parallelism strategies. As a result, the gap between the curve that only meets TTFT requirements (Dist-TTFT) and the one that only meets TPOT requirements (Dist-TPOT) is relatively small. Specifically, by analyzing the chosen placement strategy [+3] for 175B, we find the prefill instance has inter-op = 3, intra-op = 3; and the decoding instance has inter-op = 3, intra-op = 4. Under this placement, DistServe can effectively balance the load between the two instances on ShareGPT, meeting latency requirements at the lowest cost. This non-trivial placement strategy is challenging to manually find, proving the effectiveness of the algorithm. In the case of vLLM, collocating prefill and decoding greatly slows down the decoding phase, thereby significantly increasing TPOT. Due to the stringent TPOT requirements of chatbot applications, although vLLM meets the TTFT SLO for most requests, the overall SLO attainment is dragged down by a large number of requests that violate the TPOT SLO.

The second row of [Figure 8](#figure-08) indicates the robustness to the changing latency requirements of the two systems. We fix the rate and then linearly scale the two latency requirements in [Table 1](#table-01) simultaneously using a parameter called SLO Scale. As SLO Scale decreases, the latency requirement is more stringent. We aim to observe the most stringent SLO Scale that the system can withstand while still achieving the attainment target. [Figure 8](#figure-08) shows that DistServe can achieve $1.4\times$–$1.8\times$ more stringent SLO than vLLM, thus providing more engaging service quality to the users.

Code completion. [Figure 9](#figure-09)(a) shows the performance of DistServe on the code completion task when serving OPT-66B. DistServe can sustain $3.2\times$ higher request rate and $1.5\times$ more stringent SLO than vLLM. As a real-time coding assistant, the code completion task demands lower TTFT than chatbot, this leads to both systems ultimately being constrained by the TTFT requirement. However, in comparison, by eliminating the interference of the decoding jobs and automatically increasing intra-operation parallelism in prefill instances through the searching algorithm, DistServe reduces the average latency of the prefill jobs, thereby meeting the TTFT requirements of more requests.

Summarization. [Figure 9](#figure-09)(b) shows the performance of DistServe on the summarization task when serving OPT-66B. DistServe achieves $4.48\times$ higher request rate and $10.2\times$ more stringent SLO than vLLM. The requests sampled from LongBench dataset have long input lengths, which brings significant pressure to the prefill computation. However, due to the loose requirement of TTFT for the summarization task, the TPOT service quality becomes particularly important. The vLLM, which collocates prefill and decoding phases, with long prefill jobs, experiences a greater slowdown in the decoding phase and fails to meet the TPOT requirement.

<span id="figure-10"></span>

![Refer to caption](../../papers/distserve/figure-10.png)

**Figure 10.** Left: Latency breakdown when serving OPT-175B on ShareGPT dataset with DistServe. Right: The CDF function of KV Cache transmission time for OPT models.

### 6.3 Latency Breakdown

To understand DistServe’s performance in detail, we make a latency breakdown of the requests in DistServe. We divide the processing lifecycle of a request in DistServe into five stages: prefill queuing, prefill execution, transmission, decoding queuing, and decoding execution. The total time consumed by all requests in each stage is then summed up to determine their respective proportions in the system’s total execution time.

[Figure 10](#figure-10)(a) shows the latency breakdown for the OPT-175B models on ShareGPT dataset. We chose OPT-175B because the KV Cache transmission is more demanding for larger models. In fact, even for OPT-175B, the KV Cache transmission only accounts for less than 0.1% of the total latency. Even by examining the CDF of the absolute transmission time shown in [Figure 10](#figure-10)(b), we observe that over 95% of requests experience a delay of less than 30ms, despite our testbed having only limited cross-node bandwidth. This is due to the algorithm described in §[4.2](#S4.SS2 "4.2 Placement for Low Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), where we require the prefill and decoding instance to maintain the same stage on one machine, enabling the use of intra-node NVLINK bandwidth for transmission, thus significantly reducing transmission delay.

<span id="table-02"></span>

| Rate (req/s) | vLLM |  | DistServe-Low |  |
| --- | --- | --- | --- | --- |
| Real System | Simulator | Real System | Simulator |  |
| 1.0 | 97.0% | 96.8% | 100.0% | 100.0% |
| 1.5 | 65.5% | 65.1% | 100.0% | 100.0% |
| 2.0 | 52.8% | 51.0% | 99.3% | 99.3% |
| 2.5 | 44.9% | 46.1% | 87.3% | 88.3% |
| 3.0 | 36.7% | 38.3% | 83.0% | 84.1% |
| 3.5 | 27.8% | 28.0% | 77.3% | 77.0% |
| 4.0 | 23.6% | 24.1% | 70.0% | 68.9% |

**Table 2.** Comparison of the SLO attainment reported by the simulator and the real system under different rates.

<span id="figure-11"></span>

![Refer to caption](../../papers/distserve/figure-11.png)

**Figure 11.** Ablation experiments.

### 6.4 Ablation Studies

We study the effectiveness of the two key innovations in DistServe: disaggregation and the placement searching algorithm. In §[6.2](#S6.SS2 "6.2 End-to-end Experiments ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), we choose the default parallelism setting for vLLM following its original paper [Efficc23]. So we implement "vLLM++" which enumerates different parallelism strategies and chooses the best. For DistServe, We also compare the placement found by Alg. [2](#alg2 "Algorithm 2 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") (DistServe-Low) with the one found by Alg. [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") (DistServe-High) which has fewer searching constraints and assumes high cross-node bandwidth. Since vLLM does not support inter-op parallelism and our physical testbed does not have high cross-node bandwidth, we use simulation for this experiment.

Simulator accuracy. Noticing that DNN model execution [OSDIf20] has high predictability, even under parallel settings [Xivaa23, OSDIb22]. We study the accuracy of the simulator in Tab. [2](#table-02). For "vLLM" and "DistServe-Low", we compare the SLO attainment reported by the simulator and by real runs on our testbed under different rates. The error is less than 2% in all cases, verifying the accuracy of our simulator.

Results. [Figure 11](#figure-11) shows the performance of the four systems when serving OPT-13B on ShareGPT dataset. "vLLM++" has the same performance as "vLLM" because we find the default non-parallelism setting has the best per-GPU goodput. This further demonstrates the importance of disaggregation. The interference between the prefill and decoding phases significantly reduces the potential performance improvement through adjusting parallelism. In contrast, "DistLLM-High" can achieve further improvements over "DistLLM-Low" because it is not constrained by the deployment constraint that the prefill and decoding instance on one node should share the same model stage. Through disaggregation, we can use tailored parallelism strategies for prefill and decoding instances and optimize their targets without the coupling effects.

<span id="figure-12"></span>

![Refer to caption](../../papers/distserve/figure-12.png)

**Figure 12.** Algorithm Running Time

### 6.5 Algorithm Running Time

[Figure 12](#figure-12) shows the running time for Alg. [1](#alg1 "Algorithm 1 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") (DistServe-Low) and Alg. [2](#alg2 "Algorithm 2 ‣ 4.1 Placement for High Node-Affinity Cluster ‣ 4 Method ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving") (DistServe-High) on a AWS m5d.metal instance with 96 cores as the number of GPUs ($N\times M$) provided to a single instance increases. According to the results, DistServe scales well with the number of GPUs and is independent of the model size. This is because the simulator only simulates discrete events and the running time is the same no matter how big the model is. On the other hand, both algorithms are highly parallelizable, as the searches for different parallelism strategies are independent of each other, allowing the execution time of the algorithms to accelerate almost linearly with more CPU cores.

As the number of GPUs increases, the execution time of "Dist-Low" becomes higher than that of "Dist-High". This is because the search for parallelism strategies for prefill and decoding instances in "Dist-High" is independent and can be parallelized. But for "Dist-Low", due to additional restrictions on deployment, we need to enumerate all the possible intra-node parallelism combinations for prefill and decoding instances. Even so, the execution time of the algorithm is in minutes, and since it only needs to be executed once before each redeployment, this overhead is acceptable.

## 7 Related Work

Inference serving. There has been plenty of work on inference serving recently. They range from general-purpose production-grade systems like TorchServe [Serve23] and NVIDIA Triton [Triton19] to systems optimized specifically for Transformer-based LLMs [Xivaa23, OSDIa22, Xivz23, Xivac23, PPoPP21, Faster19, ATC22, System23]. Among them, Orca [OSDIa22] introduces iteration-level scheduling to increase throughput. vLLM [Efficc23] proposes a novel memory management strategy for KVCache. SARATHI [Xivz23] suggests a chunked-prefill approach, splitting a prefill request into chunks and piggyback decoding requests to improve hardware utilization. FastServe [Xivac23] implements iteration-level preemptive scheduling to mitigate the queuing delay caused by long jobs. However, they all employ a colocation approach for prefill and decoding processing, thus leading to severe interference.

Goodput-optimized systems. Optimizing goodput is a hot topic in DL applications. Pollux [July21] improves scheduling performance in DL clusters by dynamically adjusting resources for jobs to increase cluster-wide goodput. Sia [Princi23] introduces a heterogeneous-aware scheduling approach that can efficiently match cluster resources to elastic resource-adaptive jobs. Clockwork [Novemb20] and Shepherd [Servin23] provide latency-aware scheduling and preemption to improve the serving goodput, but they only target traditional small models. AlpaServe [Xivaa23] focuses on LLMs, employing model parallelism to statistically multiplex the GPU execution thus improving the resource utilization. However, it only targets the non-autoregressive generation. DistServe is the first work to optimize the goodput for autoregressive LLM inference.

Resource disaggregation. Resource disaggregated systems [OSDIh18, USA23, Comput23] decouple the hardware resources from the traditional monolithic server infrastructure and separate them into different pools to manage independently. It allows for more flexible, efficient, and scalable deployment and increases resource utilization. Many applications benefit from a truly disaggregated data center with high-speed network bandwidth and heterogenous hardware support  [Rev23, Thekka22]. DistServe adopts a similar concept by disaggregating its system components, allowing for independent resource scaling and management.

Model parallelism for training. DistServe is orthogonal to the large body of work on model parallelism in training [OSDIb22, Traini20, Effici19, Memory20, SOSPb19]. As described in §[3.3](#S3.SS3 "3.3 Practical Problems ‣ 3 Tradeoff Analysis ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), inference-serving workloads have unique characteristics not found in training settings. Where these systems do intersect with DistServe, is in their methods for implementing model parallelism along various dimensions. DistServe can integrate new parallelism optimizations into its placement searching algorithm.

## 8 Conclusion

We present DistServe, a new LLM serving architecture that disaggregates the prefill and decoding computation. DistServe maximizes the per-gpu goodput – the maximum request rate that can be served adhering to the SLO attainment goal for each GPU provisioned, hence resulting in up to $4.48\times$ lower cost per LLM query with guaranteed satisfaction of SLOs. Our findings affirm that as latency becomes an increasingly important metric for LLM services, prefill and decoding disaggregation is a vital strategy in promising improved performance and service quality guarantees.

## Appendix A Latency Model for LLM Inference

To accurately simulate the goodput of different placement strategies, we use an analytical model to predict the execution time of the prefill and decoding phases in LLM inference.

In modern LLM serving systems [Faster19, Efficc23, Xivac23], memory-bound operations like Softmax and LayerNorm are usually fused with matrix multiplication kernels for efficiency. Thus the GEMMs dominate the overall latency and our analysis primarily focuses on them.

### A.1 Symbol Definition

Here are symbols related to the architecture of the model:

-   •

    $h$: hidden size

-   •

    $n$: number of heads

-   •

    $s$: head size ($h=n\cdot s$)

-   •

    $m$: FFN intermediate size

Note: If tensor parallelism is used, $h$, $n$, and $m$ should be divided by the tensor parallelism size.

Below are symbols that characterize the batch to be executed:

-   •

    $B$: batch size

-   •

    $l_{0},l_{1},\dots,l_{B-1}$: input length of each request within the batch

-   •

    $t$: number of tokens in the batch, ($t=\sum_{i=0}^{B-1}l_{i}$)

-   •

    $t_{2}$: squared sum of the input lengths ($t_{2}=\sum_{i=0}^{B-1}l_{i}^{2}$)

-   •

    $b$: block size in the attention kernel. This parameter is used in FlashAttention [Fastb22], a common kernel optimization technique adopted by current LLM serving systems.

### A.2 Prefill Phase Latency Modeling

Since the attention operation uses specially optimized kernels, we first discuss the other four matrix multiplications in the prefill phase:

<span id="table-03"></span>

| GEMM Name | Shape of $M$ | Shape of $N$ |
| --- | --- | --- |
| QKV Linear | $(t,h)$ | $(h,3h)$ |
| Attn Output | $(t,h)$ | $(h,h)$ |
| FFN Input | $(t,h)$ | $(h,m)$ |
| FFN Output | $(t,m)$ | $(m,h)$ |

**Table 3.**

The arithmetic intensity (AI) of these operations is $O(t)$. On NVIDIA A100-80GB GPU, it is compute-bound when AI is over 156. Since $t$ usually can reach several hundred in real cases, all of these operations are compute-bound. Therefore, we can model the latency of these operations according to the total FLOPs:

$$
T_{1}=C_{1}\cdot(4\mathrm{th}^{2}+2\mathrm{thm})
$$

Next, we discuss the prefill attention operation with FlashAttention[Fastb22] optimization. Since the attention only operates among the tokens in the same request, current implementations launch attention kernels for each request in the same batch. For one attention head and a request with $l$ tokens, the attention kernel needs to perform a total of $2\mathrm{sl}+3\mathrm{sl}\cdot(l/b)\approx 3\mathrm{sl}\cdot(l/b)$ memory reads and writes, alongside $2\mathrm{sl}^{2}+\mathrm{sl}\approx 2\mathrm{sl}^{2}$ FLOPs. So the AI is $2b/3=10.677$ (when $b=16$) or $21.333$ (when $b=32$), indicating that it is a memory-bound operation on A100 GPU. Therefore, the whole attention layer latency (including all requests and all heads) can be modeled as:

$$
T_{2}=C_{2}\cdot n\cdot\sum_{i=0}^{B-1}\frac{3\mathrm{sl}_{i}^{2}}{b}=C_{2}\cdot\frac{3\mathrm{nst}_{2}}{b}=C_{2}\cdot\frac{3\mathrm{ht}_{2}}{b}
$$

Overall, the latency of the prefill phase can be modeled as:

$$
T_{\mathrm{Prefill}}=C_{1}\cdot(4\mathrm{th}^{2}+2\mathrm{thm})+C_{2}\cdot\frac{3\mathrm{ht}_{2}}{b}+C_{3}
$$

We use $C_{3}$ to quantify other overheads like Python Runtime, system noise, and so on. Then we use profiling and interpolation to figure out the values of $C_{1}$, $C_{2}$, and $C_{3}$.

### A.3 Decoding Phase Latency Modeling

Similarly, we first focus on the following GEMMs in the decoding phase:

<span id="table-04"></span>

| GEMM Name | Shape of $M$ | Shape of $N$ |
| --- | --- | --- |
| QKV Linear | $(B,h)$ | $(h,3h)$ |
| Attn Output | $(B,h)$ | $(h,h)$ |
| FFN Input | $(B,h)$ | $(h,m)$ |
| FFN Output | $(B,m)$ | $(m,h)$ |

**Table 4.**

The AI of these operations is $O(B)$. $B$ is limited by the GPU memory size and stringent latency requirements, so in existing serving scenarios, these operations are memory-bound. The total memory reads and writes is $8\mathrm{Bh}+4h^{2}+2\mathrm{hm}+2\mathrm{Bm}$, and since $h$ and $m$ are usually significantly larger than $B$, we can model the latency as:

$$
T_{3}=C_{4}\cdot(4h^{2}+2\mathrm{hm})
$$

As for the decoding attention operation, for one attention head and a request with $l$ generated tokens, it needs to perform $3\mathrm{sl}$ memory reads and writes, alongside $2\mathrm{sl}$ FLOPs. It is memory-bound, so we can model the latency of decoding attention as:

$$
T_{4}=C_{5}\cdot n\cdot 3s\sum_{i=0}^{B-1}l_{i}=C_{5}\cdot 3\mathrm{ht}
$$

Summing up, the latency of the decoding phase is:

$$
T_{\mathrm{Decoding}}=C_{4}\cdot(4h^{2}+2\mathrm{hm})+C_{5}\cdot 3\mathrm{ht}
$$

Here we do not introduce the overhead term (like $C_{3}$ in the profiling stage) because $4h^{2}+2\mathrm{hm}$ is already a constant, and the overhead can be put into $C_{4}$. Similarly, we use profiling and interpolation to figure out the values of $C_{4}$ and $C_{5}$.

## Appendix B DistLLM Placements in End-to-end Experiments

In the end-to-end experiments [6.2](#S6.SS2 "6.2 End-to-end Experiments ‣ 6 Evaluation ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving"), the tensor parallelism (TP) and pipeline parallelism (PP) configurations for prefill and decoding instances chosen by DistServe are listed on the right.

<span id="table-05"></span>

| Model | Dataset | Prefill |  | Decoding |  |
| --- | --- | --- | --- | --- | --- |
| TP | PP | TP | PP |  |  |
| OPT-13B | ShareGPT | 2 | 1 | 1 | 1 |
| OPT-66B | ShareGPT | 4 | 1 | 2 | 2 |
| OPT-66B | LongBench | 4 | 1 | 2 | 2 |
| OPT-66B | HumanEval | 4 | 1 | 2 | 2 |
| OPT-175B | ShareGPT | 3 | 3 | 4 | 3 |

**Table 5.**

[+1]: The overall request latency equals TTFT plus TPOT times the number of generated tokens in the decoding phase.

[+2]: we emphasize “execution time” instead of latency here because latency comprises both execution time and queuing delay.

[+3]: All the placements chosen by DistServe can be found in Appendix [B](#A2 "Appendix B DistLLM Placements in End-to-end Experiments ‣ DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving").
