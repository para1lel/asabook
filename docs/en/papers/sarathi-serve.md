---
title: 'Sarathi-Serve'
createTime: 2026/09/07 19:29:55
permalink: /en/papers/sarathi-serve/
---

> [Amey Agrawal](https://ameya.info/) [+author-note], [Nitin Kedia](https://kedianitin.com/), [Ashish Panwar](https://apanwariisc.github.io/), [Jayashree Mohan](https://www.microsoft.com/en-us/research/people/jamohan/), [Nipun Kwatra](https://www.microsoft.com/en-us/research/people/nkwatra/), [Bhargav S. Gulavani](https://x.com/bhargavgulavani), [Alexey Tumanov](https://faculty.cc.gatech.edu/~atumanov/), and [Ramachandran Ramjee](https://x.com/ramaramjee). First submitted to arXiv on March 4, 2024; current version v3. Published in the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), July 2024, pages 117-134. [Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve](https://arxiv.org/abs/2403.02310). <a href="/paper/sarathi-serve.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2403.02310). [TeX source](https://export.arxiv.org/e-print/2403.02310). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

Each LLM serving request goes through two phases. The first is *prefill* which processes the entire input prompt and produces the first output token and the second is *decode* which generates the rest of output tokens, one-at-a-time. Prefill iterations have high latency but saturate GPU compute due to parallel processing of the input prompt. In contrast, decode iterations have low latency but also low compute utilization because a decode iteration processes only a single token per request. This makes batching highly effective for decodes and consequently for overall throughput. However, batching multiple requests leads to an interleaving of prefill and decode iterations which makes it challenging to achieve both high throughput and low latency.

We introduce an efficient LLM inference scheduler, Sarathi-Serve, to address this throughput-latency tradeoff. Sarathi-Serve introduces *chunked-prefills* which splits a prefill request into near equal sized chunks and creates *stall-free* schedules that adds new requests in a batch without pausing ongoing decodes. Stall-free scheduling unlocks the opportunity to improve throughput with large batch sizes while minimizing the effect of batching on latency. Furthermore, uniform batches in Sarathi-Serve ameliorate the imbalance between iterations, resulting in minimal pipeline bubbles.

Our techniques yield significant improvements in inference performance across models and hardware under tail latency constraints. For Mistral-7B on single A100 GPUs, we achieve $2.6\times$ higher serving capacity and up to $3.7\times$ higher serving capacity for the Yi-34B model on two A100 GPUs as compared to vLLM. When used with pipeline parallelism on Falcon-180B, Sarathi-Serve provides up to $5.6\times$ gain in the end-to-end serving capacity. The source code for Sarathi-Serve is available at [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).

<span id="section-1"></span>

## 1 Introduction

Large language models (LLMs) [Wei22d, Bro20, Cho23a, Ope24a, Kap20] have shown impressive abilities in a wide variety of tasks spanning natural language processing, question answering, code generation, etc. This has led to tremendous increase in their usage across many applications such as chatbots [Ope24a, Cha22a, Ant23, Cha22], search [Bin23, Kom22, You21, Per22, Bar23], code assistants [Git21a, Rep22, Ama22], etc. The significant GPU compute required for running inference on large models, coupled with significant increase in their usage, has made LLM inference a dominant GPU workload today. Thus, optimizing LLM inference has been a key focus for many recent systems [Pop22, She23, Yu22a, Kwo23, Pat23, Zho24, Agr23].

<span id="figure-01"></span>

![Yi-34B running on two A100 GPUs serving 128 requests from the arxiv-summarisation trace.](../../papers/sarathi-serve/figure-01.png)

**Figure 1.** Yi-34B running on two A100 GPUs serving 128 requests from *arxiv-summarisation* trace. [Figure 1(a)](#figure-01) highlights one of the many generation stalls lasting over several seconds in vLLM [Kwo23]. [Figure 1(b)](#figure-01) shows the impact of increasing load on tail latency. Sarathi-Serve improves throughput while eliminating generation stalls.

Optimizing throughput and latency are both important objectives in LLM inference since the former helps keep serving costs tractable while the latter is necessary to meet application requirements. In this paper, we show that current LLM serving systems have to face a tradeoff between throughput and latency. In particular, LLM inference throughput can be increased significantly with batching. However, the way existing systems batch multiple requests leads to a compromise on either throughput or latency. For example, [Figure 1(b)](#figure-01) shows that increasing load can significantly increase tail latency in a state-of-the-art LLM serving system vLLM [Kwo23].

Each LLM inference request goes through two phases — a *prefill* phase followed by a *decode* phase. The *prefill* phase corresponds to the processing of the input prompt and the *decode* phase corresponds to the autoregressive token generation. The prefill phase is compute-bound because it processes all tokens of an input prompt in parallel whereas the decode phase is memory-bound because it processes only one token per-request at a time. Therefore, decodes benefit significantly from batching because larger batches can use GPUs more efficiently whereas prefills do not benefit from batching.

Current LLM inference schedulers can be broadly classified into two categories [+1], namely, *prefill-prioritizing* and *decode-prioritizing* depending on how they schedule the prefill and decode phases while batching requests. In this paper, we argue that both strategies have fundamental pitfalls that make them unsuitable for serving online inference (see [Figure 2](#figure-02)).

Traditional request-level batching systems such as FasterTransformer [Fas21] employ *decode-prioritizing* scheduling. These systems submit a batch of requests to the execution engine that first computes the prefill phase of all requests and then schedules their decode phase. The batch completes only after all requests in it have finished their decode phase i.e., new prefills are not scheduled as long as one or more requests are doing decodes. This strategy optimizes inference for latency metric time-between-tokens or TBT — an important performance metric for LLMs. This is because new requests do not affect the execution of ongoing requests in their decode phase. However, *decode-prioritizing* schedulers severely compromise on throughput because even if some requests in a batch finish early, the execution continues with reduced batch size until the completion of the last request.

<span id="figure-02"></span>

![Figure 2. Current LLM serving systems involve a tradeoff between throughput and latency depending on their scheduling policy. Prioritizing prefills optimizes throughput but sacrifices TBT (time-between-tokens) tail latency whereas prioritizing decodes has the opposite effect. Sarathi-Serve serves high throughput with low TBT latency via stall-free batching. (The figure is illustrative and actual values will depend on the model and workload characteristics.)](../../papers/sarathi-serve/figure-02.png)

**Figure 2.** Current LLM serving systems involve a tradeoff between throughput and latency depending on their scheduling policy. Prioritizing prefills optimizes throughput but sacrifices TBT (time-between-tokens) tail latency whereas prioritizing decodes has the opposite effect. Sarathi-Serve serves high throughput with low TBT latency via stall-free batching. (The figure is illustrative and actual values will depend on the model and workload characteristics.)

Orca [Yu22a] introduced iteration-level batching wherein requests can dynamically enter or exit a batch at the granularity of individual iterations. Iteration-level batching improves throughput by avoiding inefficiencies of request-level batching systems. Orca and several other recent systems like vLLM [Vll23] combine iteration-level batching with *prefill-prioritizing* scheduling wherein they eagerly schedule the prefill phase of one or more requests first i.e., whenever GPU memory becomes available. This way, *prefill-prioritizing* schedulers have better throughput because computing prefills first allows subsequent decodes to operate at high batch sizes. However, prioritizing prefills leads to high latency because it interferes with ongoing decodes. Since prefills can take arbitrarily long time depending on the lengths of the given prompts, *prefill-prioritizing* schedulers lead to an undesirable phenomenon that we refer to as *generation stalls* in this paper. For example, [Figure 1(a)](#figure-01) shows that a generation stall in vLLM can last over several seconds.

Another challenge introduced by traditional iteration-level scheduling systems like Orca [Yu22a] is pipeline stalls or bubbles [Hua19]. These appear in pipeline-parallelism (PP) deployments that are needed to scale LLM inference across several nodes. In servers with high bandwidth connectivity such as NVIDIA DGX A100 [Nvi16a], tensor-parallelism (TP) [Sho19] can enable deployment of an LLM on up to 8 GPUs, supporting large batch sizes with low latencies. However, TP can have prohibitively high latencies when hyper-clusters are unavailable [Ath22]. Thus, as an alternative to TP, pipeline-parallelism (PP) [Pip19, Ath22] is typically used across commodity networks. Existing systems rely on micro-batches to mitigate pipeline stalls or bubbles [Hua19]. However, the standard micro-batch based scheduling can still lead to pipeline bubbles due to the unique characteristics of LLM inference. Specifically, LLM inference consists of a mixture of varying length prefills and decodes. The resulting schedule can thus have wildly varying runtimes across different micro-batches that waste GPU cycles and degrade the overall system throughput.

To address these challenges, we propose Sarathi-Serve, a scheduler to balance the throughput-latency tradeoff for scalable online LLM inference serving. Sarathi-Serve is based on two key ideas: *chunked-prefills* and *stall-free* scheduling. *Chunked-prefills* splits a prefill request into equal compute-sized chunks and computes a prompt’s prefill phase over multiple iterations (each with a subset of the prompt tokens). *Stall-free* scheduling allows *new requests to join a running batch without pausing ongoing decodes*. This involves constructing a batch by coalescing all the on-going decodes with one (or more) prefill chunks from new requests such that each batch reaches the pre-configured chunk size. Sarathi-Serve builds upon iteration-level batching but with an important distinction: it throttles the number of prefill tokens in each iteration while admitting new requests in a running batch. This not only bounds the latency of each iteration, but also makes it nearly independent of the total length of input prompts. This way, Sarathi-Serve minimizes the effect of computing new prefills on the TBT of ongoing decodes enabling both high throughput and low TBT latency.

In addition, hybrid batches (consisting of prefill and decode tokens) constructed by Sarathi-Serve have a near-uniform compute requirement. With pipeline-parallelism, this allows us to create balanced micro-batching based schedules that significantly reduce pipeline bubbles and improve GPU utilization, thus allowing efficient and scalable deployments.

We evaluate Sarathi-Serve across different models and hardware — Mistral-7B on a single A100, Yi-34B on 2 A100 GPUs with 2-way tensor parallelism, LLaMA2-70B on 8 A40 GPUs, and Falcon-180B with 2-way pipeline and 4-way tensor parallelism across 8 A100 GPUs connected over commodity ethernet. For Yi-34B, Sarathi-Serve improves system serving capacity by up to $3.7\times$ under different SLO targets. Similarly for Mistral-7B, we achieve up to $2.6\times$ higher serving capacity. Sarathi-Serve also reduces pipeline bubbles, resulting in up to $5.6\times$ gains in end-to-end serving capacity for Falcon-180B deployed with pipeline parallelism.

The main contributions of our paper include:

- We identify a number of pitfalls in the current LLM serving systems, particularly in the context of navigating the throughput-latency tradeoff.
- We introduce two simple-yet-effective techniques, *chunked-prefills* and *stall-free batching*, to improve the performance of an LLM serving system.
- We show generality through extensive evaluation over multiple models, hardware, and parallelism strategies demonstrating that Sarathi-Serve improves model serving capacity by up to an order of magnitude.

<span id="section-2"></span>

## 2 Background

In this section, we describe the typical LLM model architecture along with their auto-regressive inference process. We also provide an overview of the scheduling policies and important performance metrics.

<span id="section-2-1"></span>

### 2.1 The Transformer Architecture

Popular large language models, like, GPT-3 [Ope22a], LLaMA [Tou23c], Yi [Yi23] etc. are decoder-only transformer models trained on next token prediction tasks. These models consist of a stack of layers identical in structure. Each layer contains two modules — self-attention and feed-forward network (FFN).

**Self-attention module:** The self-attention module is central to the transformer architecture [Vas17], enabling each part of a sequence to consider all previous parts for generating a contextual representation. During the computation of self-attention, first the Query ($Q$), Key ($K$) and Value ($V$) vectors corresponding to each input token are obtained via a linear transformation. Next, the *attention* operator computes a semantic relationship among all tokens of a sequence. This involves computing a dot-product of each $Q$ vector with $K$ vectors of all preceding tokens of the sequence, followed by a softmax operation to obtain a weight vector, which is then used to compute a weighted average of the $V$ vectors. This attention computation can be performed across multiple *heads*, whose outputs are combined using a linear transformation.

**Feed-forward network (FFN):** FFN typically consists of two linear transformations with a non-linear activation in between. The first linear layer transforms an input token embedding of dimension $h$ to a higher dimension $h2$. This is followed by an activation function, typically ReLU or GELU [Aga19, Hen23]. Finally, the second linear layer, transforms the token embedding back to the original dimension $h$.

<span id="section-2-2"></span>

### 2.2 LLM Inference Process

**Autoregressive decoding:** LLM inference consists of two distinct phases — a *prefill* phase followed by a *decode* phase. The prefill phase processes the user’s input prompt and produces the first output token. Subsequently, the decode phase generates output tokens one at a time wherein the token generated in the previous step is passed through the model to generate the next token until a special *end-of-sequence* token is generated. Note that the decode phase requires access to all the keys and values associated with all the previously processed tokens to perform the attention operation. To avoid repeated recomputation, contemporary LLM inference systems store activations in KV-cache [Sho19, Yu22a, Fas21].

A typical LLM prompt contains 100s-1000s of input tokens [Table 2](#table-02), [Zhe23c]. During the prefill phase all these prompt tokens are processed in parallel in a single iteration. The parallel processing allows efficient utilization of GPU compute. On the contrary, the decode phase involves a full forward pass of the model over a single token generated in the previous iteration. This leads to low compute utilization making decodes memory-bound.

**Batched LLM inference in multi-tenant environment:** A production serving system must deal with concurrent requests from multiple users. Naively processing requests in a sequential manner leads to a severe under-utilization of GPU compute. In order to achieve higher GPU utilization, LLM serving systems leverage batching to process multiple requests concurrently. This is particularly effective for the decode phase processing which has lower computational intensity at low batch sizes. Higher batch sizes allows the cost of fetching model parameters to be amortized across multiple requests.

Recently, several complementary techniques have been proposed to optimize throughput by enabling support for larger batch sizes. Kwon et al. propose PagedAttention [Kwo23], which allows more requests to concurrently execute, eliminating fragmentation in *KV-cache*. The use of Multi Query Attention (MQA) [Sha19b], Group Query Attention (GQA) [Ain23a] in leading edge LLM models like LLaMA2 [Tou23c], Falcon [Alm23] and Yi [Yi23] has also significantly helped in alleviating memory bottleneck in LLM inference. For instance, LLaMA2-70B model has a $8\times$ smaller KV-cache footprint compared to LLaMA-65B.

<span id="section-2-3"></span>

### 2.3 Multi-GPU LLM Inference

With ever-increasing growth in model sizes, it becomes necessary to scale LLMs to multi-GPU or even multi-node deployments [Pop22, Usi23]. Furthermore, LLM inference throughput, specifically that of the decode phase is limited by the maximum batch size we can fit on a GPU. Inference efficiency can therefore benefit from model-parallelism which allows larger batch sizes by sharding model weights across multiple GPUs. Prior work has employed both tensor-parallelism (TP) [Sho19] and pipeline-parallelism (PP) [Yu22a, Fas21, Wu23a] for this purpose.

TP shards each layer across the participating GPUs by splitting the model weights and KV-cache equally across GPU workers. This way, TP can linearly scale per-GPU batch size. However, TP involves a high communication cost due to two all-reduce operations per layer — one in attention computation and the other in FFN [Sho19]. Moreover, since these communication operations are in the critical path, TP is preferred only within a single node where GPUs are connected via high bandwidth interconnects like NVLink.

Compared to TP, PP splits a model layer-wise, where each GPU is responsible for a subset of layers. To keep all GPUs in the ‘pipeline’ busy, *micro-batching* is employed. These micro-batches move along the pipeline from one stage to the next at each iteration. PP has much better compute-communication ratio compared to TP, as it only needs to send activations once for multiple layers of compute. Furthermore, PP requires communication only via point-to-point communication operations, compared to the more expensive allreduces in TP. Thus, PP is more efficient than TP when high-bandwidth interconnects are unavailable *e.g.,* in cross-node deployments.

<span id="section-2-4"></span>

### 2.4 Performance Metrics

There are two primary latency metrics of interest for LLM serving: TTFT (time-to-first-token) and TBT (time-between-tokens). For a given request, TTFT measures the latency of generating the first output token from the moment a request arrives in the system. This metric reflects the initial responsiveness of the model. TBT on the other hand measures the interval between the generation of consecutive output tokens of a request, and affects the overall perceived fluidity of the response. When system is under load, low throughput can lead to large scheduling delays and consequently higher TTFT.

In addition, we use a throughput metric, *Capacity*, defined as the maximum request load (queries-per-second) a system can sustain while meeting certain latency targets. Higher capacity is desirable because it reduces the cost of serving.

<span id="section-2-5"></span>

### 2.5 Scheduling Policies for LLM Inference

The scheduler is responsible for admission control and batching policy. For the ease of exposition, we investigate existing LLM inference schedulers by broadly classifying them under two categories — *prefill-prioritizing* and *decode-prioritizing*.

Conventional inference engines like FasterTransformer [Fas21], Triton Inference Server [Tri20] use *decode-prioritizing* schedules with request-level batching *i.e.,* they pick a batch of requests and execute it until *all* requests in the batch complete ([Algorithm 1](#algorithm-01)). This approach reduces the operational complexity of the scheduling framework but at the expense of inefficient resource utilization. Different requests in a batch typically have a large variation in the number of input and output tokens. Request-level schedulers pad shorter requests with zeros to match their length with the longest request in the batch which results in wasteful compute and longer wait times for pending requests [Yu22a].

<span id="algorithm-01"></span>

**Algorithm 1: Request-level batching. New requests are admitted only if there are no decodes left (line 3). This optimizes TBT but wastes GPU compute in many decode-only iterations (line 10) with potentially small batch sizes.**

- Initialize current batch $B \leftarrow \emptyset$.
- **While** True:
  - **If** $B = \emptyset$:
    - $R_{new} \leftarrow$ `get_next_request()`.
    - **While** `can_allocate_request`$(R_{new})$:
      - $B \leftarrow B + R_{new}$.
      - $R_{new} \leftarrow$ `get_next_request()`.
    - `prefill`$(B)$.
  - **Else:**
    - `decode`$(B)$.
    - $B \leftarrow$ `filter_finished_requests`$(B)$.

To avoid wasted compute of request-level batching, Orca [Yu22a] introduced a fine-grained iteration-level batching mechanism where requests can dynamically enter and exit a batch after each model iteration.([Algorithm 2](#algorithm-02)). This approach can significantly increase system throughput and is being used in many LLM inference serving systems today *e.g.,* vLLM [Vll23], TensorRT-LLM [Ten23], and LightLLM [Lig23c].

<span id="algorithm-02"></span>

**Algorithm 2: Iteration-level batching (vLLM). Prefills are executed eagerly (lines 8-9), potentially introducing a generation stall for ongoing decodes (line 12).**

- Initialize current batch $B \leftarrow \emptyset$.
- **While** True:
  - $B_{new} \leftarrow \emptyset$.
  - $R_{new} \leftarrow$ `get_next_request()`.
  - **While** `can_allocate_request`$(R_{new})$:
    - $B_{new} \leftarrow B_{new} + R_{new}$.
    - $R_{new} \leftarrow$ `get_next_request()`.
  - **If** $B_{new} \neq \emptyset$:
    - `prefill`$(B_{new})$.
    - $B \leftarrow B + B_{new}$.
  - **Else:**
    - `decode`$(B)$.
  - $B \leftarrow$ `filter_finished_requests`$(B)$.

Current iteration-level batching systems such as vLLM [Vll23] and Orca [Yu22a] use *prefill-prioritizing* schedules that eagerly admit new requests in a running batch at the first available opportunity, e.g., whenever GPU memory becomes available. Prioritizing prefills can improve throughput because it increases the batch size of subsequent decode iterations.

<span id="section-3"></span>

## 3 Motivation

<span id="figure-03"></span>

![Figure 3. Throughput of the prefill and decode phases with different batch sizes for Mistral-7B running on a single A100 GPU. We use prompt length of 1024 for both prefill and decode experiments. Note that different y-axis, showing prefills are much more efficient than decode. Further, note that *batching boosts decode throughput almost linearly but has a marginal effect on prefill throughput.*](../../papers/sarathi-serve/figure-03.png)

**Figure 3.** Throughput of the prefill and decode phases with different batch sizes for Mistral-7B running on a single A100 GPU. We use prompt length of 1024 for both prefill and decode experiments. Note that different y-axis, showing prefills are much more efficient than decode. Further, note that *batching boosts decode throughput almost linearly but has a marginal effect on prefill throughput.*

In this section, we first analyse the cost of prefill and decode operations. We then highlight the throughput-latency trade-off and pipeline bubbles that appear in serving LLMs.

<span id="section-3-1"></span>

### 3.1 Cost Analysis of Prefill and Decode

As discussed in [Section 2.2](#section-2-2), while the *prefill* phase processes all input tokens in parallel and effectively saturates GPU compute, the *decode* phase processes only a single token at a time and is very inefficient. [Figure 3](#figure-03) illustrates throughput as a function of batch size, and we can observe that while for decode iterations throughput increases roughly linearly with batch size, prefill throughput almost saturates even with a single request.

**Takeaway-1:** *The two phases of LLM inference — prefill and decode — demonstrate contrasting behaviors wherein batching boosts decode phase throughput immensely but has little effect on prefill throughput.*

<span id="figure-04"></span>

![Figure 4. Prefill and decode time with different input sizes for Mistral-7B running on single A100 GPU. Linear layers contribute to the majority of runtime in both prefill and decode phases. Due to the low arithmetic intensity in decode batches, the cost of linear operation for 1 decode token is nearly same as 128 prefill tokens.](../../papers/sarathi-serve/figure-04.png)

**Figure 4.** Prefill and decode time with different input sizes for Mistral-7B running on single A100 GPU. Linear layers contribute to the majority of runtime in both prefill and decode phases. Due to the low arithmetic intensity in decode batches, the cost of linear operation for 1 decode token is nearly same as 128 prefill tokens.

[Figure 4](#figure-04) breaks down the prefill and decode compute times into linear, attention and others, and shows their individual contributions. From the figure, we see that linear operators contribute to the majority of the runtime cost. While attention cost grows quadratically with sequence length, linear operators still contribute more than 80% to the total time even at high sequence lengths. Therefore, optimizing linear operators is important for improving LLM inference.

**Low Compute Utilization during Decodes:** Low compute utilization during the decode phase is a waste of GPU’s processing capacity. To understand this further, we analyze the arithmetic intensity of prefill and decode iterations. Since the majority of the time in LLM inference is spent in linear operators, we focus our analysis on them.

Matrix multiplication kernels overlap memory accesses along with computation of math operations. The total execution time of an operation can be approximated to $T=\max(T_{\text{math}},T_{\text{mem}})$, where $T_{\text{math}}$ and $T_{\text{mem}}$ represent the time spent on math and memory fetch operations respectively. An operation is considered memory-bound if $T_{\text{math}}<T_{\text{mem}}$. Memory-bound operations have low Model FLOPs Utilization (MFU) [Cho23a]. On the other hand, compute-bound operations have low Model Bandwidth Utilization (MBU). When $T_{\text{math}}=T_{\text{mem}}$, both compute and memory bandwidth utilization are maximized. Arithmetic intensity quantifies the number of math operations performed per byte of data fetched from the memory. At the optimal point, the arithmetic intensity of operation matches the FLOPS-to-Bandwidth ratio of the device. [Figure 5](#figure-05) shows arithmetic intensity as a function of the number of tokens in the batch for linear layers in LLaMA2-70B running on four A100 GPUs. Prefill batches amortize the cost of fetching weights of the linear operators from HBM memory to GPU cache over a large number of tokens, allowing it to have high arithmetic intensity. In contrast, decode batches have very low computation intensity. [Figure 6](#figure-06) shows the total execution time of linear operators in an iteration for LLaMA2-70B as a function of the number of tokens. Note that execution time increases only marginally in the beginning *i.e.,* as long as the batch is in a memory-bound regime, but linearly afterwards *i.e.,* when the batch becomes compute-bound. [+2]

<span id="figure-05"></span>

![Figure 5. Arithmetic intensity trend for LLaMA2-70B linear operations with different number of token running on four A100s. Decode batches have low arithmetic intensity *i.e.,* they are bottlenecked by memory fetch time, leading to low compute utilization. Prefill batches are compute bound with sub-optimal bandwidth utilization. Sarathi-Serve forms balanced batches by combining decodes and prefill chunks to maximize both compute and bandwidth utilization.](../../papers/sarathi-serve/figure-05.png)

**Figure 5.** Arithmetic intensity trend for LLaMA2-70B linear operations with different number of token running on four A100s. Decode batches have low arithmetic intensity *i.e.,* they are bottlenecked by memory fetch time, leading to low compute utilization. Prefill batches are compute bound with sub-optimal bandwidth utilization. Sarathi-Serve forms balanced batches by combining decodes and prefill chunks to maximize both compute and bandwidth utilization.

**Takeaway-2:** *Decode batches operate in memory-bound regime leaving compute underutilized. This implies that more tokens can be processed along with a decode batch without significantly increasing its latency.*

<span id="figure-06"></span>

![Figure 6. Linear layer execution time as function of number of tokens in a batch for LLaMA2-70B on A100(s) with different tensor parallel degrees. When the number of tokens is small, execution time is dictated by the cost of fetching weights from HBM memory. Hence, execution time is largely stagnant in the 128-512 tokens range, especially for higher tensor parallel degrees. Once the number of tokens in the batch cross a critical threshold, the operation become compute bound and the runtime increases linearly with number of tokens.](../../papers/sarathi-serve/figure-06.png)

**Figure 6.** Linear layer execution time as function of number of tokens in a batch for LLaMA2-70B on A100(s) with different tensor parallel degrees. When the number of tokens is small, execution time is dictated by the cost of fetching weights from HBM memory. Hence, execution time is largely stagnant in the 128-512 tokens range, especially for higher tensor parallel degrees. Once the number of tokens in the batch cross a critical threshold, the operation become compute bound and the runtime increases linearly with number of tokens.

<span id="figure-07"></span>

![Figure 7. A generation stall occurs when one or more prefills are scheduled in between consecutive decode iterations of a request. A, B, C and D represent different requests. Subscript $d$ represents a decode iteration, $p$ represents a full prefill and $p0$, $p1$ represent two chunked prefills of a given prompt. vLLM induces generation stalls by scheduling as many prefills as possible before resuming ongoing decodes. Despite supporting hybrid batches, Orca cannot mitigate generation stalls because the execution time of batches containing long prompts remains high. FasterTransformer is free of generation stalls as it finishes all ongoing decodes before scheduling a new prefill but compromises on throughput due to low decode batch size. In contrast, Sarathi-Serve generates a schedule that eliminates generation stalls yet delivers high throughput.](../../papers/sarathi-serve/figure-07.png)

**Figure 7.** A generation stall occurs when one or more prefills are scheduled in between consecutive decode iterations of a request. A, B, C and D represent different requests. Subscript $d$ represents a decode iteration, $p$ represents a full prefill and $p0$, $p1$ represent two chunked prefills of a given prompt. vLLM induces generation stalls by scheduling as many prefills as possible before resuming ongoing decodes. Despite supporting hybrid batches, Orca cannot mitigate generation stalls because the execution time of batches containing long prompts remains high. FasterTransformer is free of generation stalls as it finishes all ongoing decodes before scheduling a new prefill but compromises on throughput due to low decode batch size. In contrast, Sarathi-Serve generates a schedule that eliminates generation stalls yet delivers high throughput.

<span id="section-3-2"></span>

### 3.2 Throughput-Latency Trade-off

Iteration-level batching improves system throughput but we show that it comes at the cost of high TBT latency due to a phenomenon we call *generation stalls*.

[Figure 7](#figure-07) compares different scheduling policies. The example shows a timeline (left to right) of requests A, B, C and D. Requests A and B are in decode phase at the start of the interval and after one iteration, requests C and D enter the system. Orca and vLLM both use FCFS iteration-level batching with eager admission of prefill requests but differ in their batch composition policy. Orca supports hybrid batches composed of both prefill and decode requests whereas vLLM only supports batches that contain either all prefill or all decode requests. Irrespective of this difference, both Orca and vLLM can improve throughput by maximizing the batch size in subsequent decode iterations. However, eagerly scheduling prefills of requests C and D delays the decodes of already running requests A and B because an iteration that computes one or more prefills can take several seconds depending on the lengths of input prompts. Therefore, *prefill-prioritizing* schedulers can introduce *generation stalls* for ongoing decodes resulting in latency spikes caused by high TBT.

In contrast to iteration-level batching, request-level batching systems such as FasterTransformer [Fas21] do not schedule new requests until *all* the already running requests complete their decode phase (line 3 in [Algorithm 1](#algorithm-01)). In [Figure 7](#figure-07), the prefills for requests C and D get stalled until requests A and B both exit the system. Therefore, *decode-prioritizing* systems provide low TBT latency albeit at the cost of low system throughput. For example, Kwon et al. [Kwo23] show that iteration-level batching with PagedAttention can achieve an order of magnitude higher throughput compared to FasterTransformer.

One way to reduce latency spikes in iteration-level batching systems is to use smaller batch sizes as recommended in Orca [Yu22a]. However, lowering batch size adversely impacts throughput as shown in [Section 2.2](#section-2-2). Therefore, existing systems are forced to trade-off between throughput and latency depending on the desired SLOs.

**Takeaway-3:** *The interleaving of prefills and decodes involves a trade-off between throughput and latency for current LLM inference schedulers. State-of-the-art systems today use prefill-prioritizing schedules that trade TBT latency for high throughput.*

<span id="figure-08"></span>

![Figure 8. A 2-way pipeline parallel iteration-level schedule in Orca across 4 requests (A,B,C,D) shows the existence of pipeline bubbles due to non-uniform batch execution times. Sarathi-Serve is able to minimize these stalls by creating uniform-compute batches.](../../papers/sarathi-serve/figure-08.png)

**Figure 8.** A 2-way pipeline parallel iteration-level schedule in Orca across 4 requests (A,B,C,D) shows the existence of pipeline bubbles due to non-uniform batch execution times. Sarathi-Serve is able to minimize these stalls by creating uniform-compute batches.

<span id="section-3-3"></span>

### 3.3 Pipeline Bubbles waste GPU Cycles

Pipeline-parallelism (PP) is a popular strategy for cross-node deployment of large models, owing to its lower communication overheads compared to Tensor Parallelism (TP). A challenge with PP, however, is that it introduces *pipeline bubbles* or periods of GPU inactivity as subsequent pipeline stages have to wait for the completion of the corresponding micro-batch in the prior stages. Pipeline bubbles is a known problem in training jobs, where they arise between the forward and backward passes due to prior stages needing to wait for the backward pass to arrive. Micro-batching is a common technique used in PP training jobs to mitigate pipeline bubbles [Ath22, Pip19, Hua19].

Inference jobs only require forward computation and therefore one might expect that micro-batching can eliminate pipeline bubbles during inference. In fact, prior work on transformer inference, such as, FasterTransformer [Fas21] and FastServe [Wu23a] use micro-batches but do not mention pipeline-bubbles. Recently proposed Orca [Yu22a] also suggests that iteration-level scheduling eliminates bubbles in pipeline scheduling (see [Figure&#32;8](https://arxiv.org/pdf/2206.02672#page=11) in [Yu22a]). However, our experiments show that even with iteration-level scheduling, pipeline bubbles can waste significant GPU cycles with PP ([Section 5.3](#section-5-3)).

Each micro-batch (or iteration) in LLM inference can require a different amount of compute (and consequently has varying execution time), depending on the composition of prefill and decode tokens in the micro-batch (see [Figure 8](#figure-08)). We identify three types of bubbles during inference: (1) bubbles like $\mathrm{PB}_{1}$ that occur due to the varying number of prefill tokens in two consecutive micro-batches (2) bubbles like $\mathrm{PB}_{2}$ that occur due to different compute times of prefill and decode stages when one is followed by the other, and (3) bubbles like $\mathrm{PB}_{3}$ that occur due to difference in decode compute times between micro-batches since the attention cost depends on the accumulated context length (size of the KV-cache) and varies across requests. For Falcon-180B, a single prompt of 4k tokens takes $\approx 1150$ ms to execute compared to a decode only iteration with batch size 32 which would take about $\approx 200$ ms to execute. Interleaving of these iteration could result in a bubble of $\approx 950$ ms. These pipeline bubbles are wasted GPU cycles and directly correspond to a loss in serving throughput and increased latency. This problem is aggravated with increase in prompt lengths and batch size, due to longer and more frequent prefill iterations respectively. If we can ensure that each micro-batch performs uniform computation, we can mitigate these pipeline bubbles.

**Takeaway-4:** *There can be a large variance in compute time of LLM iterations depending on composition of prefill- and decode-tokens in the batch. This can lead to significant bubbles when using pipeline-parallelism.*

<span id="section-4"></span>

## 4 Sarathi-Serve: Design and Implementation

We now discuss the design and implementation of Sarathi-Serve — a system that provides high throughput with predictable tail latency via two key techniques — *chunked-prefills* and *stall-free batching*.

<span id="figure-09"></span>

![Figure 9. The incremental cost of coalescing prefills with decode batches. We consider two batching schemes — (i) Decode + Full Prefill represents the hybrid batching of Orca wherein the entire prefill is executed in a single iteration along with ongoing decodes. (ii) Decode + Chunked Prefill represents Sarathi-Serve wherein prefills are chunked before being coalesced with ongoing decodes with a fixed token budget. Sarathi-Serve processes prefill tokens with much lower impact on the latency of decodes. Further, the relative impact of Sarathi-Serve on latency reduces with higher decode batch size and context lengths.](../../papers/sarathi-serve/figure-09.png)

**Figure 9.** The incremental cost of coalescing prefills with decode batches. We consider two batching schemes — (i) Decode + Full Prefill represents the hybrid batching of Orca wherein the entire prefill is executed in a single iteration along with ongoing decodes. (ii) Decode + Chunked Prefill represents Sarathi-Serve wherein prefills are chunked before being coalesced with ongoing decodes with a fixed token budget. Sarathi-Serve processes prefill tokens with much lower impact on the latency of decodes. Further, the relative impact of Sarathi-Serve on latency reduces with higher decode batch size and context lengths.

<span id="section-4-1"></span>

### 4.1 Chunked-prefills

As we show in [Section 3.1](#section-3-1), decode batches are heavily memory bound with low arithmetic intensity. This slack in arithmetic intensity presents an opportunity to piggyback additional computation in decode batches. Naively, this can be done by creating hybrid batches which combine the memory bound decodes along with compute bound prefills. However, in many practical scenarios, input prompts contain several thousand tokens on average *e.g.,* [Table 2](#table-02) shows that the median prompt size in *openchat_sharegpt4* and *arxiv_summarization* datasets is 1730 and 7059 respectively. Combining these long prefills with decode iterations would lead to high TBT latency.

To tackle this challenge, we present a technique called *chunked-prefills* which allows computing large prefills in small chunks across several iterations. *Chunked-prefills* is a prefill splitting mechanism hinged on two key insights. First, as discussed in [Section 3.1](#section-3-1), a prefill request with modest sequence length can effectively saturate GPU compute. For example, in [Figure 4](#figure-04), prefill throughput starts saturating around sequence length of 512 tokens. Second, in many practical scenarios, input prompts contain several thousand tokens on average ([Table 2](#table-02)). This provides an opportunity to break large prefill requests into smaller units of compute which are still large enough to saturate GPU compute. In Sarathi-Serve, we leverage this mechanism to form batches with appropriate number of tokens such that we can utilize the compute potential in decode batches without violating the TBT SLO.

<span id="algorithm-03"></span>

**Algorithm 3: Stall-free batching with Sarathi-Serve. First the batch is filled with with ongoing decode tokens (lines 6-8) and optionally one prefill chunk from ongoing (lines 10-12). Finally, new requests are added (lines 13-20) within the token budget so as to maximize throughput with minimal latency impact on the TBT of delaying the ongoing decodes.**

- **Input:** $T_{\max}$, Application TBT SLO.
- Initialize *token_budget*, $\tau \leftarrow$ `compute_token_buget`$(T_{\max})$.
- Initialize *batch_num_tokens*, $n_t \leftarrow 0$.
- Initialize current batch $B \leftarrow \emptyset$.
- **While** True:
  - **For** $R$ in $B$:
    - **If** `is_prefill_complete`$(R)$:
      - $n_t \leftarrow n_t + 1$.
  - **For** $R$ in $B$:
    - **If** not `is_prefill_complete`$(R)$:
      - $c \leftarrow$ `get_next_chunk_size`$(R, \tau, n_t)$.
      - $n_t \leftarrow n_t + c$.
  - $R_{new} \leftarrow$ `get_next_request()`.
  - **While** `can_allocate_request`$(R_{new}) \land n_t < \tau$:
    - $c \leftarrow$ `get_next_chunk_size`$(R_{new}, \tau, n_t)$.
    - **If** $c > 0$:
      - $n_t \leftarrow n_t + c$.
      - $B \leftarrow R_{new}$.
    - **Else:**
      - **Break.**
  - `process_hybrid_batch`$(B)$.
  - $B \leftarrow$ `filter_finished_requests`$(B)$.
  - $n_t \leftarrow 0$.

<span id="section-4-2"></span>

### 4.2 Stall-free batching

The Sarathi-Serve scheduler is an iteration-level scheduler that leverages *chunked-prefills* and coalescing of prefills and decodes to improve throughput while minimizing latency.

Unlike Orca and vLLM which stall existing decodes to execute prefills, Sarathi-Serve leverages the arithmetic intensity slack in decode iterations to execute prefills without delaying the execution of decode requests in the system. We call this approach *stall-free batching* ([Algorithm 3](#algorithm-03)). Sarathi-Serve first calculates the budget of maximum number of tokens that can be executed in a batch based on user specified SLO. We describe the considerations involved in determining this token budget in depth in [Section 4.3](#section-4-3). In every scheduling iteration, we first pack all the running decodes in the next batch (lines 6-8 in [Algorithm 3](#algorithm-03)). After that, we include any partially completed prefill (lines 9-12). Only after all the running requests have been accommodated, we admit new requests (lines 13-20). When adding prefill requests to the batch, we compute the maximum chunk size that can be accommodated within the leftover token budget for that batch (lines 11, 15). By restricting the computational load in every iteration, *stall-free batching* ensures that decodes never experience a generation stall due to a co-running prefill chunk. We compare the latency for hybrid batches with and without chunked prefills in [Figure 9](#figure-09). Naive hybrid batching leads to dramatic increase of up to $28.3\times$ in the TBT latency compared to a decode-only batch. In contrast, Sarathi-Serve provides a much tighter bound on latency with chunking.

[Figure 7](#figure-07) shows the scheduling policy of Sarathi-Serve in action, for the same example used in [Section 3.2](#section-3-2). The first iteration is decode-only as there are no prefills to be computed. However, after a new request C enters the system, Sarathi-Serve first splits the prefill of C into two chunks and schedules them in subsequent iterations. At the same time, with *stall-free batching*, it coalesces the chunked prefills with ongoing decodes of A and B. This way, Sarathi-Serve stalls neither decodes nor prefills unlike existing systems, allowing Sarathi-Serve to be largely free of latency spikes in TBT without compromising throughput. Furthermore, *stall-free batching* combined with *chunked-prefills* also ensures uniform compute hybrid batches in most cases, which helps reduce bubbles when using pipeline parallelism, thereby enabling efficient and scalable deployments.

<span id="section-4-3"></span>

### 4.3 Determining Token Budget

The token budget is determined based on two competing factors — TBT SLO requirement and *chunked-prefills* overhead. From a TBT minimization point of view, a smaller token budget is preferable because iterations with fewer prefill tokens have lower latency. However, smaller token budget can result in excessive chunking of prefills resulting in overheads due to 1) lower GPU utilization and 2) repeated KV-cache access in the attention operation which we discuss below.

During the computation of *chunked-prefills*, the attention operation for every chunk of a prompt needs to access the KV-cache of *all* prior chunks of the same prompt. This results in increased memory reads from the GPU HBM even though the computational cost is unchanged. For example, if a prefill sequence is split into $N$ chunks, then the first chunk’s KV-cache is loaded $N-1$ times, the second chunk’s KV-cache is loaded $N-2$ times, and so on. However, we find that even at small chunk sizes attention prefill operation is compute bound operation. In practice, there can be small overhead associated with chunking due to fixed overheads of kernel launch, etc. We present a detailed study of the overheads of *chunked-prefills* in [Section 5.4](#section-5-4).

Thus, one needs to take into account the trade-offs between prefill overhead and decode latency while determining the token budget. This can be handled with a one-time profiling of batches with different number of tokens and setting the token budget to maximum number of tokens that can be packed in a batch without violating TBT SLO.

Another factor that influences the choice of token budget is the *tile-quantization* effect [Mat23]. GPUs compute matmuls by partitioning the given matrices into tiles and assigning them to different thread blocks for parallel computation. Here, each thread block refers to a group of GPU threads and computes the same number of arithmetic operations. Therefore, matmuls achieve maximum GPU utilization when the matrix dimensions are divisible by the tile size. Otherwise, due to *tile-quantization*, some thread blocks perform extraneous computation [Mat23]. We observe that tile-quantization can dramatically increase prefill computation time *e.g.,* in some cases, using chunk size of 257 can increase prefill time by 32% compared to that with chunk size 256.

Finally, when using pipeline parallelism the effect of token budget on pipeline bubbles should also be taken into account. Larger chunks lead to higher inter-batch runtime variations that result in pipeline bubbles which results in lower overall system throughput. On the other hand, picking a very small token budget can lead to higher overhead due to lower arithmetic intensity and other fixed overheads.

Therefore, selecting a suitable token budget is a complex decision which depends on the desired TBT SLO, parallelism configuration, and specific hardware properties. We leverage Vidur [Agr24], a LLM inference profiler and simulator to determine the token budget that maximizes system capacity under specific deployment scenario.

<span id="section-4-4"></span>

### 4.4 Implementation

We implement Sarathi-Serve on top of the open-source implementation of vLLM [Kwo23, Vll23]. We added support for paged chunk prefill using FlashAttention v2 [Dao23a] and FlashInfer [Ye24a] kernels. We use FlashAttention backend for all the evaluations in this paper due to its support for wider set of models. We also extend the base vLLM codebase to support various scheduling policies, chunked prefills, pipeline parallelism and an extensive telemetry system. We use NCCL [Ncc15] for both pipeline and tensor parallel communication. Source code for the project is available at [https://github.com/microsoft/sarathi-serve](https://github.com/microsoft/sarathi-serve).

<span id="table-01"></span>

![Table 1. Models and GPU configurations (GQA: grouped-query attention, SW: sliding window).](../../papers/sarathi-serve/table-01.png)

**Table 1.** Models and GPU configurations (GQA: grouped-query attention, SW: sliding window).

<span id="section-5"></span>

## 5 Evaluation

We evaluate Sarathi-Serve on a variety of popular models and GPU configurations (see [Table 1](#table-01)) and two datasets (see [Table 2](#table-02)). We consider vLLM and Orca as baseline because they represent the state-of-the-art for LLM inference. Our evaluation seeks to answer the following questions:

- What is the maximum load a model replica can serve under specific Service Level Objective (SLO) constraints with different inference serving systems ([Section 5.1](#section-5-1)) and how does this load vary with varying SLO constraints ([Section 5.2](#section-5-2))?
- How does Sarathi-Serve perform under various deployments such as TP and PP? ([Section 5.3](#section-5-3))
- What is the overhead of *chunked-prefills*? ([Section 5.4.1](#section-5-4-1))
- What is the effect of each of *chunked-prefills* and *stall-free batching* in isolation as opposed to using them in tandem? ([Section 5.4.2](#section-5-4-2))

<span id="table-02"></span>

![Table 2. Datasets used for evaluation.](../../papers/sarathi-serve/table-02.png)

**Table 2.** Datasets used for evaluation.

<span id="table-03"></span>

![Table 3. SLOs for different model configurations.](../../papers/sarathi-serve/table-03.png)

**Table 3.** SLOs for different model configurations.

**Models and Environment:** We evaluate Sarathi-Serve across four different models Mistral-7B [Jia23], Yi-34B [Yi23], LLaMA2-70B [Tou23c] and Falcon-180B [Alm23] — these models are among the best in their model size categories. We use two different server configurations. For all models except LLaMA2-70B we use Azure NC96ads v4 VMs, each equipped with 4 NVIDIA 80GB A100 GPUs, connected with pairwise NVLINK. The machines are connected with a 100 Gbps ethernet connection. For LLaMA2-70B, we use a server with eight pairwise connected NVIDIA 48GB A40 GPUs. We run Yi-34B in a 2-way tensor parallel configuration (TP-2), and LLaMA2-70B and Falcon-180B in a hybrid parallel configuration with four tensor parallel workers and two pipeline stages for (TP4-PP2).

**Workloads:** In order to emulate the real-world serving scenarios, we generate traces by using the request length characteristics from the *openchat_sharegpt4* [Wan23m] and *arxiv_summarization* [Coh18] datasets ([Table 2](#table-02)). The *openchat_sharegpt4* trace contains user-shared conversations with ChatGPT-4 [Cha22a]. A conversation may contain multiple rounds of interactions between the user and chatbot. Each such interaction round is performed as a separate request to the system. This multi-round nature leads to high relative variance in the prompt lengths. In contrast, *arxiv_summarization* is a collection of scientific publications and their summaries (abstracts) on arXiv.org [Arx91]. This dataset contains longer prompts and lower variance in the number of output tokens, and is representative of LLM workloads such as Microsoft M365 Copilot [Cop23] and Google Duet AI [Due23] etc. The request arrival times are generated using Poisson distribution. We filter outliers of these datasets by removing requests with total length more than 8192 and 16384 tokens, respectively.

**Metrics:** We focus on the median value for the TTFT since this metric is obtained only once per user request and on the 99th percentile (P99) for TBT values since every decode token results in a TBT latency value.

<span id="figure-10"></span>

![Figure 10. Capacity (in queries per second) of Mistral-7B and Yi-34B with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.](../../papers/sarathi-serve/figure-10.png)

**Figure 10.** Capacity (in queries per second) of Mistral-7B and Yi-34B with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.

<span id="section-5-1"></span>

### 5.1 Capacity Evaluation

<span id="figure-11"></span>

![Figure 11. Capacity of LLaMA2-70B and Falcon-180B (models with pipeline parallelism) with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.](../../papers/sarathi-serve/figure-11.png)

**Figure 11.** Capacity of LLaMA2-70B and Falcon-180B (models with pipeline parallelism) with different schedulers under strict (SLO-S) and relaxed (SLO-R) latency SLOs.

We evaluate Sarathi-Serve, Orca and vLLM on all four models and both datasets under two different latency configurations: *relaxed* and *strict*. Similar to Patel et al. [Pat23], to account for the intrinsic performance limitations of a model and hardware pair, we define the SLO on P99 TBT to be equal to $5\times$ and $25\times$ the execution time of a decode iteration for a request (with prefill length of 4k and 32 batch size) running without any prefill interference for the *strict* and *relaxed* settings, respectively. [Table 3](#table-03) shows a summary of the absolute SLO thresholds. Note that the *strict* SLO represents the latency target desired for interactive applications like chatbots. On the other hand, the *relaxed* configuration is exemplary of systems where the complete sequence of output tokens should be generated within a predictable time limit but the TBT constraints on individual tokens is not very strict. For all load experiments, we ensure that the maximum load is sustainable, i.e., the queuing delay does not blow up (we use a limit of 2 seconds on median scheduling delay).

[Figure 10](#figure-10) and [Figure 11](#figure-11) show the results of our capacity experiments. Sarathi-Serve consistently outperforms Orca and vLLM in all cases across models and workloads. Under *strict* SLO, Sarathi-Serve can sustain up to $4.0\times$ higher load compared to Orca and $3.7\times$ higher load than vLLM under *strict* SLO (Yi-34B, *openchat_sharegpt4*). For larger models using pipeline parallelism, Sarathi-Serve achieves gains of up to $6.3\times$ and $4.3\times$ compared to Orca and vLLM respectively (LLaMA2-70B, *openchat_sharegpt4*) due to few pipeline bubbles.

We observe that in most scenarios, Orca and vLLM violate the P99 TBT latency SLO before they can reach their maximum serviceable throughput. Thus, we observe relaxing the latency target leads to considerable increase in their model serving capacity. In Sarathi-Serve, one can adjust the chunk size based on the desired SLO. Therefore, we use a strict token budget and split prompts into smaller chunks when operating under *strict* latency SLO. This reduces system efficiency marginally but allows us to achieve lower tail latency. On the other hand, when the latency constraint is relaxed, we increase the token budget to allow more efficient prefills. We use token budget of 2048 and 512 for all models under the *relaxed* and *strict* settings, respectively, except for the LLaMA2-70B *relaxed* configuration where we use token budget of 1536 to reduce the impact of pipeline bubbles. The system performance can be further enhanced by dynamically varying the token budget based on workload characteristics. We leave this exploration for future work.

We further notice that vLLM significantly outperforms Orca under relaxed setting. The reason for this is two-fold. First, Orca batches prompts for multiple requests together (*max sequence length * batch size* compared to *max sequence length* in vLLM), which can lead to even higher tail latency in some cases. Second, vLLM supports a much larger batch size compared to Orca. The lower batch size in Orca is due to the lack of PagedAttention and the large activation memory footprint associated with processing batches with excessively large number of tokens.

Finally, note that the capacity of each system is higher for *openchat_sharegpt4* dataset compared to the *arxiv_summarization* dataset. This is expected because prompts in the *arxiv_summarization* datasets are much longer - 7059 vs 1730 median tokens as shown in [Table 2](#table-02). The larger prompts makes Orca and vLLM more susceptible to latency violations due to higher processing time of these longer prefills.

<span id="section-5-2"></span>

### 5.2 Throughput-Latency Tradeoff

To fully understand the throughput-latency tradeoff in LLM serving systems, we vary the P99 TBT latency SLO and observe the impact on system capacity for vLLM and Sarathi-Serve. [Figure 12](#figure-12) shows the results for Mistral-7B and Yi-34B models with five different SLO values for the *openchat_sharegpt4* dataset.

We evaluate vLLM with three different batch sizes in an attempt to navigate the latency-throughput trade-off as prescribed by Yu et al. [Yu22a]. The maximum capacity of vLLM gets capped due to generation stalls under stringent TBT SLOs. Notably, the capacity of vLLM remains largely identical for all the three batch size settings. This implies that even though PagedAttention enables large batch sizes with efficient memory management — in practical situations with latency constraints, vLLM cannot leverage the large batch size due to the steep latency-throughput tradeoff made by it’s *prefill-prioritizing* scheduler.

On the other hand, the latency-throughput tradeoff in Sarathi-Serve can be precisely controlled by varying the token budget. Sarathi-Serve achieves $3.5\times$ higher capacity compared to vLLM under strict SLO (100ms, Mistral-7B) using a small token budget of 512. For scenarios with more relaxed SLO constraints, picking a larger token budget of 2048 allows Sarathi-Serve to operate more efficiently resulting in $1.65\times$ higher capacity compared to vLLM (1s, Yi-34B).

<span id="figure-12"></span>

![Figure 12. Latency — Throughput tradeoff in vLLM and Sarathi-Serve for Mistral-7B and Yi-34B models on *openchat_sharegpt4* dataset. We evaluate vLLM with three different max batch sizes of 32, 64 and 128. For Sarathi-Serve, we consider token budget of 512 and 2048 with max batch size of 128. Sarathi-Serve delivers $3.5\times$ higher capacity under stringent SLOs for Yi-34B using *Stall-free batching*.](../../papers/sarathi-serve/figure-12.png)

**Figure 12.** Latency — Throughput tradeoff in vLLM and Sarathi-Serve for Mistral-7B and Yi-34B models on *openchat_sharegpt4* dataset. We evaluate vLLM with three different max batch sizes of 32, 64 and 128. For Sarathi-Serve, we consider token budget of 512 and 2048 with max batch size of 128. Sarathi-Serve delivers $3.5\times$ higher capacity under stringent SLOs for Yi-34B using *Stall-free batching*.

<span id="section-5-3"></span>

### 5.3 Making Pipeline Parallel Viable

We now show that Sarathi-Serve makes it feasible to efficiently serve LLM inference across commodity networks with efficient pipeline parallelism. For these experiments, we run Falcon-180B over two nodes, each with four A100 GPUs, connected over 100 Gbps Ethernet. We evaluate model capacity under three configurations: vLLM with 8-way TP, vLLM with our pipeline-parallel implementation and Sarathi-Serve with pipeline-parallel. For PP configurations, we do 4-way TP within node and 2-way PP across nodes.

[Figure 13(a)](#figure-13) shows the latency for decode-only batches for Falcon-180B with purely tensor parallel TP-8 deployment compared to a TP-4 PP-2 hybrid parallel configuration. We observe that the median latency for tensor parallelism is $\sim 2\times$ higher than pipeline parallelism. This is because TP incurs high communication overhead due to cross-node all-reduces.

[Figure 13(b)](#figure-13) shows the capacity for tensor and hybrid parallel configurations for Falcon-180B on *openchat_sharegpt4* dataset. Note that unlike the hybrid parallel configuration, TP achieves low capacity even under the *relaxed* SLO due to high latency. Even though vLLM can support a fairly high load with hybrid parallelism under *relaxed* SLO, it’s performance drops sharply under the *strict* regime due to pipeline bubbles. Sarathi-Serve on the other hand, leverages *chunked-prefills* to reduce the variation in the execution time between microbatches to avoid pipeline bubbles, resulting in a $1.48\times$ increase in capacity under *relaxed* SLOs and $3.6\times$ increase in capacity under *strict* SLOs.

<span id="figure-13"></span>

![Figure 13. TP scales poorly across nodes. (a) Median TBT for decode-only batches: cross node TP increases median TBT by more than $2\times$ compared to a 4-way TP within node and PP across nodes. (b) Capacity under strict (SLO-S) and relaxed (SLO-R) latency SLOs: Sarathi-Serve increases Falcon-180B’s serving capacity by $4.3\times$ and $3.6\times$ over vLLM’s TP-only and hybrid-parallel configurations under strict SLOs.](../../papers/sarathi-serve/figure-13.png)

**Figure 13.** TP scales poorly across nodes. (a) Median TBT for decode-only batches: cross node TP increases median TBT by more than $2\times$ compared to a 4-way TP within node and PP across nodes. (b) Capacity under strict (SLO-S) and relaxed (SLO-R) latency SLOs: Sarathi-Serve increases Falcon-180B’s serving capacity by $4.3\times$ and $3.6\times$ over vLLM’s TP-only and hybrid-parallel configurations under strict SLOs.

<span id="section-5-4"></span>

### 5.4 Ablation Study

In this subsection, we conduct an ablation study on different aspects on Sarathi-Serve. In particular, we are interested in answering the following two questions: 1) what is the effect of chunking on prefill throughput, and 2) analyzing the impact of hybrid-batching and chunking on latency. While we provide results only for a few experiments in this section, all the trends discussed below are consistent across various model-hardware combinations.

<span id="section-5-4-1"></span>

#### 5.4.1 Overhead of chunked-prefills

[Figure 14](#figure-14) shows how much overhead chunking adds in Yi-34B — on overall prefill runtime. As expected, smaller chunks introduce higher overhead as shown by the gradually decreasing bar heights in [Figure 14](#figure-14). However, even with the smallest chunk size of 512, we observe a moderate overhead of at most $\sim$25%. Whereas with the larger token budget of 2048, chunked prefills have almost negligible overhead.

<span id="figure-14"></span>

![Figure 14. Overhead of *chunked-prefills* in prefill computation for Yi-34B (TP-2) normalized to the cost of no-chunking, shown for various prompt lengths using chunk lengths of 512, 1024 and 2048.](../../papers/sarathi-serve/figure-14.png)

**Figure 14.** Overhead of *chunked-prefills* in prefill computation for Yi-34B (TP-2) normalized to the cost of no-chunking, shown for various prompt lengths using chunk lengths of 512, 1024 and 2048.

<span id="table-04"></span>

![Table 4. TTFT and TBT latency measured in seconds for *hybrid-batching* and *chunked-prefills* used in isolation as well as when they are used in tandem, evaluated over 128 requests for Yi-34B running on two A100s with a token budget of 1024. By using both *hybrid-batching* and *chunked-prefills*, Sarathi-Serve is able to lower both TTFT and TBT.](../../papers/sarathi-serve/table-04.png)

**Table 4.** TTFT and TBT latency measured in seconds for *hybrid-batching* and *chunked-prefills* used in isolation as well as when they are used in tandem, evaluated over 128 requests for Yi-34B running on two A100s with a token budget of 1024. By using both *hybrid-batching* and *chunked-prefills*, Sarathi-Serve is able to lower both TTFT and TBT.

<span id="section-5-4-2"></span>

#### 5.4.2 Impact of individual techniques

Finally, [Table 4](#table-04) shows the TTFT and TBT latency with each component of Sarathi-Serve evaluated in isolation *i.e.,* *chunked-prefills*-only, *hybrid-batching*-only (mixed batches with both prefill and decode requests) and when they are used in tandem. These results show that the two techniques work best together: *chunked-prefills*-only increases TTFT as prefill chunks are slightly inefficient whereas *hybrid-batching*-only increases TBT because long prefills can still create generation stalls. When used together, Sarathi-Serve improves performance along both dimensions.

<span id="section-6"></span>

## 6 Related Work

**Model serving systems:** Systems such as Clipper [Cra17], TensorFlow-Serving [Ols17], Clockwork [Guj20] and BatchMaker [Gao18] study various placement, caching and batching strategies for model serving. However, these systems fail to address the challenges of auto-regressive transformer inference. More recently, systems such as Orca [Yu22a], vLLM [Kwo23], FlexGen [She23], FasterTransformers [Fas21], LightSeq [Wan21a], and TurboTransformers [Fan21] propose domain-specific optimizations for transformer inference. FlexGen [She23] optimizes LLM inference for throughput in resource-constrained offline scenarios i.e., it is not suitable for online serving. FastServe [Wu23a] proposed a preemptive scheduling framework for LLM inference to minimize the job completion times. We present a detailed comparison with Orca and vLLM as they represent the state-of-the-art in LLM inference.

Another approach that has emerged recently is to disaggregate the prefill and decode phases on different replicas as proposed in SplitWise, DistServe and TetriInfer [Pat23, Zho24, Hu24b]. These solutions can entirely eliminate the interference between prefills and decodes. However, disaggregation requires migrating the KV cache of *each* request upon the completion of its prefill phase which could be challenging in the absence of high-bandwidth interconnects between different replicas. In addition, this approach also under-utilizes the GPU memory capacity of the prefill replicas i.e., only the decode replicas are responsible for storing the KV cache. On the positive side, disaggregated approaches can execute prefills with maximum efficiency (and therefore yield better TTFT) unlike chunked prefills that are somewhat slower than full prefills. We leave a quantitative comparison between Sarathi-Serve and disaggregation-based solutions for future work.

Recently, Sheng et al. [She23b] proposed modification to iteration-level batching algorithm to ensure fairness among clients in a multi-tenant environment. FastServe [Wu23a] uses a preemption based scheduling mechanism to mitigate head-of-the-line blocking. Such algorithmic optimizations are complimentary to our approach and can benefit from lower prefill-decode interference enabled by Sarathi-Serve. Another recent system, APIServe [Abh24] adopted chunked prefills from Sarathi to utilize wasted compute in decode batches for ahead-of-time prefill recomputation for multi-turn API serving.

**Improving GPU utilization for transformers:** Recent works have proposed various optimizations to improve the hardware utilization for transformers. FasterTransformer uses model-specific GPU kernel implementations. CocoNet [Jan22] and [Wan22b] aim to overlap compute with communication to improve GPU utilization: these techniques are specially useful while using a high degree of tensor-parallel for distributed models where communication time can dominate compute. Further, the cost of computing self-attention grows quadratically with sequence length and hence can become significant for long contexts. [Rab22, Dao22c, Dao23a] have proposed various techniques to minimize the memory bottlenecks of self-attention with careful tiling and work partitioning. In addition, various parallelization strategies have been explore to optimize model placement. These techniques are orthogonal to Sarathi-Serve.

**Model optimizations:** A significant body of work around model innovations has attempted to address the shortcomings of transformer-based language models or to take the next leap forward in model architectures, beyond transformers. For example, multi-query attention [Sha19b] shares the same keys and values across all the attention heads to reduce the size of the KV-cache, allowing to fit a larger batch size on the GPUs. Several recent works have also shown that the model sizes can be compressed significantly using quantization [Xia23a, Fra23, Det23a, Det22b]. Mixture-of-expert models are aimed primarily at reducing the number of model parameters that get activated in an iteration [Art22, Li23i, Hua23a]. More recently, retentive networks have been proposed as a successor to transformers [Sun23a]. In contrast, we focus on addressing the performance issues of popular transformer models from a GPU’s perspective.

<span id="section-7"></span>

## 7 Conclusion

Optimizing LLM inference for high throughput and low latency is desirable but challenging. We presented a broad characterization of existing LLM inference schedulers by dividing them into two categories — *prefill-prioritizing* and *decode-prioritizing*. In general, we argue that the former category is better at optimizing throughput whereas the latter is better at optimizing TBT latency. However, none of them is ideal when optimizing throughput and latency are both important.

To address this tradeoff, we introduce Sarathi-Serve— a system that instantiates a novel approach comprised of *chunked-prefills* and *stall-free batching*. Sarathi-Serve chunks input prompts into smaller units of work to create stall-free schedules. This way, Sarathi-Serve can add new requests in a running batch without pausing ongoing decodes. Our evaluation shows that Sarathi-Serve improves the serving capacity of Mistral-7B by up to $2.6\times$ on a single A100 GPU and up to $5.6\times$ for Falcon-180B on 8 A100 GPUs.

## Acknowledgement

We would like to thank OSDI reviewers and our shepherd for their insightful feedback. This research is partly supported by GT Cloud Hub, under the auspices of the Institute for Data Engineering and Science (IDEaS), with funding from Microsoft, and the Center for Research into Novel Compute Hierarchies (CRNCH) at Georgia Tech.

<span id="section-8"></span>

## 8 Artifact Appendix

### Abstract

Our open source artifact is available on [GitHub](https://github.com/microsoft/sarathi-serve). This repository contains our implementation of Sarathi-Serve as well as the harnesses and scripts for running and plotting the experiments described in this paper.

This repository originally started as a fork of the vLLM project. Sarathi-Serve is a lightweight high-performance research prototype and doesn’t have complete feature parity with open-source vLLM. We have only retained the most critical features and adopted the codebase for faster research iterations.

<span id="section-8-1"></span>

### 8.1 Scope

This artifact allows the readers to validate the claims made in the Sarathi-Serve paper (the figures) and provides a means to replicate the experiments described. The artifact can be used to set up the necessary environment, execute the main results, and perform microbenchmarks, thus providing a comprehensive understanding of the key claims in Sarathi-Serve.

<span id="section-8-2"></span>

### 8.2 Contents

The repository is structured as follows, the primary source code for the system is contained in directory */sarathi*. The implementations for custom CUDA kernels are within the */csrc* directory. All the scripts to reproduce the experiments are in */osdi-experiments* and finally, the trace files used for the experiments are stored in */data*.

<span id="section-8-3"></span>

### 8.3 Hosting

You can obtain our artifacts from GitHub: [GitHub](https://github.com/microsoft/sarathi-serve). The main branch of the Github repository is actively updated, but we will maintain clear and accessible instructions about our artifacts in an easily identifiable README file. All the detailed instructions and README files to reproduce the experiments in the OSDI paper are available in the branch *osdi-sarathi-serve*.

<span id="section-8-4"></span>

### 8.4 Requirements

Sarathi-Serve has been tested with CUDA 12.1 on A100 and A40 GPUs. The specific GPU SKUs on which the experiments were performed and the parallelism strategies used are clearly explained in the README corresponding to the figures in the artifact, for ease of reproducibility.

[+1]: We classify recent schedulers Splitwise [Pat23] and DistServe [Zho24] under a third category “disaggregated” and discuss them in [Section 6](#section-6).

[+2]: Theoretically, we expect the operators to become compute-bound at $\sim$200 tokens on A100 GPUs, however, in practice we observe that it happens at $\sim$500-600 tokens for higher tensor parallel dimensions due to fixed overheads.

[+author-note]: Part of this work was done during an internship at MSR India.
