---
title: 'Autellix: Serving LLM Agents as Programs'
createTime: 2026/08/18 23:45:00
permalink: /en/papers/autellix/
---

> [Michael Luo](https://dblp.org/pid/152/0092), [Xiaoxiang Shi](https://dblp.org/pid/25/3181), [Colin Cai](https://dblp.org/pid/250/0706), [Tianjun Zhang](https://dblp.org/pid/193/9376), [Justin Wong](https://dblp.org/pid/50/6336), [Yichuan Wang](https://dblp.org/pid/60/7475), [Chi Wang](https://dblp.org/pid/09/404), [Yanping Huang](https://dblp.org/pid/00/10104), [Zhifeng Chen](https://dblp.org/pid/61/5154), [Joseph E. Gonzalez](https://dblp.org/pid/61/8262), and [Ion Stoica](https://dblp.org/pid/s/IonStoica). First submitted to arXiv on 2025-02-19; current version v1. [Autellix: An Efficient Serving Engine for LLM Agents as General Programs](https://arxiv.org/abs/2502.13965v1). [Original PDF](/paper/autellix.pdf). [DOI](https://doi.org/10.48550/arXiv.2502.13965). [TeX source](https://arxiv.org/src/2502.13965v1). The original PDF remains authoritative for exact print layout and bibliography.

## Abstract

Large language model (LLM) applications are evolving beyond simple chatbots into dynamic, general-purpose agentic programs, which scale LLM calls and output tokens to help AI agents reason, explore, and solve complex tasks. However, existing LLM serving systems ignore dependencies between programs and calls, missing significant opportunities for optimization. Our analysis reveals that programs submitted to LLM serving engines experience long cumulative wait times, primarily due to head-of-line blocking at both the individual LLM request and the program.

To address this, we introduce Autellix, an LLM serving system that treats programs as first-class citizens to minimize their end-to-end latencies. Autellix intercepts LLM calls submitted by programs, enriching schedulers with program-level context. We propose two scheduling algorithms—for single-threaded and distributed programs—that preempt and prioritize LLM calls based on their programs' previously completed calls. Our evaluation demonstrates that across diverse LLMs and agentic workloads, Autellix improves throughput of programs by 4–15× at the same latency compared to state-of-the-art systems, such as vLLM.

## 1 Introduction

Large language models (LLMs) as autonomous agents enhance their problem solving capabilities by scaling their inference computation—that is, increasing the number of output tokens or LLM calls [Eva84, Kah11, Kah03, Kum24, Zou24, Mir24]. With more calls and tokens, LLMs endow agents with improved reasoning [Wei22a, Cao23, Yao23, Dee25], planning and search capabilities [Wan24, Raf24], self-reflection from prior experiences [Yao23, Yu24, Fau24], and collaboration between multiple agents [Mor23, Wan23, Sch24]. These techniques enable agents to effectively navigate external environments via tools [Sci23, Gon23, Cao23] and solve complex tasks, such as autonomously browsing the web [Nar23, Neu24, Fau24a], resolving GitHub issues [Nar24, Pre24, Oth24], and proving difficult math problems [Dee24, Ana24].

<span id="figure-01"></span>

![Execution workflows for agentic programs](../../papers/autellix/figure-01.png)

**Figure 1. Execution workflows for Agentic Programs.** Agentic programs are highly dynamic execution workflows that follow a directed acyclic graph (DAG). It consists of LLM calls from one or more LLM agents and external interrupts (i.e. tool calls, humans).

<span id="figure-02"></span>

![Gantt chart comparing scheduling policies](../../papers/autellix/figure-02.png)

**Figure 2. Gantt chart of LLM call execution on an LLM serving engine with a max batch size (BS) of 2 (Y-axis) over decoding steps (X-axis).**

The rise of inference-time techniques and agentic applications signifies a shift from static, specialized LLM applications [Con23, Qiu24] to highly dynamic, general *agentic programs* [She24, Wan23, Inc24]. More precisely, an agentic program is a dynamic execution workflow, represented by a directed acyclic graph (DAG), that consists of LLM calls from one or more agents, and external interrupts, which include tool calls (i.e. external API calls), generic code execution, or human inputs (see [Background & Related Work](#section-02)). We assume that the LLM invocation pattern of programs emerges only at runtime, making it difficult to fully know or predict the entire graph in advance.

[Figure 1](#figure-01) illustrates the highly dynamic nature of agentic programs with single and multi-threaded examples. Single-threaded programs vary in two dimensions: 1) the length of the program, which depends on the user prompt, and 2) the sequence of LLM calls and interrupts, determined by a program's control flow. For instance, both Chatbot and ReAct (Reasoning and Acting) [Cao23] agents cycle between LLM calls and interrupts (human or tool call) and terminate based on a human or LLM's decision. Multi-threaded programs generally form DAGs. Both Map-Reduce, a classic multi-threaded program, and Monte Carlo Tree Search (MCTS) vary in the number of threads that fork and merge over time, where each thread may contain different sequences of LLM calls and interrupts. In particular, MCTS is a widely used technique for search and planning for reasoning and web-based agents [Wan24, Raf24, Cap24, Has25].

Existing LLM serving engines, like vLLM [Sto23c], focus on optimizing individual LLM calls or static LLM applications [Qiu24] by improving key-value (KV) cache efficiency [Sto23c, She24], accelerating CUDA kernels [Kas24, Jin24], and better scheduling algorithms for LLM requests [Jin24, Ram24]. However, these optimizations fail to account for the program-level context, such as the dependencies between LLM calls in the same program or program-level statistics, like total execution time. As a result, these systems often suffer from suboptimal end-to-end performance for complex programs—in particular, programs' end-to-end latencies (see [Section 2](#section-03)).

[Figure 2](#figure-02) illustrates a burst of two long programs (A, B) and two short programs (C, D) submitted to an LLM serving engine with a max batch size of 2 at t=0. Each program has one or more LLM calls with varying decoding lengths. Under a program-agnostic First-Come-First-Served (FCFS) policy, the default policy for vLLM [Sto23c], long LLM calls block other calls from running, resulting in *call-level* head-of-line (HoL) blocking. Program A and B's initial, long LLM calls execute first, delaying program C and D's execution until t=3,4. Repeated cases of HoL blocking result in a total waiting time of **18** units. To address this, preemptive scheduling, such as Multi-Level Feedback Queue (MLFQ) [Jin24], reduces HoL blocking by preempting long LLM calls to let short calls execute. However, without program-level context, newer programs are repeatedly delayed by subsequent calls from older programs, incurring *program-level* HoL blocking. MLFQ successfully preempts program A and B's long calls to start executing C and D. However, MLFQ repeatedly prioritizes A and B's subsequent calls from t=6–12, which delays program D's execution. Consequently, MLFQ incurs the same wait time of **18** units as FCFS.

We present Autellix, an LLM inference system designed to run programs, not individual LLM calls. Inspired by OS schedulers for processes, our key idea is to prioritize LLM calls by the total execution time of their program's previously completed calls; LLM calls from long programs, which are unlikely to complete soon, are deprioritized, allowing shorter programs to complete first. In [Figure 2](#figure-02), short programs C and D are no longer blocked by subsequent LLM calls from long programs A and B, effectively eliminating HoL blocking and reducing the total wait time to **12** units.

Autellix introduces a novel framework that leverages global, program-level statistics, such as program's cumulative execution time on an engine, to minimize waiting times and improve engine throughput. We propose two non-clairvoyant scheduling algorithms that assume no prior workload knowledge of programs: *PLAS* (Program-Level Attained Service) for single-threaded programs and *ATLAS* (Adaptive Thread-Level Attained Service) for multi-threaded programs represented as general, dynamic DAGs. *PLAS* prioritizes LLM calls based on the current cumulative service, or execution times, of their source program. Generalizing *PLAS*, *ATLAS* prioritizes LLM calls based on the maximum cumulative service time across all threads in the same program, which sorts calls based on their program's critical path [Zha20]. Beyond reduced wait times, *ATLAS* decreases program's makespans by prioritizing critical LLM calls that would otherwise block programs' progress (see [Autellix Design](#section-05)).

Programs comprised of tens to hundreds of LLM calls impose significant demands to the serving systems with a single LLM engine capable of handling only 0.2 programs per second for MCTS (see [Evaluation](#section-06)). Hence, Autellix also routes programs' LLM calls across multiple engines. For agentic workloads, our key observation is that LLM calls within a program often share common prefixes and cumulative conversation states, while calls across programs typically share only the system prompt [Zha24a]. To avoid recomputing the programs' KV-cache, Autellix respects a program's data locality by routing long calls to their programs' engines, while load-balancing shorter calls to other engines, where system prompts make up most of the input for shorter calls.

We implement a system prototype of Autellix as a layer on top of LLM serving engines, such as vLLM [Sto23c], and expose a stateful API that allows users to establish persistent sessions with Autellix, unlike traditional stateless APIs [Ope24a]. We evaluate Autellix across different LLMs and four representative agentic workloads (see [Evaluation](#section-06)). Our results show that Autellix improves throughput by 4–15× compared to state-of-the-art inference systems like vLLM [Sto23c]. Across engines, Autellix improves throughput by up to 1.5× over standard load-balancers.

In summary, the primary contributions of this paper are:

- This work is the first to formalize agentic programs as dynamic, non-deterministic DAGs of LLM calls and interrupts.
- Autellix utilizes program-level statistics to better inform its scheduler. Autellix's non-clairvoyant scheduler requires only the cumulative service times of LLM calls within the same program.
- Autellix leverages a simple load-balancing policy across multiple engines to balance data locality and KV-cache recomputation.
- Our system is easily deployable, seamlessly integrates a stateful API with existing programming and agent frameworks, and demonstrates significant throughput gains.

<span id="section-02"></span>

## 2 Background & Related Work

To detail relevant context for Autellix, we provide a brief overview of the emergent AI agent infrastructure and its applications, split between the LLM serving layer and higher-level agentic layer, as depicted in [Figure 3](#figure-03).

### LLM Serving Layer

**LLM Inference Process.** Large language models (LLMs), which drive chatbots and AI applications, predominantly utilize the Transformer architecture [Pol23], including decoder-only models such as GPT, Claude, and LLaMA [Oth20, Oth23, Oth23a, Say23]. For each request, LLM inference operates in two stages: the *prefill* phase, which converts the input prompt into intermediate token states, and the *decoding* phase, where new tokens are generated auto-regressively, one at a time, based on prior token sequences. To reduce computation, LLM serving systems leverage *KV-cache*, which stores intermediate token states to accelerate token generation [Chu22, Sto23c].

**LLM Serving.** LLM serving systems manage both the routing of LLM calls across engines and the execution of LLM calls within each engine ([Figure 3](#figure-03)). Within an engine, recent innovations in LLM serving mirror concepts rooted in traditional operating systems (OS), such as memory management, kernel optimization, and scheduling [Zha24, Lin24]. Existing solutions, such as vLLM, integrate virtual memory and paging techniques to reduce KV-cache fragmentation [Sto23c], introduce shared memory to cache prefixes across LLM requests [Qiu24, She24], and manage cache hierarchies between GPU, CPU, and disk [Kas24, Zha23, Sto24]. Other techniques improve GPU kernel implementations to accelerate self-attention [R22], pipeline different operators [Kas24], and implement better tensor or pipeline parallelism [Jin24, Sto23d]. Finally, LLM engines can leverage better scheduling, such as binpacking prefills and decodes together [Ram24] and preempting LLM requests [Jin24], to improve response times. Across multiple LLM engines, serving systems employ load-balancing techniques like live migration [Lin24a], disaggregate prefills and decodes [Xu24], construct prefix trees [Zha24a], and migrate KV caches across engines [Qiu24] to meet request SLOs and improve tail latencies. Overall, the above techniques optimize for *independent LLM requests*, equivalent to a function-call in a general program. Instead, Autellix focuses on program-level optimizations, particularly scheduling—akin to how traditional OSs manage entire processes across CPU cores.

<span id="figure-03"></span>

![AI agent infrastructure](../../papers/autellix/figure-03.png)

**Figure 3. AI Agent Infrastructure.** Top: Developers and users build and execute agentic programs that orchestrate execution and persist global, cumulative history across agents, tools, and humans. Bottom: LLM serving systems process agents' LLM calls and route calls across one or more LLM engines.

### Agentic Layer

**Agentic Programs.** Above the LLM inference layer, developers build sophisticated *agentic programs* to orchestrate interactions between agents, tools, and humans ([Figure 3](#figure-03)). Specifically, this work focuses on LLM agents, defined as a tuple consisting of a system prompt specifying the agent's role and the LLM model class [+2]. Similar to traditional OS processes and interrupts, agentic programs either interact directly with the LLM serving layer via LLM calls or engage in external interrupts—time spent outside an LLM engine. Specifically, agents can interact with tools to execute generic functions or external APIs, enabling control over environments such as databases, robotic systems, or the internet [Sci23, Gon23, Nar23, Bai24, Sha22, Li24]. Most importantly, agentic orchestration frameworks [Ope24, Con23, Inc24, Wan23], such as LangChain and Autogen, provide developers with primitives to manage a program's control flow, determining when to execute agents, invoke tools, or request human input. Such primitives adhere to general programming semantics, including conditional statements, loops, error handling, and terminal conditions [She24, Inc24, Wan23, Kha23]. Finally, programs maintain a global history of outputs across agents, tools, and humans [Qiu24, Inc24, Gon24, Cui24]. For instance, LLM-based chatbots accumulate messages between LLM agents' outputs and humans' inputs [Ope23]. Importantly, Autellix does not modify the program layer. Instead, it dynamically builds an internal state of the program's execution graph (DAG) when the program runs, which is stored in a process table (see [Implementation](#section-07)).

<span id="figure-04"></span>

![LLM calls in steady state](../../papers/autellix/figure-04.png)

**Figure 4. Number of LLM calls in serving engine during steady state over 1 hour.** Optimizing programs' wait times increases the volume of LLM calls at steady state.

**Agentic Applications.** Beyond standard chatbots ([Figure 1](#figure-01)), agentic applications, or instantiations of programs, automate or assist with complex tasks, including web or user-interface (UI) navigation (e.g. OpenAI's Operator) [Has25, Neu24, Fau24a, Bai24], resolving Github issues [Nar24, Pre24, Oth24], solving IMO-level problems [Dee24, Ana24], fact-checking and summarizing claims from multiple sources ([Figure 1](#figure-01)) [Gon24, Qiu24], and enabling precise robotic control [Car24]. Many applications scale inference time compute—the number of LLM calls and, correspondingly, total decode tokens—to improve their performance on complex tasks. These test-time methods include: step-by-step reasoning to decompose tasks [Wei22a, Lew23], explicit thought injection to guide reasoning [Cao23], planning or searching to explore possible solutions [Nar23, Bes24, Wan24], self-critique to evaluate actions [Sto23e, Sto24b], self-reflection to learn from failures [Yao23, Fau24], and multi-agent collaboration [Wan23, Mor23a]. In particular, a single-threaded Reasoning and Acting (ReAct) agent, which combines chain-of-thought (CoT) techniques to efficiently act in an environment ([Figure 1](#figure-01)), has recently been integrated on top of Deepseek-style (or o1-style) LLMs to enable automatic reasoning and tool calling [Dee25, Oth24, Li25]. A multi-threaded program, Monte Carlo Tree Search (MCTS) [Wan24], integrates parallel planning, self-critique, self-reflection, and multi-agent collaboration ([Figure 1](#figure-01)). Beyond MCTS, distributed programs may also incorporate best-of-N sampling, beam search, lookahead techniques, and genetic algorithms to explore and discover optimal solutions [Fau24b, Kum24, Dav24, Che25]. Given the probabilistic nature of LLMs, the breadth of inference-time techniques indicates that agentic programs and their applications exhibit three properties: (1) *dynamic*, as different user prompts over the same program can yield entirely different execution patterns, (2) *non-deterministic*, since the future is unknown, such as when a program decides to terminate, and (3) *distributed*, with many programs leveraging parallel calls. Hence, Autellix is non-clairvoyant, operating with zero prior knowledge of programs' workloads or execution graphs.

<span id="figure-05"></span>

![Program execution and wait times](../../papers/autellix/figure-05.png)

**Figure 5. Program execution and wait times, over different programs and system loads.** With moderate loads, programs spend the most time waiting. The duration of waiting depends on the workload.

<span id="section-03"></span>

## 3 Motivation

Today's AI agent infrastructure decouples LLM serving systems from agentic programs ([Section 2](#section-02)). As organizations shift from serving LLM queries to higher-level AI applications, LLM engines must optimize for program-level objectives, such as response times, or end-to-end latencies [Qiu24]. Formally, a single-threaded program's end-to-end latency comprises three components: (1) *waiting time*, the total queuing time of a program's LLM calls on the engine; (2) *execution time*, the cumulative feedforward time of LLM calls; and (3) *interceptions*, time spent waiting for external interrupts such as tool calls or human input. Since component (3) is unrelated to LLM serving, this section identifies problems and opportunities to reduce waiting and execution times, subsequently addressed in the design of Autellix's scheduling policies ([Autellix Design](#section-05)).

### Program-level Wait Times

<span id="figure-06"></span>

![Head-of-line blocking ratios](../../papers/autellix/figure-06.png)

**Figure 6. Ratio of Waiting to Execution Time for LLM Calls and Programs.** Head-of-line blocking occurs when short LLM calls and programs wait significantly longer than their execution times.

[Figure 6](#figure-06) shows that across various agentic workloads—from classic chatbots to ReAct and MCTS programs—the majority of a program's time is spent waiting as load increases. Hence, Autellix prioritizes reducing wait times, which not only improves program's latencies, but also increases LLM engine throughput. Faster call completions prompt programs to issue subsequent calls more quickly, increasing the arrival rate of LLM calls. [Figure 4](#figure-04) illustrates steady-state behavior over a one-hour trace using LLaMA-3.1-8B [Oth24a] on a single A100-80GB GPU for entire chatbot conversations [Src23]. Compared to vLLM's first-come, first-served (FCFS) policy, Autellix consistently handles 10 additional concurrent LLM calls, offering more batching opportunities to improve throughput.

**Call-level Blocking.** The first challenge is LLM call-level *head-of-line (HoL) blocking*. LLM calls with long decodes delay shorter ones, causing significant wait times [Jin24]. This issue is evident in serving engines like vLLM [Sto23c], which wait for ongoing calls to finish decoding before scheduling new ones. HoL blocking is severe in our evaluated workloads with long-tailed distributions of decoding steps ([Figure 11](#figure-11)).

To measure blocking, [Figure 6](#figure-06) measures the ratio of LLM requests' waiting time to execution time for Chatbot and MCTS workloads, as a function of output tokens. For FCFS policy, HoL blocking increases wait times for short LLM calls, increasing the ratio. Preemption, similar to how operating system schedulers interrupt long-running processes, mitigates HoL blocking by favoring shorter LLM calls. [Figure 6](#figure-06) shows that Multi-Level Feedback Queue (MLFQ), a preemptive algorithm, leads to smaller ratios for short decodes. However, preemption without program-level statistics may not fully resolve the issue, as explained next.

**Program-level Blocking.** The second challenge is *program-level HoL blocking*, where longer programs with many LLM requests delay shorter programs. Existing LLM schedulers are program-agnostic; they schedule individual LLM requests without considering their positions within the overall program, leading to suboptimal decisions. Our evaluation shows a long-tailed distribution of LLM calls per program, which increases program-level blocking ([Section 6](#section-06)).

To quantify program-level blocking, [Figure 6](#figure-06) measures the ratio of programs' waiting time to execution time, with respect to number of LLM calls. For both workloads, FCFS and MLFQ incur higher ratios when the number of LLM calls is small, suggesting that short programs wait a long time. Due to this, preemptive scheduling policies, like MLFQ, may perform close to, or even worse, than FCFS ([Section 6](#section-06)). Without program-level context, MLFQ blindly prioritizes new LLM requests, leading to starvation of shorter programs when long programs' new LLM calls are prioritized.

### Program-level Execution Times

<span id="figure-07"></span>

![Prefix cache hit rates](../../papers/autellix/figure-07.png)

**Figure 7. Prefix cache hit rate as a function of input length.**

A program's execution time largely depends on how efficiently the LLM engine manages the prefill and decoding phases. In agentic workloads, which often feature long, cumulative prefills, Autellix focuses on optimizing prefill performance. Specifically, significant portions of prefill computation can be eliminated through prefix caching. This technique stores and reuses relevant key-value (KV) cache entries—such as the system prompt—across LLM requests [Qiu24, She24].

**Data Locality.** [Figure 7](#figure-07) illustrates the average cache-hit rate as a function of input length. The cache-hit rate is defined as the percentage of precomputed input tokens in the LLM engine's KV cache for an incoming LLM call. Notably, within a single program, cache-hit rates remain above 90% across all input lengths, indicating that LLM calls within the same program share identical contexts. In contrast, when considering different programs, the cache-hit rate decays exponentially with input length, suggesting that programs only share the system prompt. These results suggest that LLM serving systems across engines should consider a program's *data locality*, as much of its KV cache can be reused for future LLM requests.

<span id="section-05"></span>

## 4 Autellix Design

We present Autellix's overall architecture ([Section 4.1](#section-04-01)) and then explore its two key components: (1) a program-aware scheduler ([Section 4.2](#section-04-02)) designed to reduce both call-level and program-level blocking, and (2) a data locality-aware load balancer ([Section 4.3](#section-04-03)).

<span id="section-04-01"></span>

### 4.1 Overview

<span id="figure-08"></span>

![Autellix system architecture](../../papers/autellix/figure-08.png)

**Figure 8. Autellix's system architecture.** Users run their programs locally, which initiates a stateful session and submits LLM calls to Autellix's backend. Autellix leverages a global process table to track sessions and better inform its custom load-balancer and scheduler.

Autellix is a higher-level serving engine designed for agentic programs rather than individual LLM requests. Autellix focuses on three primary objectives: (1) improving overall program's end-to-end latency, for users, (2) maximizing GPU utilization for providers, and (3) mitigating program starvation to improve fairness, measured via 95th and 99th percentile latencies.

**Assumptions.** Autellix is non-clairvoyant; it assumes no knowledge of program arrivals, the structure of executed workflows, or general workload distributions. When a program arrives, its execution DAG is initially unknown; Autellix dynamically constructs an internal representation (IR) as the program runs. This flexibility enables Autellix to generalize to any program that invokes LLM calls on the underlying engine. While prior work [Qiu24] submits static LLM applications to the engine, Autellix assumes users run general Python programs on their local machines, which invoke Autellix's backend ([Implementation](#section-07)).

**Architecture.** [Figure 8](#figure-08) illustrates Autellix's overall architecture. Unlike existing LLM engines, which assumes LLM calls are stateless, Autellix is stateful: programs execute from the user's local machine, establish a session with Autellix, and issue LLM calls over time with an associated session ID. We further detail the low-level implementation in [Section 5](#section-07). When a session starts, Autellix adds a corresponding entry to a global process table ([Section 4.2](#section-04-02)). This table tracks program metadata, including total service time, thread-level metadata, and waiting times across programs' LLM calls. Both the engine-level scheduler ([Section 4.2](#section-04-02)) and stateful load balancer ([Section 4.3](#section-04-03)) leverage the table to schedule LLM calls for the next decoding batch and route LLM calls to an engine based on their program's data locality.

<span id="section-04-02"></span>

### 4.2 Program-Aware Scheduler

**Algorithm 1. Autellix's Program-Aware Scheduler**

- **Procedure** `Update_Process_Table(Call c, Table pt)`
  - `pd = pt[c.pid]`
  - Comment: Total service time (PLAS), max critical path (ATLAS).
  - `pd.service = max(pd.service, c.service + c.model_time)`
  - Comment: Update other metrics...
  - `...`
- **Procedure** `Scheduler(Queues Q_1, ..., Q_K, Table pt)`
  - **For** each `c` in `C_arrived` (arriving LLM calls):
    - Comment: Fetch priority with program ID.
    - `c.service = pt[c.pid].service`
    - `c.q_idx = i` such that `Q_i^low <= c.service <= Q_i^hi`
    - Append `c` to `Q_c.q_idx`; set `c.quanta = Q_c.q_idx.quanta`.
  - **For** each `c` in `{Q_1, Q_2, ..., Q_K}`:
    - **If** `c.finished()`: update the process table and remove `c` from its queue.
    - **If** `c.quanta <= 0`: remove `c` from its queue, append it to `Q_(c.q_idx+1)`, increment `c.q_idx`, and reset its quantum.
    - Set `wait = pt[c.pid].wait + c.wait` and `service = pt[c.pid].service + c.model_time`.
    - **If** `wait / service >= beta` (anti-starvation): remove `c` from its queue, append it to `Q_1`, and reset `c.wait` and `c.model_time` to zero.
  - Set `B_out = []` (schedule the next batch of LLM calls).
  - **For** each `c` in `{Q_1, Q_2, ..., Q_K}`:
    - **If** `engine.can_fit(c)`: append `c` to `B_out`.
    - **Else** break.

We present a general, efficient scheduler designed to minimize programs' response times, or end-to-end latencies, without a-priori knowledge. To mitigate head-of-line blocking at both the program and call levels, Autellix assigns priorities to calls based on program-level statistics (e.g., total accumulated runtime, [Section 4.2.1](#section-04-02-01)) and dynamically preempts calls ([Section 4.2.2](#section-04-02-02)). The complete scheduling algorithm is shown in [Algorithm 1](#section-04-02).

<span id="section-04-02-01"></span>

#### 4.2.1 Program-level Prioritization

To implement program-level prioritization effectively, Autellix relies on a global process table that tracks essential program metrics, enabling more informed scheduling decisions across both single- and multi-threaded programs.

**Process Table.** Inspired by traditional operating systems, Autellix maintains a global process table that records the state of all running programs. When a new program arrives, Autellix adds a corresponding entry; when the program completes, this entry is removed. Each program entry in the process table tracks the following metrics:

- *Service time:* For single-threaded programs, this is the cumulative execution time of all completed calls on the LLM engine's model executor. For multi-threaded programs, it is the longest observed critical path's execution time.
- *Waiting time:* The time spent in the LLM engine's scheduler queue—used for anti-starvation.
- *Engine ID(s):* The engine(s) that the program is currently running on—used for Autellix's load-balancer.
- *Threads Metadata:* Each thread corresponds to an active LLM call. Hence, we keep track of a program's active LLM calls and their individual arrival, waiting, and service times.
- *Most recent call arrival:* The last time a new LLM call arrived for this program—used for tracking stale programs.
- *Most recent call completion:* The last time an LLM call finished—used for detecting long external interrupts.

When a program's LLM call completes, the table is updated accordingly. With the process table, the scheduler can reason about the global state of each program to schedule LLM calls.

**Single-Threaded Programs.**

<span id="figure-09"></span>

![Critical path for multi-threaded programs](../../papers/autellix/figure-09.png)

**Figure 9. Critical path for multi-threaded programs.** (Left) Example of a critical path through a DAG. (Right) Best-case scenario makespan, 14 units, versus worst-case makespan, 11 units.

Scheduling policies like Shortest-Job-First (SJF) and Shortest-Remaining-Processing-Time (SRPT) minimize response times optimally in single- and multi-server settings [Har18, Akh15]. However, these require exact knowledge of program runtimes, violating Autellix's non-clairvoyance assumption. Instead, the Least-Attained-Service (LAS) algorithm [Wie08], widely used in information-agnostic settings such as data center networking [Wan15, Cho15] and deep learning clusters [Gu19], offers a practical alternative.

We introduce *Program-Level Attained Service*, or *PLAS*, extending LAS to programs. For a single-threaded program, its service time is the total runtime of all prior completed LLM calls. Formally, if the jth LLM call `c_j` with program ID of `c_j.id` is submitted, *PLAS* assigns a priority `p(c_j)` to `c_j` based on the sum of all runtimes, `t_k`, of all prior LLM calls with the same ID:

$$
p(c_j) = \sum_{\substack{k < j \\ c_k.id = c_j.id}} t_k
$$

Here, large priority values mean lower priority. To reduce computation, the scheduler reads the program's total service time from the process table. When an LLM call completes, its program's total service time is updated. Thus, *PLAS* naturally favors calls from programs that have received less total service, helping shorter programs finish earlier and reducing response times.

**Multi-Threaded Programs.** Unlike single-threaded programs, multi-threaded programs are modeled as dynamic DAGs of LLM calls. Unfortunately, a program's completion time is dictated by the DAG's *critical path*—the longest sequence of dependent calls from start to finish, illustrated in [Figure 9](#figure-09). No matter how many parallel LLM calls an engine can process, the program only terminates when all calls along the critical path have finished. Furthermore, without considering critical paths, schedulers achieve sub-optimal completion times for programs; in [Figure 9](#figure-09), the DAG's makespan increases from 11 to 14 units.

To address this, we introduce *Adaptive Thread-Level Attained Service (ATLAS)*, a pragmatic generalization of *PLAS*, that prioritizes calls based on their service times along their programs' critical paths. *ATLAS* aims to assign each newly arrived call `c_j` a priority `p(c_j)` based on the priorities and completed service times of its parents `P(c_j)` in the same program:

$$
p(c_j) = \begin{cases} 0 & \mathrm{if}\ c_j\ \mathrm{is\ root} \\
\max_{c_k \in \mathcal{P}(c_j)} \{p(c_k) + t_k\} & \mathrm{otherwise} \end{cases}
$$

Here, `t_k` is the execution time of a parent call `c_k`. By recursively combining parent priorities and runtimes, `p(c_j)` estimates the longest chain of accumulated service time leading to `c_j`, providing a non-clairvoyant estimation of the critical path.

However, achieving both objectives—favoring short programs while also prioritizing the longest, critical-path threads—is nontrivial. To solve this, *ATLAS* maintains a single scalar per program in its process table: the longest observed critical path. Each active LLM call in a program inherits this value as its initial priority, and upon call completion, updates the scalar only if its own critical path is longer. This simple mechanism continuously refines the program's critical path estimate without tracking dependencies between LLM calls. Consequently, *ATLAS* favors programs and LLM calls with shorter critical paths, effectively approximating a Least-Attained-Service policy for dynamic DAGs. Furthermore, as all calls of a given program derive their priorities from the same entry, the scheduler naturally groups a program's parallel calls, preventing straggler threads from delaying programs' completion.

<span id="section-04-02-02"></span>

#### 4.2.2 Preemptive Scheduling

<span id="figure-10"></span>

![LLM call lifecycle based on discretized prioritization](../../papers/autellix/figure-10.png)

**Figure 10. LLM call lifecycle based on discretized prioritization.**

Autellix assigns priorities to each LLM call based on their program's history. However, scheduling and preempting programs based on continuous priorities can degrade into worst-case round-robin scheduling [Cho15], which performs worse than FCFS, and incur unnecessary context switches, including frequent KV-cache swaps between CPU and GPU [Jin24]. To avoid this, Autellix discretizes priorities into a finite set of queues, akin to multi-level feedback queues (MLFQ) in operating systems [Cho15, Gu19, Arp18].

**Multi-level Program-based Scheduling.** Autellix bins and discretizes LLM calls' priorities into K queues (`Q_1, Q_2, ..., Q_K`), where priorities decrease from `Q_1` to `Q_K`. Each queue `Q_i` covers a priority range `[Q_i^lo, Q_i^hi)`, with `Q_1^lo = 0`, `Q_K^hi = infinity`, and `Q_(i+1)^lo = Q_i^hi`.

In [Figure 10](#figure-10), when an LLM call arrives, Autellix looks up its program's priority `p(c)`, based on the process table. Unlike traditional MLFQ, where new calls all start at the highest priority queue `Q_1`, LLM calls are assigned to the ith queue based on discretized priorities, `p(c) in [Q_i^lo, Q_i^hi)`. Subsequently, calls receive the queue's time quantum and execute in FCFS order within their queue. Once a call exhausts its quantum, it is demoted to a lower priority queue. If the call waits too long, Autellix employs anti-starvation mechanisms, described next. Finally, when a call completes decoding, it updates the process table.

**Anti-Starvation.** Discrete prioritization, or MLFQ-style algorithms, incurs the starvation of long, low-priority programs [Gu19, Jin24, Cho15]. Simple anti-starvation techniques—such as promoting calls that have waited past a threshold—reduces Autellix to naive MLFQ, where long program's LLM calls, which are now in `Q_1`, interrupt short programs [Jin24, Arp18]. Hence, we also utilize the process table to measure program-level starvation. Concretely, for a program `p`, Autellix promotes call `c` to `Q_1` if the ratio of total waiting time (`W_total = W_p + W_c`) to service time (`T_total = T_p + T_c`) exceeds a threshold `beta`:

$$\frac{W_{\mathrm{total}}}{T_{\mathrm{total}}} \geq \beta$$

Varying `beta` presents a trade off between programs' average response times and fairness. After promotion, only `W_c` and `T_c`, or the calls' wait and run time, are set to zero, to ensure programs' threads, or active LLM calls, are likely all promoted together.

**Memory Management.** With preemptive scheduling, LLM engines must handle a large volume of concurrent LLM calls, leading to frequent GPU-CPU transfers as KV-cache blocks are repeatedly swapped to serve different requests [Jin24]. Prior work mitigates this swapping overhead by proactively swapping KV-cache for the next iteration of LLM requests while processing the current ones [Jin24]. However, Autellix is synchronous and requires real-time updates for each call's time quantum and the process table. Instead, Autellix employs two key optimizations to reduce both the frequency and overhead of GPU-CPU swapping respectively.

First, Autellix reduces total swaps by adopting multi-step scheduling, running the scheduler once every N decoding steps rather than at every step. As some requests may complete early, our scheduler overprovisions queued requests already on the GPU, ensuring that new requests are immediately added when some requests finish before N steps. Second, Autellix employs a more efficient GPU-CPU swap kernel. Instead of calling separate asynchronous transfers for each block, our kernel gathers all KV blocks into a contiguous buffer and transfers them in one operation—increasing PCIe bandwidth by reducing fragmentation, reducing per-block overhead, and lowering end-to-end swap latency ([Implementation](#section-07)).

<span id="section-04-03"></span>

### 4.3 Load Balancer

**Algorithm 2. Autellix's Load Balancer**

- **Procedure** `Load_Balancer(Call c, Table pt, List Engines)`
  - **If** `len(c.tokens) <= 2048` (small request), set `assigned_engine = Least_Used(Engines)`.
  - **Else**:
    - **If** `c.pid in pt` (program already assigned to engine), set `assigned_engine = pt[c.pid]`.
    - **Else**, select the least utilized engine, set `assigned_engine = Least_Used(Engines)`, and set `pt[c.pid] = assigned_engine`.
  - **Return** `assigned_engine`.
- **Procedure** `Least_Used(List Engines)`
  - Comment: Query engine workloads in parallel.
  - Set `workloads = Query_Engine_Workloads(Engines)`.
  - Set `least_used_engine = Argmin(workloads)`.
  - **Return** `least_used_engine`.

As agentic workloads scale, deploying multiple engine replicas is necessary. However, distributing requests without considering data locality yields suboptimal performance [Zha24a].

Our analysis for agentic workloads ([Section 3](#section-03)) highlights a critical distinction between short and long requests. Short requests below 2048 tokens achieve high cache hit rates (≥75%) across any engine, due to common system prompts. Enforcing data locality for these requests offers negligible gains and risks skewing engine utilization when large, parallel programs dominate specific engines. Thus, simply balancing short requests across the least-loaded engines preserves performance with minimal overhead. Conversely, longer requests are far more sensitive to their programs' data locality. Their substantial prefix overlap with a given program significantly reduces recomputation when consistently routed to the same engine, justifying occasional queuing delays.

While prior work relies on complex prefix trees to quantify data locality [Zha24a], our simple method dynamically routes short requests to the least-loaded engine and pins longer requests to their programs' corresponding engines. [Algorithm 2](#section-04-03) formalizes this approach, and our evaluation shows that Autellix's load balancer improves both throughput and latency across heterogeneous workloads ([Section 6](#section-06)).

<span id="section-07"></span>

## 5 Implementation

Autellix is a multi-engine LLM inference serving system comprising a frontend, scheduler, and load balancer—totaling 5k lines of Python and CUDA/C++ code.

**Frontend.** Autellix's frontend extends OpenAI's Chat Completion and vLLM's Python APIs [Sto23c, Ope25a] to provide a stateful interface that appears stateless to developers. Users simply import Autellix's library into their Python applications, and upon program initialization, Autellix automatically issues a `start_session` request to the backend. This operation returns a unique session identifier and creates a corresponding entry in the process table. Subsequent LLM calls are transparently annotated with the appropriate session, program, and thread IDs before being dispatched to the backend. When the program completes or encounters an error, Autellix invokes `end_session`, removing the associated entry from the process table. As a research prototype, the current frontend lacks safeguards against user modification of the underlying package; addressing this limitation remains future work.

**LLM Engine.** Autellix builds on vLLM v0.6.1 [Sto23c]. To keep changes localized, we modify only the scheduler by integrating new policies (*PLAS*, *ATLAS*, and MLFQ) and memory swapping kernels for efficiency. This ensures straightforward experimentation and clear attribution of performance gains. The scheduler follows the algorithm described in the previous section ([Autellix Design](#section-05)). We've also noticed in vLLM, each Key-Value (KV) block is transferred individually via `cudaMemcpyAsync`, creating small fragmented transfers that underutilize PCIe bandwidth and incur high overhead such as repeated DMA setups. To address this, we allocate a host buffer and consolidate all KV blocks into a single contiguous chunk, enabling one bulk transfer. The results are shown in the next section ([Evaluation](#section-06)).

**Multi-engine.** vLLM currently lacks the ability to manage multiple LLM engines at the same time. To better evaluate our load balancing strategy, we built `AsyncMultiLLMEngine` atop of `AsyncLLMEngine`. Each LLM engine replica runs in a dedicated Python process, and a coordinating meta-engine manages these replicas via standard inter-process communication (IPC) primitives such as `mp.Queue` and `mp.Pipe`. When the meta-engine receives a request, it assigns the request to the appropriate replica, returning a future-like object to the frontend without blocking. The selected engine process executes the task asynchronously and sends the completed result back through the IPC channel. Upon receiving the result, the meta-engine resolves the future and provides the output to the frontend. This design allows multiple requests to be processed in parallel, with the meta-engine acting as a non-blocking coordinator that handles routing, resource assignment, and result collection.

<span id="section-06"></span>

## 6 Evaluation

In this section, we analyze representative agentic workloads, evaluate Autellix's performance against state-of-the-art LLM serving systems, and ablate its design choices.

### Workloads

<span id="figure-11"></span>

![Workload analysis](../../papers/autellix/figure-11.png)

**Figure 11. Workload analysis.** LLM call statistics of programs from each workload. Input and output length distributions for (a) ShareGPT, (b) BFCL, and (c) LATS. Subfigure (d) plots the distribution of number of LLM calls in each workload.

<span id="figure-12"></span>

![Single-engine main results](../../papers/autellix/figure-12.png)

**Figure 12. Single-engine, main results.**

<span id="figure-13"></span>

![Single-engine tail latencies](../../papers/autellix/figure-13.png)

**Figure 13. Single-engine tail latencies.**

Our real-world experiments evaluate Autellix over four representative agentic workloads, which widely vary in the number of decode tokens, prefill tokens, and the LLM calls ([Figure 11](#figure-11)).

**Chatbot Agent: ShareGPT [Con23].** The ShareGPT dataset comprises of user-generated conversational inputs and outputs, typical for chatbot applications. The number of LLM calls follows a long-tailed distribution with a mean of 6.66 and a max of 80 ([Figure 11](#figure-11)). ShareGPT's conversational nature is evident in its decode-heavy calls, averaging 277 decode tokens versus 256 prefill tokens, where short prompts generate detailed responses ([Figure 11](#figure-11)). Our experiments replay entire conversations as a program rather than the first turn.

**ReAct Agent: BFCL [Gon24a].** The Berkeley Function Calling Leaderboard (BFCLv3) evaluates LLMs on multi-turn, multi-step tool-usage tasks. Compared to ShareGPT, BFCL's LLM calls are less long-tailed, with a mean of 10.75 and a maximum of 70 calls per program ([Figure 11](#figure-11)). BFCL is prefill-heavy, averaging 735.06 tokens per call due to long system prompts and detailed tool signatures, while decodes are short, averaging 34.14 tokens ([Figure 11](#figure-11)). BFCL thus encapsulates dynamic workflows that alternate between heavy prefills phases and short decodes with function calls.

**Monte Carlo Tree Search: LATS [Wan24].** LATS workloads, derived from running MCTS on HotpotQA [Man18], are computationally intensive and involve many parallel LLM calls. Each program instance contains on average 159.7 LLM calls—an order of magnitude more than ShareGPT or BFCL workloads ([Figure 11](#figure-11)). Moreover, the prefill and decoding phase of each call averages 467.2 and 72.6 tokens respectively ([Figure 11](#figure-11)). These distributions highlight MCTS's inherently iterative, parallel nature, pushing LLM serving systems to handle large volumes of concurrent calls efficiently.

**Mixed.** We combine all three workloads, sampling equally from each to ensure diversity. This workload stress tests Autellix's performance across different program classes.

For our experiments, we synthesize a trace by randomly sampling programs, not LLM calls, from the above workloads and generating programs' arrivals using a Poisson process `lambda`, following established methodologies [Sto23c, Jin24]. This approach ensures our setup closely reflects real-world scenarios.

### Experimental Setup

**Models & Testbed.** We evaluate on three models: LLaMA-3.1-8B, 70B and Falcon-180B, running on 1, 4, and 8 GPUs, respectively. Experiments are conducted on a GCP Compute Engine `a2-ultragpu-8g` instance with eight A100-SXM4-80GB GPUs connected via NVLink, 1360 GB host memory, PCIe-4.0×16, and 2 TB of disk space.

**Metrics.** Existing LLM serving systems focus on request-level metrics, such as Time-to-First-Token (TFTT) and Time-per-Output-Token (TPOT), also referred to as token latency [Sto23c, Jin24, Kas24]. However, these metrics overlook end-to-end latency for agentic programs. To that end, we introduce program-level token latency, defined as the total program response time divided by the number of tokens generated [+3]. A high-throughput system for programs should retain low program-level latency during high request rates. For simplicity, we refer to our metric as *latency* throughout the evaluation.

**Baselines.** Our evaluation considers three baselines. All baselines, including Autellix, use the same max batch size.

- **vLLM [Sto23c].** vLLM is the state-of-the-art, high throughput LLM serving system that integrates continuous batching [Chu22] and PagedAttention [Sto23c] to reduce KV cache fragmentation. Its default scheduling policy is FCFS, which is application-unaware and suffers from call-level and program-level HoL blocking. We use vLLM v0.6.1.
- **vLLM-opt.** An optimized version of vLLM that enables chunk-prefill [Ram23], prefix-caching [She24, Qiu24], and multi-step scheduling. Based on vLLM's blogpost [Tea24], it's performance closely matches SGLang [She24] and TensorRT [Nvi23].
- **MLFQ.** On top of vLLM-opt, it implements preemption via the Multi-Level Feedback Queue algorithm [Jin24]. This baseline ablates the impact of program and call-level blocking.

### End-to-End Single-Engine Performance

In [Figure 12](#figure-12), we evaluate the end-to-end performance of Autellix against three baselines and four workloads: ShareGPT, BFCL, LATS, and Mixed. Across all workloads, Autellix consistently achieves the highest throughput given same token latency. Conversely, vLLM performs worst due to its lack of prefix caching, which results in expensive recomputation of cumulative state ([Figure 7](#figure-07)) for LLM calls in the same program. Across workloads, the relative performance between vLLM-opt, MLFQ, and Autellix varies.

The first two rows plot the latencies for single-threaded workloads, ShareGPT and BFCL. vLLM and vLLM-opt's FCFS scheduling causes severe head-of-line (HoL) blocking, which increases latencies as arrival rates increase. In contrast, MLFQ, a preemptive algorithm, mitigates call-level HoL, improving throughput by 1.5× over vLLM-opt. However, at high load, it still suffers from program-level HoL. By employing PLAS to tackle both call- and program-level HoL, Autellix achieves up to 8× throughput of vLLM, twice that of vLLM-opt, and a 1.5× improvement over MLFQ under heavy load.

The third row presents results for the multi-threaded LATS workload. Autellix outperforms vLLM, MLFQ, and vLLM-opt by up to 5×, 2.5×, and 2×, respectively. Notably, MLFQ's preemptive scheduling, which benefits single-threaded programs, is less effective in multi-threaded settings. By aggressively prioritizing shorter requests, MLFQ inadvertently disrupts threads in the same program, exacerbating program-level HoL blocking and stalling overall progress. Autellix's ATLAS policy holistically optimizes resource allocation across all threads, maintaining balanced progress and sustaining high throughput under heavy multi-threaded workloads.

The fourth row of [Figure 12](#figure-12) illustrates performance on mixed workloads. Autellix achieves up to 15× higher throughput than vLLM, 5.5× higher than MLFQ, and 4× higher than vLLM-opt. Since Autellix reduces program and call-level blocking, Autellix performs better as programs' heterogeneity, or the diversity of LLM calls and decode lengths, increases.

**Tail latency.** Preemptive scheduling strategies can reduce average latency but risk increasing tail latency by starving long-running programs. [Figure 13](#figure-13) reports the 95th (P95) and 99th (P99) percentile latencies across different workloads on LLaMA-3.1-8B. For ShareGPT, MLFQ significantly improves average latency compared to vLLM-opt ([Figure 12](#figure-12)), but exhibits poor P95/99 tail latencies. In contrast, for BFCL, MLFQ outperforms vLLM-opt in both cases. In 7 of 8 scenarios, Autellix maintains consistently lower tail latencies than MLFQ and vLLM-opt and improves throughput by up to 1.7× for P95/99 tail latencies, demonstrating robust performance gains in both average and tail performance metrics.

### End-to-End Multi-Engine Performance

<span id="figure-14"></span>

![Multi-engine main results](../../papers/autellix/figure-14.png)

**Figure 14. Multi-engine, Main Results.** Latencies (Avg., P95/99) with respect to different load balancing policies.

To evaluate the effectiveness of Autellix's data locality-aware load balancer ([Section 4.3](#section-04-03)), we compare it against two widely used load balancing strategies under identical scheduling policies (*PLAS*, *ATLAS*) for the sake of fairness:

- **Round Robin.** Requests are assigned to engines in cyclic order—ensuring an even distribution of request counts—which is the default load-balancer policy for Kubernetes [Src17]. This strategy ignores data locality, resulting in costly KV cache misses and high recomputation overheads.
- **Least Used.** Requests are assigned to the engine with the lowest number of LLM calls in the system, effectively balancing engine workloads. However, like Round Robin, it neglects data locality and incurs frequent KV recomputations.

We conduct experiments using four replicas of LLaMA3.1-8B and two replicas of LLaMA3.1-70B with the ShareGPT and LATS workloads. The results, shown in [Figure 14](#figure-14), demonstrate Autellix's effectiveness in maintaining low average and tail latencies across all configurations. Autellix delivers up to 1.4× higher throughput compared to both baselines. The benefit is more pronounced in ShareGPT workload, where chat history reuse significantly amplifies KV-cache locality. These advantages become even more evident as the number of replicas increases, as a larger pool of engines reduces the likelihood of a request being routed to one with its locality.

<span id="figure-15"></span>

![Scalability experiments](../../papers/autellix/figure-15.png)

**Figure 15. Scalability Experiments.** Given same SLO (defined as s/tok), Autellix's max arrival rate (program/s) scales linearly with number of replicas, or LLM engines.

**Scalability.** To evaluate the scalability of Autellix, we assess its performance as the number of engine replicas increases under various latency requirements, using the ShareGPT workload with the LLaMA3.1-8B model. [Figure 15](#figure-15) shows linear scaling in all cases. Leveraging program-level load balancing, Autellix effectively scales horizontally without data locality overhead, making it a robust solution for large-scale LLM deployments.

### Ablations

We ablate Autellix over different scenarios, including offline batch inference, timing breakdown, and various design choices, such as the swap kernel. All experiments run LLaMA3.1-8B [Oth24a] over ShareGPT [Src23] and LATS [Wan24].

#### Offline inference

<span id="figure-16"></span>

![Offline batch inference](../../papers/autellix/figure-16.png)

**Figure 16. Offline batch inference.** Autellix decreases the time, or makespan, required to process a batch of programs.

In offline scenarios that prioritize throughput over latency, large batches of programs are processed in bulk rather than interactively or in a streaming fashion. We consider a use case where all programs are submitted at the start. [Figure 16](#figure-16) presents the makespan of all programs across all systems using the ShareGPT dataset. Autellix consistently outperforms the baselines, decreasing the average makespan by 10–40%. At 4000 programs, MLFQ fails to complete execution. By assigning all new requests to the highest-priority queue, it creates many active LLM requests, causing severe memory contention and frequent GPU-CPU swapping. This overwhelms system resources, resulting in Out-Of-Memory (OOM) errors despite a large swap space (>1.2TB).

#### Timing Breakdown

<span id="figure-17"></span>

![Breakdown of inference overheads](../../papers/autellix/figure-17.png)

**Figure 17. Breakdown of Inference Overheads.** Autellix significantly reduces wait time and introduces minor scheduler overheads to vLLM. Autellix also reduces swap times with its improved kernel.

[Figure 17](#figure-17) breaks down the time LLM calls spend in the LLM serving layer for Autellix and its corresponding baselines. Overall, Autellix achieves lower token latency for ShareGPT and LATS by reducing wait and swap times, attributed respectively to Autellix's program-level scheduling policy and improved swap kernels. Due to higher scheduling costs for preemption, both Autellix and MLFQ attain higher scheduling times than vLLM-OPT's naive FCFS. Yet, Autellix still incurs lower scheduling overhead than MLFQ by incorporating program-level priorities and better distributing LLM calls efficiently across different priority queues. In contrast, MLFQ assigns new LLM calls to the highest-level priority queue by default; hence, a majority of LLM calls reside in high-priority queues, leading to large scheduling overheads.

#### Comparison to Optimal Scheduling

<span id="figure-18"></span>

![Comparison to optimal scheduling policy](../../papers/autellix/figure-18.png)

**Figure 18. Comparison to optimal scheduling policy.** In simulation, Autellix outperforms other scheduling policies; however, there remains a visible gap relative to the optimal policy (SRPT).

Optimal scheduling policies like Shortest Remaining Processing Time (SRPT) assume complete knowledge of each program's runtime—an unrealistic assumption in practice. Hence, we emulate clairvoyance with a simulator by exposing each program's total LLM calls and decode steps a priori. The simulation only considers scheduling, where each continuous-batching step is identical. Under these simplified conditions, Autellix outperforms FCFS and other preemptive schedulers (e.g., Round Robin, MLFQ). Nevertheless, a noticeable gap remains between Autellix and SRPT, showing that prior knowledge can significantly boost performance.

<span id="figure-19"></span>

![Impact of Autellix's swap kernel](../../papers/autellix/figure-19.png)

**Figure 19. Impact of Autellix's swap kernel.** Autellix reduces total swaps and GPU-CPU swap times, improving throughput.

#### Impact of Swapping Kernel

Preemptive scheduling increases active LLM calls in the system, incurring high GPU memory utilization. This leads to frequent GPU-CPU swaps for fetching relevant KV cache and significant swapping overheads at high request rates [Jin24]. Autellix mitigates this by batching parallel KV block transfers into a single operation—reducing swaps by up to 18×, swap times by 3–7×, and achieving 1.3× higher throughput than vLLM's implemented kernel ([Figure 19](#figure-19)).

## 7 Discussion & Future Work

**Graph Optimizations.** Autellix assumes no prior knowledge of a program's execution DAG and dynamically constructs the graph as an internal representation (IR) during runtime. While full prior knowledge of a program's execution is unrealistic, anticipating its immediate next steps can be practical—thereby enabling *compiler optimizations* such as branch prediction and speculative execution, which enables future LLM calls to execute while prior calls are still completing. We defer such optimizations to future works.

**Post-Training.** Reasoning models, such as Deepseek-R1 [Dee25] and OpenAI's o1/o3 models [Ope25], are post-trained via end-to-end reinforcement learning (RL) to optimize the thought process. To accelerate training, distributed RL systems alternate between distributed on-policy sampling and training to collect trajectories and perform policy gradient updates [She24a, Sto18]. With more effective scheduling, Autellix reduces the total makespan for batch sampling for each RL iteration, which immediately benefits distributed post-training systems.

## 8 Conclusion

We present Autellix, a distributed LLM serving system designed for highly-dynamic and general programs, not individual LLM calls. Autellix's key innovation is to leverage program-level statistics, such as the cumulative service times, to better prioritize and schedule LLM calls, thereby improving the end-to-end response times and throughput of programs. We propose two general scheduling algorithms—for single- and multi-threaded programs—and a locality-aware load balancer that effectively reduces programs' waiting and execution times. Our experiments demonstrate that Autellix improves throughput of programs by 4×–15× at the same latency compared to state-of-the-art systems like vLLM.

### Acknowledgement

We thank Pravein Kannan, Diana Arroyo, and Marquita Ellis from IBM for their insightful discussion. We thank Google Deepmind for funding this project, providing AI infrastructure for us to run experiments. Sky Computing Lab is supported by gifts from Accenture, AMD, Anyscale, Google, IBM, Intel, Microsoft, Mohamed Bin Zayed University of Artificial Intelligence, Samsung SDS, SAP, Uber, and VMware.

[+2]: LLM agents with identical system prompts but different models (e.g., LLaMA [Oth23], Mistral [Say23]) are considered distinct [Zou24a].

[+3]: For multi-threaded programs, *program-level* token latency is computed as the critical path response time divided by the total tokens across all threads.
