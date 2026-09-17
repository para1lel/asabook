---
title: 'AgenticRAG: Enterprise Knowledge Retrieval'
createTime: 2026/09/17 09:47:11
permalink: /en/papers/agenticrag/
---

> [Susheel Suresh](https://susheels.github.io/) [+equal] [+susheel-email], [Hazel Mak](https://www.linkedin.com/in/s-n-hazel-mak) [+equal] [+hazel-email], [Shangpo Chou](https://www.linkedin.com/in/shangpochou), [Fred Kroon](https://ca.linkedin.com/in/fredkroon), [Sahil Bhatnagar](https://dblp.org/pid/273/7895.html). Microsoft Corporation. First submitted to arXiv on May 7, 2026; current version: v1. [AgenticRAG: Agentic Retrieval for Enterprise Knowledge Bases](https://arxiv.org/abs/2605.05538), [TeX source](https://arxiv.org/e-print/2605.05538), <a href="/paper/agenticrag.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

We present **AgenticRAG**, a practical agentic harness for retrieval and analysis over enterprise knowledge bases. Standard RAG pipelines place significant burden of grounding on the search stack, constraining the language model to a fixed candidate set chosen deep in the retrieval process. Our approach reduces this overdependence by layering a lightweight harness on top of existing enterprise search infrastructure, equipping a reasoning LLM with search, find, open, and summarize tools enabling the model to iteratively retrieve information, navigate within documents, and analyze evidence autonomously. On three open benchmarks we observe substantial gains: 49.6% recall@1 on BRIGHT (+21.8 pp over the best embedding baseline), 0.96 factuality on WixQA (+13% relative improvement), and 92% answer correctness on FinanceBench—within 2 pp of oracle access to true evidence. Ablation studies show that the most significant factor is the shift from single-shot retrieval to agentic tool use ($5.9\times$ improvement), while multi-query search and in-document navigation contribute to both quality and efficiency. We present various design choices in our agentic harness that were informed by pre-production deployments. Our results demonstrate its suitability for real-world enterprise production environments.

<span id="section-1"></span>

## 1 Introduction

Standard retrieval-augmented generation (RAG) pipelines follow a static retrieve-then-generate paradigm [Lew20]. In this design, the search stack effectively determines the final candidate set the large language model (LLM) will see and the model’s reasoning is constrained to that set. Modern enterprise-grade search stacks are highly optimized for scalability, latency, and multi-stage ranking pipelines built on inverted indexes, probabilistic retrieval, and learned ranking models [Liu09, Nog19, Tha21]. These systems excel at keyword and short semantic queries and are strong for high-recall candidate generation. However, they are not designed to resolve situational, multi-document, or analytically complex information needs—the kinds of queries knowledge workers issue against dense corpora such as technical manuals, compliance documents, and financial reports.

Real-world RAG systems [Azu26] attempt to compensate for these limitations through retrieval enhancement techniques such as HyDE [Gao23a], multi-query reformulation [Wan23o], and adaptive or iterative retrieval strategies [Tri22a, Jeo24b]. While these methods provide robustness to query phrasing and higher retrieval coverage, they largely preserve the same architectural assumption: retrieval decisions are finalized before substantive reasoning begins. The LLM still operates over a fixed candidate set selected deep in the search stack, without the ability to iteratively navigate documents, synthesize evidence across sources, or reassess results from a higher-level vantage point.

Recent advances in reasoning-capable language models have demonstrated strong performance on planning and iterative external tool use [Yao23b, Sch23]. Rather than hard-coding retrieval steps, we can empower the model itself to drive the process—deciding what to search for, which documents warrant deeper investigation, and when sufficient evidence has been gathered. This relaxes the pressure on the search stack: it only needs to achieve good recall, while the model handles the final precision from its broader context. We present *AgenticRAG*, a practical harness that equips a reasoning LLM with four tools—**search**, **find**, **open**, and **summarize**—layered on top of existing enterprise search infrastructure. The search tool delegates to the underlying search stack for broad candidate discovery, while find and open serve as precision instruments that let the model drill into candidate documents via in-document search and full-content retrieval (with rolling window access). To manage the growing context during long reasoning chains, the harness monitors token usage and triggers the summarize tool when a threshold is reached, allowing the model to consolidate its findings while preserving key references. Our contribution is system-level: a lightweight inference-time tool harness that requires no model fine-tuning, custom embedding model, graph construction, or corpus-specific preprocessing beyond indexing documents into the existing enterprise search backend.

We evaluate on three benchmarks spanning retrieval, enterprise QA, and financial document reasoning. Our approach achieves 49.6% recall@1 on BRIGHT (+21.8 pp over the best embedding baseline), 0.96 factuality on WixQA (+13% relative), and 92.00% answer correctness on FinanceBench—within 2 pp of oracle access. Our method is deployed for pre-production evaluation, and learnings from these deployments directly inform our design choices. We provide detailed ablations analyzing the contribution of each tool, the effect of multi-query search, and model-level differences in retrieval strategy.

<span id="section-2"></span>

## 2 Related Work

Retrieval-Augmented Generation (RAG) grounds LLM generation in external corpora to mitigate parametric memory limitations [Lew20, Guu20]. Early approaches focused on identifying relevant documents using sparse or dense vector retrieval [Kha20b, Iza21] to enhance performance on knowledge-intensive NLP tasks. As context windows expanded, research shifted toward scaling retrieval to trillions of tokens [Bor22] and optimizing in-context learning [Ram23a, Shi23f]. Despite these advancements, standard RAG pipelines often struggle with "long-tail" knowledge and can suffer from hallucinations when retrieval fails [Mal22, Gao24e]. Furthermore, static "retrieve-then-generate" paradigms lack the flexibility to handle complex, multi-hop queries that require iterative information gathering [Jia23c, Lew23].

To address the brittleness of static pipelines, the field has evolved toward *Agentic* patterns, where autonomous agents (LLMs) dynamically orchestrate the retrieval process [Sin25b, Och25]. Foundational work in agentic behaviors, such as ReAct [Yao23b] and Toolformer [Sch23], demonstrated that LLMs could effectively wield external tools to solve reasoning problems. This paradigm has been formalized in systems like Self-RAG [Asa23] and Corrective RAG [Yan24o], which employ self-reflection mechanisms to critique retrieved content and trigger fallbacks (e.g., web search) when necessary. Recent approaches propose to integrate retrieval into planning: PlanRAG [Lee24e] and Search-o1 [Li25k] separate high-level planning from low-level execution, allowing agents to decompose complex queries into sub-tasks. Similarly, Search-R1 [Jin25b] uses reinforcement learning to train LLMs for autonomous search decisions. While effective, many of these systems are designed for open-domain search or require fine-tuning, reinforcement learning, or dedicated retrieval policies, which makes them less directly applicable to proprietary enterprise corpora that cannot be exported for training. They can also incur high latency and token costs due to recursive reasoning loops [Tri22a].

Another critical limitation in standard RAG is the "flattening" of documents into disjointed chunks, which discards valuable structural priors like headings and document boundaries. RAPTOR [Sar24] addresses this by recursively clustering and summarizing text chunks into a tree structure, enabling retrieval at varying levels of abstraction. Similarly, HiQA [Che24x] constructs multi-document hierarchical contexts. Graph RAG [Edg24, Sca25a] approaches seek to build knowledge graphs from documents to support query-focused summarization. While powerful for unifying knowledge [Pan24h, Wan24ah], graph construction is often computationally prohibitive for dynamic enterprise environments.

In contrast, our Agentic RAG harness is an inference-time system that leverages a reasoning model with a "search" tool (using a fast enterprise grade search stack) alongside "find" and "open" tools for deeper information gathering and reasoning. This positions the contribution as a deployable system integration for enterprise file systems: it works with existing search infrastructure, preserves document access controls, and avoids extensive pre-computation or retraining.

<span id="section-3"></span>

## 3 Method

<span id="section-3-1"></span>

### 3.1 System Overview

We present an agentic RAG system for enterprise document search and question answering over large file systems. Unlike traditional single-pass RAG pipelines, our system employs an iterative reasoning loop where a large language model (LLM) autonomously decides when to search for documents, drill into specific passages, and retrieve full content before producing a final answer.

The system addresses several challenges in enterprise RAG: (1) multi-step reasoning: complex queries require information from multiple documents, (2) context window constraints: accumulated retrieval results must fit within LLM limits, (3) grounded responses: answers must include traceable citations to source documents, and (4) multi-turn efficiency: follow-up queries should reuse previously retrieved content rather than re-executing redundant searches. Our architecture supports multiple model families and reuses existing search infrastructure for the backend implementation of the retrieval tools. By lightweight, we mean that the harness consists of four tools, requires no model fine-tuning, no graph construction, and no custom embedding index beyond the enterprise search stack already deployed for document discovery. Overall the system comprises three main components:

1. **Agentic Loop**: Orchestrates LLM-tool interactions, bounded by maximum iterations.
2. **Retrieval Tools**: Three tools (*search*, *find*, *open*) provide hierarchical access to enterprise documents. A *summarize* tool for context management during long reasoning chains.
3. **Conversation State**: Maintains message history, token accounting, and reference ID mappings that track documents across iterations.

<span id="section-3-2"></span>

### 3.2 Agentic Loop

The agent processes each query through a bounded iteration loop ([Figure 1](#figure-01)). Each iteration, upon receiving the current conversation, the agent either selects a tool to call and appends to the conversation, or returns the final answer with citations.

The loop terminates under two conditions: (1) the model produces a text response, or (2) the iteration count reaches maximum iterations (default: 15). When maximum iterations are reached without a final answer, the agent issues a forced completion request, requiring the model to respond using available information. If the token budget is exceeded during execution, the agent triggers context management ([Section 3.4](#section-3-4)) to free space and continues the loop. For detailed algorithm, see [Section 7.1](#section-7-1).

<span id="figure-01"></span>

![Agentic loop.](../../papers/agenticrag/figure-01.png)

**Figure 1.** Agentic loop.

<span id="section-3-3"></span>

### 3.3 Retrieval Tools

The system provides three retrieval tools enabling hierarchical document exploration ([Table 1](#table-01)). The agent decides which to invoke based on current information needs.

**Search** performs enterprise-wide document discovery by delegating to the existing enterprise search stack. In the default configuration, the model may issue up to five query reformulations in one tool call. The tool returns up to 10 results per query, each containing a snippet, title, filename, file type, and other available metadata. Results from multiple queries are combined and deduplicated. Each result receives a unique reference ID (format: $\mathrm{turn}m\mathrm{search}n$) using a globally incrementing counter, enabling subsequent find and open operations.

**Find** performs targeted in-document search within a single document identified by its reference ID. Given a list of keyword patterns, lexical matching is case-insensitive substring matching; an optional semantic find mode can also be enabled. The tool returns up to 2 matching passages per pattern. Results are deduplicated by content and truncated at a bounded token limit ($\sim 11\mathrm{k}$ tokens). Find is most useful when the model knows *what* to look for, such as a revenue metric or a named concept inside a long filing.

**Open** retrieves full document content in a fixed line window. Each call returns a window of lines (default: 1,800) starting from either the beginning (line 0) or a specific line number chosen by the agent, and a response header indicating the viewing range and total document length (e.g., "Viewing lines [0-1799] of 3000 lines"). To access more than one portion from a file, the model makes subsequent calls with an explicit line number value. This enables navigation through documents exceeding the window size while keeping each response bounded. Open is most useful when the model knows *where* to read, such as context around a table, section heading, or line-numbered preview. The system prompt guides effective tool usage. See [Section 7.2](#section-7-2) for details.

<span id="table-01"></span>

![Retrieval Tool Specifications.](../../papers/agenticrag/table-01.png)

**Table 1.** Retrieval Tool Specifications.

<span id="section-3-4"></span>

### 3.4 Context Management

Since retrieval tools can load $\sim 11\mathrm{k}$ tokens from files each time, the context window can be used up quickly. To manage that, the harness monitors token usage against a 128K-token threshold: it emits an internal warning when the conversation reaches 90% of the budget and forces summarization at the threshold. The summarize tool enables the model to record current reasoning and designate which references to preserve. The system then scans tool messages and removes content not associated with preserved reference IDs, freeing tokens while retaining cited evidence. This approach extends effective context capacity. See [Figure 2](#figure-02) for an example conversation history before and after context management.

<span id="figure-02"></span>

![Example conversation history with context management via forced summarize tool call.](../../papers/agenticrag/figure-02.png)

**Figure 2.** Example conversation history with context management via forced summarize tool call.

<span id="section-4"></span>

## 4 Experiment Setup

Our goal is to evaluate AgenticRAG in realistic enterprise settings where knowledge workers issue complex situational queries requiring multi-step reasoning over large corpora of long, domain-specific documents. To this end we adopt the **BRIGHT benchmark** [Su24a] which contains StackExchange questions spanning eight domains. We choose to evaluate on the long-context setting of BRIGHT, where documents correspond to entire web pages rather than snippets and the task is to retrieve the full relevant document(s) for a given query. For our agentic setting, the full BRIGHT web pages are converted to document files and indexed into the same enterprise search backend used by the search tool. Search returns snippet previews with metadata and reference IDs, and the find and open tools then access full document content through those IDs. Standard protocol of using Recall@1 to measure relevance of retrieved documents is adopted [Su24a]. We instruct the model in AgenticRAG to provide relevancy scores for the citations it uses when producing the answer, which induces a ranking over documents for evaluation. We also test our method on **WixQA** [Coh25], which targets real-world support and troubleshooting enterprise scenarios that require multi-document and multi-step reasoning for procedural answers. We adopt the same LLM based factuality metric defined in WixQA. Finally, we run evaluations on the popular **FinanceBench** [Isl23] dataset, which contains financial questions that require deep reasoning over large company financial documents. Our metric here is answer correctness as a proxy for accurate information retrieval, since questions pertain to single documents. Detailed benchmark descriptions, query set and corpus statistics are presented in [Section 8](#section-8).

<span id="section-5"></span>

## 5 Results

<span id="section-5-1"></span>

### 5.1 Long-Context Retrieval on BRIGHT

<span id="table-02"></span>

![Long-context retrieval performance on unsplit web pages of StackExchange data from BRIGHT benchmark.](../../papers/agenticrag/table-02.png)

**Table 2.** Long-context retrieval performance on unsplit web pages of StackExchange data from BRIGHT benchmark. Scores are reported in recall@1. Best baseline per category shown; full results in [Table 9](#table-09).

[Table 2](#table-02) presents retrieval performance on the BRIGHT benchmark, showing the best baseline per category. Full results with all models are in [Table 9](#table-09), including reasoning-enhanced baselines such as LLM re-rankers over BM25/SBERT and ReDI. Our agentic harness equipped with search, find, open, and summarize tools enables both Claude Sonnet 4.5 and GPT-5-mini to achieve the highest recall@1 across all eight benchmark splits compared to all baselines. With our AgenticRAG retrieval harness, Claude Sonnet 4.5 achieves **49.6%** average recall@1 (+21.8 pp over Qwen, the best embedding model at 27.8%) and GPT-5-mini reaches 43.5% (+15.7 pp). Gains are consistent across domains, with the largest improvements in Economics (+33.0 pp), Earth Science (+24.6 pp), Robotics (+33.7 pp), and Psychology (+25.6 pp). Even with a vast corpus of 5.65K long documents averaging 16K tokens each, our method scales by leveraging traditional retrieval via the search tool, deeper reasoning enabled by the open / find tools and effective context window management by the summarize tool. The best-performing reasoning-enhanced baseline, ReDI, uses a fine-tuned Qwen3-8B decomposition and retrieval-fusion model and achieves 26.0% recall@1 in the BRIGHT long-document setting. Our method outperforms it by +23.6 pp (via Claude) and +17.5 pp (via GPT-5-mini). Static approaches such as one-time query rewriting or LLM-based re-ranking cannot match the iterative reasoning that our harness provides, and the gap is evident across all splits.

<span id="section-5-2"></span>

### 5.2 Enterprise QA on WixQA

[Figure 3](#figure-03) presents factuality results on the WixQA benchmark, which requires multi-document analysis to answer enterprise support questions. Semantic embeddings alone fail to capture the cross-document reasoning needed for these queries, whereas the iterative search and reasoning enabled by our harness excels. On the *Expert Written* split, our method with GPT-5-mini achieves a factuality score of **0.96**, compared to 0.85 for E5 retrieval and 0.83 for BM25—a **13%** relative improvement over the best baseline. We observe similar gains on the *Simulated* split; see [Section 9.3](#section-9-3) for details.

<span id="figure-03"></span>

![Factuality performance on the WixQA Expert Written dataset.](../../papers/agenticrag/figure-03.png)

**Figure 3.** Factuality performance on the WixQA Expert Written dataset (N=200). Our agentic approach (red) substantially outperforms BM25 (blue) and E5 (green) retrieval baselines across all generation models.

<span id="section-5-3"></span>

### 5.3 Financial Document QA on FinanceBench

<span id="table-03"></span>

![Evaluation results on FinanceBench.](../../papers/agenticrag/table-03.png)

**Table 3.** Evaluation results on FinanceBench (N=150).

[Table 3](#table-03) presents answer correctness on the FinanceBench dataset, which evaluates question answering over real-world financial filings. The retrieval corpus consists of 84 long financial documents averaging $\sim 116\mathrm{K}$ tokens each ($\sim 140$ pages per PDF). Our agentic approach with GPT-5-mini achieves **92.00%** correctness, substantially outperforming both traditional RAG (by **$3.8\times$**) and moreover, is more general than the agentic tool use baseline of [Sub26] (by **$2.8\times$**), which relies on keyword search tools like pdfgrep, rga, and Linux commands. We also adopt a baseline where the ground-truth full-page evidence is provided directly to GPT-5-mini, bypassing agentic retrieval entirely. This oracle setting achieves 94.00% and establishes an upper bound on the model's reasoning ability given perfect evidence. Our agentic system is within 2 pp of this upper bound which demonstrates its effectiveness. Between GPT-5-mini vs Claude Sonnet 4.5, our harness is equally effective.

<span id="section-5-4"></span>

### 5.4 Token Cost and Retrieval Efficiency

<span id="table-04"></span>

![Total token usage comparison between AgenticRAG and Single-shot Search.](../../papers/agenticrag/table-04.png)

**Table 4.** Total token usage comparison between AgenticRAG and Single-shot Search across BRIGHT splits and FinanceBench. All values are averages per query (in thousands) and include system prompt, tool calls, tool results, and any thinking tokens. Cost ratio is AgenticRAG total tokens divided by Single-shot Search total tokens.

[Table 4](#table-04) quantifies the end-to-end token cost of agentic retrieval. We measure total tokens consumed across the full interaction, including model thinking, tool-call arguments, retrieved tool results, and final answer generation. On BRIGHT, AgenticRAG averages 52.3K total tokens per query, compared to 20.4K for Single-shot Search, a $2.6\times$ token overhead. This cost yields a disproportionate quality gain: Claude Sonnet 4.5 with AgenticRAG reaches 49.6% recall@1, compared to 8.41% for Single-shot Search, a $5.9\times$ improvement. FinanceBench is more expensive, averaging 114.8K tokens per query and a $7.8\times$ ratio over single-shot search, which reflects deep navigation over long financial filings. This higher cost is paired with 92.00% answer correctness, close to the 94.00% oracle evidence upper bound. Tool usage in [Table 5](#table-05) further shows that the system operates well within the 15-iteration budget, averaging 4.48-4.79 tool calls per query. The multi-query ablation provides a direct efficiency comparison: the full system achieves comparable recall with 4.79 average tool calls versus 6.79 without multi-query search, a 29% reduction in tool calls.

<span id="section-5-5"></span>

### 5.5 Ablation Studies

<span id="table-05"></span>

![Ablation study of agentic components averaged across all BRIGHT splits.](../../papers/agenticrag/table-05.png)

**Table 5.** Ablation study of agentic components averaged across all BRIGHT splits. Performance is measured by recall (R@k), along with average tool usage and per-tool statistics. Per-split results are in [Table 10](#table-10).

**Single-shot vs Agentic Retrieval.** From [Table 5](#table-05) the most significant finding is the dramatic improvement from single-shot search to full agentic tool use. Single-shot search achieves only 8.41% recall@1 on average, while agentic tool use reaches **43.49%** with GPT-5-mini and **49.59%** with Claude Sonnet 4.5—representing **$5.2\times$** and **$5.9\times$** improvements respectively. Notably, the proprietary search stack behind our search tool trades off raw retrieval quality for speed, immense scale, and availability compared to the state-of-the-art embedding-based retrievers in [Table 2](#table-02). However, these quality differences vanish when our agentic harness is employed with a reasoning language model. The improvement is consistent across splits (ref. [Table 10](#table-10)).

**Model Comparison and Tool Usage Patterns.** Claude Sonnet 4.5 achieves a **+6.1 pp** improvement over GPT-5-mini, outperforming on seven of eight splits (detailed per-split results in [Table 10](#table-10)). The two models exhibit distinct strategies that reflect an exploration-exploitation trade-off. Claude favors *exploitation*: it uses fewer search calls (2.51 vs 3.39) but opens more documents (1.54 vs 1.22) and relies more on semantic find (0.42 vs 0.14, a $3\times$ increase), going deeper into candidate documents. GPT-5-mini favors *exploration*: it issues more search calls with reformulated queries rather than using in-document find, casting a wider net across the corpus. In the BRIGHT long-document setting, where queries have only $\sim 1.9$ golden documents on average amid a large corpus, relevant documents are sparse and broad exploration often surfaces irrelevant results.

**Failure Patterns.** The main observed weakness is broad multi-evidence retrieval, especially the Pony split, where each query has $\sim 6.9$ gold documents on average compared to $\sim 1.9$ across BRIGHT overall. This setting rewards recovering many related documents, whereas our harness is optimized for coarse-to-fine navigation toward a small number of high-value evidence sources. This explains why Pony remains difficult for both of our models despite large gains on scientific and technical splits where relevant documents are sparse. The pattern suggests that future trajectory policies should better detect broad evidence needs and shift from depth-first document reading to wider coverage before final ranking.

**Component Contributions.** We ablate individual components using GPT-5-mini to understand their contributions. The most notable finding concerns multi-query search. In the default configuration, the model can issue up to 5 queries in parallel within a single search tool call, with results de-duped and presented together. Restricting the system to single-query search (**w/o Multi-query Search**, 44.84%), where the model issues only one query per search call but receives the same number of results, achieves comparable recall@1 to the full system but at the cost of increased tool usage—6.79 average tool calls compared to 4.79 for the full system, with notably more search operations (4.38 vs 3.39) and document opens (2.16 vs 1.22). This suggests that multi-query search improves efficiency by finding relevant documents with fewer iterations. Detailed analysis of these ablations is provided in [Section 9.2](#section-9-2).

**Findings from Pre-Production Deployments** In our pre-production evaluation we identified several design choices that guide the model toward more optimal trajectories: (1) **Surfacing document metadata in search results** like title, filename, and file type helps the model disambiguate semantically similar snippets and avoid redundant searches; (2) **Line-numbered document previews** lets the model anchor on specific content and jump to relevant sections in successive open calls; (3) **Candidate reference retention after summarization** in the context window enables the model to go deeper on promising candidates via open/find, rather than restarting retrieval with reformulated queries.(4) **Having a switcher** route complex, multi-intent queries to our agentic rag harness for deeper analysis. Simple queries are routed to traditional rag for faster answers. This is vital for the tradeoff between user experience, cost, model availability. We have good early signals of this being effective and we continue to pursue this hybrid approach.

<span id="section-6"></span>

## 6 Conclusion

We presented a practical harness for AgenticRAG that equips reasoning language models with search, find, open, and summarize tools to autonomously retrieve and reason over large enterprise corpora. Across three benchmarks, our approach achieves 49.6% recall@1 on BRIGHT (+21.8 pp over the best embedding baseline), 0.96 factuality on WixQA (+13% relative), and 92.00% answer correctness on FinanceBench—within 2 pp of oracle access. Token analysis shows that these gains require a moderate $2.6\times$ token overhead on BRIGHT relative to single-shot search, while delivering a $5.9\times$ recall@1 improvement. These results demonstrate that our harness effectively extracts the value of reasoning models for enterprise information retrieval tasks requiring deep, multi-step reasoning. Future work will focus on large-scale deployment, budget-aware routing between traditional and agentic RAG, deeper failure analysis, ablations over iteration and window-size budgets, and optimizing retrieval trajectories for fast iterative reasoning via fine tuning.

## Acknowledgments

We thank Eli Coon, Kinfe Mengistu and members of the broader Copilot Studio team for feedback and discussions during internal experimentations. Special thanks to James Cai and Alejandro Gutierrez Munoz for technical guidance and project sponsorship.

<span id="section-7"></span>

## 7 Method Details

<span id="section-7-1"></span>

### 7.1 Agentic Loop Algorithm

Detailed agentic loop algorithm is shown in [Algorithm 1](#algorithm-01).

<span id="algorithm-01"></span>

**Algorithm 1: Agentic Loop.**

- **Require:** $\mathrm{user\_query}$, $\mathrm{max\_calls}$, $\mathrm{token\_threshold}$.
- **Ensure:** Formatted answer with citations.
- $\mathrm{conversation}.\mathrm{add}(\mathrm{user\_query})$.
- **For** $i=1$ **to** $\mathrm{max\_calls}$:
  - **If** $\mathrm{tokens}(\mathrm{conversation})\geq\mathrm{token\_threshold}$:
    - $\mathrm{ManageContext}()$. *Force summarize.*
  - $\mathrm{response}\leftarrow\mathrm{LLM}(\mathrm{conversation},\mathrm{tool\_schemas})$.
  - **If** $\mathrm{response}.\mathrm{has\_tool\_calls}$:
    - **For each** $\mathrm{tool\_call}$ **in** $\mathrm{response}.\mathrm{tool\_calls}$:
      - $\mathrm{result}\leftarrow\mathrm{ExecuteTool}(\mathrm{tool\_call})$.
      - $\mathrm{conversation}.\mathrm{add}(\mathrm{tool\_call},\mathrm{result})$.
  - **Else:**
    - **Return** $\mathrm{FormatAnswer}(\mathrm{response}.\mathrm{text})$.
- **Return** $\mathrm{ForceFinalAnswer}()$.

<span id="section-7-2"></span>

### 7.2 System Instructions for Tool Use

Overall instructions include:

- Search before answering when uncertain.
- Progressively explore using find or open when snippets are insufficient.
- Reuse previous results rather than performing search again.
- Cite every time when information is used from tool outputs.

When to use search:

- Primary search tool across enterprise corpus.
- First choice for any work-related query.
- When users reference current/changing information, enterprise-specific terms, or acronyms.
- To verify details rather than making assumptions.

When to use find:

- In-document pattern search for relevant files from search results.
- When search results do not give enough details.
- To get a focused view of a result in relation to certain terms.

When to use open:

- Windowed full content retrieval for relevant files from search results.
- When search results snippets are insufficient.
- To pull in more content from the most promising results.
- Can open multiple search results.
- Option to choose a line number close to the relevant content.

<span id="table-06"></span>

![Dataset statistics for the BRIGHT benchmark.](../../papers/agenticrag/table-06.png)

**Table 6.** Dataset statistics for the BRIGHT benchmark [Su24a] long-context splits used in our evaluation.

<span id="table-07"></span>

![Dataset statistics for the WixQA benchmark.](../../papers/agenticrag/table-07.png)

**Table 7.** Dataset statistics for the WixQA [Coh25] benchmark (median values).

<span id="section-8"></span>

## 8 Dataset Details

<span id="section-8-1"></span>

### 8.1 BRIGHT Benchmark

We adopt the BRIGHT benchmark [Su24a], which is designed to capture realistic enterprise scenarios of information retrieval. BRIGHT derives queries from StackExchange posts, reflecting human-authored, highly situational and domain-specific information needs. For each query, the corpus contains positive documents cited in top-voted answers and verified by human annotators, as well as negative documents collected via search engine retrieval. The corpora is normalized web content (e.g., Wikipedia pages, blogs, and reports). This construction has shown to yields realistic retrieval pools with substantial semantic overlap between relevant and irrelevant documents. We choose to evaluate on the long-context setting of BRIGHT, where documents correspond to entire web pages rather than snippets and the task is to retrieve the full relevant document(s) for a given query. Our experiments span eight domains: Biology, Earth Science, Economics, Psychology, Robotics, Stack Overflow, Sustainable Living, and Pony. These domains cover a broad range of scientific, technical, and professional areas commonly encountered in enterprise information retrieval. Across domains, corpora contain hundreds to thousands of documents, with average document lengths ranging from several thousand to over 40k tokens. Queries themselves are also non-trivial in length, with average query sizes exceeding 100 tokens in most domains and reaching several hundred tokens in technical domains such as Robotics and Stack Overflow. Benchmark statistics are detailed in [Table 6](#table-06). We follow the standard evaluation protocol of the BRIGHT benchmark and report Recall@1 for long-context document retrieval.

<span id="section-8-2"></span>

### 8.2 WixQA Benchmark

WixQA targets procedural, long-form queries that require multi-step reasoning and specialized enterprise vocabulary, closely matching real-world support and troubleshooting scenarios. We utilize both the subsets in WiXQA for our experiments: *Expert Written*, containing authentic customer queries with step-by-step answers authored and validated by human domain experts, and *Simulated*, derived from multi-turn user–chatbot interactions and curated into single-turn queries with expert-validated procedural correctness. A defining characteristic of WixQA is its multi-article dependency, where answering a query may require retrieving and synthesizing information from multiple documents. All queries are grounded in a shared enterprise-scale knowledge base of 6,221 domain-specific help articles, making WixQA well suited for evaluating agentic RAG that must coordinate retrieval and reasoning over complex, multi-document enterprise corpora. Datasets statistics are presented in [Table 7](#table-07)

<span id="section-8-3"></span>

### 8.3 FinanceBench

<span id="table-08"></span>

![FinanceBench data statistics.](../../papers/agenticrag/table-08.png)

**Table 8.** FinanceBench [Isl23] data statistics.

FinanceBench [Isl23] is a human-evaluated benchmark consisting of financial questions over public company filings (10-K, 10-Q, 8-K, and earnings reports in PDF form). Questions span metrics-generated and domain-relevant categories: metrics-generated questions target specific financial line items or ratios that require the model to locate the relevant data in the document and often perform a calculation, making them straightforward to verify since each has a single unambiguous answer. Domain-relevant questions require deeper financial reasoning, such as identifying drivers of margin changes or assessing capital intensity. Each query pertains to a single document. We choose this benchmark because of the large size of its documents (averaging $\sim 143$ pages and $\sim 117\mathrm{K}$ tokens per PDF), which is representative of enterprise domains where knowledge workers routinely work with dense, information-heavy manuals, reports, and regulatory filings. Corpus statistics are presented in [Table 8](#table-08). The evaluation metric we use is answer correctness. LLM as a judge is used for this process and manual review of the results is also conducted.

<span id="section-9"></span>

## 9 Additional Results

<span id="section-9-1"></span>

### 9.1 Full BRIGHT Retrieval Results

<span id="table-09"></span>

![Full long-context retrieval performance on unsplit web pages of StackExchange data from BRIGHT benchmark.](../../papers/agenticrag/table-09.png)

**Table 9.** Full long-context retrieval performance on unsplit web pages of StackExchange data from BRIGHT benchmark. Scores are reported in recall@1.

[Table 9](#table-09) presents the complete retrieval results on the BRIGHT benchmark across all baseline models.

<span id="section-9-2"></span>

### 9.2 Detailed Ablation Analysis

<span id="table-10"></span>

![Per-split ablation study of agentic components on BRIGHT.](../../papers/agenticrag/table-10.png)

**Table 10.** Per-split ablation study of agentic components on BRIGHT. Performance is measured by recall (R@k), along with average tool usage and per-tool usage statistics.

[Table 10](#table-10) provides per-split ablation results across all BRIGHT domains. Removing the summarization tool (**w/o Summarize**, 43.34% avg recall@1) has minimal impact, indicating that this component is rarely needed for the retrieval task. Removing semantic find (**w/o Semantic Find**, 46.34%) slightly *improves* average recall@1, likely because the lexical find fallback is sufficient for most in-document searches and removing the semantic option reduces latency, allowing more search iterations within the same compute budget.

<span id="section-9-3"></span>

### 9.3 WixQA Simulated Results

As shown in [Figure 4](#figure-04), on simulated questions with expert validated ground truth answers i.e. *Simulated* split of WixQA, our method achieves **0.94** factuality, compared to 0.77 for both E5+GPT-4o and E5+Claude 3.7. The improvement is even more pronounced on this dataset, with a **22%** relative gain. This suggests that agentic retrieval is particularly effective when questions require more complex reasoning or multi-hop information gathering.

<span id="figure-04"></span>

![Factuality performance on the WixQA Simulated dataset.](../../papers/agenticrag/figure-04.png)

**Figure 4.** Factuality performance on the WixQA Simulated dataset. The performance gap between agentic retrieval and traditional methods is even larger on synthetic questions that require more complex reasoning.

<span id="section-9-4"></span>

### 9.4 Example Conversation

[Figure 5](#figure-05) shows an example conversation from the FinanceBench.

<span id="figure-05"></span>

![Example conversation from FinanceBench.](../../papers/agenticrag/figure-05.png)

**Figure 5.** Example conversation from FinanceBench.

[+equal]: Equal contribution.

[+susheel-email]: `sussuresh@microsoft.com`

[+hazel-email]: `hazelmak@microsoft.com`
