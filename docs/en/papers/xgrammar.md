---
title: 'XGrammar: Efficient Structured Generation'
createTime: 2026/09/14 00:00:00
permalink: /en/papers/xgrammar/
pageClass: paper-reading
---

> [Yixin Dong](https://github.com/Ubospica), [Charlie F. Ruan](https://www.charlieruan.com/), [Yaxing Cai](https://dblp.org/pid/290/7679.html), [Ruihang Lai](https://ruihanglai.com/), [Ziyi Xu](https://acm.sjtu.edu.cn/~xzy2022/), [Yilong Zhao](https://ylzhao.me/), and [Tianqi Chen](https://tqch.github.io/). First submitted to arXiv on November 22, 2024; current version v3. Published at [MLSys 2025](https://proceedings.mlsys.org/paper_files/paper/2025/hash/5c20ca4b0b20b0bd2f1d839dc605e70f-Abstract-Conference.html). [XGrammar: Flexible and Efficient Structured Generation Engine for Large Language Models](https://arxiv.org/abs/2411.15100v3). <a href="/paper/xgrammar.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2411.15100). [TeX source](https://export.arxiv.org/e-print/2411.15100v3). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

The applications of LLM Agents are becoming increasingly complex and diverse, leading to a high demand for structured outputs that can be parsed into code, structured function calls, and embodied agent commands. These developments bring significant demands for structured generation in LLM inference. Context-free grammar is a flexible approach to enable structured generation via constrained decoding. However, executing context-free grammar requires going through several stack states over all tokens in vocabulary during runtime, bringing non-negligible overhead for structured generation. In this paper, we propose XGrammar, a flexible and efficient structure generation engine for large language models. XGrammar accelerates context-free grammar execution by dividing the vocabulary into context-independent tokens that can be prechecked and context-dependent tokens that need to be interpreted during runtime. We further build transformations to expand the grammar context and reduce the number of context-independent tokens. Additionally, we build an efficient persistent stack to accelerate the context-dependent token checks. Finally, we co-design the grammar engine with LLM inference engine to overlap grammar computation with GPU executions. Evaluation results show that XGrammar can achieve up to 100x speedup over existing solutions. Combined with an LLM inference engine, it can generate near-zero overhead structure generation in end-to-end low-LLM serving.

<span id="section-1"></span>

## 1 Introduction

<span id="figure-01"></span>

![Figure 1. Overview of our approach. XGrammar first uses a pushdown automaton to parse the prior LLM output, flexibly supporting diverse grammars and producing the matching stack states. It then uses the stack top to index into the adaptive token mask cache—our key optimization—to retrieve a partial mask. Most of the partial mask consists of context-independent tokens and is determined during preprocessing. A small portion, however, is context-dependent and resolved at runtime. This yields the complete token mask, thus enabling efficient constraint decoding.](../../papers/xgrammar/figure-01.png)

**Figure 1.** Overview of our approach. XGrammar first uses a pushdown automaton to parse the prior LLM output, flexibly supporting diverse grammars and producing the matching stack states. It then uses the stack top to index into the adaptive token mask cache—our key optimization—to retrieve a partial mask. Most of the partial mask consists of context-independent tokens and is determined during preprocessing. A small portion, however, is context-dependent and resolved at runtime. This yields the complete token mask, thus enabling efficient constraint decoding.

Recent advancements in large language models (LLMs) have created new possibilities for complex applications such as code generation [Che21, Wan21h], debugging [Pea22, Moz24], external tool invocation through function calling [Ope24j, Lan24], and robotic control [Liu23q]. These applications bring great demand for the *structured generation* problem, which requires the output of LLMs conforms to specific formats or grammars, such as JSON, SQL or other formats tailored to the task. The downstream applications can then organically consume the structured outputs to perform followup interactions with the system.

Constrained decoding [Deu19, Kuc23] is a commonly adopted method for structured generation. It guarantees the output of LLMs adheres to the specified structure through only allowing tokens that conform to the structure to be generated at each decoding step. At each step, constrained decoding first scans the entire vocabulary to identify invalid tokens and sets their probabilities to zero, thereby preventing them from being generated. To support the rich structure formats arising in diverse applications, a flexible mechanism is needed to specify and check the constraints. Context-free grammar (CFG) [Cho56, Poe22, Sch21d] provides a general approach for defining structures through a set of rules. Each rule contains a sequence of characters or other rules, allowing recursive composition to represent complex structures. Compared to alternative formats such as regular expressions, CFGs offer greater flexibility by allowing recursive structures, making them suitable for describing common languages such as JSON, SQL, and domain-specific languages (DSLs).

However, naively applying CFG to constrained decoding is not efficient because of its flexible nature. First, each decoding step needs to interpret CFG for every possible token in the vocabulary, which can be as large as 128k in Llama 3.1 [Dub24]. Additionally, CFG interpretation requires a stack state that tracks the recursive rules matched so far, making it impossible to precompute and cache all combinatorial combinations of stack patterns ahead of time. Finally, each token in the LLM generation comprises multiple characters, which may cross the boundaries of grammar elements and cause further recursion or stack pop during runtime execution. The misaligned boundaries bring the need to handle them carefully during grammar execution.

In this paper, we introduce XGrammar, a flexible and efficient structured generation engine for large language models to address the above challenges. XGrammar builds a byte-level pushdown automaton to represent context-free grammars (CFGs). Our main insight(shown in [Figure 1](#figure-01)) is to categorize the tokens into **context-independent** tokens that can be decided only from the local context of automata and **context-dependent** tokens that require the entire stack state. We precompute the token correctness for all context-independent tokens and store them in an adaptive token mask cache with specific storage formats tailored to each automata location. We also build algorithms to expand the context of each local rule and reduce the number of context-dependent tokens. Additionally, we build a persistent stack-based system to enable rapid state branching and rollback, expediting context-dependent token checks and cache preprocessing. Finally, we co-designed the grammar engine with LLM inference engines to overlap the grammar computations with GPU computations, bringing minimal overhead for structured generation.

Evaluation shows that XGrammar can achieve up to 100x reduction in per-token latency for context-free grammar compared to current state-of-the-art methods. Additionally, the XGrammar-integrated LLM serving engine for Llama-3.1 models achieves up to an 80x speedup in end-to-end LLM serving with structured output on the H100 GPU. We are open-sourcing XGrammar and integrating it into major open-source LLM frameworks.

The main contribution of this paper is as follows:

- We introduce an adaptive token mask cache that leverages context-independent tokens and significantly reduces mask generation overhead.

- We design a persistent execution stack that enables fast rollback operations, rapid state branching, and rollback, expediting context-dependent token processing.

- We built an efficient grammar engine co-designed with the LLM serving framework to achieve minimal structured generation overhead.

<span id="section-2"></span>

## 2 Background

<span id="section-2-1"></span>

### 2.1 LLM Constrained Generation

<span id="figure-02"></span>

![Figure 2. Constrained decoding with per-token mask. The per-token mask prevents LLM from generating tokens that would be invalid according to the structure at that step.](../../papers/xgrammar/figure-02.png)

**Figure 2.** Constrained decoding with per-token mask. The per-token mask prevents LLM from generating tokens that would be invalid according to the structure at that step.

Large Language Models (LLMs) like GPT-4 [Ope23], Llama [Dub24], and Mistral [Jia23] generate text in an auto-regressive manner, predicting one token at a time based on preceding sequence of tokens. The process starts with an initial prompt and continues as the model iteratively appends tokens until the response is complete. In LLMs, tokens serve as the basic input and output units. Each token represents a fixed string but may not correspond to a complete semantic unit or may break a Unicode character [Wan20e], creating challenges for structured text generation. At each step, the model produces a logits vector across its vocabulary, which is then converted into a probability distribution using the softmax function [Bri89]. A sampler then selects the next token from this distribution.

Constrained decoding guides the structure of LLM-generated text by restricting available tokens at each step, as illustrated in [Figure 2](#figure-02). At each step, tokens that would violate the required structure are identified as invalid. Their logits are set to $-\infty$, effectively assigning them zero probability after the softmax operation and preserving the relative probabilities of other valid tokens. This ensures that only valid tokens are sampled. Efficiently identifying and masking invalid tokens is essential, as it directly impacts generation speed.

<span id="figure-03"></span>

![A recursively composed CFG for arrays and strings, together with two possible stacks for matching a string.](../../papers/xgrammar/figure-03.png)

**Figure 3.** Up: A context-free grammar for arrays and strings that can be recursively composed. This CFG is converted into the pushdown automata in [Figure 1](#figure-01). <code>[^"\\]</code> denotes every character except <code>"</code> and <code>\\</code>. Down: Two possible matching stacks for matching the string <code>["a</code> to the CFG. Each stack represents a possible expansion of the rules in the CFG. The edges and nodes in the stack correspond to the transitions and states in the PDA in [Figure 1](#figure-01).

<span id="section-2-2"></span>

### 2.2 Context-free Grammar and Pushdown Automata

Context-free grammar (CFG) [Cho56] is widely used to define structures in structured generation. With an example shown in [Figure 3](#figure-03), CFG contains multiple rules, each including characters or references to other rules, allowing recursive composition to define complex structures. This makes CFG suitable for languages such as JSON, SQL, and various domain-specific languages. CFG’s recursive nature provides greater expressive power than simpler patterns, such as regular expressions, which are also frequently applied in LLM structured generation.

Pushdown automata (PDA) [Sch63, Eve63] are typically used to recognize languages generated by CFGs, as they employ a stack to manage nested structures. In this paper, we use a definition of PDA that is equivalent to the original one, but more conducive to explaining the algorithm. An example of PDA is shown in [Figure 1](#figure-01), and its stacks are shown in detail in [Figure 3](#figure-03). A PDA consists of multiple finite state automata (FSA), each representing a grammar rule, with the stack handling recursive rule expansions. The transitions in the FSA include two types: character edges, which accept specific characters, and rule reference edges, which allow recursive entry into other rules. A formal definition of the PDA is provided in [Section 7](#section-7). To match a string, the PDA begins with the main rule, recursively expanding child rules by pushing rule-reference edges onto the stack; once a rule is fully matched, it pops the stack to return to the previous rule. The top of the stack holds the current node reached. If the grammar is non-deterministic, meaning there can be multiple possible transitions in the PDA for the same input character, the PDA can maintain multiple parallel stacks for each path, ensuring flexibility. However, the unbounded stack length results in an infinite number of possible states, making it impractical to precompute token masks for all scenarios, thus posing challenges for efficient constrained decoding.

<span id="section-3"></span>

## 3 XGrammar

As shown in [Figure 1](#figure-01), XGrammar utilizes a byte-level pushdown automaton to interpret the context-free grammar. This byte-level design allows each character edge to include one or more bytes, handling irregular token boundaries and supporting tokens containing sub-UTF8 characters. The automaton’s structure is optimized to accelerate matching, as described in [Section 3.4](#section-3-4). In the preprocessing phase, we generate an adaptive token mask cache, as detailed in [Section 3.1](#section-3-1), which accelerates runtime mask generation by precomputing context-independent tokens. The effectiveness of this cache is further enhanced by context extension in [Section 3.2](#section-3-2). At runtime, the token mask cache quickly generates most of the mask, while the persistent execution stack in [Section 3.3](#section-3-3) efficiently processes the rest context-dependent tokens. Additionally, mask generation and LLM inference are overlapped in [Section 3.5](#section-3-5) to minimize the overhead of constrained decoding. Once the LLM generates a new token under the mask constraint, this token is then used to update the stack state of the pushdown automaton for the next mask generation.

<span id="section-3-1"></span>

### 3.1 Adaptive Token Mask Cache

<span id="figure-04"></span>

![Figure 4. An example for the token mask cache. Tokens are categorized into three types: context-independent (accepted), context-independent (rejected), and context-dependent. The first two types can be directly determined for mask generation at runtime.](../../papers/xgrammar/figure-04.png)

**Figure 4.** An example for the token mask cache. Tokens are categorized into three types: context-independent (accepted), context-independent (rejected), and context-dependent. The first two types can be directly determined for mask generation at runtime.

To accelerate the generation of the token mask cache, the adaptive token cache categorizes tokens into two types ([Figure 4](#figure-04)): context-independent tokens, which constitute the vast majority and can be pre-computed, and context-dependent tokens, which require slower, on-the-fly processing but are relatively few. This token classification relates to how tokens are validated by the pushdown automaton. We found that, considering the transition of the stack state, the process of matching tokens to the automaton can be divided into three categories:

1.  The matching process expands into a child rule, pushing new elements onto the stack.

2.  The matching process advances within the current rule, updating the stack top node to a new position.

3.  The matching process reaches the end of the current rule and returns to a parent rule, popping elements from the stack.

Validating tokens in the former two cases only relies on the stack top node, which represents the position within the current rule, so we define these tokens as *context-independent tokens*. The tokens in the third type, however, requires inspecting the entire running stack in validation, and are defined as *context-dependent tokens*. For every node of the pushdown automaton, there is a set of context-independent tokens with this node being at the top of the stack at runtime, and their validity can be determined ahead of time. Therefore, we precompute the validity of these tokens and store them in a cache with the stack top node as the key, which we refer to as the adaptive token mask cache. It also adaptively selects the most efficient storage format based on the cache’s contents, as explained in the next paragraph.

At runtime, we retrieve the validity of context-independent tokens directly based on the top of the stack to generate the token mask. The remaining few context-dependent tokens are validated by executing the pushdown automaton with the full stack. If parallel stacks exist due to the ambiguity of the grammar, the token masks for every stack is merged into a final token mask by finding the union of the accepted tokens in each mask. The computation for the token mask is significantly reduced because our method do not need to check context-independent tokens at runtime. Experiments show that context-dependent tokens account for only a minor proportion, amounting to less than 1% (1134 out of 128k) for the Llama-3.1 model using JSON grammar.

<span id="figure-05"></span>

![Figure 5. The adaptive storage format. In accept-heavy cases, we store the rejected tokens and context-dependent tokens. In reject-heavy cases, we store the accepted tokens and context-dependent tokens. In rare cases where two kinds of tokens are equal, we compress the accepted and rejected tokens into a bitset of the vocabulary size.](../../papers/xgrammar/figure-05.png)

**Figure 5.** The adaptive storage format. In accept-heavy cases, we store the rejected tokens and context-dependent tokens. In reject-heavy cases, we store the accepted tokens and context-dependent tokens. In rare cases where two kinds of tokens are equal, we compress the accepted and rejected tokens into a bitset of the vocabulary size.

**Adaptive storage.** The token mask cache adopts an adaptive storage format to reduce memory usage, as illustrated in [Figure 5](#figure-05). For each automaton node, the token mask cache divides the vocabulary into three parts: the accepted context-independent tokens, the rejected context-independent tokens, and the context-dependent tokens. Since these three parts together cover all tokens, it is sufficient to store only the two smaller subsets. We observe that, for a set of context-independent tokens, they tend to be either almost entirely accepted, namely *accept-heavy* cases, or almost entirely rejected, namely *reject-heavy* cases. This arises because, if wildcards can be matched from the current node, such as the wildcard `[^"\\]*` in the rule of string, nearly all tokens are valid; whereas if the node only accepts a few specific characters, nearly all tokens are invalid. Based on this observation, we designed the following adaptive storage format:

1.  For accept-heavy cases, we store the rejected context-independent tokens and context-dependent tokens in two arrays.

2.  For reject-heavy cases, we store the accepted context-independent tokens and context-dependent tokens in two arrays.

3.  For rare cases where the accepted and rejected tokens are roughly equal, we store the accepted and rejected context-independent tokens and compress them into a bitset matching the vocabulary size.

Thus, in both accept-heavy and reject-heavy cases, the adaptive storage format only requires storing a small subset of tokens, significantly reducing memory usage. In practice, we will enumerate the three storage types, calculate their respective costs, and choose the storage type with the smallest size. For Llama-3.1 model and JSON grammar, this adaptive storage method can effectively reduce the total memory usage to 0.2% (from 160 MB to 0.46 MB).

Additionally, when multiple parallel stacks exists, we need to merge the token masks. The merging algorithm of token masks is optimized based on storage type, as shown in [Algorithm 1](#algorithm-01). For an accept-heavy mask (many accepted tokens, storing only rejected tokens), it intersects the rejected tokens with $\mathit{PartialRej}$. For a reject-heavy mask (many rejected tokens, storing only accepted tokens), it combines accepted tokens with $\mathit{PartialAcc}$. In the final mask, the rejected tokens are the set difference $\mathit{PartialRej} \setminus \mathit{PartialAcc}$. This algorithm limits set operations to small token subsets, thus enhancing efficiency.

<span id="algorithm-01"></span>

**Algorithm 1: Efficiently Merge Token Masks.**

- **Input:** Token masks for $k$ parallel stacks $\{M_i=(\mathit{Acc}_i,\mathit{Rej}_i)\}_{i=1}^{k}$, vocabulary $\mathcal{V}$.
- **Output:** The final token mask $M=(\mathit{Acc},\mathit{Rej})$.
- **Initialize:** $\mathit{PartialAcc}\gets\emptyset$, $\mathit{PartialRej}\gets\mathcal{V}$.
- **For** $i=1$ **to** $k$:
  - **If** $M_i$ is accept-heavy:
    - $M_i$ only stores rejected token list $\mathit{Rej}_i$.
    - $\mathit{PartialRej}\gets\mathit{PartialRej}\cap\mathit{Rej}_i$.
  - **Else:**
    - $M_i$ only stores accepted token list $\mathit{Acc}_i$.
    - $\mathit{PartialAcc}\gets\mathit{PartialAcc}\cup\mathit{Acc}_i$.
- $M\gets\bigl(\mathcal{V}\setminus(\mathit{PartialRej}\setminus\mathit{PartialAcc}),\ \mathit{PartialRej}\setminus\mathit{PartialAcc}\bigr)$.

<span id="section-3-2"></span>

### 3.2 Context Expansion

<span id="figure-06"></span>

![Figure 6. The context expansion. Each rule obtains a set of expanded suffices, representing the set of strings that must be matched after completing this rule. The remaining unmatched part of the context-dependent tokens should either be a prefix of the expanded suffix or start with the expanded suffix. Otherwise, they are rejected.](../../papers/xgrammar/figure-06.png)

**Figure 6.** The context expansion. Each rule obtains a set of expanded suffices, representing the set of strings that must be matched after completing this rule. The remaining unmatched part of the context-dependent tokens should either be a prefix of the expanded suffix or start with the expanded suffix. Otherwise, they are rejected.

<span id="algorithm-02"></span>

**Algorithm 2: Extract the Expanded Suffix Automaton.**

- **Input:** Pushdown automaton $\mathcal{P}$, rule $R$.
- **Output:** Expanded context FSA $\mathcal{A}_R^{\mathrm{ctx}}$ for $R$.
- **Initialize:** $\mathcal{A}_R^{\mathrm{ctx}}$ as an empty FSA.
- **For** edge $s\xrightarrow{R}t$ in $\mathcal{P}$ referencing $R$:
  - Initialize $\mathcal{A}_{\delta}$ as an empty FSA and $\mathit{visited}\gets\{\}$.
  - Add node $t$ to $\mathcal{A}_{\delta}$.
  - $\mathrm{ExtractOne}(t,\mathcal{A}_{\delta},\mathit{visited})$.
  - $\mathcal{A}_R^{\mathrm{ctx}}\gets\mathrm{FSAUnion}(\mathcal{A}_R^{\mathrm{ctx}},\mathcal{A}_{\delta})$.
- **Function** $\mathrm{ExtractOne}(\mathit{start},\mathcal{A}_{\delta},\mathit{visited})$:
  - **If** $\mathit{start}$ is in $\mathit{visited}$:
    - **Return.**
  - Add $\mathit{start}$ to $\mathit{visited}$.
  - **If** $\mathit{start}$ is a final node in $\mathcal{P}$ **or** has an edge referencing another rule:
    - Mark $\mathit{start}$ as final in $\mathcal{A}_{\delta}$.
    - **Return.**
  - **For** edge $\mathit{start}\xrightarrow{c}\mathit{end}$ from $\mathit{start}$:
    - Add $\mathit{end}$ and $\mathit{start}\xrightarrow{c}\mathit{end}$ to $\mathcal{A}_{\delta}$.
    - $\mathrm{ExtractOne}(\mathit{end},\mathcal{A}_{\delta},\mathit{visited})$.

Although the adaptive token mask cache effectively reduces the number of tokens checked at runtime, checking all context-dependent tokens remains an efficiency bottleneck at runtime. To further reduce the number of context-dependent tokens, XGrammar introduces context expansion, which leverages the grammar’s context information to reject more context-dependent tokens during preprocessing, as shown in [Figure 6](#figure-06).

As described in the last section, a token is context-dependent when we reach the end of the current rule during matching, but there is still a remaining part of the token that requires further checking by returning to the parent rules. However, through analysis of the grammar, we can observe that in a large portion of the cases, the remaining part of a token is invalid. This is because, for many rules, when they reach their end, there are only a limited number of positions they can return to within their parent rules, and the set of strings that can be further matched from those positions is also limited.

Based on this observation, context expansion precomputes, for each rule, the set of strings that can be accepted after returning to the parent rules, called the *expanded suffix*. The remaining unmatched part of the context-dependent tokens should either be a prefix of the expanded suffix or start with the expanded suffix. Otherwise, they are rejected. This filtering process effectively reduces the number of context-dependent tokens by eliminating those that would fail in higher-level rule contexts. Applied to the Llama-3.1 model and JSON grammar, this technique reduces context-dependent tokens by 90% (from 1,134 to 120).

[Algorithm 2](#algorithm-02) describes the context expansion process that finds the expanded suffix of each rule. For a rule $R$, we utilize a finite state automaton (FSA) $\mathcal{A}_R^{\mathrm{ctx}}$ (ctx is the abbreviation for context) to represent the expanded suffix, and that is extracted from the pushdown automata. We first find all edges $e = (s, t)$ in the pushdown automata that references $R$ and belongs to rule $R'$. $R'$ is not necessarily different from $R$. Then we find a subgraph of the automaton of rule $R'$ starting from $t$ to represent the possible strings that can follow $R$ via depth-first search (DFS). However, we will not consider edges in the subgraph that reference other rules to avoid recursive references between rules, so the edges in the extracted subgraph will only have character labels. If a node has both character edges and edges referencing other rules, we will stop the search at this node. The extracted subgraph is then merged into $\mathcal{A}_R^{\mathrm{ctx}}$. This process is repeated for all rules, and the extracted $\mathcal{A}_R^{\mathrm{ctx}}$ is used to reject context-dependent tokens cannot match any string in it after finishing matching rule $R$.

Although we do not consider rule-referencing edges when extracting the expanded context automata, this algorithm can still extract many useful context information. That is because the inlining optimization introduced in [Section 3.4](#section-3-4) inlines fragment rules into their parent rules, reducing the need to check into child rules to reject context-dependent tokens.

<span id="section-3-3"></span>

### 3.3 Persistent Execution Stack

<span id="figure-07"></span>

![Figure 7. The persistent stack organizes multiple matching stacks from the current step, as well as stacks from previous steps, into a single tree. It reduces memory consumption and supports rolling the state back to previous steps.](../../papers/xgrammar/figure-07.png)

**Figure 7.** The persistent stack organizes multiple matching stacks from the current step, as well as stacks from previous steps, into a single tree. It reduces memory consumption and supports rolling the state back to previous steps.

As the grammar engine still needs to handle context-dependent tokens, we need to efficiently execute the pushdown automata for these tokens. Additionally, we also need to execute the pushdown automata for preprocessing the context-independent token sets for all positions in the pushdown automata. In both cases, we need to maintain multiple parallel stacks and branch out as we match the characters in each token. To support efficient state branching, we introduce the persistent execution stack [Dri89] to manage the multiple stacks and efficiently execute the pushdown automata. It can also manage the stacks from previous time points and enable the state rollback operation, effectively speeding up the execution of the pushdown automata on a set of tokens.

As shown in [Figure 7](#figure-07), the persistent execution stack manages a set of stacks, which are either the parallel stacks from the current time point or the stacks from previous time points, into a single tree, and every stack is represented by a path from the root node on the tree. The stack top node is stored as a pointer to the node in the tree. Since the stacks from adjacent time points often share most of the deeper elements and only a few nodes are pushed or popped, this merging avoids memory redundancy for storing multiple stacks. When matching a new character from a token, we may need to split the stack into multiple stacks due to the ambiguity of the grammar, each corresponding to a different expansion of grammar rules. In this case, we only need to split the branch for that stack instead of copying the whole stack, which reduces the overhead of state branching.

Additionally, the persistent execution stack enables fast state rollback by maintaining the stack from previous time points. At runtime, a sliding window of history is maintained. To roll back to a previous state, we only need to change the current stack pointers, which requires constant time. This rollback operation is particularly useful for checking a large set of tokens, as many tokens share a common prefix with other tokens, such as `read`, `ready`, and `reader` all sharing the prefix `read`. All the checked tokens are sorted in lexicographical order to find the maximum length of the common prefixes. Then the tokens are checked one by one, and before checking each token, the state rolls back to just after the common prefix with the previous token. Therefore, we can avoid the redundant checks of these common prefixes, reducing the number of characters that need to be checked. For Llama-3.1 model and JSON grammar, this approach reduces the number of characters that need to be checked across the entire vocabulary to 30%, significantly speeding up the preprocessing stage.

**The rollback operation enables more applications with efficient structured generation.** There are many LLM applications that involve rolling back the output to a previous token. For instance, the jump-forward decoding [Yin24a] requires retokenization, which involves rolling back some tokens in the context and then inserting new tokens. To ensure structured generation can continue after rolling back tokens, we can roll back the automaton state simultaneously with the output token rollback. There are also many LLM applications that requires LLMs generate in a tree structure, such as in Tree-of-thought [Yao23a], SGLang [She24], and the speculative model in the speculative decoding algorithm SpecInfer [Mia24a]. We can maintain the automata state for every branch of the output tree, and when the output branches, we can quickly split the automaton state, maintaining separate matching states for each output branch. This branching is fast because we only need to maintain the stack top pointer on the tree for every branch. Therefore, the persistent execution stack enables us to ensure efficient structured generation for all these applications.

<span id="section-3-4"></span>

### 3.4 Pushdown Automata Structure Optimizations

We will perform additional optimizations to improve the structure of pushdown automata to speed up the efficiency of final execution. These optimizations draw from traditional compiler optimization concepts, but we find them particularly useful for efficient constrained decoding.

**Rule inlining.** There could be many fragment rules, i.e. rules with only a few elements, in the specified context-free grammar, which are then converted into small FSA in the pushdown automaton. On the one hand, this increases the ambiguity of the grammar since we need to inspect into these fragment rules and check during the execution of the pushdown automata. On the other hands, during context expansion, references to fragment rules are not considered, so the extracted context automata will be smaller. We will miss the opportunity to reject context-dependent tokens based on the structure of these fragment rules.

To address this issue, we introduce an automatic inlining strategy [Sch77] for fragment rules. We iteratively pick rules that do not reference other rules and inline them into the parent rules. To avoid the explosion of the automaton size, we limit the size of the inlined rule and the size of inlined result to constants. This inlining process almost eliminated fragment rules, thereby improving the efficiency of token checking and enhancing the effectiveness of the context expansion.

**Pushdown automata node merging.** For pushdown automata, in many cases, the ambiguity comes from multiple outward edges of a node with the same label. When matching tokens, if we arrive at this node, and the next character just matches the label, the matching stack will be split into multiple stacks, one for each outward edge. The increase in the number of stacks increases the computation as we need to check the context-dependent tokens for each stack and merge the token masks. To reduce this kind of ambiguity, the node merging algorithm merges the subsequent nodes that satisfy: a) they are pointed to by edges with the same label originating from the same point b) they are not pointed to by other edges.

Additionally, the epsilon edge also increases the ambiguity of the matching process. An epsilon edge $s \xrightarrow{\epsilon} t$ in the automata means that the matching process can directly move from $s$ to $t$ without consuming any characters. If the matching process arrives at $s$, the execution stack will split into two stacks, one with $s$ at the top and the other with $t$, both of which can continue matching. To reduce this kind of ambiguity, the node merging algorithm also merges the nodes $s$ and $t$ into a single node, as long as $s$ has no other outward edge or $t$ has no zero inward edge.

These two optimizations preserves the equivalence of the automaton, but reduces the number of nodes and edges. At runtime, the number of stacks and the computation required for token checking are reduced, speeding up the mask generation process.

<span id="section-3-5"></span>

### 3.5 Overlapping Mask Generation and LLM Inference

<span id="figure-08"></span>

![Figure 8. Overlapping building the mask cache with LLM prefilling, and mask generation with LLM decoding to minimize the overhead.](../../papers/xgrammar/figure-08.png)

**Figure 8.** Overlapping building the mask cache with LLM prefilling, and mask generation with LLM decoding to minimize the overhead.

With the optimizations mentioned above, the token mask generation process is significantly accelerated, but it still requires CPU computation. To further eliminate the overhead of constrained decoding, we overlap the computation for mask generation with the LLM inference process, as shown in [Figure 8](#figure-08). We observed that the mask generation process and LLM inference process can be overlapped. That is because the mask generation only requires CPU, and only depends on the previously generated tokens. The LLM inference process except the sampling stage only requires GPU, and also only depends on the previously generated tokens. Therefore, we can parallelize the mask generation process on the CPU with the LLM inference process on the GPU. We will synchronize before sampling, and the GPU will obtain the mask from the CPU and perform masked sampling to generate the new token. Additionally, the preprocessing stage can also be overlapped with the LLM prefilling stage, where the LLM processes the prompt. This orchestration between CPU and GPU ensures that the token restrictions are applied seamlessly, with almost zero overhead for LLM inference. In practice, the time for mask generation is less than the time for LLM inference, so the mask generation process will not become the bottleneck of the generation process.

<span id="section-4"></span>

## 4 Evaluation

We implement XGrammar in 12,000 lines of core C++ code, and we provide Python bindings to facilitate seamless integration with LLM inference frameworks. In this section, we evaluate XGrammar to answer the following questions:

- Can XGrammar efficiently support each step of constrained decoding? ([Section 4.1](#section-4-1))

- Does XGrammar achieve minimal overhead for end-to-end structured generation in LLM serving? ([Section 4.2](#section-4-2))

- How effective is each optimization technique introduced in XGrammar? ([Section 4.3](#section-4-3))

- How does XGrammar effect downstream structured generation tasks? ([Section 4.4](#section-4-4))

<span id="section-4-1"></span>

### 4.1 Mask generation efficiency

This section evaluates the efficiency of mask generation to measure the overhead introduced by constraint decoding. We first assess regex-based methods using JSON schemas, which can be converted into regex. To test more complex cases beyond regex capabilities, we evaluate context-free grammars, including unconstrained JSON (from ECMA-404 [Ecm13]), XML (based on the XML 1.0 standard [Bra08]), and a Python DSL (adapted from the Python Grammar Specification [Pyt24a]). Unconstrained JSON cannot be handled by regex-based methods due to its support for arbitrarily nested lists and objects. The Python DSL covers basic control flow (if, for, while) and data types (str, int, float, bool) but ignores indentation. For JSON schema and unconstrained JSON, we use the JSON-mode-eval dataset [Nou24], and for XML and Python, we use a synthetic dataset. For baselines, we choose three popular constrained generation libraries: Outlines [Wil23] (v1.0), the grammar engine in llama.cpp [Ger23a] (b3998), and lm-format-enforcer [Gat25] (v0.10.9, a regex-based method that does not support CFG). All methods are evaluated on Llama-3.1-8B-Instruct using an AMD Ryzen 9 7950X CPU and an NVIDIA RTX 4090 GPU.

The results are shown in [Figure 9](#figure-09). XGrammar consistently achieves the lowest latency across all tasks, with under 40 µs per token for JSON Schema and CFG (JSON), and under 200 µs for XML and Python DSL. It delivers up to 3x speedup on JSON Schema and over 100x on CFG, compared to the best baseline in each case.

<span id="figure-09"></span>

![Figure 9. Per token mask generation latency. XGrammar consistently outperforms existing constrained decoding libraries.](../../papers/xgrammar/figure-09.png)

**Figure 9.** Per token mask generation latency. XGrammar consistently outperforms existing constrained decoding libraries.

<span id="section-4-2"></span>

### 4.2 End-to-End LLM Engine Evaluation

<span id="figure-10"></span>

![Figure 10. End-to-end evaluation on Llama 3.1 inference with structured constraints. Some results with a batch size of 32 are not reported because their API call time exceeded the API timeout limit of 600 seconds.](../../papers/xgrammar/figure-10.png)

**Figure 10.** End-to-end evaluation on Llama 3.1 inference with structured constraints. Some results with a batch size of 32 are not reported because their API call time exceeded the API timeout limit of 600 seconds.

<span id="table-01"></span>

![Table 1. End-to-end structured generation efficiency across different models, measured in time per output token (ms) on the JSON Schema task and the Llama-3.1 8B model. XGrammar demonstrates superior performance across different models.](../../papers/xgrammar/table-01.png)

**Table 1.** End-to-end structured generation efficiency across different models, measured in time per output token (ms) on the JSON Schema task and the Llama-3.1 8B model. XGrammar demonstrates superior performance across different models.

This section evaluates XGrammar in LLM serving scenarios. We co-design XGrammar with the serving engines using the overlapping technique introduced in [Section 3.5](#section-3-5). We integrate it into widely used end-to-end LLM serving engines, including the C++-based MLC-LLM [Mlc23] and the Python-based SGLang [She24], showcasing XGrammar’s adaptability and efficiency across different deployment environments.

We first compare the efficiency of several LLM engines that support structured generation, including vLLM [Kwo23] (v0.6.3) with Outlines and llama.cpp with its built-in grammar engine. Efficiency is measured by the average time per output token (TPOT), which reflects the overhead of applying constraints during token generation. The evaluations are conducted using Llama-3.1-8B-Instruct under both JSON schema and CFG (unconstrained JSON). All tests are run in an online serving setting with fixed batch sizes, on hardware with an AMD EPYC 7R13 CPU and an NVIDIA H100 GPU. To ensure a fair comparison, we set a fixed maximum output length. On average, the input has 139 tokens, and the generated output has 53 tokens.

The experiment results are shown in [Figure 10](#figure-10). XGrammar achieves the best TPOT among all baselines for both JSON Schema and CFG. The computation of vLLM and llama.cpp is hindered by their grammar engines’ longer preprocessing and per-token processing time. The decrease in TPOT speed in vLLM becomes particularly noticeable with larger batch sizes. This is because a larger batch size leads to higher throughput, putting greater pressure on grammar processing on the CPU side. Overall the XGrammar-based structured generation solutions can bring up to 80x output token rate compared to existing solutions. This proves the effectiveness of XGrammar’s system optimizations and its co-design with the serving engine.

We also study the effectiveness of XGrammar across different models in end-to-end serving scenarios. As the result shown in [Table 1](#table-01), for various models, SGLang integrated with XGrammar consistently outperforms its integration with Outlines, demonstrating the robustness of XGrammar’s effectiveness across model architectures. This experiment also provides direct evidence that XGrammar outperforms other constrained decoding library when running on the same serving engine.

Additionally, we examine the overhead of XGrammar’s constrained decoding in end-to-end scenarios by measuring performance on the MLC-LLM engine with and without XGrammar. As shown in [Table 2](#table-02), enabling XGrammar enhances output quality with nearly zero overhead in TPOT. This is attributed to efficient mask generation and the overlapping of grammar processing with GPU execution.

<span id="table-02"></span>

![Table 2. The impact on performance of enabling and disabling XGrammar tested on the MLC-LLM Engine and the Llama-3.1 8B model. TPOT (ms) is reported. XGrammar introduces minimal overhead to the serving engine with better generation quality.](../../papers/xgrammar/table-02.png)

**Table 2.** The impact on performance of enabling and disabling XGrammar tested on the MLC-LLM Engine and the Llama-3.1 8B model. TPOT (ms) is reported. XGrammar introduces minimal overhead to the serving engine with better generation quality.

<span id="section-4-3"></span>

### 4.3 Ablation Study of Optimization Techniques

<span id="table-03"></span>

![Table 3. Ablation study of optimization techniques in XGrammar.](../../papers/xgrammar/table-03.png)

**Table 3.** Ablation study of optimization techniques in XGrammar.

In this section, we investigate the impact of various optimizations introduced in xgrammar on mask generation performance, to better illustrate our design decisions. We begin by implementing a baseline using a pushdown automaton parser without any optimizations, where each token mask is generated by checking the entire vocabulary to determine whether parsing can proceed. Building on this baseline, we progressively add the optimizations described in this paper, namely node merging, the adaptive token mask cache, rule inlining, and context expansion. For each configuration, we measure the average mask generation time on the CFG (unconstrained JSON) task with the Llama-3.1 8B model and the json-mode-eval dataset. As shown in [Table 3](#table-03), the results show that the adaptive token mask cache has the greatest impact in terms of speedup, while other techniques, including node merging, rule inlining, and context expansion, also yield noticeable improvements.

<span id="section-4-4"></span>

### 4.4 Impact of XGrammar on Structured Generation Tasks

XGrammar can improve the generation quality of LLMs by ensuring that the output strictly adheres to the given format. We evaluate the impact of XGrammar on two strcutured generation tasks: function calling (i.e., JSON generation guided by a JSON schema) and XML code generation. For function calling, we use the json-mode-eval dataset, while for XML code generation, we rely on a synthetic dataset. We measure the syntactic correctness of the generated function calling outputs and XML code using LLaMA-3.1 8B. As shown in [Table 4](#table-04), XGrammar significantly improves generation accuracy. We observe that without XGrammar, the model often includes additional explanations alongside the intended code output, or the generated JSON contains an unexpected type. This makes the output unsuitable for direct use by downstream applications. XGrammar avoids this issue by enforcing grammar constraints.

<span id="table-04"></span>

![Table 4. Impact of XGrammar on structured generation tasks. XGrammar ensures 100% syntactic correctness of the generated outputs.](../../papers/xgrammar/table-04.png)

**Table 4.** Impact of XGrammar on structured generation tasks. XGrammar ensures 100% syntactic correctness of the generated outputs.

<span id="section-5"></span>

## 5 Related Work

Several works looked at algorithm improvements for structured generation. [Koo24] proposes an algorithm to convert character-level pushdown automata to token-level pushdown automata. [Wan23n] specifies LLM output structure through prompting. [Roz23, Cha23d, Li23o] explore finetuning LLMs for higher quality structured generation. XGrammar’s approach is orthogonal to these methods and can be combined with these approaches.

There has also been some previous work focusing on constrained decoding. Some methods use regex to represent syntax, such as lm-format-enforcer [Gat24], but they cannot handle more complex CFGs. Synchromesh [Poe22] and llama.cpp [Ger23a] utilizes LR parser and PDA respectively to handle CFG and generate the mask, but requires checking the entire vocabulary at runtime, which incurs large overhead. Outlines [Wil23] handles the grammar through a lexer and parser, and uses caching to accelerate mask generation, but it only considers the most recent lexer token, which may lead to incorrect judgments when LLM tokens span across multiple lexer tokens (Appendix A in [Koo24]). Syncode [Uga24] uses a lexer and parser with a cache spanning multiple lexer tokens, but requires all tokens to be processed offline, leading to substantial preprocessing overhead. In this work, XGrammar leverages a PDA to support CFGs and introduces an adaptive token mask cache to adaptively handle tokens during preprocessing and runtime, combining other system optimizations to achieve minimal overhead in both stages.

Guidance [Gui22], LMQL [Beu23a], SGLang [She24] provide flexible ways to declare the structures. XGrammar is complementary to these improvements and can be used as the backend engine to speedup their execution.

LLM serving engines [Mlc23, She24, Kwo23, Lig23c] employ various techniques to support efficient LLM generation for multiple concurrent users, including engine-level techniques such as continuous batching [Yu22a] for dyanmic request scheduling, and low-level KV cache technique PagedKVCache [Kwo23] for efficient memory management. Also, AICI [Mos24] proposes a CPU-GPU parallel computation paradigm to accelerate structured generation. These LLM serving engines can leverage XGrammar for efficient, structured generation on top of their existing LLM inference techniques.

<span id="section-6"></span>

## 6 Conclusion

We proposed XGrammar, a flexible and efficient structured generation engine for LLMs. XGrammar separates the vocabulary into context-independent tokens and context-dependent ones. It prechecks the context-dependent tokens and stores the result in an adaptive token mask cache. We further introduce a persistent stack to speed up the execution of context-dependent checks. Finally, we co-design the grammar engine with LLM inference to overlap grammar execution with GPU computation. Our system greatly speeds up the token mask generation process in token mask and enables zero overhead structure generation in end-to-end LLM inference flows. We hope our system can enable a broader range of structure generation across platforms.

## Acknowledgements

This work was supported in part by NSF award CNS-2211882, and gifts from OctoAI, Qualcomm, and CMU opensource software fellowships. We are grateful to Sasa Misailovic for his guidance as the shepherd of this paper. We also appreciate the valuable feedback and discussions from the DeepSeek, SGLang, TensorRT-LLM, vLLM, and WebLLM teams (listed alphabetically). Additionally, we thank Weihua Du, Haoran Peng, Xinyu Yang, Zihao Ye, Jieyu Zhang, Zhihao Zhang, and Ligeng Zhu for their insightful input and thoughtful conversations.

<span id="section-7"></span>

## 7 Formal Definition of the PDA Variant

In our paper, we define a variant of the pushdown automaton (PDA) that is equivalent to the original definition, but is designed to facilitate the description of the parsing algorithm and the construction of the token mask cache, since the keys of the token mask cache are precisely the states in this PDA. It is defined as the tuple

$$
P = \bigl( R, \Sigma, \{A_r\}_{r \in R}, q_{\text{main}}, \delta \bigr),
$$
where:

- $R$ is a finite set of grammar rules.

- $\Sigma$ is a finite input alphabet.

- For each rule $r \in R$, the corresponding finite state automaton is given by

  $$
  A_r = \bigl( Q_r, \Sigma \cup R, q^{\mathrm{start}}_r, F_r, \delta_r \bigr),

  $$
  where $Q_r$ is a finite set of states, $q^{\mathrm{start}}_r \in Q_r$ is the start state, $F_r \subseteq Q_r$ is the set of accepting states, and $\delta_r$ is the transition function defined over $Q_r$. The transition labels in $A_r$ are drawn from the alphabet $\Sigma \cup R$, which includes both input characters and rule references.

- $q_{\text{main}}$ is the start state corresponding to the main rule.

- $\delta$ is the global transition function that governs the operation of the PDA by handling two kinds of transitions:

  - *Character transitions*: When in a state $q \in Q_r$, reading an input symbol $a \in \Sigma$ may lead to a transition within the same automaton, i.e., $q \xrightarrow{a} q'$.

  - *Rule reference transitions*: When in a state $q \in Q_r$, a transition labeled by a rule reference to $s \in R$ allows the PDA to push the current return information onto the stack and jump to the start state $q^{\mathrm{start}}_s$ of $A_s$.

The parsing state is represented by a set of pairs $\{(s_i, q_i)\}$, where $s_i$ denotes the content of the stack (encoding return information), $q_i$ is the current state (with $q_i \in Q_r$ for some $r \in R$). There can be multiple such pairs because the pushdown automaton can contain non-deterministic transitions, which means there could be multiple possible parsing stacks and states. In the main body of the paper, for simplicity, we place the current state $q_i$ at the top of the stack. Thus, the parsing state is represented as a set of parsing stacks.

We now describe a formal transformation that converts the above variant PDA definition into a standard PDA. To obtain a standard PDA, we construct a new pushdown automaton

$$
P' = \bigl( Q, \Sigma, \Gamma, \delta', q_0, F \bigr)
$$

as follows. The state set $Q$ is defined as the union of all states from the FSAs:
$$
Q = \bigcup_{r \in R} Q_r,
$$
and we set the initial state to be $q_0 = q_{\text{main}}$. The stack alphabet $\Gamma$ is chosen to record return information; we define
$$
\Gamma = \{\, (r, q) \mid r \in R,\; q \in Q_r \,\},
$$
so that each symbol $(r, q)$ encodes the context of a rule call, namely the originating rule and the return state.

The transition function $\delta'$ of the standard PDA is then defined to simulate the behavior of the variant PDA. For each character transition in some $\delta_r$, if
$$
q \xrightarrow{a} q' \quad \mathrm{with}\ a \in \Sigma\ \mathrm{and}\ q, q' \in Q_r,
$$
we include in $\delta'$ the transition
$$
\delta'(q, a, \gamma) \ni (q', \gamma) \quad \mathrm{for\ all}\ \gamma \in \Gamma.
$$
For each rule reference transition in $\delta_r$, if
$$
q \xrightarrow{s} q' \quad \mathrm{with}\ s \in R,
$$
we simulate the recursive call by defining an $\epsilon$-transition that pushes the return information and transfers control to the called rule. Formally, we set
$$
\delta'(q, \epsilon, \gamma) \ni \Bigl( q^{\mathrm{start}}_s, \, (r,q') \cdot \gamma \Bigr) \quad \mathrm{for\ all}\ \gamma \in \Gamma,
$$
where $q$ belongs to the automaton $A_r$, and $(r,q') \cdot \gamma$ denotes the stack obtained by pushing $(r,q')$ onto $\gamma$. Finally, when an automaton $A_r$ reaches an accepting state $q \in F_r$, the standard PDA simulates the return from a recursive call by popping the top of the stack. That is, if the current stack has the form $(r',q') \cdot \gamma$, we define
$$
\delta'(q, \epsilon, (r',q')) \ni (q', \gamma).
$$
The set of accepting states $F$ is defined as those states in $\bigcup_{r \in R} F_r$ that are reached with an empty stack.

This construction shows that character transitions in the variant PDA directly correspond to state transitions without stack operations in $P'$, while rule reference transitions correspond to stack push and jump operations. Similarly, completing the match of an FSA and returning to the parent rule is achieved by a stack pop and a transition to the stored return state. In this way, the standard PDA $P'$ exactly simulates the recursive behavior of our variant PDA.

<span id="section-8"></span>

## 8 Synergy between XGrammar and Jump-forward Decoding

Jump-forward decoding is a technique designed to accelerate structured generation. When, based on the current input, the following output can be deterministically inferred from the grammar, it bypasses LLM decoding and sampling by directly tokenizing and appending the output to the context. This improves the end-to-end efficiency of LLM generation. This technique is orthogonal to the constrained decoding adopted by XGrammar. XGrammar further supports jump-forward decoding and demonstrates that the two can be effectively combined to further improve efficiency.

We evaluate performance using the Llama-3.1-8B-Instruct model served through the SGLang engine, running on a machine with an AMD Ryzen 9 7950X CPU and an NVIDIA RTX 4090 GPU. All experiments are conducted with a batch size of 1. We compare our method, XGrammar, against the existing decoding backend Outlines, testing both with and without jump-forward decoding enabled. We measure time per output token as the evaluation metric.

As shown in [Figure 11](#figure-11), XGrammar consistently outperforms Outlines, and achieves the best efficiency when combined with jump-forward decoding—demonstrating its ability to better leverage structural constraints for faster generation.

<span id="figure-11"></span>

![Figure 11. The performance comparison with and without jump-forward decoding on the JSON schema task and the SGLang engine. XGrammar combined with jump-forward can achieve optimal TPOT.](../../papers/xgrammar/figure-11.png)

**Figure 11.** The performance comparison with and without jump-forward decoding on the JSON schema task and the SGLang engine. XGrammar combined with jump-forward can achieve optimal TPOT.

<span id="section-9"></span>

## 9 Cross-platform Deployment of XGrammar

<span id="figure-12"></span>

![Figure 12. End-to-end performance comparison between structured generation with XGrammar and unstructured generation in browser JavaScript environment.](../../papers/xgrammar/figure-12.png)

**Figure 12.** End-to-end performance comparison between structured generation with XGrammar and unstructured generation in browser JavaScript environment.

We study bringing XGrammar to a wide variety of platforms. We leverage Emscripten  [Zak11] to compile XGrammar into WebAssembly  [Haa17] and build a JavaScript binding. This approach enables XGrammar to run in client-side browsers on portable devices like laptops and mobile phones. We further integrate the web-binding with the in-browser LLM inference framework WebLLM  [Web23b] to enable structured generation.

We evaluate the end-to-end performance with the JSON-mode-eval dataset, using 4-bit quantized models Llama-3.1-8B-Instruct [Dub24] on a MacBook Pro M3 Max (MacOS 14.5) with Google Chrome, and Qwen-2.5-0.5B-Instruct [Yang24b] on an iPhone 14 Pro Max (iOS 18) with Safari.

The results are shown in [Figure 12](#figure-12). We compare the time to first token (TTFT) and time per output token (TPOT) between structured generation with XGrammar and non-structured generation while ensuring the number of generated tokens is the same. The results show that XGrammar brings close to zero overhead in both settings, enabling a great potential to support future on-device agents with high performance.
