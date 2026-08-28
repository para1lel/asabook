---
title: 'Formal Conjectures'
createTime: 2026/08/28 15:15:00
permalink: /en/papers/formal-conjectures/
pageClass: paper-reading
---

> [Moritz Firsching](https://firsching.ch/) [+equal], [Paul Lezeau](https://sites.google.com/view/paul-lezeau/home/) [+equal], [Salvatore Mercuri](https://smmercuri.github.io/) [+equal], [Miklós Z. Horváth](https://miklos.ai/) [+equal], [Yaël Dillies](https://yaeldillies.github.io/), [Calle Sönne](https://dblp.org/pid/437/6124), [Eric Wieser](https://x.com/EricWieser), [Fred Zhang](https://fredzhang.me/), [Thomas Hubert](https://dblp.org/pid/211/7107), [Blaise Agüera y Arcas](https://www.blaiseaguera.com/), and [Pushmeet Kohli](https://dblp.org/pid/94/248). First submitted to arXiv on May 13, 2026; the current version is v1, dated May 13, 2026. [Formal Conjectures: An Open and Evolving Benchmark for Verified Discovery in Mathematics](https://arxiv.org/abs/2605.13171v1). <a href="/paper/formal-conjectures.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2605.13171). [TeX source](https://export.arxiv.org/e-print/2605.13171v1). The original PDF remains authoritative for the exact print layout and bibliography.

[+equal]: Equal contribution. Correspondence to: `firsching@google.com`.

## Abstract

As automated reasoning systems advance rapidly, there is a growing need for research-level formal mathematical problems to accurately evaluate their capabilities.
To address this, we present Formal Conjectures, [+repository] an evolving benchmark of currently 2615 mathematical problem statements formalized in Lean 4.
Sourced from areas of active mathematical research, the dataset features 1029 open research conjectures providing a zero-contamination benchmark for mathematical proof discovery, and 836 solved problems for proof auto-formalization.
Notably, the repository provides a structured interface connecting mathematicians who formalize and clarify problems with the AI systems and humans attempting to solve them.
Demonstrating its immediate utility, the benchmark has already been leveraged to make new mathematical discoveries, including the resolution of open research conjectures.
We describe our approach to ensuring the correctness of these formalizations in a collaborative open-source project where contributions stem from an active community.
In this framework, AI-generated proofs and disproofs serve as a valuable auditing mechanism to iteratively improve the fidelity of the benchmark.
Finally, we provide a standardized evaluation setup and report baseline results on frozen evaluation subsets, demonstrating a climbable signal that measures the current frontier of automated reasoning on research-level mathematics.

[+repository]: [Formal Conjectures repository](https://github.com/google-deepmind/formal-conjectures).

<span id="section-1"></span>

## 1 Introduction

As automated reasoning systems, including large language models (LLMs), continue improving their capabilities, the benchmarks used to measure their progress face significant challenges.
In the domain of (formal) mathematics, many existing evaluation frameworks suffer from: **(i) Data leakage:** As solutions to benchmark problems appear online, it is difficult to distinguish genuine reasoning from memorization. This is a challenge for benchmarks like [Zhe22a, Tso24, Jia25f, Yu25j]; **(ii) Secretive evaluation:** To combat leakage, some benchmarks require that the evaluation set be kept private, which hinders open and reproducible research [Cho25d, Gla24, Cen26]; **(iii) Oversimplified success criteria:** Many benchmarks rely on verifying a simple, machine-checkable final answer (such as a number), which fails to capture the complex, multi-step reasoning inherent in a full mathematical proof [Hen21, Gla24]; and **(iv) Saturation:** As models improve, some benchmarks become saturated. For instance, MiniF2F [Zhe22a] is now routinely solved with over 99%; see [Hub26, Che25h, Lin25j].

To address these limitations, we introduce **Formal Conjectures**, an open-source and growing collection of mathematical problems formalized in Lean 4 [Mou21] using Mathlib [Mat20b].
The benchmark focuses on *open conjectures* as the primary challenge, providing a zero-contamination testbed where solutions cannot be found in existing training data.
The secondary challenge consists of (informally) solved problems for auto-formalization, which do not share the same zero-contamination guarantees.
We formalize all problems in a formal language to allow for objective, rigorous, and automated evaluation: a proposed solution is definitively correct if and only if its corresponding proof is accepted by the kernel without relying on forbidden axioms like `sorry`.
Our choice of Lean 4 is pragmatic: Mathlib, the largest formal mathematics library available, is essential for stating a wide variety of advanced conjectures.
While we focus on Lean 4, the core principles of our benchmark are language-agnostic and extendable to other formal systems.

This paper details the construction and methodology of the **Formal Conjectures** benchmark.
The benchmark is publicly available under the Apache 2.0 license and has already been utilized for real-world mathematical discovery [Ale25, Goo26]. In summary, our main contributions are:

- **The Formal Conjectures Benchmark:** A large-scale, evolving dataset of 2615 mathematical statements in Lean 4, including 1029 open conjectures for zero-contamination proof discovery and 836 solved problems for auto-formalization.
- **A Unified API for Mathematical Reasoning:** A repository interface connecting mathematicians with an ecosystem of automated solvers to facilitate simultaneous statement auditing and rapid dissemination of problems to AI systems.
- **Methodology for Statement Formalization:** We introduce a collaborative pipeline and a three-level taxonomy of misformalizations (Translation, Underspecified, and Source) to ensure formalization fidelity. This includes the `answer(sorry)` mechanism, which decouples the mathematical value discovery from proof verification. We provide insights from 291 fixed cases of research-level formalization.
- **A Standardized Evaluation Framework:** We provide a rigorous evaluation setup using the Lean 4 kernel to verify multi-step reasoning without relying on oversimplified numerical answers. We establish frozen subsets, `FC100SolvedSet1` and `FC100OpenSet1`, to enable stable, reproducible model comparisons.

<span id="section-2"></span>

## 2 Related Work

Our work intersects mathematical reasoning benchmarks and formal theorem proving.

**General mathematical reasoning benchmarks.** Several benchmarks evaluate AI systems on advanced problem-solving without requiring formal proofs.
FrontierMath [Gla24] uses expert-crafted problems verified by numerical answers, which, while machine-checkable, do not fully capture the multi-step reasoning process.
In January 2026, Epoch AI launched *FrontierMath: Open Problems*, a website-only pilot of a small collection of unsolved research-level informal problems with evaluation offered as a service.
Humanity's Last Exam [Cen26] crowdsources expert questions but relies on answer-checking rather than proof verification.
MathArena [Bal25] evaluates LLMs on fresh competition problems to reduce contamination risk and assesses informal proof-writing.
The ARC Prize [Cho25d, Cho26] targets abstract reasoning, sharing our focus on measuring genuine reasoning over pattern matching.
Formal benchmarks, by contrast, allow automatic verification of full reasoning or proof.

**Formal mathematics benchmarks and automated theorem proving.** Interest in automated theorem proving has accelerated AI reasoning, leading to the rapid saturation of existing formal benchmarks.
MiniF2F [Zhe22a] pioneered high-school-level evaluation but became increasingly saturated as models improved.
This shift was defined by AlphaProof [Hub26] and AlphaGeometry 2 [Che25ac] reaching silver-medal performance at the 2024 IMO, where AlphaProof solved Lean problems and introduced the Formal-IMO [Hub26] dataset.
Systems like Aristotle [Ach25] and Seed-Prover [Che25h] achieved gold-medal equivalents at the 2025 IMO.
This capability leap cascaded to undergraduate math, where benchmarks like PutnamBench [Tso24] and Putnam 2025 are being saturated by models such as Seed-Prover 1.5 [Che25a] and Numina [Liu26a], and other industry systems.
While SorryDB [Let26] evaluates AI on practical theorem completion and FATE [Jia25f] targets varying difficulties of frontier algebra, the saturation of earlier benchmarks highlights an acute need for research-level evaluation: benchmarking reasoning on unsolved research-level mathematics.

**Adoption of Formal Conjectures.** Since its open-source release in May 2025, the repository has been adopted by both mathematicians and tool builders.
Boris Alexeev used a repository formalization to prove Erdős Problem 124 [Blo25] with Aristotle [Ach25].
Although the statement was initially misformalized and included an unintended extra hypothesis, Aristotle’s proof succeeded without relying on it.
Numerous subsequent successes such as [Ale25] highlight the repository’s value for discovery and demonstrate how AI-generated proofs serve as a vital auditing mechanism to improve formalization fidelity.
A DeepMind prover agent [Goo26] conducted a systematic evaluation of the full repository, solving several open problems.
To facilitate standardized, reproducible evaluation, this paper introduces named, versioned snapshots ([Section 4.1](#section-4-1)), moving beyond individual tests to systematic benchmarking across diverse methods.

<span id="section-3"></span>

## 3 Formal Conjectures

<span id="section-3-1"></span>

### 3.1 Problem Selection and Composition

<span id="figure-01"></span>

![Informal source and formal category distributions](../../papers/formal-conjectures/figure-01.png)

**Figure 1.** Left: informal source distribution. Right: formal category distribution, including open (1029) and solved (836) problems.

<span id="figure-02"></span>

![Repository growth over time](../../papers/formal-conjectures/figure-02.png)

**Figure 2.** Repository growth over time.

The benchmark's problems are sourced from diverse mathematical literature for broad coverage.
Key sources include the collection of problems posed by Paul Erdős, as catalogued on *erdosproblems.com* [Blo23]; the Kourovka Notebook of unsolved problems in group theory [Khu26]; the IQOQI Vienna list of open quantum problems [IQO17]; well-known Wikipedia conjectures; conjectures from recent publications in academic journals and on arXiv; and questions asked on MathOverflow.
The number of problems per informal source is in [Figure 1](#figure-01).
Furthermore, the repository has grown steadily since its initial open-source release, as visualized in [Figure 2](#figure-02).

The collection is divided into two primary categories: **Unsolved Conjectures**, open research problems where a formal or informal proof would represent a new mathematical discovery; and **Solved Problems**, established theorems with known *informal* proofs that benchmark an AI's ability to formalize existing arguments, though only a fraction currently have a *formal* proof available.
We include both categories not only because they independently serve as benchmarks for different tasks (proof generation and auto-formalization), but also because they are often intertwined.
When formalizing an open conjecture, it is natural to also state and prove simpler, solved variations.
For instance, a general conjecture about the existence of Hadamard matrices of size $4k$ for all valid $k \in \mathbb{N}$ can be presented alongside the solved cases (e.g., for all $k \leq 166$) and the first open case ($k = 167$).

Other sources of *solved* problems are collections of conjectures that include both open and solved questions. In those cases, we also formalize the statements of the solved cases. They are often related in subject to the open conjectures, and hence the formalization of their statements and the auto-formalization of their proofs support the open conjectures.

The collection of solved problems also includes simpler statements, ranging from trivial to textbook level, which help validate the correctness of the underlying formalizations.
Including such simpler problems helps detect incorrect formalizations of open problems: a disproof of an easy special case or known variant might indicate that some of the definitions used have been stated incorrectly.
See below for examples of how those initial misformalizations can lead to improved formal and informal problem statements.
We strive to always state the simplest open variant of a conjecture, as well as the conjecture in greater generality.

**The `@[category]` attribute.** To help navigate this collection, each statement is labeled with one of the following category tags: `research open` (an open research-level problem), `research solved` (a solved research-level problem), `textbook` (a textbook-level math problem ranging from high school to graduate level), `API` (statements developing foundational theory for a new definition intended for general reuse), and `test` (a statement serving as a sanity check).
The distribution of all categories is in [Figure 1](#figure-01) on the right.

**The `@[subject]` attribute.** Statements are also labeled with subject tags following the AMS Mathematics Subject Classification [Edi20].
The distribution by source collection and AMS subject classification is shown in [Tables 3](#table-03) and [4](#table-04) ([Section 7.1](#section-7-1)).
While number theory and combinatorics account for over half of all statements, the benchmark already spans over 10 AMS subjects (e.g., also quantum theory, or algebraic geometry), which we aim to broaden further.

<span id="section-3-2"></span>

### 3.2 Formalization Methodology

<span id="figure-03"></span>

![An open conjecture formalized as a Lean statement](../../papers/formal-conjectures/figure-03.png)

**Figure 3.** A statement in Formal Conjectures, an open conjecture from [Har74]. It includes an informal statement as a comment, tags, and a Lean statement.

All problems are presented as Lean theorems with informal statements. While our core goal is providing open statements with `sorry` placeholders for AI solvers to replace, we also include solved variants of conjectures and items from problem lists. For open problems, no proofs are provided. For solved items, we appreciate very short proofs or counterexamples (under 25-50 lines) that test the definitions. Longer proofs are excluded to keep the repository lightweight; instead, contributors are encouraged to host proofs in their own repositories and link to them using the `@[formal_proof]` mechanism described below.

We provide an example statement in [Figure 3](#figure-03).

**The `answer(sorry)` mechanism.** Many mathematical problems ask not just for a proof but for a specific *answer*: a number, a function, or a set.
To handle this, we introduce an `answer(sorry)` construct, implemented as a custom Lean term elaborator, that separates the task of discovering a computable answer from proving it correct.
For example, “What is the smallest $n$ such that $P(n)$?” is formalized with `answer(sorry)` as a placeholder that a solver must replace with a concrete value, reducing the problem to a verifiable proof obligation.
The construct also handles unknown truth values: when `answer(sorry)` appears in a biconditional, a solver replaces it with `True` or `False` to conjecture or refute a proposition.
See [Section 7.2](#section-7-2) for detailed examples and the full elaboration semantics.

**The “for Mathlib” pattern.** The project maintains a `FormalConjecturesForMathlib` directory containing 88 files of auxiliary definitions and lemmas not yet in Mathlib but needed for conjectures.
These results are asynchronously contributed upstream to Mathlib.
See [Section 7.3](#section-7-3) for details on the rationale and contents of this directory.

**The `@[formal_proof]` attribute.** To keep the repository lightweight, the `@[formal_proof]` attribute decouples problem statements from their solutions.
It supports three modes: (1) `formal_conjectures` for proofs solving exactly the type of the statement given here; it should point to a commit on top of a commit in the `main` branch of our repo, filling in the `sorry`;
(2) `lean4` for equivalent problems solved in Lean 4 elsewhere (e.g., in Mathlib or external repos); and
(3) `other_system` for problems solved in other systems like Lean 3, Isabelle, or Rocq.
This design links to external verification sources while keeping the repository maintainable.
The attribute applies to any solved category, though usage on a `research open` problem triggers a linter warning.
[Table 1](#table-01) summarizes proof coverage recorded via this attribute.

<span id="section-3-3"></span>

### 3.3 Misformalizations

It is notoriously difficult to formalize a mathematical problem *correctly* without providing a formal proof.
A *misformalization* is a formal statement that is incorrect.
To provide a rigorous foundation for benchmark quality, we introduce a taxonomy of misformalizations.
It includes the following levels, where in all cases the formal statement is incorrect.

1. **Translation**: the informal statement is accurate and explicitly phrased.
2. **Underspecified**: the informal statement is accurate but lacks detail.
3. **Source**: the informal statement is not as intended.

These levels are ordered with respect to the degree of contribution from the formalizer.
Level 1 misformalizations arise from the formalization only, while level 3 misformalizations arise from the informal text only.
Moreover, for a Lean expert who does not necessarily have domain expertise in the informal statements, levels are ordered with respect to difficulty to fix.
Note that level 2 and 3 misformalizations have been useful in clarifying informal statements in the literature.
An illustrative example is Erdős Problem 978: multiple attempts to formalize the statement led to a refinement of the original informal problem. [+erdos-978]
The original statement,
“Are there infinitely many $n$ for which $f(n)$ is $(k-2)$-power-free?”,
was found to require multiple additional hypotheses and became:
“If $k>3$, and for all primes $p$ there exists $n$ such that $p^{k-2}\nmid f(n)$, then are there infinitely many $n$ for which $f(n)$ is $(k-2)$-power-free?”
More broadly, informal texts may exclude trivial cases without explicitly describing them, while formalization requires these to be explicit.

[+erdos-978]: [Erdős Problem 978 discussion](https://www.erdosproblems.com/forum/thread/978).

Misformalizations can further be categorized into six types across the three levels: *syntactic*, *semantic*, and *misrepresentation* errors at the translation level; *implicit conventions* at the underspecified level; and *reporting* and *mathematical* errors at the source level.
The full taxonomy with definitions and representative pull requests is given in [Section 7.4](#section-7-4).
Across the repository a total of $291$ misformalizations have been fixed, with *misrepresentation* (48%) and *semantic* (35%) errors being the most common ([Table 5](#table-05)).
Detailed code diffs illustrating these misformalization examples can be found in [Section 7.5](#section-7-5).

<span id="section-3-4"></span>

### 3.4 Avoiding Misformalizations

<span id="figure-04"></span>

![Iterative pipeline for Formal Conjectures](../../papers/formal-conjectures/figure-04.png)

**Figure 4.** Iterative pipeline for Formal Conjectures. Formalized statements are tested by periodic AI runs, solving problems or triggering a revision loop.

In Formal Conjectures, every contribution undergoes mandatory code review by humans with both Lean expertise and relevant mathematical domain knowledge.
While early contributions to the repository were written entirely by hand, the authoring process has evolved: most recent submissions use AI tooling, including agentic coding assistants and auto-formalization systems.
To facilitate high-quality AI-assisted contributions, the repository provides an `AGENTS.md` file with structured guidance for these tools.
Similarly, the review process increasingly leverages AI: reviewers use language models to cross-check formalizations against informal sources, and AlphaProof runs on submissions to catch potential misformalizations before merging.

Beyond this review process, we employ a number of additional strategies to mitigate the risk of misformalizations across the dataset.
Firstly, we employ and encourage a test-based design for definitions.
Any new definitions introduced should be accompanied by a suite of proven `test` lemmas and `API` statements that are designed to establish expected behavior of the definition and mitigate the risk of edge-case errors.
Lean and `Mathlib`'s tooling (for example `decide`) can be employed to prove many of these `test` statements.
As of the [`bench-v1-lean4.27.0` release](https://github.com/google-deepmind/formal-conjectures/releases/tag/bench-v1-lean4.27.0), the repository contains 467 `test` statements and 155 `API` statements, amounting to 23.8% of the repository statements.

Secondly, automated theorem provers like AlphaProof regularly attempt proofs and disproofs across the repository ([Figure 4](#figure-04)); manually inspecting these results primarily reveals misformalizations.
In contrast to test-based design, which proactively reduces the risk of edge-case errors, this retroactive approach is capable of unearthing misformalizations across all categories as described in [Section 3.3](#section-3-3).
In addition, we use Gemini, AlphaProof, and other tools to automatically cross-check formalizations against their informal source texts, flagging potential discrepancies for human review.
Future work involves further automating these processes in GitHub CI in order to reduce the overhead of manual post-hoc fixes from the maintainers.

Thirdly, custom Lean linters built on Mathlib's framework enforce metadata and documentation standards across contributions.
These cover AMS classification tags, `answer(sorry)` usage, problem category annotations, and module docstrings.

Finally, as Formal Conjectures has grown in stature, community engagement and proof/disproof attempts from external automated theorem provers have been increasingly valuable.
Furthermore, insights and corrections discovered through this process are actively upstreamed to the original sources, e.g., prompting the maintainer of the Erdős problems website to regularly clarify and adjust the informal problem statements.

<span id="section-4"></span>

## 4 Benchmark Evaluation

<span id="section-4-1"></span>

### 4.1 Evaluation Framework

**Evaluation Paradigms.** Formal Conjectures provides a dynamic, rigorous benchmark for advanced automated reasoning premised on novel mathematical insight as the ultimate intellectual hurdle for complex problems. We define two distinct setups:

1. **Primary Goal: New Formal Proof Discovery.** Measuring AI's ability to discover formal proofs for open problems, these unsolved conjectures form a zero-contamination testbed for genuine mathematical discovery. Since no solutions exist in any training corpus, success provides a definitive signal of reasoning capabilities, removing the need for secretive evaluation sets common in other benchmarks.
2. **Secondary Goal: Proof Auto-formalization.** This track provides a climbable benchmark treating the non-trivial translation of established mathematics into formal code as a distinct, rigorous challenge. It measures a model's ability to work with Lean 4 and Mathlib to formalize known arguments.

**Evaluating Open Problems.** Evaluating open problems is definitive: as no formal solutions exist at release time, the first kernel-accepted proof provides a zero-contamination success signal.
However, since a proof may settle a misformalized statement rather than the intended problem, any result for an open conjecture triggers manual inspection for fidelity.

**Fixed Benchmark Subsets.** To enable stable model comparisons, we provide two frozen subsets of 100 problems each:
`FC100OpenSet1` (100 `research open` statements) and `FC100SolvedSet1` (100 `research solved` statements).
These subsets are defined in the repository files `FC100OpenSet1.lean` and `FC100SolvedSet1.lean`, which import exactly the corresponding theorem statements.
Because these files are compiled as part of the repository across all supported Lean versions, the problem sets remain fixed and comparable even as the repository evolves.
Further details on the construction of these subsets are in [Section 8.1](#section-8-1).

**Correctness.** Following established formal math benchmark methodologies [Zhe22a, Tso24], our evaluation leverages the Lean 4 kernel for an objective, rigorous, and automatable success criterion.
A proposed solution is a Lean proof term replacing the `sorry` placeholder.
Solutions are correct if and only if accepted by the kernel without relying on forbidden axioms, such as `sorry`.
This binary criterion eliminates human grading ambiguity and provides an indisputable ground truth.
While theoretically susceptible to foundational bugs in the Lean kernel or unintended axioms (e.g., a Lean Zulip thread, [+lean-kernel] which mentions a now-fixed exploit), in practice, kernel-level verification provides the highest mathematical rigor.
Consequently, all evaluations are automatically verifiable, allowing results and proof terms to be shared publicly for transparent, reproducible research.

[+lean-kernel]: [Lean Zulip discussion of the now-fixed soundness bug](https://leanprover.zulipchat.com/#narrow/channel/270676-lean4/topic/Soundness.20bug.3A.20hasLooseBVars.20is.20not.20conservative/with/520153084).

**Versioning and Reproducibility.** Evaluating on a living repository is challenging: dependency updates can alter statement semantics, and new Mathlib theorems may simplify problems.
To enable reproducible comparisons, we tag frozen benchmark snapshots using a two-part naming scheme: `bench-vN-lean4.X.Y`.
The *bench version* `vN` identifies a fixed problem set; the *Lean version* suffix pins the Mathlib tag (and hence Lean toolchain) against which statements compile and are evaluated.
Releases are issued every few months; when Lean versions bump, companion tags allow evaluating the same problem set against intermediate toolchains, e.g. `bench-v3-lean4.27.0`.
To preserve baselines, snapshots are immutable; misformalization fixes are never patched into existing versions but incorporated into the next (e.g., `bench-v(N+1)`).

**A Living Benchmark and Unified API.** Outside the frozen subsets, the repository operates as a dynamic benchmark that exhibits useful evaluation dynamics over time.
As new conjectures are added, solved, or corrected by the community, it naturally self-adjusts its difficulty, expands, and improves.
The hardest problems remain open, pushing the frontier of automated reasoning.
Crucially, the repository acts as a unified API: researchers can submit formalized conjectures to a single centralized hub for simultaneous exposure to all evaluating AI systems.
This obviates the need to manually engage with individual provers, ensuring broad and concurrent attempts at mathematical discovery.

<span id="table-01"></span>

![Proof coverage across statement categories](../../papers/formal-conjectures/table-01.png)

**Table 1.** Proof coverage across all statement categories. Proved in repo: the statement has a `sorry`-free proof in the repository; Linked proof: a proof is recorded via the `@[formal_proof]` attribute but is not present in the repository itself; With proof: either of the above.

**Open Problems Baseline.** For the primary proof-discovery task on open problems, the baseline for the first tagged release `bench-v1-lean4.27` is $0\%$ for all systems: by definition, no formal proof exists for any `research open` statement at the time of tagging (problems with known solutions are reclassified as `research solved`).
Over time, these open statements may receive formal or informal proofs.
Proofs that expose a misformalization ([Section 3.3](#section-3-3)) rather than settling the intended problem trigger revisions incorporated into the next version.
In subsequent releases, interim-solved problems are reclassified to `research solved`, with formal solutions receiving the `@[formal_proof]` attribute.
The new `research open` set combines remaining unsolved problems with new additions; thus, successive versions start with a baseline of $\geq 0\%$ solved statements and become increasingly more difficult.

**Proof Auto-Formalization Baseline.** For auto-formalization, the baseline is system-agnostic: we track all formal proofs contributed to the repository, regardless of origin.
These are tracked using the `@[formal_proof]` attribute ([Section 3.2](#section-3-2)), which records the proof system (`formal_conjectures` for proofs within the repository, `lean4` for proofs in external Lean 4 projects such as Mathlib, or `other_system` for proofs in Isabelle, Rocq, etc.) and links to the source.
[Table 1](#table-01) summarizes the current state. Notably, 17.3% of `research_solved` statements have a known formal proof (5.3% internal, 12.1% linked externally).

<span id="section-4-2"></span>

### 4.2 Illustrative Evaluation

We provide illustrative evaluations, with setup and cost details in [Section 8.2](#section-8-2).

<span id="table-02"></span>

![Results on the frozen FC100SolvedSet1](../../papers/formal-conjectures/table-02.png)

**Table 2.** Results on the frozen `FC100SolvedSet1`.

**Frozen Evaluation Subsets.** To ensure stable comparisons despite repository fluidity, we provide two frozen evaluation subsets, each containing 100 problems randomly sampled from the `bench-v1-lean4.27.0` tag: `FC100SolvedSet1` (proof auto-formalization) and `FC100OpenSet1` (proof discovery; 0% baseline).
By definition, `FC100OpenSet1` currently remains at a definitive 0% baseline across these evaluated methods, which highlights its status as a rigorous frontier for new discovery.
Results on `FC100SolvedSet1` ([Table 2](#table-02)) demonstrate a clear, climbable signal across both compute and model capabilities.
To provide a diverse evaluation, we evaluate both a slightly improved version of the AlphaProof system used in the 2024 IMO [Hub26] and a development version of a DeepMind prover agent [Goo26].
These are different systems rather than incremental versions of the same architecture.
Utilizing a constrained tree-search-only inference setup on v6e TPUs, AlphaProof achieves a solve rate of 45% on `FC100SolvedSet1` with a low compute configuration of 1,000 simulations per problem.
By increasing the compute budget to 16,000 simulations, the solve rate improves to 50%, demonstrating a clear scaling signal.
We emphasize that these evaluations are conducted using tree-search inference only and do not utilize AlphaProof's test-time reinforcement learning (TTRL) mechanism, which generates a curriculum to learn from during the proof finding process through weight updates.
Furthermore, utilizing a newer, development version of the DeepMind prover agent, the solve rate increases to 66%, illustrating that the benchmark effectively measures advancements across improved model architectures and methodologies.
Because running automated reasoning methods within Lean involves significant infrastructure overhead, we provide these specific systems as illustrative baselines to validate the benchmark's signal.
We encourage the community to report results against these subsets as well as the evolving main repository.

<span id="section-5"></span>

## 5 Discussion

**Broader Benefits.** Beyond its primary role, Formal Conjectures provides **precise mathematics** by enforcing a level of formal precision that clarifies the exact meaning of statements.
This creates a library of conjectures referenced unambiguously via unique file names and commit hashes, ensuring fixed and verifiable mathematical intent; unlike standard citations where generality or precision is often left to interpretation.
Furthermore, the benchmark serves as a compass for **Mathlib development**, as formalizing advanced conjectures reveals gaps in the existing library.
We address these via `FormalConjecturesForMathlib`, a directory of results continuously upstreamed to Mathlib. By identifying hard-to-state concepts, the repository both benchmarks reasoning and expands formalized mathematics.

**Limitations.** Problem selection faces a **scope bias** constrained by Mathlib's current coverage, which we mitigate somewhat via our `ForMathlib` directory. While some concepts remain hard to formalize, this limitation will diminish as Mathlib expands.
There is also a **selection bias** toward famous conjectures and Erdős problems, resulting in a concentration of number theory and combinatorics; we plan to mitigate this by sourcing diverse arXiv preprints in areas like computer science and physics.
Focusing on open problems introduces a **misformalization risk** from subtle statement or source errors.
While mitigated via new methods in [Section 3.4](#section-3-4), these risks remain until a full proof is provided; fixing newly discovered errors may require revisions impacting historical comparisons.
The benchmark also faces **contamination challenges**: for solved problems, models may retrieve known arguments rather than reasoning from scratch; for open problems, zero-contamination is not permanent as future solutions enter training data.
Our living repository tracks these developments and reclassifies problems accordingly.
Finally, **verification and reproducibility** rely on the Lean 4 kernel's integrity; while the theoretical exploitability of foundational bugs or unintended axioms is a general challenge for formal systems, it remains the highest standard for rigor.
To ensure stable results despite the repository's evolution, we provide frozen, versioned subsets for standardized evaluation.

**Future Work.** As an active open-source project, we are committed to maintaining and extending Formal Conjectures with new problems, improved problems, and platform features; this ongoing commitment is reflected in the repository's steady growth since its initial release (see [Figure 2](#figure-02)).
See [Section 9](#section-9) for details on planned extensions, metadata improvements, and community features.

<span id="section-6"></span>

## 6 Conclusion

We present **Formal Conjectures**, an evolving collection of mathematical problem statements formalized in Lean 4, including open conjectures.
Sourced from a diverse range of mathematical literature and areas, the benchmark provides broad coverage as an evaluation framework for AI systems' capabilities in formal math.
By providing a zero-contamination testbed of research-level problems alongside a substantial set for proof auto-formalization, we establish a rigorous environment for evaluating AI for formal mathematical research.
Crucially, to ensure stable and reproducible model comparisons as the repository evolves, we complement this dynamic dataset with versioned, frozen evaluation subsets.
We hope the work can drive the expansion of formalized math: both through its living repository and the continuous upstreaming of new definitions to Mathlib, as well as serve as a high-fidelity benchmark that bridges the gap between human mathematical expertise and automated formal reasoning.
Ultimately, we believe this resource will not only measure current progress but also help accelerate the frontier of formal mathematical discovery.

## Acknowledgements

We thank all contributors who have contributed to the repository since it was released as open source, among them Abel Doñate, Aditya Ramabadran, Amogh Parab, Anirudh Rao, Anthony Wang, Ayush Debnath, Bhavik Mehta, Bolton Bailey, Cong Lu, Daniel Chin, Felix Pernegger, Franz Huschenbeth, James Jordan, Jean-Guillaume Durand, Jofre Costa, Junseok Lee, Junyan Xu, Madhu Shree Aravindan, Mario Krenn, Martin Bruse, Michael Rothgang, Mirek Olšák, Ralf Stephan, Reklle, Seewoo Lee, The bbchallenge Collaboration (bbchallenge.org), Wojciech Nawrocki, Yan Yablonovskiy, Yoh Tanimoto, Yongxi Lin, Zeyu Zheng, and Zhen Ning David Liu.
A full list of contributors is available at [https://github.com/google-deepmind/formal-conjectures/graphs/contributors](https://github.com/google-deepmind/formal-conjectures/graphs/contributors).
Additionally, we thank Swarat Chaudhuri, George Tsoukalas, Anton Kovsharov, Henryk Michalewski, Edward Lockhart, and Goran Žužić for helpful discussions and support running experiments.

Here, we include various supplementary details.

<span id="section-7"></span>

## 7 Formal Conjectures Details

<span id="section-7-1"></span>

### 7.1 Source and AMS Classification Tables

Note that a single source problem typically gives rise to multiple formal statements due to variants (e.g., special cases, generalizations, solved sub-problems, and test lemmas). We explicitly allow and encourage the inclusion of multiple alternative formalizations for a single informal claim to capture different mathematical perspectives or resolutions of ambiguity; thus, the counts in this table reflect formal statements, not distinct source problems.

<span id="table-03"></span>

![Number of formal statements by source collection and category](../../papers/formal-conjectures/table-03.png)

**Table 3.** Number of formal statements by source collection, split by category. A single source problem may yield multiple statements (variants, special cases, tests).

[Table 4](#table-04) shows the distribution of statements across AMS subject classifications.
Other notable areas beyond number theory and combinatorics include quantum theory, linear algebra, and algebraic geometry.

<span id="table-04"></span>

![Top ten AMS subject classifications by statement count](../../papers/formal-conjectures/table-04.png)

**Table 4.** Top 10 AMS subject classifications by number of statements in Formal Conjectures.

<span id="section-7-2"></span>

### 7.2 The `answer(sorry)` Mechanism in Detail

Problems requiring a solution are formalized using `answer( )`, a custom Lean term elaborator.
Concretely, a problem like “What is the smallest $n$ such that $P(n)$?” is formalized as:

```lean
theorem smallest_n : IsLeast {n : ℕ | P n} answer(sorry) := by
  sorry
```

As usual, `sorry` is a placeholder which must be filled in order to claim that a problem is solved, but now there are two of them.
If the solution is known, then `answer(sorry)` is replaced in the statement with `answer(42)`.
This mechanism allows our statements to include computational conjectures where discovering the answer is the primary challenge and verification is routine.

Crucially, while providing a proof simply requires finding any way to replace `:= sorry` with a valid term, providing an answer is a fundamentally different task: it requires evaluating mathematical meaning to determine the correct value.
Lean's type checker can verify that a proposed answer leads to a valid proof, but it cannot judge whether the answer itself is mathematically meaningful—that remains a job for human mathematicians or AI systems with genuine mathematical reasoning capabilities.

There are a number of advantages to using the `answer( )` wrapper over directly writing a solution or a bare `sorry` placeholder:

- It makes clear which part of the formalization corresponds to the solution; this allows the solution to be masked from `research solved` statements in order to evaluate agents. This can either be done by inspecting the metadata inserted in the Lean term, or more pragmatically, by a simple regex replacement.
- It provides some protection against Lean's flexible elaboration; a problem stated as `1 / 2 * 2 = sorry` could be solved by replacing the `sorry` with either `(0 : ℕ)` or `(1 : ℚ)`, as by default, Lean only infers the type of an equality after analyzing both sides. This is clearly undesirable; we do not want the meaning of our questions to change depending on the answer provided! Stating the problem instead as `1 / 2 * 2 = answer(sorry)` protects against this, as the `answer` elaborator postpones analysis which forces Lean to decide the type of the equality before seeing the solution.
- It forces the statement to be written in such a way that does not pre-assume what the solution is.

<span id="section-7-2-1"></span>

#### 7.2.1 Propositional Solutions

A particularly common use case arises when the truth value of a proposition $P$ is itself unknown, i.e. when the problem can be phrased as a question: “Is it true that...”.
In this case, `answer(sorry)` serves as a propositional placeholder that should be replaced with either `True` or `False`:

```lean
theorem is_P_true : answer(sorry) ↔ P := by
  sorry
```

Replacing `answer(sorry)` with `True` amounts to conjecturing that $P$ holds (and the solver must prove $P$),
while replacing it with `False` amounts to conjecturing that $P$ fails (and the solver must disprove $P$).
This pattern is widely used throughout the repository for open problems where even the expected truth value is uncertain.

For the convenience of evaluating solver systems that are capable of filling proofs only,
when `answer(sorry)` is detected to be used propositionally in this way, it elaborates to `True` under its default settings.
This permits these systems to decide the proposition by either proving or disproving the statement, rather than by filling the placeholder.

<span id="section-7-3"></span>

### 7.3 The “for Mathlib” Pattern

It is typical when undertaking formalization projects to discover missing (and often trivial) results that seem suitable for Mathlib.
Contributing these results to Mathlib is of course the natural resolution, but this has a number of downsides:

- New conjectures that depend on these results find themselves at the end of a long dependency chain; the missing results must undergo Mathlib review, the merged change must land in a monthly Mathlib release, and the Formal Conjectures repository must be upgraded to that release.
- Some results may be too specialized for Mathlib, and a Mathlib contribution would require generalizing these immediately.
- When the results entail a whole new mathematical development, it may take time for the Mathlib community to find the right abstractions and evolve said development into the right shape. Bypassing the dependency chain mentioned above allows much of this evolution to happen more rapidly outside of Mathlib.

The Lean community has widely adopted the pattern of creating a `ForMathlib` directory as a staging ground for such results, the oldest example of which being [the `for_mathlib` directory](https://github.com/leanprover-community/lean-liquid/tree/master/src/for_Mathlib) in the “liquid tensor experiment”. [+lean-liquid]
The formal-conjectures project's `FormalConjecturesForMathlib` directory follows this pattern, and contains 88 files with results such as natural and logarithmic density of sets of natural numbers (with proofs of basic properties like monotonicity and the density of even numbers), arithmetic progressions and AP-free sets, hypergraph Ramsey numbers, Turing machines and busy beaver halting numbers, perfect powers with a decidability instance, VC dimension in abelian groups, Latin squares and transversals, and a library of graph invariants including the Wiener and Szeged indices, the Lovász theta function, and the Havel-Hakimi residue.
Over time, these results are asynchronously contributed upstream to Mathlib.

[+lean-liquid]: [The liquid tensor experiment](https://github.com/leanprover-community/lean-liquid).

<span id="section-7-4"></span>

### 7.4 Misformalization Taxonomy

Misformalizations can be categorized roughly as follows, with the level shown in parentheses. [+pr-sources]

- **Syntactic (1)**: where the formalizer misses a subtlety in the way Lean parses a piece of syntax. For example, this could be missing brackets changing the meaning of an expression in unexpected ways; see PR #1338.
- **Semantic (1)**: where the formalizer misses a subtlety in the way that Lean represents types or operators. For example, the natural numbers in Lean contain $0$ and have truncated subtraction; see PRs #349, #1259, #1262.
- **Misrepresentation (1)**: where the formalizer formally misrepresents aspects of the informal statement during translation. For example, the use of incorrect quantifiers, the use of incorrect logical connectives, or the failure to include hypotheses which are present in the source text; see PRs #1164, #1156.
- **Implicit conventions (2)**: where the source text assumes domain expertise and does not explicitly include certain hypotheses or conventions. For example, in analytic number theory, questions about the magnitude of a set of numbers are often implicitly asymptotic; see PRs #2136, #1151.
- **Reporting (3)**: where the statement changes in the literature through repetition, usually as the result of typos; see Erdős Problem 918 discussion.
- **Mathematical (3)**: where the source text is clearly not as intended or contains genuine mathematical errors; see Erdős Problem 728 discussion.

[+pr-sources]: Pull request (PR) numbers refer to the [Formal Conjectures repository](https://github.com/google-deepmind/formal-conjectures); Erdős Problem discussions are at [erdosproblems.com](https://www.erdosproblems.com).

<span id="table-05"></span>

![Number and proportion of misformalizations across categories](../../papers/formal-conjectures/table-05.png)

**Table 5.** Number and proportion of misformalizations across categories.

<span id="section-7-5"></span>

### 7.5 Misformalization Examples

This section contains the misformalization examples, mentioned in [Section 3.3](#section-3-3).

<span id="section-7-5-1"></span>

#### 7.5.1 Syntactical Errors

An example of a syntactical error involving the correction of misplaced parentheses is shown in [Figure 5](#figure-05).

<span id="figure-05"></span>

![Code diff correcting misplaced parentheses](../../papers/formal-conjectures/figure-05.png)

**Figure 5.** Code diff showing an example of a syntactical error, and the correction of parentheses.

<span id="section-7-5-2"></span>

#### 7.5.2 Semantic Errors

[Figures 6](#figure-06), [7](#figure-07) and [8](#figure-08) illustrate various semantic errors encountered during formalization.

<span id="figure-06"></span>

![Code diff restricting odd coefficients to polynomial support](../../papers/formal-conjectures/figure-06.png)

**Figure 6.** Code diff showing an example of a semantic error, with the fix requiring quantification over the support of the polynomial.

<span id="figure-07"></span>

![Code diff handling the natural number zero](../../papers/formal-conjectures/figure-07.png)

**Figure 7.** Code diff showing an example of a semantic error, where the formalization had failed to account for behavior at the natural number $0$.

<span id="figure-08"></span>

![Code diff replacing Partition with an exact covering structure](../../papers/formal-conjectures/figure-08.png)

**Figure 8.** Code diff showing an example of a semantic error, where the formalization missed the subtlety that `Partition` invokes `sup` for the covering condition, which means something different for subgroups than for sets.

<span id="section-7-5-3"></span>

#### 7.5.3 Misrepresentation Errors

[Figure 9](#figure-09) provides an example of a misrepresentation error where a hypothesis from the informal text was omitted, and [Figure 10](#figure-10) shows an error involving incorrect quantification.

<span id="figure-09"></span>

![Code diff restoring a hypothesis from the informal statement](../../papers/formal-conjectures/figure-09.png)

**Figure 9.** Code diff showing an example of a misrepresentation error, where the formalization was missing a hypothesis that was present in the informal text.

<span id="figure-10"></span>

![Code diff correcting the quantification of a vertex type](../../papers/formal-conjectures/figure-10.png)

**Figure 10.** Code diff showing an example of a misrepresentation error, where the vertex type `V` was incorrectly quantified. In the old version `V` was a section variable and thus appeared out of the scope of the existential quantifier.

<span id="section-7-5-4"></span>

#### 7.5.4 Implicit Conventions

Examples of errors stemming from implicit conventions in the source material can be seen in [Figures 11](#figure-11) and [12](#figure-12).

<span id="figure-11"></span>

![Code diff making asymptotic quantification explicit](../../papers/formal-conjectures/figure-11.png)

**Figure 11.** Code diff showing an example of an implicit convention, where the source [+erdos-510] implicitly assumes that the inequality holds for sufficiently large $N$. This is a common implicit convention across analytic number theory where one is typically interested in the asymptotic behavior of functions or sequences of natural numbers.

[+erdos-510]: [Erdős Problem 510](https://www.erdosproblems.com/510).

<span id="figure-12"></span>

![Code diff interpreting subgraph as induced subgraph](../../papers/formal-conjectures/figure-12.png)

**Figure 12.** Code diff showing an example of an implicit convention, where the source [+erdos-128-source] wrote “subgraph” rather than “induced subgraph”. The comments [+erdos-128-comments] demonstrate that the problem is trivial if one does not restrict to induced subgraphs, which is why we class this as implicit convention. It might also be considered to be a reporting error since the original paper explicitly writes “induced subgraph”, [Erd93, p. 344].

[+erdos-128-source]: [Source history for Erdős Problem 128](https://www.erdosproblems.com/history/128).

[+erdos-128-comments]: [Comments on Erdős Problem 128](https://www.erdosproblems.com/forum/thread/128).

<span id="section-7-5-5"></span>

#### 7.5.5 Reporting Errors

[Figure 13](#figure-13) demonstrates a reporting error caused by ambiguities in the historical source material regarding induced subgraphs.

<span id="figure-13"></span>

![Code diff clarifying a reporting error in Erdős Problem 918](../../papers/formal-conjectures/figure-13.png)

**Figure 13.** Code diff showing an example of a reporting error between an internal draft formalization of the problem. This problem appears in [Erd68] and [Erd69]. In the former citation the explicit use of $\leq \aleph_0$ is used, while this does not appear in the latter citation. This ambiguity was only clarified post draft formalization with the discovery of trivial solutions—in the (non-induced) subgraph case by the website comments [+erdos-918] and by AlphaProof in the induced subgraph case. Note also that [Erd68] explicitly writes *induced* subgraph while [Erd69] and the website do not.

[+erdos-918]: [Comments on Erdős Problem 918](https://www.erdosproblems.com/forum/thread/918).

<span id="section-8"></span>

## 8 Experimental Evaluation Details

<span id="section-8-1"></span>

### 8.1 Frozen Subsets

The frozen subsets `FC100SolvedSet1` and `FC100OpenSet1` consist of 100 problems each, sampled uniformly at random from statements in the `research solved` and `research open` categories, respectively.
They are defined by the files `FormalConjectures/Subsets/FC100OpenSet1.lean` and `FormalConjectures/Subsets/FC100SolvedSet1.lean` in the repository, which import exactly the corresponding theorem statements.
These files are compiled as part of every tagged release, ensuring that the problem sets remain well-defined and their statements compile correctly across all supported Lean versions.

The subset names (e.g., `FC100OpenSet1`) are *orthogonal* to the repository's release tags (`bench-vN-lean4.X.Y`, see [Section 4.1](#section-4-1)).
When the bench version $N$ stays the same and only the Lean version changes (e.g., from `bench-v1-lean4.27.0` to `bench-v1-lean4.29.0`), the problem set is guaranteed to contain the same statements; the new tag only ensures compilation against the updated Lean toolchain and Mathlib.
When the bench version increments (e.g., from `bench-v1` to `bench-v2`), statements in an existing set may receive fixes (e.g., corrected misformalizations) or category updates (e.g., an `open` problem reclassified as `solved`), but the *membership* of the set remains stable: no problems are added or removed.
For full reproducibility, results should specify both the set name and the release tag, e.g., `FC100SolvedSet1@bench-v1-lean4.27.0`.

In the future, we plan to release additional sets (e.g., `FC100SolvedSet2`, `FC500OpenSet1`) that may contain entirely different problems, enabling targeted evaluations at different scales or difficulty levels while preserving earlier sets.

<span id="section-8-2"></span>

### 8.2 Illustrative Evaluation Details

As noted in [Section 4.2](#section-4-2), our evaluations focus on demonstrating the benchmark’s utility and its sensitivity to scaling.
For all systems, a solution is correct if and only if the Lean 4 kernel accepts the proof term without relying on forbidden axioms.

**AlphaProof Setup.** We evaluated a slightly updated version of AlphaProof [Hub26], with tree-search inference, without the test-time reinforcement learning loop.
As reported in [Table 2](#table-02), we distinguish between two compute configurations for AlphaProof to demonstrate the benchmark's sensitivity to search budget: 1k sims utilizes 1,000 simulations and 2 attempts per problem (approximately 0.1 TPUh on v6e TPUs); and 16k sims and 2 attempts (approximately 1.6 TPUh on v6e TPUs).

**DeepMind Prover Agent Setup.** The DeepMind prover agent [Goo26] results were obtained using a development version of the method under active experimentation.
Based on current API pricing, this evaluation costs up to $100 per problem.

<span id="section-9"></span>

## 9 Extended Discussion

<span id="section-9-1"></span>

### 9.1 Future Work

We plan to continuously extend Formal Conjectures with new problems unblocked by newly added Mathlib definitions or from community contributions.
While our implementation uses Lean 4, the methodology is language-agnostic and portable to systems like Isabelle or Rocq should they develop comparable library support. We will also track future Lean updates to maintain ecosystem compatibility.
To better evaluate the difficulty and impact of conjectures, we plan to extend the metadata by recording when a conjecture was first proposed.
This objective metric will enable users to filter for long-standing open problems versus contemporary questions.
However, since difficulty and importance are often subjective, we also propose an interactive interface where the community can share insights.
Users will be able to vote on a conjecture's perceived difficulty, its mathematical significance, and their conviction regarding its truth value
(e.g., True, False, or independent of ZFC).

Furthermore, references to the mathematical literature are currently handled inconsistently across the repository, often requiring duplication in each file. We plan to improve this by centralizing reference management in future iterations.

This vision is being integrated into an accompanying website [+website] for the repository.
The website already serves as a frontend for the benchmark, allowing users to explore the source code, view LaTeX-rendered docstrings, and search, sort, or filter conjectures.
It also contains metadata recording when a `research open` problem gets solved or when a `research solved` problem receives a formal proof; in the future, we plan to make these changes more prominently visible.

Moving forward, we plan to expand this platform into a collaborative hub with interactive community features, including a public leaderboard to track successful AI-based solutions and recognize groups taking on these problems.

[+website]: [Formal Conjectures website](https://google-deepmind.github.io/formal-conjectures/).
