---
title: 'PutnamBench'
createTime: 2026/08/28 15:42:15
permalink: /en/papers/putnambench/
pageClass: paper-reading
---

> [George Tsoukalas](https://georgetsoukalas.github.io/), [Jasper Lee](https://dblp.org/pid/48/6954), [John Jennings](https://www.cs.utexas.edu/~jej2879/), [Jimmy Xin](https://dblp.org/pid/341/5979), [Michelle Ding](https://rnclncj.github.io/), [Michael Jennings](https://dblp.org/pid/40/3198), [Amitayush Thakur](https://amit9oct.github.io/aboutme/), and [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/). First submitted to arXiv on July 15, 2024; the current version is v2, dated November 3, 2024. Accepted to the [NeurIPS 2024 Datasets and Benchmarks Track](https://proceedings.neurips.cc/paper_files/paper/2024/hash/1582eaf9e0cf349e1e5a6ee453100aa1-Abstract-Datasets_and_Benchmarks_Track.html). [PutnamBench: Evaluating Neural Theorem-Provers on the Putnam Mathematical Competition](https://arxiv.org/abs/2407.11214v2). <a href="/paper/putnambench.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2407.11214). [TeX source](https://export.arxiv.org/e-print/2407.11214v2). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

We present PutnamBench, a new multi-language benchmark for evaluating the ability of neural theorem-provers to solve competition mathematics problems. PutnamBench consists of 1692 hand-constructed formalizations of 640 theorems sourced from the William Lowell Putnam Mathematical Competition, the premier undergraduate-level mathematics competition in North America.
All the problems have formalizations in Lean 4 and Isabelle; a substantial subset also has Coq formalizations. PutnamBench requires significant problem-solving ability and proficiency in a broad range of topics taught in undergraduate mathematics courses. We use PutnamBench to evaluate several established neural and symbolic theorem-provers.
These approaches can only solve a handful of the PutnamBench problems, establishing the benchmark as a difficult open challenge for research on neural theorem-proving. PutnamBench is available at [https://github.com/trishullab/PutnamBench](https://github.com/trishullab/PutnamBench).

<span id="section-1"></span>

## 1 Introduction

Automating mathematical reasoning is a longstanding goal in artificial intelligence [New57].
A prominent line of work on the problem [Li24t] uses neural models to direct theorem-proving in formal frameworks like Lean 4 [Mou21], Isabelle [Wen08], and Coq [Coq23]. These frameworks can “execute” proofs like code and offer execution feedback, which simplifies the search for correct proofs.

The design of quality benchmarks is a key challenge in this research area. The two most prominent competition-based benchmarks for neural theorem-proving are miniF2F [Zhe22a] and FIMO [Liu23s]. The former formalizes a mix of problems from high-school level courses and mathematics competitions such as AIME, AMC, and IMO; the latter consists of a collection of IMO problems. Both benchmarks have limitations. For example, miniF2F contains many problems that can be immediately solved using an SMT solver, and FIMO only targets the Lean 3 framework, which is no longer actively maintained.

More generally, as large language models (LLMs) grow in importance as a tool for neural theorem-proving [Li24t], preventing leakage between pretraining sets and evaluation sets is more important than ever. This makes the continued supply of new benchmarks an important goal.

In this paper, we respond to this challenge with PutnamBench, a new hand-curated, multi-langauge benchmark for neural theorem-provers. PutnamBench includes 1692 formalizations of 640 problems from the William Lowell Putnam Mathematical Competition, the premier college-level mathematics competition in North America. [+1]
All our problems have Lean 4 [Mou21] and Isabelle [Wen08] formalizations; a substantial fraction have formalizations in Coq [Coq23] as well. The formalizations are all manually constructed and have been carefully debugged. The benchmark also includes the original English-language problem statements with permission from the Mathematical Association of America, which organizes the Putnam competition.

One key benefit of PutnamBench is that Putnam competition problems require a broad base of mathematical knowledge and skills. Because they target undergraduate students, they cover topics such as analysis and abstract algebra that do not appear in the International Mathematical Olympiad (IMO). At the same time, success in the two competitions is correlated—top performers on the Putnam competition are often former IMO medalists as well. Hence, PutnamBench is well-aligned with the IMO Grand Challenge [Imo19] and the AI Mathematical Olympiad [Aim23], the latter of which offers a $10M prize fund for developing a system that can win a gold medal at the IMO.

Another advantage is that PutnamBench supports multiple proof assistants. Lean 4, Coq, and Isabelle are currently the three most popular formal proof languages. However, theorem-proving benchmarks typically only contain problems in a strict subset of these languages—for example, miniF2F [Zhe22a] does not include Coq problems, and FIMO [Liu23s] only targets Lean. PutnamBench is the first mathematics-competition benchmark to include problems in all three languages.

We use PutnamBench to evaluate several neural and symbolic approaches: Draft-Sketch-Prove [Jia22], COPRA [Tha23], GPT-4, Sledgehammer [Pau15], and Coqhammer [Cza18]. Collectively, these methods can only solve a handful of the PutnamBench problems, establishing PutnamBench as a hard open challenge for the neural theorem-proving community.

[+1]: PutnamBench is available at [https://github.com/trishullab/PutnamBench](https://github.com/trishullab/PutnamBench).

<span id="section-2"></span>

## 2 Background

**Formal Theorem-Proving.** Formal proof frameworks like Lean 4 [Mou21], Coq [Coq23], and Isabelle [Wen08] allow users to write machine-verifiable proofs of mathematical theorems.
To create such a proof, one first uses a framework-specific language to formally state the target theorem. The mathematical objects referenced in the theorem can be imported from an existing repository or defined by the user.
During the proof process, the proof framework maintains a *state* that includes information about the parts of the proof that remain to be completed. One can change this state by executing a *proof step*. The user's goal is to write a sequence of proof steps (in the framework's language) that changes the proof state to a special state “QED” in which there are no unmet proof obligations.

<span id="figure-01"></span>

![Figure 1. A formalization of Putnam 1988 B1 in Lean 4 together with a proof discovered through a few-shot invocation of GPT-4.](../../papers/putnambench/figure-01.png)

**Figure 1.** A formalization of Putnam 1988 B1 in Lean 4, which asserts that for all integers $a,b \geq 2$, there are positive integers $x,y,z$ such that $ab = xy + xz + yz + 1$. The formal proof begins by introducing all relevant variables and hypotheses with `intro`, then indicating the choice of $x,y,z$ with `use`, and afterwards proving all goals using the automated tactics `linarith` and `ring`. This proof was discovered through a few-shot invocation of GPT-4.

[Figure 1](#figure-01) illustrates a theorem and proof in the Lean 4 framework.

**The Putnam Competition.** The William Lowell Putnam Mathematical [Wil24], organized by the Mathematical Association of America (MAA), is the premier collegiate mathematics competition in North America. Thousands of undergraduate students from universities across the United States and Canada take the exam each year. The competition comprises two 3-hour-long sessions of six problems each, presented in approximately ascending order of difficulty within each session. While some problems require competitors to furnish a concrete solution (such as a number, a set, or the truth value of a given statement), all problems require a natural-language proof of correctness. The contest draws from a wide variety of topics in the undergraduate curriculum, often using instances of ideas from research-level mathematics.

<span id="section-3"></span>

## 3 PutnamBench

<span id="table-01"></span>

![Table 1. Comparison of existing formal theorem proving evaluation benchmarks.](../../papers/putnambench/table-01.png)

**Table 1.** Comparison of existing formal theorem proving evaluation benchmarks. PutnamBench exceeds prior benchmarks by providing support for all of Lean 4, Isabelle, and Coq, on a set of difficult competition problems using undergraduate-level mathematics. For problems requiring a numerical solution in addition to a proof, we factor the solution out of the theorem statement.

<span id="table-02"></span>

![Table 2. Quantity by domain of PutnamBench problems.](../../papers/putnambench/table-02.png)

**Table 2.** Quantity by domain of PutnamBench problems. Our formalizations generally reflect the variety of Putnam problems, though we can only formalize few geometry and probability problems due to limited support for these topics in the respective mathematical libraries.

PutnamBench is a multi-language evaluation benchmark consisting of formalized problems from the Putnam competition.
PutnamBench is a manually produced benchmark, including 640 formalizations in Lean 4 and Isabelle, and 412 formalizations in Coq. In aggregate, PutnamBench contains 1692 formalizations of Putnam competition problems. We also incorporate the informal statements and numerical solutions where applicable.

Now we elaborate on the main features of PutnamBench.

**Diversity and Breadth.** Compared to miniF2F [Zhe22a] and FIMO [Liu23s], which generally rely on high-school mathematics, PutnamBench incorporates a wider variety of problems which require definitions of the standard undergraduate mathematics curriculum. The ProofNet benchmark [Aze23] also sources problems from the undergraduate curriculum, but these problems are generally from standard textbooks as opposed to mathematical competitions. Putnam problems often require definitions from multiple fields, which standard textbooks do not necessarily target. Formalizations in PutnamBench include concepts from a wide range of mathematical fields, including:
(i) ***Analysis***: Limits, integrals, derivatives, continuity;
(ii) ***Linear Algebra***: Matrices, determinants, fields;
(iii) ***Abstract Algebra***: Rings, groups, magmas, permutations;
(iv) ***Algebra***: Polynomials, inequalities, algebraic expressions;
(v) ***Number Theory***: Primes, irrationality, base representations, divisors, palindromes;
(vi) ***Geometry***: Polygons, point sets, line intersections, Euclidean distance;
(vii) ***Set Theory & Combinatorics***: Countability, power sets, discrete structures, games.

**Multiple Languages.** PutnamBench contains formalizations of Putnam problems in Lean 4, Isabelle, and Coq. The formalizations also include concepts defined in each proof assistant's mathematical repositories—notably, Mathlib, the HOL standard library, and Coquelicot (among various Coq repositories). To the best of our knowledge, PutnamBench is the first undergraduate-level competition benchmark for each of these languages. Furthermore, we are the first to produce a human mathematics competition-style evaluation benchmark for Coq.

We hope that this contribution can enable Coq practitioners access to the rapidly-growing field of machine learning for mathematics.

Generally, the formalizations of the problems are aligned in their structure, including hypothesis naming and framing. Differences may arise according to the underlying foundations of each language. We also note that the pre-defined mathematical theory in each language differs, which can sometimes lead to difficulties formalizing certain problems.

Compared to the prior benchmarks miniF2F, FIMO, and ProofNet, PutnamBench is the first to support Lean 4 on initial release [+2].

**Factored Solutions.** Roughly 60% of Putnam problems, in their natural language form, require exhibiting a (closed-form) solution along with a proof of its correctness. Such problems do not assert propositions, and hence are not immediately formalizable as they are not directly the statement of a theorem. Prior benchmarks such as miniF2F [Zhe22a] sidestep this issue by rewording the problem statement to ask for a proof that the solution satisfies the constraints of the problem. However, this reduction diminishes the overall difficulty of the problem, as producing a solution can constitute the majority of the difficulty. To address this issue, we factor out solutions of such problems from the formalized theorem statement. We include an example in [Figure 2](#figure-02). In this way, we provide two tasks for neural theorem proving:

1. **Task 1:** Given the theorem statement, first identify the (closed-form) solution, and then provide a proof of correctness by rewriting the solution into the theorem statement.

2. **Task 2:** Given the theorem statement and solution, produce a proof of its correctness. This task aligns with the current benchmarks.

We note that the process of producing the numerical solution may be highly correlated with the proof of its correctness. In this way, our formalizations can reflect the true difficulty of the informal problem statement.

<span id="figure-02"></span>

![Figure 2. A factored-solution formalization of Putnam 2008 B5 in Lean 4.](../../papers/putnambench/figure-02.png)

**Figure 2.** A formalization of Putnam 2008 B5 in Lean 4. As the problem requires exhibiting the set of functions $f$ satisfying the specified conditions, it is not directly the statement of a theorem. We formalize the problem by instantiating a variable “solution” outside of the theorem statement. In this way, a model can either provide its own candidate, or use the correct solution we provide and attempt to produce a proof of correctness. Benchmarks such as miniF2F and FIMO only include formalizations with the solution written into the theorem statement.

**Formalization effort and challenges.** We hand-crafted our benchmark over the course of several months as a team of two doctoral and five undergraduate students with prior experience in university mathematics, computer science, and formal proof assistants. We found that the average time-to-formalize a single problem in one language was roughly 25 minutes.
Each formalization was verified by a second person at least once, and we measured that the verification of a single formalization took between 10 minutes, on average. We acknowledge that the time-to-formalize we report is higher than that of miniF2F; we believe this is largely due to the increased complexity of the Putnam problems, which oftentimes require definitions we must locate in each language's respective mathematical libraries.

We first produced formalizations in Lean 4, and then proceeded with our formalization effort in Isabelle and then Coq. Due to differences in the underlying foundations of each language, we found that formalizations in one language sometimes do not directly transfer to another; for example, Isabelle does not have a subtyping mechanism, which we made extensive use of in Lean 4. Formalizations in Coq rely on a number of mathematics repositories. Predominantly, we rely on MathComp and MathComp-Analysis [Mat15, Mat17], but also make us of Stdlib, Stdpp, Coquelicot, GeoCoq, and Coqtail [Coq15, Geo15, All20].

Some problems are not naturally amenable to formalization—for example, we found that while formalizing problems involving probabilities is possible, such formalizations often require heavy probability theory.
Similarly, support for problems involving Euclidean geometry varies across languages; in particular, Lean 4 does not yet have a sufficiently extensive library to make most geometry problems formalizable. By contrast, Coq has an extensive geometry repository called GeoCoq, which we utilize for our Coq formalizations.

<span id="figure-03"></span>

![Figure 3. Formalizations of Putnam 2006 B2 in Lean 4, Isabelle, and Coq.](../../papers/putnambench/figure-03.png)

**Figure 3.** Formalizations of Putnam 2006 B2 in (a) Lean 4, (b) Isabelle, (c) Coq. Putnam 2006 B2 asserts that given a finite subset $X \subseteq \mathbb{R}$ with $|X| = n > 0$, there is a nonempty subset $S \subseteq X$ and an $m \in \mathbb{Z}$ such that $|m + \sum_{s \in S} s| \leq \frac{1}{n+1}$.

**Dataset Contamination.** Our benchmark is unique compared to informal benchmarks such as MATH [Hen21] and GSM8K [Cob21] in the sense that the target output *has never been produced*, hence avoiding direct contamination. To the best of our knowledge, we are the first to provide formalizations of a large collection of Putnam problems in any of Lean, Isabelle, and Coq. Since writing a formal proof requires the formal theorem statement, it is highly unlikely any possible formal proof has been written for any of our problems. We performed a thorough investigation of formal mathematics repositories for each language for confirmation, finding no aligned theorems and proofs from the Putnam Competition.
We do not include any of the formal proofs in our benchmark.

Furthermore, any proofs found by automated methods in our evaluations are not included and are only mentioned in this article. Indirect contamination can occur through transfer from training on the informal proofs, though producing proofs in formal proof environments still presents a major difficulty for all current neural methods, as we find in [Section 4](#section-4).

**Licensing and Rules of Engagement.** PutnamBench is available under an Apache 2.0 license for Lean 4 and Isabelle, and under an MIT license for Coq. We align the licenses with those of the repositories we use for each language. With permission from the MAA, we include the informal statements as sourced from the competition [Ale85, Ked02, Ked20]. We host a public leaderboard at [https://trishullab.github.io/PutnamBench/](https://trishullab.github.io/PutnamBench/) and will readily accept evaluation results from future works.

[+2]: miniF2F, FIMO, and ProofNet were originally released using Lean 3, and miniF2F and FIMO have been updated to include Lean 4 formalizations following community efforts [Aze23, Rah24]. To the best of our knowledge, no open-sourced Lean 4 version of FIMO currently exists.

<span id="section-4"></span>

## 4 Experimental Evaluation

To understand the challenges that PutnamBench poses for state-of-the-art theorem-proving approaches, we attempt to solve its problems using a suite of such approaches.
Given the relative lack of tailored systems for multi-language theorem-proving, we run evaluations for each language separately. Any method that is evaluated on multiple languages is based on off-the-shelf foundation models.

**Metrics.** Our evaluation is based on the $\mathrm{pass}@n$ [Lam22] metric. This metric measures a prover's ability to produce a successful proof, as determined by the formal proof environment, given a budget of $n$ *proof attempts*. In search-based methods [Tha23], each proof attempt involves a distinct search that can query a neural model multiple times.

**Models.** For each of the languages, we perform evaluations using GPT-4 [Ope23] [+3], a highly capable foundation model. We run evaluations using in-context learning, appending several examples of successful proofs of simple theorems in each language. For evaluations with Lean 4 approaches, we note that many approaches have targeted Lean 3, which is not backward-compatible and no longer actively maintained.
We evaluate COPRA [Tha23] on PutnamBench, modifying the prompt examples of COPRA to enable search in Lean 4. Furthermore, we run evaluations LeanDojo's retrieval-augmented prover ReProver, a finetuned model designed to utilize incorporate retrieved lemmas as part of the proof search. We also include evaluations with the retrieval component held out.

For our Isabelle experiments, we run evaluations of Draft, Sketch, and Prove (DSP) [Jia22] using GPT-4 as the underlying foundation model, noting that many further works for theorem-proving in Isabelle have extended on the DSP pipeline as we mention in [Section 5](#section-5). We also run evaluations using stand-alone invocations to Sledgehammer, a powerful symbolic automation tool in Isabelle that relies on calls to external SMT solvers.

As for our Coq experiments, prior neural approaches for Coq have mostly targeted software verification tasks, as opposed to competition mathematics.
As a result, our Coq experiments use COPRA, which also supports theorem-proving in Coq. We evaluate using the Tactician [Bla20a] platform with the locality sensitive hashing model configuration. We also run evaluations using CoqHammer [Cza18], a tool similar to Isabelle's Sledgehammer, which makes calls to external constraint solvers.

[+3]: We use GPT-4o for all our evaluations.

<span id="section-4-1"></span>

### 4.1 Results

<span id="table-03"></span>

![Table 3. Results of evaluations on PutnamBench in Lean, Isabelle, and Coq.](../../papers/putnambench/table-03.png)

**Table 3.** Results of evaluations on PutnamBench in each language. We find that all tested methodologies perform poorly, solving at most a handful of problems. Notably, the only problem solved in both Lean and Coq is Putnam 1988 B1, which is not solved by any method in Isabelle. ReProver, our finetuned baseline for Lean, is unable to solve any problems with or without retrieval. Symbolic automation proves to be powerful in Isabelle, with Sledgehammer solving the most problems than GPT4 alone. DSP generates four successful proofs, two of which cannot be generated by Sledgehammer alone.

**Lean 4.** We prompt GPT-4 in a $\mathrm{pass}@10$, setting temperature $T = 0.7$ and using several examples of simple theorems and proofs, to generate a proof for each problem. The result of this experiment yields a single successful proof across all 640 Lean formalizations. The problem (Putnam 1988 B1) and the generated proof are given in [Figure 1](#figure-01). In particular, Putnam 1988 B1 is solved on the first of 10 attempts. An example of a failure mode of GPT-4 is given in [Figure 18](#figure-18).

We also run evaluations with COPRA, using their default hyperparameters for search, performing a $\mathrm{pass}@1$, and allowing 60 queries to GPT-4. However, since COPRA was originally designed for interaction with Lean 3, we make a small modification to its system prompt to enable search in Lean 4. The result of the step-wise proof search over all Lean 4 formalizations yields a correct proof to one problem (1988 B1). We find that backtracking in the search was not required for this proof, which was 10 lines long and was found at the 10th query. It is possible that affording COPRA further queries to GPT-4 can yield more successful proofs, though it is not yet feasible to perform such an experiment due to the cost of queries to GPT-4.

We found that, by default, GPT-4 produces proofs using Lean 3 syntax, which is not compatible with Lean 4. Even when directed to produce outputs in Lean 4, GPT-4 typically continues to produce outputs in Lean 3. Our prompt, which we include in [Figure 16](#figure-16), elucidates some design differences in Lean 4 to better enforce compliance with the Lean 4 syntax. However, we noticed many examples where GPT-4 continues to output terms in Lean 3 syntax. One such example is given in [Figure 17](#figure-17).

We run ReProver using the standard search parameters used in LeanDojo [Yan23e]. Our evaluation yields no successfully proven problems, with and without the inclusion of the retrieval module. We believe that Putnam 1988 B1, which the other methods solve, is not solved by ReProver as it requires an understanding that the choice of $x,y,z=1,a-1,b-1$ will eventually satisfy the conditions of the goal after simplification. Smaller models, like the one driving ReProver's search, may not be as readily capable of such understanding.

**Isabelle.** We run GPT-4 using the same configuration, with modified prompts for Isabelle, on our Isabelle formalizations. We find that GPT-4 can produce a single successful proof to Putnam 1986 B1, a geometric problem stated algebraically. We include the statement and its proof as generated by GPT-4 in [Figure 19](#figure-19).

<span id="figure-04"></span>

![Figure 4. A formalization of Putnam 2001 A1 in Isabelle and the corresponding proof discovered by DSP.](../../papers/putnambench/figure-04.png)

**Figure 4.** A formalization of Putnam 2001 A1 in Isabelle and the corresponding proof discovered by our evaluation with DSP. Sledgehammer alone can also produce a successful proof to this theorem.

DSP represents a neurosymbolic methodology which has seen significant application for theorem-proving in miniF2F. We run DSP with $\mathrm{pass}@10$, using temperature $T = 0.1$ and GPT-4 as the underlying language model. Our evaluation yields four successful proofs: of Putnam 2001 A1 and 1971 B1, two problems involving magmas (sets with a binary operation), one of Putnam 1995 A1, a problem involving a closed-under-multiplication subset of the reals, and Putnam 1986 B1. In particular, Putnam 1995 A1 and 1986 B1 cannot be solved by Sledgehammer alone. The generated proof of Putnam 1995 A1 is included in [Figure 4](#figure-04).

We run a baseline using Sledgehammer, a powerful automation tool in Isabelle which makes calls to external SMT solvers to prove a given goal. With a set timeout of $t = 120$ seconds, we run Sledgehammer on each Isabelle formalization. The result of this evaluation is 3 successfully proven problems: Putnam 1971 B1, 2001 A1, and 2012 A2. Notably, all of these problems are statements about sets with binary operations. We include the statements of 1971 B1 and 2012 A2 in [Figure 22](#figure-22).

**Coq.** We run GPT-4 with a Coq-based prompt on our Coq formalizations using the same configuration as in Lean and Isabelle. The result of the experiment is 1 solved problem, namely Putnam 1988 B1, which was also solved in Lean 4. The proof, which we include in [Figure 14](#figure-14), generally follows the same structure as the proof in Lean.

An evaluation with COPRA, in a $\mathrm{pass}@1$-with-$60$-queries and $T = 0.0$ also yields a successful proof only for Putnam 1988 B1 which we include in [Figure 14](#figure-14). In this case, backtracking was crucial for proof search on this problem. The crucial step in 1988 B1 is the choice of $x,y,z$ once $a$ and $b$ have been introduced. Initially, COPRA predicts the erroneous choice $x, y, z = 1, 1, ab-1$ and eventually reverts this choice using backtracking. Afterwards, COPRA predicts a correct choice $x, y,z = 1, a-1, b-1$ and proceeds with the proof.

We run Tactician using the locality sensitive hashing model with a timeout of $t = 600s$ per problem. Our evaluation yields no successfully proven problems. While showing favorable performance on theorems drawn from Coq's standard library [Zha21j], such methodologies do not as of yet scale to challenging olympiad-style problems.

We run CoqHammer with 8 parallel threads using an ATP timeout of 100 seconds, proof reconstruction timeout of 15 seconds, and sauto timeout of 5 seconds, for a total of 120 seconds allocated for each formalization. The evaluation yields no successful proofs—indicating that symbolic tools in Coq are not yet capable of handling PutnamBench problems. It is not surprising that CoqHammer does not match the performance of Sledgehammer even though they rely on the same external solvers. The underlying logical system of Coq is more complex than that of Isabelle and is hence less amenable to automation.

<span id="section-4-2"></span>

### 4.2 General Analysis

Aggregating over all experiments performed in all languages, we find that a total of 6 problems in PutnamBench are successfully proven. A majority of these come from evaluations in Isabelle, particularly with strong contributions from Sledgehammer. Sledgehammer can solve all three problems involving magmas which appear in our benchmark but fails to produce successful proofs for any other formalization. DSP solves an additional two problems and relies heavily on Sledgehammer to fill in the proofs of intermediate steps. The single problem solved in Lean and Coq also makes use of automated tactics like `linarith` and `lia`, and requires only a single crucial step.

Hence, we find that a few PutnamBench problems are not entirely intractable using current methods. However, anecdotally, these problems are among the easiest ever included in the Putnam competition. All admit a very short natural language proof and do not require reasoning about particularly complicated objects. We believe that significant advancements in automated mathematical reasoning are required to make progress on PutnamBench.

<span id="section-5"></span>

## 5 Related Work

**Formal Benchmarks.** Several evaluation benchmarks for formal mathematics have been developed in recent years. miniF2F [Zhe22a] is a formal-to-formal benchmark of competition problems, sourced from high school competitions such as the AMC, AIME, and IMO. miniF2F is a multi-language benchmark, comprising of 488 problems each formalized in Lean 3, Metamath, Isabelle and HOL Light. We chose not to include formalizations in Metamath and HOL Light as they have not been the focus of attention for neural theorem-proving. A similar competition-style benchmark is FIMO [Liu23s], which contains 149 Lean 3 formalizations of IMO shortlist problems produced using a back-translation procedure with GPT-4. The automatically-generated formalizations are then manually verified. Both benchmarks are designed to measure *certifying* the solution to the informal problem statement when one exists. Compfiles [Com24] is a collection of 171 Lean 4 formalizations of competition problems, predominantly from the IMO and USAMO, often accompanied by a formal proof, which has not seen use in benchmarking automated theorem-provers. ProofNet [Aze23] introduced a benchmark of 371 exercises, formalized in Lean 3, from standard textbooks in the undergraduate mathematics curriculum. While largely not competition-based, problems in ProofNet draw from a broader library of concepts than miniF2F and FIMO, which rely only on high-school mathematics. LeanDojo [Yan23e] introduces a dataset of formal mathematics and proofs derived from Lean's mathlib library [Mat20b], and trains a retrieval-augmented model towards generating proofs on their held-out test set. ProverBot9001 [San20a] introduced a dataset for theorems and proofs written in Coq derived from CompCert [Ler09], a formally verified C compiler. PISA [Jia21a] is a dataset derived from Isabelle's Archive of Formal Proofs [Afp04], which contains theorems and proofs from general mathematics as opposed to specifically competition problems.

**Informal Benchmarks.** There are also several popular benchmarks for informal (natural-language) mathematical reasoning. MATH [Hen21] is a collection of 12,500 mathematics problems, in natural language only, sourced from various high school competitions additionally supplied with step-by-step informal proofs. GSM8K [Cob21] is a collection of 8,500 grade school mathematics problems, intended to benchmark natural language reasoning for mathematics-style problems. While benefiting from the abundance of natural language data, these benchmarks fall short, since in natural language, there is no automatic mechanism for certifiable verification of the reasoning path which yielded the numerical answer. For this reason, metrics for success on these benchmarks usually rely on exact-answer match, because verifying reasoning paths is imprecise and is best done by human experts. By contrast, theorem proving in formal proof assistants comes with a high-confidence signal for correctness of the reasoning path, or *proof*, of a theorem.

**Methods for Formal Theorem-Proving.** Significant effort has been spent on developing automatic theorem-provers for formal mathematics [Li24t]. Most recent efforts train a neural module to perform proof-step prediction, which is then wrapped in a search mechanism to locate a valid proof. GPT-$f$ [Sut20] trains a transformer-based architecture on data derived from the Metamath library [Meg19] for proof synthesis. PACT expands on GPT-$f$ by incorporating auxiliary training tasks for the neural module towards theorem-proving in Lean 3. FMSCL [Pol22] alternates proof-search and training to finetune their neural model based on proofs found during search. HTPS [Lam22] uses a transformer-based neural module in an online, MCTS-inspired proof search in Lean 3 and Metamath. COPRA [Tha23] uses GPT-4 supplied with error feedback from the environment and lemmas from a retrieval mechanism for an agentic proof-search in Lean 3 and Coq. LLEMMA [Aze24] continues pretraining of Code Llama on a mathematics-based corpus dubbed Proof-Pile-2, and uses their learned model for formal proof search in Lean 4. DeepSeek-Prover [Xin24] produces synthetic Lean data en-masse for training their prover model. AlphaGeometry [Tri24] targets IMO problems in a geometry-specific proof assistant language using an interleaving search, where a neural module synthesizes auxiliary constructions and a symbolic engine produces deductive closures.

The Isabelle proof assistant [Pau94], given its declarative nature and powerful symbolic automation, has too been the focus of much attention for neural theorem proving. Isabelle features Sledgehammer [Pau15], an automated reasoning tool which calls external automated theorem provers (ATPs) for proof synthesis. Draft, Sketch, Prove (DSP) [Jia22] uses a high-caliber LLM to generate natural language proofs and converts them into formal *sketches* in Isabelle, whose gaps are then filled using Sledgehammer. Zhao et al. [Zha23o] employed a diffusion model to predict an optimal ordering of the few-shot examples provided to the LLM in the DSP pipeline. Lyra [Zhe23b] utilized error-feedback from Isabelle's execution to modify holes in the sketch which were too difficult for the symbolic prover. POETRY [Wan24ac] leverages recursion for theorem-proving and trains a neural module to produce proof sketches, as opposed to using in-context learning with an LLM. LEGO-Prover [Wan23k] extends the pipeline by incorporating a skill library which grows throughout the proof search task. Separate from approaches utilizing natural language proofs, Thor [Jia22a] trains a transformer-based architecture to predict successful invocations of Sledgehammer, along with the usual proof-step objective. Baldur [Fir23] explored repairing erroneous proofs in Isabelle through the use of LLMs.

The Coq interactive theorem prover has seen use in both software verification and general mathematics. Famously, mechanized proofs of the Four Colour Theorem [Rob97] and the Feit-Thompson theorem [Gon13] were produced in Coq. Similarly, numerous software verification projects have been undertaken in Coq, such as CompCert (a formally verified C compiler) and Verdi [Wil15] (a framework for verifying distributed systems protocols). ASTactic [Yan19b] trained a neural module involving recurrent networks and attention on data collected from various Coq repositories.
Proverbot9001 [San20a] targeted proof synthesis on a set of held-out theorems from the CompCert project. COPRA [Tha23] also evaluates on this CompCert-based task using their multi-language approach. Tactician [Bla20a] develops a platform for proof automation for the Coq practitioner, with support for experimenting with new machine learning techniques for tactic prediction and proof search. Zhang et al. [Zha21j] explores several online learning techniques inside Tactician, including an approximate $k$-nearest neighbors method via locality sensitive hashing which we use for our evaluation. Graph2Tac [Bla24b] uses graph neural networks for learning online hierarchical representations of new theorems and definitions, and is used for proof search within Tactician.

<span id="section-6"></span>

## 6 Conclusion

We presented PutnamBench, a benchmark for neural theorem-proving consisting of formalizations of Putnam competition problems. A distinctive feature of PutnamBench is that it spans a broad range of undergraduate-level mathematical topics, including algebra, analysis, and number theory. Another unique benefit is that it includes problems in Lean 4, Isabelle, and Coq, the three most popular formal proof frameworks.

As our experiments show, PutnamBench is a challenging benchmark: all current theorem-proving approaches fail to solve more than a handful of its problems. We believe that these failures include two root causes: (i) While current theorem-provers can effectively stitch together standard proof steps well-represented in the training corpus, they often fail at synthesizing new lemmas and orchestrating these lemmas into intricate proofs. (ii) Current methods often fail to leverage the deep knowledge available in mathematics repositories. Developing a new generation of neural theorem-provers in which these weaknesses are at least partly addressed is an exciting direction of future research.

## Acknowledgements

This work was supported by NSF awards CCF-2212559 and CCF-2403211, the NSF Institute for Foundations of Machine Learning, and a gift by the Aziz Family Foundation. We thank Oliver Nash, Eric Wieser, Edward Lockhart, Fabian Gloeckle, Karl Palmskog, Lasse Blaauwbroek, Jason Rute, and Kaiyu Yang for useful discussions, aiding in benchmark maintenance, and support with setting up experiments.

<span id="section-7"></span>

## 7 Checklist

1. For all authors...

   1. Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope?

      **Answer: Yes.** We support our main claims in [Section 3](#section-3) and [Section 4](#section-4).

   2. Did you describe the limitations of your work?

      **Answer: Yes.** We discussed in [Section 3](#section-3) the challenges of formalizing certain problem categories such as geometry and probability due to the nature of support for such mathematical theory in each language.

   3. Did you discuss any potential negative societal impacts of your work?

      **Answer: N/A.** We do not anticipate any negative societal impact of our work.

   4. Have you read the ethics review guidelines and ensured that your paper conforms to them?

      **Answer: Yes.** We have read the ethics review guidelines and ensured our paper conforms to them.

2. If you are including theoretical results...

   1. Did you state the full set of assumptions of all theoretical results?

      **Answer: N/A.** We do not include any theoretical results.

   2. Did you include complete proofs of all theoretical results?

      **Answer: N/A.** We do not include any theoretical results.

3. If you ran experiments (e.g. for benchmarks)...

   1. Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)?

      **Answer: Yes.** We disclosed all information related to the experiments, which use open-sourced methods. We have also included the URL to our dataset: [https://github.com/trishullab/PUTNAM/](https://github.com/trishullab/PUTNAM/).

   2. Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)?

      **Answer: N/A.** We did not perform any training.

   3. Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)?

      **Answer: No.** We evaluate our selected methodologies using established metrics accepted by the neural theorem-proving community. See [Section 4](#section-4).

   4. Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)?

      **Answer: Yes.** Most of our experiments rely on calls to GPT-4, we include sampling details. We also mention the hyperparameters to calls to symbolic methods in [Section 4](#section-4).

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

   1. If your work uses existing assets, did you cite the creators?

      **Answer: Yes.** We did cite the creators of any existing assets we used.

   2. Did you mention the license of the assets?

      **Answer: Yes.** We aligned the license of our benchmark with the license of those assets.

   3. Did you include any new assets either in the supplemental material or as a URL?

      **Answer: Yes.** We included our dataset by sharing the following URL: [https://github.com/trishullab/PUTNAM/](https://github.com/trishullab/PUTNAM/).

   4. Did you discuss whether and how consent was obtained from people whose data you're using/curating?

      **Answer: Yes.** We obtained permission from the MAA.

   5. Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content?

      **Answer: N/A.** Our data does not contain such content.

5. If you used crowdsourcing or conducted research with human subjects...

   1. Did you include the full text of instructions given to participants and screenshots, if applicable?

      **Answer: N/A.** We did not conduct research with human subjects nor crowdsource.

   2. Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable?

      **Answer: N/A.** We did not conduct research with human subjects nor crowdsource.

   3. Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation?

      **Answer: N/A.** We did not conduct research with human subjects nor crowdsource.

<span id="section-8"></span>

## 8 Appendix

We include further examples of formalizations from PutnamBench below.

<span id="figure-05"></span>

![Figure 5. A formalization of Putnam 2009 B1 in Coq relying on the MathComp repository.](../../papers/putnambench/figure-05.png)

**Figure 5.** A formalization of Putnam 2009 B1 in Coq relying on the MathComp repository.

<span id="figure-06"></span>

![Figure 6. A factored-solution formalization of Putnam 2001 B4 in Lean 4.](../../papers/putnambench/figure-06.png)

**Figure 6.** A formalization of Putnam 2001 B4 in Lean 4. As the problem requires deciding whether the infinite intersection is empty, it is not directly the statement of a theorem. We consider the associated “solution” of this problem to be a boolean value, and factor it out from the theorem statement. `sorry` is the placeholder keyword for Lean.

<span id="figure-07"></span>

![Figure 7. A factored-solution formalization of Putnam 2020 A3 in Lean 4.](../../papers/putnambench/figure-07.png)

**Figure 7.** A formalization of Putnam 2020 A3 in Lean 4. As the problem requires deciding whether the series converges, it is not directly the statement of a theorem. We consider the associated “solution” of this problem to be a boolean value, and factor it out from the theorem statement.

<span id="figure-08"></span>

![Figure 8. A formalization of Putnam 1997 A4 in Lean 4.](../../papers/putnambench/figure-08.png)

**Figure 8.** A formalization of Putnam 1997 A4, which requires knowledge of group theory, in Lean 4. The informal statement is slightly underspecified—$g_1, g_2, g_3, h_1, h_2, h_3$ are not explicitly defined to be in $G$. To produce the formalization, we must be specific about the type of $g_i, h_i$.

<span id="figure-09"></span>

![Figure 9. A formalization of Putnam 2018 B1 using mathlib4's Vector class.](../../papers/putnambench/figure-09.png)

**Figure 9.** A formalization of Putnam 2018 B1, which requires the Vector class from mathlib4.

<span id="figure-10"></span>

![Figure 10. An Isabelle formalization of Putnam 1992 B6.](../../papers/putnambench/figure-10.png)

**Figure 10.** An Isabelle formalization of Putnam 1992 B6.

<span id="figure-11"></span>

![Figure 11. A factored-solution formalization of Putnam 2012 A3 in Isabelle.](../../papers/putnambench/figure-11.png)

**Figure 11.** An Isabelle formalization of Putnam 2012 A3. The mechanism for factoring the solution out of the theorem statement is similar to that of Lean.

<span id="figure-12"></span>

![Figure 12. A Coq formalization of Putnam 1980 A5 using Coquelicot.](../../papers/putnambench/figure-12.png)

**Figure 12.** A Coq formalization of Putnam 1980 A5. This formalization is done using Coquelicot, a Coq repository outside of the standard library. The Coq equivalent of `sorry` is `Admitted`.

<span id="figure-13"></span>

![Figure 13. A factored-solution formalization of Putnam 2017 B2 in Coq.](../../papers/putnambench/figure-13.png)

**Figure 13.** A Coq formalization of Putnam 2017 B2. As the problem requires a numerical witness, we factor that out using Coq's syntax for making definitions.

<span id="figure-14"></span>

![Figure 14. A Coq proof of Putnam 1988 B1 generated through a few-shot invocation of GPT-4.](../../papers/putnambench/figure-14.png)

**Figure 14.** A Coq proof of Putnam 1988 B1 generated through a few-shot invocation of GPT-4. The proof is similar to that of the Lean version, also discovered by GPT-4. The main difficulty of the problem is to choose the values of $x,y,z$ given $a,b$. Once correctly supplied, the remainder of the proof is routine and can be done with automated methods like `lia` which handles linear arithmetic.

<span id="figure-15"></span>

![Figure 15. Examples of easy miniF2F formalizations.](../../papers/putnambench/figure-15.png)

**Figure 15.** Examples of formalizations of easy problems in miniF2F. While useful for benchmarking straightforward mathematical reasoning in a formal setting, these problems are quite simple compared to the competition problems present in PutnamBench. We note that miniF2F does include some formalizations of problems sourced directly from high school competitions, but these are fewer in number.

<span id="figure-16"></span>

![Figure 16. Parts of the system prompt used by GPT-4 for Lean 4 evaluations.](../../papers/putnambench/figure-16.png)

**Figure 16.** Parts of the “system prompt” used by GPT-4 for Lean 4 evaluations. Due to GPT-4's tendency towards producing outputs in Lean 3 syntax, our prompt places special attention towards preventing such syntactic mistakes. A similar modification is made to COPRA's system prompt for Lean 3.

<span id="figure-17"></span>

![Figure 17. A failed COPRA tactic prediction for Putnam 2011 B2 in Lean 4.](../../papers/putnambench/figure-17.png)

**Figure 17.** An example of a failed tactic prediction during proof search for Putnam 2011 B2 using COPRA in Lean 4. GPT-4 predicts a tactic involving the premise “differentiable_at.div,” which exists in Lean 3, but not Lean 4. Even with the system prompt asserting outputs should involve Lean 4 syntax alone, GPT-4 is not always capable of making the distinction.

<span id="figure-18"></span>

![Figure 18. A failed proof generated by a few-shot invocation of GPT-4.](../../papers/putnambench/figure-18.png)

**Figure 18.** A failed proof generated by few-shot invocation of GPT-4. GPT-4 misunderstands that the hypothesis `[Mul S]`, which gives an operation $\star$ and asserts it is a binary operation on $S$, also asserts associativity of the operation. The tactic `rw [←mul_assoc]`, which performs a rewrite using the associativity of $\star$, is hence not applicable in this setting.

<span id="figure-19"></span>

![Figure 19. A successful Isabelle proof discovered using GPT-4.](../../papers/putnambench/figure-19.png)

**Figure 19.** A successful proof in Isabelle discovered using GPT-4. While the theorem statement differs from the formalization, which is algebraic, we note that the official solution is also algebraic and is similar to the generated Isabelle proof.

<span id="figure-20"></span>

![Figure 20. A successful Isabelle proof discovered using DSP.](../../papers/putnambench/figure-20.png)

**Figure 20.** A successful proof in Isabelle discovered using DSP. In particular, this proof is not found using a single invocation of sledgehammer, so the sketching mechanism of DSP is crucial for this problem. We note that the DSP pipeline involves using an LLM (GPT-4) to synthesize an informal proof which is translated into a sketch in Isabelle—this can potentially be a source of indirect dataset contamination, as we cannot ensure the informal proofs are not present in GPT-4's training data.

<span id="figure-21"></span>

![Figure 21. An erroneous generation produced by DSP for Putnam 1971 B1.](../../papers/putnambench/figure-21.png)

**Figure 21.** An erroneous generation produced by DSP for Putnam 1971 B1. While a single invocation of Sledgehammer can prove this problem, the formal sketch generated in the pipeline is much more complex and is erroneous, leading to a failed proof attempt.

<span id="figure-22"></span>

![Figure 22. Two Isabelle formalizations solved with invocations of Sledgehammer.](../../papers/putnambench/figure-22.png)

**Figure 22.** The other two Isabelle formalizations solved with invocations of Sledgehammer. We note that the problems Sledgehammer was capable of solving are all problems involving binary operations on sets. It is not surprising that SMT solvers are capable of solving such problems, which do not require reasoning about complicated objects.

<span id="figure-23"></span>

![Figure 23. COPRA backtracks from an incorrect choice and successfully predicts correct values for Putnam 1988 B1.](../../papers/putnambench/figure-23.png)

**Figure 23.** Early in COPRA's attempt on Putnam 1988 B1, an incorrect prediction of $x,y,z$ given $a,b$ is made, which dooms that path of search as the most crucial step is the correct choice. Later, at step 32 of search, COPRA backtracks and then successfully predicts a correct choice for $x,y,z$. Once this step is generated, the remainder of the proof is straightforward.
