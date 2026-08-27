---
title: 'Discover and Prove'
createTime: 2026/08/27 23:57:07
permalink: /en/papers/discover-and-prove/
---

> [Chengwu Liu](https://dblp.org/pid/276/4722-1.html) [+internship], [Yichun Yin](https://dblp.org/pid/180/5934.html), [Ye Yuan](https://dblp.org/pid/33/6315-16.html), [Jiaxuan Xie](https://dblp.org/pid/250/2540.html), [Botao Li](https://dblp.org/pid/179/3223.html), [Siqi Li](https://dblp.org/pid/34/180.html), [Jianhao Shen](https://dblp.org/pid/217/2324.html), [Yan Xu](https://dblp.org/pid/03/4702.html), [Lifeng Shang](https://dblp.org/pid/70/4288.html), and [Ming Zhang](https://dblp.org/pid/73/1844-4.html). First submitted to arXiv on April 17, 2026; current version v1. Accepted to ACL 2026 Main Conference. [Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4](https://arxiv.org/abs/2604.15839v1). [Original PDF](/paper/discover-and-prove.pdf). [DOI](https://doi.org/10.48550/arXiv.2604.15839). [TeX source](https://export.arxiv.org/e-print/2604.15839v1). The original PDF remains authoritative for the exact print layout and bibliography.

[+internship]: Work done during the internship at Huawei Technologies Co., Ltd. Code and datasets are available at [GitHub](https://github.com/liuchengwucn/discover-and-prove).

## Abstract

Most ATP benchmarks embed the final answer within the formal statement — a convention we call “Easy Mode” — a design that simplifies the task relative to what human competitors face and may lead to optimistic estimates of model capability. We call the stricter, more realistic setting “Hard Mode”: the system must independently discover the answer before constructing a formal proof. To enable Hard Mode research, we make two contributions. First, we release MiniF2F-Hard and FIMO-Hard, expert-reannotated Hard Mode variants of two widely-used ATP benchmarks. Second, we introduce Discover And Prove (*DAP*), an agentic framework that uses LLM natural-language reasoning with explicit self-reflection to discover answers, then rewrites Hard Mode statements into Easy Mode ones for existing ATP provers. *DAP* sets the state of the art: on CombiBench it raises solved problems from 7 (previous SOTA, Pass@16) to 10; on PutnamBench it is the first system to formally prove 36 theorems in Hard Mode — while simultaneously revealing that state-of-the-art LLMs exceed 80% answer accuracy on the same problems where formal provers manage under 10%, exposing a substantial gap that Hard Mode benchmarks are uniquely suited to measure.

<span id="section-1"></span>

## 1 Introduction

The use of AI to solve mathematical problems has attracted considerable research interest, not only because of the potential for concrete applications in domains like education and mathematical research, but also because tackling highly abstract mathematical problems generally requires capabilities that may generalize and transfer to complex real-world tasks. These capabilities include planning, search, deductive reasoning, and induction. [Yan24k] Within the spectrum of mathematical tasks, competition problems — especially those at the International Mathematical Olympiad (IMO) level — have garnered particular attention. These problems go beyond numerical computation or simple formula application. They typically demand abstraction and modeling, rigorous logical argumentation, and often require elements of intuition and creativity. Accordingly, the ability to solve IMO-level problems is widely regarded as an important milestone for AI [Yan24k].

Existing approaches to solving mathematical problems fall into two broad categories: informal methods and formal methods. Informal methods solve mathematical problems in natural language and leverage the strong reasoning abilities of large language models (LLMs), whereas formal methods use formal languages such as Lean [Mou21] and Isabelle [Nip02] to express the solution. A key advantage of formal methods is that proofs written in formal languages can be automatically and rigorously verified by a proof assistant program. At the International Mathematical Olympiad (IMO) 2024, participating AI systems employed formal methods [Alp24]. By IMO 2025, however, most evaluated systems had shifted toward informal approaches [Luo25i, Hua25h, Wei25i, Hua25i].

We observe that current practice in many formalization efforts often embeds the final answer directly into the statement to be proved, which we refer to as “Easy Mode”. We point out that Easy Mode may substantially reduce the difficulty of formal problem-solving tasks. To address this issue, we draw inspiration from prior works (PutnamBench [Tso24] and CombiBench [Liu25ad]): answer-oriented problems are encoded in Lean 4 with two separate goals (two distinct `sorry`s). In this “Hard Mode” configuration, the model must first supply the final answer by replacing the first `sorry` with the answer and then produce a conventional formal proof for the remaining goal. This setup prevents embedding extra information that human contestants must discover by themselves in the formal statement. By definition, *Hard Mode* requires that any quantity a human competitor must derive through reasoning is not supplied as a premise in the formal statement; it must be independently discovered by the AI system. We adopt the term “Hard Mode” following the convention in the Lean community; [+hard-mode-name] CombiBench [Liu25ad] refers to the same distinction as “without solution” vs. “with solution”, while PutnamBench [Tso24] uses “no answer” vs. “with answer”. An example illustrating the difference between Easy Mode and Hard Mode is presented in [Figure 1](#figure-01). We commissioned expert annotators to reannotate two widely used ATP competition datasets, namely MiniF2F and FIMO, producing MiniF2F-Hard and FIMO-Hard. During reannotation, we corrected known alignment issues in existing formal benchmarks [Wan25l, Lin25j].

[+hard-mode-name]: [Lean community discussion](https://leanprover.zulipchat.com/#narrow/channel/208328-IMO-grand-challenge/topic/IMO.202025.20problem.20statements).

To solve Hard Mode problems, we introduce the Discover and Prove (*DAP*) framework, a fully open-source, agent-based ATP framework for Hard Mode ATP tasks. *DAP* consists of two components: a Discovery Module and a Proving Module. We prompt an open-source LLM to generate and iteratively refine its reasoning and answers using self-verification procedures. After the Discovery module “discovers” a plausible answer and fills the first `sorry`, the Proving module attempts to produce complete formal proofs by invoking traditional ATP provers. *DAP* achieves state-of-the-art results. Evaluating on the full PutnamBench dataset, *DAP* solves 36 problems in total. On solution-style problems with Hard Mode variants, it solves 19 problems — to our knowledge, this constitutes the first public result on PutnamBench under this Hard Mode evaluation setting. On CombiBench Hard Mode, it solves 10 problems, improving significantly on the previous state of the art (Kimina-Prover Preview), which solved 8 problems.

Our contributions are threefold.

1. We reannotate two commonly used ATP competition datasets, MiniF2F and FIMO, to align tasks presented to human competitors with those given to AI systems, removing the Easy Mode discrepancy and providing a more principled basis for evaluating AI mathematical capability.
2. We propose *DAP*, an open-source, agentic Hard Mode ATP framework. With a simple and straightforward design, *DAP* achieves state-of-the-art performance on PutnamBench and CombiBench.
3. We provide an analysis quantifying the individual contributions of the two modules of our proposed *DAP* prover to overall performance. These results shed light on the relative strengths of informal and formal method approaches for solving competition-level mathematical problems.

<span id="section-2"></span>

## 2 Related Work

<span id="figure-01"></span>

![Figure 1. Easy Mode and Hard Mode configurations in automated theorem proving](../../papers/discover-and-prove/figure-01.png)

**Figure 1.** Differences between Easy Mode and Hard Mode configurations in automated theorem proving. The example shown is a Lean 4 formalization of an IMO problem in two different styles. This example was intentionally selected to illustrate the kind of semantic misalignment that our re-annotation effort corrects: the Easy Mode formalization proves only that $x$ must lie within certain ranges, but does not establish that every value in those ranges is attainable, weakening the original if-and-only-if requirement to a one-directional implication. By contrast, our Hard Mode formalization represents the answer as a set, thereby fully capturing the natural-language problem's requirement in the formal statement.

<span id="section-2-1"></span>

### 2.1 Mathematical Problem-Solving with AI systems

Powered by CoT prompting and RLVR training, LLMs have made substantial progress in mathematical reasoning. Frontier models (e.g., OpenAI o1 [Ope24h], DeepSeek R1 [Dee25c], Google Gemini 2.5 Pro [Com25a]) now achieve near-saturation performance on widely-used math benchmarks, including GSM8K [Cob21], MATH-500 [Lig23], and AIME 2024/2025.

One downside of the informal methods is that the solution traces they produce are notoriously difficult to verify automatically. Assessing the validity of a generated proof would require domain experts to inspect it carefully to detect subtle errors [Lig23], which is infeasible at scale. For this reason, informal systems are typically applied to problems that require only a final numerical or symbolic answer, verified by direct comparison against ground-truth [Wen25a].

<span id="section-2-2"></span>

### 2.2 Automated Theorem Proving

The key strength of automated theorem proving (ATP) is that formal proofs can be checked rigorously and automatically by proof assistants [You25, Wan23j]. Although formal approaches have faced challenges of limited formal-data availability [Xin25], they have advanced rapidly with larger LLMs and scaling searching compute [Xin25b, Che25h]. Recent systems such as Kimina-Prover Preview [Wan25l], DeepSeek-Prover-V2 [Ren25b], and Goedel-Prover-V2 [Lin25j] have made substantial progress on MiniF2F [Zhe22a] benchmark. Seed-Prover [Che25h] is a lemma-style whole-proof reasoning model that iteratively refines proofs via Lean compiler feedback, proved lemmas, and self-summarization, achieving over 50% on PutnamBench and saturating MiniF2F. DSP [Jia23d] guides formal theorem provers with natural-language draft proofs and structured sketches; DSP+ [Cao25d] revives this paradigm with modern reasoning models. DAP differs from DSP/DSP+ in three key respects; see [Section 10](#section-10) for a detailed comparison.

Several agentic frameworks have also targeted Lean-based ATP [Tha23, Ana24, Bab25, Wan25an]; see [Section 9](#section-9).

To handle informal mathematical problems that require a final answer, prior efforts typically convert problems requiring solutions into proof problems by embedding the desired answer into the formalized statement and proving that statement [Zhe22a, Liu23s, Xio23a]. This practice raises two concerns. First, embedding the answer in the statement can reduce the intrinsic difficulty: for many solution-oriented problems, the primary challenge is discovering the answer rather than proving a consequence once known, so supplying it acts as a substantial hint. Second, the resulting formalized statements are sometimes not perfectly semantically aligned with the tasks human contestants face. As prior analyses (e.g., FMC [Xie25e], OlympiadBench [He24d]) have noted, some existing formal benchmarks are misaligned: some formal statements may capture only a subset of the goals that human solvers must address.

<span id="section-2-3"></span>

### 2.3 Formalization & Data Curation

High-quality formal datasets are scarce and require substantial expert effort [Xin25]. MiniF2F [Zhe22a] is one of the most widely used ATP benchmarks, containing formalized statements from mathematical olympiads and high-school and undergraduate courses; Kimina-Prover Preview [Wan25l] supplies corrections to a subset of its problems. FIMO [Liu23s] is constructed from IMO shortlist problems but lacked a publicly available Lean 4 version, which impeded its adoption in recent work. PutnamBench [Tso24] comprises hand-constructed formalizations of Putnam Competition problems and, for the first time, provides both “with answer” and “no answer” evaluation settings. CombiBench [Liu25ad] offers 100 combinatorics problems ranging from middle-school level through IMO and university level, complementing the number-theoretic and algebraic focus of the other benchmarks; IMOSLLean4 [San25] and IMO-Steps [You24] additionally supply complete proofs. To alleviate data scarcity, auto-formalization methods [Wu22a] automatically translate informal problems into formal statements. Representative resources include Lean Workbook [Yin25a], NuminaMath-LEAN [Wan25l], Goedel-Pset-v1 [Lin25i], and FormalMATH [Yu25j]. Because outputs are typically validated only via LLM-as-a-judge [Yin25a], they lack guaranteed semantic correctness and are more commonly used in provers' RL training phases [Wan25l, Xin25] than as evaluation benchmarks. To our knowledge, all auto-formalization techniques follow the Easy Mode paradigm, embedding the desired answer directly into each generated statement.

<span id="section-3"></span>

## 3 Methodology

<span id="figure-02"></span>

![Figure 2. Primary flowchart of the DAP framework](../../papers/discover-and-prove/figure-02.png)

**Figure 2.** Primary flowchart. A mathematical problem is first processed by the Discovery Module to generate a solution; this solution is then incorporated into the Easy Mode statement during the rewriting stage. Orange circles denote the reasoning LLM, and blue circles denote the theorem prover (another distinct LLM).

To address the challenge of proving Hard Mode Lean 4 theorems containing two `sorry` placeholders, we propose the Discover and Prove (*DAP*) framework, designed to emulate a human mathematician's approach by providing both the answer and a detailed proof. As its name suggests, *DAP* consists of two primary modules: a Discovery Module and a Proving Module. The Discovery Module operates in natural language, tasked with identifying the correct solution to the problem, and subsequently transforms the original Hard Mode Lean 4 statement into an Easy Mode statement. This transformation reduces the number of `sorry` placeholders from two to one, thereby leaving a single statement that can be resolved by conventional automated theorem provers. Then, the Proving Module utilizes the Lean 4 formal language to construct a rigorous proof for the solution identified by the Discovery Module. The complete workflow of the DAP framework is depicted in [Figure 2](#figure-02).

<span id="section-3-1"></span>

### 3.1 Discovery Module

The Discovery Module aims to solve the original mathematical problem and substitute the solution into the first `sorry` placeholder of the Lean 4 Hard Mode statement for the Proving Module to use. This process mirrors human problem-solving by hypothesizing an answer to guide the proof search. Although the reasoning capabilities of LLMs have substantially improved following the introduction of long Chain-of-Thoughts such as OpenAI o1 [Ope24h] and DeepSeek R1 [Dee25c], the one-shot resolution of highly challenging, IMO-level problems remains unsolved. Current research indicates that state-of-the-art LLMs often require auxiliary tools (e.g., information retrieval, calculators, external memory) to support deep reasoning [Nak21, Luo25c, Yua24d]. Inspired by prior work on LLM reasoning systems [Hua25h], we employ a relatively straightforward configuration where an advanced reasoning model generates solution steps and performs self-verification to enhance accuracy. Specifically, the procedure involves the following steps:

1. **Solution Generation:** Given a mathematical problem in natural language, the model's reasoning capabilities are leveraged to generate a detailed chain-of-thought describing the solution process.
2. **Self-Verification:** The reasoning LLM is instructed to inspect its steps for potential errors and produce an error report identifying any erroneous locations (a representative error report is shown in [Section 11.5](#section-11-5)). If self-verification reveals no errors, the process proceeds to step 4; otherwise, it moves to step 3.
3. **Self-Correction:** The reasoning LLM is instructed to generate a revised solution that addresses the errors identified in the error report.
4. **Rewriting:** Using the Lean 4 Hard Mode statement, the natural-language problem, and the model's chain-of-thought reasoning, the LLM is prompted to produce a rewritten Lean 4 statement containing only a single placeholder, suitable for automated theorem proving.

All four steps are implemented through meticulously designed prompts to the LLM, which are detailed in [Section 11](#section-11). The Discovery Module is crucial because incorrect solutions frequently lead to flawed formal statements that are unprovable from the outset, thus emphasizing the model's capacity for deep reasoning and reliable self-verification. This framework is released as open-source to facilitate reproducibility and serve as a baseline. For the Discovery Module, we utilize the open-source model GPT-OSS-120B, known for its strong mathematical reasoning performance. In principle, the framework can be instantiated with any model that combines robust mathematical reasoning with basic Lean proficiency.

<span id="section-3-2"></span>

### 3.2 Proving Module

Once the Discovery Module transforms a Hard Mode statement into an Easy Mode formulation, the task becomes a standard ATP problem, amenable to conventional theorem provers. For this purpose, we employ Goedel-Prover-V2 (32B), a state-of-the-art open-source theorem prover, to process the transformed problems. By decoupling the reasoning model from the ATP model, the proposed framework can improve as any of the underlying models advance, and provides a contemporary baseline for evaluating LLMs that generate formal mathematical solutions in a proof assistant language.

<span id="section-4"></span>

## 4 Data Curation

<span id="table-01"></span>

![Table 1. Automated theorem proving dataset statistics and Lean compatibility](../../papers/discover-and-prove/table-01.png)

**Table 1.** The table presents, for commonly used automated theorem proving datasets, the number of statement samples, the number of Hard Mode problems, the data sources, data curation methods, and compatibility with different versions of the Lean formal language. Notably, ProofNet and miniF2F were originally released as Lean 3 datasets, and publicly available community ports to Lean 4 exist. Although studies have reported performance on a Lean 4 variant of FIMO, no publicly available Lean 4 version of the FIMO dataset exists.

<span id="section-4-1"></span>

### 4.1 Annotation Principles

Our re-annotation is guided by three principles.

**Semantic Accuracy:** Expert-annotated datasets are small but reliable; auto-formalized datasets are large but unverifiable — current validation methods (LLM-as-a-judge [Yin25a], BEq [Liu24y]) cannot guarantee semantic alignment with the original natural-language problem. We therefore start from expert-annotated sources and re-examine each statement manually.

**Interpretability:** A formal statement should reflect exactly what a human contestant is asked to do: any quantity to be discovered must not be supplied as a premise, and any claim to be proved must appear as the goal. Current benchmarks violate this in three recurring ways: (1) encoding the final answer in the statement, (2) weakening the proof goal to a strict subset of the original, or (3) adding premises that human contestants do not have. Representative instances are shown in [Figure 1](#figure-01); we followed the IMOLean [Mye25] convention throughout.

**Consistency:** Formalization style can materially affect ATP success rates [Lin25i], so permitting annotators to choose arbitrary formalizations would introduce evaluation bias. We provided a unified convention emphasizing idiomatic Lean — “go further in the direction of idiomatic Lean rather than trying to follow a particular English version closely” — consistent with the IMOLean [Mye25] practice of providing a single canonical statement.

<span id="section-4-2"></span>

### 4.2 Data Selection

To address these deficiencies, we engaged Lean experts to reannotate two human-expert–annotated datasets, namely MiniF2F [Zhe22a] and FIMO [Liu23s]. Although ProofNet is also a high-quality, expert-annotated dataset, we observed that all of its natural-language items are inherently proof-based; consequently, we excluded it from consideration. The annotators each have more than one year of Lean-related experience. The MiniF2F and FIMO datasets were originally produced by experts; we re-examined them in light of error reports documented in the literature [Wan25l] and had each problem independently annotated by two experts to cross-validate labels and thereby safeguard correctness.

<span id="section-4-3"></span>

### 4.3 Dataset Quality Fixes

Beyond creating Hard Mode variants, our annotators performed three non-trivial categories of quality-improvement work; full details are in [Section 12](#section-12). **Porting FIMO to Lean 4:** we ported all FIMO problems from Lean 3, verifying compilation and semantic faithfulness. **Fixing semantic misalignments:** we identified and repaired $\approx 15$ errors in miniF2F and $\approx 20$ in FIMO across four error types ([Table 5](#table-05)). **Rephrasing for Hard Mode:** unknown values were promoted to free parameters with explicit side-conditions; see [Figure 3](#figure-03) ([Section 13](#section-13)).

<span id="section-5"></span>

## 5 Experiments

<span id="table-02"></span>

![Table 2. Performance of open-source approaches on standard and Hard Mode benchmarks](../../papers/discover-and-prove/table-02.png)

**Table 2.** Performance of various open-source approaches on standard and Hard Mode benchmarks. The numerals beneath each dataset name indicate the total number of examples and the number of Hard Mode examples. Best results under each configuration are indicated in boldface. In Hard Mode evaluation, each dataset contains proof-style problems (evaluated in their original form) and solution-style problems (evaluated with the answer not provided in the formal statement). Each entry $X / Y$ denotes total problems solved ($X$) and solution-style problems solved in Hard Mode ($Y$). All results are Pass@32 unless otherwise specified. $^\dagger$Kimina-Prover Preview originally reported 7 solved problems on CombiBench at Pass@16; the value of 8 shown here is our re-evaluation at Pass@32.

For the Discovery Module, we use the open-source model GPT-OSS-120B, which demonstrates strong performance on natural language mathematical reasoning tasks. For the Proving Module, we use Goedel-Prover-V2, which exhibits state-of-the-art performance in traditional automated theorem proving. Our formal verification environment uses Lean 4.15.0, with Kimina-Server [Dos25] mediating interactions with the Lean 4 REPL.

For GPT-OSS-120B, we follow the model’s recommended configuration with sampling temperature 1.0. During Self-Verification, the model is allowed up to 30 iterative attempts for self-checking and error correction. If all attempts fail, the pipeline falls back to the no-agent (ablation) configuration, where the reasoning model’s output is used directly as the answer candidate without verification. For Goedel-Prover-V2, we follow the developer-recommended sampling configuration: temperature 0.7, `max_tokens` set to 30,000, 32 samples, and Pass@32 evaluation metric. We report performance on PutnamBench, CombiBench, miniF2F-Hard, and FIMO-Hard in [Table 2](#table-02).

Our method achieves state-of-the-art performance on Hard Mode problems. On CombiBench, it solves 10 problems, improving on the prior state-of-the-art (8 problems); it is also the first method to solve problems under PutnamBench's “No Answer” configuration. Notably, on PutnamBench and miniF2F-test, our method's Hard Mode accuracy closely approaches Goedel-Prover-V2's Easy Mode performance, suggesting that our Discover and Prove approach effectively reduces Hard Mode problems to Easy Mode problem statements.

<span id="section-6"></span>

## 6 Discussion

<span id="section-6-1"></span>

### 6.1 Ablation Study on Agent Effectiveness

Advanced reasoning LLMs inherently possess significant self-reflective capabilities [Wen25a]. Consequently, explicitly introducing self-verification and self-correction mechanisms might impose an unnecessary reasoning overhead. To investigate when an agentic mode is beneficial, we conducted experiments with the agent mode disabled across the PutnamBench, CombiBench, MiniF2F-Hard, and FIMO-Hard datasets. The results are presented in [Table 2](#table-02).

Disabling the agent's explicit self-verification and self-correction mechanisms resulted in a significant performance degradation on challenging datasets like PutnamBench. Conversely, no performance degradation was observed on lower-difficulty datasets, such as CombiBench and MiniF2F. We hypothesize that this effect stems from these datasets containing a substantial proportion of relatively simple problems (e.g., middle-school textbook problems) that reasoning LLMs can solve with minimal difficulty. In such cases, the explicit inclusion of self-verification might inadvertently cause the model's instruction-following behavior to over-dominate, leading to excessive self-questioning and the introduction of additional noise to the solution.

<span id="table-03"></span>

![Table 3. Discovery Module performance on Hard Mode problems with definitive answers](../../papers/discover-and-prove/table-03.png)

**Table 3.** On the subset of Hard Mode problems with definitive (ground-truth) answers, the Discovery Module's performance in solving mathematical problems presented in natural language was evaluated.

We analyzed the Discovery Module's accuracy on Hard Mode problems with ground-truth answers, which permit direct assessment of natural language responses ([Table 3](#table-03)). Even without agentic self-reflection, the model achieves approximately 78% correctness on Putnam problems, with performance on MiniF2F approaching saturation. For lower-difficulty problems, explicit self-verification and self-correction can be superfluous given the already high one-shot accuracy of reasoning LLMs. However, for more challenging benchmarks like PutnamBench and FIMO-Hard, these agentic components remain crucial. A fine-grained failure-mode analysis of the Discovery Module on CombiBench and FIMO is provided in [Section 14](#section-14).

Ablating self-verification iteration counts shows that 10 iterations approach saturation and 30 (our default) add only marginal gains; see [Section 15](#section-15).

<span id="section-6-2"></span>

### 6.2 Ablation on Rewriting Strategies

A central design choice in *DAP* is the two-stage rewriting pipeline: the Discovery Module first derives the answer in natural language, and only then is the Hard Mode statement transformed into an Easy Mode statement for the ATP prover. To justify this design, we compare three alternative strategies. **No Rewriting:** The Hard Mode problem statement (containing two `sorry` placeholders) is fed directly to the ATP prover without any rewriting. **Straight Rewriting:** The Discovery Module is discarded; instead, an LLM is asked to simultaneously discover the answer *and* produce the rewritten Easy Mode statement in a single step. **Proposed Rewriting (Ours):** The Discovery Module first finds the answer; the Rewriting stage then transforms the statement using that answer.

<span id="table-04"></span>

![Table 4. Pass@32 comparison of three rewriting strategies](../../papers/discover-and-prove/table-04.png)

**Table 4.** Pass@32 comparison of three rewriting strategies. No Rewriting and Straight Rewriting results were manually verified to exclude spurious proofs.

[Table 4](#table-04) shows the results, from which we draw three key observations.

**No Rewriting is largely ineffective on Hard Mode problems.** We attribute this to models' lack of training on Hard Mode ATP tasks, producing large out-of-distribution performance drops when asked to both discover answers and prove them directly in the formal system.

**Straight Rewriting performs no better — and sometimes worse.** Requiring the model to handle answer discovery and formalization simultaneously in a single step causes it to frequently fail at both. The joint burden of informal mathematical reasoning and syntactically precise Lean code generation leads to degraded performance on most benchmarks.

**Spurious proofs explain the remaining gap.** We frequently observed *spurious proofs* (cheating behavior) in both No Rewriting and Straight Rewriting settings. Because the model can see the full Lean statement — including the `abbrev solution` definition — during solving, it sometimes avoids genuine mathematical reasoning by copying problem conditions directly into the proof. A concrete example from `fimo_2009_algebra_p3` is shown in [Figure 4](#figure-04) ([Section 16](#section-16)). *DAP*'s decoupled design prevents this shortcut: the prover only ever receives the rewritten Easy Mode statement, which does not expose the answer placeholder, eliminating the opportunity for such cheating behavior.

*DAP*'s modular design allows the Discovery Module and the Proving Module to be replaced independently; experiments with both lightweight small open-source models and the stronger Aristotle API confirm that the pipeline remains functional across resource constraints and that stronger informal reasoning models yield further gains (see [Section 17](#section-17)).

<span id="section-6-3"></span>

### 6.3 Natural Language Reasoning or Formal Language Reasoning?

The choice between natural-language and formal-language reasoning presents a critical dilemma for AI's advancement in complex problem-solving. Recent innovations in natural-language reasoning — such as Long Chain-of-Thought and Reinforcement Learning from Human Feedback (RLHF), exemplified by OpenAI's o1 and DeepSeek's R1 — have significantly enhanced LLMs' capabilities. Research has shown that natural-language representations can exhibit very high empirical ceilings; performance on challenging datasets like AIME has saturated rapidly [Ope24h, Wen25a]. At IMO 2025, the majority of AI systems transitioned from formal to natural-language representations and successfully solved creative reasoning problems, with several solutions attaining gold-medal recognition. In contrast, formal-language systems like Seed Prover [Che25h] and Aristotle [Ach25] achieved medal-worthy results through extensive computational search, with some solutions completed only after official submission deadlines.

This study investigates Hard Mode ATP as a neutral benchmark for comparing natural-language and formal-language reasoning. We observe a similar divergence: state-of-the-art LLMs with self-reflection exceed 80% accuracy on PutnamBench, while formal-language systems achieve less than 10%

Despite the robust empirical performance of natural-language reasoning, formal mathematical language will remain essential for future AI systems in mathematical reasoning. Natural-language proofs are prone to subtle errors that are challenging to identify without formal verification, significantly limiting their practical utility for critical applications such as education and advanced mathematics research. As evidenced by results on datasets like Putnam and FIMO, formal mathematical reasoning still requires considerable development. We therefore propose that a promising avenue for future research involves integrating the strengths of both natural-language and formal-language approaches to bridge the existing performance disparity.

<span id="section-7"></span>

## 7 Conclusion

This study identifies semantic accuracy, interpretability, and transparency problems in current Easy Mode automated theorem-proving settings. We argue that formal statements in Hard Mode, manually annotated by human experts with a unified convention, more accurately reflect the problem-solving requirements faced by participants in mathematical competitions.

To address the scarcity of Hard Mode ATP benchmarks, we engaged Lean 4 experts to reannotate high-quality datasets into the Hard Mode format, producing MiniF2F-Hard and FIMO-Hard. In parallel, we rectified existing semantic misalignments and enforced a unified convention.

To address Hard Mode problems, we introduce the *DAP* framework, designed to offer a rigorous evaluation of AI systems' capacity to solve complex formal mathematical problems. The framework achieves state-of-the-art performance on PutnamBench and CombiBench, representing the first reported progress on PutnamBench's Hard Mode configuration. These results demonstrate the efficacy of combining natural-language reasoning to derive answers with formal methods to prove them.

<span id="section-8"></span>

## 8 Limitations

The primary limitation of this work concerns potential dataset contamination, which may have led to inflated estimates of reported AI performance. The miniF2F and FIMO datasets annotated in this study, together with their source problems in natural language (like IMO shortlisted problems) and other related datasets, are publicly accessible on the Internet. The two main LLMs employed in this work — GPT-OSS-120B and Goedel-Prover-V2 — have released model weights but not their training corpora; consequently, we cannot definitively determine whether these datasets were included in training data.

By demonstrating a performance gap between the Discovery Module and the Proving Module, this study identifies the bottleneck for Hard Mode ATP as formal reasoning rather than natural-language reasoning. This suggests that approaches leveraging natural-language reasoning to assist formal-language reasoning, such as DSP [Jia23d], show promise. In our work, however, the natural-language output passed to the Proving Module is limited to answer candidates. The integration between formal-language and natural-language reasoning therefore remains limited. Ideally, the two modules would interact more tightly, with one modality providing assistance when the other encounters difficulty. Developing such tighter, cooperative integration is left to future work.

## Acknowledgments

This paper is partially supported by grants from the National Key Research and Development Program of China with Grant No. 2023YFC3341203.

<span id="section-9"></span>

## 9 Agentic Frameworks for Formal Theorem Proving

Beyond pure proof-search approaches, several agentic frameworks have been explored for Lean-based ATP. COPRA [Tha23] converts a general-purpose LLM into a Lean proof specialist via a language-agent loop. LeanAgent [Ana24] studies lifelong learning to continuously improve LLM performance on advanced mathematics. ProverAgent [Bab25] leverages non-formal models to propose auxiliary lemmas that guide formal proofs. MA-LoT [Wan25an] applies a multi-agent, Lean-based long chain-of-thought approach with iterative proof repair via Lean compiler feedback. None of these systems are directly compared to *DAP* experimentally, as they target standard (Easy Mode) ATP tasks and their setups are not directly comparable.

<span id="section-10"></span>

## 10 Comparison between DAP and DSP/DSP+

DAP differs from DSP/DSP+ in three key respects. First, **target task**: DSP/DSP+ address standard ATP where the full formal statement is already given; DAP targets Hard Mode ATP, where the system must first discover the answer before proving. Second, **inter-module communication**: DSP passes intermediate reasoning steps (draft and proof sketch) to the formal prover; DAP passes only the final answer to the rewriting stage. Passing only the answer avoids reliance on particular formal-language idioms and sidesteps the problem that natural-language solution steps are often awkward or counterproductive when injected directly into tactic-based Lean proofs [Liu23s]. Additionally, DSP's design is tightly coupled to Isabelle's subgoal syntax, making it difficult to port to Lean, while DSP+'s sketch steps require single equations expressible as Lean `have` statements, which limits applicability. Third, **verification design**: DAP front-loads self-verification inside the Discovery Module before passing to the prover; DSP includes no self-verification, and DSP+ performs repair only when the formal sketch has a syntactic error.

<span id="section-11"></span>

## 11 Prompts used in Discovery Module

<span id="section-11-1"></span>

### 11.1 Prompt for Solution Generation

````text
### Core Instructions ###

* **Rigor is Paramount:** Your primary goal is to produce a complete and rigorously justified solution. Every step in your solution must be logically sound and clearly explained. A correct final answer derived from flawed or incomplete reasoning is considered a failure.
* **Honesty About Completeness:** If you cannot find a complete solution, you must **not** guess or create a solution that appears correct but contains hidden flaws or justification gaps. Instead, you should present only significant partial results that you can rigorously prove. A partial result is considered significant if it represents a substantial advancement toward a full solution. Examples include:
  * Proving a key lemma.
  * Fully resolving one or more cases within a logically sound case-based proof.
  * Establishing a critical property of the mathematical objects in the problem.
  * For an optimization problem, proving an upper or lower bound without proving that this bound is achievable.
* **Use TeX for All Mathematics:** All mathematical variables, expressions, and relations must be enclosed in TeX delimiters (e.g., `Let $n$ be an integer.`).

### Output Format ###

Your response MUST be structured into the following sections, in this exact order.

**1. Summary**

Provide a concise overview of your findings. This section must contain two parts:

* **a. Verdict:** State clearly whether you have found a complete solution or a partial solution.
  * **For a complete solution:** State the final answer, e.g., "I have successfully solved the problem. The final answer is..."
  * **For a partial solution:** State the main rigorous conclusion(s) you were able to prove, e.g., "I have not found a complete solution, but I have rigorously proven that..."
* **b. Method Sketch:** Present a high-level, conceptual outline of your solution. This sketch should allow an expert to understand the logical flow of your argument without reading the full detail. It should include:
  * A narrative of your overall strategy.
  * The full and precise mathematical statements of any key lemmas or major intermediate results.
  * If applicable, describe any key constructions or case splits that form the backbone of your argument.

**2. Detailed Solution**

Present the full, step-by-step mathematical proof. Each step must be logically justified and clearly explained. The level of detail should be sufficient for an expert to verify the correctness of your reasoning without needing to fill in any gaps. This section must contain ONLY the complete, rigorous proof, free of any internal commentary, alternative approaches, or failed attempts.

### Self-Correction Instruction ###

Before finalizing your output, carefully review your "Method Sketch" and "Detailed Solution" to ensure they are clean, rigorous, and strictly adhere to all instructions provided above. Verify that every statement contributes directly to the final, coherent mathematical argument.
````

<span id="section-11-2"></span>

### 11.2 Prompt for Self-Verification

```text
Below is the bug report. If you agree with certain item in it, can you improve your solution so that it is complete and rigorous? Note that the evaluator who generates the bug report can misunderstand your solution and thus make mistakes. If you do not agree with certain item in the bug report, please add some detailed explanations to avoid such misunderstanding. Your new solution should strictly follow the instructions in the system prompt.
```

<span id="section-11-3"></span>

### 11.3 Prompt for Self-Correction

````text
You are an expert AI assistant specializing in Formal Mathematics with the Lean 4 proof assistant. Your task is to perform a specific and targeted code modification.

You will be given three inputs:
1. A natural language math problem (`<NATURAL_LANGUAGE_PROBLEM>`).
2. A detailed natural language solution to that problem (`<NATURAL_LANGUAGE_SOLUTION>`).
3. A Lean 4 code statement that formalizes the problem (`<LEAN4_STATEMENT>`).

The Lean 4 statement contains a "fill-in-the-blank" definition for the solution, typically an `abbrev` or `def` with `sorry` as its value.

**Your task is to:**

1. **Read the `<NATURAL_LANGUAGE_SOLUTION>` to find the final answer** to the problem. The answer is usually explicitly stated at the end (e.g., "The value is π/2", "The result is 42").
2. **Locate the `abbrev` or `def` line** in the `<LEAN4_STATEMENT>` that defines the solution variable (e.g., `abbrev my_solution : ℝ := sorry`).
3. **Replace the `sorry`** in that definition with the final answer you extracted. Ensure the syntax is correct for Lean 4 (e.g., `π` becomes `Real.pi`).
4. **Output the entire Lean 4 code block with this single modification.** Do not change any other part of the code. Specifically, the `theorem`'s proof, which is also `sorry`, must remain unchanged.

**Example:**

**<NATURAL_LANGUAGE_PROBLEM>:**
Evaluate the integral \[\int_0^1 \frac{1}{1+x^2} \,dx.\]

**<NATURAL_LANGUAGE_SOLUTION>:**
The problem is to evaluate the definite integral of \(f(x) = \frac{1}{1+x^2}\) from \(0\) to \(1\).

The antiderivative of \(\frac{1}{1+x^2}\) is \(\arctan(x)\).
Using the Fundamental Theorem of Calculus, we have:
\[
\int_0^1 \frac{1}{1+x^2} \,dx = [\arctan(x)]_0^1
\]
\[
= \arctan(1) - \arctan(0)
\]
Since \(\arctan(1) = \pi/4\) and \(\arctan(0) = 0\), the value of the integral is:
\[
\frac{\pi}{4} - 0 = \frac{\pi}{4}.
\]
The final answer is \(\pi/4\).

**<LEAN4_STATEMENT>:**
```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

open Set

abbrev integral_value : ℝ := sorry

theorem integral_example
  : ∫ x in Icc 0 1, 1 / (1 + x^2) = integral_value :=
sorry
```

**Expected Output:**
```lean
import Mathlib.Analysis.SpecialFunctions.Trigonometric.Arctan

open Set

abbrev integral_value : ℝ := Real.pi / 4

theorem integral_example
  : ∫ x in Icc 0 1, 1 / (1 + x^2) = integral_value :=
sorry
```

---

Now, perform the task for the following inputs.

**<NATURAL_LANGUAGE_PROBLEM>:**
{{NATURAL_LANGUAGE_PROBLEM}}

**<NATURAL_LANGUAGE_SOLUTION>:**
{{NATURAL_LANGUAGE_SOLUTION}}

**<LEAN4_STATEMENT>:**
```lean
{{LEAN4_STATEMENT}}
```
````

<span id="section-11-4"></span>

### 11.4 Prompt for Rewriting

````text
You are an expert mathematician and a meticulous grader for an International Mathematical Olympiad (IMO) level exam. Your primary task is to rigorously verify the provided mathematical solution. A solution is to be judged correct **only if every step is rigorously justified.** A solution that arrives at a correct final answer through flawed reasoning, educated guesses, or with gaps in its arguments must be flagged as incorrect or incomplete.

### Instructions ###

**1. Core Instructions**
* Your sole task is to find and report all issues in the provided solution. You must act as a **verifier**, NOT a solver. **Do NOT attempt to correct the errors or fill the gaps you find.**
* You must perform a **step-by-step** check of the entire solution. This analysis will be presented in a **Detailed Verification Log**, where you justify your assessment of each step: for correct steps, a brief justification suffices; for steps with errors or gaps, you must provide a detailed explanation.

**2. How to Handle Issues in the Solution**
When you identify an issue in a step, you MUST first classify it into one of the following two categories and then follow the specified procedure.

* **a. Critical Error:**
  This is any error that breaks the logical chain of the proof. This includes both **logical fallacies** (e.g., claiming that `A>B, C>D` implies `A-C>B-D`) and **factual errors** (e.g., a calculation error like `2+3=6`).
  * **Procedure:**
    * Explain the specific error and state that it **invalidates the current line of reasoning**.
    * Do NOT check any further steps that rely on this error.
    * You MUST, however, scan the rest of the solution to identify and verify any fully independent parts. For example, if a proof is split into multiple cases, an error in one case does not prevent you from checking the other cases.

* **b. Justification Gap:**
  This is for steps where the conclusion may be correct, but the provided argument is incomplete, hand-wavy, or lacks sufficient rigor.
  * **Procedure:**
    * Explain the gap in the justification.
    * State that you will **assume the step's conclusion is true** for the sake of argument.
    * Then, proceed to verify all subsequent steps to check if the remainder of the argument is sound.

**3. Output Format**
Your response MUST be structured into two main sections: a **Summary** followed by the **Detailed Verification Log**.

* **a. Summary**
  This section MUST be at the very beginning of your response. It must contain two components:
  * **Final Verdict**: A single, clear sentence declaring the overall validity of the solution. For example: "The solution is correct," "The solution contains a Critical Error and is therefore invalid," or "The solution's approach is viable but contains several Justification Gaps."
  * **List of Findings**: A bulleted list that summarizes **every** issue you discovered. For each finding, you must provide:
    * **Location:** A direct quote of the key phrase or equation where the issue occurs.
    * **Issue:** A brief description of the problem and its classification (**Critical Error** or **Justification Gap**).

* **b. Detailed Verification Log**
  Following the summary, provide the full, step-by-step verification log as defined in the Core Instructions. When you refer to a specific part of the solution, **quote the relevant text** to make your reference clear before providing your detailed analysis of that part.

**Example of the Required Summary Format**
*This is a generic example to illustrate the required format. Your findings must be based on the actual solution provided below.*

**Final Verdict:** The solution is **invalid** because it contains a Critical Error.

**List of Findings:**
* **Location:** "By interchanging the limit and the integral, we get..."
  * **Issue:** Justification Gap - The solution interchanges a limit and an integral without providing justification, such as proving uniform convergence.
* **Location:** "From $A > B$ and $C > D$, it follows that $A-C > B-D$"
  * **Issue:** Critical Error - This step is a logical fallacy. Subtracting inequalities in this manner is not a valid mathematical operation.
````

<span id="section-11-5"></span>

### 11.5 Example of Self-Verification Error Report

The Self-Verification step prompts the LLM to carefully review the generated solution step by step and produce a structured error report. Each finding identifies a specific location in the solution, describes the issue, and classifies its severity. The following is a representative example of such an error report, produced by the Discovery Module on a functional-equation problem.

```text
Final Verdict: The solution contains several Justification Gaps and therefore is not fully rigorous.

List of Findings:
* Location: Lemma 1 proof — "h(x0) is an upper bound of the set {y | y < x0} ..."
  Issue: Justification Gap — the claim that h(x0) bounds all predecessors of x0 is not proved; the argument that this contradicts the supremum property is invalid.
* Location: Lemma 2 proof — "Both lines are proved by a simple induction on the natural variable."
  Issue: Justification Gap — the induction for the second coordinate is only sketched; it is not shown that the composed beta-functions satisfy the required monotonicity and domination properties.
* Location: Lemma 2 proof — base case "For n=0 ..." while N was earlier taken to start at 1.
  Issue: Minor Justification Gap — the indexing mismatch is not addressed, though it does not affect the overall argument.
```

<span id="section-12"></span>

## 12 Dataset Quality Fixes

**Porting FIMO to Lean 4:** The publicly available FIMO dataset is written in Lean 3. Although a largely automated migration tool exists, the syntactic and library-level changes between Lean 3 and Lean 4 mean that many statements do not port cleanly: tactic names changed, Mathlib APIs were reorganised, and some constructs require manual rewriting to compile while remaining semantically faithful. Our annotators ported every FIMO problem to Lean 4, verifying that each statement compiles and is semantically faithful to the original.

**Fixing Semantic Misalignments:** Following the annotation principles described in [Section 4.1](#section-4-1), annotators identified and repaired formalization errors present in the source datasets. We found approximately 15 misalignments in miniF2F and 20 in FIMO. [Table 5](#table-05) summarises the four most common error types.

<span id="table-05"></span>

![Table 5. Semantic misalignment types corrected in miniF2F and FIMO](../../papers/discover-and-prove/table-05.png)

**Table 5.** Semantic misalignment types found and corrected in miniF2F and FIMO during our re-annotation.

**Rephrasing for Hard Mode Compatibility:** Our annotation principle requires that any quantity a human competitor must derive must not be hard-coded in the formal statement. In some cases this meant non-trivially rephrasing a problem so that the unknown value becomes a free parameter and proper side-conditions (simplicity, squarefreeness, positivity, etc.) are added explicitly. A representative before/after example is shown in [Figure 3](#figure-03) ([Section 13](#section-13)).

<span id="section-13"></span>

## 13 Hard Mode Annotation Example

[Figure 3](#figure-03) shows a representative before/after example of rephrasing a problem for Hard Mode compatibility. The original formalization hard-codes `c = 2`, leaking part of the answer; the rephrased version promotes `c` to a free parameter and adds explicit simplicity and squarefreeness conditions, requiring the solver to determine the canonical form independently.

<span id="figure-03"></span>

![Figure 3. Rephrasing mathd_algebra_320 for Hard Mode compatibility](../../papers/discover-and-prove/figure-03.png)

**Figure 3.** Rephrasing `mathd_algebra_320` for Hard Mode compatibility. The original formalization hard-codes `c = 2`, leaking part of the answer; the rephrased version promotes `c` to a free parameter and adds explicit simplicity and squarefreeness conditions, requiring the solver to determine the canonical form independently.

<span id="section-14"></span>

## 14 Failure-Mode Analysis of the Discovery Module

To understand the limitations of the Discovery Module, we analyzed all Discovery outputs on the 45 solution-style problems from CombiBench and the 70 solution-style problems from FIMO. These two sets are moderate in difficulty (avoiding the saturation seen on miniF2F-Hard) and cover a broad range of competition mathematics areas (combinatorics, number theory, and algebra), making them well-suited for diagnosing real weaknesses. We manually categorized every Discovery failure and summarize the main error types in [Table 6](#table-06).

<span id="table-06"></span>

![Table 6. Failure modes of the Discovery Module on FIMO and CombiBench](../../papers/discover-and-prove/table-06.png)

**Table 6.** Failure-mode analysis of the Discovery Module on FIMO and CombiBench, with and without self-verification (SV). Counts indicate the number of problems exhibiting each failure type.

**FIMO: high difficulty in number theory and algebra.** Many FIMO problems remain beyond the model's capabilities; self-verification cannot fully recover correct answers for the hardest instances. Function-equation problems are particularly challenging and often fail even with verification, as the underlying reasoning requires multi-step algebraic manipulation that is difficult to check automatically.

**CombiBench: comprehension failures.** On CombiBench, we frequently observed that the model did not correctly understand problem statements or even hallucinated conditions that do not appear in the original problem. These comprehension failures are not substantially reduced by self-verification. We hypothesize that certain combinatorial phrasings — though natural to humans — are less accessible to current LLMs and lead to misparsing or misinterpretation.

**Self-verification helps but is not a cure-all.** For number-theory and algebra problems, self-verification notably reduces minor computational errors ($9 \to 4$) and some shallow-reasoning failures ($4 \to 2$), rescuing answers lost due to arithmetic mistakes. For combinatorics, self-verification helps reduce instances of incorrect reasoning foundations ($5 \to 3$) but does not eliminate comprehension or hallucination errors. Overall, self-verification is most effective when the error is a localized mistake (e.g., an arithmetic slip) rather than a fundamental misunderstanding of the problem.

<span id="section-15"></span>

## 15 Self-Verification Iteration Ablation

[Table 7](#table-07) quantifies the effect of the maximum number of self-verification iterations on Discovery Module accuracy.

<span id="table-07"></span>

![Table 7. Discovery Module accuracy under different self-verification iteration budgets](../../papers/discover-and-prove/table-07.png)

**Table 7.** Discovery Module accuracy (number of correctly solved problems) under varying maximum self-verification iteration budgets. Column headers indicate the iteration limit; 0 means no self-verification. $^\dagger$Our default setting. All results are Pass@32.

Two observations emerge. First, on miniF2F-Hard the Discovery Module achieves perfect accuracy even without self-verification, consistent with the saturation effect (though possible dataset contamination should be noted). Second, for harder benchmarks like PutnamBench and FIMO-Hard, self-verification provides substantial gains; 10 iterations approach saturation and balance computational cost with accuracy, while 30 iterations adds marginal gains and serves as our default.

<span id="section-16"></span>

## 16 Spurious Proof Example

[Figure 4](#figure-04) shows a concrete example of a spurious proof observed under the No Rewriting setting. The proof closes by `rfl` because the `abbrev solution` was defined to be literally the same set as the one appearing in the theorem statement — the prover never needed to reason about the underlying mathematics.

<span id="figure-04"></span>

![Figure 4. A spurious proof under the No Rewriting setting](../../papers/discover-and-prove/figure-04.png)

**Figure 4.** A spurious proof of `fimo_2009_algebra_p3` observed under the No Rewriting setting. The `abbrev solution` is defined as the same set that appears in the theorem statement, so the prover closes the goal with `rfl` without performing any mathematical reasoning.

<span id="section-17"></span>

## 17 Cross-Model Pairing

A practical advantage of *DAP*'s modular design is that the Discovery Module and the Proving Module can be replaced independently. To demonstrate this flexibility, [Table 8](#table-08) reports results for three model combinations: our default pairing, a lightweight pairing using smaller open-source models, and a high-resource pairing using a stronger closed-source model. Because the Aristotle API [Ach25] does not support concurrent requests and each problem can take hours, we sampled five problems per dataset for that condition; results are reported as solved/total.

<span id="table-08"></span>

![Table 8. Pass@32 results for different Discovery-Proving model pairings](../../papers/discover-and-prove/table-08.png)

**Table 8.** Pass@32 results for different Discovery-Proving model pairings. $^\dagger$Aristotle API results are sampled (5 problems per dataset).

Two conclusions follow. First, the pipeline is functional even with small, resource-efficient models, supporting use in compute-constrained settings. Second, replacing the Discovery Module with a significantly stronger model yields meaningful further gains, illustrating the headroom available as informal reasoning models continue to improve.
