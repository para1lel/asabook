---
title: 'CSLib: The Lean Computer Science Library'
createTime: 2026/08/28 14:50:26
permalink: /en/papers/cslib/
pageClass: paper-reading
---

> [Clark Barrett](https://theory.stanford.edu/~barrett/), [Swarat Chaudhuri](https://www.cs.utexas.edu/~swarat/), [Fabrizio Montesi](https://www.fabriziomontesi.com/), [Jim Grundy](https://dblp.org/pid/92/2259), [Pushmeet Kohli](https://dblp.org/pid/94/248), [Leonardo de Moura](https://leodemoura.github.io/about/), [Alexandre Rademaker](https://arademaker.github.io/), and [Sorrachai Yingchareonthawornchai](https://sites.google.com/view/sorrachai). First submitted to arXiv on February 4, 2026; the current version is v1, dated February 4, 2026. [CSLib: The Lean Computer Science Library](https://arxiv.org/abs/2602.04846v1). <a href="/paper/cslib.pdf" target="_blank" rel="noopener noreferrer">Original PDF</a>. [DOI](https://doi.org/10.48550/arXiv.2602.04846). [TeX source](https://export.arxiv.org/e-print/2602.04846v1). The original PDF remains authoritative for the exact print layout and bibliography.

## Abstract

We introduce CSLib, an open-source framework for proving computer-science-related theorems and writing formally verified code in the Lean proof assistant.
CSLib aims to be for computer science what Lean’s Mathlib is for mathematics. Mathlib has been tremendously impactful: it is a key reason for Lean's popularity within the mathematics research community, and it has also played a critical role in the training of AI systems for mathematical reasoning. However, the base of computer science knowledge in Lean is currently quite limited. CSLib will vastly enhance this knowledge base and provide infrastructure for using this knowledge in real-world verification projects. By doing so, CSLib will (1) enable the broad use of Lean in computer science education and research, and (2) facilitate the manual and AI-aided engineering of large-scale formally verified systems.

<span id="section-1"></span>

## 1 Introduction

On its face, Mathlib [Mat20b] is a library of mathematical results and techniques formalized in the Lean [Mou21] proof assistant. In reality, it is much more than that—it is a community effort by mathematicians all over the world to collect the world's mathematical knowledge in a form that can easily be vetted and shared. Mathlib provides a platform for a new way to do mathematics, where many people can confidently collaborate on large and challenging projects [Rin24]. Also, by representing a wide range of mathematical definitions and proofs in a machine-checkable form, it opens up new opportunities for AI systems to aid the process of mathematical discovery [Yan24k, Hub26]. This new approach to mathematics is being embraced by a new generation of mathematicians and has gained many prominent champions [Tao24b, Kon25a]. Mathlib is at the heart of this paradigm shift, and as such, it is hard to overstate its impact.

Computer science has always been close to mathematics. Many computer science departments originally emerged as offshoots of mathematics departments, and even today, the distinction between computer science theory and mathematics is blurry. Moreover, computer scientists have long recognized formal mathematics as a power tool that enables the design of reliable and secure systems [Boy83, Cla97, Hoa03]. However, until now there has been no effort to build a Mathlib-like universal repository of formalized computer science knowledge.

In this white paper, we introduce CSLib, [+1] a library for Lean-based formal verification and computer science research that seeks to change this state of affairs. CSLib has two “pillars”:

1. *Formalizing all essential computer science concepts in Lean*. The formalizations include, but are not limited to, models of computation such as the $\lambda$-calculus and resource-bounded Turing machines; numerous algorithms and data structures accompanied by proofs of correctness and complexity; concurrency theory; foundations of programming languages; and mathematical tools relevant to computer science that do not currently appear in Mathlib.
2. *Building an infrastructure for Lean-based reasoning about everyday imperative code*. This infrastructure consists of an intermediate programming language, called Boole, that allows classical imperative constructs to be interspersed with specifications written in Lean; machinery for generating Lean-language verification conditions from Boole code; and verified Boole implementations of the algorithms and data structures covered in Pillar 1.

[+1]: The main CSLib website is at [https://cslib.io](https://cslib.io). The current CSLib codebase is at [https://github.com/leanprover/cslib/](https://github.com/leanprover/cslib/).

Collectively, the Lean code and proofs in CSLib enable the use of Lean as a medium for mathematically grounded computer science research and development. For example, we can imagine a theory researcher using CSLib's built-in abstractions to quickly mock up a novel approximation algorithm and theorems capturing its essential properties, a programming languages researcher using the framework to build verified compilers, or a systems researcher using the library to model new network protocols with worst-case guarantees. The Boole framework offers a crucial bridge from Lean to everyday programming languages. Specifically, we envision a world in which code in
mainstream languages like Rust and C++ is automatically translated into Boole at scale.
CSLib's built-in infrastructure turns the task of verifying these Boole-language programs into that of proving Lean-language theorems. This proof task is then solved using Lean-language proof machinery.

**CSLib's Impact.** We see the impact of CSLib taking two forms: addressing an urgent practical need for reliable software systems, and fulfilling a long-term need for more rigorous and scalable computer science.

On the first point, progress on software reliability and security is more time-critical than ever because of new risks that developments in AI pose to the world's computing infrastructure. Today's AI systems have a remarkable ability to analyze code, making it much easier for an AI-powered malicious actor to find and exploit vulnerabilities. At the same time, AI coding agents are writing more and more of the world's code. While this leads to overall productivity gains, AI can also introduce correctness errors and security vulnerabilities into code [Per23]. Formal methods can mitigate these risks by
mathematically proving code—whether human-written or AI-generated—to be free of bugs and vulnerabilities.

However, the impact of formal methods has been historically constrained by the immense human effort that they require.
Consider seL4 [Kle09], the world's first formally verified operating system kernel.
The system had a profound impact: an seL4-powered drone in the DARPA HACMS program famously resisted six weeks of sustained attacks by an elite red team, leading the drone to be called the “most secure drone in the world.” But seL4 also highlights the cost penalties associated with traditional formal verification. The system required over 20 person-years of proof effort and included about 480,000 lines of formal specification and proof in the Isabelle [Nip02] framework. Such costs are difficult to justify in most commercial settings.

The CSLib vision addresses this bottleneck in two ways. Once CSLib has been built, a developer would be able to derive reliable and secure systems compositionally, by putting together pre-verified CSLib components. Second, CSLib can simplify the use of AI in formal verification. AI systems have already demonstrated superhuman formal proof capabilities in the pure mathematics setting [Hub26]. However, to leverage these capabilities in broad-domain code verification, we need a comprehensive language in which to specify code properties. Lean and CSLib will provide this language. In addition, CSLib will serve as a repository of high-quality data on which to train AI systems for formalization and formal proof.

Beyond this impact on practical formal verification, CSLib will serve as a platform for 21st-century computer science research. The formal verification that CSLib enables will induce confidence in claims and allow the large-scale reuse of foundational concepts. We can also imagine AI tools trained on CSLib discovering novel algorithms and settling open claims, extending to computer science the role that AI agents are beginning to play in mathematics research.

<span id="figure-01"></span>

![Figure 1. Two topic trees show a possible organization for the models and logics, algorithms, and data structures covered by CSLib's Pillar 1.](../../papers/cslib/figure-01.png)

**Figure 1.** (a) Models & logics. (b) Algorithms and data structures. CSLib's Pillar 1: A possible organization of covered topics.

<span id="section-2"></span>

## 2 Technical Approach

Now we provide some more details on the two pillars of CSLib's technical approach.

<span id="section-2-1"></span>

### 2.1 Pillar 1: Formalizing Computer Science in Lean

CSLib's first development pillar uses Lean to formalize a comprehensive body of models of computation, algorithms, and data structures, along with properties of these artifacts and proofs of these properties. The formalizations are designed to form a coherent and integrated framework rather than a collection of unrelated modules. Such unification leads to a whole that is greater than the sum of its parts and has been key to Mathlib's impact.

The pillar has a synergistic relationship with Mathlib. On the one hand, it heavily uses Mathlib modules for, say, big-O reasoning and probability theory. On the other hand, it formalizes some core mathematics—for example, certain lemmas from combinatorics needed to analyze approximation algorithms or inequalities needed to prove the convergence of optimization algorithms—that are useful in computer science but do not currently appear in Mathlib. Depending on the circumstances, we may either contribute these formalizations to Mathlib or keep them within the CSLib repository.

Now we provide some more details on this pillar's targets.

**Models & Logics.** Over the years, computer scientists have designed a wide range of *models of computation*. These include models for deterministic, nondeterministic, probabilistic, and quantum computation; models for functional computations, online algorithms, and interactive protocols; models that solve decision problems; and models that approximately solve optimization problems. Many of these models have never been formalized; the ones that have tend to exist in isolated repositories. A key goal of CSLib's Pillar 1 is to create a unified body of formalizations of these models.

Another piece is the formalization of *specification notations* for models and algorithms. For example, temporal logics are natural for specifying properties of stateful systems; Hoare logic is a standard mechanism for specifying procedural code; separation logic simplifies the specification of low-level imperative computations; linear logic is suitable for reasoning about resource usage and concurrent behaviors. At this point, there is no unified codebase consisting of such specification notations, and CSLib will address this problem.

<span id="figure-02"></span>

![Figure 2. Lean definitions for a labeled transition system and bisimulation, followed by a theorem about inverse bisimulations.](../../papers/cslib/figure-02.png)

**Figure 2.** (a) Definition of a Labeled Transition System (“LTS”), which models the observable behavior of the possible states of a discrete computational system. (b) Definition of a bisimulation between two states. (c) Theorem establishing that the inverse of a bisimulation is a bisimulation.

<span id="figure-03"></span>

![Figure 3. Lean code defines the TimeM monadic API and simplification lemmas for elementary algorithm complexity analysis.](../../papers/cslib/figure-03.png)

**Figure 3.** A monadic API for complexity analysis of elementary algorithms. An object of type $\texttt{TimeM}~\alpha$ comprises $\texttt{ret}$, which is a value of type $\alpha$, and the $\texttt{time}$ cost incurred by the computation of this value. The $\texttt{bind}$ operation chains together two computations and defines the time cost of the composition as the sum of the time costs of the individual operations. The helper function $\texttt{tick}$ assigns the time cost. The notation $\checkmark$ is syntactic sugar for the invocation of $\texttt{tick}$. The final section asserts some basic lemmas that help simplify expressions involving $\texttt{TimeM}$.

<span id="figure-04"></span>

![Figure 4. Lean implementation of Mergesort using TimeM, with correctness and time-complexity theorem statements.](../../papers/cslib/figure-04.png)

**Figure 4.** An implementation of Mergesort using the API in [Figure 3](#figure-03). Note how each invocation of the sorting routine returns a sorted list along with a computational cost. With these definitions in place, one can formally state and prove theorems establishing the correctness of Mergesort and its worst-case number of comparisons.

While the specific set of formalizations that we will build in this pillar will necessarily evolve over time,
[Figure 1](#figure-01)-(a) spells out some of the models and logics we currently plan to cover. [Figure 2](#figure-02)-(a) and [Figure 2](#figure-02)-(b) give concrete examples of formalizations. Specifically, [Figure 2](#figure-02)-(a) shows (part of) a Lean definition of labeled transition systems [Win95], a classic model for stateful systems, and [Figure 2](#figure-02)-(b) defines bisimulations, a key notion of equivalence between system states. If two states $s_1$ and $s_2$ are related by a bisimulation, then $s_1$ can mimic all transitions of $s_2$ by a corresponding transition and vice versa, and the states reached through these original and mimicking transitions remain related by the bisimulation.

Transition systems and bisimulations have numerous applications in computer science, in areas from hardware design [Che96] to control [Gir11] to machine learning [Zha21i]. Formal definitions of these concepts and associated theorems (such as the theorem in [Figure 2](#figure-02)-(c)) are therefore valuable both from a theoretical and a practical perspective. However, such formalizations are not naturally in scope for Mathlib and are thus an example of the distinct capabilities that CSLib will provide.

**Algorithms & Data Structures.** Pillar 1 will also use Lean to build a comprehensive repository of formally verified algorithms and data structures. Our long-term ambition is for this library to cover all of computer science. In the medium term, we aim for this library to cover all algorithms and data structures that a typical computer science graduate is likely to encounter. [Figure 1](#figure-01)-(b) names some of the algorithmic categories that we plan to cover.

A key objective in this thread is to formalize analysis of algorithms, including, in particular, the *complexity* of algorithms.
We note that there is much prior work on automatic, lightweight complexity analysis of programs [Gul09, Ros89], and some work on proof-assistant-based complexity verification of specific algorithms [Cha19a, Cha21b] and formalization in Isabelle [Nip25]. However, we are only aware of one effort [Irv25] on the systematic formalization of algorithmic complexity in Lean. That formalization [Bro23] was focused on stochastic oracle computations; by contrast, we aim for a universal framework for complexity analysis.

The precise definition of the framework that we propose will require more work and evolve over time. For now, we show in
[Figure 3](#figure-03) a simple monadic API, akin to Mathlib's Writer monad and inspired by prior work on analysis of functional algorithms [Dan08, Gib11], that will be part of CSLib's complexity analysis framework. The API packages return values of procedures with the cost of computing these values;
see [Figure 4](#figure-04) for an example of its use.
Treating time complexity as a computational effect makes correctness and complexity analysis orthogonal.
Also, by giving users control over tick placement, the framework enables experimentation with different cost models, enabling a diverse set of complexity analyses.

At the same time, the API has multiple limitations. For example, it relies on manual tick annotations that a user could accidentally get wrong rather than automated verification of execution costs. Also, it cannot directly prove statements of the form “no algorithm can solve problem X faster than $f(n)$.” However, because it is lightweight and allows proof of many interesting theorems about algorithmic complexity, it is a good starting point for our efforts. In the longer run, we plan to complement this approach with heavier-weight methods that formalize complexity via explicit RAM and query models.

<span id="section-2-2"></span>

### 2.2 Pillar 2: Infrastructure for Reasoning about Everyday Code

CSLib's second pillar will develop a powerful new infrastructure for reasoning about everyday imperative code. Lean has recently added substantial support for the definition [Ull22] and verification [+2] of imperative programs, and we intend to leverage these advances in our Pillar-1 efforts. The methods we pursue in the present pillar are complementary in that they connect Lean-based formal reasoning to the verification of code in mainstream languages like Rust, Python, and C++.

[+2]: [The `mvcgen` tactic](https://lean-lang.org/doc/reference/latest/The--mvcgen--tactic/).

Our approach builds on the rich tradition of deductive verification techniques and tools [Flo67, Hoa69], in particular systems built on top of intermediate verification languages (IVLs). Perhaps best-known is the Boogie IVL [Lei08, Lei10], which is used by many systems (e.g., Dafny [Lei10a], Viper [Juh14], CIVL [Kra21], and the Move Prover [Zho20]).
CSLib's Pillar 2 will be based on a new IVL called *Boole*, partly inspired by Boogie, but also leveraging the capabilities and advantages of Lean.

<span id="figure-05"></span>

![Figure 5. CSLib code-reasoning vision from source programs and Lean specifications through Boole, verification-condition generation, and Lean hammers.](../../papers/cslib/figure-05.png)

**Figure 5.** CSLib Code Reasoning Vision.

[Figure 5](#figure-05) illustrates the long-term vision for CSLib-based code reasoning. We envision a framework with support for multiple front-end languages. A user can write their code accompanied by specifications in Lean. These are then transpiled (translated via compilation) to Boole. We plan to build a deductive verification platform built around Boole, which can use various algorithms and back-ends to generate verification conditions in Lean. These can then be discharged using various Lean automation techniques.

<span id="figure-06"></span>

![Figure 6. A Boole program in Strata computes the sum of the first n positive integers and invokes Lean verification-condition generation.](../../papers/cslib/figure-06.png)

**Figure 6.** A simple program, written in the Boole dialect of Strata, for computing the sum of the first $n$ positive integers.

This vision is guided by several forward-looking design principles, which aim to avoid shortcomings and pitfalls revealed by previous work:

1. Boole should look like pseudocode and be easy for humans to understand;
2. Boole and its ecosystem will be embedded in Lean, and both the transpilation step and the verification condition generation step should be implemented with the goal of minimizing the trusted computing base (TCB);
3. verification conditions should be generated as Lean goals and should be intuitive for humans to read and easily connectable to the code they came from; and
4. code specifications and verification conditions should be able to leverage the rich formal universe defined in CSLib's Pillar 1.

Verification condition generation from Boole will build on infrastructure and ideas from other open source projects, such as Loom [Gla26] and Strata. Loom [+3] provides a mechanism for the definition of shallowly-embedded languages in Lean (including a Dafny-like language called Velvet), with strong guarantees about correct verification condition generation. Strata [+4] is an open-source project that aims at facilitating the creation of domain-specific languages called *dialects*, which are *deeply* embedded in Lean.

[+3]: [Loom repository](https://github.com/verse-lab/loom).

[+4]: [Strata repository](https://github.com/strata-org/Strata).

<span id="figure-07"></span>

![Figure 7. Four Lean goals encode verification conditions for the Boole summation program.](../../papers/cslib/figure-07.png)

**Figure 7.** Lean verification conditions generated from the program in [Figure 6](#figure-06).

[Figure 6](#figure-06) shows a simple snippet of Lean code containing a program written in Boole. Boole is implemented as an extension of the Core dialect of Strata, which in turn is largely modeled after the classic Boogie IVL. After a prelude that imports and opens relevant packages, the program itself can be written in an easily readable form. Notice the `spec` keyword, which signals a specification. In this case, there is a requirement (precondition) that $n$ should be non-negative indicated by the `requires` keyword. We can also specify a loop invariant using the `invariant` keyword as well as more general program invariants that should hold at specific points in the code using the `assert` keyword.

We can *verify* the assertions by translating the specifications and code into verification conditions. This can be done by assigning Lean semantics to each construct in the program and then following the standard deductive verification approach, which involves proving, for certain loop-free paths in the program, that for all possible executions along that path, the precondition for the path ensures the postcondition of the path.
The `#prove_vcs` command in our example invokes a prototype verification condition generator that generates goals in Lean. A set of generated goals is shown in [Figure 7](#figure-07). The first goal, for example, represents a path from the entry to the procedure to the beginning of the loop. The precondition of the program requires $n \ge 0$, and the loop condition must be true, implying $i < n$. These should imply the loop invariant. Our verification condition generator performs some basic simplification—in this case, it knows that $i=0$ when entering the loop, so it replaces $i$ with $0$ and simplifies. The result is the goal shown. Similarly, the second goal represents the verification condition stating that the invariant is preserved around one iteration of the loop, and the other two goals correspond to the two assertions.

Goals in Lean can be proved by hand or by invoking *hammers*, which are general-purpose tactics for automating proofs of Lean goals. A number of potential hammers are shown at the bottom of [Figure 5](#figure-05). For our example, a tactic called `smt` can solve all of them. This tactic uses the Lean-SMT tool [Moh25] to translate a goal into an SMT formula, call a solver, get a proof certificate, and then replay the proof step-by-step in Lean. The number and capabilities of such hammers have been growing rapidly, and the future promises even stronger hammers, including AI-based ones.

The long-term roadmap for CSLib will involve extending Boole over time to add more functionality, including cost semantics (to support reasoning about and proving theorems about computational complexity), support for specifications using definitions from Mathlib and Pillar 1 of CSLib, support for reasoning about concurrency, and support for reasoning about low-level constructs such as pointers.
We will consider modern programming abstractions and verification techniques, for example, as found in research on choreographic programming [Mon23, Rub24] and separation logic [Jun18, Ohe19].

More significantly, we also intend for Boole to be used as a true *intermediate* language for the verification of real systems in standard programming languages. The plan is to formalize the semantics of real programming languages in Lean, use those semantics to translate code and specifications in those languages into Boole code and specifications, and then use the verification tools we are building for Boole to verify the translated code.
The Aeneas project [Ho22a], which translates a sizeable subset of Rust into several proof assistants (including Lean), offers a model of such a pipeline.

A final concrete goal of this pillar is to assemble a large library of code that has been certified using the CSLib verification infrastructure. We plan to make this library freely available and to integrate it with synergistic efforts such as VeriLib. [+5]

[+5]: [VeriLib](https://verilib.org/).

<span id="section-3"></span>

## 3 The Role of AI

The CSLib community effort is motivated by the ongoing revolution in AI-based code generation in two ways. On the one hand, we believe CSLib can be a part of our collective response to the *risks* of AI: the ability of AI-powered malicious actors to break the world's software and the possibility that engineers using AI-based coding tools would unwittingly write vulnerable code. These risks make rigorous software verification and formally verified code generation more important than ever, and CSLib will enable this. On the other hand, AI projects such as AlphaProof [Hub26], Deepseek-Prover [Ren25b], and Aristotle [Ach25] showcase the ability of AI to solve hard formal proof tasks. However, AI-based provers are bottlenecked by the abstractions available in the underlying proof assistant. By expanding the universe of CS-relevant abstractions available in Lean, CSLib will also vastly enhance the capabilities of such provers in formal verification settings.

The formalizations produced in CSLib will serve as high-quality training data for AI-based formalization and theorem-proving systems [Hub26, Ren25b, Lin25i]. At the same time, we expect that AI-based tooling for Lean proofs will vastly accelerate the development of CSLib. The ideal outcome would be a “flywheel” in which both AI-based proof tools and human experts become progressively more efficient at producing new knowledge.

One risk with the use of AI in a sophisticated development effort like CSLib is that AI-generated formalizations might be buggy. Also, while AI-generated Lean proofs of formal statements may be sound, they may not be human-comprehensible. We will mitigate these risks by ensuring that AI tools play only an advisory role in the development of CSLib, and that all formalizations and proofs that are committed to CSLib are subject to a manual, Github-based code review process.

<span id="section-4"></span>

## 4 Building a CSLib Community

A library is only as impactful as its user base. While formal verification is a vibrant subarea of computer science, we aim for CSLib's impact to transcend that research area. We would like computer science researchers of all sorts, from theorists to system-builders, to use CSLib in their everyday work. We want CSLib to drastically reduce the cost of formal verification to the extent that engineers in a broad range of industries choose to embrace formal techniques. We would like CSLib to be used in a broad swathe of universities to teach the mathematical underpinnings of computer science.

Fortunately, we can look to Mathlib [Baa25] as a blueprint for achieving our goals. The Mathlib community has demonstrated that open discussion and community activities are essential to making a large-scale formalization project such as ours successful. Following their lead, we have set up a CSLib channel [+6] on the Lean Zulip chat.
This channel is already quite active, and we encourage everyone interested in the project to join it. We also urge you to keep an eye out for the calls for contributions to specific components of CSLib that we plan to issue in the coming months. We also plan to hold many workshops and tutorials on CSLib in the coming months and years. If you are interested in joining or leading such an event, please reach out to us!

[+6]: [CSLib channel on Lean Zulip](https://leanprover.zulipchat.com/#narrow/channel/513188-CSLib).

Another dimension is the use of rigorous code review practices and the creation of tooling that lowers the cost of CSLib development and maintenance.
In particular, AI tools for automatic formalization and formal theorem-proving have progressed at a breakneck pace over the last few years. As mentioned in [Section 3](#section-3), a goal of the CSLib effort is to both leverage and improve such tools. We expect these tools to substantially lower the barrier to developing and using CSLib.

Dependently typed languages have historically existed in a separate universe from mainstream imperative languages, and this is a key reason why their adoption has been limited. CSLib's Pillar 2 aims to change this status quo. The engineering of translations from mainstream languages like Rust and C++ to Boole will be especially important for the success of this vision. We encourage community members to take on the challenge of building these translations.

Finally, we will prioritize building high-quality, searchable documentation for CSLib. Such documentation will include tutorials targeted specifically at researchers working in different areas of computer science, as well as entry-level textbooks. Please write to us if you would like to lead or contribute to such a textbook.

<span id="section-5"></span>

## 5 Governance Model

CSLib is governed by a dual-body structure designed to balance strategic direction with technical execution. A steering committee comprising leaders from academia and industry is responsible for securing financial support and guiding the project's overall vision. The founding members of this committee are Clark Barrett (Stanford University & Amazon Web Services), Swarat Chaudhuri (Google DeepMind & UT Austin), Jim Grundy (Amazon Web Services), Pushmeet Kohli (Google DeepMind), Fabrizio Montesi (University of Southern Denmark), and Leonardo de Moura (Lean FRO & Amazon Web Services). Working alongside this committee, a maintainer team manages the codebase's technical direction, quality standards, and day-to-day development.
As of the date of this paper, the team is headed by a lead maintainer (Fabrizio Montesi), who coordinates overall efforts, and includes technical leads (Alexandre Rademaker and Sorrachai Yingchareonthawornchai) for long-term cross-cutting developments, and area maintainers (Chris Henson and Kim Morrison) taking ownership of specific domains such as $\lambda$-calculi, metaprogramming, and CI/CD infrastructure.

Naturally, we expect the maintainer team and the steering committee to change as the community grows. Specifically, we plan to periodically invite new maintainers based on merit—particularly, contributions and review activity—and project needs. The goal is to ensure strategic coherence through the steering committee's guidance and technical excellence through the maintainer team's specialized expertise, while fostering a welcoming environment for a broad set of contributors.

<span id="section-6"></span>

## 6 Roadmap

Now we present a brief overview of how we see the CSLib project evolving over 2026 and 2027. This roadmap is tentative given that the project is currently only a few months old and that we plan to refine its scope with community input. A more detailed (and periodically revised) roadmap can be found on the CSLib website: [https://cslib.io](https://cslib.io).

**2026.** The CSLib repository already includes initial Lean formalizations of operational semantics, program equivalences, several automata models, a linear logic, and a few elementary sorting and searching algorithms. We expect that by the end of 2026, CSLib will include formalizations of many algorithms covered in a typical undergraduate algorithms and data structures course, and most of the models and logics covered in a typical undergraduate theory of computation course. In addition, community members will be welcome to contribute formalizations from specialized research areas that they work on. Note that progress on this Pillar-1 task does not depend on Boole to be ready.

As for Pillar 2, we have already developed an initial version of Boole on top of the Strata framework. Over the next year, we expect the framework to mature substantially. By the end of 2026, the Boole framework will have robust capabilities for generating Lean-language verification conditions and interacting with SMT-based hammers. By that point, the CSLib repository will include Boole-language representations of a sizeable number of elementary algorithms and data structures, along with Lean specifications and proofs of all relevant verification conditions.

**2027.** The project's second year will focus on scale and unification. In particular, our Pillar-1 efforts will move on to technically challenging topics such as complexity theory, concurrency, secure compilation, and randomized and quantum algorithms. We expect that as a result, CSLib will have substantially aided at least one important foundational discovery in computer science by the end of 2027. In our Pillar-2 efforts, we will extend Boole to support machinery, such as separation logic, that is needed for the verification of low-level systems code, and extensively use the Pillar-1 machinery to write more advanced system specifications. We expect that by the end of 2027, CSLib will have enabled the end-to-end verification of at least one substantially sized real-world system.

## Acknowledgements

We thank all current and future contributors to CSLib [+7] for their work on the project, and Eric Wieser and Tom Kalil for their thoughtful feedback on this paper.

[+7]: A list of current CSLib contributors is available at [https://github.com/leanprover/cslib/graphs/contributors](https://github.com/leanprover/cslib/graphs/contributors).
