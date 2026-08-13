# ASa Book Paper Style Guide

Apply these rules to English, Simplified Chinese, and Japanese paper pages. Treat the paper's TeX source and PDF as authoritative when an example does not cover a case.

## Contents

- [Page Structure](#page-structure)
- [Citations and Abbreviations](#citations-and-abbreviations)
- [Content Annotations](#content-annotations)
- [Formal Statements and Proofs](#formal-statements-and-proofs)
- [Hyphens and Dashes](#hyphens-and-dashes)
- [Math](#math)
- [Algorithms](#algorithms)
- [Figures and Tables](#figures-and-tables)
- [English](#english)
- [Simplified Chinese](#simplified-chinese)
- [Japanese](#japanese)
- [Final Consistency Pass](#final-consistency-pass)

## Page Structure

- Use the same concise English title in all languages and keep it at 50 characters or fewer. Preserve the full source title in the provenance link when shortening it.
- Preserve numbered section and subsection structure unless a heading exists only for print layout rather than content. Remove an empty generic `Appendix`, `附录`, or `付録` heading when the next heading is the actual Appendix A section.
- Preserve citations and equation, figure, table, and algorithm numbering.
- Reproduce the complete substantive paper in source order. Do not abridge, summarize, expand, or omit body material, but omit the standalone reference list.
- Render a source run-in paragraph heading such as `\paragraph{Heading.}` at the start of the same Markdown paragraph: `**Heading.** Paragraph text...`. Keep exactly one space after the closing bold marker; never separate the heading from its paragraph with a blank line or line break.
- Preserve the wording and punctuation of run-in headings in English and translate them faithfully in Chinese and Japanese. Keep actual section headings, figure and table captions, algorithm titles, and labels introducing block content as separate blocks.
- Use one shared set of image files. Localize alt text and captions, not the pixels in an original scholarly figure.
- Keep author links identical across languages and in the paper's original order.

## Citations and Abbreviations

- Write citations as the repository's compact keys, for example `[Hu21]` or `[Ope24a, Tou23]`.
- Define every cited key in `paperAbbreviations` in `docs/.vuepress/config.ts`; this enables the visible dotted underline and hover explanation.
- Include citations attached to expanded terms, for example `Low-Rank Adaptation (LoRA) [Hu21]`.
- Preserve a paper's reference identity across languages. Never translate titles in bibliography definitions.
- Prefer the canonical paper page, DOI, conference page, or arXiv abstract as the bibliography link.
- Do not render `References`, `参考文献`, or any other standalone reference-list section in a paper page. Keep inline citations and their `paperAbbreviations` definitions.

## Content Annotations

- Use Plume content annotations for all source footnotes; never use Markdown footnotes.
- Insert a marker as `[+label]` with a space before it. Keep the marker outside adjacent bold or italic markup.
- Define its content separately as `[+label]: content`. Indent continuation lines by two spaces.
- Use the same stable labels in English, Chinese, and Japanese, and preserve every link and detail from the source footnote.
- Confirm `markdown.annotation` is enabled and verify the built page renders each marker as an interactive annotation.

## Formal Statements and Proofs

- Match the published paper's run-in treatment of numbered definitions, lemmas, propositions, corollaries, theorems, and claims. Bold only the localized label and number, then begin the statement after exactly one space on the same Markdown line.
- Use `**Theorem 3.8.** Statement...`, `**定理 3.8.** 陈述...`, and `**定理 3.8。** 文...`. Do not use Markdown headings, an isolated bold-only line, or a blank line between the label and statement.
- Keep the label and first clause together even when the statement continues across several visual lines. Use `<br>` for intentional line breaks within that paragraph instead of Markdown trailing spaces. Do not put the label alone above `$$...$$`.
- Preserve stable anchors for linked statements, but place the anchor in its own line immediately before the run-in paragraph; the anchor does not change the visual run-in requirement.
- Convert each source proof environment or standalone proof passage to a default-closed Plume details container. Localize only the summary label: `Proof` in English, `证明` in Simplified Chinese, and `証明` in Japanese.
- Put the entire proof inside one container, including multiple paragraphs and display equations. Do not add an `open` option or equivalent default-expanded state.
- Omit the original visible proof prefix and terminal QED marker. Do not retain `Proof.`, `证明.`, `証明。`, `∎`, `□`, or `\square` inside or around the container.
- Leave ordinary prose that cites, summarizes, or sketches a proof outside a container unless the source presents it as an independent proof passage.

English example:

```markdown
<span id="theorem-03-08"></span>

**Theorem 3.8.** The expected improvement factor is $f(\alpha,\gamma,c)$.

::: details Proof
Let the cost of one target-model step be $T$. The result follows by dividing the total cost by the expected number of generated tokens.
:::
```

Localized summaries use the same body structure:

```markdown
::: details 证明
证明正文.
:::

::: details 証明
証明本文。
:::
```

## Hyphens and Dashes

- Never render consecutive ASCII hyphens in article headings, prose, captions, alt text, annotations, algorithms, or table content.
- Use one ASCII hyphen (`-`) when the mark links words, tokens, symbols, or the endpoints of a range, for example `producer-consumer`, `$\alpha$-$\beta$`, and `7.5-18.0%`.
- Use the full-width em dash (`—`) when the mark functions as a dash within a sentence.
- Treat conversion of source `--` and `---` sequences to these forms as a required Markdown representation change in every language, including English.
- Do not alter structural syntax such as YAML frontmatter fences, Markdown table separator rows, fenced or inline code, URLs, TeX commands, or mathematical minus signs.

## Math

- Use `$...$` for inline math and `$$...$$` for display math. Keep `\tag{N}` inside the same display-math delimiters as its equation.
- Write every matrix transpose as `^\top`, never `^{T}`.
- Verify every numbered display equation in the built page. A literal `\tag{N}` in prose or outside math is invalid.
- Put prose words and abbreviations inside `\mathrm{}` in formulas, including textual subscripts and superscripts. Do not romanize multi-index variable names such as `a_{ij}`, `A_{ij}`, or `jB`; these are mathematical symbols, not words. The checker reports possible unwrapped words as warnings for review rather than hard failures. Examples: `X_{\mathrm{FP}32}`, `X^{\mathrm{unscaled}}`, `\mathrm{round}(x)`, and `\mathrm{LZD}(b)`.
- Leave single-letter mathematical variables such as `x`, `m`, `W`, and `R` unwrapped.
- Use semantic built-in operators where available. In particular, use `\min` and `\max`, never raw `min`/`max`, `\mathrm{min}`, or `\mathrm{max}`.
- Use `\|x\|` for every norm. Never write `||x||`, `\Vert x\Vert`, or `\lVert x\rVert`.
- Use `\texttt{}` only for literal code symbols and bit operations, not mathematical names. Use `\mathbin{\texttt{<<}}` or `\mathbin{\texttt{>>}}` when shifts act as binary operators.
- Preserve source macros only when KaTeX supports them in VuePress. Expand unsupported macros into standard KaTeX commands.
- Keep punctuation outside inline math when it belongs to the sentence. Preserve commas inside display math when they are part of the expression.

## Algorithms

- Render math-heavy algorithms as unordered Markdown lists, not `pseudocode` fences.
- Put the algorithm title in bold above the list.
- Use nested list indentation to express branches, loops, and substeps.
- Put control words in bold localized prose and keep variables or expressions in KaTeX.
- Use inline code only for actual literals, masks, slices, and programming operators.
- Preserve the original execution order, inputs, outputs, conditions, and return value.
- Use compact unordered Markdown lists beginning with `-`; do not number algorithm steps.
- Indent nested loops, branches, and substeps by two spaces per level. Indentation expresses scope, so omit `end for`, `end if`, `end while`, and translated equivalents.
- Bold localized control words such as **For**, **If**, **Return**, and their Chinese or Japanese equivalents.
- Existing fenced `pseudocode` algorithm blocks may remain unchanged. For new algorithms, use a fenced code or pseudocode block only when the source is genuinely code-oriented, math is incidental, and the existing collection style supports it.

Example:

```markdown
**Algorithm 1: Quantization to lower-bit floating-point values.**

- **Input:** $X_{\mathrm{FP}32}$ and $s\in\mathbb{R}^{+}$.
- **Check overflow:**
  - **If** $x>\max(a,b)$:
    - Set $x\leftarrow\max(a,b)$.
- **Return:** $X_{\mathrm{FP}8}$.
```

## Figures and Tables

- Prefer source image files for figures when they reproduce the published figure exactly. Use PDF rendering when source assets cannot reproduce the published composite figure.
- Never typeset a paper table with Markdown or HTML table markup. Always crop the complete table directly from the published PDF, even when source TeX or extracted cell text is available.
- Render every PDF-derived figure or table screenshot at scale 4 or higher (about 288 DPI). Keep `scripts/paper-crops/<slug>.json` so the crop is reproducible, and require every referenced local PNG to have a metadata entry.
- Retain an original source image only when it supplies at least four pixels per PDF point at the size used in the paper. Never upscale or sharpen an existing low-resolution PNG in place of a fresh PDF render or a genuinely higher-resolution source.
- Crop screenshots to the figure or table itself. Remove page headers, body text, captions that will be recreated in Markdown, and excessive white margins.
- Retain a small safety margin so lines and labels at the boundary are not clipped.
- Use lossless PNG for diagrams, plots, and tables. Avoid JPEG artifacts on text and thin lines.
- Use stable two-digit numbering: `figure-01.png`, `table-01.png`.
- Verify image dimensions against the scale-4 crop dimensions and inspect the actual pixels after cropping. File existence alone is not sufficient; pay special attention to low-confidence template matches and visually similar subfigures.
- Reuse identical table pixels across English, Simplified Chinese, and Japanese pages. Localize only alt text and captions, and carry semantic context there without reproducing the table cells as text.
- Format the caption as a separate bold label, for example `**Figure 8.**`, `**图 8.**`, or `**図 8.**`.
- Put a stable two-digit HTML anchor immediately before every numbered object: `<span id="figure-08"></span>` for figures and `<span id="table-03"></span>` for tables.
- Link every reference in prose, captions, algorithms, and appendices to that local anchor. Use `[Figure 8](#figure-08)`, `[图 8](#figure-08)`, or `[図 8](#figure-08)` according to the page language.
- Link each number separately in compound references. A subfigure reference such as `[Figure 16a](#figure-16)` targets the base numbered figure.

## English

- Copy the paper's rendered English word for word and sentence for sentence in source order.
- Preserve the source's spelling, capitalization, punctuation, paragraph boundaries, qualifications, repetition, and awkward wording. Never polish or silently correct it.
- Join TeX source line wraps and remove layout-only commands only when doing so preserves the exact rendered wording shown in the PDF.
- Keep established product, model, dataset, and method names unchanged.
- Preserve qualifications such as “may,” “typically,” and “under this setting” exactly; do not weaken or strengthen claims.

## Simplified Chinese

- Use the shared concise English title.
- Translate every source sentence faithfully and in the same order, keeping the closest practical one-to-one sentence correspondence.
- Preserve every claim, qualification, logical dependency, example, repetition, citation, and degree of certainty. Do not omit, combine, expand, explain, or add material absent from the source.
- Use established, natural technical Chinese that follows local documentation conventions and is easy to understand. Prefer idiomatic phrasing over literal calques, while keeping the original meaning and sentence boundary.
- Include an English term on first use only when the source includes it or doing so is necessary to identify an otherwise ambiguous official name; do not add explanatory content.
- Use half-width punctuation in prose: comma, period, colon, semicolon, and parentheses.
- Insert one space after a half-width comma, period, colon, or semicolon when more text follows.
- Keep one space between `图` or `表` and its number, and one space between a linked `图 N` or `表 N` reference and following Han text.
- Insert one space before a left parenthesis when it follows text: `大语言模型 (LLM)`.
- Do not place spaces just inside parentheses. Insert a space after the right parenthesis when ordinary text continues.
- Put spaces between Chinese text and adjacent Latin abbreviations, product names, numbers with Latin units, links, or inline math when they form separate tokens.
- Do not mechanically translate method names, model names, APIs, datasets, or code identifiers.
- Reorder or rephrase within a sentence as required by Chinese technical prose without changing its logical relationships or claim strength.

## Japanese

- Use the shared concise English title.
- Translate every source sentence faithfully and in the same order, keeping the closest practical one-to-one sentence correspondence.
- Preserve every claim, qualification, logical dependency, example, repetition, citation, and degree of certainty. Do not omit, combine, expand, explain, or add material absent from the source.
- Use grammatical, idiomatic Japanese technical prose that follows local documentation conventions and is easy to understand. Prefer natural phrasing over literal calques, and rephrase within a sentence when needed without changing its logical relationships or claim strength.
- Use Japanese punctuation and full-width Japanese parentheses in prose.
- Add spaces around Latin technical abbreviations, product names, and inline math where the surrounding pages do so.
- Preserve established katakana terminology and official method, model, dataset, and API names.
- Keep citation tokens adjacent to the claim they support while maintaining natural Japanese sentence flow.

## Final Consistency Pass

- Compare the three heading-number sequences.
- Compare the English page against the TeX source and PDF sentence by sentence for exact wording and complete coverage.
- Compare each Chinese and Japanese sentence against the same source for additions, omissions, and semantic drift.
- Confirm every run-in paragraph heading remains bold at the start of its paragraph and that no blank line splits it from the following prose.
- Compare the canonical kind-and-number sequence of every formal statement across all three languages. Confirm every label is bold and shares its paragraph with the statement.
- Compare proof-container counts across all three languages. Confirm the summaries are `Proof`, `证明`, and `証明`, every proof is closed by default, and no visible proof prefix or terminal QED mark remains.
- Compare equation tags, citation-token sets, image basenames, and algorithm step counts.
- Compare annotation labels and definitions, and confirm no Markdown footnotes remain.
- Confirm no standalone reference-list or empty generic appendix heading remains.
- Confirm every title is at most 50 characters and every matrix transpose uses `^\top`.
- Confirm that internal section references use the localized label but the same target number.
- Confirm that the concise title, authors, author URLs, arXiv version, DOI, and publication facts match across languages, and that the provenance link preserves the full source title.
- Confirm sidebar placement uses the first arXiv submission date and is identical in all locales.
