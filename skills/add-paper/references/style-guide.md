# ASa Book Paper Style Guide

Apply these rules to English, Simplified Chinese, and Japanese paper pages. Treat the paper's TeX source and PDF as authoritative when an example does not cover a case.

## Page Structure

- Keep the paper title untranslated in all languages.
- Preserve numbered section and subsection structure unless a heading exists only for print layout rather than content.
- Preserve citations and equation, figure, table, and algorithm numbering.
- Reproduce the complete paper in source order. Do not abridge, summarize, expand, or omit material for a shorter Web reading edition.
- Use one shared set of image files. Localize alt text and captions, not the pixels in an original scholarly figure.
- Keep author links identical across languages and in the paper's original order.

## Citations and Abbreviations

- Write citations as the repository's compact keys, for example `[Hu21]` or `[Ope24a, Tou23]`.
- Define every cited key in `paperAbbreviations` in `docs/.vuepress/config.ts`; this enables the visible dotted underline and hover explanation.
- Include citations attached to expanded terms, for example `Low-Rank Adaptation (LoRA) [Hu21]`.
- Preserve a paper's reference identity across languages. Never translate titles in bibliography definitions.
- Prefer the canonical paper page, DOI, conference page, or arXiv abstract as the bibliography link.

## Math

- Use `$...$` for inline math and `$$...$$` for display math. Keep `\tag{N}` inside the same display-math delimiters as its equation.
- Verify every numbered display equation in the built page. A literal `\tag{N}` in prose or outside math is invalid.
- Put every multi-letter word or abbreviation inside `\mathrm{}` in every formula, including subscripts and superscripts. Examples: `X_{\mathrm{FP}32}`, `X^{\mathrm{unscaled}}`, `\mathrm{round}(x)`, and `\mathrm{LZD}(b)`.
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
- Use a fenced code or pseudocode block only when the source is genuinely code-oriented, math is incidental, and the existing collection style supports it.

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

- Prefer source image files over screenshots. Use PDF rendering when source assets cannot reproduce the published composite figure.
- Crop screenshots to the figure or table itself. Remove page headers, body text, captions that will be recreated in Markdown, and excessive white margins.
- Retain a small safety margin so lines and labels at the boundary are not clipped.
- Use lossless PNG for diagrams, plots, and tables. Avoid JPEG artifacts on text and thin lines.
- Use stable two-digit numbering: `figure-01.png`, `table-01.png`.
- Verify image dimensions and inspect the actual pixels after cropping. File existence alone is not sufficient.
- Format the caption as a separate bold label, for example `**Figure 8.**`, `**图 8.**`, or `**図 8.**`.

## English

- Copy the paper's rendered English word for word and sentence for sentence in source order.
- Preserve the source's spelling, capitalization, punctuation, paragraph boundaries, qualifications, repetition, and awkward wording. Never polish or silently correct it.
- Join TeX source line wraps and remove layout-only commands only when doing so preserves the exact rendered wording shown in the PDF.
- Keep established product, model, dataset, and method names unchanged.
- Preserve qualifications such as “may,” “typically,” and “under this setting” exactly; do not weaken or strengthen claims.

## Simplified Chinese

- Keep the paper title untranslated.
- Translate every source sentence faithfully and in the same order, keeping the closest practical one-to-one sentence correspondence.
- Preserve every claim, qualification, logical dependency, example, repetition, and citation. Do not summarize, paraphrase, omit, combine, expand, explain, or add material absent from the source.
- Use established technical Chinese. Include an English term on first use only when the source includes it or doing so is necessary to identify an otherwise ambiguous official name; do not add explanatory content.
- Use half-width punctuation in prose: comma, period, colon, semicolon, and parentheses.
- Insert one space after a half-width comma, period, colon, or semicolon when more text follows.
- Insert one space before a left parenthesis when it follows text: `大语言模型 (LLM)`.
- Do not place spaces just inside parentheses. Insert a space after the right parenthesis when ordinary text continues.
- Put spaces between Chinese text and adjacent Latin abbreviations, product names, numbers with Latin units, links, or inline math when they form separate tokens.
- Do not mechanically translate method names, model names, APIs, datasets, or code identifiers.
- Reorder words only as required by Chinese grammar without changing sentence boundaries or meaning.

## Japanese

- Keep the paper title untranslated.
- Translate every source sentence faithfully and in the same order, keeping the closest practical one-to-one sentence correspondence.
- Preserve every claim, qualification, logical dependency, example, repetition, and citation. Do not summarize, paraphrase, omit, combine, expand, explain, or add material absent from the source.
- Use grammatical Japanese technical prose, but reorder words only as required by Japanese grammar without changing sentence boundaries or meaning.
- Use Japanese punctuation and full-width Japanese parentheses in prose.
- Add spaces around Latin technical abbreviations, product names, and inline math where the surrounding pages do so.
- Preserve established katakana terminology and official method, model, dataset, and API names.
- Keep citation tokens adjacent to the claim they support while maintaining natural Japanese sentence flow.

## Final Consistency Pass

- Compare the three heading-number sequences.
- Compare the English page against the TeX source and PDF sentence by sentence for exact wording and complete coverage.
- Compare each Chinese and Japanese sentence against the same source for additions, omissions, and semantic drift.
- Compare equation tags, citation-token sets, image basenames, and algorithm step counts.
- Confirm that internal section references use the localized label but the same target number.
- Confirm that the title, authors, author URLs, arXiv version, DOI, and publication facts match across languages.
- Confirm sidebar placement uses the first arXiv submission date and is identical in all locales.
