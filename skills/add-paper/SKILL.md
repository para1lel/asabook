---
name: add-paper
description: Add or revise an arXiv paper in the ASa Book papers collection as English, Simplified Chinese, and Japanese pages. Use when Codex needs to add, typeset, translate, localize, synchronize, validate, revise, or commit a paper from an arXiv identifier or URL; preserve the English source, write natural localized technical Chinese and Japanese without semantic drift, use concise titles and Plume content annotations, format theorem-like statements as bold run-ins and standalone proofs as collapsed details containers, omit standalone reference lists, preserve shared figures and inline citations, crop every table directly from the published PDF instead of typesetting it, link every figure and table reference to a stable anchor, validate the production build, and commit all current repository changes together.
---

# Add an arXiv Paper

Turn one arXiv identifier into a source-faithful three-language ASa Book reading edition and finish with a verified repository-wide commit. Work autonomously unless the paper's category or requested scope is genuinely ambiguous.

## Read Project Rules

1. Read the repository `AGENTS.md` and obey any more specific instructions discovered below the target paths.
2. Read [references/style-guide.md](references/style-guide.md) completely before editing a paper.
3. Inspect nearby pages in the selected collection and all three corresponding sidebar definitions in `docs/.vuepress/config.ts`.
4. Record `git status --short` before making changes. Preserve all existing user changes without reverting or rewriting them; include the complete repository state in the final commit.

## Normalize the Request

1. Accept an identifier such as `2409.16694`, `arXiv:2409.16694`, or an arXiv abstract/PDF URL.
2. Normalize it to the bare identifier and derive a short, descriptive, lowercase kebab-case slug from the title.
3. Treat the first arXiv submission date as the paper date for collection ordering. Do not use the latest revision date, journal publication date, page creation time, or the identifier alone as a substitute for the verified date.
4. Use one concise English title of at most 50 characters in all three page frontmatters and visible titles. If the source title is longer, shorten it without losing the paper's identity, and retain the full source title in the provenance link.

## Research Primary Sources

1. Open the arXiv abstract page and verify the title, ordered author list, first submission date, current version, abstract, subjects, comments, and DOI or publication venue when present.
2. Download the current PDF to `docs/.vuepress/public/paper/<slug>.pdf`.
3. Download and unpack the current TeX source into a temporary directory. Read the root TeX file, included sections, bibliography, figure declarations, table declarations, macros, and algorithm environments. Remove the temporary directory after use.
4. Use the TeX source as the structural and mathematical authority. Use the PDF to verify source equation numbering for anchors and references, layout, captions, figure boundaries, and source ambiguities. Do not render equation numbers beside formulas.
5. Use publisher or DOI metadata only to supplement arXiv metadata. State both dates when later publication differs from the first arXiv submission.
6. Find one identity-verified link for every author, in this priority order:
   - personal academic homepage maintained by the author;
   - the author's personal X account;
   - Google Scholar profile;
   - dblp author page.
7. Verify name, affiliation, and publication overlap before selecting an author URL. Never link a search-results page, an unrelated namesake, a lab homepage in place of an available personal page, or an unverified social account.

## Build the English Reading Edition

1. Create `docs/en/papers/<slug>.md` first. Treat it as a transcription of the current paper and as the structural source for the two localized pages.
2. Add YAML frontmatter with the chosen concise title, current local `createTime`, and `/en/papers/<slug>/` permalink.
3. Start with a blockquote containing the complete linked author list in paper order, the first arXiv submission date, later venue details when verified, arXiv page, local original PDF, DOI when available, and TeX source.
4. Reproduce the complete substantive paper in source order, including the abstract, every substantive section and subsection, captions, table text, algorithms, annotation content, acknowledgements, and appendices. Omit the standalone reference list and layout-only headings such as an empty `Appendix` heading immediately followed by `Appendix A`; retain the actual appendix sections.
5. Copy the English prose word for word and sentence for sentence from the current paper. Preserve spelling, capitalization, punctuation, sentence and paragraph order, qualifications, and repetition. Do not paraphrase, summarize, polish, merge, split, reorder, expand, silently correct, or otherwise rewrite the source.
6. Make only representation changes required to render the source in Markdown and KaTeX: remove TeX-only layout commands, join source-code line wraps without changing the rendered sentence, expand textual macros to the words visible in the PDF, convert supported structure and math delimiters, map citation commands to repository citation tokens without changing their placement or referents, and normalize visible hyphens and dashes according to the style guide. Use the current TeX source as the textual authority and the current PDF to resolve rendered wording or source ambiguity.
7. Register every inline citation in `paperAbbreviations`, but do not reproduce a standalone reference section in the page. State that the original PDF remains authoritative for the exact print layout and bibliography.

## Convert Footnotes to Content Annotations

1. Do not use Markdown footnote syntax such as `[^1]` or `[^1]:`.
2. Convert every source footnote to a Plume content annotation. Insert the marker as `[+label]` with a space before it, and define it in a separate area as `[+label]: content`.
3. Reuse stable labels consistently across all three languages. Numeric source footnote labels may remain numeric.
4. Preserve the complete footnote content, Markdown links, and its logical attachment point. Keep the annotation marker outside surrounding bold or italic markup.
5. Confirm `markdown.annotation` is enabled in the Plume configuration and verify the built page contains interactive annotation components.

## Number Section Headings

1. Give every substantive section and subsection heading an Arabic decimal number in all three languages, for example `## 1 Introduction`, `### 2.1 Architecture`, and `#### 3.1.1 Dataflow`. Do not use Roman numerals, letters, or mixed forms such as `II`, `II-A`, or `III-A1` in headings.
2. Normalize the source hierarchy without changing its order or wording: map `II-A` to `2.1`, `III-A1` to `3.1.1`, and equivalent source schemes to the same decimal hierarchy. Keep the complete number sequence identical across English, Simplified Chinese, and Japanese.
3. Use matching decimal section anchors such as `section-2-1`, and update every internal section-reference label and target to the normalized number. Keep localized reference words such as `Section`, `第 ... 节`, and `第 ... 節`.
4. Treat the abstract and acknowledgements as document apparatus rather than numbered sections; leave only those headings unnumbered. Number appendices as substantive sections by continuing the top-level decimal sequence after the main body.
5. Do not apply section numbering to run-in paragraph labels, formal statements, figures, tables, algorithms, annotation definitions, or proof-container summaries.

## Format Code Blocks

1. Use two spaces for each indentation level in every fenced code block on all English, Simplified Chinese, and Japanese paper pages. This applies to Python, pseudocode, shell commands, configuration, and literal prompt examples; never use tabs or a four-space base indentation.
2. Keep the first indented level at two spaces and increase nested levels in two-space steps. Preserve the source program's behavior and visible content while normalizing indentation.
3. Keep corresponding code blocks identically indented across all three languages.

## Preserve Run-In Paragraph Headings

1. Render every source run-in paragraph heading, such as a TeX `\paragraph{...}` label, as bold text at the start of the same Markdown paragraph: `**Heading.** Paragraph text...`.
2. Keep exactly one space between the closing bold marker and the paragraph text. Never insert a blank line or line break between them, because that changes the run-in heading into a standalone block.
3. Preserve the source heading's wording and punctuation in English and translate it faithfully in Chinese and Japanese. Do not apply this rule to section headings, figure or table captions, algorithm titles, or labels that introduce a list, table, code block, or display equation.

## Format Formal Statements and Proofs

1. Render every numbered definition, lemma, proposition, corollary, theorem, and claim as a bold run-in label followed immediately by its statement on the same Markdown line, for example `**Theorem 3.8.** The expected improvement factor is ...`. Never render the label as a heading or isolated paragraph.
2. Keep a multi-line statement in the same paragraph. Put its first clause after the bold label and use `<br>` for deliberate continuation lines; do not add trailing spaces or move the label above a display block.
3. Wrap every source proof environment or otherwise standalone proof passage in a default-closed Plume details container. Use `Proof`, `证明`, and `証明` as the English, Simplified Chinese, and Japanese summary labels.
4. Put the complete proof, including all paragraphs and display equations, inside one container. Keep anchors outside the container when other content links to the statement.
5. Remove the source's visible `Proof.` label and terminal QED mark, including `∎`, `□`, and `\square`; the container summary supplies the label and boundary. Do not wrap a sentence that merely refers to a proof or sketches an argument as part of ordinary prose.
6. Follow the exact syntax and examples in [references/style-guide.md](references/style-guide.md), and keep the formal-statement sequence and proof-container count aligned across all three languages.

## Format Algorithm Descriptions

1. Render math-heavy algorithms as compact unordered Markdown lists beginning with `-`. Put the algorithm title in bold above the list.
2. Use two spaces for each nesting level to express loops, branches, and substeps. Do not number algorithm steps.
3. Do not write `end for`, `end if`, `end while`, or translated equivalents; indentation expresses scope. Preserve execution order, inputs, outputs, conditions, and return values.
4. Bold localized control words such as **For**, **If**, **Return**, and their Chinese or Japanese equivalents. Use inline code only for literal code symbols and operators.
5. Existing fenced `pseudocode` algorithm blocks may remain unchanged. For a new algorithm, use a fenced `pseudocode` block only when it is genuinely code-oriented, math is incidental, and the surrounding collection style supports it.

## Prepare Shared Media

1. Store all page-specific raster assets in `docs/papers/<slug>/` with deterministic names such as `figure-01.png` and `table-01.png`.
2. Reuse the same assets from all languages. Reference them as `./<slug>/...` from Chinese and `../../papers/<slug>/...` from English and Japanese.
3. Never recreate a paper table with Markdown or HTML table markup. Crop the complete published table directly from the paper PDF and render it at scale 4 or higher, even when source TeX or extracted cell text is available.
4. Reuse identical table pixels across all three languages. Localize only the alt text and caption; keep semantic context there without transcribing or reconstructing the table cells.
5. Prefer original source assets for figures only when they are at least as sharp as a PDF crop rendered at scale 4. When PDF rendering is necessary, use `npm run paper:refresh-screenshots -- --write --scale 4 <slug>`; scale 4 (about 288 DPI) is the minimum, not a suggested default.
6. Crop every screenshot to the complete figure or table body. Preserve every panel, label, legend, axis, table footnote, border stroke, and subfigure description that belongs to the visual object; never include the printed figure or table caption or surrounding explanatory prose when that text is reproduced in Markdown. Remove excessive whitespace, leave a small consistent safety margin on all four sides, and confirm that no meaningful stroke or glyph touches the crop edge.
7. Never enlarge, resample, sharpen, or re-encode an old low-resolution PNG as a substitute for rendering the PDF or using a higher-resolution source asset. A retained source asset must have at least four pixels per PDF point at its displayed PDF size in both dimensions.
8. Keep the reproducible crop metadata in `scripts/paper-crops/<slug>.json`. Every local PNG referenced by the page must have one metadata entry; use `preserveSource: true` only when the source-density requirement above is met.
9. Compare every cropped asset with the PDF at actual pixels. Pay particular attention to multi-panel figures and dense tables, where automatic matching or trimming can select the wrong panel or remove meaningful edge content.
10. Write localized Markdown alt text and an explicit localized bold caption below each asset.
11. Add a stable two-digit HTML anchor immediately before every numbered figure or table object, such as `<span id="figure-08"></span>` or `<span id="table-03"></span>`.
12. Link every figure and table reference in prose, captions, algorithms, and appendices to its local anchor. Use localized labels, for example `[Figure 8](#figure-08)`, `[图 8](#figure-08)`, and `[図 8](#figure-08)`.
13. Link every number in a compound reference separately. Point subfigure references such as `Figure 16a` to the base figure anchor `#figure-16`. In Chinese, keep one space between `图` or `表` and its number, and one space between the linked reference and following Han text.
14. When revising an existing paper, crop every figure and table from the published PDF itself. Do not substitute the original source image when the request requires PDF-faithful presentation; keep the crop metadata reproducible and use the same PDF crop in all languages.

## Register Citations and Navigation

1. Collect every inline citation token used by the page, including tokens that appear in captions, tables, algorithm prose, or acronym explanations. Do not collect citations solely to generate a standalone reference list.
2. Add missing definitions to the single `paperAbbreviations` object in `docs/.vuepress/config.ts`. Do not create per-paper abbreviation files.
3. Form each citation key from the first three Latin letters of the first author's surname plus the final two digits of the publication year, with the first letter uppercase and the rest lowercase, for example `Vas17`. If the surname has fewer than three letters, use the complete surname, for example `Hu21`. For a source without a personal author, use the first three Latin letters of its named project; use the organization name only when there is no distinct project name.
4. Reuse an existing key only when it denotes the same work. When different works share the same author-or-project prefix and publication year, keep the first registered key unsuffixed and append contiguous lowercase suffixes to the rest: no suffix, `a`, `b`, `c`, and so on. Reuse existing assignments and take the next available suffix; never use a paper slug, bibliography index, page number, or mnemonic suffix as the key or suffix.
5. Use the formal publication year shown in the bibliographic entry. When there is no formal publication, use the source's release year. Do not infer the year from a bibliography index, page range, access date, arXiv identifier, or the paper currently being added.
6. Update citation tokens consistently in the English, Simplified Chinese, and Japanese pages whenever a key changes.
7. Include enough bibliographic detail to identify the work and add a primary link when available. This registration is what gives citation tokens such as `[Hu21]` their underlined explanation.
8. Add the slug to the matching category in the Chinese, English, and Japanese sidebars.
9. Sort all three sidebar lists identically and chronologically by first arXiv submission date. For equal dates, keep the existing stable order unless title sorting is already the local convention.

## Localize to Chinese and Japanese

1. Create `docs/papers/<slug>.md` and `docs/ja/papers/<slug>.md` only after the English structure is stable.
2. Preserve title, author order and URLs, links, citations, math, equation anchors and reference labels, asset order, heading levels, and technical meaning across all languages. Do not add visible equation tags.
3. Translate every source sentence into Chinese and Japanese in the same order and with the closest practical one-to-one sentence correspondence. Preserve every claim, qualification, logical dependency, example, repetition, citation, and degree of certainty. Do not omit, combine, expand, explain, or add material absent from the source.
4. Do not install, invoke, or rely on local translation models or remote translation APIs or services. Translate the source directly in-agent or delegate translation to sub-agents.
5. Prefer established local technical-document wording over literal calques. Rephrase within a sentence when needed for clarity and natural Chinese or Japanese, while preserving its content, logical relationships, and claim strength.
6. Preserve official names for models, methods, APIs, datasets, and code identifiers instead of translating them mechanically. Resolve ambiguity from the TeX source and PDF; never guess by introducing a new interpretation.
7. Follow every language rule in [references/style-guide.md](references/style-guide.md), especially Chinese half-width punctuation and spacing, faithful Japanese technical prose, and the shared concise title.
8. Apply the `humanizer-zh` guidance when editing Simplified Chinese prose: remove formulaic AI phrasing, vague attribution, promotional wording, excessive connective phrases, repeated synonym substitutions, and unnecessary rhetorical padding while preserving the source meaning and sentence order.
9. In the Chinese page, use ASCII half-width punctuation throughout visible prose, captions, headings, metadata text, and list items. Keep Markdown links, code, identifiers, and mathematical delimiters intact.
10. Render all source lists as standard Markdown lists. Keep list items contiguous with no blank line between adjacent items; use nested indentation for hierarchy rather than manual bullets or paragraph-separated pseudo-lists.
11. Match the published PDF's visual structure when revising layout: preserve figure/table placement order, captions, equation grouping, paragraph boundaries, and list grouping. Do not rewrite a PDF list as prose or a prose paragraph as a list unless the source does so.

## Validate the Result

1. Run the bundled checker from the repository root:

   ```bash
   node skills/add-paper/scripts/check-paper.mjs <slug>
   ```

2. Validate screenshot resolution and crop coverage:

   ```bash
   npm run paper:check-screenshots -- <slug>
   ```

3. Resolve every error. Review warnings against the TeX source and PDF rather than suppressing them mechanically. A multi-letter sequence is not automatically a word: preserve products of adjacent single-letter variables such as `RD`, `thm`, `sl_i`, and `8Bh`, and preserve multi-index symbols such as `a_{ij}`. Use `\mathrm{}` only when the sequence is clearly prose, a textual label, a unit, or an abbreviation; use `\mathit{}` for genuine multi-letter variable names when needed to keep them distinct from products.
4. Compare the English page with the TeX source and PDF section by section and sentence by sentence. Confirm complete coverage and exact wording before comparing every Chinese and Japanese sentence with the same source for omissions, additions, weakened or strengthened claims, and changed logical relationships.
5. Search all three pages for non-decimal or missing substantive section numbers, mismatched section anchors or references, malformed math, visible equation tags, missing or misdirected equation-reference anchors, missing citation definitions, stale source paths, pseudocode fences used for math-heavy algorithms, code-block indentation that does not use two-space levels, run-in paragraph headings split into standalone bold lines, theorem-like labels rendered as headings or isolated paragraphs, standalone proofs left outside localized details containers, terminal QED marks, inconsistent captions, unlinked or misdirected figure and table references, missing figure and table anchors, Markdown footnotes, standalone reference headings, empty generic appendix headings, consecutive ASCII hyphens in rendered article content, and legacy matrix transpose notation `^{T}`.
6. Confirm every title is at most 50 characters, every matrix transpose uses `^\top`, and every annotation marker has a matching definition in all three languages.
7. Run `git diff --check`.
8. Run `npm run docs:build`. If VuePress cache behavior is suspicious, run `npm run docs:build -- --clean-cache --clean-temp`.
9. Inspect the three target routes at desktop and mobile widths when figures, tables, formulas, annotations, formal statements, proof containers, or navigation changed. Confirm that equations render without visible number tags, equation-reference links jump to the intended formulas, annotations open correctly, theorem-like labels are bold and share a paragraph with their statements, proof details are closed by default and open correctly without horizontal overflow, images are legible and tightly framed at 100% zoom on a high-density display, captions do not overflow, figure and table links jump to the intended objects, and citation abbreviations show their explanations.
10. Stop every development or preview server started for validation.

## Commit the Repository State

1. Recheck `git status --short` and review every existing change. Preserve changes that predate the paper task; do not reset, revert, or rewrite them.
2. Stage the complete repository state with `git add -A`, including all tracked and untracked changes, even when some changes are unrelated to the paper task.
3. Run `git diff --cached --check` and inspect `git diff --cached --stat` plus the complete staged diff for unexpected generated files, caches, secrets, or destructive changes.
4. Remove generated output and caches from the staging area without deleting the user's files. Stop and ask before committing if the staged state contains secrets or clearly unsafe artifacts.
5. Commit all reviewed changes together with a concise imperative message that represents the repository-wide change.
6. Verify `git status --short` is empty after the commit.
7. Report the commit hash and message, all included paths or change groups, affected routes, build result, and any residual validation limitation.
