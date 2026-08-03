---
name: add-paper
description: Add an arXiv paper to the ASa Book papers collection as complete English, Simplified Chinese, and Japanese pages. Use when the user supplies an arXiv identifier or URL and asks to add, typeset, faithfully copy, translate, synchronize, validate, or commit a paper; reproduce the English source word for word and sentence for sentence, translate Chinese and Japanese faithfully, and cover source research, author links, TeX-derived math and citations, shared figures and tables, chronological sidebars, production build, and a scoped Git commit.
---

# Add an arXiv Paper

Turn one arXiv identifier into a complete, source-faithful three-language ASa Book edition and finish with a verified commit. Work autonomously unless the paper's category or requested scope is genuinely ambiguous.

## Read Project Rules

1. Read the repository `AGENTS.md` and obey any more specific instructions discovered below the target paths.
2. Read [references/style-guide.md](references/style-guide.md) completely before editing a paper.
3. Inspect nearby pages in the selected collection and all three corresponding sidebar definitions in `docs/.vuepress/config.ts`.
4. Record `git status --short` before making changes. Preserve unrelated user changes and never include them in the paper commit.

## Normalize the Request

1. Accept an identifier such as `2409.16694`, `arXiv:2409.16694`, or an arXiv abstract/PDF URL.
2. Normalize it to the bare identifier and derive a short, descriptive, lowercase kebab-case slug from the title.
3. Treat the first arXiv submission date as the paper date for collection ordering. Do not use the latest revision date, journal publication date, page creation time, or the identifier alone as a substitute for the verified date.
4. Use the paper's unmodified title in all three page frontmatters and visible titles.

## Research Primary Sources

1. Open the arXiv abstract page and verify the title, ordered author list, first submission date, current version, abstract, subjects, comments, and DOI or publication venue when present.
2. Download the current PDF to `docs/.vuepress/public/paper/<slug>.pdf`.
3. Download and unpack the current TeX source into a temporary directory. Read the root TeX file, included sections, bibliography, figure declarations, table declarations, macros, and algorithm environments. Remove the temporary directory after use.
4. Use the TeX source as the structural and mathematical authority. Use the PDF to verify visible equation numbering, layout, captions, figure boundaries, and source ambiguities.
5. Use publisher or DOI metadata only to supplement arXiv metadata. State both dates when later publication differs from the first arXiv submission.
6. Find one identity-verified link for every author, in this priority order:
   - personal academic homepage maintained by the author;
   - the author's personal X account;
   - Google Scholar profile;
   - dblp author page.
7. Verify name, affiliation, and publication overlap before selecting an author URL. Never link a search-results page, an unrelated namesake, a lab homepage in place of an available personal page, or an unverified social account.

## Build the English Reading Edition

1. Create `docs/en/papers/<slug>.md` first. Treat it as a transcription of the current paper and as the structural source for the two localized pages.
2. Add YAML frontmatter with the exact paper title, current local `createTime`, and `/en/papers/<slug>/` permalink.
3. Start with a blockquote containing the complete linked author list in paper order, the first arXiv submission date, later venue details when verified, arXiv page, local original PDF, DOI when available, and TeX source.
4. Reproduce the complete paper in source order, including the abstract, every substantive section and subsection, captions, table text, algorithms, footnotes, acknowledgements, appendices, and reference entries. Do not abridge repeated or seemingly incidental material.
5. Copy the English prose word for word and sentence for sentence from the current paper. Preserve spelling, capitalization, punctuation, sentence and paragraph order, qualifications, and repetition. Do not paraphrase, summarize, polish, merge, split, reorder, expand, silently correct, or otherwise rewrite the source.
6. Make only representation changes required to render the source in Markdown and KaTeX: remove TeX-only layout commands, join source-code line wraps without changing the rendered sentence, expand textual macros to the words visible in the PDF, convert supported structure and math delimiters, and map citation commands to repository citation tokens without changing their placement or referents. Use the current TeX source as the textual authority and the current PDF to resolve rendered wording or source ambiguity.
7. Register every citation in `paperAbbreviations` and also reproduce the paper's complete reference section in the English page. State that the original PDF remains authoritative for the exact print layout.

## Prepare Shared Media

1. Store all page-specific raster assets in `docs/papers/<slug>/` with deterministic names such as `figure-01.png` and `table-01.png`.
2. Reuse the same assets from all languages. Reference them as `./<slug>/...` from Chinese and `../../papers/<slug>/...` from English and Japanese.
3. Prefer original source assets. When extraction from PDF is necessary, render at sufficient resolution and crop to the real figure or table bounds.
4. Remove excessive surrounding whitespace. Preserve labels, legends, axes, footnotes, and border strokes; add only a small consistent safety margin.
5. Compare every cropped asset with the PDF. Pay particular attention to multi-panel figures and dense tables, where automatic trimming can remove meaningful edge content.
6. Write localized Markdown alt text and an explicit localized bold caption below each asset.

## Register Citations and Navigation

1. Collect every citation token used by the page, including tokens that appear only in captions, tables, algorithm prose, or acronym explanations.
2. Add missing definitions to the single `paperAbbreviations` object in `docs/.vuepress/config.ts`. Do not create per-paper abbreviation files.
3. Reuse an existing key only when it denotes the same work. Resolve collisions with a stable lowercase suffix and update all three pages consistently.
4. Include enough bibliographic detail to identify the work and add a primary link when available. This registration is what gives citation tokens such as `[Hu21]` their underlined explanation.
5. Add the slug to the matching category in the Chinese, English, and Japanese sidebars.
6. Sort all three sidebar lists identically and chronologically by first arXiv submission date. For equal dates, keep the existing stable order unless title sorting is already the local convention.

## Localize to Chinese and Japanese

1. Create `docs/papers/<slug>.md` and `docs/ja/papers/<slug>.md` only after the English structure is stable.
2. Preserve title, author order and URLs, links, citations, math, equation tags, asset order, heading levels, and technical meaning across all languages.
3. Translate every source sentence into Chinese and Japanese in the same order and with the closest practical one-to-one sentence correspondence. Preserve every claim, qualification, logical dependency, example, repetition, and citation. Do not summarize, paraphrase, omit, combine, expand, explain, or add material absent from the source.
4. Allow only the grammatical reordering needed for correct Chinese or Japanese. Resolve ambiguity from the TeX source and PDF; never guess by introducing a new interpretation.
5. Follow every language rule in [references/style-guide.md](references/style-guide.md), especially Chinese half-width punctuation and spacing, faithful Japanese technical prose, and the unchanged paper title.

## Validate the Result

1. Run the bundled checker from the repository root:

   ```bash
   node skills/add-paper/scripts/check-paper.mjs <slug>
   ```

2. Resolve every error. Review warnings against the TeX source and PDF rather than suppressing them mechanically.
3. Compare the English page with the TeX source and PDF section by section and sentence by sentence. Confirm complete coverage and exact wording before comparing every Chinese and Japanese sentence with the same source for omissions, additions, weakened or strengthened claims, and changed logical relationships.
4. Search all three pages for malformed math, missing citation definitions, stale source paths, pseudocode fences used for math-heavy algorithms, and inconsistent captions.
5. Run `git diff --check`.
6. Run `npm run docs:build`. If VuePress cache behavior is suspicious, run `npm run docs:build -- --clean-cache --clean-temp`.
7. Inspect the three target routes at desktop and mobile widths when figures, tables, formulas, or navigation changed. Confirm that equations render, tags remain visible, images are legible and tightly framed, captions do not overflow, and citation abbreviations show their explanations.
8. Stop every development or preview server started for validation.

## Commit the Paper

1. Recheck `git status --short` and distinguish paper changes from pre-existing work.
2. Stage explicit in-scope paths only: the three Markdown pages, shared assets, PDF, `docs/.vuepress/config.ts`, and `.gitattributes` only if binary PDF handling must be introduced.
3. Run `git diff --cached --check` and inspect `git diff --cached --stat` plus the staged config and Markdown diff.
4. Commit with a concise imperative message, normally `docs: add <short-paper-name> paper`.
5. Verify `git status --short` after the commit. A pre-existing unrelated dirty state may remain; no staged or unstaged paper changes may remain.
6. Report the commit hash and message, affected routes, build result, and any residual validation limitation.
