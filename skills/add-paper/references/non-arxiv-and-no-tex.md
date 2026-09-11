# Non-arXiv and No-TeX Paper Workflow

Read this reference when a paper has no arXiv record, no usable TeX source, or both. The published PDF remains the authority; HTML, OCR, plain-text extraction, and GROBID TEI are aids for locating and transcribing its content.

## Choose the Source Route

Classify the source before drafting:

- **arXiv with TeX:** follow the normal TeX-plus-PDF workflow.
- **arXiv without TeX:** use arXiv for version and date metadata, but use the current PDF as the textual, structural, mathematical, and bibliographic authority.
- **non-arXiv with publisher source files:** use the publisher source files plus the published PDF, applying the same authority split as the normal workflow.
- **non-arXiv without TeX:** use the canonical publisher or proceedings record for identity and publication metadata, and the published PDF for all rendered paper content.

When the user supplies a PDF and extracted XML, first confirm that their visible title, DOI, author order, and page count identify the same paper. Do not modify the supplied originals. Copy the verified PDF to `docs/.vuepress/public/paper/<slug>.pdf` and keep extraction intermediates outside the committed tree.

## Establish Provenance and Ordering

1. Prefer the canonical publisher or proceedings page and DOI record. Use an official project or author announcement only to corroborate facts that the canonical record does not expose clearly.
2. Verify the full title, ordered authors, venue, volume or issue when applicable, page range, DOI, publication date, and PDF edition. Do not infer missing metadata from a filename or extraction output.
3. Use the first arXiv submission date for sidebar ordering whenever an arXiv record exists, even if the user supplied a publisher PDF. Otherwise use the earliest verified formal publication date from the publisher or proceedings metadata. If only a month or year is available, preserve that precision instead of inventing a day.
4. Find identity-verified author links using the normal priority order. DOI metadata and Crossref ORCID records are useful evidence, but an ORCID must belong to the named author and the paper record before it is linked.
5. In the provenance block, link the DOI or canonical paper page and the local PDF. Omit unavailable arXiv and TeX links. State plainly that no arXiv or TeX source is available and that the published PDF is authoritative for exact wording, layout, and bibliography.

## Extract a PDF-Only Draft

1. Render every PDF page to images at a readable inspection resolution and obtain layout-preserving plain text when available. Use these outputs for navigation, not as unquestioned source text.
2. If GROBID TEI is supplied or can be generated, parse its header, body divisions, paragraphs, notes, figures, tables, formulas, code-like blocks, and bibliography into a provisional outline. Retain TEI identifiers only as temporary mapping aids; do not expose them in the final page.
3. Reconstruct the English page in visible PDF order. Preserve the paper's exact rendered English, including awkward wording, repetition, capitalization, punctuation, and apparent source typos. Never silently correct the paper because an extractor produced a more plausible sentence.
4. Restore structures that extraction flattened or interleaved: headings, paragraphs, lists, code blocks, algorithms, formulas, captions, table placement, acknowledgements, and appendices. Apply only the Markdown and KaTeX representation changes allowed by the main skill and style guide.
5. For multi-column papers, verify the reading order at every column, page, figure, table, footnote, and appendix transition. A paragraph beginning near a figure or table may continue in another column or on the next page.
6. Compare the completed English page against the PDF section by section and sentence by sentence. Searchable extracted text may accelerate this comparison, but each final sentence must be supported by the visible PDF.

## Treat Extraction as Untrusted Transcription

GROBID, OCR, and PDF text extraction commonly introduce errors that require visual review:

- merge a numbered heading into its first paragraph or split one paragraph into several;
- reorder text across columns, figures, tables, captions, footnotes, or page boundaries;
- drop code indentation, mathematical notation, superscripts, subscripts, ligatures, or Unicode symbols;
- join or split words at line-end hyphens and confuse punctuation or minus signs;
- flatten lists, algorithms, run-in headings, and appendix hierarchy;
- omit a figure, table, caption, footnote, acknowledgement, or artifact appendix;
- invent, duplicate, or misclassify bibliography records and shift subsequent reference identifiers.

Never resolve a disagreement in favor of extraction merely because its XML looks structured. Inspect the corresponding PDF pixels and preserve the paper's visible content.

## Audit GROBID TEI Against the PDF

Treat TEI as a searchable index and provisional outline, not a lossless representation. Before translating or polishing, align every retained TEI object to visible PDF pages and check these failure modes:

| GROBID failure mode | Required check |
| --- | --- |
| Multi-column text, marginal content, or text surrounding a float is emitted in the wrong order. | Render every page and trace the first and last sentence of each paragraph across column, page, figure, and table boundaries. Do not trust XML node order at a transition until the visible reading order agrees. |
| A `<head>` is merged into its first paragraph, repeated as body text, assigned to the wrong parent `<div>`, or omitted. | Compare the ordered heading outline and appendix hierarchy with the PDF. Search the draft for a heading's text outside its heading line and for section numbers that skip, repeat, or move backward. |
| `<figure>`, `<table>`, `<figDesc>`, labels, and nearby `<p>` nodes duplicate or interleave captions with body prose. | Build ordered lists of visible figures, tables, captions, and their first body sentence; compare them with the PDF. Keep one Markdown caption, and crop only the visual object, excluding the printed caption and surrounding prose. |
| Formula markup is flattened into Mathematical Alphanumeric Unicode, separated symbols, or prose-like tokens, such as `𝐻 𝑙`, `𝑛 win`, `𝐿 norm`, or `𝑟 len 𝑏, 𝑗`. | Search the completed pages for U+1D400–U+1D7FF characters and visually reconstruct every hit as KaTeX, including subscripts, superscripts, accents, grouping, and operator names. A plausible Unicode transcription is not sufficient. |
| Superscript minus signs or exponents become dashes, such as `10 — 20`, `10 — 3`, or `e — 1`; `min` and `max` may become ordinary letters. | Search for spaced dash patterns next to numerals or `e`, then compare the PDF glyphs. Restore exponents with braces, for example `10^{-20}` and `e^{-1}`, and use `\min` or `\max` for operators. |
| Equation boundaries, numbering, and surrounding sentences are split or reordered. | Inventory every displayed equation in visible order, match its printed number to the surrounding prose, and verify every formula-reference link against the rendered PDF. Do not infer grouping from TEI `<formula>` boundaries alone. |
| Lists, algorithms, run-in headings, code indentation, table notes, and footnotes are flattened into paragraphs or detached from their markers. | Compare item counts, nesting, marker order, and continuation text with the PDF. Reconstruct algorithms and code from visible indentation; verify each footnote marker and note as a pair. |
| Line-end hyphenation is preserved, removed incorrectly, or joins words that were separated by layout, producing forms such as `tokenbudget`. | Compare suspicious compounds with both the visible line break and searchable PDF text. Search for unexpected long tokens and for words that differ only at a page or column boundary. |
| TEI `target`, `xml:id`, bibliography order, or generated labels drift after a missing or spurious record. | Independently map visible citation labels and cross-references from the PDF. Spot-check the beginning, middle, and end, then compare complete ordered citation and target sets; never derive printed numbering from TEI IDs. |
| Front matter, bibliography, author lists, acknowledgements, or appendices leak into adjacent body divisions. | Record the visible start and end page of each apparatus section. Compare the first and last retained sentence of the main body and every appendix, and omit only the standalone bibliography as required by this skill. |
| TEI coordinates select a partial panel, omit a table edge or note, or include the printed caption. | Use coordinates only to locate the object. Inspect the final scale-4 crop at actual pixels against the PDF, checking all four edges, panels, labels, legends, rules, and notes, with a small even safety margin. |

Use lightweight inventories before manual comparison: count and list TEI `<head>`, `<figure>`, `<table>`, `<formula>`, `<note>`, and `<ref>` nodes in document order; then compare those sequences with the Markdown headings, anchors, images, captions, equations, annotations, and links. After reconstruction, run the paper checker and resolve its Unicode-math, suspicious-dash, structure, reference, and cross-locale warnings against the PDF instead of suppressing them.

## Recover Citations Without TeX

1. Map each visible inline citation number or label to the bibliography entry printed in the PDF. Do not assume a TEI `ref` target or bibliography `xml:id` equals the printed reference number.
2. Detect numbering drift by comparing several citations from the beginning, middle, and end of the paper. One spurious or missing extracted bibliography record can shift every later mapping.
3. Register only citations actually retained inline after the standalone reference list is omitted. Use the repository key rules and keep each key identical across all three languages.
4. Account for abbreviation maintenance in `docs/.vuepress/config/papers.ts`: preserve every existing citation key, even when multiple keys refer to the same work. Never deduplicate keys by URL. Run `npm run paper:config` to synchronize local paper links and chronological ordering, then confirm citation tokens render as abbreviations.

## Recover Figures, Tables, Code, and Math

1. Use the published PDF for every table crop. With no source assets, use PDF crops for figures as well unless an official asset can be proven pixel-equivalent and sufficiently dense.
2. Inspect every crop at actual pixels. Include the whole visual object and any internal legend, panel label, border, or table note, but exclude the printed caption and surrounding prose that will be reproduced in Markdown. Leave a small even safety margin so no meaningful stroke or glyph touches an edge.
3. Reconstruct code from the PDF rather than flattened TEI text. Preserve tokens and behavior while applying the repository's two-space indentation rule. Wrap inline identifiers containing Markdown-sensitive sequences, such as `at::vec::Vectorized`, in inline code so the theme does not interpret them as icons or markup.
4. Reconstruct formulas from the visible PDF and corroborate them with extractable text only when useful. Verify every symbol, delimiter, index, operator, equation grouping, and reference number visually; never guess an unreadable expression from surrounding prose.

## Final PDF-Only Audit

In addition to the normal checker, screenshot checker, build, and browser validation:

1. Compare the English heading, paragraph, list, code-block, figure, table, acknowledgement, and appendix sequence with the PDF.
2. Compare citation tokens with the printed inline citations and bibliography, then inspect generated HTML to ensure no token remains unrendered.
3. Confirm the three locales have identical anchor targets, link targets, citation sets, image order, code-block order and content, and substantive heading numbers.
4. Using the Codex built-in browser (`iab`) required by the main validation workflow, open the local PDF link and inspect all three paper routes at desktop and mobile widths. Confirm every image loads, dense tables remain within the article layout, and no paper asset introduces document-level horizontal overflow. If the built-in browser is unavailable, report the validation limitation as directed by the main skill.
5. Run the production build after the last source, citation, crop, or configuration change. Resolve every build warning and error rather than relying on an earlier successful build.
