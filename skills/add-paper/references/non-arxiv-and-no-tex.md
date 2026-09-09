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

## Recover Citations Without TeX

1. Map each visible inline citation number or label to the bibliography entry printed in the PDF. Do not assume a TEI `ref` target or bibliography `xml:id` equals the printed reference number.
2. Detect numbering drift by comparing several citations from the beginning, middle, and end of the paper. One spurious or missing extracted bibliography record can shift every later mapping.
3. Register only citations actually retained inline after the standalone reference list is omitted. Use the repository key rules and keep each key identical across all three languages.
4. Account for abbreviation normalization in `config.ts`: two definitions with the same canonical URL may collapse to one entry. After building, confirm every citation token became a rendered abbreviation rather than remaining raw text. If a legitimate distinct work collides, use a distinct canonical version or publication URL and then choose the next valid repository key; do not weaken the bibliographic identity merely to silence the collision.

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
4. Open the local PDF link in the browser, and inspect all three paper routes at desktop and mobile widths. Confirm every image loads, dense tables remain within the article layout, and no paper asset introduces document-level horizontal overflow.
5. Run the production build after the last source, citation, crop, or configuration change. Resolve every build warning and error rather than relying on an earlier successful build.
