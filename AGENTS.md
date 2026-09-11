# Repository Guidelines

## Project Structure & Module Organization

This repository is a VuePress 2 documentation/blog site using the Plume theme and Vite bundler. Author content under `docs/`:

- `docs/README.md` is the site home page, written in Chinese.
- `docs/vndb/` contains visual novel / Galgame notes. The current entry point is `intro.md`.
- `docs/csdiy/` contains course, math, and problem-solving notes. Current pages include `aops.md` and `tst26-p18.md`.
- `docs/.vuepress/config.ts` assembles site metadata and theme settings. `docs/.vuepress/config/` contains collection modules (`papers.ts`, `gpupro.ts`, `yokubi.ts`, `csdiy.ts`, and `vndb.ts`) with their multilingual sidebars.
- `docs/.vuepress/client.ts` registers client-side styles and the LXGW WenKai webfont.
- `docs/.vuepress/styles/index.css` contains global theme overrides.

Generated output and caches live under `docs/.vuepress/dist`, `.cache`, and `.temp`; never edit or commit them. Add collection-specific images beside the Markdown page that references them when relative links are clearer, such as `docs/csdiy/miku.gif`; use `docs/.vuepress/public/` only for shared public assets.

## Build, Test, and Development Commands

- `npm install` installs the versions recorded in `package-lock.json`.
- `npm run docs:dev` starts the local documentation server with hot reload.
- `npm run docs:build` produces the static site and validates Markdown, routes, imports, and theme configuration.
- `npm run paper:config` synchronizes reading-edition citation links and sorts the single abbreviation object; add `-- --check` to validate without writing.
- `npm run paper:check-config` checks date parsing, lossless configuration updates, and language-specific citation links (use Node.js 24, matching the project tooling environment).

Run `npm run docs:build` before submitting changes.

After changing dependency versions, abbreviation definitions, or Markdown plugins, clear the Markdown cache with `npm run docs:build -- --clean-cache` or restart development with `--clean-cache`. Run development and production validation sequentially: the Windows file watcher can fail when a concurrent build rewrites `dist`.

## Dependency and Software Installation

When a task requires a missing package, library, CLI, runtime, browser, or other software, you are authorized to install it without requesting additional permission. Prefer the narrowest practical installation scope and project-local dependencies when available; use a system-wide installation when the tool genuinely requires it. Use official or otherwise verified sources, avoid unrelated upgrades, and preserve any required manifest and lockfile changes.

The current VuePress/Plume release requires Markdown-it 14 and Mermaid 11. `package.json` pins the newest compatible `@mdit/*` releases through overrides because newer releases require Markdown-it 15. TypeScript 6 supplies the compiler API used by paper configuration tooling. Recheck these constraints, `npm ls --all`, and the production build before lifting the pins. MathJax is installed to satisfy the math plugin's module imports even though this site renders formulas with KaTeX. `config/paper-links.ts` also escapes Plume rc.210's unescaped quotation marks in abbreviation aria-labels; keep the regression check when removing this compatibility fix.

## Translation Workflow

Do not use local machine-translation models for paper editions or other repository translations. Translate the source directly or delegate translation to sub-agents. Preserve the source sentence order and all protected Markdown structures, citations, links, math, code blocks, anchors, and identifiers.

## Coding Style & Naming Conventions

Use two-space indentation in TypeScript, JSON, CSS, and YAML frontmatter. Keep TypeScript imports at the top and prefer single quotes without semicolons, matching `config.ts`. Use lowercase kebab-case Markdown filenames, such as `tst26-p18.md`. Keep collection names displayed as `vndb` and `csdiy`, and preserve the site name `ASa Book`.

Keep all Markdown abbreviation definitions in `docs/.vuepress/config/papers.ts` under `paperAbbreviations`; do not create per-page or per-paper abbreviation files.

Use one object literal assignment, never `Object.assign` or later mutations. Run `npm run paper:config` after adding definitions or paper pages. It replaces or adds `Link` for existing reading editions; the Markdown plugin localizes these links to the current page language. Preserve all citation keys. Order entries oldest first using verified first-submission/publication dates (`// published: YYYY-MM-DD`), otherwise the bibliographic year (legacy key year only when the value has no year); break ties by key. Never use page creation timestamps or invent date precision.

Write content primarily in Chinese, with short Markdown sections, descriptive headings, and relative links for pages in the same collection. For paper editions governed by the add-paper skill, reproduce the English source word for word and sentence for sentence, and translate the Chinese and Japanese pages faithfully from that source. For other English translation pages, prioritize fluent, idiomatic English that reads naturally to native speakers while preserving the original meaning, links, math, and structure. Pages should include YAML frontmatter with `title`, `createTime`, and a stable `permalink` matching the configured collection path, for example `/csdiy/tst26-p18/`. Math content may use KaTeX syntax. When adding a page, also add its filename without `.md` to the appropriate collection module in `docs/.vuepress/config/`, and update navbar entry points if the first page in a collection changes.

In paper editions, write internal section references as linked localized labels such as `[Section 3](#section-3)`, `[第 3 节](#section-3)`, and `[第 3 節](#section-3)`; never prefix them with `§` or `\S`. In Chinese paper pages, leave one space between a linked formula reference such as `[公式 4](#equation-04)` and following Han text, but no space before punctuation. In every fenced code block, put the least-indented nonblank content line at column zero and indent each nested level by exactly two spaces; never add a uniform base indent, use tabs, or use four spaces as one indentation level.

No formatter or linter is configured; follow the surrounding style and avoid unrelated reformatting.

Do not change the site's body font or global prose font unless the user explicitly requests a body-font change. Requests concerning screenshot text, figure text, code fonts, or isolated rendering issues do not authorize changing the global body font.

## Testing Guidelines

The required check is a successful production build. Run `npm run paper:check-config` when changing paper configuration tooling or citation rendering; these focused tests use Node.js's built-in test runner. There is no coverage requirement. For navigation or styling changes, also inspect both `/vndb/` and `/csdiy/` in the development server at desktop and mobile widths, plus their actual entry pages `/vndb/intro/` and `/csdiy/cse291a/` (the bare collection paths currently have no pages).

For browser-based visual checks, use the Codex built-in browser (`iab`) in a visible tab. Do not substitute a connected Chrome or Edge instance, a system-installed browser, Playwright, or headless browser automation. If the Codex built-in browser is unavailable, report the validation limitation instead of silently switching browsers. Start the local development server on an explicit loopback port, inspect every affected route at desktop and mobile viewport sizes, confirm images are loaded and do not introduce horizontal overflow, then reset temporary viewport overrides and close test tabs.

For paper figure and table screenshots, crop the complete visual object with only a small, even whitespace margin on all four sides. Do not include surrounding body prose, printed captions, or explanatory text outside the figure or table. Inspect the rendered PNG at actual pixels and adjust the crop until no meaningful stroke or glyph touches an edge and no side retains excessive whitespace.

After testing, stop any local development or preview servers started for the task. Do not leave test servers running when handing off the completed work.

## Add Paper Skill

When the user supplies an arXiv identifier or asks to add, typeset, translate, synchronize, or commit a paper, read and follow `skills/add-paper/SKILL.md`. Unless the user narrows the scope, complete the English, Simplified Chinese, and Japanese pages, validate the production build, and use the repository-wide commit behavior defined by that skill.

## Commit & Pull Request Guidelines

No repository history is available to establish an existing convention. Use concise imperative commits, optionally with Conventional Commit prefixes, for example `docs: add vndb reference page` or `style: adjust global typography`.

Pull requests should summarize the change, list affected routes, and report build results. Link related issues and include before/after screenshots for visible layout or typography changes. Do not include generated files or dependency changes unrelated to the proposal.
