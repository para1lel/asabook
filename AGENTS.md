# Repository Guidelines

## Project Structure & Module Organization

This repository is a VuePress 2 documentation/blog site using the Plume theme and Vite bundler. Author content under `docs/`:

- `docs/README.md` is the site home page, written in Chinese.
- `docs/vndb/` contains visual novel / Galgame notes. The current entry point is `intro.md`.
- `docs/csdiy/` contains course, math, and problem-solving notes. Current pages include `aops.md` and `tst26-p18.md`.
- `docs/.vuepress/config.ts` defines site metadata, navigation, collection sidebars, and page entry points.
- `docs/.vuepress/client.ts` registers client-side styles and the LXGW WenKai webfont.
- `docs/.vuepress/styles/index.css` contains global theme overrides.

Generated output and caches live under `docs/.vuepress/dist`, `.cache`, and `.temp`; never edit or commit them. Add collection-specific images beside the Markdown page that references them when relative links are clearer, such as `docs/csdiy/miku.gif`; use `docs/.vuepress/public/` only for shared public assets.

## Build, Test, and Development Commands

- `npm install` installs the versions recorded in `package-lock.json`.
- `npm run docs:dev` starts the local documentation server with hot reload.
- `npm run docs:clean-dev` clears VuePress caches before starting; use this after configuration or dependency changes.
- `npm run docs:build` produces the static site and validates Markdown, routes, imports, and theme configuration.

Run `npm run docs:build` before submitting changes.

## Dependency and Software Installation

When a task requires a missing package, library, CLI, runtime, browser, or other software, you are authorized to install it without requesting additional permission. Prefer the narrowest practical installation scope and project-local dependencies when available; use a system-wide installation when the tool genuinely requires it. Use official or otherwise verified sources, avoid unrelated upgrades, and preserve any required manifest and lockfile changes.

## Coding Style & Naming Conventions

Use two-space indentation in TypeScript, JSON, CSS, and YAML frontmatter. Keep TypeScript imports at the top and prefer single quotes without semicolons, matching `config.ts`. Use lowercase kebab-case Markdown filenames, such as `tst26-p18.md`. Keep collection names displayed as `vndb` and `csdiy`, and preserve the site name `ASa Book`.

Keep all Markdown abbreviation definitions in `docs/.vuepress/config.ts` under `paperAbbreviations`; do not create per-page or per-paper abbreviation files.

Write content primarily in Chinese, with short Markdown sections, descriptive headings, and relative links for pages in the same collection. For paper editions governed by the add-paper skill, reproduce the English source word for word and sentence for sentence, and translate the Chinese and Japanese pages faithfully from that source. For other English translation pages, prioritize fluent, idiomatic English that reads naturally to native speakers while preserving the original meaning, links, math, and structure. Pages should include YAML frontmatter with `title`, `createTime`, and a stable `permalink` matching the configured collection path, for example `/csdiy/tst26-p18/`. Math content may use KaTeX syntax. When adding a page, also add its filename without `.md` to the appropriate collection sidebar in `docs/.vuepress/config.ts`, and update navbar entry points if the first page in a collection changes.

In paper editions, write internal section references as linked localized labels such as `[Section 3](#section-3)`, `[第 3 节](#section-3)`, and `[第 3 節](#section-3)`; never prefix them with `§` or `\S`. In Chinese paper pages, leave one space between a linked formula reference such as `[公式 4](#equation-04)` and following Han text, but no space before punctuation. In every fenced code block, put the least-indented nonblank content line at column zero and indent each nested level by exactly two spaces; never add a uniform base indent, use tabs, or use four spaces as one indentation level.

No formatter or linter is configured; follow the surrounding style and avoid unrelated reformatting.

Do not change the site's body font or global prose font unless the user explicitly requests a body-font change. Requests concerning screenshot text, figure text, code fonts, or isolated rendering issues do not authorize changing the global body font.

## Testing Guidelines

There is no automated test framework or coverage requirement. The required check is a successful production build. For navigation or styling changes, also inspect both `/vndb/` and `/csdiy/` in the development server at desktop and mobile widths.

For browser-based visual checks, use the connected Chrome browser when it is available. A failed generic or default browser lookup does not prove that Chrome is unavailable; when Chrome is requested, select the Chrome connection explicitly and verify it before reporting a limitation. Start the local development server on an explicit loopback port, inspect every affected route at desktop and mobile viewport sizes, confirm images are loaded and do not introduce horizontal overflow, then reset temporary viewport overrides and close test tabs.

For paper figure and table screenshots, crop the complete visual object with only a small, even whitespace margin on all four sides. Do not include surrounding body prose, printed captions, or explanatory text outside the figure or table. Inspect the rendered PNG at actual pixels and adjust the crop until no meaningful stroke or glyph touches an edge and no side retains excessive whitespace.

After testing, stop any local development or preview servers started for the task. Do not leave test servers running when handing off the completed work.

## Add Paper Skill

When the user supplies an arXiv identifier or asks to add, typeset, translate, synchronize, or commit a paper, read and follow `skills/add-paper/SKILL.md`. Unless the user narrows the scope, complete the English, Simplified Chinese, and Japanese pages, validate the production build, and use the repository-wide commit behavior defined by that skill.

## Commit & Pull Request Guidelines

No repository history is available to establish an existing convention. Use concise imperative commits, optionally with Conventional Commit prefixes, for example `docs: add vndb reference page` or `style: adjust global typography`.

Pull requests should summarize the change, list affected routes, and report build results. Link related issues and include before/after screenshots for visible layout or typography changes. Do not include generated files or dependency changes unrelated to the proposal.
