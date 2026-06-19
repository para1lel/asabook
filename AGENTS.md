# Repository Guidelines

## Project Structure & Module Organization

This repository is a VuePress 2 documentation site using the Plume theme and Vite bundler. Author content under `docs/`:

- `docs/vndb/` and `docs/csdiy/` contain the two documentation collections.
- `docs/.vuepress/config.ts` defines site metadata, navigation, collections, and sidebars.
- `docs/.vuepress/client.ts` registers client-side styles and the LXGW WenKai webfont.
- `docs/.vuepress/styles/index.css` contains global theme overrides.

Generated output and caches live under `docs/.vuepress/dist`, `.cache`, and `.temp`; never edit or commit them. Add images and other public files under `docs/.vuepress/public/` when needed.

## Build, Test, and Development Commands

- `npm install` installs the versions recorded in `package-lock.json`.
- `npm run docs:dev` starts the local documentation server with hot reload.
- `npm run docs:clean-dev` clears VuePress caches before starting; use this after configuration or dependency changes.
- `npm run docs:build` produces the static site and validates Markdown, routes, imports, and theme configuration.

Run `npm run docs:build` before submitting changes.

## Coding Style & Naming Conventions

Use two-space indentation in TypeScript, JSON, CSS, and YAML frontmatter. Keep TypeScript imports at the top and prefer single quotes without semicolons, matching `config.ts`. Use lowercase kebab-case Markdown filenames, such as `getting-started.md`. Keep collection names displayed as `vndb` and `csdiy`, and preserve the site name `ASa Book`.

Write short Markdown sections with descriptive headings and relative links for pages in the same collection. When adding a page, also add its filename (without `.md`) to the appropriate collection sidebar in `docs/.vuepress/config.ts`.

No formatter or linter is configured; follow the surrounding style and avoid unrelated reformatting.

## Testing Guidelines

There is no automated test framework or coverage requirement. The required check is a successful production build. For navigation or styling changes, also inspect both `/vndb/` and `/csdiy/` in the development server at desktop and mobile widths.

## Commit & Pull Request Guidelines

No repository history is available to establish an existing convention. Use concise imperative commits, optionally with Conventional Commit prefixes, for example `docs: add vndb reference page` or `style: adjust global typography`.

Pull requests should summarize the change, list affected routes, and report build results. Link related issues and include before/after screenshots for visible layout or typography changes. Do not include generated files or dependency changes unrelated to the proposal.
