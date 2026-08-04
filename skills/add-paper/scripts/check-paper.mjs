#!/usr/bin/env node

import fs from 'node:fs'
import path from 'node:path'
import process from 'node:process'
import { fileURLToPath } from 'node:url'

const scriptDir = path.dirname(fileURLToPath(import.meta.url))
const root = path.resolve(scriptDir, '../../..')
const slug = process.argv[2]

if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
  console.error('Usage: node skills/add-paper/scripts/check-paper.mjs <kebab-case-slug>')
  process.exit(2)
}

const configPath = path.join(root, 'docs/.vuepress/config.ts')
const pages = [
  { locale: 'zh', path: path.join(root, `docs/papers/${slug}.md`), permalink: `/papers/${slug}/` },
  { locale: 'en', path: path.join(root, `docs/en/papers/${slug}.md`), permalink: `/en/papers/${slug}/` },
  { locale: 'ja', path: path.join(root, `docs/ja/papers/${slug}.md`), permalink: `/ja/papers/${slug}/` },
]
const pdfPath = path.join(root, `docs/.vuepress/public/paper/${slug}.pdf`)
const errors = []
const warnings = []

function fail(message) {
  errors.push(message)
}

function warn(message) {
  warnings.push(message)
}

function parseFrontmatter(markdown, label) {
  const match = markdown.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/)
  if (!match) {
    fail(`${label}: missing YAML frontmatter`)
    return {}
  }

  const values = {}
  for (const line of match[1].split(/\r?\n/)) {
    const field = line.match(/^([A-Za-z][A-Za-z0-9]*):\s*(.*)$/)
    if (field) values[field[1]] = field[2].replace(/^(['"])(.*)\1$/, '$2')
  }
  return values
}

function extractMath(markdown) {
  const expressions = []
  const withoutDisplays = markdown.replace(/\$\$([\s\S]*?)\$\$/g, (_, expression) => {
    expressions.push(expression)
    return '\n'
  })

  withoutDisplays.replace(/(?<!\\)\$(?!\$)([^$\n]+?)(?<!\\)\$/g, (_, expression) => {
    expressions.push(expression)
    return ''
  })
  return expressions
}

function plainMathWords(expression) {
  const shortWords = new Set(['and', 'arg', 'cos', 'exp', 'for', 'if', 'in', 'log', 'not', 'of', 'or', 'out', 'sin', 'tan', 'to'])
  const stripped = expression
    .replace(/\\mathrm\{[^{}]*\}/g, ' ')
    .replace(/\\(?:begin|end)\{[^{}]*\}/g, ' ')
    .replace(/\\texttt\{[^{}]*\}/g, ' ')
    .replace(/\\[A-Za-z]+/g, ' ')
  const candidates = stripped.match(/[A-Za-z]{2,}/g) ?? []
  return [...new Set(candidates.filter((word) => (
    word.length >= 4 || /[A-Z]{2}|[A-Z].*[a-z].*[A-Z]/.test(word) || shortWords.has(word.toLowerCase())
  )))]
}

function extractCitations(markdown) {
  const citations = new Set()
  for (const match of markdown.matchAll(/\[([^\]\n]+)\]/g)) {
    const keys = match[1].split(/,\s*/)
    if (keys.every((key) => /^[A-Z][A-Za-z]{1,5}\d{2}[a-z]?$/.test(key))) {
      keys.forEach((key) => citations.add(key))
    }
  }
  return citations
}

function extractImageBasenames(markdown, pagePath, label) {
  const names = []
  for (const match of markdown.matchAll(/!\[[^\]]*\]\(([^)\s]+)(?:\s+['"][^)]*['"])?\)/g)) {
    const target = match[1]
    if (/^(?:https?:|data:|#)/.test(target)) continue

    const resolved = target.startsWith('/')
      ? path.join(root, 'docs/.vuepress/public', target)
      : path.resolve(path.dirname(pagePath), target)
    if (!fs.existsSync(resolved)) fail(`${label}: missing image ${target}`)
    names.push(path.basename(target))
  }
  return names
}

function numberedHeadings(markdown) {
  return [...markdown.matchAll(/^#{2,6}\s+(\d+(?:\.\d+)*)\b/gm)].map((match) => match[1])
}

function equationTags(markdown) {
  return [...markdown.matchAll(/\\tag\{([^}]+)\}/g)].map((match) => match[1])
}

function consecutiveHyphenLines(markdown) {
  const withoutComments = markdown.replace(/<!--[\s\S]*?-->/g, (comment) => comment.replace(/[^\r\n]/g, ' '))
  const lines = withoutComments.split(/\r?\n/)
  const matches = []
  let inFrontmatter = lines[0] === '---'
  let fenceMarker = null

  for (const [index, line] of lines.entries()) {
    if (inFrontmatter) {
      if (index > 0 && line === '---') inFrontmatter = false
      continue
    }

    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const tableCells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|')
    const isTableSeparator = tableCells.length > 0 && tableCells.every((cell) => /^\s*:?-{3,}:?\s*$/.test(cell))
    if (isTableSeparator || /^\s*-{3,}\s*$/.test(line)) continue

    const visibleText = line
      .replace(/(`+).*?\1/g, '')
      .replace(/\]\((?:\\.|[^)])*\)/g, ']')
    if (visibleText.includes('--')) matches.push(index + 1)
  }

  return matches
}

if (!fs.existsSync(configPath)) fail('missing docs/.vuepress/config.ts')
if (!fs.existsSync(pdfPath)) fail(`missing PDF docs/.vuepress/public/paper/${slug}.pdf`)

const pageData = []
for (const page of pages) {
  const label = `${page.locale}:${path.relative(root, page.path)}`
  if (!fs.existsSync(page.path)) {
    fail(`${label}: file does not exist`)
    continue
  }

  const markdown = fs.readFileSync(page.path, 'utf8')
  const frontmatter = parseFrontmatter(markdown, label)
  if (!frontmatter.title) fail(`${label}: missing title`)
  if (!frontmatter.createTime) fail(`${label}: missing createTime`)
  if (frontmatter.permalink !== page.permalink) {
    fail(`${label}: expected permalink ${page.permalink}, found ${frontmatter.permalink ?? '(missing)'}`)
  }

  const math = extractMath(markdown)
  math.forEach((expression, index) => {
    if (/\|\||\\(?:lVert|rVert|Vert)\b/.test(expression)) {
      fail(`${label}: math expression ${index + 1} uses a norm delimiter other than \\|`)
    }
    if (/\\(?:mathrm|operatorname)\{(?:min|max)\}/.test(expression)) {
      fail(`${label}: math expression ${index + 1} must use \\min or \\max`)
    }
    const withoutRomanWords = expression.replace(/\\mathrm\{[^{}]*\}/g, '')
    if (/(?<![\\A-Za-z])(?:min|max)(?![A-Za-z])/.test(withoutRomanWords)) {
      fail(`${label}: math expression ${index + 1} contains raw min or max`)
    }
    const words = plainMathWords(expression)
    if (words.length > 0) {
      fail(`${label}: math expression ${index + 1} has multi-letter word(s) outside \\mathrm{}: ${words.join(', ')}`)
    }
  })

  const withoutMath = markdown.replace(/\$\$[\s\S]*?\$\$/g, '').replace(/(?<!\\)\$(?!\$)[^$\n]+?(?<!\\)\$/g, '')
  if (/\\tag\{[^}]+\}/.test(withoutMath)) fail(`${label}: equation tag appears outside math delimiters`)
  if (/^```pseudocode/m.test(markdown)) {
    warn(`${label}: review pseudocode fence; math-heavy algorithms must use nested unordered lists`)
  }
  const badHyphenLines = consecutiveHyphenLines(markdown)
  if (badHyphenLines.length > 0) {
    fail(`${label}: consecutive ASCII hyphens in rendered article content at line(s) ${badHyphenLines.join(', ')}`)
  }

  pageData.push({
    ...page,
    label,
    markdown,
    frontmatter,
    citations: extractCitations(markdown),
    images: extractImageBasenames(markdown, page.path, label),
    headings: numberedHeadings(markdown),
    tags: equationTags(markdown),
  })
}

if (pageData.length === pages.length) {
  const [base, ...localized] = pageData
  for (const page of localized) {
    if (page.frontmatter.title !== base.frontmatter.title) {
      fail(`${page.label}: title differs from ${base.label}`)
    }
    for (const field of ['headings', 'tags', 'images']) {
      if (JSON.stringify(page[field]) !== JSON.stringify(base[field])) {
        fail(`${page.label}: ${field} sequence differs from ${base.label}`)
      }
    }
    const baseCitations = [...base.citations].sort()
    const localizedCitations = [...page.citations].sort()
    if (JSON.stringify(localizedCitations) !== JSON.stringify(baseCitations)) {
      fail(`${page.label}: citation set differs from ${base.label}`)
    }
  }
}

if (fs.existsSync(configPath)) {
  const config = fs.readFileSync(configPath, 'utf8')
  const abbreviationKeys = new Set([...config.matchAll(/^\s*'([^']+)'\s*:/gm)].map((match) => match[1]))
  const allCitations = new Set(pageData.flatMap((page) => [...page.citations]))
  for (const citation of [...allCitations].sort()) {
    if (!abbreviationKeys.has(citation)) fail(`citation [${citation}] is missing from paperAbbreviations`)
  }

  const sidebarMentions = [...config.matchAll(new RegExp(`['"]${slug}['"]`, 'g'))].length
  if (sidebarMentions < 3) {
    fail(`expected ${slug} in all three locale sidebars; found ${sidebarMentions} config mention(s)`)
  } else if (sidebarMentions > 3) {
    warn(`${slug} appears ${sidebarMentions} times in config; verify only the three intended sidebars reference it`)
  }
}

for (const message of warnings) console.warn(`WARN: ${message}`)
for (const message of errors) console.error(`ERROR: ${message}`)

if (errors.length > 0) {
  console.error(`\nPaper check failed with ${errors.length} error(s) and ${warnings.length} warning(s).`)
  process.exit(1)
}

console.log(`Paper check passed for ${slug} (${warnings.length} warning(s)).`)
