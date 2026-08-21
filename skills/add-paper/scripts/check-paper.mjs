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
    .replace(/\\mathit\{[^{}]*\}/g, ' ')
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
    if (keys.every((key) => /^[A-Z][A-Za-z]{1,5}\d{2}[a-z]*$/.test(key))) {
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

function validateSectionHeadings(markdown, locale, label) {
  const apparatus = {
    en: /^(?:Abstract|Acknowledgements?|Acknowledgments?)$/i,
    zh: /^(?:摘要|致谢)$/u,
    ja: /^(?:概要|要旨|謝辞)$/u,
  }[locale]
  const headings = []
  const counters = []
  const lines = markdown.split(/\r?\n/)
  let fenceMarker = null

  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const heading = line.match(/^(#{2,6})\s+(.+)$/)
    if (!heading) continue
    const depth = heading[1].length - 1
    const title = heading[2].trim()
    const titleWithoutNumber = title.replace(/^\d+(?:\.\d+)*\s+/, '')
    if (apparatus.test(titleWithoutNumber)) {
      if (titleWithoutNumber !== title) {
        fail(`${label}: document-apparatus heading at line ${index + 1} must remain unnumbered`)
      }
      continue
    }

    const numbered = title.match(/^(\d+(?:\.\d+)*)\s+\S/)
    if (!numbered) {
      fail(`${label}: substantive heading at line ${index + 1} must start with an Arabic decimal number`)
      continue
    }

    const parts = numbered[1].split('.').map(Number)
    if (parts.length !== depth) {
      fail(`${label}: heading ${numbered[1]} at line ${index + 1} has ${parts.length} number level(s), expected ${depth}`)
      continue
    }

    counters.length = depth
    const expected = (counters[depth - 1] ?? 0) + 1
    if (parts[depth - 1] !== expected) {
      fail(`${label}: heading ${numbered[1]} at line ${index + 1} is out of sequence; expected ${[...parts.slice(0, -1), expected].join('.')}`)
    }
    for (let level = 0; level < depth - 1; level += 1) {
      if (parts[level] !== counters[level]) {
        fail(`${label}: heading ${numbered[1]} at line ${index + 1} does not match its parent section`)
        break
      }
    }
    counters[depth - 1] = parts[depth - 1]
    headings.push(numbered[1])

    const expectedAnchor = `section-${numbered[1].replaceAll('.', '-')}`
    let previous = index - 1
    while (previous >= 0 && lines[previous].trim() === '') previous -= 1
    if (lines[previous]?.trim() !== `<span id="${expectedAnchor}"></span>`) {
      fail(`${label}: heading ${numbered[1]} at line ${index + 1} must be preceded by anchor ${expectedAnchor}`)
    }
  }

  return headings
}

function validateSectionReferences(markdown, locale, label) {
  const anchors = new Set()
  for (const match of markdown.matchAll(/<span\s+id="(section-[^"]+)"><\/span>/g)) {
    if (!/^section-\d+(?:-\d+)*$/.test(match[1])) {
      fail(`${label}: non-decimal section anchor ${match[1]}`)
    }
    if (anchors.has(match[1])) fail(`${label}: duplicate section anchor ${match[1]}`)
    anchors.add(match[1])
  }

  const referencePattern = {
    en: /\bSections?\s+(\d+(?:\.\d+)*)/gu,
    zh: /第\s*(\d+(?:\.\d+)*)\s*节/gu,
    ja: /第\s*(\d+(?:\.\d+)*)\s*節/gu,
  }[locale]
  const stalePattern = locale === 'en'
    ? /\bSections?\s+[IVXLCDM]+(?:[-.][A-Z]\d*)?\b/gu
    : /第\s*[IVXLCDM]+(?:[-.][A-Z]\d*)?\s*[节節]/gu

  for (const [index, line] of markdown.split(/\r?\n/).entries()) {
    if (/^#{1,6}\s/.test(line)) continue
    if (stalePattern.test(line)) {
      fail(`${label}: non-decimal section reference at line ${index + 1}`)
    }
    stalePattern.lastIndex = 0

    const links = [...line.matchAll(/\[([^\]\n]+)\]\(#(section-[^)\s]+)\)/g)].map((match) => ({
      start: match.index,
      end: match.index + match[0].length,
      text: match[1],
      target: match[2],
    }))
    for (const link of links) {
      const number = [...link.text.matchAll(referencePattern)][0]?.[1]
      referencePattern.lastIndex = 0
      if (!number) {
        fail(`${label}: section link '${link.text}' at line ${index + 1} must use a localized decimal reference label`)
        continue
      }
      const expectedTarget = `section-${number.replaceAll('.', '-')}`
      if (link.target !== expectedTarget) {
        fail(`${label}: section reference '${link.text}' at line ${index + 1} targets #${link.target}, expected #${expectedTarget}`)
      }
      if (!anchors.has(link.target)) {
        fail(`${label}: section reference '${link.text}' at line ${index + 1} targets missing anchor #${link.target}`)
      }
    }

    for (const match of line.matchAll(referencePattern)) {
      const insideLink = links.some((link) => match.index >= link.start && match.index < link.end)
      if (!insideLink) {
        fail(`${label}: unlinked section reference '${match[0]}' at line ${index + 1}`)
      }
    }
    referencePattern.lastIndex = 0
  }
}

function equationAnchors(markdown) {
  return [...markdown.matchAll(/<span id="([^"]+)"><\/span>\s*\$\$/g)].map((match) => match[1])
}

function validateFormulaReferenceTargets(markdown, label) {
  const anchors = new Set(
    [...markdown.matchAll(/<span\s+id="([^"]+)"><\/span>/g)].map((match) => match[1]),
  )
  const targetPattern = /\]\(#((?:equation-[^)\s"]+)|(?:[AS]\d+(?:\.[A-Z]+)*\.E\d+))(?:\s+["'][^)]*["'])?\)/g
  for (const match of markdown.matchAll(targetPattern)) {
    if (!anchors.has(match[1])) fail(label + ': formula reference target #' + match[1] + ' does not exist')
  }
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

function validateFencedCodeIndentation(markdown, label) {
  const lines = markdown.split(/\r?\n/)
  let fenceMarker = null
  let fenceLine = null
  let indentationWidths = []

  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) {
        fenceMarker = marker
        fenceLine = index + 1
        indentationWidths = []
      } else if (fenceMarker === marker) {
        if (indentationWidths.length > 0 && Math.min(...indentationWidths) !== 2) {
          fail(`${label}: fenced code block at line ${fenceLine} must start indentation at two spaces`)
        }
        fenceMarker = null
        fenceLine = null
        indentationWidths = []
      }
      continue
    }
    if (fenceMarker === null) continue

    const indentation = line.match(/^[ \t]+(?=\S)/)?.[0]
    if (!indentation) continue
    if (indentation.includes('\t')) {
      fail(`${label}: fenced code block at line ${fenceLine} uses a tab at line ${index + 1}`)
      continue
    }
    if (indentation.length % 2 !== 0) {
      fail(`${label}: fenced code block at line ${fenceLine} uses non-two-space indentation at line ${index + 1}`)
    }
    indentationWidths.push(indentation.length)
  }
}

function validateRunInParagraphHeadings(markdown, label) {
  const lines = markdown.split(/\r?\n/)
  let fenceMarker = null

  for (let index = 0; index + 2 < lines.length; index += 1) {
    const line = lines[index]
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const heading = line.match(/^\*\*(.+)\*\*$/)?.[1]
    if (!heading || lines[index + 1] !== '') continue
    if (/^(?:Figures?|Tables?|Algorithms?|图|表|算法|図|アルゴリズム)\s*\d/iu.test(heading)) continue

    const following = lines[index + 2]
    const startsBlock = /^(?:#{1,6}\s|!\[|<|\||>|`{3,}|~{3,}|:::|---|\[\+|\$\$|[-+*]\s|\d+\.\s)/.test(following)
    if (following && !startsBlock) {
      fail(`${label}: run-in paragraph heading at line ${index + 1} must share a line with its paragraph`)
    }
  }
}

const formalStatementKinds = new Map([
  ['Definition', 'definition'],
  ['定义', 'definition'],
  ['定義', 'definition'],
  ['Lemma', 'lemma'],
  ['引理', 'lemma'],
  ['補題', 'lemma'],
  ['Proposition', 'proposition'],
  ['命题', 'proposition'],
  ['命題', 'proposition'],
  ['Corollary', 'corollary'],
  ['推论', 'corollary'],
  ['系', 'corollary'],
  ['Theorem', 'theorem'],
  ['定理', 'theorem'],
  ['Claim', 'claim'],
  ['断言', 'claim'],
  ['主張', 'claim'],
])
const formalStatementNames = [...formalStatementKinds.keys()].join('|')
const formalStatementNumber = '(?:[A-Z]\\.)?\\d+(?:\\.\\d+)*'
const formalStatementHeading = new RegExp(`^#{1,6}\\s+(?:${formalStatementNames})\\s+${formalStatementNumber}`, 'u')
const formalStatementRunIn = new RegExp(`^\\*\\*(${formalStatementNames})\\s+(${formalStatementNumber})([.。:：]?)\\*\\*(.*)$`, 'u')
const unboldedFormalStatement = new RegExp(`^(?:${formalStatementNames})\\s+${formalStatementNumber}[.。:：]`, 'u')
const standaloneProofLabel = /^(?:#{1,6}\s+)?(?:\*\*|__|\*|_)?(?:Proof|证明|証明)[.。:：]?(?:\*\*|__|\*|_)?\s*$/u
const proofRunInPrefix = /^(?:\*\*|__|\*|_)?(?:Proof|证明|証明)[.。:：](?:\*\*|__|\*|_)?\s+\S/u

function validateFormalStatementsAndProofs(markdown, locale, label) {
  const lines = markdown.split(/\r?\n/)
  const statements = []
  const expectedProofTitle = { en: 'Proof', zh: '证明', ja: '証明' }[locale]
  let proofCount = 0
  let fenceMarker = null

  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    if (formalStatementHeading.test(line)) {
      fail(`${label}: formal statement at line ${index + 1} must use a bold run-in label, not a heading`)
    }
    if (unboldedFormalStatement.test(line)) {
      fail(`${label}: formal statement label at line ${index + 1} must be bold`)
    }

    const statement = line.match(formalStatementRunIn)
    if (statement) {
      if (!statement[4].startsWith(' ') || statement[4].trim() === '') {
        fail(`${label}: formal statement label at line ${index + 1} must share its Markdown line with the statement`)
      }
      statements.push(`${formalStatementKinds.get(statement[1])}:${statement[2]}`)
    }

    if (standaloneProofLabel.test(line) || proofRunInPrefix.test(line)) {
      fail(`${label}: standalone proof label at line ${index + 1}; use a localized details container`)
    }

    const proofContainer = line.match(/^:{3,}\s+details(?:\s+(.*?))?\s*$/u)
    const proofTitle = proofContainer?.[1]?.trim() ?? ''
    if (proofContainer && /^(?:Proof|证明|証明)(?:[.。:：]|\s|$)/iu.test(proofTitle)) {
      proofCount += 1
      if (proofTitle !== expectedProofTitle) {
        fail(`${label}: proof container at line ${index + 1} must use the localized summary '${expectedProofTitle}'`)
      }
    }

    if (/(?:[∎□]|\$?\\square\$?)\s*[.。]?\s*$/u.test(line)) {
      fail(`${label}: terminal QED mark at line ${index + 1}; the proof container supplies the boundary`)
    }
  }

  return { statements, proofCount }
}

function validateNoTypesetTables(markdown, label) {
  const lines = markdown.split(/\r?\n/)
  let fenceMarker = null

  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const cells = line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|')
    const markdownSeparator = cells.length > 1
      && cells.every((cell) => /^\s*:?-{3,}:?\s*$/.test(cell))
    if (markdownSeparator) {
      fail(`${label}: Markdown table at line ${index + 1}; crop the published table from the PDF`)
    }
    if (/<\/?(?:table|thead|tbody|tfoot|tr|th|td)\b/i.test(line)) {
      fail(`${label}: HTML table markup at line ${index + 1}; crop the published table from the PDF`)
    }
  }
}

function figureTableReferencePattern() {
  return /\b(?:Figures?|Tables?)\s+\d+(?:\([a-z]\)|[a-z])?|\b(?:Figs?|Tabs?)\.\s*\d+(?:\([a-z]\)|[a-z])?|(?:图|図|表)\s*\d+(?:\([a-z]\)|[a-z])?/giu
}

function figureTableTarget(reference) {
  const number = reference.match(/\d+/)?.[0]
  const type = /^(?:Figures?|Figs?|图|図)/iu.test(reference) ? 'figure' : 'table'
  return `${type}-${number.padStart(2, '0')}`
}

function numberedFigureTableCaptions(markdown) {
  const captions = []
  const lines = markdown.split(/\r?\n/)
  let fenceMarker = null

  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const caption = line.match(/^\s*\*\*(Figures?|Tables?|图|図|表)\s*(\d+)(?:\([a-z]\)|[a-z])?[.。:]?/iu)
    if (!caption) continue

    const type = /^(?:Figures?|图|図)$/iu.test(caption[1]) ? 'figure' : 'table'
    captions.push({
      id: `${type}-${caption[2].padStart(2, '0')}`,
      line: index + 1,
    })
  }

  return captions
}

function numberedFigureTableImages(markdown) {
  const images = []
  const lines = markdown.split(/\r?\n/)
  for (const [index, line] of lines.entries()) {
    for (const match of line.matchAll(/!\[[^\]]*\]\(([^)\s]+)(?:\s+['"][^)]*['"])?\)/g)) {
      const basename = path.basename(match[1])
      const numbered = basename.match(/^(figure|table)-(\d{2})(?:[^0-9].*)?\.[^.]+$/)
      if (!numbered) continue
      images.push({
        id: `${numbered[1]}-${numbered[2]}`,
        line: index + 1,
      })
    }
  }
  return images
}

function validateFigureTableLinks(markdown, locale, label) {
  const anchorMatches = [...markdown.matchAll(/<span\s+id="((?:figure|table)-[^"]+)"><\/span>/g)]
  const anchors = new Set()
  for (const match of anchorMatches) {
    const id = match[1]
    if (!/^(?:figure|table)-\d{2}$/.test(id)) {
      fail(`${label}: nonstandard figure/table anchor ${id}; use a two-digit number`)
    }
    if (anchors.has(id)) fail(`${label}: duplicate figure/table anchor ${id}`)
    anchors.add(id)
  }

  const captions = numberedFigureTableCaptions(markdown)
  const images = numberedFigureTableImages(markdown)
  for (const object of [...captions, ...images]) {
    if (!anchors.has(object.id)) {
      fail(`${label}: ${object.id} object at line ${object.line} has no matching anchor`)
    }
  }

  const lines = markdown.split(/\r?\n/)
  for (const [index, line] of lines.entries()) {
    const anchor = line.match(/^<span id="((?:figure|table)-\d{2})"><\/span>$/)
    if (!anchor) continue

    let objectIndex = index + 1
    while (objectIndex < lines.length && lines[objectIndex].trim() === '') objectIndex += 1
    const object = lines[objectIndex]?.trim() ?? ''
    if (!/^(?:!\[|\||`{3,}|~{3,}|<(?:div|figure|table|img)\b)/.test(object)) {
      fail(`${label}: anchor ${anchor[1]} at line ${index + 1} is not immediately before a figure/table object`)
      continue
    }

    const image = object.match(/!\[[^\]]*\]\(([^)\s]+)/)
    if (image) {
      const basename = path.basename(image[1])
      const numbered = basename.match(/^(figure|table)-(\d{2})(?:[^0-9].*)?\.[^.]+$/)
      const expectedAnchor = numbered ? `${numbered[1]}-${numbered[2]}` : null
      if (expectedAnchor && anchor[1] !== expectedAnchor) {
        fail(`${label}: anchor ${anchor[1]} at line ${index + 1} precedes image ${expectedAnchor}`)
      }
    }
  }

  let fenceMarker = null
  for (const [index, line] of lines.entries()) {
    const fence = line.match(/^\s*(`{3,}|~{3,})/)
    if (fence) {
      const marker = fence[1][0]
      if (fenceMarker === null) fenceMarker = marker
      else if (fenceMarker === marker) fenceMarker = null
      continue
    }
    if (fenceMarker !== null) continue

    const links = [...line.matchAll(/(!?)\[([^\]\n]*)\]\((#[A-Za-z0-9_-]+)\)/g)].map((match) => ({
      start: match.index,
      end: match.index + match[0].length,
      image: match[1] === '!',
      text: match[2],
      target: match[3],
    }))
    const protectedRanges = []
    for (const match of line.matchAll(/!\[[^\]]*\]\([^)]*\)|`[^`]*`|\$[^$\n]*\$/g)) {
      protectedRanges.push([match.index, match.index + match[0].length])
    }
    for (const link of links.filter((item) => item.image)) {
      protectedRanges.push([link.start, link.end])
    }
    const caption = line.match(/^\s*\*\*(?:Figures?|Tables?|图|図|表)\s*\d+(?:\([a-z]\)|[a-z])?[.。:]?/iu)
    if (caption) protectedRanges.push([0, caption[0].length])

    if (locale === 'zh') {
      for (const match of line.matchAll(/(?:图|表)\s*\d+(?:\([a-z]\)|[a-z])?/gu)) {
        if (!/^(?:图|表) \d/u.test(match[0])) {
          fail(`${label}: add a space between the Chinese figure/table label and number at line ${index + 1}`)
        }
      }
    }

    for (const match of line.matchAll(figureTableReferencePattern())) {
      const start = match.index
      const end = start + match[0].length
      const protectedContent = protectedRanges.some(([rangeStart, rangeEnd]) => (
        start < rangeEnd && end > rangeStart
      ))
      if (protectedContent) continue

      const link = links.find((item) => !item.image && start >= item.start && end <= item.end)
      if (!link) {
        fail(`${label}: unlinked figure/table reference '${match[0]}' at line ${index + 1}`)
        continue
      }

      const expectedTarget = `#${figureTableTarget(match[0])}`
      if (link.target !== expectedTarget) {
        fail(`${label}: reference '${match[0]}' at line ${index + 1} targets ${link.target}, expected ${expectedTarget}`)
      }
      if (!anchors.has(link.target.slice(1))) {
        fail(`${label}: reference '${match[0]}' at line ${index + 1} targets missing anchor ${link.target}`)
      }

      if (locale === 'zh' && /^(?:图|表)/u.test(match[0])) {
        const after = line.slice(link.end)
        const renderedAfter = after.replace(/^(?:\*\*|__|[*_])+/u, '')
        if (/^\p{Script=Han}/u.test(renderedAfter)) {
          fail(`${label}: add a space after Chinese reference '${match[0]}' at line ${index + 1}`)
        }
      }
    }

    for (const link of links.filter((item) => /^#(?:figure|table)-/.test(item.target))) {
      if (!anchors.has(link.target.slice(1))) {
        fail(`${label}: link '${link.text}' at line ${index + 1} targets missing anchor ${link.target}`)
      }
      const numericText = link.text.trim().match(/^(\d+)(?:\([a-z]\)|[a-z])?$/i)
      if (numericText && !link.target.endsWith(`-${numericText[1].padStart(2, '0')}`)) {
        fail(`${label}: link '${link.text}' at line ${index + 1} targets the wrong figure/table number`)
      }
    }

    const bareContinuation = /\[(?:[^\]]+)\]\(#(?:figure|table)-\d{2}\)(?:,\s*|、\s*|\s+(?:and|or|to|through|和|或)\s+)(\d+(?:\([a-z]\)|[a-z])?)/giu
    for (const match of line.matchAll(bareContinuation)) {
      fail(`${label}: unlinked figure/table number '${match[1]}' in a compound reference at line ${index + 1}`)
    }
  }

  return captions.map((caption) => caption.id)
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
      warn(`${label}: review ambiguous multi-letter math sequence(s): ${words.join(', ')}; use \\mathrm{} only for text, units, labels, or abbreviations, and preserve adjacent variables such as RD or thm`)
    }
  })

  if (/\\tag\s*\{[^}]+\}/.test(markdown)) fail(`${label}: visible equation tags are not allowed`)
  if (/^```pseudocode/m.test(markdown)) {
    warn(`${label}: review pseudocode fence; math-heavy algorithms must use nested unordered lists`)
  }
  validateFencedCodeIndentation(markdown, label)
  validateRunInParagraphHeadings(markdown, label)
  const formalContent = validateFormalStatementsAndProofs(markdown, page.locale, label)
  validateNoTypesetTables(markdown, label)
  validateFormulaReferenceTargets(markdown, label)
  validateSectionReferences(markdown, page.locale, label)
  const badHyphenLines = consecutiveHyphenLines(markdown)
  if (badHyphenLines.length > 0) {
    fail(`${label}: consecutive ASCII hyphens in rendered article content at line(s) ${badHyphenLines.join(', ')}`)
  }
  const figureTableCaptions = validateFigureTableLinks(markdown, page.locale, label)
  const headings = validateSectionHeadings(markdown, page.locale, label)

  pageData.push({
    ...page,
    label,
    markdown,
    frontmatter,
    citations: extractCitations(markdown),
    images: extractImageBasenames(markdown, page.path, label),
    headings,
    equationAnchors: equationAnchors(markdown),
    figureTableCaptions,
    formalStatements: formalContent.statements,
    proofCount: formalContent.proofCount,
  })
}

if (pageData.length === pages.length) {
  const [base, ...localized] = pageData
  for (const page of localized) {
    if (page.frontmatter.title !== base.frontmatter.title) {
      fail(`${page.label}: title differs from ${base.label}`)
    }
    for (const field of ['headings', 'equationAnchors', 'images', 'figureTableCaptions', 'formalStatements']) {
      if (JSON.stringify(page[field]) !== JSON.stringify(base[field])) {
        fail(`${page.label}: ${field} sequence differs from ${base.label}`)
      }
    }
    if (page.proofCount !== base.proofCount) {
      fail(`${page.label}: proof-container count differs from ${base.label}`)
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
