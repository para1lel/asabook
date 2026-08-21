import { readFileSync, writeFileSync } from 'node:fs'
import { basename } from 'node:path'
import { load } from 'cheerio'

const args = process.argv.slice(2)
const updateConfig = args.includes('--update-config')
const [input, output, configPath = 'docs/.vuepress/config.ts'] = args.filter((value) => value !== '--update-config')
if (!input || !output) {
  throw new Error('Usage: node scripts/import-arxiv-html.mjs <input.html> <output.md> [config.ts] [--update-config]')
}

const $ = load(readFileSync(input, 'utf8'))
const article = $('article.ltx_document')
const config = readFileSync(configPath, 'utf8')
const slug = basename(output, '.md')

function normalizeSpace(text) {
  return text.replace(/[\u00a0\u200b]/g, ' ').replace(/\s+/g, ' ').trim()
}

function normalizeIdentity(text) {
  return normalizeSpace(text).toLowerCase().replace(/https?:\/\//g, '').replace(/[^a-z0-9]+/g, '')
}

function normalizeUrl(url) {
  return url
    .replace(/^https?:\/\//, '')
    .replace(/^dx\.doi\.org\//, 'doi.org/')
    .replace(/arxiv\.org\/(?:pdf|abs)\/([^v?#/]+)v\d+/, 'arxiv.org/abs/$1')
    .replace(/#.*$/, '')
    .replace(/\/$/, '')
    .toLowerCase()
}

function existingAbbreviations() {
  const entries = []
  for (const match of config.matchAll(/^\s*'([^']+)'\s*:\s*'((?:\\.|[^'])*)',?$/gm)) {
    const value = match[2].replace(/\\'/g, "'").replace(/\\\\/g, '\\')
    const urls = [...value.matchAll(/\]\((https?:\/\/[^)]+)\)/g)].map((item) => normalizeUrl(item[1]))
    entries.push({ key: match[1], normalized: normalizeIdentity(value), urls, value })
  }
  return entries
}

const existing = existingAbbreviations()
const occupiedKeys = new Set(existing.map((entry) => entry.key))
const bib = new Map()

$('#bib li.ltx_bibitem').each((_, element) => {
  const item = $(element)
  const id = item.attr('id')
  const tag = normalizeSpace(item.children('.ltx_tag').first().text())
  const blocks = item.children('.ltx_bibblock').map((__, block) => normalizeSpace($(block).text())).get().filter(Boolean)
  const title = blocks[1] ?? blocks[0] ?? ''
  const text = normalizeSpace(blocks.join(' '))
  const urls = item.find('a[href]').map((__, link) => $(link).attr('href')).get().filter((url) => /^https?:/.test(url))
  if (id && text) bib.set(id, { blocks, id, tag, text, title, urls })
})

function findExisting(reference) {
  const urls = reference.urls.map(normalizeUrl)
  const arxivIds = urls.map((url) => url.match(/arxiv\.org\/(?:abs|pdf)\/([^/?#]+)/)?.[1]).filter(Boolean)
  const title = normalizeIdentity(reference.title)
  const firstAuthor = normalizeIdentity(
    reference.tag.replace(/\bet\s+al\.?/i, '').replace(/\s*\(\d{4}[a-z]?\).*$/, '').split(/\s+and\s+/i)[0],
  )
  const exactUrl = existing.find((entry) => urls.some((url) => entry.urls.includes(url)))
  if (exactUrl) return exactUrl

  const matchingArxivId = existing.find((entry) => (
    arxivIds.some((id) => entry.normalized.includes(normalizeIdentity(id)))
  ))
  if (matchingArxivId) return matchingArxivId

  if (title.length < 16) return undefined
  return existing
    .filter((entry) => entry.normalized.includes(title) && entry.normalized.includes(firstAuthor))
    .sort((left, right) => right.urls.length - left.urls.length)[0]
}

function allocateKey(reference) {
  const tag = reference.tag.replace(/\bet\s+al\.?/i, '').replace(/\s*\(\d{4}[a-z]?\).*$/, '').trim()
  const surname = tag.split(/\s+and\s+/i)[0].replace(/[^A-Za-z]/g, '') || 'Ref'
  const prefix = surname.slice(0, 3)
  const year = (reference.tag.match(/\b(\d{4})[a-z]?\b/) ?? reference.text.match(/\b(\d{4})\b/))?.[1] ?? '00'
  const base = prefix[0].toUpperCase() + prefix.slice(1).toLowerCase() + year.slice(-2)
  if (!occupiedKeys.has(base)) {
    occupiedKeys.add(base)
    return base
  }
  for (let index = 1; index < 703; index += 1) {
    let value = index
    let suffix = ''
    while (value > 0) {
      value -= 1
      suffix = String.fromCharCode('a'.charCodeAt(0) + value % 26) + suffix
      value = Math.floor(value / 26)
    }
    const candidate = `${base}${suffix}`
    if (!occupiedKeys.has(candidate)) {
      occupiedKeys.add(candidate)
      return candidate
    }
  }
  throw new Error(`Too many citation collisions for ${base}`)
}

const citedIds = new Set(article.find('cite a[href^="#bib."]').map((_, link) => $(link).attr('href').slice(1)).get())
const citationKeys = new Map()
const additions = []
const references = []
for (const id of citedIds) {
  const reference = bib.get(id)
  if (!reference) throw new Error(`Missing bibliography entry ${id}`)
  const found = findExisting(reference)
  const key = found?.key ?? allocateKey(reference)
  citationKeys.set(id, key)
  references.push({
    id,
    key,
    source: reference.text,
    sourceTitle: reference.title,
    sourceUrls: reference.urls,
    reused: found?.value,
  })
  if (!found) {
    const link = reference.urls[0] ? ` [Link](${reference.urls[0]})` : ''
    additions.push({ key, value: `${reference.text}${link}` })
  }
}

const targets = new Map()
article.find('section[id]').each((_, element) => {
  const section = $(element)
  const id = section.attr('id')
  const heading = normalizeSpace(section.children('h1,h2,h3,h4,h5,h6').first().text())
  const number = heading.match(/^(\d+(?:\.\d+)*)\./)?.[1]
  if (number) targets.set(id, `section-${number.replaceAll('.', '-')}`)
})
article.find('section.ltx_appendix[id]').each((_, element) => {
  const section = $(element)
  const id = section.attr('id')
  const heading = normalizeSpace(section.children('h1,h2,h3,h4,h5,h6').first().text())
  const letter = heading.match(/^Appendix\s+([A-Z])\b/)?.[1]
  if (letter) targets.set(id, `appendix-${letter.toLowerCase()}`)
})
article.find('figure[id]').each((_, element) => {
  const id = $(element).attr('id')
  const figure = id.match(/\.F(\d+)$/)?.[1]
  const table = id.match(/\.T(\d+)$/)?.[1]
  if (figure) targets.set(id, `figure-${figure.padStart(2, '0')}`)
  if (table) targets.set(id, `table-${table.padStart(2, '0')}`)
})
article.find('[id]').each((_, element) => {
  const id = $(element).attr('id')
  const equation = id.match(/\.E(\d+)(?:\.|$)/)?.[1]
  if (equation) targets.set(id.split('.m')[0], `equation-${equation.padStart(2, '0')}`)
})

function cleanMath(value) {
  return value
    .replace(/^\\small\s*/, '')
    .replace(/\{\{([A-Za-z]+)\}\}/g, '{$1}')
    .replace(/\{clip\}/g, '\\mathrm{clip}')
    .replace(/\|\|/g, '\\mid\\mid')
    .replace(/\^\{T\}/g, '^\\top')
}

function renderCitation(node) {
  const keys = $(node).find('a[href^="#bib."]').map((_, link) => citationKeys.get($(link).attr('href').slice(1))).get()
  return `[${keys.join(', ')}]`
}

function renderInlineNode(node) {
  if (node.type === 'text') return node.data.replace(/[\u00a0\u200b]/g, ' ')
  if (node.type !== 'tag') return ''
  const element = $(node)
  const name = node.name
  if (name === 'cite') return renderCitation(node)
  if (name === 'math') return `$${cleanMath(element.attr('alttext') ?? element.text())}$`
  if (name === 'br') return '<br>'
  if (name === 'em' || element.hasClass('ltx_font_italic')) return `*${renderInlineChildren(node)}*`
  if (name === 'strong' || element.hasClass('ltx_font_bold')) return `**${renderInlineChildren(node)}**`
  if (name === 'code' || element.hasClass('ltx_font_typewriter') && name !== 'a') return `\`${renderInlineChildren(node)}\``
  if (name === 'sup') return `<sup>${renderInlineChildren(node)}</sup>`
  if (name === 'sub') return `<sub>${renderInlineChildren(node)}</sub>`
  if (name === 'a') {
    const href = element.attr('href') ?? ''
    const label = normalizeSpace(renderInlineChildren(node)) || href
    if (href.startsWith('#bib.')) return citationKeys.get(href.slice(1)) ?? label
    if (href.startsWith('#')) {
      const target = targets.get(href.slice(1))
      return target ? `[${label}](#${target})` : label
    }
    return href ? `[${label}](${href})` : label
  }
  if (name === 'svg') return ''
  return renderInlineChildren(node)
}

function renderInlineChildren(node) {
  return (node.children ?? []).map(renderInlineNode).join('')
}

function renderInline(node) {
  return renderInlineChildren(node).replace(/\s+/g, ' ').replace(/ +([,.;:!?])/g, '$1').trim()
}

function renderList(element, depth = 0) {
  const lines = []
  $(element).children('li').each((_, item) => {
    const clone = $(item).clone()
    clone.children('ul,ol').remove()
    clone.find('.ltx_tag').remove()
    lines.push(`${'  '.repeat(depth)}- ${renderInline(clone[0])}`)
    $(item).children('ul,ol').each((__, nested) => lines.push(renderList(nested, depth + 1)))
  })
  return lines.filter(Boolean).join('\n')
}

function renderEquation(element) {
  const container = $(element)
  const rows = container.find('tr.ltx_equation').toArray().map((row) => (
    $(row).find('math').toArray().map((math) => cleanMath($(math).attr('alttext') ?? $(math).text())
      .replace(/^\\displaystyle\s*/, '')).filter(Boolean)
  )).filter((row) => row.length > 0)
  if (rows.length === 0) return ''
  const sourceId = container.attr('id') ?? $(container.find('math').first()).attr('id')?.split('.m')[0]
  const target = targets.get(sourceId)
  const anchor = target ? `<span id="${target}"></span>\n\n` : ''
  const expression = rows.length === 1 && rows[0].length === 1
    ? rows[0][0]
    : `\\begin{aligned}\n${rows.map((row) => row.join(' & ')).join(' \\\\\n')}\n\\end{aligned}`
  return `${anchor}$$\n${expression}\n$$`
}

function renderFigure(element) {
  const figure = $(element)
  const id = figure.attr('id')
  const match = id?.match(/\.(F|T)(\d+)$/)
  if (!match) return ''
  const isTable = /\.T\d+$/.test(id) || figure.hasClass('ltx_table')
  const number = Number(match[2])
  const kind = isTable ? 'table' : 'figure'
  const label = isTable ? 'Table' : 'Figure'
  const anchor = `${kind}-${String(number).padStart(2, '0')}`
  const caption = figure.find('figcaption').first().clone()
  if (!caption.length) return ''
  caption.find('.ltx_tag').first().remove()
  const captionText = renderInline(caption[0])
  return `<span id="${anchor}"></span>\n\n![${label} ${number}. ${normalizeSpace(captionText)}](../../papers/${slug}/${anchor}.png)\n\n**${label} ${number}.** ${captionText}`
}

function renderParagraphContainer(element) {
  const container = $(element)
  const heading = container.children('h5,h6').first()
  const blocks = []
  let headingUsed = false
  for (const child of container.children().toArray()) {
    if (heading.length && child === heading[0]) continue
    const rendered = renderBlock(child)
    if (!rendered) continue
    if (!headingUsed && heading.length && /^\S/.test(rendered) && !/^(?:[-#<]|\$\$|:::)/.test(rendered)) {
      blocks.push(`**${normalizeSpace(heading.text())}** ${rendered}`)
      headingUsed = true
    } else {
      blocks.push(rendered)
    }
  }
  if (heading.length && !headingUsed) blocks.unshift(`**${normalizeSpace(heading.text())}**`)
  return blocks.join('\n\n')
}

function renderSection(element) {
  const section = $(element)
  if (section.attr('id') === 'bib') return ''
  const heading = section.children('h1,h2,h3,h4,h5,h6').first()
  const title = normalizeSpace(heading.text()).replace(/^(\d+(?:\.\d+)*)\.\s+/, '$1 ')
  const level = Math.min(6, Number(heading.prop('tagName')?.slice(1) ?? 2))
  const number = title.match(/^(\d+(?:\.\d+)*)\s/)?.[1]
  const appendix = title.match(/^Appendix\s+([A-Z])\b/)?.[1]
  const anchor = number
    ? `<span id="section-${number.replaceAll('.', '-')}"></span>\n\n`
    : appendix
      ? `<span id="appendix-${appendix.toLowerCase()}"></span>\n\n`
      : ''
  const body = section.children().toArray().filter((child) => child !== heading[0]).map(renderBlock).filter(Boolean).join('\n\n')
  return `${anchor}${'#'.repeat(level)} ${title}\n\n${body}`.trim()
}

function renderBlock(node) {
  if (node.type === 'text') return normalizeSpace(node.data)
  if (node.type !== 'tag') return ''
  const element = $(node)
  const name = node.name
  if (name === 'section') return renderSection(node)
  if (name === 'figure') return renderFigure(node)
  if (name === 'p') return renderInline(node)
  if (name === 'ul' || name === 'ol') return renderList(node)
  if (name === 'table' && element.find('tr.ltx_equation').length) return renderEquation(node)
  if (element.hasClass('ltx_equationgroup') || element.hasClass('ltx_equation')) return renderEquation(node)
  if (element.hasClass('ltx_para')) return renderParagraphContainer(node)
  if (name === 'blockquote') return renderInline(node).split('\n').map((line) => `> ${line}`).join('\n')
  if (name === 'div') return element.children().toArray().map(renderBlock).filter(Boolean).join('\n\n')
  return renderInline(node)
}

const abstract = article.find('.ltx_abstract').first()
const abstractText = abstract.find('p.ltx_p').toArray().map((paragraph) => renderInline(paragraph)).filter(Boolean).join('\n\n')
const content = article.children('figure.ltx_figure, section.ltx_section, section.ltx_appendix')
  .toArray()
  .map(renderBlock)
  .filter(Boolean)
const body = [`## Abstract\n\n${abstractText}`, ...content].join('\n\n')
  .replace(/\n{3,}/g, '\n\n')
  .replace(/\b(Figure|Table|Section|Sections) \[(\d+(?:\.\d+)*)\]\(#((?:figure|table|section)-[^)]+)\)/g, '[$1 $2](#$3)')
  .replace(/\b(?:Eq\.|Equation)\.?(?: \()?\[(\d+)\]\(#(equation-[^)]+)\)\)?/g, '[Equation $1](#$2)')
  .replace(/\bSec\. \[(\d+(?:\.\d+)*)\]\(#(section-[^)]+)\)/g, '[Section $1](#$2)')
  .replace(/[ \t]+$/gm, '')
  .trim()

writeFileSync(output, `${body}\n`)
writeFileSync(`${output}.citations.json`, `${JSON.stringify({ additions, citationKeys: Object.fromEntries(citationKeys), references }, null, 2)}\n`)
if (updateConfig && additions.length > 0) {
  const marker = '\n}\n\nfunction normalizePaperAbbreviation'
  if (!config.includes(marker)) throw new Error(`Cannot find paperAbbreviations boundary in ${configPath}`)
  const entries = additions.map(({ key, value }) => {
    const escaped = value.replaceAll('\\', '\\\\').replaceAll("'", "\\'")
    return `  '${key}': '${escaped}',`
  }).join('\n')
  writeFileSync(configPath, config.replace(marker, `\n${entries}${marker}`))
}
console.log(`Wrote ${output}: ${content.length} top-level objects, ${citedIds.size} references, ${additions.length} new abbreviations`)
