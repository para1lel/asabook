import { readFileSync, writeFileSync } from 'node:fs'
import { basename, resolve } from 'node:path'

function parseArgs(argv) {
  const options = {}

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    if (!value.startsWith('--') && !options.input) options.input = value
    else if (value === '--markdown') options.markdown = argv[++index]
    else if (value === '--output') options.output = argv[++index]
    else if (value === '--rewrite') options.rewrite = true
    else throw new Error(`Unknown argument: ${value}`)
  }

  if (!options.input || !options.markdown || !options.output) {
    throw new Error('Usage: node scripts/extract-tei-bibliography.mjs <file.tei.xml> --markdown <page.md> --output <abbreviations.ts> [--rewrite]')
  }
  return options
}

function decodeXml(value = '') {
  return value
    .replace(/<[^>]+>/g, ' ')
    .replace(/&amp;/g, '&')
    .replace(/&lt;/g, '<')
    .replace(/&gt;/g, '>')
    .replace(/&quot;/g, '"')
    .replace(/&apos;/g, "'")
    .replace(/\s+/g, ' ')
    .trim()
}

function matches(block, expression) {
  return [...block.matchAll(expression)].map((match) => match[1])
}

function parseEntry(block) {
  const authors = matches(block, /<author\b[^>]*>([\s\S]*?)<\/author>/g).map((author) => {
    const names = matches(author, /<(?:forename|surname)\b[^>]*>([\s\S]*?)<\/(?:forename|surname)>/g)
    return names.map(decodeXml).filter(Boolean).join(' ')
  }).filter(Boolean)
  const titles = matches(block, /<title\b[^>]*>([\s\S]*?)<\/title>/g).map(decodeXml).filter(Boolean)
  const mainTitle = (decodeXml(block.match(/<title\b[^>]*(?:level="a"|type="main")[^>]*>([\s\S]*?)<\/title>/)?.[1]) || titles[0])
    ?.replace(/^\d+(?=[A-Z])/, '')
  const container = titles.find((title) => title !== mainTitle)
  const year = block.match(/<date\b[^>]*when="(\d{4})/)?.[1] || decodeXml(block.match(/<date\b[^>]*>([\s\S]*?)<\/date>/)?.[1]).match(/\d{4}/)?.[0]
  const url = block.match(/<ptr\b[^>]*target="([^"]+)"/)?.[1]?.replace(/\(visitedon.*$/i, '')
  const doi = decodeXml(block.match(/<idno\b[^>]*type="DOI"[^>]*>([\s\S]*?)<\/idno>/)?.[1])
  const note = decodeXml(block.match(/<note\b[^>]*>([\s\S]*?)<\/note>/)?.[1])
  const parts = []

  if (authors.length) parts.push(`${authors.join(', ')}.`)
  if (mainTitle) parts.push(`"${mainTitle}."`)
  if (container) parts.push(`${container}.`)
  if (note) parts.push(`${note}.`)
  if (year) parts.push(`${year}.`)
  if (url) parts.push(`[Link](${url})`)
  else if (doi) parts.push(`[DOI](https://doi.org/${doi})`)

  return { authors, title: mainTitle, year, text: parts.join(' ') }
}

function keyStem(entry) {
  const source = entry.authors[0]?.split(/\s+/).at(-1) || entry.title?.split(/\s+/)[0] || 'Ref'
  const ascii = source.normalize('NFKD').replace(/[\u0300-\u036f]/g, '').replace(/^[^A-Za-z]+/, '').replace(/[^A-Za-z0-9]/g, '')
  const name = (ascii || 'Ref').slice(0, 3)
  return `${name[0].toUpperCase()}${name.slice(1).toLowerCase()}${entry.year?.slice(-2) || 'Web'}`
}

function collectCitations(markdown) {
  const citations = new Set()
  for (const match of markdown.matchAll(/\[([0-9]+(?:\s*,\s*[0-9]+)*)\]/g)) {
    if (match[1].replace(/\s/g, '') === '0,1') continue
    for (const value of match[1].split(',')) citations.add(Number(value.trim()))
  }
  return [...citations].sort((left, right) => left - right)
}

const options = parseArgs(process.argv.slice(2))
const input = resolve(options.input)
const markdownPath = resolve(options.markdown)
const outputPath = resolve(options.output)
const xml = readFileSync(input, 'utf8')
let markdown = readFileSync(markdownPath, 'utf8')
const blocks = new Map()

for (const match of xml.matchAll(/<biblStruct\b[^>]*xml:id="b(\d+)"[^>]*>([\s\S]*?)<\/biblStruct>/g)) {
  blocks.set(Number(match[1]), parseEntry(match[2]))
}

const usedKeys = new Set()
const references = {}
const citationMap = new Map()
for (const number of collectCitations(markdown)) {
  const entry = blocks.get(number)
  if (!entry) throw new Error(`Citation [${number}] does not have a b${number} bibliography entry`)
  const stem = keyStem(entry)
  let key = stem
  let suffix = 0
  while (usedKeys.has(key)) key = `${stem}${String.fromCharCode(97 + suffix++)}`
  usedKeys.add(key)
  references[key] = entry.text
  citationMap.set(number, key)
}

const exportName = `${basename(options.output).replace(/\.[^.]+$/, '').replace(/[^A-Za-z0-9]+(.)/g, (_, character) => character.toUpperCase())}`
const lines = [`export const ${exportName} = {`]
for (const [key, text] of Object.entries(references)) lines.push(`  '${key}': ${JSON.stringify(text)},`)
lines.push('}', '')
writeFileSync(outputPath, lines.join('\n'))

if (options.rewrite) {
  markdown = markdown.replace(/\[([0-9]+(?:\s*,\s*[0-9]+)*)\]/g, (full, list) => {
    if (list.replace(/\s/g, '') === '0,1') return full
    return `[${list.split(',').map((value) => citationMap.get(Number(value.trim()))).join(', ')}]`
  })
  writeFileSync(markdownPath, markdown)
}

console.log(`Extracted ${citationMap.size} cited references from ${input}`)
