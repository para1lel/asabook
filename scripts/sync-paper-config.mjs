import { readFileSync, readdirSync, writeFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readPaperAbbreviations, writePaperAbbreviations } from './lib/paper-config.mjs'

const root = fileURLToPath(new URL('../', import.meta.url))
const configPath = resolve(root, 'docs/.vuepress/config/papers.ts')
const normalize = (value) => value.toLowerCase().replace(/[^a-z0-9]/g, '')
const identity = (url) => decodeURI(url).toLowerCase()
  .replace(/^https?:\/\//, '').replace(/^www\./, '')
  .replace(/(?:export\.)?arxiv.org\/(?:abs|pdf|src)\//, 'arxiv:')
  .replace(/doi.org\/10.48550\/arxiv\./, 'arxiv:')
  .replace(/(arxiv:\d{4}\.\d{4,5})(?:v\d+)?(?:\.pdf)?/, '$1')
  .replace(/[/.]+$/, '')
const urls = (value) => [...value.matchAll(/https?:\/\/[^\s)<>]+/g)].map((match) => identity(match[0]))

export function synchronizePaperConfig(source) {
  const papers = readdirSync(resolve(root, 'docs/en/papers')).filter((name) => name.endsWith('.md')).map((name) => {
    const page = readFileSync(resolve(root, 'docs/en/papers', name), 'utf8')
    const provenance = page.match(/^> .+(?:\n>.*)*/m)?.[0] ?? ''
    const title = page.match(/^title: ['"]?(.+?)['"]?$/m)?.[1] ?? ''
    const titles = [...provenance.matchAll(/\[([^\]]+)\]\((https?:\/\/[^)]+)\)/g)]
      .filter((match) => /arxiv.org\/abs\/|doi.org\/10\.|usenix.org\/conference\/[^/]+\/presentation\/|resources.nvidia.com\//.test(match[2]))
      .map((match) => match[1]).filter((label) => label.length > 15
        && !/^(?:arxiv|original|doi|tex|published|(?:\d+(?:st|nd|rd|th)\s+)?(?:proceedings|transactions|conference|symposium|usenix|acm|ieee|international|annual))/i.test(label))
    // LLMCompass was published under a different title from its arXiv edition.
    if (name === 'llmcompass.md') titles.push('LLMCompass: Enabling Efficient Hardware Design for Large Language Model Inference')
    // This early edition links its PDF and conference page without spelling out the full title.
    if (name === 'pet.md') titles.push('PET: Optimizing Tensor Programs with Partially Equivalent Transformations and Automated Corrections')
    const rawDate = provenance.match(/(?:first submitted(?: to arXiv)? on|first arXiv submission(?: date)?:?|published on)\s+([A-Z][a-z]+ \d{1,2},? \d{4}|\d{4}-\d{2}-\d{2})/i)?.[1]
    const timestamp = rawDate ? Date.parse(/^\d{4}-/.test(rawDate) ? `${rawDate}T00:00:00Z` : `${rawDate} UTC`) : NaN
    const date = Number.isNaN(timestamp) ? undefined : new Date(timestamp).toISOString().slice(0, 10)
    const arxivMonth = provenance.match(/arxiv.org\/abs\/(\d{2})(\d{2})\./)
    // Inconsistent source metadata is not a verified date; retain year precision.
    const consistentDate = date && (!arxivMonth || date.startsWith(`20${arxivMonth[1]}-${arxivMonth[2]}`)) ? date : undefined
    return { slug: name.slice(0, -3), urls: new Set(urls(provenance)), titles: titles.map(normalize), shortTitle: normalize(title), date: consistentDate }
  })
  const { entries } = readPaperAbbreviations(source)
  const linked = new Set()
  let count = 0
  for (const entry of entries) {
    const entryUrls = urls(entry.value)
    const text = normalize(entry.value)
    const matches = papers.filter((paper) => entry.value.includes(`/papers/${paper.slug}/`)
      || entryUrls.some((url) => paper.urls.has(url) && /^(?:arxiv:|doi.org\/10\.|proceedings\.[^/]+\/(?:paper|v\d)|dl.acm.org\/doi\/|usenix.org\/conference\/[^/]+\/presentation\/)/.test(url))
      || (!/\b(?:RFC\s+\d+|PyTorch Implementation)\b/i.test(entry.value)
        && (paper.titles.some((title) => text.includes(title))
          || [...entry.value.matchAll(/["“]([^"”]+)["”]/g)].some((match) => normalize(match[1]) === paper.shortTitle))))
    if (matches.length > 1) throw new Error(`Ambiguous paper identity for ${entry.key}: ${matches.map((paper) => paper.slug).join(', ')}`)
    if (!matches.length) continue
    const paper = matches[0]
    const link = `[Link](/papers/${paper.slug}/)`
    entry.value = /\[Link\]\([^)]+\)/.test(entry.value)
      ? entry.value.replace(/\[Link\]\([^)]+\)/g, link)
      : `${entry.value.trim()} ${link}`
    if (paper.date) entry.date = paper.date
    linked.add(paper.slug)
    count++
  }
  return { source: writePaperAbbreviations(source, entries), count, unmatched: papers.filter((paper) => !linked.has(paper.slug)).map((paper) => paper.slug) }
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const source = readFileSync(configPath, 'utf8')
  const result = synchronizePaperConfig(source)
  if (process.argv.includes('--check')) {
    if (source !== result.source) throw new Error('Paper links or ordering are stale; run npm run paper:config')
  } else writeFileSync(configPath, result.source)
  console.log(JSON.stringify({ linkedReferences: result.count, papersWithoutReferences: result.unmatched }))
}
