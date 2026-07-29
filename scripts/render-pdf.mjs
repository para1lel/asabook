import { mkdirSync, readFileSync, writeFileSync } from 'node:fs'
import { basename, extname, resolve } from 'node:path'
import { createCanvas, DOMMatrix, ImageData, Path2D } from '@napi-rs/canvas'

globalThis.DOMMatrix ??= DOMMatrix
globalThis.ImageData ??= ImageData
globalThis.Path2D ??= Path2D

const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs')

function parseArgs(argv) {
  const options = { scale: 2 }

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    if (!value.startsWith('--') && !options.input) options.input = value
    else if (value === '--output-dir') options.outputDir = argv[++index]
    else if (value === '--pages') options.pages = argv[++index]
    else if (value === '--scale') options.scale = Number(argv[++index])
    else if (value === '--crops') options.crops = argv[++index]
    else throw new Error(`Unknown argument: ${value}`)
  }

  if (!options.input) throw new Error('Usage: npm run paper:render -- <file.pdf> --output-dir <dir> [--pages 1,3-5] [--scale 2] [--crops crops.json]')
  if (!options.outputDir) throw new Error('--output-dir is required')
  if (!Number.isFinite(options.scale) || options.scale <= 0) throw new Error('--scale must be a positive number')
  return options
}

function parsePages(value, pageCount) {
  if (!value) return Array.from({ length: pageCount }, (_, index) => index + 1)

  const pages = new Set()
  for (const part of value.split(',')) {
    const match = part.match(/^(\d+)(?:-(\d+))?$/)
    if (!match) throw new Error(`Invalid page selection: ${part}`)
    const start = Number(match[1])
    const end = Number(match[2] ?? match[1])
    for (let page = start; page <= end; page += 1) pages.add(page)
  }

  for (const page of pages) {
    if (page < 1 || page > pageCount) throw new Error(`Page ${page} is outside 1-${pageCount}`)
  }
  return [...pages].sort((left, right) => left - right)
}

async function renderPage(pdf, pageNumber, scale) {
  const page = await pdf.getPage(pageNumber)
  const viewport = page.getViewport({ scale })
  const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height))
  const context = canvas.getContext('2d')
  context.fillStyle = '#fff'
  context.fillRect(0, 0, canvas.width, canvas.height)
  await page.render({ canvasContext: context, viewport, canvas }).promise
  return canvas
}

const options = parseArgs(process.argv.slice(2))
const input = resolve(options.input)
const outputDir = resolve(options.outputDir)
const inputName = basename(input, extname(input))
const loadingTask = getDocument({ data: new Uint8Array(readFileSync(input)), useSystemFonts: true })
const pdf = await loadingTask.promise
mkdirSync(outputDir, { recursive: true })

if (options.crops) {
  const crops = JSON.parse(readFileSync(resolve(options.crops), 'utf8'))
  const pages = new Map()

  for (const crop of crops) {
    if (!pages.has(crop.page)) pages.set(crop.page, await renderPage(pdf, crop.page, options.scale))
    const source = pages.get(crop.page)
    const x = Math.round(crop.x * options.scale)
    const y = Math.round(crop.y * options.scale)
    const width = Math.round(crop.width * options.scale)
    const height = Math.round(crop.height * options.scale)
    const canvas = createCanvas(width, height)
    const context = canvas.getContext('2d')
    context.fillStyle = '#fff'
    context.fillRect(0, 0, width, height)
    context.drawImage(source, x, y, width, height, 0, 0, width, height)
    writeFileSync(resolve(outputDir, `${crop.name}.png`), canvas.toBuffer('image/png'))
  }
  console.log(`Rendered ${crops.length} crops from ${pages.size} pages to ${outputDir}`)
} else {
  const pages = parsePages(options.pages, pdf.numPages)
  const digits = String(pdf.numPages).length
  for (const pageNumber of pages) {
    const canvas = await renderPage(pdf, pageNumber, options.scale)
    const pageName = String(pageNumber).padStart(digits, '0')
    writeFileSync(resolve(outputDir, `${inputName}-page-${pageName}.png`), canvas.toBuffer('image/png'))
  }
  console.log(`Rendered ${pages.length} of ${pdf.numPages} pages to ${outputDir}`)
}

await pdf.destroy()
