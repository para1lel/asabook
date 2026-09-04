import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  writeFileSync,
} from 'node:fs'
import { basename, extname, join, resolve } from 'node:path'
import { pathToFileURL } from 'node:url'
import { createCanvas, DOMMatrix, ImageData, loadImage, Path2D } from '@napi-rs/canvas'
import cvModule from '@techstark/opencv-js'

globalThis.DOMMatrix ??= DOMMatrix
globalThis.ImageData ??= ImageData
globalThis.Path2D ??= Path2D

const { getDocument, VerbosityLevel } = await import('pdfjs-dist/legacy/build/pdf.mjs')

const root = resolve(import.meta.dirname, '..')
const papersDir = join(root, 'docs/papers')
const publicPaperDir = join(root, 'docs/.vuepress/public/paper')
const cropsDir = join(root, 'scripts/paper-crops')
const locateScale = 2
const standardFontDataUrl = `${pathToFileURL(join(root, 'node_modules/pdfjs-dist/standard_fonts')).href}/`
const pdfWasmUrl = `${pathToFileURL(join(root, 'node_modules/pdfjs-dist/wasm')).href}/`
class LocalStandardFontDataFactory {
  constructor({ baseUrl }) {
    this.baseUrl = baseUrl
  }

  async fetch({ filename }) {
    const filePath = join(root, 'node_modules/pdfjs-dist/standard_fonts', filename)
    return new Uint8Array(readFileSync(filePath))
  }
}
class LocalWasmFactory {
  constructor({ baseUrl }) {
    this.baseUrl = baseUrl
  }

  async fetch({ filename }) {
    const filePath = join(root, 'node_modules/pdfjs-dist/wasm', filename)
    return new Uint8Array(readFileSync(filePath))
  }
}
const cv = cvModule instanceof Promise
  ? await cvModule
  : cvModule.Mat
    ? cvModule
    : await new Promise((resolveOpenCv) => {
        cvModule.onRuntimeInitialized = () => resolveOpenCv(cvModule)
      })

function parseArgs(argv) {
  const options = { scale: 4, write: false, slugs: [] }

  for (let index = 0; index < argv.length; index += 1) {
    const value = argv[index]
    if (value === '--write') options.write = true
    else if (value === '--scale') options.scale = Number(argv[++index])
    else if (value.startsWith('--')) throw new Error(`Unknown argument: ${value}`)
    else options.slugs.push(value)
  }

  if (!Number.isFinite(options.scale) || options.scale < 4) {
    throw new Error('--scale must be at least 4')
  }
  return options
}

function paperSlugs(requested) {
  const available = readdirSync(papersDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && existsSync(join(papersDir, `${entry.name}.md`)))
    .map((entry) => entry.name)
    .sort()

  if (requested.length === 0) return available
  for (const slug of requested) {
    if (!available.includes(slug)) throw new Error(`Unknown paper slug: ${slug}`)
  }
  return requested
}

function paperFiles(slug) {
  const markdown = readFileSync(join(papersDir, `${slug}.md`), 'utf8')
  const pdfTarget = markdown.match(/\]\(\/paper\/([^)]+\.pdf)\)/i)?.[1]
    ?? markdown.match(/href=["']\/paper\/([^"']+\.pdf)["']/i)?.[1]
  if (!pdfTarget) throw new Error(`${slug}: cannot find the local PDF link`)

  const localizedMarkdown = [
    markdown,
    readFileSync(join(root, `docs/en/papers/${slug}.md`), 'utf8'),
    readFileSync(join(root, `docs/ja/papers/${slug}.md`), 'utf8'),
  ]
  const imageMatches = localizedMarkdown.flatMap((content) => (
    [...content.matchAll(/(?:\.\.\/\.\.\/papers\/|\.\/)([^/]+\/[^)\s]+\.png)/g)]
      .filter((match) => match[1].startsWith(`${slug}/`))
      .map((match) => ({ content, match }))
  ))
  const imageNames = [...new Set(imageMatches.map(({ match }) => basename(match[1])))]
  if (imageNames.length === 0) throw new Error(`${slug}: no local PNG references found`)

  const imageHints = new Map()
  for (const { content, match } of imageMatches) {
    const anchors = [...content.slice(0, match.index).matchAll(/<span id="((?:figure|table)-\d{2})"><\/span>/g)]
    const anchor = anchors.at(-1)?.[1]
    if (anchor) imageHints.set(basename(match[1]), anchor)
  }

  return {
    cropPath: join(cropsDir, `${slug}.json`),
    imageNames,
    imageHints,
    outputDir: join(papersDir, slug),
    pdfPath: join(publicPaperDir, pdfTarget),
  }
}

async function imagePixels(file) {
  const image = await loadImage(file)
  const canvas = createCanvas(image.width, image.height)
  const context = canvas.getContext('2d')
  context.drawImage(image, 0, 0)
  const data = context.getImageData(0, 0, image.width, image.height).data
  return {
    canvas,
    data,
    normalized: normalizedGrayscale(canvas, 0, 0, image.width, image.height),
    pixels: new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4),
    width: image.width,
    height: image.height,
  }
}

function normalizedGrayscale(source, x, y, width, height) {
  const size = 32
  const canvas = createCanvas(size, size)
  const context = canvas.getContext('2d')
  context.fillStyle = '#fff'
  context.fillRect(0, 0, size, size)
  context.drawImage(source, x, y, width, height, 0, 0, size, size)
  const data = context.getImageData(0, 0, size, size).data
  const grayscale = new Uint8Array(size * size)
  for (let index = 0; index < grayscale.length; index += 1) {
    const offset = index * 4
    grayscale[index] = Math.round(
      data[offset] * 0.299 + data[offset + 1] * 0.587 + data[offset + 2] * 0.114,
    )
  }
  return grayscale
}

function imageDifference(left, right) {
  let difference = 0
  for (let index = 0; index < left.length; index += 1) {
    difference += Math.abs(left[index] - right[index])
  }
  return difference / left.length
}

function grayscaleMat(image) {
  const rgba = cv.matFromImageData({
    data: image.data,
    height: image.height,
    width: image.width,
  })
  const grayscale = new cv.Mat()
  cv.cvtColor(rgba, grayscale, cv.COLOR_RGBA2GRAY)
  rgba.delete()
  return grayscale
}

function matchAtScale(page, template, scale) {
  const width = Math.round(template.cvGray.cols * scale)
  const height = Math.round(template.cvGray.rows * scale)
  if (width < 12 || height < 12 || width > page.cvGray.cols || height > page.cvGray.rows) return null

  const resized = new cv.Mat()
  const result = new cv.Mat()
  cv.resize(
    template.cvGray,
    resized,
    new cv.Size(width, height),
    0,
    0,
    scale < 1 ? cv.INTER_AREA : cv.INTER_CUBIC,
  )
  cv.matchTemplate(page.cvGray, resized, result, cv.TM_CCOEFF_NORMED)
  const maximum = cv.minMaxLoc(result)
  resized.delete()
  result.delete()
  return {
    height,
    score: maximum.maxVal,
    width,
    x: maximum.maxLoc.x,
    y: maximum.maxLoc.y,
  }
}

function scaledTemplateMatch(page, template) {
  let best = null
  for (let scale = 0.15; scale <= 3; scale *= 1.12) {
    const match = matchAtScale(page, template, scale)
    if (match && (!best || match.score > best.score)) best = { ...match, scale }
  }
  if (!best) return null

  const start = Math.max(0.12, best.scale * 0.9)
  const end = Math.min(3.2, best.scale * 1.1)
  for (let scale = start; scale <= end; scale += (end - start) / 10) {
    const match = matchAtScale(page, template, scale)
    if (match && match.score > best.score) best = { ...match, scale }
  }
  return best
}

function colorChannels(color) {
  return {
    red: color & 0xff,
    green: (color >>> 8) & 0xff,
    blue: (color >>> 16) & 0xff,
    alpha: color >>> 24,
  }
}

function templateAnchors(template) {
  const frequency = new Map()
  for (const color of template.pixels) {
    const { red, green, blue, alpha } = colorChannels(color)
    if (alpha !== 255 || (red > 245 && green > 245 && blue > 245)) continue
    frequency.set(color, (frequency.get(color) ?? 0) + 1)
  }

  const candidates = []
  for (let offset = 0; offset < template.pixels.length; offset += 1) {
    const color = template.pixels[offset]
    if (!frequency.has(color)) continue
    candidates.push({
      color,
      count: frequency.get(color),
      x: offset % template.width,
      y: Math.floor(offset / template.width),
    })
  }
  candidates.sort((left, right) => left.count - right.count)

  const anchors = []
  const minimumDistance = Math.max(8, Math.min(template.width, template.height) / 8)
  for (const candidate of candidates) {
    if (anchors.every((anchor) => Math.hypot(anchor.x - candidate.x, anchor.y - candidate.y) >= minimumDistance)) {
      anchors.push(candidate)
    }
    if (anchors.length === 12) break
  }

  if (anchors.length === 0) throw new Error('image has no non-white opaque pixels')
  return anchors
}

function pageCandidates(name, pageTexts) {
  const numbered = name.match(/^(figure|table)-(\d+)/)
  if (!numbered) return new Set(pageTexts.map((_, index) => index + 1))

  const number = Number(numbered[2])
  const loosePattern = numbered[1] === 'figure'
    ? new RegExp(`\\b(?:fig(?:ure)?s?\\.?)\\s*${number}(?!\\d)`, 'i')
    : new RegExp(`\\btables?\\s*${number}(?!\\d)`, 'i')
  const captionPattern = numbered[1] === 'figure'
    ? new RegExp(`\\b(?:fig(?:ure)?\\.?)\\s*${number}\\s*[.:]`, 'i')
    : new RegExp(`\\btable\\s*${number}\\s*[.:]`, 'i')
  const captions = pageTexts
    .map((text, index) => captionPattern.test(text) ? index + 1 : null)
    .filter(Boolean)
  if (captions.length > 0) return new Set(captions)

  const matches = pageTexts
    .map((text, index) => loosePattern.test(text) ? index + 1 : null)
    .filter(Boolean)

  return new Set(matches.length > 0 ? matches : pageTexts.map((_, index) => index + 1))
}

function exactMatch(page, template) {
  const [primary, ...checks] = template.anchors
  let offset = page.pixels.indexOf(primary.color)

  while (offset !== -1) {
    const anchorX = offset % page.width
    const anchorY = Math.floor(offset / page.width)
    const x = anchorX - primary.x
    const y = anchorY - primary.y

    if (x >= 0 && y >= 0 && x + template.width <= page.width && y + template.height <= page.height) {
      const anchorsMatch = checks.every((anchor) => (
        page.pixels[(y + anchor.y) * page.width + x + anchor.x] === anchor.color
      ))
      if (anchorsMatch && allPixelsMatch(page, template, x, y)) return { x, y }
    }
    offset = page.pixels.indexOf(primary.color, offset + 1)
  }
  return null
}

function allPixelsMatch(page, template, x, y) {
  for (let row = 0; row < template.height; row += 1) {
    const pageStart = (y + row) * page.width + x
    const templateStart = row * template.width
    for (let column = 0; column < template.width; column += 1) {
      if (page.pixels[pageStart + column] !== template.pixels[templateStart + column]) return false
    }
  }
  return true
}

function drawDestination(args) {
  if (args.length === 2) return { x: args[0], y: args[1], width: args[-1]?.width, height: args[-1]?.height }
  if (args.length === 4) return { x: args[0], y: args[1], width: args[2], height: args[3] }
  if (args.length === 8) return { x: args[4], y: args[5], width: args[6], height: args[7] }
  return null
}

function transformedBounds(transform, destination, source) {
  const width = destination.width ?? source.width
  const height = destination.height ?? source.height
  const corners = [
    [destination.x, destination.y],
    [destination.x + width, destination.y],
    [destination.x, destination.y + height],
    [destination.x + width, destination.y + height],
  ].map(([x, y]) => ({
    x: transform.a * x + transform.c * y + transform.e,
    y: transform.b * x + transform.d * y + transform.f,
  }))
  const xs = corners.map((point) => point.x)
  const ys = corners.map((point) => point.y)
  const left = Math.min(...xs)
  const top = Math.min(...ys)
  return {
    x: left,
    y: top,
    width: Math.max(...xs) - left,
    height: Math.max(...ys) - top,
  }
}

async function renderPage(page, scale, captureImages = false) {
  const viewport = page.getViewport({ scale })
  const canvas = createCanvas(Math.ceil(viewport.width), Math.ceil(viewport.height))
  const context = canvas.getContext('2d')
  const draws = []
  if (captureImages) {
    const drawImage = context.drawImage.bind(context)
    context.drawImage = (source, ...args) => {
      const destination = drawDestination(args)
      if (destination) {
        draws.push({
          bounds: transformedBounds(context.getTransform(), destination, source),
          source,
        })
      }
      return drawImage(source, ...args)
    }
  }
  context.fillStyle = '#fff'
  context.fillRect(0, 0, canvas.width, canvas.height)
  await page.render({ canvasContext: context, viewport, canvas }).promise
  const data = context.getImageData(0, 0, canvas.width, canvas.height).data
  return {
    canvas,
    data,
    draws,
    pixels: new Uint32Array(data.buffer, data.byteOffset, data.byteLength / 4),
    width: canvas.width,
    height: canvas.height,
  }
}

async function locateCrops(pdf, files, existingCrops) {
  const pageTexts = []
  for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
    const page = await pdf.getPage(pageNumber)
    const content = await page.getTextContent()
    pageTexts.push(content.items.map((item) => item.str).join(' '))
  }

  const cropByName = new Map(existingCrops.map((crop) => [crop.name, crop]))
  const templates = []
  for (const imageName of files.imageNames) {
    const name = basename(imageName, extname(imageName))
    if (cropByName.has(name)) continue
    const template = await imagePixels(join(files.outputDir, imageName))
    templates.push({
      ...template,
      anchors: templateAnchors(template),
      candidatePages: pageCandidates(files.imageHints.get(imageName) ?? name, pageTexts),
      cvGray: grayscaleMat(template),
      name,
    })
  }

  const unmatched = new Set(templates)
  const approximateMatches = new Map()
  const scaledMatches = new Map()
  for (let pageNumber = 1; pageNumber <= pdf.numPages && unmatched.size > 0; pageNumber += 1) {
    const pageTemplates = [...unmatched].filter((template) => template.candidatePages.has(pageNumber))
    if (pageTemplates.length === 0) continue

    const page = await pdf.getPage(pageNumber)
    for (const scale of [1, locateScale]) {
      const scaleTemplates = pageTemplates.filter((template) => unmatched.has(template))
      if (scaleTemplates.length === 0) break
      const rendered = await renderPage(page, scale, scale === locateScale)
      rendered.cvGray = grayscaleMat(rendered)

      for (const template of scaleTemplates) {
        const match = exactMatch(rendered, template)
        if (match) {
          cropByName.set(template.name, {
            name: template.name,
            page: pageNumber,
            x: match.x / scale,
            y: match.y / scale,
            width: template.width / scale,
            height: template.height / scale,
          })
          unmatched.delete(template)
          continue
        }

        const currentScaled = scaledMatches.get(template)
        if (template.width < 1600 && (scale === 1 || !currentScaled || currentScaled.score < 0.8)) {
          const scaledMatch = scaledTemplateMatch(rendered, template)
          const previous = scaledMatches.get(template)
          if (scaledMatch && (!previous || scaledMatch.score > previous.score)) {
            scaledMatches.set(template, { ...scaledMatch, page: pageNumber, renderScale: scale })
          }
        }
        if (scale !== locateScale) continue

        for (const draw of rendered.draws) {
          const bounds = draw.bounds
          if (bounds.width < 8 || bounds.height < 8) continue
          if (bounds.x < 0 || bounds.y < 0 || bounds.x + bounds.width > rendered.width || bounds.y + bounds.height > rendered.height) continue
          if (draw.source.width === rendered.width && draw.source.height === rendered.height) continue

          const templateRatio = template.width / template.height
          const boundsRatio = bounds.width / bounds.height
          if (Math.abs(Math.log(templateRatio / boundsRatio)) > 0.12) continue

          const normalized = normalizedGrayscale(
            rendered.canvas,
            bounds.x,
            bounds.y,
            bounds.width,
            bounds.height,
          )
          const score = imageDifference(template.normalized, normalized)
          const previous = approximateMatches.get(template)
          if (!previous || score < previous.score) {
            approximateMatches.set(template, { bounds, page: pageNumber, score })
          }
        }
      }
      rendered.cvGray?.delete()
    }
  }

  for (const template of [...unmatched]) {
    const scaled = scaledMatches.get(template)
    if (scaled?.score >= 0.3) {
      const margin = 2
      const page = await pdf.getPage(scaled.page)
      const viewport = page.getViewport({ scale: 1 })
      const x = Math.max(0, scaled.x / scaled.renderScale - margin)
      const y = Math.max(0, scaled.y / scaled.renderScale - margin)
      const width = Math.min(viewport.width - x, scaled.width / scaled.renderScale + margin * 2)
      const height = Math.min(viewport.height - y, scaled.height / scaled.renderScale + margin * 2)
      cropByName.set(template.name, {
        name: template.name,
        page: scaled.page,
        x: Math.round(x * 2) / 2,
        y: Math.round(y * 2) / 2,
        width: Math.round(width * 2) / 2,
        height: Math.round(height * 2) / 2,
        matchScore: Math.round(scaled.score * 1000) / 1000,
        preserveSource: template.width >= width * 4 && template.height >= height * 4,
      })
      unmatched.delete(template)
      continue
    }

    const match = approximateMatches.get(template)
    if (!match || match.score > 24) {
      if (template.width >= 1600) {
        cropByName.set(template.name, {
          name: template.name,
          preserveSource: true,
          sourceHeight: template.height,
          sourceWidth: template.width,
        })
        unmatched.delete(template)
      }
      continue
    }
    cropByName.set(template.name, {
      name: template.name,
      page: match.page,
      x: Math.round(match.bounds.x) / locateScale,
      y: Math.round(match.bounds.y) / locateScale,
      width: Math.round(match.bounds.width) / locateScale,
      height: Math.round(match.bounds.height) / locateScale,
      preserveSource: template.width >= match.bounds.width / locateScale * 4
        && template.height >= match.bounds.height / locateScale * 4,
    })
    unmatched.delete(template)
  }

  templates.forEach((template) => template.cvGray.delete())

  return {
    crops: files.imageNames
      .map((imageName) => cropByName.get(basename(imageName, extname(imageName))))
      .filter(Boolean),
    unmatched: [...unmatched].map((template) => ({
      name: template.name,
      ...(scaledMatches.get(template) ?? approximateMatches.get(template) ?? {}),
    })),
  }
}

async function writeCrops(pdf, files, crops, scale) {
  mkdirSync(files.outputDir, { recursive: true })
  const screenshotCrops = crops.filter((crop) => !crop.preserveSource)
  const pageNumbers = [...new Set(screenshotCrops.flatMap((crop) => (
    crop.parts?.map((part) => part.page) ?? [crop.page]
  )))].sort((left, right) => left - right)
  const pages = new Map()

  for (const pageNumber of pageNumbers) {
    const page = await pdf.getPage(pageNumber)
    pages.set(pageNumber, (await renderPage(page, scale)).canvas)
  }

  for (const crop of screenshotCrops) {
    const parts = crop.parts ?? [crop]
    const renderedParts = parts.map((part) => ({
      ...part,
      height: Math.round(part.height * scale),
      width: Math.round(part.width * scale),
      x: Math.round(part.x * scale),
      y: Math.round(part.y * scale),
    }))
    const width = Math.max(...renderedParts.map((part) => part.width))
    const height = renderedParts.reduce((sum, part) => sum + part.height, 0)
    const canvas = createCanvas(width, height)
    const context = canvas.getContext('2d')
    context.fillStyle = '#fff'
    context.fillRect(0, 0, width, height)
    let offsetY = 0
    for (const part of renderedParts) {
      const offsetX = Math.floor((width - part.width) / 2)
      context.drawImage(
        pages.get(part.page),
        part.x,
        part.y,
        part.width,
        part.height,
        offsetX,
        offsetY,
        part.width,
        part.height,
      )
      offsetY += part.height
    }
    const rotation = ((crop.rotate ?? 0) % 360 + 360) % 360
    if (![0, 90, 180, 270].includes(rotation)) {
      throw new Error(`${crop.name}: rotate must be a multiple of 90 degrees`)
    }
    if (rotation === 0) {
      writeFileSync(join(files.outputDir, `${crop.name}.png`), canvas.toBuffer('image/png'))
      continue
    }

    const rotatedWidth = rotation % 180 === 0 ? width : height
    const rotatedHeight = rotation % 180 === 0 ? height : width
    const rotatedCanvas = createCanvas(rotatedWidth, rotatedHeight)
    const rotatedContext = rotatedCanvas.getContext('2d')
    rotatedContext.fillStyle = '#fff'
    rotatedContext.fillRect(0, 0, rotatedWidth, rotatedHeight)
    if (rotation === 90) {
      rotatedContext.translate(rotatedWidth, 0)
      rotatedContext.rotate(Math.PI / 2)
    } else if (rotation === 180) {
      rotatedContext.translate(rotatedWidth, rotatedHeight)
      rotatedContext.rotate(Math.PI)
    } else {
      rotatedContext.translate(0, rotatedHeight)
      rotatedContext.rotate(-Math.PI / 2)
    }
    rotatedContext.drawImage(canvas, 0, 0)
    writeFileSync(join(files.outputDir, `${crop.name}.png`), rotatedCanvas.toBuffer('image/png'))
  }
}

const options = parseArgs(process.argv.slice(2))
const failures = []

for (const slug of paperSlugs(options.slugs)) {
  const files = paperFiles(slug)
  if (!existsSync(files.pdfPath)) throw new Error(`${slug}: missing PDF ${files.pdfPath}`)

  const existingCrops = existsSync(files.cropPath)
    ? JSON.parse(readFileSync(files.cropPath, 'utf8'))
    : []
  const loadingTask = getDocument({
    data: new Uint8Array(readFileSync(files.pdfPath)),
    standardFontDataUrl,
    StandardFontDataFactory: LocalStandardFontDataFactory,
    useSystemFonts: false,
    verbosity: VerbosityLevel.ERRORS,
    wasmUrl: pdfWasmUrl,
    WasmFactory: LocalWasmFactory,
  })
  const pdf = await loadingTask.promise
  const result = await locateCrops(pdf, files, existingCrops)

  if (result.unmatched.length > 0) {
    const names = result.unmatched.map(({ name, page, score, x, y }) => (
      score === undefined ? name : `${name} (${score.toFixed(2)} @ p${page ?? '?'}, ${x ?? '?'},${y ?? '?'})`
    ))
    failures.push(`${slug}: ${names.join(', ')}`)
    console.error(`${slug}: could not locate ${result.unmatched.length}/${files.imageNames.length} image(s)`)
  } else if (options.write) {
    mkdirSync(cropsDir, { recursive: true })
    writeFileSync(files.cropPath, `${JSON.stringify(result.crops, null, 2)}\n`)
    await writeCrops(pdf, files, result.crops, options.scale)
    console.log(`${slug}: rendered ${result.crops.length} image(s) at scale ${options.scale}`)
  } else {
    console.log(`${slug}: located ${result.crops.length} image(s)`)
  }

  await pdf.destroy()
}

if (failures.length > 0) {
  console.error('\nUnmatched images:')
  failures.forEach((failure) => console.error(`- ${failure}`))
  process.exit(1)
}
