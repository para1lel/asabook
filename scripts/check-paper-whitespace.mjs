import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { basename, extname, join, resolve } from 'node:path'
import { createCanvas, loadImage } from '@napi-rs/canvas'

const root = resolve(import.meta.dirname, '..')
const papersDir = join(root, 'docs/papers')
const cropsDir = join(root, 'scripts/paper-crops')
const requested = process.argv.slice(2)
const errors = []

function paperSlugs() {
  const available = readdirSync(cropsDir, { withFileTypes: true })
    .filter((entry) => entry.isFile() && extname(entry.name) === '.json')
    .map((entry) => basename(entry.name, '.json'))
    .sort()

  if (requested.length === 0) return available
  for (const slug of requested) {
    if (!available.includes(slug)) errors.push(`unknown paper slug: ${slug}`)
  }
  return requested.filter((slug) => available.includes(slug))
}

function contentBounds(data, width, height) {
  const rowCounts = new Uint32Array(height)
  const columnCounts = new Uint32Array(width)

  for (let y = 0; y < height; y += 1) {
    for (let x = 0; x < width; x += 1) {
      const offset = (y * width + x) * 4
      const alpha = data[offset + 3]
      const isInk = alpha > 16 && (
        data[offset] < 248 || data[offset + 1] < 248 || data[offset + 2] < 248
      )
      if (!isInk) continue
      rowCounts[y] += 1
      columnCounts[x] += 1
    }
  }

  // Requiring a short run of ink avoids treating a lone antialiased pixel as
  // the edge of a figure.
  const rowThreshold = Math.max(2, Math.round(width * 0.0005))
  const columnThreshold = Math.max(2, Math.round(height * 0.0005))
  const top = rowCounts.findIndex((count) => count >= rowThreshold)
  const bottomFromEnd = [...rowCounts].reverse().findIndex((count) => count >= rowThreshold)
  const left = columnCounts.findIndex((count) => count >= columnThreshold)
  const rightFromEnd = [...columnCounts].reverse().findIndex((count) => count >= columnThreshold)
  if ([top, bottomFromEnd, left, rightFromEnd].includes(-1)) return undefined

  return {
    bottom: height - 1 - bottomFromEnd,
    left,
    right: width - 1 - rightFromEnd,
    top,
  }
}

function formatMargins(margins) {
  return `top=${margins.top}px right=${margins.right}px bottom=${margins.bottom}px left=${margins.left}px`
}

let checked = 0
for (const slug of paperSlugs()) {
  const crops = JSON.parse(readFileSync(join(cropsDir, `${slug}.json`), 'utf8'))
  for (const crop of crops) {
    const imagePath = join(papersDir, slug, `${crop.name}.png`)
    if (!existsSync(imagePath)) continue

    const image = await loadImage(imagePath)
    const canvas = createCanvas(image.width, image.height)
    const context = canvas.getContext('2d')
    context.drawImage(image, 0, 0)
    const bounds = contentBounds(
      context.getImageData(0, 0, image.width, image.height).data,
      image.width,
      image.height,
    )
    checked += 1
    if (!bounds) {
      errors.push(`${slug}/${crop.name}.png: no visible content detected`)
      continue
    }

    const margins = {
      bottom: image.height - 1 - bounds.bottom,
      left: bounds.left,
      right: image.width - 1 - bounds.right,
      top: bounds.top,
    }
    const minimum = crop.whitespace?.minimum ?? 4
    const maximum = crop.whitespace?.maximum
      ?? Math.max(32, Math.round(Math.min(image.width, image.height) * 0.03))
    const touches = Object.entries(margins).filter(([, value]) => value < minimum)
    const excessive = Object.entries(margins).filter(([, value]) => value > maximum)

    console.log(`${slug}/${crop.name}.png: ${formatMargins(margins)} (allowed ${minimum}-${maximum}px)`)
    if (touches.length > 0) {
      errors.push(`${slug}/${crop.name}.png: content touches ${touches.map(([side]) => side).join(', ')} edge; ${formatMargins(margins)}`)
    }
    if (excessive.length > 0) {
      errors.push(`${slug}/${crop.name}.png: excessive whitespace on ${excessive.map(([side]) => side).join(', ')} edge; ${formatMargins(margins)}`)
    }
  }
}

errors.forEach((message) => console.error(`ERROR: ${message}`))
if (errors.length > 0) {
  console.error(`\nWhitespace check failed with ${errors.length} error(s).`)
  process.exit(1)
}

console.log(`Whitespace check passed for ${checked} image(s).`)
