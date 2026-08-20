import { existsSync, readFileSync, readdirSync } from 'node:fs'
import { basename, extname, join, resolve } from 'node:path'
import { loadImage } from '@napi-rs/canvas'

const root = resolve(import.meta.dirname, '..')
const papersDir = join(root, 'docs/papers')
const cropsDir = join(root, 'scripts/paper-crops')
const requested = process.argv.slice(2)
const errors = []
const warnings = []

function fail(message) {
  errors.push(message)
}

function warn(message) {
  warnings.push(message)
}

function paperSlugs() {
  const available = readdirSync(papersDir, { withFileTypes: true })
    .filter((entry) => entry.isDirectory() && existsSync(join(papersDir, `${entry.name}.md`)))
    .map((entry) => entry.name)
    .sort()

  if (requested.length === 0) return available
  for (const slug of requested) {
    if (!available.includes(slug)) fail(`unknown paper slug: ${slug}`)
  }
  return requested.filter((slug) => available.includes(slug))
}

function imageNames(slug) {
  const markdownPages = [
    join(papersDir, `${slug}.md`),
    join(root, `docs/en/papers/${slug}.md`),
    join(root, `docs/ja/papers/${slug}.md`),
  ]
  return [...new Set(
    markdownPages.flatMap((pagePath) => (
      [...readFileSync(pagePath, 'utf8').matchAll(/(?:\.\.\/\.\.\/papers\/|\.\/)([^/]+\/[^)\s]+\.png)/g)]
        .filter((match) => match[1].startsWith(`${slug}/`))
        .map((match) => basename(match[1]))
    )),
  )]
}

let checkedImages = 0
for (const slug of paperSlugs()) {
  const cropPath = join(cropsDir, `${slug}.json`)
  if (!existsSync(cropPath)) {
    fail(`${slug}: missing scripts/paper-crops/${slug}.json`)
    continue
  }

  const crops = JSON.parse(readFileSync(cropPath, 'utf8'))
  const cropByName = new Map()
  for (const crop of crops) {
    if (cropByName.has(crop.name)) fail(`${slug}: duplicate crop metadata for ${crop.name}`)
    cropByName.set(crop.name, crop)
  }

  const references = imageNames(slug)
  for (const imageName of references) {
    checkedImages += 1
    const name = basename(imageName, extname(imageName))
    const crop = cropByName.get(name)
    if (!crop) {
      fail(`${slug}: ${imageName} has no crop metadata`)
      continue
    }

    const imagePath = join(papersDir, slug, imageName)
    if (!existsSync(imagePath)) {
      fail(`${slug}: missing ${imageName}`)
      continue
    }
    const image = await loadImage(imagePath)

    if (crop.page !== undefined || crop.parts?.length > 0) {
      const parts = crop.parts ?? [crop]
      const requiredWidth = Math.max(...parts.map((part) => Math.round(part.width * 4)))
      const requiredHeight = parts.reduce((sum, part) => sum + Math.round(part.height * 4), 0)
      if (image.width < requiredWidth || image.height < requiredHeight) {
        fail(`${slug}: ${imageName} is ${image.width}x${image.height}, below the scale-4 requirement ${requiredWidth}x${requiredHeight}`)
      }
    } else if (crop.preserveSource) {
      if (!crop.sourceWidth || !crop.sourceHeight) {
        fail(`${slug}: ${imageName} preserves a source without recorded dimensions`)
      } else if (image.width !== crop.sourceWidth || image.height !== crop.sourceHeight) {
        fail(`${slug}: ${imageName} dimensions differ from its preserved source metadata`)
      }
      if (image.width < 1600) {
        fail(`${slug}: ${imageName} has no PDF crop and its preserved source width is below 1600 px`)
      }
    } else {
      fail(`${slug}: ${imageName} metadata has neither a PDF crop nor a preserved source`)
    }

    if (crop.matchScore !== undefined && crop.matchScore < 0.55) {
      warn(`${slug}: visually review low-confidence crop ${imageName} (${crop.matchScore.toFixed(3)})`)
    }
  }

  for (const crop of crops) {
    if (!references.includes(`${crop.name}.png`)) {
      warn(`${slug}: unused crop metadata ${crop.name}`)
    }
  }
}

warnings.forEach((message) => console.warn(`WARN: ${message}`))
errors.forEach((message) => console.error(`ERROR: ${message}`))

if (errors.length > 0) {
  console.error(`\nScreenshot check failed with ${errors.length} error(s) and ${warnings.length} warning(s).`)
  process.exit(1)
}

console.log(`Screenshot check passed for ${checkedImages} referenced image(s) (${warnings.length} warning(s)).`)
