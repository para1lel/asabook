import { execFileSync } from 'node:child_process'
import {
  copyFileSync,
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs'
import path from 'node:path'
import process from 'node:process'

const sourceRoot = path.resolve(process.argv[2] || '')
const repositoryRoot = path.resolve(import.meta.dirname, '..')
const targetRoot = path.join(repositoryRoot, 'docs', 'en', 'yokubi')

if (!process.argv[2] || !existsSync(path.join(sourceRoot, 'src', 'SUMMARY.md'))) {
  console.error('Usage: node scripts/import-yokubi.mjs <path-to-yokubi-repository>')
  process.exit(1)
}

if (existsSync(targetRoot)) {
  console.error(`Refusing to overwrite existing directory: ${targetRoot}`)
  process.exit(1)
}

const entries = [
  ['Introduction.md', 'README.md'],
  ['Before-you-begin.md', 'before-you-begin.md'],
  ['Preamble.md', 'preamble.md'],
  ['Section1/Part1.md', 'part-1.md'],
  ['Section1/Part2.md', 'part-2.md'],
  ['Section2/Section2.md', 'absolute-territory.md'],
  ['Section2/Part3.md', 'part-3.md'],
  ['Section2/Part4.md', 'part-4.md'],
  ['ClosingWords.md', 'closing-words.md'],
  ['FAQ.md', 'faq.md'],
  ['Credits.md', 'credits.md'],
]

for (let lesson = 0; lesson <= 63; lesson += 1) {
  const part = lesson <= 18 ? 1 : lesson <= 28 ? 2 : lesson <= 44 ? 3 : 4
  const section = part <= 2 ? 'Section1' : 'Section2'
  entries.push([
    `${section}/Part${part}/Lesson${lesson}.md`,
    `part-${part}/lesson-${String(lesson).padStart(2, '0')}.md`,
  ])
}

const targetBySource = new Map(entries)

function sourceCreateTime(sourceRelativePath) {
  const history = execFileSync(
    'git',
    ['log', '--follow', '--format=%aI', '--', `src/${sourceRelativePath}`],
    { cwd: sourceRoot, encoding: 'utf8' },
  ).trim().split(/\r?\n/)
  const created = history.at(-1)

  if (!created) {
    throw new Error(`Unable to find creation time for ${sourceRelativePath}`)
  }

  return created.slice(0, 19).replaceAll('-', '/').replace('T', ' ')
}

function pageTitle(sourceRelativePath, content) {
  const heading = content.match(/^#\s+(.+)$/m)?.[1]?.trim()

  if (sourceRelativePath === 'Introduction.md') return 'Introduction'
  if (!heading) throw new Error(`Unable to find title for ${sourceRelativePath}`)

  const lesson = sourceRelativePath.match(/Lesson(\d+)\.md$/)?.[1]
  return lesson ? `Lesson ${lesson}: ${heading}` : heading
}

function pagePermalink(targetRelativePath) {
  if (targetRelativePath === 'README.md') return '/en/yokubi/'
  return `/en/yokubi/${targetRelativePath.replace(/\.md$/, '')}/`
}

function rewriteInternalLinks(content, sourceRelativePath, targetRelativePath) {
  return content.replace(/\]\(([^)]+?\.md)(#[^)]+)?\)/g, (match, href, hash = '') => {
    const resolvedSource = path.posix.normalize(
      path.posix.join(path.posix.dirname(sourceRelativePath), href),
    )
    const resolvedTarget = targetBySource.get(resolvedSource)

    if (!resolvedTarget) {
      throw new Error(`Unmapped link in ${sourceRelativePath}: ${href}`)
    }

    let relativeTarget = path.posix.relative(
      path.posix.dirname(targetRelativePath),
      resolvedTarget,
    )
    if (!relativeTarget.startsWith('.')) relativeTarget = `./${relativeTarget}`
    return `](${relativeTarget}${hash})`
  })
}

function transformContent(sourceRelativePath, targetRelativePath, rawContent) {
  let content = rawContent.replace(/\r\n/g, '\n')
  const title = pageTitle(sourceRelativePath, content)

  content = content
    .replace(/^\s*#\s+.+\n+/, '')
    .replace(/\{f\|([^{}|\n]+)\|([^{}|\n]+)\}/g, '<ruby>$1<rt>$2</rt></ruby>')
    .replace(
      /<div class="warning">\s*\n([\s\S]*?)\n<\/div>/g,
      '::: warning\n$1\n:::',
    )
    .replace(/<pre>/g, '<pre class="yokubi-example">')
    .replace(
      /<center>\s*\n<font size=50>\s*\n([\s\S]*?)\n<\/font>\s*\n<\/center>/g,
      '<p class="yokubi-closing-mark">$1</p>',
    )
    .replace(
      /<a href=(?:"([^"]+)"|([^\s>]+))>([\s\S]*?)<\/a>/g,
      (_, quotedHref, bareHref, label) => `[${label}](${quotedHref || bareHref})`,
    )
    .replace(
      /<iframe[\s\S]*?src="https:\/\/www\.youtube\.com\/embed\/([^?"\s]+)[^"]*"[\s\S]*?<\/iframe>/g,
      '@[youtube]($1)',
    )

  content = rewriteInternalLinks(content, sourceRelativePath, targetRelativePath)

  if (sourceRelativePath === 'Introduction.md') {
    content = content.replace('](images/logo.svg)', '](./logo.svg)')
  }

  if (sourceRelativePath === 'Before-you-begin.md') {
    const embeddedImage = content.match(
      /\[image1\]: <data:image\/png;base64,([^>]+)>/,
    )
    if (!embeddedImage) throw new Error('Unable to find embedded font-check image')

    writeFileSync(
      path.join(targetRoot, 'japanese-font-check.png'),
      Buffer.from(embeddedImage[1], 'base64'),
    )
    content = content.replace(embeddedImage[0], '[image1]: ./japanese-font-check.png')
  }

  const escapedTitle = title.replaceAll("'", "''")
  const frontmatter = [
    '---',
    `title: '${escapedTitle}'`,
    `createTime: ${sourceCreateTime(sourceRelativePath)}`,
    `permalink: ${pagePermalink(targetRelativePath)}`,
    'pageClass: yokubi-page',
    '---',
    '',
  ].join('\n')

  return `${frontmatter}${content.trim()}\n`
}

mkdirSync(targetRoot, { recursive: true })

for (const [sourceRelativePath, targetRelativePath] of entries) {
  const sourcePath = path.join(sourceRoot, 'src', ...sourceRelativePath.split('/'))
  const targetPath = path.join(targetRoot, ...targetRelativePath.split('/'))
  const content = transformContent(
    sourceRelativePath,
    targetRelativePath,
    readFileSync(sourcePath, 'utf8'),
  )

  mkdirSync(path.dirname(targetPath), { recursive: true })
  writeFileSync(targetPath, content)
}

copyFileSync(path.join(sourceRoot, 'src', 'images', 'logo.svg'), path.join(targetRoot, 'logo.svg'))
copyFileSync(path.join(sourceRoot, 'LICENSE'), path.join(targetRoot, 'LICENSE'))

console.log(`Imported ${entries.length} Yokubi pages into ${targetRoot}`)
