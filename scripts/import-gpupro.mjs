import { cp, mkdir, readFile, readdir, writeFile } from 'node:fs/promises'
import path from 'node:path'
import process from 'node:process'
import { execFileSync } from 'node:child_process'

const repoRoot = path.resolve(import.meta.dirname, '..')
const checkoutRoot = path.resolve(process.argv[2] || '')
const localeFlag = process.argv.find((argument) => argument.startsWith('--locale='))
const locale = localeFlag?.split('=')[1] || 'en'
const localeConfig = {
  en: {
    sourceDirectory: '',
    outputDirectory: path.join('en', 'gpupro'),
    routePrefix: '/en/gpupro',
    imagePath: '../../gpupro/images/',
  },
  zh: {
    sourceDirectory: 'zh',
    outputDirectory: 'gpupro',
    routePrefix: '/gpupro',
    imagePath: './images/',
  },
}[locale]

if (!localeConfig) {
  throw new Error(`Unsupported locale: ${locale}`)
}

const sourceRoot = path.join(checkoutRoot, localeConfig.sourceDirectory)
const outputRoot = path.join(repoRoot, 'docs', localeConfig.outputDirectory)
const imageRoot = path.join(repoRoot, 'docs', 'gpupro', 'images')
const publicRoot = path.join(repoRoot, 'docs', '.vuepress', 'public', 'gpupro')
const importedAt = '2026/08/01 00:00:00'

if (!process.argv[2]) {
  throw new Error('Usage: node scripts/import-gpupro.mjs <upstream-checkout> [--locale=en|zh]')
}

const pages = [
  { source: 'index.md', output: 'README.md', slug: '', title: { en: 'Modern GPU Programming For MLSys', zh: '面向机器学习系统的现代 GPU 编程' }, format: 'md' },
  { source: 'chapter_background/index.md', output: 'gpu-execution-model.md', slug: 'gpu-execution-model', title: { en: 'GPU Execution Model', zh: 'GPU 执行模型' }, format: 'md' },
  { source: 'chapter_performance/index.md', output: 'kernel-performance.md', slug: 'kernel-performance', title: { en: 'What Makes a Kernel Fast', zh: '高性能 Kernel 的关键' }, format: 'md' },
  { source: 'chapter_data_layout/index.md', output: 'data-layout.md', slug: 'data-layout', title: { en: 'Data Layout and Its Notation', zh: '数据布局及其表示方法' }, format: 'md' },
  { source: 'chapter_layout_generations/index.md', output: 'tensor-core-data-layouts.md', slug: 'tensor-core-data-layouts', title: { en: 'The Evolution of Tensor Core Data Layouts', zh: 'Tensor Core 数据布局的演进' }, format: 'md' },
  { source: 'chapter_tma/index.md', output: 'tma.md', slug: 'tma', title: { en: 'Async Data Movement: TMA', zh: '异步数据搬运: TMA' }, format: 'md' },
  { source: 'chapter_tensor_cores/index.md', output: 'blackwell-tensor-core.md', slug: 'blackwell-tensor-core', title: { en: 'Blackwell Tensor Core: tcgen05.mma', zh: 'Blackwell Tensor Core: tcgen05.mma' }, format: 'md' },
  { source: 'chapter_tmem/index.md', output: 'tensor-memory.md', slug: 'tensor-memory', title: { en: 'Tensor Memory (TMEM)', zh: 'Tensor Memory (TMEM)' }, format: 'md' },
  { source: 'chapter_async_barriers/index.md', output: 'mbarrier.md', slug: 'mbarrier', title: { en: 'Async Coordination: mbarrier', zh: '异步协作: mbarrier' }, format: 'md' },
  { source: 'chapter_clc/index.md', output: 'cluster-launch-control.md', slug: 'cluster-launch-control', title: { en: 'Advanced Scheduling: Cluster Launch Control', zh: '高级调度: Cluster Launch Control' }, format: 'md' },
  { source: 'chapter_intro_tirx/index.md', output: 'tirx-introduction.md', slug: 'tirx-introduction', title: { en: 'Introduction to TIRx', zh: 'TIRx 简介' }, format: 'md' },
  { source: 'chapter_tirx_layout_api/index.md', output: 'tirx-layout-api.md', slug: 'tirx-layout-api', title: { en: 'TIRx Layout API', zh: 'TIRx Layout API' }, format: 'md' },
  { source: 'chapter_gemm_basics/index.md', output: 'tiled-gemm.md', slug: 'tiled-gemm', title: { en: 'Building a Tiled GEMM', zh: '构建 Tiled GEMM' }, format: 'md' },
  { source: 'chapter_gemm_async/index.md', output: 'pipelined-gemm.md', slug: 'pipelined-gemm', title: { en: 'Pipelining GEMM with TMA', zh: '使用 TMA 流水线化 GEMM' }, format: 'md' },
  { source: 'chapter_gemm_advanced/index.md', output: 'warp-specialized-gemm.md', slug: 'warp-specialized-gemm', title: { en: 'Scaling GEMM with Warp Specialization and Clusters', zh: '使用 Warp Specialization 和 Cluster 扩展 GEMM' }, format: 'md' },
  { source: 'chapter_flash_attention/index.md', output: 'flash-attention-4.md', slug: 'flash-attention-4', title: { en: 'Flash Attention 4', zh: 'Flash Attention 4' }, format: 'md' },
  { source: 'appendix/index.md', output: 'reference.md', slug: 'reference', title: { en: 'Reference', zh: '参考资料' }, format: 'md' },
  { source: 'appendix/debugging_warp_specialized.md', output: 'debugging-warp-specialized-kernels.md', slug: 'debugging-warp-specialized-kernels', title: { en: 'Debugging Warp-Specialized Kernels', zh: '调试 Warp-Specialized Kernel' }, format: 'md' },
  { source: 'tirx_guide/language_reference/index.rst', output: 'tirx-language-reference.md', slug: 'tirx-language-reference', title: { en: 'TIRx Language Reference', zh: 'TIRx 语言参考' }, format: 'rst' },
  { source: 'tirx_guide/language_reference/cuda/parser_utils.rst', output: 'parser-utilities.md', slug: 'parser-utilities', title: { en: 'Parser Utilities', zh: '解析器工具' }, format: 'rst' },
  { source: 'tirx_guide/language_reference/cuda/data_types.rst', output: 'data-types-and-expressions.md', slug: 'data-types-and-expressions', title: { en: 'Data Types and Expressions', zh: '数据类型与表达式' }, format: 'rst' },
  { source: 'tirx_guide/language_reference/cuda/buffers.rst', output: 'buffers-and-memory.md', slug: 'buffers-and-memory', title: { en: 'Buffers and Memory', zh: 'Buffer 与内存' }, format: 'rst' },
  { source: 'tirx_guide/language_reference/cuda/control_flow.rst', output: 'control-flow.md', slug: 'control-flow', title: { en: 'Control Flow', zh: '控制流' }, format: 'rst' },
  { source: 'tirx_guide/language_reference/cuda/threads_sync.rst', output: 'cuda-ptx-intrinsics.md', slug: 'cuda-ptx-intrinsics', title: { en: 'CUDA C++/PTX Intrinsics', zh: 'CUDA C++/PTX Intrinsic' }, format: 'rst' },
  { source: 'tirx_guide/arch/index.rst', output: 'compiler-internals.md', slug: 'compiler-internals', title: { en: 'Compiler Internals', zh: '编译器内部机制' }, format: 'rst' },
  { source: 'tirx_guide/arch/lowering_pipeline.rst', output: 'tirx-lowering-pipeline.md', slug: 'tirx-lowering-pipeline', title: { en: 'TIRx Lowering Pipeline', zh: 'TIRx Lowering 流水线' }, format: 'rst' },
]

for (const page of pages) page.title = page.title[locale]

const pageBySource = new Map(pages.map((page) => [page.source.replaceAll('\\', '/'), page]))
const routeFor = (page) => `${localeConfig.routePrefix}/${page.slug ? `${page.slug}/` : ''}`

function normalize(text) {
  return text.replace(/^\uFEFF/, '').replaceAll('\r\n', '\n').replaceAll('\r', '\n')
}

function scanPythonLine(line, context) {
  let escaped = false

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index]

    if (context.quote) {
      if (context.triple && line.startsWith(context.quote.repeat(3), index)) {
        context.quote = ''
        context.triple = false
        index += 2
      } else if (!context.triple && escaped) {
        escaped = false
      } else if (!context.triple && character === '\\') {
        escaped = true
      } else if (!context.triple && character === context.quote) {
        context.quote = ''
      }
      continue
    }

    if (character === '#') break
    if (character === '"' || character === "'") {
      context.quote = character
      context.triple = line.startsWith(character.repeat(3), index)
      if (context.triple) index += 2
    } else if ('([{'.includes(character)) {
      context.depth += 1
    } else if (')]}'.includes(character)) {
      context.depth = Math.max(0, context.depth - 1)
    }
  }
}

function leadingClosingDepth(content) {
  let depth = 0
  while (depth < content.length && ')]}'.includes(content[depth])) depth += 1
  return depth
}

function structuralIndent(indentation, sourceIndentWidth) {
  return Math.floor(indentation / sourceIndentWidth) * 2 + indentation % sourceIndentWidth
}

function reindentPythonBlock(code, sourceIndentWidth = 4) {
  const context = { depth: 0, quote: '', triple: false }
  let statementIndent = 0
  let outputStatementIndent = 0

  return code.split('\n').map((line) => {
    if (!line.trim()) return ''

    const indentation = line.match(/^ */)[0].length
    const content = line.slice(indentation)
    const inMultilineString = context.triple

    if (context.depth === 0 && !context.quote) {
      statementIndent = indentation
      outputStatementIndent = structuralIndent(indentation, sourceIndentWidth)
    }

    let outputIndent
    if (inMultilineString) {
      outputIndent = Math.max(0, indentation - (statementIndent - outputStatementIndent))
    } else if (context.depth > 0) {
      const closingDepth = Math.min(context.depth, leadingClosingDepth(content))
      outputIndent = outputStatementIndent + (context.depth - closingDepth) * 2
    } else {
      outputIndent = structuralIndent(indentation, sourceIndentWidth)
    }

    const output = `${' '.repeat(outputIndent)}${content}`
    scanPythonLine(line, context)
    return output
  }).join('\n')
}

function normalizeCodeIndentation(text, sourceIndentWidth = 4) {
  return text.replace(/```python\n([\s\S]*?)\n```/g, (_, code) => {
    return `\`\`\`python\n${reindentPythonBlock(code, sourceIndentWidth)}\n\`\`\``
  })
}

function normalizeImageSyntax(text) {
  let fence = ''

  return text.split('\n').map((line) => {
    const fenceMatch = line.match(/^ {0,3}(`{3,}|~{3,})/)
    if (fenceMatch) {
      const marker = fenceMatch[1][0]
      if (!fence) fence = marker
      else if (fence === marker) fence = ''
      return line
    }
    return fence ? line : line.replace(/!\s+\[/g, '![')
  }).join('\n')
}

function slugify(text) {
  return text
    .replace(/<[^>]+>/g, '')
    .replace(/[`*_~]/g, '')
    .replace(/&[a-z]+;/gi, '')
    .toLowerCase()
    .replace(/[^\p{Letter}\p{Number}\s-]/gu, '')
    .trim()
    .replace(/[\s-]+/g, '-')
}

function headingAfter(lines, index, format) {
  for (let cursor = index + 1; cursor < lines.length; cursor += 1) {
    const line = lines[cursor]
    if (!line.trim()) continue
    if (format === 'md') {
      const match = line.match(/^#{1,6}\s+(.+)$/)
      return match ? match[1].trim() : ''
    }
    if (cursor + 1 < lines.length && /^[=\-~^]+$/.test(lines[cursor + 1].trim())) {
      return line.trim()
    }
    return ''
  }
  return ''
}

function buildLabelMap(sources) {
  const labels = new Map()

  for (const page of pages) {
    const lines = sources.get(page.source).split('\n')
    const labelPattern = page.format === 'md' ? /^\(([^)]+)\)=\s*$/ : /^\.\. _([^:]+):\s*$/

    lines.forEach((line, index) => {
      const match = line.match(labelPattern)
      if (!match) return
      const heading = headingAfter(lines, index, page.format)
      const anchor = heading && heading !== page.title && index > 2 ? `#${slugify(heading)}` : ''
      labels.set(match[1], {
        text: heading || page.title,
        href: `${routeFor(page)}${anchor}`,
      })
    })
  }

  return labels
}

function replaceReferenceRoles(text, labels, currentPage, docMap = new Map()) {
  const replaceRef = (_, value) => {
    const explicit = value.match(/^(.+?)\s*<([^>]+)>$/)
    const target = explicit ? explicit[2] : value
    const resolved = labels.get(target)
    if (!resolved) return explicit ? explicit[1] : value
    return `[${explicit ? explicit[1] : resolved.text}](${resolved.href})`
  }

  text = text.replace(/\{ref\}`([^`]+)`/g, replaceRef)
  text = text.replace(/:ref:`([^`]+)`/g, replaceRef)
  text = text.replace(/:doc:`([^`]+)`/g, (_, value) => {
    const explicit = value.match(/^(.+?)\s*<([^>]+)>$/)
    const target = explicit ? explicit[2] : value
    const resolved = docMap.get(target) || docMap.get(path.posix.join(path.posix.dirname(currentPage.source), target))
    return resolved ? `[${explicit ? explicit[1] : resolved.title}](${routeFor(resolved)})` : (explicit ? explicit[1] : value)
  })

  return text
}

function replaceRawHtml(text) {
  text = text.replace(/```\{raw\} html\n([\s\S]*?)\n```/g, (_, html) => `${html}\n`)
  text = text.replace(/<script>[\s\S]*?<\/script>/g, '')
  text = text.replaceAll('../demo/', '/gpupro/demo/')
  text = text.replaceAll('../demo_zh/', '/gpupro/demo-zh/')
  text = text.replaceAll('../_static/tirx-layout-demo/', '/gpupro/tirx-layout-demo/')
  text = text.replace(/var\(--pst-color-border,\s*#[0-9a-f]+\)/gi, 'var(--vp-c-border)')
  return text
}

function replaceMystImages(text) {
  return text.replace(/```\{image\}\s+([^\n]+)\n([\s\S]*?)\n```/g, (_, source, options) => {
    const alt = options.match(/^:alt:\s*(.+)$/m)?.[1] || ''
    const width = options.match(/^:width:\s*(.+)$/m)?.[1]
    const localSource = source.trim().replace(/^(?:\.\.\/)+img\//, localeConfig.imagePath)
    const style = width ? ` style="width:${width};"` : ''
    return `<img src="${localSource}" alt="${alt}"${style}>`
  })
}

function replaceAdmonitions(text) {
  return text.replace(/:::\{admonition\}\s+([^\n]+)\n(?::class:[^\n]*\n)?\n?([\s\S]*?)\n:::/g, (_, title, body) => {
    return `::: info ${title.trim()}\n${body.trim()}\n:::`
  })
}

function removeToctrees(text) {
  return text.replace(/```\{toctree\}\n[\s\S]*?\n```\n?/g, '')
}

function stripMarkdownTitle(text) {
  text = text.replace(/^\([^\n]+\)=\n/, '')
  return text.replace(/^#\s+[^\n]+\n+/, '')
}

function convertMarkdown(text, page, labels) {
  text = stripMarkdownTitle(text)
  text = text.replace(/^\(([^)]+)\)=\s*\n/gm, '')
  text = replaceAdmonitions(text)
  text = replaceRawHtml(text)
  text = replaceMystImages(text)
  text = removeToctrees(text)
  text = replaceReferenceRoles(text, labels, page)
  text = text.replace(/(?:\.\.\/)+img\//g, localeConfig.imagePath)

  for (const [source, linkedPage] of pageBySource) {
    const htmlSource = source.replace(/\.(md|rst)$/, '.html')
    text = text.replaceAll(`../${htmlSource}`, routeFor(linkedPage))
  }

  return text.trim()
}

function directiveBlock(lines, start) {
  const baseIndent = lines[start].match(/^\s*/)[0].length
  let cursor = start + 1
  const block = []

  while (cursor < lines.length) {
    const line = lines[cursor]
    const indent = line.match(/^\s*/)[0].length
    if (line.trim() && indent <= baseIndent) break
    block.push(line)
    cursor += 1
  }

  return { baseIndent, block, end: cursor }
}

function directiveContent(lines) {
  const content = [...lines]
  while (!content[0]?.trim()) content.shift()
  while (content[0]?.trim().match(/^:[^:]+:/)) {
    content.shift()
    while (!content[0]?.trim()) content.shift()
  }
  while (!content.at(-1)?.trim()) content.pop()

  const indentation = Math.min(
    ...content.filter((line) => line.trim()).map((line) => line.match(/^\s*/)[0].length),
  )
  return content.map((line) => line.slice(Math.min(indentation, line.length)))
}

function convertListTable(lines) {
  const rows = []
  let row = null

  for (const line of lines) {
    const firstCell = line.match(/^\*\s+-\s+(.*)$/)
    const nextCell = line.match(/^\s+-\s+(.*)$/)
    if (firstCell) {
      if (row) rows.push(row)
      row = [firstCell[1]]
    } else if (nextCell && row) {
      row.push(nextCell[1])
    } else if (line.trim() && row?.length) {
      row[row.length - 1] += ` ${line.trim()}`
    }
  }
  if (row) rows.push(row)
  if (!rows.length) return ''

  const width = Math.max(...rows.map((item) => item.length))
  const normalizedRows = rows.map((item) => [...item, ...Array(width - item.length).fill('')])
  const renderRow = (item) => `| ${item.map((cell) => cell.replaceAll('|', '\\|')).join(' | ')} |`
  return [renderRow(normalizedRows[0]), renderRow(Array(width).fill('---')), ...normalizedRows.slice(1).map(renderRow)].join('\n')
}

function convertRstBlocks(text) {
  const lines = text.split('\n')
  const output = []

  for (let index = 0; index < lines.length;) {
    const line = lines[index]
    const codeMatch = line.match(/^\s*\.\. code-block::\s*(\S*)/)
    const tableMatch = line.match(/^\s*\.\. list-table::/)
    const noteMatch = line.match(/^\s*\.\. (note|warning|tip)::\s*(.*)$/)
    const toctreeMatch = line.match(/^\s*\.\. toctree::/)

    if (codeMatch) {
      const { block, end } = directiveBlock(lines, index)
      const content = directiveContent(block)
      output.push(`\`\`\`${codeMatch[1]}`, ...content, '\`\`\`', '')
      index = end
      continue
    }

    if (tableMatch) {
      const { block, end } = directiveBlock(lines, index)
      const content = directiveContent(block)
      output.push(convertListTable(content), '')
      index = end
      continue
    }

    if (noteMatch) {
      const { block, end } = directiveBlock(lines, index)
      const content = directiveContent(block)
      const body = [noteMatch[2], ...content].filter(Boolean).join('\n')
      output.push(`::: ${noteMatch[1]}`, convertRstBlocks(body).trim(), ':::', '')
      index = end
      continue
    }

    if (toctreeMatch) {
      index = directiveBlock(lines, index).end
      continue
    }

    output.push(line)
    index += 1
  }

  return output.join('\n')
}

function convertRstInline(text, labels, page, docMap) {
  text = replaceReferenceRoles(text, labels, page, docMap)
  text = text.replace(/(?<!`)`([^`<\n]+?)\s*<([^>\n]+)>`_(?!`)/g, '[$1]($2)')
  text = text.replace(/(?<!`)`([^`\n]+)`_(?!`)/g, (_, label) => `[${label}](#${slugify(label)})`)
  text = text.replace(/(?<!`)``([^`\n]+)``(?!`)/g, '`$1`')
  return text
}

function stripRstLicense(text) {
  const lines = text.split('\n')
  const end = lines.findIndex((line) => line.trim() === 'under the License.')
  if (end === -1 || !lines.slice(0, end + 1).some((line) => line.includes('Apache Software Foundation'))) return text
  const license = lines.slice(0, end + 1).map((line) => line.replace(/^\.\.\s?/, '').trim()).filter(Boolean)
  return `<!--\n${license.join('\n')}\n-->\n\n${lines.slice(end + 1).join('\n').trimStart()}`
}

function convertRstHeadings(text) {
  const lines = text.split('\n')
  const output = []
  const levels = { '=': '#', '-': '##', '~': '###', '^': '####' }

  for (let index = 0; index < lines.length; index += 1) {
    const underline = lines[index + 1]?.trim()
    if (lines[index].trim() && /^([=\-~^])\1{2,}$/.test(underline)) {
      output.push(`${levels[underline[0]]} ${lines[index].trim()}`)
      index += 1
    } else {
      output.push(lines[index])
    }
  }

  return output.join('\n')
}

function convertRst(text, page, labels, docMap) {
  text = stripRstLicense(text)
  text = text.replace(/^\.\. _[^:]+:\s*\n/gm, '')
  text = convertRstBlocks(text)
  text = convertRstHeadings(text)
  text = text.replace(/^#\s+[^\n]+\n+/m, '')
  text = convertRstInline(text, labels, page, docMap)
  return text.trim()
}

function frontmatter(page) {
  return [
    '---',
    `title: ${JSON.stringify(page.title)}`,
    `createTime: ${importedAt}`,
    `permalink: ${routeFor(page)}`,
    'pageClass: gpupro-page',
    '---',
  ].join('\n')
}

function homeIntroduction(commit) {
  if (locale === 'zh') {
    return `::: note 来源\n本地版本镜像自 [MLC Community 原书](https://mlc.ai/modern-gpu-programming-for-mlsys/)的[上游提交 \`${commit.slice(0, 12)}\`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/${commit}). Copyright 2026 MLC Community. 正文和资源以上游中文稿为基础, 本地版本调整了 VuePress 所需的格式和链接, 并校订中文表达. 上游仓库目前没有覆盖全仓库的许可证文件; TIRx 参考页面保留 Apache License 2.0 声明.\n:::`
  }

  return `::: note Source\nThis local edition mirrors the [MLC Community book](https://mlc.ai/modern-gpu-programming-for-mlsys/) from [upstream commit \`${commit.slice(0, 12)}\`](https://github.com/mlc-ai/modern-gpu-programming-for-mlsys/tree/${commit}). Copyright 2026 MLC Community. The text and assets are unchanged except for the formatting and links required by VuePress. The upstream repository currently has no repository-wide license file; the TIRx reference pages retain their Apache License 2.0 notices.\n:::`
}

const homeContents = locale === 'zh' ? `## 目录

### 第一部分: 理解 GPU

- [GPU 执行模型](./gpu-execution-model/)
- [高性能 Kernel 的关键](./kernel-performance/)
- [数据布局及其表示方法](./data-layout/)
- [Tensor Core 数据布局的演进](./tensor-core-data-layouts/)
- [异步数据搬运: TMA](./tma/)
- [Blackwell Tensor Core: \`tcgen05.mma\`](./blackwell-tensor-core/)
- [Tensor Memory (TMEM)](./tensor-memory/)
- [异步协作: mbarrier](./mbarrier/)
- [高级调度: Cluster Launch Control](./cluster-launch-control/)

### 第二部分: TIRx 概览

- [TIRx 简介](./tirx-introduction/)
- [TIRx Layout API](./tirx-layout-api/)

### 第三部分: GEMM, 从 Tiled 到 SOTA

- [构建 Tiled GEMM](./tiled-gemm/)
- [使用 TMA 流水线化 GEMM](./pipelined-gemm/)
- [使用 Warp Specialization 和 Cluster 扩展 GEMM](./warp-specialized-gemm/)

### 第四部分: Flash Attention 4

- [Flash Attention 4](./flash-attention-4/)

### 参考资料

- [参考资料概览](./reference/)
- [TIRx 语言参考](./tirx-language-reference/)
- [调试 Warp-Specialized Kernel](./debugging-warp-specialized-kernels/)
- [编译器内部机制](./compiler-internals/)` : `## Contents

### Part I: Understanding the GPU

- [GPU Execution Model](./gpu-execution-model/)
- [What Makes a Kernel Fast](./kernel-performance/)
- [Data Layout and Its Notation](./data-layout/)
- [The Evolution of Tensor Core Data Layouts](./tensor-core-data-layouts/)
- [Async Data Movement: TMA](./tma/)
- [Blackwell Tensor Core: \`tcgen05.mma\`](./blackwell-tensor-core/)
- [Tensor Memory (TMEM)](./tensor-memory/)
- [Async Coordination: mbarrier](./mbarrier/)
- [Advanced Scheduling: Cluster Launch Control](./cluster-launch-control/)

### Part II: TIRx Overview

- [Introduction to TIRx](./tirx-introduction/)
- [TIRx Layout API](./tirx-layout-api/)

### Part III: GEMM, Tiled to SOTA

- [Building a Tiled GEMM](./tiled-gemm/)
- [Pipelining GEMM with TMA](./pipelined-gemm/)
- [Scaling GEMM with Warp Specialization and Clusters](./warp-specialized-gemm/)

### Part IV: Flash Attention 4

- [Flash Attention 4](./flash-attention-4/)

### Reference

- [Reference Overview](./reference/)
- [TIRx Language Reference](./tirx-language-reference/)
- [Debugging Warp-Specialized Kernels](./debugging-warp-specialized-kernels/)
- [Compiler Internals](./compiler-internals/)`

const languageReferenceContents = locale === 'zh' ? `## 语言主题

- [解析器工具](./parser-utilities/)
- [数据类型与表达式](./data-types-and-expressions/)
- [Buffer 与内存](./buffers-and-memory/)
- [控制流](./control-flow/)
- [CUDA C++/PTX Intrinsic](./cuda-ptx-intrinsics/)` : `## Language Topics

- [Parser Utilities](./parser-utilities/)
- [Data Types and Expressions](./data-types-and-expressions/)
- [Buffers and Memory](./buffers-and-memory/)
- [Control Flow](./control-flow/)
- [CUDA C++/PTX Intrinsics](./cuda-ptx-intrinsics/)`

const compilerContents = locale === 'zh' ? `## 编译器主题

- [TIRx Lowering 流水线](./tirx-lowering-pipeline/)` : `## Compiler Topics

- [TIRx Lowering Pipeline](./tirx-lowering-pipeline/)`

async function copyImages() {
  const sourceImages = path.join(checkoutRoot, 'img')
  await mkdir(imageRoot, { recursive: true })

  for (const entry of await readdir(sourceImages, { withFileTypes: true })) {
    if (!entry.isFile() || !['.png', '.svg'].includes(path.extname(entry.name))) continue
    if (locale === 'en' && entry.name.includes('_zh')) continue
    await cp(path.join(sourceImages, entry.name), path.join(imageRoot, entry.name))
  }
}

async function copyInteractiveAssets() {
  await mkdir(publicRoot, { recursive: true })
  const demoSource = locale === 'zh'
    ? path.join(sourceRoot, '_extra', 'demo_zh')
    : path.join(checkoutRoot, '_extra', 'demo')
  const demoTarget = locale === 'zh' ? 'demo-zh' : 'demo'
  await cp(demoSource, path.join(publicRoot, demoTarget), { recursive: true, force: true })
  await cp(path.join(checkoutRoot, 'static', 'tirx-layout-demo'), path.join(publicRoot, 'tirx-layout-demo'), { recursive: true, force: true })

  for (const file of ['viz-base.css', 'viz-base.js']) {
    await cp(path.join(checkoutRoot, '_extra', file), path.join(publicRoot, file))
  }

  for (const file of ['code-highlight.css', 'code-highlight.js']) {
    try {
      await cp(path.join(checkoutRoot, '_extra', file), path.join(publicRoot, file))
    } catch (error) {
      if (error.code !== 'ENOENT') throw error
    }
  }

  const sourceFont = path.join(repoRoot, 'docs', '.vuepress', 'assets', 'fonts', 'LHANDW.TTF')
  await mkdir(path.join(publicRoot, 'fonts'), { recursive: true })
  await cp(sourceFont, path.join(publicRoot, 'fonts', 'LHANDW.TTF'))

  const fontCss = `@font-face {\n  font-family: "Lucida Handwriting";\n  src: url("/gpupro/fonts/LHANDW.TTF") format("truetype");\n  font-display: swap;\n}\n\nhtml,\nhtml *,\nhtml *::before,\nhtml *::after {\n  font-family: "Lucida Handwriting", sans-serif !important;\n}\n`
  await writeFile(path.join(publicRoot, 'gpupro-font.css'), fontCss, 'utf8')

  const htmlRoots = [path.join(publicRoot, demoTarget), path.join(publicRoot, 'tirx-layout-demo')]
  for (const htmlRoot of htmlRoots) {
    for (const entry of await readdir(htmlRoot, { withFileTypes: true })) {
      if (!entry.isFile() || path.extname(entry.name) !== '.html') continue
      const htmlPath = path.join(htmlRoot, entry.name)
      let html = normalize(await readFile(htmlPath, 'utf8'))
      html = html.replace('</head>', '  <link rel="stylesheet" href="/gpupro/gpupro-font.css">\n</head>')
      await writeFile(htmlPath, html, 'utf8')
    }
  }
}

function normalizeChineseSegment(segment) {
  segment = segment
    .replaceAll('，', ',')
    .replaceAll('。', '.')
    .replaceAll('；', ';')
    .replaceAll('：', ':')
    .replaceAll('！', '!')
    .replaceAll('？', '?')
    .replaceAll('、', ',')
    .replaceAll('（', '(')
    .replaceAll('）', ')')

  let spaced = ''
  for (let index = 0; index < segment.length; index += 1) {
    const character = segment[index]
    spaced += character
    if (!',.;:!?'.includes(character)) continue

    const previous = segment[index - 1] || ''
    const markerMatch = segment.slice(index + 1).match(/^(\*\*|__|~~|[*_~])+/)?.[0] || ''
    const afterMarkers = segment[index + 1 + markerMatch.length] || ''
    const markersAreClosing = markerMatch && (!afterMarkers || /\s/.test(afterMarkers))
    const next = markersAreClosing ? afterMarkers : (segment[index + 1] || '')

    if (/\d/.test(previous) && /\d/.test(next) && '.,:'.includes(character)) continue
    if (!next || /[\s\])}〉》」』”’]/.test(next)) continue
    if (markersAreClosing) {
      spaced += markerMatch
      index += markerMatch.length
    }
    spaced += ' '
  }
  segment = spaced

  segment = segment.replace(/([^\s([{<])\(/g, '$1 (')
  segment = segment.replace(/\)(?=[\p{Letter}\p{Number}])/gu, ') ')
  return segment
}

function normalizeChinesePunctuation(text) {
  const protectedInline = /(`+[^`\n]*?`+|\$+[^$\n]*?\$+|<[^>]+>|\]\([^\n)]+\))/g
  let inFence = false
  let inComment = false
  let htmlTag = ''

  return text.split('\n').map((line) => {
    if (line.startsWith('```')) {
      inFence = !inFence
      return line
    }
    if (inFence) return line
    if (line.includes('<!--')) inComment = true
    if (inComment) {
      if (line.includes('-->')) inComment = false
      return line
    }
    if (htmlTag) {
      if (line.toLowerCase().includes(`</${htmlTag}>`)) htmlTag = ''
      return line
    }
    const htmlStart = line.trim().match(/^<(div|iframe|script|style|p)\b/i)?.[1]?.toLowerCase()
    if (htmlStart) {
      if (!line.toLowerCase().includes(`</${htmlStart}>`)) htmlTag = htmlStart
      return line
    }
    const directive = line.match(/^(:::\s+\w+)(?:\s+(.+))?$/)
    if (directive) return directive[2] ? `${directive[1]} ${normalizeChineseSegment(directive[2])}` : line
    if (line.trim() === ':::') return line

    let cursor = 0
    let output = ''
    for (const match of line.matchAll(protectedInline)) {
      output += normalizeChineseSegment(line.slice(cursor, match.index))
      output += match[0]
      cursor = match.index + match[0].length
    }
    return output + normalizeChineseSegment(line.slice(cursor))
  }).join('\n')
}

async function main() {
  const sources = new Map()
  for (const page of pages) {
    sources.set(page.source, normalize(await readFile(path.join(sourceRoot, page.source), 'utf8')))
  }

  const commit = execFileSync('git', ['-C', sourceRoot, 'rev-parse', 'HEAD'], { encoding: 'utf8' }).trim()
  const labels = buildLabelMap(sources)
  const docMap = new Map()
  for (const page of pages) {
    const source = page.source.replace(/\.(md|rst)$/, '')
    docMap.set(source, page)
    docMap.set(path.posix.basename(source), page)
  }

  await mkdir(outputRoot, { recursive: true })
  for (const page of pages) {
    let body = page.format === 'md'
      ? convertMarkdown(sources.get(page.source), page, labels)
      : convertRst(sources.get(page.source), page, labels, docMap)

    if (!page.slug) {
      body = `${homeIntroduction(commit)}\n\n${body}\n\n${homeContents}`
    } else if (page.slug === 'tirx-language-reference') {
      body = `${body}\n\n${languageReferenceContents}`
    } else if (page.slug === 'compiler-internals') {
      body = `${body}\n\n${compilerContents}`
    }

    body = normalizeCodeIndentation(body)
    if (locale === 'zh') body = normalizeChinesePunctuation(body)
    body = normalizeImageSyntax(body)
    await writeFile(path.join(outputRoot, page.output), `${frontmatter(page)}\n\n${body.trim()}\n`, 'utf8')
  }

  await copyImages()
  await copyInteractiveAssets()
  console.log(`Imported ${pages.length} ${locale} pages from ${commit}`)
}

async function normalizeExistingPages() {
  const directories = [
    path.join(repoRoot, 'docs', 'gpupro'),
    path.join(repoRoot, 'docs', 'en', 'gpupro'),
    path.join(repoRoot, 'docs', 'ja', 'gpupro'),
  ]
  let fileCount = 0

  for (const directory of directories) {
    for (const entry of await readdir(directory, { withFileTypes: true })) {
      if (!entry.isFile() || path.extname(entry.name) !== '.md') continue
      const file = path.join(directory, entry.name)
      const source = await readFile(file, 'utf8')
      const output = normalizeImageSyntax(normalizeCodeIndentation(source, 2))
      if (output === source) continue
      await writeFile(file, output, 'utf8')
      fileCount += 1
    }
  }

  console.log(`Normalized Markdown in ${fileCount} gpupro pages`)
}

if (process.argv.includes('--normalize-existing')) {
  await normalizeExistingPages()
} else {
  await main()
}
