import ts from 'typescript'

// Parse the declaration rather than relying on its neighbours or string escaping.
export function readPaperAbbreviations(source) {
  const file = ts.createSourceFile('papers.ts', source, ts.ScriptTarget.Latest, true)
  let object
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(file) === 'paperAbbreviations') {
      object = node.initializer
    }
    ts.forEachChild(node, visit)
  }
  visit(file)
  if (!object || !ts.isObjectLiteralExpression(object)) throw new Error('Missing paperAbbreviations object literal')
  const entries = object.properties.map((property) => {
    if (!ts.isPropertyAssignment(property) || !ts.isStringLiteral(property.initializer)) {
      throw new Error('Paper abbreviations must be literal string properties')
    }
    const comment = source.slice(property.end, source.indexOf('\n', property.end))
    return { key: property.name.text, value: property.initializer.text, date: comment.match(/published: (\d{4}(?:-\d{2}){0,2})/)?.[1] }
  })
  if (new Set(entries.map(({ key }) => key)).size !== entries.length) throw new Error('Duplicate paper abbreviation keys')
  return { entries, start: object.getStart(file), end: object.end }
}

export function paperDate({ key, value, date }) {
  if (date) return date
  // Prefer the bibliographic year; a legacy citation key supplies year precision only.
  const prose = value.replace(/https?:\/\/[^\s)]+/g, '')
    .replace(/(?:arXiv:|abs\/)\s*\S+/gi, '')
    .replace(/\b\d+\s*[-–—]+\s*\d+\b/g, '')
  const years = [...prose.matchAll(/(?<![\d.])\b((?:18|19|20)\d{2})[a-z]?\b(?!\d|\.\d)/g)].map((match) => match[1])
  const shortYear = key.match(/^[A-Za-z]+(\d{2})[a-z]*$/)?.[1]
  const citedYear = years.find((year) => year.endsWith(shortYear))
  if (citedYear) return citedYear
  if (years.length) return years.at(-1)
  return shortYear ? `${Number(shortYear) > 26 ? '19' : '20'}${shortYear}` : '9999'
}

export function writePaperAbbreviations(source, entries) {
  const { start, end } = readPaperAbbreviations(source)
  const quote = (value) => `'${value.replaceAll('\\', '\\\\').replaceAll("'", "\\'").replaceAll('\n', '\\n').replaceAll('\r', '\\r')}'`
  const sorted = [...entries].sort((a, b) => paperDate(a).localeCompare(paperDate(b)) || a.key.localeCompare(b.key, 'en'))
  const lines = sorted.map(({ key, value, date }) => `  ${quote(key)}: ${quote(value)},${date ? ` // published: ${date}` : ''}`)
  return `${source.slice(0, start)}{\n${lines.join('\n')}\n}${source.slice(end)}`
}
