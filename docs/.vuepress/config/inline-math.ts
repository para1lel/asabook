import type { RuleInline } from 'markdown-it/lib/parser_inline.mjs'
import type { Plugin } from 'vuepress/core'

const isWordCharacterOrNumber = (code: number): boolean =>
  (code >= 48 && code <= 57) ||
  (code >= 65 && code <= 90) ||
  (code >= 97 && code <= 122) ||
  code === 95

const hasOddBackslashes = (source: string, position: number): boolean => {
  let count = 0
  for (let index = position - 1; index >= 0 && source.charCodeAt(index) === 92; index--) count++
  return count % 2 === 1
}

const inlineMathWithinWord: RuleInline = (state, silent) => {
  const position = state.pos
  if (state.src.charCodeAt(position) !== 36) return false

  const previous = state.src.charCodeAt(position - 1)
  const next = state.src.charCodeAt(position + 1)
  if (
    previous === 36 ||
    next === 36 ||
    !isWordCharacterOrNumber(previous) ||
    state.md.utils.isSpace(next)
  ) {
    return false
  }

  let closing = position + 1
  while ((closing = state.src.indexOf('$', closing)) !== -1) {
    if (!hasOddBackslashes(state.src, closing)) break
    closing++
  }
  if (closing === -1 || closing === position + 1) return false

  const contentBeforeClosing = state.src.charCodeAt(closing - 1)
  if (state.md.utils.isSpace(contentBeforeClosing)) return false

  if (!silent) {
    const token = state.push('math_inline', 'math', 0)
    token.markup = '$'
    token.content = state.src.slice(position + 1, closing)
  }
  state.pos = closing + 1
  return true
}

export const inlineMathPlugin = (): Plugin => ({
  name: 'asabook:inline-math',
  extendsMarkdown(md) {
    md.inline.ruler.before('escape', 'asabook:inline-math', inlineMathWithinWord)
  },
})
