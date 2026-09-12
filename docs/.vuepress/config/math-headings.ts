import type { Plugin } from 'vuepress/core'

export const mathHeadingsPlugin = (): Plugin => ({
  name: 'asabook:math-headings',
  extendsMarkdown(md) {
    const renderMath = md.renderer.rules.math_inline
    if (!renderMath) return

    const headingMath = new WeakSet()
    md.core.ruler.after('inline', 'asabook:math-headings', (state) => {
      for (let index = 1; index < state.tokens.length; index++) {
        const token = state.tokens[index]
        if (token.type !== 'inline' || state.tokens[index - 1].type !== 'heading_open') continue
        for (const child of token.children ?? []) {
          if (child.type === 'math_inline') headingMath.add(child)
        }
      }
    })

    md.renderer.rules.math_inline = (...args) => {
      const html = renderMath(...args)
      if (!headingMath.has(args[0][args[1]])) return html

      // Plume rc.210 reads heading textContent, including MathML and its TeX annotation.
      // Its ignore-header filter removes only the cloned branch used for the outline.
      return html.replace('class="katex-mathml"', 'class="katex-mathml ignore-header"')
    }
  },
})
