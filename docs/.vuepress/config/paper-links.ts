import type { Plugin } from 'vuepress/core'

export const paperLinksPlugin = (): Plugin => ({
  name: 'asabook:paper-links',
  extendsMarkdown(md) {
    // Plume rc.210 sanitizes tooltip markup but leaves literal quotes in aria-label.
    // Escape only that attribute, preserving the original tooltip renderer and entities.
    const renderAbbreviation = md.renderer.rules.abbreviation
    if (renderAbbreviation) {
      md.renderer.rules.abbreviation = (...args) => renderAbbreviation(...args).replace(
        /(<(?:VP)?Abbreviation aria-label=")([\s\S]*?)(">)/,
        (_match, opening: string, label: string, closing: string) =>
          opening + label.replaceAll('"', '&quot;') + closing,
      )
    }
    md.core.ruler.after('abbr_replace', 'asabook:localized-paper-links', (state) => {
      const file = String(state.env.filePathRelative ?? '').replaceAll('\\', '/')
      const prefix = file.startsWith('en/') ? '/en' : file.startsWith('ja/') ? '/ja' : ''
      if (!prefix) return
      for (const token of state.tokens) {
        for (const child of token.children ?? []) {
          if (child.type === 'abbreviation') {
            child.info = child.info.replace(/\[Link\]\(\/papers\/([^)]+)\)/g, `[Link](${prefix}/papers/$1)`)
          }
        }
      }
    })
  },
})
