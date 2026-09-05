import * as compiler from 'vue/compiler-sfc'

// plugin-vue keeps SFC descriptors for the entire build. On long Markdown
// pages, retaining their template ASTs costs far more than reparsing them.
// Keep script-setup ASTs: Vue uses them to analyze template bindings.
export const lowMemoryCompiler: typeof compiler = {
  ...compiler,
  parse(source, options) {
    const result = compiler.parse(source, options)
    const { descriptor } = result

    if (options?.filename?.endsWith('.md') && !descriptor.scriptSetup && descriptor.template) {
      descriptor.template.ast = undefined
    }

    return result
  },
}
