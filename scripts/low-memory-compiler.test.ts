import assert from 'node:assert/strict'
import { test } from 'node:test'
import * as compiler from 'vue/compiler-sfc'
import katex from 'katex'
import { lowMemoryCompiler } from '../docs/.vuepress/low-memory-compiler.ts'

test('Markdown without cached AST produces identical client and SSR templates', () => {
  const math = katex.renderToString('\\sum_{i=1}^{n} i^2')
  const source = `<template><div><h2 id="math">Math</h2>${math}<img src="./figure.png"><a href="#math">Reference</a><Example :value="2"/><pre><code><span class="line">x = 1</span></code></pre></div></template><script>export default {}</script>`
  for (const ssr of [false, true]) {
    const filename = `paper-${ssr}.md`
    const original = compiler.parse(source, { filename })
    const options = {
      id: 'paper', filename, ssr, ssrCssVars: [], isProd: true,
      source: original.descriptor.template!.content,
    }
    const expected = compiler.compileTemplate({ ...options, ast: original.descriptor.template!.ast })
    const lean = lowMemoryCompiler.parse(source, { filename })
    assert.equal(lean.descriptor.template!.ast, undefined)
    const actual = compiler.compileTemplate({ ...options, ast: lean.descriptor.template!.ast })
    assert.deepEqual(actual.errors, [])
    assert.equal(actual.code, expected.code)
    // The compiler's own parse cache must not keep a second AST alive either.
    assert.equal(compiler.parse(source, { filename }).descriptor.template!.ast, undefined)
  }
})

test('Vue components and Markdown script-setup retain template binding analysis', () => {
  for (const filename of ['Component.vue', 'interactive.md']) {
    const source = `<template><div>{{ value }}</div></template><script setup>const value = 1</script>`
    const result = lowMemoryCompiler.parse(source, { filename })
    assert.ok(result.descriptor.template!.ast)
    assert.deepEqual(result.errors, [])
  }
  const result = lowMemoryCompiler.parse('<template><p>Component</p></template>', { filename: 'Plain.vue' })
  assert.ok(result.descriptor.template!.ast)
})

test('Markdown parse errors are preserved', () => {
  const source = '<template><div></template>'
  const options = { filename: 'invalid.md' }
  assert.deepEqual(lowMemoryCompiler.parse(source, options).errors, compiler.parse(source, options).errors)
  assert.ok(lowMemoryCompiler.parse(source, options).errors.length)
})
