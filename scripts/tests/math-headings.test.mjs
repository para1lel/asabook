import assert from 'node:assert/strict'
import { test } from 'node:test'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { createDevApp } from 'vuepress/core'
import { viteBundler } from '@vuepress/bundler-vite'
import { plumeTheme } from 'vuepress-theme-plume'
import { mathHeadingsPlugin } from '../../docs/.vuepress/config/math-headings.ts'

test('Plume heading formulas exclude duplicate MathML only from outline extraction', async (context) => {
  const prefix = join(tmpdir(), 'asabook-math-headings-test-')
  const source = mkdtempSync(prefix)
  context.after(() => {
    assert.ok(source.startsWith(prefix))
    rmSync(source, { recursive: true, force: true })
  })
  const app = createDevApp({
    source,
    bundler: viteBundler(),
    theme: plumeTheme({ cache: false }),
    plugins: [mathHeadingsPlugin()],
  })
  await app.init()
  const render = (markdown) => app.markdown.render(markdown, { filePathRelative: 'example.md' })

  for (const suffix of ['时定理 3.2 的证明', 'in the proof of Theorem 3.2', 'のときの定理 3.2 の証明']) {
    for (const k of [1, 2, 3]) {
      const html = render(`### 8.${k + 2} $k=${k}$ ${suffix}`)
      assert.equal((html.match(/class="katex-mathml ignore-header"/g) ?? []).length, 1)
      assert.ok(html.includes(`<annotation encoding="application/x-tex">k=${k}</annotation>`))
      assert.ok(html.includes('class="katex-html" aria-hidden="true"'))
      // MathML has no nested spans. Simulate the outline's ignored-branch removal
      // before reading text, without removing anything from the original document.
      const outline = html.replace(/<span class="katex-mathml ignore-header">[\s\S]*?<\/span>/g, '')
        .replace(/<[^>]+>/g, '')
      assert.equal((outline.match(new RegExp(`k=${k}`, 'g')) ?? []).length, 1)
      assert.ok(outline.includes(`8.${k + 2} k=${k} ${suffix}`))
    }
  }

  const mixed = render('## **$\\lambda$** and $k=1$\n\nBody $k=1$.\n\n$$\nk=1\n$$\n\n## Plain `code`')
  assert.equal((mixed.match(/katex-mathml ignore-header/g) ?? []).length, 2)
  assert.equal((mixed.match(/class="katex-mathml"/g) ?? []).length, 2)
  assert.ok(mixed.includes('>\\lambda</annotation>'))
  assert.match(mixed, /Plain <code\b[^>]*>code<\/code>/)
  assert.ok(!render('Body $k=2$.').includes('ignore-header'))
  assert.ok(!render('## `$k=1$`').includes('ignore-header'))
})
