import assert from 'node:assert/strict'
import { test } from 'node:test'
import { createDevApp } from 'vuepress/core'
import { viteBundler } from '@vuepress/bundler-vite'
import { plumeTheme } from 'vuepress-theme-plume'
import { parse } from 'vue/compiler-sfc'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { paperDate, readPaperAbbreviations, writePaperAbbreviations } from '../lib/paper-config.mjs'
import { paperLinksPlugin } from '../../docs/.vuepress/config/paper-links.ts'
import { synchronizePaperConfig } from '../sync-paper-config.mjs'

test('publication years do not come from arXiv identifiers or page ranges', () => {
  for (const [key, value, year] of [
    ['Wan18a', 'CoRR, abs/1801.04380, 2018.', '2018'],
    ['Bro20', 'pp. 1877–1901. Curran Associates, 2020a.', '2020'],
    ['Gou21', 'Journal 129(6): 1789-1819, 2021.', '2021'],
    ['Rie67', 'Dieterich, 1867.', '1867'],
    ['Ref25', 'Undated legacy bibliography entry.', '2025'],
  ]) assert.equal(paperDate({ key, value }), year)
  assert.equal(paperDate({ key: 'Ref24', value: 'Conference 2024.', date: '2023-12-05' }), '2023-12-05')
})

test('sorted writes preserve escaped strings, aliases, date precision, and neighbouring configuration', () => {
  const source = 'export const paperAbbreviations = {}\n\nexport const unrelated = 42\n'
  const entries = [
    { key: 'Zed25', value: "O'Neil. \\alpha. 2025.", date: '2024-02-01' },
    { key: 'Def24', value: '2024. [Link](https://example.com)' },
    { key: 'Abc24', value: '2024. [Link](https://example.com)' },
  ]
  const output = writePaperAbbreviations(source, entries)
  const parsed = readPaperAbbreviations(output).entries
  assert.deepEqual(parsed.map(({ key }) => key), ['Abc24', 'Def24', 'Zed25'])
  assert.equal(parsed[2].value, entries[0].value)
  assert.equal(parsed[2].date, '2024-02-01')
  assert.equal(writePaperAbbreviations(output, parsed), output)
  assert.ok(output.endsWith('export const unrelated = 42\n'))
  assert.throws(() => readPaperAbbreviations("const paperAbbreviations = { 'A': 'one', 'A': 'two' }"), /Duplicate/)
})

test('paper matching distinguishes full titles from shared venue names and implementations', () => {
  const source = writePaperAbbreviations('const paperAbbreviations = {}', [
    { key: 'Vas17b', value: 'Ashish Vaswani et al. Attention is all you need, 2017.' },
    { key: 'Han22', value: 'Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22). 2022.' },
    { key: 'Rau19', value: 'David Rau. "Sparsely-Gated Mixture-of-Experts PyTorch Implementation." 2019. [Link](https://github.com/davidmrau/mixture-of-experts)' },
    { key: 'Liu26', value: 'ECHO: Efficient KV Cache Offloading with Lossless Prefetching for Serving Native Sparse Attention LLMs. 2026.' },
  ])
  const entries = readPaperAbbreviations(synchronizePaperConfig(source).source).entries
  const attention = entries.find(({ key }) => key === 'Vas17b')
  assert.ok(attention.value.endsWith('[Link](/papers/attention-is-all-you-need/)'))
  assert.equal(attention.date, '2017-06-12')
  for (const { key, value } of entries) {
    if (key !== 'Vas17b') assert.ok(!value.includes('[Link](/papers/'), value)
  }
})

test('real Plume tooltips follow page language and compile with quoted titles', async (context) => {
  const definition = 'Author. "A & B". [Link](/papers/taso/) [Source](https://example.com)'
  const prefix = join(tmpdir(), 'asabook-paper-config-test-')
  const source = mkdtempSync(prefix)
  context.after(() => {
    assert.ok(source.startsWith(prefix))
    rmSync(source, { recursive: true, force: true })
  })
  const app = createDevApp({
    source,
    bundler: viteBundler(),
    theme: plumeTheme({ cache: false, markdown: { abbr: { Ref24: definition } } }),
    plugins: [paperLinksPlugin()],
  })
  await app.init()
  for (const [filePathRelative, prefix] of [
    ['en/papers/example.md', '/en'],
    ['ja\\papers\\example.md', '/ja'],
    ['papers/example.md', ''],
    ['en/csdiy/example.md', '/en'],
  ]) {
    const html = app.markdown.render('[Ref24]', { filePathRelative })
    assert.ok(html.includes(`href="${prefix}/papers/taso/"`), html)
    assert.ok(html.includes('href="https://example.com"'), html)
    assert.ok(html.includes('aria-label="Author. &quot;A &amp; B&quot;.'), html)
    assert.equal(parse(`<template>${html}</template>`).errors.length, 0, html)
  }
})
