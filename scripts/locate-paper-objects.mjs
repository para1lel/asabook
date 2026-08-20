import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'
import { pathToFileURL } from 'node:url'

const { getDocument } = await import('pdfjs-dist/legacy/build/pdf.mjs')
const root = resolve(import.meta.dirname, '..')
const input = resolve(process.argv[2])
const wasmUrl = `${pathToFileURL(resolve(root, 'node_modules/pdfjs-dist/wasm')).href}/`

class LocalWasmFactory {
  async fetch({ filename }) {
    return new Uint8Array(readFileSync(resolve(root, 'node_modules/pdfjs-dist/wasm', filename)))
  }
}

const pdf = await getDocument({
  data: new Uint8Array(readFileSync(input)),
  useSystemFonts: true,
  wasmUrl,
  WasmFactory: LocalWasmFactory,
}).promise

for (let pageNumber = 1; pageNumber <= pdf.numPages; pageNumber += 1) {
  const page = await pdf.getPage(pageNumber)
  const viewport = page.getViewport({ scale: 1 })
  const content = await page.getTextContent()
  const matches = content.items.filter((item) => /^(?:Figure|Table)\s+[1-7]\b/.test(item.str.trim()))
  for (const item of matches) {
    console.log(JSON.stringify({
      height: viewport.height,
      page: pageNumber,
      text: item.str.trim(),
      width: viewport.width,
      x: item.transform[4],
      y: viewport.height - item.transform[5],
    }))
  }
}

await pdf.destroy()
