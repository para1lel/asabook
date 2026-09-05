import { createCanvas, loadImage } from '@napi-rs/canvas'
import { readdirSync, writeFileSync } from 'node:fs'
import { extname, join, resolve } from 'node:path'

const dir = resolve(process.argv[2])
const out = resolve(process.argv[3])
const files = readdirSync(dir)
  .filter((f) => extname(f) === '.png')
  .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }))

const cols = 4
const cellW = 480
const cellH = 360
const titleH = 26
const rows = Math.ceil(files.length / cols)
const canvas = createCanvas(cols * cellW, rows * (cellH + titleH))
const ctx = canvas.getContext('2d')
ctx.fillStyle = '#fff'
ctx.fillRect(0, 0, canvas.width, canvas.height)

for (let i = 0; i < files.length; i += 1) {
  const file = files[i]
  const image = await loadImage(join(dir, file))
  const col = i % cols
  const row = Math.floor(i / cols)
  const x = col * cellW
  const y = row * (cellH + titleH)
  ctx.fillStyle = '#000'
  ctx.font = '14px sans-serif'
  ctx.fillText(file, x + 4, y + 16)
  const scale = Math.min(cellW / image.width, cellH / image.height)
  const w = image.width * scale
  const h = image.height * scale
  ctx.drawImage(image, x + (cellW - w) / 2, y + titleH + (cellH - h) / 2, w, h)
  ctx.strokeStyle = '#ccc'
  ctx.strokeRect(x, y, cellW, cellH + titleH)
}

writeFileSync(out, canvas.toBuffer('image/png'))
console.log(`Wrote ${files.length} crops to ${out}`)
