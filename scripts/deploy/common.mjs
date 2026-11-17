import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'
import { spawn } from 'node:child_process'

export const repo = process.env.ASABOOK_REPO || '/data1/shared/dongwu.chen/asabook'
export const state = process.env.ASABOOK_STATE || '/data1/shared/dongwu.chen/state/asabook'
export const validSha = (value) => typeof value === 'string' && /^[0-9a-f]{40}$/.test(value)
export const validId = (value) => typeof value === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(value)

export function run(command, args, capture = false) {
  return new Promise((resolve, reject) => {
    const child = spawn(command, args, { cwd: repo, stdio: ['ignore', capture ? 'pipe' : 'inherit', 'pipe'] })
    let output = ''
    let stderr = ''
    child.stdout?.on('data', (chunk) => { output += chunk })
    child.stderr.on('data', (chunk) => {
      process.stderr.write(chunk)
      stderr = (stderr + chunk).slice(-4096)
    })
    child.once('error', reject)
    child.once('close', (code) => code === 0 ? resolve(output.trim()) : reject(new Error(`${command} ${args.join(' ')} failed (${code}): ${stderr.trim()}`)))
  })
}

export async function validateCheckout() {
  if (await run('git', ['status', '--porcelain'], true)) throw new Error('Server checkout has local changes')
  if (await run('git', ['rev-parse', '--is-shallow-repository'], true) !== 'false') throw new Error('Full Git history is required')
  if (await run('git', ['remote', 'get-url', 'origin'], true) !== 'https://github.com/para1lel/asabook.git') throw new Error('Unexpected repository origin')
  if (await run('git', ['branch', '--show-current'], true) !== 'main') throw new Error('Server checkout must use main')
}

export async function saveJob(job) {
  if (!validId(job.id)) throw new Error('Invalid job ID')
  const directory = join(state, 'jobs')
  await mkdir(directory, { recursive: true })
  const target = join(directory, `${job.id}.json`)
  await writeFile(`${target}.tmp`, `${JSON.stringify(job)}\n`, { mode: 0o600 })
  await rename(`${target}.tmp`, target)
}

export async function readJob(id) {
  if (!validId(id)) throw new Error('Invalid job ID')
  try {
    return JSON.parse(await readFile(join(state, 'jobs', `${id}.json`), 'utf8'))
  } catch (error) {
    if (error.code === 'ENOENT') return null
    throw error
  }
}
