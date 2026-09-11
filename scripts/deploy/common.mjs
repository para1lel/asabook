import { mkdir, readFile, rename, writeFile } from 'node:fs/promises'
import { join } from 'node:path'

export const repo = process.env.ASABOOK_REPO || '/data0/shared/dongwu.chen/asabook'
export const state = process.env.ASABOOK_STATE || '/data0/shared/dongwu.chen/asabook-deploy'
export const validSha = (value) => typeof value === 'string' && /^[0-9a-f]{40}$/.test(value)
export const validId = (value) => typeof value === 'string' && /^[a-zA-Z0-9-]{1,80}$/.test(value)

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
