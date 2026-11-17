import { spawn } from 'node:child_process'
import { mkdir, open, readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readJob, run, saveJob, state, validateCheckout } from './common.mjs'
import { fetchMain } from './git-fetch.mjs'

await mkdir(join(state, 'logs'), { recursive: true })
await validateCheckout()
await fetchMain(run)
const sha = await run('git', ['rev-parse', 'origin/main'], true)
let published = null
try {
  published = JSON.parse(await readFile(join(state, 'current/_deployment.json'), 'utf8'))
} catch (error) {
  if (error.code !== 'ENOENT') throw error
}
if (published?.sha === sha && await run('git', ['rev-parse', 'HEAD'], true) === sha) {
  console.log(`Already published ${sha}`)
} else {
  const id = `sync-${Date.now()}-${process.pid}`
  await saveJob({ id, sha, status: 'queued', createdAt: new Date().toISOString() })
  const log = await open(join(state, 'logs', `${id}.log`), 'a', 0o600)
  try {
    const code = await new Promise((resolve, reject) => {
      const child = spawn(process.execPath, [fileURLToPath(new URL('./worker.mjs', import.meta.url)), id], {
        stdio: ['ignore', log.fd, log.fd],
      })
      child.once('error', reject)
      child.once('close', resolve)
    })
    const result = await readJob(id)
    if (code !== 0 || !['succeeded', 'superseded'].includes(result?.status)) {
      throw new Error(`Deployment ${id} failed: ${result?.error || 'worker exited unexpectedly'}. See ${join(state, 'logs', `${id}.log`)}`)
    }
    console.log(`Deployment ${id}: ${result.status} (${sha})`)
  } finally {
    await log.close()
  }
}
