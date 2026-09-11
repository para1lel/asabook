import { createServer } from 'node:http'
import { timingSafeEqual } from 'node:crypto'
import { spawn } from 'node:child_process'
import { open, mkdir, readdir } from 'node:fs/promises'
import { join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { readJob, saveJob, state, validId, validSha } from './common.mjs'

const token = process.env.ASABOOK_DEPLOY_TOKEN
if (!token || token.length < 32) throw new Error('ASABOOK_DEPLOY_TOKEN must contain at least 32 characters')
const expectedAuth = Buffer.from(`Bearer ${token}`)
const worker = fileURLToPath(new URL('./worker.mjs', import.meta.url))
await mkdir(join(state, 'logs'), { recursive: true })
await mkdir(join(state, 'jobs'), { recursive: true })

// A stopped service terminates its worker through systemd's control group.
for (const name of await readdir(join(state, 'jobs'))) {
  if (!name.endsWith('.json')) continue
  const job = await readJob(name.slice(0, -5))
  if (job && ['queued', 'running'].includes(job.status)) {
    await saveJob({ ...job, status: 'failed', error: 'Deployment service restarted; retry the workflow' })
  }
}

let active = null
const reply = (res, status, body) => {
  res.writeHead(status, { 'Content-Type': 'application/json', 'Cache-Control': 'no-store' })
  res.end(JSON.stringify(body))
}

async function startJob(job) {
  await saveJob(job)
  const log = await open(join(state, 'logs', `${job.id}.log`), 'a', 0o600)
  const { ASABOOK_DEPLOY_TOKEN: ignoredToken, ...workerEnv } = process.env
  const child = spawn('flock', ['-n', join(state, 'deploy.lock'), process.execPath, worker, job.id], {
    stdio: ['ignore', log.fd, log.fd],
    env: workerEnv,
  })
  child.once('error', async () => {
    await saveJob({ ...job, status: 'failed', error: 'Unable to start deployment worker' })
  })
  child.once('close', async () => {
    try {
      const result = await readJob(job.id)
      if (result && ['queued', 'running'].includes(result.status)) {
        await saveJob({ ...result, status: 'failed', error: 'Deployment worker exited unexpectedly' })
      }
    } finally {
      await log.close()
      active = null
    }
  })
}

const server = createServer(async (req, res) => {
  try {
    if (req.method === 'GET' && req.url === '/healthz') return reply(res, 200, { ok: true })
    const supplied = Buffer.from(req.headers.authorization || '')
    if (supplied.length !== expectedAuth.length || !timingSafeEqual(supplied, expectedAuth)) {
      return reply(res, 401, { error: 'Unauthorized' })
    }
    if (req.method === 'GET' && req.url.startsWith('/jobs/')) {
      const id = req.url.slice('/jobs/'.length)
      if (!validId(id)) return reply(res, 400, { error: 'Invalid job ID' })
      const job = await readJob(id)
      return reply(res, job ? 200 : 404, job || { error: 'Unknown job' })
    }
    if (req.method !== 'POST' || req.url !== '/deploy') return reply(res, 404, { error: 'Not found' })
    let body = ''
    for await (const chunk of req) {
      body += chunk.toString('utf8')
      if (Buffer.byteLength(body) > 4096) return reply(res, 413, { error: 'Request too large' })
    }
    let request
    try { request = JSON.parse(body) } catch { return reply(res, 400, { error: 'Invalid JSON' }) }
    if (!validId(request.id) || !validSha(request.sha)) return reply(res, 400, { error: 'Invalid deployment request' })
    if (active) return reply(res, 409, { error: 'A deployment is running' })
    // Claim the slot before asynchronous I/O to serialize simultaneous requests.
    active = request.id
    try {
      const previous = await readJob(request.id)
      if (previous) {
        active = null
        if (previous.sha !== request.sha) return reply(res, 409, { error: 'Job ID already used' })
        return reply(res, 200, previous)
      }
      const job = { id: request.id, sha: request.sha, status: 'queued', createdAt: new Date().toISOString() }
      await startJob(job)
      reply(res, 202, job)
    } catch (error) {
      active = null
      throw error
    }
  } catch (error) {
    console.error(error.message)
    // Once a worker is started it owns the slot until its close event.
    reply(res, 500, { error: 'Deployment service error' })
  }
})
server.requestTimeout = 15000
server.headersTimeout = 10000
server.listen(Number(process.env.ASABOOK_DEPLOY_PORT || 8788), '127.0.0.1')
