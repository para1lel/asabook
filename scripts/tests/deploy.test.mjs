import test from 'node:test'
import assert from 'node:assert/strict'
import { spawn } from 'node:child_process'
import { once } from 'node:events'
import { createServer } from 'node:net'
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { setTimeout as delay } from 'node:timers/promises'
import { fetchMain } from '../deploy/git-fetch.mjs'

test('Git fetch retries transient TLS failures and succeeds after recovery', async () => {
  let attempts = 0
  const waits = []
  await fetchMain(async () => {
    if (++attempts < 3) throw new Error('fatal: GnuTLS recv error (-110): The TLS connection was non-properly terminated.')
  }, async (ms) => { waits.push(ms) })
  assert.equal(attempts, 3)
  assert.deepEqual(waits, [10000, 20000])
})

test('Git fetch stops after four network failures and preserves the cause', async () => {
  let attempts = 0
  const cause = new Error('fatal: Failed to connect to github.com port 443: Connection timed out')
  await assert.rejects(fetchMain(async () => {
    attempts++
    throw cause
  }, async () => {}), (error) => error === cause)
  assert.equal(attempts, 4)
})

test('Git fetch does not retry authentication, certificate or repository errors', async () => {
  for (const message of ['Authentication failed', 'server certificate verification failed', 'Repository not found', 'cannot lock ref']) {
    let attempts = 0
    await assert.rejects(fetchMain(async () => {
      attempts++
      throw new Error(message)
    }, async () => { assert.fail('Must not retry permanent errors') }), { message })
    assert.equal(attempts, 1)
  }
})

test('deployment API rejects unauthenticated and unsafe requests and recovers interrupted jobs', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'asabook-deploy-test-'))
  await mkdir(join(directory, 'jobs'))
  const sha = 'a'.repeat(40)
  await writeFile(join(directory, 'jobs/finished.json'), JSON.stringify({ id: 'finished', sha, status: 'succeeded' }))
  await writeFile(join(directory, 'jobs/interrupted.json'), JSON.stringify({ id: 'interrupted', sha, status: 'running' }))
  const socket = createServer()
  socket.listen(0, '127.0.0.1')
  await once(socket, 'listening')
  const port = socket.address().port
  await new Promise((resolve) => socket.close(resolve))
  const token = 'test-token-'.repeat(8)
  const child = spawn(process.execPath, ['scripts/deploy/server.mjs'], {
    env: { ...process.env, ASABOOK_STATE: directory, ASABOOK_DEPLOY_PORT: String(port), ASABOOK_DEPLOY_TOKEN: token },
    stdio: 'pipe',
  })
  let errors = ''
  child.stderr.on('data', (data) => { errors += data })
  const base = `http://127.0.0.1:${port}`
  const auth = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json' }
  const post = (body, headers = auth) => fetch(`${base}/deploy`, { method: 'POST', headers, body: JSON.stringify(body) })
  try {
    let ready = false
    for (let attempt = 0; attempt < 50; attempt++) {
      try { ready = (await fetch(`${base}/healthz`)).ok } catch {}
      if (ready) break
      await delay(100)
    }
    assert.ok(ready, errors)
    assert.equal((await post({ id: 'finished', sha }, {})).status, 401)
    assert.equal((await post({ id: 'finished', sha }, { Authorization: 'Bearer wrong' })).status, 401)
    assert.equal((await post({ id: '../escape', sha })).status, 400)
    assert.equal((await post({ id: 'safe', sha: 'main; touch unsafe' })).status, 400)
    assert.equal((await fetch(`${base}/deploy`, { method: 'POST', headers: auth, body: '{bad json' })).status, 400)
    assert.equal((await post({ id: 'safe', sha, extra: 'x'.repeat(5000) })).status, 413)
    assert.equal((await fetch(`${base}/jobs/missing`, { headers: auth })).status, 404)
    const finished = await (await post({ id: 'finished', sha })).json()
    assert.equal(finished.status, 'succeeded')
    assert.equal((await post({ id: 'finished', sha: 'b'.repeat(40) })).status, 409)
    const interrupted = await (await fetch(`${base}/jobs/interrupted`, { headers: auth })).json()
    assert.equal(interrupted.status, 'failed')
    assert.match(interrupted.error, /restarted/)
    assert.equal((await post({ id: 'finished', sha })).status, 200)
  } finally {
    const exited = once(child, 'exit')
    child.kill()
    await exited
    await rm(directory, { recursive: true, force: true })
  }
})
