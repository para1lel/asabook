import { setTimeout as delay } from 'node:timers/promises'

const endpoint = process.env.ASABOOK_DEPLOY_URL
const token = process.env.ASABOOK_DEPLOY_TOKEN
if (!endpoint?.startsWith('https://') || !token) throw new Error('Deployment URL/token are not configured')
const id = `${process.env.GITHUB_RUN_ID}-${process.env.GITHUB_RUN_ATTEMPT}`
const sha = process.env.GITHUB_SHA
const deadline = Date.now() + 110 * 60 * 1000
const headers = { Authorization: `Bearer ${token}`, 'Content-Type': 'application/json', 'User-Agent': 'ASaBookDeployment/1.0' }
let accepted = false
let lastStatus = ''
while (Date.now() < deadline) {
  let response
  try {
    response = await fetch(`${endpoint}${accepted ? `/jobs/${id}` : '/deploy'}`, {
      method: accepted ? 'GET' : 'POST', headers, redirect: 'error',
      body: accepted ? undefined : JSON.stringify({ id, sha }),
      signal: AbortSignal.timeout(20000),
    })
  } catch {
    console.log('Deployment endpoint unavailable; retrying')
    await delay(15000)
    continue
  }
  if ([409, 429, 502, 503, 504, 530].includes(response.status)) {
    await delay(15000)
    continue
  }
  if (!response.ok) throw new Error(`Deployment API returned HTTP ${response.status}`)
  const job = await response.json()
  if (job.id !== id || job.sha !== sha) throw new Error('Unexpected deployment response')
  accepted = true
  if (job.status !== lastStatus) console.log(`Deployment ${id}: ${job.status}`)
  lastStatus = job.status
  if (['succeeded', 'superseded'].includes(job.status)) process.exit(0)
  if (job.status === 'failed') throw new Error(job.error || 'Server build failed; inspect the server log')
  await delay(10000)
}
throw new Error('Timed out waiting for server deployment; inspect the server before retrying')
