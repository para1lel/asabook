import { readFile } from 'node:fs/promises'

const projectId = 'prj_9sPmvcq8vnnIkoYxZC0J34wwtuPw'
const teamId = 'team_MYDiqwh9c5osbN6EnTnoKZaY'
const token = process.env.VERCEL_CLEANUP_TOKEN
const apply = process.argv.includes('--apply')

if (!token) throw new Error('VERCEL_CLEANUP_TOKEN is required')

async function api(path, method = 'GET') {
  const url = new URL(path, 'https://api.vercel.com')
  url.searchParams.set('teamId', teamId)
  const response = await fetch(url, {
    method,
    headers: { Authorization: `Bearer ${token}` },
    signal: AbortSignal.timeout(30_000),
  })
  if (!response.ok) throw new Error(`Vercel ${method} ${url.pathname}: ${response.status}`)
  return response.status === 204 ? null : response.json()
}

async function production() {
  const project = await api(`/v9/projects/${projectId}`)
  const current = project.targets?.production
  if (project.id !== projectId || current?.readyState !== 'READY' || !current.id || !current.createdAt) {
    throw new Error(`Cannot identify a healthy production deployment; nothing deleted: ${JSON.stringify({
      projectId: project.id, targetKeys: Object.keys(project.targets ?? {}),
      productionId: current?.id, state: current?.readyState, createdAt: current?.createdAt,
      productionKeys: Object.keys(current ?? {}),
    })}`)
  }
  return current
}

const current = await production()
// Never use event-provided URLs for authenticated API requests or execute event-ref code.
if (process.env.GITHUB_EVENT_NAME === 'deployment_status') {
  const event = JSON.parse(await readFile(process.env.GITHUB_EVENT_PATH, 'utf8'))
  if (event.deployment_status?.state !== 'success'
    || event.deployment?.environment !== 'Production'
    || event.deployment?.creator?.login !== 'vercel[bot]'
    || event.deployment?.sha !== current.meta?.githubCommitSha) {
    console.log('Ignoring an event that does not match the current successful production deployment')
    process.exit(0)
  }
}

const deployments = new Map()
let until
do {
  const query = new URLSearchParams({ projectId, limit: '100' })
  if (until) query.set('until', String(until))
  const page = await api(`/v6/deployments?${query}`)
  for (const deployment of page.deployments) deployments.set(deployment.uid, deployment)
  const next = page.pagination?.next
  if (next && until && next >= until) throw new Error('Deployment pagination did not advance')
  until = next
} while (until)

const terminalStates = new Set(['READY', 'ERROR', 'CANCELED', 'BLOCKED'])
const candidates = [...deployments.values()].filter(deployment =>
  deployment.projectId === projectId
  && deployment.uid !== current.id
  && deployment.created < current.createdAt
  && terminalStates.has(deployment.state))

console.log(`Keep production ${current.id}; ${candidates.length} older deployments to delete; apply=${apply}`)
let deleted = 0
for (const deployment of candidates) {
  if (apply) {
    const latest = await production()
    if (latest.id !== current.id) throw new Error('Production changed during cleanup; stopping')
    await api(`/v13/deployments/${deployment.uid}`, 'DELETE')
    deleted++
  }
  console.log(`${apply ? 'Deleted' : 'Would delete'} ${deployment.uid}`)
}
console.log(`Completed: ${deleted} deployments deleted`)
