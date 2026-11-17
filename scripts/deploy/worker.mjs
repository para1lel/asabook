import { access, cp, mkdir, readlink, rename, symlink, writeFile, rm } from 'node:fs/promises'
import { join } from 'node:path'
import { readJob, repo, run, saveJob, state, validId, validateCheckout } from './common.mjs'
import { fetchMain } from './git-fetch.mjs'

const id = process.argv[2]
if (!validId(id)) throw new Error('Invalid job ID')
let job = await readJob(id)
if (!job) throw new Error('Unknown deployment job')

async function update(fields) {
  job = { ...job, ...fields }
  await saveJob(job)
}

try {
  await update({ status: 'running', startedAt: new Date().toISOString() })
  await validateCheckout()
  await fetchMain(run)
  const latest = await run('git', ['rev-parse', 'origin/main'], true)
  if (job.sha !== latest) {
    await update({ status: 'superseded', finishedAt: new Date().toISOString() })
  } else {
    await run('git', ['merge', '--ff-only', latest])
    if (await run('git', ['rev-parse', 'HEAD'], true) !== job.sha) throw new Error('Checkout does not match the requested deployment commit')
    await run('npm', ['ci', '--no-audit', '--no-fund'])
    await run('npm', ['run', 'paper:check-config'])
    await run('npm', ['run', 'docs:build', '--', '--clean-cache'])
    const dist = join(repo, 'docs/.vuepress/dist')
    for (const path of ['index.html', 'en/index.html', 'ja/index.html', '404.html']) await access(join(dist, path))
    const releaseName = `${job.sha}-${job.id}`
    const release = join(state, 'releases', releaseName)
    await mkdir(join(state, 'releases'), { recursive: true })
    await cp(dist, release, { recursive: true, errorOnExist: true, force: false })
    await writeFile(join(release, '_deployment.json'), `${JSON.stringify({ sha: job.sha, id: job.id })}\n`)
    // Keep hashed assets across releases for browsers with older pages open.
    await cp(join(dist, 'assets'), join(state, 'assets'), { recursive: true })
    const candidate = join(state, `candidate-${job.id}`)
    await symlink(release, candidate)
    let previous = null
    try { previous = await readlink(join(state, 'current')) } catch (error) { if (error.code !== 'ENOENT') throw error }
    await rename(candidate, join(state, 'current'))
    try {
      const response = await fetch('http://127.0.0.1:8787/_deployment.json', { signal: AbortSignal.timeout(10000) })
      const online = await response.json()
      if (!response.ok || online.sha !== job.sha || online.id !== job.id) throw new Error('Published version did not pass HTTP health check')
    } catch (error) {
      if (previous) {
        await symlink(previous, candidate)
        await rename(candidate, join(state, 'current'))
      } else {
        await rm(join(state, 'current'))
      }
      throw error
    }
    await update({ status: 'succeeded', release: releaseName, finishedAt: new Date().toISOString() })
  }
} catch (error) {
  console.error(error)
  await update({ status: 'failed', error: error.message, finishedAt: new Date().toISOString() })
  process.exitCode = 1
}
