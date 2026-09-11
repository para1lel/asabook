import { spawn } from 'node:child_process'
import { access, readFile, readlink, realpath, rename, symlink } from 'node:fs/promises'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { state } from './common.mjs'

const releaseName = process.argv[2]
if (!releaseName || !/^[0-9a-f]{40}-[a-zA-Z0-9-]{1,80}$/.test(releaseName)) throw new Error('Provide a release directory name')
if (process.argv[3] !== '--locked') {
  const child = spawn('flock', ['-n', join(state, 'deploy.lock'), process.execPath, fileURLToPath(import.meta.url), releaseName, '--locked'], { stdio: 'inherit' })
  child.once('error', (error) => { console.error(error.message); process.exitCode = 1 })
  child.once('exit', (code) => { process.exitCode = code ?? 1 })
} else {
  const releases = await realpath(join(state, 'releases'))
  const release = await realpath(join(releases, releaseName))
  if (dirname(release) !== releases) throw new Error('Release is outside the release directory')
  await access(join(release, 'index.html'))
  const expected = JSON.parse(await readFile(join(release, '_deployment.json'), 'utf8'))
  if (expected.sha !== releaseName.slice(0, 40)) throw new Error('Release metadata does not match its name')
  const current = join(state, 'current')
  const previous = await readlink(current)
  const candidate = join(state, `rollback-${process.pid}`)
  await symlink(release, candidate)
  await rename(candidate, current)
  try {
    const response = await fetch('http://127.0.0.1:8787/_deployment.json', { signal: AbortSignal.timeout(10000) })
    const actual = await response.json()
    if (!response.ok || actual.sha !== expected.sha || actual.id !== expected.id) throw new Error('HTTP version check failed')
    console.log(`Published ${releaseName}`)
  } catch (error) {
    await symlink(previous, candidate)
    await rename(candidate, current)
    throw error
  }
}
