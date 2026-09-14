import { setTimeout as delay } from 'node:timers/promises'

const transientNetworkError = /GnuTLS recv error|TLS connection was non-properly terminated|Connection (?:reset|timed out)|Could not resolve host|Failed to connect|Couldn't connect to server|Operation too slow|The requested URL returned error: (?:429|502|503|504)|RPC failed; curl (?:18|28|35|52|56|92)\b/i

export async function fetchMain(run, wait = delay) {
  for (let attempt = 1; attempt <= 4; attempt++) {
    try {
      await run('git', [
        '-c', 'http.proxy=', '-c', 'https.proxy=',
        '-c', 'http.lowSpeedLimit=1024', '-c', 'http.lowSpeedTime=60',
        'fetch', '--prune', 'origin',
      ])
      return
    } catch (error) {
      if (attempt === 4 || !transientNetworkError.test(error.message)) throw error
      console.log(`Git fetch network failure; retrying (${attempt}/3)`)
      await wait(attempt * 10000)
    }
  }
}
