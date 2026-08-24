import type { Plugin } from 'vite'

export function lowMemoryBuildPlugin(): Plugin {
  // VuePress starts the client and SSR builds together; queue them to avoid overlapping peaks.
  let buildQueue = Promise.resolve()
  let releaseActiveBuild: (() => void) | undefined

  const release = () => {
    releaseActiveBuild?.()
    releaseActiveBuild = undefined
  }

  return {
    name: 'asabook:low-memory-build',
    apply: 'build',
    async buildStart() {
      const previousBuild = buildQueue
      let releaseBuild: () => void

      buildQueue = new Promise<void>((resolve) => {
        releaseBuild = resolve
      })

      await previousBuild
      releaseActiveBuild = releaseBuild!
    },
    buildEnd(error) {
      if (error) release()
    },
    closeBundle: release,
  }
}
