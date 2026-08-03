import { inject as injectAnalytics } from '@vercel/analytics'
import { injectSpeedInsights } from '@vercel/speed-insights'
import { onMounted, onUnmounted } from 'vue'
import { defineClientConfig } from 'vuepress/client'
import 'lxgw-wenkai-webfont/lxgwwenkai-regular.css'
import './styles/index.css'

function resizeGpuproDemo(event: MessageEvent) {
  if (event.origin !== window.location.origin || event.data?.type !== 'demoHeight') return

  const height = Number(event.data.height)
  if (!Number.isFinite(height) || height <= 0 || height > 10000) return

  const iframe = Array.from(document.querySelectorAll<HTMLIFrameElement>('.gpupro-page .vp-doc iframe'))
    .find((element) => element.contentWindow === event.source)
  if (!iframe) return

  // Auto-height replaces vertical scrolling; horizontal scrolling stays available for wide demos.
  const iframeRoot = iframe.contentDocument?.documentElement
  if (iframeRoot && iframeRoot.style.overflowY !== 'hidden') iframeRoot.style.overflowY = 'hidden'

  const borderHeight = iframe.offsetHeight - iframe.clientHeight
  const contentHeight = Math.ceil(height)
  const frameHeight = getComputedStyle(iframe).boxSizing === 'border-box'
    ? contentHeight + borderHeight
    : contentHeight

  iframe.style.height = `${frameHeight}px`
}

export default defineClientConfig({
  setup() {
    onMounted(() => {
      injectAnalytics()
      injectSpeedInsights()
      window.addEventListener('message', resizeGpuproDemo)
    })

    onUnmounted(() => {
      window.removeEventListener('message', resizeGpuproDemo)
    })
  },
})
