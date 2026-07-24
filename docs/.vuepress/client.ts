import { inject as injectAnalytics } from '@vercel/analytics'
import { injectSpeedInsights } from '@vercel/speed-insights'
import { onMounted } from 'vue'
import { defineClientConfig } from 'vuepress/client'
import 'lxgw-wenkai-webfont/lxgwwenkai-regular.css'
import './styles/index.css'

export default defineClientConfig({
  setup() {
    onMounted(() => {
      injectAnalytics()
      injectSpeedInsights()
    })
  },
})
