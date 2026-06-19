import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'ASa Book',
  description: 'vndb 与 csdiy 文档',

  theme: plumeTheme({
    navbar: [
      { text: 'vndb', link: '/vndb/', activeMatch: '^/vndb/' },
      { text: 'csdiy', link: '/csdiy/', activeMatch: '^/csdiy/' },
    ],
    collections: [
      {
        type: 'doc',
        title: 'vndb',
        dir: 'vndb',
        linkPrefix: '/vndb/',
        sidebar: [
          {
            text: 'vndb',
            collapsed: false,
            items: ['', 'getting-started', 'reference'],
          },
        ],
      },
      {
        type: 'doc',
        title: 'csdiy',
        dir: 'csdiy',
        linkPrefix: '/csdiy/',
        sidebar: [
          {
            text: 'csdiy',
            collapsed: false,
            items: ['', 'getting-started', 'resources'],
          },
        ],
      },
    ],
    footer: {
      message: '基于 VuePress 与 Plume 主题构建',
      copyright: 'Copyright © 2026 ASa Book',
    },
  }),

  bundler: viteBundler(),
})
