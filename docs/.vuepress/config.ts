import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'

export default defineUserConfig({
  lang: 'zh-CN',
  title: 'ASa Book',
  description: 'vndb 与 csdiy 文档',

  theme: plumeTheme({
    navbar: [
      { text: 'vndb', link: '/vndb/intro/', activeMatch: '^/vndb/' },
      { text: 'csdiy', link: '/csdiy/aops/', activeMatch: '^/csdiy/' },
    ],
    collections: [
      {
        type: 'doc',
        title: 'vndb',
        dir: 'vndb',
        linkPrefix: '/vndb/',
        sidebar: [
          {
            text: 'VNDB',
            collapsed: false,
            items: ['intro'],
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
            text: 'AoPS',
            collapsed: false,
            items: ['aops', 'tst26-p18'],
          },
        ],
      },
    ],
    footer: {
      message: '基于 VuePress 与 Plume 主题构建',
      copyright: 'Copyright © 2026 ASa Book',
    },
    lastUpdated: false,
    changelog: true,
    contributors: false,
    plugins: { git: true },
  }),

  bundler: viteBundler(),
})
