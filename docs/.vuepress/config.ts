import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'

export default defineUserConfig({
  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'ASa Book',
      description: 'vndb 与 csdiy 文档',
    },
    '/en/': {
      lang: 'en-US',
      title: 'ASa Book',
      description: 'Notes on visual novels and self-directed study',
    },
  },

  theme: plumeTheme({
    docsRepo: 'https://github.com/pare1lel/asabook',
    social: [
      { icon: 'github', link: 'https://github.com/pare1lel/asabook' },
      { icon: 'bilibili', link: 'https://space.bilibili.com/349394806' },
    ],
    navbarSocialInclude: ['github', 'bilibili'],
    locales: {
      '/': {
        selectLanguageName: '简体中文',
        navbar: [
          { text: 'vndb', link: '/vndb/intro/', activeMatch: '^/vndb/' },
          { text: 'csdiy', link: '/csdiy/cse291a/', activeMatch: '^/csdiy/' },
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
                collapsed: true,
                items: ['aops', 'tst26-p18', 'isl19-a5', 'tst26-p12'],
              },
              {
                text: 'CSE 291A',
                collapsed: false,
                items: ['cse291a', 'cse291a-week1', 'cse291a-week2', 'cse291a-week3'],
              },
            ],
          },
        ],
        footer: {
          message: '基于 VuePress 与 Plume 主题构建',
          copyright: 'Copyright © 2026 ASa Book',
        },
      },
      '/en/': {
        selectLanguageName: 'English',
        navbar: [
          { text: 'vndb', link: '/en/vndb/intro/', activeMatch: '^/en/vndb/' },
          { text: 'csdiy', link: '/en/csdiy/cse291a/', activeMatch: '^/en/csdiy/' },
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
                collapsed: true,
                items: ['aops', 'tst26-p18', 'isl19-a5', 'tst26-p12'],
              },
              {
                text: 'CSE 291A',
                collapsed: false,
                items: ['cse291a', 'cse291a-week1', 'cse291a-week2', 'cse291a-week3'],
              },
            ],
          },
        ],
        footer: {
          message: 'Built with VuePress and the Plume theme',
          copyright: 'Copyright © 2026 ASa Book',
        },
      },
    },
    lastUpdated: false,
    editLink: false,
    changelog: true,
    contributors: false,
    plugins: { git: true },
    markdown: { abbr: true, annotation: true },
  }),

  bundler: viteBundler(),
})
