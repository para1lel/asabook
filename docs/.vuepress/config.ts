import { viteBundler, type ViteBundlerOptions } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'
import { pseudocodeLanguage } from './pseudocode.js'
import { paperLinksPlugin } from './config/paper-links.js'
import { vndbCollections } from './config/vndb.js'
import { csdiyCollections } from './config/csdiy.js'
import { papersCollections, paperAbbreviations } from './config/papers.js'
import { yokubiCollections } from './config/yokubi.js'
import { gpuproCollections } from './config/gpupro.js'

export default defineUserConfig({
  locales: {
    '/': {
      lang: 'zh-CN',
      title: 'ASa Book',
      description: 'csdiy 与论文文档',
    },
    '/en/': {
      lang: 'en-US',
      title: 'ASa Book',
      description: 'Notes on self-directed study and papers',
    },
    '/ja/': {
      lang: 'ja-JP',
      title: 'ASa Book',
      description: '自主学習と論文についてのノート',
    },
  },

  plugins: [
    paperLinksPlugin(),
    {
      name: 'asabook:build-options',
      extendsBundlerOptions(options) {
        const viteOptions = ((options as ViteBundlerOptions).viteOptions ??= {})
        const build = (viteOptions.build ??= {})
        build.chunkSizeWarningLimit = 4096
      },
    },
  ],

  theme: plumeTheme({
    readAid: 'left',
    hostname: 'https://www.asabook.cc',
    docsRepo: 'https://github.com/para1lel/asabook',
    search: {
      provider: 'local',
      miniSearch: {
        options: {
          extractField(document, fieldName) {
            if (fieldName === 'id') return document.id
            if (fieldName === 'title' && !String(document.id).includes('#')) return document.title
            return ''
          },
        },
      },
    },
    comment: {
      provider: 'Giscus',
      comment: true,
      repo: 'para1lel/asabook',
      repoId: 'R_kgDOS_TuFQ',
      category: 'Announcements',
      categoryId: 'DIC_kwDOS_TuFc4DDtHK',
      mapping: 'pathname',
    },
    social: [
      { icon: 'github', link: 'https://github.com/para1lel/asabook' },
      { icon: 'bilibili', link: 'https://space.bilibili.com/349394806' },
    ],
    navbarSocialInclude: ['github', 'bilibili'],
    locales: {
      '/': {
        selectLanguageName: '简体中文',
        navbar: [
          { text: 'csdiy', link: '/csdiy/cse291a/', activeMatch: '^/csdiy/' },
          { text: 'papers', link: '/papers/taso/', activeMatch: '^/papers/' },
        ],
        collections: [
          vndbCollections['/'],
          csdiyCollections['/'],
          papersCollections['/'],
          yokubiCollections['/'],
          gpuproCollections['/'],
        ],
        footer: {
          message: '基于 VuePress 与 Plume 主题构建',
          copyright: 'Copyright © 2026 ASa Book',
        },
      },
      '/en/': {
        selectLanguageName: 'English',
        navbar: [
          { text: 'csdiy', link: '/en/csdiy/cse291a/', activeMatch: '^/en/csdiy/' },
          { text: 'papers', link: '/en/papers/taso/', activeMatch: '^/en/papers/' },
        ],
        collections: [
          vndbCollections['/en/'],
          csdiyCollections['/en/'],
          papersCollections['/en/'],
          gpuproCollections['/en/'],
          yokubiCollections['/en/'],
        ],
        footer: {
          message: 'Built with VuePress and the Plume theme',
          copyright: 'Copyright © 2026 ASa Book',
        },
      },
      '/ja/': {
        selectLanguageName: '日本語',
        navbar: [
          { text: 'csdiy', link: '/ja/csdiy/cse291a/', activeMatch: '^/ja/csdiy/' },
          { text: 'papers', link: '/ja/papers/taso/', activeMatch: '^/ja/papers/' },
        ],
        collections: [
          vndbCollections['/ja/'],
          csdiyCollections['/ja/'],
          papersCollections['/ja/'],
          yokubiCollections['/ja/'],
          gpuproCollections['/ja/'],
        ],
        footer: {
          message: 'VuePress と Plume テーマで構築',
          copyright: 'Copyright © 2026 ASa Book',
        },
      },
    },
    lastUpdated: false,
    editLink: false,
    changelog: true,
    contributors: false,
    llmstxt: true,
    plugins: {
      seo: {},
    },
    codeHighlighter: {
      langs: [pseudocodeLanguage],
      renderIndentGuides: true,
      colorizedBrackets: true
    },
    markdown: {
      abbr: paperAbbreviations,
      annotation: true,
      mermaid: true,
      youtube: true,
    },
  }),

  bundler: viteBundler({
    viteOptions: {
      build: {
        rolldownOptions: {
          checks: {
            pluginTimings: false,
          },
        },
      },
    },
  }),
})
