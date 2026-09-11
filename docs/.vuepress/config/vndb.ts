import { defineCollection } from 'vuepress-theme-plume'

export const vndbCollections = {
  '/': defineCollection({
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
  }),
  '/en/': defineCollection({
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
  }),
  '/ja/': defineCollection({
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
  }),
}
