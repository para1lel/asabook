import { defineCollection } from 'vuepress-theme-plume'

export const csdiyCollections = {
  '/': defineCollection({
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
  }),
  '/en/': defineCollection({
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
  }),
  '/ja/': defineCollection({
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
  }),
}
