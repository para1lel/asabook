import { defineCollection } from 'vuepress-theme-plume'

interface YokubiSidebarLabels {
  introduction: string
  introductionLink: string
  startHere: string
  absoluteBeginner: string
  part1: string
  part2: string
  absoluteTerritory: string
  part3: string
  part4: string
  about: string
}

const yokubiLessons = (part: number, first: number, last: number) =>
  Array.from(
    { length: last - first + 1 },
    (_, offset) => `part-${part}/lesson-${String(first + offset).padStart(2, '0')}`,
  )

const yokubiSidebar = (labels: YokubiSidebarLabels) => [
  {
    text: labels.startHere,
    collapsed: false,
    items: [
      { text: labels.introduction, link: labels.introductionLink },
      'before-you-begin',
      'preamble',
    ],
  },
  {
    text: labels.absoluteBeginner,
    collapsed: false,
    items: [
      {
        text: labels.part1,
        link: 'part-1',
        collapsed: true,
        items: yokubiLessons(1, 0, 18),
      },
      {
        text: labels.part2,
        link: 'part-2',
        collapsed: true,
        items: yokubiLessons(2, 19, 28),
      },
    ],
  },
  {
    text: labels.absoluteTerritory,
    collapsed: false,
    items: [
      'absolute-territory',
      {
        text: labels.part3,
        link: 'part-3',
        collapsed: true,
        items: yokubiLessons(3, 29, 44),
      },
      {
        text: labels.part4,
        link: 'part-4',
        collapsed: true,
        items: yokubiLessons(4, 45, 63),
      },
      'closing-words',
    ],
  },
  {
    text: labels.about,
    collapsed: false,
    items: ['faq', 'credits'],
  },
]
export const yokubiCollections = {
  '/': defineCollection({
    type: 'doc',
    title: 'Yokubi',
    dir: 'yokubi',
    linkPrefix: '/yokubi/',
    sidebar: yokubiSidebar({
      introduction: '简介',
      introductionLink: '/yokubi/',
      startHere: '开始阅读',
      absoluteBeginner: '零基础',
      part1: '第一部分: 入门',
      part2: '第二部分: 渐入佳境',
      absoluteTerritory: '绝对领域',
      part3: '第三部分: 扩展句子',
      part4: '第四部分: 添点香料与“其他内容”',
      about: '关于 Yokubi',
    }),
  }),
  '/en/': defineCollection({
    type: 'doc',
    title: 'Yokubi',
    dir: 'yokubi',
    linkPrefix: '/yokubi/',
    sidebar: [
      {
        text: 'Start Here',
        collapsed: false,
        items: [
          { text: 'Introduction', link: '/en/yokubi/' },
          'before-you-begin',
          'preamble',
        ],
      },
      {
        text: 'Absolute Beginner',
        collapsed: false,
        items: [
          {
            text: 'Part 1: Getting Started',
            link: 'part-1',
            collapsed: true,
            items: [
              'part-1/lesson-00',
              'part-1/lesson-01',
              'part-1/lesson-02',
              'part-1/lesson-03',
              'part-1/lesson-04',
              'part-1/lesson-05',
              'part-1/lesson-06',
              'part-1/lesson-07',
              'part-1/lesson-08',
              'part-1/lesson-09',
              'part-1/lesson-10',
              'part-1/lesson-11',
              'part-1/lesson-12',
              'part-1/lesson-13',
              'part-1/lesson-14',
              'part-1/lesson-15',
              'part-1/lesson-16',
              'part-1/lesson-17',
              'part-1/lesson-18',
            ],
          },
          {
            text: 'Part 2: Getting Going',
            link: 'part-2',
            collapsed: true,
            items: [
              'part-2/lesson-19',
              'part-2/lesson-20',
              'part-2/lesson-21',
              'part-2/lesson-22',
              'part-2/lesson-23',
              'part-2/lesson-24',
              'part-2/lesson-25',
              'part-2/lesson-26',
              'part-2/lesson-27',
              'part-2/lesson-28',
            ],
          },
        ],
      },
      {
        text: 'Absolute Territory',
        collapsed: false,
        items: [
          'absolute-territory',
          {
            text: 'Part 3: Growing Our Sentences',
            link: 'part-3',
            collapsed: true,
            items: [
              'part-3/lesson-29',
              'part-3/lesson-30',
              'part-3/lesson-31',
              'part-3/lesson-32',
              'part-3/lesson-33',
              'part-3/lesson-34',
              'part-3/lesson-35',
              'part-3/lesson-36',
              'part-3/lesson-37',
              'part-3/lesson-38',
              'part-3/lesson-39',
              'part-3/lesson-40',
              'part-3/lesson-41',
              'part-3/lesson-42',
              'part-3/lesson-43',
              'part-3/lesson-44',
            ],
          },
          {
            text: 'Part 4: Adding Spices and "The Rest"',
            link: 'part-4',
            collapsed: true,
            items: [
              'part-4/lesson-45',
              'part-4/lesson-46',
              'part-4/lesson-47',
              'part-4/lesson-48',
              'part-4/lesson-49',
              'part-4/lesson-50',
              'part-4/lesson-51',
              'part-4/lesson-52',
              'part-4/lesson-53',
              'part-4/lesson-54',
              'part-4/lesson-55',
              'part-4/lesson-56',
              'part-4/lesson-57',
              'part-4/lesson-58',
              'part-4/lesson-59',
              'part-4/lesson-60',
              'part-4/lesson-61',
              'part-4/lesson-62',
              'part-4/lesson-63',
            ],
          },
          'closing-words',
        ],
      },
      {
        text: 'About Yokubi',
        collapsed: false,
        items: ['faq', 'credits'],
      },
    ],
  }),
  '/ja/': defineCollection({
    type: 'doc',
    title: 'Yokubi',
    dir: 'yokubi',
    linkPrefix: '/yokubi/',
    sidebar: yokubiSidebar({
      introduction: '紹介',
      introductionLink: '/ja/yokubi/',
      startHere: 'はじめに',
      absoluteBeginner: '初学者向け',
      part1: '第 1 部: はじめの一歩',
      part2: '第 2 部: 学習を進める',
      absoluteTerritory: '絶対領域',
      part3: '第 3 部: 文を広げる',
      part4: '第 4 部: ひと味加える・「その他」',
      about: 'Yokubi について',
    }),
  }),
}
