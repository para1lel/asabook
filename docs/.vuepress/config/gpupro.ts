import { defineCollection } from 'vuepress-theme-plume'

interface GpuproSidebarLabels {
  start: string
  home: string
  gpu: string
  tirx: string
  gemm: string
  flashAttention: string
  reference: string
  languageReference: string
  compilerInternals: string
}

function createGpuproSidebar(labels: GpuproSidebarLabels, homeLink: string) {
  return [
    {
      text: labels.start,
      collapsed: false,
      items: [{ text: labels.home, link: homeLink }],
    },
    {
      text: labels.gpu,
      collapsed: false,
      items: [
        'gpu-execution-model',
        'kernel-performance',
        'data-layout',
        'tensor-core-data-layouts',
        'tma',
        'blackwell-tensor-core',
        'tensor-memory',
        'mbarrier',
        'cluster-launch-control',
      ],
    },
    {
      text: labels.tirx,
      collapsed: false,
      items: ['tirx-introduction', 'tirx-layout-api'],
    },
    {
      text: labels.gemm,
      collapsed: false,
      items: ['tiled-gemm', 'pipelined-gemm', 'warp-specialized-gemm'],
    },
    {
      text: labels.flashAttention,
      collapsed: false,
      items: ['flash-attention-4'],
    },
    {
      text: labels.reference,
      collapsed: false,
      items: [
        'reference',
        {
          text: labels.languageReference,
          link: 'tirx-language-reference',
          collapsed: true,
          items: [
            'parser-utilities',
            'data-types-and-expressions',
            'buffers-and-memory',
            'control-flow',
            'cuda-ptx-intrinsics',
          ],
        },
        'debugging-warp-specialized-kernels',
        {
          text: labels.compilerInternals,
          link: 'compiler-internals',
          collapsed: true,
          items: ['tirx-lowering-pipeline'],
        },
      ],
    },
  ]
}

export const gpuproCollections = {
  '/': defineCollection({
    type: 'doc',
    title: 'GPU 编程',
    dir: 'gpupro',
    linkPrefix: '/gpupro/',
    sidebar: createGpuproSidebar({
      start: '从这里开始',
      home: '现代 GPU 编程',
      gpu: '第一部分: 理解 GPU',
      tirx: '第二部分: TIRx 概览',
      gemm: '第三部分: 从 Tiled 到 SOTA 的 GEMM',
      flashAttention: '第四部分: Flash Attention 4',
      reference: '参考资料',
      languageReference: 'TIRx 语言参考',
      compilerInternals: '编译器内部机制',
    }, '/gpupro/'),
  }),
  '/en/': defineCollection({
    type: 'doc',
    title: 'GPU Programming',
    dir: 'gpupro',
    linkPrefix: '/gpupro/',
    sidebar: createGpuproSidebar({
      start: 'Start Here',
      home: 'Modern GPU Programming For MLSys',
      gpu: 'Part I: Understanding the GPU',
      tirx: 'Part II: TIRx Overview',
      gemm: 'Part III: GEMM, Tiled to SOTA',
      flashAttention: 'Part IV: Flash Attention 4',
      reference: 'Reference',
      languageReference: 'TIRx Language Reference',
      compilerInternals: 'Compiler Internals',
    }, '/en/gpupro/'),
  }),
  '/ja/': defineCollection({
    type: 'doc',
    title: 'GPU プログラミング',
    dir: 'gpupro',
    linkPrefix: '/gpupro/',
    sidebar: createGpuproSidebar({
      start: 'はじめに',
      home: '機械学習システムのための現代 GPU プログラミング',
      gpu: '第 1 部: GPU を理解する',
      tirx: '第 2 部: TIRx 概要',
      gemm: '第 3 部: Tiled から SOTA への GEMM',
      flashAttention: '第 4 部: Flash Attention 4',
      reference: 'リファレンス',
      languageReference: 'TIRx 言語リファレンス',
      compilerInternals: 'コンパイラ内部',
    }, '/ja/gpupro/'),
  }),
}
