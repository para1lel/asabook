import { viteBundler } from '@vuepress/bundler-vite'
import { defineUserConfig } from 'vuepress'
import { plumeTheme } from 'vuepress-theme-plume'
import { pseudocodeLanguage } from './pseudocode.js'

const paperAbbreviations = {
  'Ama17': '"Amazon EC2 P3 Instances." 2017. [Link](https://aws.amazon.com/ec2/instance-types/p3/)',
  'Ten17': '"Tensorflow graph transform creates corrupted graph." 2017. [Link](https://github.com/tensorflow/tensorflow/issues/7523)',
  'XLA17': '"XLA: Optimizing Compiler for TensorFlow." 2017. [Link](https://www.tensorflow.org/xla)',
  'Gra18': '"Graph transform: fold constant with invalid graph." 2018. [Link](https://github.com/tensorflow/tensorflow/issues/16545)',
  'Ten18': '"Tensor Cores in NVIDIA Volta Architecture." 2018. [Link](https://www.nvidia.com/en-us/data-center/tensorcore/)',
  'Aba16': 'Martín Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, Manjunath Kudlur, Josh Levenberg, Rajat Monga, Sherry Moore, Derek G Murray, Benoit Steiner, Paul Tucker, Vijay Vasudevan, Pete Warden, Martin Wicke, Yuan Yu, Xiaoqiang Zheng. "TensorFlow: A System for Large-Scale Machine Learning." Proceedings of the 12th USENIX Conference on Operating Systems Design and Implementation (OSDI). 2016.',
  'Ban06': 'Sorav Bansal, Alex Aiken. "Automatic Generation of Peephole Superoptimizers." Proceedings of the 12th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS XII). 2006.',
  'Che18': 'Tianqi Chen, Thierry Moreau, Ziheng Jiang, Haichen Shen, Eddie Q Yan, Leyuan Wang, Yuwei Hu, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. "TVM: End-to-End Optimization Stack for Deep Learning." 2018. [Link](http://arxiv.org/abs/1802.04799)',
  'Che18a': 'Tianqi Chen, Lianmin Zheng, Eddie Yan, Ziheng Jiang, Thierry Moreau, Luis Ceze, Carlos Guestrin, Arvind Krishnamurthy. "Learning to Optimize Tensor Programs." Advances in Neural Information Processing Systems. 2018.',
  'Che14': 'Sharan Chetlur, Cliff Woolley, Philippe Vandermersch, Jonathan Cohen, John Tran, Bryan Catanzaro, Evan Shelhamer. "cuDNN: Efficient Primitives for Deep Learning." 2014. [Link](http://arxiv.org/abs/1410.0759)',
  'Chu19': 'Berkeley R Churchill, Oded Padon, Rahul Sharma, Alex Aiken. "Semantic Program Alignment for Equivalence Checking." Proceedings of the 2019 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI). June 22-26, 2019. [Link](https://doi.org/10.1145/3314221.3314596)',
  'Cub16': 'Cublas. "Dense Linear Algebra on GPUs." 2016. [Link](https://developer.nvidia.com/cublas)',
  'Dah17': 'Manjeet Dahiya, Sorav Bansal. "Black-Box Equivalence Checking Across Compiler Optimizations." Programming Languages and Systems. 2017.',
  'DeM08': 'Leonardo De Moura, Nikolaj Bjørner. "Z3: An Efficient SMT Solver." Proceedings of the Theory and Practice of Software, 14th International Conference on Tools and Algorithms for the Construction and Analysis of Systems (TACAS\'08/ETAPS\'08). 2008.',
  'Dev18': 'Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova. "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding." 2018.',
  'Dum16': 'Vincent Dumoulin, Francesco Visin. "A guide to convolution arithmetic for deep learning." CoRR. 2016.',
  'Gul03': 'Sumit Gulwani, George C Necula. "Discovering Affine Equalities Using Random Interpretation." Proceedings of the 30th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL \'03). 2003.',
  'He16': 'Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Deep residual learning for image recognition." Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR). 2016.',
  'How17': 'Andrew G Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam. "MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications." 2017.',
  'Jia18': 'Zhihao Jia, Sina Lin, Charles R Qi, Alex Aiken. "Exploring Hidden Dimensions in Accelerating Convolutional Neural Networks." Proceedings of the 35th International Conference on Machine Learning (Proceedings of Machine Learning Research). 2018.',
  'Jia19': 'Zhihao Jia, James Thomas, Todd Warzawski, Mingyu Gao, Matei Zaharia, Alex Aiken. "Optimizing DNN Computation with Relaxed Graph Substitutions." Proceedings of the 2nd Conference on Systems and Machine Learning (SysML\'19). 2019.',
  'Jia19a': 'Zhihao Jia, Matei Zaharia, Alex Aiken. "Beyond Data and Model Parallelism for Deep Neural Networks." Proceedings of the 2nd Conference on Systems and Machine Learning (SysML\'19). 2019.',
  'Le14': 'Mehrdad Vu Le, Zhendong Afshari, Su. "Compiler validation via equivalence modulo inputs." ACM SIGPLAN Conference on Programming Language Design and Implementation. 2014. June 09 -11, 2014. [Link](https://doi.org/10.1145/2594291.2594334)',
  'Li16': 'Chao Li, Yi Yang, Min Feng, Srimat Chakradhar, Huiyang Zhou. "Optimizing memory efficiency for deep convolutional neural networks on GPUs." Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. 2016.',
  'Mas87': 'Henry Massalin. "Superoptimizer: a look at the smallest program." ACM SIGARCH Computer Architecture News. 1987.',
  'Mir17': 'Azalia Mirhoseini, Hieu Pham, Quoc V Le, Benoit Steiner, Rasmus Larsen, Yuefeng Zhou, Naveen Kumar, Mohammad Norouzi, Samy Bengio, and Jeff Dean. "Device Placement Optimization with Reinforcement Learning." 2017.',
  'Mkl16': 'Mkldnn. "Intel Math Kernel Library for Deep Neural Networks." 2016. [Link](https://01.org/mkl-dnn)',
  'Nai10': 'Vinod Nair, Geoffrey E Hinton. "Rectified Linear Units Improve Restricted Boltzmann Machines." Proceedings of the 27th International Conference on International Conference on Machine Learning (ICML\'10). 2010. [Link](http://dl.acm.org/citation.cfm?id=3104322.3104425)',
  'Nec00': 'George C Necula. "Translation validation for an optimizing compiler." Proceedings of the 2000 ACM SIGPLAN Conference on Programming Language Design and Implementation (PLDI), Vancouver. 2000. June 18-21, 2000. [Link](https://doi.org/10.1145/349299.349314)',
  'Pnu98': 'Amir Pnueli, Michael Siegel, Eli Singerman. "Translation Validation." Tools and Algorithms for Construction and Analysis of Systems, 4th International Conference, TACAS \'98, Held as Part of the European Joint Conferences on the Theory and Practice of Software, ETAPS\'98. 1998. March 28 -April 4, 1998. [Link](https://doi.org/10.1007/BFb0054170)',
  'PyT17': 'Pytorch. "Tensors and Dynamic neural networks in Python with strong GPU acceleration." 2017. [Link](https://pytorch.org)',
  'Rus15': 'Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael Bernstein, Alexander C Berg, Li Fei-Fei. "ImageNet Large Scale Visual Recognition Challenge." International Journal of Computer Vision. 2015. [Link](https://doi.org/10.1007/s11263-015-0816-y)',
  'Sha13': 'Rahul Sharma, Eric Schkufza, Berkeley R Churchill, Alex Aiken. "Data-driven equivalence checking." Proceedings of the 2013 ACM SIGPLAN International Conference on Object Oriented Programming Systems Languages & Applications, OOPSLA 2013, part of SPLASH 2013. October 26-31, 2013. [Link](https://doi.org/10.1145/2509136.2509509)',
  'Siv19': 'Muthian Sivathanu, Tapan Chugh, Sanjay S Singapuram, Lidong Zhou. "Astra: Exploiting Predictability to Optimize Deep Learning." Proceedings of the Twenty-Fourth International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS \'19). 2019.',
  'Tat11': 'Ross Tate, Michael Stepp, Zachary Tatlock, Sorin Lerner. "Equality Saturation: A New Approach to Optimization." Logical Methods in Computer Science. 2011. [Link](https://doi.org/10.2168/LMCS-7(1:10)2011)',
  'Ten17a': 'Tensorrt. "NVIDIA TensorRT: Programmable Inference Accelerator." 2017. [Link](https://developer.nvidia.com/tensorrt)',
  'Wu18': 'Shuang Wu, Guoqi Li, Feng Chen, Luping Shi. "Training and Inference with Integers in Deep Neural Networks." International Conference on Learning Representations. 2018.',
  'Xie16': 'Saining Xie, Ross B Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He. "Aggregated Residual Transformations for Deep Neural Networks." 2016.',
  'Zop16': 'Barret Zoph, V Quoc, Le. "Neural Architecture Search with Reinforcement Learning." 2016.',
  'Zop18': 'Barret Zoph, Vijay Vasudevan, Jonathon Shlens, Quoc V Le. "Learning transferable architectures for scalable image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.',
}

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
      description: 'Notes on visual novels, self-directed study, and papers',
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
          { text: 'papers', link: '/papers/taso/', activeMatch: '^/papers/' },
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
          {
            type: 'doc',
            title: 'papers',
            dir: 'papers',
            linkPrefix: '/papers/',
            sidebar: [
              {
                text: '论文',
                collapsed: false,
                items: ['taso'],
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
          { text: 'papers', link: '/en/papers/taso/', activeMatch: '^/en/papers/' },
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
          {
            type: 'doc',
            title: 'papers',
            dir: 'papers',
            linkPrefix: '/papers/',
            sidebar: [
              {
                text: 'Papers',
                collapsed: false,
                items: ['taso'],
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
    codeHighlighter: {
      langs: [pseudocodeLanguage],
      renderIndentGuides: true,
      colorizedBrackets: true
    },
    markdown: { abbr: paperAbbreviations, annotation: true },
  }),

  bundler: viteBundler(),
})
