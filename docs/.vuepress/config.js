module.exports = {
  title: '程序员在深圳',
  description: '一个程序员的工作学习日志',
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css' }],
    ['link', { rel: "stylesheet", href: "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.min.css" }],
  ],
  serviceWorker: { // 更新缓存网页提示及按钮名称。
    updatePopup: {
      message: "发现新内容可用",
      buttonText: "刷新"
    }
  },
  plugins: [
    ['@vuepress/back-to-top', true],
    ['@vuepress/blog'],
    ['@vuepress/last-updated'],
    ['@vuepress/google-analytics', {
      'ga': 'UA-86907063-2'
    }]
  ],
  themeConfig: {
    repo: 'https://github.com/jieniu/articles',
    docsDir: 'docs',
    editLinkText: '在 Github 上编辑此页',
    repoLabel: 'Github',
    // 默认为 true，设置为 false 来禁用
    editLinks: true,
    sidebarDepth: 2, // e'b将同时提取markdown中h2 和 h3 标题，显示在侧边栏上。
    lastUpdated: '上次更新', // string | boolean
    searchMaxSuggestions: 6,
    nav:[
      { text: 'C++', link: '/cpp/' }, 
      { text: 'Java', link: '/java/' }, 
      { text: 'AI', link: '/AI/'},
      { text: 'math', link: '/math/' }, 
      { text: 'mysql', link: '/mysql_notes/' }, 
      { text: 'tools', link: '/tools/' }, 
      { text: 'LeetCode', 
        items: [
          {text: 'articles', link: '/leetcode/'},
          { text: 'MyLeeCode', link: 'https://github.com/jieniu/LeetCode.git' }
        ]
      }
    ],
    sidebar: {
      '/cpp/': [
        'const',
        'compiler_generated_function',
        'logic_constness_and_bitwise_constness',
        'disallow_functions',
        'assignment_to_self_in_assignment_operator',
        'exception_and_destructor',
        'RAII',
        'static_initialization_fiasco',
        'virtual_destructor',
        'virtual_function_and_constructor',
        'struct-and-class',
        'resource-managing-class',
        'user-defined-type-conversion',
        'all-castings-considered',
        'inheritance_public_protect_private'
      ],
      '/java/': [
        'understanding_collections_threadsafe',
        'spring_boot_thread_pool_timer',
        'powermock_and_unittest'
      ],
      '/AI/': [
        'ready_for_machine_learning',
        'cross-validation',
        'confusion-matrix',
        'bias-variance',
        'cart1',
        'cart2',
        'rforest',
        'adaboost',
        'lr',
        'gbdt',
        'gbdt2',
        'linear-scratch',
        'softmax',
        'dnn',
        'homl-ch2',
        'homl-ch3'
      ],
      '/math/': [
        'linear_transformation'
      ],
      '/mysql_notes/': [
        '1.MySQL架构'
      ],
      '/tools/': [
        'vuepress_website',
        'python-virtualenv-commands',
        'virtualenv-migration',
        'vscode-remote',
        'remote-jupyter'
      ],
      '/leetcode/': [
        '1.two-sum',
        '2.add-two-numbers',
        '3.longest-substring',
        '4.median-of-two-sorted-arrays',
        '5.longest-palindromic-substring',
        '6.zigzag-conversion',
        '7.reverse-integer',
        '9.palindrome-number',
        '10.regular-expression-matching',
        '11.container-with-most-water'
      ]
    }
  },
    permalink: "/:regular",
//  markdown: {
 //     lineNumbers: true,
      // options for markdown-it-anchor
  //    anchor: { permalink: true },
      // options for markdown-it-toc
   //   toc: { includeLevel: [1,2] },
  //},
  extendMarkdown(md) {
      md.set({html: true})
      md.use(require("markdown-it-katex"))
  }
}
