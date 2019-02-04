const moment = require('moment');
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
  markdown: {
    lineNumbers: true // 代码块显示行号
  },
  themeConfig: {
    repo: 'https://github.com/jieniu/articles',
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
      { text: 'algorithm', 
        items: [
          {text: 'articles', link: '/AI/'},
          { text: 'MyLeeCode', link: 'https://github.com/jieniu/LeetCode.git' }
        ]
      }, 
      { text: 'math', link: '/math/' }, 
      { text: 'mysql', link: '/mysql_notes/' }, 
      { text: 'tools', link: '/tools/' }, 
    ],
    sidebar: {
      '/cpp/': [
        'assignment_to_self_in_assignment_operator',
        'exception_and_destructor',
        'RAII',
        'static_initialization_fiasco',
        'virtual_destructor',
        'virtual_function_and_constructor',
        'struct-and-class'
      ],
      '/java/': [
        'understanding_collections_threadsafe',
        'spring_boot_thread_pool_timer',
        'powermock_and_unittest'
      ],
      '/algorithm/': [
        'ready_for_machine_learning',
        'reverse-integer'
      ],
      '/math/': [
        'linear_transformation'
      ],
      '/mysql_notes/': [
        '1.MySQL架构'
      ],
      '/tools/': [
        'vuepress_website'
      ],
    }
  },
  plugins: [
    ['@vuepress/back-to-top'],
    ['@vuepress/blog'],
    [
      '@vuepress/last-updated',
      {
        transformer: (timestamp, lang) => {
          // 不要忘了安装 moment
          const moment = require('moment')
          moment.locale(lang)
          return moment(timestamp).fromNow()
        }
      }
    ]
  ],
  permalink: "/:year/:month/:day/:slug",
  markdown: {
    lineNumbers: true,
      // options for markdown-it-anchor
      anchor: { permalink: true },
      // options for markdown-it-toc
      toc: { includeLevel: [1,2] },
      config: md => {
          // use more markdown-it plugins!
          md.set({html: true})
          md.use(require("markdown-it-katex"))
      }
  }
}
