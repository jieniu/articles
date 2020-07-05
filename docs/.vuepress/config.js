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
    }],
    [
      'vuepress-plugin-comment',
      {
        choosen: 'valine',
        // options选项中的所有参数，会传给Valine的配置
        options: {
          el: '#valine-vuepress-comment',
          appId: 'NSVHICXSy81REAUABkt1jprV-gzGzoHsz',
          appKey: 'gpzYqs93NamO3WaovdVn91s0'
        }
      }
    ]
  ],
  themeConfig: {
    repo: 'https://github.com/jieniu/articles',
    docsDir: 'docs',
    editLinkText: '在 Github 上编辑此页',
    repoLabel: 'Github',
    // 默认为 true，设置为 false 来禁用
    editLinks: true,
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
    sidebarDepth: 2 // e'b将同时提取markdown中h2 和 h3 标题，显示在侧边栏上。

  },
    permalink: "/:regular",
  extendMarkdown(md) {
      md.set({html: true})
      md.use(require("markdown-it-katex"))
  }
}
