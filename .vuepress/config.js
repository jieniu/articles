module.exports = {
  title: '程序员在深圳',
  description: '一个程序员的工作学习日志',
  head: [
    ['link', { rel: 'icon', href: '/favicon.ico' }],
	['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css' }],
	['link', { rel: "stylesheet", href: "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.min.css" }],
  ],
  markdown: {
    lineNumbers: true // 代码块显示行号
  },
  themeConfig: {
    sidebarDepth: 2, // e'b将同时提取markdown中h2 和 h3 标题，显示在侧边栏上。
    lastUpdated: 'Last Updated', // 文档更新时间：每个文件git最后提交的时间
    nav:[
      { text: 'C++', link: '/cpp/' }, 
      { text: 'Java', link: '/java/' }, 
      { text: 'AI', link: '/AI/' }, 
      { text: 'math', link: '/math/' }, 
      { text: 'mysql', link: '/mysql_notes/' }, 
      { text: 'tools', link: '/tools/' }, 
      // 下拉列表
      {
        text: 'GitHub',
        items: [
          { text: 'GitHub地址', link: 'https://github.com/jieniu' },
          { text: 'MyLeeCode', link: 'https://github.com/jieniu/LeetCode.git' }
        ]
      }        
    ],
    sidebar: {
      '/cpp/': [
        'assignment_to_self_in_assignment_operator',
        'exception_and_destructor',
        'RAII',
        'static_initialization_fiasco',
        'virtual_destructor',
        'virtual_function_and_constructor'
      ],
      '/java/': [
        'understanding_collections_threadsafe',
        'spring_boot_thread_pool_timer',
        'powermock_and_unittest'
      ],
      '/AI/': [
        'ready_for_machine_learning'
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
  markdown: {
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
