# VuePress 建站速记

持续写作已经有两年多了，19年年初再次折腾建立自己的站点，主要基于以下原因

1. 和其他网站比起来，简书是比较清爽的，编辑器也非常好用，但推荐的文章层次太低，不符合程序员的气质
2. 自己建站，但不想关注除了文章内容以外的事情，即写完 Markdown 文件，提交到 github 上，文章会自动展现到页面上——让写文章和写代码保持一致的流程，方便管理

按照这个思路，我们可以把这个流程想的更具体点，例如我现在要写一篇文章，步骤是这样的：

1. 打开自己喜欢的编辑器（例如 Typora ），打开本地存放文章的目录（Typora 中敲 `control + ⌘ + 3`），新建一个 Markdown 文件
2. 开始写作
3. 写完后，打开终端，执行 `git add` -> `git commit` -> `git push` ，将文章提交到 github
4. 远端的 ECS 发现 github 上具体的 repo 下有更新，便自动执行 `git pull` 进行同步，并将 Markdown 文件渲染成静态页面，之后通过浏览器访问你的站点，就可以看到这篇文章了

这种方法有一个非常好的地方，它将写作的行为和写代码的行为保持一致，都是在本地完成，然后通过 git 进行管理，且无需担心备份、迁移等问题，这有效的减少了额外能量的消耗，能让我更快的进入写作状态，于是就可以把这个事情长期做下去。

用 github 管理自己的文章已有一段时间，但一直没有动手搭建远程站点，直到无意中看到了 VuePress 这个开源的软件，它可以很容易将 Markdown 渲染成 html 页面，同时我被它的简洁和专为技术站点打造所吸引，决定花点时间尝试一下。

大概折腾了一天，就把自己想要的效果实现了，如果不踩坑，可能不需要这么久。网上介绍 VuePress 的文章已经很详细了，这里就不重复了，有兴趣的同学可以查看文章最后的参考链接。

这里我简单讲下作为一个不懂前端的同学，我有哪些疑惑、它们是如何解除的；以及自己踩了什么坑、怎么爬出来的。

在使用 VuePress 中，我最大的疑惑是如何成功添加一个 Markdown 插件，它的步骤是什么，怎么知道引入的插件是否生效。经过了一番尝试后，总结为 3 个步骤（意味着你脑海中的额外的想法都是多余的）：

1. 安装插件
2. 将插件添加到配置中
3. 重新构建：`yarn docs:build`

拿数学公式插件 `markdown-it-katex` 作为例子，首先通过以下命令安装该插件

```
$ yarn add markdown-it-katex
```

修改 `.vuepress/config.js`  下的配置，添加一行 `md.use(require("markdown-it-katex"))`，如下：

```js
module.exports = {
  markdown: {
    config: md => {
      md.set({html: true})
      md.use(require("markdown-it-katex"))
    }
  }
}
```

构建后，一般的插件安装到此就完成了，对于 `markdown-it-katex` 来说，你还需要修改 `head` 项，依旧是 `.vuepress/config.js` 文件，在 `head` 中添加如下两行：

```js
module.exports = {
  head: [
    // ...
    ['link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.7.1/katex.min.css' }],
    ['link', { rel: "stylesheet", href: "https://cdnjs.cloudflare.com/ajax/libs/github-markdown-css/2.10.0/github-markdown.min.css" }]
  ]
  // ...
}
```

大功告成，现在 VuePress 可以识别出你文章中的数学公式了，如下所示：
$$
z = \sqrt{x^2 + y^2}
$$
最后，说下自己的采坑经历，我在修改完以上配置后，发现文章中的数学公式还是不能正常显示，于是各种 google、global 安装等，最后发现是自己写错了配置：把配置中的 `markdown` 对象写到 `themeConfig` 中了，如下

```js
  // ...
  themeConfig: {
    markdown: {
      // ...
    }
  }
```

这种错很难发现，花了不少时间，希望写出来，能帮犯同样错误的同学节省时间。



参考：

* [VuePress官网文档](https://vuepress.vuejs.org/guide/markdown.html#import-code-snippets)
* [如何使用 VuePress 编写静态博客](https://www.unaxu.com/blog/posts/005-one-how-to-generate-static-blog-with-vuepress.html)
* [VuePress 快速踩坑](https://zhuanlan.zhihu.com/p/36116211)
* [VuePress 手摸手教你搭建一个类Vue文档风格的技术文档/博客](https://segmentfault.com/a/1190000016333850)