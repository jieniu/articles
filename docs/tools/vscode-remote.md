# VSCode 远程开发插件快速使用

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/architecture-ssh.png?raw=true)

今天试用了一下 VSCode 的新插件：远程开发插件，体验很不错，它可以：

1. 让你在本地打开远程开发机上的代码，并提供和本地一样的开发体验
2. 在 VSCode 中打开远程的终端
3. 在不同远程开发机上配置不同的插件，把插件装在远程的目的是让操作更流畅
4. 你还可以在本地调试远程代码：断点、单步等一样都不会少
5. 支持 [SSH 隧道 (SSH Tunnel)](https://www.ssh.com/ssh/tunneling/example) 的连接方式，这样你便可以在家里调试公司电脑上的代码了。
6. ……

本文将以 SSH 连接的方式，做一个入门介绍，更多高级功能还需你在使用过程中慢慢发掘。

## 配置步骤

1. 在 VSCode 扩展栏中搜索 `Remote - SSH` 插件，点击安装

   ![image-20190606233135823](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190606233135823.png?raw=true)

2. 使用 `ssh-keygen` 工具在本机创建 ssh 秘钥，创建时一路回车即可 (注意：Remote-SSH 插件不支持输入账号密码的连接方式，首选的登录方式为[使用非对称秘钥登录](https://www.ssh.com/ssh/public-key-authentication))

   ```bash
   $ ssh-keygen
   Enter file in which to save the key (/Users/fengyajie/.ssh/id_rsa): 
   Enter passphrase (empty for no passphrase):
   Enter same passphrase again:
   Your identification has been saved in /Users/fengyajie/.ssh/id_rsa.
   Your public key has been saved in /Users/fengyajie/.ssh/id_rsa.pub.
   The key fingerprint is:
   The key's randomart image is:
   +---[RSA 2048]----+
   |      .oo*++.+o++|
   |       +o Oo+ + +|
   |    . . o+ o   o |
   |.. o . +o .      |
   |o o o . S+       |
   |.o . .    .      |
   |+.E   . ..       |
   |o=.+ . . ..      |
   |.=*..   ..       |
   +----[SHA256]-----+
   ```

3. 将产生好的公钥发送到远端（使用 `ssh-copy-id` 工具），你需要将下面的 `user@your_remote_host` 修改为你开发机的用户名和主机名

   ```bash
   $ ssh-copy-id -i ~/.ssh/id_rsa.pub user@your_remote_host
   usr/bin/ssh-copy-id: INFO: Source of key(s) to be installed: "/Users/fengyajie/.ssh/id_rsa.pub"
   /usr/bin/ssh-copy-id: INFO: attempting to log in with the new key(s), to filter out any that are already installed
   /usr/bin/ssh-copy-id: INFO: 1 key(s) remain to be installed -- if you are prompted now it is to install the new keys
   
   Number of key(s) added:        1
   
   Now try logging into the machine, with:   "ssh 'user@your_remote_host'"
   and check to make sure that only the key(s) you wanted were added.
   ```

4. 试一下连接，正常情况下你现在就可以直接登录到远端机器了

   ```bash
   $ ssh 'user@your_remote_host'
   Welcome to Alibaba Cloud Elastic Compute Service !
   Last login: Thu Jun  6 20:32:13 2019 from $local_ip
   root@iZwz946zuZ:~#
   ```

## 连接远端机器

上面配置完成后，你就可以使用 VSCode 连接到远端机器了，打开 VSCode，敲入 `F1` 键，输入 `Remote-SSH:Connect to Host`，回车 

![image-20190606234544034](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190606234544034.png?raw=true)

接着输入你刚才配置好的 `user@your_remote_host`，回车

![image-20190606234722725](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190606234722725.png?raw=true)

此时 VSCode 会打开一个新的窗口，在这个窗口的左下角，你会观察到一个绿色的 SSH 状态条，表示此时你的 VSCode 已经连上了远程的开发机，如下

![image-20190606235035247](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190606235035247.png?raw=true)

接下来，你就可以打开左上角的文件管理侧边栏，点击 `Open Folder`，此时你会惊讶的发现，远端的 `home` 目录被列出来了：

![image-20190606235406685](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190606235406685.png?raw=true)

至此，你已经在本地开启了一个「远端开发环境」，接下来你的开发任务再也不受远程环境的限制了，能有这样的体验，还是要感谢一下微软公司的贡献，真的是一款良心之作。



以上仅只是入门介绍，可以让你快速的把这个插件用起来，更多高级功能，还是建议你去阅读[官方文档](https://code.visualstudio.com/docs/remote/ssh)中的内容。



参考：

1. [Remote Development using SSH](https://code.visualstudio.com/docs/remote/ssh)
2. [SSH PORT FORWARDING EXAMPLE](https://www.ssh.com/ssh/tunneling/example)
3. [PUBLIC KEY AUTHENTICATION FOR SSH](https://www.ssh.com/ssh/public-key-authentication)
