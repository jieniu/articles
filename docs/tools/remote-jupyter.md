# 远程 Jupyter Notebook/Lab

今天在阿里云 ECS 上配置了一个 Jupyter Server，这样我的 iPad 就可以随时随地利用该工具写文章，撸 Python 代码了（前提是有网络的情况下）。

最简单的安装和管理 Jupyter 的方式应该数 [Anaconda](https://www.anaconda.com/) 了，这里以 Linux 操作系统为例，Python 版本为 3.7：

```bash
$ wget https://repo.anaconda.com/archive/Anaconda3-2019.03-Linux-x86_64.sh
$ sh ./Anaconda3-2019.03-Linux-x86_64.sh
```

安装完成后，你需要产生 jupyter notebook 的配置文件

```bash
$ jupyter notebook --generate-config
```

该配置文件存放在 `~/.jupyter/jupyter_notebook_config.py` 下，配置该文件

```
# 重要，允许任何源访问该服务
c.NotebookApp.allow_origin = '*'
# 第一次登陆时不弹出修改密码的提示
c.NotebookApp.allow_password_change = False
# https 所需签名
c.NotebookApp.certfile = '/your/home/.jupyter/mycert.pem'
c.NotebookApp.keyfile = '/your/home/.jupyter/mykey.key'
# 监听 ip 和 port
c.NotebookApp.ip = '127.0.0.1'
c.NotebookApp.port = 8889
# 启动时不打开浏览器
c.NotebookApp.open_browser = False
```

其中 `allow_origin` 如果不设置，会报 `Blocking Cross Origin API request for /api/contents` 错误。`certfile` 和 `keyfile` 是 https 协议的签名文件路径，详见下文。

开启远程服务，需设置登录密码：

```bash
$ jupyter notebook password
Enter password:  ****
Verify password: ****
```

该密码会记录在 `~/.jupyter/jupyter_notebook_config.json` 文件中，当然不可能是明文形式。

接下来我们来设置 https，这样所有交互数据便是加密的

```bash
$ cd ~/.jupyter
$ openssl req -x509 -nodes -days 3650 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
$ jupyter notebook --certfile=mycert.pem --keyfile mykey.key
```

上面的 `openssl` 命令会产生签名文件，该证书的过期时间是 10 年 (3650天) 后。

接下来，我们设置开机启动 jupyter lab，修改 `/etc/rc.local` 文件，添加以下行：

```
jupyter lab --allow-root --no-browser --notebook-dir=/your/workspace/ > /var/log/jupyter.log 2>&1 &
```

注意修改上面的 `/your/workspace` 为你自己的工作路径，至此，Jupyter 部分就设置完毕了，我们同样可以执行上面的命令将其启动

```bash
$ jupyter lab --allow-root --no-browser --notebook-dir=/your/workspace/ > /var/log/jupyter.log 2>&1 &
$ cat /var/log/jupyter.log
[I 16:50:56.326 LabApp] JupyterLab extension loaded from /anaconda3/lib/python3.7/site-packages/jupyterlab
[I 16:50:56.326 LabApp] JupyterLab application directory is /anaconda3/share/jupyter/lab
[I 16:50:56.328 LabApp] Serving notebooks from local directory: /your/workspace/
[I 16:50:56.328 LabApp] The Jupyter Notebook is running at:
[I 16:50:56.328 LabApp] https://127.0.0.1:8889/
[I 16:50:56.328 LabApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
```

最后，我们设置一下 nginx 反向代理：

```
map $http_upgrade $connection_upgrade {
        default upgrade;
        ''      close;
}
server {
        listen 8888;
        listen [::]:8888;
        ssl                  on;
        ssl_certificate      /your/home/.jupyter/mycert.pem;
        ssl_certificate_key  /your/home/.jupyter/mykey.key;
        location / {
                proxy_redirect     off;
                proxy_pass https://127.0.0.1:8889;
                proxy_http_version    1.1;
                proxy_set_header      Upgrade $http_upgrade;
                proxy_set_header      Connection $connection_upgrade;
        }
}
```

之后你就可以在浏览器中使用 `https://hostname:8888` 来访问 Jupyter Lab 了。当然，最后的最后，记得设置防火墙，例如打开本文中的 `8888` 端口。



参考：

* [Running a notebook server](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html)
* [https://www.cnblogs.com/students/p/10755400.html](https://www.cnblogs.com/students/p/10755400.html)

