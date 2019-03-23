# Python 工程管理及 virtualenv 的迁移

virtualenv 是管理 python 工程的利器，它可以很好的帮你维护项目中的依赖，使用 virtualenv，还能保持 global 库的干净、不会被不同项目中的第三方库所污染。

virtualenv 的默认功能简单好用，可一旦涉及到多人协作，或部署到不同的环境中时，错误的使用 virtualenv 会给你带来一些麻烦，从而你需要花很多时间在解决这些问题上。本文的目的就是总结过去使用 virtualenv 的经验，希望能帮你找到一种正确的打开方式。

首先，创建一个空的 virtualenv 时，你的目录中会包含以下文件和目录

```python
drwxr-xr-x   7 fengyajie  staff   224B Mar 21 22:49 .
drwxr-xr-x   8 fengyajie  staff   256B Mar 21 20:28 ..
lrwxr-xr-x   1 fengyajie  staff    83B Mar 21 22:49 .Python -> /usr/local/Cellar/...
drwxr-xr-x  16 fengyajie  staff   512B Mar 21 22:49 bin
drwxr-xr-x   3 fengyajie  staff    96B Mar 21 22:49 include
drwxr-xr-x   3 fengyajie  staff    96B Mar 21 22:49 lib
-rw-r--r--   1 fengyajie  staff    61B Mar 21 22:49 pip-selfcheck.json
```

接着当你执行 `source bin/activate` 后，你安装的依赖都会在 `lib` 目录下，这一点很诱人，会让你觉得一切尽在掌握，因为该应用程序所需要的一切库文件全在这个 app 的根目录下，所以当这个应用需要部署时，为了避免产生 `ImportError: No module named xxx` 错误，你会很容易的想到将本地这个 app 目录打包，然后放到远程服务器或容器中去执行。

当你这么做时，你会发现虽然在远程是可以执行 `source bin/activate` 命令以进入 virtualenv ，但此时你引用的 python 可执行文件却并不是 `${app}/bin/pyhton`，而是 global 环境中的那个 `/usr/bin/python`，所以 `${app}/lib` 下的所有依赖包路径仍然是没有被包含进 `sys.path` 的。

这时，你才发现自己的假设是错误的，并开始怀疑自己使用 virtualenv 的方式存在问题，于是便 google 各种解决方案，但项目已处于部署阶段，时间紧迫，你很可能找不到最优的办法，只能退而求其次，寻求次优解，毕竟依赖包都在嘛，改下 `sys.path` 不就好了嘛？确实很容易想到这种方法，但又不想手动改，那就写个程序改吧，也不难：

```python
# set_sys_path.py
def set_sys_path():
    import sys
    for path in sys.path:
        if os.path.split(path)[1] == 'site-packages':
            home = os.path.abspath(os.path.dirname(__file__))
            pypath = os.path.join(home, 'lib/python2.7')
            pypath_sitepackage = os.path.join(home, 'lib/python2.7/site-packages')
            pth = os.path.join(path, 'pth.pth')
            with open(pth, 'w') as f:
                    f.write("%s\n" % pypath)
                    f.write("%s\n" % pypath_sitepackage)

if __name__ == "__main__":
    set_sys_path()
```

上面的程序很简单，它将 `${app}/lib/python2.7` 和 `${app}/python2.7/site-packages` 两个依赖路径写到 `pth.pth` 文件中，并将该文件 `mv` 到 global 的 `site-packages` 目录下，这样当你启动 global 的 python 时，会自动将 `pth.pth` 里的路径添加到 `sys.path` 下，这样只需要在启动你的 app 之前，执行该脚本即可，如下：

```bash
$ python set_sys_path.py
$ python main.py
```

问题暂时解决了，这次你的 app 也顺利发布了；但还没结束，我们希望在测试机集群上把 app 的自动化测试做起来，在做自动化测试时，系统会随机给你分配一台机器资源，当测试完成后，资源会被回收。你心想，这仍然很简单嘛，本地测试已经覆盖得很全了，只要自动化系统利用 git 把代码拉下来，先执行 `set_sys_path.py` 设置 `sys.path`，再执行 `python test.py`（测试入口）就可以了。

可这时又出现问题了，自动化测试在执行 `set_sys_path.py` 时，报 `Permission denied` 错误，原因是测试机为了保持环境不被污染，不允许你将 `pth.pth` 复制到 global 的 `site-packages` 下。

遇到这个问题怎么办？其实也很容易解决：我们都知道 python 中有个环境变量 `PYTHONPATH` 可以用来设置 `sys.path`，既然没有写文件的权限，那定义环境变量总该可以吧：

```bash
$ export PYTHONPATH=$PYTHONPATH:${app}/lib/python2.7:${app}/lib/python2.7/site-packages
$ python main.py
```

果然可行，你再一次「顺利」的完成了需求。

经历过多次折腾后，我们发现这种使用 virtualenv 和修改 `sys.path` 的方法不算很好，还容易出错。于是开始思考最初的那个问题，virtualenv 该怎么迁移？有没有更好的办法？答案肯定是有的，在此之前，我们先仔细观察 virtualenv 产生的文件，会发现其中有 28 个软连接，它们的源文件均在 global 库中，如下所示

```shell
$ find . -type l
./.Python
./bin/python
./bin/python2
./include/python2.7
./lib/python2.7/lib-dynload
./lib/python2.7/encodings
...
```

所以，当你把整个 virtualenv 打包，放到另一个环境中运行时，肯定是会失败的，因为软连接失效了，于是，再一次证实这种把整个 virtualenv 打包的方法，实际上是错误的，virtualenv 就只是一个 local 方案，而不是让你可以「处处运行」的工具。

但 virtualenv 的隔离功能，可以让你只关注项目范围内的依赖包，所以我们可以利用 `pip freeze` 命令，将项目内的依赖保存到一个叫 `requirements.txt` 的文件中，这样在任何其他环境，我们只要根据 `requirements.txt` 文件来安装项目所需的依赖包，即可将本地的运行环境克隆出来，而且这种克隆出来的环境更纯粹，不会受到源环境或 global 库的影响，没有不确定性。下面我们用一个例子来具体说明下：

假设 Bob 和 Alice 同在一个团队，他们决定使用 python 来开发新项目，一开始，Bob 在 github 上创建了一个新 repo，并在本地初始化它：

```shell
# 从 github clone 项目
$ git clone https://github.com/your_group/your_repo.git
$ cd your_repo
# 创建并进入 virtualenv
$ virtualenv .
$ source bin/activate
# 修改 .gitignore，过滤掉 virtualenv 产出的文件
$ cat .gitignore
*.py[cod]
__pycache__/
.Python
bin/
include/
lib/
pip-selfcheck.json
# 在本地安装基本依赖，例如 Flask、gevent、gunicorn 等
$ pip install Flask gevent gunicorn -i https://pypi.mirrors.ustc.edu.cn/simple/
# 将本地依赖写入 requirements.txt
$ pip freeze > requirements.txt
# 将变更提交到 github
$ git add .
$ git commit -m "init project"
$ git push origin master
# 继续开发
# ...
```

Bob 完成了初始化，实际上他只提交了 `.gitignore` 和 `requirements.txt` 两个文件到 git 中，之后 Alice 也可以加入进来了：

```shell
# 从 github clone 项目
$ git clone https://github.com/your_group/your_repo.git
$ cd your_repo
# 创建并进入 virtualenv
$ virtualenv .
$ source bin/activate
# 根据 requirements.txt 文件下载项目所需的依赖
$ pip install Flask gevent gunicorn -i https://pypi.mirrors.ustc.edu.cn/simple/
# 继续开发
# ...
```

可以看到，通过这样的步骤，Bob 和 Alice 不仅有了一摸一样的开发环境，还能最小化 git 仓库的大小，且按照这样的思路，他们还可以把相同的环境克隆到测试机上，以及 Docker 镜像中。显然，这种一致性不仅可以提高开发效率，还可以提高后续的运维效率。

相关文章：

* [virtualenv 命令集](/tools/python-virtualenv-commands.md)



参考：

* [Pipenv & 虚拟环境](https://getpocket.com/redirect?url=https%3A%2F%2Fpythonguidecn.readthedocs.io%2Fzh%2Flatest%2Fdev%2Fvirtualenvs.html&formCheck=9831177a88141647c7d23b3d1995db4e)

