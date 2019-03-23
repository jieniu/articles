# virtualenv 命令集

virtualenv 是 python 中用来隔离不同项目的利器，本篇的目的主要是收集相关命令用法，便于在下次使用时快速检索

**virtualenv**

```
# 安装
$ pip install virtualenv
# 激活虚拟环境
$ source my_project/bin/activate
# 退出虚拟环境
$ deactivate
# 输出当前依赖包
$ pip freeze > requirements.txt
# 安装依赖包
$ pip install -r requirements.txt
```

**virtualenvwrapper**

```
# 安装
$ pip install virtualenvwrapper
# 修改~/.zshrc
$ cat ~/.zshrc
if [ -f /usr/local/bin/virtualenvwrapper.sh ]; then
    export WORKON_HOME=$HOME/virtualenvs
    source /usr/local/bin/virtualenvwrapper.sh
fi
# 创建虚拟环境，创建成功后 $WORKON_HOME 下会多出一个文件夹
$ mkvirtualenv my_prj
# 使用虚拟环境
$ workon my_prj
# 退出虚拟环境
$ deactivate
# 删除
$ rmvirtualenv my_prj
# 列出所有环境
$ lsvirtualenv
# 进入到当前虚拟环境中
$ cdvirtualenv
# 进入到当前虚拟环境的 site-packages 中
$ cdsitepackages
```

**.gitignore**
```
*.py[cod]     # 将匹配 .pyc、.pyo 和 .pyd文件
__pycache__/  # 排除整个文件夹
```

参考：

* http://pythonguidecn.readthedocs.io/zh/latest/dev/virtualenvs.html