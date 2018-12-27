# Machine Leanring 环境准备——搭建 Jupyter Notebook

## 目的

用 Jupyter Notebook 搭建一套 Machine Learning 学习环境

## 步骤

1. 安装 Python，建议 Python 3

   安装地址：https://www.python.org/

   过程略

2. 安装 pip

   ```shell
   $ curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   $ python get-pip.py
   ```

3. 安装机器学习库

   ```shell
   $ pip3 install --upgrade jupyter matplotlib numpy pandas scipy scikit-learn
   Collecting jupyter
     Downloading jupyter-1.0.0-py2.py3-none-any.whl
   Collecting matplotlib
     [...]
   ```

4. 验证，如果安装成功，则不会输出任何内容

   ```shell
   $ python3 -c "import jupyter, matplotlib, numpy, pandas, scipy, sklearn"
   ```

5. 打开 jupyter notebook，下面命令会拉起一个 web server，用浏览器打开 http://localhost:8888 便可以看到 jupyter notebook 页面，一般会自动打开

   ```shell
   $ jupyter notebook
   ```

## 随便逛逛

在 jupyter notebook 页面上点击按钮 New，选择 Python 版本，即可进入一个交互式 Python 环境

![](https://github.com/jieniu/articles/blob/master/pics/jupyter_new.png?raw=true)

打开交互式环境后，你可以给这个 notebook 起一个名字，如下图的第1步，我给这个 Nodebook 起了个 Housing 的名字；接着，可以在下面的命令行（cell）敲入简单的 Python 命令 `print("Hello world!")`，试着运行一下这条命令吧

![](https://github.com/jieniu/articles/blob/master/pics/jupyter_start.png?raw=true)

查看你运行 jupyter 的目录，你会看到多出了一个 Housing.ipynb 的文件，这就是 jupyter 给你创建的 notebook。