# 一文读懂神经网络

要说近几年最引人注目的技术，无疑的，非人工智能莫属。无论你是否身处科技互联网行业，随处可见人工智能的身影：从 AlphaGo 击败世界围棋冠军，到无人驾驶概念的兴起，再到科技巨头 All in AI，以及各大高校向社会输送海量的人工智能专业的毕业生。以至于人们开始萌生一个想法：新的革命就要来了，我们的世界将再次发生一次巨变；而后开始焦虑：我的工作是否会被机器取代？我该如何才能抓住这次革命？

人工智能背后的核心技术是深度神经网络（Deep Neural Network），大概是一年前这个时候，我正在回老家的高铁上学习 [3Blue1Brown 的 Neural Network](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi) 系列视频课程，短短 4 集 60 多分钟的时间，就把神经网络从 High Level 到推导细节说得清清楚楚，当时的我除了获得新知的兴奋之外，还有一点新的认知，算是给头脑中的革命性的技术泼了盆冷水：神经网络可以解决一些复杂的、以前很难通过写程序来完成的任务——例如图像、语音识别等，但它的实现机制告诉我，神经网络依然没有达到生物级别的智能，短期内期待它来取代人也是不可能的。

一年后的今天，依然在这个春运的时间点，将我对神经网络的理解写下来，算是对这部分知识的一个学习笔记，运气好的话，还可以让不了解神经网络的同学了解起来。

## 什么是神经网络

维基百科这样解释[神经网络](https://zh.wikipedia.org/wiki/%E4%BA%BA%E5%B7%A5%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)：

> 现代神经网络是一种**非线性统计性**数据建模工具，神经网络通常是通过一个基于数学统计学类型的学习方法（Learning Method）得以优化，所以也是数学统计学方法的一种实际应用。

这个定义比较宽泛，你甚至还可以用它来定义其它的机器学习算法，例如之前我们一起学习的逻辑回归和 GBDT 决策树。下面我们具体一点，下图是一个逻辑回归的示意图：

![lr & nn](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/lr_nn.png?raw=true)

其中 x1 和 x2 表示输入，w1 和 w2 是模型的参数，z 是一个线性函数：
$$
z = w_1x_1 + w_2x_2 + b
$$
接着我们对 z 做一个 sigmod 变换（图中蓝色圆），得到输出 y：
$$
y = \sigma(z)
$$
其实，上面的逻辑回归就可以看成是一个只有 1 层**输入层**， 1 层**输出层**的神经网络，图中容纳数字的圈儿被称作**神经元**；其中，层与层之间的连接 w1、w2 以及 b，是这个**神经网络的参数**，层之间如果每个神经元之间都保持着连接，这样的层被称为**全连接层**（Full Connection Layer），或**稠密层**（Dense Layer）；此外，sigmoid 函数又被称作**激活函数**（Activation Function），除了 sigmoid 外，常用的激活函数还有 ReLU、tanh 函数等，这些函数都起到将线性函数进行非线性变换的作用。我们还剩下一个重要的概念：**隐藏层**，它需要把 2 个以上的逻辑回归叠加起来加以说明：

![multi_layer](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/multi_layer.png?raw=true)

如上图所示，除输入层和输出层以外，其他的层都叫做**隐藏层**。如果我们多叠加几层，这个神经网络又可以被称作**深度神经网络**（Deep Neural Network），有同学可能会问多少层才算“深”呢？这个没有绝对的定论，个人认为 3 层以上就算吧：）

以上，便是神经网络，以及神经网络中包含的概念，可见，神经网络并不特别，广义上讲，它就是

> 一个非线性函数，或把几个非线性函数的输入输出接起来，形成一个更大的非线性函数。

可见，神经网络和人脑神经也没有任何关联，如果我们说起它的另一个名字——**多层感知机（Mutilayer Perceptron）**，就更不会觉得有多么玄乎了，多层感知机创造于 80 年代，可为什么直到 30 年后的今天才爆发呢？你想得没错，因为改了个名字……开个玩笑；实际上深度学习这项技术也经历过很长一段时间的黑暗低谷期，直到人们开始利用 GPU 来极大的提升训练模型的速度，以及几个标志性的事件：如 AlphaGo战胜李世石、Google 开源 TensorFlow 框架等等，感兴趣的同学可以翻一下这里的历史。

## 为什么需要神经网络

就拿上图中的 3 个逻辑回归组成的神经网络作为例子，它和普通的逻辑回归比起来，有什么优势呢？我们先来看下单逻辑回归有什么劣势，对于某些情况来说，逻辑回归可能永远无法使其分类，如下面数据：

| label   | x1   | x2   |
| ------- | ---- | ---- |
| class 1 | 0    | 0    |
| class 1 | 1    | 1    |
| class 2 | 0    | 1    |
| class 2 | 1    | 0    |

这 4 个样本画在坐标系中如下图所示

![lr_limit](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/lr_limit.jpg?raw=true)

因为逻辑回归的决策边界（Decision Boundary）是一条直线，所以上图中的两个分类，无论你怎么做，都无法找到一条直线将它们分开，但如果借助神经网络，就可以做到这一点。

由 3 个逻辑回归组成的网络（这里先忽略 bias）如下：

![detail_nn](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/detail_nn.png?raw=true)

观察整个网络的计算过程，在进入输出层之前，该网络所做的计算实际上是：
$$
\left[\begin{matrix}x_1'\\ x_2'\end{matrix}\right]
= \sigma(
\left[
\begin{matrix}
w_{11}\quad w_{12}\\
w_{21}\quad w_{22}
\end{matrix}\right]
\left[
\begin{matrix}
x_1\\x_2
\end{matrix}
\right])
$$
即把输入先做了一次线性变换（Linear Transformation），得到 `[z1, z2]`，再把 `[z1, z2]` 做了一个非线性变换（sigmoid），得到 `[x1', x2']`，（线性变换的概念可以参考[这个视频](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3)）。从这里开始，后面的操作就和一个普通的逻辑回归没有任何差别了，所以它们的差异在于：**我们的数据在输入到模型之前，先做了一层特征变换处理（Feature Transformation，有时又叫做特征抽取 Feature Extraction），使之前不可能被分类的数据变得可以分类了**。

我们继续来看下特征变换的效果，假设 $\left[\begin{matrix}w_{11}\quad w_{12}\\w_{21} \quad w_{22}\end{matrix}\right]$ 为 $\left[\begin{matrix}1\qquad 1\\-0.5 -0.5\end{matrix}\right]$，带入上述公式，算出 4 个样本对应的 `[x1', x2']` 如下：

| label   | x1   | x2   | x1'  | x2'  |
| ------- | ---- | ---- | ---- | ---- |
| class 1 | 0    | 0    | 0.5  | 0.5  |
| class 1 | 1    | 1    | 0.88 | 0.27 |
| class 2 | 0    | 1    | 0.73 | 0.38 |
| class 2 | 1    | 0    | 0.73 | 0.38 |

再将变换后的 4 个点绘制在坐标系中： 

![feature_trans](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/feature_trans.jpg?raw=true)

显然，在做了特征变换之后，这两个分类就可以很容易的被一条决策边界分开了。

所以，**神经网络的优势在于，它可以帮助我们自动的完成特征变换或特征提取**，尤其对于声音、图像等复杂问题，因为在面对这些问题时，人们很难清晰明确的告诉你，哪些特征是有用的。

在解决特征变换的同时，神经网络也引入了新的问题，就是我们需要设计各式各样的网络结构来针对性的应对不同的场景，例如使用卷积神经网络（CNN）来处理图像、使用长短期记忆网络（LSTM）来处理序列问题、使用生成式对抗网络（GAN）来写诗和作图等，就连去年自然语言处理（NLP）中取得突破性进展的 Transformer/Bert 也是一种特定的网络结构。所以，**学好神经网络，对理解其他更高级的网络结构也是有帮助的**。

## 神经网络是如何工作的

上面说了，神经网络可以看作一个非线性函数，该函数的参数是连接神经元的所有的 Weights 和 Biases，该函数可以简写为 `f(W, B)`，以手写数字识别的任务作为例子：识别 [MNIST 数据集](http://yann.lecun.com/exdb/mnist/)中的数字，数据集（MNIST 数据集是深度学习中的 HelloWorld）包含上万张不同的人写的数字图片，共有 0-9 十种数字，每张图片为 `28*28=784` 个像素，我们设计一个这样的网络来完成该任务：

* 输入层可容纳一张图片的所有像素，总共有 784 个神经元

* 使用 1 个隐藏层，含有 16 个神经元，那么
  * 输入层到隐藏层的参数个数为 `784*16=12544`，Bias 的个数为 16
* 输出层为 10 个神经元，分别表示 0-9 这十种情况
  * 隐藏层到输出层的参数个数为 `16*10=160`，Bias 个数为 10

* 总的 Weights 和 Biases 加起来有 `12544+16+160+10 = 12730` 个

把该网络函数所具备的属性补齐：

```
参数：12730 个 Weights 和 Biases
输入：一张 28*28 的手写数字图片
输出：0-9 这 10 个数的可能性
```

接下来的问题是，这个函数是如何产生的？这个问题本质上问的是这些参数的值是怎么确定的。

在机器学习中，有另一个函数 c 来衡量 f 的好坏，c 的参数是一堆数据集，你输入给 c 一批 Weights 和 Biases，c 输出 Bad 或 Good，当结果是 Bad 时，你需要继续调整 f 的 Weights 和 Biases，再次输入给 c，如此往复，直到 c 给出 Good 为止，这个 c 就是损失函数 Cost Function（或 Loss Function）。在手写数字识别的列子中，c 可以描述如下：

```
参数：上万张手写数字图片
输入：f 的 Weights 和 Biases
输出：一个数字，衡量分类任务的好坏，该数越小越好
```

可见，要完成手写数字识别任务，只需要调整这 12730 个参数，让损失函数输出一个足够小的值即可，推而广之，绝大部分神经网络、机器学习的问题，都可以看成是定义损失函数、以及参数调优的问题。

在手写识别任务中，我们既可以使用交叉熵（Cross Entropy）损失函数，也可以使用 MSE（Mean Squared Error）作为损失函数，接下来，就剩下如何调优参数了。

神经网络的参数调优也没有使用特别的技术，依然是大家刚接触机器学习，就学到的梯度下降算法，梯度下降解决了上面迭代过程中的遗留问题——当损失函数给出 Bad 结果时，如何调整参数，能让 Loss 减少得最快。

梯度可以理解为：

>考虑一座山，山上的点 (x,y) 的高度用 H(x,y) 表示。这一点的梯度是在该点坡度（或者说斜度）最陡的方向。梯度的大小告诉我们坡度到底有多陡。——[wiki/梯度](https://zh.wikipedia.org/wiki/%E6%A2%AF%E5%BA%A6)

具体一点，梯度就是该点 (x,y) 在每个方向的斜率（偏微分）所组成的向量：
$$
\nabla H(x, y) = \left[\frac{\partial H}{\partial x}, \frac{\partial H}{\partial y}\right]^\top
$$
把 Loss 对应到 H，12730 个参数对应到 (x,y)，则 Loss 对所有参数的梯度可以表示为下面向量，该向量的长度为 12730：
$$
\nabla L(w,b) = \left[

\frac{\partial L}{\partial w_1},
\frac{\partial L}{\partial w_2},...,
\frac{\partial L}{\partial b_{26}}

\right] ^\top
$$
有了梯度之后，沿着梯度来调整参数，就是使 Loss 减少得最快的方式。所以，每次迭代过程可以概括为

1. 向损失函数中输入模型参数
2. 通过模型参数，结合上万条样本，计算 Loss
3. 根据 Loss 来计算所有这些参数的梯度
4. 根据梯度来调整参数

用梯度来调整参数的式子如下（为了简化，这里省略了 bias）：
$$
w = w - \eta \nabla L(w)
$$
上式中，$\eta$ 是学习率，意为每次朝下降最快的方向前进一小步，避免优化过头（Overshoot）。

由于神经网络参数繁多，所以需要更高效的计算梯度的算法，于是，反向传播算法（Backpropagation）呼之欲出。

## 反向传播算法

在学习反向传播算法之前，我们先复习一下微积分中的链式法则（Chain Rule）：设 `g = u(h)`，`h = f(x)` 是两个可导函数，x 的一个很小的变化 △x 会使 h 产生一个很小的变化 △h，从而 g 也产生一个较小的变化 △g，现要求 △g/△x，可以使用链式法则：
$$
\frac{dg}{dx} = \frac{dg}{dh}\frac{dh}{dx}
$$
有了以上基础，理解反向传播算法就简单了。

假设我们的演示网络只有 2 层，输入输出都只有 2 个神经元，如下图所示：

![bp](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/bp.png?raw=true)

其中 $[x_1, x_2]$ 是输入，$[a_1^{(2)}, a_2^{(2)}]$ 是输出，$[y_1,y_2]$ 是样本的目标值，这里使用的损失函数 L 为 MSE；图中的上标 (1) 或 (2) 分别表示参数属于第 (1) 层或第 (2) 层，下标 1 或 2 分别表示该层的第 1 或 第 2 个神经元。

现在我们来计算 $\partial L/\partial w^{(2)}_1$ 和  $\partial L/\partial w^{(1)}_1$，掌握了这 2 个参数的偏导数计算之后，整个梯度的计算就掌握了。

所谓反向传播算法，指的是从右向左来计算每个参数的偏导数，先计算 $\partial L/\partial w^{(2)}_1$ ，根据链式法则
$$
\frac{\partial L}{\partial w^{(2)}_1} = 
\frac{\partial L}{\partial z^{(2)}_1}
\frac{\partial z^{(2)}_1}{\partial w^{(2)}_1}
$$
对左边项用链式法则展开
$$
\frac{\partial L}{\partial z^{(2)}_1} = \frac{\partial L}{\partial a^{(2)}_1} \frac{\partial{a^{(2)}_1}}{\partial z^{(2)}_1}
$$
又 $a^{(2)}_1$ 是输出值，$\partial L/\partial a^{(2)}_1$ 可以直接通过 MSE 的导数算出：
$$
\frac{\partial L}{\partial a^{(2)}_1} = a^{(2)}_1 - y_1
$$
而 $a_1^{(2)} = \sigma (z^{(2)}_1)$，则 $\partial a^{(2)}_1/\partial z^{(2)}_1$ 就是 sigmoid 函数的导数在 $z^{(2)}_1$ 处的值，即
$$
\frac{\partial a^{(2)}_1}{\partial z^{(2)}_1} = \sigma'(z_1^{(2)})
$$
于是 $\partial L/\partial z^{(2)}_1$ 就算出来了：
$$
\frac{\partial L}{\partial z^{(2)}_1} = \sigma'(z_1^{(2)})(a^{(2)}_1 - y_1)
$$
再来看 $\partial z^{(2)}_1 / \partial w^{(2)}_1$ 这一项，因为
$$
z^{(2)}_1 = w_1^{(2)}a^{(1)}_1 + ... + b^{(2)}_1
$$
所以 
$$
\frac{\partial z^{(2)}_1}{\partial w^{(2)}_1} = a^{(1)}_1
$$
_注意：上面式子对于所有的 $z_i$ 和 $w_j$ 都成立，且结果非常直观，即 $z_i$ 对 $w_j$ 的偏导为左边的输入 $a_j$ 的大小；同时，这里还隐含着另一层意思：需要调整哪个 $w_j$ 来影响 $z_i$，才能使 Loss 下降得最快，从该式子可以看出，当然是先调整较大的 $a_j$ 值所对应的 $w_j$，效果才最显著_。

于是，最后一层参数 $w_1^{(2)} $ 的偏导数就算出来了
$$
\frac{\partial L}{\partial w^{(2)}_1} = a_1^{(1)}\sigma'(z_1^{(2)})(a^{(2)}_1 - y_1)
$$
我们再来算上一层的 $\partial L/ \partial w^{(1)}_1$，根据链式法则 ：
$$
\frac{\partial L}{\partial w^{(1)}_1} = 
\frac{\partial L}{\partial z^{(1)}_1}
\frac{\partial z^{(1)}_1}{\partial w^{(1)}_1}
$$
继续展开左边这一项
$$
\frac{\partial L}{\partial z^{(1)}_1} = \frac{\partial L}{\partial a^{(1)}_1}
\frac{\partial a^{(1)}_1}{\partial z^{(1)}_1}
$$
你发现没有，这几乎和计算最后一层一摸一样，但需要注意的是，这里的 $a^{(1)}_1$ 对 Loss 造成的影响有多条路径，于是对于只有 2 个输出的本例来说：
$$
\frac{\partial L}{\partial a^{(1)}_1} = \frac{\partial L}{\partial z^{(2)}_1}\frac{\partial z^{(2)}_1}{\partial a^{(1)}_1} + 
\frac{\partial L}{\partial z^{(2)}_2}\frac{\partial z^{(2)}_2}{\partial a^{(1)}_1}
$$
上式中，$\partial L/\partial z^{(2)}$ 都已经在最后一层算出，下面我们来看下 $\partial z^{(2)}/\partial a^{(1)}_1$，因为
$$
z^{(2)}_1 = w_1^{(2)}a^{(1)}_1 + ... + b^{(2)}_1
$$
于是
$$
\frac{\partial z^{(2)}_1}{\partial a^{(1)}_1} = w^{(2)}_1
$$
同理
$$
\frac{\partial z^{(2)}_2}{\partial a^{(1)}_1} = w^{(2)}_2
$$
_注意：这里也引申出梯度下降的调参直觉：即要使 Loss 下降得最快，优先调整 weight 值比较大的 weight。_

至此，$\partial L/\partial w_1^{(1)}$ 也算出来了
$$
\frac{\partial L}{\partial w^{(1)}_1} = x_1\sigma'(z^{(1)}_1)
(w^{(2)}_1 \frac{\partial L}{\partial z^{(2)}_1} + w^{(2)}_2\frac{\partial L}{\partial z^{(2)}_2})
$$
观察上式，**所谓每个参数的偏导数，通过反向传播算法，都可以转换成线性加权（Weighted Sum）计算**，归纳如下：
$$
\frac{\partial L}{\partial w^{(l)}_i} = a^{(l-1)}_i\sigma'(z^{(l)}_i)
\sum_{j=1}^{n}(w^{(l+1)}_j\frac{\partial L}{\partial z^{(l+1)}_j})
$$
式子中 n 代表分类数，(l) 表示第 l 层，i 表示第 l 层的第 i 个神经元。**既然反向传播就是一个线性加权，那整个神经网络就可以借助于 GPU 的矩阵并行计算了**。

最后，当你明白了神经网络的原理，是不是越发的认为，它就是在做一堆的微积分运算，当然，作为能证明一个人是否学过微积分，神经网络还是值得学一下的。Just kidding ..

## 小结

本文我们通过

* 什么是神经网络
* 为什么需要神经网络
* 神经网络的工作原理
* 反向传播算法

这四点，全面的学习了神经网络这个知识点，希望本文能给你带来帮助。



参考：

* [3Blue1Brown: Neural Network](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
* [3Blue1Brown: Linear Transformer](https://www.youtube.com/watch?v=kYB8IZa5AuE&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab&index=3)
* [反向传播算法](https://www.youtube.com/watch?v=ibJpTrp5mcE&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=13&t=0s)
* [wikipedia: 人工神经网络](https://zh.wikipedia.org/wiki/人工神经网络)

