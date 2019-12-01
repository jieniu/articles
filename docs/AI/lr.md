## 深入理解逻辑回归算法（Logistic Regression）

在继续学习 GBDT（Gradient Boosting Dicision Tree） 决策树前，我们需要先来了解下逻辑回归算法（Logistic Regression），因为 GBDT 较为复杂，但在逻辑回归的基础上，理解起来会容易些。

逻辑回归是机器学习中最为基础的算法，也是工业界使用得最多的算法之一，究其原因，在于其简单、高效以及实用。

虽然线性回归也很简单，但却不实用，是因为逻辑回归本质上是一个概率模型，在实际应用中，预测一个 0-1 之间的概率值要比预测一个实数的场景要多得多，比如在广告业务中，我们往往求的是用户点击一条广告的概率。

逻辑回归是一个概率模型，但通过一定的转换，我们依然可以把该模型的预测范围从 0-1 转换到实数范围，所以它和线性回归都可以被归纳到「通用的线性模型」（Generalized Linear Model）中，要理解这种转换，我们需要引入一个概念：odds 和 log(odds)。

![glm](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/glm.png?raw=true)

## odds 和 log(odds)

odds 是几率、胜算的意思，据维基百科记载，这个概念主要在赌博和统计学领域中使用，且它的历史可以追溯到 16 世纪，早于概率论的发展时期。

odds 很容易理解，拿足球比赛作为例子，假设中国队打巴西队，中国队的赢面是 1，输面是 99，那么中国队赢的 odds 为 1/99，输的 odds 就是 99，odds 和概率的区别也很容易通过这个例子看出来，从概率的角度讲，中国队赢巴西队的概率为 0.01，输的概率为 0.99。

上面的例子还可以看出，中国队赢的 odds 和巴西队赢的 odds 的取值范围是不同的，中国队赢的 odds 的范围在 [0,1] 之间，而巴西队赢的范围为 [1,∞)；也就是说，中国队和巴西队比赛，这两者的输赢程度应该是相等的，但 1/99 和 99 这两个数，它们的尺度不同，就很难对此做出直观的判断；而 log(odds) 就是用来解决该问题的：

|           | 中国队赢 | 巴西队赢 |
| --------- | -------- | -------- |
| odds      | 1/99     | 99       |
| log(odds) | -4.60    | 4.60     |

可以看到，对 odds 加了 log 后，中国队赢和巴西队赢这两种情况的 log(odds) 的绝对值都是 4.6，即两者的输赢程度为 4.6；且当我们算赢面的 log(odds) 时，通过正负号就可以判断赢面多还是赢面少，如 -4.6 就表示中国队的赢面是少的；此外，当 log(odds) 为 0 时，赢面和输面一样多。

log(odds) 是一个很有用的指标，你可以写一个程序，不断产生 0-100 之间的随机数 $x$，然后把 $x$ 对应的 $\log(\frac{x}{100-x})$ 用柱状图画出来，你会发现它符合正态分布：

![image-20191128200536907](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20191128200536907.png?raw=true)

**在实际应用中，我们可以把上面的 $x$ 替换为某个网站的点击，或购买等指标，根据历史数据算出对应的 log(odds) 分布，再找一堆相关的特征来拟合这个分布，这就是我们所说的 CTR（Click Through Ratio）或 CVR（Conversion Rate） 模型**，后续来了一个用户，我们把他相关的特征带入到模型中，算出相应的 log(odds)，就是这个用户会点击或购买某个商品的几率。

有人说，这和逻辑回归有什么关系？实际上，log(odds) 还有一种计算方法：
$$
\log(odds) = \log(\frac{p}{1-p})
$$
其实也很容易理解，依然是上面的例子，中国队胜利的概率为 p=0.1，中国队胜利的 log(odds) 为
$$
\begin{aligned}
\log(odds) &= \log(\frac{1}{99}) \\&= \log(\frac{\frac{1}{100}}{\frac{99}{100}}) \\&= \log(\frac{\frac{1}{100}}{1-\frac{1}{100}}) \\&=\log(\frac{p}{1-p})
\end{aligned}
$$
我们把等式两边同时求一个 $e$ 的次方，算出 p 值，即
$$
\begin{aligned}
p &= \frac{e^{\log(odds)}}{1+e^{\log(odds)}} \\&= \frac{1}{1+e^{-\log(odds)}}
\end{aligned}
$$


这就是我们所熟知的逻辑回归，等式右边的表达式通常被称为 sigmoid 函数，而 log(odds) 又被称为 logit 函数，它们之间的转换关系如下图所示，其中 x 轴可看成特征向量。

![lr_logodds](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/lr_logodds.jpg?raw=true)

从图中可以看出，如果把逻辑回归转化为 log(odds)，有两点明显的变化：

1. log(odds) 是一条直线
2. log(odds) 可以将逻辑回归的值域从 (0, 1) 拓宽到 (-∞, +∞)

突然有点像线性回归了，但和线性回归的差异是，逻辑回归的样本只有 0 和 1 两个值，转换为 log(odds) 正好是 -∞ 和 +∞，这样你使用 MSE 来拟合时，得到的 Loss 永远都是个无穷大，所以用线性回归的方法来拟合逻辑回归是不可行的。在逻辑回归中，我们使用 Maximu Likelihood 来作为模型的 Loss。

## 最大释然估计（Maximum Likelihood）

Maximum Likelihood 也是很直观的一个概念，即我现在有一堆正样本和负样本，我用一条怎样的逻辑回归曲线去拟合这些样本，能使它们所得到概率的乘积最大。

举个例子，假设下图左边是一个关于体重和肥胖的实验数据，其中绿色点标记的是正常，而红色点为肥胖，现在要使用逻辑回归对这些样本建模，假设最佳模型如下面右图所示：

![lr_fit](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/lr_fit.jpg?raw=true)

通过该模型的计算，绿色样本对应的肥胖的概率由左至右分别为 0.01、0.02、0.03 和 0.9，绿色是正常样本，需要计算他们不是肥胖的概率，所以要用 1 减去这些值，得： 0.99、0.98、0.97 和 0.1；同理，再分别计算红色样本是肥胖的概率为 0.1、0.97、0.98 和 0.99，因为该曲线已经是最优的了，所以这 8 个点所对应的概率的乘积——0.0089，即是所有可能的模型中，能得到的最大值。可见，Maximum Likelihood 真的就只是其字面意思了。

线性回归中，我们使用 MSE 来衡量线性模型的好坏，MSE 越小，说明拟合得越好；而在逻辑回归中，使用的正是 Maximum Likelihood，该指标越大，模型越好。

对于样本 $x_i$，当它为正样本时，对应的概率为 $p(x_i)$，而当它为负样本时，对应的概率为 $1-p(x_i)$，为方便计算，我们需要用一个式子来表示这两种情况：
$$
p_i = y_i\cdot p(x_i) + (1-y_i)\cdot (1-p(x_i))
$$
这里 y 表示样本的取值，因为 y 只有两种取值，0 和 1，当 y 为正样本 1 时，带入上式得 $p_i=p(x_i)$，而当 y 为负样本 0 时，带入上式得 $p_i=1-p(x_i)$，于是每个样本的概率的表现形式得到了统一，这样总的概率就很好表示了：
$$
\begin{aligned}
{\arg\max} L(\theta) &= \prod_{i=1}^{n}p_i \\
&= \prod_{i=1}^{n}[y_i\cdot p(x_i) + (1-y_i)\cdot (1-p(x_i))]
\end{aligned}
$$
上式中，n 表示有 n 条样本，下标 i 表示第 i 条样本，x 为特征向量，y 为观察到的目标值，$\theta$ 为特征向量的权重，也是模型的参数，L 即为所有样本的 Likelihood，也是逻辑回归中的 Loss 函数，我们的目标是调整 $\theta$， 以使 L 最大。

通常我们会把连乘通过 log 转换为求和，并取负号，把求最大值转换为求最小值，如下：
$$
\begin{aligned}
\arg\min (-\log(L(\theta))) &= -\sum_{i=1}^{n}\log(p_i) \\
&= -\sum_{i=1}^{n}[y_i\cdot \log(p(x_i)) + (1-y_i)\cdot \log((1-p(x_i)))]
\end{aligned}
$$
接下来就是对 Loss 求梯度了，然后根据梯度来修改参数，并不断迭代收敛的过程，为了减少大家阅读时的不适感，这里就不继续推导了， 不过没有推导过的同学，还是建议自己在草稿上演算一下，可加深自己的理解。

## 逻辑回归和贝叶斯分类

贝叶斯分类的核心依然来自于经典的贝叶斯公式：
$$
p(c|x) = \frac{p(x|c)p(c)}{p(x|c)p(c)+p(x|\bar{c})p(\bar{c})}
$$
**在分类问题中，我们要求的实际上是当样本 x 出现时，它属于分类 c 的概率**，即上式的 p(c|x)。等式右边的 $\bar{c}$ 表示为 c 之外的其他分类，p(c) 和 $p(\bar{c})$ 可以理解为先验概率，一般情况下你可以把它们设置为均等的，如我们可以把二分类的先验概率都设为 0.5。

接着，p(x|c) 可表示为在 c 分类中观察到 x 样本出现的概率，同理，$p(x|\bar{c})$ 则为在 $\bar{c}$ 分类中观察到 x 样本的概率。于是，p(c|x) 就是一个后验概率。

理解了贝叶斯分类后，我们把等式右边的分子分母同时除以 $p(x|c)p(c)$，如下：
$$
p(c|x) = \frac{1}{1+\frac{p(x|\bar{c})p(\bar{c})}{p(x|c)p(c)}}
$$
到此，这个式子是不是像极了 sigmoid 函数，我们设：
$$
e^{-z} = \frac{p(x|\bar{c})p(\bar{c})}{p(x|c)p(c)}
$$
再设先验概率相等，同时在等式两边取 log，便得到：
$$
-z = \log(\frac{p(x|\bar{c})}{p(x|c)})
$$
将负号移到右边:
$$
z=\log(\frac{p(x|c)}{p(x|\bar{c})}) = \log(odds)
$$
最后将 z 带回原式：
$$
p(c|x) = \frac{1}{1+e^{-\log(odds)}}
$$
结论是，逻辑回归实际上就是贝叶斯分类，它们都是一个后验概率模型。

## 总结

本文我们主要通过 log(odds) 和贝叶斯分类这两个概念来学习了逻辑回归算法的原理，且了解了逻辑回归是采用 Maximum Likelihood 来作为其损失函数的，希望你和我一样，通过本文能够对逻辑回归有更深刻的理解。



参考：

* [Logistic Regression, Clearly Explained](https://www.youtube.com/playlist?list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe)
* [Classification](https://www.youtube.com/watch?v=fZAZUYEeIMg&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=10&t=0s)
* [Logistic Regression](https://www.youtube.com/watch?v=hSXFuypLukA&list=PLJV_el3uVTsPy9oCRY30oBPNLCo89yu49&index=11&t=0s)

