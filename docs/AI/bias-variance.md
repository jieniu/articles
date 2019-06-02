# 理解 Bias 和 Variance

在用机器学习算法对数据进行拟合的过程中，往往一开始得不到满意的结果，例如 AUC 值不够高，此时我们就需要对模型进行调优，那么调优的方向是什么？有没有调优方法论可遵循？答案当然是有的，bias 和 variance 这两个指标就能起到指导调优的作用。

## Bias

我们先来看一个例子，假设实验室收集了老鼠的体重和大小的数据，我们可以建立一个模型，通过输入老鼠的大小来预测老鼠的体重，部分数据散点图如下。在训练之前，我们还是将数据拆分为两部分，红色的点为训练集，绿色的点表示测试集：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190507232247920.png?raw=true)

接着我们用两个模型来拟合训练数据，第一个模型采用线性算法，如下：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190507233719525.png?raw=true)

可以看到，线性模型并不能很好的描绘真实数据，我们一般使用 MSE (Mean Squared Error) 来量化这种拟合能力，即预测值和实际值之间的差值的平方的均值。

接下来我们训练第二个较复杂的模型，该模型的曲线如下：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190507232909193.png?raw=true)

第二个模型完美的贴合了训练数据，即用这个模型来预测训练数据，获得的预测值与实际值相等。

至此，我们再给出 bias 的定义就不难理解了：

> Bias 指标衡量了在训练阶段，机器学习算法和真实数据之间的差异。

从上面的例子可以看出，模型二的 bias 远远低于模型一的 bias。

## Variance

训练完模型后，我们还需要使用测试集对模型进行评估，下图是模型一的评估结果，我们用蓝色虚线来表示测试结果中，预测值和实际情况的差异（也可以使用 MSE 来衡量）：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190507233913941.png?raw=true)

同样，模型二的评估结果如下：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190507234501259.png?raw=true)

和 Bias 相反的是，模型一的预测效果却远远好于模型二的，这说明模型二的预测能力并不稳定，同样我们也是试着给 Variance 下个定义：

> Variance 表示在不同测试集间，预测效果间的偏差程度，偏差程度越大，variance 越大，反之越小。

显然模型二的 variance 较大；而对于不同测试集，模型一预测的准确性非常接近，我们可以说模型一的 variance 较小。

## Bias & Variance

下图摘自 [Scott Fortmann-Roe's 的博客](http://scott.fortmann-roe.com/docs/BiasVariance.html)，它能够很好的描绘我们在机器学习中的调优方向，其中左上角是最理想的模型，它是终极目标，如果实在做不到，你应该朝着左下角的 High Bias + Low Variance 努力。

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190601170853678.png?raw=true)

上图中，右上角的情况又被称为**过拟合 (Overfit)**，它表示你的模型在训练时的表现非常好，但在测试过程中的表现又非常差，上文的模型二就是典型的过拟合情况。有过拟合肯定就有**欠拟合 (Underfit)**，它表示你的模型无法很好的刻画样本数据。同时，过拟合一般对应你使用了一个较复杂的模型，而欠拟合一般和简单模型相对应。很多时候，我们说模型调优，实际上指的是：

>  **在简单模型和复杂模型间寻求平衡**。

如何做到呢？这里有一些经验方法：

* 如何处理 variance 较大的问题
  1. 减少特征数量
  2. 使用更简单的模型
  3. 增大你的训练数据集
  4. 使用正则化
  5. 加入随机因子，例如采用 bagging 和 boosting 方法
* 如何处理 bias 较大的问题
  1. 增加特征数量
  2. 使用更复杂的模型
  3. 去掉正则化



参考：

[Machine Learning Fundamentals: Bias and Variance](https://www.youtube.com/watch?v=EuBBz3bI-aA&list=PLblh5JKOoLUICTaGLRoHQDuF_7q2GfuJF&index=6&t=0s)
