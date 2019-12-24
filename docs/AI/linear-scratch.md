## 使用 TensorFlow2.0 实现线性回归

本文是笔者学习 TensorFlow2.0（下文都写作 TF2.0） 的一篇笔记，使用的教材是《动手深度学习》（TF2.0版）。

之所以可以使用 TensorFlow 来实现线性回归，是因为我们可以把线性回归看成是只有一层、一个神经元的全连接网络：

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/linear1.png?raw=true)

上面这个图就是线性回归 $y = w_1x_1 + w_2x_2 + b$ 的神经网络的表示。

## 实现线性回归

要实现线性回归，我们需要

1. 定义线性回归模型
2. 定义 Loss 函数
3. 定义迭代优化算法

这些也是机器学习理论中的要点，顺带着，我们借本文来回顾一下。

### 定义线性回归模型

要实现一个算法，我们首先需要用矢量表达式来表示它，即：使用向量、矩阵来描述一个模型。这样做的好处是：矢量批量计算要比循环一条条的计算每个样本来得快得多，线性回归的矢量表达式为：
$$
\hat{y} = Xw + b 
$$
其中，$X$ 是一个 $n\times d$ 维的矩阵，$n$ 表示 n 条样本，$d$ 表示特征的维数；$w$ 是模型的参数，它是一个 $d\times 1$ 维的向量；$b$ 是偏差值，它是一个标量；$\hat{y}$ 是 n 条样本的预测值，它也是 $n \times 1$ 的向量。

该模型用 TF2.0 实现如下：

```python
import tensorflow as tf
import numpy as np
import random

def linear_reg(X, w, b):
  return tf.matmul(X, w) + b
```

### 定义 Loss 函数

一般的，回归模型的 Loss 函数为 MSE（Mean Squared Error）：
$$
Loss = \frac{1}{2n}(y-\hat{y})^2
$$
上式中，$y$ 是样本的观测值（Observed Value），$y$ 和 $\hat{y}$ 都是 $n \times 1$  的向量，n 表示对 n 个样本的 Loss 求平均，避免样本数量给 Loss 带来的影响。因为 Loss 是一个标量，所以上式还需要调整如下：
$$
Loss = \frac{1}{2n}(y-\hat{y})^{\top}(y-\hat{y})
$$
Loss 用 TF2.0 实现如下：

```python
def squared_loss(y, y_hat, n):
  y_observed = tf.reshape(y, y_hat.shape)
  return tf.matmul(tf.transpose(y_observed - y_hat), 
                   y_observed - y_hat) / 2 / n
```

### 定义迭代优化算法

深度学习大多采用小批量随机梯度下降优化算法（minibatch Stochastic Gradient Descent）来迭代模型的参数，该算法能节省内存空间，增加模型的迭代次数和加快模型的收敛速度。

SGD 算法每次会随机的从样本中选取一部分数据，例如 100 条，然后计算这 100 条数据的梯度，并根据梯度来更新当前的参数，所以这里包含 3 个步骤：

1. 随机选择样本，每次选 n 条
2. 计算这 n 条样本的 Loss，并计算梯度，使用梯度更新参数
3. 循环 1 和 2

先来看下随机选择样本的代码

```python
def data_iter(features, labels, mini_batch):
  '''
  数据迭代函数
  你只需要使用下面代码便可以随机的、小批量的抽取样本，使用方法：
  mini_batch = 100
  for X, y in data_iter(features, labels, mini_batch):
  # do gradient descent
  '''
  features = np.array(features)
  labels = np.array(labels)
  indeces = list(range(len(features)))
  random.shuffle(indeces)
  for i in range(0, len(indeces), mini_batch):
    j = np.array(indeces[i:min(i+mini_batch, len(features))])
    yield features[j], labels[j]
```

接着，我们再来看下更新模型参数的代码：

```python
def sgd(params, lr):
  '''
  计算梯度，并更新模型参数
  params:
  - params: 模型参数
  - lr: 学习率 learning rate
  '''
  for param in params:
    param.assign_sub(lr * t.gradient(l, param))
```

以上，关键代码就写完了，下面我们把它们们串起来：

```python
# 产生模拟数据
# 1000 条样本，2 维特征
num_samples = 1000
num_dim = 2
# 真实的 weight, bias
w_real = [2, -3.4]
b_real = 4.2
# 产生特征，符合正态分布，标准差为 1
features = tf.random.normal((num_samples, num_dim), stddev=1)
labels = features[:,0]*w_real[0] + features[:,1]*w_real[1] + b_real 
# 给 labels 加上噪声数据
labels += tf.random.normal(labels.shape, stddev=0.01)
# 学习率，迭代次数
lr = 0.03
num_epochs = 3
# 初始化模型参数
w = tf.Variable(tf.random.normal([num_dim, 1], stddev=0.01))
b = tf.Variable(tf.zeros(1,))
mini_batch = 10

for i in range(num_epochs):
    for X, y in data_iter(features, labels, mini_batch):
    		# 自动求梯度
        with tf.GradientTape(persistent=True) as t:
            t.watch([w, b])
            l = squared_loss(y, linear_reg(X, w, b), mini_batch)
        sgd([w, b], lr)
    # 计算本次迭代的总误差
		train_loss = squared_loss(labels, linear_reg(features, w, b), len(features))
    print('epoch %d, loss %f' % (i + 1, tf.reduce_mean(train_loss)))
```

## 简单实现

上述代码是根据线性回归的原理一步步的实现的，步骤十分清晰，但比较繁琐，实际上，TF 提供了丰富的算法库供你调用，大大的提升了你的工作效率。下面我们就用 TF 库中提供的方法来替换上述代码。

我们先用 keras 来定义一个只有 1 层的全连接网络结构：

```python
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init

model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))
```

接下来设置 Loss 函数为 MSE：

```python
from tensorflow import losses

loss = losses.MeanSquaredError()
```

设置优化策略为 SGD：

```python
from tensorflow.keras import optimizers

trainer = optimizers.SGD(learning_rate=0.03)
```

小批量随机获取数据集的代码如下：

```python
from tensorflow import data as tfdata

batch_size = 10
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(len(features)).batch(batch_size)
```

可见，构建一个模型就是设置一些配置项，把上面代码合起来，如下：

```python
from tensorflow import data as tfdata
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow import initializers as init
from tensorflow import losses
from tensorflow.keras import optimizers

# 设置网络结构：1 层全连接，初始化模型参数
model = keras.Sequential()
model.add(layers.Dense(1, kernel_initializer=init.RandomNormal(stddev=0.01)))
# loss 函数：MSE
loss = losses.MeanSquaredError()
# 优化策略：随机梯度下降
trainer = optimizers.SGD(learning_rate=0.03)
# 设置数据集，和小批量的样本数
batch_size = 10
dataset = tfdata.Dataset.from_tensor_slices((features, labels))
dataset = dataset.shuffle(len(features)).batch(batch_size)

num_epochs = 3
for epoch in range(1, num_epochs+1):
    # 取小批量进行计算
    for (batch, (X, y)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # 计算 loss
            l = loss(model(X, training=True), y)
        # 计算梯度并更新参数
        grads = tape.gradient(l, model.trainable_variables)
        trainer.apply_gradients(zip(grads, model.trainable_variables))
    
    # 本次迭代后的总 loss
    l = loss(model(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.numpy().mean()))
# 输出模型参数
print(model.get_weights())
```

上面代码直接拷贝到 Jupyter Notebook 中便可通过运行，初学的同学可以动手试试。



参考：

* [动手深度学习（TF2.0版）-线性回归从零开始实现](https://trickygo.github.io/Dive-into-DL-TensorFlow2.0/#/chapter03_DL-basics/3.1_linear-regression)
* [《动手深度学习》](https://zh.gluon.ai/)

