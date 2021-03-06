# 线性变换及其与矩阵的关系

![](https://github.com/jieniu/articles/blob/master/.vuepress/public/image-20190105220419469.png?raw=true)

> Unfortunately, no one can be told what the Matrix is. You have to see it for yourself.
>
>  -- Morpheus

### 线性变换 Linear transformation

通常 **变换（transformation）** 相当于函数（function）— $f(x)$ ，给它一定的输入，它会产生相应的输出。在线性代数的场景中，变换（transformation）可以想象为输入某个向量，然后输出另一个向量的过程。

如果是这样，为什么使用变换（transformation）这个词，而不直接使用函数（function）呢？因为变换有移动的含义在里面，而更好的理解输入向量到输出向量的过程的方式是**移动向量**。

如果一个变换（transformation）接收一个输入向量，并输出一个新的向量，我们可以想象它是从输入的向量 （vector）**移动**到了输出的向量（vector）。然后我们把这种变换当做一个整体来理解，想象整个平面内任何向量（vectors）都随着这个变换（transformation）发生了各自的**移动**，等同于平面内所有的点随着该变换（transformation）移动到了另一个点。

而线性代数中的**线性变换（Linear transformation**）是一种更易理解的、特殊的变换，它具备两个的条件:

1. 向量在变换后仍然是直线，不会被扭曲；
2. 原点不会发生移动。

把一个平面想象为彼此间均匀且平行的网格，**线性变换会让网格中的线条依然保持平行且均匀。**例如下图是细实线组成的空间变换到粗实线组成的空间后的样子：

![](https://github.com/jieniu/articles/blob/master/.vuepress/public/image-20190105223100319.png?raw=true)

理解了线性变换后，我们如何用数学的方式来表示它呢？这样我们就可以把这个“公式”制作成计算机程序，然后输入一个向量的坐标，它就会给我们返回变换后的向量的坐标。

实际上你只需要记录两个基本向量变换后的向量即可，也就是 $\hat{i}$ 和 $\hat{j}$ 变换后的向量 $\hat{i}_{transformed}$ 和 $\hat{j}_{transformed}$，因为所有向量都可以由基向量通过乘法和加法表示而来，所以任何向量变换后的结果也可以由变换后的基本向量 $\hat{i}_{transformed}$ 和 $\hat{j}_{transformed}$ 计算得出，这归因于刚才说的线性变换所具备的两个重要的条件，正是因为这两个条件，其他向量和基向量间的比例才能在变换后依然得以保持，即只要是线性变换，在新的空间中， $\hat{i}_{transformed}$ 和 $\hat{j}_{transformed}$ 依然是 1 个单位长度（相对来说）的基向量。

举个例子，例如向量 $\left[\begin{matrix}-1\\2\end{matrix}\right]$ 在变换前为 $-1\hat{i} + 2\hat{j}$，由于线性变换的**平行**和**均匀**的特性，在 $\hat{i}$ 和 $\hat{j}$ 变换后，新向量的计算方式为：

$$
\vec v_{transformed } = -1 \hat{i}_{transformed} + 2\hat{j}_{transformed}
$$

可以看到，虽然进行了线性变换，但变换前后，相同向量的线性组合并没有发生变化。所以，**只要我们知道了 $\hat{i}$ 和 $\hat{j}$ 在变换后的位置，我们就可以推断其他的向量的变换情况**，而不需要专门的观察所有其他向量的变换情况。具体一点，假设有这样的变换，$\hat{i}$ 变换到 $\left[\begin{matrix}-1\\2\end{matrix}\right]$，而 $\hat{j}$ 变换到 $\left[\begin{matrix}3\\0\end{matrix}\right]$，对于任意向量 $\left[\begin{matrix}x\\y\end{matrix}\right]$ 而言，在变换后它将落在$\left[\begin{matrix}1x+3y\\-2x + 0y\end{matrix}\right]$，如下：  

$$
\hat{i} \to \left[\begin{matrix}1\\-2\end{matrix}\right] \qquad \hat{j}\to \left[\begin{matrix}3\\0\end{matrix}\right]
$$

$$
\left[\begin{matrix}x\\y\end{matrix}\right] \to x\left[\begin{matrix}1\\-2\end{matrix}\right] + y\left[\begin{matrix}3\\0\end{matrix}\right] = \left[\begin{matrix}1x+3y\\-2x + 0y\end{matrix}\right]
$$

结论是，在二维空间中，线性变换仅需要用 4 个数字来表示，即 $\hat{i}_{transformed}$ 对应的两个坐标和 $\hat{j}_{transformed}$ 对应的两个坐标 ，一般我们把它们放到一个2乘2的“矩阵”中，即

$$
\left[\begin{matrix}a\quad b\\c\quad d\end{matrix}\right]
$$

左边的 $\left[\begin{matrix}a\\c\end{matrix}\right]$ 是 $\hat{i}$ 变换后的向量，而右边的 $\left[\begin{matrix}b\\d\end{matrix}\right]$ 是 $\hat{j}$ 变换后的向量，**这就是矩阵真正的来历**——它只是用来表示线性变换的方式而已。而对于原向量空间中的向量 $\left[\begin{matrix}x\\y\end{matrix}\right]$ ，根据线性组合，我们便知道其变换后的向量为

$$
x\left[\begin{matrix}a\\c\end{matrix}\right] + y\left[\begin{matrix}b\\d\end{matrix}\right] = \left[\begin{matrix}ax + by\\cx + dy\end{matrix}\right]
$$

同样为了方便我们记录，我们通常把上面的式子定义为：

$$
\left[\begin{matrix}a\quad b\\c \quad d\end{matrix}\right]\left[\begin{matrix}x\\y\end{matrix}\right]
$$

即**把矩阵放在原向量的左边，就像这个向量的函数一样**，把式子写完整，如下：

$$
\left[\begin{matrix}a\quad b\\c \quad d\end{matrix}\right]\left[\begin{matrix}x\\y\end{matrix}\right]=
x\left[\begin{matrix}a\\c\end{matrix}\right] + y\left[\begin{matrix}b\\d\end{matrix}\right] = \left[\begin{matrix}ax + by\\cx + dy\end{matrix}\right]
$$

看到上面的式子，会不会感觉很熟悉，这就是我们在教科书中学到的矩阵向量的乘法，现在你知道这个计算背后的意义了吧：它只是用来计算空间变换给指定向量带来的变化的工具而已。而本文的重点是：**一旦今后你看到了矩阵，你便可以将其解释为空间的一种特定的转换，理解了这一点，线性代数的一切都好理解了。**



参考：

* [Linear transformations and matrices ](https://www.youtube.com/watch?v=kYB8IZa5AuE&index=3&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)

### 几种线性变换

- Linearly dependent columns：$\hat{i}$ 和 $\hat{j}$ 在变换后形成了一条直线，意味着线性变换将 2D 空间压缩到一条直线上，例如 
  $$
  \left[\begin{matrix}2 \quad -2\\1 \quad -1\end{matrix}\right]
  $$

- $90^{\circ}$ 逆时针旋转
  $$
  \left[\begin{matrix}0 \quad -1\\ 1 \qquad  0\end{matrix}\right]
  $$

- Shear 变换：水平方向保持不变，垂直方向向右旋转 $45^{\circ}$
  $$
  \left[\begin{matrix}1 \quad 1\\0 \quad 1\end{matrix}\right]
  $$


