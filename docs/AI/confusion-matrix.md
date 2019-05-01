# 二分类评估，从混淆矩阵说起

在《理解交叉验证》一文中，我们谈到了使用 AUC 来对比不同模型的好坏，那么 AUC 是什么？它是如何衡量一个模型的好坏的呢？除了 AUC 以外，还有什么其他的手段呢？本文我们就来探讨下这几个问题。

## 混淆矩阵

要了解 AUC，我们需要从另外一个概念——混淆矩阵（Confusion Matrix）说起，混淆矩阵是一个 2 维方阵，它主要用于评估二分类问题（例如：预测患或未患心脏病、股票涨或跌这种只有两类情况的问题）的好坏，你可能会问多分类问题怎么办？实际上，多分类问题依然可以转换为二分类问题进行处理。下图展示了一张用于评判是否患有心脏病的混淆矩阵：

![cm1](/Users/fengyajie/Downloads/cm1.png)

纵向看混淆矩阵，它体现了真实情况下，患病和不患病的人数，上图中，真实患心脏病的人数为 True Positive + False Negative，未患心脏病的人数为 False Positive + True Negative；类似的，横向看混淆矩阵，它体现了模型预测出来患心脏病的人数为 True Positive + False Positive，而预测未患心脏病的人数为 False Negative + True Negative。

两个方向一起看，预测患病且实际也患病，我们称它为真阳性 (True Positive)，预测未患病且实际也未患病，被称为真阴性 (True Negative)，这两块是模型预测正确的部分；模型预测错误也分两种情况，假阳性 (False Positive) 表示预测患病，但实际并未患病，假阴性 (False Negative) 表示预测未患病，但实际却患了病的情况。概念有点多，但并不难记，可以看到，这些名词都是围绕着预测来命名的——预测患病时被称为「XX Positive」，预测未患病时被称为 「XX Negative」。

上图中，模型预测正确的部分用绿色填充，它又被称为准确率 (Accuracy)：
$$
Accuracy = \frac {TP + TN} {TP + TN + FP + FN}
$$
单用准确率，并不能足以评估模型的好坏，例如下面这种情况，虽然准确率可以达到 80%，但在实际患病的人群中，该模型的预测成功率只有 50%，很明显此模型不是一个好模型。

|            | 患心脏病 | 未患心脏病 |
| ---------- | -------- | ---------- |
| 患心脏病   | 10       | 10         |
| 未患心脏病 | 10       | 70         |

## Sensitivity 和 Specificity

所以，我们需要引入更多的衡量指标，Sensitivity (或 Recall) 表示实际患者中，预测成功的概率，而 Sensitivity 这个词也有"过敏"的意思，和患病对应，这样关联起来比较好记：
$$
Sensitivity = \frac{TP}{TP+FN}
$$
既然有衡量患病（正样例）的指标，那肯定也有衡量未患病（负样例）的指标，Specificity 就是用来表示实际未患病的人群中，预测成功的概率，即
$$
Specificity = \frac{TN}{TN+FP}
$$
Specificity 这个词有"免疫"的意思，能和未患病相关联，所以也比较好记。

这两个指标的出现，能更好的帮你比较模型间的差异，并在其中做出取舍。例如当两个模型的 Accuracy 相近时，如果你更看重于预测患病的效果，你应该选 Sensitivity 值较高的；相反，如果你更看重于预测未患病的效果，你更应该选择 Specificity 较高的。

## ROC 曲线、AUC 和 F1 Score

更进一步，我们还可以通过将这些指标图形化，以获得更直观的评估结果。ROC (receiver operating characteristic) 曲线是其中常用的一种。

我们知道，分类模型（例如"逻辑回归”）的结果为一个大于 0 且小于 1 的概率，此时我们还需要一个阈值，才能界定是否患病，假设阈值为 0.5，则可将预测结果大于 0.5 时判定为患病，否则判定为未患病。

而阈值可以取 0 到 1 之间的任意一个值，对每一个阈值，我们都可以画出一个混淆矩阵，同时还可以求出一对 Sensitivity 和 Specificity，于是可以得到一个以 1-Specificity 为横坐标，Sensitivity 为纵坐标的点，把所有这些阈值产出的点连起来，就是 ROC 曲线。

下面我们来看一个具体的例子，假设我们对老鼠做研究，希望通过老鼠的体重来预测其患心脏病的概率，我们采用逻辑回归算法来建模，下图是我们的预测结果，其中红色点代表健康老鼠，蓝色点代表患病老鼠，假设阈值设为 0.5，从下图可以得知，高于 0.5 的 5 只老鼠被预测为患病，而其他 5 只老鼠被预测未健康，预测成功率为 80%：

![image-20190430205818287](/Users/fengyajie/Library/Application Support/typora-user-images/image-20190430205818287.png)

下面我们通过以上数据，来画一条 ROC 曲线。首先取阈值为 1，此时所有的老鼠都被预测为未患病，根据样本真实患病情况，我们可以得到如下混淆矩阵

![Untitled Diagram-Page-2 (1)](/Users/fengyajie/Downloads/Untitled Diagram-Page-2 (1).png)

根据上述混淆矩阵，我们可以算出一组 Sensitivity 和 Specificity 的值。接着我们不断调整阈值，每变换一个阈值，就有一个混淆矩阵与之对应，因为这里我们样本点较少，所以让阈值根据样本点来采样即可，阈值采样情况如下：

![image-20190430224415492](/Users/fengyajie/Library/Application Support/typora-user-images/image-20190430224415492.png)

我们把这些阈值对应的所有混淆矩阵都列出来：

![Untitled Diagram-Page-3](/Users/fengyajie/Downloads/Untitled Diagram-Page-3.png)

最终，我们可以画一个 Sensitivity 和 1-Specificity  的表格

| Threshold | Sensitivity | 1- Specificity |
| --------- | ----------- | -------------- |
| 1         | 0           | 0              |
| 0.99      | 0.2         | 0              |
| 0.97      | 0.4         | 0              |
| 0.94      | 0.4         | 0.2            |
| 0.90      | 0.6         | 0.2            |
| 0.71      | 0.8         | 0.2            |
| 0.09      | 0.8         | 0.4            |
| 0.043     | 1.0         | 0.4            |
| 0.0061    | 1.0         | 0.6            |
| 0.0003    | 1.0         | 0.8            |
| 0         | 1.0         | 1.0            |

根据该表格，以 1-Specificity 为横轴，Sensitivity 为纵轴作图，通常，在画 ROC 曲线时，我们把 1-Specificity 对应的坐标轴记为 FPR (False Positive Rate)，把 Sensitivity 对应的坐标轴即为  TPR (True Positive Rate)：

![image-20190430234946551](/Users/fengyajie/Library/Application Support/typora-user-images/image-20190430234946551.png)

ROC 曲线有以下特点：

1. 从 0 点到 [1,1] 的对角线上的每个点，意味着在患者中，预测患病成功的概率，与未患病者中，预测未患病失败的概率相等，所以我们需要尽可能的使模型的 ROC 曲线沿左上角方向远离该对角线
2. ROC 曲线还可以帮助我们选择合适的域值，即 TPR 相同的情况下，ROC 上的点越靠左，效果越好，因为越靠左，意味着 FPR 越小。

根据 ROC 曲线的第 1 个特点，曲线越靠近左上角，模型的效果越好，意味着曲线下方的面积越大，模型的效果越好，于是我们把 ROC 曲线下方的面积称为 AUC (Area Under Curve)，有了这个概念后，通常情况下，我们只用一个数值就可以衡量模型的好坏了，刚才我们"训练"的模型的 AUC 如下：

![image-20190501230646710](/Users/fengyajie/Library/Application Support/typora-user-images/image-20190501230646710.png)

既然是通常情况下都使用 AUC，那肯定就有例外，当患病率 (或正样本占比)  非常小时，Ture Negative 就会非常大，这个值就会使影响 FPR，使 FPR 偏小，此时，我们通常将 FPR 用另一个指标代替：Precision
$$
Precision = \frac{TP}{TP+FP}
$$
Precision 的含义是预测为患病的样本中，正确的比例；这样，将 Precision 和 Sensitivity 结合起来，会让我们更专注于患病 (正样本) 的预测效果，和 AUC 一样，在机器学习中，我们有另一个效果指标：**F1 Score**
$$
F1~Score = 2\times \frac{Precision\times Recall}{Precision + Recall}
$$
上面的公式中，Recall 等价于 Sensitivity，和 AUC 一样，两个模型中，我们可以说，F1 Score 越大，预测效果越好，而且 F1 Score 能更好的衡量正样本的预测效果。

## 总结

本文



