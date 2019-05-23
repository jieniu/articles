# 【学习笔记】Hands On Machine Learning - Chap2. End-to-End Machine Learning Project

从标题可以看出，这一章主要从大的方向，介绍机器学习的一般步骤，虽然是介绍性的知识，但不乏一些有价值的内容，以下几点是我个人的总结：

**数据预览：**

1. 预览前 5 条数据，有个直观的感受
2. 查看数据总行数，字段类型，每个字段的非空行数
3. 查看分类字段的分布情况
4. 查看数据字段的均值、方差、最小值、最大值、25/50/75分位值
5. 查看数据字段的分布（最好是图形）

**测试集创建：**

1. 数据量大的情况下可以通过随机的方式创建数据集
2. 数据量不大的情况下，需要使用分层抽样，确保样本数据和真实数据具有一样的分层分布，避免产生采样偏差

**数据分析**

* 对属性和目标字段做相关性分析
* 属性组合：在属性组合后，再做一次相关性分析，查看组合后的属性的相关性是否变强
* 对于长尾型数据，做 log 处理

**数据清洗**

* 空值填充
* 处理文字类型属性
    * label encoding 对有顺序关系的字段进行编码
    * one-hot encoding 对非顺序关系的字段进行编码

**特征工程**

* **归一化在异常值干扰方面没有标准化好**

**选择模型进行训练**

* 欠拟合的解决方案
    * 选择一个更复杂的模型
    * 增加其他更好的特征
    * 减少模型限制，例如去掉正则化
    
* 过拟合的解决方案
    * 简化模型
    * 使用正则化
    * 加大训练数据
    
* 使用交叉验证评估模型，检查模型的泛化能力
* 使用 Grid Search 方法来选择一组较好的超参组合
* 训练后，使用特征重要性分析，将无关紧要的特征去掉，之后可以再加入新特征，重新训练，直到得到满意的模型

**上线前的总结**

* 从实验中学到了什么
* 什么可行和不可行
* 本实验中有哪些假设
* 该实验有哪些限制

**上线时需要注意什么**

* 持续监控，避免因数据的持续更新，导致模型的退化
* 采样预测数据，并对其进行评估，监控模型效果
* 定期重新训练模型，如 6 个月
* 定期全量预测

以下是具体的笔记内容：

## 数据集方面

```python
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "datasets/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
    if not os.path.isdir(housing_path):
             os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
        
# 下载数据
fetch_housing_data()
```

### 数据总览

```python
import pandas as pd
def load_housing_data(housing_path=HOUSING_PATH): 
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)

# 查看前 5 条数据
housing = load_housing_data()
housing.head()
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523220534082.png?raw=true)

```python
# 查看总行数，列类型，各列的非空条数，注意到 total_bedrooms 存在空记录
housing.info()
----
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20640 entries, 0 to 20639
Data columns (total 10 columns):
longitude             20640 non-null float64
latitude              20640 non-null float64
housing_median_age    20640 non-null float64
total_rooms           20640 non-null float64
total_bedrooms        20433 non-null float64
population            20640 non-null float64
households            20640 non-null float64
median_income         20640 non-null float64
median_house_value    20640 non-null float64
ocean_proximity       20640 non-null object
dtypes: float64(9), object(1)
memory usage: 1.6+ MB
  
# 查看分类数据的分布情况
housing["ocean_proximity"].value_counts()
----
<1H OCEAN     9136
INLAND        6551
NEAR OCEAN    2658
NEAR BAY      2290
ISLAND           5
Name: ocean_proximity, dtype: int64
    
# 查看数字字段的概览
housing.describe()
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523220729885.png?raw=true)

```python
#This tells Jupyter to set up Matplotlib so it uses Jupyter’s own backend.
%matplotlib inline 
# 输出所有数字属性的分布情况
import matplotlib.pyplot as plt 
housing.hist(bins=50, figsize=(20,15))
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523220801771.png?raw=true)

### 创建测试集
测试集样本要有代表性，能反映真实情况，否则会造成采样偏差，这是很容易被忽视的部分

```python
# 数据量相对大的情况下（相对于特征数来说），随机的方法是可行的，否则会产生抽样偏差
# 分层抽样，样本各类别的比例要符合真实情况，例如实际男女比例为6:4，那么样本中男:女就应该为6:4
# 要保证测试集符合真实情况，假设收入是预测房价的重要特征，你就需要确保测试集具有和真实情况同样的收入分布
# 构造收入分类属性
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

# sklearn 的分层抽样方法
from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
housing["income_cat"].value_counts() / len(housing)
----
3.0    0.350581
2.0    0.318847
4.0    0.176308
5.0    0.114438
1.0    0.039826
Name: income_cat, dtype: float64
    
# remove income_cat attr
for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)
```

## 深入探索和将数据可视化

```python
# 复制数据，将其可视化
housing = strat_train_set.copy()
# 以地理位置画散点图
housing.plot(kind="scatter", x="longitude", y="latitude")
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523221326162.png?raw=true)

```python
# 设置透明度，可显示数据稠密地区
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523221350684.png?raw=true)

```python
# 添加人口和房价信息
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
         s=housing["population"]/100, label="population",
         c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
     )
plt.legend()
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523221419588.png?raw=true)

### 相关性分析
计算属性间的相关性 - standard correlation coefficient (Pearson's r)，属性间是否相关可参考以下图示

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/scc.png?raw=true)

**注意：最底下的图形为非线性关系**

```python
corr_matrix = housing.corr()
# 查看属性和房价的相关性
corr_matrix["median_house_value"].sort_values(ascending=False)
----
median_house_value    1.000000
median_income         0.687160
total_rooms           0.135097
housing_median_age    0.114110
households            0.064506
total_bedrooms        0.047689
population           -0.026920
longitude            -0.047432
latitude             -0.142724
Name: median_house_value, dtype: float64
```

```python
# 使用 pandas' scatter_matrix 函数来查看两两属性的相关性
from pandas.tools.plotting import scatter_matrix
attributes = ["median_house_value", "median_income", "total_rooms",
                  "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```

![](https://github.com/jieniu/articles/blob/master/docs/.vuepress/public/image-20190523221906362.png?raw=true)

### 属性组合

* 对于长尾属性，可以对其进行 log 处理

```python
# 添加组合属性
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
# 继续查看相关性，可以看到 bedrooms_per_room 比 total_bedrooms 和 total_rooms 相关性都高
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
----
median_house_value          1.000000
median_income               0.687160
rooms_per_household         0.146285
total_rooms                 0.135097
housing_median_age          0.114110
households                  0.064506
total_bedrooms              0.047689
population_per_household   -0.021985
population                 -0.026920
longitude                  -0.047432
latitude                   -0.142724
bedrooms_per_room          -0.259984
Name: median_house_value, dtype: float64
```

## 机器学习准备

```python
# 将 label 分离，drop 操作的是复制的数据，不会影响原数据
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()
```

### 数据清洗

1. 空值填充

```python
# 方法1：删除含空值的记录
#housing.dropna(subset=["total_bedrooms"])
# 方法2：删除所有属性 
#housing.drop("total_bedrooms", axis=1)
# 方法3：用（0、中值、平均值等）填充空值
median = housing["total_bedrooms"].median()
housing["total_bedrooms"].fillna(median)

# 你也可以使用 Imputer
from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # imputer只能用于数字字段上
imputer.fit(housing_num)

imputer.statistics_
----
array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
        408.    ,    3.5409])

# 将 imputer 应用到数据中
X = imputer.transform(housing_num)
# 将输出结果转换为 Pandas Dataframe 格式
housing_tr = pd.DataFrame(X, columns=housing_num.columns)
```



### Scikit-Learn 的设计原则

* **一致性**：所有对象保持一致且简单的接口
  - Estimators: 可以基于数据训练参数的对象称为 estimator, 例如：imputer 是一个 estimator，训练使用 `fit()` 函数完成。超参数：除数据源和 label 外的其他参数
  - Transformers: 能作用于数据上且对数据做出改变的 estimators 被称为 transformers，API 为 `transform()`；`fit_transform()` 等价于 `fit()` 和 `transform()`
  - Predictors: 可以对数据进行预测的 estimators 被称为 predictors，例如 LinearRegression 模型，predictors 提供一个 `predict()` 接口，同时还提供 `score()` 接口，用来衡量预测的质量
* **可检查**：所有 estimators 的超参数都可通过公有成员访问，例如 `imputer.stategy`，训练参数也可以通过带下划线的公有成员访问，例如 `imputer.statistics_`
* **类型友好**：数据集由 NumPy 数组或 SciPy 稀疏矩阵表示
* **可组合**：很容易构建 Pipeline estimator
* **良好的默认行为**：对大多数参数来说，都提供一个合理的默认值，这样很容易创建一个基准版本

### 处理文字及分类属性

```python
# labelEncoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_encoded = encoder.fit_transform(housing_cat)
housing_cat_encoded
----
array([0, 0, 4, ..., 1, 0, 3])
```

labelEncoder 的问题问题在于两个相邻的数值被认为是相近的，但很明显在大多数情况下这种编码方式比较随机，无法体现顺序关系，解决办法是使用 one-hot 编码

```python
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1,1))
housing_cat_1hot # 返回值是一个 sparse 矩阵，sparse 矩阵只存储了有效信息，可节省空间

# 转换为 dense 矩阵
housing_cat_1hot.toarray()
----
array([[1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 1.],
       ...,
       [0., 1., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       [0., 0., 0., 1., 0.]])
```

````python
# 直接返回 dense 矩阵，如果想得到 sparse 矩阵，将 `sparse_output=True` 设入 `LabelBinarizer` 构造函数
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat_1hot = encoder.fit_transform(housing_cat)
housing_cat_1hot
----
array([[1, 0, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 0, 0, 0, 1],
       ...,
       [0, 1, 0, 0, 0],
       [1, 0, 0, 0, 0],
       [0, 0, 0, 1, 0]])
````

### 自定义 Transformers
1. 实现 `fit()`、`transform()`、`fit_transform()` 接口
2. 继承 TransformerMixin 后会自动拥有 `fit_transform()` 接口
3. 继承 BaseEstimator 类后会获得额外的 `get_params()` 和 `set_params()` 接口

例子如下：

```python
# 该例子可以让你设置一个超参数，以告诉你增加某个特征是否能对模型有帮助
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
    
    
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

### Feature Scaling

* min-max scaling: end up ranging from 0 to 1; SK-Learn: `MinMaxScaler`; `feature_range` let you change the range if you don't want 0-1 for some reason.
* standardization: **standardization 不会受到异常值的干扰**，例如：假设异常值为 100，Min-Max 会使所有的值从 0-15 归档到 0-0.15，而标准化不会太受该异常值干扰。SK-Learn：`StandardScaler`

scalers 只应该作用于训练集，不应作用于测试集和预测集

### Transformation Pipelines
Pipeline 将每个 transformer 的输出作为下一个 transformer 的输入，下面是运用在数字属性上的 pipeline 的例子

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
            # (name,estimator) 对，名字可以随便起
            # 除最后一个外，所有 estimators 必须为 transformers （实现了 fit_transform() 方法）
            ('imputer', Imputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder()),
            ('std_scaler', StandardScaler()), 
        ])
# pipeline 对象调用的方法和最后一个 estimator 的方法对应
housing_num_tr = num_pipeline.fit_transform(housing_num)
```

#### FeatureUnion
当又要处理数字特征，又要处理文字特征时，可使用 FeatureUnion，它让多个 pipeline 并行执行，当全部执行结束时，再将它们 concat 起来一起返回

```python
from sklearn.pipeline import FeatureUnion
from sklearn_features.transformers import DataFrameSelector

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
             ('selector', DataFrameSelector(num_attribs)),
             ('imputer', Imputer(strategy="median")),
             ('attribs_adder', CombinedAttributesAdder()),
             ('std_scaler', StandardScaler()),
])

cat_pipeline = Pipeline([
             ('selector', DataFrameSelector(cat_attribs)),
             ('label_binarizer', LabelBinarizer()),
])

full_pipeline = FeatureUnion(transformer_list=[
             ("num_pipeline", num_pipeline),
 #            ("cat_pipeline", cat_pipeline),
])

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape
```

## 选择和训练模型

### 训练和分析训练集
调用线性模型，该模型的 MSE 较大，意味着欠拟合，即特征未提供足够的信息来进行预测，或模型不够强大。

```python
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 对比一些预测数据和他们的标签
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:\t", lin_reg.predict(some_data_prepared))
print("Labels:\t\t", list(some_labels))
----
Predictions:	 [206563.06068576 318589.03841011 206073.20582883  71351.11544056
 185692.95569414]
Labels:		 [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]
```

````python
# 输出MSE
from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
----
69422.88161769879
````

解决欠拟合的方法为：

1. 选择一个更复杂的模型
2. 增加其他更好的特征
3. 减少模型的限制，该模型没有使用正则化，所以此选项可不考虑

下面换一个决策树回归模型（DecisionTreeRegressor）

```python
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
----
0.0
```

从上面 MSE 为 0 可以看出，该模型过拟合了

### 使用交叉验证评估模型

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())
    
display_scores(rmse_scores)
----
Scores: [74010.28770706 74680.64882796 74773.57241916 71768.12641187
 75927.45258799 74781.87802591 73151.93148335 72730.44601226
 72628.73907481 74100.34761688]
Mean: 73855.343016726
Standard deviation: 1199.2342001940942
```

```python
# 对线性模型使用交叉验证
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10) 
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
----
Scores: [67383.78417581 67985.10139708 72048.46844728 74992.50810742
 68535.66280489 71602.89821633 66059.1201932  69302.44278968
 72437.02688935 68368.6996472 ]
Mean: 69871.57126682388
Standard deviation: 2630.4324574585044
```

### 使用随机森林
**Ensemble Learning**: Building a model on top of many other models

```python
from sklearn.ensemble import RandomForestRegressor 
forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
    scoring="neg_mean_squared_error", cv=10) 
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
----
Scores: [52716.39575252 51077.36847995 53916.75005202 55501.91073001
 52624.70886263 56367.33336096 52139.5370373  53443.45594517
 55513.29552081 54751.65501867]
Mean: 53805.24107600411
Standard deviation: 1618.473853712107
```

解决过拟合的办法：

1. 简化模型
2. 使用正则化
3. 加大训练数据

## 调试模型
### Grid Search
试不同的超参数，直到找到一个最佳组合。使用 `GridSearchCV`，你只需要设置你想实验的参数，它会使用 CrossValidation 尝试所有可能的组合

例如

```python
from sklearn.model_selection import GridSearchCV
# 3*4 + 2*3 种组合，每个模型训练 5 次
param_grid = [
        {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
        {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                               scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)
# 查看最佳参数组合
grid_search.best_params_
----
{'max_features': 6, 'n_estimators': 30}
```

### 重要性分析

```python
# 根据重要性分析，你可以丢弃一些无用的 feature
feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"] 
cat_one_hot_attribs = list(encoder.classes_)
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
----
[(0.4085511709593715, 'median_income'),
 (0.1274639391269915, 'pop_per_hhold'),
 (0.10153999652040019, 'bedrooms_per_room'),
 (0.09974644399457142, 'longitude'),
 (0.09803482684236019, 'latitude'),
 (0.055005428384214745, 'housing_median_age'),
 (0.047782933377019284, 'rooms_per_hhold'),
 (0.0165562182216361, 'population'),
 (0.01549536838937868, 'total_rooms'),
 (0.014996404177845452, 'total_bedrooms'),
 (0.014827270006210978, 'households')]
```

### 在测试集上评估模型

```python
final_model = grid_search.best_estimator_
X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()
X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse) # => evaluates to 48,209.6
final_rmse
----
49746.60716972495
```

上预发前，需要展示你的解决方案：

1. 你学到了什么
2. 什么可行？什么不可行
3. 你做了哪些假设
4. 系统的限制是什么？

## 上线时需要注意什么

1. 持续监控，避免因为数据持续更新，导致模型退化
2. 采样预测数据，并对其进行评估，以监控模型效果
3. 定期重新训练模型，例如每6个月
4. 定期做全量预测



以上是该书第二张的学习笔记，你也可以下载 [Jupyter NoteBook](https://github.com/jieniu/HOML-exercises/blob/master/chapter2/ch2_note.ipynb) 来具体操练一下。