---
layout: post
title: "分类器：集成学习Random forest和Xgboost参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
header-mask: 0.4
catalog: true
tags:

  - Sklearn
  - 数据科学
  - 机器学习
---


## 随机森林案例


```python
import os
import pickle

import pandas as pd
import sklearn as skl
from sklearn.ensemble import RandomForestClassifier
```

    D:\Users\dell\Anaconda3\lib\site-packages\sklearn\ensemble\weight_boosting.py:29: DeprecationWarning: numpy.core.umath_tests is an internal NumPy module and should not be imported. It will be removed in a future NumPy release.
      from numpy.core.umath_tests import inner1d
    

### 随机森林模型详细文档

在随机森林中（参见 RandomForestClassifier 和 RandomForestRegressor 类）， 集成模型中的每棵树构建时的样本都是由训练集经过有放回抽样得来的（例如，自助采样法-bootstrap sample，这里采用西瓜书中的译法）。 另外，在构建树的过程中进行结点分割时，选择的分割点不再是所有特征中最佳分割点，而是特征的一个随机子集中的最佳分割点。 由于这种随机性，森林的偏差通常会有略微的增大（相对于单个非随机树的偏差），但是由于取了平均，其方差也会减小，通常能够补偿偏差的增加，从而产生一个总体上更好的模型。

注意scikit-learn 的实现是取每个分类器预测概率的平均，而不是让每个分类器对类别进行投票。

更多比较见[jasonfreak的博客](http://www.cnblogs.com/jasonfreak/p/5720137.html)

**用法**：
```
class sklearn.ensemble.RandomForestRegressor(n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)[source]
```

**参数**：

- `n_estimators` : 整数, 可不填 (默认=10)
- `criterion` : 字符串, 可不填 (默认="mse")
    - 决定分叉质量的依据。可以填mse代表均方差，即方差小被认为是更优的。也可以填写mae代表均绝对误差。
- `max_depth` : 整数或者None, 可不填 (默认=None)
    - 树的最大深度。若填None，则会一直分叉直到所有末端节点都只有一个特征，或者所有的叶所包含的样本量小于min_samples_split
- `min_samples_split` : 整数, 浮点数, 可不填 (默认=2)
    - 节点分叉所需最小样本量
        - 若填整数, 则认为`min_samples_split`就是最小的样本量
        - 若填浮点数, 则认为`min_samples_split`是一个最小占比。`ceil(min_samples_split * n_samples)`是最小的样本量。
- `min_samples_leaf` : 整数, 浮点数, 可不填 (默认=1)
    - 一个节点所需的最小样本数量。任意一个分叉点，如果左右侧都至少包含这么多的样本，那么才可以被认为是一个分叉点。调节这个值可以让模型更平稳，尤其是在做回归的时候。
        - 若填整数, 则认为`min_samples_leaf`就是最小的样本量
        - 若填浮点数, 则认为`min_samples_leaf`是一个最小占比。`ceil(min_samples_leaf * n_samples)`是每个节点的最小样本量。
- `min_weight_fraction_leaf` : 浮点数, 可不填 (默认=0.)
    - 叶子节点所需要的最小总权重。如果`sample_weight`未指定，那么等权处理。
- `max_features` : 整数, 浮点数, 字符串或者None, 可不填 (默认="auto")
    - 寻找最优节点分裂时考虑的特征数
        - 若填整数, 则每次分裂考虑特征数量为max_features
        - 若填浮点数, 则max_features是一个比例。每次分裂考虑特征数量为`int(max_features * n_features)`
        - 若填"auto", 则每次分裂考虑特征数量为`max_features=n_features`
        - 若填"sqrt", 则每次分裂考虑特征数量为`max_features=sqrt(n_features)`
        - 若填"log2", 则每次分裂考虑特征数量为`max_features=log2(n_features)`
        - 若填None, 则每次分裂考虑特征数量为`max_features=n_features`
- `max_leaf_nodes` : 整数或者None, 可不填 (默认=None)
    - 生成max_leaf_nodes个节点最优的树。Grow trees with `max_leaf_nodes` in best-first fashion. 节点最优被定义为impurity相对不再减小。如果填None则不限制。
- `min_impurity_decrease` : 浮点数, 可不填 (默认=0.)
    - 若分叉降低的impurity值大于或等于此数则此节点分叉
- `min_impurity_split` : 浮点数, (默认=1e-7)
    - 决定impurity是否过高需要分叉
    - 改用`min_impurity_decrease`
- `bootstrap` : 布尔值, 可不填 (默认=True)
    - 是否用bootstrap方法对样本抽样
- `oob_score` : 布尔值, 可不填 (默认=False)
    - 是否计算[袋外得分](https://scikit-learn.org/stable/auto_examples/ensemble/plot_ensemble_oob.html)
- `n_jobs` : 整数或者None, 可不填 (默认=None)
    - 并行数
        - 若填None，则并行数为1，除非`joblib.parallel_backend`里有预设。
        - 若填-1，则按处理器核的数目并行。
    - 参考[Glossary](https://scikit-learn.org/stable/glossary.html#term-n-jobs)
- `random_state` : 整数, RandomState实例或者None, 可不填 (默认=None)
    - 若填整数，`random_state`是一个随机数生成器的种子。
    - 若填RandomState实例，`random_state`是一个随机数生成器
    - 若填None，使用`np.random`生成随机数
- `verbose` : 整数, 可不填 (默认=0)
    - 控制模型在拟合和预测时的verbosity
- `warm_start` : 布尔值, 可不填 (默认=False)
    - 如果True, 会使用上次调用的结果并通过增加树的方式来集成模型。否则从零开始。参考[Glossary](https://scikit-learn.org/stable/glossary.html#term-warm-start).

**属性**:


- `estimators_` : DecisionTreeRegressor列表
    - 拟合过的子估计器集合
- `feature_importances_` : 形状如[n_features]的数组
    - 返回特征重要程度（越高越重要）
- `n_features_` : 整数
    - 随机森林拟合时的特征维度
- `n_outputs_` : 整数
    - 随机森林拟合时的特征输出维度
- `oob_score_` : 浮点数
    - 袋外得分
- `oob_prediction_` : 形状如[n_features]的数组
    - 训练集使用袋外数据估计做的预测

**方法**:

- `__init__(n_estimators='warn', criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False)`
- `apply(X)`
    - 拟合X，返回各个叶子的指标
- `decision_path(X)`
    - 拟合X，返回随机森林的决策路径
- `fit(X, y[, sample_weight])`
    - 使用训练集(X, y)构建随机森林
- `get_params([deep])`
    - 获得拟合器的参数
- `predict(X)`
    - 返回给定X的回归值
- `score(X, y[, sample_weight])`
    - 返回预测系数的R^2
- `set_params(**params)`
    - 设置拟合器的参数

附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)




**Xgboost用法**
```
class xgboost.XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, silent=True, objective='binary:logistic', booster='gbtree', n_jobs=1, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=0, seed=None, missing=None, **kwargs)
```

**参数**
- `max_depth` : 整数

    - Maximum tree depth for base learners.基学习器的树最大深度
    
    
- `learning_rate` : 浮点数

    - 学习率，控制每次迭代更新权重时的步长，值越小，训练越慢
    
    
- `n_estimators` : 整数

    - 拟合的树的数目
    
    
- `silent` : 布尔值

    - 是否在运行分类器时输出信息
    
    
- `objective` : 
    - 明确学习任务和相应的学习目标或要使用的自定义目标函数 
    
    回归任务
        reg:linear (默认)
        reg:logistic
    二分类
        binary:logistic     概率 
        binary：logitraw   类别
    多分类
        multi：softmax  num_class=n   返回类别
        multi：softprob   num_class=n  返回概率
        
        
- `booster`: 字符串

    - 指定使用哪个学习器: gbtree, gblinear or dart.
    

- `n_jobs` : 整数

    - 用于运行xgboost的并行线程数
    
    
- `gamma` : 浮点数

    - 惩罚项系数，指定节点分裂所需的最小损失函数下降值。
    
    
- `min_child_weight` : 整数

    - 决定最小叶子节点样本权重和。如果在一次分裂中，叶子节点上所有样本的权重和小于min_child_weight则停止分裂，能够有效的防止过拟合，防止学到特殊样本.值越大，越容易欠拟合；值越小，越容易过拟合（值较大时，避免模型学习到局部的特殊样本）,如果设置为0.5则意味着XGBoost将随机的冲整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合
    

- `subsample` : 浮点数

    - 训练每棵树时，使用的数据占全部训练集的比例
    
    
- `colsample_bytree` : 浮点数

    - 行采样的一种，训练每棵树时，使用的特征占全部特征的比例，防止overfitting。在建树前就随机选择一部分特征，然后之后所有叶子结点的分裂都只使用这部分特征
    
    
- `colsample_bylevel` : 浮点数

    - 行采样的一种，一种是按层随机colsample_bylevel,按层随机colsample_bylevel的意思就是，每次分裂一个结点的时候，都要遍历所有的特征和分割点，从而确定最优的分割点，那么如果加入了列采样，会在对同一层内每个结点分裂之前，先随机选择一部分特征，于是只需要遍历这部分的特征，来确定最优的分割点。
    
    
    
- `reg_alpha` : 浮点数

    - L1正则化系数
    
    
    
- `reg_lambda` : 浮点数

    - L2正则化系数
    
    
- `scale_pos_weight` : 浮点数

    - 用来调节正负样本不均衡问题的，用助于样本不平衡时训练的收敛，源代码里是增大了少数样本的学习率，二分类任务中，当正负样本比例失衡时，设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10。
    
    
- `random_state` : 整数

    - 指定了随机数生成器的种子
    
    
- `missing` : 浮点数, 可选参数

    - 数据中的值，需要以缺失值的形式显示。如果None，默认为np.nan。


**方法**
- `apply(X)`:形状如[n_samples, n_trees]的数组

    - 对于X中的每个数据点x和每棵树，返回叶子x的索引
    
- `coef_`:形状如 [n_features] or [n_classes, n_features]的数组

    - 只有当选择线性模型作为基本学习器(boost =gblinear)时，才定义系数。它没有为其他基本学习类型定义，比如树学习器(boost =gbtree)
    
- `evals_result()`

    - 返回评估结果
    
- `feature_importances_`: 形状如[n_features]的数组

    - 只有当选择决策树模型作为基学习器(boost =gbtree)时，才有特征重要性
    
    
- `fit`:
    ```fit(X, y, sample_weight=None, eval_set=None, eval_metric=None, verbose=True, xgb_model=None, sample_weight_eval_set=None)
    ```
    - 拟合梯度下降分类器

        `X ` ：特征矩阵
        
        `y ` ：标签
        
        `sample_weight `：每个实例的权重
        
        `eval_set` ：(list, optional)——一个(X, y)对的列表，用作早期停止的验证集
        
        `sample_weight_eval_set`： (list, optional)——表单[L_1, L_2，…，L_n]的列表，其中每个L_i都是第i个验证集上的实例权重列表。
        
        `eval_metric` (str，可调用的，可选的)——如果是str，那么它应该是一个内置的评估指标。如果可调用，则为自定义评估指标。调用标记是func(y_predicted, y_true)，其中y_true是一个DMatrix对象，因此可能需要调用get_label方法。它必须返回一个str值对，其中str是求值函数的名称，value是求值函数的值。这个目标总是最小化。
        
        `verbose` ：(bool)——如果使用了verbose和评估集，则将验证集上测量的评估度量写入stderr。
        
        `gb_model` ：(str) -训练前要加载的xgb模型的文件名
    
    
    
- `intercept_`

    - 只有当选择线性模型作为基本学习器(boost =gblinear)时，才定义系数。它没有为其他基本学习类型定义，比如树学习器(boost =gbtree)

- `load_model(fname)`

    - 从文件中加载模型。模型是由XGBoost内部二进制格式加载的，该格式在各种XGBoost接口之间是通用的


- `predict(X)`： 
     - 预测X的类别	


- `predict_proba(X)`:
     - 预测X为各个类别的概率值。


- `save_model(fname)`

    - 将模型保存到文件中，该模型以XGBoost内部二进制格式保存，该格式在各种XGBoost接口之间是通用的。

附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)
