---
layout: post
title: "分类器：Logistic regression参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
header-mask: 0.4
catalog: true
tags:

  - Sklearn
  - 数据科学
  - 机器学习
---
## 逻辑回归文档

**用法**：
```
class sklearn.linear_model.LogisticRegression(penalty=’l2’, dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver=’warn’, max_iter=100, multi_class=’warn’, verbose=0, warm_start=False, n_jobs=None)
```
**参数**:
- `penalty`:字符串型，’l1’ or ‘l2’，默认：’l2’；正则化类型。


- `dual`:布尔型，默认：False。如果为True，则求解对偶形式；如果为False，则求解原始形式，当样本数>特征数时，令dual=False；用于liblinear中L2正则化（只是在penalty='l2' 且solver='liblinear' 有对偶形式）。


- `tol`:浮点型，默认：1e-4；迭代终止判断的误差范围。


- `C`:浮点型，默认：1.0；其值等于正则化强度的倒数，为正的浮点数。数值越小表示正则化越强。


- `fit_intercept`:布尔型，默认：True；指定是否应该向决策函数添加常量(即偏差或截距)，即b,如果为False，则不会计算b值（模型会假设你的数据已经中心化）。。


- `intercept_scaling`:浮点型，默认为1；仅仅当solver是”liblinear”时有用,当采用 fit_intercept 时，相当于人造一个特征出来，该特征恒为 1，其权重为b,在计算正则化项的时候，该人造特征也被考虑了。因此为了降低人造特征的影响，需要提供 intercept_scaling。 。


- `class_weight`:默认为None；与“{class_label: weight}”形式中的类相关联的权重。如果为字符串 'balanced'：则每个分类的权重与该分类在样品中出现的频率成反比。如果未指定，则每个分类的权重都为 1。


- `random_state`:整型，默认None；当“solver”==“sag”或“liblinear”时使用。在变换数据时使用的伪随机数生成器的种子。如果是整数,random_state为随机数生成器使用的种子;若为RandomState实例，则random_state为随机数生成器;如果没有，随机数生成器就是' np.random '使用的RandomState实例。


- `solver`:{'newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'}，默认: 'liblinear'；用于指定优化问题的算法。
    - `newton-cg`：使用牛顿法。
    - `lbfgs`：使用L-BFGS拟牛顿法。
    - `liblinear`：使用 liblinear。
    - `sag`：使用 Stochastic Average Gradient descent 算法。
    
    
    注意：
    
    对于规模小的数据集，'liblearner'比较适用；对于规模大的数据集，'sag'比较适用。'newton-cg'、'lbfgs'、'sag' 只处理penalty=‘12’的情况。

    对于小数据集来说，“liblinear”是个不错的选择，而“sag”和'saga'对于大型数据集会更快。对于多类问题，只有'newton-cg'， 'sag'， 'saga'和'lbfgs'可以处理多项损失;“liblinear”仅限于“one-versus-rest”分类。


 
- `max_iter`:最大迭代次数，整型，默认是100；


- `multi_class`:字符串型，{ovr'， 'multinomial'}，默认:'ovr'；定对于多分类问题的策略，可以为如下的值。
    `ovr` ：采用 one-vs-rest 策略。
    `multinomial`：直接采用多分类逻辑回归策略。


- `verbose`:整型，默认是0；用于开启/关闭迭代中间输出的日志。。


- `warm_start`:布尔型，默认为False；如果为True，那么使用前一次训练结果继续训练，否则从头开始训练。对liblinear解码器无效。


- `n_jobs`:整型，默认是1；定任务并行时的 CPU 数量。如果为 -1 则使用所有了用的 CPU。


**属性**:

- `classes_`: array, shape (n_classes，)分类器已知的类标签列表。返回数组，shape=(n_samples，) if n_classes == 2 else (n_samples, n_classes)


- `coef_`:数组，shape (1, n_features)或(n_classes, n_features)
    - 决策函数中特征的系数。当给定的问题是二进制时，coef_的形状为(1,n_features)。特别是当multi_class= '多项'时，coef_对应结果1 (True)，coef_对应结果0 (False)。
    
    
- `intercept_`:数组，shape(1，)或(n_classes，)
    - 截距(又称偏差)添加到决策函数中。如果将fit_intercept设置为False，则将截距设置为零。当给定的问题是二进制时，intercept_的形状为(1，)。特别是当multi_class= '多项'时，intercept_对应结果1 (True)， -intercept_对应结果0 (False)。
    
    
- `n_iter_`:数组，形状(n_classes，)或(1，)，所有类的实际迭代次数。如果是二进制或多项式，它只返回一个元素。对于线性求解器，只给出了所有类的最大迭代次数。


在版本0.20中更改:在SciPy <= 1.0.0中，lbfgs迭代的次数可能超过max_iter。


**方法**



- `decision_function(X)`: 
    - 预测样本的置信度。


- `densify()`:
    - 系数矩阵转换为密集矩阵格式。


- `fit(X, y[, sample_weight])`：
    - 根据给定的训练数据拟合模型。


- `get_params([deep])`：
    - 获取此估计器的参数。	


- `predict(X)`： 
    - 预测样本在X中的类标签。	


- `predict_log_proba(X)`: 
    - 数据X概率估计的对数。


- `predict_proba(X)`:
    - 数据X概率的估计。


- `score(X, y[, sample_weight])`：
    - 返回给定测试数据平均准确度。


- `set_params(**params)`: 
    - 设置此估计器的参数。


- `sparsify()`:
    - 将系数矩阵转换为稀疏格式。

附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)
