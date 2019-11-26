---
layout: post
title: "分类器：多层神经网络参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
header-mask: 0.4
catalog: true
tags:
  - Sklearn
  - 数据科学
  - 机器学习
---


## 神经网络文档


**用法**：
```
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’, alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10)[source]
```
**参数**:

- `hidden_layer_sizes`，长度=n_layers-2, 默认100
    - 第i个元素表示第i个隐藏层的神经元的个数。


- `activation`
    - 激活函数，默认为relu


- `solver`，默认 ‘adam’，
    - 用来优化权重


- `alpha`，可选的，默认0.0001,
    - 正则化项参数


- `batch_size`，默认‘auto’,
    - 随机优化的minibatches的大小


- `learning_rate`，默认‘constant’，
    - 学习率，用于权重更新


- `max_iter`，默认200，
    - 最大迭代次数。


- `random_state`，可选，默认None，
    - 随机数生成器的状态或种子


- `shuffle`，可选，默认True,只有当solver=’sgd’或者‘adam’时使用，
    - 判断是否在每次迭代时对样本进行打乱。


- `tol`，可选，默认1e-4，
    - 优化的容忍度


- `learning_rate_int`，默认0.001，
    - 初始学习率，控制更新权重的补偿，只有当solver=’sgd’ 或’adam’时使用。


- `power_t`
    - 只有solver=’sgd’时使用，是逆扩展学习率的指数.当learning_rate=’invscaling’，用来更新有效学习率。


- `verbose`
    - 是否将过程打印到stdout


- `warm_start`
    - 当设置成True，使用之前的解决方法作为初始拟合，否则释放之前的解决方法。
 
 
- `momentum`,浮点数,默认0.9
    - 梯度下降的动量更新。应该在0和1之间。仅当求解器= ' sgd '时使用。

- `nesterovs_momentum `
    - 是否使用Nesterov的动量。仅当解算器= ' sgd '和动量> 0时使用。
 
 

- `early_stopping`
    - 当交叉验证分数没有提高时，是否使用early stop终止培训。如果设置为true，它将自动留出10%的训练数据作为验证，并在n_iter_no_change连续时间的验证分数至少没有提高tol时终止训练。只有当求解器= ' sgd '或' adam '时才有效。

- `validation_fraction`
    - 预留用于验证的训练数据的比例，用于提前停止。必须在0和1之间。只有在early_stop为true时才使用。

- `beta_1`
    - adam中第一个矩向量估计值的指数衰减率应为[0,1]，仅当解算器= ' adam '时使用。

- `beta_2`
    - adam中二阶矩矢量估计值的指数衰减率应为[0,1]，仅当求解器= ' adam '时使用。

- `epsilon`
    - 数值稳定性的adam值。只在solver= ' adam '时使用。

- `n_iter_no_change`
    - 不满足tol改进的最大迭代数。只有当求解器= ' sgd '或' adam '时才有效。

**属性**

- `classes_`:
    - 每个输出的类的标签


- `oss_`:
    - 损失函数计算出来的当前损失值


- `coefs_`:
    - 列表中的第i个元素表示i层的权重矩阵


- `intercepts_`:
    - 列表中第i个元素代表i+1层的偏差向量


- `n_iter_ `：
    - 迭代次数


- `n_layers_`:
    - 层数


- `n_outputs_`:
    - 输出的个数


- `out_activation_`:
    - 输出激活函数的名称。
    
**方法**

- `fit(X,y)`
    - 拟合数据


- `get_params([deep])`:
    - 获取参数


- `predict(X)`:
    - 使用MLP进行预测


- `predic_log_proba(X)`:
    - 返回X的对数概率估计


- `predic_proba(X)`：
    - X的概率估计


- `score(X,y[,sample_weight])`:
    - 返回给定测试数据和标签上的平均准确度


- `set_params(**params)`:
    - 设置参数。

附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)
