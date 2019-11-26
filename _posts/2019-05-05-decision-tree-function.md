---
layout: post
title: "分类器：决策树参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
header-mask: 0.4
catalog: true
tags:

  - Sklearn
  - 数据科学
  - 机器学习
---


## 决策树文档

**决策树用法**
```
class sklearn.tree.DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)
```

**参数**
- `criterion`:string, optional (default="gini")
    - criterion='gini',分裂节点时评价准则是Gini指数。
    - criterion='entropy',分裂节点时的评价指标是信息增益。
    
    
- `max_depth`:int or None, optional (default=None)。指定树的最大深度。
    - 如果为None，表示树的深度不限。直到所有的叶子节点都是纯净的，即叶子节点
            中所有的样本点都属于同一个类别。或者每个叶子节点包含的样本数小于min_samples_split。
            
            
- `splitte`r:string, optional (default="best")。指定分裂节点时的策略。
    - splitter='best',表示选择最优的分裂策略。
    - splitter='random',表示选择最好的随机切分策略。
    
    
- `min_samples_split`:int, float, optional (default=2)。表示分裂一个内部节点需要的最少样本数。
    - 如果为整数，则min_samples_split就是最少样本数。
    - 如果为浮点数(0到1之间)，则每次分裂最少样本数为ceil(min_samples_split * n_samples)
    
    
- `min_samples_leaf`: int, float, optional (default=1)。指定每个叶子节点需要的最少样本数。
    - 如果为整数，则min_samples_split就是最少样本数。
    - 如果为浮点数(0到1之间)，则每个叶子节点最少样本数为ceil(min_samples_leaf * n_samples)
    
    
- `min_weight_fraction_leaf`:float, optional (default=0.)，指定叶子节点中样本的最小权重。


- `max_features`:int, float, string or None, optional (default=None).搜寻最佳划分的时候考虑的特征数量。
    - 如果为整数，每次分裂只考虑max_features个特征。
    - 如果为浮点数(0到1之间)，每次切分只考虑int(max_features * n_features)个特征。
    - 如果为'auto'或者'sqrt',则每次切分只考虑sqrt(n_features)个特征
    - 如果为'log2',则每次切分只考虑log2(n_features)个特征。
    - 如果为None,则每次切分考虑n_features个特征。
    - 如果已经考虑了max_features个特征，但还是没有找到一个有效的切分，那么还会继续寻找下一个特征，直到找到一个有效的切分为止。
    
    
- `random_state`:int, RandomState instance or None, optional (default=None)
    - 如果为整数，则它指定了随机数生成器的种子。
    - 如果为RandomState实例，则指定了随机数生成器。
    - 如果为None，则使用默认的随机数生成器。
    
    
- `max_leaf_nodes`: int or None, optional (default=None)。指定了叶子节点的最大数量。
     - 如果为None,叶子节点数量不限。
     - 如果为整数，则max_depth被忽略。
     
     
- `min_impurity_decrease`:float, optional (default=0.)
     - 如果节点的分裂导致不纯度的减少(分裂后样本比分裂前更加纯净)大于或等于min_impurity_decrease，则分裂该节点。
     - 加权不纯度的减少量计算公式为：
     
         N_t / N * (impurity - N_t_R / N_t * right_impurity- N_t_L / N_t * left_impurity)
         
         
         其中
         
         N是样本的总数，N_t是当前节点的样本数，N_t_L是分裂后左子节点的样本数，
         N_t_R是分裂后右子节点的样本数。impurity指当前节点的基尼指数，right_impurity指
         分裂后右子节点的基尼指数。left_impurity指分裂后左子节点的基尼指数。
         
         
         
- `min_impurity_split`:float
     - 树生长过程中早停止的阈值。如果当前节点的不纯度高于阈值，节点将分裂，否则它是叶子节点。
         这个参数已经被弃用。用min_impurity_decrease代替了min_impurity_split。
         
         
- `class_weight`:dict, list of dicts, "balanced" or None, default=None,类别权重的形式为{class_label: weight}
     - 如果没有给出每个类别的权重，则每个类别的权重都为1。
     - 如果class_weight='balanced'，则分类的权重与样本中每个类别出现的频率成反比。
         计算公式为：n_samples / (n_classes * np.bincount(y))
     - 如果sample_weight提供了样本权重(由fit方法提供)，则这些权重都会乘以sample_weight。
     
     
- `presort`:bool, optional (default=False)
     - 指定是否需要提前排序数据从而加速训练中寻找最优切分的过程。设置为True时，对于大数据集会减慢总体的训练过程；但是对于一个小数据集或者设定了最大深度的情况下，会加速训练过程。
        
        
        
**属性**
- `classes_`:array of shape = [n_classes] or a list of such arrays
        类别的标签值。
- `feature_importances_` : array of shape = [n_features]
        特征重要性。越高，特征越重要。
        特征的重要性为该特征导致的评价准则的（标准化的）总减少量。它也被称为基尼的重要性
- `max_features_` : int
        max_features的推断值。
- `n_classes_ `: int or list
        类别的数量
- `n_features_` : int
        执行fit后，特征的数量
- `n_outputs_` : int
        执行fit后，输出的数量
- `tree_ `: Tree object
        树对象，即底层的决策树。
        
        
**方法**


- `apply(X[, check_input])`:	
     - 返回样本所在叶子的索引


- `decision_path(X[, check_input])`:
     - 返回树中的决策路径

- `fit(X,y)`:
     - 训练模型。


- `get_params([deep])`：
     - 获取此估计器的参数。	


- `predict(X)`： 
     - 预测X的类别	


- `predict_log_proba(X)`: 
     - 预测X为各个类别的概率对数值。


- `predict_proba(X)`:
     - 预测X为各个类别的概率值。

- `score(X, y[, sample_weight])`：
     - 返回给定测试数据平均准确度。


- `set_params(**params)`: 
     - 设置此估计器的参数。




附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)
