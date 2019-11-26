---
layout: post
title: "聚类：kmeans,密度，层次聚类参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
date:       2019-05-10 12:00:00
header-mask: 0.4
catalog: true
tags:
  - Sklearn
  - 数据科学
  - 机器学习
---


## K-means文档

**用法**：
```
class sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, n_init=10, max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm=’auto’)[source]
```

**参数**：

- `n_clusters`: 质心数量，也就是分类数，默认是8个。


- `init`: 初始化质心的选取方式，有三个可选值：’k-means++’， ‘random’，或者传递一个ndarray向量。 此参数指定初始化方法，默认值为 ‘k-means++’

    - k-means++’ 用一种特殊的方法选定初始质心从而能加速迭代过程的收敛
    ```       
    k-means的缺点： 
        在 K-means 算法中 K 是事先给定的，这个 K 值的选定是非常难以估计的。很多时候，事先并不知道给定的数据集应该分成多少个类别才最合适。初始聚类中心的选择对聚类结果有较大的影响，一旦初始值选择的不好，可能无法得到有效的聚类结果，这也成为 K-means算法的一个主要问题, 这在之前机器学习的ppt里有讲述.
        
    初始聚类中心的选择可以用k-means++来解决:
        1.从输入的数据点集合中随机选择一个点作为第一个聚类中心 
        2.对于数据集中的每一个点x，计算它与最近聚类中心(指已选择的聚类中心)的距离D(x) 
        3.选择一个新的数据点作为新的聚类中心，选择的原则是：D(x)较大的点，被选取作为聚类中心的概率较大 
        4.重复2和3直到k个聚类中心被选出来 
        5.利用这k个初始的聚类中心来运行标准的k-means算法
    ```
    
    - random’ 随机从训练数据中选取初始质心。 
    - 如果传递的是一个ndarray，则应该形如 (n_clusters, n_features) 并给出初始质心。 


- `n_init`: 随机初始化的次数，kmeans质心迭代的次数。



- `max_iter`: 最大迭代次数，默认是300。


- `tol`: 误差容忍度最小值。


- `precompute_distances`: 是否需要提前计算距离，auto,True,False三个参数值可选。默认值是auto，
    - 如果选择auto，当样本数*质心数>12兆(million)的时候，就不会提前进行计算，如果小于则会提前计算。提前计算距离会让聚类速度很快，但是也会消耗很多内存。
    - True : 一直提前计算距离
    - False : 从不提前计算距离


- `verbose`：输出日志


- `random_state`： 确定质心初始化的随机数生成，整数, RandomState实例或者None, 可不填 (默认=None)
    - 若填整数，`random_state`是一个随机数生成器的种子。
    - 若填RandomState实例，`random_state`是一个随机数生成器
    - 若填None，使用`np.random`生成随机数


- `copy_x`: 主要起作用于提前计算距离的情况，默认值是True,
    - 如果是True,则表示在源数据的副本上提前计算距离时，不会修改源数据。当precomputing distances时，将数据中心化会得到更准确的结果。如果把此参数值设为True，则原始数据不会被改变。
    - 如果是False，则会直接在原始数据上做修改并在函数返回值时将其还原。但是在计算过程中由于有对数据均值的加减运算，所以数据返回后，原始数据和计算前可能会有细小差别。



- `n_jobs`：整形数。　指定计算所用的进程数。内部原理是同时进行n_init指定次数的计算。 
    - 若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。 
    - 若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。 



- `algorithm`：优化算法的选择，有auto、full和elkan三种选择。
    - full就是一般意义上的K-Means算法，elkan是使用的elkan K-Means算法。默认的auto则会根据数据值是否是稀疏的(稀疏一般指是有大量缺失值)，来决定如何选择full和elkan。如果数据是稠密的，就选择elkan K-means，否则就使用普通的Kmeans算法。优化算法在此不做介绍
    

**属性**:
- `cluster_centers_`:数组，[n_clusters, n_features]

    - 输出聚类的质心。。如果算法在完全收敛之前停止(参见tol和max_iter)，这些将与labels_不一致。
    
- `labels_`:

    - 每个点的标签
    
- `inertia_`:浮点数

    - 样本到其最近聚类中心距离的平方和。
    
- `n_iter_`: 整数

    - 运行的迭代次数。


**方法**

- `fit(X[, y, sample_weight])`: 
     - 计算k-means聚类。 


- `fit_predictt(X[, y, sample_weight])`	
     - 计算簇质心并给每个样本预测类别。 


- `fit_transform(X[, y, sample_weight])`
    - 计算簇并将X转换为群集距离空间。 


- `get_params([deep])`： 
     - 取得估计器的参数。 


- `predict(X[, sample_weight])`
     - 给每个样本估计最接近的簇。 


- `score(X[, y, sample_weight])`
     - 计算聚类误差 


- `set_params(**params)`: 
     - 为这个估计器手动设定参数。 


- `transform(X[,y])`: 将X转换为群集距离空间。 
     - 在新空间中，每个维度都是到集群中心的距离。 请注意，即使X是稀疏的，转换返回的数组通常也是密集的。

## 密度聚类文档

密度聚类无法事先指定类别个数，只能通过去指定每个点的邻域，以及邻域内包含样本点的最少个数，基本概念如邻域，核心对象，密度直达，密度可达，密度相连和详细算法流程在机器学习ppt已经讲述，不再赘述。

**用法**：

```
class sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric=’euclidean’, metric_params=None, algorithm=’auto’, leaf_size=30, p=None, n_jobs=None)[source]
```

**参数**：

- `eps`:即邻域中的r值，可以理解为圆的半径，默认0.5。

- `min_samples`:要成为核心对象的必要条件，即邻域内的最小样本数，默认是5个。

- `metric`:距离计算方式，和层次聚类中的affinity参数类似，可选参数’euclidean’，’l1’，’l2’，’mantattan’，’cosine’，’precomputed’。

- `metric_params`:其他度量函数的参数。

- `algorithm`:最近邻搜索算法参数，auto、ball_tree(球树)、kd_tree（kd树）、brute(暴力搜索)，默认是auto。

- `leaf_size`:最近邻搜索算法参数，当algorithm使用kd_tree或者ball_tree时，停止建子树的叶子节点数量的阈值。

- `p`: 最近邻距离度量参数。只用于闵可夫斯基距离和带权重闵可夫斯基距离中p值的选择，p=1为曼哈顿距离，p=2为欧式距离。

- `n_jobs`：整形数。　指定计算所用的进程数。
  - 若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。 
  - 若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。

**属性**

- `core_sample_indices_`：数组，shape = [n_core_samples]
  - 核心对象的索引。

- `components_ `: 数组, shape = [n_core_samples, n_features]
  - 训练时发现的每个核心样本的副本

- `labels_`：数组，shape = [n_samples]
  - 每个样本点的对应的类别，对于噪声点将赋值为-1。

**方法**

- `fit(X[,y])`：
  - 训练样本

- `fit_predict(X[,y])`：
  - 训练模型并预测每个样本的簇标记

- `get_params([deep])`： 
  - 取得估计器的参数。 

- `set_params(**params)`: 
  - 为这个估计器手动设定参数。 

## 层次聚类文档

层次聚类（hierarchical clustering）可在不同层次上对数据集进行划分，形成树状的聚类结构。AggregativeClustering是一种常用的层次聚类算法，使用自底向上的聚类

**用法**：

```
class sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity=’euclidean’, memory=None, connectivity=None, compute_full_tree=’auto’, linkage=’ward’, pooling_func=’deprecated’)[source]
```

**参数**：

- `n_clusters` ：整数，指定分类簇的数量

- `affinity` ：一个字符串或者可调用对象，用于计算距离。可选参数：’euclidean’，’l1’，’l2’，’mantattan’，’cosine’，’precomputed’，如果linkage=’ward’，则affinity必须为’euclidean’

- `memory` ：用于缓存输出的结果，默认为不缓存

- `connectivity` ：一个数组或者可调用对象或者None，用于指定连接矩阵，根据给定的数据结构为每个样本定义相邻的样本。这可以是连接矩阵本身，也可以是将数据转换为连接矩阵

- `compute_full_tree` ：布尔值 or ‘auto’ (可选)
  通常当训练了n_clusters后，训练过程就会停止，但是如果compute_full_tree=True，则会继续训练从而生成一颗完整的树，在n_clusters上尽早停止树的构造。如果集群的数量与样本的数量相比不小，这对于减少计算时间是有用的。此选项仅在指定连接矩阵时有用

- `linkage` ：字符串，用于指定链接算法,参数可选

  - ward：单链接single-linkage，采用dmindmin
  - complete：全链接complete-linkage算法，采用dmaxdmax
  - average：均连接average-linkage算法，采用davgdavg

  ```
  如何判断两个cluster之间的距离呢？ 
  1.最小距离，单链接Single Linkage 
      两个簇的最近样本决定。 
  2.最大距离，全链接Complete Linkage 
      两个簇的最远样本决定。 
  3.平均距离，均链接Average Linkage 
      两个簇所有样本共同决定。 
  1和2都容易受极端值的影响，而3这种方法计算量比较大，不过这种度量方法更合理。
  ```

- `pooling_func` ：一个可调用对象，它的输入是一组特征的值，输出是一个数，将会在0.22版本移除



**属性**

- `labels` ：
  - 每个样本的簇标记

- `n_leaves_`：
  - 分层树的叶节点数量

- `n_components_`：
  - 连接图中连通分量的估计值

- `children_`：
  - 每个非叶节点的子节点

**方法**

- `fit(X[,y])`：
  - 训练样本

- `fit_predict(X[,y])`：
  - 训练模型并预测每个样本的簇标记

- `get_params([deep])`： 
  - 取得估计器的参数。 

- `set_params(**params)`: 
  - 为这个估计器手动设定参数。 

附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)

