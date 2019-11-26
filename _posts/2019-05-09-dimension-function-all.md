---
layout: post
title: "降维：PCA,KPCA,TSNE参数用法解读"
author: "Leo"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
header-mask: 0.4
catalog: true
tags:
  - Sklearn
  - 数据科学
  - 机器学习
---


## PCA文档

**用法**：

```
class sklearn.decomposition.PCA(n_components=None, copy=True, whiten=False, svd_solver=’auto’, tol=0.0, iterated_power=’auto’, random_state=None)[source]
```

**参数**：

- `n_components`：PCA算法中所要保留的主成分个数n，也即保留下来的特征个数n。最常用的做法是直接指定降维到的维度数目，此时n_components是一个大于等于1的整数。当然，也可以指定主成分的方差和所占的最小比例阈值，让PCA类自己去根据样本特征方差来决定降维到的维度数，此时n_components是一个（0，1]之间的数。还可以将参数设置为"mle", 此时PCA类会用MLE算法根据特征的方差分布情况自己去选择一定数量的主成分特征来降维。也可以用默认值，即不输入n_components，此时n_components=min(样本数，特征数)。

- `copy` : bool，True或者False，缺省时默认为True。
  表示是否在运行算法时，将原始训练数据复制一份。若为True，则运行PCA算法后，原始训练数据的值不会有任何改变，因为是在原始数据的副本上进行运算；若为False，则运行PCA算法后，原始训练数据的值会改，因为是在原始数据上进行降维计算。

- `whiten` ：判断是否进行白化。所谓白化，就是对降维后的数据的每个特征进行归一化，让方差都为1,即使得每个特征具有相同的方差.对于PCA降维本身来说，一般不需要白化。如果你PCA降维后有后续的数据处理动作，可以考虑白化。默认值是False，即不进行白化。

- `svd_solver`：即指定奇异值分解SVD的方法，由于特征分解是奇异值分解SVD的一个特例，一般的PCA库都是基于SVD实现的。有4个可以选择的值：{‘auto’, ‘full’, ‘arpack’, ‘randomized’}。
  - 'randomized' 一般适用于数据量大，数据维度多同时主成分数目比例又较低的PCA降维，它使用了一些加快SVD的随机算法。
  - 'full' 则是传统意义上的SVD，使用了scipy库对应的实现。
  - 'arpack' 和randomized的适用场景类似，区别是randomized使用的是scikit-learn自己的SVD实现，而arpack直接使用了scipy库的sparse SVD实现。当svd_solve设置为'arpack'时，保留的成分必须少于特征数，即不能保留所有成分。
  - 默认是'auto'，即PCA类会自己去在前面讲到的三种算法里面去权衡，选择一个合适的SVD算法来降维。一般来说，使用默认值就够了。

- `tol`: 当svd_solver == ‘arpack’时，奇异值的容忍度

- `iterated_power`: 当svd_solver == ‘randomized’时，power method的迭代次数

- `random_state`: 整数, RandomState实例或者None, 可不填 (默认=None)

  在svd_solver == ‘arpack’ or ‘randomized’时使用.

  - 若填整数，`random_state`是一个随机数生成器的种子。
  - 若填RandomState实例，`random_state`是一个随机数生成器
  - 若填None，使用`np.random`生成随机数

**属性**

- `components_` : array, shape (n_components, n_features)
  - 特征空间的主轴，表示数据中最大方差的方向。按explained_variance_排序。

- `explained_variance_` : array, shape (n_components,)
  - 代表降维后的各主成分的方差值。方差值越大，则说明越是重要的主成分。

- `explained_variance_ratio_` : array, shape (n_components,)
  - 返回所保留各个特征的方差百分比，它代表降维后的各主成分的方差值占总方差值的比例，这个比例越大，则越是重要的主成分。如果n_components没有赋值，则所有特征都会返回一个数值且解释方差之和等于1。

- `singular_values_ `: array, shape (n_components,)   
  - 对应于每个选定主成分的奇异值。奇异值等于低维空间中n_components变量的2范数。

- `n_components_`：
  - 返回所保留的特征个数。

- `mean_ `: array, shape (n_features,)
  - 每个特征的均值。

**方法**

- `fit(X)`，表示用数据X来训练PCA模型。
  - 函数返回值：调用fit方法的对象本身。比如pca.fit(X)，表示用X对pca这个对象进行训练。

- `fit_transform(X)`
  - 用X来训练PCA模型，同时返回降维后的数据。newX=pca.fit_transform(X)，newX就是降维后的数据。



- `get_covariance()`
  - 使用生成模型计算数据协方差。

- `get_precision()`	
  - 利用生成模型计算数据精度矩阵。
- `transform(X)`
  - 取得估计器的参数。 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。

- `inverse_transform()`
  - 取得估计器的参数。 将降维后的数据转换成原始数据，X=pca.inverse_transform(newX)。



- `score(X[, y])`
  - 取得估计器的参数。 返回所有样本的平均对数极大似然估计。
- `score_samples(X)`
  - 返回每个样本的对数极大似然估计。

- `get_params([deep])`： 
  - 取得估计器的参数。 

- `set_params(**params)`: 
  - 为这个估计器手动设定参数。 



## 核PCA文档

核PCA和PCA都是用来做无监督数据处理的，PCA是把m维的数据降至k维。核PCA是把m维的数据升至k维。但是他们共同的目标都是让数据在目标维度中（线性）可分，即PCA的最大可分性。

在sklearn中，核PCA和pca的使用基本一致，接口都是一样的。核PCA需要指定核函数，不然默认线性核。

**用法**：
```
class sklearn.decomposition.KernelPCA(n_components=None, kernel=’linear’, gamma=None, degree=3, coef0=1, kernel_params=None, alpha=1.0, fit_inverse_transform=False, eigen_solver=’auto’, tol=0, max_iter=None, remove_zero_eig=False, random_state=None, copy_X=True, n_jobs=None)[source]
```

**参数**：
- `n_components `: 整数
    - 主成分的数量。如果没有，则保留所有非零主成分


- `kernel `: 可选核函数，“linear” | “poly” | “rbf” | “sigmoid” | “cosine” | “precomputed”，默认“linear”


- `gamma` : 浮点数, 默认=1/n_features
    - rbf、poly和sigmoid核的核系数，其他核函数可以忽略。


- `degree` : 整数, 默认=3
    - 多项式核的维数，其他核函数可以忽略。


- `coef0` : float, 默认=1
    - 多项式 and sigmoid 核的独立项. 其他核函数可以忽略。


- `kernel_params `: mapping of string to any, default=None
    - 作为可调用对象传递的核的参数. Ignored by other kernels.


- `alpha `: int, 默认=1.0
    - 逆变换的岭回归的超参数 (当 fit_inverse_transform = True).


- `fit_inverse_transform `: 布尔值 default=False
    - 非预计算核的逆变换


- `eigen_solver` : string [‘auto’|’dense’|’arpack’], 默认=’auto’
    - 选择要使用的特征求解器。如果n_components小于训练样本的数量，arpack可能会更有效率，而不是稠密的特征解


- `tol `: float, 默认=0
    - arpack的收敛判据. 如果0，则由arpack选择最优值


- `max_iter` : int, 默认=None
    - arpack的最大迭代次数. 如果没有，则由arpack选择最优值。


- `remove_zero_eig `: boolean, 默认=False
    
- 如果为True，则删除所有特征值为零的成分
    
- `random_state`: 整数, RandomState实例或者None, 可不填 (默认=None)

    在Used when eigen_solver == ‘arpack’.时使用.

    - 若填整数，`random_state`是一个随机数生成器的种子。
    - 若填RandomState实例，`random_state`是一个随机数生成器
    - 若填None，使用`np.random`生成随机数 



- `copy_X `: 布尔值, 默认=True
    - 如果为True, 输入X将被模型复制并存储在X_fit_属性中。如果不对X做进一步的修改，设置copy_X=False可以通过存储引用来节省内存。



- `n_jobs`：整形数。　指定计算所用的进程数。
    - 若值为 -1，则用所有的CPU进行运算。若值为1，则不进行并行运算，这样的话方便调试。 
    - 若值小于-1，则用到的CPU数为(n_cpus + 1 + n_jobs)。因此如果 n_jobs值为-2，则用到的CPU数为总CPU数减1。
    

**属性**
- `lambdas_ `: array, (n_components,)
    - 中心核矩阵的特征值按递减顺序排列。如果没有设置n_components和remove_zero_eig，则存储所有值。


- `alphas_` : array, (n_samples, n_components)
    - 中心核矩阵的特征向量。如果没有设置n_components和remove_zero_eig，则存储所有组件。


- `dual_coef_` : array, (n_samples, n_features)
    - 逆变换矩阵。只有当fit_inverse_transform为True时才可用。


- `X_transformed_fit_` : array, (n_samples, n_components)
    - 拟合数据在核主成分上的投影。只有当fit_inverse_transform为True时才可用。


- `X_fit_` : (n_samples, n_features)
    - 用于拟合模型的数据。如果copy_X=False，则X_fit_是参考数据。

**方法**

- `fit(X[, y])`	
    - 根据X中的数据拟合模型。


- `fit_transform(X[, y])`
    - 根据X中的数据拟合模型并转换X。


- `inverse_transform(X)	`
    - 将X变换回原始空间。


- `transform(X)`
    - 取得估计器的参数。 将数据X转换成降维后的数据。当模型训练好后，对于新输入的数据，都可以用transform方法来降维。
    
- `get_params([deep])`： 
     - 取得估计器的参数。 


- `set_params(**params)`: 
     - 为这个估计器手动设定参数。 



## TSNE文档

TSNE即t-distributed Stochastic Neighbor Embedding，TSNE是一种可视化高维数据的工具。它将数据点之间的相似性转化为联合概率，并试图最小化低维嵌入和高维数据联合概率之间的Kullback-Leibler divergence。t-SNE具有非凸的代价函数，即初始化不同，得到的结果也不同。

t-分布随机邻域嵌入（t-Distributed Stochastic Neighbor Embedding，t-SNE），是目前一个非常流行的对高维度数据进行降维的算法, 由Laurens van der Maaten和 Geoffrey Hinton于2008年提出，其在降低维度同时，同时试图保持相似的实例相邻，并将不相似的实例分开。它主要用于可视化， 尤其是用于可视化高维空间中的实例（例如，可以将MNIST图像降维到2D可视化);
t-SNE的工作对数学领域的知识要求比较高。t-SNE的大致作用是试图保持邻域内的距离。

**用法**：

```
class sklearn.manifold.TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0, learning_rate=200.0, n_iter=1000, n_iter_without_progress=300, min_grad_norm=1e-07, metric=’euclidean’, init=’random’, verbose=0, random_state=None, method=’barnes_hut’, angle=0.5)
```

**参数**

- `n_components`：int，可选（默认值：2）
  - 嵌入式空间的维度。

- `perplexity`：浮点型，可选（默认：30）
  - 较大的数据集通常需要更大的perplexity。考虑选择一个介于5和50之间的值。由于t-SNE对这个参数非常不敏感，所以选择并不是非常重要。

- `early_exaggeration`：float，可选（默认值：4.0）
  - 控制原始空间中的自然集群在嵌入空间中的紧密程度，以及它们之间的空间大小,这个参数的选择不是非常重要。

- `learning_rate`：float，可选（默认值：1000）
  - 学习率可以是一个关键参数。它应该在100到1000之间。如果在初始优化期间成本函数增加，则早期夸大因子或学习率可能太高。如果成本函数陷入局部最小的最小值，则学习速率有时会有所帮助。

- `n_iter`：int，可选（默认值：1000）
  - 优化的最大迭代次数。至少应该250。

- `n_iter_without_progress`：int，可选（默认值：30）
  - 在中止优化之前，没有进展的最大迭代次数,控制停止条件。

- `min_grad_norm`：float，可选（默认值：1E-7）
  - 如果梯度范数低于此阈值，则优化将被中止。

- `metric`：字符串或可迭代的，可选，
  - 计算特征数组中实例之间的距离时使用的度量。如果度量标准是字符串，则它必须是scipy.spatial.distance.pdist为其度量标准参数所允许的选项之一，或者是成对列出的度量标准.PAIRWISE_DISTANCE_FUNCTIONS。如果度量是“预先计算的”，则X被假定为距离矩阵。或者，如果度量标准是可调用函数，则会在每对实例（行）上调用它，并记录结果值。可调用应该从X中获取两个数组作为输入，并返回一个表示它们之间距离的值。默认值是“euclidean”，它被解释为欧氏距离的平方。

- `init`：字符串，可选（默认值：“random”）嵌入的初始化。可能的选项是“随机”和“pca”。 
  - PCA初始化不能用于预先计算的距离，并且通常比随机初始化更全局稳定。

- `random_state`：int或RandomState实例或None（默认）
  - 伪随机数发生器种子控制。如果没有，请使用numpy.random单例。请注意，不同的初始化可能会导致成本函数的不同局部最小值。

- `method`：字符串（默认：'barnes_hut'）
  - 默认情况下，梯度计算算法使用在O（NlogN）时间内运行的Barnes-Hut近似值。 method ='exact'将运行在O（N ^ 2）时间内较慢但精确的算法上。当最近邻的误差需要好于3％时，应该使用精确的算法。但是，确切的方法无法扩展到数百万个示例。0.17新版​​功能：通过Barnes-Hut近似优化方法。

- `angle`：float（默认值：0.5）
  - 仅当method ='barnes_hut'时才使用这是Barnes-Hut T-SNE的速度和准确性之间的折衷。 'angle'是从一个点测量的远端节点的角度大小（在[3]中称为theta）。如果此大小低于'角度'，则将其用作其中包含的所有点的汇总节点。该方法对0.2-0.8范围内该参数的变化不太敏感。小于0.2的角度会迅速增加计算时间和角度，因此0.8会快速增加误差。

**属性**

- `embedding_` : array-like, shape (n_samples, n_components)
  - 存储嵌入向量。

- `kl_divergence_` : float
  - 优化后的Kullback-Leibler divergence.
- `n_iter_` : int
  - 运行的迭代次数。

**方法**

- `fit(X[, y])`	
  - 将X投射到嵌入空间中。

- `fit_transform(X[, y])`	
  - 将X投射到嵌入空间中并返回转换后的输出

- `get_params([deep])	`
  - 获取此估计器的参数。

- `set_params(**params)	`
  - 设置此估计器的参数。


附上[scikit-learn (sklearn) 官方文档中文版](http://sklearn.apachecn.org/#/)
