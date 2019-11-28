---
layout:     post
title:  "Mnist数据集各种降维技术可视化（上）"
subtitle:   ""
date:       2019-05-27 12:00:00
author:     "Leo"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
tags:
    - 机器学习
    - 降维
mathjax: false
---

人类可以在二维和三维上甚至在第四维来思考。但是机器学习经常要求使用成千上万个维度！即使是非常简单的事情，当在非常高的维度上思考的时候，也会变得难以理解。 降维技术就必不可少，可视化降维结果将更符合人类的思考习惯。幸运的是，目前很多大神为此做了很多努力，有了很多成果。

注意：此ipnb文件在python 3.7.3运行，注意不要一下子运行所有cell，否则cpu会跑不动

```python
#导入相应的库
import numpy as np
import pandas as pd
import os
np.random.seed(42)

%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 15
plt.rcParams['xtick.labelsize']  = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.decomposition import PCA
```


```python
#获取数据集，运行时需要一段时间，等待一下，否则没办法进行下一步
from six.moves import urllib
try:
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1)
    mnist.target = mnist.target.astype(np.int64)
except ImportError:
    from sklearn.datasets import fetch_mldata
    mnist = fetch_mldata('MNIST original')
```


```python
X_raw = mnist["data"]
y_raw = mnist["target"]
```


```python
#有七万张图
X_raw.shape
```

    (70000, 784)



MNIST是一种简单的计算机视觉数据集。它由28×28像素的手写数字图像组成，一个图像有28×28个像素，因此得到了一个28×28的数组，可以将每个数组变为28×28＝784维向量


```python
X_raw[0].shape
```

    (784,)

```python
X_raw[0]
```

    array([  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   3.,  18.,
            18.,  18., 126., 136., 175.,  26., 166., 255., 247., 127.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
            30.,  36.,  94., 154., 170., 253., 253., 253., 253., 253., 225.,
           172., 253., 242., 195.,  64.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,  49., 238., 253., 253., 253., 253.,
           253., 253., 253., 253., 251.,  93.,  82.,  82.,  56.,  39.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
            18., 219., 253., 253., 253., 253., 253., 198., 182., 247., 241.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,  80., 156., 107., 253.,
           253., 205.,  11.,   0.,  43., 154.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,  14.,   1., 154., 253.,  90.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
           139., 253., 190.,   2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,  11., 190., 253.,  70.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,  35., 241., 225., 160., 108.,   1.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  81., 240.,
           253., 253., 119.,  25.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,  45., 186., 253., 253., 150.,  27.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,  16.,  93., 252., 253., 187.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0., 249., 253.,
           249.,  64.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,  46., 130., 183., 253., 253., 207.,   2.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,  39., 148., 229., 253., 253., 253.,
           250., 182.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  24., 114.,
           221., 253., 253., 253., 253., 201.,  78.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,  23.,  66., 213., 253., 253., 253., 253., 198.,  81.,
             2.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,  18., 171., 219., 253., 253.,
           253., 253., 195.,  80.,   9.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,  55.,
           172., 226., 253., 253., 253., 253., 244., 133.,  11.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0., 136., 253., 253., 253., 212., 135.,
           132.,  16.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
             0.,   0.,   0.])




```python
#可视化图片
def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")
```


```python
#图片实在太多了，会运行很长时间，选取其中十分之一的图片
X, X_test, y, y_test = train_test_split(X_raw,y_raw,test_size=0.9)
```


```python
pca = PCA(n_components = 100)
#降维到100个维度
X_reduced = pca.fit_transform(X)
#降维后，在原有的维度上尝试恢复原来的维度
X_recovered = pca.inverse_transform(X_reduced)
```


```python
#任选五张图片可视化
plt.figure(figsize=(15, 4))
plt.subplot(121)
plot_digits(X[::200])
plt.title("原始数据集", fontsize=16)
plt.subplot(122)
plot_digits(X_recovered[::200])
plt.title("压缩后数据集", fontsize=16)
```

    Text(0.5, 1.0, '压缩后数据集')

<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_12_1.png'/>

降维代码很容易写，有现成的工具，太幸福了，人生苦短，我用python, 耗时间的其实是可视化。


```python
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
#重写可视化函数
def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    
    X_normalized = MinMaxScaler().fit_transform(X)
    
    neighbors = np.array([[10., 10.]])
    
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=[cmap(digit / 9)])
    plt.axis("off")
    ax = plt.gcf().gca()  # get current axes in current figure
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)
```

## Factor Analysis

```python
from sklearn.decomposition import FactorAnalysis
X_fa_reduced = FactorAnalysis(n_components = 2).fit_transform(X)
plot_digits(X_fa_reduced, y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_16_0.png'/>

