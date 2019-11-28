---
layout:     post
title:  "Mnist数据集各种降维技术可视化（下）"
subtitle:   ""
date:       2019-05-29 12:00:00
author:     "Leo"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
tags:
    - 机器学习
    - 降维
mathjax: false

---
接着中篇的内容

## 组合（PCA + LLE)


```python
from sklearn.pipeline import Pipeline
pca_lle = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("lle", LocallyLinearEmbedding(n_components=2, random_state=42)),
])

X_pca_lle_reduced = pca_lle.fit_transform(X)
plot_digits(X_pca_lle_reduced, y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_38_0.png'/>

## 组合（PCA + MDS)


```python
pca_mds = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("mds", MDS(n_components=2, random_state=42)),
])
X_pca_mds_reduced = pca_mds.fit_transform(X)
plot_digits(X_pca_mds_reduced, y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_28_0.png'/>

## 组合（PCA + TSNE)


```python
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42)),
])
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
plot_digits(X_pca_tsne_reduced, y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_42_0.png'/>
如果你在学习上有什么问题，欢迎通过<strong>1115223619@qq.com</strong>或者[这里](https://ownyulife.top/)跟我交流。
