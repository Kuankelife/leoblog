---
layout:     post
title:  "Mnist数据集各种降维技术可视化（中）"
subtitle:   ""
date:       2019-05-28 12:00:00
author:     "Leo"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
tags:

    - 机器学习
    - 降维
mathjax: false

---
接着上篇的内容
## PCA


```python
from sklearn.decomposition import PCA
X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
plot_digits(X_pca_reduced, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_18_0.png'/>

## 核PCA(线性核)


```python
from sklearn.decomposition import KernelPCA
lin_pca = KernelPCA(n_components = 2, kernel="linear").fit_transform(X)
plot_digits(lin_pca, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_20_0.png'/>
##  核PCA(高斯核)


```python
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433).fit_transform(X)
plot_digits(rbf_pca, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_22_0.png'/>
## LLE


```python
from sklearn.manifold import LocallyLinearEmbedding
X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
plot_digits(X_lle_reduced, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_24_0.png'/>

## LDA


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X,y)
plot_digits(X_lda_reduced, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_26_1.png'/>

## MDS


```python
from sklearn.manifold import MDS
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X)
plot_digits(X_mds_reduced, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_28_0.png'/>

## Isomap


```python
from sklearn.manifold import Isomap
X_reduced_isomap = Isomap(n_components=2).fit_transform(X)
plot_digits(X_reduced_isomap, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_30_0.png'/>

## ICA


```python
from sklearn.decomposition import FastICA 
X_reduced_ICA = FastICA(n_components=2).fit_transform(X)
plot_digits(X_reduced_ICA, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_32_0.png'/>

## TSNE


```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
plt.figure(figsize=(13,10))
plt.scatter(X_reduced_tsne[:, 0], X_reduced_tsne[:, 1], c=y, cmap="jet")
plt.axis('off')
plt.colorbar()
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_34_0.png'/>

## UMAP


```python
import umap.umap_ as umap
umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(X)
plot_digits(umap_data, y)
```
<img src='http://i.caigoubao.cc/627139/bgpc/ky/output_36_0.png'/>