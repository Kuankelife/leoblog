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


![png](https://thumbsnap.com/s/yxA4QyWz.png?0828)


## 核PCA(线性核)


```python
from sklearn.decomposition import KernelPCA
lin_pca = KernelPCA(n_components = 2, kernel="linear").fit_transform(X)
plot_digits(lin_pca, y)
```


![png](https://thumbsnap.com/s/okdyAZ7U.png?0828)


##  核PCA(高斯核)


```python
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433).fit_transform(X)
plot_digits(rbf_pca, y)
```


![png](https://thumbsnap.com/s/flJDdRaz.png?0828)


## LLE


```python
from sklearn.manifold import LocallyLinearEmbedding
X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
plot_digits(X_lle_reduced, y)
```


![png](https://thumbsnap.com/s/X6UPJSFv.png?0828)




## LDA


```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X,y)
plot_digits(X_lda_reduced, y)
```

    C:\Users\dell\Anaconda3\lib\site-packages\sklearn\discriminant_analysis.py:388: UserWarning: Variables are collinear.
      warnings.warn("Variables are collinear.")



![png](output_26_1.png)


## MDS


```python
from sklearn.manifold import MDS
X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X)
plot_digits(X_mds_reduced, y)
```


![png](https://thumbsnap.com/s/yRacSL4d.png?0828)


## Isomap


```python
from sklearn.manifold import Isomap
X_reduced_isomap = Isomap(n_components=2).fit_transform(X)
plot_digits(X_reduced_isomap, y)
```


![png](https://thumbsnap.com/s/yRacSL4d.png?0828)


## ICA


```python
from sklearn.decomposition import FastICA 
X_reduced_ICA = FastICA(n_components=2).fit_transform(X)
plot_digits(X_reduced_ICA, y)
```


![png](https://thumbsnap.com/s/V7BSgiby.png?0828)


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


![png](https://thumbsnap.com/i/Srq6CInG.png?0828)


## UMAP


```python
import umap.umap_ as umap
umap_data = umap.UMAP(n_neighbors=5, min_dist=0.3, n_components=2).fit_transform(X)
plot_digits(umap_data, y)
```


![png](https://thumbsnap.com/s/GJfTWXfy.png?0828)
