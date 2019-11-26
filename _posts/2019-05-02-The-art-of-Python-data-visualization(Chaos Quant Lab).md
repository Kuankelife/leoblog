---
layout: post
title: "Python数据可视化修炼之道-Chaos Quant Lab"
author: "雨栋"
header-img: "img/post-bg-css.jpg"
header-mask: 0.4
catalog: true
tags:
  -   Python
  - 数据科学
  - 量化投资
---


<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1537012717707&di=93b200d1c628327cd3686e48bccae82e&imgtype=0&src=http%3A%2F%2Fimg.mp.sohu.com%2Fupload%2F20170625%2F1f1509fb3265468b895c0515c2c9de3b.png" width="400" height="400" />

---
<center>本文只用于平时教学学习，未经得作者同意禁止直接转载或编辑后转载。</center>

---
{:toc}
# <center>Python数据可视化教程（更新至2018-10-01）



# 入门  (传剑）

## Matplotlib数据可视化

## 简介

- 在数据科学中，数据可视化是研究的其中一部份工作，它是为了将处理后的数据呈现给需要的人，因为有时候有规律的结果需要以图表的形式呈
现，即使如此，人们从呈现出来的图表获取的信息量还远远不够，这就要求画图者需要使用多样化的图表来呈现结果。


- 数据可视化是一项技能，看似简单，但是也面临着巨大的挑战，由此技术人员开发了很多可视化工具，可以说，如果一门编程语言没有可视化的功能，那它在数据科学上的推广必然很吃力，而作为Data science的编程新星**`Python`**（在2018年9月26日发布了3.71版本），必然少不了可视化工具**Matplotlib**, Matplotlib是Python中最常用的可视化工具之一，可以非常方便地创建海量类型的2D图表和一些基本的3D图表, 它就是我接下来要详细介绍的内容。


> Matplotlib库由各种可视化类构成，内部结构复杂, Matplotlib.pyplot是绘制各类可视化图形的命令子库，其实风格和Matlab的绘图风格差不多，没错，开发者就是模仿Matlab的绘图风格，所以学过Matlab绘图的同学，对Matplotlib的学习就不需要心怀恐惧，没学过Matlab的同学也不必担心，just follow me step by step, 你会少走很多弯路。

> 其实官方和网上都有关于Matplotlib的教程，但都不全面，也不够深入，有的教程也是看着看着就睡着了，我其实也是想写一篇无比详细的教程，可是写到四分之一我就放弃了，数据可视化是为呈现结果，注重的是实用性，让看图者最短时间去从图表获取最多的信息，故选择什么类型的图表就显得极为重要，同时要美观，下一步才是各种图的学习。故我把入门教程分为三个部分，**模仿篇**，**美化篇**，**拓展篇**，这是符合大多数人的学习路径。


```python
#首先导入画图的库，使用方法为
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
```

## 模仿篇

为了学得更愉快，我不会一下子塞一大堆东西给你，除非你是**AI**.

### 原始图
首先我们用五行代码就对上证指数做了简单的可视化


```python
data = get_price("000001.XSHG", start_date='2018-06-01', end_date='2018-09-29', frequency='daily').close
y = data.values
x = range(len(y))
plt.plot(x,y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_11_0.png'/>


### x轴时间
看图者只看到一条曲线，不知到x,y轴代表什么意思，而我们知道x轴应该是时间才对，那么加上去


```python
#把x轴改为时间列表
x = data.index.values
plt.plot(x,y)
plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_13_0.png'/>


### x, y轴label
即使看图者通过日期知道了x轴代表时间，但是y轴呢，y轴代表的指数没有单位，所以加上label.


```python
plt.xlabel('时间')
plt.ylabel('指数')
plt.plot(x,y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_15_0.png'/>


### 标题
画图总得有个标题吧


```python
plt.title('指数走势')
plt.xlabel('时间')
plt.ylabel('指数')
plt.plot(x,y)
plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_17_0.png'/>


### 多条曲线
很多情况下，我们都需要在同一张图里画多条曲线


```python
data_2 = get_price("000016.XSHG", start_date='2018-06-01', end_date='2018-09-28', frequency='daily').close
y2 = data_2.values
plt.xlabel('时间')
plt.ylabel('指数')
plt.title('上证指数和上证50指数')
plt.plot(x,y2)
plt.plot(x,y)
plt.show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_19_0.png'/>
### 图例
还有，怎么知道两条曲线代表哪个指数？所以需要加上图例


```python
#创建一个图例，不要问为什么，其实相当于实例化，没学过类的话可以不管，看着看着就会懂了。。。
fig,ax = plt.subplots()
ax.set_title(u'上证指数和上证50指数')
ax.set_xlabel('时间')
ax.set_ylabel('指数')
ax.plot(x,y2,label = '上证50指数')
ax.plot(x,y,label = '上证指数')
leg = ax.legend()
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_21_0.png'/>


这样，一个成型的图画出来了，似乎看起来有点丑。。。。

## 美化篇

>作为强迫症的重度患者，自然对上面的图坐视不理了!!!

### 图的大小
首先图太小了，能不能放大一点？能！


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_title(u'上证指数和上证50指数')
ax.set_xlabel('时间')
ax.set_ylabel('指数')
ax.plot(x,y2,label = '上证50指数')
ax.plot(x,y,label = '上证指数')
leg = ax.legend()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_26_0.png'/>

整个人舒畅了很多了。。。

### 图例字体样式及大小
字体也不满意


```python
#设置图例并且设置图例的字体及大小
font1 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 20,
}

fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2)
ax.plot(x,y)
leg = ax.legend(['上证50指数','上证指数'])
```

    /opt/conda/envs/python2new/lib/python2.7/site-packages/matplotlib/font_manager.py:1297: UserWarning: findfont: Font family [u'Times New Roman'] not found. Falling back to DejaVu Sans
      (prop.get_family(), self.defaultFamily[fontext]))


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_29_1.png'/>

### 曲线的颜色
- 线的颜色不合我的心意，改！

- color用于设置颜色


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r')
ax.plot(x,y,color = 'g')
leg = ax.legend(['上证50指数','上证指数'])
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_31_0.png'/>

### 线条的粗细
- 似乎线条粗一点更好看。。。。
- linewidth用于修改线条颜色


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_33_0.png'/>

耳目一新了！！！

### 图例的位置


#### 调整在图里的位置

- 图例在图内有时会和曲线重合，而图例的位置也可以调整，具体的参数就不列举了，可以自行网上查阅，比如图的图例在右上角，我设置为左上角

- 参数loc用于设置legend的位置 


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'],loc='upper left')
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_37_0.png'/>

#### 图例移动在图外的位置
- 有时候图例太大，内容太多，和曲线有重合，把图例放在图外是不错的选择。

- bbox_to_anchor用于在bbox_transform坐标（默认轴坐标）中为图例指定任意位置


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'],loc=2,bbox_to_anchor=(1.05,1.0),borderaxespad = 0.)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_39_0.png'/>

### 格子
高端一点？加个格子上去，世界杯之后对格子莫名的喜欢


```python
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
ax.grid(True)
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_41_0.png'/>

### 样式
其实matploblib自带很多**样式**

#### grayscale  style


```python
plt.style.use("grayscale")
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
ax.grid(True)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_44_0.png'/>

#### ggplot  style
作为R和Python的兼修者，我把R的ggplot样式搬到Python来了，因为这是我最满意的样式，但有时也会小问题不断。。。。


```python
plt.style.use("ggplot")
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_46_0.png'/>

#### bmh  style


```python
plt.style.use("bmh")
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
ax.grid(True)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_48_0.png'/>


#### dark_background  style


```python
plt.style.use("dark_background")
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
ax.grid(True)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_50_0.png'/>


#### fivethirtyeight  style
此处图例和背景同色，原因是上面调用太多style, 导致出错，每个style单独用是不会出现这种问题。


```python
plt.style.use("fivethirtyeight")
fig,ax = plt.subplots(figsize = (16,6))
ax.set_xlabel('time',font1)
ax.set_ylabel('index',font1)
ax.plot(x,y2,color = 'r',linewidth = 4)
ax.plot(x,y,color = 'g',linewidth = 4)
leg = ax.legend(['上证50指数','上证指数'])
ax.grid(True)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_52_0.png'/>


>如无其他说明，下面的部分我将使用ggplot样式


```python
plt.style.use("ggplot")
```

## 拓展篇

### 多个子图(网格）
- 可选参数用numRows, numCols, plotNum

---
在Matplotlib下, 一个 Figure 对象可以包含多个子图(Axes), 可以使用 subplot() 快速绘制, 其调用形式如下

>subplot(numRows, numCols, plotNum)

- 图表的整个绘图区域被分成 numRows 行和 numCols 列


- 然后按照从左到右，从上到下的顺序对每个子区域进行编号，左上的子区域的编号为1


- plotNum 参数指定创建的 Axes 对象所在的区域


---

### 规则划分

如图：numRows ＝ 3, numCols ＝ 3, 那整个绘制图表样式为 3X3 的图片区域


```python
plt.subplots(3,3,figsize = (11,11))
plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_60_0.png'/>


### 不规则划分

1. 将整个表按照 3*2 划分，本应有六张图 。


2. 前两个简单, 分别占的位置为(3, 2, 1) 和 (3, 2, 2)，而第三个图占了(3,2,3) 和 (3,2,4)，第四个图占了(3,2,5) 和 (3,2,6)。


3. 故对其重新划分, 按照 3 * 1 划分，前两个图占用了 (3, 1, 1) 的位置，第三个图占用了 (3, 1, 2) 的位置，第四个图占用了 (3, 1, 2) 的位置。



```python
plt.figure(figsize = (8,8))

plt.subplot(321)

plt.subplot(322)

plt.subplot(312)

plt.subplot(313)

plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_63_0.png'/>

所以说完全可以怎么画就怎么画

### 常见图画法

#### **说明**
- 限于篇幅，我就不一一列举，我将通过上面介绍过的多子图方法，把**`散点图`**，**`折线图`**，**`柱状图`**，**`扇形图`**，**`箱形图`**全画出来。
         
         
- 更多画法，详细请见[官方文档](https://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.plot)



```python
plt.figure(figsize = (18,18))

#散点图
plt.subplot(321)
plt.scatter(x,y)

#折线图
plt.subplot(322)
plt.plot(x,y)

#垂直柱状图
x_bar = [1,2,3,4,5]
y_bar = [-10,34,5,25,-9]
plt.subplot(323)
plt.bar(x_bar,y_bar,color = 'r')

#水平柱状图
plt.subplot(324)
plt.barh(x_bar,y_bar,color = 'y')

#箱线图
plt.subplot(325)
plt.boxplot(y)

#扇形图
sizes = [20, 30, 45, 5] #权重
explode = (0, 0.1, 0, 0) #各部分间隔
plt.subplot(326)
plt.pie(sizes, explode=explode, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.axis('equal') 
```




    (-1.1991859391653492,
     1.1308366712784419,
     -1.1617174718372747,
     1.1029389285149376)


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_67_1.png'/>


### 三维图
- 三维工具mplot3d


```python
from mpl_toolkits.mplot3d import Axes3D #引进三维画法用到的库
fig = plt.figure(figsize=(10,7))
#添加参数projection
ax = fig.add_subplot(111,projection='3d')
z = np.linspace(0,5,200)
r = 1
x_3d = r * np.sin(np.pi*2*z)
y_3d = r * np.cos(np.pi*2*z)
ax.plot(x_3d, y_3d, z,color = 'grey')

plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_69_0.png'/>


# 进阶（修炼）

## Seaborn数据可视化
虽然Matplotlib已经证明了其在数据科学数据可视化的地位，但是不得不承认，其还是有一些不支持的功能，**Seaborn**属于Matplotlib的一个高级接口,故需要跟seaborn结合起来,。


本篇参考借鉴
- [Seaborn API](http://seaborn.pydata.org/api.html)
- [Python数据可视化—seaborn简介和实例](https://blog.csdn.net/qq_34264472/article/details/53814653)

### set_style( ), set( )

- set_style( )是用来设置主题的，Seaborn有五个预设好的主题： darkgrid , whitegrid , dark , white ,和 ticks  
- 默认： darkgrid


```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
```


```python
plt.plot(np.arange(20))
plt.show()
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/75.png'/>


set( )通过设置参数可以用来设置背景，调色板等，更加常用。


```python
sns.set(style="white", palette="muted", color_codes=True)     #set( )设置主题，palette调色板更常用  
plt.plot(np.arange(20))  
plt.show()  
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_75_0.png'/>

### distplot( ), kdeplot( )
- distplot( )为hist加强版，kdeplot( )为密度曲线图 


```python
#导入pandas库
import pandas as pd
df_iris = pd.read_csv('iris.csv')
fig,axes = plt.subplots(1,2,figsize= (13,5))

#kde是密度曲线，rug是边际毛毯
sns.distplot(df_iris['PetalLength'],ax = axes[0], kde = True, rug = True) 

#shade是阴影
sns.kdeplot(df_iris['PetalLength'], ax = axes[1], shade = True)

plt.show()
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_79_0.png'/>


```python
sns.set(palette = 'muted',color_codes = True)
rs = np.random.RandomState(10)
d = rs.normal(size = 100)
f,axes = plt.subplots(2,2,figsize = (15,10), sharex = True)
sns.distplot(d,kde = False, color = 'r', ax = axes[0,0])
sns.distplot(d,hist = False, rug = True, color = 'g', ax = axes[1,0])
sns.distplot(d,hist = False, color = 'y', kde_kws = {'shade':True},ax = axes[0,1])
sns.distplot(d,color = 'grey',ax = axes[1,1])
plt.show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/80_0.png'/>

### 箱型图 boxplot( )


```python
sns.boxplot(x = df_iris['Name'], y = df_iris['SepalWidth'])
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_82_1.png'/>

### 线性回归的边缘分布


```python
import seaborn as sns
sns.set(style="darkgrid")
tips = sns.load_dataset("tips")
g = sns.jointplot("total_bill", "tip", data=tips, kind="reg",
                  xlim=(0, 60), ylim=(0, 12), color="m")
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_84_0.png'/>

 ### 散射矩阵


```python
sns.set(style="ticks")
sns.pairplot(df_iris, hue="Name")
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_86_1.png'/>


### 热力图


```python

sns.set()
#加载数据集
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")
f, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)
```



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_88_1.png'/>

### 热力图数据中的结构


```python
sns.set()
df = sns.load_dataset("brain_networks", header=[0, 1, 2], index_col=0)

#选择网络的一个子集
used_networks = [1, 5, 6, 7, 8, 12, 13, 17]
used_columns = (df.columns.get_level_values("network")
                          .astype(int)
                          .isin(used_networks))
df = df.loc[:, used_columns]

# 创建一个分类格来识别网络
network_pal = sns.husl_palette(8, s=.45)
network_lut = dict(zip(map(str, used_networks), network_pal))

# 将调色板转换成将在矩阵的一侧绘制的向量
networks = df.columns.get_level_values("network")
network_colors = pd.Series(networks, index=df.columns).map(network_lut)

#画图
sns.clustermap(df.corr(), center=0, cmap="vlag",
               row_colors=network_colors, col_colors=network_colors,
               linewidths=.75, figsize=(13, 13))
```




    <seaborn.matrix.ClusterGrid at 0x7f3908564f10>

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/90_1.png'/>


## Dataframe数据可视化
- DataFrame.plot( )函数

>DataFrame.plot(x=None, y=None, kind='line', ax=None, subplots=False, sharex=None, sharey=False, layout=None,figsize=None, use_index=True, title=None, grid=None, legend=True, style=None, logx=False, logy=False,loglog=False, xticks=None, yticks=None, xlim=None, ylim=None, rot=None, fontsize=None, colormap=None,table=False, yerr=None, xerr=None, secondary_y=False, sort_columns=False, **kwds)

> 一行代码就搞定了画图，现在精通可视化的大神都会这样画图.


```python
df = get_price("000001.XSHG", start_date='2018-06-01', end_date='2018-09-29', frequency='daily').iloc[:,:4]
df.plot()
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_94_1.png'/>



```python
df2 = pd.DataFrame(np.arange(25).reshape((5,5)),index=['a','b','c','d','e'],columns=['one','two','three','four','five'])
```

### 垂直柱状图


```python
df2.plot(kind = 'bar')
```



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_97_1.png'/>


### 散点图


```python
df.plot('close','open',kind = 'scatter')
```

    <matplotlib.axes._subplots.AxesSubplot at 0x7f39082f9b10>

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_99_1.png'/>

### 水平柱状图


```python
df2.plot(kind = 'barh')
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_101_1.png'/>
### 横向条形图


```python
df2.plot.barh(stacked = True)
```


<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_103_1.png'/>


### 垂直条形图


```python
df2.plot.bar(stacked = True)
```

<img src='http://i.caigoubao.cc/627139/bgpc/uyer/105_1.png'/>


### 直方图


```python
df3 = pd.DataFrame({'a': np.random.randn(100) , 'b': np.random.randn(100),
                 'c': np.random.randn(100), 'd':np.random.randn(100),'e':np.random.randn(100)},columns=['a', 'b', 'c','d','e'])
df3.plot.hist(alpha = 0.6)
```
<img src='http://i.caigoubao.cc/627139/bgpc/uyer/107_1.png'/>
### 子图

设置 subplot = True,同时画多个子图


```python
df.plot(subplots = True, figsize = (8,8))
```




    array([<matplotlib.axes._subplots.AxesSubplot object at 0x7f390879c810>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x7f3907ad5310>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x7f3907a58650>,
           <matplotlib.axes._subplots.AxesSubplot object at 0x7f3907a3cb10>], dtype=object)



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_109_1.png'/>


```python
fig = plt.figure()
ax_1 = fig.add_subplot(2,2,1)
ax_2 = fig.add_subplot(2,1,2)
ax_1.plot(df3)
ax_2.plot(df)
```




    [<matplotlib.lines.Line2D at 0x7f3907754d50>,
     <matplotlib.lines.Line2D at 0x7f39077b5b90>,
     <matplotlib.lines.Line2D at 0x7f39077b5350>,
     <matplotlib.lines.Line2D at 0x7f39077a9cd0>]



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_110_1.png'/>



```python
df3.diff().hist(color='y', alpha=0.5, bins=50)
```




    array([[<matplotlib.axes._subplots.AxesSubplot object at 0x7f39076e29d0>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f390768a3d0>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f390760d690>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f390756df10>],
           [<matplotlib.axes._subplots.AxesSubplot object at 0x7f390750a250>,
            <matplotlib.axes._subplots.AxesSubplot object at 0x7f390746ca10>]], dtype=object)



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_111_1.png'/>


### 箱型图


```python
df.boxplot()
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f39096d64d0>



<img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_113_1.png'/>


# 高手（忘剑）

## 可视化实战实例



```python
import matplotlib.pyplot as plt  
import numpy as np
import pandas as pd
from matplotlib import colors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker as mtk
import matplotlib.dates as mdate
from pandas import Series
import matplotlib.finance as mpf
plt.style.use("ggplot")
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体


def draw_distribution(indexcode,indextime,stockstart,stockend,factortime):

    stockname = get_index_stocks(indexcode,date=indextime)

    stockprice=[]
    return_rate=[]
    stock_cnname=[]

    for i in range(0,len(stockname),1):
        stockprice.append(get_price(stockname[i],start_date=stockstart,end_date=stockend
                                ,frequency='daily',fields=['open','close'],skip_paused=False, fq='pre'))
        return_rate.append((stockprice[i].close[-1]-stockprice[i].open[0])/stockprice[i].open[0])

    #获取多只股票在某一日期的PE和市值cap
    df1 = get_fundamentals(query(valuation.code,valuation.pe_ratio,valuation.market_cap).filter(valuation.code.in_(stockname)), date=factortime)
    df1.index=df1['code']
    del df1['code']
    #获取ROE
    df2 = get_fundamentals(query(indicator.code,indicator.roe).filter(indicator.code.in_(stockname)), date=factortime)
    df2.index=df2['code']
    del df2['code']
    #获取股票名字
    for i in range(0,len(stockname),1):
        stock_cnname.append(get_security_info(stockname[i]).display_name)
    #print stock_cnname

    result=df1.join(df2)
    result['return_rate']=return_rate
    result['cnname']=stock_cnname
    indexcnname=str(get_security_info(indexcode).display_name)
  
    #收益率分布
    plt.figure(figsize=(18,7))
    plt.title('%s至%s%s指数收益率分布'%(stockstart,stockend,indexcnname),fontsize=20) 
    sns.distplot(result.dropna(axis=0,how='any').return_rate,kde=False,bins=20)
    plt.xlabel(u'收益率',fontsize=18)
    plt.ylabel(u'频数',fontsize=18)
    
    #前五后五  
    result_aftsort=result.sort_values(by=['return_rate'],ascending=False).dropna(axis=0,how='any')
    blist=result_aftsort.head(5)
    wlist=result_aftsort.tail(5)
    best5_index=blist.index.values
    worst5_index=wlist.index.values   
    #print blist
    #print wlist
    
    x=[best5_index,worst5_index]
    y=[blist.cnname,wlist.cnname] 
    title=[u'最好',u'最差']
    
    for j in (0,1) :  
        plt.figure(figsize=(18,6))
        plt.title('%s至%s%s指数%s5支走势对比'%(stockstart,stockend,indexcnname,title[j]),fontsize=20)    
        plt.xlabel(u'时间',fontsize=18)
        closeprice=[]
        tradedate=[]
        closerate=[] 
        for i in range(0,len(x[j]),1):
            closeprice.append(get_price(x[j][i], start_date=stockstart, end_date=stockend, frequency='daily', fields=['close'],
                                skip_paused=False, fq='pre').close.values)
            closerate.append(closeprice[i]/closeprice[i][0])
            tradedate.append(get_price(x[j][i], start_date=stockstart, end_date=stockend, frequency='daily', fields=['close'],
                                skip_paused=False, fq='pre').index.values)
            plt.plot(tradedate[i],closerate[i],linewidth=2,label=str(y[j][i]))
            plt.legend(loc='best',fontsize=11)
            plt.grid()

    
    #市值市盈率散点图
    #过滤极端值（要先整合到dataframe里面！！）还要增加一列return rate
    result=result[result.pe_ratio >=-500].dropna()
    pe=result.pe_ratio.values
    cap=result.market_cap.values
    roe=result.roe.values
    return_rate=result.return_rate
    
    #
    #cm = plt.cm.get_cmap('RdYlGn_r')  
    cm2= colors.ListedColormap(['#32CD32','#F08080','#FF4500','#8B0000'],N=4)
    plt.figure(figsize=(18,7))
    plt.title(u'市值、市盈率、收益率',fontsize=20)
    plt.xlabel(u'市盈率',fontsize=18)
    plt.ylabel(u'市值',fontsize=18)
    plt.scatter(pe,cap,s=70,c=return_rate,alpha=0.8,cmap=cm2)

    #text
    #str(result.cnname[i])
    for a,b,c in zip(pe,cap,result.cnname):
            plt.text(a,b,'%s'%str(c),ha='center',va='top',fontsize=9)   
    plt.colorbar() 
    plt.grid()
       
    
    #3D+ROE
    fig = plt.figure(figsize=(18,10))
    ax = fig.add_subplot(111, projection='3d')  
    plt.title(u'市值、市盈率、净资产收益率',fontsize=20)
    ax.set_xlabel(u'市盈率',fontsize=18) 
    ax.set_ylabel(u'市值',fontsize=18) 
    ax.set_zlabel(u'净资产收益率',fontsize=18)   
    ax.scatter(pe,cap,roe,c=return_rate,s=70,alpha=1,cmap=cm2)
    plt.show()
    
def addvolume(x):
    n = 0
    volumedata = []
    for i in x:
        n = n + i
        volumedata.append(n)
    return volumedata
def x_fmt_func(x,pos=None):
    idx = np.clip(int(x+0.5), 0, df.shape[0]-1)
    return df['datetime'].iat[idx]
def decorateAx(ax, xs, ys, col, x_func):
    ax.plot(xs, ys, color=col, linewidth=1, linestyle="-")
    #plt.title("")
    #plt.xlabel("time")
    plt.ylabel("index")
    if x_func:
        ax.xaxis.set_major_formatter(mtk.FuncFormatter(x_func))
    ax.grid(True)
    return 

def draw_distribution_2(start,end,pre,df):
   
    datetime=list(df.index)
    df['datetime']=datetime
    close1 = df.close.values
    high=df.high.values
    low = df.low.values
    open1 = df.open.values
    datetime=list(df.index)
    df['datetime']=datetime
    volume = df.volume
    volume2= Series(addvolume(volume))
    idx_pxy = np.arange(df.shape[0])
    volume = df.volume
    dates = df.index
    left,width = 0.1,0.8
    rect_vol = [left,0.1,width,0.3]
    rect_main = [left,0.4,width,0.5]
    fig = plt.figure(figsize = (20,9))
    ax_vol = fig.add_axes(rect_vol)
    ax_vol.fill_between(range(240),volume,color = 'y')
    decorateAx(ax_vol, idx_pxy, df['close'],'green', x_fmt_func)
    ax_main  = fig.add_axes(rect_main)
    plt.title(u'上证指数2018年9月28日走势',fontsize=22)
    ax_main.plot(volume2,close1)
    
#调用函数1   
indexcode='000300.XSHG'#指数代码    SZ50-000016;HS300-000300;ZZ800-000906;ZZ500-000905
indextime='2017-01-01'#指数成分取值时间
stockstart='2017-01-01'#股票行情开始时间
stockend='2017-12-31'#股票行情结束时间
factortime='2017-01-01'#财务数据取值时间
draw_distribution(indexcode,indextime,stockstart,stockend,factortime)

#调用函数2   
start ='2018-09-28 9:30:00'
end = '2018-09-28 15:00:00'
pre = '2018-09-27'   
df = get_price('000001.XSHG',start_date=start, end_date=end, frequency='1m')
draw_distribution_2(start,end,pre,df)
```

<div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_0.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_1.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_2.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_3.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_4.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/uyer/output_116_5.png'/></div>

# 结语
- 我希望通过这篇教程，能够在数据可视化上帮你少走弯路，少掉坑，我的初衷是想像武侠世界一样，传授剑法给你，然后勤加修炼，当你熟练到可以抛开条条框框，从而达到融会贯通的程度时，你就差不多成为高手了，此时无剑胜有剑，岂不妙哉！加油吧！
- 如果你在学习上有什么问题，欢迎通过<strong>1115223619@qq.com</strong>或者[这里](https://ownyulife.top/)跟我交流。


---
<center>本文只用于平时教学学习，未经得作者同意禁止直接转载或编辑后转载。</center>

---
