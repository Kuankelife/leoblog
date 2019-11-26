---
layout:     post
title:  "时间序列拟合实践"
subtitle:   ""
date:       2018-09-26 00:55:00
author:     "YU"
header-img: "img/post-bg-js-module.jpg"
catalog:    true
tags:
    - 时间序列
    - 数据科学
    - Python
    - 数据分析
mathjax: true
music-id-head: 117227
---
这是我最喜欢的一首钢琴曲，希望看到这篇文章的你也能喜欢。


本文是为2019年秋季时间序列与随机过程课程的学生整理的教程，其中整理了2018年秋季课程同学的很多代码和文字，内容不全为本人所写。

# <center>时间序列及其拟合</center>

# 什么是时间序列

**数据拟合的方法很多,每种方法各有特点，有自定义函数，可以调取相应的库，比如numpy, scipy，statsmodels等，也可以结合两种方法。**

简而言之：<br>
对某一个或者一组变量 x(t) 进行观察测量，将在一系列时刻 $t_1,t_2,⋯,t_n$ 所得到的离散数字组成的序列集合，称之为时间序列。 

例如: 某股票A从2015年6月1日到2018年6月1日之间各个交易日的收盘价，可以构成一个时间序列；某地每天的最高气温可以构成一个时间序列。

一些特征:<br>
- **趋势**：是时间序列在长时期内呈现出来的持续向上或持续向下的变动。

- **季节变动**：是时间序列在一年内重复出现的周期性波动。它是诸如气候条件、生产条件、节假日或人们的风俗习惯等各种因素影响的结果。

- **循环波动**：是时间序列呈现出得非固定长度的周期性变动。循环波动的周期可能会持续一段时间，但与趋势不同，它不是朝着单一方向的持续变动，而是涨落相同的交替波动。

- **不规则波动**：是时间序列中除去趋势、季节变动和周期波动之后的随机波动。不规则波动通常总是夹杂在时间序列中，致使时间序列产生一种波浪形或震荡式的变动。只含有随机波动的序列也称为**平稳序列**。

在开始之前，你需要知道拟合要达到大概什么样的效果，并可能得去拟合，并且在后续中体现泛化的能力。


# **数据的拟合**
时间序列通常具有一定的趋势，其反映了序列总体的一个走势，而往往这个也是最宏观和最重要的。下面，我们首先解决趋势项，一个常用的方法就是拟合。

# **线性拟合**

## **简单线性拟合** 
单变量线性回归，又称简单线性回归（simple linear regression, SLR），是最简单但用途很广的回归模型。其回归式为：$$Y=\alpha+\beta X+\epsilon$$
为了从一组样本$(y_{i},x_{i})$（其中$i=1,2,...,n$）之中估计最合适（误差最小）的$\alpha$和$\beta$，通常采用最小二乘法(OLS)，其中计算目标最小化残差平方和：
$$
\sum_{i=1}^{n}\epsilon_{i}^{2} = \sum_{i=1}^{n}\left(y_{i}-\alpha-\beta x_{i}\right)^2
$$
从中可以求解得$\hat{\alpha},\hat{\beta}$，并得到拟合方程：$$\hat{Y}=\hat{\alpha}+\hat{\beta} X$$
其中$\hat{\alpha},\hat{\beta}$如下：


$$
\hat{\beta}=\frac{\sum_{i=1}^{n}(x_{i}-\bar{x})(y_{i}-\bar{y})}{\sum_{i=1}^{n}(x_{i}-\bar{x})^2}
$$

$$
\hat{\alpha} = \bar{y}-\bar{x}\hat{\beta}
$$

评价模型拟合效果通常看回归系数$R^{2}$:
$$
R^{2}= \frac{\sum(\hat{Y_{i}}-\bar{Y_{i}})^2}{\sum(Y_{i}-\bar{Y_{i}})^2}.
$$
其取值范围为[0,1]，反映了拟合函数能够解释样本的程度，越接近1，解释程度越高。


### 方法一：自定义函数拟合


```python
#范例：下面用上述普通求解方法编写一个线性拟合实现的公式，并画图
#导入库
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
#定义linefit函数
def linefit(x , y):   
    N = float(len(x))   #获取求和的上限N
    sx,sy,sxx,syy,sxy=0,0,0,0,0  #定义求和初始值
    for i in range(0,int(N)):   #循环求和  
        sx  += x[i] 
        sy  += y[i]
        sxx += x[i]*x[i]
        syy += y[i]*y[i]
        sxy += x[i]*y[i]
    b = (sy*sx/N -sxy)/( sx*sx/N -sxx)  #计算系数b的值
    a = (sy - b*sx)/N                   #计算系数a的值
    r = abs(sy*sx/N-sxy)/math.sqrt((sxx-sx*sx/N)*(syy-sy*sy/N)) 
    return a,b,r   #返回三个值
 

X=[1,2,3,4,5,6,7,8,9]                #数据x
Y=[ 2.5 ,4 ,3.5 ,6.6 ,6.47 ,7.51,8.9,6,7]     #数据y
Xsample=X[:-3]                    #样本内
Ysample=Y[:-3]                    
Xoutsample=X[-3:]                 #样本外，不计入拟合中
Youtsample=Y[-3:]  
a,b,r=linefit(Xsample,Ysample)                #调用linefit函数线性拟合
print "X=",Xsample
print "Y=",Ysample
print "拟合结果: y = %10.5f  + %10.5f x , r=%10.5f" % (a,b,r)

#绘制拟合函数曲线
x_fit = np.linspace(min(Xsample),max(Xoutsample),20)
y_fit = a + b*np.array(x_fit)
plt.scatter(Xsample,Ysample,alpha=0.6,color='b') #scatter为默认离散点的绘图程序
plt.scatter(Xoutsample,Youtsample,alpha=0.6,color='g')
plt.plot(x_fit,y_fit,alpha=0.5,color='r')
```

> X= [1, 2, 3, 4, 5, 6]
> Y= [2.5, 4, 3.5, 6.6, 6.47, 7.51]
> 拟合结果: y =    1.54067  +    1.01600 x , r=   0.93952


![png](/img/yu-img/post-img/20180926/output_10_2.png)


### 方法二：调用numpy库拟合


```python
# 线性拟合-使用numpy polyfit 函数
X=[1,2,3,4,5,6,7,8,9]            #数据x
Y=[ 2.5 ,4 ,3.5 ,6.6 ,6.47 ,7.51,8.9,6,7]     #数据y
Xsample=X[:-3]                    #样本内
Ysample=Y[:-3]                    
Xoutsample=X[-3:]                 #样本外
Youtsample=Y[-3:]  
z1 = np.polyfit(Xsample, Ysample, 1)  #一次多项式拟合，相当于线性拟合，并获取系数值
p1 = np.poly1d(z1)     #获取拟合的函数

print z1   
print p1
#绘制拟合函数曲线
x_fit = np.linspace(min(Xsample),max(Xoutsample),20)
y_fit = p1(x_fit)
plt.scatter(Xsample,Ysample,alpha=0.6,color='b')
plt.scatter(Xoutsample,Youtsample,alpha=0.6,color='g')
plt.plot(x_fit,y_fit,alpha=0.5,color='r')
```

> [ 1.016       1.54066667]
>
> 1.016 x + 1.541


![png](/img/yu-img/post-img/20180926/output_12_2.png)


效果是一样的，为什么不用现成的工具呢？

## **多项式拟合** 
多项式回归原理与线性回归基本类似。如果我们有自变量$X$,$X^2$,$X^3$,...,$X^n$，和一个因变量$Y$，通过线性回归，我们能够确定最好解释数据的线性模型：$Y = \alpha + \beta_1 X + \beta_3 X^2 +...+  \beta_n X^n$ 。
同样，对于多项式拟合，我们还是用numpy库的polyfit.


```python
#二次多项式拟合--使用numpy polyfit 函数
X=[1,2,3,4,5,6,7,8,9]                #数据x
Y=[ 2.5 ,4 ,3.5 ,6.6 ,6.47 ,7.51,8.9,6,7]     #数据y
Xsample=X[:-3]                    #样本内
Ysample=Y[:-3]                    
Xoutsample=X[-3:]                 #样本外
Youtsample=Y[-3:]  
z1 = np.polyfit(Xsample, Ysample, 3)                 #用np.polyfit（x，y,多项式阶数）获取拟合函数的系数
p1 = np.poly1d(z1)    #得到拟合的函数
print '得到这个函数：',p1
print '\n'
 # 下面计算R方
Yhat = p1(Xsample)                         
Ybar = np.sum(Ysample)/len(Ysample)         
ssreg = np.sum((Yhat-Ybar)**2)   
sstot = np.sum((Ysample - Ybar)**2)    
r = ssreg / sstot # R^2

print z1
print r

#绘制拟合函数曲线
x_fit = np.linspace(min(Xsample),max(Xoutsample),20)
y_fit = p1(x_fit)
plt.scatter(Xsample,Ysample,alpha=0.6,color='b')
plt.scatter(Xoutsample,Youtsample,alpha=0.6,color='g')
plt.plot(x_fit,y_fit,alpha=0.5,color='r')
```

> 得到这个函数：           3          2
> -0.04296 x + 0.4365 x - 0.2434 x + 2.487
> [-0.04296296  0.43646825 -0.24342593  2.48666667]
> 0.888939795429


![png](/img/yu-img/post-img/20180926/output_15_2.png)


## **曲线拟合** 
对于曲线拟合，我们开始使用scipy库optimize的curve_fit函数，其威力强大。

### **三角函数** 
- $asin(bx+c)+d$


```python
#根据某地每月的平均温度[17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18]拟合温度函数。
#构建函数y=a*sin(x*pi/6+b)+c
#使用optimize.curve_fit函数求出a、b、c的值  
def fmax(x,a,b,c):
    return a*np.sin(x*np.pi/6+b)+c

x=np.arange(1,13,1)
x1=np.arange(1,13,0.1)
ymax=np.array([17, 19, 21, 28, 33, 38, 37, 37, 31, 23, 19, 18 ])
fita,fitb=optimize.curve_fit(fmax,x,ymax,[1,1,1])
print '求出a,b,c',fita
plt.scatter(x,ymax,alpha=0.6,color='b')
plt.plot(x1,fmax(x1,fita[0],fita[1],fita[2]),alpha=0.6,color='r')
plt.show()
```

    求出a,b,c [ 10.93254951  -1.9496096   26.75      ]

![png](/img/yu-img/post-img/20180926/output_18_1.png)


### **对数函数**
- $alog(bx)+c$


```python
#  对数函数拟合  使用scipy.optimize import curve_fit 函数
from scipy import log as log
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

#先定义要拟合的曲线方程
def func(x, a, b):
    y = a * log(x) + b
    return y

X=np.arange(1, 14, 1)      #数据x
Y=[3,33,60,80,105,125,140,150,159,165,168,170,171]
Xsample=X[:-3]                    #样本内
Ysample=Y[:-3]                    
Xoutsample=X[-3:]                 #样本外
Youtsample=Y[-3:]  

z, pcov = curve_fit(func, Xsample, Ysample)  #获取拟合的系数值和协方差

#下面计算r
Yhat = func(Xsample ,z[0] ,z[1] )                         
Ybar = np.sum(Ysample)/len(Ysample)          
ssreg = np.sum((Yhat-Ybar)**2)   
sstot = np.sum((Ysample - Ybar)**2)    
r = ssreg / sstot

print z
print r

#绘制拟合函数曲线
x_fit = np.linspace(min(Xsample),max(Xoutsample),20)
y_fit = func(x_fit ,z[0] ,z[1] )  
plt.scatter(Xsample,Ysample,alpha=0.6,color='b')
plt.scatter(Xoutsample,Youtsample,alpha=0.6,color='g')
plt.plot(x_fit,y_fit,alpha=0.5,color='r')
```

    [ 75.60798846 -12.20142498]
    0.97636651944


![png](/img/yu-img/post-img/20180926/output_20_2.png)


### **指数函数**
- $ae^{bx}$


```python
from scipy.optimize import curve_fit #自定义函数 e指数形式
def func(x, a, b):    
    return a*np.exp(b/x) 
#定义x、y散点坐标
x = np.arange(1, 18, 1)
num = [3.00, 5.80, 5.900, 6.80, 7.34, 8.9, 9.86, 11.11, 12.85, 13,15,18,21,28,37,89,200]

y = np.array(num) #非线性最小二乘法拟合
popt, pcov = curve_fit(func, x, y)

#获取popt里面是拟合系数
a = popt[0]
b = popt[1]
yvals = func(x,a,b) #拟合y值
print 'a:', a
print 'b:', b 
#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'r',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```

    a: 16429066.3873
    b: -192.685937261

![png](/img/yu-img/post-img/20180926/output_22_1.png)


### 函数结合
根据官方给出的实例，讲述传递三个参数，通常为 $ae^{(\frac{b}{x})}+c$形式


```python
def func(x, a, b, c):    
    return a * np.exp(-b * x) + c 

# define the data to be fit with some noise
xdata = np.linspace(0, 4, 50)
y = func(xdata, 2.5, 1.3, 0.5)
y_noise = 0.2 * np.random.normal(size=xdata.size)
ydata = y + y_noise
plt.plot(xdata, ydata, 'b-', label='data') 

# Fit for the parameters a, b, c of the function `func`
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'r-', label='fit') 

# Constrain the optimization to the region of ``0 < a < 3``, ``0 < b < 2``# and ``0 < c < 1``:
popt, pcov = curve_fit(func, xdata, ydata)
plt.plot(xdata, func(xdata, *popt), 'g--', label='fit-with-bounds') 
plt.xlabel('x')
plt.ylabel('y')
plt.show()
```


![png](/img/yu-img/post-img/20180926/output_24_0.png)


以上只是初等函数结合一个例子，很容易想到，其他函数也可以结合，级数也是一个可以考虑的方向，所以你可以有很多方法可以尝试！！！

## 真实数据拟合
- 老是用一些随机数似乎很单调，所以接下来马上用到真实数据！！!



```python
data = get_price("000001.XSHG", start_date='2018-06-01', end_date='2018-09-25', frequency='daily').close
x = data.index.values
y = data.values
plt.figure(figsize=(15, 5))
plt.plot(x,y)
print "以下为上证指数在2018年6月1日到9月25日的走势情况"
```

    以下为上证指数在2018年6月1日到9月25日的走势情况

![png](/img/yu-img/post-img/20180926/output_27_1.png)


### statsmodels方法
再介绍一个库statsmodels


```python
from statsmodels import regression
import statsmodels.api as sm
def sinreg(x,y,t):
    X = np.column_stack((np.sin(x/40*np.pi), np.cos(x/40*np.pi),np.sin(2*x/40*np.pi),np.cos(2*x/40*np.pi),np.sin(3*x/40*np.pi),np.cos(3*x/40*np.pi)))
    
    X = sm.add_constant(X)                    #为模型增加常数项
    model = regression.linear_model.OLS(y, X).fit() #对模型进行最小二乘法的线性模型拟合
    print '%s阶傅里叶级数拟合系数为'%t
    print model.params
    a = model.params[0]                       #将拟合的第一个系数值赋值于a
    b = model.params[1]                       #...
    c = model.params[2]
    d = model.params[3]
    e = model.params[4]
    f = model.params[5]
    g = model.params[6]
    
    Y = y
    x0 = x
    x = np.linspace(min(x),max(x)+t,num=max(x)+t-min(x)+1)
    # 返回回归结果和图
    #X2 = np.linspace(x[0], x_outsample[-1], 100)  #生成拟合曲线的自变量X2，取样本内数据X[0]为下限；样本外数据X_outsample[-1](最后一个)为上限，均匀的取100个点。因此X2的数据长度为100
    
    Y_hat = a + b*np.sin(x/40*np.pi) + c*np.cos(x/40*np.pi) + d*np.sin(2*x/40*np.pi) + e*np.cos(2*x/40*np.pi) + f*np.sin(3*x/40*np.pi) + g*np.cos(3*x/40*np.pi)
    fig = plt.figure(figsize=(15, 5))
    plt.plot(x0, y, 'b')            # 画样本内数据散点图
    plt.plot(x, Y_hat, 'r')   # 添加回归直线，颜色设置为红色
    plt.xlabel('time')                      # 横坐标名称
    plt.ylabel('Cumulative Abnormal Return')                     # 纵坐标名称
    print model.summary()
    return Y_hat                # 函数返回模型的拟合结果


num = len(data)
t = 5
sinreg(np.linspace(0,num-1,num),data,t)
```

    5阶傅里叶级数拟合系数为
    const    2815.452202
    x1         83.341005
    x2         40.751444
    x3         67.021982
    x4         60.605173
    x5         79.884163
    x6         -2.145009
    dtype: float64
                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                  close   R-squared:                       0.789
    Model:                            OLS   Adj. R-squared:                  0.772
    Method:                 Least Squares   F-statistic:                     46.24
    Date:                Wed, 26 Sep 2018   Prob (F-statistic):           4.32e-23
    Time:                        00:23:37   Log-Likelihood:                -440.29
    No. Observations:                  81   AIC:                             894.6
    Df Residuals:                      74   BIC:                             911.3
    Df Model:                           6                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    const       2815.4522      6.457    436.055      0.000    2802.587    2828.317
    x1            83.3410      9.184      9.075      0.000      65.041     101.641
    x2            40.7514      9.078      4.489      0.000      22.663      58.839
    x3            67.0220      9.184      7.298      0.000      48.722      85.322
    x4            60.6052      9.078      6.676      0.000      42.517      78.693
    x5            79.8842      9.184      8.698      0.000      61.585      98.184
    x6            -2.1450      9.078     -0.236      0.814     -20.233      15.943
    ==============================================================================
    Omnibus:                        2.004   Durbin-Watson:                   0.368
    Prob(Omnibus):                  0.367   Jarque-Bera (JB):                1.342
    Skew:                           0.211   Prob(JB):                        0.511
    Kurtosis:                       3.469   Cond. No.                         1.42
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

    array([ 2914.66380996,  2949.52329863,  2981.44464373,  3009.20973522,
            3031.75490339,  3048.22335927,  3058.00788126,  3060.78147358,
            3056.51434549,  3045.4762742 ,  3028.22418008,  3005.57551907,
            2978.56884253,  2948.41354791,  2916.43140974,  2883.9929072 ,
            2852.45163086,  2823.08014124,  2797.01056218,  2775.18292722,
            2758.30387199,  2746.8177038 ,  2740.89121137,  2740.41284012,
            2745.00609104,  2754.05624693,  2766.74882774,  2782.11756671,
            2799.09921256,  2816.59212619,  2833.51547122,  2848.86580414,
            2861.76805096,  2871.51820126,  2877.61553903,  2879.78283303,
            2877.97359572,  2872.36624851,  2863.34576361,  2851.47404639,
            2837.45094082,  2822.06824742,  2806.15951325,  2790.54856473,
            2775.99979624,  2763.17309885,  2752.58602198,  2744.58532296,
            2739.32950127,  2736.7832663 ,  2736.72418842,  2738.76107016,
            2742.3628906 ,  2746.8965575 ,  2751.67118585,  2755.98623566,
            2759.1806091 ,  2760.67974049,  2760.03781593,  2756.97252776,
            2751.39018673,  2743.39955966,  2733.31344115,  2721.63766943,
            2709.04801883,  2696.35610445,  2684.46607853,  2674.32444625,
            2666.86575018,  2662.9571428 ,  2663.34496979,  2668.60641613,
            2679.10902542,  2694.98050283,  2716.09067488,  2742.04683361,
            2772.20297382,  2805.68267926,  2841.41466779,  2878.17930814,
            2914.66380996,  2949.52329863,  2981.44464373,  3009.20973522,
            3031.75490339,  3048.22335927])


![png](/img/yu-img/post-img/20180926/output_29_2.png)


# 结语

- 本章节介绍了时间序列的概念，并用数据拟合的几个方法去拟合不同的时间序列；
- 介绍了Python里可以用来数据拟合的库numpy, scipy, statsmodels,并用代码将其实现作为例子；
- 最后还用了上证指数作为例子，
- 所以讲到这里，如果你掌握了上述的方法，就可以尝试拟合这段时间的上证指数了，拟合之后尝试去评估拟合的效果，如果效果良好，下一步还可以进行预测时间序列。

Just do it！！！
