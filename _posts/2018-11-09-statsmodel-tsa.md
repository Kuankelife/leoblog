---
layout: post
title: "关于利用Statsmodels库分析预测时间序列"
author: "雨栋"
header-img: "img/post-bg-css.jpg"
header-mask: 0.4
tags:
  - Python
  - 时间序列
  - 数据分析
---



<img src="https://timgsa.baidu.com/timg?image&quality=80&size=b9999_10000&sec=1537012717707&di=93b200d1c628327cd3686e48bccae82e&imgtype=0&src=http%3A%2F%2Fimg.mp.sohu.com%2Fupload%2F20170625%2F1f1509fb3265468b895c0515c2c9de3b.png" width="400" height="400" />

---
<center>本文版权归<font color=dark red>深圳大学混沌量化投资实验室 Chaos Quant Lab</font>所有，只用于平时教学学习，未经得作者同意禁止直接转载或编辑后转载。</center>

---

# <font color = 'purple' face = '楷体'>关于利用Statsmodels库分析预测时间序列</font>

## 介绍

> 对于一般的数据探索任务来说，numpy和pandas两者来说已经基本足够了，在进行更深入的统计模型的学习时，Statsmodels无疑是一个更称手的工具。

> 在课堂的理论学习之后，我们可以马上进入实践阶段，也就是说要构建模型，但是如何利用学到的理论来对时间序列进行分析？Statsmodels里可以直接利用的函数，这就减轻了我们自己重新写代码的痛苦，毕竟人生苦短。

> 下面，我将会对一些函数的功能进行解读，不过不要指望我会把里面的函数都讲完，更多的函数的功能请参考[官方文档](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima_model.ARMA.html#statsmodels.tsa.arima_model.ARMA)

> 相关理论的详细部分在课堂已经讲述，故不再赘述。


```python
import statsmodels.api
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
```

需要获取数据


```python
df2 = get_price('000001.XSHG', start_date='2016-01-01',
                      end_date='2017-01-01', frequency='daily', fields=['close'])
df = df2.close.values
df2.plot()
df2.head()
```




<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }
    
    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>close</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-01-04</th>
      <td>3296.258</td>
    </tr>
    <tr>
      <th>2016-01-05</th>
      <td>3287.711</td>
    </tr>
    <tr>
      <th>2016-01-06</th>
      <td>3361.840</td>
    </tr>
    <tr>
      <th>2016-01-07</th>
      <td>3125.002</td>
    </tr>
    <tr>
      <th>2016-01-08</th>
      <td>3186.412</td>
    </tr>
  </tbody>
</table>
</div>



<img src='http://i.caigoubao.cc/627139/bgpc/dff/22.png'/>


获取到数据后，我们需要对它的纯随机性和平稳性进行检验，然后针对不同的序列做相应的处理，选择适合的模型建模。

### 平稳性的检验
- 时序图检验
- 自相关图检验
- 单位根检验

### 白噪声检验（即纯随机性检验）
- 常用的统计量有Q统计量，LB统计量，由样本可得到检验统计量，然后计算出对应的p值，若p值显著大于显著性水平$\alpha$, 则表示该序列不能拒绝纯随机的原假设。

> 对经过检验后被判定为平稳非白噪声序列，就可以利用ARMA模型进行建模

> 对于非平稳时间序列，因为它的均值和方差不稳定，所以需要一些手段将其变为平稳序列，常用的方法为差分

> 例如，取上证指数2018年的收盘价，计算自相关函数，检验统计量Q以及p值。可以初步判断，上证指数去年的收盘价不存在显著序列相关性。

计算自相关函数
一维数组的自相关函数
statsmodels.tsa.stattools.acf(x, unbiased=False, nlags=40, qstat=False, fft=False, alpha=None, missing='none')

参数
----------
x: 为时间序列
nlags: 延迟的自相关数目

返回
-------
acf : 数组
​    自相关函数
confint : array, 可选参数
​    Confidence intervals for the ACF. Returned if confint is not None.
qstat : 数组, 可选参数
​    The Ljung-Box Q-Statistic.  Returned if q_stat is True.
pvalues : 数组, 可选参数
​    The p-values associated with the Q-statistics.  Returned if q_stat is
​    True.

```python
acf,qstat,pvalues = statsmodels.tsa.stattools.acf(df,qstat = True,nlags=10)
acf
```




    array([1.        , 0.94074939, 0.90028989, 0.83965851, 0.80331026,
           0.75353211, 0.73236034, 0.71070701, 0.69445027, 0.67443069,
           0.65049601])



计算自协方差函数


```python
statsmodels.tsa.stattools.acovf(df)[:30]
```




    array([17786.60681013, 16732.73942942, 16013.10235166, 14934.67572538,
           14288.16374395, 13402.77932975, 13026.20534239, 12641.06614561,
           12351.91390064, 11995.83347848, 11570.11681702, 11137.15772245,
           10357.98717242,  9794.95991292,  9033.20972355,  8573.7018781 ,
            7801.52475642,  7530.69128507,  7199.60551547,  7001.59076624,
            6649.94262225,  6233.31174087,  5905.49804051,  5637.03499002,
            5350.88148122,  5094.91366941,  5022.88243591,  4698.01853717,
            4446.611484  ,  4028.8544234 ])



画自相关系数图


```python
statsmodels.graphics.tsaplots.plot_acf(df).show()
```
<img src='http://i.caigoubao.cc/627139/bgpc/dff/33.png'/>
计算偏相关系数


```python
statsmodels.tsa.stattools.pacf(df)
```




    array([ 1.        ,  0.94462078,  0.14320418, -0.18801003,  0.14995075,
           -0.07348824,  0.17645428,  0.09952893, -0.0455198 ,  0.02945173,
           -0.07569166,  0.0154618 , -0.18401555,  0.05081486, -0.0656371 ,
            0.03063943, -0.11652514,  0.11236387,  0.09624518, -0.06900354,
            0.02562782, -0.14266328,  0.13325002,  0.10692806, -0.08132526,
            0.06243219,  0.02830165, -0.10162422, -0.06167632, -0.04436114,
           -0.09349983, -0.15702505, -0.00781215,  0.04242769,  0.20635736,
            0.10785482, -0.06286746,  0.0145844 ,  0.03835777,  0.00843959,
            0.075976  ])



画偏相关系数图


```python
statsmodels.graphics.tsaplots.plot_pacf(df,lags = 20).show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/dff/44.png'/>


单位根检验(ADF test)
- 返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore


```python
print(u'原始序列的ADF检验结果为')
print ('返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore')
statsmodels.tsa.stattools.adfuller(df)
```

    原始序列的ADF检验结果为
    返回值依次为adf、pvalue、usedlag、nobs、critical values、icbest、regresults、resstore





    (-2.2717841322807417,
     0.18122966711927385,
     15,
     228,
     {'1%': -3.4593607492757554,
      '10%': -2.5735714042782396,
      '5%': -2.8743015807562924},
     2256.9605240638502)



可见，序列非平稳

对时间序列进行差分计算


```python
#一阶差分
print('一阶差分结果')
df3 = df2.diff().dropna()
print(df3.head())
df3.plot()
df3 = df3.close.values
```

    一阶差分结果
                  close
    2016-01-05   -8.547
    2016-01-06   74.129
    2016-01-07 -236.838
    2016-01-08   61.410
    2016-01-11 -169.708


<img src='http://i.caigoubao.cc/627139/bgpc/dff/55.png'/>


再进行之前的步骤


```python
statsmodels.graphics.tsaplots.plot_acf(df3).show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/dff/66.png'/>



```python
print('差分后序列的ADF检验结果为')
statsmodels.tsa.stattools.adfuller(df3)
```

    差分后序列的ADF检验结果为





    (-4.750334251126256,
     6.761987299223637e-05,
     15,
     227,
     {'1%': -3.4594900381360034,
      '10%': -2.573601605503697,
      '5%': -2.8743581895178485},
     2231.1151987962194)



故一阶差分后是平稳序列

Ljung-Box检验，检验是否为白噪声


```python
from statsmodels.stats.diagnostic import acorr_ljungbox
print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(df3, lags=1)) #返回统计量和p值
```

    差分序列的白噪声检验结果为： (array([8.64299785]), array([0.00328321]))


可见，白噪声p值小于显著性水平，故此不为白噪声序列，可进行信息提取和分析（2017年和2018年的数据差分后都为白噪声序列，找一个可以分析的非白噪声宽平稳的时间序列也不容易啊）

ARMA模型研究的对象为平稳时间序列。如果序列是非平稳的，需要通过差分的方法将其变为平稳时间序列，也就是所说的ARIMA模型。ARIMA模型需要确定三个阶数，AR模型的阶数p，差分的阶数d和MA模型的阶数q，通常写作ARIMA(p,d,q)。

> 从p值的情况可以看出，一阶差分之后就变为了平稳序列，还可以看出差分可以使得非平稳序列快速变为平稳序列。事实上，一般而言两阶差分就足够将非平稳序列变得平稳。这样，我们确定了d的阶数为1，下面确定AR和MA模型的阶数。

创建一个ARIMA模型
- statsmodels.tsa.arima_model.ARIMA(p,d,q)

先定阶,确定p,q,使用信息准则定阶方法，即bic信息量达到最小的模型阶数


```python
#定阶
pmax = 3
qmax = 3
bic_matrix = [] #bic矩阵
for p in range(pmax+1):
    tmp = []
    for q in range(qmax+1):
        try: 
            tmp.append(statsmodels.tsa.arima_model.ARIMA(df, (p,1,q)).fit().bic)
        except:
            tmp.append(None)
    bic_matrix.append(tmp)

bic_matrix = pd.DataFrame(bic_matrix) #从中可以找出最小值

p,q = bic_matrix.stack().idxmin() #先用stack展平，然后用idxmin找出最小值位置。
print(u'BIC最小的p值和q值为：%s、%s' %(p,q)) 
```

    第1次
    第2次
    第3次
    第4次
    BIC最小的p值和q值为：2、2


创建一个ARIMA模型的报告


```python
model = statsmodels.tsa.arima_model.ARIMA(df,(p,1,q)).fit() #建立ARIMA(2,1,2)模型
```


```python
model.summary()
```




<table class="simpletable">
<caption>ARIMA Model Results</caption>
<tr>
  <th>Dep. Variable:</th>        <td>D.y</td>       <th>  No. Observations:  </th>    <td>243</td>   
</tr>
<tr>
  <th>Model:</th>          <td>ARIMA(2, 1, 2)</td>  <th>  Log Likelihood     </th> <td>-1235.053</td>
</tr>
<tr>
  <th>Method:</th>             <td>css-mle</td>     <th>  S.D. of innovations</th>  <td>38.890</td>  
</tr>
<tr>
  <th>Date:</th>          <td>Fri, 09 Nov 2018</td> <th>  AIC                </th> <td>2482.106</td> 
</tr>
<tr>
  <th>Time:</th>              <td>12:47:37</td>     <th>  BIC                </th> <td>2503.064</td> 
</tr>
<tr>
  <th>Sample:</th>                <td>1</td>        <th>  HQIC               </th> <td>2490.548</td> 
</tr>
<tr>
  <th></th>                       <td> </td>        <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>z</th>      <th>P>|z|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>const</th>     <td>   -0.7183</td> <td>    2.417</td> <td>   -0.297</td> <td> 0.767</td> <td>   -5.455</td> <td>    4.019</td>
</tr>
<tr>
  <th>ar.L1.D.y</th> <td>   -1.8912</td> <td>    0.037</td> <td>  -50.753</td> <td> 0.000</td> <td>   -1.964</td> <td>   -1.818</td>
</tr>
<tr>
  <th>ar.L2.D.y</th> <td>   -0.9649</td> <td>    0.035</td> <td>  -27.770</td> <td> 0.000</td> <td>   -1.033</td> <td>   -0.897</td>
</tr>
<tr>
  <th>ma.L1.D.y</th> <td>    1.8171</td> <td>    0.041</td> <td>   44.810</td> <td> 0.000</td> <td>    1.738</td> <td>    1.897</td>
</tr>
<tr>
  <th>ma.L2.D.y</th> <td>    0.9180</td> <td>    0.045</td> <td>   20.272</td> <td> 0.000</td> <td>    0.829</td> <td>    1.007</td>
</tr>
</table>
<table class="simpletable">
<caption>Roots</caption>
<tr>
    <td></td>   <th>            Real</th>  <th>         Imaginary</th> <th>         Modulus</th>  <th>        Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>          -0.9800</td> <td>          -0.2756j</td> <td>           1.0180</td> <td>          -0.4564</td>
</tr>
<tr>
  <th>AR.2</th> <td>          -0.9800</td> <td>          +0.2756j</td> <td>           1.0180</td> <td>           0.4564</td>
</tr>
<tr>
  <th>MA.1</th> <td>          -0.9897</td> <td>          -0.3314j</td> <td>           1.0437</td> <td>          -0.4486</td>
</tr>
<tr>
  <th>MA.2</th> <td>          -0.9897</td> <td>          +0.3314j</td> <td>           1.0437</td> <td>           0.4486</td>
</tr>
</table>




```python
model.summary2() #给出一份模型报告
```




<table class="simpletable">
<tr>
        <td>Model:</td>              <td>ARIMA</td>              <td>BIC:</td>         <td>2503.0645</td>
</tr>
<tr>
  <td>Dependent Variable:</td>        <td>D.y</td>          <td>Log-Likelihood:</td>    <td>-1235.1</td> 
</tr>
<tr>
         <td>Date:</td>        <td>2018-11-09 12:43</td>        <td>Scale:</td>         <td>1.0000</td>  
</tr>
<tr>
   <td>No. Observations:</td>         <td>243</td>              <td>Method:</td>        <td>css-mle</td> 
</tr>
<tr>
       <td>Df Model:</td>              <td>5</td>               <td>Sample:</td>           <td>1</td>    
</tr>
<tr>
     <td>Df Residuals:</td>           <td>238</td>                 <td></td>               <td>4</td>    
</tr>
<tr>
      <td>Converged:</td>           <td>1.0000</td>      <td>S.D. of innovations:</td>  <td>38.890</td>  
</tr>
<tr>
    <td>No. Iterations:</td>        <td>32.0000</td>             <td>HQIC:</td>        <td>2490.548</td> 
</tr>
<tr>
         <td>AIC:</td>             <td>2482.1061</td>              <td></td>               <td></td>     
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>       <th>Coef.</th>  <th>Std.Err.</th>     <th>t</th>     <th>P>|t|</th> <th>[0.025</th>  <th>0.975]</th> 
</tr>
<tr>
  <th>const</th>     <td>-0.7183</td>  <td>2.4169</td>   <td>-0.2972</td> <td>0.7666</td> <td>-5.4553</td> <td>4.0187</td> 
</tr>
<tr>
  <th>ar.L1.D.y</th> <td>-1.8912</td>  <td>0.0373</td>  <td>-50.7528</td> <td>0.0000</td> <td>-1.9642</td> <td>-1.8181</td>
</tr>
<tr>
  <th>ar.L2.D.y</th> <td>-0.9649</td>  <td>0.0347</td>  <td>-27.7703</td> <td>0.0000</td> <td>-1.0330</td> <td>-0.8968</td>
</tr>
<tr>
  <th>ma.L1.D.y</th> <td>1.8171</td>   <td>0.0406</td>   <td>44.8105</td> <td>0.0000</td> <td>1.7376</td>  <td>1.8966</td> 
</tr>
<tr>
  <th>ma.L2.D.y</th> <td>0.9180</td>   <td>0.0453</td>   <td>20.2725</td> <td>0.0000</td> <td>0.8293</td>  <td>1.0068</td> 
</tr>
</table>
<table class="simpletable">
<tr>
    <td></td>    <th>Real</th>   <th>Imaginary</th> <th>Modulus</th> <th>Frequency</th>
</tr>
<tr>
  <th>AR.1</th> <td>-0.9800</td>  <td>-0.2756</td>  <td>1.0180</td>   <td>-0.4564</td> 
</tr>
<tr>
  <th>AR.2</th> <td>-0.9800</td>  <td>0.2756</td>   <td>1.0180</td>   <td>0.4564</td>  
</tr>
<tr>
  <th>MA.1</th> <td>-0.9897</td>  <td>-0.3314</td>  <td>1.0437</td>   <td>-0.4486</td> 
</tr>
<tr>
  <th>MA.2</th> <td>-0.9897</td>  <td>0.3314</td>   <td>1.0437</td>   <td>0.4486</td>  
</tr>
</table>



应用时间序列模型进行预测


```python
model.forecast(5) #作为期5天的预测，返回预测结果、标准误差、置信区间。
```




    (array([3098.83314629, 3101.33544042, 3098.46869172, 3098.70610347,
            3098.25334163]),
     array([38.88991532, 53.00061968, 66.18220249, 75.12873166, 85.00682791]),
     array([[3022.6103129 , 3175.05597967],
            [2997.4561347 , 3205.21474615],
            [2968.75395841, 3228.18342502],
            [2951.45649521, 3245.95571173],
            [2931.64302048, 3264.86366278]]))




```python
model.plot_predict().show()
```

<img src='http://i.caigoubao.cc/627139/bgpc/dff/77.png'/>


> 对于AR, MA, ARMA模型的构建同上，故不再赘述，在现实生活中找到不做处理的时间序列也不容易，good luck!
