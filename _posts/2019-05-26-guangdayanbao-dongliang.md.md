---
layout:     post
title:  "研报复现：《光大证券·多因子系列报告之二十二：再论动量因子》"
subtitle:   ""
date:       2019-05-26 12:00:00
author:     "@Ian666"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
tags:

    - 机器学习
    - 降维
mathjax: false

---


用于因子策略回测效果检查

本期研报来自于光大证券金融工程的多因子系列，基于大类因子动量因子的讨论——《光大证券·多因子系列报告之二十二：再论动量因子》


```python
'''
根据聚宽高频因子挖掘大赛比赛专用模板修改
初始资金：2000000
建议回测时间范围
每日调仓 一年 回测日期：20180720-20190720
每周调仓 三年 回测日期：20160720-20190720
每月调仓 五年 回测日期：20140720-20190720

股票池：中证500
每日持仓：数量固定为股票池的20只，持仓均为等权重持有
换仓时间：默认14:50
交易成本：不考虑滑点，印花税1‰，佣金2.5‱、最低5元
'''

# 导入函数库
from jqdata import *
import numpy as np
import pandas as pd
import jqfactor

################################################# 以下内容根据研究因子内容定义 ########################################################

# 定义因子
def calc_factor(context):
    '''
    用户自定义因子，要求返回一个 Series，index为股票code，value为因子值
    我们会买入「因子值最小」的20只，如果您想使用买入「因子值最大」的20只股票，只需将您的结果「乘以-1.0」即可，详见函数下方 return 部分
    '''
    # 获取股票池,g.stock_pool为因子挖掘的对象股票池，用户不可对此股票池进行二次筛选
    stocks = g.stock_pool
    # 获取当前时间
    now  =  context.current_dt
    # 获取数据
    df = get_price(stocks,end_date=context.previous_date,count=21,fields=['close'])['close'] 
    far = df.iloc[-1,:]/df.iloc[0,:] - 1
    
    ###### 中性化等数据处理模块，用户根据需要决定是否使用 ######
    # 中位数去极值
    # far = jqfactor.winsorize_med(far, scale=3, inclusive=True, inf2nan=True)
    # # 行业市值对数中性化
    far = jqfactor.neutralize(far, how=['market_cap'], date=g.d)
    # # zscore标准化
    # far = jqfactor.standardlize(far, inf2nan=True)
    # 去除 nan 值
    # far = far.dropna()
    
    return far
    #如想选择因子值最大的20只股票，请注释上方`return far`。使用下方的return：
    # return far * -1.0
    

# 开盘前运行函数
def before_market_open(context):
    '''
    盘后运行函数，可选实现
    '''
    pass

## 收盘后运行函数
def after_market_close(context):
    '''
    盘后运行函数，可选实现
    '''
    pass



################################################# 以下内容除设置运行周期，其他地方不用修改 ########################################################

# 初始化函数，设定基准等等
def initialize(context):
    # 设定500等权作为基准
    g.benchmark = '000982.XSHG'
    set_benchmark(g.benchmark)
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.00025, close_commission=0.00025, min_commission=5),type='stock')
    # 滑点
    set_slippage(FixedSlippage(0.0))
    # 初始化因子设置
    factor_analysis_initialize(context)
    # 定义股票池
    set_stockpool(context)
    # 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    run_daily(set_stockpool, time='before_open', reference_security='000300.XSHG')
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
    
    #设置策略交易时间间隔
    #run_daily(trade, time='14:50', reference_security='000300.XSHG')
    run_weekly(trade,1, time='14:50', reference_security='000300.XSHG')
    #run_monthly(trade,1, time='14:50', reference_security='000300.XSHG')
    
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')

# 定义股票池
def set_stockpool(context):
    # 获取股票池
    stocks = get_index_stocks(g.benchmark,context.previous_date)
    paused_series = get_price(stocks,end_date=context.current_dt,count=1,fields='paused')['paused'].iloc[0]
    # g.stock_pool 为因子挖掘的对象股票池，用户不可对此股票池进行二次筛选
    g.stock_pool =  paused_series[paused_series==False].index.tolist()

# 定义需要用到的全局变量
def factor_analysis_initialize(context):
    # g.weight_method 为加权方式, "avg"按平均加权
    g.weight_method = "avg"
    weight_method_model = {"avg": "平均加权"}
    # 持仓股票数量
    g.buy_num = 20
    # g.sell为卖出股票权重列表
    g.sell = pd.Series(dtype=float)
    # g.buy为买入股票权重列表
    g.buy = pd.Series(dtype=float)
    #g.ind为行业分类
    g.ind = 'jq_l1'
    # g.d 为获取昨天的时间点
    g.d = context.previous_date

# 对因子进行分析计算出每日买入或卖出的股票
def fac(context):
    # 获取因子值
    far = calc_factor(context)
    # 买入股票池
    try:
        buy = far.sort_values(ascending=True).index.tolist()[:g.buy_num]
    except:
        buy = far.order(ascending=True).index.tolist()[:g.buy_num]
    # 买卖股票权重
    if g.weight_method == "avg":
        buy_weight = pd.Series(1. / len(buy), index=buy)
    else:
        raise ValueError('invalid weight_method %s', weight_method)

    return buy_weight

#股票交易
def trade(context):
    # 计算买入卖出的股票和权重
    try:
        factor_analysis_initialize(context)
        g.buy = fac(context)
    except ValueError:
        if "Bin edges must be unique" in str(e):
            log.error("计算因子值过程出错！")
        else:
            raise
    
    for s in context.portfolio.positions.keys():
        if s not in g.buy.index:
            order_target_value(s, 0)
    

    long_cash = context.portfolio.total_value
    for s in g.buy.index:
        order_target_value(s, g.buy.loc[s] * 0.98 * long_cash)

# 买入股票
def buy(context):
    # 计算买入卖出的股票和权重
    try:
        factor_analysis_initialize(context)
        g.buy = fac(context)
    except ValueError:
        if "Bin edges must be unique" in str(e):
            log.error("计算因子值过程出错！")
        else:
            raise
    long_cash = context.portfolio.total_value
    for s in g.buy.index:
        order_target_value(s, g.buy.loc[s] * 0.98 * long_cash)

# 卖出股票
def sell(context):
    for s in context.portfolio.positions.keys():
        order_target_value(s, 0)
        
```

请先从下面内容开始

因子分析基础模板


```python
#导入需要的数据库
from jqfactor import *
from jqdata import *
import pandas as pd
import warnings  
warnings.filterwarnings('ignore') 

#获取日期列表
def get_tradeday_list(start,end,frequency=None,count=None):
    if count != None:
        df = get_price('000001.XSHG',end_date=end,count=count)
    else:
        df = get_price('000001.XSHG',start_date=start,end_date=end)
    if frequency == None or frequency =='day':
        return df.index
    else:
        df['year-month'] = [str(i)[0:7] for i in df.index]
        if frequency == 'month':
            return df.drop_duplicates('year-month').index
        elif frequency == 'quarter':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='04') | (df['month']=='07') | (df['month']=='10') ]
            return df.drop_duplicates('year-month').index
        elif frequency =='halfyear':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='06')]
            return df.drop_duplicates('year-month').index 
```

===初始化====


```python
# 设置起止时间
start='2016-07-01'
end='2019-07-01'
# 设置调仓周期
periods=(5,10,20)
# 设置分层数量
quantiles=5
#获取日期列表
date_list = get_tradeday_list(start=start,end=end,count=None)#获取回测日期间的所有交易日
date_list
```




    DatetimeIndex(['2016-07-01', '2016-07-04', '2016-07-05', '2016-07-06',
                   '2016-07-07', '2016-07-08', '2016-07-11', '2016-07-12',
                   '2016-07-13', '2016-07-14',
                   ...
                   '2019-06-18', '2019-06-19', '2019-06-20', '2019-06-21',
                   '2019-06-24', '2019-06-25', '2019-06-26', '2019-06-27',
                   '2019-06-28', '2019-07-01'],
                  dtype='datetime64[ns]', length=730, freq=None)



===原始计算因子数据===


* 进行因子值函数定义

* 循环日期获取因子值

股票价格的动量（Momentum） ， 顾名思义代表的是股价在一定时间内延续前期走势的现象。 不过与海外长期的研究和经验相悖的是， 在 A 股市场， 我们发现股价的反转（Reverse） 效应要远强于动量效应， 且短期反转因子的历史收益非常出色。

但常用动量因子也存在单调性不佳， 多头收益不稳定的问题， 因此参考研报我们尝试从不同角度出发对动量因子进行改造， 寻找提升常用动量因子选股 效果和稳定性的方法。

在该多因子系列报告中， 曾给出过动量类因子的因子测试结论， 报告中测试的几个常用动量因子，也是我们经常接触到的基础动量因子，明细如下



下面我们将以统计周期为21天的动量因子为例进行探索演示


```python
#定义要计算的动量因子
def factor_cal(pool,date):
    df = get_price(pool,end_date=date,count=21,fields=['close'])['close'] 
    far = df.iloc[-1,:]/df.iloc[0,:] - 1
    return far
factor_cal(['000001.XSHE','600000.XSHG'],'2019-07-12')
```




    000001.XSHE    0.142395
    600000.XSHG   -0.022901
    dtype: float64




```python
#定义一个空的dataframe记录因子值
factor_df = pd.DataFrame()
#循环计算给定日期范围的因子值
mark = 1
for d in date_list:
    pool = get_index_stocks('000905.XSHG',date=d)
    far = factor_cal(pool,d)
    if mark == 1:
        factor_df = far
        mark = 0
    else:
        factor_df = pd.concat([far,factor_df],axis=1)
#将columns更改为可以日期标签
factor_df.columns = date_list
factor_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2016-07-01 00:00:00</th>
      <th>2016-07-04 00:00:00</th>
      <th>2016-07-05 00:00:00</th>
      <th>2016-07-06 00:00:00</th>
      <th>2016-07-07 00:00:00</th>
      <th>2016-07-08 00:00:00</th>
      <th>2016-07-11 00:00:00</th>
      <th>2016-07-12 00:00:00</th>
      <th>2016-07-13 00:00:00</th>
      <th>2016-07-14 00:00:00</th>
      <th>2016-07-15 00:00:00</th>
      <th>2016-07-18 00:00:00</th>
      <th>2016-07-19 00:00:00</th>
      <th>2016-07-20 00:00:00</th>
      <th>2016-07-21 00:00:00</th>
      <th>2016-07-22 00:00:00</th>
      <th>2016-07-25 00:00:00</th>
      <th>2016-07-26 00:00:00</th>
      <th>2016-07-27 00:00:00</th>
      <th>2016-07-28 00:00:00</th>
      <th>2016-07-29 00:00:00</th>
      <th>2016-08-01 00:00:00</th>
      <th>2016-08-02 00:00:00</th>
      <th>2016-08-03 00:00:00</th>
      <th>2016-08-04 00:00:00</th>
      <th>2016-08-05 00:00:00</th>
      <th>2016-08-08 00:00:00</th>
      <th>2016-08-09 00:00:00</th>
      <th>2016-08-10 00:00:00</th>
      <th>2016-08-11 00:00:00</th>
      <th>2016-08-12 00:00:00</th>
      <th>2016-08-15 00:00:00</th>
      <th>2016-08-16 00:00:00</th>
      <th>2016-08-17 00:00:00</th>
      <th>2016-08-18 00:00:00</th>
      <th>2016-08-19 00:00:00</th>
      <th>2016-08-22 00:00:00</th>
      <th>2016-08-23 00:00:00</th>
      <th>2016-08-24 00:00:00</th>
      <th>2016-08-25 00:00:00</th>
      <th>...</th>
      <th>2019-05-06 00:00:00</th>
      <th>2019-05-07 00:00:00</th>
      <th>2019-05-08 00:00:00</th>
      <th>2019-05-09 00:00:00</th>
      <th>2019-05-10 00:00:00</th>
      <th>2019-05-13 00:00:00</th>
      <th>2019-05-14 00:00:00</th>
      <th>2019-05-15 00:00:00</th>
      <th>2019-05-16 00:00:00</th>
      <th>2019-05-17 00:00:00</th>
      <th>2019-05-20 00:00:00</th>
      <th>2019-05-21 00:00:00</th>
      <th>2019-05-22 00:00:00</th>
      <th>2019-05-23 00:00:00</th>
      <th>2019-05-24 00:00:00</th>
      <th>2019-05-27 00:00:00</th>
      <th>2019-05-28 00:00:00</th>
      <th>2019-05-29 00:00:00</th>
      <th>2019-05-30 00:00:00</th>
      <th>2019-05-31 00:00:00</th>
      <th>2019-06-03 00:00:00</th>
      <th>2019-06-04 00:00:00</th>
      <th>2019-06-05 00:00:00</th>
      <th>2019-06-06 00:00:00</th>
      <th>2019-06-10 00:00:00</th>
      <th>2019-06-11 00:00:00</th>
      <th>2019-06-12 00:00:00</th>
      <th>2019-06-13 00:00:00</th>
      <th>2019-06-14 00:00:00</th>
      <th>2019-06-17 00:00:00</th>
      <th>2019-06-18 00:00:00</th>
      <th>2019-06-19 00:00:00</th>
      <th>2019-06-20 00:00:00</th>
      <th>2019-06-21 00:00:00</th>
      <th>2019-06-24 00:00:00</th>
      <th>2019-06-25 00:00:00</th>
      <th>2019-06-26 00:00:00</th>
      <th>2019-06-27 00:00:00</th>
      <th>2019-06-28 00:00:00</th>
      <th>2019-07-01 00:00:00</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>000006.XSHE</th>
      <td>0.050943</td>
      <td>0.028302</td>
      <td>0.018450</td>
      <td>0.007326</td>
      <td>-0.005396</td>
      <td>0.048872</td>
      <td>0.070881</td>
      <td>0.033582</td>
      <td>0.005525</td>
      <td>0.013084</td>
      <td>0.009276</td>
      <td>-0.030576</td>
      <td>-0.019856</td>
      <td>0.000000</td>
      <td>-0.005464</td>
      <td>-0.037906</td>
      <td>-0.024164</td>
      <td>-0.029358</td>
      <td>-0.043557</td>
      <td>-0.014870</td>
      <td>-0.100170</td>
      <td>-0.098639</td>
      <td>-0.084459</td>
      <td>-0.085427</td>
      <td>-0.113238</td>
      <td>-0.146067</td>
      <td>-0.171429</td>
      <td>-0.171561</td>
      <td>-0.162037</td>
      <td>-0.183206</td>
      <td>-0.184569</td>
      <td>-0.151145</td>
      <td>-0.171898</td>
      <td>-0.182229</td>
      <td>-0.187870</td>
      <td>-0.192420</td>
      <td>-0.214599</td>
      <td>-0.208999</td>
      <td>-0.203757</td>
      <td>-0.202963</td>
      <td>...</td>
      <td>0.121523</td>
      <td>0.138235</td>
      <td>0.116312</td>
      <td>0.118741</td>
      <td>0.153515</td>
      <td>0.172805</td>
      <td>0.155492</td>
      <td>0.139403</td>
      <td>0.135838</td>
      <td>0.089744</td>
      <td>0.059829</td>
      <td>0.060734</td>
      <td>0.097561</td>
      <td>0.098694</td>
      <td>0.080344</td>
      <td>0.105036</td>
      <td>0.052260</td>
      <td>-0.018841</td>
      <td>-0.032117</td>
      <td>0.013353</td>
      <td>0.008863</td>
      <td>0.000000</td>
      <td>0.027697</td>
      <td>0.027941</td>
      <td>0.034125</td>
      <td>0.063253</td>
      <td>0.050975</td>
      <td>0.074924</td>
      <td>0.056489</td>
      <td>0.057229</td>
      <td>0.090062</td>
      <td>0.089231</td>
      <td>0.092476</td>
      <td>0.081633</td>
      <td>0.044978</td>
      <td>0.038864</td>
      <td>0.044248</td>
      <td>0.016200</td>
      <td>0.007353</td>
      <td>0.008982</td>
    </tr>
    <tr>
      <th>000008.XSHE</th>
      <td>0.021053</td>
      <td>-0.015666</td>
      <td>0.015915</td>
      <td>0.000000</td>
      <td>0.018667</td>
      <td>0.054348</td>
      <td>0.048649</td>
      <td>0.005249</td>
      <td>-0.018135</td>
      <td>-0.026178</td>
      <td>-0.038560</td>
      <td>-0.064838</td>
      <td>-0.049875</td>
      <td>-0.038071</td>
      <td>-0.045000</td>
      <td>-0.074813</td>
      <td>-0.061224</td>
      <td>-0.048223</td>
      <td>-0.062814</td>
      <td>-0.047859</td>
      <td>-0.112150</td>
      <td>-0.088095</td>
      <td>-0.158482</td>
      <td>-0.149888</td>
      <td>-0.198718</td>
      <td>-0.210300</td>
      <td>-0.206009</td>
      <td>-0.197895</td>
      <td>-0.180467</td>
      <td>-0.204167</td>
      <td>-0.186192</td>
      <td>-0.150424</td>
      <td>-0.166320</td>
      <td>-0.184265</td>
      <td>-0.175258</td>
      <td>-0.188259</td>
      <td>-0.190083</td>
      <td>-0.205645</td>
      <td>-0.189409</td>
      <td>-0.179752</td>
      <td>...</td>
      <td>-0.065789</td>
      <td>-0.065541</td>
      <td>-0.082524</td>
      <td>-0.088462</td>
      <td>-0.070260</td>
      <td>-0.102830</td>
      <td>-0.088292</td>
      <td>-0.083734</td>
      <td>-0.084615</td>
      <td>-0.103612</td>
      <td>-0.099906</td>
      <td>-0.035748</td>
      <td>-0.073446</td>
      <td>-0.068868</td>
      <td>-0.054650</td>
      <td>-0.071361</td>
      <td>-0.103286</td>
      <td>-0.089623</td>
      <td>-0.089720</td>
      <td>-0.064333</td>
      <td>-0.080074</td>
      <td>-0.035441</td>
      <td>0.011788</td>
      <td>0.031746</td>
      <td>0.046324</td>
      <td>0.042281</td>
      <td>0.012634</td>
      <td>0.007759</td>
      <td>-0.074733</td>
      <td>-0.108475</td>
      <td>-0.100085</td>
      <td>-0.111204</td>
      <td>-0.071678</td>
      <td>-0.046763</td>
      <td>-0.130833</td>
      <td>-0.122705</td>
      <td>-0.069056</td>
      <td>-0.066079</td>
      <td>-0.060579</td>
      <td>-0.072807</td>
    </tr>
    <tr>
      <th>000009.XSHE</th>
      <td>0.010395</td>
      <td>-0.024641</td>
      <td>-0.022312</td>
      <td>0.000000</td>
      <td>0.026971</td>
      <td>0.021277</td>
      <td>0.004132</td>
      <td>0.010526</td>
      <td>-0.038855</td>
      <td>-0.006466</td>
      <td>-0.004292</td>
      <td>-0.048880</td>
      <td>-0.008282</td>
      <td>0.010638</td>
      <td>0.004219</td>
      <td>-0.037815</td>
      <td>-0.032538</td>
      <td>-0.033403</td>
      <td>-0.058704</td>
      <td>-0.040900</td>
      <td>-0.064202</td>
      <td>0.010373</td>
      <td>-0.040856</td>
      <td>-0.062136</td>
      <td>-0.126812</td>
      <td>-0.139194</td>
      <td>-0.149385</td>
      <td>-0.192177</td>
      <td>-0.114130</td>
      <td>-0.163964</td>
      <td>-0.158845</td>
      <td>-0.102377</td>
      <td>-0.113761</td>
      <td>-0.142336</td>
      <td>-0.156584</td>
      <td>-0.166375</td>
      <td>-0.194056</td>
      <td>-0.198997</td>
      <td>-0.143847</td>
      <td>-0.159794</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



进行因子数据优化
* 标准化
* 风格中性化

根据研报的观点，动量（反转） 效应的成因是投资者认识出现偏差，对信息的解读不够及时充分。 也可以认为动量（反转） 效应是来自于投资者认知偏差或者说噪音交易行为。

由此可以推测， 散户集中程度或者说流动性因素是影响动量因子效果的一个重要变量。 因此， 我们会自然的联想到采用截面中性化的方法， 将衡量散户集中程度的流动性因素从原始动量因子中剥离。

因此，我们接下来对原始因子数据进行特定风格中性化处理


```python
#数据清洗、包括去极值、标准化、中性化等,并加入y值
for date in date_list:
    #对数据进行处理、标准化、去极值、中性化
    #factor_df = winsorize_med(factor_df, scale=3, inclusive=True, inf2nan=True, axis=0) #中位数去极值处理
    se = standardlize(factor_df[date], inf2nan=True) #对每列做标准化处理
    se = neutralize(se, how=['liquidity'], date=date)#剔除原始因子值与流动性相关的部分
    factor_df[date] = se
#进行转置，调整为分析可用的格式
factor_df = factor_df.T
factor_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>000006.XSHE</th>
      <th>000008.XSHE</th>
      <th>000009.XSHE</th>
      <th>000012.XSHE</th>
      <th>000021.XSHE</th>
      <th>000025.XSHE</th>
      <th>000027.XSHE</th>
      <th>000028.XSHE</th>
      <th>000030.XSHE</th>
      <th>000031.XSHE</th>
      <th>000039.XSHE</th>
      <th>000049.XSHE</th>
      <th>000050.XSHE</th>
      <th>000060.XSHE</th>
      <th>000061.XSHE</th>
      <th>000062.XSHE</th>
      <th>000066.XSHE</th>
      <th>000078.XSHE</th>
      <th>000088.XSHE</th>
      <th>000089.XSHE</th>
      <th>000090.XSHE</th>
      <th>000099.XSHE</th>
      <th>000156.XSHE</th>
      <th>000158.XSHE</th>
      <th>000301.XSHE</th>
      <th>000400.XSHE</th>
      <th>000401.XSHE</th>
      <th>000417.XSHE</th>
      <th>000418.XSHE</th>
      <th>000426.XSHE</th>
      <th>000488.XSHE</th>
      <th>000501.XSHE</th>
      <th>000513.XSHE</th>
      <th>000517.XSHE</th>
      <th>000519.XSHE</th>
      <th>000528.XSHE</th>
      <th>000536.XSHE</th>
      <th>000537.XSHE</th>
      <th>000541.XSHE</th>
      <th>000543.XSHE</th>
      <th>...</th>
      <th>603169.XSHG</th>
      <th>603188.XSHG</th>
      <th>603198.XSHG</th>
      <th>603225.XSHG</th>
      <th>603228.XSHG</th>
      <th>603233.XSHG</th>
      <th>603328.XSHG</th>
      <th>603355.XSHG</th>
      <th>603369.XSHG</th>
      <th>603377.XSHG</th>
      <th>603444.XSHG</th>
      <th>603486.XSHG</th>
      <th>603501.XSHG</th>
      <th>603515.XSHG</th>
      <th>603517.XSHG</th>
      <th>603528.XSHG</th>
      <th>603555.XSHG</th>
      <th>603556.XSHG</th>
      <th>603567.XSHG</th>
      <th>603568.XSHG</th>
      <th>603569.XSHG</th>
      <th>603589.XSHG</th>
      <th>603650.XSHG</th>
      <th>603658.XSHG</th>
      <th>603659.XSHG</th>
      <th>603698.XSHG</th>
      <th>603699.XSHG</th>
      <th>603712.XSHG</th>
      <th>603766.XSHG</th>
      <th>603799.XSHG</th>
      <th>603806.XSHG</th>
      <th>603816.XSHG</th>
      <th>603866.XSHG</th>
      <th>603868.XSHG</th>
      <th>603877.XSHG</th>
      <th>603883.XSHG</th>
      <th>603885.XSHG</th>
      <th>603888.XSHG</th>
      <th>603899.XSHG</th>
      <th>603939.XSHG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2016-07-01</th>
      <td>0.116279</td>
      <td>-0.230939</td>
      <td>-0.307165</td>
      <td>-0.018065</td>
      <td>-1.115277</td>
      <td>-0.499227</td>
      <td>0.232659</td>
      <td>-0.346900</td>
      <td>NaN</td>
      <td>0.389321</td>
      <td>0.668226</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.652505</td>
      <td>-0.651069</td>
      <td>-0.937019</td>
      <td>0.396109</td>
      <td>-1.316769</td>
      <td>NaN</td>
      <td>0.703391</td>
      <td>0.010869</td>
      <td>NaN</td>
      <td>-1.094827</td>
      <td>-0.483434</td>
      <td>0.537527</td>
      <td>-0.068281</td>
      <td>0.746313</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.537537</td>
      <td>-0.392134</td>
      <td>0.041164</td>
      <td>0.888780</td>
      <td>NaN</td>
      <td>-0.476111</td>
      <td>0.022197</td>
      <td>-0.153983</td>
      <td>-0.448221</td>
      <td>-0.196065</td>
      <td>-0.029184</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.036071</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.215926</td>
      <td>0.173149</td>
      <td>1.773804</td>
      <td>-1.312008</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.237745</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.025671</td>
      <td>NaN</td>
      <td>0.632709</td>
      <td>NaN</td>
      <td>0.346798</td>
      <td>-0.040118</td>
      <td>NaN</td>
      <td>0.128373</td>
      <td>0.359723</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.315883</td>
    </tr>
    <tr>
      <th>2016-07-04</th>
      <td>0.231738</td>
      <td>-0.280201</td>
      <td>-0.303062</td>
      <td>-0.054342</td>
      <td>-0.708107</td>
      <td>-0.221434</td>
      <td>0.410154</td>
      <td>-0.384254</td>
      <td>NaN</td>
      <td>0.312459</td>
      <td>0.584113</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.682699</td>
      <td>-0.757832</td>
      <td>-1.067445</td>
      <td>0.567712</td>
      <td>-1.425293</td>
      <td>NaN</td>
      <td>0.421777</td>
      <td>0.139029</td>
      <td>NaN</td>
      <td>-0.910318</td>
      <td>-0.872446</td>
      <td>0.751647</td>
      <td>0.068043</td>
      <td>0.191558</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.781548</td>
      <td>-0.224895</td>
      <td>0.127288</td>
      <td>0.943820</td>
      <td>NaN</td>
      <td>0.083240</td>
      <td>-0.075114</td>
      <td>-0.934088</td>
      <td>-0.378330</td>
      <td>-0.062909</td>
      <td>0.267830</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.074967</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.288378</td>
      <td>0.627841</td>
      <td>1.260432</td>
      <td>-1.129653</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.391451</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.219668</td>
      <td>NaN</td>
      <td>0.930923</td>
      <td>NaN</td>
      <td>0.998194</td>
      <td>0.677699</td>
      <td>NaN</td>
      <td>0.143252</td>
      <td>0.843557</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.788913</td>
    </tr>
    <tr>
      <th>2016-07-05</th>
      <td>0.035329</td>
      <td>0.031336</td>
      <td>-0.389137</td>
      <td>0.010385</td>
      <td>-0.971491</td>
      <td>-0.465154</td>
      <td>0.334051</td>
      <td>-0.545740</td>
      <td>NaN</td>
      <td>0.111782</td>
      <td>0.568943</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.143460</td>
      <td>-0.339332</td>
      <td>-1.027743</td>
      <td>-0.403524</td>
      <td>-1.595908</td>
      <td>NaN</td>
      <td>0.354445</td>
      <td>0.175401</td>
      <td>NaN</td>
      <td>-0.776015</td>
      <td>-0.617827</td>
      <td>0.937806</td>
      <td>-0.076594</td>
      <td>-0.094896</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.046214</td>
      <td>-0.245540</td>
      <td>-0.034545</td>
      <td>0.371119</td>
      <td>NaN</td>
      <td>0.308446</td>
      <td>-0.106853</td>
      <td>-1.002989</td>
      <td>-0.551899</td>
      <td>-0.055793</td>
      <td>-0.113595</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.324256</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.344898</td>
      <td>0.317328</td>
      <td>1.051565</td>
      <td>-1.186974</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.581834</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.096823</td>
      <td>NaN</td>
      <td>0.731554</td>
      <td>NaN</td>
      <td>0.763084</td>
      <td>0.362635</td>
      <td>NaN</td>
      <td>-0.243363</td>
      <td>0.714846</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.708468</td>
    </tr>
    <tr>
      <th>2016-07-06</th>
      <td>-0.023627</td>
      <td>-0.110300</td>
      <td>-0.071705</td>
      <td>0.118002</td>
      <td>-0.776730</td>
      <td>-0.069136</td>
      <td>0.290735</td>
      <td>-0.816949</td>
      <td>NaN</td>
      <td>0.059781</td>
      <td>0.512792</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.005006</td>
      <td>-0.069321</td>
      <td>-1.039786</td>
      <td>-0.592182</td>
      <td>-1.734341</td>
      <td>NaN</td>
      <td>0.263885</td>
      <td>0.197672</td>
      <td>NaN</td>
      <td>-0.648460</td>
      <td>-0.622340</td>
      <td>0.743690</td>
      <td>-0.223395</td>
      <td>0.732000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.132459</td>
      <td>-0.027701</td>
      <td>-0.034763</td>
      <td>0.321193</td>
      <td>NaN</td>
      <td>0.837059</td>
      <td>-0.021498</td>
      <td>-0.937775</td>
      <td>-0.567653</td>
      <td>-0.151441</td>
      <td>0.174682</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.464886</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.035668</td>
      <td>0.071702</td>
      <td>0.982961</td>
      <td>-1.789795</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.735023</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.045010</td>
      <td>NaN</td>
      <td>0.693245</td>
      <td>NaN</td>
      <td>0.929924</td>
      <td>0.280750</td>
      <td>NaN</td>
      <td>-0.573270</td>
      <td>0.507304</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.721318</td>
    </tr>
    <tr>
      <th>2016-07-07</th>
      <td>-0.195936</td>
      <td>0.129881</td>
      <td>0.289773</td>
      <td>-0.142384</td>
      <td>-0.822338</td>
      <td>-0.166412</td>
      <td>0.453323</td>
      <td>-1.004299</td>
      <td>NaN</td>
      <td>0.030255</td>
      <td>0.593486</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.160475</td>
      <td>-0.035896</td>
      <td>-0.951212</td>
      <td>-0.548201</td>
      <td>-2.091365</td>
      <td>NaN</td>
      <td>0.275797</td>
      <td>0.195717</td>
      <td>NaN</td>
      <td>-0.249111</td>
      <td>-0.351283</td>
      <td>0.738504</td>
      <td>0.026681</td>
      <td>0.614991</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.676250</td>
      <td>-0.080624</td>
      <td>-0.075823</td>
      <td>-0.322200</td>
      <td>NaN</td>
      <td>0.777190</td>
      <td>0.059828</td>
      <td>-1.196252</td>
      <td>-0.510519</td>
      <td>-0.217232</td>
      <td>0.269244</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.502855</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-0.116687</td>
      <td>0.143196</td>
      <td>0.821851</td>
      <td>-1.334644</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.621746</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.032007</td>
      <td>NaN</td>
      <td>0.401600</td>
      <td>NaN</td>
      <td>0.681191</td>
      <td>0.167255</td>
      <td>NaN</td>
      <td>-1.033837</td>
      <td>0.613974</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.615488</td>
    </tr>
  </tbody>
</table>
</div>



注意！！！
需要将处理好的factor_df格式设置为：

factor_df是dataframe格式
index 为日期
columns 是股票名称
将满足如上格式内容的df传入下面的效果模板即可

因子效果检查

在调整了因子数据格式之后，接下来的部分，我们将利用聚宽的因子分析模板，对计算好的因子进行收益分析。

我们将通过如下三个方面进行因子效果检查

1.IC信息比率

2.分组收益

3.换手率

在收益分析中, 分位数的平均收益， 第一分位数的因子值最小， 第五分位数的因子值最大。

分位数收益： 表示持仓5、10、20天后，各分位数可以获得的平均收益。


```python
#使用获取的因子值进行单因子分析
far = analyze_factor(factor=factor_df, start_date=date_list[0], end_date=date_list[-1], weight_method='avg', industry='jq_l1', quantiles=quantiles, periods=periods,max_loss=0.3)
```

IC分析


```python
# 打印信息比率（IC）相关表
far.plot_information_table(group_adjust=False, method='rank')
```

    IC 分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_5</th>
      <th>period_10</th>
      <th>period_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC Mean</th>
      <td>0.015</td>
      <td>0.022</td>
      <td>0.027</td>
    </tr>
    <tr>
      <th>IC Std.</th>
      <td>0.095</td>
      <td>0.104</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>IR</th>
      <td>0.153</td>
      <td>0.213</td>
      <td>0.236</td>
    </tr>
    <tr>
      <th>t-stat(IC)</th>
      <td>4.141</td>
      <td>5.742</td>
      <td>6.372</td>
    </tr>
    <tr>
      <th>p-value(IC)</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>IC Skew</th>
      <td>1.382</td>
      <td>2.254</td>
      <td>3.426</td>
    </tr>
    <tr>
      <th>IC Kurtosis</th>
      <td>6.463</td>
      <td>12.271</td>
      <td>21.551</td>
    </tr>
  </tbody>
</table>
</div>


IC分析：

以上展示了该因子的IC分析数据，通常来讲，我们主要观察的两个指标是IC值与IR值。

从分析结果我们看到，在持仓5天、10天、20天的分组中，持仓20天收益IC均值最大，IR信息比率最高，但是IC值不到0.03，说明该因子效果也有待提升。

分组收益


```python
# 画各分位数平均收益图
far.plot_quantile_returns_bar(by_group=False, demeaned=0, group_adjust=False)
```

<img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_21_1.png'/>

分组收益分析：

从五分组平均收益来看，基本上是有较为明显的分组效果，随着因子值1-5组从小到大的过程，分组跌幅也从大到小的变化。

从不同持仓周期看，持仓20天（绿色柱子）收益递增关系看起来更稳定一些

换手率分析


```python
# 打印换手率表
far.plot_turnover_table()
```

    换手率分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_10</th>
      <th>period_20</th>
      <th>period_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Quantile 1 Mean Turnover</th>
      <td>0.572</td>
      <td>0.791</td>
      <td>0.408</td>
    </tr>
    <tr>
      <th>Quantile 2 Mean Turnover</th>
      <td>0.723</td>
      <td>0.799</td>
      <td>0.631</td>
    </tr>
    <tr>
      <th>Quantile 3 Mean Turnover</th>
      <td>0.734</td>
      <td>0.774</td>
      <td>0.657</td>
    </tr>
    <tr>
      <th>Quantile 4 Mean Turnover</th>
      <td>0.724</td>
      <td>0.801</td>
      <td>0.626</td>
    </tr>
    <tr>
      <th>Quantile 5 Mean Turnover</th>
      <td>0.543</td>
      <td>0.781</td>
      <td>0.377</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_5</th>
      <th>period_10</th>
      <th>period_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean Factor Rank Autocorrelation</th>
      <td>0.695</td>
      <td>0.434</td>
      <td>-0.035</td>
    </tr>
  </tbody>
</table>
</div>


换手率分析：

因子的换手率是在不同的时间周期下， 观察因子个分位中个股的进出情况。 计算方法举例： 某因子第一分位持有的股票数量为30支， 5天后有一只发生变动， 换手率为：

1/30 *100% = 3.33%

对于10日、20日的换手率，在每日都会对比当日1、5分位数的成分股与10日、20日前该分位数的成分股的变化进行计算。

从该分析结果我们看到，5、10、20不同持仓周期，5日换手率最低，20日换手率最高，1-5分组无明显区别

因子效果检查综述

结论：

通过以上IC分析、分组收益检查、换手率分析，我们初步对该示例因子有了一定了解

该因子有一定的收益预测能力，但是不够显著，最好的IC均值为0.027，不到0.03，存在分组效果，换手率在与因子统计周期一致是换手率最高，因子效果有待进一步优化提升

因子分析信息全览¶

具体说明可参考：

https://www.joinquant.com/help/api/help?name=factor#%E5%9B%A0%E5%AD%90%E5%88%86%E6%9E%90%E7%BB%93%E6%9E%9C

在收益分析中, 分位数的平均收益， 各分位数的累积收益， 以及分位数的多空组合收益三方面观察因子的表现。 第一分位数的因子值最小， 第五分位数的因子值最大。

分位数收益： 表示持仓5、10、20天后，各分位数可以获得的平均收益。

分位数的累积收益： 表示各分位数持仓收益的累计值。

多空组合收益： 做多五分位（因子值最大）， 做空一分位（因子值最小）的投资组合的收益。


```python
#调用因子分析方法，进行因子信息全览
far.create_full_tear_sheet(demeaned=False, group_adjust=False, by_group=False, turnover_periods=None, avgretplot=(5, 15), std_bar=False)
```

    分位数统计



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>min</th>
      <th>max</th>
      <th>mean</th>
      <th>std</th>
      <th>count</th>
      <th>count %</th>
    </tr>
    <tr>
      <th>factor_quantile</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>-8.279978</td>
      <td>-0.500532</td>
      <td>-1.176277</td>
      <td>0.583761</td>
      <td>72161</td>
      <td>20.039267</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-0.817781</td>
      <td>-0.098838</td>
      <td>-0.451522</td>
      <td>0.132272</td>
      <td>71930</td>
      <td>19.975118</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-0.457317</td>
      <td>0.318954</td>
      <td>-0.081340</td>
      <td>0.124523</td>
      <td>71936</td>
      <td>19.976784</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.095478</td>
      <td>0.908593</td>
      <td>0.330095</td>
      <td>0.166710</td>
      <td>71930</td>
      <td>19.975118</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.306781</td>
      <td>13.933795</td>
      <td>1.392986</td>
      <td>0.944940</td>
      <td>72141</td>
      <td>20.033713</td>
    </tr>
  </tbody>
</table>
</div>


    
    -------------------------
    
    收益分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_5</th>
      <th>period_10</th>
      <th>period_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Ann. alpha</th>
      <td>0.026</td>
      <td>0.031</td>
      <td>0.031</td>
    </tr>
    <tr>
      <th>beta</th>
      <td>-0.018</td>
      <td>-0.028</td>
      <td>-0.012</td>
    </tr>
    <tr>
      <th>Mean Period Wise Return Top Quantile (bps)</th>
      <td>0.300</td>
      <td>-0.117</td>
      <td>-0.306</td>
    </tr>
    <tr>
      <th>Mean Period Wise Return Bottom Quantile (bps)</th>
      <td>-2.163</td>
      <td>-2.534</td>
      <td>-2.431</td>
    </tr>
    <tr>
      <th>Mean Period Wise Spread (bps)</th>
      <td>2.571</td>
      <td>2.557</td>
      <td>2.167</td>
    </tr>
  </tbody>
</table>
</div>


<div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_5.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_7.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_8.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_9.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_11.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_12.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_13.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_15.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_16.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_17.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_19.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_20.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_21.png'/></div>

    
    -------------------------
    
    IC 分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_5</th>
      <th>period_10</th>
      <th>period_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>IC Mean</th>
      <td>0.015</td>
      <td>0.022</td>
      <td>0.027</td>
    </tr>
    <tr>
      <th>IC Std.</th>
      <td>0.095</td>
      <td>0.104</td>
      <td>0.115</td>
    </tr>
    <tr>
      <th>IR</th>
      <td>0.153</td>
      <td>0.213</td>
      <td>0.236</td>
    </tr>
    <tr>
      <th>t-stat(IC)</th>
      <td>4.141</td>
      <td>5.742</td>
      <td>6.372</td>
    </tr>
    <tr>
      <th>p-value(IC)</th>
      <td>0.000</td>
      <td>0.000</td>
      <td>0.000</td>
    </tr>
    <tr>
      <th>IC Skew</th>
      <td>1.382</td>
      <td>2.254</td>
      <td>3.426</td>
    </tr>
    <tr>
      <th>IC Kurtosis</th>
      <td>6.463</td>
      <td>12.271</td>
      <td>21.551</td>
    </tr>
  </tbody>
</table>
</div>

<div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_25.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_27.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_29.png'/></div>
    
    -------------------------
    
    换手率分析



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_10</th>
      <th>period_20</th>
      <th>period_5</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Quantile 1 Mean Turnover</th>
      <td>0.572</td>
      <td>0.791</td>
      <td>0.408</td>
    </tr>
    <tr>
      <th>Quantile 2 Mean Turnover</th>
      <td>0.723</td>
      <td>0.799</td>
      <td>0.631</td>
    </tr>
    <tr>
      <th>Quantile 3 Mean Turnover</th>
      <td>0.734</td>
      <td>0.774</td>
      <td>0.657</td>
    </tr>
    <tr>
      <th>Quantile 4 Mean Turnover</th>
      <td>0.724</td>
      <td>0.801</td>
      <td>0.626</td>
    </tr>
    <tr>
      <th>Quantile 5 Mean Turnover</th>
      <td>0.543</td>
      <td>0.781</td>
      <td>0.377</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>period_5</th>
      <th>period_10</th>
      <th>period_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Mean Factor Rank Autocorrelation</th>
      <td>0.695</td>
      <td>0.434</td>
      <td>-0.035</td>
    </tr>
  </tbody>
</table>
</div>

<div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_34.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_35.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_36.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_38.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_39.png'/></div><div align="center"><img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_40.png'/></div>


    
    -------------------------
   <img src='http://i.caigoubao.cc/627139/bgpc/20190526/output_29_43.png'/>


错误提醒：
factor_df = pd.concat([far,factor_df],axis=1) 应为 factor_df = pd.concat([factor_df,far],axis=1)






