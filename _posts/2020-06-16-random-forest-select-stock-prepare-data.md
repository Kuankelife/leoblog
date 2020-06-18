---
layout: post
title:  "随机森林模型多因子选股：数据准备部分"
subtitle: ''
date:   2020-06-16
author: "YU"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
tags:
  - 量价分析
  - 量化投资
  - 机器学习
mathjax: False
---

这是应用随机森林的一个金融案例，更多深入研究根据研究会考虑公开。


# 数据准备 


## 1.数据获取

a) 股票池：沪深300 成份股/中证500 成份股/全A 股。剔除ST 股票，剔除每个截
面期下一交易日停牌的股票，剔除上市3 个月内的股票，每只股票视作一个样本。

b) 回测区间：2011-01-31 至 2018-07-31


```python
#导入需要用到的库
import numpy as np
import pandas as pd
from jqlib.technical_analysis import *
from jqfactor import get_factor_values
from jqfactor import neutralize
from jqfactor import winsorize_med
from jqfactor import standardlize
from jqfactor import winsorize
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import time
```


```python
#获取沪深300股票池
stocklist = get_index_stocks('000300.XSHG')
```


```python
#计算超额收益率,返回超额收益列表
def excessreturn(start,end,stocklist):
    returns = []
    #沪深300指数
    df_a = get_price('000300.XSHG', start_date= start, end_date=end,
                 frequency='daily', fields=['close']).close.values
    A = round((df_a[-1] - df_a[0])/df_a[0],4)
    for i in stocklist:
        df_onestock = get_price(i, start_date= start, end_date=end,
                 frequency='daily', fields=['close']).close.values
        onestock = round((df_onestock[-1] - df_onestock[0])/df_onestock[0],4)
        excessreturn = round(onestock - A,4)
        returns.append(excessreturn)
    return returns


def aroon(stocklist,end):
    SG2,XG2 = AROON(stocklist, check_date = end , N = 25)
    a = []
    b = []
    for i,j in SG2.items():
        a.append(i)
        b.append(j)
    aa = pd.DataFrame({'code':a,'AROON_SG2':b})   
    a = []
    b = []
    for i,j in XG2.items():
        a.append(i)
        b.append(j)   
    bb = pd.DataFrame({'code':a,'AROON_XG2':b})  
    cc = pd.merge(aa,bb,left_on  = 'code',right_on = 'code')
    cc.index = cc.code.values
    cc.drop(['code'],axis = 1,inplace = True)
    return cc


def macd(stocklist,end):
    macd_dif, macd_dea, macd_macd = MACD(stocklist,check_date=end, SHORT = 12, LONG = 26, MID = 9)
    a = []
    b = []
    for i,j in macd_dif.items():
        a.append(i)
        b.append(j)
        
    aa = pd.DataFrame({'code':a,'macd_dif':b}) 
    
    a = []
    b = []
    for i,j in macd_dea.items():
        a.append(i)
        b.append(j)
        
    bb = pd.DataFrame({'code':a,'macd_dea':b}) 
    
    a = []
    b = []
    for i,j in macd_macd.items():
        a.append(i)
        b.append(j)
        
    cc = pd.DataFrame({'code':a,'macd':b}) 
    
    dd = pd.merge(aa,bb,left_on  = 'code',right_on = 'code')
    dd = pd.merge(cc,dd,left_on  = 'code',right_on = 'code')
    dd.index = dd.code.values
    dd.drop(['code'],axis = 1,inplace = True)
   
    return dd

def psy(stocklist,end):
    PSY1 = PSY(stocklist,check_date=end, timeperiod=21)
    a = []
    b = []
    for i,j in PSY1.items():
        a.append(i)
        b.append(j)
        
    aa = pd.DataFrame({'code':a,'PSY':b}) 
    aa.index = stocklist
    aa.drop(['code'],axis = 1,inplace = True)
    
    return aa
     

def sgpf(stocklist,end):
    PF2 = SG_PF('000003.XSHG', stocklist, check_date=end)
    a = []
    b = []
    for i,j in PF2 .items():
        a.append(i)
        b.append(j)
        
    aa = pd.DataFrame({'code':a,'sgpf':b}) 
    aa.index = stocklist
    aa.drop(['code'],axis = 1,inplace = True)
    
    return aa


def rsi(stocklist,end):
    RSI1 = RSI(stocklist, check_date=end, N1=21)
    a = []
    b = []
    for i,j in RSI1.items():
        a.append(i)
        b.append(j)
        
    aa = pd.DataFrame({'code':a,'RSI':b}) 
    aa.index = stocklist
    aa.drop(['code'],axis = 1,inplace = True)
    return aa



def bias(stocklist,end):
    BIAS1, BIASMA1 = BIAS_QL(stocklist, check_date = end, N = 21, M = 21)
    a = []
    b = []
    for i,j in BIAS1.items():
        a.append(i)
        b.append(j)
    aa = pd.DataFrame({'code':a,'IAS1':b})   
    a = []
    b = []
    for i,j in BIASMA1.items():
        a.append(i)
        b.append(j)   
    bb = pd.DataFrame({'code':a,'BIASMA1':b})  
    cc = pd.merge(aa,bb,left_on  = 'code',right_on = 'code')
    cc.index = cc.code.values
    cc.drop(['code'],axis = 1,inplace = True)
    return cc


def cyf(stocklist, end):
    CYF1 = CYF(stocklist, check_date = end, N = 21)
    a = []
    b = []
    for i,j in CYF1.items():
        a.append(i)
        b.append(j)
        
    aa = pd.DataFrame({'code':a,'CYF':b}) 
    aa.index = stocklist
    aa.drop(['code'],axis = 1,inplace = True)
    return aa


def brar(stocklist,end):
    BR1,AR1 = BRAR(stocklist, check_date = end, N=26)
    a = []
    b = []
    for i,j in BR1.items():
        a.append(i)
        b.append(j)
    aa = pd.DataFrame({'code':a,'BR1':b})   
    a = []
    b = []
    for i,j in AR1.items():
        a.append(i)
        b.append(j)   
    bb = pd.DataFrame({'code':a,'AR1':b})  
    cc = pd.merge(aa,bb,left_on  = 'code',right_on = 'code')
    cc.index = cc.code.values
    cc.drop(['code'],axis = 1,inplace = True)
    return cc


def joinquant_factor(stocklist,end):
    factor = ['net_profit_ratio','ACCA','quick_ratio','operating_profit_growth_rate','current_ratio',
          'operating_profit_ratio','roa_ttm','roe_ttm','accounts_payable_turnover_rate','gross_income_ratio',
'account_receivable_turnover_rate','equity_turnover_rate','total_profit_to_cost_ratio',
'operating_cost_to_operating_revenue_ratio','VR','BR','TVMA20',
'VDIFF','WVAD','MAWVAD','VDEA','VSTD20','VOL20','DAVOL20','VMACD','AR','VOL60','VOL120',
'turnover_volatility','TVSTD20','ARBR','money_flow_20','VEMA26','VOSC',
'operating_revenue_growth_rate','total_asset_growth_rate','net_operate_cashflow_growth_rate',
'total_profit_growth_rate','net_profit_growth_rate','net_asset_growth_rate','Variance20','Skewness20',
'Kurtosis20','sharpe_ratio_20','Variance60','Skewness60','Kurtosis60','sharpe_ratio_60',
'Variance120','Skewness120','Kurtosis120','net_asset_per_share','eps_ttm']
    factor_data = get_factor_values(securities= stocklist, factors = factor, count = 1 , end_date = end)

    values = factor_data.values()
    a = pd.DataFrame()
    for i in values:
        a = pd.concat([a,i],axis = 0)
    a.index = factor
    data = a.T
    return data


def financial_factor(stocklist,end):
    df = get_fundamentals(query(
        valuation, income,indicator, balance
    ).filter(
        # 这里不能使用 in 操作, 要使用in_()函数
        valuation.code.in_(stocklist)
    ), date=end)

    ddf = df[['code','pe_ratio','pb_ratio','market_cap','ps_ratio','roe','roa','eps',
          'pcf_ratio','net_profit_margin','inc_return','gross_profit_margin',
          "circulating_market_cap",'pe_ratio_lyr','expense_to_total_revenue',
          'operation_profit_to_total_revenue','net_profit_to_total_revenue',
          'operating_expense_to_total_revenue','inc_total_revenue_year_on_year',
          'inc_revenue_year_on_year','inc_net_profit_year_on_year',
         'inc_net_profit_annual','adjusted_profit_to_profit','ocf_to_operating_profit',
         'ocf_to_revenue']]

    ddf.index = ddf.code.values
    ddf.drop(['code'],axis = 1,inplace = True)
    return ddf

def merge_raw_data(stocklist,end):
    data = [joinquant_factor(stocklist,end),macd(stocklist,end),psy(stocklist,end),sgpf(stocklist,end),
            rsi(stocklist,end),bias(stocklist,end),cyf(stocklist, end),brar(stocklist,end),
            financial_factor(stocklist,end)]
    data_first = aroon(stocklist,end)
    for i in data:
        data_first = pd.merge(data_first,i,left_index = True, right_index = True)
    return data_first

def merge_alldata(stocklist,end,next_end):
    raw = merge_raw_data(stocklist,end)
    
    raw = winsorize_med(raw, scale=1, inclusive=True, inf2nan=True, axis=0)
    raw = raw.fillna(raw.mean())
    raw = raw.T
    raw = neutralize(raw, how=['jq_l1', 'market_cap'], date = end, axis= 1)
    raw = standardlize(raw , inf2nan=True, axis = 0).T
    excess = excessreturn(end, next_end, raw.index)
    raw = raw.fillna(0)
    raw['label'] = excess
    
    result5 = raw
    result5_plus = result5[result5['label'] >=result5.label.describe()['75%']]
    result5_minus = result5[result5['label'] <= result5.label.describe()['25%']]
    result5_minus['label'] = [0]*len(result5_minus.label)
    result5_plus['label'] = [1]*len(result5_plus.label)
    frame5 = [result5_plus,result5_minus]
    raw_data = pd.concat(frame5)
    
    return raw_data
    
    
#将过去的训练时间区间的每个月样本合并，构成训练集
#样本内数据获取函数
def data_merge_func(stocklist, month_end, next_month_end):
    data_merge = []
    for i,j in zip(month_end, next_month_end):
        data_merge.append(merge_alldata(stocklist,i,j))
    data_merge = pd.concat(data_merge)
    data_merge.fillna(data_merge.mean(),inplace = True)
    return data_merge
```


```python
#2007-2010 2011
start_all = time.time()

start_time = time.time()
train_time_1 = pd.to_datetime(pd.date_range('2007-01-01', '2010-12-31',freq = 'MS'))
train_next_time_1 = pd.to_datetime(pd.date_range('2007-01-01', '2010-12-31',freq = 'BM'))
train_next_time_1

out_sample_time_1 = pd.to_datetime(pd.date_range('2011-01-01', '2011-12-31',freq = 'MS'))
out_sample_next_time_1 = pd.to_datetime(pd.date_range('2011-01-01', '2011-12-31',freq = 'BM'))
```
