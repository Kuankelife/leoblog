---
layout:     post
title:  "财务数据解释公司市值第一部分：数据获取"
subtitle:   ""
date:       2019-05-21 00:55:00
author:     "YU"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
catalog:    false
tags:
    - 量化投资

---


## 聚宽财务数据用法介绍


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
```

首先要获取财务数据，在聚宽上调用 get_fundamentals, 传入要筛选的条件即可. 聚宽上四类财务数据分别存放到如下四个表中: 市值表, 利润表, 现金表, 负债表。


```python
#获取沪深300成分股
stocks = get_index_stocks('000300.XSHG')
```

### 调用方法格式

---
get_fundamentals(query_object, date=None, statDate=None)

---

date和statDate参数只能传入一个:

- 传入date时, 查询指定日期date所能看到的最近(对市值表来说, 最近一天, 对其他表来说, 最近一个季度)的数据, 我们会查找上市公司在这个日期之前(包括此日期)发布的数据, 不会有未来函数.



- 传入statDate时, 查询statDate指定的季度或者年份的财务数据. 

    注意:
    - 由于公司发布财报不及时, 一般是看不到当季度或年份的财务报表的, 回测中使用这个数据可能会有未来函数, 请注意规避.
    - 由于估值表每天更新, 当按季度或者年份查询时, 返回季度或者年份最后一天的数据
    - 对于年报数据, 我们目前只有现金流表和利润表, 当查询其他表时, 会返回该年份最后一个季报的数据
    - 当date和statDate都不传入时, 相当于使用date参数, date的默认值下面会描述.


```python
### 获取单只股票在某一日期的市值数据
df = get_fundamentals(query(
        valuation
    ).filter(
        valuation.code == '000001.XSHE'
    ), date='2018-10-15')

df
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
      <th>id</th>
      <th>code</th>
      <th>pe_ratio</th>
      <th>turnover_ratio</th>
      <th>pb_ratio</th>
      <th>ps_ratio</th>
      <th>pcf_ratio</th>
      <th>capitalization</th>
      <th>market_cap</th>
      <th>circulating_cap</th>
      <th>circulating_market_cap</th>
      <th>day</th>
      <th>pe_ratio_lyr</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>18856883</td>
      <td>000001.XSHE</td>
      <td>7.2309</td>
      <td>0.8169</td>
      <td>0.8433</td>
      <td>1.5932</td>
      <td>1.7969</td>
      <td>1717041.125</td>
      <td>1735.9286</td>
      <td>1717024.625</td>
      <td>1735.9119</td>
      <td>2018-10-15</td>
      <td>7.486</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 取出总市值
df['market_cap'][0]
```




    1735.9286




```python
# 只选择表里的某些字段
### 获取多只股票在某一日期的市值, 利润, 现金流和负债数据
df = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
    ).filter(
        # 这里不能使用 in 操作, 要使用in_()函数
        valuation.code.in_(['000001.XSHE', '600000.XSHG'])
    ), date='2015-10-15')

df
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
      <th>code</th>
      <th>market_cap</th>
      <th>pe_ratio</th>
      <th>total_operating_revenue</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001.XSHE</td>
      <td>1598.2791</td>
      <td>7.4984</td>
      <td>2.590400e+10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>600000.XSHG</td>
      <td>2965.9021</td>
      <td>6.1440</td>
      <td>3.771000e+10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 选出所有的总市值大于1000亿元, 市盈率小于10, 营业总收入大于200亿元的股票
df = get_fundamentals(query(
        valuation.code, valuation.market_cap, valuation.pe_ratio, income.total_operating_revenue
    ).filter(
        valuation.market_cap > 1000,
        valuation.pe_ratio < 10,
        income.total_operating_revenue > 2e10
    ), date='2015-10-15')

```

---
---

## 财务数据解释公司市值


```python
stocks = get_index_stocks('399906.XSHE')
stocks
# 获取多只股票在某一日期的市值, 利润
df = get_fundamentals(query(
        valuation, income,indicator, balance
    ).filter(
    valuation.code.in_(stocks)
    ), date='2019-05-21')

```

### 给定日期的最新财务数据


```python
#查询日期为2019-05-21，获得是2019-05-20这天能获取的财务数据。
df.head()
#800 rows × 178 columns
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
      <th>id</th>
      <th>code</th>
      <th>pe_ratio</th>
      <th>turnover_ratio</th>
      <th>pb_ratio</th>
      <th>ps_ratio</th>
      <th>pcf_ratio</th>
      <th>capitalization</th>
      <th>market_cap</th>
      <th>circulating_cap</th>
      <th>circulating_market_cap</th>
      <th>day</th>
      <th>pe_ratio_lyr</th>
      <th>id.1</th>
      <th>code.1</th>
      <th>day.1</th>
      <th>pubDate</th>
      <th>statDate</th>
      <th>total_operating_revenue</th>
      <th>operating_revenue</th>
      <th>interest_income</th>
      <th>premiums_earned</th>
      <th>commission_income</th>
      <th>total_operating_cost</th>
      <th>operating_cost</th>
      <th>interest_expense</th>
      <th>commission_expense</th>
      <th>refunded_premiums</th>
      <th>net_pay_insurance_claims</th>
      <th>withdraw_insurance_contract_reserve</th>
      <th>policy_dividend_payout</th>
      <th>reinsurance_cost</th>
      <th>operating_tax_surcharges</th>
      <th>sale_expense</th>
      <th>administration_expense</th>
      <th>financial_expense</th>
      <th>asset_impairment_loss</th>
      <th>fair_value_variable_income</th>
      <th>investment_income</th>
      <th>invest_income_associates</th>
      <th>...</th>
      <th>borrowing_capital</th>
      <th>trading_liability</th>
      <th>notes_payable</th>
      <th>accounts_payable</th>
      <th>advance_peceipts</th>
      <th>sold_buyback_secu_proceeds</th>
      <th>commission_payable</th>
      <th>salaries_payable</th>
      <th>taxs_payable</th>
      <th>interest_payable</th>
      <th>dividend_payable</th>
      <th>other_payable</th>
      <th>reinsurance_payables</th>
      <th>insurance_contract_reserves</th>
      <th>proxy_secu_proceeds</th>
      <th>receivings_from_vicariously_sold_securities</th>
      <th>non_current_liability_in_one_year</th>
      <th>other_current_liability</th>
      <th>total_current_liability</th>
      <th>longterm_loan</th>
      <th>bonds_payable</th>
      <th>longterm_account_payable</th>
      <th>specific_account_payable</th>
      <th>estimate_liability</th>
      <th>deferred_tax_liability</th>
      <th>other_non_current_liability</th>
      <th>total_non_current_liability</th>
      <th>total_liability</th>
      <th>paidin_capital</th>
      <th>capital_reserve_fund</th>
      <th>treasury_stock</th>
      <th>specific_reserves</th>
      <th>surplus_reserve_fund</th>
      <th>ordinary_risk_reserve_fund</th>
      <th>retained_profit</th>
      <th>foreign_currency_report_conv_diff</th>
      <th>equities_parent_company_owners</th>
      <th>minority_interests</th>
      <th>total_owner_equities</th>
      <th>total_sheet_owner_equities</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>35356284</td>
      <td>000001.XSHE</td>
      <td>8.2812</td>
      <td>0.4580</td>
      <td>0.9354</td>
      <td>1.7544</td>
      <td>21.0987</td>
      <td>1.717041e+06</td>
      <td>2125.6970</td>
      <td>1.717025e+06</td>
      <td>2125.6765</td>
      <td>2019-05-20</td>
      <td>8.5651</td>
      <td>17903449</td>
      <td>000001.XSHE</td>
      <td>2019-05-20</td>
      <td>2019-04-24</td>
      <td>2019-03-31</td>
      <td>3.247600e+10</td>
      <td>3.247600e+10</td>
      <td>4.266100e+10</td>
      <td>NaN</td>
      <td>1.157900e+10</td>
      <td>9.924000e+09</td>
      <td>NaN</td>
      <td>2.188700e+10</td>
      <td>2.019000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.040000e+08</td>
      <td>NaN</td>
      <td>9.620000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>-642000000.0</td>
      <td>2.571000e+09</td>
      <td>NaN</td>
      <td>...</td>
      <td>2.166900e+10</td>
      <td>1.572100e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.150200e+10</td>
      <td>NaN</td>
      <td>9.848001e+09</td>
      <td>1.105300e+10</td>
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
      <td>4.017580e+11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.271000e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.279242e+12</td>
      <td>1.717000e+10</td>
      <td>5.646500e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.078100e+10</td>
      <td>3.985000e+10</td>
      <td>1.016090e+11</td>
      <td>NaN</td>
      <td>2.509380e+11</td>
      <td>NaN</td>
      <td>2.509380e+11</td>
      <td>3.530180e+12</td>
    </tr>
    <tr>
      <th>1</th>
      <td>35356285</td>
      <td>000002.XSHE</td>
      <td>9.0621</td>
      <td>0.2635</td>
      <td>1.9555</td>
      <td>0.9774</td>
      <td>6.4286</td>
      <td>1.130214e+06</td>
      <td>3080.9641</td>
      <td>9.715170e+05</td>
      <td>2648.3555</td>
      <td>2019-05-20</td>
      <td>9.1227</td>
      <td>21894414</td>
      <td>000002.XSHE</td>
      <td>2019-05-20</td>
      <td>2019-04-30</td>
      <td>2019-03-31</td>
      <td>4.837463e+10</td>
      <td>4.837463e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.225158e+10</td>
      <td>3.142224e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.437556e+09</td>
      <td>1.371392e+09</td>
      <td>2.639014e+09</td>
      <td>1.841741e+09</td>
      <td>NaN</td>
      <td>-35196480.0</td>
      <td>-2.065606e+08</td>
      <td>-312437184.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.313821e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.440450e+09</td>
      <td>1.433424e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.314350e+11</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.156035e+10</td>
      <td>4.635405e+10</td>
      <td>1.115029e+12</td>
      <td>1.261480e+11</td>
      <td>5.261106e+10</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.361696e+08</td>
      <td>4.728514e+08</td>
      <td>2.188339e+09</td>
      <td>1.999485e+11</td>
      <td>1.314977e+12</td>
      <td>1.103915e+10</td>
      <td>7.961989e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.739325e+10</td>
      <td>NaN</td>
      <td>9.284547e+10</td>
      <td>NaN</td>
      <td>1.575514e+11</td>
      <td>7.863797e+10</td>
      <td>2.361894e+11</td>
      <td>1.551166e+12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>35356288</td>
      <td>000006.XSHE</td>
      <td>10.7008</td>
      <td>0.3713</td>
      <td>1.1855</td>
      <td>3.0648</td>
      <td>3.1466</td>
      <td>1.349995e+05</td>
      <td>74.7897</td>
      <td>1.347786e+05</td>
      <td>74.6674</td>
      <td>2019-05-20</td>
      <td>8.5548</td>
      <td>21977780</td>
      <td>000006.XSHE</td>
      <td>2019-05-20</td>
      <td>2019-04-30</td>
      <td>2019-03-31</td>
      <td>4.134936e+08</td>
      <td>4.134936e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.444878e+08</td>
      <td>2.511230e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.051002e+07</td>
      <td>2.214487e+06</td>
      <td>2.797155e+07</td>
      <td>2.901531e+07</td>
      <td>13653447.0</td>
      <td>NaN</td>
      <td>6.441681e+07</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.288545e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.705216e+07</td>
      <td>6.717919e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.829181e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.374894e+08</td>
      <td>3.219049e+05</td>
      <td>5.578220e+09</td>
      <td>1.310600e+09</td>
      <td>4.993531e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.715713e+06</td>
      <td>NaN</td>
      <td>1.811669e+09</td>
      <td>7.389889e+09</td>
      <td>1.349995e+09</td>
      <td>4.843216e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.483863e+09</td>
      <td>NaN</td>
      <td>2.990372e+09</td>
      <td>NaN</td>
      <td>6.308551e+09</td>
      <td>3.212726e+08</td>
      <td>6.629824e+09</td>
      <td>1.401971e+10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35356290</td>
      <td>000008.XSHE</td>
      <td>32.6433</td>
      <td>0.8227</td>
      <td>1.4540</td>
      <td>3.9970</td>
      <td>-362.7030</td>
      <td>2.780795e+05</td>
      <td>106.5045</td>
      <td>2.599124e+05</td>
      <td>99.5465</td>
      <td>2019-05-20</td>
      <td>32.7654</td>
      <td>22710880</td>
      <td>000008.XSHE</td>
      <td>2019-05-20</td>
      <td>2019-04-27</td>
      <td>2019-03-31</td>
      <td>3.087470e+08</td>
      <td>3.087470e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.039850e+08</td>
      <td>1.448895e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7.327400e+06</td>
      <td>5.257320e+07</td>
      <td>7.052305e+07</td>
      <td>8.071832e+06</td>
      <td>NaN</td>
      <td>-5482905.0</td>
      <td>-4.299728e+05</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.919174e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.542671e+07</td>
      <td>3.612078e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3.779759e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.849081e+07</td>
      <td>2.813214e+09</td>
      <td>2.344870e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.983337e+07</td>
      <td>NaN</td>
      <td>2.830551e+08</td>
      <td>3.096269e+09</td>
      <td>2.780795e+09</td>
      <td>2.771978e+09</td>
      <td>NaN</td>
      <td>44309900.0</td>
      <td>4.685366e+07</td>
      <td>NaN</td>
      <td>1.681122e+09</td>
      <td>NaN</td>
      <td>7.325059e+09</td>
      <td>9.265987e+07</td>
      <td>7.417719e+09</td>
      <td>1.051399e+10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>35356291</td>
      <td>000009.XSHE</td>
      <td>49.0255</td>
      <td>0.7387</td>
      <td>2.2674</td>
      <td>1.0008</td>
      <td>9.1161</td>
      <td>2.149345e+05</td>
      <td>119.9334</td>
      <td>2.119789e+05</td>
      <td>118.2842</td>
      <td>2019-05-20</td>
      <td>56.1012</td>
      <td>23284630</td>
      <td>000009.XSHE</td>
      <td>2019-05-20</td>
      <td>2019-04-30</td>
      <td>2019-03-31</td>
      <td>2.249621e+09</td>
      <td>2.239609e+09</td>
      <td>1.001234e+07</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.117828e+09</td>
      <td>1.438458e+09</td>
      <td>NaN</td>
      <td>2.091037e+06</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4.292985e+07</td>
      <td>2.564405e+08</td>
      <td>1.624112e+08</td>
      <td>1.306775e+08</td>
      <td>685752.0</td>
      <td>162383584.0</td>
      <td>-6.207141e+07</td>
      <td>-6610219.5</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.233106e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.041870e+08</td>
      <td>7.710143e+08</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.438105e+09</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>5.883695e+08</td>
      <td>9.610228e+04</td>
      <td>1.208985e+10</td>
      <td>1.788000e+09</td>
      <td>4.575568e+09</td>
      <td>633191872.0</td>
      <td>NaN</td>
      <td>1.214505e+07</td>
      <td>1.124689e+08</td>
      <td>8.833930e+07</td>
      <td>7.618770e+09</td>
      <td>1.970862e+10</td>
      <td>2.149345e+09</td>
      <td>8.246493e+08</td>
      <td>NaN</td>
      <td>7931196.5</td>
      <td>2.600634e+08</td>
      <td>NaN</td>
      <td>2.054833e+09</td>
      <td>NaN</td>
      <td>5.289418e+09</td>
      <td>5.014061e+09</td>
      <td>1.030348e+10</td>
      <td>3.001209e+10</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 加载资产负债表的股票代码、总市值，资产、负债、股东权益、净利润、研发费用数据
#df[['code','market_cap','total_assets','total_liability','net_profit','development_expenditure']]
mydata = df[['code','market_cap','total_assets','total_liability','total_owner_equities','net_profit','development_expenditure']]
mydata.head()
#800 rows × 7 columns
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
      <th>code</th>
      <th>market_cap</th>
      <th>total_assets</th>
      <th>total_liability</th>
      <th>total_owner_equities</th>
      <th>net_profit</th>
      <th>development_expenditure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>000001.XSHE</td>
      <td>2125.6970</td>
      <td>3.530180e+12</td>
      <td>3.279242e+12</td>
      <td>2.509380e+11</td>
      <td>7.446000e+09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000002.XSHE</td>
      <td>3080.9641</td>
      <td>1.551166e+12</td>
      <td>1.314977e+12</td>
      <td>2.361894e+11</td>
      <td>3.205934e+09</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000006.XSHE</td>
      <td>74.7897</td>
      <td>1.401971e+10</td>
      <td>7.389889e+09</td>
      <td>6.629824e+09</td>
      <td>1.156789e+08</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>000008.XSHE</td>
      <td>106.5045</td>
      <td>1.051399e+10</td>
      <td>3.096269e+09</td>
      <td>7.417719e+09</td>
      <td>1.063877e+07</td>
      <td>14590518.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>000009.XSHE</td>
      <td>119.9334</td>
      <td>3.001209e+10</td>
      <td>1.970862e+10</td>
      <td>1.030348e+10</td>
      <td>1.859285e+08</td>
      <td>26316454.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
mydata2 = pd.DataFrame()
```


```python
#对数市值
mydata2['market_cap'] = mydata['market_cap'].apply(np.log)
```


```python
#对数净资产,净资产即所有者权益，在聚宽上称股东权益
mydata2['net_assets'] = (mydata['total_assets']-mydata['total_liability']).apply(np.log)
```


```python
#财务杠杆，负债/资产
mydata2['financial leverage'] = (mydata['total_liability']/mydata['total_assets'])
```


```python
#对数净利润
mydata2['net_profit'] = mydata['net_profit'].apply(np.log)
```


```python
#对数研发费用
mydata2['development_expenditure'] = mydata['development_expenditure'].apply(np.log).replace([np.nan,np.inf, -np.inf],0.0)
```


```python
mydata2 = mydata2.dropna()
mydata2.head()
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
      <th>market_cap</th>
      <th>net_assets</th>
      <th>financial leverage</th>
      <th>net_profit</th>
      <th>development_expenditure</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.661855</td>
      <td>26.248472</td>
      <td>0.928916</td>
      <td>22.730943</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.032998</td>
      <td>26.187900</td>
      <td>0.847734</td>
      <td>21.888269</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.314680</td>
      <td>22.614844</td>
      <td>0.527107</td>
      <td>18.566328</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.668187</td>
      <td>22.727137</td>
      <td>0.294490</td>
      <td>16.180015</td>
      <td>16.495882</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.786937</td>
      <td>23.055747</td>
      <td>0.656689</td>
      <td>19.040873</td>
      <td>17.085705</td>
    </tr>
  </tbody>
</table>
</div>



## <font color=purple> 市值解释模型 </font>

用线性模型与随机森林模型来解释市值

1) 线性市值解释模型
- 模型形式为:

$$ m_{it} = \alpha_{0t} IND_{it} + \alpha_{1t} b_{it} + \alpha_{2t} \ln NI_{it}^{+} + \alpha_{3t} I_{\lt 0} \ln NI_{it}^{-} + \alpha_{4t} LEV_{it} + \alpha_{5t} g_{it} + \alpha_{6t} RD_{it} + \epsilon_{it} $$

- 其中, $m_{it}$为股票i在t时候的对数市值，$IND_{it}$为行业哑变量，$b_{it}$为股票对数净资产，$NI$为公司净利润，这里根据净利润的正负拆分了两个变量，$LEV_{it}$为公司财务杠杆，即负债除以资产, $g_{it}$为季度同比的营业收入增长率， $RD_{it}$为开发支出。

2) 随机森林非线性市值解释模型
- 将线性模型的自变量作为随机森林模型的输入特征，对数市值作为数据的标签，进行训练。


下面用的数据未使用$IND_{it}$行业哑变量和$g_{it}$

### OLS线性回归和随机森林回归


```python
import statsmodels.api as sm
import scipy.stats as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score


#对数市值
y = mydata2['market_cap']
#输入标签
x = mydata2[['net_assets','financial leverage','net_profit','development_expenditure']]
```


```python
#OLS线性回归
model_1 = sm.OLS(y, x, missing='drop').fit()
predict_market_1 = model_1.predict(x)
model_1_resid = model_1.resid
model_1_r2 = r2_score(y, predict_market_1)
print(model_1_r2)
```

    0.5127315745827835



```python
#随机森林回归
model_2 = RandomForestRegressor()
model_2.fit(x,y)
predict_market_2 = model_2.predict(x)

model_2_resid = y - predict_market_2
model_2_r2 = r2_score(y, predict_market_2)
print(model_2_r2)
```

    0.94671393031602



```python
print('线性模型的拟合R2:%.2f'%model_1_r2)
print('随机森林模型的拟合R2:%.2f'%model_2_r2)
```

    线性模型的拟合R2:0.51
    随机森林模型的拟合R2:0.95


### 残差


```python
plt.figure(figsize = (15,5))
kwargs = dict(histtype='stepfilled', alpha=0.3, density=True, bins=40)
plt.hist(model_1_resid,label = 'OLS回归',**kwargs)
plt.hist(model_2_resid,label = '随机森林回归',**kwargs)
plt.title('2019年5月20日一期残差分布')
plt.legend()
```
<img src='http://i.caigoubao.cc/627139/bgpc/下载.png'/>

