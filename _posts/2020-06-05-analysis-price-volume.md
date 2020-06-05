---
layout: post
title:  "量价分析面向对象框架"
subtitle: ''
date:   2020-06-06
author: "YU"

header-img: "img/yu-img/post-img/post_head/keyibukeyi1.jpg"
header-mask: 0.4
tags:
  - 量价分析
  - 量化投资
  - 机器学习
mathjax: False
---

<embed frameborder="no" border="0" marginwidth="0" marginheight="0" width="340" height="86" src="//music.163.com/outchain/player?type=3&id=1423104917&auto=1&height=66">




```python
#导入常规数据处理库
#from jqdata import *
import talib
import numpy as  np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import seaborn as sns
import copy
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['font.sans-serif']=['SimHei'] #指定默认字体 SimHei为黑体
import warnings
warnings.filterwarnings("ignore")
import datetime
import time

```


```python
#特定基准的特定区间的收益率
def cal_benchmark_return(benchmark,start_date,end_date):
    First_date = get_trade_days(start_date = start_date, end_date=end_date, count=None)[0]
    tradingday = list(get_all_trade_days())
    shiftday_index = tradingday.index(First_date)-1
    pre_date = tradingday[shiftday_index]
    data = get_price(benchmark, start_date=pre_date, end_date=end_date,\
                     fields=['open','close'],frequency='daily').close
    return data/data.shift()-1

class Report(object):
    
    def __init__(self,benchmark):
        
        #初始化
        self.benchmark = benchmark
        #全A股的历史量价数据，收益，开盘价，收盘价，最高价，最低价，换手率，成交股票数，成交金额
        self.data_return = pd.read_csv('A_share_return.csv',index_col=0)
        self.data_close = pd.read_csv('A_share_close.csv',index_col=0)
        self.data_open = pd.read_csv('A_share_open.csv',index_col=0)
        self.data_high = pd.read_csv('A_share_high.csv',index_col=0)
        self.data_low = pd.read_csv('A_share_low.csv',index_col=0)
        self.data_money = pd.read_csv('A_share_money.csv',index_col=0)
        self.data_turnover = pd.read_csv('A_share_turnover_ratio.csv',index_col=0)
        self.data_volume = pd.read_csv('A_share_volume.csv',index_col=0)
        #定义基准在输入区间的收益率
        self.benchmark_return = cal_benchmark_return(benchmark,self.data_return.index[1],self.data_return.index[-1])
        
        
    #下面是为了实现特定功能的功能函数

    #获取特定周期的时间点列表
    def get_time_inverval(self,start_date,end_date,freq):
        if freq == 'M':
            interval_start = pd.to_datetime(pd.date_range(start_date,end_date,freq = 'MS'))
            interval_end = pd.to_datetime(pd.date_range(start_date,end_date,freq = 'BM'))
        elif freq == 'Y':
            interval_start = pd.to_datetime(pd.date_range(start_date,end_date,freq = 'AS'))
            interval_end = pd.to_datetime(pd.date_range(start_date,end_date,freq = 'A'))

        if len(interval_start) > len(interval_end) > 0:
            interval_start = interval_start[:-1]
        elif len(interval_end)  == 0:
            print('请输入完整的周期区间')
        return interval_start,interval_end
    
    #转datestampe格式为datetime格式
    def datetime_date(self,datestamp):
        return datetime.date(datestamp.year, datestamp.month, datestamp.day)
    
    #股票所属行业
    def get_stock_belong_industry(self,stocklist,date):
        industry = {}
        industry_all =  get_industry(security = stocklist, date=date)
        for i in stocklist:
            try:
                industry[i] = industry_all[i]['sw_l1']['industry_name']
            except:
                continue
        return pd.DataFrame([industry],index = ['industry']).T
    
    #过滤新股
    def filter_new_stock(self,stock_list,date):
        tmpList = []
        for stock in stock_list :
            days_public=(datetime_date(date) - get_security_info(stock).start_date).days
            if days_public >= 180:
                tmpList.append(stock)
        return tmpList


    #剔除ST股
    def delete_st(self,stocks,begin_date):
        st_data=get_extras('is_st',stocks, count = 1,end_date=begin_date)
        stockList = [stock for stock in stocks if not st_data[stock][0]]
        return stockList
    
    ###提取因子部分
    def get_MA(self,df_close,n):
        '''
        移动平均值是在一定范围内的价格平均值
        df_close:量价数据，dataframe或者series
        n: 回溯的天数，整数型
        '''
        ma = df_close.rolling(n).mean()
        ma = pd.DataFrame({'MA_' + str(n): ma}, index = ma.index)
        return ma

    ## 计算变化率ROC
    def get_ROC(self,df_close, n):
        '''
        ROC=(今天的收盘价-N日前的收盘价)/N日前的收盘价*100
        移动平均值是在一定范围内的价格平均值
        df_close:量价收盘数据，dataframe或者series
        n: 回溯的天数，整数型
        '''
        M = df_close
        N = df_close.shift(n)
        roc = pd.DataFrame({'ROC_' + str(n): (M-N) / N*100}, index = M.index)
        return roc

    ## 计算RSI
    def get_RSI(self,df_close,n):
        '''
        df_close:量价收盘数据，dataframe或者series
        n: 回溯的天数，整数型
        '''
        rsi = talib.RSI(df_close, timeperiod=n)
        return pd.DataFrame({'RSI_' + str(n): rsi}, index = df_close.index)

    ##计算OBV指标
    def get_OBV(self,df_close,df_volume):
        '''
        On Balance Volume 能量，通过统计成交量变动的趋势推测股价趋势
        df_close:量价收盘数据，dataframe或者series
        df_volume:J量价交易数数据，dataframe或者series
        '''
        obv = talib.OBV(df_close,df_volume)
        return pd.DataFrame({'OBV': obv}, index = df_close.index)

    #真实波幅
    def get_ATR(self,df_high,df_low,df_close,n):
        '''
        平均真实波幅，主要用来衡量价格的波动
        df_close:量价收盘数据，dataframe或者series
        df_high:量价最高价数据，dataframe或者series
        df_low:量价最低价数据，dataframe或者series
        n: 回溯的天数，整数型
        '''
        atr = talib.ATR(df_high,df_low,df_close, timeperiod=n)
        return pd.DataFrame({'ATR_' + str(n): atr}, index = df_close.index)
    #上升动向值
    def get_MOM(self,df_close,n):
        '''
        上升动向值,投资学中意思为续航，指股票(或经济指数)持续增长的能力。研究发现，赢家组合在牛市中存在着正的动量效应，输家组合在熊市中存在着负的动量效应。
        df_close:量价收盘数据，dataframe或者series
        n: 回溯的天数，整数型
        '''
        mom = talib.MOM(df_close, timeperiod=n)
        return pd.DataFrame({'MOM_' + str(n): mom}, index = df_close.index)
    
    #阿隆指标
    def get_AROON(self,df_high,df_low,n):
        aroondown, aroonup = talib.AROON(df_high,df_low, timeperiod=n)
        return pd.DataFrame({'Aroondown_' + str(n): aroondown}, index = df_high.index),pd.DataFrame({'Aroonup_' + str(n): aroonup}, index = df_high.index)
    
    
    ###因子处理合成部分
    
    #合并新特征的函数，是添加新指标或者新特征的重要函数
    def merge_raw_factors(self,df_close,df_open,df_high,df_low,df_volume,df_money):
        return pd.concat([self.get_MA(df_close,5),self.get_MA(df_close,60),self.get_AROON(df_high,df_low,14)[0],\
                          self.get_AROON(df_high,df_low,14)[1],self.get_ROC(df_close, 6),self.get_ROC(df_close, 12),self.get_RSI(df_close,6),\
        self.get_RSI(df_close,24),self.get_OBV(df_close,df_volume),self.get_ATR(df_high,df_low,df_close,14),self.get_MOM(df_close,10)],axis = 1)

    #数据去极值及标准化
    def winsorize_and_standarlize(self,data,qrange=[0.05,0.95],axis=0):
        '''
        input:
        data:Dataframe or series,输入数据
        qrange:list,list[0]下分位数，list[1]，上分位数，极值用分位数代替
        '''
        if isinstance(data,pd.DataFrame):
            if axis == 0:
                q_down = data.quantile(qrange[0])
                q_up = data.quantile(qrange[1])
                index = data.index
                col = data.columns
                for n in col:
                    data[n][data[n] > q_up[n]] = q_up[n]
                    data[n][data[n] < q_down[n]] = q_down[n]
                data = (data - data.mean())/data.std()
                data = data.fillna(0)
            else:
                data = data.stack()
                data = data.unstack(0)
                q = data.quantile(qrange)
                index = data.index
                col = data.columns
                for n in col:
                    data[n][data[n] > q[n]] = q[n]
                data = (data - data.mean())/data.std()
                data = data.stack().unstack(0)
                data = data.fillna(0)

        elif isinstance(data,pd.Series):
            name = data.name
            q = data.quantile(qrange)
            data[data>q] = q
            data = (data - data.mean())/data.std()
        return data
    
    
    
    #可取一个标的的一日或者多日原始因子数据
    def merge_raw_factor_perdate_percode(self,stock_code,start_date,end_date):
        #获取指定区间的数据来计算因子
        date_interval = get_trade_days(start_date =start_date, end_date=end_date, count=None)
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        
        if set(date_interval) <=  set(self.data_return.index) and len(self.data_return.loc[date_interval]) != 0:
            df_return = self.data_return[stock_code]
            df_close = self.data_close[stock_code]
            df_open = self.data_open[stock_code]
            df_high = self.data_high[stock_code]
            df_low = self.data_low[stock_code]
            df_money = self.data_money[stock_code]
            if stock_code != '000001.XSHG':
                df_turnover = self.data_turnover[stock_code]
            df_volume = self.data_volume[stock_code]
            merge_raw_data = self.merge_raw_factors(df_close,df_open,df_high,df_low,df_volume,df_money)
                
        else:
            print('你输入的时间区间超过了本地数据的时间区间或不含有交易日，请输入2005年1月5日到2020年2月28日内的起始时间')
        return merge_raw_data
    
    #可取一个标的的一日或者多日规整化因子数据
    def merge_regularfactor_multidate_percode(self,stock_code,start_date,end_date):
        #获取指定区间的数据来计算因子
        date_interval = get_trade_days(start_date =start_date, end_date=end_date, count=None)
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        merge_raw_data = self.merge_raw_factor_perdate_percode(stock_code,start_date,end_date)
        data_pro = self.winsorize_and_standarlize(merge_raw_data.loc[date_interval])
        return data_pro
       

        
    #可取多个标的的一天的原始因子数据
    def merge_rawfactor_multicode_perday(self,stock_list,start_date,end_date):
        #获取指定区间的数据来计算因子
        date_interval = get_trade_days(start_date =start_date, end_date=end_date, count=None)
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        factor_list = []
        for stock_code in stock_list:
            factor_list.append(self.merge_raw_factor_perdate_percode(stock_code,start_date,end_date).loc[date_interval])
        return pd.concat(factor_list)
    
    #可取多个标的的一天的规整化因子数据
    def merge_regularfactor_multicode_perday(self,stock_list,start_date,end_date):
        
        #获取指定区间的数据来计算因子
        date_interval = get_trade_days(start_date =start_date, end_date=end_date, count=None)
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        data_raw = self.merge_rawfactor_multicode_perday(stock_list,start_date,end_date)
        data_regular = self.winsorize_and_standarlize(data_raw)
        data_regular.insert(0, 'code',stock_list)
        
        return data_regular
    
    
    #贴标签方法1，贴标签方法可以自定义
    def add_label_1(self,stock_code,start_date,end_date):
        
        profit = self.data_return[stock_code]
        profit[profit > 0] = 1
        profit[profit< 0] = 0
        profit = pd.DataFrame(profit)
        profit.columns = ['Label']
        
        return profit
    
    
    #一个标的的多天的规整化因子数据+标签（方法1）
    def merge_final_factor_label1(self,stock_code,start_date,end_date):
        
        #获取指定区间的数据来计算因子
        date_interval = get_trade_days(start_date =start_date, end_date=end_date, count=None)
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        
        if set(date_interval) <=  set(self.data_return.index) and len(self.data_return.loc[date_interval]) != 0:
            data_pro = self.merge_regularfactor_multidate_percode(stock_code,start_date,end_date)
            profit = self.add_label_1(stock_code,start_date,end_date)
            data_final = pd.concat([profit.shift(),data_pro],axis = 1).loc[date_interval]
        else:
            print('你输入的时间区间超过了本地数据的时间区间或不含有交易日，请输入2005年1月5日到2020年2月28日内的起始时间')
        return data_final
    
    
    
    # 指数择时模型，参数：指数代码，开始时间，结束时间，机器学习模型（需要添加），训练好的模型的本地保存文件名
    def timing_model(self,code,start_date,end_date,out_start,out_end,model_name,file_name):
        print('开始获取合成特征和标签数据框...')
        #获取数据
        data_index = self.merge_final_factor_label1(code,start_date,end_date)
        # 特征数据和标签数据
        x_data,y_data = data_index.iloc[:,1:], data_index.iloc[:,0]
        print ('-' * 60)
        print('按照比例分割为训练集和测试集...')
        '''
        原始数据按照比例分割为“训练集”和“测试集”
        

        x_data：所要划分的样本特征集

        y_data：所要划分的样本标签

        test_size：样本占比，如果是整数的话就是样本的数量

        random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。
        '''
        x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)
        print ('-' * 60)

        # 创建预测数据框
        pred = pd.DataFrame(index=y_test.index)
        pred["Actual"] = y_test

        #可往里面添加算法
        if model_name == 'LR':
            #构建分类器，里面的默认参数可修改
            model = LogisticRegression(solver='liblinear')
        elif model_name == "LDA":
            model=QDA()
        else:
            print('不支持这个算法，请新添加这个算法')

        print('开始训练数据...')
        #训练数据
        model.fit(x_train, y_train)
        print('\n')
        print('训练结束')
        print ('-' * 60)
        print ("预测准确率:")

        pred[model_name] = model.predict(x_test)

        # 预测准确率
        score=accuracy_score(pred['Actual'], pred[model_name])
        print("%s模型: %.3f" % (model_name, score)) 
        print ('-' * 60)

        # 构建混淆矩阵
        cm = pd.crosstab(pred['Actual'], pred[model_name])
        print('输出混淆矩阵...')

        print(cm)
        # 绘制混淆矩阵图
        sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')
        print ('-' * 60)


        print('绘制曲线...')
        # 计算正例的预测概率，而非实际的预测值，用于生成ROC曲线的数据
        y_score = model.predict_proba(x_test)[:,1]
        #fpr表示1-Specificity,tpr表示Sensitivity
        fpr,tpr,threshold = roc_curve(y_test, y_score)
        # 计算AUC的值
        roc_auc = metrics.auc(fpr,tpr)
        # 绘制面积图
        plt.figure(figsize=(8,6))
        plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
        plt.plot(fpr, tpr, color='black', lw = 1)
        # 添加对角线
        plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
        # 添加文本信息
        plt.text(0.5,0.3,'ROC曲线 (area = %0.2f)' % roc_auc)
        # 添加x轴与y轴标签
        plt.title('模型预测指数涨跌的ROC曲线',size=15)
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.show()
        print ('-' * 60)


        print('输出评估报告...')
        print('\n')

        print('模型的评估报告：\n',classification_report(pred['Actual'], pred[model_name]))
        print ('-' * 60)


        print('保存模型到本地...')
        joblib.dump(model,file_name+'.pkl')
        print('\n')
        print('加载本地训练好的模型...')
        model2 = joblib.load(file_name+'.pkl')
        print('加载完毕')
        print ('-' * 60)

        
        # 样本外预测结果
        print('样本外测试结果')
        #获取样本外数据, out_start,out_end 样本外开始时间，结束时间
        data_out_sample = self.merge_final_factor_label1(index_code,out_start,out_end)

        #划分特征集和标签集
        x_out_test,y_out_test = data_out_sample.iloc[:,1:], data_out_sample.iloc[:,0]
        y_out_pred = model2.predict(x_out_test)
        
        predict_proba = model2.predict_proba(x_out_test)[:,1] #此处test_X为特征集
        
        # 样本外准确率
        accuracy_out_sample = accuracy_score(y_out_test, y_out_pred)
        print('样本外准确率',accuracy_out_sample)
        # 样本外AUC值
        roc_out_sample = roc_auc_score(y_out_test, y_out_pred)
        print('样本外AUC值',roc_out_sample)

        #返回加载好的模型
        return model2

    #贴标签方法2，贴标签方法可以自定义.data为特征数据框，带有列名为return的收益序列，前30%记为1，后30%记为0，其余数据去掉
    def add_label_2(self,data):
        percent_select = [0.3,0.3]
        #做标签
        data['Label'] = np.nan
        #根据收益排序
        data = data.sort_values(by='return',ascending=False)
        #选一定比例的样本
        n_stock = data.shape[0]
        n_stock_select = np.multiply(percent_select,n_stock)
        n_stock_select = np.around(n_stock_select).astype(int)
        #给选中的样本打上标签1 or 0
        data.iloc[0:n_stock_select[0],-1] = 1
        data.iloc[-n_stock_select[1]:,-1] = 0
        #去掉其他没选上的股票
        data = data.dropna(axis=0)
        del data['return']
        return data

 
    #合成用贴标签方法2 的一期数据框，带有特征和标签
    def data_for_model_perperiod(self,start_date,end_date,index_code):
        #在相应区间还在上市的股票
        stock_list = get_index_stocks(index_code, date=end_date)
        stock_list_notpause = list(set(stock_list) & set(get_all_securities(date=end_date).index) & set(self.data_return.columns))
        
        #去掉上市时间不到六个月的公司
        #     stock_list_fillter = filter_new_stock(stock_list_notpause,date_interval[1])
        
        #去除st，*st股
        #stock_list= delete_st(stock_list_fillter, date1)
        date_interval = get_trade_days(start_date = start_date, end_date=end_date, count=None)
        data_per_period = self.merge_regularfactor_multicode_perday(stock_list_notpause,date_interval[0],date_interval[0])
        
        
        date_interval = [date.strftime('%Y-%m-%d') for date in date_interval]
        data_return_interval = self.data_return.loc[date_interval][stock_list_notpause]
        #统计相应区间固定周期收益率
        profit = data_return_interval.apply(lambda x:(1 + x).cumprod() - 1).iloc[-1]
            
        data_per_period = data_per_period.set_index('code')
        stock_profit = pd.DataFrame(profit,columns = ['return'])

        data_merge_label = self.add_label_2(pd.concat([data_per_period,stock_profit],axis = 1))
        return data_merge_label
    
    #合成用贴标签方法2 的多期数据框，带有特征和标签
    def data_for_model_multiperiod(self,start_date,end_date,index_code):
        interval_start,interval_end = self.get_time_inverval(start_date,end_date,'M')
        factor_list = []
        for date1,date2 in dict(zip(interval_start,interval_end)).items():

            data_merge_label = self.data_for_model_perperiod(date1,date2,index_code)
            factor_list.append(data_merge_label)
        return pd.concat(factor_list,axis = 0)
    
    #多因子模型
 
    def multifactor_model(self,index_code,start_date,end_date,out_start,out_end,model_name,file_name):
        data_regular = self.data_for_model_multiperiod(start_date,end_date,index_code)

        y_data = data_regular['Label']
        x_data = data_regular.iloc[:,:-1]

        x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)
        #可往里面添加算法
        if model_name == 'LR':
            #构建分类器，里面的默认参数可修改
            model = LogisticRegression(solver='liblinear')
        elif model_name == "LDA":
            model=QDA()
        elif model_name == 'xgboost':
            model = XGBClassifier(max_depth=3,subsample=0.9,random_state=0)
        else:
            print('不支持这个算法，请新添加这个算法')




        print('开始训练数据...')
        #训练数据
        model.fit(x_train, y_train)
        print('\n')
        print('训练结束')
        print ('-' * 60)
        print ("预测准确率:")

        pred = model.predict(x_test)

        # 预测准确率
        score=accuracy_score(y_test, pred )
        print("%s模型: %.3f" % (model_name, score)) 
        print ('-' * 60)

        # 构建混淆矩阵
        cm = pd.crosstab(y_test, pred )
        print('输出混淆矩阵...')

        print(cm)
        # 绘制混淆矩阵图
        sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')
        print ('-' * 60)


        print('绘制曲线...')
        # 计算正例的预测概率，而非实际的预测值，用于生成ROC曲线的数据
        y_score = model.predict_proba(x_test)[:,1]
        #fpr表示1-Specificity,tpr表示Sensitivity
        fpr,tpr,threshold = roc_curve(y_test, y_score)
        # 计算AUC的值
        roc_auc = metrics.auc(fpr,tpr)
        # 绘制面积图
        plt.figure(figsize=(8,6))
        plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
        plt.plot(fpr, tpr, color='black', lw = 1)
        # 添加对角线
        plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
        # 添加文本信息
        plt.text(0.5,0.3,'ROC曲线 (area = %0.2f)' % roc_auc)
        # 添加x轴与y轴标签
        plt.title('模型预测的ROC曲线',size=15)
        plt.xlabel('1-Specificity')
        plt.ylabel('Sensitivity')
        plt.show()
        print ('-' * 60)


        print('输出评估报告...')
        print('\n')

        print('模型的评估报告：\n',classification_report(y_test, pred ))
        print ('-' * 60)


        print('保存模型到本地...')
        joblib.dump(model,file_name+'.pkl')
        print('\n')
        print('加载本地训练好的模型...')
        model2 = joblib.load(file_name+'.pkl')
        print('加载完毕')
        print ('-' * 60)

        #样本外预测
        test_sample_predict={}
        test_sample_score=[]
        test_sample_accuracy=[]
        test_sample_roc_auc=[]
        test_sample_date=[]

        interval_start,interval_end = self.get_time_inverval(out_start,out_end,'M')
        # 样本外预测结果
        print('样本外测试结果...')
        for date1,date2 in dict(zip(interval_start,interval_end)).items():

            data_merge_label = self.data_for_model_perperiod(date1,date2,index_code)
            y_test=data_merge_label['Label']
            X_test=data_merge_label.iloc[:,:-1]


            # 输出预测值以及预测概率
            y_pred_tmp = model2.predict(X_test)
            y_pred = pd.DataFrame(y_pred_tmp, columns=['label_predict'])  # 获得预测标签
            y_pred_proba = pd.DataFrame(model.predict_proba(X_test), columns=['pro1', 'pro2'])  # 获得预测概率
            # 将预测标签、预测数据和原始数据X合并
            y_pred.set_index(X_test.index,inplace=True)
            y_pred_proba.set_index(X_test.index,inplace=True)
            predict_pd = pd.concat((X_test, y_pred, y_pred_proba), axis=1)
            print ('Predict date:')
            print (date1)    
            print ('AUC:')
            print (roc_auc_score(y_test,y_pred))  # 打印前2条结果
            print ('Accuracy:')
            print (accuracy_score(y_test, y_pred))  # 打印前2条结果    
            print ('-' * 60)       
            ## 后续统计画图用
            test_sample_date.append(date1)
            # 样本外预测结果
            test_sample_predict[date1]=y_pred_tmp
            # 样本外准确率
            test_sample_accuracy.append(accuracy_score(y_test, y_pred))   
            # 样本外AUC值
            test_sample_roc_auc.append(roc_auc_score(y_test,y_pred))

        print ('AUC mean info')
        print (np.mean(test_sample_roc_auc))
        print ('-' * 60)    
        print ('ACCURACY mean info')
        print (np.mean(test_sample_accuracy))
        print ('-' * 60)   
        
        f = plt.figure(figsize= (15,6))
        xs_date = test_sample_date
        ys_auc = test_sample_roc_auc
        # 配置横坐标
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(xs_date, ys_auc,'r')
        # 自动旋转日期标记
        plt.gcf().autofmt_xdate() 
        # 横坐标标记
        plt.xlabel('date')
        # 纵坐标标记
        plt.ylabel("test AUC")
        plt.show()


        f = plt.figure(figsize= (15,6))
        xs_date = test_sample_date
        ys_score = test_sample_accuracy
        # 配置横坐标
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.plot(xs_date, ys_score,'r')
        # 自动旋转日期标记
        plt.gcf().autofmt_xdate() 
        # 横坐标标记
        plt.xlabel('date')
        # 纵坐标标记
        plt.ylabel("test accuracy")
        plt.show()



        f = plt.figure(figsize= (15,6))

        sns.set(style="whitegrid")
        data1 = pd.DataFrame(ys_auc, xs_date, columns={'AUC'})
        data2 = pd.DataFrame(ys_score, xs_date, columns={'accuracy'})
        data = pd.concat([data1,data2],sort=False)
        sns.lineplot(data=data, palette="tab10", linewidth=2.5)
        return model2

```


```python
#创建实例，参数（基准），数据默认2005年1月5日到2020年2月28日的全A股数据
#基准：沪深300,可选
benchmark = '000300.XSHG'
report = Report(benchmark)
```

## 获取和获取数据：量价数据调用(前复权）

### 全A股的历史量价数据，涨跌，开盘价，收盘价，最高价，最低价，换手率，成交股票数，成交金额


### 要调用的数据的格式

储存的可调用的股票历史收益率数据框


```python
#储存的可调用的股票历史收益率数据框
report.data_return.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>0.007379</td>
      <td>NaN</td>
      <td>0.009070</td>
      <td>NaN</td>
      <td>0.005272</td>
      <td>NaN</td>
      <td>-0.012048</td>
      <td>0.015015</td>
      <td>0.010638</td>
      <td>0.018405</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>-0.009992</td>
      <td>NaN</td>
      <td>-0.007904</td>
      <td>NaN</td>
      <td>-0.010741</td>
      <td>NaN</td>
      <td>-0.012195</td>
      <td>0.002959</td>
      <td>-0.015789</td>
      <td>-0.009036</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>0.004292</td>
      <td>NaN</td>
      <td>0.002265</td>
      <td>NaN</td>
      <td>0.001362</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.017699</td>
      <td>0.005348</td>
      <td>0.018237</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>0.006146</td>
      <td>NaN</td>
      <td>0.008941</td>
      <td>NaN</td>
      <td>0.011377</td>
      <td>NaN</td>
      <td>0.037037</td>
      <td>0.005797</td>
      <td>0.005319</td>
      <td>0.017910</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史收盘价数据框


```python
#储存的可调用的股票历史收盘价数据框
report.data_close.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>1242.77</td>
      <td>NaN</td>
      <td>3025.42</td>
      <td>NaN</td>
      <td>827.07</td>
      <td>NaN</td>
      <td>0.83</td>
      <td>3.33</td>
      <td>1.88</td>
      <td>3.26</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>1251.94</td>
      <td>NaN</td>
      <td>3052.86</td>
      <td>NaN</td>
      <td>831.43</td>
      <td>NaN</td>
      <td>0.82</td>
      <td>3.38</td>
      <td>1.90</td>
      <td>3.32</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>1239.43</td>
      <td>NaN</td>
      <td>3028.73</td>
      <td>NaN</td>
      <td>822.50</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>3.39</td>
      <td>1.87</td>
      <td>3.29</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>1244.75</td>
      <td>NaN</td>
      <td>3035.59</td>
      <td>NaN</td>
      <td>823.62</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>3.45</td>
      <td>1.88</td>
      <td>3.35</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>1252.40</td>
      <td>NaN</td>
      <td>3062.73</td>
      <td>NaN</td>
      <td>832.99</td>
      <td>NaN</td>
      <td>0.84</td>
      <td>3.47</td>
      <td>1.89</td>
      <td>3.41</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史最低价数据框


```python
#储存的可调用的股票历史最低价数据框
report.data_low.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>1238.18</td>
      <td>NaN</td>
      <td>3016.26</td>
      <td>NaN</td>
      <td>824.01</td>
      <td>NaN</td>
      <td>0.82</td>
      <td>3.31</td>
      <td>1.84</td>
      <td>3.26</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>1235.75</td>
      <td>NaN</td>
      <td>3017.34</td>
      <td>NaN</td>
      <td>822.97</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>3.31</td>
      <td>1.86</td>
      <td>3.24</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>1234.24</td>
      <td>NaN</td>
      <td>3016.15</td>
      <td>NaN</td>
      <td>820.34</td>
      <td>NaN</td>
      <td>0.80</td>
      <td>3.38</td>
      <td>1.87</td>
      <td>3.26</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>1235.51</td>
      <td>NaN</td>
      <td>3019.07</td>
      <td>NaN</td>
      <td>819.44</td>
      <td>NaN</td>
      <td>0.80</td>
      <td>3.39</td>
      <td>1.86</td>
      <td>3.26</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>1236.09</td>
      <td>NaN</td>
      <td>3018.49</td>
      <td>NaN</td>
      <td>821.00</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>3.43</td>
      <td>1.84</td>
      <td>3.32</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史开盘价数据框


```python
#储存的可调用的股票历史开盘价数据框
report.data_open.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>1260.78</td>
      <td>NaN</td>
      <td>3051.24</td>
      <td>NaN</td>
      <td>836.99</td>
      <td>NaN</td>
      <td>0.84</td>
      <td>3.33</td>
      <td>2.03</td>
      <td>3.32</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>1241.68</td>
      <td>NaN</td>
      <td>3020.72</td>
      <td>NaN</td>
      <td>825.71</td>
      <td>NaN</td>
      <td>0.83</td>
      <td>3.33</td>
      <td>1.86</td>
      <td>3.26</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>1252.49</td>
      <td>NaN</td>
      <td>3054.10</td>
      <td>NaN</td>
      <td>831.99</td>
      <td>NaN</td>
      <td>0.82</td>
      <td>3.39</td>
      <td>1.90</td>
      <td>3.32</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>1239.32</td>
      <td>NaN</td>
      <td>3028.69</td>
      <td>NaN</td>
      <td>822.67</td>
      <td>NaN</td>
      <td>0.81</td>
      <td>3.39</td>
      <td>1.89</td>
      <td>3.29</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>1243.58</td>
      <td>NaN</td>
      <td>3032.84</td>
      <td>NaN</td>
      <td>823.77</td>
      <td>NaN</td>
      <td>0.82</td>
      <td>3.45</td>
      <td>1.89</td>
      <td>3.32</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史最高价数据框


```python
#储存的可调用的股票历史最高价数据框
report.data_high.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>1260.78</td>
      <td>NaN</td>
      <td>3051.24</td>
      <td>NaN</td>
      <td>836.99</td>
      <td>NaN</td>
      <td>0.84</td>
      <td>3.38</td>
      <td>2.03</td>
      <td>3.32</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>1258.58</td>
      <td>NaN</td>
      <td>3067.67</td>
      <td>NaN</td>
      <td>836.43</td>
      <td>NaN</td>
      <td>0.83</td>
      <td>3.42</td>
      <td>1.94</td>
      <td>3.34</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>1252.73</td>
      <td>NaN</td>
      <td>3054.10</td>
      <td>NaN</td>
      <td>833.07</td>
      <td>NaN</td>
      <td>0.83</td>
      <td>3.43</td>
      <td>1.92</td>
      <td>3.32</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>1256.31</td>
      <td>NaN</td>
      <td>3065.28</td>
      <td>NaN</td>
      <td>832.95</td>
      <td>NaN</td>
      <td>0.82</td>
      <td>3.47</td>
      <td>1.92</td>
      <td>3.35</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>1252.72</td>
      <td>NaN</td>
      <td>3063.66</td>
      <td>NaN</td>
      <td>833.65</td>
      <td>NaN</td>
      <td>0.84</td>
      <td>3.49</td>
      <td>1.91</td>
      <td>3.41</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史换手率数据框


```python
#储存的可调用的股票历史换手率数据框
report.data_turnover.head()
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
      <th>000001.XSHE</th>
      <th>000002.XSHE</th>
      <th>000004.XSHE</th>
      <th>000005.XSHE</th>
      <th>000006.XSHE</th>
      <th>000007.XSHE</th>
      <th>000008.XSHE</th>
      <th>000009.XSHE</th>
      <th>000010.XSHE</th>
      <th>000011.XSHE</th>
      <th>...</th>
      <th>603989.XSHG</th>
      <th>603990.XSHG</th>
      <th>603991.XSHG</th>
      <th>603992.XSHG</th>
      <th>603993.XSHG</th>
      <th>603995.XSHG</th>
      <th>603996.XSHG</th>
      <th>603997.XSHG</th>
      <th>603998.XSHG</th>
      <th>603999.XSHG</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>0.1249</td>
      <td>0.6566</td>
      <td>0.6400</td>
      <td>0.2321</td>
      <td>0.1816</td>
      <td>0.3639</td>
      <td>0.2005</td>
      <td>0.1430</td>
      <td>0.5186</td>
      <td>0.5356</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>0.2286</td>
      <td>1.1155</td>
      <td>1.1179</td>
      <td>0.2236</td>
      <td>0.1860</td>
      <td>0.4015</td>
      <td>0.2400</td>
      <td>0.1395</td>
      <td>0.7805</td>
      <td>0.3907</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>0.1892</td>
      <td>1.1683</td>
      <td>1.0751</td>
      <td>0.2092</td>
      <td>0.1633</td>
      <td>0.3733</td>
      <td>0.5760</td>
      <td>0.1076</td>
      <td>0.9594</td>
      <td>0.1615</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>0.1338</td>
      <td>1.1239</td>
      <td>0.8072</td>
      <td>0.8032</td>
      <td>1.5223</td>
      <td>0.5812</td>
      <td>0.4816</td>
      <td>0.2765</td>
      <td>1.1124</td>
      <td>0.1458</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>0.1868</td>
      <td>0.4496</td>
      <td>0.7163</td>
      <td>1.9234</td>
      <td>7.8899</td>
      <td>1.0698</td>
      <td>0.5288</td>
      <td>0.6223</td>
      <td>1.9922</td>
      <td>0.1698</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3695 columns</p>
</div>



储存的可调用的股票历史成交金额数据框


```python
##储存的可调用的股票历史成交金额数据框
report.data_money.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>4.418452e+09</td>
      <td>NaN</td>
      <td>980468922.0</td>
      <td>NaN</td>
      <td>2.136409e+09</td>
      <td>NaN</td>
      <td>26134943.0</td>
      <td>20728330.0</td>
      <td>10491518.0</td>
      <td>1555757.0</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>4.916589e+09</td>
      <td>NaN</td>
      <td>807720454.0</td>
      <td>NaN</td>
      <td>1.705649e+09</td>
      <td>NaN</td>
      <td>35366812.0</td>
      <td>12969407.0</td>
      <td>8801098.0</td>
      <td>1330700.0</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>4.381370e+09</td>
      <td>NaN</td>
      <td>762259679.0</td>
      <td>NaN</td>
      <td>1.519687e+09</td>
      <td>NaN</td>
      <td>28758188.0</td>
      <td>29118085.0</td>
      <td>3727731.0</td>
      <td>1328303.0</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>5.040042e+09</td>
      <td>NaN</td>
      <td>843160298.0</td>
      <td>NaN</td>
      <td>1.640665e+09</td>
      <td>NaN</td>
      <td>29239203.0</td>
      <td>36353317.0</td>
      <td>6311149.0</td>
      <td>2460959.0</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>4.118292e+09</td>
      <td>NaN</td>
      <td>734534698.0</td>
      <td>NaN</td>
      <td>1.402314e+09</td>
      <td>NaN</td>
      <td>48798404.0</td>
      <td>15893640.0</td>
      <td>4484497.0</td>
      <td>4907955.0</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



储存的可调用的股票历史成交手数数据框


```python
##储存的可调用的股票历史成交手数数据框
report.data_volume.head()
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
      <th>000001.XSHG</th>
      <th>399006.XSHE</th>
      <th>399001.XSHE</th>
      <th>399005.XSHE</th>
      <th>000016.XSHG</th>
      <th>000906.XSHG</th>
      <th>600000.XSHG</th>
      <th>600004.XSHG</th>
      <th>600006.XSHG</th>
      <th>600007.XSHG</th>
      <th>...</th>
      <th>300813.XSHE</th>
      <th>300815.XSHE</th>
      <th>300816.XSHE</th>
      <th>300817.XSHE</th>
      <th>300818.XSHE</th>
      <th>300819.XSHE</th>
      <th>300820.XSHE</th>
      <th>300821.XSHE</th>
      <th>300822.XSHE</th>
      <th>300823.XSHE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2005-01-04</th>
      <td>816177000.0</td>
      <td>NaN</td>
      <td>138575300.0</td>
      <td>NaN</td>
      <td>403169700.0</td>
      <td>NaN</td>
      <td>31512001.0</td>
      <td>6205383.0</td>
      <td>5537612.0</td>
      <td>473684.0</td>
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
    </tr>
    <tr>
      <th>2005-01-05</th>
      <td>867865100.0</td>
      <td>NaN</td>
      <td>123303900.0</td>
      <td>NaN</td>
      <td>302086300.0</td>
      <td>NaN</td>
      <td>43229333.0</td>
      <td>3835495.0</td>
      <td>4655335.0</td>
      <td>404622.0</td>
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
    </tr>
    <tr>
      <th>2005-01-06</th>
      <td>792225400.0</td>
      <td>NaN</td>
      <td>108869300.0</td>
      <td>NaN</td>
      <td>275357400.0</td>
      <td>NaN</td>
      <td>35558905.0</td>
      <td>8549882.0</td>
      <td>1978640.0</td>
      <td>406085.0</td>
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
    </tr>
    <tr>
      <th>2005-01-07</th>
      <td>894087100.0</td>
      <td>NaN</td>
      <td>117752600.0</td>
      <td>NaN</td>
      <td>306608600.0</td>
      <td>NaN</td>
      <td>36094716.0</td>
      <td>10584134.0</td>
      <td>3332919.0</td>
      <td>743356.0</td>
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
    </tr>
    <tr>
      <th>2005-01-10</th>
      <td>723468300.0</td>
      <td>NaN</td>
      <td>94298700.0</td>
      <td>NaN</td>
      <td>247941100.0</td>
      <td>NaN</td>
      <td>58865757.0</td>
      <td>4597541.0</td>
      <td>2385315.0</td>
      <td>1449998.0</td>
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
    </tr>
  </tbody>
</table>
<p>5 rows × 3708 columns</p>
</div>



### 为了验证本地的数据是否是可靠的，随机查一下2020年1月15日平安银行的量价数据，与市面专业的商用软件通达信的数据进行对比


```python
stock_choose = '000001.XSHE'
time_choose = '2020-01-15'
print(time_choose)
print('开盘',report.data_open.loc[time_choose,stock_choose])
print('最高',report.data_high.loc[time_choose,stock_choose])
print('最低',report.data_low.loc[time_choose,stock_choose])
print('收盘',report.data_close.loc[time_choose,stock_choose])
print('总量',report.data_volume.loc[time_choose,stock_choose])
print('换手',report.data_turnover.loc[time_choose,stock_choose])
print('总额',report.data_money.loc[time_choose,stock_choose])
print('涨跌',report.data_return.loc[time_choose,stock_choose])
```

    2020-01-15
    开盘 16.79
    最高 16.86
    最低 16.45
    收盘 16.52
    总量 85943912.0
    换手 0.4429
    总额 1424889228.07
    涨跌 -0.014319809069212487
    
<div align = 'center'> <img src='http://kan.027cgb.com/627139/bgpc/20200605/1.png'/></div>


### 对比下，本地数据还是可信的

储存的可调用的历史基准收益


```python
#储存的可调用的历史基准收益
report.benchmark_return.tail()
```




    2020-02-24   -0.004013
    2020-02-25   -0.002175
    2020-02-26   -0.012326
    2020-02-27    0.002912
    2020-02-28   -0.035455
    Name: close, dtype: float64



## 处理数据，合成特征和标签：手把手利用量价数据计算技术指标


```python
#输入数据类型
close_pingan = report.data_close['000001.XSHE']
open_pingan = report.data_open['000001.XSHE']
high_pingan = report.data_high['000001.XSHE']
low_pingan = report.data_low['000001.XSHE']
volume_pingan = report.data_volume['000001.XSHE']
return_pingan = report.data_return['000001.XSHE']
money_pingan = report.data_return['000001.XSHE']
low_pingan.head()
```




    2005-01-04    1.48
    2005-01-05    1.46
    2005-01-06    1.48
    2005-01-07    1.48
    2005-01-10    1.46
    Name: 000001.XSHE, dtype: float64



### 计算均线


```python
def get_MA(df_close,n):
    '''
    移动平均值是在一定范围内的价格平均值
    df_close:量价数据，dataframe或者series
    n: 回溯的天数，整数型
    '''
    ma = df_close.rolling(n).mean()
    ma = pd.DataFrame({'MA_' + str(n): ma}, index = ma.index)
    return ma
```


```python
#计算平安银行的平均移动线
get_MA(close_pingan,20).tail()
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
      <th>MA_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>15.1320</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>15.0615</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>15.0110</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>14.9620</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>14.9100</td>
    </tr>
  </tbody>
</table>
</div>



也可以用开源库talib来计算


```python
#用talib库计算验证自己的计算结果
talib.MA(np.array(close_pingan),timeperiod=20, matype=0)
```




    array([   nan,    nan,    nan, ..., 15.011, 14.962, 14.91 ])




```python
## 计算变化率ROC
def get_ROC(df_close, n):
    '''
    ROC=(今天的收盘价-N日前的收盘价)/N日前的收盘价*100
    移动平均值是在一定范围内的价格平均值
    df_close:量价收盘数据，dataframe或者series
    n: 回溯的天数，整数型
    '''
    M = df_close
    N = df_close.shift(n)
    roc = pd.DataFrame({'ROC_' + str(n): (M-N) / N*100}, index = M.index)
    return roc
```


```python
get_ROC(close_pingan, 12).tail()
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
      <th>ROC_12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>3.114421</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>2.872777</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>3.379310</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>2.163624</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>-1.828030</td>
    </tr>
  </tbody>
</table>
</div>




```python
#用talib库计算验证自己的计算结果
talib.ROC(close_pingan,12).tail()
```




    2020-02-24    3.114421
    2020-02-25    2.872777
    2020-02-26    3.379310
    2020-02-27    2.163624
    2020-02-28   -1.828030
    dtype: float64




```python
## 计算RSI
def get_RSI(df_close,n):
    '''
    df_close:量价收盘数据，dataframe或者series
    n: 回溯的天数，整数型
    '''
    rsi = talib.RSI(df_close, timeperiod=n)
    return pd.DataFrame({'RSI_' + str(n): rsi}, index = df_close.index)
```


```python
df = get_price('000001.XSHE', start_date='2015-09-16', end_date='2020-3-22', frequency='daily')
```


```python
get_RSI(close_pingan,6).tail()
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
      <th>RSI_6</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>49.477234</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>42.341524</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>40.497130</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>47.129830</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>28.054166</td>
    </tr>
  </tbody>
</table>
</div>




```python
##计算OBV指标
def get_OBV(df_close,df_volume):
    '''
    On Balance Volume 能量，通过统计成交量变动的趋势推测股价趋势
    df_close:量价收盘数据，dataframe或者series
    df_volume:J量价交易数数据，dataframe或者series
    '''
    obv = talib.OBV(df_close,df_volume)
    return pd.DataFrame({'OBV': obv}, index = df_close.index)
```


```python
get_OBV(close_pingan,volume_pingan).tail()
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
      <th>OBV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>4.570885e+10</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>4.559439e+10</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>4.547673e+10</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>4.557426e+10</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>4.544420e+10</td>
    </tr>
  </tbody>
</table>
</div>




```python
#真实波幅
def get_ATR(df_high,df_low,df_close,n):
    '''
    平均真实波幅，主要用来衡量价格的波动
    df_close:量价收盘数据，dataframe或者series
    df_high:量价最高价数据，dataframe或者series
    df_low:量价最低价数据，dataframe或者series
    n: 回溯的天数，整数型
    '''
    atr = talib.ATR(df_high,df_low,df_close, timeperiod=n)
    return pd.DataFrame({'ATR_' + str(n): atr}, index = df_close.index)
```


```python
get_ATR(high_pingan,low_pingan,close_pingan,14).tail()
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
      <th>ATR_14</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>0.422029</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>0.424027</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>0.434454</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>0.421993</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>0.438279</td>
    </tr>
  </tbody>
</table>
</div>




```python
#上升动向值
def get_MOM(df_close,n):
    '''
    上升动向值,投资学中意思为续航，指股票(或经济指数)持续增长的能力。研究发现，赢家组合在牛市中存在着正的动量效应，输家组合在熊市中存在着负的动量效应。
    df_close:量价收盘数据，dataframe或者series
    n: 回溯的天数，整数型
    '''
    mom = talib.MOM(df_close, timeperiod=n)
    return pd.DataFrame({'MOM_' + str(n): mom}, index = df_close.index)

```


```python
def merge_raw_factors(df_close,df_open,df_high,df_low,df_volume,df_money):
    return pd.concat([get_MA(df_close,5),get_MA(df_close,20),get_MA(df_close,60),get_ROC(df_close, 6),get_ROC(df_close, 12),get_RSI(df_close,6),\
    get_RSI(df_close,12),get_RSI(df_close,24),get_OBV(df_close,df_volume),get_ATR(df_high,df_low,df_close,14),get_MOM(df_close,10)],axis = 1)
```


```python
#平安银行的每日指标
factors = merge_raw_factors(close_pingan,open_pingan,high_pingan,low_pingan,volume_pingan,money_pingan)
factors.tail()
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
      <th>MA_5</th>
      <th>MA_20</th>
      <th>MA_60</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_12</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>15.368</td>
      <td>15.1320</td>
      <td>15.831000</td>
      <td>1.330672</td>
      <td>3.114421</td>
      <td>49.477234</td>
      <td>47.099803</td>
      <td>46.196881</td>
      <td>4.570885e+10</td>
      <td>0.422029</td>
      <td>0.73</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>15.336</td>
      <td>15.0615</td>
      <td>15.821833</td>
      <td>-2.147040</td>
      <td>2.872777</td>
      <td>42.341524</td>
      <td>43.943630</td>
      <td>44.629065</td>
      <td>4.559439e+10</td>
      <td>0.424027</td>
      <td>0.25</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>15.286</td>
      <td>15.0110</td>
      <td>15.808333</td>
      <td>-1.381579</td>
      <td>3.379310</td>
      <td>40.497130</td>
      <td>43.114224</td>
      <td>44.216995</td>
      <td>4.547673e+10</td>
      <td>0.434454</td>
      <td>0.22</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>15.190</td>
      <td>14.9620</td>
      <td>15.799833</td>
      <td>-0.853018</td>
      <td>2.163624</td>
      <td>47.129830</td>
      <td>45.792942</td>
      <td>45.477725</td>
      <td>4.557426e+10</td>
      <td>0.421993</td>
      <td>0.46</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>14.974</td>
      <td>14.9100</td>
      <td>15.783667</td>
      <td>-6.991661</td>
      <td>-1.828030</td>
      <td>28.054166</td>
      <td>36.310977</td>
      <td>40.609408</td>
      <td>4.544420e+10</td>
      <td>0.438279</td>
      <td>-0.53</td>
    </tr>
  </tbody>
</table>
</div>




```python
#数据去极值及标准化
def winsorize_and_standarlize(data,qrange=[0.05,0.95],axis=0):
    '''
    input:
    data:Dataframe or series,输入数据
    qrange:list,list[0]下分位数，list[1]，上分位数，极值用分位数代替
    '''
    if isinstance(data,pd.DataFrame):
        if axis == 0:
            q_down = data.quantile(qrange[0])
            q_up = data.quantile(qrange[1])
            index = data.index
            col = data.columns
            for n in col:
                data[n][data[n] > q_up[n]] = q_up[n]
                data[n][data[n] < q_down[n]] = q_down[n]
            data = (data - data.mean())/data.std()
            data = data.fillna(0)
        else:
            data = data.stack()
            data = data.unstack(0)
            q = data.quantile(qrange)
            index = data.index
            col = data.columns
            for n in col:
                data[n][data[n] > q[n]] = q[n]
            data = (data - data.mean())/data.std()
            data = data.stack().unstack(0)
            data = data.fillna(0)
            
    elif isinstance(data,pd.Series):
        name = data.name
        q = data.quantile(qrange)
        data[data>q] = q
        data = (data - data.mean())/data.std()
    return data
```

### 数据查看


```python
data_pro = winsorize_and_standarlize(factors)
data_pro.describe()
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
      <th>MA_5</th>
      <th>MA_20</th>
      <th>MA_60</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_12</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
      <td>3.682000e+03</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.700794e-14</td>
      <td>2.515123e-14</td>
      <td>-2.662883e-14</td>
      <td>-7.065385e-16</td>
      <td>-1.112786e-16</td>
      <td>2.447526e-14</td>
      <td>2.093756e-14</td>
      <td>-1.427719e-14</td>
      <td>2.011150e-14</td>
      <td>-7.692863e-15</td>
      <td>3.948800e-16</td>
    </tr>
    <tr>
      <th>std</th>
      <td>9.994565e-01</td>
      <td>9.974158e-01</td>
      <td>9.919535e-01</td>
      <td>9.991847e-01</td>
      <td>9.983687e-01</td>
      <td>9.991847e-01</td>
      <td>9.983687e-01</td>
      <td>9.967347e-01</td>
      <td>1.000000e+00</td>
      <td>9.980965e-01</td>
      <td>9.986407e-01</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-1.705255e+00</td>
      <td>-1.728499e+00</td>
      <td>-1.780926e+00</td>
      <td>-1.824671e+00</td>
      <td>-1.810217e+00</td>
      <td>-1.702614e+00</td>
      <td>-1.659394e+00</td>
      <td>-1.596035e+00</td>
      <td>-1.534646e+00</td>
      <td>-1.351977e+00</td>
      <td>-2.025731e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-5.698024e-01</td>
      <td>-5.721285e-01</td>
      <td>-5.615280e-01</td>
      <td>-6.282979e-01</td>
      <td>-6.495794e-01</td>
      <td>-7.731055e-01</td>
      <td>-7.740355e-01</td>
      <td>-7.952646e-01</td>
      <td>-8.534952e-01</td>
      <td>-8.839974e-01</td>
      <td>-5.336564e-01</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-1.084304e-01</td>
      <td>-8.470670e-02</td>
      <td>-7.784826e-02</td>
      <td>-9.829604e-02</td>
      <td>-6.213428e-02</td>
      <td>-3.013210e-02</td>
      <td>-3.795124e-02</td>
      <td>-8.486868e-02</td>
      <td>-2.590756e-01</td>
      <td>-9.136803e-02</td>
      <td>-3.629837e-02</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.278916e-01</td>
      <td>6.541132e-01</td>
      <td>6.663069e-01</td>
      <td>5.347080e-01</td>
      <td>5.678273e-01</td>
      <td>7.716410e-01</td>
      <td>7.247613e-01</td>
      <td>7.206211e-01</td>
      <td>1.097747e+00</td>
      <td>6.497930e-01</td>
      <td>4.989021e-01</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1.994363e+00</td>
      <td>1.940143e+00</td>
      <td>1.881687e+00</td>
      <td>2.213011e+00</td>
      <td>2.117848e+00</td>
      <td>1.767631e+00</td>
      <td>1.851835e+00</td>
      <td>1.935446e+00</td>
      <td>1.523051e+00</td>
      <td>1.985683e+00</td>
      <td>2.114235e+00</td>
    </tr>
  </tbody>
</table>
</div>



平安银行的每日收益


```python
#平安银行的每日收益
profit = copy.deepcopy(return_pingan)
profit.tail(10)
```




    2020-02-17    0.022621
    2020-02-18   -0.011061
    2020-02-19    0.002632
    2020-02-20    0.022966
    2020-02-21   -0.000641
    2020-02-24   -0.022465
    2020-02-25   -0.012475
    2020-02-26   -0.003324
    2020-02-27    0.008005
    2020-02-28   -0.040371
    Name: 000001.XSHE, dtype: float64



收益率大于0标记为1，否则为0,记作标签


```python
#收益率大于0标记为1，否则为0,记作标签
profit[profit > 0] = 1
profit[profit< 0] = 0
profit = pd.DataFrame(profit)
profit.columns = ['Label']
profit.tail()
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
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
#预测的是未来一天的涨跌，所以label比特征要滞后一天
profit.shift().tail()
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
      <th>Label</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



合并后的数据框，第一列label为标签，其余列称为特征


```python
data_final = pd.concat([profit.shift(),data_pro],axis = 1)
data_final.tail()
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
      <th>Label</th>
      <th>MA_5</th>
      <th>MA_20</th>
      <th>MA_60</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_12</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-02-24</th>
      <td>0.0</td>
      <td>1.994363</td>
      <td>1.940143</td>
      <td>1.881687</td>
      <td>0.154400</td>
      <td>0.287494</td>
      <td>-0.101360</td>
      <td>-0.362483</td>
      <td>-0.602621</td>
      <td>1.523051</td>
      <td>1.427686</td>
      <td>1.499024</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>0.0</td>
      <td>1.994363</td>
      <td>1.940143</td>
      <td>1.881687</td>
      <td>-0.506021</td>
      <td>0.254781</td>
      <td>-0.510047</td>
      <td>-0.616758</td>
      <td>-0.775493</td>
      <td>1.523051</td>
      <td>1.442254</td>
      <td>0.461060</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>0.0</td>
      <td>1.994363</td>
      <td>1.940143</td>
      <td>1.881687</td>
      <td>-0.360660</td>
      <td>0.323354</td>
      <td>-0.615682</td>
      <td>-0.683579</td>
      <td>-0.820929</td>
      <td>1.523051</td>
      <td>1.518279</td>
      <td>0.396187</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>0.0</td>
      <td>1.994363</td>
      <td>1.940143</td>
      <td>1.881687</td>
      <td>-0.260285</td>
      <td>0.158779</td>
      <td>-0.235804</td>
      <td>-0.467769</td>
      <td>-0.681917</td>
      <td>1.523051</td>
      <td>1.427421</td>
      <td>0.915169</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>1.0</td>
      <td>1.994363</td>
      <td>1.940143</td>
      <td>1.881687</td>
      <td>-1.426021</td>
      <td>-0.381598</td>
      <td>-1.328334</td>
      <td>-1.231680</td>
      <td>-1.218712</td>
      <td>1.523051</td>
      <td>1.546171</td>
      <td>-1.225633</td>
    </tr>
  </tbody>
</table>
</div>



## 功能集成，方便以后调用

### 将上述函数集成，添加到上面的类对象里，作为一个新的功能, 看懂了你也可以添加新的指标，或者改进代码

### 新功能：给定时间,给定股票或者指数，就能输出各种特征  


```python
#调用类中的新功能
report.merge_final_factor_label1('600000.XSHG','2020-01-21','2020-02-28')
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
      <th>Label</th>
      <th>MA_5</th>
      <th>MA_60</th>
      <th>Aroondown_14</th>
      <th>Aroonup_14</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-21</th>
      <td>1.0</td>
      <td>2.310415</td>
      <td>1.680919</td>
      <td>1.385505</td>
      <td>0.850139</td>
      <td>-0.049595</td>
      <td>0.151672</td>
      <td>-0.627981</td>
      <td>1.315566</td>
      <td>0.974446</td>
      <td>-1.829183</td>
      <td>0.119921</td>
    </tr>
    <tr>
      <th>2020-01-22</th>
      <td>0.0</td>
      <td>2.310415</td>
      <td>1.680919</td>
      <td>1.385505</td>
      <td>0.642568</td>
      <td>-0.508179</td>
      <td>-0.039098</td>
      <td>-1.254577</td>
      <td>0.318997</td>
      <td>-0.277086</td>
      <td>-1.829183</td>
      <td>-0.030231</td>
    </tr>
    <tr>
      <th>2020-01-23</th>
      <td>0.0</td>
      <td>1.933406</td>
      <td>1.563415</td>
      <td>1.385505</td>
      <td>0.434997</td>
      <td>-0.860640</td>
      <td>-0.534339</td>
      <td>-1.550447</td>
      <td>-0.808540</td>
      <td>-1.503009</td>
      <td>-1.113230</td>
      <td>-0.573091</td>
    </tr>
    <tr>
      <th>2020-02-03</th>
      <td>0.0</td>
      <td>1.066039</td>
      <td>1.348623</td>
      <td>1.385505</td>
      <td>0.227426</td>
      <td>-1.807349</td>
      <td>-1.157115</td>
      <td>-1.550447</td>
      <td>-1.538096</td>
      <td>-1.915072</td>
      <td>1.662121</td>
      <td>-1.413946</td>
    </tr>
    <tr>
      <th>2020-02-04</th>
      <td>0.0</td>
      <td>0.282451</td>
      <td>1.128566</td>
      <td>1.203410</td>
      <td>0.019855</td>
      <td>-1.807349</td>
      <td>-1.157115</td>
      <td>-1.184782</td>
      <td>-1.538096</td>
      <td>-1.915072</td>
      <td>1.662121</td>
      <td>-1.413946</td>
    </tr>
    <tr>
      <th>2020-02-05</th>
      <td>1.0</td>
      <td>-0.397643</td>
      <td>0.913774</td>
      <td>1.021315</td>
      <td>-0.187716</td>
      <td>-1.778663</td>
      <td>-1.134754</td>
      <td>-1.047883</td>
      <td>-1.399087</td>
      <td>-1.239307</td>
      <td>1.462821</td>
      <td>-1.393156</td>
    </tr>
    <tr>
      <th>2020-02-06</th>
      <td>1.0</td>
      <td>-0.880609</td>
      <td>0.686346</td>
      <td>0.839220</td>
      <td>-0.395288</td>
      <td>-1.437094</td>
      <td>-1.055505</td>
      <td>-0.714797</td>
      <td>-1.051062</td>
      <td>-0.630247</td>
      <td>1.287606</td>
      <td>-1.081300</td>
    </tr>
    <tr>
      <th>2020-02-07</th>
      <td>1.0</td>
      <td>-0.964882</td>
      <td>0.469448</td>
      <td>0.657125</td>
      <td>-0.602859</td>
      <td>-0.927177</td>
      <td>-0.998269</td>
      <td>-0.444543</td>
      <td>-0.782738</td>
      <td>-0.119326</td>
      <td>1.216143</td>
      <td>-0.942698</td>
    </tr>
    <tr>
      <th>2020-02-10</th>
      <td>1.0</td>
      <td>-0.964882</td>
      <td>0.259921</td>
      <td>0.475030</td>
      <td>-0.810430</td>
      <td>-0.473645</td>
      <td>-0.924015</td>
      <td>-0.585924</td>
      <td>-0.946487</td>
      <td>-0.664448</td>
      <td>0.876075</td>
      <td>-1.081300</td>
    </tr>
    <tr>
      <th>2020-02-11</th>
      <td>0.0</td>
      <td>-0.875681</td>
      <td>0.079874</td>
      <td>0.292935</td>
      <td>-1.018001</td>
      <td>1.049427</td>
      <td>-0.775532</td>
      <td>-0.186210</td>
      <td>-0.595686</td>
      <td>-0.106125</td>
      <td>0.651534</td>
      <td>-1.000449</td>
    </tr>
    <tr>
      <th>2020-02-12</th>
      <td>1.0</td>
      <td>-0.796829</td>
      <td>-0.070691</td>
      <td>0.110840</td>
      <td>-1.018001</td>
      <td>0.735070</td>
      <td>-0.805060</td>
      <td>-0.186210</td>
      <td>-0.595686</td>
      <td>-0.106125</td>
      <td>0.260558</td>
      <td>-0.804095</td>
    </tr>
    <tr>
      <th>2020-02-13</th>
      <td>0.0</td>
      <td>-0.811614</td>
      <td>-0.226521</td>
      <td>-0.071255</td>
      <td>-1.018001</td>
      <td>0.507486</td>
      <td>-0.935054</td>
      <td>-0.433283</td>
      <td>-0.795829</td>
      <td>-0.468618</td>
      <td>-0.102491</td>
      <td>-0.561541</td>
    </tr>
    <tr>
      <th>2020-02-14</th>
      <td>0.0</td>
      <td>-0.811614</td>
      <td>-0.370768</td>
      <td>-0.253350</td>
      <td>-1.018001</td>
      <td>0.522711</td>
      <td>-0.655954</td>
      <td>0.118971</td>
      <td>-0.388368</td>
      <td>-0.110450</td>
      <td>-0.378784</td>
      <td>0.039070</td>
    </tr>
    <tr>
      <th>2020-02-17</th>
      <td>1.0</td>
      <td>-0.609557</td>
      <td>-0.471847</td>
      <td>-0.435445</td>
      <td>-1.018001</td>
      <td>0.920266</td>
      <td>0.031893</td>
      <td>1.313898</td>
      <td>0.788246</td>
      <td>0.551781</td>
      <td>0.003316</td>
      <td>1.123634</td>
    </tr>
    <tr>
      <th>2020-02-18</th>
      <td>1.0</td>
      <td>-0.510992</td>
      <td>-0.586614</td>
      <td>-0.617540</td>
      <td>-1.018001</td>
      <td>0.876334</td>
      <td>0.364239</td>
      <td>0.861870</td>
      <td>0.507352</td>
      <td>0.068639</td>
      <td>-0.280534</td>
      <td>1.067038</td>
    </tr>
    <tr>
      <th>2020-02-19</th>
      <td>0.0</td>
      <td>-0.387787</td>
      <td>-0.705592</td>
      <td>-0.799635</td>
      <td>-1.018001</td>
      <td>0.808748</td>
      <td>1.464518</td>
      <td>1.032863</td>
      <td>0.681504</td>
      <td>0.470690</td>
      <td>-0.361636</td>
      <td>1.078588</td>
    </tr>
    <tr>
      <th>2020-02-20</th>
      <td>1.0</td>
      <td>-0.156160</td>
      <td>-0.810882</td>
      <td>-0.981729</td>
      <td>-1.018001</td>
      <td>0.999922</td>
      <td>1.432828</td>
      <td>1.412282</td>
      <td>1.093107</td>
      <td>1.057847</td>
      <td>-0.406533</td>
      <td>1.113239</td>
    </tr>
    <tr>
      <th>2020-02-21</th>
      <td>1.0</td>
      <td>0.060681</td>
      <td>-0.888797</td>
      <td>-1.163824</td>
      <td>1.659666</td>
      <td>1.049427</td>
      <td>1.464518</td>
      <td>1.412282</td>
      <td>1.315566</td>
      <td>1.502184</td>
      <td>-0.539459</td>
      <td>1.113239</td>
    </tr>
    <tr>
      <th>2020-02-24</th>
      <td>1.0</td>
      <td>0.050825</td>
      <td>-0.973029</td>
      <td>-1.163824</td>
      <td>1.659666</td>
      <td>0.888404</td>
      <td>1.173463</td>
      <td>0.937318</td>
      <td>0.961292</td>
      <td>0.965116</td>
      <td>-0.480417</td>
      <td>1.055488</td>
    </tr>
    <tr>
      <th>2020-02-25</th>
      <td>0.0</td>
      <td>0.065610</td>
      <td>-1.067790</td>
      <td>-1.163824</td>
      <td>1.472852</td>
      <td>0.271194</td>
      <td>0.996147</td>
      <td>0.633880</td>
      <td>0.783536</td>
      <td>0.191060</td>
      <td>-0.699303</td>
      <td>0.870685</td>
    </tr>
    <tr>
      <th>2020-02-26</th>
      <td>0.0</td>
      <td>0.109964</td>
      <td>-1.159393</td>
      <td>-1.163824</td>
      <td>1.265281</td>
      <td>0.629473</td>
      <td>1.249660</td>
      <td>1.107259</td>
      <td>1.170027</td>
      <td>1.022488</td>
      <td>-0.385546</td>
      <td>0.997737</td>
    </tr>
    <tr>
      <th>2020-02-27</th>
      <td>1.0</td>
      <td>0.100107</td>
      <td>-1.239940</td>
      <td>-1.163824</td>
      <td>1.057710</td>
      <td>0.566196</td>
      <td>1.145570</td>
      <td>1.149823</td>
      <td>1.205165</td>
      <td>1.502184</td>
      <td>-0.550383</td>
      <td>1.123634</td>
    </tr>
    <tr>
      <th>2020-02-28</th>
      <td>1.0</td>
      <td>-0.121663</td>
      <td>-1.239940</td>
      <td>-1.163824</td>
      <td>0.850139</td>
      <td>-0.174968</td>
      <td>0.697302</td>
      <td>-0.213361</td>
      <td>0.299315</td>
      <td>0.748452</td>
      <td>-0.125615</td>
      <td>0.593480</td>
    </tr>
  </tbody>
</table>
</div>



## 指数择时

#### 获取自定义区间的数据


```python
#获取指标2010年到2018年的上证指数数据，作为数据集
data_index = report.merge_final_factor_label1('000001.XSHG','2010-01-01','2018-01-01')
data_index.head(20)
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
      <th>Label</th>
      <th>MA_5</th>
      <th>MA_60</th>
      <th>Aroondown_14</th>
      <th>Aroonup_14</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2010-01-04</th>
      <td>1.0</td>
      <td>0.988304</td>
      <td>0.868502</td>
      <td>-0.147992</td>
      <td>-1.385636</td>
      <td>0.990031</td>
      <td>-0.148243</td>
      <td>0.441767</td>
      <td>0.291502</td>
      <td>-0.922039</td>
      <td>0.602991</td>
      <td>1.216550</td>
    </tr>
    <tr>
      <th>2010-01-05</th>
      <td>0.0</td>
      <td>1.025477</td>
      <td>0.881695</td>
      <td>-0.342382</td>
      <td>-1.385636</td>
      <td>1.577618</td>
      <td>0.758380</td>
      <td>0.826958</td>
      <td>0.486422</td>
      <td>-0.922039</td>
      <td>0.630772</td>
      <td>1.505414</td>
    </tr>
    <tr>
      <th>2010-01-06</th>
      <td>1.0</td>
      <td>1.042376</td>
      <td>0.892517</td>
      <td>-0.536772</td>
      <td>-1.385636</td>
      <td>0.694792</td>
      <td>1.077039</td>
      <td>0.346818</td>
      <td>0.315013</td>
      <td>-0.922039</td>
      <td>0.577094</td>
      <td>1.943435</td>
    </tr>
    <tr>
      <th>2010-01-07</th>
      <td>0.0</td>
      <td>1.014588</td>
      <td>0.900080</td>
      <td>-0.731162</td>
      <td>1.062305</td>
      <td>-0.265111</td>
      <td>0.504234</td>
      <td>-0.466328</td>
      <td>-0.042070</td>
      <td>-0.922039</td>
      <td>0.676621</td>
      <td>1.109531</td>
    </tr>
    <tr>
      <th>2010-01-08</th>
      <td>0.0</td>
      <td>0.982294</td>
      <td>0.907437</td>
      <td>-0.925552</td>
      <td>0.874002</td>
      <td>-0.791835</td>
      <td>1.143227</td>
      <td>-0.416499</td>
      <td>-0.024108</td>
      <td>-0.922039</td>
      <td>0.641097</td>
      <td>0.357246</td>
    </tr>
    <tr>
      <th>2010-01-11</th>
      <td>1.0</td>
      <td>0.969952</td>
      <td>0.915472</td>
      <td>-1.119941</td>
      <td>1.250608</td>
      <td>-0.764055</td>
      <td>1.080711</td>
      <td>-0.138320</td>
      <td>0.071259</td>
      <td>-0.922039</td>
      <td>0.792552</td>
      <td>0.640891</td>
    </tr>
    <tr>
      <th>2010-01-12</th>
      <td>1.0</td>
      <td>0.966685</td>
      <td>0.923492</td>
      <td>-1.314331</td>
      <td>1.062305</td>
      <td>0.287730</td>
      <td>0.904679</td>
      <td>0.654121</td>
      <td>0.404724</td>
      <td>-0.922039</td>
      <td>0.885783</td>
      <td>0.776659</td>
    </tr>
    <tr>
      <th>2010-01-13</th>
      <td>1.0</td>
      <td>0.934224</td>
      <td>0.926494</td>
      <td>-1.314331</td>
      <td>0.874002</td>
      <td>-1.262313</td>
      <td>0.191839</td>
      <td>-0.591117</td>
      <td>-0.177093</td>
      <td>-0.922039</td>
      <td>1.012879</td>
      <td>-0.447023</td>
    </tr>
    <tr>
      <th>2010-01-14</th>
      <td>0.0</td>
      <td>0.943287</td>
      <td>0.931427</td>
      <td>-1.314331</td>
      <td>0.685699</td>
      <td>-0.482049</td>
      <td>0.152193</td>
      <td>-0.089788</td>
      <td>0.052163</td>
      <td>-0.922039</td>
      <td>0.964007</td>
      <td>-0.525294</td>
    </tr>
    <tr>
      <th>2010-01-15</th>
      <td>1.0</td>
      <td>0.954490</td>
      <td>0.937305</td>
      <td>-1.314331</td>
      <td>0.497395</td>
      <td>0.306325</td>
      <td>0.037760</td>
      <td>0.008444</td>
      <td>0.097648</td>
      <td>-0.916050</td>
      <td>0.891891</td>
      <td>-0.583775</td>
    </tr>
    <tr>
      <th>2010-01-18</th>
      <td>1.0</td>
      <td>0.964181</td>
      <td>0.941703</td>
      <td>-1.314331</td>
      <td>0.309092</td>
      <td>0.416529</td>
      <td>-0.256650</td>
      <td>0.169099</td>
      <td>0.167496</td>
      <td>-0.906526</td>
      <td>0.798950</td>
      <td>-0.127639</td>
    </tr>
    <tr>
      <th>2010-01-19</th>
      <td>1.0</td>
      <td>0.953396</td>
      <td>0.946375</td>
      <td>0.046398</td>
      <td>0.120789</td>
      <td>0.335193</td>
      <td>-0.292484</td>
      <td>0.298502</td>
      <td>0.221135</td>
      <td>-0.897364</td>
      <td>0.700342</td>
      <td>-0.409709</td>
    </tr>
    <tr>
      <th>2010-01-20</th>
      <td>1.0</td>
      <td>0.945113</td>
      <td>0.950812</td>
      <td>1.407126</td>
      <td>-0.067514</td>
      <td>-1.405118</td>
      <td>-0.774132</td>
      <td>-0.872962</td>
      <td>-0.323491</td>
      <td>-0.908522</td>
      <td>0.835592</td>
      <td>-1.069940</td>
    </tr>
    <tr>
      <th>2010-01-21</th>
      <td>0.0</td>
      <td>0.922551</td>
      <td>0.955152</td>
      <td>1.407126</td>
      <td>-0.255817</td>
      <td>-0.208462</td>
      <td>-1.007122</td>
      <td>-0.758828</td>
      <td>-0.282305</td>
      <td>-0.900460</td>
      <td>0.786744</td>
      <td>-0.396024</td>
    </tr>
    <tr>
      <th>2010-01-22</th>
      <td>1.0</td>
      <td>0.884518</td>
      <td>0.960873</td>
      <td>1.407126</td>
      <td>-0.444120</td>
      <td>-1.032629</td>
      <td>-1.033166</td>
      <td>-1.051371</td>
      <td>-0.447064</td>
      <td>-0.910173</td>
      <td>0.883784</td>
      <td>-0.725746</td>
    </tr>
    <tr>
      <th>2010-01-25</th>
      <td>0.0</td>
      <td>0.827728</td>
      <td>0.964227</td>
      <td>1.212737</td>
      <td>-0.632424</td>
      <td>-1.511881</td>
      <td>-0.836571</td>
      <td>-1.335185</td>
      <td>-0.627804</td>
      <td>-0.915699</td>
      <td>0.815792</td>
      <td>-1.227171</td>
    </tr>
    <tr>
      <th>2010-01-26</th>
      <td>0.0</td>
      <td>0.737192</td>
      <td>0.962278</td>
      <td>1.407126</td>
      <td>-0.820727</td>
      <td>-1.923946</td>
      <td>-1.453195</td>
      <td>-1.699754</td>
      <td>-0.994930</td>
      <td>-0.922039</td>
      <td>0.938067</td>
      <td>-1.970330</td>
    </tr>
    <tr>
      <th>2010-01-27</th>
      <td>0.0</td>
      <td>0.671427</td>
      <td>0.957935</td>
      <td>1.407126</td>
      <td>-1.009030</td>
      <td>-1.923946</td>
      <td>-1.834742</td>
      <td>-1.699754</td>
      <td>-1.143798</td>
      <td>-0.922039</td>
      <td>0.902419</td>
      <td>-1.893801</td>
    </tr>
    <tr>
      <th>2010-01-28</th>
      <td>0.0</td>
      <td>0.605869</td>
      <td>0.953362</td>
      <td>1.407126</td>
      <td>-1.197333</td>
      <td>-1.867684</td>
      <td>-1.873060</td>
      <td>-1.699754</td>
      <td>-1.092096</td>
      <td>-0.922039</td>
      <td>0.827277</td>
      <td>-1.970330</td>
    </tr>
    <tr>
      <th>2010-01-29</th>
      <td>1.0</td>
      <td>0.550428</td>
      <td>0.947721</td>
      <td>1.212737</td>
      <td>-1.385636</td>
      <td>-1.923946</td>
      <td>-1.517184</td>
      <td>-1.699754</td>
      <td>-1.115215</td>
      <td>-0.922039</td>
      <td>0.800694</td>
      <td>-1.970330</td>
    </tr>
  </tbody>
</table>
</div>



#### 数据分割


```python
# 将数据集分样本特征集和样本特征集

#x_data：所要划分的样本特征集

#y_data：所要划分的样本标签

#test_size：样本占比，如果是整数的话就是样本的数量

#random_state：是随机数的种子。随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样。

x_data,y_data = data_index.iloc[:,1:], data_index.iloc[:,0]
```


```python
x_data.tail(20)
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
      <th>MA_5</th>
      <th>MA_60</th>
      <th>Aroondown_14</th>
      <th>Aroonup_14</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2017-12-04</th>
      <td>1.160203</td>
      <td>1.302549</td>
      <td>0.629567</td>
      <td>-1.385636</td>
      <td>-0.529114</td>
      <td>-0.724529</td>
      <td>-1.398063</td>
      <td>-1.015399</td>
      <td>1.729006</td>
      <td>-0.574550</td>
      <td>-0.877069</td>
    </tr>
    <tr>
      <th>2017-12-05</th>
      <td>1.148271</td>
      <td>1.299805</td>
      <td>1.407126</td>
      <td>-0.444120</td>
      <td>-0.253274</td>
      <td>-0.650207</td>
      <td>-1.503913</td>
      <td>-1.094497</td>
      <td>1.729006</td>
      <td>-0.625791</td>
      <td>-1.113752</td>
    </tr>
    <tr>
      <th>2017-12-06</th>
      <td>1.130799</td>
      <td>1.296693</td>
      <td>1.407126</td>
      <td>-0.632424</td>
      <td>-0.482986</td>
      <td>-0.791368</td>
      <td>-1.670960</td>
      <td>-1.222999</td>
      <td>1.729006</td>
      <td>-0.570801</td>
      <td>-1.405963</td>
    </tr>
    <tr>
      <th>2017-12-07</th>
      <td>1.112833</td>
      <td>1.293513</td>
      <td>1.212737</td>
      <td>-0.820727</td>
      <td>-0.766524</td>
      <td>-1.083353</td>
      <td>-1.699754</td>
      <td>-1.496310</td>
      <td>1.729006</td>
      <td>-0.562992</td>
      <td>-0.848419</td>
    </tr>
    <tr>
      <th>2017-12-08</th>
      <td>1.101837</td>
      <td>1.290953</td>
      <td>1.018347</td>
      <td>-1.009030</td>
      <td>-0.348283</td>
      <td>-1.092247</td>
      <td>-1.024953</td>
      <td>-1.136449</td>
      <td>1.729006</td>
      <td>-0.544920</td>
      <td>-0.690499</td>
    </tr>
    <tr>
      <th>2017-12-11</th>
      <td>1.106843</td>
      <td>1.289108</td>
      <td>0.823957</td>
      <td>-1.197333</td>
      <td>-0.000361</td>
      <td>-0.283149</td>
      <td>0.092151</td>
      <td>-0.557712</td>
      <td>1.729006</td>
      <td>-0.540233</td>
      <td>-0.062364</td>
    </tr>
    <tr>
      <th>2017-12-12</th>
      <td>1.097741</td>
      <td>1.285750</td>
      <td>0.629567</td>
      <td>-1.385636</td>
      <td>-0.366630</td>
      <td>-0.608557</td>
      <td>-0.829096</td>
      <td>-1.079091</td>
      <td>1.729006</td>
      <td>-0.513177</td>
      <td>-0.582397</td>
    </tr>
    <tr>
      <th>2017-12-13</th>
      <td>1.101355</td>
      <td>1.282990</td>
      <td>0.435177</td>
      <td>-1.385636</td>
      <td>-0.057532</td>
      <td>-0.205210</td>
      <td>-0.236514</td>
      <td>-0.717925</td>
      <td>1.729006</td>
      <td>-0.521942</td>
      <td>-0.404885</td>
    </tr>
    <tr>
      <th>2017-12-14</th>
      <td>1.109470</td>
      <td>1.280302</td>
      <td>0.240787</td>
      <td>-1.385636</td>
      <td>-0.067255</td>
      <td>-0.371375</td>
      <td>-0.463819</td>
      <td>-0.846506</td>
      <td>1.729006</td>
      <td>-0.541387</td>
      <td>-0.305742</td>
    </tr>
    <tr>
      <th>2017-12-15</th>
      <td>1.099978</td>
      <td>1.277325</td>
      <td>0.046398</td>
      <td>-1.385636</td>
      <td>-0.116092</td>
      <td>-0.601435</td>
      <td>-0.951066</td>
      <td>-1.148089</td>
      <td>1.729006</td>
      <td>-0.540983</td>
      <td>-0.568909</td>
    </tr>
    <tr>
      <th>2017-12-18</th>
      <td>1.078375</td>
      <td>1.274094</td>
      <td>1.407126</td>
      <td>-1.197333</td>
      <td>-0.294117</td>
      <td>-0.434127</td>
      <td>-0.890981</td>
      <td>-1.118135</td>
      <td>1.729006</td>
      <td>-0.560886</td>
      <td>-0.472621</td>
    </tr>
    <tr>
      <th>2017-12-19</th>
      <td>1.084635</td>
      <td>1.272042</td>
      <td>1.212737</td>
      <td>-1.385636</td>
      <td>-0.330999</td>
      <td>-0.219780</td>
      <td>-0.029386</td>
      <td>-0.655026</td>
      <td>1.729006</td>
      <td>-0.565546</td>
      <td>-0.132365</td>
    </tr>
    <tr>
      <th>2017-12-20</th>
      <td>1.078494</td>
      <td>1.269375</td>
      <td>1.018347</td>
      <td>-1.385636</td>
      <td>0.024776</td>
      <td>-0.227254</td>
      <td>-0.260446</td>
      <td>-0.764224</td>
      <td>1.729006</td>
      <td>-0.590575</td>
      <td>-0.124587</td>
    </tr>
    <tr>
      <th>2017-12-21</th>
      <td>1.081527</td>
      <td>1.267410</td>
      <td>0.823957</td>
      <td>-1.385636</td>
      <td>-0.083261</td>
      <td>-0.087167</td>
      <td>0.095794</td>
      <td>-0.568856</td>
      <td>1.729006</td>
      <td>-0.560075</td>
      <td>0.213700</td>
    </tr>
    <tr>
      <th>2017-12-22</th>
      <td>1.093833</td>
      <td>1.265522</td>
      <td>0.629567</td>
      <td>-1.385636</td>
      <td>0.000464</td>
      <td>-0.035797</td>
      <td>-0.003177</td>
      <td>-0.607870</td>
      <td>1.729006</td>
      <td>-0.616412</td>
      <td>0.007538</td>
    </tr>
    <tr>
      <th>2017-12-25</th>
      <td>1.098824</td>
      <td>1.263443</td>
      <td>0.435177</td>
      <td>-0.632424</td>
      <td>0.108730</td>
      <td>0.005290</td>
      <td>-0.520022</td>
      <td>-0.821096</td>
      <td>1.729006</td>
      <td>-0.583946</td>
      <td>-0.473015</td>
    </tr>
    <tr>
      <th>2017-12-26</th>
      <td>1.102637</td>
      <td>1.262169</td>
      <td>0.240787</td>
      <td>-0.820727</td>
      <td>0.374023</td>
      <td>0.064116</td>
      <td>0.288744</td>
      <td>-0.413773</td>
      <td>1.729006</td>
      <td>-0.580745</td>
      <td>0.187118</td>
    </tr>
    <tr>
      <th>2017-12-27</th>
      <td>1.097928</td>
      <td>1.259804</td>
      <td>0.046398</td>
      <td>-1.009030</td>
      <td>-0.279201</td>
      <td>-0.411926</td>
      <td>-0.531291</td>
      <td>-0.793420</td>
      <td>1.729006</td>
      <td>-0.566376</td>
      <td>-0.330454</td>
    </tr>
    <tr>
      <th>2017-12-28</th>
      <td>1.096464</td>
      <td>1.258332</td>
      <td>-0.147992</td>
      <td>-1.197333</td>
      <td>0.046382</td>
      <td>0.060158</td>
      <td>0.034153</td>
      <td>-0.483688</td>
      <td>1.729006</td>
      <td>-0.542000</td>
      <td>-0.023278</td>
    </tr>
    <tr>
      <th>2017-12-29</th>
      <td>1.100487</td>
      <td>1.256911</td>
      <td>-0.342382</td>
      <td>-1.385636</td>
      <td>0.027748</td>
      <td>-0.027998</td>
      <td>0.296710</td>
      <td>-0.327705</td>
      <td>1.729006</td>
      <td>-0.594899</td>
      <td>0.341887</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_data.tail(20)
```




    2017-12-04    1.0
    2017-12-05    0.0
    2017-12-06    0.0
    2017-12-07    0.0
    2017-12-08    0.0
    2017-12-11    1.0
    2017-12-12    1.0
    2017-12-13    0.0
    2017-12-14    1.0
    2017-12-15    0.0
    2017-12-18    0.0
    2017-12-19    1.0
    2017-12-20    1.0
    2017-12-21    0.0
    2017-12-22    1.0
    2017-12-25    0.0
    2017-12-26    0.0
    2017-12-27    1.0
    2017-12-28    0.0
    2017-12-29    1.0
    Name: Label, dtype: float64




```python
# #原始数据按照比例分割为“训练集”和“测试集”
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.2)
```

####  训练模型


```python
model=QDA()
model.fit(x_train, y_train)
```




    QuadraticDiscriminantAnalysis(priors=None, reg_param=0.0,
                   store_covariance=False, store_covariances=None, tol=0.0001)



### 输出训练结果和评价指标

####  混淆矩阵


```python
# 模型在测试数据集上的预测
pred = model.predict(x_test)
# 构建混淆矩阵
cm = pd.crosstab(y_test,pred)
cm
# 绘制混淆矩阵图
sns.heatmap(cm, annot = True, cmap = 'GnBu', fmt = 'd')

print('模型的准确率为：\n',accuracy_score(y_test, pred))
print('模型的评估报告：\n',classification_report(y_test, pred))
```

    模型的准确率为：
     0.6452442159383034
    模型的评估报告：
                   precision    recall  f1-score   support
    
             0.0       0.63      0.63      0.63       186
             1.0       0.66      0.66      0.66       203
    
       micro avg       0.65      0.65      0.65       389
       macro avg       0.64      0.64      0.64       389
    weighted avg       0.65      0.65      0.65       389
    
<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_73_1-min.png'/></div>
#### ROC曲线


```python
# 计算正例的预测概率，而非实际的预测值，用于生成ROC曲线的数据
y_score = model.predict_proba(x_test)[:,1]
#fpr表示1-Specificity,tpr表示Sensitivity
fpr,tpr,threshold = roc_curve(y_test, y_score)
# 计算AUC的值
roc_auc = metrics.auc(fpr,tpr)
# 绘制面积图
plt.figure(figsize=(8,6))
plt.stackplot(fpr, tpr, color='steelblue', alpha = 0.5, edgecolor = 'black')
plt.plot(fpr, tpr, color='black', lw = 1)
# 添加对角线
plt.plot([0,1],[0,1], color = 'red', linestyle = '--')
# 添加文本信息
plt.text(0.5,0.3,'ROC曲线 (area = %0.2f)' % roc_auc)
# 添加x轴与y轴标签
plt.title('模型预测指数涨跌的ROC曲线',size=15)
plt.xlabel('1-Specificity')
plt.ylabel('Sensitivity')
plt.show()
```

<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_75_0-min.png'/></div>

#### 保存模型


```python
joblib.dump(model,'train_model.pkl')
model2 = joblib.load("train_model.pkl")
```

#### 加载模型


```python
model2 = joblib.load("train_model.pkl")
```

### 样本外测试


```python
#获取样本外数据
data_out_sample = report.merge_final_factor_label1('000001.XSHG','2018-01-01','2020-01-01')
```


```python
data_out_sample.tail(10)
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
      <th>Label</th>
      <th>MA_5</th>
      <th>MA_60</th>
      <th>Aroondown_14</th>
      <th>Aroonup_14</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2019-12-18</th>
      <td>1.0</td>
      <td>0.234736</td>
      <td>-0.061065</td>
      <td>-0.783525</td>
      <td>1.202429</td>
      <td>1.361947</td>
      <td>1.427895</td>
      <td>1.862737</td>
      <td>1.513285</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>1.522202</td>
    </tr>
    <tr>
      <th>2019-12-19</th>
      <td>0.0</td>
      <td>0.326176</td>
      <td>-0.059776</td>
      <td>-0.976240</td>
      <td>1.008527</td>
      <td>1.264220</td>
      <td>1.337858</td>
      <td>1.862737</td>
      <td>1.513665</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>1.297170</td>
    </tr>
    <tr>
      <th>2019-12-20</th>
      <td>1.0</td>
      <td>0.359787</td>
      <td>-0.059885</td>
      <td>-1.168955</td>
      <td>0.814626</td>
      <td>1.222215</td>
      <td>1.286913</td>
      <td>1.208598</td>
      <td>1.265640</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>1.036778</td>
    </tr>
    <tr>
      <th>2019-12-23</th>
      <td>0.0</td>
      <td>0.340266</td>
      <td>-0.060923</td>
      <td>-1.361671</td>
      <td>0.620724</td>
      <td>-0.038880</td>
      <td>0.665595</td>
      <td>-0.266910</td>
      <td>0.490695</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>0.565392</td>
    </tr>
    <tr>
      <th>2019-12-24</th>
      <td>0.0</td>
      <td>0.304419</td>
      <td>-0.061116</td>
      <td>-1.361671</td>
      <td>0.426822</td>
      <td>0.003651</td>
      <td>0.733870</td>
      <td>0.252256</td>
      <td>0.777277</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>0.745777</td>
    </tr>
    <tr>
      <th>2019-12-25</th>
      <td>1.0</td>
      <td>0.272703</td>
      <td>-0.059200</td>
      <td>-1.361671</td>
      <td>0.232921</td>
      <td>-0.498166</td>
      <td>0.701949</td>
      <td>0.227258</td>
      <td>0.763167</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>0.662392</td>
    </tr>
    <tr>
      <th>2019-12-26</th>
      <td>0.0</td>
      <td>0.263935</td>
      <td>-0.053531</td>
      <td>-1.361671</td>
      <td>0.039019</td>
      <td>-0.099477</td>
      <td>0.918042</td>
      <td>0.826302</td>
      <td>1.116750</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>1.023268</td>
    </tr>
    <tr>
      <th>2019-12-27</th>
      <td>1.0</td>
      <td>0.264025</td>
      <td>-0.048252</td>
      <td>-1.168955</td>
      <td>-0.154882</td>
      <td>-0.129788</td>
      <td>0.826051</td>
      <td>0.734788</td>
      <td>1.073468</td>
      <td>1.435901</td>
      <td>-1.419598</td>
      <td>0.450237</td>
    </tr>
    <tr>
      <th>2019-12-30</th>
      <td>0.0</td>
      <td>0.333726</td>
      <td>-0.038485</td>
      <td>-1.361671</td>
      <td>1.396330</td>
      <td>0.482292</td>
      <td>1.247091</td>
      <td>1.361105</td>
      <td>1.524115</td>
      <td>1.435901</td>
      <td>-1.355460</td>
      <td>0.643077</td>
    </tr>
    <tr>
      <th>2019-12-31</th>
      <td>1.0</td>
      <td>0.394560</td>
      <td>-0.028593</td>
      <td>-1.168955</td>
      <td>1.396330</td>
      <td>1.178552</td>
      <td>0.831957</td>
      <td>1.494480</td>
      <td>1.644490</td>
      <td>1.435901</td>
      <td>-1.412389</td>
      <td>0.348276</td>
    </tr>
  </tbody>
</table>
</div>




```python
#划分特征集和标签集
x_out_test,y_out_test = data_out_sample.iloc[:,1:], data_out_sample.iloc[:,0]
```

#### 加载模型后可直接使用

输出预测标签


```python
y_out_pred = model2.predict(x_out_test)
y_out_pred
```




    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
           0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 1., 1., 1., 1., 0.,
           0., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
           1., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 1., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0.,
           0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 1., 0., 0., 0., 0.,
           0., 0., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 0., 1., 1., 1., 1., 1., 0., 1., 1., 0., 0., 0., 0.,
           0., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 0., 1., 1., 0.,
           1., 1., 0., 0., 0., 1., 1., 1., 0., 0., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 1., 1., 1.,
           1., 1., 1., 0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 1., 1., 1., 1.,
           1., 1., 1., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 1., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
           1., 1., 0., 0., 0., 1., 1., 0., 1., 1., 0., 1., 1., 1., 0., 1., 1.,
           1., 1., 1., 1., 1., 1., 1., 1., 0., 0., 0., 1., 0., 1., 0., 0., 0.,
           0., 0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 0.,
           0., 0., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1., 1.,
           1., 1., 1., 1., 0., 0., 0., 1., 1., 1., 1.])



 获得预测概率


```python
predict_proba = model2.predict_proba(x_out_test)[:,1] #此处test_X为特征集
predict_proba
```




    array([0.92768637, 0.87372613, 0.93611771, 0.92504257, 0.93830129,
           0.97350163, 0.98439474, 0.98494626, 0.99302094, 0.71066415,
           0.9346553 , 0.9571639 , 0.98347517, 0.98133935, 0.97773941,
           0.8420986 , 0.89008306, 0.92629049, 0.9572436 , 0.5039008 ,
           0.15433895, 0.19618445, 0.13506135, 0.17494294, 0.54099699,
           0.03775966, 0.01478691, 0.01300564, 0.01183642, 0.03102973,
           0.06048433, 0.10077356, 0.19561205, 0.16183885, 0.18157097,
           0.40267378, 0.57780427, 0.76351945, 0.98894525, 0.96757754,
           0.94039104, 0.69392342, 0.45948081, 0.53882556, 0.54942774,
           0.55376286, 0.78704139, 0.6902328 , 0.67840564, 0.50551527,
           0.60997526, 0.46440017, 0.26347114, 0.0656017 , 0.05779543,
           0.21262375, 0.25199363, 0.50075914, 0.61394749, 0.27114971,
           0.10048141, 0.13073217, 0.13687481, 0.8528971 , 0.90548906,
           0.56790395, 0.50395654, 0.13714471, 0.08843812, 0.29217134,
           0.64426371, 0.42854564, 0.32085174, 0.5137408 , 0.35244305,
           0.30040986, 0.50240616, 0.457433  , 0.67157372, 0.69082967,
           0.77687727, 0.79582527, 0.78517435, 0.74283735, 0.68615354,
           0.80688554, 0.90175701, 0.64304015, 0.42421477, 0.75869859,
           0.81017815, 0.8514314 , 0.38881122, 0.25902116, 0.16211049,
           0.16614164, 0.12334865, 0.13226115, 0.47903855, 0.36390768,
           0.5174554 , 0.73624912, 0.75098208, 0.40788233, 0.3967154 ,
           0.2439561 , 0.54479096, 0.53777812, 0.40403812, 0.25577642,
           0.24904761, 0.09929233, 0.13741176, 0.1678991 , 0.18963611,
           0.1943378 , 0.14487027, 0.18659786, 0.42940769, 0.25579801,
           0.52254904, 0.33436851, 0.23813746, 0.37284747, 0.74879912,
           0.79164491, 0.66651398, 0.88469944, 0.74616232, 0.73853718,
           0.63936874, 0.6081675 , 0.55622609, 0.75321664, 0.7815924 ,
           0.86126268, 0.85880723, 0.6411158 , 0.65249363, 0.52455551,
           0.54540161, 0.34515664, 0.16483909, 0.09813257, 0.07875876,
           0.39561187, 0.29154285, 0.57326271, 0.59871615, 0.54026514,
           0.49279335, 0.30241696, 0.42908667, 0.26428967, 0.48671753,
           0.62122644, 0.56028551, 0.56033722, 0.62665293, 0.81940654,
           0.7387033 , 0.66592612, 0.52672122, 0.37637344, 0.2850368 ,
           0.53597262, 0.24010379, 0.18681959, 0.22768771, 0.26363987,
           0.24933921, 0.19862666, 0.51994383, 0.45421865, 0.30938515,
           0.74792445, 0.78869373, 0.73751946, 0.95793289, 0.87027078,
           0.90282423, 0.77251597, 0.79809766, 0.37687865, 0.25473459,
           0.30440401, 0.06507915, 0.09396445, 0.09478336, 0.10337532,
           0.1668443 , 0.11178433, 0.39416537, 0.87374959, 0.65201754,
           0.83738234, 0.88783166, 0.95112084, 0.82290048, 0.55409254,
           0.83058832, 0.88418062, 0.97701667, 0.96498092, 0.96815346,
           0.73549159, 0.57495442, 0.55501549, 0.45825476, 0.58237228,
           0.60024485, 0.69880294, 0.77742039, 0.88159983, 0.34161263,
           0.51951705, 0.5136713 , 0.14993804, 0.14164131, 0.13503646,
           0.44318011, 0.29003659, 0.54459399, 0.93611822, 0.88347707,
           0.79203113, 0.59555529, 0.54438551, 0.38859177, 0.25961676,
           0.38008381, 0.52648356, 0.27392474, 0.52292447, 0.3474074 ,
           0.28097051, 0.17816129, 0.15292792, 0.25848705, 0.21666034,
           0.15186081, 0.20692974, 0.2616456 , 0.1740437 , 0.18470274,
           0.6455302 , 0.74751271, 0.67978954, 0.81101168, 0.72088468,
           0.86716137, 0.49636418, 0.73066438, 0.60749457, 0.48322452,
           0.86254143, 0.90508905, 0.43830084, 0.4528298 , 0.48121765,
           0.59174571, 0.54601153, 0.62680834, 0.19902157, 0.32335333,
           0.80321108, 0.93053582, 0.90023112, 0.80395445, 0.93857671,
           0.74428307, 0.83853564, 0.80653039, 0.74841075, 0.54240611,
           0.76270491, 0.78043358, 0.79909342, 0.69770529, 0.72841322,
           0.60916924, 0.46919568, 0.23561556, 0.27600535, 0.14867822,
           0.5547519 , 0.34777691, 0.16786792, 0.03765598, 0.01467337,
           0.02226528, 0.86197099, 0.36702956, 0.22010419, 0.29279984,
           0.40326685, 0.24354445, 0.34090129, 0.27896042, 0.02739086,
           0.11002659, 0.37231314, 0.60113585, 0.28467928, 0.30574615,
           0.24217869, 0.27286212, 0.0713619 , 0.23335294, 0.20911186,
           0.18794769, 0.18560383, 0.28359213, 0.18806428, 0.38684336,
           0.23436443, 0.2736422 , 0.21226031, 0.06377722, 0.07374984,
           0.01122629, 0.03179844, 0.00436962, 0.01504322, 0.02710046,
           0.04742455, 0.21986613, 0.15414805, 0.28812503, 0.48076086,
           0.61525733, 0.29366504, 0.69023641, 0.72010134, 0.73192722,
           0.71350155, 0.43443765, 0.83757989, 0.91196111, 0.80597116,
           0.71725584, 0.72846096, 0.68139639, 0.4055958 , 0.39676744,
           0.42733194, 0.5501376 , 0.8451244 , 0.73108955, 0.71831443,
           0.51176214, 0.47541834, 0.47238053, 0.86942811, 0.93746309,
           0.90782211, 0.88088337, 0.69681599, 0.53907133, 0.57675902,
           0.49121053, 0.73568145, 0.75968351, 0.49019147, 0.43524361,
           0.48736784, 0.11410632, 0.15359888, 0.15212952, 0.18340806,
           0.3275416 , 0.58926939, 0.53889026, 0.41103409, 0.20666117,
           0.51170725, 0.29108907, 0.42503409, 0.58790145, 0.67938655,
           0.69460852, 0.67718365, 0.81981997, 0.44427158, 0.17228037,
           0.08049974, 0.05629146, 0.10224809, 0.16675326, 0.37406816,
           0.34571201, 0.71925828, 0.5392739 , 0.46973142, 0.42379824,
           0.4823324 , 0.57120789, 0.60050637, 0.4901272 , 0.68224158,
           0.73347621, 0.27967437, 0.68220636, 0.5950859 , 0.52300998,
           0.45049518, 0.86898773, 0.83804669, 0.91548209, 0.89367349,
           0.85914564, 0.85690102, 0.90204988, 0.71132515, 0.79163732,
           0.812289  , 0.20139774, 0.28918965, 0.49553488, 0.61543553,
           0.33575905, 0.54168394, 0.15658974, 0.13426621, 0.22442951,
           0.16946512, 0.1953657 , 0.42157698, 0.74126061, 0.87274383,
           0.95759955, 0.83947617, 0.55318222, 0.43796698, 0.1473707 ,
           0.14126038, 0.36556707, 0.24001217, 0.25518879, 0.58937928,
           0.80839603, 0.36331598, 0.38230496, 0.28113962, 0.55396343,
           0.72834572, 0.87771931, 0.64322465, 0.61406436, 0.3262896 ,
           0.09367672, 0.11010181, 0.10681042, 0.13281207, 0.09630662,
           0.41926085, 0.76901883, 0.46624134, 0.37215163, 0.23810044,
           0.38668379, 0.42983361, 0.40827589, 0.21950464, 0.09326784,
           0.10705637, 0.35907413, 0.22196696, 0.73719639, 0.79230004,
           0.81567024, 0.85233467, 0.96740518, 0.73752961, 0.96878856,
           0.96848615, 0.88184152, 0.86665458, 0.89069756, 0.57529245,
           0.13776376, 0.3218207 , 0.49661417, 0.67958821, 0.79617313,
           0.90001448, 0.93731303])




```python
# 样本外准确率
accuracy_out_sample = accuracy_score(y_out_test, y_out_pred)
accuracy_out_sample 
```




    0.6201232032854209




```python
# 样本外AUC值
roc_out_sample = roc_auc_score(y_out_test, y_out_pred)
roc_out_sample
```




    0.6200961376286052



### 练习：集成上述训练模型功能，加入上面类对象里，或者创建新类，集成旧类的功能，在新添功能


```python

```


```python

```

### 答案：翻看类的对应功能函数，查看相应的参数说明


```python
model_timing = report.timing_model('000001.XSHG','2010-01-01','2018-01-01','2018-01-01','2020-01-01','LR','LR_model')
```

    开始获取合成特征和标签数据框...
    ------------------------------------------------------------
    按照比例分割为训练集和测试集...
    ------------------------------------------------------------
    开始训练数据...
    
    
    训练结束
    ------------------------------------------------------------
    预测准确率:
    LR模型: 0.638
    ------------------------------------------------------------
    输出混淆矩阵...
    LR      0.0  1.0
    Actual          
    0.0     113   86
    1.0      55  135
    ------------------------------------------------------------
    绘制曲线...
    

<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_95_1-min.png'/></div><div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_95_2-min.png'/></div>


    ------------------------------------------------------------
    输出评估报告...
    
    
    模型的评估报告：
                   precision    recall  f1-score   support
    
             0.0       0.67      0.57      0.62       199
             1.0       0.61      0.71      0.66       190
    
       micro avg       0.64      0.64      0.64       389
       macro avg       0.64      0.64      0.64       389
    weighted avg       0.64      0.64      0.64       389
    
    ------------------------------------------------------------
    保存模型到本地...
    
    
    加载本地训练好的模型...
    加载完毕
    ------------------------------------------------------------
    样本外测试结果
    样本外准确率 0.6673511293634496
    样本外AUC值 0.666959015010963
    

## 多因子模型

### 数据获取和预处理

### 股票池设定

沪深300成分股，剔除ST股票，剔除每个截面期下一交易日停牌的股票，剔除上市6个月内的股票，每只股票视作一个样本。

### 时间区间
2014年1月1日-2019年12月31日的5年区间。其中前4年区间（48个月）作为训练集，后1年区间（12个月）作为测试集。


```python
#样本内时间段
start_date = '2014-01-01'
end_date = '2014-03-28'
#指数
index_code = '000300.XSHG'
```

### 特征和标签提取
每个自然月的第一个交易日，计算因子暴露度，作为样本的原始特征；计算下期收益率，作为样本的标签


```python
data_regular = report.data_for_model_multiperiod(start_date,end_date,index_code)
data_regular.tail(20)
```


```python
data_regular.shape
```




    (352, 12)




```python
y_train = data_regular['Label']  # 分割
x_train = data_regular.iloc[:,:-1]
```


```python
x_train.tail()
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
      <th>MA_5</th>
      <th>MA_60</th>
      <th>Aroondown_14</th>
      <th>Aroonup_14</th>
      <th>ROC_6</th>
      <th>ROC_12</th>
      <th>RSI_6</th>
      <th>RSI_24</th>
      <th>OBV</th>
      <th>ATR_14</th>
      <th>MOM_10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2012-05-25</th>
      <td>-0.768642</td>
      <td>-0.730667</td>
      <td>1.407126</td>
      <td>-1.385636</td>
      <td>-0.742667</td>
      <td>-0.845272</td>
      <td>-1.066789</td>
      <td>-0.593849</td>
      <td>-0.788446</td>
      <td>-0.653850</td>
      <td>-0.666870</td>
    </tr>
    <tr>
      <th>2013-12-03</th>
      <td>-1.046679</td>
      <td>-1.144522</td>
      <td>-1.119941</td>
      <td>0.685699</td>
      <td>0.556855</td>
      <td>0.965889</td>
      <td>0.907214</td>
      <td>0.662227</td>
      <td>-0.641008</td>
      <td>-0.618633</td>
      <td>0.228764</td>
    </tr>
    <tr>
      <th>2011-10-13</th>
      <td>-0.712226</td>
      <td>-0.354024</td>
      <td>1.212737</td>
      <td>-0.820727</td>
      <td>0.658966</td>
      <td>-0.151954</td>
      <td>0.580085</td>
      <td>-0.727583</td>
      <td>-0.822100</td>
      <td>0.093216</td>
      <td>-0.104109</td>
    </tr>
    <tr>
      <th>2016-04-15</th>
      <td>0.630381</td>
      <td>0.313270</td>
      <td>-0.925552</td>
      <td>0.874002</td>
      <td>0.790899</td>
      <td>1.307709</td>
      <td>0.898470</td>
      <td>0.523861</td>
      <td>1.297079</td>
      <td>0.323287</td>
      <td>0.668458</td>
    </tr>
    <tr>
      <th>2010-02-05</th>
      <td>0.443279</td>
      <td>0.911932</td>
      <td>1.018347</td>
      <td>-1.197333</td>
      <td>-0.714451</td>
      <td>-1.759498</td>
      <td>-1.198944</td>
      <td>-1.187684</td>
      <td>-0.922039</td>
      <td>0.934016</td>
      <td>-1.924716</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_train.tail()
```




    2012-05-25    0.0
    2013-12-03    0.0
    2011-10-13    1.0
    2016-04-15    1.0
    2010-02-05    0.0
    Name: Label, dtype: float64



### 模型构建和样本内训练


```python
#特征
x_train = data_regular.iloc[:,:-1]
#标签
y_train = data_regular['Label']  
```

通过Pipeline方法，将特征选择和模型构建结合起来，形成model_pipe对象，然后针对该对象做交叉验证并得到不同参数下的检验结果，辅助于最终模型的参数设置。

### 特征选择

用SelectPercentile（f_classif, percentile）来做特征选择，其中f_classif用来确定特征选择的得分标准，percentile用来确定特征选择的比例。


```python
transform = SelectPercentile(f_classif)  # 使用f_classif方法选择特征最明显的?%数量的特征

#定义训练器
model = XGBClassifier()
model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])  # 建立由特征选择和分类模型构成的“管道”对象
```


```python
# 选择最佳特征比例
# #############################################################################
# Plot the cross-validation score as a function of percentile of features
score_means = list()
score_stds = list()
percentiles = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100)
for percentile in percentiles:
    model_pipe.set_params(ANOVA__percentile=percentile)
    this_scores = cross_val_score(model_pipe, x_train, y_train, cv=5, n_jobs=-1)
    score_means.append(this_scores.mean())
    score_stds.append(this_scores.std())
plt.errorbar(percentiles, score_means, np.array(score_stds))
plt.title('Performance of the model-Anova varying the percentile of features selected')
plt.xlabel('Percentile')
plt.ylabel('Prediction rate')
plt.axis('tight')
plt.show()
```
<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_114_0-min.png'/></div>

### 交叉验证调参

特征（比例）选择完成后，根据不同的参数（n_estimators,max_depth），对模型进行交叉验证。采用StratifiedKFold来将训练集分成训练集和验证集。StratifiedKFold能够有效结合分类样本标签做数据集分割，而不是完全的随机选择和分割。完成交叉验证后，选取交叉验证集AUC（或f1-score）最高的一组参数作为模型的最优参数。


```python
transform = SelectPercentile(f_classif,percentile=100)  # 使用f_classif方法选择特征最明显的?%数量的特征

model = XGBClassifier()
model_pipe = Pipeline(steps=[('ANOVA', transform), ('model', model)])  # 建立由特征选择和分类模型构成的“管道”对象
```


```python
cv = StratifiedKFold(5)  # 设置交叉检验次数

#XGB max_depth
parameters = [3,4,5,6,7,8]
#XGB subsample
#parameters = [0.6,0.7,0.8,0.9,1]
#score_methods = ['roc_auc','accuracy', 'precision', 'recall', 'f1']  # 设置交叉检验指标
score_methods = ['roc_auc', 'f1']  # 设置交叉检验指标
#mean_list = list()  # 建立空列表用于存放不同参数方法、交叉检验评估指标的均值列表
#std_list = list()  # 建立空列表用于存放不同参数方法、交叉检验评估指标的标准差列表
for parameter in parameters:  # 循环读出每个参数值
    t1 = time.time()  # 记录训练开始的时间
    score_list = list()  # 建立空列表用于存放不同交叉检验下各个评估指标的详细数据
    print ('set parameters: %s' % parameter)  # 打印当前模型使用的参数
    for score_method in score_methods:  # 循环读出每个交叉检验指标
        #model_pipe.set_params(model__n_estimators=parameter)  # 通过“管道”设置分类模型参数
        model_pipe.set_params(model__max_depth=parameter)  # 通过“管道”设置分类模型参数            
        #model_pipe.set_params(model__subsample=parameter)  # 通过“管道”设置分类模型参数            
        score_tmp = cross_val_score(model_pipe, x_train, y_train, scoring=score_method, cv=cv, n_jobs=-1)  # 使用交叉检验计算指定指标的得分
        score_list.append(score_tmp)  # 将交叉检验得分存储到列表
    score_matrix = pd.DataFrame(np.array(score_list), index=score_methods)  # 将交叉检验详细数据转换为矩阵
    score_mean = score_matrix.mean(axis=1).rename('mean')  # 计算每个评估指标的均值
    score_std = score_matrix.std(axis=1).rename('std')  # 计算每个评估指标的标准差
    score_pd = pd.concat([score_matrix, score_mean, score_std], axis=1)  # 将原始详细数据和均值、标准差合并
    #mean_list.append(score_mean)  # 将每个参数得到的各指标均值追加到列表
    #std_list.append(score_std)  # 将每个参数得到的各指标标准差追加到列表
    print (score_pd.round(4))  # 打印每个参数得到的交叉检验指标数据，只保留4位小数
    print ('-' * 60)
    t2 = time.time()  # 计算每个参数下算法用时
    tt = t2 - t1  # 计算时间间隔
    print ('算法用时time: %s' % str(tt))  # 打印时间间隔
```

    set parameters: 3
                  0       1       2       3       4    mean     std
    roc_auc  0.6929  0.6707  0.6939  0.7505  0.7167  0.7050  0.0302
    f1       0.6588  0.6429  0.6667  0.7052  0.7086  0.6764  0.0291
    ------------------------------------------------------------
    算法用时time: 0.6454043388366699
    set parameters: 4
                  0       1       2       3       4    mean     std
    roc_auc  0.6820  0.6653  0.7065  0.7323  0.6920  0.6956  0.0254
    f1       0.6647  0.6551  0.6964  0.6866  0.6964  0.6798  0.0189
    ------------------------------------------------------------
    算法用时time: 0.7695553302764893
    set parameters: 5
                  0       1       2       3       4    mean     std
    roc_auc  0.6813  0.6473  0.6968  0.7254  0.6814  0.6865  0.0283
    f1       0.6667  0.6331  0.6725  0.7038  0.6891  0.6730  0.0266
    ------------------------------------------------------------
    算法用时time: 0.9359838962554932
    set parameters: 6
                  0       1       2       3       4    mean     std
    roc_auc  0.6896  0.6316  0.6802  0.7292  0.6718  0.6805  0.0351
    f1       0.6786  0.6243  0.6531  0.6967  0.6629  0.6631  0.0273
    ------------------------------------------------------------
    算法用时time: 1.0583992004394531
    set parameters: 7
                  0       1       2       3       4    mean     std
    roc_auc  0.6583  0.6381  0.6676  0.7175  0.6581  0.6679  0.0297
    f1       0.6509  0.6126  0.6395  0.7076  0.6552  0.6532  0.0347
    ------------------------------------------------------------
    算法用时time: 1.185107946395874
    set parameters: 8
                  0       1       2       3       4    mean     std
    roc_auc  0.6712  0.6266  0.6603  0.7188  0.6675  0.6689  0.0330
    f1       0.6527  0.6163  0.6388  0.6962  0.6648  0.6538  0.0298
    ------------------------------------------------------------
    算法用时time: 1.3339207172393799
    

### 模型构建

根据上述交叉验证的最优模型，使用XGBoosting集成学习模型对训练集进行训练。


```python
transform.fit(x_train, y_train)  # 应用特征选择对象选择要参与建模的特征变量
X_train_final = transform.transform(x_train)  # 获得具有显著性特征的特征变量
```

#### XGBoost

（1）subsample
subsample是训练集参与模型训练的比例，取值在0-1之间，可有效地防止过拟合。subsample参数的性能评价参考上面执行结果所示。随着subsample的上升，f1-score呈下降趋势，模型训练速度加快，综合训练时间和效果提升考量，选取subsample=0.9。

（2）max_depth
max_depth参数的性能评价参考表所示。随着max_depth的上升，AUC和f1-score呈下降趋势，模型训练时间变慢。选取max_depth=3。


```python
model = XGBClassifier(max_depth=3,subsample=0.9,random_state=0)
model.fit(X_train_final, y_train)  # 训练模型
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=0.9, verbosity=1)



### 样本外测试


```python
#样本外测试
out_start = '2018-01-01'
out_end = '2019-12-31'

test_sample_predict={}
test_sample_score=[]
test_sample_accuracy=[]
test_sample_roc_auc=[]
test_sample_date=[]

interval_start,interval_end = report.get_time_inverval(out_start,out_end,'M')

for date1,date2 in dict(zip(interval_start,interval_end)).items():
    
    data_merge_label = report.data_for_model_perperiod(date1,date2,index_code)
    y_test=data_merge_label['Label']
    X_test=data_merge_label.iloc[:,:-1]

    # 新数据集做预测
    
    # 输出预测值以及预测概率
    y_pred_tmp = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred_tmp, columns=['label_predict'])  # 获得预测标签
    y_pred_proba = pd.DataFrame(model.predict_proba(X_test), columns=['pro1', 'pro2'])  # 获得预测概率
    # 将预测标签、预测数据和原始数据X合并
    y_pred.set_index(X_test.index,inplace=True)
    y_pred_proba.set_index(X_test.index,inplace=True)
    predict_pd = pd.concat((X_test, y_pred, y_pred_proba), axis=1)
    print ('Predict date:')
    print (date1)    
    print ('AUC:')
    print (roc_auc_score(y_test,y_pred))  # 打印前2条结果
    print ('Accuracy:')
    print (accuracy_score(y_test, y_pred))  # 打印前2条结果    
    print ('-' * 60)       
    ## 后续统计画图用
    test_sample_date.append(date1)
    # 样本外预测结果
    test_sample_predict[date1]=y_pred_tmp
    # 样本外准确率
    test_sample_accuracy.append(accuracy_score(y_test, y_pred))   
    # 样本外AUC值
    test_sample_roc_auc.append(roc_auc_score(y_test,y_pred))

print ('AUC mean info')
print (np.mean(test_sample_roc_auc))
print ('-' * 60)    
print ('ACCURACY mean info')
print (np.mean(test_sample_accuracy))
print ('-' * 60)    
```

    Predict date:
    2018-01-01 00:00:00
    AUC:
    0.6055555555555556
    Accuracy:
    0.6055555555555555
    ------------------------------------------------------------
    Predict date:
    2018-02-01 00:00:00
    AUC:
    0.4444444444444445
    Accuracy:
    0.4444444444444444
    ------------------------------------------------------------
    Predict date:
    2018-03-01 00:00:00
    AUC:
    0.5388888888888889
    Accuracy:
    0.5388888888888889
    ------------------------------------------------------------
    Predict date:
    2018-04-01 00:00:00
    AUC:
    0.55
    Accuracy:
    0.55
    ------------------------------------------------------------
    Predict date:
    2018-05-01 00:00:00
    AUC:
    0.5277777777777778
    Accuracy:
    0.5277777777777778
    ------------------------------------------------------------
    Predict date:
    2018-06-01 00:00:00
    AUC:
    0.4444444444444444
    Accuracy:
    0.4444444444444444
    ------------------------------------------------------------
    Predict date:
    2018-07-01 00:00:00
    AUC:
    0.6166666666666667
    Accuracy:
    0.6166666666666667
    ------------------------------------------------------------
    Predict date:
    2018-08-01 00:00:00
    AUC:
    0.3611111111111111
    Accuracy:
    0.3611111111111111
    ------------------------------------------------------------
    Predict date:
    2018-09-01 00:00:00
    AUC:
    0.5166666666666667
    Accuracy:
    0.5166666666666667
    ------------------------------------------------------------
    Predict date:
    2018-10-01 00:00:00
    AUC:
    0.4555555555555556
    Accuracy:
    0.45555555555555555
    ------------------------------------------------------------
    Predict date:
    2018-11-01 00:00:00
    AUC:
    0.75
    Accuracy:
    0.75
    ------------------------------------------------------------
    Predict date:
    2018-12-01 00:00:00
    AUC:
    0.4666666666666666
    Accuracy:
    0.4666666666666667
    ------------------------------------------------------------
    Predict date:
    2019-01-01 00:00:00
    AUC:
    0.4722222222222222
    Accuracy:
    0.4722222222222222
    ------------------------------------------------------------
    Predict date:
    2019-02-01 00:00:00
    AUC:
    0.6777777777777778
    Accuracy:
    0.6777777777777778
    ------------------------------------------------------------
    Predict date:
    2019-03-01 00:00:00
    AUC:
    0.5722222222222222
    Accuracy:
    0.5722222222222222
    ------------------------------------------------------------
    Predict date:
    2019-04-01 00:00:00
    AUC:
    0.40555555555555556
    Accuracy:
    0.40555555555555556
    ------------------------------------------------------------
    Predict date:
    2019-05-01 00:00:00
    AUC:
    0.5111111111111111
    Accuracy:
    0.5111111111111111
    ------------------------------------------------------------
    Predict date:
    2019-06-01 00:00:00
    AUC:
    0.46111111111111114
    Accuracy:
    0.46111111111111114
    ------------------------------------------------------------
    Predict date:
    2019-07-01 00:00:00
    AUC:
    0.5611111111111111
    Accuracy:
    0.5611111111111111
    ------------------------------------------------------------
    Predict date:
    2019-08-01 00:00:00
    AUC:
    0.4833333333333333
    Accuracy:
    0.48333333333333334
    ------------------------------------------------------------
    Predict date:
    2019-09-01 00:00:00
    AUC:
    0.4444444444444444
    Accuracy:
    0.4444444444444444
    ------------------------------------------------------------
    Predict date:
    2019-10-01 00:00:00
    AUC:
    0.5277777777777777
    Accuracy:
    0.5277777777777778
    ------------------------------------------------------------
    Predict date:
    2019-11-01 00:00:00
    AUC:
    0.5722222222222223
    Accuracy:
    0.5722222222222222
    ------------------------------------------------------------
    Predict date:
    2019-12-01 00:00:00
    AUC:
    0.5277777777777778
    Accuracy:
    0.5277777777777778
    ------------------------------------------------------------
    AUC mean info
    0.5206018518518518
    ------------------------------------------------------------
    ACCURACY mean info
    0.5206018518518518
    ------------------------------------------------------------
    

样本外每期AUC


```python
test_sample_roc_auc
```




    [0.6055555555555556,
     0.4444444444444445,
     0.5388888888888889,
     0.55,
     0.5277777777777778,
     0.4444444444444444,
     0.6166666666666667,
     0.3611111111111111,
     0.5166666666666667,
     0.4555555555555556,
     0.75,
     0.4666666666666666,
     0.4722222222222222,
     0.6777777777777778,
     0.5722222222222222,
     0.40555555555555556,
     0.5111111111111111,
     0.46111111111111114,
     0.5611111111111111,
     0.4833333333333333,
     0.4444444444444444,
     0.5277777777777777,
     0.5722222222222223,
     0.5277777777777778]



### 预测能力


```python
xs_date = test_sample_date
ys_auc = test_sample_roc_auc
# 配置横坐标
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.plot(xs_date, ys_auc,'r')
# 自动旋转日期标记
plt.gcf().autofmt_xdate() 
# 横坐标标记
plt.xlabel('date')
# 纵坐标标记
plt.ylabel("test AUC")
plt.show()
```
<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_129_0-min.png'/></div>

```python

xs_date = test_sample_date
ys_score = test_sample_accuracy
# 配置横坐标
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.plot(xs_date, ys_score,'r')
# 自动旋转日期标记
plt.gcf().autofmt_xdate() 
# 横坐标标记
plt.xlabel('date')
# 纵坐标标记
plt.ylabel("test accuracy")
plt.show()
```
<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_130_0-min.png'/></div>

```python

f = plt.figure(figsize= (15,6))

sns.set(style="whitegrid")
data1 = pd.DataFrame(ys_auc, xs_date, columns={'AUC'})
data2 = pd.DataFrame(ys_score, xs_date, columns={'accuracy'})
data = pd.concat([data1,data2],sort=False)
sns.lineplot(data=data, palette="tab10", linewidth=2.5)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x264b0823c88>


<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_131_1-min.png'/></div>
下面特征重要度是这类算法的独有分析部分，不兼容其类机器学习算法

### 特征重要度


```python
model = XGBClassifier(max_depth=3,subsample=0.9,random_state=0)
model.fit(x_train, y_train)  # 训练模型
```




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=None,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=0.9, verbosity=1)




```python
#%matplotlib inline
fig = plt.figure(figsize= (15,6))

n_features = x_train.shape[1]
plt.barh(range(n_features),model.feature_importances_,align='center')
plt.yticks(np.arange(n_features),x_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
```




    Text(0, 0.5, 'Feature')

<img src='http://kan.027cgb.com/627139/bgpc/20200605/output_135_1.png'/>

集成，添加，调用，前面的单元格代码都可删掉


```python
#样本内时间段
start_date = '2014-01-01'
end_date = '2018-12-31'
#指数
index_code = '000300.XSHG'

#样本外时段
out_start = '2018-10-01'
out_end = '2019-12-31'

model_name = 'xgboost'
file_name = 'xgboost_model'
```

## 调用类中的多因子训练和测试模型

前面的单元格代码都可删掉，写入样本内和样本外的开始时间和结束时间，模型名，保存文件的名字（自己创建）


```python
report.multifactor_model(index_code,start_date,end_date,out_start,out_end,model_name,file_name)
```

    开始训练数据...
    
    
    训练结束
    ------------------------------------------------------------
    预测准确率:
    xgboost模型: 0.667
    ------------------------------------------------------------
    输出混淆矩阵...
    col_0  0.0  1.0
    Label          
    0.0     25   14
    1.0     10   23
    ------------------------------------------------------------
    绘制曲线...
    

<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_140_1-min.png'/></div><div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_140_2-min.png'/></div>

    ------------------------------------------------------------
    输出评估报告...
    
    
    模型的评估报告：
                   precision    recall  f1-score   support
    
             0.0       0.71      0.64      0.68        39
             1.0       0.62      0.70      0.66        33
    
       micro avg       0.67      0.67      0.67        72
       macro avg       0.67      0.67      0.67        72
    weighted avg       0.67      0.67      0.67        72
    
    ------------------------------------------------------------
    保存模型到本地...
    
    
    加载本地训练好的模型...
    加载完毕
    ------------------------------------------------------------
    样本外测试结果...
    Predict date:
    2018-10-01 00:00:00
    AUC:
    0.4833333333333334
    Accuracy:
    0.48333333333333334
    ------------------------------------------------------------
    Predict date:
    2018-11-01 00:00:00
    AUC:
    0.6722222222222222
    Accuracy:
    0.6722222222222223
    ------------------------------------------------------------
    Predict date:
    2018-12-01 00:00:00
    AUC:
    0.49444444444444446
    Accuracy:
    0.49444444444444446
    ------------------------------------------------------------
    Predict date:
    2019-01-01 00:00:00
    AUC:
    0.5611111111111111
    Accuracy:
    0.5611111111111111
    ------------------------------------------------------------
    Predict date:
    2019-02-01 00:00:00
    AUC:
    0.6166666666666666
    Accuracy:
    0.6166666666666667
    ------------------------------------------------------------
    Predict date:
    2019-03-01 00:00:00
    AUC:
    0.5277777777777778
    Accuracy:
    0.5277777777777778
    ------------------------------------------------------------
    Predict date:
    2019-04-01 00:00:00
    AUC:
    0.46111111111111114
    Accuracy:
    0.46111111111111114
    ------------------------------------------------------------
    Predict date:
    2019-05-01 00:00:00
    AUC:
    0.5166666666666667
    Accuracy:
    0.5166666666666667
    ------------------------------------------------------------
    Predict date:
    2019-06-01 00:00:00
    AUC:
    0.4555555555555556
    Accuracy:
    0.45555555555555555
    ------------------------------------------------------------
    Predict date:
    2019-07-01 00:00:00
    AUC:
    0.5611111111111111
    Accuracy:
    0.5611111111111111
    ------------------------------------------------------------
    Predict date:
    2019-08-01 00:00:00
    AUC:
    0.5111111111111111
    Accuracy:
    0.5111111111111111
    ------------------------------------------------------------
    Predict date:
    2019-09-01 00:00:00
    AUC:
    0.4555555555555556
    Accuracy:
    0.45555555555555555
    ------------------------------------------------------------
    Predict date:
    2019-10-01 00:00:00
    AUC:
    0.4222222222222222
    Accuracy:
    0.4222222222222222
    ------------------------------------------------------------
    Predict date:
    2019-11-01 00:00:00
    AUC:
    0.5444444444444444
    Accuracy:
    0.5444444444444444
    ------------------------------------------------------------
    Predict date:
    2019-12-01 00:00:00
    AUC:
    0.5666666666666667
    Accuracy:
    0.5666666666666667
    ------------------------------------------------------------
    AUC mean info
    0.5233333333333333
    ------------------------------------------------------------
    ACCURACY mean info
    0.5233333333333333
    ------------------------------------------------------------
    


<div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_140_4-min.png'/></div><div align="center"><img src='http://kan.027cgb.com/627139/bgpc/20200605/output_140_5-min.png'/></div>




    XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
           colsample_bynode=1, colsample_bytree=1, gamma=0, learning_rate=0.1,
           max_delta_step=0, max_depth=3, min_child_weight=1, missing=nan,
           n_estimators=100, n_jobs=1, nthread=None,
           objective='binary:logistic', random_state=0, reg_alpha=0,
           reg_lambda=1, scale_pos_weight=1, seed=None, silent=None,
           subsample=0.9, verbosity=1)




<img src='http://kan.027cgb.com/627139/bgpc/20200605/output_140_7-min.png'/>


```python
report.multifactor_model(index_code,start_date,end_date,out_start,out_end,model_name,file_name)
```


```python

```
