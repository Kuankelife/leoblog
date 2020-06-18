---
layout: post
title: '上证50指数复制下单策略'
subtitle: '一篮子的择机买卖'
date: 2020-06-19
author: 'YU'
header-mask: 0.3
mathjax: true
tags:
    - 量化投资

catalog: false
---


```python
# 导入函数库
from jqdata import *
import pandas as pd
# 初始化函数，设定基准等等
def initialize(context):
    # 设定沪深300作为基准
    set_benchmark('000016.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    log.set_level('order', 'error')
    log.set_level('strategy','info')
    log.set_level('system','error')
    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
    # 开盘时运行
    run_daily(market_open, time='every_bar', reference_security='000016.XSHG')
    g.index = "000016.XSHG"
    #计时器
    g.minute = 0
    #第一次总体下单次数
    g.order_num = 150
    #第二次总体下单次数
    g.order_num2 = 50
    #第一次总体下单预计表
    g.order_decision = 0
    #第二次总体下单预计表
    g.order_decision2 = 0
    g.cangwei = 0
    #股票分配资金字典
    g.code_cash = 0
    #小计时器
    g.time2 = 0
    
    
## 开盘时运行函数
def market_open(context):
    #在开盘后立即获得股票的开盘价，根据分配的资金预计今天在特定区间可买的数目和在每分钟可买的数目，得到第一次总体下单预计表
    if g.minute == 0:
        cash = context.portfolio.total_value
        code_dict = {}
        df = get_index_weights(index_id = g.index, date = context.current_dt)
        code_list = get_index_stocks(g.index, date = context.current_dt)
        for code in code_list:
            code_dict[code] = cash * df['weight'].loc[code]*0.01
        g.code_cash = code_dict
        order_df = pd.DataFrame(code_dict,index = ['cash']).T
        open_list = get_price(list(code_dict.keys()), count = 1, end_date = context.current_dt).open.T
        open_list.columns = ['cash']
        
        code_amount = pd.DataFrame()
        code_amount['cash'] = code_dict.values()
        code_amount.index = open_list.index
        code_hand = code_amount/open_list/100
        code_hand.columns = ['hands']
        
        #每次下单数量
        every_stock_order_num = []
        #下单次数
        every_stock_time = []
        
        remainder = []
        for i,j in dict(zip(code_hand.index.tolist(),list(code_hand.hands.values))).items():
            if j / g.order_num < 1:
                every_stock_order_num.append(1)
                every_stock_time.append(int(j))
                remainder.append(0)
            else:
                every_stock_order_num.append(j // g.order_num)
                every_stock_time.append(g.order_num)
                remainder.append(int(j % g.order_num))
                
                
        order_decision = pd.DataFrame()
        order_decision['every_stock_order_num'] = every_stock_order_num
        order_decision['every_stock_time'] = every_stock_time
        order_decision['remainder'] = remainder
        order_decision.index = code_hand.index.tolist()
        #得到第一次总体下单预计表，设置为全局变量
        g.order_decision = order_decision
        
    #在计算出第一次总体下单预计表后的时间段里，开始每分钟买入相对应数目的股票
    elif 0 < g.minute <= g.order_num:
        for code in g.order_decision.index.tolist():
            if  g.minute <= g.order_decision.loc[code]['every_stock_time']:
                order(code,g.order_decision.loc[code]['every_stock_order_num']*100)
            
    #因为股票价格会有变化，所以在中途会有一个检查节点，检查账户剩余资金多少 ，用剩下的钱得到第二次总体下单预计表
    elif g.order_num < g.minute <= g.order_num + 1:
        order_decision2 = {}
        long_positions_dict = context.portfolio.long_positions
        # print(context.portfolio.long_positions)
        # print(list(long_positions_dict.values()))
        print('------------')
        for position in list(long_positions_dict.values()):  
            #print("标的:{0},总仓位:{1},标的价值:{2}, 建仓时间:{3}".format(position.security, position.total_amount, position.value, position.init_time))
            # print(position)
            current_data = get_current_data() 
            if g.code_cash[position.security] - position.value > 0:
                if (g.code_cash[position.security] - position.value)/(current_data[position.security].last_price)//100 >= 1:
                    order_decision2[position.security] = (g.code_cash[position.security] - position.value)/(current_data[position.security].last_price)/100
                    
                else:
                    order_decision2[position.security] = 0
            else:
                order_decision2[position.security] = 0
        #得到第二次总体下单预计表，设置为全局变量  
        g.order_decision2 = order_decision2
    
  
    #根据第二次总体下单预计表，每分钟买入一定数目单数。
    elif g.order_num + 1 < g.minute <= g.order_num+1 + g.order_num2+1:    
        g.time2 = 1
        for code in g.order_decision.index.tolist():
            if g.order_decision2[code] > 0:
                if g.order_decision2[code]//g.order_num2 < 1:
                    if g.time2 < int(g.order_decision2[code]):
                        order(code,100)
                
                else:
                    
                    num = g.order_decision2[code]//g.order_num2
                    order(code,num * 100)
        if g.time2 == g.order_num2+1:
            for code in g.order_decision.index.tolist():
                res = g.order_decision2[code]%g.order_num2
                if g.order_decision2[code]//g.order_num2 >=1:
                    if res > 0:
                        order(code,res * 100)
        g.time2 += 1

    #在操作完后，检查当前未完成订单，循环撤销订单
    elif g.minute == g.order_num+1 + g.order_num2 + 5:
        # 得到当前未完成订单
        orders = get_open_orders()
        # 循环，撤销订单
        for _order in orders.values():
            cancel_order(_order)

    #这个时候剩下资金已经不多， 根据买入列表，每分钟买入每支股票一手，直到资金不足一万且买不了任何一只股票一手为止。
    elif g.order_num+1 + g.order_num2 + 6 < g.minute < 238:  
        if context.portfolio.available_cash > 10000:
            for code, cash_value in g.code_cash.items():
                current_data = get_current_data() 
                if context.portfolio.available_cash > current_data[code].last_price *100:
                    order(code,100)
    
     
            
    #检查剩余资金,打印今天未完成的订单
    if g.minute == 238:
        #打印今天预计要买入的各股票价值
        log.info('今天预计要买入的各股票价值',g.code_cash)
        #打印今天成交的各股票总仓位，标的价值
        for position in list(context.portfolio.long_positions.values()):  
            log.info("标的:{0},总仓位:{1},建仓标的价值:{2}, 建仓时间:{3}".format(position.security, position.total_amount, position.value, position.init_time))
        #得到还没有完成的订单
        log.info('未完成的订单',get_open_orders())
        #剩余资金
        log.info('剩余资金',context.portfolio.available_cash)
       
            
    
 
    g.minute += 1
   



```
