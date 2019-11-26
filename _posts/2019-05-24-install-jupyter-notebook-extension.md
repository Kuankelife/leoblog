---
layout:     post
title:  "在joinquant金融终端jupyter notebook安装扩展插件方法"
subtitle:   ""
date:       2019-05-24 19:55:00
author:     "YU"
header-img: "img/yu-img/post-img/u=3455037329,48387545&fm=27&gp=0.jpg"
catalog:    false
tags:
    - 量化投资

---


最近强迫症发作，就想着在joinquant金融终端里安装扩展extension时，在聚宽工作人员的指导下，折腾了一番，搞定了，现在把我的步骤分享一下。


## 1.在本地安装Anaconda, 并安装jupyter extension插件
需要在本地安装Anaconda，安装教程非常简单，就是傻瓜操作，就不再多说，安好anaconda后, 点击左下角的开始，可以看到下图里添加了Anaconda3

![图一]( https://image.joinquant.com/94c32b85631719557a0f321f2d826f4a) 


- 点击上图里红圈anconda3(64bit), 可以看到有Anaconda Prompt

![Img]( https://image.joinquant.com/dbc9d66677758bfa05ca7b85692cdf5c) 

- 点击Anaconda Prompt，然后安装jupyter notebook扩展插件，官方安装教程在[这里](https://github.com/Jupyter-contrib/jupyter_nbextensions_configurator)。

中文教程在[这里](https://www.cnblogs.com/huabaixiaoduanku/p/7858479.html)
- 安装好Anaconda后，

## 2.在金融终端设置jupyter extension插件
同样两条命令，图省事的看下面的图敲命令吧
![Img]( https://image.joinquant.com/5970346fd7dd41d175611711142d47de) 
![Img]( https://image.joinquant.com/014a2d5ec0cf4e48d2d6823d3ba068d4) 
可以从本地的路径C:\ProgramData\jupyter里面的文件夹nbextensions到D:\JoinQuant-Desktop-Py3\USERDATA\.jupyter_jq，


- 本地夹位置如图![Img]( https://image.joinquant.com/973fa2653fedddb6e80d61aa043b5f42) 


- 复制到金融终端相应目录如图
![Img]( https://image.joinquant.com/9999c6bc53020246899ed12b40f7f96d) 


- 然后打开金融终端，成功了。![Img]( https://image.joinquant.com/8f415ddd0f6e6cccd6b3410729041424) ![Img]( https://image.joinquant.com/014a2d5ec0cf4e48d2d6823d3ba068d4) 

如果有任何疑问，可以通过<a href="mailto:1115223619@qq.com"> 邮件 </a>或者[这里的联系方式](https://ownyulife.top/contact/)联系我。