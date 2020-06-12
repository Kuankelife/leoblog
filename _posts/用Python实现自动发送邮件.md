---
layout: post
title:  "Python自动化发送邮件"
subtitle: '人生苦短，我用Python'
date:   2020-06-13
author: "YU"
header-img: "img/yu-img/post-img/kaixinsudi1.jpg"
tags:
  - Python
 
mathjax: False
---
平时有太多小任务需要定期发送邮件，琐碎重复但又不得不做，所以利用python做了一个自动化发送邮件的部署。自动化后，感觉人轻松了很多。


第一个方法比较灵活，第二个方法集成方便。持续更新。

期间借鉴了网上多个教程，不胜感激，程序猿真的是史上最无私的群体之一。
```python
import smtplib
import email
# 负责构造文本
from email.mime.text import MIMEText
# 负责构造图片
from email.mime.image import MIMEImage
# 负责将多个对象集合起来
from email.mime.multipart import MIMEMultipart
from email.header import Header
```
# 第一种方法

```python
# SMTP服务器,这里使用163邮箱
mail_host = "smtp.163.com"
# 发件人邮箱
mail_sender = "****@163.com"
# 邮箱授权码,注意这里不是邮箱密码,如何获取邮箱授权码,请看本文最后教程
mail_license = "*****"
# 收件人邮箱，可以为多个收件人
mail_receivers = ["****@qq.com"]
#mail_receivers = ["1115223619@qq.com","******@outlook.com"]
```


```python
mm = MIMEMultipart('related')
```


```python
# 邮件主题
subject_content = """Python邮件测试"""
# 设置发送者,注意严格遵守格式,里面邮箱为发件人邮箱
mm["From"] = "sender_name<c**o@163.com>"
# 设置接受者,注意严格遵守格式,里面邮箱为接受者邮箱
mm["To"] = "receiver_1_name<**@qq.com>"
#mm["To"] = "receiver_1_name<1115223619@qq.com>,receiver_2_name<******@outlook.com>"
# 设置邮件主题
mm["Subject"] = Header(subject_content,'utf-8')
```


```python
# 邮件正文内容
body_content = """你好，这是一个测试邮件！"""
# 构造文本,参数1：正文内容，参数2：文本格式，参数3：编码方式
message_text = MIMEText(body_content,"plain","utf-8")
# 向MIMEMultipart对象中添加文本对象
mm.attach(message_text)
```


```python
# # 二进制读取图片
# image_data = open('timg.jpg','rb')
# # 设置读取获取的二进制数据
# message_image = MIMEImage(image_data.read())
# # 关闭刚才打开的文件
# image_data.close()
# # 添加图片文件到邮件信息当中去
# mm.attach(message_image)
```


```python
# 构造附件
atta = MIMEText(open('files.html', 'rb').read(), 'base64', 'utf-8')
# 设置附件信息
atta["Content-Disposition"] = 'attachment; filename="files.html"'
# 添加附件到邮件信息当中去
mm.attach(atta)
```


```python
# html
html = """
    <html>
        <body>
            <h1>this is a test,don't scared!</h1>
            <a href = 'https://www.baidu.com'>click it plz!</a>
        </body>

    </html>
"""
msg_html = MIMEText(html, 'html', 'utf-8')
mm.attach(msg_html)

```


```python
# 创建SMTP对象
stp = smtplib.SMTP()
# 设置发件人邮箱的域名和端口，端口地址为25
stp.connect(mail_host, 25)  
# set_debuglevel(1)可以打印出和SMTP服务器交互的所有信息
stp.set_debuglevel(1)
# 登录邮箱，传递参数1：邮箱地址，参数2：邮箱授权码
stp.login(mail_sender,mail_license)
# 发送邮件，传递参数1：发件人邮箱地址，参数2：收件人邮箱地址，参数3：把邮件内容格式改为str
stp.sendmail(mail_sender, mail_receivers, mm.as_string())
print("邮件发送成功")
# 关闭SMTP对象
stp.quit()
```
# 第二种方法
用现成的库yagmail,一键发送。
```python

import yagmail
import os
def join_current_dir(file):
    """Join filepath with current file directory"""
    cwd = os.getcwd()
    return os.path.join(cwd, file)

# 登录你的邮箱
yag = yagmail.SMTP(user = '***@163.com', password = 'h***', host = 'smtp.163.com') 
# 发送邮件
yag.send(to = ['111*****@qq.com'], subject = '邮件的主题', \
         contents = ['我要发送的内容', r'C:\\Users\\dell\\Desktop\\100jfif',r"D:\JoinQuant-Desktop-Py3\USERDATA\.joinquant-py3\notebook\f702a98f83d18867a703599d6c89c2cc\学习代码\用Python实现自动发送邮件\fig1.png"])

yag.send(to = ['1115223619@qq.com'], subject = '邮件的主题', \
         contents = ['我要发送的内容'])
#         [join_current_dir('fig1.png')])
```

