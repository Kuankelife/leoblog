---
layout: post
title:  '代码复用技术-学习面向对象编程'
subtitle: '随着我们写的代码越来越长，最终导致了越来越难以维护。为了可维护代码，分模块的代码编写是必要的'
date:   2020-11-14
author: "YU"
header-style: text
tags:
  - python
mathjax: False
---
![2401](http://5b0988e595225.cdn.sohucs.com/images/20171004/4406f28d143d47779a40ef53b3ea9171.jpeg)


到目前为止，其实我们写的python代码都是基于对象的，我们在脚本中传递对象，在表达式中使用对象和调用对象的方法等等，但是要让代码真正称得上是面向对象的（Object-Oriented),那么对象一般需要参与到所谓的继承的层次中。


在python中，面向对象的编程完全可以忽略，是可选的，因为初级阶段不需要使用类，实际上，利用函数的结构几乎可以帮助完成大部分的编程工作。由于妥善使用类需要一些预先的规划，而这种规划可以将代码分解，通过定制和复用使得代码的冗余和开发时间减少，提高了代码的可使用性，所以，类是Python能提供的最有用的工具之一。



#  类和实例

**类:** 用来描述具有相同的属性和方法的对象的集合。它定义了该集合中每个对象所共有的属性和方法。对象是类的实例。

**对象：**通过类定义的数据结构实例。对象包括两个数据成员（类变量和实例变量）和方法。

**类变量：**类变量在整个实例化的对象中是公用的。类变量定义在类中且在函数体之外。类变量通常不作为实例变量使用。

**实例变量：** 定义在方法中的变量，只作用于当前实例的类。

对“类”和“对象”的使用：
- 类就是一个模板，模板里可以包含多个函数，函数里实现一些功能。
- 对象则是根据模板创建的实例，通过实例对象可以执行类中的函数。

首先认识类的最简单形式


```python
#创建类
class FirstClass(object):
    # 类中的函数
    def first_function(self):
        #功能阐述
        pass

#根据FirstClass创建对象obj,即实例化
obj = FirstClass()
```

> 注意：创建对象的时候 记得后面加个括号

> 注意，按照Python通用规则，Class用驼峰式表示(HelloWorld),而其他的obj等等，都用一个下横线隔开(this_is_object),类中的函数第一个参数必须是self，类中定义的函数叫做“方法”。


```python
#重写类，增添两个方法
class FirstClass(object):
     
    def first_function(self):
        print('你好，python')
 
    def hello(self, name):
        print('你好 %s!' %name)
```


```python
#创建对象
obj = FirstClass()
obj.first_function()            #执行First_function方法
obj.hello('yu') #执行hello方法　
```

    你好，python
    你好 yu!


> 注意：self是为了指代它所存在的类Class之中。比如我们如果有好几个不同的obj被创建成同一个类，那么有了self，我们的class FirstClass就能很好的知道哪个指的是自己。


```python
class FirstClass(object):
    # 这里我们可以创建一个类级别的变量
    # 它不会随着由此类创建的变量而变化
    name = '中国'
    
    def first_function(self):
        print('深圳')
 
    def hello(self, name):
        print('你好 %s' %self.name)
        print('%s人' %name)
```


```python
#创建对象
obj1 = FirstClass()
obj2 = FirstClass()
obj1.hello('中国')
obj2.hello('深圳') 
```

    你好 中国
    中国人
    你好 中国
    深圳人


> 所以说，这个 self 就是个代指。代指了自己所在的class，类似于英语语法里的反身代词。你可以由 self 点进所指class本身的函数。由此可见，self 本身作为一个代词，并不一定要叫self。你也可以用个其他什么来代替。只不过，必须得是这个类的所有子方法的第一个参数


```python
class FirstClass(object):
    # 这里我们可以创建一个类级别的变量
    # 它不会随着由此类创建的变量而变化
    name = '中国'
    
    def first_function(other_name):#这里把self改成了other_name
        # 只要它作为第一参数的位置没变，它依旧是类FirstClass的自我指代
        print('深圳') 
    def hello(other_name, name):
        print('你好 %s' %other_name.name)
        print('%s人' %name) 

```

> self 本身作为一个代词，并不一定要叫self


```python
#创建对象
obj1 = FirstClass()
obj2 = FirstClass()
obj1.hello('中国')
obj2.hello('深圳') 
```

    你好 中国
    中国人
    你好 中国
    深圳人


**构造函数：** 构造函数，是一种特殊的方法。主要用来在创建对象时初始化对象， 即为对象成员变量赋初始值。

跟所有OOP语言一样，python也是有构造函数的，默认为:


```python
class FirstClass(object):
    
    def __init__(self):#这就是构造函数，它的职责是在模型创建的初期，就完成定义和赋值等
        #即自定义的初始化步骤：
        #同样，它需要self来指代本身这个class
        self.name='中国'
 
    def hello(self, name):
        print('你好，%s' %self.name)
        print('你好，%s' %name) 
        print('\n')

#每次创建一个FirstClass类的实例的时候，init会被自动跑一遍：
obj = FirstClass()
# 默认给self自己的name变量，赋值为'中国'
# 此刻，当我们调用FirstClass的hello()方法时，赋值为'深圳'
obj.hello('深圳')
```

    你好，中国
    你好，深圳
    
    


init是可以带更多的参数的，用以初始化我们的class本身。

比如说，你要初始化一个类的时候要用到一些外部参数:


```python
# 创建类
class FirstClass(object):
    
    def __init__(self, name2):# 你可以在这里附加上一些参数
        # 这些参数将是创建一个FirstClass类时的必要条件
        self.name=name2
 
    def hello(self, name):
        print('你好，%s' %self.name)
        print('你好，%s' %name) 
        print('\n')

#需要填入一个参数：name2
obj = FirstClass('广东')
#调用FirstClass的hello()方法时，赋值为'深圳'
obj.hello('深圳')
```

    你好，广东
    你好，深圳
    
    


通过这些例子，Python的OOP概念大概可以清楚：

Class(类)就是一个把一堆Object(对象)集合起来的地方，在其中有方法和属性

# 访问限制

在调用obj的时候，可以直接调出name或者使用hello()。那么怎么知道什么时候可以调用他们，什么时候不可以呢？

在Class内部，可以有属性和方法，而外部代码可以通过直接调用实例变量的方法来操作数据，这样，就隐藏了内部的复杂逻辑。如果要让内部属性不被外部访问，可以把属性的名称前加上两个下划线__，在Python中，实例的变量名如果以__开头，就变成了一个私有变量（private），只有内部可以访问，外部不能访问。

比如，创建一个类去储存一个人的信息，在外部可以访问到name,其实是可以修改name,这是不安全的。


```python
class Person(object):
    
    def __init__(self, name, age):
        self.name = name
        self.age = age  
    
    def information(self):
        print(self.name)
        print(self.age)
        
```


```python
# 输入name,age两个值，创建实例
person_1 = Person('小强', 3)
```


```python
# 访问小强的年龄
person_1.age
```




    3



这时小强年龄是3岁


```python
#修改年龄为10
person_1.age = 10
person_1.age
```




    10




```python
person_1.information()
```

    小强
    10


这是小强年龄是10岁

为了防止这种篡改年龄的操作发生，需要通过设置将一些信息隐藏起来,即在想要隐藏的信息前面加两个下横线


```python
class Person(object):
    
    def __init__(self, name, age):
        self.__name = name
        self.__age = age  
    
    def information(self):
        print(self.__name)
        print(self.__age)

```


```python
person_1 = Person('小强', 3)
```


```python
person_1.__age = 99
person_1.information()
```

    小强
    3


小强年龄并没有被改动

那么如何既保证安全，又能被外部修改呢？应该使用OOP家族传统理念：Getter+Setter

同时，下面应该学会使用Class来定义我们自己的类了

# 面向对象的特性

面向对象的三大特性是指：封装、继承和多态。

## 封装

指的就是把内容封装到某个地方，用于日后调用

它需要：

- 把内容封装在某处
- 从另一处调用被封装的内容

通过对象直接调用，我们可以在存完一个内容以后，在类以外的地方，通过这个类的对象，来直接”点“调用


```python
class Person(object):
    # 初始化Person时，记录下每个人的名字和年龄
    def __init__(self, name, age):
        self.name = name
        self.age = age  
    #用name和age存下了他们的年龄和名字
    
```


```python
obj1 = Person('张三', 15)
print(obj1.name)    # 直接调用obj1对象的name属性
print(obj1.age)   # 直接调用obj1对象的age属性
obj2 = Person('李四', 32)
print(obj2.name)    # 直接调用obj2对象的name属性
print(obj2.age)     # 直接调用obj2对象的age属性
```

    张三
    15
    李四
    32


通过self间接调用，执行类中某一个方法时，通过self来调用了类自己的变量


```python
class Person(object):
    
    def __init__(self, name, age):
        self.name = name
        self.age = age  
    
    def information(self):
        print(self.name)
        print(self.age)
```


```python
obj1 = Person('张三', 15)
obj1.information() #Python默认将obj1传给self，所以其实这里做的是obj1.information(obj1)

obj2 = Person('李四', 32)
obj2.information()
```

    张三
    15
    李四
    32


> 综上所述，对于面向对象的封装来说，其实就是使用构造方法将内容封装到 对象 中，然后通过对象直接或者self间接获取被封装的内容。

## 继承

继承，面向对象中的继承和现实生活中的继承相同，即：子可以继承父的功能和属性。

比如，每个人都有名字和年龄，这个定义的类可以作为父类

但是，每个人都可能有不同的”方法“，即每个人都有不同的特长和职业等等。


```python
# 我们首先创建一个学生类，这个类是所有学生的爸爸
class Person(object):
    
    def __init__(self, name, age):
        self.name = name
        self.age = age  
    
    def information(self):
        print(self.name)
        print(self.age)

# 然后，我们创建一个子类，子类，顾名思义，会继承父类Person的方法和属性
class Son(Person):#因为是继承于学生类，所以我们写在括号内
    # 这里我们可以不写构造函数，于是我们就是直接沿用Person类的构造函数
    def lol(self): # 我们有一些新的独有的方法，会被叠加起来
        print('不服sala！')
    
# 接下来，我们创建一个另外一个子类，这个子类有自己新的特点
class Daughter(Person):
    def __init__(self, name, age,hobby): #这里，我们改写一下构造函数
        # 于是爸爸的init会被直接overwrite
        self.name = name
        self.age = age
        self.hobby = hobby
    def more_information(self):
        print(self.hobby)

```

> 注意，子类的定义方式是：class子类名(父类名)，不是class子类名(object)


```python
#实例化
obj1 = Son('小东', 18)
obj1.lol() # 独有的方法
obj1.information()#继承与爸爸的方法
```

    不服sala！
    小东
    18



```python
#实例化
obj2 = Daughter('小雨', 20, '看书')
obj2.information()
obj2.more_information()
```

    小雨
    20
    看书


所以，对于面向对象的继承来说，其实就是将多个类共有的方法提取到父类中，子类仅需继承父类而不必一一实现每个方法。

这样可以极大的提高效率，减少代码的重复。

问题来了，如果我想多认个干爹呢？
Python和Java/C#的不同就是，Python可以多类继承，也就是，可以认很多干爹

其实这有经典类和新类之分，这是一个历史遗留问题，但python3推荐写法是新类，当本身的类是新式类的时候，就按照广度优先的方式查找 （即，找到一个爸爸，再找下一个爸爸，再找下一个爸爸，平辈之间查找），故这里只讨论新类的写法,在这个文件中，类的写法都是新类的写法


```python
class D(object):

    def bar(self):
        print('D.bar')


class C(D):

    def bar(self):
        print('C.bar')


class B(D):

    pass

class A(B, C):
    pass
```


```python
a = A()
# 执行bar方法时
# 首先去A类中查找，如果A类中没有，则继续去B类中找，如果B类中么有，则继续去C类中找，如果C类中么有，则继续去D类中找，如果还是未找到，则报错
# 所以，查找顺序：A --> B --> C --> D
# 在上述查找bar方法的过程中，一旦找到，则寻找过程立即中断，便不会再继续找了
a.bar()
```

    C.bar


Python中类的初始化都是__init__(), 所以父类和子类的初始化方式都是__init__(), 但是如果子类初始化时没有这个函数，那么他将直接调用父类的__init__(); 如果子类指定了__init__(), 就会覆盖父类的初始化函数__init__()，如果想在进行子类的初始化的同时也继承父类的__init__(), 就需要在子类中显示地通过super()来调用父类的__init__()函数。


```python
class Father(object):  # 定义一个父类
    def __init__(self):  # 父类的初始化
        self.name = '大强'
        self.role = 'parent'
        print('I am father')

class son(Father):  # 定一个继承Father的子类
    def __init__(self):  # 子类的初始化函数，此时会覆盖父类Father类的初始化函数
        super(son, self).__init__()  # 在子类进行初始化时，也想继承父类的__init__()就通过super()实现,此时会对self.name= 'animal'
        print('I am son')
        self.name = '小强'  # 定义子类的name属性,并且会把刚才的self.name= '大强'更新为'小强'

```


```python
xbai = son()#I am father,I am son
print(xbai.name)#'dog'
print(xbai.role)#'parent'
```

    I am father
    I am son
    小强
    parent


## 多态

Pyhon不支持多态并且也用不到多态，多态的概念是应用于Java和C#这一类强类型语言中

不同于强类型的语言，一个类型的obj只能一种事儿，

在Python中，只要是能“不报错运行”的类型，都可以放进参数中去


```python
class F1(object):
    pass

# 假设，S1是我们的正统类，它继承于根正苗红的F1，是我们的正统类
class S1(F1):
    def show(self):
        print('S1.show')

# S2是路人甲，是个歪瓜裂枣，但是他自己也有一个叫show的方法。
class S2:
    def show(self):
        print('S2.show')
        
        
# 在Java或C#中定义函数参数时，必须指定参数的类型，也即是说，我们如果用
# Java写下面的Func，需要告知，obj是F1类还是其他什么东西。
# 如果限定了F1，那么S2是不可以被采纳的。
# 然而，在Python中，一切都是Obj，它不care你到底是什么类，直接塞进去就可以

def Func(obj):
    """Func函数需要接收一个F1类型或者F1子类的类型"""
    obj.show()
    
s1_obj = S1()
Func(s1_obj) # 在Func函数中传入S1类的对象 s1_obj，执行 S1 的show方法，结果：S1.show

s2_obj = S2()
Func(s2_obj) # 在Func函数中传入Ss类的对象 ss_obj，执行 Ss 的show方法，结果：S2.show
```

    S1.show
    S2.show


## 获取对象信息
当我们拿到一个对象的引用时，如何知道这个对象是什么类型、有哪些方法呢？

**type()**


```python
type(123)
```




    int




```python
type('str')
```




    str




```python
type(None)
```




    NoneType




```python
type(abs)
```




    builtin_function_or_method




```python
class a(object):
    def __init__(self):
        pass 

type(a)
```




    type



如何用语句判断是不是一种type呢？


```python
type(123)==type(456)
```




    True




```python
type('abc')==type('123')
```




    True




```python
type('abc')==type(123)
```




    False




```python
type('abc')==str
```




    True




```python
type([])==list
```




    True



**isinstance()**

isinstance()可以告诉我们，一个对象是否是某种类型（包括继承关系）。


```python
class A(object):
    pass

class B(A):
    pass

class C(B):
    pass

```


```python
k=A()
g=B()
y=C()

```


```python
isinstance(y, C)
```




    True




```python
isinstance(y, B)
```




    True



同理，isinstance()也可以当type()用


```python
isinstance('a', str)
```




    True



**dir()**

如果要获得一个对象的所有属性和方法，可以使用dir()函数，它返回一个包含字符串的list，比如，获得一个str对象的所有属性和方法：


```python
dir('ABC')
```




    ['__add__',
     '__class__',
     '__contains__',
     '__delattr__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__getitem__',
     '__getnewargs__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__iter__',
     '__le__',
     '__len__',
     '__lt__',
     '__mod__',
     '__mul__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__rmod__',
     '__rmul__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     'capitalize',
     'casefold',
     'center',
     'count',
     'encode',
     'endswith',
     'expandtabs',
     'find',
     'format',
     'format_map',
     'index',
     'isalnum',
     'isalpha',
     'isdecimal',
     'isdigit',
     'isidentifier',
     'islower',
     'isnumeric',
     'isprintable',
     'isspace',
     'istitle',
     'isupper',
     'join',
     'ljust',
     'lower',
     'lstrip',
     'maketrans',
     'partition',
     'replace',
     'rfind',
     'rindex',
     'rjust',
     'rpartition',
     'rsplit',
     'rstrip',
     'split',
     'splitlines',
     'startswith',
     'strip',
     'swapcase',
     'title',
     'translate',
     'upper',
     'zfill']



> 类似__xxx__的属性和方法在Python中都是有特殊用途的，比如__len__方法返回长度。在Python中，如果你调用len()函数试图获取一个对象的长度，实际上，在len()函数内部，它自动去调用该对象的__len__()方法，所以，下面的代码是等价的.


```python
'ABC'.__len__()
```




    3



我们自己写的类，如果也想用len(myObj)的话，就自己写一个__len__()方法：


```python
class MyClass(object):
    def __len__(self):
        return 100
```


```python
obj = MyClass()
len(obj)
```




    100



仅仅把属性和方法列出来是不够的，配合getattr()、setattr()以及hasattr()，我们可以直接操作一个对象的状态：


```python
class MyClass(object):
    def __init__(self):
        self.x = 9
    def power(self):
        return self.x * self.x
```


```python
obj = MyClass()
```

可以测试该对象的属性


```python
hasattr(obj, 'x') #有木有属性'x'
```




    True




```python
obj.x
```




    9




```python
hasattr(obj, 'y') # 有属性'y'吗？
```




    False




```python
setattr(obj, 'y', 19) # 设置一个属性'y'
```


```python
hasattr(obj, 'y') # 有属性'y'吗？
```




    True




```python
getattr(obj, 'y') # 获取属性'y'
```




    19




```python
obj.y # 获取属性'y'
```




    19



可以传入一个default参数，如果属性不存在，就返回默认值：


```python
getattr(obj, 'z', 404) # 获取属性'z'，如果不存在，返回默认值404
```




    404



也可以获得对象的方法：


```python
hasattr(obj, 'power') # 有属性'power'吗？
```




    True




```python
getattr(obj, 'power') # 获取属性'power'
```




    <bound method MyClass.power of <__main__.MyClass object at 0x7f36963f1898>>




```python
fn = getattr(obj, 'power') # 获取属性'power'并赋值到变量fn
```


```python
fn # fn指向obj.power
```




    <bound method MyClass.power of <__main__.MyClass object at 0x7f36963f1898>>




```python
fn() # 调用fn()与调用obj.power()是一样的
```




    81



## 实例属性和类属性

由于Python是动态语言，根据类创建的实例可以任意绑定属性。

给实例绑定属性的方法是通过实例变量，或者通过self变量：


```python
class Student(object):
    def __init__(self, name):
        self.name = name

s = Student('小明')
s.score = 90
```

但是，如果Student类本身需要绑定一个属性呢？可以直接在class中定义属性，这种属性是类属性，归Student类所有。


```python
class Student(object):
    name = 'Student'
```

实例与类的属性差异


```python
class Person(object):
    name = '无名'

s = Person() # 创建实例s
print(s.name) # 打印name属性，因为实例并没有name属性，所以会继续查找class的name属性
```

    无名



```python
print(Person.name) # 打印类的name属性
```

    无名



```python
s.name = '小李' # 给实例绑定name属性
print(s.name) # 由于实例属性优先级比类属性高，因此，它会屏蔽掉类的name属性
```

    小李



```python
print(Person.name) # 但是类属性并未消失，用Student.name仍然可以访问
```

    无名



```python
del s.name # 如果删除实例的name属性
```


```python
print(s.name) # 再次调用s.name，由于实例的name属性没有找到，类的name属性就显示出来了
```

    无名


注意：从上面的例子可以看出，在编程的时候，千万不要把实例属性和类属性使用相同的名字，因为相同名称的实例属性将屏蔽掉类属性，但是当你删除实例属性后，再使用相同的名称，访问到的将是类属性。

### 模块和包

如果上面的内容理解了，那么就可以开始编写自己的模块和包了。


Python的程序由包（package）、模块（module）和函数组成。包是由一系列模块组成的集合。模块是处理某一类问题的函数和类的集合。

包就是一个完成特定任务的工具箱，Python提供了许多有用的工具包，如字符串处理、图形用户接口、Web应用、图形图像处理等。这些自带的工具包和模块安装在Python的安装目录下的Lib子目录中。

> 注意：
包必须至少含有一个__init__.py文件按，该文件的内容可以为空。__init__.py用于标识当前文件夹是一个包。

### 模块

在python中一个文件可以被看成一个独立模块，而包对应着文件夹，模块把python代码分成一些有组织的代码段，通过导入的方式实现代码重用。

导入模块时，是按照sys.path变量的值搜索模块，sys.path的值是包含每一个独立路径的列表，包含当前目录、python安装目录、PYTHONPATH环境变量，搜索顺序按照路径在列表中的顺序（一般当前目录优先级最高）。

想看自己的Python路径，大家可以

    import sys

    print(sys.path)

如果你发现你在某个地方写的文件（包）import错误，你就可以看看这个sys.path是否囊括了你那批文件的根目录。


```python
import sys
print(sys.path)
```

    ['/opt/conda/lib/python3.6/jqcommon', '/opt/conda/lib/python36.zip', '/opt/conda/lib/python3.6', '/opt/conda/lib/python3.6/lib-dynload', '', '/home/jquser/.local/lib/python3.6/site-packages', '/opt/conda/lib/python3.6/site-packages', '/opt/conda/lib/python3.6/site-packages/IPython/extensions', '/home/jquser/.ipython']


### 导入模块

使用import语句（不管是你自己写的，还是你下载的别人的）

    import module1

    import module2

    import module3

    import module1,module2,module3

这两种方式的效果是一样的，但是第一种可读性比第二种好，推荐按照下面的顺序导入模块，并且一般在文件首部导入所有的模块

    python标准库

    第三方模块

    自定义模块

### 使用from-import语句导入模块的属性

单行导入

    from module import name1,name2,name3

多行导入

    from module import name1,name2,name3
导入全部属性（由于容易覆盖当前名称空间中现有的名字，所以一般不推荐使用，适合模块中变量名很长并且变量很多的情况）

    from module import *

### 自定义导入模块名称

就是为了用的时候方便好记。

    import mymodule as m

### 包

包将有联系的模块组织在一起，有效避免模块名称冲突问题，让应用组织结构更加清晰。 一个普通的python应用程序目录结构：

    app/
    __init__.py
    a/
    __init__.py
    a.py
    b/
    __init__.py
    b.py
app是最顶层的包，a和b是它的子包，可以这样导入：

    from app.a import a
    from app.b.b import test

    a.test()
    test()
上面代码表示：

导入app包的子包a和子包b的属性test，然后分别调用test方法。
每个目录下都有__init__.py文件，这个是初始化模块，from-import语句导入子包时需要它，可以在里面做一些初始化工作，也可以是空文件。ps：__init__.py定义的属性直接使用 顶层包.子包 的方式导入，如在目录a的__init__.py文件中定义init_db()方法，调用如下：

    from app import a

    a.init_db()

# 自定义的类、模块、包

现在已经完全掌握了使用包的方法，自己定义类，并生成一个可运行的程序

## 例子1
编写一个自动获取数据，并且按一定比例把数据集划分为训练集和测试集的类。


```python
#导入需要用到的库
from sklearn import svm, datasets

#编写一个自动获取数据，并且按一定比例把数据集划分为训练集和测试集的类。
class DataSpilt(object):
    # 我们创造一个dataset的类，这个类会帮我们下载相关的数据集，
    # 并给我们分类好x,y
    def __init__(self, name):
        # 告诉类，我们需要哪一个数据集
        # 我们有两个选择，一个是'iris'一个是'digits'
        self.name = name
        
    def download_data(self):
        # 从sklearn的自带集中下载我们指定的数据集
        if self.name == 'iris':
            # 这里是sklearn自带的数据集下载方法，更多信息可以参照官网
            self.downloaded_data = datasets.load_iris()
        elif self.name == 'digits':
            self.downloaded_data = datasets.load_digits()
        else:
            # 如果不是我们预想的两种数据集，则报错
            print('Dataset Error: No named datasets')
    
    def generate_xy(self):
        # 通过这个过程来把我们的数据集分为原始数据以及他们的label
        # 我们先把数据下载下来
        self.download_data()
        x = self.downloaded_data.data
        y = self.downloaded_data.target
        print('\nOriginal data looks like this: \n', x)
        print('\nLabels looks like this: \n', y)
        return x,y
    
    def get_train_test_set(self, ratio):
        # 这里，我们把所有的数据分成训练集和测试集
        # 一个参数要求我们告知，我们以多少的比例来分割训练和测试集
        # 首先，我们把XY给generate出来：
        x, y = self.generate_xy()
        
        # 有个比例，我们首先得知道 一共有多少的数据
        n_samples = len(x)
        # 于是我们知道，有多少应该是训练集，多少应该是测试集
        n_train = int(n_samples * ratio)
        # 好了，接下来我们分割数据
        X_train = x[:n_train]
        y_train = y[:n_train]
        X_test = x[n_train:]
        y_test = y[n_train:]
        # 好，我们得到了所有想要的玩意儿
        return X_train, y_train, X_test, y_test
```

调用我们自己写的类


```python
# 比如，我们使用digits数据集
data = DataSpilt('digits')
# 接着，我们可以用0.7的分割率把x和y给分割出来
X_train, y_train, X_test, y_test = data.get_train_test_set(0.7)
```

    
    Original data looks like this: 
     [[0.0 0.0 5.0 ... 0.0 0.0 0.0]
     [0.0 0.0 0.0 ... 10.0 0.0 0.0]
     [0.0 0.0 0.0 ... 16.0 9.0 0.0]
     ...
     [0.0 0.0 1.0 ... 6.0 0.0 0.0]
     [0.0 0.0 2.0 ... 12.0 0.0 0.0]
     [0.0 0.0 10.0 ... 12.0 1.0 0.0]]
    
    Labels looks like this: 
     [0 1 2 ... 8 9 8]



```python
X_train
```




    array([[0.0, 0.0, 5.0, ..., 0.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, ..., 10.0, 0.0, 0.0],
           [0.0, 0.0, 0.0, ..., 16.0, 9.0, 0.0],
           ...,
           [0.0, 0.0, 0.0, ..., 3.0, 0.0, 0.0],
           [0.0, 0.0, 5.0, ..., 11.0, 3.0, 0.0],
           [0.0, 0.0, 3.0, ..., 0.0, 0.0, 0.0]])




```python
y_train
```




    array([0, 1, 2, ..., 4, 3, 1])



## 例子2

 示范了累计收益图，月度收益热力图，回撤图的写法，并且对参数和格式做出了规范的解释说明


```python
#导入需要用到的库
import matplotlib.pyplot as plt
from matplotlib import figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.ticker import FuncFormatter
import pandas as pd
import numpy as np
import matplotlib
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import empyrical as ep
import seaborn as sns
```

### **编写时间序列分析的模块**


```python
#时间序列分析的模块
class TsAnalyze(object):
    #初始化
    def __init__(self):
        pass
    
    
    def percentage(self,x, pos):
        """
        将百分比符号添加到坐标轴刻度，画图用到
        """
        return '%.0f%%' % x
    
    
    def plot_accumulated_returns(self,returns,
                             benchmark_returns=None,
                             logy=False,
                             legend_loc='best',
                             ax=None, **kwargs):
        """
        画出累计收益的曲线图。

        参数
        ----------
        returns : pandas里面的序列格式
            每日收益，记住不是累计收益。
        benchmark_returns : pandas里面的序列格式, 可选。
        logy : 布尔值，可选
            是否对y轴进行对数缩放。
        legend_loc : matplotlib图例的位置格式，可选
            matplotlib图例的位置.

        ax : matplotlib的轴,可选
            用来绘图的轴。
        **kwargs,可选
            传输到绘图函数的其他参数。

        返回
        -------
        ax : ax的对象
        """

        if ax is None:
            plt.style.use("ggplot")
            plt.figure(figsize = (15,6))
            ax = plt.gca()

        cum_rets = ep.cum_returns(returns, 1.0)
        y_axis_formatter = FuncFormatter(self.percentage)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        if benchmark_returns is not None:
            cum_benchmark_returns = ep.cum_returns(
                benchmark_returns[cum_rets.index], 1.0)
            cum_benchmark_returns.plot(lw=1.5, color='blue',
                                    label='基准收益', alpha=1.0,
                                    ax=ax, **kwargs)

        cum_rets.plot(lw=1.5, color='red', alpha=1.0,
                            label='策略收益', ax=ax, **kwargs)

        if legend_loc is not None:
            ax.legend(loc=legend_loc, frameon=True, framealpha=1)
        ax.axhline(1.0, linestyle='--', color='black', lw=1)
        ax.set_xlabel('时间')
        ax.set_ylabel('累计收益')
        ax.set_yscale('log' if logy else 'linear')
        ax.set_title('收益图')
        return ax
    
    def plot_monthly_returns_heatmap(self,returns, ax=None, **kwargs):
        """
        画出月度收益热力图

        参数
        ----------
        returns : pandas里面的序列格式
            returns : pandas里面的序列格式
                每日收益，记住不是累计收益。
        ax : matplotlib的轴,可选
                用来绘图的轴。
            **kwargs,可选
                传输到绘图函数的其他参数。
        返回
        ----------
            ax : ax的对象
        """

        if ax is None:
            plt.figure(figsize = (15,6))
            ax = plt.gca()

        monthly_ret_table = ep.aggregate_returns(returns, 'monthly')
        monthly_ret_table = monthly_ret_table.unstack().round(3)

        sns.heatmap(
            monthly_ret_table.fillna(0) *
            100.0,
            annot=True,
            annot_kws={"size": 10},
            alpha=1.0,
            center=0.0,
            cbar=True,
            cmap=matplotlib.cm.RdYlGn,
            ax=ax, **kwargs)
        ax.set_ylabel('年')
        ax.set_xlabel('月')
        ax.set_title('月度收益热力图')
        return ax


    def plot_drawdown_underwater(self,returns, ax=None, **kwargs):
        """
        画出回撤图

        参数
        ----------
        returns : pandas里面的序列格式
            returns : pandas里面的序列格式
                每日收益，记住不是累计收益。
        ax : matplotlib的轴,可选
                用来绘图的轴。
            **kwargs,可选
                传输到绘图函数的其他参数。

        返回
        ----------
            ax : ax的对象
        """

        if ax is None:
            plt.style.use("ggplot")
            plt.figure(figsize = (15,6))
            ax = plt.gca()

        y_axis_formatter = FuncFormatter(self.percentage)
        ax.yaxis.set_major_formatter(FuncFormatter(y_axis_formatter))

        df_cum_rets = ep.cum_returns(returns, starting_value=1.0)
        running_max = np.maximum.accumulate(df_cum_rets)
        underwater = -100 * ((running_max - df_cum_rets) / running_max)
        (underwater).plot(ax=ax,linewidth = 0.3, kind='area', color='red', alpha=0.7, **kwargs)
        ax.set_ylabel('回撤率')
        ax.set_title('回撤图')
        ax.set_xlabel('时间')
        return ax

```


```python
#实例化
obj = TsAnalyze()
```

获取收益率数据


```python
#平安银行收益
df_returns = get_price('000001.XSHE', start_date='2018-01-01', end_date='2020-10-30', frequency='daily', fields='close').pct_change().close
#沪深300指数收益
benchmark_returns = get_price('000300.XSHG', start_date='2018-01-01', end_date='2020-10-30', frequency='daily', fields='close').close.pct_change()
```

series格式长这个样子


```python
df_returns.head()
```




    2018-01-02         NaN
    2018-01-03   -0.027418
    2018-01-04   -0.005482
    2018-01-05    0.003150
    2018-01-08   -0.025118
    Name: close, dtype: float64



### 累计收益图


```python
#输入:策略收益和基准收益，series格式
obj.plot_accumulated_returns(df_returns,benchmark_returns)
```
<img src='http://mk.027cgb.cn/627139/bgpc/20201114/output_146_1.png'/>
```python
#输入:策略收益，series格式
obj.plot_monthly_returns_heatmap(df_returns)
```
<img src='http://mk.027cgb.cn/627139/bgpc/20201114/output_147_1.png'/>

```python
#输入:策略收益，series格式
obj.plot_drawdown_underwater(df_returns)
```
<img src='http://mk.027cgb.cn/627139/bgpc/20201114/output_148_1.png'/>
### 模块化

随着我们代码越写越多，文件中代码越来越长，最终导致越来越难以维护。为了编写可维护代码，我们可以把很多函数分组，分别放到不同文件里。在Python中，一个 .py 文件就称为一个模块（module）。为避免模块名冲突，Python引入了按目录来组织模块的方法，称为包（package）。

       使用模块优点：
       
       提高代码可维护性
       
       编写代码不必从零开始，当一个模块编写完毕，就可以被其他地方引用。
       
       避免函数名和变量名冲突，相同名字的函数和变量完全可以分别存在不同的模块中。


把写好的代码放在其他地方，调用，真正起到代码复用的的效果

在同一文件夹timeseries_analyze_package下，放置timeseries_analyze.py和__init__.py文件即可,如果要新填类或者函数，可以在timeseries_analyze.py填写代码，运用上面讲到的类的封装，继承，多态等性质。

导入自己的模块timeseries_analyze


```python
import timeseries_analyze as tsa
```


```python
#创建实例
obj = tsa.TsAnalyze()
```


```python
#输入:策略收益和基准收益，series格式
obj.plot_accumulated_returns(df_returns,benchmark_returns)
```



```python
#输入:策略收益，series格式
obj.plot_monthly_returns_heatmap(df_returns)
```
<img src='http://mk.027cgb.cn/627139/bgpc/20201114/output_155_1.png'/>
```python
#输入:策略收益，series格式
obj.plot_drawdown_underwater(df_returns)
```

<img src='http://mk.027cgb.cn/627139/bgpc/20201114/output_157_1.png'/>


## Next

现在自己动手在这个模块修复bug,或者添加有意思的功能吧

> 如果在这部分学习上遇到疑惑的话，可以通过<strong>1115223619@qq.com</strong>或者[这里](https://ownyulife.top/contact/)咨询助教。
