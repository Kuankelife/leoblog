---
layout: post
title: Algorithm in number theory 0.1(转载)
#subtitle: ''
date: 2019-11-06
author: 'Coinc1dens'
header-mask: 0.3
mathjax: true
tags:
    - Math

catalog: false



---

>算法，是彩色的。

# Baby Step Giant Step

> 又名拔山盖世/北上广深算法

用于求解 $a^x\equiv b\ (mod\ n)$ 中的最小自然数 $x$

## 引理

由欧拉定理知，对于互质的正整数 $a,p$，$a^{\phi(p)}\equiv 1\ (mod\ p)$，且 $x=k·\phi(p)$ 为 $b$ 的循环节

## 过程

1. 设 $d=gcd(a,n)$，则原式化为 $a^x\equiv b\ (mod\ {n\over{d^k}})$ 直到 $gcd(a,{n\over {d^k}})=1$
2. 采用分块思想，令 $x=im-j \in [0,\phi(n))$，则求解：${(a^m)}^i\equiv a^jb\ (mod\ {n\over{d^k}})$
3. 使用 map/others 存下对应的 $j,a^jb$，枚举 $i$ 得到答案

+ $m$ 一般取 $\sqrt n$
+ 当 ${d^k} \mid b$，原式有自然数解
+ 2，3步中，为了方便起见，可求解 ${(a^m)}^i\equiv a^jb({a\over d})^k\ (mod\ {n\over{d^k}})$，则最后的答案为 $x-k=im-j$，即 $x=im-j+k $

## 代码（[exBSGS](https://www.luogu.org/problem/P4195)）

> 期望复杂度$O(\sqrt nlog(\sqrt n))$

~~~c++
#include<cstdio>
#include<cmath>
#include<map>
#define ll long long
using namespace std;
map<ll, ll>mp;
ll a,b,p;
ll gcd(ll a,ll b){return b == 0?a:gcd(b,a % b);}
ll pow(ll a,ll b,ll c){
	ll ans = 1;
	while(b){
		if(b & 1)ans = ans * a % c;
		b >>= 1;
		a = a * a % c;
	}
	return ans;
}
void exbsgs(){
	mp.clear();
	if(b == 1 or p == 1){
		puts("0");
		return ;
	}
	ll d = gcd(a,p),ans = 1,dd = d;
	while(d > 1 and p % dd == 0) dd = dd * d;
	if(dd > 1)dd /= d,p /= dd;
	if(b % dd){
		puts("No Solution");
		return;
	}
	ll m = (ll)ceil(sqrt(p)),temp = b % p;
	mp[temp] = -1;
	for(int i = 1;i < m;++i) mp[temp = temp * a % p] = i;
	temp = pow(a,m,p);
	for(int i = 1;i <= m;++i) if(mp[ans = ans * temp % p]){
		if(mp[ans] == -1)mp[ans] = 0;
		printf("%lld\n",(i * m - mp[ans] + p) % p);
		return;
	}
	puts("No Solution");
	return ;
}
void exbsgs2(){
	mp.clear();
	if(b == 1 or p == 1){
		puts("0");
		return ;
	}
	ll d = gcd(a,p),ans = 1,num = 0;
	while(d > 1 and p % d == 0){
		num ++;
		p /= d,ans = ans * a / d % p;
		if(b % d == 0) b /= d;
		else {
			puts("No Solution");
			return ;
		}
	}
	ll m = (ll)ceil(sqrt(p)),temp = b;
	mp[b] = -1;
	for(int i = 1;i < m;++i) mp[temp = temp * a % p] = i;
	temp = pow(a,m,p);
	for(int i = 1;i <= m;++i) if(mp[ans = ans * temp % p]){
		if(mp[ans] == -1)mp[ans] = 0;
		printf("%lld\n",(i * m - mp[ans] + p + num) % p);
		return;
	}
	puts("No Solution");
	return ;
}
int main(){
	while(scanf("%lld%lld%lld",&a,&p,&b) == 3 && p){
		a %= p;
		if(!a and !b)puts("1");
		else if(a < 2)puts("No Solution");
		else exbsgs2();
	}
}

~~~



# 二次剩余

> 当存在某个 $x$，使 $$x^2\equiv a\ (mod\ p)$$ 成立时，称 $a$ 是模 $p$ 的二次剩余
>
> 本方法对于 $p$ 为奇素数成立

## 引理

$(a+b)^p\equiv a^p + b^p\ (mod\ p)$

### 证明

$$
(a+b)^p=\sum_{i=0}^p\binom{p}{i}a^ib^{p-i}\\
\binom{p}{i}\equiv 0\ (mod\ p)\\(i\ne p,i\ne 0,a,b\in \Bbb Z)
$$

## 勒让德符号

### 定义

$$
\left(\frac{a}{p}\right)=
\begin{cases}
1,& \text{$a$ 是模 $p$ 的二次剩余} \\
-1,& \text{$a$ 是模 $p$ 的非二次剩余} \\
0,& a\equiv0 \pmod p
\end{cases}\\
({a\over p})\equiv a^{ {p-1}\over 2}\ (mod\ p)
$$

+ 其中 $p$ 为奇素数

### 证明

由费马小定理：$ (a^{ {p-1}\over 2}-1 )(a^{ {p-1}\over 2}+1)\equiv 0\ (mod\ p)$

故：$a^{ {p-1}\over 2}\equiv ±1\ (mod\ p)$

+ 若为二次剩余：
  + 充分必要性显然

+ 若为非二次剩余， 只能 $a^{ {p-1}\over 2}\equiv -1\ (mod\ p)$

+ 算了上面这段待补充[here](https://okami.biz/2019/01/欧拉准则/) && [here](https://www.cnblogs.com/3200Pheathon/p/10800065.html)

## 二次剩余的数量

在 $[0,p)$ 中有 ${ {p-1}\over 2}+1$ 个（非二次剩余的数量是 ${ {p-1}\over 2}$）

### 证明

考虑两个不同的数 $x,y \in (0,p)$，若 $x^2\equiv y^2\ (mod\ p)$，则 $(x-y)(x+y)\equiv 0\ (mod\ p)$，故知 $x+y\equiv 0\ (mod\ p)$，即 $y=p-x$，故这样的数恰有 ${ {p-1}\over 2}$ 对，且每一对的二次剩余都不相同，加上0的存在得到以上<br>

则模域内剩下的数无法得到，即为非二次剩余

## 过程

1. 随机选取 $t$，使得 $t^2 - a$ 为非二次剩余（选取 $t$ 的期望次数极小）
2. 令 $b=\sqrt {t^2 - a}$，则 $x=(t+b)^{ {p+1}\over 2}$

## 证明

$$
\begin{align}
x^2&=(t+b)^{p+1}\\
&\equiv (t^{p+1}+b^{p+1})\\
&\equiv 1·t^2+(-1)·(t^2 - a)\\
&\equiv a\ (mod\ p)
\end{align}
$$

## 代码（[二次剩余](https://www.luogu.org/problem/P5491)）

> 期望复杂度$O(log^2 n)$

~~~c++
#include<cstdio>
#include<cmath>
#include<algorithm>
using namespace std;
long long T,t,n,p,vi;
struct node{
	long long x,y;
};
node mul(node a,node b,long long c){
	node ans = {0,0};
	ans.x = (a.x * b.x % c + a.y * b.y % c * vi) % c;
	ans.y = (a.x * b.y % c + a.y * b.x % c) % c;
	return ans;
}
long long pow(long long a,long long b,long long c){
	long long ans = 1;
	while(b){
		if(b & 1) ans = ans * a % c;
		b >>= 1;
		a = a * a % c;
	}
	return ans;
}
long long powp(node a,long long b,long long c){
	node ans = {1,0};
	while(b){
		if(b & 1)ans = mul(ans,a,c);
		b >>= 1;
		a = mul(a,a,c);
	}
	return ans.x;
}
int main(){
	scanf("%d",&T);
	while(T--){
		scanf("%d%d",&n,&p);
		if(!n) puts("0");
		else if(pow(n,p - 1 >> 1,p) != 1){
			puts("Hola!");
			continue;
		}
		else{
			t = sqrt(n)+1;
			while(pow(t * t - n,p - 1 >> 1,p) == 1) t++;
			vi = t * t - n;
			t = powp((node){t,1},p + 1 >> 1,p);
			long long g = p - t;
			if(!g or !t) puts("0");
			else printf("%d %d\n",min(g,t),max(g,t));
		}
	}
} 
~~~



# Miller rabin

> int范围内使用2，3，5，7验证即可

## 引理

+ 费马小定理：

  $$
  if\ p\ is\ a\ prime,\\a^{p-1}\equiv 1\ (mod\ p)
  $$

+ 二次探测定理：

  $$
  if\ p\ is\ a\ prime,\\and\ a^2\equiv 1\ (mod\ p),\\
  a\equiv ±1\ (mod\ p)
  $$

## 步骤

1. 设 $p-1 = {2^k}·t $（$t$ 为奇数）
2. 随机选取 $a\in (1,p-1)$，从 $a^{2^0·t}$ 开始，直到 $a^{2^k·t}$ 为止，套用二次探测定理及费马小定理验证
3. 若验证失败，则为合数；否则多取几个随机 $a$ 进行验证

## 代码（[线性筛素数](https://www.luogu.org/problem/P3383)）

> 期望复杂度$O(klog^2 n)$

~~~c++
#include<iostream>
using namespace std;
int p[5]={0,2,3,5,7};
long long pow(long long a,int b,int c){
	long long ans = 1;
	while(b){
		if(b & 1)ans = ans * a % c;
		b >>= 1;
		a = a * a % c;
	}
	return ans;
}
bool check(long long a,int t,int x){
	long long temp = a;
	while(k--){
		a = a * a % x;
		if(a == 1 and temp - 1 != 0 and temp + 1 != x) return 0;
		temp = a;
	}
	return temp == 1;
}
bool mb(int x){
	if(x < 10) return (x == 2 or x == 3 or x == 5 or x == 7);
	if(x % 2 == 0) return 0;
	int k = 0,t = x - 1;
	while(t % 2 == 0)
		t >>= 1,
		k ++;
	for(int i = 1;i <= 4;++i)
		if(!check(pow(p[i],t,x),t,x)) return 0;
	return 1;
}
int main(){
	int n,m,x;
	cin>>n>>m;
	for(int i = 1;i <= m;++i){
		cin>>x;
		cout << (mb(x)?"Yes":"No") << endl;
	}
}
~~~



# Pollard Rho

> 反正我觉得挺玄学的

## 原理

> 随机，生日悖论
>
> 需要选取数的个数约在 $O(N^{1\over 4})$
>
> $\{a_i\}$ 这个函数是玄学函数

1. 随机选取 $a_0\in (1,N-1)$，不断计算得到 $a_i\equiv a^2_{i-1}+k\ (mod\ N)$（$k$ 为常数）
   + 模域下，根据生日悖论，数列 $\{a_i\}$ 形成环且期望环长为 $\sqrt N$

2. 判断是否满足 $gcd(abs(a_i-a_{i-1}),N) \in (1,N)$，若满足则找到 $N$ 的一个因子
   + 成环后仍未找到因子则表示分解失败，可调整 $k$ 重新分解

3. 找到后通过 Miller rabin 算法判断因子是否为质数，若不是则使用该算法继续分解

## 优化

+ 路径倍长

  > 某题解认为 127 是个不错的 loop

  在对 $N$ 模域下，$gcd(\prod abs(a_i-a_{i-1}),N)$ 与非模域下相同，故每相隔一段时间后计算 $gcd(\prod abs(a_i-a_{i-1}),N)$ 以优化对于 $gcd$ 的时间复杂度



## 代码

+ ***咕咕咕待补充***

