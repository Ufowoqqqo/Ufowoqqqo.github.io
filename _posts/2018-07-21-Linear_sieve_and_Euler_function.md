---
layout:     post
title:      "线性筛和 Euler 函数学习笔记"
date:       2018-07-21 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 数论
    - 线性筛
---



### $\text{Eratosthenes}$ 筛法

求出 $n$ 以内的所有质数。

反向考虑，筛出所有的合数。根据定义，合数必有除了 $1$ 和其自身外的约数。

对于 $n$ 以内的每个 $i$，筛去所有的 $ki(k>1, k\in \mathbb{N})$。

总的时间复杂度为 $O(\frac{n}{2}+\frac{n}{3}+\ldots+\frac{n}{\lfloor\frac{n}{2}\rfloor}\approx n\lg n)$。

当 $n\ge 10^7$ 时无法通过。



### 线性筛

$\text{Eratosthenes}$ 筛法的低效之处在于它把每个合数筛了多次（确切地说，筛了 $[\sigma_0(i)-2]$ 次）。

如果能找到 $1$ 种方法，使得每个合数恰好只被筛 $1$ 次，那么就可以达到 $O(n)$ 的复杂度。

$\text{Euler}$ 发明了这种线性筛法，规定每个合数只被其最小的质因子筛去。

实现方法：质数表 `p[]` 中保存了不大于 $i$ 的所有质数。

$\forall1<i\le n$，若 $i$ 之前没有被筛，即 $i$ 为质数，则更新质数表。

考虑质数表中的每个质数 $p[j]$，不断尝试筛去 $k=i\times p[j]$（其中 $p[j]$ 是 $k$ 的最小质因子），**直至 $p[j] | i$**。

结束条件是因为当 $p[j]|i$ 时，$p[j]$ 已是 $i$ 的最小质因子（因为整除，所以是质因子；因为首个遇到，所以最小）。

假如不结束，那么对于下 $1$ 个被筛去的 $k'=i\times p[j+1]$，其最小质因子就不再是 $p[j+1]$，这与要求不符。

```cpp
for (int i = 2; i <= n; i++)
{
	if (!vis[i]) Prime[pn++] = i;

	for (int j = 0; j < pn; j++)
	{
		int k = Prime[j] * i;
		if (k > n) break;
		vis[k] = 1;
		if (i % Prime[j] == 0) break;
	}
}
```

同时可以发现，除了最后 $1$ 次停止以外，$i$ 与 $p[j]$ 始终互质。这个性质对后面积性函数的求解非常重要。



### 积性函数

若函数 $f(x)$ 满足当 $\gcd(a, b)=1$ 时 $f(ab)=f(a)f(b)$，则称 $f(x)$ 为积性函数。

特别地，若函数 $f(x)$ 始终满足 $f(ab)=f(a)f(b)$，则称 $f(x)$ 为完全积性函数。



### $\text{Euler}$ 函数

#### 定义

$\varphi(n)$ 表示 $[1, n]$ 范围内与 $n$ 互质的数的个数。

#### 特点

- $\varphi(1)=1$，注意到 $\gcd(1,1)=1$ 也满足互质定义。
- 显然对于任意质数 $p$，都有 $\varphi(p)=p-1$。
  - 特别地，对于只有 $1$ 个质因子的 $p^k$，$p$ 的倍数都不与其互质，即

$$
\varphi(p^k)=p^k-\frac{p^k}{p}=p^k-p^{k-1}=(p-1)p^{k-1}
$$

- **$\text{Euler}$ 函数是积性函数**。即当 $\gcd(a,b=1)$ 时，$\varphi(ab)=\varphi(a)\varphi(b)$。

  > 证明：
  >
  > 把 $1$ 至 $ab$ 范围内的数每 $b$ 个分为 $1$ 组，每组构成模 $b$ 的完全剩余系。
  >
  > 若某个数与 $ab$ 互质，显然必须先与 $b$ 互质。
  >
  > 由于当 $r<b$ 时有 $\gcd(kb+r,b)=\gcd(r,b)$，故每组中都有 $\varphi(b)$ 个数与 $b$ 互质。
  >
  > 考虑与 $b$ 互质当中的任意 $1$ 种 $r$，将每组中与 $r$ 同余的数取出，即 $r, b+r, 2b+r, \ldots, (a-1)b+r$。
  >
  > 它们构成模 $a$ 的完全剩余系（具体涉及群论知识，此处不详细阐述），其中有 $\varphi(a)$ 个数与 $a$ 互质。
  >
  > 由乘法原理可知共有 $\varphi(a)\varphi(b)$ 个数与 $ab$ 互质。

#### 求解

对 $n$ 进行唯 $1$ 分解得到 $n=p_1^{k_1}\times p_2^{k_2}\times\ldots p_m^{k_m}$。

其中各个质因子的幂两两互质，易得 
$$
\begin{align}\varphi(n)&=\varphi(p_1^{k_1})\varphi(p_2^{k_2})\ldots\varphi(p_m^{k_m})\\
 &= (p_1-1)p_1^{k_1-1}\times (p_2-1)p_2^{k_2-1}\times\ldots\times (p_3-1)p_3^{k_3-1}\\
 &=(p_1-1)(p_2-1)\ldots(p_m-1)p_1^{k_1-1}p_2^{k_2-1}\ldots p_m^{k_m-1}\\
 &=n(1-\frac{1}{p_1})(1-\frac{1}{p_2})\ldots(1-\frac{1}{p_m})\\
 &=n\prod_{i=1}^m(1-\frac{1}{p_i})\end{align} 
$$
直接根据这个式子求需要分解质因数，时间复杂度为 $O(n\sqrt n)$，显然是不够优秀的。

考虑线性筛。

分类讨论。若 $i\mod p[j]\neq 0$，即 $i$ 与 $p[j]$ 互质，根据欧拉函数的积性显然有 $\varphi(i\times p[j])=\varphi(i)\varphi(p[j])$。

若 $p[j]|i$，则 $i\times p[j]$ 与 $i$ 的质因子集合相同，即定义式中连乘号后面的部分相同，只是 $i$ 变成了 $i\times p[j]$，因此 $\varphi(i\times p[j])=\varphi(i)\times p[j]$。

```cpp
Phi[1] = 1;
for (int i = 2; i <= n; i++)
{
	if (!vis[i]) 
	{
		Prime[pn++] = i;
		Phi[i] = i - 1;
	}

	for (int j = 0; j < pn; j++)
	{
		int k = Prime[j] * i;
		if (k > n) break;
		vis[k] = 1;
		if (i % Prime[j] == 0) 
		{
			Phi[k] = Phi[i] * Prime[j];
			break;
		}
		else Phi[k] = Phi[i] * Phi[Prime[j]];
	}
}
```



### 欧拉定理

若 $\gcd(a,m)=1$，则 $a^{\varphi(n)} \equiv 1(\mathrm{mod}\ n)$。

特别地，若 $m$ 为质数，即 $\varphi(m)=m-1$，则 $a^{m-1}\equiv 1(\mathrm{mod}\ m)$，这就是费马小定理。

#### 求乘法逆元

若整数 $b,m$ 互质，且存在 $1$ 个整数 $x$，使得 $b\times x\equiv 1(\mathrm{mod}\ m)$。称 $x$ 为 $b$ 的**模 $m$ 乘法逆元**，记为 $b^{-1}(\mathrm{mod}\ m)$。

若 $m$ 为质数，将费马小定理写成 $a\times a^{m-2}\equiv 1(\mathrm{mod}\ m)$ 的形式，显然 $a^{m-2}\mod m$ 就是 $a$ 的模 $m$ 乘法逆元；若 $m$ 不为质数，可以通过解线性同余方程的手段求逆元。

#### 降幂

欧拉定理有如下推论：若正整数 $a, n$ 互质，则对于任意正整数 $b$，有 $a_b\equiv a^{b\mod \varphi(n)}(\mathrm{mod}\ n)$。

>  证明：设 $b=q\times\varphi(n)+r$，其中 $0\le r<\varphi(n)$，即 $r=b\mod \varphi(n)$。于是：
> $$
> a^b\equiv a^{q\times\varphi(n)+r}\equiv(a^{\varphi(n)})^q\times a^r\equiv 1^q\times a^r\equiv a^r\equiv a^{b\mod\varphi(n)}(\mathrm{mod}\ n)
> $$
>

求 $2^{2^{2^{\ldots}}}\mod m$。

记 $k=2^{2^{2^{\ldots}}}$，显然有 $k=2^k$。于是所求为 $2^{k\mod\varphi(m)}$。不难发现指数部分就是原问题的子问题。

递归求解，边界为 $m=1$，此时值为 $0$，再反向代回即可。易证递归层数不会超过 $\log m$ 层。

> 若 $m$ 为偶数，则 $\varphi(m)\le\lfloor\frac{m}{2}\rfloor$，即规模减小 $1$ 半；
>
> 若 $m$ 为奇数，则其所有质因子 $j$ 都为奇数，$\varphi(j)=j-1$ 都为偶数，它们相乘得到的 $\varphi(m)$ 也是偶数，即奇数下 $1$ 步也会变成偶数。
>
> 因此不断对自身求 $\varphi$ 的复杂度为 $O(\log n)$。



### 常见互质问题

求 $\sum_{i=1}^n i[\gcd(i,n)=1]$。

引理：$a<n,\ \gcd(a,n)=1 \Leftrightarrow\gcd(n-a,n)=1$。

因此当 $\gcd(i,n)=1$ 时必有 $\gcd(n-i,n)=1$，即与 $n$ 互质的数成对出现，且每对的和都是 $n$。

当 $n>1$ 时总共有 $\lfloor\frac{\varphi(n)}{2}\rfloor$ 对。因此所求即为 $\frac{\varphi(n)+[n=1]}{2}\times n$。

---

求 $\sum_{i=1}^n \sum_{j=1}^n[\gcd(i,j)=1]$。

为避免重复计数，不妨规定 $j<i$。$\forall 1\le i\le n$ 对答案的贡献为 $\varphi(i)$。

所求即为 $\sum_{i=1}^n\varphi(i)$。



### 线性筛求任意积性函数

关键是把每个合数拆成互质的 $2$ 个数之积，即最小质因子的幂及其余部分。

考虑维护每个数的最小质因子的幂 $fir[i]$。

若 $i$ 为质数，则 $fir[i]=i$；

在线性筛过程中，若 $i\mod p[j]\neq 0$，则 $fir[i\times p[j]]=p[j]$；

否则 $fir[i\times p[j]]=fir[i]\times p[j]$。

