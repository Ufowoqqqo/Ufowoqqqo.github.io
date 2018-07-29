---
layout:     post
title:      "Möbius 反演学习笔记"
date:       2018-07-24 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 数论
    - Möbius 反演
---



### 引入

考虑形如 $F(n)=\sum_{d|n}G(d)$ 的**卷积式**，用 $F(n)$ 求 $G(n)$。

由特殊到 $1$ 般推导。

直接反着推可以得到前几个，此处略去。

通过观察可以发现它们排列的规律：对于每个 $G(n)$，它等于 $1$ 系列 $F(d)$ 相加相减，其中 $d$ 为 $n$ 的约数。即枚举 $n$ 的每个约数 $d$，令 $F(d)$ 乘上对应的某个系数（$-1, 0$ 或 $1$），并相加即可得到 $G(n)$。

如何求出这个系数是关键。

特别地，对于 $n=p^k, F(n)=\sum_{i=0}^k G(p^i)$。

$n$ 的约数为 $p^0, p^1, p^2, \ldots, p^k$。

而 $F(\frac{n}{p})=\sum_{i=0}^{k-1} G(p^i)$，即 $G(p^k)=F(p^k)-F(p^{k-1})$。这是 $1$ 维前缀和的形式。

进 $1$ 步，对于 $n=p_1^{k_1}p_2^{k_2}$，其约数可以表示为 $p_1^x p_2^y(0\le x\le k_1, 0\le y\le k_2)$。

也就是 $F(n)=\sum_{i=0}^{k_1}\sum_{j=0}^{k_2}G(p_1^i p_2^j)$，易得 $G(n)=F(n)-F(\frac{n}{p_1})-F(\frac{n}{p_2})+F(\frac{n}{p_1p_2})$。这是 $2$ 维前缀和的形式。

根据容斥原理不难得到 $1$ 般形式的式子，其中每项的系数与分母中质因子个数相关。记这个系数为 $\mu (d)$。

当质因子的次数不超过 $1$ 时（即 $d$ 为 $\text{square free}$ 数），若分母中有奇数个质因子，$\mu(d)=-1$；若有偶数个则 $\mu(d)=1$。

当质因子的次数超过 $1$ 时，$\mu(d)=0$。

特别地，$\mu(1)=1$。

---

至此我们就得到了 $\text{M}\ddot{o}\text{bius}$ 反演公式
$$
F(n)=\sum_{d|n}G(d)\Leftrightarrow G(n)=\sum_{d|n}\mu(d)F(\frac{n}{d})
$$
以及 $\text{M}\ddot{o}\text{bius}$ 函数 $\mu$ 的定义。



### 线性筛求 $\text{M}\ddot{o}\text{bius}$ 函数

由定义易知 $\text{M}\ddot{o}\text{bius}$ 函数是积性函数。

对于质数 $p$，$\mu(p)=-1$；

在筛的过程中，若 $i\mod p[j]\neq 0$，则 $\mu(i\times p[j])=-\mu(i)$；

否则 $i\times p[j]$ 为非 $\text{square free}$ 数，$\mu(i\times p[j])=0$。

```cpp
void GetPrime()
{
	Miu[1] = Sum[1] = 1;
	for (int i = 2; i <= MAXN; i++)
	{
		if (!vis[i]) vis[i] = 1, Miu[i] = -1, Prime[pn++] = i;
		Sum[i] = Sum[i - 1] + Miu[i]; //求前缀和，见下
		for (int j = 0; j < pn; j++)
		{
			if (LL(i) * Prime[j] > MAXN) break;
			int k = i * Prime[j];
			vis[k] = 1;
			if (i % Prime[j] == 0) { Miu[k] = 0; break; }
			else Miu[k] = -Miu[i];
		}
	}
}
```





### 性质

$\sum_{d|n}\mu(d)=[n=1]$。

证明需要用到二项式定理，此（wǒ）处（méi）略（xué）去（guò）。



### 应用

求 $\sum_{i=1}^n\sum_{j=1}^n\gcd(i,j)$。

设 $\gcd(i,j)=d$。

所求即为
$$
\begin{align}
&\sum_{d=1}^n\sum_{x=1}^{\lfloor\frac{n}{d}\rfloor}\sum_{y=1}^{\lfloor\frac{n}{d}\rfloor}d[\gcd(i,j)=1]\\
=&\sum_{d=1}^{n}d\sum_{x=1}^{\lfloor\frac{n}{d}\rfloor}\sum_{y=1}^{\lfloor\frac{n}{d}\rfloor}\sum_{k|\gcd(x,y)}\mu(k)\\
=&\sum_{d=1}^{n}d\sum_{x=1}^{\lfloor\frac{n}{d}\rfloor}\sum_{y=1}^{\lfloor\frac{n}{d}\rfloor}\sum_{k|x, k|y}\mu(k)\\
=&\sum_{d=1}^{n}d\sum_{k=1}^{\lfloor\frac{n}{d}\rfloor}\mu(k)\sum_{x'=1}^{\lfloor\frac{\lfloor\frac{n}{d}\rfloor}{k}\rfloor}\sum_{y=1}^{\lfloor\frac{\lfloor\frac{n}{d}\rfloor}{k}\rfloor} 1\\
=&\sum_{d=1}^{n}d\sum_{k=1}^{\lfloor\frac{n}{d}\rfloor}\mu(k)\lfloor\frac{n}{dk}\rfloor^2
\end{align}
$$


### 整除分块

对于任意整数 $x\in[1,k]$，设 $g(x)=\lfloor\frac{k}{\lfloor\frac{k}{x}\rfloor}\rfloor$。显然函数 $f(x)=\frac{k}{x}$ 单调递减，而 $g(x)\ge \lfloor\frac{k}{\frac{k}{x}}\rfloor=x$，故 $\lfloor\frac{k}{g(x)}\rfloor\le\lfloor\frac{k}{x}\rfloor$。

另外，$\lfloor\frac{k}{g(x)}\rfloor\ge\lfloor\frac{k}{\frac{k}{\lfloor\frac{k}{x}\rfloor}}\rfloor=\lfloor\frac{k}{k}\times\lfloor\frac{k}{x}\rfloor \rfloor=\lfloor\frac{k}{x}\rfloor$。故 $\lfloor\frac{k}{g(x)}\rfloor=\lfloor\frac{k}{x}\rfloor$。进 $1$ 步可得，$\forall i\in [x, \lfloor \frac{k}{\lfloor\frac{k}{x}\rfloor}\rfloor]$，$\lfloor\frac{k}{i}\rfloor$ 的值都相等。

$\forall i\in[1,k]$，$\lfloor\frac{k}{i}\rfloor$最多只有 $2\sqrt k$ 个不同的值。这是因为当 $i\le \sqrt k$ 时，$i$ 只有 $\sqrt k$ 种选择，故 $\lfloor\frac{k}{i}\rfloor$ 至多只有 $\sqrt k$ 个不同的值。而当 $i>\sqrt k$ 时，$\lfloor\frac{k}{i}\rfloor<\sqrt k$，故 $\lfloor\frac{k}{i}\rfloor$ 也至多只有 $\sqrt k$ 个不同的值。

综上所述，对于 $i=1\sim k,\lfloor\frac{k}{i}\rfloor$ 由不超过 $2\sqrt k$ 段组成，每 $1$ 段 $i\in [x, \lfloor\frac{k}{\lfloor\frac{k}{x}\rfloor}\rfloor]$ 中 $\lfloor\frac{k}{i}\rfloor$ 的值都等于 $\lfloor\frac{k}{x}\rfloor$。

对于 $1$ 类参数中分母为定值、分子连续的求和问题（如上例），可以用分块处理。

记 $F(n)=\sum_{k=1}^n\mu(k)\lfloor\frac{n}{k}\rfloor^2$。预处理 $\mu$ 的前缀和，分块可以在 $O(\sqrt n)$ 时间内求解。

对于原问题 $\sum_{d=1}^ndF(\lfloor\frac{n}{d}\rfloor)$，也可以通过分块处理。

总的时间复杂度为 $O(n)$。

```cpp
int GetF(int n) //分块处理
{
	LL ans = 0;
	for (int i = 1; i <= n; )
	{
		int j = n / (n / i);
		ans += LL(n / i) * (n / i) * (Sum[j] - Sum[i - 1]);
		i = j + 1;
	}
	return ans;
}

int main()
{
	freopen("miu.in", "r", stdin);
	freopen("miu.out", "w", stdout);

	GetPrime();
	
	int n;
	scanf("%d", &n);
	LL ans = 0;
	for (int i = 1; i <= n; )
	{
		int j = n / (n / i);
		ans += LL(j - i + 1) * (j + i) / 2 * GetF(n / i); //分块处理
		i = j + 1;
	}
	printf("%I64d\n", ans);

	return 0;
}
```

