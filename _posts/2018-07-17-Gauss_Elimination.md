---
layout:     post
title:      "高斯消元学习笔记"
date:       2018-07-17 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 算法笔记
    - 高斯消元
---



#### 概念引入

> $n$ 元线性方程：含有 $n$ 个 $1$ 次未知数的方程，经过移项及合并同类项后得到 $1$ 般形式为 $a_1x_1+a_2x_2+\ldots+a_nx_n=c$
>
> $n$ 元线性方程组：$m$ 条 $n$ 元线性方程列在 $1$ 起。

线性方程组的所有系数可以写成 $1$ 个 $m$ 行 $n$ 列的系数矩阵，再加上等号右侧的常数可以写成 $m$ 行 $(n+1)$ 列的**增广矩阵**。其中每行对应原来的 $1$ 条方程，每列对应某个元的在每个方程中的系数。

在矩阵上进行运算并得出方程的解，就是高斯消元法。本文介绍与其 $10$ 分接近的**高斯-约旦消元法**。



#### 矩阵变换

回顾初中数学内容。为了得到方程（组）的解，我们往往会将其进行形式上的改造，且保证在此过程中不违反等式的基本性质，从而使方程变得易解。类似地，我们需要对矩阵进行操作，并且不改变方程的解。

- 让 $1$ 行乘上非 $0$ 的常数 $k$
- 交换 $2$ 行
- 让 $1$ 行加减另外 $1$ 行，得到新方程替代原来的 $1$ 个方程



#### 消元

解决多元方程的核心是消元，因为我们最终只会直接解含有 $1$ 个未知数的方程。

每次选 $1$ 条方程 $i$ 中尚未被消去的元，称为主元 $x_0$。尝试用主元把其他所有方程中关于 $x_0$ 的系数消去，以得到减少了未知数的子方程组。当然，只能消去 $x_0$ 系数非 $0$ 的方程。至于具体的实现，可以通过比例关系，使被消方程 $j$ 中 $x_0$ 系数等于方程 $i$ 中 $x_0$ 系数，其它元也缩放相同的倍数，并用得到的方程 $j'$ 减去方程 $i$ 的结果代替原来的方程 $j$。注意最后 $1$ 列的解也要同时改变。

主元的选取 $1$ 般从第 $1$ 个元开始顺序进行。若发现不存在使得当前要消的元系数非 $0$ 的方程，则说明条件不足，无法解出。否则，每消完 $1$ 个元，对应列上都有且只有 $1$ 个非 $0$ 系数。为了方便最终的求解，我们规定第 $i$ 个元的非 $0$ 系数恰好位于第 $i$ 行。这样，最终得到的就是**主对角线矩阵**。



#### 求解

若 $m<n$，方程数量不够，无解；

若 $m>n$，方程有多余，若不存在互相矛盾的方程，则多余方程都会被消掉，没有影响；

若 $m=n$ 且不存在等价方程（即各元系数之间为相同的倍数关系），恰好可以解出。

在最终的主对角线矩阵中，每行都是 $1$ 元 $1$ 次方程，元 $i$ 的解就是 $x_i=\frac{a[i, n]}{a[i, i]}$。



#### 细节

##### 主元的选取

根据上文，为了使最终得到主对角线矩阵，应该用第 $i$ 行的方程消第 $i$ 个未知数。然而，可能出现第 $i$ 行的方程中第 $i$ 个未知数系数为 $0$ 的情况。

解决方案是向后找出第 $i$ 个未知数系数非 $0$ 的方程 $j$，并交换第 $i$ 行和第 $j$ 行。不能向前找的原因是前面的行已经各自消过 $1$ 后面的行了，无法再消。

理论上找出任意的当前未知数系数非 $0$ 的方程均可，但由于在方程变形计算过程中对应系数要作为分母，如果系数太小，除出来的值就可能很大。为了精确起见，取对应系数绝对值最大的 $j$。

##### 只有整数的消元

在消元时，若 $\frac{a[j,i]}{a[i,i]}$ 不是整数，会导致中间结果不是整数。

可以任取 $a[i, i]$  和 $a[j, i]$ 的公倍数 $Q$，并使第 $i$ 行乘上 $\frac{Q}{a[i, i]}$，第 $j$ 行乘上 $\frac{Q}{a[j, i]}$ 后再相减即可。

要保证最终解为整数才能只用整数进行高斯消元。



#### 例子

$$\left[ \begin{array} {ccc|c} 1 & 2 & 1 & 8 \\ 2 & 2 & 3 & 15 \\ 4 & 2 & 1 & 11 \\ \end{array} \right]$$  消掉 $x_1$

$$ \left[ \begin{array} {ccc|c}1 & 2 & 1 & 8 \\ 0 & -2 & 1 & -1 \\ 0 & -6 & -3 & -21 \\ \end{array} \right] $$  消掉 $x_2$

$$ \left[ \begin{array} {ccc|c} 1 & 0 & 2 & 7 \\ 0 & -2 & 1 & -1 \\ 0 & 0 & -6 & -18 \\ \end{array} \right] $$  消掉 $x_3$

$$\left[ \begin{array} {ccc|c} 1 & 0 & 0 & 1 \\ 0 & -2 & 0 & -4 \\ 0 & 0 & -6 & -18 \\ \end{array} \right] $$  得到 $$ \left \{  \begin{array}{c} x_1=1 \\  x_2=2 \\  x_3=3 \end{array} \right.  $$ 



#### 代码实现

```cpp
int Gauss() {
	if (m < n) return 0; //判断方程数量够不够
	for (int i = 1; i <= n; i++) {
		int k = i;
		for (int j = i + 1; j <= m; j++) if (abs(A[k][i]) < abs(A[j][i])) k = j; //选出最大的主元
		for (int j = 1; j <= n+1; j++) swap(A[i][j],A[k][j]); //注意是到 n + 1

		if (abs(A[i][i]) < Eps) return 0; //无解，注意实数问题

		for (int j = 1; j <= m; j++) //消元
			if (j != i) {
				double p = A[j][i] / A[i][i];
				for (int k = 1; k <= n + 1; k++) A[j][k] -= A[i][k] * p;
			}
	}
	for (int i = n + 1; i <= m; i++) if (abs(A[i][n + 1]) > Eps) return 0;
	//判断多余的方程有没有导致无解
	for (int i = 1; i <= n; i++) Ans[i] = A[i][n + 1] / A[i][i]; //求解
	return 1;
}
```



#### 时空复杂度

时间复杂度 $O(mn^2)\approx O(n^3)$

空间复杂度 $O(mn)\approx O(n^2)$



#### 例题

##### $[\text{JSOI}2008]$ 球形空间产生器 $\texttt{sphere}$

$1$ 个球体上的所有点到球心的距离相等，因此只需求出 $1$ 个点 $(x_1, x_2, \ldots , x_n)$ 使得
$$
\sum_{j=0}^n (a_{i,j}-x_j)^2=C
$$

其中 $C$ 为常数，$i\in [1, n+1]$，球面上第 $i$ 个点的坐标为 $(a_{i,1},a_{i,2},\ldots,a_{i,n})$。该方程组由 $(n+1)$ 个 $n$ 元 $2$ 次方程构成，不是线性方程组。但是我们可以通过相邻 $2$ 个方程作差，把它变成 $n$ 个 $n$ 元 $1$ 次方程，同时消去常数 $C$：
$$
\sum_{j=1}^n (a_{i,j}^2-a_{i+1,j}^2-2x_j(a_{i,j}-a_{i+1,j}))=0\ \ \ \ \ \ \ \ (i=1,2,\ldots,n)
$$
把变量放在左边，常数放在右边：
$$
\sum_{j=1}^n 2(a_{i,j}-a_{i+1,j})x_j=\sum_{j=1}^n (a_{i,j}^2-a_{i+1,j}^2)\ \ \ \ \ \ \ \ (i=1,2,\ldots,n)
$$
这就是 $1$ 个线性方程组了。题目保证方程组有唯 $1$ 解，我们直接对增广矩阵进行高斯消元，即可得到每个 $x_j$ 的值。

##### $[\text{ANARC}2009]\ \texttt{Kind of a Blur}$

记原始矩阵和 $\text{blur}$ 矩阵分别为 $a$ 和 $b$。对于每个位置 $(i,j)$，容易求出与其 $\text{Manhattan}$ 距离不超过 $D$ 的所有位置，设共有 $k$ 个，记为 $(x_i,y_i)$，则有 $\frac{1}{k} a[x_1,y_1]+\frac{1}{k} a[x_2, y_2]+\ldots+\frac{1}{k} a[x_k,y_k]=b[i, j]$。这样总共得到 $W\times H$ 个 $W\times H$ 元 $1$ 次方程，对方程组求解即可。




#### 同余方程

增广矩阵的最后 $1$ 列不是解，而是解对 $p$ 求余的值。因此很可能会有多解， $1$ 般会指定范围，使解唯 $1$ 。可以先通过扩展欧几里得算法找到特解，进而移动到要求的范围内。

在模系下的矩阵变换并不影响求解，按照常规操作，每 $1$ 步均取模即可。

```cpp
int Gauss() {
	int x = 1, y = 1;
	for (; x <= n && y <= m; y++) {
		int k = x;
		for (int i = x + 1; i <= n; i++)
			if (Mat[i][y] > Mat[k][y]) k = i;
		if (Mat[k][y] == 0) continue;

		for (int i = 1; i <= m + 1; i++)
			swap(Mat[x][i], Mat[k][i]);

		for (int i = 1; i <= n; i++) {
			int Tmp = Mat[i][y];
			if (i != x && Mat[i][y] != 0)
				for (int j = 1; j <= m + 1; j++) Mat[i][j] = ((Mat[i][j] * Mat[x][y] - Mat[x][j] * Tmp) % 7 + 7) % 7; //MOD = 7 here
		}
		++x;

	}

	for (int i = x; i <= n; i++) if (Mat[i][m+1]) return -1; //Inconsistent Data

	if (x <= m || y <= m) return -2; //Multiple Solutions
	for (int i = 1; i <= m; i++) Ans[i] = Num[Mat[i][i]][Mat[i][m + 1]];

	return 0;
}
```





#### 异或方程组

形如 $a_1x_1\ \mathrm{xor}\ a_2x_2\ \mathrm{xor}\ \ldots a_nx_n=c $ 的方程被称作异或方程，其中所有系数都是 $0$ 或 $1$。

异或其实就是不进位加法，我们仍然可以写出增广矩阵，矩阵中的每个值要么是 $0$，要么是 $1$。然后，在执行高斯消元的过程中，把减法替换成异或，且不需要执行乘法。最终我们可以得到该异或方程组对应的主对角线矩阵。

选择主元的时候，只要找到任意 $1$ 个 $a[k, i] = 1$ 即可。

消元时，若当前行 $j$ 中当前主元 $x_i$ 的系数 $a[j,i]$ 为 $0$，那么不用消元；否则 $\forall\ 0\le k\le n$，有 $a[j, k]\leftarrow a[j, k]\ \mathrm{xor}\ a[i, k]$。

由于所有数都是 $0$ 或 $1$，我们可以使用位运算加速。

时间复杂度 $O(\frac{mn^2}{64})$。

```cpp
typedef unsigned long long ULL;

#define Get(x,y) ((Mat[(x)][(y)/64]>>ULL((y-1)%64))&1)

int Gauss() {
	if (m < n) return 0;
	int x, y;
	int Max = 0;
	for (x = 1, y = 1; x <= m && y <= n; y++, x++) {
		int k = x;
		while (Get(k, y) == 0 && k <= m) k++;
		if (k > m) return 0;
		Max = max(Max, k);

		for (int i = 0; i <= (n + 1) / 64; i++) swap(Mat[k][i], Mat[x][i]);

		for (int i = 1; i <= m; i++)
			if (i != x && Get(i,y) == 1)
				for (int j = 0; j <= (n + 1) / 64; j++) Mat[i][j] ^= Mat[x][j];
	}
	if (y == n + 1) return Max; //解出当前异或方程组所需的最少方程数
	return 0; //No solution
}
```



#### 行列式

行列式是符合 $1$ 定条件的公式的特殊写法。

$n$ 阶行列式的规则如下：

每 $1$ 行选择 $1$ 个数，假设第 $i$ 行选择的是第 $a_i$ 列的数。

要求满足 $a_i$ 互不相同，即每 $1$ 列也只能选 $1$ 个数。

将所有选到的数乘起来，然后再乘 $1$ 个系数 $(-1)^k$，其中 $k$ 是 $a_i$ 的逆序对数量。

在满足条件的情况下，把所有可能的结果加起来，就得到了 $n$ 阶行列式的展开式。

如 $ \left|\begin{array}{cccc}     1 &    2    & 3 & 4 \\     5 &    6   & 7 & 8\\     9 & 10 & 11 & 12 \\ 13 & 14 & 15 & 16 \end{array}\right| $，我们选择 $2, 7, 9, 16$，那么我们选择的 $a$ 为 $2, 3, 1, 4$。

逆序对数为 $2$，所以这 $1$ 项为 $2\times 7\times 9\times 16\times (-1)^2=2016$。



按照行列式的定义去计算很麻烦，我们需要用更好的方法。

行列式有以下的 $1$ 些性质：

- 交换任意 $2$ 行，结果是之前的相反数。
- 让 $1$ 行全部乘以 $k$，结果也乘以 $k$。
- 让 $1$ 行全部乘以 $k$ 再加到另外 $1$ 行上，结果不变。

这几个性质很像高斯消元的矩阵变换。

我们只要对行列式进行高斯消元，按照行列式的规则计算答案的变化即可。

最后剩下主对角线矩阵时，由于只有 $1$ 种选择方式不是 $0$，行列式的结果为主对角线上每 $1$ 个数乘起来。

```cpp
int Gauss(int n) {
	LL ans = 1;
	for (int x = 1, y = 1; x < n; x++, y++) {
		int k = x;
		for (; k <= n && Matrix[k][y] == 0; k++);
		if (k > n) return 0;
		if (k != x) {
			ans = MMul(ans, MOD - 1);
			for (int i = 1; i <= n; i++) swap(Matrix[x][i], Matrix[k][i]);
		}
		LL inv = GetRev(Matrix[x][y]);
		for (int i = x + 1; i <= n; i++) if (Matrix[i][y] != 0) {
				LL tmp1 = Matrix[x][y], tmp2 = Matrix[i][y];
				for (int j = y; j <= n; j++) {
					Matrix[i][j] = MMul(Matrix[i][j], tmp1) - MMul(Matrix[x][j], tmp2);
					if (Matrix[i][j] < 0) Matrix[i][j] += MOD;
				}
				ans = MMul(ans, inv);
			}
	}
	for (int i = 1; i <= n; i++) ans = MMul(ans, Matrix[i][i]);
	return ans;
}
```



#### 线性基

$n$ 维向量：$n$ 个数排在 $1$ 起 $v=<a_1, a_2, \ldots, a_n>$。

向量之间可以相加。$<a_1, a_2, \ldots,a_n>+<b_1, b_2, \ldots, b_n>=<a_1+b_1, a_2+b_2, \ldots, a_n+b_n>$。

向量整体可以乘上非 $0$ 常数 $k$。$<a_1,a_2,\ldots,a_n>\times k=<a_1\times k,a_2\times k,\ldots, a_n\times k>$。

若干个向量各自乘以系数之后相加，称作向量之间的线性组合，形如 $a_1v_1+a_2v_2+\ldots+a_nv_n$。

$n$ 维的向量最多可以有 $n$ 个向量作为**线性基**，使得任意 $1$ 个向量不可能是其他向量的线性组合。



将若干个向量排在 $1$ 起变成矩阵，高斯消元成上 $3$ 角矩阵，余下的就是这些向量的线性基。

用消元得到的线性基进行线性组合，可以得到之前所有的向量，和它们的任意线性组合。

换句话说，如果我们有 $m$ 个 $n$ 维向量，经过高斯消元得到线性基之后，我们只需要 $n$ 个 $n$ 维向量就可以表示原来的所有向量。
