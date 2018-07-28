---
layout:     post
title:      "后缀数组倍增算法学习笔记"
date:       2018-07-23 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 算法笔记
    - 后缀数组
---



> 后缀数组是处理字符串问题的有力工具。——罗穗骞

### 基数排序

先从 $1$ 个简单的问题入手：给定 $10^7$ 个 $10^6$ 以内的数，要求将它们稳定排序。

在 $1s$ 时限内使用 $1$ 般的时间复杂度为 $O(n\log n)$ 的算法（在不经常数优化的前提下）是无法通过的。

注意到数字的范围不大，考虑基数排序，利用相等数字在最终序列中是连续 $1$ 段的性质。

> 不同于常规的排序算法，基数排序不进行关键字的比较，而是利用**分配**和**收集**。

​记 `cnt[i]`  为数字 $i$ 的个数，对 `cnt[]`  求前缀和。此时，`cnt[i]` 的含义就变成了最后出现的数字 $i$ 的排名。

因此对于从后往前的第 $1$ 个数字 $i$，令 `ans[cnt[i]] = i, cnt[i]--;`。这样，在下次遇到倒数第 $2$ 个 $i$ 时，就知道了它对应的排名。

这样倒着扫 $1$ 遍就在严格 $O(n)$ 时间内完成了排序。

```cpp
//对字符集大小为 26 的串中各字符进行稳定排序
memset(Cnt, 0, sizeof(Cnt));
for (int i = 1; i <= n; i++) Cnt[S[i]]++;
for (int i = 1; i <= 26; i++) Cnt[i] += Cnt[i-1];
for (int i = n; i >= 1; i--) {
	A[Cnt[S[i]]] = i;
	Cnt[S[i]]--;
}
```

考虑双关键字的稳定排序。按先第 $2$ 关键字有序、后第 $1$ 关键字有序的顺序进行稳定排序即可。

原因是显然的。在排序结果中，第 $1$ 关键字的有序性是首要条件，第 $2$ 关键字的有序性是次要条件。假如先按第 $1$ 关键字有序，再按第 $2$ 关键字有序，就会在后 $1$ 次排序时破坏第 $1$ 关键字的有序性。反之，对于第 $1$ 关键字相等的某些元素，由于基数排序的稳定性，在对第 $1$ 关键字排序的过程中能够继承按第 $2$ 关键字排序时它们的相对顺序。

注意按第 $2$ 关键字排序的中间结果应该用临时数组保存，不能覆盖了排序过程中需要读取的原数组。

```cpp
memset(Cnt, 0, sizeof(Cnt)); //第二关键字
for (int i = 1; i <= n; i++) Cnt[S[i][2]]++;
for (int i = 1; i <= 26; i++) Cnt[i] += Cnt[i - 1];
for (int i = n; i >= 1; i--) B[Cnt[S[i][2]]--] = i;

memset(Cnt, 0, sizeof(Cnt)); //第一关键字
for (int i = 1; i <= n; i++) Cnt[S[B[i]][1]]++;
for (int i = 1; i <= 26; i++) Cnt[i] += Cnt[i - 1];
for (int i = n; i >= 1; i--) A[Cnt[S[B[i]][1]]--] = B[i];
//注意是按照第二关键字的排序结果 B 枚举的
```



### 概念

后缀 $i$：原串 `S[1..n]`  的某个子串 `S[i..n]`，其中 $i$ 为后缀 $i$ 的编号。

后缀数组（$\text{Suffix Array}$）：将串 $S$ 的所有后缀按字典序排序得到的数组。`SA[i]` 为排名为 $i$ 的后缀的编号。

名次数组 `rank[i]` 为后缀 $i$ 的排名。

不难发现 `SA[]` 与 `rank[]` 是互逆的，即 `SA[rank[i]] = rank[SA[i]]=i`。

求出后缀数组的方法不止 $1$ 种。本文介绍最为常见的倍增算法，即第 $1$ 轮对所有后缀按前 $2$ 位进行排序，第 $2$ 轮按前 $4$ 位进行排序……$1$ 直进行 $\log n$ 轮。



### 算法流程

采用从特殊到 $1$ 般的方式比较便于讲解。

考虑各前缀已经按前 $2$ 位排好序，本轮对前 $4$ 位排序。

把前 $2$ 位合起来视作第 $1$ 关键字，第 $3, 4$ 位视作第 $2$ 关键字。这样就遇到了问题：在按单个字符进行基数排序时，我们简单地把字符根据对应的 `ASCII` 码（或减去某个定值）索引成下标。对于多个字符如何索引？

对于常规的字符串确实没有好的方法，但注意到后缀的特点，即后缀 $i$ 的第 $3,4$ 位必然是后缀 $i+2$ 的开头。也就是说，**所有后缀的前 $2$ 位必然包含了所有后缀的第 $3,4$ 位**。请务必确保充分理解这个特性之后，再向下阅读。

可以借助上 $1$ 轮对排序的结果用于本轮的索引。因为我们在进行索引时只关心**相对大小**，而排序就是负责解决相对大小的顺序问题的。

具体地，在本轮排序开始前，`rank[i]` 表示后缀 $i$ 的前 $2$ 位的排名，因此 `rank[SA[i] + 2]` 就是排名为 $i$ 的后缀的第 $3, 4$ 位的排名。这样就可以索引并进行基数排序了。

对第 $3, 4​$ 位的排序完成后，保留上 $1​$ 轮遗下原有的 `rank[]` 数组，**按第 $3, 4​$位排好序之后的顺序枚举所有后缀**，按前 $2​$ 位进行排序，就得到了按前 $4​$ 位稳定排序的结果，此时再先后更新 `SA[]` 和 `rank[]` 数组。

`rank[SA[i]]` 的计算不难理解：若排名为 $i$ 的后缀与排名为 $(i-1)$ 的后缀前 $4$ 位完全相同（`rank[SA[i]] == rank[SA[i - 1]] && rank[SA[i] + 2] = rank[SA[i - 1] + 2]`），则 `rank[SA[i]] = rank[SA[i - 1]]`，否则当前相对排名 $+1$。虽然由于可能存在并列，实际排名不止 $+1$，但我们只关心相对大小，因此没有影响。

注意到在更新 `rank[]` 过程中的判断仍然要用到上 $1$ 轮中的 `rank[]`，因此新 `rank[]` 也应该用临时数组保存，计算完毕之后再复制回去。

类似地，再按前 $8$ 位、前 $16$ 位……倍增地排序，直到不存在并列排名（`rank[SA[n]] == n`）或当前轮排序长度超出 $n$ 为止。

一般地，当前轮按前 $2L$ 位排序时，第 $1, 2$ 关键字分别为 $i$ 和 $i+L$。

总的时间复杂度为 $O(n\log n)$。

```cpp
int m = 0; //最大排名
//给第 1 个字符排序
for (int i = 1; i <= n; i++) {
	Rank[i] = S[i] - 'a';
	m = max(m, Rank[i]);
}
memset(Cnt, 0, sizeof(Cnt));
for (int i = 1; i <= n; i++) Cnt[Rank[i]]++;
for (int i = 1; i <= m; i++) Cnt[i] += Cnt[i-1];
for (int i = n; i >= 1; i--) SA[Cnt[Rank[i]]--] = i;

for (int L = 1; L <= n; L *= 2) {
	memset(Cnt, 0, sizeof(Cnt)); //第 2 关键字
	for (int i = 1; i <= n; i++) Cnt[Rank[SA[i]+L]]++;
	for (int i = 1; i <= m; i++) Cnt[i] += Cnt[i - 1];
	for (int i = n; i >= 1; i--) A[Cnt[Rank[SA[i] + L]]--] = SA[i];

	memset(Cnt, 0, sizeof(Cnt)); //第 1 关键字
	for (int i = 1; i <= n; i++) Cnt[Rank[A[i]]]++;
	for (int i = 1; i <= m; i++) Cnt[i] += Cnt[i - 1];
	for (int i = n; i >= 1; i--) SA[Cnt[Rank[A[i]]]--] = A[i];

	R[SA[1]] = 1; //排名
	for (int i = 2; i <= n; i++)
		if (Rank[SA[i]] == Rank[SA[i - 1]] && Rank[SA[i] + L] == Rank[SA[i - 1] + L]) R[SA[i]] = R[SA[i - 1]];
		else R[SA[i]] = R[SA[i - 1]] + 1;

	memcpy(Rank, R, sizeof(Rank));
	m = Rank[SA[n]];//更新最大排名
}
```



### 优化及细节

#### 按第 $2$ 关键字的排序

事实上每次对第 $2$ 关键字没有必要重新进行基数排序，可以直接利用上 $1$ 轮排序的结果得到。

首先考虑第 $2$ 关键字以 `\0` 开头的所有后缀，它们肯定排在最前面，并且由于稳定排序，编号递增。

```cpp
int p = 0;
for (int i = n - L + 1; i <= n; i++) A[++p] = i;
```

对于剩余的后缀，考虑自身前 $L$ 位作为其他后缀的第 $2$ 关键字（第 $L+1\sim 2L$ 位）所作出的贡献。注意对于 $SA[i]<L$ 的后缀，并不能作为其他后缀的第 $2$ 关键字。

例如对于原先的 `SA[1]`，假设其大于 $L$，则它是 `SA[1] - L` 的第 $2$ 关键字，且在本轮排序中不存在更大的第 $2$ 关键字，因此排在所有空后缀之后的首个就是 `SA[1] - L`。其余依此类推。

```cpp
for (int i = 1; i <= n; i++) if (SA[i] > L) A[++p] = SA[i] - L;
```

当然，由于基数排序本身就是 $O(n)$ 级别的，因此这个小优化只是略微减小了常数。实际上按照常规的写法并不会受到太大影响。

#### 用双指针改造辅助数组

注意到原先有 `memcpy(Rank, R, sizeof(Rank));` 这样的 $1$ 句代码，实际上是没有必要的。

在对第 $2$ 关键字排序的过程中，我们用到了辅助数组 `A[]`，而它在对第 $1$ 关键字排序完成后就失去了作用，不妨尝试把它利用起来。

具体地，我们将新的排名直接存储在这个临时数组中，供下 $1$ 轮排序时使用。然而，直接赋值在数组中显然是不正确的，因为下 $1$ 轮的临时结果也要存储在这个数组中。

这时就要用上指针的技巧。我们总共只需开辟 $2$ 个数组 `tmp1[]` 和 `tmp2[]`，并在排序开始前令 `x` 指向 `tmp1[]`，`y` 指向 `tmp2[]`。规定在每轮开始前，`x` 指向的数组中储存着上 $1$ 轮的排名，`y` 指向的数组作为临时数组。

按第 $1$ 关键字排序完毕后，**执行 `swap(x, y);`**。此时 `x` 指向之前的临时数组，可以直接覆盖用于储存本轮的排名；`y` 指向的数组仍然储存着上轮的排名，用于更新本轮排名时的比较计算。

更新完排名后，`x` 指向的数组用于下 $1$ 轮时所需的本轮排名，`y` 指向的数组中的值相对于下 $1$ 轮而言已经是“上上轮”的排名，没有价值，用作下 $1$ 轮第 $2$ 关键字排序时的辅助数组。

最终，排序完成后，再根据定义式填写 `rank[]` 数组。

#### 其余细节

注意到判断中会出现 `Rank[SA[i] + L]`，考虑是否有数组越界导致 `RE` 的风险。

事实上是不会的。

`C++` 的逻辑判断有 $1$ 个特性：对于逻辑与，从左到右判断，$1$ 旦不为真就停止。假如 `SA[i] + L > n`，即 `SA[i] > n - L`，那么第 $1$ 关键字的末尾就已经有 `\0`，不可能相等，自然也不会判断第 $2$ 关键字。

不过为了稳妥起见，在内存限制足够的情况下，将数组空间的申请适当放宽 $1$ 些也是有必要的。



### 高度数组

> 后缀的前缀是匹配。

定义 `height[i]` 为所有后缀排好序后，排名 $i$ 的后缀与排名 $i - 1$ 的后缀的最大公共前缀（Longest Common Prefix, LCP）。

不难理解，任意后缀 $i, j(i < j)$ 之间的最大公共前缀就是 $\mathrm{LCP}(i, j)=\min\{height[i+1], height[i+2], \ldots, height[j]\}$。

求解高度数组，直接对于每对相邻后缀暴力匹配显然是不可行的，我们需要更高效的方法。

这里就要用到 $1$ 个引理：$height[rank[i]]\ge height[rank[i-1]] - 1$。

也就是说，从 $i$ 开头的后缀，它和排在它前 $1$ 个的 LCP，最少比 $i - 1$ 开头的后缀和它前 $1$ 个的 LCP 小 $1$。

换句话说，如果我们求出了从 $i - 1$ 开头的后缀和排在它前 $1$ 个的后缀的 LCP，即 `height[rank[i - 1]]`，我们在求 `height[rank[i]]` 的时候，就可以从 `height[rank[i - 1]] - 1` 开始枚举，而不是从 $0$ 开始枚举长度。这样总共最多往前 $n$ 步，总时间复杂度就是 $O(n)$ 了。

> 证明：
>
> 设 $height[rank[i]] = H[i]$，$SA[rank[i - 1] - 1] = k$。
>
> 分类讨论。当 $H[i - 1]\le 1$ 时，显然成立。
>
> 当 $H[i - 1] > 1$ 时，
>
> ![](/img/SA_0.png)
>
> 删去相等的首位后，仍然有 $1$ 部分是相同的，所以 $k + 1$ 肯定排在 $i$ 前面（`rank[k + 1] < rank[i]`），而 $\mathrm{LCP}(i, k + 1) = H[i - 1] - 1$。
>
> 即 $\min\{height[rank[k + 2]], height[rank[k + 2] + 1],\ldots, height[rank[i]]\} = H[i - 1] - 1$。
>
> 证毕。

```cpp
for (int i = 1, k = 0; i <= n; i++) //求 Height
{
	if (k > 0) k--;
	if (Rank[i] > 0) while (S[i + k] == S[SA[Rank[i] - 1] + k]) ++k;
	Height[Rank[i]] = k;
}
```



### 例题

#### $\texttt{SPOJ DISUBSTR}$

求不同子串个数。

由于“后缀的前缀是子串（匹配）”，相同的子串在排好序的 `SA[]` 中必然相邻，原问题等价于求所有后缀之间的不相同的前缀的个数。如果所有的后缀按照 $i$ 递增的顺序计算 `SA[i]`，不难发现，对于每 $1$ 次新加进来的后缀 `SA[i]`，它将产生 $(n-SA[i]+1)$ 个新的前缀。但是其中有 $height[i]$ 个是和前面的字符串的前缀是相同的。

所求即为 $\sum_{i=1}^n (n-SA[i]+1-height[i])$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 1e4;

char s[MAXN];
int cnt[MAXN], SA[MAXN], rank[MAXN], height[MAXN];
int tmp1[MAXN], tmp2[MAXN];

vector <char> v;

int main(void) {
	int T;
	scanf("%d", &T);
	for (int i = 0; i < T; i++) {
		memset(cnt, 0, sizeof cnt);
		memset(SA, 0, sizeof SA);
		memset(rank, 0, sizeof rank);
		memset(height, 0, sizeof height);
		memset(tmp1, 0, sizeof tmp1);
		memset(tmp2, 0, sizeof tmp2);
		v.clear();
		scanf("%s", s + 1);
		int n = strlen(s + 1), m = 0, *x = tmp1, *y = tmp2;
		for (int i = 1; i <= n; i++) v.push_back(s[i]);
		sort(v.begin(), v.end());
		v.erase(unique(v.begin(), v.end()), v.end());
		for (int i = 1; i <= n; i++) {
			m = max(m, x[i] = lower_bound(v.begin(), v.end(), s[i]) - v.begin() + 1);
			++cnt[x[i]];
//			printf("%d ", x[i]);
		}
//		putchar('\n');
		for (int i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
		for (int i = n; i; i--) SA[cnt[x[i]]--] = i;
		for (int L = 1; L <= n; L <<= 1) {
			int p = 0;
			for (int i = n - L + 1; i <= n; i++) y[++p] = i;
			for (int i = 1; i <= n; i++) if (SA[i] > L) y[++p] = SA[i] - L;
			memset(cnt, 0, sizeof cnt);
			for (int i = 1; i <= n; i++) ++cnt[x[y[i]]];
			for (int i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
			for (int i = n; i; i--) SA[cnt[x[y[i]]]--] = y[i];
			swap(x, y);
			x[SA[1]] = m = 1;
			for (int i = 2; i <= n; i++)
				if (y[SA[i]] == y[SA[i - 1]] && y[SA[i] + L] == y[SA[i - 1] + L]) x[SA[i]] = m;
				else x[SA[i]] = ++m;
			if (m == n) break;
		}
		for (int i = 1; i <= n; i++) rank[SA[i]] = i;

		for (int i = 1, k = 0; i <= n; i++) {
			if (k) --k;
			if (rank[i]) for (; s[i + k] == s[SA[rank[i] - 1] + k]; ++k);
			height[rank[i]] = k;
		}

		int ans = 0;
		for (int i = 1; i <= n; i++) ans += (n - i + 1) - height[i];
		printf("%d\n", ans);
	}
	return 0;
}
```



#### $\texttt{POJ 2774}$

求 $2$ 个串的最长公共子串。

把 $2$ 个串拼接在 $1$ 起，中间用串中不会出现的字符隔开。求出高度数组后，考虑 `SA[]` 中所有相邻的后缀。没有必要考虑不相邻的后缀是因为取 $\min$ 保证了范围越远结果 $1$ 定不会更优。对于每组相邻的后缀，若分属原来的 $2$ 个串，则它们的 `LCP` 就对应了原先 $2$ 个串的某组公共子串。取所有合法组中最大的 $height[]$ 即为所求。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e6;

char S[MAXN];
int cnt[MAXN], SA[MAXN], rank[MAXN], height[MAXN];
int tmp1[MAXN], tmp2[MAXN];

int main(void) {
	scanf("%s", S + 1);
	int d = strlen(S + 1);
	S[d + 1] = '~';
	scanf("%s", S + d + 2);
	int n = strlen(S + 1);
//	puts(S + 1); printf("%d %d\n", d, n);
	int *x = tmp1, *y = tmp2, m = 0;
	for (int i = 1; i <= n; i++) m = max(m, x[i] = S[i] - 'a' + 1);
	for (int i = 1; i <= n; i++) ++cnt[x[i]];
	for (int i = 2; i <= m; i++) cnt[i] += cnt[i - 1];
	for (int i = n; i; i--) SA[cnt[x[i]]--] = i;

	for (int L = 1; L <= n; L <<= 1) {
		int p = 0;
		for (int i = n - L + 1; i <= n; i++) y[++p] = i;
		for (int i = 1; i <= n; i++) if (SA[i] > L) y[++p] = SA[i] - L;

		memset(cnt, 0, sizeof cnt);
		for (int i = 1; i <= n; i++) ++cnt[x[y[i]]];
		for (int i = 2; i <= m; i++) cnt[i] += cnt[i - 1];
		for (int i = n; i; i--) SA[cnt[x[y[i]]]--] = y[i];

		swap(x, y);
		x[SA[1]] = m = 1;
		for (int i = 2; i <= n; i++)
			if (y[SA[i]] == y[SA[i - 1]] && y[SA[i] + L] == y[SA[i - 1] + L]) x[SA[i]] = m;
			else x[SA[i]] = ++m;
		if (m == n) break;
	}
	for (int i = 1; i <= n; i++) rank[SA[i]] = i;

	for (int i = 1, k = 0; i <= n; i++) {
		if (k) --k;
		if (rank[i]) for (; S[i + k] == S[SA[rank[i] - 1] + k]; ++k);
		height[rank[i]] = k;
	}

	int ans = 0;
	for (int i = 2; i <= n; i++) if (SA[i - 1] <= d && d + 1 < SA[i] || d + 1 < SA[i - 1] && SA[i] <= d) ans = max(ans, height[i]);
	printf("%d\n", ans);
	return 0;
}
```

 

#### $\texttt{HYSBZ 1031}$

将给定串的所有表示法按字典序排序，按顺序输出最后位的字符。

参考动态规划中破环成链的思想，将原串复制 $1$ 份添在末尾，则后缀 $1$ 至 $\frac{n}{2}$ 各对应了某种表示法。将所有后缀排序，忽略不合法后缀，按顺序输出对应位字符即可。

```cpp
/**************************************************************
    Problem: 1031
    User: Ufowoqqqo
    Language: C++
    Result: Accepted
    Time:1036 ms
    Memory:18024 kb
****************************************************************/

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 1e6;

char s[MAXN];
int cnt[MAXN], SA[MAXN];
int tmp1[MAXN], tmp2[MAXN];

vector <char> sigma;

int main(void) {
	scanf("%s", s + 1);
	int n = strlen(s + 1), m = 0, *x = tmp1, *y = tmp2;
	for (int i = 1; i <= n; i++) sigma.push_back(s[i]);
	sort(sigma.begin(), sigma.end());
	sigma.erase(unique(sigma.begin(), sigma.end()), sigma.end());
	for (int i = n + 1; i <= (n << 1); i++) s[i] = s[i - n];
	n <<= 1;
	for (int i = 1; i <= n; i++) {
		m = max(m, x[i] = lower_bound(sigma.begin(), sigma.end(), s[i]) - sigma.begin() + 1);
		++cnt[x[i]];
	}
	for (int i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
	for (int i = n; i; i--) SA[cnt[x[i]]--] = i;
	for (int L = 1; L <= n; L <<= 1) {
		int p = 0;
		for (int i = n - L + 1; i <= n; i++) y[++p] = i;
		for (int i = 1; i <= n; i++) if (SA[i] > L) y[++p] = SA[i] - L;
		memset(cnt, 0, sizeof cnt);
		for (int i = 1; i <= n; i++) ++cnt[x[y[i]]];
		for (int i = 1; i <= m; i++) cnt[i] += cnt[i - 1];
		for (int i = n; i; i--) SA[cnt[x[y[i]]]--] = y[i];
		swap(x, y);
		x[SA[1]] = m = 1;
		for (int i = 2; i <= n; i++)
			if (y[SA[i]] == y[SA[i - 1]] && y[SA[i] + L] == y[SA[i - 1] + L]) x[SA[i]] = m;
			else x[SA[i]] = ++m;
		if (m == n) break;
	}
	for (int i = 1; i <= n; i++) if ((SA[i] << 1) <= n) putchar(s[SA[i] + (n >> 1) - 1]);
	return 0;
}
```

