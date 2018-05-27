---
layout:     post
title:      "可持久化线段树入门学习笔记"
subtitle:   "观海听涛"
date:       2018-05-18 11:00:00
author:     "Ufowoqqqo"
header-img: "img/hjt.jpg"
mathjax:    true
catalog:    true
tags:
    - 数据结构笔记
    - 可持久化线段树
---

### $\text{Simple Segment Tree}$

**线段树**（$\text{Segment Tree}$）是一种对序列进行操作的高级数据结构，支持单点和区间的修改和询问，用于处理一类满足区间可加性的问题。

线段树的每个结点对应序列上的一段连续区间（叶子结点则对应单个点）。对于树上的每个结点 $[L, R]$，令 $mid=\lfloor\frac{L+R}{2}\rfloor$，其左儿子为 $[L, mid]$，右儿子为 $(mid, R]$。

由于线段树的父结点区间是平均分割到左右子树，因此线段树是完全二叉树，**对于包含 $n$ 个叶子结点的完全二叉树，它一定有 $(n-1)$ 个非叶结点，总共 $(2n-1)$ 个结点**，实际空间是满二叉树的结点数目。因此存储线段树需要的空间复杂度是 $O(n) $ 级别的（为了保险，一般情况下系数取 $4$ 左右，视具体题目而定）。

单点修改、询问和区间询问都不难实现，只需找到对应结点或将对应区间划分成 $\log n$ 个结点之后进行处理即可。

但对于区间修改，若某个区间被完全覆盖，其子树中的所有结点本应被全部修改，但如果真的递归下去做，会消耗大量时间，但却不一定有用，因为后面的询问也许不涉及被修改的子区间。

由此诞生了**延迟标记**（$\text{Lazy Tag}$）。每个结点新增一个标记，记录该结点是否被修改。对于任意区间的修改，先按区间询问的方式划分为线段树中的结点，并修改它们的信息。之后给这些结点标记上代表这种修改操作的标记。

在修改和询问时，若到达了某个节点 $p$，并将要递归其子结点，此时要检查结点 $p$ 是否被标记。若有，则按照标记修改其子结点的信息，并对子结点进行相应标记。最后消掉节点 $p$ 的标记。

### $\text{Weight Segment Tree}$

一般情况下，线段树对应的“区间”是序列下标。但也可以借助数组计数的思想，对应具体的数值，统计具体数字出现的次数。这就是所谓的权值线段树（$\text{Weight Segment Tree}$）。需要注意的是，大多数题目的权值范围较大，因此要离散化之后再处理。为了方便处理，可以添加 $\infty$ 和 $-\infty$ 作为边界。

例题：求 $n$ 个数中第 $k$ 小的数。

将这些数离散化，建立对应权值线段树。显然答案具有单调性，考虑二分答案。当前数为 $m$，如何确定其排名？

若有 $i$ 个数比 $m$ 小，则 $m$ 的排名（从小到大）为 $i+1$。由此，只需确定 $n$ 个数中小于 $m$ 的数的个数即可。

当然可以用 $\text{Fenwick Tree}$ 做，如果用权值线段树也很简单，求区间 $[\infty, m)$ 的权值，就得到了 $m$ 的排名。

### $\text{Chairman Tree}$

**可持久化**数据结构指保存该数据结构的所有历史版本，以便回退到任意时刻，同时利用各版本之间的共用数据减少时间和空间的消耗。顾名思义，可持久化线段树就是支持保存和回退历史版本的线段树。如下图。

![Persistent Segment Tree](https://s3.amazonaws.com/hr-challenge-images/8565/1433394877-559bcd278f-Treepers.png)

可持久化数据结构有一个通性：从不修改或删除，只新增。在建好了最初版本的线段树后，若要对某个点进行修改，步骤如下：

- 对于该点对应的叶子结点，不改动原结点，而是开一个与原结点完全相同的新结点，并对新结点进行改动；
- 对于非叶子结点，必定有且只有一个儿子被修改，则开一个与被修改的子结点完全相同的新结点，并在新结点对应的子树中继续递归修改；另一个儿子没有被修改，要充分利用不变量，因此连向旧版本的对应子结点。

容易看出，在建立了 $n$ 个版本的树之后，我们得到了 $n$ 个根结点。当要回退到某一历史版本时，只需从对应的根结点进入即可。

由于每一次只新增被修改结点的祖先，最多只会新增 $\log n$ 个，因此总的空间复杂度是 $O(n\log n)$。

---

例题：给定 $n$ 个绝对值在 $10^9$ 内的数和 $m$ 个询问，每次询问某个区间内第 $k$ 小。

与上一道例题唯一的不同之处在于此题询问的区间范围由 $[1, n]$ 变成了 $[l, r]$。同样可以二分答案。瓶颈在于如何确定区间内小于 $mid$ 的数字个数。

借助前缀和的思想，要得到 $[l, r]$ 内的权值情况，可以用 $r$ 的权值情况减去 $l-1$ 的权值情况。例如：

> array = {3 1 2 5 4}
>
> weights of version #1: 0 0 1 0 0
>
> weights of version #4: 1 1 1 0 1
>
> weight of [2, 4]: 1 1 0 0 1

由此得到启发，我们建立 $(n+1)$ 棵权值线段树，第 $i$ 棵保存 $[1, i]$ 的权值情况。考虑到第 $i$ 棵与第 $(i-1)$ 棵有且只有一个结点不同，恰好符合可持久化线段树的作用。

二分答案和查询的时间复杂度都为 $O(\log n)$，因此总的时间复杂度为 $O(m\log^2 n)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e5 + 100;

struct SegmentTree {
	struct Node {
		Node *c[2];
		int v;
		int l, r, m;
		Node (): v(0), l(0), r(0), m(0) { c[0] = c[1] = NULL; }
		Node (int x, int y): v(0), l(x), r(y), m(x + y >> 1) { c[0] = c[1] = NULL; }
		void pushup() { v = c[0] -> v + c[1] -> v; }
	} *r[MAXN], nodes[MAXN * 20], *cur;
	SegmentTree () { memset(r, 0, sizeof r); cur = nodes; }

	void make(Node *&u, int l, int r) { *cur = Node(l, r); u = cur++; if (l < r) { make(u -> c[0], l, u -> m); make(u -> c[1], u -> m + 1, r); } }

	void add(Node *last, Node *&u, int p) {
		*cur = Node(last -> l, last -> r); u = cur++; if (u -> l == p && u -> r == p) { u -> v = last -> v + 1; return; }
		int t = u -> m < p; add(last -> c[t], u -> c[t], p); u -> c[t ^ 1] = last -> c[t ^ 1]; u -> pushup();
	}
	
	int ask(Node *u, int l, int r) { if (r < u -> l || u -> r < l) return 0; if (l <= u -> l && u -> r <= r) return u -> v; return ask(u -> c[0], l, r) + ask(u -> c[1], l, r); }
} t;

pair <int, int> num[MAXN];
int rev1[MAXN], rev2[MAXN];
int n;

int main(void) {
	int m; scanf("%d%d", &n, &m);
	t.make(t.r[0], 1, n);
	for (int i = 1; i <= n; i++) { scanf("%d", &num[i].first); num[i].second = i; }
	sort(num + 1, num + n + 1);
	for (int i = 1; i <= n; i++) { rev1[i] = num[i].first; rev2[num[i].second] = i; }
	for (int i = 1; i <= n; i++) t.add(t.r[i - 1], t.r[i], rev2[i]);
	for (; m--; ) { int i, j, k; scanf("%d%d%d", &i, &j, &k); printf("%d\n", solve(i, j, k)); }
	return 0;
}
```
