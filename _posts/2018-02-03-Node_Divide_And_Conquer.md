---
layout:     post
title:      "点分治学习笔记"
subtitle:   "NOIp 提高组必备姿势之一"
date:       2018-02-03 10:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
tags:
    - 算法笔记
    - 点分治
---

> 树的分治算法是分治思想在树上的体现。通过对问题的分解和各子问题答案的合并，实现路径相关问题的高效解答。

所谓“分”，是指通过去除树上的某些部分，将整棵树分为若干互不相交的部分。根据除去对象的不同，树上的分治可分为点分治和边分治，本篇只讨论点分治。

### 重心

选取不同的点，将树进行分解，得到的子树大小是不相同的。为了使分治效果更明显，我们希望删去该点后得到的结点个数最多的子树的结点个数最少。该点就是树的重心，可以在 $O(n)$ 的时间内通过一次 $\text{DP}$ 求得。

```cpp
void getr(int u, int p) {
	f[u] = 0; w[u] = 1; //f[u] 为删去 u 后各子树中最多的结点数，w[u] 为以 u 为根的子树的结点数
	for (Edge *i = h[u]; i; i = i -> p) {
		int v = i -> t;
		if (v != p && !o[v]) { //o[v] 为 true 则结点 v 已被删去
			getr(v, u);
			f[u] = max(f[u], w[v]);
			w[u] += w[v];
		}
	}
	f[u] = max(f[u], n - w[u]);
	if (f[u] < f[r]) r = u;
}
```

可以证明，存在某点使得分出的子树的结点个数均不大于 $\frac{N}{2}$。则在点分治中每次树的结点个数减少一半，因此递归深度最坏 $O(\log N)$。

### 例题及分析

[传送门](http://poj.org/problem?id=1741)

给定一棵 $N(1\leq N\leq10000)$ 个结点的带权树和正整数 $K(1\leq K\leq10^9)$，定义 $dist(u,\ v)$ 为 $u,\ v$ 两点间的最短路径长度，求满足 $dist(u,\ v)\leq K$ 的**无序**点对个数。

考虑某个结点 $u$。以 $u$ 为根的子树中的合法路径分为两种：

- 完全在 $u$ 的某棵子树中，即不经过 $u$
- 从 $u$ 的子树 $v_1$ 出发，经过 $u$，到达 $u$ 的另一子树 $v_2$

显然根据分治思想，前一种情况在此处无需考虑，因为它被完全包含在子问题中，分治时会在子树中递归求解。

因此要考虑的就是第二种情况。先通过一次 $\text{DFS}$，可以求得以 $u$ 为根的子树中各结点到 $u$ 的距离 $d_i$。特别地，$d_u = 0$。

任意一条经过了点 $u$ 的合法路径 $(i, j)$，都由 $(u, i)$ 和 $(u, j)$ 两部分组成。即当前子树对答案的贡献为满足 $d_i + d_j\leq K$ 的点对 $(i, j)$ 数目。这里需要用到 $\text{two-pointers}$ 的方法，在 $O(n)$ 的时间内快速统计。

将 $d$ 数组升序排序后，初始时令指针 $l = 0,\ r = w(u) - 1$。对于每个 $l$，统计有多少 $r$ 能与之配对为合法路径。考虑到 $K$ 是定值，又根据数组的单调性，不难理解，随着 $l$ 的逐渐增大，能与之配对的最大 $r$ 也在不断地单调不增。

具体地，每次判断 $d_l + d_r \leq K$ 是否成立：

- 若成立，则所有满足 $l < r' < r$ 的 $r'$ 也可与 $l$ 搭配。因此答案要加上 $r - l$，并将 $l$ 右移；

- 若不成立，则将 $r$ 左移。

不断重复上述过程，直到 $l$ 与 $r$ 相遇为止。

然而，直接这样求会出现问题。有可能会出现起点和终点来自一棵子树 $v$ 的情况，即到达了 $u$ 之后再折返，则边 $(u,\ v)$ 被访问了两次，这并不符合路径的定义。解决方法也很简单，先把这个过大的答案算出来。在枚举每一棵子树时，递归求解之前先减去该子树中这些不合法的方案即可。

每层的时间复杂度不超过 $O(n\log n)$，最多递归 $\log n$ 次，因此总的时间复杂度为 $O(n\log^2n)$。

下面的代码中 `(*)` 标示了易错的语句。

```cpp
#include <algorithm>

#include <cstdio>

#include <cstdlib>

#include <cstring>

#include <iostream>

#include <vector>


using namespace std;

const int MAXN = 1e4 + 100;

struct Edge {
    int t, c;
    Edge *p;
    Edge () : p(NULL) {}
    Edge (int tt, int cc, Edge *pp) { t = tt; c = cc; p = pp; }
} *h[MAXN];

int n, k, ans;
int f[MAXN], w[MAXN], dis[MAXN];
int r;

bool ok[MAXN];

void getr(int u, int p) {
    f[u] = 0; w[u] = 1;
    for (Edge *i = h[u]; i; i = i -> p) {
        int v = i -> t;
        if (v != p && !ok[v]) {
            getr(v, u);
            f[u] = max(f[u], w[v]);
            w[u] += w[v];
        }
    }
    f[u] = max(f[u], n - w[u]);
    if (f[u] < f[r]) r = u;
}

vector <int> d;

void solve_dis(int u, int p) {
    d.push_back(dis[u]);
    for (Edge *i = h[u]; i; i = i -> p) {
        int v = i -> t;
        if (v != p && !ok[v]) {
            dis[v] = dis[u] + i -> c;
            solve_dis(v, u);
        }
    }
}

int solve_ans(int u, int o_dis) {
    d.clear();
    dis[u] = o_dis; solve_dis(u, 0);
    sort(d.begin(), d.end()); int s = 0;
    for (int l = 0, r = (signed)d.size() - 1; l < r; )
        if (d[l] + d[r] <= k) s += r - l++; else --r;
    return s;
}

void dfs(int u) {
    ans += solve_ans(u, 0); ok[u] = true;
    for (Edge *i = h[u]; i; i = i -> p) {
        int v = i -> t;
        if (!ok[v]) {
            ans -= solve_ans(v, i -> c);
            f[0] = n = w[v]; //(*)

            getr(v, r = 0);
            dfs(r);
        }
    }
}

int main(void) {
    for (; ~scanf("%d%d", &n, &k) && n; ) {
        memset(h, 0, sizeof h);
        for (int i = 1; i < n; i++) {
            int u, v, l; scanf("%d%d%d", &u, &v, &l);
            h[u] = new Edge(v, l, h[u]);
            h[v] = new Edge(u, l, h[v]);
        }
        f[r = 0] = n; memset(ok, false, sizeof ok); getr(1, 0); //(*)

        ans = 0; dfs(r); printf("%d\n", ans);
    }
    return 0;
}
```


### $\text{Another Solution}$

这题还有另一种更普遍，更好写的做法：平衡树 + 启发式合并。套不套点分治都可以。

每个结点用一棵平衡树维护以自己为根的子树中各结点的 $d$ 值，向父亲合并时，先全部计算完答案，再统一合并平衡树，这样就避免了错误方案。

计算答案也很简单。定义结点 $u$ 和 $v$ 到 $\text{LCA}$ 结点 $z$ 的距离分别为 $f_u$ 和 $f_v$，显然 $f_u = d_u - d_z$，$f_v$ 同理。

将它们代入 $f_u + f_v \leq K$ ，得到 $d_u - d_z + d_v - d_z \leq K$，整理得 $d_v \leq K + d_z \times 2 - d_u$。至此，问题迎刃而解。

总的时间复杂度与直接尺取法做点分治是相近的。

注意平衡树如果将相同值的结点压缩到一起，对数量的相关操作要时刻谨慎。因为这一点贡献了 $4$ 条 $\text{WA}$ 记录。

```cpp
#include <algorithm>

#include <cstdio>

#include <cstdlib>

#include <cstring>

#include <iostream>

#include <vector>


using namespace std;

const int MAXN = 2e4 + 100;

struct Edge {
	int t, c;
	Edge *p;
	Edge (int x, int y, Edge *z) : t(x), c(y), p(z) {}
} *h[MAXN];

struct Treap {
	struct Node {
		int f, v, c, w;
		Node *s[2];
		Node (int x, int xc = 1) : f(rand()), v(x), c(xc), w(xc) { s[0] = s[1] = NULL; } //(*)

		void upd() { w = (s[0] ? s[0] -> w : 0) + c + (s[1] ? s[1] -> w : 0); }
	} *r;
	Treap () { r = NULL; }
	void clear() { r = NULL; }
	int size() { return r -> w; }
	void rotate(Node *&u, int d) {
		Node *s = u -> s[d ^ 1];
		u -> s[d ^ 1] = s -> s[d]; u -> upd();
		s -> s[d] = u; s -> upd();
		u = s;
	}
	void insert(Node *&u, int x, int xc) { //(*)

		if (!u) { u = new Node(x, xc); return; } //(*)

		if (u -> v == x) { u -> c += xc; u -> w += xc; return; } //(*)

		int t = x > u -> v;
		insert(u -> s[t], x, xc);
		if (u -> s[t] -> f < u -> f) rotate(u, t ^ 1); else u -> upd();
	}
	int ask(Node *u, int x) {
		if (!u) return 0;
		if (x < u -> v) return ask(u -> s[0], x);
		if (x == u -> v) return (u -> s[0] ? u -> s[0] -> w : 0) + u -> c;
		return (u -> s[0] ? u -> s[0] -> w : 0) + u -> c + ask(u -> s[1], x);
	}
	int getans(Node *u, Node *v, int x) {
		if (!u) return 0;
		return getans(u -> s[0], v, x) + u -> c * ask(v, x - u -> v) + getans(u -> s[1], v, x); //(*)

	}
	void addto(Node *u, Node *&v) {
		if (!u) return;
		insert(v, u -> v, u -> c);
		addto(u -> s[0], v); addto(u -> s[1], v);
	}
} tr[MAXN];

int n, k, ans;
int d[MAXN];

void getdis(int u, int p) {
	tr[u].clear(); tr[u].insert(tr[u].r, d[u], 1);
	for (Edge *i = h[u]; i; i = i -> p) {
		int v = i -> t;
		if (v != p) {
			d[v] = d[u] + i -> c;
			getdis(v, u);
		}
	}
}

void dfs(int u, int p) {
	for (Edge *i = h[u]; i; i = i -> p)	{
		int v = i -> t;
		if (v != p) {
			dfs(v, u);
			if (tr[u].size() < tr[v].size()) swap(tr[u].r, tr[v].r); //启发式
			
			ans += tr[v].getans(tr[v].r, tr[u].r, k + (d[u] << 1));
			tr[v].addto(tr[v].r, tr[u].r);
		}
	}
}

int main(void) {
	for (; ~scanf("%d%d", &n, &k) && n; ) {
		memset(h, 0, sizeof h);
		for (int i = 1; i < n; i++) {
			int u, v, l; scanf("%d%d%d", &u, &v, &l);
			h[u] = new Edge(v, l, h[u]);
			h[v] = new Edge(u, l, h[v]);
		}
		d[1] = 0; getdis(1, 0);
		ans = 0; dfs(1, 0); printf("%d\n", ans);
	}
	return 0;
}
```
