---
layout:     post
title:      "2018 新高 1 A 班暑假测试总结"
date:       2018-07-02 22:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
tags:
    - 比赛总结
---

### $\text{#}1$

#### $\text{T}1$

先放结论：求逆序对数即可。


感性证明：

由于整个序列中忽略掉某个数之后满足单调不减，显然只需要想办法将破坏单调性的数恢复到原位即可。这样，在原位之前和当前位置之后的数都不需考虑。

因为我们的目的是使序列最终恢复单调性，所以不应该在恢复过程中进 $1$ 步破坏单调性，即必须尽可能维持现有的单调性。

由此容易发现，每 $1$ 步应该与前 $1$ 个数进行交换。但对于连续的相等数字可以压缩为 $1$ 个数，即视作与这些数字的第 $1$ 个数交换。

这就是冒泡排序的过程，所需次数就是逆序对数。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e3;
const int MAXH = 2e6;

int h[MAXN], g[MAXN], lim, t[MAXH];

void add(int p) { for (int i = p; i <= lim; i += (i & -i)) ++t[i]; }

int ask(int p) { int s = 0; for (int i = p; i; i -= (i & -i)) s += t[i]; return s; }

int main(void) {
	freopen("2532.in", "r", stdin);
	freopen("2532.out", "w", stdout);
	int N; scanf("%d", &N);
	for (int i = 0; i < N; i++) { scanf("%d", &h[i]); lim = max(lim, h[i]); }
	
	int n = 0;
	for (int i = 0; i < N; i++) if (h[i] != h[i + 1]) g[n++] = h[i];
//	for (int i = 0; i < n; i++) printf("%d\n", g[i]);
	
	int ans = 0;
	for (int i = 0; i < n; i++) { /*printf("%d %d\n", i, ask(g[i]));*/ ans += i - ask(g[i]); add(g[i]); /*for (int j = 0; j <= lim; j++) printf("%d ", t[j]); putchar('\n');*/ }
	printf("%d\n", ans);
	return 0;
}
```

#### $\text{T}2$

考虑简化版问题，不妨设坐标范围不超过 $10^5$。


$1$ 种显然的想法是：求出所有线段并集，再逐 $1$ 考虑去掉某条线段之后的损失，用总长减去最小损失。

总长容易通过差分思想 $O(n)$ 跑 $1$ 遍前缀和求出，关键在于最小损失。


不难发现，去掉 $1$ 条线段之后的损失，就是该线段所覆盖的范围内，只被 $1$ 条线段覆盖的点数。

在差分的过程中可以得到各点被覆盖的线段数，再在被 $1$ 条线段覆盖的点上打 $+1$ 标记，计算前缀和即可统计出某条线段范围内只被 $1$ 条线段覆盖的点数。


对于不超过 $10^9$ 的坐标范围，在上述算法基础上进行离散化处理即可。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 2e5;
const int INF = 1e9 + 1;

int start[MAXN], end[MAXN], f[MAXN * 3], g[MAXN * 3], len[MAXN * 3], h[MAXN * 3];
vector <int> v;

int main(void) {
	freopen("2533.in", "r", stdin);
	freopen("2533.out", "w", stdout);
	int N; scanf("%d", &N);
	for (int i = 0; i < N; i++) {
		scanf("%d%d", &start[i], &end[i]); if (start[i] > end[i]) swap(start[i], end[i]);
		--end[i]; v.push_back(start[i]); v.push_back(end[i]); v.push_back(end[i] + 1);
	}
	sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
	for (int i = 0; i < N; i++) {
		++f[lower_bound(v.begin(), v.end(), start[i]) - v.begin()];
		--f[lower_bound(v.begin(), v.end(), end[i] + 1) - v.begin()];
	}
	for (int i = 0; i < (signed)v.size(); i++) {
		g[i] = (i ? g[i - 1] : 0) + f[i];
		if (i + 1 < (signed)v.size()) { len[i] = v.at(i + 1) - v.at(i); /*printf("%d ", len[i]);*/ }
	}
	int tot = 0;
	for (int i = 0; i < (signed)v.size(); i++) {
		if (g[i]) tot += len[i];
		h[i] = (i ? h[i - 1] : 0) + (g[i] == 1 ? len[i] : 0);
	}
//	printf("%d\n", tot);
	int minlost = INF;
	for (int i = 0; i < N; i++) {
		int l = lower_bound(v.begin(), v.end(), start[i]) - v.begin();
		minlost = min(minlost, h[lower_bound(v.begin(), v.end(), end[i]) - v.begin()] - (l ? h[l - 1] : 0));
	}
	printf("%d\n", tot - minlost);
	return 0;
}
```

#### $\text{T}3$

显然，如果要卖牛，应优先卖给出价高的农场，且尽量选产奶量少的牛；如果要卖牛奶，应尽量优先卖给出价高的商铺。

问题的关键在于如何在 $2$ 种决策的权衡之间达到最优。


对于这种非此即彼的决策，往往有 $1$ 种思想，即枚举其中 $1$ 种决策，从而得到另 $1$种决策。


本题也是如此。枚举卖牛的数量，并按开头的策略进行计算即可。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define int long long

const int MAXN = 2e5;
const int INF = 1e9 + 1;

int c[MAXN], r[MAXN], fc[MAXN], fs[MAXN], fr[MAXN], fp[MAXN];
pair <int, int> sale[MAXN];

signed main(void) {
	freopen("2534.in", "r", stdin);
	freopen("2534.out", "w", stdout);
	int N, M, R; scanf("%lld%lld%lld", &N, &M, &R);
	for (int i = 0; i < N; i++) scanf("%lld", &c[i]); sort(c, c + N, greater<int>());
	fc[0] = c[0]; for (int i = 1; i < N; i++) fc[i] = fc[i - 1] + c[i];
	for (int i = 0; i < M; i++) scanf("%lld%lld", &sale[i].second, &sale[i].first); sale[M].first = 0; sale[M++].second = INF; sort(sale, sale + M, greater<pair<int, int> >());
	fs[0] = sale[0].first * sale[0].second; for (int i = 1; i < M; i++) fs[i] = fs[i - 1] + sale[i].first * sale[i].second;
	fp[0] = sale[0].second; for (int i = 1; i < M; i++) fp[i] = fp[i - 1] + sale[i].second;
	for (int i = 0; i < R; i++) scanf("%lld", &r[i]); sort(r, r + R, greater<int>());
	fr[0] = r[0]; for (int i = 1; i < R; i++) fr[i] = fr[i - 1] + r[i];
	
	int ans = 0;
	for (int i = 0; i <= min(N, R); i++) {
		int cur = i ? fr[i - 1] : 0, produce = i == N ? 0 : fc[N - i - 1];
		int p = lower_bound(fp, fp + M, produce) - fp;
		if (!p) cur += sale[0].first * produce; else cur += fs[p - 1] + sale[p].first * (produce - fp[p - 1]);
		ans = max(ans, cur);
	}
	printf("%lld\n", ans);
	return 0;
}
```

### $\text{#}2$

#### $\text{T}1$

首先有 $1$ 个性质，虽然在最终解法没有直接出现，但在思考时比较重要。即将任意路径看做 $1$ 条链，从起点开始的 $f$ 值单调不增。

考虑各边的贡献。具体地，对于询问 $k\text{ }x$，根据定义，若 $f[x][y] \ge k$，则 $x,y$ 路径上所有边均不小于 $k$。换而言之，$y$ 是由 $x$ 出发通过不小于 $k$ 的边所到达的点集。

考虑选中所有不小于 $k$ 的边，则 $x$ 所在联通块的 $\mathrm{size} - 1$（减去自身）就是答案。

但是，如果每次都重新组织并查集就会 $\texttt{TLE}$。所以应该合理安排询问和加边的顺序，对询问按 $k$ 和边按权降序，离线处理即可。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 2e5;

struct Edge { int a, b, c; bool operator < (const Edge x) const { return x.c < c; } } e[MAXN];

int p[MAXN], s[MAXN], ans[MAXN];
pair <int, pair <int, int> > q[MAXN];
inline int find(int r) { return p[r] == r ? r : p[r] = find(p[r]); }

int main(void) {
	freopen("2535.in", "r", stdin);
	freopen("2535.out", "w", stdout);
	int N, M; scanf("%d%d", &N, &M);
	for (int i = 1; i < N; i++) scanf("%d%d%d", &e[i].a, &e[i].b, &e[i].c); sort(e + 1, e + N);
	for (int i = 0; i < M; i++) { scanf("%d%d", &q[i].first, &q[i].second.first); q[i].second.second = i; } sort(q, q + M, greater<pair<int, pair<int, int> > >());
	for (int i = 1; i <= N; i++) { p[i] = i; s[i] = 1; }
	for (int i = 0, j = 1; i < M; i++) {
		for (; j < N && q[i].first <= e[j].c; j++) {
			int fa = find(e[j].a), fb = find(e[j].b);
			p[fa] = fb; s[fb] += s[fa];
		}
		ans[q[i].second.second] = s[find(q[i].second.first)];
	}
	for (int i = 0; i < M; i++) printf("%d\n", ans[i] - 1);
	return 0;
}
```
