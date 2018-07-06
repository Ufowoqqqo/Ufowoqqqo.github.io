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

### $\text{#}3$

#### $\text{T}1$

$1$ 开始的想法是容斥原理，后面发现根本推不下去。

需要从“总方案数-不合法方案”的角度思考。这 $1$ 步在考场上想到了。总方案数非常显然是 $M^N$。

根据题意，$1$ 种方案合法，当且仅当该序列中存在某段长度不小于 $K$ 的相等颜色。反之，若序列中最长的相等段长度小于 $K$，则该方案不合法。


记 $f[i]$ 为长度为 $i$ 的不存在长度为 $K$ 的相等段的序列计数。显然有边界 $\forall 0<i<K$, $f[i]=m^i$。

而 $\forall k \le i \le N$，长度为 $i$ 的序列后缀必然有相等段，不妨记其长度为 $j$。对于前 $(i-j)$ 位的方案末位颜色 $c$，当前段连续颜色不能为 $c$，否则会产生长度为 $K$ 的连续段，但可以为除了 $c$ 以外的任意颜色，而对于任意 $1$ 种方案都是如此，即

$$f[i]=(m-1)\times\sum_{j=1}^{k-1} f[i-j]$$


我的 $\text{DP}$ 还是太菜了，哎。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define int long long

const int Ghastlcon = 1e9 + 7;
const int MAXN = 1e7;

int f[MAXN], g[MAXN];

inline int p(int x, int y) {
	int r = 1;
	for (int z = x; y; y >>= 1, (z *= z) %= Ghastlcon)
		if (y & 1) (r *= z) %= Ghastlcon;
	return r;
}

signed main(void) {
	freopen("2536.in", "r", stdin);
	freopen("2536.out", "w", stdout);
	int N, M, K; scanf("%lld%lld%lld", &N, &M, &K);
	for (int i = 1; i < K; i++) { f[i] = p(M, i); g[i] = (g[i - 1] + f[i]) % Ghastlcon; }
	for (int i = K; i <= N; i++) { f[i] = ((g[i - 1] - g[i - K]) % Ghastlcon + Ghastlcon) % Ghastlcon * (M - 1) % Ghastlcon; g[i] = (g[i - 1] + f[i]) % Ghastlcon; }
	printf("%lld\n", ((p(M, N) - f[N]) % Ghastlcon + Ghastlcon) % Ghastlcon);
	return 0;
}
```

#### $\text{T}2$

本题的关键是搞清楚“警察抓小偷”的本质。

以 $K$ 为根，考虑在深度为 $d[u]$ 的叶子节点 $u$ 处放警察，则其可以保卫结点 $v$，当且仅当警察 $u$ 到达 $v$ 不晚于小偷，即 $d[u]-d[v] \le d[v]$，整理得 $d[v] \ge \frac{d[u]}{2}$，即以 $u$ 的第 $\frac{d[u]}{2}$ 层父亲为根的子树中所有结点都保障了安全。

我们希望通过最少的警察，保卫整棵树。因此根据贪心思想，在放置每个警察时，应该使得其覆盖范围尽可能大。不难想象，$d[u]$ 越小，就有越多层的结点被保卫。

由此得到选择策略，将所有叶子结点按深度升序排列，依次考虑各叶子结点，若没有警察可以保卫当前结点，则在当前结点处新增警察。警察的影响范围是某棵子树，而叶子结点是其中的单个点，利用 $\text{DFS}$ 序和 $\text{Fenwick Tree}$ 即可维护。

```cpp
#include <algorithm>
//#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 2e5;
const int MAXT = 30;

vector <int> v[MAXN], leaves;
int N, T = 17, l[MAXN], r[MAXN], cur, d[MAXN], f[MAXN][MAXT];

void dfs(int u, int p) {
//	printf("%d %d\n", u, p);
	if ((signed)v[u].size() == 1) leaves.push_back(u);
	l[u] = ++cur;
	f[u][0] = p; for (int i = 1; i <= T; i++) f[u][i] = f[f[u][i - 1]][i - 1];
	for (int i = 0; i < (signed)v[u].size(); i++)
		if (v[u].at(i) != p) { d[v[u].at(i)] = d[u] + 1; dfs(v[u].at(i), u); }
	r[u] = cur;
}

bool cmp(int x, int y) { return d[x] < d[y]; }

int t[MAXN];
void add(int p, int v) { for (int i = p; i <= N; i += (i & -i)) t[i] += v; }
int ask(int p) { int s = 0; for (int i = p; i; i -= (i & -i)) s += t[i]; return s; }

int getp(int u, int d) {
	int p = u;
	for (int i = 0; i <= T; i++) if ((1 << i) & d) p = f[p][i];
	return p;
}

int main(void) {
	freopen("2537.in", "r", stdin);
	freopen("2537.out", "w", stdout);
	int K; scanf("%d%d", &N, &K); //T = log2(N); //printf("%d %d\n", N, T);
	for (int i = 1; i < N; i++) { int a, b; scanf("%d%d", &a, &b); v[a].push_back(b); v[b].push_back(a); }
	if ((signed)v[K].size() == 1) { puts("1"); return 0; }
	dfs(K, K); sort(leaves.begin(), leaves.end(), cmp); //puts("dfs");
	int ans = 0;
	for (int i = 0; i < (signed)leaves.size(); i++)
		if (!ask(l[leaves.at(i)])) {
			++ans; int x = getp(leaves.at(i), d[leaves.at(i)] >> 1);
//			printf("%d %d %d\n", x, l[x], r[x]);
			add(l[x], 1); add(r[x] + 1, -1);
//			for (int j = 0; j <= N; j++) printf("%d ", t[j]); putchar('\n');
		}
	printf("%d\n", ans);
	return 0;
}
```

### $\text{#}4$

#### $\text{T}1$

贪心。

注意到兔子领先于乌龟的总秒数是 $1$ 定的，因此只需要合理分配给各站点停留。如果没有乌龟的威胁，当然是尽可能久地停留在权值大的站点。

考虑乌龟的追及也不复杂。仍然按权值降序考虑停留，每次都把多余时间用完（即直到乌龟赶上兔子才停止在当前站点的停留）即可。

时间复杂度为 $O(n\log n)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define int long long

const int MAXN = 1e6;
const int MAXL = 1e7;

pair <int, int> station[MAXN];
int lim, t[MAXN];

void add(int p, int v) {
	for (int i = p; i <= lim; i += (i & -i)) t[i] += v;
}
int ask(int p) {
	int s = 0;
	for (int i = p; i; i -= (i & -i)) s += t[i];
	return s;
}

signed main(void) {
	freopen("2538.in", "r", stdin);
	freopen("2538.out", "w", stdout);
	int L, N, WuGui, TuZi;
	scanf("%lld%lld%lld%lld", &L, &N, &WuGui, &TuZi);
	for (int i = 0; i < N; i++) {
		scanf("%lld%lld", &station[i].second, &station[i].first);
		lim = max(lim, station[i].second);
	}
	sort(station, station + N, greater<pair<int, int> >());
	int ans = 0;
	for (int i = 0; i < N; i++) {
		int cnt = (WuGui - TuZi) * station[i].second - ask(station[i].second);
		if (cnt > 0) {	
			add(1, cnt);
			ans += cnt * station[i].first;
		}
	}
	printf("%lld\n", ans);
	return 0;
}
```

#### $\text{T}2$

目标是最小化扔掉的鞋，等价于最大化保留的鞋。

在柱子 $i$ 进行的决策显然取决于 $D[i]$ 和当前剩余鞋的数量（从而知道栈顶的位置）。因此不妨记 $f[i]$ 为到达柱子 $i$ 时最多能保留的鞋，所求即为 $K-f[N]$。

状态转移时，由于直接扔鞋并不受 $D[i]$ 限制，可以直接扔连续的 $1$ 段，不妨设扔掉 $j$ 双鞋（$0 \le j < f[i]$），则当前穿着的就应该是第 $(K - f[i] + j + 1)$ 双鞋，为了方便表述记为 $ns$。显然 $C[ns]$ 必须不小于 $D[i]$。

$\forall 1 \le k \le S[i]$，若 $C[ns] \ge D[i + k]$，则可以考虑穿着 $ns$ 从 $i$ 跳到 $i+k$，从而用 $f[i] - j$ 尝试更新 $f[i + k]$。

时间复杂度为 $O(NKS)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 300;
const int MAXK = 300;

int D[MAXN], C[MAXK], S[MAXK], f[MAXN];

int main(void) {
	freopen("2539.in", "r", stdin);
	freopen("2539.out", "w", stdout);
	int N, K;
	scanf("%d%d", &N, &K);
	for (int i = 0; i < N; i++) scanf("%d", &D[i]);
	for (int i = 0; i < K; i++) scanf("%d%d", &C[i], &S[i]);
	f[0] = K;
	for (int i = 0; i + 1 < N; i++)
		for (int j = 0; j < f[i]; j++) { //pairs of shoes to throw away at column i
			int ns = K - f[i] + j; //the new pair of shoe
			if (C[ns] >= D[i])
				for (int k = 1; k <= S[ns] && i + k < N; k++) //steps
					if (C[ns] >= D[i + k]) f[i + k] = max(f[i + k], f[i] - j);
		}
	printf("%d\n", K - f[N - 1]);
	return 0;
}
```

### $\text{#}5$

#### $\text{T}1$

整个文件系统显然形成了树形结构，其中每个文件是叶子结点。记 $l[i]$ 为结点 $i$ 文件（夹）名长度。特别地，令叶子结点的 $l[i]$ 为实际长度减去 $1$。

考虑相对路径中文件夹之间的转移，其实就是边权，且是有向的。即若 $u$ 为 $v$ 的父亲，则 $w(u, v) = l[v] + 1, w(v, u) = 3$。

如果直接枚举出发结点，并尝试算出到达所有叶子结点的路径长度之和，会发现难以实现。



从根结点出发的答案显然是非常容易求得的。记 $f[i]$ 为根结点至 $i$ 路径上边权之和，则所求即为 $\sum_{m[i] = 0} f[i]$，其中 $m[i]$ 为 $i$ 的子结点数量。

考虑将出发点从 $u$ 转移至其子结点 $v$ 时答案产生的影响。记 $c[u]$ 为以 $u$ 为根结点的子树内叶子结点的个数。

对于以 $v$ 为根节点的子树内的叶子结点，到达它们的路径都省去了边 $(u, v)$，因此总共减去 $(l[v] + 1) * c[v]$。

对于除上述以外的其它叶子结点，到达它们的路径都增加了边 $(v, u)$，因此总共加上 $(c[1] - c[v]) * 3$。

转移过程中取所有情况下答案的最小值即为最终答案。



时间复杂度为 $O(N)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define int long long

const int MAXL = 20;
const int MAXN = 2e5;

struct Edge { int v; Edge *nxt; } e[MAXN], *cur, *h[MAXN];

int l[MAXN], m[MAXN], f[MAXN], c[MAXN], ans[MAXN], minans;

void dfs0(int u) {
	if (!m[u]) { ans[1] += f[u]; c[u] = 1; }
	for (Edge *i = h[u]; i; i = i -> nxt) {
		f[i -> v] = f[u] + l[i -> v] + 1;
		dfs0(i -> v);
		c[u] += c[i -> v];
	}
}

void dfs1(int u) {
//	printf("ans[%d] = %d\n", u, ans[u]);
	minans = min(minans, ans[u]);
	for (Edge *i = h[u]; i; i = i -> nxt)
		if (m[i -> v]) {
			ans[i -> v] = ans[u] - c[i -> v] * (l[i -> v] + 1) + (c[1] - c[i -> v]) * 3;
			dfs1(i -> v);
		}
}

signed main(void) {
	freopen("2541.in", "r", stdin);
	freopen("2541.out", "w", stdout);
	int N; scanf("%lld", &N); cur = e;
	for (int i = 1; i <= N; i++) {
		char s[MAXL]; scanf("%s%lld", s, &m[i]);
		l[i] = strlen(s); if (!m[i]) --l[i];
		for (int j = 0; j < m[i]; j++) {
			scanf("%lld", &(cur -> v));
			cur -> nxt = h[i];
			h[i] = cur++;
		}
	}
	dfs0(1); //printf("%d\n", ans[0]);
	minans = ans[1]; dfs1(1); printf("%lld\n", minans);
	return 0;
}
```

#### $\text{T}2$

对于询问 $i$，所求的实际上就是在 $N$ 个位置中选取 $i$ 个放置 $0$（根据题意，其中第 $1$ 个 $0$ 必须放在首位），每相邻 $2$ 个 $0$ 之间都是以 $1$ 为首项，公差为 $1$ 的等差数列。求由此得到的所有最终数列中，与原数列的最少不同位计数。



这个问题实际上可以动态地来看待，即考虑每 $1$ 位放什么。显然，每 $1$ 位的决策受到上 $1$ 位的影响，因此，“当前位数值”就是状态空间中的 $1$ 维。当然还有必不可少的“当前位置”。除此之外，我们还关心的是放置了多少个 $0$。因为这决定着如何确保最终方案的合法性。

基于上述考虑，不妨记 $f[i][j][k]$ 为前 $i$ 位已放置了 $j$ 个 $0$，其中第 $i$ 位的值为 $k$ 时与原数列的最少不同位计数。

若下标从 $0$ 开始，边界为 $f[0][1][0] = a[0] \neq 0$。询问 $i$ 所求为 $min\{f[N - 1][i][j]\}$，其中 $0 \le j < N$。

状态的转移可以采取填表法。已知 $f[i][j][k]$，下 $1$ 位的填写显然有 $2$ 种策略：放置 $1$ 个 $0$，或延续当前的等差数列。分别对应用 $f[i][j][k] + (a[i + 1] \neq 0)$ 尝试更新 $f[i + 1][j + 1][0]$ 和用 $f[i][j][k] + (a[i + 1] \neq k + 1)$ 更新 $f[i + 1][j][k + 1]$。



时间复杂度为 $O(N^3)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 2e2;

int f[MAXN][MAXN][MAXN], a[MAXN];

int main(void) {
	freopen("2542.in", "r", stdin);
	freopen("2542.out", "w", stdout);
	int N; scanf("%d", &N);
	for (int i = 0; i < N; i++)
		for (int j = 0; j <= N; j++)
			for (int k = 0; k < N; k++)
				f[i][j][k] = N;
	for (int i = 0; i < N; i++) scanf("%d", &a[i]);
	f[0][1][0] = (bool)a[0];
	for (int i = 0; i + 1 < N; i++)
		for (int j = 1; j <= i + 1; j++)
			for (int k = 0; k <= i; k++) {
				f[i + 1][j + 1][0] = min(f[i + 1][j + 1][0], f[i][j][k] + (bool)a[i + 1]);
				f[i + 1][j][k + 1] = min(f[i + 1][j][k + 1], f[i][j][k] + (a[i + 1] != k + 1));
			}
	for (int i = 1; i <= N; i++) {
		int ans = N;
		for (int j = 0; j < N; j++) ans = min(ans, f[N - 1][i][j]);
		printf("%d\n", ans);
	}
	return 0;
}
```
