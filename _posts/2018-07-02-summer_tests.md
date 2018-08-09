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

#### $\text{T}3$

容易证明取得最优值的 $y$ $1$ 定在所有 $e[i]$ 构成的集合中。考虑选择 $1$ 部分学生使用传送机，其中 $\sum \left\vert s[i]\right\vert $ 为定值，因此目标是最小化 $\sum \left\vert e[i]-y\right\vert$。由中位数定理可知，使总和最小的 $y$ 就是所选中的 $e[i]$ 的中位数。对于学生集合的任意子集都有该结论，因此最终的 $y$ 也必定落在某个 $e[i]$ 处。

得到了 $y$ 的备选集合之后考虑如何计算距离总和。不妨记 $f(y)$ 为传送机终点位于 $y$ 处时各学生走路距离之和，$g(i, y)$ 为传送机终点位于 $y$ 处时学生 $i$ 走路距离，显然有 $f(y)=\sum g(i,y)$。

对于每个学生 $i$，考虑 $g(i, y)$ 随 $y$ 的变化趋势。根据 $a[i]$ 与 $0$ 和 $a[i]$ 与 $b[i]$ 的大小关系分 $4$ 类讨论，不难得到如下图像。

![](http://10.3.35.134/notes/584105121a01410c2d4e6340//problemNote_2540/1.png)

可以发现 $g(i, y)$ 关于 $y$ 的图像的变化规律可以分段，并且最初都取 $\left\vert s[i]-e[i]\right\vert$。

以 $s[i]<0, s[i]<e[i]$ 为例，分界点分别为 $2s[i], e[i]$ 和 $2e[i]-2s[i]$。运用差分思想，我们在 $2s[i]$ 处打上 $-1$ 标记，$e[i]$ 处打上 $+2$ 标记，$2e[i]-2s[i]$ 处打上 $-1$ 标记。记 $delta[i]$ 为 $i$ 处的标记，$p[i]$ 为 $delta[i]$ 的前缀和，考虑相邻的分界点 $i, j$，有 $f(j)=f(i)+p[i]\times (j - i)$。

对于单个学生可以这样操作，当然也就适用于全体学生。每次将 $y$ 右移时只可能会有 $1$ 部分学生的 $g$ 值发生变动，因此在原 $f$ 值的基础上考虑变动即可。

由此，我们可以先求出所有学生都不使用传送机时的 $f$ 值，再考虑将传送机位置逐渐右移，通过类似上述的方法即可快速求得新的 $f$ 值。

坐标范围较大，需要离散化。

时间复杂度 $O(N\log N)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

#define int long long

const int MAXN = 1e6;

int a[MAXN], b[MAXN], dt[MAXN];
vector <int> v;

inline int discrete(int p) { return lower_bound(v.begin(), v.end(), p) - v.begin(); }

signed main(void) {
	freopen("2540.in", "r", stdin);
	freopen("2540.out", "w", stdout);
	int N, cur = 0; scanf("%lld", &N);
	for (int i = 0; i < N; i++) {
		scanf("%lld%lld", &a[i], &b[i]); cur += abs(a[i] - b[i]);
		if (abs(a[i]) >= abs(a[i] - b[i])) continue;
		v.push_back(b[i]);
		if (a[i] < 0 && a[i] < b[i] || a[i] >= 0 && a[i] >= b[i]) { v.push_back(0); v.push_back(b[i] << 1); }
		else { v.push_back(a[i] << 1); v.push_back(b[i] - a[i] << 1); }
	}
	sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end());
	for (int i = 0; i < N; i++)
		if (abs(a[i]) < abs(a[i] - b[i])) {
			dt[discrete(b[i])] += 2;
			if (a[i] < 0 && a[i] < b[i] || a[i] >= 0 && a[i] >= b[i]) { --dt[discrete(0)]; --dt[discrete(b[i] << 1)]; }
			else { --dt[discrete(a[i] << 1)]; --dt[discrete(b[i] - a[i] << 1)]; }
		}
	int ans = cur, dt_pf = 0;
	for (int i = 0; i < (signed)v.size(); i++) {
		if (i) { cur += dt_pf * (v.at(i) - v.at(i - 1)); ans = min(ans, cur); }
		dt_pf += dt[i];
	}
	printf("%lld\n", ans);
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

### $\text{#}6$

#### $\text{T}1$

对于质量为 $C[i]$、最大跳跃距离为 $S[i]$ 的鞋 $i$，考虑穿着它能否从 $1$ 跳到 $N$。

首先标出所有 $D[j] \le C[i]$ 的柱子 $j$，这些柱子就是穿着 $i$ 时所允许的落脚点。为了能够跳完全程，当落脚于某个柱子 $j$ 时，下 $1$ 步的范围是 $[j + 1, j + S[i]]$。只有当这个范围内存在被标记的柱子时才能继续，否则无解。

这样，问题可以转化为：求所有相邻的被标记柱子之间的最大距离 $d$。解存在，当且仅当 $d \le S[i]$。

考虑如何实现这个最大距离的维护。

首先可以发现，柱子的标记显然是存在顺序的，被标记的柱子数随着 $C[i]$ 的增大而单调不减。也就意味着我们可以把询问离线，按 $C[i]$ 升序考虑，这样每次只需新增 $C[i - 1] < D[j] \le C[i]$ 的柱子即可。每当新增 $1$ 个柱子 $m$，记其左边最近的柱子为 $l$，右边最近的柱子为 $r$，则需要删去距离 $r - l$，加入距离 $m - l$ 和 $r - m$。寻找两侧最近柱子可以通过 $\text{Fenwick Tree + Binary Search}$ 实现。

要从 $1$ 个集合中删去、加入元素并求最值，可以用 $\text{Heap}$ 实现。只需将距离与 $\text{Heap}$ 下标作映射即可。

时间复杂度为 $O(N\log ^2N)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e6;
const int INF = 1e9 + 1;

pair <int, int> D[MAXN];
pair <int, pair<int, int> > shoes[MAXN];

int N, t[MAXN];
void add(int p) { for (int i = p; i <= N; i += (i & -i)) ++t[i]; }
int ask(int p) { if (p > N) p = N; int s = 0; for (int i = p; 0 < i; i -= (i & -i)) s += t[i]; return s; }

int rv[MAXN], hpid[MAXN], ans[MAXN];

struct Heap {
	int sz, v[MAXN];
	Heap () { sz = 0; memset(v, 0, sizeof v); }
	int up(int k) {
		for (int i = k; i > 1; i >>= 1) {
			int p = i >> 1;
			if (rv[v[i]] > rv[v[p]]) { swap(v[i], v[p]); swap(hpid[v[i]], hpid[v[p]]); } else break;
		}
	}
	
	int down(int k) {
		for (int i = k; (i << 1) <= sz; ) {
			int c = i << 1; if ((c | 1) <= sz && rv[v[c | 1]] > rv[v[c]]) c |= 1;
			if (rv[v[c]] > rv[v[i]]) { swap(v[i], v[c]); swap(hpid[v[i]], hpid[v[c]]); } else break;
			i = c;
		}
	}
	
	void push(int k) { v[++sz] = k; up(hpid[k] = sz); }
	void pop(int k) { v[k] = v[sz--]; hpid[v[k]] = k; up(k); down(k); }
	int top() { return sz ? rv[v[1]] : INF; }
} hp;

int main(void) {
	freopen("2543.in", "r", stdin);
	freopen("2543.out", "w", stdout);
	int K; scanf("%d%d", &N, &K);
	for (int i = 1; i <= N; i++) { scanf("%d", &D[i].first); D[i].second = i; } sort(D + 1, D + N + 1);
	for (int i = 1; i <= K; i++) { scanf("%d%d", &shoes[i].first, &shoes[i].second.first); shoes[i].second.second = i; } sort(shoes + 1, shoes + K + 1);
	for (int i = 1, j = 1; i <= K; i++) {
//		printf("%d %d %d\n", shoes[i].first, shoes[i].second.first, shoes[i].second.second);
		for (; j <= N && D[j].first <= shoes[i].first; j++) {
//			printf("%d %d\n", D[j].first, D[j].second);
			int l, r, lid = 0, rid = 0;
			
			for (l = 0, r = D[j].second - 1; l + 1 < r; ) {
				int m = l + r >> 1;
				if (ask(D[j].second - 1) - ask(D[j].second - m - 1)) r = m; else l = m;
			}
			if (ask(D[j].second - 1) - ask(D[j].second - r - 1)) lid = D[j].second - r;
			
			for (l = 0, r = N - D[j].second; l + 1 < r; ) {
				int m = l + r >> 1;
				if (ask(D[j].second + m) - ask(D[j].second)) r = m; else l = m;
			}
			if (ask(D[j].second + r) - ask(D[j].second)) rid = D[j].second + r;
			
//			printf("%d %d\n", lid, rid);
			
			if (lid && rid) hp.pop(hpid[lid]);
			if (lid) { rv[lid] = D[j].second - lid; hp.push(lid); }
			if (rid) { rv[D[j].second] = rid - D[j].second; hp.push(D[j].second); }
			
			add(D[j].second);
		}
//		printf("%d\n", hp.top());
		ans[shoes[i].second.second] = hp.top() <= shoes[i].second.first;
//		for (int j = 1; j <= K; j++) printf("%d ", ans[j]); putchar('\n');
	}
	for (int i = 1; i <= K; i++) printf("%d\n", ans[i]);
	return 0;
}
```


#### $\text{T}2$

对于每个学生 $i$，不使用传送机的花费为 $\|s[i]-e[i]\|$，使用传送机的最小花费为 $\min\\{\|s[i]-x[j]\|+\|e[i]-y[j]\|+z[j]\\}$。

绝对值符号无法直接处理，考虑分类讨论。

按照 $s[i]$ 与 $x[j]$ 和 $e[i]$ 与 $y[j]$ 之间的大小关系，显然可以分为 $4$ 类。

以 $s[i] \ge x[j], e[i] \ge y[j]$ 为例，花费为 $s[i] - x[j] + e[i] - y[j] + z[j]$，即 $(s[i] + e[i]) + (z[j] - x[j] - y[j])$，其中前 $1$ 个括号对于特定的学生是定值，因此目标就是在 $x[j] \le s[i], y[j] \le e[i]$ 的范围内最小化 $z[j] - x[j] - y[j]$（如果存在）。对应到 $2$ 维的平面直角坐标系上，可以看作求以 $(s[i], e[i])$ 为右上角的矩形中的最小值。

这是经典问题，可以将询问离线，用扫描线配合 $\text{Fenwick Tree}$ 维护前缀最值实现。

其余 $3$ 种情况类似。注意坐标范围较大，需要将 $y$ 值离散化，且编号从 1 开始（否则无法使用 $\text{Fenwick Tree}$ 维护）。

时间复杂度为 $O((N+M)\log (N+M))$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

#define int long long

const int MAXN = 1e6;
const int MAXM = 1e6;
const int INF = 0x7fffffff;

struct SLINGSHOT {
	int x, y, t;
} slingshot[MAXN];
bool cmps_l(SLINGSHOT i, SLINGSHOT j) {
	return i.x < j.x;
}
bool cmps_g(SLINGSHOT i, SLINGSHOT j) {
	return j.x < i.x;
}

struct MANURE {
	int a, b, c;
} manure[MAXM];
int M, ans[MAXM];
bool cmpm_l(MANURE i, MANURE j) {
	return i.a < j.a;
}
bool cmpm_g(MANURE i, MANURE j) {
	return j.a < i.a;
}

int K, t[MAXM];
int add(int p, int v) {
	for (int i = p; i <= K; i += (i & -i)) t[i] = min(t[i], v);
}
int ask(int p) {
	int r = INF;
	for (int i = p; i; i -= (i & -i)) r = min(r, t[i]);
	return r;
}

vector <int> v;
inline int discrete(int p) {
	return lower_bound(v.begin(), v.end(), p) - v.begin() + 1;
}

void debug_output() {
	for (int i = 0; i < M; i++) printf("%lld ", ans[i]);
	putchar('\n');
}

signed main(void) {
	freopen("2544.in", "r", stdin);
	freopen("2544.out", "w", stdout);
	int N;
	scanf("%lld%lld", &N, &M);
	for (int i = 0; i < N; i++) {
		scanf("%lld%lld%lld", &slingshot[i].x, &slingshot[i].y, &slingshot[i].t);
		v.push_back(slingshot[i].y);
	}
	for (int i = 0; i < M; i++) {
		scanf("%lld%lld", &manure[i].a, &manure[i].b);
		manure[i].c = i;
		ans[i] = abs(manure[i].a - manure[i].b);
		v.push_back(manure[i].b);
	}
	sort(v.begin(), v.end());
	v.erase(unique(v.begin(), v.end()), v.end());
	K = (signed)v.size();

	memset(t, 0x7f, sizeof t);
	sort(slingshot, slingshot + N, cmps_l);
	sort(manure, manure + M, cmpm_l);
	for (int i = 0, j = 0; i < M; i++) {
		for (; j < N && slingshot[j].x <= manure[i].a; j++) add(discrete(slingshot[j].y), slingshot[j].t - slingshot[j].x - slingshot[j].y);
		ans[manure[i].c] = min(ans[manure[i].c], manure[i].a + manure[i].b + ask(discrete(manure[i].b)));
	}

	memset(t, 0x7f, sizeof t);
	for (int i = 0, j = 0; i < M; i++) {
		for (; j < N && slingshot[j].x <= manure[i].a; j++) add(K - discrete(slingshot[j].y) + 1, slingshot[j].y - slingshot[j].x + slingshot[j].t);
		ans[manure[i].c] = min(ans[manure[i].c], manure[i].a - manure[i].b + ask(K - discrete(manure[i].b) + 1));
	}
//	debug_output();

	memset(t, 0x7f, sizeof t);
	sort(slingshot, slingshot + N, cmps_g);
	sort(manure, manure + M, cmpm_g);
	for (int i = 0, j = 0; i < M; i++) {
		for (; j < N && manure[i].a <= slingshot[j].x; j++) add(discrete(slingshot[j].y), slingshot[j].x - slingshot[j].y + slingshot[j].t);
		ans[manure[i].c] = min(ans[manure[i].c], manure[i].b - manure[i].a + ask(discrete(manure[i].b)));
	}

	memset(t, 0x7f, sizeof t);
	for (int i = 0, j = 0; i < M; i++) {
		for (; j < N && manure[i].a <= slingshot[j].x; j++) add(K - discrete(slingshot[j].y) + 1, slingshot[j].x + slingshot[j].y + slingshot[j].t);
		ans[manure[i].c] = min(ans[manure[i].c], ask(K - discrete(manure[i].b) + 1) - manure[i].a - manure[i].b);
	}

	for (int i = 0; i < M; i++) printf("%lld\n", ans[i]);
	return 0;
}
```

### $\text{#}7$

#### $\text{T}1$

询问的限制是“当前已出现的节点编号”。考虑最终形态的森林（或通过添加虚根使其成为树以方便处理亦可），关于节点 $x$ 在时刻（数值上等于当时已出现的节点编号） $c$ 的询问，所求即为从 $x$ 出发，经过编号不大于 $c$ 的节点能够走的最长路。

编号限制的解决方案，最直接的当然就是遍历。但是直接暴力显然是不可行的。

考虑点分治，也许可以在时间复杂度得到保证的情况下暴力。

前提是如下引理：对于任意路径，路径上编号最大的节点必在路径的端点。

证明：反证法。假设路径上编号最大的节点不在端点，而在路径当中，则该点必然有子节点，且该子节点也在路径上。在原树中各节点的子节点编号均大于自身，因此与定义矛盾。假设不成立，原命题得证。

对于分治得到当前树的根节点 $r$，考虑其会对哪些询问造成影响，也就是哪些询问的路径可能经过 $r$。令 $r$ 的深度 $dep[r] = 0$。

 1. 关于 $r$ 的询问，即从 $r$ 出发到子树中节点。询问可能有多个，考虑其中某个在时刻 $c[i]$ 的询问，根据上述引理，只要保证子树中某节点 $x$ 满足 $x \le c[i]$，即可从 $r$ 到达 $x$。所求即为子树中编号不超过 $c[i]$ 的节点的最大深度。

 2. 关于子树中节点 $x$ 的询问，考虑其中某个在时刻 $c[j]$ 的询问

   2.1. 从 $x$ 出发到 $r$。类似地，若 $r \le c[j]$，即可从 $x$ 到达 $r$，尝试用 $dep[x]$ 更新该询问的答案；

   2.2. 从 $x$ 出发到 $r$，再到达另 $1$ 子树中的节点 $y$。同理，只需保证 $y \le c[j]$，即可用 $dep[x] + dep[y]$ 更新答案。换言之，需要求得以 $r$ 为根的所有子树（除 $x$ 所在子树外）中编号不大于 $c[j]$ 的节点的最大深度。

  事实上上述 $2$ 种子情况可以合并起来统 $1$ 处理。

求带修前缀最值，可以用线段树实现。先在每次点分治的暴力时将所有深度扔进线段树，排除当前子树只需在更新答案之前将当前子树中所有点的值拿出来，求完答案再扔回去即可。

时间复杂度 $O(N\log ^2N)$。实现上如果大量使用 `STL`（如多处可能用到 `vector`），会导致常数过大而 `TLE`；需要用手写的方式模拟实现即可。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#pragma GCC optimize("O3")
using namespace std;

#define lc ch[0]
#define rc ch[1]

const int MAXQ = 1e6;
const int INF = 1e9 + 1;

struct Vector { int x, y; Vector *nxt; };

Vector e[MAXQ], *cur_e, *h_e[MAXQ];
Vector q[MAXQ], *cur_q, *h_q[MAXQ];

int ans[MAXQ];
bool done[MAXQ];

int root, f[MAXQ], sz[MAXQ];

int v[MAXQ], cv;

void dfs(int u, int p) {
	f[u] = 0; sz[u] = 1; v[cv++] = u;
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (i -> x != p && !done[i -> x]) { dfs(i -> x, u); sz[u] += sz[i -> x]; f[u] = max(f[u], sz[i -> x]); }
}

void getroot(int u) {
	cv = 0; dfs(u, u); root = u;
	for (int i = 0; i < cv; i++)
		if (max(f[v[i]], sz[u] - sz[v[i]]) < max(f[root], sz[u] - sz[root])) root = v[i];
}

int dep[MAXQ];
Vector subtree[MAXQ], *cur_s, *h_s[MAXQ];

struct SegmentTree {
//	struct Node {
//		Node *ch[2];
//		int v;
//		int l, r, m;
//		Node (int x, int y): v(-INF), l(x), r(y), m(x + y >> 1) { lc = rc = NULL; }
//		void maintain() { v = max(lc -> v, rc -> v); }
//	} *root;
//	void build(Node *&u, int l, int r) { u = new Node(l, r); if (l < r) { build(u -> lc, l, u -> m); build(u -> rc, u -> m + 1, r); } }
//	void update(Node *&u, int p, int v) { if (p < u -> l || u -> r < p) return; if (u -> l == u -> r) { u -> v = v; return; } update(u -> ch[u -> m < p], p, v); u -> maintain(); }
//	int query(Node *u, int l, int r) { if (r < u -> l || u -> r < l) return -INF; if (l <= u -> l && u -> r <= r) return u -> v; return max(query(u -> lc, l, r), query(u -> rc, l, r)); }
	
	int n;
	int f[MAXQ << 2];
	void build(int l, int r)
	{
		n = 1e6 + 20;
		for(int i = 0; i < n * 2; i ++)
			f[i] = -INF;
		
		return;
	}
	
	void update(int p, int x)
	{
	    for(f[p += n] = x; p >>= 1; )
	        f[p] = max(f[p << 1], f[p << 1 | 1]);
	
	    return;
	}
	
	int query(int s, int t)
	{
	    int o;
	
	    for(o = -INF, s += n, t += n + 1; s ^ t; s >>= 1, t >>= 1)
	    {
	        if(s & 1)
	            o = max(o, f[s ++]);
	        if(t & 1)
	            o = max(o, f[-- t]);
	    }
	
	    return o;
	}
} st;

void bruteforce(int u, int p, int belong) {
	dep[u] = dep[p] + 1; //printf("dep[%d] = %d\n", u, dep[u]);
	st.update(u, dep[u]);
	cur_s -> x = u; cur_s -> nxt = h_s[belong]; h_s[belong] = cur_s++;
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (i -> x != p && !done[i -> x]) bruteforce(i -> x, u, belong);
}

void solve(int u) {
//	printf("solve(%d)\n", u);
	done[u] = true; dep[u] = 0;
	cur_s = subtree; //memset(h_s, 0, sizeof h_s);
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (!done[i -> x]) bruteforce(i -> x, u, i -> x);
	for (Vector *i = h_q[u]; i; i = i -> nxt) {
		int res = st.query(1, i -> x);
//		printf("%d %d\n", i -> x, res);
		ans[i -> y] = max(ans[i -> y], res);
	}
	st.update(u, 0);
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (!done[i -> x]) {
			for (Vector *j = h_s[i -> x]; j; j = j -> nxt) st.update(j -> x, -INF);
			for (Vector *j = h_s[i -> x]; j; j = j -> nxt)
				for (Vector *k = h_q[j -> x]; k; k = k -> nxt)
					if (k -> x >= u)
						ans[k -> y] = max(ans[k -> y], dep[j -> x] + st.query(1, k -> x));
			for (Vector *j = h_s[i -> x]; j; j = j -> nxt) st.update(j -> x, dep[j -> x]);
		}
	st.update(u, -INF);
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (!done[i -> x]) {
			for (Vector *j = h_s[i -> x]; j; j = j -> nxt) st.update(j -> x, -INF);
			h_s[i -> x] = 0;
		}
	for (Vector *i = h_e[u]; i; i = i -> nxt)
		if (!done[i -> x]) { getroot(i -> x); solve(root); }
}

int main(void) {
	freopen("2545.in", "r", stdin);
	freopen("2545.out", "w", stdout);
	int Q, id = 0; scanf("%d", &Q);
	cur_e = e; //memset(h_e, 0, sizeof h_e);
	cur_q = q; //memset(h_q, 0, sizeof h_q);
	memset(ans, -1, sizeof ans);
	for (int i = 0; i < Q; i++) {
		char request[2]; int p; scanf("%s%d", request, &p);
		if (!strcmp(request, "B")) {
			++id;
			if (p == -1) {
				cur_e -> x = Q + 1; cur_e -> nxt = h_e[id]; h_e[id] = cur_e++;
				cur_e -> x = id; cur_e -> nxt = h_e[Q + 1]; h_e[Q + 1] = cur_e++;
			} else {
				cur_e -> x = p; cur_e -> nxt = h_e[id]; h_e[id] = cur_e++;
				cur_e -> x = id; cur_e -> nxt = h_e[p]; h_e[p] = cur_e++;
			}
		}
		if (!strcmp(request, "Q")) {
			cur_q -> x = id; cur_q -> y = i; cur_q -> nxt = h_q[p]; h_q[p] = cur_q++;
			ans[i] = 0;
		}
	}
	getroot(Q + 1); //printf("%d\n", root);
	st.build(1, id); solve(root);
	for (int i = 0; i < Q; i++) if (ans[i] != -1) printf("%d\n", ans[i]);
	return 0;
}
```

#### $\text{#}2$

预处理：为了方便计算将左开右闭区间转为闭区间。

动态规划。

首先显然应该去除完全被包含的区间，除去冗余状态，去掉之后不可能使结果变得更坏。

状态的表述，如果直接记 $f[i][j]$ 为前 $i$ 个区间中去除 $j$ 个区间，会发现重复部分难以计算，需要适当增加限制。



记 $f[i][j]$ 为前 $i$ 个区间中去除 $j$ 个区间，其中保留第 $i$ 个区间的最大覆盖长度。求得之后枚举最后 $1$ 个保留的区间即可得到答案。

考虑状态转移，对上 $1$ 个保留的区间 $k$ 进行分类讨论，即区间 $(k, i)$ 都舍弃：

- 在第 $i$ 个区间之前不保留任何区间，显然有 $f[i][j] = r[i] - l[i] + 1$

- 上 $1$ 个保留的区间 $k$ 不与区间 $i$ 相交，则有 $f[i][j] = f[k][j - i + k + 1] + r[i] - l[i] +1$

- 上 $1$ 个保留的区间 $k$ 与区间 $i$ 相交，则有 $f[i][j] = f[k][j - i +k + 1] + r[i] - r[k]$

注意到枚举 $k$ 时应该保证第 $2$ 维 $j - i + k + 1$ 非负，因此不需要从 $0$ 至 $i - 1$ 枚举。



时间复杂度 $O(NK^2)$。

经过简单的常数优化即可通过。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <utility>

#pragma GCC optimize ("O3")

using namespace std;

#define MAX(a,b) ((a)>(b)?(a):(b))

typedef pair <int, int> pii;

const int MAXN = 2e5;
const int MAXK = 2e2;

struct Interval {
	int l, r;
	bool operator < (const Interval &x) const { return l != x.l ? l < x.l : x.r < r; }
} lifeguard[MAXN], useful[MAXN];

//pii lifeguard[MAXN], useful[MAXN];
//bool cmp(pii x, pii y) { return x.l != y.l ? x.l < y.l : y.r < x.r; }

int len[MAXN], f[MAXN][MAXK];

int main(void) {
	freopen("2546.in", "r", stdin);
	freopen("2546.out", "w", stdout);
	int N, K; scanf("%d%d", &N, &K);
	for (int i = 0; i < N; i++) { scanf("%d%d", &lifeguard[i].l, &lifeguard[i].r); --lifeguard[i].r; }
	sort(lifeguard, lifeguard + N);
	int rightmost = 0, M = 0;
	for (int i = 0; i < N; i++)
		if (rightmost < lifeguard[i].r) {
			len[M] = lifeguard[i].r - lifeguard[i].l + 1; useful[M++] = lifeguard[i]; rightmost = lifeguard[i].r;
		} else --K;
	K = MAX(K, 0); if (M <= K) { puts("0"); return 0; }
	for (int i = 0; i < M; i++) {
		for (int j = 0; j <= min(i, K); j++) {
			f[i][j] = len[i];
			for (int k = MAX(i - j - 1, 0), t = j - i + k + 1; k < i; k++, t++)
				if (useful[k].r < useful[i].l) f[i][j] = MAX(f[i][j], f[k][t] + len[i]);
				else f[i][j] = MAX(f[i][j], f[k][t] + useful[i].r - useful[k].r);
//			printf("%d ", f[i][j]);
		}
//		putchar('\n');
	}
	int ans = 0; for (int i = M - K - 1; i < M; i++) ans = MAX(ans, f[i][K - M + i + 1]); printf("%d\n", ans);
	return 0;
}
```

### $\text{#}8$

#### $\text{T}1$

考察找规律。

观察样例，猜想答案均为 $2$ 的非负整数次幂。通过实际编程求解进 $1$ 步肯定了猜想。

$n$ 的范围很大，因此复杂度为 $\log$ 级别或常数级别。

通过小范围内列出 $n$ 的 $2$ 进制表示与对应的答案，发现所求即为 $2^k$，其中 $k$ 为 $n$ 的 $2$ 进制表示中 $1$ 的个数。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

int main(void) {
	freopen("2547.in", "r", stdin);
	freopen("2547.out", "w", stdout);
	long long n;
	scanf("%lld", &n);
	int cnt = 0;
	for (; n; n = n & (n - 1)) ++cnt;
	printf("%lld\n", 1LL << cnt);
	return 0;
}
```

#### $\text{T}2$

关键是发掘最终形态的回文序列与原序列之间的关系。

考虑原序列中最左右 $2$ 端的数。若它们相等，则无需与旁边的数进行合并，考虑由原序列中第 $2$ 个数至倒数第 $2$ 个数组成的子问题即可。

否则，$2$ 个数最终都会各自被合并到 $1$ 个新的数中，且 $2$ 个新数相等。由于被合并的数 $1$ 定相邻，我们只需从 $2$ 端不断向中间合并，每次将总和小的 $1$ 端推进，直至 $2$ 边得到的新数相等（求解子问题）或两端相遇（结束）。

这是典型的 $\text{two-pointers}$ 做法，时间复杂度为 $O(n)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 2e6;

int a[MAXN];

int main(void) {
	freopen("2548.in", "r", stdin);
	freopen("2548.out", "w", stdout);
	int n, ans = 0;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) scanf("%d", &a[i]);
	for (int i = 1, j = n, si = a[1], sj = a[n]; i < j; )
		if (si == sj) {
			si = a[++i];
			sj = a[--j];
		} else if (si < sj) {
			si += a[++i];
			++ans;
		} else {
			sj += a[--j];
			++ans;
		}
	printf("%d\n", ans);
	return 0;
}
```

#### $\text{T}3$

$40\%$：可以任意安排顺序。典型的田忌赛马问题。应用贪心策略，按照难度递增顺序考虑每个问题，在剩下的所有能解决问题中选能力最弱的。若没有人能解决当前问题，分配剩下的人当中能力最弱的。



$100\%$：先让所有人呆在各自的理想位置。假设我们从某个位置开始进行分配。

考虑当前“能够自由支配”的人有哪些。显然，从出发开始到当前位置 $u$，设走了 $k$ 步，我们把 $1$ 路上经过的人都先收集起来，共有 $p$ 个。显然当且仅当 $k \le p$ 时，我们能够任意分配已有的人。这里“任意分配”的含义是指，对于其中的某个人 $i$，可以让他最终位于 $C[i]$ 至 $u$ 当中的任意位置，只需让之前的人占领中间的空位即可。

我们试着找出某个位置破环成链，使得全程都存在可以任意支配的人，即 $k \le p$ 始终成立。

这样的位置 $1$ 定存在。

记 $w[i]$ 为 $C[j]=i$ 的 $j$ 数量， $sum[i]$ 为 $w[]$ 的前缀和，$q[i] = sum[i]-i$。

这时，$q$ 的意义就是：从 $1$ 走到 $i$，还缺少空位的人数。$q$ 为正数时还有多余的人没有放置，$q$ 为负数时人不够，之前的位置还有空余。

显然有 $q[n]=0$。所以我们如果把环拓宽 $2$ 倍，第 $2$ 遍的 $q$ 和第 $1$ 遍完全一样。



对于所有 $1 \le x < y \le 2n$，$q[y] - q[x-1]$ 就是从 $x$ 走到 $y$ 还缺少空位的人数。

那么我们选出所有 $q$ 中间所有的 $q[m]$，由于它是最小的，所以所有 $q[m]-q[x] \le 0$，也就是说，无论从哪里走到 $m$，都不会有人缺少空位，很定不会有人需要走过 $m$ 到 $m+1$ 的路。

也就是说，我们可以把环从 $m$ 和 $m+1$ 之间断开，而不会产生影响。



对于得到的 $1$ 条链的问题，动态地进行田忌赛马解决即可。其中关键部分是怎么找“当前自由人当中刚好能解决谜题的最弱的人”。可以用线段树或者平衡树实现。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <set>

using namespace std;

const int MAXN = 1e6;

int C[MAXN], A[MAXN], B[MAXN], cnt[MAXN];
vector <int> v[MAXN];
set <int> s;

int main(void) {
	freopen("2549.in", "r", stdin);
	freopen("2549.out", "w", stdout);
	int n; scanf("%d", &n);
	for (int i = 1; i <= n; i++) { scanf("%d", &C[i]); ++cnt[C[i]]; }
	for (int i = 1; i <= n; i++) scanf("%d", &A[i]);
	for (int i = 1; i <= n; i++) { scanf("%d", &B[i]); v[C[i]].push_back(B[i]); }
	int m = 0, q = 0, least = 0;
	for (int i = 1; i <= n; i++) {
		q += cnt[i] - 1;
		if (q < least) { least = q; m = i; } 
	}
	++m; if (n < m) m = 1; int ans = 0;
//	printf("%d %d\n", n, m);
	for (int i = m; ; ) {
//		printf("i = %d\n", i);
		for (int j = 0; j < (signed)v[i].size(); j++) {
//			printf("j = %d\n", j);
			s.insert(v[i].at(j));
		}
		set<int>::iterator it = s.lower_bound(A[i]);
		if (it == s.end()) s.erase(s.begin());
		else { ++ans; s.erase(it); }
		++i; if (n < i) i = 1; if (i == m) break;
	}
	printf("%d\n", ans);
	return 0;
}
```

### $\text{#}9$

#### $\text{T}1$

按照题意，直接模拟判断即可。

如果为了处理句子最后 $1$ 个单词时，末尾只允许标点符号和小写字母的出现，需要注意句中人名长度为 $1$ 的边界情况。

```cpp
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXL = 1e4;

char s[MAXL];

bool check(char last) {
	if (!isupper(s[0])) return false;
//	printf("%c %c\n", last, s[1]);
	if (s[1] != '\0')
		for (int i = 1; s[i + 1] != '\0'; i++)
			if (!islower(s[i])) return false;
	return s[1] == '\0' || islower(last) || last == '!' || last == '?' || last == '.';
}

int main(void) {
	freopen("name.in", "r", stdin);
	freopen("name.out", "w", stdout);
	int N;
	scanf("%d", &N);
	for (int cnt = 0; ~scanf("%s", s); ) {
		char last = s[strlen(s) - 1];
//		putchar(last);
		if (check(last)) ++cnt;
//		printf("%d\n", cnt);
		if (last == '!' || last == '?' || last == '.') {
			printf("%d\n", cnt);
			cnt = 0;
		}
	}
	return 0;
}
```

#### $\text{T}2$

写出图的邻接矩阵。由于无向图的对称性，可以只看上 $3$ 角矩阵部分。目标矩阵是使所有值为 $1$。

显然，假如对某个点多次操作，每 $2$ 次就会抵消回原有的状态。因此每个点最多被操作 $1$ 次。且点与点之间的操作顺序是没有关系的，因此只需考虑操作哪些点即可。

考虑对某个点 $i$ 进行操作而产生的影响。体现在邻接矩阵上，就是第 $i$ 行和第 $i$ 列所有值取反。

记第 $i$ 个点是否操作为 $x[i]$，值为 $0$ 或 $1$。对于原始邻接矩阵中的 $a[i][j]$，若其值为 $0$，即原来 $i$ 与 $j$ 之间没有边相连，那么要么操作点 $i$，要么操作点 $j$，即 $x[i]\ \mathrm{xor}\ x[j]=1$，类似地，$a[i][j] = 1$，则有 $x[i]\ \mathrm{xor}\ x[j]=0$。

一般地，我们得到了 $n^2$ 条形如 $x[i]\ \mathrm{xor}\ x[j]=a[i][j]\ \mathrm{xor}\ 1$ 的异或方程。



自然地，第 $1$ 反应是解异或方程组，然而时间复杂度并不允许这样做。

可以发现每条方程都有且仅有 $2$ 个未知数的系数为 $1$。所以我们只需维护未知数之间的异或关系，并检查是否有矛盾。可以使用维护传递关系的并查集实现。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 2e3;

bool g[MAXN][MAXN];
int f[MAXN << 1];

inline int find(int r) {
	return f[r] == r ? r : f[r] = find(f[r]);
}

bool ok(int n) {
	for (int i = 1; i <= n; i++) f[i] = i;
	for (int i = 1; i < n; i++)
		for (int j = i + 1; j <= n; j++)
			if (g[i][j]) f[find(i)] = find(j);
	for (int i = 1; i < n; i++)
		for (int j = i + 1; j <= n; j++)
			if (!g[i][j] && find(i) == find(j)) return false;
	return true;
}

int main(void) {
	freopen("2551.in", "r", stdin);
	freopen("2551.out", "w", stdout);
	int n;
	scanf("%d", &n);
	int m;
	scanf("%d", &m);
	for (int i = 0; i < m; i++) {
		int u, v;
		scanf("%d%d", &u, &v);
		g[min(u, v)][max(u, v)] = true;
	}
	puts(ok(n) ? "DA" : "NE");
	return 0;
}
```

#### $\text{T}3$

对于 $x, y$，考虑 $\min\{x \% y, y \% x\}$ 的另 $1$ 种表达。

如果 $x < y$，则 $x \% y = x$，而 $y \% x < x$，因此最终取 $y \% x$；

如果 $y < x$，则 $y \% x = y$，而 $x \% y < y$，因此最终去 $x \% y$。

于是可以发现，$\min\{x \% y, y \% x\} = \max\{x, y} \% \min\{x, y\}$。



为了方便，规定 $x < y$，$1$ 律从前面向后面连边。

由于在完全图上求生成树，边的数量太多，需要去除冗余的边。

根据定义，$y \% x = y - kx$。

假如有另外 $1$ 个 $z$，使得 $kx \le y < z < (k+1) x$，那么显然 $x$ 连 $z$ 不如 $x$ 连 $y$，而且 $x$ 连 $z$ 也不如 $y$ 连 $z$，换句话说，$x$ 连 $z$ 这条边不可能出现在最小生成树上。

因此对于每组 $(x, k)$，我们只需保留使 $y - kx$ 最小（且非负）的 $y$ 即可。



这样产生边的数量为 $\frac{n}{1}+\frac{n}{2}+\frac{n}{3}+\ldots +\frac{n}{n}\approx n \lg⁡ n$。
对这些边使用 $\text{Kruskal}$ 算法求解即可。

实现上的细节见代码注释。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

#define MIN(x,y) ((x)<(y)?(x):(y))

const int MAXN = 1e5 + 10;
const int MAXA = 1e7 + 10;

int A[MAXN], nxt[MAXA], f[MAXN];

inline int find(int r) {
	return f[r] == r ? r : f[r] = find(f[r]);
}

vector < pair<int, int> > e[MAXA];

int main(void) {
	freopen("constellation.in", "r", stdin);
	freopen("constellation.out", "w", stdout);
	int n;
	scanf("%d", &n);
	for (int i = 1; i <= n; i++) {
		f[i] = i;
		scanf("%d", &A[i]);
	}
	sort(A + 1, A + n + 1);
	n = unique(A + 1, A + n + 1) - A - 1;
	for (int i = 1, j = 1; j <= n; i++) {
		nxt[i] = j; //找比 i 大的最近的 A[j]
		if (i == A[j]) ++j;
	}
	for (int i = 1; i < n; i++)
		if (A[i + 1] < (A[i] << 1)) e[A[i + 1] - A[i]].push_back(make_pair(i, i + 1)); //如果Ay<2Ax，由于最接近Ax的一倍的数就是Ax自己，这条边会漏掉。所以我们要加上每一条Ax+1<2Ax的边
	for (int i = 1; i <= n; i++)
		for (int j = A[i] << 1; j <= A[n]; j += A[i]) //j <= A[n], not n
			if (nxt[j] != nxt[j + A[i]]) e[A[nxt[j]] - j].push_back(make_pair(i, nxt[j])); //只有 A[i] 的下 1 个倍数不对应同一个点时，我们才记录这条边
	int ans = 0;
	for (int i = 0, j = 1; i <= A[n] && j < n; i++) //i should start from 0!!
	//由于边的数量接近 10^8，我们不能用快排。注意到边权最多 10^7，我们可以用桶排。
		for (int k = 0; k < (signed)e[i].size(); k++) {
			int fu = find(e[i].at(k).first), fv = find(e[i].at(k).second);
			if (fu != fv) {
				f[fu] = fv;
				ans += i;
				++j;
			}
		}
	printf("%d\n", ans);
	return 0;
}
```

### $\text{#}10$

#### $\text{T}1$

按照题意直接模拟即可，考察代码实现能力。

注意到字符串的长度和数量都很小，可以直接暴力匹配，不需要使用高级数据结构，以免增加不必要的潜在错误。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 100;
const int MAXL = 200;

char station[MAXN][MAXL], cur[MAXL];
bool v[MAXN];

int main(void) {
	freopen("2553.in", "r", stdin);
	freopen("2553.out", "w", stdout);
	int n;
	scanf("%d", &n);
	for (int i = 0; i < n; i++) scanf("%s", station[i]);
	scanf("%s", cur);
	int m = strlen(cur);
	for (int i = 0; i < n; i++) {
		bool matched = true;
		for (int j = 0; j < m; j++)
			if (station[i][j] != cur[j]) {
				matched = false;
				break;
			}
		if (matched) v[station[i][m]] = true;
	}
	printf("***");
	for (char i = 'A'; i <= 'E'; i++) putchar(v[i] ? i : '*');
	putchar('\n');
	for (char i = 'F'; i <= 'M'; i++) putchar(v[i] ? i : '*');
	putchar('\n');
	for (char i = 'N'; i <= 'U'; i++) putchar(v[i] ? i : '*');
	putchar('\n');
	for (char i = 'V'; i <= 'Z'; i++) putchar(v[i] ? i : '*');
	puts("***");
	return 0;
}
```

#### $\text{T}2$

注意到 $B$ 最大只有 $10^7$，可以在 $O(n)$ 的时间内用线性筛求出 $B$ 以内每个数的约数和（先不减去自身）。

具体地，注意到 $n$ 的约数和 $f(n)$ 是积性函数，可以额外维护每个数最小质因数的幂 $fir[i]$，把 $i$ 分成两个互质数的乘积，将他们的约数和相乘就得到了自己的约数和。

再遍历 $[A, B]$ 范围内的每个数，按照题目定义求解计算即可。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e7 + 10;

bool v[MAXN];
int cnt, primes[MAXN / 10], fir[MAXN];

long long f[MAXN];

int main(void) {
	freopen("2554.in", "r", stdin);
	freopen("2554.out", "w", stdout);
	int A, B;
	scanf("%d%d", &A, &B);
	for (int i = 2; i <= B; i++) {
		if (!v[i]) {
			primes[cnt++] = fir[i] = i;
			f[i] = i + 1;
		}
		for (int j = 0; j < cnt; j++) {
			int k = primes[j] * i;
			if (B < k) break;
			v[k] = true;
			if (i % primes[j]) {
				f[k] = f[i] * f[primes[j]];
				fir[k] = primes[j];
			} else {
				fir[k] = fir[i] * primes[j];
				if (fir[k] == k) f[k] = f[i] + k;
				else f[k] = f[i / fir[i]] * f[fir[i] * primes[j]];
			}
		}
	}
	long long ans = 0;
	for (int i = A; i <= B; i++) {
		ans += abs(f[i] - (i << 1));
//		printf("i = %d, ans = %lld\n", i, ans);
//		printf("%x, ", abs(f[i] - (i << 1)));
	}
	printf("%lld\n", ans - (A == 1));
	return 0;
}
```

#### $\text{T}3$

先来考虑所有 $l = 1$ 的特例，即询问 $[1, r]$ 内恰好出现 $2$ 次的数的个数。

$1$ 种直观的想法是，在每个数第 $2$ 次出现的位置打上 $+1$ 标记。这样求 $[1,r]$ 的前缀和就得到了“至少出现 $2$ 次的数”的个数。

根据容斥原理，现在需要减去的是 “至少出现 $3$ 次的数”的个数。类似地，我们在每个数第 $3$ 次出现的位置打上 $-1$ 标记。

正确性可以通过分类讨论 $r$ 与数 $x$ 的关系得到证明：若 $r$ 在 $x$ 第 $2$ 次出现前，不产生贡献；若在 $x$ 第 $2$ 次出现及其后，而在第 $3$ 次出现前，$x$ 在前缀和中贡献为 1；若在第 $3$ 次出现及其后，在前缀和中贡献为 $1 - 1 = 0$。

这就意味着当且仅当 $r$ 介于 $x$ 第 $2, 3$ 次出现之间（左闭右开）时，$x$ 会对 $[1, r]$ 的统计产生贡献。这与题意的要求是 $1$ 致的。



对于普遍情况，我们需要动态地维护标记。

在线解决询问不好处理，为了延续上面的思想，我们将操作离线，每次解决 $1$ 些 $l$ 相等的询问。

按 $l$ 递减的顺序考虑询问。当 $l$ 每左移 $1$ 位，会且只会对新位置上数对应的标记产生影响。具体地，需要将旧的标记删除，并打上新的标记。



离散化之后预处理出每个数下 $1$ 个与其相等数的出现位置 $nxt[i]$。

标记的更改体现在前缀和上，就是对 $nxt[l]$ 处 $+1$，将 $nxt[nxt[l]]$ 处原有的 $+1$ 改为 $-1$（即给该位加上 $-2$），将 $nxt[nxt[nxt[l]]]$ 处原有的 $-1$ 消去（即给该位加上 $+1$）。



对于单点修改，前（后）缀询问的问题，可以简单地使用 $\text{Fenwick Tree}$ 实现。

时间复杂度为 $O(N\log N)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 1e6;
const int MAXQ = 1e6;

int n, a[MAXN], idx[MAXN], nxt[MAXN], p[MAXN];
vector <int> v;
pair <pair<int, int>, int> queries[MAXQ];
int ans[MAXQ];

int ft[MAXN];
void add(int p, int v) {
//	printf("add(%d, %d)\n", p, v);
	for (int i = p; i <= n; i += (i & -i)) ft[i] += v;
}
int ask(int p) {
//	printf("ask(%d)\n", p);
	int s = 0;
	for (int i = p; i; i -= (i & -i)) s += ft[i];
	return s;
}

int main(void) {
	freopen("2555.in", "r", stdin);
	freopen("2555.out", "w", stdout);
	int Q;
	scanf("%d%d", &n, &Q);
	for (int i = 1; i <= n; i++) {
		scanf("%d", &a[i]);
		v.push_back(a[i]);
	}
	sort(v.begin(), v.end());
	v.erase(unique(v.begin(), v.end()), v.end());
	int N = (signed)v.size();
	for (int i = n; i; i--) {
		idx[i] = lower_bound(v.begin(), v.end(), a[i]) - v.begin() + 1;
		nxt[i] = p[idx[i]];
		p[idx[i]] = i;
	}
	for (int i = 0; i < Q; i++) {
		scanf("%d%d", &queries[i].first.first, &queries[i].first.second);
		queries[i].second = i;
	}
	sort(queries, queries + Q, greater< pair<pair<int, int>, int> >());
	for (int i = 0, j = n; i < Q; i++) {
//		printf("i = %d\n", i);
		for (; queries[i].first.first <= j; j--) {
//			printf("j = %d\n", j);
			if (nxt[j]) {
				add(n - nxt[j] + 1, 1);
				if (nxt[nxt[j]]) {
					add(n - nxt[nxt[j]] + 1, -2);
					if (nxt[nxt[nxt[j]]]) add(n - nxt[nxt[nxt[j]]] + 1, 1);
				}
			}
		}
		ans[queries[i].second] = ask(n - queries[i].first.first + 1) - ask(n - queries[i].first.second);
	}
	for (int i = 0; i < Q; i++) printf("%d\n", ans[i]);
	return 0;
}
```

### $\text{#}11$

#### $\text{T}1$

典型的线性贪心。

考虑所有相邻的城市。若距离不大于 $D$，则靠东的城市被感染时可以传染到靠西的城市；否则中间必须新建城市，为了最小化城市数目，又使得病毒能够传播，显然应该每隔 $D$ 单位距离就新建 $1$ 座城市。

时间复杂度 $O(N)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 1e6;

int a[MAXN];

int main(void) {
	freopen("2560.in", "r", stdin);
	freopen("2560.out", "w", stdout);
	int N, D; scanf("%d%d", &N, &D);
	for (int i = 1; i <= N; i++) scanf("%d", &a[i]);
	int ans = 0;
	for (int i = N - 1, j = N; i; i--)
		if (a[i]) { ans += (j - i - 1) / D; j = i; }
	printf("%d\n", ans);
	return 0;
}
```

#### $\text{T}2$

考虑对于每个人，设其能够打败 $x$ 个人，则显然可以通过安排，把这些比他弱的人分到 $1$ 起，使其连续获胜 $\lfloor\log_2(x + 1)\rfloor$ 轮，即可以进入 $N - \lfloor\log_2(x + 1)\rfloor$ 轮。

而对于计算能力值不大于自己的人数，排个序就可以统计了。

时间复杂度 $O(2^N\times N)$。

```cpp
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

using namespace std;

const int MAXN = 21;

int A[1 << MAXN], B[1 << MAXN];
vector <int> v;

int M, ft[1 << MAXN];
void add(int p) { for (int i = p; i <= M; i += (i & -i)) ++ft[i]; }
int ask(int p) { int s = 0; for (int i = p; i; i -= (i & -i)) s += ft[i]; return s; }

int scan() {
	char c = getchar();	for (; !isdigit(c); c = getchar());
	int num = 0; for (; isdigit(c); c = getchar()) num = (num << 3) + (num << 1) + c - '0'; return num;
}

int main(void) {
	freopen("2561.in", "r", stdin);
	freopen("2561.out", "w", stdout);
	int N, tot; scanf("%d", &N); tot = 1 << N;
	for (int i = 0; i < tot; i++) v.push_back(A[i] = scan());
	sort(v.begin(), v.end()); v.erase(unique(v.begin(), v.end()), v.end()); M = (signed)v.size();
	for (int i = 0; i < tot; i++) {
		B[i] = lower_bound(v.begin(), v.end(), A[i]) - v.begin() + 1;
//		printf("%d ", B[i]);
	}
	for (int i = 0; i < tot; i++) add(B[i]);
	for (int i = 0; i < tot; i++) printf("%d ", N - (int)log2(ask(B[i])));
	return 0;
}
```

#### $\text{T}3$

状态压缩动态规划。

记 $f[S]$ 为将集合 $S$ 中各个字符串任意调整顺序建成 $\text{Trie}$ 树所需的最少结点数（忽略根结点），则所求即为 $f[2^n - 1] + 1$。

状态转移考虑划分子集。

将集合 $S$ 中分为子集 $s_1$ 和 $s_2$，使它们的并集为 $S$，交集为空。所建成 $\text{Trie}$ 树中，先假设各自独立建，即 $f[s_1] + f[s_2]$，再提取出 $S$ 中所有串的 $\text{LCP}$（只关心字母的数量，原先给定的排列顺序没有意义），即为公共结点，即 $f[S]=\min\{f[s_1]+f[s_2]-L\}$。

注意到其中 $L$ 与子集无关，可以单独提出在 $O(2^n\times n\times 26)$ 的时间内预处理。

子集的枚举可以通过不断减少 $2$ 进制最右边的 $1(\text{lowbit})$ 实现。所有集合的所有子集总数为 $\sum_{i=0}^n C_n^i 2^i=\sum_{i=0}^n C_n^i 2^i 1^{n-i}=(2+1)^n=3^n$ 。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

const int MAXN = 17;
const int MAXL = 2e5;
const int INF = 1e9 + 1;

char s[MAXN][MAXL];
int N, cnt[MAXN][256], tmp[256];

int getPre(int cur) {
	memset(tmp, 0x3f, sizeof tmp);
	for (int i = 0; i < N; i++)
		if (cur & (1 << i))
			for (char j = 'a'; j <= 'z'; j++)
				tmp[j] = std::min(tmp[j], cnt[i][j]);
	int res = 0;
	for (char i = 'a'; i <= 'z'; i++)
		res += tmp[i];
	return res;
}

int f[1 << MAXN];

int main(void) {
	freopen("2562.in", "r", stdin);
	freopen("2562.out", "w", stdout);

	scanf("%d", &N);
	for (int i = 0; i < N; i++) {
		scanf("%s", s[i]);
		for (int j = 0; s[i][j] != '\0'; j++)
			++cnt[i][ s[i][j] ];
	}

	int lim = 1 << N;
	for (int i = 1; i < lim; i <<= 1) {
		f[i] = getPre(i);
//		printf("f[%d] = %d\n", i, f[i]);
	}

	for (int i = 3; i < lim; i++)
		if (!f[i]) {
			f[i] = INF;
			for (int j = (i - 1) & i; j; j = (j - 1) & i) {
//				printf("i = %d, j = %d, i ^ j = %d\n", i, j, i ^ j);
				f[i] = std::min(f[i], f[j] + f[i ^ j]);
			}
			f[i] -= getPre(i);
//			printf("f[%d] = %d\n", i, f[i]);
		}

	printf("%d\n", f[lim - 1] + 1);

	return 0;
}
```

### $\text{#}12$

#### $\text{T}1$

> 给出 $1$ 棵树，有 $n$ 个点，每 $1$ 个点有点权。
>
> 在 $1$ 条路径上，第 $i$ 个点要支付 $i$ 倍点权的代价。
>
> 每次询问 $1$ 条路径，问该路径上每个点的代价之和。

考虑线性的简化问题，求 $\sum_{i=l}^r(i-l+1)a_i$。

原式化为 $\sum_{i=l}^ri\times a_i-(l-1)\sum_{i=l}^ra_i$。

由于没有任何修改，直接用前缀和分别维护 $\sum i\times a_i$ 和 $\sum a_i$ 即可求解。

---

但将问题转化到树上之后，就涉及到拼接不同段答案的问题。因此需要考虑如何用线段树维护上述内容。

也就是说，对于线段树上某个代表区间 $[l, r]$ 的 结点 $u$，都把它看作 $1$ 个子问题去求解。

令 $m=\lfloor\frac{l+r}{2}\rfloor$，假设已经算出了左右子区间 $[l,m]$ 和 $(m, r]$ 的答案，利用它们来算出当前区间的答案。

考虑 $\sum_{i=l}^r(i-l+1)a_i$ 与 $\sum_{i=l}^m(i-l+1)a_i$ 和 $\sum_{i=m+1}^r(i-m)a_i$ 之间的关系。

容易发现左子区间的答案可以直接拿来用，右子区间中各点的系数都恰好少了 $m-l+1$。

即 $\sum_{i=l}^r(i-l+1)a_i=\sum_{i=l}^m(i-l+1)a_i+\sum_{i=m+1}^r(i-m)a_i+(m-l+1)\sum_{i=m+1}^ra_i$。

因此对于每个结点需要维护的有 $\sum_{i=l}^ra_i$ 和 $\sum_{i=l}^r(i-l+1)a_i$。

---

回到原题。对于树上不改变形态的路径相关问题，很自然地想到树链剖分。

注意到本题的特殊之处在于所求不满足交换律，因此求解时不能任意交换 $S$ 和 $T$。

在给树上结点标号时，树上每条链都满足深度小的结点编号小于深度大的结点编号，即 $T$ 侧维护的信息方向与所求 $1$ 致，但 $S$ 侧则恰好相反。

考虑将求得的 $\sum_{i=l}^r(i-l+1)a_i$ 转化为 $\sum_{i=l}^r(r-i+1)a_i$。

即将 $\sum_{i=l}^ri\times a_i-l\sum_{i=l}^ra_i+\sum_{i=l}^ra_i$ 转化为 $r\sum_{i=l}^ra_i-\sum_{i=l}^ri\times a_i+\sum_{i=l}^ra_i$。

对左式乘上 $-1$，得 $l\sum_{i=1}^ra_i-\sum_{i=l}^ri\times a_i-\sum_{i=l}^ra_i$。

加上 $(r-l)\sum_{i=l}^ra_i$，得 $r\sum_{i=l}^ra_i-\sum_{i=l}^ri\times a_i-\sum_{i=l}^ra_i$。

最后再加上 $2\sum_{i=l}^ra_i$ 即可。

---

本题除了推公式稍显繁琐，其余的都是常规的树剖操作，此处不再赘述。

时间复杂度 $O(Q\log^2n)$。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>

const int MAXN = 1e6;

std::vector <int> e[MAXN];

int dep[MAXN], fa[MAXN], son[MAXN], sz[MAXN];

void dfs1(int u, int p) {
	dep[u] = dep[p] + 1, fa[u] = p, son[u] = -1, sz[u] = 1;
	for (int i = 0; i < (signed)e[u].size(); i++)
		if (e[u].at(i) != p) {
			dfs1(e[u].at(i), u);
			sz[u] += sz[e[u].at(i)];
			if (son[u] == -1 || sz[son[u]] < sz[e[u].at(i)]) son[u] = e[u].at(i);
		}
}

int n;

struct FenwickTree {
	long long v[MAXN];
	FenwickTree() {
		memset(v, 0, sizeof v);
	}

	inline int lowBit(int x) {
		return x & -x;
	}

	void add(int p, long long x) {
		for (int i = p; i <= n; i += lowBit(i))
			v[i] += x;
	}

	long long ask(int p) {
		long long s = 0;
		for (int i = p; i; i -= lowBit(i))
			s += v[i];
		return s;
	}
} ft[2];

long long A[MAXN];

int top[MAXN], w[MAXN], clk;

void dfs2(int u) {
	if (son[u] != -1) {
		w[son[u]] = ++clk, top[son[u]] = top[u];
		ft[0].add(w[son[u]], A[son[u]]);
		ft[1].add(w[son[u]], (long long)w[son[u]] * A[son[u]]);
		dfs2(son[u]);
	}
	for (int i = 0; i < (signed)e[u].size(); i++)
		if (e[u].at(i) != fa[u] && e[u].at(i) != son[u]) {
			w[e[u].at(i)] = ++clk, top[e[u].at(i)] = e[u].at(i);
			ft[0].add(w[e[u].at(i)], A[e[u].at(i)]);
			ft[1].add(w[e[u].at(i)], (long long)w[e[u].at(i)] * A[e[u].at(i)]);
			dfs2(e[u].at(i));
		}
}

int calc(int u, int v) {
	int s = 0;
	for (; top[u] != top[v]; u = fa[top[u]]) {
		if (dep[top[u]] < dep[top[v]]) std::swap(u, v);
		s += dep[u] - dep[top[u]] + 1;
	}
	if (dep[u] < dep[v]) std::swap(u, v);
	s += dep[u] - dep[v] + 1;
	return s;
}

long long query(int u, int v) {
	long long s = 0;
	int uStep = 1, vStep = calc(u, v), curStep;
	for (; top[u] != top[v]; )
		if (dep[top[u]] < dep[top[v]]) {
			curStep = dep[v] - dep[top[v]] + 1;
//			printf("curStep = %d\n", curStep);
//			printf("%lld\n", ft[1].ask(w[v]) - ft[1].ask(w[top[v]] - 1));
			s += (ft[1].ask(w[v]) - ft[1].ask(w[top[v]] - 1)) + (long long)(vStep - w[v]) * (ft[0].ask(w[v]) - ft[0].ask(w[top[v]] - 1));
//			printf("(%d, %d), s = %lld\n", top[v], v, s);
			vStep -= curStep, v = fa[top[v]];
		} else {
			curStep = dep[u] - dep[top[u]] + 1;
			s += (long long)(uStep + w[u]) * (ft[0].ask(w[u]) - ft[0].ask(w[top[u]] - 1)) - (ft[1].ask(w[u]) - ft[1].ask(w[top[u]] - 1));
//			printf("(%d, %d), s = %lld\n", u, top[u], s);
			uStep += curStep, u = fa[top[u]];
		}
//	printf("u = %d, v = %d\n", u, v);
	if (dep[u] < dep[v]) {
		curStep = dep[v] - dep[u] + 1;
//		printf("curStep = %d\n", curStep);
		s += (ft[1].ask(w[v]) - ft[1].ask(w[u] - 1)) + (long long)(vStep - w[v]) * (ft[0].ask(w[v]) - ft[0].ask(w[u] - 1));
	} else {
		curStep = dep[u] - dep[v] + 1;
//		printf("curStep = %d\n", curStep);
//		printf("%lld\n", (uStep + w[u]) * (ft[0].ask(w[u]) - ft[0].ask(w[v] - 1)));
		s += (long long)(uStep + w[u]) * (ft[0].ask(w[u]) - ft[0].ask(w[v] - 1)) - (ft[1].ask(w[u]) - ft[1].ask(w[v] - 1));
	}
	return s;
}

int main(void) {
	freopen("2577.in", "r", stdin);
	freopen("2577.out", "w", stdout);

	int Q;
	scanf("%d%d", &n, &Q);
	for (int i = 1; i <= n; i++)
		scanf("%lld", &A[i]);
	for (int i = 1; i < n; i++) {
		int a, b;
		scanf("%d%d", &a, &b);
		e[a].push_back(b), e[b].push_back(a);
	}

	dfs1(1, 0);
	son[0] = 1;
	dfs2(0);

//	for (int i = 1; i <= n; i++)
//		printf("%d ", w[i]);
//	putchar('\n');

	for (int i = 0; i < Q; i++) {
		int S, T;
		scanf("%d%d", &S, &T);
		printf("%lld\n", query(S, T));
	}
	return 0;
}
```



#### $\text{T}2$

> 有 $n$ 个数，每个数在 $10^5$ 范围内。
>
> 要求维护 $2$ 种操作：
>
> - 区间求和，再把区间内每个数 $A_i$ 变成 $\varphi(A_i)$；
> - 区间修改，把区间内所有数变成同 $1$ 个数。

首先提供 $1$ 种非标准解法，实际测试中可以通过所有的测试数据。

考虑用线段树维护。注意到某个数不断对自身求 $\varphi$，经过不超过 $\log n$ 次就会变成 $1$。证明见[此](https://ufowoqqqo.github.io/2018/07/21/Linear_sieve_and_Euler_function/#%E9%99%8D%E5%B9%82)。

假设没有 `set` 操作，则每次可以暴力递归到底层进行修改；特别地，中途发现当前区间内全都为 $1$（可以通过维护区间最大值简单地实现）时，没有继续修改的必要，可以直接返回。

总修改次数为 $O(n\log n)$。

对于 `set` 操作，仍然像常规线段树 $1$ 样使用 `lazy-tag` 维护即可。在区间置为 $\varphi$ 的过程中，若当前区间被整段重置过，则将标记置为其 $\varphi$ 之后维护对应信息并返回即可。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
//#include <ctime>
#include <algorithm>
#include <iostream>

const int MAXN = 1e6;
const int MAXV = 1e5;

long long a[MAXN], phi[MAXN];
int primes[MAXN], cnt;

struct SegmentTree {
	struct Node {
		Node *ch[2];
		long long v, tag;
		int l, r, m;
		Node(int x, int y) : v(0LL), tag(0LL), l(x), r(y), m(x + y >> 1) {
			ch[0] = ch[1] = NULL;
		}

		void pushDown() {
			if (tag) {
				ch[0]->v = tag * (long long)(ch[0]->r - ch[0]->l + 1), ch[0]->tag = tag;
				ch[1]->v = tag * (long long)(ch[1]->r - ch[1]->l + 1), ch[1]->tag = tag;
				tag = 0LL;
			}
		}

		void maintain() {
			v = ch[0]->v + ch[1]->v;
		}
	} *root;

	SegmentTree() : root(NULL) {}

	void build(Node *&u, int l, int r) {
		u = new Node(l, r);
		if (l == r) u->v = a[l];
		else {
			build(u->ch[0], l, u->m);
			build(u->ch[1], u->m + 1, r);
			u->maintain();
		}
	}

	long long query(Node *u, int l, int r) {
		if (r < u->l || u->r < l) return 0LL;
		if (l <= u->l && u->r <= r) return u->v;
		u->pushDown();
		return query(u->ch[0], l, r) + query(u->ch[1], l, r);
	}

	void updatePhi(Node *&u, int l, int r) {
		if (r < u->l || u->r < l) return;
		if (l <= u->l && u->r <= r && u->tag) {
			u->tag = phi[u->tag];
			u->v = u->tag * (long long)(u->r - u->l + 1);
			return;
		}
		if (u->l + u->v == u->r + 1) return;
		if (u->l == u->r) {
			u->v = phi[u->v];
			return;
		}
		u->pushDown();
		updatePhi(u->ch[0], l, r);
		updatePhi(u->ch[1], l, r);
		u->maintain();
	}

	void updateSet(Node *&u, int l, int r, long long v) {
		if (r < u->l || u->r < l) return;
		if (l <= u->l && u->r <= r) {
			u->v = v * (long long)(u->r - u->l + 1);
			u->tag = v;
			return;
		}
		u->pushDown();
		updateSet(u->ch[0], l, r, v);
		updateSet(u->ch[1], l, r, v);
		u->maintain();
	}
} st;

int main(void) {
//	int startTime = clock();
	freopen("2578.in", "r", stdin);
	freopen("2578.out", "w", stdout);

	phi[1] = 1;
	for (int i = 2; i <= MAXV; i++) {
		if (!phi[i]) {
			primes[cnt++] = i;
			phi[i] = i - 1;
		}
		for (int j = 0; j < cnt; j++) {
			int k = i * primes[j];
			if (MAXV < k) break;
			if (i % primes[j]) phi[k] = phi[i] * phi[primes[j]];
			else {
				phi[k] = phi[i] * primes[j];
				break;
			}
		}
	}

	int n, Q;
	scanf("%d%d", &n, &Q);
	for (int i = 1; i <= n; i++)
		scanf("%lld", &a[i]);
	st.build(st.root, 1, n);

	for (int i = 0; i < Q; i++) {
		int op, L, R;
		scanf("%d%d%d", &op, &L, &R);
		if (!op) {
			printf("%lld\n", st.query(st.root, L, R));
			st.updatePhi(st.root, L, R);
		}
		if (op == 1) {
			long long V;
			scanf("%lld", &V);
			st.updateSet(st.root, L, R, V);
		}
	}

//	printf("time used = %d millisecond(s)\n", clock() - startTime);
	return 0;
}
```

---

正解是 $\text{Splay}$。

之前的思路中，利用“全为 $1$ 的区间没有修改意义”这点非常重要。

考虑把区间设置的整个区间看成 $1$ 个整体来操作，那么整体只会被修改 $\log n$ 次，而且每次修改可以算出总和。

具体地，$\text{Splay}$ 中每个结点代表 $1$ 个值全部相同的区间。需要维护的值有**子树中所有区间**的长度之和及**子树中所有区间**的数的总和。

这样将区间置 $\varphi$ 时，分离出对应的区间，再遍历得到的树即可。把处理完后值为 $1$ 的区间记录下来并删除。对于每次询问，询问区间的长度减去得到的由对应区间内不为 $1$ 的数组成的树的大小即为 $1$ 的数量。

---

考虑实现。

区间的分离可以用[非旋转式平衡树](https://ufowoqqqo.github.io/2018/05/26/non_rotation_treap/)中的`split`操作实现。细节此处不再赘述。

对于区间置 $\varphi$ 之后的删除全 $1$ 区间问题，可以把遍历到的全 $1$ 区间记录下来，并逐个删除。

考虑到每个点只会被删 $1$ 次，总时间复杂度为 $O(n\log n)$。



#### $\text{T}3$

> 给出 $1$ 棵 $n$ 个点的树，$1$ 开始边权都为 $1$。$n\le 10^5$
>
> 有 $3$ 种操作。
>
> 路径修改，全部边权 $+1$。
>
> 取消某 $1$ 次修改。
>
> 询问 $1$ 条路径上有多少边权为 $1$ 的边。

考虑线性的简化问题。

用特殊的删除标记即可解决。

维护当前区间被整体 $+1$ 的次数。当且仅当次数为 $0$ 时，当前区间没有被删除。

删除标记不下传，这样可以保证取消操作的复杂度也是 $O(\log n)$。事实上也没有必要下传，因为询问时遇到删除标记直接返回即可。

再套上树剖即可。

> 这题是坠简单的，我的线段树却打错爆 $0$ 了，好菜啊 $\text{:(}$

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>

const int MAXN = 1e6;
const int MAXQ = 1e6;

std::vector <int> e[MAXN];

int dep[MAXN], fa[MAXN], son[MAXN], sz[MAXN];

void dfs1(int u, int p) {
	dep[u] = dep[p] + 1, fa[u] = p, son[u] = -1, sz[u] = 1;
	for (int i = 0; i < (signed)e[u].size(); i++)
		if (e[u].at(i) != p) {
			dfs1(e[u].at(i), u);
			sz[u] += sz[e[u].at(i)];
			if (son[u] == -1 || sz[son[u]] < sz[e[u].at(i)]) son[u] = e[u].at(i);
		}
}

struct SegmentTree {
	struct Node {
		Node *ch[2];
		int v, times;
		int l, r, m;
		Node(int x, int y) : v(0), times(0), l(x), r(y), m(x + y >> 1) {
			ch[0] = ch[1] = NULL;
		}

		void maintain() {
			if (ch[0] && ch[1]) v = ch[0]->v + ch[1]->v; else v = 0;
		}
	} *root;

	void build(Node *&u, int l, int r) {
		u = new Node(l, r);
		if (l < r) {
			build(u->ch[0], l, u->m);
			build(u->ch[1], u->m + 1, r);
		}
	}

	void update(Node *&u, int l, int r, int v) {
//		printf("update(%d, %d), (%d, %d)\n", u->l, u->r, l, r);
		if (r < u->l || u->r < l) return;
		if (l <= u->l && u->r <= r) {
			u->times += v;
			if (u->times) u->v = u->r - u->l + 1; else u->maintain();
//			printf("updated, node[%d, %d] v = %d, times = %d\n", u->l, u->r, u->v, u->times);
			return;
		}
		update(u->ch[0], l, r, v);
		update(u->ch[1], l, r, v);
		if (!u->times) u->maintain();
	}

	int query(Node *u, int l, int r) {
//		printf("query(%d, %d), (%d, %d)\n", u->l, u->r, l, r);
		if (l <= u->l && u->r <= r) {
//			printf("queried, node[%d, %d] v = %d, times = %d\n", u->l, u->r, u->v, u->times);
			return u->v;
		}
		if (u->l <= l && r <= u->r && u->times) return r - l + 1;
		if (l <= u->m && r <= u->m) return query(u->ch[0], l, r);
		else if (u->m < l && u->m < r) return query(u->ch[1], l, r);
		return query(u->ch[0], l, u->m) + query(u->ch[1], u->m + 1, r);
	}
} st;

int top[MAXN], w[MAXN], clk;

void dfs2(int u) {
	if (son[u] != -1) {
		w[son[u]] = ++clk, top[son[u]] = top[u];
		dfs2(son[u]);
	}
	for (int i = 0; i < (signed)e[u].size(); i++)
		if (e[u].at(i) != fa[u] && e[u].at(i) != son[u]) {
			w[e[u].at(i)] = ++clk, top[e[u].at(i)] = e[u].at(i);
			dfs2(e[u].at(i));
		}
}

void add(int u, int v, int x) {
//	if (u == v) return;
	for (; top[u] != top[v]; u = fa[top[u]]) {
		if (dep[top[u]] < dep[top[v]]) std::swap(u, v);
//		printf("top[u] = %d, u = %d, add(%d, %d)\n", top[u], u, w[top[u]], w[u]);
		st.update(st.root, w[top[u]], w[u], x);
	}
	if (dep[u] < dep[v]) std::swap(u, v);
//	printf("v = %d, u = %d, add(%d, %d)\n", v, u, w[v] + 1, w[u]);
	if (w[v] != w[u]) st.update(st.root, w[v] + 1, w[u], x);
}

int solve(int u, int v) {
	int s = 0;
	for (; top[u] != top[v]; u = fa[top[u]]) {
		if (dep[top[u]] < dep[top[v]]) std::swap(u, v);
		s += dep[u] - dep[top[u]] + 1 - st.query(st.root, w[top[u]], w[u]);
	}
	if (dep[u] < dep[v]) std::swap(u, v);
	if (w[v] != w[u]) s += dep[u] - dep[v] - st.query(st.root, w[v] + 1, w[u]);
	return s;
}

std::pair<int, int> paths[MAXQ];

int main(void) {
	freopen("2579.in", "r", stdin);
	freopen("2579.out", "w", stdout);

	int n, Q;
	scanf("%d%d", &n, &Q);
	for (int i = 1; i < n; i++) {
		int x, y;
		scanf("%d%d", &x, &y);
		e[x].push_back(y), e[y].push_back(x);
	}
	dfs1(1, 0);
	son[0] = 1;
	st.build(st.root, 1, n);
	dfs2(0);

//	for (int i = 1; i <= n; i++)
//		printf("%d ", w[i]);
//	putchar('\n');

	int cntPaths = 0;
	for (int i = 0; i < Q; i++) {
		int op;
		scanf("%d", &op);
		if (!op) {
			++cntPaths;
			scanf("%d%d", &paths[cntPaths].first, &paths[cntPaths].second);
			add(paths[cntPaths].first, paths[cntPaths].second, 1);
		}
		if (op == 1) {
			int p;
			scanf("%d", &p);
			add(paths[p].first, paths[p].second, -1);
		}
		if (op == 2) {
			int x, y;
			scanf("%d%d", &x, &y);
			printf("%d\n", solve(x, y));
		}
//		putchar('\n');
	}
	return 0;
}
```

### $\text{#}13$

#### $\text{T}1$

直接枚举每 $1$ 个串的每 $1$ 个位置匹配即可。

时间复杂度 $O(nmQ)$。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#define N 1020
#define M 1020
using namespace std;

int text[N][M], pattern[M];

int Scan(void)
{
	char c;
	int o, k;

	for(o = 0, k = 1; (c = getchar()) != EOF && !isdigit(c) && c != '-'; )
		;
	if(c == '-')
	{
		k = -1;
		c = getchar();
	}
	do
		o = (o << 3) + (o << 1) + c - '0';
	while((c = getchar()) != EOF && isdigit(c));

	return o * k;
}

bool match(int *x, int *y, int length)
{
	int index;

	for(index = 0; index < length; index ++)
		if(x[index] != y[index] && y[index] != -1)
			return false;

	return true;
}

int main(void)
{
	int n, m, Q, answer;
	int index, j;

	freopen("2596.in", "r", stdin);
	freopen("2596.out", "w", stdout);

	n = Scan(), m = Scan();
	for(index = 0; index < n; index ++)
		for(j = 0; j < m; j ++)
			text[index][j] = Scan();
	Q = Scan();
	for(index = 0; index < Q; index ++)
	{
		for(j = 0; j < m; j ++)
			pattern[j] = Scan();
		for(j = 0, answer = 0; j < n; j ++)
			answer += match(text[j], pattern, m);
		printf("%d\n", answer);
	}

	return 0;
}
```

#### $\text{T}2$

题解见[此](https://ufowoqqqo.github.io/2018/02/02/con261/#textt1-deda)。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#define INFINITY 1000000001
using namespace std;

class SegmentTree
{
public:
	class Node
	{
	public:
		Node *child[2];
		int minimumValue;
		int left, right, middle;
		Node(int x, int y) : minimumValue(INFINITY), left(x), right(y), middle((x + y )>> 1)
		{
			child[0] = child[1] = NULL;

			return;
		}

		void maintain(void)
		{
			minimumValue = min(child[0]->minimumValue, child[1]->minimumValue);

			return;
		}
	} *root;

	SegmentTree(void) : root(NULL)
	{
		return;
	}

	void build(Node *¤t, int left, int right)
	{
		current = new Node(left, right);
		if(left < right)
		{
			build(current->child[0], left               , current->middle);
			build(current->child[1], current->middle + 1, right          );
		}

		return;
	}

	void update(Node *¤t, int position, int value)
	{
		if(current->left == current->right)
		{
			current->minimumValue = min(current->minimumValue, value);

			return;
		}
		update(current->child[current->middle < position], position, value);
		current->maintain();

		return;
	}

	int query(Node *current, int position, int value)
	{
		int temp;

		if(value < current->minimumValue || current->right < position)
			return -1;
		if(current->left == current->right)
			return current->left;
		if(current->child[0]->minimumValue <= value)
		{
			temp = query(current->child[0], position, value);
			if(temp != -1)
				return temp;
		}

		return query(current->child[1], position, value);
	}
} ghaSTLcon;

int main(void)
{
	int N, Q, XY, AB;
	int index;
	char op[2];

	freopen("2597.in", "r", stdin);
	freopen("2597.out", "w", stdout);

	scanf("%d %d", &N, &Q);
	ghaSTLcon.build(ghaSTLcon.root, 1, N);
	for(index = 0; index < Q; index ++)
	{
		scanf("%s %d %d", op, &XY, &AB);
		if(!strcmp(op, "M"))
			ghaSTLcon.update(ghaSTLcon.root, AB, XY);
		if(!strcmp(op, "D"))
			printf("%d\n", ghaSTLcon.query(ghaSTLcon.root, AB, XY));
	}
	return 0;
}
```

#### $\text{T}3$

设 $E[i]$ 为已经匹配了前 $i$ 个数字后， 匹配完整个串还需要的期望生成数字数。

边界条件是显然的。对于长度为 $k$ 的前缀，有 $E[k] = 0$。

转移时，由于在当前状态下在 $n$ 种可能的分支中有且只有 $1$ 种能够使得匹配位数增加 $1$ 位，其余都不会使已有的匹配结果变得更优。

即 $E[i]=\frac{E[i+1]}{n}+\frac{\sum E[j]}{n}+1$，其中 $j$ 是其余的 $(n - 1)$ 种分支到达的新匹配位置。

先忽略 $j$ 的求解方法，假设已经通过某种手段求得 $(n - 1)$ 个分支对应的 $j$。

---

注意到上面的式子中，$E[i]$ 既与前面有关，又与后面有关，这使问题看似复杂。

但后面永远是 $i + 1$，不妨稍作变形，得到 $E[i+1]=n\times E[i]-\sum E[j] -n$。

这样就能够从前往后递推了。

---

然后我们就发现，已知的边界与递推的方向恰好是相反的。

考虑逆推。

设 $E[0] = x$，最终可以得到 $E[k] = ax+b = 0$，即 $x = -\frac{b}{a}$。

通过简单的数学归纳法不难得到，所有 $E[i]$ 中 $x$ 的系数均为 $1$，因此只需记录常数项即可。

---

最后考虑“当前情况下再填上某个字符转移到的位置”如何求得。

可以使用 $\text{KMP}$ 算法，但是由于每位有多种失配方式，会使复杂度高达 $O(M^2)$。

注意到当前位置与失配位置有 $1$ 部分前缀相等，下 $1$ 位的结果可以继承。也就是构建只有 $1$ 个串的 $\text{AC}$ 自动机即可。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#define MAXN 120
#define MAXM 1000020
#define ghaSTLcon 1000000007LL
using namespace std;

int Scan(void)
{
	char c;
	int o;

	for(o = 0; (c = getchar()) != EOF && !isdigit(c); )
		;
	do
		o = (o << 3) + (o << 1) + c - '0';
	while((c = getchar()) != EOF && isdigit(c));

	return o;
}

int a[MAXM], next[MAXM][MAXN];
long long E[MAXM];

int main(void) //2598.cpp
{
	int N, M;
	int index, j, k;

	freopen("2598.in" , "r", stdin);
	freopen("2598.out", "w", stdout);

	N = Scan();
	M = Scan();
	for(index = 1; index <= M; index ++)
		a[index] = Scan();

	for(next[0][a[1]] = 1, index = 1, j = 0; index < M; index ++)
	{
		for(k = 1; k <= N; k ++)
			next[index][k] = next[j][k];
		next[index][a[index + 1]] = index + 1;
		j = next[j][a[index + 1]];
	}

	for(index = 0; index < M; index ++)
	{
		E[index + 1] = (((E[index] - 1LL) * (long long )N) % ghaSTLcon + ghaSTLcon) % ghaSTLcon;
		for(j = 1; j <= N; j ++)
			if(next[index][j] != index + 1) (((E[index + 1] -= E[next[index][j]]) %= ghaSTLcon) += ghaSTLcon) %= ghaSTLcon;
	}

	for(index = 1; index <= M; index ++)
		printf("%lld\n", ((-E[index] % ghaSTLcon) + ghaSTLcon) % ghaSTLcon);
	return 0;
}
```

### $\text{#}14$

#### $\text{T}1$

> 给出 $1$ 棵 $n$ 个点的树，树上每 $1$ 个点 $i$ 都有点权 $A_i$。求
> $$
> \sum_{i=1}^n\sum_{j=1}^n(\max(i, j) - \min(i, j))\times len(i,j)
> $$
> 其中 $\max(i,j)$ 表示从 $i$ 到 $j$ 的路径的点权的最大值，$\min(i, j)$ 则为最小值，$len(i, j)$ 表示该路径的点数。

这题询问树上所有路径，考虑点分治。

将原式拆开，根据乘法分配律得到所求为
$$
\sum_{i=1}^n\sum_{j=1}^n\max(i,j)\times len(i,j)-\sum_{i=1}^n\sum_{j=1}^n\min(i,j)\times len(i,j)
$$
注意到 $2$ 部分的求解是类似的。我们可以只考虑如何计算 $\max$ 的情况，然后将所有点权值取反，此时所有点权为负数，原来最小的现在变成了最大，恰好可以用相同的方式求解。

---

对于分治重心 $x$，考虑各子树经过 $x$ 到达其他子树的路径对答案的贡献。

由于询问的是无序点对，为了避免重复计算，只考虑“当前子树”与“之前子树”之间的路径。

对于当前子树中的某个点 $p$，在暴力时预处理出 $x$ 到 $p$ 经过的点数 $dist[p]$ 以及 $x$ 到 $p$ 路径上的最大值 $A[p]$。

在与之前子树中的某个结点 $q$ 组成路径时，显然有
$$
len(p, q)=dist[p]+dist[q]-1\\
\max(p,q)=\max\{A[p],A[q]\}
$$
正如之前遇到过的绝对值符号 $1$ 样，我们发现取 $\max$ 也是无法直接统计的。

---

考虑分类讨论。

若 $A[q]\le A[p]$，则每个 $q$ 都有对答案的贡献为
$$
A[p]\times(dist[p]+dist[q]-1)=A[p]\times(dist[p]-1)+A[p]\times dist[q]
$$
所有满足条件的 $q$ 的总贡献即为
$$
A[p]\times(dist[p]-1)\times\sum[A[q]\le A[p]]+A[p]\times\sum dist[q][A[q]\le A[p]]
$$
类似地，若 $A[p] < A[q]$，则每个 $q$ 都有对答案的贡献为
$$
A[q]\times(dist[p]+dist[q]-1)=A[q]\times(dist[p]-1)+A[q]\times dist[q]
$$
所有满足条件的 $q$ 的总贡献即为
$$
(dist[p]-1)\times\sum A[q][A[p]<A[q]]+\sum A[q]dist[q][A[p]<A[q]]
$$
注意到上述 $4$ 个 $\sum$ 都满足关于 $A$ 的前缀和形式，可以简单地使用 $\text{Fenwick Tree}$ 维护。

每 $1$ 层递归时间复杂度 $O(n\log n)$。

由 $\text{Master}$ 定理可知总时间复杂度 $O(n\log n)$。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#define N 50020
#define MAXA 10020
using namespace std;

long long a[N], dist[N];
int size[N], maxSubtree[N];
vector<int> edges[N];
bool visited[N];

class Queue
{
public:
    int value[N];
    int head, tail;
    Queue(void) : head(1), tail(0)
    {
        memset(value, 0, sizeof value);

        return;
    }

    bool empty(void)
    {
        return tail < head;
    }

    int front(void)
    {
        return value[head];
    }

    void push(int x)
    {
        value[++ tail] = x;

        return;
    }

    void pop(void)
    {
        value[head ++] = 0;
        if(empty())
            head = 1, tail = 0;

        return;
    }
} queueTemp, queueAll;

long long A[N];

void depthFirstSearch(int x)
{
    int i;

    visited[x] = true;
    size[x] = 1;
    maxSubtree[x] = -1;
    queueTemp.push(x);
    for(i = 0; i < (signed)edges[x].size(); i ++)
        if(!visited[edges[x].at(i)])
        {
            A[edges[x].at(i)] = max(A[x], a[edges[x].at(i)]);
            dist[edges[x].at(i)] = dist[x] + 1;
            depthFirstSearch(edges[x].at(i));
            size[x] += size[edges[x].at(i)];
            maxSubtree[x] = max(maxSubtree[x], size[edges[x].at(i)]);
        }
    visited[x] = false;

    return;
}

class FenwickTree
{
public:
    long long value[MAXA << 1];
    FenwickTree(void)
    {
        memset(value, 0, sizeof value);

        return;
    }

    inline int lowBit(int x)
    {
        return x & -x;
    }

    void update(int position, long long delta)
    {
        int i;

        for(i = position; i < (MAXA << 1); i += lowBit(i))
            value[i] += delta;

        return;
    }

    long long query(int position)
    {
        long long result;
        int i;

        for(result = 0LL, i = position; i; i -= lowBit(i))
            result += value[i];

        return result;
    }
} lessOrEqualCount, dPrefixSum, aPrefixSum, timesPrefixSum;

int offset;

long long solve(int x)
{
    int total;
    long long answer;
    int i, j, p;

    for(; !queueTemp.empty(); queueTemp.pop())
        ;
    depthFirstSearch(x);
    for(total = size[x]; !queueTemp.empty(); queueTemp.pop())
    {
        maxSubtree[queueTemp.front()] = max(maxSubtree[queueTemp.front()], total - size[queueTemp.front()]);
        if(maxSubtree[queueTemp.front()] < maxSubtree[x])
            x = queueTemp.front();
    }

    visited[x] = true;
    for(; !queueAll.empty(); queueAll.pop())
        ;
    lessOrEqualCount.update(a[x] + offset, 1LL);
    dPrefixSum.update(a[x] + offset, 1LL);
    aPrefixSum.update(a[x] + offset, a[x]);
    timesPrefixSum.update(a[x] + offset, a[x]);
    for(i = answer = 0; i < (signed)edges[x].size(); i ++)
        if(!visited[edges[x].at(i)])
        {
            A[edges[x].at(i)] = max(a[x], a[edges[x].at(i)]);
            dist[edges[x].at(i)] = 2LL;
            depthFirstSearch(edges[x].at(i));
            for(j = queueTemp.head; j <= queueTemp.tail; j ++)
            {
                p = queueTemp.value[j];
                answer += A[p] * (dist[p] - 1LL) * lessOrEqualCount.query(A[p] + offset) + A[p] * dPrefixSum.query(A[p] + offset)
                       +  (aPrefixSum.query(MAXA - 1 + offset) - aPrefixSum.query(A[p] + offset)) * (dist[p] - 1LL) + (timesPrefixSum.query(MAXA - 1 + offset) - timesPrefixSum.query(A[p] + offset));
            }
            for(; !queueTemp.empty(); queueTemp.pop())
            {
                p = queueTemp.front();
                lessOrEqualCount.update(A[p] + offset, 1LL);
                dPrefixSum.update(A[p] + offset, dist[p]);
                aPrefixSum.update(A[p] + offset, A[p]);
                timesPrefixSum.update(A[p] + offset, A[p] * dist[p]);
                queueAll.push(p);
            }
        }

    for(; !queueAll.empty(); queueAll.pop())
    {
        p = queueAll.front();
        lessOrEqualCount.update(A[p] + offset, -1LL);
        dPrefixSum.update(A[p] + offset, -dist[p]);
        aPrefixSum.update(A[p] + offset, -A[p]);
        timesPrefixSum.update(A[p] + offset, -(A[p] * dist[p]));
    }
    lessOrEqualCount.update(a[x] + offset, -1LL);
    dPrefixSum.update(a[x] + offset, -1LL);
    aPrefixSum.update(a[x] + offset, -a[x]);
    timesPrefixSum.update(a[x] + offset, -a[x]);

    for(i = 0; i < (signed)edges[x].size(); i ++)
        if(!visited[edges[x].at(i)])
            answer += solve(edges[x].at(i));

    return answer;
}

int main(void) //2565.cpp
{
    int n, u, v;
    long long answer;
    int i;

    freopen("2565.in" , "r", stdin);
    freopen("2565.out", "w", stdout);

    scanf("%d", &n);
    for(i = 0; i < n; i ++)
        scanf("%lld", &a[i]);
    for(i = 1; i < n; i ++)
    {
        scanf("%d %d", &u, &v);
        edges[-- u].push_back(-- v);
        edges[   v].push_back(   u);
    }
    offset = 1;
    answer = solve(0);

    for(i = 0; i < n; i ++)
        a[i] *= -1LL;
    memset(visited, false, sizeof visited);
    offset = 10001;
    printf("%lld\n", answer + solve(0));

    return 0;
}
```

#### $\text{T}2$

> 给出 $1$ 颗 $n$ 个点的树，某些节点间是敌对关系。
>
> 如果 $1$ 个点的祖先和另外 $1$ 个点的祖先是独对关系，那么这 $2$ 个点也是敌对关系。
>
> 询问 $Q$ 次，每次问 $2$ 个点是不是敌对关系。

根据题意，$1$ 旦与某个点为敌，就会与以该点为根的子树为敌。

$1$ 棵子树在 $\text{DFS}$ 序中总是连续的 $1$ 段，可以考虑用线段树上标记对应段维护每个点的敌对关系。

注意到敌对关系是可以继承的，每个节点的所有敌人都是其祖先的所有敌人与自身敌人的并集。

因此每次只需要在父亲结点的基础上新标记自身独有的敌人即可，这符合可持久化的特征。

在实现上需要注意，进行区间修改不好操作，由于是单点询问，可以转为差分的形式，即在敌对点子树的 $\text{DFS}$ 序左端点 $+1$，右端点后 $1$ 个点 $-1$，询问某个点时只需查询前缀和，若大于 $0$ 则是敌对关系。

时间复杂度 $O(n\log n)$。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#include <vector>
#define N 30020
#define MAXQ 100020
using namespace std;

vector<int> edges[N], enemies[N];
vector<pair<int, int> > queries[N];

bool answer[MAXQ];

int currentEnemies[N];

int clk;
int l[N], r[N];

void preVisit(int x)
{
	int i;

	l[x] = clk ++;
	for(i = 0; i < (signed)edges[x].size(); i ++)
		preVisit(edges[x].at(i));
	r[x] = clk - 1;
}

class PersistentSegmentTree
{
public:
	class Node
	{
	public:
		Node *child[2];
		int sum;
		int left, right, middle;
		Node(int x = 0, int y = 0) : sum(0), left(x), right(y), middle((x + y) >> 1)
		{
			child[0] = child[1] = NULL;

			return;
		}

		void maintain(void)
		{
			sum = child[0]->sum + child[1]->sum;

			return;
		}
	} *roots[N];
	PersistentSegmentTree(void)
	{
		memset(roots, 0, sizeof roots);

		return;
	}

	void build(Node *¤t, int left, int right)
	{
		current = new Node(left, right);
		if(left < right)
		{
			build(current->child[0], left               , current->middle);
			build(current->child[1], current->middle + 1, right          );
		}

		return;
	}

	void update(Node *¤t, Node *last, int position, int value)
	{
		int direction;
		current = new Node();
		*current = *last;
		if(current->left == current->right)
		{
			current->sum += value;

			return;
		}
		direction = current->middle < position;
		update(current->child[direction], last->child[direction], position, value);
		current->maintain();

		return;
	}

	int query(Node *current, int left, int right)
	{
		if(right < current->left || current->right < left)
			return 0;
		if(left <= current->left && current->right <= right)
			return current->sum;
		return query(current->child[0], left, right) + query(current->child[1], left, right);
	}
} ghaSTLcon;

int n, parent[N];

void depthFirstSearch(int x)
{
	int i;
	bool f__k;

	if(!enemies[x].size())
	{
		ghaSTLcon.roots[x] = new PersistentSegmentTree::Node();
		*ghaSTLcon.roots[x] = *ghaSTLcon.roots[parent[x]];
	}

	for(i = 0, f__k = true; i < (signed)enemies[x].size(); i ++)
	{
		ghaSTLcon.update(ghaSTLcon.roots[x], f__k ? ghaSTLcon.roots[parent[x]] : ghaSTLcon.roots[x], l[enemies[x].at(i)], 1);
		f__k = false;
		if(r[enemies[x].at(i)] + 1 < n)
			ghaSTLcon.update(ghaSTLcon.roots[x], ghaSTLcon.roots[x], r[enemies[x].at(i)] + 1, -1);
	}
	for(i = 0; i < (signed)edges[x].size(); i ++)
		depthFirstSearch(edges[x].at(i));

	return;
}

int main(void) //2599.cpp
{
	int m, x, y, Q;
	int i;

	freopen("2599.in" , "r", stdin);
	freopen("2599.out", "w", stdout);

	scanf("%d", &n);
	for(i = 1; i < n; i ++)
	{
		scanf("%d", &parent[i]);
		edges[-- parent[i]].push_back(i);
	}
	scanf("%d", &m);
	for(i = 0; i < m; i ++)
	{
		scanf("%d %d", &x, &y);
		enemies[-- x].push_back(-- y);
		enemies[   y].push_back(   x);
	}

	preVisit(0);
	ghaSTLcon.build(ghaSTLcon.roots[n], 0, n - 1);
	parent[0] = n;
	depthFirstSearch(0);

	scanf("%d", &Q);
	for(i = 0; i < Q; i ++)
	{
		scanf("%d %d", &x, &y);
		puts(ghaSTLcon.query(ghaSTLcon.roots[-- x], 0, l[-- y]) > 0 ? "Yes" : "No");
	}

	return 0;
}
```

#### $\text{T}3$

> 有 $1$ 个 $n\times m$ 的平面，第 $i$ 排的第 $L_i$ 到 $R_i$ 列中每 $1$ 格的数都是 $A_i$，其余格子为 $0$。
>
> 有 $Q$ 次询问，问 $1$ 个矩形的总和。
>
> 强制在线。

很容易想到 $2$ 维前缀和的形式，但是会发现数据范围过大而无法直接实现。

但是这启示我们，部分和可以差分，考虑使用线段树维护每列的前缀和。

注意到我们在直接维护前缀和的过程中，实际上每行只有 $L_i$ 到 $R_i$ 这部分的值会发生改变，其余都与上 $1$ 行完全 $1$ 致。使用标记永久化的方式即可。

```cpp
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <iostream>
#define N 100020
using namespace std;

class PersistentSegmentTree
{
public:
	class Node
	{
	public:
		Node *child[2];
		long long sum, delta;
		int left, right, middle;
		Node(int x = 0, int y = 0) : sum(0LL), delta(0LL), left(x), right(y), middle((x + y) >> 1)
		{
			child[0] = child[1] = NULL;

			return;
		}
	} *roots[N];
	PersistentSegmentTree(void)
	{
		memset(roots, 0, sizeof roots);

		return;
	}

	void build(Node *¤t, int left, int right)
	{
		current = new Node(left, right);
		if(left < right)
		{
			build(current->child[0], left               , current->middle);
			build(current->child[1], current->middle + 1, right          );
		}

		return;
	}

	void update(Node *¤t, Node *last, int left, int right, long long delta)
	{
		if(right < last->left || last->right < left)
			return;
		current = new Node();
		*current = *last;
		current->sum += delta * (long long)(min(current->right, right) - max(current->left, left) + 1);
		if(left <= current->left && current->right <= right)
		{
			current->delta += delta;

			return;
		}
		update(current->child[0], last->child[0], left, right, delta);
		update(current->child[1], last->child[1], left, right, delta);

		return;
	}

	long long query(Node *current, long long deltaSum, int left, int right)
	{
		if(right < current->left || current->right < left)
			return 0LL;
		if(left <= current->left && current->right <= right)
			return current->sum + deltaSum * (long long)(current->right - current->left + 1);
		return query(current->child[0], deltaSum + current->delta, left, right) + query(current->child[1], deltaSum + current->delta, left, right);
	}
} ghaSTLcon;

int L[N], R[N];
long long A[N];

int main(void) //2600.cpp
{
	int n, m;
	int Q;
	long long a, b, c, d, ans;
	int i;

	freopen("2600.in" , "r", stdin);
	freopen("2600.out", "w", stdout);

	scanf("%d %d", &n, &m);
	ghaSTLcon.build(ghaSTLcon.roots[0], 1, m);
	for(i = 1; i <= n; i ++)
	{
		scanf("%d %d %lld", &L[i], &R[i], &A[i]);
		ghaSTLcon.update(ghaSTLcon.roots[i], ghaSTLcon.roots[i - 1], L[i], R[i], A[i]);
	}
	scanf("%d", &Q);
	for(ans = 0LL, i = 0; i < Q; i ++)
	{
		scanf("%lld %lld %lld %lld", &a, &b, &c, &d);
		a ^= ans;
		b ^= ans;
		c ^= ans;
		d ^= ans;
		ans = ghaSTLcon.query(ghaSTLcon.roots[c], 0, b, d) - ghaSTLcon.query(ghaSTLcon.roots[a - 1], 0, b, d);
		printf("%lld\n", ans);
	}

	return 0;
}
```
