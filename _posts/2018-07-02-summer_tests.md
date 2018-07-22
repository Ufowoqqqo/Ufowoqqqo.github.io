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
