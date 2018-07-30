---
layout:     post
title:      "2 分图学习笔记"
date:       2018-07-26 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 二分图
---



#### 定义

$2$ 分图指能够将点集分为 $2$ 个互不相交的子集，使得每条边所连接的顶点分属于 $2$ 个子集的图。



#### 判定

从所有未被染色的点出发，对图进行 `dfs` 染色，每次将相邻点染成与当前点相反的颜色，若在此过程中产生冲突，则图不是 $2$ 分图。

不难发现，当图中存在奇环时，$1$ 定不是 $2$ 分图。



#### 最大匹配

选出最多的边，使得任意点最多只连了 $1$ 条边。

可以建模成最大流解决，也有专门的匹配算法——匈牙利算法。

---

匹配点和未匹配点：已选取的匹配边 $2$ 端的点称作匹配点，其余为未匹配点。

增广路：从 $1$ 端的未匹配点开始，沿着非匹配边 $\Rightarrow$ 匹配边 $\Rightarrow$ 非匹配边 $\Rightarrow$ 匹配边 $\Rightarrow$ …… $\Rightarrow$ 非匹配边，最后到达另 $1$ 端的匹配点的路径。

对于找到的增广路，将匹配边与非匹配边交换，匹配数 $+1$。

每次用 `dfs` 找增广路，直至图中不存在增广路，所得的就是最大匹配。时间复杂度 $O(VE)$。

```cpp
#include <cstdio>
#include <vector>
#include <cstring>

using namespace std;

const int maxn = 1000;

int match[maxn];
int n, m;
bool vis[maxn];

vector <int> E[maxn];

bool DFS(int);

int main()
{
	#ifdef LOCAL
		freopen("perfectstall.in", "r", stdin);
		freopen("perfectstall.out", "w", stdout);
	#endif

	while (~scanf("%d%d", &n, &m))
	{
		memset(match, 0, sizeof(match));
		for (int i = 1; i <= n; i++)
		{
			E[i].clear();
			int s;
			scanf("%d", &s);
			int x;
			for (int j = 0; j != s; j++ )
			{
				scanf("%d", &x);
				E[i].push_back(x);
			}
		}

		int ans = 0;//匈牙利
		for (int i = 1; i <= n; i++ )
		{
			memset(vis, 0, sizeof(vis));
			if (DFS(i)) ans++;
		}
		printf("%d\n",ans);
	}

	return 0;
}

bool DFS(int x)
{
	for (int i = 0; i != E[x].size(); i++)
	{
		int y = E[x][i];
		if (vis[y]) continue;
		vis[y] = 1;
		if (match[y] == 0 || DFS(match[y]))//寻找增广路
		{
			match[y] = x;//增广路取反
			return 1;
		}
	}
	return 0;
}
```



#### 最小点覆盖

在 $2$ 分图上选取最少的点，使所有边都连到至少 $1$ 个被选的点。

$\text{K}\ddot{o}\text{ing}$ 定理：**最小点覆盖 = 最大边匹配**。



#### 最大独立集

选最多点，使任意 $2$ 点之间没有连边。

把最小点覆盖所选点取反即可。

> - 这样选出来的是独立集。假如有 $2$ 个点不在最小点覆盖中，而且它们之间有边，则么这条边就没有被覆盖，就违反了前提。
> - 由于最小点覆盖选择了最少的点，所以剩下来的就是最大独立集。



#### $\text{DAG}$ 的最小路径覆盖

##### 路径不相交

在 $\text{DAG}$ 中选取最少路径，使每个点**恰好**在 $1$ 条路径中，即不允许路径之间相交。单独 $1$ 个点也算作 $1$ 条路径。

把每个点拆成入点 $u^-$和出点 $u^+$，对于原图中的边 $(u, v)$，由 $u^+$ 向 $v^-$ 连边，这样就得到了 $2$ 分图。

点数与最大边匹配之差就是最小路径覆盖。

> 每 $1$ 个点只能在 $1$ 条路径里面，所以每 $1$ 个点只能选择 $1$ 条出边，$1$ 条入边。
>
> 相当于在 $2$ 分图中，每 $1$ 个出点只能匹配 $1$ 个入点。每 $1$ 个匹配相当于用 $1$ 条路径把 $2$ 个点连了起来。
>
> 假设原来每 $1$ 个点是 $1$ 条路径，那么每 $1$ 条匹配就会减少 $1$ 条路径。

##### 路径相交

在 $\text{DAG}$ 中选取最少路径，使每个点**至少**在 $1$ 条路径中，即允许路径之间相交。

>  假如 $1$ 条路径和另外 $1$ 条路径相交了，那么肯定有 $1$ 个点同时在这 $2$ 条路径上。
>
> 我们可以视为：其中 $1$ 条路径**跳过**了这个点，这样仍然满足每个点都在且只在 $1$ 条路径上。

对于原图中的顶点 $u$，向其**可达**的所有点 $v$ 都直接连边。

可以用 $\text{Floyd}$ 传递闭包来算出新的图，然后按不相交路径覆盖来计算。

