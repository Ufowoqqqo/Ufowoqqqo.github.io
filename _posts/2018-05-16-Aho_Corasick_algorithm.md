---
layout:     post
title:      "AC 自动机学习笔记"
subtitle:   "Let's get Accepted"
date:       2018-05-16 12:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 算法笔记
    - AC 自动机
---

### $\text{What}$

$\text{AC}$ 自动机是一种多模匹配算法，即用于解决多个模式串的匹配问题。它很好地把 $\text{kmp}$ 算法和 $\text{Trie}$ 进行了有机的结合，使得 $next$ 数组的思想在树上得到了应用。

### $\text{Preparations}$
#### $\text{kmp}$
$\text{kmp}$ 算法通过对模式串进行分析，得到 $next$ 数组，其中 $next_i = j$ 表示模式串中满足 $s_{[0,j-1]} = s_{[i - j + 1,i]}$ 的**次大**的 $j$（即排除了 $j=i+1$，后缀不能与前缀重合，否则没有意义）。如下图：

![](https://blog.monad.ga/img/post/KMP/Prefix.svg)

当模式串与文本串在某一位置失配时，朴素的做法是将模式串整体右移一位，并再次从头开始匹配，不难发现这样做了许多无用功。由上图可以知道，在已匹配的 `ABCDAB` 中，有长度相等且完全相同的前缀 `AB` 与后缀，而要使前 `AB` 与后 `AB` 重合，显然应该移动 $4$ 位，而且移动更少的位数是不可能匹配的。

类似地，由 $next$ 的定义易知，若模式串的前 $k$ 位已匹配而在第 $(k+1)$ 位失配，则可以令 $k'=next_k$，即将模式串右移 $k-next_k$ 位，此时前 $k'$ 位必然仍然匹配，可以再尝试匹配第 $(k'+1)$ 位，否则重复上述过程，直至文本串当前位匹配成功或模式串被迫从头开始匹配。

由于篇幅限制，这里不再展开阐述，更进一步的解析参见[这里](https://blog.monad.ga/2018/02/07/KMP/)。

总的来说，$\text{kmp}$ 算法的精华在于 $next$ 数组，它实现了在每次失配时充分利用模式串重叠的部分，使下一次匹配在合适的位置重新开始，而不是不加思考地直接向右移一位。

#### Trie

又被称为字典树，主要特点是利用多个字符串的公共前缀来节省空间。如下图：

![](https://blog.monad.ga/img/post/Trie/Trie_tree_example.svg)

注意 $\text{Trie}$ 中的键值保存在边上而并非结点上，这一点与很多数据结构不同。由根结点出发，到达任意被标记结点的一条路径上各边字母，就组成了一个单词。如上图代表保存了 `{car, cat, cut}`。

插入和查询时都是从根结点出发，每次从当前结点找一条表示当前字母的边，若找不到，则查询失败，插入时则要新建边和结点；找到则继续往下索引。为了区分单词与前缀，插入时要在单词对应的结点处打上标记。

由于篇幅限制，这里不再展开阐述，更进一步的解析参见[这里](https://blog.monad.ga/2018/02/12/Trie/)。

### $\text{Fail Pointer}$

将 $\text{kmp}$ 的思想放到 $\text{Trie}$ 上，就得到了 $\text{AC}$ 自动机。

首先，我们照常建立 $\text{Trie}$，将所有模式串插入其中。

之后在匹配的过程中，仍然是从根结点出发，一路向下，若某处不存在对应的字母边，则需要退而求其次，寻找相对短一些的后缀进行匹配。此时朴素的想法当然是再回到根结点开始尝试，但类比 $next$ 数组令模式串右移适当位置的效果，我们在 $\text{Trie}$ 树上添加一些有向边，使匹配转移到合理的位置继续进行下去。事实上，我们得到了一张 $\text{Trie}$ 图。

定义 $\text{Fail}$ 指针如下：若 $x$ 的 $\text{Fail}$ 指针指向 $y$，根结点到 $y$ 路径上各边字母组成的串为 $t$，则 $x$ 向上 $depth(y)$ 代祖先到 $x$ 的路径上各边字母组成的串 $s$ 与 $t$ 相等。且 $y$ 的深度尽量大。

如下图：

![](http://s11.sinaimg.cn/mw690/001L9wH0gy6SB1FHRwK1a&690)

不难发现，$\text{Fail}$ 指针的工作机制仍然是基于“后缀=前缀”，这与 $\text{kmp}$ 在本质上是相通的。

### $\text{How}$

考虑 $next$ 数组的求值过程，尽量地利用了之前已经求解出的值 。类似地，$\text{Fail}$ 指针的建立，依赖于父结点。

对于根结点的直接子结点，显然其 $\text{Fail}$ 指针只能指向根结点（对应 $next_0=next_1=1$）。

而对于更深的结点 $u$，显然应该指向深度严格小于自身的结点 $v$。对于 $u$ 的父结点 $f$ 的 $\text{Fail}$ 指针所指的 $v'$，令边 $(u, v)$ 上的字母为 $c$，若 $v'$ 有一条 $c$ 的边连向 $w$，则 $w$ 即为所求的 $v$；否则令 $v'$ 指向其 $\text{Fail}$ 指针，重复上述过程，直到找到有 $c$ 的边，或回到根结点为止。

这里就存在求解顺序的问题，因为要按深度递增求解，因此以 $\text{BFS}$ 实现。

### $\text{Code}$

此处以[模板题](https://www.luogu.org/problemnew/show/P3808)为例。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <queue>

using namespace std;

const int MAXL = 1e6 + 100;
const int SIGMA_SIZE = 30;

struct Trie {
	struct Node {
		int ch[SIGMA_SIZE];
		int v, f, d;
	} nodes[MAXL];
	int sz;
	Trie () { memset(nodes[0].ch, 0, sizeof nodes[0].ch); sz = 1; }
	int idx(char c) { return c - 'a'; }

	void insert(char *s) {
		int u = 0, n = strlen(s);
		for (int i = 0; i < n; i++) {
			int c = idx(s[i]);
			if (!nodes[u].ch[c]) {
				memset(nodes[sz].ch, 0, sizeof nodes[sz].ch);
				nodes[sz].v = 0;
				nodes[u].ch[c] = sz++;
			}
			u = nodes[u].ch[c];
		}
		++nodes[u].v;
	}

	void bfs() { //连边构 Trie 图 
		queue <int> q;
		for (; !q.empty(); q.pop());
		nodes[0].f = nodes[0].d = 0;
		for (q.push(0); !q.empty(); ) {
			int u = q.front(); q.pop();
			for (int i = 0; i < SIGMA_SIZE; i++) {
				int v = nodes[u].ch[i];
				if (!v) continue;
				nodes[v].d = nodes[u].d + 1; q.push(v);
				if (nodes[v].d == 1) nodes[v].f = 0; //特判 
				else {
					int p = nodes[u].f;
					for (; p && !nodes[p].ch[i]; p = nodes[p].f); //往回匹配，直至找到或回到根结点 
					nodes[v].f = nodes[p].ch[i];
				}
			}
		}
	}

	int match(char *s) {
		int n = strlen(s), ans = 0;
		for (int i = 0, u = 0; i < n; i++) {
			int c = idx(s[i]);
			for (; u && !nodes[u].ch[c]; u = nodes[u].f); //匹配当前字母 
			if (nodes[u].ch[c]) {
				u = nodes[u].ch[c]; //(*)
				for (int v = u; v && nodes[v].v; v = nodes[v].f) { //遇到非单词结点就可以停下了
					ans += nodes[v].v; nodes[v].v = 0; //避免重复访问 
				}
			}
		}
		return ans;
	}

	void debug_output() {
		for (int i = 0; i < sz; i++) printf("%d ", nodes[i].f); putchar('\n');
	}
} t;

char tmp[MAXL], str[MAXL];

int main(void) {
	int n; scanf("%d", &n);
	for (int i = 0; i < n; i++) {
		scanf("%s", tmp); 
		t.insert(tmp);
	}
	t.bfs();
//	t.debug_output();
	scanf("%s", str);
	printf("%d\n", t.match(str));
	return 0;
}
```

