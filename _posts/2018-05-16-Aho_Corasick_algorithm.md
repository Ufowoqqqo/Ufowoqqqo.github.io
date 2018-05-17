---
layout:     post
title:      "AC 自动机学习笔记"
subtitle:   "Let's get ACcepted"
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

### $\text{Example Problems}$

#### $\text{SMOJ }2504$

既然允许重复，则只需要考虑“是否存在”，问题变得简单了许多。

不难注意到，对于某个前缀的判定可以划分为子问题，即去掉该前缀末尾的某个元素后得到的是仍然某个更短的合法前缀。

于是用 $\texttt{bool}$ 的 $f_i$ 表示以第 $i$ 位为结尾的前缀能否分解。可以枚举子问题，尝试将 $[0,i]$ 划分为 $[0,j]$ 这个子问题和 $(j,i]$ 这个元素。特别地，要注意 $[0,i]$ 自身就是元素的情况。最终答案就是使得 $f_i=1$ 的最大的 $i+1$。

直接转移是 $O(n^2)$ 的，但这中间完全可以省去很多无用功。首先，对于当前正在求解的 $i$，只要往前找到一种合法的 $j$，就可以停止本轮了。毕竟具体的方案数不是我们需要关心的问题，具体是怎么串联的没有必要考虑。其次，设 $P$ 中最长的元素长度为 $l_{max}$，则 $j$ 的枚举范围从 $i-l_{max}$ 开始即可，这是因为每次只考虑作为一个元素添加在末尾，避免对于状态空间的重复求解，复杂度极大降低。

#### $\text{SMOJ }1766$

虽然 $K$ 可能比较大，但在计算的时候其实只有最近 $15$ 位是有用的，因为 $\text{combo}$ 串的长度不可能超过 $15$。又由于每 $1$ 位最多只有 $3$ 种可能，容易想到一种极其大力的 $\text{DP}$ 方法：即第 $1$ 维为当前长度，第 $2$ 维为 $3^{15}$，表示最后 $15$ 位的情况。转移方程是非常显然的，每次尝试添加字母后累加新增的匹配到 $\text{combo}$ 串数。

然而这个状态空间过于奢侈了，在时间和空间上都无法承受。

事实上，大部分状态是不必要的。因为只有那些能够加分的串，才有被构造的意义。这样一来就可以把第 $2$ 维改为 $\text{Trie}$ 树中的结点编号，在添加字母后索引到对应子结点而进行状态转移。但有的结点不一定存在，而构造串的过程中又不一定全程都在加分，因此在建立 $\text{AC}$ 自动机的过程中，对于不存在的子结点应指向自身 $\text{Fail}$ 指针所指结点的对应子结点。并且每个结点除了自身有的值之外，还应继承 $\text{Fail}$ 指针所指结点对应的数值。这是因为 $\text{Fail}$ 所指的对应串本身就是当前串的子串，若当前得分了则在 $\text{Fail}$ 处也必然得分。

还需要注意的细节是 $\text{DP}$ 时数组的初值。有的状态不一定合法，因此一开始只将 $(0, 0)$ 赋为 $0$，其余都赋为负数，以防无意义状态进行拓展。

#### $\text{SMOJ }1767$

一开始考虑的是 $3$ 维的状态，分别为长度、匹配的子串数和当前在 $\text{Trie}$ 中对应的结点编号。然而子串可能被重复匹配，但却不应被重复计算，即问题的关键在于“有没有”而非“多少次”。又结合相当少的串数，自然而然地就会想到应该要把第 $2$ 维改为状压，各子串的被匹配情况。

类似于上一题，转移的时候尝试在末尾添加一个字母并进行匹配，得到新状态的结点编号。各结点上记录的值也应该是一个集合，
表示其到 $\text{Fail}$ 树的根的路径上子串结点的情况。由于 $\text{Fail}$ 指针特有的性质，一旦当前串为最终串的子串，对应的 $\text{Fail}$ 串必然也是，因此计算的时候要“打包”在一起。建立 $\text{AC}$ 自动机的时候，当前结点要继承 $\text{Fail}$ 结点的值（取并集）。

最后稍微总结一下最近做的几题（特别是 $1766$ 和 $1767$），都是 $\text{AC}$ 自动机优化 $\text{DP}$，主要是对状态进行了压缩（不是进制那种），只考虑自动机中的合法状态集。并且在建立 $\text{AC}$ 自动机的时候，都要相应继承 $\text{Fail}$ 结点的一些信息。

#### $\text{SMOJ }1768$

题目所述的串 $x$ 在串 $y$ 中**出现**，等价于 $x$ 为 $y$ 的子串，即所求转化为求 $y$ 的所有子串中 $x$ 的个数。

如何考虑 $y$ 的所有子串？若强行枚举，可以发现有很多状态似乎是不会出现的，即应该排除这些冗余状态。

结合题面所述过程，不难发现这类似于从 $\text{Trie}$ 树的根结点出发，每次向下走（打字母）、向上走（按 $\texttt{B}$）或打标记（按 $\texttt{P}$）。为了便于表述，以下将串 $A$ 在 $\text{Trie}$ 树上构造后对应的末结点称为结点 $A$；同理，将根结点到 $A$ 的路径上各边字母构成的串称作串 $A$。

对于根结点到结点 $y$ 的路径上的任意结点 $u$，串 $u$ 显然是 $y$ 的前缀。

若将 $\text{AC}$ 自动机中的 $\text{Fail}$ 指针看作边，则可以得到一棵**树**（除根结点外，每个结点有且只有一个深度小于自身的父亲），不妨称之为 $\text{Fail}$ 树。由定义易知，对于  $\text{Fail}$ 树上任意结点 $u$，以其为根的子树中任意结点 $v$，都满足串 $u$ 为 串 $v$ 的后缀，自然也属于串 $v$ 的子串。

考虑从根结点出发，处于某个结点 $y$ 处。若有询问 $(x,y)$，应统计满足下述条件的结点 $p$ 的个数：

- 在 $\text{Trie}$ 上，$p$ 在根结点到 $y$ 的路径中（串 $p$ 为串 $y$ 前缀）；
- 在 $\text{Fail}$ 树上，$p$ 在以 $x$ 为根的子树中（串 $x$ 为串 $p$ 后缀）。

注意括号内的性质。事实上，这启示我们，**$x$ 为 $y$ 子串恰好可以看作 $x$ 为 $y$ 某个前缀的后缀**。需要注意的是，$p$ **不一定**为单词结点。

-----

条件 $1$ 是容易实现的，只需在 $\text{DFS}$ 的过程中对已遍历而未回溯的结点作标记即可。

如何实现条件 $2$？即，$x$ 的子树中各结点有何通性？

若对 $\text{Fail}$ 树进行 $\text{DFS}$，则以某结点为根的子树中各结点 $\text{DFS}$ 序连续。

由此，则可以对 $\text{Trie}$ 树上的结点与其在 $\text{Fail}$ 树上的时间戳进行映射，打标记是单点增加，统计时区间查询，可以用 $\text{Fenwick}$ 树 或线段树实现。由于到了具体结点才考虑自身需要处理的询问，需要在输入时将询问离线。
