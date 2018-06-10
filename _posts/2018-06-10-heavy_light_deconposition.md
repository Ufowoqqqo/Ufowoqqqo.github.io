---
layout:     post
title:      "樹鏈剖分學習筆記"
date:       2018-06-10 20:30:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 数据结构笔记
    - 树链剖分
---


### 輕重路徑剖分

在競賽中常常會遇到 $1$ 類樹上路徑之維護及詢問，主要爲針對邊權及點權。暴力做法顯然爲通過求 $\text{LCA}$ 後從 $2$ 個點處逐步向 $\text{LCA}$ 靠近。但這類問題往往都滿足區間加法之性質，很容易聯想到能否藉助 $\text{Segment Tree}$ 之類之高級數據結構進行維護，答案爲肯定之。

我們知道，以 $\text{Segment Tree}$ 爲例，要用其進行區間操作之前提爲「編號連續之 $1$ 段」。考慮樹上之對應情形。平時所說樹上之「鏈」所指即爲退化成線性形態之樹，其父子層級關係依然存在。若於普遍情形之樹上引入相類似之概念，首先需要通過對樹上相鄰之邊分組，得到 $1$ 條條鏈。剖分之原則即根據「輕重子結點」。

對於任意非葉結點 $u$，其**重兒子** $v$ 定義爲對應子樹 $\mathrm{size}$ 最大之子結點，邊 $(u, v)$ 則爲**重邊**，$u$ 與其餘子結點之連邊則爲**輕邊**。將相鄰之重邊連在 $1$ 起，就得到了**重鏈**。這些重鏈即爲 $\text{Segment Tree}$ 所維護之對象。

![](http://s16.sinaimg.cn/large/6974c8b2gb4c1e1110f6f&690)

### 定理及結論

- 對於 $u$ 之子結點 $v$，當 $(u, v)$ 爲輕邊時，$\mathrm{size}(v) < \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$。

  證明：根據定義，所有非葉結點往下都有 $1$ 條重邊。不妨假設 $\mathrm{size}(v) \ge \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$，則对于 $u$ 向下之重邊 $(u, v')$ 而言，$\mathrm{size}(v') \ge \mathrm{size}(v) \ge \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$，可得 $\mathrm{size}(u) \ge \mathrm{size}(v)+\mathrm{size}(v')+1 \ge \mathrm{size}(u)+1$，與假設矛盾。

- 由上述定理得，對於任意非根結點 $u$，在 $u$ 到根之路徑上，輕邊及重鏈之條數均不超過 $\log n$，因爲每遇到 $1$ 條輕邊，$\mathrm{size}$ 值就會減半。

- 衆所周知 $\text{Segment Tree}$ 之基本操作複雜度爲 $\log n$，因此樹鏈剖分之複雜度爲 $O(n\log^2n)$。

- 在 $\text{DFS}$ 序中，以某結點爲根之子樹之時間戳爲連續 $1$ 段。



### 實現

由於樹鏈剖分是針對邊進行的，因此在維護結點相關信息時要採取小技巧，將自身結點信息放到與父親結點之連邊上進行維護，因此需要新增虛根結點。

通過 $2$ 次 $\text{DFS}$ 實現預處理。

- 第 $1?$ 次求出樹上各結點之對應信息，包括 $\mathrm{depth,heavy\_son,parent,size}?$。此處不再贅述。
- 第 $2$ 次將各重邊連成鏈，並初始化 $\text{Segment Tree}$。具體地，
  - 定義 $\mathrm{top}(u)$ 爲 $u$ 所在重鏈之頂端結點編號，$w(u)$ 爲 $u$ 與父結點之連邊在 $\text{Segment Tree}$ 中對應下標。
  - 對於任意非葉結點 $u$，令 $\mathrm{top}(\mathrm{heavy\_son}(u))\leftarrow\mathrm{top}(u)$，$w(\mathrm{heavy\_son}(u))$ 爲時間戳累加。若初始時有權值，則在 $\text{Segment Tree}$ 上相應結點進行單點修改。考慮到重鏈上各邊下標須連續，立即遞歸 $\mathrm{heavy\_son}(u)$。
  - 再處理其餘各輕兒子 $v$。令 $\mathrm{top}(v)\leftarrow v$，$w(v)$ 爲時間戳累加。若初始時有權值，則在 $\text{Segment Tree}$ 上相應結點進行單點修改。遞歸 $v$。

對於 $(u,v)$ 路徑之修改或詢問，需要將其剖成若干重鏈進行處理。具體地，重複如下流程，直至 $u=v$。

- 若 $\mathrm{top}(u)=\mathrm{top}(v)$，即 $u, v$ 已在同 $1$ 重鏈上，則在 $\text{Segment Tree}$ 上處理對應段，並結束本次操作；
- 否則，考慮 $\mathrm{depth}(u)\ge\mathrm{depth}(v)$，則在 $\text{Segment Tree}$ 上處理 $[\mathrm{top}(u),u]$ 對應段，並令 $u\leftarrow \mathrm{parent}(\mathrm{top}(u))$。

需要注意，處理點權時，若最終並非在同 $1$ 重鏈處結束而是會聚至同 $1$ 點，則該點尚未被處理，需要額外操作。



### 模板

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

#define lc ch[0]
#define rc ch[1]

const int MAXN = 1e5 + 100;

struct Edge {
    int v;
    Edge *nxt;
} e[MAXN << 1], *cur, *h[MAXN];

int N, dep[MAXN], fa[MAXN], siz[MAXN], son[MAXN];
long long Ghastlcon, A[MAXN];

void dfs0(int u, int p) {
    dep[u] = dep[p] + 1;
    fa[u] = p;
    siz[u] = 1;
    son[u] = -1;

    for(Edge *i = h[u]; i; i = i -> nxt)
        if(i -> v != p) {
            dfs0(i -> v, u);
            siz[u] += siz[i -> v];

            if(son[u] == -1 || siz[i -> v] > siz[son[u]]) son[u] = i -> v;
        }
}

struct SegmentTree {
    struct Node {
        Node *ch[2];
        long long v, delta;
        int l, r, m, len;
        Node(int l, int r): v(0LL), delta(0LL), l(l), r(r), m(l + r >> 1), len(r - l + 1) {
            lc = rc = NULL;
        }
        void maintain() {
            v = lc -> v + rc -> v;
        }
        void pushdown() {
            if(delta) {
                (lc -> v += delta * lc -> len) %= Ghastlcon;
                (lc -> delta += delta) %= Ghastlcon;
                (rc -> v += delta * rc -> len) %= Ghastlcon;
                (rc -> delta += delta) %= Ghastlcon;
                delta = 0;
            }
        }
    } *root;
    SegmentTree(): root(NULL) {}

    void build(Node *&u, int l, int r) {
        u = new Node(l, r);

        if(l == r) return;

        build(u -> lc, l, u -> m);
        build(u -> rc, u -> m + 1, r);
    }

    void update(Node *&u, int l, int r, long long v) {
        if(r < u -> l || u -> r < l) return;

        if(l <= u -> l && u -> r <= r) {
            (u -> v += v * u -> len) %= Ghastlcon;
            (u -> delta += v) %= Ghastlcon;
            return;
        }

        u -> pushdown();
        update(u -> lc, l, r, v);
        update(u -> rc, l, r, v);
        u -> maintain();
    }

    long long query(Node *u, int l, int r) {
        if(r < u -> l || u -> r < l) return 0;

        if(l <= u -> l && u -> r <= r) return u -> v;

        u -> pushdown();
        return (query(u -> lc, l, r) + query(u -> rc, l, r)) % Ghastlcon;
    }
} st;
int cnt_segnode;

int top[MAXN], w[MAXN], maxw[MAXN];

void dfs1(int u) {
    if(son[u] != -1) { //非葉結點優先處理重兒子
        top[son[u]] = top[u];
        w[son[u]] = ++cnt_segnode;
        st.update(st.root, cnt_segnode, cnt_segnode, A[son[u]]);
        dfs1(son[u]);
    }

    for(Edge *i = h[u]; i; i = i -> nxt)
        if(i -> v != fa[u] && i -> v != son[u]) {
            top[i -> v] = i -> v;
            w[i -> v] = ++cnt_segnode;
            st.update(st.root, cnt_segnode, cnt_segnode, A[i -> v]);
            dfs1(i -> v);
        }

    maxw[u] = cnt_segnode;
}

void add_path(int u, int v, long long d) { //路徑上各點權加上 d
    for(; u != v;) {
        int f1 = top[u], f2 = top[v];

        if(f1 != f2) {
            if(dep[f1] < dep[f2]) {
                swap(f1, f2);
                swap(u, v);
            }

            st.update(st.root, w[f1], w[u], d);
            u = fa[f1];
        } else {
            if(dep[u] < dep[v]) swap(u, v);

            st.update(st.root, w[v], w[u], d);
            return;
        }
    }

    st.update(st.root, w[u], w[u], d);
}

long long query_path(int u, int v) { //路徑上各點權之和
    long long s = 0;

    for(; u != v;) {
        int f1 = top[u], f2 = top[v];

        if(f1 != f2) {
            if(dep[f1] < dep[f2]) {
                swap(f1, f2);
                swap(u, v);
            }

            (s += st.query(st.root, w[f1], w[u])) %= Ghastlcon;
            u = fa[f1];
        } else {
            if(dep[u] < dep[v]) swap(u, v);

            (s += st.query(st.root, w[v], w[u])) %= Ghastlcon;
            return s;
        }
    }

    (s += st.query(st.root, w[u], w[u])) %= Ghastlcon;
    return s;
}

int main(void) {
    int M, R;
    scanf("%d%d%d%lld", &N, &M, &R, &Ghastlcon);

    for(int i = 1; i <= N; i++) {
        scanf("%lld", &A[i]);
        A[i] %= Ghastlcon;
    }

    memset(h, 0, sizeof h);
    cur = e;

    for(int i = 1; i < N; i++) {
        int x, y;
        scanf("%d%d", &x, &y);
        *cur = (Edge) {
            y, h[x]
        };
        h[x] = cur++;
        *cur = (Edge) {
            x, h[y]
        };
        h[y] = cur++;
    }

    dfs0(R, 0);
    st.build(st.root, 1, N);
    son[0] = R;
    dfs1(0);

    for(; M--;) {
        int op, x, y;
        long long z;
        scanf("%d%d", &op, &x);

        if(op == 1) {
            scanf("%d%lld", &y, &z);
            add_path(x, y, z % Ghastlcon);
        }

        if(op == 2) {
            scanf("%d", &y);
            printf("%lld\n", query_path(x, y));
        }

        if(op == 3) { //子樹上各結點權值加上 z
            scanf("%lld", &z);
            st.update(st.root, w[x], maxw[x], z % Ghastlcon);
        }

        if(op == 4) printf("%lld\n", st.query(st.root, w[x], maxw[x])); //子樹結點權值之和
    }

    return 0;
}
```

