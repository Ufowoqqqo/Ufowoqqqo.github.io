---
layout:     post
title:      "非旋转式 Treap 学习笔记"
date:       2018-05-26 21:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 数据结构笔记
    - Treap
---

### $\text{Introduction}$

传统的 $\text{Treap}$ 我们都已经很熟悉，其特性是永远在键值上维护 $\text{BST}$  性质，在修正值上满足堆性质，以左旋或右旋为手段，调整自身形态结构，从而达到较理想（期望树高 $\log n$）的平衡状态。然而，这也是要付出一定代价的。虽然中序遍历不会受到影响，进而仍然始终可以进行一系列的维护和查询，但由于树的父子层级关系发生了本质上的改变，导致传统 $\text{Treap}$ 无法实现可持久化。

为了解决这个瑕疵，一种新型的、无需旋转的 $\text{Treap}$ 应运而生，满足了可持久化的需要。

![](https://images2015.cnblogs.com/blog/1000791/201706/1000791-20170606085830043-1517925596.jpg)

### $\text{Symbols}$

结点 $u$ 的左子结点记为 $left(u)$，结点自身及其左子结点组合记为 $left'(u)$，右子结点记为 $right(u)$，结点自身及其右子结点组合记为 $right'(u)$。

### $\text{Basic Operations}$

下述 $2$ 种基本操作的单次复杂度均为 $O(\log n)$。

#### $\text{Merge(}a,b\text{)}$

给定以 $a,b$ 为根的 $2$ 棵 $\text{Treap}$ 且保证 $a$ 中最大元素不大于 $b$ 中最小元素，返回 $2$ 棵 $\text{Treap}$ 合并后的根。

若 $a, b$ 中有 $1$ 棵空树，则返回非空 $\text{Treap}$ 的根即可。

若均不为空，则考虑新 $\text{Treap}$ 的根为 $a$ 或 $b$，原有的另 $1$ 根则为该根的子结点。由修正值的堆性质可知，$a$ 和 $b$ 中修正值更高者为新根。具体地，若 $a$ 的修正值优先级更高，则令 $a$ 为新根，其原有左子树保持不变，右子树变为 $\text{Merge(}right(a),b\text{)}$，反之亦然。

#### $\text{Split(}u,x\text{)}$

返回将以 $u$ 为根的 $\text{Treap}$ 分裂为小于等于 $x$ 的元素组成的 $\text{Treap}$ 的根 $a$ 和大于 $x$ 的元素组成的 $\text{Treap}$ 的根 $b$ 组成的有序 $2$ 元组 $(a,b)$。

考虑结点 $u$ 自身键值 $k$ 与 $x$ 的大小关系。

若 $x<k$，则令 $\text{Split(}left(u),x\text{)}=(r_1,r_2)$，并返回 $(r_1,\text{Merge}(r_2,right'(u)))$；

否则令 $\text{Split}(right(u),x)=(r_1,r_2)$，并返回 $(\text{Merge}(left'(u),r_1),r_2)$。

### $\text{Persistence}$

在[可持久化线段树入门学习笔记](https://ufowoqqqo.github.io/2018/05/18/Chairman_Tree/)中曾经提到过，

> 可持久化数据结构有一个通性：从不修改或删除，只新增。 

基于上述原则，我们容易得到可持久化 $\text{Treap}$ 的维护方法。所谓可持久化的具体表现其实就是进入 $\text{Treap}$ 的时候借用了历史版本的根结点，参见下面的模板代码。

### $\text{Maintenance}$

#### $\text{Insert}(u,x)$

首先得到 $\text{Split}(u,x)=(r_1,r_2)$，再返回 $\text{Merge}(\text{Merge}(r_1,x),r_2))$ 即可。

#### $\text{Delete}(u,x)$

首先得到 $\text{Split}(u,x)=(r_1,r_2)$，再得到 $\text{Split}(r_1,x-1)=(r_1,r_3)$，则此时得到的 $r_1$ 为小于 $x$ 的元素组成 $\text{Treap}$ 的根，$r_2$ 则为大于 $x$ 的元素，$r_3$ 为所有等于 $x$ 的元素。若 $r_3$ 非空，令 $r_3=\text{Merge}(left(r_3),right(r_3))$ 即删除了单个 $x$。最终返回 $\text{Merge}(\text{Merge}(r_1,r_3),r_2)$ 即可。

### $\text{Code}$

```cpp
//Waring: LARGE memory.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int INF = 0x7fffffff;
const int MAXN = 5e5 + 100;

struct Node {
    Node *c[2];
    int v, s; short f;
    Node (): v(0), f(rand()), s(0) { c[0] = c[1] = NULL; }
    Node (int x): v(x), f(rand()), s(1) { c[0] = c[1] = NULL; }
    void upd() { s = (c[0] ? c[0] -> s : 0) + 1 + (c[1] ? c[1] -> s : 0); }
} nodes[MAXN * 20], *cur; //memory pool

struct Treap {
    Node *r;
    Treap (): r(NULL) {}
    
    void debug_output(Node *u) {
        if (!u) return;
        debug_output(u -> c[0]);
        printf("%d*%d ", u -> v, 1);
        debug_output(u -> c[1]);
    }
    
    Node *merge(Node *a, Node *b) {
        if (!a) return b; if (!b) return a;
        Node *t = cur++;
        if (a -> f < b -> f) { *t = *a; t -> c[1] = merge(t -> c[1], b); t -> upd(); return t; }
        else { *t = *b; t -> c[0] = merge(a, t -> c[0]); t -> upd(); return t; }
    }
    
    void split(Node *u, int n, Node *&x, Node *&y) { //try not to return <pair>
        if (!u) { x = y = NULL; return; }
        if (n < u -> v) { y = cur++; *y = *u; split(y -> c[0], n, x, y -> c[0]); y -> upd(); }
        else { x = cur++; *x = *u; split(x -> c[1], n, x -> c[1], y); x -> upd(); }
    }
    
    void ins(Node *&u, int x) {
        Node *a = NULL, *b = NULL;
        split(u, x, a, b); cur -> v = x; cur -> s = 1; Node *t = cur++;
        u = merge(merge(a, t), b);
    }
    
    void del(Node *&u, int x) {
        Node *a = NULL, *b = NULL, *c = NULL;
        split(u, x, a, b); split(a, x - 1, a, c);
        if (c) c = merge(c -> c[0], c -> c[1]); //special judge, otherwise it will get RE
        u = merge(merge(a, c), b);
    }
    
    //queries below are the same as the ones in a traditional Treap
    int rank(Node *u, int x) {
        if (!u) return 1;
        int ls = u -> c[0] ? u -> c[0] -> s : 0;
        if (x <= u -> v) return rank(u -> c[0], x); else return ls + 1 + rank(u -> c[1], x);
    }
    
    int kth(Node *u, int x) {
        if (!u || x > u -> s) return 0;
        int ls = u -> c[0] ? u -> c[0] -> s : 0;
        if (ls < x) {
            if (x == ls + 1) return u -> v; else return kth(u -> c[1], x - ls - 1);
        } else return kth(u -> c[0], x);
    }
    
    int prev(Node *u, int x, int best) {
        if (!u) return best;
        if (u -> v < x) return prev(u -> c[1], x, u -> v); else return prev(u -> c[0], x, best);
    }
    
    int succ(Node *u, int x, int best) {
        if (!u) return best;
        if (u -> v > x) return succ(u -> c[0], x, u -> v); else return succ(u -> c[1], x, best);
    }
} t[MAXN];

int main(void) {
    int n; scanf("%d", &n); cur = nodes;
    for (int i = 1; i <= n; i++) {
        int v, opt, x; scanf("%d%d%d", &v, &opt, &x); t[i].r = t[v].r;
        if (opt == 1) t[i].ins(t[i].r, x);
        if (opt == 2) t[i].del(t[i].r, x);
        if (opt == 3) printf("%d\n", t[i].rank(t[i].r, x));
        if (opt == 4) printf("%d\n", t[i].kth(t[i].r, x));
        if (opt == 5) printf("%d\n", t[i].prev(t[i].r, x, -INF));
        if (opt == 6) printf("%d\n", t[i].succ(t[i].r, x, INF));
    }
    return 0;
}
```



### $\text{References}$

- 陈立杰《可持久化数据结构研究》
- 范浩强《谈数据结构》