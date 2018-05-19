---
layout:     post
title:      "Middle(clj)"
subtitle:   "%%%%% WJMZBMR"
date:       2018-05-19 18:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 题解
    - 可持久化线段树
    - 二分答案
---

## $\text{Description}$

一个长度为 $n$ 的序列 $a$，设其排过序之后为 $b$，其中位数定义为$b[\lfloor\frac{n}{2}\rfloor]$，其中 $a,b$ 从 $0$ 开始标号。给你一个长度为 $n$ 的序列 $s$。回答 $Q$ 个这样的询问：$s$ 的左端点在 $[a,b]$ 之间，右端点在 $[c,d]$ 之间的子序列中，最大的中位数。

其中 $a<b<c<d$。位置也从 $0$ 开始标号。**我会使用一些方式强制你在线。**

## $\text{Input}$

第一行序列长度 $n$。接下来 $n$ 行按顺序给出 $a$ 中的数。

接下来一行 $Q$。然后 $Q$ 行每行 $a,b,c,d$，我们令上个询问的答案是 $x$（如果这是第一个询问则 $x=0$）。

令数组 $$q=\left\{(a+x)\mod n,(b+x)\mod n,(c+x)\mod n,(d+x)\mod n\right\}$$

将 $q$ 从小到大排序之后，令真正的要询问的 $$a=q[0],b=q[1],c=q[2],d=q[3]$$

输入保证满足条件。

第一行所谓“排过序”指的是从小到大排序！

$n\leq 20000,Q\leq25000$

## $\text{Output}$

$Q$ 行依次给出询问的答案。

## $\text{Sample Input}$

5  
170337785  
271451044  
22430280  
969056313  
206452321  
3  
3 1 0 2  
2 3 1 4  
3 1 4 0  

## $\text{Sample Output}$

271451044  
271451044  
969056313

---------------------

强行按照中位数的定义去解决，似乎需要对区间进行排序。但实际上可以相对地看待这个问题。对于长度为 $l$ 的区间，有 $\lfloor\frac{n}{2}\rfloor$ 个数比中位数小。对于给定的数 $x$，不妨将数列中大于或等于 $x$ 的数替换为 $1$，小于 $x$ 的数替换为 $-1$，则对任意区间 $[l,r]$ 求子段和 $s$，即可得到 $x$ 与区间中位数 $y$ 的相对大小关系，即若 $s\geq0$ 则 $x\leq y$，反之则 $x>y$。

假设先去除关于 $a,b,c,d$ 的区间端点限制。不显然，所求的中位数具有单调性。考虑二分答案，问题的关键在于如何快速得到关于 $mid$ 的 $-1/1$ 序列。

根据数值从小到大考虑每个数的 $-1/1$ 序列。显然最小的数（设其原下标为 $i$）的序列全为 $1$，对于次小的数，其序列有且仅有第 $i$ 位为 $-1$，其余全为 $1$。同理，每个数的 $-1/1$ 序列在自己对应的原始下标处打上 $-1$ 标记后，就得到了下一个数的序列。这符合可持久化的要求。只需离散化后按数值从小到大建立主席树即可。

对于区间限制，注意到 $[b,c]$ 必须选。在此基础上，考虑到最大化答案，也就需要使 $s$ 尽可能大。只需对左边 $[a,b)$ 部分求连续最大右子段和，对右边 $(c, d]$ 求连续最大左子段和即可。稍微注意一下边界。

时间复杂度为 $O(Q\log^2n)$，空间复杂度为 $O(n\log n)$。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

//#define debugging

const int MAXN = 2e4 + 100;

int n;

struct SegmentTree {
    struct Node {
        Node *c[2];
        int sum, lmax, rmax;
        int l, r, m;
        Node (int x, int y): sum(0), lmax(0), rmax(0), l(x), r(y), m(x + y >> 1) { c[0] = c[1] = NULL; }
        void pushup() {
            sum = c[0] -> sum + c[1] -> sum;
            lmax = max(c[0] -> lmax, c[0] -> sum + c[1] -> lmax);
            rmax = max(c[1] -> rmax, c[0] -> rmax + c[1] -> sum);
        }
    } *r[MAXN];
    SegmentTree () { memset(r, 0, sizeof r); }
    
    void make(Node *&u, int l, int r) {
        u = new Node(l, r);
        if (l == r) { u -> sum = u -> lmax = u -> rmax = 1; return; }
        make(u -> c[0], l, u -> m); make(u -> c[1], u -> m + 1, r);
        u -> pushup();
    }
    
    void update(Node *last, Node *&u, int p, int v) {
        u = new Node(last -> l, last -> r);
        if (u -> l == p && u -> r == p) { u -> sum = u -> lmax = u -> rmax = v; return; }
        int t = u -> m < p;
        update(last -> c[t], u -> c[t], p, v); u -> c[t ^ 1] = last -> c[t ^ 1];
        u -> pushup();
    }
    
    int ask0(Node *u, int l, int r) {
        if (r < u -> l || u -> r < l) return 0;
        if (l <= u -> l && u -> r <= r) return u -> sum;
        return ask0(u -> c[0], l, r) + ask0(u -> c[1], l, r);
    }
    
    int ask1(Node *u, int l, int r) {
//		if (r < u -> l || u -> r < l) return 0;
        if (l == u -> l && u -> r == r) return u -> rmax;
        if (r <= u -> m) return ask1(u -> c[0], l, r);
        else if (l > u -> m) return ask1(u -> c[1], l, r);
        else return max(ask1(u -> c[1], u -> m + 1, r), ask1(u -> c[0], l, u -> m) + ask0(u -> c[1], u -> m + 1, r));
                                         // error here: 'u -> c[0]' -> 'u'!!!!!!!!!
        //mistaken:          u -> c[0], l, u -> m
    }
    
    int ask2(Node *u, int l, int r) {
//		if (r < u -> l || u -> r < l) return 0;
        if (l == u -> l && u -> r == r) return u -> lmax;
        if (r <= u -> m) return ask2(u -> c[0], l, r);
        else if (l > u -> m) return ask2(u -> c[1], l, r);
        else return max(ask2(u -> c[0], l, u -> m), ask0(u -> c[0], l, u -> m) + ask2(u -> c[1], u -> m + 1, r));
        //mistaken:          u -> c[1], u -> m + 1, r
    }
    
    void debug_output(int id) {
        for (int i = 0; i < n; i++) printf("%d ", ask0(r[id], i, i));
    }
} t;

pair <int, int> a[MAXN];
int b[MAXN], c[MAXN];

int main(void) {
    #ifdef debugging
    freopen("Temp.in", "r", stdin);
    freopen("Temp.out", "w", stdout);
    #endif
    scanf("%d", &n);
    for (int i = 0; i < n; i++) { scanf("%d", &a[i].first); a[i].second = i; }
    t.make(t.r[0], 0, n - 1);
    sort(a, a + n);
    for (int i = 0; i + 1 < n; i++) {
//		b[a[i].second] = i; c[i] = a[i].first;
        t.update(t.r[i], t.r[i + 1], a[i].second, -1);
    }
    
    /*
    for (int i = 0; i < n; i++) {
        printf("the %d-th tree(b[%d] = %d):", i, i, b[i]);
        for (int j = 0; j < n; j++) printf("%d ", t.ask0(t.r[b[i]], j, j));
        putchar('\n');
    }
    */
    
    int Q, lastans = 0; scanf("%d", &Q);
    for (int i = 0; i < Q; i++) {
        int tmp[5];
        for (int i = 0; i < 4; i++) { scanf("%d", &tmp[i]); (tmp[i] += lastans) %= n; }
        sort(tmp, tmp + 4);
        int l = 0, r = n - 1, id; //[l, r]
        for (; l <= r; ) {
//			printf("%d %d\n", l, r);
            int mid = l + r >> 1;
//			printf("d[%d] = %d\n", mid, d[mid]);
            #ifdef debugging
            printf("mid = %d(%d), t[mid] = { ", mid, a[mid].first);  t.debug_output(mid); puts("}");
            #endif
            int res0 = tmp[1] + 1 < tmp[2] ? t.ask0(t.r[mid], tmp[1] + 1, tmp[2] - 1) : 0;
            int res1 = t.ask1(t.r[mid], tmp[0], tmp[1]);
            int res2 = t.ask2(t.r[mid], tmp[2], tmp[3]);
            #ifdef debugging
            printf("There are %d number(s) greater than or equal to %d (in general) in the range[%d, %d].\n", res0, c[mid], tmp[1], tmp[2]);
            printf("The maximum sum going left from %d is %d\n", tmp[1], res1);
            printf("The maximum sum going right from %d is %d\n",tmp[2], res2);
            #endif
            if (res0 + res1 + res2 >= 0) { l = mid + 1; id = mid; } else r = mid - 1;
        }
//		printf("%d\n", l);
        printf("%d\n", lastans = a[id].first);
    }
    return 0;
}
```
