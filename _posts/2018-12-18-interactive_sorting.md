---
layout:     post
title:      "交互式排序初探"
subtitle:   "Interactive Sorting"
date:       2018-12-18 21:45:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
tags:
    - 题解
    - 非传统题
---

题目在[这里](https://atcoder.jp/contests/practice/tasks/practice_2)。

对于子任务 $1$，进行 $N^2$ 次比较即可。

对于子任务 $2$，可以考虑归并排序，可以用略低于 $N\log N$ 次比较完成。

子任务 $3$ 比较有趣，大概是小学奥数题，解析如下。

> Compare A to B and C to D. WLOG, suppose A>B and C>D. Compare A to C. WLOG, suppose A>C. Sort E into A-C-D. This can be done with two comparisons. Sort B into {E,C,D}. This can be done with two comparisons, for a total of seven.

---

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#define N 30
using namespace std;

char a[N], b[N];

bool cmp(char x, char y)
{
    char c[2];

    printf("? %c %c\n", x, y);
    fflush(stdout);
    scanf("%s", c);

    return !strcmp(c, "<");
}

void merge(int l, int m, int r)
{
    int i, j, k;

    // printf("merge(l = %d, m = %d, r = %d)\n", l, m, r);

    for(i = k = l, j = m + 1; i <= m || j <= r; )
    {
        if(m < i)
            b[k ++] = a[j ++];
        if(r < j)
            b[k ++] = a[i ++];
        if(i <= m && j <= r)
            b[k ++] = (cmp(a[i], a[j]) ? a[i ++] : a[j ++]);
    }

    for(i = l; i <= r; i ++)
        a[i] = b[i];

    return;
}

void mergesort(int l, int r)
{
    int m;

    if(l == r)
        return;

    // printf("mergesort(l = %d, r = %d)\n", l, r);

    m = (l + r) >> 1;
    mergesort(l    , m);
    mergesort(m + 1, r);
    merge(l, m, r);

    return;
}

int main(void)
{
    int n, q;
    int i;

    scanf("%d %d", &n, &q);
    for(i = 0; i < n; i ++)
        a[i] = 'A' + i;

    if(n == 26)
        mergesort(0, n - 1);
    else
    {
        if(cmp(a[1], a[0]))
            swap(a[0], a[1]);
        if(cmp(a[3], a[2]))
            swap(a[2], a[3]);
        if(cmp(a[2], a[0]))
        {
            swap(a[0], a[2]);
            swap(a[1], a[3]);
        }

        if(cmp(a[2], a[4]))
        {
            if(cmp(a[4], a[3]))
                swap(a[3], a[4]);
            if(cmp(a[1], a[3]))
            {
                if(cmp(a[2], a[1]))
                    swap(a[1], a[2]);
            }
            else
            {
                swap(a[1], a[2]);
                swap(a[2], a[3]);
                if(cmp(a[4], a[3]))
                    swap(a[3], a[4]);
            }
        }
        else
        {
            if(cmp(a[0], a[4]))
            {
                if(cmp(a[1], a[2]))
                {
                    swap(a[3], a[4]);
                    swap(a[2], a[3]);
                    if(cmp(a[2], a[1]))
                        swap(a[1], a[2]);
                }
                else
                {
                    swap(a[1], a[4]);
                    if(cmp(a[4], a[3]))
                        swap(a[3], a[4]);
                }
            }
            else
            {
                swap(a[0], a[4]);
                if(cmp(a[1], a[2]))
                {
                    swap(a[3], a[4]);
                    swap(a[2], a[3]);
                    swap(a[1], a[2]);
                }
                else
                {
                    swap(a[1], a[4]);
                    if(cmp(a[4], a[3]))
                        swap(a[3], a[4]);
                }
            }
        }
    }

    printf("! %s\n", a);
    fflush(stdout);

    return 0;
}
```

