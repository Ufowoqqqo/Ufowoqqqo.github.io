---
layout:     post
title:      "Proof of NHOI 2015_factory"
date:       2018-05-22 11:00:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 线段树
    - NHOI
---

考虑每个结点 $u=[l,r]$，令 $len_u=r-l$。维护该区间内的
$$sa_u=\sum_{i=l}^{r\uparrow} a_i$$
$$sb_u=\sum_{i=l}^{r\uparrow} b_i$$
和
$$sab_u=\sum_{i=l}^{r\uparrow} a_i\times b_i$$



不妨分别讨论两个标记 $\delta a_u$ 和 $\delta b_u$ 的下传顺序。考虑接收标记的子结点 $v$。

### 先 $a$ 后 $b$

若先下传 $\delta a_u$，则有
$$\delta a_v' = \delta a_v + \delta a_u$$
$$sa_v' = sa_v +\delta a_u\times len_v$$
$$sab_v'=sab_v+\delta a_u\times sb_v$$

再下传 $\delta b_u$，则有
$$\delta b_v=\delta b_u$$
$$sb_v=\delta b_u\times len_v$$
$$sab_v=\delta b_u\times sa_v'=\delta b_u\times (sa_v+\delta a_u\times len_v)=\delta b_u\times sa_v+\delta a_u\times sb_v$$

### 先 $b$ 后 $a$

若先下传 $\delta b_u$，则有
$$\delta b_v=\delta b_u$$
$$sb_v=\delta b_u\times len_v$$
$$sab_v=\delta b_u\times sa_v$$

再下传 $\delta a_u$，则有
$$\delta a_v'=\delta a_v+\delta a_u$$
$$sa_v'=sa_v+\delta a_u\times len_v$$
$$sab_v'=sab_v+\delta a_u\times sb_v=\delta b_u\times sa_v + \delta a_u\times sb_v$$

于是可以发现两种操作顺序后，$sab_v'$ 的值是相等的，任选其中一种方式进行维护即可。
