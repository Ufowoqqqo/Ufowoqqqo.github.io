---
layout:     post
title:      "����ʷ֌W���Pӛ"
date:       2018-06-10 20:30:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - ���ݽṹ�ʼ�
    - �����ʷ�
---

### �p��·���ʷ�

�ڸ�ِ�г��������� $1$ ���·��֮�S�o��ԃ������Ҫ��ᘌ�߅�༰�c�ࡣ���������@Ȼ��ͨ�^�� $\text{LCA}$ ��� $2$ ���c̎���� $\text{LCA}$ ���������@��}�������M��^�g�ӷ�֮���|���������뵽�ܷ���� $\text{Segment Tree}$ ֮�֮�߼������Y���M�оS�o���𰸠��϶�֮��

�҂�֪������ $\text{Segment Tree}$ ������Ҫ�����M�Ѕ^�g����֮ǰ�᠑����̖�B�m֮ $1$ �Ρ������]����֮�������Ρ�ƽ�r���f����֮��朡���ָ�����˻��ɾ����ΑB֮�䣬�丸�ӌӼ��P�S��Ȼ���ڡ�����ձ�����֮�������������֮���������Ҫͨ�^����������֮߅�ֽM���õ� $1$ �l�l朡��ʷ�֮ԭ�t���������p���ӽY�c����

���������~�Y�c $u$����**�؃���** $v$ ���x�������Ә� $\mathrm{size}$ ���֮�ӽY�c��߅ $(u, v)$ �t��**��߅**��$u$ �c���N�ӽY�c֮�B߅�t��**�p߅**��������֮��߅�B�� $1$ �𣬾͵õ���**���**���@Щ��朼��� $\text{Segment Tree}$ ���S�o֮����

![](http://s16.sinaimg.cn/large/6974c8b2gb4c1e1110f6f&690)

### �����YՓ

- ��� $u$ ֮�ӽY�c $v$���� $(u, v)$ ���p߅�r��$\mathrm{size}(v) < \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$��

  �C�����������x�����з��~�Y�c���¶��� $1$ �l��߅���������O $\mathrm{size}(v) \ge \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$���t���� $u$ ����֮��߅ $(u, v')$ ���ԣ�$\mathrm{size}(v') \ge \mathrm{size}(v) \ge \lfloor\frac{\mathrm{size}(u)}{2}\rfloor$���ɵ� $\mathrm{size}(u) \ge \mathrm{size}(v)+\mathrm{size}(v')+1 \ge \mathrm{size}(u)+1$���c���Oì�ܡ�

- ����������ã��������Ǹ��Y�c $u$���� $u$ ����֮·���ϣ��p߅�����֮�l���������^ $\log n$����ÿ���� $1$ �l�p߅��$\mathrm{size}$ ֵ�͕��p�롣

- �\����֪ $\text{Segment Tree}$ ֮���������}�s�Ƞ� $\log n$����˘���ʷ�֮�}�s�Ƞ� $O(n\log^2n)$��

- �� $\text{DFS}$ ���У���ĳ�Y�c����֮�Ә�֮�r�g�����B�m $1$ �Ρ�



### ���F

��춘���ʷ���ᘌ�߅�M�еģ�����ھS�o�Y�c���P��Ϣ�rҪ��ȡС���ɣ�������Y�c��Ϣ�ŵ��c���H�Y�c֮�B߅���M�оS�o�������Ҫ����̓���Y�c��

ͨ�^ $2$ �� $\text{DFS}$ ���F�A̎��

- �� $1?$ ��������ϸ��Y�c֮������Ϣ������ $\mathrm{depth,heavy\_son,parent,size}?$����̎����٘����
- �� $2$ �Ό�����߅�B��朣��K��ʼ�� $\text{Segment Tree}$�����w�أ�
  - ���x $\mathrm{top}(u)$ �� $u$ �������֮픶˽Y�c��̖��$w(u)$ �� $u$ �c���Y�c֮�B߅�� $\text{Segment Tree}$ �Ќ����ˡ�
  - ���������~�Y�c $u$���� $\mathrm{top}(\mathrm{heavy\_son}(u))\leftarrow\mathrm{top}(u)$��$w(\mathrm{heavy\_son}(u))$ ���r�g���ۼӡ�����ʼ�r�Й�ֵ���t�� $\text{Segment Tree}$ �������Y�c�M�І��c�޸ġ����]������ϸ�߅����B�m�������f�w $\mathrm{heavy\_son}(u)$��
  - ��̎�����N���p���� $v$���� $\mathrm{top}(v)\leftarrow v$��$w(v)$ ���r�g���ۼӡ�����ʼ�r�Й�ֵ���t�� $\text{Segment Tree}$ �������Y�c�M�І��c�޸ġ��f�w $v$��

��� $(u,v)$ ·��֮�޸Ļ�ԃ������Ҫ�����ʳ���������M��̎�����w�أ����}�������̣�ֱ�� $u=v$��

- �� $\mathrm{top}(u)=\mathrm{top}(v)$���� $u, v$ ����ͬ $1$ ����ϣ��t�� $\text{Segment Tree}$ ��̎�팦���Σ��K�Y�����β�����
- ��t�����] $\mathrm{depth}(u)\ge\mathrm{depth}(v)$���t�� $\text{Segment Tree}$ ��̎�� $[\mathrm{top}(u),u]$ �����Σ��K�� $u\leftarrow \mathrm{parent}(\mathrm{top}(u))$��

��Ҫע�⣬̎���c���r������K�K����ͬ $1$ ���̎�Y�����Ǖ�����ͬ $1$ �c���tԓ�c��δ��̎����Ҫ�~�������



### ģ��

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
    if(son[u] != -1) { //���~�Y�c����̎���؃���
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

void add_path(int u, int v, long long d) { //·���ϸ��c����� d
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

long long query_path(int u, int v) { //·���ϸ��c��֮��
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

        if(op == 3) { //�Ә��ϸ��Y�c��ֵ���� z
            scanf("%lld", &z);
            st.update(st.root, w[x], maxw[x], z % Ghastlcon);
        }

        if(op == 4) printf("%lld\n", st.query(st.root, w[x], maxw[x])); //�Ә�Y�c��ֵ֮��
    }

    return 0;
}
```

