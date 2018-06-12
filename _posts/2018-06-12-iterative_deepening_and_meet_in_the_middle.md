---
layout:     post
title:      "迭代加深及 Meet in the Middle 學習筆記"
date:       2018-06-12 12:30:00
author:     "Ufowoqqqo"
header-img: "img/solution_bg.png"
mathjax:    true
catalog:    true
tags:
    - 算法笔记
    - 搜索
---


#### 迭代加深

衆所周知，$\text{DFS}$ 的順序是儘量訪問搜索子樹的葉子結點，再回溯去訪問搜索樹的其他分支。這就可能會導致 $1$ 種可怕的浪費：當合法的狀態空間可能很大，但最終答案的深度卻並不大時，在其餘搜索子樹上的遍歷就是無用的。

爲了避免這種情況，$1$ 種可行的解決方案當然是改 $\text{DFS}$ 爲 $\text{BFS}$，但更多情況下由於狀態的龐大，沒有辦法用 $\text{BFS}$ 解決問題。這種情況下就可以藉助 $\text{BFS}$ 的思想來改造 $\text{DFS}$，也就是迭代加深搜索。

迭代加深其實很好理解。從字面意思上來看，「迭代」意味着在之前的基礎上進行擴展，加深則顧名思義，是逐步增加求解的深度。以下 $2$ 幅圖很好地說明了常規 $\text{DFS}$ 與迭代加深的異同之處。

![Normal DFS](/img/0612_normal_dfs.png)

---

![ID-DFS](/img/0612_id_dfs.png)

實現上與常規 $\text{DFS}$ 並不太多不同。只是從小至大限制搜索深度，若在當前深度下搜不到答案，就把深度限制增加，重新進行 $1$ 次搜索，這就是**迭代加深**思想。

有 $1$ 個問題。當我們將深度限制爲 $lim$ 時，得到的搜索樹上深度小於 $lim$ 的結點事實上都已經在深度限制更小的時候被考察過了。看起來似乎造成了重複。

然而，$1$ 般地，隨着搜索深度的增加，搜索樹結點數目也會快速（例如指數級別地）增長。與之相比，這點重複搜索不足爲道。

當然，迭代加深並不意味着將當前深度的**所有**結點都考慮。對於常規 $\text{DFS}$ 中的剪枝策略，迭代加深 $1$ 樣適用。

**例題**：$\texttt{POJ2248}$

給定不大於 $100$ 的正整數 $n$，按以下要求構造正整數序列 $x$，

- $x_1=1$
- $x_m=n$
- 序列各元素**嚴格**單調遞增
- $\forall 1<k\le m$ 都存在 $2$ 個整數 $i$ 和 $j$ 使得 $x_k=x_i+x_j$，其中 $1\le i, j<k$ 且允許 $i=j$
- 在滿足上述條件的所有序列中，找出長度最小的任意 $1$ 種方案

基本的搜索框架是顯然的。首位填好 $1$，之後每次在前面的數中選取 $2$ 個數 $a$ 和 $b$（允許相同），若 $a+b$ 大於所填的前 $1$ 位且不大於 $n$ 則嘗試將其填在當前位上。

考慮剪枝及優化。

首先搜索順序上，爲了儘快得到 $n$，應該優先填大的數。這樣最初找到的解就是長度最短的。

還可以排除等效冗餘，即若存在 $x_i+x_j=x_k+x_l$，不應將本質相同的和考慮多次。考慮到數值較小，用數組計數的方法判重即可。

經過探索可以發現最終答案長度不會太大。然而，每 $1$ 步的分支是驚人的。假設不存在重複的和，則對於已填了 $k$  個數，當前位有 $\frac{n(n+1)}{2}$ 種填法，意味着搜索樹的結點數目會以平方級別速率增長。

這 $2$ 個條件啓示我們採取迭代加深的方法搜索。實際應用的效果很好。

本題需要稍微注意的是「停止搜索」的方法。對於 $1$ 組數據，可以在找到解並輸出後直接 `exit(0)`，但對於多組數據的情況，可以令 `dfs()` 返回 $1$ 個 $\texttt{bool}$ 值，表示是否找到解，以便在子樹中得到解時及時回溯。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXL = 15;
const int MAXM = 200;

int m, n, ans[MAXL];
bool f[MAXM];

bool dfs(int k) { //是否找到當前數據的解
    if(k == n) {
        if(ans[n] == m) {
            for(int i = 1; i <= n; i++) printf("%d ", ans[i]);
            return true;
        }
        return false;
    }

    memset(f, false, sizeof f);

    for(int i = 1; i <= k; i++) for(int j = 1; j <= k; j++) f[ans[i] + ans[j]] = true;

    for(int i = m; ans[k] < i; i--) if(f[i]) {
            ans[k + 1] = i;

            if(dfs(k + 1)) return true;
        }

    return false;
}

int main(void) {
    for(ans[1] = 1; ~scanf("%d", &m) && m;) for(n = 1; !dfs(1); n++); //n 爲搜索深度
    return 0;
}
```



#### $\text{Meet in the Middle}$

有 $1$ 類搜索問題，「初態」和「終態」都是確定的。這種情況下，可以從 $2$ 端同時出發，到中間匯合。

![](/img/0612_normal_dfs1.png)

---

![](/img/0612_meet_in_the_middle.png)



不難理解，若原複雜度爲 $n^m$，則 $\text{Meet in the Middle}$ 可將複雜度降至 $2\times n^{\frac{m}{2}}$。

然而，知道了終態就 $1$ 定能 $\text{Meet in the Middle}$ 嗎？未必。有 $1$ 點要求，即逆向搜索同樣能夠覆蓋問題的所有狀態空間。換言之，假如我們採用樸素的 $\text{DFS}$ 方法，交換初態和終態之後的效果應該與原來等價。這樣，才能夠保證雙方搜索到中間時，可以交會、組合得到最終的答案。

**例題**：$\texttt{TYVJ1340}$

大體積的揹包問題，使用常規的 $\text{DP}$ 方法將會 $\text{TLE+MLE}$。

考慮搜索。直接枚舉「選或不選」的複雜度爲 $O(2^n)$，無法承受。

顯然可以進行可行性剪枝——若當前總和超過 $W$ 則放棄。但最壞情況下複雜度不變。

採用 $\text{Meet in the Middle}$ 的思想。將所有物品平均分成 $2$ 半（若總數爲奇數則分成數量只相差 $1$ 的 $2$ 半），則最終方案可視爲在左右部分中各選取 $1$ 些物品（不選也算），分開搜索。

考慮對於右部分的某種選取方案，總和爲 $w$，則與之搭配的最理想情況當然是在左部分找到總和爲 $W-w$ 的方案。若找不到，則在所有總和小於 $W-w$ 的方案中，找盡可能總和大的方案。顯然所求具有單調性，因此可以將左部分的所有總和排序，對於右部分的每種方案在其中 $2$ 分查找進行搭配。此處有 $1$ 個小優化是將左部分中總和相同的方案去重（可以用 `std::unique()`，此處不具體展開介紹）。

直接這樣做的理論複雜度是 $O(2^{\frac{n}{2}}\times\log 2^{\frac{n}{2}})=O(N\times 2^{\frac{N}{2}})$。但我們還可以考慮對搜索順序進行優化，在分組前先把物品按照重量降序排序，從而減少左部分方案數。

但我在最初做這題的時候犯了個很嚴重的錯誤。出於習慣，我用 `for `循環進行二進制枚舉，而不是以遞歸的形式。這就導致所謂的 `break` 其實只是停止對於搜索樹上**當前結點**的考慮，但對**當前子樹**沒有放棄。這是需要警惕的。另外需要稍微注意的是涉及重量的變量需採用 `unsigned int` 或範圍更大的類型儲存，否則可能在相加之和超過 $2^{31}$ 後溢出。

```cpp
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>

using namespace std;

const int MAXN = 48;

unsigned W, G[MAXN], f[1 << (MAXN >> 1)], ans;
int N, half, cnt;

void dfs0(int k, unsigned s) {
    if(k == half) {
        f[cnt++] = s;
        return;
    }

    dfs0(k + 1, s);
    if(s + G[k] <= W) dfs0(k + 1, s + G[k]);
}

void dfs1(int k, unsigned s) {
    if(k == N) {
        int l = 0, r = cnt;
        unsigned k = W - s; //[l, r)

        for(; l + 1 < r;) {
            int mid = l + r >> 1;
            if(f[mid] <= k) l = mid;
            else r = mid;
        }

        ans = max(ans, s + f[l]);
        return;
    }

    dfs1(k + 1, s);
    if(s + G[k] <= W) dfs1(k + 1, s + G[k]);
}

int main(void) {
    scanf("%d%d", &W, &N);
    for(int i = 0; i < N; i++) scanf("%d", &G[i]);

    sort(G, G + N, greater<unsigned>());
    half = (N >> 1) + 1; //[0, half), [half, N)
    dfs0(0, 0);
    sort(f, f + cnt);
    cnt = unique(f, f + cnt) - f;
    dfs1(half, 0);
    printf("%d\n", ans);
    return 0;
}
```

