---
title: 第 2 週
createTime: 2026/07/15 12:52:00
permalink: /ja/csdiy/cse291a-week2/
---

> 繰り返す孤独な夜に　願うのは微かな温もり  
> 惹かれ合う惑星（ほし）のように　廻り逢う瞬間（とき）　夢視（み）て

第 2 週は、先週に続いて計算グラフと自動微分を扱い、そこから計算最適化へ進みます。最後に、行列積演算のタイル分割による最適化にも少し触れます。自動微分の要点は、逆伝播の処理自体も計算グラフ（中間表現）へ組み込み、深層学習の意味論と具体的な計算処理を分離することです。これにより、その後のシステムレベルの最適化が容易になります。

この講義で扱う計算最適化は、次の 4 つに分けられます。

- 演算子最適化：さまざまな計算要件とハードウェアに対して、高性能な演算子（GPU では kernel と呼びます）を書く方法。
- グラフレベル最適化：演算子融合、定数畳み込み、デッドコード除去など。
- ランタイム環境：計算のスケジューリングやメモリ管理など。
- 複数デバイスでの並列化：10,000 台を超えるデバイスを管理し、共同で計算タスクを実行します。

最適化の中心は、メモリの読み書きにかかるコストを減らし、プロセッサの演算能力を使い切ることです。  
現代の ML フレームワークは一般に、tensor を==ストライド形式==で表します。各次元 $i$ について $\mathrm{shape}[i]$ と $\mathrm{stride}[i]$ を記録し、それぞれインデックスの範囲と、その次元のインデックスが 1 増えたときにメモリアドレスがどれだけ進むかを表します。これにより、多くの tensor 操作を shape と stride の変更だけで実現し、コピーを避けられます。もちろん、実際の計算が連続したメモリアクセスを必要とする場合は、必要に応じて連続領域へコピーできます。

$$A[i,j]=A.\mathrm{data}[\mathrm{offset}+i\cdot A.\mathrm{stride}[0]+j\cdot A.\mathrm{stride}[1]]$$

単純化したメモリ階層モデルを考えます。行列 A、B、C は DRAM に置き、計算は高速だが容量の小さいレジスタ上で行います。

```python
def matmul(A: Float[Tensor, "n/v1 n v1", "DRAM"],
           B: Float[Tensor, "n/v2 n v2", "DRAM"],
           C: Float[Tensor, "n/v1 n/v2 v1 v2", "DRAM"]):
  for i in range(n/v1):
    for j in range(n/v2):
      c: Float[Tensor, "v1 v2", "reg"] = 0
      for k in range(n):
        a: Float[Tensor, "v1", "reg"] = A[i, k]
        b: Float[Tensor, "v2", "reg"] = B[j, k]
        c += outer(a, b)
      C[i, j] = c
```

A と B の読み取り回数はそれぞれ $n^3/v_2$ と $n^3/v_1$ になり、C の書き込み回数は $n^2$ のままです。  
読み取り回数が減るのは、1 回の計算で B の $v_2$ 個の要素が A の 1 回の読み取りを共有し、A の読み取り回数が $v_2$ 分の 1 になるためです。

<p style="text-align: center;">
  <img src="../../csdiy/hasumi.svg" alt="hasumi.svg" style="height: 300px; width: auto;">
</p>

この手法を==レジスタタイリング==と呼びます。メモリ階層が増えれば、それに応じて多段のタイリングを行えます。

## PA1

PA1 はウォームアップ課題で、Python を使って計算グラフの定義と自動微分に慣れれば十分です。  
少し難しいのは Softmax と LayerNorm の逆伝播なので、ここに記録しておきます。

### Softmax

$y_i:=\mathrm{softmax}(x)_i=\mathrm e^{x_i}/\sum_j\mathrm e^{x_j}$、$g_i:=\partial\mathcal L/\partial y_i$ が既知であるとし、$\partial\mathcal L/\partial x_i$ を求めます。

微分則から $\partial y_j/\partial x_i=y_j(\mathbf 1_{\{i=j\}}-y_i)$ なので、
$$\frac{\partial\mathcal L}{\partial x_i}=\sum_{j=1}^N g_j\frac{\partial y_j}{\partial x_i}=y_i\left(g_i-\sum_{j=1}^N g_jy_j\right).$$

### LayerNorm

$\mu:=\frac 1N\sum_i x_i$、$\hat x_i:=x_i-\mu$、$\sigma^2:=\frac 1N\sum_i\hat x_i^2$、$s:=\sqrt{\sigma^2+\varepsilon}$、$y_i:=\hat x_i/s$ とします。  
$g_i:=\partial\mathcal L/\partial y_i$ が既知であるとし、$\partial\mathcal L/\partial x_i$ を求めます。

これは「手動自動微分」のようなものです。
$$
\begin{aligned}
\frac{\partial\sigma^2}{\partial x_i}&=\frac 2N\sum_{j=1}^N\hat x_j\frac{\partial\hat x_j}{\partial x_i}=\frac{2\hat x_i}N, \\
\frac{\partial s}{\partial x_i}&=\frac 1{2s}\frac{\partial\sigma^2}{\partial x_i}=\frac{\hat x_i}{Ns}, \\
\frac{\partial y_j}{\partial x_i}&=\frac 1s\frac{\partial\hat x_j}{\partial x_i}-\frac{\hat x_j}{s^2}\frac{\partial s}{\partial x_i}=\frac 1s\left(\mathbf 1_{\{i=j\}}-\frac 1N-\frac{y_iy_j}{N}\right), \\
\frac{\partial\mathcal L}{\partial x_i}&=\sum_{j=1}^N g_j\frac{\partial y_j}{\partial x_i}=\frac 1s(g_i-\mathrm{mean}(g)-y_i\cdot\mathrm{mean}(gy)).
\end{aligned}
$$
