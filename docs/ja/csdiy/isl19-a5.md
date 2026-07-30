---
title: 代数恒等式
createTime: 2026/06/26 17:19:45
permalink: /ja/csdiy/isl19-a5/
---

> どんな概念も、自分自身で発見することが最良の学び方である。

$x_1,\ldots,x_n$ を相異なる実数とするとき、次を証明せよ。
$$\sum_{i=1}^n\prod_{j\ne i}\frac{1-x_ix_j}{x_i-x_j}=n\bmod 2.$$

## 解答

これは 2019 年 IMO Shortlist の代数第 5 問です。[AoPS の議論](https://artofproblemsolving.com/community/c6h2279006p17828803)も参照してください。

$x_1,\ldots,x_n$ の中に $\pm 1$ がない場合だけを示せば十分です。それ以外の場合は、連続性を使って極限を取ればよいからです。次の有理関数 $f:\mathbb C\to\mathbb C$ を考えます。
$$f(z):=\frac{\prod_{i=1}^n(1-x_iz)}{(1-z^2)\prod_{i=1}^n(z-x_i)}.$$
$f$ の特異点 $\pm 1,x_1,\ldots,x_n$ はすべて 1 位の極であり、無限遠での極限は $0$ です。各極と無限遠点における留数を計算すると、
$$
\begin{aligned}
\mathrm{Res}(f,1)&=\lim_{z\to 1}(z-1)f(z)=-\frac{\prod_{i=1}^n(1-x_i)}{2\prod_{i=1}^n(1-x_i)}=-\frac12,\\
\mathrm{Res}(f,-1)&=\lim_{z\to -1}(z+1)f(z)=\frac{\prod_{i=1}^n(1+x_i)}{2(-1)^n\prod_{i=1}^n(1+x_i)}=\frac{(-1)^n}{2},\\
\mathrm{Res}(f,x_i)&=\lim_{z\to x_i}(z-x_i)f(z)=\prod_{j\ne i}\frac{1-x_ix_j}{x_i-x_j},\\
\mathrm{Res}(f,\infty)&=-\lim_{|z|\to\infty}zf(z)=0.
\end{aligned}
$$
[留数定理](https://en.wikipedia.org/wiki/Residue_theorem)により、すべての特異点と無限遠点における留数の和は $0$ です。したがって、
$$\mathrm{LHS}=\sum_{i=1}^n\mathrm{Res}(f,x_i)=\frac{1-(-1)^n}{2}=n\bmod 2.$$

## どういうことか

留数定理をまだ習っていなければ、どうすればよいのか。留数定理の証明から書けばよいのです😃

<p style="text-align: center;">
  <img src="../../csdiy/yua.jpg" alt="yua.jpg" style="height: 250px; width: auto;">
</p>

微分の定義を振り返ります。$f:\mathbb C\to\mathbb C$ が $a$ で微分可能であるとは、次の極限が存在することです。
$$f'(a):=\lim\limits_{z\to a}\frac{f(z)-f(a)}{z-a}$$

- $f'(a)\ne 0$ なら、$z-a$ を一定の倍率で拡大・縮小し、回転させたものが $f(z)-f(a)$ になります。このとき $f$ は $a$ で[等角](https://en.wikipedia.org/wiki/Conformal_map)であるといいます。
- $f'(a)=0$ なら、$f(z)-f(a)$ は $z-a$ よりはるかに小さくなります。このとき $a$ を $f$ の[臨界点](https://en.wikipedia.org/wiki/Critical_point_(mathematics))といいます。

特に、実軸方向と虚軸方向の変化量は、写像後も直交します。これが[コーシー・リーマンの方程式](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations)を与えます。  
$f$ が $x+\mathrm iy$ を $u+\mathrm iv$ へ写すとき、$\partial_y v=\partial_x u$、$\partial_y u=-\partial_x v$ です。

次に、[複素線積分](https://en.wikipedia.org/wiki/Contour_integration)を振り返ります。長さをもつ曲線 $\gamma$ と関数 $f:\mathbb C\to\mathbb C$ に対し、積分 $\int_\gamma f(z)\,\mathrm{d}z$ は、分割 $z_0,\ldots,z_n$ に対する和 $\sum_{k=1}^n f(z_k)(z_k-z_{k-1})$ の極限です。

[コーシーの積分定理](https://www.bananaspace.org/wiki/Cauchy_%E7%A7%AF%E5%88%86%E5%AE%9A%E7%90%86)は、複素関数の積分に関する基本定理です。開集合 $U\subseteq\mathbb C$ と正則関数 $f:U\to\mathbb C$ に対し、閉曲線 $\gamma$ が $U$ の中で[零ホモトピー](https://mathworld.wolfram.com/Null-Homotopic.html)ならば、つまり $\gamma$ を $U$ の中で==連続的に一点まで変形==できるならば、
$$\int_\gamma f(z)\,\mathrm dz=0.$$
が成り立ちます。証明は複雑で技術的なので、詳しくは上の[バナナスペース](https://www.bananaspace.org/wiki/%E9%A6%96%E9%A1%B5)のリンクを参照してください。

この定理には、積分計算への応用が数多くあります。たとえば[コーシーの積分公式](https://en.wikipedia.org/wiki/Cauchy_integral_formula)によれば、同じ条件のもとで $a\in U\setminus\gamma$ に対し、
$$\frac 1{2\pi\mathrm i}\int_\gamma \frac{f(z)}{z-a}\,\mathrm dz=\nu(\gamma,a)f(a).$$
が成り立ちます。ここで $\nu(\gamma,a)$ は、$a$ に関する $\gamma$ の[回転数](https://en.wikipedia.org/wiki/Winding_number)です。  
[Hopf の定理](https://en.wikipedia.org/wiki/Hopf_theorem)により、$\gamma$ は $U\setminus\{a\}$ の中で、$a$ の周りを $\nu(\gamma,a)$ 回まわる小円へ変形できるため、小円上の積分へ帰着できます。

<p style="text-align: center;">
  <img src="../../csdiy/cauchy-cut.svg" alt="コーシーの積分公式における切断法の図" style="max-width: 100%; height: auto;">
</p>

同様に、$U\subseteq\mathbb C$ を単連結な開集合とし、$f$ は有限個の特異点 $z_1,\ldots,z_n$ を除いて $U$ 上で正則とします。$z_k$ における $f$ の**留数**を
$$\mathrm{Res}(f,z_k):=\frac 1{2\pi\mathrm i}\int_\gamma f(z)\,\mathrm dz,$$
と定めます。ここで $\gamma$ は $z_k$ を正の向きに囲む小円です。[留数定理](https://en.wikipedia.org/wiki/Residue_theorem)によれば、$U$ 上の閉曲線 $\gamma$ に対して、
$$\frac 1{2\pi\mathrm i}\int_\gamma f(z)\,\mathrm dz=\sum_{k=1}^n \nu(\gamma,z_k)\cdot\mathrm{Res}(f,z_k).$$
が成り立ちます。たとえば $f$ が円環 $r\le|z-a|\le R$ の開近傍で正則とします。内側と外側の境界をそれぞれ $\gamma_\mathrm{in}$、$\gamma_\mathrm{out}$ とすれば、
$$f(z)=\frac 1{2\pi\mathrm i}\left(\int_{\gamma_\mathrm{out}}\frac{f(\zeta)}{\zeta-z}\,\mathrm d\zeta-\int_{\gamma_\mathrm{in}}\frac{f(\zeta)}{\zeta-z}\,\mathrm d\zeta\right).$$

二つの積分に現れる $1/(\zeta-z)$ を、それぞれ次のべき級数に展開します。

$$
\begin{aligned}
\frac 1{\zeta-z}&=\frac 1{\zeta-a}\cdot\frac 1{1-\frac{z-a}{\zeta-a}}=\sum_{k=0}^\infty\frac{(z-a)^k}{(\zeta-a)^{k+1}},\\
\frac 1{\zeta-z}&=-\frac 1{z-a}\cdot\frac 1{1-\frac{\zeta-a}{z-a}}=-\sum_{k=0}^\infty\frac{(\zeta-a)^k}{(z-a)^{k+1}}.
\end{aligned}
$$

二つのべき級数は円環上で一様収束するため、項別に積分して[ローラン級数](https://en.wikipedia.org/wiki/Laurent_series) $f(z)=\sum_{k=-\infty}^\infty a_k(z-a)^k$ を得ます。ここで、
$$a_k:=\frac 1{2\pi\mathrm i}\int_\gamma\frac{f(\zeta)}{(\zeta-a)^{k+1}}\,\mathrm d\zeta.$$
特に $a_{-1}=\mathrm{Res}(f,a)$ です。最も次数の低い非零項の係数が $a_{-m}$ なら、$a$ を $f$ の $m$ 位の[極](https://zh.wikipedia.org/zh-hans/%E6%9E%81%E7%82%B9_(%E5%A4%8D%E5%88%86%E6%9E%90))といい、このとき、
$$\mathrm{Res}(f,a)=\frac 1{(m-1)!}\lim_{z\to a}\frac{\mathrm d^{m-1}}{\mathrm dz^{m-1}}\!\left((z-a)^m f(z)\right).$$

## 部分分数分解

実のところ、留数定理を使うのはいささか大げさです。~~でも、これほど短く書けると、どこか神々しく見えませんか？~~

$Q(z):=\prod_{i=1}^n(z-x_i)$ とおき、点 $x_1,\ldots,x_n$ における多項式 $P(z)$ の Lagrange 補間を考えます。
$$
\begin{aligned}
P(z)&=\sum_{i=1}^n P(x_i)\prod_{j\ne i}\frac{z-x_j}{x_i-x_j}=\sum_{i=1}^n\frac{P(x_i)Q(z)}{Q'(x_i)(z-x_i)}, \\
\frac{P(z)}{Q(z)}&=\sum_{i=1}^n\frac{P(x_i)}{Q'(x_i)}\cdot\frac{1}{z-x_i}.
\end{aligned}
$$
つまり、==Lagrange 補間は $P(z)/Q(z)$ の部分分数分解と等価==です。元の問題に戻り、この方法で $f(z)$ を処理して、両辺に $z$ を掛けてから $z\to\infty$ とすれば、留数定理と同じ結果が得られます。

## 代数的な解法

多変数多項式の恒等判定を使って証明することもできます。元の式を $E_n$ とし、多項式
$$F_n:=\prod_{i<j}(x_i-x_j)\cdot(E_n-n\bmod 2).$$
を考えると、$\deg F_n<\binom{n+1}2$ です。$F_n=0$ を示すには、$F_n$ が次数 $\binom{n+1}2$ の多項式で割り切れることを示せば十分です。

$n$ に関する帰納法を使います。$n=1$ のときは明らかです。$n>1$ とし、すべての $k<n$ について $F_k=0$ と仮定します。

- $x_1=1$ のとき $E_n=1-E_{n-1}$ なので、帰納法の仮定から $F_n=0$ です。よって $x_1-1$ は $F_n$ を割ります。
- $x_1=x_2$ のとき、$E_n$ のうち分母に $x_1-x_2$ を含む最初の 2 項だけを考えればよく、簡単な計算で互いに打ち消し合うことが分かります。

以上と $E_n$ の対称性から、すべての $x_i-x_j$ と $x_i-1$ が $F_n$ を割るため、$F_n=0$ です。  
実はこの解法も少し「範囲外」で、多項式環の一意分解性を使いますが、普通はそのまま利用されています。
