---
title: Algebraic Identity
createTime: 2026/06/26 17:19:45
permalink: /en/csdiy/isl19-a5/
---

> The best way to learn any concept is to discover it by yourself.

Let $x_1,\ldots,x_n$ be pairwise distinct real numbers. Prove that
$$\sum_{i=1}^n\prod_{j\ne i}\frac{1-x_ix_j}{x_i-x_j}=n\bmod 2.$$

## Solution

This is Algebra Problem 5 from the 2019 IMO Shortlist. See the [AoPS discussion thread](https://artofproblemsolving.com/community/c6h2279006p17828803).

It is enough to prove the case where none of $x_1,\ldots,x_n$ is $\pm 1$; the general case then follows by taking limits, using continuity.  
Consider the rational function $f:\mathbb C\to\mathbb C$ defined by
$$f(z):=\frac{\prod_{i=1}^n(1-x_iz)}{(1-z^2)\prod_{i=1}^n(z-x_i)}.$$
All singularities of $f$, namely $\pm 1,x_1,\ldots,x_n$, are simple poles, and the limit at infinity is $0$. Computing the residues at these poles and at infinity gives
$$
\begin{aligned}
\mathrm{Res}(f,1)&=\lim_{z\to 1}(z-1)f(z)=-\frac{\prod_{i=1}^n(1-x_i)}{2\prod_{i=1}^n(1-x_i)}=-\frac12,\\
\mathrm{Res}(f,-1)&=\lim_{z\to -1}(z+1)f(z)=\frac{\prod_{i=1}^n(1+x_i)}{2(-1)^n\prod_{i=1}^n(1+x_i)}=\frac{(-1)^n}{2},\\
\mathrm{Res}(f,x_i)&=\lim_{z\to x_i}(z-x_i)f(z)=\prod_{j\ne i}\frac{1-x_ix_j}{x_i-x_j},\\
\mathrm{Res}(f,\infty)&=-\lim_{|z|\to\infty}zf(z)=0.
\end{aligned}
$$
By the [residue theorem](https://en.wikipedia.org/wiki/Residue_theorem), the sum of all residues is $0$. Hence
$$\mathrm{LHS}=\sum_{i=1}^n\mathrm{Res}(f,x_i)=\frac{1-(-1)^n}{2}=n\bmod 2.$$

## What Does It Mean?

Someone might ask: what if we have not learned the residue theorem yet? Then let us just write down its proof.

<p style="text-align: center;">
  <img src="./yua.jpg" alt="yua.jpg" style="height: 250px; width: auto;">
</p>

Recall the definition of a derivative. A function $f:\mathbb C\to\mathbb C$ is differentiable at $a$ if
$$f'(a):=\lim\limits_{z\to a}\frac{f(z)-f(a)}{z-a}$$
exists.

- If $f'(a)\ne 0$, then $f(z)-f(a)$ is obtained from $z-a$ by a fixed scaling and rotation. In this case, $f$ is [conformal](https://en.wikipedia.org/wiki/Conformal_map) at $a$.
- Otherwise $f'(a)=0$, meaning $f(z)-f(a)$ is much smaller than $z-a$. In this case, $a$ is a [critical point](https://en.wikipedia.org/wiki/Critical_point_(mathematics)).

In particular, the images of the real-axis and imaginary-axis directions remain orthogonal. This gives the [Cauchy-Riemann equations](https://en.wikipedia.org/wiki/Cauchy%E2%80%93Riemann_equations):  
if $f$ sends $x+\mathrm iy$ to $u+\mathrm iv$, then $\partial_y v=\partial_x u$ and $\partial_y u=-\partial_x v$.

Now recall [contour integration](https://en.wikipedia.org/wiki/Contour_integration). Given a rectifiable curve $\gamma$ and a function $f:\mathbb C\to\mathbb C$, the integral $\int_\gamma f(z)\,\mathrm{d}z$ is the limit of sums $\sum_{k=1}^n f(z_k)(z_k-z_{k-1})$.

The [Cauchy integral theorem](https://www.bananaspace.org/wiki/Cauchy_%E7%A7%AF%E5%88%86%E5%AE%9A%E7%90%86) is the fundamental theorem for integration of complex functions: given an open set $U\subseteq\mathbb C$ and a holomorphic function $f:U\to\mathbb C$, if a closed curve $\gamma$ is [null-homotopic](https://mathworld.wolfram.com/Null-Homotopic.html) in $U$ (meaning $\gamma$ can be ==continuously deformed== to a single point; topologically, this expresses that the interior of $\gamma$ lies in $U$), then
$$\int_\gamma f(z)\,\mathrm dz=0.$$
The proof is rather technical. For details, see the [Banana Space](https://www.bananaspace.org/wiki/%E9%A6%96%E9%A1%B5) link above.

This theorem has a series of applications to integral computation, such as the [Cauchy integral formula](https://en.wikipedia.org/wiki/Cauchy_integral_formula). Under the same hypotheses, for $a\in U\setminus\gamma$, we have
$$\frac 1{2\pi\mathrm i}\int_\gamma \frac{f(z)}{z-a}\,\mathrm dz=\nu(\gamma,a)f(a).$$
Here $\nu(\gamma,a)$ is the [winding number](https://en.wikipedia.org/wiki/Winding_number) of $\gamma$ around $a$.  
By the [Hopf theorem](https://en.wikipedia.org/wiki/Hopf_theorem), $\gamma$ can be deformed in $U\setminus\{a\}$ to a small circle winding around $a$ exactly $\nu(\gamma,a)$ times, reducing the formula to an integral over that small circle.

<p style="text-align: center;">
  <img src="./cauchy-cut.svg" alt="Diagram of the cut method in the Cauchy integral formula" style="max-width: 100%; height: auto;">
</p>

Similarly, suppose $U\subseteq\mathbb C$ is a simply connected open set, and $f$ is holomorphic on $U$ except at finitely many singularities $z_1,\ldots,z_n$. Define the **residue** of $f$ at $z_k$ by
$$\mathrm{Res}(f,z_k):=\frac 1{2\pi\mathrm i}\int_\gamma f(z)\,\mathrm dz,$$
where $\gamma$ is a small positively oriented circle around $z_k$. Then the [residue theorem](https://en.wikipedia.org/wiki/Residue_theorem) says that, for a closed curve $\gamma$ in $U$,
$$\frac 1{2\pi\mathrm i}\int_\gamma f(z)\,\mathrm dz=\sum_{k=1}^n \nu(\gamma,z_k)\cdot\mathrm{Res}(f,z_k).$$
For example, suppose $f$ is holomorphic on an open neighborhood of the annulus $r\le|z-a|\le R$. Let its inner and outer boundary curves be $\gamma_\mathrm{in}$ and $\gamma_\mathrm{out}$. Then
$$f(z)=\frac 1{2\pi\mathrm i}\left(\int_{\gamma_\mathrm{out}}\frac{f(\zeta)}{\zeta-z}\,\mathrm d\zeta-\int_{\gamma_\mathrm{in}}\frac{f(\zeta)}{\zeta-z}\,\mathrm d\zeta\right).$$

Expand the two factors $1/(\zeta-z)$ in these integrals as the following power series:

$$
\begin{aligned}
\frac 1{\zeta-z}&=\frac 1{\zeta-a}\cdot\frac 1{1-\frac{z-a}{\zeta-a}}=\sum_{k=0}^\infty\frac{(z-a)^k}{(\zeta-a)^{k+1}},\\
\frac 1{\zeta-z}&=-\frac 1{z-a}\cdot\frac 1{1-\frac{\zeta-a}{z-a}}=-\sum_{k=0}^\infty\frac{(\zeta-a)^k}{(z-a)^{k+1}}.
\end{aligned}
$$

These two power series converge uniformly on the annulus, so integrating term by term gives the [Laurent series](https://en.wikipedia.org/wiki/Laurent_series)
$$f(z)=\sum_{k=-\infty}^\infty a_k(z-a)^k,$$
where
$$a_k:=\frac 1{2\pi\mathrm i}\int_\gamma\frac{f(\zeta)}{(\zeta-a)^{k+1}}\,\mathrm d\zeta.$$
In particular, $a_{-1}=\mathrm{Res}(f,a)$. If the lowest-degree nonzero term has coefficient $a_{-m}$, then $a$ is called a pole of order $m$ of $f$, and
$$\mathrm{Res}(f,a)=\frac 1{(m-1)!}\lim_{z\to a}\frac{\mathrm d^{m-1}}{\mathrm dz^{m-1}}\!\left((z-a)^m f(z)\right).$$

## Partial Fractions

In fact, the residue theorem is indeed a bit of overkill here, although the proof does look wonderfully short.

Let $Q(z):=\prod_{i=1}^n(z-x_i)$. Consider the Lagrange interpolation formula for a polynomial $P(z)$ at the points $x_1,\ldots,x_n$:
$$
\begin{aligned}
P(z)&=\sum_{i=1}^n P(x_i)\prod_{j\ne i}\frac{z-x_j}{x_i-x_j}=\sum_{i=1}^n\frac{P(x_i)Q(z)}{Q'(x_i)(z-x_i)}, \\
\frac{P(z)}{Q(z)}&=\sum_{i=1}^n\frac{P(x_i)}{Q'(x_i)}\cdot\frac{1}{z-x_i}.
\end{aligned}
$$
Thus ==Lagrange interpolation is equivalent to the partial fraction decomposition of $P(z)/Q(z)$==. Returning to the original problem, if we treat $f(z)$ by this method, multiply both sides by $z$, and then let $z\to\infty$, we obtain the same result.

## Algebraic Solution

We can also prove the identity by using polynomial identity testing in $n$ variables. Let the original expression be $E_n$, and define
$$F_n:=\prod_{i<j}(x_i-x_j)\cdot(E_n-n\bmod 2).$$
Then $\deg F_n<\binom{n+1}2$. To prove $F_n=0$, it is enough to show that $F_n$ is divisible by a polynomial of degree $\binom{n+1}2$.

We induct on $n$. The case $n=1$ is clear. Otherwise, assume $F_k=0$ for every $k<n$.

- If $x_1=1$, then $E_n=1-E_{n-1}$, so the induction hypothesis gives $F_n=0$.  
  Hence $x_1-1$ divides $F_n$.
- If $x_1=x_2$, then we only need to consider the two terms in $E_n$ whose denominators contain $x_1-x_2$; a direct computation shows that these two terms cancel.

By the symmetry of $E_n$, all factors $x_i-x_j$ and $x_i-1$ divide $F_n$. Therefore $F_n=0$.  
This solution is also somewhat beyond the usual syllabus, since it uses unique factorization in polynomial rings, but in practice people use that fact directly.
