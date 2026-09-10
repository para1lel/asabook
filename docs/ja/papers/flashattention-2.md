---
title: 'FlashAttention-2'
createTime: 2026/09/10 00:00:00
permalink: /ja/papers/flashattention-2/
pageClass: paper-reading
---

> [Tri Dao](https://tridao.me/)。2023 年 7 月 17 日に arXiv へ初投稿、現行版は v1、[ICLR 2024](https://openreview.net/forum?id=mZn2Xyh9Ec) で発表。[FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691)。<a href="/paper/flashattention-2.pdf" target="_blank" rel="noopener noreferrer">原 PDF</a>。[DOI](https://doi.org/10.48550/arXiv.2307.08691)。[TeX ソース](https://export.arxiv.org/e-print/2307.08691v1)。正確な印刷レイアウトと参考文献については、原 PDF を正本とする。

## 概要

Transformer をより長い系列長へスケールさせることは、ここ数年の大きな課題であり、言語モデリングや高解像度画像理解の性能向上をもたらすだけでなく、コード、音声、動画生成における新たな応用を切り開く可能性がある。Attention 層は、実行時間とメモリが系列長に対して二次的に増加するため、より長い系列へスケールする際の主なボトルネックである。FlashAttention [Dao22] は、GPU の非対称なメモリ階層を活用することで、近似を用いずに、大幅なメモリ節約（二次ではなく線形）と実行時間の高速化（最適化されたベースラインと比べて 2-4$\times$）を実現する。しかし、FlashAttention は依然として最適化された行列乗算（GEMM）演算ほど高速ではなく、理論上の最大 FLOPs/s の 25-40% にしか到達しない。この非効率性は、GPU 上の異なるスレッドブロックと warp の間で作業分割が最適化されておらず、その結果、占有率が低くなるか、共有メモリの不要な読み書きが発生するためであることが分かった。本稿では、これらの問題に対処するため、作業分割を改善した FlashAttention-2 を提案する。具体的には、（1）非 matmul FLOP の数を減らすようアルゴリズムを調整し、（2）単一の head であっても attention 計算を異なるスレッドブロック間で並列化して占有率を高め、（3）各スレッドブロック内で warp 間に作業を分配して共有メモリを介した通信を削減する。これにより FlashAttention と比べて約 2$\times$ の高速化を実現し、A100 上で理論上の最大 FLOPs/s の 50-73% に達して、GEMM 演算の効率に近づく。GPT 型モデルの end-to-end 学習に用いた場合、FlashAttention-2 が A100 GPU 1 基当たり最大 225 TFLOPs/s（モデル FLOPs 利用率 72%）の学習速度に達することを実験的に検証する。[+1]

<span id="section-1"></span>

## 1 はじめに

Transformer [Vas17] のコンテキスト長を拡大することは、その中核にある attention 層の実行時間とメモリ要件が入力系列長に対して二次であるため、困難な課題である。理想的には、標準的な系列長 2k の制限を超え、書籍、高解像度画像、長尺動画を理解するモデルを学習したい。この 1 年だけでも、従来よりはるかに長いコンテキストを持つ言語モデルがいくつか登場している。すなわち、コンテキスト長 32k の GPT-4 [Ope23]、コンテキスト長 65k の MosaicML の MPT、コンテキスト長 100k の Anthropic の Claude である。長文書への問い合わせや物語執筆といった新たなユースケースは、このような長いコンテキストを持つモデルの必要性を示している。

このような長いコンテキストにおける attention の計算要件を削減するため、attention を近似する多数の手法が提案されてきた [Kit20, Roy21, Wan20a, Kat20, Cho20a, Bel20a, Zah20, Che21b]。これらの手法はいくつかのユースケースで利用されているものの、著者の知る限り、大規模な学習の大半では依然として標準的な attention が用いられている。これを動機として、Dao ら [Dao22] は attention 計算を並べ替え、古典的な手法（タイリング、再計算）を活用して計算を大幅に高速化し、メモリ使用量を系列長に対して二次から線形へ削減することを提案した。これにより、近似を用いずに、最適化されたベースラインと比べて wall-clock 時間を 2-4$\times$ 高速化し、メモリを最大 10-20$\times$ 節約できる。その結果、FlashAttention は Transformer の大規模な学習と推論で広く採用されている。

しかし、コンテキスト長がさらに増加するにつれ、FlashAttention は依然として行列乗算（GEMM）などの他のプリミティブほど効率的ではない。具体的には、FlashAttention は標準的な attention 実装よりすでに 2-4$\times$ 高速であるものの、forward pass はデバイスの理論上の最大 FLOPs/s の 30-50% にしか達せず（[図 5](#figure-05)）、backward pass はさらに難しく、A100 GPU の最大スループットの 25-35% にしか達しない（[図 6](#figure-06)）。これに対し、最適化された GEMM はデバイスの理論上の最大スループットの最大 80-90% に到達できる。慎重なプロファイリングを通じて、FlashAttention では GPU 上の異なるスレッドブロックと warp の間の作業分割が依然として最適ではなく、その結果、占有率が低くなるか、共有メモリの不要な読み書きが発生することが分かった。

FlashAttention を基に、本稿ではこれらの課題に対処するため、並列性と作業分割を改善した FlashAttention-2 を提案する。

1. [第 3.1 節](#section-3-1)では、出力を変えずに非 matmul FLOP の数を減らすようアルゴリズムを調整する。非 matmul FLOP は総 FLOP 数のごく一部しか占めないが、GPU には行列乗算専用のユニットがあるため、実行にはより長い時間がかかり、その結果 matmul のスループットは非 matmul のスループットより最大 16$\times$ 高くなり得る。したがって、非 matmul FLOP を減らし、可能な限り多くの時間を matmul FLOP の実行に費やすことが重要である。

2. batch と head 数の次元に加え、系列長の次元に沿って forward pass と backward pass の両方を並列化することを提案する。これにより、系列が長い場合（したがって batch size が小さいことが多い）に占有率（GPU リソースの利用率）が向上する。

3. attention 計算の 1 ブロック内でも、スレッドブロックの異なる warp 間で作業を分割し、通信と共有メモリの読み書きを削減する。

[第 4 節](#section-4)では、FlashAttention-2 が FlashAttention と比べても大幅な高速化をもたらすことを実験的に検証する。異なる設定（causal mask の有無、異なる head dimension）でのベンチマークにより、FlashAttention-2 が FlashAttention に対して約 2$\times$ の高速化を達成し、forward pass では理論上の最大スループットの最大 73%、backward pass では最大 63% に達することを示す。GPT 型モデルの end-to-end 学習に用いると、A100 GPU 1 基当たり最大 225 TFLOPs/s の学習速度に達する。

<span id="section-2"></span>

## 2 背景

GPU の性能特性と実行モデルについて背景を説明する。また、attention の標準的な実装と FlashAttention についても説明する。

<span id="section-2-1"></span>

### 2.1 ハードウェア特性

**GPU の性能特性。** GPU は計算要素（たとえば浮動小数点演算ユニット）とメモリ階層から構成される。最新 GPU の多くは、低精度の行列乗算を高速化する専用ユニット（たとえば Nvidia GPU における FP16/BF16 行列乗算用の Tensor Core）を備えている。メモリ階層は、高帯域幅メモリ（HBM）とオンチップ SRAM（別名 shared memory）からなる。一例として、A100 GPU は帯域幅 1.5-2.0TB/s の高帯域幅メモリ（HBM）を 40-80GB 備え、108 基ある各 streaming multiprocessor には、帯域幅が約 19TB/s と推定される 192KB のオンチップ SRAM が搭載されている [Jia18a, Jia21]。L2 cache はプログラマが直接制御できないため、ここでの議論では HBM と SRAM に焦点を当てる。

**実行モデル。** GPU は、演算（kernel と呼ばれる）を実行するために膨大な数の thread を備えている。thread は thread block に編成され、streaming multiprocessor（SM）上で実行されるようスケジュールされる。各 thread block 内では、thread は warp（32 thread のグループ）にまとめられる。warp 内の thread は、高速な shuffle 命令で通信したり、協調して行列乗算を実行したりできる。thread block 内の warp は、shared memory の読み書きによって通信できる。各 kernel は入力を HBM から register と SRAM へロードし、計算を行った後、出力を HBM へ書き込む。

<span id="section-2-2"></span>

### 2.2 標準的なアテンション実装

$N$ を系列長、$d$ を head dimension とする入力系列 $\mathbf{Q}, \mathbf{K}, \mathbf{V}\in \mathbb{R}^{N \times d}$ が与えられたとき、attention の出力 $\mathbf{O}\in \mathbb{R}^{N \times d}$ を次のように計算したい。

$$
\mathbf{S}= \mathbf{Q}\mathbf{K}^\top \in \mathbb{R}^{N \times N}, \quad \mathbf{P}= \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \quad \mathbf{O}= \mathbf{P}\mathbf{V}\in \mathbb{R}^{N \times d},
$$

ここで $\mathrm{softmax}$ は行ごとに適用される。[+2] Multi-head attention（MHA）では、この同じ計算が多数の head にわたって並列に実行され、batch dimension（1 batch 内の入力系列数）についても並列に実行される。

attention の backward pass は次のように進む。$\mathbf{dO}\in \mathbb{R}^{N \times d}$ を、ある損失関数に関する $\mathbf{O}$ の勾配とする。このとき連鎖律（別名 backpropagation）により、次を得る。

$$
\begin{aligned}
\mathbf{dV}&= \mathbf{P}^\top \mathbf{dO}\in \mathbb{R}^{N \times d} \\
  \mathbf{dP}&= \mathbf{dO}\mathbf{V}^\top \in \mathbb{R}^{N \times N} \\
  \mathbf{dS}&= \mathrm{dsoftmax}(\mathbf{dP}) \in \mathbb{R}^{N \times N} \\
  \mathbf{dQ}&= \mathbf{dS}\mathbf{K}\in \mathbb{R}^{N \times d} \\
  \mathbf{dK}&= \mathbf{Q}\mathbf{dS}^\top \in \mathbb{R}^{N \times d},
\end{aligned}
$$

ここで $\mathrm{dsoftmax}$ は、行ごとに適用される softmax の勾配（backward pass）である。あるベクトル $s$ と $p$ について $p = \mathrm{softmax}(s)$ であるとき、出力勾配を $dp$ とすれば、入力勾配が $ds = (\mathrm{diag}(p) - p p^\top)dp$ となることを導出できる。

標準的な attention 実装では、行列 $\mathbf{S}$ と $\mathbf{P}$ を HBM 上に実体化するため、$O(N^2)$ のメモリを要する。多くの場合 $N \gg d$ である（通常、$N$ は 1k-8k 程度、$d$ は 64-128 程度）。標準的な attention 実装は、（1）行列乗算（GEMM）サブルーチンを呼び出して $\mathbf{S}= \mathbf{Q}\mathbf{K}^\top$ を計算し、その結果を HBM に書き込み、次に（2）$\mathbf{S}$ を HBM からロードして softmax を計算し、結果 $\mathbf{P}$ を HBM に書き込み、最後に（3）GEMM を呼び出して $\mathbf{O}= \mathbf{P}\mathbf{V}$ を得る。ほとんどの演算はメモリ帯域幅によって律速されるため、大量のメモリアクセスは wall-clock 時間の増大につながる。さらに、$\mathbf{S}$ と $\mathbf{P}$ を実体化する必要があるため、必要なメモリは $O(N^2)$ である。加えて、backward pass で勾配を計算するために、$\mathbf{P}\in \mathbb{R}^{N \times N}$ を保存しなければならない。

<span id="section-2-3"></span>

### 2.3 FlashAttention

GPU などの hardware accelerator 上で attention を高速化するため、[Dao22] は、同じ出力を維持しながら（近似を用いずに）メモリの読み書きを減らすアルゴリズムを提案している。

<span id="section-2-3-1"></span>

#### 2.3.1 フォワードパス

FlashAttention は古典的なタイリング手法を適用してメモリ IO を削減する。具体的には、（1）入力のブロックを HBM から SRAM にロードし、（2）そのブロックに関する attention を計算し、（3）大きな中間行列 $\mathbf{S}$ と $\mathbf{P}$ を HBM に書き込まずに出力を更新する。softmax は行全体または行ブロック全体を結び付けるため、online softmax [Mil18, Rab21] は attention 計算をブロックに分割し、各ブロックの出力を再スケーリングして最終的に正しい結果を得られる（近似は用いない）。メモリの読み書き量を大幅に削減することで、FlashAttention は最適化されたベースラインの attention 実装に対して wall-clock 時間を 2-4$\times$ 高速化する。

online softmax 手法 [Mil18] と、それが attention でどのように使われるか [Rab21] を説明する。簡単のため、attention 行列 $\mathbf{S}$ の 1 つの行ブロックだけを考える。これは、ある行列 $\mathbf{S}^{(1)}, \mathbf{S}^{(2)} \in \mathbb{R}^{B_r \times B_c}$ に対して $\begin{bmatrix} \mathbf{S}^{(1)} & \mathbf{S}^{(2)} \end{bmatrix}$ の形をしており、$B_r$ と $B_c$ は行ブロックと列ブロックのサイズである。この行ブロックの softmax を計算し、ある行列 $\mathbf{V}^{(1)}, \mathbf{V}^{(2)} \in \mathbb{R}^{B_c \times d}$ に対して $\begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix}$ の形をした value と乗算したい。標準的な softmax は次を計算する。

$$
\begin{aligned}
m &= \max(\mathrm{rowmax}(\mathbf{S}^{(1)}), \mathrm{rowmax}(\mathbf{S}^{(2)})) \in \mathbb{R}^{B_r}  \\
  \ell &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) \in \mathbb{R}^{B_r}  \\
  \mathbf{P}&= \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} = \mathrm{diag}(\ell)^{-1}\begin{bmatrix} e^{\mathbf{S}^{(1)} - m} & e^{\mathbf{S}^{(2)} - m} \end{bmatrix} \in \mathbb{R}^{B_r \times 2B_c} \\
  \mathbf{O}&= \begin{bmatrix} \mathbf{P}^{(1)} & \mathbf{P}^{(2)} \end{bmatrix} \begin{bmatrix} \mathbf{V}^{(1)} \\ \mathbf{V}^{(2)} \end{bmatrix} = \mathrm{diag}(\ell)^{-1} e^{\mathbf{S}^{(1)} - m} \mathbf{V}^{(1)} + e^{\mathbf{S}^{(2)} - m} \mathbf{V}^{(2)} \in \mathbb{R}^{B_r \times d}.
\end{aligned}
$$

これに対して online softmax は、各ブロックに関して「local」softmax を計算し、再スケーリングによって最終的に正しい出力を得る。

$$
\begin{aligned}
m^{(1)} &= \mathrm{rowmax}(\mathbf{S}^{(1)})  \in \mathbb{R}^{B_r}\\
  \ell^{(1)} &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R}^{B_r} \\
  \tilde{\mathbf{P}}^{(1)} &= \mathrm{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \in \mathbb{R}^{B_r \times B_c}\\
  \mathbf{O}^{(1)} &= \tilde{\mathbf{P}}^{(1)} \mathbf{V}^{(1)} = \mathrm{diag}(\ell^{(1)})^{-1} e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d}\\
  m^{(2)} &= \max(m^{(1)}, \mathrm{rowmax}(\mathbf{S}^{(2)})) = m \\
  \ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\
  \tilde{\mathbf{P}}^{(2)} &= \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\
  \mathbf{O}^{(2)} &= \mathrm{diag}(\ell^{(1)} / \ell^{(2)})^{-1} \mathbf{O}^{(1)} + \tilde{\mathbf{P}}^{(2)} \mathbf{V}^{(2)} = \mathrm{diag}(\ell^{(2)})^{-1} e^{s^{(1)} - m} \mathbf{V}^{(1)} + \mathrm{diag}(\ell^{(2)})^{-1} e^{s^{(2)} - m} \mathbf{V}^{(2)} = \mathbf{O}.
\end{aligned}
$$

FlashAttention が online softmax を用いてタイリングを可能にし（[図 1](#figure-01)）、メモリの読み書きを削減する方法を示す。

<span id="figure-01"></span>

![FlashAttention のフォワードパスの模式図](../../papers/flashattention-2/figure-01.png)

**図 1。** key $\mathbf{K}$ を 2 ブロックに分割し、value $\mathbf{V}$ も 2 ブロックに分割した場合に、FlashAttention のフォワードパスがどのように実行されるかを示す模式図。各ブロックに関してアテンションを計算し、出力を再スケーリングすることで、中間行列 $\mathbf{S}$ と $\mathbf{P}$ の高コストなメモリ読み書きを避けながら、最終的に正しい答えを得る。図を簡略化するため、softmax で各要素から行ごとの最大値を引くステップは省略している。

<span id="section-2-3-2"></span>

#### 2.3.2 バックワードパス

backward pass では、入力 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ のブロックが SRAM にロードされた後で attention 行列 $\mathbf{S}$ と $\mathbf{P}$ の値を再計算することにより、FlashAttention は大きな中間値を保存せずに済む。サイズ $N \times N$ の大きな行列 $\mathbf{S}$ と $\mathbf{P}$ を保存する必要がないため、FlashAttention は系列長に応じて 10-20$\times$ のメモリ節約を実現する（必要なメモリは系列長 $N$ に対して二次ではなく線形）。メモリの読み書きが減るため、backward pass も wall-clock 時間を 2-4$\times$ 高速化する。

backward pass は、[第 2.2 節](#section-2-2)の式にタイリングを適用する。概念的には backward pass は forward pass より単純であるものの（softmax の再スケーリングがない）、実装はかなり複雑である。これは、forward pass では行列乗算がわずか 2 回なのに対し、backward pass では 5 回の行列乗算を行うため、SRAM に保持すべき値が多いからである。

<span id="section-3"></span>

## 3 FlashAttention-2：アルゴリズム、並列性、作業分割

FlashAttention-2 のアルゴリズムについて説明する。このアルゴリズムには、非 matmul FLOP の数を減らすための FlashAttention に対するいくつかの調整が含まれる。次に、GPU リソースを十分に活用するため、異なるスレッドブロック上で計算を並列化する方法を説明する。最後に、共有メモリアクセスの量を減らすため、1 つのスレッドブロック内の異なる warp 間で作業を分割する方法を説明する。これらの改善により、[第 4 節](#section-4)で検証するように、2-3$\times$ の高速化が得られる。

<span id="section-3-1"></span>

### 3.1 アルゴリズム

FlashAttention のアルゴリズムを調整し、非 matmul FLOP の数を減らす。これは、最新 GPU が matmul を大幅に高速化する専用の計算ユニット（たとえば Nvidia GPU の Tensor Core）を備えているためである。一例として、A100 GPU における FP16/BF16 matmul の理論上の最大スループットは 312 TFLOPs/s だが、非 matmul FP32 ではわずか 19.5 TFLOPs/s である。別の見方をすれば、各非 matmul FLOP は matmul FLOP より 16$\times$ 高コストである。高いスループット（たとえば理論上の最大 TFLOPs/s の 50% 超）を維持するため、可能な限り多くの時間を matmul FLOP に費やしたい。

<span id="section-3-1-1"></span>

#### 3.1.1 フォワードパス

[第 2.3 節](#section-2-3)に示した online softmax の技法を改めて検討し、非 matmul FLOP を減らすために 2 つの小さな調整を加える。

1. 出力更新の両方の項を $\mathrm{diag}(\ell^{(2)})^{-1}$ で再スケーリングする必要はない。

    $$
    \mathbf{O}^{(2)} = \mathrm{diag}(\ell^{(1)} / \ell^{(2)})^{-1} \mathbf{O}^{(1)} + \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}.
    $$

    代わりに $\mathbf{O}^{(2)}$ の「un-scaled」版を保持し、統計量 $\ell^{(2)}$ も保持できる。

    $$
    \tilde{\mathbf{O}}^{(2)} = \mathrm{diag}(\ell^{(1)})^{-1} \mathbf{O}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)}.
    $$

    ループのまさに最後でのみ、最終的な $\tilde{\mathbf{O}}^{(\mathrm{last})}$ を $\mathrm{diag}(\ell^{(\mathrm{last})})^{-1}$ でスケーリングして正しい出力を得る。

2. backward pass のために、最大値 $m^{(j)}$ と指数関数の和 $\ell^{(j)}$ の両方を保存する必要はない。logsumexp $L^{(j)} = m^{(j)} + \log(\ell^{(j)})$ だけを保存すればよい。

[第 2.3 節](#section-2-3)の 2 ブロックという単純な場合、online softmax の技法は次のようになる。

$$
\begin{aligned}
m^{(1)} &= \mathrm{rowmax}(\mathbf{S}^{(1)})  \in \mathbb{R}^{B_r}\\
  \ell^{(1)} &= \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m^{(1)}}) \in \mathbb{R}^{B_r} \\
  \tilde{\mathbf{O}^{(1)}} &= e^{\mathbf{S}^{(1)} - m^{(1)}} \mathbf{V}^{(1)} \in \mathbb{R}^{B_r \times d}\\
  m^{(2)} &= \max(m^{(1)}, \mathrm{rowmax}(\mathbf{S}^{(2)})) = m \\
  \ell^{(2)} &= e^{m^{(1)} - m^{(2)}} \ell^{(1)} + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m^{(2)}}) = \mathrm{rowsum}(e^{\mathbf{S}^{(1)} - m}) + \mathrm{rowsum}(e^{\mathbf{S}^{(2)} - m}) = \ell \\
  \tilde{\mathbf{P}}^{(2)} &= \mathrm{diag}(\ell^{(2)})^{-1} e^{\mathbf{S}^{(2)} - m^{(2)}} \\
  \tilde{\mathbf{O}}^{(2)} &= \mathrm{diag}(e^{m^{(1)} - m^{(2)}})^{-1} \tilde{\mathbf{O}}^{(1)} + e^{\mathbf{S}^{(2)} - m^{(2)}} \mathbf{V}^{(2)} = e^{s^{(1)} - m} \mathbf{V}^{(1)} + e^{s^{(2)} - m} \mathbf{V}^{(2)} \\
  \mathbf{O}^{(2)} &= \mathrm{diag}(\ell^{(2)})^{-1} \tilde{\mathbf{O}}^{(2)} = \mathbf{O}.
\end{aligned}
$$

FlashAttention-2 の forward pass 全体を [アルゴリズム 1](#algorithm-01)に示す。

<span id="algorithm-01"></span>

**アルゴリズム 1：FlashAttention-2 のフォワードパス**

- **必要条件：** HBM 内の行列 $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$、ブロックサイズ $B_c$, $B_r$。
- $\mathbf{Q}$ を、それぞれサイズ $B_r \times d$ の $T_r = \left\lceil\frac{N}{B_r} \right\rceil$ 個のブロック $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_r}$ に分割し、$\mathbf{K}, \mathbf{V}$ を、それぞれサイズ $B_c \times d$ の $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ 個のブロック $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ および $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$ に分割する。
- 出力 $\mathbf{O} \in \mathbb{R}^{N \times d}$ を、それぞれサイズ $B_r \times d$ の $T_r$ 個のブロック $\mathbf{O}_i, \dots, \mathbf{O}_{T_r}$ に分割し、logsumexp $L$ を、それぞれサイズ $B_r$ の $T_r$ 個のブロック $L_i, \dots, L_{T_r}$ に分割する。
- **各** $1 \le i \le T_r$ **について：**
  - $\mathbf{Q}_i$ を HBM からオンチップ SRAM へロードする。
  - オンチップで、$\mathbf{O}_{i}^{(0)} = (0)_{B_r \times d} \in \mathbb{R}^{B_r \times d}, \ell_{i}^{(0)} = (0)_{B_r} \in \mathbb{R}^{B_r}, m_{i}^{(0)} = (-\infty)_{B_r} \in \mathbb{R}^{B_r}$ を初期化する。
  - **各** $1 \le j \le T_c$ **について：**
    - $\mathbf{K}_j, \mathbf{V}_j$ を HBM からオンチップ SRAM へロードする。
    - オンチップで、$\mathbf{S}_{i}^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c}$ を計算する。
    - オンチップで、$m_{i}^{(j)} = \max(m_{i}^{(j-1)}, \mathrm{rowmax}(\mathbf{S}_{i}^{(j)})) \in \mathbb{R}^{B_r}$、$\tilde{\mathbf{P}}_{i}^{(j)} = \exp(\mathbf{S}_{i}^{(j)} - m_{i}^{(j)}) \in \mathbb{R}^{B_r \times B_c}$（要素ごと）、$\ell_{i}^{(j)} = e^{m_{i}^{j-1} - m_{i}^{(j)}} \ell_{i}^{(j-1)} + \mathrm{row\ sum}(\tilde{\mathbf{P}}_{i}^{(j)}) \in \mathbb{R}^{B_r}$ を計算する。
    - オンチップで、$\mathbf{O}_{i}^{(j)} = \mathrm{diag}(e^{m_{i}^{(j-1)} - m_{i}^{(j)}})^{-1} \mathbf{O}_{i}^{(j-1)} + \tilde{\mathbf{P}}_{i}^{(j)} \mathbf{V}_j$ を計算する。
  - オンチップで、$\mathbf{O}_{i} = \mathrm{diag}(\ell_{i}^{(T_c)})^{-1} \mathbf{O}_{i}^{(T_c)}$ を計算する。
  - オンチップで、$L_{i} = m_{i}^{(T_c)} + \log(\ell_i^{(T_c)})$ を計算する。
  - $\mathbf{O}_{i}$ を $\mathbf{O}$ の $i$ 番目のブロックとして HBM に書き込む。
  - $L_{i}$ を $L$ の $i$ 番目のブロックとして HBM に書き込む。
- **返す：** 出力 $\mathbf{O}$ と logsumexp $L$。

**因果マスキング。** attention の一般的なユースケースの 1 つは自己回帰言語モデリングであり、そこでは attention 行列 $\mathbf{S}$ に causal mask を適用する必要がある（すなわち、$j > i$ を満たす任意の要素 $\mathbf{S}_{ij}$ を $-\infty$ に設定する）。

1. FlashAttention と FlashAttention-2 はすでにブロック単位で動作するため、すべての列 index が行 index より大きい任意のブロック（系列長が大きい場合、ブロックの約半分）について、そのブロックの計算を省略できる。これにより、causal mask を使用しない attention と比べて約 1.7-1.8$\times$ の高速化が得られる。

2. 行 index が列 index より厳密に小さいことが保証されるブロックには、causal mask を適用する必要がない。つまり、各行では causal mask を 1 ブロックにだけ適用すればよい（正方形ブロックを仮定）。

**正しさ、実行時間、メモリ要件。** FlashAttention と同様に、[アルゴリズム 1](#algorithm-01)は、$O(N^2d)$ FLOP を用い、入力と出力に加えて $O(N)$ の追加メモリ（logsumexp $L$ の保存用）を必要としながら、正しい出力 $\mathbf{O}= \mathrm{softmax}(\mathbf{Q}\mathbf{K}^\top)\mathbf{V}$ を返す（近似は用いない）。証明は Dao ら [Dao22] の証明（定理 1）とほぼ同じであるため、ここでは省略する。

<span id="section-3-1-2"></span>

#### 3.1.2 バックワードパス

FlashAttention-2 の backward pass は FlashAttention のものとほぼ同じである。softmax における行ごとの最大値と指数関数の行ごとの和の両方ではなく、行ごとの logsumexp $L$ だけを用いるように小さな調整を加える。完全を期すため、backward pass の記述を [アルゴリズム 2](#algorithm-02)に示す。

<span id="algorithm-02"></span>

**アルゴリズム 2：FlashAttention-2 のバックワードパス**

- **必要条件：** HBM 内の行列 $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO} \in \mathbb{R}^{N \times d}$、HBM 内のベクトル $L \in \mathbb{R}^N$、ブロックサイズ $B_c$, $B_r$。
- $\mathbf{Q}$ を、それぞれサイズ $B_r \times d$ の $T_r = \left\lceil\frac{N}{B_r} \right\rceil$ 個のブロック $\mathbf{Q}_1, \dots, \mathbf{Q}_{T_r}$ に分割し、$\mathbf{K}, \mathbf{V}$ を、それぞれサイズ $B_c \times d$ の $T_c = \left\lceil \frac{N}{B_c} \right\rceil$ 個のブロック $\mathbf{K}_1, \dots, \mathbf{K}_{T_c}$ および $\mathbf{V}_1, \dots, \mathbf{V}_{T_c}$ に分割する。
- $\mathbf{O}$ を、それぞれサイズ $B_r \times d$ の $T_r$ 個のブロック $\mathbf{O}_i, \dots, \mathbf{O}_{T_r}$ に分割し、$\mathbf{dO}$ を、それぞれサイズ $B_r \times d$ の $T_r$ 個のブロック $\mathbf{dO}_i, \dots, \mathbf{dO}_{T_r}$ に分割し、$L$ を、それぞれサイズ $B_r$ の $T_r$ 個のブロック $L_i, \dots, L_{T_r}$ に分割する。
- HBM 内で $\mathbf{dQ} = (0)_{N \times d}$ を初期化し、それぞれサイズ $B_r \times d$ の $T_r$ 個のブロック $\mathbf{dQ}_1, \dots, \mathbf{dQ}_{T_r}$ に分割する。$\mathbf{dK}, \mathbf{dV} \in \mathbb{R}^{N \times d}$ を、それぞれサイズ $B_c \times d$ の $T_c$ 個のブロック $\mathbf{dK}_1, \dots, \mathbf{dK}_{T_c}$ および $\mathbf{dV}_1, \dots, \mathbf{dV}_{T_c}$ に分割する。
- $D = \mathrm{rowsum}(\mathbf{dO} \circ \mathbf{O}) \in \mathbb{R}^d$（要素ごとの乗算）を計算し、$D$ を HBM に書き込んで、それぞれサイズ $B_r$ の $T_r$ 個のブロック $D_1, \dots, D_{T_r}$ に分割する。
- **各** $1 \le j \le T_c$ **について：**
  - $\mathbf{K}_j, \mathbf{V}_j$ を HBM からオンチップ SRAM へロードする。
  - SRAM 上で $\mathbf{dK}_j = (0)_{B_c \times d}, \mathbf{dV}_j = (0)_{B_c \times d}$ を初期化する。
  - **各** $1 \le i \le T_r$ **について：**
    - $\mathbf{Q}_i, \mathbf{O}_i, \mathbf{dO}_i, \mathbf{dQ}_i, L_i, D_i$ を HBM からオンチップ SRAM へロードする。
    - オンチップで、$\mathbf{S}_{i}^{(j)} = \mathbf{Q}_i \mathbf{K}_j^\top \in \mathbb{R}^{B_r \times B_c}$ を計算する。
    - オンチップで、$\mathbf{P}_{i}^{(j)} = \exp(\mathbf{S}_{ij} - L_{i}) \in \mathbb{R}^{B_r \times B_c}$ を計算する。
    - オンチップで、$\mathbf{dV}_j \leftarrow \mathbf{dV}_j + (\mathbf{P}_{i}^{(j)})^\top \mathbf{dO}_i \in \mathbb{R}^{B_c \times d}$ を計算する。
    - オンチップで、$\mathbf{dP}_{i}^{(j)} = \mathbf{dO}_{i} \mathbf{V}_j^\top \in \mathbb{R}^{B_r \times B_c}$ を計算する。
    - オンチップで、$\mathbf{dS}_{i}^{(j)} = \mathbf{P}_{i}^{(j)} \circ (\mathbf{dP}_{i}^{(j)} - D_i) \in \mathbb{R}^{B_r \times B_c}$ を計算する。
    - $\mathbf{dQ}_i$ を HBM から SRAM へロードし、続いてオンチップで $\mathbf{dQ}_{i} \leftarrow \mathbf{dQ}_i + \mathbf{dS}_{i}^{(j)} \mathbf{K}_j \in \mathbb{R}^{B_r \times d}$ と更新し、HBM に書き戻す。
    - オンチップで、$\mathbf{dK}_{j} \leftarrow \mathbf{dK}_j + {\mathbf{dS}_{i}^{(j)}}^\top \mathbf{Q}_i \in \mathbb{R}^{B_c \times d}$ を計算する。
  - $\mathbf{dK}_j, \mathbf{dV}_j$ を HBM に書き込む。
- **返す：** $\mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$。

**Multi-query attention と grouped-query attention。** Multi-query attention（MQA）[Sha19] と grouped-query attention（GQA）[Ain23] は、推論時の KV cache のサイズを削減するため、複数の query head が同じ key と value の head を参照する attention の変種である。計算のために key head と value head を複製する代わりに、暗黙的に head への index を操作して同じ計算を実行する。backward pass では、暗黙的に複製された異なる head にわたって勾配 $\mathbf{dK}$ と $\mathbf{dV}$ を合計する必要がある。

<span id="section-3-2"></span>

### 3.2 並列性

FlashAttention の最初のバージョンは、batch size と head 数について並列化する。1 つの attention head を処理するために 1 thread block を用い、全体では $\mathrm{batch size} \cdot \mathrm{number of heads}$ 個の thread block が存在する。各 thread block は streaming multiprocessor（SM）上で実行されるようスケジュールされ、たとえば A100 GPU にはこの SM が 108 基ある。この数が大きい場合（たとえば $\geq 80$）、GPU の計算リソースをほぼすべて効果的に利用できるため、このスケジューリングは効率的である。

長い系列の場合（通常は batch size または head 数が小さいことを意味する）、GPU 上の multiprocessor をより有効に活用するため、ここではさらに系列長の次元についても並列化する。このレジームでは、これによって大幅な高速化が得られる。

**フォワードパス。** 外側のループ（系列長にわたるループ）は embarrassingly parallel であり、相互に通信する必要のない異なる thread block 上へスケジュールできることが分かる。また、FlashAttention と同様に、batch dimension と head 数の次元についても並列化する。系列長に沿った並列性を高めることで、batch size と head 数が小さい場合の占有率（使用されている GPU リソースの割合）が向上し、この場合の高速化につながる。

ループの順序を入れ替えるという考え方（元の FlashAttention 論文とは逆に、外側のループを行ブロック、内側のループを列ブロックにする）と、系列長の次元について並列化するという考え方は、Phil Tillet が Triton [Til19] の実装で最初に提案し、実装した。[+3]

**バックワードパス。** 異なる列ブロック間で共有される唯一の計算は、[アルゴリズム 2](#algorithm-02)で $\mathbf{dQ}$ を更新する部分である。そこでは $\mathbf{dQ}_i$ を HBM から SRAM へロードし、続いてオンチップで $\mathbf{dQ}_{i} \leftarrow \mathbf{dQ}_i + \mathbf{dS}_{i}^{(j)} \mathbf{K}_j$ と更新し、HBM に書き戻す必要がある。そこで系列長の次元についても並列化し、backward pass の各列ブロックに 1 thread block をスケジュールする。異なる thread block 間で通信して $\mathbf{dQ}$ を更新するために atomic add を用いる。

並列化方式を [図 2](#figure-02)に示す。

<span id="figure-02"></span>

![フォワードパスとバックワードパスの並列性](../../papers/flashattention-2/figure-02.png)

**図 2。** フォワードパス（左）では、各 worker（thread block）がアテンション行列の行ブロック 1 つを担当するように worker を並列化する。バックワードパス（右）では、各 worker がアテンション行列の列ブロック 1 つを担当する。

<span id="section-3-3"></span>

### 3.3 Warp 間の作業分割

[第 3.2 節](#section-3-2)では thread block のスケジューリング方法を説明したが、各 thread block の内部でも、異なる warp 間で作業をどのように分割するかを決める必要がある。通常は thread block 1 つ当たり 4 または 8 warp を使用し、その分割を [図 3](#figure-03)に示す。

**フォワードパス。** 各ブロックについて、FlashAttention は $\mathbf{Q}$ をすべての warp からアクセス可能に保ちながら、$\mathbf{K}$ と $\mathbf{V}$ を 4 warp に分割する。各 warp は乗算によって $\mathbf{Q}\mathbf{K}^\top$ の一部を得た後、$\mathbf{V}$ の一部と乗算し、通信によって結果を加算する必要がある。これは「split-K」方式と呼ばれる。しかし、すべての warp が中間結果を shared memory に書き出し、同期してから中間結果を加算する必要があるため、これは非効率である。shared memory のこうした読み書きが FlashAttention の forward pass を遅くする。

FlashAttention-2 では代わりに、$\mathbf{K}$ と $\mathbf{V}$ をすべての warp からアクセス可能に保ちながら、$\mathbf{Q}$ を 4 warp に分割する。各 warp が行列乗算を実行して $\mathbf{Q}\mathbf{K}^\top$ の一部を得た後、共有されている $\mathbf{V}$ の一部と乗算するだけで、対応する出力の一部を得られる。warp 間で通信する必要はない。shared memory の読み書きが減ることで高速化が得られる（[第 4 節](#section-4)）。

<span id="figure-03"></span>

![Warp 間の作業分割](../../papers/flashattention-2/figure-03.png)

**図 3。** フォワードパスにおける異なる warp 間の作業分割

**バックワードパス。** backward pass でも同様に、「split-K」方式を避けるように warp を分割する。ただし、さまざまな入力と勾配 $\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{O}, \mathbf{dO}, \mathbf{dQ}, \mathbf{dK}, \mathbf{dV}$ の間にある、より複雑な依存関係のため、依然として何らかの同期が必要である。それでも、「split-K」を避けることで shared memory の読み書きが減り、ここでも高速化が得られる（[第 4 節](#section-4)）。

**ブロックサイズの調整。** 一般にブロックサイズを大きくすると shared memory の load/store は減るが、必要な register 数と shared memory の総量は増える。ブロックサイズがある値を超えると、register spilling によって大幅に遅くなるか、必要な shared memory の量が GPU で利用可能な量を上回り、kernel をまったく実行できなくなる。通常は、head dimension $d$ とデバイスの shared memory サイズに応じて、$\{64, 128\} \times \{64, 128\}$ のブロックサイズを選ぶ。

ブロックサイズの選択肢は本質的に 4 つしかないため、各 head dimension に対して手動で調整しているが、この手作業を避けるために auto-tuning を活用できる可能性がある。これは今後の課題とする。

<span id="section-4"></span>

## 4 実験的検証

Transformer モデルの学習に FlashAttention-2 を用いた場合の影響を評価する。

- **アテンションのベンチマーク。** 異なる系列長における FlashAttention-2 の実行時間を測定し、PyTorch の標準実装、FlashAttention、Triton の FlashAttention と比較する。FlashAttention-2 は FlashAttention より 1.7-3.0$\times$、Triton の FlashAttention より 1.3-2.5$\times$、標準的な attention 実装より 3-10$\times$ 高速であることを確認する。FlashAttention-2 は最大 230 TFLOPs/s、すなわち A100 GPU の理論上の最大 TFLOPs/s の 73% に達する。

- **エンドツーエンドの学習速度。** 系列長 2k または 8k で、サイズ 1.3B および 2.7B の GPT 型モデルを end-to-end で学習する際に使用すると、FlashAttention-2 は FlashAttention に対して最大 1.3$\times$、FlashAttention を使用しないベースラインに対して 2.8$\times$ の高速化をもたらす。FlashAttention-2 は A100 GPU 1 基当たり最大 225 TFLOPs/s（モデル FLOPs 利用率 72%）に達する。

<span id="section-4-1"></span>

### 4.1 アテンションのベンチマーク

A100 80GB SXM4 GPU 上で、異なる設定（causal mask なし／あり、head dimension 64 または 128）について各 attention 手法の実行時間を測定する。結果を [図 4](#figure-04)、[図 5](#figure-05)、[図 6](#figure-06)に示す。FlashAttention-2 は FlashAttention および `xformers` の FlashAttention（「cutlass」実装）より約 2$\times$ 高速である。FlashAttention-2 は forward pass では Triton の FlashAttention より約 1.3-1.5$\times$、backward pass では約 2$\times$ 高速である。PyTorch の標準的な attention 実装と比べると、FlashAttention-2 は最大 10$\times$ 高速になり得る。

ベンチマーク設定は次のとおりである。系列長を 512、1k、...、16k と変化させ、token の総数が 16k になるよう batch size を設定する。hidden dimension を 2048、head dimension を 64 または 128（すなわち 32 head または 16 head）に設定する。forward pass の FLOP 数の計算には、次を用いる。

$$
4 \cdot \mathrm{seqlen}^2 \cdot \mathrm{head dimension} \cdot \mathrm{number of heads}.
$$

causal mask を用いる場合、計算される要素はおよそ半分にすぎないという事実を考慮して、この数を 2 で割る。backward pass の FLOP 数を得るには、forward pass の FLOP 数に 2.5 を掛ける（再計算のため、forward pass には 2 回、backward pass には 5 回の matmul がある）。

<span id="figure-04"></span>

![A100 GPU におけるアテンションのフォワードとバックワードの速度](../../papers/flashattention-2/figure-04.png)

**図 4。** A100 GPU におけるアテンションのフォワード + バックワード速度

<span id="figure-05"></span>

![A100 GPU におけるアテンションのフォワード速度](../../papers/flashattention-2/figure-05.png)

**図 5。** A100 GPU におけるアテンションのフォワード速度

<span id="figure-06"></span>

![A100 GPU におけるアテンションのバックワード速度](../../papers/flashattention-2/figure-06.png)

**図 6。** A100 GPU におけるアテンションのバックワード速度

新機能を利用するための特別な命令（TMA や第 4 世代 Tensor Core など）を使わず、同じ実装を H100 GPU 上で実行するだけで、最大 335 TFLOPs/s が得られる（[図 7](#figure-07)）。新しい命令を使えば、H100 GPU 上でさらに 1.5x-2x の高速化が得られると期待する。これは今後の課題とする。

<span id="figure-07"></span>

![H100 GPU におけるアテンションのフォワードとバックワードの速度](../../papers/flashattention-2/figure-07.png)

**図 7。** H100 GPU におけるアテンションのフォワード + バックワード速度

<span id="section-4-2"></span>

### 4.2 エンドツーエンド性能

8$\times$A100 80GB SXM 上で、1.3B または 2.7B の parameter を持つ GPT 型モデルの学習スループットを測定する。[表 1](#table-01)に示すように、FlashAttention-2 は FlashAttention を用いないベースラインに対して 2.8$\times$、FlashAttention-2 に対して 1.3$\times$ の高速化をもたらし、A100 GPU 1 基当たり最大 225 TFLOPs/s に達する。

Megatron-LM [Sho19]（および他の多くの論文やライブラリ）にならい、次の式で FLOP 数を計算することに注意されたい。

$$
6 \cdot \mathrm{seqlen} \cdot \mathrm{number of params} + 12 \cdot \mathrm{number of
    layers} \cdot \mathrm{hidden dim} \cdot \mathrm{seqlen}^2.
$$

第 1 項は weight-input multiplication による FLOP 数を、第 2 項は attention による FLOP 数を表す。しかし、causal mask を用いる場合、attention で計算する必要がある要素数はおよそ半分にすぎないため、第 2 項は半分にすべきだとも論じられる。一貫性を保つため、本稿では文献の式（attention の FLOP 数を 2 で割らない式）に従う。

<span id="table-01"></span>

![A100 GPU における GPT 型モデルの学習速度](../../papers/flashattention-2/table-01.png)

**表 1。** 8$\times$A100 GPU における GPT 型モデルの学習速度（TFLOPs/s/GPU）。FlashAttention-2 は最大 225 TFLOPs/s（モデル FLOPs 利用率 72%）に達する。FlashAttention を用いずに実行するベースラインと比較する。

<span id="section-5"></span>

## 5 議論と今後の方向性

FlashAttention-2 は FlashAttention より 2$\times$ 高速であり、これは、以前に 8k のコンテキストを持つモデルを学習していたのと同じ費用で、16k のより長いコンテキストを持つモデルを学習できることを意味する。これが長い書籍やレポート、高解像度画像、音声、動画の理解にどのように活用できるかを楽しみにしている。FlashAttention-2 は、既存モデルの学習、finetuning、推論も高速化する。

近い将来、FlashAttention をさまざまな種類のデバイス（たとえば H100 GPU、AMD GPU）や FP8 などの新しい data type に広く適用できるよう、研究者やエンジニアと協力する予定である。直近の次のステップとして、新しいハードウェア機能（TMA、第 4 世代 Tensor Core、fp8）を用いるよう FlashAttention-2 を H100 GPU 向けに最適化する予定である。FlashAttention-2 の low-level 最適化と high-level のアルゴリズム変更（たとえば local、dilated、block-sparse attention）を組み合わせることで、はるかに長いコンテキストを持つ AI モデルを学習できる可能性がある。また、これらの最適化手法を容易にプログラムできるようにするため、compiler の研究者と協力することも楽しみにしている。

## 謝辞

Triton [Til19] と `xformers` ライブラリ [Lef22] で FlashAttention の各バージョンを実装した Phil Tillet と Daniel Haziza に感謝する。FlashAttention-2 は、attention を実装するさまざまな方法の間で交わされたアイデアから着想を得た。Nvidia CUTLASS チーム（特に Vijay Thakkar、Cris Cecka、Haicheng Wu、Andrew Kerr）には、その CUTLASS ライブラリ、なかでも FlashAttention-2 の実装に簡潔な抽象化と強力な構成要素を提供する CUTLASS 3.x リリースについて感謝する。FlashAttention を PyTorch に統合した Driss Guessous に感謝する。FlashAttention-2 は、Phil Wang、Markus Rabe、James Bradbury、Young-Jun Ko、Julien Launay、Daniel Hesslow、Michaël Benesty、Horace He、Ashish Vaswani、Erich Elsen との有益な議論から恩恵を受けた。計算資源の支援を提供した Stanford CRFM と Stanford NLP に感謝する。ハードウェア効率の高いアルゴリズムを設計するこの一連の研究における協力、建設的なフィードバック、絶え間ない励ましについて、Dan Fu と Christopher Ré に感謝する。このテクニカルレポートの初期草稿に有益な提案を寄せた Albert Gu と Beidi Chen に感謝する。


[+1]: FlashAttention-2 は <https://github.com/Dao-AILab/flash-attention> で公開されている

[+2]: 説明を明確にするため、$\mathbf{Q}\mathbf{K}^\top$ のスケーリング（通常は $1/\mathrm{d}$）と、$\mathbf{S}$ への任意の要素単位のマスキング、および／または $\mathbf{P}$ に適用する dropout は省略している

[+3]: <https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py>
