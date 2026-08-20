---
title: 'FlashAttention-4'
createTime: 2026/08/20 17:49:32
permalink: /ja/papers/flashattention-4/
pageClass: paper-reading
---

> [Ted Zadouri](https://tedzadouri.com/) [+equal]、[Markus Hoehnerbach](https://www.linkedin.com/in/markus-h%C3%B6hnerbach-0b6391166) [+equal]、[Jay Shah](https://developer.nvidia.com/blog/author/jayshah/) [+equal]、[Timmy Liu](https://www.linkedin.com/in/jian-timmy-liu)、[Vijay Thakkar](https://cse.gatech.edu/people/vijay-thakkar)、[Tri Dao](https://tridao.me/)。2026 年 3 月 5 日に arXiv へ初投稿。現行版は v1。[Proceedings of Machine Learning and Systems 8（MLSys 2026）](https://proceedings.mlsys.org/paper_files/paper/2026/hash/ae8b0b5838ba510daff1198474e7b984-Abstract-Conference.html) に掲載。[FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling](https://arxiv.org/abs/2603.05451v1)。[原論文 PDF](/paper/flashattention-4.pdf)。[DOI](https://doi.org/10.48550/arXiv.2603.05451)。[TeX ソース](https://export.arxiv.org/e-print/2603.05451v1)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 要旨

広く使われる Transformer アーキテクチャの中核層である attention は、大規模言語モデルと長文脈アプリケーションのボトルネックである。FlashAttention-3 は非同期実行と warp specialization によって Hopper GPU 向けに attention を最適化したが、主として H100 アーキテクチャを対象としている。AI 業界では B200 や GB200 などの Blackwell ベースのシステムへの移行が急速に進んでいる。これらは、ハードウェアの非対称なスケーリングにより、根本的に異なる性能特性を示す。Tensor Core のスループットが 2 倍になる一方で、共有メモリ帯域幅や指数演算ユニットなどの機能ユニットは、より緩やかにしか向上しないか、まったく変わらない。Blackwell GPU で変化したボトルネックに対処するため、我々は複数の手法を開発した。（1）完全非同期 MMA 演算と大きな tile を活用するよう再設計したパイプライン、（2）非 matmul 演算を減らすソフトウェア実装の指数関数と条件付き softmax rescaling、（3）Tensor Memory と 2-CTA MMA モードを活用し、backward pass における共有メモリトラフィックと atomic add を削減する手法である。我々の手法 FlashAttention-4 は、BF16 を用いる B200 GPU 上で cuDNN 9.13 に対して最大 1.3$\times$、Triton に対して最大 2.7$\times$ の高速化を達成し、最大 1613 TFLOPs/s（使用率 71%）に到達する。アルゴリズム上の工夫に加え、FlashAttention-4 は Python に埋め込まれた CuTe-DSL だけで実装されており、完全な表現力を保ったまま、従来の C++ template ベースの手法より 20-30$\times$ 高速にコンパイルできる。

## 1 はじめに

Transformer アーキテクチャ [Vas17] は、大規模言語モデル [Bro20]、vision [Dos20]、multimodal system に至るまで、ほぼすべての AI アプリケーションで主要な backbone であり続けている。Transformer では attention mechanism が主要な計算ボトルネックであり、query と key の間で計算する self-attention score は系列長に対して二次で増加する。Attention をより長い文脈へ拡張すると、複数文書にまたがる推論 [Guo21a, Sha22a]、コードベース全体のモデリング [Roz23]、高解像度動画の処理 [Che22a, Ho22] などの新しい能力が得られる。一方、accelerator hardware は急速に進化を続けており [Nvi24d]、世代ごとにピーク計算スループットが大きく向上している。しかし、その進化は非対称である。行列乗算ユニットが急速にスケールする一方、メモリ帯域幅や専用計算ユニットなどの機能ユニットはより緩やかにしかスケールしないため、ハードウェアパイプラインはますます不均衡になる。この状況では、アルゴリズムとハードウェアを慎重に協調設計する必要がある。

このため、GPU hardware の特性を深く取り込んだアルゴリズム上の工夫によって attention を高速化する研究が継続している。Dao ら [Dao22] は FlashAttention を提案し、新しい tiling と kernel fusion によって低速な global memory への中間結果の read/write を除去した。Dao [Dao23b] はこれを FlashAttention-2 として再構成し、系列長方向に並列化することで GPU occupancy を改善した。Shah ら [Sha24b] はさらに、warp specialization による非同期実行と FP8 対応を取り入れ、アルゴリズムを Hopper GPU 向けの FlashAttention-3 へ適応した。近年は低精度 attention も研究されている。SageAttention [Lin24d] は INT8 quantization によって高速化し、SageAttention2 [Lin24e] は INT4/FP8 quantization へ拡張し、SageAttention3 [Lin25e] は Blackwell consumer GPU 上で FP4 quantization を示した。しかし、これらの手法は主に consumer GPU を対象としている一方、AI 計算の大部分は datacenter GPU 上に配備されている。また、FlashAttention-3 は NVIDIA Hopper H100 アーキテクチャを主な対象とするが、AI 業界では B200 や GB200 など、根本的に異なる性能特性を持つ新世代の Blackwell datacenter system [Nvi24d] への移行が急速に進んでいる。

Accelerator の進化における重要な傾向は、hardware unit の非対称なスケーリングである。Blackwell B200 は Hopper H100 と比べて Tensor Core のスループットが 2 倍（FP16/BF16 で 2.25 PFLOPS 対 1 PFLOPS）になるが、共有メモリ帯域幅、指数演算ユニット、整数／浮動小数点 ALU などの機能ユニットは、より緩やかにしかスケールしないか、変化しない。その結果、非 MMA リソースがボトルネックになる。我々の roofline analysis（[3.1 節](#section-03-01)および[3.2 節](#section-03-02)）によれば、Blackwell 上の典型的な attention workload では、意外にも共有メモリトラフィックと指数演算が実行時間を支配し、MMA 計算を 25-60% 上回る。さらに、Blackwell には新しいアーキテクチャ機能がある。Tensor Core の中間結果を格納する SM あたり 256 KB の Tensor Memory（TMEM）、Hopper の $64 \times 128$ の 2 倍の面積を持つ $128 \times 128$ MMA tile、そして TMEM へ直接書き込む完全非同期 Tensor Core 演算である。既存の attention algorithm を単純にこの hardware へ移植すると、多くの性能を取りこぼすか、Hopper MMA 命令に forward compatibility がないために移植そのものが不可能になる。

そこで我々は FlashAttention-4 を提案する。これは、現代の GPU アーキテクチャで変化したボトルネックに対応するため、アルゴリズムと kernel 実装を協調設計するものである。Hardware を一様な計算資源として扱うのではなく、非 matmul unit のボトルネックを明示的に特定し、アルゴリズム上の工夫で緩和する。

1. **最大限の overlap を実現する再設計パイプライン：** Blackwell の完全非同期 MMA 演算と大きな tile を活用し、Tensor Core、softmax 計算、メモリ演算の overlap を最大化する forward pass と backward pass の新しい software pipeline を開発する。

2. **指数演算ユニットのボトルネック緩和：** Forward pass では、FMA unit 上の多項式近似によって指数関数を software emulation し、指数演算のスループットを高める。また、不要な rescaling を省略する conditional softmax rescaling を導入する。

3. **共有メモリトラフィックの削減：** Backward pass では Tensor Memory により多くの中間結果を格納し、共有メモリトラフィックを減らす。さらに Blackwell の 2-CTA MMA モードを利用し、各 CTA が operand B の半分だけを stage・load することで共有メモリトラフィックをさらに減らす。これを利用して dQ step を再構成し、atomic reduction の回数を半減させる。また、性能 overhead の小さい deterministic execution mode を実装し、reinforcement learning application で再現可能な学習を可能にする。

4. **Scheduling と resource allocation の改善：** Blackwell の resource constraint と大きな tile に合わせ、新しい CTA scheduling strategy と register allocation scheme を開発する。

アルゴリズム上の工夫に加え、FlashAttention-4 は Python に埋め込まれた CuTe-DSL だけで実装されており、完全な表現力を保ったまま、従来の C++ template ベースの手法より 20-30$\times$ 高速にコンパイルできる。この framework は開発生産性を大幅に高め、参入障壁を下げるため、C++ template metaprogramming に深い専門知識を持たない研究者でも、新しい attention variant を素早く prototype・deploy できる。

我々の手法を実験的に検証するため、B200 GPU 上で FlashAttention-4 を benchmark し、次を示す。（1）BF16 は cuDNN に対して最大 1.3$\times$、Triton 実装に対して最大 2.7$\times$ 高速である。（2）移行後のボトルネックリソースでピークに近い使用率を達成し、最大 $\sim$1600 TFLOPS（理論最大値の 71%）に到達する。（3）長い系列では FlashAttention-4 が他の attention 実装を上回る。

我々は FlashAttention-4 を permissive license で open source 化し、より多くの研究者と開発者が利用できるよう、一般的な library への統合を進めている。コードは [https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn/cute) で公開している。

## 2 背景

### 2.1 Multi-Head Attention

単一の head に対応する query、key、value の入力系列を $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{N \times d}$ とする。ここで $N$ は系列長、$d$ は head dimension である。Attention の出力 $\mathbf{O} \in \mathbb{R}^{N \times d}$ は次のように計算する。

$$
\begin{aligned}
\mathbf{S} &= \alpha \mathbf{Q} \mathbf{K}^\top \in \mathbb{R}^{N \times N}, \\
\mathbf{P} &= \mathrm{softmax}(\mathbf{S}) \in \mathbb{R}^{N \times N}, \\
\mathbf{O} &= \mathbf{P}\mathbf{V} \in \mathbb{R}^{N \times d},
\end{aligned}
$$

ここで $\mathrm{softmax}$ は行ごとに適用し、$\alpha = 1/\sqrt{d}$ は scaling factor である。実際には数値安定性のため、$\mathbf{S}$ から $\mathrm{rowmax}(\mathbf{S})$ を減算する。Multi-head attention（MHA）では各 head が独自の projection を持ち、この計算は複数の head と batch にまたがって並列化できる。

出力勾配 $\mathbf{dO} \in \mathbb{R}^{N \times d}$ が与えられると、backward は次を計算する。

$$
\begin{aligned}
\mathbf{dV} &= \mathbf{P}^\top \mathbf{dO}, \quad \mathbf{dP} = \mathbf{dO} \mathbf{V}^\top, \\
\mathbf{dS} &= \mathrm{dsoftmax}(\mathbf{dP}), \\
\mathbf{dQ} &= \alpha \mathbf{dS} \mathbf{K}, \quad \mathbf{dK} = \alpha \mathbf{dS}^\top \mathbf{Q},
\end{aligned}
$$

ここで $\mathrm{dsoftmax}(\mathbf{dP})$ は行ごとの softmax gradient を表し、$p = \mathrm{softmax}(s)$ に対して $\mathbf{d}s = (\mathrm{diag}(p) - p p^\top)\mathbf{d}p$ である。

### 2.2 GPU Hardware の特性と実行モデル

FlashAttention-4 に関係する GPU execution model について、NVIDIA Blackwell アーキテクチャ（B200 および GB200）を中心に説明する。FlashAttention-4 の最適化を動機づけた、先行する Hopper アーキテクチャとの主な違いを示す。

**Memory hierarchy：** GPU memory は data locale の階層として構成され、容量と帯域幅は反比例する。Global memory（GMEM）は HBM とも呼ばれ、すべての streaming multiprocessor（SM）からアクセスできる off-chip DRAM である。GMEM からの data は on-chip L2 cache に透過的に cache される。次に、各 SM には容量が小さく、programmer-managed で、多数の bank を持つ on-chip cache である shared memory（SMEM）がある。最後に、各 SM 内に register file がある。

Blackwell は *Tensor Memory*（TMEM）という新しい memory level を導入する。これは SM あたり 256 KB の on-chip memory で、Tensor Core 演算の中間結果を格納するために設計されている。Shared memory とは異なり、TMEM は warp-synchronous で Tensor Core と密接に結合しており、matrix multiply-accumulate（MMA）unit は register を消費せずに出力を TMEM へ直接書き込める。これにより Hopper kernel の深刻な register pressure が緩和され、より大きな tile が可能になる。TMEM は 32 column（16 KB）単位で割り当てられ、allocation、deallocation、data movement を programmer が明示的に管理する必要がある。

**Thread hierarchy：** GPU programming model は thread と呼ばれる論理的な execution unit のグループを中心に構成される。細かい順に、thread、warp（32 thread）、warpgroup（連続する 4 warp）、threadblock（cooperative thread array、すなわち CTA）、threadblock cluster、grid から成る。同じ CTA の thread は同じ SM 上に co-schedule され、同じ cluster の CTA は同じ GPC 上に co-schedule される。SMEM は CTA 内のすべての thread から直接 address できる一方、各 thread は最大 256 個の private register（RMEM）を持つ。

**Tensor Core と高い非同期性：** Blackwell は従来のアーキテクチャより大幅に大きな tile を扱う第 5 世代 Tensor Core を備える。各 MMA Tensor Core 命令は $128 \times N$ tile（通常 $N =$ 128 または 256）を処理し、Hopper の $64 \times N$ と対照的である。重要なのは、Hopper MMA が register に書き込むのに対し、Blackwell MMA は出力を TMEM へ非同期に直接書き込む点である。MMA unit が register writeback で block されないため、この完全な非同期性によって計算と他の演算をよりよく overlap できる。

非同期実行の hardware support により、warp-specialized kernel を構成できる。CTA の warp を producer または consumer role に分け、data movement か計算の一方だけを発行させる [Bau11]。

**2-CTA Tensor Core：** Blackwell は 2-CTA Tensor Core MMA モードをサポートする。同じ thread block cluster 内の CTA pair が 1 回の MMA を協調実行し、両方の CTA の Tensor Memory を読み書きできる。Pair の一方の thread が MMA を開始するが、peer CTA は開始済みで、演算中も active でなければならない。Single-CTA MMA では M dimension が 128 に制限されるのに対し、paired mode は M = 128 または 256 をサポートする。A tile と accumulator を M dimension に沿って pair 間で分割し、B tile を N dimension に沿って 2 CTA 間で分割するため、各 CTA は B の半分だけを自身の shared memory に stage すればよく、hardware は乗算時に結合された B tile を使う。これにより冗長な shared memory capacity と bandwidth を削減できる。しかし、これらの演算は CTA pair をまたいで Tensor Memory に触れるため、kernel は CTA を固定 pair で起動し、kernel 全体で Tensor Memory 演算と Tensor Core 演算に一貫した 2-CTA mode を使わなければならない。

**移行するボトルネック：** Blackwell に表れている重要な傾向は、Tensor Core のスループットが他の機能ユニットより速くスケールすることである。Blackwell は Hopper と比べて FP16/BF16 Tensor Core スループットを 2 倍にする（GPU あたり 2.25 PFLOPS [Nvi24d] 対 1 PFLOPS [Nvi22]）が、共有メモリ帯域幅と指数演算ユニットのスループットは変わらないか、より緩やかにしか向上しない。この不均衡により、性能ボトルネックは行列乗算から共有メモリトラフィックや softmax などの非 matmul 演算へ移る。[3.1 節](#section-03-01)および[3.2 節](#section-03-02)の roofline analysis が示すように、MMA 演算とこれらのボトルネックリソースの overlap を最大化するよう、kernel を慎重に設計する必要がある。

B200（および GB200）の複数の hardware component のスループットを以下に示す。

1. Tensor Core：BF16 MMA のスループットは SM あたり 1 clock に 8192 ops であり、Hopper の 4096 ops / clock / SM の 2 倍である。これは理論最大 FLOPS から、2.25 PFLOPS / 1850 Mhz clock speed / 148 SM = 8192 ops / clock / SM と導出できる。
2. Exponential unit：B200 と GB200 の multifunction unit（MUFU）は 16 ops / clock / SM を実行でき、Hopper と同じである [Nvi24c]。B300 と GB300 GPU では指数スループットが 32 ops / clock / SM へ倍増しているが、執筆時点ではまだ広く利用されていない。
3. SMEM：read throughput は 128 bytes / clock / SM であり、microbenchmark [Luo25h] で測定した Hopper と同じ値である。

Blackwell の MMA スループットは Hopper と比べて 2 倍になったが、他の hardware unit が同じ割合で高速化するとは限らない。これは accelerator design の一般的な傾向を反映している。すなわち、同程度の power／silicon area constraint の下で高い性能を得るため、最も重要な component（通常は matrix multiply unit）のスループットを高める設計である。

## 3 アルゴリズム

<span id="section-03-01"></span>

### 3.1 Attention forward pass

まず roofline analysis によって attention forward pass のボトルネックを示す。この分析が、新しい pipeline design、指数演算ユニットのスループットを高める FlashAttention algorithm の変更、softmax rescaling の大部分を回避する変更の動機になる。

#### 3.1.1 Feeds and Speeds

Kernel design と最適化の直観を示すため、matmul unit（Tensor Core）、shared memory（SMEM）、exponential unit のスループットに基づく roofline analysis を行う。これは GPU のすべてのリソース（たとえば floating point math、register bandwidth、L2 bandwidth）を考慮しない簡略化した分析である。それでもボトルネックを特定できる。

$\mathbf{Q}$ と $\mathbf{K}$ の系列長方向における tile shape を $M \times N$、head dimension を $d$ とする。性能ボトルネックを特定するため、計算量と memory traffic の要件を分析する。

**MMA compute：** Forward pass は各 iteration で 2 回の matrix multiply-accumulate（MMA）を行う。$\mathbf{Q}\mathbf{K}^\top$ は $M \times d$ と $d \times N$ の入力から $M \times N$ の出力を計算し、$\mathbf{P}\mathbf{V}$ は $M \times N$ と $N \times d$ の入力から $M \times d$ の出力を計算する。各 MMA は $2MNd$ floating-point operation を必要とする。Tensor Core のスループットを 1 cycle あたり 8192 FLOPs とすると、総計算時間は

<span id="equation-01"></span>

$$
T_{\mathrm{MMA}} = \frac{4MNd}{8192} \mathrm{\ cycles}.
$$

**Shared memory traffic：** 2 回の MMA のうち一方は shared-shared（SS）で、両 operand を shared memory から読み出す（$\mathbf{Q}\mathbf{K}^\top$）。もう一方は tensor-shared（TS）で、operand $A$ を Tensor Memory から、operand $B$ を shared memory から読み出す（$\mathbf{P}\mathbf{V}$）。各 MMA 命令は $128 \times 128$ tile を処理するため、$M \times N$ の出力には $\lceil M/128 \rceil \times \lceil N/128 \rceil$ 個の MMA 命令が必要である。重要なのは、複数の MMA 命令が必要な場合、shared memory operand が複数回 read されることである。

$\mathbf{Q}\mathbf{K}^\top$（SS）で $M \times N$ の出力を計算するには $\lceil M/128 \rceil \times \lceil N/128 \rceil$ 個の MMA 命令が必要であり、各命令は shared memory から $\mathbf{Q}$ の $128 \times d$ chunk と $\mathbf{K}^\top$ の $d \times 128$ chunk を読み出す。Shared memory の総 read 数は $\lceil M/128 \rceil \times \lceil N/128 \rceil \times (128d + 128d) = \lceil M/128 \rceil \lceil N/128 \rceil \times 256d$ element である。$\mathbf{P}\mathbf{V}$（TS）で $M \times d$ の出力を計算するには $\lceil M/128 \rceil \times \lceil d/128 \rceil$ 個の MMA 命令が必要であり、各命令は shared memory から $\mathbf{V}$ の $N \times 128$ chunk を読み出す。合計は $\lceil M/128 \rceil \times \lceil d/128 \rceil \times 128N$ element である。1 element は 2 byte（bf16）、bandwidth は 1 cycle あたり 128 byte なので、shared memory の read 時間（$T_{\mathrm{smem}}$）は

<span id="equation-02"></span>

$$
= 2\Big\lceil\tfrac{M}{128}\Big\rceil\Big\lceil\tfrac{N}{128}\Big\rceil 256d
+ 2\Big\lceil\tfrac{M}{128}\Big\rceil\Big\lceil\tfrac{d}{128}\Big\rceil 128N
= \tfrac{3MNd}{8192}\ \mathrm{cycles}
$$

となる（$M$、$N$、$d$ が 128 の倍数であると仮定する）。

**Exponential unit：** Exponential unit は softmax 計算に必要な elementwise operation を実行する。Forward pass では $M \times N$ 個の値（attention matrix $\mathbf{S}$ に対応）に対して exponential operation が必要である。スループットは 1 cycle あたり 16 operation なので、必要時間は

<span id="equation-03"></span>

$$
T_{\mathrm{exp}} = \frac{MN}{16} \mathrm{\ cycles}.
$$

[表 1](#table-01) は 2 つの典型的な tile configuration の分析をまとめている。$M = N = d = 128$ の場合、resource はよく均衡しており、shared memory は 768 cycle、MMA compute と exponential unit はともに 1024 cycle である。より大きな $M = 256, N = d = 128$ の tile では、MMA operand を複数回読み出すため shared memory traffic は 1536 cycle に増え、MMA compute と exponential unit は 2048 cycle に倍増する。この分析から、kernel design には（1）大きな tile を使い MMA operation と softmax computation の overlap を最大化すること、（2）他の hardware unit を使って exponential throughput を高めること、（3）不要な非 matmul operation の時間を減らすことが必要だと分かる。

<span id="figure-01"></span>

![FlashAttention-4 forward pipeline](../../papers/flashattention-4/figure-01.png)

**図 1.** FlashAttention-4 forward pipeline。上付きの $^H$ は「high」Q tile に対応する行列を、$^L$ は「low」Q tile に対応する行列を表す。各 Q tile は 128 query token に対応する。

<span id="table-01"></span>

![原論文の表 1](../../papers/flashattention-4/table-01.png)

**表 1.** Attention forward pass の roofline analysis（cycle 数）。どちらの tile size でも、MMA compute と exponential unit が主要なボトルネックである。

#### 3.1.2 Matmul と softmax を overlap する新しいパイプライン

Blackwell アーキテクチャでは Tensor Core FLOPS が再び 2 倍になったため、softmax と Tensor Core operation を overlap させることは Hopper の場合以上に重要である。我々は FA-3 と同様の ping-pong schedule を用い、thread block ごとに 2 つの output tile を計算する。一方の tile で Tensor Core operation を実行している間、もう一方の tile で softmax を計算する。Hopper Tensor Core は accumulator を register に保持し、1 行を 4 thread が interleaved pattern で処理するのに対し、Blackwell Tensor Core は accumulator を Tensor Memory に保持する。また、Blackwell の単一 accumulator tile は 128×128 element であり、Hopper の tile size は 64×128 である。

これらの tile に作業を分配する自然な方法は、各 128 thread の 2 つの warpgroup を用い、各 thread が 1 行全体を処理することである。これにより row max を reduction する inter-warp shuffle と、thread ごとの複数の statistics register が不要になる。FA-3 と同様に、2 つの softmax warpgroup が exponential computation の critical section で overlap しないよう明示的に同期する。各 softmax warpgroup は、まず行全体を register に load し、maximum を計算し、softmax（max の減算、rescale、exponentiate、input precision への変換）を計算し、最後に row sum を計算する。

FA-3 とのもう 1 つの違いは、$\mathbf{P}$ を register file ではなく Tensor Memory 経由で転送するため、output rescaling を別の「correction」warpgroup に分離し、critical path から外せることである。

この pipeline overlap を実現する Tensor Memory の partition は複数考えられる。いずれも output 2 tile 分を割り当てる必要があり、head dimension 128 では残り半分の Tensor Memory に $\mathbf{S}$ と $\mathbf{P}$ を格納する。その領域には $\mathbf{S}$ を 2 copy、または $\mathbf{P}$ を 4 copy 格納できる（FP16 または BF16 Tensor Core の input を仮定）。したがって、残りの Tensor Memory には概ね 2 つの選択肢がある。$\mathbf{S}$ 1 tile と $\mathbf{P}$ 2 tile、または $\mathbf{P}$ と overlap する $\mathbf{S}$ 2 tile である。我々は後者を選ぶ。Software pipeline の開始時に 2 つの $\mathbf{S}$ tile を直ちに計算できるためである。また、rescale statistics を correction warpgroup へ渡す Tensor Memory も残せる。

Blackwell の tile size が大きく、上記の thread assignment を採ると、Tensor Memory から reload しない限り、128 element の 1 行全体を register に保持しなければならない。2 つの softmax warpgroup、1 つの correction warpgroup、Tensor Core と TMA unit を駆動する 1 つの warpgroup を使うため、softmax に十分な register を割り当て、register spill を防ぐことが重要である。BF16 input data type では input に 128 register、output に最大 64 register が必要になる（このほか miscellaneous register と temporary register がある）。Register pressure を減らすため、$\mathbf{P}$ の store を段階化する。最初の 3 quarter を一度 store して対応する MMA operation を trigger し、最後の quarter は別に store する。

#### 3.1.3 指数関数のエミュレーション

**指数スループットのボトルネック：** 現代の GPU では指数関数を multi-function unit（MUFU）が計算するが、そのスループットは行列乗算に使う Tensor Core より大幅に低い。B200 および GB200 GPU では、MUFU は 16 operations / clock / SM、行列乗算は 8192 operations / clock / SM である。Softmax は多数の指数評価を必要とするため、この差によって指数関数が attention kernel の重大なボトルネックになる。

**多項式近似による software emulation：** 指数スループットを高めるため、MUFU と並列動作できる floating-point FMA unit を用いて $2^x$ を software emulation する。古典的な range reduction（Cody-Waite）と多項式近似 [Mul18] を使う。中心となる考えは、指数計算を次のように分解することである。

<span id="equation-04"></span>

$$
2^x = 2^{\lfloor x \rfloor}\,2^{x-\lfloor x \rfloor}
$$

ここで $\lfloor x \rfloor$ は整数部、$x - \lfloor x \rfloor \in [0, 1)$ は小数部である。

整数部 $2^{\lfloor x \rfloor}$ は IEEE 754 floating-point representation の bit 操作で効率よく計算できる。Exponent field は 2 の冪を直接表すため、$2^{\lfloor x \rfloor}$ の計算は exponent bit の shift と add に相当し、integer ALU instruction で実行できる。

小数部については、$x_{\mathrm{frac}} \in [0, 1)$ に対する $2^{x_{\mathrm{frac}}}$ を次の多項式で近似する。

<span id="equation-05"></span>

$$
2^{x_{\mathrm{frac}}} \approx \sum_{i=0}^{n} p_i\, x_{\mathrm{frac}}^i
$$

$p_0 = 1.0$ とし、残りの係数は Sollya software package [Che10] により、$\lbrack 0, 1)$ 上の relative approximation error を最小にするよう選ぶ。多項式は Horner’s method と FMA instruction で評価し、高いスループットを実現する。

完全なアルゴリズムは次のとおりである。

- Underflow を避けるため、$x$ を最低 $-127$ に clamp する。
- Round-down mode で $\lfloor x \rfloor$ を計算する。$x$ に $2^{23} + 2^{22}$ を加えて小数 bit を mantissa へ押し込み、その値を round-down mode で減算する。
- 小数部 $x_{\mathrm{frac}} = x - \lfloor x \rfloor$ を計算する。
- 多項式を評価し、$2^{x_{\mathrm{frac}}}$ を得る。
- 整数部と小数部を結合する。$\lfloor x \rfloor$ を exponent field へ shift し、$2^{x_{\mathrm{frac}}}$ の mantissa bit を加える。

指数計算を MUFU と FMA unit の両方へ分配することで、実効指数スループットが向上し、attention computation の重要なボトルネックが緩和される。

**部分的なエミュレーション：** 多項式 emulation は指数スループットを高める一方、中間値と係数を保持する追加 register、より高い register bandwidth consumption、MUFU instruction より長い latency というコストがある。すべての指数評価に emulation を使うと register pressure が増加し、spill によってスループットの利点が失われる可能性がある。そこで、各 softmax row の一部（10-25%）だけに emulation を適用し、残りは hardware `MUFU.EX2` で計算する。正確な割合は、与えられた tile configuration の MMA throughput と exponential throughput の比に基づき経験的に調整する。

**数値精度：** [表 2](#table-02) は、$[0, 1)$ から得た 4M 個の random input に対し、次数の異なる polynomial approximation と hardware `MUFU.EX2` instruction の精度を比較する。2 つの metric、すなわち quantization 前の FP32-level error と、FP32 output を BF16 へ丸めた後の BF16-level error を、いずれも FP64 reference に対して報告する。

FP32 level では、degree-3 polynomial の maximum relative error は $8.8 \times 10^{-5}$ であり、hardware のおよそ $600\times$ である。しかし BF16 へ丸めると error はほぼ区別できなくなる。BF16 quantization error（${\sim}3.9 \times 10^{-3}$）が、すべての degree $\geq 3$ の polynomial approximation error を上回るためである。Degree-3 polynomial は 99% の input で hardware と 1 BF16 ULP 以内に一致する。Softmax output は BF16 precision で消費されるため、attention computation には十分である。高次の polynomial は FP32 の差を縮め、degree 5 では maximum relative error が hardware の $2\times$ 以内になるが、評価あたり 2 個の FMA instruction が追加で必要になる。

<span id="table-02"></span>

![原論文の表 2](../../papers/flashattention-4/table-02.png)

**表 2.** $[0, 1)$ 上の $2^x$ polynomial emulation の精度。4M 個の random input について FP64 reference と比較した。FP32 column は polynomial の raw output、BF16 column は BF16 へ丸めた後を測定する。すべての degree $\geq 3$ で BF16 quantization error が支配的である。

#### 3.1.4 Online softmax rescaling の省略

**FlashAttention online softmax：** FlashAttention は memory traffic を最小化するため、attention $\mathrm{softmax}(QK^\top)V$ を block 単位で計算する。数値安定性のため、block を処理しながら running statistic を維持する。Block $j$ を計算するとき、その block の attention score を $S_j = Q K_j^\top$ とする。Online softmax algorithm は次を追跡する。

$$
\begin{aligned}
m_j &= \max(m_{j-1}, \mathrm{rowmax}(S_j)) \\
\ell_j &= e^{m_{j-1} - m_j} \ell_{j-1} + \mathrm{rowsum}(e^{S_j - m_j})
\end{aligned}
$$

ここで $m_j$ は running max、$\ell_j$ は exponential の running sum（normalizer）である。中間出力 $O_j$ は $O_j = e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j.$ と更新する。Rescaling factor $e^{m_{j-1} - m_j}$ は、より大きな値が現れたときに以前の結果を renormalize し、数値安定性を保証する。

**Conditional rescaling：** $e^{m_{j-1} - m_j} O_{j-1}$ の step は vector multiplication を必要とする。我々は 2 つの単純な点に着目する。

1. Rescaling が必要なのは、より大きな値が新たに見つかり $m_j > m_{j-1}$ となる場合だけである。
2. Rescaling にはある程度の「slack」を許容できる。$m_j - m_{j-1} > \tau$ の場合だけ rescale する。Threshold $\tau$ は通常 $\log_2(256) = 8.0$ とし、256.0 の rescaling factor に対応する。Statistics（これまでに行った total scaling）を追跡すれば、最後に真の denominator を得て、正しい final output を計算できる。

FlashAttention-4 ではアルゴリズムを次のように変更する。

<span id="equation-06"></span>

$$
O_j = \begin{cases}
e^{m_{j-1} - m_j} O_{j-1} + e^{S_j - m_j} V_j & \mathrm{if}\ m_j - m_{j-1} > \tau \\
O_{j-1} + e^{S_j - m_{j-1}} V_j & \mathrm{otherwise}
\end{cases}
$$

$m_j - m_{j-1} \leq \tau$ の場合、$m$ の更新を省略して $m_{j-1}$ を使い続ける。計算の最後に、すべての accumulated value を真の maximum $m_{\mathrm{final}}$ と final normalizer $\ell_{\mathrm{final}}$ で renormalize するため、correctness は保たれる。

$$
\mathrm{Output} = \frac{1}{\ell_{\mathrm{final}}} O_{\mathrm{final}}
$$

この変更は numerical accuracy を保ったまま rescaling operation の回数を大幅に減らす。Final normalization step が、中間 rescaling を省略したことで生じる小さな差を補正するためである。

実装では warp divergence を避けるため、warp 内のいずれかの thread が rescaling を必要とする場合に rescale する。

<span id="section-03-02"></span>

### 3.2 Attention backward pass

#### 3.2.1 Feeds and Speeds

Forward pass と同様に、matmul unit（Tensor Core）、shared memory（SMEM）、exponential unit のスループットに基づく roofline analysis により、kernel design と最適化の直観を示す。

$\mathbf{Q}$ と $\mathbf{K}$ の系列長方向の tile shape を $M \times N$、head dimension を $d$ とする。性能ボトルネックを特定するため、計算量と memory traffic の要件を分析する。Forward pass とは異なり、SMEM cycle 数の式を簡単にするため $M = N = d = 128$ を仮定するが、分かりやすさのため変数名は残す。

**MMA compute：** Backward pass は iteration あたり 5 回の matrix multiply-accumulate（MMA）を実行する。各 MMA は $M \times N$ matrix、$M \times d$ matrix、$d \times N$ matrix を含み（どれが output matrix になるかが異なる）、$2MNd$ floating-point operation を必要とする。Tensor Core のスループットを 1 cycle あたり 8192 FLOPs とすると、総計算時間は

<span id="equation-07"></span>

$$
T_{\mathrm{MMA}} = \frac{10MNd}{8192} \mathrm{\ cycles}.
$$

**Shared memory traffic：** 5 回の MMA のうち 3 回、すなわち $\mathbf{S}^\top = \mathbf{K} \mathbf{Q}^\top$、$\mathbf{dP}^\top = \mathbf{V} \mathbf{dO}^\top$、$\mathbf{dQ} = \mathbf{dS} \mathbf{K}$ は、両 operand を shared memory から読み出す shared-shared（SS）operation である。残る 2 回、$\mathbf{dV} = \mathbf{P}^\top \mathbf{dO}$ と $\mathbf{dK} = \mathbf{dS}^\top \mathbf{Q}$ は、operand $A$ を Tensor Memory から、operand $B$ を shared memory から読み出す tensor-shared（TS）operation である。SS MMA は合計 $2Md + 3Nd + MN$ element、TS MMA は合計 $2Md$ element を shared memory から読み出す。Shared memory bandwidth が 1 cycle あたり 128 byte、各 element が 2 byte（bf16）なので、この部分は

<span id="equation-08"></span>

$$
T_{\mathrm{smem,MMA}} = \frac{4 M d + 3 N d + M N}{64} \mathrm{\ cycles}.
$$

さらに、アルゴリズムは $M \times N$ の中間 gradient $\mathbf{dS}$ を bf16 で shared memory へ書き込み、$2MN$ byte、すなわち $MN/64$ cycle を必要とする。$M \times d$ の gradient $\mathbf{dQ}$ は fp32（1 element あたり 4 byte）で shared memory へ書き込み、reduction のため TMA で読み戻すので、shared memory traffic は合計 $8Md$ byte、すなわち $Md/16$ cycle である。

したがって shared memory の総 access time（$T_{\mathrm{smem}}$）は

<span id="equation-09"></span>

$$
\frac{4 M d + 3 N d + M N}{64} + \frac{MN}{64} + \frac{Md}{16} \mathrm{\ cycles}.
$$

<span id="figure-02"></span>

![FlashAttention-4 backward computation graph](../../papers/flashattention-4/figure-02.png)

**図 2.** FlashAttention-4 backward computation graph（5 MMA operation + 2 elementwise operation）。Prologue、main loop、tail にまたがる 1-CTA MMA mode の software pipeline order を示す。

**Exponential unit：** Exponential unit は softmax とその gradient に必要な elementwise operation（exponential、logarithm、関連する nonlinear function）を実行する。Backward pass では $M \times N$ 個の値（attention matrix $\mathbf{S}$ と関連項）に exponential operation が必要である。スループットは 1 cycle あたり 16 operation なので、必要時間は

<span id="equation-10"></span>

$$
T_{\mathrm{exp}} = \frac{MN}{16} \mathrm{\ cycles}.
$$

[表 3](#table-03) は典型的な tile configuration $M = N = d = 128$ の分析をまとめている。Shared memory traffic の 3328 cycle は、MMA compute の 2560 cycle と exponential unit の 1024 cycle の双方を上回る。したがって shared memory bandwidth が主要なボトルネックであるが、global memory traffic が支配する場合ほど深刻ではない。この結果から、shared memory latency を隠すために MMA operation と他の計算の overlap を最大化する kernel design が必要になる。

<span id="table-03"></span>

![原論文の表 3](../../papers/flashattention-4/table-03.png)

**表 3.** $M = N = d = 128$ における attention backward pass の roofline analysis。Shared memory traffic がボトルネックであり、MMA compute time を約 30% 上回る。$M = 256$、$N = d = 128$ の 2-CTA setting（$\mathbf{dQ}$ MMA だけは $M = N = 128$、$d = 256$）では、shared memory traffic が MMA compute time を約 5% 上回る。

#### 3.2.2 Matmul と softmax を overlap する新しいパイプライン

FlashAttention の backward pass は 5 回の MMA operation を実行する。$\mathbf{S}$ の再計算、$\mathbf{Q} \mathbf{K}$ から生じる 2 つの gradient computation（$\mathbf{dQ}$ と $\mathbf{dK}$）、$\mathbf{P} \mathbf{V}$ から生じる 2 つの gradient computation（$\mathbf{dP}$ と $\mathbf{dV}$）である。FA-3 では accumulator を register に格納するが、register は限られた resource である。この制約は operation order を強く制限し、実質的に compute graph を $\mathbf{S}, \mathbf{dP}, \mathbf{dV}, \mathbf{dQ}, \mathbf{dK}$ の順に serialize する。TMA load だけがこの順序から大きく外れて動作する。これ以外のアルゴリズムは同様である。KV 系列長方向に iterate し、forward pass に対して transpose した値を計算する。$\mathbf{dV}$ と $\mathbf{dK}$ の gradient calculation が operand の 1 つを Tensor Memory から読むために必要な layout だからである。$\mathbf{dQ}$ は atomic operation によって accumulate する。

FA-4 では TMEM により、FA-3 より多くの schedule が可能になり、MMA operation と非 MMA operation を大きく overlap できる。具体的には forward pass と同様に softmax calculation の latency を隠そうとする。FA-3 では softmax computation を $\mathbf{dP}$ の MMA と overlap する。前節の分析から、Blackwell では少なくとも 2 つの MMA operation を同時実行する必要がある。

そこで前 iteration の $\mathbf{dQ}$ MMA と $\mathbf{dK}$ MMA を使う。これには load、MMA、compute、reduction の間で shared memory と Tensor Memory resource を慎重に管理する必要がある。特に、5 つの accumulator tile を格納できるだけの Tensor Memory はない。128×128 element の tile は最大 4 つしか入らず、$\mathbf{dV}$ と $\mathbf{dK}$ は accumulate するため領域を共有できない。我々の実装では、$\mathbf{S}$ と $\mathbf{P}$ が TMEM block の 1 つ（offset 0）を共有し、$\mathbf{dP}$、$\mathbf{dS}$、$\mathbf{dQ}$ がもう 1 つを共有する。[図 2](#figure-02) に FA-4 backward の computational graph を示す。

<span id="figure-03"></span>

![2-CTA backward dQ step](../../papers/flashattention-4/figure-03.png)

**図 3.** 2-CTA backward の $dQ$ step では、CTA pair が DSMEM を使って $dS$ tile の半分を交換する。これにより各 CTA は $(\frac{M}{2} \times 2N)$ operand を形成し、reduction を 2 倍にした CTA-pair UMMA を実行できる。

#### 3.2.3 2-CTA backward pass：共有メモリトラフィックと global atomic add の削減

Pipelining を改善し、10 個の GEMM operand のうち 2 個を Tensor Memory に置いても、shared memory bandwidth は backward pass を支配する。5 回の GEMM では、残る 8 個の BF16 operand を Tensor Core へ供給するため shared memory から load する必要があり、この shared memory traffic は Tensor Core compute より約 30% 多くの cycle を要する。このボトルネックをさらに緩和するため、Blackwell が導入した 2-CTA MMA mode を用い、output accumulator を M dimension で分割する。MMA tile shape が $M=256$、$N=K=128$ のとき、2 つの CTA は 1 つの大きな tile として動作する。各 CTA は operand B の半分を load・stage し、自身の accumulator slice だけを保持する。

**Shared memory traffic：** Backward pass の 5 GEMM では $M=256$、$N=K=128$ の MMA tile shape を用い、operand B の shared memory traffic を概ね半減させる。FlashAttention backward pass では、各 CTA が固定 KV tile を保持し（outer loop を $N$ CTA に並列化）、inner loop で M tile を stream する。$\mathbf{dQ}$ の accumulation は outer loop の KV sequence に沿う reduction である。しかし 2-CTA MMA は output tile だけを分割し、reduction axis は分割しない。また、$\mathbf{dQ}$ MMA の reduction dimension は $N$ であり、CTA pair に自然に分かれている。その結果、各 CTA は自身が担当する row について full reduction を必要とする。Reduction axis 上のこの衝突を解決するため、2 つの CTA が同じ cluster にあることを利用し、distributed shared memory（DSMEM）で dS の半分を交換する。この方法は $\mathbf{dS}$ を non-reduction axis に沿って partition し直す。各 CTA は $\frac{M}{2}$ row と完全な $2N$ reduction を保持する。その結果、CTA ごとの $\mathbf{dQ}$ MMA tile shape は $(\frac{M}{2}, 2N)(2N, d)$ となり、Tensor Memory に $(\frac{M}{2}, d)$ tile を accumulate する。2-CTA MMA mode では、$\mathbf{S}$、$\mathbf{dP}$、$\mathbf{dV}$、$\mathbf{dK}$ の MMA は $M=256$ tile を使い、$\mathbf{dQ}$ は $M=128$ だが reduction を $2N=256$ に倍増する。次に DSMEM latency を隠すため、1-CTA variant に対して software pipeline の順序を変更する。現在の tile の $\mathbf{dP}$ を、前 iteration の tile の $\mathbf{dQ}$ より先に計算する。$\mathbf{dQ}$ tile は $\mathbf{P}$ とともに TMEM に収まるほど小さく、$\mathbf{S}$ と同じ TMEM region を再利用する。このため 1-CTA mode と異なり、$\mathbf{dP}$ と $\mathbf{dQ}$ は同じ TMEM region を再利用しない。この新しい pipelining order では、現在の tile の elementwise $\mathbf{dS}$ と、前 iteration の tile の $\mathbf{dQ}$ MMA を並列に計算できる。[図 3](#figure-03) は $\mathbf{dQ}$ step の分解を示す。

**$\mathbf{dQ}$ atomic add：** この $\mathbf{dQ}$ decomposition には global atomic reduction の回数を半減させる効果もある。Atomic update は nondeterminism をもたらし、inner loop の iteration ごとに行われるため高コストである。その結果、各 CTA は $\mathbf{dQ}$ tile の半分だけを書き込み、global atomic reduction の回数も 1-CTA counterpart の半分になる。

#### 3.2.4 Deterministic backward pass

我々の backward kernel は global memory で inter-CTA reduction を行うため、gradient computation に nondeterminism が生じる（通常は $\mathbf{dQ}$、GQA では $\mathbf{dK}$/$\mathbf{dV}$ にも影響する）。学習の reproducibility を保証し、信頼できる debugging を可能にするため、deterministic execution mode も提供する。標準的な解決策は semaphore lock で global reduction を serialize することであり、我々もこれを採用する。具体的には、共通の $\mathbf{dQ}$ tile に書き込む各 CTA が定められた順序で lock を取得し、reduction を行い、semaphore counter を increment して lock を解放する。

この lock-based approach は主に 2 つの理由で性能へ影響する。（1）semaphore write の device-wide visibility を保証するため memory fence を発行する必要がある（正しい acquire-release semantics に必要）。（2）各 CTA が共通の $\mathbf{dQ}$ tile を reduction する先行 CTA の完了を待つため stall が生じる。Load imbalance がある場合、単純な CTA order は性能を著しく低下させる。一般には head dimension と batch dimension で CTA swizzling を行い、L2 cache capacity の範囲内で stall を減らす（[3.3 節](#section-03-03)を参照）。Causal masking ではさらに、KV block を降順に launch し、diagonal から query block を昇順に traverse し、$\mathbf{dQ}$ reduction を query block index の降順に並べる。この「shortest-processing-time-first」（SPT）schedule により、最初の $\mathbf{dQ}$ write で CTA が stall しない。

<span id="section-03-03"></span>

### 3.3 Scheduling

Causal masking や variable sequence length（varlen）など多くの場合、attention kernel は自然に load imbalance を持つ。割り当てられた worktile により SM の mainloop length が異なり、一部の worktile は他より多くの load と MMA を必要とするためである。さらに、grid coordinate の望ましい linearization を定めるなど、SM が tile を処理する順序を選べる。Attention 固有の性質を抽象化すれば、同一の parallel processor に対する makespan minimization の一般的な結果を適用できる。FlashAttention-4 では古典的な longest-processing-time-first（LPT）scheduling [Gra69] を使う。この適用方法はすべての GPU architecture で機能し、Hopper GPU 上の FlashAttention-3 でも改善になることを確認した。

**Causal masking の LPT：** 標準 attention grid は（mblocks, heads, batches）であり、左から右へ昇順に計算する。しかし diagonal より上の score は mask されるため、固定した head と batch に対し、SM は worktile を短いものから長いものへ非効率に処理する。一方、naive LPT order も最適ではない。Batch が異なると mainloop KV load が L2 cache に hit せず、すべての KV head を先に load すると、それらが L2 capacity を超えた場合に L2 cache thrashing が起きる。その代わり、常に batch を outermost dimension とし、head を swizzle する。つまり、L2 cache から overflow しない section に head を分割し、tile scheduler が section 内の head、逆順の mblock、section、最後に batch の順で grid を traverse する。特に MQA または GQA では、mblock を変える前に、KV head ごとのすべての query head を必ず traverse する。実験では、この LPT order が非常に有効である。たとえば H200 GPU 上の BF16、head dimension 128 では、MHA で 4-8%、MQA 8 で 7-14% の FLOPS 向上を得た。

**Variable sequence length の LPT：** Varlen では batch 間のばらつきによる load imbalance にも対処する必要がある。たとえば decode workload では batch ごとに参照する context length が異なる場合があり、mixed batching や continuous batching では、一部の batch が prefill、別の batch が decode である場合がある。Batch ごとの query と KV sequence length の list は通常、attention metadata として device に格納される。標準の varlen attention kernel は runtime にこれらの integer を読み、batch を昇順に処理する。しかし、与えられた batch order は load balancing に対して任意に悪くなり得る。たとえば短い square prefill の後に long-context decode が並ぶ場合である。これを改善するには preprocessing kernel を起動し、worktile ごとの最大実行時間に従って batch を sort することで LPT order を強制できる。Preprocessing kernel は virtual batch index から actual batch index への mapping metadata を追加で書き出し、attention kernel が後でそれを読み、sorted order で batch を traverse する。この metadata は cache できるので、sorting による性能低下はない。

## 4 言語とフレームワーク

FlashAttention-4 は Python に埋め込まれた CuTe-DSL [Nvi25d] だけで記述し、CUDA C++ component は含まない。CuTe-DSL compiler は Python source code を受け取り、PTX へ lower し、PTX compiler（ptxas）を使って最終的に assembly code（SASS）を生成する。

**明瞭な abstraction と完全な表現力：** CuTe-DSL programming model は CUTLASS C++ と isomorphic である。したがって FlashAttention-4 は、C++ の代わりに Python で meta-programming できる生産性と高速な JIT compilation を得ながら、low-level GPU programming の完全な表現力を保つ。CuTe-DSL は escape hatch として PTX への直接 access を提供し、framework の制約なしに必要な機能を実装できる。たとえば、CuTe-DSL API にまだ完全には公開されていない operation に custom PTX sequence を使っている（これらは将来の release へ統合される）。この例は、framework が GPU capability の限られた subset に開発者を制約しないことを示す。

**JIT による高速 compilation：** 過去の FlashAttention 実装では、複雑な C++ template metaprogram によって compile time がボトルネックになっていた。CuTe-DSL を Python に埋め込み just-in-time（JIT）compile することで、FlashAttention-4 は従来の C++ template-based approach より高速に build できる。[表 4](#table-04)に示すように、FlashAttention-4 は FlashAttention-3 と比べて compile time を 20-30$\times$ 短縮する。この短い iteration cycle により開発者の生産性が大きく向上し、kernel development 中の実験と debugging を高速化できる。

<span id="table-04"></span>

![原論文の表 4](../../papers/flashattention-4/table-04.png)

**表 4.** 単一 kernel の compile time：FA3（C++ template）と FA4（CuTe-DSL）。通常、FA2 と FA3 は異なる attention variant 向けに数百の kernel を precompile する必要がある。

**柔軟性と利用しやすさ：** Python-based framework は実際に柔軟性を示している。開発者は core framework を変更せず、FlashAttention-4 上に FlexAttention と block-sparse attention variant を構築した。参入障壁が下がるため、GPU programming の経験が数か月しかない研究者や engineer でも、C++ template metaprogramming の深い専門知識なしに意味のある extension を追加できる。この利用しやすさは innovation を速め、attention mechanism の研究コミュニティが新しい algorithmic variant をより素早く探索できるようにする。

我々の構想は、あらゆる種類の attention variant を best-in-class performance で構築する包括的な framework を提供することである。Attention variant ごとに一から実装するのではなく、FlashAttention-4 は共通機能を独立して composable な primitive へ分解する。Block-sparse pattern、masking strategy、variable sequence length handling、work scheduling などはすべて orthogonal primitive として公開し、自由に組み合わせられる。この modular design により、最適化と新機能は framework 上のすべての attention 実装へ反映され、同時に efficient GPU kernel へ compile して最高の性能を達成できる。

## 5 実験評価

FlashAttention-4 の効率を、複数の open-source および closed-source baseline と比較する。

**Attention の benchmark：** 異なる系列長と head dimension で FlashAttention-4 の runtime を測定し、PyTorch の標準実装、FlashAttention-2 [+fa3]、Triton（B200 固有命令を使用 [Til19]）、Gluon（Triton より low-level で細かな制御が可能な GPU programming language [Tri24b]）、cuDNN（B200 GPU 向けに最適化された vendor library）と比較する。FlashAttention-4 は cuDNN 9.13 より最大 1.3$\times$、Triton より最大 2.7$\times$ 高速である。FlashAttention-4 は最大 1613 TFLOPs/s に達し、B200 GPU の theoretical maximum TFLOPs/s の約 71% である。

**Benchmark setting：** B200 GPU 上で、causal mask の有無、head dimension 64、128、（192, 128）の各 setting について BF16 input の runtime を測定する。系列長は 1k、2k、...、32k とし、token の総数が 32k になるよう batch size を設定する。Hidden dimension は 2048、head dimension は 64 または 128（32 head または 16 head）とする。DeepSeek V3 [Dee24a] が使う（192, 128）configuration では、query dimension 192、key/value dimension 128 の 16 head を使う。Forward pass の FLOPs は $4 \cdot \mathrm{seqlen}^2 \cdot \mathrm{head\ dimension} \cdot \mathrm{number\ of\ heads}$ で計算する。Causal masking では約半分の entry だけが計算されるため、この値を 2 で割る。Backward pass の FLOPs は forward pass の 2.5 倍とする（forward pass は 2 matmul、backward pass は recomputation により 5 matmul である）。

### 5.1 Forward pass

<span id="figure-04"></span>

![B200 上の head dimension 128 における forward pass TFLOPS](../../papers/flashattention-4/figure-04.png)

**図 4.** B200（FP16/BF16）上の head dimension 128 における forward pass TFLOPS。左：non-causal attention。右：causal attention。FA4 は系列長全体で cuDNN 9.13.0 に対して 1.1-1.3$\times$、Triton に対して 2.1-2.7$\times$ 高速である。我々の実装の初回 release 以降、新しい cuDNN version は本論文の多くの手法を取り込み、FA4 と同程度の性能を得ている。

[図 4](#figure-04)および[図 5](#figure-05)に forward pass の結果を示す。FlashAttention-4 は cuDNN 9.13 より 1.1-1.3$\times$、Triton より 2.1-2.7$\times$ 高速である。中程度および長い系列（4k 以上）では、異なる head dimension と causal masking setting のすべてで、FlashAttention-4 は一貫して全 baseline を上回る。Causal case で gain が大きいのは、longest-processing-time-first（LPT）scheduler によるものと考える。

<span id="figure-05"></span>

![Causal attention、head dimension 192, 128 の forward pass TFLOPS](../../papers/flashattention-4/figure-05.png)

**図 5.** B200（FP16/BF16）上の causal attention における cuDNN と FA4 の forward pass TFLOPS 比較。Head dimension は（192, 128）であり、通常 DeepSeek V3 architecture で使われる。

### 5.2 Backward pass

<span id="figure-06"></span>

![B200 上の head dimension 128 における backward pass TFLOPS](../../papers/flashattention-4/figure-06.png)

**図 6.** B200（FP16/BF16）上の head dimension 128 における backward pass TFLOPS。左：non-causal attention。右：causal attention。

[図 6](#figure-06)に backward pass の結果を示す。FlashAttention-4 は長い系列と causal masking で一貫した高速化を達成し、2-CTA backward pass の有効性を示す。

[図 7](#figure-07)には deterministic backward pass の性能も示す。慎重な swizzling と scheduling により deterministic backward pass は大幅に高速化し、1-CTA の nondeterministic backward pass の最大 75% の速度に達する。

<span id="figure-07"></span>

![Causal attention の deterministic backward pass ablation](../../papers/flashattention-4/figure-07.png)

**図 7.** B200（FP16/BF16）上、head dimension 128 の deterministic backward pass ablation。Causal attention に対する SPT、reverse mblock order の LPT、LPT、batch/head swizzle なしの naive setting。

## 6 議論と結論

FlashAttention-4 は、Tensor Core が非常に高速になり、主要なボトルネックが shared-memory traffic と exponential throughput へ移る非対称な hardware scaling に対処する。この制約を緩和するには algorithm と kernel の協調設計が必要である。我々は完全非同期 MMA を中心に pipeline を再設計し、softmax と大きな tile の matmul を overlap する。また software-emulated exponential と conditional softmax rescaling を導入し、非 matmul operation を減らす。Tensor Memory と 2-CTA MMA mode を活用して shared memory traffic を削減する。加えて、2-CTA により global atomic accumulation を再構成し、global atomic add の回数を半減する。FlashAttention-4 は Python に埋め込まれた CuTe-DSL だけで実装され、low-level control を保ちながら、C++ template-based kernel より 20-30× 高速に compile できる。Blackwell GPU 向けに最適化しているが、compute が非 matmul unit を上回る速度で向上し続ければ、これらのアルゴリズムの一部は他の accelerator にも拡張できる。

#### 謝辞

Compute support を提供した Together AI、Meta、xAI、Princeton Language and Intelligence（PLI）に感謝する。Schmidt Sciences AI2050 fellowship、Google ML and Systems Junior Faculty Awards、Google Research Scholar program の支援に深く感謝する。また、Nvidia の CuDNN、TensorRT-LLM、Cutlass team には、継続的な議論、アイデア、feedback をいただいたことに感謝する。

## 付録 A 実験と benchmark の補足

### A.1 System と library

B100 180GB SXM6（1000W）上で速度を benchmark する。5 run で warmup し、その後 benchmark を 10 回繰り返して平均時間を取る。

通常は執筆時点（2025 年 3 月）の最新 library version を用いた。具体的には次を使う。

- CUDA 13.1
- FlashAttention 2.8.3
- Triton 3.6
- PyTorch 2.10.0
- CuTe-DSL 4.4.1

cuDNN について、main paper では cuDNN 9.13 と最新 version cuDNN 9.19.1.2 を比較する。Version 9.13 と 9.14 [Nvi25c] 以降、我々は cuDNN team と協力し、FlashAttention-4 の一部の手法を cuDNN に取り込んだ。これにより本研究をできるだけ多くの practitioner に届けられる。

### A.2 Non-causal deterministic backward

完全を期すため、causal masking なしの deterministic backward kernel の性能値も、causal masking と並べて[図 8](#figure-08)に示す。

<span id="figure-08"></span>

![Causal masking の有無に対する deterministic backward pass ablation](../../papers/flashattention-4/figure-08.png)

**図 8.** B200 上、head dimension 128 の deterministic backward pass ablation。左：batch/head swizzle を用いる non-causal attention と naive setting。右：causal attention の SPT、reverse mblock order の LPT、LPT、batch/head swizzle なしの naive setting。

[+equal]: Equal contribution

[+fa3]: FlashAttention-3 は B200 上で動作しない
