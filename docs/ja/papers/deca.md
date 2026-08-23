---
title: DECA Accelerator
createTime: 2026-08-23
permalink: /ja/papers/deca/
---

> Gerasimos Gerogiannis、Stijn Eyerman、Evangelos Georganas、Wim Heirman、Josep Torrellas。arXiv への初回提出日は 2025-05-25、現行版は v2（2025-08-08）。Intel Corporation、Intel Labs、University of Illinois at Urbana-Champaign。[arXiv:2505.19349](https://arxiv.org/abs/2505.19349)。[原 PDF](/paper/deca.pdf)。[TeX ソース](https://arxiv.org/e-print/2505.19349)。原論文の題名は「DECA: A Near-Core LLM Decompression Accelerator Grounded on a 3D Roofline Model」。

## 概要

大規模言語モデル（LLM）の推論におけるメモリ帯域幅のボトルネックを緩和するため、重み行列は量子化およびスパース化された形式でメモリに格納される。そのため、これらの行列 tile をコア内の汎用行列乗算（GeMM）ハードウェアエンジンで処理する前に、逆量子化とデスパース化を行う必要がある。現在、この処理はソフトウェアのベクトル演算によって実行されているが、得られる性能は限定的である。さらに、GeMM 全体の性能はメモリ資源、ベクトルユニット、ハードウェア行列エンジンの相互作用に依存するため、システムの改善方法を把握することも難しい。

コア内 GeMM エンジンと HBM を備えた先進的なプラットフォームで LLM 推論の性能を高めるため、本論文は三つの主要な貢献を示す。第一に、メモリ資源、ベクトルユニット、ハードウェア行列エンジンがどのように連携して圧縮 GeMM の性能を実現するかを把握できる、三次元可視化を伴う解析性能モデルを構築する。第二に、新しい近傍コア ML モデル解凍アクセラレータ DECA を提案する。DECA は tile のデスパース化と逆量子化を CPU からオフロードし、コア内 GeMM エンジンがそのまま利用できる tile を生成する。第三に、近傍コアアクセラレータをアウト・オブ・オーダーで呼び出す新しい ISA 拡張を導入する。この拡張により、アクセラレータとコアの計算を高い性能で交互に実行し、重ね合わせることができる。シミュレーションした HBM 搭載 56 コア Xeon 4 サーバでは、DECA は最適化された Intel ソフトウェア kernel に比べて圧縮 GeMM の実行を最大 4 倍高速化する。また、Llama2-70B と OPT-66B の次 token 生成時間を 1.6-2.6 倍短縮する。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル（LLM）は、チャットボット、翻訳、テキスト要約、コンテンツ生成で重要な機械学習（ML）ワークロードである。LLM は Transformer を使用し、主にマルチヘッド注意層と全結合（FC）層で構成される。最大規模のモデルは FC 層に数兆個のパラメータを含む。推論時には小さい batch でこれらの重みの再利用率が低くなり、メモリ容量と帯域幅を圧迫する。

GPU は計算能力とメモリ帯域幅に優れるため、LLM 推論の標準的なプラットフォームである。最近の Intel Xeon 4 サーバは、TMUL と呼ばれるコア内の汎用行列乗算（GeMM）エンジンを備え、High Bandwidth Memory（HBM）を搭載できる。TMUL は AMX 命令でプログラムされ、ベクトル SIMD ユニットと比べて GeMM スループットを一桁高める。一方、HBM は DDR システムの三倍から四倍の帯域幅を供給する。

Xeon サーバ上の LLM 推論はメモリ帯域幅に制約される。FC 層の大規模 GeMM は、Llama2-70B の次 token 生成時間の 90% 以上を占める [Lla23]。したがって、これらの GeMM の高速化が推論高速化の中心となる。

低ビット重み量子化やスパース化などのモデル圧縮技術はメモリトラフィックを減らす。しかし、TMUL は密な BF16 または INT8 tile を必要とし、任意の量子化方式やスパースパターンを直接処理できない。このため Intel libxsmm は、AVX ベクトル命令で圧縮 tile を読み出し、デスパース化と逆量子化を行い、密な tile を AMX ユニットへ渡す。この協調動作はベクトル領域と行列領域を組み合わせ、それぞれが別の命令と機能ユニットを持つ。

プロファイリングでは、libxsmm は中程度に圧縮された GeMM と DDR メモリには有効だが、HBM では性能が低下する。従来の二次元 Roofline モデルはメモリ、ベクトル、行列資源の相互作用を扱わないため、この低下を説明できない。本論文では Roof-Surface と呼ぶ三次元モデルを構築し、この相互作用を明らかにして、性能低下の原因を AVX 解凍に帰着させる。

本論文は、tile のデスパース化と逆量子化を CPU コアからオフロードする近傍コアアクセラレータ DECA を提案する。DECA は 1 bit から 8 bit の量子化形式、非構造化スパース、グループ量子化を扱う。さらに、DECA をアウト・オブ・オーダーで呼び出し、CPU とアクセラレータの通信遅延を隠す ISA 拡張 Tile External Preprocess and Load（TEPL）を導入する。

異なるスパース度の BF8 と MXFP4 を評価した結果、シミュレーションした HBM 搭載 56 コア Xeon 4 サーバで、DECA は圧縮 GeMM を最大 4 倍高速化した。Llama2-70B と OPT-66B の次 token 生成時間は、ソフトウェア解凍に対して 1.6-2.6 倍、非圧縮ベースラインに対して 2.5-5.0 倍短縮された。

本論文の貢献は、Roof-Surface 性能モデル、DECA 近傍コア解凍アクセラレータ、アウト・オブ・オーダー呼び出しのための TEPL 拡張、および LLM 推論の圧縮 GeMM に対するシミュレーション評価である。

<span id="section-2"></span>

## 2 背景

<span id="section-2-1"></span>

### 2.1 LLM 推論

LLM は埋め込み層、全結合層、注意層で構成される。推論は入力 token を符号化して最初の token を生成する prompt フェーズと、その後に出力 token を生成する生成フェーズから成る。本論文では算術強度の低い生成フェーズに注目する。このフェーズは多くの実用的な場面でエンド・ツー・エンド推論の大半を占める [Yua24]。HBM とコア内 GeMM エンジンを備えた CPU は有望なプラットフォームであるため、現代の CPU サーバにおける推論を研究する。

<span id="section-2-2"></span>

### 2.2 モデル圧縮

量子化は重みを FP8 や FP4 などの低ビット形式で保存する。グループ量子化はグループごとの scale factor を追加する。本論文では BF8 と MXFP4 を評価する。MXFP4 は 4 bit 値を使用し、32 個の重みごとに一つの scale を共有する。スパース化はゼロに近い重みを取り除く。非構造化スパースでは任意の位置を除去でき、bitmask から非ゼロ位置を復元する。50% から 5% の密度を評価する。

密な BF16 モデルに対して、密度係数が $d$ の Q-bit モデルはモデルサイズを $16/(Qd+1)$ 分の一にする。ここで 1 は bitmask を表す。この比率を Compression Factor（CF）と呼ぶ。圧縮はオフライン、解凍はオンラインで行われ、後者は性能に影響する。

<span id="figure-01"></span>

![図 1．オフライン圧縮とオンライン解凍。](../../papers/deca/figure-01.png)

**図 1．** オフライン圧縮とオンライン解凍。

<span id="figure-02"></span>

```text
// Ti+1 を解凍
Ti+1 の各行 r について:
  AVX ベクトル演算を適用
// GeMM Ti
AMX で Ti をロード
AMX で Tout を計算
// Ti+2 を解凍し、GeMM Ti+1 を実行
```

**図 2．** Libxsmm 圧縮 GeMM kernel の疑似コード。

<span id="section-2-3"></span>

### 2.3 行列拡張

Intel Advanced Matrix Extensions（AMX）は 8 個の tile レジスタを追加する。各レジスタは最大 16 行、1 行あたり 64 byte を保持し、BF16 または INT8 要素として解釈される。TMUL は activation tile $A$ と weight tile $W^\top$ を乗算する。batch size $N\leq16$ のとき、1 回の演算は 16 cycle で $512N$ 回の fused multiply-add を実行する。本論文では fused multiply-add を FLOP と呼ぶ。

<span id="section-2-4"></span>

### 2.4 GeMM 解凍

TMUL は特定の密な形式だけを受け付けるため、圧縮重みは乗算前に tile へ解凍しなければならない。Libxsmm は double buffer を使用し、tile $T_{i+1}$ の AVX 解凍と tile $T_i$ の AMX 処理を重ね合わせる。解凍シーケンスは permute と mask 付きベクトル展開を使用する。AVX は cache line サイズの行を処理し、AMX は tile 全体を処理するため、AVX 命令数は AMX 命令数を大きく上回る。

<span id="section-3"></span>

## 3 動機

<span id="section-3-1"></span>

### 3.1 FC 層の GeMM が推論を支配する

<span id="table-01"></span>

![表 1．次 token 時間に対する FC 層 GeMM の寄与。](../../papers/deca/table-01.png)

**表 1．** 次 token 時間に対する FC 層 GeMM の寄与。

次 token 生成時間の大半は FC 層の GeMM に費やされ、DDR5 では 95% 以上、HBM では 85%-90% を占める。したがって、これらの GeMM を高速化すればエンド・ツー・エンド性能が直接改善される。

<span id="section-3-2"></span>

### 3.2 FC 層の GeMM は帯域幅に制約される

<span id="figure-03"></span>

![図 3．N=4 の GeMM に対する従来の Roofline。](../../papers/deca/figure-03.png)

**図 3．** $N=4$ の GeMM に対する従来の Roofline。

非圧縮 BF16 の点はメモリ帯域幅に制約される。圧縮によって算術強度が上がり、点は右へ移動する。高圧縮率では AVX 解凍がメモリ帯域幅や TMUL スループットに追いつかないため、観測性能は Roofline を下回る。

<span id="section-3-3"></span>

### 3.3 圧縮 GeMM は非効率を招く場合がある

Roofline 解析からは、解凍ボトルネックの解消に必要なベクトルスループットを判断できない。コアを拡張すると、ベクトルユニット、スーパースカラ幅、cache port を過剰に用意するおそれがある。次節では、このハードウェア支援の設計を導くモデルを構築する。

<span id="section-4"></span>

## 4 Roof-Surface モデル

三次元 Roof-Surface と、その二次元投影である Bounding Region Diagram（BORD）を用いて、行列、ベクトル、メモリ演算の相互作用をモデル化する。

<span id="section-4-1"></span>

### 4.1 三次元 Roof-Surface 性能モデル

メモリは 1 秒あたり $\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}$ tile の速度で圧縮 tile を供給する。ベクトルハードウェアは $\mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}$ の速度で解凍する。行列ハードウェアの処理速度は $\mathrm{MOS}$ である。達成可能な tile rate は次式となる。

<span id="equation-01"></span>

$$
\mathrm{TPS} = \min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

一つの TMUL tile は $512N$ 回の FMA を実行するため、Roof-Surface の式は次のようになる。

<span id="equation-02"></span>

$$
\mathrm{FLOPS} = 512N\cdot\min\{\mathrm{MBW}\cdot \mathrm{AI}_{\mathrm{XM}}, \mathrm{VOS}\cdot \mathrm{AI}_{\mathrm{XV}}, \mathrm{MOS}\}.
$$

kernel に依存する二つの変数は $\mathrm{AI}_{\mathrm{XM}}$ と $\mathrm{AI}_{\mathrm{XV}}$ である。これらの値が kernel signature を構成するため、同じ signature を持つ二つの kernel は同じ投影性能を持つ。

<span id="figure-04"></span>

![図 4．三次元 Roof-Surface モデルとその性能予測。](../../papers/deca/figure-04.png)

**図 4．** 三次元 Roof-Surface モデルとその性能予測。

この surface には、メモリ制約領域、ベクトル制約領域、行列制約領域がある。surface より下の測定点は制約資源を示し、surface より上の点は達成できない。

<span id="section-4-2"></span>

### 4.2 二次元 Bounding Region Diagram

BORD は Roof-Surface を $\mathrm{AI}_{\mathrm{XM}}$-$\mathrm{AI}_{\mathrm{XV}}$ 平面へ投影する。境界線は $y=(\mathrm{MBW}/\mathrm{VOS})x$、$x=\mathrm{MOS}/\mathrm{MBW}$、$y=\mathrm{MOS}/\mathrm{VOS}$ である。

<span id="figure-05"></span>

![図 5．HBM および DDR システムの BORD。](../../papers/deca/figure-05.png)

**図 5．** HBM および DDR システムの BORD。

HBM システムでは大半の kernel がベクトル制約領域に入る。一方、DDR はメモリ帯域幅が小さいため、メモリ制約領域が広くなる。

<span id="figure-06"></span>

![図 6．ベクトルスループットを 4 倍にした HBM の BORD。](../../papers/deca/figure-06.png)

**図 6．** ベクトルスループットを 4 倍にした HBM の BORD。

ベクトルスループットを 4 倍にしても、すべての kernel からベクトルボトルネックを取り除くことはできない。コアはすでに動的命令の大半を解凍に使い、commit slot の 40%-80% を使用しているため、従来方式の拡張は高コストになる。

<span id="section-5"></span>

## 5 DECA の概要とアウト・オブ・オーダー呼び出し

<span id="section-5-1"></span>

### 5.1 DECA の配置とシステム統合

各 CPU コアには、processing element、control register、tile-output register を持つ DECA が対応する。コアは privileged store によって量子化とスパースの設定を行う。DECA は L2 を介して圧縮 tile を読み出して解凍し、そのまま利用できる tile を tile-output register へ書き込む。

<span id="figure-07"></span>

![図 7．コアの隣に配置した DECA。](../../papers/deca/figure-07.png)

**図 7．** コアの隣に配置した DECA。

DECA は L2 TLB と CPU 仮想アドレス空間を共有する。context switch の際に状態を保存でき、別の process が使用するときには trap して再設定することもできる。

<span id="section-5-2"></span>

### 5.2 DECA とコアによる協調 tile 処理

二つの loader と二つの tile-output register がハードウェア double buffering を提供する。loader はデータ、bitmask、scale factor を取得する。DECA が tile $T_i$ をロード、解凍、書き込みしている間に、コアは $T_{i-1}$ を読み出して乗算し、その後 $T_{i+1}$ の取得を開始する。

<span id="figure-08"></span>

![図 8．DECA と CPU コアによる協調 tile 処理。](../../papers/deca/figure-08.png)

**図 8．** DECA と CPU コアによる協調 tile 処理。

<span id="figure-09"></span>

```text
TLoad TReg1, TOut1
TComp TReg2, TReg1
T(i+1) のメタデータを Loader2 へ保存
fence
```

**図 9．** memory-mapped DECA 呼び出しを行う CPU の疑似コード。

<span id="section-5-3"></span>

### 5.3 アウト・オブ・オーダー呼び出しの ISA 支援

Tile External Preprocess and Load（TEPL）はメタデータの送信と tile のロードを組み合わせる。DECA が tile をコアの tile register へ解凍すると命令が完了する。二つの loader に対応して、同時に実行できる TEPL 命令は最大 2 個である。

<span id="figure-10"></span>

```text
TEPL TReg1, M(i+1)
TComp TReg2, TReg1
TEPL TReg2, M(i+2)
```

**図 10．** TEPL 命令を使用する CPU の疑似コード。

TEPL は専用 queue に入り、source register と execution port が利用可能になると実行される。DECA はメモリを更新しないため、speculative invocation は安全である。pipeline flush は squash signal を送る。この仕組みで fence を取り除き、複数 tile の処理を重ね、コアと DECA の通信を隠す。

<span id="section-6"></span>

## 6 DECA のマイクロアーキテクチャ設計

<span id="section-6-1"></span>

### 6.1 DECA processing element

<span id="figure-11"></span>

![図 11．DECA processing element のマイクロアーキテクチャ。](../../papers/deca/figure-11.png)

**図 11．** DECA processing element のマイクロアーキテクチャ。

各 processing element は二つの Load Queue（LDQ）と prefetcher、bitmask queue、scale-factor queue、vector pipeline、look-up-table（LUT）array、展開 stage、scaling stage、tile-output register を備える。pipeline は圧縮データを読み出し、bitmask に従って非ゼロ値を展開し、LUT を用いた逆量子化と scaling を適用して、密な BF16 tile を書き出す。

<span id="section-6-2"></span>

### 6.2 定量的なマイクロアーキテクチャ設計

vector pipeline の幅を $W$、latency を $L$ とする。LUT array は 8 bit 量子化を支援し、各 8 bit 入力に対応する値を格納する。bitmask queue と scale-factor queue は一つの tile に合わせた容量を持つ。スパース入力では処理する要素が少ないため、pipeline は自然に高いスループットを達成できる。Roof-Surface モデルはバランスの取れた設計として $W=32$ と $L=8$ を選択する。

<span id="section-7"></span>

## 7 解凍ボトルネックを扱う DECA の代替方式

従来のベクトル資源を拡張するには、AVX unit の追加、ベクトル幅の拡大、スーパースカラ幅の拡大、cache port の追加が必要になる。コア内の行列拡張は一部のスパース形式や低ビット形式に対応できるが、形式ごとのハードウェアが必要で、将来の方式には適応できない。分離型ベクトルアクセラレータは一部のコア変更を避けられるが、中程度にスパースな ML モデルで TMUL のスループットを活用できない。DECA は解凍を独立させ、LUT 設定によって多様な形式を支援し、小さな ISA 変更で行列ユニットと協調する。

<span id="table-02"></span>

![表 2．ほかのコア内および近傍コアアクセラレータとの比較。](../../papers/deca/table-02.png)

**表 2．** ほかのコア内および近傍コアアクセラレータとの比較。

<span id="section-8"></span>

## 8 評価方法

2.5 GHz で動作し、260 GB/s の DDR5 または 850 GB/s の HBM を備えた Xeon 4 相当の 56 コアサーバをシミュレーションする。Sniper ベースの simulator に DECA processing element と TEPL queue を追加する。baseline DECA は $W=32$ と $L=8$ を使用する。

ソフトウェア baseline は Intel libxsmm である。その AVX 解凍シーケンスを TEPL 命令に置き換え、約 2 億 5000 万パラメータを持つ全結合層の cascade を評価する。batch size は 1 から 16 とし、Llama2 と OPT の推論には Intel Tensor Processing Primitives framework を使用する。

Q16、Q8、Q4 圧縮を評価し、Q16 と Q8 では密度を 50% から 5% とする。面積は CACTI と公表済み回路モデルから推定する。56 個の DECA processing element は約 $2.51\,\mathrm{mm}^2$ を占め、56 コア Xeon 4 die の 0.2% 未満である。

<span id="section-9"></span>

## 9 評価

<span id="section-9-1"></span>

### 9.1 圧縮 GeMM に対する DECA

<span id="figure-12"></span>

![図 12．DDR、N=1 における圧縮 GeMM の speedup。](../../papers/deca/figure-12.png)

**図 12．** DDR、$N=1$ における圧縮 GeMM の speedup。

<span id="figure-13"></span>

![図 13．HBM、N=1 における圧縮 GeMM の speedup。](../../papers/deca/figure-13.png)

**図 13．** HBM、$N=1$ における圧縮 GeMM の speedup。

DECA は DDR で 1.7 倍、HBM で 4.0 倍の speedup を達成する。ベクトル処理のオーバーヘッドが隠されるため、性能は最適値に近い。batch size を 16 まで増やしても同様の結果が得られる。

<span id="figure-14"></span>

![図 14．DDR、N=4 における各圧縮方式の TFLOPS。](../../papers/deca/figure-14.png)

**図 14．** DDR、$N=4$ における各圧縮方式の TFLOPS。

DECA を備えた 16 コアは従来の 56 コアを上回るため、ほかの workload にコアを割り当てたり、power gating したりできる。

<span id="table-03"></span>

![表 3．Q8、N=1、HBM における component utilization。](../../papers/deca/table-03.png)

**表 3．** Q8、$N=1$、HBM における component utilization。

DECA はメモリ利用率を高める一方、ベクトル解凍ユニットがボトルネックになることを防ぐ。この結果は Roof-Surface の予測を裏づける。

<span id="figure-15"></span>

![図 15．DECA と従来のベクトル拡張の比較。](../../papers/deca/figure-15.png)

**図 15．** HBM、$N=1$ における DECA と従来のベクトル拡張の比較。

AVX unit を四つ追加する場合も AVX unit の幅を 4 倍にする場合も、性能は DECA を大きく下回る。従来システムではスーパースカラ幅や L1 port を同時に拡張しないためである。

<span id="section-9-2"></span>

### 9.2 Roof-Surface による設計空間探索

<span id="figure-16"></span>

![図 16．DECA なし、および異なる DECA サイズでの HBM BORD。](../../papers/deca/figure-16.png)

**図 16．** DECA なし、および異なる DECA サイズでの HBM BORD。

モデルは、すべての kernel をベクトル制約領域から移動させる最小設計として $W=32,L=8$ を選択する。過剰構成の $W=64,L=64$ は 3% 未満しか高速化しないが、LUT 数は 8 倍になる。

<span id="section-9-3"></span>

### 9.3 DECA 統合と TEPL の解析

<span id="figure-17"></span>

![図 17．HBM、N=4 における DECA 統合機能。](../../papers/deca/figure-17.png)

**図 17．** HBM、$N=4$ における DECA 統合機能。

圧縮重みを L2 から読み出すこと、DECA prefetcher を使用すること、tile-output register へ書き込むこと、TEPL を使用することにより、性能は段階的に向上する。密度 5% では、TEPL が通信遅延を隠すことで性能を 2 倍にする。

<span id="section-9-4"></span>

### 9.4 LLM 推論に対する DECA

<span id="table-04"></span>

![表 4．Llama2-70B と OPT-66B の次 token latency。](../../papers/deca/table-04.png)

**表 4．** Llama2-70B と OPT-66B の次 token latency。

入力 token と出力 token がそれぞれ 128 個の場合、DECA はソフトウェア解凍に対して次 token 時間を 1.6-2.6 倍短縮し、非圧縮 baseline に対して 2.5-5.0 倍の speedup を実現する。

<span id="section-10"></span>

## 10 その他の関連研究

分離型アクセラレータはスパース、量子化、注意を対象とするが、面積、電力、データ移動のコストを伴うことが多い。CPU 統合型アクセラレータはこれらのコストを減らす。協調型ベクトル・行列 processor には Tandem、AWS Trainium、TPU、Tensor Core と SIMT core を持つ GPU がある。GPU Tensor Core が受け付ける形式も限られるため、DECA のような解凍エンジンで TMA を拡張し、shared memory への負荷を軽減できる可能性がある。

<span id="section-11"></span>

## 11 結論

本論文は、コア内 GeMM エンジンと HBM を備えた CPU プラットフォームを対象に、Roof-Surface 性能モデル、DECA 近傍コア ML モデル解凍アクセラレータ、アクセラレータをアウト・オブ・オーダーで呼び出す TEPL ISA 拡張を提示した。DECA は圧縮 GeMM と LLM 推論を高速化し、評価システムにおける面積オーバーヘッドを 0.2% 未満に抑える。
