---
title: 'Kelle: KV Cache and eDRAM Co-design'
createTime: 2026/08/23 20:00:00
permalink: /ja/papers/kelle/
pageClass: paper-reading
---

> [Tianhua Xia](https://scholar.google.com/citations?user=cC4Aw_4AAAAJ&hl=en) and [Sai Qian Zhang](https://www.saiqianzhang.com/). 2025 年 10 月 16 日に arXiv へ初回投稿され、現在の版は v1 である。第 58 回 IEEE/ACM International Symposium on Microarchitecture (MICRO '25) の予稿集に掲載された。本リーディング版は [Kelle: Co-design KV Caching and eDRAM for Efficient LLM Serving in Edge Computing](https://arxiv.org/abs/2510.16040v1) を転記したものである。[原論文 PDF](/paper/kelle.pdf)。[DOI](https://doi.org/10.1145/3725843.3756071)。[TeX ソース](https://export.arxiv.org/e-print/2510.16040v1)。正確な印刷レイアウトと参考文献については原論文 PDF を正とする。

## 概要

エッジデバイス上で大規模言語モデル (LLM) を実行することは、レイテンシの短縮、リアルタイム処理の改善、プライバシーの強化に不可欠である。デバイス上で直接推論すれば、データをクラウドへ送る必要がなくなり、応答が高速化し、ネットワーク接続への依存も減る。しかし、エッジデバイスへの LLM 実装には課題があり、特に LLM サービングで中心的な役割を担う Key-Value (KV) キャッシュの管理が問題となる。入力テキストが長くなるにつれて KV キャッシュのサイズは系列長に対して線形に増加し、大きなメモリ占有量とデータアクセスコストをもたらす。一方、エッジデバイスではメモリと計算能力が限られるため、LLM 推論に必要な大規模キャッシュを格納し、効率よくアクセスすることが難しい。

KV キャッシュによる大きなオーバーヘッドを軽減するため、SRAM より高い記憶密度を持つ組込み DRAM (eDRAM) を、エッジデバイスにおける LLM サービングの主要ストレージとして使用することを提案する。ただし、データ完全性を保つには eDRAM を定期的にリフレッシュする必要があり、その消費電力は大きい。eDRAM のコストを抑え、システム全体の性能を改善するため、eDRAM ベースのエッジシステムへの LLM 配備に最適化したソフトウェア・ハードウェア協調設計 *Kelle* を提案する。細粒度のメモリ追い出し、再計算、リフレッシュ制御アルゴリズムと組み合わせることで、*Kelle* アクセラレータは既存のベースラインに対して $3.9\times$ の高速化と $4.5\times$ の省エネルギーを実現する。

<span id="section-1"></span>

## 1 はじめに

大規模言語モデル (LLM) は、幅広い領域で顕著な能力を示している。クラウドへの配備には処理能力を高められる利点がある一方、通信レイテンシの大きさやセキュリティリスクといった制約もある。LLM の発展に伴い、その能力をエッジデバイスへ直接持ち込むことがますます重要になっている [Shi16a]。LLM をエッジデバイスへ統合すれば利用可能性が広がるだけでなく、個人や産業のニーズに合わせた堅牢な体験を提供できる。この動きは学術界 [Fra22, Zha24x, Cai24c, Yu24c, Yu24b] だけでなく、Intel [She23e]、NVIDIA [Nvi22a]、Microsoft [Wan23]、Qualcomm [Sor23] などの企業でも進んでいる。

しかし、エッジデバイスへの LLM 実装には課題があり、特に LLM の token 生成を高速化するうえで重要な Key-Value (KV) キャッシュ [Rad19b] の管理が問題となる。この機構は、attention の計算中に過去に計算した *Key と Value* のベクトル (KV ベクトル) を保存し、後続 token の生成時に再利用する。これにより、新しい token を生成するたびに、それ以前の token のベクトルを再計算せずに済む。一方、KV キャッシュのメモリ占有量は、モデルサイズと生成テキスト長の増加に伴って急速に増える [Hoo24, Zha24y]。例えば、LLaMA 2-7B が長さ 8192 の系列を FP16 で処理すると、KV キャッシュは 4GB のメモリを消費し、片上 SRAM と片外 DRAM の間で頻発するメモリアクセスが総実行レイテンシを主に制約する [Pop22, Yu24b]。エッジデバイスのように片上 SRAM の容量が限られるシステムでは、この問題が特に深刻である [Zha24y]。例えば、エッジ GPU Jetson Orin NX の L3 キャッシュは 4MB しかない [Nvi22a]。

単純な対策は片上 SRAM を増やすことであり、高コストな片外メモリアクセスを減らし、システム全体の性能を改善できる [Tu18, Che14a]。しかし、エッジデバイスの面積と電力の予算は限られており、SRAM の拡大は計算コアなど他の重要な部品が利用できる資源を減らす [Kun86, Wan22k, Che19c]。そこで本研究では、LLM 実行中の KV ベクトルを格納する主要な片上媒体として組込み DRAM (eDRAM) を用いる。メモリセル当たりのトランジスタ数は、SRAM セルの 6T に対して eDRAM セルは 3T などと少ないため、eDRAM はより高いデータ記憶密度を持ち、容量は 2 倍を超える [Git20, Che14a]。この記憶密度により、同じチップ面積で片上記憶容量を増やせる。eDRAM のリーク電力も SRAM より大幅に低い (先行研究 [Chu11] によれば約 $3.5\times$)。これらの利点から、eDRAM はエッジデバイスで KV ベクトルを格納するのに適している。

<span id="figure-01"></span>

![(a) LLM の token 生成。(b) 中間データを格納する KV キャッシュ。N は token のインデックスを表す。](../../papers/kelle/figure-01.png)

**図 1.** (a) LLM の token 生成。(b) 中間データを格納する KV キャッシュ。N は token のインデックスを表す。

ただし、eDRAM にはリークによるデータ損失を防ぐため定期的なリフレッシュが必要という大きな欠点がある。具体的には、eDRAM セルのリフレッシュには読み書き操作が必要であり、レイテンシと消費電力が増え、LLM の効率的な配備に大きな影響を及ぼし得る。eDRAM 統合の課題に対処するため、KV キャッシュアルゴリズムと eDRAM ベースのハードウェアシステムを協調設計し、精度を損なわずに高効率な KV キャッシュを実現する。本研究の貢献は次のとおりである。

- eDRAM ベースのエッジシステムにおけるデバイス内 LLM サービングのためのアルゴリズム・システム協調設計 *Kelle* を提案する。eDRAM 統合コストを抑え、LLM の実行効率を改善するため、KV キャッシュを効率よく実現する *attention に基づく追い出し・再計算ポリシー* (AERP) と *2 次元適応リフレッシュポリシー* (2DRP) を導入する ([第 4.1 節](#section-4-1)および[第 4.2 節](#section-4-2))。
- eDRAM を主要な片上ストレージとし、専用メモリレイアウトを備えた *Kelle アクセラレータ* を設計する。効率を最大化するため、専用 eDRAM コントローラと *systolic evictor* を統合し、AERP と 2DRP を効率よく実装する ([第 5 節](#section-5))。
- eDRAM のデータ生存期間と LLM サービングのレイテンシを最適化する効率的な計算パターンを採用し、eDRAM のリフレッシュエネルギーとメモリトラフィックを大幅に削減する Kelle scheduler ([第 6 節](#section-6)) も導入する。
- 評価の結果、Kelle は LLM の精度への影響を無視できる程度に抑えながら、他のベースラインハードウェアに対して $3.9\times$ の高速化と $4.5\times$ の省エネルギーを達成した ([第 7 節](#section-7)、[第 8 節](#section-8))。

<span id="section-2"></span>

## 2 背景と関連研究

<span id="section-2-1"></span>

### 2.1 LLM のワークフロー

現代の LLM (Llama 系列 [Oth23, Oth23a]、GPT 系列 [Rad19b, Oth20] など) は Transformer decoder の積層で構成され、各 decoder は Self-Attention (SA) block と feed-forward network (FFN) という 2 つの基本要素を持つ。LLM サービングでは、まず SA block の入力に 3 つの重み行列 $W_{Q}$、$W_{K}$、$W_{V}$ を乗じ、それぞれ query ($q$)、key ($k$)、value ($v$) と呼ぶ出力を得る。得られた $q$ と $k$ は $v$ とともに、乗算、softmax、残差加算を経て SA の出力を生成する。SA の出力は FFN でさらに処理され、通常は標準 MLP [Rad18, Rad19b] または gated MLP [Liu21c, Oth23, Oth23a] が用いられる。FFN は複数の fully connected (FC) layer と、GeLU [Hen16a] などの中間活性化関数で構成される。

LLM サービングには pre-filling と decoding という 2 つの主要段階がある。pre-filling 段階では、モデルが context token を並列処理する。decoding 段階では、現在と過去の token に基づいて次の token を予測する。これは、現在の入力を、過去の token から得た Key と Value (KV) ベクトルとして表される情報と組み合わせることで行われる。この処理は自己回帰的に反復される ([図 1a](#figure-01))。

<span id="section-2-2"></span>

### 2.2 KV キャッシュ

decoding 段階では、生成速度を高めるため、新たに生成された各 token の KV ベクトルを KV キャッシュに格納する ([図 1b](#figure-01))。これにより、新しい token の生成ごとに過去の token のベクトルを再計算せずに済む。具体的には、LLM block の出力を生成するため、長さ $C$ の N 番目の token の入力ベクトルに $W_{Q}$、$W_{K}$、$W_{V}$ を乗じ、query $q_{N}$、key $k_{N}$、value $v_{N}$ という次元 $1\times C$ の 3 ベクトルを生成する。$C$ は channel size を表す ([図 1b](#figure-01))。次に、$q_{N}$ と他の KV ベクトルを channel 次元に沿って複数の部分に分割し、それぞれの次元を $1\times\frac{C}{H}$ とする。$H$ は head 数を表す。head h のベクトルをそれぞれ $q^{h}_{N}$、$k^{h}_{N}$、$v^{h}_{N}$ と表す。続いて、過去 $N-1$ token の KV ベクトルをメモリから読み出す。各 head h について、$q^{h}_{N}$ と各 key ベクトル $k^{h}_{n},1\leq n\leq N$ の内積を計算し、その結果を softmax 関数へ渡して、次元 $1\times N$ の attention score ベクトル $A^{h}_{N}$ を得る。次に、この attention score ベクトルと各 value ベクトル $v^{h}_{n},1\leq n\leq N$ の内積を計算し、長さ $\frac{C}{H}$ の結果ベクトル $y^{h}_{N}$ を得る。複数の head にわたって $y^{h}_{N}$ を連結し、得られたベクトル $y_{N}$ に $W_{O}$ を乗じる。この処理を[図 2](#figure-02)に示し、次式で表す。

<span id="equation-01"></span>

$$
A^{h}_{N}=\mathrm{softmax}([q^{h^{\top}}_{N}k^{h}_{1},q^{h^{\top}}_{N}k^{h}_{2},\ldots,q^{h^{\top}}_{N}k^{h}_{N}])
$$

<span id="equation-02"></span>

$$
y^{h}_{N}=\sum_{1\leq n\leq N}(A^{h}_{N,n}\cdot v^{h}_{n})
$$

ここで $A^{h}_{N,n}$ は $A^{h}_{N}$ の $n$ 番目の要素を表す。[式 1](#equation-01)と[式 2](#equation-02)から、KV ベクトル対 $[k^{h}_{n},v^{h}_{n}]$ の相対的な順序は decoding の計算に影響しないことが分かる。すなわち、2 組の KV ベクトル対の値を交換しても (例えば $[k^{h}_{1},v^{h}_{1}]$ と $[k^{h}_{2},v^{h}_{2}]$ を交換しても)、[式 1](#equation-01)と[式 2](#equation-02)から得られる $y^{h}_{N}$ は変わらない。

<span id="figure-02"></span>

![KV ベクトル計算の例。](../../papers/kelle/figure-02.png)

**図 2.** KV ベクトル計算の例。

KV キャッシュ圧縮技術は、token dropping [Xia24a, Zha23g, Ge24, Liu24p, Yan24f, Liu24m] と KV キャッシュ量子化 [Liu24c, Hoo24, Xia24h] の 2 つに大別できる。token dropping は重要でない token を特定して永続的に破棄し、その後はアクセス不能とする。StreamLLM [Xia24a] は、系列先頭にあり LLM の性能に重要な *sink token* を特定し、性能を保つため直近の token も保持する。H2O [Zha23g] は累積 attention score の高い *heavy hitter token* を特定する。KIVI [Liu24c] は KV ベクトルを channel 単位でグループ化し、2-bit 非対称量子化を実現する。QuaRot [Ash24] は zero-shot Hadamard 変換によりモデルの外れ値を削減し、4-bit 量子化を可能にする。speculative decoding [Mcd25, Hu25i, Hu25j, Lev23] は、軽量な draft model が複数の token を提案し、full model が選択的に検証することで LLM を高速化する別の推論技術である。Kelle は speculative decoding 技術と直交的に併用できる。

<span id="section-2-3"></span>

### 2.3 組込み DRAM

<span id="table-01"></span>

![SRAM と eDRAM の比較。](../../papers/kelle/table-01.png)

**表 1.** SRAM と eDRAM の比較。

SRAM の代替としてさまざまな eDRAM 回路 [Git20, Yu20] が登場しており、中には 2 トランジスタしか必要としないものもある。その中でも 3T-eDRAM は、SRAM に比べて 2 倍を超える密度を実現し、静的消費電力を $3.5\times$ 削減する [Chu11, Cha13]。[表 1](#table-01)は 3T-eDRAM と SRAM を比較したもので、65nm technology node において Destiny [Por15] でシミュレーションした結果である。eDRAM は記憶密度が高く、アクセスレイテンシとエネルギーが低いため、LLM の実装に適している。一方、電荷リークによるデータ破損を防ぐ定期的なリフレッシュが必要という重大な欠点がある。したがって eDRAM は、頻繁なリフレッシュを避けられる**大量の一時データ**の格納に最も適している。

アクセラレータシステムで eDRAM を用いて CNN 計算を支援する研究 [Che14a, Zha23l, Tu18, Ngu19] が進められ、リフレッシュ電力のオーバーヘッドを抑える手法が提案されてきた。DaDianNao [Che14a] は eDRAM を bank に分割してリフレッシュ故障を軽減するが、リフレッシュエネルギーやデータ保持の課題には対処しない。RANA [Tu18] は CNN の学習中に bit retention error を注入し、低いリフレッシュ頻度による精度低下を緩和する。CAMEL [Zha23l] は CNN のモデルアーキテクチャを最適化し、学習中のデータ生存期間を短縮する。先行研究は Convolutional Neural Network (CNN) の推論と学習における eDRAM の効率を示してきたが、LLM における可能性はまだ調べられていない。これに対して Kelle は、eDRAM を用いて LLM の KV キャッシュによる片外メモリアクセスを最小化するという、従来未探索の領域を扱う。

<span id="section-2-4"></span>

### 2.4 エッジ LLM アクセラレータ

エッジデバイスへの LLM 配備を可能にするため、量子化 Transformer の精度を改善する複数の手法が提案されている [Guo23, Zad20, Lee24, Par18a, Fan22a, Lu20, Tra22, Liu24x, Xio25b]。Tender [Lee24] は scale factor を 2 のべき乗とすることで、ハードウェア効率のよい LLM 量子化手法を提案する。COMET [Liu24x] は 4-bit LLM 量子化向けの効率的な mixed-precision GPU kernel を設計する。FlexGen [Sto24]、InfiniGen [Lee24c]、InstInfer [Pan24g]、LLM.npu [Xu24f] などは、資源の限られたデバイスに LLM を効率よく配備するため、片上 unit と main storage の間でモデルを offload する戦略を検討する。Cambricon-LLM [Yu24b] は、NPU と専用 NAND flash chip を備える chiplet ベースのハイブリッドアーキテクチャを提案し、デバイス上の効率的な推論を可能にする。

<span id="figure-03"></span>

![(a) モデルと系列長ごとに比較した、4MB と 8MB SRAM を持つエッジシステムの正規化レイテンシ。(b) 8MB eDRAM と 8MB SRAM を持つエッジシステムの面積内訳。赤線は面積予算を表す。(c) eDRAM を統合したエッジシステムのエネルギー内訳。pre-filling 長 512 に対する decoding 長を示す。decoding 中、一部 layer の KV キャッシュを 8MB eDRAM に格納する。報告する DRAM エネルギーにはモデル重みのアクセスと eDRAM からの KV キャッシュ offload の両方を含む。](../../papers/kelle/figure-03.png)

**図 3.** (a) モデルと系列長ごとに比較した、4MB と 8MB SRAM を持つエッジシステムの正規化レイテンシ。(b) 8MB eDRAM と 8MB SRAM を持つエッジシステムの面積内訳。赤線は面積予算を表す。(c) eDRAM を統合したエッジシステムのエネルギー内訳。pre-filling 長 512 に対する decoding 長を示す。decoding 中、一部 layer の KV キャッシュを 8MB eDRAM に格納する。報告する DRAM エネルギーにはモデル重みのアクセスと eDRAM からの KV キャッシュ offload の両方を含む。

<span id="section-3"></span>

## 3 エッジデバイスの LLM に eDRAM を使う理由

<span id="section-3-1"></span>

### 3.1 片上メモリ拡大の利点と課題

先行研究 [Zha24w, Zha24y, Yu24b] が示すように、LLM のサービング速度は片外メモリ帯域幅に大きく制約される。特に KV キャッシュへのアクセスは、LLM の decoding 段階で最も重大なボトルネックとなる [Zha24y, Zha23g, Lee24c]。片外メモリ使用量を最小化する直接的な方法は片上 SRAM を拡大することであり、高コストな片外メモリアクセスを減らし、システム性能を高められる [Tu18, Che14a]。これを示すため、4MB と 8MB の SRAM を持つ 2 つのエッジ計算システムについて、異なる系列長で LLaMA2-7B を実行した際のレイテンシを評価する。テストには、8-bit MAC 演算用の $32\times 32$ systolic array と、帯域幅 64GB/s の 16GB DRAM を持つシミュレーションプラットフォームを用いる。これは Google Coral エッジデバイス [Sur20] に似た edge tensor processing unit (TPU) を表す。[図 3a](#figure-03)に示すように、SRAM 容量を 2 倍にすると平均 $1.27\times$ の高速化が得られる。しかし評価プラットフォームで SRAM を 4MB から 8MB に拡大すると、消費電力とチップ面積がそれぞれ $29\%$ と $26\%$ 増加する。エッジ環境では面積と電力の予算が限られるため、SRAM の増加は他の重要な部品が利用できる資源を減らし、システム性能を最適でないものにする [Kun86, Wan22k, Che19c]。したがって、次の知見を得る。

**知見 1.** 大きな片上メモリは LLM の KV キャッシュボトルネックを緩和するが、SRAM を片上ストレージとするエッジデバイスでは面積と電力のペナルティを伴う。

<span id="section-3-2"></span>

### 3.2 eDRAM 統合の長所と短所

面積を増やさずに片上メモリを拡大する方法の 1 つは、SRAM を eDRAM に置き換えることである。eDRAM は同じ SRAM 面積で 2 倍を超える容量を提供するだけでなく、[表 1](#table-01)に示すようにアクセスエネルギーとリークエネルギーも低い。[図 3b](#figure-03)に示すように、8MB eDRAM を持つ評価システムは 8MB SRAM を持つシステムより面積が小さく、より小さなチップで LLM サービングのレイテンシを短縮できる。広範な研究 [Xia24i, Agr14, Cho14b] と商用製品 [Git20, Tec24, Wen10, Flu14, Tim13] は、eDRAM を主要な片上記憶媒体として統合できることを示している。しかし、エッジデバイスでの LLM サービングに対する利点はまだ調べられていない。

<span id="figure-04"></span>

![65nm eDRAM の $105^{\circ}C$ における retention failure 分布 [Kon08]。](../../papers/kelle/figure-04.png)

**図 4.** 65nm eDRAM の $105^{\circ}C$ における retention failure 分布 [Kon08]。

<span id="figure-05"></span>

![Kelle アクセラレータの概要。](../../papers/kelle/figure-05.png)

**図 5.** Kelle アクセラレータの概要。

eDRAM には複数の利点があるが、先行研究 [Tu18, Zha23l] はリフレッシュ操作がシステム全体のエネルギー消費における大きなボトルネックとなり得ることを示している。さらに、生存期間の長いデータを eDRAM に格納すると、リフレッシュ頻度が低い場合に読み出しエラーの危険が高まる ([図 4](#figure-04))。retention failure rate は、リフレッシュ間隔を変えたときに retention error が生じる bit の割合として表す。この問題を示すため、[第 3.1 節](#section-3-1)のシステムで 4MB SRAM を 8MB eDRAM に置き換える。データ破損を防ぐため、eDRAM のリフレッシュ間隔を $45\mu s$ とする。異なるモデルと系列長で eDRAM システムのエネルギー消費を評価する。[図 3c](#figure-03)に示すように、最適化を行わない場合、eDRAM のリフレッシュは総エネルギーの最大 $46\%$ を占め、平均エネルギー消費を $1.7\times$ 増加させる。

**知見 2.** 同じチップ面積で、eDRAM はエッジデバイス上の LLM サービングにおいて SRAM より低いレイテンシをもたらす。しかし電力面の利点を十分に活用するには、eDRAM のリフレッシュ操作を大幅に削減する必要がある。

<span id="section-3-3"></span>

### 3.3 Kelle: KV キャッシュと eDRAM の協調設計

eDRAM のエネルギー消費を最小化する有効な戦略は、データのリフレッシュ頻度、格納データ量、データ生存期間を減らすことである。eDRAM によりエッジデバイスでの LLM サービング性能を高めるため、eDRAM のリフレッシュエネルギーを最小化し、KV キャッシュを効率よく管理するハードウェア・アルゴリズム協調設計 *Kelle* を提案する。

<span id="section-3-3-1"></span>

#### 3.3.1 eDRAM リフレッシュ制御

データのリフレッシュ頻度を下げると retention failure の危険が高まり、データが破損する可能性がある。ここから重要な問いが生じる。*精度を損なわず、LLM は KV キャッシュ内のデータ破損をどの程度許容できるか。* この問いに基づき、eDRAM のメモリレイアウトとコントローラを *2 次元適応リフレッシュポリシー* (2DRP) と協調設計し、[第 4.2 節](#section-4-2)で述べる細粒度で動的なリフレッシュ間隔を設定する。

<span id="section-3-3-2"></span>

#### 3.3.2 KV キャッシュの追い出し

KV キャッシュを小さくすると eDRAM のデータ格納量を大幅に削減でき、リフレッシュエネルギーを抑え、システム性能を改善できる。先行研究では、重要でない token を追い出しても生成品質が損なわれないことが観測されている。しかし重要でない token を特定するため、従来手法は系列の profiling [Ge24, Liu24p] または追加計算 [Zha23g, Xia24a] を必要とする。KV キャッシュを効率よく管理するため、[第 4.1 節](#section-4-1)で述べる *attention に基づく追い出し・再計算ポリシー* (AERP) の処理を高速化する新しい *systolic evictor* アーキテクチャを提案する。

<span id="section-3-3-3"></span>

#### 3.3.3 KV ベクトルの再計算

系列長が増えると、片外メモリへのアクセス時間が一部の KV tensor の再計算時間を上回る閾値に達し、KV キャッシュの利点が小さくなる。再計算は、[第 4.1 節](#section-4-1)に示すように、一時データの格納に適する eDRAM の長所とよく合う。ただし、再計算と格納の釣り合いを取るには、ハードウェアの特徴を考慮した慎重なスケジューリングが必要である。そこで計算パターンを設計して KV ベクトルのデータ生存期間を短縮する *Kelle Scheduler* を提案する ([第 6 節](#section-6))。

<span id="section-4"></span>

## 4 Kelle アルゴリズム

本節では、[図 5](#figure-05)の概要に沿って Kelle フレームワーク内の効率的なアルゴリズムを示す。実行中、Kelle は *attention に基づく追い出し・再計算ポリシー* (AERP) と *2 次元適応リフレッシュポリシー* (2DRP) を使用して eDRAM の動作を管理する ([第 4.1 節](#section-4-1)、[第 4.2 節](#section-4-2))。

<span id="section-4-1"></span>

### 4.1 Attention に基づく追い出し・再計算ポリシー

まず decoding 段階で eDRAM が満杯になったときの追い出しポリシーを説明する。

<span id="section-4-1-1"></span>

#### 4.1.1 追い出しポリシー

最大 $N^{\prime}$ token を保持できる容量の限られた KV キャッシュでは、decoding 段階で $(N^{\prime}+1)$ 番目の token が到着すると、いずれかの token $n$ ($1\leq n\leq N^{\prime}$) の KV ベクトル $[k^{h}_{n},v^{h}_{n}]$ を追い出す必要がある。追い出す $h$ 番目の head、$n$ 番目の token の KV ベクトルは、その重要度 $s^{h}_{n}$ に基づいて選ぶ。重要度は KV キャッシュ内の他の全 token との attention score ([式 1](#equation-01)) の和として計算する。

<span id="equation-03"></span>

$$
s^{h}_{n}=\sum_{1\leq i\leq n}A^{h}_{n,i}
$$

[図 6](#figure-06)に追い出し処理の例を示す。KV キャッシュには合計 $N^{\prime}=4$ ベクトルを格納できるものとする。3 つの attention head がある場合を考える。分かりやすさのため、最初の head の計算だけを描き、head の表記は省略する。$[k_{5},v_{5}]$ が到着すると、まず[式 3](#equation-03)で重要度を計算する ([図 6a](#figure-06))。次に、重要度が最も小さい token (3 番目の token) の KV ベクトルを追い出す ([図 6b](#figure-06))。$y_{N}$ の計算が KV ベクトルの相対順序に影響されないことを利用すると、元の token index を気にせず、キャッシュから順番に KV ベクトルを読み出して[式 1](#equation-01)と[式 2](#equation-02)を計算できる。同じ token $n$ の重要度 $s_{n}^{h}$ は attention head によって異なる場合がある点に注意が必要である。このため、KV ベクトルの追い出しパターンも head h ごとに異なる。

context token 長が $N_{cxt}$ の pre-filling 段階では、すべての context token を並列処理する。各 layer の各 head について、N 番目の token の重要度を $s^{h}_{N}=\sum_{1\leq n\leq N_{cxt}}A^{h}_{n,N}$ と計算する。$s^{h}_{n}$ が最も高い上位 $N^{\prime}$ token を decoding 用の KV キャッシュに保持する。

重要度 $s^{h}_{n}$ が最も高い token に加え、初期 token と直近 token も保持する。これらがモデル性能に影響することは先行研究 [Xia24a, Zha23g] で示され、本研究の実験でも確認されているためである。

<span id="section-4-1-2"></span>

#### 4.1.2 再計算ポリシー

<span id="figure-06"></span>

![(a) 3 つの head それぞれにおける重要度の計算。(b) 最低スコアの token の KV ベクトルを新しい KV ベクトルで置き換える。第 4 token は 3 head 中 2 head で重要なため、入力ベクトル $x_{4}$ を格納する。$x_{4}$ の格納で eDRAM entry が 1 つ空き、eDRAM のリフレッシュコストが減る。(c) eDRAM の記憶容量を節約するため、第 4 token の KV ベクトルを再計算する。](../../papers/kelle/figure-06.png)

**図 6.** (a) 3 つの head それぞれにおける重要度の計算。(b) 最低スコアの token の KV ベクトルを新しい KV ベクトルで置き換える。第 4 token は 3 head 中 2 head で重要なため、入力ベクトル $x_{4}$ を格納する。$x_{4}$ の格納で eDRAM entry が 1 つ空き、eDRAM のリフレッシュコストが減る。(c) eDRAM の記憶容量を節約するため、第 4 token の KV ベクトルを再計算する。

[第 2.3 節](#section-2-3)で述べたように、eDRAM は一時データの格納に適している。追い出しポリシーはモデル実行中に保持する KV ベクトル数を減らすが、生存期間の長い KV ベクトルを eDRAM に格納すると、必要なリフレッシュによるコストは依然として大きい。このコストを抑えるため、再計算をさらに適用できる。具体的には、KV キャッシュ内の token の部分集合 $N_{\mathrm{recomp}}$ について、対応する入力ベクトル $x_{N}$ を用いて KV ベクトルを再計算する。$x_{N}$ は[図 1b](#figure-01)に示す $W_{Q},W_{K},W_{V}$ への入力である。再計算を利用すると、2 ベクトル (K と V) の保持から 1 ベクトル (入力 x) の保持へ記憶要件を減らせる。この方式では必要に応じて K と V を再計算でき、KV ベクトルの長いデータ生存期間を実質的に短縮できる。

decoding 段階では、まず入力ベクトル $x_{N}$ に $W_{K}$ と $W_{V}$ を乗じて、全 head $h\in H$ の KV ベクトル $k^{h}_{N}$ と $v^{h}_{N}$ を再計算し ([図 6c](#figure-06))、decoding に使用する。再計算で KV キャッシュ容量を節約するには、次元 $1\times C$ の入力ベクトル $x_{N}$ の格納コストが、再計算する KV ベクトルのコストより小さくなければならない。この条件を満たすため、再計算しなければ token $N$ の KV ベクトルが少なくとも $\theta\gt 50\%$ の head で保持される場合、$x_{N}$ から KV ベクトルを再計算する。$\theta$ は token の**人気度**を表す。KV ベクトルの格納コスト $2\times\frac{C}{H}\times\theta H$ が $x_{N}$ のサイズ ($C$) を上回るため、この選択が妥当である。[図 6b](#figure-06)では第 4 token が 3 head 中 2 head で人気であるため、KV ベクトルを格納せず入力ベクトル $x_{4}$ を保持する。

記憶容量の節約に加え、再計算した KV ベクトルは[式 1](#equation-01)の計算に短時間だけ使う一時データとなり、eDRAM の利点をさらに活用できる。さらに、[第 5.2 節](#section-5-2)で説明する計算エンジンの systolic array アーキテクチャにより、再計算の追加コストは小さい。

pre-filling 段階では、まず head $h$ の各 token $n$ について重要度 $s_{n}^{h}$ を計算する。次に各 head $h$ で、対応する token の重要度に基づき KV ベクトルを追い出す。重要度の高い token のうち、少なくとも $50\%$ の head で KV ベクトルを保持するもの (人気 token) は入力ベクトル $x_{n}$ を格納し、それ以外は KV ベクトルを格納する。decoding 中、新しい token の格納形式は人気度 $\theta$ を計算して動的に決定する。[図 7a](#figure-07)に AERP 全体をまとめる。token の人気度は decoding 中に変動し得るが、実験では変動が限定的であり、$50\%$ を超える head で重要な token の重要度が下がることはほとんどない。したがって Kelle では、一度入力ベクトル形式で格納した token は、追い出されない限り decoding 全体を通して形式を変えない。

<span id="section-4-2"></span>

### 4.2 2 次元適応リフレッシュポリシー

<span id="figure-07"></span>

![(a) AERP の概要。簡単のため head h を 1 つだけ示す。(b) 2 次元適応リフレッシュポリシー。(c) 2DRP の例。$k_{5}[15:8]$ は第 5 token の key ベクトルの bit 8 から 15 を表す。濃い色ほど頻繁にリフレッシュされ、retention error rate が低い。](../../papers/kelle/figure-07.png)

**図 7.** (a) AERP の概要。簡単のため head h を 1 つだけ示す。(b) 2 次元適応リフレッシュポリシー。(c) 2DRP の例。$k_{5}[15:8]$ は第 5 token の key ベクトルの bit 8 から 15 を表す。濃い色ほど頻繁にリフレッシュされ、retention error rate が低い。

<span id="figure-08"></span>

![(a) bit-flip error rate P に対する PPL。(b) (a) HST と LST、(b) MSB と LSB のみに bit flipping を適用した場合の、bit-flip error rate ごとの LLM 精度。$P$ は error rate を表す。PPL が低いほど性能がよく、赤い数字は PPL の差を表す。](../../papers/kelle/figure-08.png)

**図 8.** (a) bit-flip error rate P に対する PPL。(b) (a) HST と LST、(b) MSB と LSB のみに bit flipping を適用した場合の、bit-flip error rate ごとの LLM 精度。$P$ は error rate を表す。PPL が低いほど性能がよく、赤い数字は PPL の差を表す。

精度を損なわずに LLM が KV キャッシュ内のデータ破損をどこまで許容できるか調べるため、eDRAM メモリセルに bit-flip error を導入して retention failure をシミュレーションする。具体的には、Wikitext-2 [Mer17] データセットを使用し、LLaMA2-7B の perplexity (PPL) への影響を評価する。PPL が低いほど性能がよい。実行中、KV キャッシュに一様な確率で bit-flip error を導入する。[図 8a](#figure-08)の結果では、error rate が $10^{-3}$ 未満なら PPL の増加は 0.1 未満と小さい。しかし bit-flip error が増えると PPL は大きく増加する。したがって LLM は一定量の KV キャッシュエラーを許容できる。ここから次の問いが生じる。*精度を保ちながら、さらに低いリフレッシュ頻度を可能にする細粒度なリフレッシュポリシーを開発できるか。*

[第 4.1 節](#section-4-1)では、[式 3](#equation-03)の重要度に基づいて token を追い出した。同様の方法を eDRAM のリフレッシュにも適用し、重要度の低い token の KV ベクトルまたは入力ベクトルには低い頻度を、重要度の高い token には高い頻度を割り当てられると仮定する。この仮説を検証するため、適応的なリフレッシュポリシーを実装し、実験を反復した。簡単のため、重要度に基づいて token を high score token (HST) group と low score token (LST) group に分けた。KV ベクトルにおける bit retention failure (bit-flip error) の確率 $p$ を、HST と LST の対応する token の KV ベクトルへ別々に適用した。[図 8b](#figure-08)では、retention failure が LST group より HST group に生じたときに LLM の性能低下が大きい。これは HST group の token に高いリフレッシュ頻度が必要であることを示し、仮説を支持する。

同様に、least significant bit (LSB) の bit-flip error は値の変化が小さいため、LSB は most significant bit (MSB) より retention failure error の影響を受けにくいと仮定できる。KV ベクトルの各値について、MSB (bit 15-8) または LSB (bit 7-0) のいずれかに bit retention error を導入する。[図 8c](#figure-08)の結果では、同じ bit-flip error rate のもとで MSB は LSB より retention error に敏感であり、仮説をさらに裏付ける。

以上の観察に基づき、[図 7b](#figure-07)に示す *2 次元適応リフレッシュポリシー* (2DRP) を提案する。この戦略は、KV ベクトルまたは入力ベクトル内の各値における bit 位置と各 token の重要度の両方に基づいて、各 eDRAM セルのリフレッシュ頻度を調整する。[図 7c](#figure-07)は KV キャッシュが最大 $N^{\prime}=3$ token を保持する 2DRP の例である。token の重要度と bit 位置の重要性が高いほど、リフレッシュ頻度を上げる。実行中、KV ベクトルと入力ベクトルの重要度を動的に計算し、重要度と bit 位置に応じてリフレッシュ頻度を割り当てる。

<span id="section-5"></span>

## 5 Kelle エッジアクセラレータ

<span id="figure-09"></span>

![Kelle ハードウェアアクセラレータの概要。](../../papers/kelle/figure-09.png)

**図 9.** Kelle ハードウェアアクセラレータの概要。

[図 9](#figure-09)は Kelle アクセラレータの概要を示す。ハイブリッド eDRAM-SRAM メモリサブシステム、reconfigurable systolic array (RSA)、specialized function unit (SFU) を組み込む。重みは 8 bit に量子化し、activation と KV ベクトルは 16 bit のままとする。重みを SRAM に、activation と KV ベクトルを eDRAM に格納する。実行中、systolic evictor は attention score を累積し、eDRAM コントローラは KV ベクトルの追い出しと再計算を処理しながら、[第 4.1 節](#section-4-1)と[第 4.2 節](#section-4-2)で述べたようにリフレッシュ頻度を動的に調整する。RSA の各 processing element (PE) は 8-bit multiply-accumulate (MAC) 演算を行う。

SFU は活性化関数、softmax、正規化、position embedding などの非線形演算を処理する。先行研究 [Wan20b, Xia24g, Das22, Qin25b] が示すように、非線形演算のエネルギー消費は入力系列長に伴って増加する。中でも softmax は多くの資源を消費する。メモリアクセスを最小化するため Softermax [Ste21] の online max 計算を採用する。他の非線形演算では、計算フローに従い lookup table (LUT) を使って計算する。

<span id="section-5-1"></span>

### 5.1 メモリサブシステム

[図 10](#figure-10)は Kelle アクセラレータのメモリサブシステムを示す。この設計では 2MB SRAM が重みを格納し、activation と KV ベクトルをそれぞれ 256KB の *activation eDRAM* と 4MB の *KV cache eDRAM* に格納する。Kelle は重要度と bit 位置に基づいて KV ベクトルを 4 group に分け、それぞれにリフレッシュ頻度を適用することで 2DRP を実装する。具体的には、HST group の token の KV ベクトルにおける MSB (bit 15-8) は最高頻度で、LST group の token における LSB (bit 7-0) は最低頻度でリフレッシュする。AERP を支援するため、一部の token では KV ベクトルの代わりに入力ベクトルを KV cache eDRAM に格納する。これらの入力ベクトルも重要度と bit width によって 4 group に分け、KV ベクトルと同じ方法で制御する。簡単のため、以降は入力ベクトルに言及せず KV ベクトルを用いてメモリサブシステムを説明する。

LLM 推論中に 2DRP を実行するため、KV ベクトルの各要素を bit ごとに分割し、異なる eDRAM bank に格納する。具体的には、KV ベクトルの MSB と LSB を *MSB bank* と *LSB bank* と呼ぶ別々の KV cache eDRAM bank に格納し、[図 10](#figure-10)では濃色と淡色で示す。各 token の重要度は[式 3](#equation-03)を使って 4-bit precision で動的に計算し、register file に格納する。各 entry は 4 bank にまたがる KV ベクトルに対応する。同じ token の KV ベクトルは、異なる eDRAM bank で同じ address を共有する。全 4 bank の AERP を管理する 1 つの eviction controller と、MSB bank と LSB bank に対して 2DRP を個別に実行する 2 つの refresh controller を備える。

各 MSB bank と LSB bank では token を attention score によってさらに 2 group に分け、refresh controller 内の counter が各 group のリフレッシュ間隔を監視する。controller は eDRAM entry を走査し、register file から attention score を読み取って各 token の group を特定する。特定 group のリフレッシュ間隔が満了すると *refresh* signal を発行する。その group の token に対応する KV ベクトルの address を計算し、KV ベクトルを読み出して書き戻す。モデルが KV ベクトルを使わないときにリフレッシュを行うため、そのレイテンシは隠蔽できる。KV キャッシュが満杯のときに新しい token が到着すると、eviction controller は systolic evictor から追い出す token index を受け取り、その token を新しい token で置き換える。

$32\times 32$ RSA へ並列にデータを供給して bank conflict を避け、RSA を最大限に利用するため、Kelle KV キャッシュを 32 bank に分割する。Key MSB、Key LSB、Value MSB、Value LSB の各 group に 8 bank を割り当てる。この設計と pipeline 化した cache read により、Kelle eDRAM は bank conflict を起こさず RSA を十分に活用できる帯域幅を提供する。token read や token eviction など他の eDRAM access も独立して動作するため、bank conflict を実質的に軽減する。

LLM 実行中、RSA I/O controller は異なる bank のデータを小さなオーバーヘッドで効率よく再構成し、計算に用いる。また Kelle は一部の LLM layer の KV ベクトルを eDRAM に格納し、layer 数は具体的な LLM のサイズとテキスト長から決める。eDRAM により片外メモリアクセスのオーバーヘッドが大幅に減る。

<span id="figure-10"></span>

![Kelle メモリサブシステム。一部 token の入力ベクトルを赤い行で示し、KV キャッシュに格納する。](../../papers/kelle/figure-10.png)

**図 10.** Kelle メモリサブシステム。一部 token の入力ベクトルを赤い行で示し、KV キャッシュに格納する。

<span id="section-5-2"></span>

### 5.2 Reconfigurable Systolic Array

systolic array core は $32\times 32$ の 2 次元 array であり、入力をずらしながら処理し、計算した部分和を accumulator と SFU へ送る。[図 11a](#figure-11)に示す weight-stationary dataflow を使用する。FAST [Zha21g] と同様の reconfigurable strategy を採用し、その場で転置行列乗算を行う。

[第 4.1 節](#section-4-1)の再計算が LLM decoding 段階にもたらすオーバーヘッドは小さい。行列演算に適する systolic array の長所を利用し、再計算する token ベクトルと現在の token の入力ベクトルを効率よく結合して入力行列を作れる。[図 6b](#figure-06)の記法を用いると、[図 11a](#figure-11)は現在の token の入力ベクトル $x_{5}$ を RSA へ送り、KV ベクトルを計算する様子を示す。第 4 token の KV ベクトルを再計算するには、$x_{4}$ と $x_{5}$ を 1 つの行列に結合でき、[図 11b](#figure-11)に示すようにレイテンシとエネルギーの増加は小さい。

<span id="figure-11"></span>

![(a)、(b) 再計算が RSA の動作へ与える影響。(c) RSA と systolic evictor の統合。(d) systolic array と systolic evictor の実行順。赤い円内の数字は順番を表す。](../../papers/kelle/figure-11.png)

**図 11.** (a)、(b) 再計算が RSA の動作へ与える影響。(c) RSA と systolic evictor の統合。(d) systolic array と systolic evictor の実行順。赤い円内の数字は順番を表す。

<span id="section-5-3"></span>

### 5.3 Systolic Evictor

<span id="figure-12"></span>

![(a)、(b) ベースラインと Kelle scheduler における SA block の計算パターンと eDRAM データ生存期間。SM は softmax 演算を表す。](../../papers/kelle/figure-12.png)

**図 12.** (a)、(b) ベースラインと Kelle scheduler における SA block の計算パターンと eDRAM データ生存期間。SM は softmax 演算を表す。

AERP における token 追い出し処理は、[式 1](#equation-01)の attention score の計算、[式 3](#equation-03)に基づく重要度の更新、重要度が最低の token の特定、KV キャッシュの更新からなる。

追い出しを効率よく実装するため、systolic に動作し、RSA と統合して最小の重要度を on-the-fly に探索する systolic evictor (SE) を提案する。重要度は、[式 1](#equation-01)の $Q K^{T}$ の結果を softmax に通さず合計して計算する。この統合により、新しい token の attention score を RSA が計算すると同時に、重要度が最小の token を特定できる。最小の重要度を持つ token index が見つかると、SE はその index を eDRAM controller 内の eviction controller へ送り、該当 token を追い出す。[図 11c](#figure-11)に SE の設計と RSA への統合を示す。SE は[図 11c](#figure-11)で *S* と表す register の列を持ち、過去の token の重要度を preload する。また *M* と表す register chain が、最小重要度 (min) を上から下へ周期的に伝播する。[図 11d](#figure-11)は RSA と SE の実行順を示す。1 cycle で RSA の $i$ 番目の row が attention score を計算し、続いて SE の $i$ 番目の row が重要度と最小重要度の index を更新する。これらを Step 1 と Step 2 とする。次の cycle では RSA と SE の次の row で同じ操作を行い、Step 3 と Step 4 とする。systolic evictor により最小値探索がもたらす追加の LLM 実行レイテンシを避けられる。

<span id="section-6"></span>

## 6 Kelle Scheduler

eDRAM のリフレッシュエネルギーをさらに最小化するため、精度を損なわずにデータ生存期間を短縮し、LLM 推論を高速化する新しい計算パターンを導入する。

まず LLM decoding 段階の self-attention (SA) アーキテクチャにおけるデータ生存期間を数値的に解析する。[第 2.1 節](#section-2-1)で示したように、SA では最初に入力 $X$ と重み行列 $W_{Q}$、$W_{K}$、$W_{V}$ の行列乗算を行い、出力 $Q,K,V$ を得る。各処理を $\mathrm{MM}_{Q}$、$\mathrm{MM}_{K}$、$\mathrm{MM}_{V}$ と表す。次に $Q$ と $K$ を乗算し、softmax で attention score $A$ を計算する。各処理を $\mathrm{MM}_{qk}$ と $\mathrm{SM}$ と表す。最後に $A$ と重み行列 $W_{O}$ を乗じて SA の出力を得る。これを $\mathrm{MM}_{O}$ と表す。行列乗算のレイテンシ ($T_{\mathrm{MM}}$) を次のように見積もる。

<span id="equation-04"></span>

$$
T_{\mathrm{MM}}=\frac{N_{\mathrm{MM}}}{\mathrm{TOP}_{\mathrm{RSA}}}
$$

$N_{\mathrm{MM}}$ は行列乗算に必要な MAC 演算数を表す。$\mathrm{TOP}_{\mathrm{RSA}}$ は[第 5.2 節](#section-5-2)で述べた RSA の throughput である。KV ベクトルの eDRAM access operation のレイテンシ $T_{\mathrm{eDRAM}}$ は次のようにモデル化する。

<span id="equation-05"></span>

$$
T_{\mathrm{eDRAM}}=\frac{S_{\mathrm{KV}}}{B_{\mathrm{eDRAM}}}
$$

$S_{\mathrm{KV}}$ は KV ベクトルの byte 数、$B_{\mathrm{eDRAM}}$ は eDRAM の帯域幅を表す。同様に、重みの SRAM access operation のレイテンシ $T_{\mathrm{SRAM}}$ を次のようにモデル化する。

<span id="equation-06"></span>

$$
T_{\mathrm{SRAM}}=\frac{S_{W}}{B_{\mathrm{SRAM}}}
$$

$S_{W}$ は重みの byte 数、$B_{\mathrm{SRAM}}$ は SRAM の帯域幅を表す。

<span id="table-02"></span>

![各手法の精度。FP16 は KV キャッシュを削減しない FP16 での LLM 精度を表す。](../../papers/kelle/table-02.png)

**表 2.** 各手法の精度。FP16 は KV キャッシュを削減しない FP16 での LLM 精度を表す。

[図 12a](#figure-12)は、行列乗算 $\mathrm{MM}_{Q}$、$\mathrm{MM}_{K}$、$\mathrm{MM}_{V}$、$\mathrm{MM}_{qk}$ を順に実行するベースラインの計算パターンを示す。この順序は入力 $X$、$Q$、$K$、$V$ のデータ生存期間を延ばす。データ生存期間は、データが計算されてから後続の操作で使われるまでの期間と定義する。例えば[図 12a](#figure-12)では、SRAM から重み行列 $W_{Q}$ へアクセスした後、$t_{1}$ でベクトル $Q$ の計算を開始し、$t_{4}$ で $Q$ と $K$ の乗算が始まると $Q$ が消費される。$t_{1}$ と $t_{4}$ の間に SRAM から $W_{K}$ と $W_{V}$ を読み込み、eDRAM KV キャッシュから $K$ にアクセスする。$W_{K}$ と $W_{V}$ へのアクセスレイテンシはともに $T_{\mathrm{SRAM}}$、$K$ へのアクセスレイテンシは $T_{\mathrm{eDRAM}}$ である。したがって $Q$ のデータ生存期間は $2\times T_{\mathrm{SRAM}}+T_{\mathrm{eDRAM}}$ となる。すべての activation は eDRAM に格納され、リフレッシュが必要なため、全 activation の総データ生存期間は各 activation の生存期間の和となる。[式 4](#equation-04)の計算時間 $T_{\mathrm{MM}}$ は $T_{\mathrm{SRAM}}$ と $T_{\mathrm{eDRAM}}$ に比べて無視できるため省略する。この長い生存期間は eDRAM のリフレッシュコストを増やす。ベースラインスケジュールの一時データの総生存期間 $L_{bl}$ を次のようにモデル化する。

<span id="equation-07"></span>

$$
\begin{split}L_{X}&=3\times T_{\mathrm{SRAM}},L_{Q}=2\times T_{\mathrm{SRAM}}+T_{\mathrm{eDRAM}}\\
L_{K}&=T_{\mathrm{SRAM}}+T_{\mathrm{eDRAM}},L_{V}=2T_{\mathrm{eDRAM}}\\
L_{bl}&=L_{X}+L_{Q}+L_{K}+L_{V}=6T_{\mathrm{SRAM}}+4T_{\mathrm{eDRAM}}\end{split}
$$

$L_{X}$、$L_{Q}$、$L_{K}$、$L_{V}$ はそれぞれ $X$、$Q$、$K$、$V$ のデータ生存期間を表す。$T_{\mathrm{SRAM}}$ と $T_{\mathrm{eDRAM}}$ は[式 6](#equation-06)と[式 5](#equation-05)で定義した。これに対し、Kelle の計算パターンを[図 12b](#figure-12)に示す。独立した片上メモリの統合により、重みと KV ベクトルへのメモリアクセスを並列化する。この構成は activation のデータ生存期間を短縮し、次のように見積もれる。

<span id="equation-08"></span>

$$
\begin{split}L_{X}&=3\times T_{\mathrm{SRAM}},L_{Q}=T_{\mathrm{SRAM}}+T_{\mathrm{eDRAM}}\\
L_{\mathrm{Kelle}}&=L_{X}+L_{Q}=4T_{\mathrm{SRAM}}+T_{\mathrm{eDRAM}}\end{split}
$$

key と value のベクトルは各計算に直ちに使うため、長期格納は不要であり、データ生存期間を無視できる。ベースラインと比べ、Kelle scheduler は eDRAM の一時データ生存期間を大幅に短縮し、リフレッシュエネルギーを減らしてシステム性能を改善する。

<span id="section-7"></span>

## 7 精度評価

<span id="section-7-1"></span>

### 7.1 主な精度結果

異なるサイズの Llama2 [Oth23a]、Llama3 [Dub24]、Llama3.2 [Dub24]、Mistral [Jia23a]、QWEN [Yang24b]、OPT [Zha22] を含む各種 LLM で Kelle を評価する。言語生成タスクでは WikiText-2 (WK2) [Mer17] と PG19 [Rae20] の perplexity により評価する。WK2 の系列長は数百から数千 token、PG19 は数万から数百万 token に及ぶ。Cold Compress framework [Pyt23, Ada24] を使った PG19 テキスト生成タスクでは、モデルへ書名と短い説明を与え、系列生成長を 8192 に設定する。PIQA (PQ) [Bis20]、Lambada (LA) [Rad19b]、Arc Easy (A-e) [Cla18]、Arc Challenge (A-c) [Cla18]、TriviaQA (TQ) [Jos17]、Qasper (QP) [Das21] などの zero-shot task でも Kelle を評価する。LM Evaluation Harness [Gao21] を既定パラメータで使用する。

KV ベクトルの追い出しでは、pre-filling と decoding の両段階で、KV キャッシュに保持する token 数をデータセットに応じて動的に調整する。低い eDRAM リフレッシュ頻度による bit-flip error を模擬するため、リフレッシュ間隔に基づく所定の確率で bit-level retention failure を導入する。HST の MSB (bit 15-8)、HST の LSB (bit 7-0)、LST の MSB、LST の LSB のリフレッシュ間隔をそれぞれ 0.36ms、5.4ms、1.44ms、7.2ms とし、平均 retention time を 1.05ms とする。平均 retention failure rate は 2e-3 となる。

<span id="table-03"></span>

![異なるキャッシュサイズにおける LLaMA2-7B の精度。](../../papers/kelle/table-03.png)

**表 3.** 異なるキャッシュサイズにおける LLaMA2-7B の精度。

Kelle の精度を最新の量子化 framework QuaRot (QR) [Ash24] と比較する。さらに、近年の KV キャッシュ追い出し手法 StreamLLM [Xia24a] と H2O [Zha23g] も比較対象とする。すべての手法でモデル重みを 8 bit に量子化する。量子化と KV キャッシュ追い出しのベースラインで KV キャッシュ予算を揃えるため、QuaRot の KV ベクトルを 4 bit に量子化し、StreamLLM、H2O、Kelle は量子化せず 16 bit とする。Kelle の token 格納予算は、PQ、LA、A-e、A-c で $N^{\prime}=128$、WK2 で $N^{\prime}=512$、TQ と QP で $N^{\prime}=1024$、PG19 で $N^{\prime}=2048$ とする。token 予算内で直近 token の window size は PQ、LA、A-e、A-c で 64、WK2 で 256、TQ と QP で 512、PG-19 で 1024 とする。各データセットで最初の 10 token も保持する。StreamLLM と H2O の token 格納予算も Kelle と同じにする。KV キャッシュを追い出さない元の FP16 モデルも比較し、FP16 と表す。[表 2](#table-02)に示すように、Kelle は full KV キャッシュの元モデルと同等の精度を保ち、他手法を上回るか同等の性能を達成する。これは AERP と 2DRP の高い精度を示す。

<span id="section-7-2"></span>

### 7.2 アブレーション研究

Llama2-7B の予算 $N^{\prime}$ を変え、各タスクへの影響を調べる。Kelle の他の設定 (量子化 bit width、retention failure rate など) は変えない。[表 3](#table-03)では、予算 $N^{\prime}$ の減少につれて精度が一貫して下がるが、$N^{\prime}\geq 128$ なら pruning しない KV キャッシュ ([表 3](#table-03)の Full) と比べても妥当な性能を保つ。

<span id="table-04"></span>

![異なるリフレッシュ間隔における LLaMA2-7B の精度。](../../papers/kelle/table-04.png)

**表 4.** 異なるリフレッシュ間隔における LLaMA2-7B の精度。

<span id="figure-13"></span>

![Kelle とベースラインシステムの比較。正規化エネルギー効率と speedup で性能を評価する。円グラフは Kelle+eDRAM の主要部品における片上エネルギーの内訳を示す。赤い点線は各設定の speedup を表す。](../../papers/kelle/figure-13.png)

**図 13.** Kelle とベースラインシステムの比較。正規化エネルギー効率と speedup で性能を評価する。円グラフは Kelle+eDRAM の主要部品における片上エネルギーの内訳を示す。赤い点線は各設定の speedup を表す。

次に、2DRP が LLM 精度へ与える影響を調べる。具体的には、すべての eDRAM セルが同じリフレッシュ間隔を共有し、平均 retention failure rate を 2DRP と同じに保つ条件と 2DRP を比較する。Kelle の他の条件は変えない。[表 4](#table-04)では、Llama2-7B の各タスクについて、リフレッシュ間隔を変えたときの精度変化を示す。[表 4](#table-04)の Uniform ($\mu$s) は eDRAM に適用する一様なリフレッシュ間隔を表す。HST 行の 2 数値は HST の MSB と LSB のリフレッシュ間隔を表し、LST 行も同様である。すべての条件とデータセットで、2DRP は一様な eDRAM リフレッシュより精度を改善する。

エッジへの配備はユーザ向けアプリケーションを伴うことが多いため、2DRP が導入する近似的なメモリ動作がテキスト生成の定性指標へ与える影響を評価する必要がある。一貫性の評価では、CNN/DailyMail [Nal16] (CNN) 要約データセットを使用して LLaMA2-7B と Mistral-7B で Kelle を実行し、ROUGE-1 score を報告する。事実の正確性は TruthfulQA benchmark [Lin22] (Truth) で Kelle をテストし、選択肢が 1 つの multiple-choice accuracy を報告する。bias の傾向は BBQ benchmark [Par22b] で評価し、両モデルの bias score を報告する。[表 6](#table-06)では、Kelle がすべての基準で FP16 モデルと同等の性能を達成する。

最後に、Hadamard 変換で low-bit LLM 量子化を可能にする QuaRot framework [Ash24] を用いて Llama2-7B を量子化する。モデル重みを 4 bit、KV ベクトルと activation を 8 bit に量子化する。[表 6](#table-06)に示すように、量子化により精度への影響を小さく保ちながら Kelle のシステム性能をさらに改善できると見込まれる。これは Kelle とモデル量子化技術の互換性を示す。

<span id="table-05"></span>

![Kelle の定性指標](../../papers/kelle/table-05.png)

**表 5.** Kelle の定性指標

<span id="table-06"></span>

![量子化した Kelle の精度](../../papers/kelle/table-06.png)

**表 6.** 量子化した Kelle の精度

<span id="section-8"></span>

## 8 ハードウェア評価

本節では、[第 5 節](#section-5)で述べた Kelle エッジアクセラレータのハードウェア評価結果を報告する。Kelle エッジアクセラレータは 2 次元 $32\times 32$ RSA、SFU、必要な interface、memory controller で構成され、すべて SystemVerilog により RTL で実装し、周波数を 1GHz とする。45nm NanGate Open Cell Library [Nan10] と Synopsys Design Compiler [Bal19] を用いて部品を合成し、Kelle の面積と電力を報告する。重み格納用 SRAM を 2MB、KV キャッシュと activation 格納用 eDRAM をそれぞれ 4MB と 256KB とする。SRAM と eDRAM の帯域幅はそれぞれ 128GB/s と 256GB/s とする。Destiny [Por15] を用い、$105^{\circ}C$、65nm technology node における eDRAM と SRAM の面積、電力、timing を評価する。eDRAM の retention time 分布は[図 4](#figure-04)のデータと一致する [Kon08, Zha23l]。$105^{\circ}C$ 未満で動作する eDRAM は retention time がさらに長く、システム性能も改善する。Cacti 7 [Bal17] を用い、Google Coral エッジデバイス [Sur20] の DRAM に近い、帯域幅 64GB/s の 16GB LPDDR4 DRAM の性能をシミュレーションする。この設定で片上の総面積は $9.5mm^{2}$、RSA、eDRAM、SRAM、SFU の面積比はそれぞれ $23\%$、$33\%$、$37\%$、$7\%$ である。DRAM の面積は $16mm^{2}$ である。片上電力は 6.52W、RSA、eDRAM、SRAM、SFU の電力比はそれぞれ $17\%$、$29\%$、$41\%$、$13\%$ である。DRAM の電力は 11.74W である。Kelle は 4.13 INT8 TOPs を達成する。[第 6 節](#section-6)の Kelle scheduler により eDRAM のコストをさらに削減する。

Lambada (LA) [Rad19b]、TriviaQA (TQ) [Jos17]、Qasper (QA) [Das21]、PG19 [Rae20] を含む複数のタスクと LLM アーキテクチャで Kelle のハードウェア性能を評価する。context 長はそれぞれ 128、512、1024、512、decoding 長は 512、2048、5120、8192 とする。batch size は 16 とする。すべての評価結果に片外 DRAM のアクセスレイテンシとエネルギーを含める。

<span id="figure-14"></span>

![LLM アクセラレータの比較](../../papers/kelle/figure-14.png)

**図 14.** LLM アクセラレータの比較

<span id="section-8-1"></span>

### 8.1 End-to-End 性能評価

<span id="section-8-1-1"></span>

#### 8.1.1 評価ベースライン

[第 5 節](#section-5)の Kelle アルゴリズムと eDRAM ベースのアクセラレータの寄与を個別に理解するため、Kelle アルゴリズムと eDRAM ベースの Kelle アクセラレータを組み合わせた **Kelle+eDRAM** を 4 つのベースラインと比較する。

第 1 の **Original+SRAM** は、SRAM を主要な片上ストレージとするシステムで元の LLM を実行する。モデル重みを 8 bit に量子化し、activation と KV ベクトルは 16 bit のまま、8-bit MAC 演算に設定した Kelle RSA で処理する。AERP は適用せず KV キャッシュを完全に保持する。SRAM システムの総片上面積は Kelle+eDRAM と一致させる。計算/メモリ IO 比が均衡するよう SRAM と systolic array のサイズを調整し、$24\times 24$ 個の 8-bit PE、4MB 片上 SRAM、16GB 片外 DRAM とする。第 2 の **Original+eDRAM** は、KV キャッシュを完全に保持しながら、eDRAM ベースの Kelle アクセラレータで元の LLM を実行する。モデルは 8-bit MAC 演算に設定した Kelle RSA で処理する。このベースラインはアルゴリズム上の工夫をすべて除き、eDRAM システムのみの性能を評価する。第 3 の **AEP+SRAM** では、[第 7.1 節](#section-7-1)の設定で attention に基づく追い出しを適用して KV キャッシュを pruning し、Original+SRAM と同じ SRAM システムに実装する。これはキャッシュ追い出しアルゴリズムが SRAM システムへ与える影響を評価するためである。このベースラインは再計算を行わない。第 4 の **AERP+SRAM** は、SRAM ベースの Kelle アクセラレータで AERP を実行する。

<span id="section-8-1-2"></span>

#### 8.1.2 End-to-End 性能の改善

[図 13](#figure-13)は、複数の LLM とデータセットにおける上記ベースラインのエネルギー効率と処理レイテンシを比較する。平均して Kelle+eDRAM は Original+SRAM に対し、レイテンシを $3.94\times$、エネルギー効率を $4.46\times$ 改善し、decoding 系列が長いほど性能差が大きくなる。Kelle の高い性能は AERP と 2DRP のアルゴリズムに加え、効率的な eDRAM memory controller、systolic evictor、Kelle scheduler などのハードウェア上の利点による。

<span id="section-8-1-3"></span>

#### 8.1.3 各最適化による性能改善

本節では Kelle の各最適化の影響を調べる。まず Original+SRAM と比べ、Original+eDRAM は平均 speedup を $32\%$ 改善するが、エネルギー効率を $39\%$ 低下させる。エネルギー消費の増加は、アルゴリズムやハードウェアによる最適化を行わない eDRAM リフレッシュに起因する。eDRAM は SRAM より容量が大きくアクセスが速いため speedup を改善する。次に AEP+SRAM と Original+SRAM を比較すると、systolic evictor で高速化した attention-based eviction policy はレイテンシを $2.39\times$ 削減し、エネルギー効率を $2.41\times$ 改善する。attention-based recomputation policy により、AERP+SRAM は AEP+SRAM に対して speedup を $1.19\times$、エネルギー効率を $1.27\times$ 改善する。最後に AERP でモデルを実行するシステムでは、2DRP と Kelle scheduler で最適化した eDRAM により、Kelle+eDRAM は AERP+SRAM に対して speedup を $1.29\times$、エネルギー効率を $1.45\times$ 改善する。2DRP はリフレッシュエネルギーを大幅に減らし、Kelle が eDRAM を十分に活用できるようにする。

<span id="section-8-1-4"></span>

#### 8.1.4 オーバーヘッド解析

[図 13](#figure-13)の円グラフは Kelle+eDRAM のエネルギー内訳を示す。KV キャッシュのエネルギー比率が下がることは、eDRAM と Kelle のアルゴリズムがメモリアクセスのボトルネックを緩和することを示す。systolic array は行列間乗算を効率よく処理するため、KV 再計算のハードウェアオーバーヘッドは小さく、RSA は片上エネルギーのわずかな割合しか消費しない。

token 追い出しを高速化するため、RSA と結合する小型計算 unit である Systolic Evictor を導入する。面積は $0.06mm^{2}$ (片上面積の $0.6\%$)、電力は 0.028W (片上電力の $0.4\%$) である。systolic evictor は KV キャッシュ追い出しや冗長なメモリ・計算アクセスによる LLM 実行の停止を避ける。これによりシステムのエネルギー効率を $5\%$ 改善し、レイテンシを $7\%$ 削減する。

<span id="section-8-2"></span>

### 8.2 他のアクセラレータとの比較

Kelle+eDRAM を他の先進 LLM アクセラレータと比較する。LLM.npu [Xu24f] は prompt とモデルを再構成し、デバイス上の Neural Processing Unit (NPU) への offload を改善して pre-filling のレイテンシを減らす。DynaX [Xio25b] は動的で細粒度な structured pruning を提案して sparse attention の計算効率を高め、$90\%$ の attention sparsity を実現する。DynaX は pre-filling 段階の計算ボトルネックを緩和する。COMET [Liu24x] は LLM を 4 bit に量子化し、mixed-precision 計算を支援する高性能 GPU kernel を設計する。本論文の主眼は高度な量子化ではないため、COMET の LLM 重みを 8 bit、activation と KV ベクトルを 4 bit に量子化し、Kelle+eDRAM と同等の KV キャッシュ予算を確保する。最後に、NVIDIA Jetson Orin エッジ GPU [Nvi22a] 上で FP8 を用いる LLM 実装と Kelle を比較する。測定には pynvml [Pyn17] と nvidia-smi [Nvi24e] を用いる。

[図 14](#figure-14)に示すように、Kelle+eDRAM は他の LLM アクセラレータより speedup とエネルギー効率を改善する。LLM.npu と DynaX は計算量の多い pre-filling 段階を最適化するが、LLM decoding 段階の KV キャッシュボトルネックには対処しない。Kelle が COMET を上回ることは、専用ハードウェアアクセラレータなしで KV キャッシュ圧縮だけに依存する方式の限界を示す。

<span id="section-8-3"></span>

### 8.3 アブレーション研究

<span id="section-8-3-1"></span>

#### 8.3.1 KV キャッシュ予算の影響

<span id="table-07"></span>

![複数の KV キャッシュ予算におけるエネルギー効率。](../../papers/kelle/table-07.png)

**表 7.** 複数の KV キャッシュ予算におけるエネルギー効率。

[表 7](#table-07)は異なる KV キャッシュ予算 $N^{\prime}$ における Kelle+eDRAM のエネルギー効率改善を示す。追い出しを行わない場合、PG19 で保持できる最大 token 数は $N^{\prime}=8750$ となる。この条件でも Kelle は Original+SRAM に対して約 $3\times$ 高いエネルギー効率を実現し、Kelle の堅牢性を示す。

<span id="section-8-3-2"></span>

#### 8.3.2 再計算の影響

<span id="figure-15"></span>

![(a) Kelle+eDRAM における KV キャッシュ再計算の影響。(b) 2DRP と Kelle scheduler の評価。](../../papers/kelle/figure-15.png)

**図 15.** (a) Kelle+eDRAM における KV キャッシュ再計算の影響。(b) 2DRP と Kelle scheduler の評価。

<span id="figure-16"></span>

![(a) KV キャッシュ再計算の影響。(b) 長い入力系列での評価。P と D はそれぞれ pre-filling と decoding 段階を表す。](../../papers/kelle/figure-16.png)

**図 16.** (a) KV キャッシュ再計算の影響。(b) 長い入力系列での評価。P と D はそれぞれ pre-filling と decoding 段階を表す。

KV キャッシュを再計算する場合としない場合の Kelle+eDRAM のエネルギー消費を比較する。[図 15a](#figure-15)に示すように、再計算アルゴリズムは RSA のエネルギーをわずかに増やすだけで、KV キャッシュのエネルギーを効果的に削減する。また複数の LLM アーキテクチャとタスクについて、pre-filling と decoding 段階での token 人気度の変化を profile する。平均すると、pre-filling 段階の人気 token の $86\%$ 超が decoding 段階でも人気を保ち、[第 4.1.2 節](#section-4-1-2)の実行戦略を裏付ける。

再計算により Kelle は片上に多くの token を格納し、DRAM access を減らせる。LLaMA2-7B の処理では、DRAM から 1 つの KV ベクトルへアクセスするのに約 $1.1\ \mu s$ かかる。これに対し、RSA による KV ベクトルの再計算は $3.2\ \mu s$ の追加レイテンシをもたらす。再計算は計算とメモリアクセスを重ねてメモリ停止を隠蔽し、総レイテンシを削減してエネルギー効率を平均 $25\%$ 改善する。例えば DRAM から 4 つの KV ベクトルを読み込むには $4.4\ \mu s$ かかる。再計算を用いると 3 ベクトルを読み込み、その間に 1 ベクトルを並列に再計算するため、総レイテンシは $3.3\ \mu s$ となる。入力ベクトル数にかかわらず RSA は動作するため、再計算による追加エネルギーは無視できる。

[図 16a](#figure-16)は No Recomp (再計算なし)、Recomp (適度な再計算)、Over Recomp (過剰な再計算) の 3 設定における Kelle の roofline model を示す。再計算は実効メモリ帯域幅を高めて性能を改善する。しかし再計算する KV ベクトルが増えると RSA がボトルネックとなる。Over Recomp の線は、Kelle が memory-bound から compute-bound へ移るこの動作を表す。

<span id="section-8-3-3"></span>

#### 8.3.3 2DRP と Kelle Scheduler の影響

Llama2-7B で PG19 を実行する Kelle+eDRAM を 4 つの戦略で評価する。**Org** は eDRAM を retention time に合わせて $45\mu s$ 間隔でリフレッシュし、データ破損をほぼ完全に防ぐ。**Uni** は一様な $0.36ms$ 間隔を用い、2DRP と同じ LLM 精度を実現する。**2D** と表す 2DRP は attention score と bit 位置に基づき異なる間隔を適用する。**2K** は 2DRP と Kelle scheduler の両方を組み合わせる。[図 15b](#figure-15)に示すように、2DRP の細粒度なリフレッシュはエネルギー効率を改善する。2DRP と Kelle scheduler の両方を用いると Kelle は最良の性能を達成する。

<span id="section-8-3-4"></span>

#### 8.3.4 eDRAM Retention Time の影響

bit failure rate への影響を考慮し、eDRAM retention time が Kelle の性能に与える影響を評価する。retention time は設計、technology node、温度などの要因に左右される [Git20, Kon08, Zha23l]。異なる retention time の 2DRP を使い、TriviaQA と PG19 で Kelle+eDRAM を評価する。具体的には、Kelle の retention time ($45\mu s$) を、平均リフレッシュ間隔 $525\mu s$、$262\mu s$、$131\mu s$ へそれぞれ短縮する。[表 8](#table-08)は Original+SRAM に対するこれら 2 設定のエネルギー効率を示す。AERP により KV キャッシュアクセスのオーバーヘッドは総エネルギーの小さな割合に留まる。そのため retention time の短縮によるエネルギー増加は小さく、Kelle+eDRAM は性能上の利点を維持する。

<span id="section-8-3-5"></span>

#### 8.3.5 入力系列長の影響

Llama2-7B と PG-19 を使い、異なる入出力系列長について、長い入力系列における Kelle+eDRAM のエネルギー消費を評価する。入力長-出力長 (例: "16K-128") という形式で各設定を表す。[図 16b](#figure-16)に示すように、入力系列が長く decoding 長が短い場合、pre-filling が総エネルギーの大半を占め、システムは compute-bound となる。この場合 Kelle は Original+SRAM に対して $2.1\times$ のエネルギー効率改善を達成する。入出力系列長がともに増えると、activation の DRAM access energy も増える。この memory-intensive な場合、効率的な KV キャッシュ管理により、Kelle は Original+SRAM に対して平均 $5.6\times$、AERP+SRAM に対して $1.8\times$ のエネルギー効率改善を達成する。

<span id="section-8-3-6"></span>

#### 8.3.6 Batch Size の影響

<span id="table-08"></span>

![リフレッシュ間隔ごとのエネルギー効率。](../../papers/kelle/table-08.png)

**表 8.** リフレッシュ間隔ごとのエネルギー効率。

<span id="table-09"></span>

![batch size ごとのエネルギー効率。](../../papers/kelle/table-09.png)

**表 9.** batch size ごとのエネルギー効率。

[表 9](#table-09)に示すように、Llama2-7B と PG-19 を用いて異なる batch size の Kelle を比較する。小さな batch size では RSA 利用率とモデル重みの転送効率が低いため、Original+SRAM に対するエネルギー効率の改善は小さくなるが、Kelle はすべてのベースラインを一貫して上回る。batch size 1 で Kelle は Original+SRAM、AEP+SRAM、AERP+SRAM に対し、それぞれ $71\%$、$37\%$、$25\%$ の speedup を達成する。

<span id="section-8-3-7"></span>

#### 8.3.7 eDRAM 帯域幅の影響

bank 数を半分、各 bank の容量を 2 倍にし、eDRAM の総面積と総容量を一定に保つことで、帯域幅を 128GB/s に下げた Kelle を評価する。Llama2-7B の PG-19 と TriviaQA で、Kelle+eDRAM は AERP+SRAM に対して $1.47\times$ と $1.35\times$、Original+SRAM に対して $6.31\times$ と $5.42\times$ のエネルギー利得を達成する。完全な帯域幅の Kelle よりわずかに低いが、帯域幅が減っても eDRAM 容量を増やせば、高コストな DRAM access を効果的に削減して帯域幅効率を改善できる。

<span id="section-8-4"></span>

### 8.4 議論

<span id="section-8-4-1"></span>

#### 8.4.1 長い Context の推論

長い context の推論では eDRAM 容量が限られるため、余分な KV データを 16GB DRAM へ offload する。LLaMA 2 7B の簡単な解析では、8-bit 重みが 16GB DRAM のうち 6.5GB を占め、各 token が 32 layer に 16-bit KV pair を持つとき、AERP なしの Kelle は最大 19,000 input token を扱える。AERP を導入すると各 layer の実行直後に KV キャッシュを削減し、後の layer で入力系列全体を収めるメモリを確保できる。これにより Kelle は約 60K token の入力系列を扱える。KV ベクトルを 4 bit に量子化すると最大 240K token を扱える。上限は残るものの、一般的な LLM 入力長である数万 token [Oth23a, Yang24b, Bai25a] を超える。

長い入力系列はオーバーヘッドを増やすが、[式 1](#equation-01)と[式 2](#equation-02)の permutation invariant property により、新しい KV ベクトルを追い出したベクトルと同じ位置へ配置でき、paging が大幅に単純化する。ベクトルは複雑な lookup mechanism なしに順次 prefetch できる。このため prefetch overhead は入力長に対して線形に増え、不釣り合いな増加を避けられる。

<span id="section-8-4-2"></span>

#### 8.4.2 Kelle の GPU への統合

Kelle は systolic array で実装するが、AERP は GPU にも適用できる。ただし systolic evictor がないため、attention score が最低の token の特定は非効率になり得る。2DRP は eDRAM に固有であり、リフレッシュエネルギーを減らす。eDRAM を GPU の既存メモリシステムと結合して KV ベクトルを格納できる。Kelle scheduler は CUDA で容易に実装できる。

<span id="section-9"></span>

## 9 結論

KV キャッシュは LLM の効率を高めるうえで不可欠である。しかし大量の KV ベクトルを格納すると、メモリ占有量とデータアクセスコストが増える。本研究では、KV ベクトルの主要記憶媒体として eDRAM を使用する Kelle システムを提案した。Kelle の高い性能は KV キャッシュ機構の実装における eDRAM の大きな可能性を示し、今後の研究への道を開く。
